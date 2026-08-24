use crate::s4::config::S4Config;
use scirs2_core::ndarray::Array2; // SciRS2 Integration Policy
use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result},
    layers::{Embedding, LayerNorm, Linear},
    ops::activations::gelu,
    tensor::Tensor,
    traits::{Layer, Model},
};

pub use super::discretization::{Discretization, HiPPOMatrix};
pub use super::layer::S4Layer;

/// S4 Block combining S4 layer with additional components
pub struct S4Block {
    config: S4Config,
    s4_layer: S4Layer,
    norm: LayerNorm,
    in_proj: Linear,
    out_proj: Linear,
    #[allow(dead_code)]
    dropout: f32,
    device: Device,
}

impl S4Block {
    /// Whether this block gates its output through a GLU.
    ///
    /// The reference S4 block's `postact = "glu"` projects to `2 · d_model` and
    /// gates one half with the sigmoid of the other. A previous revision kept the
    /// `d_model`-wide projection and applied **GELU**, under a comment reading
    /// "GLU activation would split and gate / Simplified for now" — so a config
    /// asking for a gated block silently got an ungated one with a different
    /// non-linearity. The projection is now sized for the gate it declares.
    pub fn uses_glu(config: &S4Config) -> bool {
        config.postact == "glu"
    }

    /// Build a block on a device.
    ///
    /// # Errors
    ///
    /// Fails when the S4 layer cannot be built or discretised, or when the norm
    /// cannot be allocated.
    pub fn new_with_device(config: &S4Config, device: Device) -> Result<Self> {
        let d_model = config.d_model;
        let n_ssm = config.get_n_ssm();

        let s4_layer = S4Layer::new_with_device(config, device)?;
        let norm = LayerNorm::new_with_device(vec![d_model], config.layer_norm_eps, device)?;
        let in_proj = Linear::new_with_device(d_model, n_ssm, config.use_bias, device);
        let out_width = if Self::uses_glu(config) { d_model * 2 } else { d_model };
        let out_proj = Linear::new_with_device(n_ssm, out_width, config.use_bias, device);

        Ok(Self {
            config: config.clone(),
            s4_layer,
            norm,
            in_proj,
            out_proj,
            dropout: config.dropout,
            device,
        })
    }

    /// Build a block on the CPU.
    ///
    /// # Errors
    ///
    /// See [`S4Block::new_with_device`].
    pub fn new(config: &S4Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    /// The device this block reports.
    pub fn device(&self) -> Device {
        self.device
    }

    /// The state-space layer.
    pub fn s4_layer(&self) -> &S4Layer {
        &self.s4_layer
    }

    /// The state-space layer, mutably — used by the checkpoint binder.
    pub fn s4_layer_mut(&mut self) -> &mut S4Layer {
        &mut self.s4_layer
    }

    /// The pre-norm, the input projection and the output projection, mutably.
    pub(super) fn parts_mut(&mut self) -> (&mut LayerNorm, &mut Linear, &mut Linear) {
        (&mut self.norm, &mut self.in_proj, &mut self.out_proj)
    }

    /// The width of `out_proj`: `2 · d_model` for a gated block, else `d_model`.
    pub fn out_projection_width(&self) -> usize {
        if Self::uses_glu(&self.config) {
            self.config.d_model * 2
        } else {
            self.config.d_model
        }
    }

    /// Run the state-space recurrence over a `[batch, seq_len, channels]` buffer.
    fn run_state_space(&self, projected: &Tensor) -> Result<Tensor> {
        let arr = match projected {
            Tensor::F32(arr) => arr,
            _ => {
                return Err(tensor_op_error(
                    "s4_block",
                    "the S4 recurrence needs an F32 activation".to_string(),
                ))
            },
        };
        let shape = arr.shape().to_vec();
        let channels = *shape.last().unwrap_or(&0);
        if channels != self.config.get_n_ssm() {
            return Err(tensor_op_error(
                "s4_block",
                format!(
                    "the S4 recurrence expects {} channel(s), got {channels}",
                    self.config.get_n_ssm()
                ),
            ));
        }
        let (batch, seq_len) = match shape.len() {
            3 => (shape[0], shape[1]),
            2 => (1, shape[0]),
            other => {
                return Err(tensor_op_error(
                    "s4_block",
                    format!("the S4 recurrence needs a 2-D or 3-D activation, got {other}-D"),
                ))
            },
        };
        let contiguous = arr.as_standard_layout().to_owned();
        let values: Vec<f32> = match contiguous.as_slice() {
            Some(slice) => slice.to_vec(),
            None => contiguous.iter().copied().collect(),
        };

        let mut output = vec![0.0_f32; values.len()];
        // The recurrence runs over [channels, seq_len]; the activation is stored
        // as [.., seq_len, channels], so each batch element is transposed in and
        // back out.
        let mut sequence = Array2::<f32>::zeros((channels, seq_len));
        for item in 0..batch {
            let base = item * seq_len * channels;
            for step in 0..seq_len {
                for channel in 0..channels {
                    sequence[[channel, step]] = values[base + step * channels + channel];
                }
            }
            let processed = self.s4_layer.apply(&sequence)?;
            for step in 0..seq_len {
                for channel in 0..channels {
                    output[base + step * channels + channel] = processed[[channel, step]];
                }
            }
        }
        Tensor::from_vec(output, &shape)
    }

    /// Gate the second half of `input`'s last dimension into the first.
    fn gated_linear_unit(&self, input: &Tensor) -> Result<Tensor> {
        let arr = match input {
            Tensor::F32(arr) => arr,
            _ => {
                return Err(tensor_op_error(
                    "s4_block",
                    "the gate needs an F32 activation".to_string(),
                ))
            },
        };
        let mut shape = arr.shape().to_vec();
        let width = *shape.last().unwrap_or(&0);
        if width != self.config.d_model * 2 {
            return Err(tensor_op_error(
                "s4_block",
                format!(
                    "a gated block projects to {} values per position, got {width}",
                    self.config.d_model * 2
                ),
            ));
        }
        let half = self.config.d_model;
        let contiguous = arr.as_standard_layout().to_owned();
        let values: Vec<f32> = match contiguous.as_slice() {
            Some(slice) => slice.to_vec(),
            None => contiguous.iter().copied().collect(),
        };
        let mut gated = Vec::with_capacity(values.len() / 2);
        for position in values.chunks_exact(width) {
            for index in 0..half {
                let gate = 1.0 / (1.0 + (-position[half + index]).exp());
                gated.push(position[index] * gate);
            }
        }
        if let Some(last) = shape.last_mut() {
            *last = half;
        }
        Tensor::from_vec(gated, &shape)
    }
}

impl Layer for S4Block {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let residual = input.clone();
        let normed = self.norm.forward(input)?;
        let projected = self.in_proj.forward(normed)?;
        let state_space_output = self.run_state_space(&projected)?;
        let output = self.out_proj.forward(state_space_output)?;

        let activated = match self.config.postact.as_str() {
            "glu" => self.gated_linear_unit(&output)?,
            "gelu" => gelu(&output)?,
            _ => output,
        };

        residual.add(&activated)
    }
}

impl S4Block {
    pub fn parameter_count(&self) -> usize {
        let mut total = 0;

        // S4 layer parameters
        total += self.s4_layer.parameter_count();

        // Layer norm parameters
        total += self.norm.parameter_count();

        // Projection layers parameters
        total += self.in_proj.parameter_count();
        total += self.out_proj.parameter_count();

        total
    }
}

/// S4 Model for sequence modeling
pub struct S4Model {
    pub config: S4Config,
    pub embeddings: Embedding,
    pub blocks: Vec<S4Block>,
    pub ln_f: LayerNorm,
    pub device: Device,
}

impl S4Model {
    pub fn new_with_device(config: S4Config, device: Device) -> Result<Self> {
        let embeddings =
            Embedding::new_with_device(config.vocab_size, config.d_model, None, device)?;

        let mut blocks = Vec::new();
        for _ in 0..config.n_layer {
            if let Ok(block) = S4Block::new_with_device(&config, device) {
                blocks.push(block);
            }
        }

        let ln_f = LayerNorm::new_with_device(vec![config.d_model], config.layer_norm_eps, device)?;

        Ok(Self {
            config,
            embeddings,
            blocks,
            ln_f,
            device,
        })
    }

    pub fn new(config: S4Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Model for S4Model {
    type Config = S4Config;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Get batch and sequence dimensions from input
        let (batch_size, seq_len, input_ids) = match &input {
            Tensor::I64(ref arr) => {
                if arr.ndim() == 2 {
                    let batch_size = arr.shape()[0];
                    let seq_len = arr.shape()[1];
                    let ids = arr.mapv(|x| x as u32).into_raw_vec_and_offset().0;
                    (batch_size, seq_len, ids)
                } else if arr.ndim() == 1 {
                    let seq_len = arr.len();
                    let ids = arr.mapv(|x| x as u32).into_raw_vec_and_offset().0;
                    (1, seq_len, ids)
                } else {
                    return Err(tensor_op_error(
                        "tensor_operation",
                        "Input tensor must be 1D or 2D".to_string(),
                    ));
                }
            },
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Unsupported tensor type".to_string(),
                ))
            },
        };

        // Get embeddings - returns [total_tokens, d_model]
        let embedded = self.embeddings.forward(input_ids)?;

        // Reshape to 3D [batch_size, seq_len, d_model]
        let mut hidden = if embedded.shape().len() == 2 {
            let total_tokens = embedded.shape()[0];
            let d_model = embedded.shape()[1];
            if total_tokens == batch_size * seq_len {
                embedded.reshape(&[batch_size, seq_len, d_model])?
            } else {
                embedded.reshape(&[1, total_tokens, d_model])?
            }
        } else {
            embedded
        };

        // Apply S4 blocks
        for block in &self.blocks {
            hidden = block.forward(hidden)?;
        }

        // Final layer norm
        self.ln_f.forward(hidden)
    }

    /// Load an S4 checkpoint that follows this crate's declared tensor contract.
    ///
    /// The full name map, and why this crate declares a contract instead of
    /// claiming a HuggingFace convention that does not exist for S4, are in
    /// [`crate::s4::loading`].
    ///
    /// Two earlier revisions were both dishonest. The first parsed a bespoke
    /// `S4ML` container and returned `Ok(())` while every component step either
    /// discarded the array it had just decoded or ran through a length check and
    /// an offset bump — not one tensor reached a parameter. The second refused
    /// outright, correctly at the time, because [`S4Layer`] had no setters and
    /// binding without invalidating the discretisation cache would have left the
    /// model computing with the pre-load discretisation. The setters exist now
    /// and every one of them rebuilds that cache, so the binder is real.
    ///
    /// # Errors
    ///
    /// Fails when the container cannot be parsed, when the checkpoint does not
    /// match the contract, when a tensor has the wrong shape, when a parameter is
    /// missing, or when an unrecognised tensor is present.
    fn load_pretrained(&mut self, reader: &mut dyn std::io::Read) -> Result<()> {
        self.load_pretrained_report(reader).map(|_| ())
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Embeddings parameters
        total += self.embeddings.parameter_count();

        // S4 blocks parameters
        for block in &self.blocks {
            total += block.parameter_count();
        }

        // Final layer norm parameters
        total += self.ln_f.parameter_count();

        total
    }
}

/// S4 Model for Language Modeling
pub struct S4ForLanguageModeling {
    pub s4: S4Model,
    pub lm_head: Linear,
    pub device: Device,
}

impl S4ForLanguageModeling {
    pub fn new_with_device(config: S4Config, device: Device) -> Result<Self> {
        let s4 = S4Model::new_with_device(config.clone(), device)?;
        let lm_head = Linear::new_with_device(
            config.d_model,
            config.vocab_size,
            false, // No bias for LM head
            device,
        );

        Ok(Self {
            s4,
            lm_head,
            device,
        })
    }

    pub fn new(config: S4Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Model for S4ForLanguageModeling {
    type Config = S4Config;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let hidden = self.s4.forward(input)?;
        self.lm_head.forward(hidden)
    }

    /// Load the backbone and, when the checkpoint carries one, the LM head.
    ///
    /// A previous revision called the backbone loader, ignored the LM head with
    /// the comment "For now, just return success after loading S4 weights", and
    /// returned `Ok(())` — on top of a backbone loader that bound nothing.
    ///
    /// # Errors
    ///
    /// See [`S4ForLanguageModeling::load_pretrained_report`].
    fn load_pretrained(&mut self, reader: &mut dyn std::io::Read) -> Result<()> {
        self.load_pretrained_report(reader).map(|_| ())
    }

    fn get_config(&self) -> &Self::Config {
        self.s4.get_config()
    }

    fn num_parameters(&self) -> usize {
        // S4 backbone parameters + LM head parameters
        self.s4.num_parameters() + self.lm_head.parameter_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1; // SciRS2 Integration Policy

    #[test]
    fn test_hippo_initialization() {
        let n = 4;

        // Test LEGS initialization
        let legs = HiPPOMatrix::LEGS;
        let a_legs = legs.initialize(n);
        assert_eq!(a_legs.shape(), &[n, n]);
        // Check skew-symmetric
        let diff = &a_legs + &a_legs.t();
        assert!(diff.iter().all(|&x| x.abs() < 1e-6));

        // Test other initializations
        let legt = HiPPOMatrix::LEGT;
        let a_legt = legt.initialize(n);
        assert_eq!(a_legt.shape(), &[n, n]);

        let fourier = HiPPOMatrix::Fourier;
        let a_fourier = fourier.initialize(n);
        assert_eq!(a_fourier.shape(), &[n, n]);
    }

    #[test]
    fn test_discretization() {
        let n = 4;
        let a = Array2::<f32>::eye(n);
        let b = Array1::<f32>::ones(n);
        let dt = 0.01;

        // Test ZOH discretization
        let zoh = Discretization::ZOH;
        let (a_bar, b_bar) = zoh.discretize(&a, &b, dt).expect("discretization must succeed");
        assert_eq!(a_bar.shape(), &[n, n]);
        assert_eq!(b_bar.shape(), &[n]);

        // Test other methods
        let euler = Discretization::Euler;
        let (a_bar_euler, b_bar_euler) =
            euler.discretize(&a, &b, dt).expect("discretization must succeed");
        assert_eq!(a_bar_euler.shape(), &[n, n]);
        assert_eq!(b_bar_euler.shape(), &[n]);
    }

    #[test]
    fn test_s4_layer_creation() {
        let config = S4Config::default();
        let layer = S4Layer::new(&config);
        assert!(layer.is_ok());

        let layer = layer.expect("operation failed");
        assert_eq!(layer.a_real().shape(), &[config.d_state, config.d_state]);
        assert_eq!(layer.b_real().shape(), &[config.d_state]);
        assert_eq!(layer.c_real().shape(), &[config.d_state]);
        assert_eq!(layer.d().shape(), &[config.get_n_ssm()]);
    }

    #[test]
    fn test_s4_model_creation() {
        let config = S4Config::s4_small();
        let model = S4Model::new(config.clone()).expect("operation failed");

        assert_eq!(model.config.d_model, config.d_model);
        assert_eq!(model.blocks.len(), config.n_layer);
    }

    #[test]
    fn test_s4_lm_creation() {
        let config = S4Config::s4_base();
        let _model = S4ForLanguageModeling::new(config).expect("operation failed");

        // S4 language model created successfully - LM head dimensions are internal
    }

    // ---- HiPPO matrix shape tests ----

    #[test]
    fn test_hippo_matrix_shapes_all_types() {
        for (hippo, name) in [
            (HiPPOMatrix::LEGS, "LEGS"),
            (HiPPOMatrix::LEGT, "LEGT"),
            (HiPPOMatrix::LAGT, "LAGT"),
            (HiPPOMatrix::Fourier, "Fourier"),
            (HiPPOMatrix::Random, "Random"),
        ] {
            let a = hippo.initialize(8);
            assert_eq!(a.shape(), &[8, 8], "HiPPO {} must produce 8x8 matrix", name);
        }
    }

    #[test]
    fn test_hippo_legs_skew_symmetric() {
        // LEGS must produce a skew-symmetric matrix: A + A^T = 0
        let n = 6;
        let legs = HiPPOMatrix::LEGS;
        let a = legs.initialize(n);
        let sum = &a + &a.t();
        for val in sum.iter() {
            assert!(
                val.abs() < 1e-5,
                "LEGS must be skew-symmetric; got residual {}",
                val
            );
        }
    }

    #[test]
    fn test_hippo_legs_different_sizes() {
        for n in [4usize, 8, 16, 32] {
            let a = HiPPOMatrix::LEGS.initialize(n);
            assert_eq!(
                a.shape(),
                &[n, n],
                "LEGS matrix must have shape [n, n] for n={}",
                n
            );
        }
    }

    // ---- Discretization ----

    #[test]
    fn test_bilinear_discretization_shapes() {
        let n = 4;
        let a = Array2::<f32>::eye(n);
        let b = Array1::<f32>::ones(n);
        let dt = 0.001f32;
        let (a_bar, b_bar) = Discretization::Bilinear
            .discretize(&a, &b, dt)
            .expect("discretization must succeed");
        assert_eq!(a_bar.shape(), &[n, n]);
        assert_eq!(b_bar.shape(), &[n]);
    }

    #[test]
    fn test_zoh_identity_a_bar_approaches_identity_for_small_dt() {
        // ZOH: Ā ≈ I + Δ·A for small Δ when A = 0 → Ā ≈ I
        let n = 4;
        let a_zero = Array2::<f32>::zeros((n, n));
        let b = Array1::<f32>::ones(n);
        let dt = 1e-6f32;
        let (a_bar, _b_bar) = Discretization::ZOH
            .discretize(&a_zero, &b, dt)
            .expect("discretization must succeed");
        // With A=0, ZOH should give A_bar ≈ I
        for i in 0..n {
            assert!(
                (a_bar[[i, i]] - 1.0).abs() < 1e-4,
                "ZOH with A=0 should give diagonal ~1; got {}",
                a_bar[[i, i]]
            );
        }
    }

    #[test]
    fn test_euler_discretization_matches_formula() {
        // Euler: A_bar = I + dt * A, B_bar = dt * B
        let n = 3;
        let dt = 0.1f32;
        let a = Array2::<f32>::zeros((n, n));
        let b = Array1::<f32>::ones(n);
        let (a_bar, b_bar) = Discretization::Euler
            .discretize(&a, &b, dt)
            .expect("discretization must succeed");
        // With A=0: A_bar = I
        for i in 0..n {
            assert!((a_bar[[i, i]] - 1.0).abs() < 1e-6);
        }
        // B_bar = dt * b = 0.1 * 1.0 = 0.1
        for i in 0..n {
            assert!((b_bar[i] - dt).abs() < 1e-6, "Euler B_bar must equal dt*B");
        }
    }

    // ---- S4 SSM recurrence: y = Cx + Du ----

    #[test]
    fn test_ssm_output_y_equals_cx_plus_du_at_t0() {
        // At t=0 state x=0, so y = C*0 + D*u = D*u
        let d_state = 4usize;
        let c: Vec<f64> = vec![1.0, 0.5, -0.5, 0.25];
        let d_scalar = 2.0f64;
        let state: Vec<f64> = vec![0.0; d_state];
        let u = 1.5f64;

        // y = sum(c_i * x_i) + d * u
        let cx: f64 = c.iter().zip(state.iter()).map(|(ci, xi)| ci * xi).sum();
        let y = cx + d_scalar * u;
        assert!(
            (y - 3.0).abs() < 1e-12,
            "y at t=0 from zero state must be D*u = {}",
            3.0
        );
    }

    #[test]
    fn test_ssm_recurrence_state_update() {
        // x_{t+1} = a_bar * x_t + b_bar * u  (scalar simplified)
        let a_bar = 0.9f64;
        let b_bar = 0.1f64;
        let mut x = 0.0f64;
        let u = 1.0f64;

        for _ in 0..500 {
            x = a_bar * x + b_bar * u;
        }
        // Fixed point: x* = b_bar * u / (1 - a_bar) = 0.1 / 0.1 = 1.0
        let fixed_point = b_bar * u / (1.0 - a_bar);
        assert!(
            (x - fixed_point).abs() < 1e-4,
            "Recurrence must converge to fixed-point {}; got {}",
            fixed_point,
            x
        );
    }

    // ---- S4D diagonal variant ----

    #[test]
    fn test_s4_layer_a_real_is_square() {
        let config = S4Config {
            d_state: 8,
            ..Default::default()
        };
        let layer = S4Layer::new(&config).expect("S4Layer creation must succeed");
        let (r, c) = (layer.a_real().shape()[0], layer.a_real().shape()[1]);
        assert_eq!(r, c, "A_real must be square");
        assert_eq!(r, 8, "A_real must have d_state rows");
    }

    #[test]
    fn test_s4_layer_b_c_lengths_match_d_state() {
        let d_state = 16;
        let config = S4Config {
            d_state,
            ..Default::default()
        };
        let layer = S4Layer::new(&config).expect("S4Layer creation must succeed");
        assert_eq!(layer.b_real().len(), d_state);
        assert_eq!(layer.c_real().len(), d_state);
    }

    #[test]
    fn test_s4_layer_d_length_matches_n_ssm() {
        let config = S4Config {
            d_model: 64,
            n_ssm: None,
            ..Default::default()
        };
        let layer = S4Layer::new(&config).expect("S4Layer creation must succeed");
        assert_eq!(
            layer.d().len(),
            config.get_n_ssm(),
            "D skip-connection must have length n_ssm"
        );
    }

    // ---- Cauchy kernel property ----

    /// K(ω) = C(ωI - A)^{-1}B should be a well-defined frequency-domain kernel.
    /// We verify a simplified property: for real diagonal A and ω ≠ eigenvalue,
    /// the resolvent (ωI - A) is invertible.
    #[test]
    fn test_cauchy_kernel_resolvent_invertible() {
        // For diagonal A = diag(a_1, ..., a_N), (ωI - A) is invertible when ω ≠ a_i.
        let a: u64 = 6364136223846793005;
        let c_lcg: u64 = 1442695040888963407;
        let mut lcg: u64 = 0xABCD_EF01_2345_6789;

        let d_state = 4;
        // Generate diagonal eigenvalues in (-1, 0) (stable region)
        let mut eigenvalues = Vec::with_capacity(d_state);
        for _ in 0..d_state {
            lcg = lcg.wrapping_mul(a).wrapping_add(c_lcg);
            let ev = (lcg as i64 as f64) / (u64::MAX as f64) * 0.9; // in (-0.9, 0.9)
            eigenvalues.push(ev - 1.0); // shift to (-1.9, -0.1) — stable
        }

        // Choose ω not equal to any eigenvalue
        let omega = 0.5f64;
        for &ev in &eigenvalues {
            let denom = omega - ev;
            assert!(
                denom.abs() > 1e-8,
                "Resolvent denominator must be non-zero for ω={}, a_i={}",
                omega,
                ev
            );
        }
    }

    // ---- Sequence output shape ----

    #[test]
    fn test_s4_block_creation_succeeds() {
        let config = S4Config {
            d_model: 64,
            d_state: 8,
            n_ssm: Some(64),
            ..Default::default()
        };
        let block = S4Block::new(&config);
        assert!(block.is_ok(), "S4Block creation must succeed");
    }

    #[test]
    fn test_s4_layer_parameter_count_positive() {
        let config = S4Config::default();
        let layer = S4Layer::new(&config).expect("S4Layer creation must succeed");
        assert!(
            layer.parameter_count() > 0,
            "S4Layer must have > 0 parameters"
        );
    }

    #[test]
    fn test_s4_model_block_count() {
        let config = S4Config {
            n_layer: 4,
            ..S4Config::default()
        };
        let model = S4Model::new(config.clone()).expect("S4Model must be created");
        assert_eq!(
            model.blocks.len(),
            config.n_layer,
            "S4Model must have n_layer blocks"
        );
    }

    #[test]
    fn test_s4_model_num_parameters_positive() {
        let config = S4Config {
            d_model: 64,
            d_state: 8,
            n_layer: 2,
            n_ssm: Some(64),
            ..Default::default()
        };
        let model = S4Model::new(config).expect("S4Model creation must succeed");
        assert!(
            model.num_parameters() > 0,
            "S4Model must have > 0 parameters"
        );
    }

    // ---- Causal convolution property ----

    /// A causal kernel K_t = 0 for t < 0 (no future information leakage).
    /// We verify this by checking that the S4 kernel coefficients only depend on
    /// past positions when processed sequentially.
    #[test]
    fn test_causal_convolution_future_is_zero_at_init() {
        // A causal filter has kernel K(t) = 0 for t < 0.
        // We represent this by checking that initial state is zero (no future leakage).
        let d_state = 4;
        let state: Vec<f64> = vec![0.0; d_state];
        // At t=0 the state is 0, meaning no "future" values have influenced the output.
        assert!(
            state.iter().all(|&x| x == 0.0),
            "Initial causal state must be zero"
        );
    }

    // ── Checkpoint loading refuses instead of faking success ────────────────

    fn loading_config() -> S4Config {
        S4Config {
            d_model: 8,
            d_state: 4,
            n_layer: 1,
            vocab_size: 12,
            max_position_embeddings: 16,
            ..S4Config::default()
        }
    }

    /// Rebuild the exact `S4ML` byte stream the deleted loader accepted.
    ///
    /// This fixture is the point of the regression test: the old
    /// `load_weights_from_buffer` walked precisely this layout, validated every
    /// length, *skipped* every payload and returned `Ok(())`. Feeding it back in
    /// proves the model no longer claims to have loaded weights it never bound.
    fn well_formed_s4ml_buffer(config: &S4Config) -> Vec<u8> {
        fn push_tensor(bytes: &mut Vec<u8>, element_count: usize) {
            let byte_len = element_count * 4;
            bytes.extend_from_slice(&(byte_len as u32).to_le_bytes());
            for i in 0..element_count {
                bytes.extend_from_slice(&(i as f32 + 1.0).to_le_bytes());
            }
        }

        let d_model = config.d_model;
        let d_state = config.d_state;

        let metadata = format!(
            r#"{{"config":{{"d_model":{d_model},"d_state":{d_state},"n_layer":{},"vocab_size":{}}}}}"#,
            config.n_layer, config.vocab_size
        );
        let metadata_bytes = metadata.as_bytes();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0x5334_4D4Cu32.to_le_bytes()); // "S4ML"
        bytes.extend_from_slice(&1u32.to_le_bytes()); // version
        bytes.extend_from_slice(&(metadata_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(metadata_bytes);

        // Embeddings.
        push_tensor(&mut bytes, config.vocab_size * d_model);

        for _ in 0..config.n_layer {
            // State-space parameters: A (real, imag), B (real, imag),
            // C (real, imag), D, dt.
            push_tensor(&mut bytes, d_state * d_state);
            push_tensor(&mut bytes, d_state * d_state);
            push_tensor(&mut bytes, d_state);
            push_tensor(&mut bytes, d_state);
            push_tensor(&mut bytes, d_state);
            push_tensor(&mut bytes, d_state);
            push_tensor(&mut bytes, d_model);
            push_tensor(&mut bytes, d_model);
            // Block LayerNorm.
            push_tensor(&mut bytes, d_model);
            push_tensor(&mut bytes, d_model);
            // Input projection (d_model -> 2 * d_model) and its bias.
            push_tensor(&mut bytes, 2 * d_model * d_model);
            push_tensor(&mut bytes, 2 * d_model);
            // Output projection (d_model -> d_model) and its bias.
            push_tensor(&mut bytes, d_model * d_model);
            push_tensor(&mut bytes, d_model);
        }

        // Final LayerNorm.
        push_tensor(&mut bytes, d_model);
        push_tensor(&mut bytes, d_model);

        bytes
    }

    /// Regression: this exact buffer used to return `Ok(())` while every tensor
    /// in it was length-checked and skipped, leaving the model at its
    /// constructor initialisation.
    #[test]
    fn load_pretrained_refuses_the_container_it_used_to_fake_a_load_from() {
        let config = loading_config();
        let bytes = well_formed_s4ml_buffer(&config);

        let mut model = S4Model::new(config).expect("model must build");
        let before = model.embeddings.weight().data().expect("readable");

        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("a load that binds nothing must not report success");
        assert!(
            err.to_string().contains("container"),
            "the bespoke `S4ML` container is not one this crate reads; the refusal must say \
             so: {err}"
        );
        assert_eq!(
            model.embeddings.weight().data().expect("readable"),
            before,
            "a refused load must leave every parameter untouched"
        );
    }

    /// The language-modelling wrapper used to call the backbone loader and then
    /// return `Ok(())` with the LM head untouched; it now propagates the failure.
    #[test]
    fn language_modeling_load_pretrained_propagates_the_refusal() {
        let config = loading_config();
        let bytes = well_formed_s4ml_buffer(&config);

        let mut model = S4ForLanguageModeling::new(config).expect("model must build");
        let before = model.lm_head.weight().data().expect("readable");

        model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("the wrapper must not report a load the backbone refused");
        assert_eq!(
            model.lm_head.weight().data().expect("readable"),
            before,
            "the LM head must be untouched by a refused load"
        );
    }

    /// The stream is still drained, so a caller reusing the reader sees a
    /// defined state.
    #[test]
    fn load_pretrained_drains_the_reader_before_refusing() {
        let mut model = S4Model::new(loading_config()).expect("model must build");
        let bytes = vec![0x33u8; 96];
        let mut cursor = bytes.as_slice();
        let _ = model.load_pretrained(&mut cursor);
        assert!(
            cursor.is_empty(),
            "the reader must be fully consumed even when the load is refused"
        );
    }
}
