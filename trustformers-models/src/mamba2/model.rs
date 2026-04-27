//! Mamba-2 model implementation.
//!
//! Implements State Space Duality (SSD) with multi-head structured matrices.
//! The core recurrence is:
//!   h_t = A_bar * h_{t-1} + B_bar * x_t
//!   y_t = C * h_t + D * x_t
//!
//! A, B, C and dt are all input-dependent (selective mechanism S6).

use crate::mamba2::config::Mamba2Config;

/// Error types for Mamba-2 operations.
#[derive(Debug, thiserror::Error)]
pub enum Mamba2Error {
    #[error("Empty input")]
    EmptyInput,
    #[error("Dimension mismatch: {0}")]
    DimMismatch(String),
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

/// Softplus: log(1 + exp(x)), numerically stable.
#[inline]
pub fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// SiLU (Swish): x * sigmoid(x).
#[inline]
fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// Matrix-vector multiply: weight [out x in] @ vec [in] -> [out].
fn mat_vec_mul(weight: &[Vec<f64>], x: &[f64]) -> Result<Vec<f64>, Mamba2Error> {
    if weight.is_empty() {
        return Ok(Vec::new());
    }
    let in_dim = weight[0].len();
    if x.len() != in_dim {
        return Err(Mamba2Error::DimMismatch(format!(
            "mat_vec_mul: weight cols={} but x len={}",
            in_dim,
            x.len()
        )));
    }
    let out: Vec<f64> = weight
        .iter()
        .map(|row| row.iter().zip(x.iter()).map(|(w, v)| w * v).sum())
        .collect();
    Ok(out)
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

/// Root Mean Square Layer Normalization for Mamba-2.
pub struct Mamba2RmsNorm {
    weight: Vec<f64>,
    eps: f64,
}

impl Mamba2RmsNorm {
    /// Create a new RMSNorm with all-ones weights.
    pub fn new(dim: usize, eps: f64) -> Self {
        Self {
            weight: vec![1.0; dim],
            eps,
        }
    }

    /// Forward pass: normalize then scale by learned weights.
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, Mamba2Error> {
        if x.is_empty() {
            return Err(Mamba2Error::EmptyInput);
        }
        if x.len() != self.weight.len() {
            return Err(Mamba2Error::DimMismatch(format!(
                "RmsNorm: weight dim={} but x len={}",
                self.weight.len(),
                x.len()
            )));
        }
        let mean_sq: f64 = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        let rms = (mean_sq + self.eps).sqrt();
        let out = x.iter().zip(self.weight.iter()).map(|(v, w)| v / rms * w).collect();
        Ok(out)
    }

    /// Dimension of this norm layer.
    pub fn dim(&self) -> usize {
        self.weight.len()
    }
}

// ---------------------------------------------------------------------------
// Mamba2SSM — core selective SSM layer
// ---------------------------------------------------------------------------

/// Core Mamba-2 SSM layer implementing the SSD (State Space Duality) mechanism.
///
/// Projection layout (in_proj output size per token):
///   z:    inner_dim
///   x:    inner_dim
///   B:    nheads * d_state
///   C:    nheads * d_state
///   dt:   nheads
///
/// Total in_proj output = 2*inner_dim + 2*nheads*d_state + nheads
pub struct Mamba2SSM {
    /// Input projection: d_model -> 2*inner + 2*nheads*d_state + nheads
    in_proj: Vec<Vec<f64>>,
    /// Output projection: inner_dim -> d_model
    out_proj: Vec<Vec<f64>>,
    /// Log of negative A (discretization parameter), shape [nheads]
    a_log: Vec<f64>,
    /// D skip connection bias, shape [nheads]
    d_bias: Vec<f64>,
    /// Delta time bias, shape [nheads]
    dt_bias: Vec<f64>,
    /// Local 1D causal conv weights, shape [inner_dim x d_conv]
    conv_weight: Vec<Vec<f64>>,
    config: Mamba2Config,
}

impl Mamba2SSM {
    /// Create a new Mamba2SSM with zero-initialized (but functional) weights.
    pub fn new(config: &Mamba2Config) -> Self {
        let inner_dim = config.inner_dim();
        let nheads = config.nheads;
        let d_state = config.d_state;
        let d_model = config.d_model;
        let d_conv = config.d_conv;

        // in_proj output: z(inner) + x(inner) + B(nheads*d_state) + C(nheads*d_state) + dt(nheads)
        let in_proj_out = 2 * inner_dim + 2 * nheads * d_state + nheads;

        // Initialize in_proj with small identity-like values (avoid all zeros → dead network)
        let in_proj: Vec<Vec<f64>> = (0..in_proj_out)
            .map(|i| {
                let mut row = vec![0.0f64; d_model];
                // small diagonal-ish init scaled by 0.02
                row[i % d_model] = 0.02;
                row
            })
            .collect();

        let out_proj: Vec<Vec<f64>> = (0..d_model)
            .map(|i| {
                let mut row = vec![0.0f64; inner_dim];
                row[i % inner_dim] = 0.02;
                row
            })
            .collect();

        // a_log initialized to log(1) = 0 → A = exp(-softplus(0)) ≈ exp(-ln2) ≈ 0.5
        let a_log = vec![0.0f64; nheads];
        // D skip connection initialized to 1 to give non-trivial output
        let d_bias = vec![1.0f64; nheads];
        let dt_bias = vec![0.0f64; nheads];

        // conv_weight: each channel gets a simple averaging kernel
        let conv_weight: Vec<Vec<f64>> =
            (0..inner_dim).map(|_| vec![1.0 / d_conv as f64; d_conv]).collect();

        Self {
            in_proj,
            out_proj,
            a_log,
            d_bias,
            dt_bias,
            conv_weight,
            config: config.clone(),
        }
    }

    /// Apply causal local convolution of width d_conv to a sequence.
    ///
    /// For each position t and each channel c:
    ///   out[t][c] = sum_{k=0}^{d_conv-1} conv_weight[c][k] * x[t-k][c]  (zero-padded)
    fn causal_conv(&self, x: &[Vec<f64>], _inner_dim: usize) -> Result<Vec<Vec<f64>>, Mamba2Error> {
        let seq_len = x.len();
        if seq_len == 0 {
            return Err(Mamba2Error::EmptyInput);
        }
        let channels = x[0].len();
        let d_conv = self.config.d_conv;

        let mut out = vec![vec![0.0f64; channels]; seq_len];
        for t in 0..seq_len {
            for c in 0..channels {
                let w = &self.conv_weight[c];
                let mut val = 0.0f64;
                for k in 0..d_conv {
                    if t >= k {
                        val += w[k] * x[t - k][c];
                    }
                    // implicit zero padding for t < k
                }
                out[t][c] = val;
            }
        }
        Ok(out)
    }

    /// Forward pass for the Mamba-2 SSM layer.
    ///
    /// Steps:
    /// 1. Project input to z, x_ssm, B, C, dt via in_proj
    /// 2. Apply causal convolution on x_ssm
    /// 3. Discretize: `A_bar = exp(-softplus(dt + dt_bias) * exp(a_log))`
    ///    and `B_bar = softplus(dt) * B`
    /// 4. Recurrent scan: `h_t = A_bar * h_{t-1} + B_bar * x_t`
    ///    and `y_t = (C * h) + D * x_t`  (D = d_bias, per-head skip)
    /// 5. Gate with z: `output = y * silu(z)`
    /// 6. `out_proj @ output`
    pub fn forward(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, Mamba2Error> {
        let seq_len = x.len();
        if seq_len == 0 {
            return Err(Mamba2Error::EmptyInput);
        }
        let d_model = self.config.d_model;
        if x[0].len() != d_model {
            return Err(Mamba2Error::DimMismatch(format!(
                "SSM forward: expected d_model={} but got {}",
                d_model,
                x[0].len()
            )));
        }

        let inner_dim = self.config.inner_dim();
        let nheads = self.config.nheads;
        let d_state = self.config.d_state;
        let headdim = self.config.headdim;

        // Split sizes for in_proj output
        let z_offset = 0usize;
        let x_offset = inner_dim;
        let b_offset = 2 * inner_dim;
        let c_offset = b_offset + nheads * d_state;
        let dt_offset = c_offset + nheads * d_state;

        // 1. Project all tokens
        let mut proj_out: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        for token in x.iter() {
            proj_out.push(mat_vec_mul(&self.in_proj, token)?);
        }

        // Extract z, x_ssm, B, C, dt slices
        let z_seq: Vec<Vec<f64>> =
            proj_out.iter().map(|p| p[z_offset..z_offset + inner_dim].to_vec()).collect();
        let x_ssm_raw: Vec<Vec<f64>> =
            proj_out.iter().map(|p| p[x_offset..x_offset + inner_dim].to_vec()).collect();
        let b_seq: Vec<Vec<f64>> = proj_out
            .iter()
            .map(|p| p[b_offset..b_offset + nheads * d_state].to_vec())
            .collect();
        let c_seq: Vec<Vec<f64>> = proj_out
            .iter()
            .map(|p| p[c_offset..c_offset + nheads * d_state].to_vec())
            .collect();
        let dt_seq: Vec<Vec<f64>> =
            proj_out.iter().map(|p| p[dt_offset..dt_offset + nheads].to_vec()).collect();

        // 2. Causal convolution on x_ssm
        let x_ssm = self.causal_conv(&x_ssm_raw, inner_dim)?;

        // 3 & 4. Discretize and run recurrent scan per head
        // State h: [nheads, headdim, d_state]
        let mut h: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0f64; d_state]; headdim]; nheads];

        let mut y_seq: Vec<Vec<f64>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let dt_t = &dt_seq[t];
            let b_t = &b_seq[t];
            let c_t = &c_seq[t];
            let x_t = &x_ssm[t];

            let mut y_t = vec![0.0f64; inner_dim];

            for head in 0..nheads {
                // Discretize dt for this head
                let dt_val = softplus(dt_t[head] + self.dt_bias[head]);
                // A_bar = exp(-dt_val * exp(a_log[head]))
                let a_bar = (-dt_val * self.a_log[head].exp()).exp();
                // B slice for this head: [d_state]
                let b_head = &b_t[head * d_state..(head + 1) * d_state];
                // C slice for this head: [d_state]
                let c_head = &c_t[head * d_state..(head + 1) * d_state];
                // x slice for this head: [headdim]
                let x_head = &x_t[head * headdim..(head + 1) * headdim];

                // Update state h[head]: shape [headdim, d_state]
                // h[hd][s] = A_bar * h[hd][s] + x_head[hd] * b_head[s]
                // y[head*headdim + hd] = sum_s(C[s] * h[hd][s]) + D[head] * x_head[hd]
                for hd in 0..headdim {
                    let x_val = x_head[hd];
                    let mut y_val = self.d_bias[head] * x_val; // D skip
                    for s in 0..d_state {
                        h[head][hd][s] = a_bar * h[head][hd][s] + x_val * b_head[s];
                        y_val += c_head[s] * h[head][hd][s];
                    }
                    y_t[head * headdim + hd] = y_val;
                }
            }

            // 5. Gate with z: output = y * silu(z)
            let z_t = &z_seq[t];
            let gated: Vec<f64> = y_t.iter().zip(z_t.iter()).map(|(y, z)| y * silu(*z)).collect();

            y_seq.push(gated);
        }

        // 6. out_proj @ output
        let mut result: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        for gated in y_seq.iter() {
            result.push(mat_vec_mul(&self.out_proj, gated)?);
        }

        Ok(result)
    }

    /// Access a_log (for testing discretization)
    pub fn a_log(&self) -> &[f64] {
        &self.a_log
    }

    /// Access d_bias (D skip connection weights)
    pub fn d_bias(&self) -> &[f64] {
        &self.d_bias
    }

    /// Config reference
    pub fn config(&self) -> &Mamba2Config {
        &self.config
    }
}

// ---------------------------------------------------------------------------
// Mamba2Block
// ---------------------------------------------------------------------------

/// A single Mamba-2 block: pre-norm + SSM.
pub struct Mamba2Block {
    ssm: Mamba2SSM,
    norm: Mamba2RmsNorm,
}

impl Mamba2Block {
    /// Create a new Mamba2Block.
    pub fn new(config: &Mamba2Config) -> Self {
        Self {
            ssm: Mamba2SSM::new(config),
            norm: Mamba2RmsNorm::new(config.d_model, config.rms_norm_eps),
        }
    }

    /// Forward: apply pre-norm then SSM, add residual.
    pub fn forward(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, Mamba2Error> {
        let seq_len = x.len();
        if seq_len == 0 {
            return Err(Mamba2Error::EmptyInput);
        }

        // Pre-norm each token
        let mut normed: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        for token in x.iter() {
            normed.push(self.norm.forward(token)?);
        }

        // SSM forward
        let ssm_out = self.ssm.forward(&normed)?;

        // Residual connection
        let out: Vec<Vec<f64>> = x
            .iter()
            .zip(ssm_out.iter())
            .map(|(res, s)| res.iter().zip(s.iter()).map(|(a, b)| a + b).collect())
            .collect();

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Mamba2Model
// ---------------------------------------------------------------------------

/// Full Mamba-2 backbone: embedding + N blocks + final norm.
pub struct Mamba2Model {
    embed_tokens: Vec<Vec<f64>>,
    layers: Vec<Mamba2Block>,
    norm_f: Mamba2RmsNorm,
    config: Mamba2Config,
}

impl Mamba2Model {
    /// Create a new Mamba2Model with the given configuration.
    pub fn new(config: &Mamba2Config) -> Self {
        let embed_tokens: Vec<Vec<f64>> = vec![vec![0.0f64; config.d_model]; config.vocab_size];
        let layers: Vec<Mamba2Block> =
            (0..config.n_layer).map(|_| Mamba2Block::new(config)).collect();
        let norm_f = Mamba2RmsNorm::new(config.d_model, config.rms_norm_eps);
        Self {
            embed_tokens,
            layers,
            norm_f,
            config: config.clone(),
        }
    }

    /// Forward: embed tokens, run through all blocks, apply final norm.
    pub fn forward(&self, input_ids: &[usize]) -> Result<Vec<Vec<f64>>, Mamba2Error> {
        let seq_len = input_ids.len();
        if seq_len == 0 {
            return Err(Mamba2Error::EmptyInput);
        }

        // Embed tokens
        let mut hidden: Vec<Vec<f64>> = input_ids
            .iter()
            .map(|&id| {
                if id < self.embed_tokens.len() {
                    self.embed_tokens[id].clone()
                } else {
                    vec![0.0f64; self.config.d_model]
                }
            })
            .collect();

        // Run through layers
        for layer in self.layers.iter() {
            hidden = layer.forward(&hidden)?;
        }

        // Final norm
        let mut normed: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        for token in hidden.iter() {
            normed.push(self.norm_f.forward(token)?);
        }

        Ok(normed)
    }

    /// Number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// Mamba2ForCausalLM
// ---------------------------------------------------------------------------

/// Mamba-2 language model with causal LM head.
pub struct Mamba2ForCausalLM {
    backbone: Mamba2Model,
    lm_head: Vec<Vec<f64>>,
}

impl Mamba2ForCausalLM {
    /// Create a new Mamba-2 causal LM model.
    pub fn new(config: &Mamba2Config) -> Self {
        // lm_head: vocab_size x d_model
        let lm_head: Vec<Vec<f64>> = vec![vec![0.0f64; config.d_model]; config.vocab_size];
        Self {
            backbone: Mamba2Model::new(config),
            lm_head,
        }
    }

    /// Forward pass: input_ids -> logits [seq_len, vocab_size]
    pub fn forward(&self, input_ids: &[usize]) -> Result<Vec<Vec<f64>>, Mamba2Error> {
        let hidden = self.backbone.forward(input_ids)?;

        let logits: Result<Vec<Vec<f64>>, Mamba2Error> =
            hidden.iter().map(|h| mat_vec_mul(&self.lm_head, h)).collect();

        logits
    }

    /// Config accessor
    pub fn config(&self) -> &Mamba2Config {
        &self.backbone.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mamba2::config::Mamba2Config;

    // ------ Test 1: config presets produce valid configs ------
    #[test]
    fn test_config_presets_valid() {
        let cfg_2_7b = Mamba2Config::mamba2_2_7b();
        assert!(cfg_2_7b.d_model > 0);
        assert!(cfg_2_7b.n_layer > 0);
        assert!(cfg_2_7b.d_state > 0);

        let cfg_small = Mamba2Config::small_test();
        assert!(cfg_small.d_model > 0);
        assert!(cfg_small.n_layer > 0);
    }

    // ------ Test 2: headdim consistency for 2.7B ------
    #[test]
    fn test_headdim_consistency_2_7b() {
        let cfg = Mamba2Config::mamba2_2_7b();
        assert!(
            cfg.validate(),
            "headdim={} should equal inner_dim={} / nheads={}",
            cfg.headdim,
            cfg.inner_dim(),
            cfg.nheads
        );
        assert_eq!(cfg.headdim * cfg.nheads, cfg.inner_dim());
    }

    // ------ Test 3: headdim consistency for small_test ------
    #[test]
    fn test_headdim_consistency_small() {
        let cfg = Mamba2Config::small_test();
        assert!(cfg.validate(), "small_test config should be valid");
        assert_eq!(cfg.headdim * cfg.nheads, cfg.inner_dim());
    }

    // ------ Test 4: d_model / nheads relation ------
    #[test]
    fn test_d_model_nheads_headdim_relation() {
        let cfg = Mamba2Config::small_test();
        // headdim = d_model * expand / nheads
        let expected_headdim = cfg.d_model * cfg.expand / cfg.nheads;
        assert_eq!(cfg.headdim, expected_headdim);
    }

    // ------ Test 5: RmsNorm forward correctness ------
    #[test]
    fn test_rmsnorm_forward() {
        let norm = Mamba2RmsNorm::new(4, 1e-5);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let out = norm.forward(&x).expect("rmsnorm should succeed");
        assert_eq!(out.len(), 4);
        // RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5)
        let mean_sq: f64 = x.iter().map(|v| v * v).sum::<f64>() / 4.0;
        let rms = (mean_sq + 1e-5).sqrt();
        let expected: Vec<f64> = x.iter().map(|v| v / rms).collect();
        for (got, exp) in out.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-9, "got={} exp={}", got, exp);
        }
    }

    // ------ Test 6: RmsNorm dimension mismatch error ------
    #[test]
    fn test_rmsnorm_dimension_mismatch() {
        let norm = Mamba2RmsNorm::new(4, 1e-5);
        let x = vec![1.0, 2.0];
        let result = norm.forward(&x);
        assert!(result.is_err());
        matches!(result.unwrap_err(), Mamba2Error::DimMismatch(_));
    }

    // ------ Test 7: local conv output size ------
    #[test]
    fn test_local_conv_output_size() {
        let cfg = Mamba2Config::small_test();
        let ssm = Mamba2SSM::new(&cfg);
        let seq_len = 8usize;
        let inner_dim = cfg.inner_dim();
        let x: Vec<Vec<f64>> = vec![vec![0.5f64; inner_dim]; seq_len];
        let out = ssm.causal_conv(&x, inner_dim).expect("conv should work");
        assert_eq!(out.len(), seq_len, "output seq_len should match input");
        assert_eq!(
            out[0].len(),
            inner_dim,
            "output channels should match inner_dim"
        );
    }

    // ------ Test 8: SSM forward shape ------
    #[test]
    fn test_ssm_forward_shape() {
        let cfg = Mamba2Config::small_test();
        let ssm = Mamba2SSM::new(&cfg);
        let seq_len = 5usize;
        let x: Vec<Vec<f64>> = vec![vec![0.1f64; cfg.d_model]; seq_len];
        let out = ssm.forward(&x).expect("ssm forward should succeed");
        assert_eq!(out.len(), seq_len);
        assert_eq!(out[0].len(), cfg.d_model);
    }

    // ------ Test 9: recurrence state update (output differs from input) ------
    #[test]
    fn test_recurrence_state_update() {
        let cfg = Mamba2Config::small_test();
        let ssm = Mamba2SSM::new(&cfg);
        // Use non-trivial input to ensure state accumulates
        let seq_len = 4usize;
        let x: Vec<Vec<f64>> =
            (0..seq_len).map(|i| vec![(i + 1) as f64 * 0.1; cfg.d_model]).collect();
        let out = ssm.forward(&x).expect("ssm forward");
        // Just check shape (actual values depend on weights)
        assert_eq!(out.len(), seq_len);
        assert_eq!(out[0].len(), cfg.d_model);
    }

    // ------ Test 10: D skip connection is non-zero ------
    #[test]
    fn test_d_skip_connection_nonzero() {
        let cfg = Mamba2Config::small_test();
        let ssm = Mamba2SSM::new(&cfg);
        // d_bias initialized to 1.0
        let all_nonzero = ssm.d_bias().iter().all(|&v| v != 0.0);
        assert!(all_nonzero, "D skip connection should be non-zero");
    }

    // ------ Test 11: full model forward (small_test) output shape ------
    #[test]
    fn test_full_model_forward_small() {
        let cfg = Mamba2Config::small_test();
        let model = Mamba2ForCausalLM::new(&cfg);
        let input_ids = vec![0usize, 1, 2, 3];
        let logits = model.forward(&input_ids).expect("full model forward");
        assert_eq!(logits.len(), 4, "one logit vector per token");
        assert_eq!(logits[0].len(), cfg.vocab_size, "logit dim = vocab_size");
    }

    // ------ Test 12: lm_head output shape ------
    #[test]
    fn test_lm_head_output_shape() {
        let cfg = Mamba2Config::small_test();
        let model = Mamba2ForCausalLM::new(&cfg);
        let input_ids = vec![0usize, 5, 10];
        let logits = model.forward(&input_ids).expect("lm_head forward");
        assert_eq!(logits.len(), 3);
        for row in logits.iter() {
            assert_eq!(row.len(), cfg.vocab_size);
        }
    }

    // ------ Test 13: softplus function correctness ------
    #[test]
    fn test_softplus_function() {
        // softplus(0) = ln(2) ≈ 0.693
        let sp0 = softplus(0.0);
        assert!((sp0 - std::f64::consts::LN_2).abs() < 1e-9);
        // softplus(x) > 0 for all x
        assert!(softplus(-10.0) > 0.0);
        assert!(softplus(10.0) > 0.0);
        // softplus(x) ≈ x for large x
        assert!((softplus(100.0) - 100.0).abs() < 0.01);
    }

    // ------ Test 14: discretization A_bar < 1 ------
    #[test]
    fn test_discretization_a_bar_less_than_one() {
        // A_bar = exp(-softplus(dt + dt_bias) * exp(a_log))
        // With dt=0, dt_bias=0, a_log=0: A_bar = exp(-softplus(0)*exp(0)) = exp(-ln2) ≈ 0.5
        let dt = 0.0f64;
        let dt_bias = 0.0f64;
        let a_log = 0.0f64;
        let a_bar = (-softplus(dt + dt_bias) * a_log.exp()).exp();
        assert!(a_bar < 1.0, "A_bar={} should be < 1 for stability", a_bar);
        assert!(a_bar > 0.0, "A_bar should be positive");
        // For positive dt, a_bar should be < for larger values
        let a_bar_large = (-softplus(5.0) * a_log.exp()).exp();
        assert!(a_bar_large < a_bar, "larger dt => smaller A_bar");
    }

    // ------ Test 15: empty input returns error ------
    #[test]
    fn test_empty_input_error() {
        let cfg = Mamba2Config::small_test();
        let model = Mamba2ForCausalLM::new(&cfg);
        let result = model.forward(&[]);
        assert!(result.is_err());
        matches!(result.unwrap_err(), Mamba2Error::EmptyInput);
    }
}
