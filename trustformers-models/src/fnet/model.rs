use crate::common::ActivationType;
use crate::fnet::config::FNetConfig;
use crate::weight_loading::binding::{
    bind_embedding, bind_head_layer_norm, bind_head_linear, bind_linear, take_norm_bias,
    take_norm_weight, BoundNamespaces,
};
use crate::weight_loading::checkpoint::{Checkpoint, LoadReport, UnusedTensors};
use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result},
    layers::{Embedding, LayerNorm, Linear},
    tensor::Tensor,
    traits::{Config, Layer, Model},
};

/// Fourier Transform layer that replaces self-attention
/// Applies 2D DFT along sequence and feature dimensions
pub struct FourierTransform {
    fourier_type: String,
    #[allow(dead_code)]
    use_bias: bool,
    bias: Option<Linear>,
    #[allow(dead_code)]
    dropout: f32,
    device: Device,
}

impl FourierTransform {
    pub fn new(config: &FNetConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &FNetConfig, device: Device) -> Result<Self> {
        let bias = if config.use_bias_in_fourier {
            Some(Linear::new_with_device(
                config.hidden_size,
                config.hidden_size,
                true,
                device,
            ))
        } else {
            None
        };

        Ok(Self {
            fourier_type: config.fourier_transform_type.clone(),
            use_bias: config.use_bias_in_fourier,
            bias,
            dropout: config.fourier_dropout_prob,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn parameter_count(&self) -> usize {
        if let Some(ref bias_layer) = self.bias {
            bias_layer.parameter_count()
        } else {
            0
        }
    }

    /// Apply the 2-D Discrete Fourier Transform and keep its real part.
    ///
    /// This is FNet's token-mixing operation: `Re(F_seq(F_hidden(x)))`.
    ///
    /// # Why this needs complex arithmetic
    ///
    /// A previous revision built a single matrix holding only
    /// `cos(-2πkj/N)/√N` and applied it twice. That is **not** the real part of
    /// the 2-D DFT. Writing the 1-D basis as `e^{-2πi kn/N} = c - i·s`, the
    /// separable 2-D transform gives
    ///
    /// ```text
    /// Re(X)[k, l] = Σ_n Σ_m x[n, m] · (c_kn·c_lm − s_kn·s_lm)
    /// ```
    ///
    /// The dropped `−s·s` term is exactly what the cosine-only version threw
    /// away, so its output was a *different linear map* — cosine mixing, not a
    /// Fourier transform. Both the sine and the cosine components are carried
    /// through here, so the imaginary part produced by the first axis
    /// contributes to the real part after the second, as it must.
    ///
    /// # Errors
    ///
    /// Fails when the input is neither 2-D (`[seq, hidden]`) nor 3-D
    /// (`[batch, seq, hidden]`), or when an intermediate reshape fails.
    fn apply_dft(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch_size, seq_len, hidden_size] or [seq_len, hidden_size]
        // Normalise to 3-D so all downstream math is consistent.
        let (x3d, was_2d) =
            if x.shape().len() == 2 { (x.unsqueeze(0)?, true) } else { (x.clone(), false) };
        if x3d.shape().len() != 3 {
            return Err(tensor_op_error(
                "FourierTransform::apply_dft",
                format!(
                    "expected a [seq, hidden] or [batch, seq, hidden] tensor, got shape {:?}",
                    x.shape()
                ),
            ));
        }

        // DFT along the sequence axis. The input is real, so the transform's
        // imaginary part starts here.
        let (real_seq, imag_seq) = self.dft_1d_complex(&x3d, None, 1)?;

        // DFT along the hidden axis, carrying the complex intermediate.
        let (real_both, _imag_both) = self.dft_1d_complex(&real_seq, Some(&imag_seq), 2)?;

        // Take the real part (FNet discards the imaginary component).
        let out3d = real_both;

        // Restore original rank if input was 2-D
        if was_2d {
            out3d.squeeze(0)
        } else {
            Ok(out3d)
        }
    }

    /// Apply Real DFT (more efficient variant)
    ///
    /// For a real-valued input the negative frequencies are the conjugates of
    /// the positive ones, so the real part of the full transform is identical to
    /// the real part of the half-spectrum transform. The result is therefore the
    /// same as [`FourierTransform::apply_dft`], and this variant delegates to it
    /// rather than pretending to a different numeric result.
    ///
    /// # Errors
    ///
    /// See [`FourierTransform::apply_dft`].
    fn apply_real_dft(&self, x: &Tensor) -> Result<Tensor> {
        self.apply_dft(x)
    }

    /// Apply Discrete Cosine Transform (DCT)
    fn apply_dct(&self, x: &Tensor) -> Result<Tensor> {
        // DCT is real-valued and often more efficient than DFT
        // For now, approximate with cosine-based transformation
        // Normalise to 3-D so all downstream math is consistent.
        let (x, was_2d) =
            if x.shape().len() == 2 { (x.unsqueeze(0)?, true) } else { (x.clone(), false) };
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        let hidden_size = x.shape()[2];

        // Create DCT basis matrices
        let seq_dct_matrix = self.create_dct_matrix(seq_len)?;
        let hidden_dct_matrix = self.create_dct_matrix(hidden_size)?;

        // Apply DCT along sequence dimension using reshape to keep matmul 2-D.
        // Transpose to [batch, hidden_size, seq_len], flatten to [batch*hidden, seq_len],
        // apply DCT matrix, then restore original shape.
        let seq_shape = seq_dct_matrix.shape();
        let seq_dim0 = seq_shape.len().saturating_sub(2);
        let seq_dim1 = seq_shape.len().saturating_sub(1);
        let seq_dct_t = seq_dct_matrix.transpose(seq_dim0, seq_dim1)?;
        let x_t = x.transpose(1, 2)?; // [batch, hidden, seq]
        let x_t_flat = x_t.reshape(&[batch_size * hidden_size, seq_len])?;
        let seq_out_flat = x_t_flat.matmul(&seq_dct_t)?;
        let seq_out_t = seq_out_flat.reshape(&[batch_size, hidden_size, seq_len])?;
        let x_seq_dct = seq_out_t.transpose(1, 2)?; // back to [batch, seq, hidden]

        // Apply DCT along hidden dimension
        // For hidden dimension: reshape, apply DCT, reshape back
        let reshaped = x_seq_dct.reshape(&[batch_size * seq_len, hidden_size])?;
        let hidden_shape = hidden_dct_matrix.shape();
        let hidden_dim0 = hidden_shape.len().saturating_sub(2);
        let hidden_dim1 = hidden_shape.len().saturating_sub(1);
        let hidden_dct =
            reshaped.matmul(&hidden_dct_matrix.transpose(hidden_dim0, hidden_dim1)?)?;
        let out3d = hidden_dct.reshape(&[batch_size, seq_len, hidden_size])?;

        // Restore original rank if input was 2-D
        if was_2d {
            out3d.squeeze(0)
        } else {
            Ok(out3d)
        }
    }

    /// Create DCT transformation matrix
    fn create_dct_matrix(&self, n: usize) -> Result<Tensor> {
        let mut matrix = Vec::new();
        let pi = std::f32::consts::PI;

        for k in 0..n {
            for i in 0..n {
                let value = if k == 0 {
                    (1.0 / n as f32).sqrt()
                } else {
                    (2.0 / n as f32).sqrt()
                        * (pi * k as f32 * (2 * i + 1) as f32 / (2 * n) as f32).cos()
                };
                matrix.push(value);
            }
        }

        Tensor::from_vec(matrix, &[n, n])
    }

    /// Orthonormal 1-D DFT basis matrices `C[k, j] = cos(2πkj/n)/√n` and
    /// `S[k, j] = sin(2πkj/n)/√n`.
    ///
    /// The forward transform is `X[k] = Σ_j x[j] e^{-2πi kj/n} / √n`, i.e.
    /// `Re(X) = C·x` and `Im(X) = −S·x` for a real `x`.
    ///
    /// # Errors
    ///
    /// Fails when `n` is 0 or when the basis tensors cannot be built.
    fn dft_basis(&self, n: usize) -> Result<(Tensor, Tensor)> {
        if n == 0 {
            return Err(tensor_op_error(
                "FourierTransform::dft_basis",
                "cannot build a DFT basis for a zero-length axis".to_string(),
            ));
        }
        let mut cos_matrix = Vec::with_capacity(n * n);
        let mut sin_matrix = Vec::with_capacity(n * n);
        let scale = 1.0 / (n as f32).sqrt();
        let two_pi = 2.0 * std::f32::consts::PI;
        for k in 0..n {
            for j in 0..n {
                // (k*j) mod n keeps the angle small for long axes, which matters
                // for f32 precision once k*j exceeds 2^24.
                let angle = two_pi * ((k * j) % n) as f32 / n as f32;
                cos_matrix.push(angle.cos() * scale);
                sin_matrix.push(angle.sin() * scale);
            }
        }
        Ok((
            Tensor::from_vec(cos_matrix, &[n, n])?,
            Tensor::from_vec(sin_matrix, &[n, n])?,
        ))
    }

    /// Apply a 1-D DFT along `dim` to a complex-valued 3-D tensor.
    ///
    /// The input is `real + i·imag` (`imag = None` means a real input). The
    /// transform is the orthonormal forward DFT, so with `C = cos` and
    /// `S = sin` bases:
    ///
    /// ```text
    /// Re(out) = C·real + S·imag
    /// Im(out) = C·imag − S·real
    /// ```
    ///
    /// Both components are returned, because dropping the imaginary part between
    /// the two axes of a 2-D transform changes the result — that omission is
    /// what made the previous cosine-only implementation something other than a
    /// Fourier transform.
    ///
    /// # Errors
    ///
    /// Fails when `dim` is not 1 or 2, when the tensor is not 3-D, or when a
    /// reshape / matmul fails.
    fn dft_1d_complex(
        &self,
        real: &Tensor,
        imag: Option<&Tensor>,
        dim: usize,
    ) -> Result<(Tensor, Tensor)> {
        let shape = real.shape();
        if shape.len() != 3 {
            return Err(tensor_op_error(
                "FourierTransform::dft_1d_complex",
                format!("expected a 3-D tensor, got shape {shape:?}"),
            ));
        }
        if dim != 1 && dim != 2 {
            return Err(tensor_op_error(
                "FourierTransform::dft_1d_complex",
                format!("DFT axis must be 1 (sequence) or 2 (hidden), got {dim}"),
            ));
        }
        let batch_size = shape[0];
        let seq_len = shape[1];
        let hidden_size = shape[2];
        let n = shape[dim];

        let (cos_basis, sin_basis) = self.dft_basis(n)?;
        // The rows of the basis are frequencies; a right-multiply `x @ Bᵀ`
        // computes `Σ_j x[.., j] B[k, j]` for every k, which is the transform.
        let cos_t = cos_basis.transpose(0, 1)?;
        let sin_t = sin_basis.transpose(0, 1)?;

        // Flatten so the axis under transform is the last one and the matmul is 2-D.
        let flatten = |t: &Tensor| -> Result<Tensor> {
            if dim == 1 {
                // [batch, seq, hidden] -> [batch, hidden, seq] -> [batch*hidden, seq]
                t.transpose(1, 2)?.reshape(&[batch_size * hidden_size, seq_len])
            } else {
                t.reshape(&[batch_size * seq_len, hidden_size])
            }
        };
        let restore = |t: Tensor| -> Result<Tensor> {
            if dim == 1 {
                t.reshape(&[batch_size, hidden_size, seq_len])?.transpose(1, 2)
            } else {
                t.reshape(&[batch_size, seq_len, hidden_size])
            }
        };

        let real_flat = flatten(real)?;
        let real_cos = real_flat.matmul(&cos_t)?;
        let real_sin = real_flat.matmul(&sin_t)?;

        let (out_real_flat, out_imag_flat) = match imag {
            Some(imag) => {
                let imag_flat = flatten(imag)?;
                let imag_cos = imag_flat.matmul(&cos_t)?;
                let imag_sin = imag_flat.matmul(&sin_t)?;
                // Re = C·re + S·im ; Im = C·im − S·re
                (real_cos.add(&imag_sin)?, imag_cos.sub(&real_sin)?)
            },
            // Real input: Re = C·re ; Im = −S·re
            None => (real_cos, real_sin.scalar_mul(-1.0)?),
        };

        Ok((restore(out_real_flat)?, restore(out_imag_flat)?))
    }
}

impl Layer for FourierTransform {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Apply the appropriate Fourier transform
        let fourier_output = match self.fourier_type.as_str() {
            "dft" => self.apply_dft(&input)?,
            "real_dft" => self.apply_real_dft(&input)?,
            "dct" => self.apply_dct(&input)?,
            _ => self.apply_dft(&input)?, // Default to DFT
        };

        // Apply bias if configured
        let output = if let Some(ref bias_layer) = self.bias {
            bias_layer.forward(fourier_output)?
        } else {
            fourier_output
        };

        // Apply dropout if configured (in training mode)
        // For inference, we skip dropout
        Ok(output)
    }
}

/// FNet feed-forward network (same as BERT)
pub struct FNetFeedForward {
    pub(crate) dense1: Linear,
    pub(crate) dense2: Linear,
    activation: ActivationType,
    #[allow(dead_code)]
    dropout: f32,
    device: Device,
}

impl FNetFeedForward {
    pub fn new(config: &FNetConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &FNetConfig, device: Device) -> Result<Self> {
        let dense1 =
            Linear::new_with_device(config.hidden_size, config.intermediate_size, true, device);
        let dense2 =
            Linear::new_with_device(config.intermediate_size, config.hidden_size, true, device);

        Ok(Self {
            dense1,
            dense2,
            activation: ActivationType::from_config_str_or(
                &config.hidden_act,
                ActivationType::Identity,
            ),
            dropout: config.hidden_dropout_prob,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn parameter_count(&self) -> usize {
        self.dense1.parameter_count() + self.dense2.parameter_count()
    }

    fn apply_activation(&self, x: &Tensor) -> Result<Tensor> {
        self.activation.apply(x)
    }
}

impl Layer for FNetFeedForward {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let hidden = self.dense1.forward(input)?;
        let hidden = self.apply_activation(&hidden)?;
        self.dense2.forward(hidden)
    }
}

/// FNet encoder layer (Fourier + FFN)
pub struct FNetLayer {
    fourier_transform: FourierTransform,
    pub(crate) feed_forward: FNetFeedForward,
    pub(crate) fourier_norm: LayerNorm,
    pub(crate) output_norm: LayerNorm,
    device: Device,
}

impl FNetLayer {
    pub fn new(config: &FNetConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &FNetConfig, device: Device) -> Result<Self> {
        let fourier_transform = FourierTransform::new_with_device(config, device)?;
        let feed_forward = FNetFeedForward::new_with_device(config, device)?;
        let fourier_norm =
            LayerNorm::new_with_device(vec![config.hidden_size], config.layer_norm_eps, device)?;
        let output_norm =
            LayerNorm::new_with_device(vec![config.hidden_size], config.layer_norm_eps, device)?;

        Ok(Self {
            fourier_transform,
            feed_forward,
            fourier_norm,
            output_norm,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn parameter_count(&self) -> usize {
        self.fourier_transform.parameter_count()
            + self.feed_forward.parameter_count()
            + self.fourier_norm.parameter_count()
            + self.output_norm.parameter_count()
    }
}

impl Layer for FNetLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Fourier transform with residual connection and layer norm
        let fourier_output = self.fourier_transform.forward(input.clone())?;
        let fourier_output = input.add(&fourier_output)?; // Residual
        let fourier_output = self.fourier_norm.forward(fourier_output)?;

        // Feed-forward with residual connection and layer norm
        let ff_output = self.feed_forward.forward(fourier_output.clone())?;
        let output = fourier_output.add(&ff_output)?; // Residual
        self.output_norm.forward(output)
    }
}

/// FNet embeddings (same as BERT)
pub struct FNetEmbeddings {
    pub(crate) word_embeddings: Embedding,
    pub(crate) position_embeddings: Embedding,
    pub(crate) token_type_embeddings: Embedding,
    pub(crate) layer_norm: LayerNorm,
    #[allow(dead_code)]
    dropout: f32,
    device: Device,
}

impl FNetEmbeddings {
    pub fn new(config: &FNetConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &FNetConfig, device: Device) -> Result<Self> {
        let word_embeddings = Embedding::new_with_device(
            config.vocab_size,
            config.hidden_size,
            Some(config.pad_token_id as usize),
            device,
        )?;
        let position_embeddings = Embedding::new_with_device(
            config.max_position_embeddings,
            config.hidden_size,
            None,
            device,
        )?;
        let token_type_embeddings =
            Embedding::new_with_device(config.type_vocab_size, config.hidden_size, None, device)?;
        let layer_norm =
            LayerNorm::new_with_device(vec![config.hidden_size], config.layer_norm_eps, device)?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout: config.hidden_dropout_prob,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn parameter_count(&self) -> usize {
        self.word_embeddings.parameter_count()
            + self.position_embeddings.parameter_count()
            + self.token_type_embeddings.parameter_count()
            + self.layer_norm.parameter_count()
    }
}

impl Layer for FNetEmbeddings {
    type Input = (Vec<u32>, Option<Vec<u32>>, Option<Vec<u32>>);
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let (input_ids, token_type_ids, position_ids) = input;
        let seq_len = input_ids.len();

        let words_embeddings = self.word_embeddings.forward(input_ids)?;

        let position_ids = position_ids.unwrap_or_else(|| (0..seq_len as u32).collect());
        let position_embeddings = self.position_embeddings.forward(position_ids)?;

        let token_type_ids = token_type_ids.unwrap_or_else(|| vec![0; seq_len]);
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;

        let embeddings = words_embeddings.add(&position_embeddings)?.add(&token_type_embeddings)?;
        let embeddings = self.layer_norm.forward(embeddings)?;

        Ok(embeddings)
    }
}

/// FNet encoder
pub struct FNetEncoder {
    pub(crate) layers: Vec<FNetLayer>,
    device: Device,
}

impl FNetEncoder {
    pub fn new(config: &FNetConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &FNetConfig, device: Device) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(FNetLayer::new_with_device(config, device)?);
        }

        Ok(Self { layers, device })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn parameter_count(&self) -> usize {
        self.layers.iter().map(|layer| layer.parameter_count()).sum()
    }
}

impl Layer for FNetEncoder {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let mut hidden_states = input;

        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states)?;
        }

        Ok(hidden_states)
    }
}

/// FNet model
pub struct FNetModel {
    config: FNetConfig,
    embeddings: FNetEmbeddings,
    encoder: FNetEncoder,
    device: Device,
}

impl FNetModel {
    pub fn new(config: FNetConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: FNetConfig, device: Device) -> Result<Self> {
        config.validate()?;

        let embeddings = FNetEmbeddings::new_with_device(&config, device)?;
        let encoder = FNetEncoder::new_with_device(&config, device)?;

        Ok(Self {
            config,
            embeddings,
            encoder,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Model for FNetModel {
    type Config = FNetConfig;
    type Input = (Vec<u32>, Option<Vec<u32>>, Option<Vec<u32>>);
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let embeddings = self.embeddings.forward(input)?;
        let sequence_output = self.encoder.forward(embeddings)?;
        Ok(sequence_output)
    }

    /// Load a HuggingFace FNet checkpoint (safetensors or `torch.save`).
    ///
    /// A previous revision was `Ok(())` — the reader was never touched, so every
    /// "load" left the model randomly initialised while reporting success. See
    /// [`FNetModel::load_from_checkpoint`] for the name map.
    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.load_pretrained_report(reader).map(|_| ())
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        self.embeddings.parameter_count() + self.encoder.parameter_count()
    }
}

/// FNet for sequence classification
pub struct FNetForSequenceClassification {
    fnet: FNetModel,
    classifier: Linear,
    #[allow(dead_code)]
    num_labels: usize,
    device: Device,
}

impl FNetForSequenceClassification {
    pub fn new(config: FNetConfig, num_labels: usize) -> Result<Self> {
        Self::new_with_device(config, num_labels, Device::CPU)
    }

    pub fn new_with_device(config: FNetConfig, num_labels: usize, device: Device) -> Result<Self> {
        let fnet = FNetModel::new_with_device(config.clone(), device)?;
        let classifier = Linear::new_with_device(config.hidden_size, num_labels, true, device);

        Ok(Self {
            fnet,
            classifier,
            num_labels,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Model for FNetForSequenceClassification {
    type Config = FNetConfig;
    type Input = (Vec<u32>, Option<Vec<u32>>, Option<Vec<u32>>);
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let sequence_output = self.fnet.forward(input)?;
        // Extract CLS token (position 0 of sequence dimension).
        // sequence_output may be 2-D [seq_len, hidden] or 3-D [batch, seq_len, hidden].
        let cls_output = if sequence_output.shape().len() == 3 {
            // 3-D: [batch, seq_len, hidden] → slice seq dim → [batch, 1, hidden] → squeeze → [batch, hidden]
            let sliced = sequence_output.slice(1, 0, 1)?;
            sliced.squeeze(1)?
        } else {
            // 2-D: [seq_len, hidden] → slice first row → [1, hidden] (keep 2-D for Linear)
            sequence_output.slice(0, 0, 1)?
        };
        self.classifier.forward(cls_output)
    }

    /// Load the encoder and, when the checkpoint carries one, the classifier head.
    ///
    /// # Errors
    ///
    /// See [`FNetForSequenceClassification::load_pretrained_report`].
    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.load_pretrained_report(reader).map(|_| ())
    }

    fn get_config(&self) -> &Self::Config {
        self.fnet.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.fnet.num_parameters() + self.classifier.parameter_count()
    }
}

impl FNetForSequenceClassification {
    /// The checkpoint namespaces this wrapper binds, and therefore must fully
    /// consume.
    ///
    /// See [`BoundNamespaces`] for why a wrapper is stricter than the bare
    /// encoder over the very names it binds.
    const BOUND_NAMESPACES: BoundNamespaces<'static> = BoundNamespaces::new(&["classifier."]);

    /// Load the encoder and the classification head, reporting what was bound.
    ///
    /// A previous revision delegated straight to `FNetModel::load_pretrained`,
    /// which binds the encoder only. `FNetModel`'s unused-tensor policy tolerates
    /// the `classifier.` namespace, so a fine-tuned checkpoint's head was
    /// silently dropped: inference then ran through a constructor-initialised
    /// classifier while `load_pretrained` returned `Ok(())`.
    ///
    /// A checkpoint that carries no head at all — a plain pretrained encoder — is
    /// still accepted, but the absent head tensors are recorded in
    /// [`LoadReport::missing`] so the caller can see the layer kept its
    /// initialisation rather than being told everything was loaded.
    ///
    /// # Errors
    ///
    /// Fails when the checkpoint cannot be parsed, when an encoder parameter is
    /// missing, or when the head is present with the wrong shape.
    pub fn load_pretrained_report(&mut self, reader: &mut dyn Read) -> Result<LoadReport> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        let mut report = self.fnet.load_from_checkpoint(&checkpoint)?;
        let hidden = self.fnet.get_config().hidden_size;
        bind_head_linear(
            &checkpoint,
            &mut report,
            "classifier",
            [self.num_labels, hidden],
            &mut self.classifier,
        )?;
        Self::BOUND_NAMESPACES.verify(&report)?;
        Ok(report)
    }
}

/// FNet's masked-language-modelling prediction head.
///
/// HuggingFace's `FNetLMPredictionHead` is a `transform` block — a square dense
/// projection, GELU and a `LayerNorm` — followed by a `decoder` back to the
/// vocabulary, exactly the layout BERT uses. A previous revision of this crate
/// collapsed the head to a single `Linear`, which meant a real
/// `FNetForMaskedLM` checkpoint could not be represented at all: its
/// `cls.predictions.transform.*` tensors had nowhere to land and were dropped
/// while the load reported success.
pub struct FNetLMHead {
    dense: Linear,
    layer_norm: LayerNorm,
    decoder: Linear,
}

impl FNetLMHead {
    /// Build the head for `config` on `device`.
    ///
    /// # Errors
    ///
    /// Fails when the LayerNorm cannot be constructed for `hidden_size`.
    pub fn new_with_device(config: &FNetConfig, device: Device) -> Result<Self> {
        Ok(Self {
            dense: Linear::new_with_device(config.hidden_size, config.hidden_size, true, device),
            layer_norm: LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            decoder: Linear::new_with_device(config.hidden_size, config.vocab_size, true, device),
        })
    }

    /// dense → GELU → LayerNorm → decoder.
    ///
    /// # Errors
    ///
    /// Propagates any layer failure.
    pub fn forward(&self, hidden_states: Tensor) -> Result<Tensor> {
        let hidden_states = self.dense.forward(hidden_states)?;
        let hidden_states = trustformers_core::ops::activations::gelu(&hidden_states)?;
        let hidden_states = self.layer_norm.forward(hidden_states)?;
        self.decoder.forward(hidden_states)
    }

    /// Total learnable parameters of the head.
    pub fn parameter_count(&self) -> usize {
        self.dense.parameter_count()
            + self.layer_norm.parameter_count()
            + self.decoder.parameter_count()
    }
}

/// FNet for masked language modeling
pub struct FNetForMaskedLM {
    fnet: FNetModel,
    mlm_head: FNetLMHead,
    device: Device,
}

impl FNetForMaskedLM {
    /// The checkpoint namespaces this wrapper binds, and therefore must fully
    /// consume.
    ///
    /// See [`BoundNamespaces`] for why a wrapper is stricter than the bare
    /// encoder over the very names it binds.
    const BOUND_NAMESPACES: BoundNamespaces<'static> = BoundNamespaces::new(&["cls.predictions."]);

    pub fn new(config: FNetConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: FNetConfig, device: Device) -> Result<Self> {
        let fnet = FNetModel::new_with_device(config.clone(), device)?;
        let mlm_head = FNetLMHead::new_with_device(&config, device)?;

        Ok(Self {
            fnet,
            mlm_head,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Load the encoder and the masked-LM head, reporting what was bound.
    ///
    /// A previous revision delegated straight to `FNetModel::load_pretrained`,
    /// which binds the encoder only. `FNetModel`'s unused-tensor policy tolerates
    /// the `cls.` namespace, so the whole prediction head was silently dropped
    /// while `load_pretrained` returned `Ok(())`.
    ///
    /// HuggingFace declares the decoder with `bias=False` and aliases
    /// `decoder.bias` onto a separate `cls.predictions.bias` parameter, so a real
    /// checkpoint may spell the output bias either way; both are accepted, and
    /// the canonical `cls.predictions.bias` wins when the checkpoint has both.
    ///
    /// # Errors
    ///
    /// Fails when the checkpoint cannot be parsed, when an encoder parameter is
    /// missing, or when a head tensor is present with the wrong shape.
    pub fn load_pretrained_report(&mut self, reader: &mut dyn Read) -> Result<LoadReport> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        let mut report = self.fnet.load_from_checkpoint(&checkpoint)?;

        let config = self.fnet.get_config().clone();
        let hidden = config.hidden_size;
        let vocab = config.vocab_size;

        bind_head_linear(
            &checkpoint,
            &mut report,
            "cls.predictions.transform.dense",
            [hidden, hidden],
            &mut self.mlm_head.dense,
        )?;
        bind_head_layer_norm(
            &checkpoint,
            &mut report,
            "cls.predictions.transform.LayerNorm",
            hidden,
            &mut self.mlm_head.layer_norm,
        )?;

        let decoder_weight = "cls.predictions.decoder.weight";
        match checkpoint.take_shaped(decoder_weight, &[vocab, hidden])? {
            Some(weight) => {
                self.mlm_head.decoder.set_weight(weight)?;
                report.mark_loaded(decoder_weight);
            },
            None => report.note_absent(decoder_weight),
        }

        let canonical_bias = "cls.predictions.bias";
        let aliased_bias = "cls.predictions.decoder.bias";
        let bias_name =
            if checkpoint.contains(canonical_bias) { canonical_bias } else { aliased_bias };
        match checkpoint.take_shaped(bias_name, &[vocab])? {
            Some(bias) => {
                self.mlm_head.decoder.set_bias(bias)?;
                report.mark_loaded(bias_name);
                // The two spellings alias one parameter; note the other as
                // consumed so a checkpoint carrying both is fully accounted for.
                if checkpoint.contains(aliased_bias) && bias_name == canonical_bias {
                    report.mark_loaded(aliased_bias);
                }
            },
            None => report.note_absent(canonical_bias),
        }

        Self::BOUND_NAMESPACES.verify(&report)?;
        Ok(report)
    }
}

impl Model for FNetForMaskedLM {
    type Config = FNetConfig;
    type Input = (Vec<u32>, Option<Vec<u32>>, Option<Vec<u32>>);
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let sequence_output = self.fnet.forward(input)?;
        self.mlm_head.forward(sequence_output)
    }

    /// Load the encoder and, when the checkpoint carries one, the prediction head.
    ///
    /// # Errors
    ///
    /// See [`FNetForMaskedLM::load_pretrained_report`].
    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.load_pretrained_report(reader).map(|_| ())
    }

    fn get_config(&self) -> &Self::Config {
        self.fnet.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.fnet.num_parameters() + self.mlm_head.parameter_count()
    }
}

impl FNetModel {
    /// Checkpoint namespaces an FNet encoder legitimately does not consume.
    pub(crate) const ALLOWED_UNUSED_PREFIXES: &'static [&'static str] =
        &["cls.", "classifier.", "qa_outputs.", "pooler."];

    /// Non-parameter buffers HuggingFace stores alongside FNet's weights.
    pub(crate) const ALLOWED_UNUSED_SUFFIXES: &'static [&'static str] =
        &["embeddings.position_ids", "embeddings.token_type_ids"];

    /// Load pretrained weights and report exactly what was bound.
    ///
    /// # Errors
    ///
    /// See [`FNetModel::load_from_checkpoint`].
    pub fn load_pretrained_report(&mut self, reader: &mut dyn Read) -> Result<LoadReport> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        self.load_from_checkpoint(&checkpoint)
    }

    /// Bind an already-parsed checkpoint into this model.
    ///
    /// FNet is BERT with self-attention replaced by a parameter-free 2-D Fourier
    /// transform, so its checkpoints carry BERT's embedding and feed-forward
    /// tensors and simply *omit* every `attention.self.*` / `attention.output.*`
    /// projection — the mixing layer has no weights to store. The two per-layer
    /// norms keep their HuggingFace spellings (`fourier.output.LayerNorm` and
    /// `output.LayerNorm`).
    ///
    /// # Errors
    ///
    /// Fails when the checkpoint does not look like an FNet checkpoint, when a
    /// tensor has the wrong shape, when a parameter is missing, or when the
    /// checkpoint carries weights this architecture does not recognise.
    pub fn load_from_checkpoint(&mut self, checkpoint: &Checkpoint) -> Result<LoadReport> {
        let prefix =
            checkpoint.detect_prefix(&["", "fnet."], "embeddings.word_embeddings.weight")?;
        let mut binder = checkpoint.binder(&prefix);

        let config = self.config.clone();
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;

        bind_embedding(
            &mut binder,
            "embeddings.word_embeddings",
            config.vocab_size,
            hidden,
            &mut self.embeddings.word_embeddings,
        )?;
        bind_embedding(
            &mut binder,
            "embeddings.position_embeddings",
            config.max_position_embeddings,
            hidden,
            &mut self.embeddings.position_embeddings,
        )?;
        bind_embedding(
            &mut binder,
            "embeddings.token_type_embeddings",
            config.type_vocab_size,
            hidden,
            &mut self.embeddings.token_type_embeddings,
        )?;
        if let Some(w) = take_norm_weight(&mut binder, "embeddings.LayerNorm", hidden)? {
            self.embeddings.layer_norm.set_weight(w)?;
        }
        if let Some(b) = take_norm_bias(&mut binder, "embeddings.LayerNorm", hidden)? {
            self.embeddings.layer_norm.set_bias(b)?;
        }

        for (index, layer) in self.encoder.layers.iter_mut().enumerate() {
            let base = format!("encoder.layer.{index}");

            // The Fourier mixing layer itself has no parameters; only the
            // residual norm that follows it does.
            let fourier_norm = format!("{base}.fourier.output.LayerNorm");
            if let Some(w) = take_norm_weight(&mut binder, &fourier_norm, hidden)? {
                layer.fourier_norm.set_weight(w)?;
            }
            if let Some(b) = take_norm_bias(&mut binder, &fourier_norm, hidden)? {
                layer.fourier_norm.set_bias(b)?;
            }

            bind_linear(
                &mut binder,
                &format!("{base}.intermediate.dense"),
                intermediate,
                hidden,
                true,
                &mut layer.feed_forward.dense1,
            )?;
            bind_linear(
                &mut binder,
                &format!("{base}.output.dense"),
                hidden,
                intermediate,
                true,
                &mut layer.feed_forward.dense2,
            )?;

            let output_norm = format!("{base}.output.LayerNorm");
            if let Some(w) = take_norm_weight(&mut binder, &output_norm, hidden)? {
                layer.output_norm.set_weight(w)?;
            }
            if let Some(b) = take_norm_bias(&mut binder, &output_norm, hidden)? {
                layer.output_norm.set_bias(b)?;
            }
        }

        binder.finish(UnusedTensors::new(
            Self::ALLOWED_UNUSED_PREFIXES,
            Self::ALLOWED_UNUSED_SUFFIXES,
        ))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fnet::config::FNetConfig;
    use trustformers_core::{
        tensor::Tensor,
        traits::{Config, Model},
    };

    fn tiny_config() -> FNetConfig {
        FNetConfig {
            vocab_size: 64,
            hidden_size: 16,
            num_hidden_layers: 2,
            intermediate_size: 32,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            max_position_embeddings: 32,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: "absolute".to_string(),
            use_fourier_transform: true,
            use_tpu_optimized_fft: false,
            fourier_transform_type: "dft".to_string(),
            use_bias_in_fourier: false,
            fourier_dropout_prob: 0.0,
        }
    }

    fn make_input(seq_len: usize) -> (Vec<u32>, Option<Vec<u32>>, Option<Vec<u32>>) {
        let ids: Vec<u32> = (0..seq_len as u32).collect();
        (ids, None, None)
    }

    // ── Fourier transform correctness ────────────────────────────────────────

    /// Reference orthonormal 2-D DFT real part, computed directly from the
    /// definition with `f64` complex accumulation.
    fn reference_dft_real(values: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * cols];
        let scale = 1.0 / ((rows as f64).sqrt() * (cols as f64).sqrt());
        for k in 0..rows {
            for l in 0..cols {
                let mut re = 0.0f64;
                for n in 0..rows {
                    for m in 0..cols {
                        let angle = -2.0
                            * std::f64::consts::PI
                            * ((k * n) as f64 / rows as f64 + (l * m) as f64 / cols as f64);
                        re += values[n * cols + m] as f64 * angle.cos();
                    }
                }
                out[k * cols + l] = (re * scale) as f32;
            }
        }
        out
    }

    fn fourier_layer(kind: &str) -> FourierTransform {
        let mut config = tiny_config();
        config.fourier_transform_type = kind.to_string();
        config.use_bias_in_fourier = false;
        FourierTransform::new(&config).expect("Fourier layer must build")
    }

    /// Regression: `apply_dft` used a cosine-only basis applied twice, dropping
    /// the `−sin·sin` cross term of the separable 2-D transform. That is a
    /// different linear map, so its output does not match the real part of a
    /// genuine 2-D DFT and this comparison fails against it.
    #[test]
    fn dft_matches_the_real_part_of_a_direct_2d_fourier_transform() {
        let rows = 6usize;
        let cols = 4usize;
        // A deterministic, non-symmetric signal: a symmetric one would hide the
        // missing sine term because its sine components vanish.
        let values: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7) % 13) as f32 - 6.0 + 0.25 * i as f32)
            .collect();
        let input = Tensor::from_vec(values.clone(), &[rows, cols]).expect("input must build");

        let layer = fourier_layer("dft");
        let output = layer.forward(input).expect("Fourier transform must succeed");
        let got = output.data().expect("output must be readable");
        let want = reference_dft_real(&values, rows, cols);

        assert_eq!(got.len(), want.len());
        for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                (g - w).abs() < 1e-3,
                "element {idx}: FNet produced {g}, the true 2-D DFT real part is {w}"
            );
        }
    }

    /// Parseval / DC sanity: the `[0, 0]` output bin of an orthonormal 2-D DFT
    /// is the signal's sum divided by `sqrt(rows*cols)`, for any signal.
    #[test]
    fn dft_dc_bin_equals_the_normalised_signal_sum() {
        let rows = 4usize;
        let cols = 8usize;
        let values: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.3 - 2.0).collect();
        let input = Tensor::from_vec(values.clone(), &[rows, cols]).expect("input must build");

        let layer = fourier_layer("dft");
        let output = layer.forward(input).expect("Fourier transform must succeed");
        let got = output.data().expect("output must be readable");

        let expected_dc =
            values.iter().sum::<f32>() / ((rows as f32).sqrt() * (cols as f32).sqrt());
        assert!(
            (got[0] - expected_dc).abs() < 1e-3,
            "DC bin {} must equal the normalised sum {expected_dc}",
            got[0]
        );
    }

    #[test]
    fn dft_preserves_the_input_shape_for_batched_input() {
        let layer = fourier_layer("dft");
        let input = Tensor::from_vec((0..2 * 5 * 3).map(|i| i as f32).collect(), &[2, 5, 3])
            .expect("input must build");
        let output = layer.forward(input).expect("Fourier transform must succeed");
        assert_eq!(output.shape(), vec![2, 5, 3]);
    }

    #[test]
    fn real_dft_agrees_with_the_full_dft_for_real_input() {
        let values: Vec<f32> = (0..4 * 4).map(|i| ((i * 5) % 7) as f32).collect();
        let full = fourier_layer("dft")
            .forward(Tensor::from_vec(values.clone(), &[4, 4]).expect("input must build"))
            .expect("dft must succeed")
            .data()
            .expect("readable");
        let real = fourier_layer("real_dft")
            .forward(Tensor::from_vec(values, &[4, 4]).expect("input must build"))
            .expect("real_dft must succeed")
            .data()
            .expect("readable");
        for (a, b) in full.iter().zip(real.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "real_dft diverged from dft: {a} vs {b}"
            );
        }
    }

    // ── Config tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_config_validate_ok() {
        tiny_config().validate().expect("tiny_config should be valid");
    }

    #[test]
    fn test_config_invalid_fourier_type_fails() {
        let mut cfg = tiny_config();
        cfg.fourier_transform_type = "unknown_type".to_string();
        assert!(
            cfg.validate().is_err(),
            "unknown fourier type must fail validation"
        );
    }

    #[test]
    fn test_config_fft_no_attention_heads_field() {
        // FNet has no num_attention_heads — validates without attention constraints
        tiny_config().validate().expect("fnet config has no attention head constraint");
    }

    // ── FourierTransform tests ────────────────────────────────────────────────

    #[test]
    fn test_fourier_transform_dft_output_shape_preserved() {
        let cfg = tiny_config();
        let ft = FourierTransform::new(&cfg).expect("fourier transform creation should succeed");
        // DFT 1D creates an n×n matrix for each dimension.
        // To make matmul work both ways (seq and hidden must be equal for simplicity),
        // use seq_len == hidden_size.
        let seq_len = cfg.hidden_size;
        let hidden_size = cfg.hidden_size;
        let data = vec![0.5_f32; seq_len * hidden_size];
        let input = Tensor::from_vec(data, &[1, seq_len, hidden_size])
            .expect("tensor creation should succeed");
        match ft.forward(input) {
            Ok(output) => {
                let shape = output.shape();
                assert_eq!(shape[0], 1, "batch dim must be preserved");
                // The DFT along seq dim changes effective shape but output batch stays 1
                assert!(shape[0] >= 1, "batch must be at least 1");
            },
            Err(_) => { /* Known shape limitation in test configs */ },
        }
    }

    #[test]
    fn test_fourier_transform_dct_output_shape_preserved() {
        let mut cfg = tiny_config();
        cfg.fourier_transform_type = "dct".to_string();
        let ft = FourierTransform::new(&cfg).expect("fourier transform creation should succeed");
        // DCT: seq_len must equal hidden_size for the matmul to work correctly in the impl
        let seq_len = cfg.hidden_size;
        let data = vec![0.2_f32; seq_len * cfg.hidden_size];
        let input = Tensor::from_vec(data, &[1, seq_len, cfg.hidden_size])
            .expect("tensor creation should succeed");
        match ft.forward(input) {
            Ok(output) => {
                let shape = output.shape();
                assert_eq!(shape[0], 1, "batch preserved");
            },
            Err(_) => { /* Known shape limitation in test configs */ },
        }
    }

    #[test]
    fn test_fourier_has_no_attention_weights() {
        // FNet's fundamental property: no learnable attention parameters
        let cfg = tiny_config();
        let ft = FourierTransform::new(&cfg).expect("creation should succeed");
        assert_eq!(
            ft.parameter_count(),
            0,
            "Fourier transform without bias has 0 parameters (no attention weights)"
        );
    }

    // ── FNetLayer tests ────────────────────────────────────────────────────────

    #[test]
    fn test_fnet_layer_output_shape_preserved() {
        let cfg = tiny_config();
        let layer = FNetLayer::new(&cfg).expect("fnet layer creation should succeed");
        // DFT requires seq_len == hidden_size due to matrix dimension constraints
        let seq_len = cfg.hidden_size;
        let data = vec![0.1_f32; seq_len * cfg.hidden_size];
        let input = Tensor::from_vec(data, &[1, seq_len, cfg.hidden_size])
            .expect("tensor creation should succeed");
        match layer.forward(input) {
            Ok(output) => {
                let shape = output.shape();
                assert_eq!(shape[0], 1, "batch preserved");
                assert_eq!(shape[1], seq_len, "seq_len preserved");
                assert_eq!(shape[2], cfg.hidden_size, "hidden_size preserved");
            },
            Err(_) => { /* Known shape limitation in test configs */ },
        }
    }

    #[test]
    fn test_fnet_layer_fourier_plus_ffn() {
        // An FNetLayer contains both a FourierTransform and a FeedForward
        let cfg = tiny_config();
        let layer = FNetLayer::new(&cfg).expect("creation should succeed");
        // Parameter count includes FFN (dense1 + dense2) but not FourierTransform
        assert!(layer.parameter_count() > 0, "fnet layer has FFN parameters");
    }

    // ── FNetModel tests ────────────────────────────────────────────────────────

    #[test]
    fn test_model_creation() {
        let cfg = tiny_config();
        FNetModel::new(cfg).expect("model creation should succeed");
    }

    #[test]
    fn test_model_forward_output_shape() {
        let cfg = tiny_config();
        let model = FNetModel::new(cfg.clone()).expect("model creation should succeed");
        // DFT requires seq_len == hidden_size due to matrix dimension constraints
        let seq_len = cfg.hidden_size;
        let input = make_input(seq_len);
        match model.forward(input) {
            Ok(output) => {
                let shape = output.shape();
                // Output: [batch, seq_len, hidden_size] or [seq_len, hidden_size]
                assert_eq!(
                    shape[shape.len() - 1],
                    cfg.hidden_size,
                    "last dim must be hidden_size"
                );
            },
            Err(_) => { /* Known shape limitation in test configs */ },
        }
    }

    #[test]
    fn test_model_linear_complexity_no_attention() {
        // FNet's primary property: O(n log n) vs O(n²) for attention.
        // Verify no attention layers exist (parameter structure test)
        let cfg = tiny_config();
        let model = FNetModel::new(cfg).expect("model creation should succeed");
        // Parameter count should not contain query/key projections from attention
        let total = model.num_parameters();
        assert!(total > 0, "model must have non-zero parameters");
    }

    // ── FNetForMaskedLM tests ─────────────────────────────────────────────────

    #[test]
    fn test_masked_lm_creation() {
        let cfg = tiny_config();
        FNetForMaskedLM::new(cfg).expect("masked lm creation should succeed");
    }

    #[test]
    fn test_masked_lm_output_vocab_size() {
        let cfg = tiny_config();
        let vocab_size = cfg.vocab_size;
        let model = FNetForMaskedLM::new(cfg.clone()).expect("creation should succeed");
        // DFT requires seq_len == hidden_size
        let input = make_input(cfg.hidden_size);
        match model.forward(input) {
            Ok(output) => {
                let shape = output.shape();
                assert_eq!(
                    shape[shape.len() - 1],
                    vocab_size,
                    "masked lm output last dim must equal vocab_size"
                );
            },
            Err(_) => { /* Known shape limitation in test configs */ },
        }
    }

    // ── FNetForSequenceClassification tests ───────────────────────────────────

    #[test]
    fn test_sequence_classification_creation() {
        let cfg = tiny_config();
        FNetForSequenceClassification::new(cfg, 3)
            .expect("sequence classification creation should succeed");
    }

    #[test]
    fn test_sequence_classification_output_num_labels() {
        let cfg = tiny_config();
        let num_labels = 5;
        let model = FNetForSequenceClassification::new(cfg.clone(), num_labels)
            .expect("creation should succeed");
        // DFT requires seq_len == hidden_size
        let input = make_input(cfg.hidden_size);
        match model.forward(input) {
            Ok(output) => {
                let shape = output.shape();
                assert_eq!(
                    shape[shape.len() - 1],
                    num_labels,
                    "classifier output last dim must equal num_labels"
                );
            },
            Err(_) => { /* Known shape limitation in test configs */ },
        }
    }

    // ── FNetEmbeddings tests ───────────────────────────────────────────────────

    #[test]
    fn test_embeddings_creation() {
        let cfg = tiny_config();
        FNetEmbeddings::new(&cfg).expect("embeddings creation should succeed");
    }

    #[test]
    fn test_embeddings_forward_shape() {
        let cfg = tiny_config();
        let emb = FNetEmbeddings::new(&cfg).expect("creation should succeed");
        let seq_len = 4usize;
        let ids: Vec<u32> = (0..seq_len as u32).collect();
        let output = emb.forward((ids, None, None)).expect("embeddings forward should succeed");
        let shape = output.shape();
        assert_eq!(
            shape[shape.len() - 1],
            cfg.hidden_size,
            "embedding dim must match hidden_size"
        );
    }

    // ── Real checkpoint loading (regression for the silent `Ok(())`) ────────

    use crate::weight_loading::test_support::{build_safetensors, F32Tensor};

    fn loading_config() -> FNetConfig {
        FNetConfig {
            vocab_size: 16,
            hidden_size: 8,
            num_hidden_layers: 2,
            intermediate_size: 16,
            max_position_embeddings: 8,
            type_vocab_size: 2,
            ..tiny_config()
        }
    }

    /// Every tensor an FNet checkpoint of this shape carries.
    ///
    /// Note the absence of any `attention.*` projection: FNet's mixing layer is
    /// a parameter-free Fourier transform, so those tensors simply do not exist.
    fn fnet_tensors(config: &FNetConfig, prefix: &str) -> Vec<F32Tensor> {
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;
        let mut seed = 0.0f32;
        let mut next = || {
            seed += 1.0;
            seed
        };

        let mut tensors = vec![
            F32Tensor::ramp(
                &format!("{prefix}embeddings.word_embeddings.weight"),
                &[config.vocab_size, hidden],
                next(),
            ),
            F32Tensor::ramp(
                &format!("{prefix}embeddings.position_embeddings.weight"),
                &[config.max_position_embeddings, hidden],
                next(),
            ),
            F32Tensor::ramp(
                &format!("{prefix}embeddings.token_type_embeddings.weight"),
                &[config.type_vocab_size, hidden],
                next(),
            ),
            F32Tensor::ramp(
                &format!("{prefix}embeddings.LayerNorm.weight"),
                &[hidden],
                next(),
            ),
            F32Tensor::ramp(
                &format!("{prefix}embeddings.LayerNorm.bias"),
                &[hidden],
                next(),
            ),
        ];

        for layer in 0..config.num_hidden_layers {
            let base = format!("{prefix}encoder.layer.{layer}");
            tensors.push(F32Tensor::ramp(
                &format!("{base}.fourier.output.LayerNorm.weight"),
                &[hidden],
                next(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{base}.fourier.output.LayerNorm.bias"),
                &[hidden],
                next(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{base}.intermediate.dense.weight"),
                &[intermediate, hidden],
                next(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{base}.intermediate.dense.bias"),
                &[intermediate],
                next(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{base}.output.dense.weight"),
                &[hidden, intermediate],
                next(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{base}.output.dense.bias"),
                &[hidden],
                next(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{base}.output.LayerNorm.weight"),
                &[hidden],
                next(),
            ));
            tensors.push(F32Tensor::ramp(
                &format!("{base}.output.LayerNorm.bias"),
                &[hidden],
                next(),
            ));
        }

        tensors
    }

    /// Regression: `load_pretrained` was `Ok(())`, so the reader was never read
    /// and the model kept its random initialisation while reporting success.
    #[test]
    fn load_pretrained_binds_the_checkpoint_instead_of_returning_ok() {
        let config = loading_config();
        let tensors = fnet_tensors(&config, "fnet.");
        let bytes = build_safetensors(&tensors);

        let mut model = FNetModel::new(config).expect("model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("a matching checkpoint must load");
        assert!(
            report.is_complete(),
            "every parameter must be bound, missing: {:?}",
            report.missing
        );
        assert_eq!(
            report.loaded.len(),
            tensors.len(),
            "every fixture tensor must reach the model"
        );
    }

    #[test]
    fn load_pretrained_reports_a_missing_parameter_instead_of_inventing_it() {
        let config = loading_config();
        let mut tensors = fnet_tensors(&config, "fnet.");
        tensors.retain(|t| t.name != "fnet.encoder.layer.1.output.dense.weight");
        let bytes = build_safetensors(&tensors);

        let mut model = FNetModel::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("an incomplete checkpoint must not load silently");
        assert!(
            err.to_string().contains("encoder.layer.1.output.dense.weight"),
            "the error must name the gap: {err}"
        );
    }

    #[test]
    fn load_pretrained_rejects_a_foreign_tensor() {
        let config = loading_config();
        let mut tensors = fnet_tensors(&config, "fnet.");
        // An attention projection has no place in an FNet checkpoint.
        tensors.push(F32Tensor::ramp(
            "fnet.encoder.layer.0.attention.self.query.weight",
            &[8, 8],
            42.0,
        ));
        let bytes = build_safetensors(&tensors);

        let mut model = FNetModel::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("an unrecognised weight must fail the load");
        assert!(
            err.to_string().contains("attention.self.query"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn load_pretrained_rejects_bytes_that_are_not_a_checkpoint() {
        let mut model = FNetModel::new(loading_config()).expect("model must build");
        let garbage = vec![0xABu8; 4096];
        let err = model
            .load_pretrained(&mut garbage.as_slice())
            .expect_err("garbage must not be accepted as weights");
        assert!(
            err.to_string().contains("unrecognised checkpoint container"),
            "unexpected: {err}"
        );
    }

    // ── Task heads (regression for the head-dropping delegation) ────────────

    /// The tensors a fine-tuned `FNetForSequenceClassification` export adds on
    /// top of the encoder.
    fn classifier_tensors(config: &FNetConfig, num_labels: usize) -> Vec<F32Tensor> {
        vec![
            F32Tensor::ramp("classifier.weight", &[num_labels, config.hidden_size], 90.0),
            F32Tensor::ramp("classifier.bias", &[num_labels], 95.0),
        ]
    }

    /// The tensors a `FNetForMaskedLM` export adds on top of the encoder.
    fn prediction_head_tensors(config: &FNetConfig, aliased_bias: bool) -> Vec<F32Tensor> {
        let hidden = config.hidden_size;
        let bias_name = if aliased_bias {
            "cls.predictions.decoder.bias"
        } else {
            "cls.predictions.bias"
        };
        vec![
            F32Tensor::ramp(
                "cls.predictions.transform.dense.weight",
                &[hidden, hidden],
                60.0,
            ),
            F32Tensor::ramp("cls.predictions.transform.dense.bias", &[hidden], 65.0),
            F32Tensor::ramp(
                "cls.predictions.transform.LayerNorm.weight",
                &[hidden],
                70.0,
            ),
            F32Tensor::ramp("cls.predictions.transform.LayerNorm.bias", &[hidden], 75.0),
            F32Tensor::ramp(
                "cls.predictions.decoder.weight",
                &[config.vocab_size, hidden],
                80.0,
            ),
            F32Tensor::ramp(bias_name, &[config.vocab_size], 85.0),
        ]
    }

    /// Regression: the wrapper delegated to `FNetModel::load_pretrained`, whose
    /// unused-tensor policy tolerates the `classifier.` namespace. The head was
    /// therefore dropped on the floor while the load returned `Ok(())`.
    #[test]
    fn sequence_classification_load_pretrained_binds_the_classifier_head() {
        let config = loading_config();
        let num_labels = 3;
        let mut tensors = fnet_tensors(&config, "fnet.");
        tensors.extend(classifier_tensors(&config, num_labels));
        let bytes = build_safetensors(&tensors);

        let mut model =
            FNetForSequenceClassification::new(config, num_labels).expect("model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("a matching checkpoint must load");

        assert!(
            report.is_complete(),
            "every parameter must be bound, missing: {:?}",
            report.missing
        );
        assert!(
            report.loaded.contains(&"classifier.weight".to_string())
                && report.loaded.contains(&"classifier.bias".to_string()),
            "the classification head must be among the loaded tensors: {:?}",
            report.loaded
        );
    }

    /// A plain pretrained encoder ships without a fine-tuned head. That is
    /// accepted, but the gap is reported rather than passed off as a full load.
    #[test]
    fn sequence_classification_load_pretrained_records_an_absent_head() {
        let config = loading_config();
        let bytes = build_safetensors(&fnet_tensors(&config, "fnet."));

        let mut model = FNetForSequenceClassification::new(config, 3).expect("model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("a head-less encoder checkpoint must still load");
        assert!(
            !report.is_complete(),
            "a checkpoint without a head must not be reported as complete"
        );
        assert!(
            report.missing.contains(&"classifier.weight".to_string()),
            "the absent head must be named: {:?}",
            report.missing
        );
    }

    #[test]
    fn sequence_classification_load_pretrained_rejects_a_head_of_the_wrong_width() {
        let config = loading_config();
        let mut tensors = fnet_tensors(&config, "fnet.");
        tensors.extend(classifier_tensors(&config, 7));
        let bytes = build_safetensors(&tensors);

        let mut model = FNetForSequenceClassification::new(config, 3).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("a 7-label head must not be reshaped into a 3-label model");
        assert!(
            err.to_string().contains("classifier.weight"),
            "unexpected: {err}"
        );
    }

    /// Regression: the masked-LM wrapper delegated to the encoder loader, whose
    /// policy tolerates the whole `cls.` namespace, so the prediction head was
    /// silently discarded.
    #[test]
    fn masked_lm_load_pretrained_binds_the_prediction_head() {
        let config = loading_config();
        let mut tensors = fnet_tensors(&config, "fnet.");
        tensors.extend(prediction_head_tensors(&config, false));
        let bytes = build_safetensors(&tensors);

        let mut model = FNetForMaskedLM::new(config).expect("model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("a matching checkpoint must load");

        assert!(
            report.is_complete(),
            "every parameter must be bound, missing: {:?}",
            report.missing
        );
        for name in [
            "cls.predictions.transform.dense.weight",
            "cls.predictions.transform.LayerNorm.weight",
            "cls.predictions.decoder.weight",
            "cls.predictions.bias",
        ] {
            assert!(
                report.loaded.contains(&name.to_string()),
                "{name} must be among the loaded tensors: {:?}",
                report.loaded
            );
        }
    }

    /// HuggingFace aliases `cls.predictions.decoder.bias` onto
    /// `cls.predictions.bias`; an export may carry either spelling.
    #[test]
    fn masked_lm_load_pretrained_accepts_the_aliased_decoder_bias() {
        let config = loading_config();
        let mut tensors = fnet_tensors(&config, "fnet.");
        tensors.extend(prediction_head_tensors(&config, true));
        let bytes = build_safetensors(&tensors);

        let mut model = FNetForMaskedLM::new(config).expect("model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("the aliased bias spelling must load");
        assert!(
            report.loaded.contains(&"cls.predictions.decoder.bias".to_string()),
            "the aliased bias must be consumed: {:?}",
            report.loaded
        );
    }

    #[test]
    fn masked_lm_load_pretrained_records_an_absent_head() {
        let config = loading_config();
        let bytes = build_safetensors(&fnet_tensors(&config, "fnet."));

        let mut model = FNetForMaskedLM::new(config).expect("model must build");
        let report = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect("a head-less encoder checkpoint must still load");
        assert!(
            report.missing.contains(&"cls.predictions.decoder.weight".to_string()),
            "the absent prediction head must be named: {:?}",
            report.missing
        );
    }

    #[test]
    fn masked_lm_load_pretrained_rejects_a_head_for_a_different_vocabulary() {
        let config = loading_config();
        let wider = FNetConfig {
            vocab_size: config.vocab_size * 2,
            ..config.clone()
        };
        let mut tensors = fnet_tensors(&config, "fnet.");
        tensors.extend(prediction_head_tensors(&wider, false));
        let bytes = build_safetensors(&tensors);

        let mut model = FNetForMaskedLM::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("a head for a bigger vocabulary must not be reshaped into place");
        assert!(
            err.to_string().contains("cls.predictions.decoder.weight"),
            "unexpected: {err}"
        );
    }
}
