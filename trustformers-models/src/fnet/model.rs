use crate::fnet::config::FNetConfig;
use std::io::Read;
use trustformers_core::{
    errors::Result,
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
}

impl FourierTransform {
    pub fn new(config: &FNetConfig) -> Result<Self> {
        let bias = if config.use_bias_in_fourier {
            Some(Linear::new(config.hidden_size, config.hidden_size, true))
        } else {
            None
        };

        Ok(Self {
            fourier_type: config.fourier_transform_type.clone(),
            use_bias: config.use_bias_in_fourier,
            bias,
            dropout: config.fourier_dropout_prob,
        })
    }

    pub fn parameter_count(&self) -> usize {
        if let Some(ref bias_layer) = self.bias {
            bias_layer.parameter_count()
        } else {
            0
        }
    }

    /// Apply Discrete Fourier Transform (DFT)
    fn apply_dft(&self, x: &Tensor) -> Result<Tensor> {
        // x: [batch_size, seq_len, hidden_size]
        let _batch_size = x.shape()[0];
        let _seq_len = x.shape()[1];
        let _hidden_size = x.shape()[2];

        // Apply DFT along sequence dimension first
        let x_seq_dft = self.dft_1d(x, 1)?; // DFT along dimension 1 (seq_len)

        // Apply DFT along hidden dimension
        let x_both_dft = self.dft_1d(&x_seq_dft, 2)?; // DFT along dimension 2 (hidden_size)

        // Take real part only (common practice in FNet)
        self.real_part(&x_both_dft)
    }

    /// Apply Real DFT (more efficient variant)
    fn apply_real_dft(&self, x: &Tensor) -> Result<Tensor> {
        // Similar to DFT but optimized for real inputs
        // For simplicity, we'll implement this as regular DFT taking real part
        self.apply_dft(x)
    }

    /// Apply Discrete Cosine Transform (DCT)
    fn apply_dct(&self, x: &Tensor) -> Result<Tensor> {
        // DCT is real-valued and often more efficient than DFT
        // For now, approximate with cosine-based transformation
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        let hidden_size = x.shape()[2];

        // Create DCT basis matrices
        let seq_dct_matrix = self.create_dct_matrix(seq_len)?;
        let hidden_dct_matrix = self.create_dct_matrix(hidden_size)?;

        // Apply DCT along sequence dimension
        // x @ seq_dct_matrix^T
        let seq_shape = seq_dct_matrix.shape();
        let seq_dim0 = seq_shape.len().saturating_sub(2);
        let seq_dim1 = seq_shape.len().saturating_sub(1);
        let x_seq_dct = x.matmul(&seq_dct_matrix.transpose(seq_dim0, seq_dim1)?)?;

        // Apply DCT along hidden dimension
        // For hidden dimension: reshape, apply DCT, reshape back
        let reshaped = x_seq_dct.reshape(&[batch_size * seq_len, hidden_size])?;
        let hidden_shape = hidden_dct_matrix.shape();
        let hidden_dim0 = hidden_shape.len().saturating_sub(2);
        let hidden_dim1 = hidden_shape.len().saturating_sub(1);
        let hidden_dct =
            reshaped.matmul(&hidden_dct_matrix.transpose(hidden_dim0, hidden_dim1)?)?;
        hidden_dct.reshape(&[batch_size, seq_len, hidden_size])
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

    /// 1D DFT implementation (simplified)
    fn dft_1d(&self, x: &Tensor, dim: i32) -> Result<Tensor> {
        // This is a simplified implementation
        // In practice, you'd use an efficient FFT library

        let shape = x.shape();
        let n = shape[dim as usize];

        // For simplicity, we'll approximate DFT with a learned transformation
        // that captures the frequency domain mixing behavior

        // Create a pseudo-DFT matrix that mixes elements
        let mut dft_matrix = Vec::new();
        let pi = std::f32::consts::PI;

        for k in 0..n {
            for j in 0..n {
                let angle = -2.0 * pi * (k * j) as f32 / n as f32;
                let real_part = angle.cos() / (n as f32).sqrt();
                dft_matrix.push(real_part);
            }
        }

        let dft_tensor = Tensor::from_vec(dft_matrix, &[n, n])?;

        // Apply transformation along the specified dimension
        if dim == 1 {
            // Along sequence dimension
            let dft_shape = dft_tensor.shape();
            let dft_dim0 = dft_shape.len().saturating_sub(2);
            let dft_dim1 = dft_shape.len().saturating_sub(1);
            x.matmul(&dft_tensor.transpose(dft_dim0, dft_dim1)?)
        } else {
            // Along hidden dimension - need to reshape
            let batch_size = shape[0];
            let seq_len = shape[1];
            let hidden_size = shape[2];

            let reshaped = x.reshape(&[batch_size * seq_len, hidden_size])?;
            let dft_shape = dft_tensor.shape();
            let dft_dim0 = dft_shape.len().saturating_sub(2);
            let dft_dim1 = dft_shape.len().saturating_sub(1);
            let transformed = reshaped.matmul(&dft_tensor.transpose(dft_dim0, dft_dim1)?)?;
            transformed.reshape(&[batch_size, seq_len, hidden_size])
        }
    }

    /// Extract real part of complex tensor
    fn real_part(&self, x: &Tensor) -> Result<Tensor> {
        // Since we're working with real tensors, just return as-is
        // In a full implementation, this would handle complex numbers
        Ok(x.clone())
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
    dense1: Linear,
    dense2: Linear,
    activation: String,
    #[allow(dead_code)]
    dropout: f32,
}

impl FNetFeedForward {
    pub fn new(config: &FNetConfig) -> Result<Self> {
        let dense1 = Linear::new(config.hidden_size, config.intermediate_size, true);
        let dense2 = Linear::new(config.intermediate_size, config.hidden_size, true);

        Ok(Self {
            dense1,
            dense2,
            activation: config.hidden_act.clone(),
            dropout: config.hidden_dropout_prob,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.dense1.parameter_count() + self.dense2.parameter_count()
    }

    fn apply_activation(&self, x: &Tensor) -> Result<Tensor> {
        match self.activation.as_str() {
            "gelu" => x.gelu(),
            "relu" => x.relu(),
            "silu" | "swish" => x.silu(),
            _ => Ok(x.clone()),
        }
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
    feed_forward: FNetFeedForward,
    fourier_norm: LayerNorm,
    output_norm: LayerNorm,
}

impl FNetLayer {
    pub fn new(config: &FNetConfig) -> Result<Self> {
        let fourier_transform = FourierTransform::new(config)?;
        let feed_forward = FNetFeedForward::new(config)?;
        let fourier_norm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?;
        let output_norm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?;

        Ok(Self {
            fourier_transform,
            feed_forward,
            fourier_norm,
            output_norm,
        })
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
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    #[allow(dead_code)]
    dropout: f32,
}

impl FNetEmbeddings {
    pub fn new(config: &FNetConfig) -> Result<Self> {
        let word_embeddings = Embedding::new(
            config.vocab_size,
            config.hidden_size,
            Some(config.pad_token_id as usize),
        )?;
        let position_embeddings =
            Embedding::new(config.max_position_embeddings, config.hidden_size, None)?;
        let token_type_embeddings =
            Embedding::new(config.type_vocab_size, config.hidden_size, None)?;
        let layer_norm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout: config.hidden_dropout_prob,
        })
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
    layers: Vec<FNetLayer>,
}

impl FNetEncoder {
    pub fn new(config: &FNetConfig) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(FNetLayer::new(config)?);
        }

        Ok(Self { layers })
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
}

impl FNetModel {
    pub fn new(config: FNetConfig) -> Result<Self> {
        config.validate()?;

        let embeddings = FNetEmbeddings::new(&config)?;
        let encoder = FNetEncoder::new(&config)?;

        Ok(Self {
            config,
            embeddings,
            encoder,
        })
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

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Ok(())
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
}

impl FNetForSequenceClassification {
    pub fn new(config: FNetConfig, num_labels: usize) -> Result<Self> {
        let fnet = FNetModel::new(config.clone())?;
        let classifier = Linear::new(config.hidden_size, num_labels, true);

        Ok(Self {
            fnet,
            classifier,
            num_labels,
        })
    }
}

impl Model for FNetForSequenceClassification {
    type Config = FNetConfig;
    type Input = (Vec<u32>, Option<Vec<u32>>, Option<Vec<u32>>);
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let sequence_output = self.fnet.forward(input)?;
        let cls_output = sequence_output.slice(1, 0, 1)?; // Get first token (CLS) from sequence
        let cls_output = cls_output.squeeze(1)?; // Remove singleton sequence dimension
        self.classifier.forward(cls_output)
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.fnet.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.fnet.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.fnet.num_parameters() + self.classifier.parameter_count()
    }
}

/// FNet for masked language modeling
pub struct FNetForMaskedLM {
    fnet: FNetModel,
    mlm_head: Linear,
}

impl FNetForMaskedLM {
    pub fn new(config: FNetConfig) -> Result<Self> {
        let fnet = FNetModel::new(config.clone())?;
        let mlm_head = Linear::new(config.hidden_size, config.vocab_size, true);

        Ok(Self { fnet, mlm_head })
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

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.fnet.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.fnet.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.fnet.num_parameters() + self.mlm_head.parameter_count()
    }
}
