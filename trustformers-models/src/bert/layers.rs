use crate::bert::config::BertConfig;
use scirs2_core::ndarray::s; // SciRS2 Integration Policy
use trustformers_core::device::Device;
use trustformers_core::errors::{tensor_op_error, Result};
use trustformers_core::layers::{FeedForward, LayerNorm, MultiHeadAttention};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Layer;

#[derive(Debug, Clone)]
pub struct BertEmbeddings {
    word_embeddings: trustformers_core::layers::Embedding,
    position_embeddings: trustformers_core::layers::Embedding,
    token_type_embeddings: trustformers_core::layers::Embedding,
    layer_norm: LayerNorm,
    #[allow(dead_code)]
    dropout_prob: f32,
    device: Device,
}

impl BertEmbeddings {
    pub fn new(config: &BertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &BertConfig, device: Device) -> Result<Self> {
        Ok(Self {
            word_embeddings: trustformers_core::layers::Embedding::new_with_device(
                config.vocab_size,
                config.hidden_size,
                Some(config.pad_token_id as usize),
                device,
            )?,
            position_embeddings: trustformers_core::layers::Embedding::new_with_device(
                config.max_position_embeddings,
                config.hidden_size,
                None,
                device,
            )?,
            token_type_embeddings: trustformers_core::layers::Embedding::new_with_device(
                config.type_vocab_size,
                config.hidden_size,
                None,
                device,
            )?,
            layer_norm: LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            dropout_prob: config.hidden_dropout_prob,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(&self, input_ids: Vec<u32>, token_type_ids: Option<Vec<u32>>) -> Result<Tensor> {
        let seq_length = input_ids.len();
        let position_ids: Vec<u32> = (0..seq_length as u32).collect();

        let word_embeddings = self.word_embeddings.forward(input_ids)?;
        let position_embeddings = self.position_embeddings.forward(position_ids)?;

        let mut embeddings = word_embeddings.add(&position_embeddings)?;

        if let Some(token_type_ids) = token_type_ids {
            let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;
            embeddings = embeddings.add(&token_type_embeddings)?;
        }

        self.layer_norm.forward(embeddings)
    }

    pub fn parameter_count(&self) -> usize {
        self.word_embeddings.parameter_count()
            + self.position_embeddings.parameter_count()
            + self.token_type_embeddings.parameter_count()
            + self.layer_norm.parameter_count()
    }
}

#[derive(Debug, Clone)]
pub struct BertLayer {
    attention: BertAttention,
    intermediate: FeedForward,
    output_layer_norm: LayerNorm,
    device: Device,
}

impl BertLayer {
    pub fn new(config: &BertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &BertConfig, device: Device) -> Result<Self> {
        Ok(Self {
            attention: BertAttention::new_with_device(config, device)?,
            intermediate: FeedForward::new_with_device(
                config.hidden_size,
                config.intermediate_size,
                config.hidden_dropout_prob,
                device,
            ),
            output_layer_norm: LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn parameter_count(&self) -> usize {
        self.attention.parameter_count()
            + self.intermediate.parameter_count()
            + self.output_layer_norm.parameter_count()
    }
}

impl Layer for BertLayer {
    type Input = (Tensor, Option<Tensor>);
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let (hidden_states, attention_mask) = input;

        let attention_output = self.attention.forward((hidden_states.clone(), attention_mask))?;
        let intermediate_output = self.intermediate.forward(attention_output.clone())?;

        let layer_output = intermediate_output.add(&attention_output)?;
        self.output_layer_norm.forward(layer_output)
    }
}

#[derive(Debug, Clone)]
pub struct BertAttention {
    self_attention: MultiHeadAttention,
    output_layer_norm: LayerNorm,
    device: Device,
}

impl BertAttention {
    pub fn new(config: &BertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &BertConfig, device: Device) -> Result<Self> {
        Ok(Self {
            self_attention: MultiHeadAttention::new_with_device(
                config.hidden_size,
                config.num_attention_heads,
                config.attention_probs_dropout_prob,
                true,
                device,
            )?,
            output_layer_norm: LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(&self, input: (Tensor, Option<Tensor>)) -> Result<Tensor> {
        let (hidden_states, attention_mask) = input;

        let attention_output = self.self_attention.forward_self_attention(
            &hidden_states,
            attention_mask.as_ref(),
            false, // causal
        )?;
        let output = attention_output.add(&hidden_states)?;
        self.output_layer_norm.forward(output)
    }

    pub fn parameter_count(&self) -> usize {
        self.self_attention.parameter_count() + self.output_layer_norm.parameter_count()
    }
}

#[derive(Debug, Clone)]
pub struct BertEncoder {
    layers: Vec<BertLayer>,
    device: Device,
}

impl BertEncoder {
    pub fn new(config: &BertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &BertConfig, device: Device) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(BertLayer::new_with_device(config, device)?);
        }
        Ok(Self { layers, device })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(&self, hidden_states: Tensor, attention_mask: Option<Tensor>) -> Result<Tensor> {
        let mut hidden_states = hidden_states;

        for layer in &self.layers {
            hidden_states = layer.forward((hidden_states, attention_mask.clone()))?;
        }

        Ok(hidden_states)
    }

    pub fn parameter_count(&self) -> usize {
        self.layers.iter().map(|layer| layer.parameter_count()).sum()
    }
}

#[derive(Debug, Clone)]
pub struct BertPooler {
    dense: trustformers_core::layers::Linear,
    device: Device,
}

impl BertPooler {
    pub fn new(config: &BertConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &BertConfig, device: Device) -> Result<Self> {
        Ok(Self {
            dense: trustformers_core::layers::Linear::new_with_device(
                config.hidden_size,
                config.hidden_size,
                true,
                device,
            ),
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn parameter_count(&self) -> usize {
        self.dense.parameter_count()
    }
}

impl Layer for BertPooler {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        match &input {
            Tensor::F32(arr) => {
                // Input shape is [seq_len, hidden_size] (2D)
                // We want to extract the first token: [1, hidden_size]
                let shape = arr.shape();
                if shape.len() != 2 {
                    return Err(tensor_op_error(
                        "tensor_operation",
                        format!(
                            "BertPooler expects 2D input, got {} dimensions",
                            shape.len()
                        ),
                    ));
                }

                // Extract first token and keep it 2D: [1, hidden_size]
                let first_token = arr.slice(s![0..1, ..]).to_owned().into_dyn();
                let pooled = self.dense.forward(Tensor::F32(first_token))?;
                trustformers_core::ops::activations::tanh(&pooled)
            },
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Unsupported tensor type for pooler".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bert::config::BertConfig;
    use trustformers_core::device::Device;
    use trustformers_core::traits::Layer;

    // --- LCG for deterministic pseudo-random data ---
    fn lcg_next(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
        *state
    }

    fn lcg_f32(state: &mut u64) -> f32 {
        let v = lcg_next(state);
        ((v >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    fn small_config() -> BertConfig {
        BertConfig {
            vocab_size: 512,
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            intermediate_size: 256,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            max_position_embeddings: 32,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: Some("absolute".to_string()),
            use_cache: Some(true),
            classifier_dropout: None,
        }
    }

    // --- BertEmbeddings ---

    #[test]
    fn test_bert_embeddings_new_cpu() {
        let cfg = small_config();
        let embeddings = BertEmbeddings::new(&cfg).expect("BertEmbeddings::new must succeed");
        assert_eq!(embeddings.device(), Device::CPU);
    }

    #[test]
    fn test_bert_embeddings_forward_output_shape() {
        let cfg = small_config();
        let embeddings = BertEmbeddings::new(&cfg).expect("BertEmbeddings::new must succeed");
        let seq_len = 8usize;
        let input_ids: Vec<u32> = (0..seq_len as u32).collect();
        let output = embeddings
            .forward(input_ids, None)
            .expect("BertEmbeddings forward must succeed");
        let shape = output.shape();
        assert_eq!(shape[0], seq_len, "First dim must equal seq_len");
        assert_eq!(
            shape[1], cfg.hidden_size,
            "Second dim must equal hidden_size"
        );
    }

    #[test]
    fn test_bert_embeddings_with_token_type_ids() {
        let cfg = small_config();
        let embeddings = BertEmbeddings::new(&cfg).expect("BertEmbeddings::new must succeed");
        let seq_len = 6usize;
        let input_ids: Vec<u32> = (0..seq_len as u32).collect();
        let token_type_ids: Vec<u32> = vec![0, 0, 0, 1, 1, 1];
        let output = embeddings
            .forward(input_ids, Some(token_type_ids))
            .expect("BertEmbeddings with token_type_ids must succeed");
        let shape = output.shape();
        assert_eq!(shape[0], seq_len);
        assert_eq!(shape[1], cfg.hidden_size);
    }

    #[test]
    fn test_bert_embeddings_parameter_count_positive() {
        let cfg = small_config();
        let embeddings = BertEmbeddings::new(&cfg).expect("BertEmbeddings::new must succeed");
        let params = embeddings.parameter_count();
        assert!(params > 0, "Embedding parameter count must be positive");
    }

    #[test]
    fn test_bert_embeddings_parameter_count_includes_all_tables() {
        let cfg = small_config();
        let embeddings = BertEmbeddings::new(&cfg).expect("BertEmbeddings::new must succeed");
        // word + position + token_type + layer_norm (2 * hidden_size)
        let min_expected = cfg.vocab_size * cfg.hidden_size
            + cfg.max_position_embeddings * cfg.hidden_size
            + cfg.type_vocab_size * cfg.hidden_size;
        assert!(
            embeddings.parameter_count() >= min_expected,
            "parameter_count must cover at least word+position+token_type tables"
        );
    }

    // --- BertAttention ---

    #[test]
    fn test_bert_attention_new_cpu() {
        let cfg = small_config();
        let attn = BertAttention::new(&cfg).expect("BertAttention::new must succeed");
        assert_eq!(attn.device(), Device::CPU);
    }

    #[test]
    fn test_bert_attention_output_shape() {
        let cfg = small_config();
        let attn = BertAttention::new(&cfg).expect("BertAttention::new must succeed");
        let batch = 1usize;
        let seq_len = 4usize;
        let hidden_size = cfg.hidden_size;
        let mut state: u64 = 12345;
        // MultiHeadAttention forward_self_attention expects 3D: [batch, seq_len, hidden]
        let data: Vec<f32> =
            (0..batch * seq_len * hidden_size).map(|_| lcg_f32(&mut state)).collect();
        let input = Tensor::from_vec(data, &[batch, seq_len, hidden_size])
            .expect("Tensor creation must succeed");
        let output = attn.forward((input, None)).expect("BertAttention forward must succeed");
        let shape = output.shape();
        // Output shape must have hidden_size as last dim
        assert_eq!(*shape.last().expect("shape must not be empty"), hidden_size);
    }

    #[test]
    fn test_bert_attention_parameter_count_positive() {
        let cfg = small_config();
        let attn = BertAttention::new(&cfg).expect("BertAttention::new must succeed");
        assert!(attn.parameter_count() > 0);
    }

    // --- BertLayer ---

    #[test]
    fn test_bert_layer_new_cpu() {
        let cfg = small_config();
        let layer = BertLayer::new(&cfg).expect("BertLayer::new must succeed");
        assert_eq!(layer.device(), Device::CPU);
    }

    #[test]
    fn test_bert_layer_output_shape() {
        let cfg = small_config();
        let layer = BertLayer::new(&cfg).expect("BertLayer::new must succeed");
        let batch = 1usize;
        let seq_len = 5usize;
        let hidden_size = cfg.hidden_size;
        let mut state: u64 = 99999;
        // BertLayer attention expects 3D: [batch, seq_len, hidden]
        let data: Vec<f32> =
            (0..batch * seq_len * hidden_size).map(|_| lcg_f32(&mut state)).collect();
        let input = Tensor::from_vec(data, &[batch, seq_len, hidden_size])
            .expect("Tensor creation must succeed");
        let output = layer.forward((input, None)).expect("BertLayer forward must succeed");
        let shape = output.shape();
        assert_eq!(
            *shape.last().expect("shape must not be empty"),
            hidden_size,
            "BertLayer must preserve hidden_size in last dim"
        );
        assert!(
            shape.contains(&seq_len),
            "BertLayer must preserve seq_len in output shape"
        );
    }

    #[test]
    fn test_bert_layer_parameter_count_positive() {
        let cfg = small_config();
        let layer = BertLayer::new(&cfg).expect("BertLayer::new must succeed");
        assert!(layer.parameter_count() > 0);
    }

    // --- BertEncoder ---

    #[test]
    fn test_bert_encoder_new() {
        let cfg = small_config();
        let encoder = BertEncoder::new(&cfg).expect("BertEncoder::new must succeed");
        assert_eq!(encoder.device(), Device::CPU);
    }

    #[test]
    fn test_bert_encoder_output_shape() {
        let cfg = small_config();
        let encoder = BertEncoder::new(&cfg).expect("BertEncoder::new must succeed");
        let batch = 1usize;
        let seq_len = 4usize;
        let hidden_size = cfg.hidden_size;
        let mut state: u64 = 777;
        // Encoder expects 3D input: [batch, seq_len, hidden_size]
        let data: Vec<f32> =
            (0..batch * seq_len * hidden_size).map(|_| lcg_f32(&mut state)).collect();
        let input = Tensor::from_vec(data, &[batch, seq_len, hidden_size])
            .expect("Tensor creation must succeed");
        let output = encoder.forward(input, None).expect("BertEncoder forward must succeed");
        let shape = output.shape();
        assert_eq!(
            *shape.last().expect("shape must not be empty"),
            hidden_size,
            "Encoder output must have hidden_size in last dim"
        );
        assert!(
            shape.contains(&seq_len),
            "Encoder output must preserve seq_len"
        );
    }

    #[test]
    fn test_bert_encoder_parameter_count_scales_with_layers() {
        let cfg2 = small_config();
        let cfg4 = BertConfig {
            num_hidden_layers: 4,
            ..small_config()
        };
        let enc2 = BertEncoder::new(&cfg2).expect("BertEncoder 2 layers must succeed");
        let enc4 = BertEncoder::new(&cfg4).expect("BertEncoder 4 layers must succeed");
        assert!(
            enc4.parameter_count() > enc2.parameter_count(),
            "More layers must lead to more parameters"
        );
    }

    // --- BertPooler ---

    #[test]
    fn test_bert_pooler_new() {
        let cfg = small_config();
        let pooler = BertPooler::new(&cfg).expect("BertPooler::new must succeed");
        assert_eq!(pooler.device(), Device::CPU);
    }

    #[test]
    fn test_bert_pooler_output_shape() {
        let cfg = small_config();
        let pooler = BertPooler::new(&cfg).expect("BertPooler::new must succeed");
        // BertPooler expects exactly 2D input: [seq_len, hidden_size]
        // It extracts the first token (CLS) and applies dense+tanh
        let seq_len = 6usize;
        let hidden_size = cfg.hidden_size;
        let mut state: u64 = 54321;
        let data: Vec<f32> = (0..seq_len * hidden_size).map(|_| lcg_f32(&mut state)).collect();
        let input =
            Tensor::from_vec(data, &[seq_len, hidden_size]).expect("Tensor creation must succeed");
        let output = pooler.forward(input).expect("BertPooler forward must succeed");
        let shape = output.shape();
        // Pooler returns [1, hidden_size] (the CLS token linear transform)
        assert!(
            shape.contains(&hidden_size),
            "BertPooler output must contain hidden_size dimension, got {:?}",
            shape
        );
    }

    #[test]
    fn test_bert_pooler_output_tanh_bounded() {
        let cfg = small_config();
        let pooler = BertPooler::new(&cfg).expect("BertPooler::new must succeed");
        // BertPooler expects 2D input [seq_len, hidden_size]
        let seq_len = 3usize;
        let hidden_size = cfg.hidden_size;
        let mut state: u64 = 11111;
        let data: Vec<f32> = (0..seq_len * hidden_size).map(|_| lcg_f32(&mut state)).collect();
        let input =
            Tensor::from_vec(data, &[seq_len, hidden_size]).expect("Tensor creation must succeed");
        let output = pooler.forward(input).expect("BertPooler forward must succeed with 2D input");
        // tanh output must be in (-1, 1)
        if let trustformers_core::tensor::Tensor::F32(arr) = &output {
            for &v in arr.iter() {
                assert!(
                    (-1.0..=1.0).contains(&v),
                    "BertPooler tanh output must be in [-1, 1], got {}",
                    v
                );
            }
        }
    }

    #[test]
    fn test_bert_pooler_parameter_count_positive() {
        let cfg = small_config();
        let pooler = BertPooler::new(&cfg).expect("BertPooler::new must succeed");
        assert!(pooler.parameter_count() > 0);
    }

    // --- Attention head size property ---

    #[test]
    fn test_attention_head_size_equals_hidden_div_heads() {
        let cfg = small_config();
        // hidden=64, heads=8 -> head_size=8
        let head_size = cfg.hidden_size / cfg.num_attention_heads;
        assert_eq!(
            head_size, 8,
            "head_size must be hidden_size / num_attention_heads"
        );
    }
}
