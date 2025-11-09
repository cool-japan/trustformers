use serde::{Deserialize, Serialize};
use trustformers_core::errors::invalid_config;
use trustformers_core::traits::Config;

/// RetNet model configuration
/// Reference: "Retentive Network: A Successor to Transformer for Large Language Models"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetNetConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub hidden_dropout_prob: f32,
    pub attention_dropout_prob: f32,
    pub max_position_embeddings: usize,
    pub initializer_range: f32,
    pub layer_norm_eps: f32,
    pub pad_token_id: u32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,

    // RetNet-specific parameters
    pub use_bias: bool,                // Use bias in linear layers
    pub use_glu: bool,                 // Use GLU activation in FFN
    pub use_norm_bias: bool,           // Use bias in normalization layers
    pub deepnorm: bool,                // Use DeepNorm for better scaling
    pub dropout_module: String,        // Dropout module type
    pub activation_dropout: f32,       // Dropout for activations
    pub attention_dropout: f32,        // Dropout for retention mechanism
    pub retention_heads: usize,        // Number of retention heads
    pub value_factor: f32,             // Value scaling factor
    pub gate_fn: String,               // Gate function type
    pub tensor_parallel_degree: usize, // Tensor parallelism degree
    pub sequence_parallel: bool,       // Enable sequence parallelism
    pub fuse_norm: bool,               // Fuse normalization operations
    pub no_output_layer: bool,         // Remove output layer
    pub layernorm_embedding: bool,     // Apply layernorm to embeddings
    pub chunking: bool,                // Enable chunked processing
    pub chunk_size: usize,             // Chunk size for processing
}

impl Default for RetNetConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 2048,
            num_hidden_layers: 24,
            num_heads: 16,
            intermediate_size: 8192,
            hidden_act: "swish".to_string(),
            hidden_dropout_prob: 0.0,
            attention_dropout_prob: 0.0,
            max_position_embeddings: 2048,
            initializer_range: 0.02,
            layer_norm_eps: 1e-6,
            pad_token_id: 0,
            bos_token_id: 1,
            eos_token_id: 2,

            // RetNet defaults
            use_bias: false,
            use_glu: true,
            use_norm_bias: false,
            deepnorm: true,
            dropout_module: "dropout".to_string(),
            activation_dropout: 0.0,
            attention_dropout: 0.0,
            retention_heads: 16,
            value_factor: 2.0,
            gate_fn: "swish".to_string(),
            tensor_parallel_degree: 1,
            sequence_parallel: false,
            fuse_norm: false,
            no_output_layer: false,
            layernorm_embedding: false,
            chunking: false,
            chunk_size: 512,
        }
    }
}

impl Config for RetNetConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.hidden_size % self.num_heads != 0 {
            return Err(invalid_config(
                "config_field",
                "hidden_size must be divisible by num_heads".to_string(),
            ));
        }

        if self.hidden_size % self.retention_heads != 0 {
            return Err(invalid_config(
                "config_field",
                "hidden_size must be divisible by retention_heads".to_string(),
            ));
        }

        if self.chunk_size > self.max_position_embeddings {
            return Err(invalid_config(
                "config_field",
                "chunk_size should not exceed max_position_embeddings".to_string(),
            ));
        }

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "RetNet"
    }
}

impl RetNetConfig {
    /// RetNet-Small configuration (1.3B parameters)
    pub fn retnet_small() -> Self {
        Self {
            hidden_size: 2048,
            num_hidden_layers: 24,
            num_heads: 16,
            intermediate_size: 8192,
            retention_heads: 16,
            max_position_embeddings: 2048,
            ..Self::default()
        }
    }

    /// RetNet-Medium configuration (2.7B parameters)
    pub fn retnet_medium() -> Self {
        Self {
            hidden_size: 2560,
            num_hidden_layers: 32,
            num_heads: 20,
            intermediate_size: 10240,
            retention_heads: 20,
            max_position_embeddings: 2048,
            ..Self::default()
        }
    }

    /// RetNet-Large configuration (6.7B parameters)
    pub fn retnet_large() -> Self {
        Self {
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_heads: 32,
            intermediate_size: 16384,
            retention_heads: 32,
            max_position_embeddings: 2048,
            ..Self::default()
        }
    }

    /// RetNet-XL configuration (13B parameters)
    pub fn retnet_xl() -> Self {
        Self {
            hidden_size: 5120,
            num_hidden_layers: 40,
            num_heads: 40,
            intermediate_size: 20480,
            retention_heads: 40,
            max_position_embeddings: 2048,
            deepnorm: true,
            ..Self::default()
        }
    }

    /// Long-context RetNet for extended sequences
    pub fn retnet_long() -> Self {
        Self {
            max_position_embeddings: 8192,
            chunking: true,
            chunk_size: 1024,
            sequence_parallel: true,
            ..Self::retnet_medium()
        }
    }

    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    /// Get the retention head dimension
    pub fn retention_head_dim(&self) -> usize {
        self.hidden_size / self.retention_heads
    }

    /// Get the effective head dimension for retention
    pub fn retention_dim(&self) -> usize {
        (self.hidden_size as f32 / self.value_factor) as usize
    }

    /// Check if using efficient chunked processing
    pub fn uses_chunking(&self) -> bool {
        self.chunking && self.chunk_size > 0
    }

    /// Get memory complexity advantage over attention
    pub fn memory_advantage(&self) -> f32 {
        let seq_len = self.max_position_embeddings as f32;
        let attention_memory = seq_len * seq_len;
        let retnet_memory = seq_len; // Linear complexity
        attention_memory / retnet_memory
    }

    /// Check if configuration supports very long sequences
    pub fn supports_long_sequences(&self) -> bool {
        self.max_position_embeddings >= 4096 || self.uses_chunking()
    }

    /// Get the deepnorm scaling factor
    pub fn deepnorm_alpha(&self) -> f32 {
        // DeepNorm scaling factor based on number of layers
        (2.0 * self.num_hidden_layers as f32).powf(0.25)
    }

    /// Get the deepnorm beta factor
    pub fn deepnorm_beta(&self) -> f32 {
        // DeepNorm initialization factor
        (8.0 * self.num_hidden_layers as f32).powf(-0.25)
    }
}
