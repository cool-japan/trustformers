use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// Recursive Transformer configuration
/// Reference: "Recursive Transformers for Long Sequences" (Universal Transformers, PaLM)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecursiveConfig {
    // Base transformer configuration
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub initializer_range: f32,
    pub layer_norm_eps: f32,
    pub use_cache: bool,
    pub pad_token_id: Option<u32>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,

    // Recursive configuration
    pub num_recursive_layers: usize,
    pub recursion_depth: usize,
    pub chunk_size: usize,
    pub overlap_size: usize,
    pub use_adaptive_depth: bool,
    pub min_depth: usize,
    pub max_depth: usize,
    pub depth_threshold: f32,

    // Memory configuration
    pub memory_size: usize,
    pub use_memory_compression: bool,
    pub compression_ratio: f32,
    pub memory_update_strategy: String,

    // Hierarchical processing
    pub use_hierarchical_attention: bool,
    pub hierarchy_levels: usize,
    pub level_compression_ratios: Vec<f32>,
    pub cross_level_attention: bool,

    // Universal Transformer features
    pub use_universal_transformer: bool,
    pub max_steps: usize,
    pub adaptive_computation_time: bool,
    pub act_threshold: f32,
    pub act_loss_weight: f32,

    // Optimization
    pub use_gradient_checkpointing: bool,
    pub use_flash_attention: bool,
    pub dropout: f32,
    pub attention_dropout: f32,

    // Model type
    pub model_type: String,
}

impl Default for RecursiveConfig {
    fn default() -> Self {
        Self {
            // Base transformer defaults
            vocab_size: 32000,
            hidden_size: 768,
            intermediate_size: 3072,
            num_attention_heads: 12,
            num_key_value_heads: None,
            hidden_act: "gelu".to_string(),
            max_position_embeddings: 16384, // Long sequences
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            use_cache: true,
            pad_token_id: Some(0),
            bos_token_id: 1,
            eos_token_id: 2,

            // Recursive defaults
            num_recursive_layers: 6,
            recursion_depth: 3,
            chunk_size: 512,
            overlap_size: 64,
            use_adaptive_depth: true,
            min_depth: 1,
            max_depth: 5,
            depth_threshold: 0.5,

            // Memory defaults
            memory_size: 1024,
            use_memory_compression: true,
            compression_ratio: 0.5,
            memory_update_strategy: "gated".to_string(),

            // Hierarchical defaults
            use_hierarchical_attention: true,
            hierarchy_levels: 3,
            level_compression_ratios: vec![1.0, 0.5, 0.25],
            cross_level_attention: true,

            // Universal Transformer defaults
            use_universal_transformer: false,
            max_steps: 10,
            adaptive_computation_time: false,
            act_threshold: 0.01,
            act_loss_weight: 0.01,

            // Optimization defaults
            use_gradient_checkpointing: true,
            use_flash_attention: true,
            dropout: 0.1,
            attention_dropout: 0.1,

            model_type: "recursive".to_string(),
        }
    }
}

impl Config for RecursiveConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        // Validate base transformer configuration
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "hidden_size must be divisible by num_attention_heads".to_string(),
                ),
            );
        }

        if let Some(num_kv_heads) = self.num_key_value_heads {
            if self.num_attention_heads % num_kv_heads != 0 {
                return Err(
                    trustformers_core::errors::TrustformersError::invalid_config(
                        "num_attention_heads must be divisible by num_key_value_heads".to_string(),
                    ),
                );
            }
        }

        // Validate recursive configuration
        if self.recursion_depth == 0 {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "recursion_depth must be greater than 0".to_string(),
                ),
            );
        }

        if self.chunk_size == 0 {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "chunk_size must be greater than 0".to_string(),
                ),
            );
        }

        if self.overlap_size >= self.chunk_size {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "overlap_size must be less than chunk_size".to_string(),
                ),
            );
        }

        if self.use_adaptive_depth && self.min_depth >= self.max_depth {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "min_depth must be less than max_depth for adaptive depth".to_string(),
                ),
            );
        }

        // Validate hierarchical configuration
        if self.use_hierarchical_attention {
            if self.hierarchy_levels == 0 {
                return Err(
                    trustformers_core::errors::TrustformersError::invalid_config(
                        "hierarchy_levels must be greater than 0".to_string(),
                    ),
                );
            }

            if self.level_compression_ratios.len() != self.hierarchy_levels {
                return Err(
                    trustformers_core::errors::TrustformersError::invalid_config(
                        "level_compression_ratios length must equal hierarchy_levels".to_string(),
                    ),
                );
            }

            for ratio in &self.level_compression_ratios {
                if *ratio <= 0.0 || *ratio > 1.0 {
                    return Err(
                        trustformers_core::errors::TrustformersError::invalid_config(
                            "compression ratios must be between 0.0 and 1.0".to_string(),
                        ),
                    );
                }
            }
        }

        // Validate Universal Transformer configuration
        if self.use_universal_transformer && self.max_steps == 0 {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "max_steps must be greater than 0 for Universal Transformer".to_string(),
                ),
            );
        }

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "RecursiveTransformer"
    }
}

impl RecursiveConfig {
    /// Create a recursive transformer for long documents
    pub fn long_document() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 4096,
            num_attention_heads: 16,
            max_position_embeddings: 32768,
            recursion_depth: 4,
            chunk_size: 1024,
            overlap_size: 128,
            memory_size: 2048,
            hierarchy_levels: 4,
            level_compression_ratios: vec![1.0, 0.75, 0.5, 0.25],
            model_type: "recursive-long-document".to_string(),
            ..Self::default()
        }
    }

    /// Create a recursive transformer with Universal Transformer features
    pub fn universal() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 3072,
            num_attention_heads: 12,
            num_recursive_layers: 1, // Single layer, repeated
            use_universal_transformer: true,
            max_steps: 8,
            adaptive_computation_time: true,
            act_threshold: 0.01,
            recursion_depth: 1, // Depth through time, not hierarchy
            model_type: "recursive-universal".to_string(),
            ..Self::default()
        }
    }

    /// Create a memory-efficient recursive transformer
    pub fn memory_efficient() -> Self {
        Self {
            hidden_size: 512,
            intermediate_size: 2048,
            num_attention_heads: 8,
            use_gradient_checkpointing: true,
            use_memory_compression: true,
            compression_ratio: 0.25,
            chunk_size: 256,
            overlap_size: 32,
            memory_size: 512,
            model_type: "recursive-memory-efficient".to_string(),
            ..Self::default()
        }
    }

    /// Create a hierarchical recursive transformer
    pub fn hierarchical() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 3072,
            num_attention_heads: 12,
            use_hierarchical_attention: true,
            hierarchy_levels: 3,
            level_compression_ratios: vec![1.0, 0.5, 0.25],
            cross_level_attention: true,
            recursion_depth: 3,
            chunk_size: 512,
            model_type: "recursive-hierarchical".to_string(),
            ..Self::default()
        }
    }

    /// Create configuration for code understanding
    pub fn code_understanding() -> Self {
        Self {
            vocab_size: 50000, // Larger vocabulary for code
            hidden_size: 1024,
            intermediate_size: 4096,
            num_attention_heads: 16,
            max_position_embeddings: 8192,
            recursion_depth: 3,
            chunk_size: 512,
            overlap_size: 64,
            use_hierarchical_attention: true,
            hierarchy_levels: 3, // File -> Function -> Statement level
            memory_size: 1024,
            model_type: "recursive-code".to_string(),
            ..Self::default()
        }
    }

    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get the number of key-value heads
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Get effective sequence length per chunk
    pub fn effective_chunk_size(&self) -> usize {
        self.chunk_size - self.overlap_size
    }

    /// Get total memory capacity across all levels
    pub fn total_memory_capacity(&self) -> usize {
        if self.use_hierarchical_attention {
            self.level_compression_ratios
                .iter()
                .map(|ratio| (self.memory_size as f32 * ratio) as usize)
                .sum()
        } else {
            self.memory_size
        }
    }

    /// Create configuration from pretrained model name
    pub fn from_pretrained_name(name: &str) -> Option<Self> {
        match name {
            "recursive-long-document" => Some(Self::long_document()),
            "recursive-universal" => Some(Self::universal()),
            "recursive-memory-efficient" => Some(Self::memory_efficient()),
            "recursive-hierarchical" => Some(Self::hierarchical()),
            "recursive-code" => Some(Self::code_understanding()),
            _ => None,
        }
    }

    /// Configure memory settings
    pub fn with_memory(&mut self, size: usize, compression: bool, ratio: f32) -> &mut Self {
        self.memory_size = size;
        self.use_memory_compression = compression;
        self.compression_ratio = ratio;
        self
    }

    /// Configure chunking settings
    pub fn with_chunks(&mut self, chunk_size: usize, overlap_size: usize) -> &mut Self {
        self.chunk_size = chunk_size;
        self.overlap_size = overlap_size;
        self
    }

    /// Configure recursion depth
    pub fn with_depth(&mut self, depth: usize, adaptive: bool) -> &mut Self {
        self.recursion_depth = depth;
        self.use_adaptive_depth = adaptive;
        if adaptive {
            self.max_depth = depth * 2;
        }
        self
    }

    /// Configure hierarchical attention
    pub fn with_hierarchy(&mut self, levels: usize, ratios: Vec<f32>) -> &mut Self {
        self.use_hierarchical_attention = true;
        self.hierarchy_levels = levels;
        self.level_compression_ratios = ratios;
        self
    }

    /// Configure Universal Transformer settings
    pub fn with_universal(&mut self, max_steps: usize, act: bool) -> &mut Self {
        self.use_universal_transformer = true;
        self.max_steps = max_steps;
        self.adaptive_computation_time = act;
        self
    }
}

/// Recursive processing strategy
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum RecursionStrategy {
    /// Divide and conquer - split input into chunks
    #[default]
    DivideAndConquer,
    /// Hierarchical - process at multiple resolutions
    Hierarchical,
    /// Universal - repeat same layer multiple times
    Universal,
    /// Memory-augmented - use external memory
    MemoryAugmented,
    /// Adaptive - dynamically determine recursion depth
    Adaptive,
}

/// Memory update strategy for recursive processing
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum MemoryUpdateStrategy {
    /// Simple concatenation
    Concatenate,
    /// Gated update mechanism
    #[default]
    Gated,
    /// Attention-based update
    Attention,
    /// LSTM-style update
    LSTM,
    /// Differentiable Neural Computer style
    DNC,
}
