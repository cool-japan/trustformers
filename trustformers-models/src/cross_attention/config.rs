use serde::{Deserialize, Serialize};

/// Configuration for cross-attention mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossAttentionConfig {
    /// Hidden size of the model
    pub hidden_size: usize,

    /// Number of attention heads
    pub num_heads: usize,

    /// Dimension of each attention head
    pub head_dim: Option<usize>,

    /// Dropout probability for attention weights
    pub attention_dropout: f32,

    /// Whether to use bias in linear projections
    pub bias: bool,

    /// Scaling factor for attention scores
    pub scale: Option<f32>,

    /// Maximum sequence length for positional encoding
    pub max_seq_len: usize,

    /// Cross-attention variant type
    pub attention_type: CrossAttentionType,

    /// Configuration for sparse attention
    pub sparse_config: Option<SparseAttentionConfig>,

    /// Configuration for hierarchical attention
    pub hierarchical_config: Option<HierarchicalAttentionConfig>,

    /// Configuration for adaptive attention
    pub adaptive_config: Option<AdaptiveAttentionConfig>,

    /// Configuration for gated attention
    pub gated_config: Option<GatedAttentionConfig>,
}

/// Types of cross-attention mechanisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrossAttentionType {
    /// Standard cross-attention
    Standard,
    /// Multi-head cross-attention
    MultiHead,
    /// Sparse cross-attention
    Sparse,
    /// Hierarchical cross-attention
    Hierarchical,
    /// Adaptive cross-attention
    Adaptive,
    /// Gated cross-attention
    Gated,
}

/// Configuration for sparse cross-attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseAttentionConfig {
    /// Sparsity pattern type
    pub pattern: SparsePattern,

    /// Sparsity ratio (0.0 to 1.0)
    pub sparsity_ratio: f32,

    /// Block size for block-sparse attention
    pub block_size: Option<usize>,

    /// Number of random connections for random sparse attention
    pub random_connections: Option<usize>,
}

/// Sparse attention patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SparsePattern {
    /// Random sparsity
    Random,
    /// Block-based sparsity
    Block,
    /// Strided sparsity
    Strided,
    /// Local window sparsity
    Local,
    /// Top-k sparsity
    TopK,
}

/// Configuration for hierarchical cross-attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalAttentionConfig {
    /// Number of hierarchy levels
    pub num_levels: usize,

    /// Pooling factor for each level
    pub pooling_factor: usize,

    /// Whether to use learnable pooling
    pub learnable_pooling: bool,

    /// Aggregation method for multi-level features
    pub aggregation_method: AggregationMethod,
}

/// Methods for aggregating hierarchical features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// Weighted sum
    WeightedSum,
    /// Concatenation
    Concatenation,
    /// Max pooling
    MaxPooling,
    /// Average pooling
    AvgPooling,
}

/// Configuration for adaptive cross-attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveAttentionConfig {
    /// Number of attention patterns to learn
    pub num_patterns: usize,

    /// Dimension of pattern embeddings
    pub pattern_dim: usize,

    /// Temperature for pattern selection
    pub temperature: f32,

    /// Whether to use hard or soft pattern selection
    pub hard_selection: bool,
}

/// Configuration for gated cross-attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatedAttentionConfig {
    /// Gate activation function
    pub gate_activation: GateActivation,

    /// Whether to use separate gates for query, key, and value
    pub separate_gates: bool,

    /// Dimension of gate hidden layer
    pub gate_hidden_dim: Option<usize>,

    /// Whether to use bias in gate computations
    pub gate_bias: bool,
}

/// Activation functions for gates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GateActivation {
    /// Sigmoid activation
    Sigmoid,
    /// Tanh activation
    Tanh,
    /// ReLU activation
    ReLU,
    /// GELU activation
    GELU,
    /// Swish activation
    Swish,
}

impl Default for CrossAttentionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 512,
            num_heads: 8,
            head_dim: None,
            attention_dropout: 0.1,
            bias: true,
            scale: None,
            max_seq_len: 1024,
            attention_type: CrossAttentionType::Standard,
            sparse_config: None,
            hierarchical_config: None,
            adaptive_config: None,
            gated_config: None,
        }
    }
}

impl Default for SparseAttentionConfig {
    fn default() -> Self {
        Self {
            pattern: SparsePattern::Random,
            sparsity_ratio: 0.1,
            block_size: Some(64),
            random_connections: Some(32),
        }
    }
}

impl Default for HierarchicalAttentionConfig {
    fn default() -> Self {
        Self {
            num_levels: 3,
            pooling_factor: 2,
            learnable_pooling: true,
            aggregation_method: AggregationMethod::WeightedSum,
        }
    }
}

impl Default for AdaptiveAttentionConfig {
    fn default() -> Self {
        Self {
            num_patterns: 4,
            pattern_dim: 64,
            temperature: 1.0,
            hard_selection: false,
        }
    }
}

impl Default for GatedAttentionConfig {
    fn default() -> Self {
        Self {
            gate_activation: GateActivation::Sigmoid,
            separate_gates: false,
            gate_hidden_dim: None,
            gate_bias: true,
        }
    }
}

impl CrossAttentionConfig {
    /// Get the actual head dimension
    pub fn get_head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_heads)
    }

    /// Get the attention scale factor
    pub fn get_scale(&self) -> f32 {
        self.scale.unwrap_or(1.0 / (self.get_head_dim() as f32).sqrt())
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.hidden_size % self.num_heads != 0 {
            return Err("hidden_size must be divisible by num_heads".to_string());
        }

        if self.attention_dropout < 0.0 || self.attention_dropout > 1.0 {
            return Err("attention_dropout must be between 0.0 and 1.0".to_string());
        }

        if let Some(sparse_config) = &self.sparse_config {
            if sparse_config.sparsity_ratio < 0.0 || sparse_config.sparsity_ratio > 1.0 {
                return Err("sparsity_ratio must be between 0.0 and 1.0".to_string());
            }
        }

        if let Some(hierarchical_config) = &self.hierarchical_config {
            if hierarchical_config.num_levels == 0 {
                return Err("num_levels must be greater than 0".to_string());
            }
            if hierarchical_config.pooling_factor == 0 {
                return Err("pooling_factor must be greater than 0".to_string());
            }
        }

        if let Some(adaptive_config) = &self.adaptive_config {
            if adaptive_config.num_patterns == 0 {
                return Err("num_patterns must be greater than 0".to_string());
            }
            if adaptive_config.temperature <= 0.0 {
                return Err("temperature must be greater than 0".to_string());
            }
        }

        Ok(())
    }

    /// Create configuration for standard cross-attention
    pub fn standard(hidden_size: usize, num_heads: usize) -> Self {
        Self {
            hidden_size,
            num_heads,
            attention_type: CrossAttentionType::Standard,
            ..Default::default()
        }
    }

    /// Create configuration for sparse cross-attention
    pub fn sparse(hidden_size: usize, num_heads: usize, sparsity_ratio: f32) -> Self {
        Self {
            hidden_size,
            num_heads,
            attention_type: CrossAttentionType::Sparse,
            sparse_config: Some(SparseAttentionConfig {
                sparsity_ratio,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// Create configuration for hierarchical cross-attention
    pub fn hierarchical(hidden_size: usize, num_heads: usize, num_levels: usize) -> Self {
        Self {
            hidden_size,
            num_heads,
            attention_type: CrossAttentionType::Hierarchical,
            hierarchical_config: Some(HierarchicalAttentionConfig {
                num_levels,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// Create configuration for adaptive cross-attention
    pub fn adaptive(hidden_size: usize, num_heads: usize, num_patterns: usize) -> Self {
        Self {
            hidden_size,
            num_heads,
            attention_type: CrossAttentionType::Adaptive,
            adaptive_config: Some(AdaptiveAttentionConfig {
                num_patterns,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// Create configuration for gated cross-attention
    pub fn gated(hidden_size: usize, num_heads: usize) -> Self {
        Self {
            hidden_size,
            num_heads,
            attention_type: CrossAttentionType::Gated,
            gated_config: Some(GatedAttentionConfig::default()),
            ..Default::default()
        }
    }
}
