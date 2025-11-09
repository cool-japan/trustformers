use serde::{Deserialize, Serialize};
use trustformers_core::{errors::invalid_config, traits::Config};

/// Configuration for hierarchical transformer models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalConfig {
    /// Base hidden size of the model
    pub hidden_size: usize,

    /// Number of hierarchical levels
    pub num_levels: usize,

    /// Number of attention heads per level
    pub num_heads: usize,

    /// Reduction factor between levels
    pub reduction_factor: usize,

    /// Number of layers per level
    pub num_layers_per_level: usize,

    /// Intermediate size for feed-forward networks
    pub intermediate_size: usize,

    /// Dropout rate
    pub dropout: f32,

    /// Attention dropout rate
    pub attention_dropout: f32,

    /// Layer norm epsilon
    pub layer_norm_eps: f32,

    /// Type of hierarchical architecture
    pub hierarchical_type: HierarchicalType,

    /// Method for reducing sequence length between levels
    pub reduction_method: ReductionMethod,

    /// Method for aggregating multi-level features
    pub aggregation_method: AggregationMethod,

    /// Maximum sequence length per level
    pub max_seq_lengths: Vec<usize>,

    /// Whether to use residual connections across levels
    pub cross_level_residual: bool,

    /// Whether to use position embeddings
    pub use_position_embeddings: bool,

    /// Configuration for tree-structured attention
    pub tree_config: Option<TreeConfig>,

    /// Configuration for pyramid architecture
    pub pyramid_config: Option<PyramidConfig>,

    /// Configuration for nested transformers
    pub nested_config: Option<NestedConfig>,
}

/// Types of hierarchical architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HierarchicalType {
    /// Standard hierarchical attention
    Hierarchical,
    /// Pyramid transformer
    Pyramid,
    /// Tree-structured transformer
    Tree,
    /// Nested transformer
    Nested,
    /// Hybrid architecture
    Hybrid,
}

/// Methods for reducing sequence length between levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReductionMethod {
    /// Average pooling
    AveragePooling,
    /// Max pooling
    MaxPooling,
    /// Learnable pooling
    LearnablePooling,
    /// Strided convolution
    StridedConvolution,
    /// Attention-based pooling
    AttentionPooling,
    /// Token merging
    TokenMerging,
}

/// Methods for aggregating features across levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// Sum aggregation
    Sum,
    /// Concatenation
    Concatenation,
    /// Weighted sum
    WeightedSum,
    /// Attention-based aggregation
    AttentionAggregation,
    /// Gated aggregation
    GatedAggregation,
}

/// Configuration for tree-structured attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeConfig {
    /// Branching factor of the tree
    pub branching_factor: usize,

    /// Maximum tree depth
    pub max_depth: usize,

    /// Whether to use learnable tree structure
    pub learnable_structure: bool,

    /// Method for constructing the tree
    pub tree_construction: TreeConstruction,
}

/// Methods for constructing tree structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TreeConstruction {
    /// Binary tree
    Binary,
    /// Balanced k-ary tree
    Balanced,
    /// Learned tree structure
    Learned,
    /// Syntax-guided tree
    SyntaxGuided,
}

/// Configuration for pyramid transformers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyramidConfig {
    /// Scaling factors for each level
    pub scaling_factors: Vec<f32>,

    /// Whether to use skip connections
    pub skip_connections: bool,

    /// Method for upsampling
    pub upsampling_method: UpsamplingMethod,

    /// Whether to use feature pyramid networks
    pub use_fpn: bool,
}

/// Methods for upsampling in pyramid architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UpsamplingMethod {
    /// Linear interpolation
    Linear,
    /// Transposed convolution
    TransposedConvolution,
    /// Learned upsampling
    Learned,
    /// Pixel shuffle
    PixelShuffle,
}

/// Configuration for nested transformers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NestedConfig {
    /// Number of nested levels
    pub num_nested_levels: usize,

    /// Share parameters across nested levels
    pub share_parameters: bool,

    /// Method for information flow between levels
    pub information_flow: InformationFlow,

    /// Whether to use progressive training
    pub progressive_training: bool,
}

/// Methods for information flow in nested architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InformationFlow {
    /// Bottom-up only
    BottomUp,
    /// Top-down only
    TopDown,
    /// Bidirectional
    Bidirectional,
    /// Skip connections
    SkipConnections,
}

impl Default for HierarchicalConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            num_levels: 4,
            num_heads: 12,
            reduction_factor: 2,
            num_layers_per_level: 3,
            intermediate_size: 3072,
            dropout: 0.1,
            attention_dropout: 0.1,
            layer_norm_eps: 1e-5,
            hierarchical_type: HierarchicalType::Hierarchical,
            reduction_method: ReductionMethod::AveragePooling,
            aggregation_method: AggregationMethod::WeightedSum,
            max_seq_lengths: vec![512, 256, 128, 64],
            cross_level_residual: true,
            use_position_embeddings: true,
            tree_config: None,
            pyramid_config: None,
            nested_config: None,
        }
    }
}

impl Default for TreeConfig {
    fn default() -> Self {
        Self {
            branching_factor: 2,
            max_depth: 8,
            learnable_structure: false,
            tree_construction: TreeConstruction::Binary,
        }
    }
}

impl Default for PyramidConfig {
    fn default() -> Self {
        Self {
            scaling_factors: vec![1.0, 0.5, 0.25, 0.125],
            skip_connections: true,
            upsampling_method: UpsamplingMethod::Linear,
            use_fpn: false,
        }
    }
}

impl Default for NestedConfig {
    fn default() -> Self {
        Self {
            num_nested_levels: 3,
            share_parameters: false,
            information_flow: InformationFlow::Bidirectional,
            progressive_training: false,
        }
    }
}

impl HierarchicalConfig {
    /// Create hierarchical attention configuration
    pub fn hierarchical(hidden_size: usize, num_levels: usize) -> Self {
        Self {
            hidden_size,
            num_levels,
            hierarchical_type: HierarchicalType::Hierarchical,
            max_seq_lengths: (0..num_levels).map(|i| 512 / (2_usize.pow(i as u32))).collect(),
            ..Default::default()
        }
    }

    /// Create pyramid transformer configuration
    pub fn pyramid(hidden_size: usize, num_levels: usize) -> Self {
        Self {
            hidden_size,
            num_levels,
            hierarchical_type: HierarchicalType::Pyramid,
            pyramid_config: Some(PyramidConfig::default()),
            max_seq_lengths: (0..num_levels).map(|i| 512 / (2_usize.pow(i as u32))).collect(),
            ..Default::default()
        }
    }

    /// Create tree transformer configuration
    pub fn tree(hidden_size: usize, branching_factor: usize, max_depth: usize) -> Self {
        Self {
            hidden_size,
            num_levels: max_depth,
            hierarchical_type: HierarchicalType::Tree,
            tree_config: Some(TreeConfig {
                branching_factor,
                max_depth,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// Create nested transformer configuration
    pub fn nested(hidden_size: usize, num_nested_levels: usize) -> Self {
        Self {
            hidden_size,
            num_levels: num_nested_levels,
            hierarchical_type: HierarchicalType::Nested,
            nested_config: Some(NestedConfig {
                num_nested_levels,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    /// Get the hidden size for a specific level
    pub fn get_hidden_size(&self, level: usize) -> usize {
        match self.hierarchical_type {
            HierarchicalType::Pyramid => {
                if let Some(pyramid_config) = &self.pyramid_config {
                    if level < pyramid_config.scaling_factors.len() {
                        (self.hidden_size as f32 * pyramid_config.scaling_factors[level]) as usize
                    } else {
                        self.hidden_size
                    }
                } else {
                    self.hidden_size
                }
            },
            _ => self.hidden_size,
        }
    }

    /// Get the sequence length for a specific level
    pub fn get_seq_length(&self, level: usize) -> usize {
        if level < self.max_seq_lengths.len() {
            self.max_seq_lengths[level]
        } else {
            512 / (2_usize.pow(level as u32))
        }
    }

    /// Get the reduction factor for a specific level
    pub fn get_reduction_factor(&self, level: usize) -> usize {
        self.reduction_factor.pow(level as u32)
    }

    /// Validate the configuration
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.num_levels == 0 {
            return Err("num_levels must be greater than 0".to_string());
        }

        if self.reduction_factor == 0 {
            return Err("reduction_factor must be greater than 0".to_string());
        }

        if self.num_heads == 0 {
            return Err("num_heads must be greater than 0".to_string());
        }

        if self.hidden_size % self.num_heads != 0 {
            return Err("hidden_size must be divisible by num_heads".to_string());
        }

        if self.dropout < 0.0 || self.dropout > 1.0 {
            return Err("dropout must be between 0.0 and 1.0".to_string());
        }

        if self.attention_dropout < 0.0 || self.attention_dropout > 1.0 {
            return Err("attention_dropout must be between 0.0 and 1.0".to_string());
        }

        if !self.max_seq_lengths.is_empty() && self.max_seq_lengths.len() != self.num_levels {
            return Err("max_seq_lengths length must match num_levels".to_string());
        }

        // Validate tree config if present
        if let Some(tree_config) = &self.tree_config {
            if tree_config.branching_factor == 0 {
                return Err("branching_factor must be greater than 0".to_string());
            }
            if tree_config.max_depth == 0 {
                return Err("max_depth must be greater than 0".to_string());
            }
        }

        // Validate pyramid config if present
        if let Some(pyramid_config) = &self.pyramid_config {
            if pyramid_config.scaling_factors.is_empty() {
                return Err("scaling_factors cannot be empty".to_string());
            }
            for &factor in &pyramid_config.scaling_factors {
                if factor <= 0.0 {
                    return Err("scaling_factors must be positive".to_string());
                }
            }
        }

        // Validate nested config if present
        if let Some(nested_config) = &self.nested_config {
            if nested_config.num_nested_levels == 0 {
                return Err("num_nested_levels must be greater than 0".to_string());
            }
        }

        Ok(())
    }

    /// Get the total number of parameters (rough estimate)
    pub fn estimate_parameters(&self) -> usize {
        let mut total = 0;

        for level in 0..self.num_levels {
            let hidden_size = self.get_hidden_size(level);
            let _seq_len = self.get_seq_length(level);

            // Attention parameters
            total += hidden_size * hidden_size * 4; // Q, K, V, O projections

            // FFN parameters
            total += hidden_size * self.intermediate_size * 2; // Up and down projections

            // Layer norm parameters
            total += hidden_size * 2; // Weight and bias

            // Multiply by number of layers at this level
            total *= self.num_layers_per_level;
        }

        total
    }
}

impl Config for HierarchicalConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        // Validate hierarchical configuration parameters
        if self.num_levels == 0 {
            return Err(invalid_config(
                "config_field",
                "num_levels must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "hierarchical"
    }
}
