# Model Implementation Best Practices

This guide outlines best practices for implementing high-quality, performant, and maintainable transformer models in TrustformeRS.

## Table of Contents

1. [Code Organization](#code-organization)
2. [Configuration Design](#configuration-design)
3. [Module Architecture](#module-architecture)
4. [Performance Optimization](#performance-optimization)
5. [Memory Management](#memory-management)
6. [Error Handling](#error-handling)
7. [Testing Strategies](#testing-strategies)
8. [Documentation Standards](#documentation-standards)
9. [Compatibility Guidelines](#compatibility-guidelines)
10. [Security Considerations](#security-considerations)

## Code Organization

### 1. File Structure Standards

Follow the established directory pattern for consistency:

```
src/model_name/
├── mod.rs              # Public interface and re-exports
├── config.rs           # Configuration structures
├── model.rs            # Main model implementation
├── layers.rs           # Layer implementations
├── attention.rs        # Attention mechanisms
├── heads.rs            # Task-specific heads
├── utils.rs            # Model-specific utilities
└── tests.rs            # Comprehensive tests
```

### 2. Module Organization

**Good Practice:**
```rust
// mod.rs - Clean public interface
pub mod config;
pub mod model;
pub mod layers;
pub mod attention;
pub mod heads;

// Re-export main types
pub use config::*;
pub use model::*;
pub use heads::*;

#[cfg(test)]
mod tests;
```

**Avoid:**
```rust
// Don't expose internal implementation details
pub mod internal_utils; // Should be private
pub use layers::*;      // Avoid wildcard exports of internals
```

### 3. Line Length Policy

Maintain files under 2000 lines as per project policy. Split large files:

```rust
// Instead of one large model.rs (3000+ lines)
// Split into:
model/
├── core.rs         # Core model structure
├── layers.rs       # Layer implementations  
├── forward.rs      # Forward pass logic
├── generation.rs   # Generation methods
└── mod.rs          # Module coordination
```

## Configuration Design

### 1. Configuration Structure

Design configurations for extensibility and validation:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    // Required core parameters
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    
    // Optional parameters with sensible defaults
    pub intermediate_size: Option<usize>,
    pub hidden_dropout_prob: f32,
    pub attention_probs_dropout_prob: f32,
    
    // Model-specific parameters
    #[serde(flatten)]
    pub model_specific: ModelSpecificConfig,
    
    // Metadata
    pub model_type: String,
    pub architecture_version: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50257,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: None, // Will default to 4 * hidden_size
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            model_specific: ModelSpecificConfig::default(),
            model_type: "transformer".to_string(),
            architecture_version: "1.0".to_string(),
        }
    }
}
```

### 2. Configuration Validation

Implement comprehensive validation:

```rust
impl Config for ModelConfig {
    fn validate(&self) -> Result<()> {
        // Basic constraints
        if self.hidden_size == 0 {
            return Err(ConfigError::InvalidParameter {
                param: "hidden_size".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        
        // Architectural constraints
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(ConfigError::InvalidParameter {
                param: "hidden_size".to_string(),
                reason: format!(
                    "must be divisible by num_attention_heads ({})",
                    self.num_attention_heads
                ),
            });
        }
        
        // Logical constraints
        let intermediate_size = self.intermediate_size
            .unwrap_or(4 * self.hidden_size);
        if intermediate_size < self.hidden_size {
            log::warn!(
                "intermediate_size ({}) is smaller than hidden_size ({})",
                intermediate_size, self.hidden_size
            );
        }
        
        // Model-specific validation
        self.model_specific.validate()?;
        
        Ok(())
    }
}
```

### 3. Configuration Presets

Provide convenient preset methods:

```rust
impl ModelConfig {
    // Standard model sizes
    pub fn small() -> Self {
        Self {
            hidden_size: 512,
            num_hidden_layers: 6,
            num_attention_heads: 8,
            ..Default::default()
        }
    }
    
    pub fn base() -> Self {
        Self::default()
    }
    
    pub fn large() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            ..Default::default()
        }
    }
    
    // Domain-specific presets
    pub fn for_code_generation() -> Self {
        Self {
            vocab_size: 50400, // Extended vocabulary
            max_position_embeddings: 8192, // Longer context
            ..Self::base()
        }
    }
    
    // Hardware-optimized presets
    pub fn for_mobile() -> Self {
        Self {
            hidden_size: 256,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            hidden_dropout_prob: 0.0, // Reduce computation
            attention_probs_dropout_prob: 0.0,
            ..Default::default()
        }
    }
}
```

## Module Architecture

### 1. Layer Design Principles

Design layers to be composable and reusable:

```rust
// Good: Modular design
pub struct TransformerLayer {
    attention: Box<dyn AttentionLayer>,
    feed_forward: Box<dyn FeedForwardLayer>,
    norm1: Box<dyn NormalizationLayer>,
    norm2: Box<dyn NormalizationLayer>,
    dropout: Dropout,
}

impl TransformerLayer {
    pub fn new(config: &LayerConfig) -> Result<Self> {
        Ok(Self {
            attention: create_attention_layer(config)?,
            feed_forward: create_ff_layer(config)?,
            norm1: create_norm_layer(config)?,
            norm2: create_norm_layer(config)?,
            dropout: Dropout::new(config.dropout),
        })
    }
}

// Factory functions for different layer types
fn create_attention_layer(config: &LayerConfig) -> Result<Box<dyn AttentionLayer>> {
    match config.attention_type {
        AttentionType::MultiHead => Ok(Box::new(MultiHeadAttention::new(config)?)),
        AttentionType::GroupedQuery => Ok(Box::new(GroupedQueryAttention::new(config)?)),
        AttentionType::SlidingWindow => Ok(Box::new(SlidingWindowAttention::new(config)?)),
    }
}
```

### 2. Trait Design

Use traits for common interfaces:

```rust
pub trait AttentionLayer: Module {
    fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
        past_key_value: Option<(&Tensor, &Tensor)>,
        use_cache: bool,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)>;
    
    fn num_heads(&self) -> usize;
    fn head_dim(&self) -> usize;
}

pub trait FeedForwardLayer: Module {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
    fn intermediate_size(&self) -> usize;
}

pub trait NormalizationLayer: Module {
    fn forward(&self, x: &Tensor) -> Result<Tensor>;
    fn normalized_shape(&self) -> &[usize];
}
```

### 3. Builder Pattern for Complex Models

Use builders for models with many configuration options:

```rust
pub struct ModelBuilder {
    config: ModelConfig,
    custom_layers: Vec<Box<dyn TransformerLayer>>,
    custom_heads: Vec<Box<dyn TaskHead>>,
}

impl ModelBuilder {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            custom_layers: Vec::new(),
            custom_heads: Vec::new(),
        }
    }
    
    pub fn with_custom_layer(mut self, layer: Box<dyn TransformerLayer>) -> Self {
        self.custom_layers.push(layer);
        self
    }
    
    pub fn with_task_head(mut self, head: Box<dyn TaskHead>) -> Self {
        self.custom_heads.push(head);
        self
    }
    
    pub fn build(self) -> Result<Model> {
        self.config.validate()?;
        Model::from_builder(self)
    }
}

// Usage
let model = ModelBuilder::new(config)
    .with_custom_layer(Box::new(CustomAttentionLayer::new()?))
    .with_task_head(Box::new(ClassificationHead::new(num_classes)?))
    .build()?;
```

## Performance Optimization

### 1. Memory-Efficient Operations

Optimize tensor operations for memory usage:

```rust
impl Model {
    pub fn forward(&self, input_ids: &Tensor) -> Result<ModelOutput> {
        // Use in-place operations where possible
        let mut hidden_states = self.embeddings.forward(input_ids)?;
        
        // Reuse buffers for attention computation
        let mut attention_buffer = AttentionBuffer::new(
            self.config.max_batch_size,
            self.config.max_sequence_length,
            self.config.hidden_size,
        )?;
        
        for layer in &self.layers {
            hidden_states = layer.forward_with_buffer(
                &hidden_states,
                &mut attention_buffer,
            )?;
        }
        
        Ok(ModelOutput {
            last_hidden_state: hidden_states,
            ..Default::default()
        })
    }
}

// Buffer management for attention
pub struct AttentionBuffer {
    query_buffer: Tensor,
    key_buffer: Tensor,
    value_buffer: Tensor,
    scores_buffer: Tensor,
}

impl AttentionBuffer {
    pub fn resize_if_needed(&mut self, batch_size: usize, seq_len: usize) -> Result<()> {
        if self.query_buffer.shape()[0] < batch_size ||
           self.query_buffer.shape()[1] < seq_len {
            self.query_buffer = Tensor::zeros(&[batch_size, seq_len, self.hidden_size])?;
            // ... resize other buffers
        }
        Ok(())
    }
}
```

### 2. Efficient Attention Implementation

Implement attention patterns efficiently:

```rust
impl MultiHeadAttention {
    pub fn forward_optimized(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = hidden_states.shape3()?;
        
        // Fused QKV projection
        let qkv = self.qkv_proj.forward(hidden_states)?;
        let qkv = qkv.view(&[batch_size, seq_len, 3, self.num_heads, self.head_dim])?;
        let qkv = qkv.transpose(1, 3)?; // [batch, head, seq, 3, dim]
        
        let q = qkv.select(3, 0)?; // [batch, head, seq, dim]
        let k = qkv.select(3, 1)?;
        let v = qkv.select(3, 2)?;
        
        // Use flash attention if available
        let attn_output = if self.use_flash_attention {
            self.flash_attention(&q, &k, &v, attention_mask)?
        } else {
            self.standard_attention(&q, &k, &v, attention_mask)?
        };
        
        // Reshape and project output
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .view(&[batch_size, seq_len, hidden_size])?;
        
        self.o_proj.forward(&attn_output)
    }
    
    fn flash_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Use optimized flash attention kernel
        flash_attn_func(q, k, v, mask, self.dropout_p, self.scale)
    }
}
```

### 3. Gradient Checkpointing

Implement gradient checkpointing for large models:

```rust
pub struct Model {
    config: ModelConfig,
    layers: ModuleList<TransformerLayer>,
    gradient_checkpointing: bool,
}

impl Model {
    pub fn enable_gradient_checkpointing(&mut self) {
        self.gradient_checkpointing = true;
    }
    
    pub fn forward(&self, input_ids: &Tensor) -> Result<ModelOutput> {
        let mut hidden_states = self.embeddings.forward(input_ids)?;
        
        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = if self.gradient_checkpointing && self.training {
                // Use checkpointing for memory efficiency
                checkpoint_layer(layer, &hidden_states)?
            } else {
                layer.forward(&hidden_states)?
            };
        }
        
        Ok(ModelOutput {
            last_hidden_state: hidden_states,
        })
    }
}

fn checkpoint_layer(
    layer: &TransformerLayer,
    input: &Tensor,
) -> Result<Tensor> {
    // Recompute forward pass during backward
    input.requires_grad_(false);
    let output = layer.forward(input)?;
    output.requires_grad_(true);
    Ok(output)
}
```

## Memory Management

### 1. Resource Cleanup

Implement proper resource management:

```rust
pub struct Model {
    // ... fields
    _cleanup: Vec<Box<dyn Drop>>,
}

impl Model {
    pub fn new(config: ModelConfig) -> Result<Self> {
        let mut cleanup = Vec::new();
        
        // Register cleanup for large allocations
        let large_buffer = allocate_large_buffer()?;
        cleanup.push(Box::new(BufferCleanup::new(large_buffer)));
        
        Ok(Self {
            // ... initialization
            _cleanup: cleanup,
        })
    }
}

struct BufferCleanup {
    buffer: Tensor,
}

impl BufferCleanup {
    fn new(buffer: Tensor) -> Self {
        Self { buffer }
    }
}

impl Drop for BufferCleanup {
    fn drop(&mut self) {
        // Explicit cleanup if needed
        log::debug!("Cleaning up large buffer");
    }
}
```

### 2. Memory Pool Management

Use memory pools for frequent allocations:

```rust
pub struct TensorPool {
    available: Vec<Tensor>,
    max_size: usize,
}

impl TensorPool {
    pub fn new(max_size: usize) -> Self {
        Self {
            available: Vec::with_capacity(max_size),
            max_size,
        }
    }
    
    pub fn get(&mut self, shape: &[usize]) -> Result<Tensor> {
        // Try to reuse existing tensor
        for (i, tensor) in self.available.iter().enumerate() {
            if tensor.shape() == shape {
                return Ok(self.available.swap_remove(i));
            }
        }
        
        // Allocate new tensor
        Tensor::zeros(shape)
    }
    
    pub fn return_tensor(&mut self, tensor: Tensor) {
        if self.available.len() < self.max_size {
            self.available.push(tensor);
        }
        // Otherwise let it drop
    }
}

// Use in model
impl Model {
    pub fn forward_with_pool(&self, input: &Tensor, pool: &mut TensorPool) -> Result<ModelOutput> {
        let temp_tensor = pool.get(&[input.shape()[0], self.config.hidden_size])?;
        
        // Use temp_tensor for computations
        let result = self.compute_with_temp(&temp_tensor)?;
        
        // Return to pool when done
        pool.return_tensor(temp_tensor);
        
        Ok(result)
    }
}
```

## Error Handling

### 1. Comprehensive Error Types

Define specific error types for different failure modes:

```rust
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("Configuration error: {0}")]
    Config(#[from] ConfigError),
    
    #[error("Tensor operation failed: {0}")]
    Tensor(#[from] TensorError),
    
    #[error("Invalid input shape: expected {expected:?}, got {actual:?}")]
    InvalidShape { expected: Vec<usize>, actual: Vec<usize> },
    
    #[error("Model not initialized")]
    NotInitialized,
    
    #[error("Incompatible model version: {version}")]
    IncompatibleVersion { version: String },
    
    #[error("Memory allocation failed: {size} bytes")]
    OutOfMemory { size: usize },
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Invalid parameter '{param}': {reason}")]
    InvalidParameter { param: String, reason: String },
    
    #[error("Missing required parameter: {param}")]
    MissingParameter { param: String },
    
    #[error("Validation failed: {reason}")]
    ValidationFailed { reason: String },
}
```

### 2. Input Validation

Validate inputs at API boundaries:

```rust
impl Model {
    pub fn forward(&self, input_ids: &Tensor, attention_mask: Option<&Tensor>) -> Result<ModelOutput> {
        // Validate input shapes
        self.validate_input_shape(input_ids)?;
        
        if let Some(mask) = attention_mask {
            self.validate_attention_mask(input_ids, mask)?;
        }
        
        // Validate tensor properties
        if input_ids.dtype() != DataType::Int64 {
            return Err(ModelError::InvalidShape {
                expected: vec![],
                actual: vec![],
            });
        }
        
        // Main forward logic
        self.forward_impl(input_ids, attention_mask)
    }
    
    fn validate_input_shape(&self, input_ids: &Tensor) -> Result<()> {
        let shape = input_ids.shape();
        
        if shape.len() != 2 {
            return Err(ModelError::InvalidShape {
                expected: vec![0, 0], // batch_size, seq_len
                actual: shape.to_vec(),
            });
        }
        
        let seq_len = shape[1];
        if seq_len > self.config.max_position_embeddings {
            return Err(ModelError::InvalidShape {
                expected: vec![0, self.config.max_position_embeddings],
                actual: shape.to_vec(),
            });
        }
        
        Ok(())
    }
    
    fn validate_attention_mask(&self, input_ids: &Tensor, mask: &Tensor) -> Result<()> {
        if input_ids.shape() != mask.shape() {
            return Err(ModelError::InvalidShape {
                expected: input_ids.shape().to_vec(),
                actual: mask.shape().to_vec(),
            });
        }
        Ok(())
    }
}
```

### 3. Graceful Degradation

Handle edge cases gracefully:

```rust
impl Model {
    pub fn generate_with_fallback(
        &mut self,
        input_ids: &Tensor,
        config: &GenerationConfig,
    ) -> Result<GenerationOutput> {
        // Try optimized generation first
        match self.generate_optimized(input_ids, config) {
            Ok(output) => Ok(output),
            Err(ModelError::OutOfMemory { .. }) => {
                log::warn!("Optimized generation failed, falling back to standard generation");
                self.generate_standard(input_ids, config)
            },
            Err(e) => Err(e),
        }
    }
    
    fn generate_optimized(&mut self, input_ids: &Tensor, config: &GenerationConfig) -> Result<GenerationOutput> {
        // High-performance generation with caching
        todo!()
    }
    
    fn generate_standard(&mut self, input_ids: &Tensor, config: &GenerationConfig) -> Result<GenerationOutput> {
        // Fallback generation without caching
        todo!()
    }
}
```

## Testing Strategies

### 1. Layered Testing Approach

Test at multiple levels:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    // Unit tests for individual components
    mod unit_tests {
        use super::*;
        
        #[test]
        fn test_attention_layer() {
            let config = AttentionConfig::default();
            let attention = MultiHeadAttention::new(&config).unwrap();
            
            let input = Tensor::randn(&[2, 10, 64]);
            let output = attention.forward(&input, &input, &input, None, None, false).unwrap();
            
            assert_eq!(output.0.shape(), input.shape());
        }
        
        #[test]
        fn test_feed_forward_layer() {
            let config = FFNConfig::default();
            let ffn = FeedForward::new(&config).unwrap();
            
            let input = Tensor::randn(&[2, 10, 64]);
            let output = ffn.forward(&input).unwrap();
            
            assert_eq!(output.shape(), input.shape());
        }
    }
    
    // Integration tests for complete model
    mod integration_tests {
        use super::*;
        
        #[test]
        fn test_full_model_forward() {
            let config = ModelConfig::default();
            let mut model = Model::new(config).unwrap();
            
            let input_ids = Tensor::randint(0, 1000, &[2, 10]);
            let output = model.forward(&input_ids, None).unwrap();
            
            assert_eq!(output.last_hidden_state.shape(), &[2, 10, config.hidden_size]);
        }
        
        #[test]
        fn test_generation_pipeline() {
            let config = ModelConfig::default();
            let mut model = Model::new(config).unwrap();
            
            let input_ids = Tensor::randint(0, 1000, &[1, 5]);
            let gen_config = GenerationConfig::default();
            
            let output = model.generate(&input_ids, &gen_config).unwrap();
            assert!(output.sequences.shape()[1] > input_ids.shape()[1]);
        }
    }
    
    // Property-based tests
    mod property_tests {
        use super::*;
        use proptest::prelude::*;
        
        proptest! {
            #[test]
            fn test_model_shape_invariants(
                batch_size in 1..8usize,
                seq_len in 1..128usize,
            ) {
                let config = ModelConfig::default();
                let mut model = Model::new(config.clone()).unwrap();
                
                let input_ids = Tensor::randint(0, config.vocab_size as i64, &[batch_size, seq_len]);
                let output = model.forward(&input_ids, None).unwrap();
                
                prop_assert_eq!(output.last_hidden_state.shape(), &[batch_size, seq_len, config.hidden_size]);
            }
        }
    }
}
```

### 2. Performance Benchmarks

Include performance tests:

```rust
#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn benchmark_forward_pass() {
        let config = ModelConfig::default();
        let mut model = Model::new(config).unwrap();
        
        let input_ids = Tensor::randint(0, 1000, &[8, 512]); // Large batch
        
        // Warmup
        for _ in 0..5 {
            let _ = model.forward(&input_ids, None).unwrap();
        }
        
        // Benchmark
        let start = Instant::now();
        for _ in 0..100 {
            let _ = model.forward(&input_ids, None).unwrap();
        }
        let duration = start.elapsed();
        
        let avg_duration = duration.as_secs_f32() / 100.0;
        println!("Average forward pass time: {:.2}ms", avg_duration * 1000.0);
        
        // Assert reasonable performance bounds
        assert!(avg_duration < 0.1, "Forward pass too slow: {:.2}ms", avg_duration * 1000.0);
    }
    
    #[test]
    fn benchmark_memory_usage() {
        let config = ModelConfig::default();
        let model = Model::new(config).unwrap();
        
        let memory_usage = get_memory_usage();
        println!("Model memory usage: {} MB", memory_usage / 1024 / 1024);
        
        // Assert reasonable memory bounds
        assert!(memory_usage < 500 * 1024 * 1024, "Model uses too much memory");
    }
}
```

## Documentation Standards

### 1. Module Documentation

Document modules comprehensively:

```rust
//! # Transformer Model Implementation
//!
//! This module provides a complete implementation of the Transformer architecture
//! as described in "Attention Is All You Need" (Vaswani et al., 2017).
//!
//! ## Features
//!
//! - Multi-head self-attention
//! - Position-wise feed-forward networks
//! - Layer normalization
//! - Residual connections
//! - Positional encodings
//!
//! ## Architecture
//!
//! The model consists of:
//! 1. Token embeddings
//! 2. Positional encodings  
//! 3. N transformer layers
//! 4. Final layer normalization
//!
//! ## Usage
//!
//! ```rust
//! use trustformers_models::transformer::{TransformerConfig, TransformerModel};
//!
//! // Create model configuration
//! let config = TransformerConfig {
//!     vocab_size: 50257,
//!     hidden_size: 768,
//!     num_hidden_layers: 12,
//!     num_attention_heads: 12,
//!     ..Default::default()
//! };
//!
//! // Initialize model
//! let mut model = TransformerModel::new(config)?;
//!
//! // Forward pass
//! let input_ids = Tensor::randint(0, 50257, &[1, 10]);
//! let output = model.forward(&input_ids, None)?;
//! ```
//!
//! ## Performance Considerations
//!
//! - Enable gradient checkpointing for large models
//! - Use mixed precision training when available
//! - Consider using flash attention for long sequences
//!
//! ## References
//!
//! - Vaswani, A., et al. (2017). "Attention is all you need."
//! - Radford, A., et al. (2019). "Language models are unsupervised multitask learners."
```

### 2. Function Documentation

Document functions with examples:

```rust
impl Model {
    /// Performs a forward pass through the model.
    ///
    /// # Arguments
    ///
    /// * `input_ids` - Input token IDs of shape [batch_size, sequence_length]
    /// * `attention_mask` - Optional attention mask of shape [batch_size, sequence_length]
    ///
    /// # Returns
    ///
    /// Returns a `ModelOutput` containing:
    /// - `last_hidden_state`: Final hidden states of shape [batch_size, sequence_length, hidden_size]
    /// - `past_key_values`: Optional cached key-value pairs for generation
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Input shapes are invalid
    /// - Sequence length exceeds maximum position embeddings
    /// - Tensor operations fail
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use trustformers_models::transformer::*;
    /// # let config = TransformerConfig::default();
    /// # let mut model = TransformerModel::new(config)?;
    /// let input_ids = Tensor::tensor([[1, 2, 3, 4, 5]]);
    /// let output = model.forward(&input_ids, None)?;
    /// 
    /// assert_eq!(output.last_hidden_state.shape(), &[1, 5, config.hidden_size]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Performance Notes
    ///
    /// This function allocates temporary tensors for attention computation.
    /// For memory-constrained environments, consider using `forward_with_cache`.
    pub fn forward(
        &mut self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<ModelOutput> {
        // Implementation...
    }
}
```

## Compatibility Guidelines

### 1. HuggingFace Compatibility

Ensure compatibility with HuggingFace models:

```rust
impl Model {
    /// Load weights from HuggingFace format
    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        let hub_api = HubApi::new()?;
        let repo = hub_api.model(model_name.to_string());
        
        // Download config
        let config_path = repo.get("config.json")?;
        let config: ModelConfig = serde_json::from_reader(File::open(config_path)?)?;
        
        // Download model weights
        let model_path = repo.get("pytorch_model.bin")?;
        let state_dict = load_pytorch_weights(&model_path)?;
        
        // Create model and load weights
        let mut model = Self::new(config)?;
        model.load_state_dict(&state_dict)?;
        
        Ok(model)
    }
    
    /// Convert to HuggingFace format
    pub fn to_hf_format(&self) -> Result<HashMap<String, Tensor>> {
        let mut state_dict = HashMap::new();
        
        // Map internal parameter names to HuggingFace convention
        for (name, param) in self.named_parameters() {
            let hf_name = self.map_parameter_name(&name);
            state_dict.insert(hf_name, param.clone());
        }
        
        Ok(state_dict)
    }
    
    fn map_parameter_name(&self, internal_name: &str) -> String {
        // Convert internal naming to HuggingFace convention
        internal_name
            .replace("transformer.layers", "transformer.h")
            .replace("attention.q_proj", "attn.c_attn")
            .replace("attention.k_proj", "attn.c_attn")
            .replace("attention.v_proj", "attn.c_attn")
            .replace("attention.o_proj", "attn.c_proj")
    }
}
```

### 2. Version Compatibility

Handle version differences gracefully:

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    // Core parameters (always present)
    pub vocab_size: usize,
    pub hidden_size: usize,
    
    // Version-specific parameters
    #[serde(default)]
    pub version: String,
    
    // Optional parameters (added in later versions)
    #[serde(default)]
    pub use_bias: Option<bool>,
    
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
}

impl ModelConfig {
    pub fn migrate_from_v1(mut self) -> Self {
        if self.version == "1.0" || self.version.is_empty() {
            // Set defaults for new parameters
            self.use_bias = Some(true);
            self.version = "2.0".to_string();
        }
        self
    }
    
    pub fn is_compatible_with(&self, other_version: &str) -> bool {
        let major_version = self.version.split('.').next().unwrap_or("1");
        let other_major = other_version.split('.').next().unwrap_or("1");
        major_version == other_major
    }
}
```

## Security Considerations

### 1. Input Sanitization

Sanitize untrusted inputs:

```rust
impl Model {
    pub fn forward_safe(&mut self, input_ids: &Tensor) -> Result<ModelOutput> {
        // Validate input ranges
        self.validate_token_ids(input_ids)?;
        
        // Limit computational resources
        self.validate_computational_limits(input_ids)?;
        
        // Proceed with normal forward pass
        self.forward(input_ids, None)
    }
    
    fn validate_token_ids(&self, input_ids: &Tensor) -> Result<()> {
        let min_val = input_ids.min()?.item::<i64>();
        let max_val = input_ids.max()?.item::<i64>();
        
        if min_val < 0 {
            return Err(ModelError::InvalidInput("Negative token IDs not allowed".into()));
        }
        
        if max_val >= self.config.vocab_size as i64 {
            return Err(ModelError::InvalidInput(
                format!("Token ID {} exceeds vocabulary size {}", max_val, self.config.vocab_size)
            ));
        }
        
        Ok(())
    }
    
    fn validate_computational_limits(&self, input_ids: &Tensor) -> Result<()> {
        let batch_size = input_ids.shape()[0];
        let seq_len = input_ids.shape()[1];
        
        // Prevent excessive memory usage
        let estimated_memory = batch_size * seq_len * self.config.hidden_size * 4; // bytes
        const MAX_MEMORY: usize = 2 * 1024 * 1024 * 1024; // 2GB
        
        if estimated_memory > MAX_MEMORY {
            return Err(ModelError::OutOfMemory { size: estimated_memory });
        }
        
        // Prevent excessive computation
        const MAX_SEQUENCE_LENGTH: usize = 8192;
        if seq_len > MAX_SEQUENCE_LENGTH {
            return Err(ModelError::InvalidInput(
                format!("Sequence length {} exceeds maximum {}", seq_len, MAX_SEQUENCE_LENGTH)
            ));
        }
        
        Ok(())
    }
}
```

### 2. Safe Deserialization

Handle model loading securely:

```rust
impl Model {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        
        // Read and validate magic number
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != b"TFRS_MDL" {
            return Err(ModelError::InvalidFormat("Invalid file format".into()));
        }
        
        // Read version
        let mut version = [0u8; 4];
        reader.read_exact(&mut version)?;
        let version = u32::from_le_bytes(version);
        
        if version > CURRENT_FORMAT_VERSION {
            return Err(ModelError::IncompatibleVersion { 
                version: version.to_string() 
            });
        }
        
        // Deserialize with size limits
        let config: ModelConfig = bincode::deserialize_from(
            reader.take(MAX_CONFIG_SIZE)
        )?;
        
        // Validate config before proceeding
        config.validate()?;
        
        // Create model
        let mut model = Self::new(config)?;
        
        // Load weights with validation
        model.load_weights_safe(&mut reader)?;
        
        Ok(model)
    }
    
    fn load_weights_safe<R: Read>(&mut self, reader: &mut R) -> Result<()> {
        for (name, param) in self.named_parameters() {
            // Read parameter metadata
            let mut meta = [0u8; 16];
            reader.read_exact(&mut meta)?;
            
            let dtype = u32::from_le_bytes([meta[0], meta[1], meta[2], meta[3]]);
            let ndim = u32::from_le_bytes([meta[4], meta[5], meta[6], meta[7]]);
            let size = u64::from_le_bytes([
                meta[8], meta[9], meta[10], meta[11],
                meta[12], meta[13], meta[14], meta[15]
            ]);
            
            // Validate parameter size
            if size > MAX_PARAMETER_SIZE {
                return Err(ModelError::InvalidInput(
                    format!("Parameter {} too large: {} bytes", name, size)
                ));
            }
            
            // Load parameter data
            let mut data = vec![0u8; size as usize];
            reader.read_exact(&mut data)?;
            
            // Convert and validate
            let tensor = Tensor::from_bytes(&data, dtype.into(), &param.shape())?;
            param.copy_(&tensor)?;
        }
        
        Ok(())
    }
}

const MAX_CONFIG_SIZE: u64 = 1024 * 1024; // 1MB
const MAX_PARAMETER_SIZE: u64 = 100 * 1024 * 1024; // 100MB per parameter
const CURRENT_FORMAT_VERSION: u32 = 1;
```

Following these best practices ensures that your model implementations are robust, performant, maintainable, and secure.