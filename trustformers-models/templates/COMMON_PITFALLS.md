# Common Pitfalls in Model Implementation

This document catalogs common mistakes, gotchas, and anti-patterns encountered when implementing transformer models, along with solutions and prevention strategies.

## Table of Contents

1. [Configuration Pitfalls](#configuration-pitfalls)
2. [Architecture Implementation Issues](#architecture-implementation-issues)
3. [Memory Management Problems](#memory-management-problems)
4. [Performance Anti-patterns](#performance-anti-patterns)
5. [Numerical Stability Issues](#numerical-stability-issues)
6. [Testing Oversights](#testing-oversights)
7. [Generation Implementation Bugs](#generation-implementation-bugs)
8. [Weight Loading Errors](#weight-loading-errors)
9. [Training Compatibility Issues](#training-compatibility-issues)
10. [Debugging Challenges](#debugging-challenges)

## Configuration Pitfalls

### 1. Inconsistent Dimension Validation

**Problem:** Not validating that dimensions are compatible across model components.

```rust
// ❌ BAD: No validation
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
}

impl ModelConfig {
    // Missing validation - will cause runtime errors
}
```

**Issues:**
- `hidden_size` not divisible by `num_attention_heads` causes invalid head dimensions
- Inconsistent intermediate sizes can cause shape mismatches

**Solution:**
```rust
// ✅ GOOD: Comprehensive validation
impl Config for ModelConfig {
    fn validate(&self) -> Result<()> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(ConfigError::InvalidParameter {
                param: "hidden_size".to_string(),
                reason: format!(
                    "must be divisible by num_attention_heads ({})", 
                    self.num_attention_heads
                ),
            });
        }
        
        if self.intermediate_size == 0 {
            return Err(ConfigError::InvalidParameter {
                param: "intermediate_size".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        
        if self.num_attention_heads == 0 {
            return Err(ConfigError::InvalidParameter {
                param: "num_attention_heads".to_string(),
                reason: "must be greater than 0".to_string(),
            });
        }
        
        Ok(())
    }
}
```

### 2. Missing Default Values

**Problem:** Configurations without sensible defaults make the API harder to use.

```rust
// ❌ BAD: No defaults, difficult to use
#[derive(Debug, Clone)]
pub struct ComplexConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub dropout: f32,
    pub epsilon: f32,
    pub initializer_range: f32,
    // ... 20 more parameters
}

// User must specify all parameters
let config = ComplexConfig {
    vocab_size: 50257,
    hidden_size: 768,
    num_layers: 12,
    dropout: 0.1,
    epsilon: 1e-5,
    initializer_range: 0.02,
    // ... 20 more parameters
};
```

**Solution:**
```rust
// ✅ GOOD: Provide sensible defaults
impl Default for ComplexConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50257,
            hidden_size: 768,
            num_layers: 12,
            dropout: 0.1,
            epsilon: 1e-5,
            initializer_range: 0.02,
            // ... sensible defaults for all parameters
        }
    }
}

// Also provide convenience constructors
impl ComplexConfig {
    pub fn small() -> Self {
        Self {
            hidden_size: 512,
            num_layers: 6,
            ..Default::default()
        }
    }
    
    pub fn for_domain(domain: Domain) -> Self {
        match domain {
            Domain::Code => Self {
                vocab_size: 50400,
                max_position_embeddings: 8192,
                ..Default::default()
            },
            Domain::Math => Self {
                vocab_size: 60000,
                ..Default::default()
            },
        }
    }
}
```

## Architecture Implementation Issues

### 3. Incorrect Attention Mask Handling

**Problem:** Mishandling attention masks, especially for causal (decoder) models.

```rust
// ❌ BAD: Incorrect mask application
impl Attention {
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let scores = q.matmul(&k.transpose(-2, -1))?;
        
        // Wrong: adding mask directly without proper format
        let scores = if let Some(mask) = mask {
            scores.add(mask)?  // This is often wrong!
        } else {
            scores
        };
        
        let attn_weights = scores.softmax(-1)?;
        attn_weights.matmul(v)
    }
}
```

**Issues:**
- Attention masks need specific format (0 for attend, large negative for mask)
- Causal masks need to be combined with padding masks correctly
- Shape broadcasting issues

**Solution:**
```rust
// ✅ GOOD: Proper mask handling
impl Attention {
    fn forward(&self, q: &Tensor, k: &Tensor, v: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let scores = q.matmul(&k.transpose(-2, -1))?
            .div_scalar((self.head_dim as f32).sqrt())?;
        
        // Create causal mask for decoder
        let causal_mask = if self.is_causal {
            Some(self.create_causal_mask(q.shape()[2], k.shape()[2])?)
        } else {
            None
        };
        
        // Combine masks properly
        let final_mask = self.combine_masks(attention_mask, causal_mask.as_ref())?;
        
        let scores = if let Some(mask) = final_mask {
            // Proper mask format: 0 for attend, large negative for mask
            scores.add(&mask)?
        } else {
            scores
        };
        
        let attn_weights = scores.softmax(-1)?;
        attn_weights.matmul(v)
    }
    
    fn create_causal_mask(&self, q_len: usize, k_len: usize) -> Result<Tensor> {
        let mask = Tensor::ones(&[q_len, k_len])?;
        let mask = mask.tril(k_len as i64 - q_len as i64)?; // Lower triangular
        
        // Convert to attention mask format
        Tensor::where_scalar(&mask, 0.0, f32::NEG_INFINITY)
    }
    
    fn combine_masks(&self, padding_mask: Option<&Tensor>, causal_mask: Option<&Tensor>) -> Result<Option<Tensor>> {
        match (padding_mask, causal_mask) {
            (Some(pad), Some(causal)) => {
                // Convert padding mask to proper format
                let pad_mask = pad.unsqueeze(1)?.unsqueeze(2)?;
                let pad_mask = Tensor::where_scalar(pad_mask, 0.0, f32::NEG_INFINITY)?;
                
                // Combine masks
                Ok(Some(pad_mask.add(causal)?))
            },
            (Some(pad), None) => {
                let pad_mask = pad.unsqueeze(1)?.unsqueeze(2)?;
                Ok(Some(Tensor::where_scalar(pad_mask, 0.0, f32::NEG_INFINITY)?))
            },
            (None, Some(causal)) => Ok(Some(causal.clone())),
            (None, None) => Ok(None),
        }
    }
}
```

### 4. Incorrect Position Encoding Implementation

**Problem:** Position encodings not applied correctly or with wrong dimensions.

```rust
// ❌ BAD: Several issues
impl Model {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let embeddings = self.token_embeddings.forward(input_ids)?;
        
        // Wrong: Position encodings should be added, not concatenated
        let positions = self.position_embeddings.forward(&input_ids)?;
        let hidden_states = Tensor::cat(&[&embeddings, &positions], -1)?;
        
        // ... rest of forward pass
    }
}
```

**Issues:**
- Position encodings should be added to token embeddings, not concatenated
- Position IDs need to be generated correctly
- RoPE needs to be applied to queries and keys, not embeddings

**Solution:**
```rust
// ✅ GOOD: Correct position encoding
impl Model {
    fn forward(&self, input_ids: &Tensor, position_ids: Option<&Tensor>) -> Result<Tensor> {
        let embeddings = self.token_embeddings.forward(input_ids)?;
        
        // Generate position IDs if not provided
        let position_ids = if let Some(pos_ids) = position_ids {
            pos_ids.clone()
        } else {
            let seq_len = input_ids.shape()[1];
            let batch_size = input_ids.shape()[0];
            Tensor::arange(0, seq_len as i64, 1)?
                .unsqueeze(0)?
                .expand(&[batch_size, seq_len])?
        };
        
        // Add position embeddings (not concatenate)
        let position_embeddings = self.position_embeddings.forward(&position_ids)?;
        let hidden_states = embeddings.add(&position_embeddings)?;
        
        // ... rest of forward pass
        Ok(hidden_states)
    }
}

// For RoPE models
impl RotaryEmbedding {
    fn apply_to_attention(&self, q: &Tensor, k: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let (cos, sin) = self.forward(position_ids)?;
        
        // Apply to queries and keys, not embeddings
        let q_embed = self.apply_rotary_pos_emb(&q, &cos, &sin)?;
        let k_embed = self.apply_rotary_pos_emb(&k, &cos, &sin)?;
        
        Ok((q_embed, k_embed))
    }
}
```

### 5. Layer Normalization Placement Errors

**Problem:** Incorrect placement of layer normalization (pre-norm vs post-norm).

```rust
// ❌ BAD: Inconsistent or incorrect norm placement
impl TransformerLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Wrong: This mixes pre-norm and post-norm
        let attn_out = self.attention.forward(&x)?;
        let x = self.norm1.forward(&x.add(&attn_out)?)?; // Post-norm for attention
        
        let ffn_out = self.norm2.forward(&x)?; // Pre-norm for FFN
        let ffn_out = self.ffn.forward(&ffn_out)?;
        Ok(x.add(&ffn_out)?)
    }
}
```

**Solution:**
```rust
// ✅ GOOD: Consistent normalization strategy
impl TransformerLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Pre-norm (modern approach)
        let normed_x = self.norm1.forward(x)?;
        let attn_out = self.attention.forward(&normed_x)?;
        let x = x.add(&attn_out)?; // Residual connection
        
        let normed_x = self.norm2.forward(&x)?;
        let ffn_out = self.ffn.forward(&normed_x)?;
        Ok(x.add(&ffn_out)?) // Residual connection
    }
    
    // Alternative: Post-norm (traditional approach)
    fn forward_post_norm(&self, x: &Tensor) -> Result<Tensor> {
        let attn_out = self.attention.forward(x)?;
        let x = self.norm1.forward(&x.add(&attn_out)?)?;
        
        let ffn_out = self.ffn.forward(&x)?;
        let x = self.norm2.forward(&x.add(&ffn_out)?)?;
        Ok(x)
    }
}
```

## Memory Management Problems

### 6. Memory Leaks in Generation

**Problem:** Accumulating tensors during generation without proper cleanup.

```rust
// ❌ BAD: Memory accumulates indefinitely
impl Model {
    fn generate(&mut self, input_ids: &Tensor, max_length: usize) -> Result<Tensor> {
        let mut generated = input_ids.clone();
        let mut past_key_values = Vec::new(); // Grows indefinitely!
        
        for _ in 0..max_length {
            let output = self.forward(&generated, Some(&past_key_values))?;
            
            // Keep accumulating past states
            past_key_values.push(output.past_key_values);
            
            let next_token = output.logits.argmax(-1)?;
            generated = Tensor::cat(&[&generated, &next_token], 1)?;
        }
        
        Ok(generated)
    }
}
```

**Solution:**
```rust
// ✅ GOOD: Proper memory management
impl Model {
    fn generate(&mut self, input_ids: &Tensor, max_length: usize) -> Result<Tensor> {
        let mut generated = input_ids.clone();
        let mut past_key_values: Option<Vec<(Tensor, Tensor)>> = None;
        
        for step in 0..max_length {
            // Only pass new token if using cache
            let model_input = if past_key_values.is_some() && step > 0 {
                generated.slice(1, -1, generated.shape()[1] as i64, 1)? // Only last token
            } else {
                generated.clone()
            };
            
            let output = self.forward(&model_input, past_key_values.as_ref())?;
            
            // Update cache efficiently
            past_key_values = output.past_key_values;
            
            let next_token = output.logits.slice(1, -1, output.logits.shape()[1] as i64, 1)?
                .argmax(-1)?;
            
            // Check for EOS token
            if self.is_eos_token(&next_token) {
                break;
            }
            
            generated = Tensor::cat(&[&generated, &next_token.unsqueeze(1)?], 1)?;
        }
        
        Ok(generated)
    }
}
```

### 7. Inefficient Tensor Operations

**Problem:** Creating unnecessary intermediate tensors.

```rust
// ❌ BAD: Creates many intermediate tensors
impl Attention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;
        
        // Many intermediate tensors
        let q = q.view(&[batch_size, seq_len, self.num_heads, self.head_dim])?;
        let q = q.transpose(1, 2)?;
        let k = k.view(&[batch_size, seq_len, self.num_heads, self.head_dim])?;
        let k = k.transpose(1, 2)?;
        let v = v.view(&[batch_size, seq_len, self.num_heads, self.head_dim])?;
        let v = v.transpose(1, 2)?;
        
        // More operations creating temporaries...
    }
}
```

**Solution:**
```rust
// ✅ GOOD: Minimize intermediate tensors
impl Attention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = x.shape3()?;
        
        // Use fused operations where possible
        let qkv = self.qkv_proj.forward(x)?; // Single projection
        let qkv = qkv.view(&[batch_size, seq_len, 3, self.num_heads, self.head_dim])?
            .transpose(1, 3)?; // [batch, head, seq, 3, dim]
        
        let q = qkv.select(3, 0)?; // No copy, just view
        let k = qkv.select(3, 1)?;
        let v = qkv.select(3, 2)?;
        
        // Use in-place operations where safe
        self.scaled_dot_product_attention(&q, &k, &v)
    }
    
    // Use buffer pools for repeated operations
    fn forward_with_buffer(&self, x: &Tensor, buffer: &mut AttentionBuffer) -> Result<Tensor> {
        buffer.resize_if_needed(x.shape())?;
        
        // Reuse pre-allocated buffers
        self.qkv_proj.forward_into(&x, &mut buffer.qkv_buffer)?;
        // ... use buffers for computation
    }
}
```

## Performance Anti-patterns

### 8. Inefficient Batch Processing

**Problem:** Not batching operations efficiently or forcing unnecessary synchronization.

```rust
// ❌ BAD: Sequential processing
impl Model {
    fn process_batch(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        let mut outputs = Vec::new();
        
        for input in inputs {
            // Process one at a time - very inefficient!
            let output = self.forward(input)?;
            outputs.push(output);
        }
        
        Ok(outputs)
    }
}
```

**Solution:**
```rust
// ✅ GOOD: Proper batching
impl Model {
    fn process_batch(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        // Batch inputs together
        let batched_input = Tensor::stack(inputs, 0)?;
        let batched_output = self.forward(&batched_input)?;
        
        // Split outputs
        let outputs = batched_output.unbind(0)?;
        Ok(outputs)
    }
    
    // For variable-length sequences
    fn process_variable_batch(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        // Pad to same length
        let max_len = inputs.iter().map(|t| t.shape()[1]).max().unwrap_or(0);
        let padded_inputs: Vec<Tensor> = inputs.iter()
            .map(|t| self.pad_to_length(t, max_len))
            .collect::<Result<Vec<_>>>()?;
        
        let batched_input = Tensor::stack(&padded_inputs, 0)?;
        
        // Create attention mask for padding
        let attention_mask = self.create_padding_mask(&inputs, max_len)?;
        
        let output = self.forward(&batched_input, Some(&attention_mask))?;
        
        // Unpad outputs to original lengths
        let outputs = inputs.iter().zip(output.unbind(0)?)
            .map(|(orig, out)| self.unpad_to_length(&out, orig.shape()[1]))
            .collect::<Result<Vec<_>>>()?;
        
        Ok(outputs)
    }
}
```

### 9. Unnecessary CPU-GPU Transfers

**Problem:** Moving tensors between devices frequently.

```rust
// ❌ BAD: Frequent device transfers
impl Model {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = input.to_device(Device::Cpu)?; // Unnecessary transfer
        let embedded = self.embeddings.forward(&x)?;
        
        let x = embedded.to_device(Device::Cuda(0))?; // Another transfer
        let attention_out = self.attention.forward(&x)?;
        
        let x = attention_out.to_device(Device::Cpu)?; // Yet another transfer
        self.output_layer.forward(&x)
    }
}
```

**Solution:**
```rust
// ✅ GOOD: Minimize device transfers
impl Model {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Ensure input is on correct device once
        let device = self.parameters()[0].device();
        let x = if input.device() != device {
            input.to_device(device)?
        } else {
            input.clone()
        };
        
        // All operations stay on same device
        let embedded = self.embeddings.forward(&x)?;
        let attention_out = self.attention.forward(&embedded)?;
        let output = self.output_layer.forward(&attention_out)?;
        
        Ok(output)
    }
    
    // Helper to ensure model is on correct device
    pub fn to_device(&mut self, device: Device) -> Result<()> {
        for param in self.parameters_mut() {
            *param = param.to_device(device)?;
        }
        Ok(())
    }
}
```

## Numerical Stability Issues

### 10. Softmax Overflow

**Problem:** Softmax operations on large logits causing overflow/underflow.

```rust
// ❌ BAD: Can overflow with large logits
impl Attention {
    fn compute_attention(&self, scores: &Tensor) -> Result<Tensor> {
        // scores can be very large, causing overflow
        let attn_weights = scores.softmax(-1)?;
        Ok(attn_weights)
    }
}
```

**Solution:**
```rust
// ✅ GOOD: Numerically stable softmax
impl Attention {
    fn compute_attention(&self, scores: &Tensor) -> Result<Tensor> {
        // Subtract max for numerical stability (softmax is translation invariant)
        let max_scores = scores.max_dim(-1, true)?.0;
        let stable_scores = scores.sub(&max_scores)?;
        
        let attn_weights = stable_scores.softmax(-1)?;
        Ok(attn_weights)
    }
    
    // For very large sequences, consider using sparse attention
    fn compute_sparse_attention(&self, scores: &Tensor, sparsity_pattern: &Tensor) -> Result<Tensor> {
        // Apply sparsity mask before softmax
        let masked_scores = Tensor::where_tensor(
            sparsity_pattern,
            scores,
            &Tensor::scalar(f32::NEG_INFINITY)?
        )?;
        
        self.compute_attention(&masked_scores)
    }
}
```

### 11. Gradient Explosion/Vanishing

**Problem:** Poor initialization or missing gradient clipping leading to training instability.

```rust
// ❌ BAD: Poor initialization
impl Linear {
    fn new(in_features: usize, out_features: usize) -> Result<Self> {
        // Wrong: Random initialization without proper scaling
        let weight = Tensor::randn(&[out_features, in_features])?;
        let bias = Tensor::zeros(&[out_features])?;
        
        Ok(Self { weight, bias })
    }
}
```

**Solution:**
```rust
// ✅ GOOD: Proper initialization
impl Linear {
    fn new(in_features: usize, out_features: usize) -> Result<Self> {
        // Xavier/Glorot initialization for better gradient flow
        let scale = (2.0 / (in_features + out_features) as f32).sqrt();
        let weight = Tensor::randn(&[out_features, in_features])?.mul_scalar(scale)?;
        let bias = Tensor::zeros(&[out_features])?;
        
        Ok(Self { weight, bias })
    }
    
    // For ReLU activations, use Kaiming initialization
    fn new_kaiming(in_features: usize, out_features: usize) -> Result<Self> {
        let scale = (2.0 / in_features as f32).sqrt();
        let weight = Tensor::randn(&[out_features, in_features])?.mul_scalar(scale)?;
        let bias = Tensor::zeros(&[out_features])?;
        
        Ok(Self { weight, bias })
    }
}

// Implement gradient clipping
impl Model {
    pub fn clip_gradients(&mut self, max_norm: f32) -> Result<f32> {
        let mut total_norm = 0.0f32;
        
        // Compute total gradient norm
        for param in self.parameters() {
            if let Some(grad) = param.grad() {
                total_norm += grad.norm()?.item::<f32>().powi(2);
            }
        }
        total_norm = total_norm.sqrt();
        
        // Clip if necessary
        if total_norm > max_norm {
            let scale = max_norm / total_norm;
            for param in self.parameters() {
                if let Some(grad) = param.grad() {
                    let clipped_grad = grad.mul_scalar(scale)?;
                    param.set_grad(Some(clipped_grad))?;
                }
            }
        }
        
        Ok(total_norm)
    }
}
```

## Testing Oversights

### 12. Inadequate Shape Testing

**Problem:** Not testing edge cases for tensor shapes.

```rust
// ❌ BAD: Only tests typical cases
#[cfg(test)]
mod tests {
    #[test]
    fn test_forward() {
        let model = Model::new(Config::default()).unwrap();
        let input = Tensor::randint(0, 1000, &[2, 10]); // Only one shape
        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape()[0], 2);
    }
}
```

**Solution:**
```rust
// ✅ GOOD: Test various shapes and edge cases
#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    
    #[test]
    fn test_forward_various_shapes() {
        let model = Model::new(Config::default()).unwrap();
        
        let test_cases = vec![
            (1, 1),    // Single token
            (1, 512),  // Long sequence
            (8, 128),  // Batch processing
            (1, 2048), // Very long sequence
        ];
        
        for (batch_size, seq_len) in test_cases {
            let input = Tensor::randint(0, 1000, &[batch_size, seq_len]);
            let output = model.forward(&input).unwrap();
            assert_eq!(output.shape(), &[batch_size, seq_len, model.config.hidden_size]);
        }
    }
    
    // Property-based testing
    proptest! {
        #[test]
        fn test_forward_arbitrary_shapes(
            batch_size in 1..16usize,
            seq_len in 1..1024usize,
        ) {
            let model = Model::new(Config::default()).unwrap();
            let input = Tensor::randint(0, 1000, &[batch_size, seq_len]);
            
            let output = model.forward(&input).unwrap();
            prop_assert_eq!(output.shape(), &[batch_size, seq_len, model.config.hidden_size]);
            
            // Test numerical properties
            prop_assert!(!output.isnan().any().unwrap());
            prop_assert!(!output.isinf().any().unwrap());
        }
    }
    
    #[test]
    fn test_empty_sequences() {
        let model = Model::new(Config::default()).unwrap();
        
        // Test zero-length sequences (should fail gracefully)
        let empty_input = Tensor::empty(&[1, 0]);
        assert!(model.forward(&empty_input).is_err());
    }
    
    #[test]
    fn test_maximum_sequence_length() {
        let config = Config {
            max_position_embeddings: 1024,
            ..Default::default()
        };
        let model = Model::new(config).unwrap();
        
        // Test at maximum length
        let input = Tensor::randint(0, 1000, &[1, 1024]);
        assert!(model.forward(&input).is_ok());
        
        // Test exceeding maximum length
        let input = Tensor::randint(0, 1000, &[1, 1025]);
        assert!(model.forward(&input).is_err());
    }
}
```

### 13. Missing Determinism Tests

**Problem:** Not ensuring reproducible results.

```rust
// ❌ BAD: No determinism testing
#[test]
fn test_model() {
    let model = Model::new(Config::default()).unwrap();
    let input = Tensor::randn(&[1, 10]);
    let output = model.forward(&input).unwrap();
    // No check for reproducibility
}
```

**Solution:**
```rust
// ✅ GOOD: Test determinism
#[test]
fn test_deterministic_output() {
    let config = Config::default();
    
    // Set random seed
    torch::manual_seed(42);
    let mut model1 = Model::new(config.clone()).unwrap();
    let input = Tensor::randint(0, 1000, &[2, 10]);
    let output1 = model1.forward(&input).unwrap();
    
    // Reset seed and create identical model
    torch::manual_seed(42);
    let mut model2 = Model::new(config).unwrap();
    let output2 = model2.forward(&input).unwrap();
    
    // Outputs should be identical
    let diff = output1.sub(&output2).unwrap().abs().max().unwrap().item::<f32>();
    assert!(diff < 1e-6, "Non-deterministic output: max diff = {}", diff);
}

#[test]
fn test_training_determinism() {
    torch::manual_seed(42);
    let mut model1 = Model::new(Config::default()).unwrap();
    
    torch::manual_seed(42);
    let mut model2 = Model::new(Config::default()).unwrap();
    
    let input = Tensor::randint(0, 1000, &[2, 10]);
    let target = Tensor::randint(0, 1000, &[2, 10]);
    
    // Run identical training steps
    for _ in 0..5 {
        let loss1 = model1.compute_loss(&input, &target).unwrap();
        loss1.backward().unwrap();
        model1.step_optimizer().unwrap();
        
        let loss2 = model2.compute_loss(&input, &target).unwrap();
        loss2.backward().unwrap();
        model2.step_optimizer().unwrap();
        
        // Losses should be identical
        let loss_diff = (loss1.item::<f32>() - loss2.item::<f32>()).abs();
        assert!(loss_diff < 1e-6, "Non-deterministic training");
    }
}
```

## Generation Implementation Bugs

### 14. Incorrect Beam Search Implementation

**Problem:** Beam search with incorrect scoring or pruning logic.

```rust
// ❌ BAD: Incorrect beam search
impl Model {
    fn beam_search(&mut self, input_ids: &Tensor, num_beams: usize) -> Result<Tensor> {
        let mut beams = vec![BeamSequence::new(input_ids.clone(), 0.0)];
        
        for _ in 0..100 {
            let mut new_beams = Vec::new();
            
            for beam in &beams {
                let output = self.forward(&beam.sequence)?;
                let logits = output.logits.slice(1, -1, output.logits.shape()[1] as i64, 1)?;
                
                // Wrong: Not considering length normalization
                let (top_scores, top_indices) = logits.topk(num_beams, -1, false)?;
                
                for i in 0..num_beams {
                    let token = top_indices.get_item(&[i])? as u32;
                    let score = top_scores.get_item(&[i])?;
                    
                    // Wrong: Adding raw scores without proper normalization
                    let new_score = beam.score + score;
                    
                    let new_sequence = Tensor::cat(&[
                        &beam.sequence,
                        &Tensor::scalar(token as f32)?.unsqueeze(0)?
                    ], 1)?;
                    
                    new_beams.push(BeamSequence::new(new_sequence, new_score));
                }
            }
            
            // Wrong: Simple sorting without considering completion
            new_beams.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            beams = new_beams.into_iter().take(num_beams).collect();
        }
        
        Ok(beams[0].sequence.clone())
    }
}
```

**Solution:**
```rust
// ✅ GOOD: Correct beam search implementation
impl Model {
    fn beam_search(&mut self, input_ids: &Tensor, config: &BeamSearchConfig) -> Result<Vec<Tensor>> {
        let mut active_beams = vec![BeamSequence::new(input_ids.clone(), 0.0)];
        let mut completed_beams = Vec::new();
        
        for step in 0..config.max_length {
            let mut candidates = Vec::new();
            
            for beam in &active_beams {
                if beam.is_complete(config.eos_token_id) {
                    completed_beams.push(beam.clone());
                    continue;
                }
                
                let output = self.forward(&beam.sequence)?;
                let logits = output.logits.slice(1, -1, output.logits.shape()[1] as i64, 1)?;
                let log_probs = logits.log_softmax(-1)?;
                
                let (top_scores, top_indices) = log_probs.topk(config.num_beams, -1, false)?;
                
                for i in 0..config.num_beams {
                    let token = top_indices.get_item(&[i])? as u32;
                    let log_prob = top_scores.get_item(&[i])?;
                    
                    let new_sequence = Tensor::cat(&[
                        &beam.sequence,
                        &Tensor::scalar(token as f32)?.unsqueeze(0)?
                    ], 1)?;
                    
                    // Proper score accumulation (log probabilities)
                    let new_score = beam.score + log_prob;
                    
                    // Length normalization
                    let normalized_score = new_score / new_sequence.shape()[1] as f32;
                    
                    candidates.push(BeamSequence::new(new_sequence, normalized_score));
                }
            }
            
            // Separate active and completed beams
            candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            
            active_beams.clear();
            for candidate in candidates.into_iter().take(config.num_beams) {
                if candidate.is_complete(config.eos_token_id) {
                    completed_beams.push(candidate);
                } else {
                    active_beams.push(candidate);
                }
            }
            
            // Early stopping condition
            if config.early_stopping && active_beams.is_empty() {
                break;
            }
            
            // Ensure we have enough beams
            if completed_beams.len() >= config.num_return_sequences {
                break;
            }
        }
        
        // Combine and sort all beams
        completed_beams.extend(active_beams);
        completed_beams.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        
        Ok(completed_beams.into_iter()
           .take(config.num_return_sequences)
           .map(|beam| beam.sequence)
           .collect())
    }
}

#[derive(Clone)]
struct BeamSequence {
    sequence: Tensor,
    score: f32,
}

impl BeamSequence {
    fn new(sequence: Tensor, score: f32) -> Self {
        Self { sequence, score }
    }
    
    fn is_complete(&self, eos_token_id: Option<u32>) -> bool {
        if let Some(eos_id) = eos_token_id {
            if let Ok(last_token) = self.sequence.slice(1, -1, self.sequence.shape()[1] as i64, 1) {
                return last_token.item::<i64>() == eos_id as i64;
            }
        }
        false
    }
}
```

## Weight Loading Errors

### 15. Incorrect Parameter Mapping

**Problem:** Mismatching parameter names when loading from external formats.

```rust
// ❌ BAD: Hardcoded parameter mapping
impl Model {
    fn load_huggingface_weights(&mut self, state_dict: &HashMap<String, Tensor>) -> Result<()> {
        // Wrong: Assumes exact name matches
        for (name, param) in self.named_parameters() {
            if let Some(weight) = state_dict.get(&name) {
                param.copy_(weight)?;
            }
        }
        Ok(())
    }
}
```

**Solution:**
```rust
// ✅ GOOD: Flexible parameter mapping
impl Model {
    fn load_huggingface_weights(&mut self, state_dict: &HashMap<String, Tensor>) -> Result<()> {
        let parameter_mapping = self.create_parameter_mapping();
        let mut loaded_params = HashSet::new();
        let mut missing_params = Vec::new();
        
        for (internal_name, param) in self.named_parameters() {
            if let Some(external_name) = parameter_mapping.get(&internal_name) {
                if let Some(weight) = state_dict.get(external_name) {
                    // Validate shapes before loading
                    if weight.shape() != param.shape() {
                        return Err(ModelError::ShapeMismatch {
                            param_name: internal_name.clone(),
                            expected: param.shape().to_vec(),
                            actual: weight.shape().to_vec(),
                        });
                    }
                    
                    param.copy_(weight)?;
                    loaded_params.insert(external_name.clone());
                } else {
                    missing_params.push(external_name.clone());
                }
            }
        }
        
        // Check for missing parameters
        if !missing_params.is_empty() {
            log::warn!("Missing parameters: {:?}", missing_params);
        }
        
        // Check for unexpected parameters
        let unexpected: Vec<_> = state_dict.keys()
            .filter(|k| !loaded_params.contains(*k))
            .collect();
        if !unexpected.is_empty() {
            log::warn!("Unexpected parameters: {:?}", unexpected);
        }
        
        Ok(())
    }
    
    fn create_parameter_mapping(&self) -> HashMap<String, String> {
        let mut mapping = HashMap::new();
        
        // Define mapping between internal and external parameter names
        mapping.insert("embeddings.word_embeddings.weight".to_string(), 
                      "transformer.wte.weight".to_string());
        mapping.insert("embeddings.position_embeddings.weight".to_string(), 
                      "transformer.wpe.weight".to_string());
        
        for i in 0..self.config.num_hidden_layers {
            let layer_prefix = format!("transformer.layers.{}", i);
            let hf_prefix = format!("transformer.h.{}", i);
            
            mapping.insert(format!("{}.attention.q_proj.weight", layer_prefix),
                         format!("{}.attn.c_attn.weight", hf_prefix));
            mapping.insert(format!("{}.attention.k_proj.weight", layer_prefix),
                         format!("{}.attn.c_attn.weight", hf_prefix)); // Shared in HF
            mapping.insert(format!("{}.attention.v_proj.weight", layer_prefix),
                         format!("{}.attn.c_attn.weight", hf_prefix)); // Shared in HF
            
            // ... more mappings
        }
        
        mapping
    }
    
    // Handle special cases like tied embeddings
    fn handle_tied_weights(&mut self, state_dict: &HashMap<String, Tensor>) -> Result<()> {
        if self.config.tie_word_embeddings {
            if let Some(embed_weight) = state_dict.get("transformer.wte.weight") {
                // Use embedding weights for output layer
                if let Some(output_weight) = self.get_parameter_mut("lm_head.weight") {
                    output_weight.copy_(embed_weight)?;
                }
            }
        }
        Ok(())
    }
}
```

## Prevention Strategies

### General Prevention Guidelines

1. **Comprehensive Testing**
   - Test edge cases and boundary conditions
   - Use property-based testing for invariants
   - Test with various input sizes and shapes
   - Include performance regression tests

2. **Code Review Checklist**
   - Verify attention mask handling
   - Check dimension compatibility
   - Validate position encoding implementation
   - Ensure proper memory management
   - Confirm numerical stability

3. **Documentation Standards**
   - Document tensor shapes at each step
   - Explain coordinate systems and conventions
   - Provide usage examples
   - Document performance characteristics

4. **Debugging Tools**
   - Implement shape tracking utilities
   - Add gradient flow visualization
   - Create memory profiling helpers
   - Build attention pattern visualizers

Remember that many of these pitfalls can be avoided through careful design, comprehensive testing, and following established patterns. When in doubt, refer to well-tested reference implementations and adapt them to your specific needs.