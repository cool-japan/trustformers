# TrustformeRS Error Codes Reference

This document provides detailed information about error codes, their meanings, and recovery strategies.

## Error Code Format

All TrustformeRS errors follow the format `E####` where:
- E0001-E1000: Core errors (tensor operations, dimensions, memory)
- E1001-E2000: Model and configuration errors
- E2001-E3000: Training and optimization errors
- E3001-E4000: Export and deployment errors
- E9999: Generic/other errors

## Core Errors

### E0001: DimensionMismatch

**Description**: Tensor dimensions do not match expected values for an operation.

**Common Causes**:
- Incorrect input shape to a model
- Mismatched batch sizes
- Wrong sequence length
- Incompatible tensor operations

**Example**:
```
❌ Error [E0001]
────────────────────────────────────────────────────────────
📍 Dimension mismatch: expected [batch_size, 512, 768], got [batch_size, 256, 768]

📋 Context:
   Operation: MultiHeadAttention.forward
   Component: BERT
   layer: 12
   head_count: 12

💡 Suggestions:
   1. Check that input tensors have shape [batch_size, 512, 768], not [batch_size, 256, 768]
   2. Verify the model configuration matches your input dimensions
   3. Use .view() or .reshape() to adjust tensor dimensions

📚 For more information, see: https://docs.trustformers.ai/errors/E0001
────────────────────────────────────────────────────────────
```

**Recovery Strategies**:
1. **Padding**: Use padding to match expected sequence length
   ```rust
   let padded = tensor.pad(&[(0, 0), (0, 256), (0, 0)], 0.0)?;
   ```

2. **Truncation**: Truncate sequences to maximum length
   ```rust
   let truncated = tensor.slice(1, 0, 512)?;
   ```

3. **Dynamic shapes**: Use models that support variable sequence lengths

### E0002: OutOfMemory

**Description**: Insufficient memory to complete the requested operation.

**Common Causes**:
- Batch size too large
- Model too large for available GPU memory
- Memory fragmentation
- Memory leaks from previous operations

**Recovery Strategies**:
1. **Reduce batch size**:
   ```rust
   let config = GenerationConfig {
       batch_size: 16, // Reduced from 32
       ..Default::default()
   };
   ```

2. **Enable gradient checkpointing**:
   ```rust
   model.enable_gradient_checkpointing();
   ```

3. **Use mixed precision**:
   ```rust
   let amp_config = AMPConfig {
       enabled: true,
       precision: Precision::FP16,
       ..Default::default()
   };
   ```

4. **Model parallelism**: Split model across multiple devices

### E0003: InvalidConfiguration

**Description**: Model or component configuration contains invalid values.

**Common Causes**:
- Incompatible parameter combinations
- Out-of-range values
- Missing required fields
- Type mismatches

**Example Configuration Issues**:
- `num_attention_heads` not divisible by `num_key_value_heads`
- `hidden_size` not divisible by `num_attention_heads`
- Negative or zero values for sizes
- Vocabulary size mismatch with embeddings

**Recovery Strategies**:
1. **Use validated configs**:
   ```rust
   let config = BertConfig::from_pretrained("bert-base-uncased")?;
   ```

2. **Validate before creation**:
   ```rust
   config.validate()?;
   let model = BertModel::new(&config)?;
   ```

### E0004: ModelNotFound

**Description**: Requested model cannot be found in the specified location.

**Common Causes**:
- Typo in model name
- Model is private/gated
- No internet connection
- Incorrect revision/branch

**Recovery Strategies**:
1. **Check available models**:
   ```rust
   let models = Model::list_available()?;
   ```

2. **Authenticate for private models**:
   ```bash
   huggingface-cli login
   ```

3. **Use local cache**:
   ```rust
   let model = Model::from_pretrained(
       "model-name",
       Some(&LoadOptions {
           local_files_only: true,
           ..Default::default()
       })
   )?;
   ```

## Model and Configuration Errors

### E0005: WeightLoadingError

**Description**: Failed to load model weights from file.

**Common Causes**:
- Corrupted weight file
- Incompatible format
- Version mismatch
- Missing weight keys

**Recovery Strategies**:
1. **Re-download weights**
2. **Check format compatibility**
3. **Use weight converter tools**

### E0006: TokenizationError

**Description**: Failed to tokenize input text.

**Common Causes**:
- Invalid Unicode sequences
- Tokenizer vocabulary mismatch
- Special tokens not handled
- Maximum length exceeded

**Recovery Strategies**:
1. **Clean input text**
2. **Use appropriate tokenizer**
3. **Handle special tokens explicitly**

### E0007: QuantizationError

**Description**: Failed to quantize model or tensors.

**Common Causes**:
- Unsupported layer types
- Calibration data issues
- Precision loss too high
- Hardware incompatibility

**Recovery Strategies**:
1. **Exclude problematic layers**:
   ```rust
   let config = QuantizationConfig {
       exclude_layers: vec!["embed_tokens", "lm_head"],
       ..Default::default()
   };
   ```

2. **Use different quantization method**
3. **Provide calibration data**

## Best Practices for Error Handling

### 1. Use the ResultExt trait
```rust
use trustformers_core::errors::ResultExt;

fn process() -> Result<(), TrustformersError> {
    load_model()?
        .with_operation("process")
        .with_component("Pipeline")?;
    Ok(())
}
```

### 2. Add context progressively
```rust
let result = compute_attention(query, key, value)
    .with_context("layer", layer_idx.to_string())
    .with_context("head", head_idx.to_string())?;
```

### 3. Provide helpful suggestions
```rust
if tokens.len() > max_length {
    return Err(invalid_config("sequence_length", "exceeds maximum")
        .with_context("length", tokens.len().to_string())
        .with_context("maximum", max_length.to_string())
        .with_suggestion(format!("Truncate to {} tokens", max_length))
        .with_suggestion("Use a model with longer context window"));
}
```

### 4. Use error helpers
```rust
// Instead of manually creating errors
use trustformers_core::errors::{dimension_mismatch, out_of_memory};

// Use helpers for common errors
return Err(dimension_mismatch(expected_shape, actual_shape));
```

## Error Monitoring and Debugging

### Enable detailed error logging
```rust
std::env::set_var("TRUSTFORMERS_ERROR_DETAIL", "full");
```

### Error metrics collection
```rust
let metrics = ErrorMetrics::new();
metrics.record_error(&error);
println!("Most common errors: {:?}", metrics.top_errors(5));
```

### Debug mode for tensor operations
```rust
std::env::set_var("TRUSTFORMERS_DEBUG_TENSORS", "1");
```

This will add tensor statistics to error messages when numerical issues occur.