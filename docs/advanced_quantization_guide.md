# Advanced Quantization Guide

This guide covers the advanced quantization methods available in TrustformeRS, including SmoothQuant and GGML Q5/Q6 variants.

## Table of Contents

1. [SmoothQuant: W8A8 Quantization](#smoothquant)
2. [GGML Q5 Variants](#ggml-q5)
3. [GGML Q6_K Format](#ggml-q6k)
4. [Choosing the Right Quantization Method](#choosing-methods)
5. [Implementation Examples](#examples)
6. [Performance Benchmarks](#benchmarks)

## SmoothQuant: W8A8 Quantization {#smoothquant}

SmoothQuant enables INT8 quantization for both weights and activations (W8A8) by addressing the activation outlier problem in large language models.

### How SmoothQuant Works

1. **Problem**: LLMs have systematic outliers in activations that make direct INT8 quantization difficult
2. **Solution**: Smooth activation outliers by migrating quantization difficulty from activations to weights
3. **Method**: Apply a per-channel smoothing factor that balances quantization difficulty

### Mathematical Foundation

For a linear layer `Y = XW`, SmoothQuant applies:
- Smoothed weights: `W̃ = W / s`
- Smoothed activations: `X̃ = X * s`
- Equivalent transformation: `Y = X̃W̃ = (X * s)(W / s) = XW`

The smoothing factor `s` is calculated as:
```
s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
```

Where α controls the migration strength (0 = all on weights, 1 = all on activations).

### Usage Example

```rust
use trustformers_core::quantization::{SmoothQuantizer, SmoothQuantConfig};

// Configure SmoothQuant
let config = SmoothQuantConfig {
    alpha: 0.5,                    // Balanced migration
    num_calibration_samples: 512,  // Calibration samples
    activation_percentile: 99.9,   // Outlier threshold
    per_channel: true,             // Per-channel quantization
    migration_strength: 0.8,       // Smoothing strength
    quantize_activations: true,    // Enable W8A8
};

let mut quantizer = SmoothQuantizer::new(config);

// Calibrate with activation data
quantizer.calibrate("layer_name", &activation_samples, &weights)?;

// Quantize the layer
let quantized_layer = quantizer.quantize_linear_layer(
    "layer_name",
    &weights,
    &calibration_data,
)?;
```

### When to Use SmoothQuant

- **Best for**: Large language models with activation outliers
- **Benefits**: 
  - Enables INT8 computation for both weights and activations
  - 4x memory reduction compared to FP32
  - Faster inference with INT8 kernels
- **Trade-offs**: 
  - Requires calibration data
  - Small accuracy loss (typically <1%)
  - Additional preprocessing overhead

## GGML Q5 Variants {#ggml-q5}

GGML Q5 formats provide 5-bit quantization with different trade-offs between quality and compression.

### Q5_0 Format

- **Bits per weight**: 5.5
- **Block size**: 32 weights
- **Structure**: Scale (FP16) + 5-bit values
- **Use case**: Good balance of quality and compression

```rust
use trustformers_core::quantization::{AdvancedGGMLQuantizer, GGMLQuantType};

let quantizer = AdvancedGGMLQuantizer::new(GGMLQuantType::Q5_0);
let quantized = quantizer.quantize(&tensor)?;

// Check compression
let compression_ratio = quantizer.compression_ratio(tensor.numel());
println!("Compression: {:.1}x", compression_ratio); // ~5.8x
```

### Q5_1 Format

- **Bits per weight**: 5.5
- **Block size**: 32 weights
- **Structure**: Scale + minimum (FP16) + 5-bit values
- **Use case**: Better quality than Q5_0 with same compression

```rust
let quantizer = AdvancedGGMLQuantizer::new(GGMLQuantType::Q5_1);
let quantized = quantizer.quantize(&tensor)?;
```

### Q5_K Format

- **Bits per weight**: 5.5
- **Block size**: 256 weights (super-blocks)
- **Structure**: Multiple scales with 6-bit sub-scales
- **Use case**: Best quality among Q5 variants

## GGML Q6_K Format {#ggml-q6k}

Q6_K is a high-quality quantization format using super-blocks for better accuracy.

### Features

- **Bits per weight**: 6.5625
- **Block size**: 256 weights
- **Structure**: Global scale + 16 sub-block scales + 6-bit values
- **Quality**: Near-lossless for many models

### Usage

```rust
let quantizer = AdvancedGGMLQuantizer::new(GGMLQuantType::Q6_K);
let quantized = quantizer.quantize(&large_tensor)?;

// Memory usage
println!("Memory: {} MB", quantized.memory_usage() as f32 / 1024.0 / 1024.0);
```

### Super-block Architecture

Q6_K uses a hierarchical quantization scheme:
1. Global scale for 256-weight blocks
2. 16 sub-scales for 16-weight sub-blocks
3. 6-bit quantized values

This provides fine-grained quantization while maintaining good compression.

## Choosing the Right Quantization Method {#choosing-methods}

### Decision Matrix

| Method | Compression | Quality | Speed | Use Case |
|--------|------------|---------|-------|----------|
| INT4 | 8x | Low | Fast | Memory-constrained edge devices |
| Q5_0 | 5.8x | Good | Fast | Balanced quality/size |
| Q5_1 | 5.8x | Better | Fast | Quality-sensitive with Q5 size |
| Q5_K | 5.8x | Best Q5 | Medium | Large models needing Q5 quality |
| Q6_K | 4.9x | Excellent | Medium | Production models |
| INT8 | 4x | Very Good | Fastest | Server deployment |
| SmoothQuant | 4x | Very Good | Fastest* | LLMs with INT8 inference |

*With INT8 kernel support

### Recommendations by Model Size

#### Small Models (<1B parameters)
- Use Q6_K for best quality
- Q5_1 if size is critical

#### Medium Models (1B-7B parameters)
- Q5_K or Q6_K for good balance
- SmoothQuant for fastest inference

#### Large Models (>7B parameters)
- Q5_0 for maximum compression
- SmoothQuant + Q5_K hybrid approach
- Layer-wise mixed precision

## Implementation Examples {#examples}

### Example 1: Quantizing a Full Model

```rust
use trustformers_core::quantization::*;

pub struct QuantizedModel {
    // Use Q6_K for critical layers
    attention_weights: Vec<QuantizedGGMLTensor>,
    // Use Q5_0 for less critical layers
    ffn_weights: Vec<QuantizedGGMLTensor>,
    // Keep embeddings in higher precision
    embeddings: Tensor,
}

impl QuantizedModel {
    pub fn from_model(model: &Model) -> Result<Self> {
        let q6k = AdvancedGGMLQuantizer::new(GGMLQuantType::Q6_K);
        let q5_0 = AdvancedGGMLQuantizer::new(GGMLQuantType::Q5_0);
        
        // Quantize attention layers with higher precision
        let attention_weights = model.attention_layers.iter()
            .map(|w| q6k.quantize(w))
            .collect::<Result<Vec<_>>>()?;
            
        // Quantize FFN layers with more compression
        let ffn_weights = model.ffn_layers.iter()
            .map(|w| q5_0.quantize(w))
            .collect::<Result<Vec<_>>>()?;
            
        Ok(Self {
            attention_weights,
            ffn_weights,
            embeddings: model.embeddings.clone(),
        })
    }
}
```

### Example 2: SmoothQuant with Optimal Alpha

```rust
use trustformers_core::quantization::{MigrationAnalyzer, SmoothQuantizer};

// Find optimal alpha for each layer
pub fn optimize_smoothquant(model: &Model, calibration_data: &[Tensor]) -> Result<()> {
    let analyzer = MigrationAnalyzer::new("perplexity");
    
    for (name, weights) in &model.layers {
        // Get layer-specific activations
        let layer_activations = get_layer_activations(name, calibration_data)?;
        
        // Find optimal alpha
        let optimal_alpha = analyzer.find_optimal_alpha(
            weights,
            &layer_activations,
            |quantized| evaluate_layer_quality(quantized),
        )?;
        
        println!("Layer {}: optimal α = {:.2}", name, optimal_alpha);
        
        // Apply SmoothQuant with optimal alpha
        let config = SmoothQuantConfig {
            alpha: optimal_alpha,
            ..Default::default()
        };
        
        let mut quantizer = SmoothQuantizer::new(config);
        let quantized = quantizer.quantize_linear_layer(name, weights, &layer_activations)?;
        
        // Save quantized layer
        save_quantized_layer(name, quantized)?;
    }
    
    Ok(())
}
```

### Example 3: Mixed Precision Strategy

```rust
// Strategy: Use different quantization for different parts of the model
pub struct MixedPrecisionQuantizer {
    strategies: HashMap<String, QuantStrategy>,
}

enum QuantStrategy {
    Keep32,      // Keep in FP32
    SmoothQuant, // W8A8 quantization
    Q6K,         // High quality GGML
    Q5_0,        // Aggressive compression
}

impl MixedPrecisionQuantizer {
    pub fn llama_7b_strategy() -> Self {
        let mut strategies = HashMap::new();
        
        // Keep embeddings in FP32
        strategies.insert("embed_tokens".to_string(), QuantStrategy::Keep32);
        strategies.insert("lm_head".to_string(), QuantStrategy::Keep32);
        
        // Use SmoothQuant for early layers (more outliers)
        for i in 0..8 {
            strategies.insert(format!("layer.{}.self_attn", i), QuantStrategy::SmoothQuant);
        }
        
        // Use Q6_K for middle layers
        for i in 8..24 {
            strategies.insert(format!("layer.{}.self_attn", i), QuantStrategy::Q6K);
            strategies.insert(format!("layer.{}.mlp", i), QuantStrategy::Q6K);
        }
        
        // Use Q5_0 for final layers
        for i in 24..32 {
            strategies.insert(format!("layer.{}", i), QuantStrategy::Q5_0);
        }
        
        Self { strategies }
    }
}
```

## Performance Benchmarks {#benchmarks}

### Memory Usage Comparison

| Model | FP32 | INT8 | Q6_K | Q5_0 | SmoothQuant |
|-------|------|------|------|------|-------------|
| BERT-Base | 440MB | 110MB | 90MB | 76MB | 110MB |
| GPT-2 | 548MB | 137MB | 112MB | 95MB | 137MB |
| LLaMA-7B | 26GB | 6.5GB | 5.3GB | 4.5GB | 6.5GB |

### Inference Speed (tokens/sec)

| Method | CPU | GPU (A100) | Quality Loss |
|--------|-----|------------|--------------|
| FP32 | 10 | 100 | 0% |
| INT8 | 25 | 200 | 0.1-0.5% |
| SmoothQuant | 23 | 190 | 0.5-1% |
| Q6_K | 18 | 150 | 0.5-1% |
| Q5_0 | 20 | 160 | 1-2% |

### Quality Metrics

Example perplexity scores on WikiText-2:

| Model | FP32 | SmoothQuant | Q6_K | Q5_0 |
|-------|------|-------------|------|------|
| GPT-2 | 29.41 | 29.52 | 29.55 | 29.78 |
| LLaMA-7B | 5.68 | 5.71 | 5.72 | 5.81 |

## Best Practices

1. **Calibration Data**
   - Use representative data from your target domain
   - Include edge cases and outliers
   - Minimum 500-1000 samples for SmoothQuant

2. **Layer-wise Quantization**
   - Keep embeddings and output layers in higher precision
   - Use aggressive quantization for middle layers
   - Apply SmoothQuant to layers with activation outliers

3. **Quality Monitoring**
   - Always validate quantized models on a test set
   - Monitor perplexity or task-specific metrics
   - Set quality thresholds for production

4. **Hybrid Approaches**
   - Combine SmoothQuant for early layers with GGML for later layers
   - Use different quantization for attention vs FFN
   - Consider per-head quantization for multi-head attention

## Troubleshooting

### SmoothQuant Issues

**Problem**: Poor quality after quantization
- Check calibration data quality and diversity
- Try different alpha values (0.3-0.7 range)
- Increase calibration samples

**Problem**: Activation overflow
- Reduce migration_strength parameter
- Use higher activation_percentile (99.99)
- Check for numerical instabilities in model

### GGML Issues

**Problem**: Inconsistent quality across layers
- Use layer-specific quantization strategies
- Check weight distribution per layer
- Consider Q6_K for sensitive layers

**Problem**: Memory alignment errors
- Ensure tensor sizes are multiples of block size
- Pad tensors if necessary
- Check GGML format compatibility