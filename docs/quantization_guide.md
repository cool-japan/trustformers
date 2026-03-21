# TrustformeRS Quantization Guide

This guide covers the quantization capabilities in TrustformeRS, including BitsAndBytes compatibility, GPTQ, AWQ, and quantization-aware training.

## Table of Contents

1. [Overview](#overview)
2. [Quantization Methods](#quantization-methods)
3. [BitsAndBytes Compatibility](#bitsandbytes-compatibility)
4. [GPTQ Quantization](#gptq-quantization)
5. [AWQ Quantization](#awq-quantization)
6. [Quantization-Aware Training](#quantization-aware-training)
7. [Performance Considerations](#performance-considerations)
8. [Migration from Python](#migration-from-python)

## Overview

Quantization reduces model size and improves inference speed by representing weights and activations with lower precision. TrustformeRS supports multiple quantization methods:

- **INT8/INT4**: Standard integer quantization
- **BitsAndBytes**: Compatible with the popular bitsandbytes library
- **GPTQ**: Gradient-based post-training quantization
- **AWQ**: Activation-aware weight quantization
- **Dynamic**: Runtime quantization
- **QAT**: Quantization-aware training

## Quantization Methods

### Basic Quantization

```rust
use trustformers_core::{Tensor, QuantizationConfig, QuantizationScheme, Quantizer};

// Configure INT8 quantization
let config = QuantizationConfig {
    scheme: QuantizationScheme::Int8,
    symmetric: true,
    per_channel: false,
    calibration_samples: Some(128),
    group_size: None,
    bnb_config: None,
};

// Quantize a tensor
let tensor = Tensor::randn(&[256, 256])?;
let quantized = Quantizer::quantize(&tensor, &config)?;

// Dequantize back to float
let dequantized = quantized.dequantize()?;
```

### Per-Channel Quantization

```rust
// Per-channel quantization for better accuracy
let config = QuantizationConfig {
    scheme: QuantizationScheme::Int8,
    symmetric: false,
    per_channel: true,  // Quantize each channel separately
    ..Default::default()
};

let quantized = Quantizer::quantize(&tensor, &config)?;
```

### INT4 Quantization

```rust
// 4-bit quantization for extreme compression
let config = QuantizationConfig {
    scheme: QuantizationScheme::Int4,
    symmetric: false,
    per_channel: true,
    group_size: Some(128),  // Group-wise quantization
    ..Default::default()
};

let quantized = Quantizer::quantize(&tensor, &config)?;
```

## BitsAndBytes Compatibility

TrustformeRS provides full compatibility with the bitsandbytes library, supporting LLM.int8() and NF4/FP4 quantization.

### INT8 Quantization (LLM.int8())

```rust
use trustformers_core::{BitsAndBytesConfig, quantize_int8, dequantize_bitsandbytes};

// Configure BitsAndBytes INT8 quantization
let config = BitsAndBytesConfig {
    bits: 8,
    dynamic_tree: false,
    block_size: 256,
    stochastic: false,
    outlier_threshold: 0.99,  // Detect outliers
    nested_quantization: false,
};

// Quantize tensor
let tensor = Tensor::randn(&[1024, 1024])?;
let quantized = quantize_int8(&tensor, &config)?;

// Access quantization components
println!("Quantized data shape: {:?}", quantized.data.shape());
println!("Scale factors: {}", quantized.scale.shape()[0]);
println!("Outliers detected: {:?}", quantized.outliers.as_ref().map(|o| o.len()));

// Dequantize
let dequantized = dequantize_bitsandbytes(&quantized, &config)?;
```

### 4-bit Quantization (NF4)

```rust
use trustformers_core::{quantize_4bit};

// Configure NF4 quantization
let config = BitsAndBytesConfig {
    bits: 4,
    block_size: 64,  // Smaller blocks for 4-bit
    nested_quantization: true,  // Quantize scales too
    ..Default::default()
};

// Quantize with NF4 (Normal Float 4-bit)
let quantized = quantize_4bit(&tensor, &config)?;

// NF4 uses predefined quantization levels based on normal distribution
// This provides better accuracy for normally distributed weights
```

### Dynamic Tree Quantization

```rust
use trustformers_core::{quantize_dynamic_tree};

// Configure dynamic tree quantization
let config = BitsAndBytesConfig {
    bits: 8,
    dynamic_tree: true,  // Enable tree quantization
    ..Default::default()
};

// Build quantization tree based on data distribution
let quantized = quantize_dynamic_tree(&tensor, &config)?;
```

### Format Conversion

Convert between TrustformeRS and bitsandbytes formats:

```rust
use trustformers_core::{to_bitsandbytes_format, from_bitsandbytes_format};

// Convert to bitsandbytes format
let bnb_format = to_bitsandbytes_format(&tensor, &config)?;

// Components in bitsandbytes format
let data = &bnb_format["data"];
let scale = &bnb_format["scale"];
let zero_point = bnb_format.get("zero_point");
let outliers = bnb_format.get("outliers");

// Convert back from bitsandbytes format
let reconstructed = from_bitsandbytes_format(bnb_format, &config)?;
```

## GPTQ Quantization

GPTQ uses gradient information to minimize quantization error:

```rust
use trustformers_core::{GPTQQuantizer};

let config = QuantizationConfig {
    scheme: QuantizationScheme::GPTQ,
    symmetric: false,
    per_channel: true,
    group_size: Some(128),
    ..Default::default()
};

let gptq = GPTQQuantizer::new(config);

// Full GPTQ requires Hessian matrix (computed during calibration)
let hessian = None; // Would be computed from calibration data
let quantized = gptq.quantize(&tensor, hessian)?;
```

## AWQ Quantization

AWQ considers activation magnitudes for better weight quantization:

```rust
use trustformers_core::{AWQQuantizer};

let config = QuantizationConfig {
    scheme: QuantizationScheme::AWQ,
    symmetric: false,
    per_channel: true,
    ..Default::default()
};

let mut awq = AWQQuantizer::new(config);

// Set activation scales from calibration
let activation_scales = vec![1.2; num_channels];
awq.set_activation_scales(activation_scales);

let quantized = awq.quantize(&tensor)?;
```

## Quantization-Aware Training

Quantization-Aware Training (QAT) simulates quantization during training to produce models that maintain high accuracy after quantization. TrustformeRS provides comprehensive QAT support.

### Basic QAT Setup

```rust
use trustformers_training::{QATConfig, QATModel, QATTrainer};
use trustformers_core::QuantizationScheme;

// Configure QAT
let qat_config = QATConfig {
    qscheme: QuantizationScheme::Int8,
    bits: 8,
    symmetric: true,
    per_channel: false,
    start_step: 1000,        // Start QAT after warmup
    freeze_step: Some(5000), // Freeze quantization params
    learnable_step_size: false,
    observer_momentum: 0.99,
};

// Prepare model for QAT
let mut qat_model = QATModel::new(original_model, qat_config);
qat_model.prepare()?;
```

### QAT Layers

Replace standard layers with QAT versions:

```rust
use trustformers_training::{QATLinear, QATConv2d};

// QAT Linear layer
let linear = QATLinear::new(
    original_linear,
    qat_config.clone()
);

// QAT Convolution with activation quantization
let conv = QATConv2d::new(
    original_conv,
    qat_config.clone(),
    true  // Also quantize activations
);
```

### Calibration

Initialize quantization parameters using representative data:

```rust
use trustformers_training::CalibrationDataset;

// Create calibration dataset
let calibration_dataset = CalibrationDataset::new(
    calibration_samples,
    calibration_labels
);

// Run calibration
calibration_dataset.calibrate(&mut qat_model)?;
```

### Training with QAT

```rust
use trustformers_training::{Trainer, QATTrainer, qat_loss};

// Create QAT trainer for quantization parameters
let qat_trainer = QATTrainer::new(
    1e-4,  // Learning rate for quantization params
    0.0,   // Weight decay
);

// Custom loss function with quantization penalty
let loss = qat_loss(
    &predictions,
    &targets,
    quantization_error,
    0.01,  // Alpha: weight for quant error
)?;

// Train model
let trainer = Trainer::new(
    qat_model,
    training_args,
    train_dataset,
    optimizer,
);

trainer.train()?;
```

### Fake Quantization

The core of QAT is fake quantization - simulating quantization during forward pass while maintaining differentiability:

```rust
use trustformers_training::fake_quantize;

// Apply fake quantization
let fake_quantized = fake_quantize(
    &tensor,
    &scale,
    zero_point.as_ref(),
    bits,
    symmetric
)?;

// Gradients pass through unchanged (straight-through estimator)
```

### Quantization Parameters

Track and update quantization parameters during training:

```rust
use trustformers_training::{QuantizationParams, QuantizationGradients};

// Initialize parameters
let mut quant_params = QuantizationParams::new(&shape, symmetric);

// Update statistics during forward pass
quant_params.update_stats(&tensor, momentum)?;

// Compute scale and zero point
quant_params.compute_params(bits, symmetric)?;

// Update with gradients
let gradients = QuantizationGradients {
    scale_grad: Some(scale_gradient),
    zero_point_grad: zero_point_gradient,
};

qat_trainer.update_quant_params(&mut quant_params, &gradients)?;
```

### Converting to Quantized Model

After QAT training, convert to a fully quantized model:

```rust
// Get quantization statistics
let stats = qat_model.get_statistics();
for (layer, stat) in stats {
    println!("{}: scale={:.4}, range=[{:.2}, {:.2}]",
             layer, stat.scale, stat.min_val, stat.max_val);
}

// Convert to quantized model
let quantized_model = qat_model.convert()?;

// Model is now ready for INT8 inference
```

### Advanced QAT Techniques

#### Per-Channel Quantization

```rust
let config = QATConfig {
    per_channel: true,  // Separate scale per output channel
    ..Default::default()
};
```

#### Learnable Quantization

```rust
let config = QATConfig {
    learnable_step_size: true,  // Learn optimal scale
    ..Default::default()
};

// Scale becomes a learnable parameter
```

#### Progressive Quantization

Start with higher precision and gradually reduce:

```rust
// Start with INT16
let mut config = QATConfig {
    bits: 16,
    ..Default::default()
};

// After N epochs, switch to INT8
if epoch > 5 {
    config.bits = 8;
    qat_model.update_config(config);
}
```

### QAT Best Practices

#### 1. Proper Initialization

```rust
// Calibrate on representative data
let calibration_size = 1000;
let calibration_data = train_data.take(calibration_size);
calibration_dataset.calibrate(&mut qat_model)?;
```

#### 2. Gradual Introduction

```rust
let config = QATConfig {
    start_step: 1000,  // Let model stabilize first
    ..Default::default()
};
```

#### 3. Monitor Quantization Error

```rust
// Track quantization impact
let quant_error = compute_quantization_error(&original, &quantized)?;
if quant_error > threshold {
    // Adjust QAT parameters
}
```

#### 4. Separate Learning Rates

```rust
// Different learning rates for weights and quantization params
let weight_lr = 1e-3;
let quant_lr = 1e-4;  // Usually lower
```

### QAT for Different Architectures

#### Transformer Models

```rust
// QAT for attention layers
let qat_attention = QATMultiHeadAttention::new(
    attention_layer,
    QATConfig {
        per_channel: true,  // Important for attention
        ..Default::default()
    }
);
```

#### Convolutional Networks

```rust
// QAT for CNN layers
let qat_conv = QATConv2d::new(
    conv_layer,
    QATConfig {
        bits: 8,
        symmetric: false,  // Asymmetric often better for CNNs
        ..Default::default()
    },
    true  // Quantize activations
);
```

### Debugging QAT

Enable detailed logging:

```rust
std::env::set_var("TRUSTFORMERS_QAT_DEBUG", "1");

// Monitor quantization statistics
qat_model.set_debug_callback(|layer, stats| {
    println!("{}: scale={:.6}, zero={:.2}", 
             layer, stats.scale, stats.zero_point);
});
```

### Observer Pattern

The Observer pattern collects statistics for quantization:

```rust
use trustformers_core::Observer;

// Create observer
let mut observer = Observer::new();

// Collect statistics during calibration
for batch in calibration_data {
    observer.update(&batch);
}

// Compute quantization parameters
let (scale, zero_point) = observer.get_quantization_params(
    symmetric,
    bits
)?;
```

### QAT Performance Tips

1. **Batch Normalization Folding**
   ```rust
   // Fold BN into conv before QAT
   let folded_conv = fold_bn_into_conv(conv, bn)?;
   let qat_conv = QATConv2d::new(folded_conv, config);
   ```

2. **Mixed Precision QAT**
   ```rust
   // Different precision for different layers
   let backbone_config = QATConfig { bits: 8, ..Default::default() };
   let head_config = QATConfig { bits: 16, ..Default::default() };
   ```

3. **Gradient Clipping**
   ```rust
   // Prevent gradient explosion in quantization params
   trainer.set_gradient_clip_value(1.0);
   ```

## Performance Considerations

### Memory Efficiency

Different quantization schemes have different memory footprints:

| Method | Bits/Weight | Compression | Accuracy |
|--------|-------------|-------------|----------|
| FP32 | 32 | 1x | Baseline |
| INT8 | 8 | 4x | ~99% |
| INT4 | 4 | 8x | ~98% |
| NF4 | 4 | 8x | ~98.5% |

### Speed Optimization

```rust
// Block-wise quantization for cache efficiency
let config = BitsAndBytesConfig {
    block_size: 256,  // Tune based on cache size
    ..Default::default()
};

// Use per-tensor quantization for simpler operations
let config = QuantizationConfig {
    per_channel: false,  // Faster than per-channel
    ..Default::default()
};
```

### Outlier Handling

```rust
// Configure outlier detection
let config = BitsAndBytesConfig {
    outlier_threshold: 0.99,  // Keep top 1% as FP16
    ..Default::default()
};

// Mixed precision for outliers
let quantized = quantize_int8(&tensor, &config)?;
if let Some(outliers) = &quantized.outliers {
    println!("Outliers: {} values kept at full precision", outliers.len());
}
```

## Migration from Python

### From bitsandbytes

Python:
```python
import bitsandbytes as bnb

# INT8 quantization
linear_int8 = bnb.nn.Linear8bitLt(in_features, out_features)

# 4-bit quantization
linear_4bit = bnb.nn.Linear4bit(
    in_features, 
    out_features,
    compress_statistics=True,
    quant_type='nf4'
)
```

Rust:
```rust
use trustformers_core::*;

// INT8 quantization
let config = BitsAndBytesConfig {
    bits: 8,
    ..Default::default()
};

// 4-bit quantization
let config = BitsAndBytesConfig {
    bits: 4,
    nested_quantization: true,  // compress_statistics
    ..Default::default()
};
```

### From PyTorch

Python:
```python
import torch.quantization as quant

# Dynamic quantization
model_int8 = quant.quantize_dynamic(
    model, 
    {torch.nn.Linear}, 
    dtype=torch.qint8
)

# Static quantization
model.qconfig = quant.get_default_qconfig('fbgemm')
quant.prepare(model, inplace=True)
# ... calibration ...
quant.convert(model, inplace=True)
```

Rust:
```rust
// Dynamic quantization
let config = QuantizationConfig {
    scheme: QuantizationScheme::Dynamic,
    ..Default::default()
};

// Static quantization with calibration
let samples = load_calibration_data()?;
let calibrated_config = Quantizer::calibrate(&samples, &config)?;
let quantized = Quantizer::quantize(&tensor, &calibrated_config)?;
```

## Best Practices

### 1. Choose the Right Method

- **INT8**: Best balance of speed and accuracy
- **INT4/NF4**: Maximum compression for large models
- **Dynamic**: When weights vary significantly during inference
- **GPTQ/AWQ**: For minimal accuracy loss

### 2. Calibration is Key

```rust
// Collect representative samples
let calibration_samples = vec![
    sample1, sample2, sample3, // ...
];

// Calibrate quantization parameters
let config = QuantizationConfig {
    calibration_samples: Some(calibration_samples.len()),
    ..Default::default()
};

let calibrated = Quantizer::calibrate(&calibration_samples, &config)?;
```

### 3. Profile Before and After

```rust
use std::time::Instant;

// Measure quantization impact
let start = Instant::now();
let output_float = model.forward(&input)?;
let float_time = start.elapsed();

let start = Instant::now();
let output_quant = quantized_model.forward(&input)?;
let quant_time = start.elapsed();

println!("Speedup: {:.2}x", float_time.as_secs_f32() / quant_time.as_secs_f32());
```

### 4. Validate Accuracy

```rust
// Check quantization error
let error = original.sub(&dequantized)?.abs()?.mean()?;
let snr = calculate_snr(&original, &dequantized)?;

assert!(error < threshold, "Quantization error too high");
assert!(snr > min_snr, "Signal-to-noise ratio too low");
```

## Advanced Topics

### Custom Quantization Schemes

```rust
// Implement custom quantization
pub struct MyCustomQuantizer {
    // Custom parameters
}

impl MyCustomQuantizer {
    pub fn quantize(&self, tensor: &Tensor) -> Result<QuantizedTensor> {
        // Custom quantization logic
    }
}
```

### Hardware-Specific Optimization

```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Use AVX2 for INT8 operations
unsafe fn quantize_avx2(data: &[f32], scale: f32) -> Vec<u8> {
    // SIMD quantization
}
```

### Quantization Debugging

```rust
// Enable quantization debugging
std::env::set_var("TRUSTFORMERS_QUANTIZATION_DEBUG", "1");

// Log quantization statistics
let stats = QuantizationStats::from_tensor(&tensor);
println!("Quantization stats: {:?}", stats);
```

## Conclusion

TrustformeRS provides comprehensive quantization support with:
- Full BitsAndBytes compatibility
- Multiple quantization schemes
- Flexible configuration options
- High-performance implementations
- Easy migration from Python

Choose the appropriate quantization method based on your model size, accuracy requirements, and deployment constraints.