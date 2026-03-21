# On-Device Fine-Tuning Guide

This guide covers TrustformeRS's comprehensive on-device fine-tuning capabilities, including federated learning, differential privacy, and advanced parameter-efficient training methods.

## Table of Contents

1. [Overview](#overview)
2. [Basic On-Device Training](#basic-on-device-training)
3. [Advanced Training Methods](#advanced-training-methods)
4. [Federated Learning](#federated-learning)
5. [Differential Privacy](#differential-privacy)
6. [Best Practices](#best-practices)
7. [Performance Optimization](#performance-optimization)
8. [Examples](#examples)

## Overview

TrustformeRS provides state-of-the-art on-device fine-tuning capabilities optimized for mobile devices:

- **Memory-Efficient Methods**: LoRA, QLoRA, BitFit, and more
- **Privacy-Preserving**: Differential privacy and federated learning
- **Mobile-Optimized**: Adaptive to device capabilities
- **Production-Ready**: Checkpoint management and error recovery

### Key Features

- **Parameter-Efficient Fine-Tuning**: Train models with minimal memory overhead
- **Federated Learning**: Collaborate without sharing raw data
- **Differential Privacy**: Mathematically proven privacy guarantees
- **Advanced Methods**: QLoRA, P-tuning, UniPELT, and more
- **Hardware Adaptation**: Automatic optimization for device capabilities

## Basic On-Device Training

### Quick Start

```rust
use trustformers_mobile::training::{
    OnDeviceTrainer, OnDeviceTrainingConfig, FineTuningMethod
};

// Configure training
let config = OnDeviceTrainingConfig {
    learning_rate: 1e-4,
    epochs: 3,
    batch_size: 1,
    gradient_accumulation_steps: 8,
    max_sequence_length: 128,
    gradient_checkpointing: true,
    method: FineTuningMethod::LoRA { rank: 8, alpha: 16.0 },
    memory_optimization: MemoryOptimization::Balanced,
    max_training_memory_mb: 512,
};

// Create trainer
let mut trainer = OnDeviceTrainer::new(config, mobile_config)?;

// Initialize with base model
trainer.initialize_training(base_model_params)?;

// Train on local data
let stats = trainer.train(&training_data)?;
```

### Fine-Tuning Methods

#### LoRA (Low-Rank Adaptation)
```rust
FineTuningMethod::LoRA { 
    rank: 8,        // Rank of adaptation matrices
    alpha: 16.0,    // Scaling factor
}
```
- **Memory**: ~0.1% of full model
- **Performance**: Near full fine-tuning
- **Best for**: General adaptation

#### Adapter Layers
```rust
FineTuningMethod::Adapter { 
    bottleneck_size: 64  // Hidden dimension
}
```
- **Memory**: ~1-2% of full model
- **Performance**: Good for task-specific adaptation
- **Best for**: Multi-task learning

#### Prefix Tuning
```rust
FineTuningMethod::PrefixTuning { 
    prefix_length: 20  // Number of virtual tokens
}
```
- **Memory**: Minimal (only prefix embeddings)
- **Performance**: Good for generation tasks
- **Best for**: Prompt-based tasks

### Memory Optimization

Configure memory optimization based on device:

```rust
// Ultra-low memory devices (<1GB)
let config = OnDeviceTrainingConfig {
    method: FineTuningMethod::LoRA { rank: 4, alpha: 8.0 },
    memory_optimization: MemoryOptimization::Maximum,
    max_training_memory_mb: 256,
    gradient_checkpointing: true,
    ..Default::default()
};

// High-end devices (>4GB)
let config = OnDeviceTrainingConfig {
    method: FineTuningMethod::LoRA { rank: 16, alpha: 32.0 },
    memory_optimization: MemoryOptimization::Minimal,
    max_training_memory_mb: 2048,
    gradient_checkpointing: false,
    ..Default::default()
};
```

## Advanced Training Methods

### QLoRA (Quantized LoRA)

4-bit quantized LoRA for extreme memory efficiency:

```rust
use trustformers_mobile::advanced_training::{
    AdvancedTrainer, AdvancedTrainingMethod
};

let method = AdvancedTrainingMethod::QLoRA {
    rank: 8,
    alpha: 16.0,
    quantization_bits: 4,
    double_quantization: true,
    nf4_quantization: true,  // Normal Float 4
};

let mut trainer = AdvancedTrainer::new(method, training_config, mobile_config)?;
```

**Benefits**:
- 16x memory reduction vs full precision
- Maintains >95% of LoRA performance
- Enables larger models on mobile

### Prompt Tuning / P-tuning

Train soft prompts instead of model weights:

```rust
let method = AdvancedTrainingMethod::PromptTuning {
    num_virtual_tokens: 20,
    prompt_embedding_dim: 768,
    encoder_type: PromptEncoderType::MLP,
    init_method: PromptInitMethod::FromTask,
};
```

**Encoder Types**:
- `Embedding`: Simple lookup table
- `MLP`: Non-linear transformation
- `LSTM`: Sequential encoding
- `Prefix`: Prepended to all layers

### IA³ (Infused Adapter by Inhibiting and Amplifying)

Efficient scaling-based adaptation:

```rust
let method = AdvancedTrainingMethod::IA3 {
    target_modules: vec!["attention".to_string(), "mlp".to_string()],
    scaling_rank: 1,
    init_scale: 1.0,
};
```

**Benefits**:
- Extremely parameter efficient
- No additional inference latency
- Good for instruction tuning

### UniPELT

Unified parameter-efficient learning combining multiple methods:

```rust
let method = AdvancedTrainingMethod::UniPELT {
    lora_rank: 8,
    adapter_size: 64,
    prefix_length: 10,
    gate_type: GateType::Attention,
};
```

**Gate Types**:
- `Linear`: Simple weighted combination
- `Attention`: Dynamic routing
- `Mixture`: Layer-specific weights

### BitFit

Train only bias parameters:

```rust
let method = AdvancedTrainingMethod::BitFit {
    target_layers: vec!["layer".to_string()],
    learning_rate_scale: 10.0,
};
```

**Benefits**:
- Minimal parameters (~0.1%)
- Surprisingly effective
- Zero inference overhead

## Federated Learning

### Basic Setup

```rust
use trustformers_mobile::federated::{
    FederatedLearningClient, FederatedLearningConfig,
    ClientSelectionStrategy, AggregationStrategy
};

let fl_config = FederatedLearningConfig {
    server_endpoint: "https://fl.example.com".to_string(),
    client_id: device_id,
    local_epochs: 5,
    min_clients_for_aggregation: 10,
    enable_differential_privacy: true,
    enable_secure_aggregation: true,
    communication_rounds: 100,
    enable_compression: true,
    compression_ratio: 0.1,
    client_selection: ClientSelectionStrategy::ResourceBased,
    aggregation_strategy: AggregationStrategy::FedAvg,
};

let mut fl_client = FederatedLearningClient::new(
    fl_config,
    training_config,
    mobile_config,
)?;
```

### Client Selection Strategies

- **Random**: Probabilistic selection
- **ResourceBased**: Based on device capabilities
- **QualityBased**: Based on data quality
- **RoundRobin**: Deterministic rotation
- **SpeedOptimized**: Prefer fast devices

### Aggregation Strategies

- **FedAvg**: Simple averaging
- **WeightedAvg**: Weighted by data size
- **FedMomentum**: Momentum-based aggregation
- **FedYogi**: Adaptive aggregation
- **PersonalizedFed**: Client-specific personalization

### Privacy in Federated Learning

```rust
use trustformers_mobile::federated::DifferentialPrivacyConfig;

let dp_config = DifferentialPrivacyConfig {
    epsilon: 1.0,              // Privacy budget
    delta: 1e-6,               // Failure probability
    clipping_norm: 1.0,        // Gradient clipping
    noise_mechanism: NoiseMechanism::Gaussian,
    per_layer_budget: false,   // Uniform budget allocation
};
```

### Communication Efficiency

```rust
// Estimate communication cost
let cost = FederatedLearningUtils::estimate_communication_cost(
    model_size_mb: 100.0,
    compression_ratio: 0.1,
    rounds: 50,
);

println!("Total upload: {} MB", cost.total_upload_mb);
println!("Total download: {} MB", cost.total_download_mb);
```

## Differential Privacy

### Privacy Levels

```rust
use trustformers_mobile::differential_privacy::{
    DifferentialPrivacyEngine, PrivacyConfig, PrivacyLevel
};

// Preset privacy levels
let config = PrivacyConfig::from_privacy_level(PrivacyLevel::High);
```

**Privacy Levels**:
- **Low**: ε ≈ 10 (basic privacy)
- **Medium**: ε ≈ 3 (balanced)
- **High**: ε ≈ 1 (strong privacy)
- **VeryHigh**: ε ≈ 0.1 (maximum privacy)

### Custom Privacy Configuration

```rust
let config = PrivacyConfig {
    privacy_level: PrivacyLevel::Custom,
    total_epsilon: 2.0,
    total_delta: 1e-7,
    noise_multiplier: 1.5,
    clipping_threshold: 0.5,
    per_example_clipping: true,
    adaptive_clipping: true,
    subsampling_rate: 0.01,
    composition_method: CompositionMethod::Moments,
};
```

### Privacy-Preserving Data Augmentation

```rust
let dp_engine = DifferentialPrivacyEngine::new(config);

// Apply privacy to training data
let private_data = dp_engine.privatize_data(
    &training_data,
    DataPrivacyMethod::Mixup { alpha: 1.0 },
)?;
```

**Methods**:
- **InputPerturbation**: Add noise to inputs
- **LabelSmoothing**: Smooth one-hot labels
- **Mixup**: Interpolate between samples
- **CutMix**: Mix regions of samples

### Privacy Cost Estimation

```rust
// Estimate privacy cost before training
let estimate = DifferentialPrivacyEngine::estimate_privacy_cost(
    &training_config,
    &privacy_config,
    dataset_size,
);

if estimate.meets_budget {
    println!("Training will use ε={:.2}", estimate.estimated_epsilon);
} else {
    println!("Warning: Exceeds privacy budget!");
}
```

### Privacy Accounting

```rust
// During training
let report = dp_engine.privatize_gradients(
    &mut gradients,
    batch_size,
    step,
)?;

println!("Step ε: {:.4}", report.epsilon_spent);
println!("Total ε: {:.4}", report.total_epsilon_spent);
println!("Remaining budget: {:.4}", 
         config.total_epsilon - report.total_epsilon_spent);
```

## Best Practices

### 1. Device-Aware Configuration

```rust
// Auto-detect and configure
let device_info = MobileDeviceDetector::detect()?;
let config = MobileTrainingUtils::create_mobile_training_config(
    device_info.memory_info.available_mb,
    device_info.performance_scores.tier,
);
```

### 2. Checkpoint Management

```rust
// Save checkpoints regularly
let checkpoint = trainer.save_checkpoint()?;

// Resume from checkpoint
trainer.load_checkpoint(checkpoint)?;

// Export for deployment
let exported = trainer.export_parameters()?;
```

### 3. Error Recovery

```rust
// Implement retry logic
let mut retries = 3;
loop {
    match trainer.train(&data) {
        Ok(stats) => break,
        Err(e) if retries > 0 => {
            eprintln!("Training failed: {}, retrying...", e);
            retries -= 1;
            // Reduce batch size or sequence length
            config.batch_size = config.batch_size.max(1) / 2;
        }
        Err(e) => return Err(e),
    }
}
```

### 4. Memory Management

```rust
// Monitor memory usage
let memory_stats = trainer.get_memory_stats();
if memory_stats.total_memory_bytes > threshold {
    // Switch to more aggressive optimization
    config.memory_optimization = MemoryOptimization::Maximum;
}
```

### 5. Privacy-Utility Tradeoff

```rust
// Start with low privacy, increase if performance is good
let mut privacy_level = PrivacyLevel::Low;
let baseline_accuracy = evaluate_model(&model)?;

if baseline_accuracy > 0.9 {
    privacy_level = PrivacyLevel::High;
}
```

## Performance Optimization

### Memory Optimization Techniques

1. **Gradient Checkpointing**
   ```rust
   config.gradient_checkpointing = true;  // Trade compute for memory
   ```

2. **Mixed Precision**
   ```rust
   mobile_config.use_fp16 = true;  // 2x memory savings
   ```

3. **Gradient Accumulation**
   ```rust
   config.gradient_accumulation_steps = 8;  // Simulate larger batches
   ```

4. **Sparse Updates**
   ```rust
   // Use methods that support sparsity
   AdvancedTrainingMethod::IA3 { .. }  // Naturally sparse
   ```

### Speed Optimization

1. **Operator Fusion**
   ```rust
   mobile_config.backend = MobileBackend::CoreML;  // Hardware acceleration
   ```

2. **Quantization**
   ```rust
   // Use QLoRA for 4-bit training
   AdvancedTrainingMethod::QLoRA { quantization_bits: 4, .. }
   ```

3. **Efficient Data Loading**
   ```rust
   // Preprocess and cache data
   let preprocessed = preprocess_data(&raw_data)?;
   ```

### Battery Optimization

```rust
use trustformers_mobile::battery::MobileBatteryManager;

let battery_manager = MobileBatteryManager::new(battery_config)?;

// Adjust training based on battery
if battery_manager.get_battery_level()? < 0.3 {
    config.epochs = 1;  // Reduce training
    config.learning_rate *= 2.0;  // Faster convergence
}
```

## Examples

### Example 1: Personal Assistant Fine-Tuning

```rust
// Fine-tune a model for personal writing style
let config = OnDeviceTrainingConfig {
    method: FineTuningMethod::LoRA { rank: 4, alpha: 8.0 },
    epochs: 5,
    learning_rate: 5e-5,
    max_sequence_length: 128,
    ..Default::default()
};

// Use differential privacy for personal data
let privacy_config = PrivacyConfig::from_privacy_level(PrivacyLevel::High);
let dp_engine = DifferentialPrivacyEngine::new(privacy_config);

// Train with privacy
let mut trainer = OnDeviceTrainer::new(config, mobile_config)?;
trainer.train_with_privacy(&personal_messages, &dp_engine)?;
```

### Example 2: Federated Medical Model

```rust
// Federated learning for medical diagnosis
let fl_config = FederatedLearningConfig {
    enable_differential_privacy: true,
    dp_config: Some(DifferentialPrivacyConfig {
        epsilon: 0.1,  // Very high privacy for medical data
        delta: 1e-8,
        ..Default::default()
    }),
    enable_secure_aggregation: true,
    client_selection: ClientSelectionStrategy::QualityBased,
    ..Default::default()
};

// Use QLoRA for memory efficiency
let method = AdvancedTrainingMethod::QLoRA {
    rank: 4,
    quantization_bits: 4,
    nf4_quantization: true,
    ..Default::default()
};
```

### Example 3: Multilingual Adaptation

```rust
// Adapt model to new language using prompt tuning
let method = AdvancedTrainingMethod::PromptTuning {
    num_virtual_tokens: 50,
    prompt_embedding_dim: 768,
    encoder_type: PromptEncoderType::LSTM,
    init_method: PromptInitMethod::FromTask,
};

// Configure for cross-lingual transfer
let config = OnDeviceTrainingConfig {
    epochs: 10,
    learning_rate: 1e-3,  // Higher LR for prompts
    ..Default::default()
};
```

## Troubleshooting

### Out of Memory

1. Reduce batch size and sequence length
2. Enable gradient checkpointing
3. Use more aggressive quantization (QLoRA with 4-bit)
4. Switch to BitFit or IA³

### Slow Training

1. Reduce model complexity (lower LoRA rank)
2. Use hardware acceleration (CoreML/NNAPI)
3. Enable operator fusion
4. Reduce communication rounds in FL

### Poor Accuracy

1. Increase training epochs
2. Use larger LoRA rank
3. Try different initialization methods
4. Reduce differential privacy noise

### Privacy Budget Exceeded

1. Reduce number of epochs
2. Increase batch size
3. Use subsampling
4. Switch to weaker privacy guarantees

## References

- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [QLoRA: Efficient Finetuning](https://arxiv.org/abs/2305.14314)
- [UniPELT: Unified Parameter-Efficient Learning](https://arxiv.org/abs/2110.07577)
- [Differential Privacy in ML](https://arxiv.org/abs/1607.00133)
- [Federated Learning](https://arxiv.org/abs/1602.05629)