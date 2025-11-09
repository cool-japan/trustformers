# trustformers-training

Comprehensive training infrastructure for transformer models with distributed training, hyperparameter optimization, and advanced training techniques.

## Current State

This crate provides **production-ready training capabilities** including distributed training across multiple nodes, mixed precision training, hyperparameter tuning, and quantization-aware training. The design closely follows HuggingFace Transformers' Trainer API for familiarity.

## Features

### Core Training Infrastructure
- **Trainer API**: HuggingFace-compatible training interface
- **TrainingArguments**: Comprehensive configuration system
- **Gradient Management**: Accumulation, clipping, and scaling
- **Checkpoint System**: Save/resume with full state preservation
- **Early Stopping**: Patience-based training termination
- **Callback System**: Extensible hooks for custom behavior

### Distributed Training
- **Data Parallel Training**: Multi-GPU with gradient synchronization
- **Model Parallel Support**: Tensor and pipeline parallelism
- **ZeRO Optimization**: All three stages implemented
  - Stage 1: Optimizer state sharding
  - Stage 2: Optimizer + gradient sharding
  - Stage 3: Full parameter sharding
- **Multi-Node Support**: MPI backend for cluster training
- **Process Groups**: NCCL, Gloo, MPI backends

### Mixed Precision Training
- **Automatic Mixed Precision (AMP)**: FP16/BF16 training
- **Dynamic Loss Scaling**: Prevent gradient underflow
- **Gradient Scaling**: Automatic scale adjustment
- **Memory Savings**: ~50% reduction in memory usage
- **Performance**: 1.5-2x training speedup

### Hyperparameter Optimization
- **Random Search**: Basic hyperparameter exploration
- **Bayesian Optimization**: Efficient search with Gaussian processes
- **Grid Search**: Exhaustive parameter combinations
- **Early Stopping**: Resource-efficient search termination
- **Parallel Trials**: Multiple experiments simultaneously

### Advanced Training Features
- **Quantization-Aware Training (QAT)**: Train with simulated quantization
- **Gradient Checkpointing**: Trade compute for memory
- **Learning Rate Schedulers**: 
  - Linear, cosine, polynomial decay
  - Warmup strategies
  - Exponential and step decay
- **Custom Loss Functions**: CrossEntropy, MSE, with label smoothing
- **Metrics System**: Accuracy, F1, perplexity, custom metrics

## Usage Example

```rust
use trustformers_training::{
    Trainer, TrainingArguments,
    optimizers::{AdamW, AdamWConfig},
    schedulers::{LinearScheduler, SchedulerConfig},
};

// Configure training
let args = TrainingArguments {
    output_dir: "output".to_string(),
    num_train_epochs: 3,
    per_device_train_batch_size: 32,
    learning_rate: 5e-5,
    warmup_steps: 500,
    logging_steps: 100,
    save_steps: 1000,
    fp16: true,
    gradient_accumulation_steps: 4,
    ..Default::default()
};

// Create optimizer
let optimizer = AdamW::new(AdamWConfig {
    lr: args.learning_rate,
    weight_decay: 0.01,
    ..Default::default()
})?;

// Create trainer
let trainer = Trainer::new(
    model,
    args,
    train_dataset,
    eval_dataset,
    optimizer,
)?;

// Train
trainer.train()?;
```

## Distributed Training Example

```rust
use trustformers_training::distributed::{
    DistributedTrainer,
    ProcessGroup,
    ZeroStage,
};

// Initialize distributed environment
let process_group = ProcessGroup::new_from_env()?;

// Configure ZeRO
let trainer = DistributedTrainer::new(
    model,
    args,
    process_group,
    ZeroStage::Three, // Full parameter sharding
)?;

// Train across multiple GPUs/nodes
trainer.train()?;
```

## Architecture

```
trustformers-training/
├── src/
│   ├── trainer.rs          # Main trainer implementation
│   ├── args.rs            # Training arguments
│   ├── distributed/       # Distributed training
│   │   ├── data_parallel.rs
│   │   ├── zero.rs       # ZeRO optimizer
│   │   └── process_group.rs
│   ├── mixed_precision/   # AMP implementation
│   ├── hyperparameter/    # HP optimization
│   ├── loss/             # Loss functions
│   ├── metrics/          # Evaluation metrics
│   ├── callbacks/        # Callback system
│   └── schedulers/       # LR schedulers
```

## Performance Features

### Memory Optimization
- **Gradient Checkpointing**: Re-compute activations during backward pass
- **CPU Offloading**: Move optimizer states to CPU (ZeRO Stage 3)
- **Mixed Precision**: Reduce memory with FP16 weights
- **Gradient Accumulation**: Larger effective batch sizes

### Speed Optimization
- **Data Parallel**: Linear scaling with multiple GPUs
- **Fused Optimizers**: Combined update operations
- **Efficient Communication**: Optimized all-reduce
- **Overlapped Computation**: Hide communication latency

## Benchmarks

| Configuration | GPUs | Model | Throughput | Speedup |
|--------------|------|-------|------------|---------|
| Single GPU | 1 | BERT-Large | 250 samples/s | 1.0x |
| Data Parallel | 8 | BERT-Large | 1,920 samples/s | 7.7x |
| ZeRO Stage 2 | 8 | GPT-2 1.5B | 450 samples/s | - |
| ZeRO Stage 3 | 16 | LLaMA 7B | 320 samples/s | - |

*Benchmarks on NVIDIA A100 GPUs with NVLink*

## Advanced Features

### Callbacks
- **EarlyStoppingCallback**: Stop training when metric plateaus
- **ModelCheckpoint**: Save best models during training
- **TensorBoardCallback**: Log metrics to TensorBoard
- **Custom Callbacks**: Implement the `TrainerCallback` trait

### Metrics
- **Classification**: Accuracy, F1, Precision, Recall
- **Generation**: Perplexity, BLEU
- **Custom Metrics**: Implement the `Metric` trait
- **Metric Collections**: Combine multiple metrics

## Testing

- Unit tests for all components
- Integration tests for full training runs
- Distributed training tests
- Memory leak detection
- Performance benchmarks

## Future Enhancements

Planned additions:
- RLHF (Reinforcement Learning from Human Feedback)
- Continual learning support
- Neural Architecture Search (NAS)
- Federated learning
- More optimization algorithms

## License

MIT OR Apache-2.0