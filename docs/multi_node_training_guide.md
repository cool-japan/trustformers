# Multi-Node Training with ZeRO Optimization

This guide explains how to use TrustformeRS for distributed training across multiple nodes using MPI and ZeRO (Zero Redundancy Optimizer) optimizations.

## Prerequisites

1. **MPI Installation**: Install OpenMPI or MPICH on all nodes
   ```bash
   # Ubuntu/Debian
   sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev
   
   # macOS
   brew install open-mpi
   
   # CentOS/RHEL
   sudo yum install openmpi openmpi-devel
   ```

2. **Build with MPI Support**:
   ```bash
   cargo build --release --features mpi
   ```

## ZeRO Optimization Stages

TrustformeRS implements all three ZeRO optimization stages:

### ZeRO Stage 1: Optimizer State Partitioning
- Partitions optimizer states (momentum, variance) across devices
- Memory reduction: ~4x for Adam optimizer
- No communication overhead during forward/backward

### ZeRO Stage 2: Optimizer State + Gradient Partitioning
- Includes Stage 1 benefits
- Additionally partitions gradients across devices
- Memory reduction: ~8x for typical models
- Minimal communication overhead (one reduce-scatter per backward)

### ZeRO Stage 3: Full Parameter Partitioning
- Includes Stage 1 & 2 benefits
- Partitions model parameters across devices
- Memory reduction: ~Nx where N is number of devices
- Higher communication overhead but enables huge models

## Usage Examples

### Basic Multi-Node Training

```rust
use trustformers::optim::{AdamW, ZeroOptimizer, ZeroStage};
use trustformers::training::{Trainer, TrainingArguments};
use trustformers::parallel::{CommunicationBackend, ModelParallelContext};

fn main() -> Result<()> {
    // Initialize MPI for multi-node communication
    let ctx = ModelParallelContext::new(
        CommunicationBackend::Mpi,
        DeviceMesh::Linear(num_gpus()),
    )?;
    
    // Create base optimizer
    let base_optimizer = AdamW::new(
        learning_rate: 1e-4,
        weight_decay: 0.01,
    );
    
    // Wrap with ZeRO Stage 2 optimization
    let optimizer = ZeroOptimizer::new(
        base_optimizer,
        ZeroStage::Stage2,
        ctx.communicator(),
    )?;
    
    // Configure training
    let training_args = TrainingArguments {
        per_device_batch_size: 8,
        gradient_accumulation_steps: 4,
        num_epochs: 3,
        // ... other args
    };
    
    // Create trainer
    let mut trainer = Trainer::new(
        model,
        optimizer,
        training_args,
        train_dataset,
        eval_dataset,
    )?;
    
    // Train across multiple nodes
    trainer.train()?;
    
    Ok(())
}
```

### Running with MPI

1. **Single Node, Multiple GPUs**:
   ```bash
   mpirun -np 4 ./target/release/your_training_script
   ```

2. **Multiple Nodes**:
   Create a hostfile:
   ```
   node1 slots=4
   node2 slots=4
   node3 slots=4
   ```
   
   Run:
   ```bash
   mpirun -hostfile hosts -np 12 ./target/release/your_training_script
   ```

### Advanced Configuration

```rust
// Configure ZeRO Stage 3 with custom settings
let zero_config = ZeroConfig {
    stage: ZeroStage::Stage3,
    offload_optimizer: true,      // Offload optimizer states to CPU
    offload_param: false,          // Keep parameters on GPU
    overlap_comm: true,            // Overlap communication with computation
    reduce_bucket_size: 5e8,       // 500MB reduce buckets
    allgather_bucket_size: 5e8,    // 500MB allgather buckets
    contiguous_gradients: true,    // Use contiguous gradient buffer
};

let optimizer = ZeroOptimizer::with_config(
    base_optimizer,
    zero_config,
    ctx.communicator(),
)?;
```

### Memory Monitoring

```rust
// Monitor memory usage across nodes
let stats = optimizer.get_memory_stats();
println!("Rank {}: Memory allocated: {:.2} GB", 
         ctx.rank(), 
         stats.allocated_bytes as f64 / 1e9);
println!("Rank {}: Memory reserved: {:.2} GB", 
         ctx.rank(), 
         stats.reserved_bytes as f64 / 1e9);
```

## Performance Tips

1. **Batch Size Scaling**: When using ZeRO, you can increase batch size proportionally to the number of devices:
   ```rust
   let effective_batch_size = per_device_batch_size * num_devices * gradient_accumulation_steps;
   ```

2. **Communication Optimization**:
   - Use high-speed interconnect (InfiniBand, RoCE)
   - Enable NUMA binding for optimal memory access
   - Use process affinity to bind MPI ranks to specific CPUs

3. **Stage Selection**:
   - Stage 1: Good for small models, minimal overhead
   - Stage 2: Best balance for most use cases
   - Stage 3: Required for very large models that don't fit on single GPU

## Troubleshooting

### Common Issues

1. **MPI Initialization Failure**:
   ```
   Error: Failed to initialize MPI
   ```
   Solution: Ensure MPI is properly installed and `mpirun` is in PATH

2. **Communication Timeout**:
   ```
   Error: MPI communication timeout
   ```
   Solution: Check network connectivity between nodes, firewall settings

3. **Memory Errors with Stage 3**:
   ```
   Error: Out of memory during parameter gathering
   ```
   Solution: Reduce batch size or enable parameter offloading

### Environment Variables

```bash
# Increase communication buffer size
export OMPI_MCA_btl_tcp_sndbuf=8388608
export OMPI_MCA_btl_tcp_rcvbuf=8388608

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Enable NCCL debugging (if using NCCL backend)
export NCCL_DEBUG=INFO
```

## Benchmarks

Expected memory savings and performance with ZeRO:

| Model Size | Stage 0 (DDP) | Stage 1 | Stage 2 | Stage 3 |
|------------|---------------|---------|---------|---------|
| 1.5B params| 12 GB/GPU     | 10 GB   | 8 GB    | 4 GB    |
| 6B params  | 48 GB/GPU     | 40 GB   | 32 GB   | 16 GB   |
| 20B params | OOM           | OOM     | 80 GB   | 40 GB   |

## Example: Training Large Language Model

```rust
use trustformers::models::{LlamaModel, LlamaConfig};
use trustformers::optim::{AdamW, ZeroOptimizer, ZeroStage};
use trustformers::training::distributed::DistributedTrainer;

fn train_large_llm() -> Result<()> {
    // Initialize distributed context
    let ctx = ModelParallelContext::new(
        CommunicationBackend::Mpi,
        DeviceMesh::Linear(world_size()),
    )?;
    
    // Create large model (13B parameters)
    let config = LlamaConfig::llama_13b();
    let model = LlamaModel::new(config)?;
    
    // Use ZeRO Stage 3 for memory efficiency
    let optimizer = ZeroOptimizer::new(
        AdamW::new(1e-4, 0.01),
        ZeroStage::Stage3,
        ctx.communicator(),
    )?;
    
    // Distributed trainer handles data parallelism
    let trainer = DistributedTrainer::new(
        model,
        optimizer,
        ctx,
        TrainingArguments {
            per_device_batch_size: 2,
            gradient_accumulation_steps: 8,
            gradient_checkpointing: true,  // Further memory savings
            fp16: true,                    // Mixed precision training
            // ...
        },
    )?;
    
    // Train across all nodes
    trainer.train(train_dataloader, eval_dataloader)?;
    
    Ok(())
}

// Run with: mpirun -np 32 ./train_llm
```

## Next Steps

- [Pipeline Parallelism Guide](./pipeline_parallelism.md) for even larger models
- [Performance Tuning Guide](./performance_tuning.md) for optimization tips
- [Checkpoint Guide](./checkpointing.md) for saving/resuming distributed training

For more examples, see the `examples/distributed/` directory.