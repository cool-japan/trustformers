# Migration from PyTorch to TrustformeRS

This guide focuses specifically on migrating PyTorch code to TrustformeRS, covering core concepts, API mappings, and practical examples.

## Quick Reference

| PyTorch | TrustformeRS | Notes |
|---------|--------------|-------|
| `torch.Tensor` | `Tensor` | Strongly typed, explicit error handling |
| `torch.nn.Module` | `impl Model` trait | Rust traits instead of inheritance |
| `tensor.cuda()` | `tensor.to_device()` | Multi-backend support |
| `torch.no_grad()` | N/A | Explicit gradient tracking |
| `torch.autograd` | Built-in AD | Automatic differentiation integrated |

## Core Concepts

### Tensor Creation and Manipulation

#### PyTorch
```python
import torch

# Creation
x = torch.tensor([1, 2, 3], dtype=torch.float32)
y = torch.zeros(3, 4)
z = torch.randn(2, 3, requires_grad=True)

# Device management
x = x.cuda()
x = x.to('cuda:0')

# Operations
result = torch.matmul(a, b)
sum_result = x.sum(dim=1, keepdim=True)
```

#### TrustformeRS
```rust
use trustformers_core::tensor::{Tensor, Device};

// Creation
let x = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3])?;
let y = Tensor::zeros(&[3, 4])?;
let z = Tensor::randn(&[2, 3])?.requires_grad(true);

// Device management
let x = x.to_device(Device::cuda(0)?)?;

// Operations
let result = a.matmul(&b)?;
let sum_result = x.sum_keepdim(&[1])?;
```

### Neural Network Layers

#### PyTorch Linear Layer
```python
import torch.nn as nn

# Define
linear = nn.Linear(784, 256)

# Use
output = linear(input)

# Access parameters
weight = linear.weight
bias = linear.bias
```

#### TrustformeRS Linear Layer
```rust
use trustformers_core::layers::Linear;

// Define
let linear = Linear::new(784, 256)?;

// Use
let output = linear.forward(&input)?;

// Access parameters
let weight = linear.weight();
let bias = linear.bias();
```

### Custom Modules

#### PyTorch
```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

#### TrustformeRS
```rust
use trustformers_core::{Model, ModelOutput, layers::*};

pub struct MLP {
    fc1: Linear,
    activation: ReLU,
    dropout: Dropout,
    fc2: Linear,
}

impl MLP {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Result<Self> {
        Ok(Self {
            fc1: Linear::new(input_dim, hidden_dim)?,
            activation: ReLU::new(),
            dropout: Dropout::new(0.1),
            fc2: Linear::new(hidden_dim, output_dim)?,
        })
    }
}

impl Model for MLP {
    type Input = Tensor;
    type Output = ModelOutput;
    type Config = MLPConfig;
    
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let x = self.fc1.forward(&input)?;
        let x = self.activation.forward(&x)?;
        let x = self.dropout.forward(&x)?;
        let x = self.fc2.forward(&x)?;
        Ok(ModelOutput::new(x))
    }
}
```

### Optimizers

#### PyTorch
```python
# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.01)

# Training step
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Learning rate scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
scheduler.step()
```

#### TrustformeRS
```rust
use trustformers_optim::{AdamW, CosineScheduler, Optimizer};

// Create optimizer
let mut optimizer = AdamW::new(1e-3, (0.9, 0.999), 1e-8, 0.01);

// Training step
optimizer.zero_grad();
loss.backward()?;
for (param, grad) in model.parameters_and_gradients()? {
    optimizer.update(param, grad)?;
}
optimizer.step();

// Learning rate scheduling
let mut scheduler = CosineScheduler::new(1e-3, 1000, 10000, 1e-5);
let lr = scheduler.get_lr(step);
optimizer.set_lr(lr);
```

### Data Loading

#### PyTorch DataLoader
```python
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

#### TrustformeRS DataLoader
```rust
use trustformers_data::{Dataset, DataLoader, Transform};

pub struct ImageDataset {
    images: Vec<Tensor>,
    labels: Vec<i64>,
    transform: Option<Box<dyn Transform>>,
}

impl Dataset for ImageDataset {
    type Item = (Tensor, i64);
    
    fn len(&self) -> usize {
        self.images.len()
    }
    
    fn get(&self, idx: usize) -> Result<Self::Item> {
        let mut image = self.images[idx].clone();
        if let Some(transform) = &self.transform {
            image = transform.apply(&image)?;
        }
        Ok((image, self.labels[idx]))
    }
}

let loader = DataLoader::builder()
    .dataset(dataset)
    .batch_size(32)
    .shuffle(true)
    .num_workers(4)
    .build()?;
```

### Loss Functions

#### PyTorch
```python
# Classification
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, labels)

# Regression
criterion = nn.MSELoss()
loss = criterion(predictions, targets)

# Custom loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs, targets):
        # Implementation
        pass
```

#### TrustformeRS
```rust
use trustformers_core::loss::{CrossEntropyLoss, MSELoss, Loss};

// Classification
let criterion = CrossEntropyLoss::new();
let loss = criterion.forward(&logits, &labels)?;

// Regression
let criterion = MSELoss::new();
let loss = criterion.forward(&predictions, &targets)?;

// Custom loss
pub struct FocalLoss {
    gamma: f32,
    alpha: f32,
}

impl Loss for FocalLoss {
    fn forward(&self, inputs: &Tensor, targets: &Tensor) -> Result<Tensor> {
        // Implementation
    }
}
```

### Mixed Precision Training

#### PyTorch
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

#### TrustformeRS
```rust
use trustformers_core::amp::{autocast, GradScaler};

let mut scaler = GradScaler::new();

for batch in dataloader {
    let (data, target) = batch?;
    optimizer.zero_grad();
    
    let (output, loss) = autocast(|| {
        let output = model.forward(&data)?;
        let loss = criterion.forward(&output, &target)?;
        Ok((output, loss))
    })?;
    
    scaler.scale_loss(&loss)?.backward()?;
    scaler.step(&mut optimizer)?;
    scaler.update();
}
```

### Model Checkpointing

#### PyTorch
```python
# Save
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# Load
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

#### TrustformeRS
```rust
use trustformers_core::checkpoint::{Checkpoint, save_checkpoint, load_checkpoint};

// Save
let checkpoint = Checkpoint {
    epoch,
    model_state_dict: model.state_dict()?,
    optimizer_state_dict: optimizer.state_dict()?,
    loss: loss.item(),
    metadata: HashMap::new(),
};
save_checkpoint(&checkpoint, "checkpoint.safetensors")?;

// Load
let checkpoint = load_checkpoint("checkpoint.safetensors")?;
model.load_state_dict(&checkpoint.model_state_dict)?;
optimizer.load_state_dict(&checkpoint.optimizer_state_dict)?;
let epoch = checkpoint.epoch;
let loss = checkpoint.loss;
```

## Advanced Topics

### Distributed Training

#### PyTorch DDP
```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize
dist.init_process_group("nccl")
model = model.cuda()
model = DDP(model)

# Training
output = model(input)
loss.backward()
```

#### TrustformeRS
```rust
use trustformers_core::distributed::{init_process_group, DistributedDataParallel};

// Initialize
init_process_group("nccl")?;
let model = model.to_device(Device::cuda(local_rank)?)?;
let model = DistributedDataParallel::new(model)?;

// Training
let output = model.forward(&input)?;
loss.backward()?;
```

### Hooks and Callbacks

#### PyTorch
```python
def print_grad_norm(grad):
    print(f"Gradient norm: {grad.norm()}")

# Register hook
handle = model.fc1.weight.register_hook(print_grad_norm)

# Remove hook
handle.remove()
```

#### TrustformeRS
```rust
use trustformers_core::hooks::{Hook, register_hook};

// Define hook
fn print_grad_norm(grad: &Tensor) -> Result<()> {
    println!("Gradient norm: {}", grad.norm()?);
    Ok(())
}

// Register hook
let handle = model.fc1.weight_mut().register_hook(Box::new(print_grad_norm))?;

// Remove hook
handle.remove()?;
```

## Performance Tips

### Memory Management

1. **Explicit tensor cleanup**:
   ```rust
   {
       let temp_tensor = large_computation()?;
       // temp_tensor dropped here, memory freed immediately
   }
   ```

2. **Reuse allocations**:
   ```rust
   let mut buffer = Tensor::zeros(&[batch_size, hidden_dim])?;
   for batch in dataloader {
       model.forward_into(&batch, &mut buffer)?;
   }
   ```

3. **Gradient checkpointing**:
   ```rust
   use trustformers_core::checkpoint;
   
   let output = checkpoint::checkpoint(|| {
       expensive_layer.forward(&input)
   })?;
   ```

### Parallelization

```rust
// Parallel data preprocessing
use rayon::prelude::*;

let processed: Vec<_> = data.par_iter()
    .map(|x| preprocess(x))
    .collect::<Result<Vec<_>>>()?;

// Parallel model inference
let outputs: Vec<_> = batches.par_iter()
    .map(|batch| model.forward(batch))
    .collect::<Result<Vec<_>>>()?;
```

## Common Pitfalls and Solutions

### 1. Forgetting Error Handling
```rust
// Bad: Will panic on error
let output = model.forward(&input).unwrap();

// Good: Proper error propagation
let output = model.forward(&input)?;
```

### 2. Device Mismatches
```rust
// Ensure tensors are on same device
let input = input.to_device(model.device())?;
let output = model.forward(&input)?;
```

### 3. Shape Mismatches
```rust
// Validate shapes before operations
assert_eq!(a.shape()[1], b.shape()[0], "Matrix dimensions must match");
let result = a.matmul(&b)?;
```

### 4. Memory Leaks in Loops
```rust
// Clear gradients and intermediate tensors
for epoch in 0..num_epochs {
    for batch in &dataloader {
        optimizer.zero_grad();
        // Forward and backward pass
        drop(intermediate_tensors); // Explicit cleanup if needed
    }
}
```

## Testing Your Migration

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_model_output_matches_pytorch() {
        // Load PyTorch reference output
        let pytorch_output = load_reference_output("pytorch_output.json")?;
        
        // Run TrustformeRS model
        let model = MyModel::new()?;
        let trustformers_output = model.forward(&test_input)?;
        
        // Compare outputs
        assert_relative_eq!(
            trustformers_output.as_slice(),
            pytorch_output.as_slice(),
            epsilon = 1e-5
        );
    }
}
```

## Resources

- [PyTorch to TrustformeRS Cheatsheet](./pytorch_cheatsheet.md)
- [Complete Migration Examples](../examples/migration/)
- [Performance Comparison](./benchmarks.md)
- [FAQ](./faq.md#pytorch-migration)

For more detailed examples, check out the `examples/pytorch_migration/` directory in the repository.