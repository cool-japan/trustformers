# Basic Training Examples

This directory contains foundational examples demonstrating core training functionality in TrustformeRS.

## Examples

### 🚀 Simple Classification (`simple_classification.rs`)

A comprehensive example showing basic supervised learning with a feedforward neural network.

#### What You'll Learn
- Setting up models, optimizers, and loss functions
- Creating training and evaluation loops
- Using callbacks for monitoring progress
- Implementing custom metrics
- Saving and loading checkpoints
- Working with synthetic data

#### Key Features Demonstrated
- **Model Definition**: Custom neural network implementation
- **Training Loop**: Complete training with evaluation
- **Progress Monitoring**: Real-time loss and accuracy tracking
- **Early Stopping**: Prevent overfitting with patience-based stopping
- **Checkpointing**: Save model state during training
- **Metrics**: Custom accuracy metric implementation

#### Running the Example

```bash
# Basic run with default configuration
cargo run --example simple_classification

# With custom configuration
cargo run --example simple_classification -- --config config.json

# In release mode for better performance
cargo run --release --example simple_classification
```

#### Configuration

The example uses a JSON configuration file (`config.json`) that you can modify:

```json
{
  "model": {
    "input_size": 20,        // Input feature dimension
    "hidden_size": 128,      // Hidden layer size
    "num_classes": 5         // Number of output classes
  },
  "training": {
    "learning_rate": 0.001,  // Adam learning rate
    "batch_size": 64,        // Training batch size
    "num_epochs": 15,        // Number of training epochs
    "weight_decay": 0.0001   // L2 regularization
  },
  "data": {
    "num_train_samples": 2000,  // Training samples
    "num_eval_samples": 400,    // Validation samples
    "noise_level": 0.15         // Data noise for difficulty
  }
}
```

#### Expected Output

```
🚀 TrustformeRS Simple Classification Training Example
=================================================
Configuration:
  Input size: 20
  Hidden size: 128
  Number of classes: 5
  Learning rate: 0.001
  Batch size: 64
  Number of epochs: 15

📊 Generating synthetic data...
  Training samples: 2000
  Evaluation samples: 400

🧠 Creating model...
  Model parameters: 15,877

🎯 Initializing trainer...
  Checkpoints will be saved to: ./checkpoints

🔥 Starting training...

Starting epoch 1
  Batch 0: loss = 1.6247
  Batch 15: loss = 1.2384
Epoch 1 completed:
  Train loss: 1.1245
  Eval loss: 1.0876
  Accuracy: 0.4250

...

✅ Training completed successfully!

📈 Final model evaluation:
  eval_loss: 0.3421
  accuracy: 0.8975

🎉 Example completed successfully!
```

#### Understanding the Results

- **Loss**: Should decrease over epochs (convergence)
- **Accuracy**: Should increase (better classification)
- **Final accuracy**: Typically 85-95% on synthetic data

#### Customization Ideas

1. **Change Model Architecture**:
   ```rust
   let model = SimpleClassifier::new(
       50,    // input_size
       256,   // hidden_size  
       10,    // num_classes
   )?;
   ```

2. **Add More Layers**:
   ```rust
   // Modify the SimpleClassifier to have more layers
   struct DeepClassifier {
       layers: Vec<LinearLayer>,
   }
   ```

3. **Different Loss Functions**:
   ```rust
   // Implement focal loss for imbalanced data
   struct FocalLoss {
       alpha: f32,
       gamma: f32,
   }
   ```

4. **Custom Metrics**:
   ```rust
   struct F1ScoreMetric;
   
   impl Metric for F1ScoreMetric {
       fn compute(&self, predictions: &Tensor, targets: &Tensor) -> MetricResult {
           // F1 score implementation
       }
   }
   ```

#### Troubleshooting

**Loss Not Decreasing**:
- Try lowering learning rate (0.0001)
- Increase model capacity (hidden_size)
- Check data quality and labels

**Overfitting** (train accuracy >> eval accuracy):
- Add dropout or weight decay
- Reduce model size
- Increase training data

**Memory Issues**:
- Reduce batch_size
- Use smaller model
- Enable gradient checkpointing

**Slow Training**:
- Use `--release` mode
- Increase batch_size (if memory allows)
- Enable mixed precision

#### Next Steps

After mastering this example:

1. **Try Advanced Features**: Move to `../advanced_features/`
2. **Explore Distributed Training**: Check `../distributed_training/`
3. **Learn RLHF**: See `../rlhf_training/`
4. **Benchmark Performance**: Use `../benchmarks/`

#### Code Structure

```
simple_classification.rs
├── SimpleClassifier        # Model definition
├── AccuracyMetric         # Custom metric
├── CrossEntropyLoss       # Loss function
├── ProgressCallback       # Training monitoring
├── generate_data()        # Synthetic data creation
└── main()                 # Training orchestration
```

#### Key Takeaways

- ✅ Understanding the basic training loop structure
- ✅ Model definition and forward pass implementation
- ✅ Loss functions and metrics integration
- ✅ Callback system for monitoring and control
- ✅ Configuration and checkpointing patterns
- ✅ Data handling and batch processing

This example provides the foundation for all other training scenarios in TrustformeRS!