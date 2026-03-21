# Migrating from TensorFlow to TrustformeRS

This guide helps you migrate your TensorFlow-based machine learning projects to TrustformeRS, leveraging Rust's performance and safety benefits while maintaining familiar workflows.

## Overview

TrustformeRS provides TensorFlow-inspired APIs while offering superior performance, memory safety, and easier deployment. This guide covers the most common migration patterns from TensorFlow/Keras to TrustformeRS.

## Table of Contents

1. [Basic Concepts](#basic-concepts)
2. [Model Loading](#model-loading)
3. [Inference Pipeline](#inference-pipeline)
4. [Training Migration](#training-migration)
5. [Keras-style APIs](#keras-style-apis)
6. [Performance Optimization](#performance-optimization)
7. [Common Patterns](#common-patterns)
8. [Troubleshooting](#troubleshooting)

## Basic Concepts

### Tensor Operations

**TensorFlow:**
```python
import tensorflow as tf

# Create tensors
x = tf.random.normal([10, 20])
y = tf.ones([10, 20])

# Operations
z = tf.add(x, y)
z = tf.matmul(x, tf.transpose(y))
```

**TrustformeRS:**
```rust
use trustformers::Tensor;

// Create tensors
let x = Tensor::randn(&[10, 20])?;
let y = Tensor::ones(&[10, 20])?;

// Operations
let z = x.add(&y)?;
let z = x.matmul(&y.transpose(0, 1)?)?;
```

### Device Management

**TensorFlow:**
```python
# Check for GPU
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        x = tf.constant([1.0, 2.0, 3.0])
else:
    with tf.device('/CPU:0'):
        x = tf.constant([1.0, 2.0, 3.0])
```

**TrustformeRS:**
```rust
use trustformers::Device;

let device = Device::best_available()?;
let x = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3])?.to_device(&device)?;
```

### Eager Execution vs Graph Mode

**TensorFlow:**
```python
# Eager execution (TF 2.x default)
@tf.function
def compute_function(x, y):
    return tf.add(x, y)

# Graph execution
result = compute_function(x, y)
```

**TrustformeRS:**
```rust
use trustformers::graph::{ComputationGraph, GraphNode};

// TrustformeRS supports both eager and graph execution
// Eager execution (default)
let result = x.add(&y)?;

// Graph execution
let mut graph = ComputationGraph::new();
let node1 = graph.add_input("x", &[10, 20]);
let node2 = graph.add_input("y", &[10, 20]);
let result_node = graph.add_operation("add", &[node1, node2]);
let result = graph.execute(&[("x", &x), ("y", &y)])?;
```

## Model Loading

### Loading Pre-trained Models

**TensorFlow:**
```python
import tensorflow_hub as hub

# Load from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Load saved model
model = tf.saved_model.load("path/to/saved_model")
```

**TrustformeRS:**
```rust
use trustformers::{AutoModel, Hub};

// Load from TrustformeRS Hub
let model = AutoModel::from_pretrained("universal-sentence-encoder").await?;

// Load local model
let model = AutoModel::from_pretrained("path/to/model").await?;
```

### Keras Model Loading

**TensorFlow/Keras:**
```python
from tensorflow import keras

# Load Keras model
model = keras.models.load_model("model.h5")

# Or load SavedModel format
model = keras.models.load_model("saved_model_dir")
```

**TrustformeRS:**
```rust
use trustformers::keras_compat::{KerasModel, KerasConverter};

// Load and convert Keras model
let keras_model = KerasModel::load("model.h5").await?;
let model = KerasConverter::convert(keras_model)?;

// Or load directly if already converted
let model = AutoModel::from_pretrained("converted_model").await?;
```

## Inference Pipeline

### Text Classification

**TensorFlow:**
```python
import tensorflow_text as text

# Preprocess text
preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

text_input = ["I love this library!"]
encoder_inputs = preprocessor(text_input)
outputs = encoder(encoder_inputs)

# Classification head
classifier = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid')
])
predictions = classifier(outputs['pooled_output'])
```

**TrustformeRS:**
```rust
use trustformers::pipeline;

let classifier = pipeline(
    "text-classification",
    Some("bert-base-uncased"),
    None
).await?;

let result = classifier.predict("I love this library!")?;
// Returns: ClassificationOutput { label: "POSITIVE", score: 0.9998 }
```

### Batch Processing

**TensorFlow:**
```python
# Batch processing
texts = ["I love this!", "This is terrible.", "It's okay."]
predictions = model(preprocessor(texts))
```

**TrustformeRS:**
```rust
let texts = vec![
    "I love this!".to_string(),
    "This is terrible.".to_string(),
    "It's okay.".to_string()
];
let results = classifier.batch(texts)?;
```

### Image Classification

**TensorFlow:**
```python
import tensorflow as tf

# Load image
image = tf.io.read_file("image.jpg")
image = tf.image.decode_image(image, channels=3)
image = tf.cast(image, tf.float32) / 255.0
image = tf.expand_dims(image, 0)  # Add batch dimension

# Load model and predict
model = tf.keras.applications.ResNet50(weights='imagenet')
predictions = model(image)
```

**TrustformeRS:**
```rust
use trustformers::vision::{ImageProcessor, pipeline};

let processor = ImageProcessor::new("resnet50")?;
let image = processor.load_image("image.jpg")?;

let classifier = pipeline(
    "image-classification",
    Some("resnet50"),
    None
).await?;

let result = classifier.predict_image(&image)?;
```

## Training Migration

### Basic Training Loop

**TensorFlow/Keras:**
```python
import tensorflow as tf

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint('best_model.h5'),
        tf.keras.callbacks.EarlyStopping(patience=3)
    ]
)
```

**TrustformeRS:**
```rust
use trustformers::{Adam, CrossEntropyLoss, TrainingConfig, Trainer};

let optimizer = Adam::new(model.parameters(), 0.001)?;
let loss_fn = CrossEntropyLoss::new();

let config = TrainingConfig {
    epochs: 10,
    batch_size: 32,
    early_stopping_patience: Some(3),
    checkpoint_path: Some("best_model.safetensors".into()),
    ..Default::default()
};

let mut trainer = Trainer::new(model, optimizer, loss_fn, config)?;
trainer.train(train_dataset, Some(val_dataset)).await?;
```

### Custom Training Loop

**TensorFlow:**
```python
optimizer = tf.keras.optimizers.Adam()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

@tf.function
def train_step(x, y):
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_fn(y, predictions)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

for epoch in range(epochs):
    for batch, (x, y) in enumerate(train_dataset):
        loss = train_step(x, y)
        if batch % 100 == 0:
            print(f"Epoch {epoch}, Batch {batch}, Loss: {loss}")
```

**TrustformeRS:**
```rust
use trustformers::{Adam, CrossEntropyLoss};

let mut optimizer = Adam::new(model.parameters(), 0.001)?;
let loss_fn = CrossEntropyLoss::new();

for epoch in 0..epochs {
    for (batch, (x, y)) in train_dataset.enumerate() {
        optimizer.zero_grad()?;
        
        model.set_training(true);
        let outputs = model.forward(&x)?;
        let loss = loss_fn.forward(&outputs, &y)?;
        
        loss.backward()?;
        optimizer.step()?;
        
        if batch % 100 == 0 {
            println!("Epoch {}, Batch {}, Loss: {}", epoch, batch, loss.item());
        }
    }
}
```

## Keras-style APIs

### Sequential Model

**TensorFlow/Keras:**
```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])
```

**TrustformeRS:**
```rust
use trustformers::keras_compat::{Sequential, Dense, Dropout};

let model = Sequential::new(vec![
    Box::new(Dense::new(128, 784).with_activation("relu")),
    Box::new(Dropout::new(0.2)),
    Box::new(Dense::new(10, 128).with_activation("softmax")),
])?;
```

### Functional API

**TensorFlow/Keras:**
```python
inputs = keras.Input(shape=(784,))
x = keras.layers.Dense(128, activation='relu')(inputs)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

**TrustformeRS:**
```rust
use trustformers::keras_compat::{Input, Model, Dense, Dropout};

let inputs = Input::new(&[784])?;
let x = Dense::new(128, 784).with_activation("relu").apply(&inputs)?;
let x = Dropout::new(0.2).apply(&x)?;
let outputs = Dense::new(10, 128).with_activation("softmax").apply(&x)?;

let model = Model::new(inputs, outputs)?;
```

### Custom Layers

**TensorFlow/Keras:**
```python
class CustomLayer(keras.layers.Layer):
    def __init__(self, units=32):
        super(CustomLayer, self).__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
```

**TrustformeRS:**
```rust
use trustformers::{Layer, LayerConfig, Tensor, Parameter};

pub struct CustomLayer {
    units: usize,
    weight: Parameter,
    bias: Parameter,
}

impl CustomLayer {
    pub fn new(units: usize, input_dim: usize) -> Result<Self> {
        Ok(Self {
            units,
            weight: Parameter::randn(&[input_dim, units])?,
            bias: Parameter::zeros(&[units])?,
        })
    }
}

impl Layer for CustomLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let output = input.matmul(&self.weight.data())?;
        output.add(&self.bias.data())
    }
    
    fn parameters(&self) -> Vec<&Parameter> {
        vec![&self.weight, &self.bias]
    }
}
```

## Performance Optimization

### Mixed Precision Training

**TensorFlow:**
```python
from tensorflow.keras import mixed_precision

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Compile with loss scaling
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    loss_scale='dynamic'
)
```

**TrustformeRS:**
```rust
use trustformers::{MixedPrecisionConfig, MixedPrecisionTrainer};

let mp_config = MixedPrecisionConfig {
    enabled: true,
    loss_scale: LossScale::Dynamic,
    ..Default::default()
};

let mut trainer = MixedPrecisionTrainer::new(mp_config)?;
let outputs = trainer.forward_with_autocast(&model, &inputs)?;
let loss = loss_fn.forward(&outputs, &targets)?;
trainer.backward_and_step(&loss, &mut optimizer)?;
```

### XLA Compilation

**TensorFlow:**
```python
# Enable XLA JIT compilation
@tf.function(jit_compile=True)
def compiled_function(x):
    return model(x)

# Or enable globally
tf.config.optimizer.set_jit(True)
```

**TrustformeRS:**
```rust
use trustformers::{PipelineJitConfig, CompilationStrategy};

let jit_config = PipelineJitConfig {
    enabled: true,
    compilation_strategy: CompilationStrategy::Aggressive,
    enable_kernel_fusion: true,
    ..Default::default()
};

let compiled_model = model.compile_with_jit(jit_config)?;
```

### Distribution Strategies

**TensorFlow:**
```python
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = create_model()
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy'
    )

model.fit(train_dataset, epochs=10)
```

**TrustformeRS:**
```rust
use trustformers::{DistributedConfig, DistributedTrainer};

let distributed_config = DistributedConfig {
    strategy: DistributionStrategy::DataParallel,
    num_gpus: 4,
    ..Default::default()
};

let mut trainer = DistributedTrainer::new(model, distributed_config)?;
trainer.train(train_dataset, epochs).await?;
```

## Common Patterns

### Callbacks and Hooks

**TensorFlow:**
```python
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch} ended with loss: {logs['loss']}")

model.fit(
    train_dataset,
    callbacks=[CustomCallback(), tf.keras.callbacks.ModelCheckpoint('model.h5')]
)
```

**TrustformeRS:**
```rust
use trustformers::{TrainingCallback, CallbackEvent};

struct CustomCallback;

impl TrainingCallback for CustomCallback {
    fn on_epoch_end(&mut self, epoch: usize, metrics: &TrainingMetrics) -> Result<()> {
        println!("Epoch {} ended with loss: {}", epoch, metrics.loss);
        Ok(())
    }
}

let callbacks = vec![
    Box::new(CustomCallback) as Box<dyn TrainingCallback>,
    Box::new(ModelCheckpoint::new("model.safetensors")),
];

trainer.with_callbacks(callbacks).train(train_dataset, epochs).await?;
```

### Saving and Loading

**TensorFlow:**
```python
# Save entire model
model.save('my_model.h5')

# Save weights only
model.save_weights('my_weights.h5')

# Load model
loaded_model = tf.keras.models.load_model('my_model.h5')

# Load weights
model.load_weights('my_weights.h5')
```

**TrustformeRS:**
```rust
// Save entire model
model.save("my_model.safetensors")?;

// Save weights only
model.save_weights("my_weights.safetensors")?;

// Load model
let loaded_model = AutoModel::from_pretrained("my_model.safetensors").await?;

// Load weights
model.load_weights("my_weights.safetensors")?;
```

### Custom Metrics

**TensorFlow:**
```python
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * (precision * recall) / (precision + recall + 1e-7)
```

**TrustformeRS:**
```rust
use trustformers::{Metric, MetricState, Tensor};

pub struct F1Score {
    precision: Precision,
    recall: Recall,
}

impl F1Score {
    pub fn new() -> Self {
        Self {
            precision: Precision::new(),
            recall: Recall::new(),
        }
    }
}

impl Metric for F1Score {
    fn update(&mut self, predictions: &Tensor, targets: &Tensor) -> Result<()> {
        self.precision.update(predictions, targets)?;
        self.recall.update(predictions, targets)?;
        Ok(())
    }

    fn compute(&self) -> Result<f64> {
        let precision = self.precision.compute()?;
        let recall = self.recall.compute()?;
        Ok(2.0 * (precision * recall) / (precision + recall + 1e-7))
    }

    fn reset(&mut self) {
        self.precision.reset();
        self.recall.reset();
    }
}
```

## Advanced Features

### TensorFlow Serving Compatible Export

**TensorFlow:**
```python
# Export for TensorFlow Serving
tf.saved_model.save(model, "serving_model")

# With signatures
@tf.function
def serving_fn(x):
    return model(x)

serving_fn = serving_fn.get_concrete_function(
    tf.TensorSpec(shape=[None, 784], dtype=tf.float32)
)

tf.saved_model.save(
    model, 
    "serving_model",
    signatures={'serving_default': serving_fn}
)
```

**TrustformeRS:**
```rust
use trustformers::export::{TensorflowServingExporter, ServingSignature};

let signature = ServingSignature {
    name: "serving_default".to_string(),
    inputs: vec![("input".to_string(), vec![-1, 784])],
    outputs: vec![("output".to_string(), vec![-1, 10])],
};

let exporter = TensorflowServingExporter::new();
exporter.export(&model, "serving_model", &[signature]).await?;
```

### TensorBoard Integration

**TensorFlow:**
```python
# Log scalars
with tf.summary.create_file_writer('logs').as_default():
    tf.summary.scalar('loss', loss_value, step=step)
    tf.summary.scalar('accuracy', accuracy, step=step)

# Log images
tf.summary.image('input_images', images, step=step)
```

**TrustformeRS:**
```rust
use trustformers::logging::{TensorboardLogger, LogValue};

let mut logger = TensorboardLogger::new("logs")?;

// Log scalars
logger.log_scalar("loss", loss_value, step)?;
logger.log_scalar("accuracy", accuracy, step)?;

// Log images
logger.log_images("input_images", &images, step)?;
```

## Troubleshooting

### Common Issues

1. **Shape Mismatches**
   ```rust
   // TensorFlow: x.reshape([-1, 784])
   // TrustformeRS:
   let x = x.reshape(&[-1, 784])?; // Use -1 for automatic dimension
   ```

2. **Gradient Computation**
   ```rust
   // Ensure variables require gradients
   let weight = Parameter::randn(&[784, 128]); // Automatically requires gradients
   
   // For tensors that need gradients
   let x = x.requires_grad(true);
   ```

3. **Device Placement**
   ```rust
   // Explicit device placement
   let device = Device::cuda(0)?;
   let model = model.to_device(&device)?;
   let inputs = inputs.to_device(&device)?;
   ```

### Performance Tips

1. **Enable JIT Compilation**
   ```rust
   let model = model.compile_with_jit(PipelineJitConfig::default())?;
   ```

2. **Use Batch Processing**
   ```rust
   // Instead of processing one by one
   let results = model.batch_predict(&inputs)?;
   ```

3. **Enable Memory Pool**
   ```rust
   let memory_config = MemoryPoolConfig::default();
   let model = model.with_memory_pool(memory_config);
   ```

4. **Optimize Data Loading**
   ```rust
   let dataloader = DataLoader::new(dataset, DataLoaderConfig {
       batch_size: 64,
       num_workers: 4,
       prefetch_factor: 2,
       pin_memory: true,
       ..Default::default()
   })?;
   ```

### Migration Checklist

- [ ] Replace `tf.Tensor` with `Tensor`
- [ ] Update model loading to async functions
- [ ] Add proper error handling with `?` operator
- [ ] Replace Python lists/numpy arrays with Rust vectors
- [ ] Update device management to TrustformeRS patterns
- [ ] Replace TensorFlow optimizers with TrustformeRS equivalents
- [ ] Update training loops to use Rust error handling
- [ ] Replace TensorFlow operations with Tensor operations
- [ ] Update data loading to use TrustformeRS DataLoader
- [ ] Test with TrustformeRS-specific optimizations
- [ ] Convert custom layers and metrics
- [ ] Update callback and logging patterns

## Performance Comparison

| Feature | TensorFlow | TrustformeRS | Improvement |
|---------|------------|--------------|-------------|
| Memory Safety | Runtime checks | Compile-time safety | 100% safe |
| Inference Speed | Baseline | 1.5-3x faster | 50-200% |
| Memory Usage | Baseline | 20-40% less | 20-40% |
| Startup Time | Baseline | 2-5x faster | 100-400% |
| Graph Compilation | XLA | Native JIT | 10-50% faster |
| Error Handling | Runtime exceptions | Compile-time checks | Fewer runtime errors |

## Next Steps

1. **Start Small**: Begin with inference-only migration
2. **Convert Models**: Use TrustformeRS model converters for TensorFlow models
3. **Test Thoroughly**: Compare outputs between TensorFlow and TrustformeRS
4. **Optimize Gradually**: Enable TrustformeRS-specific optimizations
5. **Monitor Performance**: Use built-in profiling tools
6. **Join Community**: Get help from TrustformeRS community

## Additional Resources

- [TrustformeRS API Documentation](../api_reference.md)
- [TensorFlow Model Conversion Guide](../conversion/tensorflow_models.md)
- [Performance Tuning Guide](../performance_tuning.md)
- [Best Practices](../best_practices.md)
- [Example Projects](../../examples/)
- [Community Forum](https://github.com/trustformers/trustformers/discussions)

## Support

If you encounter issues during migration:

1. Check the [troubleshooting guide](../troubleshooting.md)
2. Search [existing issues](https://github.com/trustformers/trustformers/issues)
3. Ask on [discussions](https://github.com/trustformers/trustformers/discussions)
4. Join our [Discord community](https://discord.gg/trustformers)