# ML Framework Integration Examples

This guide demonstrates how to integrate TrustformeRS Tokenizers with popular machine learning frameworks for seamless model training and inference workflows.

## Table of Contents

- [PyTorch Integration](#pytorch-integration)
- [TensorFlow Integration](#tensorflow-integration)
- [JAX/Flax Integration](#jax-flax-integration)
- [ONNX Integration](#onnx-integration)
- [Hugging Face Transformers](#hugging-face-transformers)
- [Production Deployment](#production-deployment)

## PyTorch Integration

TrustformeRS provides native PyTorch integration for efficient tensor operations and seamless model training.

### Basic PyTorch Integration

```rust
#[cfg(feature = "pytorch")]
use trustformers_tokenizers::{PyTorchTokenizer, PyTorchConfig, TensorDType, PaddingStrategy, TruncationStrategy};

#[cfg(feature = "pytorch")]
fn basic_pytorch_integration() -> Result<(), Box<dyn std::error::Error>> {
    // Configure PyTorch tokenizer
    let config = PyTorchConfig {
        device: "cuda:0".to_string(),
        dtype: TensorDType::Long,
        return_attention_mask: true,
        return_token_type_ids: true,
        max_length: Some(512),
        padding: Some(PaddingStrategy::MaxLength),
        truncation: Some(TruncationStrategy::LongestFirst),
    };
    
    let tokenizer = PyTorchTokenizer::from_pretrained("bert-base-uncased")?
        .with_config(config);
    
    // Single text encoding to tensors
    let text = "Hello, how are you today?";
    let encoding = tokenizer.encode_to_tensors(text)?;
    
    println!("Input IDs shape: {:?}", encoding.input_ids.size());
    println!("Attention mask shape: {:?}", encoding.attention_mask.size());
    
    // Batch encoding
    let texts = vec![
        "First example sentence.",
        "Second example with different length.",
        "Third example that is much longer and contains more words.",
    ];
    
    let batch = tokenizer.encode_batch_to_tensors(&texts)?;
    println!("Batch input IDs shape: {:?}", batch.input_ids.size()); // [batch_size, seq_len]
    println!("Batch attention mask shape: {:?}", batch.attention_mask.size());
    
    Ok(())
}

// PyTorch model training example
#[cfg(feature = "pytorch")]
fn pytorch_model_training() -> Result<(), Box<dyn std::error::Error>> {
    use tch::{nn, nn::Module, Device, Tensor};
    
    let device = Device::cuda_if_available();
    let tokenizer = PyTorchTokenizer::from_pretrained("bert-base-uncased")?
        .with_config(PyTorchConfig {
            device: device.to_string(),
            max_length: Some(128),
            padding: Some(PaddingStrategy::MaxLength),
            truncation: Some(TruncationStrategy::LongestFirst),
            return_attention_mask: true,
            ..Default::default()
        });
    
    // Create a simple classifier model
    let vs = nn::VarStore::new(device);
    let bert_hidden_size = 768;
    let num_classes = 2;
    
    let classifier = nn::seq()
        .add(nn::linear(&vs.root(), bert_hidden_size, 128, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::dropout(&vs.root(), 0.1))
        .add(nn::linear(&vs.root(), 128, num_classes, Default::default()));
    
    let mut opt = nn::Adam::default().build(&vs, 1e-4)?;
    
    // Training data
    let training_data = vec![
        ("This movie is great!", 1),
        ("I loved this film", 1),
        ("Terrible movie, waste of time", 0),
        ("Boring and predictable", 0),
        ("Amazing storyline and acting", 1),
        ("Disappointing ending", 0),
    ];
    
    // Training loop
    for epoch in 0..10 {
        let mut total_loss = 0.0;
        
        for (text, label) in &training_data {
            // Tokenize input
            let encoding = tokenizer.encode_to_tensors(text)?;
            let input_ids = encoding.input_ids.unsqueeze(0); // Add batch dimension
            let attention_mask = encoding.attention_mask.unsqueeze(0);
            
            // In a real scenario, you'd use BERT embeddings here
            // For this example, we'll use a simple embedding lookup
            let embeddings = input_ids.to_kind(tch::Kind::Float) * 0.1; // Dummy embeddings
            let pooled = embeddings.mean_dim(&[1], false, tch::Kind::Float); // Simple pooling
            
            // Forward pass
            let logits = classifier.forward(&pooled);
            let target = Tensor::of_slice(&[*label as i64]).to(device);
            
            // Compute loss
            let loss = logits.cross_entropy_for_logits(&target);
            
            // Backward pass
            opt.zero_grad();
            loss.backward();
            opt.step();
            
            total_loss += f64::from(&loss);
        }
        
        println!("Epoch {}: Average loss = {:.4}", epoch, total_loss / training_data.len() as f64);
    }
    
    // Inference example
    let test_text = "This is an amazing movie!";
    let encoding = tokenizer.encode_to_tensors(test_text)?;
    let input_ids = encoding.input_ids.unsqueeze(0);
    
    let embeddings = input_ids.to_kind(tch::Kind::Float) * 0.1;
    let pooled = embeddings.mean_dim(&[1], false, tch::Kind::Float);
    let logits = classifier.forward(&pooled);
    let prediction = logits.argmax(-1, false);
    
    println!("Test text: {}", test_text);
    println!("Prediction: {}", i64::from(&prediction));
    
    Ok(())
}

// PyTorch DataLoader integration
#[cfg(feature = "pytorch")]
fn pytorch_dataloader_integration() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::Arc;
    use tch::Tensor;
    
    let tokenizer = Arc::new(PyTorchTokenizer::from_pretrained("bert-base-uncased")?);
    
    // Custom dataset
    struct TextDataset {
        texts: Vec<String>,
        labels: Vec<i64>,
        tokenizer: Arc<PyTorchTokenizer>,
    }
    
    impl TextDataset {
        fn new(texts: Vec<String>, labels: Vec<i64>, tokenizer: Arc<PyTorchTokenizer>) -> Self {
            Self { texts, labels, tokenizer }
        }
        
        fn len(&self) -> usize {
            self.texts.len()
        }
        
        fn get_item(&self, idx: usize) -> Result<(Tensor, Tensor, i64), Box<dyn std::error::Error>> {
            let text = &self.texts[idx];
            let label = self.labels[idx];
            
            let encoding = self.tokenizer.encode_to_tensors(text)?;
            Ok((encoding.input_ids, encoding.attention_mask, label))
        }
    }
    
    // Create dataset
    let texts = vec![
        "Positive example text".to_string(),
        "Negative example text".to_string(),
        "Another positive example".to_string(),
        "Another negative example".to_string(),
    ];
    let labels = vec![1, 0, 1, 0];
    
    let dataset = TextDataset::new(texts, labels, tokenizer);
    
    // Batch processing
    let batch_size = 2;
    for batch_start in (0..dataset.len()).step_by(batch_size) {
        let batch_end = std::cmp::min(batch_start + batch_size, dataset.len());
        let mut batch_input_ids = Vec::new();
        let mut batch_attention_masks = Vec::new();
        let mut batch_labels = Vec::new();
        
        for idx in batch_start..batch_end {
            let (input_ids, attention_mask, label) = dataset.get_item(idx)?;
            batch_input_ids.push(input_ids);
            batch_attention_masks.push(attention_mask);
            batch_labels.push(label);
        }
        
        // Stack tensors to create batch
        let input_ids_batch = Tensor::stack(&batch_input_ids, 0);
        let attention_mask_batch = Tensor::stack(&batch_attention_masks, 0);
        let labels_batch = Tensor::of_slice(&batch_labels);
        
        println!("Batch {}: input shape {:?}, labels shape {:?}", 
                 batch_start / batch_size, 
                 input_ids_batch.size(), 
                 labels_batch.size());
        
        // Process batch...
    }
    
    Ok(())
}
```

## TensorFlow Integration

TrustformeRS provides comprehensive TensorFlow integration including tf.data pipelines and SavedModel export.

### Basic TensorFlow Integration

```rust
#[cfg(feature = "tensorflow")]
use trustformers_tokenizers::{TensorFlowTokenizer, TensorFlowConfig, TfDType, TfPaddingStrategy};

#[cfg(feature = "tensorflow")]
fn basic_tensorflow_integration() -> Result<(), Box<dyn std::error::Error>> {
    let config = TensorFlowConfig {
        dtype: TfDType::Int32,
        padding_strategy: TfPaddingStrategy::Longest,
        max_length: Some(256),
        return_ragged: false,
        return_attention_mask: true,
    };
    
    let tokenizer = TensorFlowTokenizer::from_pretrained("bert-base-uncased")?
        .with_config(config);
    
    // Single text encoding
    let text = "TensorFlow integration example";
    let tf_tensors = tokenizer.encode_to_tf_tensors(text)?;
    
    println!("TensorFlow tensor shape: {:?}", tf_tensors.input_ids.shape());
    
    // Batch encoding
    let texts = vec![
        "First example",
        "Second example with more words",
        "Third example",
    ];
    
    let batch_tensors = tokenizer.encode_batch_to_tf_tensors(&texts)?;
    println!("Batch tensor shape: {:?}", batch_tensors.input_ids.shape());
    
    Ok(())
}

// TensorFlow tf.data pipeline integration
#[cfg(feature = "tensorflow")]
fn tensorflow_data_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    use tensorflow::{DataType, Graph, Session, SessionOptions, SessionRunArgs, Tensor};
    
    let tokenizer = TensorFlowTokenizer::from_pretrained("bert-base-uncased")?;
    
    // Sample text data
    let texts = vec![
        "Sample text for TensorFlow pipeline",
        "Another example with different length",
        "Short text",
        "Much longer text that demonstrates the padding and truncation capabilities",
    ];
    
    // Create tf.data dataset
    let tf_dataset = tokenizer.create_tf_dataset(&texts)?;
    
    // Configure dataset
    let processed_dataset = tf_dataset
        .batch(2)?
        .prefetch(1)?
        .repeat(None)?; // Repeat indefinitely
    
    // Create iterator
    let iterator = processed_dataset.make_one_shot_iterator()?;
    let next_element = iterator.get_next()?;
    
    // TensorFlow session
    let graph = Graph::new();
    let session = Session::new(&SessionOptions::new(), &graph)?;
    
    // Training loop simulation
    for epoch in 0..3 {
        println!("Epoch {}", epoch);
        
        for batch_idx in 0..2 { // Process 2 batches per epoch
            let mut run_args = SessionRunArgs::new();
            
            // Fetch next batch
            let input_ids_tensor = next_element.input_ids.clone();
            let attention_mask_tensor = next_element.attention_mask.clone();
            
            run_args.add_fetch(&input_ids_tensor, 0);
            run_args.add_fetch(&attention_mask_tensor, 1);
            
            session.run(&mut run_args)?;
            
            let input_ids: Tensor<i32> = run_args.fetch(0)?;
            let attention_mask: Tensor<i32> = run_args.fetch(1)?;
            
            println!("  Batch {}: input_ids shape {:?}, attention_mask shape {:?}", 
                     batch_idx, input_ids.shape(), attention_mask.shape());
            
            // Process batch (model forward pass, loss computation, etc.)
        }
    }
    
    Ok(())
}

// TensorFlow SavedModel export
#[cfg(feature = "tensorflow")]
fn tensorflow_savedmodel_export() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = TensorFlowTokenizer::from_pretrained("bert-base-uncased")?;
    
    // Export configuration
    let export_config = TfServingExportConfig {
        model_name: "text_classifier".to_string(),
        version: 1,
        signature_name: "serving_default".to_string(),
        include_preprocessing: true,
        optimize_for_inference: true,
    };
    
    // Export tokenizer as part of SavedModel
    tokenizer.export_for_serving("./exported_model", export_config)?;
    
    println!("Model exported to ./exported_model");
    
    // The exported model can be served with TensorFlow Serving:
    // tensorflow_model_server --model_base_path=./exported_model --model_name=text_classifier
    
    Ok(())
}

// TensorFlow Keras integration
#[cfg(feature = "tensorflow")]
fn tensorflow_keras_integration() -> Result<(), Box<dyn std::error::Error>> {
    use tensorflow::{Graph, Operation, Session, SessionOptions, SessionRunArgs, Tensor};
    
    let tokenizer = TensorFlowTokenizer::from_pretrained("bert-base-uncased")?;
    
    // Build Keras-style model graph
    let mut graph = Graph::new();
    let vocab_size = tokenizer.get_vocab_size();
    let hidden_size = 128;
    let num_classes = 2;
    
    // Input placeholders
    let input_ids = graph.new_operation("Placeholder", "input_ids")?
        .set_attr_type("dtype", DataType::Int32)?
        .set_attr_shape("shape", &[-1, -1])?  // [batch_size, seq_len]
        .finish()?;
    
    let attention_mask = graph.new_operation("Placeholder", "attention_mask")?
        .set_attr_type("dtype", DataType::Int32)?
        .set_attr_shape("shape", &[-1, -1])?
        .finish()?;
    
    // Embedding layer (simplified)
    let embedding_weights = graph.new_operation("Variable", "embedding_weights")?
        .set_attr_type("dtype", DataType::Float)?
        .set_attr_shape("shape", &[vocab_size as i64, hidden_size as i64])?
        .finish()?;
    
    let embeddings = graph.new_operation("Gather", "embeddings")?
        .add_input(embedding_weights.clone())
        .add_input(input_ids.clone())
        .finish()?;
    
    // Global average pooling
    let pooled = graph.new_operation("ReduceMean", "pooled")?
        .add_input(embeddings)
        .set_attr_int_list("reduction_indices", &[1])?  // Pool over sequence dimension
        .set_attr_bool("keep_dims", false)?
        .finish()?;
    
    // Dense layer
    let dense_weights = graph.new_operation("Variable", "dense_weights")?
        .set_attr_type("dtype", DataType::Float)?
        .set_attr_shape("shape", &[hidden_size as i64, num_classes as i64])?
        .finish()?;
    
    let dense_bias = graph.new_operation("Variable", "dense_bias")?
        .set_attr_type("dtype", DataType::Float)?
        .set_attr_shape("shape", &[num_classes as i64])?
        .finish()?;
    
    let logits = graph.new_operation("Add", "logits")?
        .add_input(
            graph.new_operation("MatMul", "dense_matmul")?
                .add_input(pooled)
                .add_input(dense_weights)
                .finish()?
        )
        .add_input(dense_bias)
        .finish()?;
    
    // Training setup
    let labels = graph.new_operation("Placeholder", "labels")?
        .set_attr_type("dtype", DataType::Int32)?
        .set_attr_shape("shape", &[-1])?
        .finish()?;
    
    let loss = graph.new_operation("SparseSoftmaxCrossEntropyWithLogits", "loss")?
        .add_input(logits.clone())
        .add_input(labels.clone())
        .finish()?;
    
    let mean_loss = graph.new_operation("ReduceMean", "mean_loss")?
        .add_input(loss)
        .finish()?;
    
    // Create session and initialize variables
    let session = Session::new(&SessionOptions::new(), &graph)?;
    
    // Initialize variables (simplified)
    let mut init_args = SessionRunArgs::new();
    // In a real implementation, you'd properly initialize the variables
    
    // Training data
    let training_texts = vec![
        "Positive example",
        "Negative example", 
        "Another positive",
        "Another negative",
    ];
    let training_labels = vec![1, 0, 1, 0];
    
    // Training loop
    for epoch in 0..5 {
        let mut total_loss = 0.0;
        
        for (text, label) in training_texts.iter().zip(&training_labels) {
            let encoding = tokenizer.encode_to_tf_tensors(text)?;
            
            let mut run_args = SessionRunArgs::new();
            
            // Feed inputs
            run_args.add_feed(&input_ids, 0, &encoding.input_ids);
            run_args.add_feed(&attention_mask, 0, &encoding.attention_mask);
            run_args.add_feed(&labels, 0, &Tensor::from(*label));
            
            // Fetch loss
            run_args.add_fetch(&mean_loss, 0);
            
            session.run(&mut run_args)?;
            
            let loss_value: f32 = run_args.fetch::<f32>(0)?[0];
            total_loss += loss_value;
        }
        
        println!("Epoch {}: Average loss = {:.4}", epoch, total_loss / training_texts.len() as f32);
    }
    
    Ok(())
}
```

## JAX/Flax Integration

TrustformeRS provides JAX integration with XLA compilation support for high-performance training and inference.

### Basic JAX Integration

```rust
#[cfg(feature = "jax")]
use trustformers_tokenizers::{JaxTokenizer, JaxConfig, JaxDType, JaxDevice, JaxPaddingStrategy};

#[cfg(feature = "jax")]
fn basic_jax_integration() -> Result<(), Box<dyn std::error::Error>> {
    let config = JaxConfig {
        dtype: JaxDType::Int32,
        device: JaxDevice::GPU(0),
        padding_strategy: JaxPaddingStrategy::Longest,
        max_length: Some(128),
        return_attention_mask: true,
    };
    
    let tokenizer = JaxTokenizer::from_pretrained("bert-base-uncased")?
        .with_config(config);
    
    // Single text encoding
    let text = "JAX integration example with XLA compilation";
    let jax_arrays = tokenizer.encode_to_jax_arrays(text)?;
    
    println!("JAX array shape: {:?}", jax_arrays.input_ids.shape());
    println!("Device: {:?}", jax_arrays.input_ids.device());
    
    // Batch encoding
    let texts = vec![
        "First JAX example",
        "Second example with XLA optimization",
        "Third example for batch processing",
    ];
    
    let batch_arrays = tokenizer.encode_batch_to_jax_arrays(&texts)?;
    println!("Batch shape: {:?}", batch_arrays.input_ids.shape());
    
    Ok(())
}

// JAX compiled tokenization
#[cfg(feature = "jax")]
fn jax_compiled_tokenization() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = JaxTokenizer::from_pretrained("bert-base-uncased")?;
    
    // Compile tokenization function with XLA
    let compiled_tokenizer = tokenizer.compile(JaxCompilationConfig {
        static_argnums: vec![],
        donate_argnums: vec![],
        backend: JaxBackend::GPU,
        optimization_level: JaxOptimizationLevel::O3,
    })?;
    
    // Warm-up compilation
    let warmup_text = "Compilation warmup text";
    let _ = compiled_tokenizer.encode(warmup_text)?;
    
    // Benchmark compiled vs non-compiled
    let test_texts: Vec<String> = (0..1000)
        .map(|i| format!("Test text number {} for benchmarking", i))
        .collect();
    
    // Non-compiled baseline
    let start = std::time::Instant::now();
    for text in &test_texts {
        let _ = tokenizer.encode_to_jax_arrays(text)?;
    }
    let non_compiled_time = start.elapsed();
    
    // Compiled version
    let start = std::time::Instant::now();
    for text in &test_texts {
        let _ = compiled_tokenizer.encode(text)?;
    }
    let compiled_time = start.elapsed();
    
    println!("Non-compiled: {:?}", non_compiled_time);
    println!("Compiled: {:?}", compiled_time);
    println!("Speedup: {:.2}x", non_compiled_time.as_secs_f64() / compiled_time.as_secs_f64());
    
    Ok(())
}

// JAX/Flax model training
#[cfg(feature = "jax")]
fn jax_flax_model_training() -> Result<(), Box<dyn std::error::Error>> {
    use jax::{Array, Device};
    use flax::{linen as nn, training as flax_training};
    
    let tokenizer = JaxTokenizer::from_pretrained("bert-base-uncased")?;
    
    // Define Flax model
    #[derive(Clone)]
    struct TextClassifier {
        vocab_size: usize,
        hidden_size: usize,
        num_classes: usize,
    }
    
    impl nn::Module for TextClassifier {
        fn setup(&self) -> Self::ModuleSetup {
            nn::ModuleSetup::new()
                .add_submodule("embedding", nn::Embed::new(self.vocab_size, self.hidden_size))
                .add_submodule("dense", nn::Dense::new(self.num_classes))
        }
        
        fn __call__(&self, input_ids: &Array, attention_mask: &Array) -> Array {
            let embeddings = self.embedding.apply(input_ids);
            
            // Simple global average pooling with attention mask
            let masked_embeddings = embeddings * attention_mask.expand_dims(-1);
            let pooled = masked_embeddings.sum_axis(-2) / attention_mask.sum_axis(-1).expand_dims(-1);
            
            self.dense.apply(&pooled)
        }
    }
    
    let model = TextClassifier {
        vocab_size: tokenizer.get_vocab_size(),
        hidden_size: 128,
        num_classes: 2,
    };
    
    // Initialize model parameters
    let key = jax::random::PRNGKey::new(42);
    let dummy_input_ids = Array::zeros(&[1, 10], jax::DType::Int32);
    let dummy_attention_mask = Array::ones(&[1, 10], jax::DType::Int32);
    
    let params = model.init(&key, &dummy_input_ids, &dummy_attention_mask)?;
    
    // Training data
    let training_data = vec![
        ("Positive sentiment example", 1),
        ("Negative sentiment example", 0),
        ("Another positive example", 1),
        ("Another negative example", 0),
    ];
    
    // Training configuration
    let learning_rate = 1e-3;
    let optimizer = flax_training::adam(learning_rate);
    let mut opt_state = optimizer.init(&params);
    
    // Training function (JIT compiled)
    let train_step = jax::jit(|params, opt_state, input_ids, attention_mask, labels| {
        let loss_fn = |params| {
            let logits = model.apply(params, input_ids, attention_mask);
            nn::sparse_categorical_crossentropy(&logits, labels).mean()
        };
        
        let (loss, grads) = jax::value_and_grad(loss_fn)(params);
        let (updates, new_opt_state) = optimizer.update(&grads, &opt_state, &params);
        let new_params = optax::apply_updates(&params, &updates);
        
        (new_params, new_opt_state, loss)
    });
    
    // Training loop
    for epoch in 0..10 {
        let mut total_loss = 0.0;
        
        for (text, label) in &training_data {
            let encoding = tokenizer.encode_to_jax_arrays(text)?;
            let input_ids = encoding.input_ids.expand_dims(0); // Add batch dimension
            let attention_mask = encoding.attention_mask.expand_dims(0);
            let labels = Array::from_vec(vec![*label], &[1]);
            
            let (new_params, new_opt_state, loss) = train_step(
                &params, &opt_state, &input_ids, &attention_mask, &labels
            );
            
            params = new_params;
            opt_state = new_opt_state;
            total_loss += loss.to_scalar::<f32>();
        }
        
        println!("Epoch {}: Average loss = {:.4}", epoch, total_loss / training_data.len() as f32);
    }
    
    Ok(())
}

// JAX distributed training
#[cfg(feature = "jax")]
fn jax_distributed_training() -> Result<(), Box<dyn std::error::Error>> {
    use jax::{Array, Device, sharding};
    
    let tokenizer = JaxTokenizer::from_pretrained("bert-base-uncased")?;
    
    // Setup device mesh for multi-GPU training
    let devices = jax::devices();
    let mesh = sharding::create_device_mesh(&devices, &[2, 2])?; // 2x2 mesh
    
    // Configure sharding for model parallelism
    let sharding_config = JaxSharding::new()
        .with_data_parallel_axis("batch")
        .with_model_parallel_axis("hidden")
        .with_mesh(mesh);
    
    let distributed_tokenizer = tokenizer.with_sharding(sharding_config);
    
    // Large batch for distributed processing
    let large_batch: Vec<String> = (0..1000)
        .map(|i| format!("Distributed training example {}", i))
        .collect();
    
    // Process batch across multiple devices
    let sharded_batch = distributed_tokenizer.encode_batch_distributed(&large_batch)?;
    
    println!("Distributed batch processing completed");
    println!("Sharded across {} devices", devices.len());
    
    Ok(())
}
```

## ONNX Integration

TrustformeRS supports ONNX export for cross-platform inference and deployment.

### ONNX Model Export

```rust
#[cfg(feature = "onnx")]
use trustformers_tokenizers::{OnnxTokenizerExporter, OnnxExportConfig, OnnxOptimizationLevel};

#[cfg(feature = "onnx")]
fn onnx_model_export() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = TokenizerImpl::from_pretrained("bert-base-uncased")?;
    
    let export_config = OnnxExportConfig {
        opset_version: 11,
        optimization_level: OnnxOptimizationLevel::All,
        target_platform: OnnxTargetPlatform::CPU,
        include_tokenizer: true,
        include_preprocessing: true,
        dynamic_axes: vec![
            ("input_text".to_string(), vec![0]), // Dynamic batch size
        ],
        metadata: OnnxModelMetadata {
            model_name: "bert_tokenizer".to_string(),
            description: "BERT tokenizer for text classification".to_string(),
            version: "1.0".to_string(),
            author: "TrustformeRS".to_string(),
        },
    };
    
    let exporter = OnnxTokenizerExporter::new();
    
    // Export tokenizer to ONNX format
    exporter.export_tokenizer(&tokenizer, "bert_tokenizer.onnx", export_config)?;
    
    println!("Tokenizer exported to bert_tokenizer.onnx");
    
    // Validate exported model
    let validation_result = exporter.validate_exported_model("bert_tokenizer.onnx")?;
    
    if validation_result.is_valid {
        println!("ONNX model validation passed");
        println!("Input names: {:?}", validation_result.input_names);
        println!("Output names: {:?}", validation_result.output_names);
    } else {
        println!("Validation issues: {:?}", validation_result.issues);
    }
    
    Ok(())
}

// ONNX Runtime inference
#[cfg(feature = "onnx")]
fn onnx_runtime_inference() -> Result<(), Box<dyn std::error::Error>> {
    use onnxruntime::{environment::Environment, LoggingLevel, SessionBuilder};
    
    // Create ONNX Runtime environment
    let environment = Environment::builder()
        .with_name("tokenizer_inference")
        .with_log_level(LoggingLevel::Warning)
        .build()?;
    
    // Load the exported model
    let session = SessionBuilder::new(&environment)?
        .with_optimization_level(onnxruntime::GraphOptimizationLevel::All)?
        .with_number_threads(4)?
        .with_model_from_file("bert_tokenizer.onnx")?;
    
    // Input text
    let input_text = "This is a test sentence for ONNX inference";
    
    // Prepare input tensor
    let input_tensor = onnxruntime::ndarray::Array1::from_vec(
        input_text.chars().map(|c| c as i64).collect()
    ).into_dyn();
    
    // Run inference
    let outputs = session.run(vec![input_tensor])?;
    
    // Extract token IDs
    let token_ids = outputs[0].try_extract::<i64>()?;
    println!("Token IDs: {:?}", token_ids.view());
    
    // Extract attention mask
    let attention_mask = outputs[1].try_extract::<i64>()?;
    println!("Attention mask: {:?}", attention_mask.view());
    
    Ok(())
}

// ONNX deployment pipeline
#[cfg(feature = "onnx")]
fn onnx_deployment_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = TokenizerImpl::from_pretrained("bert-base-uncased")?;
    
    // Export for different target platforms
    let platforms = vec![
        (OnnxTargetPlatform::CPU, "bert_tokenizer_cpu.onnx"),
        (OnnxTargetPlatform::GPU, "bert_tokenizer_gpu.onnx"),
        (OnnxTargetPlatform::EdgeTPU, "bert_tokenizer_edge.onnx"),
    ];
    
    let exporter = OnnxTokenizerExporter::new();
    
    for (platform, filename) in platforms {
        let config = OnnxExportConfig {
            target_platform: platform,
            optimization_level: match platform {
                OnnxTargetPlatform::CPU => OnnxOptimizationLevel::All,
                OnnxTargetPlatform::GPU => OnnxOptimizationLevel::Basic,
                OnnxTargetPlatform::EdgeTPU => OnnxOptimizationLevel::Extended,
            },
            opset_version: 11,
            ..Default::default()
        };
        
        exporter.export_tokenizer(&tokenizer, filename, config)?;
        println!("Exported for {:?}: {}", platform, filename);
        
        // Platform-specific optimizations
        match platform {
            OnnxTargetPlatform::CPU => {
                exporter.optimize_for_cpu(filename)?;
            },
            OnnxTargetPlatform::GPU => {
                exporter.optimize_for_gpu(filename)?;
            },
            OnnxTargetPlatform::EdgeTPU => {
                exporter.quantize_for_edge(filename)?;
            },
        }
    }
    
    // Create deployment package
    let deployment_config = DeploymentConfig {
        model_files: vec![
            "bert_tokenizer_cpu.onnx".to_string(),
            "bert_tokenizer_gpu.onnx".to_string(),
        ],
        config_file: "tokenizer_config.json".to_string(),
        vocabulary_file: "vocab.txt".to_string(),
        runtime_requirements: RuntimeRequirements {
            min_onnx_version: "1.8.0".to_string(),
            supported_providers: vec!["CPUExecutionProvider", "CUDAExecutionProvider"],
        },
    };
    
    exporter.create_deployment_package("deployment_package/", deployment_config)?;
    println!("Deployment package created in deployment_package/");
    
    Ok(())
}
```

## Hugging Face Transformers

Seamless integration with Hugging Face Transformers library for easy model usage.

### Transformers Integration

```rust
use trustformers_tokenizers::{TokenizerImpl, HuggingFaceCompat};

fn transformers_integration() -> Result<(), Box<dyn std::error::Error>> {
    // Load tokenizer with HuggingFace compatibility
    let tokenizer = TokenizerImpl::from_pretrained("bert-base-uncased")?
        .with_huggingface_compat(true);
    
    // Text classification pipeline
    let texts = vec![
        "I love this movie!",
        "This film is terrible.",
        "Amazing storyline and great acting.",
        "Boring and predictable plot.",
    ];
    
    // Tokenize for model input
    let encoded_batch = tokenizer.encode_batch(&texts)?;
    
    // Convert to format compatible with transformers
    let model_inputs = HuggingFaceCompat::create_model_inputs(&encoded_batch)?;
    
    println!("Model inputs prepared for transformers:");
    println!("Input IDs shape: {:?}", model_inputs.input_ids.len());
    println!("Attention mask shape: {:?}", model_inputs.attention_mask.len());
    
    // Simulate model inference (in real usage, you'd use transformers library)
    let predictions = simulate_bert_inference(&model_inputs)?;
    
    for (text, prediction) in texts.iter().zip(predictions) {
        println!("Text: {} -> Prediction: {:.3}", text, prediction);
    }
    
    Ok(())
}

// Helper function to simulate BERT inference
fn simulate_bert_inference(inputs: &ModelInputs) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // In real usage, this would be actual model inference
    // For demo purposes, we'll return random predictions
    use rand::Rng;
    
    let mut rng = rand::thread_rng();
    let predictions: Vec<f32> = (0..inputs.input_ids.len())
        .map(|_| rng.gen_range(0.0..1.0))
        .collect();
    
    Ok(predictions)
}

// Custom model integration
fn custom_model_integration() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer = TokenizerImpl::from_pretrained("distilbert-base-uncased")?;
    
    // Define custom model pipeline
    struct CustomTextClassifier {
        tokenizer: TokenizerImpl,
        model_weights: Vec<f32>, // Simplified representation
    }
    
    impl CustomTextClassifier {
        fn new(tokenizer: TokenizerImpl) -> Self {
            // Initialize with dummy weights
            let vocab_size = tokenizer.get_vocab_size();
            let model_weights = vec![0.1; vocab_size * 2]; // Simplified
            
            Self { tokenizer, model_weights }
        }
        
        fn classify(&self, text: &str) -> Result<f32, Box<dyn std::error::Error>> {
            // Tokenize input
            let encoded = self.tokenizer.encode(text)?;
            
            // Simple scoring (in real usage, this would be actual model inference)
            let score = encoded.ids().iter()
                .map(|&id| self.model_weights.get(id as usize).unwrap_or(&0.0))
                .sum::<f32>() / encoded.ids().len() as f32;
            
            Ok(score)
        }
        
        fn classify_batch(&self, texts: &[&str]) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            let mut scores = Vec::new();
            for text in texts {
                scores.push(self.classify(text)?);
            }
            Ok(scores)
        }
    }
    
    let classifier = CustomTextClassifier::new(tokenizer);
    
    let test_texts = vec![
        "This is a positive example",
        "This is a negative example",
        "Neutral text example",
    ];
    
    let scores = classifier.classify_batch(&test_texts)?;
    
    for (text, score) in test_texts.iter().zip(scores) {
        println!("Text: {} -> Score: {:.3}", text, score);
    }
    
    Ok(())
}
```

## Production Deployment

Examples for deploying tokenizers in production environments.

### Microservice Deployment

```rust
use trustformers_tokenizers::{TokenizerImpl, PerformanceProfiler};
use std::sync::Arc;
use tokio::sync::RwLock;

// Production tokenizer service
#[derive(Clone)]
struct TokenizerService {
    tokenizer: Arc<TokenizerImpl>,
    profiler: Arc<PerformanceProfiler>,
    cache: Arc<RwLock<std::collections::HashMap<String, TokenizedInput>>>,
}

impl TokenizerService {
    pub async fn new(model_name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let tokenizer = Arc::new(TokenizerImpl::from_pretrained(model_name)?);
        let profiler = Arc::new(PerformanceProfiler::new());
        let cache = Arc::new(RwLock::new(std::collections::HashMap::new()));
        
        Ok(Self {
            tokenizer,
            profiler,
            cache,
        })
    }
    
    pub async fn tokenize(&self, text: &str) -> Result<TokenizedInput, Box<dyn std::error::Error>> {
        // Check cache first
        {
            let cache_read = self.cache.read().await;
            if let Some(cached_result) = cache_read.get(text) {
                return Ok(cached_result.clone());
            }
        }
        
        // Profile the tokenization
        let start = std::time::Instant::now();
        let result = self.tokenizer.encode(text)?;
        let duration = start.elapsed();
        
        // Record metrics
        self.profiler.record_timing("tokenize", duration);
        self.profiler.record_throughput("tokens_per_second", result.tokens().len(), duration);
        
        // Cache result
        {
            let mut cache_write = self.cache.write().await;
            cache_write.insert(text.to_string(), result.clone());
            
            // Limit cache size
            if cache_write.len() > 10000 {
                cache_write.clear(); // Simple eviction strategy
            }
        }
        
        Ok(result)
    }
    
    pub async fn tokenize_batch(&self, texts: &[String]) -> Result<Vec<TokenizedInput>, Box<dyn std::error::Error>> {
        let start = std::time::Instant::now();
        
        // Process batch in parallel
        let futures: Vec<_> = texts.iter()
            .map(|text| self.tokenize(text))
            .collect();
        
        let results = futures_util::future::try_join_all(futures).await?;
        let duration = start.elapsed();
        
        // Record batch metrics
        let total_tokens: usize = results.iter().map(|r| r.tokens().len()).sum();
        self.profiler.record_timing("tokenize_batch", duration);
        self.profiler.record_throughput("batch_tokens_per_second", total_tokens, duration);
        
        Ok(results)
    }
    
    pub fn get_metrics(&self) -> String {
        self.profiler.generate_report()
    }
}

// REST API server
#[cfg(feature = "web")]
async fn tokenizer_web_service() -> Result<(), Box<dyn std::error::Error>> {
    use warp::Filter;
    use serde::{Deserialize, Serialize};
    
    #[derive(Deserialize)]
    struct TokenizeRequest {
        text: String,
        model: Option<String>,
    }
    
    #[derive(Deserialize)]
    struct BatchTokenizeRequest {
        texts: Vec<String>,
        model: Option<String>,
    }
    
    #[derive(Serialize)]
    struct TokenizeResponse {
        tokens: Vec<String>,
        ids: Vec<u32>,
        attention_mask: Vec<u32>,
    }
    
    // Initialize services for different models
    let bert_service = Arc::new(TokenizerService::new("bert-base-uncased").await?);
    let distilbert_service = Arc::new(TokenizerService::new("distilbert-base-uncased").await?);
    
    // Single tokenization endpoint
    let tokenize = warp::path("tokenize")
        .and(warp::post())
        .and(warp::body::json())
        .and(warp::any().map(move || bert_service.clone()))
        .and_then(|req: TokenizeRequest, service: Arc<TokenizerService>| async move {
            match service.tokenize(&req.text).await {
                Ok(result) => {
                    let response = TokenizeResponse {
                        tokens: result.tokens().clone(),
                        ids: result.ids().clone(),
                        attention_mask: result.attention_mask().clone(),
                    };
                    Ok(warp::reply::json(&response))
                },
                Err(e) => Err(warp::reject::custom(TokenizerError::new(e.to_string())))
            }
        });
    
    // Batch tokenization endpoint
    let batch_tokenize = warp::path("batch_tokenize")
        .and(warp::post())
        .and(warp::body::json())
        .and(warp::any().map(move || bert_service.clone()))
        .and_then(|req: BatchTokenizeRequest, service: Arc<TokenizerService>| async move {
            match service.tokenize_batch(&req.texts).await {
                Ok(results) => {
                    let responses: Vec<TokenizeResponse> = results.into_iter().map(|result| {
                        TokenizeResponse {
                            tokens: result.tokens().clone(),
                            ids: result.ids().clone(),
                            attention_mask: result.attention_mask().clone(),
                        }
                    }).collect();
                    Ok(warp::reply::json(&responses))
                },
                Err(e) => Err(warp::reject::custom(TokenizerError::new(e.to_string())))
            }
        });
    
    // Metrics endpoint
    let metrics = warp::path("metrics")
        .and(warp::get())
        .and(warp::any().map(move || bert_service.clone()))
        .and_then(|service: Arc<TokenizerService>| async move {
            let metrics = service.get_metrics();
            Ok::<_, warp::Rejection>(warp::reply::html(metrics))
        });
    
    let routes = tokenize
        .or(batch_tokenize)
        .or(metrics)
        .with(warp::cors().allow_any_origin());
    
    println!("Starting tokenizer service on http://localhost:3030");
    warp::serve(routes)
        .run(([127, 0, 0, 1], 3030))
        .await;
    
    Ok(())
}

// Load balancing and auto-scaling
#[cfg(feature = "cluster")]
async fn distributed_tokenizer_service() -> Result<(), Box<dyn std::error::Error>> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    
    struct LoadBalancer {
        services: Vec<Arc<TokenizerService>>,
        current: AtomicUsize,
    }
    
    impl LoadBalancer {
        fn new(services: Vec<Arc<TokenizerService>>) -> Self {
            Self {
                services,
                current: AtomicUsize::new(0),
            }
        }
        
        fn get_service(&self) -> Arc<TokenizerService> {
            let index = self.current.fetch_add(1, Ordering::Relaxed) % self.services.len();
            self.services[index].clone()
        }
        
        async fn scale_up(&mut self) -> Result<(), Box<dyn std::error::Error>> {
            let new_service = Arc::new(TokenizerService::new("bert-base-uncased").await?);
            self.services.push(new_service);
            println!("Scaled up to {} services", self.services.len());
            Ok(())
        }
    }
    
    // Initialize load balancer with multiple services
    let initial_services = vec![
        Arc::new(TokenizerService::new("bert-base-uncased").await?),
        Arc::new(TokenizerService::new("bert-base-uncased").await?),
    ];
    
    let mut load_balancer = LoadBalancer::new(initial_services);
    
    // Auto-scaling based on load
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
            
            // Check load metrics and scale if needed
            // This is a simplified example
            let current_load = get_current_load().await;
            if current_load > 0.8 {
                if let Err(e) = load_balancer.scale_up().await {
                    eprintln!("Failed to scale up: {}", e);
                }
            }
        }
    });
    
    // Main request handling loop
    loop {
        let service = load_balancer.get_service();
        
        // Handle incoming requests
        // This would integrate with your web framework
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }
}

async fn get_current_load() -> f64 {
    // Simplified load calculation
    0.5 // Return dummy value
}

#[derive(Debug)]
struct TokenizerError {
    message: String,
}

impl TokenizerError {
    fn new(message: String) -> Self {
        Self { message }
    }
}

impl warp::reject::Reject for TokenizerError {}
```

This comprehensive guide shows how to integrate TrustformeRS Tokenizers with various ML frameworks and deploy them in production environments. Each framework integration provides optimized paths for high-performance training and inference.