//! Demonstration of checkpoint format conversion between frameworks
#![allow(unused_variables)]

use anyhow::Result;
use trustformers_core::checkpoint::{
    compare_checkpoints, convert_checkpoint, detect_format, get_checkpoint_info, load_checkpoint,
    merge_checkpoints, shard_checkpoint, Checkpoint, CheckpointConverter,
    CheckpointConverterBuilder, CheckpointFormat, ModelType, PyTorchCheckpoint,
    TensorFlowCheckpoint, TrustformersCheckpoint, WeightMapping, WeightTensor,
};

#[tokio::main]
async fn main() -> Result<()> {
    println!("TrustformeRS Checkpoint Converter Demo");
    println!("=====================================\n");

    // Example 1: Basic conversion
    println!("Example 1: Basic Checkpoint Conversion");
    demo_basic_conversion().await?;

    // Example 2: Model-specific conversion
    println!("\nExample 2: Model-Specific Conversion (BERT)");
    demo_bert_conversion().await?;

    // Example 3: Custom mappings
    println!("\nExample 3: Custom Weight Mappings");
    demo_custom_mappings().await?;

    // Example 4: Checkpoint utilities
    println!("\nExample 4: Checkpoint Utilities");
    demo_checkpoint_utils()?;

    // Example 5: Batch conversion
    println!("\nExample 5: Batch Conversion");
    demo_batch_conversion().await?;

    // Example 6: Checkpoint validation
    println!("\nExample 6: Checkpoint Validation");
    demo_checkpoint_validation().await?;

    Ok(())
}

async fn demo_basic_conversion() -> Result<()> {
    // Create a sample PyTorch checkpoint
    let mut pytorch_checkpoint = PyTorchCheckpoint::new();

    // Add some sample weights
    pytorch_checkpoint.set_weight(
        "model.embeddings.word_embeddings.weight",
        WeightTensor::new(vec![0.1; 768 * 50000], vec![50000, 768]),
    )?;
    pytorch_checkpoint.set_weight(
        "model.embeddings.position_embeddings.weight",
        WeightTensor::new(vec![0.2; 768 * 512], vec![512, 768]),
    )?;
    pytorch_checkpoint.set_weight(
        "model.encoder.layer.0.attention.self.query.weight",
        WeightTensor::new(vec![0.3; 768 * 768], vec![768, 768]),
    )?;

    // Save the checkpoint
    let temp_dir = tempfile::tempdir()?;
    let pytorch_path = temp_dir.path().join("model.pt");
    pytorch_checkpoint.save(&pytorch_path)?;

    // Convert to TensorFlow format
    let tf_path = temp_dir.path().join("model.ckpt");
    let result = convert_checkpoint(&pytorch_path, &tf_path, CheckpointFormat::TensorFlow).await?;

    println!("Conversion completed:");
    println!("  Source format: {:?}", result.source_format);
    println!("  Target format: {:?}", result.target_format);
    println!("  Weights converted: {}", result.weights_converted);
    println!("  Conversion time: {}ms", result.conversion_time_ms);

    Ok(())
}

async fn demo_bert_conversion() -> Result<()> {
    println!("Creating BERT-specific converter...");

    // Create converter with BERT-specific mappings
    let converter = CheckpointConverterBuilder::new()
        .model_type(ModelType::BERT)
        .validate(true)
        .parallel(true)
        .build();

    // Create sample BERT checkpoint
    let mut bert_checkpoint = PyTorchCheckpoint::new();

    // Add BERT-specific weights
    bert_checkpoint.set_weight(
        "embeddings.word_embeddings.weight",
        WeightTensor::new(vec![0.1; 768 * 30000], vec![30000, 768]),
    )?;
    bert_checkpoint.set_weight(
        "embeddings.LayerNorm.weight",
        WeightTensor::new(vec![1.0; 768], vec![768]),
    )?;
    bert_checkpoint.set_weight(
        "embeddings.LayerNorm.bias",
        WeightTensor::new(vec![0.0; 768], vec![768]),
    )?;
    bert_checkpoint.set_weight(
        "encoder.layer.0.attention.self.query.weight",
        WeightTensor::new(vec![0.2; 768 * 768], vec![768, 768]),
    )?;

    // Save and convert
    let temp_dir = tempfile::tempdir()?;
    let source_path = temp_dir.path().join("bert_pytorch.pt");
    bert_checkpoint.save(&source_path)?;

    let target_path = temp_dir.path().join("bert_tensorflow.ckpt");
    let result = converter
        .convert(&source_path, &target_path, CheckpointFormat::TensorFlow)
        .await?;

    println!("BERT conversion completed:");
    println!("  Weights converted: {}", result.weights_converted);
    println!("  Warnings: {:?}", result.warnings);

    // Verify the conversion
    let tf_checkpoint = TensorFlowCheckpoint::load(&target_path)?;
    let tf_names = tf_checkpoint.weight_names();
    println!("\nConverted weight names:");
    for name in tf_names.iter().take(5) {
        println!("  {}", name);
    }

    Ok(())
}

async fn demo_custom_mappings() -> Result<()> {
    // Create converter with custom weight mappings
    let converter = CheckpointConverterBuilder::new()
        .add_custom_mapping("old_name.weight", "new_name.kernel")
        .add_custom_mapping("layer.0.weight", "layer_0/weight")
        .exclude_weight("unused_weight")
        .build();

    // Create checkpoint with custom weights
    let mut checkpoint = TrustformersCheckpoint::new();
    checkpoint.set_weight(
        "old_name.weight",
        WeightTensor::new(vec![0.5; 100], vec![10, 10]),
    )?;
    checkpoint.set_weight(
        "layer.0.weight",
        WeightTensor::new(vec![0.6; 200], vec![20, 10]),
    )?;
    checkpoint.set_weight("unused_weight", WeightTensor::new(vec![0.7; 50], vec![50]))?;
    checkpoint.set_weight(
        "normal_weight",
        WeightTensor::new(vec![0.8; 150], vec![15, 10]),
    )?;

    let temp_dir = tempfile::tempdir()?;
    let source_path = temp_dir.path().join("custom.trust");
    checkpoint.save(&source_path)?;

    let target_path = temp_dir.path().join("custom_converted.pt");
    let result = converter.convert(&source_path, &target_path, CheckpointFormat::PyTorch).await?;

    println!("Custom mapping conversion:");
    println!("  Weights converted: {}", result.weights_converted);
    println!("  Weights skipped: {:?}", result.weights_skipped);

    Ok(())
}

fn demo_checkpoint_utils() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;

    // Create test checkpoints
    let mut checkpoint1 = TrustformersCheckpoint::new();
    checkpoint1.set_weight("weight1", WeightTensor::new(vec![1.0; 100], vec![10, 10]))?;
    checkpoint1.set_weight("weight2", WeightTensor::new(vec![2.0; 200], vec![20, 10]))?;
    checkpoint1.set_weight("common", WeightTensor::new(vec![3.0; 300], vec![30, 10]))?;

    let mut checkpoint2 = TrustformersCheckpoint::new();
    checkpoint2.set_weight("weight3", WeightTensor::new(vec![4.0; 400], vec![40, 10]))?;
    checkpoint2.set_weight("weight4", WeightTensor::new(vec![5.0; 500], vec![50, 10]))?;
    checkpoint2.set_weight("common", WeightTensor::new(vec![3.0; 300], vec![30, 10]))?;

    let path1 = temp_dir.path().join("checkpoint1.trust");
    let path2 = temp_dir.path().join("checkpoint2.trust");
    checkpoint1.save(&path1)?;
    checkpoint2.save(&path2)?;

    // Test format detection
    let detected_format = detect_format(&path1)?;
    println!("Detected format: {:?}", detected_format);

    // Test checkpoint info
    let info = get_checkpoint_info(&path1)?;
    println!("\nCheckpoint info:");
    println!("  Format: {:?}", info.format);
    println!("  File size: {} bytes", info.file_size_bytes);

    // Test checkpoint comparison
    let comparison = compare_checkpoints(&path1, &path2)?;
    println!("\n{}", comparison.summary());

    // Test checkpoint merging
    println!("\nMerging checkpoints...");
    let merged_path = temp_dir.path().join("merged.trust");
    merge_checkpoints(
        &[&path1, &path2],
        &merged_path,
        CheckpointFormat::Trustformers,
    )?;

    let merged = load_checkpoint(&merged_path)?;
    println!(
        "Merged checkpoint contains {} weights",
        merged.weight_names().len()
    );

    // Test checkpoint sharding
    println!("\nSharding checkpoint...");
    let shard_dir = temp_dir.path().join("shards");
    std::fs::create_dir(&shard_dir)?;
    let shards = shard_checkpoint(&merged_path, &shard_dir, 1)?; // 1MB shards
    println!("Created {} shards", shards.len());

    Ok(())
}

async fn demo_batch_conversion() -> Result<()> {
    println!("Performing batch conversion...");

    let temp_dir = tempfile::tempdir()?;
    let converter = CheckpointConverter::new();

    // Create multiple checkpoints
    let models = vec!["model1", "model2", "model3"];
    let mut tasks = Vec::new();

    for model_name in models {
        let mut checkpoint = PyTorchCheckpoint::new();
        checkpoint.set_weight(
            &format!("{}.weight", model_name),
            WeightTensor::new(vec![0.1; 1000], vec![100, 10]),
        )?;

        let source_path = temp_dir.path().join(format!("{}.pt", model_name));
        checkpoint.save(&source_path)?;

        let target_path = temp_dir.path().join(format!("{}.trust", model_name));

        // Spawn async conversion task
        let converter_clone = CheckpointConverter::new();
        let task = tokio::spawn(async move {
            converter_clone
                .convert(&source_path, &target_path, CheckpointFormat::Trustformers)
                .await
        });
        tasks.push(task);
    }

    // Wait for all conversions to complete
    let mut total_time = 0u64;
    for (i, task) in tasks.into_iter().enumerate() {
        let result = task.await??;
        println!(
            "  Model {} converted in {}ms",
            i + 1,
            result.conversion_time_ms
        );
        total_time += result.conversion_time_ms;
    }
    println!("Total batch conversion time: {}ms", total_time);

    Ok(())
}

async fn demo_checkpoint_validation() -> Result<()> {
    println!("Validating checkpoint conversions...");

    let temp_dir = tempfile::tempdir()?;

    // Create original checkpoint
    let mut original = PyTorchCheckpoint::new();
    original.set_weight(
        "layer1.weight",
        WeightTensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![2, 2]),
    )?;
    original.set_weight("layer2.bias", WeightTensor::new(vec![0.5, 0.6], vec![2]))?;

    let original_path = temp_dir.path().join("original.pt");
    original.save(&original_path)?;

    // Test round-trip conversion
    println!("\nTesting round-trip conversion (PyTorch -> TF -> PyTorch)...");

    // Convert to TensorFlow
    let tf_path = temp_dir.path().join("intermediate.ckpt");
    convert_checkpoint(&original_path, &tf_path, CheckpointFormat::TensorFlow).await?;

    // Convert back to PyTorch
    let roundtrip_path = temp_dir.path().join("roundtrip.pt");
    convert_checkpoint(&tf_path, &roundtrip_path, CheckpointFormat::PyTorch).await?;

    // Compare original and round-trip
    let comparison = compare_checkpoints(&original_path, &roundtrip_path)?;
    if comparison.is_compatible() {
        println!("✅ Round-trip conversion successful - checkpoints are compatible");
    } else {
        println!("❌ Round-trip conversion failed - checkpoints differ");
        println!("{}", comparison.summary());
    }

    Ok(())
}

// Helper function to demonstrate weight mapping rules
#[allow(dead_code)]
fn print_weight_mappings() {
    println!("\n=== Weight Mapping Examples ===");

    let mapping = WeightMapping::new(ModelType::BERT);

    let pytorch_names = vec![
        "embeddings.word_embeddings.weight",
        "embeddings.LayerNorm.weight",
        "embeddings.LayerNorm.bias",
        "encoder.layer.0.attention.self.query.weight",
    ];

    println!("\nBERT PyTorch -> TensorFlow mappings:");
    for name in pytorch_names {
        match mapping.pytorch_to_tensorflow(name) {
            Ok((tf_name, transform)) => {
                print!("  {} -> {}", name, tf_name);
                if let Some(t) = transform {
                    print!(" (transform: {:?})", t);
                }
                println!();
            },
            Err(e) => println!("  {} -> Error: {}", name, e),
        }
    }

    println!("\nGPT-2 mappings:");
    let gpt2_mapping = WeightMapping::new(ModelType::GPT2);
    let gpt2_names = vec!["wte.weight", "h.0.attn.c_attn.weight", "ln_f.weight"];

    for name in gpt2_names {
        if let Ok((tf_name, _)) = gpt2_mapping.pytorch_to_tensorflow(name) {
            println!("  {} -> {}", name, tf_name);
        }
    }
}
