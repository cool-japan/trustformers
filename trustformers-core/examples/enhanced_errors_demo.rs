//! Demonstration of enhanced error handling system
#![allow(unused_variables)]

use anyhow::Result;
use trustformers_core::errors::{
    compute_error, dimension_mismatch, invalid_config, model_not_found, out_of_memory, ErrorKind,
    ResultExt, TrustformersError,
};

fn main() -> Result<()> {
    println!("TrustformeRS Enhanced Error Handling Demo");
    println!("========================================\n");

    // Example 1: Dimension mismatch error
    println!("Example 1: Dimension Mismatch Error");
    demo_dimension_mismatch();

    // Example 2: Out of memory error
    println!("\nExample 2: Out of Memory Error");
    demo_out_of_memory();

    // Example 3: Model not found error
    println!("\nExample 3: Model Not Found Error");
    demo_model_not_found();

    // Example 4: Configuration error
    println!("\nExample 4: Configuration Error");
    demo_config_error();

    // Example 5: Compute error with context
    println!("\nExample 5: Compute Error with Context");
    demo_compute_error();

    // Example 6: Chained error context
    println!("\nExample 6: Chained Error Context");
    demo_chained_context()?;

    Ok(())
}

fn demo_dimension_mismatch() {
    let error = dimension_mismatch("[batch_size, 512, 768]", "[batch_size, 256, 768]")
        .with_operation("MultiHeadAttention.forward")
        .with_component("BERT-Base")
        .with_context("layer", "12".to_string())
        .with_context("head_count", "12".to_string())
        .with_suggestion("Consider using a padding strategy to match sequence lengths")
        .with_suggestion("Check if you're using the correct tokenizer max_length setting");

    println!("{}", error);
}

fn demo_out_of_memory() {
    let required = 8_000_000_000; // 8GB
    let available = 4_000_000_000; // 4GB

    let error = out_of_memory(required, available)
        .with_operation("forward_pass")
        .with_component("LLaMA-7B")
        .with_context("batch_size", "32".to_string())
        .with_context("sequence_length", "2048".to_string())
        .with_suggestion("Try batch_size=16 which would require ~4GB");

    println!("{}", error);
}

fn demo_model_not_found() {
    let error = model_not_found("meta-llama/llama-3-70b")
        .with_operation("AutoModel.from_pretrained")
        .with_context("source", "HuggingFace Hub".to_string())
        .with_context("revision", "main".to_string())
        .with_suggestion("Did you mean 'meta-llama/Llama-2-70b-hf'?")
        .with_suggestion("Check if you need to authenticate with `huggingface-cli login`");

    println!("{}", error);
}

fn demo_config_error() {
    let error = invalid_config(
        "num_attention_heads",
        "must be divisible by num_key_value_heads",
    )
    .with_component("MistralConfig")
    .with_context("num_attention_heads", "32".to_string())
    .with_context("num_key_value_heads", "7".to_string())
    .with_suggestion("Use num_key_value_heads=8 for grouped-query attention")
    .with_suggestion("Or use num_key_value_heads=32 for standard multi-head attention");

    println!("{}", error);
}

fn demo_compute_error() {
    let error = compute_error("softmax", "numerical overflow detected (inf values)")
        .with_component("FlashAttention")
        .with_context("input_dtype", "fp16".to_string())
        .with_context("max_value", "65504".to_string())
        .with_context("temperature", "0.1".to_string())
        .with_suggestion("Scale down logits before softmax: logits / temperature")
        .with_suggestion("Consider using fp32 for this operation")
        .with_suggestion("Enable autocast with loss scaling");

    println!("{}", error);
}

#[allow(clippy::result_large_err)]
fn demo_chained_context() -> Result<(), TrustformersError> {
    // Simulate a chain of operations that might fail
    load_model("gpt-5")
        .with_operation("pipeline_setup")
        .with_component("TextGenerationPipeline")?;

    Ok(())
}

#[allow(clippy::result_large_err)]
fn load_model(name: &str) -> Result<(), TrustformersError> {
    // Simulate model loading that fails
    Err(model_not_found(name)
        .with_operation("load_weights")
        .with_context("cache_dir", "~/.cache/trustformers".to_string()))
}

// Example of custom error creation
#[allow(dead_code)]
fn create_custom_error() -> TrustformersError {
    TrustformersError::new(ErrorKind::QuantizationError {
        reason: "Model contains layers that cannot be quantized to INT4".to_string(),
    })
    .with_component("GPTQ")
    .with_context(
        "problematic_layers",
        "['embed_tokens', 'lm_head']".to_string(),
    )
    .with_suggestion("Exclude embedding and output layers from quantization")
    .with_suggestion("Use mixed quantization: INT4 for attention, INT8 for FFN")
    .with_suggestion("Consider AWQ quantization which handles these layers better")
}

// Example of error handling in a function
#[allow(dead_code)]
#[allow(clippy::result_large_err)]
fn process_batch(batch_size: usize) -> Result<(), TrustformersError> {
    const MAX_BATCH_SIZE: usize = 64;

    if batch_size > MAX_BATCH_SIZE {
        return Err(invalid_config(
            "batch_size",
            format!("exceeds maximum of {}", MAX_BATCH_SIZE),
        )
        .with_context("requested", batch_size.to_string())
        .with_context("maximum", MAX_BATCH_SIZE.to_string())
        .with_suggestion(format!("Use batch_size <= {}", MAX_BATCH_SIZE))
        .with_suggestion("Enable gradient accumulation for effective larger batches"));
    }

    // Process batch...
    Ok(())
}

// Example showing error documentation links
#[allow(dead_code)]
fn show_error_documentation() {
    let error = TrustformersError::new(ErrorKind::AttentionError {
        reason: "Key-value cache overflow".to_string(),
    })
    .with_component("PagedAttention")
    .with_context("max_cache_size", "4GB".to_string())
    .with_context("requested_size", "4.5GB".to_string());

    println!("\nError with documentation link:");
    println!("{}", error);
    println!("\nNote: The error includes a link to detailed documentation!");
}
