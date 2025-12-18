#[cfg(all(target_os = "macos", feature = "metal"))]
use std::time::Instant;
#[cfg(all(target_os = "macos", feature = "metal"))]
use trustformers_core::device::Device;
#[cfg(all(target_os = "macos", feature = "metal"))]
use trustformers_models::gpt2::config::Gpt2Config;
#[cfg(all(target_os = "macos", feature = "metal"))]
use trustformers_models::gpt2::model::Gpt2LMHeadModel;

#[cfg(all(target_os = "macos", feature = "metal"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("===============================================");
    println!("🧪 Testing GPT-2 Text Generation with Fixed Multi-Head Attention");
    println!("===============================================\n");

    // Initialize small GPT-2 config
    let config = Gpt2Config {
        vocab_size: 50257,
        n_positions: 1024,
        n_embd: 768,
        n_layer: 12,
        n_head: 12,
        n_inner: Some(3072),
        ..Default::default()
    };

    println!("📦 Loading model on CPU...");
    let mut model = Gpt2LMHeadModel::new(config.clone())?;

    // Test prompt: "The capital of France is"
    // GPT-2 tokenization (approximate - real tokenizer would give exact tokens)
    // For now, using simple token IDs
    println!("\n📝 Test Prompt: \"Hello, world!\"");
    let input_ids = vec![15496u32, 11, 995, 0]; // "Hello", ",", "world", "!"

    println!("\n🔥 Testing GPU Generation with Fixed Multi-Head Attention...\n");

    // Move to GPU
    println!("📤 Moving model to Metal GPU...");
    model.weights_to_gpu(&Device::Metal(0))?;

    let start = Instant::now();
    let max_length = 20;
    let gpu_result = model.generate_greedy_with_cache(input_ids.clone(), max_length)?;
    let gpu_elapsed = start.elapsed();

    println!("\n✅ GPU Generation Complete!");
    println!("⏱️  GPU Time: {:.2}s", gpu_elapsed.as_secs_f64());
    println!(
        "📊 Tokens generated: {}",
        gpu_result.len() - input_ids.len()
    );
    println!(
        "⚡ GPU Tokens/sec: {:.2}",
        (gpu_result.len() - input_ids.len()) as f64 / gpu_elapsed.as_secs_f64()
    );
    println!("\n🔢 GPU Output token IDs:");
    println!("   {:?}", gpu_result);

    // Check if output looks reasonable (not all same token)
    let all_same = gpu_result.windows(2).all(|w| w[0] == w[1]);
    let has_repetitive_pattern = gpu_result.windows(3).any(|w| w[0] == w[1] && w[1] == w[2]);

    println!("\n🔍 Generation Quality Check:");
    println!("   All tokens same: {}", all_same);
    println!("   Has 3+ repetitive tokens: {}", has_repetitive_pattern);

    if all_same {
        println!("\n❌ FAILED: All tokens are the same (generation stuck)");
    } else if has_repetitive_pattern {
        println!("\n⚠️  WARNING: Detected repetitive token pattern (may indicate issues)");
    } else {
        println!("\n✅ PASSED: Token diversity looks good!");
    }

    // Show unique token count
    let mut unique_tokens = gpu_result.clone();
    unique_tokens.sort();
    unique_tokens.dedup();
    println!(
        "   Unique tokens: {} / {}",
        unique_tokens.len(),
        gpu_result.len()
    );

    println!("\n===============================================");
    Ok(())
}


#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn main() {
    println!("This example requires macOS with Metal support.");
    println!("Please compile with --features metal on macOS.");
}
