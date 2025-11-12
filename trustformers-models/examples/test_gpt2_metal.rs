use trustformers_models::gpt2::config::Gpt2Config;
use trustformers_models::gpt2::model::Gpt2LMHeadModel;
use trustformers_core::device::Device;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("🚀 Testing GPT-2 with Metal GPU Attention");

    // Initialize model (smaller size for testing)
    let config = Gpt2Config {
        vocab_size: 50257,
        n_positions: 1024,
        n_embd: 768,
        n_layer: 12,
        n_head: 12,
        n_inner: Some(3072),
        ..Default::default()
    };

    eprintln!("Loading model on CPU...");
    let mut model = Gpt2LMHeadModel::new(config.clone())?;

    eprintln!("Uploading model weights to Metal GPU...");
    model.weights_to_gpu(&Device::Metal(0))?;

    // Test input: "Hello, world!" = [15496, 11, 995, 0]
    let input_ids = vec![15496u32, 11, 995, 0];

    eprintln!("\n📝 Input: {:?}", input_ids);
    eprintln!("   (tokens: 'Hello', ',', 'world', '!')\n");

    // Run generation with timing
    eprintln!("🔥 Starting generation (first forward pass should use GPU attention)...\n");
    let start = Instant::now();

    let max_length = 20;  // Generate 16 more tokens (total 20)
    let result = model.generate_greedy_with_cache(input_ids.clone(), max_length)?;

    let elapsed = start.elapsed();

    eprintln!("\n✅ Generation complete!");
    eprintln!("⏱️  Total time: {:.2}s", elapsed.as_secs_f64());
    eprintln!("📊 Tokens generated: {}", result.len() - input_ids.len());
    eprintln!("⚡ Tokens/sec: {:.2}", (result.len() - input_ids.len()) as f64 / elapsed.as_secs_f64());
    eprintln!("\n🔢 Output token IDs: {:?}", result);

    Ok(())
}
