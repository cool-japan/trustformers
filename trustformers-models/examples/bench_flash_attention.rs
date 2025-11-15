// Microbenchmark: Flash Attention vs Standard Attention
use std::time::Instant;
use trustformers_core::gpu_ops::metal::functions::get_metal_backend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let backend = get_metal_backend()?;

    // Test parameters - realistic GPT-2 attention
    let batch_size = 1;
    let num_heads = 12;
    let head_dim = 64;

    println!("🔬 Attention Microbenchmark: Flash vs Standard");
    println!("{}", "=".repeat(60));

    // Test multiple sequence lengths
    for seq_len in &[16, 32, 64, 128, 256, 512] {
        println!("\n📊 Sequence Length: {}", seq_len);

        // Create test data
        let q_data = vec![0.1f32; batch_size * num_heads * seq_len * head_dim];
        let k_data = vec![0.2f32; batch_size * num_heads * seq_len * head_dim];
        let v_data = vec![0.3f32; batch_size * num_heads * seq_len * head_dim];

        let q_id = backend.create_persistent_buffer(&q_data)?;
        let k_id = backend.create_persistent_buffer(&k_data)?;
        let v_id = backend.create_persistent_buffer(&v_data)?;

        // Warmup
        for _ in 0..5 {
            let _ = backend.flash_attention_with_cache(
                &q_id, &k_id, &v_id, batch_size, *seq_len, *seq_len, num_heads, head_dim,
            )?;
        }

        // Benchmark Flash Attention
        let iterations = 100;
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = backend.flash_attention_with_cache(
                &q_id, &k_id, &v_id, batch_size, *seq_len, *seq_len, num_heads, head_dim,
            )?;
        }
        let flash_time = start.elapsed().as_secs_f64() / iterations as f64;

        // Benchmark Standard Attention
        let start = Instant::now();
        for _ in 0..iterations {
            let _ = backend.attention_with_cache_gpu_to_gpu(
                &q_id, &k_id, &v_id, batch_size, *seq_len, *seq_len, num_heads, head_dim,
            )?;
        }
        let standard_time = start.elapsed().as_secs_f64() / iterations as f64;

        let speedup = standard_time / flash_time;

        println!("  Flash Attention:    {:.3} ms", flash_time * 1000.0);
        println!("  Standard Attention: {:.3} ms", standard_time * 1000.0);
        println!("  ⚡ Speedup: {:.2}x", speedup);

        if speedup < 1.0 {
            println!("  ⚠️  Flash is SLOWER!");
        }
    }

    Ok(())
}
