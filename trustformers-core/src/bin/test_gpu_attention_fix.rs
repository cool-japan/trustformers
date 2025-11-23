use anyhow::Result;

#[cfg(all(target_os = "macos", feature = "metal"))]
use trustformers_core::device::Device;
#[cfg(all(target_os = "macos", feature = "metal"))]
use trustformers_core::gpu_ops::metal::get_metal_backend;

#[cfg(all(target_os = "macos", feature = "metal"))]
fn main() -> Result<()> {
    println!("===============================================");
    println!("🧪 Testing GPU Multi-Head Attention Fix");
    println!("===============================================\n");

    let device = Device::metal_if_available(0);
    println!("Device: {:?}\n", device);

    let backend = get_metal_backend()?;

    // Test configuration: GPT-2 small parameters
    let seq_len = 5;
    let num_heads = 12;
    let head_dim = 64;
    let hidden_size = num_heads * head_dim; // 768

    println!("Configuration:");
    println!("  seq_len: {}", seq_len);
    println!("  num_heads: {}", num_heads);
    println!("  head_dim: {}", head_dim);
    println!("  hidden_size: {}\n", hidden_size);

    // Create simple test data for Q, K, V [seq_len, hidden_size]
    let total_size = seq_len * hidden_size;

    let mut q_data = vec![0.0f32; total_size];
    let mut k_data = vec![0.0f32; total_size];
    let mut v_data = vec![0.0f32; total_size];

    // Fill with simple patterns to test multi-head separation
    for i in 0..seq_len {
        for j in 0..hidden_size {
            q_data[i * hidden_size + j] = ((i + 1) as f32) * 0.1 + ((j % 64) as f32) * 0.01;
            k_data[i * hidden_size + j] = ((i + 1) as f32) * 0.1 + ((j % 64) as f32) * 0.01;
            v_data[i * hidden_size + j] = ((j / 64) as f32) + (i as f32) * 0.1; // Different per head
        }
    }

    println!("📥 Creating GPU buffers...");
    let q_buffer_id = backend.create_persistent_buffer(&q_data)?;
    let k_buffer_id = backend.create_persistent_buffer(&k_data)?;
    let v_buffer_id = backend.create_persistent_buffer(&v_data)?;

    // Run multi-head attention
    println!("\n🚀 Running GPU Multi-Head Attention...\n");
    let output_buffer_id = backend.attention_gpu_to_gpu(
        &q_buffer_id,
        &k_buffer_id,
        &v_buffer_id,
        1, // batch_size
        seq_len,
        num_heads,
        head_dim,
    )?;

    // Download result
    println!("\n📤 Downloading result...");
    let output_data = backend.download_buffer_to_vec(&output_buffer_id)?;

    println!("\n✅ Multi-Head Attention Complete!");
    println!("\nOutput shape: [{}, {}]", seq_len, hidden_size);
    println!("Output size: {} elements\n", output_data.len());

    // Print first few elements of first and last position
    println!("First position output (first 10 elements):");
    print!("  ");
    for i in 0..10.min(hidden_size) {
        print!("{:.4} ", output_data[i]);
    }
    println!("\n");

    println!("Last position output (first 10 elements):");
    let last_pos_offset = (seq_len - 1) * hidden_size;
    print!("  ");
    for i in 0..10.min(hidden_size) {
        print!("{:.4} ", output_data[last_pos_offset + i]);
    }
    println!("\n");

    // Verify output is not all zeros or NaN
    let has_valid_output = output_data.iter().any(|&x| x.abs() > 1e-6 && !x.is_nan());
    let has_nan = output_data.iter().any(|&x| x.is_nan());
    let has_inf = output_data.iter().any(|&x| x.is_infinite());

    println!("Verification:");
    println!("  Has valid (non-zero) output: {}", has_valid_output);
    println!("  Has NaN values: {}", has_nan);
    println!("  Has Inf values: {}", has_inf);

    if has_valid_output && !has_nan && !has_inf {
        println!("\n✅ GPU Multi-Head Attention appears to be working correctly!");
    } else {
        println!("\n❌ GPU Multi-Head Attention may have issues!");
    }

    println!("\n===============================================");
    Ok(())
}

#[cfg(not(all(target_os = "macos", feature = "metal")))]
fn main() -> Result<()> {
    println!("This test requires macOS and the 'metal' feature to be enabled.");
    Ok(())
}
