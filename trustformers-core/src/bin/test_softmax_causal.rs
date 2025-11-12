use anyhow::Result;
use trustformers_core::device::Device;
use trustformers_core::gpu_ops::metal::get_metal_backend;

fn main() -> Result<()> {
    println!("===============================================");
    println!("🧪 Testing Softmax Causal GPU Kernel");
    println!("===============================================\n");

    let device = Device::metal_if_available(0);
    println!("Device: {:?}\n", device);

    let backend = get_metal_backend()?;

    // Create test input: simple attention scores [seq_len, seq_len]
    // For seq_len=5, test with known values
    let seq_len = 5;
    let mut input_data = vec![0.0f32; seq_len * seq_len];

    // Fill with simple pattern: row i has values [1, 2, 3, ..., seq_len]
    for i in 0..seq_len {
        for j in 0..seq_len {
            input_data[i * seq_len + j] = (j + 1) as f32;
        }
    }

    println!("📥 Input scores ({} x {}):", seq_len, seq_len);
    for i in 0..seq_len {
        print!("   Row {}: [", i);
        for j in 0..seq_len {
            print!("{:5.2}", input_data[i * seq_len + j]);
            if j < seq_len - 1 {
                print!(", ");
            }
        }
        println!("]");
    }
    println!();

    // Upload to GPU
    let input_buffer_id = backend.create_persistent_buffer(&input_data)?;

    // Run softmax_causal
    println!("🚀 Running softmax_causal GPU kernel...\n");
    let output_buffer_id = backend.softmax_causal_gpu_to_gpu(&input_buffer_id, seq_len)?;

    // Download result
    let output_data = backend.download_buffer_to_vec(&output_buffer_id)?;

    println!("📤 Softmax output ({} x {}):", seq_len, seq_len);
    for i in 0..seq_len {
        print!("   Row {}: [", i);
        let mut row_sum = 0.0f32;
        for j in 0..seq_len {
            let val = output_data[i * seq_len + j];
            print!("{:7.4}", val);
            if j < seq_len - 1 {
                print!(", ");
            }
            row_sum += val;
        }
        println!("]  sum={:.6}", row_sum);
    }
    println!();

    // Verify causal masking
    println!("🔍 Verification:");
    let mut all_correct = true;

    for i in 0..seq_len {
        for j in 0..seq_len {
            let val = output_data[i * seq_len + j];

            if j > i {
                // Future positions should be zero
                if val > 1e-6 {
                    println!(
                        "   ❌ Position ({}, {}) should be 0.0 but is {:.6}",
                        i, j, val
                    );
                    all_correct = false;
                }
            } else {
                // Valid positions should be positive
                if val < 1e-6 {
                    println!(
                        "   ⚠️  Position ({}, {}) is {:.6} (seems too small)",
                        i, j, val
                    );
                }
            }
        }

        // Check row sum
        let row_sum: f32 = output_data[i * seq_len..(i + 1) * seq_len].iter().sum();
        if (row_sum - 1.0).abs() > 0.01 {
            println!("   ❌ Row {} sum is {:.6}, expected ~1.0", i, row_sum);
            all_correct = false;
        }
    }

    if all_correct {
        println!("   ✅ All checks passed!");
    }

    println!("\n===============================================");
    Ok(())
}
