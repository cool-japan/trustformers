/// Test MPS matmul correctness by comparing with CPU
/// This will help identify if MPS is computing the wrong operation
use anyhow::Result;

fn main() -> Result<()> {
    println!("================================================================================");
    println!("🧪 MPS Matmul Correctness Test");
    println!("================================================================================\n");

    // Simple 2x3 @ 3x2 = 2x2 matrix multiplication
    println!("Test Case: 2x3 @ 3x2 = 2x2\n");

    // Matrix A (2x3)
    let a_data = vec![
        1.0, 2.0, 3.0, // Row 0
        4.0, 5.0, 6.0, // Row 1
    ];
    println!("Matrix A (2x3):");
    println!("  [{}, {}, {}]", a_data[0], a_data[1], a_data[2]);
    println!("  [{}, {}, {}]\n", a_data[3], a_data[4], a_data[5]);

    // Matrix B (3x2)
    let b_data = vec![
        7.0, 8.0, // Row 0
        9.0, 10.0, // Row 1
        11.0, 12.0, // Row 2
    ];
    println!("Matrix B (3x2):");
    println!("  [{}, {}]", b_data[0], b_data[1]);
    println!("  [{}, {}]", b_data[2], b_data[3]);
    println!("  [{}, {}]\n", b_data[4], b_data[5]);

    // Expected result (CPU computation)
    // C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
    // C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
    // C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
    // C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154

    println!("Expected Result (CPU):");
    println!("  [58.0, 64.0]");
    println!("  [139.0, 154.0]\n");

    #[cfg(all(target_os = "macos", feature = "metal"))]
    {
        use trustformers_core::gpu_ops::metal::get_metal_backend;

        println!("Testing MPS matmul...");

        // Get Metal backend
        let backend = get_metal_backend()?;

        // Upload matrices to GPU
        let a_buffer_id = backend.create_persistent_buffer(&a_data)?;
        let b_buffer_id = backend.create_persistent_buffer(&b_data)?;

        // Perform MPS matmul: C = A @ B
        let m = 2; // Rows in A and C
        let k = 3; // Cols in A, Rows in B
        let n = 2; // Cols in B and C

        let c_buffer_id = backend.matmul_gpu_to_gpu_mps(&a_buffer_id, &b_buffer_id, m, k, n)?;

        // Download result from GPU
        let c_data = backend.download_buffer_to_vec(&c_buffer_id)?;

        println!("MPS Result:");
        println!("  [{:.1}, {:.1}]", c_data[0], c_data[1]);
        println!("  [{:.1}, {:.1}]\n", c_data[2], c_data[3]);

        // Verify
        let expected = [58.0, 64.0, 139.0, 154.0];
        let mut all_correct = true;
        for i in 0..4 {
            let diff = (c_data[i] - expected[i]).abs();
            if diff > 0.001 {
                println!(
                    "❌ Mismatch at position {}: expected {}, got {} (diff: {})",
                    i, expected[i], c_data[i], diff
                );
                all_correct = false;
            }
        }

        if all_correct {
            println!("✅ MPS matmul is CORRECT!");
        } else {
            println!("🔴 MPS matmul is INCORRECT!");
        }
    }

    #[cfg(not(all(target_os = "macos", feature = "metal")))]
    {
        println!("⚠️  Metal not available, skipping GPU test");
    }

    Ok(())
}
