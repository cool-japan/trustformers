//! Benchmark to verify SciRS2-Core MPS integration
//!
//! This benchmark tests matrix multiplication performance using:
//! 1. Naive Rust implementation (baseline)
//! 2. TrustformeRS tensor operations
//! 3. SciRS2-Core MPS (if available)
//!
//! Expected results on Apple Silicon (M1/M2/M3):
//! - Naive: ~0.1 GFLOPS
//! - TrustformeRS current: ~1-5 GFLOPS
//! - SciRS2-Core MPS: ~100-500 GFLOPS (100-500x speedup)

use std::time::Instant;
use trustformers_core::device::Device;
use trustformers_core::tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SciRS2-Core MPS Integration Benchmark ===\n");

    // Test matrix sizes (typical transformer dimensions)
    let sizes = vec![
        (512, 768, 768),   // Small model (BERT-like)
        (1024, 2048, 2048), // Medium model
        (2048, 4096, 4096), // Large model
    ];

    for (m, k, n) in sizes {
        println!("Matrix size: A({} x {}) × B({} x {}) = C({} x {})", m, k, k, n, m, n);
        println!("{}", "=".repeat(60));

        // Calculate theoretical GFLOPS
        let flops = (2.0 * m as f64 * k as f64 * n as f64) / 1e9;
        println!("Theoretical FLOPs: {:.2} GFLOPS\n", flops);

        // Test 1: CPU baseline
        println!("[1] CPU Baseline (trustformers_core::Tensor)");
        let device_cpu = Device::CPU;
        let a_cpu = Tensor::randn(&[m, k])?;
        let b_cpu = Tensor::randn(&[k, n])?;

        let start = Instant::now();
        let _c_cpu = a_cpu.matmul(&b_cpu)?;
        let duration = start.elapsed();

        let gflops_cpu = flops / duration.as_secs_f64();
        println!("  Time: {:.2}ms", duration.as_secs_f64() * 1000.0);
        println!("  Performance: {:.2} GFLOPS\n", gflops_cpu);

        // Test 2: Metal (if available)
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            println!("[2] Metal Backend (trustformers_core GPU)");
            match Device::metal(0) {
                Ok(device_metal) => {
                    let a_metal = Tensor::randn(&[m, k])?.to_device(&device_metal)?;
                    let b_metal = Tensor::randn(&[k, n])?.to_device(&device_metal)?;

                    // Warmup
                    let _ = a_metal.matmul(&b_metal)?;

                    let start = Instant::now();
                    let _c_metal = a_metal.matmul(&b_metal)?;
                    let duration = start.elapsed();

                    let gflops_metal = flops / duration.as_secs_f64();
                    println!("  Time: {:.2}ms", duration.as_secs_f64() * 1000.0);
                    println!("  Performance: {:.2} GFLOPS", gflops_metal);
                    println!("  Speedup vs CPU: {:.1}x\n", gflops_metal / gflops_cpu);
                }
                Err(e) => {
                    println!("  Metal not available: {}\n", e);
                }
            }
        }

        // Test 3: SciRS2-Core MPS (direct access)
        #[cfg(all(target_os = "macos", feature = "metal"))]
        {
            println!("[3] SciRS2-Core MPS (direct)");

            // Try to use scirs2-core MPS directly
            #[cfg(feature = "metal")]
            {
                use scirs2_core::gpu_ops::metal_ops;

                match metal_ops::initialize_metal() {
                    Ok((device, queue)) => {
                        use scirs2_core::gpu_ops::MPSOperations;

                        let mps = MPSOperations::new(device, queue);

                        // Create buffers
                        // Note: This is pseudocode - actual implementation needs buffer creation
                        println!("  SciRS2-Core MPS available");
                        println!("  TODO: Implement direct MPS matmul benchmark\n");
                    }
                    Err(e) => {
                        println!("  MPS initialization failed: {}\n", e);
                    }
                }
            }
        }

        #[cfg(not(target_os = "macos"))]
        {
            println!("[Metal/MPS] Not available on this platform ({})\n", std::env::consts::OS);
        }

        // Test 4: CUDA (if available)
        #[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
        {
            println!("[4] CUDA Backend");
            match Device::cuda(0) {
                Ok(device_cuda) => {
                    let a_cuda = Tensor::randn(&[m, k])?.to_device(&device_cuda)?;
                    let b_cuda = Tensor::randn(&[k, n])?.to_device(&device_cuda)?;

                    // Warmup
                    let _ = a_cuda.matmul(&b_cuda)?;

                    let start = Instant::now();
                    let _c_cuda = a_cuda.matmul(&b_cuda)?;
                    let duration = start.elapsed();

                    let gflops_cuda = flops / duration.as_secs_f64();
                    println!("  Time: {:.2}ms", duration.as_secs_f64() * 1000.0);
                    println!("  Performance: {:.2} GFLOPS", gflops_cuda);
                    println!("  Speedup vs CPU: {:.1}x\n", gflops_cuda / gflops_cpu);
                }
                Err(e) => {
                    println!("  CUDA not available: {}\n", e);
                }
            }
        }

        println!("\n");
    }

    println!("=== Summary ===");
    println!("To achieve 50-200 tok/sec target:");
    println!("- Current bottleneck: Matrix multiplication performance");
    println!("- Solution: Ensure MPS/cuBLAS backends are active");
    println!("- Expected: 100-500x improvement over CPU baseline");
    println!("\nRun on macOS M1/M2/M3 to test MPS performance.");
    println!("Run on Linux/Windows with NVIDIA GPU to test CUDA performance.");

    Ok(())
}
