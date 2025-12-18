//! Benchmark to verify SciRS2-Core SIMD/BLAS integration
//!
//! This benchmark tests matrix multiplication performance using:
//! 1. TrustformeRS tensor operations (with scirs2-core BLAS)
//!
//! Expected results after BLAS integration:
//! - CPU with BLAS: 10-50 GFLOPS (via Accelerate/MKL/OpenBLAS)
//! - Without BLAS: ~0.5-2 GFLOPS

use std::time::Instant;
use trustformers_core::tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SciRS2-Core SIMD/BLAS Integration Benchmark ===\n");

    // Test matrix sizes (typical transformer dimensions)
    let sizes = vec![
        (512, 768, 768),    // Small model (BERT-like)
        (1024, 2048, 2048), // Medium model
        (2048, 4096, 4096), // Large model
    ];

    for (m, k, n) in sizes {
        println!(
            "Matrix size: A({} x {}) x B({} x {}) = C({} x {})",
            m, k, k, n, m, n
        );
        println!("{}", "=".repeat(60));

        // Calculate theoretical GFLOPS
        let flops = (2.0 * m as f64 * k as f64 * n as f64) / 1e9;
        println!("Theoretical FLOPs: {:.2} GFLOPS\n", flops);

        // Test: CPU with scirs2-core BLAS
        println!("[1] CPU with SciRS2-Core BLAS (Accelerate/MKL)");
        let a = Tensor::randn(&[m, k])?;
        let b = Tensor::randn(&[k, n])?;

        // Warmup run
        let _ = a.matmul(&b)?;

        // Timed runs
        let num_runs = 5;
        let mut durations = Vec::new();

        for _ in 0..num_runs {
            let start = Instant::now();
            let _c = a.matmul(&b)?;
            durations.push(start.elapsed());
        }

        // Calculate average (excluding first warmup-like run)
        let avg_duration: f64 =
            durations.iter().map(|d| d.as_secs_f64()).sum::<f64>() / num_runs as f64;
        let min_duration = durations.iter().map(|d| d.as_secs_f64()).fold(f64::INFINITY, f64::min);

        let gflops_avg = flops / avg_duration;
        let gflops_peak = flops / min_duration;

        println!("  Average time: {:.2}ms", avg_duration * 1000.0);
        println!("  Best time: {:.2}ms", min_duration * 1000.0);
        println!("  Average performance: {:.2} GFLOPS", gflops_avg);
        println!("  Peak performance: {:.2} GFLOPS\n", gflops_peak);

        // Reference performance targets
        if gflops_avg > 10.0 {
            println!("  [OK] BLAS acceleration detected (>10 GFLOPS)");
        } else if gflops_avg > 2.0 {
            println!("  [WARN] Moderate performance - BLAS may not be fully utilized");
        } else {
            println!("  [SLOW] Low performance - BLAS may not be active");
        }
        println!();
    }

    println!("=== Summary ===");
    println!("This benchmark tests scirs2-core's SIMD/BLAS integration.");
    println!();
    println!("Performance expectations:");
    println!("  - With Accelerate (macOS): 20-100+ GFLOPS");
    println!("  - With MKL (Intel): 50-200+ GFLOPS");
    println!("  - With OpenBLAS: 10-50+ GFLOPS");
    println!("  - Without BLAS: ~0.5-2 GFLOPS");
    println!();
    println!("If performance is low, ensure scirs2-core has 'linalg' feature enabled");
    println!("and appropriate BLAS library is linked.");

    Ok(())
}
