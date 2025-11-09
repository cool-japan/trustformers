//! Mobile Inference Demo
//!
//! Demonstrates optimized inference on mobile devices with real-world use cases

use std::time::{Duration, Instant};
use trustformers_core::{Result, Tensor};
use trustformers_mobile::{
    battery::MobileBatteryManager, device_info::MobileDeviceDetector,
    inference::MobileInferenceEngine, optimization::PowerMode, profiler::MobilePerformanceProfiler,
    thermal_power::ThermalPowerManager, MobileConfig,
};

fn main() -> Result<()> {
    println!("TrustformeRS Mobile Inference Demo");
    println!("==================================\n");

    // 1. Detect device capabilities
    let device_info = MobileDeviceDetector::detect()?;
    println!("Device Information:");
    println!("{}\n", device_info.summary());

    // 2. Create optimized configuration based on device
    let config = MobileDeviceDetector::generate_optimized_config(&device_info);
    println!("Optimized Configuration:");
    println!("- Backend: {:?}", config.backend);
    println!("- Memory limit: {} MB", config.max_memory_mb);
    println!("- Threads: {}", config.get_thread_count());
    println!("- FP16: {}", config.use_fp16);
    println!("- Quantization: {:?}\n", config.quantization);

    // 3. Initialize inference engine
    let mut engine = MobileInferenceEngine::new(config.clone())?;

    // 4. Load model (simulated)
    println!("Loading optimized model...");
    let _model_path = "model.tfm"; // Would be actual model file
                                   // engine.load_model(model_path)?;

    // 5. Initialize monitoring systems
    let thermal_manager = ThermalPowerManager::new(Default::default(), &device_info)?;
    let battery_manager = MobileBatteryManager::new(Default::default(), &device_info)?;
    let mut profiler = MobilePerformanceProfiler::new(Default::default(), &device_info)?;

    // 6. Run inference benchmarks
    run_performance_benchmark(&mut engine, &mut profiler)?;
    run_power_aware_inference(&mut engine, &thermal_manager, &battery_manager)?;
    run_adaptive_inference(&mut engine)?;

    // 7. Display profiling results
    display_profiling_results(&mut profiler)?;

    Ok(())
}

/// Run performance benchmark
fn run_performance_benchmark(
    engine: &mut MobileInferenceEngine,
    profiler: &mut MobilePerformanceProfiler,
) -> Result<()> {
    println!("Running Performance Benchmark");
    println!("-----------------------------");

    // Create sample input
    let input = Tensor::randn(&[1, 3, 224, 224])?;

    // Warmup
    println!("Warming up...");
    for _ in 0..5 {
        let _ = simulate_inference(engine, &input)?;
    }

    // Benchmark
    println!("Running benchmark (100 iterations)...");
    profiler.start_session("benchmark_session".to_string())?;

    let start = Instant::now();
    let mut latencies = Vec::new();

    for i in 0..100 {
        let iter_start = Instant::now();
        let _ = simulate_inference(engine, &input)?;
        let latency = iter_start.elapsed();
        latencies.push(latency);

        if i % 10 == 0 {
            print!(".");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
    }
    println!();

    let total_time = start.elapsed();
    let _session_stats = profiler.stop_session()?;

    // Calculate statistics
    let avg_latency = latencies.iter().sum::<Duration>() / latencies.len() as u32;
    let min_latency = *latencies.iter().min().unwrap();
    let max_latency = *latencies.iter().max().unwrap();

    // Calculate percentiles
    latencies.sort();
    let p50 = latencies[latencies.len() / 2];
    let p90 = latencies[latencies.len() * 9 / 10];
    let p95 = latencies[latencies.len() * 95 / 100];
    let p99 = latencies[latencies.len() * 99 / 100];

    println!("\nBenchmark Results:");
    println!("- Total time: {:?}", total_time);
    println!("- Average latency: {:?}", avg_latency);
    println!("- Min latency: {:?}", min_latency);
    println!("- Max latency: {:?}", max_latency);
    println!("- P50: {:?}", p50);
    println!("- P90: {:?}", p90);
    println!("- P95: {:?}", p95);
    println!("- P99: {:?}", p99);
    println!(
        "- Throughput: {:.2} inferences/sec\n",
        100.0 / total_time.as_secs_f32()
    );

    Ok(())
}

/// Run power-aware inference
fn run_power_aware_inference(
    engine: &mut MobileInferenceEngine,
    thermal_manager: &ThermalPowerManager,
    battery_manager: &MobileBatteryManager,
) -> Result<()> {
    println!("Running Power-Aware Inference");
    println!("-----------------------------");

    // Check current thermal and battery state
    let thermal_reading = thermal_manager.get_current_reading()?;
    let battery_reading = battery_manager.get_current_reading()?;

    println!("Current State:");
    println!(
        "- Temperature: {:.1}°C",
        thermal_reading.temperature_celsius
    );
    println!("- Thermal state: {:?}", thermal_reading.thermal_state);
    println!(
        "- Battery level: {:.1}%",
        battery_reading.level_percent.unwrap_or(0) as f32
    );
    println!("- Charging: {:?}", battery_reading.charging_status);

    // Adjust inference based on conditions
    if thermal_reading.temperature_celsius > 45.0 {
        println!("\n⚠️  High temperature detected - enabling thermal throttling");
        engine.set_power_mode(PowerMode::PowerSaving)?;
    } else if battery_reading.level_percent.unwrap_or(100) < 20
        && battery_reading.charging_status
            != trustformers_mobile::device_info::ChargingStatus::Charging
    {
        println!("\n🔋 Low battery detected - enabling power saving mode");
        engine.set_power_mode(PowerMode::PowerSaving)?;
    } else {
        println!("\n✅ Normal conditions - using balanced mode");
        engine.set_power_mode(PowerMode::Balanced)?;
    }

    // Run inference with power monitoring
    let input = Tensor::randn(&[1, 3, 224, 224])?;

    println!("\nRunning 10 inferences with power monitoring...");
    for i in 0..10 {
        let start = Instant::now();
        let _ = simulate_inference(engine, &input)?;
        let duration = start.elapsed();

        // Check if we need to adjust
        if i % 3 == 0 {
            let current_temp = thermal_manager.get_current_reading()?.temperature_celsius;
            if current_temp > thermal_reading.temperature_celsius + 2.0 {
                println!(
                    "Temperature rising ({:.1}°C) - reducing performance",
                    current_temp
                );
                engine.reduce_performance(0.8)?;
            }
        }

        println!("Inference {}: {:?}", i + 1, duration);
    }

    println!();
    Ok(())
}

/// Run adaptive inference based on conditions
fn run_adaptive_inference(engine: &mut MobileInferenceEngine) -> Result<()> {
    println!("Running Adaptive Inference");
    println!("-------------------------");

    // Simulate different network conditions
    let test_cases = vec![
        ("High-res image (good network)", vec![1, 3, 1024, 1024]),
        ("Medium-res image (moderate network)", vec![1, 3, 512, 512]),
        ("Low-res image (poor network)", vec![1, 3, 224, 224]),
    ];

    for (desc, shape) in test_cases {
        println!("\nTest case: {}", desc);

        let input = Tensor::randn(&shape)?;
        let input_size_mb = (shape.iter().product::<usize>() * 4) as f32 / (1024.0 * 1024.0);
        println!("Input size: {:.2} MB", input_size_mb);

        // Adapt batch size based on input size
        let batch_size = if input_size_mb > 10.0 {
            1
        } else if input_size_mb > 5.0 {
            2
        } else {
            4
        };

        engine.set_batch_size(batch_size)?;
        println!("Adaptive batch size: {}", batch_size);

        // Run inference
        let start = Instant::now();
        let _ = simulate_inference(engine, &input)?;
        let duration = start.elapsed();

        println!("Inference time: {:?}", duration);
        println!(
            "Throughput: {:.2} MB/s",
            input_size_mb / duration.as_secs_f32()
        );
    }

    println!();
    Ok(())
}

/// Simulate inference (placeholder for actual model inference)
fn simulate_inference(_engine: &MobileInferenceEngine, _input: &Tensor) -> Result<Tensor> {
    // In real implementation, this would run actual inference
    // For demo, simulate with computation
    std::thread::sleep(Duration::from_millis(50));

    // Return dummy output
    Ok(Tensor::zeros(&[1, 1000])?)
}

/// Display profiling results
fn display_profiling_results(profiler: &mut MobilePerformanceProfiler) -> Result<()> {
    println!("Profiling Results");
    println!("-----------------");

    let snapshot = profiler.collect_snapshot()?;

    println!("\nInference Metrics:");
    println!(
        "- Latency: {:.2} ms",
        snapshot.metrics.inference_metrics.latency_ms
    );
    println!(
        "- Throughput: {:.2} inferences/sec",
        snapshot.metrics.inference_metrics.throughput_ips
    );
    println!(
        "- Queue depth: {}",
        snapshot.metrics.inference_metrics.queue_depth
    );
    println!(
        "- Memory footprint: {} MB",
        snapshot.metrics.inference_metrics.memory_footprint_mb
    );

    println!("\nCPU Metrics:");
    println!(
        "- Utilization: {:.1}%",
        snapshot.metrics.platform_metrics.cpu_metrics.utilization_percent
    );
    println!(
        "- Load average: {:.2}",
        snapshot.metrics.platform_metrics.cpu_metrics.load_average[0]
    );

    println!("\nMemory Metrics:");
    println!(
        "- Total usage: {} MB",
        snapshot.metrics.platform_metrics.memory_metrics.total_usage_mb
    );
    println!(
        "- Available: {} MB",
        snapshot.metrics.platform_metrics.memory_metrics.available_mb
    );
    println!(
        "- Pressure level: {:?}",
        snapshot.metrics.platform_metrics.memory_metrics.pressure_level
    );

    if let Some(bottleneck) = snapshot.bottlenecks.first() {
        println!("\n⚠️  Performance Bottleneck Detected:");
        println!("- Type: {:?}", bottleneck.bottleneck_type);
        println!("- Severity: {:?}", bottleneck.severity);
        println!("- Description: {}", bottleneck.description);

        if let Some(suggestion) = snapshot.optimization_suggestions.first() {
            println!("\n💡 Optimization Suggestion:");
            println!("- {}", suggestion.description);
            println!(
                "- Expected improvement: {:.1}%",
                suggestion.expected_improvement_percent
            );
        }
    }

    println!();
    Ok(())
}

// Mock implementations for demo
#[allow(dead_code)]
mod inference {
    use super::*;

    #[derive(Debug, Clone, Copy)]
    pub enum InferencePowerMode {
        HighPerformance,
        Balanced,
        PowerSaving,
    }

    pub struct MobileInferenceEngine {
        config: MobileConfig,
        power_mode: InferencePowerMode,
        batch_size: usize,
        performance_scale: f32,
    }

    impl MobileInferenceEngine {
        pub fn new(config: MobileConfig) -> Result<Self> {
            Ok(Self {
                config,
                power_mode: InferencePowerMode::Balanced,
                batch_size: 1,
                performance_scale: 1.0,
            })
        }

        pub fn set_power_mode(&mut self, mode: InferencePowerMode) -> Result<()> {
            self.power_mode = mode;
            Ok(())
        }

        pub fn set_batch_size(&mut self, size: usize) -> Result<()> {
            self.batch_size = size;
            Ok(())
        }

        pub fn reduce_performance(&mut self, scale: f32) -> Result<()> {
            self.performance_scale *= scale;
            Ok(())
        }
    }
}
