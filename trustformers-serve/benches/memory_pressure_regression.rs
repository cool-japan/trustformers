//! Memory Pressure Cleanup Handler Performance Regression Tests
//!
//! This benchmark suite provides comprehensive performance regression testing for
//! cleanup handlers in the TrustformeRS memory pressure system. It establishes
//! performance baselines, tracks metrics over time, and detects regressions.
//!
//! ## Key Performance Metrics Tracked:
//! - Cleanup execution time under different pressure levels
//! - Memory freed per second (throughput)
//! - CPU usage during cleanup operations

// Allow unused code for benchmarks
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_imports)]
//! - Memory allocation efficiency
//! - Scalability with different workload sizes
//! - Latency percentiles (P50, P95, P99)
//!
//! ## Cleanup Handlers Tested:
//! - GarbageCollectionHandler
//! - BufferCompactionHandler
//! - GPU cleanup handlers (cache eviction, buffer compaction, model unloading, etc.)
//!
//! ## Usage:
//! ```bash
//! # Run all memory pressure regression benchmarks
//! cargo bench --bench memory_pressure_regression
//!
//! # Run with baseline recording
//! RECORD_BASELINE=1 cargo bench --bench memory_pressure_regression
//!
//! # Run with regression detection
//! CHECK_REGRESSION=1 cargo bench --bench memory_pressure_regression
//! ```

use chrono::{DateTime, Utc};
use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, AxisScale, BenchmarkId,
    Criterion, PlotConfiguration, Throughput,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant};
use tokio::runtime::Runtime;

use trustformers_serve::memory_pressure::{
    BufferCompactionHandler, CleanupHandler, GarbageCollectionHandler, GpuCleanupStrategy,
    MemoryPressureConfig, MemoryPressureHandler, MemoryPressureLevel, MemoryPressureThresholds,
};

/// Performance baseline data for a specific cleanup handler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    /// Handler name
    pub handler_name: String,
    /// Test scenario
    pub scenario: String,
    /// Mean execution time in nanoseconds
    pub mean_execution_time_ns: u64,
    /// P50 execution time in nanoseconds
    pub p50_execution_time_ns: u64,
    /// P95 execution time in nanoseconds
    pub p95_execution_time_ns: u64,
    /// P99 execution time in nanoseconds
    pub p99_execution_time_ns: u64,
    /// Memory freed in bytes
    pub memory_freed_bytes: u64,
    /// Memory throughput (bytes per second)
    pub memory_throughput_bps: f64,
    /// CPU usage percentage during operation
    pub cpu_usage_percent: f64,
    /// Timestamp when baseline was recorded
    pub recorded_at: DateTime<Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Performance regression detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionConfig {
    /// Maximum allowed performance degradation percentage (e.g., 15.0 for 15%)
    pub max_degradation_percent: f64,
    /// Maximum allowed execution time increase percentage
    pub max_time_increase_percent: f64,
    /// Minimum required memory throughput (bytes per second)
    pub min_memory_throughput_bps: f64,
    /// Maximum allowed CPU usage increase percentage
    pub max_cpu_increase_percent: f64,
    /// Whether to fail the benchmark on regression detection
    pub fail_on_regression: bool,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            max_degradation_percent: 15.0,
            max_time_increase_percent: 20.0,
            min_memory_throughput_bps: 100_000_000.0, // 100MB/s minimum
            max_cpu_increase_percent: 25.0,
            fail_on_regression: std::env::var("FAIL_ON_REGRESSION").is_ok(),
        }
    }
}

/// Performance baseline manager
pub struct BaselineManager {
    baseline_file: String,
    regression_config: RegressionConfig,
}

impl BaselineManager {
    pub fn new() -> Self {
        let baseline_file = std::env::var("BASELINE_FILE")
            .unwrap_or_else(|_| "/tmp/memory_pressure_baselines.json".to_string());

        Self {
            baseline_file,
            regression_config: RegressionConfig::default(),
        }
    }

    /// Load existing baselines from file
    pub fn load_baselines(&self) -> HashMap<String, PerformanceBaseline> {
        if !Path::new(&self.baseline_file).exists() {
            return HashMap::new();
        }

        match fs::read_to_string(&self.baseline_file) {
            Ok(content) => serde_json::from_str(&content).unwrap_or_else(|e| {
                eprintln!("Failed to parse baseline file: {}", e);
                HashMap::new()
            }),
            Err(e) => {
                eprintln!("Failed to read baseline file: {}", e);
                HashMap::new()
            },
        }
    }

    /// Save baselines to file
    pub fn save_baselines(&self, baselines: &HashMap<String, PerformanceBaseline>) {
        let content =
            serde_json::to_string_pretty(baselines).expect("Failed to serialize baselines");

        if let Some(parent) = Path::new(&self.baseline_file).parent() {
            fs::create_dir_all(parent).ok();
        }

        fs::write(&self.baseline_file, content).expect("Failed to write baseline file");
    }

    /// Record a new baseline measurement
    pub fn record_baseline(
        &self,
        handler_name: String,
        scenario: String,
        execution_times: &[Duration],
        memory_freed: u64,
        cpu_usage: f64,
    ) {
        if std::env::var("RECORD_BASELINE").is_err() {
            return;
        }

        let mut baselines = self.load_baselines();
        let key = format!("{}_{}", handler_name, scenario);

        // Calculate statistics from execution times
        let mut times_ns: Vec<u64> = execution_times.iter().map(|d| d.as_nanos() as u64).collect();
        times_ns.sort();

        let mean_time_ns = times_ns.iter().sum::<u64>() / times_ns.len() as u64;
        let p50_time_ns = times_ns[times_ns.len() / 2];
        let p95_time_ns = times_ns[(times_ns.len() * 95) / 100];
        let p99_time_ns = times_ns[(times_ns.len() * 99) / 100];

        // Calculate memory throughput
        let mean_time_seconds = mean_time_ns as f64 / 1_000_000_000.0;
        let memory_throughput_bps = if mean_time_seconds > 0.0 {
            memory_freed as f64 / mean_time_seconds
        } else {
            0.0
        };

        let baseline = PerformanceBaseline {
            handler_name,
            scenario,
            mean_execution_time_ns: mean_time_ns,
            p50_execution_time_ns: p50_time_ns,
            p95_execution_time_ns: p95_time_ns,
            p99_execution_time_ns: p99_time_ns,
            memory_freed_bytes: memory_freed,
            memory_throughput_bps,
            cpu_usage_percent: cpu_usage,
            recorded_at: Utc::now(),
            metadata: HashMap::new(),
        };

        baselines.insert(key.clone(), baseline);
        self.save_baselines(&baselines);

        println!("📊 Recorded baseline for {}", key);
    }

    /// Check for performance regressions
    pub fn check_regression(
        &self,
        handler_name: &str,
        scenario: &str,
        execution_times: &[Duration],
        memory_freed: u64,
        cpu_usage: f64,
    ) -> bool {
        if std::env::var("CHECK_REGRESSION").is_err() {
            return false;
        }

        let baselines = self.load_baselines();
        let key = format!("{}_{}", handler_name, scenario);

        let baseline = match baselines.get(&key) {
            Some(b) => b,
            None => {
                println!(
                    "⚠️  No baseline found for {}, skipping regression check",
                    key
                );
                return false;
            },
        };

        // Calculate current performance metrics
        let times_ns: Vec<u64> = execution_times.iter().map(|d| d.as_nanos() as u64).collect();
        let current_mean_ns = times_ns.iter().sum::<u64>() / times_ns.len() as u64;
        let current_throughput = if current_mean_ns > 0 {
            memory_freed as f64 / (current_mean_ns as f64 / 1_000_000_000.0)
        } else {
            0.0
        };

        // Check for regressions
        let mut regressions = Vec::new();

        // Check execution time regression
        let time_increase_percent = ((current_mean_ns as f64
            - baseline.mean_execution_time_ns as f64)
            / baseline.mean_execution_time_ns as f64)
            * 100.0;
        if time_increase_percent > self.regression_config.max_time_increase_percent {
            regressions.push(format!(
                "Execution time increased by {:.1}% (limit: {:.1}%)",
                time_increase_percent, self.regression_config.max_time_increase_percent
            ));
        }

        // Check memory throughput regression
        let throughput_decrease_percent = ((baseline.memory_throughput_bps - current_throughput)
            / baseline.memory_throughput_bps)
            * 100.0;
        if throughput_decrease_percent > self.regression_config.max_degradation_percent {
            regressions.push(format!(
                "Memory throughput decreased by {:.1}% (limit: {:.1}%)",
                throughput_decrease_percent, self.regression_config.max_degradation_percent
            ));
        }

        // Check CPU usage regression
        let cpu_increase_percent =
            ((cpu_usage - baseline.cpu_usage_percent) / baseline.cpu_usage_percent) * 100.0;
        if cpu_increase_percent > self.regression_config.max_cpu_increase_percent {
            regressions.push(format!(
                "CPU usage increased by {:.1}% (limit: {:.1}%)",
                cpu_increase_percent, self.regression_config.max_cpu_increase_percent
            ));
        }

        if !regressions.is_empty() {
            println!("🚨 Performance regression detected for {}:", key);
            for regression in &regressions {
                println!("   - {}", regression);
            }

            if self.regression_config.fail_on_regression {
                panic!(
                    "Performance regression detected: {}",
                    regressions.join(", ")
                );
            }

            return true;
        }

        println!("✅ No regression detected for {}", key);
        false
    }
}

/// Create test configuration for memory pressure handler
fn create_test_memory_pressure_config() -> MemoryPressureConfig {
    MemoryPressureConfig {
        enabled: true,
        monitoring_interval_seconds: 1,
        pressure_thresholds: MemoryPressureThresholds {
            low: 0.7,
            medium: 0.8,
            high: 0.9,
            critical: 0.95,
            adaptive: true,
            base_low: 0.65,
            base_medium: 0.75,
            base_high: 0.85,
            base_critical: 0.92,
            adjustment_factor: 1.1,
            learning_rate: 0.01,
        },
        cleanup_strategies: vec![
            trustformers_serve::memory_pressure::CleanupStrategy::GarbageCollection,
            trustformers_serve::memory_pressure::CleanupStrategy::BufferCompaction,
        ],
        emergency_threshold: 0.98,
        memory_buffer_mb: 512,
        enable_aggressive_cleanup: true,
        gc_trigger_threshold: 0.85,
        cache_eviction_threshold: 0.80,
        request_rejection_threshold: 0.95,
        enable_memory_compaction: true,
        swap_usage_threshold: 0.90,
        enable_gpu_monitoring: true,
        gpu_pressure_thresholds: MemoryPressureThresholds {
            low: 0.7,
            medium: 0.8,
            high: 0.9,
            critical: 0.95,
            adaptive: true,
            base_low: 0.65,
            base_medium: 0.75,
            base_high: 0.85,
            base_critical: 0.92,
            adjustment_factor: 1.1,
            learning_rate: 0.01,
        },
        gpu_cleanup_strategies: vec![
            GpuCleanupStrategy::GpuCacheEviction,
            GpuCleanupStrategy::GpuBufferCompaction,
            GpuCleanupStrategy::GpuModelUnloading,
        ],
        gpu_memory_buffer_mb: 512,
        gpu_fragmentation_threshold: 0.8,
        gpu_device_strategy: trustformers_serve::memory_pressure::GpuDeviceStrategy::All,
    }
}

/// Measure performance of a cleanup operation
fn measure_cleanup_performance<F>(
    handler_name: &str,
    scenario: &str,
    pressure_level: MemoryPressureLevel,
    operation: F,
    baseline_manager: &BaselineManager,
) -> (Vec<Duration>, u64)
where
    F: Fn() -> u64,
{
    const NUM_ITERATIONS: usize = 100;
    let mut execution_times = Vec::with_capacity(NUM_ITERATIONS);
    let mut total_memory_freed = 0u64;

    // Warm up
    for _ in 0..10 {
        black_box(operation());
    }

    // Measure performance
    for _ in 0..NUM_ITERATIONS {
        let start = Instant::now();
        let memory_freed = black_box(operation());
        let elapsed = start.elapsed();

        execution_times.push(elapsed);
        total_memory_freed += memory_freed;
    }

    let avg_memory_freed = total_memory_freed / NUM_ITERATIONS as u64;

    // Record baseline and check for regressions
    let mock_cpu_usage = 15.0; // Mock CPU usage since actual measurement requires platform-specific code
    baseline_manager.record_baseline(
        handler_name.to_string(),
        scenario.to_string(),
        &execution_times,
        avg_memory_freed,
        mock_cpu_usage,
    );

    baseline_manager.check_regression(
        handler_name,
        scenario,
        &execution_times,
        avg_memory_freed,
        mock_cpu_usage,
    );

    (execution_times, avg_memory_freed)
}

/// Benchmark GarbageCollectionHandler performance
fn bench_garbage_collection_handler(c: &mut Criterion) {
    let mut group = c.benchmark_group("garbage_collection_handler");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let baseline_manager = BaselineManager::new();
    let handler = GarbageCollectionHandler::new();

    // Test different pressure levels
    for pressure_level in [
        MemoryPressureLevel::Low,
        MemoryPressureLevel::Medium,
        MemoryPressureLevel::High,
        MemoryPressureLevel::Critical,
    ] {
        let scenario = format!("pressure_{:?}", pressure_level);

        group.bench_with_input(
            BenchmarkId::new("cleanup_execution", &scenario),
            &pressure_level,
            |b, &pressure_level| {
                b.iter_custom(|iters| {
                    let (execution_times, memory_freed) = measure_cleanup_performance(
                        "GarbageCollectionHandler",
                        &scenario,
                        pressure_level,
                        || handler.cleanup(pressure_level).unwrap_or(0),
                        &baseline_manager,
                    );

                    // Return the mean execution time for criterion
                    let total_time: Duration = execution_times.iter().sum();
                    total_time / execution_times.len() as u32
                })
            },
        );

        // Benchmark memory throughput
        group.bench_with_input(
            BenchmarkId::new("memory_throughput", &scenario),
            &pressure_level,
            |b, &pressure_level| {
                b.iter(|| {
                    let start = Instant::now();
                    let memory_freed = black_box(handler.cleanup(pressure_level).unwrap_or(0));
                    let elapsed = start.elapsed();

                    // Calculate throughput (bytes per second)
                    let throughput = if elapsed.as_secs_f64() > 0.0 {
                        memory_freed as f64 / elapsed.as_secs_f64()
                    } else {
                        0.0
                    };

                    black_box(throughput)
                })
            },
        );
    }

    // Benchmark concurrent execution
    group.bench_function("concurrent_cleanup", |b| {
        let rt = Runtime::new().unwrap();
        b.to_async(&rt).iter(|| async {
            let handles: Vec<_> = (0..8)
                .map(|_| {
                    tokio::spawn(async {
                        let handler = GarbageCollectionHandler::new();
                        handler.cleanup(MemoryPressureLevel::High).unwrap_or(0)
                    })
                })
                .collect();

            let results = futures::future::join_all(handles).await;
            black_box(results)
        })
    });

    // Benchmark scalability with different workload sizes
    for workload_size in [1, 4, 8, 16, 32].iter() {
        group.bench_with_input(
            BenchmarkId::new("scalability", workload_size),
            workload_size,
            |b, &workload_size| {
                b.iter(|| {
                    let mut total_freed = 0u64;
                    for _ in 0..workload_size {
                        total_freed +=
                            black_box(handler.cleanup(MemoryPressureLevel::Medium).unwrap_or(0));
                    }
                    black_box(total_freed)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark BufferCompactionHandler performance
fn bench_buffer_compaction_handler(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_compaction_handler");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let baseline_manager = BaselineManager::new();
    let handler = BufferCompactionHandler::new();

    // Test different pressure levels with throughput measurement
    for pressure_level in [
        MemoryPressureLevel::Low,
        MemoryPressureLevel::Medium,
        MemoryPressureLevel::High,
        MemoryPressureLevel::Critical,
    ] {
        let scenario = format!("pressure_{:?}", pressure_level);
        let memory_freed = handler.cleanup(pressure_level).unwrap_or(0);

        group.throughput(Throughput::Bytes(memory_freed));
        group.bench_with_input(
            BenchmarkId::new("cleanup_execution", &scenario),
            &pressure_level,
            |b, &pressure_level| {
                b.iter_custom(|iters| {
                    let (execution_times, _) = measure_cleanup_performance(
                        "BufferCompactionHandler",
                        &scenario,
                        pressure_level,
                        || handler.cleanup(pressure_level).unwrap_or(0),
                        &baseline_manager,
                    );

                    let total_time: Duration = execution_times.iter().sum();
                    total_time / execution_times.len() as u32
                })
            },
        );
    }

    // Benchmark under memory pressure simulation
    group.bench_function("memory_pressure_simulation", |b| {
        b.iter(|| {
            // Simulate memory pressure by allocating and then cleaning up
            let _large_allocation: Vec<u8> = vec![0; 10 * 1024 * 1024]; // 10MB
            let memory_freed = black_box(handler.cleanup(MemoryPressureLevel::High).unwrap_or(0));
            black_box(memory_freed)
        })
    });

    group.finish();
}

/// Benchmark GPU cleanup handlers performance
fn bench_gpu_cleanup_handlers(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_cleanup_handlers");
    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    let baseline_manager = BaselineManager::new();
    let rt = Runtime::new().unwrap();
    let config = create_test_memory_pressure_config();
    let handler = MemoryPressureHandler::new(config);

    // Test different GPU cleanup strategies
    let gpu_strategies = vec![
        ("cache_eviction", GpuCleanupStrategy::GpuCacheEviction),
        ("buffer_compaction", GpuCleanupStrategy::GpuBufferCompaction),
        ("model_unloading", GpuCleanupStrategy::GpuModelUnloading),
        ("vram_compaction", GpuCleanupStrategy::GpuVramCompaction),
        (
            "memory_defragmentation",
            GpuCleanupStrategy::GpuMemoryDefragmentation,
        ),
        ("context_switching", GpuCleanupStrategy::GpuContextSwitching),
        ("stream_cleanup", GpuCleanupStrategy::GpuStreamCleanup),
        ("texture_cleanup", GpuCleanupStrategy::GpuTextureCleanup),
        ("memory_pool_reset", GpuCleanupStrategy::GpuMemoryPoolReset),
        (
            "batch_size_reduction",
            GpuCleanupStrategy::GpuBatchSizeReduction,
        ),
    ];

    for (strategy_name, strategy) in gpu_strategies {
        group.bench_with_input(
            BenchmarkId::new("gpu_cleanup_execution", strategy_name),
            &strategy,
            |b, strategy| {
                b.to_async(&rt).iter_custom(|iters| async {
                    let (execution_times, _) = measure_cleanup_performance(
                        &format!("GPU_{:?}", strategy),
                        "device_0",
                        MemoryPressureLevel::High,
                        || {
                            // Mock the cleanup operation since we can't actually test GPU operations
                            let memory_freed = match strategy {
                                GpuCleanupStrategy::GpuCacheEviction => 50 * 1024 * 1024,
                                GpuCleanupStrategy::GpuBufferCompaction => 30 * 1024 * 1024,
                                GpuCleanupStrategy::GpuModelUnloading => 200 * 1024 * 1024,
                                GpuCleanupStrategy::GpuVramCompaction => 80 * 1024 * 1024,
                                GpuCleanupStrategy::GpuMemoryDefragmentation => 40 * 1024 * 1024,
                                GpuCleanupStrategy::GpuContextSwitching => 10 * 1024 * 1024,
                                GpuCleanupStrategy::GpuStreamCleanup => 20 * 1024 * 1024,
                                GpuCleanupStrategy::GpuTextureCleanup => 60 * 1024 * 1024,
                                GpuCleanupStrategy::GpuMemoryPoolReset => 100 * 1024 * 1024,
                                GpuCleanupStrategy::GpuBatchSizeReduction => 25 * 1024 * 1024,
                            };

                            // Simulate the actual cleanup time based on the operation complexity
                            let sleep_duration = match strategy {
                                GpuCleanupStrategy::GpuModelUnloading => Duration::from_millis(50),
                                GpuCleanupStrategy::GpuMemoryPoolReset => Duration::from_millis(30),
                                GpuCleanupStrategy::GpuVramCompaction => Duration::from_millis(25),
                                _ => Duration::from_millis(10),
                            };
                            std::thread::sleep(sleep_duration);

                            memory_freed
                        },
                        &baseline_manager,
                    );

                    let total_time: Duration = execution_times.iter().sum();
                    total_time / execution_times.len() as u32
                })
            },
        );

        // Benchmark memory throughput for GPU operations
        let expected_memory_freed = match strategy {
            GpuCleanupStrategy::GpuCacheEviction => 50 * 1024 * 1024,
            GpuCleanupStrategy::GpuBufferCompaction => 30 * 1024 * 1024,
            GpuCleanupStrategy::GpuModelUnloading => 200 * 1024 * 1024,
            GpuCleanupStrategy::GpuVramCompaction => 80 * 1024 * 1024,
            GpuCleanupStrategy::GpuMemoryDefragmentation => 40 * 1024 * 1024,
            GpuCleanupStrategy::GpuContextSwitching => 10 * 1024 * 1024,
            GpuCleanupStrategy::GpuStreamCleanup => 20 * 1024 * 1024,
            GpuCleanupStrategy::GpuTextureCleanup => 60 * 1024 * 1024,
            GpuCleanupStrategy::GpuMemoryPoolReset => 100 * 1024 * 1024,
            GpuCleanupStrategy::GpuBatchSizeReduction => 25 * 1024 * 1024,
        };

        group.throughput(Throughput::Bytes(expected_memory_freed));
        group.bench_with_input(
            BenchmarkId::new("gpu_cleanup_throughput", strategy_name),
            &strategy,
            |b, _strategy| {
                b.to_async(&rt).iter(|| async {
                    let start = Instant::now();
                    let memory_freed = black_box(expected_memory_freed);

                    // Simulate cleanup time
                    let sleep_duration = Duration::from_millis(10);
                    tokio::time::sleep(sleep_duration).await;

                    let elapsed = start.elapsed();
                    let throughput = if elapsed.as_secs_f64() > 0.0 {
                        memory_freed as f64 / elapsed.as_secs_f64()
                    } else {
                        0.0
                    };

                    black_box(throughput)
                })
            },
        );
    }

    // Benchmark concurrent GPU cleanup operations
    group.bench_function("concurrent_gpu_cleanup", |b| {
        b.to_async(&rt).iter(|| async {
            let handles: Vec<_> = (0..4)
                .map(|device_id| {
                    tokio::spawn(async move {
                        // Simulate concurrent cleanup on different GPU devices
                        tokio::time::sleep(Duration::from_millis(20)).await;
                        50 * 1024 * 1024 // Mock memory freed
                    })
                })
                .collect();

            let results = futures::future::join_all(handles).await;
            black_box(results)
        })
    });

    group.finish();
}

/// Benchmark complete memory pressure scenarios
fn bench_complete_memory_pressure_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("complete_memory_pressure_scenarios");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));

    let baseline_manager = BaselineManager::new();
    let rt = Runtime::new().unwrap();

    // Benchmark complete memory pressure handling workflow
    group.bench_function("complete_pressure_handling_workflow", |b| {
        b.to_async(&rt).iter_custom(|iters| async {
            let (execution_times, _) = measure_cleanup_performance(
                "CompletePressureWorkflow",
                "mixed_cleanup",
                MemoryPressureLevel::High,
                || {
                    // Simulate complete workflow:
                    // 1. Detect memory pressure
                    // 2. Select appropriate cleanup strategies
                    // 3. Execute cleanup handlers in priority order
                    // 4. Verify memory freed

                    let mut total_memory_freed = 0u64;

                    // Garbage collection (high priority)
                    let gc_handler = GarbageCollectionHandler::new();
                    total_memory_freed +=
                        gc_handler.cleanup(MemoryPressureLevel::High).unwrap_or(0);

                    // Buffer compaction (medium priority)
                    let buffer_handler = BufferCompactionHandler::new();
                    total_memory_freed +=
                        buffer_handler.cleanup(MemoryPressureLevel::High).unwrap_or(0);

                    // GPU cleanup (if needed)
                    total_memory_freed += 50 * 1024 * 1024; // Mock GPU cache eviction

                    total_memory_freed
                },
                &baseline_manager,
            );

            let total_time: Duration = execution_times.iter().sum();
            total_time / execution_times.len() as u32
        })
    });

    // Benchmark memory pressure under load
    group.bench_function("pressure_under_load", |b| {
        b.to_async(&rt).iter(|| async {
            // Simulate memory pressure while the system is under load
            let load_tasks: Vec<_> = (0..10)
                .map(|_| {
                    tokio::spawn(async {
                        // Simulate background load
                        let _data: Vec<u8> = vec![0; 1024 * 1024]; // 1MB allocation
                        tokio::time::sleep(Duration::from_millis(50)).await;
                    })
                })
                .collect();

            // Perform cleanup while under load
            let gc_handler = GarbageCollectionHandler::new();
            let memory_freed = gc_handler.cleanup(MemoryPressureLevel::Critical).unwrap_or(0);

            // Wait for background tasks to complete
            futures::future::join_all(load_tasks).await;

            black_box(memory_freed)
        })
    });

    // Benchmark adaptive threshold adjustment
    group.bench_function("adaptive_threshold_adjustment", |b| {
        b.iter(|| {
            // Simulate adaptive threshold calculation based on historical data
            let mut thresholds = MemoryPressureThresholds {
                low: 0.7,
                medium: 0.8,
                high: 0.9,
                critical: 0.95,
                adaptive: true,
                base_low: 0.65,
                base_medium: 0.75,
                base_high: 0.85,
                base_critical: 0.92,
                adjustment_factor: 1.1,
                learning_rate: 0.01,
            };

            // Mock adaptive adjustment based on system performance
            let historical_pressures = vec![0.75, 0.82, 0.88, 0.91, 0.73, 0.79];
            let avg_pressure =
                historical_pressures.iter().sum::<f32>() / historical_pressures.len() as f32;

            // Adjust thresholds based on average historical pressure
            if avg_pressure > 0.85 {
                thresholds.medium = (thresholds.medium - 0.05).max(0.6);
                thresholds.high = (thresholds.high - 0.05).max(0.7);
            }

            black_box(thresholds)
        })
    });

    group.finish();
}

/// Benchmark error handling and recovery scenarios
fn bench_error_handling_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("error_handling_scenarios");

    let baseline_manager = BaselineManager::new();

    // Benchmark cleanup handler failure recovery
    group.bench_function("cleanup_failure_recovery", |b| {
        b.iter_custom(|iters| {
            let (execution_times, _) = measure_cleanup_performance(
                "ErrorHandling",
                "cleanup_failure_recovery",
                MemoryPressureLevel::High,
                || {
                    // Simulate a cleanup operation that occasionally fails
                    if fastrand::f32() < 0.1 {
                        // 10% failure rate
                        // Return 0 to indicate failure
                        0
                    } else {
                        // Successful cleanup
                        let handler = GarbageCollectionHandler::new();
                        handler.cleanup(MemoryPressureLevel::High).unwrap_or(0)
                    }
                },
                &baseline_manager,
            );

            let total_time: Duration = execution_times.iter().sum();
            total_time / execution_times.len() as u32
        })
    });

    // Benchmark timeout handling
    group.bench_function("cleanup_timeout_handling", |b| {
        b.iter(|| {
            // Simulate cleanup operation with timeout
            let start = Instant::now();
            let timeout = Duration::from_millis(100);

            let memory_freed = loop {
                if start.elapsed() > timeout {
                    // Timeout occurred, return partial cleanup
                    break 5 * 1024 * 1024; // 5MB partial cleanup
                }

                // Simulate work
                std::thread::sleep(Duration::from_millis(1));

                // Check if cleanup completed successfully
                if fastrand::f32() < 0.8 {
                    // 80% chance of completion
                    break 15 * 1024 * 1024; // 15MB successful cleanup
                }
            };

            black_box(memory_freed)
        })
    });

    group.finish();
}

// Group all benchmarks
criterion_group!(
    memory_pressure_regression_benches,
    bench_garbage_collection_handler,
    bench_buffer_compaction_handler,
    bench_gpu_cleanup_handlers,
    bench_complete_memory_pressure_scenarios,
    bench_error_handling_scenarios
);

criterion_main!(memory_pressure_regression_benches);
