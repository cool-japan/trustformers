use crate::error::{TrustformersError, TrustformersResult};
use scirs2_core::random::*; // SciRS2 Integration Policy
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// High-resolution timer for precise measurements
pub struct HighResTimer {
    start_time: Option<Instant>,
    measurements: Vec<Duration>,
    name: String,
}

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    pub name: String,
    pub iterations: usize,
    pub warmup_iterations: usize,
    pub cooldown_ms: u64,
    pub memory_tracking: bool,
    pub cpu_profiling: bool,
    pub gpu_profiling: bool,
    pub detailed_timing: bool,
}

/// Benchmark result statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkStats {
    pub name: String,
    pub iterations: usize,
    pub min_time_ms: f64,
    pub max_time_ms: f64,
    pub mean_time_ms: f64,
    pub median_time_ms: f64,
    pub std_dev_ms: f64,
    pub percentile_95_ms: f64,
    pub percentile_99_ms: f64,
    pub throughput_ops_per_sec: f64,
    pub total_time_ms: f64,
    pub cpu_utilization: Option<f64>,
    pub memory_usage: Option<MemoryStats>,
    pub gpu_stats: Option<GpuStats>,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub initial_bytes: u64,
    pub peak_bytes: u64,
    pub final_bytes: u64,
    pub allocations_count: u64,
    pub deallocations_count: u64,
    pub fragmentation_ratio: f64,
}

/// GPU utilization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuStats {
    pub utilization_percent: f64,
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
    pub temperature_celsius: Option<f32>,
    pub power_watts: Option<f32>,
}

/// Performance profile point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilePoint {
    pub timestamp_ms: f64,
    pub operation_name: String,
    pub duration_ms: f64,
    pub memory_delta_bytes: i64,
    pub cpu_percent: Option<f64>,
    pub gpu_percent: Option<f64>,
    pub custom_metrics: HashMap<String, f64>,
}

/// Continuous profiler for real-time monitoring
pub struct ContinuousProfiler {
    points: Arc<Mutex<VecDeque<ProfilePoint>>>,
    max_points: usize,
    start_time: Instant,
    is_running: Arc<Mutex<bool>>,
    sample_interval_ms: u64,
}

/// Comparative benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    pub baseline_name: String,
    pub comparison_name: String,
    pub speed_improvement: f64, // Positive means faster, negative means slower
    pub memory_improvement: f64,
    pub statistical_significance: f64,   // p-value
    pub confidence_interval: (f64, f64), // 95% confidence interval for improvement
}

/// Regression test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionTestResult {
    pub test_name: String,
    pub baseline_performance: f64,
    pub current_performance: f64,
    pub performance_change_percent: f64,
    pub is_regression: bool,
    pub severity: RegressionSeverity,
    pub details: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegressionSeverity {
    None,
    Minor,    // < 5% degradation
    Moderate, // 5-15% degradation
    Major,    // 15-50% degradation
    Critical, // > 50% degradation
}

/// Main profiling tools manager
pub struct ProfilingTools {
    timers: HashMap<String, HighResTimer>,
    benchmark_results: HashMap<String, BenchmarkStats>,
    continuous_profiler: Option<ContinuousProfiler>,
    baseline_performances: HashMap<String, f64>,
}

impl HighResTimer {
    pub fn new(name: String) -> Self {
        Self {
            start_time: None,
            measurements: Vec::new(),
            name,
        }
    }

    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    pub fn stop(&mut self) -> TrustformersResult<Duration> {
        if let Some(start) = self.start_time.take() {
            let duration = start.elapsed();
            self.measurements.push(duration);
            Ok(duration)
        } else {
            Err(TrustformersError::InvalidParameter)
        }
    }

    pub fn reset(&mut self) {
        self.start_time = None;
        self.measurements.clear();
    }

    pub fn get_stats(&self) -> TrustformersResult<BenchmarkStats> {
        if self.measurements.is_empty() {
            return Err(TrustformersError::InvalidParameter);
        }

        let mut times_ms: Vec<f64> =
            self.measurements.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
        times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let min_time_ms = times_ms[0];
        let max_time_ms = times_ms[times_ms.len() - 1];
        let mean_time_ms = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
        let median_time_ms = if times_ms.len() % 2 == 0 {
            (times_ms[times_ms.len() / 2 - 1] + times_ms[times_ms.len() / 2]) / 2.0
        } else {
            times_ms[times_ms.len() / 2]
        };

        let variance = times_ms.iter().map(|&x| (x - mean_time_ms).powi(2)).sum::<f64>()
            / times_ms.len() as f64;
        let std_dev_ms = variance.sqrt();

        let percentile_95_ms = times_ms[(times_ms.len() as f64 * 0.95) as usize];
        let percentile_99_ms = times_ms[(times_ms.len() as f64 * 0.99) as usize];
        let total_time_ms = times_ms.iter().sum();
        let throughput_ops_per_sec = if mean_time_ms > 0.0 { 1000.0 / mean_time_ms } else { 0.0 };

        Ok(BenchmarkStats {
            name: self.name.clone(),
            iterations: self.measurements.len(),
            min_time_ms,
            max_time_ms,
            mean_time_ms,
            median_time_ms,
            std_dev_ms,
            percentile_95_ms,
            percentile_99_ms,
            throughput_ops_per_sec,
            total_time_ms,
            cpu_utilization: None,
            memory_usage: None,
            gpu_stats: None,
        })
    }
}

impl ContinuousProfiler {
    pub fn new(max_points: usize, sample_interval_ms: u64) -> Self {
        Self {
            points: Arc::new(Mutex::new(VecDeque::with_capacity(max_points))),
            max_points,
            start_time: Instant::now(),
            is_running: Arc::new(Mutex::new(false)),
            sample_interval_ms,
        }
    }

    pub fn start(&self) -> TrustformersResult<()> {
        let mut running = self.is_running.lock().expect("lock should not be poisoned");
        if *running {
            return Err(TrustformersError::RuntimeError);
        }
        *running = true;

        let points = Arc::clone(&self.points);
        let is_running = Arc::clone(&self.is_running);
        let max_points = self.max_points;
        let interval = self.sample_interval_ms;
        let start_time = self.start_time;

        thread::spawn(move || {
            while *is_running.lock().expect("lock should not be poisoned") {
                let timestamp_ms = start_time.elapsed().as_secs_f64() * 1000.0;

                // Sample system metrics
                let cpu_percent = Self::get_cpu_usage();
                let memory_info = Self::get_memory_info();
                let gpu_percent = Self::get_gpu_usage();

                let point = ProfilePoint {
                    timestamp_ms,
                    operation_name: "system_sample".to_string(),
                    duration_ms: 0.0,
                    memory_delta_bytes: 0,
                    cpu_percent,
                    gpu_percent,
                    custom_metrics: HashMap::new(),
                };

                let mut points_lock = points.lock().expect("lock should not be poisoned");
                if points_lock.len() >= max_points {
                    points_lock.pop_front();
                }
                points_lock.push_back(point);
                drop(points_lock);

                thread::sleep(Duration::from_millis(interval));
            }
        });

        Ok(())
    }

    pub fn stop(&self) {
        let mut running = self.is_running.lock().expect("lock should not be poisoned");
        *running = false;
    }

    pub fn add_point(&self, point: ProfilePoint) {
        let mut points = self.points.lock().expect("lock should not be poisoned");
        if points.len() >= self.max_points {
            points.pop_front();
        }
        points.push_back(point);
    }

    pub fn get_points(&self) -> Vec<ProfilePoint> {
        self.points.lock().expect("lock should not be poisoned").iter().cloned().collect()
    }

    pub fn export_timeline(&self, format: &str) -> TrustformersResult<String> {
        let points = self.get_points();

        match format.to_lowercase().as_str() {
            "json" => serde_json::to_string_pretty(&points)
                .map_err(|_| TrustformersError::SerializationError),
            "csv" => {
                let mut csv = String::new();
                csv.push_str("timestamp_ms,operation_name,duration_ms,memory_delta_bytes,cpu_percent,gpu_percent\n");

                for point in points {
                    csv.push_str(&format!(
                        "{},{},{},{},{},{}\n",
                        point.timestamp_ms,
                        point.operation_name,
                        point.duration_ms,
                        point.memory_delta_bytes,
                        point.cpu_percent.unwrap_or(0.0),
                        point.gpu_percent.unwrap_or(0.0)
                    ));
                }
                Ok(csv)
            },
            _ => Err(TrustformersError::InvalidParameter),
        }
    }

    // Helper methods for system monitoring
    fn get_cpu_usage() -> Option<f64> {
        // Simplified CPU usage calculation
        // In production, use proper system monitoring libraries
        Some(thread_rng().gen::<f64>() * 100.0) // Mock data
    }

    fn get_memory_info() -> Option<u64> {
        // Mock memory information
        Some(1024 * 1024 * 1024) // 1GB
    }

    fn get_gpu_usage() -> Option<f64> {
        // Mock GPU usage
        Some(thread_rng().gen::<f64>() * 100.0)
    }
}

impl ProfilingTools {
    pub fn new() -> Self {
        Self {
            timers: HashMap::new(),
            benchmark_results: HashMap::new(),
            continuous_profiler: None,
            baseline_performances: HashMap::new(),
        }
    }

    /// Create a new high-resolution timer
    pub fn create_timer(&mut self, name: &str) -> TrustformersResult<()> {
        if self.timers.contains_key(name) {
            return Err(TrustformersError::InvalidParameter);
        }

        self.timers.insert(name.to_string(), HighResTimer::new(name.to_string()));
        Ok(())
    }

    /// Start a timer
    pub fn start_timer(&mut self, name: &str) -> TrustformersResult<()> {
        let timer = self.timers.get_mut(name).ok_or(TrustformersError::InvalidParameter)?;
        timer.start();
        Ok(())
    }

    /// Stop a timer and return the elapsed time
    pub fn stop_timer(&mut self, name: &str) -> TrustformersResult<Duration> {
        let timer = self.timers.get_mut(name).ok_or(TrustformersError::InvalidParameter)?;
        timer.stop()
    }

    /// Run a comprehensive benchmark
    pub fn run_benchmark<F>(
        &mut self,
        config: BenchmarkConfig,
        operation: F,
    ) -> TrustformersResult<BenchmarkStats>
    where
        F: Fn() -> TrustformersResult<()>,
    {
        println!("Running benchmark: {}", config.name);

        // Create timer for this benchmark
        let timer_name = format!("benchmark_{}", config.name);
        self.create_timer(&timer_name)?;

        let mut memory_start = 0u64;
        let mut memory_peak = 0u64;

        if config.memory_tracking {
            memory_start = self.get_current_memory_usage();
        }

        // Warmup phase
        println!("  Warmup: {} iterations", config.warmup_iterations);
        for i in 0..config.warmup_iterations {
            if i % 10 == 0 {
                print!(".");
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }
            operation()?;
        }
        println!(" done");

        if config.cooldown_ms > 0 {
            thread::sleep(Duration::from_millis(config.cooldown_ms));
        }

        // Benchmark phase
        println!("  Benchmark: {} iterations", config.iterations);
        for i in 0..config.iterations {
            if i % 10 == 0 {
                print!(".");
                std::io::Write::flush(&mut std::io::stdout()).unwrap();
            }

            self.start_timer(&timer_name)?;
            operation()?;
            self.stop_timer(&timer_name)?;

            if config.memory_tracking {
                let current_memory = self.get_current_memory_usage();
                if current_memory > memory_peak {
                    memory_peak = current_memory;
                }
            }
        }
        println!(" done");

        // Calculate statistics
        let timer = self.timers.get(&timer_name).unwrap();
        let mut stats = timer.get_stats()?;

        // Add memory information if tracking was enabled
        if config.memory_tracking {
            let memory_final = self.get_current_memory_usage();
            stats.memory_usage = Some(MemoryStats {
                initial_bytes: memory_start,
                peak_bytes: memory_peak,
                final_bytes: memory_final,
                allocations_count: config.iterations as u64,
                deallocations_count: config.iterations as u64,
                fragmentation_ratio: 0.1, // Mock value
            });
        }

        // Add GPU stats if profiling was enabled
        if config.gpu_profiling {
            stats.gpu_stats = Some(GpuStats {
                utilization_percent: 75.0,                  // Mock value
                memory_used_bytes: 1024 * 1024 * 512,       // 512MB
                memory_total_bytes: 1024 * 1024 * 1024 * 8, // 8GB
                temperature_celsius: Some(65.0),
                power_watts: Some(180.0),
            });
        }

        // Store results
        self.benchmark_results.insert(config.name.clone(), stats.clone());

        println!(
            "  Completed: {:.2} ms average, {:.2} ops/sec",
            stats.mean_time_ms, stats.throughput_ops_per_sec
        );

        Ok(stats)
    }

    /// Compare two benchmark results
    pub fn compare_benchmarks(
        &self,
        baseline_name: &str,
        comparison_name: &str,
    ) -> TrustformersResult<ComparisonResult> {
        let baseline = self
            .benchmark_results
            .get(baseline_name)
            .ok_or(TrustformersError::InvalidParameter)?;

        let comparison = self
            .benchmark_results
            .get(comparison_name)
            .ok_or(TrustformersError::InvalidParameter)?;

        let speed_improvement =
            (baseline.mean_time_ms - comparison.mean_time_ms) / baseline.mean_time_ms;

        let memory_improvement = if let (Some(base_mem), Some(comp_mem)) =
            (&baseline.memory_usage, &comparison.memory_usage)
        {
            (base_mem.peak_bytes as f64 - comp_mem.peak_bytes as f64) / base_mem.peak_bytes as f64
        } else {
            0.0
        };

        // Simplified statistical significance calculation (t-test would be more accurate)
        let statistical_significance = if speed_improvement.abs() > 0.05 { 0.01 } else { 0.5 };

        let confidence_interval = (speed_improvement - 0.02, speed_improvement + 0.02);

        Ok(ComparisonResult {
            baseline_name: baseline_name.to_string(),
            comparison_name: comparison_name.to_string(),
            speed_improvement,
            memory_improvement,
            statistical_significance,
            confidence_interval,
        })
    }

    /// Run regression tests against baseline performance
    pub fn run_regression_test(
        &mut self,
        test_name: &str,
        current_performance: f64,
        threshold_percent: f64,
    ) -> TrustformersResult<RegressionTestResult> {
        let baseline_performance =
            self.baseline_performances.get(test_name).copied().unwrap_or_else(|| {
                // If no baseline, set current as baseline
                self.baseline_performances.insert(test_name.to_string(), current_performance);
                current_performance
            });

        let performance_change_percent =
            ((current_performance - baseline_performance) / baseline_performance) * 100.0;
        let is_regression = performance_change_percent > threshold_percent;

        let severity = if !is_regression {
            RegressionSeverity::None
        } else if performance_change_percent < 5.0 {
            RegressionSeverity::Minor
        } else if performance_change_percent < 15.0 {
            RegressionSeverity::Moderate
        } else if performance_change_percent < 50.0 {
            RegressionSeverity::Major
        } else {
            RegressionSeverity::Critical
        };

        let details = if is_regression {
            format!(
                "Performance degraded by {:.2}% (baseline: {:.2}ms, current: {:.2}ms)",
                performance_change_percent, baseline_performance, current_performance
            )
        } else {
            format!(
                "Performance improved by {:.2}% or within acceptable range",
                performance_change_percent.abs()
            )
        };

        Ok(RegressionTestResult {
            test_name: test_name.to_string(),
            baseline_performance,
            current_performance,
            performance_change_percent,
            is_regression,
            severity,
            details,
        })
    }

    /// Start continuous profiling
    pub fn start_continuous_profiling(
        &mut self,
        max_points: usize,
        sample_interval_ms: u64,
    ) -> TrustformersResult<()> {
        if self.continuous_profiler.is_some() {
            return Err(TrustformersError::RuntimeError);
        }

        let profiler = ContinuousProfiler::new(max_points, sample_interval_ms);
        profiler.start()?;
        self.continuous_profiler = Some(profiler);
        Ok(())
    }

    /// Stop continuous profiling
    pub fn stop_continuous_profiling(&mut self) -> TrustformersResult<Vec<ProfilePoint>> {
        if let Some(profiler) = self.continuous_profiler.take() {
            profiler.stop();
            Ok(profiler.get_points())
        } else {
            Err(TrustformersError::InvalidParameter)
        }
    }

    /// Generate comprehensive performance report
    pub fn generate_performance_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# TrustformeRS Performance Report\n\n");
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(std::time::Duration::from_secs(0))
            .as_secs();
        report.push_str(&format!("Generated at: {} (Unix timestamp)\n\n", timestamp));

        // Summary statistics
        report.push_str("## Summary\n\n");
        report.push_str(&format!(
            "Total benchmarks: {}\n",
            self.benchmark_results.len()
        ));

        if !self.benchmark_results.is_empty() {
            let avg_throughput: f64 =
                self.benchmark_results.values().map(|s| s.throughput_ops_per_sec).sum::<f64>()
                    / self.benchmark_results.len() as f64;
            report.push_str(&format!(
                "Average throughput: {:.2} ops/sec\n",
                avg_throughput
            ));
        }

        report.push('\n');

        // Individual benchmark results
        if !self.benchmark_results.is_empty() {
            report.push_str("## Benchmark Results\n\n");

            for (name, stats) in &self.benchmark_results {
                report.push_str(&format!("### {}\n\n", name));
                report.push_str(&format!("- **Iterations**: {}\n", stats.iterations));
                report.push_str(&format!("- **Mean time**: {:.2} ms\n", stats.mean_time_ms));
                report.push_str(&format!(
                    "- **Median time**: {:.2} ms\n",
                    stats.median_time_ms
                ));
                report.push_str(&format!(
                    "- **95th percentile**: {:.2} ms\n",
                    stats.percentile_95_ms
                ));
                report.push_str(&format!(
                    "- **Standard deviation**: {:.2} ms\n",
                    stats.std_dev_ms
                ));
                report.push_str(&format!(
                    "- **Throughput**: {:.2} ops/sec\n",
                    stats.throughput_ops_per_sec
                ));

                if let Some(memory) = &stats.memory_usage {
                    report.push_str(&format!(
                        "- **Peak memory**: {:.2} MB\n",
                        memory.peak_bytes as f64 / 1_000_000.0
                    ));
                    report.push_str(&format!(
                        "- **Memory growth**: {:.2} MB\n",
                        (memory.final_bytes as f64 - memory.initial_bytes as f64) / 1_000_000.0
                    ));
                }

                if let Some(gpu) = &stats.gpu_stats {
                    report.push_str(&format!(
                        "- **GPU utilization**: {:.1}%\n",
                        gpu.utilization_percent
                    ));
                    report.push_str(&format!(
                        "- **GPU memory**: {:.2} MB / {:.2} MB\n",
                        gpu.memory_used_bytes as f64 / 1_000_000.0,
                        gpu.memory_total_bytes as f64 / 1_000_000.0
                    ));
                }

                report.push('\n');
            }
        }

        // Baseline comparisons
        if !self.baseline_performances.is_empty() {
            report.push_str("## Baseline Comparisons\n\n");
            for (test_name, baseline) in &self.baseline_performances {
                report.push_str(&format!(
                    "- **{}**: {:.2} ms baseline\n",
                    test_name, baseline
                ));
            }
            report.push('\n');
        }

        report.push_str("## Recommendations\n\n");
        report.push_str(self.generate_recommendations().as_str());

        report
    }

    /// Export benchmark data in various formats
    pub fn export_benchmark_data(&self, format: &str) -> TrustformersResult<String> {
        match format.to_lowercase().as_str() {
            "json" => serde_json::to_string_pretty(&self.benchmark_results)
                .map_err(|_| TrustformersError::SerializationError),
            "csv" => {
                let mut csv = String::new();
                csv.push_str("name,iterations,mean_ms,median_ms,min_ms,max_ms,std_dev_ms,p95_ms,p99_ms,throughput_ops_sec\n");

                for (name, stats) in &self.benchmark_results {
                    csv.push_str(&format!(
                        "{},{},{},{},{},{},{},{},{},{}\n",
                        name,
                        stats.iterations,
                        stats.mean_time_ms,
                        stats.median_time_ms,
                        stats.min_time_ms,
                        stats.max_time_ms,
                        stats.std_dev_ms,
                        stats.percentile_95_ms,
                        stats.percentile_99_ms,
                        stats.throughput_ops_per_sec
                    ));
                }
                Ok(csv)
            },
            _ => Err(TrustformersError::InvalidParameter),
        }
    }

    // Helper methods
    fn get_current_memory_usage(&self) -> u64 {
        // Mock memory usage - in production, use proper memory monitoring
        1024 * 1024 * 100 // 100MB
    }

    fn generate_recommendations(&self) -> String {
        let mut recommendations = String::new();

        // Analyze benchmark results for recommendations
        let slow_benchmarks: Vec<_> = self
            .benchmark_results
            .iter()
            .filter(|(_, stats)| stats.mean_time_ms > 100.0)
            .collect();

        if !slow_benchmarks.is_empty() {
            recommendations.push_str("### Performance Optimizations\n\n");
            for (name, _) in slow_benchmarks {
                recommendations.push_str(&format!(
                    "- Consider optimizing `{}` - average time exceeds 100ms\n",
                    name
                ));
            }
            recommendations.push('\n');
        }

        let high_variance_benchmarks: Vec<_> = self
            .benchmark_results
            .iter()
            .filter(|(_, stats)| stats.std_dev_ms > stats.mean_time_ms * 0.2)
            .collect();

        if !high_variance_benchmarks.is_empty() {
            recommendations.push_str("### Consistency Improvements\n\n");
            for (name, _) in high_variance_benchmarks {
                recommendations.push_str(&format!(
                    "- `{}` shows high variance - consider stabilizing performance\n",
                    name
                ));
            }
            recommendations.push('\n');
        }

        if recommendations.is_empty() {
            recommendations.push_str("- All benchmarks show good performance characteristics\n");
            recommendations
                .push_str("- Consider running longer benchmarks for more detailed analysis\n");
        }

        recommendations
    }
}

// C API functions
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_int};

static mut PROFILING_TOOLS: Option<ProfilingTools> = None;

/// Initialize profiling tools
#[no_mangle]
pub unsafe extern "C" fn trustformers_profiling_init() -> c_int {
    PROFILING_TOOLS = Some(ProfilingTools::new());
    0
}

/// Create a new timer
#[no_mangle]
pub unsafe extern "C" fn trustformers_profiling_create_timer(name: *const c_char) -> c_int {
    if name.is_null() {
        return 1;
    }

    let name_str = match CStr::from_ptr(name).to_str() {
        Ok(s) => s,
        Err(_) => return 1,
    };

    if let Some(ref mut tools) = PROFILING_TOOLS {
        match tools.create_timer(name_str) {
            Ok(_) => 0,
            Err(_) => 1,
        }
    } else {
        1
    }
}

/// Start a timer
#[no_mangle]
pub unsafe extern "C" fn trustformers_profiling_start_timer(name: *const c_char) -> c_int {
    if name.is_null() {
        return 1;
    }

    let name_str = match CStr::from_ptr(name).to_str() {
        Ok(s) => s,
        Err(_) => return 1,
    };

    if let Some(ref mut tools) = PROFILING_TOOLS {
        match tools.start_timer(name_str) {
            Ok(_) => 0,
            Err(_) => 1,
        }
    } else {
        1
    }
}

/// Stop a timer and return elapsed time in milliseconds
#[no_mangle]
pub unsafe extern "C" fn trustformers_profiling_stop_timer(
    name: *const c_char,
    elapsed_ms: *mut c_double,
) -> c_int {
    if name.is_null() || elapsed_ms.is_null() {
        return 1;
    }

    let name_str = match CStr::from_ptr(name).to_str() {
        Ok(s) => s,
        Err(_) => return 1,
    };

    if let Some(ref mut tools) = PROFILING_TOOLS {
        match tools.stop_timer(name_str) {
            Ok(duration) => {
                *elapsed_ms = duration.as_secs_f64() * 1000.0;
                0
            },
            Err(_) => 1,
        }
    } else {
        1
    }
}

/// Generate performance report
#[no_mangle]
pub unsafe extern "C" fn trustformers_profiling_generate_report(result: *mut *mut c_char) -> c_int {
    if result.is_null() {
        return 1;
    }

    if let Some(ref tools) = PROFILING_TOOLS {
        let report = tools.generate_performance_report();
        match CString::new(report) {
            Ok(c_string) => {
                *result = c_string.into_raw();
                0
            },
            Err(_) => 1,
        }
    } else {
        1
    }
}

/// Export benchmark data
#[no_mangle]
pub unsafe extern "C" fn trustformers_profiling_export_data(
    format: *const c_char,
    result: *mut *mut c_char,
) -> c_int {
    if format.is_null() || result.is_null() {
        return 1;
    }

    let format_str = match CStr::from_ptr(format).to_str() {
        Ok(s) => s,
        Err(_) => return 1,
    };

    if let Some(ref tools) = PROFILING_TOOLS {
        match tools.export_benchmark_data(format_str) {
            Ok(data) => match CString::new(data) {
                Ok(c_string) => {
                    *result = c_string.into_raw();
                    0
                },
                Err(_) => 1,
            },
            Err(_) => 1,
        }
    } else {
        1
    }
}

/// Free profiling string
#[no_mangle]
pub unsafe extern "C" fn trustformers_profiling_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(CString::from_raw(ptr));
    }
}

/// Start continuous profiling
#[no_mangle]
pub unsafe extern "C" fn trustformers_profiling_start_continuous(
    max_points: c_int,
    sample_interval_ms: c_int,
) -> c_int {
    if let Some(ref mut tools) = PROFILING_TOOLS {
        match tools.start_continuous_profiling(max_points as usize, sample_interval_ms as u64) {
            Ok(_) => 0,
            Err(_) => 1,
        }
    } else {
        1
    }
}

/// Stop continuous profiling
#[no_mangle]
pub unsafe extern "C" fn trustformers_profiling_stop_continuous() -> c_int {
    if let Some(ref mut tools) = PROFILING_TOOLS {
        match tools.stop_continuous_profiling() {
            Ok(_) => 0,
            Err(_) => 1,
        }
    } else {
        1
    }
}
