//! Performance Benchmarking and Timing Utilities
//!
//! This module provides comprehensive performance measurement capabilities including
//! timing, benchmarking, statistics calculation, and performance monitoring utilities.

use super::types::*;
use crate::error::{TrustformersError, TrustformersResult};
use std::time::Instant;

/// Performance timer for benchmarking operations
pub struct PerformanceTimer {
    measurements: Vec<f64>,
    start_time: Option<Instant>,
}

impl PerformanceTimer {
    /// Create a new performance timer
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
            start_time: None,
        }
    }

    /// Start timing an operation
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Stop timing and record the measurement
    pub fn stop(&mut self) -> TrustformersResult<f64> {
        if let Some(start) = self.start_time.take() {
            let elapsed = start.elapsed().as_secs_f64() * 1000.0; // Convert to milliseconds
            self.measurements.push(elapsed);
            Ok(elapsed)
        } else {
            Err(TrustformersError::RuntimeError)
        }
    }

    /// Get comprehensive statistics for all measurements
    pub fn get_statistics(&self) -> TrustformersBenchmarkResult {
        if self.measurements.is_empty() {
            return TrustformersBenchmarkResult::default();
        }

        let total_time: f64 = self.measurements.iter().sum();
        let avg_time = total_time / self.measurements.len() as f64;
        let min_time = self.measurements.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_time = self.measurements.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Calculate standard deviation
        let variance: f64 = self.measurements.iter().map(|&x| (x - avg_time).powi(2)).sum::<f64>()
            / self.measurements.len() as f64;
        let std_dev = variance.sqrt();

        // Calculate throughput (operations per second)
        let throughput = if avg_time > 0.0 {
            1000.0 / avg_time // Convert from ms to seconds
        } else {
            0.0
        };

        TrustformersBenchmarkResult {
            iterations: self.measurements.len() as u64,
            total_time_ms: total_time,
            avg_time_ms: avg_time,
            min_time_ms: min_time,
            max_time_ms: max_time,
            std_dev_ms: std_dev,
            throughput_ops: throughput,
        }
    }

    /// Reset all measurements
    pub fn reset(&mut self) {
        self.measurements.clear();
        self.start_time = None;
    }

    /// Get the number of measurements recorded
    pub fn measurement_count(&self) -> usize {
        self.measurements.len()
    }

    /// Get all raw measurements
    pub fn get_measurements(&self) -> &[f64] {
        &self.measurements
    }

    /// Add a manual measurement (for pre-calculated timings)
    pub fn add_measurement(&mut self, measurement_ms: f64) {
        self.measurements.push(measurement_ms);
    }

    /// Get percentile statistics
    pub fn get_percentiles(&self) -> PercentileStats {
        if self.measurements.is_empty() {
            return PercentileStats::default();
        }

        let mut sorted = self.measurements.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let len = sorted.len();
        let get_percentile = |p: f64| {
            let index = ((len as f64 - 1.0) * p / 100.0).round() as usize;
            sorted[index.min(len - 1)]
        };

        PercentileStats {
            p50: get_percentile(50.0),
            p75: get_percentile(75.0),
            p90: get_percentile(90.0),
            p95: get_percentile(95.0),
            p99: get_percentile(99.0),
        }
    }
}

impl Default for PerformanceTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Percentile statistics for performance analysis
#[derive(Debug, Default)]
pub struct PercentileStats {
    pub p50: f64, // Median
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

/// High-resolution timer for micro-benchmarking
pub struct MicroTimer {
    start: Option<Instant>,
}

impl MicroTimer {
    /// Create a new micro timer
    pub fn new() -> Self {
        Self { start: None }
    }

    /// Start the timer
    pub fn start(&mut self) {
        self.start = Some(Instant::now());
    }

    /// Stop and get elapsed time in nanoseconds
    pub fn stop_nanos(&mut self) -> TrustformersResult<u128> {
        if let Some(start) = self.start.take() {
            Ok(start.elapsed().as_nanos())
        } else {
            Err(TrustformersError::RuntimeError)
        }
    }

    /// Stop and get elapsed time in microseconds
    pub fn stop_micros(&mut self) -> TrustformersResult<u128> {
        if let Some(start) = self.start.take() {
            Ok(start.elapsed().as_micros())
        } else {
            Err(TrustformersError::RuntimeError)
        }
    }

    /// Stop and get elapsed time in milliseconds
    pub fn stop_millis(&mut self) -> TrustformersResult<f64> {
        if let Some(start) = self.start.take() {
            Ok(start.elapsed().as_secs_f64() * 1000.0)
        } else {
            Err(TrustformersError::RuntimeError)
        }
    }
}

impl Default for MicroTimer {
    fn default() -> Self {
        Self::new()
    }
}

/// Benchmarking utilities for performance testing
pub mod benchmark_utils {
    use super::*;

    /// Run a benchmark function multiple times and collect statistics
    pub fn benchmark_function<F>(
        mut func: F,
        iterations: usize,
        warmup_iterations: usize,
    ) -> TrustformersBenchmarkResult
    where
        F: FnMut(),
    {
        // Warmup phase
        for _ in 0..warmup_iterations {
            func();
        }

        // Actual benchmarking
        let mut timer = PerformanceTimer::new();
        for _ in 0..iterations {
            timer.start();
            func();
            let _ = timer.stop();
        }

        timer.get_statistics()
    }

    /// Benchmark a function with automatic iteration count determination
    pub fn benchmark_auto<F>(mut func: F, target_duration_ms: f64) -> TrustformersBenchmarkResult
    where
        F: FnMut(),
    {
        // Quick calibration run to estimate timing
        let mut micro_timer = MicroTimer::new();
        micro_timer.start();
        func();
        let single_run_ms = micro_timer.stop_millis().unwrap_or(1.0);

        // Calculate how many iterations we can fit in target duration
        let estimated_iterations = (target_duration_ms / single_run_ms).max(1.0) as usize;

        benchmark_function(func, estimated_iterations, estimated_iterations / 10)
    }

    /// Compare the performance of two functions
    pub fn benchmark_compare<F1, F2>(
        mut func1: F1,
        mut func2: F2,
        iterations: usize,
    ) -> BenchmarkComparison
    where
        F1: FnMut(),
        F2: FnMut(),
    {
        let warmup = iterations / 10;
        let stats1 = benchmark_function(&mut func1, iterations, warmup);
        let stats2 = benchmark_function(&mut func2, iterations, warmup);

        BenchmarkComparison {
            function1_stats: stats1.clone(),
            function2_stats: stats2.clone(),
            speedup_ratio: if stats2.avg_time_ms > 0.0 {
                stats1.avg_time_ms / stats2.avg_time_ms
            } else {
                0.0
            },
        }
    }

    /// Run a memory allocation benchmark
    pub fn benchmark_allocation<F>(
        mut alloc_func: F,
        iterations: usize,
    ) -> AllocationBenchmarkResult
    where
        F: FnMut() -> Box<dyn std::any::Any>,
    {
        let mut allocations = Vec::with_capacity(iterations);
        let mut timer = PerformanceTimer::new();

        timer.start();
        for _ in 0..iterations {
            let allocation = alloc_func();
            allocations.push(allocation);
        }
        let total_time = timer.stop().unwrap_or(0.0);

        AllocationBenchmarkResult {
            iterations: iterations as u64,
            total_allocation_time_ms: total_time,
            avg_allocation_time_ms: total_time / iterations as f64,
            allocations_per_second: if total_time > 0.0 {
                (iterations as f64 / total_time) * 1000.0
            } else {
                0.0
            },
        }
    }
}

/// Result of comparing two benchmark functions
#[derive(Debug)]
pub struct BenchmarkComparison {
    pub function1_stats: TrustformersBenchmarkResult,
    pub function2_stats: TrustformersBenchmarkResult,
    pub speedup_ratio: f64, // function1_time / function2_time
}

/// Result of allocation benchmarking
#[derive(Debug)]
pub struct AllocationBenchmarkResult {
    pub iterations: u64,
    pub total_allocation_time_ms: f64,
    pub avg_allocation_time_ms: f64,
    pub allocations_per_second: f64,
}

/// Performance monitoring utilities
pub mod monitoring {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    /// Global performance monitor for tracking system-wide performance
    pub struct PerformanceMonitor {
        timers: Arc<Mutex<HashMap<String, PerformanceTimer>>>,
        counters: Arc<Mutex<HashMap<String, u64>>>,
        gauges: Arc<Mutex<HashMap<String, f64>>>,
    }

    impl PerformanceMonitor {
        /// Create a new performance monitor
        pub fn new() -> Self {
            Self {
                timers: Arc::new(Mutex::new(HashMap::new())),
                counters: Arc::new(Mutex::new(HashMap::new())),
                gauges: Arc::new(Mutex::new(HashMap::new())),
            }
        }

        /// Start timing an operation by name
        pub fn start_timer(&self, name: &str) {
            let mut timers = self.timers.lock().expect("lock should not be poisoned");
            let timer = timers.entry(name.to_string()).or_default();
            timer.start();
        }

        /// Stop timing an operation and record the measurement
        pub fn stop_timer(&self, name: &str) -> TrustformersResult<f64> {
            let mut timers = self.timers.lock().expect("lock should not be poisoned");
            if let Some(timer) = timers.get_mut(name) {
                timer.stop()
            } else {
                Err(TrustformersError::InvalidParameter)
            }
        }

        /// Increment a counter
        pub fn increment_counter(&self, name: &str, value: u64) {
            let mut counters = self.counters.lock().expect("lock should not be poisoned");
            *counters.entry(name.to_string()).or_insert(0) += value;
        }

        /// Set a gauge value
        pub fn set_gauge(&self, name: &str, value: f64) {
            let mut gauges = self.gauges.lock().expect("lock should not be poisoned");
            gauges.insert(name.to_string(), value);
        }

        /// Get timer statistics by name
        pub fn get_timer_stats(&self, name: &str) -> Option<TrustformersBenchmarkResult> {
            let timers = self.timers.lock().expect("lock should not be poisoned");
            timers.get(name).map(|timer| timer.get_statistics())
        }

        /// Get counter value by name
        pub fn get_counter(&self, name: &str) -> u64 {
            let counters = self.counters.lock().expect("lock should not be poisoned");
            counters.get(name).copied().unwrap_or(0)
        }

        /// Get gauge value by name
        pub fn get_gauge(&self, name: &str) -> Option<f64> {
            let gauges = self.gauges.lock().expect("lock should not be poisoned");
            gauges.get(name).copied()
        }

        /// Get all timer names
        pub fn get_timer_names(&self) -> Vec<String> {
            let timers = self.timers.lock().expect("lock should not be poisoned");
            timers.keys().cloned().collect()
        }

        /// Reset all metrics
        pub fn reset_all(&self) {
            let mut timers = self.timers.lock().expect("lock should not be poisoned");
            let mut counters = self.counters.lock().expect("lock should not be poisoned");
            let mut gauges = self.gauges.lock().expect("lock should not be poisoned");

            for timer in timers.values_mut() {
                timer.reset();
            }
            counters.clear();
            gauges.clear();
        }
    }

    impl Default for PerformanceMonitor {
        fn default() -> Self {
            Self::new()
        }
    }

    /// RAII timer that automatically measures scope duration
    pub struct ScopeTimer {
        name: String,
        monitor: Arc<PerformanceMonitor>,
        start: Instant,
    }

    impl ScopeTimer {
        /// Create a new scope timer
        pub fn new(name: String, monitor: Arc<PerformanceMonitor>) -> Self {
            Self {
                name,
                monitor,
                start: Instant::now(),
            }
        }
    }

    impl Drop for ScopeTimer {
        fn drop(&mut self) {
            let elapsed = self.start.elapsed().as_secs_f64() * 1000.0;
            let mut timers = self.monitor.timers.lock().expect("lock should not be poisoned");
            let timer = timers.entry(self.name.clone()).or_default();
            timer.add_measurement(elapsed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_performance_timer() {
        let mut timer = PerformanceTimer::new();

        timer.start();
        thread::sleep(Duration::from_millis(10));
        let elapsed = timer.stop().unwrap();

        assert!(elapsed >= 10.0);
        assert!(elapsed < 50.0); // Should be close to 10ms, allowing for some variance

        let stats = timer.get_statistics();
        assert_eq!(stats.iterations, 1);
        assert!(stats.avg_time_ms >= 10.0);
    }

    #[test]
    fn test_micro_timer() {
        let mut timer = MicroTimer::new();

        timer.start();
        thread::sleep(Duration::from_millis(1));
        let elapsed_micros = timer.stop_micros().unwrap();

        assert!(elapsed_micros >= 1000); // At least 1ms in microseconds
    }

    #[test]
    fn test_benchmark_function() {
        let result = benchmark_utils::benchmark_function(
            || {
                // Simulate some work with measurable duration
                std::thread::sleep(Duration::from_micros(100));
            },
            10,
            2,
        );

        assert_eq!(result.iterations, 10);
        // avg_time_ms should be at least 0.1ms (100 microseconds)
        assert!(result.avg_time_ms >= 0.1, "avg_time_ms was {}", result.avg_time_ms);
    }
}
