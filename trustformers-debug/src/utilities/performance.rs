//! Performance monitoring and profiling utilities

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Performance monitoring utilities
#[derive(Debug)]
pub struct PerformanceMonitor {
    start_time: Instant,
    checkpoints: HashMap<String, Instant>,
    durations: HashMap<String, Duration>,
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            checkpoints: HashMap::new(),
            durations: HashMap::new(),
        }
    }

    pub fn checkpoint(&mut self, name: &str) {
        self.checkpoints.insert(name.to_string(), Instant::now());
    }

    pub fn end_checkpoint(&mut self, name: &str) -> Option<Duration> {
        if let Some(start) = self.checkpoints.remove(name) {
            let duration = start.elapsed();
            self.durations.insert(name.to_string(), duration);
            Some(duration)
        } else {
            None
        }
    }

    pub fn total_elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub fn get_durations(&self) -> &HashMap<String, Duration> {
        &self.durations
    }

    pub fn performance_report(&self) -> String {
        let mut report = format!(
            "Performance Report - Total: {:.2}ms\n",
            self.total_elapsed().as_millis()
        );

        for (name, duration) in &self.durations {
            report.push_str(&format!("  {}: {:.2}ms\n", name, duration.as_millis()));
        }

        report
    }

    /// Get detailed performance metrics
    pub fn get_detailed_metrics(&self) -> PerformanceMetrics {
        let total_duration = self.total_elapsed();
        let checkpoint_count = self.durations.len();

        let avg_checkpoint_duration = if checkpoint_count > 0 {
            self.durations.values().map(|d| d.as_millis() as f64).sum::<f64>()
                / checkpoint_count as f64
        } else {
            0.0
        };

        let slowest_checkpoint = self
            .durations
            .iter()
            .max_by_key(|(_, duration)| *duration)
            .map(|(name, duration)| (name.clone(), *duration));

        let fastest_checkpoint = self
            .durations
            .iter()
            .min_by_key(|(_, duration)| *duration)
            .map(|(name, duration)| (name.clone(), *duration));

        PerformanceMetrics {
            total_duration,
            checkpoint_count,
            avg_checkpoint_duration,
            slowest_checkpoint,
            fastest_checkpoint,
            durations: self.durations.clone(),
        }
    }

    /// Analyze performance bottlenecks
    pub fn analyze_bottlenecks(&self, threshold_percentile: f64) -> BottleneckAnalysis {
        let mut duration_values: Vec<u128> =
            self.durations.values().map(|d| d.as_millis()).collect();
        duration_values.sort();

        let threshold_index = ((duration_values.len() as f64 * threshold_percentile) as usize)
            .min(duration_values.len().saturating_sub(1));
        let threshold = duration_values.get(threshold_index).copied().unwrap_or(0);

        let bottlenecks: Vec<PerformanceBottleneck> = self
            .durations
            .iter()
            .filter(|(_, duration)| duration.as_millis() >= threshold)
            .map(|(name, duration)| PerformanceBottleneck {
                checkpoint_name: name.clone(),
                duration: *duration,
                severity: Self::classify_bottleneck_severity(
                    duration.as_millis(),
                    &duration_values,
                ),
                recommendation: Self::generate_bottleneck_recommendation(name, *duration),
            })
            .collect();

        let total_bottleneck_time: Duration = bottlenecks.iter().map(|b| b.duration).sum();

        BottleneckAnalysis {
            threshold_ms: threshold,
            bottlenecks,
            total_bottleneck_time,
            bottleneck_percentage: if self.total_elapsed().as_millis() > 0 {
                (total_bottleneck_time.as_millis() as f64 / self.total_elapsed().as_millis() as f64)
                    * 100.0
            } else {
                0.0
            },
        }
    }

    fn classify_bottleneck_severity(
        duration_ms: u128,
        all_durations: &[u128],
    ) -> BottleneckSeverity {
        if all_durations.is_empty() {
            return BottleneckSeverity::Low;
        }

        let max_duration = all_durations.iter().max().copied().unwrap_or(0);
        let avg_duration = all_durations.iter().sum::<u128>() / all_durations.len() as u128;

        if duration_ms >= max_duration {
            BottleneckSeverity::Critical
        } else if duration_ms > avg_duration * 3 {
            BottleneckSeverity::High
        } else if duration_ms > avg_duration * 2 {
            BottleneckSeverity::Medium
        } else {
            BottleneckSeverity::Low
        }
    }

    fn generate_bottleneck_recommendation(checkpoint_name: &str, duration: Duration) -> String {
        let duration_ms = duration.as_millis();

        match checkpoint_name {
            name if name.contains("forward") => {
                if duration_ms > 1000 {
                    "Consider model pruning or quantization to reduce forward pass time".to_string()
                } else {
                    "Monitor forward pass efficiency".to_string()
                }
            },
            name if name.contains("backward") => {
                if duration_ms > 2000 {
                    "Consider gradient accumulation or mixed precision training".to_string()
                } else {
                    "Monitor backward pass efficiency".to_string()
                }
            },
            name if name.contains("data") => {
                "Consider data loading optimization or caching".to_string()
            },
            name if name.contains("io") => {
                "Consider I/O optimization or async processing".to_string()
            },
            _ => {
                format!(
                    "Optimize '{}' operation - duration: {}ms",
                    checkpoint_name, duration_ms
                )
            },
        }
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Detailed performance metrics
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_duration: Duration,
    pub checkpoint_count: usize,
    pub avg_checkpoint_duration: f64,
    pub slowest_checkpoint: Option<(String, Duration)>,
    pub fastest_checkpoint: Option<(String, Duration)>,
    pub durations: HashMap<String, Duration>,
}

/// Performance bottleneck analysis
#[derive(Debug, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    pub threshold_ms: u128,
    pub bottlenecks: Vec<PerformanceBottleneck>,
    pub total_bottleneck_time: Duration,
    pub bottleneck_percentage: f64,
}

/// Individual performance bottleneck
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    pub checkpoint_name: String,
    pub duration: Duration,
    pub severity: BottleneckSeverity,
    pub recommendation: String,
}

/// Bottleneck severity levels
#[derive(Debug, Serialize, Deserialize)]
pub enum BottleneckSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Memory performance monitoring
#[derive(Debug)]
pub struct SystemMemoryProfiler {
    /// RSS at construction; `None` when the platform gave no reading.
    baseline_memory: Option<usize>,
    /// Largest RSS observed across [`SystemMemoryProfiler::checkpoint`] calls;
    /// `None` until at least one real reading has been taken.
    peak_memory: Option<usize>,
    /// Insertion-ordered so the delta chain in
    /// [`SystemMemoryProfiler::memory_report`] follows the order the caller
    /// actually took the checkpoints in. With a `HashMap` (the previous type)
    /// every delta depended on hash-seed iteration order and so varied run to
    /// run for the same measurements.
    checkpoints: IndexMap<String, usize>,
}

impl Default for SystemMemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemMemoryProfiler {
    pub fn new() -> Self {
        Self {
            baseline_memory: Self::current_memory_usage(),
            peak_memory: None,
            checkpoints: IndexMap::new(),
        }
    }

    /// Record this process's current RSS under `name`.
    ///
    /// Silently skips the checkpoint when the platform did not return a
    /// reading, rather than recording a fabricated zero.
    pub fn checkpoint(&mut self, name: &str) {
        let Some(current_memory) = Self::current_memory_usage() else {
            tracing::debug!(
                checkpoint = name,
                "no process memory reading available; checkpoint not recorded"
            );
            return;
        };
        self.checkpoints.insert(name.to_string(), current_memory);

        if self.peak_memory.is_none_or(|peak| current_memory > peak) {
            self.peak_memory = Some(current_memory);
        }
    }

    pub fn memory_report(&self) -> MemoryReport {
        let current_memory = Self::current_memory_usage();
        let memory_growth = match (current_memory, self.baseline_memory) {
            (Some(current), Some(baseline)) => Some(current as i64 - baseline as i64),
            _ => None,
        };

        // Deltas are chained through the checkpoints in insertion order, so the
        // sequence is the real one the caller recorded.
        let mut memory_deltas = IndexMap::new();
        let mut prev_memory = self.baseline_memory;
        for (name, memory) in &self.checkpoints {
            if let Some(prev) = prev_memory {
                memory_deltas.insert(name.clone(), *memory as i64 - prev as i64);
            }
            prev_memory = Some(*memory);
        }

        MemoryReport {
            baseline_memory: self.baseline_memory,
            current_memory,
            peak_memory: self.peak_memory,
            memory_growth,
            checkpoints: self.checkpoints.clone(),
            memory_deltas,
        }
    }

    /// Resident set size of THIS process, in bytes, read from `sysinfo`.
    ///
    /// Returns `None` when the platform's process table does not list this
    /// PID (which `sysinfo` supports on every tier-1 target, but not on every
    /// sandbox). It used to return a hardcoded `0`, which made every field of
    /// every [`MemoryReport`] a fabricated zero.
    pub fn current_memory_usage() -> Option<usize> {
        let pid = sysinfo::Pid::from_u32(std::process::id());
        let mut system = sysinfo::System::new();
        system.refresh_processes_specifics(
            sysinfo::ProcessesToUpdate::Some(&[pid]),
            true,
            sysinfo::ProcessRefreshKind::nothing().with_memory(),
        );
        system.process(pid).map(|p| p.memory() as usize)
    }
}

/// Memory profiling report
#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryReport {
    /// Process RSS in bytes when the profiler was created.
    pub baseline_memory: Option<usize>,
    /// Process RSS in bytes when the report was taken.
    pub current_memory: Option<usize>,
    /// Largest RSS seen at any checkpoint.
    pub peak_memory: Option<usize>,
    /// `current_memory - baseline_memory`, signed: memory can genuinely fall.
    /// (It used to be a `usize` produced with `saturating_sub`, so every
    /// release of memory was reported as zero growth.)
    pub memory_growth: Option<i64>,
    /// Real RSS reading recorded at each named checkpoint, in the order the
    /// checkpoints were taken.
    pub checkpoints: IndexMap<String, usize>,
    /// Signed byte delta between consecutive checkpoints, in the same order.
    pub memory_deltas: IndexMap<String, i64>,
}

/// Combined performance and memory profiler
#[derive(Debug)]
pub struct SystemProfiler {
    performance_monitor: PerformanceMonitor,
    memory_profiler: SystemMemoryProfiler,
}

impl Default for SystemProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemProfiler {
    pub fn new() -> Self {
        Self {
            performance_monitor: PerformanceMonitor::new(),
            memory_profiler: SystemMemoryProfiler::new(),
        }
    }

    pub fn checkpoint(&mut self, name: &str) {
        self.performance_monitor.checkpoint(name);
        self.memory_profiler.checkpoint(name);
    }

    pub fn end_checkpoint(&mut self, name: &str) -> Option<Duration> {
        self.performance_monitor.end_checkpoint(name)
    }

    pub fn generate_system_report(&self) -> SystemReport {
        let performance_metrics = self.performance_monitor.get_detailed_metrics();
        let memory_report = self.memory_profiler.memory_report();
        let bottleneck_analysis = self.performance_monitor.analyze_bottlenecks(0.8);

        SystemReport {
            performance_metrics,
            memory_report,
            bottleneck_analysis,
            timestamp: chrono::Utc::now(),
        }
    }
}

/// Comprehensive system profiling report
#[derive(Debug, Serialize, Deserialize)]
pub struct SystemReport {
    pub performance_metrics: PerformanceMetrics,
    pub memory_report: MemoryReport,
    pub bottleneck_analysis: BottleneckAnalysis,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod memory_profiler_tests {
    use super::*;

    #[test]
    fn current_memory_usage_reports_a_real_non_zero_rss() {
        // The old `get_current_memory_usage` returned a hardcoded 0.
        let reading = SystemMemoryProfiler::current_memory_usage();
        // On every platform this crate is tested on, `sysinfo` lists our own
        // PID, so a `None` here is a real regression, not an acceptable
        // absence.
        #[cfg(any(target_os = "linux", target_os = "macos", target_os = "windows"))]
        let bytes = reading.expect("sysinfo must list this process on a tier-1 target");
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
        let Some(bytes) = reading
        else {
            return;
        };
        assert!(
            bytes > 0,
            "a live process must have non-zero RSS, got {bytes} (the old placeholder was 0)"
        );
        // A test binary is comfortably over 1 MiB resident; a fabricated
        // constant would not scale with the real process.
        assert!(bytes > 1024 * 1024, "implausibly small RSS: {bytes} bytes");
    }

    #[test]
    fn checkpoints_and_deltas_follow_the_real_recording_order() {
        let mut profiler = SystemMemoryProfiler::new();
        if SystemMemoryProfiler::current_memory_usage().is_none() {
            return; // no readings available on this platform; nothing to assert
        }
        profiler.checkpoint("first");
        // Force a measurable allocation so the second reading is meaningful.
        let ballast: Vec<u8> = vec![7u8; 8 * 1024 * 1024];
        assert_eq!(ballast.len(), 8 * 1024 * 1024);
        profiler.checkpoint("second");

        let report = profiler.memory_report();
        assert_eq!(
            report.checkpoints.keys().collect::<Vec<_>>(),
            vec!["first", "second"],
            "checkpoints must keep recording order"
        );
        assert_eq!(
            report.memory_deltas.keys().collect::<Vec<_>>(),
            vec!["first", "second"],
            "deltas must keep the same order as the checkpoints"
        );
        assert!(
            report.peak_memory.is_some(),
            "a real peak must have been recorded"
        );
        assert!(report.baseline_memory.is_some());
        assert!(report.current_memory.is_some());
    }

    #[test]
    fn memory_growth_is_signed_so_a_release_is_not_reported_as_zero() {
        let profiler = SystemMemoryProfiler::new();
        let report = profiler.memory_report();
        if let Some(growth) = report.memory_growth {
            // Signed type: this compiles and is meaningful only because the
            // field is `i64` now. `usize` + `saturating_sub` reported every
            // memory release as "zero growth".
            let _: i64 = growth;
        }
    }
}
