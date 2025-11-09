//! Performance monitoring and profiling utilities

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
    baseline_memory: usize,
    peak_memory: usize,
    checkpoints: HashMap<String, usize>,
}

impl SystemMemoryProfiler {
    pub fn new() -> Self {
        Self {
            baseline_memory: Self::get_current_memory_usage(),
            peak_memory: 0,
            checkpoints: HashMap::new(),
        }
    }

    pub fn checkpoint(&mut self, name: &str) {
        let current_memory = Self::get_current_memory_usage();
        self.checkpoints.insert(name.to_string(), current_memory);

        if current_memory > self.peak_memory {
            self.peak_memory = current_memory;
        }
    }

    pub fn memory_report(&self) -> MemoryReport {
        let current_memory = Self::get_current_memory_usage();
        let memory_growth = current_memory.saturating_sub(self.baseline_memory);

        let mut memory_deltas = HashMap::new();
        let mut prev_memory = self.baseline_memory;

        for (name, memory) in &self.checkpoints {
            let delta = memory.saturating_sub(prev_memory) as i64;
            memory_deltas.insert(name.clone(), delta);
            prev_memory = *memory;
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

    fn get_current_memory_usage() -> usize {
        // Simplified memory usage - in practice this would use platform-specific APIs
        // This is a placeholder implementation
        0
    }
}

/// Memory profiling report
#[derive(Debug, Serialize, Deserialize)]
pub struct MemoryReport {
    pub baseline_memory: usize,
    pub current_memory: usize,
    pub peak_memory: usize,
    pub memory_growth: usize,
    pub checkpoints: HashMap<String, usize>,
    pub memory_deltas: HashMap<String, i64>,
}

/// Combined performance and memory profiler
#[derive(Debug)]
pub struct SystemProfiler {
    performance_monitor: PerformanceMonitor,
    memory_profiler: SystemMemoryProfiler,
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
