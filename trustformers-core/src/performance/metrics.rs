//! Performance metrics for latency, throughput, and memory

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Duration;

/// Latency metrics with detailed statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMetrics {
    /// Total number of measurements
    pub count: usize,
    /// Average latency
    pub mean_ms: f64,
    /// Median latency
    pub median_ms: f64,
    /// Standard deviation
    pub std_dev_ms: f64,
    /// Minimum latency
    pub min_ms: f64,
    /// Maximum latency
    pub max_ms: f64,
    /// 50th percentile
    pub p50_ms: f64,
    /// 90th percentile
    pub p90_ms: f64,
    /// 95th percentile
    pub p95_ms: f64,
    /// 99th percentile
    pub p99_ms: f64,
    /// 99.9th percentile
    pub p999_ms: f64,
    /// Time window for these metrics
    pub window_duration: Duration,
}

impl LatencyMetrics {
    /// Calculate metrics from a collection of durations
    pub fn from_durations(durations: &[Duration]) -> Self {
        if durations.is_empty() {
            return Self::default();
        }

        let mut sorted: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let count = sorted.len();
        let mean_ms = sorted.iter().sum::<f64>() / count as f64;

        let variance = sorted.iter().map(|&x| (x - mean_ms).powi(2)).sum::<f64>() / count as f64;
        let std_dev_ms = variance.sqrt();

        let percentile = |p: f64| -> f64 {
            let index = ((count - 1) as f64 * p / 100.0) as usize;
            sorted[index]
        };

        Self {
            count,
            mean_ms,
            median_ms: percentile(50.0),
            std_dev_ms,
            min_ms: sorted[0],
            max_ms: sorted[count - 1],
            p50_ms: percentile(50.0),
            p90_ms: percentile(90.0),
            p95_ms: percentile(95.0),
            p99_ms: percentile(99.0),
            p999_ms: percentile(99.9),
            window_duration: durations.iter().sum(),
        }
    }
}

impl Default for LatencyMetrics {
    fn default() -> Self {
        Self {
            count: 0,
            mean_ms: 0.0,
            median_ms: 0.0,
            std_dev_ms: 0.0,
            min_ms: 0.0,
            max_ms: 0.0,
            p50_ms: 0.0,
            p90_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
            p999_ms: 0.0,
            window_duration: Duration::ZERO,
        }
    }
}

/// Throughput metrics for different measurement types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    /// Tokens processed per second
    pub tokens_per_second: f64,
    /// Batches processed per second
    pub batches_per_second: f64,
    /// Samples processed per second
    pub samples_per_second: f64,
    /// Average batch size
    pub avg_batch_size: f64,
    /// Average sequence length
    pub avg_sequence_length: f64,
    /// Total tokens processed
    pub total_tokens: usize,
    /// Total batches processed
    pub total_batches: usize,
    /// Total time elapsed
    pub total_duration: Duration,
}

impl ThroughputMetrics {
    /// Calculate throughput from processing statistics
    pub fn calculate(
        total_tokens: usize,
        total_batches: usize,
        total_samples: usize,
        duration: Duration,
    ) -> Self {
        let seconds = duration.as_secs_f64();
        let avg_batch_size = if total_batches > 0 {
            total_samples as f64 / total_batches as f64
        } else {
            0.0
        };
        let avg_sequence_length =
            if total_samples > 0 { total_tokens as f64 / total_samples as f64 } else { 0.0 };

        Self {
            tokens_per_second: total_tokens as f64 / seconds,
            batches_per_second: total_batches as f64 / seconds,
            samples_per_second: total_samples as f64 / seconds,
            avg_batch_size,
            avg_sequence_length,
            total_tokens,
            total_batches,
            total_duration: duration,
        }
    }
}

/// Memory usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Current memory usage in bytes
    pub current_bytes: usize,
    /// Peak memory usage in bytes
    pub peak_bytes: usize,
    /// Allocated memory in bytes
    pub allocated_bytes: usize,
    /// Reserved memory in bytes
    pub reserved_bytes: usize,
    /// Number of allocations
    pub num_allocations: usize,
    /// Number of deallocations
    pub num_deallocations: usize,
    /// Memory fragmentation percentage
    pub fragmentation_percent: f64,
}

impl MemoryMetrics {
    /// Create new memory metrics
    pub fn new(current: usize, peak: usize, allocated: usize, reserved: usize) -> Self {
        let fragmentation = if reserved > 0 {
            (1.0 - allocated as f64 / reserved as f64) * 100.0
        } else {
            0.0
        };

        Self {
            current_bytes: current,
            peak_bytes: peak,
            allocated_bytes: allocated,
            reserved_bytes: reserved,
            num_allocations: 0,
            num_deallocations: 0,
            fragmentation_percent: fragmentation,
        }
    }

    /// Get memory usage in MB
    pub fn current_mb(&self) -> f64 {
        self.current_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get peak memory usage in MB
    pub fn peak_mb(&self) -> f64 {
        self.peak_bytes as f64 / (1024.0 * 1024.0)
    }
}

/// Real-time metrics tracker
pub struct MetricsTracker {
    /// Window size for rolling metrics
    window_size: usize,
    /// Latency measurements
    latencies: VecDeque<Duration>,
    /// Token counts per batch
    token_counts: VecDeque<usize>,
    /// Batch sizes
    batch_sizes: VecDeque<usize>,
    /// Memory snapshots
    memory_snapshots: VecDeque<MemoryMetrics>,
    /// Start time
    start_time: std::time::Instant,
}

impl MetricsTracker {
    /// Create new metrics tracker with specified window size
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            latencies: VecDeque::with_capacity(window_size),
            token_counts: VecDeque::with_capacity(window_size),
            batch_sizes: VecDeque::with_capacity(window_size),
            memory_snapshots: VecDeque::with_capacity(window_size),
            start_time: std::time::Instant::now(),
        }
    }

    /// Record a single inference
    pub fn record_inference(
        &mut self,
        latency: Duration,
        batch_size: usize,
        sequence_length: usize,
    ) {
        // Maintain window size
        if self.latencies.len() >= self.window_size {
            self.latencies.pop_front();
            self.token_counts.pop_front();
            self.batch_sizes.pop_front();
        }

        self.latencies.push_back(latency);
        self.token_counts.push_back(batch_size * sequence_length);
        self.batch_sizes.push_back(batch_size);
    }

    /// Record memory snapshot
    pub fn record_memory(&mut self, metrics: MemoryMetrics) {
        if self.memory_snapshots.len() >= self.window_size {
            self.memory_snapshots.pop_front();
        }
        self.memory_snapshots.push_back(metrics);
    }

    /// Get current latency metrics
    pub fn latency_metrics(&self) -> LatencyMetrics {
        let durations: Vec<Duration> = self.latencies.iter().copied().collect();
        LatencyMetrics::from_durations(&durations)
    }

    /// Get current throughput metrics
    pub fn throughput_metrics(&self) -> ThroughputMetrics {
        let total_tokens: usize = self.token_counts.iter().sum();
        let total_batches = self.latencies.len();
        let total_samples: usize = self.batch_sizes.iter().sum();
        let duration = self.start_time.elapsed();

        ThroughputMetrics::calculate(total_tokens, total_batches, total_samples, duration)
    }

    /// Get latest memory metrics
    pub fn memory_metrics(&self) -> Option<&MemoryMetrics> {
        self.memory_snapshots.back()
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.latencies.clear();
        self.token_counts.clear();
        self.batch_sizes.clear();
        self.memory_snapshots.clear();
        self.start_time = std::time::Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_metrics() {
        let durations = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(15),
            Duration::from_millis(25),
            Duration::from_millis(30),
        ];

        let metrics = LatencyMetrics::from_durations(&durations);

        assert_eq!(metrics.count, 5);
        assert_eq!(metrics.mean_ms, 20.0);
        assert_eq!(metrics.min_ms, 10.0);
        assert_eq!(metrics.max_ms, 30.0);
        assert!(metrics.std_dev_ms > 0.0);
    }

    #[test]
    fn test_throughput_metrics() {
        let metrics = ThroughputMetrics::calculate(
            10000, // tokens
            100,   // batches
            400,   // samples (4 per batch)
            Duration::from_secs(10),
        );

        assert_eq!(metrics.tokens_per_second, 1000.0);
        assert_eq!(metrics.batches_per_second, 10.0);
        assert_eq!(metrics.samples_per_second, 40.0);
        assert_eq!(metrics.avg_batch_size, 4.0);
        assert_eq!(metrics.avg_sequence_length, 25.0);
    }

    #[test]
    fn test_metrics_tracker() {
        let mut tracker = MetricsTracker::new(10);

        // Record some inferences
        for i in 0..5 {
            tracker.record_inference(Duration::from_millis(10 + i * 5), 4, 128);
        }

        let latency = tracker.latency_metrics();
        assert_eq!(latency.count, 5);

        let throughput = tracker.throughput_metrics();
        assert_eq!(throughput.total_batches, 5);
        assert_eq!(throughput.total_tokens, 5 * 4 * 128);
    }
}
