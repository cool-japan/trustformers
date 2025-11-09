use crate::error::{Result, TrustformersError};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Configuration for adaptive batch sizing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveBatchConfig {
    /// Minimum batch size to consider
    pub min_batch_size: usize,
    /// Maximum batch size to test
    pub max_batch_size: usize,
    /// Number of performance samples to collect per batch size
    pub samples_per_size: usize,
    /// Warmup iterations before collecting metrics
    pub warmup_iterations: usize,
    /// Target latency percentile (e.g., 95.0 for 95th percentile)
    pub target_latency_percentile: f64,
    /// Target latency in milliseconds
    pub target_latency_ms: f64,
    /// Throughput weight in optimization (0.0 to 1.0)
    pub throughput_weight: f64,
    /// Latency weight in optimization (0.0 to 1.0)
    pub latency_weight: f64,
    /// Memory weight in optimization (0.0 to 1.0)
    pub memory_weight: f64,
    /// Re-evaluation interval in seconds
    pub reevaluation_interval_secs: u64,
}

impl Default for AdaptiveBatchConfig {
    fn default() -> Self {
        Self {
            min_batch_size: 1,
            max_batch_size: 64,
            samples_per_size: 10,
            warmup_iterations: 3,
            target_latency_percentile: 95.0,
            target_latency_ms: 100.0,
            throughput_weight: 0.4,
            latency_weight: 0.4,
            memory_weight: 0.2,
            reevaluation_interval_secs: 300, // 5 minutes
        }
    }
}

/// Performance sample for a specific batch size
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSample {
    pub batch_size: usize,
    pub latency_ms: f64,
    pub throughput_rps: f64,
    pub memory_usage_mb: f64,
    pub gpu_memory_mb: f64,
    pub cpu_utilization: f32,
    pub gpu_utilization: f32,
    pub timestamp: std::time::SystemTime,
}

/// Statistics for a batch size over multiple samples
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchSizeStats {
    pub batch_size: usize,
    pub sample_count: usize,
    pub avg_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub avg_throughput_rps: f64,
    pub max_throughput_rps: f64,
    pub avg_memory_mb: f64,
    pub max_memory_mb: f64,
    pub avg_cpu_utilization: f32,
    pub avg_gpu_utilization: f32,
    pub score: f64, // Combined optimization score
}

/// Adaptive batch sizing optimizer
#[derive(Debug)]
pub struct AdaptiveBatchOptimizer {
    config: AdaptiveBatchConfig,
    performance_data: Arc<RwLock<HashMap<usize, VecDeque<PerformanceSample>>>>,
    batch_stats: Arc<RwLock<HashMap<usize, BatchSizeStats>>>,
    optimal_batch_size: Arc<RwLock<Option<usize>>>,
    last_evaluation: Arc<RwLock<Instant>>,
    current_test_size: Arc<RwLock<Option<usize>>>,
    test_iteration: Arc<RwLock<usize>>,
}

/// Alias for backward compatibility
pub type AdaptiveBatchManager = AdaptiveBatchOptimizer;

impl AdaptiveBatchOptimizer {
    /// Create a new adaptive batch optimizer
    pub fn new(config: AdaptiveBatchConfig) -> Self {
        Self {
            config,
            performance_data: Arc::new(RwLock::new(HashMap::new())),
            batch_stats: Arc::new(RwLock::new(HashMap::new())),
            optimal_batch_size: Arc::new(RwLock::new(None)),
            last_evaluation: Arc::new(RwLock::new(Instant::now())),
            current_test_size: Arc::new(RwLock::new(None)),
            test_iteration: Arc::new(RwLock::new(0)),
        }
    }

    /// Get the current optimal batch size
    pub fn get_optimal_batch_size(&self) -> Option<usize> {
        *self.optimal_batch_size.read().unwrap()
    }

    /// Record a performance sample for a given batch size
    pub fn record_sample(&self, sample: PerformanceSample) -> Result<()> {
        let batch_size = sample.batch_size;

        // Add sample to performance data
        {
            let mut data = self.performance_data.write().unwrap();
            let samples = data.entry(batch_size).or_default();
            samples.push_back(sample);

            // Keep only recent samples
            while samples.len() > self.config.samples_per_size * 2 {
                samples.pop_front();
            }
        }

        // Update statistics for this batch size
        self.update_batch_stats(batch_size)?;

        // Check if we should re-evaluate optimal size
        let should_reevaluate = {
            let last_eval = self.last_evaluation.read().unwrap();
            last_eval.elapsed().as_secs() >= self.config.reevaluation_interval_secs
        };

        if should_reevaluate {
            self.evaluate_optimal_batch_size()?;
        }

        Ok(())
    }

    /// Get the next batch size to test (for exploration)
    pub fn get_next_test_size(&self) -> Option<usize> {
        let mut current_test = self.current_test_size.write().unwrap();
        let mut iteration = self.test_iteration.write().unwrap();

        match *current_test {
            None => {
                // Start testing from minimum size
                *current_test = Some(self.config.min_batch_size);
                *iteration = 0;
                Some(self.config.min_batch_size)
            },
            Some(size) => {
                *iteration += 1;

                // Check if we've collected enough samples for current size
                let enough_samples = {
                    let data = self.performance_data.read().unwrap();
                    data.get(&size)
                        .map(|samples| samples.len() >= self.config.samples_per_size)
                        .unwrap_or(false)
                };

                if enough_samples {
                    // Move to next batch size
                    let next_size = self.get_next_size_to_test(size);
                    *current_test = next_size;
                    *iteration = 0;
                    next_size
                } else {
                    // Continue testing current size
                    Some(size)
                }
            },
        }
    }

    /// Get the next size to test in sequence
    fn get_next_size_to_test(&self, current_size: usize) -> Option<usize> {
        // Exponential progression for testing
        let progression = [1, 2, 4, 8, 16, 32, 64, 128];

        if let Some(pos) = progression.iter().position(|&x| x == current_size) {
            if pos + 1 < progression.len() && progression[pos + 1] <= self.config.max_batch_size {
                Some(progression[pos + 1])
            } else {
                None // Testing complete
            }
        } else {
            // Custom progression if current size not in standard progression
            let next = current_size * 2;
            if next <= self.config.max_batch_size {
                Some(next)
            } else {
                None
            }
        }
    }

    /// Update statistics for a specific batch size
    fn update_batch_stats(&self, batch_size: usize) -> Result<()> {
        let samples = {
            let data = self.performance_data.read().unwrap();
            data.get(&batch_size).cloned().unwrap_or_default()
        };

        if samples.len() < 3 {
            return Ok(()); // Need at least 3 samples for meaningful stats
        }

        let sample_vec: Vec<_> = samples.iter().collect();

        // Calculate statistics
        let avg_latency =
            sample_vec.iter().map(|s| s.latency_ms).sum::<f64>() / sample_vec.len() as f64;
        let avg_throughput =
            sample_vec.iter().map(|s| s.throughput_rps).sum::<f64>() / sample_vec.len() as f64;
        let max_throughput = sample_vec.iter().map(|s| s.throughput_rps).fold(0.0, f64::max);
        let avg_memory =
            sample_vec.iter().map(|s| s.memory_usage_mb).sum::<f64>() / sample_vec.len() as f64;
        let max_memory = sample_vec.iter().map(|s| s.memory_usage_mb).fold(0.0, f64::max);
        let avg_cpu =
            sample_vec.iter().map(|s| s.cpu_utilization).sum::<f32>() / sample_vec.len() as f32;
        let avg_gpu =
            sample_vec.iter().map(|s| s.gpu_utilization).sum::<f32>() / sample_vec.len() as f32;

        // Calculate percentiles
        let mut latencies: Vec<f64> = sample_vec.iter().map(|s| s.latency_ms).collect();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p95_index = ((latencies.len() as f64) * 0.95) as usize;
        let p99_index = ((latencies.len() as f64) * 0.99) as usize;
        let p95_latency = latencies.get(p95_index).copied().unwrap_or(avg_latency);
        let p99_latency = latencies.get(p99_index).copied().unwrap_or(avg_latency);

        // Calculate optimization score
        let score =
            self.calculate_optimization_score(avg_latency, p95_latency, avg_throughput, avg_memory);

        let stats = BatchSizeStats {
            batch_size,
            sample_count: sample_vec.len(),
            avg_latency_ms: avg_latency,
            p95_latency_ms: p95_latency,
            p99_latency_ms: p99_latency,
            avg_throughput_rps: avg_throughput,
            max_throughput_rps: max_throughput,
            avg_memory_mb: avg_memory,
            max_memory_mb: max_memory,
            avg_cpu_utilization: avg_cpu,
            avg_gpu_utilization: avg_gpu,
            score,
        };

        // Store stats
        {
            let mut batch_stats = self.batch_stats.write().unwrap();
            batch_stats.insert(batch_size, stats);
        }

        Ok(())
    }

    /// Calculate optimization score for a configuration
    fn calculate_optimization_score(
        &self,
        avg_latency: f64,
        p95_latency: f64,
        avg_throughput: f64,
        avg_memory: f64,
    ) -> f64 {
        // Normalize metrics to 0-1 scale
        let latency_score = if p95_latency <= self.config.target_latency_ms {
            1.0 // Perfect score for meeting target
        } else {
            (self.config.target_latency_ms / p95_latency).max(0.1) // Penalty for exceeding target
        };

        // Throughput score (higher is better)
        let throughput_score = (avg_throughput / 100.0).min(1.0); // Normalize to 100 RPS baseline

        // Memory score (lower is better)
        let memory_score = (1000.0 / (avg_memory + 100.0)).min(1.0); // Normalize with 1GB baseline

        // Weighted combination

        (self.config.latency_weight * latency_score)
            + (self.config.throughput_weight * throughput_score)
            + (self.config.memory_weight * memory_score)
    }

    /// Evaluate and update the optimal batch size
    fn evaluate_optimal_batch_size(&self) -> Result<()> {
        let stats = self.batch_stats.read().unwrap();

        if stats.is_empty() {
            return Ok(());
        }

        // Find batch size with highest score
        let optimal = stats
            .values()
            .filter(|s| s.sample_count >= self.config.samples_per_size)
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .map(|s| s.batch_size);

        if let Some(optimal_size) = optimal {
            let mut current_optimal = self.optimal_batch_size.write().unwrap();
            *current_optimal = Some(optimal_size);

            tracing::info!(
                "Updated optimal batch size to {}: score {:.3}",
                optimal_size,
                stats.get(&optimal_size).unwrap().score
            );
        }

        // Update last evaluation time
        {
            let mut last_eval = self.last_evaluation.write().unwrap();
            *last_eval = Instant::now();
        }

        Ok(())
    }

    /// Get comprehensive performance report
    pub fn get_performance_report(&self) -> PerformanceReport {
        let stats = self.batch_stats.read().unwrap();
        let optimal = *self.optimal_batch_size.read().unwrap();

        let mut batch_performances: Vec<_> = stats.values().cloned().collect();
        batch_performances.sort_by_key(|s| s.batch_size);

        PerformanceReport {
            optimal_batch_size: optimal,
            batch_performances,
            total_evaluations: stats.len(),
            last_evaluation: *self.last_evaluation.read().unwrap(),
        }
    }

    /// Export performance data for analysis
    pub fn export_data(&self) -> Result<String> {
        let stats = self.batch_stats.read().unwrap();
        let data = stats.values().collect::<Vec<_>>();
        serde_json::to_string_pretty(&data)
            .map_err(|e| TrustformersError::runtime_error(format!("Failed to export data: {}", e)))
    }

    /// Import performance data from previous runs
    pub fn import_data(&self, data: &str) -> Result<()> {
        let imported_stats: Vec<BatchSizeStats> = serde_json::from_str(data).map_err(|e| {
            TrustformersError::runtime_error(format!("Failed to import data: {}", e))
        })?;

        {
            let mut stats = self.batch_stats.write().unwrap();
            for stat in imported_stats {
                stats.insert(stat.batch_size, stat);
            }
        }

        // Re-evaluate optimal batch size with new data
        self.evaluate_optimal_batch_size()?;

        Ok(())
    }
}

/// Performance report containing optimization results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub optimal_batch_size: Option<usize>,
    pub batch_performances: Vec<BatchSizeStats>,
    pub total_evaluations: usize,
    #[serde(skip, default = "Instant::now")]
    pub last_evaluation: Instant,
}

impl PerformanceReport {
    /// Get the best performing batch sizes (top N)
    pub fn get_top_performers(&self, n: usize) -> Vec<&BatchSizeStats> {
        let mut sorted = self.batch_performances.iter().collect::<Vec<_>>();
        sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        sorted.into_iter().take(n).collect()
    }

    /// Get performance comparison between two batch sizes
    pub fn compare_batch_sizes(&self, size1: usize, size2: usize) -> Option<BatchComparison> {
        let stats1 = self.batch_performances.iter().find(|s| s.batch_size == size1)?;
        let stats2 = self.batch_performances.iter().find(|s| s.batch_size == size2)?;

        Some(BatchComparison {
            size1,
            size2,
            latency_improvement: stats1.avg_latency_ms / stats2.avg_latency_ms,
            throughput_improvement: stats2.avg_throughput_rps / stats1.avg_throughput_rps,
            memory_difference: stats2.avg_memory_mb - stats1.avg_memory_mb,
            score_difference: stats2.score - stats1.score,
        })
    }
}

/// Comparison between two batch sizes
#[derive(Debug, Clone)]
pub struct BatchComparison {
    pub size1: usize,
    pub size2: usize,
    pub latency_improvement: f64,    // >1.0 means size1 is faster
    pub throughput_improvement: f64, // >1.0 means size2 has higher throughput
    pub memory_difference: f64,      // Positive means size2 uses more memory
    pub score_difference: f64,       // Positive means size2 has better score
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    #[test]
    fn test_adaptive_batch_optimizer() {
        let config = AdaptiveBatchConfig::default();
        let optimizer = AdaptiveBatchOptimizer::new(config);

        // Record some sample data
        let sample = PerformanceSample {
            batch_size: 8,
            latency_ms: 50.0,
            throughput_rps: 160.0,
            memory_usage_mb: 512.0,
            gpu_memory_mb: 1024.0,
            cpu_utilization: 0.7,
            gpu_utilization: 0.8,
            timestamp: SystemTime::now(),
        };

        optimizer.record_sample(sample).unwrap();

        // Test next test size progression
        assert_eq!(optimizer.get_next_test_size(), Some(1));
        assert_eq!(optimizer.get_next_test_size(), Some(1)); // Should continue with 1
    }

    #[test]
    fn test_optimization_score_calculation() {
        let config = AdaptiveBatchConfig::default();
        let optimizer = AdaptiveBatchOptimizer::new(config);

        // Test perfect score (meets latency target)
        let score1 = optimizer.calculate_optimization_score(80.0, 95.0, 50.0, 500.0);

        // Test penalty score (exceeds latency target)
        let score2 = optimizer.calculate_optimization_score(120.0, 150.0, 50.0, 500.0);

        assert!(
            score1 > score2,
            "Score should be higher when meeting latency target"
        );
    }

    #[test]
    fn test_performance_report() {
        let config = AdaptiveBatchConfig::default();
        let optimizer = AdaptiveBatchOptimizer::new(config);

        // Add some test data
        for batch_size in [2, 4, 8, 16] {
            for _ in 0..5 {
                let sample = PerformanceSample {
                    batch_size,
                    latency_ms: 50.0 + batch_size as f64 * 5.0,
                    throughput_rps: batch_size as f64 * 10.0,
                    memory_usage_mb: batch_size as f64 * 64.0,
                    gpu_memory_mb: batch_size as f64 * 128.0,
                    cpu_utilization: 0.6,
                    gpu_utilization: 0.7,
                    timestamp: SystemTime::now(),
                };
                optimizer.record_sample(sample).unwrap();
            }
        }

        let report = optimizer.get_performance_report();
        assert!(!report.batch_performances.is_empty());

        let top_performers = report.get_top_performers(2);
        assert!(top_performers.len() <= 2);
    }
}
