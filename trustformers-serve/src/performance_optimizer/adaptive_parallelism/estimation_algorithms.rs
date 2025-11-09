//! Estimation Algorithms for Optimal Parallelism
//!
//! This module provides various algorithms for estimating optimal parallelism levels
//! including linear regression, resource-based estimation, historical averaging,
//! and CPU affinity-based approaches. Each algorithm provides different perspectives
//! on optimal parallelism based on different heuristics and data analysis methods.

use anyhow::Result;
use std::collections::HashMap;

use crate::performance_optimizer::types::*;

// Note: EstimationAlgorithm trait is defined in types.rs and imported above
// Re-export the trait for use by parent module
pub use crate::performance_optimizer::types::EstimationAlgorithm;

// =============================================================================
// LINEAR REGRESSION ESTIMATOR
// =============================================================================

/// Linear regression based estimation algorithm
pub struct LinearRegressionEstimator {
    name: String,
}

impl LinearRegressionEstimator {
    pub fn new() -> Self {
        Self {
            name: "linear_regression".to_string(),
        }
    }
}

impl EstimationAlgorithm for LinearRegressionEstimator {
    fn estimate_optimal_parallelism(
        &self,
        historical_data: &[PerformanceDataPoint],
        _current_characteristics: &TestCharacteristics,
        system_state: &SystemState,
    ) -> Result<ParallelismEstimate> {
        if historical_data.len() < 3 {
            // Not enough data for regression, use simple heuristic
            let optimal = (system_state.available_cores as f32 * 0.8) as usize;
            return Ok(ParallelismEstimate {
                optimal_parallelism: optimal.max(1),
                confidence: 0.3,
                expected_improvement: 0.1,
                method: self.name.clone(),
                metadata: HashMap::new(),
            });
        }

        // Simple linear regression: throughput = a * parallelism + b
        let n = historical_data.len() as f64;
        let sum_x: f64 = historical_data.iter().map(|p| p.parallelism as f64).sum();
        let sum_y: f64 = historical_data.iter().map(|p| p.throughput).sum();
        let sum_xy: f64 = historical_data.iter().map(|p| p.parallelism as f64 * p.throughput).sum();
        let sum_x2: f64 = historical_data.iter().map(|p| (p.parallelism as f64).powi(2)).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));
        let intercept = (sum_y - slope * sum_x) / n;

        // Find optimal parallelism (where derivative becomes negative or reaches system limit)
        let max_cores = system_state.available_cores;
        let mut optimal_parallelism = 1;
        let mut max_throughput = intercept + slope;

        for cores in 2..=max_cores {
            let predicted_throughput = intercept + slope * cores as f64;
            if predicted_throughput > max_throughput {
                max_throughput = predicted_throughput;
                optimal_parallelism = cores;
            } else {
                break; // Diminishing returns
            }
        }

        let confidence = if historical_data.len() >= 10 { 0.8 } else { 0.5 };
        let current_throughput = historical_data.last().map(|p| p.throughput).unwrap_or(1.0);
        let expected_improvement =
            ((max_throughput - current_throughput) / current_throughput).max(0.0) as f32;

        Ok(ParallelismEstimate {
            optimal_parallelism: optimal_parallelism.max(1),
            confidence,
            expected_improvement: expected_improvement.min(1.0),
            method: self.name.clone(),
            metadata: HashMap::new(),
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn confidence(&self, data_points: usize) -> f32 {
        (data_points as f32 / 20.0).min(0.9)
    }
}

// =============================================================================
// RESOURCE-BASED ESTIMATOR
// =============================================================================

/// Resource-based estimation algorithm
pub struct ResourceBasedEstimator {
    name: String,
}

impl ResourceBasedEstimator {
    pub fn new() -> Self {
        Self {
            name: "resource_based".to_string(),
        }
    }
}

impl EstimationAlgorithm for ResourceBasedEstimator {
    fn estimate_optimal_parallelism(
        &self,
        _historical_data: &[PerformanceDataPoint],
        current_characteristics: &TestCharacteristics,
        system_state: &SystemState,
    ) -> Result<ParallelismEstimate> {
        let intensity = &current_characteristics.resource_intensity;

        // Calculate resource-aware parallelism
        let cpu_limit =
            (system_state.available_cores as f32 / intensity.cpu_intensity.max(0.1)) as usize;
        let memory_limit = if intensity.memory_intensity > 0.1 {
            ((system_state.available_memory_mb as f32 / 1024.0) / intensity.memory_intensity)
                as usize
        } else {
            system_state.available_cores * 4
        };

        let optimal_parallelism = cpu_limit.clamp(1, memory_limit);

        // Higher confidence for resource-intensive workloads
        let confidence = (intensity.cpu_intensity + intensity.memory_intensity) / 2.0;

        Ok(ParallelismEstimate {
            optimal_parallelism,
            confidence: confidence.clamp(0.3, 0.9),
            expected_improvement: 0.15,
            method: self.name.clone(),
            metadata: HashMap::new(),
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn confidence(&self, _data_points: usize) -> f32 {
        0.7 // Resource-based estimation is generally reliable
    }
}

// =============================================================================
// HISTORICAL AVERAGE ESTIMATOR
// =============================================================================

/// Historical average estimation algorithm
pub struct HistoricalAverageEstimator {
    name: String,
}

impl HistoricalAverageEstimator {
    pub fn new() -> Self {
        Self {
            name: "historical_average".to_string(),
        }
    }
}

impl EstimationAlgorithm for HistoricalAverageEstimator {
    fn estimate_optimal_parallelism(
        &self,
        historical_data: &[PerformanceDataPoint],
        _current_characteristics: &TestCharacteristics,
        _system_state: &SystemState,
    ) -> Result<ParallelismEstimate> {
        if historical_data.is_empty() {
            return Ok(ParallelismEstimate {
                optimal_parallelism: num_cpus::get(),
                confidence: 0.2,
                expected_improvement: 0.05,
                method: self.name.clone(),
                metadata: HashMap::new(),
            });
        }

        // Find the parallelism level with best average performance
        let mut parallelism_performance: HashMap<usize, Vec<f64>> = HashMap::new();

        for point in historical_data {
            parallelism_performance
                .entry(point.parallelism)
                .or_default()
                .push(point.throughput);
        }

        let mut best_parallelism = 1;
        let mut best_avg_throughput = 0.0;

        for (parallelism, throughputs) in parallelism_performance {
            let avg_throughput = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
            if avg_throughput > best_avg_throughput {
                best_avg_throughput = avg_throughput;
                best_parallelism = parallelism;
            }
        }

        let confidence = (historical_data.len() as f32 / 50.0).min(0.8);

        Ok(ParallelismEstimate {
            optimal_parallelism: best_parallelism,
            confidence,
            expected_improvement: 0.1,
            method: self.name.clone(),
            metadata: HashMap::new(),
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn confidence(&self, data_points: usize) -> f32 {
        (data_points as f32 / 30.0).min(0.8)
    }
}

// =============================================================================
// CPU AFFINITY ESTIMATOR
// =============================================================================

/// CPU affinity-based estimation algorithm
pub struct CpuAffinityEstimator {
    name: String,
}

impl CpuAffinityEstimator {
    pub fn new() -> Self {
        Self {
            name: "cpu_affinity".to_string(),
        }
    }
}

impl EstimationAlgorithm for CpuAffinityEstimator {
    fn estimate_optimal_parallelism(
        &self,
        _historical_data: &[PerformanceDataPoint],
        current_characteristics: &TestCharacteristics,
        system_state: &SystemState,
    ) -> Result<ParallelismEstimate> {
        let intensity = &current_characteristics.resource_intensity;

        // Estimate based on CPU affinity and workload characteristics
        let base_parallelism = if intensity.cpu_intensity > 0.8 {
            // CPU-intensive workloads: limit to physical cores
            system_state.available_cores
        } else if intensity.io_intensity > 0.6 {
            // I/O-intensive workloads: can oversubscribe
            system_state.available_cores * 2
        } else {
            // Balanced workloads
            (system_state.available_cores as f32 * 1.5) as usize
        };

        // Adjust for system load
        let load_factor = (1.0 - system_state.load_average).max(0.1);
        let optimal_parallelism = (base_parallelism as f32 * load_factor) as usize;

        Ok(ParallelismEstimate {
            optimal_parallelism: optimal_parallelism.max(1),
            confidence: 0.6,
            expected_improvement: 0.2,
            method: self.name.clone(),
            metadata: HashMap::new(),
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn confidence(&self, _data_points: usize) -> f32 {
        0.6 // Moderate confidence as it's heuristic-based
    }
}
