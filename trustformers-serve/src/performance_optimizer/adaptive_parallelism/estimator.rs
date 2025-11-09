//! Optimal Parallelism Estimator Implementation
//!
//! This module provides the OptimalParallelismEstimator which coordinates multiple
//! estimation algorithms to determine the optimal parallelism level. It includes
//! ensemble estimation, historical data management, accuracy tracking, and model
//! training capabilities.

use anyhow::Result;
use chrono::Utc;
use parking_lot::Mutex;
use std::{collections::HashMap, sync::Arc, time::Duration};

// EstimationAlgorithm trait is now imported from types (not from estimation_algorithms)
// CacheHierarchy is also in types::*
use crate::performance_optimizer::types::*;

// =============================================================================
// OPTIMAL PARALLELISM ESTIMATOR IMPLEMENTATION
// =============================================================================

impl OptimalParallelismEstimator {
    /// Create a new optimal parallelism estimator
    pub async fn new() -> Result<Self> {
        let mut algorithms: Vec<Box<dyn EstimationAlgorithm + Send + Sync>> = Vec::new();

        // Add default estimation algorithms
        algorithms.push(Box::new(
            super::estimation_algorithms::LinearRegressionEstimator::new(),
        ));
        algorithms.push(Box::new(
            super::estimation_algorithms::ResourceBasedEstimator::new(),
        ));
        algorithms.push(Box::new(
            super::estimation_algorithms::HistoricalAverageEstimator::new(),
        ));
        algorithms.push(Box::new(
            super::estimation_algorithms::CpuAffinityEstimator::new(),
        ));

        Ok(Self {
            performance_model: Arc::new(Mutex::new(PerformanceModel {
                model_type: PerformanceModelType::Ensemble {
                    models: vec![
                        PerformanceModelType::LinearRegression,
                        PerformanceModelType::PolynomialRegression { degree: 2 },
                    ],
                },
                parameters: HashMap::new(),
                accuracy: 0.8,
                last_updated: Utc::now(),
                training_data_size: 0,
                validation_results: ModelValidationResults {
                    r_squared: 0.8,
                    mean_absolute_error: 0.1,
                    root_mean_squared_error: 0.15,
                    cross_validation_scores: vec![0.8, 0.82, 0.78, 0.85, 0.79],
                    validated_at: Utc::now(),
                },
            })),
            historical_data: Arc::new(Mutex::new(Vec::new())),
            resource_model: Arc::new(Mutex::new(Self::create_default_resource_model())),
            algorithms: Arc::new(Mutex::new(algorithms)),
            accuracy_tracker: Arc::new(Mutex::new(EstimationAccuracyTracker::default())),
        })
    }

    /// Estimate optimal parallelism using ensemble of algorithms
    pub async fn estimate_optimal_parallelism(
        &self,
        historical_data: &[PerformanceDataPoint],
        current_characteristics: &TestCharacteristics,
        system_state: &SystemState,
    ) -> Result<ParallelismEstimate> {
        let algorithms = self.algorithms.lock();
        let mut estimates = Vec::new();

        // Get estimates from all algorithms
        for algorithm in algorithms.iter() {
            match algorithm.estimate_optimal_parallelism(
                historical_data,
                current_characteristics,
                system_state,
            ) {
                Ok(estimate) => estimates.push(estimate),
                Err(e) => {
                    log::warn!("Algorithm {} failed: {}", algorithm.name(), e);
                },
            }
        }

        if estimates.is_empty() {
            return Err(anyhow::anyhow!("No algorithms produced valid estimates"));
        }

        // Combine estimates using weighted average
        let combined_estimate = self.combine_estimates(&estimates)?;

        Ok(combined_estimate)
    }

    /// Combine multiple estimates into a single estimate
    fn combine_estimates(&self, estimates: &[ParallelismEstimate]) -> Result<ParallelismEstimate> {
        if estimates.is_empty() {
            return Err(anyhow::anyhow!("Cannot combine empty estimates"));
        }

        // Weight estimates by confidence
        let total_weight: f32 = estimates.iter().map(|e| e.confidence).sum();
        let weighted_parallelism: f32 = estimates
            .iter()
            .map(|e| e.optimal_parallelism as f32 * e.confidence)
            .sum::<f32>()
            / total_weight;

        let weighted_improvement: f32 =
            estimates.iter().map(|e| e.expected_improvement * e.confidence).sum::<f32>()
                / total_weight;

        let combined_confidence = (total_weight / estimates.len() as f32).min(1.0);

        let mut metadata = HashMap::new();
        metadata.insert("algorithms_used".to_string(), estimates.len().to_string());
        metadata.insert(
            "combination_method".to_string(),
            "weighted_average".to_string(),
        );

        Ok(ParallelismEstimate {
            optimal_parallelism: weighted_parallelism.round() as usize,
            confidence: combined_confidence,
            expected_improvement: weighted_improvement,
            method: "ensemble_weighted_average".to_string(),
            metadata,
        })
    }

    /// Add performance data point for training
    pub async fn add_performance_data(&self, data_point: PerformanceDataPoint) -> Result<()> {
        self.historical_data.lock().push(data_point);

        // Limit historical data size
        let mut data = self.historical_data.lock();
        if data.len() > 1000 {
            data.drain(0..100); // Remove oldest 100 points
        }

        Ok(())
    }

    /// Record estimation for accuracy tracking
    pub async fn record_estimation(&self, estimate: &ParallelismEstimate) -> Result<()> {
        let record = EstimationRecord {
            timestamp: Utc::now(),
            estimated_parallelism: estimate.optimal_parallelism,
            actual_optimal_parallelism: None, // Will be filled later
            algorithm: estimate.method.clone(),
            confidence: estimate.confidence,
            error: None,
        };

        self.accuracy_tracker.lock().estimation_history.push(record);
        self.accuracy_tracker.lock().total_estimations += 1;

        Ok(())
    }

    /// Update estimation accuracy with actual results
    pub async fn update_estimation_accuracy(
        &self,
        estimated_parallelism: usize,
        actual_optimal: usize,
    ) -> Result<()> {
        let mut tracker = self.accuracy_tracker.lock();

        // Find corresponding estimation record
        if let Some(record) = tracker.estimation_history.iter_mut().rev().find(|r| {
            r.estimated_parallelism == estimated_parallelism
                && r.actual_optimal_parallelism.is_none()
        }) {
            record.actual_optimal_parallelism = Some(actual_optimal);
            let error = (estimated_parallelism as f32 - actual_optimal as f32).abs()
                / actual_optimal.max(1) as f32;
            record.error = Some(error);

            // Update accuracy statistics
            if error < 0.1 {
                // Within 10% is considered correct
                tracker.correct_estimations += 1;
            }

            // Update average error
            let errors: Vec<f32> =
                tracker.estimation_history.iter().filter_map(|r| r.error).collect();

            if !errors.is_empty() {
                tracker.average_error = errors.iter().sum::<f32>() / errors.len() as f32;
            }
        }

        Ok(())
    }

    /// Get accuracy statistics
    pub fn get_accuracy_stats(&self) -> EstimationAccuracyTracker {
        self.accuracy_tracker.lock().clone()
    }

    /// Train performance model with historical data
    pub async fn train_performance_model(&self) -> Result<()> {
        let historical_data = self.historical_data.lock().clone();

        if historical_data.len() < 10 {
            return Ok(()); // Not enough data to train
        }

        // Convert historical data to training format
        let training_data = self.convert_to_training_data(&historical_data)?;

        // Update model with new training data
        let mut model = self.performance_model.lock();
        model.training_data_size = training_data.len();
        model.last_updated = Utc::now();

        // Simple accuracy calculation based on data consistency
        model.accuracy = self.calculate_model_accuracy(&historical_data);

        log::info!(
            "Trained performance model with {} data points, accuracy: {:.2}",
            training_data.len(),
            model.accuracy
        );

        Ok(())
    }

    /// Convert performance data to training format
    fn convert_to_training_data(
        &self,
        data: &[PerformanceDataPoint],
    ) -> Result<Vec<TrainingExample>> {
        let mut training_examples = Vec::new();

        for point in data {
            let features = vec![
                point.parallelism as f64,
                point.cpu_utilization as f64,
                point.memory_utilization as f64,
                point.resource_efficiency as f64,
                point.test_characteristics.average_duration.as_millis() as f64,
            ];

            let example = TrainingExample {
                features,
                target: point.throughput,
                weight: 1.0,
                timestamp: point.timestamp,
                metadata: HashMap::new(),
            };

            training_examples.push(example);
        }

        Ok(training_examples)
    }

    /// Calculate model accuracy from historical consistency
    fn calculate_model_accuracy(&self, data: &[PerformanceDataPoint]) -> f32 {
        if data.len() < 2 {
            return 0.5;
        }

        // Simple measure: how consistent throughput is for similar parallelism levels
        let mut consistency_scores = Vec::new();

        for i in 1..data.len() {
            let current = &data[i];
            let previous = &data[i - 1];

            if (current.parallelism as i32 - previous.parallelism as i32).abs() <= 1 {
                let expected_change =
                    (current.parallelism as f64 - previous.parallelism as f64) * 10.0; // Expected 10x throughput per core
                let actual_change = current.throughput - previous.throughput;
                let error = (expected_change - actual_change).abs() / previous.throughput.max(1.0);
                consistency_scores.push((1.0 / (1.0 + error)) as f32);
            }
        }

        if consistency_scores.is_empty() {
            0.5
        } else {
            consistency_scores.iter().sum::<f32>() / consistency_scores.len() as f32
        }
    }

    /// Create default system resource model
    fn create_default_resource_model() -> SystemResourceModel {
        SystemResourceModel {
            cpu_model: CpuModel {
                core_count: num_cpus::get(),
                thread_count: num_cpus::get() * 2,
                base_frequency_mhz: 2400,
                max_frequency_mhz: 3600,
                cache_hierarchy: CacheHierarchy {
                    l1_cache_kb: 32,
                    l2_cache_kb: 256,
                    l3_cache_kb: Some(8192),
                    cache_line_size: 64,
                },
                performance_characteristics: CpuPerformanceCharacteristics {
                    instructions_per_clock: 2.5,
                    context_switch_overhead: Duration::from_nanos(1000),
                    thread_creation_overhead: Duration::from_micros(50),
                    numa_topology: None,
                },
            },
            memory_model: MemoryModel {
                total_memory_mb: 16384,
                memory_type: MemoryType::Ddr4,
                memory_speed_mhz: 3200,
                bandwidth_gbps: 51.2,
                latency: Duration::from_nanos(14),
                page_size_kb: 4,
            },
            io_model: IoModel {
                storage_devices: vec![],
                total_bandwidth_mbps: 6000.0,
                average_latency: Duration::from_micros(100),
                queue_depth: 32,
            },
            network_model: NetworkModel {
                interfaces: vec![],
                total_bandwidth_mbps: 1000.0,
                latency: Duration::from_millis(1),
                packet_loss_rate: 0.0,
            },
            gpu_model: None,
            last_updated: Utc::now(),
        }
    }
}
