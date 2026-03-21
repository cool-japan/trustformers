//! Safe Concurrency Estimator
//!
//! Implements multiple sophisticated algorithms for determining optimal concurrency
//! levels including conservative, optimistic, adaptive, and machine learning approaches.

use super::super::types::*;
use anyhow::Result;
use chrono::Utc;
use parking_lot::{Mutex, RwLock};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

pub struct SafeConcurrencyEstimator {
    /// Estimation algorithms
    algorithms: Arc<Mutex<Vec<Box<dyn ConcurrencyEstimationAlgorithm + Send + Sync>>>>,

    /// Algorithm performance history
    algorithm_performance: Arc<Mutex<HashMap<String, AlgorithmPerformance>>>,

    /// Estimation history for learning
    estimation_history: Arc<Mutex<Vec<EstimationRecord>>>,

    /// Safety constraints
    safety_constraints: Arc<RwLock<EstimationSafetyConstraints>>,

    /// Configuration
    config: EstimationConfig,
}

impl SafeConcurrencyEstimator {
    /// Creates a new safe concurrency estimator
    pub async fn new(config: EstimationConfig) -> Result<Self> {
        let mut algorithms: Vec<Box<dyn ConcurrencyEstimationAlgorithm + Send + Sync>> = Vec::new();

        // Initialize estimation algorithms
        algorithms.push(Box::new(ConservativeEstimationAlgorithm::new(0.2, true)));
        algorithms.push(Box::new(OptimisticEstimationAlgorithm::new(false, 1.5)));
        algorithms.push(Box::new(MLBasedEstimationAlgorithm::new(
            "default".to_string(),
            0.85,
        )));

        Ok(Self {
            algorithms: Arc::new(Mutex::new(algorithms)),
            algorithm_performance: Arc::new(Mutex::new(HashMap::new())),
            estimation_history: Arc::new(Mutex::new(Vec::new())),
            safety_constraints: Arc::new(RwLock::new(EstimationSafetyConstraints::default())),
            config,
        })
    }

    /// Estimates safe concurrency level using multiple algorithms
    pub async fn estimate_safe_concurrency(
        &self,
        test_data: &TestExecutionData,
    ) -> Result<ConcurrencyEstimationResult> {
        let start_time = Utc::now();

        // Create preliminary analysis result for algorithms
        let preliminary_result = ConcurrencyAnalysisResult {
            timestamp: Utc::now(),
            test_id: test_data.test_id.clone(),
            max_safe_concurrency: num_cpus::get(), // Default to CPU count
            recommended_concurrency: (num_cpus::get() / 2).max(1), // Conservative initial estimate
            resource_conflicts: Vec::new(),
            lock_dependencies: Vec::new(),
            sharing_capabilities: Vec::new(),
            safety_constraints: SafetyConstraints::default(),
            recommendations: Vec::new(),
            confidence: 0.5, // Low initial confidence
            performance_impact: 0.5,
            requirements: ConcurrencyRequirements::default(),
            estimation_details: String::new(),
            conflict_analysis: String::new(),
            sharing_analysis: String::new(),
            deadlock_analysis: String::new(),
            risk_assessment: String::new(),
            thread_analysis: String::new(),
            lock_analysis: String::new(),
            pattern_analysis: String::new(),
            analysis_duration: Duration::from_secs(0),
            safety_validation: String::new(),
        };

        // Run all algorithms sequentially (trait objects can't be easily cloned for parallel execution)
        let algorithms = self.algorithms.lock();
        let mut estimations = Vec::new();

        for algorithm in algorithms.iter() {
            let algorithm_name = algorithm.name().to_string();
            let estimation_start = Instant::now();
            let result = algorithm.estimate_safe_concurrency(&preliminary_result);
            let duration = estimation_start.elapsed();

            match result {
                Ok(estimation) => {
                    estimations.push(EstimationResult {
                        algorithm: algorithm_name.clone(),
                        concurrency: estimation,
                        confidence: self
                            .calculate_algorithm_confidence(&algorithm_name, test_data)
                            .await? as f64,
                        duration,
                    });

                    // Update algorithm performance
                    self.update_algorithm_performance(&algorithm_name, true, duration).await;
                },
                Err(e) => {
                    log::warn!("Algorithm {} failed: {}", algorithm_name, e);
                    self.update_algorithm_performance(&algorithm_name, false, duration).await;
                },
            }
        }

        drop(algorithms);

        if estimations.is_empty() {
            anyhow::bail!("All estimation algorithms failed");
        }

        // Select best estimation using ensemble approach
        let (recommended, optimal) = self.select_best_estimation(&estimations, test_data).await?;

        let estimation_summaries: Vec<String> = estimations
            .iter()
            .map(|e| {
                format!(
                    "{}: {} (conf: {:.2})",
                    e.algorithm, e.concurrency, e.confidence
                )
            })
            .collect();

        let result = ConcurrencyEstimationResult {
            recommended_concurrency: recommended,
            optimal_concurrency: optimal,
            max_safe_concurrency: ((optimal as f64) * (1.0 + self.config.safety_margin)) as usize,
            is_parallelizable: optimal > 1, // Test is parallelizable if optimal concurrency > 1
            estimations: estimation_summaries,
            analysis_confidence: self.calculate_overall_estimation_confidence(&estimations) as f64,
            estimation_duration: Utc::now()
                .signed_duration_since(start_time)
                .to_std()
                .unwrap_or_default(),
            safety_margin: self.config.safety_margin,
            performance_impact: (self.calculate_overall_estimation_confidence(&estimations) * 0.8)
                as f64, // Impact proportional to confidence
        };

        // Store estimation for learning
        self.store_estimation_record(test_data, &result).await?;

        Ok(result)
    }

    /// Calculates confidence for a specific algorithm
    async fn calculate_algorithm_confidence(
        &self,
        algorithm_name: &str,
        test_data: &TestExecutionData,
    ) -> Result<f32> {
        let performance = self.algorithm_performance.lock();

        if let Some(perf) = performance.get(algorithm_name) {
            let base_confidence = perf.success_rate;

            // Adjust confidence based on test characteristics
            let complexity_factor = self.assess_test_complexity(test_data);
            let adjusted_confidence = base_confidence * (1.0 - (complexity_factor as f64) * 0.2);

            Ok((adjusted_confidence.clamp(0.0, 1.0)) as f32)
        } else {
            Ok(0.5) // Default confidence for new algorithms
        }
    }

    /// Assesses test complexity for confidence adjustment
    fn assess_test_complexity(&self, test_data: &TestExecutionData) -> f32 {
        let trace_complexity = (test_data.execution_traces.len() as f32).log10() / 3.0;
        let resource_complexity = (test_data.resource_access_patterns.len() as f32).log10() / 2.0;
        let lock_complexity = (test_data.lock_usage.len() as f32).log10() / 2.0;

        (trace_complexity + resource_complexity + lock_complexity) / 3.0
    }

    /// Selects best estimation using ensemble approach
    async fn select_best_estimation(
        &self,
        estimations: &[EstimationResult],
        _test_data: &TestExecutionData,
    ) -> Result<(usize, usize)> {
        if estimations.is_empty() {
            return Ok((1, 1));
        }

        // Weighted average based on confidence and performance
        let mut weighted_sum = 0.0_f64;
        let mut total_weight = 0.0_f64;

        for estimation in estimations {
            let weight = estimation.confidence;
            weighted_sum += (estimation.concurrency as f64) * weight;
            total_weight += weight;
        }

        let recommended = if total_weight > 0.0 {
            (weighted_sum / total_weight).round() as usize
        } else {
            estimations.iter().map(|e| e.concurrency).min().unwrap_or(1)
        };

        // Apply safety margin
        let safety_margin = self.config.safety_margin;
        let safe_recommended = ((recommended as f64) * (1.0 - safety_margin)).ceil() as usize;

        // Calculate optimal (without safety margin)
        let optimal = recommended;

        Ok((safe_recommended.max(1), optimal.max(1)))
    }

    /// Calculates overall estimation confidence
    fn calculate_overall_estimation_confidence(&self, estimations: &[EstimationResult]) -> f32 {
        if estimations.is_empty() {
            return 0.0;
        }

        let avg_confidence: f32 =
            estimations.iter().map(|e| e.confidence).sum::<f64>() as f32 / estimations.len() as f32;
        let consensus_factor = self.calculate_consensus_factor(estimations);

        avg_confidence * consensus_factor
    }

    /// Calculates consensus factor based on estimation agreement
    fn calculate_consensus_factor(&self, estimations: &[EstimationResult]) -> f32 {
        if estimations.len() < 2 {
            return 1.0;
        }

        let concurrencies: Vec<usize> = estimations.iter().map(|e| e.concurrency).collect();
        let mean = concurrencies.iter().sum::<usize>() as f32 / concurrencies.len() as f32;

        let variance = concurrencies.iter().map(|&c| (c as f32 - mean).powi(2) as f64).sum::<f64>()
            as f32
            / concurrencies.len() as f32;

        let std_dev = variance.sqrt();
        let coefficient_of_variation = if mean > 0.0 { std_dev / mean } else { 1.0 };

        // Higher consensus (lower variation) = higher factor
        (1.0 - coefficient_of_variation.min(1.0)).max(0.1)
    }

    /// Generates timeout requirements based on estimations
    fn generate_timeout_requirements(
        &self,
        estimations: &[EstimationResult],
    ) -> TimeoutRequirements {
        let max_duration = estimations
            .iter()
            .map(|e| e.duration)
            .max()
            .unwrap_or(Duration::from_millis(100));

        TimeoutRequirements {
            estimation_timeout: max_duration * 2,
            execution_timeout: max_duration * 10,
            cleanup_timeout: max_duration,
        }
    }

    /// Updates algorithm performance metrics
    async fn update_algorithm_performance(
        &self,
        algorithm_name: &str,
        success: bool,
        duration: Duration,
    ) {
        let mut performance = self.algorithm_performance.lock();
        let entry = performance.entry(algorithm_name.to_string()).or_default();

        entry.total_runs += 1;
        if success {
            entry.successful_runs += 1;
        }
        entry.total_duration += duration;
        entry.success_rate = entry.successful_runs as f64 / entry.total_runs as f64;
        let avg_secs = entry.total_duration.as_secs_f64() / entry.total_runs as f64;
        entry.avg_duration = Duration::from_secs_f64(avg_secs);
    }

    /// Stores estimation record for learning
    async fn store_estimation_record(
        &self,
        test_data: &TestExecutionData,
        result: &ConcurrencyEstimationResult,
    ) -> Result<()> {
        // TODO: from_test_data requires 4 arguments, using defaults for missing data
        let record = EstimationRecord {
            timestamp: Utc::now(),
            test_id: test_data.test_id.clone(),
            test_characteristics: TestCharacteristics::from_test_data(
                test_data.test_id.clone(),
                ResourceIntensity {
                    cpu_intensity: 0.0,
                    memory_intensity: 0.0,
                    io_intensity: 0.0,
                    network_intensity: 0.0,
                    gpu_intensity: 0.0,
                    overall_intensity: 0.0,
                    peak_periods: Vec::new(),
                    usage_variance: 0.0,
                    baseline_comparison: 0.0,
                    calculation_method: IntensityCalculationMethod::MovingAverage,
                },
                ConcurrencyRequirements::default(),
                SynchronizationRequirements {
                    synchronization_points: Vec::new(),
                    lock_usage_patterns: Vec::new(),
                    coordination_requirements: Vec::new(),
                    synchronization_overhead: 0.0,
                    deadlock_prevention: Vec::new(),
                    optimization_opportunities: Vec::new(),
                    complexity_score: 0.0,
                    performance_impact: 0.0,
                    alternative_strategies: Vec::new(),
                    average_wait_time: Duration::from_millis(0),
                    ordered_locking: false,
                    timeout_based_locking: false,
                    resource_ordering: Vec::new(),
                    lock_free_alternatives: Vec::new(),
                    custom_requirements: Vec::new(),
                },
            ),
            estimation_result: EstimationResult {
                algorithm: "ensemble".to_string(),
                concurrency: result.recommended_concurrency,
                confidence: result.analysis_confidence,
                duration: result.estimation_duration,
            },
        };

        let mut history = self.estimation_history.lock();
        history.push(record);

        // Cleanup old records if needed
        let retention_limit = self.config.history_retention_limit;
        if history.len() > retention_limit {
            let history_len = history.len();
            history.drain(0..history_len - retention_limit);
        }

        Ok(())
    }
}
