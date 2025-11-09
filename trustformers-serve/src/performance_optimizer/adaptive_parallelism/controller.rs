//! Adaptive Parallelism Controller Implementation
//!
//! This module provides the main AdaptiveParallelismController which orchestrates
//! parallelism adjustments based on performance feedback, machine learning models,
//! and system conditions. It includes adaptive adjustment strategies, conservative
//! mode, and comprehensive performance tracking.

use anyhow::Result;
use chrono::Utc;
use parking_lot::{Mutex, RwLock};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::{task::JoinHandle, time};

use crate::performance_optimizer::types::*;
// OptimalParallelismEstimator, PerformanceFeedbackSystem, and AdaptiveLearningModel
// are imported from crate::performance_optimizer::types::* above

// =============================================================================
// ADAPTIVE PARALLELISM CONTROLLER IMPLEMENTATION
// =============================================================================

impl AdaptiveParallelismController {
    /// Create a new adaptive parallelism controller
    ///
    /// Initializes the controller with default configuration and sets up
    /// all necessary components for adaptive parallelism management.
    pub async fn new(config: AdaptiveParallelismConfig) -> Result<Self> {
        let optimal_estimator = Arc::new(OptimalParallelismEstimator::new().await?);
        let feedback_system = Arc::new(PerformanceFeedbackSystem::new().await?);
        let learning_model = Arc::new(AdaptiveLearningModel::new().await?);

        Ok(Self {
            current_parallelism: Arc::new(AtomicUsize::new(config.min_parallelism)),
            optimal_estimator,
            adjustment_history: Arc::new(Mutex::new(Vec::new())),
            feedback_system,
            learning_model,
            config: Arc::new(RwLock::new(config)),
        })
    }

    /// Get current parallelism level
    pub fn current_parallelism(&self) -> usize {
        self.current_parallelism.load(Ordering::Relaxed)
    }

    /// Set parallelism level with bounds checking
    pub fn set_parallelism(&self, level: usize) -> Result<()> {
        let config = self.config.read();
        let bounded_level = level.clamp(config.min_parallelism, config.max_parallelism);

        self.current_parallelism.store(bounded_level, Ordering::Relaxed);

        if bounded_level != level {
            log::warn!(
                "Parallelism level {} was clamped to {} (bounds: {}-{})",
                level,
                bounded_level,
                config.min_parallelism,
                config.max_parallelism
            );
        }

        Ok(())
    }

    /// Recommend optimal parallelism level
    ///
    /// Uses multiple estimation algorithms and machine learning models to
    /// determine the optimal parallelism level for given test characteristics.
    pub async fn recommend_parallelism(
        &self,
        characteristics: &TestCharacteristics,
    ) -> Result<ParallelismEstimate> {
        // Get system state
        let system_state = self.get_current_system_state().await?;

        // Get historical data
        let historical_data = {
            let data = self.optimal_estimator.historical_data.lock();
            data.clone()
        };

        // Get estimate from the optimal estimator
        let estimate = self
            .optimal_estimator
            .estimate_optimal_parallelism(&historical_data, characteristics, &system_state)
            .await?;

        // Apply learning model adjustments
        let adjusted_estimate = self
            .learning_model
            .adjust_estimate(&estimate, characteristics, &system_state)
            .await?;

        // Record estimation for accuracy tracking
        self.optimal_estimator.record_estimation(&adjusted_estimate).await?;

        Ok(adjusted_estimate)
    }

    /// Adjust parallelism level based on performance feedback
    ///
    /// Dynamically adjusts the parallelism level based on real-time
    /// performance feedback and system conditions.
    pub async fn adjust_parallelism(
        &self,
        reason: AdjustmentReason,
        performance_before: PerformanceMeasurement,
        target_characteristics: &TestCharacteristics,
    ) -> Result<usize> {
        let previous_level = self.current_parallelism();

        // Get recommendation
        let estimate = self.recommend_parallelism(target_characteristics).await?;
        let new_level = estimate.optimal_parallelism;

        // Apply conservative mode if enabled
        let config = self.config.read();
        let final_level = if config.conservative_mode {
            self.apply_conservative_adjustment(previous_level, new_level)
        } else {
            new_level
        };

        // Set new parallelism level
        self.set_parallelism(final_level)?;

        // Record adjustment
        let adjustment = ParallelismAdjustment {
            timestamp: Utc::now(),
            previous_level,
            new_level: final_level,
            reason,
            performance_before,
            performance_after: None, // Will be filled later
            effectiveness: None,     // Will be calculated later
        };

        self.adjustment_history.lock().push(adjustment);

        log::info!(
            "Adjusted parallelism from {} to {} (reason: {:?})",
            previous_level,
            final_level,
            reason
        );

        Ok(final_level)
    }

    /// Apply conservative adjustment strategy
    fn apply_conservative_adjustment(&self, current: usize, target: usize) -> usize {
        let config = self.config.read();
        let max_change = ((current as f32) * config.stability_threshold).max(1.0) as usize;

        if target > current {
            (current + max_change).min(target)
        } else {
            current.saturating_sub(max_change).max(target)
        }
    }

    /// Process performance feedback
    ///
    /// Processes performance feedback from various sources and updates
    /// the learning model accordingly.
    pub async fn process_feedback(&self, feedback: PerformanceFeedback) -> Result<()> {
        self.feedback_system.add_feedback(feedback).await?;

        // Trigger learning model update if enough feedback accumulated
        let feedback_count = self.feedback_system.get_feedback_count().await?;
        if feedback_count % 10 == 0 {
            // Update every 10 feedback items
            self.learning_model.update_from_feedback(&self.feedback_system).await?;
        }

        Ok(())
    }

    /// Update performance measurement after adjustment
    ///
    /// Updates the adjustment history with performance measurements
    /// taken after a parallelism adjustment.
    pub async fn update_adjustment_performance(
        &self,
        performance_after: PerformanceMeasurement,
    ) -> Result<()> {
        let mut history = self.adjustment_history.lock();
        if let Some(last_adjustment) = history.last_mut() {
            last_adjustment.performance_after = Some(performance_after.clone());

            // Calculate effectiveness
            let effectiveness = self.calculate_adjustment_effectiveness(
                &last_adjustment.performance_before,
                &performance_after,
            );
            last_adjustment.effectiveness = Some(effectiveness);

            log::debug!("Updated adjustment effectiveness: {:.2}", effectiveness);
        }

        Ok(())
    }

    /// Calculate adjustment effectiveness
    fn calculate_adjustment_effectiveness(
        &self,
        before: &PerformanceMeasurement,
        after: &PerformanceMeasurement,
    ) -> f32 {
        // Calculate relative improvement in throughput and efficiency
        let throughput_improvement =
            (after.throughput - before.throughput) / before.throughput.max(0.001);
        let efficiency_improvement = (after.resource_efficiency - before.resource_efficiency)
            / before.resource_efficiency.max(0.001);

        // Combined effectiveness score
        (throughput_improvement as f32 * 0.6 + efficiency_improvement * 0.4).clamp(-1.0, 1.0)
    }

    /// Get current system state
    async fn get_current_system_state(&self) -> Result<SystemState> {
        // In a real implementation, this would collect actual system metrics
        Ok(SystemState {
            available_cores: num_cpus::get(),
            available_memory_mb: 8192, // Placeholder
            load_average: 0.5,
            active_processes: 100,
            io_wait_percent: 2.0,
            network_utilization: 0.1,
            temperature_metrics: None,
        })
    }

    /// Start adaptive adjustment background task
    ///
    /// Starts a background task that continuously monitors performance
    /// and adjusts parallelism levels automatically.
    pub async fn start_adaptive_adjustment(
        self: Arc<Self>,
        shutdown_signal: Arc<AtomicBool>,
    ) -> Result<JoinHandle<()>> {
        let controller = Arc::clone(&self);
        let shutdown = Arc::clone(&shutdown_signal);

        let task = tokio::spawn(async move {
            let mut interval = time::interval(controller.config.read().adjustment_interval);

            while !shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                if let Err(e) = controller.perform_adaptive_adjustment().await {
                    log::error!("Adaptive adjustment failed: {}", e);
                }
            }
        });

        Ok(task)
    }

    /// Perform periodic adaptive adjustment
    async fn perform_adaptive_adjustment(&self) -> Result<()> {
        // Get current performance metrics (placeholder implementation)
        let current_performance = self.get_current_performance().await?;

        // Analyze performance trends
        let trend_analysis = self.analyze_performance_trends().await?;

        // Determine if adjustment is needed
        if self.should_adjust_parallelism(&current_performance, &trend_analysis).await? {
            let test_characteristics = self.get_current_test_characteristics().await?;

            self.adjust_parallelism(
                AdjustmentReason::AlgorithmRecommendation,
                current_performance,
                &test_characteristics,
            )
            .await?;
        }

        Ok(())
    }

    /// Get current performance metrics
    async fn get_current_performance(&self) -> Result<PerformanceMeasurement> {
        // Placeholder implementation - in real system would collect actual metrics
        Ok(PerformanceMeasurement {
            throughput: 100.0,
            average_latency: Duration::from_millis(50),
            cpu_utilization: 0.6,
            memory_utilization: 0.4,
            resource_efficiency: 0.8,
            timestamp: Utc::now(),
            measurement_duration: Duration::from_secs(30),
            cpu_usage: 0.6,
            memory_usage: 0.4,
            latency: Duration::from_millis(50),
        })
    }

    /// Analyze performance trends
    async fn analyze_performance_trends(&self) -> Result<PerformanceTrend> {
        // Get recent performance data
        let historical_data = {
            let data = self.optimal_estimator.historical_data.lock();
            data.iter().rev().take(10).cloned().collect::<Vec<_>>()
        };

        if historical_data.len() < 2 {
            return Ok(PerformanceTrend {
                direction: crate::test_performance_monitoring::TrendDirection::Stable,
                strength: 0.0,
                confidence: 0.0,
                period: Duration::from_secs(300),
                data_points: historical_data,
            });
        }

        // Simple trend analysis (in real implementation would use more sophisticated methods)
        let recent_throughput: f64 =
            historical_data.iter().take(3).map(|p| p.throughput).sum::<f64>() / 3.0;
        let older_throughput: f64 =
            historical_data.iter().skip(3).map(|p| p.throughput).sum::<f64>()
                / (historical_data.len() - 3).max(1) as f64;

        let direction = if recent_throughput > older_throughput * 1.05 {
            crate::test_performance_monitoring::TrendDirection::Improving
        } else if recent_throughput < older_throughput * 0.95 {
            crate::test_performance_monitoring::TrendDirection::Degrading
        } else {
            crate::test_performance_monitoring::TrendDirection::Stable
        };

        let strength = ((recent_throughput - older_throughput) / older_throughput).abs() as f32;

        Ok(PerformanceTrend {
            direction,
            strength,
            confidence: (historical_data.len() as f32 / 10.0).min(1.0),
            period: Duration::from_secs(300),
            data_points: historical_data,
        })
    }

    /// Determine if parallelism adjustment is needed
    async fn should_adjust_parallelism(
        &self,
        _performance: &PerformanceMeasurement,
        trend: &PerformanceTrend,
    ) -> Result<bool> {
        let config = self.config.read();

        // Adjust if performance is degrading significantly
        if matches!(
            trend.direction,
            crate::test_performance_monitoring::TrendDirection::Degrading
        ) && trend.strength > config.stability_threshold
            && trend.confidence > 0.7
        {
            return Ok(true);
        }

        // Explore if performance is stable and exploration is enabled
        if matches!(
            trend.direction,
            crate::test_performance_monitoring::TrendDirection::Stable
        ) && config.exploration_rate > 0.0
        {
            use scirs2_core::random::*;
            let mut rng = thread_rng();
            if rng.random::<f32>() < config.exploration_rate {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Get current test characteristics
    async fn get_current_test_characteristics(&self) -> Result<TestCharacteristics> {
        // Placeholder implementation
        Ok(TestCharacteristics {
            category_distribution: HashMap::new(),
            average_duration: Duration::from_millis(100),
            resource_intensity: ResourceIntensity::default(),
            concurrency_requirements: ConcurrencyRequirements::default(),
            dependency_complexity: 0.3,
        })
    }

    /// Get adjustment history
    pub fn get_adjustment_history(&self) -> Vec<ParallelismAdjustment> {
        self.adjustment_history.lock().clone()
    }

    /// Get effectiveness statistics
    pub fn get_effectiveness_stats(&self) -> (f32, usize) {
        let history = self.adjustment_history.lock();
        let effective_adjustments: Vec<f32> =
            history.iter().filter_map(|adj| adj.effectiveness).collect();

        if effective_adjustments.is_empty() {
            (0.0, 0)
        } else {
            let avg_effectiveness =
                effective_adjustments.iter().sum::<f32>() / effective_adjustments.len() as f32;
            (avg_effectiveness, effective_adjustments.len())
        }
    }
}
