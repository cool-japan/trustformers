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

/// Raised when the periodic adjustment task is asked for a measurement the
/// controller has never been given.
///
/// [`AdaptiveParallelismController`] does not run the test suite and does not
/// instrument it: throughput, latency and workload shape all arrive from the
/// caller, through [`AdaptiveParallelismController::adjust_parallelism`] and
/// [`AdaptiveParallelismController::update_adjustment_performance`]. Until one
/// of those has been called there is nothing to adapt from, and the background
/// task says so instead of adapting against invented numbers.
#[derive(Debug, thiserror::Error)]
#[error(
    "adaptive parallelism has no {what} to work from yet: the controller is fed by \
     `adjust_parallelism`/`update_adjustment_performance` and neither has been called"
)]
pub struct NoRecordedTelemetry {
    /// What the adjustment task was looking for.
    pub what: &'static str,
}

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
            last_characteristics: Arc::new(Mutex::new(None)),
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

        // Remember the workload the caller described. The periodic adjustment
        // task has no other way to learn what is being run.
        *self.last_characteristics.lock() = Some(target_characteristics.clone());

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

    /// Get current system state.
    ///
    /// Every field is sampled from the running machine through `sysinfo` at
    /// call time. `load_average` is the kernel's 1-minute figure, which is
    /// `0.0` on platforms that keep no load accounting (Windows); that is
    /// `sysinfo`'s report, not a substituted default.
    ///
    /// `temperature_metrics` is populated only when the platform exposes
    /// thermal components with a reading; otherwise it stays `None` rather
    /// than carrying an assumed temperature.
    async fn get_current_system_state(&self) -> Result<SystemState> {
        let mut system = sysinfo::System::new();
        system.refresh_memory();
        system.refresh_processes(sysinfo::ProcessesToUpdate::All, true);

        let temperature_metrics = Self::read_temperature_metrics();

        Ok(SystemState {
            available_cores: num_cpus::get(),
            available_memory_mb: system.available_memory() / (1024 * 1024),
            load_average: sysinfo::System::load_average().one as f32,
            active_processes: system.processes().len(),
            temperature_metrics,
        })
    }

    /// Read CPU/system temperatures from the platform's thermal components.
    ///
    /// Returns `None` when the platform exposes no components at all, or none
    /// of them reports a temperature — a machine whose sensors are not
    /// readable has no temperature to report, and zero would be a lie.
    fn read_temperature_metrics() -> Option<TemperatureMetrics> {
        let components = sysinfo::Components::new_with_refreshed_list();
        let readings: Vec<f32> = components.iter().filter_map(|c| c.temperature()).collect();
        if readings.is_empty() {
            return None;
        }

        let hottest = readings.iter().copied().fold(f32::MIN, f32::max);
        let mean = readings.iter().sum::<f32>() / readings.len() as f32;
        let cpu_temperature = components
            .iter()
            .find(|c| {
                let label = c.label().to_ascii_lowercase();
                label.contains("cpu") || label.contains("core") || label.contains("package")
            })
            .and_then(|c| c.temperature())
            .unwrap_or(hottest);

        // `sysinfo` exposes a per-component critical threshold; treat the
        // machine as throttling only when a component that publishes one is at
        // or past it. No threshold published means no claim either way.
        let thermal_throttling = components.iter().any(|c| match (c.temperature(), c.critical()) {
            (Some(current), Some(critical)) => current >= critical,
            _ => false,
        });

        Some(TemperatureMetrics {
            cpu_temperature,
            gpu_temperature: None,
            system_temperature: mean,
            thermal_throttling,
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
                    // Having no telemetry yet is the normal state of a
                    // controller nobody has driven, not a failure; anything
                    // else is.
                    if e.downcast_ref::<NoRecordedTelemetry>().is_some() {
                        log::debug!("Adaptive adjustment skipped: {}", e);
                    } else {
                        log::error!("Adaptive adjustment failed: {}", e);
                    }
                }
            }
        });

        Ok(task)
    }

    /// Perform periodic adaptive adjustment
    async fn perform_adaptive_adjustment(&self) -> Result<()> {
        // The most recent measurement a caller supplied. Errors out (and the
        // caller logs it) while the controller has never been fed one.
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

    /// Get the most recent performance measurement the controller was given.
    ///
    /// Prefers the post-adjustment measurement of the latest adjustment, since
    /// that describes the level currently in force; falls back to the
    /// pre-adjustment measurement when
    /// [`AdaptiveParallelismController::update_adjustment_performance`] has not
    /// been called for it yet.
    ///
    /// Returns [`NoRecordedTelemetry`] when no measurement has ever been
    /// supplied. The controller has no instrumentation of its own, so there is
    /// nothing else it could truthfully return.
    async fn get_current_performance(&self) -> Result<PerformanceMeasurement> {
        let history = self.adjustment_history.lock();
        let latest = history.last().ok_or(NoRecordedTelemetry {
            what: "performance measurement",
        })?;
        Ok(latest
            .performance_after
            .clone()
            .unwrap_or_else(|| latest.performance_before.clone()))
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

        // Trend = mean throughput of the 3 most recent recorded points against
        // the mean of everything older in the window.
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

    /// Get the workload description most recently supplied by a caller.
    ///
    /// Recorded by [`AdaptiveParallelismController::adjust_parallelism`].
    /// Returns [`NoRecordedTelemetry`] before the first call: the controller
    /// cannot see the test suite, so it has no way to characterise a workload
    /// nobody has described to it.
    async fn get_current_test_characteristics(&self) -> Result<TestCharacteristics> {
        self.last_characteristics.lock().clone().ok_or_else(|| {
            NoRecordedTelemetry {
                what: "test characteristics",
            }
            .into()
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
