//! Network adaptation manager - the main coordinator for all network adaptation components.
//!
//! This module provides the NetworkAdaptationManager which orchestrates all aspects of
//! network adaptation including monitoring, scheduling, bandwidth optimization, synchronization,
//! prediction, and statistics collection for mobile federated learning.

use std::time::Instant;
use trustformers_core::Result;

use crate::device_info::MobileDeviceInfo;

use super::bandwidth::BandwidthOptimizer;
use super::monitoring::NetworkMonitor;
use super::prediction::NetworkPredictor;
use super::scheduling::FederatedScheduler;
use super::stats::NetworkAdaptationStats;
use super::sync::ModelSyncCoordinator;
use super::sync::SyncStatus;
use super::types::{
    FederatedTask, NetworkAdaptationConfig, NetworkConditions, NetworkQuality, TaskPriority,
};

/// Network adaptation manager for federated learning
///
/// This is the main entry point for all network adaptation functionality.
/// It coordinates between all the specialized components to provide intelligent
/// network-aware federated learning capabilities.
pub struct NetworkAdaptationManager {
    config: NetworkAdaptationConfig,
    network_monitor: NetworkMonitor,
    communication_scheduler: FederatedScheduler,
    bandwidth_optimizer: BandwidthOptimizer,
    sync_coordinator: ModelSyncCoordinator,
    network_predictor: NetworkPredictor,
    adaptation_stats: NetworkAdaptationStats,
}

impl NetworkAdaptationManager {
    /// Create new network adaptation manager
    ///
    /// # Arguments
    /// * `config` - Network adaptation configuration
    /// * `device_info` - Mobile device information for optimization
    ///
    /// # Returns
    /// * `Result<Self>` - New network adaptation manager instance
    pub fn new(config: NetworkAdaptationConfig, device_info: &MobileDeviceInfo) -> Result<Self> {
        let network_monitor = NetworkMonitor::new(config.clone())?;
        let communication_scheduler = FederatedScheduler::new(config.clone())?;
        let bandwidth_optimizer = BandwidthOptimizer::new(config.clone())?;
        let sync_coordinator = ModelSyncCoordinator::new(config.clone())?;
        let network_predictor = NetworkPredictor::new(config.clone())?;
        let adaptation_stats = NetworkAdaptationStats::new();

        Ok(Self {
            config,
            network_monitor,
            communication_scheduler,
            bandwidth_optimizer,
            sync_coordinator,
            network_predictor,
            adaptation_stats,
        })
    }

    /// Start network adaptation system
    ///
    /// Initializes and starts all subsystems including monitoring, scheduling,
    /// bandwidth optimization, synchronization coordination, and prediction.
    ///
    /// # Returns
    /// * `Result<()>` - Success or error
    pub fn start(&mut self) -> Result<()> {
        self.network_monitor.start()?;
        self.communication_scheduler.start()?;
        self.bandwidth_optimizer.start()?;
        self.sync_coordinator.start()?;

        if self.config.prediction_config.enable_prediction {
            self.network_predictor.start()?;
        }

        Ok(())
    }

    /// Stop network adaptation system
    ///
    /// Gracefully shuts down all subsystems and releases resources.
    ///
    /// # Returns
    /// * `Result<()>` - Success or error
    pub fn stop(&mut self) -> Result<()> {
        self.network_monitor.stop()?;
        self.communication_scheduler.stop()?;
        self.bandwidth_optimizer.stop()?;
        self.sync_coordinator.stop()?;
        self.network_predictor.stop()?;

        Ok(())
    }

    /// Schedule federated learning task
    ///
    /// Intelligently schedules a federated learning task based on current network
    /// conditions, device state, and predictive analysis. If conditions are suitable,
    /// the task is executed immediately; otherwise, it's scheduled for an optimal time.
    ///
    /// # Arguments
    /// * `task` - The federated learning task to schedule
    ///
    /// # Returns
    /// * `Result<String>` - Task ID for tracking
    pub fn schedule_task(&mut self, task: FederatedTask) -> Result<String> {
        // Record task scheduling for statistics
        self.adaptation_stats.record_task_scheduled(&task);

        // Assess current network conditions
        let current_conditions = self.network_monitor.get_current_conditions();

        // Check if task can be executed now
        if self.can_execute_task(&task, &current_conditions) {
            self.communication_scheduler.schedule_immediate(&task)?;
        } else {
            // Find optimal scheduling time using prediction
            let optimal_time = self.find_optimal_schedule_time(&task)?;
            self.communication_scheduler.schedule_for_time(&task, optimal_time)?;
        }

        Ok(task.task_id)
    }

    /// Execute pending tasks
    ///
    /// Processes all pending federated learning tasks that are ready to execute
    /// based on current network conditions and scheduling constraints.
    ///
    /// # Returns
    /// * `Result<Vec<String>>` - IDs of executed tasks
    pub fn execute_pending_tasks(&mut self) -> Result<Vec<String>> {
        let current_conditions = self.network_monitor.get_current_conditions();

        // Get tasks ready for execution
        let mut executed_task_ids = Vec::new();

        while let Some(task) = self.communication_scheduler.get_next_ready_task() {
            match self.execute_task(&task) {
                Ok(task_id) => {
                    executed_task_ids.push(task_id);
                    // Record successful task completion
                    let task_performance = super::types::TaskPerformance {
                        task_type: task.task_type,
                        network_conditions: current_conditions.clone(),
                        execution_time_ms: 1000,      // placeholder
                        data_transferred_mb: 1.0,     // placeholder
                        success_rate: 1.0,            // success
                        battery_consumption_mwh: 0.1, // placeholder
                    };
                    let _ = self
                        .communication_scheduler
                        .record_task_completion(&task, task_performance);
                },
                Err(e) => {
                    // Record task failure in stats
                    self.adaptation_stats.record_task_failed(&task, &e.to_string());
                },
            }
        }

        Ok(executed_task_ids)
    }

    /// Execute individual task
    ///
    /// Executes a single federated learning task with full network adaptation,
    /// including bandwidth optimization, compression, and synchronization.
    ///
    /// # Arguments
    /// * `task` - The task to execute
    ///
    /// # Returns
    /// * `Result<String>` - Task ID on success
    fn execute_task(&mut self, task: &FederatedTask) -> Result<String> {
        let start_time = Instant::now();
        let current_conditions = self.network_monitor.get_current_conditions();

        // Optimize task data for transmission
        let optimized_data = self.optimize_task_data(task, &current_conditions)?;

        // Execute the task with bandwidth optimization
        let result = self.execute_optimized_task(task, &optimized_data)?;

        // Record completion statistics
        let completion_time = start_time.elapsed().as_millis() as u64;
        let data_used = optimized_data.len() as u64;

        self.adaptation_stats.record_task_completed(
            task,
            completion_time,
            data_used,
            current_conditions.connection_type,
        );

        Ok(task.task_id.clone())
    }

    /// Optimize task data for transmission
    ///
    /// Applies intelligent compression and optimization strategies based on
    /// current network conditions and task requirements.
    ///
    /// # Arguments
    /// * `task` - The task to optimize
    /// * `conditions` - Current network conditions
    ///
    /// # Returns
    /// * `Result<Vec<u8>>` - Optimized task data
    fn optimize_task_data(
        &mut self,
        task: &FederatedTask,
        conditions: &NetworkConditions,
    ) -> Result<Vec<u8>> {
        // Create mock task data - in practice this would be the actual model/gradient data
        let task_data = vec![0u8; task.estimated_size_mb * 1024 * 1024];

        // Apply bandwidth optimization
        let data_type = match task.task_type {
            super::types::FederatedTaskType::ModelDownload => "model",
            super::types::FederatedTaskType::GradientUpload => "gradient",
            super::types::FederatedTaskType::FullModelSync => "model",
            super::types::FederatedTaskType::IncrementalSync => "differential",
            _ => "general",
        };

        self.bandwidth_optimizer.optimize_transmission(task_data, data_type)
    }

    /// Execute optimized task
    ///
    /// Performs the actual task execution with the optimized data.
    ///
    /// # Arguments
    /// * `task` - The task to execute
    /// * `optimized_data` - Pre-optimized task data
    ///
    /// # Returns
    /// * `Result<Vec<u8>>` - Task execution result
    fn execute_optimized_task(
        &mut self,
        task: &FederatedTask,
        optimized_data: &[u8],
    ) -> Result<Vec<u8>> {
        // In a real implementation, this would:
        // 1. Send data to server/peers
        // 2. Receive responses
        // 3. Apply model updates
        // 4. Handle synchronization conflicts

        // For now, return a mock result
        Ok(optimized_data.to_vec())
    }

    /// Check if task can be executed under current conditions
    ///
    /// Evaluates whether a task meets all requirements for immediate execution
    /// including network quality, bandwidth, latency, and data usage constraints.
    ///
    /// # Arguments
    /// * `task` - The task to evaluate
    /// * `conditions` - Current network conditions
    ///
    /// # Returns
    /// * `bool` - Whether task can be executed now
    fn can_execute_task(&self, task: &FederatedTask, conditions: &NetworkConditions) -> bool {
        // Check if network monitoring meets requirements
        if !self.network_monitor.meets_requirements(&task.network_requirements) {
            return false;
        }

        // Check network quality requirements based on task priority
        match conditions.quality_assessment {
            NetworkQuality::Poor => task.priority >= TaskPriority::Critical,
            NetworkQuality::Fair => task.priority >= TaskPriority::High,
            NetworkQuality::Good => task.priority >= TaskPriority::Normal,
            NetworkQuality::Excellent => true,
        }
    }

    /// Find optimal scheduling time for task
    ///
    /// Uses network prediction capabilities to identify the best time window
    /// for executing a task based on forecasted network conditions.
    ///
    /// # Arguments
    /// * `task` - The task to schedule
    ///
    /// # Returns
    /// * `Result<Instant>` - Optimal execution time
    fn find_optimal_schedule_time(&self, task: &FederatedTask) -> Result<Instant> {
        // Use network predictor to find best time window
        let prediction_result = self
            .network_predictor
            .predict_conditions(self.config.prediction_config.prediction_window_minutes)?;

        let mut best_time = Instant::now();
        let mut best_score = 0.0;

        for (time, predicted_conditions) in prediction_result.get_predicted_conditions() {
            let score = self.calculate_scheduling_score(task, predicted_conditions);
            if score > best_score {
                best_score = score;
                best_time = *time;
            }
        }

        Ok(best_time)
    }

    /// Calculate scheduling score for task at given conditions
    ///
    /// Computes a composite score that considers bandwidth, latency, quality,
    /// and stability to determine the suitability of executing a task.
    ///
    /// # Arguments
    /// * `task` - The task to score
    /// * `conditions` - Network conditions to evaluate
    ///
    /// # Returns
    /// * `f32` - Scheduling score (higher is better)
    fn calculate_scheduling_score(
        &self,
        task: &FederatedTask,
        conditions: &NetworkConditions,
    ) -> f32 {
        let mut score = 0.0;

        // Bandwidth score (0-60 points)
        let bandwidth_ratio =
            conditions.bandwidth_mbps / task.network_requirements.min_bandwidth_mbps;
        score += bandwidth_ratio.min(2.0) * 30.0;

        // Latency score (0-20 points)
        let latency_score = if conditions.latency_ms <= task.network_requirements.max_latency_ms {
            (task.network_requirements.max_latency_ms - conditions.latency_ms)
                / task.network_requirements.max_latency_ms
                * 20.0
        } else {
            0.0
        };
        score += latency_score;

        // Quality score (0-20 points)
        let quality_score = match conditions.quality_assessment {
            NetworkQuality::Excellent => 20.0,
            NetworkQuality::Good => 15.0,
            NetworkQuality::Fair => 10.0,
            NetworkQuality::Poor => 0.0,
        };
        score += quality_score;

        // Stability score (0-10 points)
        score += conditions.stability_score * 10.0;

        // Priority bonus - higher priority tasks get scheduling preference
        let priority_bonus = match task.priority {
            TaskPriority::Emergency => 20.0,
            TaskPriority::Critical => 15.0,
            TaskPriority::High => 10.0,
            TaskPriority::Normal => 5.0,
            TaskPriority::Low => 0.0,
        };
        score += priority_bonus;

        score
    }

    /// Update network conditions
    ///
    /// Updates the network monitor with new condition measurements.
    /// This method should be called regularly with fresh network measurements.
    ///
    /// # Arguments
    /// * `conditions` - Latest network condition measurements
    ///
    /// # Returns
    /// * `Result<()>` - Success or error
    pub fn update_network_conditions(&mut self, conditions: NetworkConditions) -> Result<()> {
        // Update network monitor
        self.network_monitor.update_conditions(conditions.clone())?;

        // Update network predictor with historical data
        self.network_predictor.add_historical_data(conditions.clone())?;

        // Record quality assessment for statistics
        self.adaptation_stats.record_quality_assessment(conditions.quality_assessment);

        Ok(())
    }

    /// Get current network conditions
    ///
    /// # Returns
    /// * `NetworkConditions` - Current network state
    pub fn get_current_network_conditions(&self) -> NetworkConditions {
        self.network_monitor.get_current_conditions()
    }

    /// Get current network adaptation statistics
    ///
    /// # Returns
    /// * `&NetworkAdaptationStats` - Current statistics
    pub fn get_statistics(&self) -> &NetworkAdaptationStats {
        &self.adaptation_stats
    }

    /// Get comprehensive performance metrics
    ///
    /// # Returns
    /// * `super::stats::PerformanceMetrics` - Performance metrics
    pub fn get_performance_metrics(&self) -> super::stats::PerformanceMetrics {
        self.adaptation_stats.get_performance_metrics()
    }

    /// Update configuration dynamically
    ///
    /// Updates the configuration for all components without requiring a restart.
    /// This allows for runtime optimization and adaptation.
    ///
    /// # Arguments
    /// * `new_config` - Updated configuration
    ///
    /// # Returns
    /// * `Result<()>` - Success or error
    pub fn update_config(&mut self, new_config: NetworkAdaptationConfig) -> Result<()> {
        self.config = new_config.clone();

        // Update all components with new configuration
        self.network_monitor.update_config(new_config.clone())?;
        self.communication_scheduler.update_config(new_config.clone())?;
        self.bandwidth_optimizer.update_config(new_config.clone())?;
        self.sync_coordinator.update_config(new_config.clone())?;
        self.network_predictor.update_config(new_config)?;

        Ok(())
    }

    /// Check if the system is suitable for federated learning
    ///
    /// # Returns
    /// * `bool` - Whether conditions are suitable for federated learning
    pub fn is_suitable_for_federated_learning(&self) -> bool {
        self.network_monitor.is_suitable_for_federated_learning()
    }

    /// Get network stability score
    ///
    /// # Returns
    /// * `f32` - Network stability score (0.0 to 1.0)
    pub fn get_network_stability_score(&self) -> f32 {
        self.network_monitor.get_stability_score()
    }

    /// Get bandwidth utilization
    ///
    /// # Returns
    /// * `f32` - Current bandwidth utilization ratio (0.0 to 1.0)
    pub fn get_bandwidth_utilization(&self) -> f32 {
        self.bandwidth_optimizer.get_bandwidth_utilization()
    }

    /// Get compression statistics
    ///
    /// # Returns
    /// * `&super::types::CompressionStats` - Current compression statistics
    pub fn get_compression_stats(&self) -> &super::types::CompressionStats {
        self.bandwidth_optimizer.get_compression_stats()
    }

    /// Check if approaching data usage limits
    ///
    /// # Returns
    /// * `bool` - Whether data usage is approaching configured limits
    pub fn is_approaching_data_limits(&self) -> bool {
        self.bandwidth_optimizer.is_approaching_limit()
    }

    /// Get network trend analysis
    ///
    /// # Returns
    /// * `&super::monitoring::NetworkTrendAnalyzer` - Current trend analysis
    pub fn get_network_trends(&self) -> &super::monitoring::NetworkTrendAnalyzer {
        self.network_monitor.get_trend_analysis()
    }

    /// Get synchronization status
    ///
    /// # Arguments
    /// * `sync_id` - Synchronization ID to check
    ///
    /// # Returns
    /// * `Option<&super::types::SyncStatus>` - Synchronization status if found
    pub fn get_sync_status(&self, sync_id: &str) -> Option<&SyncStatus> {
        self.sync_coordinator.get_sync_status(sync_id)
    }

    /// Force immediate synchronization
    ///
    /// Forces an immediate model synchronization, bypassing normal scheduling.
    /// This should be used sparingly and only for critical updates.
    ///
    /// # Arguments
    /// * `model_data` - Model data to synchronize
    /// * `reason` - Reason for forced synchronization
    ///
    /// # Returns
    /// * `Result<String>` - Synchronization ID
    pub fn force_sync(&mut self, model_data: Vec<u8>, reason: String) -> Result<String> {
        self.sync_coordinator.force_sync(model_data, reason)
    }

    /// Execute pending synchronizations
    ///
    /// Processes all pending model synchronizations that are ready to execute.
    ///
    /// # Returns
    /// * `Result<Vec<super::sync::SyncResponse>>` - Synchronization responses
    pub fn execute_pending_syncs(&mut self) -> Result<Vec<super::sync::SyncResponse>> {
        self.sync_coordinator.execute_pending_syncs()
    }

    /// Get prediction accuracy
    ///
    /// # Returns
    /// * `f32` - Current prediction accuracy (0.0 to 1.0)
    pub fn get_prediction_accuracy(&self) -> f32 {
        let stats = self.network_predictor.get_prediction_stats();
        stats.get("average_accuracy").copied().unwrap_or(0.0)
    }

    /// Optimize communication strategy
    ///
    /// Updates the communication strategy based on current conditions and
    /// historical performance data.
    ///
    /// # Returns
    /// * `Result<()>` - Success or error
    pub fn optimize_communication_strategy(&mut self) -> Result<()> {
        let current_conditions = self.network_monitor.get_current_conditions();
        let stats = &self.adaptation_stats;

        // Generate optimization recommendations
        let recommendations =
            super::stats::NetworkAdaptationUtils::generate_optimization_recommendations(
                stats,
                &self.config,
                // In practice, we'd need device info here - for now use a simplified approach
                &crate::device_info::MobileDeviceInfo::default(),
            );

        // Apply high-priority recommendations automatically
        if recommendations.priority >= 8 {
            for recommendation in recommendations.config_recommendations {
                match recommendation.parameter.as_str() {
                    "sync_frequency.base_frequency_minutes" => {
                        if let Ok(new_freq) = recommendation.recommended_value.parse::<u32>() {
                            self.config.sync_frequency.base_frequency_minutes = new_freq;
                        }
                    },
                    "communication_strategy.compression_config.model_compression_ratio" => {
                        if let Ok(new_ratio) = recommendation.recommended_value.parse::<f32>() {
                            self.config
                                .communication_strategy
                                .compression_config
                                .model_compression_ratio = new_ratio;
                        }
                    },
                    "enable_bandwidth_optimization" => {
                        if let Ok(enable) = recommendation.recommended_value.parse::<bool>() {
                            self.config.enable_bandwidth_optimization = enable;
                        }
                    },
                    _ => {
                        // Log unhandled recommendations for manual review
                    },
                }
            }

            // Update all components with the optimized configuration
            self.update_config(self.config.clone())?;
        }

        Ok(())
    }

    /// Get system health summary
    ///
    /// # Returns
    /// * `super::stats::NetworkHealthAssessment` - Comprehensive health assessment
    pub fn get_system_health(&self) -> super::stats::NetworkHealthAssessment {
        let current_conditions = self.network_monitor.get_current_conditions();
        super::stats::NetworkAdaptationUtils::analyze_network_health(&current_conditions)
    }

    /// Generate performance report
    ///
    /// # Returns
    /// * `String` - Formatted performance report
    pub fn generate_performance_report(&self) -> String {
        let stats_summary = self.adaptation_stats.generate_summary();
        let health_assessment = self.get_system_health();
        let performance_metrics = self.get_performance_metrics();

        format!(
            "Network Adaptation Performance Report\n\
             =====================================\n\
             \n\
             {}\n\
             \n\
             Network Health Assessment:\n\
             - Overall Health Score: {:.1}/100\n\
             - Bandwidth Score: {:.1}/100\n\
             - Latency Score: {:.1}/100\n\
             - Stability Score: {:.1}/100\n\
             - Reliability Score: {:.1}/100\n\
             \n\
             Performance Metrics:\n\
             - Throughput: {:.2} tasks/minute\n\
             - Network Utilization: {:.1}%\n\
             - Battery Efficiency: {:.2} tasks/mAh\n\
             - Compression Efficiency: {:.1}%\n\
             - Prediction Accuracy: {:.1}%\n\
             \n\
             Recommendations:\n\
             {}\n",
            stats_summary,
            health_assessment.overall_health_score,
            health_assessment.bandwidth_score,
            health_assessment.latency_score,
            health_assessment.stability_score,
            health_assessment.reliability_score,
            performance_metrics.throughput_tasks_per_minute,
            performance_metrics.network_utilization * 100.0,
            performance_metrics.battery_efficiency,
            performance_metrics.compression_efficiency * 100.0,
            performance_metrics.prediction_accuracy * 100.0,
            health_assessment.recommendations.join("\n- ")
        )
    }
}

// Default implementations for convenience
impl Default for NetworkAdaptationManager {
    fn default() -> Self {
        let config = NetworkAdaptationConfig::default();
        let device_info = MobileDeviceInfo::default();

        Self::new(config, &device_info).unwrap_or_else(|_| {
            // Fallback implementation for default
            Self {
                config: NetworkAdaptationConfig::default(),
                network_monitor: NetworkMonitor::new(NetworkAdaptationConfig::default()).unwrap(),
                communication_scheduler:
                    FederatedScheduler::new(NetworkAdaptationConfig::default()).unwrap(),
                bandwidth_optimizer: BandwidthOptimizer::new(NetworkAdaptationConfig::default())
                    .unwrap(),
                sync_coordinator: ModelSyncCoordinator::new(NetworkAdaptationConfig::default())
                    .unwrap(),
                network_predictor: NetworkPredictor::new(NetworkAdaptationConfig::default())
                    .unwrap(),
                adaptation_stats: NetworkAdaptationStats::new(),
            }
        })
    }
}
