//! Federated task scheduling and coordination.

use super::types::*;
use crate::battery::BatteryLevel;
use crate::profiler::NetworkConnectionType;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;
use trustformers_core::error::Result;

/// Federated learning scheduler
pub struct FederatedScheduler {
    config: NetworkAdaptationConfig,
    scheduled_tasks: VecDeque<FederatedTask>,
    active_tasks: HashMap<String, FederatedTask>,
    task_prioritizer: TaskPrioritizer,
    schedule_optimizer: ScheduleOptimizer,
}

/// Task prioritizer for intelligent scheduling
pub struct TaskPrioritizer {
    priority_weights: HashMap<FederatedTaskType, f32>,
    network_weights: HashMap<NetworkConnectionType, f32>,
    battery_weights: HashMap<BatteryLevel, f32>,
}

/// Schedule optimizer for network-aware planning
pub struct ScheduleOptimizer {
    optimization_strategy: OptimizationStrategy,
    constraints: SchedulingConstraints,
    performance_predictor: PerformancePredictor,
}

/// Performance predictor for scheduling optimization
pub struct PerformancePredictor {
    historical_performance: VecDeque<TaskPerformance>,
    prediction_models: HashMap<FederatedTaskType, PredictionModel>,
    accuracy_tracker: AccuracyTracker,
}

/// Prediction model for task performance
#[derive(Debug, Clone)]
struct PredictionModel {
    model_type: ModelType,
    parameters: Vec<f32>,
    accuracy: f32,
    last_updated: Instant,
}

/// Types of prediction models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelType {
    Linear,
    Polynomial,
    ExponentialSmoothing,
    NeuralNetwork,
}

/// Accuracy tracking for predictions
pub struct AccuracyTracker {
    prediction_errors: VecDeque<f32>,
    mean_absolute_error: f32,
    root_mean_square_error: f32,
    confidence_intervals: HashMap<FederatedTaskType, f32>,
}

impl FederatedScheduler {
    /// Create new federated scheduler
    pub fn new(config: NetworkAdaptationConfig) -> Result<Self> {
        Ok(Self {
            config,
            scheduled_tasks: VecDeque::new(),
            active_tasks: HashMap::new(),
            task_prioritizer: TaskPrioritizer::new(),
            schedule_optimizer: ScheduleOptimizer::new(),
        })
    }

    /// Start the scheduler
    pub fn start(&mut self) -> Result<()> {
        // Initialize scheduling subsystem
        Ok(())
    }

    /// Stop the scheduler
    pub fn stop(&mut self) -> Result<()> {
        // Stop scheduling subsystem
        Ok(())
    }

    /// Schedule a task for immediate execution
    pub fn schedule_immediate(&mut self, task: &FederatedTask) -> Result<()> {
        let prioritized_task = self.task_prioritizer.prioritize_task(task);
        self.active_tasks.insert(task.task_id.clone(), prioritized_task);
        Ok(())
    }

    /// Schedule a task for specific time
    pub fn schedule_for_time(&mut self, task: &FederatedTask, time: Instant) -> Result<()> {
        let mut scheduled_task = task.clone();
        scheduled_task.scheduled_time = time;

        // Insert in chronological order
        let mut insert_position = None;
        for (i, existing_task) in self.scheduled_tasks.iter().enumerate() {
            if scheduled_task.scheduled_time < existing_task.scheduled_time {
                insert_position = Some(i);
                break;
            }
        }

        if let Some(position) = insert_position {
            self.scheduled_tasks.insert(position, scheduled_task);
        } else {
            self.scheduled_tasks.push_back(scheduled_task);
        }

        Ok(())
    }

    /// Get next task ready for execution
    pub fn get_next_ready_task(&mut self) -> Option<FederatedTask> {
        let now = Instant::now();

        if let Some(task) = self.scheduled_tasks.front() {
            if task.scheduled_time <= now {
                return self.scheduled_tasks.pop_front();
            }
        }

        None
    }

    /// Cancel a scheduled task
    pub fn cancel_task(&mut self, task_id: &str) -> Result<bool> {
        // Remove from active tasks
        if self.active_tasks.remove(task_id).is_some() {
            return Ok(true);
        }

        // Remove from scheduled tasks
        let original_len = self.scheduled_tasks.len();
        self.scheduled_tasks.retain(|task| task.task_id != task_id);

        Ok(self.scheduled_tasks.len() < original_len)
    }

    /// Get all active tasks
    pub fn get_active_tasks(&self) -> Vec<&FederatedTask> {
        self.active_tasks.values().collect()
    }

    /// Get all scheduled tasks
    pub fn get_scheduled_tasks(&self) -> &VecDeque<FederatedTask> {
        &self.scheduled_tasks
    }

    /// Optimize schedule based on current network conditions
    pub fn optimize_schedule(&mut self, network_conditions: &NetworkConditions) -> Result<()> {
        self.schedule_optimizer.optimize(&mut self.scheduled_tasks, network_conditions)
    }

    /// Update scheduler configuration
    pub fn update_config(&mut self, config: NetworkAdaptationConfig) -> Result<()> {
        self.config = config;
        Ok(())
    }

    /// Get scheduling statistics
    pub fn get_statistics(&self) -> SchedulingStatistics {
        SchedulingStatistics {
            active_task_count: self.active_tasks.len(),
            scheduled_task_count: self.scheduled_tasks.len(),
            total_tasks_processed: self.performance_predictor().get_total_tasks_processed(),
            average_completion_time: self.performance_predictor().get_average_completion_time(),
        }
    }

    /// Get performance predictor reference
    pub fn performance_predictor(&self) -> &PerformancePredictor {
        &self.schedule_optimizer.performance_predictor
    }

    /// Record task completion for learning
    pub fn record_task_completion(
        &mut self,
        task: &FederatedTask,
        performance: TaskPerformance,
    ) -> Result<()> {
        self.schedule_optimizer.performance_predictor.record_performance(performance);
        self.active_tasks.remove(&task.task_id);
        Ok(())
    }
}

impl TaskPrioritizer {
    /// Create new task prioritizer
    pub fn new() -> Self {
        let mut priority_weights = HashMap::new();
        priority_weights.insert(FederatedTaskType::Heartbeat, 1.0);
        priority_weights.insert(FederatedTaskType::GradientUpload, 2.0);
        priority_weights.insert(FederatedTaskType::ModelDownload, 3.0);
        priority_weights.insert(FederatedTaskType::FullModelSync, 4.0);
        priority_weights.insert(FederatedTaskType::IncrementalSync, 2.5);
        priority_weights.insert(FederatedTaskType::Checkpoint, 3.5);

        let mut network_weights = HashMap::new();
        network_weights.insert(NetworkConnectionType::WiFi, 1.0);
        network_weights.insert(NetworkConnectionType::Cellular5G, 0.8);
        network_weights.insert(NetworkConnectionType::Cellular4G, 0.6);

        let mut battery_weights = HashMap::new();
        battery_weights.insert(BatteryLevel::Critical, 0.2);
        battery_weights.insert(BatteryLevel::Low, 0.5);
        battery_weights.insert(BatteryLevel::Medium, 0.8);
        battery_weights.insert(BatteryLevel::High, 1.0);
        battery_weights.insert(BatteryLevel::Full, 1.0);

        Self {
            priority_weights,
            network_weights,
            battery_weights,
        }
    }

    /// Calculate priority score for a task
    pub fn calculate_priority_score(
        &self,
        task: &FederatedTask,
        network_type: NetworkConnectionType,
        battery_level: BatteryLevel,
    ) -> f32 {
        let task_weight = self.priority_weights.get(&task.task_type).unwrap_or(&1.0);
        let network_weight = self.network_weights.get(&network_type).unwrap_or(&1.0);
        let battery_weight = self.battery_weights.get(&battery_level).unwrap_or(&1.0);

        let priority_multiplier = match task.priority {
            TaskPriority::Emergency => 5.0,
            TaskPriority::Critical => 4.0,
            TaskPriority::High => 3.0,
            TaskPriority::Normal => 2.0,
            TaskPriority::Low => 1.0,
        };

        task_weight * network_weight * battery_weight * priority_multiplier
    }

    /// Prioritize and potentially modify a task
    pub fn prioritize_task(&self, task: &FederatedTask) -> FederatedTask {
        // For now, return the task as-is
        // In a real implementation, this might adjust scheduling parameters
        task.clone()
    }

    /// Update priority weights based on learning
    pub fn update_weights(&mut self, task_type: FederatedTaskType, success_rate: f32) {
        if let Some(weight) = self.priority_weights.get_mut(&task_type) {
            // Adjust weight based on success rate
            *weight *= 0.9 + (success_rate * 0.2);
        }
    }
}

impl ScheduleOptimizer {
    /// Create new schedule optimizer
    pub fn new() -> Self {
        Self {
            optimization_strategy: OptimizationStrategy::BalancedOptimization,
            constraints: SchedulingConstraints::default(),
            performance_predictor: PerformancePredictor::new(),
        }
    }

    /// Optimize task schedule based on network conditions
    pub fn optimize(
        &mut self,
        tasks: &mut VecDeque<FederatedTask>,
        network_conditions: &NetworkConditions,
    ) -> Result<()> {
        match self.optimization_strategy {
            OptimizationStrategy::MinimizeLatency => {
                self.optimize_for_latency(tasks, network_conditions)
            },
            OptimizationStrategy::MinimizeDataUsage => self.optimize_for_data_usage(tasks),
            OptimizationStrategy::MaximizeReliability => {
                self.optimize_for_reliability(tasks, network_conditions)
            },
            OptimizationStrategy::BalancedOptimization => {
                self.optimize_balanced(tasks, network_conditions)
            },
            OptimizationStrategy::BatteryAware => self.optimize_for_battery(tasks),
            OptimizationStrategy::NetworkAware => {
                self.optimize_for_network(tasks, network_conditions)
            },
        }
    }

    fn optimize_for_latency(
        &mut self,
        tasks: &mut VecDeque<FederatedTask>,
        _network_conditions: &NetworkConditions,
    ) -> Result<()> {
        // Sort by priority and estimated completion time
        let mut task_vec: Vec<_> = tasks.drain(..).collect();
        task_vec.sort_by(|a, b| {
            a.priority
                .cmp(&b.priority)
                .reverse()
                .then_with(|| a.estimated_size_mb.cmp(&b.estimated_size_mb))
        });

        for task in task_vec {
            tasks.push_back(task);
        }

        Ok(())
    }

    fn optimize_for_data_usage(&mut self, tasks: &mut VecDeque<FederatedTask>) -> Result<()> {
        // Sort by data size (smallest first)
        let mut task_vec: Vec<_> = tasks.drain(..).collect();
        task_vec.sort_by_key(|task| task.estimated_size_mb);

        for task in task_vec {
            tasks.push_back(task);
        }

        Ok(())
    }

    fn optimize_for_reliability(
        &mut self,
        tasks: &mut VecDeque<FederatedTask>,
        _network_conditions: &NetworkConditions,
    ) -> Result<()> {
        // Prioritize tasks with higher reliability requirements
        let mut task_vec: Vec<_> = tasks.drain(..).collect();
        task_vec.sort_by(|a, b| {
            b.network_requirements
                .required_reliability_percent
                .partial_cmp(&a.network_requirements.required_reliability_percent)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for task in task_vec {
            tasks.push_back(task);
        }

        Ok(())
    }

    fn optimize_balanced(
        &mut self,
        tasks: &mut VecDeque<FederatedTask>,
        network_conditions: &NetworkConditions,
    ) -> Result<()> {
        // Combine multiple optimization criteria
        let mut task_vec: Vec<_> = tasks.drain(..).collect();

        task_vec.sort_by(|a, b| {
            let a_score = self.calculate_balanced_score(a, network_conditions);
            let b_score = self.calculate_balanced_score(b, network_conditions);
            b_score.partial_cmp(&a_score).unwrap_or(std::cmp::Ordering::Equal)
        });

        for task in task_vec {
            tasks.push_back(task);
        }

        Ok(())
    }

    fn optimize_for_battery(&mut self, tasks: &mut VecDeque<FederatedTask>) -> Result<()> {
        // Prioritize less resource-intensive tasks
        let mut task_vec: Vec<_> = tasks.drain(..).collect();
        task_vec.sort_by_key(|task| task.estimated_size_mb);

        for task in task_vec {
            tasks.push_back(task);
        }

        Ok(())
    }

    fn optimize_for_network(
        &mut self,
        tasks: &mut VecDeque<FederatedTask>,
        network_conditions: &NetworkConditions,
    ) -> Result<()> {
        // Adapt to current network quality
        let mut task_vec: Vec<_> = tasks.drain(..).collect();

        task_vec.sort_by(|a, b| {
            let a_network_fit = self.calculate_network_fitness(a, network_conditions);
            let b_network_fit = self.calculate_network_fitness(b, network_conditions);
            b_network_fit.partial_cmp(&a_network_fit).unwrap_or(std::cmp::Ordering::Equal)
        });

        for task in task_vec {
            tasks.push_back(task);
        }

        Ok(())
    }

    fn calculate_balanced_score(
        &self,
        task: &FederatedTask,
        network_conditions: &NetworkConditions,
    ) -> f32 {
        let priority_score = match task.priority {
            TaskPriority::Emergency => 5.0,
            TaskPriority::Critical => 4.0,
            TaskPriority::High => 3.0,
            TaskPriority::Normal => 2.0,
            TaskPriority::Low => 1.0,
        };

        let size_penalty = 1.0 / (1.0 + task.estimated_size_mb as f32 / 100.0);
        let network_bonus = network_conditions.bandwidth_mbps / 10.0;

        priority_score * size_penalty * (1.0 + network_bonus)
    }

    fn calculate_network_fitness(
        &self,
        task: &FederatedTask,
        network_conditions: &NetworkConditions,
    ) -> f32 {
        let bandwidth_match =
            if network_conditions.bandwidth_mbps >= task.network_requirements.min_bandwidth_mbps {
                1.0
            } else {
                network_conditions.bandwidth_mbps / task.network_requirements.min_bandwidth_mbps
            };

        let latency_match =
            if network_conditions.latency_ms <= task.network_requirements.max_latency_ms {
                1.0
            } else {
                task.network_requirements.max_latency_ms / network_conditions.latency_ms
            };

        (bandwidth_match + latency_match) / 2.0
    }

    /// Set optimization strategy
    pub fn set_strategy(&mut self, strategy: OptimizationStrategy) {
        self.optimization_strategy = strategy;
    }

    /// Get current optimization strategy
    pub fn get_strategy(&self) -> OptimizationStrategy {
        self.optimization_strategy
    }

    /// Update scheduling constraints
    pub fn update_constraints(&mut self, constraints: SchedulingConstraints) {
        self.constraints = constraints;
    }
}

impl PerformancePredictor {
    /// Create new performance predictor
    pub fn new() -> Self {
        Self {
            historical_performance: VecDeque::new(),
            prediction_models: HashMap::new(),
            accuracy_tracker: AccuracyTracker::new(),
        }
    }

    /// Record task performance for learning
    pub fn record_performance(&mut self, performance: TaskPerformance) {
        self.historical_performance.push_back(performance.clone());

        // Maintain reasonable history size
        if self.historical_performance.len() > 1000 {
            self.historical_performance.pop_front();
        }

        // Update prediction models
        self.update_prediction_model(performance.task_type, &performance);
    }

    /// Predict task completion time
    pub fn predict_completion_time(
        &self,
        task: &FederatedTask,
        network_conditions: &NetworkConditions,
    ) -> f32 {
        if let Some(model) = self.prediction_models.get(&task.task_type) {
            self.apply_prediction_model(model, task, network_conditions)
        } else {
            // Fallback to simple heuristic
            (task.estimated_size_mb as f32 / network_conditions.bandwidth_mbps) * 8.0 * 1000.0
            // Convert to milliseconds
        }
    }

    /// Get average completion time for task type
    pub fn get_average_completion_time(&self) -> f32 {
        if self.historical_performance.is_empty() {
            return 0.0;
        }

        let total_time: u64 = self.historical_performance.iter().map(|p| p.execution_time_ms).sum();

        total_time as f32 / self.historical_performance.len() as f32
    }

    /// Get total number of tasks processed
    pub fn get_total_tasks_processed(&self) -> usize {
        self.historical_performance.len()
    }

    fn update_prediction_model(
        &mut self,
        task_type: FederatedTaskType,
        performance: &TaskPerformance,
    ) {
        // Ensure model exists
        self.prediction_models.entry(task_type).or_insert_with(|| {
            PredictionModel {
                model_type: ModelType::Linear,
                parameters: vec![1.0, 0.0], // slope, intercept
                accuracy: 0.5,
                last_updated: Instant::now(),
            }
        });

        // Create dummy task for prediction
        let dummy_task = FederatedTask {
            task_id: "dummy".to_string(),
            task_type,
            priority: TaskPriority::Normal,
            estimated_size_mb: performance.network_conditions.bandwidth_mbps as usize,
            network_requirements: Default::default(),
            scheduled_time: Instant::now(),
            deadline: Instant::now(),
            retry_count: 0,
            status: super::types::TaskStatus::Pending,
        };

        // Get prediction (this borrows self immutably)
        let predicted_time = if let Some(model) = self.prediction_models.get(&task_type) {
            self.apply_prediction_model(model, &dummy_task, &performance.network_conditions)
        } else {
            performance.execution_time_ms as f32 // fallback
        };

        // Update model parameters (now we can get mutable access)
        if let Some(model) = self.prediction_models.get_mut(&task_type) {
            let error = (predicted_time - performance.execution_time_ms as f32).abs();
            let learning_rate = 0.01;
            model.parameters[0] *= 1.0 - learning_rate * error / predicted_time.max(1.0);
            model.last_updated = Instant::now();
        }
    }

    fn apply_prediction_model(
        &self,
        model: &PredictionModel,
        task: &FederatedTask,
        network_conditions: &NetworkConditions,
    ) -> f32 {
        match model.model_type {
            ModelType::Linear => {
                let size_factor = task.estimated_size_mb as f32;
                let bandwidth_factor = 1.0 / (network_conditions.bandwidth_mbps + 0.1);
                model.parameters[0] * size_factor * bandwidth_factor + model.parameters[1]
            },
            _ => {
                // Fallback to simple calculation
                (task.estimated_size_mb as f32 / network_conditions.bandwidth_mbps) * 8.0 * 1000.0
            },
        }
    }
}

impl AccuracyTracker {
    /// Create new accuracy tracker
    pub fn new() -> Self {
        Self {
            prediction_errors: VecDeque::new(),
            mean_absolute_error: 0.0,
            root_mean_square_error: 0.0,
            confidence_intervals: HashMap::new(),
        }
    }

    /// Record prediction error
    pub fn record_error(&mut self, error: f32) {
        self.prediction_errors.push_back(error);

        if self.prediction_errors.len() > 100 {
            self.prediction_errors.pop_front();
        }

        self.update_metrics();
    }

    /// Get current mean absolute error
    pub fn get_mean_absolute_error(&self) -> f32 {
        self.mean_absolute_error
    }

    /// Get current root mean square error
    pub fn get_root_mean_square_error(&self) -> f32 {
        self.root_mean_square_error
    }

    fn update_metrics(&mut self) {
        if self.prediction_errors.is_empty() {
            return;
        }

        // Calculate MAE
        let sum_abs_errors: f32 = self.prediction_errors.iter().map(|e| e.abs()).sum();
        self.mean_absolute_error = sum_abs_errors / self.prediction_errors.len() as f32;

        // Calculate RMSE
        let sum_squared_errors: f32 = self.prediction_errors.iter().map(|e| e * e).sum();
        self.root_mean_square_error =
            (sum_squared_errors / self.prediction_errors.len() as f32).sqrt();
    }
}

/// Scheduling statistics
#[derive(Debug, Clone)]
pub struct SchedulingStatistics {
    pub active_task_count: usize,
    pub scheduled_task_count: usize,
    pub total_tasks_processed: usize,
    pub average_completion_time: f32,
}

impl Default for TaskPrioritizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for ScheduleOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for PerformancePredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AccuracyTracker {
    fn default() -> Self {
        Self::new()
    }
}
