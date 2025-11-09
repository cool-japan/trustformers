// Allow dead code for infrastructure under development
#![allow(dead_code)]

//! Operator Scheduling Module
//!
//! Provides intelligent scheduling of computational operators for optimal resource
//! utilization, task prioritization, and performance optimization across different
//! execution contexts and device types.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, info, trace};

/// Operator scheduling service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorSchedulingConfig {
    /// Maximum number of concurrent operators per device
    pub max_concurrent_operators_per_device: usize,

    /// Enable priority-based scheduling
    pub enable_priority_scheduling: bool,

    /// Enable resource-aware scheduling
    pub enable_resource_aware_scheduling: bool,

    /// Enable dependency-aware scheduling
    pub enable_dependency_aware_scheduling: bool,

    /// Enable load balancing across devices
    pub enable_load_balancing: bool,

    /// Scheduling algorithm to use
    pub scheduling_algorithm: SchedulingAlgorithm,

    /// Time slice for round-robin scheduling (milliseconds)
    pub time_slice_ms: u64,

    /// Memory threshold for scheduling decisions (percentage)
    pub memory_threshold_percent: f64,

    /// CPU utilization threshold for scheduling decisions (percentage)
    pub cpu_threshold_percent: f64,

    /// GPU utilization threshold for scheduling decisions (percentage)
    pub gpu_threshold_percent: f64,

    /// Enable preemption for higher priority tasks
    pub enable_preemption: bool,

    /// Maximum queue size per device
    pub max_queue_size_per_device: usize,

    /// Task timeout in milliseconds
    pub task_timeout_ms: u64,

    /// Enable performance profiling
    pub enable_performance_profiling: bool,
}

impl Default for OperatorSchedulingConfig {
    fn default() -> Self {
        Self {
            max_concurrent_operators_per_device: 4,
            enable_priority_scheduling: true,
            enable_resource_aware_scheduling: true,
            enable_dependency_aware_scheduling: true,
            enable_load_balancing: true,
            scheduling_algorithm: SchedulingAlgorithm::Adaptive,
            time_slice_ms: 100,
            memory_threshold_percent: 85.0,
            cpu_threshold_percent: 90.0,
            gpu_threshold_percent: 95.0,
            enable_preemption: true,
            max_queue_size_per_device: 1000,
            task_timeout_ms: 300000, // 5 minutes
            enable_performance_profiling: true,
        }
    }
}

/// Scheduling algorithms
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SchedulingAlgorithm {
    /// First-Come, First-Served
    FCFS,
    /// Shortest Job First
    SJF,
    /// Priority scheduling
    Priority,
    /// Round-robin scheduling
    RoundRobin,
    /// Earliest Deadline First
    EDF,
    /// Resource-aware scheduling
    ResourceAware,
    /// Adaptive scheduling (combines multiple strategies)
    Adaptive,
    /// Custom scheduling logic
    Custom,
}

/// Task priority levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TaskPriority {
    /// Critical system tasks
    Critical = 4,
    /// High priority user tasks
    High = 3,
    /// Normal priority tasks
    Normal = 2,
    /// Low priority background tasks
    Low = 1,
    /// Best-effort tasks
    BestEffort = 0,
}

impl Default for TaskPriority {
    fn default() -> Self {
        TaskPriority::Normal
    }
}

/// Device types for scheduling
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// CPU device
    CPU,
    /// CUDA GPU device
    CUDA(usize),
    /// Metal GPU device
    Metal,
    /// OpenCL device
    OpenCL(usize),
    /// TPU device
    TPU(usize),
}

/// Operator task representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorTask {
    /// Unique task identifier
    pub id: String,

    /// Task name/description
    pub name: String,

    /// Operation type
    pub operation_type: OperationType,

    /// Task priority
    pub priority: TaskPriority,

    /// Estimated execution time in milliseconds
    pub estimated_duration_ms: Option<u64>,

    /// Memory requirements in bytes
    pub memory_requirements: Option<usize>,

    /// CPU requirements (cores)
    pub cpu_requirements: Option<f64>,

    /// GPU requirements (percentage of device)
    pub gpu_requirements: Option<f64>,

    /// Preferred device type
    pub preferred_device: Option<DeviceType>,

    /// Task dependencies (task IDs)
    pub dependencies: Vec<String>,

    /// Deadline for execution
    pub deadline: Option<SystemTime>,

    /// Task creation time
    pub created_at: SystemTime,

    /// Task metadata
    pub metadata: HashMap<String, String>,

    /// Task affinity (device preferences)
    pub device_affinity: HashMap<DeviceType, f64>,
}

/// Operation types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OperationType {
    /// Matrix multiplication
    MatMul,
    /// Convolution
    Convolution,
    /// Activation functions
    Activation(String),
    /// Normalization
    Normalization(String),
    /// Pooling
    Pooling(String),
    /// Attention computation
    Attention,
    /// Embedding lookup
    Embedding,
    /// Reduction operations
    Reduction(String),
    /// Element-wise operations
    ElementWise(String),
    /// Memory operations
    Memory(String),
    /// Custom operation
    Custom(String),
}

/// Task execution state
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum TaskState {
    /// Task is queued for execution
    Queued,
    /// Task is currently running
    Running,
    /// Task completed successfully
    Completed,
    /// Task failed with error
    Failed,
    /// Task was cancelled
    Cancelled,
    /// Task timed out
    TimedOut,
    /// Task is waiting for dependencies
    WaitingForDependencies,
    /// Task is preempted
    Preempted,
}

/// Device resource information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceResource {
    /// Device type and identifier
    pub device: DeviceType,

    /// Current memory usage in bytes
    pub memory_used: usize,

    /// Total memory available in bytes
    pub memory_total: usize,

    /// Current CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f64,

    /// Current GPU utilization (0.0 to 1.0)
    pub gpu_utilization: f64,

    /// Number of currently running tasks
    pub active_tasks: usize,

    /// Number of queued tasks
    pub queued_tasks: usize,

    /// Device availability
    pub is_available: bool,

    /// Last update timestamp
    pub last_updated: SystemTime,
}

/// Task execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskExecutionResult {
    /// Task identifier
    pub task_id: String,

    /// Final task state
    pub state: TaskState,

    /// Actual execution time
    pub execution_time: Duration,

    /// Device used for execution
    pub executed_on: DeviceType,

    /// Memory peak usage during execution
    pub peak_memory_usage: Option<usize>,

    /// Error message if failed
    pub error_message: Option<String>,

    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,

    /// Task completion timestamp
    pub completed_at: SystemTime,
}

/// Scheduling statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingStats {
    /// Total tasks scheduled
    pub total_tasks_scheduled: u64,

    /// Tasks completed successfully
    pub tasks_completed: u64,

    /// Tasks failed
    pub tasks_failed: u64,

    /// Tasks cancelled/timed out
    pub tasks_cancelled: u64,

    /// Average task execution time
    pub average_execution_time: Duration,

    /// Average queue wait time
    pub average_queue_time: Duration,

    /// Current queue size across all devices
    pub current_queue_size: usize,

    /// Device utilization statistics
    pub device_utilization: HashMap<DeviceType, f64>,

    /// Scheduling algorithm performance
    pub algorithm_metrics: HashMap<String, f64>,

    /// Throughput (tasks per second)
    pub throughput: f64,

    /// Task priority distribution
    pub priority_distribution: HashMap<TaskPriority, u64>,
}

/// Scheduling decision information
#[derive(Debug, Clone)]
pub struct SchedulingDecision {
    /// Selected task to execute
    pub task: OperatorTask,

    /// Selected device for execution
    pub device: DeviceType,

    /// Scheduling score/priority
    pub score: f64,

    /// Reasoning for the decision
    pub reasoning: String,

    /// Expected start time
    pub expected_start_time: SystemTime,

    /// Alternative devices considered
    pub alternatives: Vec<(DeviceType, f64)>,
}

/// Operator scheduling errors
#[derive(Debug, Error)]
pub enum OperatorSchedulingError {
    #[error("No suitable device found for task {0}")]
    NoSuitableDevice(String),

    #[error("Task queue full for device {0:?}")]
    QueueFull(DeviceType),

    #[error("Task {0} timed out")]
    TaskTimeout(String),

    #[error("Dependency cycle detected in task {0}")]
    DependencyCycle(String),

    #[error("Invalid task configuration: {0}")]
    InvalidTask(String),

    #[error("Device {0:?} is not available")]
    DeviceUnavailable(DeviceType),

    #[error("Resource constraints violated: {0}")]
    ResourceConstraints(String),

    #[error("Internal scheduling error: {0}")]
    Internal(#[from] anyhow::Error),
}

/// Task wrapper for priority queue
#[derive(Debug, Clone)]
struct PriorityTask {
    task: OperatorTask,
    priority_score: f64,
    queue_time: SystemTime,
}

impl PartialEq for PriorityTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority_score == other.priority_score
    }
}

impl Eq for PriorityTask {}

impl PartialOrd for PriorityTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority score comes first
        other
            .priority_score
            .partial_cmp(&self.priority_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.queue_time.cmp(&other.queue_time))
    }
}

/// Operator scheduling service
pub struct OperatorSchedulingService {
    config: OperatorSchedulingConfig,
    task_queues: Arc<RwLock<HashMap<DeviceType, BinaryHeap<PriorityTask>>>>,
    running_tasks: Arc<RwLock<HashMap<String, (OperatorTask, DeviceType, SystemTime)>>>,
    device_resources: Arc<RwLock<HashMap<DeviceType, DeviceResource>>>,
    stats: Arc<RwLock<SchedulingStats>>,
    task_results: Arc<RwLock<HashMap<String, TaskExecutionResult>>>,
    dependency_graph: Arc<RwLock<HashMap<String, HashSet<String>>>>,
    scheduling_history: Arc<RwLock<Vec<SchedulingDecision>>>,
}

impl OperatorSchedulingService {
    /// Create a new operator scheduling service
    pub fn new(config: OperatorSchedulingConfig) -> Self {
        Self {
            config,
            task_queues: Arc::new(RwLock::new(HashMap::new())),
            running_tasks: Arc::new(RwLock::new(HashMap::new())),
            device_resources: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(SchedulingStats {
                total_tasks_scheduled: 0,
                tasks_completed: 0,
                tasks_failed: 0,
                tasks_cancelled: 0,
                average_execution_time: Duration::from_secs(0),
                average_queue_time: Duration::from_secs(0),
                current_queue_size: 0,
                device_utilization: HashMap::new(),
                algorithm_metrics: HashMap::new(),
                throughput: 0.0,
                priority_distribution: HashMap::new(),
            })),
            task_results: Arc::new(RwLock::new(HashMap::new())),
            dependency_graph: Arc::new(RwLock::new(HashMap::new())),
            scheduling_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a device for scheduling
    pub async fn register_device(
        &self,
        device: DeviceType,
        resource: DeviceResource,
    ) -> Result<()> {
        let mut devices = self.device_resources.write().await;
        let mut queues = self.task_queues.write().await;

        devices.insert(device, resource);
        queues.insert(device, BinaryHeap::new());

        info!("Registered device {:?} for operator scheduling", device);
        Ok(())
    }

    /// Submit a task for scheduling
    pub async fn submit_task(&self, task: OperatorTask) -> Result<(), OperatorSchedulingError> {
        info!("Submitting task {} for scheduling", task.id);

        // Validate task
        self.validate_task(&task).await?;

        // Check for dependency cycles
        if self.config.enable_dependency_aware_scheduling {
            self.check_dependency_cycles(&task).await?;
        }

        // Update dependency graph
        self.update_dependency_graph(&task).await;

        // Select device and schedule task
        let decision = self.make_scheduling_decision(&task).await?;

        // Add task to appropriate queue
        let mut queues = self.task_queues.write().await;
        if let Some(queue) = queues.get_mut(&decision.device) {
            if queue.len() >= self.config.max_queue_size_per_device {
                return Err(OperatorSchedulingError::QueueFull(decision.device));
            }

            let priority_task = PriorityTask {
                task: task.clone(),
                priority_score: decision.score,
                queue_time: SystemTime::now(),
            };

            queue.push(priority_task);
        } else {
            return Err(OperatorSchedulingError::DeviceUnavailable(decision.device));
        }

        // Update statistics
        self.update_submission_stats(&task).await;

        // Store scheduling decision and capture device for later use
        let device = decision.device;
        self.scheduling_history.write().await.push(decision);

        // Trigger scheduling if possible
        self.try_schedule_next_task(device).await?;

        debug!("Task {} queued for device {:?}", task.id, device);
        Ok(())
    }

    /// Try to schedule the next task for a device
    async fn try_schedule_next_task(
        &self,
        device: DeviceType,
    ) -> Result<(), OperatorSchedulingError> {
        // Check device availability first
        let device_available = {
            let devices = self.device_resources.read().await;
            if let Some(device_resource) = devices.get(&device) {
                device_resource.is_available
                    && device_resource.active_tasks
                        < self.config.max_concurrent_operators_per_device
            } else {
                return Err(OperatorSchedulingError::DeviceUnavailable(device));
            }
        };

        if !device_available {
            return Ok(());
        }

        // Get next task from queue
        let task_to_schedule = {
            let mut queues = self.task_queues.write().await;
            if let Some(queue) = queues.get_mut(&device) {
                // Try to find a task whose dependencies are satisfied
                let mut checked_tasks = Vec::new();
                let mut found_task = None;

                while let Some(priority_task) = queue.pop() {
                    let task = priority_task.task.clone();

                    // Check dependencies without holding locks
                    if self.config.enable_dependency_aware_scheduling {
                        // Check if task dependencies are satisfied
                        let dependencies_satisfied = self.are_dependencies_satisfied(&task).await;

                        if dependencies_satisfied {
                            found_task = Some(task);
                            break;
                        } else {
                            // Dependencies not satisfied, check next task
                            checked_tasks.push(priority_task);
                        }
                    } else {
                        // When dependency checking is disabled, take the first available task
                        found_task = Some(task);
                        break;
                    }
                }

                // Put back unchecked tasks
                for task in checked_tasks {
                    queue.push(task);
                }

                found_task
            } else {
                None
            }
        };

        if let Some(task) = task_to_schedule {
            // Start task execution
            let start_time = SystemTime::now();
            {
                let mut running = self.running_tasks.write().await;
                running.insert(task.id.clone(), (task.clone(), device, start_time));
            }

            info!(
                "Starting execution of task {} on device {:?}",
                task.id, device
            );

            // In a real implementation, this would trigger actual task execution
            // For now, we'll simulate it directly without spawning
            let task_id = task.id.clone();
            // Call simulate_task_execution directly to avoid Send issues
            tokio::task::spawn({
                let scheduler = self.clone();
                async move {
                    // Use hash-based randomness (Send-safe)
                    use std::collections::hash_map::DefaultHasher;
                    use std::hash::{Hash, Hasher};
                    let mut hasher = DefaultHasher::new();
                    task_id.hash(&mut hasher);
                    let hash_value = hasher.finish();

                    // Simple simulation without complex async operations
                    let execution_time = Duration::from_millis(100 + (hash_value % 1000));
                    tokio::time::sleep(execution_time).await;

                    // Complete the task with basic result
                    let result = TaskExecutionResult {
                        task_id: task_id.clone(),
                        state: TaskState::Completed,
                        execution_time,
                        executed_on: device,
                        peak_memory_usage: Some(1024 * 1024),
                        error_message: None,
                        performance_metrics: HashMap::new(),
                        completed_at: SystemTime::now(),
                    };

                    // Update task results directly without complex operations
                    {
                        let mut results = scheduler.task_results.write().await;
                        results.insert(task_id.clone(), result.clone());
                    }

                    // Update basic stats
                    {
                        let mut stats = scheduler.stats.write().await;
                        stats.tasks_completed += 1;
                    }
                }
            });
        }

        Ok(())
    }

    /// Make scheduling decision for a task
    async fn make_scheduling_decision(
        &self,
        task: &OperatorTask,
    ) -> Result<SchedulingDecision, OperatorSchedulingError> {
        let devices = self.device_resources.read().await;
        let mut candidates = Vec::new();

        // Evaluate all available devices
        for (device_type, resource) in devices.iter() {
            if !resource.is_available {
                continue;
            }

            let score = self.calculate_device_score(task, device_type, resource).await;
            candidates.push((*device_type, score));
        }

        if candidates.is_empty() {
            return Err(OperatorSchedulingError::NoSuitableDevice(task.id.clone()));
        }

        // Sort by score (highest first)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let (selected_device, score) = candidates[0];
        let alternatives = candidates[1..].to_vec();

        Ok(SchedulingDecision {
            task: task.clone(),
            device: selected_device,
            score,
            reasoning: self.generate_decision_reasoning(task, selected_device, score).await,
            expected_start_time: SystemTime::now(),
            alternatives,
        })
    }

    /// Calculate scoring for device assignment
    async fn calculate_device_score(
        &self,
        task: &OperatorTask,
        device: &DeviceType,
        resource: &DeviceResource,
    ) -> f64 {
        let mut score = 0.0;

        // Base score from device affinity
        if let Some(affinity) = task.device_affinity.get(device) {
            score += affinity * 40.0;
        }

        // Preferred device bonus
        if let Some(preferred) = &task.preferred_device {
            if preferred == device {
                score += 30.0;
            }
        }

        // Resource availability scoring
        let memory_availability =
            1.0 - (resource.memory_used as f64 / resource.memory_total as f64);
        score += memory_availability * 20.0;

        let cpu_availability = 1.0 - resource.cpu_utilization;
        score += cpu_availability * 15.0;

        let gpu_availability = 1.0 - resource.gpu_utilization;
        score += gpu_availability * 25.0;

        // Queue length penalty
        let queue_penalty = resource.queued_tasks as f64 * 2.0;
        score -= queue_penalty;

        // Active tasks penalty
        let active_penalty = resource.active_tasks as f64 * 3.0;
        score -= active_penalty;

        // Operation type specific scoring
        score += self.calculate_operation_affinity_score(&task.operation_type, device);

        score.max(0.0)
    }

    /// Calculate operation affinity score for device
    fn calculate_operation_affinity_score(
        &self,
        operation: &OperationType,
        device: &DeviceType,
    ) -> f64 {
        match (operation, device) {
            (OperationType::MatMul, DeviceType::CUDA(_)) => 15.0,
            (OperationType::MatMul, DeviceType::Metal) => 12.0,
            (OperationType::Convolution, DeviceType::CUDA(_)) => 20.0,
            (OperationType::Convolution, DeviceType::Metal) => 15.0,
            (OperationType::Attention, DeviceType::CUDA(_)) => 18.0,
            (OperationType::Attention, DeviceType::Metal) => 14.0,
            (OperationType::Memory(_), DeviceType::CPU) => 10.0,
            (OperationType::ElementWise(_), DeviceType::CUDA(_)) => 12.0,
            (OperationType::ElementWise(_), DeviceType::Metal) => 10.0,
            _ => 5.0,
        }
    }

    /// Generate reasoning for scheduling decision
    async fn generate_decision_reasoning(
        &self,
        task: &OperatorTask,
        device: DeviceType,
        score: f64,
    ) -> String {
        format!(
            "Selected device {:?} for task {} (priority: {:?}) with score {:.2}. \
             Operation type: {:?}, Device affinity: {:?}",
            device,
            task.name,
            task.priority,
            score,
            task.operation_type,
            task.device_affinity.get(&device).unwrap_or(&0.0)
        )
    }

    /// Validate task before scheduling
    async fn validate_task(&self, task: &OperatorTask) -> Result<(), OperatorSchedulingError> {
        if task.id.is_empty() {
            return Err(OperatorSchedulingError::InvalidTask(
                "Task ID cannot be empty".to_string(),
            ));
        }

        if task.name.is_empty() {
            return Err(OperatorSchedulingError::InvalidTask(
                "Task name cannot be empty".to_string(),
            ));
        }

        // Check if task already exists
        let running = self.running_tasks.read().await;
        if running.contains_key(&task.id) {
            return Err(OperatorSchedulingError::InvalidTask(format!(
                "Task {} is already running",
                task.id
            )));
        }

        Ok(())
    }

    /// Check for dependency cycles
    async fn check_dependency_cycles(
        &self,
        task: &OperatorTask,
    ) -> Result<(), OperatorSchedulingError> {
        let dep_graph = self.dependency_graph.read().await;
        let mut visited = HashSet::new();
        let mut rec_stack = HashSet::new();

        for dep in &task.dependencies {
            if self.has_cycle_util(&dep_graph, dep, &mut visited, &mut rec_stack) {
                return Err(OperatorSchedulingError::DependencyCycle(task.id.clone()));
            }
        }

        Ok(())
    }

    /// Utility function for cycle detection
    fn has_cycle_util(
        &self,
        graph: &HashMap<String, HashSet<String>>,
        node: &str,
        visited: &mut HashSet<String>,
        rec_stack: &mut HashSet<String>,
    ) -> bool {
        visited.insert(node.to_string());
        rec_stack.insert(node.to_string());

        if let Some(deps) = graph.get(node) {
            for dep in deps {
                if !visited.contains(dep) {
                    if self.has_cycle_util(graph, dep, visited, rec_stack) {
                        return true;
                    }
                } else if rec_stack.contains(dep) {
                    return true;
                }
            }
        }

        rec_stack.remove(node);
        false
    }

    /// Update dependency graph
    async fn update_dependency_graph(&self, task: &OperatorTask) {
        let mut dep_graph = self.dependency_graph.write().await;
        dep_graph.insert(task.id.clone(), task.dependencies.iter().cloned().collect());
    }

    /// Check if task dependencies are satisfied
    async fn are_dependencies_satisfied(&self, task: &OperatorTask) -> bool {
        let results = self.task_results.read().await;

        for dep_id in &task.dependencies {
            if let Some(result) = results.get(dep_id) {
                if result.state != TaskState::Completed {
                    return false;
                }
            } else {
                return false; // Dependency not found
            }
        }

        true
    }

    /// Simulate task execution (for testing/demo purposes)
    async fn simulate_task_execution(&self, task_id: String, device: DeviceType) {
        // Use hash-based randomness (Send-safe)
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        task_id.hash(&mut hasher);
        let hash_value = hasher.finish();
        let execution_time = Duration::from_millis(100 + (hash_value % 1000));
        tokio::time::sleep(execution_time).await;

        // Complete the task
        let result = TaskExecutionResult {
            task_id: task_id.clone(),
            state: TaskState::Completed,
            execution_time,
            executed_on: device,
            peak_memory_usage: Some(1024 * 1024), // 1MB placeholder
            error_message: None,
            performance_metrics: HashMap::new(),
            completed_at: SystemTime::now(),
        };

        self.complete_task(task_id, result).await;
    }

    /// Complete task execution
    async fn complete_task(&self, task_id: String, result: TaskExecutionResult) {
        // Remove from running tasks
        let mut running = self.running_tasks.write().await;
        running.remove(&task_id);

        // Store result
        let mut results = self.task_results.write().await;
        results.insert(task_id.clone(), result.clone());

        // Update statistics
        self.update_completion_stats(&result).await;

        info!("Task {} completed with state {:?}", task_id, result.state);

        // Try to schedule next task on the device
        let _ = self.try_schedule_next_task(result.executed_on).await;
    }

    /// Update submission statistics
    async fn update_submission_stats(&self, task: &OperatorTask) {
        let mut stats = self.stats.write().await;
        stats.total_tasks_scheduled += 1;
        *stats.priority_distribution.entry(task.priority).or_insert(0) += 1;

        // Update current queue size
        let queues = self.task_queues.read().await;
        stats.current_queue_size = queues.values().map(|q| q.len()).sum();
    }

    /// Update completion statistics
    async fn update_completion_stats(&self, result: &TaskExecutionResult) {
        let mut stats = self.stats.write().await;

        match result.state {
            TaskState::Completed => stats.tasks_completed += 1,
            TaskState::Failed => stats.tasks_failed += 1,
            TaskState::Cancelled | TaskState::TimedOut => stats.tasks_cancelled += 1,
            _ => {},
        }

        // Update average execution time
        let total_completed = stats.tasks_completed as f64;
        if total_completed > 0.0 {
            let current_avg = stats.average_execution_time.as_secs_f64();
            let new_avg = (current_avg * (total_completed - 1.0)
                + result.execution_time.as_secs_f64())
                / total_completed;
            stats.average_execution_time = Duration::from_secs_f64(new_avg);
        }

        // Calculate throughput
        let total_time = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        if total_time > 0.0 {
            stats.throughput = stats.tasks_completed as f64 / total_time;
        }
    }

    /// Get current scheduling statistics
    pub async fn get_stats(&self) -> SchedulingStats {
        self.stats.read().await.clone()
    }

    /// Get task result
    pub async fn get_task_result(&self, task_id: &str) -> Option<TaskExecutionResult> {
        self.task_results.read().await.get(task_id).cloned()
    }

    /// Cancel a task
    pub async fn cancel_task(&self, task_id: &str) -> Result<(), OperatorSchedulingError> {
        // Remove from queues
        let mut queues = self.task_queues.write().await;
        for queue in queues.values_mut() {
            queue.retain(|pt| pt.task.id != task_id);
        }

        // Remove from running tasks
        let mut running = self.running_tasks.write().await;
        if let Some((_task, device, _)) = running.remove(task_id) {
            let result = TaskExecutionResult {
                task_id: task_id.to_string(),
                state: TaskState::Cancelled,
                execution_time: Duration::from_secs(0),
                executed_on: device,
                peak_memory_usage: None,
                error_message: Some("Task cancelled".to_string()),
                performance_metrics: HashMap::new(),
                completed_at: SystemTime::now(),
            };

            self.task_results.write().await.insert(task_id.to_string(), result.clone());
            self.update_completion_stats(&result).await;
        }

        info!("Task {} cancelled", task_id);
        Ok(())
    }

    /// Update device resource information
    pub async fn update_device_resource(
        &self,
        device: DeviceType,
        resource: DeviceResource,
    ) -> Result<()> {
        let mut devices = self.device_resources.write().await;
        devices.insert(device, resource);

        trace!("Updated resource information for device {:?}", device);
        Ok(())
    }

    /// Get current device resources
    pub async fn get_device_resources(&self) -> HashMap<DeviceType, DeviceResource> {
        self.device_resources.read().await.clone()
    }

    /// Get scheduling history
    pub async fn get_scheduling_history(&self, limit: Option<usize>) -> Vec<SchedulingDecision> {
        let history = self.scheduling_history.read().await;
        match limit {
            Some(n) => history.iter().rev().take(n).cloned().collect(),
            None => history.clone(),
        }
    }
}

// Manual Clone implementation for OperatorSchedulingService
impl Clone for OperatorSchedulingService {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            task_queues: Arc::clone(&self.task_queues),
            running_tasks: Arc::clone(&self.running_tasks),
            device_resources: Arc::clone(&self.device_resources),
            stats: Arc::clone(&self.stats),
            task_results: Arc::clone(&self.task_results),
            dependency_graph: Arc::clone(&self.dependency_graph),
            scheduling_history: Arc::clone(&self.scheduling_history),
        }
    }
}

/// Summary statistics for the operator scheduling service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorSchedulingStatsSummary {
    /// Total tasks scheduled
    pub total_tasks_scheduled: u64,

    /// Success rate percentage
    pub success_rate_percent: f64,

    /// Average execution time in seconds
    pub average_execution_time_seconds: f64,

    /// Current throughput (tasks per second)
    pub throughput: f64,

    /// Current total queue size
    pub current_queue_size: usize,

    /// Number of active devices
    pub active_devices: usize,

    /// Most used priority level
    pub most_used_priority: Option<TaskPriority>,
}

impl OperatorSchedulingService {
    /// Get summary statistics
    pub async fn get_stats_summary(&self) -> OperatorSchedulingStatsSummary {
        let stats = self.stats.read().await;
        let devices = self.device_resources.read().await;

        let success_rate = if stats.total_tasks_scheduled > 0 {
            (stats.tasks_completed as f64 / stats.total_tasks_scheduled as f64) * 100.0
        } else {
            0.0
        };

        let most_used_priority = stats
            .priority_distribution
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(priority, _)| *priority);

        OperatorSchedulingStatsSummary {
            total_tasks_scheduled: stats.total_tasks_scheduled,
            success_rate_percent: success_rate,
            average_execution_time_seconds: stats.average_execution_time.as_secs_f64(),
            throughput: stats.throughput,
            current_queue_size: stats.current_queue_size,
            active_devices: devices.len(),
            most_used_priority,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    #[tokio::test]
    async fn test_operator_scheduling_service_creation() {
        let config = OperatorSchedulingConfig::default();
        let service = OperatorSchedulingService::new(config);
        let stats = service.get_stats().await;

        assert_eq!(stats.total_tasks_scheduled, 0);
        assert_eq!(stats.tasks_completed, 0);
    }

    #[tokio::test]
    async fn test_device_registration() {
        let service = OperatorSchedulingService::new(OperatorSchedulingConfig::default());

        let device = DeviceType::CPU;
        let resource = DeviceResource {
            device,
            memory_used: 0,
            memory_total: 1024 * 1024 * 1024, // 1GB
            cpu_utilization: 0.5,
            gpu_utilization: 0.0,
            active_tasks: 0,
            queued_tasks: 0,
            is_available: true,
            last_updated: SystemTime::now(),
        };

        let result = service.register_device(device, resource).await;
        assert!(result.is_ok());

        let devices = service.get_device_resources().await;
        assert!(devices.contains_key(&device));
    }

    #[tokio::test]
    async fn test_task_submission() {
        let service = OperatorSchedulingService::new(OperatorSchedulingConfig::default());

        // Register a device first
        let device = DeviceType::CPU;
        let resource = DeviceResource {
            device,
            memory_used: 0,
            memory_total: 1024 * 1024 * 1024,
            cpu_utilization: 0.3,
            gpu_utilization: 0.0,
            active_tasks: 0,
            queued_tasks: 0,
            is_available: true,
            last_updated: SystemTime::now(),
        };
        service.register_device(device, resource).await.unwrap();

        // Create and submit a task
        let task = OperatorTask {
            id: "test_task_1".to_string(),
            name: "Test MatMul".to_string(),
            operation_type: OperationType::MatMul,
            priority: TaskPriority::High,
            estimated_duration_ms: Some(100),
            memory_requirements: Some(1024 * 1024),
            cpu_requirements: Some(0.5),
            gpu_requirements: None,
            preferred_device: Some(DeviceType::CPU),
            dependencies: Vec::new(),
            deadline: None,
            created_at: SystemTime::now(),
            metadata: HashMap::new(),
            device_affinity: {
                let mut affinity = HashMap::new();
                affinity.insert(DeviceType::CPU, 0.8);
                affinity
            },
        };

        let result = service.submit_task(task).await;
        assert!(result.is_ok());

        let stats = service.get_stats().await;
        assert_eq!(stats.total_tasks_scheduled, 1);
    }

    #[tokio::test]
    async fn test_task_validation() {
        let service = OperatorSchedulingService::new(OperatorSchedulingConfig::default());

        // Test empty task ID
        let invalid_task = OperatorTask {
            id: "".to_string(),
            name: "Test Task".to_string(),
            operation_type: OperationType::MatMul,
            priority: TaskPriority::Normal,
            estimated_duration_ms: None,
            memory_requirements: None,
            cpu_requirements: None,
            gpu_requirements: None,
            preferred_device: None,
            dependencies: Vec::new(),
            deadline: None,
            created_at: SystemTime::now(),
            metadata: HashMap::new(),
            device_affinity: HashMap::new(),
        };

        let result = service.validate_task(&invalid_task).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_task_priority_ordering() {
        assert!(TaskPriority::Critical > TaskPriority::High);
        assert!(TaskPriority::High > TaskPriority::Normal);
        assert!(TaskPriority::Normal > TaskPriority::Low);
        assert!(TaskPriority::Low > TaskPriority::BestEffort);
    }

    #[tokio::test]
    async fn test_device_scoring() {
        let service = OperatorSchedulingService::new(OperatorSchedulingConfig::default());

        let task = OperatorTask {
            id: "test_task".to_string(),
            name: "Test Task".to_string(),
            operation_type: OperationType::MatMul,
            priority: TaskPriority::High,
            estimated_duration_ms: Some(100),
            memory_requirements: Some(1024),
            cpu_requirements: Some(0.5),
            gpu_requirements: Some(0.8),
            preferred_device: Some(DeviceType::CUDA(0)),
            dependencies: Vec::new(),
            deadline: None,
            created_at: SystemTime::now(),
            metadata: HashMap::new(),
            device_affinity: {
                let mut affinity = HashMap::new();
                affinity.insert(DeviceType::CUDA(0), 0.9);
                affinity.insert(DeviceType::CPU, 0.3);
                affinity
            },
        };

        let cpu_resource = DeviceResource {
            device: DeviceType::CPU,
            memory_used: 512 * 1024 * 1024,
            memory_total: 1024 * 1024 * 1024,
            cpu_utilization: 0.5,
            gpu_utilization: 0.0,
            active_tasks: 1,
            queued_tasks: 2,
            is_available: true,
            last_updated: SystemTime::now(),
        };

        let gpu_resource = DeviceResource {
            device: DeviceType::CUDA(0),
            memory_used: 256 * 1024 * 1024,
            memory_total: 2048 * 1024 * 1024,
            cpu_utilization: 0.2,
            gpu_utilization: 0.3,
            active_tasks: 0,
            queued_tasks: 1,
            is_available: true,
            last_updated: SystemTime::now(),
        };

        let cpu_score =
            service.calculate_device_score(&task, &DeviceType::CPU, &cpu_resource).await;
        let gpu_score =
            service.calculate_device_score(&task, &DeviceType::CUDA(0), &gpu_resource).await;

        // GPU should score higher for MatMul with GPU preference
        assert!(gpu_score > cpu_score);
    }
}
