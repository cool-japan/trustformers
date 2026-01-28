//! Background Task Management Types
//!
//! This module contains types for managing background tasks, execution context,
//! and resource requirements.

use crate::lifecycle::config::{SchedulingStrategy, TaskPriority, TaskType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Background task definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackgroundTask {
    /// Unique task identifier
    pub task_id: String,
    /// Task type
    pub task_type: TaskType,
    /// Task priority
    pub priority: TaskPriority,
    /// Scheduling strategy
    pub scheduling_strategy: SchedulingStrategy,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
    /// Execution constraints
    pub execution_constraints: ExecutionConstraints,
    /// Task metadata
    pub metadata: TaskMetadata,
}

/// Resource requirements for tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    /// Minimum CPU allocation (%)
    pub min_cpu_percent: u8,
    /// Maximum CPU allocation (%)
    pub max_cpu_percent: u8,
    /// Minimum memory allocation (MB)
    pub min_memory_mb: usize,
    /// Maximum memory allocation (MB)
    pub max_memory_mb: usize,
    /// Network bandwidth requirement (Mbps)
    pub network_bandwidth_mbps: Option<f32>,
    /// GPU requirement
    pub requires_gpu: bool,
    /// Storage I/O requirement (MB/s)
    pub storage_io_mbps: Option<f32>,
    /// Execution time estimate (seconds)
    pub estimated_execution_time_seconds: u64,
}

/// Task execution constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConstraints {
    /// Maximum execution time (seconds)
    pub max_execution_time_seconds: u64,
    /// Retry attempts on failure
    pub retry_attempts: u32,
    /// Retry delay (seconds)
    pub retry_delay_seconds: u64,
    /// Requires network connectivity
    pub requires_network: bool,
    /// Minimum battery level (%)
    pub min_battery_percent: u8,
    /// Maximum thermal level
    pub max_thermal_level: crate::lifecycle::config::ThermalLevel,
    /// Can run in background
    pub background_eligible: bool,
    /// User presence required
    pub requires_user_presence: bool,
}

/// Task metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetadata {
    /// Task name
    pub name: String,
    /// Task description
    pub description: String,
    /// Creation timestamp
    pub created_timestamp: u64,
    /// Scheduled execution timestamp
    pub scheduled_timestamp: Option<u64>,
    /// Last execution timestamp
    pub last_execution_timestamp: Option<u64>,
    /// Execution count
    pub execution_count: usize,
    /// Success count
    pub success_count: usize,
    /// Failure count
    pub failure_count: usize,
    /// Tags for categorization
    pub tags: Vec<String>,
}

/// Task execution status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Task is pending execution
    Pending,
    /// Task is scheduled to run
    Scheduled,
    /// Task is currently running
    Running,
    /// Task is paused
    Paused,
    /// Task completed successfully
    Completed,
    /// Task failed
    Failed,
    /// Task was cancelled
    Cancelled,
    /// Task timed out
    TimedOut,
    /// Task was deferred
    Deferred,
    /// Task is waiting for resources
    WaitingForResources,
    /// Task is waiting for conditions
    WaitingForConditions,
}

/// Task execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    /// Task identifier
    pub task_id: String,
    /// Execution status
    pub status: TaskStatus,
    /// Start timestamp
    pub start_timestamp: u64,
    /// End timestamp
    pub end_timestamp: Option<u64>,
    /// Actual execution time (seconds)
    pub execution_time_seconds: f64,
    /// Resource usage during execution
    pub resource_usage: TaskResourceUsage,
    /// Output data (if any)
    pub output_data: Option<Vec<u8>>,
    /// Error information (if failed)
    pub error_info: Option<TaskError>,
    /// Performance metrics
    pub performance_metrics: TaskPerformanceMetrics,
    /// Quality metrics
    pub quality_metrics: Option<TaskQualityMetrics>,
}

/// Task resource usage during execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResourceUsage {
    /// Peak CPU usage (%)
    pub peak_cpu_percent: f32,
    /// Average CPU usage (%)
    pub avg_cpu_percent: f32,
    /// Peak memory usage (MB)
    pub peak_memory_mb: usize,
    /// Average memory usage (MB)
    pub avg_memory_mb: usize,
    /// Network data transferred (MB)
    pub network_data_mb: f32,
    /// Storage I/O performed (MB)
    pub storage_io_mb: f32,
    /// GPU usage (%)
    pub gpu_usage_percent: Option<f32>,
    /// Battery consumption (mAh)
    pub battery_consumption_mah: f32,
}

/// Task error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskError {
    /// Error code
    pub error_code: u32,
    /// Error message
    pub error_message: String,
    /// Error category
    pub error_category: TaskErrorCategory,
    /// Recoverable error flag
    pub recoverable: bool,
    /// Retry recommended flag
    pub retry_recommended: bool,
    /// Additional error details
    pub details: HashMap<String, String>,
}

/// Task error categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskErrorCategory {
    ResourceUnavailable,
    NetworkError,
    AuthenticationError,
    PermissionError,
    DataError,
    SystemError,
    TimeoutError,
    UserCancellation,
    InternalError,
    ConfigurationError,
}

/// Task performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskPerformanceMetrics {
    /// Throughput (operations/second)
    pub throughput_ops_per_second: f32,
    /// Latency percentiles (ms)
    pub latency_percentiles: LatencyPercentiles,
    /// Error rate (%)
    pub error_rate_percent: f32,
    /// Resource efficiency score (0-100)
    pub resource_efficiency_score: f32,
    /// Completion rate (%)
    pub completion_rate_percent: f32,
}

/// Latency percentile measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyPercentiles {
    /// 50th percentile (median) in ms
    pub p50_ms: f32,
    /// 90th percentile in ms
    pub p90_ms: f32,
    /// 95th percentile in ms
    pub p95_ms: f32,
    /// 99th percentile in ms
    pub p99_ms: f32,
    /// Maximum latency in ms
    pub max_ms: f32,
}

/// Task quality metrics (for ML tasks)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskQualityMetrics {
    /// Accuracy score (0-100)
    pub accuracy_score: f32,
    /// Precision score (0-100)
    pub precision_score: f32,
    /// Recall score (0-100)
    pub recall_score: f32,
    /// F1 score (0-100)
    pub f1_score: f32,
    /// Model drift score (0-100)
    pub model_drift_score: Option<f32>,
    /// Data quality score (0-100)
    pub data_quality_score: Option<f32>,
}

/// Background task coordinator
pub struct BackgroundCoordinator {
    task_queue: Arc<std::sync::Mutex<Vec<BackgroundTask>>>,
    running_tasks: Arc<std::sync::Mutex<HashMap<String, TaskExecutionContext>>>,
    completed_tasks: Arc<std::sync::Mutex<Vec<TaskResult>>>,
    execution_context: BackgroundExecutionContext,
    task_registry: TaskRegistry,
    max_concurrent_tasks: usize,
}

/// Task execution context
pub struct TaskExecutionContext {
    pub task: BackgroundTask,
    pub start_time: Instant,
    pub allocated_resources: AllocatedResources,
    pub status: TaskStatus,
    pub progress: f32, // 0.0 to 1.0
}

/// Allocated resources for task execution
#[derive(Debug, Clone)]
pub struct AllocatedResources {
    pub cpu_percent: u8,
    pub memory_mb: usize,
    pub network_mbps: Option<f32>,
    pub gpu_allocation: Option<f32>,
    pub storage_io_mbps: Option<f32>,
}

/// Background execution context
pub struct BackgroundExecutionContext {
    pub available_cpu_percent: u8,
    pub available_memory_mb: usize,
    pub available_network_mbps: f32,
    pub battery_level_percent: u8,
    pub thermal_level: crate::lifecycle::config::ThermalLevel,
    pub network_connected: bool,
    pub user_present: bool,
    pub system_load: f32,
}

/// Task registry for managing task definitions
pub struct TaskRegistry {
    registered_tasks: HashMap<TaskType, Vec<BackgroundTask>>,
    task_templates: HashMap<TaskType, BackgroundTask>,
}

impl BackgroundCoordinator {
    /// Create new background coordinator
    pub fn new(max_concurrent_tasks: usize) -> Self {
        Self {
            task_queue: Arc::new(std::sync::Mutex::new(Vec::new())),
            running_tasks: Arc::new(std::sync::Mutex::new(HashMap::new())),
            completed_tasks: Arc::new(std::sync::Mutex::new(Vec::new())),
            execution_context: BackgroundExecutionContext::default(),
            task_registry: TaskRegistry::new(),
            max_concurrent_tasks,
        }
    }

    /// Schedule a background task
    pub fn schedule_task(&self, task: BackgroundTask) -> Result<(), Box<dyn std::error::Error>> {
        let mut queue = self.task_queue.lock().expect("Operation failed");
        queue.push(task);

        // Sort by priority and scheduling strategy
        queue.sort_by(|a, b| {
            b.priority.cmp(&a.priority).then_with(|| {
                a.scheduling_strategy
                    .priority_order()
                    .cmp(&b.scheduling_strategy.priority_order())
            })
        });

        Ok(())
    }

    /// Execute next available task
    pub fn execute_next_task(&mut self) -> Result<Option<TaskResult>, Box<dyn std::error::Error>> {
        let running_count = self.running_tasks.lock().expect("Operation failed").len();
        if running_count >= self.max_concurrent_tasks {
            return Ok(None);
        }

        let task = {
            let mut queue = self.task_queue.lock().expect("Operation failed");
            if queue.is_empty() {
                return Ok(None);
            }

            // Find first executable task based on constraints
            let mut task_index = None;
            for (index, task) in queue.iter().enumerate() {
                if self.can_execute_task(task) {
                    task_index = Some(index);
                    break;
                }
            }

            match task_index {
                Some(index) => queue.remove(index),
                None => return Ok(None),
            }
        };

        // Allocate resources and start execution
        let allocated_resources = self.allocate_resources(&task)?;
        let execution_context = TaskExecutionContext {
            task: task.clone(),
            start_time: Instant::now(),
            allocated_resources,
            status: TaskStatus::Running,
            progress: 0.0,
        };

        self.running_tasks
            .lock()
            .expect("Operation failed")
            .insert(task.task_id.clone(), execution_context);

        // Execute task (simplified - in real implementation this would be async)
        let result = self.execute_task_impl(&task)?;

        // Remove from running tasks and add to completed
        self.running_tasks.lock().expect("Operation failed").remove(&task.task_id);
        self.completed_tasks.lock().expect("Operation failed").push(result.clone());

        Ok(Some(result))
    }

    /// Check if task can be executed given current constraints
    fn can_execute_task(&self, task: &BackgroundTask) -> bool {
        // Check battery constraints
        if self.execution_context.battery_level_percent
            < task.execution_constraints.min_battery_percent
        {
            return false;
        }

        // Check thermal constraints
        if self.execution_context.thermal_level > task.execution_constraints.max_thermal_level {
            return false;
        }

        // Check network constraints
        if task.execution_constraints.requires_network && !self.execution_context.network_connected
        {
            return false;
        }

        // Check user presence constraints
        if task.execution_constraints.requires_user_presence && !self.execution_context.user_present
        {
            return false;
        }

        // Check resource availability
        if task.resource_requirements.min_cpu_percent > self.execution_context.available_cpu_percent
        {
            return false;
        }

        if task.resource_requirements.min_memory_mb > self.execution_context.available_memory_mb {
            return false;
        }

        true
    }

    /// Allocate resources for task execution
    fn allocate_resources(
        &self,
        task: &BackgroundTask,
    ) -> Result<AllocatedResources, Box<dyn std::error::Error>> {
        let cpu_percent = task
            .resource_requirements
            .min_cpu_percent
            .min(self.execution_context.available_cpu_percent);
        let memory_mb = task
            .resource_requirements
            .min_memory_mb
            .min(self.execution_context.available_memory_mb);

        Ok(AllocatedResources {
            cpu_percent,
            memory_mb,
            network_mbps: task.resource_requirements.network_bandwidth_mbps,
            gpu_allocation: if task.resource_requirements.requires_gpu { Some(50.0) } else { None },
            storage_io_mbps: task.resource_requirements.storage_io_mbps,
        })
    }

    /// Execute task implementation (simplified)
    fn execute_task_impl(
        &self,
        task: &BackgroundTask,
    ) -> Result<TaskResult, Box<dyn std::error::Error>> {
        let start_timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Operation failed")
            .as_secs();

        // Simulate task execution
        std::thread::sleep(Duration::from_millis(100));

        let end_timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("Operation failed")
            .as_secs();

        Ok(TaskResult {
            task_id: task.task_id.clone(),
            status: TaskStatus::Completed,
            start_timestamp,
            end_timestamp: Some(end_timestamp),
            execution_time_seconds: (end_timestamp - start_timestamp) as f64,
            resource_usage: TaskResourceUsage {
                peak_cpu_percent: 25.0,
                avg_cpu_percent: 20.0,
                peak_memory_mb: 100,
                avg_memory_mb: 80,
                network_data_mb: 1.0,
                storage_io_mb: 0.5,
                gpu_usage_percent: None,
                battery_consumption_mah: 5.0,
            },
            output_data: None,
            error_info: None,
            performance_metrics: TaskPerformanceMetrics {
                throughput_ops_per_second: 10.0,
                latency_percentiles: LatencyPercentiles {
                    p50_ms: 50.0,
                    p90_ms: 80.0,
                    p95_ms: 90.0,
                    p99_ms: 100.0,
                    max_ms: 120.0,
                },
                error_rate_percent: 0.0,
                resource_efficiency_score: 85.0,
                completion_rate_percent: 100.0,
            },
            quality_metrics: None,
        })
    }

    /// Get running tasks status
    pub fn get_running_tasks(&self) -> Vec<(String, TaskStatus, f32)> {
        self.running_tasks
            .lock()
            .expect("Operation failed")
            .iter()
            .map(|(id, context)| (id.clone(), context.status, context.progress))
            .collect()
    }

    /// Get completed tasks results
    pub fn get_completed_tasks(&self) -> Vec<TaskResult> {
        self.completed_tasks.lock().expect("Operation failed").clone()
    }
}

impl Default for BackgroundExecutionContext {
    fn default() -> Self {
        Self {
            available_cpu_percent: 50,
            available_memory_mb: 512,
            available_network_mbps: 10.0,
            battery_level_percent: 80,
            thermal_level: crate::lifecycle::config::ThermalLevel::Normal,
            network_connected: true,
            user_present: true,
            system_load: 0.5,
        }
    }
}

impl TaskRegistry {
    /// Create new task registry
    pub fn new() -> Self {
        Self {
            registered_tasks: HashMap::new(),
            task_templates: HashMap::new(),
        }
    }

    /// Register a task template
    pub fn register_task_template(&mut self, task_type: TaskType, template: BackgroundTask) {
        self.task_templates.insert(task_type, template);
    }

    /// Create task from template
    pub fn create_task_from_template(
        &self,
        task_type: TaskType,
        task_id: String,
    ) -> Option<BackgroundTask> {
        self.task_templates.get(&task_type).map(|template| {
            let mut task = template.clone();
            task.task_id = task_id;
            task
        })
    }

    /// Get all registered tasks by type
    pub fn get_tasks_by_type(&self, task_type: TaskType) -> Vec<&BackgroundTask> {
        self.registered_tasks
            .get(&task_type)
            .map(|tasks| tasks.iter().collect())
            .unwrap_or_default()
    }
}

impl Default for TaskRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl SchedulingStrategy {
    /// Get priority order for scheduling (lower number = higher priority)
    pub fn priority_order(&self) -> u8 {
        match self {
            SchedulingStrategy::Immediate => 0,
            SchedulingStrategy::NetworkOptimal => 1,
            SchedulingStrategy::BatteryOptimal => 2,
            SchedulingStrategy::ThermalOptimal => 3,
            SchedulingStrategy::UserIdle => 4,
            SchedulingStrategy::OpportunisticAgg => 5,
            SchedulingStrategy::Deferred => 6,
        }
    }
}

impl Default for ResourceRequirements {
    fn default() -> Self {
        Self {
            min_cpu_percent: 10,
            max_cpu_percent: 50,
            min_memory_mb: 50,
            max_memory_mb: 200,
            network_bandwidth_mbps: None,
            requires_gpu: false,
            storage_io_mbps: None,
            estimated_execution_time_seconds: 30,
        }
    }
}

impl Default for ExecutionConstraints {
    fn default() -> Self {
        Self {
            max_execution_time_seconds: 300, // 5 minutes
            retry_attempts: 3,
            retry_delay_seconds: 5,
            requires_network: false,
            min_battery_percent: 20,
            max_thermal_level: crate::lifecycle::config::ThermalLevel::Moderate,
            background_eligible: true,
            requires_user_presence: false,
        }
    }
}
