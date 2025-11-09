//! GPU Memory Scheduling
//!
//! Provides intelligent GPU memory scheduling for optimized resource utilization
//! in multi-GPU environments and memory-constrained scenarios.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::{
    sync::{broadcast, mpsc, Mutex, RwLock, Semaphore},
    time::{sleep, timeout},
};

/// GPU memory scheduling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSchedulerConfig {
    /// Enable GPU memory scheduling
    pub enabled: bool,

    /// Maximum memory utilization percentage (0-100)
    pub max_memory_utilization: f32,

    /// Memory reservation buffer (in MB)
    pub memory_buffer_mb: usize,

    /// Scheduling algorithm
    pub scheduling_algorithm: SchedulingAlgorithm,

    /// Maximum queue size for pending tasks
    pub max_queue_size: usize,

    /// Task timeout in seconds
    pub task_timeout_seconds: u64,

    /// Memory monitoring interval in seconds
    pub memory_monitoring_interval_seconds: u64,

    /// GPU configurations
    pub gpu_configs: Vec<GpuConfig>,

    /// Enable preemption
    pub enable_preemption: bool,

    /// Preemption threshold (memory percentage)
    pub preemption_threshold: f32,
}

impl Default for GpuSchedulerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_memory_utilization: 90.0,
            memory_buffer_mb: 512,
            scheduling_algorithm: SchedulingAlgorithm::BestFit,
            max_queue_size: 1000,
            task_timeout_seconds: 300,
            memory_monitoring_interval_seconds: 5,
            gpu_configs: vec![GpuConfig {
                gpu_id: 0,
                total_memory_mb: 24000, // 24GB
                max_utilization: 90.0,
                priority: 1,
                enabled: true,
            }],
            enable_preemption: true,
            preemption_threshold: 95.0,
        }
    }
}

/// GPU configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// GPU ID
    pub gpu_id: usize,

    /// Total memory in MB
    pub total_memory_mb: usize,

    /// Maximum utilization percentage
    pub max_utilization: f32,

    /// GPU priority (lower is higher priority)
    pub priority: u32,

    /// Enable this GPU
    pub enabled: bool,
}

/// Scheduling algorithms
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum SchedulingAlgorithm {
    /// First-fit: Assign to first GPU with enough memory
    FirstFit,

    /// Best-fit: Assign to GPU with least wasted memory
    BestFit,

    /// Worst-fit: Assign to GPU with most free memory
    WorstFit,

    /// Round-robin: Cycle through GPUs
    RoundRobin,

    /// Priority-based: Consider GPU priority
    Priority,

    /// Load-balanced: Balance memory usage across GPUs
    LoadBalanced,
}

/// GPU memory task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuTask {
    /// Task ID
    pub task_id: String,

    /// Required memory in MB
    pub required_memory_mb: usize,

    /// Estimated execution time in seconds
    pub estimated_duration_seconds: u64,

    /// Task priority (lower is higher priority)
    pub priority: u32,

    /// Task type/category
    pub task_type: String,

    /// Client ID
    pub client_id: Option<String>,

    /// Task metadata
    pub metadata: HashMap<String, String>,

    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,

    /// Can be preempted
    pub preemptible: bool,
}

/// GPU memory status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuMemoryStatus {
    /// GPU ID
    pub gpu_id: usize,

    /// Total memory in MB
    pub total_memory_mb: usize,

    /// Used memory in MB
    pub used_memory_mb: usize,

    /// Available memory in MB
    pub available_memory_mb: usize,

    /// Memory utilization percentage
    pub utilization_percent: f32,

    /// Active tasks count
    pub active_tasks: usize,

    /// Is GPU available
    pub is_available: bool,

    /// Last updated timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Task execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskResult {
    /// Task ID
    pub task_id: String,

    /// Assigned GPU ID
    pub gpu_id: usize,

    /// Execution status
    pub status: TaskStatus,

    /// Actual memory used in MB
    pub actual_memory_mb: usize,

    /// Actual execution time in seconds
    pub actual_duration_seconds: u64,

    /// Start time
    pub started_at: chrono::DateTime<chrono::Utc>,

    /// End time
    pub completed_at: chrono::DateTime<chrono::Utc>,

    /// Error message if failed
    pub error_message: Option<String>,
}

/// Task execution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
    Preempted,
    TimedOut,
}

/// GPU scheduler statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuSchedulerStats {
    /// Total tasks processed
    pub total_tasks: u64,

    /// Tasks currently running
    pub running_tasks: u64,

    /// Tasks in queue
    pub queued_tasks: u64,

    /// Average wait time (seconds)
    pub avg_wait_time_seconds: f64,

    /// Average execution time (seconds)
    pub avg_execution_time_seconds: f64,

    /// GPU utilization statistics
    pub gpu_stats: Vec<GpuStats>,

    /// Memory efficiency (0-1)
    pub memory_efficiency: f64,

    /// Preemption count
    pub preemption_count: u64,

    /// Task failure rate
    pub failure_rate: f64,
}

/// GPU-specific statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuStats {
    /// GPU ID
    pub gpu_id: usize,

    /// Current memory status
    pub memory_status: GpuMemoryStatus,

    /// Tasks executed on this GPU
    pub tasks_executed: u64,

    /// Average task duration
    pub avg_task_duration_seconds: f64,

    /// Memory efficiency for this GPU
    pub memory_efficiency: f64,

    /// Uptime percentage
    pub uptime_percentage: f64,
}

/// GPU memory scheduler
pub struct GpuScheduler {
    config: GpuSchedulerConfig,

    /// GPU memory status
    gpu_status: Arc<RwLock<HashMap<usize, GpuMemoryStatus>>>,

    /// Task queue
    task_queue: Arc<Mutex<VecDeque<GpuTask>>>,

    /// Running tasks
    running_tasks: Arc<RwLock<HashMap<String, (GpuTask, usize)>>>, // task_id -> (task, gpu_id)

    /// Task results
    task_results: Arc<Mutex<HashMap<String, TaskResult>>>,

    /// Statistics
    stats: Arc<RwLock<GpuSchedulerStats>>,

    /// Event broadcaster
    event_sender: broadcast::Sender<GpuSchedulerEvent>,

    /// Task processing channels
    task_sender: mpsc::UnboundedSender<GpuTask>,

    /// Background task handles
    task_handles: Arc<Mutex<Vec<tokio::task::JoinHandle<()>>>>,

    /// GPU semaphores for resource management
    gpu_semaphores: Arc<RwLock<HashMap<usize, Arc<Semaphore>>>>,

    /// Round-robin counter
    round_robin_counter: Arc<Mutex<usize>>,
}

/// GPU scheduler events
#[derive(Debug, Clone, Serialize)]
pub enum GpuSchedulerEvent {
    /// Task queued
    TaskQueued {
        task_id: String,
        required_memory_mb: usize,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// Task started
    TaskStarted {
        task_id: String,
        gpu_id: usize,
        allocated_memory_mb: usize,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// Task completed
    TaskCompleted {
        task_id: String,
        gpu_id: usize,
        duration_seconds: u64,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// Task failed
    TaskFailed {
        task_id: String,
        gpu_id: Option<usize>,
        error: String,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// Task preempted
    TaskPreempted {
        task_id: String,
        gpu_id: usize,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// GPU memory updated
    GpuMemoryUpdated {
        gpu_id: usize,
        utilization_percent: f32,
        timestamp: chrono::DateTime<chrono::Utc>,
    },

    /// GPU overloaded
    GpuOverloaded {
        gpu_id: usize,
        utilization_percent: f32,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
}

impl GpuScheduler {
    /// Create a new GPU scheduler
    pub fn new(config: GpuSchedulerConfig) -> Self {
        let (event_sender, _) = broadcast::channel(1000);
        let (task_sender, task_receiver) = mpsc::unbounded_channel();

        let mut gpu_status = HashMap::new();
        let mut gpu_semaphores = HashMap::new();

        for gpu_config in &config.gpu_configs {
            if gpu_config.enabled {
                let status = GpuMemoryStatus {
                    gpu_id: gpu_config.gpu_id,
                    total_memory_mb: gpu_config.total_memory_mb,
                    used_memory_mb: 0,
                    available_memory_mb: gpu_config.total_memory_mb,
                    utilization_percent: 0.0,
                    active_tasks: 0,
                    is_available: true,
                    last_updated: chrono::Utc::now(),
                };

                gpu_status.insert(gpu_config.gpu_id, status);

                // Create semaphore for GPU (single permit for simplicity)
                gpu_semaphores.insert(gpu_config.gpu_id, Arc::new(Semaphore::new(1)));
            }
        }

        let scheduler = Self {
            config,
            gpu_status: Arc::new(RwLock::new(gpu_status)),
            task_queue: Arc::new(Mutex::new(VecDeque::new())),
            running_tasks: Arc::new(RwLock::new(HashMap::new())),
            task_results: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(RwLock::new(GpuSchedulerStats::default())),
            event_sender,
            task_sender,
            task_handles: Arc::new(Mutex::new(Vec::new())),
            gpu_semaphores: Arc::new(RwLock::new(gpu_semaphores)),
            round_robin_counter: Arc::new(Mutex::new(0)),
        };

        scheduler.start_background_tasks(task_receiver);

        scheduler
    }

    /// Start the GPU scheduler
    pub async fn start(&self) -> Result<()> {
        if !self.config.enabled {
            tracing::info!("GPU scheduler is disabled");
            return Ok(());
        }

        tracing::info!("Starting GPU memory scheduler");

        // Start memory monitoring
        self.start_memory_monitoring().await;

        Ok(())
    }

    /// Stop the GPU scheduler
    pub async fn stop(&self) -> Result<()> {
        let mut handles = self.task_handles.lock().await;
        for handle in handles.drain(..) {
            handle.abort();
        }

        tracing::info!("GPU scheduler stopped");
        Ok(())
    }

    /// Submit a task for GPU execution
    pub async fn submit_task(&self, task: GpuTask) -> Result<String> {
        if !self.config.enabled {
            return Err(anyhow::anyhow!("GPU scheduler is disabled"));
        }

        let task_id = task.task_id.clone();

        // Check queue capacity
        {
            let queue = self.task_queue.lock().await;
            if queue.len() >= self.config.max_queue_size {
                return Err(anyhow::anyhow!("Task queue is full"));
            }
        }

        // Send task queued event
        let _ = self.event_sender.send(GpuSchedulerEvent::TaskQueued {
            task_id: task_id.clone(),
            required_memory_mb: task.required_memory_mb,
            timestamp: chrono::Utc::now(),
        });

        // Send task for processing
        self.task_sender.send(task)?;

        Ok(task_id)
    }

    /// Get task status
    pub async fn get_task_status(&self, task_id: &str) -> Option<TaskStatus> {
        // Check if task is running
        {
            let running_tasks = self.running_tasks.read().await;
            if running_tasks.contains_key(task_id) {
                return Some(TaskStatus::Running);
            }
        }

        // Check if task is completed
        {
            let results = self.task_results.lock().await;
            if let Some(result) = results.get(task_id) {
                return Some(result.status.clone());
            }
        }

        // Check if task is in queue
        {
            let queue = self.task_queue.lock().await;
            if queue.iter().any(|task| task.task_id == task_id) {
                return Some(TaskStatus::Pending);
            }
        }

        None
    }

    /// Get task result
    pub async fn get_task_result(&self, task_id: &str) -> Option<TaskResult> {
        let results = self.task_results.lock().await;
        results.get(task_id).cloned()
    }

    /// Cancel a task
    pub async fn cancel_task(&self, task_id: &str) -> Result<bool> {
        // Try to remove from queue first
        {
            let mut queue = self.task_queue.lock().await;
            if let Some(pos) = queue.iter().position(|task| task.task_id == task_id) {
                queue.remove(pos);

                // Store cancelled result
                let result = TaskResult {
                    task_id: task_id.to_string(),
                    gpu_id: 0,
                    status: TaskStatus::Cancelled,
                    actual_memory_mb: 0,
                    actual_duration_seconds: 0,
                    started_at: chrono::Utc::now(),
                    completed_at: chrono::Utc::now(),
                    error_message: Some("Task cancelled".to_string()),
                };

                self.task_results.lock().await.insert(task_id.to_string(), result);

                return Ok(true);
            }
        }

        // Handle cancellation of running tasks
        {
            let mut running_tasks = self.running_tasks.write().await;
            if let Some((task, gpu_id)) = running_tasks.remove(task_id) {
                // Store cancelled result
                let result = TaskResult {
                    task_id: task_id.to_string(),
                    gpu_id,
                    status: TaskStatus::Cancelled,
                    actual_memory_mb: task.required_memory_mb,
                    actual_duration_seconds: 0,
                    started_at: task.created_at,
                    completed_at: chrono::Utc::now(),
                    error_message: Some("Task cancelled during execution".to_string()),
                };

                self.task_results.lock().await.insert(task_id.to_string(), result);

                // Update statistics
                {
                    let mut stats = self.stats.write().await;
                    stats.running_tasks = stats.running_tasks.saturating_sub(1);
                }

                // Free up GPU memory
                {
                    let mut gpu_status = self.gpu_status.write().await;
                    if let Some(status) = gpu_status.get_mut(&gpu_id) {
                        status.used_memory_mb =
                            status.used_memory_mb.saturating_sub(task.required_memory_mb);
                        status.available_memory_mb = status.total_memory_mb - status.used_memory_mb;
                    }
                }

                // Send cancellation event
                let _ = self.event_sender.send(GpuSchedulerEvent::TaskFailed {
                    task_id: task_id.to_string(),
                    gpu_id: Some(gpu_id),
                    error: "Task cancelled during execution".to_string(),
                    timestamp: chrono::Utc::now(),
                });

                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Get GPU memory status
    pub async fn get_gpu_status(&self) -> Vec<GpuMemoryStatus> {
        let status = self.gpu_status.read().await;
        status.values().cloned().collect()
    }

    /// Get scheduler statistics
    pub async fn get_stats(&self) -> GpuSchedulerStats {
        let stats = self.stats.read().await;
        stats.clone()
    }

    /// Subscribe to scheduler events
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<GpuSchedulerEvent> {
        self.event_sender.subscribe()
    }

    /// Start background tasks
    fn start_background_tasks(&self, mut task_receiver: mpsc::UnboundedReceiver<GpuTask>) {
        let scheduler = self.clone();

        // Task processing loop
        let handle = tokio::spawn(async move {
            while let Some(task) = task_receiver.recv().await {
                scheduler.process_task(task).await;
            }
        });

        let scheduler_clone = self.clone();
        tokio::spawn(async move {
            let mut handles = scheduler_clone.task_handles.lock().await;
            handles.push(handle);
        });
    }

    /// Process a single task
    async fn process_task(&self, task: GpuTask) {
        // Add to queue
        {
            let mut queue = self.task_queue.lock().await;
            queue.push_back(task.clone());
        }

        // Try to schedule immediately
        self.schedule_tasks().await;
    }

    /// Schedule tasks from queue
    async fn schedule_tasks(&self) {
        let mut scheduled_tasks = Vec::new();

        // Process queue
        {
            let mut queue = self.task_queue.lock().await;
            let mut remaining_tasks = VecDeque::new();

            while let Some(task) = queue.pop_front() {
                if let Some(gpu_id) = self.find_suitable_gpu(&task).await {
                    scheduled_tasks.push((task, gpu_id));
                } else {
                    remaining_tasks.push_back(task);
                }
            }

            *queue = remaining_tasks;
        }

        // Execute scheduled tasks
        for (task, gpu_id) in scheduled_tasks {
            self.execute_task(task, gpu_id).await;
        }
    }

    /// Find suitable GPU for a task
    async fn find_suitable_gpu(&self, task: &GpuTask) -> Option<usize> {
        let gpu_status = self.gpu_status.read().await;

        match self.config.scheduling_algorithm {
            SchedulingAlgorithm::FirstFit => self.find_first_fit_gpu(task, &gpu_status),
            SchedulingAlgorithm::BestFit => self.find_best_fit_gpu(task, &gpu_status),
            SchedulingAlgorithm::WorstFit => self.find_worst_fit_gpu(task, &gpu_status),
            SchedulingAlgorithm::RoundRobin => self.find_round_robin_gpu(task, &gpu_status).await,
            SchedulingAlgorithm::Priority => self.find_priority_gpu(task, &gpu_status),
            SchedulingAlgorithm::LoadBalanced => self.find_load_balanced_gpu(task, &gpu_status),
        }
    }

    /// Find first-fit GPU
    fn find_first_fit_gpu(
        &self,
        task: &GpuTask,
        gpu_status: &HashMap<usize, GpuMemoryStatus>,
    ) -> Option<usize> {
        for gpu_config in &self.config.gpu_configs {
            if let Some(status) = gpu_status.get(&gpu_config.gpu_id) {
                if status.is_available
                    && status.available_memory_mb
                        >= task.required_memory_mb + self.config.memory_buffer_mb
                {
                    return Some(gpu_config.gpu_id);
                }
            }
        }
        None
    }

    /// Find best-fit GPU
    fn find_best_fit_gpu(
        &self,
        task: &GpuTask,
        gpu_status: &HashMap<usize, GpuMemoryStatus>,
    ) -> Option<usize> {
        let mut best_gpu = None;
        let mut best_waste = usize::MAX;

        for gpu_config in &self.config.gpu_configs {
            if let Some(status) = gpu_status.get(&gpu_config.gpu_id) {
                if status.is_available
                    && status.available_memory_mb
                        >= task.required_memory_mb + self.config.memory_buffer_mb
                {
                    let waste = status.available_memory_mb - task.required_memory_mb;
                    if waste < best_waste {
                        best_waste = waste;
                        best_gpu = Some(gpu_config.gpu_id);
                    }
                }
            }
        }

        best_gpu
    }

    /// Find worst-fit GPU
    fn find_worst_fit_gpu(
        &self,
        task: &GpuTask,
        gpu_status: &HashMap<usize, GpuMemoryStatus>,
    ) -> Option<usize> {
        let mut best_gpu = None;
        let mut best_free_memory = 0;

        for gpu_config in &self.config.gpu_configs {
            if let Some(status) = gpu_status.get(&gpu_config.gpu_id) {
                if status.is_available
                    && status.available_memory_mb
                        >= task.required_memory_mb + self.config.memory_buffer_mb
                    && status.available_memory_mb > best_free_memory
                {
                    best_free_memory = status.available_memory_mb;
                    best_gpu = Some(gpu_config.gpu_id);
                }
            }
        }

        best_gpu
    }

    /// Find round-robin GPU
    async fn find_round_robin_gpu(
        &self,
        task: &GpuTask,
        gpu_status: &HashMap<usize, GpuMemoryStatus>,
    ) -> Option<usize> {
        let mut counter = self.round_robin_counter.lock().await;
        let start_idx = *counter;

        for i in 0..self.config.gpu_configs.len() {
            let idx = (start_idx + i) % self.config.gpu_configs.len();
            let gpu_config = &self.config.gpu_configs[idx];

            if let Some(status) = gpu_status.get(&gpu_config.gpu_id) {
                if status.is_available
                    && status.available_memory_mb
                        >= task.required_memory_mb + self.config.memory_buffer_mb
                {
                    *counter = (idx + 1) % self.config.gpu_configs.len();
                    return Some(gpu_config.gpu_id);
                }
            }
        }

        None
    }

    /// Find priority GPU
    fn find_priority_gpu(
        &self,
        task: &GpuTask,
        gpu_status: &HashMap<usize, GpuMemoryStatus>,
    ) -> Option<usize> {
        let mut sorted_gpus = self.config.gpu_configs.clone();
        sorted_gpus.sort_by_key(|gpu| gpu.priority);

        for gpu_config in sorted_gpus {
            if let Some(status) = gpu_status.get(&gpu_config.gpu_id) {
                if status.is_available
                    && status.available_memory_mb
                        >= task.required_memory_mb + self.config.memory_buffer_mb
                {
                    return Some(gpu_config.gpu_id);
                }
            }
        }

        None
    }

    /// Find load-balanced GPU
    fn find_load_balanced_gpu(
        &self,
        task: &GpuTask,
        gpu_status: &HashMap<usize, GpuMemoryStatus>,
    ) -> Option<usize> {
        let mut best_gpu = None;
        let mut best_utilization = f32::MAX;

        for gpu_config in &self.config.gpu_configs {
            if let Some(status) = gpu_status.get(&gpu_config.gpu_id) {
                if status.is_available
                    && status.available_memory_mb
                        >= task.required_memory_mb + self.config.memory_buffer_mb
                    && status.utilization_percent < best_utilization
                {
                    best_utilization = status.utilization_percent;
                    best_gpu = Some(gpu_config.gpu_id);
                }
            }
        }

        best_gpu
    }

    /// Execute a task on a specific GPU
    async fn execute_task(&self, task: GpuTask, gpu_id: usize) {
        let task_id = task.task_id.clone();

        // Get GPU semaphore
        let semaphore = {
            let semaphores = self.gpu_semaphores.read().await;
            semaphores.get(&gpu_id).cloned()
        };

        if let Some(semaphore) = semaphore {
            // Acquire GPU resource
            let _permit = semaphore.acquire().await.unwrap();

            // Update GPU status
            self.update_gpu_memory_usage(gpu_id, task.required_memory_mb as i64).await;

            // Add to running tasks
            {
                let mut running_tasks = self.running_tasks.write().await;
                running_tasks.insert(task_id.clone(), (task.clone(), gpu_id));
            }

            // Send task started event
            let _ = self.event_sender.send(GpuSchedulerEvent::TaskStarted {
                task_id: task_id.clone(),
                gpu_id,
                allocated_memory_mb: task.required_memory_mb,
                timestamp: chrono::Utc::now(),
            });

            let start_time = Instant::now();
            let started_at = chrono::Utc::now();

            // Simulate task execution
            let execution_result = timeout(
                Duration::from_secs(self.config.task_timeout_seconds),
                self.simulate_task_execution(&task, gpu_id),
            )
            .await;

            let duration = start_time.elapsed();
            let completed_at = chrono::Utc::now();

            // Create task result
            let result = match execution_result {
                Ok(Ok(_)) => TaskResult {
                    task_id: task_id.clone(),
                    gpu_id,
                    status: TaskStatus::Completed,
                    actual_memory_mb: task.required_memory_mb,
                    actual_duration_seconds: duration.as_secs(),
                    started_at,
                    completed_at,
                    error_message: None,
                },
                Ok(Err(e)) => TaskResult {
                    task_id: task_id.clone(),
                    gpu_id,
                    status: TaskStatus::Failed,
                    actual_memory_mb: task.required_memory_mb,
                    actual_duration_seconds: duration.as_secs(),
                    started_at,
                    completed_at,
                    error_message: Some(e.to_string()),
                },
                Err(_) => TaskResult {
                    task_id: task_id.clone(),
                    gpu_id,
                    status: TaskStatus::TimedOut,
                    actual_memory_mb: task.required_memory_mb,
                    actual_duration_seconds: duration.as_secs(),
                    started_at,
                    completed_at,
                    error_message: Some("Task timed out".to_string()),
                },
            };

            // Store result
            {
                let mut results = self.task_results.lock().await;
                results.insert(task_id.clone(), result.clone());
            }

            // Remove from running tasks
            {
                let mut running_tasks = self.running_tasks.write().await;
                running_tasks.remove(&task_id);
            }

            // Update GPU status
            self.update_gpu_memory_usage(gpu_id, -(task.required_memory_mb as i64)).await;

            // Send completion event
            match result.status {
                TaskStatus::Completed => {
                    let _ = self.event_sender.send(GpuSchedulerEvent::TaskCompleted {
                        task_id: task_id.clone(),
                        gpu_id,
                        duration_seconds: duration.as_secs(),
                        timestamp: chrono::Utc::now(),
                    });
                },
                _ => {
                    let _ = self.event_sender.send(GpuSchedulerEvent::TaskFailed {
                        task_id: task_id.clone(),
                        gpu_id: Some(gpu_id),
                        error: result.error_message.as_ref().unwrap_or(&String::new()).clone(),
                        timestamp: chrono::Utc::now(),
                    });
                },
            }

            // Update statistics
            self.update_stats(&result).await;
        }
    }

    /// Simulate task execution
    async fn simulate_task_execution(&self, task: &GpuTask, _gpu_id: usize) -> Result<()> {
        // Use hash-based randomness (Send-safe)
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        task.task_id.hash(&mut hasher);
        let hash1 = hasher.finish();

        task.task_type.hash(&mut hasher);
        let hash2 = hasher.finish();

        // Simulate processing time
        let processing_time = Duration::from_millis(
            (task.estimated_duration_seconds * 1000) + (hash1 % 1000), // Add some randomness
        );

        sleep(processing_time).await;

        // Simulate occasional failures
        let failure_value = (hash2 % 100) as f64 / 100.0;
        if failure_value < 0.05 {
            // 5% failure rate
            return Err(anyhow::anyhow!("Simulated task failure"));
        }

        Ok(())
    }

    /// Update GPU memory usage
    async fn update_gpu_memory_usage(&self, gpu_id: usize, memory_delta: i64) {
        let mut gpu_status = self.gpu_status.write().await;

        if let Some(status) = gpu_status.get_mut(&gpu_id) {
            if memory_delta > 0 {
                status.used_memory_mb += memory_delta as usize;
                status.active_tasks += 1;
            } else {
                status.used_memory_mb =
                    status.used_memory_mb.saturating_sub((-memory_delta) as usize);
                status.active_tasks = status.active_tasks.saturating_sub(1);
            }

            status.available_memory_mb =
                status.total_memory_mb.saturating_sub(status.used_memory_mb);
            status.utilization_percent =
                (status.used_memory_mb as f32 / status.total_memory_mb as f32) * 100.0;
            status.last_updated = chrono::Utc::now();

            // Send memory update event
            let _ = self.event_sender.send(GpuSchedulerEvent::GpuMemoryUpdated {
                gpu_id,
                utilization_percent: status.utilization_percent,
                timestamp: chrono::Utc::now(),
            });

            // Check for overload
            if status.utilization_percent > self.config.preemption_threshold {
                let _ = self.event_sender.send(GpuSchedulerEvent::GpuOverloaded {
                    gpu_id,
                    utilization_percent: status.utilization_percent,
                    timestamp: chrono::Utc::now(),
                });
            }
        }
    }

    /// Start memory monitoring
    async fn start_memory_monitoring(&self) {
        let scheduler = self.clone();
        let monitoring_interval =
            Duration::from_secs(self.config.memory_monitoring_interval_seconds);

        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(monitoring_interval);

            loop {
                interval.tick().await;

                // Update GPU memory status (simulate actual GPU monitoring)
                scheduler.update_gpu_memory_monitoring().await;

                // Schedule pending tasks
                scheduler.schedule_tasks().await;
            }
        });

        let scheduler_clone = self.clone();
        tokio::spawn(async move {
            let mut handles = scheduler_clone.task_handles.lock().await;
            handles.push(handle);
        });
    }

    /// Update GPU memory monitoring
    async fn update_gpu_memory_monitoring(&self) {
        // This would integrate with actual GPU monitoring libraries
        // For now, we just update the timestamp
        let mut gpu_status = self.gpu_status.write().await;
        for status in gpu_status.values_mut() {
            status.last_updated = chrono::Utc::now();
        }
    }

    /// Update statistics
    async fn update_stats(&self, result: &TaskResult) {
        let mut stats = self.stats.write().await;
        stats.total_tasks += 1;

        // Update failure rate
        if matches!(result.status, TaskStatus::Failed | TaskStatus::TimedOut) {
            stats.failure_rate = (stats.failure_rate * (stats.total_tasks - 1) as f64 + 1.0)
                / stats.total_tasks as f64;
        } else {
            stats.failure_rate =
                (stats.failure_rate * (stats.total_tasks - 1) as f64) / stats.total_tasks as f64;
        }

        // Update average execution time
        stats.avg_execution_time_seconds = (stats.avg_execution_time_seconds
            * (stats.total_tasks - 1) as f64
            + result.actual_duration_seconds as f64)
            / stats.total_tasks as f64;

        // Update running tasks count
        stats.running_tasks = self.running_tasks.read().await.len() as u64;
        stats.queued_tasks = self.task_queue.lock().await.len() as u64;
    }
}

impl Clone for GpuScheduler {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            gpu_status: Arc::clone(&self.gpu_status),
            task_queue: Arc::clone(&self.task_queue),
            running_tasks: Arc::clone(&self.running_tasks),
            task_results: Arc::clone(&self.task_results),
            stats: Arc::clone(&self.stats),
            event_sender: self.event_sender.clone(),
            task_sender: self.task_sender.clone(),
            task_handles: Arc::clone(&self.task_handles),
            gpu_semaphores: Arc::clone(&self.gpu_semaphores),
            round_robin_counter: Arc::clone(&self.round_robin_counter),
        }
    }
}

impl Default for GpuSchedulerStats {
    fn default() -> Self {
        Self {
            total_tasks: 0,
            running_tasks: 0,
            queued_tasks: 0,
            avg_wait_time_seconds: 0.0,
            avg_execution_time_seconds: 0.0,
            gpu_stats: vec![],
            memory_efficiency: 0.0,
            preemption_count: 0,
            failure_rate: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_scheduler_config_default() {
        let config = GpuSchedulerConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_memory_utilization, 90.0);
        assert_eq!(config.scheduling_algorithm, SchedulingAlgorithm::BestFit);
    }

    #[tokio::test]
    async fn test_gpu_scheduler_creation() {
        let config = GpuSchedulerConfig::default();
        let scheduler = GpuScheduler::new(config);

        let stats = scheduler.get_stats().await;
        assert_eq!(stats.total_tasks, 0);
    }

    #[tokio::test]
    async fn test_task_submission() {
        let config = GpuSchedulerConfig::default();
        let scheduler = GpuScheduler::new(config);

        let task = GpuTask {
            task_id: "test-task".to_string(),
            required_memory_mb: 1000,
            estimated_duration_seconds: 5,
            priority: 1,
            task_type: "inference".to_string(),
            client_id: Some("test-client".to_string()),
            metadata: HashMap::new(),
            created_at: chrono::Utc::now(),
            preemptible: false,
        };

        let task_id = scheduler.submit_task(task).await.unwrap();
        assert_eq!(task_id, "test-task");

        // Wait for processing
        tokio::time::sleep(Duration::from_millis(100)).await;

        let status = scheduler.get_task_status(&task_id).await;
        assert!(status.is_some());
    }
}
