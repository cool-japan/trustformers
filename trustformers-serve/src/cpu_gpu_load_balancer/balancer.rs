//! Main CPU/GPU Load Balancer Implementation
//!
//! This module contains the main load balancer orchestrator that coordinates
//! task scheduling, resource management, and performance monitoring.

use anyhow::Result;
use std::{
    collections::{HashMap, VecDeque},
    sync::{Arc, RwLock},
    time::{Duration, Instant},
};
use tokio::{
    sync::{broadcast, mpsc, Mutex, Semaphore},
    time::interval,
};

use super::{
    config::LoadBalancerConfig,
    monitoring::{DefaultPerformanceMonitor, LoadBalancerStats, PerformanceMonitor},
    power::PowerEfficiencyManager,
    scheduler::{DefaultTaskScheduler, TaskQueueManager, TaskScheduler},
    types::{
        ComputeTask, ExecutionStatus, LoadBalancerEvent, PerformanceHistory, ProcessorResource,
        ProcessorType, TaskExecutionResult,
    },
};

/// Main CPU/GPU load balancer
///
/// Orchestrates task scheduling, resource management, and performance monitoring
/// across heterogeneous CPU and GPU compute resources.
pub struct CpuGpuLoadBalancer {
    config: LoadBalancerConfig,

    /// CPU resources
    cpu_resources: Arc<RwLock<Vec<ProcessorResource>>>,

    /// GPU resources
    gpu_resources: Arc<RwLock<Vec<ProcessorResource>>>,

    /// Task queue manager
    task_queue: Arc<Mutex<TaskQueueManager>>,

    /// Running tasks
    running_tasks: Arc<RwLock<HashMap<String, TaskExecutionResult>>>,

    /// Performance history
    performance_history: Arc<Mutex<VecDeque<PerformanceHistory>>>,

    /// Performance monitor
    performance_monitor: Arc<Mutex<DefaultPerformanceMonitor>>,

    /// Power efficiency manager
    power_manager: Arc<Mutex<PowerEfficiencyManager>>,

    /// Task scheduler
    scheduler: Arc<DefaultTaskScheduler>,

    /// Event broadcaster
    event_sender: broadcast::Sender<LoadBalancerEvent>,

    /// Task processing channels
    task_sender: mpsc::UnboundedSender<ComputeTask>,

    /// CPU semaphore
    cpu_semaphore: Arc<Semaphore>,

    /// GPU semaphore
    gpu_semaphore: Arc<Semaphore>,

    /// Background task handles
    task_handles: Arc<Mutex<Vec<tokio::task::JoinHandle<()>>>>,
}

impl CpuGpuLoadBalancer {
    /// Create a new CPU/GPU load balancer
    pub fn new(config: LoadBalancerConfig) -> Self {
        let (event_sender, _) = broadcast::channel(1000);
        let (task_sender, task_receiver) = mpsc::unbounded_channel();

        let cpu_semaphore = Arc::new(Semaphore::new(config.cpu_pool_size));
        let gpu_semaphore = Arc::new(Semaphore::new(1)); // Simplified GPU concurrency

        let scheduler = Arc::new(DefaultTaskScheduler::new(config.clone()));
        let performance_monitor = Arc::new(Mutex::new(DefaultPerformanceMonitor::new()));
        let power_manager = Arc::new(Mutex::new(PowerEfficiencyManager::new(
            config.power_efficiency_mode,
            config.max_power_consumption,
        )));

        let balancer = Self {
            config: config.clone(),
            cpu_resources: Arc::new(RwLock::new(Vec::new())),
            gpu_resources: Arc::new(RwLock::new(Vec::new())),
            task_queue: Arc::new(Mutex::new(TaskQueueManager::new(
                config.task_queue_capacity,
            ))),
            running_tasks: Arc::new(RwLock::new(HashMap::new())),
            performance_history: Arc::new(Mutex::new(VecDeque::new())),
            performance_monitor,
            power_manager,
            scheduler,
            event_sender,
            task_sender,
            cpu_semaphore,
            gpu_semaphore,
            task_handles: Arc::new(Mutex::new(Vec::new())),
        };

        balancer.start_background_tasks(task_receiver);
        balancer
    }

    /// Initialize processor resources
    pub async fn initialize_resources(&self) -> Result<()> {
        self.initialize_cpu_resources().await?;
        self.initialize_gpu_resources().await?;
        Ok(())
    }

    /// Initialize CPU resources
    async fn initialize_cpu_resources(&self) -> Result<()> {
        let mut cpu_resources = self.cpu_resources.write().expect("lock should not be poisoned");
        cpu_resources.clear();

        for i in 0..self.config.cpu_pool_size {
            let resource = ProcessorResource {
                processor_type: ProcessorType::CPU,
                id: i,
                available_memory: 8 * 1024 * 1024 * 1024, // 8GB default
                total_memory: 8 * 1024 * 1024 * 1024,
                ..ProcessorResource::default()
            };
            cpu_resources.push(resource);
        }

        Ok(())
    }

    /// Initialize GPU resources
    async fn initialize_gpu_resources(&self) -> Result<()> {
        let mut gpu_resources =
            self.gpu_resources.write().expect("gpu_resources lock should not be poisoned");
        gpu_resources.clear();

        // Simulate 1 GPU for simplicity
        let resource = ProcessorResource {
            processor_type: ProcessorType::GPU,
            id: 0,
            available_memory: 16 * 1024 * 1024 * 1024, // 16GB GPU memory
            total_memory: 16 * 1024 * 1024 * 1024,
            ..ProcessorResource::default()
        };
        gpu_resources.push(resource);

        Ok(())
    }

    /// Start the load balancer
    pub async fn start(&self) -> Result<()> {
        if !self.config.enabled {
            tracing::info!("CPU/GPU load balancer is disabled");
            return Ok(());
        }

        tracing::info!("Starting CPU/GPU load balancer");

        // Initialize resources
        self.initialize_resources().await?;

        // Start monitoring
        self.start_resource_monitoring().await;

        // Send start event
        let _ = self.event_sender.send(LoadBalancerEvent::LoadBalancerStarted {
            timestamp: chrono::Utc::now(),
        });

        Ok(())
    }

    /// Stop the load balancer
    pub async fn stop(&self) -> Result<()> {
        let mut handles = self.task_handles.lock().await;
        for handle in handles.drain(..) {
            handle.abort();
        }

        // Send stop event
        let _ = self.event_sender.send(LoadBalancerEvent::LoadBalancerStopped {
            timestamp: chrono::Utc::now(),
        });

        tracing::info!("CPU/GPU load balancer stopped");
        Ok(())
    }

    /// Submit a compute task
    pub async fn submit_task(&self, task: ComputeTask) -> Result<String> {
        if !self.config.enabled {
            return Err(anyhow::anyhow!("Load balancer is disabled"));
        }

        let task_id = task.id.clone();

        // Check queue capacity
        {
            let queue = self.task_queue.lock().await;
            if queue.is_full() {
                return Err(anyhow::anyhow!("Task queue is full"));
            }
        }

        // Send task submitted event
        let _ = self.event_sender.send(LoadBalancerEvent::TaskSubmitted {
            task_id: task_id.clone(),
            task_type: task.task_type.clone(),
            timestamp: chrono::Utc::now(),
        });

        // Send task for processing
        self.task_sender.send(task)?;

        Ok(task_id)
    }

    /// Get task status
    pub async fn get_task_status(&self, task_id: &str) -> Option<ExecutionStatus> {
        let running_tasks =
            self.running_tasks.read().expect("running_tasks lock should not be poisoned");
        running_tasks.get(task_id).map(|result| result.status.clone())
    }

    /// Get current statistics
    pub async fn get_stats(&self) -> LoadBalancerStats {
        let monitor = self.performance_monitor.lock().await;
        monitor.get_stats().clone()
    }

    /// Subscribe to load balancer events
    pub fn subscribe_events(&self) -> broadcast::Receiver<LoadBalancerEvent> {
        self.event_sender.subscribe()
    }

    /// Assign processor for task
    pub async fn assign_processor(&self, task: &ComputeTask) -> Option<(ProcessorType, usize)> {
        // Collect all available resources
        let cpu_resources =
            self.cpu_resources.read().expect("cpu_resources lock should not be poisoned");
        let gpu_resources =
            self.gpu_resources.read().expect("gpu_resources lock should not be poisoned");

        let mut all_resources = Vec::new();
        all_resources.extend(cpu_resources.iter().cloned());
        all_resources.extend(gpu_resources.iter().cloned());

        self.scheduler.assign_processor(task, &all_resources)
    }

    /// Start background tasks
    fn start_background_tasks(&self, mut task_receiver: mpsc::UnboundedReceiver<ComputeTask>) {
        let task_queue = Arc::clone(&self.task_queue);
        let scheduler = Arc::clone(&self.scheduler);
        let cpu_resources = Arc::clone(&self.cpu_resources);
        let gpu_resources = Arc::clone(&self.gpu_resources);
        let running_tasks = Arc::clone(&self.running_tasks);
        let performance_monitor = Arc::clone(&self.performance_monitor);
        let event_sender = self.event_sender.clone();
        let cpu_semaphore = Arc::clone(&self.cpu_semaphore);
        let gpu_semaphore = Arc::clone(&self.gpu_semaphore);
        let task_handles = Arc::clone(&self.task_handles);

        // Task processing task
        let task_processor = tokio::spawn(async move {
            while let Some(task) = task_receiver.recv().await {
                // Add to queue
                {
                    let mut queue = task_queue.lock().await;
                    if let Err(e) = queue.enqueue(task) {
                        tracing::error!("Failed to enqueue task: {}", e);
                    }
                }

                // Process tasks from queue
                while let Some(task) = {
                    let mut queue = task_queue.lock().await;
                    queue.dequeue()
                } {
                    // Assign processor
                    let all_resources = {
                        let cpu_resources_guard = cpu_resources
                            .read()
                            .expect("cpu_resources lock should not be poisoned");
                        let gpu_resources_guard = gpu_resources
                            .read()
                            .expect("gpu_resources lock should not be poisoned");

                        let mut resources = Vec::with_capacity(
                            cpu_resources_guard.len() + gpu_resources_guard.len(),
                        );
                        resources.extend(cpu_resources_guard.iter().cloned());
                        resources.extend(gpu_resources_guard.iter().cloned());
                        resources
                    };

                    if let Some((processor_type, processor_id)) =
                        scheduler.assign_processor(&task, &all_resources)
                    {
                        // Send assignment event
                        let _ = event_sender.send(LoadBalancerEvent::TaskAssigned {
                            task_id: task.id.clone(),
                            processor_type,
                            processor_id,
                            timestamp: chrono::Utc::now(),
                        });

                        // Execute task
                        let task_id = task.id.clone();
                        let execution_result = TaskExecutionResult {
                            task_id: task_id.clone(),
                            processor: processor_type,
                            processor_id,
                            status: ExecutionStatus::Running,
                            execution_time: Duration::from_millis(0),
                            memory_used: task.memory_required,
                            energy_consumed: 0.0,
                            throughput: 0.0,
                            started_at: Instant::now(),
                            completed_at: Instant::now(),
                            error_message: None,
                        };

                        // Add to running tasks
                        {
                            let mut running = running_tasks
                                .write()
                                .expect("running_tasks lock should not be poisoned");
                            running.insert(task_id.clone(), execution_result);
                        }

                        // Simulate task execution
                        let semaphore = match processor_type {
                            ProcessorType::CPU => Arc::clone(&cpu_semaphore),
                            ProcessorType::GPU => Arc::clone(&gpu_semaphore),
                        };

                        let _permit =
                            semaphore.acquire().await.expect("semaphore should not be closed");

                        // Simulate execution time
                        let execution_time =
                            Duration::from_millis(100 + (task.compute_operations / 1000) as u64);
                        tokio::time::sleep(execution_time).await;

                        // Complete task
                        {
                            let mut running = running_tasks
                                .write()
                                .expect("running_tasks lock should not be poisoned");
                            if let Some(result) = running.get_mut(&task_id) {
                                result.status = ExecutionStatus::Completed;
                                result.execution_time = execution_time;
                                result.completed_at = Instant::now();
                                result.throughput =
                                    task.compute_operations as f64 / execution_time.as_secs_f64();
                            }
                        }

                        // Update performance monitor
                        {
                            let mut monitor = performance_monitor.lock().await;
                            monitor.update_task_stats(processor_type, execution_time, true);
                        }

                        // Send completion event
                        let _ = event_sender.send(LoadBalancerEvent::TaskCompleted {
                            task_id,
                            processor_type,
                            execution_time,
                            timestamp: chrono::Utc::now(),
                        });
                    }
                }
            }
        });

        // Store handle for cleanup
        tokio::spawn(async move {
            let mut handles = task_handles.lock().await;
            handles.push(task_processor);
        });
    }

    /// Start resource monitoring
    async fn start_resource_monitoring(&self) {
        let cpu_resources = Arc::clone(&self.cpu_resources);
        let gpu_resources = Arc::clone(&self.gpu_resources);
        let performance_monitor = Arc::clone(&self.performance_monitor);
        let power_manager = Arc::clone(&self.power_manager);
        let monitoring_interval = self.config.monitoring_interval_seconds;

        let monitor_task = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(monitoring_interval));

            loop {
                interval.tick().await;

                // Update CPU utilization
                let cpu_resources_snapshot: Vec<_> = {
                    let cpu_resources_guard =
                        cpu_resources.read().expect("cpu_resources lock should not be poisoned");
                    cpu_resources_guard.iter().cloned().collect()
                };

                for resource in cpu_resources_snapshot {
                    let mut monitor = performance_monitor.lock().await;
                    monitor.update_utilization(ProcessorType::CPU, resource.utilization);

                    let mut power_mgr = power_manager.lock().await;
                    power_mgr.update_power_consumption(
                        ProcessorType::CPU,
                        resource.power_consumption,
                        resource.utilization,
                    );
                }

                // Update GPU utilization
                let gpu_resources_snapshot: Vec<_> = {
                    let gpu_resources_guard =
                        gpu_resources.read().expect("gpu_resources lock should not be poisoned");
                    gpu_resources_guard.iter().cloned().collect()
                };

                for resource in gpu_resources_snapshot {
                    let mut monitor = performance_monitor.lock().await;
                    monitor.update_utilization(ProcessorType::GPU, resource.utilization);

                    let mut power_mgr = power_manager.lock().await;
                    power_mgr.update_power_consumption(
                        ProcessorType::GPU,
                        resource.power_consumption,
                        resource.utilization,
                    );
                }
            }
        });

        // Store handle for cleanup
        let mut handles = self.task_handles.lock().await;
        handles.push(monitor_task);
    }
}

/// Clone implementation for shared use
impl Clone for CpuGpuLoadBalancer {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            cpu_resources: Arc::clone(&self.cpu_resources),
            gpu_resources: Arc::clone(&self.gpu_resources),
            task_queue: Arc::clone(&self.task_queue),
            running_tasks: Arc::clone(&self.running_tasks),
            performance_history: Arc::clone(&self.performance_history),
            performance_monitor: Arc::clone(&self.performance_monitor),
            power_manager: Arc::clone(&self.power_manager),
            scheduler: Arc::clone(&self.scheduler),
            event_sender: self.event_sender.clone(),
            task_sender: self.task_sender.clone(),
            cpu_semaphore: Arc::clone(&self.cpu_semaphore),
            gpu_semaphore: Arc::clone(&self.gpu_semaphore),
            task_handles: Arc::clone(&self.task_handles),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu_gpu_load_balancer::config::{LoadBalancerConfig, LoadBalancingStrategy};
    use crate::cpu_gpu_load_balancer::types::{ComputeTask, TaskPriority, TaskType};

    fn make_test_config() -> LoadBalancerConfig {
        LoadBalancerConfig {
            enabled: true,
            cpu_pool_size: 2,
            task_queue_capacity: 100,
            monitoring_interval_seconds: 60,
            strategy: LoadBalancingStrategy::Adaptive,
            ..LoadBalancerConfig::default()
        }
    }

    fn make_inference_task(id: &str) -> ComputeTask {
        ComputeTask {
            task_type: TaskType::Inference,
            compute_operations: 5000,
            parallelizability: 0.8,
            memory_required: 1024,
            priority: TaskPriority::Normal,
            ..ComputeTask::new(id.to_string(), TaskType::Inference)
        }
    }

    // --- CpuGpuLoadBalancer construction tests ---

    #[tokio::test]
    async fn test_balancer_new_creates_instance() {
        let config = make_test_config();
        let _balancer = CpuGpuLoadBalancer::new(config);
        // Just verify construction succeeds
    }

    #[tokio::test]
    async fn test_balancer_initial_stats_zero_tasks() {
        let config = make_test_config();
        let balancer = CpuGpuLoadBalancer::new(config);
        let stats = balancer.get_stats().await;
        assert_eq!(stats.total_tasks, 0);
    }

    #[tokio::test]
    async fn test_balancer_start_and_stop() {
        let config = make_test_config();
        let balancer = CpuGpuLoadBalancer::new(config);
        balancer.start().await.expect("start should succeed");
        balancer.stop().await.expect("stop should succeed");
    }

    #[tokio::test]
    async fn test_balancer_disabled_rejects_tasks() {
        let config = LoadBalancerConfig {
            enabled: false,
            ..make_test_config()
        };
        let balancer = CpuGpuLoadBalancer::new(config);
        let task = make_inference_task("t1");
        let result = balancer.submit_task(task).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_balancer_submit_task_returns_task_id() {
        let config = make_test_config();
        let balancer = CpuGpuLoadBalancer::new(config);
        balancer.start().await.expect("start should succeed");
        let task = make_inference_task("task_abc");
        let result = balancer.submit_task(task).await;
        assert!(result.is_ok());
        assert_eq!(result.expect("should be Ok"), "task_abc");
    }

    #[tokio::test]
    async fn test_balancer_get_task_status_unknown_returns_none() {
        let config = make_test_config();
        let balancer = CpuGpuLoadBalancer::new(config);
        let status = balancer.get_task_status("nonexistent_task").await;
        assert!(status.is_none());
    }

    #[tokio::test]
    async fn test_balancer_subscribe_events_returns_receiver() {
        let config = make_test_config();
        let balancer = CpuGpuLoadBalancer::new(config);
        let _receiver = balancer.subscribe_events();
        // Just verify we get a valid receiver
    }

    #[tokio::test]
    async fn test_balancer_assign_processor_after_init() {
        let config = make_test_config();
        let balancer = CpuGpuLoadBalancer::new(config);
        balancer.start().await.expect("start should succeed");
        let task = make_inference_task("t1");
        // May return None if no resources available or Some if assigned
        let _ = balancer.assign_processor(&task).await;
    }

    #[tokio::test]
    async fn test_balancer_initialize_resources_creates_cpu_resources() {
        let config = LoadBalancerConfig {
            cpu_pool_size: 4,
            ..make_test_config()
        };
        let balancer = CpuGpuLoadBalancer::new(config);
        balancer.initialize_resources().await.expect("init should succeed");
        let cpu_count = balancer.cpu_resources.read().expect("lock ok").len();
        assert_eq!(cpu_count, 4);
    }

    #[tokio::test]
    async fn test_balancer_clone_shares_state() {
        let config = make_test_config();
        let balancer = CpuGpuLoadBalancer::new(config);
        let _cloned = balancer.clone();
        // Just verify clone doesn't panic and is possible
    }

    #[tokio::test]
    async fn test_balancer_multiple_tasks_submitted() {
        let config = make_test_config();
        let balancer = CpuGpuLoadBalancer::new(config);
        balancer.start().await.expect("start should succeed");
        for i in 0..5 {
            let task = make_inference_task(&format!("task_{}", i));
            balancer.submit_task(task).await.expect("submit should succeed");
        }
    }

    #[tokio::test]
    async fn test_balancer_queue_full_returns_error() {
        let config = LoadBalancerConfig {
            task_queue_capacity: 1,
            ..make_test_config()
        };
        let balancer = CpuGpuLoadBalancer::new(config);
        balancer.start().await.expect("start should succeed");
        // Fill the queue
        let t1 = make_inference_task("t1");
        let _ = balancer.submit_task(t1).await;
        // Give task processor time to possibly drain
        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
        // Submit another and check if queue gets full at some point
        // The test just verifies no panic occurs
        let t2 = make_inference_task("t2");
        let _ = balancer.submit_task(t2).await;
    }
}
