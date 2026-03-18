//! Analysis Scheduler
//!
//! Scheduler for intelligent analysis task scheduling.

use super::super::types::*;
use super::*;

use crate::parallel_execution_engine::SchedulingConfig;
use crate::test_parallelization::SchedulingStrategy;
use anyhow::{anyhow, Result};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{Mutex as TokioMutex, RwLock as TokioRwLock, Semaphore};
use tokio::task::{spawn, JoinHandle};
use tracing::{debug, error, info, instrument};

#[derive(Debug)]
pub struct AnalysisScheduler {
    /// Scheduling configuration
    config: Arc<TokioRwLock<SchedulingConfig>>,
    /// Task queues by priority
    task_queues: Arc<TokioMutex<BTreeMap<u8, VecDeque<AnalysisTask>>>>,
    /// Active tasks
    active_tasks: Arc<TokioMutex<HashMap<String, ActiveTask>>>,
    /// Component manager reference
    component_manager: Arc<ComponentManager>,
    /// Error recovery manager reference
    error_recovery_manager: Arc<ErrorRecoveryManager>,
    /// Task semaphore for concurrency control
    task_semaphore: Arc<Semaphore>,
    /// Scheduler statistics
    scheduler_stats: Arc<SchedulerStatistics>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
}

/// Analysis task for scheduling
#[derive(Debug, Clone)]
pub struct AnalysisTask {
    /// Task ID
    pub id: String,
    /// Test data
    pub test_data: TestExecutionData,
    /// Profiling options
    pub options: ProfilingOptions,
    /// Task priority (0 = highest)
    pub priority: u8,
    /// Task creation time
    pub created_at: SystemTime,
    /// Estimated duration
    pub estimated_duration: Option<Duration>,
    /// Required resources
    pub required_resources: ResourceRequirements,
}

/// Active task information
#[derive(Debug)]
pub struct ActiveTask {
    /// Task information
    pub task: AnalysisTask,
    /// Task start time
    pub started_at: SystemTime,
    /// Task handle
    pub handle: JoinHandle<Result<TestCharacteristics>>,
}

/// Resource requirements for a task
#[derive(Debug, Clone, Default)]
pub struct ResourceRequirements {
    /// CPU cores required
    pub cpu_cores: usize,
    /// Memory required in MB
    pub memory_mb: usize,
    /// I/O priority required
    pub io_priority: u8,
}

/// Scheduler statistics
#[derive(Debug, Default)]
pub struct SchedulerStatistics {
    /// Total tasks scheduled
    pub total_scheduled: AtomicU64,
    /// Total tasks completed
    pub total_completed: AtomicU64,
    /// Total tasks failed
    pub total_failed: AtomicU64,
    /// Average task wait time
    pub average_wait_time_ms: AtomicU64,
    /// Average task execution time
    pub average_execution_time_ms: AtomicU64,
    /// Current queue size
    pub current_queue_size: AtomicUsize,
    /// Current active tasks
    pub current_active_tasks: AtomicUsize,
}

impl AnalysisScheduler {
    /// Create a new analysis scheduler
    pub async fn new(
        config: SchedulingConfig,
        component_manager: Arc<ComponentManager>,
        error_recovery_manager: Arc<ErrorRecoveryManager>,
    ) -> Result<Self> {
        let task_semaphore = Arc::new(Semaphore::new(config.queue_management.max_queue_size));

        Ok(Self {
            config: Arc::new(TokioRwLock::new(config)),
            task_queues: Arc::new(TokioMutex::new(BTreeMap::new())),
            active_tasks: Arc::new(TokioMutex::new(HashMap::new())),
            component_manager,
            error_recovery_manager,
            task_semaphore,
            scheduler_stats: Arc::new(SchedulerStatistics::default()),
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Schedule an analysis task
    #[instrument(skip(self, task))]
    pub async fn schedule_task(&self, task: AnalysisTask) -> Result<String> {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(anyhow!("Scheduler is shutting down"));
        }

        let task_id = task.id.clone();
        let priority = task.priority;

        debug!(
            "Scheduling analysis task: {} with priority {}",
            task_id, priority
        );

        // Add task to appropriate priority queue
        {
            let mut queues = self.task_queues.lock().await;
            let queue = queues.entry(priority).or_insert_with(VecDeque::new);
            queue.push_back(task);
        }

        // Update statistics
        self.scheduler_stats.total_scheduled.fetch_add(1, Ordering::Relaxed);
        let queue_size = {
            let queues = self.task_queues.lock().await;
            queues.values().map(|q| q.len()).sum::<usize>()
        };
        self.scheduler_stats.current_queue_size.store(queue_size, Ordering::Relaxed);

        // Try to execute task immediately if resources are available
        self.try_execute_next_task().await?;

        Ok(task_id)
    }

    /// Try to execute the next task from the queue
    async fn try_execute_next_task(&self) -> Result<()> {
        // Check if we can acquire a permit for task execution
        if let Ok(permit) = self.task_semaphore.try_acquire() {
            if let Some(task) = self.get_next_task().await {
                self.execute_task(task, permit).await?;
            } else {
                // No task available, release the permit
                permit.forget();
            }
        }

        Ok(())
    }

    /// Get the next task based on scheduling algorithm
    async fn get_next_task(&self) -> Option<AnalysisTask> {
        let mut queues = self.task_queues.lock().await;
        let config = self.config.read().await;

        match config.strategy {
            SchedulingStrategy::Priority => {
                // Get task from highest priority queue (lowest number)
                for (_, queue) in queues.iter_mut() {
                    if let Some(task) = queue.pop_front() {
                        return Some(task);
                    }
                }
            },
            SchedulingStrategy::Fifo => {
                // Get oldest task across all queues
                let mut oldest_task: Option<(u8, usize, AnalysisTask)> = None;

                for (priority_key, queue) in queues.iter() {
                    for (index, task) in queue.iter().enumerate() {
                        if let Some((_, _, ref current_oldest)) = oldest_task {
                            if task.created_at < current_oldest.created_at {
                                oldest_task = Some((*priority_key, index, task.clone()));
                            }
                        } else {
                            oldest_task = Some((*priority_key, index, task.clone()));
                        }
                    }
                }

                if let Some((priority, index, task)) = oldest_task {
                    if let Some(queue) = queues.get_mut(&priority) {
                        queue.remove(index);
                        return Some(task);
                    }
                }
            },
            SchedulingStrategy::ShortestJobFirst => {
                // Get task with shortest estimated duration
                let mut shortest_task: Option<(u8, usize, AnalysisTask)> = None;

                for (priority_key, queue) in queues.iter() {
                    for (index, task) in queue.iter().enumerate() {
                        if let Some(estimated_duration) = task.estimated_duration {
                            if let Some((_, _, ref current_shortest)) = shortest_task {
                                if let Some(current_duration) = current_shortest.estimated_duration
                                {
                                    if estimated_duration < current_duration {
                                        shortest_task = Some((*priority_key, index, task.clone()));
                                    }
                                }
                            } else {
                                shortest_task = Some((*priority_key, index, task.clone()));
                            }
                        }
                    }
                }

                if let Some((priority, index, task)) = shortest_task {
                    if let Some(queue) = queues.get_mut(&priority) {
                        queue.remove(index);
                        return Some(task);
                    }
                }
            },
            _ => {
                // Default to priority scheduling
                for (_, queue) in queues.iter_mut() {
                    if let Some(task) = queue.pop_front() {
                        return Some(task);
                    }
                }
            },
        }

        None
    }

    /// Execute a task
    async fn execute_task(
        &self,
        task: AnalysisTask,
        _permit: tokio::sync::SemaphorePermit<'_>,
    ) -> Result<()> {
        let task_id = task.id.clone();
        let started_at = SystemTime::now();

        debug!("Executing analysis task: {}", task_id);

        // Calculate wait time
        if let Ok(wait_duration) = started_at.duration_since(task.created_at) {
            let current_avg = self.scheduler_stats.average_wait_time_ms.load(Ordering::Relaxed);
            let new_avg = if current_avg == 0 {
                wait_duration.as_millis() as u64
            } else {
                (current_avg + wait_duration.as_millis() as u64) / 2
            };
            self.scheduler_stats.average_wait_time_ms.store(new_avg, Ordering::Relaxed);
        }

        // Create task execution future
        let component_manager = self.component_manager.clone();
        let error_recovery_manager = self.error_recovery_manager.clone();
        let scheduler_stats = self.scheduler_stats.clone();
        let task_clone = task.clone();

        let handle = spawn(async move {
            let execution_start = Instant::now();

            // Execute the analysis
            let result = Self::execute_analysis_task(&task_clone, component_manager).await;

            let execution_duration = execution_start.elapsed();

            // Update statistics
            let current_avg = scheduler_stats.average_execution_time_ms.load(Ordering::Relaxed);
            let new_avg = if current_avg == 0 {
                execution_duration.as_millis() as u64
            } else {
                (current_avg + execution_duration.as_millis() as u64) / 2
            };
            scheduler_stats.average_execution_time_ms.store(new_avg, Ordering::Relaxed);

            match result {
                Ok(characteristics) => {
                    scheduler_stats.total_completed.fetch_add(1, Ordering::Relaxed);
                    Ok(characteristics)
                },
                Err(e) => {
                    scheduler_stats.total_failed.fetch_add(1, Ordering::Relaxed);

                    // Try error recovery
                    match error_recovery_manager.recover_from_task_error(&task_clone, &e).await {
                        Ok(recovered_result) => {
                            scheduler_stats.total_completed.fetch_add(1, Ordering::Relaxed);
                            Ok(recovered_result)
                        },
                        Err(recovery_error) => {
                            error!("Task execution and recovery failed: {}", recovery_error);
                            Err(e)
                        },
                    }
                },
            }
        });

        // Track active task
        {
            let mut active_tasks = self.active_tasks.lock().await;
            active_tasks.insert(
                task_id.clone(),
                ActiveTask {
                    task,
                    started_at,
                    handle,
                },
            );
        }

        // Update active task count
        let active_count = {
            let active_tasks = self.active_tasks.lock().await;
            active_tasks.len()
        };
        self.scheduler_stats.current_active_tasks.store(active_count, Ordering::Relaxed);

        Ok(())
    }

    /// Execute an analysis task
    async fn execute_analysis_task(
        task: &AnalysisTask,
        _component_manager: Arc<ComponentManager>,
    ) -> Result<TestCharacteristics> {
        // This would integrate with the AnalysisOrchestrator to perform the actual analysis
        // For now, we'll simulate the execution

        debug!("Executing analysis for task: {}", task.id);

        // Simulate some analysis work
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Return default characteristics for now
        // In a real implementation, this would call the analysis orchestrator
        Ok(TestCharacteristics::default())
    }

    /// Get scheduler statistics
    pub async fn get_statistics(&self) -> SchedulerStatistics {
        SchedulerStatistics {
            total_scheduled: AtomicU64::new(
                self.scheduler_stats.total_scheduled.load(Ordering::Relaxed),
            ),
            total_completed: AtomicU64::new(
                self.scheduler_stats.total_completed.load(Ordering::Relaxed),
            ),
            total_failed: AtomicU64::new(self.scheduler_stats.total_failed.load(Ordering::Relaxed)),
            average_wait_time_ms: AtomicU64::new(
                self.scheduler_stats.average_wait_time_ms.load(Ordering::Relaxed),
            ),
            average_execution_time_ms: AtomicU64::new(
                self.scheduler_stats.average_execution_time_ms.load(Ordering::Relaxed),
            ),
            current_queue_size: AtomicUsize::new(
                self.scheduler_stats.current_queue_size.load(Ordering::Relaxed),
            ),
            current_active_tasks: AtomicUsize::new(
                self.scheduler_stats.current_active_tasks.load(Ordering::Relaxed),
            ),
        }
    }

    /// Shutdown the scheduler
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down AnalysisScheduler");

        self.shutdown.store(true, Ordering::Release);

        // Cancel all active tasks
        {
            let mut active_tasks = self.active_tasks.lock().await;
            for (_, active_task) in active_tasks.drain() {
                active_task.handle.abort();
            }
        }

        // Clear task queues
        {
            let mut queues = self.task_queues.lock().await;
            queues.clear();
        }

        info!("AnalysisScheduler shutdown completed");
        Ok(())
    }
}
