use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJob {
    pub job_id: String,
    pub job_name: String,
    pub model_config: ModelConfig,
    pub training_config: TrainingJobConfig,
    pub resource_requirements: ResourceRequirements,
    pub priority: JobPriority,
    pub status: JobStatus,
    pub created_at: u64,
    pub started_at: Option<u64>,
    pub completed_at: Option<u64>,
    pub error_message: Option<String>,
    pub progress: f32,
    pub checkpoints: Vec<CheckpointInfo>,
    pub metrics: TrainingMetrics,
    pub dependencies: Vec<String>,
    pub retry_count: u32,
    pub max_retries: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_type: String,
    pub model_size: String,
    pub architecture_params: HashMap<String, String>,
    pub pretrained_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJobConfig {
    pub dataset_path: String,
    pub validation_split: f32,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub epochs: u32,
    pub optimizer: String,
    pub loss_function: String,
    pub regularization: HashMap<String, f32>,
    pub early_stopping: Option<EarlyStoppingConfig>,
    pub checkpointing: CheckpointConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequirements {
    pub cpu_cores: u32,
    pub memory_gb: u32,
    pub gpu_count: u32,
    pub gpu_memory_gb: u32,
    pub storage_gb: u32,
    pub network_bandwidth_mbps: Option<u32>,
    pub node_requirements: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    pub monitor_metric: String,
    pub patience: u32,
    pub min_delta: f32,
    pub mode: String, // "min" or "max"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    pub save_frequency: u32, // epochs
    pub keep_best_only: bool,
    pub save_weights_only: bool,
    pub monitor_metric: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointInfo {
    pub checkpoint_id: String,
    pub epoch: u32,
    pub metrics: HashMap<String, f32>,
    pub file_path: String,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub current_epoch: u32,
    pub loss: f32,
    pub accuracy: f32,
    pub validation_loss: f32,
    pub validation_accuracy: f32,
    pub learning_rate: f32,
    pub throughput: f32, // samples per second
    pub custom_metrics: HashMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JobPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum JobStatus {
    Pending,
    Queued,
    Running,
    Paused,
    Completed,
    Failed,
    Cancelled,
    Retrying,
}

#[derive(Debug, Clone)]
pub struct ResourceNode {
    pub node_id: String,
    pub node_name: String,
    pub available_resources: ResourceRequirements,
    pub allocated_resources: ResourceRequirements,
    pub status: NodeStatus,
    pub last_heartbeat: Instant,
    pub running_jobs: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum NodeStatus {
    Available,
    Busy,
    Maintenance,
    Failed,
}

pub struct TrainingOrchestrator {
    jobs: Arc<RwLock<HashMap<String, TrainingJob>>>,
    job_queue: Arc<RwLock<VecDeque<String>>>,
    nodes: Arc<RwLock<HashMap<String, ResourceNode>>>,
    scheduler: Arc<Mutex<JobScheduler>>,
    job_sender: mpsc::UnboundedSender<JobEvent>,
    #[allow(dead_code)]
    job_receiver: Arc<Mutex<mpsc::UnboundedReceiver<JobEvent>>>,
    statistics: Arc<RwLock<OrchestrationStatistics>>,
}

#[derive(Debug, Clone)]
pub enum JobEvent {
    JobSubmitted(String),
    JobStarted(String),
    JobCompleted(String),
    JobFailed(String, String),
    JobCancelled(String),
    ProgressUpdate(String, f32),
    MetricsUpdate(String, TrainingMetrics),
    CheckpointCreated(String, CheckpointInfo),
    NodeRegistered(String),
    NodeFailed(String),
}

#[derive(Debug, Default, Clone)]
pub struct OrchestrationStatistics {
    pub total_jobs_submitted: usize,
    pub jobs_completed: usize,
    pub jobs_failed: usize,
    pub jobs_cancelled: usize,
    pub average_completion_time: Duration,
    pub resource_utilization: f32,
    pub queue_length: usize,
    pub active_jobs: usize,
}

pub struct JobScheduler {
    scheduling_strategy: SchedulingStrategy,
    resource_allocator: ResourceAllocator,
}

#[derive(Debug, Clone)]
pub enum SchedulingStrategy {
    FIFO,
    Priority,
    ShortestJobFirst,
    FairShare,
    BackfillAware,
}

pub struct ResourceAllocator {
    allocation_strategy: AllocationStrategy,
}

#[derive(Debug, Clone)]
pub enum AllocationStrategy {
    FirstFit,
    BestFit,
    WorstFit,
    LoadBalanced,
}

impl TrainingOrchestrator {
    pub fn new(
        scheduling_strategy: SchedulingStrategy,
        allocation_strategy: AllocationStrategy,
    ) -> Self {
        let (job_sender, job_receiver) = mpsc::unbounded_channel();

        Self {
            jobs: Arc::new(RwLock::new(HashMap::new())),
            job_queue: Arc::new(RwLock::new(VecDeque::new())),
            nodes: Arc::new(RwLock::new(HashMap::new())),
            scheduler: Arc::new(Mutex::new(JobScheduler {
                scheduling_strategy,
                resource_allocator: ResourceAllocator {
                    allocation_strategy,
                },
            })),
            job_sender,
            job_receiver: Arc::new(Mutex::new(job_receiver)),
            statistics: Arc::new(RwLock::new(OrchestrationStatistics::default())),
        }
    }

    pub fn submit_job(&self, mut job: TrainingJob) -> Result<String> {
        job.job_id = Uuid::new_v4().to_string();
        job.status = JobStatus::Pending;
        job.created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();

        // Add to jobs registry
        {
            let mut jobs = self
                .jobs
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on jobs"))?;
            jobs.insert(job.job_id.clone(), job.clone());
        }

        // Add to queue
        {
            let mut queue = self
                .job_queue
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on job queue"))?;
            queue.push_back(job.job_id.clone());
        }

        // Update statistics
        {
            let mut stats = self
                .statistics
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on statistics"))?;
            stats.total_jobs_submitted += 1;
            stats.queue_length = {
                let queue = self
                    .job_queue
                    .read()
                    .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on job queue"))?;
                queue.len()
            };
        }

        // Send event
        self.job_sender
            .send(JobEvent::JobSubmitted(job.job_id.clone()))
            .map_err(|_| anyhow::anyhow!("Failed to send job submitted event"))?;

        Ok(job.job_id)
    }

    pub fn cancel_job(&self, job_id: &str) -> Result<()> {
        let mut jobs = self
            .jobs
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on jobs"))?;

        if let Some(job) = jobs.get_mut(job_id) {
            match job.status {
                JobStatus::Pending | JobStatus::Queued | JobStatus::Running | JobStatus::Paused => {
                    job.status = JobStatus::Cancelled;
                    job.completed_at = Some(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .expect("SystemTime should be after UNIX_EPOCH")
                            .as_secs(),
                    );

                    // Remove from queue if still queued
                    let mut queue = self.job_queue.write().map_err(|_| {
                        anyhow::anyhow!("Failed to acquire write lock on job queue")
                    })?;
                    queue.retain(|id| id != job_id);

                    // Update statistics
                    let mut stats = self.statistics.write().map_err(|_| {
                        anyhow::anyhow!("Failed to acquire write lock on statistics")
                    })?;
                    stats.jobs_cancelled += 1;

                    // Send event
                    self.job_sender
                        .send(JobEvent::JobCancelled(job_id.to_string()))
                        .map_err(|_| anyhow::anyhow!("Failed to send job cancelled event"))?;

                    Ok(())
                },
                _ => Err(anyhow::anyhow!(
                    "Job cannot be cancelled in current status: {:?}",
                    job.status
                )),
            }
        } else {
            Err(anyhow::anyhow!("Job not found: {}", job_id))
        }
    }

    pub fn pause_job(&self, job_id: &str) -> Result<()> {
        let mut jobs = self
            .jobs
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on jobs"))?;

        if let Some(job) = jobs.get_mut(job_id) {
            if matches!(job.status, JobStatus::Running) {
                job.status = JobStatus::Paused;
                Ok(())
            } else {
                Err(anyhow::anyhow!(
                    "Job cannot be paused in current status: {:?}",
                    job.status
                ))
            }
        } else {
            Err(anyhow::anyhow!("Job not found: {}", job_id))
        }
    }

    pub fn resume_job(&self, job_id: &str) -> Result<()> {
        let mut jobs = self
            .jobs
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on jobs"))?;

        if let Some(job) = jobs.get_mut(job_id) {
            if matches!(job.status, JobStatus::Paused) {
                job.status = JobStatus::Queued;

                // Add back to queue
                let mut queue = self
                    .job_queue
                    .write()
                    .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on job queue"))?;
                queue.push_back(job_id.to_string());

                Ok(())
            } else {
                Err(anyhow::anyhow!(
                    "Job cannot be resumed in current status: {:?}",
                    job.status
                ))
            }
        } else {
            Err(anyhow::anyhow!("Job not found: {}", job_id))
        }
    }

    pub fn register_node(&self, node: ResourceNode) -> Result<()> {
        let mut nodes = self
            .nodes
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on nodes"))?;

        nodes.insert(node.node_id.clone(), node.clone());

        // Send event
        self.job_sender
            .send(JobEvent::NodeRegistered(node.node_id))
            .map_err(|_| anyhow::anyhow!("Failed to send node registered event"))?;

        Ok(())
    }

    pub fn unregister_node(&self, node_id: &str) -> Result<()> {
        let mut nodes = self
            .nodes
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on nodes"))?;

        if let Some(node) = nodes.remove(node_id) {
            // Cancel any running jobs on this node
            for job_id in &node.running_jobs {
                let _ = self.cancel_job(job_id);
            }
            Ok(())
        } else {
            Err(anyhow::anyhow!("Node not found: {}", node_id))
        }
    }

    pub fn schedule_jobs(&self) -> Result<Vec<(String, String)>> {
        let scheduler = self
            .scheduler
            .lock()
            .map_err(|_| anyhow::anyhow!("Failed to acquire lock on scheduler"))?;

        let jobs = self
            .jobs
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on jobs"))?;

        let nodes = self
            .nodes
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on nodes"))?;

        let queue = self
            .job_queue
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on job queue"))?;

        let mut assignments = Vec::new();

        // Get jobs ready for scheduling based on priority and dependencies
        let mut ready_jobs: Vec<_> = queue
            .iter()
            .filter_map(|job_id| jobs.get(job_id))
            .filter(|job| self.are_dependencies_satisfied(&job.dependencies, &jobs))
            .collect();

        // Sort based on scheduling strategy
        match scheduler.scheduling_strategy {
            SchedulingStrategy::Priority => {
                ready_jobs
                    .sort_by(|a, b| (b.priority.clone() as u8).cmp(&(a.priority.clone() as u8)));
            },
            SchedulingStrategy::ShortestJobFirst => {
                ready_jobs.sort_by(|a, b| a.training_config.epochs.cmp(&b.training_config.epochs));
            },
            SchedulingStrategy::FIFO => {
                ready_jobs.sort_by(|a, b| a.created_at.cmp(&b.created_at));
            },
            _ => {}, // Keep original order for other strategies
        }

        // Try to assign jobs to nodes
        for job in ready_jobs {
            if let Some(node_id) = scheduler
                .resource_allocator
                .find_suitable_node(&job.resource_requirements, &nodes)
            {
                assignments.push((job.job_id.clone(), node_id));
            }
        }

        Ok(assignments)
    }

    fn are_dependencies_satisfied(
        &self,
        dependencies: &[String],
        jobs: &HashMap<String, TrainingJob>,
    ) -> bool {
        dependencies.iter().all(|dep_id| {
            jobs.get(dep_id)
                .map(|job| matches!(job.status, JobStatus::Completed))
                .unwrap_or(false)
        })
    }

    pub fn start_job(&self, job_id: &str, node_id: &str) -> Result<()> {
        // Update job status
        {
            let mut jobs = self
                .jobs
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on jobs"))?;

            if let Some(job) = jobs.get_mut(job_id) {
                job.status = JobStatus::Running;
                job.started_at = Some(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .expect("SystemTime should be after UNIX_EPOCH")
                        .as_secs(),
                );
            } else {
                return Err(anyhow::anyhow!("Job not found: {}", job_id));
            }
        }

        // Update node allocation
        {
            let mut nodes = self
                .nodes
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on nodes"))?;

            if let Some(node) = nodes.get_mut(node_id) {
                node.running_jobs.push(job_id.to_string());

                // Update allocated resources (simplified)
                let jobs = self
                    .jobs
                    .read()
                    .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on jobs"))?;
                if let Some(job) = jobs.get(job_id) {
                    node.allocated_resources.cpu_cores += job.resource_requirements.cpu_cores;
                    node.allocated_resources.memory_gb += job.resource_requirements.memory_gb;
                    node.allocated_resources.gpu_count += job.resource_requirements.gpu_count;
                }
            }
        }

        // Remove from queue
        {
            let mut queue = self
                .job_queue
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on job queue"))?;
            queue.retain(|id| id != job_id);
        }

        // Update statistics
        {
            let mut stats = self
                .statistics
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on statistics"))?;
            stats.active_jobs += 1;
        }

        // Send event
        self.job_sender
            .send(JobEvent::JobStarted(job_id.to_string()))
            .map_err(|_| anyhow::anyhow!("Failed to send job started event"))?;

        Ok(())
    }

    pub fn complete_job(&self, job_id: &str) -> Result<()> {
        let completion_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime should be after UNIX_EPOCH")
            .as_secs();

        // Update job status
        let (node_id, job_duration) = {
            let mut jobs = self
                .jobs
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on jobs"))?;

            if let Some(job) = jobs.get_mut(job_id) {
                job.status = JobStatus::Completed;
                job.completed_at = Some(completion_time);
                job.progress = 100.0;

                let duration = job
                    .started_at
                    .map(|start| Duration::from_secs(completion_time - start))
                    .unwrap_or_default();

                // Find which node was running this job
                let nodes = self
                    .nodes
                    .read()
                    .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on nodes"))?;
                let node_id = nodes
                    .iter()
                    .find(|(_, node)| node.running_jobs.contains(&job_id.to_string()))
                    .map(|(id, _)| id.clone());

                (node_id, duration)
            } else {
                return Err(anyhow::anyhow!("Job not found: {}", job_id));
            }
        };

        // Free up node resources
        if let Some(node_id) = node_id {
            let mut nodes = self
                .nodes
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on nodes"))?;

            if let Some(node) = nodes.get_mut(&node_id) {
                node.running_jobs.retain(|id| id != job_id);

                // Free allocated resources (simplified)
                let jobs = self
                    .jobs
                    .read()
                    .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on jobs"))?;
                if let Some(job) = jobs.get(job_id) {
                    node.allocated_resources.cpu_cores = node
                        .allocated_resources
                        .cpu_cores
                        .saturating_sub(job.resource_requirements.cpu_cores);
                    node.allocated_resources.memory_gb = node
                        .allocated_resources
                        .memory_gb
                        .saturating_sub(job.resource_requirements.memory_gb);
                    node.allocated_resources.gpu_count = node
                        .allocated_resources
                        .gpu_count
                        .saturating_sub(job.resource_requirements.gpu_count);
                }
            }
        }

        // Update statistics
        {
            let mut stats = self
                .statistics
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on statistics"))?;
            stats.jobs_completed += 1;
            stats.active_jobs = stats.active_jobs.saturating_sub(1);
            stats.average_completion_time = (stats.average_completion_time + job_duration) / 2;
        }

        // Send event
        self.job_sender
            .send(JobEvent::JobCompleted(job_id.to_string()))
            .map_err(|_| anyhow::anyhow!("Failed to send job completed event"))?;

        Ok(())
    }

    pub fn fail_job(&self, job_id: &str, error_message: String) -> Result<()> {
        let mut should_retry = false;

        // Update job status
        {
            let mut jobs = self
                .jobs
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on jobs"))?;

            if let Some(job) = jobs.get_mut(job_id) {
                job.retry_count += 1;

                if job.retry_count <= job.max_retries {
                    job.status = JobStatus::Retrying;
                    should_retry = true;
                } else {
                    job.status = JobStatus::Failed;
                    job.completed_at = Some(
                        SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .expect("SystemTime should be after UNIX_EPOCH")
                            .as_secs(),
                    );
                }

                job.error_message = Some(error_message.clone());
            } else {
                return Err(anyhow::anyhow!("Job not found: {}", job_id));
            }
        }

        if should_retry {
            // Add back to queue for retry
            let mut queue = self
                .job_queue
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on job queue"))?;
            queue.push_back(job_id.to_string());
        } else {
            // Update statistics for permanent failure
            let mut stats = self
                .statistics
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on statistics"))?;
            stats.jobs_failed += 1;
            stats.active_jobs = stats.active_jobs.saturating_sub(1);
        }

        // Send event
        self.job_sender
            .send(JobEvent::JobFailed(job_id.to_string(), error_message))
            .map_err(|_| anyhow::anyhow!("Failed to send job failed event"))?;

        Ok(())
    }

    pub fn update_job_progress(&self, job_id: &str, progress: f32) -> Result<()> {
        let mut jobs = self
            .jobs
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on jobs"))?;

        if let Some(job) = jobs.get_mut(job_id) {
            job.progress = progress.clamp(0.0, 100.0);
        } else {
            return Err(anyhow::anyhow!("Job not found: {}", job_id));
        }

        // Send event
        self.job_sender
            .send(JobEvent::ProgressUpdate(job_id.to_string(), progress))
            .map_err(|_| anyhow::anyhow!("Failed to send progress update event"))?;

        Ok(())
    }

    pub fn update_job_metrics(&self, job_id: &str, metrics: TrainingMetrics) -> Result<()> {
        let mut jobs = self
            .jobs
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on jobs"))?;

        if let Some(job) = jobs.get_mut(job_id) {
            job.metrics = metrics.clone();
        } else {
            return Err(anyhow::anyhow!("Job not found: {}", job_id));
        }

        // Send event
        self.job_sender
            .send(JobEvent::MetricsUpdate(job_id.to_string(), metrics))
            .map_err(|_| anyhow::anyhow!("Failed to send metrics update event"))?;

        Ok(())
    }

    pub fn get_job(&self, job_id: &str) -> Result<TrainingJob> {
        let jobs = self
            .jobs
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on jobs"))?;

        jobs.get(job_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Job not found: {}", job_id))
    }

    pub fn list_jobs(&self, status_filter: Option<JobStatus>) -> Result<Vec<TrainingJob>> {
        let jobs = self
            .jobs
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on jobs"))?;

        let filtered_jobs: Vec<_> = jobs
            .values()
            .filter(|job| {
                status_filter.as_ref().map_or(true, |status| {
                    std::mem::discriminant(&job.status) == std::mem::discriminant(status)
                })
            })
            .cloned()
            .collect();

        Ok(filtered_jobs)
    }

    pub fn get_statistics(&self) -> Result<OrchestrationStatistics> {
        let stats = self
            .statistics
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on statistics"))?;
        Ok((*stats).clone())
    }

    pub fn get_queue_status(&self) -> Result<Vec<String>> {
        let queue = self
            .job_queue
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on job queue"))?;
        Ok(queue.iter().cloned().collect())
    }

    pub fn get_node_status(&self) -> Result<Vec<ResourceNode>> {
        let nodes = self
            .nodes
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on nodes"))?;
        Ok(nodes.values().cloned().collect())
    }
}

impl ResourceAllocator {
    fn find_suitable_node(
        &self,
        requirements: &ResourceRequirements,
        nodes: &HashMap<String, ResourceNode>,
    ) -> Option<String> {
        let suitable_nodes: Vec<_> = nodes
            .iter()
            .filter(|(_, node)| self.can_allocate_resources(requirements, node))
            .collect();

        if suitable_nodes.is_empty() {
            return None;
        }

        match self.allocation_strategy {
            AllocationStrategy::FirstFit => suitable_nodes.first().map(|(id, _)| (*id).clone()),
            AllocationStrategy::BestFit => suitable_nodes
                .iter()
                .min_by_key(|(_, node)| self.calculate_remaining_resources(requirements, node))
                .map(|(id, _)| (*id).clone()),
            AllocationStrategy::WorstFit => suitable_nodes
                .iter()
                .max_by_key(|(_, node)| self.calculate_remaining_resources(requirements, node))
                .map(|(id, _)| (*id).clone()),
            AllocationStrategy::LoadBalanced => suitable_nodes
                .iter()
                .min_by_key(|(_, node)| node.running_jobs.len())
                .map(|(id, _)| (*id).clone()),
        }
    }

    fn can_allocate_resources(
        &self,
        requirements: &ResourceRequirements,
        node: &ResourceNode,
    ) -> bool {
        matches!(node.status, NodeStatus::Available)
            && node.available_resources.cpu_cores
                >= node.allocated_resources.cpu_cores + requirements.cpu_cores
            && node.available_resources.memory_gb
                >= node.allocated_resources.memory_gb + requirements.memory_gb
            && node.available_resources.gpu_count
                >= node.allocated_resources.gpu_count + requirements.gpu_count
            && node.available_resources.storage_gb >= requirements.storage_gb
    }

    fn calculate_remaining_resources(
        &self,
        requirements: &ResourceRequirements,
        node: &ResourceNode,
    ) -> u32 {
        let remaining_cpu = node.available_resources.cpu_cores
            - node.allocated_resources.cpu_cores
            - requirements.cpu_cores;
        let remaining_memory = node.available_resources.memory_gb
            - node.allocated_resources.memory_gb
            - requirements.memory_gb;
        let remaining_gpu = node.available_resources.gpu_count
            - node.allocated_resources.gpu_count
            - requirements.gpu_count;

        remaining_cpu + remaining_memory + remaining_gpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_orchestrator_creation() {
        let orchestrator =
            TrainingOrchestrator::new(SchedulingStrategy::FIFO, AllocationStrategy::FirstFit);

        let stats = orchestrator.get_statistics().unwrap();
        assert_eq!(stats.total_jobs_submitted, 0);
    }

    #[test]
    fn test_job_submission() {
        let orchestrator =
            TrainingOrchestrator::new(SchedulingStrategy::FIFO, AllocationStrategy::FirstFit);

        let job = TrainingJob {
            job_id: String::new(),
            job_name: "test_job".to_string(),
            model_config: ModelConfig {
                model_type: "transformer".to_string(),
                model_size: "small".to_string(),
                architecture_params: HashMap::new(),
                pretrained_model: None,
            },
            training_config: TrainingJobConfig {
                dataset_path: "/path/to/dataset".to_string(),
                validation_split: 0.2,
                batch_size: 32,
                learning_rate: 0.001,
                epochs: 10,
                optimizer: "Adam".to_string(),
                loss_function: "CrossEntropy".to_string(),
                regularization: HashMap::new(),
                early_stopping: None,
                checkpointing: CheckpointConfig {
                    save_frequency: 1,
                    keep_best_only: false,
                    save_weights_only: false,
                    monitor_metric: "loss".to_string(),
                },
            },
            resource_requirements: ResourceRequirements {
                cpu_cores: 4,
                memory_gb: 8,
                gpu_count: 1,
                gpu_memory_gb: 8,
                storage_gb: 100,
                network_bandwidth_mbps: None,
                node_requirements: vec![],
            },
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            created_at: 0,
            started_at: None,
            completed_at: None,
            error_message: None,
            progress: 0.0,
            checkpoints: vec![],
            metrics: TrainingMetrics {
                current_epoch: 0,
                loss: 0.0,
                accuracy: 0.0,
                validation_loss: 0.0,
                validation_accuracy: 0.0,
                learning_rate: 0.001,
                throughput: 0.0,
                custom_metrics: HashMap::new(),
            },
            dependencies: vec![],
            retry_count: 0,
            max_retries: 3,
        };

        let job_id = orchestrator.submit_job(job).unwrap();
        assert!(!job_id.is_empty());

        let retrieved_job = orchestrator.get_job(&job_id).unwrap();
        assert_eq!(retrieved_job.job_name, "test_job");
        assert_eq!(retrieved_job.status, JobStatus::Pending);
    }

    #[test]
    fn test_node_registration() {
        let orchestrator =
            TrainingOrchestrator::new(SchedulingStrategy::FIFO, AllocationStrategy::FirstFit);

        let node = ResourceNode {
            node_id: "node1".to_string(),
            node_name: "Test Node".to_string(),
            available_resources: ResourceRequirements {
                cpu_cores: 16,
                memory_gb: 64,
                gpu_count: 4,
                gpu_memory_gb: 32,
                storage_gb: 1000,
                network_bandwidth_mbps: Some(1000),
                node_requirements: vec![],
            },
            allocated_resources: ResourceRequirements {
                cpu_cores: 0,
                memory_gb: 0,
                gpu_count: 0,
                gpu_memory_gb: 0,
                storage_gb: 0,
                network_bandwidth_mbps: None,
                node_requirements: vec![],
            },
            status: NodeStatus::Available,
            last_heartbeat: Instant::now(),
            running_jobs: vec![],
        };

        orchestrator.register_node(node).unwrap();

        let nodes = orchestrator.get_node_status().unwrap();
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].node_id, "node1");
    }

    #[test]
    fn test_job_cancellation() {
        let orchestrator =
            TrainingOrchestrator::new(SchedulingStrategy::FIFO, AllocationStrategy::FirstFit);

        let job = TrainingJob {
            job_id: String::new(),
            job_name: "test_job".to_string(),
            model_config: ModelConfig {
                model_type: "transformer".to_string(),
                model_size: "small".to_string(),
                architecture_params: HashMap::new(),
                pretrained_model: None,
            },
            training_config: TrainingJobConfig {
                dataset_path: "/path/to/dataset".to_string(),
                validation_split: 0.2,
                batch_size: 32,
                learning_rate: 0.001,
                epochs: 10,
                optimizer: "Adam".to_string(),
                loss_function: "CrossEntropy".to_string(),
                regularization: HashMap::new(),
                early_stopping: None,
                checkpointing: CheckpointConfig {
                    save_frequency: 1,
                    keep_best_only: false,
                    save_weights_only: false,
                    monitor_metric: "loss".to_string(),
                },
            },
            resource_requirements: ResourceRequirements {
                cpu_cores: 4,
                memory_gb: 8,
                gpu_count: 1,
                gpu_memory_gb: 8,
                storage_gb: 100,
                network_bandwidth_mbps: None,
                node_requirements: vec![],
            },
            priority: JobPriority::Normal,
            status: JobStatus::Pending,
            created_at: 0,
            started_at: None,
            completed_at: None,
            error_message: None,
            progress: 0.0,
            checkpoints: vec![],
            metrics: TrainingMetrics {
                current_epoch: 0,
                loss: 0.0,
                accuracy: 0.0,
                validation_loss: 0.0,
                validation_accuracy: 0.0,
                learning_rate: 0.001,
                throughput: 0.0,
                custom_metrics: HashMap::new(),
            },
            dependencies: vec![],
            retry_count: 0,
            max_retries: 3,
        };

        let job_id = orchestrator.submit_job(job).unwrap();
        orchestrator.cancel_job(&job_id).unwrap();

        let retrieved_job = orchestrator.get_job(&job_id).unwrap();
        assert_eq!(retrieved_job.status, JobStatus::Cancelled);
    }
}
