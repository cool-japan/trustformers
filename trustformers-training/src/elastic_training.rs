/// Elastic training capabilities for dynamic scaling and fault tolerance
///
/// This module provides advanced distributed training bookkeeping:
/// - A local registry of workers that self-register via `register_worker`
///   and report in via `update_heartbeat`
/// - Scale-up/scale-down/rebalance *decisions* computed from that registry
/// - Checkpoint bookkeeping for fault tolerance
///
/// # No cluster substrate
///
/// [`ElasticTrainingCoordinator`] does not itself start, stop, or otherwise
/// control worker processes: there is no cluster substrate (Kubernetes,
/// Slurm, a cloud autoscaling group, ...) wired into this crate. Without a
/// [`WorkerProvisioner`] attached via
/// [`ElasticTrainingCoordinator::with_provisioner`], `scale_up`, the
/// termination step of `scale_down`, `rebalance_workers`, and checkpoint
/// recovery all return [`ElasticTrainingError::NoProvisioner`] rather than
/// reporting success for something that did not happen. Everything else
/// (worker registration, heartbeats, the in-memory worker/checkpoint
/// registries, and the scaling *decisions* themselves) is real, local
/// bookkeeping and works with no provisioner at all.
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use thiserror::Error;
use trustformers_core::tensor::Tensor;

/// Configuration for elastic training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElasticTrainingConfig {
    /// Minimum number of workers
    pub min_workers: usize,
    /// Maximum number of workers
    pub max_workers: usize,
    /// Enable dynamic scaling
    pub dynamic_scaling: bool,
    /// Enable fault tolerance
    pub fault_tolerance: bool,
    /// Scaling threshold based on throughput
    pub scale_up_threshold: f32,
    /// Scaling threshold for scaling down
    pub scale_down_threshold: f32,
    /// Checkpoint interval for fault tolerance
    pub checkpoint_interval: Duration,
    /// Maximum failed attempts before worker removal
    pub max_failed_attempts: usize,
    /// Heartbeat interval for health monitoring
    pub heartbeat_interval: Duration,
    /// Resource monitoring interval
    pub resource_monitor_interval: Duration,
    /// Enable load balancing
    pub load_balancing: bool,
    /// Enable heterogeneous hardware support
    pub heterogeneous_support: bool,
}

impl Default for ElasticTrainingConfig {
    fn default() -> Self {
        Self {
            min_workers: 1,
            max_workers: 16,
            dynamic_scaling: true,
            fault_tolerance: true,
            scale_up_threshold: 0.8,
            scale_down_threshold: 0.3,
            checkpoint_interval: Duration::from_secs(300), // 5 minutes
            max_failed_attempts: 3,
            heartbeat_interval: Duration::from_secs(30),
            resource_monitor_interval: Duration::from_secs(60),
            load_balancing: true,
            heterogeneous_support: false,
        }
    }
}

/// Worker information and status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub worker_id: String,
    pub rank: usize,
    pub status: WorkerStatus,
    #[serde(skip, default = "Instant::now")]
    pub last_heartbeat: Instant,
    pub hardware_info: HardwareInfo,
    pub performance_metrics: WorkerPerformanceMetrics,
    pub failed_attempts: usize,
    pub workload: f32,
}

/// Worker status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerStatus {
    Active,
    Idle,
    Failed,
    Scaling,
    Recovering,
    Shutdown,
}

/// Hardware information for heterogeneous support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub gpu_count: usize,
    pub gpu_memory: usize,
    pub cpu_cores: usize,
    pub ram: usize,
    pub network_bandwidth: f32,
    pub compute_capability: f32,
}

/// Performance metrics for each worker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerPerformanceMetrics {
    pub throughput: f32,
    pub latency: Duration,
    pub memory_usage: f32,
    pub cpu_usage: f32,
    pub gpu_utilization: f32,
    pub network_usage: f32,
    /// Self-reported current workload (e.g. queued batches / assigned
    /// shard fraction), on whatever scale the caller's workers use
    /// consistently with each other. `update_heartbeat` mirrors this into
    /// the coordinator's per-worker registry, which is what
    /// `should_rebalance` reads to detect imbalance.
    pub workload: f32,
}

impl Default for WorkerPerformanceMetrics {
    fn default() -> Self {
        Self {
            throughput: 0.0,
            latency: Duration::from_secs(0),
            memory_usage: 0.0,
            cpu_usage: 0.0,
            gpu_utilization: 0.0,
            network_usage: 0.0,
            workload: 0.0,
        }
    }
}

/// Something that can actually provision, terminate, restore, and rebalance
/// worker processes on a real cluster substrate (Kubernetes, Slurm, a cloud
/// autoscaling group, an in-house fleet manager, ...).
///
/// [`ElasticTrainingCoordinator`] has no such substrate of its own -- see
/// the module docs. Attach an implementation via
/// [`ElasticTrainingCoordinator::with_provisioner`] to make `scale_up`, the
/// termination step of `scale_down`, `rebalance_workers`, and checkpoint
/// recovery actually act instead of returning
/// [`ElasticTrainingError::NoProvisioner`].
pub trait WorkerProvisioner: Send + Sync {
    /// Request `count` additional worker processes. Returns the ids of the
    /// workers that were *actually* started -- implementations must not
    /// fabricate ids for processes that were not actually started, and may
    /// return fewer than `count` if capacity is limited. The started
    /// workers are expected to call `register_worker` themselves once they
    /// come online; this call only requests capacity, it does not update
    /// the coordinator's registry.
    fn provision_workers(&self, count: usize) -> Result<Vec<String>>;

    /// Terminate the given worker processes. `worker_ids` have already been
    /// selected by the coordinator (idle workers, preferentially). Return
    /// `Err` if any could not be confirmed terminated -- the coordinator
    /// only forgets about workers whose termination this call confirmed
    /// with `Ok(())`.
    fn terminate_workers(&self, worker_ids: &[String]) -> Result<()>;

    /// Restore `worker_id`'s training state from `checkpoint` (e.g. by
    /// pushing the checkpoint to a replacement process, or restarting the
    /// worker with `--resume-from`). Returning `Ok(())` is taken as
    /// confirmation the worker's state was actually restored.
    fn restore_worker(&self, worker_id: &str, checkpoint: &CheckpointInfo) -> Result<()>;

    /// Ask the real workers named in `worker_ids` (all currently `Active`)
    /// to rebalance workload amongst themselves. What "rebalance" means in
    /// practice is entirely up to the implementation; the coordinator only
    /// tracks that a rebalance was requested and confirmed.
    fn rebalance_workers(&self, worker_ids: &[String]) -> Result<()>;
}

/// Errors from [`ElasticTrainingCoordinator`] operations that need
/// capabilities the coordinator does not have by itself.
#[derive(Debug, Error)]
pub enum ElasticTrainingError {
    /// The requested operation needs a [`WorkerProvisioner`] and none was
    /// attached via [`ElasticTrainingCoordinator::with_provisioner`].
    #[error("{operation}: {detail} (no WorkerProvisioner configured)")]
    NoProvisioner {
        operation: &'static str,
        detail: String,
    },
    /// `scale_up` asked its [`WorkerProvisioner`] for `requested` workers
    /// but only `provisioned` were actually started. Reported as a failure
    /// rather than silently accepting a partial scale-up as success.
    #[error(
        "scale_up requested {requested} worker(s) but the WorkerProvisioner \
         only started {provisioned}"
    )]
    PartialProvisioning {
        requested: usize,
        provisioned: usize,
    },
}

/// Scaling decision information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingDecision {
    pub decision_type: ScalingType,
    pub target_workers: usize,
    pub reason: String,
    pub confidence: f32,
    pub estimated_benefit: f32,
}

/// Types of scaling decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScalingType {
    ScaleUp,
    ScaleDown,
    NoChange,
    Rebalance,
}

/// Elastic training coordinator
///
/// Manages exactly one thing for real: an in-memory registry of workers
/// that have called `register_worker`, their checkpoints, and a log of
/// scaling decisions. It does not manage any external fleet by itself --
/// see the module docs and [`WorkerProvisioner`].
pub struct ElasticTrainingCoordinator {
    config: ElasticTrainingConfig,
    workers: Arc<Mutex<HashMap<String, WorkerInfo>>>,
    checkpoints: HashMap<String, CheckpointInfo>,
    scaling_history: Vec<ScalingEvent>,
    /// Optional real cluster substrate. `None` means this coordinator only
    /// tracks self-registered workers and cannot provision, terminate,
    /// restore, or rebalance real processes.
    provisioner: Option<Arc<dyn WorkerProvisioner>>,
}

impl ElasticTrainingCoordinator {
    pub fn new(config: ElasticTrainingConfig) -> Self {
        Self {
            config,
            workers: Arc::new(Mutex::new(HashMap::new())),
            checkpoints: HashMap::new(),
            scaling_history: Vec::new(),
            provisioner: None,
        }
    }

    /// Attach a [`WorkerProvisioner`] so `scale_up`, the termination step of
    /// `scale_down`, `rebalance_workers`, and checkpoint recovery can act on
    /// a real cluster substrate instead of returning
    /// [`ElasticTrainingError::NoProvisioner`].
    #[must_use]
    pub fn with_provisioner(mut self, provisioner: Arc<dyn WorkerProvisioner>) -> Self {
        self.provisioner = Some(provisioner);
        self
    }

    /// Register a new worker
    pub fn register_worker(
        &mut self,
        worker_id: String,
        hardware_info: HardwareInfo,
    ) -> Result<usize> {
        let mut workers = self.workers.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        let rank = workers.len();
        let worker_info = WorkerInfo {
            worker_id: worker_id.clone(),
            rank,
            status: WorkerStatus::Active,
            last_heartbeat: Instant::now(),
            hardware_info,
            performance_metrics: WorkerPerformanceMetrics::default(),
            failed_attempts: 0,
            workload: 0.0,
        };

        workers.insert(worker_id, worker_info);

        tracing::info!("Registered worker with rank {}", rank);
        Ok(rank)
    }

    /// Update worker heartbeat
    pub fn update_heartbeat(
        &mut self,
        worker_id: &str,
        metrics: WorkerPerformanceMetrics,
    ) -> Result<()> {
        let mut workers = self.workers.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        if let Some(worker) = workers.get_mut(worker_id) {
            worker.last_heartbeat = Instant::now();
            worker.workload = metrics.workload;
            worker.performance_metrics = metrics;
            worker.failed_attempts = 0; // Reset on successful heartbeat

            // Update status based on performance
            if worker.performance_metrics.throughput > 0.0 {
                worker.status = WorkerStatus::Active;
            } else {
                worker.status = WorkerStatus::Idle;
            }
        }

        Ok(())
    }

    /// Monitor workers and detect failures
    pub fn monitor_workers(&mut self) -> Result<Vec<String>> {
        let mut failed_workers = Vec::new();
        let now = Instant::now();

        {
            let mut workers = self.workers.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

            for (worker_id, worker) in workers.iter_mut() {
                // Check heartbeat timeout
                if now.duration_since(worker.last_heartbeat) > self.config.heartbeat_interval * 2 {
                    worker.failed_attempts += 1;

                    if worker.failed_attempts >= self.config.max_failed_attempts {
                        worker.status = WorkerStatus::Failed;
                        failed_workers.push(worker_id.clone());
                    }
                }
            }
        }

        // Handle failed workers
        for worker_id in &failed_workers {
            self.handle_worker_failure(worker_id)?;
        }

        Ok(failed_workers)
    }

    /// Handle worker failure
    fn handle_worker_failure(&mut self, worker_id: &str) -> Result<()> {
        if !self.config.fault_tolerance {
            return Ok(());
        }

        tracing::warn!("Handling failure for worker: {}", worker_id);

        // Attempt real recovery only if a checkpoint exists. Recovery
        // itself needs a `WorkerProvisioner` that can actually push state
        // to a (possibly replacement) process; without one we cannot claim
        // recovery happened. Log exactly why and continue with the
        // bookkeeping this coordinator CAN honestly do on its own --
        // deregistering the confirmed-dead worker and re-evaluating
        // scaling -- rather than aborting the whole failure-handling path.
        if let Some(checkpoint) = self.checkpoints.get(worker_id).cloned() {
            if let Err(e) = self.recover_from_checkpoint(worker_id, &checkpoint) {
                tracing::warn!(
                    "Worker {} was NOT recovered from checkpoint (step {}): {}. Its \
                     in-flight state is lost; the worker is being deregistered.",
                    worker_id,
                    checkpoint.step,
                    e
                );
            }
        }

        // Remove failed worker from active set
        {
            let mut workers = self.workers.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            workers.remove(worker_id);
        }

        // Trigger scaling if needed
        if self.config.dynamic_scaling {
            self.evaluate_scaling_decision()?;
        }

        Ok(())
    }

    /// Evaluate whether scaling is needed
    pub fn evaluate_scaling_decision(&mut self) -> Result<Option<ScalingDecision>> {
        if !self.config.dynamic_scaling {
            return Ok(None);
        }

        let workers = self.workers.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let active_workers =
            workers.iter().filter(|(_, w)| matches!(w.status, WorkerStatus::Active)).count();
        // `calculate_system_performance` and `should_rebalance` below both
        // take `self.workers.lock()` too; holding this guard across those
        // calls would deadlock (`Mutex` has no reentrancy at all, unlike an
        // `RwLock` this is not even contention-dependent).
        drop(workers);

        if active_workers < self.config.min_workers {
            return Ok(Some(ScalingDecision {
                decision_type: ScalingType::ScaleUp,
                target_workers: self.config.min_workers,
                reason: "Below minimum worker count".to_string(),
                confidence: 1.0,
                estimated_benefit: 0.5,
            }));
        }

        if active_workers > self.config.max_workers {
            return Ok(Some(ScalingDecision {
                decision_type: ScalingType::ScaleDown,
                target_workers: self.config.max_workers,
                reason: "Above maximum worker count".to_string(),
                confidence: 1.0,
                estimated_benefit: 0.3,
            }));
        }

        // Calculate system performance
        let system_performance = self.calculate_system_performance();

        // Check for scale-up conditions
        if system_performance.overall_utilization > self.config.scale_up_threshold
            && active_workers < self.config.max_workers
        {
            return Ok(Some(ScalingDecision {
                decision_type: ScalingType::ScaleUp,
                target_workers: (active_workers + 1).min(self.config.max_workers),
                reason: format!(
                    "High utilization: {:.2}",
                    system_performance.overall_utilization
                ),
                confidence: 0.8,
                estimated_benefit: 0.6,
            }));
        }

        // Check for scale-down conditions
        if system_performance.overall_utilization < self.config.scale_down_threshold
            && active_workers > self.config.min_workers
        {
            return Ok(Some(ScalingDecision {
                decision_type: ScalingType::ScaleDown,
                target_workers: (active_workers - 1).max(self.config.min_workers),
                reason: format!(
                    "Low utilization: {:.2}",
                    system_performance.overall_utilization
                ),
                confidence: 0.7,
                estimated_benefit: 0.4,
            }));
        }

        // Check for rebalancing needs
        if self.config.load_balancing && self.should_rebalance() {
            return Ok(Some(ScalingDecision {
                decision_type: ScalingType::Rebalance,
                target_workers: active_workers,
                reason: "Load imbalance detected".to_string(),
                confidence: 0.6,
                estimated_benefit: 0.3,
            }));
        }

        Ok(None)
    }

    /// Calculate system performance metrics
    fn calculate_system_performance(&self) -> SystemPerformanceSnapshot {
        let workers = self.workers.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let active_workers: Vec<_> = workers
            .iter()
            .filter(|(_, w)| matches!(w.status, WorkerStatus::Active))
            .collect();

        if active_workers.is_empty() {
            return SystemPerformanceSnapshot::default();
        }

        let total_throughput: f32 =
            active_workers.iter().map(|(_, w)| w.performance_metrics.throughput).sum();

        let avg_latency = Duration::from_secs_f32(
            active_workers
                .iter()
                .map(|(_, w)| w.performance_metrics.latency.as_secs_f32())
                .sum::<f32>()
                / active_workers.len() as f32,
        );

        let avg_utilization = active_workers
            .iter()
            .map(|(_, w)| {
                (w.performance_metrics.cpu_usage + w.performance_metrics.gpu_utilization) / 2.0
            })
            .sum::<f32>()
            / active_workers.len() as f32;

        SystemPerformanceSnapshot {
            timestamp: Instant::now(),
            active_workers: active_workers.len(),
            total_throughput,
            average_latency: avg_latency,
            overall_utilization: avg_utilization,
            memory_usage: active_workers
                .iter()
                .map(|(_, w)| w.performance_metrics.memory_usage)
                .sum::<f32>()
                / active_workers.len() as f32,
        }
    }

    /// Check if load rebalancing is needed
    fn should_rebalance(&self) -> bool {
        if !self.config.load_balancing {
            return false;
        }

        let workers = self.workers.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let workloads: Vec<f32> = workers
            .iter()
            .filter(|(_, w)| matches!(w.status, WorkerStatus::Active))
            .map(|(_, w)| w.workload)
            .collect();

        if workloads.len() < 2 {
            return false;
        }

        let avg_workload = workloads.iter().sum::<f32>() / workloads.len() as f32;
        let max_workload = workloads.iter().fold(0.0f32, |a, &b| a.max(b));
        let min_workload = workloads.iter().fold(f32::MAX, |a, &b| a.min(b));

        // Rebalance if there's significant imbalance
        (max_workload - min_workload) > avg_workload * 0.3
    }

    /// Execute scaling decision
    ///
    /// The recorded [`ScalingEvent::success`] reflects exactly what the
    /// underlying operation reported -- it is never fabricated. This
    /// function still returns the operation's `Err` (after recording the
    /// failed event), so callers keep the error detail.
    pub fn execute_scaling(&mut self, decision: &ScalingDecision) -> Result<()> {
        let result = match decision.decision_type {
            ScalingType::ScaleUp => self.scale_up(decision.target_workers),
            ScalingType::ScaleDown => self.scale_down(decision.target_workers),
            ScalingType::Rebalance => self.rebalance_workers(),
            ScalingType::NoChange => Ok(()),
        };

        // Record scaling event with the operation's REAL outcome.
        self.scaling_history.push(ScalingEvent {
            timestamp: Instant::now(),
            decision: decision.clone(),
            success: result.is_ok(),
        });

        result
    }

    /// Scale up workers.
    ///
    /// Requires a [`WorkerProvisioner`] (see [`Self::with_provisioner`]):
    /// this coordinator has no cluster substrate of its own to request new
    /// worker instances from. Without one, returns
    /// [`ElasticTrainingError::NoProvisioner`] instead of reporting success
    /// for workers that were never requested.
    fn scale_up(&mut self, target_workers: usize) -> Result<()> {
        let current_workers =
            self.workers.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).len();
        let workers_to_add = target_workers.saturating_sub(current_workers);

        if workers_to_add == 0 {
            return Ok(());
        }

        let provisioner =
            self.provisioner.as_ref().ok_or_else(|| ElasticTrainingError::NoProvisioner {
                operation: "scale_up",
                detail: format!(
                    "cannot request {workers_to_add} new worker instance(s): this coordinator \
                 only tracks workers that self-register via register_worker() and has no \
                 cluster substrate of its own"
                ),
            })?;

        tracing::info!(
            "Scaling up: requesting {} worker(s) from provisioner",
            workers_to_add
        );
        let requested = provisioner.provision_workers(workers_to_add)?;

        if requested.len() < workers_to_add {
            return Err(ElasticTrainingError::PartialProvisioning {
                requested: workers_to_add,
                provisioned: requested.len(),
            }
            .into());
        }

        tracing::info!(
            "Provisioner started {} worker instance(s); they must call register_worker() \
             once online for the coordinator to see them",
            requested.len()
        );

        Ok(())
    }

    /// Scale down workers.
    ///
    /// Always deregisters the selected (idle) workers from this
    /// coordinator's own registry -- that is real, local bookkeeping the
    /// coordinator can do unconditionally. It only *terminates* the
    /// underlying processes when a [`WorkerProvisioner`] is attached (see
    /// [`Self::with_provisioner`]); without one, the processes (if any are
    /// real) keep running, and this logs a warning saying so rather than
    /// silently claiming they were shut down.
    fn scale_down(&mut self, target_workers: usize) -> Result<()> {
        let workers_to_remove_ids: Vec<String> = {
            let workers = self.workers.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let current_workers = workers.len();
            let workers_to_remove = current_workers.saturating_sub(target_workers);

            // Select workers to remove (prefer idle workers)
            workers
                .iter()
                .filter(|(_, w)| matches!(w.status, WorkerStatus::Idle))
                .take(workers_to_remove)
                .map(|(id, _)| id.clone())
                .collect()
        };

        if workers_to_remove_ids.is_empty() {
            return Ok(());
        }

        tracing::info!(
            "Scaling down: removing {} worker(s)",
            workers_to_remove_ids.len()
        );

        // Confirm real termination BEFORE forgetting about them locally --
        // otherwise a failed/absent terminate call would leave real
        // processes running but untracked, which is worse than not
        // scaling down at all.
        match self.provisioner.as_ref() {
            Some(provisioner) => provisioner.terminate_workers(&workers_to_remove_ids)?,
            None => tracing::warn!(
                "Deregistering {} idle worker(s) but no WorkerProvisioner is configured: \
                 their underlying processes (if any) will NOT be terminated, only forgotten \
                 by this coordinator.",
                workers_to_remove_ids.len()
            ),
        }

        let mut workers = self.workers.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        for worker_id in &workers_to_remove_ids {
            workers.remove(worker_id);
        }

        Ok(())
    }

    /// Rebalance workload across active workers.
    ///
    /// Requires a [`WorkerProvisioner`] (see [`Self::with_provisioner`]):
    /// actually redistributing work means telling real worker processes to
    /// change what they're doing, which this coordinator cannot do on its
    /// own. Without one, returns [`ElasticTrainingError::NoProvisioner`]
    /// instead of reporting a rebalance that never reached any worker.
    fn rebalance_workers(&mut self) -> Result<()> {
        let active_ids: Vec<String> = {
            let workers = self.workers.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            workers
                .iter()
                .filter(|(_, w)| matches!(w.status, WorkerStatus::Active))
                .map(|(id, _)| id.clone())
                .collect()
        };

        let provisioner =
            self.provisioner.as_ref().ok_or_else(|| ElasticTrainingError::NoProvisioner {
                operation: "rebalance_workers",
                detail: format!(
                    "cannot redistribute workload across {} active worker(s): this coordinator \
                 has no channel to real worker processes without a WorkerProvisioner",
                    active_ids.len()
                ),
            })?;

        provisioner.rebalance_workers(&active_ids)?;
        tracing::info!(
            "Requested rebalance across {} active worker(s)",
            active_ids.len()
        );

        Ok(())
    }

    /// Create checkpoint for fault tolerance.
    ///
    /// `step` is the caller's real current training step -- it is recorded
    /// as-is, never guessed or hardcoded, so callers own the responsibility
    /// of passing their actual step counter.
    pub fn create_checkpoint(
        &mut self,
        worker_id: &str,
        step: usize,
        model_state: HashMap<String, Tensor>,
    ) -> Result<()> {
        if !self.config.fault_tolerance {
            return Ok(());
        }

        let checkpoint = CheckpointInfo {
            timestamp: Instant::now(),
            worker_id: worker_id.to_string(),
            model_state,
            step,
        };

        self.checkpoints.insert(worker_id.to_string(), checkpoint);
        tracing::info!(
            "Created checkpoint for worker {} at step {}",
            worker_id,
            step
        );

        Ok(())
    }

    /// Recover a worker's training state from `checkpoint`.
    ///
    /// Requires a [`WorkerProvisioner`] (see [`Self::with_provisioner`]):
    /// this coordinator cannot push `checkpoint`'s state to `worker_id` (or
    /// to a replacement process) on its own. Without one, returns
    /// [`ElasticTrainingError::NoProvisioner`] instead of reporting a
    /// recovery that never restored anything.
    fn recover_from_checkpoint(
        &mut self,
        worker_id: &str,
        checkpoint: &CheckpointInfo,
    ) -> Result<()> {
        let provisioner =
            self.provisioner.as_ref().ok_or_else(|| ElasticTrainingError::NoProvisioner {
                operation: "recover_from_checkpoint",
                detail: format!(
                    "cannot restore worker {worker_id}'s state from checkpoint step {} without \
                 a WorkerProvisioner",
                    checkpoint.step
                ),
            })?;

        provisioner.restore_worker(worker_id, checkpoint)?;
        tracing::info!(
            "Recovered worker {} from checkpoint (step {})",
            worker_id,
            checkpoint.step
        );

        Ok(())
    }

    /// Get current system status.
    ///
    /// Reflects exactly this coordinator's own registry -- workers that
    /// called `register_worker` and have (or have not) sent recent
    /// heartbeats. It says nothing about any external fleet; if a
    /// [`WorkerProvisioner`] is managing real processes elsewhere, this
    /// coordinator only knows about the ones that registered.
    pub fn get_system_status(&self) -> SystemStatus {
        // Calculate performance first to avoid deadlock (it also acquires workers lock)
        let performance = self.calculate_system_performance();

        // Now acquire workers lock for remaining stats
        let workers = self.workers.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let active_count =
            workers.iter().filter(|(_, w)| matches!(w.status, WorkerStatus::Active)).count();

        SystemStatus {
            total_workers: workers.len(),
            active_workers: active_count,
            failed_workers: workers
                .iter()
                .filter(|(_, w)| matches!(w.status, WorkerStatus::Failed))
                .count(),
            scaling_in_progress: workers
                .iter()
                .any(|(_, w)| matches!(w.status, WorkerStatus::Scaling)),
            performance_snapshot: performance,
            fault_tolerance_enabled: self.config.fault_tolerance,
            dynamic_scaling_enabled: self.config.dynamic_scaling,
        }
    }
}

/// Checkpoint information for fault tolerance
#[derive(Debug, Clone)]
pub struct CheckpointInfo {
    pub timestamp: Instant,
    pub worker_id: String,
    pub model_state: HashMap<String, Tensor>,
    pub step: usize,
}

/// System performance snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPerformanceSnapshot {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub active_workers: usize,
    pub total_throughput: f32,
    pub average_latency: Duration,
    pub overall_utilization: f32,
    pub memory_usage: f32,
}

impl Default for SystemPerformanceSnapshot {
    fn default() -> Self {
        Self {
            timestamp: Instant::now(),
            active_workers: 0,
            total_throughput: 0.0,
            average_latency: Duration::from_secs(0),
            overall_utilization: 0.0,
            memory_usage: 0.0,
        }
    }
}

/// Scaling event for history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingEvent {
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    pub decision: ScalingDecision,
    pub success: bool,
}

/// Overall system status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStatus {
    pub total_workers: usize,
    pub active_workers: usize,
    pub failed_workers: usize,
    pub scaling_in_progress: bool,
    pub performance_snapshot: SystemPerformanceSnapshot,
    pub fault_tolerance_enabled: bool,
    pub dynamic_scaling_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elastic_coordinator_creation() {
        let config = ElasticTrainingConfig::default();
        let coordinator = ElasticTrainingCoordinator::new(config);

        assert_eq!(
            coordinator.workers.lock().expect("lock should not be poisoned").len(),
            0
        );
        assert_eq!(coordinator.checkpoints.len(), 0);
    }

    #[test]
    fn test_worker_registration() {
        let config = ElasticTrainingConfig::default();
        let mut coordinator = ElasticTrainingCoordinator::new(config);

        let hardware_info = HardwareInfo {
            gpu_count: 1,
            gpu_memory: 8000000000,
            cpu_cores: 8,
            ram: 16000000000,
            network_bandwidth: 1000.0,
            compute_capability: 7.5,
        };

        let result = coordinator.register_worker("worker1".to_string(), hardware_info);
        assert!(result.is_ok());
        assert_eq!(result.expect("operation failed in test"), 0);
        assert_eq!(
            coordinator.workers.lock().expect("lock should not be poisoned").len(),
            1
        );
    }

    #[test]
    fn test_heartbeat_update() {
        let config = ElasticTrainingConfig::default();
        let mut coordinator = ElasticTrainingCoordinator::new(config);

        let hardware_info = HardwareInfo {
            gpu_count: 1,
            gpu_memory: 8000000000,
            cpu_cores: 8,
            ram: 16000000000,
            network_bandwidth: 1000.0,
            compute_capability: 7.5,
        };

        coordinator
            .register_worker("worker1".to_string(), hardware_info)
            .expect("operation failed in test");

        let metrics = WorkerPerformanceMetrics {
            throughput: 100.0,
            latency: Duration::from_millis(50),
            memory_usage: 0.5,
            cpu_usage: 0.6,
            gpu_utilization: 0.8,
            network_usage: 0.3,
            workload: 0.4,
        };

        let result = coordinator.update_heartbeat("worker1", metrics);
        assert!(result.is_ok());

        // The heartbeat's self-reported workload must actually land in the
        // coordinator's registry, not be silently dropped.
        let workers = coordinator.workers.lock().expect("lock should not be poisoned");
        assert_eq!(
            workers.get("worker1").expect("worker1 must be registered").workload,
            0.4
        );
    }

    #[test]
    fn test_scaling_decision() {
        let config = ElasticTrainingConfig {
            min_workers: 2,
            max_workers: 8,
            dynamic_scaling: true,
            ..Default::default()
        };
        let mut coordinator = ElasticTrainingCoordinator::new(config);

        let decision = coordinator.evaluate_scaling_decision().expect("operation failed in test");
        assert!(decision.is_some());

        let decision = decision.expect("operation failed in test");
        assert!(matches!(decision.decision_type, ScalingType::ScaleUp));
        assert_eq!(decision.target_workers, 2);
    }

    /// Regression: `evaluate_scaling_decision` used to hold its own
    /// `self.workers.lock()` guard for its whole body, including across the
    /// calls to `calculate_system_performance` and `should_rebalance`, which
    /// both also lock `self.workers`. `Mutex` has no reentrancy at all (this
    /// is not contention-dependent like an `RwLock` read-read case), so any
    /// call that fell through the below/above worker-count early returns --
    /// i.e. the normal case, active worker count within `[min, max]` --
    /// deadlocked unconditionally. `test_scaling_decision` above never
    /// reached this: with zero workers registered, `active_workers (0) <
    /// min_workers (2)` returns before ever calling
    /// `calculate_system_performance`.
    ///
    /// A genuinely deadlocked call hangs rather than erroring, so this runs
    /// it on a background thread and fails on a bounded timeout instead of
    /// hanging the whole suite.
    #[test]
    fn test_scaling_decision_in_normal_range_does_not_deadlock() {
        use std::sync::mpsc;
        use std::time::Duration;

        let config = ElasticTrainingConfig {
            min_workers: 1,
            max_workers: 8,
            dynamic_scaling: true,
            ..Default::default()
        };
        let mut coordinator = ElasticTrainingCoordinator::new(config);

        let hardware_info = HardwareInfo {
            gpu_count: 1,
            gpu_memory: 8000000000,
            cpu_cores: 8,
            ram: 16000000000,
            network_bandwidth: 1000.0,
            compute_capability: 7.5,
        };
        coordinator
            .register_worker("worker1".to_string(), hardware_info)
            .expect("register_worker failed");

        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let result = coordinator.evaluate_scaling_decision();
            // Only sent if evaluate_scaling_decision did not hang.
            let _ = tx.send(result.is_ok());
        });

        let completed = rx.recv_timeout(Duration::from_secs(10)).expect(
            "evaluate_scaling_decision must return promptly for an in-range worker \
             count, not deadlock on its own workers lock",
        );
        assert!(completed, "evaluate_scaling_decision returned an error");
    }

    #[test]
    fn test_checkpoint_creation() {
        let config = ElasticTrainingConfig::default();
        let mut coordinator = ElasticTrainingCoordinator::new(config);

        let model_state = HashMap::new();
        let result = coordinator.create_checkpoint("worker1", 42, model_state);

        assert!(result.is_ok());
        assert_eq!(coordinator.checkpoints.len(), 1);
        assert_eq!(
            coordinator.checkpoints.get("worker1").expect("checkpoint missing").step,
            42,
            "the checkpoint must carry the caller's real step, never a hardcoded 0"
        );
    }

    #[test]
    fn test_system_status() {
        let config = ElasticTrainingConfig::default();
        let coordinator = ElasticTrainingCoordinator::new(config);

        let status = coordinator.get_system_status();
        assert_eq!(status.total_workers, 0);
        assert_eq!(status.active_workers, 0);
        assert_eq!(status.failed_workers, 0);
        assert!(!status.scaling_in_progress);
    }

    // ---- Honest-contract tests for scale_up/scale_down/rebalance/recover ----

    fn test_hardware_info() -> HardwareInfo {
        HardwareInfo {
            gpu_count: 1,
            gpu_memory: 1,
            cpu_cores: 1,
            ram: 1,
            network_bandwidth: 1.0,
            compute_capability: 1.0,
        }
    }

    fn scaling_decision(decision_type: ScalingType, target_workers: usize) -> ScalingDecision {
        ScalingDecision {
            decision_type,
            target_workers,
            reason: "test".to_string(),
            confidence: 1.0,
            estimated_benefit: 0.5,
        }
    }

    /// A [`WorkerProvisioner`] test double that records every call it
    /// receives (so tests can prove it was actually invoked, not just that
    /// no error surfaced) and can be configured to under-provision.
    #[derive(Default)]
    struct MockProvisioner {
        provision_calls: Mutex<Vec<usize>>,
        terminate_calls: Mutex<Vec<Vec<String>>>,
        restore_calls: Mutex<Vec<(String, usize)>>,
        rebalance_calls: Mutex<Vec<Vec<String>>>,
        provision_shortfall: usize,
    }

    impl MockProvisioner {
        fn new() -> Self {
            Self::default()
        }

        fn with_shortfall(provision_shortfall: usize) -> Self {
            Self {
                provision_shortfall,
                ..Self::default()
            }
        }
    }

    impl WorkerProvisioner for MockProvisioner {
        fn provision_workers(&self, count: usize) -> Result<Vec<String>> {
            self.provision_calls.lock().expect("lock should not be poisoned").push(count);
            let actually_started = count.saturating_sub(self.provision_shortfall);
            Ok((0..actually_started).map(|i| format!("provisioned-{i}")).collect())
        }

        fn terminate_workers(&self, worker_ids: &[String]) -> Result<()> {
            self.terminate_calls
                .lock()
                .expect("lock should not be poisoned")
                .push(worker_ids.to_vec());
            Ok(())
        }

        fn restore_worker(&self, worker_id: &str, checkpoint: &CheckpointInfo) -> Result<()> {
            self.restore_calls
                .lock()
                .expect("lock should not be poisoned")
                .push((worker_id.to_string(), checkpoint.step));
            Ok(())
        }

        fn rebalance_workers(&self, worker_ids: &[String]) -> Result<()> {
            self.rebalance_calls
                .lock()
                .expect("lock should not be poisoned")
                .push(worker_ids.to_vec());
            Ok(())
        }
    }

    #[test]
    fn test_scale_up_without_provisioner_is_a_structured_error_not_fake_success() {
        let mut coordinator = ElasticTrainingCoordinator::new(ElasticTrainingConfig::default());

        let decision = scaling_decision(ScalingType::ScaleUp, 3);
        let result = coordinator.execute_scaling(&decision);

        let err =
            result.expect_err("scale_up with no provisioner must fail, not fabricate success");
        assert!(
            err.downcast_ref::<ElasticTrainingError>().is_some_and(|e| matches!(
                e,
                ElasticTrainingError::NoProvisioner {
                    operation: "scale_up",
                    ..
                }
            )),
            "expected ElasticTrainingError::NoProvisioner from scale_up, got: {err}"
        );
        // The failed attempt is recorded honestly as success:false, never true.
        assert_eq!(coordinator.scaling_history.len(), 1);
        assert!(!coordinator.scaling_history[0].success);
    }

    #[test]
    fn test_scale_up_with_provisioner_succeeds_and_is_actually_invoked() {
        let provisioner = Arc::new(MockProvisioner::new());
        let mut coordinator = ElasticTrainingCoordinator::new(ElasticTrainingConfig::default())
            .with_provisioner(provisioner.clone());

        let decision = scaling_decision(ScalingType::ScaleUp, 3);
        let result = coordinator.execute_scaling(&decision);

        assert!(
            result.is_ok(),
            "scale_up must succeed once a provisioner covers the request"
        );
        assert_eq!(
            provisioner
                .provision_calls
                .lock()
                .expect("lock should not be poisoned")
                .as_slice(),
            [3],
            "the provisioner must actually be asked for the workers, not bypassed"
        );
        assert!(coordinator.scaling_history[0].success);
    }

    #[test]
    fn test_scale_up_partial_provisioning_is_reported_as_failure() {
        let provisioner = Arc::new(MockProvisioner::with_shortfall(1));
        let mut coordinator = ElasticTrainingCoordinator::new(ElasticTrainingConfig::default())
            .with_provisioner(provisioner);

        let decision = scaling_decision(ScalingType::ScaleUp, 3);
        let result = coordinator.execute_scaling(&decision);

        let err = result.expect_err("provisioning only 2 of 3 requested workers must be an error");
        assert!(
            err.downcast_ref::<ElasticTrainingError>().is_some_and(|e| matches!(
                e,
                ElasticTrainingError::PartialProvisioning {
                    requested: 3,
                    provisioned: 2
                }
            )),
            "expected PartialProvisioning{{requested:3,provisioned:2}}, got: {err}"
        );
        assert!(!coordinator.scaling_history[0].success);
    }

    #[test]
    fn test_scale_down_without_provisioner_still_deregisters_idle_workers_locally() {
        let mut coordinator = ElasticTrainingCoordinator::new(ElasticTrainingConfig::default());
        coordinator
            .register_worker("worker1".to_string(), test_hardware_info())
            .expect("register_worker failed");
        // throughput 0.0 (the default) marks the worker Idle, which is what
        // scale_down selects for removal.
        coordinator
            .update_heartbeat("worker1", WorkerPerformanceMetrics::default())
            .expect("update_heartbeat failed");

        let decision = scaling_decision(ScalingType::ScaleDown, 0);
        let result = coordinator.execute_scaling(&decision);

        assert!(
            result.is_ok(),
            "scale_down's local deregistration must succeed with no provisioner"
        );
        assert_eq!(
            coordinator.workers.lock().expect("lock should not be poisoned").len(),
            0
        );
        assert!(coordinator.scaling_history[0].success);
    }

    #[test]
    fn test_scale_down_with_provisioner_terminates_the_real_workers() {
        let provisioner = Arc::new(MockProvisioner::new());
        let mut coordinator = ElasticTrainingCoordinator::new(ElasticTrainingConfig::default())
            .with_provisioner(provisioner.clone());
        coordinator
            .register_worker("worker1".to_string(), test_hardware_info())
            .expect("register_worker failed");
        coordinator
            .update_heartbeat("worker1", WorkerPerformanceMetrics::default())
            .expect("update_heartbeat failed");

        let decision = scaling_decision(ScalingType::ScaleDown, 0);
        coordinator.execute_scaling(&decision).expect("scale_down failed");

        assert_eq!(
            provisioner
                .terminate_calls
                .lock()
                .expect("lock should not be poisoned")
                .as_slice(),
            [vec!["worker1".to_string()]],
            "the provisioner must actually be asked to terminate the selected worker"
        );
    }

    #[test]
    fn test_rebalance_without_provisioner_is_a_structured_error_not_fake_success() {
        let mut coordinator = ElasticTrainingCoordinator::new(ElasticTrainingConfig::default());

        let decision = scaling_decision(ScalingType::Rebalance, 0);
        let result = coordinator.execute_scaling(&decision);

        let err =
            result.expect_err("rebalance with no provisioner must fail, not fabricate success");
        assert!(
            err.downcast_ref::<ElasticTrainingError>().is_some_and(|e| matches!(
                e,
                ElasticTrainingError::NoProvisioner {
                    operation: "rebalance_workers",
                    ..
                }
            )),
            "expected ElasticTrainingError::NoProvisioner from rebalance_workers, got: {err}"
        );
        assert!(!coordinator.scaling_history[0].success);
    }

    #[test]
    fn test_rebalance_with_provisioner_is_actually_invoked() {
        let provisioner = Arc::new(MockProvisioner::new());
        let mut coordinator = ElasticTrainingCoordinator::new(ElasticTrainingConfig::default())
            .with_provisioner(provisioner.clone());
        coordinator
            .register_worker("worker1".to_string(), test_hardware_info())
            .expect("register_worker failed");

        let decision = scaling_decision(ScalingType::Rebalance, 1);
        coordinator.execute_scaling(&decision).expect("rebalance failed");

        assert_eq!(
            provisioner
                .rebalance_calls
                .lock()
                .expect("lock should not be poisoned")
                .as_slice(),
            [vec!["worker1".to_string()]]
        );
    }

    /// Regression: `workload` used to only ever be set to 0.0 at
    /// registration and nothing ever updated it afterwards, so
    /// `should_rebalance`'s imbalance check (`max - min > avg * 0.3`) was
    /// structurally always false (0.0 - 0.0 = 0.0 for every worker,
    /// forever) -- the same "detector that can never fire" shape as the
    /// hardcoded stats elsewhere in this wave. Heartbeats now carry a real
    /// `workload` value into the registry, so genuine imbalance is
    /// detectable.
    #[test]
    fn test_should_rebalance_fires_on_real_workload_imbalance_from_heartbeats() {
        let config = ElasticTrainingConfig {
            load_balancing: true,
            ..Default::default()
        };
        let mut coordinator = ElasticTrainingCoordinator::new(config);
        coordinator
            .register_worker("worker1".to_string(), test_hardware_info())
            .expect("register_worker failed");
        coordinator
            .register_worker("worker2".to_string(), test_hardware_info())
            .expect("register_worker failed");

        let busy = WorkerPerformanceMetrics {
            throughput: 100.0,
            workload: 0.9,
            ..Default::default()
        };
        let mostly_idle = WorkerPerformanceMetrics {
            throughput: 100.0,
            workload: 0.1,
            ..Default::default()
        };
        coordinator.update_heartbeat("worker1", busy).expect("update_heartbeat failed");
        coordinator
            .update_heartbeat("worker2", mostly_idle)
            .expect("update_heartbeat failed");

        assert!(
            coordinator.should_rebalance(),
            "a 0.9 vs 0.1 workload split reported by real heartbeats must be detected"
        );
    }

    #[test]
    fn test_recover_from_checkpoint_without_provisioner_is_a_structured_error() {
        let mut coordinator = ElasticTrainingCoordinator::new(ElasticTrainingConfig::default());
        let checkpoint = CheckpointInfo {
            timestamp: Instant::now(),
            worker_id: "worker1".to_string(),
            model_state: HashMap::new(),
            step: 5,
        };

        let result = coordinator.recover_from_checkpoint("worker1", &checkpoint);

        let err = result.expect_err("recovery with no provisioner must fail, not claim success");
        assert!(
            err.downcast_ref::<ElasticTrainingError>().is_some_and(|e| matches!(
                e,
                ElasticTrainingError::NoProvisioner {
                    operation: "recover_from_checkpoint",
                    ..
                }
            )),
            "expected ElasticTrainingError::NoProvisioner from recover_from_checkpoint, got: {err}"
        );
    }

    #[test]
    fn test_recover_from_checkpoint_with_provisioner_actually_restores() {
        let provisioner = Arc::new(MockProvisioner::new());
        let mut coordinator = ElasticTrainingCoordinator::new(ElasticTrainingConfig::default())
            .with_provisioner(provisioner.clone());
        let checkpoint = CheckpointInfo {
            timestamp: Instant::now(),
            worker_id: "worker1".to_string(),
            model_state: HashMap::new(),
            step: 5,
        };

        coordinator
            .recover_from_checkpoint("worker1", &checkpoint)
            .expect("recover_from_checkpoint failed");

        assert_eq!(
            provisioner
                .restore_calls
                .lock()
                .expect("lock should not be poisoned")
                .as_slice(),
            [("worker1".to_string(), 5)]
        );
    }

    #[test]
    fn test_handle_worker_failure_without_provisioner_still_deregisters_locally() {
        // Recovery cannot succeed without a provisioner, but the coordinator
        // must still do the real, local part of failure handling --
        // deregistering the confirmed-dead worker -- rather than hard-failing
        // the whole failure path just because recovery is unavailable.
        let mut coordinator = ElasticTrainingCoordinator::new(ElasticTrainingConfig::default());
        coordinator
            .register_worker("worker1".to_string(), test_hardware_info())
            .expect("register_worker failed");
        coordinator
            .create_checkpoint("worker1", 3, HashMap::new())
            .expect("create_checkpoint failed");

        let result = coordinator.handle_worker_failure("worker1");

        assert!(
            result.is_ok(),
            "handle_worker_failure must not hard-fail on missing recovery"
        );
        assert_eq!(
            coordinator.workers.lock().expect("lock should not be poisoned").len(),
            0
        );
    }
}
