/// Elastic training capabilities for dynamic scaling and fault tolerance
///
/// This module provides advanced distributed training features including:
/// - Dynamic worker scaling during training
/// - Fault tolerance and automatic recovery
/// - Checkpoint-restart optimization
/// - Resource monitoring and auto-scaling
/// - Load balancing across heterogeneous hardware
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
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
        }
    }
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
#[allow(dead_code)]
pub struct ElasticTrainingCoordinator {
    config: ElasticTrainingConfig,
    workers: Arc<Mutex<HashMap<String, WorkerInfo>>>,
    checkpoints: HashMap<String, CheckpointInfo>,
    scaling_history: Vec<ScalingEvent>,
    #[allow(dead_code)]
    performance_history: Vec<SystemPerformanceSnapshot>,
    resource_monitor: ResourceMonitor,
    fault_detector: FaultDetector,
    load_balancer: LoadBalancer,
}

impl ElasticTrainingCoordinator {
    pub fn new(config: ElasticTrainingConfig) -> Self {
        Self {
            config,
            workers: Arc::new(Mutex::new(HashMap::new())),
            checkpoints: HashMap::new(),
            scaling_history: Vec::new(),
            performance_history: Vec::new(),
            resource_monitor: ResourceMonitor::new(),
            fault_detector: FaultDetector::new(),
            load_balancer: LoadBalancer::new(),
        }
    }

    /// Register a new worker
    pub fn register_worker(
        &mut self,
        worker_id: String,
        hardware_info: HardwareInfo,
    ) -> Result<usize> {
        let mut workers = self.workers.lock().unwrap();

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

        println!("Registered worker with rank {}", rank);
        Ok(rank)
    }

    /// Update worker heartbeat
    pub fn update_heartbeat(
        &mut self,
        worker_id: &str,
        metrics: WorkerPerformanceMetrics,
    ) -> Result<()> {
        let mut workers = self.workers.lock().unwrap();

        if let Some(worker) = workers.get_mut(worker_id) {
            worker.last_heartbeat = Instant::now();
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
            let mut workers = self.workers.lock().unwrap();

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

        println!("Handling failure for worker: {}", worker_id);

        // Try to recover from checkpoint
        if let Some(checkpoint) = self.checkpoints.get(worker_id).cloned() {
            self.recover_from_checkpoint(worker_id, &checkpoint)?;
        }

        // Remove failed worker from active set
        {
            let mut workers = self.workers.lock().unwrap();
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

        let workers = self.workers.lock().unwrap();
        let active_workers =
            workers.iter().filter(|(_, w)| matches!(w.status, WorkerStatus::Active)).count();

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
        let workers = self.workers.lock().unwrap();
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

        let workers = self.workers.lock().unwrap();
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
    pub fn execute_scaling(&mut self, decision: &ScalingDecision) -> Result<()> {
        match decision.decision_type {
            ScalingType::ScaleUp => {
                self.scale_up(decision.target_workers)?;
            },
            ScalingType::ScaleDown => {
                self.scale_down(decision.target_workers)?;
            },
            ScalingType::Rebalance => {
                self.rebalance_workers()?;
            },
            ScalingType::NoChange => {},
        }

        // Record scaling event
        self.scaling_history.push(ScalingEvent {
            timestamp: Instant::now(),
            decision: decision.clone(),
            success: true,
        });

        Ok(())
    }

    /// Scale up workers
    fn scale_up(&mut self, target_workers: usize) -> Result<()> {
        let current_workers = self.workers.lock().unwrap().len();
        let workers_to_add = target_workers.saturating_sub(current_workers);

        println!("Scaling up: adding {} workers", workers_to_add);

        // In a real implementation, this would:
        // 1. Request new worker instances from resource manager
        // 2. Wait for workers to come online
        // 3. Redistribute workload
        // 4. Update routing tables

        Ok(())
    }

    /// Scale down workers
    fn scale_down(&mut self, target_workers: usize) -> Result<()> {
        let mut workers = self.workers.lock().unwrap();
        let current_workers = workers.len();
        let workers_to_remove = current_workers.saturating_sub(target_workers);

        println!("Scaling down: removing {} workers", workers_to_remove);

        // Select workers to remove (prefer idle workers)
        let mut workers_to_remove_ids = Vec::new();
        for (id, worker) in workers.iter() {
            if matches!(worker.status, WorkerStatus::Idle)
                && workers_to_remove_ids.len() < workers_to_remove
            {
                workers_to_remove_ids.push(id.clone());
            }
        }

        // Remove selected workers
        for worker_id in workers_to_remove_ids {
            workers.remove(&worker_id);
        }

        Ok(())
    }

    /// Rebalance workload across workers
    fn rebalance_workers(&mut self) -> Result<()> {
        println!("Rebalancing workload across workers");

        // In a real implementation, this would:
        // 1. Calculate optimal workload distribution
        // 2. Migrate work between workers
        // 3. Update load balancing weights
        // 4. Verify balance improvement

        Ok(())
    }

    /// Create checkpoint for fault tolerance
    pub fn create_checkpoint(
        &mut self,
        worker_id: &str,
        model_state: HashMap<String, Tensor>,
    ) -> Result<()> {
        if !self.config.fault_tolerance {
            return Ok(());
        }

        let checkpoint = CheckpointInfo {
            timestamp: Instant::now(),
            worker_id: worker_id.to_string(),
            model_state,
            step: 0, // Would be actual training step
        };

        self.checkpoints.insert(worker_id.to_string(), checkpoint);
        println!("Created checkpoint for worker: {}", worker_id);

        Ok(())
    }

    /// Recover from checkpoint
    fn recover_from_checkpoint(
        &mut self,
        worker_id: &str,
        _checkpoint: &CheckpointInfo,
    ) -> Result<()> {
        println!("Recovering worker {} from checkpoint", worker_id);

        // In a real implementation, this would:
        // 1. Restore model state from checkpoint
        // 2. Redistribute work from failed worker
        // 3. Update global state
        // 4. Resume training from checkpoint step

        Ok(())
    }

    /// Get current system status
    pub fn get_system_status(&self) -> SystemStatus {
        // Calculate performance first to avoid deadlock (it also acquires workers lock)
        let performance = self.calculate_system_performance();

        // Now acquire workers lock for remaining stats
        let workers = self.workers.lock().unwrap();
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

/// Resource monitoring for scaling decisions
pub struct ResourceMonitor {
    monitoring_active: bool,
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceMonitor {
    pub fn new() -> Self {
        Self {
            monitoring_active: false,
        }
    }

    pub fn start_monitoring(&mut self) {
        self.monitoring_active = true;
    }

    pub fn stop_monitoring(&mut self) {
        self.monitoring_active = false;
    }
}

/// Fault detection system
pub struct FaultDetector {
    detection_active: bool,
}

impl Default for FaultDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl FaultDetector {
    pub fn new() -> Self {
        Self {
            detection_active: false,
        }
    }

    pub fn start_detection(&mut self) {
        self.detection_active = true;
    }

    pub fn stop_detection(&mut self) {
        self.detection_active = false;
    }
}

/// Load balancing system
pub struct LoadBalancer {
    balancing_active: bool,
}

impl Default for LoadBalancer {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadBalancer {
    pub fn new() -> Self {
        Self {
            balancing_active: false,
        }
    }

    pub fn start_balancing(&mut self) {
        self.balancing_active = true;
    }

    pub fn stop_balancing(&mut self) {
        self.balancing_active = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elastic_coordinator_creation() {
        let config = ElasticTrainingConfig::default();
        let coordinator = ElasticTrainingCoordinator::new(config);

        assert_eq!(coordinator.workers.lock().unwrap().len(), 0);
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
        assert_eq!(result.unwrap(), 0);
        assert_eq!(coordinator.workers.lock().unwrap().len(), 1);
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

        coordinator.register_worker("worker1".to_string(), hardware_info).unwrap();

        let metrics = WorkerPerformanceMetrics {
            throughput: 100.0,
            latency: Duration::from_millis(50),
            memory_usage: 0.5,
            cpu_usage: 0.6,
            gpu_utilization: 0.8,
            network_usage: 0.3,
        };

        let result = coordinator.update_heartbeat("worker1", metrics);
        assert!(result.is_ok());
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

        let decision = coordinator.evaluate_scaling_decision().unwrap();
        assert!(decision.is_some());

        let decision = decision.unwrap();
        assert!(matches!(decision.decision_type, ScalingType::ScaleUp));
        assert_eq!(decision.target_workers, 2);
    }

    #[test]
    fn test_checkpoint_creation() {
        let config = ElasticTrainingConfig::default();
        let mut coordinator = ElasticTrainingCoordinator::new(config);

        let model_state = HashMap::new();
        let result = coordinator.create_checkpoint("worker1", model_state);

        assert!(result.is_ok());
        assert_eq!(coordinator.checkpoints.len(), 1);
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
}
