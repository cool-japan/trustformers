// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! Hardware operation scheduling components
//!
//! This module provides scheduling algorithms and implementations for distributing
//! operations across available hardware devices efficiently.

#![allow(unused_variables)] // Hardware scheduling

use super::traits::{HardwareOperation, HardwareScheduler, SchedulerStatistics};
use super::{HardwareResult, OperationParameter};
use crate::errors::TrustformersError;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

/// Default scheduler implementation with priority-based scheduling
#[derive(Debug)]
pub struct DefaultScheduler {
    /// Scheduler statistics
    statistics: Arc<Mutex<SchedulerStatistics>>,
    /// Device priorities (higher is better)
    device_priorities: Arc<Mutex<HashMap<String, f64>>>,
    /// Operation queue
    operation_queue: Arc<Mutex<Vec<QueuedOperation>>>,
    /// Scheduling configuration
    config: SchedulerConfig,
}

/// Queued operation waiting for execution
#[derive(Debug, Clone)]
pub struct QueuedOperation {
    /// Operation identifier
    pub id: String,
    /// Operation to execute
    pub operation_type: String,
    /// Input tensors
    pub inputs: Vec<TensorInfo>,
    /// Operation parameters
    pub params: HashMap<String, OperationParameter>,
    /// Priority level
    pub priority: f64,
    /// Enqueue timestamp
    pub enqueued_at: SystemTime,
    /// Expected execution time
    pub estimated_duration: Duration,
}

/// Tensor information for scheduling
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor shape
    pub shape: Vec<usize>,
    /// Data type size in bytes
    pub dtype_size: usize,
    /// Memory layout
    pub layout: String,
}

/// Scheduler configuration
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Queue timeout
    pub queue_timeout: Duration,
    /// Enable priority scheduling
    pub enable_priority_scheduling: bool,
    /// Load balancing weight
    pub load_balancing_weight: f64,
    /// Performance weight
    pub performance_weight: f64,
    /// Availability weight
    pub availability_weight: f64,
}

/// Advanced scheduler with multiple scheduling algorithms
#[derive(Debug)]
pub struct AdvancedScheduler {
    /// Scheduling algorithm to use
    algorithm: SchedulingAlgorithm,
    /// Device load tracking
    device_loads: Arc<Mutex<HashMap<String, DeviceLoad>>>,
    /// Performance history
    performance_history: Arc<Mutex<HashMap<String, Vec<PerformanceRecord>>>>,
    /// Configuration
    config: AdvancedSchedulerConfig,
    /// Round-robin cursor over `device_loads`' (sorted) keys.
    round_robin_cursor: Arc<Mutex<usize>>,
    /// Real, running scheduling statistics (see `record_schedule_result`).
    statistics: Arc<Mutex<SchedulerStatistics>>,
}

/// Available scheduling algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingAlgorithm {
    /// First Come First Served
    FCFS,
    /// Shortest Job First
    SJF,
    /// Priority-based scheduling
    Priority,
    /// Round Robin
    RoundRobin,
    /// Load-aware scheduling
    LoadAware,
    /// Performance-based scheduling
    PerformanceBased,
    /// Machine learning-based scheduling
    MLBased,
}

/// Device load information
#[derive(Debug, Clone)]
pub struct DeviceLoad {
    /// Current utilization (0.0 - 1.0)
    pub utilization: f64,
    /// Active operations count
    pub active_operations: u32,
    /// Queued operations count
    pub queued_operations: u32,
    /// Last update timestamp
    pub last_updated: SystemTime,
    /// Average response time
    pub avg_response_time: Duration,
}

/// Performance record for scheduling decisions
#[derive(Debug, Clone)]
pub struct PerformanceRecord {
    /// Operation type
    pub operation_type: String,
    /// Execution time
    pub execution_time: Duration,
    /// Throughput achieved
    pub throughput: f64,
    /// Resource utilization
    pub resource_utilization: HashMap<String, f64>,
    /// Timestamp
    pub timestamp: SystemTime,
}

/// Advanced scheduler configuration
#[derive(Debug, Clone)]
pub struct AdvancedSchedulerConfig {
    /// Learning rate for ML-based scheduling
    pub learning_rate: f64,
    /// History window size
    pub history_window: usize,
    /// Performance prediction accuracy threshold
    pub prediction_threshold: f64,
    /// Load balancing aggressiveness
    pub load_balancing_factor: f64,
}

impl DefaultScheduler {
    /// Create a new default scheduler
    pub fn new() -> Self {
        Self {
            statistics: Arc::new(Mutex::new(SchedulerStatistics::default())),
            device_priorities: Arc::new(Mutex::new(HashMap::new())),
            operation_queue: Arc::new(Mutex::new(Vec::new())),
            config: SchedulerConfig::default(),
        }
    }

    /// Create scheduler with custom configuration
    pub fn with_config(config: SchedulerConfig) -> Self {
        Self {
            statistics: Arc::new(Mutex::new(SchedulerStatistics::default())),
            device_priorities: Arc::new(Mutex::new(HashMap::new())),
            operation_queue: Arc::new(Mutex::new(Vec::new())),
            config,
        }
    }

    /// Enqueue an operation for scheduling
    pub fn enqueue_operation(&self, operation: QueuedOperation) -> HardwareResult<()> {
        let mut queue = self.operation_queue.lock().map_err(|_| {
            TrustformersError::model_error("Failed to lock operation queue".to_string())
        })?;

        if queue.len() >= self.config.max_queue_size {
            return Err(TrustformersError::model_error(
                "Operation queue is full".to_string(),
            ));
        }

        // Insert operation maintaining priority order
        let insert_pos = queue
            .iter()
            .position(|op| op.priority < operation.priority)
            .unwrap_or(queue.len());

        queue.insert(insert_pos, operation);

        // Update statistics
        if let Ok(mut stats) = self.statistics.lock() {
            stats.total_operations += 1;
        }

        Ok(())
    }

    /// Get next operation from queue
    pub fn dequeue_operation(&self) -> Option<QueuedOperation> {
        if let Ok(mut queue) = self.operation_queue.lock() {
            if !queue.is_empty() {
                return Some(queue.remove(0));
            }
        }
        None
    }

    /// Find best device for operation based on current scheduling strategy
    fn find_best_device(&self, operation: &QueuedOperation) -> HardwareResult<String> {
        let priorities = self.device_priorities.lock().map_err(|_| {
            TrustformersError::model_error("Failed to lock device priorities".to_string())
        })?;

        if priorities.is_empty() {
            return Err(TrustformersError::model_error(
                "No devices available".to_string(),
            ));
        }

        // Find device with highest priority that can handle the operation
        let best_device = priorities
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(::std::cmp::Ordering::Equal))
            .map(|(device_id, _)| device_id.clone())
            .ok_or_else(|| {
                TrustformersError::model_error("No suitable device found".to_string())
            })?;

        Ok(best_device)
    }

    /// Update device performance metrics
    pub fn update_device_metrics(&self, device_id: &str, performance: &PerformanceRecord) {
        // Update internal performance tracking for better scheduling decisions
        if let Ok(mut priorities) = self.device_priorities.lock() {
            let current_priority = priorities.get(device_id).cloned().unwrap_or(1.0);

            // Adjust priority based on performance
            let performance_factor = match performance.execution_time.as_millis() {
                0..=100 => 1.2,    // Very fast
                101..=500 => 1.0,  // Normal
                501..=1000 => 0.8, // Slow
                _ => 0.5,          // Very slow
            };

            let new_priority = current_priority * performance_factor;
            priorities.insert(device_id.to_string(), new_priority);
        }
    }
}

/// Record a scheduling decision (or failure) into a shared
/// `SchedulerStatistics`: total/per-device operation counts, a running
/// average of scheduling latency, and per-device utilization approximated
/// as that device's share of all operations scheduled so far. Shared by
/// both `DefaultScheduler` and `AdvancedScheduler` so the four fields the
/// hardcoded `statistics()` implementations used to leave frozen at their
/// `Default` values (`operations_per_device`, `avg_scheduling_time`,
/// `device_utilization`, `failed_operations`) are actually updated.
fn record_schedule_result(
    stats: &Arc<Mutex<SchedulerStatistics>>,
    device_id: Option<&str>,
    elapsed: Duration,
) {
    let Ok(mut stats) = stats.lock() else { return };

    stats.total_operations += 1;
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let n = stats.total_operations as f64;
    // Running mean: new_avg = old_avg + (sample - old_avg) / n
    stats.avg_scheduling_time += (elapsed_ms - stats.avg_scheduling_time) / n;

    match device_id {
        Some(device_id) => {
            *stats.operations_per_device.entry(device_id.to_string()).or_insert(0) += 1;
        },
        None => {
            stats.failed_operations += 1;
        },
    }

    let total_successful: u64 = stats.operations_per_device.values().sum();
    if total_successful > 0 {
        let shares: Vec<(String, f64)> = stats
            .operations_per_device
            .iter()
            .map(|(device, count)| (device.clone(), *count as f64 / total_successful as f64))
            .collect();
        for (device, share) in shares {
            stats.device_utilization.insert(device, share);
        }
    }
}

impl HardwareScheduler for DefaultScheduler {
    fn schedule_operation(
        &self,
        operation: &dyn HardwareOperation,
        inputs: &[Tensor],
        params: &HashMap<String, OperationParameter>,
    ) -> HardwareResult<String> {
        let start = Instant::now();

        // Convert to queued operation
        let queued_op = QueuedOperation {
            id: format!(
                "op_{}",
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            ),
            operation_type: operation.name().to_string(),
            inputs: inputs
                .iter()
                .map(|t| TensorInfo {
                    shape: t.shape(),
                    dtype_size: 4, // Assume f32 for now
                    layout: "contiguous".to_string(),
                })
                .collect(),
            params: params.clone(),
            priority: 1.0, // Default priority since HardwareOperation doesn't have priority()
            enqueued_at: SystemTime::now(),
            estimated_duration: Duration::from_millis(100), // Default estimate since no estimated_duration()
        };

        // Find best device for this operation
        let result = self.find_best_device(&queued_op);
        record_schedule_result(&self.statistics, result.as_deref().ok(), start.elapsed());
        result
    }

    fn statistics(&self) -> SchedulerStatistics {
        self.statistics.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone()
    }

    fn update_priorities(&mut self, priorities: HashMap<String, f64>) {
        if let Ok(mut device_priorities) = self.device_priorities.lock() {
            *device_priorities = priorities;
        }
    }
}

impl AdvancedScheduler {
    /// Create a new advanced scheduler
    pub fn new(algorithm: SchedulingAlgorithm) -> Self {
        Self {
            algorithm,
            device_loads: Arc::new(Mutex::new(HashMap::new())),
            performance_history: Arc::new(Mutex::new(HashMap::new())),
            config: AdvancedSchedulerConfig::default(),
            round_robin_cursor: Arc::new(Mutex::new(0)),
            statistics: Arc::new(Mutex::new(SchedulerStatistics::default())),
        }
    }

    /// Devices known to this scheduler: the (sorted, for determinism) key
    /// set of `device_loads`, populated by `update_device_load`. Scheduling
    /// must be restricted to devices the caller has actually registered,
    /// never a hardcoded guess like `["cpu", "gpu"]` that could name a
    /// device this scheduler (or this machine) does not have.
    fn known_devices(&self) -> Vec<String> {
        let Ok(loads) = self.device_loads.lock() else {
            return Vec::new();
        };
        let mut devices: Vec<String> = loads.keys().cloned().collect();
        devices.sort();
        devices
    }

    /// Update device load information
    pub fn update_device_load(&self, device_id: &str, load: DeviceLoad) {
        if let Ok(mut loads) = self.device_loads.lock() {
            loads.insert(device_id.to_string(), load);
        }
    }

    /// Record performance for learning-based scheduling
    pub fn record_performance(&self, device_id: &str, record: PerformanceRecord) {
        if let Ok(mut history) = self.performance_history.lock() {
            let device_history = history.entry(device_id.to_string()).or_default();
            device_history.push(record);

            // Keep only recent history within window
            if device_history.len() > self.config.history_window {
                device_history.drain(..device_history.len() - self.config.history_window);
            }
        }
    }

    /// Predict performance for a given operation on a device
    pub fn predict_performance(&self, device_id: &str, operation_type: &str) -> Option<Duration> {
        if let Ok(history) = self.performance_history.lock() {
            if let Some(device_history) = history.get(device_id) {
                let matching_ops: Vec<_> = device_history
                    .iter()
                    .filter(|record| record.operation_type == operation_type)
                    .collect();

                if !matching_ops.is_empty() {
                    let avg_duration = matching_ops
                        .iter()
                        .map(|record| record.execution_time.as_millis())
                        .sum::<u128>()
                        / matching_ops.len() as u128;

                    return Some(Duration::from_millis(avg_duration as u64));
                }
            }
        }
        None
    }

    /// Schedule using the configured algorithm
    pub fn schedule_advanced(
        &self,
        operation: &QueuedOperation,
        available_devices: &[String],
    ) -> HardwareResult<String> {
        match self.algorithm {
            SchedulingAlgorithm::FCFS => self.schedule_fcfs(available_devices),
            SchedulingAlgorithm::SJF => self.schedule_sjf(operation, available_devices),
            SchedulingAlgorithm::Priority => self.schedule_priority(operation, available_devices),
            SchedulingAlgorithm::RoundRobin => self.schedule_round_robin(available_devices),
            SchedulingAlgorithm::LoadAware => self.schedule_load_aware(available_devices),
            SchedulingAlgorithm::PerformanceBased => {
                self.schedule_performance_based(operation, available_devices)
            },
            SchedulingAlgorithm::MLBased => self.schedule_ml_based(operation, available_devices),
        }
    }

    fn schedule_fcfs(&self, available_devices: &[String]) -> HardwareResult<String> {
        available_devices
            .first()
            .ok_or_else(|| TrustformersError::model_error("No devices available".to_string()))
            .cloned()
    }

    fn schedule_sjf(
        &self,
        operation: &QueuedOperation,
        available_devices: &[String],
    ) -> HardwareResult<String> {
        // Select device with shortest predicted execution time
        let mut best_device = None;
        let mut best_time = Duration::from_secs(u64::MAX);

        for device_id in available_devices {
            if let Some(predicted_time) =
                self.predict_performance(device_id, &operation.operation_type)
            {
                if predicted_time < best_time {
                    best_time = predicted_time;
                    best_device = Some(device_id.clone());
                }
            }
        }

        best_device
            .or_else(|| available_devices.first().cloned())
            .ok_or_else(|| TrustformersError::model_error("No devices available".to_string()))
    }

    fn schedule_priority(
        &self,
        operation: &QueuedOperation,
        available_devices: &[String],
    ) -> HardwareResult<String> {
        // Select device based on operation priority and device capability
        // For now, return first available device
        available_devices
            .first()
            .ok_or_else(|| TrustformersError::model_error("No devices available".to_string()))
            .cloned()
    }

    fn schedule_round_robin(&self, available_devices: &[String]) -> HardwareResult<String> {
        if available_devices.is_empty() {
            return Err(TrustformersError::model_error(
                "No devices available".to_string(),
            ));
        }

        // Cycle through `available_devices` using a cursor carried across
        // calls, so repeated calls actually round-robin instead of always
        // returning the first entry.
        let mut cursor = self
            .round_robin_cursor
            .lock()
            .map_err(|_| TrustformersError::model_error("Failed to lock RR cursor".to_string()))?;
        let idx = *cursor % available_devices.len();
        *cursor = cursor.wrapping_add(1);
        Ok(available_devices[idx].clone())
    }

    fn schedule_load_aware(&self, available_devices: &[String]) -> HardwareResult<String> {
        if let Ok(loads) = self.device_loads.lock() {
            let mut best_device = None;
            let mut lowest_load = f64::MAX;

            for device_id in available_devices {
                if let Some(load) = loads.get(device_id) {
                    if load.utilization < lowest_load {
                        lowest_load = load.utilization;
                        best_device = Some(device_id.clone());
                    }
                }
            }

            return best_device
                .or_else(|| available_devices.first().cloned())
                .ok_or_else(|| TrustformersError::model_error("No devices available".to_string()));
        }

        self.schedule_fcfs(available_devices)
    }

    fn schedule_performance_based(
        &self,
        operation: &QueuedOperation,
        available_devices: &[String],
    ) -> HardwareResult<String> {
        // Select device with best historical performance for this operation type
        if let Ok(history) = self.performance_history.lock() {
            let mut best_device = None;
            let mut best_throughput = 0.0;

            for device_id in available_devices {
                if let Some(device_history) = history.get(device_id) {
                    let avg_throughput = device_history
                        .iter()
                        .filter(|record| record.operation_type == operation.operation_type)
                        .map(|record| record.throughput)
                        .sum::<f64>()
                        / device_history.len() as f64;

                    if avg_throughput > best_throughput {
                        best_throughput = avg_throughput;
                        best_device = Some(device_id.clone());
                    }
                }
            }

            return best_device
                .or_else(|| available_devices.first().cloned())
                .ok_or_else(|| TrustformersError::model_error("No devices available".to_string()));
        }

        self.schedule_fcfs(available_devices)
    }

    fn schedule_ml_based(
        &self,
        _operation: &QueuedOperation,
        available_devices: &[String],
    ) -> HardwareResult<String> {
        // Placeholder for ML-based scheduling
        // Would implement neural network or other ML model for scheduling decisions
        self.schedule_load_aware(available_devices)
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 1000,
            queue_timeout: Duration::from_secs(30),
            enable_priority_scheduling: true,
            load_balancing_weight: 0.3,
            performance_weight: 0.4,
            availability_weight: 0.3,
        }
    }
}

impl Default for AdvancedSchedulerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            history_window: 1000,
            prediction_threshold: 0.8,
            load_balancing_factor: 1.0,
        }
    }
}

impl Default for DeviceLoad {
    fn default() -> Self {
        Self {
            utilization: 0.0,
            active_operations: 0,
            queued_operations: 0,
            last_updated: SystemTime::now(),
            avg_response_time: Duration::from_millis(100),
        }
    }
}

impl Default for DefaultScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl HardwareScheduler for AdvancedScheduler {
    fn schedule_operation(
        &self,
        operation: &dyn HardwareOperation,
        inputs: &[Tensor],
        params: &HashMap<String, OperationParameter>,
    ) -> crate::hardware::HardwareResult<String> {
        let start = Instant::now();

        // Convert to queued operation for advanced scheduling
        let queued_op = QueuedOperation {
            id: format!(
                "op_{}",
                SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
            ),
            operation_type: operation.name().to_string(),
            inputs: inputs
                .iter()
                .map(|t| TensorInfo {
                    shape: t.shape(),
                    dtype_size: 4, // Assume f32 for now
                    layout: "contiguous".to_string(),
                })
                .collect(),
            params: params.clone(),
            priority: 1.0, // Default priority
            enqueued_at: SystemTime::now(),
            estimated_duration: Duration::from_millis(100), // Default estimate
        };

        // Devices actually registered via `update_device_load`, never a
        // hardcoded guess: a machine with no GPU must not have "gpu" handed
        // back as a schedulable target.
        let available_devices = self.known_devices();

        let result = if available_devices.is_empty() {
            Err(TrustformersError::model_error(
                "No devices registered with this scheduler; call update_device_load first"
                    .to_string(),
            ))
        } else {
            match self.algorithm {
                SchedulingAlgorithm::FCFS => self.schedule_fcfs(&available_devices),
                SchedulingAlgorithm::SJF => self.schedule_sjf(&queued_op, &available_devices),
                SchedulingAlgorithm::Priority => {
                    self.schedule_priority(&queued_op, &available_devices)
                },
                SchedulingAlgorithm::RoundRobin => self.schedule_round_robin(&available_devices),
                SchedulingAlgorithm::LoadAware => self.schedule_load_aware(&available_devices),
                SchedulingAlgorithm::PerformanceBased => {
                    self.schedule_performance_based(&queued_op, &available_devices)
                },
                SchedulingAlgorithm::MLBased => {
                    self.schedule_ml_based(&queued_op, &available_devices)
                },
            }
        };

        record_schedule_result(&self.statistics, result.as_deref().ok(), start.elapsed());
        result
    }

    fn statistics(&self) -> SchedulerStatistics {
        self.statistics.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone()
    }

    fn update_priorities(&mut self, priorities: HashMap<String, f64>) {
        // `AdvancedScheduler` ranks devices by `DeviceLoad::utilization`
        // (see `schedule_load_aware`) rather than by a separate priority
        // map, so an externally supplied priority is translated into a
        // synthetic utilization for any device not already tracked: higher
        // priority -> lower (better) utilization. Devices that already have
        // real load data from `update_device_load` are left untouched
        // rather than overwritten with a synthetic value.
        if let Ok(mut loads) = self.device_loads.lock() {
            for (device_id, priority) in priorities {
                loads.entry(device_id).or_insert_with(|| DeviceLoad {
                    utilization: (1.0 - priority.clamp(0.0, 1.0)).clamp(0.0, 1.0),
                    ..DeviceLoad::default()
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::traits::{HardwareDevice, OperationRequirements};

    /// Minimal `HardwareOperation` used only to exercise the scheduler by
    /// name; `execute`/`validate_params`/`requirements`/`estimate_cost` are
    /// not invoked by `schedule_operation`.
    struct NamedOp(&'static str);

    #[async_trait::async_trait]
    impl HardwareOperation for NamedOp {
        fn name(&self) -> &str {
            self.0
        }

        async fn execute(
            &self,
            _device: &mut dyn HardwareDevice,
            _inputs: &[Tensor],
            _outputs: &mut [Tensor],
            _params: &HashMap<String, OperationParameter>,
        ) -> HardwareResult<()> {
            Ok(())
        }

        fn validate_params(
            &self,
            _params: &HashMap<String, OperationParameter>,
        ) -> HardwareResult<()> {
            Ok(())
        }

        fn requirements(&self) -> OperationRequirements {
            OperationRequirements {
                min_memory: 0,
                compute_units: None,
                data_types: vec![],
                capabilities: vec![],
                performance: Default::default(),
            }
        }

        fn estimate_cost(
            &self,
            _inputs: &[Tensor],
            _params: &HashMap<String, OperationParameter>,
        ) -> f64 {
            0.0
        }
    }

    /// Regression test: `AdvancedScheduler::schedule_operation` used to
    /// hardcode `available_devices = vec!["cpu", "gpu"]`, so it would
    /// happily return `"gpu"` even when no GPU (or any device at all) had
    /// ever been registered via `update_device_load`. With no devices
    /// registered it must now error instead of naming a phantom device.
    #[test]
    fn test_advanced_scheduler_errors_with_no_registered_devices() {
        let scheduler = AdvancedScheduler::new(SchedulingAlgorithm::FCFS);
        let op = NamedOp("add");
        let result = scheduler.schedule_operation(&op, &[], &HashMap::new());
        assert!(
            result.is_err(),
            "must not schedule onto a hardcoded phantom device list"
        );
    }

    /// Regression test: scheduling must only ever name devices that were
    /// actually registered, never the old hardcoded "cpu"/"gpu" pair.
    #[test]
    fn test_advanced_scheduler_only_schedules_registered_devices() {
        let scheduler = AdvancedScheduler::new(SchedulingAlgorithm::FCFS);
        scheduler.update_device_load("accel_7", DeviceLoad::default());

        let op = NamedOp("add");
        let device = scheduler
            .schedule_operation(&op, &[], &HashMap::new())
            .expect("schedule_operation should succeed with a registered device");
        assert_eq!(device, "accel_7");
    }

    /// Regression test: round-robin must actually cycle through devices
    /// across repeated calls instead of always returning the first entry.
    #[test]
    fn test_advanced_scheduler_round_robin_cycles() {
        let scheduler = AdvancedScheduler::new(SchedulingAlgorithm::RoundRobin);
        scheduler.update_device_load("dev_a", DeviceLoad::default());
        scheduler.update_device_load("dev_b", DeviceLoad::default());

        let op = NamedOp("add");
        let mut seen = std::collections::HashSet::new();
        for _ in 0..4 {
            let device = scheduler
                .schedule_operation(&op, &[], &HashMap::new())
                .expect("schedule_operation failed");
            seen.insert(device);
        }
        assert_eq!(
            seen.len(),
            2,
            "round-robin over two devices must eventually visit both, got {seen:?}"
        );
    }

    /// Regression test: `statistics()` used to return a constant
    /// `SchedulerStatistics` (10.0ms average, empty maps) regardless of how
    /// many operations had actually been scheduled.
    #[test]
    fn test_advanced_scheduler_statistics_reflect_real_activity() {
        let scheduler = AdvancedScheduler::new(SchedulingAlgorithm::FCFS);
        scheduler.update_device_load("dev_a", DeviceLoad::default());

        let op = NamedOp("add");
        for _ in 0..3 {
            scheduler
                .schedule_operation(&op, &[], &HashMap::new())
                .expect("schedule failed");
        }

        let stats = scheduler.statistics();
        assert_eq!(stats.total_operations, 3);
        assert_eq!(stats.operations_per_device.get("dev_a").copied(), Some(3));
        assert_eq!(stats.device_utilization.get("dev_a").copied(), Some(1.0));
        assert_eq!(stats.failed_operations, 0);
    }

    /// Regression test: a scheduling failure (no registered devices) must
    /// be counted in `failed_operations`, not silently dropped.
    #[test]
    fn test_advanced_scheduler_statistics_count_failures() {
        let scheduler = AdvancedScheduler::new(SchedulingAlgorithm::FCFS);
        let op = NamedOp("add");

        let _ = scheduler.schedule_operation(&op, &[], &HashMap::new());
        let _ = scheduler.schedule_operation(&op, &[], &HashMap::new());

        let stats = scheduler.statistics();
        assert_eq!(stats.total_operations, 2);
        assert_eq!(stats.failed_operations, 2);
    }

    /// Regression test: `DefaultScheduler::statistics` used to leave
    /// `operations_per_device`/`device_utilization`/`failed_operations`
    /// frozen at their `Default` (empty/zero) values forever, even as
    /// `total_operations` incremented.
    #[test]
    fn test_default_scheduler_statistics_track_per_device_counts() {
        let mut scheduler = DefaultScheduler::new();
        let mut priorities = HashMap::new();
        priorities.insert("dev_x".to_string(), 1.0);
        scheduler.update_priorities(priorities);

        let op = NamedOp("add");
        scheduler
            .schedule_operation(&op, &[], &HashMap::new())
            .expect("schedule failed");
        scheduler
            .schedule_operation(&op, &[], &HashMap::new())
            .expect("schedule failed");

        let stats = scheduler.statistics();
        assert_eq!(stats.total_operations, 2);
        assert_eq!(stats.operations_per_device.get("dev_x").copied(), Some(2));
        assert!(
            stats.device_utilization.contains_key("dev_x"),
            "device_utilization must be populated, not left empty"
        );
    }
}
