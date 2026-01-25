use crate::gpu_scheduler::GpuScheduler;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use tokio::sync::Notify;

/// Dynamic GPU allocation manager
pub struct DynamicGpuAllocator {
    /// GPU resources available
    gpu_resources: Arc<RwLock<HashMap<u32, GpuResource>>>,
    /// Allocation requests queue
    allocation_queue: Arc<Mutex<VecDeque<AllocationRequest>>>,
    /// Active allocations
    active_allocations: Arc<RwLock<HashMap<AllocationId, GpuAllocation>>>,
    /// Configuration
    config: DynamicAllocationConfig,
    /// Scheduler
    scheduler: Arc<GpuScheduler>,
    /// Metrics
    metrics: Arc<Mutex<AllocationMetrics>>,
    /// Notification for queue changes
    queue_notify: Arc<Notify>,
    /// Resource availability notifier
    resource_notify: Arc<Notify>,
    /// Allocation strategies
    strategies: HashMap<AllocationStrategy, Box<dyn AllocationStrategyImpl>>,
}

/// GPU resource information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuResource {
    /// Device ID
    pub device_id: u32,
    /// Total memory in bytes
    pub total_memory: u64,
    /// Available memory in bytes
    pub available_memory: u64,
    /// Compute capability
    pub compute_capability: (u32, u32),
    /// Current utilization (0.0 to 1.0)
    pub utilization: f32,
    /// Temperature in Celsius
    pub temperature: f32,
    /// Power consumption in watts
    pub power_consumption: f32,
    /// Status
    pub status: GpuStatus,
    /// Active allocations
    pub active_allocations: Vec<AllocationId>,
    /// Reserved memory
    pub reserved_memory: u64,
    /// Performance profile
    pub performance_profile: PerformanceProfile,
}

/// GPU status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuStatus {
    /// Available for allocation
    Available,
    /// Busy with current tasks
    Busy,
    /// Maintenance mode
    Maintenance,
    /// Error state
    Error(String),
    /// Overheating
    Overheating,
    /// Low memory
    LowMemory,
}

/// Performance profile for GPU
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    /// FP32 performance (TFLOPS)
    pub fp32_performance: f32,
    /// FP16 performance (TFLOPS)
    pub fp16_performance: f32,
    /// INT8 performance (TOPS)
    pub int8_performance: f32,
    /// Memory bandwidth (GB/s)
    pub memory_bandwidth: f32,
    /// Tensor core availability
    pub tensor_cores: bool,
}

/// Allocation request
#[derive(Debug, Serialize, Deserialize)]
pub struct AllocationRequest {
    /// Request ID
    pub id: AllocationId,
    /// Memory requirement in bytes
    pub memory_required: u64,
    /// Compute requirement (relative)
    pub compute_required: f32,
    /// Priority
    pub priority: AllocationPriority,
    /// Preferred devices
    pub preferred_devices: Option<Vec<u32>>,
    /// Allocation strategy
    pub strategy: AllocationStrategy,
    /// Duration hint
    pub duration_hint: Option<Duration>,
    /// Request timestamp
    pub requested_at: DateTime<Utc>,
    /// Timeout
    pub timeout: Duration,
    /// Callback for allocation result
    #[serde(skip, default)]
    pub callback: Option<tokio::sync::oneshot::Sender<AllocationResult>>,
}

/// Allocation ID
pub type AllocationId = u64;

/// Allocation priority
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AllocationPriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Allocation strategy
#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum AllocationStrategy {
    /// First available GPU
    FirstAvailable,
    /// Best fit based on memory
    BestFit,
    /// Least loaded GPU
    LeastLoaded,
    /// Round robin
    RoundRobin,
    /// Affinity-based allocation
    Affinity,
    /// Performance-based allocation
    PerformanceBased,
    /// Power-efficient allocation
    PowerEfficient,
}

/// GPU allocation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuAllocation {
    /// Allocation ID
    pub id: AllocationId,
    /// Assigned device ID
    pub device_id: u32,
    /// Allocated memory in bytes
    pub allocated_memory: u64,
    /// Allocation timestamp
    pub allocated_at: DateTime<Utc>,
    /// Expected duration
    pub expected_duration: Option<Duration>,
    /// Allocation type
    pub allocation_type: AllocationType,
    /// Session ID
    pub session_id: Option<String>,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Allocation type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationType {
    /// Inference allocation
    Inference,
    /// Training allocation
    Training,
    /// Batch processing
    BatchProcessing,
    /// Streaming
    Streaming,
    /// Cache allocation
    Cache,
    /// Temporary allocation
    Temporary,
}

/// Allocation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationResult {
    /// Allocation successful
    Success(GpuAllocation),
    /// Allocation failed
    Failed(AllocationError),
    /// Allocation queued
    Queued,
    /// Allocation timeout
    Timeout,
}

/// Allocation error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AllocationError {
    /// No available GPU
    NoAvailableGpu,
    /// Insufficient memory
    InsufficientMemory,
    /// Device error
    DeviceError(String),
    /// Invalid request
    InvalidRequest(String),
    /// Timeout
    Timeout,
    /// Resource contention
    ResourceContention,
}

/// Dynamic allocation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicAllocationConfig {
    /// Enable preemption
    pub enable_preemption: bool,
    /// Memory overcommit ratio
    pub memory_overcommit_ratio: f32,
    /// Allocation timeout
    pub allocation_timeout: Duration,
    /// Queue timeout
    pub queue_timeout: Duration,
    /// Rebalancing interval
    pub rebalancing_interval: Duration,
    /// Memory fragmentation threshold
    pub memory_fragmentation_threshold: f32,
    /// Utilization threshold for rebalancing
    pub utilization_threshold: f32,
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Enable auto-scaling
    pub enable_auto_scaling: bool,
    /// Auto-scaling strategy
    pub auto_scaling_strategy: AutoScalingStrategy,
}

impl Default for DynamicAllocationConfig {
    fn default() -> Self {
        Self {
            enable_preemption: true,
            memory_overcommit_ratio: 1.2,
            allocation_timeout: Duration::from_secs(30),
            queue_timeout: Duration::from_secs(300),
            rebalancing_interval: Duration::from_secs(60),
            memory_fragmentation_threshold: 0.3,
            utilization_threshold: 0.8,
            max_queue_size: 1000,
            enable_auto_scaling: true,
            auto_scaling_strategy: AutoScalingStrategy::Reactive,
        }
    }
}

/// Auto-scaling strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AutoScalingStrategy {
    /// Reactive scaling
    Reactive,
    /// Predictive scaling
    Predictive,
    /// Scheduled scaling
    Scheduled,
    /// Manual scaling
    Manual,
}

/// Allocation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllocationMetrics {
    /// Total allocations
    pub total_allocations: u64,
    /// Successful allocations
    pub successful_allocations: u64,
    /// Failed allocations
    pub failed_allocations: u64,
    /// Average allocation time
    pub avg_allocation_time: Duration,
    /// Queue length
    pub queue_length: usize,
    /// Memory utilization per GPU
    pub memory_utilization: HashMap<u32, f32>,
    /// GPU utilization per device
    pub gpu_utilization: HashMap<u32, f32>,
    /// Fragmentation ratio
    pub fragmentation_ratio: f32,
    /// Preemption count
    pub preemption_count: u64,
}

/// Allocation strategy implementation trait
pub trait AllocationStrategyImpl: Send + Sync {
    /// Select best GPU for allocation
    fn select_gpu(
        &self,
        request: &AllocationRequest,
        resources: &HashMap<u32, GpuResource>,
    ) -> Option<u32>;

    /// Calculate allocation score
    fn calculate_score(&self, resource: &GpuResource, request: &AllocationRequest) -> f32;
}

/// First available strategy
pub struct FirstAvailableStrategy;

impl AllocationStrategyImpl for FirstAvailableStrategy {
    fn select_gpu(
        &self,
        request: &AllocationRequest,
        resources: &HashMap<u32, GpuResource>,
    ) -> Option<u32> {
        resources
            .iter()
            .find(|(_, resource)| {
                matches!(resource.status, GpuStatus::Available)
                    && resource.available_memory >= request.memory_required
            })
            .map(|(device_id, _)| *device_id)
    }

    fn calculate_score(&self, resource: &GpuResource, _request: &AllocationRequest) -> f32 {
        match resource.status {
            GpuStatus::Available => 1.0,
            _ => 0.0,
        }
    }
}

/// Best fit strategy
pub struct BestFitStrategy;

impl AllocationStrategyImpl for BestFitStrategy {
    fn select_gpu(
        &self,
        request: &AllocationRequest,
        resources: &HashMap<u32, GpuResource>,
    ) -> Option<u32> {
        resources
            .iter()
            .filter(|(_, resource)| {
                matches!(resource.status, GpuStatus::Available)
                    && resource.available_memory >= request.memory_required
            })
            .min_by(|(_, a), (_, b)| a.available_memory.cmp(&b.available_memory))
            .map(|(device_id, _)| *device_id)
    }

    fn calculate_score(&self, resource: &GpuResource, request: &AllocationRequest) -> f32 {
        if resource.available_memory < request.memory_required {
            return 0.0;
        }

        let waste_ratio = (resource.available_memory - request.memory_required) as f32
            / resource.total_memory as f32;
        1.0 - waste_ratio
    }
}

/// Least loaded strategy
pub struct LeastLoadedStrategy;

impl AllocationStrategyImpl for LeastLoadedStrategy {
    fn select_gpu(
        &self,
        request: &AllocationRequest,
        resources: &HashMap<u32, GpuResource>,
    ) -> Option<u32> {
        resources
            .iter()
            .filter(|(_, resource)| {
                matches!(resource.status, GpuStatus::Available)
                    && resource.available_memory >= request.memory_required
            })
            .min_by(|(_, a), (_, b)| {
                a.utilization.partial_cmp(&b.utilization).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(device_id, _)| *device_id)
    }

    fn calculate_score(&self, resource: &GpuResource, request: &AllocationRequest) -> f32 {
        if resource.available_memory < request.memory_required {
            return 0.0;
        }

        1.0 - resource.utilization
    }
}

/// Performance-based strategy
pub struct PerformanceBasedStrategy;

impl AllocationStrategyImpl for PerformanceBasedStrategy {
    fn select_gpu(
        &self,
        request: &AllocationRequest,
        resources: &HashMap<u32, GpuResource>,
    ) -> Option<u32> {
        resources
            .iter()
            .filter(|(_, resource)| {
                matches!(resource.status, GpuStatus::Available)
                    && resource.available_memory >= request.memory_required
            })
            .max_by(|(_, a), (_, b)| {
                let score_a = self.calculate_score(a, request);
                let score_b = self.calculate_score(b, request);
                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(device_id, _)| *device_id)
    }

    fn calculate_score(&self, resource: &GpuResource, request: &AllocationRequest) -> f32 {
        if resource.available_memory < request.memory_required {
            return 0.0;
        }

        let performance_score =
            resource.performance_profile.fp32_performance * (1.0 - resource.utilization);
        let memory_score = resource.available_memory as f32 / resource.total_memory as f32;

        performance_score * 0.7 + memory_score * 0.3
    }
}

impl DynamicGpuAllocator {
    /// Create new dynamic GPU allocator
    pub fn new(scheduler: Arc<GpuScheduler>, config: DynamicAllocationConfig) -> Self {
        let mut strategies: HashMap<AllocationStrategy, Box<dyn AllocationStrategyImpl>> =
            HashMap::new();
        strategies.insert(
            AllocationStrategy::FirstAvailable,
            Box::new(FirstAvailableStrategy),
        );
        strategies.insert(AllocationStrategy::BestFit, Box::new(BestFitStrategy));
        strategies.insert(
            AllocationStrategy::LeastLoaded,
            Box::new(LeastLoadedStrategy),
        );
        strategies.insert(
            AllocationStrategy::PerformanceBased,
            Box::new(PerformanceBasedStrategy),
        );

        Self {
            gpu_resources: Arc::new(RwLock::new(HashMap::new())),
            allocation_queue: Arc::new(Mutex::new(VecDeque::new())),
            active_allocations: Arc::new(RwLock::new(HashMap::new())),
            config,
            scheduler,
            metrics: Arc::new(Mutex::new(AllocationMetrics::default())),
            queue_notify: Arc::new(Notify::new()),
            resource_notify: Arc::new(Notify::new()),
            strategies,
        }
    }

    /// Initialize GPU resources
    pub async fn initialize_resources(&self) -> Result<(), AllocationError> {
        let mut resources = self.gpu_resources.write().expect("GPU resources RwLock poisoned");

        // Detect available GPUs
        let gpu_count = self.detect_gpu_count().await?;

        for device_id in 0..gpu_count {
            let resource = self.query_gpu_resource(device_id).await?;
            resources.insert(device_id, resource);
        }

        Ok(())
    }

    /// Request GPU allocation
    pub async fn request_allocation(
        &self,
        memory_required: u64,
        compute_required: f32,
        priority: AllocationPriority,
        strategy: AllocationStrategy,
        duration_hint: Option<Duration>,
    ) -> Result<AllocationResult, AllocationError> {
        let id = self.generate_allocation_id();
        let (tx, rx) = tokio::sync::oneshot::channel();

        let request = AllocationRequest {
            id,
            memory_required,
            compute_required,
            priority,
            preferred_devices: None,
            strategy,
            duration_hint,
            requested_at: Utc::now(),
            timeout: self.config.allocation_timeout,
            callback: Some(tx),
        };

        // Try immediate allocation first
        if let Some(allocation) = self.try_immediate_allocation(&request).await? {
            return Ok(AllocationResult::Success(allocation));
        }

        // Queue for later allocation
        self.queue_allocation_request(request).await?;

        // Wait for allocation result
        match tokio::time::timeout(self.config.allocation_timeout, rx).await {
            Ok(Ok(result)) => Ok(result),
            Ok(Err(_)) => Ok(AllocationResult::Failed(AllocationError::Timeout)),
            Err(_) => Ok(AllocationResult::Timeout),
        }
    }

    /// Try immediate allocation
    async fn try_immediate_allocation(
        &self,
        request: &AllocationRequest,
    ) -> Result<Option<GpuAllocation>, AllocationError> {
        let device_id = {
            let resources = self.gpu_resources.read().expect("GPU resources RwLock poisoned");

            let strategy = self.strategies.get(&request.strategy).ok_or(
                AllocationError::InvalidRequest("Invalid strategy".to_string()),
            )?;

            strategy.select_gpu(request, &resources)
        };

        if let Some(device_id) = device_id {
            // Try to allocate on selected device
            if let Ok(allocation) = self.allocate_on_device(request, device_id).await {
                return Ok(Some(allocation));
            }
        }

        Ok(None)
    }

    /// Queue allocation request
    async fn queue_allocation_request(
        &self,
        request: AllocationRequest,
    ) -> Result<(), AllocationError> {
        let mut queue = self.allocation_queue.lock().expect("Allocation queue lock poisoned");

        if queue.len() >= self.config.max_queue_size {
            return Err(AllocationError::ResourceContention);
        }

        // Insert request in priority order
        let insert_index = queue
            .iter()
            .position(|req| req.priority < request.priority)
            .unwrap_or(queue.len());

        queue.insert(insert_index, request);

        // Update metrics
        {
            let mut metrics = self.metrics.lock().expect("Metrics lock poisoned");
            metrics.queue_length = queue.len();
        }

        // Notify queue processor
        self.queue_notify.notify_one();

        Ok(())
    }

    /// Allocate on specific device
    async fn allocate_on_device(
        &self,
        request: &AllocationRequest,
        device_id: u32,
    ) -> Result<GpuAllocation, AllocationError> {
        let mut resources = self.gpu_resources.write().expect("GPU resources RwLock poisoned");

        let resource = resources
            .get_mut(&device_id)
            .ok_or(AllocationError::DeviceError("Device not found".to_string()))?;

        // Check availability
        if resource.available_memory < request.memory_required {
            return Err(AllocationError::InsufficientMemory);
        }

        // Allocate memory
        resource.available_memory -= request.memory_required;

        let allocation = GpuAllocation {
            id: request.id,
            device_id,
            allocated_memory: request.memory_required,
            allocated_at: Utc::now(),
            expected_duration: request.duration_hint,
            allocation_type: AllocationType::Inference,
            session_id: None,
            metadata: HashMap::new(),
        };

        resource.active_allocations.push(allocation.id);

        // Update active allocations
        {
            let mut active =
                self.active_allocations.write().expect("Active allocations RwLock poisoned");
            active.insert(allocation.id, allocation.clone());
        }

        // Update metrics
        {
            let mut metrics = self.metrics.lock().expect("Metrics lock poisoned");
            metrics.total_allocations += 1;
            metrics.successful_allocations += 1;
            metrics.memory_utilization.insert(
                device_id,
                1.0 - (resource.available_memory as f32 / resource.total_memory as f32),
            );
        }

        Ok(allocation)
    }

    /// Release allocation
    pub async fn release_allocation(
        &self,
        allocation_id: AllocationId,
    ) -> Result<(), AllocationError> {
        let allocation = {
            let mut active =
                self.active_allocations.write().expect("Active allocations RwLock poisoned");
            active.remove(&allocation_id).ok_or(AllocationError::InvalidRequest(
                "Allocation not found".to_string(),
            ))?
        };

        // Release GPU memory
        {
            let mut resources = self.gpu_resources.write().expect("GPU resources RwLock poisoned");
            let resource = resources
                .get_mut(&allocation.device_id)
                .ok_or(AllocationError::DeviceError("Device not found".to_string()))?;

            resource.available_memory += allocation.allocated_memory;
            resource.active_allocations.retain(|&id| id != allocation_id);
        }

        // Update metrics
        {
            let mut metrics = self.metrics.lock().expect("Metrics lock poisoned");
            metrics.memory_utilization.insert(allocation.device_id, {
                let resources = self.gpu_resources.read().expect("GPU resources RwLock poisoned");
                let resource =
                    resources.get(&allocation.device_id).expect("Device should exist in resources");
                1.0 - (resource.available_memory as f32 / resource.total_memory as f32)
            });
        }

        // Notify waiting allocations
        self.resource_notify.notify_waiters();

        Ok(())
    }

    /// Process allocation queue
    pub async fn process_allocation_queue(&self) {
        loop {
            // Wait for queue notifications
            self.queue_notify.notified().await;

            let mut processed = 0;

            loop {
                let request = {
                    let mut queue =
                        self.allocation_queue.lock().expect("Allocation queue lock poisoned");
                    queue.pop_front()
                };

                let request = match request {
                    Some(req) => req,
                    None => break,
                };

                // Check if request has timed out
                if Utc::now()
                    .signed_duration_since(request.requested_at)
                    .to_std()
                    .unwrap_or(Duration::from_secs(0))
                    > request.timeout
                {
                    if let Some(callback) = request.callback {
                        let _ = callback.send(AllocationResult::Timeout);
                    }
                    continue;
                }

                // Try to allocate
                match self.try_immediate_allocation(&request).await {
                    Ok(Some(allocation)) => {
                        if let Some(callback) = request.callback {
                            let _ = callback.send(AllocationResult::Success(allocation));
                        }
                        processed += 1;
                    },
                    Ok(None) => {
                        // Put back in queue
                        self.queue_allocation_request(request).await.ok();
                    },
                    Err(error) => {
                        if let Some(callback) = request.callback {
                            let _ = callback.send(AllocationResult::Failed(error));
                        }
                    },
                }
            }

            // Update queue metrics
            {
                let mut metrics = self.metrics.lock().expect("Metrics lock poisoned");
                let queue = self.allocation_queue.lock().expect("Allocation queue lock poisoned");
                metrics.queue_length = queue.len();
            }

            // If no requests were processed, wait for resources
            if processed == 0 {
                tokio::select! {
                    _ = self.resource_notify.notified() => {},
                    _ = tokio::time::sleep(Duration::from_millis(100)) => {},
                }
            }
        }
    }

    /// Start background tasks
    pub async fn start_background_tasks(&self) {
        let allocator = self.clone();
        tokio::spawn(async move {
            allocator.process_allocation_queue().await;
        });

        let allocator = self.clone();
        tokio::spawn(async move {
            allocator.resource_monitoring_loop().await;
        });

        let allocator = self.clone();
        tokio::spawn(async move {
            allocator.rebalancing_loop().await;
        });
    }

    /// Resource monitoring loop
    async fn resource_monitoring_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(1));

        loop {
            interval.tick().await;

            // Update GPU resource information
            let device_ids: Vec<u32> = {
                let resources = self.gpu_resources.read().expect("GPU resources RwLock poisoned");
                resources.keys().cloned().collect()
            };

            for device_id in device_ids {
                if let Ok(updated_resource) = self.query_gpu_resource(device_id).await {
                    let mut resources =
                        self.gpu_resources.write().expect("GPU resources RwLock poisoned");
                    if let Some(resource) = resources.get_mut(&device_id) {
                        // Update utilization and temperature
                        resource.utilization = updated_resource.utilization;
                        resource.temperature = updated_resource.temperature;
                        resource.power_consumption = updated_resource.power_consumption;

                        // Update status based on conditions
                        if resource.temperature > 90.0 {
                            resource.status = GpuStatus::Overheating;
                        } else if resource.available_memory < resource.total_memory / 10 {
                            resource.status = GpuStatus::LowMemory;
                        } else if resource.utilization > 0.95 {
                            resource.status = GpuStatus::Busy;
                        } else {
                            resource.status = GpuStatus::Available;
                        }
                    }
                }
            }
        }
    }

    /// Rebalancing loop
    async fn rebalancing_loop(&self) {
        let mut interval = tokio::time::interval(self.config.rebalancing_interval);

        loop {
            interval.tick().await;

            if let Err(e) = self.rebalance_allocations().await {
                eprintln!("Rebalancing failed: {:?}", e);
            }
        }
    }

    /// Rebalance allocations across GPUs
    async fn rebalance_allocations(&self) -> Result<(), AllocationError> {
        let resources = self.gpu_resources.read().expect("GPU resources RwLock poisoned");

        // Calculate load imbalance
        let total_utilization: f32 = resources.values().map(|r| r.utilization).sum();
        let avg_utilization = total_utilization / resources.len() as f32;

        let mut imbalanced_devices = Vec::new();

        for (device_id, resource) in resources.iter() {
            let deviation = (resource.utilization - avg_utilization).abs();
            if deviation > self.config.utilization_threshold {
                imbalanced_devices.push((*device_id, resource.utilization));
            }
        }

        // If significant imbalance, consider rebalancing
        if imbalanced_devices.len() > 1 {
            // This is a simplified rebalancing - in practice, you'd implement
            // more sophisticated algorithms like live migration
            println!("Rebalancing needed for devices: {:?}", imbalanced_devices);
        }

        Ok(())
    }

    /// Get allocation metrics
    pub fn get_metrics(&self) -> AllocationMetrics {
        let metrics = self.metrics.lock().expect("Metrics lock poisoned");
        metrics.clone()
    }

    /// Helper functions
    fn generate_allocation_id(&self) -> AllocationId {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        COUNTER.fetch_add(1, Ordering::SeqCst)
    }

    async fn detect_gpu_count(&self) -> Result<u32, AllocationError> {
        // Placeholder - in real implementation would use CUDA/ROCm APIs
        Ok(2)
    }

    async fn query_gpu_resource(&self, device_id: u32) -> Result<GpuResource, AllocationError> {
        // Placeholder - in real implementation would query actual GPU
        Ok(GpuResource {
            device_id,
            total_memory: 8 * 1024 * 1024 * 1024,     // 8GB
            available_memory: 6 * 1024 * 1024 * 1024, // 6GB
            compute_capability: (7, 5),
            utilization: 0.3,
            temperature: 65.0,
            power_consumption: 150.0,
            status: GpuStatus::Available,
            active_allocations: Vec::new(),
            reserved_memory: 0,
            performance_profile: PerformanceProfile {
                fp32_performance: 13.0,
                fp16_performance: 26.0,
                int8_performance: 104.0,
                memory_bandwidth: 448.0,
                tensor_cores: true,
            },
        })
    }
}

impl Clone for DynamicGpuAllocator {
    fn clone(&self) -> Self {
        Self {
            gpu_resources: Arc::clone(&self.gpu_resources),
            allocation_queue: Arc::clone(&self.allocation_queue),
            active_allocations: Arc::clone(&self.active_allocations),
            config: self.config.clone(),
            scheduler: Arc::clone(&self.scheduler),
            metrics: Arc::clone(&self.metrics),
            queue_notify: Arc::clone(&self.queue_notify),
            resource_notify: Arc::clone(&self.resource_notify),
            strategies: HashMap::new(), // Strategies are not cloned
        }
    }
}

impl Default for AllocationMetrics {
    fn default() -> Self {
        Self {
            total_allocations: 0,
            successful_allocations: 0,
            failed_allocations: 0,
            avg_allocation_time: Duration::from_millis(0),
            queue_length: 0,
            memory_utilization: HashMap::new(),
            gpu_utilization: HashMap::new(),
            fragmentation_ratio: 0.0,
            preemption_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_scheduler::GpuScheduler;

    #[tokio::test]
    async fn test_dynamic_gpu_allocator_creation() {
        let scheduler = Arc::new(GpuScheduler::new(Default::default()));
        let config = DynamicAllocationConfig::default();
        let allocator = DynamicGpuAllocator::new(scheduler, config);

        assert!(allocator.initialize_resources().await.is_ok());
    }

    #[tokio::test]
    async fn test_allocation_request() {
        let scheduler = Arc::new(GpuScheduler::new(Default::default()));
        let config = DynamicAllocationConfig::default();
        let allocator = DynamicGpuAllocator::new(scheduler, config);

        let _ = allocator.initialize_resources().await;

        let result = allocator
            .request_allocation(
                1024 * 1024 * 1024, // 1GB
                0.5,
                AllocationPriority::Normal,
                AllocationStrategy::FirstAvailable,
                Some(Duration::from_secs(60)),
            )
            .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_allocation_strategies() {
        let strategy = FirstAvailableStrategy;
        let mut resources = HashMap::new();

        resources.insert(
            0,
            GpuResource {
                device_id: 0,
                total_memory: 8 * 1024 * 1024 * 1024,
                available_memory: 6 * 1024 * 1024 * 1024,
                compute_capability: (7, 5),
                utilization: 0.3,
                temperature: 65.0,
                power_consumption: 150.0,
                status: GpuStatus::Available,
                active_allocations: Vec::new(),
                reserved_memory: 0,
                performance_profile: PerformanceProfile {
                    fp32_performance: 13.0,
                    fp16_performance: 26.0,
                    int8_performance: 104.0,
                    memory_bandwidth: 448.0,
                    tensor_cores: true,
                },
            },
        );

        let request = AllocationRequest {
            id: 1,
            memory_required: 1024 * 1024 * 1024,
            compute_required: 0.5,
            priority: AllocationPriority::Normal,
            preferred_devices: None,
            strategy: AllocationStrategy::FirstAvailable,
            duration_hint: None,
            requested_at: Utc::now(),
            timeout: Duration::from_secs(30),
            callback: None,
        };

        let selected = strategy.select_gpu(&request, &resources);
        assert_eq!(selected, Some(0));
    }

    #[tokio::test]
    async fn test_allocation_metrics() {
        let scheduler = Arc::new(GpuScheduler::new(Default::default()));
        let config = DynamicAllocationConfig::default();
        let allocator = DynamicGpuAllocator::new(scheduler, config);

        let metrics = allocator.get_metrics();
        assert_eq!(metrics.total_allocations, 0);
        assert_eq!(metrics.successful_allocations, 0);
        assert_eq!(metrics.failed_allocations, 0);
    }

    #[tokio::test]
    async fn test_gpu_resource_monitoring() {
        let scheduler = Arc::new(GpuScheduler::new(Default::default()));
        let config = DynamicAllocationConfig::default();
        let allocator = DynamicGpuAllocator::new(scheduler, config);

        let _ = allocator.initialize_resources().await;

        // Test resource querying
        let resource = allocator
            .query_gpu_resource(0)
            .await
            .expect("Query GPU resource should succeed");
        assert_eq!(resource.device_id, 0);
        assert!(resource.total_memory > 0);
        assert!(resource.available_memory <= resource.total_memory);
    }

    #[tokio::test]
    async fn test_allocation_priority_ordering() {
        let high_priority = AllocationPriority::High;
        let normal_priority = AllocationPriority::Normal;
        let low_priority = AllocationPriority::Low;

        assert!(high_priority > normal_priority);
        assert!(normal_priority > low_priority);
        assert!(high_priority > low_priority);
    }

    #[tokio::test]
    async fn test_allocation_configuration() {
        let config = DynamicAllocationConfig::default();

        assert!(config.enable_preemption);
        assert_eq!(config.memory_overcommit_ratio, 1.2);
        assert_eq!(config.allocation_timeout, Duration::from_secs(30));
        assert_eq!(config.max_queue_size, 1000);
    }

    #[tokio::test]
    async fn test_performance_based_strategy() {
        let strategy = PerformanceBasedStrategy;
        let mut resources = HashMap::new();

        // Add high-performance GPU
        resources.insert(
            0,
            GpuResource {
                device_id: 0,
                total_memory: 8 * 1024 * 1024 * 1024,
                available_memory: 6 * 1024 * 1024 * 1024,
                compute_capability: (8, 0),
                utilization: 0.1,
                temperature: 60.0,
                power_consumption: 200.0,
                status: GpuStatus::Available,
                active_allocations: Vec::new(),
                reserved_memory: 0,
                performance_profile: PerformanceProfile {
                    fp32_performance: 20.0,
                    fp16_performance: 40.0,
                    int8_performance: 160.0,
                    memory_bandwidth: 900.0,
                    tensor_cores: true,
                },
            },
        );

        // Add lower-performance GPU
        resources.insert(
            1,
            GpuResource {
                device_id: 1,
                total_memory: 4 * 1024 * 1024 * 1024,
                available_memory: 3 * 1024 * 1024 * 1024,
                compute_capability: (7, 0),
                utilization: 0.2,
                temperature: 70.0,
                power_consumption: 150.0,
                status: GpuStatus::Available,
                active_allocations: Vec::new(),
                reserved_memory: 0,
                performance_profile: PerformanceProfile {
                    fp32_performance: 10.0,
                    fp16_performance: 20.0,
                    int8_performance: 80.0,
                    memory_bandwidth: 400.0,
                    tensor_cores: false,
                },
            },
        );

        let request = AllocationRequest {
            id: 1,
            memory_required: 1024 * 1024 * 1024,
            compute_required: 0.8,
            priority: AllocationPriority::High,
            preferred_devices: None,
            strategy: AllocationStrategy::PerformanceBased,
            duration_hint: None,
            requested_at: Utc::now(),
            timeout: Duration::from_secs(30),
            callback: None,
        };

        let selected = strategy.select_gpu(&request, &resources);
        assert_eq!(selected, Some(0)); // Should select the higher-performance GPU
    }
}
