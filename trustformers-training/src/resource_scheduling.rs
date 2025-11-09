use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequest {
    pub request_id: String,
    pub job_id: String,
    pub resource_type: ResourceType,
    pub quantity: u64,
    pub duration: Option<Duration>,
    pub priority: Priority,
    pub constraints: ResourceConstraints,
    pub created_at: u64,
    pub deadline: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceType {
    CPU {
        cores: u32,
    },
    Memory {
        gb: u32,
    },
    GPU {
        count: u32,
        memory_gb: u32,
        model: Option<String>,
    },
    Storage {
        gb: u64,
        speed: StorageSpeed,
    },
    Network {
        bandwidth_mbps: u32,
    },
    Custom {
        name: String,
        units: u64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageSpeed {
    HDD,
    SSD,
    NVMe,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConstraints {
    pub node_affinity: Vec<String>,
    pub node_anti_affinity: Vec<String>,
    pub require_dedicated: bool,
    pub allow_preemption: bool,
    pub max_nodes: Option<u32>,
    pub locality_preference: LocalityPreference,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LocalityPreference {
    None,
    SameRack,
    SameDatacenter,
    SameRegion,
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, Hash, PartialEq)]
pub enum Priority {
    Critical = 4,
    High = 3,
    Normal = 2,
    Low = 1,
    BestEffort = 0,
}

#[derive(Debug, Clone)]
pub struct ResourcePool {
    pub pool_id: String,
    pub name: String,
    pub resource_type: ResourceType,
    pub total_capacity: u64,
    pub available_capacity: u64,
    pub reserved_capacity: u64,
    pub allocations: HashMap<String, ResourceAllocation>,
    pub maintenance_windows: Vec<MaintenanceWindow>,
    pub utilization_history: VecDeque<UtilizationSnapshot>,
    pub cost_per_unit: f64,
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    pub allocation_id: String,
    pub request_id: String,
    pub job_id: String,
    pub allocated_amount: u64,
    pub start_time: Instant,
    pub end_time: Option<Instant>,
    pub actual_usage: Option<u64>,
    pub status: AllocationStatus,
}

#[derive(Debug, Clone)]
pub enum AllocationStatus {
    Pending,
    Active,
    Completed,
    Failed,
    Preempted,
}

#[derive(Debug, Clone)]
pub struct MaintenanceWindow {
    pub window_id: String,
    pub start_time: u64,
    pub end_time: u64,
    pub description: String,
    pub affects_capacity: bool,
    pub capacity_reduction: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct UtilizationSnapshot {
    pub timestamp: Instant,
    pub utilized_capacity: u64,
    pub total_requests: u32,
    pub average_request_size: f64,
}

pub struct ResourceScheduler {
    pools: Arc<RwLock<HashMap<String, ResourcePool>>>,
    pending_requests: Arc<RwLock<BTreeMap<u64, ResourceRequest>>>, // Ordered by priority and timestamp
    active_allocations: Arc<RwLock<HashMap<String, ResourceAllocation>>>,
    scheduling_policies: Arc<RwLock<SchedulingPolicies>>,
    statistics: Arc<RwLock<SchedulingStatistics>>,
    cost_optimizer: Arc<RwLock<CostOptimizer>>,
}

#[derive(Debug, Clone)]
pub struct SchedulingPolicies {
    pub default_scheduling_algorithm: SchedulingAlgorithm,
    pub preemption_enabled: bool,
    pub overcommit_ratio: f32,
    pub fragmentation_threshold: f32,
    pub load_balancing_enabled: bool,
    pub cost_optimization_enabled: bool,
    pub priority_weights: HashMap<Priority, f32>,
}

#[derive(Debug, Clone)]
pub enum SchedulingAlgorithm {
    FirstFit,
    BestFit,
    WorstFit,
    NextFit,
    QuickFit,
    BuddySystem,
    LoadBalanced,
    CostOptimized,
}

#[derive(Debug, Default, Clone)]
pub struct SchedulingStatistics {
    pub total_requests: u64,
    pub successful_allocations: u64,
    pub failed_allocations: u64,
    pub preempted_allocations: u64,
    pub average_allocation_time: Duration,
    pub average_utilization: f64,
    pub fragmentation_ratio: f64,
    pub cost_efficiency: f64,
}

pub struct CostOptimizer {
    pub optimization_strategy: CostOptimizationStrategy,
    pub cost_history: VecDeque<CostSnapshot>,
    pub budget_limit: Option<f64>,
    pub cost_alerts: Vec<CostAlert>,
}

#[derive(Debug, Clone)]
pub enum CostOptimizationStrategy {
    MinimizeCost,
    MaximizeUtilization,
    BalanceCostPerformance,
    SpotInstanceOptimization,
}

#[derive(Debug, Clone)]
pub struct CostSnapshot {
    pub timestamp: Instant,
    pub total_cost: f64,
    pub cost_by_resource_type: HashMap<String, f64>,
    pub efficiency_score: f64,
}

#[derive(Debug, Clone)]
pub struct CostAlert {
    pub alert_id: String,
    pub threshold: f64,
    pub current_value: f64,
    pub message: String,
    pub severity: AlertSeverity,
}

#[derive(Debug, Clone)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

impl Default for ResourceScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceScheduler {
    pub fn new() -> Self {
        Self {
            pools: Arc::new(RwLock::new(HashMap::new())),
            pending_requests: Arc::new(RwLock::new(BTreeMap::new())),
            active_allocations: Arc::new(RwLock::new(HashMap::new())),
            scheduling_policies: Arc::new(RwLock::new(SchedulingPolicies::default())),
            statistics: Arc::new(RwLock::new(SchedulingStatistics::default())),
            cost_optimizer: Arc::new(RwLock::new(CostOptimizer::new())),
        }
    }

    pub fn register_resource_pool(&self, pool: ResourcePool) -> Result<()> {
        let mut pools = self
            .pools
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on pools"))?;

        pools.insert(pool.pool_id.clone(), pool);
        Ok(())
    }

    pub fn submit_resource_request(&self, mut request: ResourceRequest) -> Result<String> {
        request.created_at = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

        let priority_score = self.calculate_priority_score(&request)?;

        {
            let mut pending = self
                .pending_requests
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on pending requests"))?;
            pending.insert(priority_score, request.clone());
        }

        // Update statistics
        {
            let mut stats = self
                .statistics
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on statistics"))?;
            stats.total_requests += 1;
        }

        // Try immediate scheduling
        self.schedule_pending_requests()?;

        Ok(request.request_id)
    }

    fn calculate_priority_score(&self, request: &ResourceRequest) -> Result<u64> {
        let policies = self
            .scheduling_policies
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on scheduling policies"))?;

        let priority_weight = policies.priority_weights.get(&request.priority).unwrap_or(&1.0);

        // Combine priority, age, and deadline urgency
        let age_factor = (SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
            - request.created_at) as f64;

        let deadline_urgency = if let Some(deadline) = request.deadline {
            let time_to_deadline = deadline
                .saturating_sub(SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs())
                as f64;
            1.0 / (time_to_deadline + 1.0) // Higher urgency as deadline approaches
        } else {
            0.0
        };

        let score =
            (*priority_weight as f64 * 1000000.0) + age_factor + (deadline_urgency * 100000.0);
        Ok(score as u64)
    }

    pub fn schedule_pending_requests(&self) -> Result<Vec<String>> {
        let mut allocated_requests = Vec::new();

        let pending_requests = {
            let pending = self
                .pending_requests
                .read()
                .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on pending requests"))?;
            pending.values().cloned().collect::<Vec<_>>()
        };

        for request in pending_requests.iter().rev() {
            // Process highest priority first
            if let Ok(allocation_id) = self.try_allocate_request(request) {
                allocated_requests.push(allocation_id);

                // Remove from pending
                let priority_score = self.calculate_priority_score(request)?;
                let mut pending = self.pending_requests.write().map_err(|_| {
                    anyhow::anyhow!("Failed to acquire write lock on pending requests")
                })?;
                pending.remove(&priority_score);
            }
        }

        Ok(allocated_requests)
    }

    fn try_allocate_request(&self, request: &ResourceRequest) -> Result<String> {
        let policies = self
            .scheduling_policies
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on scheduling policies"))?;

        match policies.default_scheduling_algorithm {
            SchedulingAlgorithm::FirstFit => self.first_fit_allocation(request),
            SchedulingAlgorithm::BestFit => self.best_fit_allocation(request),
            SchedulingAlgorithm::WorstFit => self.worst_fit_allocation(request),
            SchedulingAlgorithm::LoadBalanced => self.load_balanced_allocation(request),
            SchedulingAlgorithm::CostOptimized => self.cost_optimized_allocation(request),
            _ => self.first_fit_allocation(request), // Default fallback
        }
    }

    fn first_fit_allocation(&self, request: &ResourceRequest) -> Result<String> {
        let pools = self
            .pools
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on pools"))?;

        for (pool_id, pool) in pools.iter() {
            if self.can_satisfy_request(pool, request)? {
                let pool_id_clone = pool_id.clone();
                drop(pools); // Release read lock before acquiring write lock
                return self.create_allocation(&pool_id_clone, request);
            }
        }

        Err(anyhow::anyhow!("No suitable pool found for request"))
    }

    fn best_fit_allocation(&self, request: &ResourceRequest) -> Result<String> {
        let pools = self
            .pools
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on pools"))?;

        let mut best_pool = None;
        let mut min_waste = u64::MAX;

        for (pool_id, pool) in pools.iter() {
            if self.can_satisfy_request(pool, request)? {
                let waste = pool.available_capacity.saturating_sub(request.quantity);
                if waste < min_waste {
                    min_waste = waste;
                    best_pool = Some(pool_id.clone());
                }
            }
        }

        if let Some(pool_id) = best_pool {
            drop(pools);
            self.create_allocation(&pool_id, request)
        } else {
            Err(anyhow::anyhow!("No suitable pool found for request"))
        }
    }

    fn worst_fit_allocation(&self, request: &ResourceRequest) -> Result<String> {
        let pools = self
            .pools
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on pools"))?;

        let mut best_pool = None;
        let mut max_remaining = 0;

        for (pool_id, pool) in pools.iter() {
            if self.can_satisfy_request(pool, request)? {
                let remaining = pool.available_capacity.saturating_sub(request.quantity);
                if remaining > max_remaining {
                    max_remaining = remaining;
                    best_pool = Some(pool_id.clone());
                }
            }
        }

        if let Some(pool_id) = best_pool {
            drop(pools);
            self.create_allocation(&pool_id, request)
        } else {
            Err(anyhow::anyhow!("No suitable pool found for request"))
        }
    }

    fn load_balanced_allocation(&self, request: &ResourceRequest) -> Result<String> {
        let pools = self
            .pools
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on pools"))?;

        let mut best_pool = None;
        let mut min_utilization = f64::MAX;

        for (pool_id, pool) in pools.iter() {
            if self.can_satisfy_request(pool, request)? {
                let utilization = (pool.total_capacity - pool.available_capacity) as f64
                    / pool.total_capacity as f64;
                if utilization < min_utilization {
                    min_utilization = utilization;
                    best_pool = Some(pool_id.clone());
                }
            }
        }

        if let Some(pool_id) = best_pool {
            drop(pools);
            self.create_allocation(&pool_id, request)
        } else {
            Err(anyhow::anyhow!("No suitable pool found for request"))
        }
    }

    fn cost_optimized_allocation(&self, request: &ResourceRequest) -> Result<String> {
        let pools = self
            .pools
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on pools"))?;

        let mut best_pool = None;
        let mut min_cost = f64::MAX;

        for (pool_id, pool) in pools.iter() {
            if self.can_satisfy_request(pool, request)? {
                let cost = pool.cost_per_unit * request.quantity as f64;
                if cost < min_cost {
                    min_cost = cost;
                    best_pool = Some(pool_id.clone());
                }
            }
        }

        if let Some(pool_id) = best_pool {
            drop(pools);
            self.create_allocation(&pool_id, request)
        } else {
            Err(anyhow::anyhow!("No suitable pool found for request"))
        }
    }

    fn can_satisfy_request(&self, pool: &ResourcePool, request: &ResourceRequest) -> Result<bool> {
        // Check basic capacity
        if pool.available_capacity < request.quantity {
            return Ok(false);
        }

        // Check resource type compatibility
        if !self.is_resource_type_compatible(&pool.resource_type, &request.resource_type) {
            return Ok(false);
        }

        // Check constraints
        if !request.constraints.node_affinity.is_empty() {
            let has_affinity =
                request.constraints.node_affinity.iter().any(|tag| pool.tags.contains_key(tag));
            if !has_affinity {
                return Ok(false);
            }
        }

        if !request.constraints.node_anti_affinity.is_empty() {
            let has_anti_affinity = request
                .constraints
                .node_anti_affinity
                .iter()
                .any(|tag| pool.tags.contains_key(tag));
            if has_anti_affinity {
                return Ok(false);
            }
        }

        // Check maintenance windows
        if self.is_maintenance_conflicting(pool, request)? {
            return Ok(false);
        }

        Ok(true)
    }

    fn is_resource_type_compatible(
        &self,
        pool_type: &ResourceType,
        request_type: &ResourceType,
    ) -> bool {
        match (pool_type, request_type) {
            (ResourceType::CPU { .. }, ResourceType::CPU { .. }) => true,
            (ResourceType::Memory { .. }, ResourceType::Memory { .. }) => true,
            (ResourceType::GPU { .. }, ResourceType::GPU { .. }) => true,
            (ResourceType::Storage { .. }, ResourceType::Storage { .. }) => true,
            (ResourceType::Network { .. }, ResourceType::Network { .. }) => true,
            (
                ResourceType::Custom {
                    name: pool_name, ..
                },
                ResourceType::Custom { name: req_name, .. },
            ) => pool_name == req_name,
            _ => false,
        }
    }

    fn is_maintenance_conflicting(
        &self,
        pool: &ResourcePool,
        request: &ResourceRequest,
    ) -> Result<bool> {
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();

        let request_end_time = if let Some(duration) = request.duration {
            current_time + duration.as_secs()
        } else {
            request.deadline.unwrap_or(current_time + 3600) // Default 1 hour if no duration or deadline
        };

        for window in &pool.maintenance_windows {
            if window.affects_capacity
                && current_time < window.end_time
                && request_end_time > window.start_time
            {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn create_allocation(&self, pool_id: &str, request: &ResourceRequest) -> Result<String> {
        let allocation_id = uuid::Uuid::new_v4().to_string();

        // Update pool capacity
        {
            let mut pools = self
                .pools
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on pools"))?;

            if let Some(pool) = pools.get_mut(pool_id) {
                if pool.available_capacity < request.quantity {
                    return Err(anyhow::anyhow!("Insufficient capacity"));
                }

                pool.available_capacity -= request.quantity;

                let allocation = ResourceAllocation {
                    allocation_id: allocation_id.clone(),
                    request_id: request.request_id.clone(),
                    job_id: request.job_id.clone(),
                    allocated_amount: request.quantity,
                    start_time: Instant::now(),
                    end_time: request.duration.map(|d| Instant::now() + d),
                    actual_usage: None,
                    status: AllocationStatus::Active,
                };

                pool.allocations.insert(allocation_id.clone(), allocation.clone());

                // Add to active allocations
                let mut active = self.active_allocations.write().map_err(|_| {
                    anyhow::anyhow!("Failed to acquire write lock on active allocations")
                })?;
                active.insert(allocation_id.clone(), allocation);
            } else {
                return Err(anyhow::anyhow!("Pool not found: {}", pool_id));
            }
        }

        // Update statistics
        {
            let mut stats = self
                .statistics
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on statistics"))?;
            stats.successful_allocations += 1;
        }

        Ok(allocation_id)
    }

    pub fn release_allocation(&self, allocation_id: &str) -> Result<()> {
        // Find and remove from active allocations
        let allocation = {
            let mut active = self.active_allocations.write().map_err(|_| {
                anyhow::anyhow!("Failed to acquire write lock on active allocations")
            })?;
            active
                .remove(allocation_id)
                .ok_or_else(|| anyhow::anyhow!("Allocation not found: {}", allocation_id))?
        };

        // Find the pool and release capacity
        {
            let mut pools = self
                .pools
                .write()
                .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on pools"))?;

            for pool in pools.values_mut() {
                if let Some(mut pool_allocation) = pool.allocations.remove(allocation_id) {
                    pool_allocation.status = AllocationStatus::Completed;
                    pool_allocation.end_time = Some(Instant::now());

                    pool.available_capacity += allocation.allocated_amount;
                    break;
                }
            }
        }

        // Try to schedule pending requests now that resources are available
        self.schedule_pending_requests()?;

        Ok(())
    }

    pub fn update_actual_usage(&self, allocation_id: &str, actual_usage: u64) -> Result<()> {
        let mut active = self
            .active_allocations
            .write()
            .map_err(|_| anyhow::anyhow!("Failed to acquire write lock on active allocations"))?;

        if let Some(allocation) = active.get_mut(allocation_id) {
            allocation.actual_usage = Some(actual_usage);
        } else {
            return Err(anyhow::anyhow!("Allocation not found: {}", allocation_id));
        }

        Ok(())
    }

    pub fn get_resource_utilization(&self) -> Result<HashMap<String, f64>> {
        let pools = self
            .pools
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on pools"))?;

        let mut utilization = HashMap::new();

        for (pool_id, pool) in pools.iter() {
            let util =
                (pool.total_capacity - pool.available_capacity) as f64 / pool.total_capacity as f64;
            utilization.insert(pool_id.clone(), util);
        }

        Ok(utilization)
    }

    pub fn get_pending_requests_count(&self) -> Result<usize> {
        let pending = self
            .pending_requests
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on pending requests"))?;
        Ok(pending.len())
    }

    pub fn get_active_allocations_count(&self) -> Result<usize> {
        let active = self
            .active_allocations
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on active allocations"))?;
        Ok(active.len())
    }

    pub fn get_statistics(&self) -> Result<SchedulingStatistics> {
        let stats = self
            .statistics
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on statistics"))?;
        Ok((*stats).clone())
    }

    pub fn optimize_costs(&self) -> Result<Vec<CostOptimizationRecommendation>> {
        let _cost_optimizer = self
            .cost_optimizer
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on cost optimizer"))?;

        let pools = self
            .pools
            .read()
            .map_err(|_| anyhow::anyhow!("Failed to acquire read lock on pools"))?;

        let mut recommendations = Vec::new();

        // Identify underutilized resources
        for (pool_id, pool) in pools.iter() {
            let utilization =
                (pool.total_capacity - pool.available_capacity) as f64 / pool.total_capacity as f64;

            if utilization < 0.3 {
                // Less than 30% utilized
                recommendations.push(CostOptimizationRecommendation {
                    pool_id: pool_id.clone(),
                    recommendation_type: RecommendationType::ScaleDown,
                    description: format!(
                        "Pool {} is only {:.1}% utilized. Consider scaling down.",
                        pool_id,
                        utilization * 100.0
                    ),
                    potential_savings: pool.cost_per_unit * (pool.available_capacity as f64 * 0.7),
                });
            }

            if utilization > 0.9 {
                // More than 90% utilized
                recommendations.push(CostOptimizationRecommendation {
                    pool_id: pool_id.clone(),
                    recommendation_type: RecommendationType::ScaleUp,
                    description: format!(
                        "Pool {} is {:.1}% utilized. Consider scaling up to avoid bottlenecks.",
                        pool_id,
                        utilization * 100.0
                    ),
                    potential_savings: 0.0, // This is actually a cost increase but prevents performance issues
                });
            }
        }

        Ok(recommendations)
    }
}

#[derive(Debug, Clone)]
pub struct CostOptimizationRecommendation {
    pub pool_id: String,
    pub recommendation_type: RecommendationType,
    pub description: String,
    pub potential_savings: f64,
}

#[derive(Debug, Clone)]
pub enum RecommendationType {
    ScaleUp,
    ScaleDown,
    ChangeInstanceType,
    UseSpotInstances,
    ConsolidateWorkloads,
}

impl SchedulingPolicies {
    fn default() -> Self {
        let mut priority_weights = HashMap::new();
        priority_weights.insert(Priority::Critical, 4.0);
        priority_weights.insert(Priority::High, 3.0);
        priority_weights.insert(Priority::Normal, 2.0);
        priority_weights.insert(Priority::Low, 1.0);
        priority_weights.insert(Priority::BestEffort, 0.5);

        Self {
            default_scheduling_algorithm: SchedulingAlgorithm::FirstFit,
            preemption_enabled: false,
            overcommit_ratio: 1.0,
            fragmentation_threshold: 0.1,
            load_balancing_enabled: true,
            cost_optimization_enabled: false,
            priority_weights,
        }
    }
}

impl CostOptimizer {
    fn new() -> Self {
        Self {
            optimization_strategy: CostOptimizationStrategy::BalanceCostPerformance,
            cost_history: VecDeque::with_capacity(1000),
            budget_limit: None,
            cost_alerts: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_scheduler_creation() {
        let scheduler = ResourceScheduler::new();
        assert_eq!(scheduler.get_pending_requests_count().unwrap(), 0);
        assert_eq!(scheduler.get_active_allocations_count().unwrap(), 0);
    }

    #[test]
    fn test_resource_pool_registration() {
        let scheduler = ResourceScheduler::new();

        let pool = ResourcePool {
            pool_id: "cpu-pool-1".to_string(),
            name: "CPU Pool 1".to_string(),
            resource_type: ResourceType::CPU { cores: 16 },
            total_capacity: 16,
            available_capacity: 16,
            reserved_capacity: 0,
            allocations: HashMap::new(),
            maintenance_windows: Vec::new(),
            utilization_history: VecDeque::new(),
            cost_per_unit: 0.1,
            tags: HashMap::new(),
        };

        scheduler.register_resource_pool(pool).unwrap();

        let utilization = scheduler.get_resource_utilization().unwrap();
        assert_eq!(utilization.get("cpu-pool-1"), Some(&0.0));
    }

    #[test]
    fn test_resource_request_submission() {
        let scheduler = ResourceScheduler::new();

        // Register a pool first
        let pool = ResourcePool {
            pool_id: "cpu-pool-1".to_string(),
            name: "CPU Pool 1".to_string(),
            resource_type: ResourceType::CPU { cores: 16 },
            total_capacity: 16,
            available_capacity: 16,
            reserved_capacity: 0,
            allocations: HashMap::new(),
            maintenance_windows: Vec::new(),
            utilization_history: VecDeque::new(),
            cost_per_unit: 0.1,
            tags: HashMap::new(),
        };
        scheduler.register_resource_pool(pool).unwrap();

        let request = ResourceRequest {
            request_id: "req-1".to_string(),
            job_id: "job-1".to_string(),
            resource_type: ResourceType::CPU { cores: 4 },
            quantity: 4,
            duration: Some(Duration::from_secs(3600)),
            priority: Priority::Normal,
            constraints: ResourceConstraints {
                node_affinity: vec![],
                node_anti_affinity: vec![],
                require_dedicated: false,
                allow_preemption: false,
                max_nodes: None,
                locality_preference: LocalityPreference::None,
            },
            created_at: 0,
            deadline: None,
        };

        let request_id = scheduler.submit_resource_request(request).unwrap();
        assert_eq!(request_id, "req-1");

        // Check that resources were allocated
        assert_eq!(scheduler.get_active_allocations_count().unwrap(), 1);

        let utilization = scheduler.get_resource_utilization().unwrap();
        assert_eq!(utilization.get("cpu-pool-1"), Some(&0.25)); // 4/16 = 0.25
    }

    #[test]
    fn test_allocation_release() {
        let scheduler = ResourceScheduler::new();

        // Register a pool
        let pool = ResourcePool {
            pool_id: "cpu-pool-1".to_string(),
            name: "CPU Pool 1".to_string(),
            resource_type: ResourceType::CPU { cores: 16 },
            total_capacity: 16,
            available_capacity: 16,
            reserved_capacity: 0,
            allocations: HashMap::new(),
            maintenance_windows: Vec::new(),
            utilization_history: VecDeque::new(),
            cost_per_unit: 0.1,
            tags: HashMap::new(),
        };
        scheduler.register_resource_pool(pool).unwrap();

        // Submit a request
        let request = ResourceRequest {
            request_id: "req-1".to_string(),
            job_id: "job-1".to_string(),
            resource_type: ResourceType::CPU { cores: 4 },
            quantity: 4,
            duration: Some(Duration::from_secs(3600)),
            priority: Priority::Normal,
            constraints: ResourceConstraints {
                node_affinity: vec![],
                node_anti_affinity: vec![],
                require_dedicated: false,
                allow_preemption: false,
                max_nodes: None,
                locality_preference: LocalityPreference::None,
            },
            created_at: 0,
            deadline: None,
        };

        scheduler.submit_resource_request(request).unwrap();

        // Get allocation ID (in a real implementation, this would be returned)
        let active_allocations = scheduler.active_allocations.read().unwrap();
        let allocation_id = active_allocations.keys().next().unwrap().clone();
        drop(active_allocations);

        // Release the allocation
        scheduler.release_allocation(&allocation_id).unwrap();

        // Check that resources were freed
        assert_eq!(scheduler.get_active_allocations_count().unwrap(), 0);

        let utilization = scheduler.get_resource_utilization().unwrap();
        assert_eq!(utilization.get("cpu-pool-1"), Some(&0.0));
    }

    #[test]
    fn test_priority_scheduling() {
        let scheduler = ResourceScheduler::new();

        // Register a small pool to force queueing
        let pool = ResourcePool {
            pool_id: "cpu-pool-1".to_string(),
            name: "CPU Pool 1".to_string(),
            resource_type: ResourceType::CPU { cores: 4 },
            total_capacity: 4,
            available_capacity: 4,
            reserved_capacity: 0,
            allocations: HashMap::new(),
            maintenance_windows: Vec::new(),
            utilization_history: VecDeque::new(),
            cost_per_unit: 0.1,
            tags: HashMap::new(),
        };
        scheduler.register_resource_pool(pool).unwrap();

        // Submit low priority request first
        let low_priority_request = ResourceRequest {
            request_id: "req-low".to_string(),
            job_id: "job-low".to_string(),
            resource_type: ResourceType::CPU { cores: 4 },
            quantity: 4,
            duration: Some(Duration::from_secs(3600)),
            priority: Priority::Low,
            constraints: ResourceConstraints {
                node_affinity: vec![],
                node_anti_affinity: vec![],
                require_dedicated: false,
                allow_preemption: false,
                max_nodes: None,
                locality_preference: LocalityPreference::None,
            },
            created_at: 0,
            deadline: None,
        };

        scheduler.submit_resource_request(low_priority_request).unwrap();

        // Submit high priority request that should be prioritized
        let high_priority_request = ResourceRequest {
            request_id: "req-high".to_string(),
            job_id: "job-high".to_string(),
            resource_type: ResourceType::CPU { cores: 2 },
            quantity: 2,
            duration: Some(Duration::from_secs(1800)),
            priority: Priority::High,
            constraints: ResourceConstraints {
                node_affinity: vec![],
                node_anti_affinity: vec![],
                require_dedicated: false,
                allow_preemption: false,
                max_nodes: None,
                locality_preference: LocalityPreference::None,
            },
            created_at: 0,
            deadline: None,
        };

        scheduler.submit_resource_request(high_priority_request).unwrap();

        // In a real scenario, we would verify that the high priority request
        // gets allocated first when resources become available
        assert!(scheduler.get_pending_requests_count().unwrap() > 0);
    }
}
