//! Resource allocation coordination and conflict detection.

// Re-export types for external access
pub use super::types::{AllocationEvent, ExecutionState, ExecutionStatus};
use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use std::{collections::HashMap, sync::Arc, time::Duration};
use tracing::{debug, info};

use crate::parallel_execution_engine::ResourceRequirement;
use crate::test_parallelization::{ConflictResolutionConfig, ResourceAllocation};

/// Resource allocator for coordinating resource allocation
#[derive(Debug)]
pub struct ResourceAllocator {
    /// Active allocations
    active_allocations: Arc<Mutex<HashMap<String, ResourceAllocation>>>,
    /// Allocation history
    allocation_history: Arc<Mutex<Vec<AllocationEvent>>>,
    /// Allocation statistics
    allocation_stats: Arc<Mutex<AllocationStatistics>>,
}

/// Conflict detector for identifying resource conflicts
#[derive(Debug)]
pub struct ConflictDetector {
    /// Configuration
    config: Arc<Mutex<ConflictResolutionConfig>>,
    /// Conflict detection rules
    detection_rules: Arc<Mutex<Vec<ConflictDetectionRule>>>,
    /// Conflict history
    conflict_history: Arc<Mutex<Vec<ConflictEvent>>>,
}

/// Allocation statistics
#[derive(Debug, Default, Clone)]
pub struct AllocationStatistics {
    /// Total allocations performed
    pub total_allocations: u64,
    /// Currently active allocations
    pub active_allocations: usize,
    /// Failed allocations
    pub failed_allocations: u64,
    /// Average allocation duration
    pub average_allocation_duration: Duration,
    /// Resource utilization efficiency
    pub utilization_efficiency: f32,
    /// Conflict rate
    pub conflict_rate: f32,
}

/// Conflict detection rule
#[derive(Debug, Clone)]
pub struct ConflictDetectionRule {
    /// Rule name
    pub name: String,
    /// Resource types this rule applies to
    pub resource_types: Vec<String>,
    /// Conflict detection logic
    pub detection_logic: ConflictDetectionLogic,
    /// Rule priority
    pub priority: f32,
    /// Rule enabled
    pub enabled: bool,
}

/// Conflict detection logic types
#[derive(Debug, Clone)]
pub enum ConflictDetectionLogic {
    /// Port range overlap detection
    PortRangeOverlap,
    /// Directory path conflict
    DirectoryPathConflict,
    /// GPU device exclusive access
    GpuExclusiveAccess,
    /// Database connection limit
    DatabaseConnectionLimit,
    /// Memory usage limit
    MemoryUsageLimit,
    /// CPU usage limit
    CpuUsageLimit,
    /// Custom conflict logic
    Custom(String),
}

/// Conflict event record
#[derive(Debug, Clone)]
pub struct ConflictEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Conflict type
    pub conflict_type: ConflictType,
    /// Conflicting test IDs
    pub conflicting_tests: Vec<String>,
    /// Resource involved in conflict
    pub resource_id: String,
    /// Conflict severity
    pub severity: ConflictSeverity,
    /// Resolution applied
    pub resolution: Option<ConflictResolution>,
    /// Event details
    pub details: HashMap<String, String>,
}

/// Types of resource conflicts
#[derive(Debug, Clone)]
pub enum ConflictType {
    /// Port already allocated
    PortConflict,
    /// Directory path collision
    DirectoryConflict,
    /// GPU device busy
    GpuConflict,
    /// Database connection exhaustion
    DatabaseConflict,
    /// Memory limit exceeded
    MemoryConflict,
    /// CPU limit exceeded
    CpuConflict,
    /// Custom conflict type
    Custom(String),
}

/// Conflict severity levels
#[derive(Debug, Clone)]
pub enum ConflictSeverity {
    /// Low severity - can be resolved automatically
    Low,
    /// Medium severity - may require intervention
    Medium,
    /// High severity - requires immediate attention
    High,
    /// Critical severity - system integrity at risk
    Critical,
}

/// Conflict resolution strategies
#[derive(Debug, Clone)]
pub enum ConflictResolution {
    /// Retry allocation with different resources
    RetryWithAlternative,
    /// Queue the allocation for later
    QueueForLater,
    /// Reject the allocation
    Reject,
    /// Force allocation (override conflicts)
    ForceAllocate,
    /// Scale up resources
    ScaleUp,
    /// Custom resolution
    Custom(String),
}

/// Resource allocation coordinator
#[derive(Debug)]
pub struct AllocationCoordinator {
    /// Resource allocator
    allocator: Arc<ResourceAllocator>,
    /// Conflict detector
    conflict_detector: Arc<ConflictDetector>,
    /// Allocation queue
    allocation_queue: Arc<Mutex<Vec<AllocationRequest>>>,
    /// Coordination statistics
    coordination_stats: Arc<Mutex<CoordinationStatistics>>,
}

/// Allocation request
#[derive(Debug, Clone)]
pub struct AllocationRequest {
    /// Request ID
    pub request_id: String,
    /// Test ID requesting resources
    pub test_id: String,
    /// Resource requirements
    pub requirements: ResourceRequirement,
    /// Request timestamp
    pub requested_at: DateTime<Utc>,
    /// Request priority
    pub priority: f32,
    /// Timeout for allocation
    pub timeout: Duration,
    /// Retry count
    pub retry_count: usize,
}

/// Coordination statistics
#[derive(Debug, Default, Clone)]
pub struct CoordinationStatistics {
    /// Total requests processed
    pub total_requests: u64,
    /// Successful allocations
    pub successful_allocations: u64,
    /// Failed allocations
    pub failed_allocations: u64,
    /// Conflicts detected
    pub conflicts_detected: u64,
    /// Conflicts resolved
    pub conflicts_resolved: u64,
    /// Average resolution time
    pub average_resolution_time: Duration,
}

impl ResourceAllocator {
    /// Create new resource allocator
    pub async fn new() -> Result<Self> {
        Ok(Self {
            active_allocations: Arc::new(Mutex::new(HashMap::new())),
            allocation_history: Arc::new(Mutex::new(Vec::new())),
            allocation_stats: Arc::new(Mutex::new(AllocationStatistics::default())),
        })
    }

    /// Track a new allocation
    pub async fn track_allocation(&self, allocation: &ResourceAllocation) -> Result<()> {
        debug!("Tracking allocation: {}", allocation.resource_id);

        // Add to active allocations
        let mut active_allocations = self.active_allocations.lock();
        active_allocations.insert(allocation.resource_id.clone(), allocation.clone());

        // Record allocation event
        let event = AllocationEvent {
            timestamp: Utc::now(),
            resource_id: allocation.resource_id.clone(),
            test_id: format!("test-{}", allocation.resource_id), // Extract from resource_id
            event_type: "allocation_tracked".to_string(),
            details: HashMap::new(),
        };

        let mut allocation_history = self.allocation_history.lock();
        allocation_history.push(event);

        // Update statistics
        let mut stats = self.allocation_stats.lock();
        stats.total_allocations += 1;
        stats.active_allocations = active_allocations.len();

        info!(
            "Successfully tracked allocation: {}",
            allocation.resource_id
        );
        Ok(())
    }

    /// Mark allocation as deallocated
    pub async fn mark_deallocated(&self, allocation: &ResourceAllocation) -> Result<()> {
        debug!(
            "Marking allocation as deallocated: {}",
            allocation.resource_id
        );

        // Remove from active allocations
        let mut active_allocations = self.active_allocations.lock();
        active_allocations.remove(&allocation.resource_id);

        // Record deallocation event
        let event = AllocationEvent {
            timestamp: Utc::now(),
            resource_id: allocation.resource_id.clone(),
            test_id: format!("test-{}", allocation.resource_id), // Extract from resource_id
            event_type: "allocation_deallocated".to_string(),
            details: HashMap::new(),
        };

        let mut allocation_history = self.allocation_history.lock();
        allocation_history.push(event);

        // Update statistics
        let mut stats = self.allocation_stats.lock();
        stats.active_allocations = active_allocations.len();

        info!(
            "Successfully marked allocation as deallocated: {}",
            allocation.resource_id
        );
        Ok(())
    }

    /// Untrack allocation for a test
    pub async fn untrack_allocation(&self, test_id: &str) -> Result<()> {
        debug!("Untracking allocations for test: {}", test_id);

        let mut active_allocations = self.active_allocations.lock();

        // Find and remove allocations for this test
        let allocations_to_remove: Vec<String> = active_allocations
            .iter()
            .filter(|(_, allocation)| {
                // Simple test ID extraction from resource_id
                allocation.resource_id.contains(test_id)
            })
            .map(|(resource_id, _)| resource_id.clone())
            .collect();

        for resource_id in &allocations_to_remove {
            active_allocations.remove(resource_id);

            // Record untrack event
            let event = AllocationEvent {
                timestamp: Utc::now(),
                resource_id: resource_id.clone(),
                test_id: test_id.to_string(),
                event_type: "allocation_untracked".to_string(),
                details: HashMap::new(),
            };

            let mut allocation_history = self.allocation_history.lock();
            allocation_history.push(event);
        }

        // Update statistics
        let mut stats = self.allocation_stats.lock();
        stats.active_allocations = active_allocations.len();

        info!(
            "Untracked {} allocations for test: {}",
            allocations_to_remove.len(),
            test_id
        );
        Ok(())
    }

    /// Get allocation statistics
    pub async fn get_statistics(&self) -> Result<AllocationStatistics> {
        let stats = self.allocation_stats.lock();
        // MutexGuard doesn't implement Clone, dereference to clone the inner value
        Ok((*stats).clone())
    }

    /// Get active allocations
    pub async fn get_active_allocations(&self) -> Result<Vec<ResourceAllocation>> {
        let active_allocations = self.active_allocations.lock();
        Ok(active_allocations.values().cloned().collect())
    }

    /// Get allocation history
    pub async fn get_allocation_history(&self) -> Result<Vec<AllocationEvent>> {
        let allocation_history = self.allocation_history.lock();
        Ok(allocation_history.clone())
    }
}

impl ConflictDetector {
    /// Create new conflict detector
    pub async fn new(config: ConflictResolutionConfig) -> Result<Self> {
        Ok(Self {
            config: Arc::new(Mutex::new(config)),
            detection_rules: Arc::new(Mutex::new(Self::default_detection_rules())),
            conflict_history: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Check for conflicts with resource requirements
    pub async fn check_conflicts(
        &self,
        _requirements: &ResourceRequirement,
        test_id: &str,
    ) -> Result<Option<String>> {
        debug!("Checking conflicts for test: {}", test_id);

        // In a real implementation, this would:
        // 1. Check each detection rule against current allocations
        // 2. Identify potential conflicts
        // 3. Return conflict details or None

        // For now, return no conflicts
        Ok(None)
    }

    /// Add a conflict detection rule
    pub async fn add_detection_rule(&self, rule: ConflictDetectionRule) -> Result<()> {
        let mut detection_rules = self.detection_rules.lock();
        detection_rules.push(rule);
        Ok(())
    }

    /// Remove a conflict detection rule
    pub async fn remove_detection_rule(&self, rule_name: &str) -> Result<bool> {
        let mut detection_rules = self.detection_rules.lock();
        let initial_len = detection_rules.len();
        detection_rules.retain(|rule| rule.name != rule_name);
        Ok(detection_rules.len() < initial_len)
    }

    /// Get conflict history
    pub async fn get_conflict_history(&self) -> Result<Vec<ConflictEvent>> {
        let conflict_history = self.conflict_history.lock();
        Ok(conflict_history.clone())
    }

    /// Default detection rules
    fn default_detection_rules() -> Vec<ConflictDetectionRule> {
        vec![
            ConflictDetectionRule {
                name: "port_range_overlap".to_string(),
                resource_types: vec!["network_port".to_string()],
                detection_logic: ConflictDetectionLogic::PortRangeOverlap,
                priority: 1.0,
                enabled: true,
            },
            ConflictDetectionRule {
                name: "directory_path_conflict".to_string(),
                resource_types: vec!["temp_directory".to_string()],
                detection_logic: ConflictDetectionLogic::DirectoryPathConflict,
                priority: 1.0,
                enabled: true,
            },
            ConflictDetectionRule {
                name: "gpu_exclusive_access".to_string(),
                resource_types: vec!["gpu_device".to_string()],
                detection_logic: ConflictDetectionLogic::GpuExclusiveAccess,
                priority: 1.0,
                enabled: true,
            },
        ]
    }
}

impl AllocationCoordinator {
    /// Create new allocation coordinator
    pub async fn new(
        allocator: Arc<ResourceAllocator>,
        conflict_detector: Arc<ConflictDetector>,
    ) -> Result<Self> {
        Ok(Self {
            allocator,
            conflict_detector,
            allocation_queue: Arc::new(Mutex::new(Vec::new())),
            coordination_stats: Arc::new(Mutex::new(CoordinationStatistics::default())),
        })
    }

    /// Submit allocation request
    pub async fn submit_allocation_request(&self, request: AllocationRequest) -> Result<()> {
        info!("Submitting allocation request: {}", request.request_id);

        let mut allocation_queue = self.allocation_queue.lock();
        allocation_queue.push(request);

        // Sort by priority (highest first)
        allocation_queue.sort_by(|a, b| {
            b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(())
    }

    /// Process allocation queue
    pub async fn process_allocation_queue(&self) -> Result<usize> {
        let mut processed_count = 0;

        // Process requests in priority order
        let mut allocation_queue = self.allocation_queue.lock();
        let mut remaining_requests = Vec::new();

        while let Some(request) = allocation_queue.pop() {
            match self.process_allocation_request(&request).await {
                Ok(true) => {
                    processed_count += 1;
                    self.update_coordination_stats(true).await?;
                },
                Ok(false) => {
                    // Could not allocate, requeue if not expired
                    if request.requested_at.signed_duration_since(Utc::now())
                        < chrono::Duration::from_std(request.timeout)?
                    {
                        remaining_requests.push(request);
                    } else {
                        self.update_coordination_stats(false).await?;
                    }
                },
                Err(_) => {
                    self.update_coordination_stats(false).await?;
                },
            }
        }

        // Put back unprocessed requests
        allocation_queue.extend(remaining_requests);

        Ok(processed_count)
    }

    /// Process a single allocation request
    async fn process_allocation_request(&self, request: &AllocationRequest) -> Result<bool> {
        debug!("Processing allocation request: {}", request.request_id);

        // Check for conflicts
        if let Some(conflict) = self
            .conflict_detector
            .check_conflicts(&request.requirements, &request.test_id)
            .await?
        {
            debug!(
                "Conflict detected for request {}: {}",
                request.request_id, conflict
            );
            return Ok(false);
        }

        // If no conflicts, proceed with allocation
        // In a real implementation, this would coordinate with resource managers
        info!(
            "Successfully processed allocation request: {}",
            request.request_id
        );
        Ok(true)
    }

    /// Update coordination statistics
    async fn update_coordination_stats(&self, success: bool) -> Result<()> {
        let mut stats = self.coordination_stats.lock();
        stats.total_requests += 1;

        if success {
            stats.successful_allocations += 1;
        } else {
            stats.failed_allocations += 1;
        }

        Ok(())
    }

    /// Get coordination statistics
    pub async fn get_coordination_statistics(&self) -> Result<CoordinationStatistics> {
        let stats = self.coordination_stats.lock();
        // MutexGuard doesn't implement Clone, dereference to clone the inner value
        Ok((*stats).clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_parallelization::{ConflictResolutionConfig, ResourceAllocation};
    use std::collections::HashMap;

    fn make_resource_allocation(resource_id: &str) -> ResourceAllocation {
        ResourceAllocation {
            resource_type: "network_port".to_string(),
            resource_id: resource_id.to_string(),
            allocated_at: chrono::Utc::now(),
            deallocated_at: None,
            duration: std::time::Duration::from_secs(0),
            utilization: 0.5,
            efficiency: 0.8,
        }
    }

    #[test]
    fn test_allocation_statistics_default() {
        let s = AllocationStatistics::default();
        assert_eq!(s.total_allocations, 0);
        assert_eq!(s.active_allocations, 0);
        assert_eq!(s.failed_allocations, 0);
        assert_eq!(s.conflict_rate, 0.0);
        assert_eq!(s.utilization_efficiency, 0.0);
    }

    #[test]
    fn test_coordination_statistics_default() {
        let s = CoordinationStatistics::default();
        assert_eq!(s.total_requests, 0);
        assert_eq!(s.successful_allocations, 0);
        assert_eq!(s.failed_allocations, 0);
        assert_eq!(s.conflicts_detected, 0);
        assert_eq!(s.conflicts_resolved, 0);
    }

    #[test]
    fn test_conflict_type_variants() {
        let v = [
            format!("{:?}", ConflictType::PortConflict),
            format!("{:?}", ConflictType::DirectoryConflict),
            format!("{:?}", ConflictType::GpuConflict),
            format!("{:?}", ConflictType::DatabaseConflict),
            format!("{:?}", ConflictType::MemoryConflict),
            format!("{:?}", ConflictType::CpuConflict),
        ];
        assert_eq!(v[0], "PortConflict");
        assert_eq!(v[2], "GpuConflict");
    }

    #[test]
    fn test_conflict_severity_variants() {
        assert_eq!(format!("{:?}", ConflictSeverity::Low), "Low");
        assert_eq!(format!("{:?}", ConflictSeverity::Medium), "Medium");
        assert_eq!(format!("{:?}", ConflictSeverity::High), "High");
        assert_eq!(format!("{:?}", ConflictSeverity::Critical), "Critical");
    }

    #[test]
    fn test_conflict_resolution_variants() {
        let variants = [
            format!("{:?}", ConflictResolution::RetryWithAlternative),
            format!("{:?}", ConflictResolution::QueueForLater),
            format!("{:?}", ConflictResolution::Reject),
            format!("{:?}", ConflictResolution::ForceAllocate),
            format!("{:?}", ConflictResolution::ScaleUp),
        ];
        assert_eq!(variants[2], "Reject");
        assert_eq!(variants[4], "ScaleUp");
    }

    #[test]
    fn test_conflict_detection_logic_variants() {
        let v = [
            format!("{:?}", ConflictDetectionLogic::PortRangeOverlap),
            format!("{:?}", ConflictDetectionLogic::DirectoryPathConflict),
            format!("{:?}", ConflictDetectionLogic::GpuExclusiveAccess),
            format!("{:?}", ConflictDetectionLogic::DatabaseConnectionLimit),
            format!("{:?}", ConflictDetectionLogic::MemoryUsageLimit),
            format!("{:?}", ConflictDetectionLogic::CpuUsageLimit),
        ];
        assert_eq!(v[0], "PortRangeOverlap");
        assert_eq!(v[2], "GpuExclusiveAccess");
    }

    #[test]
    fn test_conflict_detection_rule_creation() {
        let rule = ConflictDetectionRule {
            name: "port-overlap".to_string(),
            resource_types: vec!["network_port".to_string()],
            detection_logic: ConflictDetectionLogic::PortRangeOverlap,
            priority: 0.9,
            enabled: true,
        };
        assert_eq!(rule.name, "port-overlap");
        assert!(rule.enabled);
        assert!((rule.priority - 0.9).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_resource_allocator_new() {
        let allocator = ResourceAllocator::new().await;
        assert!(allocator.is_ok());
    }

    #[tokio::test]
    async fn test_resource_allocator_get_active_empty() {
        let allocator = ResourceAllocator::new().await.unwrap_or_else(|_| panic!("failed"));
        let active = allocator.get_active_allocations().await.unwrap_or_default();
        assert!(active.is_empty());
    }

    #[tokio::test]
    async fn test_resource_allocator_get_history_empty() {
        let allocator = ResourceAllocator::new().await.unwrap_or_else(|_| panic!("failed"));
        let history = allocator.get_allocation_history().await.unwrap_or_default();
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn test_resource_allocator_track_allocation() {
        let allocator = ResourceAllocator::new().await.unwrap_or_else(|_| panic!("failed"));
        let alloc = make_resource_allocation("port-8080");
        let r = allocator.track_allocation(&alloc).await;
        assert!(r.is_ok());
        let stats = allocator.get_statistics().await.unwrap_or_default();
        assert_eq!(stats.total_allocations, 1);
    }

    #[tokio::test]
    async fn test_resource_allocator_mark_deallocated() {
        let allocator = ResourceAllocator::new().await.unwrap_or_else(|_| panic!("failed"));
        let alloc = make_resource_allocation("port-8081");
        allocator.track_allocation(&alloc).await.unwrap_or(());
        let r = allocator.mark_deallocated(&alloc).await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn test_resource_allocator_untrack_allocation() {
        let allocator = ResourceAllocator::new().await.unwrap_or_else(|_| panic!("failed"));
        let alloc = make_resource_allocation("port-8082");
        allocator.track_allocation(&alloc).await.unwrap_or(());
        let r = allocator.untrack_allocation("t-001").await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn test_resource_allocator_get_statistics() {
        let allocator = ResourceAllocator::new().await.unwrap_or_else(|_| panic!("failed"));
        let alloc = make_resource_allocation("port-8083");
        allocator.track_allocation(&alloc).await.unwrap_or(());
        let stats = allocator.get_statistics().await.unwrap_or_default();
        assert!(stats.total_allocations >= 1);
    }

    #[tokio::test]
    async fn test_conflict_detector_new() {
        let config = ConflictResolutionConfig::default();
        let detector = ConflictDetector::new(config).await;
        assert!(detector.is_ok());
    }

    #[tokio::test]
    async fn test_conflict_detector_history_empty() {
        let config = ConflictResolutionConfig::default();
        let detector = ConflictDetector::new(config).await.unwrap_or_else(|_| panic!("failed"));
        let history = detector.get_conflict_history().await.unwrap_or_default();
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn test_conflict_detector_add_rule() {
        let config = ConflictResolutionConfig::default();
        let detector = ConflictDetector::new(config).await.unwrap_or_else(|_| panic!("failed"));
        let rule = ConflictDetectionRule {
            name: "test-rule".to_string(),
            resource_types: vec!["gpu".to_string()],
            detection_logic: ConflictDetectionLogic::GpuExclusiveAccess,
            priority: 1.0,
            enabled: true,
        };
        let r = detector.add_detection_rule(rule).await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn test_conflict_detector_remove_nonexistent_rule() {
        let config = ConflictResolutionConfig::default();
        let detector = ConflictDetector::new(config).await.unwrap_or_else(|_| panic!("failed"));
        let removed = detector.remove_detection_rule("no-such-rule").await.unwrap_or(false);
        assert!(!removed);
    }

    #[tokio::test]
    async fn test_allocation_coordinator_new() {
        let config = ConflictResolutionConfig::default();
        let allocator =
            Arc::new(ResourceAllocator::new().await.unwrap_or_else(|_| panic!("failed")));
        let detector =
            Arc::new(ConflictDetector::new(config).await.unwrap_or_else(|_| panic!("failed")));
        let coordinator = AllocationCoordinator::new(allocator, detector).await;
        assert!(coordinator.is_ok());
    }

    #[tokio::test]
    async fn test_allocation_coordinator_get_stats_initial() {
        let config = ConflictResolutionConfig::default();
        let allocator =
            Arc::new(ResourceAllocator::new().await.unwrap_or_else(|_| panic!("failed")));
        let detector =
            Arc::new(ConflictDetector::new(config).await.unwrap_or_else(|_| panic!("failed")));
        let coordinator = AllocationCoordinator::new(allocator, detector)
            .await
            .unwrap_or_else(|_| panic!("failed"));
        let stats = coordinator.get_coordination_statistics().await.unwrap_or_default();
        assert_eq!(stats.total_requests, 0);
        assert_eq!(stats.successful_allocations, 0);
    }

    #[test]
    fn test_allocation_request_creation() {
        use crate::parallel_execution_engine::ResourceRequirement;
        let req = AllocationRequest {
            request_id: "req-001".to_string(),
            test_id: "t-001".to_string(),
            requirements: ResourceRequirement {
                resource_type: "gpu".to_string(),
                min_amount: 0.0,
                cpu_cores: 2.0,
                memory_mb: 1024,
                gpu_devices: vec![],
                network_ports: 1,
                temp_directories: 0,
                database_connections: 0,
                custom_resources: HashMap::new(),
            },
            requested_at: chrono::Utc::now(),
            priority: 0.8,
            timeout: std::time::Duration::from_secs(30),
            retry_count: 0,
        };
        assert_eq!(req.request_id, "req-001");
        assert_eq!(req.retry_count, 0);
        assert!((req.priority - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_conflict_event_creation() {
        let event = ConflictEvent {
            timestamp: chrono::Utc::now(),
            conflict_type: ConflictType::PortConflict,
            conflicting_tests: vec!["t-a".to_string(), "t-b".to_string()],
            resource_id: "port-8080".to_string(),
            severity: ConflictSeverity::Medium,
            resolution: Some(ConflictResolution::RetryWithAlternative),
            details: HashMap::new(),
        };
        assert_eq!(event.conflicting_tests.len(), 2);
        assert!(event.resolution.is_some());
    }
}
