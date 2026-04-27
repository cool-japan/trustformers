//! Resource allocation strategies and tracking for test parallelization.
//!
//! This module provides intelligent resource allocation algorithms, load balancing,
//! and allocation tracking for optimal resource utilization.

use anyhow::Result;
use chrono::Utc;
use parking_lot::Mutex;
use std::{collections::HashMap, sync::Arc};
use tracing::{debug, info};

use super::types::{
    AllocationEvent, DistributionEvent, ExecutionPerformanceMetrics, ExecutionState,
};
use crate::test_parallelization::ResourceAllocation;

// Re-export types needed by other modules
pub use super::types::LoadMetrics;

/// Resource allocator for distribution and tracking
pub struct ResourceAllocator {
    /// Active allocations
    active_allocations: Arc<Mutex<HashMap<String, ResourceAllocation>>>,
    /// Allocation history
    allocation_history: Arc<Mutex<Vec<AllocationEvent>>>,
    /// Worker pool
    worker_pool: Arc<WorkerPool>,
}

/// Worker pool for resource allocation
pub struct WorkerPool {
    /// Workers
    workers: Arc<Mutex<HashMap<String, Worker>>>,
    /// Load metrics
    load_metrics: Arc<Mutex<LoadMetrics>>,
    /// Distribution events
    distribution_events: Arc<Mutex<Vec<DistributionEvent>>>,
}

/// Individual worker
struct Worker {
    /// Worker ID
    worker_id: String,
    /// Current state
    state: ExecutionState,
    /// Performance metrics
    performance_metrics: ExecutionPerformanceMetrics,
}

impl ResourceAllocator {
    /// Create new resource allocator
    pub async fn new() -> Result<Self> {
        Ok(Self {
            active_allocations: Arc::new(Mutex::new(HashMap::new())),
            allocation_history: Arc::new(Mutex::new(Vec::new())),
            worker_pool: Arc::new(WorkerPool::new()),
        })
    }

    /// Track allocation
    pub async fn track_allocation(&self, allocation: &ResourceAllocation) -> Result<()> {
        let mut active_allocations = self.active_allocations.lock();
        active_allocations.insert(allocation.resource_id.clone(), allocation.clone());

        // Record allocation event
        let event = AllocationEvent {
            timestamp: chrono::Utc::now(),
            event_type: "Allocated".to_string(),
            test_id: "unknown".to_string(), // Would be extracted from allocation
            resource_type: allocation.resource_type.clone(),
            resource_id: allocation.resource_id.clone(),
            details: HashMap::new(),
        };

        let mut allocation_history = self.allocation_history.lock();
        allocation_history.push(event);

        // Limit history size
        if allocation_history.len() > 10000 {
            allocation_history.remove(0);
        }

        debug!("Tracked allocation: {}", allocation.resource_id);
        Ok(())
    }

    /// Mark allocation as deallocated
    pub async fn mark_deallocated(&self, allocation: &ResourceAllocation) -> Result<()> {
        let mut active_allocations = self.active_allocations.lock();
        active_allocations.remove(&allocation.resource_id);

        // Record deallocation event
        let event = AllocationEvent {
            timestamp: chrono::Utc::now(),
            event_type: "Deallocated".to_string(),
            test_id: "unknown".to_string(), // Would be extracted from allocation
            resource_type: allocation.resource_type.clone(),
            resource_id: allocation.resource_id.clone(),
            details: HashMap::new(),
        };

        let mut allocation_history = self.allocation_history.lock();
        allocation_history.push(event);

        debug!(
            "Marked allocation as deallocated: {}",
            allocation.resource_id
        );
        Ok(())
    }

    /// Untrack allocation
    pub async fn untrack_allocation(&self, _test_id: &str) -> Result<()> {
        // Placeholder implementation
        Ok(())
    }

    /// Get active allocations
    pub async fn get_active_allocations(&self) -> HashMap<String, ResourceAllocation> {
        let active_allocations = self.active_allocations.lock();
        active_allocations.clone()
    }

    /// Get allocation history
    pub async fn get_allocation_history(&self) -> Vec<AllocationEvent> {
        let allocation_history = self.allocation_history.lock();
        allocation_history.clone()
    }

    /// Get load metrics
    pub async fn get_load_metrics(&self) -> LoadMetrics {
        self.worker_pool.get_load_metrics().await
    }
}

impl WorkerPool {
    /// Create new worker pool
    pub fn new() -> Self {
        Self {
            workers: Arc::new(Mutex::new(HashMap::new())),
            load_metrics: Arc::new(Mutex::new(LoadMetrics {
                cpu_usage: 0.0,
                memory_usage: 0.0,
                active_tasks: 0,
                queue_length: 0,
                timestamp: Utc::now(),
            })),
            distribution_events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Add worker
    pub async fn add_worker(&self, worker_id: &str) -> Result<()> {
        let mut workers = self.workers.lock();

        let worker_start_time = chrono::Utc::now();
        let worker = Worker {
            worker_id: worker_id.to_string(),
            state: ExecutionState {
                status: super::types::ExecutionStatus::Running,
                start_time: worker_start_time,
                end_time: None,
                progress: 0.0,
                metadata: HashMap::new(),
                worker_id: worker_id.to_string(),
                state: "Idle".to_string(),
                current_task: None,
                state_since: worker_start_time,
                last_heartbeat: worker_start_time,
            },
            performance_metrics: ExecutionPerformanceMetrics::default(),
        };

        workers.insert(worker_id.to_string(), worker);

        info!("Added worker to pool: {}", worker_id);
        Ok(())
    }

    /// Remove worker
    pub async fn remove_worker(&self, worker_id: &str) -> Result<()> {
        let mut workers = self.workers.lock();

        if workers.remove(worker_id).is_some() {
            info!("Removed worker from pool: {}", worker_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Worker {} not found", worker_id))
        }
    }

    /// Get load metrics
    pub async fn get_load_metrics(&self) -> LoadMetrics {
        let load_metrics = self.load_metrics.lock();
        load_metrics.clone()
    }

    /// Update load metrics
    pub async fn update_load_metrics(&self, metrics: LoadMetrics) {
        let mut load_metrics = self.load_metrics.lock();
        *load_metrics = metrics;
    }

    /// Get worker count
    pub async fn get_worker_count(&self) -> usize {
        let workers = self.workers.lock();
        workers.len()
    }

    /// Get active worker count
    pub async fn get_active_worker_count(&self) -> usize {
        let workers = self.workers.lock();
        workers.values().filter(|worker| worker.state.state == "Busy").count()
    }
}

impl Default for WorkerPool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_parallelization::ResourceAllocation;
    use std::time::Duration;

    fn make_allocation(id: &str) -> ResourceAllocation {
        ResourceAllocation {
            resource_type: "network_port".to_string(),
            resource_id: id.to_string(),
            allocated_at: chrono::Utc::now(),
            deallocated_at: None,
            duration: Duration::from_secs(0),
            utilization: 0.5,
            efficiency: 0.8,
        }
    }

    #[tokio::test]
    async fn test_resource_allocator_creation() {
        let allocator = ResourceAllocator::new().await;
        assert!(allocator.is_ok());
    }

    #[tokio::test]
    async fn test_resource_allocator_track_allocation() {
        let allocator =
            ResourceAllocator::new().await.unwrap_or_else(|_| panic!("creation failed"));
        let alloc = make_allocation("port-8080");
        let result = allocator.track_allocation(&alloc).await;
        assert!(result.is_ok());
        let active = allocator.get_active_allocations().await;
        assert_eq!(active.len(), 1);
    }

    #[tokio::test]
    async fn test_resource_allocator_mark_deallocated() {
        let allocator =
            ResourceAllocator::new().await.unwrap_or_else(|_| panic!("creation failed"));
        let alloc = make_allocation("port-9090");
        allocator.track_allocation(&alloc).await.unwrap_or(());
        let result = allocator.mark_deallocated(&alloc).await;
        assert!(result.is_ok());
        let active = allocator.get_active_allocations().await;
        assert!(active.is_empty());
    }

    #[tokio::test]
    async fn test_resource_allocator_history_grows() {
        let allocator =
            ResourceAllocator::new().await.unwrap_or_else(|_| panic!("creation failed"));
        allocator.track_allocation(&make_allocation("p1")).await.unwrap_or(());
        allocator.track_allocation(&make_allocation("p2")).await.unwrap_or(());
        let history = allocator.get_allocation_history().await;
        assert_eq!(history.len(), 2);
    }

    #[tokio::test]
    async fn test_worker_pool_new() {
        let pool = WorkerPool::new();
        assert_eq!(pool.get_worker_count().await, 0);
    }

    #[tokio::test]
    async fn test_worker_pool_add_workers() {
        let pool = WorkerPool::new();
        pool.add_worker("w1").await.unwrap_or(());
        pool.add_worker("w2").await.unwrap_or(());
        assert_eq!(pool.get_worker_count().await, 2);
    }

    #[tokio::test]
    async fn test_worker_pool_remove_worker() {
        let pool = WorkerPool::new();
        pool.add_worker("w-1").await.unwrap_or(());
        let result = pool.remove_worker("w-1").await;
        assert!(result.is_ok());
        assert_eq!(pool.get_worker_count().await, 0);
    }

    #[tokio::test]
    async fn test_worker_pool_remove_nonexistent() {
        let pool = WorkerPool::new();
        let result = pool.remove_worker("nonexistent").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_worker_pool_load_metrics() {
        let pool = WorkerPool::new();
        let metrics = pool.get_load_metrics().await;
        assert_eq!(metrics.cpu_usage, 0.0);
        assert_eq!(metrics.active_tasks, 0);
    }

    #[tokio::test]
    async fn test_resource_allocator_multiple_allocs() {
        let allocator =
            ResourceAllocator::new().await.unwrap_or_else(|_| panic!("creation failed"));
        for i in 0..5_usize {
            allocator
                .track_allocation(&make_allocation(&format!("r{}", i)))
                .await
                .unwrap_or(());
        }
        assert_eq!(allocator.get_active_allocations().await.len(), 5);
    }
}
