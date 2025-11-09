//! Network port allocation and management for test parallelization.

// Re-export types for external access
pub use super::types::{PortUsageStatistics, PortUsageType};
use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
    time::Duration,
};
use tracing::{debug, info};

use crate::test_parallelization::PortPoolConfig;

/// Network port management
pub struct NetworkPortManager {
    /// Port pool configuration
    config: Arc<RwLock<PortPoolConfig>>,
    /// Available ports
    available_ports: Arc<Mutex<HashSet<u16>>>,
    /// Allocated ports
    allocated_ports: Arc<Mutex<HashMap<u16, PortAllocation>>>,
    /// Port reservation system
    reservation_system: Arc<PortReservationSystem>,
    /// Port usage statistics
    usage_stats: Arc<Mutex<PortUsageStatistics>>,
}

/// Port allocation information
#[derive(Debug, Clone)]
pub struct PortAllocation {
    /// Allocated port
    pub port: u16,
    /// Test ID that allocated the port
    pub test_id: String,
    /// Allocation timestamp
    pub allocated_at: DateTime<Utc>,
    /// Expected release time
    pub expected_release: Option<DateTime<Utc>>,
    /// Port usage type
    pub usage_type: PortUsageType,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Port reservation system
#[derive(Debug)]
pub struct PortReservationSystem {
    /// Reserved ports by test ID
    reservations: Arc<Mutex<HashMap<String, Vec<u16>>>>,
    /// Reservation expiry times
    expiry_times: Arc<Mutex<HashMap<u16, DateTime<Utc>>>>,
    /// Reservation queue
    reservation_queue: Arc<Mutex<VecDeque<PortReservationRequest>>>,
}

impl Default for PortReservationSystem {
    fn default() -> Self {
        Self {
            reservations: Arc::new(Mutex::new(HashMap::new())),
            expiry_times: Arc::new(Mutex::new(HashMap::new())),
            reservation_queue: Arc::new(Mutex::new(VecDeque::new())),
        }
    }
}

/// Port reservation request
#[derive(Debug, Clone)]
pub struct PortReservationRequest {
    /// Test ID requesting reservation
    pub test_id: String,
    /// Number of ports requested
    pub port_count: usize,
    /// Preferred port range
    pub preferred_range: Option<(u16, u16)>,
    /// Usage type
    pub usage_type: PortUsageType,
    /// Request timestamp
    pub requested_at: DateTime<Utc>,
    /// Request priority
    pub priority: f32,
    /// Timeout for reservation
    pub timeout: Duration,
}

impl NetworkPortManager {
    /// Create new network port manager
    pub async fn new(config: PortPoolConfig) -> Result<Self> {
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            available_ports: Arc::new(Mutex::new(HashSet::new())),
            allocated_ports: Arc::new(Mutex::new(HashMap::new())),
            reservation_system: Arc::new(PortReservationSystem::default()),
            usage_stats: Arc::new(Mutex::new(PortUsageStatistics::default())),
        })
    }

    /// Allocate ports for a test
    pub async fn allocate_ports(&self, count: usize, test_id: &str) -> Result<Vec<u16>> {
        info!("Allocating {} ports for test: {}", count, test_id);

        // For now, return a simple port allocation
        // In a real implementation, this would:
        // 1. Check available ports
        // 2. Reserve the requested number of ports
        // 3. Track the allocation
        // 4. Update statistics

        let ports = vec![8080]; // Placeholder

        // Update allocation tracking
        for &port in &ports {
            let allocation = PortAllocation {
                port,
                test_id: test_id.to_string(),
                allocated_at: Utc::now(),
                expected_release: None,
                usage_type: PortUsageType::HttpServer,
                metadata: HashMap::new(),
            };

            let mut allocated_ports = self.allocated_ports.lock();
            allocated_ports.insert(port, allocation);
        }

        // Update statistics
        let mut stats = self.usage_stats.lock();
        stats.total_allocated += ports.len() as u64;
        stats.currently_allocated += ports.len();

        info!("Allocated ports {:?} for test: {}", ports, test_id);
        Ok(ports)
    }

    /// Deallocate a specific port
    pub async fn deallocate_port(&self, port: u16) -> Result<()> {
        debug!("Deallocating port: {}", port);

        let mut allocated_ports = self.allocated_ports.lock();
        if let Some(_allocation) = allocated_ports.remove(&port) {
            // Update statistics
            let mut stats = self.usage_stats.lock();
            stats.currently_allocated = stats.currently_allocated.saturating_sub(1);

            info!("Successfully deallocated port: {}", port);
        } else {
            debug!("Port {} was not allocated or already deallocated", port);
        }

        Ok(())
    }

    /// Deallocate all ports for a test
    pub async fn deallocate_ports_for_test(&self, test_id: &str) -> Result<()> {
        debug!("Deallocating ports for test: {}", test_id);

        let mut allocated_ports = self.allocated_ports.lock();
        let ports_to_remove: Vec<u16> = allocated_ports
            .iter()
            .filter(|(_, allocation)| allocation.test_id == test_id)
            .map(|(&port, _)| port)
            .collect();

        for port in &ports_to_remove {
            allocated_ports.remove(port);
        }

        // Update statistics
        let mut stats = self.usage_stats.lock();
        stats.currently_allocated = stats.currently_allocated.saturating_sub(ports_to_remove.len());

        info!(
            "Released {} ports for test: {}",
            ports_to_remove.len(),
            test_id
        );
        Ok(())
    }

    /// Check if requested number of ports are available
    pub async fn check_availability(&self, count: usize) -> Result<bool> {
        // In a real implementation, this would check against available port ranges
        // and current allocations
        let allocated_ports = self.allocated_ports.lock();
        let currently_allocated = allocated_ports.len();

        // Simple check assuming a maximum of 1000 ports available
        Ok(currently_allocated + count <= 1000)
    }

    /// Get port usage statistics
    pub async fn get_statistics(&self) -> Result<PortUsageStatistics> {
        let stats = self.usage_stats.lock();
        // MutexGuard doesn't implement Clone, dereference to clone the inner value
        Ok((*stats).clone())
    }

    /// Reserve ports for future allocation
    pub async fn reserve_ports(&self, request: PortReservationRequest) -> Result<Vec<u16>> {
        info!(
            "Processing port reservation request for test: {}",
            request.test_id
        );

        // Add to reservation queue
        let mut queue = self.reservation_system.reservation_queue.lock();
        queue.push_back(request.clone());

        // For now, return a simple reservation
        let reserved_ports = vec![8080]; // Placeholder

        // Track reservation
        let mut reservations = self.reservation_system.reservations.lock();
        reservations.insert(request.test_id.clone(), reserved_ports.clone());

        info!(
            "Reserved ports {:?} for test: {}",
            reserved_ports, request.test_id
        );
        Ok(reserved_ports)
    }

    /// Release port reservations for a test
    pub async fn release_reservations(&self, test_id: &str) -> Result<()> {
        debug!("Releasing port reservations for test: {}", test_id);

        let mut reservations = self.reservation_system.reservations.lock();
        if let Some(reserved_ports) = reservations.remove(test_id) {
            info!(
                "Released {} reserved ports for test: {}",
                reserved_ports.len(),
                test_id
            );
        }

        Ok(())
    }

    /// Get current port allocations
    pub async fn get_allocations(&self) -> Result<Vec<PortAllocation>> {
        let allocated_ports = self.allocated_ports.lock();
        Ok(allocated_ports.values().cloned().collect())
    }

    /// Get allocations for a specific test
    pub async fn get_allocations_for_test(&self, test_id: &str) -> Result<Vec<PortAllocation>> {
        let allocated_ports = self.allocated_ports.lock();
        Ok(allocated_ports
            .values()
            .filter(|allocation| allocation.test_id == test_id)
            .cloned()
            .collect())
    }

    /// Force deallocate expired allocations
    pub async fn cleanup_expired_allocations(&self) -> Result<usize> {
        let mut allocated_ports = self.allocated_ports.lock();
        let now = Utc::now();
        let mut expired_count = 0;

        allocated_ports.retain(|_port, allocation| {
            if let Some(expected_release) = allocation.expected_release {
                if now > expected_release {
                    expired_count += 1;
                    false // Remove expired allocation
                } else {
                    true // Keep allocation
                }
            } else {
                true // Keep allocation without expiry time
            }
        });

        if expired_count > 0 {
            // Update statistics
            let mut stats = self.usage_stats.lock();
            stats.currently_allocated = stats.currently_allocated.saturating_sub(expired_count);
            info!("Cleaned up {} expired port allocations", expired_count);
        }

        Ok(expired_count)
    }

    /// Update port manager configuration
    pub async fn update_config(&self, config: PortPoolConfig) -> Result<()> {
        let mut current_config = self.config.write();
        *current_config = config;
        info!("Updated port manager configuration");
        Ok(())
    }

    /// Get available port count
    pub async fn get_available_count(&self) -> Result<usize> {
        let allocated_count = self.allocated_ports.lock().len();
        let config = self.config.read();

        // Calculate available ports based on configuration
        // This is a simplified calculation
        let total_ports = (config.end_port - config.start_port + 1) as usize;
        Ok(total_ports.saturating_sub(allocated_count))
    }

    /// Force release all allocations (emergency cleanup)
    pub async fn force_release_all(&self) -> Result<usize> {
        let mut allocated_ports = self.allocated_ports.lock();
        let count = allocated_ports.len();
        allocated_ports.clear();

        // Reset statistics
        let mut stats = self.usage_stats.lock();
        stats.currently_allocated = 0;

        info!("Force released all {} port allocations", count);
        Ok(count)
    }
}
