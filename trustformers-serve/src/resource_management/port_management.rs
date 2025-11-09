//! Network port management for test resource allocation.
//!
//! This module provides comprehensive network port management capabilities including
//! port allocation, reservation, conflict detection, and usage statistics for
//! parallel test execution in distributed systems.

use anyhow::Result;
use chrono::Utc;
use parking_lot::{Mutex, RwLock};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};
use tracing::{debug, info, warn};

use super::types::{PortPoolConfig, PortReservationRequest, PortUsageStatistics, PortUsageType};

// Re-export types needed by other modules
pub use super::types::PortAllocation;

/// Network port management system
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

/// Port reservation system for advanced port management
pub struct PortReservationSystem {
    /// Active reservations
    reservations: Arc<Mutex<HashMap<u16, PortReservationRequest>>>,
    /// Reservation history
    reservation_history: Arc<Mutex<Vec<PortReservationRequest>>>,
    /// Conflict detection enabled
    conflict_detection: bool,
}

impl NetworkPortManager {
    /// Create new network port manager
    pub async fn new(config: PortPoolConfig) -> Result<Self> {
        let mut available_ports = HashSet::new();

        // Initialize available ports from configuration
        for port in config.port_range.0..=config.port_range.1 {
            // Skip well-known ports and reserved ranges
            let is_reserved =
                config.reserved_ranges.iter().any(|(start, end)| port >= *start && port <= *end);
            if !Self::is_well_known_port(port) && !is_reserved {
                available_ports.insert(port);
            }
        }

        info!(
            "Initialized port manager with {} available ports in range {}-{}",
            available_ports.len(),
            config.port_range.0,
            config.port_range.1
        );

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            available_ports: Arc::new(Mutex::new(available_ports)),
            allocated_ports: Arc::new(Mutex::new(HashMap::new())),
            reservation_system: Arc::new(PortReservationSystem::new()),
            usage_stats: Arc::new(Mutex::new(PortUsageStatistics::default())),
        })
    }

    /// Allocate ports for a test
    pub async fn allocate_ports(&self, count: usize, test_id: &str) -> Result<Vec<u16>> {
        if count == 0 {
            return Ok(vec![]);
        }

        let mut available_ports = self.available_ports.lock();
        let mut allocated_ports = self.allocated_ports.lock();
        let mut usage_stats = self.usage_stats.lock();

        if available_ports.len() < count {
            return Err(anyhow::anyhow!(
                "Insufficient available ports: requested {}, available {}",
                count,
                available_ports.len()
            ));
        }

        let mut allocated = Vec::new();
        let now = Utc::now();

        // Allocate ports
        for _ in 0..count {
            if let Some(&port) = available_ports.iter().next() {
                available_ports.remove(&port);

                let allocation = PortAllocation {
                    port,
                    test_id: test_id.to_string(),
                    allocated_at: now,
                    expected_release: None,
                    usage_type: PortUsageType::Custom("test".to_string()),
                    metadata: HashMap::new(),
                };

                allocated_ports.insert(port, allocation);
                allocated.push(port);
            } else {
                // Rollback partial allocation
                for &port in &allocated {
                    if let Some(allocation) = allocated_ports.remove(&port) {
                        available_ports.insert(allocation.port);
                    }
                }
                return Err(anyhow::anyhow!("Failed to allocate ports"));
            }
        }

        // Update statistics
        usage_stats.total_allocated += count as u64;
        usage_stats.currently_allocated = allocated_ports.len();
        usage_stats.peak_usage = usage_stats.peak_usage.max(allocated_ports.len());

        info!(
            "Allocated {} ports for test {}: {:?}",
            allocated.len(),
            test_id,
            allocated
        );

        Ok(allocated)
    }

    /// Deallocate a specific port
    pub async fn deallocate_port(&self, port: u16) -> Result<()> {
        let mut available_ports = self.available_ports.lock();
        let mut allocated_ports = self.allocated_ports.lock();
        let mut usage_stats = self.usage_stats.lock();

        if let Some(allocation) = allocated_ports.remove(&port) {
            available_ports.insert(port);
            usage_stats.currently_allocated = allocated_ports.len();

            // Update average duration statistics
            let duration = allocation.allocated_at.signed_duration_since(Utc::now()).abs();
            let duration_std = Duration::from_secs(duration.num_seconds().max(0) as u64);

            if usage_stats.total_allocated > 0 {
                let total_duration = usage_stats.average_allocation_duration.as_secs() as f64
                    * (usage_stats.total_allocated - 1) as f64;
                let new_average = (total_duration + duration_std.as_secs() as f64)
                    / usage_stats.total_allocated as f64;
                usage_stats.average_allocation_duration = Duration::from_secs(new_average as u64);
            }

            info!("Deallocated port {} for test {}", port, allocation.test_id);
            Ok(())
        } else {
            warn!(
                "Attempted to deallocate port {} that was not allocated",
                port
            );
            Err(anyhow::anyhow!("Port {} was not allocated", port))
        }
    }

    /// Deallocate all ports for a specific test
    pub async fn deallocate_ports_for_test(&self, test_id: &str) -> Result<()> {
        debug!("Deallocating ports for test: {}", test_id);

        let mut available_ports = self.available_ports.lock();
        let mut allocated_ports = self.allocated_ports.lock();
        let mut usage_stats = self.usage_stats.lock();

        let mut deallocated_ports = Vec::new();

        // Find and collect ports to deallocate
        allocated_ports.retain(|&port, allocation| {
            if allocation.test_id == test_id {
                available_ports.insert(port);
                deallocated_ports.push(port);
                false // Remove from allocated_ports
            } else {
                true // Keep in allocated_ports
            }
        });

        usage_stats.currently_allocated = allocated_ports.len();

        if !deallocated_ports.is_empty() {
            info!(
                "Released {} network ports for test {}: {:?}",
                deallocated_ports.len(),
                test_id,
                deallocated_ports
            );
        }

        Ok(())
    }

    /// Check if requested number of ports are available
    pub async fn check_availability(&self, count: usize) -> Result<bool> {
        let available_ports = self.available_ports.lock();
        Ok(available_ports.len() >= count)
    }

    /// Get current port usage statistics
    pub async fn get_statistics(&self) -> Result<PortUsageStatistics> {
        let stats = self.usage_stats.lock();
        Ok(stats.clone())
    }

    /// Reserve ports for future allocation
    pub async fn reserve_ports(
        &self,
        count: usize,
        test_id: &str,
        duration: Duration,
        usage_type: PortUsageType,
    ) -> Result<Vec<u16>> {
        self.reservation_system
            .reserve_ports(count, test_id, duration, usage_type, &self.available_ports)
            .await
    }

    /// Cancel port reservations for a test
    pub async fn cancel_reservations(&self, test_id: &str) -> Result<()> {
        self.reservation_system.cancel_reservations(test_id).await
    }

    /// Get detailed port allocation information
    pub async fn get_port_allocations(&self) -> HashMap<u16, PortAllocation> {
        let allocated_ports = self.allocated_ports.lock();
        allocated_ports.clone()
    }

    /// Get available port count
    pub async fn get_available_port_count(&self) -> usize {
        let available_ports = self.available_ports.lock();
        available_ports.len()
    }

    /// Get allocated port count
    pub async fn get_allocated_port_count(&self) -> usize {
        let allocated_ports = self.allocated_ports.lock();
        allocated_ports.len()
    }

    /// Check if a specific port is available
    pub async fn is_port_available(&self, port: u16) -> bool {
        let available_ports = self.available_ports.lock();
        available_ports.contains(&port)
    }

    /// Check if a specific port is allocated
    pub async fn is_port_allocated(&self, port: u16) -> bool {
        let allocated_ports = self.allocated_ports.lock();
        allocated_ports.contains_key(&port)
    }

    /// Get allocation details for a specific port
    pub async fn get_port_allocation(&self, port: u16) -> Option<PortAllocation> {
        let allocated_ports = self.allocated_ports.lock();
        allocated_ports.get(&port).cloned()
    }

    /// Update port manager configuration
    pub async fn update_config(&self, new_config: PortPoolConfig) -> Result<()> {
        let mut config = self.config.write();
        let mut available_ports = self.available_ports.lock();

        // Recalculate available ports based on new configuration
        available_ports.clear();
        for port in new_config.port_range.0..=new_config.port_range.1 {
            let is_reserved = new_config
                .reserved_ranges
                .iter()
                .any(|(start, end)| port >= *start && port <= *end);
            if !Self::is_well_known_port(port) && !is_reserved {
                available_ports.insert(port);
            }
        }

        // Remove any ports that are no longer in the valid range
        let allocated_ports = self.allocated_ports.lock();
        for &port in allocated_ports.keys() {
            available_ports.remove(&port);
        }

        *config = new_config;

        info!(
            "Updated port manager configuration, {} available ports",
            available_ports.len()
        );

        Ok(())
    }

    /// Check if port is well-known (system reserved)
    fn is_well_known_port(port: u16) -> bool {
        port < 1024 || matches!(port, 1433 | 1521 | 1883 | 3306 | 5432 | 6379 | 27017)
    }

    /// Get port utilization percentage
    pub async fn get_utilization(&self) -> f32 {
        let available_count = self.get_available_port_count().await;
        let allocated_count = self.get_allocated_port_count().await;
        let total_count = available_count + allocated_count;

        if total_count == 0 {
            0.0
        } else {
            allocated_count as f32 / total_count as f32
        }
    }

    /// Clean up expired allocations
    pub async fn cleanup_expired_allocations(&self) -> Result<usize> {
        let mut available_ports = self.available_ports.lock();
        let mut allocated_ports = self.allocated_ports.lock();
        let mut cleaned_count = 0;

        let config = self.config.read();
        let max_allocation_duration = Duration::from_secs(config.allocation_timeout_secs);
        let now = Utc::now();

        // Collect expired ports
        let mut expired_ports = Vec::new();
        for (&port, allocation) in allocated_ports.iter() {
            let duration_since_allocation = now.signed_duration_since(allocation.allocated_at);
            if duration_since_allocation.to_std().unwrap_or(Duration::ZERO)
                > max_allocation_duration
            {
                expired_ports.push(port);
            }
        }

        // Clean up expired allocations
        for port in expired_ports {
            if let Some(allocation) = allocated_ports.remove(&port) {
                available_ports.insert(port);
                cleaned_count += 1;
                warn!(
                    "Cleaned up expired port allocation: port {} from test {} (allocated at {})",
                    port, allocation.test_id, allocation.allocated_at
                );
            }
        }

        if cleaned_count > 0 {
            info!("Cleaned up {} expired port allocations", cleaned_count);
        }

        Ok(cleaned_count)
    }

    /// Generate port allocation report
    pub async fn generate_allocation_report(&self) -> String {
        let stats = self.get_statistics().await.unwrap_or_default();
        let available_count = self.get_available_port_count().await;
        let allocated_count = self.get_allocated_port_count().await;
        let utilization = self.get_utilization().await;

        format!(
            "Port Allocation Report:\n\
             - Available ports: {}\n\
             - Allocated ports: {}\n\
             - Total allocations: {}\n\
             - Peak usage: {}\n\
             - Current utilization: {:.1}%\n\
             - Average allocation duration: {}s",
            available_count,
            allocated_count,
            stats.total_allocated,
            stats.peak_usage,
            utilization * 100.0,
            stats.average_allocation_duration.as_secs()
        )
    }
}

impl PortReservationSystem {
    /// Create new port reservation system
    pub fn new() -> Self {
        Self {
            reservations: Arc::new(Mutex::new(HashMap::new())),
            reservation_history: Arc::new(Mutex::new(Vec::new())),
            conflict_detection: true,
        }
    }

    /// Reserve ports for future allocation
    pub async fn reserve_ports(
        &self,
        count: usize,
        test_id: &str,
        duration: Duration,
        usage_type: PortUsageType,
        available_ports: &Arc<Mutex<HashSet<u16>>>,
    ) -> Result<Vec<u16>> {
        if count == 0 {
            return Ok(vec![]);
        }

        let available = available_ports.lock();
        let mut reservations = self.reservations.lock();

        if available.len() < count {
            return Err(anyhow::anyhow!(
                "Insufficient available ports for reservation: requested {}, available {}",
                count,
                available.len()
            ));
        }

        let mut reserved_ports = Vec::new();
        let now = Utc::now();

        // Reserve ports
        for _ in 0..count {
            if let Some(&port) = available.iter().find(|&p| !reservations.contains_key(p)) {
                let reservation = PortReservationRequest {
                    test_id: test_id.to_string(),
                    port_count: count,
                    preferred_range: None,
                    usage_type: usage_type.clone(),
                    requested_at: now,
                    priority: 0.5, // Default priority
                    timeout: duration,
                    port: Some(port),
                    duration: Some(duration),
                };

                reservations.insert(port, reservation.clone());
                reserved_ports.push(port);

                // Add to history
                let mut history = self.reservation_history.lock();
                history.push(reservation);
                if history.len() > 1000 {
                    history.remove(0);
                }
            } else {
                // Rollback partial reservation
                for &port in &reserved_ports {
                    reservations.remove(&port);
                }
                return Err(anyhow::anyhow!("Failed to reserve sufficient ports"));
            }
        }

        info!(
            "Reserved {} ports for test {}: {:?}",
            reserved_ports.len(),
            test_id,
            reserved_ports
        );

        Ok(reserved_ports)
    }

    /// Cancel reservations for a test
    pub async fn cancel_reservations(&self, test_id: &str) -> Result<()> {
        let mut reservations = self.reservations.lock();
        let mut cancelled_ports = Vec::new();

        reservations.retain(|&port, reservation| {
            if reservation.test_id == test_id {
                cancelled_ports.push(port);
                false
            } else {
                true
            }
        });

        if !cancelled_ports.is_empty() {
            info!(
                "Cancelled {} port reservations for test {}: {:?}",
                cancelled_ports.len(),
                test_id,
                cancelled_ports
            );
        }

        Ok(())
    }

    /// Get active reservations
    pub fn get_active_reservations(&self) -> HashMap<u16, PortReservationRequest> {
        let reservations = self.reservations.lock();
        reservations.clone()
    }

    /// Check if port is reserved
    pub fn is_port_reserved(&self, port: u16) -> bool {
        let reservations = self.reservations.lock();
        reservations.contains_key(&port)
    }

    /// Get reservation for a specific port
    pub fn get_port_reservation(&self, port: u16) -> Option<PortReservationRequest> {
        let reservations = self.reservations.lock();
        reservations.get(&port).cloned()
    }

    /// Clean up expired reservations
    pub async fn cleanup_expired_reservations(&self) -> Result<usize> {
        let mut reservations = self.reservations.lock();
        let now = Utc::now();
        let mut expired_ports = Vec::new();

        for (&port, reservation) in reservations.iter() {
            let duration_since_request = now.signed_duration_since(reservation.requested_at);
            if let Some(duration) = reservation.duration {
                if duration_since_request.to_std().unwrap_or(Duration::ZERO) > duration {
                    expired_ports.push(port);
                }
            }
        }

        let cleaned_count = expired_ports.len();
        for port in expired_ports {
            if let Some(reservation) = reservations.remove(&port) {
                warn!(
                    "Cleaned up expired port reservation: port {} from test {}",
                    port, reservation.test_id
                );
            }
        }

        if cleaned_count > 0 {
            info!("Cleaned up {} expired port reservations", cleaned_count);
        }

        Ok(cleaned_count)
    }

    /// Get reservation statistics
    pub fn get_reservation_statistics(&self) -> (usize, usize) {
        let reservations = self.reservations.lock();
        let history = self.reservation_history.lock();
        (reservations.len(), history.len())
    }
}

impl Default for PortReservationSystem {
    fn default() -> Self {
        Self::new()
    }
}
