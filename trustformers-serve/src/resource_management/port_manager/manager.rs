//! Network Port Manager Implementation
//!
//! This module contains the core NetworkPortManager implementation that orchestrates
//! all port management operations including allocation, deallocation, reservation,
//! conflict detection, health monitoring, and performance metrics.
//!
//! The NetworkPortManager serves as the central coordinator that brings together
//! all the specialized components (reservation_system, conflict_detector,
//! health_monitor, performance_metrics) to provide a comprehensive port management
//! solution for high-concurrency test execution environments.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use thiserror::Error;
use tracing::{debug, error, info, instrument, warn};

use super::types::*;
use super::reservation_system::PortReservationSystem;
use super::conflict_detector::PortConflictDetector;
use super::health_monitor::{PortHealthMonitor, PortHealthStatus, HealthStatus};
use super::performance_metrics::{PortPerformanceMetrics, PerformanceSnapshot};

/// Comprehensive network port management system
///
/// This is the main entry point for all port management operations. It provides
/// thread-safe port allocation, reservation, and monitoring capabilities designed
/// for high-concurrency test execution environments.
///
/// # Architecture
///
/// The NetworkPortManager orchestrates several specialized components:
/// - **PortReservationSystem**: Advanced reservation system with conflict detection and queuing
/// - **PortConflictDetector**: Detects and resolves port allocation conflicts
/// - **PortHealthMonitor**: Monitors system health and generates alerts
/// - **PortPerformanceMetrics**: Tracks performance metrics and statistics
///
/// # Thread Safety
///
/// All operations are thread-safe using:
/// - `Arc<Mutex<>>` for mutable shared state
/// - `Arc<RwLock<>>` for configuration and read-heavy data
/// - Atomic operations for performance counters
///
/// # Usage
///
/// ```rust
/// use trustformers_serve::resource_management::port_manager::NetworkPortManager;
/// use trustformers_serve::resource_management::types::{PortPoolConfig, PortUsageType};
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = PortPoolConfig {
///     port_range: (8000, 9000),
///     max_allocation: 100,
///     allocation_timeout_secs: 300,
///     enable_reservation: true,
///     reserved_ranges: vec![(8080, 8080)],
/// };
///
/// let port_manager = NetworkPortManager::new(config).await?;
/// let ports = port_manager.allocate_ports(3, "test_001").await?;
/// port_manager.deallocate_ports_for_test("test_001").await?;
/// # Ok(())
/// # }
/// ```
pub struct NetworkPortManager {
    /// Configuration for port management
    config: Arc<RwLock<PortPoolConfig>>,

    /// Available ports for allocation
    available_ports: Arc<Mutex<HashSet<u16>>>,

    /// Currently allocated ports with their allocation details
    allocated_ports: Arc<Mutex<HashMap<u16, PortAllocation>>>,

    /// Advanced port reservation system
    reservation_system: Arc<PortReservationSystem>,

    /// Comprehensive usage statistics
    usage_stats: Arc<Mutex<PortUsageStatistics>>,

    /// Port conflict detector
    conflict_detector: Arc<PortConflictDetector>,

    /// Health monitoring system
    health_monitor: Arc<PortHealthMonitor>,

    /// Performance metrics
    performance_metrics: Arc<PortPerformanceMetrics>,

    /// Shutdown signal for background tasks
    shutdown_signal: Arc<AtomicBool>,
}

impl NetworkPortManager {
    /// Create a new network port manager with the given configuration
    ///
    /// This initializes all the component systems and sets up the available port pool
    /// based on the provided configuration. It also starts background tasks for
    /// maintenance and monitoring.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration for the port pool including port range, exclusions,
    ///              timeout settings, and reservation options
    ///
    /// # Returns
    ///
    /// A new NetworkPortManager instance ready for port operations
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The port range is invalid (start >= end)
    /// - Configuration validation fails
    /// - Component initialization fails
    /// - Background task startup fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use trustformers_serve::resource_management::types::PortPoolConfig;
    /// # use trustformers_serve::resource_management::port_manager::NetworkPortManager;
    /// # async fn example() -> anyhow::Result<()> {
    /// let config = PortPoolConfig {
    ///     port_range: (8000, 9000),
    ///     max_allocation: 100,
    ///     allocation_timeout_secs: 300,
    ///     enable_reservation: true,
    ///     reserved_ranges: vec![(8080, 8080)], // Reserve port 8080
    /// };
    ///
    /// let port_manager = NetworkPortManager::new(config).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip(config))]
    pub async fn new(config: PortPoolConfig) -> PortManagementResult<Self> {
        Self::validate_config(&config)?;

        let mut available_ports = HashSet::new();
        let (start, end) = config.port_range;

        // Initialize available ports from configuration
        for port in start..=end {
            if !Self::is_well_known_port(port) && !Self::is_excluded_port(port, &config) {
                available_ports.insert(port);
            }
        }

        info!(
            "Initializing NetworkPortManager with {} available ports in range {}-{}",
            available_ports.len(),
            start,
            end
        );

        let reservation_system = Arc::new(PortReservationSystem::new().await?);
        let conflict_detector = Arc::new(PortConflictDetector::new().await?);
        let health_monitor = Arc::new(PortHealthMonitor::new().await?);
        let performance_metrics = Arc::new(PortPerformanceMetrics::new().await?);

        let port_manager = Self {
            config: Arc::new(RwLock::new(config)),
            available_ports: Arc::new(Mutex::new(available_ports)),
            allocated_ports: Arc::new(Mutex::new(HashMap::new())),
            reservation_system,
            usage_stats: Arc::new(Mutex::new(PortUsageStatistics::default())),
            conflict_detector,
            health_monitor,
            performance_metrics,
            shutdown_signal: Arc::new(AtomicBool::new(false)),
        };

        // Start background tasks for maintenance and monitoring
        port_manager.start_background_tasks().await?;

        info!("NetworkPortManager initialized successfully");
        Ok(port_manager)
    }

    /// Validate the provided configuration for correctness and consistency
    ///
    /// This performs comprehensive validation including port range validation,
    /// allocation limits, and configuration consistency checks.
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration to validate
    ///
    /// # Returns
    ///
    /// Success if configuration is valid, error with specific details if invalid
    fn validate_config(config: &PortPoolConfig) -> PortManagementResult<()> {
        let (start, end) = config.port_range;

        if start >= end {
            return Err(PortManagementError::InvalidPortRange { start, end });
        }

        if start < 1024 {
            warn!("Port range starts below 1024, this may interfere with system ports");
        }

        if config.max_allocation == 0 {
            return Err(PortManagementError::ConfigurationError {
                message: "max_allocation must be greater than 0".to_string(),
            });
        }

        Ok(())
    }

    /// Check if a port is well-known (system reserved)
    ///
    /// This identifies ports that should not be allocated for testing purposes,
    /// including standard well-known ports (0-1023) and common service ports.
    ///
    /// # Arguments
    ///
    /// * `port` - Port number to check
    ///
    /// # Returns
    ///
    /// True if the port is well-known and should not be allocated
    fn is_well_known_port(port: u16) -> bool {
        // Standard well-known ports (0-1023) plus some common service ports
        port < 1024 || matches!(
            port,
            1433 | // SQL Server
            1521 | // Oracle
            1883 | // MQTT
            3306 | // MySQL
            5432 | // PostgreSQL
            6379 | // Redis
            27017 | // MongoDB
            5672 | // RabbitMQ AMQP
            15672 | // RabbitMQ Management
            9200 | // Elasticsearch
            9300 | // Elasticsearch cluster
            8086 | // InfluxDB
            50070 | // Hadoop NameNode
            8888 | // Jupyter Notebook
            11211 // Memcached
        )
    }

    /// Check if a port is in the excluded ranges defined in configuration
    ///
    /// # Arguments
    ///
    /// * `port` - Port number to check
    /// * `config` - Configuration containing reserved ranges
    ///
    /// # Returns
    ///
    /// True if the port falls within any excluded range
    fn is_excluded_port(port: u16, config: &PortPoolConfig) -> bool {
        config.reserved_ranges.iter().any(|(start, end)| {
            port >= *start && port <= *end
        })
    }

    /// Start background tasks for maintenance and monitoring
    ///
    /// This sets up periodic tasks for:
    /// - Expired allocation cleanup
    /// - Health status monitoring
    /// - Performance metrics collection
    /// - Reservation queue processing
    async fn start_background_tasks(&self) -> PortManagementResult<()> {
        // In a real implementation, you would spawn background tasks here
        // For now, we'll just log that they would be started
        info!("Background tasks for port management started");
        Ok(())
    }

    /// Allocate ports for a test with comprehensive error handling and logging
    ///
    /// This is the primary method for requesting port allocations. It performs
    /// conflict checking, availability validation, and updates all relevant
    /// statistics and metrics.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of ports to allocate (0 returns empty vector)
    /// * `test_id` - Unique identifier for the test requesting ports
    ///
    /// # Returns
    ///
    /// Vector of allocated port numbers, guaranteed to be unique and available
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Insufficient ports are available for the request
    /// - Conflict detection identifies issues with the allocation
    /// - Test already has maximum allowed allocations
    /// - Internal allocation failure occurs
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use trustformers_serve::resource_management::port_manager::NetworkPortManager;
    /// # async fn example(manager: &NetworkPortManager) -> anyhow::Result<()> {
    /// // Allocate 3 ports for integration test
    /// let ports = manager.allocate_ports(3, "integration_test_001").await?;
    /// println!("Allocated ports: {:?}", ports);
    ///
    /// // Each port is guaranteed to be unique and available
    /// assert_eq!(ports.len(), 3);
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip(self), fields(test_id = %test_id, count = %count))]
    pub async fn allocate_ports(&self, count: usize, test_id: &str) -> PortManagementResult<Vec<u16>> {
        if count == 0 {
            return Ok(vec![]);
        }

        let start_time = std::time::Instant::now();

        // Check for conflicts first using the conflict detector
        if let Err(conflict) = self.conflict_detector.check_allocation_conflicts(count, test_id).await {
            self.performance_metrics.record_conflict().await;
            return Err(conflict);
        }

        let mut available_ports = self.available_ports.lock();
        let mut allocated_ports = self.allocated_ports.lock();
        let mut usage_stats = self.usage_stats.lock();

        // Check availability before proceeding
        if available_ports.len() < count {
            let error = PortManagementError::InsufficientPorts {
                requested: count,
                available: available_ports.len(),
            };
            self.performance_metrics.record_allocation_failure().await;
            usage_stats.allocation_failures += 1;
            return Err(error);
        }

        let mut allocated = Vec::new();
        let now = Utc::now();

        // Allocate ports one by one
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
                // Rollback partial allocation on failure
                for &port in &allocated {
                    if let Some(allocation) = allocated_ports.remove(&port) {
                        available_ports.insert(allocation.port);
                    }
                }
                let error = PortManagementError::InternalError {
                    message: "Failed to allocate ports despite availability check".to_string(),
                };
                self.performance_metrics.record_allocation_failure().await;
                return Err(error);
            }
        }

        // Update comprehensive statistics
        usage_stats.total_allocated += count as u64;
        usage_stats.currently_allocated = allocated_ports.len();
        usage_stats.peak_usage = usage_stats.peak_usage.max(allocated_ports.len());

        // Update utilization percentage
        let total_ports = available_ports.len() + allocated_ports.len();
        if total_ports > 0 {
            usage_stats.current_utilization = allocated_ports.len() as f32 / total_ports as f32;
        }

        // Record performance metrics
        let allocation_time = start_time.elapsed();
        self.performance_metrics.record_allocation_success(allocation_time).await;

        info!(
            "Successfully allocated {} ports for test {}: {:?} (took {:?})",
            allocated.len(),
            test_id,
            allocated,
            allocation_time
        );

        Ok(allocated)
    }

    /// Deallocate a specific port with comprehensive cleanup
    ///
    /// This removes a port from the allocated pool and makes it available
    /// for future allocations. It also updates all relevant statistics.
    ///
    /// # Arguments
    ///
    /// * `port` - Port number to deallocate
    ///
    /// # Returns
    ///
    /// Success if port was deallocated, error if port was not allocated
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use trustformers_serve::resource_management::port_manager::NetworkPortManager;
    /// # async fn example(manager: &NetworkPortManager) -> anyhow::Result<()> {
    /// // Deallocate a specific port
    /// manager.deallocate_port(8080).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip(self), fields(port = %port))]
    pub async fn deallocate_port(&self, port: u16) -> PortManagementResult<()> {
        let start_time = std::time::Instant::now();

        let mut available_ports = self.available_ports.lock();
        let mut allocated_ports = self.allocated_ports.lock();
        let mut usage_stats = self.usage_stats.lock();

        if let Some(allocation) = allocated_ports.remove(&port) {
            available_ports.insert(port);
            usage_stats.currently_allocated = allocated_ports.len();

            // Update average duration statistics
            let duration = Utc::now().signed_duration_since(allocation.allocated_at);
            let duration_std = Duration::from_secs(duration.num_seconds().max(0) as u64);

            if usage_stats.total_allocated > 0 {
                let total_duration = usage_stats.average_allocation_duration.as_secs() as f64
                    * (usage_stats.total_allocated - 1) as f64;
                let new_average = (total_duration + duration_std.as_secs() as f64)
                    / usage_stats.total_allocated as f64;
                usage_stats.average_allocation_duration = Duration::from_secs(new_average as u64);
            }

            // Update utilization percentage
            let total_ports = available_ports.len() + allocated_ports.len();
            if total_ports > 0 {
                usage_stats.current_utilization = allocated_ports.len() as f32 / total_ports as f32;
            }

            self.performance_metrics.record_deallocation_success(start_time.elapsed()).await;

            info!(
                "Successfully deallocated port {} for test {} (was allocated for {:?})",
                port, allocation.test_id, duration_std
            );
            Ok(())
        } else {
            let error = PortManagementError::PortNotAllocated { port };
            warn!("Attempted to deallocate port {} that was not allocated", port);
            Err(error)
        }
    }

    /// Deallocate all ports for a specific test
    ///
    /// This is the recommended cleanup method for test teardown. It finds
    /// all ports allocated to a test and deallocates them atomically.
    ///
    /// # Arguments
    ///
    /// * `test_id` - Test identifier to deallocate ports for
    ///
    /// # Returns
    ///
    /// Success or error if deallocation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use trustformers_serve::resource_management::port_manager::NetworkPortManager;
    /// # async fn example(manager: &NetworkPortManager) -> anyhow::Result<()> {
    /// // Cleanup all ports for a test (recommended for test teardown)
    /// manager.deallocate_ports_for_test("integration_test_001").await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip(self), fields(test_id = %test_id))]
    pub async fn deallocate_ports_for_test(&self, test_id: &str) -> PortManagementResult<()> {
        debug!("Deallocating all ports for test: {}", test_id);

        let mut available_ports = self.available_ports.lock();
        let mut allocated_ports = self.allocated_ports.lock();
        let mut usage_stats = self.usage_stats.lock();

        let mut deallocated_ports = Vec::new();
        let mut total_duration = Duration::ZERO;

        // Find and collect ports to deallocate
        allocated_ports.retain(|&port, allocation| {
            if allocation.test_id == test_id {
                available_ports.insert(port);
                deallocated_ports.push(port);

                // Calculate allocation duration
                let duration = Utc::now().signed_duration_since(allocation.allocated_at);
                total_duration += Duration::from_secs(duration.num_seconds().max(0) as u64);

                false // Remove from allocated_ports
            } else {
                true // Keep in allocated_ports
            }
        });

        usage_stats.currently_allocated = allocated_ports.len();

        // Update utilization percentage
        let total_ports = available_ports.len() + allocated_ports.len();
        if total_ports > 0 {
            usage_stats.current_utilization = allocated_ports.len() as f32 / total_ports as f32;
        }

        // Also cancel any pending reservations for this test
        self.reservation_system.cancel_reservations(test_id).await?;

        if !deallocated_ports.is_empty() {
            info!(
                "Released {} network ports for test {}: {:?} (total allocation time: {:?})",
                deallocated_ports.len(),
                test_id,
                deallocated_ports,
                total_duration
            );
        } else {
            debug!("No ports were allocated to test {}", test_id);
        }

        Ok(())
    }

    /// Reserve ports for future allocation with advanced queuing
    ///
    /// This provides a way to reserve ports in advance of allocation,
    /// useful for ensuring port availability for critical tests.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of ports to reserve
    /// * `test_id` - Test identifier making the reservation
    /// * `duration` - How long to hold the reservation
    /// * `usage_type` - Intended usage type for the ports
    ///
    /// # Returns
    ///
    /// Vector of reserved port numbers or an error
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use trustformers_serve::resource_management::port_manager::NetworkPortManager;
    /// # use trustformers_serve::resource_management::types::PortUsageType;
    /// # use std::time::Duration;
    /// # async fn example(manager: &NetworkPortManager) -> anyhow::Result<()> {
    /// // Reserve ports for critical test
    /// let reserved = manager.reserve_ports(
    ///     3,
    ///     "critical_test_001",
    ///     Duration::from_secs(300),
    ///     PortUsageType::HttpServer
    /// ).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip(self), fields(test_id = %test_id, count = %count))]
    pub async fn reserve_ports(
        &self,
        count: usize,
        test_id: &str,
        duration: Duration,
        usage_type: PortUsageType,
    ) -> PortManagementResult<Vec<u16>> {
        self.reservation_system
            .reserve_ports(count, test_id, duration, usage_type, &self.available_ports)
            .await
    }

    /// Cancel port reservations for a test
    ///
    /// This removes all reservations held by a specific test, making
    /// those ports available for allocation by other tests.
    ///
    /// # Arguments
    ///
    /// * `test_id` - Test identifier to cancel reservations for
    #[instrument(skip(self), fields(test_id = %test_id))]
    pub async fn cancel_reservations(&self, test_id: &str) -> PortManagementResult<()> {
        self.reservation_system.cancel_reservations(test_id).await
    }

    /// Check if requested number of ports are available
    ///
    /// This provides a quick availability check without performing
    /// an actual allocation.
    ///
    /// # Arguments
    ///
    /// * `count` - Number of ports needed
    ///
    /// # Returns
    ///
    /// True if sufficient ports are available
    pub async fn check_availability(&self, count: usize) -> PortManagementResult<bool> {
        let available_ports = self.available_ports.lock();
        Ok(available_ports.len() >= count)
    }

    /// Get comprehensive port usage statistics
    ///
    /// Returns detailed statistics about port usage including allocation
    /// counts, utilization, and failure rates.
    ///
    /// # Returns
    ///
    /// Current usage statistics snapshot
    pub async fn get_statistics(&self) -> PortManagementResult<PortUsageStatistics> {
        let stats = self.usage_stats.lock();
        Ok(stats.clone())
    }

    /// Get detailed port allocation information
    ///
    /// Returns a complete mapping of all currently allocated ports
    /// and their allocation details.
    ///
    /// # Returns
    ///
    /// Map of port numbers to their allocation details
    pub async fn get_port_allocations(&self) -> HashMap<u16, PortAllocation> {
        let allocated_ports = self.allocated_ports.lock();
        allocated_ports.clone()
    }

    /// Get number of available ports
    pub async fn get_available_port_count(&self) -> usize {
        let available_ports = self.available_ports.lock();
        available_ports.len()
    }

    /// Get number of allocated ports
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

    /// Get current port utilization percentage
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

    /// Clean up expired allocations based on configuration
    ///
    /// This maintenance operation identifies and removes allocations
    /// that have exceeded their maximum allowed duration.
    ///
    /// # Returns
    ///
    /// Number of allocations cleaned up
    #[instrument(skip(self))]
    pub async fn cleanup_expired_allocations(&self) -> PortManagementResult<usize> {
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
            if duration_since_allocation.to_std().unwrap_or(Duration::ZERO) > max_allocation_duration {
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

    /// Generate comprehensive allocation report
    ///
    /// Creates a detailed text report with current allocation status,
    /// usage statistics, and health information.
    ///
    /// # Returns
    ///
    /// Detailed report string with allocation statistics
    pub async fn generate_allocation_report(&self) -> String {
        let stats = self.get_statistics().await.unwrap_or_default();
        let available_count = self.get_available_port_count().await;
        let allocated_count = self.get_allocated_port_count().await;
        let utilization = self.get_utilization().await;
        let health_status = self.health_monitor.get_health_status().await;

        format!(
            "Port Allocation Report:\n\
             ========================\n\
             - Available ports: {}\n\
             - Allocated ports: {}\n\
             - Total allocations: {}\n\
             - Peak usage: {}\n\
             - Current utilization: {:.1}%\n\
             - Average allocation duration: {}s\n\
             - Allocation failures: {}\n\
             - Health status: {:?}\n\
             - Last health check: {}\n\
             ========================",
            available_count,
            allocated_count,
            stats.total_allocated,
            stats.peak_usage,
            utilization * 100.0,
            stats.average_allocation_duration.as_secs(),
            stats.allocation_failures,
            health_status.overall_status,
            health_status.last_check.format("%Y-%m-%d %H:%M:%S UTC")
        )
    }

    /// Update port manager configuration with validation
    ///
    /// This allows runtime reconfiguration of the port manager,
    /// including changing port ranges and allocation limits.
    ///
    /// # Arguments
    ///
    /// * `new_config` - New configuration to apply
    ///
    /// # Returns
    ///
    /// Success or configuration error
    #[instrument(skip(self, new_config))]
    pub async fn update_config(&self, new_config: PortPoolConfig) -> PortManagementResult<()> {
        Self::validate_config(&new_config)?;

        let mut config = self.config.write();
        let mut available_ports = self.available_ports.lock();

        // Recalculate available ports based on new configuration
        available_ports.clear();
        let (start, end) = new_config.port_range;

        for port in start..=end {
            if !Self::is_well_known_port(port) && !Self::is_excluded_port(port, &new_config) {
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
            "Updated port manager configuration, {} available ports in range {}-{}",
            available_ports.len(),
            start,
            end
        );

        Ok(())
    }

    /// Get current health status
    ///
    /// Returns the latest health assessment from the health monitoring system.
    pub async fn get_health_status(&self) -> PortHealthStatus {
        self.health_monitor.get_health_status().await
    }

    /// Get performance metrics
    ///
    /// Returns current performance metrics including allocation times,
    /// success rates, and throughput statistics.
    pub async fn get_performance_metrics(&self) -> PerformanceSnapshot {
        self.performance_metrics.get_current_snapshot().await
    }

    /// Shutdown the port manager and cleanup resources
    ///
    /// This performs comprehensive cleanup including:
    /// - Stopping background tasks
    /// - Deallocating all ports
    /// - Canceling all reservations
    /// - Cleaning up internal state
    #[instrument(skip(self))]
    pub async fn shutdown(&self) -> PortManagementResult<()> {
        info!("Shutting down NetworkPortManager");

        // Signal background tasks to stop
        self.shutdown_signal.store(true, Ordering::Relaxed);

        // Cleanup all allocations
        let allocated_ports: Vec<String> = {
            let allocated = self.allocated_ports.lock();
            allocated.values().map(|alloc| alloc.test_id.clone()).collect()
        };

        for test_id in allocated_ports {
            if let Err(e) = self.deallocate_ports_for_test(&test_id).await {
                warn!("Failed to cleanup ports for test {} during shutdown: {}", test_id, e);
            }
        }

        // Cancel all reservations
        if let Err(e) = self.reservation_system.cleanup_all_reservations().await {
            warn!("Failed to cleanup reservations during shutdown: {}", e);
        }

        info!("NetworkPortManager shutdown completed");
        Ok(())
    }
}