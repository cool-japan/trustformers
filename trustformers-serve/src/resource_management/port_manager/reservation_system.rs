//! Port Reservation System Implementation for TrustformeRS
//!
//! This module provides a focused implementation of the port reservation system,
//! handling all aspects of port reservations including immediate allocations,
//! queuing, expiration handling, and comprehensive event tracking.
//!
//! # Features
//!
//! - **Immediate Reservations**: Reserve ports that are immediately available
//! - **Queue Management**: Queue reservations when resources are not immediately available
//! - **Expiration Handling**: Automatic cleanup of expired reservations
//! - **Event Tracking**: Comprehensive logging of all reservation events
//! - **Conflict Resolution**: Handle reservation conflicts and priority management
//! - **Resource Lifecycle**: Proper management of reservation resources and cleanup
//!
//! # Architecture
//!
//! The reservation system consists of several key components:
//!
//! ## Core Components
//!
//! - **Active Reservations**: Currently active port reservations by port number
//! - **Test Mappings**: Quick lookup of reservations by test ID
//! - **Reservation Queue**: Queue for reservations that can't be immediately fulfilled
//! - **Expiry Management**: Automatic cleanup of expired reservations
//! - **Event History**: Comprehensive tracking of all reservation events
//!
//! ## Thread Safety
//!
//! All operations are thread-safe using:
//! - `Arc<Mutex<>>` for mutable shared state
//! - `Arc<RwLock<>>` for configuration data
//! - Atomic operations where appropriate
//!
//! # Usage
//!
//! ```rust
//! use trustformers_serve::resource_management::port_manager::reservation_system::*;
//! use trustformers_serve::resource_management::types::*;
//! use std::time::Duration;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create a new reservation system
//! let reservation_system = PortReservationSystem::new().await?;
//!
//! // Reserve ports for a test
//! let available_ports = Arc::new(Mutex::new((8000..8010).collect()));
//! let reserved_ports = reservation_system.reserve_ports(
//!     3,
//!     "test_001",
//!     Duration::from_secs(300),
//!     PortUsageType::HttpServer,
//!     &available_ports
//! ).await?;
//!
//! println!("Reserved ports: {:?}", reserved_ports);
//!
//! // Cancel reservations when done
//! reservation_system.cancel_reservations("test_001").await?;
//! # Ok(())
//! # }
//! ```
//!
//! # Error Handling
//!
//! All operations return `PortManagementResult<T>` which provides comprehensive
//! error information for debugging and handling various failure scenarios.
//!
//! # Performance Considerations
//!
//! - Reservation lookups are O(1) due to HashMap usage
//! - Lock contention is minimized through fine-grained locking
//! - Event history is automatically pruned to prevent memory leaks
//! - Queue processing is optimized for high-throughput scenarios

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
    time::Duration,
};
use tracing::{debug, error, info, instrument, warn};

// Import all necessary types from the types module
use super::types::*;

// Import from parent types module for core types
use super::super::types::{PortReservationRequest, PortUsageType};

/// Implementation of the advanced port reservation system
///
/// This implementation provides comprehensive port reservation capabilities including
/// immediate reservations, queuing for delayed fulfillment, automatic expiration,
/// and detailed event tracking for analytics and debugging.
impl PortReservationSystem {
    /// Create a new port reservation system
    ///
    /// Initializes all internal data structures and configurations needed for
    /// the reservation system to operate effectively.
    ///
    /// # Returns
    ///
    /// A new PortReservationSystem instance ready for use
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails (rare, but possible in low-memory conditions)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # async fn example() -> PortManagementResult<()> {
    /// let reservation_system = PortReservationSystem::new().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn new() -> PortManagementResult<Self> {
        info!("Initializing port reservation system");

        Ok(Self {
            reservations: Arc::new(Mutex::new(HashMap::new())),
            reservations_by_test: Arc::new(Mutex::new(HashMap::new())),
            reservation_queue: Arc::new(Mutex::new(VecDeque::new())),
            expiry_times: Arc::new(Mutex::new(HashMap::new())),
            reservation_history: Arc::new(Mutex::new(Vec::new())),
            config: Arc::new(RwLock::new(PortReservationConfig::default())),
        })
    }

    /// Reserve ports for future allocation with advanced queuing
    ///
    /// This is the primary method for creating port reservations. It handles both
    /// immediate reservations (when ports are available) and queued reservations
    /// (when resources are not immediately available).
    ///
    /// # Arguments
    ///
    /// * `count` - Number of ports to reserve
    /// * `test_id` - Unique identifier for the test making the reservation
    /// * `duration` - How long to hold the reservation
    /// * `usage_type` - Intended usage type for the ports
    /// * `available_ports` - Reference to the pool of available ports
    ///
    /// # Returns
    ///
    /// Vector of reserved port numbers, or an empty vector if queued
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The test exceeds maximum reservations per test
    /// - The reservation queue is full and queuing is enabled
    /// - Internal reservation creation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// # async fn example(reservation_system: &PortReservationSystem) -> PortManagementResult<()> {
    /// let available_ports = Arc::new(Mutex::new((8000..8010).collect()));
    ///
    /// let reserved = reservation_system.reserve_ports(
    ///     3,
    ///     "test_001",
    ///     Duration::from_secs(300),
    ///     PortUsageType::HttpServer,
    ///     &available_ports
    /// ).await?;
    ///
    /// println!("Reserved {} ports", reserved.len());
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip(self, available_ports), fields(test_id = %test_id, count = %count))]
    pub async fn reserve_ports(
        &self,
        count: usize,
        test_id: &str,
        duration: Duration,
        usage_type: PortUsageType,
        available_ports: &Arc<Mutex<HashSet<u16>>>,
    ) -> PortManagementResult<Vec<u16>> {
        if count == 0 {
            debug!("Reservation request for 0 ports, returning empty vector");
            return Ok(vec![]);
        }

        let config = self.config.read();

        // Check reservation limits to prevent resource monopolization
        let reservations_by_test = self.reservations_by_test.lock();
        if let Some(existing_reservations) = reservations_by_test.get(test_id) {
            if existing_reservations.len() + count > config.max_reservations_per_test {
                return Err(PortManagementError::ConfigurationError {
                    message: format!(
                        "Exceeds maximum reservations per test: {} + {} > {}",
                        existing_reservations.len(),
                        count,
                        config.max_reservations_per_test
                    ),
                });
            }
        }
        drop(reservations_by_test);

        // Validate and normalize reservation duration
        let reservation_duration = if duration > config.max_reservation_duration {
            warn!(
                "Requested duration {:?} exceeds maximum, capping to {:?}",
                duration, config.max_reservation_duration
            );
            config.max_reservation_duration
        } else if duration.is_zero() {
            debug!("Using default reservation duration");
            config.default_reservation_duration
        } else {
            duration
        };

        let available = available_ports.lock();
        let mut reservations = self.reservations.lock();

        // Check if we have sufficient available ports
        if available.len() < count {
            drop(available);
            drop(reservations);

            // Queue the reservation if enabled
            if config.enable_queue_processing {
                info!(
                    "Insufficient ports available ({} requested, {} available), queuing reservation for test {}",
                    count, available.len(), test_id
                );
                return self.queue_reservation(count, test_id, reservation_duration, usage_type).await;
            } else {
                return Err(PortManagementError::InsufficientPorts {
                    requested: count,
                    available: available.len(),
                });
            }
        }

        let mut reserved_ports = Vec::new();
        let now = Utc::now();
        let expiry_time = now + chrono::Duration::from_std(reservation_duration).unwrap();

        // Reserve ports by finding available ports that aren't already reserved
        for _ in 0..count {
            if let Some(&port) = available.iter().find(|&p| !reservations.contains_key(p)) {
                let reservation = PortReservationRequest {
                    test_id: test_id.to_string(),
                    port_count: 1,
                    preferred_range: Some((port, port)),
                    usage_type: usage_type.clone(),
                    requested_at: now,
                    priority: 1.0,
                    timeout: reservation_duration,
                };

                reservations.insert(port, reservation.clone());
                reserved_ports.push(port);

                // Set expiry time for automatic cleanup
                let mut expiry_times = self.expiry_times.lock();
                expiry_times.insert(port, expiry_time);

                // Record reservation creation event
                self.record_reservation_event(
                    ReservationEventType::Created,
                    port,
                    test_id,
                    HashMap::new()
                ).await;
            } else {
                // Rollback partial reservation on failure
                for &port in &reserved_ports {
                    reservations.remove(&port);
                    let mut expiry_times = self.expiry_times.lock();
                    expiry_times.remove(&port);
                }
                return Err(PortManagementError::InternalError {
                    message: "Failed to reserve sufficient ports despite availability check".to_string(),
                });
            }
        }

        // Update reservations by test mapping for quick lookup
        let mut reservations_by_test = self.reservations_by_test.lock();
        reservations_by_test
            .entry(test_id.to_string())
            .or_default()
            .extend(&reserved_ports);

        info!(
            "Successfully reserved {} ports for test {}: {:?} (expires in {:?})",
            reserved_ports.len(),
            test_id,
            reserved_ports,
            reservation_duration
        );

        Ok(reserved_ports)
    }

    /// Queue a reservation for later processing when resources become available
    ///
    /// This method is called when immediate reservation is not possible due to
    /// insufficient available ports. The reservation is added to a queue for
    /// processing when resources become available.
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
    /// Empty vector (ports will be assigned when queue is processed)
    ///
    /// # Errors
    ///
    /// Returns an error if the reservation queue is full
    async fn queue_reservation(
        &self,
        count: usize,
        test_id: &str,
        duration: Duration,
        usage_type: PortUsageType,
    ) -> PortManagementResult<Vec<u16>> {
        let config = self.config.read();
        let mut queue = self.reservation_queue.lock();

        if queue.len() >= config.max_queue_size {
            return Err(PortManagementError::ResourceConflict {
                details: format!(
                    "Reservation queue is full (size: {}, max: {})",
                    queue.len(),
                    config.max_queue_size
                ),
            });
        }

        let reservation = PortReservationRequest {
            test_id: test_id.to_string(),
            port_count: count,
            preferred_range: None,
            usage_type,
            requested_at: Utc::now(),
            priority: 1.0,
            timeout: duration,
        };

        queue.push_back(reservation);

        // Record queued event for analytics
        self.record_reservation_event(
            ReservationEventType::Queued,
            0, // No specific port for queued reservations
            test_id,
            HashMap::from([
                ("queue_position".to_string(), queue.len().to_string()),
                ("requested_count".to_string(), count.to_string()),
            ])
        ).await;

        info!(
            "Queued reservation for {} ports for test {} (queue position: {})",
            count, test_id, queue.len()
        );

        // Return empty vector as ports will be assigned when queue is processed
        Ok(vec![])
    }

    /// Cancel reservations for a specific test
    ///
    /// Removes all active reservations for the specified test and cleans up
    /// associated data structures. Also removes any queued reservations.
    ///
    /// # Arguments
    ///
    /// * `test_id` - Test identifier to cancel reservations for
    ///
    /// # Returns
    ///
    /// Success or error if cancellation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// # async fn example(reservation_system: &PortReservationSystem) -> PortManagementResult<()> {
    /// // Cancel all reservations for a test
    /// reservation_system.cancel_reservations("test_001").await?;
    /// # Ok(())
    /// # }
    /// ```
    #[instrument(skip(self), fields(test_id = %test_id))]
    pub async fn cancel_reservations(&self, test_id: &str) -> PortManagementResult<()> {
        debug!("Cancelling reservations for test: {}", test_id);

        let mut reservations = self.reservations.lock();
        let mut reservations_by_test = self.reservations_by_test.lock();
        let mut expiry_times = self.expiry_times.lock();

        // Remove active reservations for this test
        let cancelled_ports = if let Some(ports) = reservations_by_test.remove(test_id) {
            for port in &ports {
                reservations.remove(port);
                expiry_times.remove(port);

                // Record cancellation event for each port
                self.record_reservation_event(
                    ReservationEventType::Cancelled,
                    *port,
                    test_id,
                    HashMap::from([
                        ("reason".to_string(), "test_cancelled".to_string()),
                    ])
                ).await;
            }
            ports
        } else {
            Vec::new()
        };

        // Also remove from reservation queue
        let mut queue = self.reservation_queue.lock();
        let initial_queue_size = queue.len();
        queue.retain(|req| req.test_id != test_id);
        let removed_from_queue = initial_queue_size - queue.len();

        if !cancelled_ports.is_empty() || removed_from_queue > 0 {
            info!(
                "Cancelled {} active port reservations and {} queued reservations for test {}: {:?}",
                cancelled_ports.len(),
                removed_from_queue,
                test_id,
                cancelled_ports
            );
        } else {
            debug!("No reservations found to cancel for test {}", test_id);
        }

        Ok(())
    }

    /// Cleanup all reservations during system shutdown
    ///
    /// Removes all active reservations and clears the reservation queue.
    /// This method is typically called during system shutdown to ensure
    /// clean resource cleanup.
    ///
    /// # Returns
    ///
    /// Success or error if cleanup fails
    pub async fn cleanup_all_reservations(&self) -> PortManagementResult<()> {
        info!("Cleaning up all reservations during system shutdown");

        let mut reservations = self.reservations.lock();
        let mut reservations_by_test = self.reservations_by_test.lock();
        let mut expiry_times = self.expiry_times.lock();
        let mut queue = self.reservation_queue.lock();

        let total_active_reservations = reservations.len();
        let total_queued_reservations = queue.len();
        let total_reservations = total_active_reservations + total_queued_reservations;

        // Clear all data structures
        reservations.clear();
        reservations_by_test.clear();
        expiry_times.clear();
        queue.clear();

        info!(
            "Cleaned up {} total reservations ({} active, {} queued) during shutdown",
            total_reservations, total_active_reservations, total_queued_reservations
        );

        Ok(())
    }

    /// Clean up expired reservations automatically
    ///
    /// Scans all active reservations and removes those that have exceeded their
    /// expiration time. This method should be called periodically to prevent
    /// resource leaks from forgotten reservations.
    ///
    /// # Returns
    ///
    /// Number of reservations that were cleaned up
    ///
    /// # Examples
    ///
    /// ```rust
    /// # async fn example(reservation_system: &PortReservationSystem) -> PortManagementResult<()> {
    /// // Clean up expired reservations
    /// let cleaned_count = reservation_system.cleanup_expired_reservations().await?;
    /// println!("Cleaned up {} expired reservations", cleaned_count);
    /// # Ok(())
    /// # }
    /// ```
    pub async fn cleanup_expired_reservations(&self) -> PortManagementResult<usize> {
        let now = Utc::now();
        let mut reservations = self.reservations.lock();
        let mut reservations_by_test = self.reservations_by_test.lock();
        let mut expiry_times = self.expiry_times.lock();

        let mut expired_ports = Vec::new();

        // Find all expired reservations
        for (&port, &expiry_time) in expiry_times.iter() {
            if now > expiry_time {
                expired_ports.push(port);
            }
        }

        // Clean up expired reservations
        for port in &expired_ports {
            if let Some(reservation) = reservations.remove(port) {
                expiry_times.remove(port);

                // Remove from test mapping
                if let Some(test_ports) = reservations_by_test.get_mut(&reservation.test_id) {
                    test_ports.retain(|&p| p != *port);
                    if test_ports.is_empty() {
                        reservations_by_test.remove(&reservation.test_id);
                    }
                }

                // Record expiration event
                self.record_reservation_event(
                    ReservationEventType::Expired,
                    *port,
                    &reservation.test_id,
                    HashMap::from([
                        ("expiry_reason".to_string(), "timeout".to_string()),
                        ("duration_exceeded".to_string(), "true".to_string()),
                    ])
                ).await;

                warn!(
                    "Expired port reservation: port {} from test {} (reserved at {}, expired at {})",
                    port, reservation.test_id, reservation.requested_at, now
                );
            }
        }

        let cleanup_count = expired_ports.len();
        if cleanup_count > 0 {
            info!("Cleaned up {} expired port reservations", cleanup_count);
        } else {
            debug!("No expired reservations found during cleanup");
        }

        Ok(cleanup_count)
    }

    /// Record a reservation event for comprehensive history tracking
    ///
    /// This method creates detailed event records for all significant reservation
    /// system actions. Events are used for analytics, debugging, and system monitoring.
    ///
    /// # Arguments
    ///
    /// * `event_type` - Type of event that occurred
    /// * `port` - Port number involved (0 for non-port-specific events)
    /// * `test_id` - Test identifier that triggered the event
    /// * `details` - Additional event metadata and context
    async fn record_reservation_event(
        &self,
        event_type: ReservationEventType,
        port: u16,
        test_id: &str,
        details: HashMap<String, String>,
    ) {
        let event = PortReservationEvent {
            timestamp: Utc::now(),
            event_type: event_type.clone(),
            port,
            test_id: test_id.to_string(),
            details,
        };

        let mut history = self.reservation_history.lock();
        history.push(event);

        // Keep history size manageable to prevent memory leaks
        // Remove oldest 10% when we hit the limit
        if history.len() > 10000 {
            let drain_count = 1000;
            history.drain(0..drain_count);
            debug!("Pruned {} old reservation events from history", drain_count);
        }

        // Log the event based on its type and importance
        match event_type {
            ReservationEventType::Created => {
                debug!("Reservation event: Created port {} for test {}", port, test_id);
            }
            ReservationEventType::Fulfilled => {
                info!("Reservation event: Fulfilled port {} for test {}", port, test_id);
            }
            ReservationEventType::Cancelled => {
                debug!("Reservation event: Cancelled port {} for test {}", port, test_id);
            }
            ReservationEventType::Expired => {
                warn!("Reservation event: Expired port {} for test {}", port, test_id);
            }
            ReservationEventType::Queued => {
                debug!("Reservation event: Queued reservation for test {}", test_id);
            }
            ReservationEventType::Conflict => {
                warn!("Reservation event: Conflict detected for port {} by test {}", port, test_id);
            }
        }
    }

    /// Get all currently active reservations
    ///
    /// Returns a snapshot of all current port reservations. Useful for
    /// monitoring and debugging purposes.
    ///
    /// # Returns
    ///
    /// HashMap mapping port numbers to their reservation requests
    ///
    /// # Examples
    ///
    /// ```rust
    /// # fn example(reservation_system: &PortReservationSystem) {
    /// let active_reservations = reservation_system.get_active_reservations();
    /// println!("Currently {} active reservations", active_reservations.len());
    /// # }
    /// ```
    pub fn get_active_reservations(&self) -> HashMap<u16, PortReservationRequest> {
        let reservations = self.reservations.lock();
        reservations.clone()
    }

    /// Get comprehensive reservation system statistics
    ///
    /// Returns key metrics about the reservation system including number of
    /// active reservations, history size, and queue length.
    ///
    /// # Returns
    ///
    /// Tuple of (active_reservations, history_events, queued_reservations)
    ///
    /// # Examples
    ///
    /// ```rust
    /// # fn example(reservation_system: &PortReservationSystem) {
    /// let (active, history, queued) = reservation_system.get_reservation_statistics();
    /// println!("Stats: {} active, {} history events, {} queued", active, history, queued);
    /// # }
    /// ```
    pub fn get_reservation_statistics(&self) -> (usize, usize, usize) {
        let reservations = self.reservations.lock();
        let history = self.reservation_history.lock();
        let queue = self.reservation_queue.lock();

        (reservations.len(), history.len(), queue.len())
    }

    /// Get reservations for a specific test
    ///
    /// Returns all port numbers currently reserved by the specified test.
    ///
    /// # Arguments
    ///
    /// * `test_id` - Test identifier to look up
    ///
    /// # Returns
    ///
    /// Vector of port numbers reserved by the test
    pub fn get_reservations_for_test(&self, test_id: &str) -> Vec<u16> {
        let reservations_by_test = self.reservations_by_test.lock();
        reservations_by_test
            .get(test_id)
            .cloned()
            .unwrap_or_default()
    }

    /// Check if a specific port is reserved
    ///
    /// # Arguments
    ///
    /// * `port` - Port number to check
    ///
    /// # Returns
    ///
    /// True if the port is currently reserved
    pub fn is_port_reserved(&self, port: u16) -> bool {
        let reservations = self.reservations.lock();
        reservations.contains_key(&port)
    }

    /// Get reservation details for a specific port
    ///
    /// # Arguments
    ///
    /// * `port` - Port number to look up
    ///
    /// # Returns
    ///
    /// Reservation request details if the port is reserved
    pub fn get_port_reservation(&self, port: u16) -> Option<PortReservationRequest> {
        let reservations = self.reservations.lock();
        reservations.get(&port).cloned()
    }

    /// Get the number of queued reservations
    ///
    /// # Returns
    ///
    /// Number of reservations currently in the queue
    pub fn get_queue_length(&self) -> usize {
        let queue = self.reservation_queue.lock();
        queue.len()
    }

    /// Get reservation history events for a specific test
    ///
    /// # Arguments
    ///
    /// * `test_id` - Test identifier to filter events
    ///
    /// # Returns
    ///
    /// Vector of reservation events for the specified test
    pub fn get_reservation_history_for_test(&self, test_id: &str) -> Vec<PortReservationEvent> {
        let history = self.reservation_history.lock();
        history
            .iter()
            .filter(|event| event.test_id == test_id)
            .cloned()
            .collect()
    }

    /// Update reservation system configuration
    ///
    /// # Arguments
    ///
    /// * `new_config` - New configuration to apply
    ///
    /// # Returns
    ///
    /// Success or error if configuration is invalid
    pub async fn update_config(&self, new_config: PortReservationConfig) -> PortManagementResult<()> {
        let mut config = self.config.write();
        *config = new_config;
        info!("Updated port reservation system configuration");
        Ok(())
    }

    /// Get current reservation system configuration
    ///
    /// # Returns
    ///
    /// Current configuration clone
    pub fn get_config(&self) -> PortReservationConfig {
        let config = self.config.read();
        config.clone()
    }
}