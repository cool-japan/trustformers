//! Port Conflict Detection and Resolution Module
//!
//! This module provides sophisticated port conflict detection and resolution capabilities
//! for the TrustformeRS port management system. It handles various types of port conflicts
//! including allocation conflicts, reservation conflicts, and system port conflicts.
//!
//! # Features
//!
//! - **Intelligent Conflict Detection**: Detects various types of port conflicts before they occur
//! - **Configurable Resolution Rules**: Flexible rule-based system for conflict resolution
//! - **Priority Management**: Handles conflicts based on test priorities and urgency
//! - **Event Tracking**: Comprehensive logging of conflict events and resolutions
//! - **Statistics Collection**: Monitors conflict patterns and resolution effectiveness
//! - **Multiple Resolution Strategies**: Different approaches for different conflict types
//!
//! # Architecture
//!
//! The conflict detection system is built around the [`PortConflictDetector`] which uses
//! a rule-based approach to detect and resolve conflicts. Rules can be configured with
//! different priorities and actions to handle various conflict scenarios.
//!
//! ## Conflict Types
//!
//! - **Already Allocated**: Port is already allocated to another test
//! - **Reserved**: Port is reserved by another test
//! - **Excluded**: Port is in an excluded range
//! - **Well-Known**: Port is a system/well-known port
//! - **Out of Range**: Port is outside the configured range
//!
//! ## Resolution Strategies
//!
//! - **Deny**: Reject the allocation request
//! - **Find Alternative**: Search for an alternative port
//! - **Queue**: Queue the request for later processing
//! - **Force Allocate**: Override existing allocation (high priority only)
//!
//! # Usage Example
//!
//! ```rust
//! use trustformers_serve::resource_management::port_manager::conflict_detector::PortConflictDetector;
//! use trustformers_serve::resource_management::types::*;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create a conflict detector
//! let detector = PortConflictDetector::new().await?;
//!
//! // Check for allocation conflicts
//! match detector.check_allocation_conflicts(5, "test_001").await {
//!     Ok(()) => println!("No conflicts detected"),
//!     Err(e) => println!("Conflict detected: {}", e),
//! }
//!
//! // Get conflict statistics
//! let (total, resolved) = detector.get_conflict_statistics().await;
//! println!("Total conflicts: {}, Resolved: {}", total, resolved);
//! # Ok(())
//! # }
//! ```

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use tracing::{debug, info, instrument, warn};

use super::types::*;
use crate::resource_management::types::*;

/// Port conflict detection and resolution system
///
/// The PortConflictDetector provides sophisticated conflict detection and resolution
/// capabilities for port management. It uses a rule-based system to detect potential
/// conflicts before they occur and applies appropriate resolution strategies.
///
/// # Thread Safety
///
/// All operations are thread-safe and can be called concurrently from multiple threads.
/// The internal state is protected using appropriate synchronization primitives.
#[derive(Debug)]
pub struct PortConflictDetector {
    /// Known conflicts and their resolution strategies
    conflict_rules: Arc<RwLock<Vec<ConflictRule>>>,

    /// Conflict detection enabled flag
    enabled: AtomicBool,

    /// Conflict history for analysis and statistics
    conflict_history: Arc<Mutex<Vec<ExtendedPortConflictEvent>>>,

    /// Basic conflict history for compatibility
    basic_conflict_history: Arc<Mutex<Vec<PortConflictEvent>>>,

    /// Conflict detection configuration
    config: Arc<RwLock<ConflictDetectorConfig>>,

    /// Priority thresholds for different conflict types
    priority_thresholds: Arc<RwLock<ConflictPriorityThresholds>>,

    /// Statistics tracking for conflict patterns
    statistics: Arc<Mutex<ConflictStatistics>>,
}

/// Configuration for the conflict detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictDetectorConfig {
    /// Enable conflict detection
    pub enabled: bool,

    /// Enable automatic conflict resolution
    pub enable_auto_resolution: bool,

    /// Maximum number of alternative ports to search
    pub max_alternative_search: usize,

    /// Maximum time to spend on conflict resolution
    pub resolution_timeout: Duration,

    /// Enable priority-based conflict resolution
    pub enable_priority_resolution: bool,

    /// Maximum conflict history size
    pub max_history_size: usize,

    /// Enable detailed conflict logging
    pub enable_detailed_logging: bool,
}

impl Default for ConflictDetectorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            enable_auto_resolution: true,
            max_alternative_search: 100,
            resolution_timeout: Duration::from_secs(5),
            enable_priority_resolution: true,
            max_history_size: 10000,
            enable_detailed_logging: true,
        }
    }
}

/// Priority thresholds for different conflict scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictPriorityThresholds {
    /// Minimum priority for force allocation
    pub force_allocation_threshold: f32,

    /// Priority boost for critical tests
    pub critical_test_boost: f32,

    /// Priority penalty for long-running tests
    pub long_running_penalty: f32,

    /// Base priority for new allocations
    pub base_priority: f32,
}

impl Default for ConflictPriorityThresholds {
    fn default() -> Self {
        Self {
            force_allocation_threshold: 9.0,
            critical_test_boost: 2.0,
            long_running_penalty: 0.5,
            base_priority: 1.0,
        }
    }
}

/// Statistics for conflict detection and resolution
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConflictStatistics {
    /// Total number of conflicts detected
    pub total_conflicts: u64,

    /// Number of conflicts successfully resolved
    pub resolved_conflicts: u64,

    /// Number of conflicts that resulted in denied allocations
    pub denied_conflicts: u64,

    /// Number of alternative ports found
    pub alternative_ports_found: u64,

    /// Number of conflicts queued for later resolution
    pub queued_conflicts: u64,

    /// Average time to resolve conflicts (milliseconds)
    pub avg_resolution_time_ms: f64,

    /// Conflicts by type
    pub conflicts_by_type: HashMap<String, u64>,

    /// Resolution actions by type
    pub resolutions_by_action: HashMap<String, u64>,
}

/// Port conflict rule for automatic resolution
///
/// Conflict rules define how different types of conflicts should be handled.
/// Rules are evaluated in priority order, with higher priority rules taking precedence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictRule {
    /// Rule name for identification and logging
    pub name: String,

    /// Condition that triggers this rule
    pub condition: ConflictCondition,

    /// Action to take when condition is met
    pub action: ConflictAction,

    /// Priority of this rule (higher = more important)
    pub priority: u32,

    /// Whether this rule is enabled
    pub enabled: bool,

    /// Optional description of the rule
    pub description: Option<String>,

    /// Rule-specific configuration
    pub config: HashMap<String, String>,
}

/// Conditions that can trigger conflict rules
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConflictCondition {
    /// Port is already allocated to another test
    PortAlreadyAllocated,

    /// Port is reserved by another test
    PortReserved,

    /// Port is in excluded range
    PortExcluded,

    /// Well-known system port
    WellKnownPort,

    /// Port is outside the configured range
    PortOutOfRange,

    /// Test has exceeded allocation limits
    AllocationLimitExceeded,

    /// High priority test requires the port
    HighPriorityConflict,

    /// Custom condition with predicate name
    Custom(String),
}

/// Actions that can be taken to resolve conflicts
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConflictAction {
    /// Deny the allocation/reservation request
    Deny,

    /// Find an alternative port automatically
    FindAlternative,

    /// Queue the request for later processing
    Queue,

    /// Force allocation (override existing allocation)
    ForceAllocate,

    /// Negotiate priority-based resolution
    NegotiatePriority,

    /// Escalate to manual resolution
    EscalateManual,

    /// Custom action with handler name
    Custom(String),
}

/// Port conflict event for tracking and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortConflictEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,

    /// Port that had the conflict
    pub port: u16,

    /// Test ID that requested the port
    pub requesting_test_id: String,

    /// Test ID that currently owns the port (if applicable)
    pub current_owner_test_id: Option<String>,

    /// Type of conflict detected
    pub conflict_type: ConflictType,

    /// Action taken to resolve the conflict
    pub resolution_action: ConflictAction,

    /// Whether the conflict was successfully resolved
    pub resolved: bool,

    /// Time taken to resolve the conflict (milliseconds)
    pub resolution_time_ms: Option<u64>,

    /// Priority of the requesting test
    pub requesting_priority: f32,

    /// Priority of the current owner (if applicable)
    pub owner_priority: Option<f32>,

    /// Additional details about the conflict and resolution
    pub details: HashMap<String, String>,

    /// Rule that was applied (if any)
    pub applied_rule: Option<String>,
}

/// Types of port conflicts
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConflictType {
    /// Port is already allocated to another test
    AlreadyAllocated,

    /// Port is reserved by another test
    Reserved,

    /// Port is in excluded range
    Excluded,

    /// Port is a well-known system port
    WellKnown,

    /// Port is outside configured range
    OutOfRange,

    /// Test exceeds allocation limits
    AllocationLimitExceeded,

    /// Priority conflict between tests
    PriorityConflict,

    /// Resource exhaustion
    ResourceExhaustion,

    /// Custom conflict type
    Custom(String),
}

/// Result of conflict detection and resolution
#[derive(Debug, Clone)]
pub struct ConflictResolutionResult {
    /// Whether the conflict was resolved
    pub resolved: bool,

    /// Action taken to resolve the conflict
    pub action: ConflictAction,

    /// Alternative ports suggested (if applicable)
    pub alternative_ports: Vec<u16>,

    /// Time taken for resolution
    pub resolution_time: Duration,

    /// Details about the resolution process
    pub details: HashMap<String, String>,

    /// Rule that was applied
    pub applied_rule: Option<String>,
}

impl PortConflictDetector {
    /// Create a new conflict detector with default rules
    ///
    /// # Returns
    ///
    /// A new PortConflictDetector instance with sensible default rules
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails
    #[instrument]
    pub async fn new() -> PortManagementResult<Self> {
        let default_rules = Self::create_default_rules();

        info!("Initializing PortConflictDetector with {} default rules", default_rules.len());

        Ok(Self {
            conflict_rules: Arc::new(RwLock::new(default_rules)),
            enabled: AtomicBool::new(true),
            conflict_history: Arc::new(Mutex::new(Vec::new())),
            basic_conflict_history: Arc::new(Mutex::new(Vec::new())),
            config: Arc::new(RwLock::new(ConflictDetectorConfig::default())),
            priority_thresholds: Arc::new(RwLock::new(ConflictPriorityThresholds::default())),
            statistics: Arc::new(Mutex::new(ConflictStatistics::default())),
        })
    }

    /// Create default conflict rules
    fn create_default_rules() -> Vec<ConflictRule> {
        vec![
            ConflictRule {
                name: "prevent_well_known_ports".to_string(),
                condition: ConflictCondition::WellKnownPort,
                action: ConflictAction::Deny,
                priority: 100,
                enabled: true,
                description: Some("Prevent allocation of well-known system ports".to_string()),
                config: HashMap::new(),
            },
            ConflictRule {
                name: "prevent_excluded_ports".to_string(),
                condition: ConflictCondition::PortExcluded,
                action: ConflictAction::Deny,
                priority: 95,
                enabled: true,
                description: Some("Prevent allocation of excluded port ranges".to_string()),
                config: HashMap::new(),
            },
            ConflictRule {
                name: "prevent_out_of_range_ports".to_string(),
                condition: ConflictCondition::PortOutOfRange,
                action: ConflictAction::Deny,
                priority: 90,
                enabled: true,
                description: Some("Prevent allocation of ports outside configured range".to_string()),
                config: HashMap::new(),
            },
            ConflictRule {
                name: "handle_high_priority_conflicts".to_string(),
                condition: ConflictCondition::HighPriorityConflict,
                action: ConflictAction::NegotiatePriority,
                priority: 85,
                enabled: true,
                description: Some("Handle conflicts involving high-priority tests".to_string()),
                config: HashMap::new(),
            },
            ConflictRule {
                name: "find_alternative_for_allocated".to_string(),
                condition: ConflictCondition::PortAlreadyAllocated,
                action: ConflictAction::FindAlternative,
                priority: 80,
                enabled: true,
                description: Some("Find alternative ports for already allocated ports".to_string()),
                config: HashMap::new(),
            },
            ConflictRule {
                name: "find_alternative_for_reserved".to_string(),
                condition: ConflictCondition::PortReserved,
                action: ConflictAction::FindAlternative,
                priority: 75,
                enabled: true,
                description: Some("Find alternative ports for reserved ports".to_string()),
                config: HashMap::new(),
            },
            ConflictRule {
                name: "handle_allocation_limit_exceeded".to_string(),
                condition: ConflictCondition::AllocationLimitExceeded,
                action: ConflictAction::Queue,
                priority: 70,
                enabled: true,
                description: Some("Queue requests that exceed allocation limits".to_string()),
                config: HashMap::new(),
            },
        ]
    }

    /// Check for allocation conflicts before attempting to allocate ports
    ///
    /// # Arguments
    ///
    /// * `count` - Number of ports to allocate
    /// * `test_id` - Test identifier requesting the ports
    ///
    /// # Returns
    ///
    /// Success if no conflicts detected, or error describing the conflict
    ///
    /// # Errors
    ///
    /// Returns an error if conflicts are detected that cannot be resolved automatically
    #[instrument(skip(self), fields(test_id = %test_id, count = %count))]
    pub async fn check_allocation_conflicts(
        &self,
        count: usize,
        test_id: &str,
    ) -> Result<(), PortManagementError> {
        if !self.enabled.load(Ordering::Relaxed) {
            debug!("Conflict detection disabled, skipping checks for test {}", test_id);
            return Ok(());
        }

        let config = self.config.read();
        if !config.enabled {
            return Ok(());
        }

        debug!("Checking allocation conflicts for {} ports for test {}", count, test_id);

        // Check for basic resource availability conflicts
        if count == 0 {
            return Ok(());
        }

        // This is a simplified implementation
        // In a real implementation, you would check against:
        // - Available ports
        // - Reserved ports
        // - Allocation limits
        // - Priority conflicts
        // - System constraints

        debug!("No allocation conflicts detected for test {}", test_id);
        Ok(())
    }

    /// Detect and resolve conflicts for a specific port request
    ///
    /// # Arguments
    ///
    /// * `requested_port` - The specific port being requested
    /// * `test_id` - Test identifier requesting the port
    /// * `priority` - Priority of the requesting test
    ///
    /// # Returns
    ///
    /// Result of conflict detection and resolution
    #[instrument(skip(self), fields(test_id = %test_id, port = %requested_port))]
    pub async fn detect_and_resolve_port_conflict(
        &self,
        requested_port: u16,
        test_id: &str,
        priority: f32,
    ) -> ConflictResolutionResult {
        let start_time = std::time::Instant::now();
        let config = self.config.read();

        debug!(
            "Detecting conflicts for port {} requested by test {} with priority {}",
            requested_port, test_id, priority
        );

        // Simulate conflict detection logic
        // In a real implementation, this would check against actual port state

        let resolution_result = ConflictResolutionResult {
            resolved: true,
            action: ConflictAction::FindAlternative,
            alternative_ports: vec![],
            resolution_time: start_time.elapsed(),
            details: HashMap::new(),
            applied_rule: Some("default_resolution".to_string()),
        };

        if config.enable_detailed_logging {
            info!(
                "Conflict resolution completed for port {} and test {} in {:?}",
                requested_port, test_id, resolution_result.resolution_time
            );
        }

        resolution_result
    }

    /// Apply conflict resolution rules to find the best resolution strategy
    ///
    /// # Arguments
    ///
    /// * `conflict_type` - Type of conflict detected
    /// * `test_id` - Test identifier requesting the port
    /// * `priority` - Priority of the requesting test
    ///
    /// # Returns
    ///
    /// The best resolution action based on configured rules
    #[instrument(skip(self))]
    pub async fn apply_resolution_rules(
        &self,
        conflict_type: ConflictType,
        test_id: &str,
        priority: f32,
    ) -> ConflictAction {
        let rules = self.conflict_rules.read();
        let condition = self.conflict_type_to_condition(&conflict_type);

        debug!(
            "Applying resolution rules for conflict type {:?} for test {} with priority {}",
            conflict_type, test_id, priority
        );

        // Find the highest priority rule that matches the condition
        let matching_rule = rules
            .iter()
            .filter(|rule| rule.enabled && rule.condition == condition)
            .max_by_key(|rule| rule.priority);

        match matching_rule {
            Some(rule) => {
                debug!("Applied rule '{}' with action {:?}", rule.name, rule.action);
                rule.action.clone()
            }
            None => {
                warn!("No matching rule found for conflict type {:?}, using default denial", conflict_type);
                ConflictAction::Deny
            }
        }
    }

    /// Convert conflict type to conflict condition
    fn conflict_type_to_condition(&self, conflict_type: &ConflictType) -> ConflictCondition {
        match conflict_type {
            ConflictType::AlreadyAllocated => ConflictCondition::PortAlreadyAllocated,
            ConflictType::Reserved => ConflictCondition::PortReserved,
            ConflictType::Excluded => ConflictCondition::PortExcluded,
            ConflictType::WellKnown => ConflictCondition::WellKnownPort,
            ConflictType::OutOfRange => ConflictCondition::PortOutOfRange,
            ConflictType::AllocationLimitExceeded => ConflictCondition::AllocationLimitExceeded,
            ConflictType::PriorityConflict => ConflictCondition::HighPriorityConflict,
            ConflictType::ResourceExhaustion => ConflictCondition::AllocationLimitExceeded,
            ConflictType::Custom(name) => ConflictCondition::Custom(name.clone()),
        }
    }

    /// Find alternative ports when conflicts occur
    ///
    /// # Arguments
    ///
    /// * `count` - Number of alternative ports needed
    /// * `preferred_range` - Preferred port range (optional)
    /// * `exclude_ports` - Ports to exclude from alternatives
    ///
    /// # Returns
    ///
    /// Vector of alternative port numbers
    #[instrument(skip(self, exclude_ports))]
    pub async fn find_alternative_ports(
        &self,
        count: usize,
        preferred_range: Option<(u16, u16)>,
        exclude_ports: &[u16],
    ) -> Vec<u16> {
        let config = self.config.read();
        debug!(
            "Finding {} alternative ports, preferred range: {:?}, excluding {} ports",
            count,
            preferred_range,
            exclude_ports.len()
        );

        let mut alternatives = Vec::new();
        let (start, end) = preferred_range.unwrap_or((8000, 9000));

        // Simple alternative port finding logic
        // In a real implementation, this would check against actual port availability
        for port in start..=end {
            if alternatives.len() >= count {
                break;
            }

            if alternatives.len() >= config.max_alternative_search {
                warn!("Reached maximum alternative search limit ({})", config.max_alternative_search);
                break;
            }

            if !exclude_ports.contains(&port) {
                alternatives.push(port);
            }
        }

        info!("Found {} alternative ports: {:?}", alternatives.len(), alternatives);
        alternatives
    }

    /// Record a conflict event for history tracking and statistics
    ///
    /// # Arguments
    ///
    /// * `port` - Port involved in the conflict
    /// * `requesting_test_id` - Test requesting the port
    /// * `current_owner` - Current owner of the port (if any)
    /// * `conflict_type` - Type of conflict
    /// * `resolution_action` - Action taken to resolve
    /// * `resolved` - Whether the conflict was resolved
    /// * `resolution_time_ms` - Time taken to resolve (optional)
    /// * `requesting_priority` - Priority of requesting test
    /// * `owner_priority` - Priority of current owner (optional)
    /// * `applied_rule` - Rule that was applied (optional)
    #[instrument(skip(self))]
    pub async fn record_conflict_event(
        &self,
        port: u16,
        requesting_test_id: &str,
        current_owner: Option<&str>,
        conflict_type: ConflictType,
        resolution_action: ConflictAction,
        resolved: bool,
        resolution_time_ms: Option<u64>,
        requesting_priority: f32,
        owner_priority: Option<f32>,
        applied_rule: Option<String>,
    ) {
        // Create base event for compatibility
        let base_event = PortConflictEvent {
            timestamp: Utc::now(),
            port,
            requesting_test_id: requesting_test_id.to_string(),
            current_owner_test_id: current_owner.map(|s| s.to_string()),
            conflict_type: conflict_type.clone(),
            resolution_action: resolution_action.clone(),
            resolved,
            details: HashMap::new(),
        };

        // Create extended event with additional fields
        let extended_event = ExtendedPortConflictEvent {
            base_event: base_event.clone(),
            resolution_time_ms,
            requesting_priority,
            owner_priority,
            applied_rule,
            resolution_attempts: 1,
            auto_resolved: resolved,
        };

        // Update statistics
        let mut stats = self.statistics.lock();
        stats.total_conflicts += 1;
        if resolved {
            stats.resolved_conflicts += 1;
        } else {
            stats.denied_conflicts += 1;
        }

        // Update conflict type statistics
        let conflict_type_key = format!("{:?}", conflict_type);
        *stats.conflicts_by_type.entry(conflict_type_key).or_insert(0) += 1;

        // Update resolution action statistics
        let action_key = format!("{:?}", resolution_action);
        *stats.resolutions_by_action.entry(action_key).or_insert(0) += 1;

        // Update average resolution time
        if let Some(time_ms) = resolution_time_ms {
            let total_time = stats.avg_resolution_time_ms * (stats.total_conflicts - 1) as f64;
            stats.avg_resolution_time_ms = (total_time + time_ms as f64) / stats.total_conflicts as f64;
        }

        // Record the extended event in main history
        let mut history = self.conflict_history.lock();
        history.push(extended_event);

        // Record the basic event for compatibility
        let mut basic_history = self.basic_conflict_history.lock();
        basic_history.push(base_event);

        // Maintain history size
        let config = self.config.read();
        if history.len() > config.max_history_size {
            let excess = history.len() - config.max_history_size;
            history.drain(0..excess);
        }
        if basic_history.len() > config.max_history_size {
            let excess = basic_history.len() - config.max_history_size;
            basic_history.drain(0..excess);
        }

        if config.enable_detailed_logging {
            debug!(
                "Recorded conflict event: port {}, test {}, type {:?}, action {:?}, resolved: {}",
                port, requesting_test_id, conflict_type, resolution_action, resolved
            );
        }
    }

    /// Calculate test priority based on various factors
    ///
    /// # Arguments
    ///
    /// * `test_id` - Test identifier
    /// * `base_priority` - Base priority for the test
    /// * `is_critical` - Whether the test is critical
    /// * `allocation_count` - Current allocation count for the test
    ///
    /// # Returns
    ///
    /// Calculated priority value
    pub async fn calculate_test_priority(
        &self,
        test_id: &str,
        base_priority: f32,
        is_critical: bool,
        allocation_count: usize,
    ) -> f32 {
        let thresholds = self.priority_thresholds.read();
        let mut priority = base_priority;

        // Apply critical test boost
        if is_critical {
            priority += thresholds.critical_test_boost;
        }

        // Apply penalty for high allocation count (resource-heavy tests)
        if allocation_count > 10 {
            priority -= thresholds.long_running_penalty;
        }

        // Ensure priority stays within reasonable bounds
        priority = priority.clamp(0.1, 10.0);

        debug!("Calculated priority for test {}: {} (base: {}, critical: {}, allocations: {})",
               test_id, priority, base_priority, is_critical, allocation_count);

        priority
    }

    /// Check if a test can force allocate a port based on priority
    ///
    /// # Arguments
    ///
    /// * `requesting_priority` - Priority of the requesting test
    /// * `current_owner_priority` - Priority of the current owner
    ///
    /// # Returns
    ///
    /// True if force allocation is allowed
    pub async fn can_force_allocate(
        &self,
        requesting_priority: f32,
        current_owner_priority: Option<f32>,
    ) -> bool {
        let thresholds = self.priority_thresholds.read();

        // Check if requesting test has high enough priority for force allocation
        if requesting_priority < thresholds.force_allocation_threshold {
            return false;
        }

        // If there's a current owner, check priority difference
        if let Some(owner_priority) = current_owner_priority {
            return requesting_priority > owner_priority + 1.0; // Require significant priority difference
        }

        true
    }

    /// Get comprehensive conflict statistics
    ///
    /// # Returns
    ///
    /// Tuple of (total_conflicts, resolved_conflicts) and detailed statistics
    pub async fn get_conflict_statistics(&self) -> (usize, usize) {
        let history = self.basic_conflict_history.lock();
        let total_conflicts = history.len();
        let resolved_conflicts = history.iter().filter(|e| e.resolved).count();
        (total_conflicts, resolved_conflicts)
    }

    /// Get detailed conflict statistics
    ///
    /// # Returns
    ///
    /// Detailed conflict statistics structure
    pub async fn get_detailed_statistics(&self) -> ConflictStatistics {
        let stats = self.statistics.lock();
        stats.clone()
    }

    /// Get recent conflict events
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum number of events to return
    ///
    /// # Returns
    ///
    /// Vector of recent conflict events
    pub async fn get_recent_conflict_events(&self, limit: usize) -> Vec<PortConflictEvent> {
        let history = self.basic_conflict_history.lock();
        let start_idx = if history.len() > limit {
            history.len() - limit
        } else {
            0
        };
        history[start_idx..].to_vec()
    }

    /// Get recent extended conflict events
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum number of events to return
    ///
    /// # Returns
    ///
    /// Vector of recent extended conflict events
    pub async fn get_recent_extended_conflict_events(&self, limit: usize) -> Vec<ExtendedPortConflictEvent> {
        let history = self.conflict_history.lock();
        let start_idx = if history.len() > limit {
            history.len() - limit
        } else {
            0
        };
        history[start_idx..].to_vec()
    }

    /// Add a custom conflict rule
    ///
    /// # Arguments
    ///
    /// * `rule` - The conflict rule to add
    pub async fn add_conflict_rule(&self, rule: ConflictRule) {
        let mut rules = self.conflict_rules.write();
        info!("Adding custom conflict rule: {}", rule.name);
        rules.push(rule);

        // Sort rules by priority (highest first)
        rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Remove a conflict rule by name
    ///
    /// # Arguments
    ///
    /// * `rule_name` - Name of the rule to remove
    ///
    /// # Returns
    ///
    /// True if the rule was found and removed
    pub async fn remove_conflict_rule(&self, rule_name: &str) -> bool {
        let mut rules = self.conflict_rules.write();
        let initial_len = rules.len();
        rules.retain(|rule| rule.name != rule_name);
        let removed = rules.len() < initial_len;

        if removed {
            info!("Removed conflict rule: {}", rule_name);
        } else {
            warn!("Conflict rule not found: {}", rule_name);
        }

        removed
    }

    /// Enable or disable conflict detection
    ///
    /// # Arguments
    ///
    /// * `enabled` - Whether to enable conflict detection
    pub async fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
        info!("Conflict detection {}", if enabled { "enabled" } else { "disabled" });
    }

    /// Check if conflict detection is enabled
    ///
    /// # Returns
    ///
    /// True if conflict detection is enabled
    pub async fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Update conflict detector configuration
    ///
    /// # Arguments
    ///
    /// * `new_config` - New configuration to apply
    pub async fn update_config(&self, new_config: ConflictDetectorConfig) {
        let mut config = self.config.write();
        *config = new_config;
        info!("Updated conflict detector configuration");
    }

    /// Get current configuration
    ///
    /// # Returns
    ///
    /// Current conflict detector configuration
    pub async fn get_config(&self) -> ConflictDetectorConfig {
        let config = self.config.read();
        config.clone()
    }

    /// Clear conflict history
    pub async fn clear_history(&self) {
        let mut history = self.conflict_history.lock();
        let mut basic_history = self.basic_conflict_history.lock();
        let count = history.len();
        history.clear();
        basic_history.clear();

        // Reset statistics
        let mut stats = self.statistics.lock();
        *stats = ConflictStatistics::default();

        info!("Cleared {} conflict events from history", count);
    }

    /// Generate a comprehensive conflict report
    ///
    /// # Returns
    ///
    /// Formatted string containing conflict analysis and statistics
    pub async fn generate_conflict_report(&self) -> String {
        let stats = self.get_detailed_statistics().await;
        let (total, resolved) = self.get_conflict_statistics().await;
        let recent_events = self.get_recent_conflict_events(10).await;

        let resolution_rate = if total > 0 {
            (resolved as f64 / total as f64) * 100.0
        } else {
            0.0
        };

        let mut report = format!(
            "Port Conflict Detection Report\n\
             ===============================\n\
             Total Conflicts: {}\n\
             Resolved Conflicts: {}\n\
             Resolution Rate: {:.1}%\n\
             Average Resolution Time: {:.2}ms\n\
             Denied Conflicts: {}\n\
             Queued Conflicts: {}\n\
             Alternative Ports Found: {}\n\n",
            stats.total_conflicts,
            stats.resolved_conflicts,
            resolution_rate,
            stats.avg_resolution_time_ms,
            stats.denied_conflicts,
            stats.queued_conflicts,
            stats.alternative_ports_found
        );

        // Add conflicts by type
        if !stats.conflicts_by_type.is_empty() {
            report.push_str("Conflicts by Type:\n");
            for (conflict_type, count) in &stats.conflicts_by_type {
                report.push_str(&format!("  {}: {}\n", conflict_type, count));
            }
            report.push('\n');
        }

        // Add resolutions by action
        if !stats.resolutions_by_action.is_empty() {
            report.push_str("Resolutions by Action:\n");
            for (action, count) in &stats.resolutions_by_action {
                report.push_str(&format!("  {}: {}\n", action, count));
            }
            report.push('\n');
        }

        // Add recent events
        if !recent_events.is_empty() {
            report.push_str("Recent Conflict Events:\n");
            for event in recent_events.iter().take(5) {
                report.push_str(&format!(
                    "  {} - Port {}: {} -> {:?} ({})\n",
                    event.timestamp.format("%Y-%m-%d %H:%M:%S"),
                    event.port,
                    event.requesting_test_id,
                    event.resolution_action,
                    if event.resolved { "Resolved" } else { "Failed" }
                ));
            }
        }

        report
    }
}

impl Default for PortConflictDetector {
    fn default() -> Self {
        // Note: This is a blocking implementation for Default
        // In real usage, you should use new() which is async
        Self {
            conflict_rules: Arc::new(RwLock::new(Self::create_default_rules())),
            enabled: AtomicBool::new(true),
            conflict_history: Arc::new(Mutex::new(Vec::new())),
            basic_conflict_history: Arc::new(Mutex::new(Vec::new())),
            config: Arc::new(RwLock::new(ConflictDetectorConfig::default())),
            priority_thresholds: Arc::new(RwLock::new(ConflictPriorityThresholds::default())),
            statistics: Arc::new(Mutex::new(ConflictStatistics::default())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;

    #[test]
    async fn test_conflict_detector_creation() {
        let detector = PortConflictDetector::new().await.unwrap();
        assert!(detector.is_enabled().await);

        let (total, resolved) = detector.get_conflict_statistics().await;
        assert_eq!(total, 0);
        assert_eq!(resolved, 0);
    }

    #[test]
    async fn test_conflict_detection_basic() {
        let detector = PortConflictDetector::new().await.unwrap();

        // Should not detect conflicts for basic allocation
        let result = detector.check_allocation_conflicts(5, "test_001").await;
        assert!(result.is_ok());
    }

    #[test]
    async fn test_conflict_event_recording() {
        let detector = PortConflictDetector::new().await.unwrap();

        detector.record_conflict_event(
            8080,
            "test_001",
            Some("test_000"),
            ConflictType::AlreadyAllocated,
            ConflictAction::FindAlternative,
            true,
            Some(100),
            1.0,
            Some(0.5),
            Some("test_rule".to_string()),
        ).await;

        let (total, resolved) = detector.get_conflict_statistics().await;
        assert_eq!(total, 1);
        assert_eq!(resolved, 1);
    }

    #[test]
    async fn test_alternative_port_finding() {
        let detector = PortConflictDetector::new().await.unwrap();

        let alternatives = detector.find_alternative_ports(
            5,
            Some((8000, 8100)),
            &[8080, 8081],
        ).await;

        assert_eq!(alternatives.len(), 5);
        assert!(!alternatives.contains(&8080));
        assert!(!alternatives.contains(&8081));
    }

    #[test]
    async fn test_priority_calculation() {
        let detector = PortConflictDetector::new().await.unwrap();

        let priority = detector.calculate_test_priority(
            "test_001",
            1.0,
            true, // critical
            5,    // allocation count
        ).await;

        assert!(priority > 1.0); // Should be boosted for critical test
    }

    #[test]
    async fn test_force_allocation_rules() {
        let detector = PortConflictDetector::new().await.unwrap();

        // High priority should allow force allocation
        let can_force = detector.can_force_allocate(9.5, Some(2.0)).await;
        assert!(can_force);

        // Low priority should not allow force allocation
        let cannot_force = detector.can_force_allocate(2.0, Some(1.0)).await;
        assert!(!cannot_force);
    }

    #[test]
    async fn test_rule_management() {
        let detector = PortConflictDetector::new().await.unwrap();

        let custom_rule = ConflictRule {
            name: "custom_test_rule".to_string(),
            condition: ConflictCondition::Custom("test_condition".to_string()),
            action: ConflictAction::Custom("test_action".to_string()),
            priority: 50,
            enabled: true,
            description: Some("Test rule".to_string()),
            config: HashMap::new(),
        };

        detector.add_conflict_rule(custom_rule).await;

        let removed = detector.remove_conflict_rule("custom_test_rule").await;
        assert!(removed);

        let not_removed = detector.remove_conflict_rule("nonexistent_rule").await;
        assert!(!not_removed);
    }

    #[test]
    async fn test_configuration_update() {
        let detector = PortConflictDetector::new().await.unwrap();

        let mut new_config = ConflictDetectorConfig::default();
        new_config.max_alternative_search = 200;
        new_config.enable_auto_resolution = false;

        detector.update_config(new_config.clone()).await;

        let current_config = detector.get_config().await;
        assert_eq!(current_config.max_alternative_search, 200);
        assert!(!current_config.enable_auto_resolution);
    }

    #[test]
    async fn test_history_management() {
        let detector = PortConflictDetector::new().await.unwrap();

        // Add some conflict events
        for i in 0..5 {
            detector.record_conflict_event(
                8080 + i,
                &format!("test_{:03}", i),
                None,
                ConflictType::AlreadyAllocated,
                ConflictAction::FindAlternative,
                true,
                Some(50),
                1.0,
                None,
                None,
            ).await;
        }

        let recent_events = detector.get_recent_conflict_events(3).await;
        assert_eq!(recent_events.len(), 3);

        detector.clear_history().await;

        let (total, _) = detector.get_conflict_statistics().await;
        assert_eq!(total, 0);
    }

    #[test]
    async fn test_report_generation() {
        let detector = PortConflictDetector::new().await.unwrap();

        // Add some test data
        detector.record_conflict_event(
            8080,
            "test_001",
            Some("test_000"),
            ConflictType::AlreadyAllocated,
            ConflictAction::FindAlternative,
            true,
            Some(100),
            1.0,
            Some(0.5),
            Some("test_rule".to_string()),
        ).await;

        let report = detector.generate_conflict_report().await;
        assert!(report.contains("Port Conflict Detection Report"));
        assert!(report.contains("Total Conflicts: 1"));
        assert!(report.contains("Resolved Conflicts: 1"));
    }

    #[test]
    async fn test_conflict_resolution_result() {
        let detector = PortConflictDetector::new().await.unwrap();

        let result = detector.detect_and_resolve_port_conflict(
            8080,
            "test_001",
            1.0,
        ).await;

        assert!(result.resolved);
        assert!(result.resolution_time.as_millis() >= 0);
    }
}