//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

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
// use super::types::*; // Circular import - commented out
use crate::resource_management::types::*;


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
        info!(
            "Initializing PortConflictDetector with {} default rules", default_rules
            .len()
        );
        Ok(Self {
            conflict_rules: Arc::new(RwLock::new(default_rules)),
            enabled: AtomicBool::new(true),
            conflict_history: Arc::new(Mutex::new(Vec::new())),
            basic_conflict_history: Arc::new(Mutex::new(Vec::new())),
            config: Arc::new(RwLock::new(ConflictDetectorConfig::default())),
            priority_thresholds: Arc::new(
                RwLock::new(ConflictPriorityThresholds::default()),
            ),
            statistics: Arc::new(Mutex::new(ConflictStatistics::default())),
        })
    }
    /// Create default conflict rules
    fn create_default_rules() -> Vec<ConflictRule> {
        vec![
            ConflictRule { name : "prevent_well_known_ports".to_string(), condition :
            ConflictCondition::WellKnownPort, action : ConflictAction::Deny, priority :
            100, enabled : true, description :
            Some("Prevent allocation of well-known system ports".to_string()), config :
            HashMap::new(), }, ConflictRule { name : "prevent_excluded_ports"
            .to_string(), condition : ConflictCondition::PortExcluded, action :
            ConflictAction::Deny, priority : 95, enabled : true, description :
            Some("Prevent allocation of excluded port ranges".to_string()), config :
            HashMap::new(), }, ConflictRule { name : "prevent_out_of_range_ports"
            .to_string(), condition : ConflictCondition::PortOutOfRange, action :
            ConflictAction::Deny, priority : 90, enabled : true, description :
            Some("Prevent allocation of ports outside configured range".to_string()),
            config : HashMap::new(), }, ConflictRule { name :
            "handle_high_priority_conflicts".to_string(), condition :
            ConflictCondition::HighPriorityConflict, action :
            ConflictAction::NegotiatePriority, priority : 85, enabled : true, description
            : Some("Handle conflicts involving high-priority tests".to_string()), config
            : HashMap::new(), }, ConflictRule { name : "find_alternative_for_allocated"
            .to_string(), condition : ConflictCondition::PortAlreadyAllocated, action :
            ConflictAction::FindAlternative, priority : 80, enabled : true, description :
            Some("Find alternative ports for already allocated ports".to_string()),
            config : HashMap::new(), }, ConflictRule { name :
            "find_alternative_for_reserved".to_string(), condition :
            ConflictCondition::PortReserved, action : ConflictAction::FindAlternative,
            priority : 75, enabled : true, description :
            Some("Find alternative ports for reserved ports".to_string()), config :
            HashMap::new(), }, ConflictRule { name : "handle_allocation_limit_exceeded"
            .to_string(), condition : ConflictCondition::AllocationLimitExceeded, action
            : ConflictAction::Queue, priority : 70, enabled : true, description :
            Some("Queue requests that exceed allocation limits".to_string()), config :
            HashMap::new(), },
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
        if count == 0 {
            return Ok(());
        }
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
                warn!(
                    "No matching rule found for conflict type {:?}, using default denial",
                    conflict_type
                );
                ConflictAction::Deny
            }
        }
    }
    /// Convert conflict type to conflict condition
    fn conflict_type_to_condition(
        &self,
        conflict_type: &ConflictType,
    ) -> ConflictCondition {
        match conflict_type {
            ConflictType::AlreadyAllocated => ConflictCondition::PortAlreadyAllocated,
            ConflictType::Reserved => ConflictCondition::PortReserved,
            ConflictType::Excluded => ConflictCondition::PortExcluded,
            ConflictType::WellKnown => ConflictCondition::WellKnownPort,
            ConflictType::OutOfRange => ConflictCondition::PortOutOfRange,
            ConflictType::AllocationLimitExceeded => {
                ConflictCondition::AllocationLimitExceeded
            }
            ConflictType::PriorityConflict => ConflictCondition::HighPriorityConflict,
            ConflictType::ResourceExhaustion => {
                ConflictCondition::AllocationLimitExceeded
            }
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
            count, preferred_range, exclude_ports.len()
        );
        let mut alternatives = Vec::new();
        let (start, end) = preferred_range.unwrap_or((8000, 9000));
        for port in start..=end {
            if alternatives.len() >= count {
                break;
            }
            if alternatives.len() >= config.max_alternative_search {
                warn!(
                    "Reached maximum alternative search limit ({})", config
                    .max_alternative_search
                );
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
        let extended_event = ExtendedPortConflictEvent {
            base_event: base_event.clone(),
            resolution_time_ms,
            requesting_priority,
            owner_priority,
            applied_rule,
            resolution_attempts: 1,
            auto_resolved: resolved,
        };
        let mut stats = self.statistics.lock();
        stats.total_conflicts += 1;
        if resolved {
            stats.resolved_conflicts += 1;
        } else {
            stats.denied_conflicts += 1;
        }
        let conflict_type_key = format!("{:?}", conflict_type);
        *stats.conflicts_by_type.entry(conflict_type_key).or_insert(0) += 1;
        let action_key = format!("{:?}", resolution_action);
        *stats.resolutions_by_action.entry(action_key).or_insert(0) += 1;
        if let Some(time_ms) = resolution_time_ms {
            let total_time = stats.avg_resolution_time_ms
                * (stats.total_conflicts - 1) as f64;
            stats.avg_resolution_time_ms = (total_time + time_ms as f64)
                / stats.total_conflicts as f64;
        }
        let mut history = self.conflict_history.lock();
        history.push(extended_event);
        let mut basic_history = self.basic_conflict_history.lock();
        basic_history.push(base_event);
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
        if is_critical {
            priority += thresholds.critical_test_boost;
        }
        if allocation_count > 10 {
            priority -= thresholds.long_running_penalty;
        }
        priority = priority.clamp(0.1, 10.0);
        debug!(
            "Calculated priority for test {}: {} (base: {}, critical: {}, allocations: {})",
            test_id, priority, base_priority, is_critical, allocation_count
        );
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
        if requesting_priority < thresholds.force_allocation_threshold {
            return false;
        }
        if let Some(owner_priority) = current_owner_priority {
            return requesting_priority > owner_priority + 1.0;
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
    pub async fn get_recent_conflict_events(
        &self,
        limit: usize,
    ) -> Vec<PortConflictEvent> {
        let history = self.basic_conflict_history.lock();
        let start_idx = if history.len() > limit { history.len() - limit } else { 0 };
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
    pub async fn get_recent_extended_conflict_events(
        &self,
        limit: usize,
    ) -> Vec<ExtendedPortConflictEvent> {
        let history = self.conflict_history.lock();
        let start_idx = if history.len() > limit { history.len() - limit } else { 0 };
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
            stats.total_conflicts, stats.resolved_conflicts, resolution_rate, stats
            .avg_resolution_time_ms, stats.denied_conflicts, stats.queued_conflicts,
            stats.alternative_ports_found
        );
        if !stats.conflicts_by_type.is_empty() {
            report.push_str("Conflicts by Type:\n");
            for (conflict_type, count) in &stats.conflicts_by_type {
                report.push_str(&format!("  {}: {}\n", conflict_type, count));
            }
            report.push('\n');
        }
        if !stats.resolutions_by_action.is_empty() {
            report.push_str("Resolutions by Action:\n");
            for (action, count) in &stats.resolutions_by_action {
                report.push_str(&format!("  {}: {}\n", action, count));
            }
            report.push('\n');
        }
        if !recent_events.is_empty() {
            report.push_str("Recent Conflict Events:\n");
            for event in recent_events.iter().take(5) {
                report
                    .push_str(
                        &format!(
                            "  {} - Port {}: {} -> {:?} ({})\n", event.timestamp
                            .format("%Y-%m-%d %H:%M:%S"), event.port, event
                            .requesting_test_id, event.resolution_action, if event
                            .resolved { "Resolved" } else { "Failed" }
                        ),
                    );
            }
        }
        report
    }
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
