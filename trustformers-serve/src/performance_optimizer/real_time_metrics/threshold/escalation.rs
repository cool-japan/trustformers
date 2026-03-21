//! Escalation management system for alert severity progression.

use super::super::types::*;
use super::error::{Result, ThresholdError};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use tokio::sync::Mutex as TokioMutex;
use tokio::time::interval;
use tracing::info;

// =============================================================================
// ESCALATION MANAGEMENT SYSTEM
// =============================================================================

/// Escalation manager for alert escalation
///
/// Advanced escalation manager that handles multi-level alert escalation
/// with time-based triggers, severity-based routing, and customizable policies.
pub struct EscalationManager {
    /// Escalation policies
    policies: Arc<RwLock<HashMap<String, EscalationPolicy>>>,

    /// Escalation state tracking
    state: Arc<TokioMutex<HashMap<String, EscalationState>>>,

    /// Escalation statistics
    stats: Arc<EscalationStats>,

    /// Configuration
    config: Arc<RwLock<EscalationConfig>>,

    /// Processing task handle
    processing_handle: Arc<TokioMutex<Option<tokio::task::JoinHandle<()>>>>,

    /// Shutdown signal
    shutdown_signal: Arc<AtomicBool>,
}

/// Escalation policy configuration
#[derive(Debug, Clone)]
pub struct EscalationPolicy {
    /// Policy name
    pub name: String,

    /// Escalation levels
    pub levels: Vec<EscalationLevel>,

    /// Maximum escalation level
    pub max_level: u8,

    /// Auto-escalation enabled
    pub auto_escalation: bool,

    /// Escalation triggers
    pub triggers: Vec<EscalationTrigger>,

    /// Policy enabled
    pub enabled: bool,
}

/// Escalation level configuration
#[derive(Debug, Clone)]
pub struct EscalationLevel {
    /// Level number
    pub level: u8,

    /// Time before escalation
    pub time_to_escalate: Duration,

    /// Recipients at this level
    pub recipients: Vec<String>,

    /// Notification channels
    pub channels: Vec<String>,

    /// Actions to take
    pub actions: Vec<EscalationAction>,
}

/// Current escalation state
#[derive(Debug, Clone)]
pub struct EscalationState {
    /// Alert ID being escalated
    pub alert_id: String,

    /// Current escalation level
    pub current_level: u8,

    /// Escalation start time
    pub started_at: DateTime<Utc>,

    /// Next escalation time
    pub next_escalation: DateTime<Utc>,

    /// Escalation history
    pub escalation_history: Vec<EscalationEvent>,

    /// Acknowledged
    pub acknowledged: bool,

    /// Acknowledgment time
    pub acknowledged_at: Option<DateTime<Utc>>,
}

/// Escalation trigger conditions
#[derive(Debug, Clone)]
pub enum EscalationTrigger {
    /// Time-based escalation
    TimeBasedDuration(Duration),

    /// Severity-based escalation
    SeverityLevel(SeverityLevel),

    /// Repeat alert count
    RepeatCount(u32),

    /// No acknowledgment
    NoAcknowledgment(Duration),

    /// Custom trigger
    Custom(String),
}

/// Actions to take during escalation
#[derive(Debug, Clone)]
pub enum EscalationAction {
    /// Send notification
    Notify {
        channel: String,
        recipients: Vec<String>,
    },

    /// Execute webhook
    Webhook {
        url: String,
        payload: HashMap<String, String>,
    },

    /// Create incident
    CreateIncident { severity: String, assignee: String },

    /// Auto-remediation
    AutoRemediate {
        script: String,
        parameters: HashMap<String, String>,
    },

    /// Custom action
    Custom(String),
}

/// Escalation event record
#[derive(Debug, Clone)]
pub struct EscalationEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,

    /// Event type
    pub event_type: EscalationEventType,

    /// Event level
    pub level: u8,

    /// Event details
    pub details: HashMap<String, String>,
}

/// Types of escalation events
#[derive(Debug, Clone)]
pub enum EscalationEventType {
    /// Escalation started
    Started,

    /// Escalation level increased
    LevelIncreased,

    /// Escalation acknowledged
    Acknowledged,

    /// Escalation resolved
    Resolved,

    /// Escalation failed
    Failed,

    /// Manual escalation
    Manual,
}

/// Statistics for escalation management
#[derive(Debug, Default)]
pub struct EscalationStats {
    /// Total escalations
    pub total_escalations: AtomicU64,

    /// Active escalations
    pub active_escalations: AtomicU64,

    /// Escalations by level
    pub escalations_by_level: Arc<Mutex<HashMap<u8, u64>>>,

    /// Average escalation time
    pub avg_escalation_time: Arc<Mutex<Duration>>,

    /// Acknowledgment rate
    pub acknowledgment_rate: Arc<Mutex<f32>>,
}

/// Configuration for escalation management
#[derive(Debug, Clone)]
pub struct EscalationConfig {
    /// Enable auto-escalation
    pub enable_auto_escalation: bool,

    /// Default escalation timeout
    pub default_escalation_timeout: Duration,

    /// Maximum escalation levels
    pub max_escalation_levels: u8,

    /// Escalation check interval
    pub check_interval: Duration,

    /// Enable escalation notifications
    pub enable_notifications: bool,
}

impl EscalationManager {
    /// Create a new escalation manager
    pub async fn new() -> Result<Self> {
        let manager = Self {
            policies: Arc::new(RwLock::new(HashMap::new())),
            state: Arc::new(TokioMutex::new(HashMap::new())),
            stats: Arc::new(EscalationStats::default()),
            config: Arc::new(RwLock::new(EscalationConfig::default())),
            processing_handle: Arc::new(TokioMutex::new(None)),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
        };

        manager.initialize_default_policies().await?;
        Ok(manager)
    }

    /// Start escalation manager
    pub async fn start(&self) -> Result<()> {
        let mut handle = self.processing_handle.lock().await;

        if handle.is_some() {
            return Err(ThresholdError::EscalationError(
                "Escalation manager already started".to_string(),
            ));
        }

        let state = Arc::clone(&self.state);
        let policies = Arc::clone(&self.policies);
        let stats = Arc::clone(&self.stats);
        let config = Arc::clone(&self.config);
        let shutdown_signal = Arc::clone(&self.shutdown_signal);

        let escalation_task = tokio::spawn(async move {
            Self::escalation_loop(state, policies, stats, config, shutdown_signal).await;
        });

        *handle = Some(escalation_task);
        info!("Escalation manager started");
        Ok(())
    }

    /// Stop escalation manager
    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown_signal.store(true, Ordering::Relaxed);

        let mut handle = self.processing_handle.lock().await;
        if let Some(task) = handle.take() {
            task.abort();
            info!("Escalation manager stopped");
        }

        Ok(())
    }

    /// Start escalation for an alert
    pub async fn start_escalation(&self, alert: &AlertEvent, policy_name: &str) -> Result<()> {
        let policies = self.policies.read().expect("Policies RwLock poisoned");
        let policy = policies.get(policy_name).ok_or_else(|| {
            ThresholdError::EscalationError(format!("Policy {} not found", policy_name))
        })?;

        if !policy.enabled {
            return Ok(());
        }

        let mut state = self.state.lock().await;

        // Check if escalation already exists
        if state.contains_key(&alert.alert_id) {
            return Ok(());
        }

        let escalation_state = EscalationState {
            alert_id: alert.alert_id.clone(),
            current_level: 0,
            started_at: Utc::now(),
            next_escalation: Utc::now()
                + chrono::Duration::from_std(policy.levels[0].time_to_escalate)
                    .expect("Duration conversion failed"),
            escalation_history: vec![EscalationEvent {
                timestamp: Utc::now(),
                event_type: EscalationEventType::Started,
                level: 0,
                details: HashMap::new(),
            }],
            acknowledged: false,
            acknowledged_at: None,
        };

        state.insert(alert.alert_id.clone(), escalation_state);
        self.stats.total_escalations.fetch_add(1, Ordering::Relaxed);
        self.stats.active_escalations.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Acknowledge escalation
    pub async fn acknowledge_escalation(&self, alert_id: &str, acknowledger: &str) -> Result<()> {
        let mut state = self.state.lock().await;

        if let Some(escalation_state) = state.get_mut(alert_id) {
            escalation_state.acknowledged = true;
            escalation_state.acknowledged_at = Some(Utc::now());
            escalation_state.escalation_history.push(EscalationEvent {
                timestamp: Utc::now(),
                event_type: EscalationEventType::Acknowledged,
                level: escalation_state.current_level,
                details: {
                    let mut details = HashMap::new();
                    details.insert("acknowledger".to_string(), acknowledger.to_string());
                    details
                },
            });

            self.stats.active_escalations.fetch_sub(1, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Resolve escalation
    pub async fn resolve_escalation(&self, alert_id: &str) -> Result<()> {
        let mut state = self.state.lock().await;

        if let Some(mut escalation_state) = state.remove(alert_id) {
            escalation_state.escalation_history.push(EscalationEvent {
                timestamp: Utc::now(),
                event_type: EscalationEventType::Resolved,
                level: escalation_state.current_level,
                details: HashMap::new(),
            });

            self.stats.active_escalations.fetch_sub(1, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Main escalation processing loop
    async fn escalation_loop(
        state: Arc<TokioMutex<HashMap<String, EscalationState>>>,
        policies: Arc<RwLock<HashMap<String, EscalationPolicy>>>,
        stats: Arc<EscalationStats>,
        config: Arc<RwLock<EscalationConfig>>,
        shutdown_signal: Arc<AtomicBool>,
    ) {
        let mut interval = interval(Duration::from_secs(30)); // Check every 30 seconds

        while !shutdown_signal.load(Ordering::Relaxed) {
            interval.tick().await;

            let mut state_guard = state.lock().await;
            let auto_escalation_enabled = {
                let config_read = config.read().expect("Config RwLock poisoned");
                config_read.enable_auto_escalation
            };

            if !auto_escalation_enabled {
                continue;
            }

            let now = Utc::now();
            let mut escalations_to_process = Vec::new();

            // Find escalations that need processing
            for (alert_id, escalation_state) in state_guard.iter() {
                if !escalation_state.acknowledged && now >= escalation_state.next_escalation {
                    escalations_to_process.push(alert_id.clone());
                }
            }

            // Process escalations
            for alert_id in escalations_to_process {
                if let Some(escalation_state) = state_guard.get_mut(&alert_id) {
                    Self::process_escalation(escalation_state, &policies, &stats);
                }
            }
        }
    }

    /// Process a single escalation
    fn process_escalation(
        escalation_state: &mut EscalationState,
        policies: &Arc<RwLock<HashMap<String, EscalationPolicy>>>,
        stats: &Arc<EscalationStats>,
    ) {
        let policies_guard = policies.read().expect("Policies RwLock poisoned");

        // Find appropriate policy (simplified - in practice, this would be more sophisticated)
        if let Some(policy) = policies_guard.values().next() {
            if escalation_state.current_level < policy.max_level
                && escalation_state.current_level < policy.levels.len() as u8
            {
                escalation_state.current_level += 1;
                let level_index = escalation_state.current_level as usize;

                if level_index < policy.levels.len() {
                    let level = &policy.levels[level_index];

                    // Schedule next escalation
                    escalation_state.next_escalation = Utc::now()
                        + chrono::Duration::from_std(level.time_to_escalate)
                            .expect("Duration conversion failed");

                    // Record escalation event
                    escalation_state.escalation_history.push(EscalationEvent {
                        timestamp: Utc::now(),
                        event_type: EscalationEventType::LevelIncreased,
                        level: escalation_state.current_level,
                        details: HashMap::new(),
                    });

                    // Execute escalation actions
                    for action in &level.actions {
                        Self::execute_escalation_action(action);
                    }

                    // Update statistics
                    let mut level_stats =
                        stats.escalations_by_level.lock().expect("Escalation stats lock poisoned");
                    *level_stats.entry(escalation_state.current_level).or_insert(0) += 1;
                }
            }
        }
    }

    /// Execute escalation action
    fn execute_escalation_action(action: &EscalationAction) {
        match action {
            EscalationAction::Notify {
                channel,
                recipients,
            } => {
                info!(
                    "Sending escalation notification via {} to {:?}",
                    channel, recipients
                );
                // Implementation would send actual notifications
            },
            EscalationAction::Webhook { url, payload } => {
                info!("Executing webhook {} with payload {:?}", url, payload);
                // Implementation would make HTTP request
            },
            EscalationAction::CreateIncident { severity, assignee } => {
                info!(
                    "Creating incident with severity {} assigned to {}",
                    severity, assignee
                );
                // Implementation would create incident in ticketing system
            },
            EscalationAction::AutoRemediate { script, parameters } => {
                info!(
                    "Executing auto-remediation script {} with parameters {:?}",
                    script, parameters
                );
                // Implementation would execute remediation script
            },
            EscalationAction::Custom(action) => {
                info!("Executing custom escalation action: {}", action);
                // Implementation would handle custom action
            },
        }
    }

    /// Initialize default escalation policies
    async fn initialize_default_policies(&self) -> Result<()> {
        let default_policy = EscalationPolicy {
            name: "default".to_string(),
            levels: vec![
                EscalationLevel {
                    level: 1,
                    time_to_escalate: Duration::from_secs(300), // 5 minutes
                    recipients: vec!["team-lead@company.com".to_string()],
                    channels: vec!["email".to_string()],
                    actions: vec![EscalationAction::Notify {
                        channel: "email".to_string(),
                        recipients: vec!["team-lead@company.com".to_string()],
                    }],
                },
                EscalationLevel {
                    level: 2,
                    time_to_escalate: Duration::from_secs(900), // 15 minutes
                    recipients: vec!["manager@company.com".to_string()],
                    channels: vec!["email".to_string(), "slack".to_string()],
                    actions: vec![
                        EscalationAction::Notify {
                            channel: "email".to_string(),
                            recipients: vec!["manager@company.com".to_string()],
                        },
                        EscalationAction::CreateIncident {
                            severity: "high".to_string(),
                            assignee: "on-call-engineer".to_string(),
                        },
                    ],
                },
            ],
            max_level: 2,
            auto_escalation: true,
            triggers: vec![
                EscalationTrigger::TimeBasedDuration(Duration::from_secs(300)),
                EscalationTrigger::NoAcknowledgment(Duration::from_secs(600)),
            ],
            enabled: true,
        };

        let performance_policy = EscalationPolicy {
            name: "performance".to_string(),
            levels: vec![EscalationLevel {
                level: 1,
                time_to_escalate: Duration::from_secs(180), // 3 minutes
                recipients: vec!["performance-team@company.com".to_string()],
                channels: vec!["slack".to_string()],
                actions: vec![EscalationAction::Notify {
                    channel: "slack".to_string(),
                    recipients: vec!["performance-team@company.com".to_string()],
                }],
            }],
            max_level: 1,
            auto_escalation: true,
            triggers: vec![EscalationTrigger::SeverityLevel(SeverityLevel::Critical)],
            enabled: true,
        };

        let mut policies = self.policies.write().expect("Policies RwLock poisoned");
        policies.insert(default_policy.name.clone(), default_policy);
        policies.insert(performance_policy.name.clone(), performance_policy);

        Ok(())
    }

    /// Add escalation policy
    pub fn add_policy(&self, policy: EscalationPolicy) {
        let mut policies = self.policies.write().expect("Policies RwLock poisoned");
        policies.insert(policy.name.clone(), policy);
    }

    /// Remove escalation policy
    pub fn remove_policy(&self, policy_name: &str) {
        let mut policies = self.policies.write().expect("Policies RwLock poisoned");
        policies.remove(policy_name);
    }

    /// Get escalation statistics
    pub fn get_stats(&self) -> EscalationStats {
        EscalationStats {
            total_escalations: AtomicU64::new(self.stats.total_escalations.load(Ordering::Relaxed)),
            active_escalations: AtomicU64::new(
                self.stats.active_escalations.load(Ordering::Relaxed),
            ),
            escalations_by_level: Arc::new(Mutex::new(
                self.stats
                    .escalations_by_level
                    .lock()
                    .expect("Escalation stats lock poisoned")
                    .clone(),
            )),
            avg_escalation_time: Arc::new(Mutex::new(
                *self
                    .stats
                    .avg_escalation_time
                    .lock()
                    .expect("Avg escalation time lock poisoned"),
            )),
            acknowledgment_rate: Arc::new(Mutex::new(
                *self
                    .stats
                    .acknowledgment_rate
                    .lock()
                    .expect("Acknowledgment rate lock poisoned"),
            )),
        }
    }

    /// Get escalation state
    pub async fn get_escalation_state(&self, alert_id: &str) -> Option<EscalationState> {
        let state = self.state.lock().await;
        state.get(alert_id).cloned()
    }
}

impl Default for EscalationConfig {
    fn default() -> Self {
        Self {
            enable_auto_escalation: true,
            default_escalation_timeout: Duration::from_secs(300), // 5 minutes
            max_escalation_levels: 3,
            check_interval: Duration::from_secs(30),
            enable_notifications: true,
        }
    }
}
