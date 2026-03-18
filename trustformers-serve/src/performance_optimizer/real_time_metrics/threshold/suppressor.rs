//! Alert suppression system for reducing alert noise.

use super::super::types::*;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use tokio::sync::Mutex as TokioMutex;

// =============================================================================
// ALERT SUPPRESSION SYSTEM
// =============================================================================

/// Alert suppression system for reducing alert noise
///
/// Advanced alert suppression system that reduces alert noise through
/// deduplication, frequency limiting, and intelligent suppression policies.
pub struct AlertSuppressor {
    /// Suppression rules
    rules: Arc<RwLock<Vec<SuppressionRule>>>,

    /// Alert fingerprints for deduplication
    fingerprints: Arc<TokioMutex<HashMap<String, AlertFingerprint>>>,

    /// Suppression statistics
    stats: Arc<SuppressionStats>,

    /// Configuration
    config: Arc<RwLock<SuppressionConfig>>,
}

/// Suppression rule for alerts
#[derive(Debug, Clone)]
pub struct SuppressionRule {
    /// Rule ID
    pub id: String,

    /// Rule name
    pub name: String,

    /// Rule type
    pub rule_type: SuppressionRuleType,

    /// Matching criteria
    pub criteria: SuppressionCriteria,

    /// Suppression action
    pub action: SuppressionAction,

    /// Rule priority
    pub priority: u8,

    /// Rule enabled
    pub enabled: bool,
}

/// Types of suppression rules
#[derive(Debug, Clone)]
pub enum SuppressionRuleType {
    /// Frequency-based suppression
    Frequency,

    /// Duplicate suppression
    Duplicate,

    /// Time-based suppression
    TimeBased,

    /// Pattern-based suppression
    Pattern,

    /// Severity-based suppression
    Severity,

    /// Custom suppression logic
    Custom(String),
}

/// Criteria for alert suppression
#[derive(Debug, Clone)]
pub struct SuppressionCriteria {
    /// Metric patterns to match
    pub metric_patterns: Vec<String>,

    /// Severity levels to match
    pub severity_levels: Vec<SeverityLevel>,

    /// Time window for suppression
    pub time_window: Duration,

    /// Maximum alerts per window
    pub max_alerts_per_window: u32,

    /// Threshold patterns to match
    pub threshold_patterns: Vec<String>,

    /// Additional conditions
    pub conditions: HashMap<String, String>,
}

/// Suppression action to take
#[derive(Debug, Clone)]
pub enum SuppressionAction {
    /// Suppress completely
    Suppress,

    /// Reduce frequency
    ReduceFrequency { factor: f32 },

    /// Aggregate alerts
    Aggregate { window: Duration },

    /// Downgrade severity
    DowngradeSeverity { levels: u8 },

    /// Route to different channel
    Reroute { channel: String },
}

/// Alert fingerprint for deduplication
#[derive(Debug, Clone)]
pub struct AlertFingerprint {
    /// Fingerprint hash
    pub hash: String,

    /// First occurrence
    pub first_seen: DateTime<Utc>,

    /// Last occurrence
    pub last_seen: DateTime<Utc>,

    /// Occurrence count
    pub count: u32,

    /// Suppression status
    pub suppressed: bool,

    /// Associated alert IDs
    pub alert_ids: Vec<String>,
}

/// Statistics for alert suppression
#[derive(Debug, Default)]
pub struct SuppressionStats {
    /// Total alerts processed
    pub total_processed: AtomicU64,

    /// Total alerts suppressed
    pub total_suppressed: AtomicU64,

    /// Suppression by rule type
    pub suppression_by_rule: Arc<Mutex<HashMap<String, u64>>>,

    /// Average suppression rate
    pub avg_suppression_rate: Arc<Mutex<f32>>,

    /// Peak suppression rate
    pub peak_suppression_rate: Arc<Mutex<f32>>,
}

/// Configuration for alert suppression
#[derive(Debug, Clone)]
pub struct SuppressionConfig {
    /// Enable alert deduplication
    pub enable_deduplication: bool,

    /// Deduplication window
    pub deduplication_window: Duration,

    /// Maximum fingerprint cache size
    pub max_fingerprint_cache: usize,

    /// Default suppression window
    pub default_suppression_window: Duration,

    /// Enable adaptive suppression
    pub enable_adaptive_suppression: bool,
}

impl Default for AlertSuppressor {
    fn default() -> Self {
        Self::new()
    }
}

impl AlertSuppressor {
    /// Create a new alert suppressor
    pub fn new() -> Self {
        Self {
            rules: Arc::new(RwLock::new(Vec::new())),
            fingerprints: Arc::new(TokioMutex::new(HashMap::new())),
            stats: Arc::new(SuppressionStats::default()),
            config: Arc::new(RwLock::new(SuppressionConfig::default())),
        }
    }

    /// Check if alert should be suppressed
    pub async fn should_suppress(&self, alert: &AlertEvent) -> bool {
        self.stats.total_processed.fetch_add(1, Ordering::Relaxed);

        let suppressed_by_rule = {
            let rules = self.rules.read().expect("Rules RwLock poisoned");
            let mut suppressed = false;

            for rule in rules.iter() {
                if !rule.enabled {
                    continue;
                }

                if self.matches_criteria(&rule.criteria, alert)
                    && matches!(rule.action, SuppressionAction::Suppress)
                {
                    self.stats.total_suppressed.fetch_add(1, Ordering::Relaxed);
                    self.update_rule_stats(&rule.id);
                    suppressed = true;
                    break;
                }
            }

            suppressed
        };

        if suppressed_by_rule {
            return true;
        }

        // Check deduplication
        let enable_deduplication = {
            let config = self.config.read().expect("Config RwLock poisoned");
            config.enable_deduplication
        };

        if enable_deduplication && self.is_duplicate(alert).await {
            self.stats.total_suppressed.fetch_add(1, Ordering::Relaxed);
            return true;
        }

        false
    }

    /// Suppress an alert
    pub async fn suppress_alert(&self, alert: &mut AlertEvent) {
        alert.suppression_info = Some(SuppressionInfo {
            reason: "Suppressed by alert suppressor".to_string(),
            start_time: Utc::now(), // Changed from suppressed_at to match types.rs
            duration: Duration::from_secs(300), // 5 minutes default
            suppressed_count: 1,
        });

        // Update fingerprint if deduplication is enabled
        let enable_deduplication = {
            let config = self.config.read().expect("Config RwLock poisoned");
            config.enable_deduplication
        };

        if enable_deduplication {
            self.update_fingerprint(alert).await;
        }
    }

    /// Check if alert matches criteria
    fn matches_criteria(&self, criteria: &SuppressionCriteria, alert: &AlertEvent) -> bool {
        // Check metric patterns
        if !criteria.metric_patterns.is_empty() {
            let metric_matches = criteria
                .metric_patterns
                .iter()
                .any(|pattern| alert.threshold.metric.contains(pattern));
            if !metric_matches {
                return false;
            }
        }

        // Check severity levels
        if !criteria.severity_levels.is_empty()
            && !criteria.severity_levels.contains(&alert.severity)
        {
            return false;
        }

        // Check threshold patterns
        if !criteria.threshold_patterns.is_empty() {
            let threshold_matches = criteria
                .threshold_patterns
                .iter()
                .any(|pattern| alert.threshold.name.contains(pattern));
            if !threshold_matches {
                return false;
            }
        }

        // Check additional conditions
        for (key, expected_value) in &criteria.conditions {
            if let Some(actual_value) = alert.context.get(key) {
                if actual_value != expected_value {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }

    /// Check if alert is a duplicate
    async fn is_duplicate(&self, alert: &AlertEvent) -> bool {
        let fingerprint_hash = self.calculate_fingerprint(alert);
        let mut fingerprints = self.fingerprints.lock().await;

        if let Some(existing) = fingerprints.get_mut(&fingerprint_hash) {
            existing.last_seen = Utc::now();
            existing.count += 1;
            existing.alert_ids.push(alert.alert_id.clone());

            // Check if within deduplication window
            let time_diff = existing.last_seen.signed_duration_since(existing.first_seen);
            let dedup_window = chrono::Duration::from_std(
                self.config.read().expect("Config RwLock poisoned").deduplication_window,
            )
            .expect("Duration conversion failed");

            return time_diff <= dedup_window;
        }

        // Create new fingerprint
        let fingerprint = AlertFingerprint {
            hash: fingerprint_hash.clone(),
            first_seen: Utc::now(),
            last_seen: Utc::now(),
            count: 1,
            suppressed: false,
            alert_ids: vec![alert.alert_id.clone()],
        };

        fingerprints.insert(fingerprint_hash, fingerprint);

        // Maintain cache size
        let max_cache_size =
            self.config.read().expect("Config RwLock poisoned").max_fingerprint_cache;
        if fingerprints.len() > max_cache_size {
            // Remove oldest entries - collect keys first to avoid borrow conflict
            let mut entries: Vec<_> =
                fingerprints.iter().map(|(k, v)| (k.clone(), v.first_seen)).collect();
            entries.sort_by_key(|(_, first_seen)| *first_seen);

            let to_remove = entries.len() - max_cache_size;
            let keys_to_remove: Vec<_> =
                entries.iter().take(to_remove).map(|(k, _)| k.clone()).collect();

            for hash in keys_to_remove {
                fingerprints.remove(&hash);
            }
        }

        false
    }

    /// Calculate alert fingerprint
    fn calculate_fingerprint(&self, alert: &AlertEvent) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        alert.threshold.metric.hash(&mut hasher);
        alert.threshold.name.hash(&mut hasher);
        alert.severity.hash(&mut hasher);

        // Include relevant context in fingerprint
        for (key, value) in &alert.context {
            if key != "timestamp" && key != "alert_id" {
                key.hash(&mut hasher);
                value.hash(&mut hasher);
            }
        }

        format!("{:x}", hasher.finish())
    }

    /// Update fingerprint for alert
    async fn update_fingerprint(&self, alert: &AlertEvent) {
        let fingerprint_hash = self.calculate_fingerprint(alert);
        let mut fingerprints = self.fingerprints.lock().await;

        if let Some(fingerprint) = fingerprints.get_mut(&fingerprint_hash) {
            fingerprint.suppressed = true;
            if let Some(ref suppression_info) = alert.suppression_info {
                fingerprint.count = suppression_info.suppressed_count;
            }
        }
    }

    /// Update rule statistics
    fn update_rule_stats(&self, rule_id: &str) {
        let mut stats =
            self.stats.suppression_by_rule.lock().expect("Suppression stats lock poisoned");
        *stats.entry(rule_id.to_string()).or_insert(0) += 1;
    }

    /// Add suppression rule
    pub fn add_rule(&self, rule: SuppressionRule) {
        let mut rules = self.rules.write().expect("Rules RwLock poisoned");
        rules.push(rule);

        // Sort by priority (higher priority first)
        rules.sort_by_key(|x| std::cmp::Reverse(x.priority));
    }

    /// Remove suppression rule
    pub fn remove_rule(&self, rule_id: &str) {
        let mut rules = self.rules.write().expect("Rules RwLock poisoned");
        rules.retain(|rule| rule.id != rule_id);
    }

    /// Get suppression statistics
    pub fn get_stats(&self) -> SuppressionStats {
        SuppressionStats {
            total_processed: AtomicU64::new(self.stats.total_processed.load(Ordering::Relaxed)),
            total_suppressed: AtomicU64::new(self.stats.total_suppressed.load(Ordering::Relaxed)),
            suppression_by_rule: Arc::new(Mutex::new(
                self.stats
                    .suppression_by_rule
                    .lock()
                    .expect("Suppression stats lock poisoned")
                    .clone(),
            )),
            avg_suppression_rate: Arc::new(Mutex::new(
                *self
                    .stats
                    .avg_suppression_rate
                    .lock()
                    .expect("Avg suppression rate lock poisoned"),
            )),
            peak_suppression_rate: Arc::new(Mutex::new(
                *self
                    .stats
                    .peak_suppression_rate
                    .lock()
                    .expect("Peak suppression rate lock poisoned"),
            )),
        }
    }
}

impl Default for SuppressionConfig {
    fn default() -> Self {
        Self {
            enable_deduplication: true,
            deduplication_window: Duration::from_secs(300), // 5 minutes
            max_fingerprint_cache: 10000,
            default_suppression_window: Duration::from_secs(600), // 10 minutes
            enable_adaptive_suppression: true,
        }
    }
}
