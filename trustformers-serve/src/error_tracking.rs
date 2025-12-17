//! Error Tracking System
//!
//! Provides comprehensive error tracking and monitoring capabilities for
//! capturing, categorizing, and analyzing errors for better debugging and observability.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::{broadcast, mpsc, RwLock};
use uuid::Uuid;

#[derive(Debug, Error)]
pub enum ErrorTrackingError {
    #[error("Failed to initialize error tracking: {0}")]
    InitializationError(String),

    #[error("Failed to serialize error data: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Failed to store error: {0}")]
    StorageError(String),

    #[error("Failed to export error data: {0}")]
    ExportError(String),

    #[error("Error not found: {0}")]
    ErrorNotFound(String),

    #[error("Invalid error configuration: {0}")]
    ConfigurationError(String),
}

/// Error tracking configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorTrackingConfig {
    /// Enable error tracking
    pub enabled: bool,

    /// Maximum number of errors to keep in memory
    pub max_errors_in_memory: usize,

    /// Error retention period in hours
    pub retention_hours: u64,

    /// Enable error grouping
    pub enable_grouping: bool,

    /// Enable automatic error deduplication
    pub enable_deduplication: bool,

    /// Enable error severity classification
    pub enable_severity_classification: bool,

    /// Enable error trend analysis
    pub enable_trend_analysis: bool,

    /// Enable real-time notifications
    pub enable_notifications: bool,

    /// Notification thresholds
    pub notification_thresholds: NotificationThresholds,

    /// Enable error sampling
    pub enable_sampling: bool,

    /// Sampling rate (0.0 to 1.0)
    pub sampling_rate: f64,

    /// Enable stack trace capture
    pub enable_stack_traces: bool,

    /// Maximum stack trace depth
    pub max_stack_trace_depth: usize,

    /// Enable context capture
    pub enable_context_capture: bool,

    /// Export interval in seconds
    pub export_interval_seconds: u64,

    /// Export endpoints
    pub export_endpoints: Vec<String>,
}

impl Default for ErrorTrackingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_errors_in_memory: 10000,
            retention_hours: 72,
            enable_grouping: true,
            enable_deduplication: true,
            enable_severity_classification: true,
            enable_trend_analysis: true,
            enable_notifications: true,
            notification_thresholds: NotificationThresholds::default(),
            enable_sampling: false,
            sampling_rate: 1.0,
            enable_stack_traces: true,
            max_stack_trace_depth: 50,
            enable_context_capture: true,
            export_interval_seconds: 300,
            export_endpoints: Vec::new(),
        }
    }
}

/// Notification thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationThresholds {
    /// Critical error threshold (errors per minute)
    pub critical_errors_per_minute: f64,

    /// High error rate threshold (errors per minute)
    pub high_error_rate_per_minute: f64,

    /// New error type threshold (new error types per hour)
    pub new_error_types_per_hour: u64,

    /// Error spike threshold (percentage increase)
    pub error_spike_threshold_percent: f64,

    /// Spike threshold (occurrences of same error type in 5 minutes)
    pub spike_threshold: u64,

    /// Consecutive error threshold (same error pattern consecutively)
    pub consecutive_threshold: u64,
}

impl Default for NotificationThresholds {
    fn default() -> Self {
        Self {
            critical_errors_per_minute: 10.0,
            high_error_rate_per_minute: 50.0,
            new_error_types_per_hour: 5,
            error_spike_threshold_percent: 200.0,
            spike_threshold: 10,
            consecutive_threshold: 5,
        }
    }
}

/// Error entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEntry {
    /// Error ID
    pub id: String,

    /// Error type
    pub error_type: String,

    /// Error message
    pub message: String,

    /// Error severity
    pub severity: ErrorSeverity,

    /// Error category
    pub category: ErrorCategory,

    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Context information
    pub context: ErrorContext,

    /// Stack trace
    pub stack_trace: Option<Vec<StackFrame>>,

    /// Tags
    pub tags: HashMap<String, String>,

    /// Fingerprint for grouping
    pub fingerprint: String,

    /// Occurrence count
    pub occurrence_count: u64,

    /// First seen timestamp
    pub first_seen: chrono::DateTime<chrono::Utc>,

    /// Last seen timestamp
    pub last_seen: chrono::DateTime<chrono::Utc>,

    /// Related errors
    pub related_errors: Vec<String>,

    /// Resolution status
    pub resolution_status: ResolutionStatus,

    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ErrorEntry {
    pub fn new(error_type: String, message: String) -> Self {
        let now = chrono::Utc::now();
        let fingerprint = Self::generate_fingerprint(&error_type, &message);

        Self {
            id: Uuid::new_v4().to_string(),
            error_type,
            message,
            severity: ErrorSeverity::Medium,
            category: ErrorCategory::Unknown,
            timestamp: now,
            context: ErrorContext::new(),
            stack_trace: None,
            tags: HashMap::new(),
            fingerprint,
            occurrence_count: 1,
            first_seen: now,
            last_seen: now,
            related_errors: Vec::new(),
            resolution_status: ResolutionStatus::Unresolved,
            metadata: HashMap::new(),
        }
    }

    pub fn with_severity(mut self, severity: ErrorSeverity) -> Self {
        self.severity = severity;
        self
    }

    pub fn with_category(mut self, category: ErrorCategory) -> Self {
        self.category = category;
        self
    }

    pub fn with_context(mut self, context: ErrorContext) -> Self {
        self.context = context;
        self
    }

    pub fn with_stack_trace(mut self, stack_trace: Vec<StackFrame>) -> Self {
        self.stack_trace = Some(stack_trace);
        self
    }

    pub fn with_tags(mut self, tags: HashMap<String, String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_metadata(mut self, metadata: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = metadata;
        self
    }

    fn generate_fingerprint(error_type: &str, message: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        error_type.hash(&mut hasher);
        message.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    pub fn increment_occurrence(&mut self) {
        self.occurrence_count += 1;
        self.last_seen = chrono::Utc::now();
    }
}

/// Error severity
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Error category
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    Unknown,
    Authentication,
    Authorization,
    Validation,
    Network,
    Database,
    FileSystem,
    External,
    Internal,
    Configuration,
    Resource,
    Performance,
    Security,
    Business,
}

/// Error context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    /// Request ID
    pub request_id: Option<String>,

    /// User ID
    pub user_id: Option<String>,

    /// Session ID
    pub session_id: Option<String>,

    /// IP address
    pub ip_address: Option<String>,

    /// User agent
    pub user_agent: Option<String>,

    /// HTTP method
    pub method: Option<String>,

    /// URL path
    pub path: Option<String>,

    /// Query parameters
    pub query_params: HashMap<String, String>,

    /// Headers
    pub headers: HashMap<String, String>,

    /// Environment
    pub environment: Option<String>,

    /// Service name
    pub service_name: Option<String>,

    /// Service version
    pub service_version: Option<String>,

    /// Additional context
    pub additional_context: HashMap<String, serde_json::Value>,
}

impl ErrorContext {
    pub fn new() -> Self {
        Self {
            request_id: None,
            user_id: None,
            session_id: None,
            ip_address: None,
            user_agent: None,
            method: None,
            path: None,
            query_params: HashMap::new(),
            headers: HashMap::new(),
            environment: None,
            service_name: None,
            service_version: None,
            additional_context: HashMap::new(),
        }
    }

    pub fn with_request_id(mut self, request_id: String) -> Self {
        self.request_id = Some(request_id);
        self
    }

    pub fn with_user_id(mut self, user_id: String) -> Self {
        self.user_id = Some(user_id);
        self
    }

    pub fn with_http_info(mut self, method: String, path: String) -> Self {
        self.method = Some(method);
        self.path = Some(path);
        self
    }

    pub fn with_service_info(mut self, service_name: String, service_version: String) -> Self {
        self.service_name = Some(service_name);
        self.service_version = Some(service_version);
        self
    }
}

/// Stack frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackFrame {
    /// Function name
    pub function: String,

    /// File path
    pub file: Option<String>,

    /// Line number
    pub line: Option<u32>,

    /// Column number
    pub column: Option<u32>,

    /// Module name
    pub module: Option<String>,

    /// Source code context
    pub context: Option<SourceContext>,
}

/// Source context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceContext {
    /// Pre-context lines
    pub pre_context: Vec<String>,

    /// Context line
    pub context_line: String,

    /// Post-context lines
    pub post_context: Vec<String>,
}

/// Resolution status
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResolutionStatus {
    Unresolved,
    Resolved,
    Ignored,
    InProgress,
}

/// Error group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorGroup {
    /// Group ID
    pub id: String,

    /// Group fingerprint
    pub fingerprint: String,

    /// Group title
    pub title: String,

    /// Group description
    pub description: String,

    /// Error count
    pub error_count: u64,

    /// Affected users
    pub affected_users: u64,

    /// First seen
    pub first_seen: chrono::DateTime<chrono::Utc>,

    /// Last seen
    pub last_seen: chrono::DateTime<chrono::Utc>,

    /// Severity
    pub severity: ErrorSeverity,

    /// Category
    pub category: ErrorCategory,

    /// Resolution status
    pub resolution_status: ResolutionStatus,

    /// Sample errors
    pub sample_errors: Vec<String>,

    /// Tags
    pub tags: HashMap<String, String>,
}

/// Error statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStatistics {
    /// Total errors
    pub total_errors: u64,

    /// Unique errors
    pub unique_errors: u64,

    /// Error groups
    pub error_groups: u64,

    /// Errors by severity
    pub errors_by_severity: HashMap<ErrorSeverity, u64>,

    /// Errors by category
    pub errors_by_category: HashMap<ErrorCategory, u64>,

    /// Errors per minute
    pub errors_per_minute: f64,

    /// Top error types
    pub top_error_types: Vec<(String, u64)>,

    /// Top error messages
    pub top_error_messages: Vec<(String, u64)>,

    /// Recent error trends
    pub recent_trends: Vec<ErrorTrend>,

    /// Resolution statistics
    pub resolution_stats: ResolutionStatistics,
}

/// Error trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorTrend {
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Error count
    pub error_count: u64,

    /// Unique error count
    pub unique_error_count: u64,

    /// Average severity
    pub average_severity: f64,
}

/// Resolution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionStatistics {
    /// Resolved errors
    pub resolved_errors: u64,

    /// Unresolved errors
    pub unresolved_errors: u64,

    /// Ignored errors
    pub ignored_errors: u64,

    /// In-progress errors
    pub in_progress_errors: u64,

    /// Average resolution time
    pub average_resolution_time_hours: f64,
}

/// Error notification
#[derive(Debug, Clone, Serialize)]
pub struct ErrorNotification {
    /// Notification ID
    pub id: String,

    /// Notification type
    pub notification_type: NotificationType,

    /// Message
    pub message: String,

    /// Severity
    pub severity: ErrorSeverity,

    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,

    /// Error details
    pub error_details: HashMap<String, serde_json::Value>,
}

/// Notification type
#[derive(Debug, Clone, Serialize)]
pub enum NotificationType {
    HighErrorRate,
    CriticalError,
    NewErrorType,
    ErrorSpike,
    ErrorResolution,
}

/// Error tracking system
pub struct ErrorTrackingSystem {
    config: ErrorTrackingConfig,

    /// Error storage
    errors: Arc<RwLock<HashMap<String, ErrorEntry>>>,

    /// Error groups
    error_groups: Arc<RwLock<HashMap<String, ErrorGroup>>>,

    /// Error history for trend analysis
    error_history: Arc<RwLock<VecDeque<ErrorEntry>>>,

    /// Error sender
    error_sender: mpsc::UnboundedSender<ErrorEntry>,

    /// Notification sender
    notification_sender: broadcast::Sender<ErrorNotification>,

    /// Background task handles
    task_handles: Arc<RwLock<Vec<tokio::task::JoinHandle<()>>>>,
}

impl ErrorTrackingSystem {
    pub fn new(config: ErrorTrackingConfig) -> Self {
        let (error_sender, error_receiver) = mpsc::unbounded_channel();
        let (notification_sender, _) = broadcast::channel(1000);

        let system = Self {
            config,
            errors: Arc::new(RwLock::new(HashMap::new())),
            error_groups: Arc::new(RwLock::new(HashMap::new())),
            error_history: Arc::new(RwLock::new(VecDeque::new())),
            error_sender,
            notification_sender,
            task_handles: Arc::new(RwLock::new(Vec::new())),
        };

        // Start background processing
        system.start_background_processing(error_receiver);

        system
    }

    fn start_background_processing(&self, mut error_receiver: mpsc::UnboundedReceiver<ErrorEntry>) {
        let config = self.config.clone();
        let errors = Arc::clone(&self.errors);
        let error_groups = Arc::clone(&self.error_groups);
        let error_history = Arc::clone(&self.error_history);
        let notification_sender = self.notification_sender.clone();

        // Error processing task
        let error_task = tokio::spawn(async move {
            while let Some(error) = error_receiver.recv().await {
                // Process error
                Self::process_error(
                    &config,
                    &errors,
                    &error_groups,
                    &error_history,
                    &notification_sender,
                    error,
                )
                .await;
            }
        });

        // Cleanup task
        let config_clone = self.config.clone();
        let errors_clone = Arc::clone(&self.errors);
        let error_groups_clone = Arc::clone(&self.error_groups);
        let error_history_clone = Arc::clone(&self.error_history);

        let cleanup_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(3600)); // Every hour

            loop {
                interval.tick().await;
                Self::cleanup_old_errors(
                    &config_clone,
                    &errors_clone,
                    &error_groups_clone,
                    &error_history_clone,
                )
                .await;
            }
        });

        // Store task handles
        let task_handles = Arc::clone(&self.task_handles);
        tokio::spawn(async move {
            let mut handles = task_handles.write().await;
            handles.push(error_task);
            handles.push(cleanup_task);
        });
    }

    async fn process_error(
        config: &ErrorTrackingConfig,
        errors: &Arc<RwLock<HashMap<String, ErrorEntry>>>,
        error_groups: &Arc<RwLock<HashMap<String, ErrorGroup>>>,
        error_history: &Arc<RwLock<VecDeque<ErrorEntry>>>,
        notification_sender: &broadcast::Sender<ErrorNotification>,
        mut error: ErrorEntry,
    ) {
        // Apply sampling using hash-based approach (Send-safe)
        if config.enable_sampling {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            error.id.hash(&mut hasher);
            let hash_value = hasher.finish();
            let sample_value = (hash_value % 10000) as f64 / 10000.0;
            if sample_value > config.sampling_rate {
                return;
            }
        }

        // Classify severity if enabled
        if config.enable_severity_classification {
            error.severity = Self::classify_severity(&error);
        }

        // Categorize error
        error.category = Self::categorize_error(&error);

        // Store or update error
        let is_new_error = {
            let mut errors_map = errors.write().await;

            if config.enable_deduplication {
                if let Some(existing_error) = errors_map.get_mut(&error.fingerprint) {
                    existing_error.increment_occurrence();
                    false
                } else {
                    errors_map.insert(error.fingerprint.clone(), error.clone());
                    true
                }
            } else {
                errors_map.insert(error.id.clone(), error.clone());
                true
            }
        };

        // Update error groups
        if config.enable_grouping {
            Self::update_error_groups(error_groups, &error, is_new_error).await;
        }

        // Add to history
        {
            let mut history = error_history.write().await;
            history.push_back(error.clone());

            // Limit history size
            if history.len() > config.max_errors_in_memory {
                history.pop_front();
            }
        }

        // Send notifications
        if config.enable_notifications {
            Self::check_notification_thresholds(config, notification_sender, &error, is_new_error)
                .await;
        }
    }

    fn classify_severity(error: &ErrorEntry) -> ErrorSeverity {
        // Simple severity classification based on error type and message
        let error_type_lower = error.error_type.to_lowercase();
        let message_lower = error.message.to_lowercase();

        if error_type_lower.contains("panic")
            || error_type_lower.contains("fatal")
            || message_lower.contains("out of memory")
            || message_lower.contains("segfault")
        {
            ErrorSeverity::Critical
        } else if error_type_lower.contains("error")
            || message_lower.contains("failed")
            || message_lower.contains("timeout")
        {
            ErrorSeverity::High
        } else if error_type_lower.contains("warning") || message_lower.contains("deprecated") {
            ErrorSeverity::Low
        } else {
            ErrorSeverity::Medium
        }
    }

    fn categorize_error(error: &ErrorEntry) -> ErrorCategory {
        let error_type_lower = error.error_type.to_lowercase();
        let message_lower = error.message.to_lowercase();

        if error_type_lower.contains("auth") || message_lower.contains("authentication") {
            ErrorCategory::Authentication
        } else if error_type_lower.contains("permission") || message_lower.contains("authorization")
        {
            ErrorCategory::Authorization
        } else if error_type_lower.contains("validation") || message_lower.contains("invalid") {
            ErrorCategory::Validation
        } else if error_type_lower.contains("network") || message_lower.contains("connection") {
            ErrorCategory::Network
        } else if error_type_lower.contains("database") || error_type_lower.contains("sql") {
            ErrorCategory::Database
        } else if error_type_lower.contains("file") || error_type_lower.contains("io") {
            ErrorCategory::FileSystem
        } else if error_type_lower.contains("performance") || message_lower.contains("slow") {
            ErrorCategory::Performance
        } else if error_type_lower.contains("security") || message_lower.contains("security") {
            ErrorCategory::Security
        } else {
            ErrorCategory::Internal
        }
    }

    async fn update_error_groups(
        error_groups: &Arc<RwLock<HashMap<String, ErrorGroup>>>,
        error: &ErrorEntry,
        is_new_error: bool,
    ) {
        let mut groups = error_groups.write().await;

        if let Some(group) = groups.get_mut(&error.fingerprint) {
            group.error_count += 1;
            group.last_seen = error.timestamp;

            if group.sample_errors.len() < 10 {
                group.sample_errors.push(error.id.clone());
            }
        } else if is_new_error {
            let group = ErrorGroup {
                id: Uuid::new_v4().to_string(),
                fingerprint: error.fingerprint.clone(),
                title: error.error_type.clone(),
                description: error.message.clone(),
                error_count: 1,
                affected_users: 1,
                first_seen: error.timestamp,
                last_seen: error.timestamp,
                severity: error.severity.clone(),
                category: error.category.clone(),
                resolution_status: ResolutionStatus::Unresolved,
                sample_errors: vec![error.id.clone()],
                tags: error.tags.clone(),
            };

            groups.insert(error.fingerprint.clone(), group);
        }
    }

    async fn check_notification_thresholds(
        config: &ErrorTrackingConfig,
        notification_sender: &broadcast::Sender<ErrorNotification>,
        error: &ErrorEntry,
        is_new_error: bool,
    ) {
        // Check for critical errors
        if error.severity == ErrorSeverity::Critical {
            let notification = ErrorNotification {
                id: Uuid::new_v4().to_string(),
                notification_type: NotificationType::CriticalError,
                message: format!("Critical error detected: {}", error.message),
                severity: error.severity.clone(),
                timestamp: chrono::Utc::now(),
                error_details: HashMap::new(),
            };

            let _ = notification_sender.send(notification);
        }

        // Check for new error types
        if is_new_error {
            let notification = ErrorNotification {
                id: Uuid::new_v4().to_string(),
                notification_type: NotificationType::NewErrorType,
                message: format!("New error type detected: {}", error.error_type),
                severity: error.severity.clone(),
                timestamp: chrono::Utc::now(),
                error_details: HashMap::new(),
            };

            let _ = notification_sender.send(notification);
        }

        // Additional threshold checks

        // Check for high error rate patterns
        Self::check_high_error_rate_threshold(config, notification_sender, error).await;

        // Check for error spike detection
        Self::check_error_spike_threshold(config, notification_sender, error).await;

        // Check for consecutive error patterns
        Self::check_consecutive_error_pattern(config, notification_sender, error).await;
    }

    /// Check for high error rate threshold violations
    async fn check_high_error_rate_threshold(
        config: &ErrorTrackingConfig,
        notification_sender: &broadcast::Sender<ErrorNotification>,
        _error: &ErrorEntry,
    ) {
        // Check if we're exceeding error rate thresholds
        // This is a simplified implementation - in production, you'd want to track error rates over time
        static RECENT_ERROR_COUNT: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        static LAST_RESET: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

        let now = chrono::Utc::now().timestamp() as u64;
        let last_reset = LAST_RESET.load(std::sync::atomic::Ordering::Relaxed);

        // Reset counter every minute
        if now - last_reset > 60 {
            RECENT_ERROR_COUNT.store(0, std::sync::atomic::Ordering::Relaxed);
            LAST_RESET.store(now, std::sync::atomic::Ordering::Relaxed);
        }

        let current_count =
            RECENT_ERROR_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;

        // Alert if we exceed configured errors per minute threshold
        let error_rate_threshold = config.notification_thresholds.high_error_rate_per_minute as u64;

        if current_count >= error_rate_threshold {
            let notification = ErrorNotification {
                id: Uuid::new_v4().to_string(),
                notification_type: NotificationType::HighErrorRate,
                message: format!(
                    "High error rate detected: {} errors in the last minute",
                    current_count
                ),
                severity: ErrorSeverity::High,
                timestamp: chrono::Utc::now(),
                error_details: {
                    let mut details = HashMap::new();
                    details.insert(
                        "error_count".to_string(),
                        serde_json::Value::Number(current_count.into()),
                    );
                    details.insert(
                        "threshold".to_string(),
                        serde_json::Value::Number(error_rate_threshold.into()),
                    );
                    details
                },
            };

            let _ = notification_sender.send(notification);
        }
    }

    /// Check for error spike patterns
    async fn check_error_spike_threshold(
        config: &ErrorTrackingConfig,
        notification_sender: &broadcast::Sender<ErrorNotification>,
        error: &ErrorEntry,
    ) {
        // Detect sudden spikes in specific error types
        static ERROR_TYPE_COUNTS: std::sync::LazyLock<
            std::sync::Arc<std::sync::Mutex<HashMap<String, (u64, u64)>>>,
        > = std::sync::LazyLock::new(|| std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())));

        let mut counts = ERROR_TYPE_COUNTS.lock().unwrap();
        let now = chrono::Utc::now().timestamp() as u64;
        let entry = counts.entry(error.error_type.clone()).or_insert((0, now));

        // Reset counter every 5 minutes
        if now - entry.1 > 300 {
            entry.0 = 0;
            entry.1 = now;
        }

        entry.0 += 1;

        // Alert if specific error type occurs more than configured times in 5 minutes
        let spike_threshold = config.notification_thresholds.spike_threshold;

        if entry.0 >= spike_threshold {
            let notification = ErrorNotification {
                id: Uuid::new_v4().to_string(),
                notification_type: NotificationType::ErrorSpike,
                message: format!(
                    "Error spike detected for '{}': {} occurrences in 5 minutes",
                    error.error_type, entry.0
                ),
                severity: ErrorSeverity::High,
                timestamp: chrono::Utc::now(),
                error_details: {
                    let mut details = HashMap::new();
                    details.insert(
                        "error_type".to_string(),
                        serde_json::Value::String(error.error_type.clone()),
                    );
                    details.insert(
                        "occurrence_count".to_string(),
                        serde_json::Value::Number(entry.0.into()),
                    );
                    details.insert(
                        "threshold".to_string(),
                        serde_json::Value::Number(spike_threshold.into()),
                    );
                    details
                },
            };

            let _ = notification_sender.send(notification);

            // Reset counter after notification to avoid spam
            entry.0 = 0;
        }
    }

    /// Check for consecutive error patterns
    async fn check_consecutive_error_pattern(
        config: &ErrorTrackingConfig,
        notification_sender: &broadcast::Sender<ErrorNotification>,
        error: &ErrorEntry,
    ) {
        // Track consecutive errors of the same type or from the same service
        static CONSECUTIVE_PATTERNS: std::sync::LazyLock<
            std::sync::Arc<std::sync::Mutex<HashMap<String, u64>>>,
        > = std::sync::LazyLock::new(|| std::sync::Arc::new(std::sync::Mutex::new(HashMap::new())));

        let mut patterns = CONSECUTIVE_PATTERNS.lock().unwrap();
        let source = error.context.service_name.as_deref().unwrap_or("unknown");
        let pattern_key = format!("{}:{}", error.error_type, source);

        let count = patterns.entry(pattern_key.clone()).or_insert(0);
        *count += 1;

        // Alert if we see the same error pattern consecutively
        let consecutive_threshold = config.notification_thresholds.consecutive_threshold;

        if *count >= consecutive_threshold {
            let notification = ErrorNotification {
                id: Uuid::new_v4().to_string(),
                notification_type: NotificationType::ErrorSpike,
                message: format!("Consecutive error pattern detected: '{}' from source '{}' occurred {} times consecutively",
                    error.error_type, source, count),
                severity: ErrorSeverity::Medium,
                timestamp: chrono::Utc::now(),
                error_details: {
                    let mut details = HashMap::new();
                    details.insert("error_type".to_string(), serde_json::Value::String(error.error_type.clone()));
                    details.insert("source".to_string(), serde_json::Value::String(source.to_string()));
                    details.insert("consecutive_count".to_string(), serde_json::Value::Number((*count).into()));
                    details.insert("threshold".to_string(), serde_json::Value::Number(consecutive_threshold.into()));
                    details
                },
            };

            let _ = notification_sender.send(notification);

            // Reset counter after notification
            *count = 0;
        }
    }

    async fn cleanup_old_errors(
        config: &ErrorTrackingConfig,
        errors: &Arc<RwLock<HashMap<String, ErrorEntry>>>,
        error_groups: &Arc<RwLock<HashMap<String, ErrorGroup>>>,
        error_history: &Arc<RwLock<VecDeque<ErrorEntry>>>,
    ) {
        let cutoff_time =
            chrono::Utc::now() - chrono::Duration::hours(config.retention_hours as i64);

        // Clean up old errors
        {
            let mut errors_map = errors.write().await;
            errors_map.retain(|_, error| error.timestamp > cutoff_time);
        }

        // Clean up old error groups
        {
            let mut groups = error_groups.write().await;
            groups.retain(|_, group| group.last_seen > cutoff_time);
        }

        // Clean up old history
        {
            let mut history = error_history.write().await;
            while let Some(front) = history.front() {
                if front.timestamp <= cutoff_time {
                    history.pop_front();
                } else {
                    break;
                }
            }
        }
    }

    pub async fn track_error(&self, error: ErrorEntry) -> Result<(), ErrorTrackingError> {
        if !self.config.enabled {
            return Ok(());
        }

        self.error_sender
            .send(error)
            .map_err(|_| ErrorTrackingError::StorageError("Failed to send error".to_string()))?;

        Ok(())
    }

    pub async fn get_error(&self, error_id: &str) -> Result<ErrorEntry, ErrorTrackingError> {
        let errors = self.errors.read().await;
        errors
            .get(error_id)
            .cloned()
            .ok_or_else(|| ErrorTrackingError::ErrorNotFound(error_id.to_string()))
    }

    pub async fn get_error_group(&self, group_id: &str) -> Result<ErrorGroup, ErrorTrackingError> {
        let groups = self.error_groups.read().await;
        groups
            .values()
            .find(|group| group.id == group_id)
            .cloned()
            .ok_or_else(|| ErrorTrackingError::ErrorNotFound(group_id.to_string()))
    }

    pub async fn get_statistics(&self) -> ErrorStatistics {
        let errors = self.errors.read().await;
        let error_groups = self.error_groups.read().await;

        let total_errors = errors.len() as u64;
        let unique_errors = errors.len() as u64;
        let error_groups_count = error_groups.len() as u64;

        let mut errors_by_severity = HashMap::new();
        let mut errors_by_category = HashMap::new();
        let mut top_error_types = HashMap::new();
        let mut top_error_messages = HashMap::new();

        for error in errors.values() {
            *errors_by_severity.entry(error.severity.clone()).or_insert(0) += 1;
            *errors_by_category.entry(error.category.clone()).or_insert(0) += 1;
            *top_error_types.entry(error.error_type.clone()).or_insert(0) += 1;
            *top_error_messages.entry(error.message.clone()).or_insert(0) += 1;
        }

        let mut top_error_types_vec: Vec<(String, u64)> = top_error_types.into_iter().collect();
        top_error_types_vec.sort_by(|a, b| b.1.cmp(&a.1));
        top_error_types_vec.truncate(10);

        let mut top_error_messages_vec: Vec<(String, u64)> =
            top_error_messages.into_iter().collect();
        top_error_messages_vec.sort_by(|a, b| b.1.cmp(&a.1));
        top_error_messages_vec.truncate(10);

        // Calculate errors per minute from the last 60 minutes
        let now = chrono::Utc::now();
        let one_hour_ago = now - chrono::Duration::minutes(60);
        let recent_errors: Vec<_> =
            errors.values().filter(|error| error.timestamp > one_hour_ago).collect();

        let errors_per_minute =
            if !recent_errors.is_empty() { recent_errors.len() as f64 / 60.0 } else { 0.0 };

        ErrorStatistics {
            total_errors,
            unique_errors,
            error_groups: error_groups_count,
            errors_by_severity,
            errors_by_category,
            errors_per_minute,
            top_error_types: top_error_types_vec,
            top_error_messages: top_error_messages_vec,
            recent_trends: Self::calculate_error_trends(&errors, 24), // Calculate 24-hour trends
            resolution_stats: ResolutionStatistics {
                resolved_errors: 0,
                unresolved_errors: total_errors,
                ignored_errors: 0,
                in_progress_errors: 0,
                average_resolution_time_hours: 0.0,
            },
        }
    }

    /// Calculate error trends over specified number of hours
    fn calculate_error_trends(errors: &HashMap<String, ErrorEntry>, hours: i64) -> Vec<ErrorTrend> {
        let now = chrono::Utc::now();
        let mut trends = Vec::new();

        // Create hourly buckets for the specified time period
        for hour_offset in (0..hours).rev() {
            let hour_start = now - chrono::Duration::hours(hour_offset + 1);
            let hour_end = now - chrono::Duration::hours(hour_offset);

            // Filter errors for this hour
            let hour_errors: Vec<&ErrorEntry> = errors
                .values()
                .filter(|error| error.timestamp >= hour_start && error.timestamp < hour_end)
                .collect();

            // Calculate unique error count (distinct error types)
            let unique_errors: std::collections::HashSet<&String> =
                hour_errors.iter().map(|error| &error.error_type).collect();

            // Calculate average severity
            let average_severity = if !hour_errors.is_empty() {
                let severity_sum: f64 = hour_errors
                    .iter()
                    .map(|error| match error.severity {
                        ErrorSeverity::Low => 1.0,
                        ErrorSeverity::Medium => 2.0,
                        ErrorSeverity::High => 3.0,
                        ErrorSeverity::Critical => 4.0,
                    })
                    .sum();
                severity_sum / hour_errors.len() as f64
            } else {
                0.0
            };

            let trend = ErrorTrend {
                timestamp: hour_start,
                error_count: hour_errors.len() as u64,
                unique_error_count: unique_errors.len() as u64,
                average_severity,
            };

            trends.push(trend);
        }

        trends
    }

    pub async fn resolve_error(&self, error_id: &str) -> Result<(), ErrorTrackingError> {
        let mut errors = self.errors.write().await;
        let error = errors
            .get_mut(error_id)
            .ok_or_else(|| ErrorTrackingError::ErrorNotFound(error_id.to_string()))?;

        error.resolution_status = ResolutionStatus::Resolved;

        // Send notification
        let notification = ErrorNotification {
            id: Uuid::new_v4().to_string(),
            notification_type: NotificationType::ErrorResolution,
            message: format!("Error resolved: {}", error.message),
            severity: error.severity.clone(),
            timestamp: chrono::Utc::now(),
            error_details: HashMap::new(),
        };

        let _ = self.notification_sender.send(notification);

        Ok(())
    }

    pub fn subscribe_to_notifications(&self) -> broadcast::Receiver<ErrorNotification> {
        self.notification_sender.subscribe()
    }

    pub async fn export_errors(
        &self,
        format: ErrorExportFormat,
    ) -> Result<Vec<u8>, ErrorTrackingError> {
        let errors = self.errors.read().await;
        let error_list: Vec<&ErrorEntry> = errors.values().collect();

        match format {
            ErrorExportFormat::Json => {
                let json = serde_json::to_string_pretty(&error_list)
                    .map_err(ErrorTrackingError::SerializationError)?;
                Ok(json.into_bytes())
            },
            ErrorExportFormat::Csv => {
                let mut csv_content = String::new();

                // CSV Header
                csv_content.push_str("ID,Error Type,Message,Severity,Category,Occurrence Count,First Seen,Last Seen,Resolution Status,Context\n");

                // CSV Rows
                for error in error_list {
                    // Build context string from ErrorContext fields
                    let mut context_parts = Vec::new();
                    if let Some(ref request_id) = error.context.request_id {
                        context_parts.push(format!("request_id={}", request_id));
                    }
                    if let Some(ref user_id) = error.context.user_id {
                        context_parts.push(format!("user_id={}", user_id));
                    }
                    if let Some(ref session_id) = error.context.session_id {
                        context_parts.push(format!("session_id={}", session_id));
                    }
                    if let Some(ref ip_address) = error.context.ip_address {
                        context_parts.push(format!("ip_address={}", ip_address));
                    }
                    if let Some(ref method) = error.context.method {
                        context_parts.push(format!("method={}", method));
                    }
                    if let Some(ref path) = error.context.path {
                        context_parts.push(format!("path={}", path));
                    }
                    let context_str = context_parts.join(";");

                    // Escape CSV values by wrapping in quotes and escaping quotes
                    let escaped_values = [
                        escape_csv_field(&error.id),
                        escape_csv_field(&error.error_type),
                        escape_csv_field(&error.message),
                        escape_csv_field(&format!("{:?}", error.severity)),
                        escape_csv_field(&format!("{:?}", error.category)),
                        error.occurrence_count.to_string(),
                        error.first_seen.to_rfc3339(),
                        error.last_seen.to_rfc3339(),
                        escape_csv_field(&format!("{:?}", error.resolution_status)),
                        escape_csv_field(&context_str),
                    ];

                    csv_content.push_str(&escaped_values.join(","));
                    csv_content.push('\n');
                }

                Ok(csv_content.into_bytes())
            },
        }
    }
}

/// Helper function to escape CSV field values
fn escape_csv_field(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') || field.contains('\r') {
        // Escape quotes by doubling them and wrap the field in quotes
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorExportFormat {
    Json,
    Csv,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_entry_creation() {
        let error = ErrorEntry::new("TestError".to_string(), "Test message".to_string());
        assert!(!error.id.is_empty());
        assert_eq!(error.error_type, "TestError");
        assert_eq!(error.message, "Test message");
        assert_eq!(error.occurrence_count, 1);
        assert_eq!(error.resolution_status, ResolutionStatus::Unresolved);
    }

    #[test]
    fn test_error_fingerprint_generation() {
        let error1 = ErrorEntry::new("TestError".to_string(), "Test message".to_string());
        let error2 = ErrorEntry::new("TestError".to_string(), "Test message".to_string());

        assert_eq!(error1.fingerprint, error2.fingerprint);
    }

    #[test]
    fn test_error_context_builder() {
        let context = ErrorContext::new()
            .with_request_id("req-123".to_string())
            .with_user_id("user-456".to_string())
            .with_http_info("GET".to_string(), "/api/test".to_string());

        assert_eq!(context.request_id, Some("req-123".to_string()));
        assert_eq!(context.user_id, Some("user-456".to_string()));
        assert_eq!(context.method, Some("GET".to_string()));
        assert_eq!(context.path, Some("/api/test".to_string()));
    }

    #[tokio::test]
    async fn test_error_tracking_system() {
        let mut config = ErrorTrackingConfig::default();
        config.enable_deduplication = false; // Disable deduplication so error is stored by ID
        let system = ErrorTrackingSystem::new(config);

        let error = ErrorEntry::new("TestError".to_string(), "Test message".to_string());
        let error_id = error.id.clone();

        system.track_error(error).await.unwrap();

        // Give some time for background processing
        tokio::time::sleep(Duration::from_millis(100)).await;

        let retrieved_error = system.get_error(&error_id).await.unwrap();
        assert_eq!(retrieved_error.error_type, "TestError");
        assert_eq!(retrieved_error.message, "Test message");
    }

    #[tokio::test]
    async fn test_error_statistics() {
        let config = ErrorTrackingConfig::default();
        let system = ErrorTrackingSystem::new(config);

        let error1 = ErrorEntry::new("Error1".to_string(), "Message 1".to_string());
        let error2 = ErrorEntry::new("Error2".to_string(), "Message 2".to_string());

        system.track_error(error1).await.unwrap();
        system.track_error(error2).await.unwrap();

        // Give some time for background processing
        tokio::time::sleep(Duration::from_millis(100)).await;

        let stats = system.get_statistics().await;
        assert_eq!(stats.total_errors, 2);
        assert_eq!(stats.unique_errors, 2);
    }
}
