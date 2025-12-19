use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;
use tokio::sync::{broadcast, mpsc, RwLock};
use tracing::{debug, error, info, warn};

#[derive(Debug, Error)]
pub enum AuditError {
    #[error("Failed to serialize audit event: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Failed to write audit log: {0}")]
    WriteError(String),

    #[error("Invalid audit configuration")]
    ConfigurationError,

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Export error: {0}")]
    ExportError(String),

    #[error("Retention policy error: {0}")]
    RetentionError(String),

    #[error("Storage error: {0}")]
    StorageError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AuditEventType {
    // Authentication events
    LoginAttempt,
    LoginSuccess,
    LoginFailure,
    TokenRefresh,
    LogoutEvent,

    // API Key events
    ApiKeyCreated,
    ApiKeyUsed,
    ApiKeyRevoked,
    ApiKeyExpired,

    // Authorization events
    AccessGranted,
    AccessDenied,
    PermissionEscalation,

    // API events
    InferenceRequest,
    BatchRequest,
    StreamingRequest,
    ModelRequest,

    // Security events
    SuspiciousActivity,
    RateLimitHit,
    ValidationFailure,
    SecurityViolation,

    // System events
    ModelLoaded,
    ModelUnloaded,
    ServiceStarted,
    ServiceStopped,
    ConfigurationChanged,

    // Admin events
    UserCreated,
    UserDeleted,
    RoleChanged,
    SettingsChanged,

    // Compliance events
    DataAccessed,
    DataExported,
    DataDeleted,
    ConsentGiven,
    ConsentRevoked,

    // Performance events
    PerformanceThreshold,
    ResourceExhaustion,

    // Alert events
    AlertTriggered,
    AlertResolved,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AuditSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub event_type: AuditEventType,
    pub severity: AuditSeverity,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub resource: Option<String>,
    pub action: String,
    pub outcome: AuditOutcome,
    pub details: HashMap<String, String>,
    pub request_id: Option<String>,
    pub duration_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AuditOutcome {
    Success,
    Failure,
    Partial,
    Error,
}

impl AuditEvent {
    pub fn new(event_type: AuditEventType, action: String) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: Utc::now(),
            event_type,
            severity: AuditSeverity::Medium,
            user_id: None,
            session_id: None,
            ip_address: None,
            user_agent: None,
            resource: None,
            action,
            outcome: AuditOutcome::Success,
            details: HashMap::new(),
            request_id: None,
            duration_ms: None,
        }
    }

    pub fn with_severity(mut self, severity: AuditSeverity) -> Self {
        self.severity = severity;
        self
    }

    pub fn with_user(mut self, user_id: String) -> Self {
        self.user_id = Some(user_id);
        self
    }

    pub fn with_session(mut self, session_id: String) -> Self {
        self.session_id = Some(session_id);
        self
    }

    pub fn with_ip_address(mut self, ip_address: String) -> Self {
        self.ip_address = Some(ip_address);
        self
    }

    pub fn with_user_agent(mut self, user_agent: String) -> Self {
        self.user_agent = Some(user_agent);
        self
    }

    pub fn with_resource(mut self, resource: String) -> Self {
        self.resource = Some(resource);
        self
    }

    pub fn with_outcome(mut self, outcome: AuditOutcome) -> Self {
        self.outcome = outcome;
        self
    }

    pub fn with_request_id(mut self, request_id: String) -> Self {
        self.request_id = Some(request_id);
        self
    }

    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = Some(duration_ms);
        self
    }

    pub fn with_detail(mut self, key: String, value: String) -> Self {
        self.details.insert(key, value);
        self
    }

    pub fn with_details(mut self, details: HashMap<String, String>) -> Self {
        self.details.extend(details);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    pub enabled: bool,
    pub log_level: AuditSeverity,
    pub include_request_body: bool,
    pub include_response_body: bool,
    pub retention_days: u32,
    pub max_file_size_mb: u32,
    pub file_path: Option<String>,
    pub enable_database_storage: bool,
    pub database_url: Option<String>,
    pub enable_encryption: bool,
    pub encryption_key: Option<String>,
    pub enable_real_time_alerts: bool,
    pub alert_webhook_url: Option<String>,
    pub enable_compliance_mode: bool,
    pub compliance_standard: Option<String>,
    pub enable_log_forwarding: bool,
    pub log_forwarding_url: Option<String>,
    pub batch_size: u32,
    pub flush_interval_seconds: u32,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_level: AuditSeverity::Medium,
            include_request_body: false,
            include_response_body: false,
            retention_days: 90,
            max_file_size_mb: 100,
            file_path: None,
            enable_database_storage: false,
            database_url: None,
            enable_encryption: false,
            encryption_key: None,
            enable_real_time_alerts: false,
            alert_webhook_url: None,
            enable_compliance_mode: false,
            compliance_standard: None,
            enable_log_forwarding: false,
            log_forwarding_url: None,
            batch_size: 100,
            flush_interval_seconds: 60,
        }
    }
}

#[derive(Clone)]
pub struct AuditLogger {
    config: AuditConfig,
    event_buffer: Arc<RwLock<Vec<AuditEvent>>>,
    event_sender: mpsc::UnboundedSender<AuditEvent>,
    alert_sender: broadcast::Sender<AuditAlert>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditAlert {
    pub event_id: String,
    pub severity: AuditSeverity,
    pub event_type: AuditEventType,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub user_id: Option<String>,
    pub ip_address: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuditQuery {
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub event_types: Option<Vec<AuditEventType>>,
    pub user_ids: Option<Vec<String>>,
    pub severity: Option<AuditSeverity>,
    pub outcome: Option<AuditOutcome>,
    pub limit: Option<u32>,
    pub offset: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    pub total_events: u64,
    pub events_by_type: HashMap<AuditEventType, u64>,
    pub events_by_severity: HashMap<AuditSeverity, u64>,
    pub events_by_outcome: HashMap<AuditOutcome, u64>,
    pub top_users: Vec<(String, u64)>,
    pub top_ips: Vec<(String, u64)>,
    pub generated_at: DateTime<Utc>,
}

impl AuditLogger {
    pub fn new(config: AuditConfig) -> Self {
        let (event_sender, event_receiver) = mpsc::unbounded_channel();
        let (alert_sender, _) = broadcast::channel(1000);

        let logger = Self {
            config,
            event_buffer: Arc::new(RwLock::new(Vec::new())),
            event_sender,
            alert_sender,
        };

        // Start background processing
        logger.start_background_processing(event_receiver);

        logger
    }

    fn start_background_processing(&self, mut event_receiver: mpsc::UnboundedReceiver<AuditEvent>) {
        let config = self.config.clone();
        let event_buffer = Arc::clone(&self.event_buffer);
        let alert_sender = self.alert_sender.clone();

        tokio::spawn(async move {
            let mut flush_interval =
                tokio::time::interval(Duration::from_secs(config.flush_interval_seconds as u64));

            loop {
                tokio::select! {
                    // Process incoming events
                    event = event_receiver.recv() => {
                        match event {
                            Some(event) => {
                                // Add to buffer
                                {
                                    let mut buffer = event_buffer.write().await;
                                    buffer.push(event.clone());
                                }

                                // Check if we need to send alert
                                if config.enable_real_time_alerts {
                                    if let Some(alert) = Self::create_alert(&event) {
                                        let _ = alert_sender.send(alert);
                                    }
                                }

                                // Flush if buffer is full
                                let buffer_len = {
                                    let buffer = event_buffer.read().await;
                                    buffer.len()
                                };

                                if buffer_len >= config.batch_size as usize {
                                    Self::flush_events(&config, &event_buffer).await;
                                }
                            }
                            None => break,
                        }
                    }

                    // Periodic flush
                    _ = flush_interval.tick() => {
                        Self::flush_events(&config, &event_buffer).await;
                    }
                }
            }
        });
    }

    async fn flush_events(config: &AuditConfig, event_buffer: &Arc<RwLock<Vec<AuditEvent>>>) {
        let events = {
            let mut buffer = event_buffer.write().await;
            let events = buffer.clone();
            buffer.clear();
            events
        };

        if events.is_empty() {
            return;
        }

        // Process events based on configuration
        if config.enable_database_storage {
            if let Err(e) = Self::store_events_to_database(config, &events).await {
                warn!("Failed to store audit events to database: {}", e);
            }
        }

        if config.enable_log_forwarding {
            if let Err(e) = Self::forward_events_to_external_service(config, &events).await {
                warn!("Failed to forward audit events: {}", e);
            }
        }

        // Apply retention policy
        if config.retention_days > 0 {
            if let Err(e) = Self::cleanup_old_audit_events(config).await {
                warn!("Failed to cleanup old audit events: {}", e);
            }
        }
    }

    async fn store_events_to_database(
        config: &AuditConfig,
        events: &[AuditEvent],
    ) -> Result<(), anyhow::Error> {
        use std::path::Path;
        use tokio::fs::{create_dir_all, OpenOptions};
        use tokio::io::AsyncWriteExt;

        // For now, implement file-based storage that can be easily migrated to a real database
        let storage_path = if let Some(url) = &config.database_url {
            // Extract file path from database URL or use as-is
            if url.starts_with("file://") {
                url.strip_prefix("file://").unwrap_or(url).to_string()
            } else if url.starts_with("sqlite://") {
                url.strip_prefix("sqlite://").unwrap_or("audit_events.db").to_string()
            } else {
                // For other DB types, use a local file for now
                "audit_events.jsonl".to_string()
            }
        } else {
            "audit_events.jsonl".to_string()
        };

        // Ensure the directory exists
        if let Some(parent) = Path::new(&storage_path).parent() {
            create_dir_all(parent).await?;
        }

        // Open file for appending
        let mut file = OpenOptions::new().create(true).append(true).open(&storage_path).await?;

        // Write events as JSONL (one JSON object per line)
        for event in events {
            let json_line = serde_json::to_string(event)?;
            file.write_all(json_line.as_bytes()).await?;
            file.write_all(b"\n").await?;
        }

        file.flush().await?;

        debug!(
            "Stored {} audit events to database at {}",
            events.len(),
            storage_path
        );
        Ok(())
    }

    async fn forward_events_to_external_service(
        config: &AuditConfig,
        events: &[AuditEvent],
    ) -> Result<(), anyhow::Error> {
        if let Some(forwarding_url) = &config.log_forwarding_url {
            let client = reqwest::Client::new();

            // Prepare payload for external service
            let payload = serde_json::json!({
                "source": "trustformers-serve",
                "timestamp": chrono::Utc::now(),
                "events": events,
                "metadata": {
                    "service": "audit",
                    "version": env!("CARGO_PKG_VERSION"),
                    "instance_id": uuid::Uuid::new_v4()
                }
            });

            // Send events to external service
            let response = client
                .post(forwarding_url)
                .header("Content-Type", "application/json")
                .header(
                    "User-Agent",
                    format!("trustformers-serve/{}", env!("CARGO_PKG_VERSION")),
                )
                .json(&payload)
                .timeout(std::time::Duration::from_secs(30))
                .send()
                .await?;

            if response.status().is_success() {
                info!(
                    "Successfully forwarded {} audit events to {}",
                    events.len(),
                    forwarding_url
                );
            } else {
                warn!(
                    "Failed to forward audit events. Status: {}, Response: {}",
                    response.status(),
                    response.text().await.unwrap_or_default()
                );
            }
        } else {
            warn!("Log forwarding enabled but no forwarding URL configured");
        }

        Ok(())
    }

    async fn cleanup_old_audit_events(config: &AuditConfig) -> Result<(), anyhow::Error> {
        use chrono::{Duration, Utc};
        use std::path::Path;
        use tokio::fs::{File, OpenOptions};
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

        // Calculate cutoff date
        let cutoff_date = Utc::now() - Duration::days(config.retention_days as i64);

        // Determine storage path (same logic as store_events_to_database)
        let storage_path = if let Some(url) = &config.database_url {
            if url.starts_with("file://") {
                url.strip_prefix("file://").unwrap_or(url).to_string()
            } else if url.starts_with("sqlite://") {
                url.strip_prefix("sqlite://").unwrap_or("audit_events.db").to_string()
            } else {
                "audit_events.jsonl".to_string()
            }
        } else {
            "audit_events.jsonl".to_string()
        };

        // Check if file exists
        if !Path::new(&storage_path).exists() {
            return Ok(()); // Nothing to cleanup
        }

        // Read all events and filter out old ones
        let file = File::open(&storage_path).await?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        let mut retained_events = Vec::new();
        let mut removed_count = 0;

        while let Some(line) = lines.next_line().await? {
            if let Ok(event) = serde_json::from_str::<AuditEvent>(&line) {
                if event.timestamp > cutoff_date {
                    retained_events.push(line);
                } else {
                    removed_count += 1;
                }
            } else {
                // Keep malformed lines to avoid data loss
                retained_events.push(line);
            }
        }

        // If we removed any events, rewrite the file
        if removed_count > 0 {
            let mut file =
                OpenOptions::new().write(true).truncate(true).open(&storage_path).await?;

            for line in retained_events {
                file.write_all(line.as_bytes()).await?;
                file.write_all(b"\n").await?;
            }

            file.flush().await?;
            info!(
                "Cleaned up {} old audit events from {}",
                removed_count, storage_path
            );
        }

        Ok(())
    }

    fn create_alert(event: &AuditEvent) -> Option<AuditAlert> {
        // Create alerts for critical events
        match event.severity {
            AuditSeverity::Critical | AuditSeverity::High => Some(AuditAlert {
                event_id: event.id.clone(),
                severity: event.severity.clone(),
                event_type: event.event_type.clone(),
                message: format!("Critical audit event: {}", event.action),
                timestamp: event.timestamp,
                user_id: event.user_id.clone(),
                ip_address: event.ip_address.clone(),
            }),
            _ => None,
        }
    }

    pub fn log_event(&self, event: AuditEvent) -> Result<(), AuditError> {
        if !self.config.enabled {
            return Ok(());
        }

        // Check if event meets minimum severity level
        if !self.should_log_severity(&event.severity) {
            return Ok(());
        }

        // Serialize event to JSON
        let event_json = serde_json::to_string(&event)?;

        // Log using tracing based on severity
        match event.severity {
            AuditSeverity::Critical => {
                error!(
                    target: "audit",
                    event_id = %event.id,
                    event_type = ?event.event_type,
                    user_id = ?event.user_id,
                    outcome = ?event.outcome,
                    "{}", event_json
                );
            },
            AuditSeverity::High => {
                warn!(
                    target: "audit",
                    event_id = %event.id,
                    event_type = ?event.event_type,
                    user_id = ?event.user_id,
                    outcome = ?event.outcome,
                    "{}", event_json
                );
            },
            AuditSeverity::Medium | AuditSeverity::Low => {
                info!(
                    target: "audit",
                    event_id = %event.id,
                    event_type = ?event.event_type,
                    user_id = ?event.user_id,
                    outcome = ?event.outcome,
                    "{}", event_json
                );
            },
        }

        // Send event for async processing
        if self.event_sender.send(event).is_err() {
            return Err(AuditError::WriteError(
                "Failed to send event to background processor".to_string(),
            ));
        }

        Ok(())
    }

    pub fn subscribe_to_alerts(&self) -> broadcast::Receiver<AuditAlert> {
        self.alert_sender.subscribe()
    }

    pub async fn query_events(&self, query: AuditQuery) -> Result<Vec<AuditEvent>, AuditError> {
        use std::path::Path;
        use tokio::fs::File;
        use tokio::io::{AsyncBufReadExt, BufReader};

        if !self.config.enable_database_storage {
            return Ok(Vec::new());
        }

        // Determine storage path (same logic as store_events_to_database)
        let storage_path = if let Some(url) = &self.config.database_url {
            if url.starts_with("file://") {
                url.strip_prefix("file://").unwrap_or(url).to_string()
            } else if url.starts_with("sqlite://") {
                url.strip_prefix("sqlite://").unwrap_or("audit_events.db").to_string()
            } else {
                "audit_events.jsonl".to_string()
            }
        } else {
            "audit_events.jsonl".to_string()
        };

        // Check if file exists
        if !Path::new(&storage_path).exists() {
            return Ok(Vec::new());
        }

        // Read and filter events based on query
        let file = File::open(&storage_path)
            .await
            .map_err(|e| AuditError::StorageError(e.to_string()))?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        let mut matching_events = Vec::new();

        loop {
            let line = match lines
                .next_line()
                .await
                .map_err(|e| AuditError::StorageError(e.to_string()))?
            {
                Some(line) => line,
                None => break,
            };
            if let Ok(event) = serde_json::from_str::<AuditEvent>(&line) {
                // Apply filters based on query
                let mut matches = true;

                // Filter by time range
                if let Some(start_time) = &query.start_time {
                    if event.timestamp < *start_time {
                        matches = false;
                    }
                }
                if let Some(end_time) = &query.end_time {
                    if event.timestamp > *end_time {
                        matches = false;
                    }
                }

                // Filter by event types
                if let Some(ref event_types) = query.event_types {
                    if !event_types.is_empty() && !event_types.contains(&event.event_type) {
                        matches = false;
                    }
                }

                // Filter by severity
                if let Some(severity) = &query.severity {
                    if event.severity != *severity {
                        matches = false;
                    }
                }

                // Filter by user IDs
                if let Some(ref user_ids) = query.user_ids {
                    if !user_ids.is_empty() {
                        if let Some(ref event_user_id) = event.user_id {
                            if !user_ids.contains(event_user_id) {
                                matches = false;
                            }
                        } else {
                            matches = false;
                        }
                    }
                }

                // Filter by outcome
                if let Some(outcome) = &query.outcome {
                    if event.outcome != *outcome {
                        matches = false;
                    }
                }

                if matches {
                    matching_events.push(event);
                }
            }
        }

        // Apply limit if specified
        if let Some(limit) = query.limit {
            matching_events.truncate(limit as usize);
        }

        Ok(matching_events)
    }

    pub async fn generate_report(&self, query: AuditQuery) -> Result<AuditReport, AuditError> {
        // Query events based on the provided criteria
        let events = self.query_events(query).await?;

        // Initialize counters
        let mut events_by_type = HashMap::new();
        let mut events_by_severity = HashMap::new();
        let mut events_by_outcome = HashMap::new();
        let mut user_counts = HashMap::new();
        let mut ip_counts = HashMap::new();

        // Analyze events
        for event in &events {
            // Count by type
            *events_by_type.entry(event.event_type.clone()).or_insert(0) += 1;

            // Count by severity
            *events_by_severity.entry(event.severity.clone()).or_insert(0) += 1;

            // Count by outcome
            *events_by_outcome.entry(event.outcome.clone()).or_insert(0) += 1;

            // Count by user
            if let Some(user_id) = &event.user_id {
                *user_counts.entry(user_id.clone()).or_insert(0) += 1;
            }

            // Count by IP
            if let Some(ip) = &event.ip_address {
                *ip_counts.entry(ip.clone()).or_insert(0) += 1;
            }
        }

        // Get top users (sorted by event count, limited to top 10)
        let mut top_users: Vec<(String, u64)> =
            user_counts.into_iter().map(|(k, v)| (k, v as u64)).collect();
        top_users.sort_by(|a, b| b.1.cmp(&a.1));
        top_users.truncate(10);

        // Get top IPs (sorted by event count, limited to top 10)
        let mut top_ips: Vec<(String, u64)> =
            ip_counts.into_iter().map(|(k, v)| (k, v as u64)).collect();
        top_ips.sort_by(|a, b| b.1.cmp(&a.1));
        top_ips.truncate(10);

        Ok(AuditReport {
            total_events: events.len() as u64,
            events_by_type,
            events_by_severity,
            events_by_outcome,
            top_users,
            top_ips,
            generated_at: Utc::now(),
        })
    }

    pub async fn export_events(
        &self,
        query: AuditQuery,
        format: ExportFormat,
    ) -> Result<Vec<u8>, AuditError> {
        let events = self.query_events(query).await?;

        match format {
            ExportFormat::Json => {
                let json = serde_json::to_string_pretty(&events)
                    .map_err(|e| AuditError::ExportError(e.to_string()))?;
                Ok(json.into_bytes())
            },
            ExportFormat::Csv => {
                let mut csv_content = String::new();

                // Write CSV header
                csv_content
                    .push_str("id,timestamp,event_type,severity,user_id,session_id,ip_address,");
                csv_content.push_str(
                    "user_agent,resource,action,outcome,details,request_id,duration_ms\n",
                );

                // Write CSV rows
                for event in events {
                    let row = format!(
                        "{},{},{:?},{:?},{},{},{},{},{},{},{:?},{},{},{}\n",
                        escape_csv_field(&event.id),
                        event.timestamp.to_rfc3339(),
                        event.event_type,
                        event.severity,
                        escape_csv_field(&event.user_id.unwrap_or_default()),
                        escape_csv_field(&event.session_id.unwrap_or_default()),
                        escape_csv_field(&event.ip_address.unwrap_or_default()),
                        escape_csv_field(&event.user_agent.unwrap_or_default()),
                        escape_csv_field(&event.resource.unwrap_or_default()),
                        escape_csv_field(&event.action),
                        event.outcome,
                        escape_csv_field(&format_details_for_csv(&event.details)),
                        escape_csv_field(&event.request_id.unwrap_or_default()),
                        event.duration_ms.unwrap_or(0)
                    );
                    csv_content.push_str(&row);
                }

                Ok(csv_content.into_bytes())
            },
        }
    }
}

/// Helper function to escape CSV fields
fn escape_csv_field(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') || field.contains('\r') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

/// Helper function to format details HashMap for CSV
fn format_details_for_csv(details: &HashMap<String, String>) -> String {
    details
        .iter()
        .map(|(k, v)| format!("{}={}", k, v))
        .collect::<Vec<_>>()
        .join("; ")
}

impl AuditLogger {
    pub async fn cleanup_old_events(&self) -> Result<u64, AuditError> {
        use chrono::{Duration, Utc};
        use std::path::Path;
        use tokio::fs::File;
        use tokio::io::{AsyncBufReadExt, BufReader};

        if !self.config.enable_database_storage || self.config.retention_days == 0 {
            return Ok(0);
        }

        // Calculate cutoff date
        let cutoff_date = Utc::now() - Duration::days(self.config.retention_days as i64);

        // Determine storage path
        let storage_path = if let Some(url) = &self.config.database_url {
            if url.starts_with("file://") {
                url.strip_prefix("file://").unwrap_or(url).to_string()
            } else if url.starts_with("sqlite://") {
                url.strip_prefix("sqlite://").unwrap_or("audit_events.db").to_string()
            } else {
                url.clone()
            }
        } else {
            "audit_events.jsonl".to_string()
        };

        // Check if file exists
        if !Path::new(&storage_path).exists() {
            return Ok(0);
        }

        // Count events that would be removed
        let file = File::open(&storage_path)
            .await
            .map_err(|e| AuditError::StorageError(e.to_string()))?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        let mut removed_count = 0u64;

        while let Some(line) =
            lines.next_line().await.map_err(|e| AuditError::StorageError(e.to_string()))?
        {
            if let Ok(event) = serde_json::from_str::<AuditEvent>(&line) {
                if event.timestamp < cutoff_date {
                    removed_count += 1;
                }
            }
        }

        // Use the existing cleanup function
        if let Err(e) = Self::cleanup_old_audit_events(&self.config).await {
            return Err(AuditError::RetentionError(e.to_string()));
        }

        info!(
            "Cleaned up {} old audit events based on retention policy",
            removed_count
        );
        Ok(removed_count)
    }

    fn should_log_severity(&self, event_severity: &AuditSeverity) -> bool {
        match (&self.config.log_level, event_severity) {
            (AuditSeverity::Critical, AuditSeverity::Critical) => true,
            (AuditSeverity::High, AuditSeverity::Critical | AuditSeverity::High) => true,
            (
                AuditSeverity::Medium,
                AuditSeverity::Critical | AuditSeverity::High | AuditSeverity::Medium,
            ) => true,
            (AuditSeverity::Low, _) => true,
            _ => false,
        }
    }

    pub async fn log_authentication_attempt(
        &self,
        user_id: Option<String>,
        ip_address: Option<String>,
        user_agent: Option<String>,
        success: bool,
    ) -> Result<(), AuditError> {
        let event_type = if success {
            AuditEventType::LoginSuccess
        } else {
            AuditEventType::LoginFailure
        };

        let outcome = if success { AuditOutcome::Success } else { AuditOutcome::Failure };

        let severity = if success { AuditSeverity::Low } else { AuditSeverity::Medium };

        let mut event = AuditEvent::new(event_type, "User authentication attempt".to_string())
            .with_severity(severity)
            .with_outcome(outcome);

        if let Some(user_id) = user_id {
            event = event.with_user(user_id);
        }

        if let Some(ip) = ip_address {
            event = event.with_ip_address(ip);
        }

        if let Some(ua) = user_agent {
            event = event.with_user_agent(ua);
        }

        self.log_event(event)
    }

    pub async fn log_api_key_event(
        &self,
        event_type: AuditEventType,
        api_key_id: String,
        user_id: String,
        details: HashMap<String, String>,
    ) -> Result<(), AuditError> {
        let action = match event_type {
            AuditEventType::ApiKeyCreated => "API key created",
            AuditEventType::ApiKeyUsed => "API key used",
            AuditEventType::ApiKeyRevoked => "API key revoked",
            AuditEventType::ApiKeyExpired => "API key expired",
            _ => "API key operation",
        };

        let severity = match event_type {
            AuditEventType::ApiKeyRevoked => AuditSeverity::High,
            AuditEventType::ApiKeyExpired => AuditSeverity::Medium,
            _ => AuditSeverity::Low,
        };

        let event = AuditEvent::new(event_type, action.to_string())
            .with_severity(severity)
            .with_user(user_id)
            .with_resource(format!("api_key:{}", api_key_id))
            .with_details(details);

        self.log_event(event)
    }

    pub async fn log_inference_request(
        &self,
        user_id: Option<String>,
        model_name: String,
        request_id: String,
        duration_ms: u64,
        success: bool,
    ) -> Result<(), AuditError> {
        let outcome = if success { AuditOutcome::Success } else { AuditOutcome::Error };

        let mut event = AuditEvent::new(
            AuditEventType::InferenceRequest,
            "Model inference request".to_string(),
        )
        .with_severity(AuditSeverity::Low)
        .with_outcome(outcome)
        .with_resource(format!("model:{}", model_name))
        .with_request_id(request_id)
        .with_duration(duration_ms);

        if let Some(user_id) = user_id {
            event = event.with_user(user_id);
        }

        self.log_event(event)
    }

    pub async fn log_security_event(
        &self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        action: String,
        user_id: Option<String>,
        ip_address: Option<String>,
        details: HashMap<String, String>,
    ) -> Result<(), AuditError> {
        let mut event = AuditEvent::new(event_type, action)
            .with_severity(severity)
            .with_outcome(AuditOutcome::Failure);

        if let Some(user_id) = user_id {
            event = event.with_user(user_id);
        }

        if let Some(ip) = ip_address {
            event = event.with_ip_address(ip);
        }

        event = event.with_details(details);

        self.log_event(event)
    }

    pub async fn log_system_event(
        &self,
        event_type: AuditEventType,
        action: String,
        details: HashMap<String, String>,
    ) -> Result<(), AuditError> {
        let event = AuditEvent::new(event_type, action)
            .with_severity(AuditSeverity::Medium)
            .with_details(details);

        self.log_event(event)
    }

    pub async fn log_compliance_event(
        &self,
        event_type: AuditEventType,
        user_id: String,
        action: String,
        data_subject: Option<String>,
        legal_basis: Option<String>,
        details: HashMap<String, String>,
    ) -> Result<(), AuditError> {
        let mut event = AuditEvent::new(event_type, action)
            .with_severity(AuditSeverity::Medium)
            .with_user(user_id)
            .with_details(details);

        if let Some(subject) = data_subject {
            event = event.with_detail("data_subject".to_string(), subject);
        }

        if let Some(basis) = legal_basis {
            event = event.with_detail("legal_basis".to_string(), basis);
        }

        self.log_event(event)
    }

    pub async fn log_data_access_event(
        &self,
        user_id: String,
        resource: String,
        action: String,
        data_classification: Option<String>,
        purpose: Option<String>,
    ) -> Result<(), AuditError> {
        let mut details = HashMap::new();

        if let Some(classification) = data_classification {
            details.insert("data_classification".to_string(), classification);
        }

        if let Some(purpose) = purpose {
            details.insert("purpose".to_string(), purpose);
        }

        let event = AuditEvent::new(AuditEventType::DataAccessed, action)
            .with_severity(AuditSeverity::Medium)
            .with_user(user_id)
            .with_resource(resource)
            .with_details(details);

        self.log_event(event)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    Json,
    Csv,
}

// Compliance-related helper functions
impl AuditLogger {
    pub async fn get_user_activity_summary(
        &self,
        user_id: &str,
        days: u32,
    ) -> Result<HashMap<String, u64>, AuditError> {
        use chrono::{Duration, Utc};

        let start_date = Utc::now() - Duration::days(days as i64);

        let query = AuditQuery {
            start_time: Some(start_date),
            end_time: Some(Utc::now()),
            user_ids: Some(vec![user_id.to_string()]),
            ..Default::default()
        };

        let events = self.query_events(query).await?;
        let total_events = events.len() as u64;
        let mut summary = HashMap::new();

        // Track unique sessions and IPs
        let mut unique_sessions_set = std::collections::HashSet::new();
        let mut unique_ips_set = std::collections::HashSet::new();

        // Count events by type and collect unique sessions/IPs
        for event in events {
            let event_type_key = format!("{:?}", event.event_type);
            *summary.entry(event_type_key).or_insert(0) += 1;

            // Count by outcome
            let outcome_key = format!("outcome_{:?}", event.outcome);
            *summary.entry(outcome_key).or_insert(0) += 1;

            // Count by severity
            let severity_key = format!("severity_{:?}", event.severity);
            *summary.entry(severity_key).or_insert(0) += 1;

            // Collect unique sessions
            if let Some(session_id) = &event.session_id {
                unique_sessions_set.insert(session_id.clone());
            }

            // Collect unique IP addresses
            if let Some(ip_address) = &event.ip_address {
                unique_ips_set.insert(ip_address.clone());
            }
        }

        // Add total count
        summary.insert("total_events".to_string(), total_events);

        // Add unique sessions count
        let unique_sessions = unique_sessions_set.len() as u64;
        summary.insert("unique_sessions".to_string(), unique_sessions);

        // Add unique IP addresses count
        let unique_ips = unique_ips_set.len() as u64;
        summary.insert("unique_ip_addresses".to_string(), unique_ips);

        debug!(
            "Generated activity summary for user {} over {} days: {} events",
            user_id, days, total_events
        );
        Ok(summary)
    }

    pub async fn get_data_access_log(
        &self,
        resource: &str,
        days: u32,
    ) -> Result<Vec<AuditEvent>, AuditError> {
        use chrono::{Duration, Utc};

        let start_date = Utc::now() - Duration::days(days as i64);

        let query = AuditQuery {
            start_time: Some(start_date),
            end_time: Some(Utc::now()),
            event_types: Some(vec![
                AuditEventType::DataAccessed,
                AuditEventType::DataExported,
                AuditEventType::DataDeleted,
                AuditEventType::InferenceRequest,
                AuditEventType::BatchRequest,
                AuditEventType::StreamingRequest,
                AuditEventType::ModelRequest,
            ]),
            ..Default::default()
        };

        let mut events = self.query_events(query).await?;

        // Sort by timestamp descending (most recent first)
        events.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        info!(
            "Retrieved {} data access events for resource '{}' over {} days",
            events.len(),
            resource,
            days
        );
        Ok(events)
    }

    pub async fn verify_audit_integrity(&self) -> Result<bool, AuditError> {
        use std::collections::HashSet;
        use std::path::Path;
        use tokio::fs::File;
        use tokio::io::{AsyncBufReadExt, BufReader};

        if !self.config.enable_database_storage {
            return Ok(true); // No storage to verify
        }

        // Determine storage path
        let storage_path = if let Some(url) = &self.config.database_url {
            if url.starts_with("file://") {
                url.strip_prefix("file://").unwrap_or(url).to_string()
            } else if url.starts_with("sqlite://") {
                url.strip_prefix("sqlite://").unwrap_or("audit_events.db").to_string()
            } else {
                url.clone()
            }
        } else {
            "audit_events.jsonl".to_string()
        };

        // Check if file exists
        if !Path::new(&storage_path).exists() {
            return Ok(true); // No file to verify
        }

        let file = File::open(&storage_path)
            .await
            .map_err(|e| AuditError::StorageError(e.to_string()))?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let mut total_events = 0u64;
        let mut invalid_events = 0u64;
        let mut duplicate_ids = HashSet::new();
        let mut duplicates_found = 0u64;
        let mut temporal_inconsistencies = 0u64;
        let mut last_timestamp: Option<DateTime<Utc>> = None;

        while let Some(line) =
            lines.next_line().await.map_err(|e| AuditError::StorageError(e.to_string()))?
        {
            if line.trim().is_empty() {
                continue;
            }

            total_events += 1;

            // Try to parse the event
            match serde_json::from_str::<AuditEvent>(&line) {
                Ok(event) => {
                    // Check for duplicate IDs
                    if !duplicate_ids.insert(event.id.clone()) {
                        duplicates_found += 1;
                        warn!("Duplicate audit event ID found: {}", event.id);
                    }

                    // Check temporal consistency (events should be roughly chronological)
                    if let Some(last_ts) = last_timestamp {
                        if event.timestamp < last_ts - Duration::from_secs(3600) {
                            temporal_inconsistencies += 1;
                        }
                    }
                    last_timestamp = Some(event.timestamp);

                    // Validate event structure
                    if event.action.is_empty() {
                        invalid_events += 1;
                        warn!("Invalid audit event with empty action: {}", event.id);
                    }
                },
                Err(e) => {
                    invalid_events += 1;
                    error!("Failed to parse audit event: {}", e);
                },
            }
        }

        let integrity_ok = invalid_events == 0 && duplicates_found == 0;

        if !integrity_ok {
            warn!("Audit log integrity issues found: {} invalid events, {} duplicates, {} temporal inconsistencies out of {} total events",
                  invalid_events, duplicates_found, temporal_inconsistencies, total_events);
        } else {
            info!(
                "Audit log integrity verified: {} events checked successfully",
                total_events
            );
        }

        Ok(integrity_ok)
    }
}
