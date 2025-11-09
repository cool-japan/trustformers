use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::{broadcast, mpsc, RwLock};
use tracing::{debug, error, info, instrument, warn};
use tracing_subscriber::{
    fmt::{format::FmtSpan, layer as fmt_layer},
    layer::SubscriberExt,
    util::SubscriberInitExt,
    Layer,
};

#[derive(Debug, Error)]
pub enum LoggingError {
    #[error("Failed to initialize logging: {0}")]
    InitializationError(String),

    #[error("Failed to configure logging: {0}")]
    ConfigurationError(String),

    #[error("Failed to serialize log data: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Log forwarding error: {0}")]
    ForwardingError(String),

    #[error("Log aggregation error: {0}")]
    AggregationError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl From<LogLevel> for tracing::Level {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Trace => tracing::Level::TRACE,
            LogLevel::Debug => tracing::Level::DEBUG,
            LogLevel::Info => tracing::Level::INFO,
            LogLevel::Warn => tracing::Level::WARN,
            LogLevel::Error => tracing::Level::ERROR,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogFormat {
    Json,
    Pretty,
    Compact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: LogLevel,
    pub format: LogFormat,
    pub enable_colors: bool,
    pub enable_spans: bool,
    pub enable_targets: bool,
    pub enable_timestamps: bool,
    pub enable_thread_ids: bool,
    pub enable_thread_names: bool,
    pub file_output: Option<String>,
    pub max_file_size_mb: u64,
    pub max_files: u32,
    pub include_module_path: bool,
    pub include_line_numbers: bool,
    pub enable_log_forwarding: bool,
    pub log_forwarding_url: Option<String>,
    pub enable_log_aggregation: bool,
    pub aggregation_interval_seconds: u64,
    pub buffer_size: usize,
    pub enable_correlation_ids: bool,
    pub environment: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            format: LogFormat::Json,
            enable_colors: false,
            enable_spans: true,
            enable_targets: true,
            enable_timestamps: true,
            enable_thread_ids: false,
            enable_thread_names: false,
            file_output: None,
            max_file_size_mb: 100,
            max_files: 10,
            include_module_path: true,
            include_line_numbers: false,
            enable_log_forwarding: false,
            log_forwarding_url: None,
            enable_log_aggregation: false,
            aggregation_interval_seconds: 60,
            buffer_size: 1000,
            enable_correlation_ids: true,
            environment: "development".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct RequestContext {
    pub request_id: String,
    pub user_id: Option<String>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
    pub method: String,
    pub path: String,
    pub query_params: HashMap<String, String>,
    pub started_at: chrono::DateTime<chrono::Utc>,
}

impl RequestContext {
    pub fn new(request_id: String, method: String, path: String) -> Self {
        Self {
            request_id,
            user_id: None,
            ip_address: None,
            user_agent: None,
            method,
            path,
            query_params: HashMap::new(),
            started_at: chrono::Utc::now(),
        }
    }

    pub fn with_user(mut self, user_id: String) -> Self {
        self.user_id = Some(user_id);
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

    pub fn with_query_params(mut self, params: HashMap<String, String>) -> Self {
        self.query_params = params;
        self
    }

    pub fn elapsed_ms(&self) -> u64 {
        let elapsed = chrono::Utc::now() - self.started_at;
        elapsed.num_milliseconds() as u64
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct PerformanceMetrics {
    pub request_id: String,
    pub operation: String,
    pub duration_ms: u64,
    pub cpu_usage_percent: Option<f64>,
    pub memory_usage_bytes: Option<u64>,
    pub gpu_memory_usage_bytes: Option<u64>,
    pub model_name: Option<String>,
    pub batch_size: Option<usize>,
    pub sequence_length: Option<usize>,
    pub tokens_per_second: Option<f64>,
}

impl PerformanceMetrics {
    pub fn new(request_id: String, operation: String, duration_ms: u64) -> Self {
        Self {
            request_id,
            operation,
            duration_ms,
            cpu_usage_percent: None,
            memory_usage_bytes: None,
            gpu_memory_usage_bytes: None,
            model_name: None,
            batch_size: None,
            sequence_length: None,
            tokens_per_second: None,
        }
    }

    pub fn with_resource_usage(
        mut self,
        cpu_percent: Option<f64>,
        memory_bytes: Option<u64>,
        gpu_memory_bytes: Option<u64>,
    ) -> Self {
        self.cpu_usage_percent = cpu_percent;
        self.memory_usage_bytes = memory_bytes;
        self.gpu_memory_usage_bytes = gpu_memory_bytes;
        self
    }

    pub fn with_model_info(
        mut self,
        model_name: String,
        batch_size: Option<usize>,
        sequence_length: Option<usize>,
    ) -> Self {
        self.model_name = Some(model_name);
        self.batch_size = batch_size;
        self.sequence_length = sequence_length;
        self
    }

    pub fn with_throughput(mut self, tokens_per_second: f64) -> Self {
        self.tokens_per_second = Some(tokens_per_second);
        self
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorContext {
    pub error_id: String,
    pub request_id: Option<String>,
    pub error_type: String,
    pub error_message: String,
    pub stack_trace: Option<String>,
    pub user_id: Option<String>,
    pub operation: String,
    pub severity: String,
    pub recoverable: bool,
    pub metadata: HashMap<String, String>,
}

impl ErrorContext {
    pub fn new(error_type: String, error_message: String, operation: String) -> Self {
        Self {
            error_id: uuid::Uuid::new_v4().to_string(),
            request_id: None,
            error_type,
            error_message,
            stack_trace: None,
            user_id: None,
            operation,
            severity: "error".to_string(),
            recoverable: false,
            metadata: HashMap::new(),
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

    pub fn with_stack_trace(mut self, stack_trace: String) -> Self {
        self.stack_trace = Some(stack_trace);
        self
    }

    pub fn with_severity(mut self, severity: String) -> Self {
        self.severity = severity;
        self
    }

    pub fn with_recoverable(mut self, recoverable: bool) -> Self {
        self.recoverable = recoverable;
        self
    }

    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }
}

pub struct StructuredLogger {
    config: LoggingConfig,
    log_buffer: Arc<RwLock<Vec<LogEntry>>>,
    log_sender: mpsc::UnboundedSender<LogEntry>,
    alert_sender: broadcast::Sender<LogAlert>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LogEntry {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub level: String,
    pub target: String,
    pub message: String,
    pub fields: HashMap<String, serde_json::Value>,
    pub span_id: Option<String>,
    pub trace_id: Option<String>,
    pub correlation_id: Option<String>,
    pub environment: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct LogAlert {
    pub alert_id: String,
    pub severity: String,
    pub message: String,
    pub source: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub context: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LogMetrics {
    pub total_logs: u64,
    pub logs_by_level: HashMap<String, u64>,
    pub logs_by_target: HashMap<String, u64>,
    pub errors_per_minute: f64,
    pub warnings_per_minute: f64,
    pub average_log_size_bytes: f64,
    pub buffer_usage_percent: f64,
}

impl StructuredLogger {
    pub fn new(config: LoggingConfig) -> Result<Self, LoggingError> {
        let (log_sender, log_receiver) = mpsc::unbounded_channel();
        let (alert_sender, _) = broadcast::channel(1000);

        let logger = Self {
            config,
            log_buffer: Arc::new(RwLock::new(Vec::new())),
            log_sender,
            alert_sender,
        };

        logger.initialize()?;

        // Start background log processing
        logger.start_background_processing(log_receiver);

        Ok(logger)
    }

    fn start_background_processing(&self, mut log_receiver: mpsc::UnboundedReceiver<LogEntry>) {
        let config = self.config.clone();
        let log_buffer = Arc::clone(&self.log_buffer);
        let alert_sender = self.alert_sender.clone();

        tokio::spawn(async move {
            let mut aggregation_interval = tokio::time::interval(std::time::Duration::from_secs(
                config.aggregation_interval_seconds,
            ));

            loop {
                tokio::select! {
                    // Process incoming log entries
                    log_entry = log_receiver.recv() => {
                        match log_entry {
                            Some(entry) => {
                                // Check for alert conditions
                                if Self::should_alert(&entry) {
                                    let alert = LogAlert {
                                        alert_id: uuid::Uuid::new_v4().to_string(),
                                        severity: entry.level.clone(),
                                        message: entry.message.clone(),
                                        source: entry.target.clone(),
                                        timestamp: entry.timestamp,
                                        context: entry.fields.iter().map(|(k, v)| (k.clone(), v.to_string())).collect(),
                                    };

                                    let _ = alert_sender.send(alert);
                                }

                                // Add to buffer if aggregation is enabled
                                if config.enable_log_aggregation {
                                    let mut buffer = log_buffer.write().await;
                                    buffer.push(entry);

                                    // Flush buffer if it's full
                                    if buffer.len() >= config.buffer_size {
                                        let entries = buffer.drain(..).collect::<Vec<_>>();
                                        drop(buffer);
                                        Self::process_log_entries(&config, entries).await;
                                    }
                                }
                            }
                            None => break,
                        }
                    }

                    // Periodic aggregation flush
                    _ = aggregation_interval.tick() => {
                        if config.enable_log_aggregation {
                            let mut buffer = log_buffer.write().await;
                            if !buffer.is_empty() {
                                let entries = buffer.drain(..).collect::<Vec<_>>();
                                drop(buffer);
                                Self::process_log_entries(&config, entries).await;
                            }
                        }
                    }
                }
            }
        });
    }

    fn should_alert(entry: &LogEntry) -> bool {
        matches!(entry.level.as_str(), "ERROR" | "WARN") && entry.target.contains("security")
    }

    async fn process_log_entries(config: &LoggingConfig, entries: Vec<LogEntry>) {
        if config.enable_log_forwarding {
            if let Some(ref forwarding_url) = config.log_forwarding_url {
                match Self::forward_logs(forwarding_url, &entries).await {
                    Ok(_) => {
                        tracing::debug!(
                            "Successfully forwarded {} log entries to {}",
                            entries.len(),
                            forwarding_url
                        );
                    },
                    Err(e) => {
                        tracing::error!("Failed to forward logs to {}: {}", forwarding_url, e);
                    },
                }
            } else {
                tracing::warn!("Log forwarding enabled but no forwarding URL configured");
            }
        }
    }

    /// Forward logs to external logging service
    async fn forward_logs(url: &str, entries: &[LogEntry]) -> Result<(), LoggingError> {
        use std::time::Duration;

        // Create HTTP client with timeout
        let client =
            reqwest::Client::builder()
                .timeout(Duration::from_secs(10))
                .build()
                .map_err(|e| {
                    LoggingError::ForwardingError(format!("Failed to create HTTP client: {}", e))
                })?;

        // Prepare log payload
        let payload = serde_json::json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "service": "trustformers-serve",
            "version": env!("CARGO_PKG_VERSION"),
            "environment": std::env::var("ENVIRONMENT").unwrap_or_else(|_| "development".to_string()),
            "logs": entries
        });

        // Send POST request to forwarding endpoint
        let response = client
            .post(url)
            .json(&payload)
            .header("Content-Type", "application/json")
            .header(
                "User-Agent",
                format!("trustformers-serve/{}", env!("CARGO_PKG_VERSION")),
            )
            .send()
            .await
            .map_err(|e| LoggingError::ForwardingError(format!("Failed to send logs: {}", e)))?;

        // Check response status
        if response.status().is_success() {
            tracing::debug!("Log forwarding successful, status: {}", response.status());
            Ok(())
        } else {
            let status = response.status();
            let error_body = response.text().await.unwrap_or_else(|_| "Unknown error".to_string());
            Err(LoggingError::ForwardingError(format!(
                "Log forwarding failed with status {}: {}",
                status, error_body
            )))
        }
    }

    pub fn initialize(&self) -> Result<(), LoggingError> {
        let level: tracing::Level = self.config.level.clone().into();

        let registry = tracing_subscriber::registry();

        match self.config.format {
            LogFormat::Json => {
                let fmt_layer = fmt_layer()
                    .json()
                    .with_span_events(if self.config.enable_spans {
                        FmtSpan::FULL
                    } else {
                        FmtSpan::NONE
                    })
                    .with_target(self.config.enable_targets)
                    .with_thread_ids(self.config.enable_thread_ids)
                    .with_thread_names(self.config.enable_thread_names);

                registry
                    .with(
                        fmt_layer.with_filter(tracing_subscriber::filter::LevelFilter::from(level)),
                    )
                    .init();
            },
            LogFormat::Pretty => {
                let fmt_layer = fmt_layer()
                    .pretty()
                    .with_ansi(self.config.enable_colors)
                    .with_span_events(if self.config.enable_spans {
                        FmtSpan::FULL
                    } else {
                        FmtSpan::NONE
                    })
                    .with_target(self.config.enable_targets)
                    .with_thread_ids(self.config.enable_thread_ids)
                    .with_thread_names(self.config.enable_thread_names);

                registry
                    .with(
                        fmt_layer.with_filter(tracing_subscriber::filter::LevelFilter::from(level)),
                    )
                    .init();
            },
            LogFormat::Compact => {
                let fmt_layer = fmt_layer()
                    .compact()
                    .with_ansi(self.config.enable_colors)
                    .with_span_events(if self.config.enable_spans {
                        FmtSpan::FULL
                    } else {
                        FmtSpan::NONE
                    })
                    .with_target(self.config.enable_targets)
                    .with_thread_ids(self.config.enable_thread_ids)
                    .with_thread_names(self.config.enable_thread_names);

                registry
                    .with(
                        fmt_layer.with_filter(tracing_subscriber::filter::LevelFilter::from(level)),
                    )
                    .init();
            },
        }

        Ok(())
    }

    #[instrument(skip(self, context))]
    pub fn log_request_start(&self, context: &RequestContext) {
        info!(
            target: "http_request",
            request_id = %context.request_id,
            user_id = ?context.user_id,
            method = %context.method,
            path = %context.path,
            ip_address = ?context.ip_address,
            user_agent = ?context.user_agent,
            "Request started"
        );
    }

    #[instrument(skip(self, context))]
    pub fn log_request_end(
        &self,
        context: &RequestContext,
        status_code: u16,
        response_size: Option<usize>,
    ) {
        let duration_ms = context.elapsed_ms();

        info!(
            target: "http_request",
            request_id = %context.request_id,
            user_id = ?context.user_id,
            method = %context.method,
            path = %context.path,
            status_code = status_code,
            duration_ms = duration_ms,
            response_size = ?response_size,
            "Request completed"
        );
    }

    #[instrument(skip(self, metrics))]
    pub fn log_performance_metrics(&self, metrics: &PerformanceMetrics) {
        info!(
            target: "performance",
            request_id = %metrics.request_id,
            operation = %metrics.operation,
            duration_ms = metrics.duration_ms,
            cpu_usage_percent = ?metrics.cpu_usage_percent,
            memory_usage_bytes = ?metrics.memory_usage_bytes,
            gpu_memory_usage_bytes = ?metrics.gpu_memory_usage_bytes,
            model_name = ?metrics.model_name,
            batch_size = ?metrics.batch_size,
            sequence_length = ?metrics.sequence_length,
            tokens_per_second = ?metrics.tokens_per_second,
            "Performance metrics"
        );
    }

    #[instrument(skip(self, error_context))]
    pub fn log_error(&self, error_context: &ErrorContext) {
        match error_context.severity.as_str() {
            "critical" => {
                error!(
                    target: "application_error",
                    error_id = %error_context.error_id,
                    request_id = ?error_context.request_id,
                    error_type = %error_context.error_type,
                    operation = %error_context.operation,
                    user_id = ?error_context.user_id,
                    recoverable = error_context.recoverable,
                    stack_trace = ?error_context.stack_trace,
                    metadata = ?error_context.metadata,
                    "Critical error: {}", error_context.error_message
                );
            },
            "warn" => {
                warn!(
                    target: "application_error",
                    error_id = %error_context.error_id,
                    request_id = ?error_context.request_id,
                    error_type = %error_context.error_type,
                    operation = %error_context.operation,
                    user_id = ?error_context.user_id,
                    recoverable = error_context.recoverable,
                    metadata = ?error_context.metadata,
                    "Warning: {}", error_context.error_message
                );
            },
            _ => {
                error!(
                    target: "application_error",
                    error_id = %error_context.error_id,
                    request_id = ?error_context.request_id,
                    error_type = %error_context.error_type,
                    operation = %error_context.operation,
                    user_id = ?error_context.user_id,
                    recoverable = error_context.recoverable,
                    stack_trace = ?error_context.stack_trace,
                    metadata = ?error_context.metadata,
                    "Error: {}", error_context.error_message
                );
            },
        }
    }

    #[instrument(skip(self))]
    pub fn log_security_event(
        &self,
        event_type: &str,
        user_id: Option<&str>,
        details: &HashMap<String, String>,
    ) {
        warn!(
            target: "security",
            event_type = event_type,
            user_id = ?user_id,
            details = ?details,
            "Security event"
        );
    }

    #[instrument(skip(self))]
    pub fn log_model_operation(
        &self,
        operation: &str,
        model_name: &str,
        duration_ms: u64,
        success: bool,
    ) {
        info!(
            target: "model_operation",
            operation = operation,
            model_name = model_name,
            duration_ms = duration_ms,
            success = success,
            "Model operation"
        );
    }

    #[instrument(skip(self))]
    pub fn log_cache_operation(
        &self,
        operation: &str,
        cache_type: &str,
        hit: bool,
        latency_ms: Option<u64>,
    ) {
        debug!(
            target: "cache",
            operation = operation,
            cache_type = cache_type,
            hit = hit,
            latency_ms = ?latency_ms,
            "Cache operation"
        );
    }

    #[instrument(skip(self))]
    pub fn log_inference_metrics(
        &self,
        request_id: &str,
        model_name: &str,
        tokens_generated: usize,
        duration_ms: u64,
        batch_size: usize,
    ) {
        let tokens_per_second = if duration_ms > 0 {
            (tokens_generated as f64 * 1000.0) / duration_ms as f64
        } else {
            0.0
        };

        info!(
            target: "inference",
            request_id = request_id,
            model_name = model_name,
            tokens_generated = tokens_generated,
            duration_ms = duration_ms,
            batch_size = batch_size,
            tokens_per_second = tokens_per_second,
            "Inference completed"
        );
    }

    pub fn subscribe_to_alerts(&self) -> broadcast::Receiver<LogAlert> {
        self.alert_sender.subscribe()
    }

    pub async fn get_log_metrics(&self) -> LogMetrics {
        let buffer = self.log_buffer.read().await;
        let buffer_len = buffer.len();

        // Calculate metrics from buffer
        let mut logs_by_level = HashMap::new();
        let mut logs_by_target = HashMap::new();
        let mut total_size = 0;

        // Calculate time-based metrics
        let now = chrono::Utc::now();
        let one_hour_ago = now - chrono::Duration::minutes(60);
        let mut recent_errors = 0;
        let mut recent_warnings = 0;

        for entry in buffer.iter() {
            *logs_by_level.entry(entry.level.clone()).or_insert(0) += 1;
            *logs_by_target.entry(entry.target.clone()).or_insert(0) += 1;
            total_size += entry.message.len()
                + entry.fields.iter().map(|(k, v)| k.len() + v.to_string().len()).sum::<usize>();

            // Count recent errors and warnings for rate calculation
            if entry.timestamp > one_hour_ago {
                match entry.level.as_str() {
                    "ERROR" => recent_errors += 1,
                    "WARN" => recent_warnings += 1,
                    _ => {},
                }
            }
        }

        let errors_per_minute = recent_errors as f64 / 60.0;
        let warnings_per_minute = recent_warnings as f64 / 60.0;

        LogMetrics {
            total_logs: buffer_len as u64,
            logs_by_level,
            logs_by_target,
            errors_per_minute,
            warnings_per_minute,
            average_log_size_bytes: if buffer_len > 0 {
                total_size as f64 / buffer_len as f64
            } else {
                0.0
            },
            buffer_usage_percent: (buffer_len as f64 / self.config.buffer_size as f64) * 100.0,
        }
    }

    #[instrument(skip(self))]
    pub fn log_structured_event(
        &self,
        level: &str,
        target: &str,
        message: &str,
        fields: HashMap<String, serde_json::Value>,
    ) {
        let entry = LogEntry {
            timestamp: chrono::Utc::now(),
            level: level.to_string(),
            target: target.to_string(),
            message: message.to_string(),
            fields,
            span_id: Self::extract_span_id(),
            trace_id: Self::extract_trace_id(),
            correlation_id: if self.config.enable_correlation_ids {
                Some(uuid::Uuid::new_v4().to_string())
            } else {
                None
            },
            environment: self.config.environment.clone(),
        };

        // Send to background processor
        let _ = self.log_sender.send(entry);
    }

    #[instrument(skip(self))]
    pub fn log_business_event(
        &self,
        event_type: &str,
        entity_id: &str,
        action: &str,
        details: HashMap<String, serde_json::Value>,
    ) {
        let mut fields = details;
        fields.insert(
            "event_type".to_string(),
            serde_json::Value::String(event_type.to_string()),
        );
        fields.insert(
            "entity_id".to_string(),
            serde_json::Value::String(entity_id.to_string()),
        );
        fields.insert(
            "action".to_string(),
            serde_json::Value::String(action.to_string()),
        );

        info!(
            target: "business",
            event_type = event_type,
            entity_id = entity_id,
            action = action,
            details = ?fields,
            "Business event"
        );

        self.log_structured_event("INFO", "business", "Business event", fields);
    }

    #[instrument(skip(self))]
    pub fn log_system_health(&self, component: &str, status: &str, metrics: HashMap<String, f64>) {
        let mut fields = HashMap::new();
        fields.insert(
            "component".to_string(),
            serde_json::Value::String(component.to_string()),
        );
        fields.insert(
            "status".to_string(),
            serde_json::Value::String(status.to_string()),
        );

        for (key, value) in &metrics {
            fields.insert(
                key.clone(),
                serde_json::Value::Number(serde_json::Number::from_f64(*value).unwrap()),
            );
        }

        info!(
            target: "system_health",
            component = component,
            status = status,
            metrics = ?metrics,
            "System health check"
        );

        self.log_structured_event("INFO", "system_health", "System health check", fields);
    }

    #[instrument(skip(self))]
    pub fn log_compliance_event(
        &self,
        regulation: &str,
        action: &str,
        subject: &str,
        details: HashMap<String, String>,
    ) {
        let mut fields = HashMap::new();
        fields.insert(
            "regulation".to_string(),
            serde_json::Value::String(regulation.to_string()),
        );
        fields.insert(
            "action".to_string(),
            serde_json::Value::String(action.to_string()),
        );
        fields.insert(
            "subject".to_string(),
            serde_json::Value::String(subject.to_string()),
        );

        for (key, value) in details {
            fields.insert(key, serde_json::Value::String(value));
        }

        info!(
            target: "compliance",
            regulation = regulation,
            action = action,
            subject = subject,
            "Compliance event"
        );

        self.log_structured_event("INFO", "compliance", "Compliance event", fields);
    }

    pub async fn flush_logs(&self) -> Result<(), LoggingError> {
        if self.config.enable_log_aggregation {
            let mut buffer = self.log_buffer.write().await;
            if !buffer.is_empty() {
                let entries = buffer.drain(..).collect::<Vec<_>>();
                drop(buffer);
                Self::process_log_entries(&self.config, entries).await;
            }
        }
        Ok(())
    }

    /// Extract span ID from current tracing context
    fn extract_span_id() -> Option<String> {
        tracing::Span::current().id().map(|id| format!("{:?}", id))
    }

    /// Extract trace ID from current tracing context
    fn extract_trace_id() -> Option<String> {
        // Try to extract trace ID from span metadata
        let current_span = tracing::Span::current();
        if current_span != tracing::Span::none() {
            // Generate a consistent trace ID based on the root span
            // This is a simplified implementation - in a real system you might use OpenTelemetry
            let span_name = current_span.metadata().map(|m| m.name()).unwrap_or("unknown");
            let span_target = current_span.metadata().map(|m| m.target()).unwrap_or("unknown");

            // Create a deterministic trace ID from span information
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            span_name.hash(&mut hasher);
            span_target.hash(&mut hasher);

            // Include current time to make it unique per execution
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            now.hash(&mut hasher);

            Some(format!("trace-{:x}", hasher.finish()))
        } else {
            None
        }
    }
}

// Helper macros for structured logging
#[macro_export]
macro_rules! log_request_start {
    ($logger:expr, $context:expr) => {
        $logger.log_request_start($context)
    };
}

#[macro_export]
macro_rules! log_request_end {
    ($logger:expr, $context:expr, $status:expr, $size:expr) => {
        $logger.log_request_end($context, $status, $size)
    };
}

#[macro_export]
macro_rules! log_performance {
    ($logger:expr, $metrics:expr) => {
        $logger.log_performance_metrics($metrics)
    };
}

#[macro_export]
macro_rules! log_structured_error {
    ($logger:expr, $error_context:expr) => {
        $logger.log_error($error_context)
    };
}
