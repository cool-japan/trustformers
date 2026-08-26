//! # Notification Channels for Delivery
//!
//! Channel implementations used by the notification manager.
//!
//! ## Delivery is real or it is an error
//!
//! Every channel in this module either performs the delivery it advertises or
//! returns a structured error. None of them fabricates a delivery outcome.
//!
//! * [`LogNotificationChannel`] writes the notification through `tracing`. It
//!   always succeeds because logging is the delivery.
//! * [`WebhookNotificationChannel`], [`SlackNotificationChannel`] and
//!   [`PagerDutyNotificationChannel`] issue a real HTTP request through
//!   [`reqwest`] and report the transport's own status code, latency and error
//!   text. Constructed without an endpoint they refuse to send rather than
//!   pretend to.
//! * [`EmailNotificationChannel`] and [`SmsNotificationChannel`] have no
//!   transport: SMTP and the SMS gateway APIs need dependencies this crate does
//!   not declare. They return [`ChannelError::TransportUnavailable`] from every
//!   send, and report themselves unhealthy.
//!
//! Channel health is derived from [`ChannelCounters`], which counts what the
//! channel actually did. `success_rate` is `successes / attempts`, not a
//! constant, and a channel that has never been used reports no health history
//! rather than inventing one.

use super::health_monitor::ChannelHealth;
use super::types::*;
use anyhow::Result;
use chrono::Utc;
use std::fmt::Debug;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};
use tracing::{debug, error, info, warn};

/// Errors returned by notification channels.
#[derive(Debug, thiserror::Error)]
pub enum ChannelError {
    /// The channel was constructed without the endpoint it needs to deliver.
    #[error("{channel} notification channel has no endpoint configured; construct it with `{channel}NotificationChannel::with_endpoint`")]
    NotConfigured {
        /// Channel name, used verbatim in the message.
        channel: &'static str,
    },

    /// The channel has no transport implementation in this crate.
    #[error("{channel} notification delivery is not implemented: {reason}")]
    TransportUnavailable {
        /// Channel name.
        channel: &'static str,
        /// Why no transport exists.
        reason: &'static str,
    },
}

/// Counters describing what a channel actually did.
///
/// Health checks read these instead of returning a hard-coded success rate.
#[derive(Debug, Default)]
pub struct ChannelCounters {
    attempts: AtomicU64,
    successes: AtomicU64,
    consecutive_failures: AtomicU64,
}

impl ChannelCounters {
    /// Record a delivery attempt and its outcome.
    pub fn record(&self, success: bool) {
        self.attempts.fetch_add(1, Ordering::Relaxed);
        if success {
            self.successes.fetch_add(1, Ordering::Relaxed);
            self.consecutive_failures.store(0, Ordering::Relaxed);
        } else {
            self.consecutive_failures.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Number of delivery attempts made so far.
    pub fn attempts(&self) -> u64 {
        self.attempts.load(Ordering::Relaxed)
    }

    /// Number of attempts that succeeded.
    pub fn successes(&self) -> u64 {
        self.successes.load(Ordering::Relaxed)
    }

    /// Failures since the last success.
    pub fn consecutive_failures(&self) -> u64 {
        self.consecutive_failures.load(Ordering::Relaxed)
    }

    /// Observed success rate, or `None` when nothing has been attempted.
    ///
    /// Returning `None` matters: a channel that has never delivered anything
    /// has no measured success rate, and reporting one would be a fabrication.
    pub fn success_rate(&self) -> Option<f32> {
        let attempts = self.attempts();
        if attempts == 0 {
            None
        } else {
            Some(self.successes() as f32 / attempts as f32)
        }
    }

    /// Build a [`ChannelHealth`] from the observed counters.
    ///
    /// `reachable` is the outcome of whatever liveness probe the channel just
    /// performed; `success_rate` falls back to `reachable` when no delivery has
    /// been attempted yet, so an unused-but-reachable channel reads as 1.0 and
    /// an unusable one as 0.0.
    pub fn to_health(&self, reachable: bool) -> ChannelHealth {
        let consecutive_failures = self.consecutive_failures();
        ChannelHealth {
            healthy: reachable && consecutive_failures == 0,
            last_check: Utc::now(),
            consecutive_failures: consecutive_failures as usize,
            success_rate: self.success_rate().unwrap_or(if reachable { 1.0 } else { 0.0 }),
        }
    }
}

/// Endpoint configuration shared by the HTTP-delivered channels.
#[derive(Debug, Clone)]
pub struct HttpChannelConfig {
    /// Absolute URL the notification is POSTed to.
    pub endpoint: String,

    /// Per-request timeout.
    pub timeout: Duration,

    /// Optional value for the `Authorization` header.
    pub auth_header: Option<String>,

    /// Additional headers sent with every request.
    pub extra_headers: HashMap<String, String>,
}

impl HttpChannelConfig {
    /// Configuration pointing at `endpoint` with a 30 second timeout.
    pub fn new(endpoint: impl Into<String>) -> Self {
        Self {
            endpoint: endpoint.into(),
            timeout: Duration::from_secs(30),
            auth_header: None,
            extra_headers: HashMap::new(),
        }
    }

    /// Set the `Authorization` header value.
    pub fn with_auth_header(mut self, value: impl Into<String>) -> Self {
        self.auth_header = Some(value.into());
        self
    }

    /// Set the per-request timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Add an extra header sent with every request.
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.insert(name.into(), value.into());
        self
    }
}

/// POST `body` to the configured endpoint and turn the transport outcome into a
/// [`DeliveryResult`].
///
/// Every field of the returned result comes from the request that was actually
/// made: `latency_ms` is measured, `status_code` is the server's, and `error`
/// carries the transport or HTTP error text.
async fn post_json(
    client: &reqwest::Client,
    config: &HttpChannelConfig,
    body: serde_json::Value,
    counters: &ChannelCounters,
) -> DeliveryResult {
    let start_time = Instant::now();

    let mut request = client.post(&config.endpoint).timeout(config.timeout).json(&body);
    if let Some(auth) = &config.auth_header {
        request = request.header(reqwest::header::AUTHORIZATION, auth);
    }
    for (name, value) in &config.extra_headers {
        request = request.header(name, value);
    }

    let mut response_data = HashMap::new();
    match request.send().await {
        Ok(response) => {
            let status = response.status();
            response_data.insert("status_code".to_string(), status.as_u16().to_string());
            let latency_ms = start_time.elapsed().as_millis() as u64;
            let body_text = response.text().await.unwrap_or_default();
            if !body_text.is_empty() {
                response_data.insert("response_body".to_string(), body_text.clone());
            }
            let success = status.is_success();
            counters.record(success);
            DeliveryResult {
                success,
                delivered_at: success.then(Utc::now),
                attempts: 1,
                error: (!success)
                    .then(|| format!("endpoint returned HTTP {}: {}", status, body_text)),
                latency_ms: Some(latency_ms),
                response_data,
            }
        },
        Err(error) => {
            counters.record(false);
            DeliveryResult {
                success: false,
                delivered_at: None,
                attempts: 1,
                error: Some(error.to_string()),
                latency_ms: Some(start_time.elapsed().as_millis() as u64),
                response_data,
            }
        },
    }
}

/// Probe an endpoint's reachability with a real request.
async fn probe(client: &reqwest::Client, config: &HttpChannelConfig) -> bool {
    client.head(&config.endpoint).timeout(config.timeout).send().await.is_ok()
}

/// Enhanced trait for notification channels with async support and health monitoring
#[async_trait::async_trait]
pub trait NotificationChannel: Debug {
    /// Send a notification through this channel
    async fn send_notification(&self, notification: &Notification) -> Result<DeliveryResult>;

    /// Get channel name
    fn name(&self) -> &str;

    /// Check if channel supports the given notification type
    fn supports(&self, notification_type: &str) -> bool;

    /// Perform health check
    async fn health_check(&self) -> Result<ChannelHealth>;

    /// Get channel capabilities
    fn capabilities(&self) -> Vec<String> {
        vec!["basic".to_string()]
    }

    /// Get maximum message size
    fn max_message_size(&self) -> Option<usize> {
        None
    }
}

/// Log notification channel implementation.
///
/// Writing the record through `tracing` *is* the delivery, so this channel is
/// the only one that can report success without a transport.
#[derive(Debug)]
pub struct LogNotificationChannel {
    name: String,
    counters: Arc<ChannelCounters>,
}

impl LogNotificationChannel {
    /// Create the log channel.
    pub async fn new() -> Result<Self> {
        Ok(Self {
            name: "log".to_string(),
            counters: Arc::new(ChannelCounters::default()),
        })
    }
}

#[async_trait::async_trait]
impl NotificationChannel for LogNotificationChannel {
    async fn send_notification(&self, notification: &Notification) -> Result<DeliveryResult> {
        let start_time = Instant::now();

        // Log the notification
        match notification.priority {
            NotificationPriority::Emergency | NotificationPriority::Critical => {
                error!(
                    "CRITICAL ALERT: {} - {} (ID: {})",
                    notification.subject, notification.content, notification.id
                );
            },
            NotificationPriority::High => {
                warn!(
                    "HIGH PRIORITY: {} - {} (ID: {})",
                    notification.subject, notification.content, notification.id
                );
            },
            NotificationPriority::Normal => {
                info!(
                    "NOTIFICATION: {} - {} (ID: {})",
                    notification.subject, notification.content, notification.id
                );
            },
            NotificationPriority::Low => {
                debug!(
                    "INFO: {} - {} (ID: {})",
                    notification.subject, notification.content, notification.id
                );
            },
        }

        let latency_ms = start_time.elapsed().as_millis() as u64;
        self.counters.record(true);

        Ok(DeliveryResult {
            success: true,
            delivered_at: Some(Utc::now()),
            attempts: 1,
            error: None,
            latency_ms: Some(latency_ms),
            response_data: {
                let mut data = HashMap::new();
                data.insert("logged_at".to_string(), Utc::now().to_rfc3339());
                data.insert(
                    "log_level".to_string(),
                    match notification.priority {
                        NotificationPriority::Emergency | NotificationPriority::Critical => {
                            "ERROR".to_string()
                        },
                        NotificationPriority::High => "WARN".to_string(),
                        NotificationPriority::Normal => "INFO".to_string(),
                        NotificationPriority::Low => "DEBUG".to_string(),
                    },
                );
                data
            },
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supports(&self, notification_type: &str) -> bool {
        notification_type == "log" || notification_type == "*"
    }

    async fn health_check(&self) -> Result<ChannelHealth> {
        Ok(self.counters.to_health(true))
    }

    fn capabilities(&self) -> Vec<String> {
        vec![
            "basic".to_string(),
            "always_available".to_string(),
            "immediate".to_string(),
        ]
    }
}

/// Email notification channel.
///
/// This crate declares no SMTP client, so the channel has no way to deliver
/// mail. Every send returns [`ChannelError::TransportUnavailable`] instead of
/// reporting a delivery that did not happen.
#[derive(Debug)]
pub struct EmailNotificationChannel {
    name: String,
}

impl EmailNotificationChannel {
    /// Create the email channel.
    pub async fn new() -> Result<Self> {
        Ok(Self {
            name: "email".to_string(),
        })
    }
}

#[async_trait::async_trait]
impl NotificationChannel for EmailNotificationChannel {
    async fn send_notification(&self, _notification: &Notification) -> Result<DeliveryResult> {
        Err(ChannelError::TransportUnavailable {
            channel: "email",
            reason: "trustformers-serve declares no SMTP client dependency",
        }
        .into())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supports(&self, notification_type: &str) -> bool {
        notification_type == "email"
    }

    async fn health_check(&self) -> Result<ChannelHealth> {
        Ok(ChannelHealth {
            healthy: false,
            last_check: Utc::now(),
            consecutive_failures: 0,
            success_rate: 0.0,
        })
    }

    fn capabilities(&self) -> Vec<String> {
        Vec::new()
    }

    fn max_message_size(&self) -> Option<usize> {
        None
    }
}

/// Generic HTTP webhook notification channel.
///
/// Constructed with [`WebhookNotificationChannel::with_endpoint`] it POSTs the
/// notification as JSON and reports the endpoint's own response. Constructed
/// with [`WebhookNotificationChannel::new`] it has no endpoint and refuses to
/// send.
#[derive(Debug)]
pub struct WebhookNotificationChannel {
    name: String,
    config: Option<HttpChannelConfig>,
    client: reqwest::Client,
    counters: Arc<ChannelCounters>,
}

impl WebhookNotificationChannel {
    /// Create an unconfigured webhook channel that refuses to send.
    pub async fn new() -> Result<Self> {
        Ok(Self {
            name: "webhook".to_string(),
            config: None,
            client: reqwest::Client::new(),
            counters: Arc::new(ChannelCounters::default()),
        })
    }

    /// Create a webhook channel that POSTs to `endpoint`.
    pub fn with_endpoint(endpoint: impl Into<String>) -> Self {
        Self::with_config(HttpChannelConfig::new(endpoint))
    }

    /// Create a webhook channel from a full [`HttpChannelConfig`].
    pub fn with_config(config: HttpChannelConfig) -> Self {
        Self {
            name: "webhook".to_string(),
            config: Some(config),
            client: reqwest::Client::new(),
            counters: Arc::new(ChannelCounters::default()),
        }
    }

    /// JSON body sent to the endpoint.
    fn payload(notification: &Notification) -> serde_json::Value {
        serde_json::json!({
            "id": notification.id,
            "alert_id": notification.alert_id,
            "subject": notification.subject,
            "content": notification.content,
            "priority": format!("{:?}", notification.priority),
            "severity": format!("{:?}", notification.severity),
            "recipients": notification.recipients,
            "created_at": notification.created_at.to_rfc3339(),
        })
    }
}

#[async_trait::async_trait]
impl NotificationChannel for WebhookNotificationChannel {
    async fn send_notification(&self, notification: &Notification) -> Result<DeliveryResult> {
        let config =
            self.config.as_ref().ok_or(ChannelError::NotConfigured { channel: "webhook" })?;
        Ok(post_json(
            &self.client,
            config,
            Self::payload(notification),
            &self.counters,
        )
        .await)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supports(&self, notification_type: &str) -> bool {
        notification_type == "webhook" || notification_type == "*"
    }

    async fn health_check(&self) -> Result<ChannelHealth> {
        match &self.config {
            Some(config) => {
                let reachable = probe(&self.client, config).await;
                Ok(self.counters.to_health(reachable))
            },
            None => Ok(self.counters.to_health(false)),
        }
    }

    fn capabilities(&self) -> Vec<String> {
        vec![
            "basic".to_string(),
            "json".to_string(),
            "custom_headers".to_string(),
        ]
    }

    fn max_message_size(&self) -> Option<usize> {
        Some(1024 * 1024) // 1MB limit
    }
}

/// Slack incoming-webhook notification channel.
///
/// Delivers through Slack's incoming-webhook API: a JSON `{"text": ...}` POST
/// to the workspace URL. Without that URL the channel refuses to send.
#[derive(Debug)]
pub struct SlackNotificationChannel {
    name: String,
    config: Option<HttpChannelConfig>,
    client: reqwest::Client,
    counters: Arc<ChannelCounters>,
}

impl SlackNotificationChannel {
    /// Create an unconfigured Slack channel that refuses to send.
    pub async fn new() -> Result<Self> {
        Ok(Self {
            name: "slack".to_string(),
            config: None,
            client: reqwest::Client::new(),
            counters: Arc::new(ChannelCounters::default()),
        })
    }

    /// Create a Slack channel posting to an incoming-webhook URL.
    pub fn with_endpoint(endpoint: impl Into<String>) -> Self {
        Self::with_config(HttpChannelConfig::new(endpoint))
    }

    /// Create a Slack channel from a full [`HttpChannelConfig`].
    pub fn with_config(config: HttpChannelConfig) -> Self {
        Self {
            name: "slack".to_string(),
            config: Some(config),
            client: reqwest::Client::new(),
            counters: Arc::new(ChannelCounters::default()),
        }
    }

    /// Slack incoming-webhook payload for `notification`.
    fn payload(notification: &Notification) -> serde_json::Value {
        let marker = match notification.priority {
            NotificationPriority::Emergency => "[EMERGENCY]",
            NotificationPriority::Critical => "[CRITICAL]",
            NotificationPriority::High => "[HIGH]",
            NotificationPriority::Normal => "[NORMAL]",
            NotificationPriority::Low => "[LOW]",
        };
        serde_json::json!({
            "text": format!("{} {}\n{}", marker, notification.subject, notification.content),
        })
    }
}

#[async_trait::async_trait]
impl NotificationChannel for SlackNotificationChannel {
    async fn send_notification(&self, notification: &Notification) -> Result<DeliveryResult> {
        let config =
            self.config.as_ref().ok_or(ChannelError::NotConfigured { channel: "slack" })?;
        Ok(post_json(
            &self.client,
            config,
            Self::payload(notification),
            &self.counters,
        )
        .await)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supports(&self, notification_type: &str) -> bool {
        notification_type == "slack" || notification_type == "*"
    }

    async fn health_check(&self) -> Result<ChannelHealth> {
        match &self.config {
            Some(config) => {
                let reachable = probe(&self.client, config).await;
                Ok(self.counters.to_health(reachable))
            },
            None => Ok(self.counters.to_health(false)),
        }
    }

    fn capabilities(&self) -> Vec<String> {
        vec![
            "basic".to_string(),
            "markdown".to_string(),
            "emoji".to_string(),
            "mentions".to_string(),
        ]
    }

    fn max_message_size(&self) -> Option<usize> {
        Some(40000) // Slack's message limit
    }
}

/// SMS notification channel.
///
/// Every SMS gateway needs a provider SDK or REST credential set that this
/// crate does not carry, so the channel reports the gap instead of claiming a
/// delivery.
#[derive(Debug)]
pub struct SmsNotificationChannel {
    name: String,
}

impl SmsNotificationChannel {
    /// Create the SMS channel.
    pub async fn new() -> Result<Self> {
        Ok(Self {
            name: "sms".to_string(),
        })
    }
}

#[async_trait::async_trait]
impl NotificationChannel for SmsNotificationChannel {
    async fn send_notification(&self, _notification: &Notification) -> Result<DeliveryResult> {
        Err(ChannelError::TransportUnavailable {
            channel: "sms",
            reason: "trustformers-serve integrates no SMS gateway",
        }
        .into())
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supports(&self, notification_type: &str) -> bool {
        notification_type == "sms"
    }

    async fn health_check(&self) -> Result<ChannelHealth> {
        Ok(ChannelHealth {
            healthy: false,
            last_check: Utc::now(),
            consecutive_failures: 0,
            success_rate: 0.0,
        })
    }

    fn capabilities(&self) -> Vec<String> {
        Vec::new()
    }
}

/// PagerDuty Events API v2 notification channel.
///
/// Delivery is a JSON POST to the Events API enqueue endpoint carrying the
/// integration's routing key. Without a routing key the channel refuses to
/// send.
#[derive(Debug)]
pub struct PagerDutyNotificationChannel {
    name: String,
    routing_key: Option<String>,
    config: HttpChannelConfig,
    client: reqwest::Client,
    counters: Arc<ChannelCounters>,
}

impl PagerDutyNotificationChannel {
    /// Default PagerDuty Events API v2 enqueue endpoint.
    pub const EVENTS_V2_ENDPOINT: &'static str = "https://events.pagerduty.com/v2/enqueue";

    /// Create an unconfigured PagerDuty channel that refuses to send.
    pub async fn new() -> Result<Self> {
        Ok(Self {
            name: "pagerduty".to_string(),
            routing_key: None,
            config: HttpChannelConfig::new(Self::EVENTS_V2_ENDPOINT),
            client: reqwest::Client::new(),
            counters: Arc::new(ChannelCounters::default()),
        })
    }

    /// Create a PagerDuty channel using `routing_key` against the public
    /// Events API v2 endpoint.
    pub fn with_routing_key(routing_key: impl Into<String>) -> Self {
        Self {
            name: "pagerduty".to_string(),
            routing_key: Some(routing_key.into()),
            config: HttpChannelConfig::new(Self::EVENTS_V2_ENDPOINT),
            client: reqwest::Client::new(),
            counters: Arc::new(ChannelCounters::default()),
        }
    }

    /// Create a PagerDuty channel using `routing_key` against a specific
    /// endpoint, for EU-region accounts or a local relay.
    pub fn with_routing_key_and_config(
        routing_key: impl Into<String>,
        config: HttpChannelConfig,
    ) -> Self {
        Self {
            name: "pagerduty".to_string(),
            routing_key: Some(routing_key.into()),
            config,
            client: reqwest::Client::new(),
            counters: Arc::new(ChannelCounters::default()),
        }
    }

    /// Events API v2 payload for `notification`.
    fn payload(routing_key: &str, notification: &Notification) -> serde_json::Value {
        let severity = match notification.priority {
            NotificationPriority::Emergency | NotificationPriority::Critical => "critical",
            NotificationPriority::High => "error",
            NotificationPriority::Normal => "warning",
            NotificationPriority::Low => "info",
        };
        serde_json::json!({
            "routing_key": routing_key,
            "event_action": "trigger",
            "dedup_key": notification.alert_id,
            "payload": {
                "summary": notification.subject,
                "source": "trustformers-serve",
                "severity": severity,
                "custom_details": { "content": notification.content },
            },
        })
    }
}

#[async_trait::async_trait]
impl NotificationChannel for PagerDutyNotificationChannel {
    async fn send_notification(&self, notification: &Notification) -> Result<DeliveryResult> {
        let routing_key = self.routing_key.as_ref().ok_or(ChannelError::NotConfigured {
            channel: "pagerduty",
        })?;
        Ok(post_json(
            &self.client,
            &self.config,
            Self::payload(routing_key, notification),
            &self.counters,
        )
        .await)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supports(&self, notification_type: &str) -> bool {
        notification_type == "pagerduty"
    }

    async fn health_check(&self) -> Result<ChannelHealth> {
        if self.routing_key.is_none() {
            return Ok(self.counters.to_health(false));
        }
        let reachable = probe(&self.client, &self.config).await;
        Ok(self.counters.to_health(reachable))
    }

    fn capabilities(&self) -> Vec<String> {
        vec!["basic".to_string(), "deduplication".to_string()]
    }
}
