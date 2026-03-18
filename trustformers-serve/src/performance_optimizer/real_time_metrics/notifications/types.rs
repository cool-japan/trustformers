//! # Core Types and Configuration for Notification System
//!
//! This module contains all the fundamental types, enums, and configuration structures
//! used throughout the notification system.

use super::super::types::*;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, AtomicUsize},
    time::Duration,
};

// =============================================================================
// CONFIGURATION TYPES
// =============================================================================

/// Comprehensive configuration for the notification system
///
/// Provides extensive configuration options for all aspects of notification
/// processing, delivery, and monitoring with sensible defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    /// Maximum concurrent notifications
    pub max_concurrent_notifications: usize,

    /// Default delivery guarantee level
    pub delivery_guarantee: DeliveryGuarantee,

    /// Maximum retry attempts for failed notifications
    pub retry_attempts: usize,

    /// Base retry delay in milliseconds
    pub base_retry_delay_ms: u64,

    /// Maximum retry delay in milliseconds
    pub max_retry_delay_ms: u64,

    /// Rate limiting: maximum notifications per minute
    pub rate_limit_per_minute: usize,

    /// Rate limiting: burst allowance
    pub rate_limit_burst: usize,

    /// Enable health monitoring for channels
    pub enable_health_monitoring: bool,

    /// Health check interval
    pub health_check_interval: Duration,

    /// Channel failure threshold for circuit breaker
    pub failure_threshold: usize,

    /// Circuit breaker recovery timeout
    pub circuit_breaker_timeout: Duration,

    /// Enable notification auditing
    pub enable_auditing: bool,

    /// Audit retention period
    pub audit_retention_period: Duration,

    /// Maximum audit history size
    pub max_audit_history: usize,

    /// Enable message templating
    pub enable_templating: bool,

    /// Template cache size
    pub template_cache_size: usize,

    /// Notification timeout
    pub notification_timeout: Duration,

    /// Enable escalation integration
    pub enable_escalation: bool,

    /// Escalation check interval
    pub escalation_check_interval: Duration,
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            max_concurrent_notifications: 1000,
            delivery_guarantee: DeliveryGuarantee::AtLeastOnce,
            retry_attempts: 3,
            base_retry_delay_ms: 1000,
            max_retry_delay_ms: 30000,
            rate_limit_per_minute: 100,
            rate_limit_burst: 10,
            enable_health_monitoring: true,
            health_check_interval: Duration::from_secs(30),
            failure_threshold: 5,
            circuit_breaker_timeout: Duration::from_secs(60),
            enable_auditing: true,
            audit_retention_period: Duration::from_secs(7 * 24 * 3600), // 7 days
            max_audit_history: 10000,
            enable_templating: true,
            template_cache_size: 1000,
            notification_timeout: Duration::from_secs(30),
            enable_escalation: true,
            escalation_check_interval: Duration::from_secs(60),
        }
    }
}

/// Delivery guarantee levels for notifications
///
/// Different levels of delivery guarantees with varying reliability
/// and performance characteristics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeliveryGuarantee {
    /// Best effort delivery - no guarantees, highest performance
    BestEffort,

    /// At least once delivery - guarantees delivery with possible duplicates
    AtLeastOnce,

    /// Exactly once delivery - guarantees single delivery (highest overhead)
    ExactlyOnce,
}

/// Configuration for individual notification channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    /// Channel name
    pub name: String,

    /// Channel enabled status
    pub enabled: bool,

    /// Channel priority (higher = preferred)
    pub priority: u8,

    /// Channel-specific configuration
    pub config: HashMap<String, String>,

    /// Rate limit for this channel
    pub rate_limit: Option<usize>,

    /// Timeout for this channel
    pub timeout: Duration,

    /// Retry policy for this channel
    pub retry_policy: RetryPolicy,

    /// Health check settings
    pub health_check: HealthCheckConfig,
}

/// Retry policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum retry attempts
    pub max_attempts: usize,

    /// Base delay between retries
    pub base_delay: Duration,

    /// Maximum delay between retries
    pub max_delay: Duration,

    /// Backoff multiplier
    pub backoff_multiplier: f64,

    /// Enable jitter to prevent thundering herd
    pub enable_jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(1000),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            enable_jitter: true,
        }
    }
}

/// Health check configuration for channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Enable health checks
    pub enabled: bool,

    /// Health check interval
    pub interval: Duration,

    /// Health check timeout
    pub timeout: Duration,

    /// Failure threshold before marking unhealthy
    pub failure_threshold: usize,

    /// Success threshold for recovery
    pub success_threshold: usize,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            failure_threshold: 3,
            success_threshold: 2,
        }
    }
}

// =============================================================================
// CORE NOTIFICATION TYPES
// =============================================================================

/// Enhanced notification structure with comprehensive metadata
///
/// Provides rich notification data with delivery tracking, templating
/// support, and extensive metadata for routing and processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notification {
    /// Unique notification ID
    pub id: String,

    /// Alert ID that triggered this notification
    pub alert_id: String,

    /// Target channels for delivery
    pub channels: Vec<String>,

    /// Recipients list
    pub recipients: Vec<String>,

    /// Notification subject/title
    pub subject: String,

    /// Notification content/body
    pub content: String,

    /// Notification priority level
    pub priority: NotificationPriority,

    /// Severity level from original alert
    pub severity: SeverityLevel,

    /// Delivery guarantee for this notification
    pub delivery_guarantee: DeliveryGuarantee,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Delivery deadline
    pub deadline: Option<DateTime<Utc>>,

    /// Template name for formatting
    pub template: Option<String>,

    /// Template variables for substitution
    pub template_vars: HashMap<String, String>,

    /// Notification metadata
    pub metadata: HashMap<String, String>,

    /// Tags for classification and routing
    pub tags: Vec<String>,

    /// Escalation policy reference
    pub escalation_policy: Option<String>,

    /// Correlation ID for grouping related notifications
    pub correlation_id: Option<String>,
}

/// Notification priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NotificationPriority {
    /// Low priority - can be delayed
    Low = 1,

    /// Normal priority - standard delivery
    Normal = 2,

    /// High priority - expedited delivery
    High = 3,

    /// Critical priority - immediate delivery
    Critical = 4,

    /// Emergency priority - bypass rate limits
    Emergency = 5,
}

impl Default for NotificationPriority {
    fn default() -> Self {
        NotificationPriority::Normal
    }
}

/// Processed notification with delivery information
#[derive(Debug, Clone)]
pub struct ProcessedNotification {
    /// Original notification
    pub notification: Notification,

    /// Processing timestamp
    pub processed_at: DateTime<Utc>,

    /// Processing results by channel
    pub results: HashMap<String, DeliveryResult>,

    /// Overall processing status
    pub status: ProcessingStatus,

    /// Processing metadata
    pub metadata: HashMap<String, String>,
}

/// Delivery result for individual channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryResult {
    /// Delivery success status
    pub success: bool,

    /// Delivery timestamp
    pub delivered_at: Option<DateTime<Utc>>,

    /// Delivery attempts made
    pub attempts: usize,

    /// Error message if delivery failed
    pub error: Option<String>,

    /// Delivery latency in milliseconds
    pub latency_ms: Option<u64>,

    /// Channel-specific response data
    pub response_data: HashMap<String, String>,
}

/// Processing status for notifications
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessingStatus {
    /// Notification is pending processing
    Pending,

    /// Notification is being processed
    Processing,

    /// Notification delivered successfully to all channels
    Delivered,

    /// Notification partially delivered (some channels failed)
    PartiallyDelivered,

    /// Notification delivery failed on all channels
    Failed,

    /// Notification delivery timed out
    TimedOut,

    /// Notification was cancelled
    Cancelled,
}

/// Statistics for notification processing
#[derive(Debug, Default)]
pub struct NotificationStats {
    /// Total notifications processed
    pub total_processed: AtomicU64,

    /// Successful deliveries
    pub successful_deliveries: AtomicU64,

    /// Failed deliveries
    pub failed_deliveries: AtomicU64,

    /// Partial deliveries
    pub partial_deliveries: AtomicU64,

    /// Rate limited notifications
    pub rate_limited: AtomicU64,

    /// Retry attempts
    pub retry_attempts: AtomicU64,

    /// Average processing latency (ms)
    pub avg_processing_latency_ms: AtomicF32,

    /// Average delivery latency (ms)
    pub avg_delivery_latency_ms: AtomicF32,

    /// Current queue size
    pub queue_size: AtomicUsize,

    /// Processing errors
    pub processing_errors: AtomicU64,
}

// Re-export AtomicF32 from parent types module
pub use crate::performance_optimizer::real_time_metrics::types::common::AtomicF32;
