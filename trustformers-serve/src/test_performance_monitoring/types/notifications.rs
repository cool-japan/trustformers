//! Notifications Type Definitions

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationTarget {
    pub target_id: String,
    pub target_type: String,
    pub contact_info: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationRateLimiter {
    pub max_per_minute: u32,
    pub burst_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryTracker {
    pub tracking_enabled: bool,
    pub delivery_log: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationMetrics {
    pub sent_count: u64,
    pub failed_count: u64,
    pub avg_delivery_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryRequirements {
    pub require_acknowledgment: bool,
    // TODO: Re-enable once utils::duration_serde is implemented
    // #[serde(with = "crate::utils::duration_serde")]
    pub max_delivery_time: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationError {
    pub error_type: String,
    pub message: String,
    pub retry_possible: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryCapabilities {
    pub supports_attachments: bool,
    pub supports_html: bool,
    pub max_message_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailRateLimiter {
    pub max_per_hour: u32,
    pub burst_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmsLimits {
    pub max_length: usize,
    pub max_per_day: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookAuthentication {
    pub auth_type: String,
    pub credentials: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportNotificationManager {
    pub manager_id: String,
    pub notification_channels: Vec<String>,
    pub throttle_config: HashMap<String, String>,
}
