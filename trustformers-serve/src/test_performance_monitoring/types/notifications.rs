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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }
        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            self.state
        }
        fn next_f32(&mut self) -> f32 {
            (self.next() >> 11) as f32 / (1u64 << 53) as f32
        }
    }

    #[test]
    fn test_notification_target_construction() {
        let nt = NotificationTarget {
            target_id: "tgt-001".to_string(),
            target_type: "email".to_string(),
            contact_info: "alert@example.com".to_string(),
        };
        assert_eq!(nt.target_id, "tgt-001");
        assert_eq!(nt.target_type, "email");
        assert!(!nt.contact_info.is_empty());
    }

    #[test]
    fn test_notification_target_slack() {
        let nt = NotificationTarget {
            target_id: "tgt-slack".to_string(),
            target_type: "slack".to_string(),
            contact_info: "#alerts-channel".to_string(),
        };
        assert_eq!(nt.target_type, "slack");
        assert!(nt.contact_info.starts_with('#'));
    }

    #[test]
    fn test_notification_rate_limiter_construction() {
        let rl = NotificationRateLimiter {
            max_per_minute: 60,
            burst_size: 10,
        };
        assert_eq!(rl.max_per_minute, 60);
        assert_eq!(rl.burst_size, 10);
        assert!(rl.burst_size <= rl.max_per_minute);
    }

    #[test]
    fn test_notification_rate_limiter_burst() {
        let rl = NotificationRateLimiter {
            max_per_minute: 100,
            burst_size: 20,
        };
        assert!(rl.max_per_minute > rl.burst_size);
    }

    #[test]
    fn test_delivery_tracker_enabled() {
        let dt = DeliveryTracker {
            tracking_enabled: true,
            delivery_log: vec!["msg-1".to_string(), "msg-2".to_string()],
        };
        assert!(dt.tracking_enabled);
        assert_eq!(dt.delivery_log.len(), 2);
    }

    #[test]
    fn test_delivery_tracker_disabled() {
        let dt = DeliveryTracker {
            tracking_enabled: false,
            delivery_log: Vec::new(),
        };
        assert!(!dt.tracking_enabled);
        assert!(dt.delivery_log.is_empty());
    }

    #[test]
    fn test_notification_metrics_construction() {
        let nm = NotificationMetrics {
            sent_count: 1000,
            failed_count: 5,
            avg_delivery_time_ms: 42.5,
        };
        assert_eq!(nm.sent_count, 1000);
        assert_eq!(nm.failed_count, 5);
        assert!(nm.avg_delivery_time_ms > 0.0);
        assert!(nm.sent_count > nm.failed_count);
    }

    #[test]
    fn test_notification_metrics_zero_failures() {
        let nm = NotificationMetrics {
            sent_count: 100,
            failed_count: 0,
            avg_delivery_time_ms: 15.0,
        };
        assert_eq!(nm.failed_count, 0);
    }

    #[test]
    fn test_delivery_requirements_construction() {
        let dr = DeliveryRequirements {
            require_acknowledgment: true,
            max_delivery_time: Duration::from_secs(30),
        };
        assert!(dr.require_acknowledgment);
        assert_eq!(dr.max_delivery_time, Duration::from_secs(30));
    }

    #[test]
    fn test_delivery_requirements_no_ack() {
        let dr = DeliveryRequirements {
            require_acknowledgment: false,
            max_delivery_time: Duration::from_secs(60),
        };
        assert!(!dr.require_acknowledgment);
    }

    #[test]
    fn test_notification_error_retryable() {
        let ne = NotificationError {
            error_type: "timeout".to_string(),
            message: "Connection timed out".to_string(),
            retry_possible: true,
        };
        assert_eq!(ne.error_type, "timeout");
        assert!(ne.retry_possible);
    }

    #[test]
    fn test_notification_error_not_retryable() {
        let ne = NotificationError {
            error_type: "invalid_address".to_string(),
            message: "Invalid email address".to_string(),
            retry_possible: false,
        };
        assert!(!ne.retry_possible);
    }

    #[test]
    fn test_delivery_capabilities_full() {
        let dc = DeliveryCapabilities {
            supports_attachments: true,
            supports_html: true,
            max_message_size: 10 * 1024 * 1024,
        };
        assert!(dc.supports_attachments);
        assert!(dc.supports_html);
        assert!(dc.max_message_size > 0);
    }

    #[test]
    fn test_delivery_capabilities_sms() {
        let dc = DeliveryCapabilities {
            supports_attachments: false,
            supports_html: false,
            max_message_size: 160,
        };
        assert!(!dc.supports_attachments);
        assert!(!dc.supports_html);
        assert_eq!(dc.max_message_size, 160);
    }

    #[test]
    fn test_email_rate_limiter_construction() {
        let erl = EmailRateLimiter {
            max_per_hour: 100,
            burst_size: 10,
        };
        assert_eq!(erl.max_per_hour, 100);
        assert_eq!(erl.burst_size, 10);
    }

    #[test]
    fn test_sms_limits_construction() {
        let sl = SmsLimits {
            max_length: 160,
            max_per_day: 50,
        };
        assert_eq!(sl.max_length, 160);
        assert_eq!(sl.max_per_day, 50);
    }

    #[test]
    fn test_webhook_authentication_construction() {
        let wa = WebhookAuthentication {
            auth_type: "bearer".to_string(),
            credentials: "token_value".to_string(),
        };
        assert_eq!(wa.auth_type, "bearer");
        assert!(!wa.credentials.is_empty());
    }

    #[test]
    fn test_report_notification_manager_construction() {
        let rnm = ReportNotificationManager {
            manager_id: "mgr-1".to_string(),
            notification_channels: vec!["email".to_string(), "slack".to_string()],
            throttle_config: HashMap::new(),
        };
        assert_eq!(rnm.manager_id, "mgr-1");
        assert_eq!(rnm.notification_channels.len(), 2);
    }

    #[test]
    fn test_report_notification_manager_with_throttle() {
        let mut config = HashMap::new();
        config.insert("max_rate".to_string(), "10".to_string());
        let rnm = ReportNotificationManager {
            manager_id: "mgr-2".to_string(),
            notification_channels: Vec::new(),
            throttle_config: config,
        };
        assert_eq!(rnm.throttle_config.len(), 1);
    }

    #[test]
    fn test_notification_metrics_random_values() {
        let mut lcg = Lcg::new(271828);
        for _ in 0..5 {
            let sent = (lcg.next_f32() * 1000.0) as u64;
            let failed = (lcg.next_f32() * sent as f32) as u64;
            let avg = lcg.next_f32() as f64 * 100.0;
            let nm = NotificationMetrics {
                sent_count: sent,
                failed_count: failed,
                avg_delivery_time_ms: avg,
            };
            assert!(nm.sent_count >= nm.failed_count);
        }
    }

    #[test]
    fn test_delivery_tracker_growing_log() {
        let entries: Vec<String> = (0..5).map(|i| format!("entry-{}", i)).collect();
        let dt = DeliveryTracker {
            tracking_enabled: true,
            delivery_log: entries.clone(),
        };
        assert_eq!(dt.delivery_log.len(), 5);
        assert_eq!(dt.delivery_log[0], "entry-0");
        assert_eq!(dt.delivery_log[4], "entry-4");
    }
}
