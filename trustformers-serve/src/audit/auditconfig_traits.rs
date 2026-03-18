//! # AuditConfig - Trait Implementations
//!
//! This module contains trait implementations for `AuditConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{AuditConfig, AuditSeverity};

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
