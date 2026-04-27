//! Auto-generated module structure

pub mod errorcontext_traits;
pub mod errortrackingconfig_traits;
pub mod functions;
pub mod notificationthresholds_traits;
pub mod types;

// Re-export all types
pub use functions::*;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_error_entry_new_sets_defaults() {
        let entry = ErrorEntry::new("IoError".to_string(), "file not found".to_string());
        assert!(!entry.id.is_empty());
        assert_eq!(entry.error_type, "IoError");
        assert_eq!(entry.message, "file not found");
        assert_eq!(entry.occurrence_count, 1);
        assert_eq!(entry.resolution_status, ResolutionStatus::Unresolved);
        assert!(!entry.fingerprint.is_empty());
    }

    #[test]
    fn test_error_entry_fingerprint_consistency() {
        let e1 = ErrorEntry::new("TypeError".to_string(), "null deref".to_string());
        let e2 = ErrorEntry::new("TypeError".to_string(), "null deref".to_string());
        assert_eq!(e1.fingerprint, e2.fingerprint);
    }

    #[test]
    fn test_error_entry_different_types_have_different_fingerprints() {
        let e1 = ErrorEntry::new("TypeError".to_string(), "msg".to_string());
        let e2 = ErrorEntry::new("IoError".to_string(), "msg".to_string());
        assert_ne!(e1.fingerprint, e2.fingerprint);
    }

    #[test]
    fn test_error_entry_builder_with_severity() {
        let entry = ErrorEntry::new("CriticalError".to_string(), "crash".to_string())
            .with_severity(ErrorSeverity::Critical);
        assert_eq!(entry.severity, ErrorSeverity::Critical);
    }

    #[test]
    fn test_error_entry_builder_with_category() {
        let entry = ErrorEntry::new("AuthError".to_string(), "unauthorized".to_string())
            .with_category(ErrorCategory::Authentication);
        assert_eq!(entry.category, ErrorCategory::Authentication);
    }

    #[test]
    fn test_error_entry_increment_occurrence() {
        let mut entry = ErrorEntry::new("TestError".to_string(), "test".to_string());
        entry.increment_occurrence();
        assert_eq!(entry.occurrence_count, 2);
    }

    #[test]
    fn test_error_context_builder_chain() {
        let ctx = ErrorContext::new()
            .with_request_id("req-abc".to_string())
            .with_user_id("user-xyz".to_string())
            .with_http_info("POST".to_string(), "/api/data".to_string())
            .with_service_info("serve".to_string(), "0.1.0".to_string());
        assert_eq!(ctx.request_id, Some("req-abc".to_string()));
        assert_eq!(ctx.user_id, Some("user-xyz".to_string()));
        assert_eq!(ctx.method, Some("POST".to_string()));
        assert_eq!(ctx.path, Some("/api/data".to_string()));
        assert_eq!(ctx.service_name, Some("serve".to_string()));
        assert_eq!(ctx.service_version, Some("0.1.0".to_string()));
    }

    #[test]
    fn test_error_context_default_all_none() {
        let ctx = ErrorContext::new();
        assert!(ctx.request_id.is_none());
        assert!(ctx.user_id.is_none());
        assert!(ctx.method.is_none());
        assert!(ctx.query_params.is_empty());
    }

    #[test]
    fn test_error_severity_ordering() {
        assert!(ErrorSeverity::Critical > ErrorSeverity::High);
        assert!(ErrorSeverity::High > ErrorSeverity::Medium);
        assert!(ErrorSeverity::Medium > ErrorSeverity::Low);
    }

    #[test]
    fn test_resolution_status_variants() {
        let unresolved = ResolutionStatus::Unresolved;
        let resolved = ResolutionStatus::Resolved;
        let ignored = ResolutionStatus::Ignored;
        let in_progress = ResolutionStatus::InProgress;
        assert_ne!(unresolved, resolved);
        assert_ne!(ignored, in_progress);
    }

    #[test]
    fn test_error_tracking_config_default() {
        let config = ErrorTrackingConfig::default();
        assert!(config.enabled);
        assert!(config.max_errors_in_memory > 0);
    }

    #[tokio::test]
    async fn test_error_tracking_system_new() {
        let config = ErrorTrackingConfig::default();
        let system = ErrorTrackingSystem::new(config);
        let stats = system.get_statistics().await;
        assert_eq!(stats.total_errors, 0);
    }

    #[tokio::test]
    async fn test_error_tracking_system_track_and_retrieve() {
        let mut config = ErrorTrackingConfig::default();
        config.enable_deduplication = false;
        let system = ErrorTrackingSystem::new(config);
        let error = ErrorEntry::new("TrackTest".to_string(), "test msg".to_string());
        let error_id = error.id.clone();
        let send_result = system.track_error(error).await;
        assert!(send_result.is_ok());
        // Allow background processor to handle the event
        tokio::time::sleep(Duration::from_millis(150)).await;
        let retrieved = system.get_error(&error_id).await;
        assert!(retrieved.is_ok());
        if let Ok(e) = retrieved {
            assert_eq!(e.error_type, "TrackTest");
        }
    }

    #[tokio::test]
    async fn test_error_tracking_system_statistics_count() {
        let config = ErrorTrackingConfig::default();
        let system = ErrorTrackingSystem::new(config);
        let e1 = ErrorEntry::new("Type1".to_string(), "msg1".to_string());
        let e2 = ErrorEntry::new("Type2".to_string(), "msg2".to_string());
        system.track_error(e1).await.unwrap_or_default();
        system.track_error(e2).await.unwrap_or_default();
        tokio::time::sleep(Duration::from_millis(150)).await;
        let stats = system.get_statistics().await;
        assert_eq!(stats.total_errors, 2);
    }

    #[tokio::test]
    async fn test_error_tracking_export_json() {
        let config = ErrorTrackingConfig::default();
        let system = ErrorTrackingSystem::new(config);
        let result = system.export_errors(ErrorExportFormat::Json).await;
        assert!(result.is_ok());
        if let Ok(bytes) = result {
            let s = String::from_utf8_lossy(&bytes);
            assert!(s.contains('[') || s.contains(']'));
        }
    }

    #[tokio::test]
    async fn test_error_tracking_export_csv() {
        let mut config = ErrorTrackingConfig::default();
        config.enable_deduplication = false;
        let system = ErrorTrackingSystem::new(config);
        let e = ErrorEntry::new("CsvTest".to_string(), "csv message".to_string());
        system.track_error(e).await.unwrap_or_default();
        tokio::time::sleep(Duration::from_millis(150)).await;
        let result = system.export_errors(ErrorExportFormat::Csv).await;
        assert!(result.is_ok());
        if let Ok(bytes) = result {
            let s = String::from_utf8_lossy(&bytes);
            assert!(s.contains("ID,Error Type,Message"));
        }
    }

    #[test]
    fn test_escape_csv_field_with_comma() {
        let result = escape_csv_field("hello, world");
        assert!(result.starts_with('"'));
        assert!(result.ends_with('"'));
    }

    #[test]
    fn test_escape_csv_field_plain() {
        let result = escape_csv_field("hello");
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_error_category_variants_distinct() {
        assert_ne!(ErrorCategory::Authentication, ErrorCategory::Authorization);
        assert_ne!(ErrorCategory::Network, ErrorCategory::Database);
        assert_ne!(ErrorCategory::Security, ErrorCategory::Unknown);
    }

    #[test]
    fn test_notification_thresholds_accessible() {
        let config = ErrorTrackingConfig::default();
        assert!(config.notification_thresholds.critical_errors_per_minute >= 0.0);
    }
}
