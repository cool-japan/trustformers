//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// Helper function to escape CSV field values
pub fn escape_csv_field(field: &str) -> String {
    if field.contains(',') || field.contains('"') || field.contains('\n') || field.contains('\r') {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}
#[cfg(test)]
mod tests {

    use crate::{
        ErrorContext, ErrorEntry, ErrorTrackingConfig, ErrorTrackingSystem, ResolutionStatus,
    };
    use std::time::Duration;
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
        config.enable_deduplication = false;
        let system = ErrorTrackingSystem::new(config);
        let error = ErrorEntry::new("TestError".to_string(), "Test message".to_string());
        let error_id = error.id.clone();
        system.track_error(error).await.unwrap();
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
        tokio::time::sleep(Duration::from_millis(100)).await;
        let stats = system.get_statistics().await;
        assert_eq!(stats.total_errors, 2);
        assert_eq!(stats.unique_errors, 2);
    }
}
