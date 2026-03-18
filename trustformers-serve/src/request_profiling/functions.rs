//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {
    use super::super::*;
    use std::time::{Duration, SystemTime};
    #[tokio::test]
    async fn test_profiling_service_creation() {
        let config = RequestProfilingConfig::default();
        let service = RequestProfilingService::new(config);
        let stats = service.get_stats().await;
        assert_eq!(stats.total_profiles, 0);
        assert_eq!(stats.active_profiles, 0);
    }
    #[tokio::test]
    async fn test_start_and_complete_profile() {
        let mut config = RequestProfilingConfig::default();
        config.sampling_rate = 1.0;
        config.min_duration_to_profile_ms = 0;
        let service = RequestProfilingService::new(config);
        let profile_id = service
            .start_profile("req_123".to_string(), "inference".to_string())
            .await
            .unwrap();
        assert!(!profile_id.is_empty());
        let stats = service.get_stats().await;
        assert_eq!(stats.total_profiles, 1);
        assert_eq!(stats.active_profiles, 1);
        service.complete_profile(&profile_id).await.unwrap();
        let stats = service.get_stats().await;
        assert_eq!(stats.active_profiles, 0);
        assert_eq!(stats.completed_profiles, 1);
    }
    #[tokio::test]
    async fn test_record_timing() {
        let mut config = RequestProfilingConfig::default();
        config.sampling_rate = 1.0;
        let service = RequestProfilingService::new(config);
        let profile_id = service
            .start_profile("req_123".to_string(), "inference".to_string())
            .await
            .unwrap();
        let inference_duration = Duration::from_millis(100);
        service
            .record_timing(&profile_id, "inference", inference_duration)
            .await
            .unwrap();
        let profile = service.get_profile(&profile_id).await.unwrap();
        assert_eq!(
            profile.timing_breakdown.inference_duration,
            Some(inference_duration)
        );
    }
    #[tokio::test]
    async fn test_resource_usage_recording() {
        let mut config = RequestProfilingConfig::default();
        config.sampling_rate = 1.0;
        let service = RequestProfilingService::new(config);
        let profile_id = service
            .start_profile("req_123".to_string(), "inference".to_string())
            .await
            .unwrap();
        let resource_usage = ResourceUsage {
            peak_cpu_percent: Some(85.5),
            peak_memory_bytes: Some(1024 * 1024 * 100),
            ..Default::default()
        };
        service
            .record_resource_usage(&profile_id, resource_usage.clone())
            .await
            .unwrap();
        let profile = service.get_profile(&profile_id).await.unwrap();
        assert_eq!(profile.resource_usage.peak_cpu_percent, Some(85.5));
        assert_eq!(
            profile.resource_usage.peak_memory_bytes,
            Some(1024 * 1024 * 100)
        );
    }
    #[tokio::test]
    async fn test_error_recording() {
        let mut config = RequestProfilingConfig::default();
        config.sampling_rate = 1.0;
        let service = RequestProfilingService::new(config);
        let profile_id = service
            .start_profile("req_123".to_string(), "inference".to_string())
            .await
            .unwrap();
        let error_info = ErrorInfo {
            error_type: "TimeoutError".to_string(),
            error_message: "Request timed out".to_string(),
            stack_trace: vec!["function1".to_string(), "function2".to_string()],
            location: Some("inference.rs:123".to_string()),
            timestamp: SystemTime::now(),
            category: ErrorCategory::Timeout,
        };
        service.record_error(&profile_id, error_info).await.unwrap();
        let profile = service.get_profile(&profile_id).await.unwrap();
        assert!(profile.error_info.is_some());
        assert_eq!(profile.status, ProfileStatus::Failed);
    }
    #[tokio::test]
    async fn test_profile_filtering() {
        let mut config = RequestProfilingConfig::default();
        config.sampling_rate = 1.0;
        config.min_duration_to_profile_ms = 0;
        let service = RequestProfilingService::new(config);
        let profile1 = service
            .start_profile("req_1".to_string(), "inference".to_string())
            .await
            .unwrap();
        let profile2 = service
            .start_profile("req_2".to_string(), "preprocessing".to_string())
            .await
            .unwrap();
        service.complete_profile(&profile1).await.unwrap();
        let inference_profiles = service.list_profiles(Some("inference"), None, None).await;
        assert_eq!(inference_profiles.len(), 1);
        assert_eq!(inference_profiles[0].request_type, "inference");
        let active_profiles = service.list_profiles(None, Some(ProfileStatus::Active), None).await;
        assert_eq!(active_profiles.len(), 1);
        assert_eq!(active_profiles[0].id, profile2);
    }
    #[test]
    fn test_issue_severity_ordering() {
        assert!(IssueSeverity::Critical > IssueSeverity::High);
        assert!(IssueSeverity::High > IssueSeverity::Medium);
        assert!(IssueSeverity::Medium > IssueSeverity::Low);
        assert!(IssueSeverity::Low > IssueSeverity::Info);
    }
    #[test]
    fn test_difficulty_level_ordering() {
        assert!(DifficultyLevel::Expert > DifficultyLevel::Hard);
        assert!(DifficultyLevel::Hard > DifficultyLevel::Medium);
        assert!(DifficultyLevel::Medium > DifficultyLevel::Easy);
    }
}
