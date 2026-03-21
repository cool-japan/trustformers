//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {

    use super::super::types::*;
    #[tokio::test]
    async fn test_dr_manager_creation() {
        let config = DisasterRecoveryConfig::default();
        let manager = DisasterRecoveryManager::new(config).expect("failed to create DR manager");
        assert!(manager.config.enabled);
    }
    #[tokio::test]
    async fn test_failover_trigger() {
        let mut config = DisasterRecoveryConfig::default();
        config.failover.traffic_splitting.gradual_failover = false;
        let manager = DisasterRecoveryManager::new(config).expect("failed to create DR manager");
        let result = manager
            .trigger_failover(Some("dr-site-1".to_string()), "Test failover".to_string())
            .await;
        assert!(result.is_ok());
        let status = manager.get_status().await;
        assert_eq!(status.active_site_id, "dr-site-1");
    }
    #[tokio::test]
    async fn test_event_recording() {
        let config = DisasterRecoveryConfig::default();
        let manager = DisasterRecoveryManager::new(config).expect("failed to create DR manager");
        let result = manager.record_event(DREventType::TestStarted, "Test event", None).await;
        assert!(result.is_ok());
        let events = manager.get_recent_events(Some(10)).await;
        assert_eq!(events.len(), 1);
    }
}
