//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
mod tests {

    use crate::production::DeploymentStatus;
    use crate::{ProductionConfig, ProductionManager, UpdateStrategy};
    #[tokio::test]
    async fn test_production_manager_creation() {
        let config = ProductionConfig::default();
        let manager = ProductionManager::new(config);
        let status = manager.get_status().await;
        assert!(matches!(
            status.deployment_status,
            DeploymentStatus::Running
        ));
        assert!(!status.maintenance_mode);
    }
    #[tokio::test]
    async fn test_maintenance_mode() {
        let config = ProductionConfig::default();
        let manager = ProductionManager::new(config);
        let result = manager.enable_maintenance_mode(Some("Testing".to_string())).await;
        assert!(result.is_ok());
        let status = manager.get_status().await;
        assert!(status.maintenance_mode);
        assert!(matches!(
            status.deployment_status,
            DeploymentStatus::Maintenance
        ));
        let result = manager.disable_maintenance_mode().await;
        assert!(result.is_ok());
        let status = manager.get_status().await;
        assert!(!status.maintenance_mode);
        assert!(matches!(
            status.deployment_status,
            DeploymentStatus::Running
        ));
    }
    #[tokio::test]
    async fn test_rolling_update() {
        let config = ProductionConfig::default();
        let manager = ProductionManager::new(config);
        let strategy = UpdateStrategy::Rolling {
            max_unavailable: 1,
            max_surge: 1,
            batch_size: 1,
        };
        let result = manager.start_rolling_update("test-update".to_string(), strategy).await;
        assert!(result.is_ok());
        let metrics = manager.get_metrics().await;
        assert_eq!(metrics.update_count, 1);
        assert_eq!(metrics.successful_updates, 1);
    }
}
