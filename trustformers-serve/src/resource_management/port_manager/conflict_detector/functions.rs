//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use crate::resource_management::types::*;

use super::types::{ConflictAction, ConflictCondition, ConflictDetectorConfig, ConflictRule, ConflictType, PortConflictDetector};

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;
    #[test]
    async fn test_conflict_detector_creation() {
        let detector = PortConflictDetector::new().await.unwrap();
        assert!(detector.is_enabled(). await);
        let (total, resolved) = detector.get_conflict_statistics().await;
        assert_eq!(total, 0);
        assert_eq!(resolved, 0);
    }
    #[test]
    async fn test_conflict_detection_basic() {
        let detector = PortConflictDetector::new().await.unwrap();
        let result = detector.check_allocation_conflicts(5, "test_001").await;
        assert!(result.is_ok());
    }
    #[test]
    async fn test_conflict_event_recording() {
        let detector = PortConflictDetector::new().await.unwrap();
        detector
            .record_conflict_event(
                8080,
                "test_001",
                Some("test_000"),
                ConflictType::AlreadyAllocated,
                ConflictAction::FindAlternative,
                true,
                Some(100),
                1.0,
                Some(0.5),
                Some("test_rule".to_string()),
            )
            .await;
        let (total, resolved) = detector.get_conflict_statistics().await;
        assert_eq!(total, 1);
        assert_eq!(resolved, 1);
    }
    #[test]
    async fn test_alternative_port_finding() {
        let detector = PortConflictDetector::new().await.unwrap();
        let alternatives = detector
            .find_alternative_ports(5, Some((8000, 8100)), &[8080, 8081])
            .await;
        assert_eq!(alternatives.len(), 5);
        assert!(! alternatives.contains(& 8080));
        assert!(! alternatives.contains(& 8081));
    }
    #[test]
    async fn test_priority_calculation() {
        let detector = PortConflictDetector::new().await.unwrap();
        let priority = detector.calculate_test_priority("test_001", 1.0, true, 5).await;
        assert!(priority > 1.0);
    }
    #[test]
    async fn test_force_allocation_rules() {
        let detector = PortConflictDetector::new().await.unwrap();
        let can_force = detector.can_force_allocate(9.5, Some(2.0)).await;
        assert!(can_force);
        let cannot_force = detector.can_force_allocate(2.0, Some(1.0)).await;
        assert!(! cannot_force);
    }
    #[test]
    async fn test_rule_management() {
        let detector = PortConflictDetector::new().await.unwrap();
        let custom_rule = ConflictRule {
            name: "custom_test_rule".to_string(),
            condition: ConflictCondition::Custom("test_condition".to_string()),
            action: ConflictAction::Custom("test_action".to_string()),
            priority: 50,
            enabled: true,
            description: Some("Test rule".to_string()),
            config: HashMap::new(),
        };
        detector.add_conflict_rule(custom_rule).await;
        let removed = detector.remove_conflict_rule("custom_test_rule").await;
        assert!(removed);
        let not_removed = detector.remove_conflict_rule("nonexistent_rule").await;
        assert!(! not_removed);
    }
    #[test]
    async fn test_configuration_update() {
        let detector = PortConflictDetector::new().await.unwrap();
        let mut new_config = ConflictDetectorConfig::default();
        new_config.max_alternative_search = 200;
        new_config.enable_auto_resolution = false;
        detector.update_config(new_config.clone()).await;
        let current_config = detector.get_config().await;
        assert_eq!(current_config.max_alternative_search, 200);
        assert!(! current_config.enable_auto_resolution);
    }
    #[test]
    async fn test_history_management() {
        let detector = PortConflictDetector::new().await.unwrap();
        for i in 0..5 {
            detector
                .record_conflict_event(
                    8080 + i,
                    &format!("test_{:03}", i),
                    None,
                    ConflictType::AlreadyAllocated,
                    ConflictAction::FindAlternative,
                    true,
                    Some(50),
                    1.0,
                    None,
                    None,
                )
                .await;
        }
        let recent_events = detector.get_recent_conflict_events(3).await;
        assert_eq!(recent_events.len(), 3);
        detector.clear_history().await;
        let (total, _) = detector.get_conflict_statistics().await;
        assert_eq!(total, 0);
    }
    #[test]
    async fn test_report_generation() {
        let detector = PortConflictDetector::new().await.unwrap();
        detector
            .record_conflict_event(
                8080,
                "test_001",
                Some("test_000"),
                ConflictType::AlreadyAllocated,
                ConflictAction::FindAlternative,
                true,
                Some(100),
                1.0,
                Some(0.5),
                Some("test_rule".to_string()),
            )
            .await;
        let report = detector.generate_conflict_report().await;
        assert!(report.contains("Port Conflict Detection Report"));
        assert!(report.contains("Total Conflicts: 1"));
        assert!(report.contains("Resolved Conflicts: 1"));
    }
    #[test]
    async fn test_conflict_resolution_result() {
        let detector = PortConflictDetector::new().await.unwrap();
        let result = detector
            .detect_and_resolve_port_conflict(8080, "test_001", 1.0)
            .await;
        assert!(result.resolved);
        assert!(result.resolution_time.as_millis() >= 0);
    }
}
