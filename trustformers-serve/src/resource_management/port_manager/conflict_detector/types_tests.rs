//! Comprehensive tests for conflict_detector/types.rs
//!
//! Tests for PortConflictDetector, ConflictRule, ConflictStatistics,
//! ConflictResolutionResult, and related types.

#[cfg(test)]
mod tests {
    use super::super::types::*;
    use std::collections::HashMap;
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

    #[tokio::test]
    async fn test_conflict_detector_creation() {
        let result = PortConflictDetector::new().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_detector_default_rules() {
        if let Ok(detector) = PortConflictDetector::new().await {
            let rules = detector.conflict_rules.read();
            assert!(!rules.is_empty());
            // Should have at least the 7 default rules
            assert!(rules.len() >= 7);
        }
    }

    #[tokio::test]
    async fn test_check_allocation_no_conflicts() {
        if let Ok(detector) = PortConflictDetector::new().await {
            let result = detector
                .check_allocation_conflicts(5, "test_001")
                .await;
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_check_allocation_zero_count() {
        if let Ok(detector) = PortConflictDetector::new().await {
            let result = detector
                .check_allocation_conflicts(0, "test_001")
                .await;
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_detect_and_resolve_port_conflict() {
        if let Ok(detector) = PortConflictDetector::new().await {
            let result = detector
                .detect_and_resolve_port_conflict(8080, "test_001", 1.0)
                .await;
            assert!(result.resolved);
        }
    }

    #[tokio::test]
    async fn test_apply_resolution_rules_allocated() {
        if let Ok(detector) = PortConflictDetector::new().await {
            let action = detector
                .apply_resolution_rules(ConflictType::AlreadyAllocated, "test_001", 1.0)
                .await;
            assert!(matches!(action, ConflictAction::FindAlternative));
        }
    }

    #[tokio::test]
    async fn test_apply_resolution_rules_well_known() {
        if let Ok(detector) = PortConflictDetector::new().await {
            let action = detector
                .apply_resolution_rules(ConflictType::WellKnown, "test_001", 1.0)
                .await;
            assert!(matches!(action, ConflictAction::Deny));
        }
    }

    #[tokio::test]
    async fn test_apply_resolution_rules_excluded() {
        if let Ok(detector) = PortConflictDetector::new().await {
            let action = detector
                .apply_resolution_rules(ConflictType::Excluded, "test_001", 1.0)
                .await;
            assert!(matches!(action, ConflictAction::Deny));
        }
    }

    #[tokio::test]
    async fn test_find_alternative_ports() {
        if let Ok(detector) = PortConflictDetector::new().await {
            let alternatives = detector
                .find_alternative_ports(3, Some((8000, 8100)), &[8000, 8001, 8002])
                .await;
            assert_eq!(alternatives.len(), 3);
            assert!(!alternatives.contains(&8000));
            assert!(!alternatives.contains(&8001));
            assert!(!alternatives.contains(&8002));
        }
    }

    #[tokio::test]
    async fn test_find_alternative_ports_no_exclusions() {
        if let Ok(detector) = PortConflictDetector::new().await {
            let alternatives = detector
                .find_alternative_ports(5, Some((9000, 9100)), &[])
                .await;
            assert_eq!(alternatives.len(), 5);
        }
    }

    #[tokio::test]
    async fn test_record_conflict_event() {
        if let Ok(detector) = PortConflictDetector::new().await {
            detector
                .record_conflict_event(
                    8080,
                    "test_001",
                    Some("test_002"),
                    ConflictType::AlreadyAllocated,
                    ConflictAction::FindAlternative,
                    true,
                    Some(15),
                    1.0,
                    Some(0.5),
                    Some("find_alternative_for_allocated".to_string()),
                )
                .await;
            let (total, resolved) = detector.get_conflict_statistics().await;
            assert_eq!(total, 1);
            assert_eq!(resolved, 1);
        }
    }

    #[tokio::test]
    async fn test_record_multiple_conflict_events() {
        if let Ok(detector) = PortConflictDetector::new().await {
            let mut rng = Lcg::new(42);
            for i in 0..10 {
                let port = (rng.next() % 1000 + 8000) as u16;
                let resolved = rng.next() % 2 == 0;
                detector
                    .record_conflict_event(
                        port,
                        &format!("test_{:03}", i),
                        None,
                        ConflictType::AlreadyAllocated,
                        ConflictAction::FindAlternative,
                        resolved,
                        Some(10),
                        1.0,
                        None,
                        None,
                    )
                    .await;
            }
            let (total, _) = detector.get_conflict_statistics().await;
            assert_eq!(total, 10);
        }
    }

    #[tokio::test]
    async fn test_detailed_statistics() {
        if let Ok(detector) = PortConflictDetector::new().await {
            detector
                .record_conflict_event(
                    8080,
                    "test_001",
                    None,
                    ConflictType::AlreadyAllocated,
                    ConflictAction::FindAlternative,
                    true,
                    Some(20),
                    1.0,
                    None,
                    None,
                )
                .await;
            let stats = detector.get_detailed_statistics().await;
            assert_eq!(stats.total_conflicts, 1);
            assert_eq!(stats.resolved_conflicts, 1);
            assert!(!stats.conflicts_by_type.is_empty());
        }
    }

    #[tokio::test]
    async fn test_calculate_test_priority_normal() {
        if let Ok(detector) = PortConflictDetector::new().await {
            let priority = detector
                .calculate_test_priority("test_001", 1.0, false, 5)
                .await;
            assert!((priority - 1.0).abs() < f32::EPSILON);
        }
    }

    #[tokio::test]
    async fn test_calculate_test_priority_critical() {
        if let Ok(detector) = PortConflictDetector::new().await {
            let normal = detector
                .calculate_test_priority("test_001", 1.0, false, 0)
                .await;
            let critical = detector
                .calculate_test_priority("test_001", 1.0, true, 0)
                .await;
            assert!(critical > normal);
        }
    }

    #[tokio::test]
    async fn test_calculate_test_priority_long_running_penalty() {
        if let Ok(detector) = PortConflictDetector::new().await {
            let short = detector
                .calculate_test_priority("test_001", 5.0, false, 5)
                .await;
            let long = detector
                .calculate_test_priority("test_001", 5.0, false, 15)
                .await;
            assert!(short > long);
        }
    }

    #[tokio::test]
    async fn test_can_force_allocate_high_priority() {
        if let Ok(detector) = PortConflictDetector::new().await {
            let can_force = detector.can_force_allocate(9.0, Some(1.0)).await;
            assert!(can_force);
        }
    }

    #[tokio::test]
    async fn test_can_force_allocate_low_priority() {
        if let Ok(detector) = PortConflictDetector::new().await {
            let can_force = detector.can_force_allocate(1.0, Some(5.0)).await;
            assert!(!can_force);
        }
    }

    #[tokio::test]
    async fn test_recent_conflict_events() {
        if let Ok(detector) = PortConflictDetector::new().await {
            for i in 0..5 {
                detector
                    .record_conflict_event(
                        (8000 + i) as u16,
                        &format!("test_{}", i),
                        None,
                        ConflictType::Reserved,
                        ConflictAction::Queue,
                        false,
                        None,
                        1.0,
                        None,
                        None,
                    )
                    .await;
            }
            let recent = detector.get_recent_conflict_events(3).await;
            assert_eq!(recent.len(), 3);
        }
    }

    #[tokio::test]
    async fn test_add_custom_rule() {
        if let Ok(detector) = PortConflictDetector::new().await {
            let initial_count = detector.conflict_rules.read().len();
            let rule = ConflictRule {
                name: "custom_rule".to_string(),
                condition: ConflictCondition::Custom("custom".to_string()),
                action: ConflictAction::EscalateManual,
                priority: 50,
                enabled: true,
                description: Some("Test custom rule".to_string()),
                config: HashMap::new(),
            };
            detector.add_conflict_rule(rule).await;
            let new_count = detector.conflict_rules.read().len();
            assert_eq!(new_count, initial_count + 1);
        }
    }

    #[tokio::test]
    async fn test_remove_conflict_rule() {
        if let Ok(detector) = PortConflictDetector::new().await {
            let removed = detector
                .remove_conflict_rule("prevent_well_known_ports")
                .await;
            assert!(removed);
            let not_removed = detector
                .remove_conflict_rule("nonexistent_rule")
                .await;
            assert!(!not_removed);
        }
    }

    #[tokio::test]
    async fn test_enable_disable() {
        if let Ok(detector) = PortConflictDetector::new().await {
            assert!(detector.is_enabled().await);
            detector.set_enabled(false).await;
            assert!(!detector.is_enabled().await);
            detector.set_enabled(true).await;
            assert!(detector.is_enabled().await);
        }
    }

    #[tokio::test]
    async fn test_disabled_skips_checks() {
        if let Ok(detector) = PortConflictDetector::new().await {
            detector.set_enabled(false).await;
            let result = detector
                .check_allocation_conflicts(1000, "test_001")
                .await;
            assert!(result.is_ok());
        }
    }

    #[tokio::test]
    async fn test_clear_history() {
        if let Ok(detector) = PortConflictDetector::new().await {
            detector
                .record_conflict_event(
                    8080,
                    "test_001",
                    None,
                    ConflictType::AlreadyAllocated,
                    ConflictAction::Deny,
                    false,
                    None,
                    1.0,
                    None,
                    None,
                )
                .await;
            detector.clear_history().await;
            let (total, _) = detector.get_conflict_statistics().await;
            assert_eq!(total, 0);
        }
    }

    #[tokio::test]
    async fn test_generate_conflict_report() {
        if let Ok(detector) = PortConflictDetector::new().await {
            detector
                .record_conflict_event(
                    8080,
                    "test_001",
                    None,
                    ConflictType::AlreadyAllocated,
                    ConflictAction::FindAlternative,
                    true,
                    Some(10),
                    1.0,
                    None,
                    None,
                )
                .await;
            let report = detector.generate_conflict_report().await;
            assert!(report.contains("Port Conflict Detection Report"));
            assert!(report.contains("Total Conflicts:"));
            assert!(report.contains("Resolution Rate:"));
        }
    }

    #[test]
    fn test_conflict_condition_equality() {
        let a = ConflictCondition::PortAlreadyAllocated;
        let b = ConflictCondition::PortAlreadyAllocated;
        let c = ConflictCondition::PortReserved;
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_conflict_action_equality() {
        let a = ConflictAction::Deny;
        let b = ConflictAction::Deny;
        let c = ConflictAction::FindAlternative;
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_conflict_type_equality() {
        let a = ConflictType::AlreadyAllocated;
        let b = ConflictType::AlreadyAllocated;
        let c = ConflictType::WellKnown;
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_conflict_statistics_default() {
        let stats = ConflictStatistics::default();
        assert_eq!(stats.total_conflicts, 0);
        assert_eq!(stats.resolved_conflicts, 0);
        assert_eq!(stats.denied_conflicts, 0);
        assert!(stats.conflicts_by_type.is_empty());
    }
}
