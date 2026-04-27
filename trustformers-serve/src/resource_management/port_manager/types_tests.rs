//! Comprehensive tests for port_manager/types.rs
//!
//! Tests for PortManagementError, PortReservationConfig, PortHealthConfig,
//! PortHealthThresholds, PerformanceConfig, ConflictRule, ConflictCondition,
//! ConflictAction, ConflictType, and related types.

#[cfg(test)]
mod tests {
    use super::super::types::*;
    use crate::resource_management::types::*;
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

    #[test]
    fn test_error_insufficient_ports() {
        let err = PortManagementError::InsufficientPorts {
            requested: 10,
            available: 3,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Insufficient"));
        assert!(msg.contains("10"));
        assert!(msg.contains("3"));
    }

    #[test]
    fn test_error_port_not_allocated() {
        let err = PortManagementError::PortNotAllocated { port: 8080 };
        let msg = format!("{}", err);
        assert!(msg.contains("8080"));
        assert!(msg.contains("not allocated"));
    }

    #[test]
    fn test_error_port_already_allocated() {
        let err = PortManagementError::PortAlreadyAllocated {
            port: 8080,
            test_id: "test_001".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("8080"));
        assert!(msg.contains("test_001"));
    }

    #[test]
    fn test_error_port_reserved() {
        let err = PortManagementError::PortReserved {
            port: 8080,
            test_id: "test_001".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("reserved"));
    }

    #[test]
    fn test_error_port_excluded() {
        let err = PortManagementError::PortExcluded { port: 80 };
        let msg = format!("{}", err);
        assert!(msg.contains("excluded"));
    }

    #[test]
    fn test_error_invalid_port_range() {
        let err = PortManagementError::InvalidPortRange {
            start: 9000,
            end: 8000,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Invalid port range"));
    }

    #[test]
    fn test_error_configuration_error() {
        let err = PortManagementError::ConfigurationError {
            message: "max_allocation must be > 0".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Configuration error"));
    }

    #[test]
    fn test_error_resource_conflict() {
        let err = PortManagementError::ResourceConflict {
            details: "Port 8080 conflict".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Resource conflict"));
    }

    #[test]
    fn test_error_operation_timeout() {
        let err = PortManagementError::OperationTimeout { timeout_secs: 300 };
        let msg = format!("{}", err);
        assert!(msg.contains("timeout"));
        assert!(msg.contains("300"));
    }

    #[test]
    fn test_error_internal_error() {
        let err = PortManagementError::InternalError {
            message: "unexpected state".to_string(),
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Internal error"));
    }

    #[test]
    fn test_reservation_config_default() {
        let config = PortReservationConfig::default();
        assert_eq!(config.max_reservations_per_test, 10);
        assert_eq!(
            config.max_reservation_duration,
            Duration::from_secs(3600)
        );
        assert_eq!(
            config.default_reservation_duration,
            Duration::from_secs(300)
        );
        assert!(config.enable_queue_processing);
        assert_eq!(config.max_queue_size, 1000);
    }

    #[test]
    fn test_health_config_default() {
        let config = PortHealthConfig::default();
        assert!(config.enabled);
        assert_eq!(config.check_interval, Duration::from_secs(60));
        assert_eq!(config.history_size, 1000);
        assert!(config.enable_alerts);
    }

    #[test]
    fn test_health_thresholds_default() {
        let thresholds = PortHealthThresholds::default();
        assert!((thresholds.utilization_warning - 80.0).abs() < f32::EPSILON);
        assert!((thresholds.utilization_critical - 95.0).abs() < f32::EPSILON);
        assert!(thresholds.utilization_warning < thresholds.utilization_critical);
        assert!(
            thresholds.conflicts_per_minute_warning
                < thresholds.conflicts_per_minute_critical
        );
        assert!(
            thresholds.allocation_time_warning_ms
                < thresholds.allocation_time_critical_ms
        );
    }

    #[test]
    fn test_performance_config_default() {
        let config = PerformanceConfig::default();
        assert!(config.enabled);
        assert_eq!(config.snapshot_interval, Duration::from_secs(300));
        assert_eq!(config.history_size, 288);
        assert!(!config.enable_detailed_timing);
    }

    #[test]
    fn test_conflict_rule_creation() {
        let rule = ConflictRule {
            name: "test_rule".to_string(),
            condition: ConflictCondition::PortAlreadyAllocated,
            action: ConflictAction::FindAlternative,
            priority: 80,
            enabled: true,
        };
        assert_eq!(rule.name, "test_rule");
        assert_eq!(rule.priority, 80);
        assert!(rule.enabled);
    }

    #[test]
    fn test_conflict_condition_variants() {
        let conditions = vec![
            ConflictCondition::PortAlreadyAllocated,
            ConflictCondition::PortReserved,
            ConflictCondition::PortExcluded,
            ConflictCondition::WellKnownPort,
            ConflictCondition::Custom("test".to_string()),
        ];
        assert_eq!(conditions.len(), 5);
    }

    #[test]
    fn test_conflict_action_variants() {
        let actions = vec![
            ConflictAction::Deny,
            ConflictAction::FindAlternative,
            ConflictAction::Queue,
            ConflictAction::ForceAllocate,
            ConflictAction::Custom("test".to_string()),
        ];
        assert_eq!(actions.len(), 5);
    }

    #[test]
    fn test_conflict_type_variants() {
        let types = vec![
            ConflictType::AlreadyAllocated,
            ConflictType::Reserved,
            ConflictType::Excluded,
            ConflictType::WellKnown,
            ConflictType::OutOfRange,
            ConflictType::Custom("test".to_string()),
        ];
        assert_eq!(types.len(), 6);
    }

    #[test]
    fn test_reservation_system_default() {
        let system = PortReservationSystem::default();
        assert!(system.reservations.lock().is_empty());
        assert!(system.reservations_by_test.lock().is_empty());
        assert!(system.reservation_queue.lock().is_empty());
        assert!(system.expiry_times.lock().is_empty());
        assert!(system.reservation_history.lock().is_empty());
    }

    #[test]
    fn test_port_health_status_creation() {
        let status = PortHealthStatus {
            overall_status: HealthStatus::Healthy,
            last_check: chrono::Utc::now(),
            available_ports: 100,
            allocated_ports: 10,
            reserved_ports: 5,
            utilization_percent: 10.0,
            recent_conflicts: 0,
            avg_allocation_time_ms: 5.0,
            active_alerts: Vec::new(),
        };
        assert_eq!(status.available_ports, 100);
        assert_eq!(status.allocated_ports, 10);
        assert!(status.active_alerts.is_empty());
    }

    #[test]
    fn test_performance_snapshot_creation() {
        let snapshot = PerformanceSnapshot {
            timestamp: chrono::Utc::now(),
            ops_per_second: 100.0,
            avg_allocation_time_ms: 5.0,
            success_rate_percent: 99.5,
            conflict_rate_percent: 0.5,
            utilization_percent: 25.0,
            metrics: HashMap::new(),
        };
        assert!((snapshot.ops_per_second - 100.0).abs() < f64::EPSILON);
        assert!((snapshot.success_rate_percent - 99.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_reservation_event_creation() {
        let event = PortReservationEvent {
            timestamp: chrono::Utc::now(),
            event_type: ReservationEventType::Created,
            port: 8080,
            test_id: "test_001".to_string(),
            details: HashMap::new(),
        };
        assert_eq!(event.port, 8080);
        assert_eq!(event.test_id, "test_001");
        assert!(matches!(
            event.event_type,
            ReservationEventType::Created
        ));
    }

    #[test]
    fn test_conflict_event_creation() {
        let event = PortConflictEvent {
            timestamp: chrono::Utc::now(),
            port: 8080,
            requesting_test_id: "test_001".to_string(),
            current_owner_test_id: Some("test_002".to_string()),
            conflict_type: ConflictType::AlreadyAllocated,
            resolution_action: ConflictAction::FindAlternative,
            resolved: true,
            details: HashMap::new(),
        };
        assert_eq!(event.port, 8080);
        assert!(event.resolved);
    }

    #[test]
    fn test_health_event_creation() {
        let event = PortHealthEvent {
            timestamp: chrono::Utc::now(),
            status: HealthStatus::Healthy,
            metrics: HashMap::new(),
            alerts: vec!["test alert".to_string()],
            details: HashMap::new(),
        };
        assert_eq!(event.alerts.len(), 1);
    }

    #[test]
    fn test_port_management_result_type_alias() {
        let ok_result: PortManagementResult<u16> = Ok(8080);
        assert!(ok_result.is_ok());
        let err_result: PortManagementResult<u16> =
            Err(PortManagementError::PortNotAllocated { port: 8080 });
        assert!(err_result.is_err());
    }

    #[test]
    fn test_lcg_produces_varied_values() {
        let mut rng = Lcg::new(42);
        let mut values = Vec::new();
        for _ in 0..20 {
            values.push(rng.next());
        }
        // All values should be different
        let mut unique = values.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), 20);
    }

    #[test]
    fn test_lcg_f32_range() {
        let mut rng = Lcg::new(42);
        for _ in 0..100 {
            let val = rng.next_f32();
            assert!(val >= 0.0);
            assert!(val <= 1.0);
        }
    }
}
