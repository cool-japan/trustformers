//! Tests for disaster_recovery types

#[cfg(test)]
mod tests {
    use super::super::types::*;
    use std::path::PathBuf;
    use std::time::Duration;

    // ===== SiteStatus tests =====

    #[test]
    fn test_site_status_variants_distinct() {
        let active = SiteStatus::Active;
        let standby = SiteStatus::Standby;
        let unhealthy = SiteStatus::Unhealthy;
        let maintenance = SiteStatus::Maintenance;
        let activating = SiteStatus::Activating;
        let deactivating = SiteStatus::Deactivating;
        let unknown = SiteStatus::Unknown;
        assert_ne!(active, standby);
        assert_ne!(active, unhealthy);
        assert_ne!(standby, maintenance);
        assert_ne!(activating, deactivating);
        assert_ne!(maintenance, unknown);
    }

    #[test]
    fn test_site_status_equality() {
        let a = SiteStatus::Active;
        let b = SiteStatus::Active;
        assert_eq!(a, b);
        let c = SiteStatus::Unhealthy;
        assert_ne!(a, c);
    }

    // ===== SiteType tests =====

    #[test]
    fn test_site_type_variants_debug() {
        let primary = SiteType::Primary;
        let dr = SiteType::DisasterRecovery;
        let hot = SiteType::HotStandby;
        let cold = SiteType::ColdStandby;
        let backup = SiteType::Backup;
        assert!(format!("{:?}", primary).contains("Primary"));
        assert!(format!("{:?}", dr).contains("DisasterRecovery"));
        assert!(format!("{:?}", hot).contains("HotStandby"));
        assert!(format!("{:?}", cold).contains("ColdStandby"));
        assert!(format!("{:?}", backup).contains("Backup"));
    }

    // ===== FailoverStrategy tests =====

    #[test]
    fn test_failover_strategy_highest_priority() {
        let strategy = FailoverStrategy::HighestPriority;
        assert!(format!("{:?}", strategy).contains("HighestPriority"));
    }

    #[test]
    fn test_failover_strategy_geographic() {
        let strategy = FailoverStrategy::Geographic {
            preferred_regions: vec!["us-east-1".to_string(), "eu-west-1".to_string()],
        };
        if let FailoverStrategy::Geographic { preferred_regions } = &strategy {
            assert_eq!(preferred_regions.len(), 2);
            assert_eq!(preferred_regions[0], "us-east-1");
        } else {
            panic!("Expected Geographic variant");
        }
    }

    #[test]
    fn test_failover_strategy_custom() {
        let strategy = FailoverStrategy::Custom {
            strategy_name: "latency-aware".to_string(),
        };
        if let FailoverStrategy::Custom { strategy_name } = &strategy {
            assert_eq!(strategy_name, "latency-aware");
        } else {
            panic!("Expected Custom variant");
        }
    }

    // ===== FailoverTrigger tests =====

    #[test]
    fn test_failover_trigger_site_unavailable() {
        let trigger = FailoverTrigger::SiteUnavailable {
            site_id: "primary-us-east".to_string(),
        };
        if let FailoverTrigger::SiteUnavailable { site_id } = &trigger {
            assert_eq!(site_id, "primary-us-east");
        } else {
            panic!("Expected SiteUnavailable variant");
        }
    }

    #[test]
    fn test_failover_trigger_high_error_rate() {
        let trigger = FailoverTrigger::HighErrorRate {
            threshold: 0.05,
            duration_seconds: 300,
        };
        if let FailoverTrigger::HighErrorRate {
            threshold,
            duration_seconds,
        } = &trigger
        {
            assert!(*threshold > 0.0 && *threshold < 1.0);
            assert_eq!(*duration_seconds, 300);
        } else {
            panic!("Expected HighErrorRate variant");
        }
    }

    #[test]
    fn test_failover_trigger_high_latency() {
        let trigger = FailoverTrigger::HighLatency {
            threshold_ms: 5000,
            duration_seconds: 60,
        };
        if let FailoverTrigger::HighLatency {
            threshold_ms,
            duration_seconds,
        } = &trigger
        {
            assert_eq!(*threshold_ms, 5000);
            assert_eq!(*duration_seconds, 60);
        } else {
            panic!("Expected HighLatency variant");
        }
    }

    // ===== RollbackCondition tests =====

    #[test]
    fn test_rollback_condition_variants() {
        let primary_recovered = RollbackCondition::PrimarySiteRecovered;
        let dr_unhealthy = RollbackCondition::DRSiteUnhealthy;
        let perf_degraded = RollbackCondition::PerformanceDegradation { threshold: 0.8 };
        let manual = RollbackCondition::Manual {
            reason: "operator request".to_string(),
        };

        assert!(format!("{:?}", primary_recovered).contains("PrimarySiteRecovered"));
        assert!(format!("{:?}", dr_unhealthy).contains("DRSiteUnhealthy"));
        if let RollbackCondition::PerformanceDegradation { threshold } = &perf_degraded {
            assert!(*threshold > 0.0 && *threshold <= 1.0);
        }
        if let RollbackCondition::Manual { reason } = &manual {
            assert!(!reason.is_empty());
        }
    }

    // ===== DRAlertThresholds tests =====

    #[test]
    fn test_dr_alert_thresholds_creation() {
        let thresholds = DRAlertThresholds {
            rto_threshold_seconds: 3600,
            rpo_threshold_seconds: 900,
            replication_lag_threshold_seconds: 60,
            backup_failure_threshold: 3,
            site_unavailable_threshold_seconds: 300,
        };
        assert!(thresholds.rto_threshold_seconds >= thresholds.rpo_threshold_seconds);
        assert!(thresholds.backup_failure_threshold > 0);
        assert!(thresholds.replication_lag_threshold_seconds <= thresholds.rpo_threshold_seconds);
    }

    // ===== NotificationSeverity tests =====

    #[test]
    fn test_notification_severity_ordering() {
        let severities = vec![
            NotificationSeverity::Low,
            NotificationSeverity::Medium,
            NotificationSeverity::High,
            NotificationSeverity::Critical,
        ];
        assert_eq!(severities.len(), 4);
        assert!(format!("{:?}", NotificationSeverity::Critical).contains("Critical"));
        assert!(format!("{:?}", NotificationSeverity::Low).contains("Low"));
    }

    // ===== ReplicationHealth tests =====

    #[test]
    fn test_replication_health_variants() {
        let health_healthy = ReplicationHealth::Healthy;
        let health_lagging = ReplicationHealth::Lagging;
        let health_failed = ReplicationHealth::Failed;
        let health_unknown = ReplicationHealth::Unknown;
        assert!(format!("{:?}", health_healthy).contains("Healthy"));
        assert!(format!("{:?}", health_lagging).contains("Lagging"));
        assert!(format!("{:?}", health_failed).contains("Failed"));
        assert!(format!("{:?}", health_unknown).contains("Unknown"));
    }

    // ===== ConsistencyLevel tests =====

    #[test]
    fn test_consistency_level_bounded_staleness() {
        let bounded = ConsistencyLevel::BoundedStaleness {
            max_lag_seconds: 30,
        };
        if let ConsistencyLevel::BoundedStaleness { max_lag_seconds } = bounded {
            assert_eq!(max_lag_seconds, 30);
        } else {
            panic!("Expected BoundedStaleness");
        }
    }

    #[test]
    fn test_consistency_level_variants() {
        let strong = ConsistencyLevel::StrongConsistency;
        let eventual = ConsistencyLevel::EventualConsistency;
        let session = ConsistencyLevel::SessionConsistency;
        assert!(format!("{:?}", strong).contains("Strong"));
        assert!(format!("{:?}", eventual).contains("Eventual"));
        assert!(format!("{:?}", session).contains("Session"));
    }

    // ===== BackupStatus tests =====

    #[test]
    fn test_backup_status_success_rate_range() {
        let status = BackupStatus {
            last_backup_time: None,
            backup_in_progress: false,
            last_backup_size_bytes: 1024 * 1024 * 512,
            success_rate: 0.98,
            next_backup_time: None,
        };
        assert!(status.success_rate >= 0.0 && status.success_rate <= 1.0);
        assert!(!status.backup_in_progress);
    }

    // ===== SiteCapacity tests =====

    #[test]
    fn test_site_capacity_compute_capacity_range() {
        let capacity = SiteCapacity {
            max_requests_per_second: 10000,
            max_concurrent_users: 50000,
            storage_capacity_gb: 10000,
            compute_capacity: 0.75,
        };
        assert!(capacity.compute_capacity >= 0.0 && capacity.compute_capacity <= 1.0);
        assert!(capacity.max_requests_per_second > 0);
    }

    // ===== RetentionPolicy tests =====

    #[test]
    fn test_retention_policy_creation() {
        let policy = RetentionPolicy {
            daily_backups: 7,
            weekly_backups: 4,
            monthly_backups: 12,
            yearly_backups: 3,
        };
        assert!(policy.daily_backups <= 365);
        assert!(policy.weekly_backups <= 52);
        assert!(policy.monthly_backups <= 24);
    }

    // ===== DREventType tests =====

    #[test]
    fn test_dr_event_type_all_variants() {
        let event_types = [
            DREventType::FailoverTriggered,
            DREventType::FailoverCompleted,
            DREventType::FailoverFailed,
            DREventType::BackupStarted,
            DREventType::BackupCompleted,
            DREventType::BackupFailed,
            DREventType::RestoreStarted,
            DREventType::RestoreCompleted,
            DREventType::TestStarted,
            DREventType::TestCompleted,
        ];
        assert_eq!(event_types.len(), 10);
    }

    // ===== DRError tests =====

    #[test]
    fn test_dr_error_display() {
        let config_err = DRError::ConfigurationError {
            message: "invalid config".to_string(),
        };
        let site_err = DRError::SiteNotFound {
            site_id: "site-xyz".to_string(),
        };
        let failover_err = DRError::FailoverError {
            message: "failover timeout".to_string(),
        };
        let replication_err = DRError::ReplicationError {
            message: "lag exceeded".to_string(),
        };
        let backup_err = DRError::BackupError {
            message: "disk full".to_string(),
        };
        assert!(config_err.to_string().contains("Configuration error"));
        assert!(site_err.to_string().contains("site-xyz"));
        assert!(failover_err.to_string().contains("Failover error"));
        assert!(replication_err.to_string().contains("Replication error"));
        assert!(backup_err.to_string().contains("Backup error"));
    }

    // ===== StorageType tests =====

    #[test]
    fn test_storage_type_local() {
        let storage = StorageType::Local {
            path: PathBuf::from("/data/backups"),
        };
        if let StorageType::Local { path } = storage {
            assert!(path.starts_with("/data"));
        } else {
            panic!("Expected Local variant");
        }
    }

    #[test]
    fn test_storage_type_s3() {
        let storage = StorageType::S3 {
            bucket: "my-backups".to_string(),
            region: "us-east-1".to_string(),
            access_key_id: "AKID...".to_string(),
            secret_access_key: "secret...".to_string(),
        };
        if let StorageType::S3 { bucket, region, .. } = &storage {
            assert_eq!(bucket, "my-backups");
            assert_eq!(region, "us-east-1");
        } else {
            panic!("Expected S3 variant");
        }
    }

    // ===== BackupType tests =====

    #[test]
    fn test_backup_type_variants() {
        let full = BackupType::Full;
        let incremental = BackupType::Incremental;
        let differential = BackupType::Differential;
        assert!(format!("{:?}", full).contains("Full"));
        assert!(format!("{:?}", incremental).contains("Incremental"));
        assert!(format!("{:?}", differential).contains("Differential"));
    }

    // ===== BackupStrategy tests =====

    #[test]
    fn test_backup_strategy_full() {
        let strategy = BackupStrategy::Full {
            interval: Duration::from_secs(24 * 3600),
        };
        if let BackupStrategy::Full { interval } = strategy {
            assert_eq!(interval.as_secs(), 24 * 3600);
        } else {
            panic!("Expected Full strategy");
        }
    }

    #[test]
    fn test_backup_strategy_incremental() {
        let strategy = BackupStrategy::Incremental {
            full_backup_interval: Duration::from_secs(7 * 24 * 3600),
            incremental_interval: Duration::from_secs(3600),
        };
        if let BackupStrategy::Incremental {
            full_backup_interval,
            incremental_interval,
        } = strategy
        {
            assert!(full_backup_interval > incremental_interval);
        } else {
            panic!("Expected Incremental strategy");
        }
    }

    // ===== TrafficStage tests =====

    #[test]
    fn test_traffic_stage_percentage_bounds() {
        let stage = TrafficStage {
            percentage: 25,
            duration_seconds: 300,
        };
        assert!(stage.percentage <= 100);
        assert!(stage.duration_seconds > 0);
    }
}
