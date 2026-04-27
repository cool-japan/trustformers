//! Tests for encryption/performance types

#[cfg(test)]
mod tests {
    use super::super::types::*;
    use std::time::{Duration, SystemTime};
    use std::collections::HashMap;

    // ===== TaskPriority tests =====

    #[test]
    fn test_task_priority_ordering() {
        assert!(TaskPriority::Critical > TaskPriority::High);
        assert!(TaskPriority::High > TaskPriority::Normal);
        assert!(TaskPriority::Normal > TaskPriority::Low);
    }

    #[test]
    fn test_task_priority_equality() {
        let a = TaskPriority::High;
        let b = TaskPriority::High;
        assert_eq!(a, b);
        let c = TaskPriority::Low;
        assert_ne!(a, c);
    }

    #[test]
    fn test_task_priority_debug_format() {
        assert!(format!("{:?}", TaskPriority::Critical).contains("Critical"));
        assert!(format!("{:?}", TaskPriority::Normal).contains("Normal"));
    }

    // ===== BlockStatus tests =====

    #[test]
    fn test_block_status_variants_distinct() {
        let available = BlockStatus::Available;
        let allocated = BlockStatus::Allocated;
        let reserved = BlockStatus::Reserved;
        assert_ne!(available, allocated);
        assert_ne!(allocated, reserved);
        assert_ne!(available, reserved);
    }

    // ===== AllocationStrategy tests =====

    #[test]
    fn test_allocation_strategy_variants() {
        let first_fit = AllocationStrategy::FirstFit;
        let best_fit = AllocationStrategy::BestFit;
        let worst_fit = AllocationStrategy::WorstFit;
        let next_fit = AllocationStrategy::NextFit;
        assert!(format!("{:?}", first_fit).contains("FirstFit"));
        assert!(format!("{:?}", best_fit).contains("BestFit"));
        assert!(format!("{:?}", worst_fit).contains("WorstFit"));
        assert!(format!("{:?}", next_fit).contains("NextFit"));
    }

    // ===== OperationType tests =====

    #[test]
    fn test_operation_type_equality_for_hash() {
        use std::collections::HashSet;
        let mut ops = HashSet::new();
        ops.insert(OperationType::Encryption);
        ops.insert(OperationType::Decryption);
        ops.insert(OperationType::KeyDerivation);
        ops.insert(OperationType::HashComputation);
        ops.insert(OperationType::DigitalSignature);
        ops.insert(OperationType::KeyExchange);
        assert_eq!(ops.len(), 6);
    }

    #[test]
    fn test_operation_type_encryption_distinct_decryption() {
        let enc = OperationType::Encryption;
        let dec = OperationType::Decryption;
        assert_ne!(enc, dec);
    }

    // ===== BatchStatus tests =====

    #[test]
    fn test_batch_status_lifecycle() {
        // Simulate lifecycle: Building -> Ready -> Processing -> Completed
        let states = vec![
            BatchStatus::Building,
            BatchStatus::Ready,
            BatchStatus::Processing,
            BatchStatus::Completed,
        ];
        let mut saw_building = false;
        let mut saw_completed = false;
        for status in &states {
            match status {
                BatchStatus::Building => saw_building = true,
                BatchStatus::Completed => saw_completed = true,
                _ => {}
            }
        }
        assert!(saw_building);
        assert!(saw_completed);
    }

    #[test]
    fn test_batch_status_failed_distinct_completed() {
        let failed = BatchStatus::Failed;
        let completed = BatchStatus::Completed;
        assert_ne!(failed, completed);
    }

    // ===== PressureLevel tests =====

    #[test]
    fn test_pressure_level_variants() {
        let levels = vec![
            PressureLevel::None,
            PressureLevel::Low,
            PressureLevel::Medium,
            PressureLevel::High,
            PressureLevel::Critical,
        ];
        assert_eq!(levels.len(), 5);
    }

    #[test]
    fn test_pressure_level_equality() {
        assert_eq!(PressureLevel::None, PressureLevel::None);
        assert_ne!(PressureLevel::Low, PressureLevel::Critical);
    }

    // ===== LoadBalancingStrategy tests =====

    #[test]
    fn test_load_balancing_strategy_variants() {
        let strategies = vec![
            LoadBalancingStrategy::RoundRobin,
            LoadBalancingStrategy::LeastLoaded,
            LoadBalancingStrategy::Random,
            LoadBalancingStrategy::PriorityBased,
        ];
        for s in &strategies {
            let debug_str = format!("{:?}", s);
            assert!(!debug_str.is_empty());
        }
    }

    // ===== TaskDistributionStrategy tests =====

    #[test]
    fn test_task_distribution_strategy_fifo_debug() {
        let fifo = TaskDistributionStrategy::FIFO;
        assert!(format!("{:?}", fifo).contains("FIFO"));
    }

    #[test]
    fn test_task_distribution_strategy_all_variants() {
        let strategies = vec![
            TaskDistributionStrategy::FIFO,
            TaskDistributionStrategy::Priority,
            TaskDistributionStrategy::ShortestJobFirst,
            TaskDistributionStrategy::LoadAware,
        ];
        assert_eq!(strategies.len(), 4);
    }

    // ===== TaskType tests =====

    #[test]
    fn test_task_type_variants_distinct() {
        let encryption = TaskType::Encryption;
        let decryption = TaskType::Decryption;
        let batch = TaskType::BatchOperation;
        let background = TaskType::BackgroundComputation;
        assert_ne!(encryption, decryption);
        assert_ne!(batch, background);
        assert_ne!(encryption, batch);
    }

    // ===== ThroughputStats tests =====

    #[test]
    fn test_throughput_stats_default() {
        let stats = ThroughputStats::default();
        assert_eq!(stats.ops_per_second, 0.0);
        assert_eq!(stats.bytes_per_second, 0.0);
        assert_eq!(stats.peak_throughput, 0.0);
        assert_eq!(stats.average_throughput, 0.0);
    }

    #[test]
    fn test_throughput_stats_peak_gte_average() {
        let mut stats = ThroughputStats::default();
        stats.average_throughput = 1000.0;
        stats.peak_throughput = 2000.0;
        assert!(stats.peak_throughput >= stats.average_throughput);
    }

    // ===== BenchmarkMetrics tests =====

    #[test]
    fn test_benchmark_metrics_default() {
        let metrics = BenchmarkMetrics::default();
        assert_eq!(metrics.operations, 0);
        assert_eq!(metrics.bytes_processed, 0);
        assert_eq!(metrics.cpu_usage, 0.0);
    }

    #[test]
    fn test_benchmark_metrics_clone() {
        let mut metrics = BenchmarkMetrics::default();
        metrics.operations = 1000;
        metrics.bytes_processed = 1024 * 1024;
        metrics.cpu_usage = 0.75;
        let cloned = metrics.clone();
        assert_eq!(cloned.operations, metrics.operations);
        assert_eq!(cloned.bytes_processed, metrics.bytes_processed);
    }

    // ===== AccelerationMetrics tests =====

    #[test]
    fn test_acceleration_metrics_default() {
        let metrics = AccelerationMetrics::default();
        assert_eq!(metrics.operations_accelerated, 0);
        assert_eq!(metrics.performance_gain, 0.0);
        assert_eq!(metrics.energy_savings, 0.0);
        assert_eq!(metrics.error_rate, 0.0);
    }

    #[test]
    fn test_acceleration_metrics_error_rate_range() {
        let mut metrics = AccelerationMetrics::default();
        metrics.error_rate = 0.001; // 0.1% error rate
        assert!(metrics.error_rate >= 0.0 && metrics.error_rate <= 1.0);
    }

    // ===== PoolGrowthStrategy tests =====

    #[test]
    fn test_pool_growth_strategy_dynamic() {
        let strategy = PoolGrowthStrategy::Dynamic { max_size: 1024 };
        if let PoolGrowthStrategy::Dynamic { max_size } = strategy {
            assert!(max_size > 0);
        } else {
            panic!("Expected Dynamic variant");
        }
    }

    #[test]
    fn test_pool_growth_strategy_variants() {
        let fixed = PoolGrowthStrategy::Fixed;
        let on_demand = PoolGrowthStrategy::OnDemand;
        assert!(format!("{:?}", fixed).contains("Fixed"));
        assert!(format!("{:?}", on_demand).contains("OnDemand"));
    }

    // ===== HardwareFeatures tests =====

    #[test]
    fn test_hardware_features_detect_cpu_cores_nonzero() {
        let features = HardwareFeatures::detect();
        assert!(features.cpu_cores > 0, "CPU core count must be positive");
    }

    #[test]
    fn test_hardware_features_debug_format() {
        let features = HardwareFeatures::detect();
        let debug_str = format!("{:?}", features);
        assert!(debug_str.contains("HardwareFeatures"));
        assert!(debug_str.contains("aes_ni"));
    }

    // ===== PressureThresholds tests =====

    #[test]
    fn test_pressure_thresholds_ordering() {
        let thresholds = PressureThresholds {
            low: 0.5,
            medium: 0.7,
            high: 0.85,
            critical: 0.95,
        };
        assert!(thresholds.low < thresholds.medium);
        assert!(thresholds.medium < thresholds.high);
        assert!(thresholds.high < thresholds.critical);
        assert!(thresholds.critical <= 1.0);
    }

    // ===== ResourceUtilization tests =====

    #[test]
    fn test_resource_utilization_default() {
        let util = ResourceUtilization::default();
        assert_eq!(util.cpu_usage, 0.0);
        assert_eq!(util.memory_usage, 0.0);
        assert_eq!(util.cache_hit_ratio, 0.0);
        assert!(util.gpu_usage.is_none());
    }

    #[test]
    fn test_resource_utilization_gpu_optional() {
        let mut util = ResourceUtilization::default();
        util.gpu_usage = Some(0.82);
        assert!(util.gpu_usage.is_some());
        let gpu = util.gpu_usage.expect("GPU usage should be set");
        assert!(gpu >= 0.0 && gpu <= 1.0);
    }
}
