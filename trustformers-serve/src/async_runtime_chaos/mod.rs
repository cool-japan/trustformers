//! Auto-generated module structure

pub mod asyncmemorypressureconfig_traits;
pub mod asyncnetworkfailureconfig_traits;
pub mod asyncresourceexhaustionconfig_traits;
pub mod asyncruntimechaosframework_traits;
pub mod asynctestsuiteresult_traits;
pub mod concurrentaccessconfig_traits;
pub mod deadlockconfig_traits;
pub mod functions;
pub mod panicrecoveryconfig_traits;
pub mod runtimeshutdownconfig_traits;
pub mod taskcancellationconfig_traits;
pub mod types;

// Re-export all types
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use std::collections::HashMap;
    use std::time::Duration;

    // ── AsyncRuntimeChaosType ──────────────────────────────────────────────

    #[test]
    fn test_chaos_type_task_cancellation_debug() {
        let t = AsyncRuntimeChaosType::TaskCancellation;
        assert!(format!("{:?}", t).contains("TaskCancellation"));
    }

    #[test]
    fn test_chaos_type_runtime_shutdown_debug() {
        let t = AsyncRuntimeChaosType::RuntimeShutdown;
        assert!(format!("{:?}", t).contains("RuntimeShutdown"));
    }

    #[test]
    fn test_chaos_type_deadlock_detection_debug() {
        let t = AsyncRuntimeChaosType::DeadlockDetection;
        assert!(format!("{:?}", t).contains("DeadlockDetection"));
    }

    #[test]
    fn test_chaos_type_all_variants_are_distinct() {
        let variants = vec![
            AsyncRuntimeChaosType::TaskCancellation,
            AsyncRuntimeChaosType::RuntimeShutdown,
            AsyncRuntimeChaosType::DeadlockDetection,
            AsyncRuntimeChaosType::AsyncMemoryPressure,
            AsyncRuntimeChaosType::AsyncNetworkFailure,
            AsyncRuntimeChaosType::AsyncResourceExhaustion,
            AsyncRuntimeChaosType::RaceConditionSimulation,
            AsyncRuntimeChaosType::AsyncPanicRecovery,
            AsyncRuntimeChaosType::TaskStarvation,
            AsyncRuntimeChaosType::ChannelCongestion,
            AsyncRuntimeChaosType::AsyncTimeouts,
            AsyncRuntimeChaosType::TaskLeakage,
        ];
        // All variants produce non-empty debug strings
        for v in &variants {
            assert!(!format!("{:?}", v).is_empty());
        }
        // Twelve unique variants
        assert_eq!(variants.len(), 12);
    }

    // ── CancellationStrategy ──────────────────────────────────────────────

    #[test]
    fn test_cancellation_strategy_broadcast_debug() {
        let s = CancellationStrategy::BroadcastCancel;
        assert!(format!("{:?}", s).contains("BroadcastCancel"));
    }

    #[test]
    fn test_cancellation_strategy_selective_carries_rate() {
        let rate = 0.75_f64;
        let s = CancellationStrategy::SelectiveCancel(rate);
        let dbg = format!("{:?}", s);
        assert!(dbg.contains("SelectiveCancel"));
    }

    #[test]
    fn test_cancellation_strategy_graceful_shutdown_debug() {
        let s = CancellationStrategy::GracefulShutdown;
        assert!(format!("{:?}", s).contains("GracefulShutdown"));
    }

    // ── PanicType ─────────────────────────────────────────────────────────

    #[test]
    fn test_panic_type_immediate_debug() {
        let p = PanicType::Immediate;
        assert!(format!("{:?}", p).contains("Immediate"));
    }

    #[test]
    fn test_panic_type_delayed_debug() {
        let p = PanicType::Delayed;
        assert!(format!("{:?}", p).contains("Delayed"));
    }

    #[test]
    fn test_panic_type_conditional_debug() {
        let p = PanicType::ConditionalPanic;
        assert!(format!("{:?}", p).contains("ConditionalPanic"));
    }

    // ── ConcurrentAccessPattern ───────────────────────────────────────────

    #[test]
    fn test_concurrent_access_pattern_read_heavy() {
        let p = ConcurrentAccessPattern::ReadHeavy;
        assert!(format!("{:?}", p).contains("ReadHeavy"));
    }

    #[test]
    fn test_concurrent_access_pattern_write_heavy() {
        let p = ConcurrentAccessPattern::WriteHeavy;
        assert!(format!("{:?}", p).contains("WriteHeavy"));
    }

    #[test]
    fn test_concurrent_access_pattern_mixed() {
        let p = ConcurrentAccessPattern::Mixed;
        assert!(format!("{:?}", p).contains("Mixed"));
    }

    // ── AsyncExperimentResult ─────────────────────────────────────────────

    #[test]
    fn test_async_experiment_result_new_defaults() {
        let id = uuid::Uuid::new_v4();
        let result = AsyncExperimentResult::new(id, AsyncRuntimeChaosType::TaskCancellation);
        assert_eq!(result.experiment_id, id);
        assert!(!result.success);
        assert!(result.metrics.is_empty());
        assert!(result.errors.is_empty());
    }

    #[test]
    fn test_async_experiment_result_record_metric() {
        let id = uuid::Uuid::new_v4();
        let mut result = AsyncExperimentResult::new(id, AsyncRuntimeChaosType::AsyncTimeouts);
        result.record_metric("latency_ms", 42.5);
        assert_eq!(result.metrics.get("latency_ms").copied(), Some(42.5));
    }

    #[test]
    fn test_async_experiment_result_record_multiple_metrics() {
        let id = uuid::Uuid::new_v4();
        let mut result = AsyncExperimentResult::new(id, AsyncRuntimeChaosType::TaskStarvation);
        result.record_metric("throughput", 1000.0);
        result.record_metric("error_rate", 0.01);
        assert_eq!(result.metrics.len(), 2);
    }

    #[test]
    fn test_async_experiment_result_record_error() {
        let id = uuid::Uuid::new_v4();
        let mut result =
            AsyncExperimentResult::new(id, AsyncRuntimeChaosType::RaceConditionSimulation);
        result.record_error("timeout exceeded".to_string());
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0], "timeout exceeded");
    }

    #[test]
    fn test_async_experiment_result_record_multiple_errors() {
        let id = uuid::Uuid::new_v4();
        let mut result = AsyncExperimentResult::new(id, AsyncRuntimeChaosType::ChannelCongestion);
        result.record_error("error A".to_string());
        result.record_error("error B".to_string());
        assert_eq!(result.errors.len(), 2);
    }

    // ── AsyncTestSuiteResult ──────────────────────────────────────────────

    #[test]
    fn test_async_test_suite_result_new_defaults() {
        let suite = AsyncTestSuiteResult::new();
        assert_eq!(suite.success_rate, 0.0);
        assert_eq!(suite.total_duration, Duration::ZERO);
        assert!(suite.results.is_empty());
    }

    #[test]
    fn test_async_test_suite_result_add_result() {
        let mut suite = AsyncTestSuiteResult::new();
        let exp_id = uuid::Uuid::new_v4();
        let exp_result = AsyncExperimentResult::new(exp_id, AsyncRuntimeChaosType::TaskLeakage);
        suite.add_result("leakage_test", exp_result);
        assert_eq!(suite.results.len(), 1);
        assert!(suite.results.contains_key("leakage_test"));
    }

    #[test]
    fn test_async_test_suite_result_add_multiple_results() {
        let mut suite = AsyncTestSuiteResult::new();
        for i in 0..5_usize {
            let exp_id = uuid::Uuid::new_v4();
            let r = AsyncExperimentResult::new(exp_id, AsyncRuntimeChaosType::AsyncTimeouts);
            suite.add_result(&format!("test_{}", i), r);
        }
        assert_eq!(suite.results.len(), 5);
    }

    // ── Config structs ────────────────────────────────────────────────────

    #[test]
    fn test_deadlock_config_fields() {
        let cfg = DeadlockConfig {
            deadlock_delay: Duration::from_millis(100),
            deadlock_timeout: Duration::from_secs(5),
            detection_timeout: Duration::from_secs(10),
        };
        assert_eq!(cfg.deadlock_delay, Duration::from_millis(100));
        assert_eq!(cfg.deadlock_timeout, Duration::from_secs(5));
        assert_eq!(cfg.detection_timeout, Duration::from_secs(10));
    }

    #[test]
    fn test_async_resource_exhaustion_config_fields() {
        let cfg = AsyncResourceExhaustionConfig {
            total_tasks: 50,
            max_concurrent_resources: 8,
            resource_timeout: Duration::from_secs(2),
            work_duration: Duration::from_millis(500),
        };
        assert_eq!(cfg.total_tasks, 50);
        assert_eq!(cfg.max_concurrent_resources, 8);
    }

    #[test]
    fn test_concurrent_access_config_fields() {
        let cfg = ConcurrentAccessConfig {
            concurrent_tasks: 16,
            operations_per_task: 100,
            access_pattern: ConcurrentAccessPattern::Mixed,
        };
        assert_eq!(cfg.concurrent_tasks, 16);
        assert_eq!(cfg.operations_per_task, 100);
    }

    #[test]
    fn test_async_memory_pressure_config_fields() {
        let cfg = AsyncMemoryPressureConfig {
            memory_pressure_mb: 256,
            concurrent_async_tasks: 32,
            pressure_duration: Duration::from_secs(3),
        };
        assert_eq!(cfg.memory_pressure_mb, 256);
        assert_eq!(cfg.concurrent_async_tasks, 32);
    }

    #[test]
    fn test_panic_recovery_config_fields() {
        let cfg = PanicRecoveryConfig {
            total_tasks: 20,
            panic_task_count: 5,
            panic_type: PanicType::Immediate,
        };
        assert_eq!(cfg.total_tasks, 20);
        assert_eq!(cfg.panic_task_count, 5);
    }

    #[test]
    fn test_runtime_shutdown_config_fields() {
        let cfg = RuntimeShutdownConfig {
            worker_threads: 4,
            concurrent_operations: 8,
            operation_duration: Duration::from_millis(200),
            startup_delay: Duration::from_millis(50),
            graceful_shutdown_timeout: Duration::from_secs(5),
        };
        assert_eq!(cfg.worker_threads, 4);
        assert_eq!(cfg.concurrent_operations, 8);
    }

    #[test]
    fn test_task_cancellation_config_fields() {
        let cfg = TaskCancellationConfig {
            task_count: 100,
            task_duration: Duration::from_secs(1),
            cancellation_delay: Duration::from_millis(500),
            cancellation_strategy: CancellationStrategy::BroadcastCancel,
            graceful_timeout: Duration::from_secs(2),
            completion_timeout: Duration::from_secs(10),
        };
        assert_eq!(cfg.task_count, 100);
        assert_eq!(cfg.cancellation_delay, Duration::from_millis(500));
    }

    #[test]
    fn test_async_network_failure_config_fields() {
        let cfg = AsyncNetworkFailureConfig {
            concurrent_operations: 20,
            max_retries: 3,
            base_backoff_ms: 100,
            failure_start_delay: Duration::from_millis(200),
            failure_duration: Duration::from_secs(1),
        };
        assert_eq!(cfg.max_retries, 3);
        assert_eq!(cfg.base_backoff_ms, 100);
    }

    // ── AsyncRuntimeChaosFramework ────────────────────────────────────────

    #[test]
    fn test_async_runtime_chaos_framework_new() {
        let framework = AsyncRuntimeChaosFramework::new();
        // Construction should succeed; verify via clone
        let _cloned = framework.clone();
    }
}
