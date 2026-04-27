//! Auto-generated module structure

pub mod batchconfig_traits;
pub mod types;
pub mod functions;

// Re-export all types
pub use batchconfig_traits::*;
pub use types::*;
pub use functions::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    // ── TaskPriority ──────────────────────────────────────────────────────

    #[test]
    fn test_task_priority_ordering_critical_highest() {
        assert!(TaskPriority::Critical > TaskPriority::High);
        assert!(TaskPriority::High > TaskPriority::Normal);
        assert!(TaskPriority::Normal > TaskPriority::Low);
    }

    #[test]
    fn test_task_priority_equality() {
        assert_eq!(TaskPriority::Normal, TaskPriority::Normal);
        assert_ne!(TaskPriority::Low, TaskPriority::High);
    }

    #[test]
    fn test_task_priority_clone() {
        let p = TaskPriority::Critical;
        let q = p.clone();
        assert_eq!(p, q);
    }

    // ── BlockStatus ───────────────────────────────────────────────────────

    #[test]
    fn test_block_status_all_variants_distinct() {
        let a = BlockStatus::Available;
        let b = BlockStatus::Allocated;
        let c = BlockStatus::Reserved;
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    #[test]
    fn test_block_status_available_debug() {
        assert!(format!("{:?}", BlockStatus::Available).contains("Available"));
    }

    // ── AllocationStrategy ────────────────────────────────────────────────

    #[test]
    fn test_allocation_strategy_all_variants_debug() {
        assert!(format!("{:?}", AllocationStrategy::FirstFit).contains("FirstFit"));
        assert!(format!("{:?}", AllocationStrategy::BestFit).contains("BestFit"));
        assert!(format!("{:?}", AllocationStrategy::WorstFit).contains("WorstFit"));
        assert!(format!("{:?}", AllocationStrategy::NextFit).contains("NextFit"));
    }

    // ── OperationType ─────────────────────────────────────────────────────

    #[test]
    fn test_operation_type_all_variants_debug() {
        assert!(format!("{:?}", OperationType::Encryption).contains("Encryption"));
        assert!(format!("{:?}", OperationType::Decryption).contains("Decryption"));
        assert!(format!("{:?}", OperationType::KeyDerivation).contains("KeyDerivation"));
        assert!(format!("{:?}", OperationType::HashComputation).contains("HashComputation"));
        assert!(format!("{:?}", OperationType::DigitalSignature).contains("DigitalSignature"));
        assert!(format!("{:?}", OperationType::KeyExchange).contains("KeyExchange"));
    }

    #[test]
    fn test_operation_type_hash_equality() {
        use std::collections::HashMap;
        let mut m: HashMap<OperationType, usize> = HashMap::new();
        m.insert(OperationType::Encryption, 1);
        m.insert(OperationType::Decryption, 2);
        assert_eq!(m.get(&OperationType::Encryption), Some(&1));
        assert_ne!(m.get(&OperationType::Encryption), m.get(&OperationType::Decryption));
    }

    // ── GCStrategy ────────────────────────────────────────────────────────

    #[test]
    fn test_gc_strategy_all_variants_debug() {
        assert!(format!("{:?}", GCStrategy::ReferenceCounting).contains("ReferenceCounting"));
        assert!(format!("{:?}", GCStrategy::MarkAndSweep).contains("MarkAndSweep"));
        assert!(format!("{:?}", GCStrategy::Generational).contains("Generational"));
        assert!(format!("{:?}", GCStrategy::Incremental).contains("Incremental"));
    }

    // ── BatchStatus ───────────────────────────────────────────────────────

    #[test]
    fn test_batch_status_all_variants_debug() {
        assert!(format!("{:?}", BatchStatus::Building).contains("Building"));
        assert!(format!("{:?}", BatchStatus::Ready).contains("Ready"));
        assert!(format!("{:?}", BatchStatus::Processing).contains("Processing"));
        assert!(format!("{:?}", BatchStatus::Completed).contains("Completed"));
        assert!(format!("{:?}", BatchStatus::Failed).contains("Failed"));
    }

    #[test]
    fn test_batch_status_equality() {
        assert_eq!(BatchStatus::Completed, BatchStatus::Completed);
        assert_ne!(BatchStatus::Building, BatchStatus::Failed);
    }

    // ── ProfilerOutputFormat ──────────────────────────────────────────────

    #[test]
    fn test_profiler_output_format_variants_debug() {
        assert!(format!("{:?}", ProfilerOutputFormat::FlameGraph).contains("FlameGraph"));
        assert!(format!("{:?}", ProfilerOutputFormat::CallGraph).contains("CallGraph"));
        assert!(format!("{:?}", ProfilerOutputFormat::TreeView).contains("TreeView"));
        assert!(format!("{:?}", ProfilerOutputFormat::Statistical).contains("Statistical"));
    }

    // ── BatchConfig ───────────────────────────────────────────────────────

    #[test]
    fn test_batch_config_default_fields() {
        let cfg = BatchConfig::default();
        assert_eq!(cfg.max_batch_size, 100);
        assert!(cfg.auto_flush);
        assert!(!cfg.compression);
        assert_eq!(cfg.batch_timeout, Duration::from_millis(100));
    }

    #[test]
    fn test_batch_config_custom_fields() {
        let cfg = BatchConfig {
            max_batch_size: 500,
            batch_timeout: Duration::from_secs(1),
            auto_flush: false,
            compression: true,
        };
        assert_eq!(cfg.max_batch_size, 500);
        assert!(!cfg.auto_flush);
        assert!(cfg.compression);
    }

    // ── BenchmarkMetrics ──────────────────────────────────────────────────

    #[test]
    fn test_benchmark_metrics_default() {
        let m = BenchmarkMetrics::default();
        assert_eq!(m.operations, 0);
        assert_eq!(m.bytes_processed, 0);
        assert_eq!(m.cpu_usage, 0.0);
    }

    #[test]
    fn test_benchmark_metrics_fields() {
        let m = BenchmarkMetrics {
            execution_time: Duration::from_millis(250),
            operations: 10_000,
            bytes_processed: 1_048_576,
            cpu_usage: 0.72_f64,
            memory_usage: 256 * 1024 * 1024,
        };
        assert_eq!(m.operations, 10_000);
        assert_eq!(m.bytes_processed, 1_048_576);
        assert!((m.cpu_usage - 0.72).abs() < 1e-9);
    }

    // ── ResourceUtilization ───────────────────────────────────────────────

    #[test]
    fn test_resource_utilization_default() {
        let r = ResourceUtilization::default();
        assert_eq!(r.cpu_usage, 0.0);
        assert_eq!(r.memory_usage, 0.0);
        assert!(r.gpu_usage.is_none());
    }

    #[test]
    fn test_resource_utilization_with_gpu() {
        let r = ResourceUtilization {
            cpu_usage: 0.65_f64,
            memory_usage: 0.45_f64,
            gpu_usage: Some(0.80_f64),
            cache_hit_ratio: 0.92_f64,
        };
        assert!(r.gpu_usage.is_some());
        assert!((r.gpu_usage.unwrap_or(0.0) - 0.80).abs() < 1e-9);
    }

    // ── ExecutionTimeStats ────────────────────────────────────────────────

    #[test]
    fn test_execution_time_stats_default() {
        let s = ExecutionTimeStats::default();
        assert_eq!(s.mean, Duration::ZERO);
        assert_eq!(s.min, Duration::ZERO);
        assert_eq!(s.max, Duration::ZERO);
    }

    #[test]
    fn test_execution_time_stats_percentile_ordering() {
        let s = ExecutionTimeStats {
            mean: Duration::from_millis(50),
            median: Duration::from_millis(48),
            std_dev: Duration::from_millis(10),
            min: Duration::from_millis(10),
            max: Duration::from_millis(200),
            p95: Duration::from_millis(120),
            p99: Duration::from_millis(180),
        };
        assert!(s.p95 < s.p99);
        assert!(s.min < s.max);
    }

    // ── ParallelStats / HardwareStats ─────────────────────────────────────

    #[test]
    fn test_parallel_stats_default_zeroed() {
        let ps = ParallelStats::default();
        assert_eq!(ps.parallel_tasks.load(Ordering::Relaxed), 0);
        assert_eq!(ps.queue_size.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_hardware_stats_default_zeroed() {
        let hs = HardwareStats::default();
        assert_eq!(hs.hardware_operations.load(Ordering::Relaxed), 0);
        assert_eq!(hs.hardware_errors.load(Ordering::Relaxed), 0);
    }

    // ── CacheStats ────────────────────────────────────────────────────────

    #[test]
    fn test_cache_stats_default_zeroed() {
        let cs = CacheStats::default();
        assert_eq!(cs.hits.load(Ordering::Relaxed), 0);
        assert_eq!(cs.misses.load(Ordering::Relaxed), 0);
        assert_eq!(cs.evictions.load(Ordering::Relaxed), 0);
    }

    // ── BatchProcessor ────────────────────────────────────────────────────

    #[test]
    fn test_batch_processor_new() {
        let cfg = BatchConfig::default();
        let _processor = BatchProcessor::new(cfg);
        // No panic on construction
    }

    // ── Profiler ──────────────────────────────────────────────────────────

    #[test]
    fn test_profiler_new() {
        let profiler = Profiler::new();
        // Results should start empty
        assert!(profiler.results.read().is_empty());
    }
}
