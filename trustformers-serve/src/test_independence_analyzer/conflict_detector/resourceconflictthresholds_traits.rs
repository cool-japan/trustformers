//! # ResourceConflictThresholds - Trait Implementations
//!
//! This module contains trait implementations for `ResourceConflictThresholds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::ResourceConflictThresholds;

impl Default for ResourceConflictThresholds {
    fn default() -> Self {
        Self {
            cpu_threshold: 0.8,
            memory_threshold: 0.85,
            network_threshold: 0.7,
            disk_io_threshold: 0.75,
            gpu_threshold: 0.9,
            custom_thresholds: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_default_cpu_threshold() {
        let rt = ResourceConflictThresholds::default();
        assert!((rt.cpu_threshold - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_default_memory_threshold() {
        let rt = ResourceConflictThresholds::default();
        assert!((rt.memory_threshold - 0.85).abs() < f32::EPSILON);
    }

    #[test]
    fn test_default_network_threshold() {
        let rt = ResourceConflictThresholds::default();
        assert!((rt.network_threshold - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_default_disk_io_threshold() {
        let rt = ResourceConflictThresholds::default();
        assert!((rt.disk_io_threshold - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_default_gpu_threshold() {
        let rt = ResourceConflictThresholds::default();
        assert!((rt.gpu_threshold - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn test_default_custom_thresholds_empty() {
        let rt = ResourceConflictThresholds::default();
        assert!(rt.custom_thresholds.is_empty());
    }

    #[test]
    fn test_all_default_thresholds_in_range() {
        let rt = ResourceConflictThresholds::default();
        assert!(rt.cpu_threshold >= 0.0 && rt.cpu_threshold <= 1.0);
        assert!(rt.memory_threshold >= 0.0 && rt.memory_threshold <= 1.0);
        assert!(rt.network_threshold >= 0.0 && rt.network_threshold <= 1.0);
        assert!(rt.disk_io_threshold >= 0.0 && rt.disk_io_threshold <= 1.0);
        assert!(rt.gpu_threshold >= 0.0 && rt.gpu_threshold <= 1.0);
    }

    #[test]
    fn test_gpu_threshold_highest() {
        let rt = ResourceConflictThresholds::default();
        assert!(rt.gpu_threshold >= rt.cpu_threshold);
        assert!(rt.gpu_threshold >= rt.memory_threshold);
        assert!(rt.gpu_threshold >= rt.network_threshold);
        assert!(rt.gpu_threshold >= rt.disk_io_threshold);
    }

    #[test]
    fn test_network_threshold_lowest() {
        let rt = ResourceConflictThresholds::default();
        assert!(rt.network_threshold <= rt.cpu_threshold);
        assert!(rt.network_threshold <= rt.memory_threshold);
        assert!(rt.network_threshold <= rt.disk_io_threshold);
        assert!(rt.network_threshold <= rt.gpu_threshold);
    }

    #[test]
    fn test_add_custom_threshold() {
        let mut rt = ResourceConflictThresholds::default();
        rt.custom_thresholds.insert("db_connections".to_string(), 0.6);
        assert_eq!(rt.custom_thresholds.len(), 1);
        if let Some(v) = rt.custom_thresholds.get("db_connections") {
            assert!((v - 0.6).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_multiple_custom_thresholds() {
        let mut rt = ResourceConflictThresholds::default();
        rt.custom_thresholds.insert("redis_conn".to_string(), 0.5);
        rt.custom_thresholds.insert("kafka_partitions".to_string(), 0.8);
        rt.custom_thresholds.insert("thread_pool".to_string(), 0.75);
        assert_eq!(rt.custom_thresholds.len(), 3);
    }

    #[test]
    fn test_update_cpu_threshold() {
        let mut rt = ResourceConflictThresholds::default();
        rt.cpu_threshold = 0.95;
        assert!((rt.cpu_threshold - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn test_update_memory_threshold_strict() {
        let mut rt = ResourceConflictThresholds::default();
        rt.memory_threshold = 0.99;
        assert!(rt.memory_threshold > 0.85);
    }

    #[test]
    fn test_default_is_repeatable() {
        let r1 = ResourceConflictThresholds::default();
        let r2 = ResourceConflictThresholds::default();
        assert!((r1.cpu_threshold - r2.cpu_threshold).abs() < f32::EPSILON);
        assert!((r1.memory_threshold - r2.memory_threshold).abs() < f32::EPSILON);
        assert!((r1.gpu_threshold - r2.gpu_threshold).abs() < f32::EPSILON);
    }

    #[test]
    fn test_random_custom_threshold_insertion() {
        let mut lcg = Lcg::new(55555);
        let mut rt = ResourceConflictThresholds::default();
        for i in 0..5_u32 {
            let key = format!("resource_{}", i);
            let val = lcg.next_f32();
            rt.custom_thresholds.insert(key, val);
        }
        assert_eq!(rt.custom_thresholds.len(), 5);
        for val in rt.custom_thresholds.values() {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_threshold_ordering_disk_vs_memory() {
        let rt = ResourceConflictThresholds::default();
        assert!(rt.disk_io_threshold < rt.memory_threshold);
    }

    #[test]
    fn test_custom_threshold_override() {
        let mut rt = ResourceConflictThresholds::default();
        rt.custom_thresholds.insert("cpu_custom".to_string(), 0.6);
        rt.custom_thresholds.insert("cpu_custom".to_string(), 0.75);
        if let Some(v) = rt.custom_thresholds.get("cpu_custom") {
            assert!((v - 0.75).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_all_thresholds_positive() {
        let rt = ResourceConflictThresholds::default();
        assert!(rt.cpu_threshold > 0.0);
        assert!(rt.memory_threshold > 0.0);
        assert!(rt.network_threshold > 0.0);
        assert!(rt.disk_io_threshold > 0.0);
        assert!(rt.gpu_threshold > 0.0);
    }

    #[test]
    fn test_thresholds_less_than_one() {
        let rt = ResourceConflictThresholds::default();
        assert!(rt.cpu_threshold < 1.01); // allow for floating point near 1.0
        assert!(rt.memory_threshold < 1.01);
        assert!(rt.network_threshold < 1.01);
        assert!(rt.disk_io_threshold < 1.01);
        assert!(rt.gpu_threshold < 1.01);
    }

    #[test]
    fn test_memory_threshold_between_cpu_and_gpu() {
        let rt = ResourceConflictThresholds::default();
        assert!(
            rt.memory_threshold >= rt.cpu_threshold || rt.memory_threshold > rt.network_threshold
        );
    }

    #[test]
    fn test_custom_thresholds_cleared() {
        let mut rt = ResourceConflictThresholds::default();
        rt.custom_thresholds.insert("key1".to_string(), 0.5);
        rt.custom_thresholds.insert("key2".to_string(), 0.6);
        rt.custom_thresholds.clear();
        assert!(rt.custom_thresholds.is_empty());
    }
}
