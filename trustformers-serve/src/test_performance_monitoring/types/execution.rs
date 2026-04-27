//! Execution Type Definitions

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExecutionInfo {
    pub test_id: String,
    pub test_name: String,
    pub test_suite: Option<String>,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub status: String,
    pub configuration: std::collections::HashMap<String, String>,
    pub expected_duration: Option<Duration>,
    pub resource_requirements: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]

pub struct TestFilter {
    pub filter_id: String,
    pub criteria: HashMap<String, String>,
    pub include_pattern: Option<String>,
    pub exclude_pattern: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequentialIoResult {
    pub throughput: f64,
    pub latency: Duration,
    pub block_size: usize,
    pub total_bytes: usize,
    pub operation_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomIoResult {
    pub iops: f64,
    pub latency: Duration,
    pub queue_depth: usize,
    pub total_operations: usize,
    pub operation_type: String,
}

#[cfg(test)]
mod tests {
    use super::*;
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
        fn next_f64(&mut self) -> f64 {
            (self.next() >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    #[test]
    fn test_test_execution_info_construction() {
        let now = chrono::Utc::now();
        let tei = TestExecutionInfo {
            test_id: "test-001".to_string(),
            test_name: "integration_test_auth".to_string(),
            test_suite: Some("auth_suite".to_string()),
            start_time: now,
            end_time: None,
            status: "running".to_string(),
            configuration: HashMap::new(),
            expected_duration: Some(Duration::from_secs(30)),
            resource_requirements: None,
        };
        assert_eq!(tei.test_id, "test-001");
        assert_eq!(tei.test_name, "integration_test_auth");
        assert!(tei.test_suite.is_some());
        assert!(tei.end_time.is_none());
        assert_eq!(tei.status, "running");
    }

    #[test]
    fn test_test_execution_info_completed() {
        let now = chrono::Utc::now();
        let mut config = HashMap::new();
        config.insert("env".to_string(), "staging".to_string());
        let tei = TestExecutionInfo {
            test_id: "test-002".to_string(),
            test_name: "load_test".to_string(),
            test_suite: None,
            start_time: now,
            end_time: Some(now),
            status: "passed".to_string(),
            configuration: config,
            expected_duration: Some(Duration::from_secs(120)),
            resource_requirements: None,
        };
        assert!(tei.end_time.is_some());
        assert_eq!(tei.status, "passed");
        assert_eq!(tei.configuration.len(), 1);
    }

    #[test]
    fn test_test_execution_info_no_suite() {
        let now = chrono::Utc::now();
        let tei = TestExecutionInfo {
            test_id: "test-003".to_string(),
            test_name: "unit_test_parser".to_string(),
            test_suite: None,
            start_time: now,
            end_time: None,
            status: "pending".to_string(),
            configuration: HashMap::new(),
            expected_duration: None,
            resource_requirements: None,
        };
        assert!(tei.test_suite.is_none());
        assert!(tei.expected_duration.is_none());
    }

    #[test]
    fn test_test_filter_construction() {
        let mut criteria = HashMap::new();
        criteria.insert("tag".to_string(), "smoke".to_string());
        let tf = TestFilter {
            filter_id: "filter-001".to_string(),
            criteria,
            include_pattern: Some("test_*".to_string()),
            exclude_pattern: Some("*_ignored".to_string()),
        };
        assert_eq!(tf.filter_id, "filter-001");
        assert_eq!(tf.criteria.len(), 1);
        assert!(tf.include_pattern.is_some());
        assert!(tf.exclude_pattern.is_some());
    }

    #[test]
    fn test_test_filter_no_patterns() {
        let tf = TestFilter {
            filter_id: "filter-002".to_string(),
            criteria: HashMap::new(),
            include_pattern: None,
            exclude_pattern: None,
        };
        assert!(tf.include_pattern.is_none());
        assert!(tf.exclude_pattern.is_none());
        assert!(tf.criteria.is_empty());
    }

    #[test]
    fn test_sequential_io_result_construction() {
        let sr = SequentialIoResult {
            throughput: 500.0,
            latency: Duration::from_millis(10),
            block_size: 4096,
            total_bytes: 1024 * 1024,
            operation_type: "write".to_string(),
        };
        assert!(sr.throughput > 0.0);
        assert!(sr.latency > Duration::from_secs(0));
        assert_eq!(sr.block_size, 4096);
        assert_eq!(sr.total_bytes, 1024 * 1024);
        assert_eq!(sr.operation_type, "write");
    }

    #[test]
    fn test_sequential_io_result_read() {
        let sr = SequentialIoResult {
            throughput: 800.0,
            latency: Duration::from_millis(5),
            block_size: 65536,
            total_bytes: 10 * 1024 * 1024,
            operation_type: "read".to_string(),
        };
        assert_eq!(sr.operation_type, "read");
        assert!(sr.throughput > 500.0);
        assert!(sr.block_size > 4096);
    }

    #[test]
    fn test_random_io_result_construction() {
        let rr = RandomIoResult {
            iops: 10000.0,
            latency: Duration::from_micros(100),
            queue_depth: 32,
            total_operations: 100_000,
            operation_type: "mixed".to_string(),
        };
        assert!(rr.iops > 0.0);
        assert_eq!(rr.queue_depth, 32);
        assert_eq!(rr.total_operations, 100_000);
        assert_eq!(rr.operation_type, "mixed");
    }

    #[test]
    fn test_random_io_result_write() {
        let rr = RandomIoResult {
            iops: 5000.0,
            latency: Duration::from_micros(200),
            queue_depth: 16,
            total_operations: 50_000,
            operation_type: "write".to_string(),
        };
        assert_eq!(rr.operation_type, "write");
        assert!(rr.queue_depth < 32);
    }

    #[test]
    fn test_sequential_io_throughput_consistency() {
        let mut lcg = Lcg::new(161803);
        for _ in 0..5 {
            let throughput = lcg.next_f64() * 1000.0;
            let block_size = ((lcg.next_f64() * 65536.0) as usize).max(512);
            let sr = SequentialIoResult {
                throughput,
                latency: Duration::from_millis(10),
                block_size,
                total_bytes: block_size * 100,
                operation_type: "read".to_string(),
            };
            assert!(sr.throughput >= 0.0);
            assert!(sr.block_size >= 512);
            assert_eq!(sr.total_bytes, sr.block_size * 100);
        }
    }

    #[test]
    fn test_random_io_high_iops() {
        let rr = RandomIoResult {
            iops: 500_000.0,
            latency: Duration::from_nanos(200),
            queue_depth: 256,
            total_operations: 5_000_000,
            operation_type: "read".to_string(),
        };
        assert!(rr.iops > 100_000.0);
        assert!(rr.queue_depth > 32);
        assert!(rr.latency < Duration::from_micros(1));
    }

    #[test]
    fn test_test_execution_info_with_resources() {
        let now = chrono::Utc::now();
        let mut resources = HashMap::new();
        resources.insert("cpu_cores".to_string(), "4".to_string());
        resources.insert("memory_gb".to_string(), "8".to_string());
        let tei = TestExecutionInfo {
            test_id: "test-004".to_string(),
            test_name: "perf_test".to_string(),
            test_suite: Some("performance_suite".to_string()),
            start_time: now,
            end_time: None,
            status: "running".to_string(),
            configuration: HashMap::new(),
            expected_duration: Some(Duration::from_secs(300)),
            resource_requirements: Some(resources),
        };
        if let Some(reqs) = &tei.resource_requirements {
            assert_eq!(reqs.len(), 2);
        }
    }

    #[test]
    fn test_sequential_io_zero_throughput() {
        let sr = SequentialIoResult {
            throughput: 0.0,
            latency: Duration::from_secs(0),
            block_size: 4096,
            total_bytes: 0,
            operation_type: "none".to_string(),
        };
        assert_eq!(sr.throughput, 0.0);
        assert_eq!(sr.total_bytes, 0);
    }

    #[test]
    fn test_random_io_zero_iops() {
        let rr = RandomIoResult {
            iops: 0.0,
            latency: Duration::from_secs(0),
            queue_depth: 0,
            total_operations: 0,
            operation_type: "none".to_string(),
        };
        assert_eq!(rr.iops, 0.0);
        assert_eq!(rr.total_operations, 0);
    }

    #[test]
    fn test_test_filter_with_multiple_criteria() {
        let mut criteria = HashMap::new();
        criteria.insert("category".to_string(), "integration".to_string());
        criteria.insert("priority".to_string(), "high".to_string());
        criteria.insert("owner".to_string(), "team-alpha".to_string());
        let tf = TestFilter {
            filter_id: "filter-multi".to_string(),
            criteria,
            include_pattern: None,
            exclude_pattern: None,
        };
        assert_eq!(tf.criteria.len(), 3);
        assert_eq!(
            tf.criteria.get("category"),
            Some(&"integration".to_string())
        );
    }

    #[test]
    fn test_test_execution_info_failed_status() {
        let now = chrono::Utc::now();
        let tei = TestExecutionInfo {
            test_id: "test-005".to_string(),
            test_name: "failed_test".to_string(),
            test_suite: None,
            start_time: now,
            end_time: Some(now),
            status: "failed".to_string(),
            configuration: HashMap::new(),
            expected_duration: None,
            resource_requirements: None,
        };
        assert_eq!(tei.status, "failed");
        assert!(tei.end_time.is_some());
    }

    #[test]
    fn test_sequential_io_large_block_size() {
        let sr = SequentialIoResult {
            throughput: 2000.0,
            latency: Duration::from_millis(2),
            block_size: 1024 * 1024,        // 1MB
            total_bytes: 1024 * 1024 * 100, // 100MB
            operation_type: "read".to_string(),
        };
        assert_eq!(sr.block_size, 1024 * 1024);
        assert!(sr.total_bytes > sr.block_size);
        assert!(sr.throughput > 0.0);
    }

    #[test]
    fn test_random_io_deep_queue() {
        let rr = RandomIoResult {
            iops: 50000.0,
            latency: Duration::from_micros(500),
            queue_depth: 128,
            total_operations: 5_000_000,
            operation_type: "read".to_string(),
        };
        assert!(rr.queue_depth >= 64);
        assert!(rr.iops > 10000.0);
    }

    #[test]
    fn test_test_execution_info_skipped_status() {
        let now = chrono::Utc::now();
        let tei = TestExecutionInfo {
            test_id: "test-006".to_string(),
            test_name: "skipped_test".to_string(),
            test_suite: Some("flaky_suite".to_string()),
            start_time: now,
            end_time: None,
            status: "skipped".to_string(),
            configuration: HashMap::new(),
            expected_duration: None,
            resource_requirements: None,
        };
        assert_eq!(tei.status, "skipped");
        assert!(tei.end_time.is_none());
    }

    #[test]
    fn test_sequential_io_random_values() {
        let mut lcg = Lcg::new(31415);
        for _ in 0..5 {
            let throughput = lcg.next_f64() * 1000.0;
            let ops = ["read", "write"];
            let op_idx = (lcg.next() % 2) as usize;
            let sr = SequentialIoResult {
                throughput,
                latency: Duration::from_millis(10),
                block_size: 4096,
                total_bytes: 4096 * 1000,
                operation_type: ops[op_idx].to_string(),
            };
            assert!(sr.throughput >= 0.0);
            assert!(!sr.operation_type.is_empty());
        }
    }

    #[test]
    fn test_test_filter_id_uniqueness() {
        let tf1 = TestFilter {
            filter_id: "filter-a".to_string(),
            criteria: HashMap::new(),
            include_pattern: None,
            exclude_pattern: None,
        };
        let tf2 = TestFilter {
            filter_id: "filter-b".to_string(),
            criteria: HashMap::new(),
            include_pattern: None,
            exclude_pattern: None,
        };
        assert_ne!(tf1.filter_id, tf2.filter_id);
    }
}
