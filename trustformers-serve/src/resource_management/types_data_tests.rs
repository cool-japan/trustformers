//! Tests for resource management types_data

use super::types::*;
use super::types_data::*;
use chrono::Utc;
use std::time::Duration;

/// Simple LCG for deterministic pseudo-random values
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() % 10000) as f32 / 10000.0
    }
}

#[test]
fn test_port_usage_type_exclusive_access() {
    assert!(PortUsageType::HttpServer.requires_exclusive_access());
    assert!(PortUsageType::HttpsServer.requires_exclusive_access());
    assert!(PortUsageType::Database.requires_exclusive_access());
}

#[test]
fn test_port_usage_type_non_exclusive() {
    assert!(!PortUsageType::TcpSocket.requires_exclusive_access());
    assert!(!PortUsageType::Custom("foo".into()).requires_exclusive_access());
}

#[test]
fn test_port_usage_type_default_port_range_http() {
    let range = PortUsageType::HttpServer.default_port_range();
    if let Some((start, end)) = range {
        assert_eq!(start, 8000);
        assert_eq!(end, 8999);
    } else {
        panic!("expected Some for HttpServer");
    }
}

#[test]
fn test_port_usage_type_default_port_range_https() {
    let range = PortUsageType::HttpsServer.default_port_range();
    if let Some((start, end)) = range {
        assert_eq!(start, 8443);
        assert_eq!(end, 8499);
    } else {
        panic!("expected Some for HttpsServer");
    }
}

#[test]
fn test_port_usage_type_default_port_range_database() {
    let range = PortUsageType::Database.default_port_range();
    if let Some((start, end)) = range {
        assert_eq!(start, 5432);
        assert_eq!(end, 5499);
    } else {
        panic!("expected Some for Database");
    }
}

#[test]
fn test_port_usage_type_default_port_range_none() {
    let range = PortUsageType::TcpSocket.default_port_range();
    assert!(range.is_none());
}

#[test]
fn test_gpu_capability_supports_cuda() {
    let cap = GpuCapability::Cuda("12.0".to_string());
    assert!(cap.supports_framework("cuda"));
    assert!(cap.supports_framework("CUDA"));
    assert!(!cap.supports_framework("opencl"));
}

#[test]
fn test_gpu_capability_supports_opencl() {
    let cap = GpuCapability::OpenCl("3.0".to_string());
    assert!(cap.supports_framework("opencl"));
    assert!(cap.supports_framework("OpenCL"));
    assert!(!cap.supports_framework("cuda"));
}

#[test]
fn test_gpu_capability_supports_vulkan() {
    let cap = GpuCapability::Vulkan("1.3".to_string());
    assert!(cap.supports_framework("vulkan"));
    assert!(!cap.supports_framework("cuda"));
}

#[test]
fn test_gpu_capability_supports_ml() {
    let cap = GpuCapability::MachineLearning(vec!["tensorflow".to_string(), "pytorch".to_string()]);
    assert!(cap.supports_framework("tensorflow"));
    assert!(cap.supports_framework("pytorch"));
    assert!(!cap.supports_framework("jax"));
}

#[test]
fn test_gpu_capability_custom() {
    let cap = GpuCapability::Custom("metal".to_string(), "2.0".to_string());
    assert!(cap.supports_framework("metal"));
    assert!(cap.supports_framework("Metal"));
    assert!(!cap.supports_framework("cuda"));
}

#[test]
fn test_alert_severity_priority() {
    assert_eq!(AlertSeverity::Info.priority(), 0);
    assert_eq!(AlertSeverity::Warning.priority(), 1);
    assert_eq!(AlertSeverity::Error.priority(), 2);
    assert_eq!(AlertSeverity::Critical.priority(), 3);
}

#[test]
fn test_alert_severity_requires_immediate_attention() {
    assert!(!AlertSeverity::Info.requires_immediate_attention());
    assert!(!AlertSeverity::Warning.requires_immediate_attention());
    assert!(AlertSeverity::Error.requires_immediate_attention());
    assert!(AlertSeverity::Critical.requires_immediate_attention());
}

#[test]
fn test_execution_status_terminal() {
    assert!(ExecutionStatus::Completed.is_terminal());
    assert!(ExecutionStatus::Failed.is_terminal());
    assert!(ExecutionStatus::Cancelled.is_terminal());
    assert!(!ExecutionStatus::Running.is_terminal());
    assert!(!ExecutionStatus::Queued.is_terminal());
}

#[test]
fn test_execution_status_active() {
    assert!(ExecutionStatus::Running.is_active());
    assert!(!ExecutionStatus::Queued.is_active());
    assert!(!ExecutionStatus::Completed.is_active());
}

#[test]
fn test_directory_permissions_default() {
    let perms = DirectoryPermissions::default();
    assert!(perms.owner_read);
    assert!(perms.owner_write);
    assert!(perms.owner_execute);
}

#[test]
fn test_port_usage_statistics_default() {
    let stats = PortUsageStatistics::default();
    assert_eq!(stats.total_allocated, 0);
    assert_eq!(stats.currently_allocated, 0);
    assert_eq!(stats.peak_usage, 0);
}

#[test]
fn test_allocation_event_type_variants() {
    let types = [
        AllocationEventType::Allocated,
        AllocationEventType::Deallocated,
        AllocationEventType::Failed,
    ];
    assert_eq!(types.len(), 3);
}

#[test]
fn test_worker_state_variants() {
    let states = [
        WorkerState::Idle,
        WorkerState::Busy,
        WorkerState::Starting,
        WorkerState::Stopping,
    ];
    assert_eq!(states.len(), 4);
}

#[test]
fn test_cleanup_type_variants() {
    let types = [
        CleanupType::DeleteDirectory,
        CleanupType::DeleteOldFiles(Duration::from_secs(3600)),
        CleanupType::EmptyDirectory,
        CleanupType::CompressFiles,
        CleanupType::Custom("custom".into()),
    ];
    assert_eq!(types.len(), 5);
}

#[test]
fn test_resource_lifecycle_stage_variants() {
    let stages = [
        ResourceLifecycleStage::Creating,
        ResourceLifecycleStage::Created,
        ResourceLifecycleStage::Active,
        ResourceLifecycleStage::Idle,
        ResourceLifecycleStage::Cleaning,
        ResourceLifecycleStage::Destroying,
        ResourceLifecycleStage::Destroyed,
    ];
    assert_eq!(stages.len(), 7);
}

#[test]
fn test_gpu_alert_statistics_default() {
    let stats = GpuAlertStatistics::default();
    assert_eq!(stats.total_alerts, 0);
    assert_eq!(stats.active_alerts, 0);
    assert!(stats.last_alert_time.is_none());
}

#[test]
fn test_resource_statistics_default() {
    let stats = ResourceStatistics::default();
    assert_eq!(stats.total_allocated, 0);
    assert_eq!(stats.active_resources, 0);
}

#[test]
fn test_execution_performance_metrics_default() {
    let metrics = ExecutionPerformanceMetrics::default();
    assert_eq!(metrics.total_executions, 0);
}

#[test]
fn test_worker_pool_default() {
    let pool = WorkerPool::default();
    let workers = pool.workers.lock();
    assert!(workers.is_empty());
}

#[test]
fn test_directory_purpose_default() {
    let purpose = DirectoryPurpose::default();
    if let DirectoryPurpose::General = purpose {
        // expected
    } else {
        panic!("expected General as default");
    }
}
