//! Tests for the advanced_gpu_profiler module.

use super::*;
use std::time::Duration;

// ── AdvancedGpuProfilingConfig ────────────────────────────────────────────────

#[test]
fn test_gpu_profiling_config_default() {
    let cfg = AdvancedGpuProfilingConfig::default();
    assert!(cfg.enable_gpu_profiling);
    assert_eq!(cfg.device_count, 1);
    assert!(cfg.enable_memory_profiling);
    assert!(cfg.enable_kernel_profiling);
    assert!(cfg.enable_bandwidth_monitoring);
    assert_eq!(cfg.max_tracked_allocations, 10000);
    assert!((cfg.profiling_sampling_rate - 1.0).abs() < 1e-6);
    assert!(cfg.enable_fragmentation_analysis);
}

#[test]
fn test_gpu_profiling_config_clone() {
    let cfg = AdvancedGpuProfilingConfig::default();
    let cloned = cfg.clone();
    assert_eq!(cfg.device_count, cloned.device_count);
    assert_eq!(cfg.max_tracked_allocations, cloned.max_tracked_allocations);
}

// ── GpuMemoryType variants ────────────────────────────────────────────────────

#[test]
fn test_gpu_memory_type_variants() {
    let variants = [
        GpuMemoryType::Global,
        GpuMemoryType::Shared,
        GpuMemoryType::Constant,
        GpuMemoryType::Texture,
        GpuMemoryType::Local,
        GpuMemoryType::Unified,
        GpuMemoryType::Pinned,
    ];
    for v in &variants {
        let _cloned = v.clone();
        let _debug = format!("{:?}", v);
    }
}

// ── AllocationSource variants ─────────────────────────────────────────────────

#[test]
fn test_allocation_source_variants() {
    let variants = [
        AllocationSource::TensorCreation,
        AllocationSource::KernelLaunch,
        AllocationSource::IntermediateBuffer,
        AllocationSource::GradientBuffer,
        AllocationSource::WeightBuffer,
        AllocationSource::ActivationBuffer,
        AllocationSource::CacheBuffer,
    ];
    for v in &variants {
        let _cloned = v.clone();
    }
}

// ── CrossDeviceTransferType variants ─────────────────────────────────────────

#[test]
fn test_cross_device_transfer_type_variants() {
    let variants = [
        CrossDeviceTransferType::DirectMemoryAccess,
        CrossDeviceTransferType::PeerToPeer,
        CrossDeviceTransferType::HostBounced,
        CrossDeviceTransferType::NvLink,
        CrossDeviceTransferType::Infinity,
    ];
    for v in &variants {
        let _cloned = v.clone();
    }
}

// ── MemoryPressureLevel variants ──────────────────────────────────────────────

#[test]
fn test_memory_pressure_level_variants() {
    let variants = [
        MemoryPressureLevel::Low,
        MemoryPressureLevel::Medium,
        MemoryPressureLevel::High,
        MemoryPressureLevel::Critical,
    ];
    for v in &variants {
        let _cloned = v.clone();
    }
}

// ── AdvancedGpuMemoryProfiler construction ────────────────────────────────────

#[test]
fn test_profiler_new_single_device() {
    let result = AdvancedGpuMemoryProfiler::new(1);
    assert!(result.is_ok(), "Should construct profiler for 1 device");
}

#[test]
fn test_profiler_new_no_devices() {
    let result = AdvancedGpuMemoryProfiler::new(0);
    assert!(result.is_ok(), "Should construct profiler for 0 devices");
}

#[test]
fn test_profiler_new_multiple_devices() {
    let result = AdvancedGpuMemoryProfiler::new(2);
    assert!(result.is_ok(), "Should construct profiler for 2 devices");
}

// ── track_allocation ──────────────────────────────────────────────────────────

#[test]
fn test_track_allocation_basic() {
    let mut profiler = AdvancedGpuMemoryProfiler::new(1).expect("profiler");
    let context = AllocationContext {
        kernel_name: Some("matmul".to_string()),
        tensor_name: Some("weight_0".to_string()),
        layer_name: Some("linear_1".to_string()),
        allocation_source: AllocationSource::WeightBuffer,
        stack_trace: Vec::new(),
    };
    let result = profiler.track_allocation(0, 1024 * 1024, GpuMemoryType::Global, context);
    assert!(result.is_ok());
    let alloc_id = result.expect("allocation id");
    let _debug = format!("{:?}", alloc_id);
}

#[test]
fn test_track_allocation_returns_unique_ids() {
    let mut profiler = AdvancedGpuMemoryProfiler::new(1).expect("profiler");
    let ctx1 = AllocationContext {
        kernel_name: None,
        tensor_name: Some("t1".to_string()),
        layer_name: None,
        allocation_source: AllocationSource::TensorCreation,
        stack_trace: Vec::new(),
    };
    let ctx2 = AllocationContext {
        kernel_name: None,
        tensor_name: Some("t2".to_string()),
        layer_name: None,
        allocation_source: AllocationSource::TensorCreation,
        stack_trace: Vec::new(),
    };
    let id1 = profiler.track_allocation(0, 512, GpuMemoryType::Global, ctx1).expect("id1");
    let id2 = profiler.track_allocation(0, 1024, GpuMemoryType::Global, ctx2).expect("id2");
    assert_ne!(id1, id2);
}

// ── track_deallocation ────────────────────────────────────────────────────────

#[test]
fn test_track_deallocation_valid_id() {
    let mut profiler = AdvancedGpuMemoryProfiler::new(1).expect("profiler");
    let context = AllocationContext {
        kernel_name: None,
        tensor_name: Some("tensor_to_free".to_string()),
        layer_name: None,
        allocation_source: AllocationSource::IntermediateBuffer,
        stack_trace: Vec::new(),
    };
    let id = profiler.track_allocation(0, 4096, GpuMemoryType::Shared, context).expect("id");
    let result = profiler.track_deallocation(id);
    assert!(result.is_ok());
}

#[test]
fn test_track_deallocation_unknown_id() {
    let mut profiler = AdvancedGpuMemoryProfiler::new(1).expect("profiler");
    let unknown_id = Uuid::new_v4();
    // Deallocating an unknown id should be a no-op or return Ok
    let result = profiler.track_deallocation(unknown_id);
    assert!(result.is_ok());
}

// ── analyze_fragmentation ─────────────────────────────────────────────────────

#[test]
fn test_analyze_fragmentation_empty() {
    let mut profiler = AdvancedGpuMemoryProfiler::new(1).expect("profiler");
    let result = profiler.analyze_fragmentation();
    assert!(result.is_ok());
    let snapshots = result.expect("snapshots");
    // Should have snapshots for each device (1 device)
    assert_eq!(snapshots.len(), 1);
}

// ── track_cross_device_transfer ───────────────────────────────────────────────

#[test]
fn test_track_cross_device_transfer() {
    let mut profiler = AdvancedGpuMemoryProfiler::new(2).expect("profiler");
    let result = profiler.track_cross_device_transfer(
        0,
        1,
        1024 * 1024,
        CrossDeviceTransferType::PeerToPeer,
        Duration::from_millis(10),
    );
    assert!(result.is_ok());
}

// ── get_memory_analysis_report ────────────────────────────────────────────────

#[test]
fn test_memory_analysis_report_empty() {
    let profiler = AdvancedGpuMemoryProfiler::new(1).expect("profiler");
    let report = profiler.get_memory_analysis_report();
    // AllocationPatternSummary.total_allocations should be 0
    assert_eq!(report.allocation_summary.total_allocations, 0);
}

#[test]
fn test_memory_analysis_report_after_allocation() {
    let mut profiler = AdvancedGpuMemoryProfiler::new(1).expect("profiler");
    let context = AllocationContext {
        kernel_name: None,
        tensor_name: Some("t".to_string()),
        layer_name: None,
        allocation_source: AllocationSource::TensorCreation,
        stack_trace: Vec::new(),
    };
    let result = profiler.track_allocation(0, 2048, GpuMemoryType::Global, context);
    assert!(result.is_ok());
    // Report should be obtainable without panicking
    let report = profiler.get_memory_analysis_report();
    // total_allocations is derived from allocation tracking; verify report is accessible
    let _total = report.allocation_summary.total_allocations;
}

// ── KernelOptimizationSummaryReport ──────────────────────────────────────────

#[test]
fn test_kernel_optimization_summary_report() {
    let report = KernelOptimizationSummaryReport {
        total_kernels_analyzed: 10,
        optimization_opportunities_found: 3,
        high_impact_optimizations: Vec::new(),
        fusion_opportunities: 2,
        regression_alerts: 0,
        overall_optimization_score: 0.75,
        top_recommendations: vec!["Use tensor cores".to_string()],
    };
    assert_eq!(report.total_kernels_analyzed, 10);
    assert!(report.overall_optimization_score > 0.0);
}

// ── HighImpactOptimization ────────────────────────────────────────────────────

#[test]
fn test_high_impact_optimization_construction() {
    let opt = HighImpactOptimization {
        kernel_name: "conv_forward".to_string(),
        optimization_type: "TensorCore".to_string(),
        expected_speedup: 2.5,
        implementation_difficulty: "Medium".to_string(),
        description: "Switch to FP16 tensor cores".to_string(),
    };
    assert_eq!(opt.kernel_name, "conv_forward");
    assert!(opt.expected_speedup > 1.0);
}

// ── AllocationContext ─────────────────────────────────────────────────────────

#[test]
fn test_allocation_context_construction() {
    let ctx = AllocationContext {
        kernel_name: Some("softmax_kernel".to_string()),
        tensor_name: Some("attention_weights".to_string()),
        layer_name: Some("self_attention".to_string()),
        allocation_source: AllocationSource::ActivationBuffer,
        stack_trace: vec!["frame_0".to_string(), "frame_1".to_string()],
    };
    assert_eq!(ctx.kernel_name, Some("softmax_kernel".to_string()));
    assert_eq!(ctx.stack_trace.len(), 2);
}

// ── MemoryFragmentationSnapshot ───────────────────────────────────────────────

#[test]
fn test_memory_fragmentation_snapshot_fields() {
    let snapshot = MemoryFragmentationSnapshot {
        timestamp: chrono::Utc::now(),
        device_id: 0,
        total_memory: 8 * 1024 * 1024 * 1024,
        free_memory: 4 * 1024 * 1024 * 1024,
        largest_free_block: 1024 * 1024 * 1024,
        fragmentation_ratio: 0.15,
        free_block_distribution: vec![1024, 2048, 4096],
        external_fragmentation: 0.10,
        internal_fragmentation: 0.05,
    };
    assert_eq!(snapshot.device_id, 0);
    assert!(snapshot.fragmentation_ratio >= 0.0 && snapshot.fragmentation_ratio <= 1.0);
    assert!(!snapshot.free_block_distribution.is_empty());
}
