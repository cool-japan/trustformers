//! Tests for performance profiler types
//!
//! Tests covering enum variants, struct construction, field access,
//! Clone/PartialEq, and profiler stub methods.

use super::*;

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(1442695040888963407u64);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next() >> 11) as f32 / (1u64 << 53) as f32
    }
}

#[test]
fn test_profiling_phase_initializing() {
    let phase = ProfilingPhase::Initializing;
    let _ = format!("{:?}", phase);
}

#[test]
fn test_profiling_phase_profiling() {
    let phase = ProfilingPhase::Profiling;
    let _ = format!("{:?}", phase);
}

#[test]
fn test_profiling_phase_processing() {
    let phase = ProfilingPhase::Processing;
    let _ = format!("{:?}", phase);
}

#[test]
fn test_profiling_phase_completed() {
    let phase = ProfilingPhase::Completed;
    let _ = format!("{:?}", phase);
}

#[test]
fn test_profiling_phase_failed() {
    let phase = ProfilingPhase::Failed("connection timeout".to_string());
    if let ProfilingPhase::Failed(msg) = &phase {
        assert!(!msg.is_empty());
        assert_eq!(msg, "connection timeout");
    } else {
        panic!("Expected Failed variant");
    }
}

#[test]
fn test_profiling_phase_validating() {
    let phase = ProfilingPhase::Validating;
    let _ = format!("{:?}", phase);
}

#[test]
fn test_profiling_phase_clone() {
    let phase = ProfilingPhase::Profiling;
    let cloned = phase.clone();
    let _ = format!("{:?}", cloned);
}

#[test]
fn test_profiling_phase_failed_clone() {
    let phase = ProfilingPhase::Failed("test error".to_string());
    let cloned = phase.clone();
    if let ProfilingPhase::Failed(msg) = cloned {
        assert_eq!(msg, "test error");
    }
}

#[test]
fn test_cpu_vendor_variants() {
    let intel = CpuVendor::Intel;
    let amd = CpuVendor::Amd;
    let arm = CpuVendor::Arm;
    let other = CpuVendor::Other("RISC-V".to_string());
    let _ = format!("{:?}", intel);
    let _ = format!("{:?}", amd);
    let _ = format!("{:?}", arm);
    if let CpuVendor::Other(name) = &other {
        assert_eq!(name, "RISC-V");
    } else {
        panic!("Expected Other variant");
    }
}

#[test]
fn test_cpu_vendor_clone() {
    let vendor = CpuVendor::Intel;
    let cloned = vendor.clone();
    let _ = format!("{:?}", cloned);
}

#[test]
fn test_cpu_vendor_other_clone() {
    let vendor = CpuVendor::Other("custom".to_string());
    let cloned = vendor.clone();
    if let CpuVendor::Other(name) = cloned {
        assert_eq!(name, "custom");
    }
}

#[test]
fn test_memory_hierarchy_analyzer_new() {
    let analyzer = MemoryHierarchyAnalyzer::new();
    let _ = format!("{:?}", analyzer);
}

#[test]
fn test_memory_hierarchy_analyzer_clone() {
    let analyzer = MemoryHierarchyAnalyzer::new();
    let cloned = analyzer.clone();
    let _ = format!("{:?}", cloned);
}

#[test]
fn test_memory_hierarchy_analyzer_analyze() {
    let analyzer = MemoryHierarchyAnalyzer::new();
    let result = analyzer.analyze_memory_hierarchy();
    assert!(result.is_ok());
    if let Ok(analysis) = result {
        assert!(analysis.hierarchy_levels.is_empty());
        assert!(analysis.bandwidth_matrix.is_empty());
        assert!(analysis.latency_matrix.is_empty());
    }
}

#[test]
fn test_cpu_vendor_detector_new() {
    let detector = CpuVendorDetector::new();
    let _ = format!("{:?}", detector);
}

#[test]
fn test_cpu_vendor_detector_detect_capabilities() {
    let detector = CpuVendorDetector::new();
    let result = detector.detect_cpu_capabilities();
    assert!(result.is_ok());
    if let Ok(info) = result {
        assert!(!info.model.is_empty());
        assert!(info.core_count > 0);
        assert!(info.thread_count > 0);
    }
}

#[test]
fn test_gpu_vendor_detector_new() {
    let detector = GpuVendorDetector::new();
    let _ = format!("{:?}", detector);
}

#[test]
fn test_gpu_vendor_detector_detect_capabilities() {
    let detector = GpuVendorDetector::new();
    let result = detector.detect_gpu_capabilities();
    assert!(result.is_ok());
    if let Ok(info) = result {
        assert!(!info.model.is_empty());
        assert!(!info.compute_capability.is_empty());
    }
}

#[test]
fn test_cpu_benchmark_suite_new() {
    let suite = CpuBenchmarkSuite::new();
    let _ = format!("{:?}", suite);
}

#[test]
fn test_cpu_benchmark_suite_execute_benchmarks() {
    let suite = CpuBenchmarkSuite::new();
    let detector = CpuVendorDetector::new();
    let cpu_info = detector.detect_cpu_capabilities();
    assert!(cpu_info.is_ok());
    if let Ok(info) = cpu_info {
        let result = suite.execute_comprehensive_benchmarks(&info);
        assert!(result.is_ok());
        if let Ok(benchmarks) = result {
            assert!(benchmarks.single_thread_score >= 0.0);
            assert!(benchmarks.multi_thread_score >= 0.0);
            assert!(benchmarks.efficiency_score >= 0.0);
        }
    }
}

#[test]
fn test_io_latency_analyzer_new() {
    let analyzer = IoLatencyAnalyzer::new();
    let _ = format!("{:?}", analyzer);
}

#[test]
fn test_io_latency_analyzer_analyze() {
    let analyzer = IoLatencyAnalyzer::new();
    let result = analyzer.analyze_comprehensive_latency();
    assert!(result.is_ok());
    if let Ok(analysis) = result {
        assert!(analysis.average_latency_ns >= 0.0);
        assert!(analysis.p50_latency_ns >= 0.0);
        assert!(analysis.p99_latency_ns >= 0.0);
    }
}

#[test]
fn test_network_latency_tester_new() {
    let tester = NetworkLatencyTester::new();
    let _ = format!("{:?}", tester);
}

#[test]
fn test_network_latency_tester_analyze() {
    let tester = NetworkLatencyTester::new();
    let result = tester.analyze_comprehensive_latency();
    assert!(result.is_ok());
    if let Ok(analysis) = result {
        assert!(analysis.min_latency_ms <= analysis.avg_latency_ms || analysis.avg_latency_ms == 0.0);
        assert!(analysis.avg_latency_ms <= analysis.max_latency_ms || analysis.max_latency_ms == 0.0);
    }
}

#[test]
fn test_lcg_generates_different_values() {
    let mut rng = Lcg::new(12345);
    let v1 = rng.next_f32();
    let v2 = rng.next_f32();
    let v3 = rng.next_f32();
    assert!((v1 - v2).abs() > f32::EPSILON || (v2 - v3).abs() > f32::EPSILON);
}

#[test]
fn test_profiling_phase_all_variants_fmt() {
    let phases = vec![
        ProfilingPhase::Initializing,
        ProfilingPhase::Profiling,
        ProfilingPhase::Processing,
        ProfilingPhase::Validating,
        ProfilingPhase::Completed,
        ProfilingPhase::Failed("err".to_string()),
    ];
    for phase in &phases {
        let debug_str = format!("{:?}", phase);
        assert!(!debug_str.is_empty());
    }
}

#[test]
fn test_cache_analysis_config_default() {
    let config = CacheAnalysisConfig::default();
    let _ = format!("{:?}", config);
}

#[tokio::test]
async fn test_cache_analyzer_new() {
    let config = CacheAnalysisConfig::default();
    let result = CacheAnalyzer::new(config).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_gpu_profiler_new() {
    let config = GpuProfilingConfig::default();
    let result = GpuProfiler::new(config).await;
    assert!(result.is_ok());
}
