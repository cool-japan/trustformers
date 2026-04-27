//! Comprehensive Types Module for Resource Modeling System
//!
//! This module contains all types extracted from the resource modeling system,
//! organized into logical categories for optimal maintainability and comprehension.
//! The resource modeling system provides comprehensive hardware detection, performance
//! profiling, thermal monitoring, and topology analysis capabilities.
//!
//! # Features
//!
//! * **Core Configuration Types**: Configuration structs for all aspects of resource modeling
//! * **Hardware Model Types**: Detailed hardware characterization and modeling types
//! * **Monitoring Types**: Real-time monitoring and tracking infrastructure
//! * **Detection Types**: Hardware detection and vendor-specific capabilities
//! * **Profiling Types**: Performance profiling and benchmarking systems
//! * **Topology Types**: System topology analysis and NUMA optimization
//! * **Utility Types**: Supporting types for resource management
//! * **Enums**: State and type enumerations for resource modeling
//! * **Error Types**: Comprehensive error handling for resource operations
//! * **Trait Definitions**: Extensible interfaces for hardware detection

// Import and re-export types from the main performance optimizer types module
pub use crate::performance_optimizer::types::{
    CacheHierarchy, CpuModel, CpuPerformanceCharacteristics, GpuDeviceModel, GpuModel,
    GpuUtilizationCharacteristics, IoModel, MemoryModel, MemoryType, NetworkInterface,
    NetworkInterfaceStatus, NetworkInterfaceType, NetworkModel, NumaTopology, StorageDevice,
    StorageDeviceType, SystemResourceModel, SystemState, TemperatureMetrics,
};

// Import ResourceModelingConfig from manager module (canonical definition)
pub use super::manager::ResourceModelingConfig;

// Module declarations
pub mod config;
pub mod detection;
pub mod enums;
pub mod error;
pub mod hardware;
pub mod monitoring;
pub mod profiling;
pub mod topology;
pub mod traits;
pub mod traits_analysis;
pub mod traits_profiling;
pub mod utility;

// Re-export all public types from submodules
pub use config::*;
pub use detection::*;
pub use enums::*;
pub use error::*;
pub use hardware::*;
pub use monitoring::*;
pub use profiling::*;
pub use topology::*;
pub use traits::*;
// traits_analysis and traits_profiling are re-exported via traits.rs
pub use utility::*;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::Duration;

    // ── ThermalState ──────────────────────────────────────────────────────

    #[test]
    fn test_thermal_state_default_is_normal() {
        let state = ThermalState::default();
        assert_eq!(state, ThermalState::Normal);
    }

    #[test]
    fn test_thermal_state_all_variants_debug() {
        assert!(format!("{:?}", ThermalState::Normal).contains("Normal"));
        assert!(format!("{:?}", ThermalState::Warning).contains("Warning"));
        assert!(format!("{:?}", ThermalState::Critical).contains("Critical"));
        assert!(format!("{:?}", ThermalState::Emergency).contains("Emergency"));
    }

    #[test]
    fn test_thermal_state_equality() {
        assert_eq!(ThermalState::Normal, ThermalState::Normal);
        assert_ne!(ThermalState::Normal, ThermalState::Warning);
        assert_ne!(ThermalState::Critical, ThermalState::Emergency);
    }

    #[test]
    fn test_thermal_state_copy() {
        let s = ThermalState::Warning;
        let t = s; // Copy semantics
        assert_eq!(s, t);
    }

    // ── HealthStatus ──────────────────────────────────────────────────────

    #[test]
    fn test_health_status_all_variants_debug() {
        assert!(format!("{:?}", HealthStatus::Healthy).contains("Healthy"));
        assert!(format!("{:?}", HealthStatus::Warning).contains("Warning"));
        assert!(format!("{:?}", HealthStatus::Critical).contains("Critical"));
        assert!(format!("{:?}", HealthStatus::Failed).contains("Failed"));
        assert!(format!("{:?}", HealthStatus::Unknown).contains("Unknown"));
    }

    #[test]
    fn test_health_status_equality() {
        assert_eq!(HealthStatus::Healthy, HealthStatus::Healthy);
        assert_ne!(HealthStatus::Healthy, HealthStatus::Failed);
        assert_ne!(HealthStatus::Critical, HealthStatus::Unknown);
    }

    // ── Severity ──────────────────────────────────────────────────────────

    #[test]
    fn test_severity_all_variants_debug() {
        let variants = [
            Severity::Info,
            Severity::Low,
            Severity::Medium,
            Severity::High,
            Severity::Critical,
        ];
        let expected = ["Info", "Low", "Medium", "High", "Critical"];
        for (v, name) in variants.iter().zip(expected.iter()) {
            assert!(format!("{:?}", v).contains(name));
        }
    }

    #[test]
    fn test_severity_equality() {
        assert_eq!(Severity::Info, Severity::Info);
        assert_ne!(Severity::Low, Severity::High);
    }

    // ── OptimizationTarget ────────────────────────────────────────────────

    #[test]
    fn test_optimization_target_throughput_debug() {
        let t = OptimizationTarget::Throughput;
        assert!(format!("{:?}", t).contains("Throughput"));
    }

    #[test]
    fn test_optimization_target_latency_debug() {
        let t = OptimizationTarget::Latency;
        assert!(format!("{:?}", t).contains("Latency"));
    }

    #[test]
    fn test_optimization_target_custom_carries_name() {
        let t = OptimizationTarget::Custom("balance_cost".to_string());
        let dbg = format!("{:?}", t);
        assert!(dbg.contains("Custom"));
    }

    #[test]
    fn test_optimization_target_resource_efficiency_debug() {
        let t = OptimizationTarget::ResourceEfficiency;
        assert!(format!("{:?}", t).contains("ResourceEfficiency"));
    }

    // ── DetectionAlgorithmType ────────────────────────────────────────────

    #[test]
    fn test_detection_algorithm_statistical_debug() {
        let d = DetectionAlgorithmType::Statistical;
        assert!(format!("{:?}", d).contains("Statistical"));
    }

    #[test]
    fn test_detection_algorithm_machine_learning_debug() {
        let d = DetectionAlgorithmType::MachineLearning;
        assert!(format!("{:?}", d).contains("MachineLearning"));
    }

    #[test]
    fn test_detection_algorithm_custom_carries_name() {
        let d = DetectionAlgorithmType::Custom("knn_detector".to_string());
        assert!(format!("{:?}", d).contains("Custom"));
    }

    // ── ProfilingMethodology ──────────────────────────────────────────────

    #[test]
    fn test_profiling_methodology_all_variants() {
        let variants = [
            ProfilingMethodology::Microbenchmark,
            ProfilingMethodology::Application,
            ProfilingMethodology::SyntheticWorkload,
            ProfilingMethodology::RealWorkload,
            ProfilingMethodology::Hybrid,
        ];
        for v in &variants {
            assert!(!format!("{:?}", v).is_empty());
        }
    }

    // ── ProfilingConfig ───────────────────────────────────────────────────

    #[test]
    fn test_profiling_config_fields() {
        let cfg = ProfilingConfig {
            cpu_benchmark_iterations: 1000,
            enable_gpu_profiling: false,
            cache_results: true,
            profiling_timeout: Duration::from_secs(30),
        };
        assert_eq!(cfg.cpu_benchmark_iterations, 1000);
        assert!(!cfg.enable_gpu_profiling);
        assert!(cfg.cache_results);
        assert_eq!(cfg.profiling_timeout, Duration::from_secs(30));
    }

    // ── TemperatureThresholds ─────────────────────────────────────────────

    #[test]
    fn test_temperature_thresholds_ordering() {
        let thresholds = TemperatureThresholds {
            warning_temperature: 70.0_f32,
            critical_temperature: 85.0_f32,
            shutdown_temperature: 95.0_f32,
        };
        assert!(thresholds.warning_temperature < thresholds.critical_temperature);
        assert!(thresholds.critical_temperature < thresholds.shutdown_temperature);
    }

    #[test]
    fn test_temperature_thresholds_fields() {
        let t = TemperatureThresholds {
            warning_temperature: 75.0,
            critical_temperature: 90.0,
            shutdown_temperature: 100.0,
        };
        assert!((t.warning_temperature - 75.0).abs() < 1e-4);
    }

    // ── UtilizationTrackingConfig ─────────────────────────────────────────

    #[test]
    fn test_utilization_tracking_config_fields() {
        let cfg = UtilizationTrackingConfig {
            sample_interval: Duration::from_millis(500),
            history_size: 200,
            detailed_tracking: true,
        };
        assert_eq!(cfg.history_size, 200);
        assert!(cfg.detailed_tracking);
        assert_eq!(cfg.sample_interval, Duration::from_millis(500));
    }

    // ── HardwareDetectionConfig ───────────────────────────────────────────

    #[test]
    fn test_hardware_detection_config_fields() {
        let cfg = HardwareDetectionConfig {
            enable_intel_detection: true,
            enable_amd_detection: true,
            enable_nvidia_detection: false,
            cache_detection_results: true,
            detection_timeout: Duration::from_secs(10),
        };
        assert!(cfg.enable_intel_detection);
        assert!(!cfg.enable_nvidia_detection);
        assert!(cfg.cache_detection_results);
    }

    // ── SystemCapabilities ────────────────────────────────────────────────

    #[test]
    fn test_system_capabilities_defaults() {
        let caps = SystemCapabilities {
            virtualization_support: false,
            hardware_acceleration: vec!["AVX2".to_string(), "SSE4.2".to_string()],
            security_features: vec!["SEV".to_string()],
            power_management: vec!["DVFS".to_string()],
            custom_capabilities: HashMap::new(),
        };
        assert!(!caps.virtualization_support);
        assert_eq!(caps.hardware_acceleration.len(), 2);
        assert_eq!(caps.security_features.len(), 1);
    }

    #[test]
    fn test_system_capabilities_custom_map() {
        let mut custom = HashMap::new();
        custom.insert("hugepages".to_string(), true);
        custom.insert("io_uring".to_string(), true);
        custom.insert("ebpf".to_string(), false);
        let caps = SystemCapabilities {
            virtualization_support: true,
            hardware_acceleration: vec![],
            security_features: vec![],
            power_management: vec![],
            custom_capabilities: custom,
        };
        assert_eq!(caps.custom_capabilities.len(), 3);
        assert_eq!(caps.custom_capabilities.get("hugepages"), Some(&true));
        assert_eq!(caps.custom_capabilities.get("ebpf"), Some(&false));
    }

    // ── CpuProfile ────────────────────────────────────────────────────────

    #[test]
    fn test_cpu_profile_default_values() {
        let profile = CpuProfile::default();
        assert_eq!(profile.instructions_per_second, 0.0);
        assert_eq!(profile.branch_prediction_accuracy, 0.0);
        assert_eq!(profile.floating_point_performance, 0.0);
        assert_eq!(profile.context_switch_overhead, Duration::from_nanos(0));
    }

    // ── OptimizationComplexity ────────────────────────────────────────────

    #[test]
    fn test_optimization_complexity_all_variants() {
        assert!(format!("{:?}", OptimizationComplexity::Low).contains("Low"));
        assert!(format!("{:?}", OptimizationComplexity::Medium).contains("Medium"));
        assert!(format!("{:?}", OptimizationComplexity::High).contains("High"));
        assert!(format!("{:?}", OptimizationComplexity::Critical).contains("Critical"));
    }

    // ── CacheAnalysis ─────────────────────────────────────────────────────

    #[test]
    fn test_cache_analysis_default() {
        let ca = CacheAnalysis::default();
        assert!(ca.cache_levels.is_empty());
        assert_eq!(ca.total_cache_size_kb, 0);
    }

    // ── MemoryTopology ────────────────────────────────────────────────────

    #[test]
    fn test_memory_topology_default() {
        let mt = MemoryTopology::default();
        assert_eq!(mt.memory_channels, 0);
        assert!(mt.dimm_configuration.is_empty());
        assert!(!mt.interleaving_enabled);
        assert!(!mt.ecc_enabled);
    }

    // ── PerformanceProfiler ───────────────────────────────────────────────

    #[test]
    fn test_performance_profiler_new() {
        let cfg = ProfilingConfig {
            cpu_benchmark_iterations: 500,
            enable_gpu_profiling: false,
            cache_results: false,
            profiling_timeout: Duration::from_secs(5),
        };
        let profiler = PerformanceProfiler::new(cfg.clone());
        assert_eq!(profiler.config.cpu_benchmark_iterations, 500);
        assert!(!profiler.config.enable_gpu_profiling);
    }

    #[test]
    fn test_performance_profiler_starts_with_empty_profiles() {
        let cfg = ProfilingConfig {
            cpu_benchmark_iterations: 100,
            enable_gpu_profiling: false,
            cache_results: true,
            profiling_timeout: Duration::from_secs(2),
        };
        let profiler = PerformanceProfiler::new(cfg);
        assert!(profiler.cpu_profiles.lock().is_empty());
        assert!(profiler.memory_profiles.lock().is_empty());
        assert!(profiler.io_profiles.lock().is_empty());
    }
}
