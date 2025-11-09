//! Resource Modeling Module for Performance Optimizer
//!
//! This module provides comprehensive system resource modeling functionality including
//! hardware detection, performance profiling, resource utilization tracking, thermal
//! monitoring, and topology analysis for optimal performance optimization decisions.
//!
//! ## Modular Architecture (Phase 41 Refactoring)
//!
//! The original 2,578-line monolithic file has been systematically refactored into 7 focused modules
//! for improved maintainability, comprehension, and organization:
//!
//! ### Core Modules
//!
//! - **types** (2,642 lines): Core types, configurations, and trait definitions
//! - **performance_profiler** (2,537 lines): PerformanceProfiler system with comprehensive benchmarking
//! - **temperature_monitor** (1,400+ lines): TemperatureMonitor system for thermal management
//! - **topology_analyzer** (5,911 lines): TopologyAnalyzer system for NUMA and hardware topology
//! - **utilization_tracker** (1,648 lines): ResourceUtilizationTracker system for continuous monitoring
//! - **hardware_detector** (1,453 lines): HardwareDetector system with vendor-specific optimizations
//! - **manager** (1,400+ lines): ResourceModelingManager orchestrating all components
//!
//! ## Key Features
//!
//! - **Complete System Detection**: Automatic detection and characterization of CPU,
//!   memory, I/O, network, and GPU resources
//! - **Performance Profiling**: Dynamic profiling of hardware performance characteristics
//! - **Thermal Monitoring**: Real-time temperature monitoring and thermal management
//! - **Topology Analysis**: NUMA topology detection and optimization
//! - **Resource Tracking**: Continuous resource utilization monitoring
//! - **Advanced Hardware Detection**: Deep hardware inspection with vendor-specific optimizations
//! - **Vendor Optimization**: Intel, AMD, and NVIDIA specific hardware optimizations
//! - **Real-time Analytics**: Continuous monitoring with intelligent alerting
//! - **Performance Validation**: Comprehensive system capability assessment
//!
//! ## Usage Examples
//!
//! ```rust
//! use crate::performance_optimizer::resource_modeling::{
//!     ResourceModelingManager, ResourceModelingConfig,
//!     PerformanceProfiler, TemperatureMonitor,
//!     TopologyAnalyzer, HardwareDetector,
//! };
//!
//! // Initialize resource modeling with comprehensive configuration
//! let config = ResourceModelingConfig::default()
//!     .with_detailed_detection(true)
//!     .with_profiling_enabled(true)
//!     .with_temperature_monitoring(true)
//!     .with_numa_analysis(true);
//!
//! let manager = ResourceModelingManager::new(config).await?;
//!
//! // Get current system resource model
//! let resource_model = manager.get_resource_model();
//! println!("Detected {} CPU cores", resource_model.cpu_model.core_count);
//!
//! // Profile system performance characteristics
//! let performance_results = manager.profile_performance().await?;
//! println!("CPU performance: {} instructions/sec",
//!          performance_results.cpu_profile.instructions_per_second);
//!
//! // Monitor resource utilization
//! let utilization_report = manager.monitor_utilization(Duration::from_secs(60)).await?;
//! println!("Average CPU utilization: {}%", utilization_report.cpu_utilization.average);
//!
//! // Analyze hardware topology
//! let topology_results = manager.analyze_topology().await?;
//! if let Some(numa) = topology_results.numa_topology {
//!     println!("NUMA nodes detected: {}", numa.node_count);
//! }
//! ```
//!
//! ## Backward Compatibility
//!
//! All original APIs are preserved through comprehensive re-exports. Existing code
//! will continue to work without modification while benefiting from the improved
//! modular organization and enhanced performance.
//!
//! ## Module Organization
//!
//! The internal structure is organized as follows:
//!
//! ```text
//! resource_modeling/
//! ├── types.rs                - Core types, configurations, and traits
//! ├── performance_profiler.rs - Performance profiling and benchmarking
//! ├── temperature_monitor.rs  - Thermal monitoring and management
//! ├── topology_analyzer.rs    - Hardware topology and NUMA analysis
//! ├── utilization_tracker.rs  - Resource utilization monitoring
//! ├── hardware_detector.rs    - Hardware detection and characterization
//! └── manager.rs              - Main orchestrating manager
//! ```

// =============================================================================
// MODULE DECLARATIONS
// =============================================================================

/// Core types, configurations, and trait definitions for resource modeling
pub mod types;

/// Performance profiling engine with comprehensive benchmarking capabilities
pub mod performance_profiler;

/// Temperature monitoring system for thermal management and throttling
pub mod temperature_monitor;

/// Topology analyzer for NUMA detection and hardware layout optimization
pub mod topology_analyzer;

/// Resource utilization tracker for continuous monitoring and analysis
pub mod utilization_tracker;

/// Hardware detection engine with vendor-specific optimizations
pub mod hardware_detector;

/// Main orchestrating manager that coordinates all resource modeling components
pub mod manager;

// Import and re-export SystemResourceModel
pub use crate::performance_optimizer::system_models::SystemResourceModel;

// =============================================================================
// COMPREHENSIVE RE-EXPORTS FOR BACKWARD COMPATIBILITY
// =============================================================================

// Re-export all types for complete backward compatibility
pub use types::*;

// =============================================================================
// CORE RESOURCE MODELING MANAGER
// =============================================================================

// Main orchestrating engine (from manager module)
pub use manager::{
    AnalysisPriority,
    // Analysis Quality and Priority
    AnalysisQuality,
    AnalysisResultData,
    // Analysis Infrastructure
    AnalysisTask,
    AnalysisTaskResult,
    AnalysisTaskType,
    // Workflow Management
    AnalysisWorkflow,
    ComponentCoordinator,
    // Component Health and Status
    ComponentHealth,
    ComponentPerformanceMetrics,

    ComponentStatus,
    ModelingOrchestrator,

    // Core Manager Components
    ResourceModelingConfig, // Import ResourceModelingConfig from manager.rs
    ResourceModelingManager,
    RetryPolicy,

    TaskExecutionStatus,
    TaskResourceUsage,

    WorkflowExecution,
    WorkflowExecutionStatus,
    WorkflowStep,
};

// =============================================================================
// PERFORMANCE PROFILING SYSTEM
// =============================================================================

// Re-export performance profiling components
pub use performance_profiler::{
    // Profiling Infrastructure
    BenchmarkExecutor,
    CacheAnalyzer,
    CpuProfiler,
    EnhancedCpuProfile,
    EnhancedIoProfile,
    EnhancedMemoryProfile,
    EnhancedNetworkProfile,
    GpuProfiler,

    IoProfiler,
    MemoryProfiler,
    NetworkProfiler,
    // Core Profiler Components
    PerformanceProfiler,
    PerformanceValidator,

    // Enhanced Profile Results
    ProfileResult,
    ProfileResultsProcessor,
    // Configuration Types
    ProfilingConfig,

    ProfilingPhase,
    // Profiling Session Management
    ProfilingSessionState,
    ProfilingStatus,
    SessionMetadata,
    SystemInfo,
};

// =============================================================================
// TEMPERATURE MONITORING SYSTEM
// =============================================================================

// Re-export temperature monitoring components
pub use temperature_monitor::{
    CoolingController,
    HeatDissipationAnalyzer,

    // Alerting and Reporting
    TemperatureAlerting,
    // Core Temperature Monitor
    TemperatureMonitor,
    TemperatureMonitorConfig,

    ThermalAnalyzer,

    // Calibration and Optimization
    ThermalCalibrator,
    ThermalReporting,

    // Thermal Management Systems
    ThermalSensorManager,
    ThermalStateManager,
    // Prediction and Analysis
    ThrottlingPredictor,
};

// =============================================================================
// TOPOLOGY ANALYSIS SYSTEM
// =============================================================================

// Re-export topology analysis components
pub use topology_analyzer::{
    CacheAnalysisAdvanced,
    CacheBandwidthCharacteristics,

    // Cache Hierarchy Analysis
    CacheHierarchyAnalyzer,
    CacheHierarchyNode,
    CacheLatencyCharacteristics,
    CacheLevelInfoAdvanced,
    CacheSharingPattern,

    // Cache Topology
    CacheTopologyMapping,
    MemoryRegion,
    NumaDomain,
    NumaDomainMetrics,

    NumaTopologyAdvanced,
    // NUMA Analysis
    NumaTopologyDetector,
    PrecisionLevel,
    TopologyAnalysisCache,

    TopologyAnalysisConfig,
    // Core Topology Analyzer
    TopologyAnalyzer,
    // Configuration and Validation
    ValidationLevel,
};

// =============================================================================
// RESOURCE UTILIZATION TRACKING
// =============================================================================

// Re-export utilization tracking components
pub use utilization_tracker::{
    AlertingSystem,
    // Configuration Types
    CpuMonitorConfig,
    // Resource Monitors
    CpuUtilizationMonitor,
    GpuMonitorConfig,

    GpuUtilizationMonitor,

    IoMonitorConfig,
    IoUtilizationMonitor,
    MemoryMonitorConfig,
    MemoryUtilizationMonitor,
    MonitoringState,
    NetworkMonitorConfig,
    NetworkUtilizationMonitor,
    ReportGenerator,

    // Core Utilization Tracker
    ResourceUtilizationTracker,
    TrendAnalyzer,
    UtilizationEvent,
    // Utilization Data Types
    UtilizationHistory,
    // Analysis and Management
    UtilizationHistoryManager,
    UtilizationTrackingConfig,
};

// =============================================================================
// HARDWARE DETECTION SYSTEM
// =============================================================================

// Re-export hardware detection components
pub use hardware_detector::{
    AmdCpuDetector,
    ArmCpuDetector,

    CapabilityAssessor,
    CpuDetectionCache,
    // Detection Configuration
    CpuDetectionConfig,
    // Specialized Detectors
    CpuDetector,
    // Vendor Detection Traits
    CpuVendorDetector,
    GpuDetector,
    GpuVendorDetector,
    HardwareDetectionCache,

    HardwareDetectionConfig,
    // Core Hardware Detector
    HardwareDetector,
    HardwareValidator,

    // Vendor-Specific CPU Detectors
    IntelCpuDetector,
    MemoryDetectionCache,

    MemoryDetector,
    MotherboardDetector,

    NetworkDetector,
    StorageDetector,
    ValidationRule,
    // Detection Infrastructure
    VendorOptimizationEngine,
};

// =============================================================================
// CONVENIENCE FUNCTIONS AND INITIALIZATION
// =============================================================================

/// Create a new resource modeling manager with default configuration
pub async fn create_default_resource_manager() -> anyhow::Result<ResourceModelingManager> {
    let config = ResourceModelingConfig::default();
    ResourceModelingManager::new(config).await
}

/// Create a resource modeling manager with high-performance configuration
pub async fn create_high_performance_manager() -> anyhow::Result<ResourceModelingManager> {
    let config = ResourceModelingConfig::default()
        .with_detailed_detection(true)
        .with_profiling_enabled(true)
        .with_temperature_monitoring(true)
        .with_numa_analysis(true)
        .with_profiling_samples(20)
        .with_cache_profiling_results(true);

    ResourceModelingManager::new(config).await
}

/// Create a resource modeling manager with minimal overhead configuration
pub async fn create_minimal_overhead_manager() -> anyhow::Result<ResourceModelingManager> {
    use std::time::Duration;

    let config = ResourceModelingConfig::default()
        .with_detailed_detection(false)
        .with_profiling_enabled(false)
        .with_temperature_monitoring(true)
        .with_numa_analysis(false)
        .with_update_interval(Duration::from_secs(300))
        .with_profiling_samples(3)
        .with_cache_profiling_results(false);

    ResourceModelingManager::new(config).await
}

/// Quick system resource detection without full initialization
pub async fn quick_system_detection() -> anyhow::Result<SystemResourceModel> {
    let manager = create_minimal_overhead_manager().await?;
    Ok(manager.get_resource_model())
}

/// Quick system performance profile
pub async fn quick_performance_profile() -> anyhow::Result<PerformanceProfileResults> {
    let manager = create_default_resource_manager().await?;
    manager.profile_performance().await
}

/// Quick system temperature check
pub async fn quick_temperature_check() -> anyhow::Result<TemperatureMetrics> {
    let config = TemperatureMonitorConfig::default();
    let monitor = TemperatureMonitor::new(config).await?;
    monitor.get_current_temperature().await
}

// =============================================================================
// TYPE ALIASES FOR COMMON USE CASES
// =============================================================================

/// Alias for the main resource modeling manager
pub type ResourceManager = ResourceModelingManager;

/// Alias for resource modeling configuration
pub type ResourceConfig = ResourceModelingConfig;

/// Alias for system resource model
pub type SystemModel = SystemResourceModel;

/// Alias for performance profile results
pub type PerformanceResults = PerformanceProfileResults;

/// Alias for utilization monitoring report
pub type UtilizationMonitoringReport = UtilizationReport;

/// Alias for topology analysis results
pub type TopologyResults = TopologyAnalysisResults;

/// Alias for hardware detection cache
pub type DetectionCache = HardwareDetectionCache;

// =============================================================================
// LEGACY COMPATIBILITY EXPORTS
// =============================================================================

// Legacy function aliases for smooth migration
pub use create_default_resource_manager as new_resource_manager;
pub use quick_performance_profile as profile_system_performance;
pub use quick_system_detection as detect_system_resources;
pub use quick_temperature_check as check_system_temperature;

// Legacy type aliases
pub use HardwareDetector as SystemHardwareDetector;
pub use PerformanceProfiler as SystemProfiler;
pub use ResourceModelingManager as ResourceModelingEngine;
pub use ResourceUtilizationTracker as SystemUtilizationTracker;
pub use TemperatureMonitor as ThermalMonitor;
pub use TopologyAnalyzer as SystemTopologyAnalyzer;
