//! # TrustformeRS Debug
//!
//! Advanced debugging tools for TrustformeRS models including tensor inspection,
//! gradient debugging, and model diagnostics.

// Allow ambiguous glob re-exports - documented below with clear guidance on which version to use
#![allow(ambiguous_glob_reexports)]
// Allow large error types in Result (TrustformersError is large by design)
#![allow(clippy::result_large_err)]
// Allow large enum variants (debug reports contain comprehensive data)
#![allow(clippy::large_enum_variant)]
// Allow common patterns in debugging/profiling code
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::excessive_nesting)]

// New visualization and analysis modules
pub mod activation_visualizer;
pub mod attention_visualizer;
pub mod graph_visualizer;
pub mod mlflow_integration;
pub mod netron_export;
pub mod performance_tuning;
pub mod stability_checker;
pub mod tensorboard_integration;
pub mod unified_debug_session;
pub mod visualization_plugins;
pub mod weight_analyzer;

pub mod advanced_gpu_profiler;
pub mod advanced_ml_debugging;
pub mod ai_code_analyzer;
pub mod anomaly_detector;
pub mod architecture_analysis;
pub mod auto_debugger;
pub mod behavior_analysis;
pub mod cicd_integration;
pub mod collaboration;
pub mod computation_graph;
pub mod dashboard;
pub mod data_export;
pub mod differential_debugging;
pub mod distributed_debugger;
pub mod distributed_profiling;
pub mod environmental_monitor;
pub mod error_recovery;
pub mod flame_graph_profiler;
pub mod gradient_debugger;
pub mod health_checker;
pub mod hooks;
pub mod ide_integration;
pub mod interactive_debugger;
pub mod interpretability_tools;
pub mod kernel_optimizer;
pub mod large_model_viz;
pub mod llm_debugging;
pub mod memory_profiler;
pub mod model_diagnostics;
pub mod model_diagnostics_main;
pub use model_diagnostics_main::{ModelDiagnostics, ModelDiagnosticsReport};

// Import specific types from model_diagnostics to avoid conflicts
pub use model_diagnostics::{
    ActivationHeatmap,
    ActiveAlert,
    // Advanced analytics
    AdvancedAnalytics,
    AlertConfig,
    // Alert system
    AlertManager,
    AlertSeverity,
    AlertStatistics,

    AlertStatus,
    AlertThresholds,
    AnalyticsConfig,
    AnalyticsReport,
    AnomalyDetectionResults,
    ArchitecturalAnalysis,
    AttentionVisualization,
    AutoDebugConfig,
    // Auto-debugging system
    AutoDebugger,
    ConvergenceStatus,
    DebuggingRecommendation,
    DebuggingReport,
    HiddenStateAnalysis,

    IdentifiedIssue,
    IssueCategory,
    IssueSeverity,

    LayerActivationStats,
    LayerAnalysis,
    LayerAnalysisConfig,

    // Layer analysis (prefixed to avoid conflicts)
    LayerAnalyzer,
    ModelArchitectureInfo,
    ModelDiagnosticAlert,
    // Core types that don't conflict
    ModelPerformanceMetrics,
    OverfittingIndicator,
    // Performance analysis (prefixed to avoid conflicts)
    PerformanceAnalyzer,
    PerformanceAnomaly,

    PerformanceSummary,
    PlateauInfo,
    StatisticalAnalysis,
    TrainingDynamics,
    // Training analysis (prefixed to avoid conflicts)
    TrainingDynamicsAnalyzer,

    TrainingStability,
    UnderfittingIndicator,
    WeightDistribution,
};
pub mod neural_network_debugging;
pub mod profiler;
pub mod quantum_debugging;
pub mod realtime_dashboard;
pub mod regression_detector;
pub mod report_generation;
pub mod simulation_tools;
pub mod streaming_debugger;
pub mod team_dashboard;
pub mod tensor_inspector;
pub mod training_dynamics;
pub mod utilities;
pub mod visualization;
#[cfg(feature = "wasm")]
pub mod wasm_interface;

// GPU profiling imports (specific to avoid conflicts)
pub use advanced_gpu_profiler::{
    AdvancedGpuMemoryProfiler, AdvancedGpuProfilingConfig, CrossDeviceTransfer,
    GpuMemoryAllocation, GpuMemoryType, HighImpactOptimization, KernelOptimizationSummaryReport,
    MemoryAnalysisReport, MemoryFragmentationSnapshot,
};

// Kernel optimization imports (specific)
pub use kernel_optimizer::{
    KernelOptimizationAnalyzer, KernelOptimizationConfig, KernelOptimizationReport,
    KernelProfileData,
};

// ============================================================================
// New Visualization and Analysis Tools (TODO.md implementations)
// ============================================================================

// TensorBoard Integration
pub use tensorboard_integration::{
    create_graph_node, tensor_to_histogram_values, GraphDef, GraphNode as TensorBoardGraphNode,
    HistogramEvent, ScalarEvent, TensorBoardWriter, TextEvent,
};

// Netron/ONNX Export
pub use netron_export::{
    AttributeValue, ExportFormat, GraphNode as NetronGraphNode, ModelGraph, ModelMetadata,
    NetronExporter, NetronModel, TensorData, TensorInfo,
};

// Activation Visualizer
pub use activation_visualizer::{
    ActivationConfig, ActivationData, ActivationHeatmap as ActivationVisualizerHeatmap,
    ActivationHistogram, ActivationStatistics, ActivationVisualizer,
};

// Attention Visualizer
pub use attention_visualizer::{
    AttentionAnalysis, AttentionFlow, AttentionHeatmap as AttentionVisualizerHeatmap,
    AttentionType, AttentionVisualizer, AttentionVisualizerConfig, AttentionWeights, ColorScheme,
};

// Stability Checker
pub use stability_checker::{
    IssueKind, StabilityChecker, StabilityConfig, StabilityIssue, StabilitySummary,
};

// Graph Visualizer
pub use graph_visualizer::{
    ComputationGraph, GraphColorScheme, GraphEdge, GraphNode as GraphVisualizerNode,
    GraphStatistics, GraphVisualizer, GraphVisualizerConfig, LayoutDirection,
};

// Unified Debug Session Manager
pub use unified_debug_session::{SessionSummary, UnifiedDebugSession, UnifiedDebugSessionConfig};

// Weight Analyzer
pub use weight_analyzer::{
    InitializationScheme, WeightAnalysis, WeightAnalyzer, WeightAnalyzerConfig, WeightHistogram,
    WeightStatistics,
};

// MLflow Integration
pub use mlflow_integration::{
    ArtifactType, MLflowClient, MLflowConfig, MLflowDebugSession, MetricPoint, RunInfo, RunStatus,
};

// Visualization Plugin System
pub use visualization_plugins::{
    OutputFormat as PluginOutputFormat, PluginConfig, PluginManager, PluginMetadata, PluginResult,
    VisualizationData, VisualizationPlugin,
};

// Performance Tuning
pub use performance_tuning::{
    Difficulty, HardwareType, ImpactEstimate, PerformanceSnapshot,
    PerformanceSummary as TuningPerformanceSummary, PerformanceTuner, Priority, Recommendation,
    RecommendationCategory, TunerConfig, TuningReport,
};

// ============================================================================
// Module Re-exports
// ============================================================================
//
// ⚠️  TYPE NAME CONFLICTS (Documented for clarity):
// The following types are defined in multiple modules. The LAST import wins in Rust.
// If you need a specific version, import directly from the module:
//
// - `LRScheduleType`: defined in `training_dynamics` (PRIMARY) and `advanced_ml_debugging`
//   → Use `training_dynamics::LRScheduleType` for training schedules
//   → Use `advanced_ml_debugging::LRScheduleType` for ML debugging contexts
//
// - `RiskLevel`: defined in `llm_debugging` (PRIMARY) and `advanced_ml_debugging`
//   → Use `llm_debugging::RiskLevel` for LLM safety analysis
//   → Use `advanced_ml_debugging::RiskLevel` for general ML risk assessment
//
// - `InteractionType`: defined in `simulation_tools` (PRIMARY) and `advanced_ml_debugging`
//   → Use `simulation_tools::InteractionType` for simulation interactions
//   → Use `advanced_ml_debugging::InteractionType` for ML component interactions
//
// - `BottleneckType`: defined in `profiler` (PRIMARY) and `advanced_ml_debugging`
//   → Use `profiler::BottleneckType` for performance bottlenecks
//   → Use `advanced_ml_debugging::BottleneckType` for ML-specific bottlenecks
//
// - `FeatureSensitivityAnalysis`: defined in `simulation_tools` (PRIMARY) and `advanced_ml_debugging`
//   → Use `simulation_tools::FeatureSensitivityAnalysis` for simulation feature analysis
//   → Use `advanced_ml_debugging::FeatureSensitivityAnalysis` for ML feature analysis
//
// - `RobustnessAssessment`: defined in `simulation_tools` (PRIMARY) and `advanced_ml_debugging`
//   → Use `simulation_tools::RobustnessAssessment` for simulation robustness
//   → Use `advanced_ml_debugging::RobustnessAssessment` for ML robustness
//
// - `PatternType`: defined in `memory_profiler` (PRIMARY) and `ai_code_analyzer`
//   → Use `memory_profiler::PatternType` for memory allocation patterns
//   → Use `ai_code_analyzer::PatternType` for code patterns
//
// - `IssueType`: defined in `auto_debugger` (PRIMARY) and `ai_code_analyzer`
//   → Use `auto_debugger::IssueType` for debugging issues
//   → Use `ai_code_analyzer::IssueType` for code analysis issues
//
// ============================================================================

// Primary exports (order determines which type wins for ambiguous names)
// Note: New visualization modules are explicitly imported above to avoid conflicts
pub use advanced_ml_debugging::*;
pub use ai_code_analyzer::*;
pub use anomaly_detector::*;
pub use architecture_analysis::*;
pub use auto_debugger::*;
pub use behavior_analysis::*;
pub use cicd_integration::*;
pub use collaboration::*;
pub use computation_graph::*;
pub use dashboard::*;
pub use data_export::*;
pub use differential_debugging::*;
pub use distributed_debugger::*;
pub use distributed_profiling::*;
pub use environmental_monitor::*;
pub use error_recovery::*;
pub use flame_graph_profiler::*;
pub use gradient_debugger::*;
pub use health_checker::*;
pub use hooks::*;
pub use ide_integration::*;
pub use interactive_debugger::*;
pub use large_model_viz::*;
pub use llm_debugging::*;
pub use memory_profiler::*;
pub use model_diagnostics::*;
pub use neural_network_debugging::*;
pub use profiler::*;
pub use quantum_debugging::*;
pub use realtime_dashboard::{AlertSeverity as DashboardAlertSeverity, *};
pub use regression_detector::*;
pub use report_generation::*;
pub use simulation_tools::*;
pub use streaming_debugger::*;
pub use team_dashboard::*;
pub use tensor_inspector::*;
pub use training_dynamics::*; // LRScheduleType from here is PRIMARY
pub use utilities::*;
pub use visualization::*;
#[cfg(feature = "wasm")]
pub use wasm_interface::*;

use scirs2_core::ndarray::ArrayD; // SciRS2 Integration Policy

// ============================================================================
// NEW MODULAR ARCHITECTURE
// ============================================================================

/// Core debugging session and configuration management
pub mod core;

/// Simplified debugging interface with one-line functions
pub mod interface;

/// Guided debugging system with step-by-step workflows
pub mod guided;

/// Interactive tutorial and learning system
pub mod tutorial;

/// Context-aware help system
pub mod help;

/// Performance optimization system for production debugging
pub mod performance;

// Re-export all public items from modules for backward compatibility
pub use core::*;
pub use guided::*;
pub use help::*;
pub use interface::*;
pub use performance::*;
pub use tutorial::*;
