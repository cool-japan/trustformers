//! Gradient Debugging and Analysis Tools
//!
//! This module provides comprehensive gradient debugging capabilities including
//! real-time monitoring, anomaly detection, performance tracking, conflict analysis,
//! visualization, and enhanced analysis for gradient flow in neural networks.
//!
//! ## Architecture
//!
//! The gradient debugging system is organized into several focused modules:
//! - [`types`] - Core types, enums, and configuration for gradient debugging
//! - [`monitoring`] - Real-time gradient monitoring and adaptive thresholds
//! - [`anomaly_detection`] - Advanced gradient anomaly detection system
//! - [`performance_tracking`] - Performance tracking and bottleneck analysis
//! - [`conflict_analysis`] - Gradient conflict analysis between layers
//! - [`visualization`] - Gradient flow visualization data generation
//! - [`enhanced_analysis`] - Enhanced layer analysis and network-level insights
//! - [`debugger`] - Main GradientDebugger orchestrating all components

pub mod anomaly_detection;
pub mod conflict_analysis;
pub mod debugger;
pub mod enhanced_analysis;
pub mod monitoring;
pub mod performance_tracking;
pub mod types;
pub mod visualization;

// Re-export core types for backward compatibility
pub use debugger::GradientDebugger;
pub use types::*;

// Re-export component types for easy access
pub use monitoring::{
    AdaptiveThresholds, MonitoringConfig, MonitoringResults, RealTimeGradientMonitor,
};

pub use anomaly_detection::{
    AnomalyContext, AnomalySummary, AnomalyTrend, AnomalyType, BaselineGradientStats,
    GradientAnomaly, GradientAnomalyDetector,
};

pub use performance_tracking::{
    GradientPerformanceTracker, LayerPerformanceMetrics, OptimizationIssue,
    OptimizationRecommendation, OptimizationSeverity, PerformanceSnapshot, PerformanceTimer,
    PerformanceTrends, ResourceUtilization,
};

pub use conflict_analysis::{
    ConflictLevel, ConflictMitigationStrategy, ConflictReport, ConflictType, GradientConflict,
    GradientConflictAnalysis, MitigationComplexity,
};

pub use visualization::{
    CriticalGradientPath, ExplodingRegion, FlowEdge, FlowNode, GradientDeadZone,
    GradientFlowNetwork, GradientFlowVisualization, GradientLayerFlow, GradientVisualizationConfig,
    TemporalGradientFlow, VanishingRegion,
};

pub use enhanced_analysis::{
    EnhancedLayerGradientAnalysis, GradientHierarchy, LayerGradientDetails,
    LayerOptimizationSuggestion, NetworkLevelAnalysis, OptimizationPriority,
};

pub use debugger::{
    ComprehensiveGradientReport, GradientDebugStatus, GradientRecommendation, LayerGradientStatus,
    PerformanceInsights, RecommendationType,
};
pub use trustformers_core::RecommendationPriority;

// Backward compatibility alias
pub type GradientDebugReport = ComprehensiveGradientReport;
