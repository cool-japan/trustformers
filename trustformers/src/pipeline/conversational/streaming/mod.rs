//! Streaming response system modules for conversational AI.
//!
//! This module contains the streaming functionality split into focused submodules:
//! - `types`: Common types and data structures
//! - `coordinator`: Stream coordination and management
//! - `chunking`: Response chunking and token streaming
//! - `typing_simulation`: Natural typing simulation and conversation flow
//! - `state_management`: Stream state management and error recovery
//! - `backpressure`: Backpressure handling and flow control algorithms
//! - `quality_analyzer`: Quality analysis and streaming algorithms
//! - `pipeline`: Main streaming pipeline orchestrator and implementation
//! - `compatibility`: Backward compatibility and legacy support for existing APIs

pub mod backpressure;
pub mod chunking;
pub mod compatibility;
pub mod coordinator;
pub mod pipeline;
pub mod quality_analyzer;
pub mod state_management;
pub mod types;
pub mod typing_simulation;

// Re-export commonly used types for convenience
pub use backpressure::{
    BackpressureController, BackpressureMetrics, BackpressureStrategy, EnhancedBufferState,
    FlowAction, FlowControlStrategies, FlowState, LoadBalanceAction, OverflowAction, PressureLevel,
    ResourceMonitor, SystemHealth,
};
pub use chunking::ResponseChunker;
pub use coordinator::StreamingCoordinator;
pub use pipeline::ConversationalStreamingPipeline;
pub use quality_analyzer::{
    AdvancedQualityMetrics, ComplexityLevel, DegradationIndicators, OptimizationRecommendation,
    OptimizationType, PerceptualQuality, PerformanceBenchmarks, QualityAnalysis, QualityAnalyzer,
    QualityArea, QualityMeasurement, QualityThresholds, QualityTrends, StatisticalAnalysis,
    StreamingQuality,
};
pub use state_management::{
    ErrorRecoveryManager, ErrorSeverity, HealthStatus, LatencyMetrics, NetworkMetrics,
    OverallHealthStatus, RecoveryStrategy, ResourceUsage, StateTransition, StreamError,
    StreamErrorType, StreamPerformance, StreamState, StreamStateManager,
};
pub use types::*;
pub use typing_simulation::{
    TypingAnalysis, TypingCharacteristics, TypingEvent, TypingEventType, TypingPattern,
    TypingPatterns, TypingPersonality, TypingSimulator,
};

// Legacy compatibility re-exports for backward compatibility
// These are deprecated and will be removed in a future version
#[allow(deprecated)]
pub use compatibility::{
    LegacyStreamingConfig, LegacyStreamingResponse, LegacyStreamingState, StreamingManager,
    StreamingSession, StreamingStats,
};
