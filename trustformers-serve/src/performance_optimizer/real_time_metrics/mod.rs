//! Real-Time Metrics Module
//!
//! This module provides comprehensive real-time monitoring and metrics collection
//! functionality for the TrustformeRS performance optimization system. It is organized
//! into focused submodules for optimal maintainability and comprehension.
//!
//! ## Module Structure
//!
//! - **types**: Comprehensive type definitions for all real-time metrics functionality
//! - **collector**: Core real-time metrics collection and streaming functionality
//! - **monitor**: Parallel performance monitoring with real-time analysis
//! - **aggregator**: Real-time data aggregation and statistical processing
//! - **threshold**: Comprehensive threshold monitoring and alerting system
//! - **optimization**: Live optimization engine with intelligent recommendation generation
//! - **buffer**: High-performance circular buffers and data storage systems
//! - **analytics**: Advanced analytics engine with statistical analysis, trend detection, and forecasting
//! - **notifications**: Multi-channel notification system for alert processing and delivery
//!
//! ## Design Philosophy
//!
//! This module follows the systematic refactoring approach (Phase 39) of breaking down
//! large monolithic files into focused, well-organized modules. The types module contains
//! 147+ carefully organized types covering all aspects of real-time metrics collection,
//! monitoring, alerting, optimization, and quality control.
//!
//! ## Usage
//!
//! ```rust
//! use crate::performance_optimizer::real_time_metrics::types::{
//!     MetricsCollectionConfig,
//!     RealTimeMetricsError,
//!     SeverityLevel,
//!     StatisticalProcessor,
//! };
//! ```

pub mod aggregator;
pub mod analytics;
pub mod buffer;
pub mod collector;
pub mod monitor;
pub mod notifications;
pub mod optimization;
pub mod threshold; // Re-enabled: Implemented from backup file
pub mod types;

// Re-export commonly used types for convenience
pub use types::{
    // Enums
    ActionType,
    // Core Configuration Types
    AggregationConfig,
    // Metrics and Analysis Types
    AggregationResult,
    // Data Structure Types
    AggregationWindow,
    // Monitoring Types
    AlertEvent,
    // Additional Support Types
    AlgorithmStatistics,
    BufferStatistics,
    CheckerStatistics,
    CircularBuffer,
    ConfidenceIntervals,
    DistributionAnalysis,
    EnforcementLevel,
    // Error Types
    ErrorHandlingPolicy,
    // Alerting Types
    EvaluationStatistics,

    ImpactAlert,
    ImpactAnalysis,
    ImpactAssessment,
    ImpactMonitorConfig,
    InsightType,
    // Trait Definitions
    LiveOptimizationAlgorithm,
    MetricsCollectionConfig,
    MonitorConfiguration,
    MonitorThread,
    MonitoringEvent,
    MonitoringEventType,
    MonitoringScope,
    ObjectiveType,
    OptimizationContext,
    OptimizationDirection,
    OptimizationEngineConfig,
    OptimizationObjective,
    OptimizationRecommendation,
    OverheadMeasurement,
    PerformanceBaseline,
    PerformanceInsight,
    PipelineConfig,
    PipelineInput,
    PipelineOutput,
    PipelineStage,
    PipelineStatistics,
    ProcessingContext,
    ProcessingError,
    ProcessingResults,
    ProcessorStatistics,
    QualityCheckResult,
    QualityChecker,
    QualityControlConfig,
    QualityIssue,
    QualityIssueType,
    QualityMetrics,

    QualityRequirements,
    QualityStandards,
    QualityViolation,
    RateAdjustment,
    RateControllerStats,
    RealTimeMetricsError,

    RiskFactor,
    SampleRateAlgorithm,
    SampleRateConfig,
    SeverityLevel,
    StatisticalProcessor,
    StatisticalResult,
    ThreadStatistics,
    ThresholdConfig,

    ThresholdDirection,

    ThresholdEvaluation,
    ThresholdEvaluator,

    ThresholdMonitoringState,
    TimestampedMetrics,
    TrendAnalysis,
    VariabilityBounds,

    WindowStatistics,
};

// Re-export collector components for convenience
pub use collector::{
    // Trait Definitions
    CollectionErrorHandler,
    // Event Types
    CollectionEvent,

    // Collection Infrastructure
    CollectionStatistics,
    // Configuration Types
    DeliveryConfig,
    DeliveryGuarantee,

    MetricsPublisher,

    PerformanceImpactMonitor,

    PublishErrorHandler,
    PublisherType,
    // Core Collector Components
    RealTimeMetricsCollector,
    SampleRateController,
};

// Re-export aggregator components for convenience
pub use aggregator::RealTimeDataAggregator;

// Note: Threshold module components are not re-exported
// The threshold module currently only has stub implementations (monitor.rs, alerts.rs, etc.)
// Re-enabling these exports without implementations would cause 1,700+ compilation errors
// See threshold.rs.bak2 for the original full implementation that needs restoration
// pub use threshold::*;

// Re-export optimization components for convenience
pub use optimization::{
    AdaptiveLearner,

    AdaptiveLearnerConfig,

    // Enum Types
    AlgorithmSelectionStrategy,

    // Analysis Types
    AnalysisResult,
    BatchingOptimizationAlgorithm,
    BottleneckAnalysisAlgorithm,
    ConfidenceScorer,
    ConfidenceScorerConfig,
    ConfidenceScorerStats,
    ConfidenceScoringAlgorithm,
    ConsensusConfidenceAlgorithm,

    GenerationAlgorithmMetrics,

    // Recommendation Generation Algorithms
    HeuristicRecommendationAlgorithm,
    // Confidence Scoring Algorithms
    HistoricalConfidenceAlgorithm,
    IOOptimizationAlgorithm,
    ImpactAssessor,
    ImpactAssessorConfig,
    // Prediction and ML Types
    ImpactPredictionModel,
    // LearningModel, // NOTE: Internal type not exported from optimization module
    // Trait Definitions
    LiveOptimizationAlgorithm as OptimizationAlgorithmTrait,
    // Core Optimization Components
    LiveOptimizationEngine,
    MLBasedRecommendationAlgorithm,

    MemoryOptimizationAlgorithm,
    // ModelPerformanceMetrics, // NOTE: Internal type not exported from optimization module
    NetworkOptimizationAlgorithm,
    OptimizationEngineStatistics,

    // Statistics Types
    OptimizationEngineStats,
    // Optimization Algorithms
    ParallelismOptimizationAlgorithm,
    PatternBasedRecommendationAlgorithm,
    // Real-time Analysis Algorithms
    PerformanceAnalysisAlgorithm,
    PerformanceTuningAlgorithm,
    PredictiveAnalysisAlgorithm,

    RealTimeAnalysisAlgorithm,

    RealTimeAnalysisConfig,
    RealTimeAnalysisStats,
    RealTimeAnalyzer,
    RecommendationGenerationAlgorithm,
    RecommendationGenerator,
    // Configuration Types
    RecommendationGeneratorConfig,
    RecommendationGeneratorStats,
    ResourceOptimizationAlgorithm,
    RiskBasedConfidenceAlgorithm,
    StatisticalConfidenceAlgorithm,
    // StrategyPerformance, // NOTE: Internal type not exported from optimization module
    StrategySelector,
    StrategySelectorConfig,
    ThreadPoolOptimizationAlgorithm,
    // TrainingExample, // NOTE: Internal type not exported from optimization module
};

// Re-export analytics components for convenience
pub use analytics::{
    AdvancedStatistics,
    // Configuration Types
    AnalyticsConfig,
    // Core Analytics Components
    AnalyticsEngine,
    // Statistics Types
    AnalyticsEngineStats,
    // Result Types
    AnalyticsResult,
    AnomalousPeriod,
    AnomalyAnalysisResult,
    // Anomaly Detection Types
    AnomalyDetection,
    AnomalyDetector,
    AnomalyPattern,
    AnomalySeverity,
    AnomalyType,
    AvailabilityAnalysis,
    AvailabilityTrend,
    BaselineModelPerformance,

    // Statistical Analysis Types
    BasicStatistics,
    BottleneckAnalysis,
    BottleneckType,
    CausalRelationship,
    ChangeDirection,
    ChangePoint,
    ChangeType,

    ConditionalDependency,
    CorrelationAnalysisResult,
    CorrelationAnalyzer,
    CorrelationDirection,
    CorrelationMatrix,
    // Correlation Analysis Types
    CorrelationMeasure,
    CorrelationPattern,

    CorrelationStrength,
    CyclicalPattern,
    DataQualityIssue,
    DependencyAnalysis,
    DescriptiveStatistics,
    // Pattern Analysis Types
    DetectedPattern,
    DistributionAnalysisResult,
    DistributionAnalyzer,
    DistributionCharacteristics,
    DistributionComparison,

    // Distribution Analysis Types
    DistributionFit,
    DowntimeAnalysis,
    DowntimeIncident,
    DowntimePattern,
    EfficiencyAnalysis,
    EfficiencyComponents,
    EfficiencyTrend,
    EnsembleForecast,
    ErrorPattern,
    ErrorRateAnalysis,
    ErrorTypeAnalysis,
    ForecastAccuracyMetrics,
    ForecastPoint,
    ForecastingEngine,
    // Forecasting Types
    ForecastingModel,
    ForecastingModelType,
    ForecastingResult,
    GoodnessOfFitStatistics,

    HistogramAnalysis,
    HistogramData,
    HistogramPeak,
    ImplementationEffort,

    LatencyAnalysis,
    LatencyDistribution,
    LatencyStatistics,
    LatencyTrend,
    LeadLagRelationship,
    MitigationStrategy,
    ModelPerformanceMetrics,

    NormalityAssessment,
    NormalityTestResult,
    OpportunityType,
    OptimizationOpportunity,
    OutlierAnalysis,
    OutlierDetectionMethod,

    PatternAnalysisResult,
    PatternAnalyzer,

    PatternClassification,
    PatternRelationship,
    PatternType,
    PerformanceAnalysisResult,

    PerformanceBottleneck,
    // Performance Analysis Types
    PerformanceMetricsAnalysis,
    PerformanceThresholds,
    PerformanceTrendAnalysis,
    QualityAnalysisResult,
    QualityAnalyzer,
    // Quality Analysis Types
    QualityDimensions,
    QualityRecommendation,

    QualityThresholds,
    QualityTrend,
    RelationshipType,

    ReliabilityMetrics,
    ResourceUtilizationAnalysis,
    SeasonalComponent,
    SeasonalEffect,
    ShapeAssessment,
    SignificantCorrelation,
    SlaCompliance,
    SlaViolation,
    StatisticalAnalysisResult,
    StatisticalAnalyzer,
    StatisticalAnalyzerConfig,
    StatisticalAnalyzerStats,
    StatisticalMetadata,
    StatisticalTest,
    TailAnalysis,
    TrendAnalysisResult,
    TrendAnalyzer,
    // Trend Analysis Types
    TrendComponent,
    TrendComponents,
    TrendDataPoint,
    TrendForecast,
    UtilizationMetrics,
};

// Re-export buffer components for convenience
pub use buffer::{
    BufferIterator,
    BufferManager,

    BufferManagerConfig,
    BufferManagerStats,
    // Core Buffer Components
    BufferPool,
    // Configuration Types
    BufferPoolConfig,
    // Statistics Types
    BufferPoolStatistics,
    CompressionAlgorithm,

    CompressionConfig,
    // Utilities
    CompressionUtils,
    FileRotationConfig,

    FileStorage,

    // Storage Backends
    MemoryStorage,
    // Enums
    OverflowStrategy,
    PreallocationStrategy,
    StorageStats,
};

// Re-export notifications components for convenience
pub use notifications::{
    AdaptiveRateConfig,

    AdaptiveRateController,
    AuditSystem,
    ChannelConfig,
    // Health Monitoring
    ChannelHealth,
    // Notification Channels
    // (Already imported from threshold module)

    // Trait Definitions
    // NotificationChannel already imported from threshold
    ChannelHealthMonitor,
    ChannelRateLimitStats,
    // Rate Limiting
    ChannelRateLimiter,
    CompiledTemplate,

    DeliveryEngine,
    DeliveryRecord,
    DeliveryResult,

    DeliveryStats,
    EscalationEngine,

    FormattingStats,
    GlobalRateLimitStats,
    GlobalRateLimiter,
    HealthCheckConfig,

    LoadMetrics,
    MessageFormatter,
    // Configuration Types
    NotificationConfig,
    // Core Notification Components
    NotificationManager,
    // Statistics Types
    NotificationStats,
    // Delivery Components
    PendingDelivery,
    PriorityQueue,
    // Message Types
    ProcessedNotification,
    // Enums
    ProcessingStatus,

    RateLimiter,
    RateLimitingStats,

    RetryItem,

    RetryPolicy,
    RetryScheduler,
    RetryStats,
    // Template Engine
    TemplateEngine,
    ThrottledNotification,
    TokenBucket,
};
