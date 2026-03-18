//! # Threshold Monitoring and Alerting Module
//!
//! This module provides comprehensive threshold monitoring and alerting functionality for the
//! TrustformeRS real-time metrics system. It includes intelligent threshold evaluation,
//! multi-level alerting, escalation policies, dynamic threshold adaptation, and comprehensive
//! alert management with suppression and correlation capabilities.
//!
//! ## Key Components
//!
//! - **ThresholdMonitor**: Core threshold monitoring system with real-time evaluation
//! - **ThresholdEvaluator**: Multiple evaluation algorithms (Simple, Statistical, Adaptive)
//! - **AlertManager**: Comprehensive alert processing and notification management
//! - **EscalationManager**: Multi-level alert escalation with severity classification
//! - **AdaptiveThresholdController**: Dynamic threshold adjustment based on system behavior
//! - **AlertSuppressor**: Alert deduplication and suppression algorithms
//! - **AlertCorrelator**: Alert correlation and relationship analysis
//! - **PerformanceAnalyzer**: Threshold performance impact analysis and optimization
//!
//! ## Features
//!
//! - Real-time threshold monitoring with microsecond precision
//! - Multiple threshold evaluation strategies with adaptive capabilities
//! - Intelligent alert generation with severity classification
//! - Advanced escalation policies with time-based triggers
//! - Dynamic threshold adaptation based on historical data and system behavior
//! - Alert suppression and deduplication to reduce noise
//! - Alert correlation to identify related issues
//! - Performance impact monitoring for threshold evaluation overhead
//! - Comprehensive notification system with multiple channels
//! - Thread-safe concurrent processing with minimal performance impact
//! - Extensive configuration options and real-time monitoring
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use threshold::{ThresholdMonitor, ThresholdConfig, ThresholdDirection};
//! use std::time::Duration;
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Create threshold monitor
//!     let monitor = ThresholdMonitor::new().await?;
//!
//!     // Configure CPU threshold
//!     let cpu_threshold = ThresholdConfig {
//!         name: "high_cpu_usage".to_string(),
//!         metric: "cpu_utilization".to_string(),
//!         warning_threshold: 0.8,
//!         critical_threshold: 0.95,
//!         direction: ThresholdDirection::Above,
//!         adaptive: true,
//!         evaluation_window: Duration::from_secs(60),
//!         min_trigger_count: 3,
//!         cooldown_period: Duration::from_secs(300),
//!         escalation_policy: "default".to_string(),
//!     };
//!
//!     // Add threshold and start monitoring
//!     monitor.add_threshold(cpu_threshold).await?;
//!     monitor.start_monitoring().await?;
//!
//!     // Evaluate metrics continuously
//!     loop {
//!         let metrics = collect_current_metrics().await?;
//!         let alerts = monitor.evaluate_thresholds(&metrics).await?;
//!
//!         if !alerts.is_empty() {
//!             println!("Generated {} alerts", alerts.len());
//!         }
//!
//!         tokio::time::sleep(Duration::from_secs(1)).await;
//!     }
//! }
//! ```

// Module declarations
pub mod adaptive_controller;
pub mod adaptive_evaluator;
pub mod alert_manager;
pub mod correlator;
pub mod error;
pub mod escalation;
pub mod evaluator;
pub mod monitor;
pub mod processors;
pub mod simple_evaluator;
pub mod statistical_evaluator;
pub mod suppressor;
pub mod types;

// Re-export commonly used items
pub use adaptive_controller::{
    AdaptationStats, AdaptiveThresholdConfig, AdaptiveThresholdController, ThresholdAdaptation,
    ThresholdAdaptationAlgorithm,
};
pub use adaptive_evaluator::{
    AdaptationEngine, AdaptationRecord, AdaptiveEvaluatorConfig, AdaptiveThreshold,
    AdaptiveThresholdEvaluator, DetectedPattern, PatternDetector, SeasonalAnalyzer,
};
pub use alert_manager::{AlertManager, AlertManagerConfig, AlertManagerStats};
pub use correlator::{
    AlertCorrelation, AlertCorrelator, CorrelationConfig, CorrelationCriteria, CorrelationRule,
    CorrelationRuleType, CorrelationStats, SeverityMatching,
};
pub use error::{Result, ThresholdError};
pub use escalation::{
    EscalationAction, EscalationConfig, EscalationEvent, EscalationEventType, EscalationLevel,
    EscalationManager, EscalationPolicy, EscalationState, EscalationStats, EscalationTrigger,
};
pub use evaluator::ThresholdEvaluator;
pub use monitor::{
    AlgorithmStats, CompletedEvaluation, EvaluationTrack, MLAdaptationConfig, MLModelState,
    MachineLearningAdaptationAlgorithm, PerformanceAnalyzer, PerformanceAnalyzerConfig,
    PerformanceAnalyzerStats, PerformanceMetrics, PerformanceSnapshot, PerformanceTrends,
    ResourceUtilization, SeasonalPattern, SeasonalPatternType, StatisticalAdaptationAlgorithm,
    StatisticalAdaptationConfig, ThresholdMonitor, ThresholdMonitorConfig, ThresholdStatistics,
    ThroughputAnalysis, TrendAnalysis, TrendAnalysisAlgorithm, TrendAnalysisConfig, TrendDirection,
    TrendState,
};
pub use processors::{
    ChannelStats, CriticalAlertProcessor, CriticalProcessorConfig, DefaultAlertProcessor,
    DefaultProcessorConfig, EmailChannelConfig, EmailNotificationChannel, LogChannelConfig,
    LogNotificationChannel, PerformanceAlertProcessor, PerformanceProcessorConfig, ProcessorStats,
    ResourceAlertProcessor, ResourceProcessorConfig, SlackChannelConfig, SlackNotificationChannel,
    WebhookChannelConfig, WebhookNotificationChannel,
};
pub use simple_evaluator::{EvaluatorStats, SimpleEvaluatorConfig, SimpleThresholdEvaluator};
pub use statistical_evaluator::{
    ConfidenceMethod, StatisticalDataPoint, StatisticalEvaluatorConfig,
    StatisticalThresholdEvaluator,
};
pub use suppressor::{
    AlertFingerprint, AlertSuppressor, SuppressionAction, SuppressionConfig, SuppressionCriteria,
    SuppressionRule, SuppressionRuleType, SuppressionStats,
};
pub use types::{
    CorrelationInfo, CorrelationType, EvaluationPerformance, EvaluationStatistics,
    ThresholdEvaluation, ThresholdMonitoringState,
};
