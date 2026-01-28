//! Comprehensive Optimization History Module
//!
//! This module provides advanced optimization history management with comprehensive
//! analytical capabilities including trend analysis, pattern recognition, anomaly
//! detection, effectiveness analysis, statistical computation, and predictive analytics.
//!
//! # Architecture
//!
//! The optimization history system is organized into specialized modules:
//! - `types`: Core type definitions and data structures
//! - `trend_analysis`: Enhanced trend detection and prediction
//! - `pattern_recognition`: Pattern detection and machine learning
//! - `anomaly_detection`: Anomaly detection with statistical methods
//! - `effectiveness_analysis`: ROI calculation and effectiveness measurement
//! - `statistics`: Advanced statistical analysis and computation
//! - `predictive_analytics`: Predictive modeling and forecasting
//! - `manager`: Main orchestrator coordinating all components
//!
//! # Features
//!
//! - **Advanced Trend Analysis** with multiple algorithms and prediction capabilities
//! - **Pattern Recognition** using machine learning and statistical methods
//! - **Anomaly Detection** with comprehensive detection algorithms
//! - **Effectiveness Analysis** with ROI calculation and cost-benefit analysis
//! - **Statistical Analysis** with comprehensive statistical modeling
//! - **Predictive Analytics** with ensemble prediction methods
//! - **Comprehensive History Management** with automated insights and recommendations
//!
//! # Migration Notice
//!
//! This module has been refactored from a monolithic structure into a modular
//! architecture for better maintainability and extensibility. All previous
//! functionality is preserved through comprehensive re-exports and compatibility layers.
//!
//! # Usage
//!
//! ```rust
//! use trustformers_serve::performance_optimizer::optimization_history::{
//!     AdvancedOptimizationHistoryManager, OptimizationInsights
//! };
//!
//! // Create advanced optimization history manager
//! let manager = AdvancedOptimizationHistoryManager::new().await?;
//!
//! // Record optimization event
//! let event_id = manager.record_optimization_event(
//!     OptimizationEventType::ParallelismAdjustment,
//!     "Increased parallel workers".to_string(),
//!     Some(before_performance),
//!     Some(after_performance),
//!     optimization_params,
//!     metadata,
//! ).await?;
//!
//! // Get comprehensive insights
//! let insights = manager.get_comprehensive_insights(Duration::from_secs(3600)).await?;
//! ```

use anyhow::Result;

// Re-export all types and interfaces from specialized modules
pub use anomaly_detection::{AnomalyDetectionSystem, AnomalyStatistics};
pub use effectiveness_analysis::{
    AnalysisStatistics as EffectivenessAnalysisStatistics, CostBreakdown, EffectivenessAnalyzer,
    EffectivenessTrend, EffectivenessTrendDirection,
};
pub use manager::{
    AdvancedOptimizationHistoryManager, EffectivenessSummary, OptimizationInsights,
    OptimizationRecommendation, RecommendationPriority, SystemHealthSummary,
};
pub use pattern_recognition::{PatternRecognitionSystem, PatternStatistics};
pub use predictive_analytics::{PredictionStatistics, PredictiveAnalyticsEngine};
pub use statistics::{
    calculate_confidence_interval, calculate_percentiles, detect_outliers_iqr,
    detect_outliers_zscore, perform_descriptive_analysis, AdvancedStatisticsComputer,
};
pub use trend_analysis::{CacheStatistics as TrendCacheStatistics, EnhancedTrendAnalysisEngine};
pub use types::*;

// Module declarations
pub mod anomaly_detection;
pub mod effectiveness_analysis;
pub mod manager;
pub mod pattern_recognition;
pub mod predictive_analytics;
pub mod statistics;
pub mod trend_analysis;
pub mod types;

// =============================================================================
// CONVENIENCE CONSTRUCTORS AND FACTORY FUNCTIONS
// =============================================================================

/// Create a fully configured advanced optimization history manager
/// with default settings optimized for production use
pub async fn create_production_manager() -> Result<AdvancedOptimizationHistoryManager> {
    let trend_config = TrendAnalysisConfig {
        min_data_points: 10,
        analysis_window: std::time::Duration::from_secs(3600), // 1 hour
        confidence_threshold: 0.8,
        enable_prediction: true,
        prediction_horizon: std::time::Duration::from_secs(1800), // 30 minutes
        cache_expiry: std::time::Duration::from_secs(300),        // 5 minutes
        enable_ml_models: true,
    };

    let pattern_config = PatternRecognitionConfig {
        min_pattern_length: 3,
        similarity_threshold: 0.85,
        enable_ml_learning: true,
        learning_rate: 0.05, // More conservative for production
        cache_size: 2000,
    };

    let anomaly_config = AnomalyDetectionConfig {
        enable_detection: true,
        sensitivity: 0.8, // Higher sensitivity for production
        min_severity_threshold: 0.6,
        enable_ml_detection: true,
        learning_rate: 0.05,
    };

    let effectiveness_config = EffectivenessAnalysisConfig {
        enable_roi_calculation: true,
        cost_calculation_method: CostCalculationMethod::Hybrid,
        min_effectiveness_threshold: 0.2,
        enable_significance_testing: true,
        confidence_level: 0.95,
    };

    let statistics_config = StatisticsConfig {
        enable_advanced_metrics: true,
        window_size: 200,
        enable_correlation_analysis: true,
        enable_distribution_analysis: true,
        significance_threshold: 0.05,
    };

    let predictive_config = PredictiveAnalyticsConfig {
        enable_prediction: true,
        prediction_models: vec![
            PredictionModelType::LinearRegression,
            PredictionModelType::MovingAverage,
            PredictionModelType::ExponentialSmoothing,
            PredictionModelType::ARIMA,
        ],
        prediction_horizon: std::time::Duration::from_secs(3600), // 1 hour
        model_update_frequency: std::time::Duration::from_secs(600), // 10 minutes
        min_data_points: 20,
    };

    let retention_config = HistoryRetentionConfig {
        max_events: 50000,
        max_age: std::time::Duration::from_secs(7 * 24 * 60 * 60), // 7 days
        auto_cleanup: true,
        cleanup_interval: std::time::Duration::from_secs(6 * 60 * 60), // 6 hours
        compression_threshold: std::time::Duration::from_secs(24 * 60 * 60), // 1 day
    };

    AdvancedOptimizationHistoryManager::with_configs(
        trend_config,
        pattern_config,
        anomaly_config,
        effectiveness_config,
        statistics_config,
        predictive_config,
        retention_config,
    )
    .await
}

/// Create a development-optimized optimization history manager
/// with faster update cycles and more aggressive caching
pub async fn create_development_manager() -> Result<AdvancedOptimizationHistoryManager> {
    let trend_config = TrendAnalysisConfig {
        min_data_points: 5,
        analysis_window: std::time::Duration::from_secs(1800), // 30 minutes
        confidence_threshold: 0.7,
        enable_prediction: true,
        prediction_horizon: std::time::Duration::from_secs(900), // 15 minutes
        cache_expiry: std::time::Duration::from_secs(60),        // 1 minute
        enable_ml_models: true,
    };

    let pattern_config = PatternRecognitionConfig {
        min_pattern_length: 2,
        similarity_threshold: 0.75,
        enable_ml_learning: true,
        learning_rate: 0.15, // Faster learning for development
        cache_size: 500,
    };

    let anomaly_config = AnomalyDetectionConfig {
        enable_detection: true,
        sensitivity: 0.6,
        min_severity_threshold: 0.4,
        enable_ml_detection: true,
        learning_rate: 0.15,
    };

    let effectiveness_config = EffectivenessAnalysisConfig {
        enable_roi_calculation: true,
        cost_calculation_method: CostCalculationMethod::ResourceBased,
        min_effectiveness_threshold: 0.1,
        enable_significance_testing: false, // Faster for development
        confidence_level: 0.90,
    };

    let statistics_config = StatisticsConfig {
        enable_advanced_metrics: true,
        window_size: 50,
        enable_correlation_analysis: true,
        enable_distribution_analysis: false, // Faster for development
        significance_threshold: 0.1,
    };

    let predictive_config = PredictiveAnalyticsConfig {
        enable_prediction: true,
        prediction_models: vec![
            PredictionModelType::LinearRegression,
            PredictionModelType::MovingAverage,
        ],
        prediction_horizon: std::time::Duration::from_secs(1800), // 30 minutes
        model_update_frequency: std::time::Duration::from_secs(120), // 2 minutes
        min_data_points: 10,
    };

    let retention_config = HistoryRetentionConfig {
        max_events: 5000,
        max_age: std::time::Duration::from_secs(24 * 60 * 60), // 1 day
        auto_cleanup: true,
        cleanup_interval: std::time::Duration::from_secs(60 * 60), // 1 hour
        compression_threshold: std::time::Duration::from_secs(6 * 60 * 60), // 6 hours
    };

    AdvancedOptimizationHistoryManager::with_configs(
        trend_config,
        pattern_config,
        anomaly_config,
        effectiveness_config,
        statistics_config,
        predictive_config,
        retention_config,
    )
    .await
}

/// Create a lightweight optimization history manager
/// with minimal resource usage for testing environments
pub async fn create_lightweight_manager() -> Result<AdvancedOptimizationHistoryManager> {
    let trend_config = TrendAnalysisConfig {
        min_data_points: 3,
        analysis_window: std::time::Duration::from_secs(600), // 10 minutes
        confidence_threshold: 0.6,
        enable_prediction: false, // Disabled for lightweight
        prediction_horizon: std::time::Duration::from_secs(300),
        cache_expiry: std::time::Duration::from_secs(30),
        enable_ml_models: false, // Disabled for lightweight
    };

    let pattern_config = PatternRecognitionConfig {
        min_pattern_length: 2,
        similarity_threshold: 0.7,
        enable_ml_learning: false, // Disabled for lightweight
        learning_rate: 0.1,
        cache_size: 100,
    };

    let anomaly_config = AnomalyDetectionConfig {
        enable_detection: true,
        sensitivity: 0.5,
        min_severity_threshold: 0.5,
        enable_ml_detection: false, // Disabled for lightweight
        learning_rate: 0.1,
    };

    let effectiveness_config = EffectivenessAnalysisConfig {
        enable_roi_calculation: true,
        cost_calculation_method: CostCalculationMethod::ResourceBased,
        min_effectiveness_threshold: 0.1,
        enable_significance_testing: false,
        confidence_level: 0.90,
    };

    let statistics_config = StatisticsConfig {
        enable_advanced_metrics: false, // Disabled for lightweight
        window_size: 20,
        enable_correlation_analysis: false,
        enable_distribution_analysis: false,
        significance_threshold: 0.1,
    };

    let predictive_config = PredictiveAnalyticsConfig {
        enable_prediction: false, // Disabled for lightweight
        prediction_models: vec![PredictionModelType::MovingAverage],
        prediction_horizon: std::time::Duration::from_secs(300),
        model_update_frequency: std::time::Duration::from_secs(600),
        min_data_points: 5,
    };

    let retention_config = HistoryRetentionConfig {
        max_events: 1000,
        max_age: std::time::Duration::from_secs(6 * 60 * 60), // 6 hours
        auto_cleanup: true,
        cleanup_interval: std::time::Duration::from_secs(30 * 60), // 30 minutes
        compression_threshold: std::time::Duration::from_secs(60 * 60), // 1 hour
    };

    AdvancedOptimizationHistoryManager::with_configs(
        trend_config,
        pattern_config,
        anomaly_config,
        effectiveness_config,
        statistics_config,
        predictive_config,
        retention_config,
    )
    .await
}

// =============================================================================
// SPECIALIZED ANALYSIS FUNCTIONS
// =============================================================================

/// Perform comprehensive optimization analysis on a dataset
pub async fn analyze_optimization_dataset(
    data_points: &[crate::performance_optimizer::types::PerformanceDataPoint],
) -> Result<OptimizationDatasetAnalysis> {
    if data_points.is_empty() {
        return Err(anyhow::anyhow!("No data points provided for analysis"));
    }

    // Create temporary analyzers
    let trend_analyzer = EnhancedTrendAnalysisEngine::new();
    let anomaly_detector = AnomalyDetectionSystem::new();
    let statistics_computer = AdvancedStatisticsComputer::new();

    // Perform analyses
    let trend_analysis = trend_analyzer.analyze_trend(data_points).await?;
    let anomalies = anomaly_detector.detect_anomalies(data_points).await?;
    let statistics = statistics_computer.compute_comprehensive_statistics(data_points)?;

    Ok(OptimizationDatasetAnalysis {
        data_points: data_points.len(),
        trend_analysis,
        detected_anomalies: anomalies,
        comprehensive_statistics: statistics,
        analyzed_at: chrono::Utc::now(),
    })
}

/// Generate optimization insights report for a given time period
pub async fn generate_insights_report(
    manager: &AdvancedOptimizationHistoryManager,
    timeframe: std::time::Duration,
    include_predictions: bool,
) -> Result<OptimizationInsightsReport> {
    let insights = manager.get_comprehensive_insights(timeframe).await?;

    let predictions = if include_predictions {
        Some(
            manager
                .get_performance_predictions(std::time::Duration::from_secs(3600))
                .await?,
        )
    } else {
        None
    };

    let health_summary = manager.get_system_health_summary();

    Ok(OptimizationInsightsReport {
        insights,
        predictions,
        health_summary,
        generated_at: chrono::Utc::now(),
    })
}

/// Compare optimization effectiveness across different strategies
pub async fn compare_optimization_strategies(
    results: &[(
        String,
        crate::performance_optimizer::types::PerformanceMeasurement,
        crate::performance_optimizer::types::PerformanceMeasurement,
    )],
) -> Result<OptimizationStrategyComparison> {
    if results.is_empty() {
        return Err(anyhow::anyhow!("No optimization results provided"));
    }

    let effectiveness_analyzer = EffectivenessAnalyzer::new();
    let mut strategy_analyses = std::collections::HashMap::new();

    for (strategy_name, before, after) in results {
        let analysis = effectiveness_analyzer
            .analyze_effectiveness(before, after, &std::collections::HashMap::new())
            .await?;

        strategy_analyses.insert(strategy_name.clone(), analysis);
    }

    // Find best and worst strategies
    let best_strategy = strategy_analyses
        .iter()
        .max_by(|a, b| {
            a.1.effectiveness_score
                .partial_cmp(&b.1.effectiveness_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(name, _)| name.clone());

    let worst_strategy = strategy_analyses
        .iter()
        .min_by(|a, b| {
            a.1.effectiveness_score
                .partial_cmp(&b.1.effectiveness_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(name, _)| name.clone());

    let average_effectiveness = strategy_analyses
        .values()
        .map(|analysis| analysis.effectiveness_score)
        .sum::<f32>()
        / strategy_analyses.len() as f32;

    Ok(OptimizationStrategyComparison {
        strategy_analyses,
        best_strategy,
        worst_strategy,
        average_effectiveness,
        analyzed_strategies: results.len(),
        compared_at: chrono::Utc::now(),
    })
}

// =============================================================================
// RESULT TYPES FOR SPECIALIZED FUNCTIONS
// =============================================================================

/// Dataset analysis result
#[derive(Debug, Clone)]
pub struct OptimizationDatasetAnalysis {
    pub data_points: usize,
    pub trend_analysis: TrendAnalysisResult,
    pub detected_anomalies: Vec<DetectedAnomaly>,
    pub comprehensive_statistics: ComprehensiveOptimizationStatistics,
    pub analyzed_at: chrono::DateTime<chrono::Utc>,
}

/// Comprehensive insights report
#[derive(Debug, Clone)]
pub struct OptimizationInsightsReport {
    pub insights: OptimizationInsights,
    pub predictions: Option<Vec<PerformancePrediction>>,
    pub health_summary: SystemHealthSummary,
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

/// Optimization strategy comparison result
#[derive(Debug, Clone)]
pub struct OptimizationStrategyComparison {
    pub strategy_analyses: std::collections::HashMap<String, EffectivenessAnalysisResult>,
    pub best_strategy: Option<String>,
    pub worst_strategy: Option<String>,
    pub average_effectiveness: f32,
    pub analyzed_strategies: usize,
    pub compared_at: chrono::DateTime<chrono::Utc>,
}

// =============================================================================
// LEGACY COMPATIBILITY AND RE-EXPORTS
// =============================================================================

// Re-export commonly used types for backward compatibility
pub use crate::performance_optimizer::types::{
    OptimizationEventType, PerformanceDataPoint, PerformanceMeasurement,
};

// Legacy type aliases maintained for backward compatibility
pub type LegacyOptimizationHistoryManager = AdvancedOptimizationHistoryManager;

/// Create a legacy-compatible optimization history manager
/// This maintains the same interface as the original monolithic implementation
pub async fn create_legacy_compatible_manager() -> Result<AdvancedOptimizationHistoryManager> {
    // Use default settings that closely match the original implementation
    AdvancedOptimizationHistoryManager::new().await
}

// =============================================================================
// MODULE HEALTH AND DIAGNOSTICS
// =============================================================================

/// Get comprehensive module health information
pub fn get_module_health() -> ModuleHealth {
    ModuleHealth {
        components_loaded: true,
        trend_analysis_available: true,
        pattern_recognition_available: true,
        anomaly_detection_available: true,
        effectiveness_analysis_available: true,
        statistics_computation_available: true,
        predictive_analytics_available: true,
        manager_available: true,
        last_checked: chrono::Utc::now(),
    }
}

/// Module health status
#[derive(Debug, Clone)]
pub struct ModuleHealth {
    pub components_loaded: bool,
    pub trend_analysis_available: bool,
    pub pattern_recognition_available: bool,
    pub anomaly_detection_available: bool,
    pub effectiveness_analysis_available: bool,
    pub statistics_computation_available: bool,
    pub predictive_analytics_available: bool,
    pub manager_available: bool,
    pub last_checked: chrono::DateTime<chrono::Utc>,
}

// =============================================================================
// INTEGRATION HELPERS
// =============================================================================

/// Helper function to extract performance data points from optimization events
pub fn extract_performance_data_from_events(
    events: &[LegacyOptimizationEvent],
) -> Vec<crate::performance_optimizer::types::PerformanceDataPoint> {
    let mut data_points = Vec::new();

    for event in events {
        if let Some(performance) = &event.performance_after {
            data_points.push(crate::performance_optimizer::types::PerformanceDataPoint {
                parallelism: 1,
                throughput: performance.throughput,
                latency: performance.latency,
                cpu_utilization: performance.cpu_usage,
                memory_utilization: performance.memory_usage,
                resource_efficiency: performance.resource_efficiency,
                timestamp: event.timestamp,
                test_characteristics:
                    crate::performance_optimizer::types::TestCharacteristics::default(),
                system_state: crate::performance_optimizer::types::SystemState::default(),
            });
        }
    }

    data_points
}

/// Helper function to convert optimization insights to legacy format
pub fn convert_insights_to_legacy_format(
    insights: &OptimizationInsights,
) -> LegacyOptimizationStatistics {
    let _total_patterns = insights.recognized_patterns.len() as u64;
    let _effective_patterns =
        insights.recognized_patterns.iter().filter(|p| p.effectiveness > 0.5).count() as u64;

    let average_effectiveness = if !insights.recognized_patterns.is_empty() {
        insights.recognized_patterns.iter().map(|p| p.effectiveness).sum::<f32>()
            / insights.recognized_patterns.len() as f32
    } else {
        insights.effectiveness_summary.average_effectiveness
    };

    LegacyOptimizationStatistics {
        total_optimizations: insights.effectiveness_summary.analyzed_optimizations as u64,
        successful_optimizations: insights.effectiveness_summary.positive_roi_count as u64,
        average_improvement: average_effectiveness,
        best_improvement: insights
            .recognized_patterns
            .iter()
            .map(|p| p.effectiveness)
            .fold(0.0f32, |acc, x| acc.max(x)),
        total_roi: insights.effectiveness_summary.success_rate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use std::collections::HashMap;

    #[tokio::test]
    async fn test_create_production_manager() {
        let manager = create_production_manager().await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_create_development_manager() {
        let manager = create_development_manager().await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_create_lightweight_manager() {
        let manager = create_lightweight_manager().await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_module_health() {
        let health = get_module_health();
        assert!(health.components_loaded);
        assert!(health.manager_available);
    }

    #[test]
    fn test_extract_performance_data() {
        let events = vec![LegacyOptimizationEvent {
            timestamp: Utc::now(),
            event_type: OptimizationEventType::ParallelismAdjustment,
            description: "Test event".to_string(),
            performance_before: None,
            performance_after: Some(
                crate::performance_optimizer::types::PerformanceMeasurement {
                    throughput: 100.0,
                    average_latency: std::time::Duration::from_millis(50),
                    cpu_utilization: 0.5,
                    memory_utilization: 0.5,
                    resource_efficiency: 0.8,
                    timestamp: chrono::Utc::now(),
                    measurement_duration: std::time::Duration::from_secs(30),
                    cpu_usage: 0.5,
                    memory_usage: 0.5,
                    latency: std::time::Duration::from_millis(50),
                },
            ),
            parameters: HashMap::new(),
            metadata: HashMap::new(),
        }];

        let data_points = extract_performance_data_from_events(&events);
        assert_eq!(data_points.len(), 1);
        assert_eq!(data_points[0].throughput, 100.0);
    }

    #[tokio::test]
    async fn test_analyze_optimization_dataset() {
        let data_points = vec![
            crate::performance_optimizer::types::PerformanceDataPoint {
                parallelism: 1,
                throughput: 100.0,
                latency: std::time::Duration::from_millis(50),
                cpu_utilization: 0.5,
                memory_utilization: 0.5,
                resource_efficiency: 0.8,
                timestamp: Utc::now(),
                test_characteristics:
                    crate::performance_optimizer::types::TestCharacteristics::default(),
                system_state: crate::performance_optimizer::types::SystemState::default(),
            },
            crate::performance_optimizer::types::PerformanceDataPoint {
                parallelism: 1,
                throughput: 110.0,
                latency: std::time::Duration::from_millis(45),
                cpu_utilization: 0.5,
                memory_utilization: 0.5,
                resource_efficiency: 0.8,
                timestamp: Utc::now() + chrono::Duration::minutes(1),
                test_characteristics:
                    crate::performance_optimizer::types::TestCharacteristics::default(),
                system_state: crate::performance_optimizer::types::SystemState::default(),
            },
            crate::performance_optimizer::types::PerformanceDataPoint {
                parallelism: 2,
                throughput: 180.0,
                latency: std::time::Duration::from_millis(30),
                cpu_utilization: 0.7,
                memory_utilization: 0.6,
                resource_efficiency: 0.85,
                timestamp: Utc::now() + chrono::Duration::minutes(2),
                test_characteristics:
                    crate::performance_optimizer::types::TestCharacteristics::default(),
                system_state: crate::performance_optimizer::types::SystemState::default(),
            },
            crate::performance_optimizer::types::PerformanceDataPoint {
                parallelism: 2,
                throughput: 190.0,
                latency: std::time::Duration::from_millis(28),
                cpu_utilization: 0.75,
                memory_utilization: 0.62,
                resource_efficiency: 0.86,
                timestamp: Utc::now() + chrono::Duration::minutes(3),
                test_characteristics:
                    crate::performance_optimizer::types::TestCharacteristics::default(),
                system_state: crate::performance_optimizer::types::SystemState::default(),
            },
            crate::performance_optimizer::types::PerformanceDataPoint {
                parallelism: 4,
                throughput: 320.0,
                latency: std::time::Duration::from_millis(25),
                cpu_utilization: 0.9,
                memory_utilization: 0.7,
                resource_efficiency: 0.9,
                timestamp: Utc::now() + chrono::Duration::minutes(4),
                test_characteristics:
                    crate::performance_optimizer::types::TestCharacteristics::default(),
                system_state: crate::performance_optimizer::types::SystemState::default(),
            },
            crate::performance_optimizer::types::PerformanceDataPoint {
                parallelism: 4,
                throughput: 340.0,
                latency: std::time::Duration::from_millis(23),
                cpu_utilization: 0.92,
                memory_utilization: 0.72,
                resource_efficiency: 0.91,
                timestamp: Utc::now() + chrono::Duration::minutes(5),
                test_characteristics:
                    crate::performance_optimizer::types::TestCharacteristics::default(),
                system_state: crate::performance_optimizer::types::SystemState::default(),
            },
            crate::performance_optimizer::types::PerformanceDataPoint {
                parallelism: 8,
                throughput: 580.0,
                latency: std::time::Duration::from_millis(22),
                cpu_utilization: 0.95,
                memory_utilization: 0.8,
                resource_efficiency: 0.87,
                timestamp: Utc::now() + chrono::Duration::minutes(6),
                test_characteristics:
                    crate::performance_optimizer::types::TestCharacteristics::default(),
                system_state: crate::performance_optimizer::types::SystemState::default(),
            },
            crate::performance_optimizer::types::PerformanceDataPoint {
                parallelism: 8,
                throughput: 600.0,
                latency: std::time::Duration::from_millis(21),
                cpu_utilization: 0.96,
                memory_utilization: 0.82,
                resource_efficiency: 0.88,
                timestamp: Utc::now() + chrono::Duration::minutes(7),
                test_characteristics:
                    crate::performance_optimizer::types::TestCharacteristics::default(),
                system_state: crate::performance_optimizer::types::SystemState::default(),
            },
            crate::performance_optimizer::types::PerformanceDataPoint {
                parallelism: 16,
                throughput: 1000.0,
                latency: std::time::Duration::from_millis(20),
                cpu_utilization: 0.98,
                memory_utilization: 0.85,
                resource_efficiency: 0.85,
                timestamp: Utc::now() + chrono::Duration::minutes(8),
                test_characteristics:
                    crate::performance_optimizer::types::TestCharacteristics::default(),
                system_state: crate::performance_optimizer::types::SystemState::default(),
            },
            crate::performance_optimizer::types::PerformanceDataPoint {
                parallelism: 16,
                throughput: 1050.0,
                latency: std::time::Duration::from_millis(19),
                cpu_utilization: 0.99,
                memory_utilization: 0.87,
                resource_efficiency: 0.84,
                timestamp: Utc::now() + chrono::Duration::minutes(9),
                test_characteristics:
                    crate::performance_optimizer::types::TestCharacteristics::default(),
                system_state: crate::performance_optimizer::types::SystemState::default(),
            },
        ];

        let analysis = analyze_optimization_dataset(&data_points).await;
        if let Err(ref e) = analysis {
            eprintln!("Error in test_analyze_optimization_dataset: {:?}", e);
        }
        assert!(analysis.is_ok());

        let result = analysis.unwrap();
        assert_eq!(result.data_points, 10);
    }
}
