//! Adaptive Parallelism Module for Performance Optimizer
//!
//! This module has been refactored into a modular architecture for better maintainability.
//! All functionality has been organized into focused sub-modules while maintaining
//! backward compatibility through comprehensive re-exports.
//!
//! # Architecture Overview
//!
//! The adaptive parallelism system is organized into the following modules:
//! - [`controller`] - Main adaptive parallelism controller implementation
//! - [`estimator`] - Optimal parallelism estimation coordination
//! - [`estimation_algorithms`] - Various estimation algorithms (linear regression, resource-based, etc.)
//! - [`feedback_system`] - Performance feedback collection and processing
//! - [`feedback_processors`] - Specialized feedback processors for different metrics
//! - [`aggregation`] - Feedback aggregation strategies and coordination
//! - [`learning_model`] - Adaptive learning model and algorithms
//! - [`validation`] - Model validation strategies and frameworks
//!
//! # Key Features
//!
//! - **Adaptive Parallelism Control**: Dynamic adjustment of parallelism levels based on performance feedback
//! - **Multiple Estimation Algorithms**: Linear regression, resource-based, historical average, and CPU affinity estimators
//! - **Real-time Feedback Processing**: Immediate processing and aggregation of performance feedback
//! - **Machine Learning Integration**: Adaptive learning models for continuous improvement
//! - **Comprehensive Validation**: Cross-validation and holdout validation strategies
//! - **Conservative Adjustment Modes**: Stability-focused adjustment strategies
//! - **Performance Tracking**: Historical tracking of adjustments and effectiveness
//!
//! # Usage Examples
//!
//! ```rust
//! use crate::performance_optimizer::adaptive_parallelism::{
//!     AdaptiveParallelismController, AdaptiveParallelismConfig,
//!     OptimalParallelismEstimator, PerformanceFeedbackSystem
//! };
//!
//! // Initialize adaptive parallelism controller
//! let config = AdaptiveParallelismConfig::default();
//! let controller = AdaptiveParallelismController::new(config).await?;
//!
//! // Get optimal parallelism recommendation
//! let characteristics = TestCharacteristics::default();
//! let estimate = controller.recommend_parallelism(&characteristics).await?;
//! println!("Recommended parallelism: {}", estimate.optimal_parallelism);
//!
//! // Adjust parallelism based on performance
//! let performance = PerformanceMeasurement::default();
//! let new_level = controller.adjust_parallelism(
//!     AdjustmentReason::PerformanceDegradation,
//!     performance,
//!     &characteristics,
//! ).await?;
//! ```
//!
//! # Backward Compatibility
//!
//! All original APIs are preserved through comprehensive re-exports. Existing code
//! will continue to work without modification while benefiting from the improved
//! modular organization and enhanced performance.

// =============================================================================
// MODULE DECLARATIONS
// =============================================================================

/// Adaptive parallelism controller and orchestration
pub mod controller;

/// Optimal parallelism estimation coordination
pub mod estimator;

/// Estimation algorithms for optimal parallelism
pub mod estimation_algorithms;

/// Performance feedback collection and processing
pub mod feedback_system;

/// Specialized feedback processors for different metrics
pub mod feedback_processors;

/// Feedback aggregation strategies and coordination
pub mod aggregation;

/// Adaptive learning model and machine learning algorithms
pub mod learning_model;

/// Model validation strategies and frameworks
pub mod validation;

// =============================================================================
// COMPREHENSIVE RE-EXPORTS FOR BACKWARD COMPATIBILITY
// =============================================================================

// Re-export everything from the parent types module
pub use crate::performance_optimizer::types::*;

// =============================================================================
// CORE CONTROLLER EXPORTS
// =============================================================================

// Main controller implementation

// =============================================================================
// ESTIMATION SYSTEM EXPORTS
// =============================================================================

// Optimal parallelism estimator

// Estimation algorithms
pub use estimation_algorithms::{
    CpuAffinityEstimator, EstimationAlgorithm, HistoricalAverageEstimator,
    LinearRegressionEstimator, ResourceBasedEstimator,
};

// =============================================================================
// FEEDBACK SYSTEM EXPORTS
// =============================================================================

// Feedback system coordination

// Feedback processors
pub use feedback_processors::{
    FeedbackProcessor, LatencyFeedbackProcessor, ResourceUtilizationFeedbackProcessor,
    ThroughputFeedbackProcessor,
};

// =============================================================================
// AGGREGATION SYSTEM EXPORTS
// =============================================================================

// Aggregation coordination and strategies
pub use aggregation::{
    AggregationStrategy, ConfidenceWeightedAggregation, ConsensusAggregation,
    WeightedAverageAggregation,
};

// Re-export FeedbackAggregator from types module (for backward compatibility)
pub use crate::performance_optimizer::types::FeedbackAggregator;

// =============================================================================
// LEARNING MODEL EXPORTS
// =============================================================================

// Learning model and algorithms
pub use learning_model::{AdaptiveLinearRegression, LearningAlgorithmExt};

// =============================================================================
// VALIDATION SYSTEM EXPORTS
// =============================================================================

// Validation strategies and frameworks
pub use validation::{CrossValidationStrategy, HoldoutValidationStrategy, ValidationStrategy};

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/// Create a default adaptive parallelism controller
pub async fn create_default_adaptive_controller() -> anyhow::Result<AdaptiveParallelismController> {
    let config = AdaptiveParallelismConfig::default();
    AdaptiveParallelismController::new(config).await
}

/// Create an adaptive parallelism controller optimized for high-performance scenarios
pub async fn create_high_performance_controller() -> anyhow::Result<AdaptiveParallelismController> {
    let mut config = AdaptiveParallelismConfig::default();
    config.conservative_mode = false;
    config.exploration_rate = 0.2;
    config.stability_threshold = 0.1;

    AdaptiveParallelismController::new(config).await
}

/// Create an adaptive parallelism controller optimized for stability
pub async fn create_conservative_controller() -> anyhow::Result<AdaptiveParallelismController> {
    let mut config = AdaptiveParallelismConfig::default();
    config.conservative_mode = true;
    config.exploration_rate = 0.05;
    config.stability_threshold = 0.05;

    AdaptiveParallelismController::new(config).await
}

/// Create a machine learning enhanced controller with adaptive learning
pub async fn create_ml_enhanced_controller() -> anyhow::Result<AdaptiveParallelismController> {
    let mut config = AdaptiveParallelismConfig::default();
    config.exploration_rate = 0.15;

    let controller = AdaptiveParallelismController::new(config).await?;

    // The controller automatically includes ML components
    Ok(controller)
}

/// Quick parallelism estimation for immediate decision making
pub async fn quick_parallelism_estimate(
    characteristics: &TestCharacteristics,
) -> anyhow::Result<ParallelismEstimate> {
    let controller = create_default_adaptive_controller().await?;
    controller.recommend_parallelism(characteristics).await
}

/// Quick performance assessment for current parallelism level
pub async fn assess_current_parallelism_performance(
    current_level: usize,
    characteristics: &TestCharacteristics,
) -> anyhow::Result<f32> {
    let controller = create_default_adaptive_controller().await?;
    let estimate = controller.recommend_parallelism(characteristics).await?;

    // Calculate relative performance score
    let optimal_level = estimate.optimal_parallelism;
    let performance_score = if current_level == optimal_level {
        1.0
    } else {
        let distance = (current_level as f32 - optimal_level as f32).abs();
        (1.0 / (1.0 + distance * 0.1)).max(0.1)
    };

    Ok(performance_score * estimate.confidence)
}

// =============================================================================
// LEGACY COMPATIBILITY
// =============================================================================

// Legacy type aliases for smooth migration
pub type AdaptiveParallelismControllerLegacy = AdaptiveParallelismController;
pub type OptimalParallelismEstimatorLegacy = OptimalParallelismEstimator;
pub type PerformanceFeedbackSystemLegacy = PerformanceFeedbackSystem;

// Legacy function aliases
pub use create_default_adaptive_controller as new_adaptive_controller;
pub use quick_parallelism_estimate as estimate_optimal_parallelism;

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn test_adaptive_parallelism_controller_creation() {
        let controller = create_default_adaptive_controller().await;
        assert!(controller.is_ok());
    }

    #[tokio::test]
    async fn test_high_performance_controller() {
        let controller = create_high_performance_controller().await;
        assert!(controller.is_ok());
    }

    #[tokio::test]
    async fn test_conservative_controller() {
        let controller = create_conservative_controller().await;
        assert!(controller.is_ok());
    }

    #[tokio::test]
    async fn test_ml_enhanced_controller() {
        let controller = create_ml_enhanced_controller().await;
        assert!(controller.is_ok());
    }

    #[tokio::test]
    async fn test_quick_parallelism_estimate() {
        let characteristics = TestCharacteristics {
            category_distribution: std::collections::HashMap::new(),
            average_duration: Duration::from_millis(100),
            resource_intensity: ResourceIntensity::default(),
            concurrency_requirements: ConcurrencyRequirements::default(),
            dependency_complexity: 0.3,
        };

        let estimate = quick_parallelism_estimate(&characteristics).await;
        assert!(estimate.is_ok());

        if let Ok(est) = estimate {
            assert!(est.optimal_parallelism >= 1);
            assert!(est.confidence >= 0.0 && est.confidence <= 1.0);
        }
    }

    #[tokio::test]
    async fn test_performance_assessment() {
        let characteristics = TestCharacteristics::default();
        let score = assess_current_parallelism_performance(4, &characteristics).await;
        if let Err(ref e) = score {
            eprintln!("Error in test_performance_assessment: {:?}", e);
        }
        assert!(score.is_ok());

        if let Ok(s) = score {
            assert!((0.0..=1.0).contains(&s));
        }
    }

    #[test]
    fn test_backward_compatibility() {
        // Test that legacy type aliases work
        let _: Option<AdaptiveParallelismControllerLegacy> = None;
        let _: Option<OptimalParallelismEstimatorLegacy> = None;
        let _: Option<PerformanceFeedbackSystemLegacy> = None;
    }

    #[test]
    fn test_module_integration() {
        // Test that all modules export their types correctly
        let _: Option<Box<dyn EstimationAlgorithm>> = None;
        let _: Option<Box<dyn FeedbackProcessor>> = None;
        let _: Option<Box<dyn AggregationStrategy>> = None;
        let _: Option<Box<dyn LearningAlgorithm>> = None;
        let _: Option<Box<dyn ValidationStrategy>> = None;
    }
}
