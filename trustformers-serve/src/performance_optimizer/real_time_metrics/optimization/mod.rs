//! Comprehensive Optimization Module for Real-Time Metrics System
//!
//! This module provides an advanced LiveOptimizationEngine that analyzes streaming performance
//! data to generate real-time optimization recommendations with confidence scoring, impact
//! assessment, and machine learning-based adaptive optimization capabilities.
//!
//! ## Key Components
//!
//! - **LiveOptimizationEngine**: Main optimization engine with real-time recommendation generation
//! - **Optimization Algorithms**: Multiple strategies for different optimization scenarios
//! - **Recommendation Generation**: Intelligent recommendation system with confidence scoring
//! - **Real-time Analysis**: Continuous performance analysis and bottleneck identification
//! - **Impact Assessment**: Predictive impact analysis and validation
//! - **Adaptive Learning**: Machine learning-based optimization improvement
//! - **Strategy Selection**: Intelligent algorithm selection based on system characteristics
//! - **Historical Tracking**: Optimization history and learning from past results
//!
//! ## Features
//!
//! - Real-time optimization with multiple algorithmic strategies
//! - Intelligent recommendation generation with confidence scoring
//! - Performance bottleneck identification and resolution
//! - Machine learning-based adaptive optimization
//! - Historical optimization tracking and learning
//! - Real-time impact assessment and validation
//! - Comprehensive error handling and recovery
//! - Thread-safe concurrent optimization with minimal overhead
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use trustformers_serve::performance_optimizer::real_time_metrics::optimization::LiveOptimizationEngine;
//! use trustformers_serve::performance_optimizer::real_time_metrics::OptimizationEngineConfig;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create and start optimization engine with default configuration
//!     let engine = LiveOptimizationEngine::new(OptimizationEngineConfig::default()).await?;
//!     engine.start().await?;
//!
//!     // Subscribe to recommendations
//!     let mut receiver = engine.subscribe_to_recommendations();
//!
//!     // Process optimization recommendations
//!     while let Ok(recommendation) = receiver.recv().await {
//!         println!("New optimization recommendation: {:?}", recommendation);
//!     }
//!
//!     Ok(())
//! }
//! ```

// Module declarations
mod advanced_algorithms;
mod algorithms;
mod analysis;
mod components;
mod confidence;
mod engine;
mod recommendation;
mod support;

// Re-export public types from each module
pub use advanced_algorithms::*;
pub use algorithms::*;
pub use analysis::*;
pub use components::*;
pub use confidence::*;
pub use engine::*;
pub use recommendation::*;
pub use support::*;

// =============================================================================
// UNIT TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::super::types::*;
    use super::*;
    use chrono::Utc;
    use std::{
        collections::HashMap,
        time::{Duration, Instant},
    };

    #[tokio::test]
    async fn test_live_optimization_engine_creation() {
        let config = OptimizationEngineConfig::default();
        let engine = LiveOptimizationEngine::new(config).await;
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_recommendation_generator() {
        let generator = RecommendationGenerator::new()
            .await
            .expect("Failed to create recommendation generator");

        let context =
            OptimizationContext::new(SystemState::default(), TestCharacteristics::default());

        let metrics = RealTimeMetrics::default();

        let recommendations = generator.generate_recommendations(&context, &metrics).await;
        assert!(recommendations.is_ok());
    }

    #[tokio::test]
    async fn test_confidence_scorer() {
        let scorer = ConfidenceScorer::new().await.expect("Failed to create confidence scorer");

        let recommendation = OptimizationRecommendation {
            id: "test_rec".to_string(),
            timestamp: Utc::now(),
            actions: vec![RecommendedAction {
                action_type: ActionType::IncreaseParallelism,
                parameters: HashMap::new(),
                priority: 1.0,
                expected_impact: 0.8,
                estimated_duration: Duration::from_secs(60),
                reversible: true,
            }],
            expected_impact: ImpactAssessment::default(),
            confidence: 0.8,
            analysis: "Test recommendation".to_string(),
            risks: Vec::new(),
            priority: 1,
            implementation_time: Duration::from_secs(60),
        };

        let confidence = scorer.score_recommendation(&recommendation).await;
        assert!(confidence.is_ok());
        let confidence_val = confidence.expect("Confidence is None");
        assert!((0.0..=1.0).contains(&confidence_val));
    }

    #[tokio::test]
    async fn test_parallelism_optimization_algorithm() {
        let algorithm = ParallelismOptimizationAlgorithm::new();

        let metrics = RealTimeMetrics::default();

        let context =
            OptimizationContext::new(SystemState::default(), TestCharacteristics::default());

        let recommendations = algorithm.optimize(&metrics, &[], &context);
        assert!(recommendations.is_ok());

        let recs = recommendations.expect("Recommendations is None");
        if !recs.is_empty() {
            // Should recommend increasing parallelism due to low CPU utilization
            assert!(recs.iter().any(|r| r
                .actions
                .iter()
                .any(|a| matches!(a.action_type, ActionType::IncreaseParallelism))));
        }
    }

    #[tokio::test]
    async fn test_real_time_analyzer() {
        let analyzer = RealTimeAnalyzer::new().await.expect("Failed to create analyzer");

        let metrics = RealTimeMetrics::default();

        let history = vec![TimestampedMetrics {
            timestamp: Utc::now() - chrono::Duration::seconds(60),
            precise_timestamp: Instant::now(),
            metrics: metrics.clone(),
            system_state: SystemState::default(),
            quality_score: 1.0,
            source: "test".to_string(),
            metadata: HashMap::new(),
        }];

        let results = analyzer.analyze(&metrics, &history).await;
        assert!(results.is_ok());

        let analysis_results = results.expect("Results is None");
        assert!(!analysis_results.is_empty());
    }

    #[test]
    fn test_impact_assessment_overall_score() {
        let impact = ImpactAssessment {
            performance_impact: 0.3,
            resource_impact: 0.1,
            complexity: 0.4,
            risk_level: 0.2,
            estimated_benefit: 0.5,
            implementation_time: Duration::from_secs(120),
        };

        let score = impact.overall_score();
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_optimization_context_creation() {
        let system_state = SystemState::default();
        let test_characteristics = TestCharacteristics::default();

        let context = OptimizationContext::new(system_state, test_characteristics);
        assert!(context.system_state.available_cores > 0);
    }
}
