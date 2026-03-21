//! Live Optimization Engine
//!
//! Main optimization engine with real-time recommendation generation

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use std::{
    collections::VecDeque,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::{sync::broadcast, task::JoinHandle, time::interval};
use tracing::{debug, error, info, warn};

use super::super::types::*;
use super::{
    advanced_algorithms::*,
    algorithms::*,
    analysis::RealTimeAnalyzer,
    components::{AdaptiveLearner, ImpactAssessor, StrategySelector},
    confidence::ConfidenceScorer,
    recommendation::RecommendationGenerator,
    support::OptimizationEngineStatistics,
};

// LIVE OPTIMIZATION ENGINE
// =============================================================================

/// Live optimization engine for real-time recommendations
///
/// Advanced optimization engine that analyzes streaming performance data to
/// generate real-time optimization recommendations with confidence scoring,
/// impact assessment, and machine learning-based adaptive optimization.
pub struct LiveOptimizationEngine {
    /// Optimization algorithms
    algorithms: Arc<Mutex<Vec<Box<dyn LiveOptimizationAlgorithm + Send + Sync>>>>,

    /// Recommendation generator
    recommendation_generator: Arc<RecommendationGenerator>,

    /// Confidence scorer
    confidence_scorer: Arc<ConfidenceScorer>,

    /// Optimization history
    optimization_history: Arc<Mutex<VecDeque<OptimizationRecommendation>>>,

    /// Real-time analyzer
    realtime_analyzer: Arc<RealTimeAnalyzer>,

    /// Impact assessor
    impact_assessor: Arc<ImpactAssessor>,

    /// Strategy selector
    strategy_selector: Arc<StrategySelector>,

    /// Adaptive learner
    adaptive_learner: Arc<AdaptiveLearner>,

    /// Engine configuration
    config: Arc<RwLock<OptimizationEngineConfig>>,

    /// Recommendation publisher
    recommendation_publisher: Arc<broadcast::Sender<OptimizationRecommendation>>,

    /// Engine statistics
    stats: Arc<OptimizationEngineStats>,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,

    /// Background tasks
    background_tasks: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

/// Statistics for optimization engine
#[derive(Debug, Default)]
pub struct OptimizationEngineStats {
    /// Total recommendations generated
    pub recommendations_generated: AtomicU64,

    /// Successful optimizations
    pub successful_optimizations: AtomicU64,

    /// Average confidence score
    pub average_confidence: AtomicF32,

    /// Average impact score
    pub average_impact: AtomicF32,

    /// Engine uptime
    pub uptime: AtomicU64,

    /// Processing latency
    pub processing_latency_ms: AtomicF32,

    /// Active algorithms count
    pub active_algorithms: AtomicUsize,
}

impl LiveOptimizationEngine {
    /// Create a new live optimization engine
    ///
    /// Initializes a comprehensive optimization engine that analyzes streaming
    /// performance data to generate real-time optimization recommendations.
    pub async fn new(config: OptimizationEngineConfig) -> Result<Self> {
        let (recommendation_sender, _) = broadcast::channel(1000);

        let engine = Self {
            algorithms: Arc::new(Mutex::new(Vec::new())),
            recommendation_generator: Arc::new(RecommendationGenerator::new().await?),
            confidence_scorer: Arc::new(ConfidenceScorer::new().await?),
            optimization_history: Arc::new(Mutex::new(VecDeque::new())),
            realtime_analyzer: Arc::new(RealTimeAnalyzer::new().await?),
            impact_assessor: Arc::new(ImpactAssessor::new().await?),
            strategy_selector: Arc::new(StrategySelector::new().await?),
            adaptive_learner: Arc::new(AdaptiveLearner::new().await?),
            config: Arc::new(RwLock::new(config)),
            recommendation_publisher: Arc::new(recommendation_sender),
            stats: Arc::new(OptimizationEngineStats::default()),
            shutdown: Arc::new(AtomicBool::new(false)),
            background_tasks: Arc::new(Mutex::new(Vec::new())),
        };

        engine.initialize_algorithms().await?;
        Ok(engine)
    }

    /// Start live optimization engine
    ///
    /// Begins continuous analysis of performance data and generation of
    /// real-time optimization recommendations.
    pub async fn start(&self) -> Result<()> {
        info!("Starting live optimization engine");

        // Start components
        self.recommendation_generator.start().await?;
        self.realtime_analyzer.start().await?;
        self.impact_assessor.start().await?;
        self.strategy_selector.start().await?;
        self.adaptive_learner.start().await?;

        // Start background tasks
        self.start_background_tasks().await?;

        info!("Live optimization engine started successfully");
        Ok(())
    }

    /// Generate optimization recommendations
    ///
    /// Analyzes current performance data and generates actionable optimization
    /// recommendations with confidence scoring and impact assessment.
    pub async fn generate_recommendations(
        &self,
        metrics: &RealTimeMetrics,
        history: &[TimestampedMetrics],
        context: &OptimizationContext,
    ) -> Result<Vec<OptimizationRecommendation>> {
        let start_time = Instant::now();

        // Select optimal strategies based on context
        let selected_algorithms = self.strategy_selector.select_algorithms(context).await?;

        let algorithms = self.algorithms.lock();
        let mut all_recommendations = Vec::new();

        // Generate recommendations using selected algorithms
        for algorithm in algorithms.iter() {
            if selected_algorithms.contains(&algorithm.name().to_string()) {
                match algorithm.optimize(metrics, history, context) {
                    Ok(mut recommendations) => {
                        all_recommendations.append(&mut recommendations);
                    },
                    Err(e) => {
                        error!("Optimization algorithm '{}' error: {}", algorithm.name(), e);
                    },
                }
            }
        }

        drop(algorithms);

        // Score recommendations for confidence
        let scored_recommendations = self.score_recommendations(&all_recommendations).await?;

        // Assess impact for each recommendation
        let impact_assessed_recommendations = self.assess_impact(&scored_recommendations).await?;

        // Filter by confidence threshold and limit count
        let config = self.config.read();
        let filtered_recommendations: Vec<OptimizationRecommendation> =
            impact_assessed_recommendations
                .into_iter()
                .filter(|rec| rec.confidence >= config.min_confidence_threshold)
                .take(config.max_recommendations)
                .collect();

        // Store in history and update statistics
        self.update_history(&filtered_recommendations).await?;
        self.update_statistics(&filtered_recommendations, start_time.elapsed()).await?;

        // Publish recommendations
        for recommendation in &filtered_recommendations {
            let _ = self.recommendation_publisher.send(recommendation.clone());
        }

        // Update adaptive learner with new data
        self.adaptive_learner
            .update_with_recommendations(&filtered_recommendations)
            .await?;

        debug!(
            "Generated {} optimization recommendations in {:?}",
            filtered_recommendations.len(),
            start_time.elapsed()
        );

        Ok(filtered_recommendations)
    }

    /// Subscribe to optimization recommendations
    ///
    /// Creates a subscription to receive real-time optimization recommendations
    /// as they are generated by the engine.
    pub fn subscribe_to_recommendations(&self) -> broadcast::Receiver<OptimizationRecommendation> {
        self.recommendation_publisher.subscribe()
    }

    /// Get optimization history
    ///
    /// Returns the history of optimization recommendations within the specified
    /// time range for analysis and tracking.
    pub async fn get_optimization_history(
        &self,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Vec<OptimizationRecommendation> {
        let history = self.optimization_history.lock();
        history
            .iter()
            .filter(|rec| rec.timestamp >= start_time && rec.timestamp <= end_time)
            .cloned()
            .collect()
    }

    /// Get engine statistics
    ///
    /// Returns comprehensive statistics about the optimization engine performance.
    pub fn get_statistics(&self) -> OptimizationEngineStatistics {
        OptimizationEngineStatistics {
            recommendations_generated: self.stats.recommendations_generated.load(Ordering::Relaxed),
            successful_optimizations: self.stats.successful_optimizations.load(Ordering::Relaxed),
            average_confidence: self.stats.average_confidence.load(Ordering::Relaxed),
            average_impact: self.stats.average_impact.load(Ordering::Relaxed),
            uptime_seconds: self.stats.uptime.load(Ordering::Relaxed),
            processing_latency_ms: self.stats.processing_latency_ms.load(Ordering::Relaxed),
            active_algorithms: self.stats.active_algorithms.load(Ordering::Relaxed),
        }
    }

    /// Update engine configuration
    ///
    /// Updates the optimization engine configuration at runtime.
    pub async fn update_config(&self, config: OptimizationEngineConfig) -> Result<()> {
        let mut current_config = self.config.write();
        *current_config = config;
        info!("Optimization engine configuration updated");
        Ok(())
    }

    /// Shutdown optimization engine
    ///
    /// Gracefully shuts down the optimization engine and cleans up resources.
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down live optimization engine");

        self.shutdown.store(true, Ordering::Relaxed);

        // Stop background tasks
        let mut tasks = self.background_tasks.lock();
        for task in tasks.drain(..) {
            task.abort();
        }

        // Shutdown components
        self.recommendation_generator.shutdown().await?;
        self.realtime_analyzer.shutdown().await?;
        self.impact_assessor.shutdown().await?;
        self.strategy_selector.shutdown().await?;
        self.adaptive_learner.shutdown().await?;

        info!("Live optimization engine shutdown complete");
        Ok(())
    }

    // Private helper methods

    async fn initialize_algorithms(&self) -> Result<()> {
        let mut algorithms = self.algorithms.lock();

        // Add core optimization algorithms
        algorithms.push(Box::new(ParallelismOptimizationAlgorithm::new()));
        algorithms.push(Box::new(ResourceOptimizationAlgorithm::new()));
        algorithms.push(Box::new(BatchingOptimizationAlgorithm::new()));
        algorithms.push(Box::new(PerformanceTuningAlgorithm::new()));

        // Add enhanced algorithms
        algorithms.push(Box::new(MemoryOptimizationAlgorithm::new()));
        algorithms.push(Box::new(IOOptimizationAlgorithm::new()));
        algorithms.push(Box::new(NetworkOptimizationAlgorithm::new()));
        algorithms.push(Box::new(ThreadPoolOptimizationAlgorithm::new()));

        self.stats.active_algorithms.store(algorithms.len(), Ordering::Relaxed);

        info!("Initialized {} optimization algorithms", algorithms.len());
        Ok(())
    }

    async fn score_recommendations(
        &self,
        recommendations: &[OptimizationRecommendation],
    ) -> Result<Vec<OptimizationRecommendation>> {
        let mut scored_recommendations = Vec::new();

        for recommendation in recommendations {
            let mut scored_recommendation = recommendation.clone();

            match self.confidence_scorer.score_recommendation(recommendation).await {
                Ok(confidence) => {
                    scored_recommendation.confidence = confidence;
                    scored_recommendations.push(scored_recommendation);
                },
                Err(e) => {
                    warn!(
                        "Failed to score recommendation {}: {}",
                        recommendation.id, e
                    );
                    scored_recommendation.confidence = 0.5; // Default confidence
                    scored_recommendations.push(scored_recommendation);
                },
            }
        }

        Ok(scored_recommendations)
    }

    async fn assess_impact(
        &self,
        recommendations: &[OptimizationRecommendation],
    ) -> Result<Vec<OptimizationRecommendation>> {
        let mut impact_assessed_recommendations = Vec::new();

        for recommendation in recommendations {
            let mut assessed_recommendation = recommendation.clone();

            match self.impact_assessor.assess_impact(recommendation).await {
                Ok(impact) => {
                    assessed_recommendation.expected_impact = impact;
                    impact_assessed_recommendations.push(assessed_recommendation);
                },
                Err(e) => {
                    warn!(
                        "Failed to assess impact for recommendation {}: {}",
                        recommendation.id, e
                    );
                    impact_assessed_recommendations.push(assessed_recommendation);
                },
            }
        }

        Ok(impact_assessed_recommendations)
    }

    async fn update_history(&self, recommendations: &[OptimizationRecommendation]) -> Result<()> {
        let mut history = self.optimization_history.lock();

        for recommendation in recommendations {
            history.push_back(recommendation.clone());

            // Limit history size
            if history.len() > 10000 {
                history.pop_front();
            }
        }

        Ok(())
    }

    async fn update_statistics(
        &self,
        recommendations: &[OptimizationRecommendation],
        processing_time: Duration,
    ) -> Result<()> {
        self.stats
            .recommendations_generated
            .fetch_add(recommendations.len() as u64, Ordering::Relaxed);
        self.stats
            .processing_latency_ms
            .store(processing_time.as_millis() as f32, Ordering::Relaxed);

        if !recommendations.is_empty() {
            let avg_confidence = recommendations.iter().map(|r| r.confidence).sum::<f32>()
                / recommendations.len() as f32;
            let avg_impact =
                recommendations.iter().map(|r| r.expected_impact.overall_score()).sum::<f32>()
                    / recommendations.len() as f32;

            self.stats.average_confidence.store(avg_confidence, Ordering::Relaxed);
            self.stats.average_impact.store(avg_impact, Ordering::Relaxed);
        }

        Ok(())
    }

    async fn start_background_tasks(&self) -> Result<()> {
        let mut tasks = self.background_tasks.lock();

        // Background optimization monitoring task
        let engine_clone = Arc::new(self.clone());
        let monitoring_task = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));

            while !engine_clone.shutdown.load(Ordering::Relaxed) {
                interval.tick().await;

                if let Err(e) = engine_clone.perform_background_optimization().await {
                    error!("Background optimization error: {}", e);
                }
            }
        });

        tasks.push(monitoring_task);

        Ok(())
    }

    async fn perform_background_optimization(&self) -> Result<()> {
        // Update uptime
        self.stats.uptime.fetch_add(60, Ordering::Relaxed);

        // Perform adaptive learning updates
        self.adaptive_learner.perform_background_learning().await?;

        // Update strategy selection weights
        self.strategy_selector.update_strategy_weights().await?;

        // Clean up old history entries
        self.cleanup_old_history().await?;

        Ok(())
    }

    async fn cleanup_old_history(&self) -> Result<()> {
        let mut history = self.optimization_history.lock();
        let cutoff_time = Utc::now() - chrono::Duration::hours(24);

        history.retain(|recommendation| recommendation.timestamp > cutoff_time);

        Ok(())
    }
}

// Clone implementation for LiveOptimizationEngine
impl Clone for LiveOptimizationEngine {
    fn clone(&self) -> Self {
        Self {
            algorithms: Arc::clone(&self.algorithms),
            recommendation_generator: Arc::clone(&self.recommendation_generator),
            confidence_scorer: Arc::clone(&self.confidence_scorer),
            optimization_history: Arc::clone(&self.optimization_history),
            realtime_analyzer: Arc::clone(&self.realtime_analyzer),
            impact_assessor: Arc::clone(&self.impact_assessor),
            strategy_selector: Arc::clone(&self.strategy_selector),
            adaptive_learner: Arc::clone(&self.adaptive_learner),
            config: Arc::clone(&self.config),
            recommendation_publisher: Arc::clone(&self.recommendation_publisher),
            stats: Arc::clone(&self.stats),
            shutdown: Arc::clone(&self.shutdown),
            background_tasks: Arc::clone(&self.background_tasks),
        }
    }
}
