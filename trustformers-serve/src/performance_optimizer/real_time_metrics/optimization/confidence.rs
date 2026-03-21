//! Confidence Scoring System
//!
//! Provides confidence scoring for optimization recommendations

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tracing::{info, warn};

use super::super::types::*;

// =============================================================================

/// Confidence scorer for optimization recommendations
///
/// Evaluates and scores the confidence level of optimization recommendations
/// based on multiple factors including historical success, data quality, and risk assessment.
pub struct ConfidenceScorer {
    /// Scoring algorithms
    algorithms: Arc<Mutex<Vec<Box<dyn ConfidenceScoringAlgorithm + Send + Sync>>>>,

    /// Scoring statistics
    stats: Arc<ConfidenceScorerStats>,

    /// Configuration
    config: Arc<RwLock<ConfidenceScorerConfig>>,

    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
}

/// Statistics for confidence scorer
#[derive(Debug, Default)]
pub struct ConfidenceScorerStats {
    /// Recommendations scored
    pub recommendations_scored: AtomicU64,

    /// Average confidence score
    pub average_confidence: AtomicF32,

    /// Scoring accuracy
    pub scoring_accuracy: AtomicF32,

    /// Processing time
    pub avg_processing_time_ms: AtomicF32,
}

/// Configuration for confidence scorer
#[derive(Debug, Clone)]
pub struct ConfidenceScorerConfig {
    /// Scoring algorithm weights
    pub algorithm_weights: HashMap<String, f32>,

    /// Minimum confidence threshold
    pub min_confidence: f32,

    /// Maximum confidence cap
    pub max_confidence: f32,

    /// Scoring timeout
    pub scoring_timeout: Duration,
}

impl Default for ConfidenceScorerConfig {
    fn default() -> Self {
        let mut algorithm_weights = HashMap::new();
        algorithm_weights.insert("historical".to_string(), 0.4);
        algorithm_weights.insert("statistical".to_string(), 0.3);
        algorithm_weights.insert("risk_based".to_string(), 0.2);
        algorithm_weights.insert("consensus".to_string(), 0.1);

        Self {
            algorithm_weights,
            min_confidence: 0.1,
            max_confidence: 0.95,
            scoring_timeout: Duration::from_secs(10),
        }
    }
}

/// Trait for confidence scoring algorithms
pub trait ConfidenceScoringAlgorithm: Send + Sync {
    /// Score recommendation confidence
    fn score(
        &self,
        recommendation: &OptimizationRecommendation,
    ) -> Result<f32, RealTimeMetricsError>;

    /// Get algorithm name
    fn name(&self) -> &str;

    /// Get algorithm weight
    fn weight(&self) -> f32;

    /// Update algorithm with feedback
    fn update_with_feedback(
        &mut self,
        recommendation: &OptimizationRecommendation,
        actual_outcome: bool,
    );
}

impl ConfidenceScorer {
    /// Create a new confidence scorer
    pub async fn new() -> Result<Self> {
        let scorer = Self {
            algorithms: Arc::new(Mutex::new(Vec::new())),
            stats: Arc::new(ConfidenceScorerStats::default()),
            config: Arc::new(RwLock::new(ConfidenceScorerConfig::default())),
            shutdown: Arc::new(AtomicBool::new(false)),
        };

        scorer.initialize_algorithms().await?;
        Ok(scorer)
    }

    /// Score recommendation confidence
    pub async fn score_recommendation(
        &self,
        recommendation: &OptimizationRecommendation,
    ) -> Result<f32> {
        let start_time = Instant::now();
        let algorithms = self.algorithms.lock();
        let config = self.config.read();

        let mut weighted_score = 0.0;
        let mut total_weight = 0.0;

        for algorithm in algorithms.iter() {
            let weight = config.algorithm_weights.get(algorithm.name()).copied().unwrap_or(1.0);

            match algorithm.score(recommendation) {
                Ok(score) => {
                    weighted_score += score * weight;
                    total_weight += weight;
                },
                Err(e) => {
                    warn!("Confidence scoring error from {}: {}", algorithm.name(), e);
                },
            }
        }

        let final_score = if total_weight > 0.0 {
            (weighted_score / total_weight)
                .max(config.min_confidence)
                .min(config.max_confidence)
        } else {
            0.5 // Default confidence
        };

        // Update statistics
        self.stats.recommendations_scored.fetch_add(1, Ordering::Relaxed);
        self.stats
            .avg_processing_time_ms
            .store(start_time.elapsed().as_millis() as f32, Ordering::Relaxed);

        // Update running average confidence
        let current_count = self.stats.recommendations_scored.load(Ordering::Relaxed) as f32;
        let current_avg = self.stats.average_confidence.load(Ordering::Relaxed);
        let new_avg = (current_avg * (current_count - 1.0) + final_score) / current_count;
        self.stats.average_confidence.store(new_avg, Ordering::Relaxed);

        Ok(final_score)
    }

    /// Shutdown confidence scorer
    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown.store(true, Ordering::Relaxed);
        info!("Confidence scorer shutdown complete");
        Ok(())
    }

    async fn initialize_algorithms(&self) -> Result<()> {
        let mut algorithms = self.algorithms.lock();

        algorithms.push(Box::new(HistoricalConfidenceAlgorithm::new()));
        algorithms.push(Box::new(StatisticalConfidenceAlgorithm::new()));
        algorithms.push(Box::new(RiskBasedConfidenceAlgorithm::new()));
        algorithms.push(Box::new(ConsensusConfidenceAlgorithm::new()));

        info!(
            "Initialized {} confidence scoring algorithms",
            algorithms.len()
        );
        Ok(())
    }
}

/// Historical confidence algorithm
///
/// Scores confidence based on historical success rates of similar recommendations.
pub struct HistoricalConfidenceAlgorithm {
    historical_data: HashMap<String, HistoricalRecord>,
}

#[derive(Debug, Clone)]
struct HistoricalRecord {
    total_attempts: u32,
    successful_attempts: u32,
    last_updated: DateTime<Utc>,
    average_impact: f32,
}

impl Default for HistoricalConfidenceAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl HistoricalConfidenceAlgorithm {
    pub fn new() -> Self {
        Self {
            historical_data: HashMap::new(),
        }
    }

    fn get_recommendation_key(&self, recommendation: &OptimizationRecommendation) -> String {
        // Create a key based on recommendation characteristics
        let action_types: Vec<String> = recommendation
            .actions
            .iter()
            .map(|action| format!("{:?}", action.action_type))
            .collect();

        format!(
            "{}_priority_{}",
            action_types.join("_"),
            recommendation.priority
        )
    }
}

impl ConfidenceScoringAlgorithm for HistoricalConfidenceAlgorithm {
    fn score(
        &self,
        recommendation: &OptimizationRecommendation,
    ) -> Result<f32, RealTimeMetricsError> {
        let key = self.get_recommendation_key(recommendation);

        if let Some(record) = self.historical_data.get(&key) {
            let success_rate = record.successful_attempts as f32 / record.total_attempts as f32;
            let confidence = success_rate * 0.9 + 0.1; // Minimum 10% confidence
            Ok(confidence.min(0.95))
        } else {
            // No historical data, use moderate confidence
            Ok(0.6)
        }
    }

    fn name(&self) -> &str {
        "historical"
    }

    fn weight(&self) -> f32 {
        0.4
    }

    fn update_with_feedback(
        &mut self,
        recommendation: &OptimizationRecommendation,
        actual_outcome: bool,
    ) {
        let key = self.get_recommendation_key(recommendation);

        let record = self.historical_data.entry(key).or_insert(HistoricalRecord {
            total_attempts: 0,
            successful_attempts: 0,
            last_updated: Utc::now(),
            average_impact: 0.0,
        });

        record.total_attempts += 1;
        if actual_outcome {
            record.successful_attempts += 1;
        }
        record.last_updated = Utc::now();

        // Update average impact
        let current_impact = recommendation.expected_impact.overall_score();
        record.average_impact = (record.average_impact * (record.total_attempts - 1) as f32
            + current_impact)
            / record.total_attempts as f32;
    }
}

/// Statistical confidence algorithm
///
/// Scores confidence based on statistical analysis of recommendation data and metrics.
pub struct StatisticalConfidenceAlgorithm {
    data_quality_weights: HashMap<String, f32>,
}

impl Default for StatisticalConfidenceAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticalConfidenceAlgorithm {
    pub fn new() -> Self {
        let mut weights = HashMap::new();
        weights.insert("impact_score".to_string(), 0.3);
        weights.insert("risk_assessment".to_string(), 0.25);
        weights.insert("implementation_complexity".to_string(), 0.2);
        weights.insert("data_completeness".to_string(), 0.15);
        weights.insert("consistency".to_string(), 0.1);

        Self {
            data_quality_weights: weights,
        }
    }

    fn calculate_data_quality(&self, recommendation: &OptimizationRecommendation) -> f32 {
        let impact_score = recommendation.expected_impact.overall_score().min(1.0);
        let risk_score = 1.0 - recommendation.expected_impact.risk_level;
        let complexity_score = 1.0 - recommendation.expected_impact.complexity;

        // Data completeness based on available fields
        let data_completeness =
            if !recommendation.analysis.is_empty() && !recommendation.actions.is_empty() {
                1.0
            } else {
                0.7
            };

        // Consistency score based on relationship between confidence and impact
        let consistency =
            if (recommendation.confidence - impact_score).abs() < 0.3 { 1.0 } else { 0.8 };

        impact_score * self.data_quality_weights["impact_score"]
            + risk_score * self.data_quality_weights["risk_assessment"]
            + complexity_score * self.data_quality_weights["implementation_complexity"]
            + data_completeness * self.data_quality_weights["data_completeness"]
            + consistency * self.data_quality_weights["consistency"]
    }
}

impl ConfidenceScoringAlgorithm for StatisticalConfidenceAlgorithm {
    fn score(
        &self,
        recommendation: &OptimizationRecommendation,
    ) -> Result<f32, RealTimeMetricsError> {
        let data_quality = self.calculate_data_quality(recommendation);
        let base_confidence = recommendation.confidence;

        // Adjust confidence based on statistical analysis
        let statistical_confidence = (base_confidence + data_quality) / 2.0;

        Ok(statistical_confidence.clamp(0.1, 0.95))
    }

    fn name(&self) -> &str {
        "statistical"
    }

    fn weight(&self) -> f32 {
        0.3
    }

    fn update_with_feedback(
        &mut self,
        _recommendation: &OptimizationRecommendation,
        _actual_outcome: bool,
    ) {
        // Statistical algorithm can adapt weights based on feedback
        // Implementation would involve updating data_quality_weights based on outcomes
    }
}

/// Risk-based confidence algorithm
///
/// Scores confidence based on risk assessment and mitigation strategies.
pub struct RiskBasedConfidenceAlgorithm {
    risk_factors: HashMap<RiskType, f32>,
}

impl Default for RiskBasedConfidenceAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl RiskBasedConfidenceAlgorithm {
    pub fn new() -> Self {
        let mut risk_factors = HashMap::new();
        risk_factors.insert(RiskType::Performance, 0.8);
        risk_factors.insert(RiskType::Operational, 0.9);
        risk_factors.insert(RiskType::Resource, 0.6);
        risk_factors.insert(RiskType::Security, 0.95);
        risk_factors.insert(RiskType::Operational, 1.0);

        Self { risk_factors }
    }

    fn assess_overall_risk(&self, recommendation: &OptimizationRecommendation) -> f32 {
        let base_risk = recommendation.expected_impact.risk_level;

        let risk_penalty = recommendation
            .risks
            .iter()
            .map(|risk| {
                // Convert String risk_type to RiskType enum
                let risk_type_enum = match risk.risk_type.as_str() {
                    "Performance" => RiskType::Performance,
                    "Security" => RiskType::Security,
                    "Resource" => RiskType::Resource,
                    "Operational" => RiskType::Operational,
                    "Financial" => RiskType::Financial,
                    "Technical" => RiskType::Technical,
                    _ => RiskType::Operational, // Default fallback
                };
                let factor = self.risk_factors.get(&risk_type_enum).copied().unwrap_or(0.7);
                risk.impact * risk.probability * factor
            })
            .sum::<f32>();

        (base_risk + risk_penalty).min(1.0)
    }
}

impl ConfidenceScoringAlgorithm for RiskBasedConfidenceAlgorithm {
    fn score(
        &self,
        recommendation: &OptimizationRecommendation,
    ) -> Result<f32, RealTimeMetricsError> {
        let overall_risk = self.assess_overall_risk(recommendation);
        let risk_adjusted_confidence = (1.0 - overall_risk) * recommendation.confidence;

        Ok(risk_adjusted_confidence.clamp(0.1, 0.95))
    }

    fn name(&self) -> &str {
        "risk_based"
    }

    fn weight(&self) -> f32 {
        0.2
    }

    fn update_with_feedback(
        &mut self,
        recommendation: &OptimizationRecommendation,
        actual_outcome: bool,
    ) {
        // Update risk factors based on actual outcomes
        for risk in &recommendation.risks {
            // Convert String risk_type to RiskType enum
            let risk_type_enum = match risk.risk_type.as_str() {
                "Performance" => RiskType::Performance,
                "Security" => RiskType::Security,
                "Resource" => RiskType::Resource,
                "Operational" => RiskType::Operational,
                "Financial" => RiskType::Financial,
                "Technical" => RiskType::Technical,
                _ => RiskType::Operational, // Default fallback
            };

            if let Some(factor) = self.risk_factors.get_mut(&risk_type_enum) {
                if actual_outcome {
                    // Successful outcome, slightly reduce risk factor
                    *factor *= 0.99;
                } else {
                    // Failed outcome, slightly increase risk factor
                    *factor = (*factor * 1.01).min(1.0);
                }
            }
        }
    }
}

/// Consensus confidence algorithm
///
/// Scores confidence based on consensus among multiple recommendation sources.
pub struct ConsensusConfidenceAlgorithm {
    consensus_threshold: f32,
    agreement_bonus: f32,
}

impl Default for ConsensusConfidenceAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsensusConfidenceAlgorithm {
    pub fn new() -> Self {
        Self {
            consensus_threshold: 0.7,
            agreement_bonus: 0.1,
        }
    }

    fn calculate_consensus_score(&self, recommendation: &OptimizationRecommendation) -> f32 {
        // Analyze consistency across multiple aspects of the recommendation
        let action_consistency = if recommendation.actions.len() > 1 {
            // Check if actions are complementary
            let priorities: Vec<f32> = recommendation.actions.iter().map(|a| a.priority).collect();
            let priority_variance = self.calculate_variance(&priorities);

            if priority_variance < 1.0 {
                1.0
            } else {
                0.8
            }
        } else {
            1.0
        };

        let confidence_impact_alignment = {
            let confidence = recommendation.confidence;
            let expected_benefit = recommendation.expected_impact.estimated_benefit;
            let alignment = 1.0 - (confidence - expected_benefit).abs();
            alignment.max(0.0)
        };

        (action_consistency + confidence_impact_alignment) / 2.0
    }

    fn calculate_variance(&self, values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance =
            values.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / values.len() as f32;

        variance
    }
}

impl ConfidenceScoringAlgorithm for ConsensusConfidenceAlgorithm {
    fn score(
        &self,
        recommendation: &OptimizationRecommendation,
    ) -> Result<f32, RealTimeMetricsError> {
        let consensus_score = self.calculate_consensus_score(recommendation);
        let base_confidence = recommendation.confidence;

        let consensus_confidence = if consensus_score >= self.consensus_threshold {
            base_confidence + self.agreement_bonus
        } else {
            base_confidence * consensus_score
        };

        Ok(consensus_confidence.clamp(0.1, 0.95))
    }

    fn name(&self) -> &str {
        "consensus"
    }

    fn weight(&self) -> f32 {
        0.1
    }

    fn update_with_feedback(
        &mut self,
        _recommendation: &OptimizationRecommendation,
        actual_outcome: bool,
    ) {
        // Adjust consensus parameters based on feedback
        if actual_outcome {
            self.agreement_bonus = (self.agreement_bonus * 1.01).min(0.2);
        } else {
            self.agreement_bonus *= 0.99;
        }
    }
}
