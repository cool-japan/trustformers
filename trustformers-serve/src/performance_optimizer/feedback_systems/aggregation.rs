//! Advanced Feedback Aggregation Strategies
//!
//! This module provides sophisticated aggregation strategies for combining
//! multiple feedback sources into coherent insights. Includes weighted confidence
//! aggregation, time-series analysis, consensus-based aggregation, and
//! multi-objective optimization approaches.

use anyhow::Result;
use chrono::Utc;
use std::collections::HashMap;

use super::types::*;
use crate::performance_optimizer::types::{
    AggregatedFeedback, AggregationStrategy, FeedbackSource, ProcessedFeedback, RecommendedAction,
};

// =============================================================================
// WEIGHTED CONFIDENCE AGGREGATION STRATEGY
// =============================================================================

impl WeightedConfidenceAggregationStrategy {
    /// Create new weighted confidence aggregation strategy
    pub fn new() -> Self {
        let mut source_weights = HashMap::new();
        source_weights.insert(FeedbackSource::PerformanceMonitor, 1.0);
        source_weights.insert(FeedbackSource::ResourceMonitor, 0.9);
        source_weights.insert(FeedbackSource::TestExecutionEngine, 0.8);
        source_weights.insert(FeedbackSource::ExternalSystem, 0.6);
        source_weights.insert(FeedbackSource::UserInput, 0.4);

        Self {
            confidence_weight: 0.7,
            recency_weight: 0.3,
            source_weights,
        }
    }

    /// Create with custom configuration
    pub fn with_config(_config: super::types::AggregationManagerConfig) -> Self {
        // For now, use default implementation
        Self::new()
    }

    /// Calculate effective weight for feedback
    fn calculate_effective_weight(&self, feedback: &ProcessedFeedback) -> f32 {
        let source_weight =
            self.source_weights.get(&feedback.original_feedback.source).unwrap_or(&0.5);

        let age = Utc::now()
            .signed_duration_since(feedback.original_feedback.timestamp)
            .num_seconds() as f32;
        let recency_factor = (-age / 3600.0).exp(); // Exponential decay over hours

        let confidence_component = feedback.confidence * self.confidence_weight;
        let recency_component = recency_factor * self.recency_weight;
        let source_component = source_weight * 0.2;

        confidence_component + recency_component + source_component
    }
}

impl AggregationStrategy for WeightedConfidenceAggregationStrategy {
    fn aggregate(&self, feedback: &[ProcessedFeedback]) -> Result<AggregatedFeedback> {
        if feedback.is_empty() {
            return Err(anyhow::anyhow!("No feedback to aggregate"));
        }

        let mut total_weight = 0.0f32;
        let mut weighted_sum = 0.0f64;
        let mut all_actions = Vec::new();

        for fb in feedback {
            let weight = self.calculate_effective_weight(fb);
            weighted_sum += fb.processed_value * weight as f64;
            total_weight += weight;

            if let Some(action) = &fb.recommended_action {
                all_actions.push(action.clone());
            }
        }

        let aggregated_value = if total_weight > 0.0 {
            weighted_sum / total_weight as f64
        } else {
            feedback.iter().map(|f| f.processed_value).sum::<f64>() / feedback.len() as f64
        };

        Ok(AggregatedFeedback {
            aggregated_value,
            confidence: total_weight / feedback.len() as f32,
            contributing_count: feedback.len(),
            contributing_feedback_count: feedback.len(),
            aggregation_method: "weighted_confidence".to_string(),
            timestamp: Utc::now(),
            recommended_actions: all_actions,
        })
    }

    fn name(&self) -> &str {
        "weighted_confidence"
    }

    fn is_applicable(&self, feedback: &[ProcessedFeedback]) -> bool {
        !feedback.is_empty()
    }
}

// =============================================================================
// TIME-SERIES AGGREGATION STRATEGY
// =============================================================================

impl TimeSeriesAggregationStrategy {
    /// Create new time-series aggregation strategy
    pub fn new(time_window: std::time::Duration) -> Self {
        Self {
            time_window,
            smoothing_factor: 0.3,
            trend_detection: true,
            seasonal_adjustment: false,
        }
    }

    /// Filter feedback within time window
    fn filter_by_time_window<'a>(
        &self,
        feedback: &'a [ProcessedFeedback],
    ) -> Vec<&'a ProcessedFeedback> {
        let cutoff_time =
            Utc::now() - chrono::Duration::from_std(self.time_window).unwrap_or_default();

        feedback
            .iter()
            .filter(|fb| fb.original_feedback.timestamp >= cutoff_time)
            .collect()
    }

    /// Apply exponential smoothing
    fn apply_exponential_smoothing(&self, values: &[f64]) -> Vec<f64> {
        if values.is_empty() {
            return Vec::new();
        }

        let mut smoothed = Vec::with_capacity(values.len());
        smoothed.push(values[0]);

        for &value in values.iter().skip(1) {
            let last_smoothed = *smoothed.last().unwrap();
            let new_smoothed = (self.smoothing_factor as f64) * value
                + (1.0 - (self.smoothing_factor as f64)) * last_smoothed;
            smoothed.push(new_smoothed);
        }

        smoothed
    }

    /// Detect trend in time series
    fn detect_trend(&self, values: &[f64]) -> Option<(f64, f64)> {
        if values.len() < 3 {
            return None;
        }

        // Simple linear regression for trend detection
        let n = values.len() as f64;
        let x_values: Vec<f64> = (0..values.len()).map(|i| i as f64).collect();

        let x_mean = x_values.iter().sum::<f64>() / n;
        let y_mean = values.iter().sum::<f64>() / n;

        let numerator: f64 = x_values
            .iter()
            .zip(values.iter())
            .map(|(x, y)| (x - x_mean) * (y - y_mean))
            .sum();

        let denominator: f64 = x_values.iter().map(|x| (x - x_mean).powi(2)).sum();

        if denominator != 0.0 {
            let slope = numerator / denominator;
            let intercept = y_mean - slope * x_mean;
            Some((slope, intercept))
        } else {
            None
        }
    }
}

impl AggregationStrategy for TimeSeriesAggregationStrategy {
    fn aggregate(&self, feedback: &[ProcessedFeedback]) -> Result<AggregatedFeedback> {
        let filtered_feedback = self.filter_by_time_window(feedback);

        if filtered_feedback.is_empty() {
            return Err(anyhow::anyhow!("No feedback within time window"));
        }

        // Sort by timestamp
        let mut sorted_feedback = filtered_feedback;
        sorted_feedback.sort_by_key(|fb| fb.original_feedback.timestamp);

        let values: Vec<f64> = sorted_feedback.iter().map(|fb| fb.processed_value).collect();

        // Apply exponential smoothing
        let smoothed_values = self.apply_exponential_smoothing(&values);
        let aggregated_value = smoothed_values.last().copied().unwrap_or(0.0);

        // Detect trend if enabled
        let trend_info =
            if self.trend_detection { self.detect_trend(&smoothed_values) } else { None };

        // Calculate confidence based on data consistency and trend strength
        let confidence = if let Some((slope, _)) = trend_info {
            let trend_strength = slope.abs().min(1.0);
            0.8 + 0.2 * (1.0 - trend_strength) as f32
        } else {
            0.8
        };

        // Collect recommended actions
        let recommended_actions: Vec<RecommendedAction> =
            sorted_feedback.iter().filter_map(|fb| fb.recommended_action.clone()).collect();

        Ok(AggregatedFeedback {
            aggregated_value,
            confidence,
            contributing_count: sorted_feedback.len(),
            contributing_feedback_count: sorted_feedback.len(),
            aggregation_method: "time_series".to_string(),
            timestamp: Utc::now(),
            recommended_actions,
        })
    }

    fn name(&self) -> &str {
        "time_series"
    }

    fn is_applicable(&self, feedback: &[ProcessedFeedback]) -> bool {
        !feedback.is_empty()
    }
}

// =============================================================================
// CONSENSUS AGGREGATION STRATEGY
// =============================================================================

impl ConsensusAggregationStrategy {
    /// Create new consensus aggregation strategy
    pub fn new() -> Self {
        Self {
            consensus_threshold: 0.7,
            outlier_detection: true,
            outlier_threshold: 2.0,
            voting_mechanism: VotingMechanism::WeightedVoting,
        }
    }

    /// Detect outliers using statistical methods
    fn detect_outliers(&self, values: &[f64]) -> Vec<bool> {
        if values.len() < 3 || !self.outlier_detection {
            return vec![false; values.len()];
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        values
            .iter()
            .map(|&value| {
                if std_dev > 0.0 {
                    let z_score = (value - mean).abs() / std_dev;
                    z_score as f64 > self.outlier_threshold as f64
                } else {
                    false
                }
            })
            .collect()
    }

    /// Apply voting mechanism
    fn apply_voting(&self, feedback: &[ProcessedFeedback], outliers: &[bool]) -> f64 {
        match self.voting_mechanism {
            VotingMechanism::SimpleMajority => self.simple_majority_vote(feedback, outliers),
            VotingMechanism::WeightedVoting => self.weighted_vote(feedback, outliers),
            VotingMechanism::RankedChoice => self.ranked_choice_vote(feedback, outliers),
            VotingMechanism::FuzzyConsensus => self.fuzzy_consensus_vote(feedback, outliers),
        }
    }

    fn simple_majority_vote(&self, feedback: &[ProcessedFeedback], outliers: &[bool]) -> f64 {
        let valid_values: Vec<f64> = feedback
            .iter()
            .zip(outliers.iter())
            .filter_map(
                |(fb, &is_outlier)| if !is_outlier { Some(fb.processed_value) } else { None },
            )
            .collect();

        if valid_values.is_empty() {
            feedback.iter().map(|f| f.processed_value).sum::<f64>() / feedback.len() as f64
        } else {
            valid_values.iter().sum::<f64>() / valid_values.len() as f64
        }
    }

    fn weighted_vote(&self, feedback: &[ProcessedFeedback], outliers: &[bool]) -> f64 {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (i, fb) in feedback.iter().enumerate() {
            if !outliers[i] {
                let weight = fb.confidence as f64;
                weighted_sum += fb.processed_value * weight;
                total_weight += weight;
            }
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            feedback.iter().map(|f| f.processed_value).sum::<f64>() / feedback.len() as f64
        }
    }

    fn ranked_choice_vote(&self, feedback: &[ProcessedFeedback], outliers: &[bool]) -> f64 {
        // Simplified ranked choice - sort by confidence and take median
        let mut valid_feedback: Vec<_> = feedback
            .iter()
            .zip(outliers.iter())
            .filter_map(|(fb, &is_outlier)| if !is_outlier { Some(fb) } else { None })
            .collect();

        valid_feedback.sort_by(|a, b| {
            b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
        });

        if valid_feedback.is_empty() {
            return 0.0;
        }

        let mid = valid_feedback.len() / 2;
        if valid_feedback.len() % 2 == 0 {
            (valid_feedback[mid - 1].processed_value + valid_feedback[mid].processed_value) / 2.0
        } else {
            valid_feedback[mid].processed_value
        }
    }

    fn fuzzy_consensus_vote(&self, feedback: &[ProcessedFeedback], outliers: &[bool]) -> f64 {
        // Fuzzy consensus using membership functions
        let valid_values: Vec<f64> = feedback
            .iter()
            .zip(outliers.iter())
            .filter_map(
                |(fb, &is_outlier)| if !is_outlier { Some(fb.processed_value) } else { None },
            )
            .collect();

        if valid_values.is_empty() {
            return 0.0;
        }

        let min_val = valid_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = valid_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        if range == 0.0 {
            return min_val;
        }

        // Calculate fuzzy membership and consensus
        let mut consensus_sum = 0.0;
        let mut membership_sum = 0.0;

        for &value in &valid_values {
            let normalized = (value - min_val) / range;
            let membership = 1.0 - (normalized - 0.5).abs(); // Peak at center
            consensus_sum += value * membership;
            membership_sum += membership;
        }

        if membership_sum > 0.0 {
            consensus_sum / membership_sum
        } else {
            valid_values.iter().sum::<f64>() / valid_values.len() as f64
        }
    }
}

impl AggregationStrategy for ConsensusAggregationStrategy {
    fn aggregate(&self, feedback: &[ProcessedFeedback]) -> Result<AggregatedFeedback> {
        if feedback.is_empty() {
            return Err(anyhow::anyhow!("No feedback to aggregate"));
        }

        let values: Vec<f64> = feedback.iter().map(|f| f.processed_value).collect();
        let outliers = self.detect_outliers(&values);

        let aggregated_value = self.apply_voting(feedback, &outliers);

        // Calculate consensus level
        let valid_count = outliers.iter().filter(|&&is_outlier| !is_outlier).count();
        let consensus_level = valid_count as f32 / feedback.len() as f32;

        let confidence = if consensus_level >= self.consensus_threshold {
            0.9 * consensus_level
        } else {
            0.5 * consensus_level
        };

        // Collect actions from non-outlier feedback
        let recommended_actions: Vec<RecommendedAction> = feedback
            .iter()
            .zip(outliers.iter())
            .filter_map(
                |(fb, &is_outlier)| {
                    if !is_outlier {
                        fb.recommended_action.clone()
                    } else {
                        None
                    }
                },
            )
            .collect();

        Ok(AggregatedFeedback {
            aggregated_value,
            confidence,
            contributing_count: valid_count,
            contributing_feedback_count: valid_count,
            aggregation_method: "consensus".to_string(),
            timestamp: Utc::now(),
            recommended_actions,
        })
    }

    fn name(&self) -> &str {
        "consensus"
    }

    fn is_applicable(&self, feedback: &[ProcessedFeedback]) -> bool {
        feedback.len() >= 3 // Need at least 3 items for meaningful consensus
    }
}

// =============================================================================
// MULTI-OBJECTIVE AGGREGATION STRATEGY
// =============================================================================

impl MultiObjectiveAggregationStrategy {
    /// Create new multi-objective aggregation strategy
    pub fn new(objectives: Vec<OptimizationObjective>, weights: Vec<f32>) -> Self {
        Self {
            objectives,
            weights,
            pareto_optimization: false,
            constraints: Vec::new(),
        }
    }

    /// Evaluate objectives for feedback
    fn evaluate_objectives(&self, feedback: &[ProcessedFeedback]) -> Vec<f64> {
        let mut objective_values = Vec::new();

        for objective in &self.objectives {
            let value = match objective.objective_type {
                ObjectiveType::Performance => {
                    // Average performance value
                    feedback.iter().map(|f| f.processed_value).sum::<f64>() / feedback.len() as f64
                },
                ObjectiveType::ResourceEfficiency => {
                    // Calculate efficiency based on confidence and value
                    let efficiency_sum: f64 =
                        feedback.iter().map(|f| f.processed_value * f.confidence as f64).sum();
                    efficiency_sum / feedback.len() as f64
                },
                ObjectiveType::Stability => {
                    // Calculate stability as inverse of variance
                    let values: Vec<f64> = feedback.iter().map(|f| f.processed_value).collect();
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                        / values.len() as f64;
                    1.0 / (1.0 + variance) // Higher stability for lower variance
                },
                ObjectiveType::Reliability => {
                    // Average confidence as reliability measure
                    feedback.iter().map(|f| f.confidence as f64).sum::<f64>()
                        / feedback.len() as f64
                },
                ObjectiveType::Custom(_) => {
                    // Default value for custom objectives
                    0.5
                },
            };

            objective_values.push(value);
        }

        objective_values
    }

    /// Apply constraints
    fn check_constraints(&self, objective_values: &[f64]) -> bool {
        for constraint in &self.constraints {
            let constraint_value = match constraint.constraint_type {
                ConstraintType::Performance => objective_values.get(0).copied().unwrap_or(0.0),
                ConstraintType::Resource => objective_values.get(1).copied().unwrap_or(0.0),
                ConstraintType::Safety => objective_values
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .copied()
                    .unwrap_or(0.0),
                ConstraintType::Policy => {
                    objective_values.iter().sum::<f64>() / objective_values.len() as f64
                },
                ConstraintType::Custom(_) => 0.5,
            };

            let constraint_satisfied = match constraint.operator {
                ConstraintOperator::LessThan => constraint_value < constraint.value,
                ConstraintOperator::LessThanOrEqual => constraint_value <= constraint.value,
                ConstraintOperator::Equal => (constraint_value - constraint.value).abs() < 0.001,
                ConstraintOperator::GreaterThanOrEqual => constraint_value >= constraint.value,
                ConstraintOperator::GreaterThan => constraint_value > constraint.value,
                ConstraintOperator::NotEqual => {
                    (constraint_value - constraint.value).abs() >= 0.001
                },
            };

            if !constraint_satisfied {
                return false;
            }
        }

        true
    }

    /// Calculate weighted objective value
    fn calculate_weighted_value(&self, objective_values: &[f64]) -> f64 {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for (i, (&obj_value, objective)) in
            objective_values.iter().zip(&self.objectives).enumerate()
        {
            let weight = self.weights.get(i).copied().unwrap_or(1.0) as f64;

            let normalized_value = match objective.direction {
                OptimizationDirection::Maximize => obj_value,
                OptimizationDirection::Minimize => 1.0 - obj_value.clamp(0.0, 1.0),
            };

            weighted_sum += normalized_value * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.0
        }
    }
}

impl AggregationStrategy for MultiObjectiveAggregationStrategy {
    fn aggregate(&self, feedback: &[ProcessedFeedback]) -> Result<AggregatedFeedback> {
        if feedback.is_empty() {
            return Err(anyhow::anyhow!("No feedback to aggregate"));
        }

        let objective_values = self.evaluate_objectives(feedback);

        // Check constraints
        if !self.check_constraints(&objective_values) {
            return Err(anyhow::anyhow!("Constraints not satisfied"));
        }

        let aggregated_value = self.calculate_weighted_value(&objective_values);

        // Calculate confidence based on objective achievement
        let confidence = objective_values.iter().sum::<f64>() / objective_values.len() as f64;

        // Collect all recommended actions
        let recommended_actions: Vec<RecommendedAction> =
            feedback.iter().filter_map(|f| f.recommended_action.clone()).collect();

        Ok(AggregatedFeedback {
            aggregated_value,
            confidence: confidence as f32,
            contributing_count: feedback.len(),
            contributing_feedback_count: feedback.len(),
            aggregation_method: "multi_objective".to_string(),
            timestamp: Utc::now(),
            recommended_actions,
        })
    }

    fn name(&self) -> &str {
        "multi_objective"
    }

    fn is_applicable(&self, feedback: &[ProcessedFeedback]) -> bool {
        !feedback.is_empty() && !self.objectives.is_empty()
    }
}
