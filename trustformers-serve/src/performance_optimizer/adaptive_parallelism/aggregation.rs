//! Feedback Aggregation System
//!
//! This module provides feedback aggregation capabilities including multiple
//! aggregation strategies such as weighted average, consensus, and confidence-weighted
//! approaches. The FeedbackAggregator coordinates these strategies to combine
//! feedback from multiple sources into coherent recommendations.

use anyhow::Result;
use chrono::Utc;
use parking_lot::Mutex;
use std::{collections::HashMap, sync::Arc};

use crate::performance_optimizer::types::*;

// Re-export types needed by other modules
pub use crate::performance_optimizer::types::{AggregationStrategy, FeedbackAggregator};

// =============================================================================
// FEEDBACK AGGREGATOR IMPLEMENTATION
// =============================================================================

impl FeedbackAggregator {
    /// Create a new feedback aggregator
    pub async fn new() -> Result<Self> {
        let mut strategies: Vec<Box<dyn AggregationStrategy + Send + Sync>> = Vec::new();

        // Add default aggregation strategies
        strategies.push(Box::new(WeightedAverageAggregation::new()));
        strategies.push(Box::new(ConsensusAggregation::new()));
        strategies.push(Box::new(ConfidenceWeightedAggregation::new()));

        Ok(Self {
            strategies: Arc::new(Mutex::new(strategies)),
            aggregated_cache: Arc::new(Mutex::new(HashMap::new())),
            aggregation_history: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Aggregate feedback using multiple strategies
    pub async fn aggregate_feedback(&self, feedback_items: &[ProcessedFeedback]) -> Result<()> {
        if feedback_items.is_empty() {
            return Ok(());
        }

        let strategies = self.strategies.lock();
        let mut aggregated_results = Vec::new();

        // Apply each strategy
        for strategy in strategies.iter() {
            match strategy.aggregate(feedback_items) {
                Ok(result) => aggregated_results.push(result),
                Err(e) => log::warn!("Aggregation strategy {} failed: {}", strategy.name(), e),
            }
        }

        // Store aggregation results
        if !aggregated_results.is_empty() {
            let strategy_names: Vec<String> =
                strategies.iter().map(|s| s.name().to_string()).collect();

            let record = AggregationRecord {
                timestamp: Utc::now(),
                input_count: feedback_items.len(),
                strategy: strategy_names.first().cloned().unwrap_or_else(|| "unknown".to_string()),
                result: aggregated_results.first().cloned().unwrap_or_default(),
                duration: std::time::Duration::from_millis(0),
                input_feedback_count: feedback_items.len(),
                strategies_used: strategy_names,
                aggregated_results: aggregated_results.clone(),
            };

            self.aggregation_history.lock().push(record);
        }

        Ok(())
    }

    /// Get recent aggregated feedback
    pub async fn get_recent_aggregated_feedback(&self) -> Result<Vec<AggregatedFeedback>> {
        let history = self.aggregation_history.lock();
        let recent_results = history
            .iter()
            .rev()
            .take(10)
            .flat_map(|record| record.aggregated_results.clone())
            .collect();

        Ok(recent_results)
    }
}

// =============================================================================
// WEIGHTED AVERAGE AGGREGATION STRATEGY
// =============================================================================

/// Weighted average aggregation strategy
pub struct WeightedAverageAggregation {
    name: String,
}

impl Default for WeightedAverageAggregation {
    fn default() -> Self {
        Self::new()
    }
}

impl WeightedAverageAggregation {
    pub fn new() -> Self {
        Self {
            name: "weighted_average".to_string(),
        }
    }
}

impl AggregationStrategy for WeightedAverageAggregation {
    fn aggregate(&self, feedback_items: &[ProcessedFeedback]) -> Result<AggregatedFeedback> {
        if feedback_items.is_empty() {
            return Err(anyhow::anyhow!("Cannot aggregate empty feedback"));
        }

        // Calculate weighted average based on confidence
        let total_weight: f32 = feedback_items.iter().map(|f| f.confidence).sum();
        let weighted_value: f32 = feedback_items
            .iter()
            .map(|f| f.processed_value * (f.confidence as f64))
            .sum::<f64>() as f32
            / total_weight;

        Ok(AggregatedFeedback {
            aggregated_value: weighted_value as f64,
            aggregation_method: self.name.clone(),
            confidence: total_weight / feedback_items.len() as f32,
            contributing_count: feedback_items.len(),
            contributing_feedback_count: feedback_items.len(),
            timestamp: Utc::now(),
            recommended_actions: feedback_items
                .iter()
                .filter_map(|f| f.recommended_action.clone())
                .collect(),
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_applicable(&self, feedback: &[ProcessedFeedback]) -> bool {
        !feedback.is_empty()
    }
}

// =============================================================================
// CONSENSUS AGGREGATION STRATEGY
// =============================================================================

/// Consensus aggregation strategy
pub struct ConsensusAggregation {
    name: String,
}

impl Default for ConsensusAggregation {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsensusAggregation {
    pub fn new() -> Self {
        Self {
            name: "consensus".to_string(),
        }
    }
}

impl AggregationStrategy for ConsensusAggregation {
    fn aggregate(&self, feedback_items: &[ProcessedFeedback]) -> Result<AggregatedFeedback> {
        if feedback_items.is_empty() {
            return Err(anyhow::anyhow!("Cannot aggregate empty feedback"));
        }

        // Find consensus by taking median value
        let mut values: Vec<f32> =
            feedback_items.iter().map(|f| f.processed_value as f32).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if values.len() % 2 == 0 {
            (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
        } else {
            values[values.len() / 2]
        };

        // Calculate consensus confidence based on variance
        let variance =
            values.iter().map(|v| (v - median).powi(2)).sum::<f32>() / values.len() as f32;
        let consensus_confidence = (1.0 / (1.0 + variance)).min(0.9);

        Ok(AggregatedFeedback {
            aggregated_value: median as f64,
            aggregation_method: self.name.clone(),
            confidence: consensus_confidence,
            contributing_count: feedback_items.len(),
            contributing_feedback_count: feedback_items.len(),
            timestamp: Utc::now(),
            recommended_actions: feedback_items
                .iter()
                .filter_map(|f| f.recommended_action.clone())
                .collect(),
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_applicable(&self, feedback: &[ProcessedFeedback]) -> bool {
        feedback.len() >= 3 // Consensus needs at least 3 data points
    }
}

// =============================================================================
// CONFIDENCE WEIGHTED AGGREGATION STRATEGY
// =============================================================================

/// Confidence weighted aggregation strategy
pub struct ConfidenceWeightedAggregation {
    name: String,
}

impl Default for ConfidenceWeightedAggregation {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfidenceWeightedAggregation {
    pub fn new() -> Self {
        Self {
            name: "confidence_weighted".to_string(),
        }
    }
}

impl AggregationStrategy for ConfidenceWeightedAggregation {
    fn aggregate(&self, feedback_items: &[ProcessedFeedback]) -> Result<AggregatedFeedback> {
        if feedback_items.is_empty() {
            return Err(anyhow::anyhow!("Cannot aggregate empty feedback"));
        }

        // Weight by confidence squared to emphasize high-confidence feedback
        let weights: Vec<f32> = feedback_items.iter().map(|f| f.confidence.powi(2)).collect();
        let total_weight: f32 = weights.iter().sum();

        let weighted_value: f32 = feedback_items
            .iter()
            .zip(weights.iter())
            .map(|(f, w)| f.processed_value * (*w as f64))
            .sum::<f64>() as f32
            / total_weight;

        let confidence = (total_weight / feedback_items.len() as f32).sqrt().min(0.95);

        Ok(AggregatedFeedback {
            aggregated_value: weighted_value as f64,
            aggregation_method: self.name.clone(),
            confidence,
            contributing_count: feedback_items.len(),
            contributing_feedback_count: feedback_items.len(),
            timestamp: Utc::now(),
            recommended_actions: feedback_items
                .iter()
                .filter_map(|f| f.recommended_action.clone())
                .collect(),
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_applicable(&self, feedback: &[ProcessedFeedback]) -> bool {
        !feedback.is_empty() && feedback.iter().any(|f| f.confidence > 0.5)
    }
}
