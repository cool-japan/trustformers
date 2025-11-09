//! Gradient Conflict Analysis Between Layers
//!
//! This module provides comprehensive analysis of gradient conflicts between layers,
//! including conflict detection, classification, and mitigation strategy generation.

use super::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Analysis of gradient conflicts between layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientConflictAnalysis {
    pub total_conflicts: usize,
    pub conflicts: Vec<GradientConflict>,
    pub overall_conflict_level: ConflictLevel,
    pub mitigation_strategies: Vec<ConflictMitigationStrategy>,
}

/// Individual gradient conflict between two layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientConflict {
    pub layer1: String,
    pub layer2: String,
    pub conflict_score: f64,
    pub conflict_type: ConflictType,
    pub recommendations: Vec<String>,
}

/// Types of gradient conflicts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictType {
    None,
    Mild,
    Moderate,
    Severe,
}

/// Overall level of conflicts in the network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Strategy for mitigating gradient conflicts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictMitigationStrategy {
    pub strategy_name: String,
    pub description: String,
    pub target_conflicts: Vec<String>,
    pub effectiveness: f64,
    pub implementation_complexity: MitigationComplexity,
    pub expected_outcome: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MitigationComplexity {
    Simple,
    Moderate,
    Complex,
    RequiresArchitectureChange,
}

/// Gradient conflict analyzer
#[derive(Debug)]
pub struct GradientConflictAnalyzer {
    conflict_threshold: f64,
    analysis_window: usize,
}

impl Default for GradientConflictAnalyzer {
    fn default() -> Self {
        Self {
            conflict_threshold: 0.5,
            analysis_window: 10,
        }
    }
}

impl GradientConflictAnalyzer {
    pub fn new(threshold: f64, window: usize) -> Self {
        Self {
            conflict_threshold: threshold,
            analysis_window: window,
        }
    }

    pub fn analyze_conflicts(
        &self,
        gradient_histories: &HashMap<String, GradientHistory>,
    ) -> GradientConflictAnalysis {
        let mut conflicts = Vec::new();
        let mut layer_gradients: Vec<(String, Vec<f64>)> = Vec::new();

        // Collect recent gradient norms for each layer
        for (layer_name, history) in gradient_histories {
            if let Some(recent_gradients) = self.get_recent_gradients(history, self.analysis_window)
            {
                layer_gradients.push((layer_name.clone(), recent_gradients));
            }
        }

        // Analyze conflicts between pairs of layers
        for i in 0..layer_gradients.len() {
            for j in (i + 1)..layer_gradients.len() {
                let (layer1_name, layer1_grads) = &layer_gradients[i];
                let (layer2_name, layer2_grads) = &layer_gradients[j];

                let conflict_score = self.compute_gradient_conflict(layer1_grads, layer2_grads);

                if conflict_score > self.conflict_threshold {
                    conflicts.push(GradientConflict {
                        layer1: layer1_name.clone(),
                        layer2: layer2_name.clone(),
                        conflict_score,
                        conflict_type: self.classify_conflict_type(conflict_score),
                        recommendations: self.get_conflict_recommendations(conflict_score),
                    });
                }
            }
        }

        let overall_conflict_level = self.compute_overall_conflict_level(&conflicts);
        let mitigation_strategies = self.generate_conflict_mitigation_strategies(&conflicts);

        GradientConflictAnalysis {
            total_conflicts: conflicts.len(),
            conflicts,
            overall_conflict_level,
            mitigation_strategies,
        }
    }

    fn get_recent_gradients(&self, history: &GradientHistory, count: usize) -> Option<Vec<f64>> {
        if history.gradient_norms.len() < count {
            return None;
        }

        Some(history.gradient_norms.iter().rev().take(count).cloned().collect())
    }

    fn compute_gradient_conflict(&self, grads1: &[f64], grads2: &[f64]) -> f64 {
        if grads1.len() != grads2.len() || grads1.is_empty() {
            return 0.0;
        }

        // Compute cosine similarity between gradient patterns
        let dot_product: f64 = grads1.iter().zip(grads2.iter()).map(|(a, b)| a * b).sum();
        let norm1: f64 = grads1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = grads2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return 1.0; // Maximum conflict if one has zero gradients
        }

        let cosine_similarity = dot_product / (norm1 * norm2);

        // Convert to conflict score (0 = no conflict, 1 = maximum conflict)
        (1.0 - cosine_similarity.abs()).max(0.0)
    }

    fn classify_conflict_type(&self, conflict_score: f64) -> ConflictType {
        match conflict_score {
            x if x > 0.8 => ConflictType::Severe,
            x if x > 0.6 => ConflictType::Moderate,
            x if x > 0.3 => ConflictType::Mild,
            _ => ConflictType::None,
        }
    }

    fn get_conflict_recommendations(&self, conflict_score: f64) -> Vec<String> {
        let mut recommendations = Vec::new();

        match conflict_score {
            x if x > 0.8 => {
                recommendations.push("Critical gradient conflict detected".to_string());
                recommendations.push("Consider gradient clipping or normalization".to_string());
                recommendations.push("Review learning rates for affected layers".to_string());
                recommendations.push("Consider architectural changes".to_string());
            },
            x if x > 0.6 => {
                recommendations.push("Moderate gradient conflict detected".to_string());
                recommendations.push("Consider adjusting learning rates".to_string());
                recommendations.push("Monitor gradient flow patterns".to_string());
            },
            x if x > 0.3 => {
                recommendations.push("Mild gradient conflict detected".to_string());
                recommendations.push("Continue monitoring conflict patterns".to_string());
            },
            _ => {
                recommendations.push("No significant conflict detected".to_string());
            },
        }

        recommendations
    }

    fn compute_overall_conflict_level(&self, conflicts: &[GradientConflict]) -> ConflictLevel {
        if conflicts.is_empty() {
            return ConflictLevel::Low;
        }

        let severe_conflicts = conflicts
            .iter()
            .filter(|c| matches!(c.conflict_type, ConflictType::Severe))
            .count();
        let moderate_conflicts = conflicts
            .iter()
            .filter(|c| matches!(c.conflict_type, ConflictType::Moderate))
            .count();

        let total_layers_with_conflicts = self.count_layers_with_conflicts(conflicts);

        if severe_conflicts > 0 || total_layers_with_conflicts > 10 {
            ConflictLevel::Critical
        } else if moderate_conflicts > 3 || total_layers_with_conflicts > 5 {
            ConflictLevel::High
        } else if moderate_conflicts > 0 || total_layers_with_conflicts > 2 {
            ConflictLevel::Medium
        } else {
            ConflictLevel::Low
        }
    }

    fn count_layers_with_conflicts(&self, conflicts: &[GradientConflict]) -> usize {
        let mut layers = std::collections::HashSet::new();
        for conflict in conflicts {
            layers.insert(&conflict.layer1);
            layers.insert(&conflict.layer2);
        }
        layers.len()
    }

    fn generate_conflict_mitigation_strategies(
        &self,
        conflicts: &[GradientConflict],
    ) -> Vec<ConflictMitigationStrategy> {
        let mut strategies = Vec::new();

        if conflicts.is_empty() {
            return strategies;
        }

        // Gradient clipping strategy
        let severe_conflicts = conflicts
            .iter()
            .filter(|c| matches!(c.conflict_type, ConflictType::Severe))
            .count();
        if severe_conflicts > 0 {
            strategies.push(ConflictMitigationStrategy {
                strategy_name: "Gradient Clipping".to_string(),
                description: "Apply gradient clipping to prevent extreme gradient values"
                    .to_string(),
                target_conflicts: conflicts
                    .iter()
                    .filter(|c| matches!(c.conflict_type, ConflictType::Severe))
                    .map(|c| format!("{}-{}", c.layer1, c.layer2))
                    .collect(),
                effectiveness: 0.8,
                implementation_complexity: MitigationComplexity::Simple,
                expected_outcome: "Reduced gradient magnitude conflicts".to_string(),
            });
        }

        // Learning rate adjustment strategy
        if conflicts.len() > 2 {
            strategies.push(ConflictMitigationStrategy {
                strategy_name: "Adaptive Learning Rates".to_string(),
                description: "Use layer-specific learning rates to balance gradient flows"
                    .to_string(),
                target_conflicts: conflicts
                    .iter()
                    .map(|c| format!("{}-{}", c.layer1, c.layer2))
                    .collect(),
                effectiveness: 0.7,
                implementation_complexity: MitigationComplexity::Moderate,
                expected_outcome: "Better gradient balance across layers".to_string(),
            });
        }

        // Normalization strategy
        let high_conflict_count = conflicts
            .iter()
            .filter(|c| {
                matches!(
                    c.conflict_type,
                    ConflictType::Severe | ConflictType::Moderate
                )
            })
            .count();

        if high_conflict_count > 1 {
            strategies.push(ConflictMitigationStrategy {
                strategy_name: "Gradient Normalization".to_string(),
                description: "Normalize gradients to reduce scale conflicts".to_string(),
                target_conflicts: conflicts
                    .iter()
                    .filter(|c| {
                        matches!(
                            c.conflict_type,
                            ConflictType::Severe | ConflictType::Moderate
                        )
                    })
                    .map(|c| format!("{}-{}", c.layer1, c.layer2))
                    .collect(),
                effectiveness: 0.6,
                implementation_complexity: MitigationComplexity::Simple,
                expected_outcome: "More consistent gradient scales".to_string(),
            });
        }

        // Architecture modification strategy for critical conflicts
        if severe_conflicts > 3 {
            strategies.push(ConflictMitigationStrategy {
                strategy_name: "Architecture Modification".to_string(),
                description: "Consider residual connections or attention mechanisms".to_string(),
                target_conflicts: conflicts
                    .iter()
                    .filter(|c| matches!(c.conflict_type, ConflictType::Severe))
                    .map(|c| format!("{}-{}", c.layer1, c.layer2))
                    .collect(),
                effectiveness: 0.9,
                implementation_complexity: MitigationComplexity::RequiresArchitectureChange,
                expected_outcome: "Fundamental improvement in gradient flow".to_string(),
            });
        }

        strategies
    }

    pub fn generate_conflict_report(&self, analysis: &GradientConflictAnalysis) -> ConflictReport {
        let mut layer_conflict_counts = HashMap::new();
        #[allow(dead_code)]
        #[allow(unused_assignments)]
        let mut most_problematic_pairs = Vec::new();

        // Count conflicts per layer
        for conflict in &analysis.conflicts {
            *layer_conflict_counts.entry(conflict.layer1.clone()).or_insert(0) += 1;
            *layer_conflict_counts.entry(conflict.layer2.clone()).or_insert(0) += 1;
        }

        // Find most problematic layer pairs
        let mut sorted_conflicts = analysis.conflicts.clone();
        sorted_conflicts.sort_by(|a, b| b.conflict_score.partial_cmp(&a.conflict_score).unwrap());
        most_problematic_pairs = sorted_conflicts.into_iter().take(5).collect();

        // Find most problematic layers
        let mut layer_scores: Vec<(String, usize)> = layer_conflict_counts.into_iter().collect();
        layer_scores.sort_by(|a, b| b.1.cmp(&a.1));
        let most_problematic_layers: Vec<String> =
            layer_scores.into_iter().take(5).map(|(name, _)| name).collect();

        ConflictReport {
            total_conflicts: analysis.total_conflicts,
            overall_level: analysis.overall_conflict_level.clone(),
            most_problematic_layers,
            most_problematic_pairs,
            recommended_strategies: analysis.mitigation_strategies.clone(),
            summary: self.generate_conflict_summary(analysis),
        }
    }

    fn generate_conflict_summary(&self, analysis: &GradientConflictAnalysis) -> String {
        match analysis.overall_conflict_level {
            ConflictLevel::Critical => {
                format!("Critical gradient conflicts detected ({} total). Immediate action required to stabilize training.", analysis.total_conflicts)
            },
            ConflictLevel::High => {
                format!("High level of gradient conflicts ({} total). Consider implementing mitigation strategies.", analysis.total_conflicts)
            },
            ConflictLevel::Medium => {
                format!("Moderate gradient conflicts detected ({} total). Monitor and consider optimization.", analysis.total_conflicts)
            },
            ConflictLevel::Low => {
                format!(
                    "Low conflict level ({} total). Gradient flow appears stable.",
                    analysis.total_conflicts
                )
            },
        }
    }
}

/// Comprehensive conflict analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictReport {
    pub total_conflicts: usize,
    pub overall_level: ConflictLevel,
    pub most_problematic_layers: Vec<String>,
    pub most_problematic_pairs: Vec<GradientConflict>,
    pub recommended_strategies: Vec<ConflictMitigationStrategy>,
    pub summary: String,
}
