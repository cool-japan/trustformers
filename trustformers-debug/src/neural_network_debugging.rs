//! Advanced Neural Network Debugging Utilities
//!
//! This module provides specialized debugging utilities for modern neural network architectures,
//! with particular focus on transformer models, attention mechanisms, and large-scale training scenarios.

use anyhow::Result;
use scirs2_core::ndarray::*; // SciRS2 Integration Policy - was: use ndarray::{Array, ArrayD, IxDyn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Advanced attention mechanism debugger for transformer architectures
#[derive(Debug)]
pub struct AttentionDebugger {
    pub config: AttentionDebugConfig,
    attention_maps: Vec<AttentionMap>,
    head_analysis: HashMap<usize, AttentionHeadAnalysis>,
}

/// Configuration for attention debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionDebugConfig {
    pub enable_attention_visualization: bool,
    pub enable_head_analysis: bool,
    pub enable_pattern_detection: bool,
    pub attention_threshold: f32,
    pub max_heads_to_analyze: usize,
}

impl Default for AttentionDebugConfig {
    fn default() -> Self {
        Self {
            enable_attention_visualization: true,
            enable_head_analysis: true,
            enable_pattern_detection: true,
            attention_threshold: 0.01,
            max_heads_to_analyze: 16,
        }
    }
}

/// Attention map for a specific layer and head
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionMap {
    pub layer_index: usize,
    pub head_index: usize,
    pub sequence_length: usize,
    pub attention_weights: Vec<Vec<f32>>,
    pub attention_pattern: AttentionPattern,
    pub attention_entropy: f32,
    pub sparsity_ratio: f32,
}

/// Analysis of individual attention head behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionHeadAnalysis {
    pub head_id: usize,
    pub layer_id: usize,
    pub specialization_type: HeadSpecializationType,
    pub attention_distribution: AttentionDistribution,
    pub redundancy_score: f32,
    pub importance_score: f32,
    pub patterns_detected: Vec<AttentionPattern>,
}

/// Types of attention head specializations commonly found in transformers
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HeadSpecializationType {
    LocalSyntax,  // Focuses on local syntactic patterns
    LongRange,    // Captures long-range dependencies
    Positional,   // Primarily position-based attention
    ContentBased, // Content-driven attention patterns
    Copying,      // Copy mechanisms
    Delimiter,    // Focuses on delimiters and boundaries
    Mixed,        // Mixed functionality
    Redundant,    // Highly redundant with other heads
}

/// Attention pattern types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttentionPattern {
    Diagonal,     // Attention along diagonal (local patterns)
    Block,        // Block-structured attention
    Sparse,       // Sparse attention patterns
    Uniform,      // Uniform attention distribution
    Concentrated, // Highly concentrated attention
    Strided,      // Strided patterns
    Random,       // Random/chaotic patterns
}

/// Distribution characteristics of attention weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionDistribution {
    pub mean_attention: f32,
    pub std_attention: f32,
    pub max_attention: f32,
    pub min_attention: f32,
    pub entropy: f32,
    pub effective_context_length: f32,
}

impl AttentionDebugger {
    /// Create a new attention debugger
    pub fn new(config: AttentionDebugConfig) -> Self {
        Self {
            config,
            attention_maps: Vec::new(),
            head_analysis: HashMap::new(),
        }
    }

    /// Analyze attention weights for a transformer layer
    pub fn analyze_attention_layer(
        &mut self,
        layer_index: usize,
        attention_weights: &[ArrayD<f32>], // One array per head
    ) -> Result<LayerAttentionAnalysis> {
        let mut head_analyses = Vec::new();
        let mut attention_maps = Vec::new();

        for (head_index, weights) in attention_weights.iter().enumerate() {
            if head_index >= self.config.max_heads_to_analyze {
                break;
            }

            // Create attention map
            let attention_map = self.create_attention_map(layer_index, head_index, weights)?;
            attention_maps.push(attention_map.clone());
            self.attention_maps.push(attention_map);

            // Analyze head behavior
            let head_analysis = self.analyze_attention_head(layer_index, head_index, weights)?;
            head_analyses.push(head_analysis.clone());
            self.head_analysis.insert(head_index, head_analysis);
        }

        let layer_diversity_score = self.compute_layer_diversity(&head_analyses);
        let redundancy_analysis = self.analyze_head_redundancy(&head_analyses);

        Ok(LayerAttentionAnalysis {
            layer_index,
            num_heads: attention_weights.len(),
            head_analyses,
            attention_maps,
            layer_diversity_score,
            redundancy_analysis,
        })
    }

    /// Create attention map from weights
    fn create_attention_map(
        &self,
        layer_index: usize,
        head_index: usize,
        weights: &ArrayD<f32>,
    ) -> Result<AttentionMap> {
        let shape = weights.shape();
        if shape.len() != 2 {
            return Err(anyhow::anyhow!(
                "Expected 2D attention weights, got {}D",
                shape.len()
            ));
        }

        let seq_len = shape[0];
        let attention_weights: Vec<Vec<f32>> =
            (0..seq_len).map(|i| (0..shape[1]).map(|j| weights[[i, j]]).collect()).collect();

        let pattern = self.detect_attention_pattern(&attention_weights);
        let entropy = self.compute_attention_entropy(&attention_weights);
        let sparsity = self.compute_sparsity_ratio(&attention_weights);

        Ok(AttentionMap {
            layer_index,
            head_index,
            sequence_length: seq_len,
            attention_weights,
            attention_pattern: pattern,
            attention_entropy: entropy,
            sparsity_ratio: sparsity,
        })
    }

    /// Analyze individual attention head
    fn analyze_attention_head(
        &self,
        layer_index: usize,
        head_index: usize,
        weights: &ArrayD<f32>,
    ) -> Result<AttentionHeadAnalysis> {
        let specialization = self.classify_head_specialization(weights)?;
        let distribution = self.compute_attention_distribution(weights)?;
        let patterns = vec![self.detect_attention_pattern_from_weights(weights)?];

        Ok(AttentionHeadAnalysis {
            head_id: head_index,
            layer_id: layer_index,
            specialization_type: specialization,
            attention_distribution: distribution,
            redundancy_score: 0.0, // Will be computed later
            importance_score: self.compute_head_importance(weights)?,
            patterns_detected: patterns,
        })
    }

    /// Detect attention pattern from weights matrix
    fn detect_attention_pattern(&self, weights: &[Vec<f32>]) -> AttentionPattern {
        let seq_len = weights.len();
        if seq_len == 0 {
            return AttentionPattern::Random;
        }

        // Check for diagonal pattern
        let diagonal_strength = self.measure_diagonal_strength(weights);
        if diagonal_strength > 0.7 {
            return AttentionPattern::Diagonal;
        }

        // Check for sparse pattern
        let sparsity = self.compute_sparsity_ratio(weights);
        if sparsity > 0.8 {
            return AttentionPattern::Sparse;
        }

        // Check for uniform pattern
        let uniformity = self.measure_uniformity(weights);
        if uniformity > 0.8 {
            return AttentionPattern::Uniform;
        }

        // Check for block pattern
        if self.has_block_structure(weights) {
            return AttentionPattern::Block;
        }

        AttentionPattern::Random
    }

    /// Measure diagonal strength in attention pattern
    fn measure_diagonal_strength(&self, weights: &[Vec<f32>]) -> f32 {
        let seq_len = weights.len();
        if seq_len == 0 {
            return 0.0;
        }

        let mut diagonal_sum = 0.0;
        let mut total_sum = 0.0;
        let window_size = 3; // Look at diagonal +/- window

        for i in 0..seq_len {
            for j in 0..weights[i].len() {
                let weight = weights[i][j];
                total_sum += weight;

                if (i as i32 - j as i32).abs() <= window_size {
                    diagonal_sum += weight;
                }
            }
        }

        if total_sum > 0.0 {
            diagonal_sum / total_sum
        } else {
            0.0
        }
    }

    /// Measure uniformity of attention distribution
    fn measure_uniformity(&self, weights: &[Vec<f32>]) -> f32 {
        let seq_len = weights.len();
        if seq_len == 0 {
            return 0.0;
        }

        let expected_weight = 1.0 / seq_len as f32;
        let mut deviation_sum = 0.0;
        let mut count = 0;

        for row in weights {
            for &weight in row {
                deviation_sum += (weight - expected_weight).abs();
                count += 1;
            }
        }

        if count > 0 {
            1.0 - (deviation_sum / count as f32)
        } else {
            0.0
        }
    }

    /// Check if attention has block structure
    fn has_block_structure(&self, weights: &[Vec<f32>]) -> bool {
        // Simplified block detection - look for concentrated regions
        let seq_len = weights.len();
        if seq_len < 4 {
            return false;
        }

        let block_size = seq_len / 4;
        let mut block_concentrations = Vec::new();

        for block_start in (0..seq_len).step_by(block_size) {
            let block_end = (block_start + block_size).min(seq_len);
            let mut block_sum = 0.0;
            let mut block_count = 0;

            for i in block_start..block_end {
                for j in block_start..(block_end.min(weights[i].len())) {
                    block_sum += weights[i][j];
                    block_count += 1;
                }
            }

            if block_count > 0 {
                block_concentrations.push(block_sum / block_count as f32);
            }
        }

        // Check if some blocks have significantly higher concentration
        if block_concentrations.len() < 2 {
            return false;
        }

        let max_concentration = block_concentrations.iter().cloned().fold(0.0f32, f32::max);
        let avg_concentration =
            block_concentrations.iter().sum::<f32>() / block_concentrations.len() as f32;

        max_concentration > avg_concentration * 2.0
    }

    /// Classify attention head specialization
    fn classify_head_specialization(
        &self,
        weights: &ArrayD<f32>,
    ) -> Result<HeadSpecializationType> {
        let shape = weights.shape();
        if shape.len() != 2 {
            return Ok(HeadSpecializationType::Mixed);
        }

        let seq_len = shape[0];

        // Convert to 2D vec for analysis
        let weights_2d: Vec<Vec<f32>> =
            (0..seq_len).map(|i| (0..shape[1]).map(|j| weights[[i, j]]).collect()).collect();

        // Analyze patterns to determine specialization
        let diagonal_strength = self.measure_diagonal_strength(&weights_2d);
        let long_range_strength = self.measure_long_range_attention(&weights_2d);
        let positional_bias = self.measure_positional_bias(&weights_2d);

        Ok(if diagonal_strength > 0.7 {
            HeadSpecializationType::LocalSyntax
        } else if long_range_strength > 0.6 {
            HeadSpecializationType::LongRange
        } else if positional_bias > 0.8 {
            HeadSpecializationType::Positional
        } else {
            HeadSpecializationType::ContentBased
        })
    }

    /// Measure long-range attention strength
    fn measure_long_range_attention(&self, weights: &[Vec<f32>]) -> f32 {
        let seq_len = weights.len();
        if seq_len < 4 {
            return 0.0;
        }

        let mut long_range_sum = 0.0;
        let mut total_sum = 0.0;
        let long_range_threshold = seq_len / 4; // Consider distances > 1/4 sequence length as long-range

        for i in 0..seq_len {
            for j in 0..weights[i].len() {
                let weight = weights[i][j];
                total_sum += weight;

                if (i as i32 - j as i32).abs() > long_range_threshold as i32 {
                    long_range_sum += weight;
                }
            }
        }

        if total_sum > 0.0 {
            long_range_sum / total_sum
        } else {
            0.0
        }
    }

    /// Measure positional bias in attention
    fn measure_positional_bias(&self, weights: &[Vec<f32>]) -> f32 {
        let seq_len = weights.len();
        if seq_len == 0 {
            return 0.0;
        }

        // Measure how much attention depends on absolute position vs content
        let mut position_correlation = 0.0;
        let mut count = 0;

        for i in 0..seq_len {
            for j in 0..weights[i].len().min(seq_len) {
                let position_similarity = 1.0 - (i as f32 - j as f32).abs() / seq_len as f32;
                position_correlation += weights[i][j] * position_similarity;
                count += 1;
            }
        }

        if count > 0 {
            position_correlation / count as f32
        } else {
            0.0
        }
    }

    /// Compute attention distribution statistics
    fn compute_attention_distribution(
        &self,
        weights: &ArrayD<f32>,
    ) -> Result<AttentionDistribution> {
        let values: Vec<f32> = weights.iter().cloned().collect();

        if values.is_empty() {
            return Ok(AttentionDistribution {
                mean_attention: 0.0,
                std_attention: 0.0,
                max_attention: 0.0,
                min_attention: 0.0,
                entropy: 0.0,
                effective_context_length: 0.0,
            });
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / values.len() as f32;
        let std_dev = variance.sqrt();
        let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_val = values.iter().cloned().fold(f32::INFINITY, f32::min);

        // Compute entropy
        let entropy = self.compute_entropy(&values);

        // Compute effective context length (based on attention concentration)
        let effective_length = self.compute_effective_context_length(&values);

        Ok(AttentionDistribution {
            mean_attention: mean,
            std_attention: std_dev,
            max_attention: max_val,
            min_attention: min_val,
            entropy,
            effective_context_length: effective_length,
        })
    }

    /// Compute entropy of attention distribution
    fn compute_entropy(&self, values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let sum: f32 = values.iter().sum();
        if sum <= 0.0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &value in values {
            if value > 0.0 {
                let prob = value / sum;
                entropy -= prob * prob.log2();
            }
        }

        entropy
    }

    /// Compute effective context length based on attention concentration
    fn compute_effective_context_length(&self, values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let sum: f32 = values.iter().sum();
        if sum <= 0.0 {
            return 0.0;
        }

        // Compute 90th percentile threshold
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let mut cumulative_sum = 0.0;
        let target_sum = sum * 0.9; // 90% of attention mass

        for (i, &value) in sorted_values.iter().enumerate() {
            cumulative_sum += value;
            if cumulative_sum >= target_sum {
                return (i + 1) as f32;
            }
        }

        values.len() as f32
    }

    /// Detect attention pattern from weights array
    fn detect_attention_pattern_from_weights(
        &self,
        weights: &ArrayD<f32>,
    ) -> Result<AttentionPattern> {
        let shape = weights.shape();
        if shape.len() != 2 {
            return Ok(AttentionPattern::Random);
        }

        let weights_2d: Vec<Vec<f32>> = (0..shape[0])
            .map(|i| (0..shape[1]).map(|j| weights[[i, j]]).collect())
            .collect();

        Ok(self.detect_attention_pattern(&weights_2d))
    }

    /// Compute head importance score
    fn compute_head_importance(&self, weights: &ArrayD<f32>) -> Result<f32> {
        let values: Vec<f32> = weights.iter().cloned().collect();

        if values.is_empty() {
            return Ok(0.0);
        }

        // Importance based on entropy (higher entropy = more important for diverse attention)
        let entropy = self.compute_entropy(&values);
        let max_entropy = (values.len() as f32).log2();

        if max_entropy > 0.0 {
            Ok(entropy / max_entropy)
        } else {
            Ok(0.0)
        }
    }

    /// Compute attention entropy
    fn compute_attention_entropy(&self, weights: &[Vec<f32>]) -> f32 {
        let values: Vec<f32> = weights.iter().flatten().cloned().collect();
        self.compute_entropy(&values)
    }

    /// Compute sparsity ratio
    fn compute_sparsity_ratio(&self, weights: &[Vec<f32>]) -> f32 {
        let total_count = weights.iter().map(|row| row.len()).sum::<usize>();
        if total_count == 0 {
            return 0.0;
        }

        let sparse_count = weights
            .iter()
            .flatten()
            .filter(|&&w| w < self.config.attention_threshold)
            .count();

        sparse_count as f32 / total_count as f32
    }

    /// Compute layer diversity score
    fn compute_layer_diversity(&self, head_analyses: &[AttentionHeadAnalysis]) -> f32 {
        if head_analyses.len() < 2 {
            return 0.0;
        }

        // Measure diversity based on different specialization types
        let mut specialization_counts: HashMap<HeadSpecializationType, usize> = HashMap::new();
        for analysis in head_analyses {
            *specialization_counts.entry(analysis.specialization_type.clone()).or_insert(0) += 1;
        }

        let num_types = specialization_counts.len() as f32;
        let max_types = 8.0; // Maximum possible specialization types

        num_types / max_types
    }

    /// Analyze head redundancy
    fn analyze_head_redundancy(
        &self,
        head_analyses: &[AttentionHeadAnalysis],
    ) -> RedundancyAnalysis {
        let mut redundant_heads = Vec::new();
        let redundancy_groups = Vec::new();

        // Group heads by similar behavior
        for i in 0..head_analyses.len() {
            for j in (i + 1)..head_analyses.len() {
                let similarity = self.compute_head_similarity(&head_analyses[i], &head_analyses[j]);
                if similarity > 0.8 {
                    redundant_heads.push((i, j, similarity));
                }
            }
        }

        RedundancyAnalysis {
            redundant_head_pairs: redundant_heads,
            redundancy_groups,
            overall_redundancy_score: self.compute_overall_redundancy(head_analyses),
        }
    }

    /// Compute similarity between two attention heads
    fn compute_head_similarity(
        &self,
        head1: &AttentionHeadAnalysis,
        head2: &AttentionHeadAnalysis,
    ) -> f32 {
        // Similarity based on specialization type and attention distribution
        let type_similarity =
            if head1.specialization_type == head2.specialization_type { 1.0 } else { 0.0 };

        let dist_similarity = {
            let d1 = &head1.attention_distribution;
            let d2 = &head2.attention_distribution;

            let mean_diff = (d1.mean_attention - d2.mean_attention).abs();
            let std_diff = (d1.std_attention - d2.std_attention).abs();
            let entropy_diff = (d1.entropy - d2.entropy).abs();

            1.0 - (mean_diff + std_diff + entropy_diff) / 3.0
        };

        (type_similarity + dist_similarity) / 2.0
    }

    /// Compute overall redundancy score for the layer
    fn compute_overall_redundancy(&self, head_analyses: &[AttentionHeadAnalysis]) -> f32 {
        if head_analyses.len() < 2 {
            return 0.0;
        }

        let mut total_similarity = 0.0;
        let mut pair_count = 0;

        for i in 0..head_analyses.len() {
            for j in (i + 1)..head_analyses.len() {
                total_similarity +=
                    self.compute_head_similarity(&head_analyses[i], &head_analyses[j]);
                pair_count += 1;
            }
        }

        if pair_count > 0 {
            total_similarity / pair_count as f32
        } else {
            0.0
        }
    }
}

/// Analysis results for a transformer layer's attention mechanism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAttentionAnalysis {
    pub layer_index: usize,
    pub num_heads: usize,
    pub head_analyses: Vec<AttentionHeadAnalysis>,
    pub attention_maps: Vec<AttentionMap>,
    pub layer_diversity_score: f32,
    pub redundancy_analysis: RedundancyAnalysis,
}

/// Analysis of attention head redundancy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedundancyAnalysis {
    pub redundant_head_pairs: Vec<(usize, usize, f32)>, // (head1, head2, similarity)
    pub redundancy_groups: Vec<Vec<usize>>,
    pub overall_redundancy_score: f32,
}

/// Transformer-specific debugging utilities
#[derive(Debug)]
pub struct TransformerDebugger {
    pub config: TransformerDebugConfig,
    layer_analyses: Vec<LayerAttentionAnalysis>,
    attention_debugger: AttentionDebugger,
}

/// Configuration for transformer debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerDebugConfig {
    pub attention_config: AttentionDebugConfig,
    pub enable_layer_analysis: bool,
    pub enable_cross_layer_analysis: bool,
    pub max_layers_to_analyze: usize,
}

impl Default for TransformerDebugConfig {
    fn default() -> Self {
        Self {
            attention_config: AttentionDebugConfig::default(),
            enable_layer_analysis: true,
            enable_cross_layer_analysis: true,
            max_layers_to_analyze: 48, // Support for large models
        }
    }
}

impl TransformerDebugger {
    /// Create a new transformer debugger
    pub fn new(config: TransformerDebugConfig) -> Self {
        let attention_debugger = AttentionDebugger::new(config.attention_config.clone());

        Self {
            config,
            layer_analyses: Vec::new(),
            attention_debugger,
        }
    }

    /// Analyze entire transformer model attention patterns
    pub fn analyze_transformer_attention(
        &mut self,
        model_attention_weights: &[Vec<ArrayD<f32>>], // [layer][head] -> attention weights
    ) -> Result<TransformerAttentionAnalysis> {
        let mut layer_analyses = Vec::new();

        for (layer_idx, layer_weights) in model_attention_weights.iter().enumerate() {
            if layer_idx >= self.config.max_layers_to_analyze {
                break;
            }

            let layer_analysis =
                self.attention_debugger.analyze_attention_layer(layer_idx, layer_weights)?;
            layer_analyses.push(layer_analysis);
        }

        self.layer_analyses = layer_analyses.clone();

        // Perform cross-layer analysis
        let cross_layer_analysis = if self.config.enable_cross_layer_analysis {
            Some(self.perform_cross_layer_analysis(&layer_analyses)?)
        } else {
            None
        };

        Ok(TransformerAttentionAnalysis {
            num_layers: model_attention_weights.len(),
            layer_analyses,
            cross_layer_analysis,
            model_attention_summary: self.generate_model_attention_summary()?,
        })
    }

    /// Perform cross-layer attention analysis
    fn perform_cross_layer_analysis(
        &self,
        layer_analyses: &[LayerAttentionAnalysis],
    ) -> Result<CrossLayerAnalysis> {
        let attention_evolution = self.analyze_attention_evolution(layer_analyses)?;
        let head_consistency = self.analyze_head_consistency(layer_analyses)?;
        let pattern_progression = self.analyze_pattern_progression(layer_analyses)?;

        Ok(CrossLayerAnalysis {
            attention_evolution,
            head_consistency,
            pattern_progression,
            layer_diversity_trend: self.compute_layer_diversity_trend(layer_analyses),
        })
    }

    /// Analyze how attention patterns evolve across layers
    fn analyze_attention_evolution(
        &self,
        layer_analyses: &[LayerAttentionAnalysis],
    ) -> Result<AttentionEvolution> {
        let mut entropy_trend = Vec::new();
        let mut sparsity_trend = Vec::new();

        for layer in layer_analyses {
            let layer_entropy: f32 =
                layer.attention_maps.iter().map(|map| map.attention_entropy).sum::<f32>()
                    / layer.attention_maps.len() as f32;
            let layer_sparsity: f32 =
                layer.attention_maps.iter().map(|map| map.sparsity_ratio).sum::<f32>()
                    / layer.attention_maps.len() as f32;

            entropy_trend.push(layer_entropy);
            sparsity_trend.push(layer_sparsity);
        }

        let evolution_type = self.classify_evolution_type(&entropy_trend);

        Ok(AttentionEvolution {
            entropy_trend,
            sparsity_trend,
            evolution_type,
        })
    }

    /// Classify the type of attention evolution
    fn classify_evolution_type(&self, entropy_trend: &[f32]) -> EvolutionType {
        if entropy_trend.len() < 3 {
            return EvolutionType::Stable;
        }

        let start_entropy = entropy_trend[0];
        let end_entropy = entropy_trend[entropy_trend.len() - 1];
        let change_ratio = (end_entropy - start_entropy) / start_entropy.max(1e-8);

        match change_ratio {
            r if r > 0.2 => EvolutionType::Increasing,
            r if r < -0.2 => EvolutionType::Decreasing,
            _ => EvolutionType::Stable,
        }
    }

    /// Analyze head consistency across layers
    fn analyze_head_consistency(
        &self,
        layer_analyses: &[LayerAttentionAnalysis],
    ) -> Result<HeadConsistency> {
        let mut specialization_consistency = HashMap::new();
        let pattern_consistency = HashMap::new();

        // Track how specialization types are distributed across layers
        for layer in layer_analyses {
            for head in &layer.head_analyses {
                let spec_type = &head.specialization_type;
                let layer_counts =
                    specialization_consistency.entry(spec_type.clone()).or_insert_with(Vec::new);
                layer_counts.push(layer.layer_index);
            }
        }

        Ok(HeadConsistency {
            specialization_consistency,
            pattern_consistency,
            consistency_score: self.compute_consistency_score(layer_analyses),
        })
    }

    /// Compute overall consistency score
    fn compute_consistency_score(&self, layer_analyses: &[LayerAttentionAnalysis]) -> f32 {
        if layer_analyses.len() < 2 {
            return 1.0;
        }

        // Measure how similar the distribution of head types is across layers
        let mut layer_distributions = Vec::new();

        for layer in layer_analyses {
            let mut distribution: HashMap<HeadSpecializationType, f32> = HashMap::new();
            for head in &layer.head_analyses {
                *distribution.entry(head.specialization_type.clone()).or_insert(0.0) += 1.0;
            }

            // Normalize
            let total: f32 = distribution.values().sum();
            if total > 0.0 {
                for value in distribution.values_mut() {
                    *value /= total;
                }
            }

            layer_distributions.push(distribution);
        }

        // Compute average pairwise similarity
        let mut total_similarity = 0.0;
        let mut pair_count = 0;

        for i in 0..layer_distributions.len() {
            for j in (i + 1)..layer_distributions.len() {
                let similarity = self.compute_distribution_similarity(
                    &layer_distributions[i],
                    &layer_distributions[j],
                );
                total_similarity += similarity;
                pair_count += 1;
            }
        }

        if pair_count > 0 {
            total_similarity / pair_count as f32
        } else {
            1.0
        }
    }

    /// Compute similarity between two distributions
    fn compute_distribution_similarity(
        &self,
        dist1: &HashMap<HeadSpecializationType, f32>,
        dist2: &HashMap<HeadSpecializationType, f32>,
    ) -> f32 {
        let mut all_keys: std::collections::HashSet<_> = dist1.keys().collect();
        all_keys.extend(dist2.keys());

        let mut similarity = 0.0;
        for key in all_keys {
            let val1 = dist1.get(key).unwrap_or(&0.0);
            let val2 = dist2.get(key).unwrap_or(&0.0);
            similarity += (val1 - val2).abs();
        }

        1.0 - (similarity / 2.0) // Normalize to [0, 1]
    }

    /// Analyze pattern progression across layers
    fn analyze_pattern_progression(
        &self,
        layer_analyses: &[LayerAttentionAnalysis],
    ) -> Result<PatternProgression> {
        let mut pattern_evolution = Vec::new();

        for layer in layer_analyses {
            let mut pattern_counts: HashMap<AttentionPattern, usize> = HashMap::new();
            for map in &layer.attention_maps {
                *pattern_counts.entry(map.attention_pattern.clone()).or_insert(0) += 1;
            }
            pattern_evolution.push(pattern_counts);
        }

        let dominant_pattern_sequence = self.extract_dominant_patterns(&pattern_evolution);

        Ok(PatternProgression {
            pattern_evolution,
            dominant_pattern_sequence,
        })
    }

    /// Extract sequence of dominant patterns across layers
    fn extract_dominant_patterns(
        &self,
        pattern_evolution: &[HashMap<AttentionPattern, usize>],
    ) -> Vec<AttentionPattern> {
        pattern_evolution
            .iter()
            .map(|patterns| {
                patterns
                    .iter()
                    .max_by_key(|(_, &count)| count)
                    .map(|(pattern, _)| pattern.clone())
                    .unwrap_or(AttentionPattern::Random)
            })
            .collect()
    }

    /// Compute layer diversity trend
    fn compute_layer_diversity_trend(&self, layer_analyses: &[LayerAttentionAnalysis]) -> Vec<f32> {
        layer_analyses.iter().map(|layer| layer.layer_diversity_score).collect()
    }

    /// Generate model attention summary
    fn generate_model_attention_summary(&self) -> Result<ModelAttentionSummary> {
        if self.layer_analyses.is_empty() {
            return Ok(ModelAttentionSummary::default());
        }

        let total_heads: usize = self.layer_analyses.iter().map(|layer| layer.num_heads).sum();
        let avg_diversity: f32 =
            self.layer_analyses.iter().map(|layer| layer.layer_diversity_score).sum::<f32>()
                / self.layer_analyses.len() as f32;
        let avg_redundancy: f32 = self
            .layer_analyses
            .iter()
            .map(|layer| layer.redundancy_analysis.overall_redundancy_score)
            .sum::<f32>()
            / self.layer_analyses.len() as f32;

        // Count specialization types across all layers
        let mut specialization_distribution: HashMap<HeadSpecializationType, usize> =
            HashMap::new();
        for layer in &self.layer_analyses {
            for head in &layer.head_analyses {
                *specialization_distribution
                    .entry(head.specialization_type.clone())
                    .or_insert(0) += 1;
            }
        }

        Ok(ModelAttentionSummary {
            total_layers: self.layer_analyses.len(),
            total_heads,
            average_diversity_score: avg_diversity,
            average_redundancy_score: avg_redundancy,
            specialization_distribution,
            model_attention_health: self
                .assess_model_attention_health(avg_diversity, avg_redundancy),
        })
    }

    /// Assess overall model attention health
    fn assess_model_attention_health(
        &self,
        diversity: f32,
        redundancy: f32,
    ) -> AttentionHealthStatus {
        let health_score = diversity * (1.0 - redundancy); // High diversity, low redundancy is good

        match health_score {
            s if s > 0.7 => AttentionHealthStatus::Excellent,
            s if s > 0.5 => AttentionHealthStatus::Good,
            s if s > 0.3 => AttentionHealthStatus::Fair,
            _ => AttentionHealthStatus::Poor,
        }
    }
}

/// Complete transformer attention analysis results
#[derive(Debug, Serialize, Deserialize)]
pub struct TransformerAttentionAnalysis {
    pub num_layers: usize,
    pub layer_analyses: Vec<LayerAttentionAnalysis>,
    pub cross_layer_analysis: Option<CrossLayerAnalysis>,
    pub model_attention_summary: ModelAttentionSummary,
}

/// Cross-layer attention analysis
#[derive(Debug, Serialize, Deserialize)]
pub struct CrossLayerAnalysis {
    pub attention_evolution: AttentionEvolution,
    pub head_consistency: HeadConsistency,
    pub pattern_progression: PatternProgression,
    pub layer_diversity_trend: Vec<f32>,
}

/// Attention evolution across layers
#[derive(Debug, Serialize, Deserialize)]
pub struct AttentionEvolution {
    pub entropy_trend: Vec<f32>,
    pub sparsity_trend: Vec<f32>,
    pub evolution_type: EvolutionType,
}

/// Types of attention evolution patterns
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvolutionType {
    Increasing, // Attention becomes more diverse
    Decreasing, // Attention becomes more focused
    Stable,     // Attention patterns remain consistent
}

/// Head consistency analysis across layers
#[derive(Debug, Serialize, Deserialize)]
pub struct HeadConsistency {
    pub specialization_consistency: HashMap<HeadSpecializationType, Vec<usize>>,
    pub pattern_consistency: HashMap<AttentionPattern, Vec<usize>>,
    pub consistency_score: f32,
}

/// Pattern progression analysis
#[derive(Debug, Serialize, Deserialize)]
pub struct PatternProgression {
    pub pattern_evolution: Vec<HashMap<AttentionPattern, usize>>,
    pub dominant_pattern_sequence: Vec<AttentionPattern>,
}

/// Model-level attention summary
#[derive(Debug, Serialize, Deserialize)]
pub struct ModelAttentionSummary {
    pub total_layers: usize,
    pub total_heads: usize,
    pub average_diversity_score: f32,
    pub average_redundancy_score: f32,
    pub specialization_distribution: HashMap<HeadSpecializationType, usize>,
    pub model_attention_health: AttentionHealthStatus,
}

impl Default for ModelAttentionSummary {
    fn default() -> Self {
        Self {
            total_layers: 0,
            total_heads: 0,
            average_diversity_score: 0.0,
            average_redundancy_score: 0.0,
            specialization_distribution: HashMap::new(),
            model_attention_health: AttentionHealthStatus::Poor,
        }
    }
}

/// Overall attention health status
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttentionHealthStatus {
    Excellent,
    Good,
    Fair,
    Poor,
}

// Convenience macro for creating attention debugger with default config
#[macro_export]
macro_rules! debug_attention {
    ($attention_weights:expr) => {{
        let mut debugger = $crate::neural_network_debugging::AttentionDebugger::new(
            $crate::neural_network_debugging::AttentionDebugConfig::default(),
        );
        debugger.analyze_attention_layer(0, $attention_weights)
    }};
}

// Convenience macro for transformer debugging
#[macro_export]
macro_rules! debug_transformer {
    ($model_weights:expr) => {{
        let mut debugger = $crate::neural_network_debugging::TransformerDebugger::new(
            $crate::neural_network_debugging::TransformerDebugConfig::default(),
        );
        debugger.analyze_transformer_attention($model_weights)
    }};
}
