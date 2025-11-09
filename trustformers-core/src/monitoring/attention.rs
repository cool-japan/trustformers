// Attention visualization and analysis tools
#![allow(unused_variables)] // Attention monitoring with reserved parameters

use crate::tensor::Tensor;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Attention visualization and analysis system
#[derive(Debug, Clone)]
pub struct AttentionVisualizer {
    config: AttentionVisualizerConfig,
    active_sessions: HashMap<String, AttentionSession>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionVisualizerConfig {
    pub enabled: bool,
    pub max_layers_to_track: usize,
    pub max_heads_to_track: usize,
    pub save_attention_matrices: bool,
    pub compute_statistics: bool,
    pub track_attention_entropy: bool,
    pub visualize_patterns: bool,
}

impl Default for AttentionVisualizerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_layers_to_track: 12,
            max_heads_to_track: 12,
            save_attention_matrices: false, // Can be memory intensive
            compute_statistics: true,
            track_attention_entropy: true,
            visualize_patterns: true,
        }
    }
}

#[derive(Debug, Clone)]
struct AttentionSession {
    #[allow(dead_code)]
    id: String,
    layer_data: HashMap<usize, LayerAttentionData>,
    input_tokens: Option<Vec<String>>,
}

#[derive(Debug, Clone)]
struct LayerAttentionData {
    layer_idx: usize,
    #[allow(dead_code)]
    attention_weights: Option<Vec<f32>>, // Flattened attention matrix
    attention_shape: Vec<usize>,
    head_statistics: Vec<HeadStatistics>,
    patterns: Vec<AttentionPattern>,
}

/// Statistics for individual attention heads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadStatistics {
    pub head_idx: usize,
    pub entropy: f64,
    pub max_attention: f64,
    pub sparsity: f64,
    pub diagonal_ratio: f64,
    pub local_attention_ratio: f64,
    pub global_attention_ratio: f64,
}

/// Detected attention patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionPattern {
    pub pattern_type: AttentionPatternType,
    pub confidence: f64,
    pub layer_idx: usize,
    pub head_idx: usize,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttentionPatternType {
    Diagonal,     // Attending to nearby tokens
    Vertical,     // Attending to specific positions
    Horizontal,   // Attending across all positions
    Block,        // Block-structured attention
    Sparse,       // Sparse attention pattern
    Global,       // Global attention to special tokens
    Local,        // Local window attention
    Causal,       // Causal/autoregressive pattern
    Broadcast,    // Broadcasting from specific tokens
    Hierarchical, // Hierarchical attention structure
}

/// Complete attention analysis report
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AttentionReport {
    pub session_id: String,
    pub num_layers: usize,
    pub num_heads: usize,
    pub sequence_length: usize,
    pub layer_reports: Vec<LayerAttentionReport>,
    pub global_statistics: GlobalAttentionStatistics,
    pub pattern_summary: PatternSummary,
    pub attention_flow: AttentionFlow,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAttentionReport {
    pub layer_idx: usize,
    pub head_statistics: Vec<HeadStatistics>,
    pub patterns: Vec<AttentionPattern>,
    pub average_entropy: f64,
    pub sparsity_distribution: Vec<f64>,
    pub attention_concentration: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalAttentionStatistics {
    pub total_entropy: f64,
    pub average_sparsity: f64,
    pub attention_variance: f64,
    pub layer_similarity: Vec<f64>,
    pub head_diversity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PatternSummary {
    pub pattern_counts: HashMap<String, usize>,
    pub dominant_patterns: Vec<AttentionPattern>,
    pub pattern_evolution: Vec<LayerPatternEvolution>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerPatternEvolution {
    pub layer_idx: usize,
    pub pattern_changes: Vec<PatternTransition>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternTransition {
    pub from_pattern: AttentionPatternType,
    pub to_pattern: AttentionPatternType,
    pub transition_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionFlow {
    pub information_flow: Vec<FlowVector>,
    pub bottlenecks: Vec<AttentionBottleneck>,
    pub flow_concentration: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowVector {
    pub from_layer: usize,
    pub to_layer: usize,
    pub flow_strength: f64,
    pub flow_direction: FlowDirection,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlowDirection {
    Forward,
    Backward,
    Lateral,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionBottleneck {
    pub layer_idx: usize,
    pub position: usize,
    pub bottleneck_strength: f64,
    pub affected_heads: Vec<usize>,
}

impl Default for GlobalAttentionStatistics {
    fn default() -> Self {
        Self {
            total_entropy: 0.0,
            average_sparsity: 0.0,
            attention_variance: 0.0,
            layer_similarity: Vec::new(),
            head_diversity: 0.0,
        }
    }
}

impl Default for AttentionFlow {
    fn default() -> Self {
        Self {
            information_flow: Vec::new(),
            bottlenecks: Vec::new(),
            flow_concentration: 0.0,
        }
    }
}

impl Default for AttentionVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

impl AttentionVisualizer {
    pub fn new() -> Self {
        Self {
            config: AttentionVisualizerConfig::default(),
            active_sessions: HashMap::new(),
        }
    }

    pub fn with_config(config: AttentionVisualizerConfig) -> Self {
        Self {
            config,
            active_sessions: HashMap::new(),
        }
    }

    /// Start tracking attention for a session
    pub fn start_tracking(&mut self, session_id: &str) -> Result<()> {
        let session = AttentionSession {
            id: session_id.to_string(),
            layer_data: HashMap::new(),
            input_tokens: None,
        };

        self.active_sessions.insert(session_id.to_string(), session);
        Ok(())
    }

    /// Track attention weights for a specific layer
    pub fn track_attention(
        &mut self,
        session_id: &str,
        layer_idx: usize,
        attention_weights: &Tensor,
        input_tokens: Option<&[String]>,
    ) -> Result<()> {
        if !self.config.enabled || layer_idx >= self.config.max_layers_to_track {
            return Ok(());
        }

        // Get existing session
        if !self.active_sessions.contains_key(session_id) {
            return Ok(()); // Session not found, skip tracking
        }

        // Update input tokens if provided
        if let Some(tokens) = input_tokens {
            if let Some(session) = self.active_sessions.get_mut(session_id) {
                session.input_tokens = Some(tokens.to_vec());
            }
        }

        // Analyze attention weights
        let layer_data = self.analyze_attention_layer(layer_idx, attention_weights)?;

        // Insert the analyzed data
        if let Some(session) = self.active_sessions.get_mut(session_id) {
            session.layer_data.insert(layer_idx, layer_data);
        }

        Ok(())
    }

    /// Generate attention analysis report
    pub fn get_report(&mut self, session_id: &str) -> Result<AttentionReport> {
        let session = self
            .active_sessions
            .remove(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session not found: {}", session_id))?;

        let mut layer_reports = Vec::new();
        let mut num_heads = 0;
        let mut sequence_length = 0;

        // Process each layer
        for (&layer_idx, layer_data) in &session.layer_data {
            let layer_report = self.generate_layer_report(layer_data)?;

            if num_heads == 0 {
                num_heads = layer_report.head_statistics.len();
            }

            if sequence_length == 0 && !layer_data.attention_shape.is_empty() {
                sequence_length = layer_data.attention_shape[layer_data.attention_shape.len() - 1];
            }

            layer_reports.push(layer_report);
        }

        // Sort by layer index
        layer_reports.sort_by_key(|r| r.layer_idx);

        // Compute global statistics
        let global_statistics = self.compute_global_statistics(&layer_reports)?;

        // Analyze patterns
        let pattern_summary = self.analyze_pattern_evolution(&layer_reports)?;

        // Compute attention flow
        let attention_flow = self.compute_attention_flow(&layer_reports)?;

        Ok(AttentionReport {
            session_id: session_id.to_string(),
            num_layers: layer_reports.len(),
            num_heads,
            sequence_length,
            layer_reports,
            global_statistics,
            pattern_summary,
            attention_flow,
        })
    }

    /// Clear all session data
    pub fn clear(&mut self) -> Result<()> {
        self.active_sessions.clear();
        Ok(())
    }

    /// Analyze attention weights for a single layer
    fn analyze_attention_layer(
        &self,
        layer_idx: usize,
        attention_weights: &Tensor,
    ) -> Result<LayerAttentionData> {
        let shape = attention_weights.shape();
        let data = attention_weights.data()?;

        // Expected shape: [batch_size, num_heads, seq_len, seq_len]
        if shape.len() != 4 {
            return Err(anyhow::anyhow!(
                "Expected 4D attention tensor, got {}D",
                shape.len()
            ));
        }

        let batch_size = shape[0];
        let num_heads = shape[1].min(self.config.max_heads_to_track);
        let seq_len = shape[2];

        let mut head_statistics = Vec::new();
        let mut patterns = Vec::new();

        // Analyze each head
        for head_idx in 0..num_heads {
            let head_start = head_idx * seq_len * seq_len;
            let head_end = head_start + seq_len * seq_len;

            if head_end <= data.len() {
                let head_data = &data[head_start..head_end];
                let stats = self.compute_head_statistics(head_idx, head_data, seq_len)?;
                let head_patterns =
                    self.detect_attention_patterns(layer_idx, head_idx, head_data, seq_len)?;

                head_statistics.push(stats);
                patterns.extend(head_patterns);
            }
        }

        let attention_weights_data =
            if self.config.save_attention_matrices { Some(data) } else { None };

        Ok(LayerAttentionData {
            layer_idx,
            attention_weights: attention_weights_data,
            attention_shape: shape,
            head_statistics,
            patterns,
        })
    }

    /// Compute statistics for a single attention head
    fn compute_head_statistics(
        &self,
        head_idx: usize,
        head_data: &[f32],
        seq_len: usize,
    ) -> Result<HeadStatistics> {
        let mut entropy = 0.0;
        let mut max_attention: f32 = 0.0;
        let mut diagonal_sum = 0.0;
        let mut local_sum = 0.0;
        let mut global_sum = 0.0;
        let mut non_zero_count = 0;

        for i in 0..seq_len {
            for j in 0..seq_len {
                let idx = i * seq_len + j;
                let val = head_data[idx];

                max_attention = max_attention.max(val);

                if val > 1e-8 {
                    entropy -= val * val.ln();
                    non_zero_count += 1;
                }

                // Diagonal attention
                if i == j {
                    diagonal_sum += val;
                }

                // Local attention (within window of size 3)
                if (i as i32 - j as i32).abs() <= 1 {
                    local_sum += val;
                }

                // Global attention (to first/last tokens)
                if j == 0 || j == seq_len - 1 {
                    global_sum += val;
                }
            }
        }

        let total_attention = head_data.iter().sum::<f32>();
        let sparsity = 1.0 - (non_zero_count as f64 / (seq_len * seq_len) as f64);
        let diagonal_ratio = diagonal_sum as f64 / total_attention as f64;
        let local_attention_ratio = local_sum as f64 / total_attention as f64;
        let global_attention_ratio = global_sum as f64 / total_attention as f64;

        Ok(HeadStatistics {
            head_idx,
            entropy: entropy as f64,
            max_attention: max_attention as f64,
            sparsity,
            diagonal_ratio,
            local_attention_ratio,
            global_attention_ratio,
        })
    }

    /// Detect attention patterns in a head
    fn detect_attention_patterns(
        &self,
        layer_idx: usize,
        head_idx: usize,
        head_data: &[f32],
        seq_len: usize,
    ) -> Result<Vec<AttentionPattern>> {
        let mut patterns = Vec::new();

        // Detect diagonal pattern
        let diagonal_strength = self.measure_diagonal_pattern(head_data, seq_len);
        if diagonal_strength > 0.5 {
            patterns.push(AttentionPattern {
                pattern_type: AttentionPatternType::Diagonal,
                confidence: diagonal_strength,
                layer_idx,
                head_idx,
                description: "Strong diagonal attention pattern".to_string(),
            });
        }

        // Detect causal pattern
        let causal_strength = self.measure_causal_pattern(head_data, seq_len);
        if causal_strength > 0.7 {
            patterns.push(AttentionPattern {
                pattern_type: AttentionPatternType::Causal,
                confidence: causal_strength,
                layer_idx,
                head_idx,
                description: "Causal attention pattern".to_string(),
            });
        }

        // Detect sparse pattern
        let sparse_strength = self.measure_sparse_pattern(head_data, seq_len);
        if sparse_strength > 0.8 {
            patterns.push(AttentionPattern {
                pattern_type: AttentionPatternType::Sparse,
                confidence: sparse_strength,
                layer_idx,
                head_idx,
                description: "Sparse attention pattern".to_string(),
            });
        }

        // Detect global pattern
        let global_strength = self.measure_global_pattern(head_data, seq_len);
        if global_strength > 0.6 {
            patterns.push(AttentionPattern {
                pattern_type: AttentionPatternType::Global,
                confidence: global_strength,
                layer_idx,
                head_idx,
                description: "Global attention to special tokens".to_string(),
            });
        }

        Ok(patterns)
    }

    /// Measure diagonal pattern strength
    fn measure_diagonal_pattern(&self, head_data: &[f32], seq_len: usize) -> f64 {
        let mut diagonal_sum = 0.0;
        let total_sum: f32 = head_data.iter().sum();

        for i in 0..seq_len {
            diagonal_sum += head_data[i * seq_len + i];
        }

        diagonal_sum as f64 / total_sum as f64
    }

    /// Measure causal pattern strength
    fn measure_causal_pattern(&self, head_data: &[f32], seq_len: usize) -> f64 {
        let mut lower_triangular_sum = 0.0;
        let total_sum: f32 = head_data.iter().sum();

        for i in 0..seq_len {
            for j in 0..=i {
                lower_triangular_sum += head_data[i * seq_len + j];
            }
        }

        lower_triangular_sum as f64 / total_sum as f64
    }

    /// Measure sparse pattern strength
    fn measure_sparse_pattern(&self, head_data: &[f32], seq_len: usize) -> f64 {
        let threshold = 0.1;
        let non_zero_count = head_data.iter().filter(|&&x| x > threshold).count();
        let total_elements = seq_len * seq_len;

        1.0 - (non_zero_count as f64 / total_elements as f64)
    }

    /// Measure global attention pattern
    fn measure_global_pattern(&self, head_data: &[f32], seq_len: usize) -> f64 {
        let mut global_attention = 0.0;
        let total_sum: f32 = head_data.iter().sum();

        // Attention to first and last tokens
        for i in 0..seq_len {
            global_attention += head_data[i * seq_len]; // First token
            global_attention += head_data[i * seq_len + (seq_len - 1)]; // Last token
        }

        global_attention as f64 / total_sum as f64
    }

    /// Generate report for a single layer
    fn generate_layer_report(
        &self,
        layer_data: &LayerAttentionData,
    ) -> Result<LayerAttentionReport> {
        let average_entropy = layer_data.head_statistics.iter().map(|s| s.entropy).sum::<f64>()
            / layer_data.head_statistics.len() as f64;

        let sparsity_distribution: Vec<f64> =
            layer_data.head_statistics.iter().map(|s| s.sparsity).collect();

        let attention_concentration =
            layer_data.head_statistics.iter().map(|s| s.max_attention).sum::<f64>()
                / layer_data.head_statistics.len() as f64;

        Ok(LayerAttentionReport {
            layer_idx: layer_data.layer_idx,
            head_statistics: layer_data.head_statistics.clone(),
            patterns: layer_data.patterns.clone(),
            average_entropy,
            sparsity_distribution,
            attention_concentration,
        })
    }

    /// Compute global statistics across all layers
    fn compute_global_statistics(
        &self,
        layer_reports: &[LayerAttentionReport],
    ) -> Result<GlobalAttentionStatistics> {
        if layer_reports.is_empty() {
            return Ok(GlobalAttentionStatistics::default());
        }

        let total_entropy = layer_reports.iter().map(|r| r.average_entropy).sum::<f64>()
            / layer_reports.len() as f64;

        let average_sparsity =
            layer_reports.iter().flat_map(|r| &r.sparsity_distribution).sum::<f64>()
                / layer_reports.iter().map(|r| r.sparsity_distribution.len()).sum::<usize>() as f64;

        let attention_variance = self.compute_attention_variance(layer_reports);
        let layer_similarity = self.compute_layer_similarity(layer_reports);
        let head_diversity = self.compute_head_diversity(layer_reports);

        Ok(GlobalAttentionStatistics {
            total_entropy,
            average_sparsity,
            attention_variance,
            layer_similarity,
            head_diversity,
        })
    }

    /// Compute attention variance across layers
    fn compute_attention_variance(&self, layer_reports: &[LayerAttentionReport]) -> f64 {
        let concentrations: Vec<f64> =
            layer_reports.iter().map(|r| r.attention_concentration).collect();

        let mean = concentrations.iter().sum::<f64>() / concentrations.len() as f64;
        let variance = concentrations.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / concentrations.len() as f64;

        variance
    }

    /// Compute similarity between adjacent layers
    fn compute_layer_similarity(&self, layer_reports: &[LayerAttentionReport]) -> Vec<f64> {
        let mut similarities = Vec::new();

        for i in 0..layer_reports.len().saturating_sub(1) {
            let sim = self.compute_pattern_similarity(&layer_reports[i], &layer_reports[i + 1]);
            similarities.push(sim);
        }

        similarities
    }

    /// Compute pattern similarity between two layers
    fn compute_pattern_similarity(
        &self,
        layer1: &LayerAttentionReport,
        layer2: &LayerAttentionReport,
    ) -> f64 {
        // Simplified similarity based on entropy correlation
        let entropy1: Vec<f64> = layer1.head_statistics.iter().map(|s| s.entropy).collect();
        let entropy2: Vec<f64> = layer2.head_statistics.iter().map(|s| s.entropy).collect();

        if entropy1.len() != entropy2.len() {
            return 0.0;
        }

        let mean1 = entropy1.iter().sum::<f64>() / entropy1.len() as f64;
        let mean2 = entropy2.iter().sum::<f64>() / entropy2.len() as f64;

        let covariance = entropy1
            .iter()
            .zip(&entropy2)
            .map(|(x, y)| (x - mean1) * (y - mean2))
            .sum::<f64>()
            / entropy1.len() as f64;

        let std1 = (entropy1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>()
            / entropy1.len() as f64)
            .sqrt();
        let std2 = (entropy2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>()
            / entropy2.len() as f64)
            .sqrt();

        if std1 > 0.0 && std2 > 0.0 {
            covariance / (std1 * std2)
        } else {
            0.0
        }
    }

    /// Compute head diversity within layers
    fn compute_head_diversity(&self, layer_reports: &[LayerAttentionReport]) -> f64 {
        let mut total_diversity = 0.0;
        let mut count = 0;

        for layer in layer_reports {
            if layer.head_statistics.len() > 1 {
                let entropies: Vec<f64> = layer.head_statistics.iter().map(|s| s.entropy).collect();
                let mean_entropy = entropies.iter().sum::<f64>() / entropies.len() as f64;
                let variance = entropies.iter().map(|x| (x - mean_entropy).powi(2)).sum::<f64>()
                    / entropies.len() as f64;

                total_diversity += variance.sqrt();
                count += 1;
            }
        }

        if count > 0 {
            total_diversity / count as f64
        } else {
            0.0
        }
    }

    /// Analyze pattern evolution across layers
    fn analyze_pattern_evolution(
        &self,
        layer_reports: &[LayerAttentionReport],
    ) -> Result<PatternSummary> {
        let mut pattern_counts = HashMap::new();
        let mut dominant_patterns = Vec::new();
        let mut pattern_evolution = Vec::new();

        // Count patterns
        for layer in layer_reports {
            for pattern in &layer.patterns {
                let pattern_name = format!("{:?}", pattern.pattern_type);
                *pattern_counts.entry(pattern_name).or_insert(0) += 1;

                if pattern.confidence > 0.8 {
                    dominant_patterns.push(pattern.clone());
                }
            }
        }

        // Analyze evolution between adjacent layers
        for i in 0..layer_reports.len().saturating_sub(1) {
            let evolution =
                self.analyze_layer_pattern_evolution(&layer_reports[i], &layer_reports[i + 1]);
            pattern_evolution.push(evolution);
        }

        Ok(PatternSummary {
            pattern_counts,
            dominant_patterns,
            pattern_evolution,
        })
    }

    /// Analyze pattern evolution between two adjacent layers
    fn analyze_layer_pattern_evolution(
        &self,
        layer1: &LayerAttentionReport,
        layer2: &LayerAttentionReport,
    ) -> LayerPatternEvolution {
        let pattern_changes = Vec::new();

        // Simplified pattern transition analysis
        // In practice, would track specific pattern transitions

        LayerPatternEvolution {
            layer_idx: layer1.layer_idx,
            pattern_changes,
        }
    }

    /// Compute attention flow analysis
    fn compute_attention_flow(
        &self,
        layer_reports: &[LayerAttentionReport],
    ) -> Result<AttentionFlow> {
        let mut information_flow = Vec::new();
        let mut bottlenecks = Vec::new();
        let mut flow_concentration = 0.0;

        // Compute flow between adjacent layers
        for i in 0..layer_reports.len().saturating_sub(1) {
            let flow_strength =
                self.compute_flow_strength(&layer_reports[i], &layer_reports[i + 1]);

            information_flow.push(FlowVector {
                from_layer: layer_reports[i].layer_idx,
                to_layer: layer_reports[i + 1].layer_idx,
                flow_strength,
                flow_direction: FlowDirection::Forward,
            });
        }

        // Detect bottlenecks
        for (idx, layer) in layer_reports.iter().enumerate() {
            if let Some(bottleneck) = self.detect_bottleneck(layer, idx) {
                bottlenecks.push(bottleneck);
            }
        }

        // Compute overall flow concentration
        if !information_flow.is_empty() {
            flow_concentration = information_flow.iter().map(|f| f.flow_strength).sum::<f64>()
                / information_flow.len() as f64;
        }

        Ok(AttentionFlow {
            information_flow,
            bottlenecks,
            flow_concentration,
        })
    }

    /// Compute flow strength between two layers
    fn compute_flow_strength(
        &self,
        layer1: &LayerAttentionReport,
        layer2: &LayerAttentionReport,
    ) -> f64 {
        // Simplified flow computation based on entropy correlation
        self.compute_pattern_similarity(layer1, layer2).abs()
    }

    /// Detect attention bottlenecks
    fn detect_bottleneck(
        &self,
        layer: &LayerAttentionReport,
        layer_idx: usize,
    ) -> Option<AttentionBottleneck> {
        // Detect if there's unusually high concentration in few heads
        let max_concentration =
            layer.head_statistics.iter().map(|s| s.max_attention).fold(0.0, f64::max);

        if max_concentration > 0.9 {
            Some(AttentionBottleneck {
                layer_idx,
                position: 0, // Simplified
                bottleneck_strength: max_concentration,
                affected_heads: layer
                    .head_statistics
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.max_attention > 0.8)
                    .map(|(i, _)| i)
                    .collect(),
            })
        } else {
            None
        }
    }
}

impl AttentionReport {
    /// Print a summary of the attention report
    pub fn print_summary(&self) {
        println!("Attention Analysis Summary");
        println!("=========================");
        println!(
            "Layers: {}, Heads: {}, Sequence Length: {}",
            self.num_layers, self.num_heads, self.sequence_length
        );
        println!(
            "Average Entropy: {:.4}",
            self.global_statistics.total_entropy
        );
        println!(
            "Average Sparsity: {:.4}",
            self.global_statistics.average_sparsity
        );
        println!(
            "Head Diversity: {:.4}",
            self.global_statistics.head_diversity
        );

        println!("\nPattern Summary:");
        for (pattern_type, count) in &self.pattern_summary.pattern_counts {
            println!("  {}: {} occurrences", pattern_type, count);
        }

        println!(
            "\nDominant Patterns: {}",
            self.pattern_summary.dominant_patterns.len()
        );
        println!(
            "Attention Bottlenecks: {}",
            self.attention_flow.bottlenecks.len()
        );
        println!(
            "Flow Concentration: {:.4}",
            self.attention_flow.flow_concentration
        );
    }

    /// Export attention matrices for visualization
    pub fn export_for_visualization(&self, output_dir: &str) -> Result<()> {
        std::fs::create_dir_all(output_dir)?;

        // Export layer-wise statistics
        for layer_report in &self.layer_reports {
            let filename = format!("{}/layer_{}_stats.json", output_dir, layer_report.layer_idx);
            let json = serde_json::to_string_pretty(layer_report)?;
            std::fs::write(filename, json)?;
        }

        // Export global report
        let global_filename = format!("{}/global_report.json", output_dir);
        let global_json = serde_json::to_string_pretty(self)?;
        std::fs::write(global_filename, global_json)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_visualizer_creation() {
        let visualizer = AttentionVisualizer::new();
        assert!(visualizer.config.enabled);
        assert_eq!(visualizer.config.max_layers_to_track, 12);
    }

    #[test]
    fn test_attention_visualizer_config() {
        let config = AttentionVisualizerConfig {
            enabled: true,
            max_layers_to_track: 6,
            max_heads_to_track: 8,
            save_attention_matrices: true,
            compute_statistics: true,
            track_attention_entropy: true,
            visualize_patterns: true,
        };

        let visualizer = AttentionVisualizer::with_config(config.clone());
        assert_eq!(visualizer.config.max_layers_to_track, 6);
        assert_eq!(visualizer.config.max_heads_to_track, 8);
        assert!(visualizer.config.save_attention_matrices);
    }

    #[test]
    fn test_attention_tracking() -> Result<()> {
        let mut visualizer = AttentionVisualizer::new();

        // Create dummy attention tensor
        let attention_data = vec![0.5; 4 * 8 * 16 * 16]; // batch=4, heads=8, seq=16
        let attention_tensor = Tensor::from_vec(attention_data, &[4, 8, 16, 16])?;

        let session_id = "test_session";
        let tokens = vec!["hello".to_string(), "world".to_string()];

        // Start tracking session first
        visualizer.start_tracking(session_id)?;
        visualizer.track_attention(session_id, 0, &attention_tensor, Some(&tokens))?;

        let report = visualizer.get_report(session_id)?;
        assert_eq!(report.session_id, session_id);
        assert_eq!(report.num_layers, 1);

        Ok(())
    }

    #[test]
    fn test_pattern_detection() {
        let visualizer = AttentionVisualizer::new();

        // Create diagonal pattern
        let seq_len = 4;
        let mut head_data = vec![0.0; seq_len * seq_len];
        for i in 0..seq_len {
            head_data[i * seq_len + i] = 1.0; // Diagonal
        }

        let diagonal_strength = visualizer.measure_diagonal_pattern(&head_data, seq_len);
        assert!(diagonal_strength > 0.9); // Should detect strong diagonal pattern
    }

    #[test]
    fn test_head_statistics() -> Result<()> {
        let visualizer = AttentionVisualizer::new();

        // Create uniform attention
        let seq_len = 4;
        let uniform_val = 1.0 / (seq_len as f32);
        let head_data = vec![uniform_val; seq_len * seq_len];

        let stats = visualizer.compute_head_statistics(0, &head_data, seq_len)?;

        assert_eq!(stats.head_idx, 0);
        assert!(stats.entropy > 0.0); // Should have some entropy
        assert_eq!(stats.max_attention, uniform_val as f64);

        Ok(())
    }

    #[test]
    fn test_attention_pattern_types() {
        // Test that all pattern types can be created
        let patterns = [
            AttentionPatternType::Diagonal,
            AttentionPatternType::Vertical,
            AttentionPatternType::Horizontal,
            AttentionPatternType::Block,
            AttentionPatternType::Sparse,
            AttentionPatternType::Global,
            AttentionPatternType::Local,
            AttentionPatternType::Causal,
            AttentionPatternType::Broadcast,
            AttentionPatternType::Hierarchical,
        ];

        for pattern_type in patterns.iter() {
            let pattern = AttentionPattern {
                pattern_type: pattern_type.clone(),
                confidence: 0.8,
                layer_idx: 0,
                head_idx: 0,
                description: "Test pattern".to_string(),
            };

            assert_eq!(pattern.confidence, 0.8);
            assert_eq!(pattern.layer_idx, 0);
        }
    }

    #[test]
    fn test_attention_report_serialization() -> Result<()> {
        let report = AttentionReport::default();

        // Test that the report can be serialized/deserialized
        let json = serde_json::to_string(&report)?;
        let _deserialized: AttentionReport = serde_json::from_str(&json)?;

        Ok(())
    }

    #[test]
    fn test_sparse_pattern_measurement() {
        let visualizer = AttentionVisualizer::new();

        // Create sparse pattern (mostly zeros)
        let seq_len = 4;
        let mut head_data = vec![0.0; seq_len * seq_len];
        head_data[0] = 1.0; // Only one non-zero element

        let sparse_strength = visualizer.measure_sparse_pattern(&head_data, seq_len);
        assert!(sparse_strength > 0.8); // Should detect high sparsity
    }
}
