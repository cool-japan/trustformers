//! Layer-level analysis and activation monitoring.
//!
//! This module provides comprehensive layer-level diagnostics including
//! activation analysis, weight distribution monitoring, attention visualization,
//! and layer health assessment for deep learning models.
// reason: debug/profiling scaffolding — structs are constructed and their fields/methods
// are retained for the data model, serialization completeness, and future consumers that
// do not yet read every member. Consolidated from many item-level #[allow(dead_code)].
#![allow(dead_code)]

use anyhow::{Context, Result};
use std::collections::HashMap;

use super::analytics::{AdvancedAnalytics, HiddenStateData};
use super::types::{
    ActivationHeatmap, AttentionVisualization, HiddenStateAnalysis, LayerActivationStats,
    LayerAnalysis, WeightDistribution,
};

/// Layer analyzer for monitoring and analyzing individual layer behavior.
#[derive(Debug)]
pub struct LayerAnalyzer {
    /// Layer activation statistics history
    layer_activations: HashMap<String, Vec<LayerActivationStats>>,
    /// Layer health monitoring configuration
    config: LayerAnalysisConfig,
    /// Current layer states
    layer_states: HashMap<String, LayerState>,
    /// Real weight tensors (flattened), recorded by the caller via
    /// [`LayerAnalyzer::record_layer_weights`]. Drives
    /// [`LayerAnalyzer::analyze_layer_weight_distribution`] -- no tensor
    /// recorded means no distribution can be reported, honestly.
    layer_weights: HashMap<String, Vec<f64>>,
    /// Real activation grids (e.g. `[batch][feature]` or a spatial slice),
    /// recorded by the caller via [`LayerAnalyzer::record_activation_grid`].
    /// Drives [`LayerAnalyzer::create_activation_heatmap`].
    activation_grids: HashMap<String, Vec<Vec<f64>>>,
    /// Real hidden-state vector history per layer, recorded by the caller via
    /// [`LayerAnalyzer::record_hidden_state_sample`]. Drives
    /// [`LayerAnalyzer::analyze_layer_hidden_states`] (dimensionality,
    /// information content, clustering, temporal dynamics, representation
    /// stability), delegated to [`AdvancedAnalytics`]'s real implementations.
    hidden_state_history: HashMap<String, Vec<HiddenStateData>>,
    /// Real attention-weight matrices recorded per layer, with the tokens
    /// they were computed over. Drives
    /// [`LayerAnalyzer::create_attention_visualization`].
    attention_samples: HashMap<String, AttentionSample>,
    /// Real numeric routines (k-means clustering, temporal-dynamics and
    /// representation-stability statistics) shared with [`AdvancedAnalytics`]
    /// rather than reimplemented here.
    analytics: AdvancedAnalytics,
}

/// A real, caller-supplied attention weight matrix plus the tokens it was
/// computed over.
#[derive(Debug, Clone)]
struct AttentionSample {
    weights: Vec<Vec<f64>>,
    input_tokens: Vec<String>,
    output_tokens: Vec<String>,
}

/// Configuration for layer analysis.
#[derive(Debug, Clone)]
pub struct LayerAnalysisConfig {
    /// Threshold for dead neuron detection
    pub dead_neuron_threshold: f64,
    /// Threshold for saturated neuron detection
    pub saturated_neuron_threshold: f64,
    /// Maximum acceptable activation variance
    pub max_activation_variance: f64,
    /// Minimum acceptable layer health score
    pub min_health_score: f64,
    /// History length for temporal analysis
    pub history_length: usize,
}

impl Default for LayerAnalysisConfig {
    fn default() -> Self {
        Self {
            dead_neuron_threshold: 0.1,
            saturated_neuron_threshold: 0.1,
            max_activation_variance: 2.0,
            min_health_score: 0.7,
            history_length: 100,
        }
    }
}

/// Current state information for a layer.
#[derive(Debug, Clone, Default)]
struct LayerState {
    /// Health score history
    health_scores: Vec<f64>,
    /// Issues detected in the layer
    detected_issues: Vec<String>,
    /// Last analysis timestamp
    last_analysis_step: usize,
}

impl LayerAnalyzer {
    /// Create a new layer analyzer.
    pub fn new() -> Self {
        Self {
            layer_activations: HashMap::new(),
            config: LayerAnalysisConfig::default(),
            layer_states: HashMap::new(),
            layer_weights: HashMap::new(),
            activation_grids: HashMap::new(),
            hidden_state_history: HashMap::new(),
            attention_samples: HashMap::new(),
            analytics: AdvancedAnalytics::new(),
        }
    }

    /// Create a new layer analyzer with custom configuration.
    pub fn with_config(config: LayerAnalysisConfig) -> Self {
        Self {
            layer_activations: HashMap::new(),
            config,
            layer_states: HashMap::new(),
            layer_weights: HashMap::new(),
            activation_grids: HashMap::new(),
            hidden_state_history: HashMap::new(),
            attention_samples: HashMap::new(),
            analytics: AdvancedAnalytics::new(),
        }
    }

    /// Record a real weight tensor (flattened) for a layer. Required before
    /// [`Self::analyze_layer_weight_distribution`] can report anything.
    pub fn record_layer_weights(&mut self, layer_name: &str, weights: Vec<f64>) {
        self.layer_weights.insert(layer_name.to_string(), weights);
    }

    /// Record a real captured activation grid (e.g. `[batch][feature]`, or a
    /// spatial `[height][width]` slice) for a layer. Required before
    /// [`Self::create_activation_heatmap`] can report anything.
    pub fn record_activation_grid(&mut self, layer_name: &str, grid: Vec<Vec<f64>>) {
        self.activation_grids.insert(layer_name.to_string(), grid);
    }

    /// Record one real hidden-state sample (a batch of hidden-state vectors
    /// captured at one point in time/training) for a layer. Accumulates into
    /// that layer's history; required before
    /// [`Self::analyze_layer_hidden_states`] can report anything.
    pub fn record_hidden_state_sample(&mut self, layer_name: &str, hidden_states: Vec<Vec<f64>>) {
        let history = self.hidden_state_history.entry(layer_name.to_string()).or_default();
        let training_step = history.len();
        history.push(HiddenStateData {
            layer_name: layer_name.to_string(),
            hidden_states,
            labels: None,
            timestamp: chrono::Utc::now(),
            training_step,
        });
    }

    /// Record a real attention-weight matrix for a layer. `input_tokens` /
    /// `output_tokens` default to positional labels (`pos_0`, `pos_1`, ...)
    /// when not supplied -- a real (if generic) label derived from the
    /// matrix's own shape, never fabricated content. Required before
    /// [`Self::create_attention_visualization`] can report anything.
    pub fn record_attention_weights(
        &mut self,
        layer_name: &str,
        weights: Vec<Vec<f64>>,
        input_tokens: Option<Vec<String>>,
        output_tokens: Option<Vec<String>>,
    ) {
        let seq_len = weights.len();
        let positional = || (0..seq_len).map(|i| format!("pos_{i}")).collect();
        let input_tokens = input_tokens.unwrap_or_else(positional);
        let output_tokens = output_tokens.unwrap_or_else(positional);
        self.attention_samples.insert(
            layer_name.to_string(),
            AttentionSample {
                weights,
                input_tokens,
                output_tokens,
            },
        );
    }

    /// Record layer activation statistics.
    pub fn record_layer_activations(&mut self, layer_name: &str, stats: LayerActivationStats) {
        // Calculate health score before mutable borrow
        let health_score = self.calculate_layer_health_score(&stats);

        let layer_stats = self.layer_activations.entry(layer_name.to_string()).or_default();
        layer_stats.push(stats);

        // Maintain reasonable history length
        if layer_stats.len() > self.config.history_length {
            layer_stats.remove(0);
        }

        // Update layer state
        let layer_state = self.layer_states.entry(layer_name.to_string()).or_default();
        layer_state.health_scores.push(health_score);

        if layer_state.health_scores.len() > 50 {
            layer_state.health_scores.remove(0);
        }

        layer_state.last_analysis_step += 1;
    }

    /// Record layer statistics (extracts layer name and calls record_layer_activations).
    pub fn record_layer_stats(&mut self, stats: LayerActivationStats) {
        let layer_name = stats.layer_name.clone();
        self.record_layer_activations(&layer_name, stats);
    }

    /// Get layer activation statistics for a specific layer.
    pub fn get_layer_activations(&self, layer_name: &str) -> Option<&[LayerActivationStats]> {
        self.layer_activations.get(layer_name).map(|v| v.as_slice())
    }

    /// Perform comprehensive layer-by-layer analysis.
    pub fn perform_layer_by_layer_analysis(&self) -> Vec<LayerAnalysis> {
        let mut analyses = Vec::new();

        for (layer_name, stats_history) in &self.layer_activations {
            if let Some(latest_stats) = stats_history.last() {
                let analysis = self.analyze_single_layer(layer_name, latest_stats, stats_history);
                analyses.push(analysis);
            }
        }

        analyses.sort_by(|a, b| {
            a.health_score.partial_cmp(&b.health_score).unwrap_or(std::cmp::Ordering::Equal)
        });
        analyses
    }

    /// Analyze a single layer comprehensively.
    pub fn analyze_single_layer(
        &self,
        layer_name: &str,
        current_stats: &LayerActivationStats,
        stats_history: &[LayerActivationStats],
    ) -> LayerAnalysis {
        let layer_type = self.infer_layer_type(layer_name);
        let health_score = self.calculate_layer_health_score(current_stats);
        let issues = self.identify_layer_issues(current_stats, stats_history);
        let recommendations = self.generate_layer_recommendations(&issues, &layer_type);
        let activation_summary = self.generate_activation_summary(current_stats);

        LayerAnalysis {
            layer_name: layer_name.to_string(),
            layer_type,
            health_score,
            issues,
            recommendations,
            activation_summary,
        }
    }

    /// Calculate layer health score.
    pub fn calculate_layer_health_score(&self, stats: &LayerActivationStats) -> f64 {
        let mut score = 1.0;

        // Penalize dead neurons
        if stats.dead_neurons_ratio > self.config.dead_neuron_threshold {
            score -= stats.dead_neurons_ratio * 0.5;
        }

        // Penalize saturated neurons
        if stats.saturated_neurons_ratio > self.config.saturated_neuron_threshold {
            score -= stats.saturated_neurons_ratio * 0.3;
        }

        // Penalize extreme activation ranges
        let activation_range = stats.max_activation - stats.min_activation;
        if activation_range > 10.0 {
            score -= 0.2;
        }

        // Penalize high variance
        if stats.std_activation > self.config.max_activation_variance {
            score -= 0.2;
        }

        // Bonus for good sparsity
        if stats.sparsity > 0.1 && stats.sparsity < 0.8 {
            score += 0.1;
        }

        score.max(0.0).min(1.0)
    }

    /// Identify issues in a layer.
    pub fn identify_layer_issues(
        &self,
        current_stats: &LayerActivationStats,
        stats_history: &[LayerActivationStats],
    ) -> Vec<String> {
        let mut issues = Vec::new();

        // Dead neuron issues
        if current_stats.dead_neurons_ratio > self.config.dead_neuron_threshold {
            issues.push(format!(
                "High dead neuron ratio: {:.1}%",
                current_stats.dead_neurons_ratio * 100.0
            ));
        }

        // Saturated neuron issues
        if current_stats.saturated_neurons_ratio > self.config.saturated_neuron_threshold {
            issues.push(format!(
                "High saturated neuron ratio: {:.1}%",
                current_stats.saturated_neurons_ratio * 100.0
            ));
        }

        // Activation range issues
        if current_stats.max_activation - current_stats.min_activation > 100.0 {
            issues.push("Extremely wide activation range detected".to_string());
        }

        // Variance issues
        if current_stats.std_activation > self.config.max_activation_variance {
            issues.push("High activation variance detected".to_string());
        }

        // Temporal issues (if history is available)
        if stats_history.len() > 5 {
            let variance_trend = self.analyze_variance_trend(stats_history);
            if variance_trend > 0.1 {
                issues.push("Increasing activation variance over time".to_string());
            }
        }

        // Zero activation issues
        if current_stats.mean_activation.abs() < 1e-6 {
            issues.push("Near-zero mean activation detected".to_string());
        }

        issues
    }

    /// Generate recommendations for layer improvement.
    pub fn generate_layer_recommendations(
        &self,
        issues: &[String],
        layer_type: &str,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        for issue in issues {
            if issue.contains("dead neuron") {
                match layer_type {
                    "Linear" => recommendations
                        .push("Consider using LeakyReLU or ELU activation".to_string()),
                    "Convolutional" => recommendations.push(
                        "Consider batch normalization or different initialization".to_string(),
                    ),
                    _ => recommendations.push(
                        "Consider different activation function or initialization".to_string(),
                    ),
                }
            }

            if issue.contains("saturated neuron") {
                recommendations
                    .push("Consider gradient clipping or learning rate reduction".to_string());
                recommendations.push("Consider batch normalization".to_string());
            }

            if issue.contains("activation range") {
                recommendations.push("Consider activation clipping or normalization".to_string());
            }

            if issue.contains("variance") {
                recommendations.push("Consider weight initialization adjustment".to_string());
                recommendations.push("Consider adding regularization".to_string());
            }

            if issue.contains("zero activation") {
                recommendations
                    .push("Check weight initialization and input preprocessing".to_string());
            }
        }

        recommendations.dedup();
        recommendations
    }

    /// Analyze weight distributions for every layer with a real weight
    /// tensor on record (see [`Self::record_layer_weights`]). Layers without
    /// one are simply absent from the result -- never filled with a
    /// fabricated distribution.
    pub fn analyze_weight_distributions(&self) -> HashMap<String, WeightDistribution> {
        let mut distributions = HashMap::new();

        for layer_name in self.layer_weights.keys() {
            if let Ok(distribution) = self.analyze_layer_weight_distribution(layer_name) {
                distributions.insert(layer_name.clone(), distribution);
            }
        }

        distributions
    }

    /// Generate activation heatmaps for every layer with a real recorded
    /// activation grid (see [`Self::record_activation_grid`]).
    pub fn generate_activation_heatmaps(&self) -> HashMap<String, ActivationHeatmap> {
        let mut heatmaps = HashMap::new();

        for layer_name in self.activation_grids.keys() {
            if let Ok(heatmap) = self.create_activation_heatmap(layer_name) {
                heatmaps.insert(layer_name.clone(), heatmap);
            }
        }

        heatmaps
    }

    /// Generate attention visualizations for every layer with real recorded
    /// attention weights (see [`Self::record_attention_weights`]).
    pub fn generate_attention_visualizations(&self) -> HashMap<String, AttentionVisualization> {
        let mut visualizations = HashMap::new();

        for layer_name in self.attention_samples.keys() {
            if let Ok(visualization) = self.create_attention_visualization(layer_name) {
                visualizations.insert(layer_name.clone(), visualization);
            }
        }

        visualizations
    }

    /// Analyze hidden states for every layer with real recorded hidden-state
    /// samples (see [`Self::record_hidden_state_sample`]). A layer is
    /// omitted (never fabricated) when its sample count is too small for a
    /// statistically meaningful analysis -- see
    /// [`Self::analyze_layer_hidden_states`].
    pub fn analyze_hidden_states(&self) -> HashMap<String, HiddenStateAnalysis> {
        let mut analyses = HashMap::new();

        for layer_name in self.hidden_state_history.keys() {
            if let Ok(analysis) = self.analyze_layer_hidden_states(layer_name) {
                analyses.insert(layer_name.clone(), analysis);
            }
        }

        analyses
    }

    // Helper methods

    fn infer_layer_type(&self, layer_name: &str) -> String {
        let name_lower = layer_name.to_lowercase();

        if name_lower.contains("attention") || name_lower.contains("attn") {
            "Attention".to_string()
        } else if name_lower.contains("linear")
            || name_lower.contains("dense")
            || name_lower.contains("fc")
        {
            "Linear".to_string()
        } else if name_lower.contains("conv") {
            "Convolutional".to_string()
        } else if name_lower.contains("norm")
            || name_lower.contains("bn")
            || name_lower.contains("ln")
        {
            "Normalization".to_string()
        } else if name_lower.contains("dropout") {
            "Dropout".to_string()
        } else if name_lower.contains("embed") {
            "Embedding".to_string()
        } else {
            "Unknown".to_string()
        }
    }

    fn generate_activation_summary(&self, stats: &LayerActivationStats) -> String {
        format!(
            "Mean: {:.3}, Std: {:.3}, Range: [{:.3}, {:.3}], Dead: {:.1}%, Saturated: {:.1}%, Sparsity: {:.1}%",
            stats.mean_activation,
            stats.std_activation,
            stats.min_activation,
            stats.max_activation,
            stats.dead_neurons_ratio * 100.0,
            stats.saturated_neurons_ratio * 100.0,
            stats.sparsity * 100.0
        )
    }

    fn analyze_variance_trend(&self, stats_history: &[LayerActivationStats]) -> f64 {
        if stats_history.len() < 2 {
            return 0.0;
        }

        let variances: Vec<f64> = stats_history.iter().map(|s| s.std_activation.powi(2)).collect();
        self.calculate_trend(&variances)
    }

    fn calculate_trend(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = values.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &y) in values.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (y - y_mean);
            denominator += (x - x_mean).powi(2);
        }

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Compute a real weight distribution from the tensor recorded via
    /// [`Self::record_layer_weights`]. Errors (rather than fabricating) when
    /// no tensor has been recorded for this layer.
    fn analyze_layer_weight_distribution(&self, layer_name: &str) -> Result<WeightDistribution> {
        let weights = self.layer_weights.get(layer_name).with_context(|| {
            format!(
                "no weight tensor recorded for layer '{layer_name}'; call \
                 LayerAnalyzer::record_layer_weights first"
            )
        })?;
        if weights.is_empty() {
            anyhow::bail!("recorded weight tensor for layer '{layer_name}' is empty");
        }

        let n = weights.len() as f64;
        let mean = weights.iter().sum::<f64>() / n;
        let variance = weights.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();
        let min = weights.iter().copied().fold(f64::INFINITY, f64::min);
        let max = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let near_zero_count = weights.iter().filter(|w| w.abs() < 1e-6).count();
        let sparsity = near_zero_count as f64 / n;

        // Real (Fisher-Pearson) skewness of the actual data, rather than
        // always reporting "Normal".
        let distribution_shape = if std_dev > 0.0 {
            let skewness = weights.iter().map(|w| ((w - mean) / std_dev).powi(3)).sum::<f64>() / n;
            if skewness > 0.5 {
                "Right-skewed"
            } else if skewness < -0.5 {
                "Left-skewed"
            } else {
                "Approximately symmetric"
            }
        } else {
            "Degenerate (zero variance)"
        }
        .to_string();

        Ok(WeightDistribution {
            mean,
            std_dev,
            min,
            max,
            sparsity,
            distribution_shape,
        })
    }

    /// Build a real activation heatmap from the grid recorded via
    /// [`Self::record_activation_grid`]. Errors (rather than fabricating)
    /// when no grid has been recorded for this layer.
    fn create_activation_heatmap(&self, layer_name: &str) -> Result<ActivationHeatmap> {
        let grid = self.activation_grids.get(layer_name).with_context(|| {
            format!(
                "no activation sample recorded for layer '{layer_name}'; call \
                 LayerAnalyzer::record_activation_grid first"
            )
        })?;
        if grid.is_empty() || grid[0].is_empty() {
            anyhow::bail!("recorded activation grid for layer '{layer_name}' is empty");
        }

        let height = grid.len();
        let width = grid[0].len();
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for row in grid {
            for &v in row {
                min_v = min_v.min(v);
                max_v = max_v.max(v);
            }
        }

        Ok(ActivationHeatmap {
            data: grid.clone(),
            dimensions: (height, width),
            value_range: (min_v, max_v),
            interpretation: format!(
                "Real captured activations for {} layer ({height}x{width})",
                self.infer_layer_type(layer_name)
            ),
        })
    }

    /// Build a real attention visualization from the weights recorded via
    /// [`Self::record_attention_weights`]. Errors (rather than fabricating)
    /// when no weights have been recorded for this layer. `patterns` is a
    /// real, deterministic description derived from the matrix's own
    /// diagonal mass, not a fixed literal list.
    fn create_attention_visualization(&self, layer_name: &str) -> Result<AttentionVisualization> {
        let sample = self.attention_samples.get(layer_name).with_context(|| {
            format!(
                "no attention weights recorded for layer '{layer_name}'; call \
                 LayerAnalyzer::record_attention_weights first"
            )
        })?;
        if sample.weights.is_empty() {
            anyhow::bail!("recorded attention weights for layer '{layer_name}' are empty");
        }

        let seq_len = sample.weights.len();
        let diagonal_mass: f64 =
            (0..seq_len).map(|i| sample.weights[i].get(i).copied().unwrap_or(0.0)).sum();
        let total_mass: f64 = sample.weights.iter().flatten().sum();

        let mut patterns = Vec::new();
        if total_mass > 0.0 {
            let diagonal_ratio = diagonal_mass / total_mass;
            let description = if diagonal_ratio > 0.5 {
                "Strongly self-attending (diagonal-dominant)"
            } else if diagonal_ratio > 0.2 {
                "Partially local/self-attending"
            } else {
                "Diffuse/global attention (low diagonal mass)"
            };
            patterns.push(format!(
                "{description}: {:.1}% of total attention mass on the diagonal",
                diagonal_ratio * 100.0
            ));
        } else {
            patterns.push("All-zero attention weights recorded".to_string());
        }

        Ok(AttentionVisualization {
            attention_weights: sample.weights.clone(),
            input_tokens: sample.input_tokens.clone(),
            output_tokens: sample.output_tokens.clone(),
            patterns,
        })
    }

    /// Compute a real hidden-state analysis from the samples recorded via
    /// [`Self::record_hidden_state_sample`]. Errors (rather than
    /// fabricating) when no samples have been recorded, or when there are
    /// too few for a statistically meaningful analysis (clustering and
    /// temporal-dynamics both require a minimum sample count, enforced by
    /// [`AdvancedAnalytics`] and propagated here rather than worked around).
    fn analyze_layer_hidden_states(&self, layer_name: &str) -> Result<HiddenStateAnalysis> {
        let history = self.hidden_state_history.get(layer_name).with_context(|| {
            format!(
                "no hidden-state samples recorded for layer '{layer_name}'; call \
                 LayerAnalyzer::record_hidden_state_sample first"
            )
        })?;
        if history.is_empty() {
            anyhow::bail!("recorded hidden-state history for layer '{layer_name}' is empty");
        }

        let all_states: Vec<Vec<f64>> =
            history.iter().flat_map(|sample| sample.hidden_states.iter().cloned()).collect();
        if all_states.is_empty() || all_states[0].is_empty() {
            anyhow::bail!(
                "recorded hidden-state samples for layer '{layer_name}' contain no vectors"
            );
        }

        let dimensionality = all_states[0].len();
        let information_content = information_content(&all_states);

        // Real k-means clustering / temporal-dynamics / stability statistics,
        // shared with `AdvancedAnalytics` rather than reimplemented here.
        let clustering_results = self
            .analytics
            .perform_clustering_analysis(&all_states)
            .with_context(|| format!("clustering analysis for layer '{layer_name}'"))?;

        let layer_refs: Vec<&HiddenStateData> = history.iter().collect();
        let temporal_dynamics = self
            .analytics
            .analyze_temporal_dynamics(&layer_refs)
            .with_context(|| format!("temporal-dynamics analysis for layer '{layer_name}'"))?;

        let representation_stability =
            self.analytics.assess_representation_stability(&all_states).with_context(|| {
                format!("representation-stability analysis for layer '{layer_name}'")
            })?;

        Ok(HiddenStateAnalysis {
            dimensionality,
            information_content,
            clustering_results,
            temporal_dynamics,
            representation_stability,
        })
    }

    /// Clear all layer analysis data, including recorded real tensors.
    pub fn clear(&mut self) {
        self.layer_activations.clear();
        self.layer_states.clear();
        self.layer_weights.clear();
        self.activation_grids.clear();
        self.hidden_state_history.clear();
        self.attention_samples.clear();
    }
}

/// Real information-content proxy: total per-dimension variance across the
/// given hidden-state vectors, normalized by dimensionality. Zero for a
/// single (or no) sample, since variance is undefined with fewer than two
/// observations.
fn information_content(hidden_states: &[Vec<f64>]) -> f64 {
    if hidden_states.len() < 2 {
        return 0.0;
    }
    let dimensions = hidden_states[0].len();
    if dimensions == 0 {
        return 0.0;
    }

    let mut total_variance = 0.0;
    for dim in 0..dimensions {
        let values: Vec<f64> =
            hidden_states.iter().filter_map(|state| state.get(dim).copied()).collect();
        if values.len() > 1 {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
            total_variance += variance;
        }
    }
    total_variance / dimensions as f64
}

impl Default for LayerAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_layer_stats(layer_name: &str) -> LayerActivationStats {
        LayerActivationStats {
            layer_name: layer_name.to_string(),
            mean_activation: 0.5,
            std_activation: 0.2,
            min_activation: 0.0,
            max_activation: 1.0,
            dead_neurons_ratio: 0.05,
            saturated_neurons_ratio: 0.03,
            sparsity: 0.3,
            output_shape: vec![128, 256],
        }
    }

    #[test]
    fn test_layer_analyzer_creation() {
        let analyzer = LayerAnalyzer::new();
        assert_eq!(analyzer.layer_activations.len(), 0);
    }

    #[test]
    fn test_record_layer_activations() {
        let mut analyzer = LayerAnalyzer::new();
        let stats = create_test_layer_stats("test_layer");

        analyzer.record_layer_activations("test_layer", stats);
        assert_eq!(analyzer.layer_activations.len(), 1);
        assert!(analyzer.layer_activations.contains_key("test_layer"));
    }

    #[test]
    fn test_layer_health_score_calculation() {
        let analyzer = LayerAnalyzer::new();
        let stats = create_test_layer_stats("test_layer");

        let health_score = analyzer.calculate_layer_health_score(&stats);
        assert!(health_score > 0.0 && health_score <= 1.0);
    }

    #[test]
    fn test_layer_type_inference() {
        let analyzer = LayerAnalyzer::new();

        assert_eq!(analyzer.infer_layer_type("attention_layer"), "Attention");
        assert_eq!(analyzer.infer_layer_type("linear_projection"), "Linear");
        assert_eq!(analyzer.infer_layer_type("conv2d_layer"), "Convolutional");
        assert_eq!(analyzer.infer_layer_type("batch_norm"), "Normalization");
    }

    #[test]
    fn test_issue_identification() {
        let analyzer = LayerAnalyzer::new();
        let mut stats = create_test_layer_stats("test_layer");
        stats.dead_neurons_ratio = 0.2; // High dead neuron ratio

        let issues = analyzer.identify_layer_issues(&stats, &[]);
        assert!(!issues.is_empty());
        assert!(issues[0].contains("dead neuron"));
    }

    #[test]
    fn test_layer_analysis() {
        let analyzer = LayerAnalyzer::new();
        let stats = create_test_layer_stats("attention_layer");
        let history = vec![stats.clone()];

        let analysis = analyzer.analyze_single_layer("attention_layer", &stats, &history);
        assert_eq!(analysis.layer_name, "attention_layer");
        assert_eq!(analysis.layer_type, "Attention");
        assert!(analysis.health_score > 0.0);
    }

    #[test]
    fn test_weight_distribution_errors_without_recorded_weights() {
        let analyzer = LayerAnalyzer::new();
        let result = analyzer.analyze_layer_weight_distribution("unknown_layer");
        assert!(
            result.is_err(),
            "must error rather than fabricate a distribution"
        );
    }

    #[test]
    fn test_weight_distribution_is_computed_from_real_tensor() {
        let mut analyzer = LayerAnalyzer::new();
        // Deterministic tensor: mean 0, known std_dev, known sparsity.
        let weights = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        analyzer.record_layer_weights("fc1", weights.clone());

        let d1 = analyzer.analyze_layer_weight_distribution("fc1").expect("weights recorded");
        let d2 = analyzer.analyze_layer_weight_distribution("fc1").expect("weights recorded");

        // The old implementation drew fresh random numbers on every call; a
        // real computation over the same recorded tensor is deterministic.
        assert_eq!(d1.mean, d2.mean);
        assert_eq!(d1.std_dev, d2.std_dev);

        let n = weights.len() as f64;
        let expected_mean = weights.iter().sum::<f64>() / n;
        let expected_variance =
            weights.iter().map(|w| (w - expected_mean).powi(2)).sum::<f64>() / n;
        assert!((d1.mean - expected_mean).abs() < 1e-12);
        assert!((d1.std_dev - expected_variance.sqrt()).abs() < 1e-12);
        assert_eq!(d1.min, -2.0);
        assert_eq!(d1.max, 2.0);
        assert!((d1.sparsity - 0.2).abs() < 1e-12); // exactly one of five is ~0
    }

    #[test]
    fn test_activation_heatmap_errors_without_recorded_grid() {
        let analyzer = LayerAnalyzer::new();
        let result = analyzer.create_activation_heatmap("unknown_layer");
        assert!(
            result.is_err(),
            "must error rather than fabricate a heatmap"
        );
    }

    #[test]
    fn test_activation_heatmap_reflects_real_grid() {
        let mut analyzer = LayerAnalyzer::new();
        let grid = vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]];
        analyzer.record_activation_grid("conv1", grid.clone());

        let heatmap = analyzer.create_activation_heatmap("conv1").expect("grid recorded");
        assert_eq!(heatmap.data, grid);
        assert_eq!(heatmap.dimensions, (2, 3));
        assert_eq!(heatmap.value_range, (0.0, 5.0));
    }

    #[test]
    fn test_attention_visualization_errors_without_recorded_weights() {
        let analyzer = LayerAnalyzer::new();
        let result = analyzer.create_attention_visualization("unknown_layer");
        assert!(
            result.is_err(),
            "must error rather than fabricate attention weights"
        );
    }

    #[test]
    fn test_attention_visualization_uses_real_weights_and_tokens() {
        let mut analyzer = LayerAnalyzer::new();
        // Diagonal-dominant matrix: each position mostly attends to itself.
        let weights = vec![
            vec![0.9, 0.05, 0.05],
            vec![0.05, 0.9, 0.05],
            vec![0.05, 0.05, 0.9],
        ];
        analyzer.record_attention_weights(
            "self_attn",
            weights.clone(),
            Some(vec!["a".to_string(), "b".to_string(), "c".to_string()]),
            None,
        );

        let viz = analyzer.create_attention_visualization("self_attn").expect("weights recorded");
        assert_eq!(viz.attention_weights, weights);
        assert_eq!(viz.input_tokens, vec!["a", "b", "c"]);
        // No input tokens were supplied for the output side, so it must fall
        // back to real positional labels derived from the matrix shape --
        // never fabricated token text.
        assert_eq!(viz.output_tokens, vec!["pos_0", "pos_1", "pos_2"]);
        assert!(viz.patterns.iter().any(|p| p.contains("self-attending")));
    }

    #[test]
    fn test_hidden_state_analysis_errors_without_recorded_samples() {
        let analyzer = LayerAnalyzer::new();
        let result = analyzer.analyze_layer_hidden_states("unknown_layer");
        assert!(
            result.is_err(),
            "must error rather than fabricate a hidden-state analysis"
        );
    }

    #[test]
    fn test_hidden_state_analysis_computes_real_statistics() {
        let mut analyzer = LayerAnalyzer::new();

        // Two samples (>= 2 for temporal dynamics), 30 vectors each (60 total
        // >= the 50-sample floor `AdvancedAnalytics` requires for clustering
        // to be statistically meaningful), all derived deterministically --
        // never randomly.
        for sample_idx in 0..2 {
            let batch: Vec<Vec<f64>> = (0..30)
                .map(|i| {
                    let x = (sample_idx * 30 + i) as f64;
                    vec![x, x * 2.0, x.sin()]
                })
                .collect();
            analyzer.record_hidden_state_sample("hidden1", batch);
        }

        let analysis = analyzer.analyze_layer_hidden_states("hidden1").expect("samples recorded");
        assert_eq!(analysis.dimensionality, 3);
        assert!(analysis.information_content > 0.0);
        assert!(analysis.clustering_results.num_clusters > 0);
        assert_eq!(analysis.clustering_results.cluster_assignments.len(), 60);
    }
}
