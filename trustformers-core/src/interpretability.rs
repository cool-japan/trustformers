use crate::errors::{TrustformersError, Result};
use crate::tensor::Tensor;
use crate::layers::attention::MultiHeadAttention;
use scirs2_core::ndarray::{Array2, ArrayD, Axis, IxDyn, s};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for interpretability analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpretabilityConfig {
    pub enable_attention_analysis: bool,
    pub enable_gradient_analysis: bool,
    pub enable_activation_analysis: bool,
    pub enable_feature_importance: bool,
    pub save_visualizations: bool,
    pub output_dir: Option<String>,
}

impl Default for InterpretabilityConfig {
    fn default() -> Self {
        Self {
            enable_attention_analysis: true,
            enable_gradient_analysis: true,
            enable_activation_analysis: true,
            enable_feature_importance: true,
            save_visualizations: false,
            output_dir: None,
        }
    }
}

/// Attention pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionPattern {
    pub layer_idx: usize,
    pub head_idx: usize,
    pub attention_weights: Vec<Vec<f32>>,
    pub entropy: f32,
    pub sparsity: f32,
    pub pattern_type: AttentionPatternType,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionPatternType {
    Local,      // Attention focused on nearby tokens
    Global,     // Attention spread across entire sequence
    Diagonal,   // Attention following diagonal pattern
    Vertical,   // Attention focused on specific positions
    Block,      // Attention in block patterns
    Random,     // No clear pattern
}

/// Feature importance analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportance {
    pub token_importance: Vec<f32>,
    pub position_importance: Vec<f32>,
    pub layer_importance: Vec<f32>,
    pub head_importance: Vec<Vec<f32>>, // [layer][head]
}

/// Gradient-based attribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientAttribution {
    pub input_gradients: Vec<f32>,
    pub integrated_gradients: Vec<f32>,
    pub saliency_scores: Vec<f32>,
    pub attribution_method: AttributionMethod,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttributionMethod {
    Gradients,
    IntegratedGradients,
    SmoothGrad,
    GradCam,
    LayerGradCam,
}

/// Activation analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationAnalysis {
    pub neuron_activation_patterns: HashMap<String, Vec<f32>>,
    pub layer_activation_statistics: HashMap<String, ActivationStats>,
    pub dead_neuron_count: HashMap<String, usize>,
    pub activation_clusters: Vec<ActivationCluster>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationStats {
    pub mean: f32,
    pub std: f32,
    pub min: f32,
    pub max: f32,
    pub sparsity: f32,
    pub skewness: f32,
    pub kurtosis: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationCluster {
    pub layer_name: String,
    pub cluster_id: usize,
    pub neuron_indices: Vec<usize>,
    pub centroid: Vec<f32>,
    pub variance: f32,
}

/// Main interpretability analyzer
pub struct InterpretabilityAnalyzer {
    config: InterpretabilityConfig,
    attention_patterns: Vec<AttentionPattern>,
    feature_importance: Option<FeatureImportance>,
    gradient_attribution: Option<GradientAttribution>,
    activation_analysis: Option<ActivationAnalysis>,
}

impl InterpretabilityAnalyzer {
    pub fn new(config: InterpretabilityConfig) -> Self {
        Self {
            config,
            attention_patterns: Vec::new(),
            feature_importance: None,
            gradient_attribution: None,
            activation_analysis: None,
        }
    }

    /// Analyze attention patterns from attention weights
    pub fn analyze_attention_patterns(
        &mut self,
        attention_weights: &[Tensor],
        layer_idx: usize,
    ) -> Result<()> {
        if !self.config.enable_attention_analysis {
            return Ok(());
        }

        for (head_idx, attention_tensor) in attention_weights.iter().enumerate() {
            let pattern = self.extract_attention_pattern(attention_tensor, layer_idx, head_idx)?;
            self.attention_patterns.push(pattern);
        }

        Ok(())
    }

    /// Extract attention pattern from attention weights tensor
    fn extract_attention_pattern(
        &self,
        attention_weights: &Tensor,
        layer_idx: usize,
        head_idx: usize,
    ) -> Result<AttentionPattern> {
        match attention_weights {
            Tensor::F32(arr) => {
                // Convert to 2D matrix for analysis (seq_len x seq_len)
                let shape = arr.shape();
                if shape.len() < 2 {
                    return Err(TrustformersError::invalid_operation(
                        "Attention weights must be at least 2D".into()
                    ));
                }

                let seq_len = shape[shape.len() - 1];
                let attention_matrix = arr.slice(s![.., ..]).to_owned();

                // Convert to nested Vec for serialization
                let attention_weights_vec: Vec<Vec<f32>> = (0..seq_len)
                    .map(|i| {
                        (0..seq_len)
                            .map(|j| {
                                attention_matrix[[i, j]]
                            })
                            .collect()
                    })
                    .collect();

                // Calculate entropy
                let entropy = self.calculate_attention_entropy(&attention_weights_vec);

                // Calculate sparsity
                let sparsity = self.calculate_attention_sparsity(&attention_weights_vec);

                // Determine pattern type
                let pattern_type = self.classify_attention_pattern(&attention_weights_vec);

                Ok(AttentionPattern {
                    layer_idx,
                    head_idx,
                    attention_weights: attention_weights_vec,
                    entropy,
                    sparsity,
                    pattern_type,
                })
            }
            _ => Err(TrustformersError::tensor_op_error("Unsupported tensor type for attention analysis", "analyze_attention")),
        }
    }

    /// Calculate attention entropy
    fn calculate_attention_entropy(&self, attention_weights: &[Vec<f32>]) -> f32 {
        let mut total_entropy = 0.0;
        let seq_len = attention_weights.len();

        for row in attention_weights {
            let mut entropy = 0.0;
            for &weight in row {
                if weight > 1e-8 {
                    entropy -= weight * weight.ln();
                }
            }
            total_entropy += entropy;
        }

        total_entropy / seq_len as f32
    }

    /// Calculate attention sparsity
    fn calculate_attention_sparsity(&self, attention_weights: &[Vec<f32>]) -> f32 {
        let total_elements = attention_weights.len() * attention_weights[0].len();
        let mut zero_count = 0;

        for row in attention_weights {
            for &weight in row {
                if weight < 1e-6 {
                    zero_count += 1;
                }
            }
        }

        zero_count as f32 / total_elements as f32
    }

    /// Classify attention pattern type
    fn classify_attention_pattern(&self, attention_weights: &[Vec<f32>]) -> AttentionPatternType {
        let seq_len = attention_weights.len();

        // Calculate various pattern scores
        let local_score = self.calculate_local_pattern_score(attention_weights);
        let diagonal_score = self.calculate_diagonal_pattern_score(attention_weights);
        let vertical_score = self.calculate_vertical_pattern_score(attention_weights);
        let block_score = self.calculate_block_pattern_score(attention_weights);

        // Determine dominant pattern
        let scores = vec![
            (local_score, AttentionPatternType::Local),
            (diagonal_score, AttentionPatternType::Diagonal),
            (vertical_score, AttentionPatternType::Vertical),
            (block_score, AttentionPatternType::Block),
        ];

        let (max_score, pattern_type) = scores.into_iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).expect("Partial comparison failed"))
            .unwrap();

        if max_score > 0.3 {
            pattern_type
        } else {
            // Check if it's global or random
            let global_score = self.calculate_global_pattern_score(attention_weights);
            if global_score > 0.5 {
                AttentionPatternType::Global
            } else {
                AttentionPatternType::Random
            }
        }
    }

    fn calculate_local_pattern_score(&self, attention_weights: &[Vec<f32>]) -> f32 {
        let seq_len = attention_weights.len();
        let mut local_score = 0.0;
        let window_size = 5; // Local window

        for i in 0..seq_len {
            let start = (i as i32 - window_size as i32 / 2).max(0) as usize;
            let end = (i + window_size / 2 + 1).min(seq_len);

            let local_sum: f32 = attention_weights[i][start..end].iter().sum();
            local_score += local_sum;
        }

        local_score / seq_len as f32
    }

    fn calculate_diagonal_pattern_score(&self, attention_weights: &[Vec<f32>]) -> f32 {
        let seq_len = attention_weights.len();
        let mut diagonal_score = 0.0;

        for i in 0..seq_len {
            if i < seq_len {
                diagonal_score += attention_weights[i][i];
            }
        }

        diagonal_score / seq_len as f32
    }

    fn calculate_vertical_pattern_score(&self, attention_weights: &[Vec<f32>]) -> f32 {
        let seq_len = attention_weights.len();
        let mut max_col_sum = 0.0;

        for j in 0..seq_len {
            let col_sum: f32 = attention_weights.iter().map(|row| row[j]).sum();
            max_col_sum = max_col_sum.max(col_sum);
        }

        max_col_sum / seq_len as f32
    }

    fn calculate_block_pattern_score(&self, attention_weights: &[Vec<f32>]) -> f32 {
        let seq_len = attention_weights.len();
        let block_size = seq_len / 4; // Quarter of sequence
        if block_size == 0 { return 0.0; }

        let mut block_score = 0.0;
        let num_blocks = seq_len / block_size;

        for block_i in 0..num_blocks {
            for block_j in 0..num_blocks {
                let start_i = block_i * block_size;
                let end_i = (start_i + block_size).min(seq_len);
                let start_j = block_j * block_size;
                let end_j = (start_j + block_size).min(seq_len);

                let mut block_sum = 0.0;
                for i in start_i..end_i {
                    for j in start_j..end_j {
                        block_sum += attention_weights[i][j];
                    }
                }

                block_score = block_score.max(block_sum);
            }
        }

        block_score / (num_blocks * block_size) as f32
    }

    fn calculate_global_pattern_score(&self, attention_weights: &[Vec<f32>]) -> f32 {
        let seq_len = attention_weights.len();
        let mut variance_sum = 0.0;

        for row in attention_weights {
            let mean: f32 = row.iter().sum::<f32>() / seq_len as f32;
            let variance: f32 = row.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / seq_len as f32;
            variance_sum += variance;
        }

        let avg_variance = variance_sum / seq_len as f32;
        1.0 / (1.0 + avg_variance * 100.0) // Lower variance = more global
    }

    /// Analyze feature importance using various methods
    pub fn analyze_feature_importance(
        &mut self,
        inputs: &Tensor,
        outputs: &Tensor,
        model_fn: &dyn Fn(&Tensor) -> Result<Tensor>,
    ) -> Result<()> {
        if !self.config.enable_feature_importance {
            return Ok(());
        }

        let token_importance = self.calculate_token_importance(inputs, outputs, model_fn)?;
        let position_importance = self.calculate_position_importance(inputs, outputs, model_fn)?;
        let layer_importance = vec![1.0; 12]; // Placeholder - would need layer-wise analysis
        let head_importance = vec![vec![1.0; 8]; 12]; // Placeholder

        self.feature_importance = Some(FeatureImportance {
            token_importance,
            position_importance,
            layer_importance,
            head_importance,
        });

        Ok(())
    }

    /// Calculate token-level importance using occlusion
    fn calculate_token_importance(
        &self,
        inputs: &Tensor,
        original_output: &Tensor,
        model_fn: &dyn Fn(&Tensor) -> Result<Tensor>,
    ) -> Result<Vec<f32>> {
        match inputs {
            Tensor::F32(arr) => {
                let shape = arr.shape();
                let seq_len = shape[shape.len() - 1];
                let mut importance_scores = Vec::with_capacity(seq_len);

                for i in 0..seq_len {
                    // Create occluded input
                    let mut occluded_input = arr.clone();

                    // Zero out token i
                    if shape.len() == 2 {
                        occluded_input[[0, i]] = 0.0;
                    } else if shape.len() == 3 {
                        for j in 0..shape[1] {
                            occluded_input[[0, j, i]] = 0.0;
                        }
                    }

                    let occluded_tensor = Tensor::F32(occluded_input);
                    let occluded_output = model_fn(&occluded_tensor)?;

                    // Calculate importance as difference in output
                    let importance = self.calculate_output_difference(original_output, &occluded_output)?;
                    importance_scores.push(importance);
                }

                Ok(importance_scores)
            }
            _ => Err(TrustformersError::tensor_op_error("Unsupported tensor type for token importance", "analyze_token_importance")),
        }
    }

    /// Calculate position-level importance
    fn calculate_position_importance(
        &self,
        inputs: &Tensor,
        original_output: &Tensor,
        model_fn: &dyn Fn(&Tensor) -> Result<Tensor>,
    ) -> Result<Vec<f32>> {
        match inputs {
            Tensor::F32(arr) => {
                let shape = arr.shape();
                let seq_len = shape[shape.len() - 1];
                let mut importance_scores = Vec::with_capacity(seq_len);

                for pos in 0..seq_len {
                    // Create position-masked input
                    let mut masked_input = arr.clone();

                    // Apply position mask
                    if shape.len() == 2 {
                        masked_input[[0, pos]] *= 0.1; // Reduce but don't zero
                    } else if shape.len() == 3 {
                        for j in 0..shape[1] {
                            masked_input[[0, j, pos]] *= 0.1;
                        }
                    }

                    let masked_tensor = Tensor::F32(masked_input);
                    let masked_output = model_fn(&masked_tensor)?;

                    let importance = self.calculate_output_difference(original_output, &masked_output)?;
                    importance_scores.push(importance);
                }

                Ok(importance_scores)
            }
            _ => Err(TrustformersError::tensor_op_error("Unsupported tensor type for position importance", "analyze_position_importance")),
        }
    }

    /// Calculate difference between two outputs
    fn calculate_output_difference(&self, original: &Tensor, modified: &Tensor) -> Result<f32> {
        match (original, modified) {
            (Tensor::F32(orig), Tensor::F32(modif)) => {
                let diff_sum: f32 = orig.iter()
                    .zip(modif.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum();
                Ok(diff_sum / orig.len() as f32)
            }
            _ => Err(TrustformersError::invalid_operation("Tensor type mismatch in output difference".into())),
        }
    }

    /// Perform gradient-based attribution analysis
    pub fn analyze_gradient_attribution(
        &mut self,
        inputs: &Tensor,
        gradients: &Tensor,
        method: AttributionMethod,
    ) -> Result<()> {
        if !self.config.enable_gradient_analysis {
            return Ok(());
        }

        let input_gradients = self.extract_input_gradients(gradients)?;
        let integrated_gradients = self.calculate_integrated_gradients(inputs, gradients)?;
        let saliency_scores = self.calculate_saliency_scores(&input_gradients);

        self.gradient_attribution = Some(GradientAttribution {
            input_gradients,
            integrated_gradients,
            saliency_scores,
            attribution_method: method,
        });

        Ok(())
    }

    /// Extract input gradients from gradient tensor
    fn extract_input_gradients(&self, gradients: &Tensor) -> Result<Vec<f32>> {
        match gradients {
            Tensor::F32(arr) => {
                Ok(arr.iter().cloned().collect())
            }
            _ => Err(TrustformersError::invalid_operation("Unsupported tensor type for gradient extraction".into())),
        }
    }

    /// Calculate integrated gradients
    fn calculate_integrated_gradients(&self, inputs: &Tensor, gradients: &Tensor) -> Result<Vec<f32>> {
        // Simplified integrated gradients - in practice would need multiple evaluations
        match (inputs, gradients) {
            (Tensor::F32(inp), Tensor::F32(grad)) => {
                let integrated: Vec<f32> = inp.iter()
                    .zip(grad.iter())
                    .map(|(input, gradient)| input * gradient)
                    .collect();
                Ok(integrated)
            }
            _ => Err(TrustformersError::invalid_operation("Tensor type mismatch in integrated gradients".into())),
        }
    }

    /// Calculate saliency scores from gradients
    fn calculate_saliency_scores(&self, gradients: &[f32]) -> Vec<f32> {
        gradients.iter().map(|&grad| grad.abs()).collect()
    }

    /// Analyze activation patterns
    pub fn analyze_activations(
        &mut self,
        activations: &HashMap<String, Tensor>,
    ) -> Result<()> {
        if !self.config.enable_activation_analysis {
            return Ok(());
        }

        let mut neuron_activation_patterns = HashMap::new();
        let mut layer_activation_statistics = HashMap::new();
        let mut dead_neuron_count = HashMap::new();
        let mut activation_clusters = Vec::new();

        for (layer_name, activation_tensor) in activations {
            // Extract activation patterns
            let patterns = self.extract_activation_patterns(activation_tensor)?;
            neuron_activation_patterns.insert(layer_name.clone(), patterns);

            // Calculate statistics
            let stats = self.calculate_activation_statistics(activation_tensor)?;
            layer_activation_statistics.insert(layer_name.clone(), stats);

            // Count dead neurons
            let dead_count = self.count_dead_neurons(activation_tensor)?;
            dead_neuron_count.insert(layer_name.clone(), dead_count);

            // Perform clustering analysis
            let clusters = self.cluster_activations(layer_name, activation_tensor)?;
            activation_clusters.extend(clusters);
        }

        self.activation_analysis = Some(ActivationAnalysis {
            neuron_activation_patterns,
            layer_activation_statistics,
            dead_neuron_count,
            activation_clusters,
        });

        Ok(())
    }

    /// Extract activation patterns from tensor
    fn extract_activation_patterns(&self, tensor: &Tensor) -> Result<Vec<f32>> {
        match tensor {
            Tensor::F32(arr) => {
                // Calculate mean activation per neuron
                let shape = arr.shape();
                if shape.len() < 2 {
                    return Ok(vec![0.0]);
                }

                let num_neurons = shape[shape.len() - 1];
                let mut patterns = Vec::with_capacity(num_neurons);

                for neuron_idx in 0..num_neurons {
                    let mut sum = 0.0;
                    let mut count = 0;

                    // Sum activations across batch and sequence dimensions
                    for elem in arr.iter() {
                        sum += elem;
                        count += 1;
                    }

                    patterns.push(if count > 0 { sum / count as f32 } else { 0.0 });
                }

                Ok(patterns)
            }
            _ => Err(TrustformersError::invalid_operation("Unsupported tensor type for activation analysis".into())),
        }
    }

    /// Calculate activation statistics
    fn calculate_activation_statistics(&self, tensor: &Tensor) -> Result<ActivationStats> {
        match tensor {
            Tensor::F32(arr) => {
                let data: Vec<f32> = arr.iter().cloned().collect();

                if data.is_empty() {
                    return Ok(ActivationStats {
                        mean: 0.0,
                        std: 0.0,
                        min: 0.0,
                        max: 0.0,
                        sparsity: 0.0,
                        skewness: 0.0,
                        kurtosis: 0.0,
                    });
                }

                let mean = data.iter().sum::<f32>() / data.len() as f32;
                let variance = data.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f32>() / data.len() as f32;
                let std = variance.sqrt();

                let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                let zero_count = data.iter().filter(|&&x| x.abs() < 1e-6).count();
                let sparsity = zero_count as f32 / data.len() as f32;

                // Calculate skewness and kurtosis
                let skewness = if std > 0.0 {
                    data.iter()
                        .map(|x| ((x - mean) / std).powi(3))
                        .sum::<f32>() / data.len() as f32
                } else { 0.0 };

                let kurtosis = if std > 0.0 {
                    data.iter()
                        .map(|x| ((x - mean) / std).powi(4))
                        .sum::<f32>() / data.len() as f32 - 3.0
                } else { 0.0 };

                Ok(ActivationStats {
                    mean,
                    std,
                    min,
                    max,
                    sparsity,
                    skewness,
                    kurtosis,
                })
            }
            _ => Err(TrustformersError::invalid_operation("Unsupported tensor type for activation statistics".into())),
        }
    }

    /// Count dead neurons (always zero activation)
    fn count_dead_neurons(&self, tensor: &Tensor) -> Result<usize> {
        match tensor {
            Tensor::F32(arr) => {
                let shape = arr.shape();
                if shape.is_empty() {
                    return Ok(0);
                }

                let num_neurons = shape[shape.len() - 1];
                let mut dead_count = 0;

                for neuron_idx in 0..num_neurons {
                    let mut is_dead = true;

                    // Check if neuron ever activates
                    for val in arr.iter() {
                        if val.abs() > 1e-6 {
                            is_dead = false;
                            break;
                        }
                    }

                    if is_dead {
                        dead_count += 1;
                    }
                }

                Ok(dead_count)
            }
            _ => Err(TrustformersError::invalid_operation("Unsupported tensor type for dead neuron counting".into())),
        }
    }

    /// Cluster activations to find similar patterns
    fn cluster_activations(&self, layer_name: &str, tensor: &Tensor) -> Result<Vec<ActivationCluster>> {
        // Simplified k-means clustering
        let k = 3; // Number of clusters
        let mut clusters = Vec::with_capacity(k);

        match tensor {
            Tensor::F32(arr) => {
                let shape = arr.shape();
                if shape.is_empty() {
                    return Ok(clusters);
                }

                let num_neurons = shape[shape.len() - 1];

                // For simplicity, create clusters based on activation magnitude
                for cluster_id in 0..k {
                    let mut neuron_indices = Vec::new();
                    let threshold_min = cluster_id as f32 / k as f32;
                    let threshold_max = (cluster_id + 1) as f32 / k as f32;

                    for neuron_idx in 0..num_neurons {
                        let activation_sum: f32 = arr.iter().sum();
                        let normalized = activation_sum / arr.len() as f32;

                        if normalized >= threshold_min && normalized < threshold_max {
                            neuron_indices.push(neuron_idx);
                        }
                    }

                    if !neuron_indices.is_empty() {
                        clusters.push(ActivationCluster {
                            layer_name: layer_name.to_string(),
                            cluster_id,
                            neuron_indices,
                            centroid: vec![0.5; 10], // Placeholder centroid
                            variance: 0.1,
                        });
                    }
                }

                Ok(clusters)
            }
            _ => Err(TrustformersError::invalid_operation("Unsupported tensor type for activation clustering".into())),
        }
    }

    /// Generate comprehensive interpretability report
    pub fn generate_report(&self) -> InterpretabilityReport {
        InterpretabilityReport {
            attention_patterns: self.attention_patterns.clone(),
            feature_importance: self.feature_importance.clone(),
            gradient_attribution: self.gradient_attribution.clone(),
            activation_analysis: self.activation_analysis.clone(),
            summary: self.generate_summary(),
        }
    }

    /// Generate summary of interpretability analysis
    fn generate_summary(&self) -> InterpretabilitySummary {
        let total_attention_patterns = self.attention_patterns.len();
        let avg_attention_entropy = if !self.attention_patterns.is_empty() {
            self.attention_patterns.iter()
                .map(|p| p.entropy)
                .sum::<f32>() / self.attention_patterns.len() as f32
        } else { 0.0 };

        let most_important_tokens = self.feature_importance
            .as_ref()
            .map(|fi| {
                fi.token_importance.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).expect("Partial comparison failed"))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .unwrap_or(0);

        let total_dead_neurons = self.activation_analysis
            .as_ref()
            .map(|aa| aa.dead_neuron_count.values().sum::<usize>())
            .unwrap_or(0);

        InterpretabilitySummary {
            total_attention_patterns,
            avg_attention_entropy,
            most_important_tokens,
            total_dead_neurons,
            has_gradient_attribution: self.gradient_attribution.is_some(),
            has_feature_importance: self.feature_importance.is_some(),
        }
    }

    /// Get attention patterns
    pub fn get_attention_patterns(&self) -> &[AttentionPattern] {
        &self.attention_patterns
    }

    /// Get feature importance results
    pub fn get_feature_importance(&self) -> Option<&FeatureImportance> {
        self.feature_importance.as_ref()
    }

    /// Get gradient attribution results
    pub fn get_gradient_attribution(&self) -> Option<&GradientAttribution> {
        self.gradient_attribution.as_ref()
    }

    /// Get activation analysis results
    pub fn get_activation_analysis(&self) -> Option<&ActivationAnalysis> {
        self.activation_analysis.as_ref()
    }
}

/// Complete interpretability report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpretabilityReport {
    pub attention_patterns: Vec<AttentionPattern>,
    pub feature_importance: Option<FeatureImportance>,
    pub gradient_attribution: Option<GradientAttribution>,
    pub activation_analysis: Option<ActivationAnalysis>,
    pub summary: InterpretabilitySummary,
}

/// Summary of interpretability analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpretabilitySummary {
    pub total_attention_patterns: usize,
    pub avg_attention_entropy: f32,
    pub most_important_tokens: usize,
    pub total_dead_neurons: usize,
    pub has_gradient_attribution: bool,
    pub has_feature_importance: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpretability_analyzer_creation() {
        let config = InterpretabilityConfig::default();
        let analyzer = InterpretabilityAnalyzer::new(config);

        assert_eq!(analyzer.attention_patterns.len(), 0);
        assert!(analyzer.feature_importance.is_none());
        assert!(analyzer.gradient_attribution.is_none());
        assert!(analyzer.activation_analysis.is_none());
    }

    #[test]
    fn test_attention_pattern_analysis() {
        let config = InterpretabilityConfig::default();
        let mut analyzer = InterpretabilityAnalyzer::new(config);

        // Create dummy attention weights
        let attention_data = vec![0.1, 0.2, 0.3, 0.4];
        let attention_tensor = Tensor::from_vec(attention_data, &[2, 2]).expect("Tensor from_vec failed");
        let attention_weights = vec![attention_tensor];

        let result = analyzer.analyze_attention_patterns(&attention_weights, 0);
        assert!(result.is_ok());
        assert_eq!(analyzer.attention_patterns.len(), 1);
    }

    #[test]
    fn test_attention_entropy_calculation() {
        let config = InterpretabilityConfig::default();
        let analyzer = InterpretabilityAnalyzer::new(config);

        let attention_weights = vec![
            vec![0.5, 0.3, 0.2],
            vec![0.1, 0.8, 0.1],
            vec![0.3, 0.3, 0.4],
        ];

        let entropy = analyzer.calculate_attention_entropy(&attention_weights);
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_attention_sparsity_calculation() {
        let config = InterpretabilityConfig::default();
        let analyzer = InterpretabilityAnalyzer::new(config);

        let attention_weights = vec![
            vec![0.5, 0.0, 0.0],
            vec![0.0, 0.8, 0.0],
            vec![0.0, 0.0, 0.4],
        ];

        let sparsity = analyzer.calculate_attention_sparsity(&attention_weights);
        assert!(sparsity > 0.0);
        assert!(sparsity < 1.0);
    }

    #[test]
    fn test_attention_pattern_classification() {
        let config = InterpretabilityConfig::default();
        let analyzer = InterpretabilityAnalyzer::new(config);

        // Diagonal pattern
        let diagonal_weights = vec![
            vec![0.8, 0.1, 0.1],
            vec![0.1, 0.8, 0.1],
            vec![0.1, 0.1, 0.8],
        ];

        let pattern_type = analyzer.classify_attention_pattern(&diagonal_weights);
        assert_eq!(pattern_type, AttentionPatternType::Diagonal);
    }

    #[test]
    fn test_activation_statistics() {
        let config = InterpretabilityConfig::default();
        let analyzer = InterpretabilityAnalyzer::new(config);

        let activation_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0, 0.0];
        let activation_tensor = Tensor::from_vec(activation_data, &[2, 4]).expect("Tensor from_vec failed");

        let stats = analyzer.calculate_activation_statistics(&activation_tensor).unwrap();
        assert!(stats.mean > 0.0);
        assert!(stats.std > 0.0);
        assert!(stats.sparsity > 0.0);
    }

    #[test]
    fn test_dead_neuron_detection() {
        let config = InterpretabilityConfig::default();
        let analyzer = InterpretabilityAnalyzer::new(config);

        // Create tensor with some dead neurons (all zeros)
        let activation_data = vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0];
        let activation_tensor = Tensor::from_vec(activation_data, &[2, 4]).expect("Tensor from_vec failed");

        let dead_count = analyzer.count_dead_neurons(&activation_tensor).unwrap();
        assert!(dead_count > 0);
    }

    #[test]
    fn test_gradient_attribution_analysis() {
        let config = InterpretabilityConfig::default();
        let mut analyzer = InterpretabilityAnalyzer::new(config);

        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let gradient_data = vec![0.1, 0.2, 0.3, 0.4];

        let inputs = Tensor::from_vec(input_data, &[2, 2]).expect("Tensor from_vec failed");
        let gradients = Tensor::from_vec(gradient_data, &[2, 2]).expect("Tensor from_vec failed");

        let result = analyzer.analyze_gradient_attribution(&inputs, &gradients, AttributionMethod::Gradients);
        assert!(result.is_ok());
        assert!(analyzer.gradient_attribution.is_some());
    }

    #[test]
    fn test_comprehensive_activation_analysis() {
        let config = InterpretabilityConfig::default();
        let mut analyzer = InterpretabilityAnalyzer::new(config);

        let mut activations = HashMap::new();
        let activation_data = vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0];
        let activation_tensor = Tensor::from_vec(activation_data, &[2, 4]).expect("Tensor from_vec failed");
        activations.insert("layer1".to_string(), activation_tensor);

        let result = analyzer.analyze_activations(&activations);
        assert!(result.is_ok());

        let analysis = analyzer.get_activation_analysis().unwrap();
        assert!(analysis.layer_activation_statistics.contains_key("layer1"));
        assert!(analysis.dead_neuron_count.contains_key("layer1"));
    }

    #[test]
    fn test_interpretability_report_generation() {
        let config = InterpretabilityConfig::default();
        let analyzer = InterpretabilityAnalyzer::new(config);

        let report = analyzer.generate_report();
        assert_eq!(report.attention_patterns.len(), 0);
        assert!(report.feature_importance.is_none());
        assert!(report.gradient_attribution.is_none());
        assert!(report.activation_analysis.is_none());
    }

    #[test]
    fn test_config_serialization() {
        let config = InterpretabilityConfig {
            enable_attention_analysis: true,
            enable_gradient_analysis: false,
            enable_activation_analysis: true,
            enable_feature_importance: false,
            save_visualizations: true,
            output_dir: Some("/tmp/interpretability".to_string()),
        };

        let serialized = serde_json::to_string(&config).expect("JSON serialization failed");
        let deserialized: InterpretabilityConfig = serde_json::from_str(&serialized).expect("JSON deserialization failed");

        assert_eq!(config.enable_attention_analysis, deserialized.enable_attention_analysis);
        assert_eq!(config.enable_gradient_analysis, deserialized.enable_gradient_analysis);
        assert_eq!(config.save_visualizations, deserialized.save_visualizations);
        assert_eq!(config.output_dir, deserialized.output_dir);
    }
}