//! Model interpretability analysis.
//!
//! Provides attention-pattern classification, occlusion-based feature
//! importance, gradient attribution and activation statistics.
//!
//! Every metric produced by this module is computed from data the caller
//! supplies. Methods that would need information the caller cannot provide
//! (for example per-layer ablation without a per-layer forward hook) return an
//! empty result rather than a synthesised constant.

use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use scirs2_core::ndarray::s;
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
    Local,    // Attention focused on nearby tokens
    Global,   // Attention spread across entire sequence
    Diagonal, // Attention following diagonal pattern
    Vertical, // Attention focused on specific positions
    Block,    // Attention in block patterns
    Random,   // No clear pattern
}

/// Feature importance analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportance {
    /// Per-token occlusion importance (zeroing each token and measuring the
    /// mean absolute output change).
    pub token_importance: Vec<f32>,
    /// Per-position masking importance (scaling each position by 0.1 and
    /// measuring the mean absolute output change).
    pub position_importance: Vec<f32>,
    /// Per-layer attention concentration, averaged over the heads recorded for
    /// that layer. `1 - H(row) / ln(seq_len)`, i.e. 0 for uniform attention and
    /// 1 for a one-hot row. Empty when no attention patterns were recorded via
    /// [`InterpretabilityAnalyzer::analyze_attention_patterns`].
    pub layer_importance: Vec<f32>,
    /// Per-head attention concentration, indexed `[layer][head]`. Empty when no
    /// attention patterns were recorded.
    pub head_importance: Vec<Vec<f32>>, // [layer][head]
}

/// Gradient-based attribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientAttribution {
    /// Raw input gradients, flattened.
    pub input_gradients: Vec<f32>,
    /// Element-wise `input * gradient` ("gradient x input" attribution).
    ///
    /// This is *not* integrated gradients: it is a single-point attribution.
    /// Use [`InterpretabilityAnalyzer::compute_integrated_gradients`] for a
    /// real Riemann-sum path integral against a baseline.
    pub gradient_x_input: Vec<f32>,
    /// Integrated gradients along a baseline->input path. Only populated by
    /// [`InterpretabilityAnalyzer::analyze_integrated_gradients`]; `None`
    /// otherwise.
    pub integrated_gradients: Option<Vec<f32>>,
    /// Absolute value of the input gradients.
    pub saliency_scores: Vec<f32>,
    /// Which attribution method produced this result.
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
                        "Attention weights must be at least 2D".into(),
                    ));
                }

                let seq_len = shape[shape.len() - 1];
                let attention_matrix = arr.slice(s![.., ..]).to_owned();

                // Convert to nested Vec for serialization
                let attention_weights_vec: Vec<Vec<f32>> = (0..seq_len)
                    .map(|i| (0..seq_len).map(|j| attention_matrix[[i, j]]).collect())
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
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor type for attention analysis",
                "analyze_attention",
            )),
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
        // Calculate various pattern scores
        let local_score = self.calculate_local_pattern_score(attention_weights);
        let diagonal_score = self.calculate_diagonal_pattern_score(attention_weights);
        let vertical_score = self.calculate_vertical_pattern_score(attention_weights);
        let block_score = self.calculate_block_pattern_score(attention_weights);

        // Determine dominant pattern. `max_by` returns the *last* of several
        // equal maxima, so the list is ordered least-specific first: a matrix
        // that is simultaneously a one-column block and a vertical pattern is
        // reported as `Vertical`.
        let scores = [
            (block_score, AttentionPatternType::Block),
            (local_score, AttentionPatternType::Local),
            (diagonal_score, AttentionPatternType::Diagonal),
            (vertical_score, AttentionPatternType::Vertical),
        ];

        let (max_score, pattern_type) = scores
            .into_iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0.0, AttentionPatternType::Random));

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

    /// Normalise an observed attention mass against the mass a *uniform* row
    /// would place on the same number of cells.
    ///
    /// Returns the "excess over uniform", rescaled to `[0, 1]`:
    /// `(observed - uniform) / (1 - uniform)`. This makes the four pattern
    /// scores directly comparable: without it a window that happens to cover
    /// the whole row scores 1.0 for every possible attention matrix, and short
    /// sequences are always classified `Local`.
    fn excess_over_uniform(observed: f32, cells: usize, seq_len: usize) -> f32 {
        if seq_len == 0 || cells == 0 {
            return 0.0;
        }
        let uniform = cells as f32 / seq_len as f32;
        let denominator = 1.0 - uniform;
        if denominator <= f32::EPSILON {
            // The region covers the whole row: it carries no information.
            return 0.0;
        }
        ((observed - uniform) / denominator).clamp(0.0, 1.0)
    }

    fn calculate_local_pattern_score(&self, attention_weights: &[Vec<f32>]) -> f32 {
        let seq_len = attention_weights.len();
        if seq_len == 0 {
            return 0.0;
        }
        let window_size = 5usize.min(seq_len); // Local window, clamped to the sequence
        let half = window_size / 2;

        let mut score_sum = 0.0;
        for (i, row) in attention_weights.iter().enumerate() {
            let start = i.saturating_sub(half);
            let end = (i + half + 1).min(seq_len);
            let cells = end - start;
            let local_mass: f32 = row[start..end].iter().sum();
            score_sum += Self::excess_over_uniform(local_mass, cells, seq_len);
        }

        score_sum / seq_len as f32
    }

    fn calculate_diagonal_pattern_score(&self, attention_weights: &[Vec<f32>]) -> f32 {
        let seq_len = attention_weights.len();
        if seq_len == 0 {
            return 0.0;
        }
        let mut score_sum = 0.0;

        for (i, row) in attention_weights.iter().enumerate() {
            let diagonal_mass = row.get(i).copied().unwrap_or(0.0);
            score_sum += Self::excess_over_uniform(diagonal_mass, 1, seq_len);
        }

        score_sum / seq_len as f32
    }

    fn calculate_vertical_pattern_score(&self, attention_weights: &[Vec<f32>]) -> f32 {
        let seq_len = attention_weights.len();
        if seq_len == 0 {
            return 0.0;
        }
        let mut max_col_mean: f32 = 0.0;

        for j in 0..seq_len {
            let col_sum: f32 =
                attention_weights.iter().map(|row| row.get(j).copied().unwrap_or(0.0)).sum();
            max_col_mean = max_col_mean.max(col_sum / seq_len as f32);
        }

        Self::excess_over_uniform(max_col_mean, 1, seq_len)
    }

    fn calculate_block_pattern_score(&self, attention_weights: &[Vec<f32>]) -> f32 {
        let seq_len = attention_weights.len();
        let block_size = seq_len / 4; // Quarter of sequence
        if block_size == 0 {
            return 0.0;
        }

        let mut best_block_mean: f32 = 0.0;
        let num_blocks = seq_len / block_size;

        for block_i in 0..num_blocks {
            for block_j in 0..num_blocks {
                let start_i = block_i * block_size;
                let end_i = (start_i + block_size).min(seq_len);
                let start_j = block_j * block_size;
                let end_j = (start_j + block_size).min(seq_len);

                let mut block_sum = 0.0;
                for row in attention_weights.iter().take(end_i).skip(start_i) {
                    for j in start_j..end_j {
                        block_sum += row.get(j).copied().unwrap_or(0.0);
                    }
                }

                // Mean attention mass placed by each row of the block inside it.
                let rows = (end_i - start_i).max(1);
                best_block_mean = best_block_mean.max(block_sum / rows as f32);
            }
        }

        Self::excess_over_uniform(best_block_mean, block_size, seq_len)
    }

    /// Global-pattern score: mean row entropy normalised by `ln(seq_len)`.
    ///
    /// 1.0 for perfectly uniform attention (fully global), 0.0 for a one-hot row.
    fn calculate_global_pattern_score(&self, attention_weights: &[Vec<f32>]) -> f32 {
        let seq_len = attention_weights.len();
        if seq_len < 2 {
            return 0.0;
        }
        let max_entropy = (seq_len as f32).ln();
        let mut entropy_sum = 0.0;

        for row in attention_weights {
            let mass: f32 = row.iter().sum();
            if mass <= f32::EPSILON {
                continue;
            }
            let mut entropy = 0.0;
            for &weight in row {
                let p = weight / mass;
                if p > 1e-8 {
                    entropy -= p * p.ln();
                }
            }
            entropy_sum += entropy;
        }

        (entropy_sum / seq_len as f32 / max_entropy).clamp(0.0, 1.0)
    }

    /// Attention concentration of one head: `1 - H(row) / ln(seq_len)`.
    ///
    /// 0.0 when the head attends uniformly (carries no positional preference),
    /// 1.0 when it is one-hot.
    fn attention_concentration(&self, pattern: &AttentionPattern) -> f32 {
        1.0 - self.calculate_global_pattern_score(&pattern.attention_weights)
    }

    /// Per-layer / per-head attention concentration derived from the attention
    /// patterns recorded so far.
    ///
    /// Returns `(layer_scores, head_scores)`. Both are empty when
    /// [`Self::analyze_attention_patterns`] has not been called: layer-level
    /// ablation is not reachable through the single `model_fn` closure this
    /// analyzer is given, so no value is invented for it.
    fn attention_derived_importance(&self) -> (Vec<f32>, Vec<Vec<f32>>) {
        if self.attention_patterns.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let num_layers = self.attention_patterns.iter().map(|p| p.layer_idx).max().unwrap_or(0) + 1;
        let mut head_scores: Vec<Vec<f32>> = vec![Vec::new(); num_layers];

        for pattern in &self.attention_patterns {
            let score = self.attention_concentration(pattern);
            let heads = &mut head_scores[pattern.layer_idx];
            if heads.len() <= pattern.head_idx {
                heads.resize(pattern.head_idx + 1, 0.0);
            }
            heads[pattern.head_idx] = score;
        }

        let layer_scores = head_scores
            .iter()
            .map(|heads| {
                if heads.is_empty() {
                    0.0
                } else {
                    heads.iter().sum::<f32>() / heads.len() as f32
                }
            })
            .collect();

        (layer_scores, head_scores)
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
        let (layer_importance, head_importance) = self.attention_derived_importance();

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
                    let importance =
                        self.calculate_output_difference(original_output, &occluded_output)?;
                    importance_scores.push(importance);
                }

                Ok(importance_scores)
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor type for token importance",
                "analyze_token_importance",
            )),
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

                    let importance =
                        self.calculate_output_difference(original_output, &masked_output)?;
                    importance_scores.push(importance);
                }

                Ok(importance_scores)
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor type for position importance",
                "analyze_position_importance",
            )),
        }
    }

    /// Calculate difference between two outputs
    fn calculate_output_difference(&self, original: &Tensor, modified: &Tensor) -> Result<f32> {
        match (original, modified) {
            (Tensor::F32(orig), Tensor::F32(modif)) => {
                let diff_sum: f32 = orig.iter().zip(modif.iter()).map(|(a, b)| (a - b).abs()).sum();
                Ok(diff_sum / orig.len() as f32)
            },
            _ => Err(TrustformersError::invalid_operation(
                "Tensor type mismatch in output difference".into(),
            )),
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

        if method == AttributionMethod::IntegratedGradients {
            return Err(TrustformersError::invalid_input(
                "Integrated gradients cannot be computed from a single gradient tensor; \
                 use `analyze_integrated_gradients(inputs, baseline, grad_fn, steps)` \
                 which evaluates the gradient along the baseline->input path"
                    .to_string(),
            ));
        }

        let input_gradients = self.extract_input_gradients(gradients)?;
        let gradient_x_input = self.calculate_gradient_x_input(inputs, gradients)?;
        let saliency_scores = self.calculate_saliency_scores(&input_gradients);

        self.gradient_attribution = Some(GradientAttribution {
            input_gradients,
            gradient_x_input,
            integrated_gradients: None,
            saliency_scores,
            attribution_method: method,
        });

        Ok(())
    }

    /// Extract input gradients from gradient tensor
    fn extract_input_gradients(&self, gradients: &Tensor) -> Result<Vec<f32>> {
        match gradients {
            Tensor::F32(arr) => Ok(arr.iter().cloned().collect()),
            _ => Err(TrustformersError::invalid_operation(
                "Unsupported tensor type for gradient extraction".into(),
            )),
        }
    }

    /// Element-wise `input * gradient` attribution (a.k.a. "gradient x input").
    ///
    /// This is a single-point attribution, not a path integral. See
    /// [`Self::compute_integrated_gradients`] for the real Riemann-sum variant.
    fn calculate_gradient_x_input(&self, inputs: &Tensor, gradients: &Tensor) -> Result<Vec<f32>> {
        match (inputs, gradients) {
            (Tensor::F32(inp), Tensor::F32(grad)) => {
                if inp.len() != grad.len() {
                    return Err(TrustformersError::dimension_mismatch(
                        format!("{} input elements", inp.len()),
                        format!("{} gradient elements", grad.len()),
                    ));
                }
                let attributed: Vec<f32> =
                    inp.iter().zip(grad.iter()).map(|(input, gradient)| input * gradient).collect();
                Ok(attributed)
            },
            _ => Err(TrustformersError::invalid_operation(
                "Tensor type mismatch in gradient x input attribution".into(),
            )),
        }
    }

    /// Compute integrated gradients along the straight-line path from
    /// `baseline` to `inputs` (Sundararajan et al., 2017).
    ///
    /// `IG_i = (x_i - x'_i) * (1/m) * sum_{k=1..m} d f / d x_i evaluated at
    /// `x' + (k/m) * (x - x')`. `grad_fn` must return the gradient of the model
    /// output with respect to the tensor it is given; it is called `steps`
    /// times, so this is genuinely a multi-evaluation path integral.
    pub fn compute_integrated_gradients(
        &self,
        inputs: &Tensor,
        baseline: &Tensor,
        grad_fn: &dyn Fn(&Tensor) -> Result<Tensor>,
        steps: usize,
    ) -> Result<Vec<f32>> {
        if steps == 0 {
            return Err(TrustformersError::invalid_input(
                "integrated gradients requires at least one Riemann step".to_string(),
            ));
        }

        let (input_arr, baseline_arr) = match (inputs, baseline) {
            (Tensor::F32(a), Tensor::F32(b)) => (a, b),
            _ => {
                return Err(TrustformersError::invalid_operation(
                    "integrated gradients requires F32 tensors".into(),
                ))
            },
        };

        if input_arr.shape() != baseline_arr.shape() {
            return Err(TrustformersError::dimension_mismatch(
                format!("{:?}", baseline_arr.shape()),
                format!("{:?}", input_arr.shape()),
            ));
        }

        let mut accumulated = vec![0.0f32; input_arr.len()];

        for step in 1..=steps {
            let alpha = step as f32 / steps as f32;
            let mut interpolated = baseline_arr.clone();
            for (dst, (base, inp)) in
                interpolated.iter_mut().zip(baseline_arr.iter().zip(input_arr.iter()))
            {
                *dst = base + alpha * (inp - base);
            }

            let gradient = grad_fn(&Tensor::F32(interpolated))?;
            let gradient_values = self.extract_input_gradients(&gradient)?;
            if gradient_values.len() != accumulated.len() {
                return Err(TrustformersError::dimension_mismatch(
                    format!("{} input elements", accumulated.len()),
                    format!("{} gradient elements", gradient_values.len()),
                ));
            }
            for (acc, g) in accumulated.iter_mut().zip(gradient_values.iter()) {
                *acc += g;
            }
        }

        let scale = 1.0 / steps as f32;
        let integrated: Vec<f32> = accumulated
            .iter()
            .zip(baseline_arr.iter().zip(input_arr.iter()))
            .map(|(acc, (base, inp))| (inp - base) * acc * scale)
            .collect();

        Ok(integrated)
    }

    /// Run integrated-gradients attribution and store the result.
    pub fn analyze_integrated_gradients(
        &mut self,
        inputs: &Tensor,
        baseline: &Tensor,
        grad_fn: &dyn Fn(&Tensor) -> Result<Tensor>,
        steps: usize,
    ) -> Result<()> {
        if !self.config.enable_gradient_analysis {
            return Ok(());
        }

        let integrated = self.compute_integrated_gradients(inputs, baseline, grad_fn, steps)?;
        let end_point_gradients = self.extract_input_gradients(&grad_fn(inputs)?)?;
        let gradient_x_input =
            self.calculate_gradient_x_input(inputs, &grad_fn(inputs)?).unwrap_or_default();
        let saliency_scores = self.calculate_saliency_scores(&end_point_gradients);

        self.gradient_attribution = Some(GradientAttribution {
            input_gradients: end_point_gradients,
            gradient_x_input,
            integrated_gradients: Some(integrated),
            saliency_scores,
            attribution_method: AttributionMethod::IntegratedGradients,
        });

        Ok(())
    }

    /// Calculate saliency scores from gradients
    fn calculate_saliency_scores(&self, gradients: &[f32]) -> Vec<f32> {
        gradients.iter().map(|&grad| grad.abs()).collect()
    }

    /// Analyze activation patterns
    pub fn analyze_activations(&mut self, activations: &HashMap<String, Tensor>) -> Result<()> {
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

    /// Per-neuron activation columns.
    ///
    /// The last tensor axis is the neuron axis; every preceding axis (batch,
    /// sequence, ...) is flattened into the sample axis. Returns
    /// `columns[neuron][sample]`.
    fn neuron_columns(arr: &scirs2_core::ndarray::ArrayD<f32>) -> Vec<Vec<f32>> {
        let shape = arr.shape();
        if shape.is_empty() {
            return Vec::new();
        }
        let num_neurons = shape[shape.len() - 1];
        if num_neurons == 0 {
            return Vec::new();
        }
        let num_samples = arr.len() / num_neurons;

        let mut columns = vec![Vec::with_capacity(num_samples); num_neurons];
        // Standard (row-major) iteration order visits the neuron axis fastest.
        for (flat_index, value) in arr.iter().enumerate() {
            columns[flat_index % num_neurons].push(*value);
        }
        columns
    }

    /// Extract activation patterns from tensor
    ///
    /// Returns the mean activation of each neuron over all samples.
    fn extract_activation_patterns(&self, tensor: &Tensor) -> Result<Vec<f32>> {
        match tensor {
            Tensor::F32(arr) => {
                let shape = arr.shape();
                if shape.len() < 2 {
                    return Ok(vec![0.0]);
                }

                let patterns = Self::neuron_columns(arr)
                    .into_iter()
                    .map(|column| {
                        if column.is_empty() {
                            0.0
                        } else {
                            column.iter().sum::<f32>() / column.len() as f32
                        }
                    })
                    .collect();

                Ok(patterns)
            },
            _ => Err(TrustformersError::invalid_operation(
                "Unsupported tensor type for activation analysis".into(),
            )),
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
                let variance =
                    data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
                let std = variance.sqrt();

                let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                let zero_count = data.iter().filter(|&&x| x.abs() < 1e-6).count();
                let sparsity = zero_count as f32 / data.len() as f32;

                // Calculate skewness and kurtosis
                let skewness = if std > 0.0 {
                    data.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f32>() / data.len() as f32
                } else {
                    0.0
                };

                let kurtosis = if std > 0.0 {
                    data.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f32>() / data.len() as f32
                        - 3.0
                } else {
                    0.0
                };

                Ok(ActivationStats {
                    mean,
                    std,
                    min,
                    max,
                    sparsity,
                    skewness,
                    kurtosis,
                })
            },
            _ => Err(TrustformersError::invalid_operation(
                "Unsupported tensor type for activation statistics".into(),
            )),
        }
    }

    /// Count dead neurons (never activate across any sample).
    ///
    /// A neuron is the last tensor axis; a neuron is dead when *every* value in
    /// its column is within 1e-6 of zero.
    fn count_dead_neurons(&self, tensor: &Tensor) -> Result<usize> {
        match tensor {
            Tensor::F32(arr) => {
                if arr.shape().is_empty() {
                    return Ok(0);
                }

                let dead_count = Self::neuron_columns(arr)
                    .into_iter()
                    .filter(|column| column.iter().all(|value| value.abs() <= 1e-6))
                    .count();

                Ok(dead_count)
            },
            _ => Err(TrustformersError::invalid_operation(
                "Unsupported tensor type for dead neuron counting".into(),
            )),
        }
    }

    /// Cluster activations to find similar patterns
    fn cluster_activations(
        &self,
        layer_name: &str,
        tensor: &Tensor,
    ) -> Result<Vec<ActivationCluster>> {
        // Lloyd's k-means over the per-neuron activation profiles.
        const K: usize = 3;
        const MAX_ITERATIONS: usize = 50;

        match tensor {
            Tensor::F32(arr) => {
                if arr.shape().is_empty() {
                    return Ok(Vec::new());
                }

                let columns = Self::neuron_columns(arr);
                if columns.is_empty() || columns[0].is_empty() {
                    return Ok(Vec::new());
                }
                let dimension = columns[0].len();
                let k = K.min(columns.len());

                // Deterministic k-means++ style seeding: first centroid is
                // neuron 0, each subsequent centroid is the neuron farthest
                // from every centroid chosen so far.
                let mut centroids: Vec<Vec<f32>> = vec![columns[0].clone()];
                while centroids.len() < k {
                    let mut best_index = 0usize;
                    let mut best_distance = -1.0f32;
                    for (index, column) in columns.iter().enumerate() {
                        let distance = centroids
                            .iter()
                            .map(|centroid| squared_distance(column, centroid))
                            .fold(f32::INFINITY, f32::min);
                        if distance > best_distance {
                            best_distance = distance;
                            best_index = index;
                        }
                    }
                    centroids.push(columns[best_index].clone());
                }

                let mut assignments = vec![0usize; columns.len()];
                for _ in 0..MAX_ITERATIONS {
                    let mut changed = false;
                    for (index, column) in columns.iter().enumerate() {
                        let mut best_cluster = 0usize;
                        let mut best_distance = f32::INFINITY;
                        for (cluster_id, centroid) in centroids.iter().enumerate() {
                            let distance = squared_distance(column, centroid);
                            if distance < best_distance {
                                best_distance = distance;
                                best_cluster = cluster_id;
                            }
                        }
                        if assignments[index] != best_cluster {
                            assignments[index] = best_cluster;
                            changed = true;
                        }
                    }

                    // Recompute centroids from the current assignment.
                    let mut sums = vec![vec![0.0f32; dimension]; k];
                    let mut counts = vec![0usize; k];
                    for (index, column) in columns.iter().enumerate() {
                        let cluster_id = assignments[index];
                        counts[cluster_id] += 1;
                        for (accumulator, value) in sums[cluster_id].iter_mut().zip(column.iter()) {
                            *accumulator += value;
                        }
                    }
                    for (cluster_id, centroid) in centroids.iter_mut().enumerate() {
                        if counts[cluster_id] > 0 {
                            let inverse = 1.0 / counts[cluster_id] as f32;
                            for (target, sum) in centroid.iter_mut().zip(sums[cluster_id].iter()) {
                                *target = sum * inverse;
                            }
                        }
                    }

                    if !changed {
                        break;
                    }
                }

                let mut clusters = Vec::with_capacity(k);
                for (cluster_id, centroid) in centroids.into_iter().enumerate() {
                    let neuron_indices: Vec<usize> = assignments
                        .iter()
                        .enumerate()
                        .filter(|(_, assigned)| **assigned == cluster_id)
                        .map(|(index, _)| index)
                        .collect();

                    if neuron_indices.is_empty() {
                        continue;
                    }

                    // Mean squared distance of the members to their centroid.
                    let variance = neuron_indices
                        .iter()
                        .map(|index| squared_distance(&columns[*index], &centroid))
                        .sum::<f32>()
                        / neuron_indices.len() as f32;

                    clusters.push(ActivationCluster {
                        layer_name: layer_name.to_string(),
                        cluster_id,
                        neuron_indices,
                        centroid,
                        variance,
                    });
                }

                Ok(clusters)
            },
            _ => Err(TrustformersError::invalid_operation(
                "Unsupported tensor type for activation clustering".into(),
            )),
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
            self.attention_patterns.iter().map(|p| p.entropy).sum::<f32>()
                / self.attention_patterns.len() as f32
        } else {
            0.0
        };

        let most_important_tokens = self
            .feature_importance
            .as_ref()
            .map(|fi| {
                fi.token_importance
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(::std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .unwrap_or(0);

        let total_dead_neurons = self
            .activation_analysis
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

/// Squared Euclidean distance between two equal-length activation profiles.
fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
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
        let attention_tensor =
            Tensor::from_vec(attention_data, &[2, 2]).expect("Tensor from_vec failed");
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
        let activation_tensor =
            Tensor::from_vec(activation_data, &[2, 4]).expect("Tensor from_vec failed");

        let stats = analyzer
            .calculate_activation_statistics(&activation_tensor)
            .expect("tensor operation failed");
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
        let activation_tensor =
            Tensor::from_vec(activation_data, &[2, 4]).expect("Tensor from_vec failed");

        let dead_count = analyzer
            .count_dead_neurons(&activation_tensor)
            .expect("tensor operation failed");
        assert!(dead_count > 0);
    }

    /// Regression test: `count_dead_neurons` used to scan the *whole* tensor for
    /// every neuron index, so it could only ever return 0 or `num_neurons`.
    /// Neurons 1 and 2 below are dead, neurons 0 and 3 are alive.
    #[test]
    fn test_dead_neuron_count_is_per_neuron_not_all_or_nothing() {
        let analyzer = InterpretabilityAnalyzer::new(InterpretabilityConfig::default());

        // shape [3, 4]; last axis is the neuron axis.
        //   neuron 0 -> [1.0, 5.0, 9.0]  (alive)
        //   neuron 1 -> [0.0, 0.0, 0.0]  (dead)
        //   neuron 2 -> [0.0, 0.0, 0.0]  (dead)
        //   neuron 3 -> [4.0, 8.0, 12.0] (alive)
        #[rustfmt::skip]
        let data = vec![
            1.0, 0.0, 0.0, 4.0,
            5.0, 0.0, 0.0, 8.0,
            9.0, 0.0, 0.0, 12.0,
        ];
        let tensor = Tensor::from_vec(data, &[3, 4]).expect("Tensor from_vec failed");

        let dead_count = analyzer.count_dead_neurons(&tensor).expect("dead neuron count failed");
        assert_eq!(
            dead_count, 2,
            "expected exactly the two all-zero neuron columns to be dead"
        );

        // The old implementation also gave every neuron the same activation
        // pattern; the per-neuron means must now differ.
        let patterns = analyzer
            .extract_activation_patterns(&tensor)
            .expect("activation patterns failed");
        assert_eq!(patterns.len(), 4);
        assert!((patterns[0] - 5.0).abs() < 1e-6, "got {:?}", patterns);
        assert!((patterns[1] - 0.0).abs() < 1e-6, "got {:?}", patterns);
        assert!((patterns[3] - 8.0).abs() < 1e-6, "got {:?}", patterns);
    }

    /// Regression test: `cluster_activations` used to give every neuron the same
    /// score (`arr.iter().sum()`) and a hardcoded `centroid: vec![0.5; 10]`.
    #[test]
    fn test_activation_clustering_uses_real_centroids() {
        let analyzer = InterpretabilityAnalyzer::new(InterpretabilityConfig::default());

        // Two clearly separated neuron groups: low (~0.1) and high (~10.0).
        #[rustfmt::skip]
        let data = vec![
            0.1, 0.1, 10.0, 10.0,
            0.1, 0.1, 10.0, 10.0,
        ];
        let tensor = Tensor::from_vec(data, &[2, 4]).expect("Tensor from_vec failed");

        let clusters = analyzer.cluster_activations("layer0", &tensor).expect("clustering failed");
        assert!(!clusters.is_empty());

        for cluster in &clusters {
            // Centroid must have the sample dimension (2), not the old fixed 10.
            assert_eq!(
                cluster.centroid.len(),
                2,
                "centroid must be a real mean vector"
            );
            // Every centroid must sit on one of the two real groups.
            let value = cluster.centroid[0];
            assert!(
                (value - 0.1).abs() < 1e-4 || (value - 10.0).abs() < 1e-4,
                "centroid {} is neither of the two real activation groups",
                value
            );
            assert!(cluster.variance >= 0.0);
            assert!(
                (cluster.variance - 0.1).abs() > 1e-9 || cluster.neuron_indices.len() > 1,
                "variance must be measured, not the old hardcoded 0.1"
            );
        }

        // The two groups must not be collapsed into one cluster.
        let populated = clusters.iter().filter(|c| !c.neuron_indices.is_empty()).count();
        assert!(
            populated >= 2,
            "expected the low and high groups to separate"
        );
    }

    /// Regression test: `analyze_gradient_attribution` used to label
    /// `input * gradient` as integrated gradients. Real integrated gradients
    /// need multiple evaluations along a baseline->input path.
    #[test]
    fn test_integrated_gradients_are_a_real_path_integral() {
        let analyzer = InterpretabilityAnalyzer::new(InterpretabilityConfig::default());

        // f(x) = sum(x^2) => df/dx_i = 2 x_i.
        let grad_fn = |t: &Tensor| -> Result<Tensor> {
            match t {
                Tensor::F32(arr) => {
                    let doubled = arr.mapv(|v| 2.0 * v);
                    Ok(Tensor::F32(doubled))
                },
                _ => Err(TrustformersError::invalid_operation("expected F32".into())),
            }
        };

        let inputs = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("from_vec failed");
        let baseline = Tensor::zeros(&[3]).expect("zeros failed");

        let integrated = analyzer
            .compute_integrated_gradients(&inputs, &baseline, &grad_fn, 512)
            .expect("integrated gradients failed");

        // Completeness axiom: sum(IG) == f(x) - f(baseline) == sum(x^2).
        // For f(x)=x^2 the exact per-feature attribution is x_i^2.
        for (value, expected) in integrated.iter().zip([1.0f32, 4.0, 9.0]) {
            assert!(
                (value - expected).abs() < 0.05,
                "integrated gradients {:?} do not satisfy the completeness axiom",
                integrated
            );
        }

        // gradient x input for the same function would be 2*x*x = 2 x^2, i.e.
        // twice the correct attribution: the two must not be interchangeable.
        let gradients = grad_fn(&inputs).expect("grad_fn failed");
        let gxi = analyzer
            .calculate_gradient_x_input(&inputs, &gradients)
            .expect("gradient x input failed");
        assert!((gxi[2] - 18.0).abs() < 1e-4, "got {:?}", gxi);
    }

    /// Regression test: requesting `IntegratedGradients` from the single-tensor
    /// entry point must fail loudly instead of silently returning gradient x input.
    #[test]
    fn test_single_tensor_attribution_rejects_integrated_gradients() {
        let mut analyzer = InterpretabilityAnalyzer::new(InterpretabilityConfig::default());
        let inputs = Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("from_vec failed");
        let gradients = Tensor::from_vec(vec![0.5, 0.5], &[2]).expect("from_vec failed");

        let result = analyzer.analyze_gradient_attribution(
            &inputs,
            &gradients,
            AttributionMethod::IntegratedGradients,
        );
        assert!(result.is_err(), "must not fabricate integrated gradients");
    }

    /// Regression test: `layer_importance` / `head_importance` used to be
    /// `vec![1.0; 12]` / `vec![vec![1.0; 8]; 12]` regardless of the model.
    #[test]
    fn test_feature_importance_reports_no_layer_scores_without_attention_data() {
        let mut analyzer = InterpretabilityAnalyzer::new(InterpretabilityConfig::default());

        let inputs = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).expect("from_vec failed");
        let model_fn = |t: &Tensor| -> Result<Tensor> { Ok(t.clone()) };
        let outputs = model_fn(&inputs).expect("model failed");

        analyzer
            .analyze_feature_importance(&inputs, &outputs, &model_fn)
            .expect("feature importance failed");

        let importance = analyzer.feature_importance.as_ref().expect("importance recorded");
        assert!(
            importance.layer_importance.is_empty(),
            "no attention patterns were recorded, so no layer scores may be invented"
        );
        assert!(importance.head_importance.is_empty());
        assert_eq!(importance.token_importance.len(), 4);
    }

    /// Regression test: the local-window score used to saturate at 1.0 for any
    /// short sequence, so a clearly diagonal matrix was classified `Local`.
    #[test]
    fn test_pattern_scores_are_normalised_against_uniform_attention() {
        let analyzer = InterpretabilityAnalyzer::new(InterpretabilityConfig::default());

        // A uniform matrix has no pattern at all: every score must be ~0.
        let uniform = vec![vec![0.25f32; 4]; 4];
        assert!(analyzer.calculate_local_pattern_score(&uniform) < 1e-5);
        assert!(analyzer.calculate_diagonal_pattern_score(&uniform) < 1e-5);
        assert!(analyzer.calculate_vertical_pattern_score(&uniform) < 1e-5);
        assert_eq!(
            analyzer.classify_attention_pattern(&uniform),
            AttentionPatternType::Global
        );

        // A vertical (single-column) pattern must beat the local window score.
        let vertical: Vec<Vec<f32>> = (0..4).map(|_| vec![1.0, 0.0, 0.0, 0.0]).collect();
        assert_eq!(
            analyzer.classify_attention_pattern(&vertical),
            AttentionPatternType::Vertical
        );
    }

    #[test]
    fn test_gradient_attribution_analysis() {
        let config = InterpretabilityConfig::default();
        let mut analyzer = InterpretabilityAnalyzer::new(config);

        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let gradient_data = vec![0.1, 0.2, 0.3, 0.4];

        let inputs = Tensor::from_vec(input_data, &[2, 2]).expect("Tensor from_vec failed");
        let gradients = Tensor::from_vec(gradient_data, &[2, 2]).expect("Tensor from_vec failed");

        let result = analyzer.analyze_gradient_attribution(
            &inputs,
            &gradients,
            AttributionMethod::Gradients,
        );
        assert!(result.is_ok());
        assert!(analyzer.gradient_attribution.is_some());
    }

    #[test]
    fn test_comprehensive_activation_analysis() {
        let config = InterpretabilityConfig::default();
        let mut analyzer = InterpretabilityAnalyzer::new(config);

        let mut activations = HashMap::new();
        let activation_data = vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0];
        let activation_tensor =
            Tensor::from_vec(activation_data, &[2, 4]).expect("Tensor from_vec failed");
        activations.insert("layer1".to_string(), activation_tensor);

        let result = analyzer.analyze_activations(&activations);
        assert!(result.is_ok());

        let analysis = analyzer.get_activation_analysis().expect("operation failed in test");
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
            output_dir: Some(
                std::env::temp_dir().join("interpretability").to_string_lossy().to_string(),
            ),
        };

        let serialized = serde_json::to_string(&config).expect("JSON serialization failed");
        let deserialized: InterpretabilityConfig =
            serde_json::from_str(&serialized).expect("JSON deserialization failed");

        assert_eq!(
            config.enable_attention_analysis,
            deserialized.enable_attention_analysis
        );
        assert_eq!(
            config.enable_gradient_analysis,
            deserialized.enable_gradient_analysis
        );
        assert_eq!(config.save_visualizations, deserialized.save_visualizations);
        assert_eq!(config.output_dir, deserialized.output_dir);
    }
}
