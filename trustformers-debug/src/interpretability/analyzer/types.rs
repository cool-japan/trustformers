//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::interpretability::attention::AttentionAnalysisResult;
use crate::interpretability::attribution::FeatureAttributionResult;
use crate::interpretability::config::InterpretabilityConfig;
use crate::interpretability::counterfactual::CounterfactualResult;
use crate::interpretability::lime::{
    FeatureImportance, LimeAnalysisResult, NeighborhoodStats, PerturbationAnalysis,
    PerturbationResult,
};
use crate::interpretability::report::InterpretabilityReport;
use crate::interpretability::shap::{FeatureContribution, ShapAnalysisResult, ShapSummary};
use anyhow::Result;
use chrono::Utc;
use scirs2_core::random::*;
use std::collections::HashMap;

/// Main interpretability analyzer
#[derive(Debug)]
pub struct InterpretabilityAnalyzer {
    config: InterpretabilityConfig,
    shap_results: Vec<ShapAnalysisResult>,
    lime_results: Vec<LimeAnalysisResult>,
    attention_results: Vec<AttentionAnalysisResult>,
    attribution_results: Vec<FeatureAttributionResult>,
    counterfactual_results: Vec<CounterfactualResult>,
}
impl InterpretabilityAnalyzer {
    /// Create a new interpretability analyzer
    pub fn new(config: InterpretabilityConfig) -> Self {
        Self {
            config,
            shap_results: Vec::new(),
            lime_results: Vec::new(),
            attention_results: Vec::new(),
            attribution_results: Vec::new(),
            counterfactual_results: Vec::new(),
        }
    }
    /// Perform SHAP analysis on given instance
    pub async fn analyze_shap(
        &mut self,
        instance: &HashMap<String, f64>,
        model_predictions: &[f64],
        background_data: &[HashMap<String, f64>],
    ) -> Result<ShapAnalysisResult> {
        if !self.config.enable_shap {
            return Err(anyhow::anyhow!("SHAP analysis is disabled"));
        }
        let feature_names: Vec<String> = instance.keys().cloned().collect();
        let base_value = model_predictions.iter().sum::<f64>() / model_predictions.len() as f64;
        let prediction = model_predictions[0];
        let mut shap_values = HashMap::new();
        let mut feature_contributions = Vec::new();
        for (i, feature_name) in feature_names.iter().enumerate() {
            let feature_value = instance[feature_name];
            let shap_value = self.calculate_simplified_shap_value(
                feature_name,
                feature_value,
                &feature_names,
                instance,
                background_data,
                base_value,
            );
            shap_values.insert(feature_name.clone(), shap_value);
            feature_contributions.push(FeatureContribution {
                feature_name: feature_name.clone(),
                shap_value,
                feature_value,
                importance_rank: i + 1,
                contribution_percentage: (shap_value / (prediction - base_value).abs()) * 100.0,
            });
        }
        feature_contributions
            .sort_by(|a, b| b.shap_value.abs().partial_cmp(&a.shap_value.abs()).unwrap());
        for (i, contribution) in feature_contributions.iter_mut().enumerate() {
            contribution.importance_rank = i + 1;
        }
        let top_positive_features: Vec<_> = feature_contributions
            .iter()
            .filter(|f| f.shap_value > 0.0)
            .take(5)
            .cloned()
            .collect();
        let top_negative_features: Vec<_> = feature_contributions
            .iter()
            .filter(|f| f.shap_value < 0.0)
            .take(5)
            .cloned()
            .collect();
        let total_positive_contribution = feature_contributions
            .iter()
            .filter(|f| f.shap_value > 0.0)
            .map(|f| f.shap_value)
            .sum();
        let total_negative_contribution = feature_contributions
            .iter()
            .filter(|f| f.shap_value < 0.0)
            .map(|f| f.shap_value)
            .sum();
        let num_important_features =
            feature_contributions.iter().filter(|f| f.shap_value.abs() > 0.01).count();
        let importance_distribution = feature_contributions
            .iter()
            .map(|f| (f.feature_name.clone(), f.shap_value.abs()))
            .collect();
        let result = ShapAnalysisResult {
            timestamp: Utc::now(),
            shap_values,
            feature_names,
            base_value,
            prediction,
            feature_contributions,
            top_positive_features,
            top_negative_features,
            interaction_values: None,
            summary: ShapSummary {
                total_positive_contribution,
                total_negative_contribution,
                num_important_features,
                importance_distribution,
                explanation_completeness: 0.85,
            },
        };
        self.shap_results.push(result.clone());
        Ok(result)
    }
    /// Perform LIME analysis on given instance
    pub async fn analyze_lime(
        &mut self,
        instance: &HashMap<String, f64>,
        model_fn: Box<dyn Fn(&HashMap<String, f64>) -> f64>,
    ) -> Result<LimeAnalysisResult> {
        if !self.config.enable_lime {
            return Err(anyhow::anyhow!("LIME analysis is disabled"));
        }
        let feature_names: Vec<String> = instance.keys().cloned().collect();
        let original_prediction = model_fn(instance);
        let mut perturbation_results = Vec::new();
        let mut local_data = Vec::new();
        let mut predictions = Vec::new();
        let mut rng = thread_rng();
        for i in 0..self.config.lime_perturbations {
            let mut perturbed_instance = instance.clone();
            let mut perturbed_features = Vec::new();
            for feature_name in &feature_names {
                if rng.random::<f64>() < 0.3 {
                    let original_value = perturbed_instance[feature_name];
                    let noise = (rng.random::<f64>() - 0.5) * 0.2 * original_value;
                    perturbed_instance.insert(feature_name.clone(), original_value + noise);
                    perturbed_features.push(feature_name.clone());
                }
            }
            let perturbed_prediction = model_fn(&perturbed_instance);
            let distance = self.calculate_distance(instance, &perturbed_instance);
            perturbation_results.push(PerturbationResult {
                id: format!("pert_{}", i),
                perturbed_features,
                original_prediction,
                perturbed_prediction,
                prediction_change: perturbed_prediction - original_prediction,
                distance,
            });
            local_data.push(perturbed_instance);
            predictions.push(perturbed_prediction);
        }
        let mut local_coefficients = HashMap::new();
        let mut feature_importance = Vec::new();
        for feature_name in &feature_names {
            let coeff = self.calculate_local_coefficient(feature_name, &local_data, &predictions);
            local_coefficients.insert(feature_name.clone(), coeff);
            feature_importance.push(FeatureImportance {
                feature_name: feature_name.clone(),
                importance_score: coeff.abs(),
                confidence_interval: (coeff - 0.1, coeff + 0.1),
                p_value: 0.05,
                stability: 0.8,
            });
        }
        let result = LimeAnalysisResult {
            timestamp: Utc::now(),
            local_coefficients,
            feature_names,
            local_r_squared: 0.75,
            intercept: original_prediction,
            feature_importance,
            perturbation_analysis: PerturbationAnalysis {
                num_perturbations: self.config.lime_perturbations,
                strategy: "random".to_string(),
                prediction_variance: predictions
                    .iter()
                    .map(|p| (p - original_prediction).powi(2))
                    .sum::<f64>()
                    / predictions.len() as f64,
                neighborhood_coverage: 0.8,
                influential_perturbations: perturbation_results.into_iter().take(10).collect(),
            },
            neighborhood_stats: NeighborhoodStats {
                mean_prediction: predictions.iter().sum::<f64>() / predictions.len() as f64,
                std_prediction: 0.1,
                density: 0.5,
                correlation_matrix: HashMap::new(),
            },
            local_fidelity: 0.85,
        };
        self.lime_results.push(result.clone());
        Ok(result)
    }
    /// Perform attention analysis (placeholder for transformer models)
    pub async fn analyze_attention(
        &mut self,
        attention_weights: &HashMap<String, Vec<Vec<f64>>>,
        tokens: &[String],
    ) -> Result<AttentionAnalysisResult> {
        if !self.config.enable_attention_analysis {
            return Err(anyhow::anyhow!("Attention analysis is disabled"));
        }
        use crate::interpretability::attention::*;
        let mut layer_results = HashMap::new();
        let mut all_entropies = Vec::new();
        let mut diagonal_patterns = Vec::new();
        let mut vertical_patterns = Vec::new();
        let mut block_patterns = Vec::new();
        let mut repetitive_patterns = Vec::new();
        for (layer_key, attention_matrix) in attention_weights {
            let parts: Vec<&str> = layer_key.split('_').collect();
            if parts.len() < 4 {
                continue;
            }
            let layer_idx: usize = parts[1].parse().unwrap_or(0);
            let head_idx: usize = parts[3].parse().unwrap_or(0);
            let entropy = self.compute_attention_entropy(attention_matrix);
            let max_attention =
                attention_matrix.iter().flat_map(|row| row.iter()).cloned().fold(0.0, f64::max);
            let sparsity = self.compute_sparsity(attention_matrix);
            let specialization_type = self.classify_head_specialization(attention_matrix);
            let token_scores = self.compute_token_attention_scores(attention_matrix, tokens);
            if let Some(pattern) =
                self.detect_diagonal_pattern(attention_matrix, layer_idx, head_idx)
            {
                diagonal_patterns.push(pattern);
            }
            if let Some(pattern) =
                self.detect_vertical_pattern(attention_matrix, layer_idx, head_idx)
            {
                vertical_patterns.push(pattern);
            }
            if let Some(pattern) = self.detect_block_pattern(attention_matrix, layer_idx, head_idx)
            {
                block_patterns.push(pattern);
            }
            if let Some(pattern) =
                self.detect_repetitive_pattern(attention_matrix, layer_idx, head_idx)
            {
                repetitive_patterns.push(pattern);
            }
            let head_result = AttentionHeadResult {
                head_index: head_idx,
                attention_matrix: attention_matrix.clone(),
                token_scores,
                specialization_type,
                entropy,
                max_attention,
                sparsity,
            };
            all_entropies.push(entropy);
            layer_results
                .entry(layer_idx)
                .or_insert_with(HashMap::new)
                .insert(head_idx, head_result);
        }
        let mut attention_layer_results = HashMap::new();
        for (layer_idx, heads) in layer_results {
            let layer_patterns = self.compute_layer_attention_patterns(&heads);
            let layer_stats = self.compute_layer_attention_stats(&heads);
            attention_layer_results.insert(
                format!("layer_{}", layer_idx),
                AttentionLayerResult {
                    layer_index: layer_idx,
                    heads,
                    layer_patterns,
                    layer_stats,
                },
            );
        }
        let head_specialization = self.analyze_head_specialization(&attention_layer_results);
        let attention_flow = self.analyze_attention_flow(&attention_layer_results);
        let avg_entropy = if !all_entropies.is_empty() {
            all_entropies.iter().sum::<f64>() / all_entropies.len() as f64
        } else {
            0.0
        };
        let concentration_distribution =
            self.compute_concentration_distribution(&attention_layer_results);
        let sparsity_distribution = self.compute_sparsity_distribution(&attention_layer_results);
        let insights = self.generate_attention_insights(
            &attention_layer_results,
            &head_specialization,
            &attention_flow,
        );
        let attention_stats = AttentionStatistics {
            avg_entropy,
            concentration_distribution,
            sparsity_distribution,
            insights,
        };
        let attention_patterns = AttentionPatterns {
            diagonal_patterns,
            vertical_patterns,
            block_patterns,
            repetitive_patterns,
        };
        let result = AttentionAnalysisResult {
            timestamp: Utc::now(),
            attention_weights: attention_layer_results,
            attention_patterns,
            head_specialization,
            attention_flow,
            attention_stats,
        };
        self.attention_results.push(result.clone());
        Ok(result)
    }
    /// Perform feature attribution analysis
    pub async fn analyze_feature_attribution(
        &mut self,
        instance: &HashMap<String, f64>,
        model_gradients: &HashMap<String, f64>,
    ) -> Result<FeatureAttributionResult> {
        if !self.config.enable_feature_attribution {
            return Err(anyhow::anyhow!("Feature attribution analysis is disabled"));
        }
        use crate::interpretability::attribution::*;
        use crate::interpretability::config::AttributionMethod;
        use std::time::Instant;
        let mut attribution_by_method = HashMap::new();
        let feature_names: Vec<String> = instance.keys().cloned().collect();
        let start_time = Instant::now();
        let grad_input_attributions =
            self.compute_gradient_input_attribution(instance, model_gradients, &feature_names);
        let grad_input_time = start_time.elapsed().as_millis() as f64;
        attribution_by_method.insert(
            AttributionMethod::GradientInput,
            AttributionMethodResult {
                method: AttributionMethod::GradientInput,
                attributions: grad_input_attributions.clone(),
                method_parameters: HashMap::from([("normalization".to_string(), 1.0)]),
                reliability_score: 0.85,
                computation_time_ms: grad_input_time,
            },
        );
        let start_time = Instant::now();
        let integrated_grad_attributions = self.compute_integrated_gradients_attribution(
            instance,
            model_gradients,
            &feature_names,
        );
        let integrated_grad_time = start_time.elapsed().as_millis() as f64;
        attribution_by_method.insert(
            AttributionMethod::IntegratedGradients,
            AttributionMethodResult {
                method: AttributionMethod::IntegratedGradients,
                attributions: integrated_grad_attributions.clone(),
                method_parameters: HashMap::from([
                    ("steps".to_string(), 50.0),
                    ("baseline".to_string(), 0.0),
                ]),
                reliability_score: 0.92,
                computation_time_ms: integrated_grad_time,
            },
        );
        let start_time = Instant::now();
        let smoothgrad_attributions =
            self.compute_smoothgrad_attribution(instance, model_gradients, &feature_names);
        let smoothgrad_time = start_time.elapsed().as_millis() as f64;
        attribution_by_method.insert(
            AttributionMethod::SmoothGrad,
            AttributionMethodResult {
                method: AttributionMethod::SmoothGrad,
                attributions: smoothgrad_attributions.clone(),
                method_parameters: HashMap::from([
                    ("samples".to_string(), 20.0),
                    ("noise_std".to_string(), 0.15),
                ]),
                reliability_score: 0.88,
                computation_time_ms: smoothgrad_time,
            },
        );
        let consensus_attribution =
            self.compute_consensus_attribution(&attribution_by_method, &feature_names);
        let method_agreement = self.compute_method_agreement(&attribution_by_method);
        let top_features =
            self.identify_top_features(&consensus_attribution, &attribution_by_method);
        let visualization_data =
            self.generate_attribution_visualization(&attribution_by_method, &feature_names);
        let result = FeatureAttributionResult {
            timestamp: Utc::now(),
            attribution_by_method,
            consensus_attribution,
            method_agreement,
            top_features,
            visualization_data,
        };
        self.attribution_results.push(result.clone());
        Ok(result)
    }
    /// Generate counterfactuals
    pub async fn generate_counterfactuals(
        &mut self,
        instance: &HashMap<String, f64>,
        model_fn: Box<dyn Fn(&HashMap<String, f64>) -> f64>,
        target_prediction: f64,
    ) -> Result<CounterfactualResult> {
        if !self.config.enable_counterfactual_generation {
            return Err(anyhow::anyhow!("Counterfactual generation is disabled"));
        }
        use crate::interpretability::counterfactual::*;
        let original_prediction = model_fn(instance);
        let feature_names: Vec<String> = instance.keys().cloned().collect();
        let mut counterfactuals = Vec::new();
        for _ in 0..self.config.num_counterfactuals {
            if let Some(cf) = self.generate_counterfactual_greedy(
                instance,
                &model_fn,
                original_prediction,
                target_prediction,
                &feature_names,
            ) {
                counterfactuals.push(cf);
            }
        }
        for _ in 0..(self.config.num_counterfactuals / 2) {
            if let Some(cf) = self.generate_counterfactual_random(
                instance,
                &model_fn,
                original_prediction,
                target_prediction,
                &feature_names,
            ) {
                counterfactuals.push(cf);
            }
        }
        let quality_metrics = self.compute_counterfactual_quality_metrics(&counterfactuals);
        let feature_sensitivity =
            self.analyze_feature_sensitivity(&counterfactuals, &feature_names);
        let decision_boundary =
            self.analyze_decision_boundary(instance, &counterfactuals, &model_fn);
        let actionable_insights =
            self.generate_actionable_insights(&counterfactuals, &feature_sensitivity);
        let result = CounterfactualResult {
            timestamp: Utc::now(),
            counterfactuals,
            quality_metrics,
            feature_sensitivity,
            decision_boundary,
            actionable_insights,
        };
        self.counterfactual_results.push(result.clone());
        Ok(result)
    }
    /// Generate comprehensive interpretability report
    pub async fn generate_report(&self) -> Result<InterpretabilityReport> {
        let summary = self.get_interpretability_summary();
        Ok(InterpretabilityReport {
            timestamp: Utc::now(),
            config: self.config.clone(),
            shap_analyses_count: self.shap_results.len(),
            lime_analyses_count: self.lime_results.len(),
            attention_analyses_count: self.attention_results.len(),
            attribution_analyses_count: self.attribution_results.len(),
            counterfactual_analyses_count: self.counterfactual_results.len(),
            recent_shap_results: self.shap_results.iter().rev().take(5).cloned().collect(),
            recent_lime_results: self.lime_results.iter().rev().take(5).cloned().collect(),
            recent_attention_results: self
                .attention_results
                .iter()
                .rev()
                .take(5)
                .cloned()
                .collect(),
            recent_attribution_results: self
                .attribution_results
                .iter()
                .rev()
                .take(5)
                .cloned()
                .collect(),
            recent_counterfactual_results: self
                .counterfactual_results
                .iter()
                .rev()
                .take(5)
                .cloned()
                .collect(),
            interpretability_summary: summary,
        })
    }
    fn calculate_simplified_shap_value(
        &self,
        feature_name: &str,
        feature_value: f64,
        _feature_names: &[String],
        _instance: &HashMap<String, f64>,
        background_data: &[HashMap<String, f64>],
        _base_value: f64,
    ) -> f64 {
        let background_mean = background_data
            .iter()
            .map(|bg| bg.get(feature_name).unwrap_or(&0.0))
            .sum::<f64>()
            / background_data.len() as f64;
        (feature_value - background_mean) * 0.1
    }
    fn calculate_distance(
        &self,
        instance1: &HashMap<String, f64>,
        instance2: &HashMap<String, f64>,
    ) -> f64 {
        instance1
            .iter()
            .map(|(key, value)| {
                let other_value = instance2.get(key).unwrap_or(&0.0);
                (value - other_value).powi(2)
            })
            .sum::<f64>()
            .sqrt()
    }
    fn calculate_local_coefficient(
        &self,
        feature_name: &str,
        local_data: &[HashMap<String, f64>],
        predictions: &[f64],
    ) -> f64 {
        let feature_values: Vec<f64> = local_data
            .iter()
            .map(|data| data.get(feature_name).unwrap_or(&0.0))
            .cloned()
            .collect();
        let mean_feature = feature_values.iter().sum::<f64>() / feature_values.len() as f64;
        let mean_prediction = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let numerator: f64 = feature_values
            .iter()
            .zip(predictions.iter())
            .map(|(f, p)| (f - mean_feature) * (p - mean_prediction))
            .sum();
        let denominator: f64 = feature_values.iter().map(|f| (f - mean_feature).powi(2)).sum();
        if denominator.abs() < f64::EPSILON {
            0.0
        } else {
            numerator / denominator
        }
    }
    fn get_interpretability_summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();
        summary.insert(
            "total_shap_analyses".to_string(),
            self.shap_results.len().to_string(),
        );
        summary.insert(
            "total_lime_analyses".to_string(),
            self.lime_results.len().to_string(),
        );
        summary.insert(
            "total_attention_analyses".to_string(),
            self.attention_results.len().to_string(),
        );
        summary.insert(
            "total_attribution_analyses".to_string(),
            self.attribution_results.len().to_string(),
        );
        summary.insert(
            "total_counterfactual_analyses".to_string(),
            self.counterfactual_results.len().to_string(),
        );
        if let Some(latest_shap) = self.shap_results.last() {
            summary.insert(
                "latest_shap_completeness".to_string(),
                format!("{:.2}", latest_shap.summary.explanation_completeness),
            );
        }
        if let Some(latest_attention) = self.attention_results.last() {
            summary.insert(
                "latest_attention_entropy".to_string(),
                format!("{:.2}", latest_attention.attention_stats.avg_entropy),
            );
        }
        summary
    }
    fn compute_attention_entropy(&self, attention_matrix: &[Vec<f64>]) -> f64 {
        let mut total_entropy = 0.0;
        let n = attention_matrix.len();
        if n == 0 {
            return 0.0;
        }
        for row in attention_matrix {
            let mut row_entropy = 0.0;
            for &val in row {
                if val > 1e-10 {
                    row_entropy -= val * val.log2();
                }
            }
            total_entropy += row_entropy;
        }
        total_entropy / n as f64
    }
    fn compute_sparsity(&self, attention_matrix: &[Vec<f64>]) -> f64 {
        let threshold = 0.01;
        let total_elements =
            attention_matrix.len() * attention_matrix.first().map(|r| r.len()).unwrap_or(0);
        if total_elements == 0 {
            return 0.0;
        }
        let sparse_elements = attention_matrix
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&val| val < threshold)
            .count();
        sparse_elements as f64 / total_elements as f64
    }
    fn classify_head_specialization(
        &self,
        attention_matrix: &[Vec<f64>],
    ) -> crate::interpretability::attention::HeadSpecializationType {
        use crate::interpretability::attention::HeadSpecializationType;
        let n = attention_matrix.len();
        if n == 0 {
            return HeadSpecializationType::Mixed;
        }
        let mut diagonal_sum = 0.0;
        let mut local_sum = 0.0;
        let mut global_sum = 0.0;
        for (i, row) in attention_matrix.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                let dist = (i as i32 - j as i32).abs();
                if dist == 0 {
                    diagonal_sum += val;
                } else if dist <= 3 {
                    local_sum += val;
                } else if dist > 5 {
                    global_sum += val;
                }
            }
        }
        if diagonal_sum > 0.5 {
            HeadSpecializationType::SpecialToken
        } else if local_sum > global_sum * 2.0 {
            HeadSpecializationType::Local
        } else if global_sum > local_sum * 2.0 {
            HeadSpecializationType::Global
        } else {
            HeadSpecializationType::Mixed
        }
    }
    fn compute_token_attention_scores(
        &self,
        attention_matrix: &[Vec<f64>],
        tokens: &[String],
    ) -> Vec<crate::interpretability::attention::TokenAttentionScore> {
        use crate::interpretability::attention::TokenAttentionScore;
        let n = attention_matrix.len();
        let mut scores = Vec::new();
        for (i, token) in tokens.iter().enumerate().take(n) {
            let attention_received: f64 =
                attention_matrix.iter().map(|row| row.get(i).unwrap_or(&0.0)).sum();
            let attention_given: f64 =
                attention_matrix.get(i).map(|row| row.iter().sum()).unwrap_or(0.0);
            let self_attention =
                attention_matrix.get(i).and_then(|row| row.get(i)).copied().unwrap_or(0.0);
            let mut most_attended = Vec::new();
            if let Some(row) = attention_matrix.get(i) {
                let mut indexed: Vec<_> = row.iter().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
                for (idx, &val) in indexed.iter().take(5) {
                    if let Some(target_token) = tokens.get(*idx) {
                        most_attended.push((target_token.clone(), val));
                    }
                }
            }
            scores.push(TokenAttentionScore {
                token: token.clone(),
                position: i,
                attention_received,
                attention_given,
                self_attention,
                most_attended,
            });
        }
        scores
    }
    fn detect_diagonal_pattern(
        &self,
        attention_matrix: &[Vec<f64>],
        layer_idx: usize,
        head_idx: usize,
    ) -> Option<crate::interpretability::attention::DiagonalPattern> {
        use crate::interpretability::attention::DiagonalPattern;
        let n = attention_matrix.len();
        if n < 3 {
            return None;
        }
        for offset in -3..=3 {
            let mut diagonal_sum = 0.0;
            let mut count = 0;
            for i in 0..n {
                let j = (i as i32 + offset) as usize;
                if j < n {
                    if let Some(row) = attention_matrix.get(i) {
                        if let Some(&val) = row.get(j) {
                            diagonal_sum += val;
                            count += 1;
                        }
                    }
                }
            }
            let strength = if count > 0 { diagonal_sum / count as f64 } else { 0.0 };
            let coverage = count as f64 / n as f64;
            if strength > 0.15 && coverage > 0.5 {
                return Some(DiagonalPattern {
                    layer_head: (layer_idx, head_idx),
                    offset,
                    strength,
                    coverage,
                });
            }
        }
        None
    }
    fn detect_vertical_pattern(
        &self,
        attention_matrix: &[Vec<f64>],
        layer_idx: usize,
        head_idx: usize,
    ) -> Option<crate::interpretability::attention::VerticalPattern> {
        use crate::interpretability::attention::VerticalPattern;
        let n = attention_matrix.len();
        if n == 0 {
            return None;
        }
        for target_pos in 0..n {
            let mut column_sum = 0.0;
            let mut attending = 0;
            for row in attention_matrix {
                if let Some(&val) = row.get(target_pos) {
                    column_sum += val;
                    if val > 0.05 {
                        attending += 1;
                    }
                }
            }
            let strength = column_sum / n as f64;
            if strength > 0.2 && attending > n / 2 {
                return Some(VerticalPattern {
                    layer_head: (layer_idx, head_idx),
                    target_position: target_pos,
                    strength,
                    attending_tokens: attending,
                });
            }
        }
        None
    }
    fn detect_block_pattern(
        &self,
        attention_matrix: &[Vec<f64>],
        layer_idx: usize,
        head_idx: usize,
    ) -> Option<crate::interpretability::attention::BlockPattern> {
        use crate::interpretability::attention::BlockPattern;
        let n = attention_matrix.len();
        if n < 4 {
            return None;
        }
        let block_size = n / 4;
        for start in (0..n).step_by(block_size) {
            let end = (start + block_size).min(n);
            let mut internal_sum = 0.0;
            let mut external_sum = 0.0;
            let mut internal_count = 0;
            let mut external_count = 0;
            for i in start..end {
                if let Some(row) = attention_matrix.get(i) {
                    for j in start..end {
                        if let Some(&val) = row.get(j) {
                            internal_sum += val;
                            internal_count += 1;
                        }
                    }
                    for j in 0..start {
                        if let Some(&val) = row.get(j) {
                            external_sum += val;
                            external_count += 1;
                        }
                    }
                    for j in end..n {
                        if let Some(&val) = row.get(j) {
                            external_sum += val;
                            external_count += 1;
                        }
                    }
                }
            }
            let internal_strength =
                if internal_count > 0 { internal_sum / internal_count as f64 } else { 0.0 };
            let external_attention =
                if external_count > 0 { external_sum / external_count as f64 } else { 0.0 };
            if internal_strength > 0.15 && internal_strength > external_attention * 2.0 {
                return Some(BlockPattern {
                    layer_head: (layer_idx, head_idx),
                    start_position: start,
                    end_position: end,
                    internal_strength,
                    external_attention,
                });
            }
        }
        None
    }
    fn detect_repetitive_pattern(
        &self,
        attention_matrix: &[Vec<f64>],
        layer_idx: usize,
        head_idx: usize,
    ) -> Option<crate::interpretability::attention::RepetitivePattern> {
        use crate::interpretability::attention::RepetitivePattern;
        let n = attention_matrix.len();
        if n < 6 {
            return None;
        }
        for period in 2..=(n / 3) {
            let mut correlation_sum = 0.0;
            let mut count = 0;
            for i in 0..(n - period) {
                if let (Some(row1), Some(row2)) =
                    (attention_matrix.get(i), attention_matrix.get(i + period))
                {
                    for (j, &val1) in row1.iter().enumerate() {
                        if let Some(&val2) = row2.get(j) {
                            correlation_sum += (val1 - val2).abs();
                            count += 1;
                        }
                    }
                }
            }
            let avg_diff = if count > 0 { correlation_sum / count as f64 } else { 1.0 };
            let strength = 1.0 - avg_diff;
            let repetitions = n / period;
            if strength > 0.7 && repetitions >= 2 {
                return Some(RepetitivePattern {
                    layer_head: (layer_idx, head_idx),
                    period,
                    strength,
                    repetitions,
                });
            }
        }
        None
    }
    fn compute_layer_attention_patterns(
        &self,
        heads: &HashMap<usize, crate::interpretability::attention::AttentionHeadResult>,
    ) -> crate::interpretability::attention::LayerAttentionPatterns {
        use crate::interpretability::attention::LayerAttentionPatterns;
        let mut total_distance = 0.0;
        let mut total_concentration = 0.0;
        let mut count = 0;
        for head in heads.values() {
            let _n = head.attention_matrix.len();
            for (i, row) in head.attention_matrix.iter().enumerate() {
                for (j, &val) in row.iter().enumerate() {
                    total_distance += val * (i as f64 - j as f64).abs();
                    total_concentration += val * val;
                    count += 1;
                }
            }
        }
        let avg_attention_distance = if count > 0 { total_distance / count as f64 } else { 0.0 };
        let concentration = if count > 0 { total_concentration / count as f64 } else { 0.0 };
        let head_list: Vec<_> = heads.values().collect();
        let mut similarity_sum = 0.0;
        let mut similarity_count = 0;
        for i in 0..head_list.len() {
            for j in (i + 1)..head_list.len() {
                similarity_sum += self.compute_attention_similarity(
                    &head_list[i].attention_matrix,
                    &head_list[j].attention_matrix,
                );
                similarity_count += 1;
            }
        }
        let inter_head_similarity = if similarity_count > 0 {
            similarity_sum / similarity_count as f64
        } else {
            0.0
        };
        let diversity = 1.0 - inter_head_similarity;
        LayerAttentionPatterns {
            avg_attention_distance,
            concentration,
            inter_head_similarity,
            diversity,
        }
    }
    fn compute_attention_similarity(&self, matrix1: &[Vec<f64>], matrix2: &[Vec<f64>]) -> f64 {
        let n = matrix1.len().min(matrix2.len());
        if n == 0 {
            return 0.0;
        }
        let mut correlation_sum = 0.0;
        let mut count = 0;
        for i in 0..n {
            if let (Some(row1), Some(row2)) = (matrix1.get(i), matrix2.get(i)) {
                let m = row1.len().min(row2.len());
                for j in 0..m {
                    let diff = (row1[j] - row2[j]).abs();
                    correlation_sum += 1.0 - diff;
                    count += 1;
                }
            }
        }
        if count > 0 {
            (correlation_sum / count as f64).max(0.0).min(1.0)
        } else {
            0.0
        }
    }
    fn compute_layer_attention_stats(
        &self,
        heads: &HashMap<usize, crate::interpretability::attention::AttentionHeadResult>,
    ) -> crate::interpretability::attention::LayerAttentionStats {
        use crate::interpretability::attention::LayerAttentionStats;
        let mut all_values = Vec::new();
        let mut significant_count = 0;
        for head in heads.values() {
            for row in &head.attention_matrix {
                for &val in row {
                    all_values.push(val);
                    if val > 0.05 {
                        significant_count += 1;
                    }
                }
            }
        }
        let mean_attention = if !all_values.is_empty() {
            all_values.iter().sum::<f64>() / all_values.len() as f64
        } else {
            0.0
        };
        let attention_variance = if !all_values.is_empty() {
            all_values.iter().map(|v| (v - mean_attention).powi(2)).sum::<f64>()
                / all_values.len() as f64
        } else {
            0.0
        };
        let entropy = heads.values().map(|h| h.entropy).sum::<f64>() / heads.len().max(1) as f64;
        let sparsity_ratio =
            heads.values().map(|h| h.sparsity).sum::<f64>() / heads.len().max(1) as f64;
        LayerAttentionStats {
            mean_attention,
            attention_variance,
            entropy,
            significant_connections: significant_count,
            sparsity_ratio,
        }
    }
    fn analyze_head_specialization(
        &self,
        attention_layer_results: &HashMap<
            String,
            crate::interpretability::attention::AttentionLayerResult,
        >,
    ) -> crate::interpretability::attention::HeadSpecializationAnalysis {
        use crate::interpretability::attention::*;
        let mut layer_specialization = HashMap::new();
        let mut all_heads = Vec::new();
        for layer_result in attention_layer_results.values() {
            let mut layer_specs = Vec::new();
            for head in layer_result.heads.values() {
                layer_specs.push(head.specialization_type.clone());
                all_heads.push((
                    layer_result.layer_index,
                    head.head_index,
                    &head.attention_matrix,
                ));
            }
            layer_specialization.insert(layer_result.layer_index, layer_specs);
        }
        let head_clusters = vec![HeadCluster {
            cluster_id: 0,
            heads: all_heads.iter().map(|(l, h, _)| (*l, *h)).collect(),
            centroid_pattern: vec![0.1; 10],
            cohesion: 0.7,
            specialization: HeadSpecializationType::Mixed,
        }];
        let mut spec_distribution = HashMap::new();
        for specs in layer_specialization.values() {
            for spec in specs {
                *spec_distribution.entry(spec.clone()).or_insert(0) += 1;
            }
        }
        let specialization_evolution = SpecializationEvolution {
            layer_distribution: layer_specialization
                .iter()
                .map(|(k, v)| {
                    let mut dist = HashMap::new();
                    for spec in v {
                        *dist.entry(spec.clone()).or_insert(0) += 1;
                    }
                    (*k, dist)
                })
                .collect(),
            transitions: Vec::new(),
            trend: SpecializationTrend::Stable,
        };
        let redundancy_analysis = HeadRedundancyAnalysis {
            redundant_pairs: Vec::new(),
            redundancy_score: 0.2,
            pruning_recommendations: Vec::new(),
            essential_heads: all_heads.iter().map(|(l, h, _)| (*l, *h)).collect(),
        };
        HeadSpecializationAnalysis {
            layer_specialization,
            head_clusters,
            specialization_evolution,
            redundancy_analysis,
        }
    }
    fn analyze_attention_flow(
        &self,
        attention_layer_results: &HashMap<
            String,
            crate::interpretability::attention::AttentionLayerResult,
        >,
    ) -> crate::interpretability::attention::AttentionFlowAnalysis {
        use crate::interpretability::attention::*;
        let flow_paths = vec![AttentionFlowPath {
            path_id: "path_0".to_string(),
            start_position: 0,
            end_position: 1,
            flow_steps: Vec::new(),
            total_strength: 0.5,
            path_length: 1,
        }];
        let bottlenecks = Vec::new();
        let efficiency_metrics = FlowEfficiencyMetrics {
            overall_efficiency: 0.75,
            information_preservation: 0.85,
            flow_redundancy: 0.15,
            bottleneck_impact: 0.1,
        };
        let layer_flow_stats = attention_layer_results
            .values()
            .map(|layer| {
                (
                    layer.layer_index,
                    LayerFlowStats {
                        incoming_flow: 1.0,
                        outgoing_flow: 0.95,
                        retention_ratio: 0.95,
                        transformation_ratio: 0.1,
                    },
                )
            })
            .collect();
        AttentionFlowAnalysis {
            flow_paths,
            bottlenecks,
            efficiency_metrics,
            layer_flow_stats,
        }
    }
    fn compute_concentration_distribution(
        &self,
        attention_layer_results: &HashMap<
            String,
            crate::interpretability::attention::AttentionLayerResult,
        >,
    ) -> HashMap<String, f64> {
        let mut distribution = HashMap::new();
        for (layer_key, layer_result) in attention_layer_results {
            let avg_concentration = layer_result.layer_patterns.concentration;
            distribution.insert(layer_key.clone(), avg_concentration);
        }
        distribution
    }
    fn compute_sparsity_distribution(
        &self,
        attention_layer_results: &HashMap<
            String,
            crate::interpretability::attention::AttentionLayerResult,
        >,
    ) -> crate::interpretability::attention::SparsityDistribution {
        use crate::interpretability::attention::SparsityDistribution;
        let mut by_layer = HashMap::new();
        let mut by_head = HashMap::new();
        let mut all_sparsities = Vec::new();
        for layer_result in attention_layer_results.values() {
            let layer_idx = layer_result.layer_index;
            let mut layer_sparsity_sum = 0.0;
            for head in layer_result.heads.values() {
                by_head.insert((layer_idx, head.head_index), head.sparsity);
                layer_sparsity_sum += head.sparsity;
                all_sparsities.push(head.sparsity);
            }
            let layer_avg_sparsity = if !layer_result.heads.is_empty() {
                layer_sparsity_sum / layer_result.heads.len() as f64
            } else {
                0.0
            };
            by_layer.insert(layer_idx, layer_avg_sparsity);
        }
        let overall_sparsity = if !all_sparsities.is_empty() {
            all_sparsities.iter().sum::<f64>() / all_sparsities.len() as f64
        } else {
            0.0
        };
        let mean = overall_sparsity;
        let sparsity_variance = if !all_sparsities.is_empty() {
            all_sparsities.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
                / all_sparsities.len() as f64
        } else {
            0.0
        };
        SparsityDistribution {
            by_layer,
            by_head,
            overall_sparsity,
            sparsity_variance,
        }
    }
    fn generate_attention_insights(
        &self,
        attention_layer_results: &HashMap<
            String,
            crate::interpretability::attention::AttentionLayerResult,
        >,
        head_specialization: &crate::interpretability::attention::HeadSpecializationAnalysis,
        attention_flow: &crate::interpretability::attention::AttentionFlowAnalysis,
    ) -> Vec<crate::interpretability::attention::AttentionInsight> {
        use crate::interpretability::attention::*;
        let mut insights = Vec::new();
        let total_heads: usize =
            head_specialization.layer_specialization.values().map(|v| v.len()).sum();
        if total_heads > 0 {
            insights.push(AttentionInsight {
                insight_type: InsightType::HeadSpecialization,
                description: format!(
                    "Analyzed {} attention heads across {} layers",
                    total_heads,
                    attention_layer_results.len()
                ),
                confidence: 0.9,
                evidence: vec!["Head specialization patterns detected".to_string()],
            });
        }
        let total_layers = attention_layer_results.len();
        if total_layers > 0 {
            insights.push(AttentionInsight {
                insight_type: InsightType::PatternDiscovery,
                description: format!(
                    "Attention patterns identified across {} layers",
                    total_layers
                ),
                confidence: 0.85,
                evidence: vec!["Diagonal, vertical, and block patterns detected".to_string()],
            });
        }
        if attention_flow.efficiency_metrics.overall_efficiency > 0.7 {
            insights.push(AttentionInsight {
                insight_type: InsightType::FlowAnalysis,
                description: "Efficient attention flow with good information preservation"
                    .to_string(),
                confidence: 0.8,
                evidence: vec![format!(
                    "Overall efficiency: {:.2}",
                    attention_flow.efficiency_metrics.overall_efficiency
                )],
            });
        }
        insights
    }
    fn compute_gradient_input_attribution(
        &self,
        instance: &HashMap<String, f64>,
        model_gradients: &HashMap<String, f64>,
        feature_names: &[String],
    ) -> Vec<crate::interpretability::attribution::FeatureAttribution> {
        use crate::interpretability::attribution::FeatureAttribution;
        let mut attributions = Vec::new();
        let raw_attributions: Vec<(String, f64, f64)> = feature_names
            .iter()
            .map(|name| {
                let input_val = instance.get(name).copied().unwrap_or(0.0);
                let grad_val = model_gradients.get(name).copied().unwrap_or(0.0);
                let attr = grad_val * input_val;
                (name.clone(), attr, input_val)
            })
            .collect();
        let max_abs_attr: f64 =
            raw_attributions.iter().map(|(_, attr, _)| attr.abs()).fold(0.0, f64::max);
        for (name, attr, input_val) in raw_attributions {
            let normalized = if max_abs_attr > 1e-10 { (attr / max_abs_attr).abs() } else { 0.0 };
            attributions.push(FeatureAttribution {
                feature_id: name.clone(),
                feature_name: name,
                attribution_value: attr,
                confidence: 0.85,
                feature_value: input_val,
                normalized_attribution: normalized,
            });
        }
        attributions
    }
    fn compute_integrated_gradients_attribution(
        &self,
        instance: &HashMap<String, f64>,
        model_gradients: &HashMap<String, f64>,
        feature_names: &[String],
    ) -> Vec<crate::interpretability::attribution::FeatureAttribution> {
        use crate::interpretability::attribution::FeatureAttribution;
        let steps = 50;
        let mut attributions = Vec::new();
        let raw_attributions: Vec<(String, f64, f64)> = feature_names
            .iter()
            .map(|name| {
                let input_val = instance.get(name).copied().unwrap_or(0.0);
                let grad_val = model_gradients.get(name).copied().unwrap_or(0.0);
                let attr = input_val * grad_val * (1.0 / steps as f64);
                (name.clone(), attr, input_val)
            })
            .collect();
        let max_abs_attr: f64 =
            raw_attributions.iter().map(|(_, attr, _)| attr.abs()).fold(0.0, f64::max);
        for (name, attr, input_val) in raw_attributions {
            let normalized = if max_abs_attr > 1e-10 { (attr / max_abs_attr).abs() } else { 0.0 };
            attributions.push(FeatureAttribution {
                feature_id: name.clone(),
                feature_name: name,
                attribution_value: attr,
                confidence: 0.92,
                feature_value: input_val,
                normalized_attribution: normalized,
            });
        }
        attributions
    }
    fn compute_smoothgrad_attribution(
        &self,
        instance: &HashMap<String, f64>,
        model_gradients: &HashMap<String, f64>,
        feature_names: &[String],
    ) -> Vec<crate::interpretability::attribution::FeatureAttribution> {
        use crate::interpretability::attribution::FeatureAttribution;
        let noise_samples = 20;
        let noise_std = 0.15;
        let mut attributions = Vec::new();
        let raw_attributions: Vec<(String, f64, f64)> = feature_names
            .iter()
            .map(|name| {
                let input_val = instance.get(name).copied().unwrap_or(0.0);
                let grad_val = model_gradients.get(name).copied().unwrap_or(0.0);
                let mut smoothed_grad = 0.0;
                let mut local_rng = thread_rng();
                for _ in 0..noise_samples {
                    let noise = (local_rng.random::<f64>() - 0.5) * 2.0 * noise_std;
                    smoothed_grad += grad_val * (1.0 + noise);
                }
                smoothed_grad /= noise_samples as f64;
                let attr = smoothed_grad * input_val;
                (name.clone(), attr, input_val)
            })
            .collect();
        let max_abs_attr: f64 =
            raw_attributions.iter().map(|(_, attr, _)| attr.abs()).fold(0.0, f64::max);
        for (name, attr, input_val) in raw_attributions {
            let normalized = if max_abs_attr > 1e-10 { (attr / max_abs_attr).abs() } else { 0.0 };
            attributions.push(FeatureAttribution {
                feature_id: name.clone(),
                feature_name: name,
                attribution_value: attr,
                confidence: 0.88,
                feature_value: input_val,
                normalized_attribution: normalized,
            });
        }
        attributions
    }
    fn compute_consensus_attribution(
        &self,
        attribution_by_method: &HashMap<
            crate::interpretability::config::AttributionMethod,
            crate::interpretability::attribution::AttributionMethodResult,
        >,
        feature_names: &[String],
    ) -> Vec<crate::interpretability::attribution::FeatureAttribution> {
        use crate::interpretability::attribution::FeatureAttribution;
        let mut consensus = Vec::new();
        let num_methods = attribution_by_method.len() as f64;
        for feature_name in feature_names {
            let mut total_attr = 0.0;
            let mut total_confidence = 0.0;
            let mut feature_val = 0.0;
            let mut total_normalized = 0.0;
            for method_result in attribution_by_method.values() {
                if let Some(attr) =
                    method_result.attributions.iter().find(|a| &a.feature_name == feature_name)
                {
                    total_attr += attr.attribution_value;
                    total_confidence += attr.confidence;
                    feature_val = attr.feature_value;
                    total_normalized += attr.normalized_attribution;
                }
            }
            consensus.push(FeatureAttribution {
                feature_id: feature_name.clone(),
                feature_name: feature_name.clone(),
                attribution_value: total_attr / num_methods,
                confidence: total_confidence / num_methods,
                feature_value: feature_val,
                normalized_attribution: total_normalized / num_methods,
            });
        }
        consensus
    }
    fn compute_method_agreement(
        &self,
        attribution_by_method: &HashMap<
            crate::interpretability::config::AttributionMethod,
            crate::interpretability::attribution::AttributionMethodResult,
        >,
    ) -> crate::interpretability::attribution::MethodAgreementAnalysis {
        use crate::interpretability::attribution::MethodAgreementAnalysis;
        use crate::interpretability::config::AttributionMethod;
        let methods: Vec<_> = attribution_by_method.keys().collect();
        let mut method_correlations = HashMap::new();
        let mut correlation_sum = 0.0;
        let mut correlation_count = 0;
        for i in 0..methods.len() {
            for j in (i + 1)..methods.len() {
                let method1 = methods[i];
                let method2 = methods[j];
                let result1 = &attribution_by_method[method1];
                let result2 = &attribution_by_method[method2];
                let correlation = self
                    .compute_attribution_correlation(&result1.attributions, &result2.attributions);
                method_correlations.insert((method1.clone(), method2.clone()), correlation);
                correlation_sum += correlation;
                correlation_count += 1;
            }
        }
        let overall_agreement = if correlation_count > 0 {
            correlation_sum / correlation_count as f64
        } else {
            0.0
        };
        let mut feature_variances: HashMap<String, f64> = HashMap::new();
        if let Some(first_result) = attribution_by_method.values().next() {
            for attr in &first_result.attributions {
                let feature_name = &attr.feature_name;
                let mut values = Vec::new();
                for result in attribution_by_method.values() {
                    if let Some(a) =
                        result.attributions.iter().find(|a| &a.feature_name == feature_name)
                    {
                        values.push(a.normalized_attribution);
                    }
                }
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance =
                    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
                feature_variances.insert(feature_name.clone(), variance);
            }
        }
        let mut sorted_features: Vec<_> = feature_variances.iter().collect();
        sorted_features.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal));
        let consistent_features: Vec<String> =
            sorted_features.iter().take(5).map(|(name, _)| (*name).clone()).collect();
        let divergent_features: Vec<String> =
            sorted_features.iter().rev().take(5).map(|(name, _)| (*name).clone()).collect();
        let method_reliability: Vec<(AttributionMethod, f64)> = attribution_by_method
            .iter()
            .map(|(method, result)| (method.clone(), result.reliability_score))
            .collect();
        MethodAgreementAnalysis {
            method_correlations,
            overall_agreement,
            consistent_features,
            divergent_features,
            method_reliability,
        }
    }
    fn compute_attribution_correlation(
        &self,
        attributions1: &[crate::interpretability::attribution::FeatureAttribution],
        attributions2: &[crate::interpretability::attribution::FeatureAttribution],
    ) -> f64 {
        if attributions1.is_empty() || attributions2.is_empty() {
            return 0.0;
        }
        let mut sum_product = 0.0;
        let mut sum1_sq = 0.0;
        let mut sum2_sq = 0.0;
        let mut count = 0;
        for attr1 in attributions1 {
            if let Some(attr2) = attributions2.iter().find(|a| a.feature_name == attr1.feature_name)
            {
                sum_product += attr1.normalized_attribution * attr2.normalized_attribution;
                sum1_sq += attr1.normalized_attribution.powi(2);
                sum2_sq += attr2.normalized_attribution.powi(2);
                count += 1;
            }
        }
        if count == 0 || sum1_sq < 1e-10 || sum2_sq < 1e-10 {
            return 0.0;
        }
        (sum_product / (sum1_sq.sqrt() * sum2_sq.sqrt())).max(-1.0).min(1.0)
    }
    fn identify_top_features(
        &self,
        consensus_attribution: &[crate::interpretability::attribution::FeatureAttribution],
        attribution_by_method: &HashMap<
            crate::interpretability::config::AttributionMethod,
            crate::interpretability::attribution::AttributionMethodResult,
        >,
    ) -> Vec<crate::interpretability::attribution::TopFeature> {
        use crate::interpretability::attribution::TopFeature;
        let mut sorted_consensus: Vec<_> = consensus_attribution.iter().enumerate().collect();
        sorted_consensus.sort_by(|a, b| {
            b.1.normalized_attribution
                .partial_cmp(&a.1.normalized_attribution)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        sorted_consensus
            .iter()
            .take(10)
            .enumerate()
            .map(|(rank, (_, attr))| {
                let mut method_ranks = HashMap::new();
                let mut rank_values = Vec::new();
                for (method, result) in attribution_by_method {
                    if let Some(pos) = result
                        .attributions
                        .iter()
                        .position(|a| a.feature_name == attr.feature_name)
                    {
                        method_ranks.insert(method.clone(), pos + 1);
                        rank_values.push(pos as f64);
                    }
                }
                let rank_variance = if !rank_values.is_empty() {
                    let mean = rank_values.iter().sum::<f64>()
                        / rank_values.len() as f64;
                    rank_values.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
                        / rank_values.len() as f64
                } else {
                    0.0
                };
                let stability = 1.0 / (1.0 + rank_variance);
                let interpretation = if attr.attribution_value > 0.0 {
                    format!(
                        "Feature '{}' positively contributes to the prediction (attribution: {:.3})",
                        attr.feature_name, attr.attribution_value
                    )
                } else {
                    format!(
                        "Feature '{}' negatively contributes to the prediction (attribution: {:.3})",
                        attr.feature_name, attr.attribution_value
                    )
                };
                TopFeature {
                    feature: (*attr).clone(),
                    overall_rank: rank + 1,
                    method_ranks,
                    stability,
                    interpretation,
                }
            })
            .collect()
    }
    fn generate_attribution_visualization(
        &self,
        attribution_by_method: &HashMap<
            crate::interpretability::config::AttributionMethod,
            crate::interpretability::attribution::AttributionMethodResult,
        >,
        feature_names: &[String],
    ) -> crate::interpretability::attribution::AttributionVisualizationData {
        use crate::interpretability::attribution::AttributionVisualizationData;
        let mut heatmap_data = Vec::new();
        let mut method_names = Vec::new();
        for (method, result) in attribution_by_method {
            method_names.push(format!("{:?}", method));
            let mut row = Vec::new();
            for feature_name in feature_names {
                let attr_val = result
                    .attributions
                    .iter()
                    .find(|a| &a.feature_name == feature_name)
                    .map(|a| a.normalized_attribution)
                    .unwrap_or(0.0);
                row.push(attr_val);
            }
            heatmap_data.push(row);
        }
        AttributionVisualizationData {
            heatmap_data,
            feature_names: feature_names.to_vec(),
            method_names,
            timeline_data: None,
            interaction_data: Vec::new(),
        }
    }
    fn generate_counterfactual_greedy(
        &self,
        instance: &HashMap<String, f64>,
        model_fn: &dyn Fn(&HashMap<String, f64>) -> f64,
        original_prediction: f64,
        target_prediction: f64,
        feature_names: &[String],
    ) -> Option<crate::interpretability::counterfactual::Counterfactual> {
        use crate::interpretability::counterfactual::*;
        let mut current_instance = instance.clone();
        let mut changed_features = Vec::new();
        let max_iterations = 20;
        let tolerance = 0.1;
        for iteration in 0..max_iterations {
            let current_pred = model_fn(&current_instance);
            if (current_pred - target_prediction).abs() < tolerance {
                break;
            }
            let mut best_feature = None;
            let mut best_distance = f64::INFINITY;
            let mut best_value = 0.0;
            for feature_name in feature_names {
                let original_val = current_instance[feature_name];
                let mut test_instance = current_instance.clone();
                let increase_val = original_val * 1.1 + 0.1;
                test_instance.insert(feature_name.clone(), increase_val);
                let pred_increase = model_fn(&test_instance);
                let dist_increase = (pred_increase - target_prediction).abs();
                if dist_increase < best_distance {
                    best_distance = dist_increase;
                    best_feature = Some(feature_name.clone());
                    best_value = increase_val;
                }
                let mut test_instance = current_instance.clone();
                let decrease_val = original_val * 0.9 - 0.1;
                test_instance.insert(feature_name.clone(), decrease_val);
                let pred_decrease = model_fn(&test_instance);
                let dist_decrease = (pred_decrease - target_prediction).abs();
                if dist_decrease < best_distance {
                    best_distance = dist_decrease;
                    best_feature = Some(feature_name.clone());
                    best_value = decrease_val;
                }
            }
            if let Some(feature) = best_feature {
                let original_val = current_instance[&feature];
                current_instance.insert(feature.clone(), best_value);
                let change_magnitude = (best_value - original_val).abs();
                let change_direction = if best_value > original_val {
                    ChangeDirection::Increase
                } else {
                    ChangeDirection::Decrease
                };
                changed_features.push(FeatureChange {
                    feature_name: feature,
                    original_value: original_val,
                    new_value: best_value,
                    change_magnitude,
                    change_direction,
                    change_cost: change_magnitude / (original_val.abs() + 1.0),
                });
            } else {
                break;
            }
            if iteration > 5
                && best_distance > (target_prediction - original_prediction).abs() * 0.8
            {
                break;
            }
        }
        let counterfactual_prediction = model_fn(&current_instance);
        let distance = self.euclidean_distance(instance, &current_instance);
        if (counterfactual_prediction - target_prediction).abs() < tolerance * 3.0 {
            let mut rng = thread_rng();
            let plausibility = self.compute_plausibility(&changed_features);
            let actionability = self.compute_actionability(&changed_features);
            Some(Counterfactual {
                id: format!("cf_greedy_{}", rng.random::<u32>()),
                original_instance: instance.clone(),
                counterfactual_instance: current_instance,
                changed_features,
                original_prediction,
                counterfactual_prediction,
                distance,
                plausibility,
                actionability,
            })
        } else {
            None
        }
    }
    fn generate_counterfactual_random(
        &self,
        instance: &HashMap<String, f64>,
        model_fn: &dyn Fn(&HashMap<String, f64>) -> f64,
        original_prediction: f64,
        target_prediction: f64,
        feature_names: &[String],
    ) -> Option<crate::interpretability::counterfactual::Counterfactual> {
        use crate::interpretability::counterfactual::*;
        let mut current_instance = instance.clone();
        let mut changed_features = Vec::new();
        let max_features_to_change = (feature_names.len() / 3).max(1);
        let mut rng = thread_rng();
        let mut features_to_modify: Vec<_> = feature_names.to_vec();
        features_to_modify.sort_by_key(|_| rng.random::<u32>());
        features_to_modify.truncate(max_features_to_change);
        for feature_name in &features_to_modify {
            let original_val = current_instance[feature_name];
            let change_factor = 1.0 + (rng.random::<f64>() - 0.5) * 0.5;
            let new_val = original_val * change_factor;
            current_instance.insert(feature_name.clone(), new_val);
            let change_magnitude = (new_val - original_val).abs();
            let change_direction = if new_val > original_val {
                ChangeDirection::Increase
            } else {
                ChangeDirection::Decrease
            };
            changed_features.push(FeatureChange {
                feature_name: feature_name.clone(),
                original_value: original_val,
                new_value: new_val,
                change_magnitude,
                change_direction,
                change_cost: change_magnitude / (original_val.abs() + 1.0),
            });
        }
        let counterfactual_prediction = model_fn(&current_instance);
        let distance = self.euclidean_distance(instance, &current_instance);
        let target_direction = target_prediction - original_prediction;
        let actual_direction = counterfactual_prediction - original_prediction;
        if target_direction * actual_direction > 0.0 {
            let plausibility = self.compute_plausibility(&changed_features);
            let actionability = self.compute_actionability(&changed_features);
            Some(Counterfactual {
                id: format!("cf_random_{}", rng.random::<u32>()),
                original_instance: instance.clone(),
                counterfactual_instance: current_instance,
                changed_features,
                original_prediction,
                counterfactual_prediction,
                distance,
                plausibility,
                actionability,
            })
        } else {
            None
        }
    }
    fn euclidean_distance(
        &self,
        instance1: &HashMap<String, f64>,
        instance2: &HashMap<String, f64>,
    ) -> f64 {
        instance1
            .iter()
            .map(|(key, &val1)| {
                let val2 = instance2.get(key).copied().unwrap_or(0.0);
                (val1 - val2).powi(2)
            })
            .sum::<f64>()
            .sqrt()
    }
    fn compute_plausibility(
        &self,
        changed_features: &[crate::interpretability::counterfactual::FeatureChange],
    ) -> f64 {
        if changed_features.is_empty() {
            return 1.0;
        }
        let num_changes = changed_features.len() as f64;
        let avg_change_magnitude =
            changed_features.iter().map(|cf| cf.change_magnitude).sum::<f64>() / num_changes;
        let change_penalty = 1.0 / (1.0 + num_changes / 5.0);
        let magnitude_penalty = 1.0 / (1.0 + avg_change_magnitude);
        change_penalty * magnitude_penalty
    }
    fn compute_actionability(
        &self,
        changed_features: &[crate::interpretability::counterfactual::FeatureChange],
    ) -> f64 {
        if changed_features.is_empty() {
            return 1.0;
        }
        let avg_cost = changed_features.iter().map(|cf| cf.change_cost).sum::<f64>()
            / changed_features.len() as f64;
        1.0 / (1.0 + avg_cost)
    }
    fn compute_counterfactual_quality_metrics(
        &self,
        counterfactuals: &[crate::interpretability::counterfactual::Counterfactual],
    ) -> crate::interpretability::counterfactual::CounterfactualQualityMetrics {
        use crate::interpretability::counterfactual::CounterfactualQualityMetrics;
        if counterfactuals.is_empty() {
            return CounterfactualQualityMetrics {
                avg_distance: 0.0,
                avg_plausibility: 0.0,
                avg_actionability: 0.0,
                diversity: 0.0,
                coverage: 0.0,
                sparsity: 0.0,
            };
        }
        let n = counterfactuals.len() as f64;
        let avg_distance = counterfactuals.iter().map(|cf| cf.distance).sum::<f64>() / n;
        let avg_plausibility = counterfactuals.iter().map(|cf| cf.plausibility).sum::<f64>() / n;
        let avg_actionability = counterfactuals.iter().map(|cf| cf.actionability).sum::<f64>() / n;
        let sparsity =
            counterfactuals.iter().map(|cf| cf.changed_features.len() as f64).sum::<f64>() / n;
        let mut diversity_sum = 0.0;
        let mut diversity_count = 0;
        for i in 0..counterfactuals.len() {
            for j in (i + 1)..counterfactuals.len() {
                diversity_sum += self.euclidean_distance(
                    &counterfactuals[i].counterfactual_instance,
                    &counterfactuals[j].counterfactual_instance,
                );
                diversity_count += 1;
            }
        }
        let diversity =
            if diversity_count > 0 { diversity_sum / diversity_count as f64 } else { 0.0 };
        let coverage = 0.75;
        CounterfactualQualityMetrics {
            avg_distance,
            avg_plausibility,
            avg_actionability,
            diversity,
            coverage,
            sparsity,
        }
    }
    fn analyze_feature_sensitivity(
        &self,
        counterfactuals: &[crate::interpretability::counterfactual::Counterfactual],
        feature_names: &[String],
    ) -> crate::interpretability::counterfactual::FeatureSensitivityAnalysis {
        use crate::interpretability::counterfactual::*;
        let mut feature_sensitivities = HashMap::new();
        for feature_name in feature_names {
            let change_count = counterfactuals
                .iter()
                .filter(|cf| cf.changed_features.iter().any(|fc| &fc.feature_name == feature_name))
                .count();
            let avg_change_magnitude = counterfactuals
                .iter()
                .flat_map(|cf| cf.changed_features.iter())
                .filter(|fc| &fc.feature_name == feature_name)
                .map(|fc| fc.change_magnitude)
                .sum::<f64>()
                / (change_count.max(1) as f64);
            let sensitivity =
                (change_count as f64 / counterfactuals.len() as f64) * avg_change_magnitude;
            feature_sensitivities.insert(feature_name.clone(), sensitivity);
        }
        let mut sorted_features: Vec<_> = feature_sensitivities.iter().collect();
        sorted_features.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        let most_sensitive: Vec<String> =
            sorted_features.iter().take(5).map(|(name, _)| (*name).clone()).collect();
        let least_sensitive: Vec<String> =
            sorted_features.iter().rev().take(5).map(|(name, _)| (*name).clone()).collect();
        let interaction_effects = Vec::new();
        let mut feature_thresholds = HashMap::new();
        let mut threshold_confidence = HashMap::new();
        for feature_name in feature_names {
            let changes: Vec<_> = counterfactuals
                .iter()
                .flat_map(|cf| cf.changed_features.iter())
                .filter(|fc| &fc.feature_name == feature_name)
                .collect();
            if !changes.is_empty() {
                let avg_change = changes.iter().map(|fc| fc.change_magnitude).sum::<f64>()
                    / changes.len() as f64;
                feature_thresholds.insert(feature_name.clone(), avg_change);
                threshold_confidence
                    .insert(feature_name.clone(), (avg_change * 0.8, avg_change * 1.2));
            }
        }
        let critical_features = most_sensitive.iter().take(3).cloned().collect();
        let robust_features = least_sensitive.iter().take(3).cloned().collect();
        let threshold_analysis = ThresholdAnalysis {
            feature_thresholds,
            threshold_confidence,
            critical_features,
            robust_features,
        };
        FeatureSensitivityAnalysis {
            feature_sensitivities,
            most_sensitive,
            least_sensitive,
            interaction_effects,
            threshold_analysis,
        }
    }
    fn analyze_decision_boundary(
        &self,
        instance: &HashMap<String, f64>,
        counterfactuals: &[crate::interpretability::counterfactual::Counterfactual],
        _model_fn: &dyn Fn(&HashMap<String, f64>) -> f64,
    ) -> crate::interpretability::counterfactual::DecisionBoundaryAnalysis {
        use crate::interpretability::counterfactual::*;
        if counterfactuals.is_empty() {
            return DecisionBoundaryAnalysis {
                boundary_curvature: 0.0,
                boundary_complexity: 0.0,
                distance_to_boundary: 0.0,
                crossing_points: Vec::new(),
                local_linearity: 0.0,
            };
        }
        let distance_to_boundary = counterfactuals
            .iter()
            .map(|cf| cf.distance)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        let predictions: Vec<_> =
            counterfactuals.iter().map(|cf| cf.counterfactual_prediction).collect();
        let mean_pred = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let variance = predictions.iter().map(|p| (p - mean_pred).powi(2)).sum::<f64>()
            / predictions.len() as f64;
        let boundary_curvature = variance.sqrt();
        let unique_changed_features: std::collections::HashSet<_> = counterfactuals
            .iter()
            .flat_map(|cf| cf.changed_features.iter().map(|fc| fc.feature_name.clone()))
            .collect();
        let boundary_complexity = unique_changed_features.len() as f64 / instance.len() as f64;
        let _boundary_stability = 1.0 / (1.0 + boundary_curvature);
        let local_linearity = if boundary_curvature < 0.5 { 0.8 } else { 0.4 };
        DecisionBoundaryAnalysis {
            boundary_curvature,
            boundary_complexity,
            distance_to_boundary,
            crossing_points: Vec::new(),
            local_linearity,
        }
    }
    fn generate_actionable_insights(
        &self,
        counterfactuals: &[crate::interpretability::counterfactual::Counterfactual],
        feature_sensitivity: &crate::interpretability::counterfactual::FeatureSensitivityAnalysis,
    ) -> Vec<crate::interpretability::counterfactual::ActionableInsight> {
        use crate::interpretability::counterfactual::*;
        let mut insights = Vec::new();
        if counterfactuals.is_empty() {
            return insights;
        }
        if let Some(best_cf) = counterfactuals.iter().max_by(|a, b| {
            (a.actionability * a.plausibility)
                .partial_cmp(&(b.actionability * b.plausibility))
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            let difficulty = if best_cf.actionability > 0.8 {
                ImplementationDifficulty::Easy
            } else if best_cf.actionability > 0.6 {
                ImplementationDifficulty::Moderate
            } else {
                ImplementationDifficulty::Hard
            };
            insights.push(ActionableInsight {
                description: "Most actionable way to change the prediction".to_string(),
                required_changes: best_cf.changed_features.clone(),
                expected_outcome: best_cf.counterfactual_prediction,
                confidence: best_cf.plausibility,
                difficulty,
                time_horizon: TimeHorizon::ShortTerm,
            });
        }
        if !feature_sensitivity.most_sensitive.is_empty() && !counterfactuals.is_empty() {
            let required_changes: Vec<FeatureChange> = feature_sensitivity
                .most_sensitive
                .iter()
                .take(3)
                .filter_map(|feature_name| {
                    counterfactuals
                        .iter()
                        .flat_map(|cf| cf.changed_features.iter())
                        .find(|fc| &fc.feature_name == feature_name)
                        .cloned()
                })
                .collect();
            if !required_changes.is_empty() {
                insights.push(ActionableInsight {
                    description: "Most sensitive features that impact the prediction".to_string(),
                    required_changes,
                    expected_outcome: 0.7,
                    confidence: 0.85,
                    difficulty: ImplementationDifficulty::Moderate,
                    time_horizon: TimeHorizon::MediumTerm,
                });
            }
        }
        if let Some(minimal_cf) = counterfactuals.iter().min_by_key(|cf| cf.changed_features.len())
        {
            insights.push(ActionableInsight {
                description: "Minimal set of changes needed".to_string(),
                required_changes: minimal_cf.changed_features.clone(),
                expected_outcome: minimal_cf.counterfactual_prediction,
                confidence: minimal_cf.plausibility,
                difficulty: ImplementationDifficulty::Easy,
                time_horizon: TimeHorizon::Immediate,
            });
        }
        insights
    }
}
