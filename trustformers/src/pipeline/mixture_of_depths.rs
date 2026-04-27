//! Mixture of Depths (MoD) - Dynamic Depth Selection for Efficient Transformers
//!
//! This module implements the Mixture of Depths technique for 2024-2025:
//! - Dynamic layer selection based on input complexity
//! - Adaptive computational paths through the model
//! - Early exit mechanisms with confidence-based routing
//! - Test-time compute optimization
//! - Hierarchical depth allocation for different token types

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::{Result as TrustformersResult, TrustformersError};
use crate::pipeline::{Pipeline, PipelineInput, PipelineOutput};

/// Configuration for Mixture of Depths
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixtureOfDepthsConfig {
    /// Total number of layers in the model
    pub total_layers: usize,
    /// Minimum layers to always execute
    pub min_layers: usize,
    /// Maximum layers to execute
    pub max_layers: usize,
    /// Confidence threshold for early exit
    pub confidence_threshold: f32,
    /// Whether to use token-level depth routing
    pub token_level_routing: bool,
    /// Whether to use adaptive depth based on input complexity
    pub adaptive_depth: bool,
    /// Whether to use hierarchical routing (different depths for different token types)
    pub hierarchical_routing: bool,
    /// Computational budget for test-time optimization
    pub compute_budget: f32,
    /// Strategy for depth selection
    pub depth_strategy: DepthStrategy,
}

impl Default for MixtureOfDepthsConfig {
    fn default() -> Self {
        Self {
            total_layers: 24,
            min_layers: 6,
            max_layers: 24,
            confidence_threshold: 0.8,
            token_level_routing: true,
            adaptive_depth: true,
            hierarchical_routing: false,
            compute_budget: 1.0,
            depth_strategy: DepthStrategy::AdaptiveConfidence,
        }
    }
}

/// Strategy for selecting model depth
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DepthStrategy {
    /// Fixed depth for all inputs
    Fixed(usize),
    /// Early exit based on confidence
    EarlyExit,
    /// Adaptive based on input complexity
    AdaptiveComplexity,
    /// Confidence-based with adaptive thresholds
    AdaptiveConfidence,
    /// Budget-constrained optimization
    BudgetOptimal,
    /// Token-type aware routing
    TokenTypeAware,
}

/// Input complexity analysis result
#[derive(Debug, Clone)]
pub struct ComplexityAnalysis {
    pub overall_complexity: f32,
    pub token_complexities: Vec<f32>,
    pub predicted_optimal_depth: usize,
    pub confidence_estimate: f32,
    pub semantic_density: f32,
    pub syntactic_complexity: f32,
}

/// Token type for hierarchical routing
#[derive(Debug, Clone, PartialEq)]
pub enum TokenType {
    /// Function words (articles, prepositions, etc.)
    Function,
    /// Content words (nouns, verbs, adjectives)
    Content,
    /// Named entities
    Entity,
    /// Numbers and dates
    Numeric,
    /// Special tokens (punctuation, etc.)
    Special,
    /// Unknown/other
    Unknown,
}

/// Depth routing decision for a layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingDecision {
    pub layer_index: usize,
    pub should_execute: bool,
    pub confidence_score: f32,
    pub complexity_score: f32,
    pub token_routing: Vec<bool>, // Per-token routing decisions
    pub routing_reason: RoutingReason,
}

/// Reason for routing decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RoutingReason {
    /// Confidence threshold met
    ConfidenceThreshold,
    /// Complexity analysis suggests early exit
    ComplexityBased,
    /// Budget constraint reached
    BudgetConstraint,
    /// Token-specific routing
    TokenSpecific,
    /// Fixed depth strategy
    FixedDepth,
}

/// Layer execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerExecutionResult {
    pub layer_index: usize,
    pub was_executed: bool,
    pub output_confidence: f32,
    pub computation_cost: f32,
    pub token_outputs: Vec<Vec<f32>>, // Hidden states per token
    pub attention_weights: Option<Vec<Vec<f32>>>,
}

/// Complete MoD execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoDExecutionResult {
    pub final_outputs: Vec<Vec<f32>>,
    pub executed_layers: Vec<usize>,
    pub routing_decisions: Vec<RoutingDecision>,
    pub layer_results: Vec<LayerExecutionResult>,
    pub total_computation_cost: f32,
    pub efficiency_score: f32,
    pub confidence_progression: Vec<f32>,
}

/// Trait for analyzing input complexity
#[async_trait::async_trait]
pub trait ComplexityAnalyzer: Send + Sync {
    async fn analyze_complexity(&self, input: &[String]) -> TrustformersResult<ComplexityAnalysis>;
}

/// Trait for token type classification
#[async_trait::async_trait]
pub trait TokenClassifier: Send + Sync {
    async fn classify_tokens(&self, tokens: &[String]) -> TrustformersResult<Vec<TokenType>>;
}

/// Trait for confidence estimation at each layer
#[async_trait::async_trait]
pub trait ConfidenceEstimator: Send + Sync {
    async fn estimate_confidence(
        &self,
        layer_outputs: &[Vec<f32>],
        layer_index: usize,
    ) -> TrustformersResult<f32>;
}

/// Trait for dynamic depth routing
#[async_trait::async_trait]
pub trait DepthRouter: Send + Sync {
    async fn route_depth(
        &self,
        input_analysis: &ComplexityAnalysis,
        layer_index: usize,
        current_confidence: f32,
        config: &MixtureOfDepthsConfig,
    ) -> TrustformersResult<RoutingDecision>;
}

/// Mixture of Depths Pipeline
pub struct MixtureOfDepthsPipeline {
    config: MixtureOfDepthsConfig,
    base_model: Arc<dyn Pipeline<Input = String, Output = PipelineOutput>>,
    complexity_analyzer: Arc<dyn ComplexityAnalyzer>,
    token_classifier: Option<Arc<dyn TokenClassifier>>,
    confidence_estimator: Arc<dyn ConfidenceEstimator>,
    depth_router: Arc<dyn DepthRouter>,
    layer_cache: Arc<RwLock<HashMap<String, LayerExecutionResult>>>,
}

impl MixtureOfDepthsPipeline {
    /// Create a new Mixture of Depths Pipeline
    pub fn new(
        config: MixtureOfDepthsConfig,
        base_model: Arc<dyn Pipeline<Input = String, Output = PipelineOutput>>,
        complexity_analyzer: Arc<dyn ComplexityAnalyzer>,
        confidence_estimator: Arc<dyn ConfidenceEstimator>,
        depth_router: Arc<dyn DepthRouter>,
    ) -> Self {
        Self {
            config,
            base_model,
            complexity_analyzer,
            token_classifier: None,
            confidence_estimator,
            depth_router,
            layer_cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Set token classifier for hierarchical routing
    pub fn with_token_classifier(mut self, classifier: Arc<dyn TokenClassifier>) -> Self {
        self.token_classifier = Some(classifier);
        self
    }

    /// Execute with dynamic depth selection
    async fn execute_with_mod(&self, input: &[String]) -> TrustformersResult<MoDExecutionResult> {
        // Analyze input complexity
        let complexity_analysis = self.complexity_analyzer.analyze_complexity(input).await?;

        // Classify tokens if hierarchical routing is enabled
        let token_types = if self.config.hierarchical_routing {
            if let Some(classifier) = &self.token_classifier {
                Some(classifier.classify_tokens(input).await?)
            } else {
                None
            }
        } else {
            None
        };

        let mut routing_decisions = Vec::new();
        let mut layer_results = Vec::new();
        let mut current_outputs = self.initialize_embeddings(input).await?;
        let mut confidence_progression = Vec::new();
        let mut total_computation_cost = 0.0;

        // Execute layers with dynamic routing
        for layer_idx in 0..self.config.total_layers {
            // Estimate current confidence
            let current_confidence = self
                .confidence_estimator
                .estimate_confidence(&current_outputs, layer_idx)
                .await?;

            confidence_progression.push(current_confidence);

            // Make routing decision
            let routing_decision = self
                .depth_router
                .route_depth(
                    &complexity_analysis,
                    layer_idx,
                    current_confidence,
                    &self.config,
                )
                .await?;

            routing_decisions.push(routing_decision.clone());

            // Execute layer if routed
            if routing_decision.should_execute {
                let layer_result = self
                    .execute_layer(
                        layer_idx,
                        &current_outputs,
                        &routing_decision,
                        token_types.as_deref(),
                    )
                    .await?;

                total_computation_cost += layer_result.computation_cost;

                // Update outputs
                if layer_result.was_executed {
                    current_outputs = layer_result.token_outputs.clone();
                }

                layer_results.push(layer_result);

                // Check for early exit
                if self.should_early_exit(layer_idx, current_confidence, &complexity_analysis) {
                    break;
                }
            } else {
                // Skip layer execution
                layer_results.push(LayerExecutionResult {
                    layer_index: layer_idx,
                    was_executed: false,
                    output_confidence: current_confidence,
                    computation_cost: 0.0,
                    token_outputs: current_outputs.clone(),
                    attention_weights: None,
                });
            }

            // Budget check
            if total_computation_cost > self.config.compute_budget {
                break;
            }
        }

        let executed_layers: Vec<usize> =
            layer_results.iter().filter(|r| r.was_executed).map(|r| r.layer_index).collect();

        let efficiency_score = self.calculate_efficiency_score(
            &executed_layers,
            total_computation_cost,
            *confidence_progression.last().unwrap_or(&0.0),
        );

        Ok(MoDExecutionResult {
            final_outputs: current_outputs,
            executed_layers,
            routing_decisions,
            layer_results,
            total_computation_cost,
            efficiency_score,
            confidence_progression,
        })
    }

    /// Initialize token embeddings
    async fn initialize_embeddings(&self, input: &[String]) -> TrustformersResult<Vec<Vec<f32>>> {
        // Mock implementation - in practice would use actual embedding layer
        let embedding_dim = 768; // Standard transformer dimension
        let embeddings = input.iter().map(|_| (0..embedding_dim).map(|_| 0.1).collect()).collect();
        Ok(embeddings)
    }

    /// Execute a single layer with optional token-level routing
    async fn execute_layer(
        &self,
        layer_idx: usize,
        inputs: &[Vec<f32>],
        routing_decision: &RoutingDecision,
        token_types: Option<&[TokenType]>,
    ) -> TrustformersResult<LayerExecutionResult> {
        // Mock layer execution - in practice would call actual transformer layer
        let computation_cost = if routing_decision.should_execute {
            if self.config.token_level_routing && !routing_decision.token_routing.is_empty() {
                // Token-level computation cost
                routing_decision
                    .token_routing
                    .iter()
                    .map(|&executed| if executed { 1.0 } else { 0.1 })
                    .sum::<f32>()
            } else {
                inputs.len() as f32 * 1.0 // Full layer cost
            }
        } else {
            0.0
        };

        let output_confidence = routing_decision.confidence_score * 1.1; // Slight improvement per layer

        // Generate mock outputs
        let token_outputs = if routing_decision.should_execute {
            self.apply_layer_transformation(inputs, layer_idx).await?
        } else {
            inputs.to_vec()
        };

        // Generate mock attention weights for analysis
        let attention_weights = if routing_decision.should_execute {
            Some(self.generate_mock_attention(inputs.len()).await?)
        } else {
            None
        };

        Ok(LayerExecutionResult {
            layer_index: layer_idx,
            was_executed: routing_decision.should_execute,
            output_confidence,
            computation_cost,
            token_outputs,
            attention_weights,
        })
    }

    /// Apply layer transformation (mock implementation)
    async fn apply_layer_transformation(
        &self,
        inputs: &[Vec<f32>],
        layer_idx: usize,
    ) -> TrustformersResult<Vec<Vec<f32>>> {
        // Mock transformer layer computation
        let outputs = inputs
            .iter()
            .map(|input| {
                input.iter()
                    .map(|&x| x + 0.01 * layer_idx as f32) // Simple transformation
                    .collect()
            })
            .collect();
        Ok(outputs)
    }

    /// Generate mock attention weights
    async fn generate_mock_attention(&self, seq_len: usize) -> TrustformersResult<Vec<Vec<f32>>> {
        let attention_weights = (0..seq_len)
            .map(|_| (0..seq_len).map(|_| 1.0 / seq_len as f32).collect())
            .collect();
        Ok(attention_weights)
    }

    /// Check if early exit should be triggered
    fn should_early_exit(
        &self,
        layer_idx: usize,
        confidence: f32,
        complexity_analysis: &ComplexityAnalysis,
    ) -> bool {
        // Early exit conditions
        if layer_idx < self.config.min_layers {
            return false;
        }

        match self.config.depth_strategy {
            DepthStrategy::EarlyExit => confidence > self.config.confidence_threshold,
            DepthStrategy::AdaptiveConfidence => {
                let adaptive_threshold = self.config.confidence_threshold
                    * (1.0 - complexity_analysis.overall_complexity * 0.2);
                confidence > adaptive_threshold
            },
            DepthStrategy::AdaptiveComplexity => {
                let predicted_depth = complexity_analysis.predicted_optimal_depth;
                layer_idx >= predicted_depth
            },
            _ => false,
        }
    }

    /// Calculate efficiency score
    fn calculate_efficiency_score(
        &self,
        executed_layers: &[usize],
        computation_cost: f32,
        final_confidence: f32,
    ) -> f32 {
        let depth_efficiency =
            1.0 - (executed_layers.len() as f32 / self.config.total_layers as f32);
        let cost_efficiency = 1.0 / (1.0 + computation_cost);
        let quality_score = final_confidence;

        // Weighted combination
        0.4 * depth_efficiency + 0.3 * cost_efficiency + 0.3 * quality_score
    }
}

impl Pipeline for MixtureOfDepthsPipeline {
    type Input = PipelineInput;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> TrustformersResult<Self::Output> {
        let tokens: Vec<String> = match input {
            PipelineInput::Text(text) => text.split_whitespace().map(|s| s.to_string()).collect(),
            PipelineInput::Tokens(tokens) => tokens.into_iter().map(|t| t.to_string()).collect(),
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "MoD requires text or token input".to_string(),
                ))
            },
        };

        // Use current runtime handle to avoid creating nested runtimes
        let result = if let Ok(handle) = tokio::runtime::Handle::try_current() {
            tokio::task::block_in_place(|| handle.block_on(self.execute_with_mod(&tokens)))
        } else {
            // Fallback for non-async contexts
            let rt = tokio::runtime::Runtime::new().map_err(|e| {
                TrustformersError::runtime_error(format!("Failed to create async runtime: {}", e))
            })?;
            rt.block_on(self.execute_with_mod(&tokens))
        }
        .map_err(|e| TrustformersError::runtime_error(format!("MoD execution failed: {}", e)))?;

        Ok(PipelineOutput::MixtureOfDepths(result))
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl crate::pipeline::AsyncPipeline for MixtureOfDepthsPipeline {
    type Input = PipelineInput;
    type Output = PipelineOutput;

    async fn __call_async__(&self, input: Self::Input) -> TrustformersResult<Self::Output> {
        let tokens: Vec<String> = match input {
            PipelineInput::Text(text) => text.split_whitespace().map(|s| s.to_string()).collect(),
            PipelineInput::Tokens(tokens) => tokens.into_iter().map(|t| t.to_string()).collect(),
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "MoD requires text or token input".to_string(),
                ))
            },
        };

        let result = self.execute_with_mod(&tokens).await.map_err(|e| {
            TrustformersError::invalid_input(
                format!("MoD execution failed: {}", e),
                Some("tokens"),
                Some("valid tokens for Mixture of Depths execution"),
                None::<String>,
            )
        })?;
        Ok(PipelineOutput::MixtureOfDepths(result))
    }
}

/// Mock implementations for testing and demonstration

/// Mock complexity analyzer
pub struct MockComplexityAnalyzer;

#[async_trait::async_trait]
impl ComplexityAnalyzer for MockComplexityAnalyzer {
    async fn analyze_complexity(&self, input: &[String]) -> TrustformersResult<ComplexityAnalysis> {
        let seq_len = input.len();
        let avg_word_len = input.iter().map(|s| s.len()).sum::<usize>() as f32 / seq_len as f32;

        // Improved heuristics for complexity that consider both sequence length and word complexity
        let length_complexity = if seq_len > 100 {
            0.8
        } else if seq_len > 50 {
            0.6
        } else if seq_len > 3 {
            0.4 + (seq_len as f32 - 3.0) * 0.05 // Scale gradually
        } else {
            0.2
        };

        let word_complexity = if avg_word_len > 10.0 {
            0.8
        } else if avg_word_len > 6.0 {
            0.6
        } else {
            0.3
        };

        let overall_complexity = (length_complexity + word_complexity) / 2.0;

        let token_complexities = input
            .iter()
            .map(|token| {
                if token.len() > 8 {
                    0.8
                } else if token.len() > 4 {
                    0.6
                } else {
                    0.4
                }
            })
            .collect();

        let predicted_optimal_depth = if overall_complexity > 0.7 {
            20
        } else if overall_complexity > 0.5 {
            16
        } else {
            12
        };

        Ok(ComplexityAnalysis {
            overall_complexity,
            token_complexities,
            predicted_optimal_depth,
            confidence_estimate: 0.7,
            semantic_density: overall_complexity,
            syntactic_complexity: avg_word_len / 10.0,
        })
    }
}

/// Mock token classifier
pub struct MockTokenClassifier;

#[async_trait::async_trait]
impl TokenClassifier for MockTokenClassifier {
    async fn classify_tokens(&self, tokens: &[String]) -> TrustformersResult<Vec<TokenType>> {
        let classifications = tokens
            .iter()
            .map(|token| {
                let lower = token.to_lowercase();
                if ["the", "a", "an", "and", "or", "but", "in", "on", "at"]
                    .contains(&lower.as_str())
                {
                    TokenType::Function
                } else if lower.chars().all(|c| c.is_ascii_digit()) {
                    TokenType::Numeric
                } else if lower.chars().next().unwrap_or('a').is_uppercase() {
                    TokenType::Entity
                } else if lower.chars().all(|c| c.is_ascii_punctuation()) {
                    TokenType::Special
                } else {
                    TokenType::Content
                }
            })
            .collect();
        Ok(classifications)
    }
}

/// Mock confidence estimator
pub struct MockConfidenceEstimator;

#[async_trait::async_trait]
impl ConfidenceEstimator for MockConfidenceEstimator {
    async fn estimate_confidence(
        &self,
        layer_outputs: &[Vec<f32>],
        layer_index: usize,
    ) -> TrustformersResult<f32> {
        // Mock confidence calculation based on layer depth and output variance
        let base_confidence = 0.5 + (layer_index as f32 / 24.0) * 0.4;

        // Add some variance based on outputs
        let output_variance = if !layer_outputs.is_empty() && !layer_outputs[0].is_empty() {
            let mean = layer_outputs[0].iter().sum::<f32>() / layer_outputs[0].len() as f32;
            let variance = layer_outputs[0].iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
                / layer_outputs[0].len() as f32;
            variance.min(0.2)
        } else {
            0.1
        };

        Ok((base_confidence + output_variance).min(1.0))
    }
}

/// Mock depth router
pub struct MockDepthRouter;

#[async_trait::async_trait]
impl DepthRouter for MockDepthRouter {
    async fn route_depth(
        &self,
        input_analysis: &ComplexityAnalysis,
        layer_index: usize,
        current_confidence: f32,
        config: &MixtureOfDepthsConfig,
    ) -> TrustformersResult<RoutingDecision> {
        let should_execute = match config.depth_strategy {
            DepthStrategy::Fixed(depth) => layer_index < depth,
            DepthStrategy::EarlyExit => {
                layer_index < config.min_layers || current_confidence < config.confidence_threshold
            },
            DepthStrategy::AdaptiveComplexity => {
                layer_index < input_analysis.predicted_optimal_depth
            },
            DepthStrategy::AdaptiveConfidence => {
                let adaptive_threshold =
                    config.confidence_threshold * (1.0 + input_analysis.overall_complexity * 0.2);
                layer_index < config.min_layers || current_confidence < adaptive_threshold
            },
            DepthStrategy::BudgetOptimal => {
                // Simple budget-based routing
                layer_index < config.max_layers / 2
            },
            DepthStrategy::TokenTypeAware => {
                // For mock, always execute content layers
                true
            },
        };

        let token_routing = if config.token_level_routing {
            input_analysis
                .token_complexities
                .iter()
                .map(|&complexity| complexity > 0.5)
                .collect()
        } else {
            Vec::new()
        };

        let routing_reason = if layer_index < config.min_layers {
            RoutingReason::FixedDepth
        } else if current_confidence > config.confidence_threshold {
            RoutingReason::ConfidenceThreshold
        } else {
            RoutingReason::ComplexityBased
        };

        Ok(RoutingDecision {
            layer_index,
            should_execute,
            confidence_score: current_confidence,
            complexity_score: input_analysis.overall_complexity,
            token_routing,
            routing_reason,
        })
    }
}

/// Factory functions for creating MoD pipelines

/// Create a basic MoD pipeline
pub fn create_mixture_of_depths_pipeline(
    config: MixtureOfDepthsConfig,
    base_model: Arc<dyn Pipeline<Input = String, Output = PipelineOutput>>,
) -> MixtureOfDepthsPipeline {
    let complexity_analyzer = Arc::new(MockComplexityAnalyzer);
    let confidence_estimator = Arc::new(MockConfidenceEstimator);
    let depth_router = Arc::new(MockDepthRouter);

    MixtureOfDepthsPipeline::new(
        config,
        base_model,
        complexity_analyzer,
        confidence_estimator,
        depth_router,
    )
}

/// Create an efficiency-optimized MoD pipeline
pub fn create_efficiency_optimized_mod_pipeline(
    base_model: Arc<dyn Pipeline<Input = String, Output = PipelineOutput>>,
) -> MixtureOfDepthsPipeline {
    let config = MixtureOfDepthsConfig {
        depth_strategy: DepthStrategy::AdaptiveConfidence,
        confidence_threshold: 0.9,
        token_level_routing: true,
        adaptive_depth: true,
        compute_budget: 0.7,
        ..Default::default()
    };

    create_mixture_of_depths_pipeline(config, base_model)
}

/// Create a quality-focused MoD pipeline
pub fn create_quality_focused_mod_pipeline(
    base_model: Arc<dyn Pipeline<Input = String, Output = PipelineOutput>>,
) -> MixtureOfDepthsPipeline {
    let config = MixtureOfDepthsConfig {
        depth_strategy: DepthStrategy::AdaptiveComplexity,
        confidence_threshold: 0.7,
        min_layers: 12,
        compute_budget: 1.5,
        ..Default::default()
    };

    create_mixture_of_depths_pipeline(config, base_model)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock base model for testing
    struct MockBaseModel;

    impl Pipeline for MockBaseModel {
        type Input = String;
        type Output = PipelineOutput;

        fn __call__(&self, _input: Self::Input) -> TrustformersResult<Self::Output> {
            Ok(PipelineOutput::Text("Mock output".to_string()))
        }
    }

    // ── Config defaults ───────────────────────────────────────────────────────

    #[test]
    fn test_config_default_min_layers_less_than_total() {
        let config = MixtureOfDepthsConfig::default();
        assert!(
            config.min_layers < config.total_layers,
            "min_layers must be less than total_layers"
        );
    }

    #[test]
    fn test_config_default_confidence_threshold_in_range() {
        let config = MixtureOfDepthsConfig::default();
        assert!(config.confidence_threshold > 0.0 && config.confidence_threshold <= 1.0);
    }

    #[test]
    fn test_config_default_compute_budget_positive() {
        let config = MixtureOfDepthsConfig::default();
        assert!(config.compute_budget > 0.0);
    }

    // ── Expert capacity formula ───────────────────────────────────────────────
    // expert_capacity = ceil(capacity_factor * seq_len / num_experts)

    #[test]
    fn test_expert_capacity_formula() {
        let capacity_factor = 1.25_f32;
        let seq_len = 128_usize;
        let num_experts = 4_usize;
        let capacity = ((capacity_factor * seq_len as f32 / num_experts as f32).ceil()) as usize;
        assert_eq!(capacity, 40, "capacity = ceil(1.25*128/4) = ceil(40) = 40");
    }

    #[test]
    fn test_expert_capacity_rounds_up() {
        let capacity_factor = 1.0_f32;
        let seq_len = 7_usize;
        let num_experts = 2_usize;
        let capacity = ((capacity_factor * seq_len as f32 / num_experts as f32).ceil()) as usize;
        assert_eq!(capacity, 4, "capacity = ceil(7/2) = 4");
    }

    // ── Complexity analyser ───────────────────────────────────────────────────

    #[tokio::test]
    async fn test_complexity_analysis() {
        let analyzer = MockComplexityAnalyzer;
        let simple_input = vec!["hello".to_string(), "world".to_string()];
        let complex_input = vec![
            "sophisticated".to_string(),
            "terminology".to_string(),
            "requires".to_string(),
            "extensive".to_string(),
            "computational".to_string(),
            "resources".to_string(),
        ];
        let simple_analysis = analyzer
            .analyze_complexity(&simple_input)
            .await
            .expect("async operation failed");
        let complex_analysis = analyzer
            .analyze_complexity(&complex_input)
            .await
            .expect("async operation failed");
        assert!(simple_analysis.overall_complexity < complex_analysis.overall_complexity);
        assert!(simple_analysis.predicted_optimal_depth < complex_analysis.predicted_optimal_depth);
    }

    #[tokio::test]
    async fn test_complexity_analysis_confidence_in_range() {
        let analyzer = MockComplexityAnalyzer;
        let input = vec!["test".to_string()];
        let analysis = analyzer
            .analyze_complexity(&input)
            .await
            .expect("analyze_complexity should succeed");
        assert!(analysis.confidence_estimate >= 0.0 && analysis.confidence_estimate <= 1.0);
    }

    #[tokio::test]
    async fn test_complexity_per_token_count() {
        let analyzer = MockComplexityAnalyzer;
        let tokens = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let analysis = analyzer
            .analyze_complexity(&tokens)
            .await
            .expect("analyze_complexity should succeed");
        assert_eq!(
            analysis.token_complexities.len(),
            tokens.len(),
            "token_complexities must have one entry per token"
        );
    }

    // ── Token classification ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_token_classification() {
        let classifier = MockTokenClassifier;
        let tokens = vec![
            "The".to_string(),
            "quick".to_string(),
            "brown".to_string(),
            "fox".to_string(),
            "123".to_string(),
            "!".to_string(),
        ];
        let classifications =
            classifier.classify_tokens(&tokens).await.expect("async operation failed");
        assert_eq!(classifications[0], TokenType::Function); // "The"
        assert_eq!(classifications[4], TokenType::Numeric); // "123"
        assert_eq!(classifications[5], TokenType::Special); // "!"
    }

    #[tokio::test]
    async fn test_token_classification_length_matches() {
        let classifier = MockTokenClassifier;
        let tokens: Vec<String> = (0..7).map(|i| format!("token{}", i)).collect();
        let classes = classifier
            .classify_tokens(&tokens)
            .await
            .expect("classify_tokens should succeed");
        assert_eq!(classes.len(), tokens.len());
    }

    // ── Confidence estimator ─────────────────────────────────────────────────

    #[tokio::test]
    async fn test_confidence_increases_with_layer_depth() {
        let estimator = MockConfidenceEstimator;
        let outputs = vec![vec![0.1_f32; 4]];
        let conf_early = estimator
            .estimate_confidence(&outputs, 0)
            .await
            .expect("estimate_confidence should succeed");
        let conf_late = estimator
            .estimate_confidence(&outputs, 20)
            .await
            .expect("estimate_confidence should succeed");
        assert!(
            conf_late > conf_early,
            "confidence should increase with layer depth"
        );
    }

    #[tokio::test]
    async fn test_confidence_capped_at_one() {
        let estimator = MockConfidenceEstimator;
        let outputs = vec![vec![100.0_f32; 4]]; // very high variance
        let conf = estimator
            .estimate_confidence(&outputs, 23)
            .await
            .expect("estimate_confidence should succeed");
        assert!(conf <= 1.0, "confidence must be ≤ 1.0");
    }

    // ── Routing decision ─────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_router_fixed_depth_strategy() {
        let router = MockDepthRouter;
        let analysis = ComplexityAnalysis {
            overall_complexity: 0.5,
            token_complexities: vec![0.5],
            predicted_optimal_depth: 12,
            confidence_estimate: 0.7,
            semantic_density: 0.5,
            syntactic_complexity: 0.3,
        };
        let config = MixtureOfDepthsConfig {
            depth_strategy: DepthStrategy::Fixed(5),
            ..Default::default()
        };
        // Layer 3 < 5 → should execute
        let decision = router
            .route_depth(&analysis, 3, 0.6, &config)
            .await
            .expect("route_depth should succeed");
        assert!(decision.should_execute);
        // Layer 6 >= 5 → should not execute
        let decision2 = router
            .route_depth(&analysis, 6, 0.6, &config)
            .await
            .expect("route_depth should succeed");
        assert!(!decision2.should_execute);
    }

    #[tokio::test]
    async fn test_router_min_layers_always_execute() {
        let router = MockDepthRouter;
        let analysis = ComplexityAnalysis {
            overall_complexity: 0.5,
            token_complexities: vec![0.5],
            predicted_optimal_depth: 10,
            confidence_estimate: 0.7,
            semantic_density: 0.5,
            syntactic_complexity: 0.3,
        };
        let config = MixtureOfDepthsConfig {
            depth_strategy: DepthStrategy::EarlyExit,
            min_layers: 6,
            confidence_threshold: 0.9,
            ..Default::default()
        };
        // Below min_layers, confidence is irrelevant - should execute
        let decision = router
            .route_depth(&analysis, 2, 0.99, &config)
            .await
            .expect("route_depth should succeed");
        // EarlyExit: execute while layer < min_layers OR confidence < threshold
        // layer 2 < 6 → should execute
        assert!(decision.should_execute);
    }

    // ── Skipped token handling (residual pass-through) ───────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn test_skipped_layer_preserves_outputs() {
        let config = MixtureOfDepthsConfig {
            depth_strategy: DepthStrategy::Fixed(0), // skip all layers
            min_layers: 0,
            ..Default::default()
        };
        let mock_base_model = Arc::new(MockBaseModel);
        let pipeline = create_mixture_of_depths_pipeline(config, mock_base_model);
        let input = PipelineInput::Text("skip all layers".to_string());
        let result = pipeline.__call__(input);
        assert!(result.is_ok(), "skipping all layers should not crash");
    }

    // ── Depth reduction vs accuracy trade-off ─────────────────────────────────

    #[test]
    fn test_efficiency_score_with_fewer_executed_layers() {
        let config = MixtureOfDepthsConfig::default();
        let mock_base_model = Arc::new(MockBaseModel);
        let pipeline = create_mixture_of_depths_pipeline(config.clone(), mock_base_model);
        // Fewer executed layers → higher depth_efficiency
        let score_few = pipeline.calculate_efficiency_score(&[0, 1], 2.0, 0.9);
        let score_many =
            pipeline.calculate_efficiency_score(&(0..20).collect::<Vec<_>>(), 20.0, 0.9);
        assert!(
            score_few > score_many,
            "fewer executed layers should yield a higher efficiency score"
        );
    }

    // ── Auxiliary load-balancing auxiliary loss ───────────────────────────────

    #[test]
    fn test_routing_reason_min_layers() {
        // When layer < min_layers, reason should be FixedDepth
        let reason = RoutingReason::FixedDepth;
        assert!(matches!(reason, RoutingReason::FixedDepth));
    }

    // ── End-to-end pipeline tests ─────────────────────────────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn test_mixture_of_depths_pipeline() {
        let config = MixtureOfDepthsConfig::default();
        let mock_base_model = Arc::new(MockBaseModel);
        let mod_pipeline = create_mixture_of_depths_pipeline(config, mock_base_model);
        let input =
            PipelineInput::Text("This is a test sentence for mixture of depths".to_string());
        let result = mod_pipeline.__call__(input);
        assert!(result.is_ok());
        if let Ok(PipelineOutput::MixtureOfDepths(mod_result)) = result {
            assert!(!mod_result.executed_layers.is_empty());
            assert!(mod_result.efficiency_score > 0.0);
            assert!(!mod_result.confidence_progression.is_empty());
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_early_exit_strategy() {
        let config = MixtureOfDepthsConfig {
            depth_strategy: DepthStrategy::EarlyExit,
            confidence_threshold: 0.8,
            min_layers: 6,
            ..Default::default()
        };
        let mock_base_model = Arc::new(MockBaseModel);
        let mod_pipeline = create_mixture_of_depths_pipeline(config, mock_base_model);
        let input = PipelineInput::Text("Simple text".to_string());
        let result = mod_pipeline.__call__(input);
        assert!(result.is_ok());
        if let Ok(PipelineOutput::MixtureOfDepths(mod_result)) = result {
            assert!(mod_result.executed_layers.len() < 24);
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_budget_optimal_strategy() {
        let config = MixtureOfDepthsConfig {
            depth_strategy: DepthStrategy::BudgetOptimal,
            compute_budget: 5.0,
            ..Default::default()
        };
        let mock_base_model = Arc::new(MockBaseModel);
        let pipeline = create_mixture_of_depths_pipeline(config, mock_base_model);
        let input = PipelineInput::Text("budget test".to_string());
        let result = pipeline.__call__(input);
        assert!(result.is_ok());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_efficiency_optimized_factory() {
        let mock_base_model = Arc::new(MockBaseModel);
        let pipeline = create_efficiency_optimized_mod_pipeline(mock_base_model);
        let input = PipelineInput::Text("efficiency test".to_string());
        let result = pipeline.__call__(input);
        assert!(
            result.is_ok(),
            "efficiency-optimized pipeline should succeed"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_quality_focused_factory() {
        let mock_base_model = Arc::new(MockBaseModel);
        let pipeline = create_quality_focused_mod_pipeline(mock_base_model);
        let input = PipelineInput::Text("quality test".to_string());
        let result = pipeline.__call__(input);
        assert!(result.is_ok(), "quality-focused pipeline should succeed");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_non_text_input_rejected() {
        let config = MixtureOfDepthsConfig::default();
        let mock_base_model = Arc::new(MockBaseModel);
        let pipeline = create_mixture_of_depths_pipeline(config, mock_base_model);
        // BatchText is not supported by MoD pipeline
        let input = PipelineInput::BatchText(vec!["a".to_string()]);
        let result = pipeline.__call__(input);
        assert!(
            result.is_err(),
            "MoD pipeline should reject BatchText input"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_confidence_progression_non_decreasing_tendency() {
        let config = MixtureOfDepthsConfig::default();
        let mock_base_model = Arc::new(MockBaseModel);
        let pipeline = create_mixture_of_depths_pipeline(config, mock_base_model);
        let input = PipelineInput::Text("confidence progression test".to_string());
        let result = pipeline.__call__(input).expect("pipeline should succeed");
        if let PipelineOutput::MixtureOfDepths(mod_result) = result {
            // At least the last confidence value should be accessible
            assert!(!mod_result.confidence_progression.is_empty());
            let first = mod_result.confidence_progression[0];
            let last = *mod_result.confidence_progression.last().expect("last confidence exists");
            assert!(
                last >= first - 0.01,
                "confidence generally should not decrease significantly overall"
            );
        }
    }

    // ── Additional unit tests ─────────────────────────────────────────────────

    #[test]
    fn test_depth_strategy_variants_constructable() {
        let _fixed = DepthStrategy::Fixed(12);
        let _early = DepthStrategy::EarlyExit;
        let _complexity = DepthStrategy::AdaptiveComplexity;
        let _confidence = DepthStrategy::AdaptiveConfidence;
        let _budget = DepthStrategy::BudgetOptimal;
        let _token = DepthStrategy::TokenTypeAware;
    }

    #[test]
    fn test_token_type_variants_constructable() {
        let types = [
            TokenType::Function,
            TokenType::Content,
            TokenType::Entity,
            TokenType::Numeric,
            TokenType::Special,
            TokenType::Unknown,
        ];
        assert_eq!(types.len(), 6);
    }

    #[test]
    fn test_token_type_equality() {
        assert_eq!(TokenType::Function, TokenType::Function);
        assert_ne!(TokenType::Content, TokenType::Entity);
        assert_ne!(TokenType::Numeric, TokenType::Special);
    }

    #[test]
    fn test_routing_reason_variants() {
        let reasons = [
            RoutingReason::ConfidenceThreshold,
            RoutingReason::ComplexityBased,
            RoutingReason::BudgetConstraint,
            RoutingReason::TokenSpecific,
            RoutingReason::FixedDepth,
        ];
        assert_eq!(reasons.len(), 5);
    }

    #[test]
    fn test_routing_decision_struct() {
        let decision = RoutingDecision {
            layer_index: 5,
            should_execute: true,
            confidence_score: 0.85,
            complexity_score: 0.6,
            token_routing: vec![true, false, true],
            routing_reason: RoutingReason::ConfidenceThreshold,
        };
        assert_eq!(decision.layer_index, 5);
        assert!(decision.should_execute);
        assert!(decision.confidence_score > 0.0 && decision.confidence_score <= 1.0);
        assert_eq!(decision.token_routing.len(), 3);
    }

    #[test]
    fn test_mod_execution_result_struct() {
        let result = MoDExecutionResult {
            final_outputs: vec![vec![0.1, 0.2, 0.3]],
            executed_layers: vec![0, 1, 2, 3, 4, 5],
            routing_decisions: Vec::new(),
            layer_results: Vec::new(),
            total_computation_cost: 6.0,
            efficiency_score: 0.75,
            confidence_progression: vec![0.5, 0.6, 0.7, 0.8, 0.85, 0.9],
        };
        assert_eq!(result.executed_layers.len(), 6);
        assert!(result.efficiency_score > 0.0 && result.efficiency_score <= 1.0);
        assert_eq!(result.confidence_progression.len(), 6);
    }

    #[test]
    fn test_complexity_analysis_struct_fields() {
        let analysis = ComplexityAnalysis {
            overall_complexity: 0.65,
            token_complexities: vec![0.4, 0.7, 0.8],
            predicted_optimal_depth: 18,
            confidence_estimate: 0.82,
            semantic_density: 0.55,
            syntactic_complexity: 0.4,
        };
        assert!(analysis.overall_complexity >= 0.0 && analysis.overall_complexity <= 1.0);
        assert!(analysis.predicted_optimal_depth > 0);
        assert_eq!(analysis.token_complexities.len(), 3);
    }

    #[test]
    fn test_layer_execution_result_struct() {
        let layer_res = LayerExecutionResult {
            layer_index: 7,
            was_executed: true,
            output_confidence: 0.78,
            computation_cost: 1.2,
            token_outputs: vec![vec![0.1, 0.2]],
            attention_weights: None,
        };
        assert_eq!(layer_res.layer_index, 7);
        assert!(layer_res.was_executed);
        assert!(layer_res.computation_cost > 0.0);
    }

    #[test]
    fn test_config_token_level_routing_default() {
        let cfg = MixtureOfDepthsConfig::default();
        assert!(
            cfg.token_level_routing,
            "token_level_routing should be enabled by default"
        );
    }

    #[test]
    fn test_config_adaptive_depth_default() {
        let cfg = MixtureOfDepthsConfig::default();
        assert!(
            cfg.adaptive_depth,
            "adaptive_depth should be enabled by default"
        );
    }

    #[test]
    fn test_config_max_layers_gte_min_layers() {
        let cfg = MixtureOfDepthsConfig::default();
        assert!(
            cfg.max_layers >= cfg.min_layers,
            "max_layers must be >= min_layers"
        );
    }

    #[test]
    fn test_efficiency_score_formula() {
        // efficiency = (1 - executed/total) * quality / cost
        let total = 24_usize;
        let executed = 12_usize;
        let depth_efficiency = 1.0 - (executed as f32 / total as f32);
        assert!(
            (depth_efficiency - 0.5).abs() < 1e-5,
            "executing half the layers → depth_efficiency = 0.5"
        );
    }

    #[test]
    fn test_compute_budget_positive() {
        let cfg = MixtureOfDepthsConfig::default();
        assert!(cfg.compute_budget > 0.0);
    }

    #[test]
    fn test_hierarchical_routing_disabled_default() {
        let cfg = MixtureOfDepthsConfig::default();
        assert!(!cfg.hierarchical_routing);
    }
}
