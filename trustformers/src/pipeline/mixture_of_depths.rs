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
    layer_cache: Arc<RwLock<LayerCache>>,
    /// Maximum number of entries `layer_cache` may hold; see
    /// [`DEFAULT_LAYER_CACHE_CAPACITY`] and
    /// [`Self::with_layer_cache_capacity`].
    layer_cache_capacity: usize,
    /// Produces the real token embeddings the router reasons about.
    embedder: Option<Arc<dyn TokenEmbedder>>,
    /// Executes a real transformer layer.
    layer_executor: Option<Arc<dyn TransformerLayerExecutor>>,
}

/// Produces per-token embeddings for the routing decision.
///
/// Routing on constant vectors carries no information, so the pipeline requires
/// a real embedder rather than substituting one.
pub trait TokenEmbedder: Send + Sync {
    /// Embed each input token/segment.
    ///
    /// # Errors
    ///
    /// Whatever the underlying model's embedding step can fail with.
    fn embed(&self, inputs: &[String]) -> TrustformersResult<Vec<Vec<f32>>>;

    /// Dimensionality of the produced embeddings.
    fn embedding_dim(&self) -> usize;
}

/// Executes one transformer layer over a batch of token vectors.
pub trait TransformerLayerExecutor: Send + Sync {
    /// Number of layers available.
    fn num_layers(&self) -> usize;

    /// Run `layer_index` over `inputs`.
    ///
    /// `token_mask`, when present, marks which tokens the router selected for
    /// this layer; unselected tokens must be passed through unchanged.
    ///
    /// # Errors
    ///
    /// Whatever the underlying layer computation can fail with.
    fn execute(
        &self,
        layer_index: usize,
        inputs: &[Vec<f32>],
        token_mask: Option<&[bool]>,
    ) -> TrustformersResult<Vec<Vec<f32>>>;
}

/// Default maximum number of entries [`MixtureOfDepthsPipeline`]'s
/// layer-execution cache may hold before the least-recently-used entry is
/// evicted. `MixtureOfDepthsConfig` carries no cache-size field of its own
/// (it describes routing depth, not caching), so this is a separate,
/// implementation-level default, overridable via
/// [`MixtureOfDepthsPipeline::with_layer_cache_capacity`].
const DEFAULT_LAYER_CACHE_CAPACITY: usize = 128;

/// A capacity-bounded, least-recently-used cache of per-layer execution
/// results, keyed on `(layer_index, content_hash_of_inputs_and_mask)`.
///
/// Bounded so a long-running pipeline processing many distinct inputs cannot
/// grow this cache without limit: once `capacity` entries are held, the
/// least-recently-used entry is evicted before a new one is inserted.
#[derive(Debug, Default)]
struct LayerCache {
    entries: HashMap<(usize, u64), LayerExecutionResult>,
    /// Recency order, oldest (least-recently-used) first. Each key appears
    /// at most once; `touch` moves a key to the back (most-recently-used).
    order: std::collections::VecDeque<(usize, u64)>,
}

impl LayerCache {
    /// Number of entries currently held.
    fn len(&self) -> usize {
        self.entries.len()
    }

    /// Look up `key`, marking it most-recently-used on a hit.
    fn get(&mut self, key: (usize, u64)) -> Option<LayerExecutionResult> {
        let hit = self.entries.get(&key).cloned();
        if hit.is_some() {
            self.touch(key);
        }
        hit
    }

    /// Move `key` to the most-recently-used end of the recency order.
    fn touch(&mut self, key: (usize, u64)) {
        if let Some(pos) = self.order.iter().position(|k| *k == key) {
            self.order.remove(pos);
        }
        self.order.push_back(key);
    }

    /// Insert (or update) `value` under `key`, evicting the
    /// least-recently-used entry if `capacity` would otherwise be exceeded.
    ///
    /// A `capacity` of `0` disables caching: nothing is stored, so every
    /// lookup misses and every call recomputes.
    fn insert(&mut self, key: (usize, u64), value: LayerExecutionResult, capacity: usize) {
        if capacity == 0 {
            return;
        }
        // A plain upsert handles both the new-key and existing-key cases
        // uniformly (`HashMap::insert` replaces in place for an existing
        // key without changing its length); `touch` likewise both
        // repositions an existing `order` entry and appends a brand new
        // one, so no separate existing/new branch is needed here.
        self.entries.insert(key, value);
        self.touch(key);
        while self.entries.len() > capacity {
            // `touch` above just moved `key` to the most-recently-used
            // (back) end, so the front of `order` is never the entry we
            // just inserted or updated.
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            self.entries.remove(&oldest);
        }
    }
}

/// Content hash of a layer's inputs and routing mask, combined with the
/// layer index, for [`LayerCache`]'s key.
///
/// The mask is part of the key, not just the input tensor: it changes what
/// the executor actually does to those inputs (a masked-out token passes
/// through unchanged; a selected one does not), so a mask-blind key would
/// silently serve a stale result computed under a different mask. Assuming a
/// deterministic [`TransformerLayerExecutor`], two calls that hash equal are
/// guaranteed to produce the same output.
fn layer_cache_key(
    layer_index: usize,
    inputs: &[Vec<f32>],
    token_mask: Option<&[bool]>,
) -> (usize, u64) {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    inputs.len().hash(&mut hasher);
    for row in inputs {
        row.len().hash(&mut hasher);
        for value in row {
            value.to_bits().hash(&mut hasher);
        }
    }
    match token_mask {
        Some(mask) => {
            1u8.hash(&mut hasher);
            mask.hash(&mut hasher);
        },
        None => 0u8.hash(&mut hasher),
    }
    (layer_index, hasher.finish())
}

/// Mean absolute activation of `outputs`, squashed into 0..=1.
///
/// A measured summary of how strongly the layer responded — used as the
/// per-layer confidence signal instead of multiplying the previous value by a
/// fixed factor.
fn mean_activation_confidence(outputs: &[Vec<f32>]) -> f32 {
    let mut sum = 0.0f32;
    let mut count = 0usize;
    for row in outputs {
        for value in row {
            sum += value.abs();
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }
    (sum / count as f32).tanh()
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
            embedder: None,
            layer_executor: None,
            layer_cache: Arc::new(RwLock::new(LayerCache::default())),
            layer_cache_capacity: DEFAULT_LAYER_CACHE_CAPACITY,
        }
    }

    /// Set token classifier for hierarchical routing
    pub fn with_token_classifier(mut self, classifier: Arc<dyn TokenClassifier>) -> Self {
        self.token_classifier = Some(classifier);
        self
    }

    /// Override the layer-execution cache's maximum entry count (default
    /// `DEFAULT_LAYER_CACHE_CAPACITY`).
    ///
    /// A capacity of `0` disables caching entirely: every call to
    /// `execute_layer` recomputes.
    pub fn with_layer_cache_capacity(mut self, capacity: usize) -> Self {
        self.layer_cache_capacity = capacity;
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

    /// Attach the component that produces real token embeddings.
    pub fn with_embedder(mut self, embedder: Arc<dyn TokenEmbedder>) -> Self {
        self.embedder = Some(embedder);
        self
    }

    /// Attach the component that executes real transformer layers.
    pub fn with_layer_executor(mut self, executor: Arc<dyn TransformerLayerExecutor>) -> Self {
        self.layer_executor = Some(executor);
        self
    }

    /// The base pipeline this `MixtureOfDepthsPipeline` was constructed with.
    ///
    /// `execute_with_mod` performs its own routing/embedding/layer-execution
    /// via `embedder`/`layer_executor`/`depth_router` rather than delegating
    /// to it, so this is exposed for callers that want a text-generating
    /// fallback (or a baseline to compare MoD routing against) without
    /// having to keep a second `Arc` clone of their own around.
    pub fn base_model(&self) -> &Arc<dyn Pipeline<Input = String, Output = PipelineOutput>> {
        &self.base_model
    }

    /// Number of entries currently stored in the layer-execution cache.
    ///
    /// `execute_layer` checks this cache before running the attached
    /// [`TransformerLayerExecutor`] and inserts into it after a real
    /// execution, keyed on `(layer_index, content_hash(inputs,
    /// routing_mask))` (see `layer_cache_key`). Entries beyond
    /// [`Self::with_layer_cache_capacity`]'s bound (default
    /// `DEFAULT_LAYER_CACHE_CAPACITY`) are evicted least-recently-used
    /// first.
    pub async fn cached_layer_count(&self) -> usize {
        self.layer_cache.read().await.len()
    }

    /// Embed the input with the attached embedder.
    ///
    /// # Errors
    ///
    /// Fails when no embedder is attached: routing over constant vectors would
    /// make every decision meaningless while still reporting efficiency gains.
    async fn initialize_embeddings(&self, input: &[String]) -> TrustformersResult<Vec<Vec<f32>>> {
        let embedder = self.embedder.as_ref().ok_or_else(|| {
            TrustformersError::feature_unavailable(
                "no token embedder is attached to the Mixture-of-Depths pipeline; routing \
                 decisions need real embeddings. Attach one with \
                 `MixtureOfDepthsPipeline::with_embedder`."
                    .to_string(),
                "mixture-of-depths",
            )
        })?;
        embedder.embed(input)
    }

    /// Execute a single layer with optional token-level routing.
    ///
    /// The transformation is performed by the attached
    /// [`TransformerLayerExecutor`]; on a cache miss, the reported computation
    /// cost is the real wall-clock time the layer took (see below for the
    /// cache-hit case), and the token routing mask is passed through so
    /// unselected tokens are genuinely skipped.
    ///
    /// Memoised on `(layer_idx, content_hash(inputs, token_mask))` via
    /// [`layer_cache_key`]: a cache hit returns the earlier, real
    /// `token_outputs`/`output_confidence` without calling the executor
    /// again, but with `computation_cost` reset to `0.0` -- the wall-clock
    /// cost recorded on the original (miss) call was paid then, not on this
    /// replay, and reporting it again would double-count it into
    /// `execute_with_mod`'s `total_computation_cost` (its compute-budget
    /// cutoff and efficiency score) every time the same layer is replayed. A
    /// miss computes the result for real and stores it, subject to
    /// [`Self::with_layer_cache_capacity`]'s bound. A skipped layer
    /// (`!routing_decision.should_execute`) is a pure, zero-cost
    /// pass-through and is not cached.
    ///
    /// # Errors
    ///
    /// Fails when no layer executor is attached, or when the executor rejects
    /// the layer index or input shape.
    async fn execute_layer(
        &self,
        layer_idx: usize,
        inputs: &[Vec<f32>],
        routing_decision: &RoutingDecision,
        _token_types: Option<&[TokenType]>,
    ) -> TrustformersResult<LayerExecutionResult> {
        if !routing_decision.should_execute {
            return Ok(LayerExecutionResult {
                layer_index: layer_idx,
                was_executed: false,
                output_confidence: routing_decision.confidence_score,
                computation_cost: 0.0,
                token_outputs: inputs.to_vec(),
                attention_weights: None,
            });
        }

        let executor = self.layer_executor.as_ref().ok_or_else(|| {
            TrustformersError::feature_unavailable(
                "no transformer layer executor is attached to the Mixture-of-Depths pipeline. \
                 Attach one with `MixtureOfDepthsPipeline::with_layer_executor`."
                    .to_string(),
                "mixture-of-depths",
            )
        })?;

        let token_mask: Option<Vec<bool>> =
            if self.config.token_level_routing && !routing_decision.token_routing.is_empty() {
                Some(routing_decision.token_routing.clone())
            } else {
                None
            };

        let cache_key = layer_cache_key(layer_idx, inputs, token_mask.as_deref());
        if let Some(cached) = self.layer_cache.write().await.get(cache_key) {
            // No computation happened on this call -- the executor was not
            // invoked -- so the honest cost of *this* call is zero, not the
            // wall-clock time measured back when the result was first
            // computed. Returning the stale cost here would let a replayed
            // layer that cost ~0 this time still be charged its original
            // cost in `execute_with_mod`'s running `total_computation_cost`.
            return Ok(LayerExecutionResult {
                computation_cost: 0.0,
                ..cached
            });
        }

        let started = std::time::Instant::now();
        let token_outputs = executor.execute(layer_idx, inputs, token_mask.as_deref())?;
        // Measured, not synthesised from the routing flags.
        let computation_cost = started.elapsed().as_secs_f32() * 1000.0;

        // Confidence after the layer is measured from its own output rather
        // than scaled up by a fixed factor.
        let output_confidence = mean_activation_confidence(&token_outputs);

        let result = LayerExecutionResult {
            layer_index: layer_idx,
            was_executed: true,
            output_confidence,
            computation_cost,
            token_outputs,
            // The executor interface does not surface attention matrices, and a
            // uniform stand-in would be indistinguishable from a real one.
            attention_weights: None,
        };

        self.layer_cache
            .write()
            .await
            .insert(cache_key, result.clone(), self.layer_cache_capacity);

        Ok(result)
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
#[path = "mixture_of_depths_tests.rs"]
mod tests;
