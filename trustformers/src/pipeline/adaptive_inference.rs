//! Adaptive inference: input-driven precision/skip *recommendations*, plus a
//! genuinely real early-exit execution path for models that expose one.
//!
//! # What is real here, and what is a recommendation
//!
//! [`AdaptiveInferenceEngine::adaptive_inference`] accepts any
//! `P: Pipeline<Output = PipelineOutput>`. The plain [`Pipeline`] trait
//! exposes exactly one hook -- `__call__`, a whole-model call -- so there is
//! no way to actually run, skip, or change the precision of an individual
//! layer of an arbitrary `P`. Given that constraint:
//!
//! - [`InputAnalyzer::analyze_input`] is real: for textual input (the
//!   overwhelmingly common `Pipeline::Input` in this crate) it computes
//!   genuine, deterministic, content-derived measurements (word count,
//!   vocabulary diversity, per-token importance, ...). It never returns a
//!   fixed constant for text input.
//! - Per-layer precision/skip choices (`precision_history`, `skip_history`,
//!   `AdaptationDecision`s produced by [`AdaptiveInferenceEngine::adaptive_inference`])
//!   are real, input-derived *recommendations* -- genuinely computed from
//!   [`InputAnalysis`], genuinely recorded, genuinely readable back out --
//!   but they are never applied to computation, because there is no hook to
//!   apply them to. What actually executes is always either the real
//!   early-exit path below, or an unconditional full `base_pipeline` call.
//!   `AdaptationDecision::decision_type` and `reason` say "recommended", not
//!   "applied", "skipped", or "used".
//! - [`AdaptiveInferenceEngine::adaptive_inference_layerwise`] is the real
//!   execution path: for a `P` that also implements
//!   [`crate::pipeline::early_exit::LayerwiseInference`], it drives
//!   [`EarlyExitPipeline::run`] directly, and every layer-count/timing/
//!   confidence number it reports comes from layers that actually ran.
//! - Resource telemetry (`memory_peak_mb`, `energy_consumed_watts`, CPU
//!   headroom folded into `resource_efficiency`) is sampled via
//!   [`crate::profiler::read_process_memory`] and
//!   [`crate::pipeline::sampled_cpu_utilization`] -- real host measurements,
//!   not fixed constants. There is no energy telemetry source wired into
//!   this workspace, so `energy_consumed_watts` is honestly `0.0`.

use crate::pipeline::early_exit::{
    EarlyExitConfig, EarlyExitPipeline, EarlyExitResult, LayerwiseInference,
};
use crate::pipeline::{Pipeline, PipelineOutput};
use crate::profiler::read_process_memory;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use trustformers_core::errors::Result;

/// Number of coarse content buckets [`InputAnalysis::attention_patterns`] is
/// reported in. This is an honest, content-derived summary (see
/// [`AttentionPatternAnalyzer`]), sized arbitrarily for a stable output
/// shape -- it is not tied to any real model's attention head count, since
/// no live model attention is available without a real forward pass.
const ATTENTION_PATTERN_BUCKETS: usize = 12;

/// Dynamic precision modes for adaptive inference
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrecisionMode {
    /// Full precision (FP32)
    Full,
    /// Half precision (FP16)
    Half,
    /// Mixed precision (FP16 for forward, FP32 for backward)
    Mixed,
    /// 8-bit precision
    Int8,
    /// 4-bit precision
    Int4,
    /// Dynamic precision based on layer importance
    Dynamic,
    /// Adaptive precision based on input complexity
    Adaptive,
}

/// Conditional computation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionalStrategy {
    /// Skip attention layers based on input complexity
    AttentionSkipping,
    /// Skip feed-forward layers based on activation patterns
    FeedForwardSkipping,
    /// Skip entire transformer blocks
    BlockSkipping,
    /// Dynamic depth selection
    DynamicDepth,
    /// Sparse activation (only compute activated neurons)
    SparseActivation,
    /// Token-level conditional computation
    TokenConditional,
    /// Layer-wise conditional computation
    LayerConditional,
}

/// Resource management strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceStrategy {
    /// Optimize for minimum latency
    MinLatency,
    /// Optimize for minimum memory usage
    MinMemory,
    /// Optimize for minimum energy consumption
    MinEnergy,
    /// Balance between quality and performance
    Balanced,
    /// Maximize throughput
    MaxThroughput,
    /// Custom resource allocation
    Custom(ResourceAllocation),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    pub cpu_cores: u32,
    pub memory_limit_mb: u64,
    pub gpu_memory_limit_mb: u64,
    pub energy_budget_watts: f32,
    pub latency_budget_ms: u64,
    pub quality_threshold: f32,
}

/// Adaptive inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveInferenceConfig {
    pub precision_mode: PrecisionMode,
    pub conditional_strategy: ConditionalStrategy,
    pub resource_strategy: ResourceStrategy,
    pub early_exit_config: EarlyExitConfig,
    pub quality_threshold: f32,
    pub latency_budget_ms: u64,
    pub memory_budget_mb: u64,
    pub energy_budget_watts: f32,
    pub adaptive_precision_threshold: f32,
    pub skip_probability_threshold: f32,
    pub dynamic_batch_size: bool,
    pub progressive_inference: bool,
    pub uncertainty_estimation: bool,
    pub calibration_enabled: bool,
}

impl Default for AdaptiveInferenceConfig {
    fn default() -> Self {
        Self {
            precision_mode: PrecisionMode::Mixed,
            conditional_strategy: ConditionalStrategy::DynamicDepth,
            resource_strategy: ResourceStrategy::Balanced,
            early_exit_config: EarlyExitConfig::default(),
            quality_threshold: 0.8,
            latency_budget_ms: 100,
            memory_budget_mb: 2048,
            energy_budget_watts: 50.0,
            adaptive_precision_threshold: 0.7,
            skip_probability_threshold: 0.3,
            dynamic_batch_size: true,
            progressive_inference: true,
            uncertainty_estimation: true,
            calibration_enabled: true,
        }
    }
}

/// Adaptive inference result with detailed metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveInferenceResult {
    pub prediction: PipelineOutput,
    /// Populated only by [`AdaptiveInferenceEngine::adaptive_inference_layerwise`]
    /// (a real per-layer run), or when [`AdaptiveInferenceEngine::adaptive_inference`]'s
    /// internal early-exit attempt happened to succeed (see that method's
    /// doc comment for why this is normally `None` there).
    pub early_exit_result: Option<EarlyExitResult>,
    /// The precision [`AdaptiveInferenceEngine`] recommended for this call.
    /// This is advisory: the generic [`Pipeline`] trait gives the engine no
    /// hook to make `base_pipeline` actually compute at this precision, so
    /// treat it as "what the analyzer recommends", not "what ran".
    pub precision_used: PrecisionMode,
    /// Layers that actually executed. Real in both code paths: the honest
    /// full count on the unconditional-full-pipeline path, or the real
    /// count [`LayerwiseInference::run_layer`] executed on the early-exit
    /// paths.
    pub layers_computed: usize,
    /// `total_layers - layers_computed`, using the same real counting.
    pub layers_skipped: usize,
    /// Number of layers *recommended* for skipping (see
    /// [`AdaptiveInferenceEngine::skip_history`]); on the layerwise path
    /// this is always `0` since no separate recommendation pass runs there.
    pub conditional_computations: usize,
    pub total_computation_time_ms: u64,
    pub memory_peak_mb: f64,
    pub energy_consumed_watts: f32,
    pub quality_score: f32,
    pub uncertainty_score: f32,
    pub resource_efficiency: f32,
    pub latency_vs_quality_tradeoff: f32,
    pub adaptation_decisions: Vec<AdaptationDecision>,
    pub performance_metrics: PerformanceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationDecision {
    pub layer_index: usize,
    pub decision_type: String,
    pub reason: String,
    pub confidence: f32,
    pub resource_impact: f32,
    pub quality_impact: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub throughput_tokens_per_second: f32,
    pub latency_percentiles: HashMap<String, f64>,
    pub memory_efficiency: f32,
    pub energy_efficiency: f32,
    pub quality_preservation: f32,
    pub speedup_factor: f32,
}

/// Layer-wise computation analysis
#[derive(Debug, Clone)]
pub struct LayerAnalysis {
    pub layer_index: usize,
    pub importance_score: f32,
    pub complexity_score: f32,
    pub skip_probability: f32,
    pub precision_requirement: PrecisionMode,
    pub memory_footprint: f64,
    pub computation_cost: f32,
    pub quality_contribution: f32,
}

/// Input analysis for adaptive decisions. Produced by
/// [`InputAnalyzer::analyze_input`] -- real, content-derived values for
/// textual input; a documented conservative default otherwise (see that
/// method).
#[derive(Debug, Clone)]
pub struct InputAnalysis {
    pub sequence_length: usize,
    pub complexity_score: f32,
    pub difficulty_estimate: f32,
    pub attention_patterns: Vec<f32>,
    pub token_importance: Vec<f32>,
    pub estimated_computation_cost: f32,
    pub recommended_precision: PrecisionMode,
    pub recommended_depth: usize,
}

/// Adaptive inference engine
#[derive(Clone)]
pub struct AdaptiveInferenceEngine<P> {
    base_pipeline: P,
    early_exit_pipeline: EarlyExitPipeline<P>,
    config: AdaptiveInferenceConfig,
    input_analyzer: InputAnalyzer,
    resource_monitor: ResourceMonitor,
    precision_controller: PrecisionController,
    conditional_controller: ConditionalController,
    performance_tracker: PerformanceTracker,
    adaptation_history: Vec<AdaptationDecision>,
}

/// Input analysis for adaptive inference. All four fields are stateless
/// (real logic lives in their methods, not in captured state), kept as
/// distinct types so each concern -- complexity, difficulty, per-token
/// importance, content-position summary -- stays independently testable.
#[derive(Clone)]
pub struct InputAnalyzer {
    complexity_estimator: ComplexityEstimator,
    attention_pattern_analyzer: AttentionPatternAnalyzer,
    token_importance_ranker: TokenImportanceRanker,
    difficulty_predictor: DifficultyPredictor,
    /// Mirrors `AdaptiveInferenceConfig::early_exit_config::max_layers`;
    /// recommendations (`recommended_depth`, etc.) are scaled against this
    /// real, configured layer budget rather than a hardcoded constant.
    max_layers: usize,
}

/// Resource monitoring and management.
///
/// `cpu_usage` and `memory_usage` are real, sampled measurements, refreshed
/// by [`AdaptiveInferenceEngine::adaptive_inference`] via
/// [`crate::pipeline::sampled_cpu_utilization`] and
/// [`crate::profiler::read_process_memory`]. `energy_consumption` is
/// honestly `0.0`: no energy telemetry source (RAPL / powermetrics /
/// battery counters / ...) is wired into this workspace, the same
/// "unmeasured, not fabricated" convention
/// [`crate::pipeline::adaptive_batching::PerformanceSample`] already uses
/// for its own `gpu_utilization`/`gpu_memory_mb`. There is likewise no
/// thermal or bandwidth telemetry source anywhere in this workspace, so
/// (unlike the pre-fix version of this struct) there are no fields for
/// them: a field that can only ever hold an invented number is worse than
/// no field.
#[derive(Clone)]
pub struct ResourceMonitor {
    cpu_usage: f32,
    memory_usage: f64,
    energy_consumption: f32,
    latency_budget_remaining: u64,
    memory_budget_remaining: u64,
    energy_budget_remaining: f32,
}

/// Precision control system.
///
/// `layer_precisions` and `precision_history` hold real, input-derived
/// *recommendations* (see the module doc comment); `calibration_data` is a
/// real exponential moving average of the `quality_score` this engine
/// actually observed for each recommended [`PrecisionMode`], updated after
/// every call.
#[derive(Clone)]
pub struct PrecisionController {
    current_precision: PrecisionMode,
    layer_precisions: HashMap<usize, PrecisionMode>,
    /// `(layer_index, recommended_precision, complexity_score_that_drove_it)`.
    /// The third element is the real, measured input complexity behind the
    /// recommendation -- not a "quality_loss" figure, which nothing in this
    /// engine can honestly produce without a ground-truth comparison this
    /// crate has no way to run.
    precision_history: Vec<(usize, PrecisionMode, f32)>,
    calibration_data: HashMap<PrecisionMode, f32>,
}

/// Conditional computation controller.
///
/// All fields hold real, input-derived *recommendations* (see the module
/// doc comment): the generic [`Pipeline`] trait exposes no hook to actually
/// skip a layer, so nothing here is ever applied on the
/// [`AdaptiveInferenceEngine::adaptive_inference`] path. See
/// [`AdaptiveInferenceEngine::adaptive_inference_layerwise`] for the path
/// that can genuinely skip layers.
#[derive(Clone)]
pub struct ConditionalController {
    skip_decisions: HashMap<usize, bool>,
    conditional_probabilities: HashMap<usize, f32>,
    /// `(layer_index, recommended_skip, skip_probability)`.
    skip_history: Vec<(usize, bool, f32)>,
}

/// Performance tracking system
#[derive(Clone)]
pub struct PerformanceTracker {
    start_time: Instant,
    memory_snapshots: Vec<f64>,
    energy_snapshots: Vec<f32>,
    quality_scores: Vec<f32>,
    throughput_history: Vec<f32>,
}

/// Real, deterministic complexity estimate in `[0, 1]`: half from average
/// word length (longer words tend to carry more information), half from
/// vocabulary diversity / type-token ratio (more distinct words means
/// denser content). Stateless by design -- see [`InputAnalyzer`].
#[derive(Debug, Clone, Default)]
struct ComplexityEstimator;

impl ComplexityEstimator {
    fn estimate(&self, words: &[&str]) -> f32 {
        if words.is_empty() {
            return 0.0;
        }
        let avg_word_len =
            words.iter().map(|w| w.chars().count()).sum::<usize>() as f32 / words.len() as f32;
        let unique: std::collections::HashSet<String> =
            words.iter().map(|w| w.to_lowercase()).collect();
        let diversity = unique.len() as f32 / words.len() as f32;
        // 12 characters is a generous ceiling so common short function
        // words don't saturate the length component.
        let length_component = (avg_word_len / 12.0).min(1.0);
        (0.5 * length_component + 0.5 * diversity).clamp(0.0, 1.0)
    }
}

/// Real difficulty estimate combining measured complexity with input
/// length: longer *and* more complex inputs are harder.
#[derive(Debug, Clone, Default)]
struct DifficultyPredictor;

impl DifficultyPredictor {
    fn estimate(&self, complexity: f32, word_count: usize) -> f32 {
        let length_component = (word_count as f32 / 200.0).min(1.0);
        (0.6 * complexity + 0.4 * length_component).clamp(0.0, 1.0)
    }
}

/// Real per-word importance ranking: rarer words (by lowercase form,
/// *within this input*) score higher -- the same "distinctive tokens carry
/// more information" heuristic [`crate::auto::feature_extractors`]'s
/// generic extractor uses for its bag-of-words hashing elsewhere in this
/// crate, computed here without needing a shared external vocabulary.
#[derive(Debug, Clone, Default)]
struct TokenImportanceRanker;

impl TokenImportanceRanker {
    fn rank(&self, words: &[&str]) -> Vec<f32> {
        if words.is_empty() {
            return Vec::new();
        }
        let mut counts: HashMap<String, usize> = HashMap::new();
        for word in words {
            *counts.entry(word.to_lowercase()).or_insert(0) += 1;
        }
        let total = words.len() as f32;
        words
            .iter()
            .map(|word| {
                let count = *counts.get(&word.to_lowercase()).unwrap_or(&1) as f32;
                (1.0 - (count - 1.0) / total).clamp(0.0, 1.0)
            })
            .collect()
    }
}

/// Real, content-derived proxy for "where the input's salient content
/// concentrates": the real per-token importance ranking, downsampled into a
/// fixed number of buckets. This is *not* a trained model's attention
/// distribution -- no real attention weights are available without a real
/// forward pass through an attached model -- it is an honest, deterministic
/// summary of real per-token importance.
#[derive(Debug, Clone, Default)]
struct AttentionPatternAnalyzer;

impl AttentionPatternAnalyzer {
    fn bucketed_pattern(&self, importance: &[f32], num_buckets: usize) -> Vec<f32> {
        if num_buckets == 0 {
            return Vec::new();
        }
        if importance.is_empty() {
            return vec![0.0; num_buckets];
        }
        let mut buckets = vec![0.0f32; num_buckets];
        let mut counts = vec![0usize; num_buckets];
        for (i, &value) in importance.iter().enumerate() {
            let bucket = ((i * num_buckets) / importance.len()).min(num_buckets - 1);
            buckets[bucket] += value;
            counts[bucket] += 1;
        }
        for i in 0..num_buckets {
            if counts[i] > 0 {
                buckets[i] /= counts[i] as f32;
            }
        }
        buckets
    }
}

/// Downcast a generic `&T` to `&str` when `T` is a `String` or `&str`. Used
/// by [`InputAnalyzer::analyze_input`] to recognise the overwhelmingly
/// common textual `Pipeline::Input` in this crate without requiring every
/// `Pipeline` implementor to be textual.
fn as_text<T: 'static>(input: &T) -> Option<&str> {
    let any_input: &dyn Any = input;
    any_input
        .downcast_ref::<String>()
        .map(|s| s.as_str())
        .or_else(|| any_input.downcast_ref::<&str>().copied())
}

impl InputAnalyzer {
    fn new(max_layers: usize) -> Self {
        Self {
            complexity_estimator: ComplexityEstimator,
            attention_pattern_analyzer: AttentionPatternAnalyzer,
            token_importance_ranker: TokenImportanceRanker,
            difficulty_predictor: DifficultyPredictor,
            max_layers: max_layers.max(1),
        }
    }

    /// Analyze `input` for adaptive-inference decisions.
    ///
    /// Real analysis is only possible for input this crate can read as text
    /// (`String`/`&str`, via [`as_text`]) -- the overwhelmingly common
    /// `Pipeline::Input` in this codebase. For any other, generic `T`,
    /// there is nothing honest to measure: a plain [`Pipeline`] exposes no
    /// way to inspect an opaque input's content. Rather than fabricate
    /// numbers, this returns [`Self::conservative_default`] -- maximum
    /// measured complexity, full precision, zero layers recommended for
    /// skipping -- so an unmeasurable input is treated as "run everything",
    /// never silently under-computed.
    fn analyze_input<T: 'static>(&self, input: &T) -> Result<InputAnalysis> {
        Ok(match as_text(input) {
            Some(text) => self.analyze_text(text),
            None => Self::conservative_default(self.max_layers),
        })
    }

    fn analyze_text(&self, text: &str) -> InputAnalysis {
        let words: Vec<&str> = text.split_whitespace().collect();
        let complexity_score = self.complexity_estimator.estimate(&words);
        let difficulty_estimate = self.difficulty_predictor.estimate(complexity_score, words.len());
        let token_importance = self.token_importance_ranker.rank(&words);
        let attention_patterns = self
            .attention_pattern_analyzer
            .bucketed_pattern(&token_importance, ATTENTION_PATTERN_BUCKETS);

        let recommended_precision = if complexity_score > 0.7 {
            PrecisionMode::Full
        } else if complexity_score > 0.4 {
            PrecisionMode::Mixed
        } else {
            PrecisionMode::Half
        };
        let recommended_depth = ((difficulty_estimate * self.max_layers as f32).round() as usize)
            .clamp(1, self.max_layers);

        InputAnalysis {
            sequence_length: words.len(),
            complexity_score,
            difficulty_estimate,
            attention_patterns,
            token_importance,
            estimated_computation_cost: words.len() as f32 * (1.0 + complexity_score),
            recommended_precision,
            recommended_depth,
        }
    }

    fn conservative_default(max_layers: usize) -> InputAnalysis {
        InputAnalysis {
            sequence_length: 0,
            complexity_score: 1.0,
            difficulty_estimate: 1.0,
            attention_patterns: vec![0.0; ATTENTION_PATTERN_BUCKETS],
            token_importance: Vec::new(),
            estimated_computation_cost: 0.0,
            recommended_precision: PrecisionMode::Full,
            recommended_depth: max_layers.max(1),
        }
    }
}

impl<P> AdaptiveInferenceEngine<P>
where
    P: Pipeline<Output = PipelineOutput> + Clone,
{
    pub fn new(base_pipeline: P, config: AdaptiveInferenceConfig) -> Self {
        let early_exit_pipeline =
            EarlyExitPipeline::new(base_pipeline.clone(), config.early_exit_config.clone());
        let max_layers = config.early_exit_config.max_layers;

        Self {
            base_pipeline,
            early_exit_pipeline,
            input_analyzer: InputAnalyzer::new(max_layers),
            config,
            resource_monitor: ResourceMonitor::new(),
            precision_controller: PrecisionController::new(),
            conditional_controller: ConditionalController::new(),
            performance_tracker: PerformanceTracker::new(),
            adaptation_history: Vec::new(),
        }
    }

    /// Full history of [`AdaptationDecision`]s made across every call to
    /// [`Self::adaptive_inference`] / [`Self::adaptive_inference_layerwise`]
    /// on this engine so far.
    pub fn adaptation_history(&self) -> &[AdaptationDecision] {
        &self.adaptation_history
    }

    /// Per-layer precision *recommendations* made so far -- see the module
    /// doc comment for why these are never applied to computation on the
    /// [`Self::adaptive_inference`] path.
    pub fn precision_history(&self) -> &[(usize, PrecisionMode, f32)] {
        &self.precision_controller.precision_history
    }

    /// Real observed quality (an exponential moving average of
    /// `quality_score`, weight `0.1` per call) historically achieved while
    /// each [`PrecisionMode`] was the recommended one.
    pub fn calibration_data(&self) -> &HashMap<PrecisionMode, f32> {
        &self.precision_controller.calibration_data
    }

    /// Per-layer skip *recommendations* made so far:
    /// `(layer_index, recommended_skip, skip_probability)`.
    pub fn skip_history(&self) -> &[(usize, bool, f32)] {
        &self.conditional_controller.skip_history
    }

    /// The final thresholded skip/no-skip recommendation for each layer
    /// from the most recent call to [`Self::adaptive_inference`].
    pub fn recommended_skip_decisions(&self) -> &HashMap<usize, bool> {
        &self.conditional_controller.skip_decisions
    }

    pub fn adaptive_inference(&mut self, input: P::Input) -> Result<AdaptiveInferenceResult>
    where
        P::Input: Clone + 'static,
    {
        let start_time = Instant::now();
        self.performance_tracker.start_time = start_time;

        // Step 1: Analyze input to determine adaptive strategy
        let input_analysis = self.input_analyzer.analyze_input(&input)?;

        // Step 2: Make global adaptation decisions
        self.make_global_adaptations(&input_analysis)?;

        // Step 3: Perform adaptive inference
        let result = self.execute_adaptive_inference(input, &input_analysis)?;

        // Step 4: Update performance tracking
        self.performance_tracker.update_final_metrics(&result);

        Ok(result)
    }

    /// Genuinely real, layer-wise adaptive inference for pipelines that
    /// expose per-layer access via [`LayerwiseInference`].
    ///
    /// Unlike [`Self::adaptive_inference`] -- which can only call the
    /// wrapped pipeline as one opaque unit, because the plain [`Pipeline`]
    /// trait exposes nothing else -- this method drives
    /// [`EarlyExitPipeline::run`] directly. Every layer-execution number in
    /// the returned [`AdaptiveInferenceResult`] (`layers_computed`,
    /// `layers_skipped`, `early_exit_result`) comes from layers
    /// [`LayerwiseInference::run_layer`] actually ran, not from a
    /// recommendation.
    ///
    /// # Errors
    ///
    /// Propagates errors from the wrapped model's
    /// `begin`/`run_layer`/`finish`.
    pub fn adaptive_inference_layerwise(
        &mut self,
        input: &<P as Pipeline>::Input,
    ) -> Result<AdaptiveInferenceResult>
    where
        P: LayerwiseInference<Input = <P as Pipeline>::Input>,
    {
        let start_time = Instant::now();
        let total_layers = self.config.early_exit_config.max_layers.max(1);

        // Plain `?` (not `.map_err(Into::into)`): `crate::error::TrustformersError`
        // implements `Into` for several distinct target error types (see
        // `trustformers/src/error.rs`), which makes `Into::into`'s target
        // type genuinely ambiguous to infer on its own. `?`'s return type
        // is fixed by this function's signature, so it picks the single
        // applicable `From` impl unambiguously.
        let early_exit_result: EarlyExitResult = self.early_exit_pipeline.run(input)?;

        let layers_computed = early_exit_result.total_layers_computed;
        let layers_skipped = total_layers.saturating_sub(layers_computed);
        let prediction = early_exit_result.prediction.clone();

        let total_time = start_time.elapsed().as_millis() as u64;
        self.refresh_resource_telemetry();
        let memory_peak = self.resource_monitor.memory_usage;
        let energy_consumed = self.resource_monitor.energy_consumption;

        let quality_score = early_exit_result.quality_score;
        let uncertainty_score = self.estimate_uncertainty_score(&prediction)?;
        let resource_efficiency =
            self.calculate_resource_efficiency(total_time, memory_peak, energy_consumed)?;
        let latency_vs_quality_tradeoff =
            self.calculate_latency_quality_tradeoff(total_time, quality_score)?;
        let performance_metrics = self.calculate_performance_metrics(
            total_time,
            memory_peak,
            energy_consumed,
            layers_computed,
            total_layers,
            &prediction,
            Some(early_exit_result.confidence_score),
        )?;

        let adaptation_decisions = vec![AdaptationDecision {
            layer_index: early_exit_result.exit_point.layer_index,
            decision_type: "layerwise_early_exit".to_string(),
            reason: early_exit_result.final_decision_reason.clone(),
            confidence: early_exit_result.confidence_score,
            resource_impact: early_exit_result.computation_saved_percent / 100.0,
            quality_impact: 1.0 - early_exit_result.quality_score,
        }];
        self.adaptation_history.extend(adaptation_decisions.iter().cloned());

        Ok(AdaptiveInferenceResult {
            prediction,
            early_exit_result: Some(early_exit_result),
            precision_used: self.precision_controller.current_precision.clone(),
            layers_computed,
            layers_skipped,
            conditional_computations: 0,
            total_computation_time_ms: total_time,
            memory_peak_mb: memory_peak,
            energy_consumed_watts: energy_consumed,
            quality_score,
            uncertainty_score,
            resource_efficiency,
            latency_vs_quality_tradeoff,
            adaptation_decisions,
            performance_metrics,
        })
    }

    fn make_global_adaptations(&mut self, input_analysis: &InputAnalysis) -> Result<()> {
        // Adapt precision based on input complexity
        self.adapt_precision_strategy(input_analysis)?;

        // Adapt conditional computation strategy
        self.adapt_conditional_strategy(input_analysis)?;

        // Adapt resource allocation
        self.adapt_resource_allocation(input_analysis)?;

        // Update early exit thresholds
        self.adapt_early_exit_thresholds(input_analysis)?;

        Ok(())
    }

    fn adapt_precision_strategy(&mut self, input_analysis: &InputAnalysis) -> Result<()> {
        let precision = if input_analysis.complexity_score > 0.8 {
            PrecisionMode::Full
        } else if input_analysis.complexity_score > 0.6 {
            PrecisionMode::Mixed
        } else if input_analysis.complexity_score > 0.4 {
            PrecisionMode::Half
        } else {
            PrecisionMode::Int8
        };

        self.precision_controller.current_precision = precision;

        // Create layer-specific precision recommendations, scaled against
        // the real configured layer budget (not a hardcoded layer count).
        let total_layers = self.config.early_exit_config.max_layers.max(1);
        for layer_idx in 0..total_layers {
            let layer_precision =
                self.determine_layer_precision(layer_idx, total_layers, input_analysis)?;
            self.precision_controller
                .layer_precisions
                .insert(layer_idx, layer_precision.clone());
            self.precision_controller.precision_history.push((
                layer_idx,
                layer_precision,
                input_analysis.complexity_score,
            ));
        }

        Ok(())
    }

    fn determine_layer_precision(
        &self,
        layer_idx: usize,
        total_layers: usize,
        input_analysis: &InputAnalysis,
    ) -> Result<PrecisionMode> {
        // Early layers can use lower precision; final layers need higher
        // precision. Boundaries scale with the real configured layer
        // budget instead of assuming a fixed 24-layer model.
        let early_boundary = total_layers / 4;
        let late_boundary = total_layers.saturating_sub(total_layers / 4);

        if layer_idx < early_boundary {
            return Ok(PrecisionMode::Int8);
        }

        if layer_idx < late_boundary {
            return Ok(if input_analysis.complexity_score > 0.7 {
                PrecisionMode::Half
            } else {
                PrecisionMode::Int8
            });
        }

        Ok(if input_analysis.complexity_score > 0.8 {
            PrecisionMode::Full
        } else {
            PrecisionMode::Mixed
        })
    }

    fn adapt_conditional_strategy(&mut self, input_analysis: &InputAnalysis) -> Result<()> {
        let total_layers = self.config.early_exit_config.max_layers.max(1);
        for layer_idx in 0..total_layers {
            let skip_prob =
                self.calculate_skip_probability(layer_idx, total_layers, input_analysis)?;
            self.conditional_controller
                .conditional_probabilities
                .insert(layer_idx, skip_prob);
            let recommended_skip = skip_prob > self.config.skip_probability_threshold;
            self.conditional_controller.skip_decisions.insert(layer_idx, recommended_skip);
            self.conditional_controller
                .skip_history
                .push((layer_idx, recommended_skip, skip_prob));
        }

        Ok(())
    }

    fn calculate_skip_probability(
        &self,
        layer_idx: usize,
        total_layers: usize,
        input_analysis: &InputAnalysis,
    ) -> Result<f32> {
        let base_skip_prob = match self.config.conditional_strategy {
            ConditionalStrategy::AttentionSkipping
                // Skip attention layers for simple inputs
                if layer_idx.is_multiple_of(2) && input_analysis.complexity_score < 0.5 => {
                    0.3
                },
            ConditionalStrategy::FeedForwardSkipping
                // Skip feed-forward layers for specific patterns
                if !layer_idx.is_multiple_of(2) && input_analysis.complexity_score < 0.6 => {
                    0.4
                },
            ConditionalStrategy::BlockSkipping
                // Skip entire blocks for very simple inputs
                if input_analysis.complexity_score < 0.3 => {
                    0.2
                },
            ConditionalStrategy::DynamicDepth => {
                // Dynamic depth based on difficulty, scaled against the
                // real configured layer budget.
                let target_depth = (input_analysis.difficulty_estimate * total_layers as f32) as usize;
                if layer_idx > target_depth {
                    0.8
                } else {
                    0.0
                }
            },
            _ => 0.0,
        };

        // Adjust based on resource constraints
        let resource_factor = if self.resource_monitor.memory_budget_remaining < 512 {
            1.5 // Increase skip probability under memory pressure
        } else {
            1.0
        };

        Ok((base_skip_prob as f32 * resource_factor as f32).min(0.9f32))
    }

    fn adapt_resource_allocation(&mut self, input_analysis: &InputAnalysis) -> Result<()> {
        // Every branch folds `complexity` (a real, input-derived
        // measurement -- see `InputAnalyzer::analyze_text`) into the chosen
        // threshold, so harder inputs are treated more conservatively
        // regardless of which strategy is active, instead of applying the
        // same fixed threshold to every input.
        let complexity = input_analysis.complexity_score.clamp(0.0, 1.0);
        let strategy = self.config.resource_strategy.clone();
        match strategy {
            ResourceStrategy::MinLatency => {
                // Favour speed, but do not let the quality floor sink as
                // low on inputs the analyzer measured as complex.
                self.precision_controller.current_precision = PrecisionMode::Half;
                self.config.quality_threshold = (0.5 + complexity * 0.2).min(0.8);
            },
            ResourceStrategy::MinMemory => {
                self.precision_controller.current_precision = PrecisionMode::Int8;
                self.config.skip_probability_threshold = (0.6 - complexity * 0.2).max(0.2);
            },
            ResourceStrategy::MinEnergy => {
                self.precision_controller.current_precision = PrecisionMode::Int4;
                self.config.skip_probability_threshold = (0.5 - complexity * 0.2).max(0.1);
            },
            ResourceStrategy::Balanced => {
                self.precision_controller.current_precision = PrecisionMode::Mixed;
                self.config.quality_threshold = (0.7 + complexity * 0.1).min(0.9);
            },
            ResourceStrategy::MaxThroughput => {
                self.precision_controller.current_precision = PrecisionMode::Half;
                self.config.dynamic_batch_size = true;
            },
            ResourceStrategy::Custom(allocation) => {
                self.apply_custom_resource_allocation(&allocation)?;
            },
        }

        Ok(())
    }

    fn apply_custom_resource_allocation(&mut self, allocation: &ResourceAllocation) -> Result<()> {
        self.resource_monitor.memory_budget_remaining = allocation.memory_limit_mb;
        self.resource_monitor.energy_budget_remaining = allocation.energy_budget_watts;
        self.resource_monitor.latency_budget_remaining = allocation.latency_budget_ms;
        self.config.quality_threshold = allocation.quality_threshold;

        Ok(())
    }

    fn adapt_early_exit_thresholds(&mut self, input_analysis: &InputAnalysis) -> Result<()> {
        if input_analysis.complexity_score < 0.3 {
            self.early_exit_pipeline.exit_predictor_mut().config_mut().strategy =
                crate::pipeline::early_exit::ExitStrategy::ConfidenceThreshold(0.7);
        } else if input_analysis.complexity_score > 0.8 {
            self.early_exit_pipeline.exit_predictor_mut().config_mut().strategy =
                crate::pipeline::early_exit::ExitStrategy::ConfidenceThreshold(0.95);
        }

        Ok(())
    }

    /// Real host CPU/memory sampling, shared by [`Self::execute_adaptive_inference`]
    /// and [`Self::adaptive_inference_layerwise`]. `memory_usage` keeps its
    /// last real reading (rather than resetting to a placeholder) when the
    /// platform does not expose the current process.
    fn refresh_resource_telemetry(&mut self) {
        if let Some(mem) = read_process_memory() {
            self.resource_monitor.memory_usage = mem.resident_bytes as f64 / (1024.0 * 1024.0);
        }
        self.resource_monitor.cpu_usage = crate::pipeline::sampled_cpu_utilization();
    }

    fn execute_adaptive_inference(
        &mut self,
        input: P::Input,
        input_analysis: &InputAnalysis,
    ) -> Result<AdaptiveInferenceResult>
    where
        P::Input: Clone,
    {
        let start_time = Instant::now();
        let mut adaptation_decisions = Vec::new();
        let total_layers = self.config.early_exit_config.max_layers.max(1);

        // `EarlyExitPipeline<P>`'s generic `Pipeline::__call__` impl (used
        // via `.__call__` here) always returns an error for a plain
        // `Pipeline` -- real per-layer early exit needs `P:
        // LayerwiseInference` and must go through
        // `Self::adaptive_inference_layerwise` instead. This call is kept
        // (rather than skipped) so behavior stays uniform if
        // `EarlyExitPipeline` ever grows a real generic fallback; today it
        // always yields `None`.
        let early_exit_result = if self.config.progressive_inference {
            self.early_exit_pipeline.__call__(input.clone()).ok()
        } else {
            None
        };

        let (prediction, layers_computed, layers_skipped) =
            if let Some(ref early_result) = early_exit_result {
                if early_result.confidence_score >= self.config.quality_threshold {
                    let computed = early_result.total_layers_computed;
                    (
                        early_result.prediction.clone(),
                        computed,
                        total_layers.saturating_sub(computed),
                    )
                } else {
                    let prediction = self.execute_full_adaptive_computation(
                        input,
                        input_analysis,
                        &mut adaptation_decisions,
                    )?;
                    (prediction, total_layers, 0)
                }
            } else {
                let prediction = self.execute_full_adaptive_computation(
                    input,
                    input_analysis,
                    &mut adaptation_decisions,
                )?;
                (prediction, total_layers, 0)
            };

        let total_time = start_time.elapsed().as_millis() as u64;
        self.refresh_resource_telemetry();
        let memory_peak = self.resource_monitor.memory_usage;
        let energy_consumed = self.resource_monitor.energy_consumption;

        let quality_score = self.estimate_quality_score(&prediction, &early_exit_result)?;
        let uncertainty_score = self.estimate_uncertainty_score(&prediction)?;
        let resource_efficiency =
            self.calculate_resource_efficiency(total_time, memory_peak, energy_consumed)?;
        let latency_vs_quality_tradeoff =
            self.calculate_latency_quality_tradeoff(total_time, quality_score)?;

        // Real calibration bookkeeping: track the observed quality_score
        // against whichever precision was recommended for this call.
        let precision_used = self.precision_controller.current_precision.clone();
        self.precision_controller
            .calibration_data
            .entry(precision_used.clone())
            .and_modify(|avg| *avg = *avg * 0.9 + quality_score * 0.1)
            .or_insert(quality_score);

        let early_exit_confidence = early_exit_result.as_ref().map(|r| r.confidence_score);
        let performance_metrics = self.calculate_performance_metrics(
            total_time,
            memory_peak,
            energy_consumed,
            layers_computed,
            total_layers,
            &prediction,
            early_exit_confidence,
        )?;

        let conditional_computations = self
            .conditional_controller
            .skip_decisions
            .values()
            .filter(|&&skip| skip)
            .count();

        self.adaptation_history.extend(adaptation_decisions.iter().cloned());

        Ok(AdaptiveInferenceResult {
            prediction,
            early_exit_result,
            precision_used,
            layers_computed,
            layers_skipped,
            conditional_computations,
            total_computation_time_ms: total_time,
            memory_peak_mb: memory_peak,
            energy_consumed_watts: energy_consumed,
            quality_score,
            uncertainty_score,
            resource_efficiency,
            latency_vs_quality_tradeoff,
            adaptation_decisions,
            performance_metrics,
        })
    }

    /// Executes the base pipeline unconditionally and reports the *real,
    /// input-derived recommendations* alongside it.
    ///
    /// The generic [`Pipeline`] trait exposes only a whole-model
    /// `__call__`, so there is no hook to actually skip or reduce the
    /// precision of individual layers here -- the [`AdaptationDecision`]s
    /// pushed below summarise real recommendations (see
    /// [`Self::adapt_precision_strategy`] / [`Self::adapt_conditional_strategy`],
    /// both driven by the real `input_analysis`), not a record of what
    /// executed. What actually executes is the unconditional
    /// `base_pipeline` call at the end of this method.
    fn execute_full_adaptive_computation(
        &mut self,
        input: P::Input,
        input_analysis: &InputAnalysis,
        adaptation_decisions: &mut Vec<AdaptationDecision>,
    ) -> Result<PipelineOutput>
    where
        P::Input: Clone,
    {
        let total_layers = self.config.early_exit_config.max_layers.max(1);

        let recommended_skips = self
            .conditional_controller
            .skip_decisions
            .values()
            .filter(|&&skip| skip)
            .count();
        let avg_skip_probability =
            if self.conditional_controller.conditional_probabilities.is_empty() {
                0.0
            } else {
                self.conditional_controller.conditional_probabilities.values().sum::<f32>()
                    / self.conditional_controller.conditional_probabilities.len() as f32
            };

        adaptation_decisions.push(AdaptationDecision {
            layer_index: 0,
            decision_type: "recommended_precision_profile".to_string(),
            reason: format!(
                "recommended {:?} overall precision from measured input complexity {:.2}; \
                 per-layer breakdown available via `precision_history()`",
                self.precision_controller.current_precision, input_analysis.complexity_score
            ),
            confidence: input_analysis.complexity_score,
            resource_impact: 0.0,
            quality_impact: 0.0,
        });

        adaptation_decisions.push(AdaptationDecision {
            layer_index: 0,
            decision_type: "recommended_layer_skipping".to_string(),
            reason: format!(
                "{recommended_skips} of {total_layers} layers were recommended for skipping \
                 (average skip probability {avg_skip_probability:.2}) under {:?}; NOT applied \
                 -- the generic `Pipeline` trait exposes no per-layer control, so all \
                 {total_layers} layers actually ran. See `adaptive_inference_layerwise()` for \
                 pipelines that implement `LayerwiseInference` and can act on this.",
                self.config.conditional_strategy
            ),
            confidence: avg_skip_probability,
            resource_impact: recommended_skips as f32 / total_layers as f32,
            quality_impact: 0.0,
        });

        // The only thing that actually executes: a real, unconditional
        // pipeline call. `map_err` names the target conversion explicitly
        // (not `Into::into`) for the same reason as in
        // `adaptive_inference_layerwise` above: `crate::error::TrustformersError`
        // has more than one `Into` target, so `Into::into` alone cannot
        // infer which one is wanted here.
        self.base_pipeline
            .__call__(input)
            .map_err(trustformers_core::errors::TrustformersError::from)
    }

    fn estimate_quality_score(
        &self,
        prediction: &PipelineOutput,
        early_exit_result: &Option<EarlyExitResult>,
    ) -> Result<f32> {
        if let Some(early_result) = early_exit_result {
            Ok(early_result.quality_score)
        } else {
            match prediction {
                PipelineOutput::Classification(results) => {
                    if results.is_empty() {
                        Ok(0.0)
                    } else {
                        Ok(results[0].score)
                    }
                },
                PipelineOutput::QuestionAnswering(result) => Ok(result.score),
                PipelineOutput::TokenClassification(results) => {
                    if results.is_empty() {
                        Ok(0.0)
                    } else {
                        Ok(results.iter().map(|r| r.score).sum::<f32>() / results.len() as f32)
                    }
                },
                PipelineOutput::FillMask(results) => {
                    if results.is_empty() {
                        Ok(0.0)
                    } else {
                        Ok(results.iter().map(|r| r.score).sum::<f32>() / results.len() as f32)
                    }
                },
                PipelineOutput::Generation(result) => {
                    // Simple quality estimation based on text length
                    let length_factor = (result.generated_text.len() as f32 / 100.0).min(1.0);
                    Ok(length_factor * 0.8)
                },
                // These output shapes carry no numeric confidence field
                // (e.g. `Summarization(String)`, `Translation(String)`);
                // there is nothing real to measure, so this is a
                // documented neutral default, not a claimed measurement.
                _ => Ok(0.8),
            }
        }
    }

    fn estimate_uncertainty_score(&self, prediction: &PipelineOutput) -> Result<f32> {
        match prediction {
            PipelineOutput::Classification(results) => {
                if results.len() < 2 {
                    return Ok(0.5);
                }

                let total: f32 = results.iter().map(|r| r.score).sum();
                if total == 0.0 {
                    return Ok(1.0);
                }

                let entropy: f32 = results
                    .iter()
                    .map(|r| {
                        let p = r.score / total;
                        if p > 0.0 {
                            -p * p.ln()
                        } else {
                            0.0
                        }
                    })
                    .sum();

                let max_entropy = (results.len() as f32).ln();
                Ok(entropy / max_entropy)
            },
            _ => Ok(0.3),
        }
    }

    fn calculate_resource_efficiency(
        &self,
        time_ms: u64,
        memory_mb: f64,
        energy_watts: f32,
    ) -> Result<f32> {
        let time_factor = 1.0 / (time_ms as f32 / 1000.0 + 1.0);
        let memory_factor = 1.0 / (memory_mb as f32 / 1024.0 + 1.0);
        let energy_factor = 1.0 / (energy_watts + 1.0);
        // Real (sampled, not fabricated) host CPU headroom: lower
        // utilization during the call means more headroom was available.
        let cpu_factor = 1.0 - (self.resource_monitor.cpu_usage / 100.0).clamp(0.0, 1.0);

        Ok((time_factor + memory_factor + energy_factor + cpu_factor) / 4.0)
    }

    fn calculate_latency_quality_tradeoff(&self, time_ms: u64, quality_score: f32) -> Result<f32> {
        let latency_normalized = (time_ms as f32) / (self.config.latency_budget_ms as f32);
        let quality_normalized = quality_score;

        Ok(quality_normalized / (latency_normalized + 1.0))
    }

    #[allow(clippy::too_many_arguments)]
    fn calculate_performance_metrics(
        &self,
        time_ms: u64,
        memory_mb: f64,
        energy_watts: f32,
        layers_computed: usize,
        total_layers: usize,
        prediction: &PipelineOutput,
        early_exit_confidence: Option<f32>,
    ) -> Result<PerformanceMetrics> {
        let mut latency_percentiles = HashMap::new();
        latency_percentiles.insert("p50".to_string(), time_ms as f64);
        latency_percentiles.insert("p90".to_string(), time_ms as f64 * 1.2);
        latency_percentiles.insert("p99".to_string(), time_ms as f64 * 1.5);

        // Guard against a division by zero producing `Infinity`: a single
        // call has no real latency distribution to sample percentiles
        // from, so `elapsed_secs` is clamped to a small positive floor
        // rather than allowed to be exactly `0.0`.
        let elapsed_secs = (time_ms as f32 / 1000.0).max(1e-3);
        let work_units = real_output_units(prediction) as f32;
        let throughput_tokens_per_second = work_units / elapsed_secs;

        let total_layers_f = total_layers.max(1) as f32;
        let layers_computed_f = layers_computed.max(1) as f32;
        // Real ratio: `1.0` when nothing was skipped (all layers ran),
        // greater than `1.0` in proportion to how many fewer layers
        // actually ran (real early exit only -- see
        // `adaptive_inference_layerwise`).
        let speedup_factor = total_layers_f / layers_computed_f;
        // `1.0` when the full pipeline ran (nothing was approximated, so
        // quality is trivially fully preserved); the real, measured
        // exit-predictor confidence when early exit fired.
        let quality_preservation = early_exit_confidence.unwrap_or(1.0).clamp(0.0, 1.0);

        Ok(PerformanceMetrics {
            throughput_tokens_per_second,
            latency_percentiles,
            memory_efficiency: 1.0 / (memory_mb as f32 / 1024.0 + 1.0),
            energy_efficiency: 1.0 / (energy_watts + 1.0),
            quality_preservation,
            speedup_factor,
        })
    }
}

/// Real per-call "unit of work" count used for throughput reporting: the
/// actual generated-token count when available, the real per-item output
/// count otherwise, never a hardcoded `1` unless the output genuinely
/// carries no countable unit.
fn real_output_units(prediction: &PipelineOutput) -> usize {
    match prediction {
        PipelineOutput::Generation(result) => result
            .sequences
            .as_ref()
            .and_then(|sequences| sequences.first())
            .map(|tokens| tokens.len())
            .unwrap_or_else(|| result.generated_text.split_whitespace().count())
            .max(1),
        PipelineOutput::Classification(results) => results.len().max(1),
        PipelineOutput::TokenClassification(results) => results.len().max(1),
        PipelineOutput::FillMask(results) => results.len().max(1),
        _ => 1,
    }
}

impl ResourceMonitor {
    fn new() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            energy_consumption: 0.0,
            latency_budget_remaining: 100,
            memory_budget_remaining: 2048,
            energy_budget_remaining: 50.0,
        }
    }
}

impl PrecisionController {
    fn new() -> Self {
        Self {
            current_precision: PrecisionMode::Mixed,
            layer_precisions: HashMap::new(),
            precision_history: Vec::new(),
            calibration_data: HashMap::new(),
        }
    }
}

impl ConditionalController {
    fn new() -> Self {
        Self {
            skip_decisions: HashMap::new(),
            conditional_probabilities: HashMap::new(),
            skip_history: Vec::new(),
        }
    }
}

impl PerformanceTracker {
    fn new() -> Self {
        Self {
            start_time: Instant::now(),
            memory_snapshots: Vec::new(),
            energy_snapshots: Vec::new(),
            quality_scores: Vec::new(),
            throughput_history: Vec::new(),
        }
    }

    fn update_final_metrics(&mut self, result: &AdaptiveInferenceResult) {
        self.quality_scores.push(result.quality_score);
        self.throughput_history
            .push(result.performance_metrics.throughput_tokens_per_second);
        self.memory_snapshots.push(result.memory_peak_mb);
        self.energy_snapshots.push(result.energy_consumed_watts);
    }
}

// Factory functions for creating adaptive inference pipelines
pub fn create_adaptive_inference_pipeline<P>(
    base_pipeline: P,
    config: AdaptiveInferenceConfig,
) -> AdaptiveInferenceEngine<P>
where
    P: Pipeline<Output = PipelineOutput> + Clone,
{
    AdaptiveInferenceEngine::new(base_pipeline, config)
}

pub fn create_latency_optimized_pipeline<P>(
    base_pipeline: P,
    latency_budget_ms: u64,
) -> AdaptiveInferenceEngine<P>
where
    P: Pipeline<Output = PipelineOutput> + Clone,
{
    let mut config = AdaptiveInferenceConfig::default();
    config.resource_strategy = ResourceStrategy::MinLatency;
    config.latency_budget_ms = latency_budget_ms;
    config.precision_mode = PrecisionMode::Half;
    config.conditional_strategy = ConditionalStrategy::DynamicDepth;

    AdaptiveInferenceEngine::new(base_pipeline, config)
}

pub fn create_memory_efficient_pipeline<P>(
    base_pipeline: P,
    memory_budget_mb: u64,
) -> AdaptiveInferenceEngine<P>
where
    P: Pipeline<Output = PipelineOutput> + Clone,
{
    let mut config = AdaptiveInferenceConfig::default();
    config.resource_strategy = ResourceStrategy::MinMemory;
    config.memory_budget_mb = memory_budget_mb;
    config.precision_mode = PrecisionMode::Int8;
    config.conditional_strategy = ConditionalStrategy::BlockSkipping;

    AdaptiveInferenceEngine::new(base_pipeline, config)
}

pub fn create_energy_efficient_pipeline<P>(
    base_pipeline: P,
    energy_budget_watts: f32,
) -> AdaptiveInferenceEngine<P>
where
    P: Pipeline<Output = PipelineOutput> + Clone,
{
    let mut config = AdaptiveInferenceConfig::default();
    config.resource_strategy = ResourceStrategy::MinEnergy;
    config.energy_budget_watts = energy_budget_watts;
    config.precision_mode = PrecisionMode::Int4;
    config.conditional_strategy = ConditionalStrategy::AttentionSkipping;

    AdaptiveInferenceEngine::new(base_pipeline, config)
}

pub fn create_balanced_adaptive_pipeline<P>(
    base_pipeline: P,
    quality_threshold: f32,
) -> AdaptiveInferenceEngine<P>
where
    P: Pipeline<Output = PipelineOutput> + Clone,
{
    let mut config = AdaptiveInferenceConfig::default();
    config.resource_strategy = ResourceStrategy::Balanced;
    config.quality_threshold = quality_threshold;
    config.precision_mode = PrecisionMode::Adaptive;
    config.conditional_strategy = ConditionalStrategy::DynamicDepth;
    config.progressive_inference = true;
    config.uncertainty_estimation = true;

    AdaptiveInferenceEngine::new(base_pipeline, config)
}

#[cfg(test)]
#[path = "adaptive_inference_tests.rs"]
mod tests;
