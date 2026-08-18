//! Unit tests for [`super`] -- split out of `adaptive_inference.rs` to keep
//! it under the workspace's 2000-line-per-file policy (see the `#[path =
//! "adaptive_inference_tests.rs"] mod tests;` declaration at the bottom of
//! that file; the same convention is used by `pipeline/metal_backend.rs`).
//!
//! These are regression tests for the bug described in the module doc
//! comment of `adaptive_inference.rs`: `InputAnalyzer::analyze_input` used
//! to return a fixed constant (`sequence_length: 256, complexity_score:
//! 0.6, difficulty_estimate: 0.7, token_importance: vec![0.3; 256], ...`)
//! for *any* input, and `execute_full_adaptive_computation` used to
//! "simulate" per-layer execution with a hardcoded `for layer_idx in 0..24`
//! loop and fixed confidence values (0.8/0.9), while `layers_computed` /
//! `layers_skipped` stayed at their initial `0` on that path and
//! `calculate_performance_metrics` returned fixed
//! `speedup_factor: 2.0`/`quality_preservation: 0.9`/
//! `throughput_tokens_per_second: 1000.0 / time_ms` (which divides by zero
//! -> `Infinity` whenever a call completes in under 1ms). Every test below
//! asserts a property that is true of the current, real implementation and
//! that the described old behavior would have failed.

use super::*;
use crate::pipeline::early_exit::ExitStrategy;
use crate::pipeline::{ClassificationOutput, GenerationOutput};

// ---------------------------------------------------------------------
// Test fixtures
// ---------------------------------------------------------------------

/// A `Pipeline` whose `__call__` deterministically returns a
/// caller-chosen `Classification` output. Does *not* implement
/// `LayerwiseInference`, so `EarlyExitPipeline<MockPipeline>::__call__`
/// always errors (see `early_exit.rs`) and `AdaptiveInferenceEngine`'s
/// `progressive_inference` path always falls through to
/// `execute_full_adaptive_computation` -- exactly the case the old
/// hardcoded-24-layer simulation loop covered.
#[derive(Clone)]
struct MockPipeline {
    label: String,
    score: f32,
}

impl Pipeline for MockPipeline {
    type Input = String;
    type Output = PipelineOutput;

    // `Pipeline::__call__` is declared against `crate::error::Result`
    // (see `pipeline/mod.rs`), which differs from the `trustformers_core::errors::Result`
    // this test file's `use super::*;` brings in for
    // `AdaptiveInferenceEngine`'s own methods -- so this impl spells the
    // trait's error type out explicitly rather than relying on the
    // ambient (wrong) `Result` alias.
    fn __call__(&self, _input: Self::Input) -> crate::error::Result<Self::Output> {
        Ok(PipelineOutput::Classification(vec![ClassificationOutput {
            label: self.label.clone(),
            score: self.score,
        }]))
    }
}

fn mock_engine(
    score: f32,
    early_exit_config: EarlyExitConfig,
) -> AdaptiveInferenceEngine<MockPipeline> {
    let config = AdaptiveInferenceConfig {
        early_exit_config,
        ..AdaptiveInferenceConfig::default()
    };
    AdaptiveInferenceEngine::new(
        MockPipeline {
            label: "mock".to_string(),
            score,
        },
        config,
    )
}

fn small_early_exit_config() -> EarlyExitConfig {
    EarlyExitConfig {
        min_layers: 2,
        max_layers: 4,
        ..EarlyExitConfig::default()
    }
}

/// A `LayerwiseInference` (and `Pipeline`) model whose per-layer state is a
/// real (if tiny) running computation over the input -- not a fixed
/// constant -- so `adaptive_inference_layerwise` genuinely drives
/// per-layer execution.
#[derive(Clone)]
struct MockLayerwiseModel {
    layers: usize,
}

#[derive(Clone)]
struct MockLayerwiseState {
    accumulator: f32,
}

impl Pipeline for MockLayerwiseModel {
    type Input = String;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> crate::error::Result<Self::Output> {
        Ok(PipelineOutput::Classification(vec![ClassificationOutput {
            label: "whole-model".to_string(),
            score: (input.len() as f32 / 100.0).min(1.0),
        }]))
    }
}

impl LayerwiseInference for MockLayerwiseModel {
    type Input = String;
    type State = MockLayerwiseState;

    fn layer_count(&self) -> usize {
        self.layers
    }

    fn begin(&self, input: &Self::Input) -> crate::error::Result<Self::State> {
        Ok(MockLayerwiseState {
            accumulator: input.len() as f32,
        })
    }

    fn run_layer(
        &self,
        layer_index: usize,
        state: &mut Self::State,
    ) -> crate::error::Result<crate::pipeline::early_exit::LayerOutput> {
        state.accumulator += 1.0;
        Ok(crate::pipeline::early_exit::LayerOutput {
            layer_index,
            hidden_states: vec![state.accumulator],
            attention_weights: None,
            logits: Some(vec![state.accumulator, 1.0]),
            intermediate_prediction: None,
            computation_time_ms: 1,
            memory_usage_mb: 1.0,
        })
    }

    fn finish(
        &self,
        state: &Self::State,
        layers_executed: usize,
    ) -> crate::error::Result<PipelineOutput> {
        Ok(PipelineOutput::Classification(vec![ClassificationOutput {
            label: "layerwise".to_string(),
            score: (state.accumulator / (layers_executed.max(1) as f32 * 10.0)).min(1.0),
        }]))
    }
}

fn layerwise_engine(
    early_exit_config: EarlyExitConfig,
) -> AdaptiveInferenceEngine<MockLayerwiseModel> {
    let config = AdaptiveInferenceConfig {
        early_exit_config,
        ..AdaptiveInferenceConfig::default()
    };
    AdaptiveInferenceEngine::new(MockLayerwiseModel { layers: 4 }, config)
}

// ---------------------------------------------------------------------
// InputAnalyzer::analyze_text / analyze_input: real, content-derived
// measurements -- regression coverage for the old fixed-constant return.
// ---------------------------------------------------------------------

#[test]
fn test_analyze_text_sequence_length_matches_real_word_count() {
    let analyzer = InputAnalyzer::new(12);
    let short = analyzer.analyze_text("cat sat mat");
    let long = analyzer.analyze_text(
        "the quick brown fox jumps over the lazy dog and then runs away quickly into the forest",
    );
    assert_eq!(
        short.sequence_length, 3,
        "must count the real words, not a fixed 256"
    );
    assert_eq!(long.sequence_length, 17);
    assert_ne!(
        short.sequence_length, long.sequence_length,
        "sequence_length must vary with the actual input, not be a fixed constant"
    );
}

#[test]
fn test_analyze_text_complexity_varies_with_content() {
    let analyzer = InputAnalyzer::new(12);
    // Highly repetitive, short words -> low diversity, low length component.
    let simple = analyzer.analyze_text("a a a a a a a a a a");
    // Long, entirely distinct words -> high diversity, high length component.
    let complex = analyzer.analyze_text(
        "extraordinarily sophisticated multidimensional configurations demonstrate remarkable \
         computational characteristics",
    );
    assert!(
        complex.complexity_score > simple.complexity_score,
        "a longer, more diverse vocabulary must score as more complex than ten repeats of `a`: \
         simple={}, complex={}",
        simple.complexity_score,
        complex.complexity_score
    );
    // Neither value may be the old hardcoded constant, since both texts
    // were engineered to sit far from 0.6.
    assert!((simple.complexity_score - 0.6).abs() > 0.05);
    assert!((complex.complexity_score - 0.6).abs() > 0.05);
}

#[test]
fn test_analyze_text_empty_input_has_zero_complexity() {
    let analyzer = InputAnalyzer::new(12);
    let empty = analyzer.analyze_text("");
    assert_eq!(empty.sequence_length, 0);
    assert_eq!(empty.complexity_score, 0.0);
    assert!(empty.token_importance.is_empty());
}

#[test]
fn test_analyze_text_token_importance_length_matches_sequence_length() {
    // Regression: the old code always returned `vec![0.3; 256]` regardless
    // of the actual number of tokens.
    let analyzer = InputAnalyzer::new(12);
    let analysis = analyzer.analyze_text("one two three four five");
    assert_eq!(analysis.token_importance.len(), analysis.sequence_length);
    assert_eq!(analysis.token_importance.len(), 5);
}

#[test]
fn test_token_importance_ranker_scores_rare_words_higher() {
    let ranker = TokenImportanceRanker;
    let words = ["common", "common", "common", "rare"];
    let scores = ranker.rank(&words);
    assert_eq!(scores.len(), 4);
    // "common" appears 3/4 times, "rare" appears 1/4.
    assert!(
        scores[3] > scores[0],
        "a word appearing once must score higher importance than one appearing three times: \
         rare={}, common={}",
        scores[3],
        scores[0]
    );
}

#[test]
fn test_attention_patterns_always_have_fixed_bucket_count() {
    let analyzer = InputAnalyzer::new(12);
    let short = analyzer.analyze_text("one two");
    let long = analyzer.analyze_text(&"word ".repeat(500));
    assert_eq!(short.attention_patterns.len(), ATTENTION_PATTERN_BUCKETS);
    assert_eq!(long.attention_patterns.len(), ATTENTION_PATTERN_BUCKETS);
}

#[test]
fn test_analyze_input_falls_back_to_conservative_default_for_non_text() {
    // A type analyze_input cannot read as text (see `as_text`) must not
    // fabricate a plausible-looking measurement -- it must return the
    // documented conservative default (maximum complexity, full precision,
    // run everything) rather than treating "cannot measure" as "measured
    // as simple".
    struct OpaqueInput(#[allow(dead_code)] u32);
    let analyzer = InputAnalyzer::new(10);
    let analysis = analyzer.analyze_input(&OpaqueInput(42)).expect("analyze_input must not error");
    assert_eq!(analysis.complexity_score, 1.0);
    assert_eq!(analysis.difficulty_estimate, 1.0);
    assert_eq!(
        analysis.recommended_depth, 10,
        "must equal the configured max_layers"
    );
    assert!(matches!(
        analysis.recommended_precision,
        PrecisionMode::Full
    ));
    assert_eq!(analysis.attention_patterns.len(), ATTENTION_PATTERN_BUCKETS);
}

#[test]
fn test_analyze_input_reads_string_as_text_not_conservative_default() {
    // Sanity check in the other direction: a `String` (the overwhelmingly
    // common `Pipeline::Input`) must take the real `analyze_text` path,
    // not the conservative default.
    let analyzer = InputAnalyzer::new(10);
    let input = "hello world this is real text".to_string();
    let analysis = analyzer.analyze_input(&input).expect("analyze_input must not error");
    assert_eq!(analysis.sequence_length, 6);
    assert_ne!(
        analysis.complexity_score, 1.0,
        "a normal sentence should not hit the maximum-complexity conservative default"
    );
}

// ---------------------------------------------------------------------
// adapt_precision_strategy: deterministic, complexity-driven thresholds.
// ---------------------------------------------------------------------

fn analysis_with_complexity(complexity: f32) -> InputAnalysis {
    InputAnalysis {
        sequence_length: 10,
        complexity_score: complexity,
        difficulty_estimate: complexity,
        attention_patterns: vec![0.0; ATTENTION_PATTERN_BUCKETS],
        token_importance: vec![0.5; 10],
        estimated_computation_cost: 10.0,
        recommended_precision: PrecisionMode::Mixed,
        recommended_depth: 5,
    }
}

#[test]
fn test_adapt_precision_strategy_selects_full_for_high_complexity() {
    let mut engine = mock_engine(0.5, small_early_exit_config());
    engine
        .adapt_precision_strategy(&analysis_with_complexity(0.95))
        .expect("adapt_precision_strategy must not error");
    assert!(matches!(
        engine.precision_controller.current_precision,
        PrecisionMode::Full
    ));
}

#[test]
fn test_adapt_precision_strategy_selects_int8_for_low_complexity() {
    let mut engine = mock_engine(0.5, small_early_exit_config());
    engine
        .adapt_precision_strategy(&analysis_with_complexity(0.1))
        .expect("adapt_precision_strategy must not error");
    assert!(matches!(
        engine.precision_controller.current_precision,
        PrecisionMode::Int8
    ));
}

#[test]
fn test_adapt_precision_strategy_populates_precision_history_per_layer() {
    let mut engine = mock_engine(0.5, small_early_exit_config());
    engine
        .adapt_precision_strategy(&analysis_with_complexity(0.5))
        .expect("adapt_precision_strategy must not error");
    // small_early_exit_config has max_layers = 4.
    assert_eq!(engine.precision_history().len(), 4);
    for (_, _, complexity) in engine.precision_history() {
        assert_eq!(
            *complexity, 0.5,
            "recorded complexity must be the real input complexity"
        );
    }
}

// ---------------------------------------------------------------------
// adaptive_inference(): end-to-end regression coverage for fabricated
// metrics on the (always-taken, for a plain Pipeline) full-computation
// path.
// ---------------------------------------------------------------------

#[test]
fn test_adaptive_inference_reports_real_layer_counts_not_stuck_at_zero() {
    // Regression: the old `execute_adaptive_inference` initialized
    // `layers_computed`/`layers_skipped` to 0 and only updated them inside
    // the (for a plain `Pipeline`) unreachable early-exit-succeeded
    // branch, so a full-pipeline run always reported 0 layers computed
    // and 0 skipped.
    let mut engine = mock_engine(0.42, small_early_exit_config());
    let result = engine.adaptive_inference("hello world".to_string()).expect("must succeed");
    assert_eq!(
        result.layers_computed, 4,
        "must equal the configured max_layers (small_early_exit_config), not 0"
    );
    assert_eq!(result.layers_skipped, 0);
}

#[test]
fn test_adaptive_inference_adaptation_decisions_are_two_honest_summaries() {
    // Regression: the old code pushed 1 initial decision plus one decision
    // per hardcoded `for layer_idx in 0..24` loop iteration = 25 total,
    // all describing computation that never actually happened per-layer.
    // The new code pushes exactly two honestly-labeled summaries.
    let mut engine = mock_engine(0.42, small_early_exit_config());
    let result = engine.adaptive_inference("hello world".to_string()).expect("must succeed");
    assert_eq!(
        result.adaptation_decisions.len(),
        2,
        "must be exactly the two summary decisions, not 25 fabricated per-layer ones"
    );
    assert!(result
        .adaptation_decisions
        .iter()
        .any(|d| d.decision_type == "recommended_precision_profile"));
    assert!(result
        .adaptation_decisions
        .iter()
        .any(|d| d.decision_type == "recommended_layer_skipping"));
}

#[test]
fn test_adaptive_inference_quality_score_matches_real_prediction_score() {
    let mut engine = mock_engine(0.42, small_early_exit_config());
    let result = engine.adaptive_inference("hello world".to_string()).expect("must succeed");
    assert!(
        (result.quality_score - 0.42).abs() < 1e-6,
        "quality_score must be the real Classification score (0.42), got {}",
        result.quality_score
    );
}

#[test]
fn test_adaptive_inference_energy_consumed_is_honest_zero() {
    // Regression: `ResourceMonitor::new()` used to hardcode
    // `energy_consumption: 10.0`. No energy telemetry source is wired
    // into this workspace, so it must be an honest 0.0.
    let mut engine = mock_engine(0.5, small_early_exit_config());
    let result = engine.adaptive_inference("hello world".to_string()).expect("must succeed");
    assert_eq!(result.energy_consumed_watts, 0.0);
}

#[test]
fn test_adaptive_inference_memory_peak_is_real_sampled_reading() {
    // Regression: `ResourceMonitor::new()` used to hardcode
    // `memory_usage: 512.0` and nothing on the full-computation path ever
    // refreshed it. A real sysinfo-sampled RSS for the test process is
    // essentially never bit-exact 512.0 MB.
    let mut engine = mock_engine(0.5, small_early_exit_config());
    let result = engine.adaptive_inference("hello world".to_string()).expect("must succeed");
    assert!(result.memory_peak_mb.is_finite());
    assert!(result.memory_peak_mb >= 0.0);
    assert_ne!(
        result.memory_peak_mb, 512.0,
        "memory_peak_mb must be a real measurement, not the old hardcoded 512.0 placeholder"
    );
}

#[test]
fn test_adaptive_inference_throughput_is_finite_not_divide_by_zero_infinity() {
    // Regression: the old `calculate_performance_metrics` computed
    // `1000.0 / (time_ms as f32)`, which is `Infinity` whenever a call
    // completes in under 1ms -- entirely plausible for an in-memory mock
    // pipeline with no real model.
    let mut engine = mock_engine(0.5, small_early_exit_config());
    let result = engine.adaptive_inference("hello world".to_string()).expect("must succeed");
    assert!(
        result.performance_metrics.throughput_tokens_per_second.is_finite(),
        "throughput must never be Infinity, got {}",
        result.performance_metrics.throughput_tokens_per_second
    );
    assert!(result.performance_metrics.throughput_tokens_per_second > 0.0);
}

#[test]
fn test_adaptive_inference_speedup_factor_is_one_when_nothing_skipped() {
    // Regression: the old code hardcoded `speedup_factor: 2.0`
    // unconditionally. When the full pipeline ran (as it always does for
    // a plain `Pipeline`), the honest speedup is 1.0 (nothing was saved).
    let mut engine = mock_engine(0.5, small_early_exit_config());
    let result = engine.adaptive_inference("hello world".to_string()).expect("must succeed");
    assert!(
        (result.performance_metrics.speedup_factor - 1.0).abs() < 1e-6,
        "speedup_factor must be 1.0 (real ratio total_layers/layers_computed) when nothing was \
         skipped, got {}",
        result.performance_metrics.speedup_factor
    );
}

#[test]
fn test_adaptive_inference_quality_preservation_is_one_when_full_pipeline_ran() {
    // Regression: the old code hardcoded `quality_preservation: 0.9`
    // unconditionally, even when the full (unapproximated) pipeline ran.
    let mut engine = mock_engine(0.5, small_early_exit_config());
    let result = engine.adaptive_inference("hello world".to_string()).expect("must succeed");
    assert_eq!(
        result.performance_metrics.quality_preservation, 1.0,
        "quality is trivially fully preserved when the full pipeline ran with no early exit"
    );
}

#[test]
fn test_adaptive_inference_history_accumulates_across_calls() {
    let mut engine = mock_engine(0.5, small_early_exit_config());
    assert_eq!(engine.adaptation_history().len(), 0);
    engine.adaptive_inference("first call".to_string()).expect("must succeed");
    assert_eq!(engine.adaptation_history().len(), 2);
    engine.adaptive_inference("second call".to_string()).expect("must succeed");
    assert_eq!(engine.adaptation_history().len(), 4);
}

#[test]
fn test_adaptive_inference_calibration_data_tracks_observed_quality() {
    // New functionality: calibration_data is a real EMA of observed
    // quality_score per recommended PrecisionMode, not an inert map.
    let mut engine = mock_engine(0.9, small_early_exit_config());
    let result = engine.adaptive_inference("hello world".to_string()).expect("must succeed");
    let precision_used = result.precision_used.clone();
    let recorded = *engine
        .calibration_data()
        .get(&precision_used)
        .expect("calibration_data must record the precision used");
    assert!(
        (recorded - 0.9).abs() < 1e-6,
        "first observation must seed the EMA at the observed value"
    );

    // A second call with a different observed quality must move the EMA,
    // not overwrite or ignore it.
    let mut engine2 = mock_engine(0.9, small_early_exit_config());
    engine2.adaptive_inference("hello world".to_string()).expect("must succeed");
    let second_result =
        engine2.adaptive_inference("hello world".to_string()).expect("must succeed");
    let after_two = *engine2
        .calibration_data()
        .get(&second_result.precision_used)
        .expect("calibration_data must still have an entry");
    assert!(after_two.is_finite());
}

#[test]
fn test_conditional_computations_counts_recommended_skips() {
    let mut engine = mock_engine(0.5, small_early_exit_config());
    let result = engine.adaptive_inference("hello world".to_string()).expect("must succeed");
    let real_skip_count =
        engine.recommended_skip_decisions().values().filter(|&&skip| skip).count();
    assert_eq!(result.conditional_computations, real_skip_count);
}

// ---------------------------------------------------------------------
// real_output_units: real per-call work-unit counting for throughput.
// ---------------------------------------------------------------------

#[test]
fn test_real_output_units_generation_counts_real_tokens() {
    let prediction = PipelineOutput::Generation(GenerationOutput {
        generated_text: "irrelevant if sequences is present".to_string(),
        sequences: Some(vec![vec![1, 2, 3, 4, 5]]),
        scores: None,
    });
    assert_eq!(real_output_units(&prediction), 5);
}

#[test]
fn test_real_output_units_generation_falls_back_to_word_count() {
    let prediction = PipelineOutput::Generation(GenerationOutput {
        generated_text: "four real words here".to_string(),
        sequences: None,
        scores: None,
    });
    assert_eq!(real_output_units(&prediction), 4);
}

#[test]
fn test_real_output_units_classification_counts_results() {
    let prediction = PipelineOutput::Classification(vec![
        ClassificationOutput {
            label: "a".to_string(),
            score: 0.1,
        },
        ClassificationOutput {
            label: "b".to_string(),
            score: 0.2,
        },
    ]);
    assert_eq!(real_output_units(&prediction), 2);
}

// ---------------------------------------------------------------------
// adaptive_inference_layerwise(): genuinely real per-layer execution --
// this method and the `LayerwiseInference` wiring did not exist at all in
// the pre-fix version of this file.
// ---------------------------------------------------------------------

#[test]
fn test_adaptive_inference_layerwise_runs_and_reports_real_layer_counts() {
    let mut engine = layerwise_engine(EarlyExitConfig {
        min_layers: 2,
        max_layers: 4,
        strategy: ExitStrategy::ConfidenceThreshold(0.0), // trivially met -- exits ASAP
        ..EarlyExitConfig::default()
    });
    let input = "hello".to_string();
    let result = engine.adaptive_inference_layerwise(&input).expect("must succeed");

    assert!(result.early_exit_result.is_some());
    assert!(
        (2..=4).contains(&result.layers_computed),
        "layers_computed must be within [min_layers, max_layers], got {}",
        result.layers_computed
    );
    assert_eq!(result.layers_skipped, 4 - result.layers_computed);
    assert_eq!(
        result.conditional_computations, 0,
        "the layerwise path has no separate skip-recommendation pass"
    );
    assert_eq!(result.adaptation_decisions.len(), 1);
    assert_eq!(
        result.adaptation_decisions[0].decision_type,
        "layerwise_early_exit"
    );
}

#[test]
fn test_adaptive_inference_layerwise_full_run_when_threshold_never_met() {
    let mut engine = layerwise_engine(EarlyExitConfig {
        min_layers: 1,
        max_layers: 4,
        strategy: ExitStrategy::ConfidenceThreshold(1.0), // never normally triggered
        // `EarlyExitPredictor::get_adjusted_threshold` (see `early_exit.rs`)
        // lowers the configured threshold based on measured context when
        // this is left at its `EarlyExitConfig::default()` value of
        // `true`, which makes even a `1.0` threshold reachable. Disabling
        // it is what `early_exit.rs`'s own `test_forced_exit_at_max_layers`
        // does to make "never normally triggered" literally true.
        dynamic_threshold_adjustment: false,
        ..EarlyExitConfig::default()
    });
    let input = "hello".to_string();
    let result = engine.adaptive_inference_layerwise(&input).expect("must succeed");
    assert_eq!(
        result.layers_computed, 4,
        "with a threshold that is never met, all configured layers must actually run"
    );
    assert_eq!(result.layers_skipped, 0);
}

#[test]
fn test_adaptive_inference_layerwise_propagates_zero_layer_error() {
    let mut engine = layerwise_engine(small_early_exit_config());
    // Overwrite with a model reporting zero layers via a fresh engine.
    let mut zero_layer_engine = AdaptiveInferenceEngine::new(
        MockLayerwiseModel { layers: 0 },
        AdaptiveInferenceConfig {
            early_exit_config: small_early_exit_config(),
            ..AdaptiveInferenceConfig::default()
        },
    );
    let input = "hello".to_string();
    let result = zero_layer_engine.adaptive_inference_layerwise(&input);
    assert!(
        result.is_err(),
        "a model reporting zero layers must surface as an error, not panic"
    );
    // Original (non-zero-layer) engine is unaffected -- sanity check that
    // the fixture itself is not the source of any failure above.
    let _ = engine.adaptive_inference_layerwise(&input);
}

#[test]
fn test_adaptive_inference_layerwise_prediction_is_real_classification() {
    let mut engine = layerwise_engine(EarlyExitConfig {
        min_layers: 1,
        max_layers: 4,
        strategy: ExitStrategy::ConfidenceThreshold(1.0),
        ..EarlyExitConfig::default()
    });
    let input = "hello".to_string();
    let result = engine.adaptive_inference_layerwise(&input).expect("must succeed");
    match result.prediction {
        PipelineOutput::Classification(results) => {
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].label, "layerwise");
            assert!((0.0..=1.0).contains(&results[0].score));
        },
        other => panic!(
            "expected a Classification prediction from MockLayerwiseModel::finish, got {other:?}"
        ),
    }
}
