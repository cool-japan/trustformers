use super::*;

// -----------------------------------------------------------------------
// Regression: the pipeline used to embed every token as a constant 0.1
// vector and to "execute" layers by adding 0.01 * layer_index. Routing over
// identical vectors carries no information, yet efficiency scores were
// still reported. Both tests below fail against that implementation.
// -----------------------------------------------------------------------

/// A real embedder over a tiny hashing feature space.
struct HashingEmbedder {
    dim: usize,
}

impl TokenEmbedder for HashingEmbedder {
    fn embed(&self, inputs: &[String]) -> TrustformersResult<Vec<Vec<f32>>> {
        Ok(inputs
            .iter()
            .map(|text| {
                let mut features = vec![0.0f32; self.dim];
                for (position, byte) in text.bytes().enumerate() {
                    let slot = (byte as usize + position) % self.dim;
                    features[slot] += 1.0;
                }
                let norm = features.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
                features.iter().map(|v| v / norm).collect()
            })
            .collect())
    }

    fn embedding_dim(&self) -> usize {
        self.dim
    }
}

/// A real (if simple) layer: a per-layer scaled non-linearity that honours
/// the routing mask.
struct ScalingLayerExecutor {
    layers: usize,
}

impl TransformerLayerExecutor for ScalingLayerExecutor {
    fn num_layers(&self) -> usize {
        self.layers
    }

    fn execute(
        &self,
        layer_index: usize,
        inputs: &[Vec<f32>],
        token_mask: Option<&[bool]>,
    ) -> TrustformersResult<Vec<Vec<f32>>> {
        if layer_index >= self.layers {
            return Err(TrustformersError::invalid_input_simple(format!(
                "layer {layer_index} is out of range for a {}-layer model",
                self.layers
            )));
        }
        Ok(inputs
            .iter()
            .enumerate()
            .map(|(token_index, row)| {
                let selected = token_mask.map(|m| *m.get(token_index).unwrap_or(&true));
                if selected == Some(false) {
                    return row.clone();
                }
                row.iter().map(|v| (v * (1.0 + layer_index as f32 * 0.1)).tanh()).collect()
            })
            .collect())
    }
}

/// A layer executor that counts real invocations, so a test can prove
/// the cache actually avoids calling it again for a cached
/// `(layer, input, mask)` combination rather than merely returning a
/// plausible-looking result.
struct CountingLayerExecutor {
    inner: ScalingLayerExecutor,
    calls: std::sync::atomic::AtomicUsize,
}

impl CountingLayerExecutor {
    fn new(layers: usize) -> Self {
        Self {
            inner: ScalingLayerExecutor { layers },
            calls: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.load(std::sync::atomic::Ordering::SeqCst)
    }
}

impl TransformerLayerExecutor for CountingLayerExecutor {
    fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    fn execute(
        &self,
        layer_index: usize,
        inputs: &[Vec<f32>],
        token_mask: Option<&[bool]>,
    ) -> TrustformersResult<Vec<Vec<f32>>> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        self.inner.execute(layer_index, inputs, token_mask)
    }
}

fn arbitrary_decision(layer_index: usize, token_routing: Vec<bool>) -> RoutingDecision {
    RoutingDecision {
        layer_index,
        should_execute: true,
        confidence_score: 0.5,
        complexity_score: 0.5,
        routing_reason: RoutingReason::ConfidenceThreshold,
        token_routing,
    }
}

// -----------------------------------------------------------------------
// Regression: `layer_cache` had a real read path (`cached_layer_count`)
// but nothing ever wrote to it -- every call to `execute_layer`
// recomputed from scratch even for an identical `(layer, input, mask)`.
// The four tests below fail against that implementation.
// -----------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn execute_layer_caches_identical_input() {
    let executor = Arc::new(CountingLayerExecutor::new(4));
    let pipeline = MixtureOfDepthsPipeline::new(
        MixtureOfDepthsConfig::default(),
        Arc::new(MockBaseModel),
        Arc::new(MockComplexityAnalyzer),
        Arc::new(MockConfidenceEstimator),
        Arc::new(MockDepthRouter),
    )
    .with_layer_executor(executor.clone());

    let decision = arbitrary_decision(1, Vec::new());
    let inputs = vec![vec![0.5f32, -0.25, 0.75, 0.0]];

    let first = pipeline
        .execute_layer(1, &inputs, &decision, None)
        .await
        .expect("first execution should succeed");
    let second = pipeline
        .execute_layer(1, &inputs, &decision, None)
        .await
        .expect("second execution should succeed");

    assert_eq!(
        executor.call_count(),
        1,
        "an identical (layer, input, mask) call must be served from the cache, not \
             recomputed"
    );
    assert_eq!(
        first.token_outputs, second.token_outputs,
        "a cache hit must return the same output as the original computation"
    );
    assert_eq!(
        pipeline.cached_layer_count().await,
        1,
        "the cache must hold the one distinct (layer, input) key seen so far"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn execute_layer_misses_on_different_input() {
    let executor = Arc::new(CountingLayerExecutor::new(4));
    let pipeline = MixtureOfDepthsPipeline::new(
        MixtureOfDepthsConfig::default(),
        Arc::new(MockBaseModel),
        Arc::new(MockComplexityAnalyzer),
        Arc::new(MockConfidenceEstimator),
        Arc::new(MockDepthRouter),
    )
    .with_layer_executor(executor.clone());

    let decision = arbitrary_decision(1, Vec::new());

    pipeline
        .execute_layer(1, &[vec![0.1, 0.2]], &decision, None)
        .await
        .expect("first execution should succeed");
    pipeline
        .execute_layer(1, &[vec![0.9, 0.4]], &decision, None)
        .await
        .expect("second execution should succeed");

    assert_eq!(
        executor.call_count(),
        2,
        "distinct inputs must not share a cache entry"
    );
    assert_eq!(
        pipeline.cached_layer_count().await,
        2,
        "two distinct inputs should produce two cache entries"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn execute_layer_cache_key_includes_the_routing_mask() {
    // Same layer, same input tensor, but a different routing mask must
    // NOT hit the same cache entry: the mask changes what the executor
    // actually does to the input (a masked-out token passes through
    // unchanged; a selected one does not), so a mask-blind cache key
    // would silently serve a wrong result for the second call.
    let executor = Arc::new(CountingLayerExecutor::new(4));
    let config = MixtureOfDepthsConfig {
        token_level_routing: true,
        ..Default::default()
    };
    let pipeline = MixtureOfDepthsPipeline::new(
        config,
        Arc::new(MockBaseModel),
        Arc::new(MockComplexityAnalyzer),
        Arc::new(MockConfidenceEstimator),
        Arc::new(MockDepthRouter),
    )
    .with_layer_executor(executor.clone());

    let inputs = vec![vec![0.5f32, -0.25], vec![0.1, 0.2]];
    let all_selected = arbitrary_decision(1, vec![true, true]);
    let partially_masked = arbitrary_decision(1, vec![true, false]);

    let first = pipeline
        .execute_layer(1, &inputs, &all_selected, None)
        .await
        .expect("first execution should succeed");
    let second = pipeline
        .execute_layer(1, &inputs, &partially_masked, None)
        .await
        .expect("second execution should succeed");

    assert_eq!(
        executor.call_count(),
        2,
        "a different routing mask over the same input tensor must miss the cache"
    );
    assert_ne!(
        first.token_outputs[1], second.token_outputs[1],
        "the second token, masked out only in the second call, must differ between \
             the two real (uncached) results"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn execute_layer_cache_respects_capacity_bound() {
    let executor = Arc::new(CountingLayerExecutor::new(1));
    let pipeline = MixtureOfDepthsPipeline::new(
        MixtureOfDepthsConfig::default(),
        Arc::new(MockBaseModel),
        Arc::new(MockComplexityAnalyzer),
        Arc::new(MockConfidenceEstimator),
        Arc::new(MockDepthRouter),
    )
    .with_layer_executor(executor.clone())
    .with_layer_cache_capacity(2);

    let decision = arbitrary_decision(0, Vec::new());

    for value in [0.1f32, 0.2, 0.3, 0.4] {
        pipeline
            .execute_layer(0, &[vec![value]], &decision, None)
            .await
            .expect("execution should succeed");
    }

    assert_eq!(
        pipeline.cached_layer_count().await,
        2,
        "the cache must never grow past its configured capacity"
    );

    // The least-recently-used entries (0.1, 0.2) were evicted to make
    // room for 0.3 and 0.4; replaying 0.1 must recompute rather than
    // phantom-hit an evicted slot.
    let calls_before = executor.call_count();
    pipeline
        .execute_layer(0, &[vec![0.1f32]], &decision, None)
        .await
        .expect("execution should succeed");
    assert_eq!(
        executor.call_count(),
        calls_before + 1,
        "an evicted entry must be recomputed, not phantom-hit"
    );
}

/// A layer executor that sleeps for a small, fixed duration before
/// delegating to a [`ScalingLayerExecutor`], so a cache-miss call's
/// measured `computation_cost` is reliably, unambiguously nonzero --
/// unlike the featherweight scaling executor alone, whose real cost can
/// round to `0.0` on a fast host with a coarse timer, which would make a
/// cache-hit's `computation_cost == 0.0` assertion indistinguishable
/// from measurement noise rather than proof the fix does something.
struct SleepingLayerExecutor {
    inner: ScalingLayerExecutor,
}

impl TransformerLayerExecutor for SleepingLayerExecutor {
    fn num_layers(&self) -> usize {
        self.inner.num_layers()
    }

    fn execute(
        &self,
        layer_index: usize,
        inputs: &[Vec<f32>],
        token_mask: Option<&[bool]>,
    ) -> TrustformersResult<Vec<Vec<f32>>> {
        std::thread::sleep(std::time::Duration::from_millis(5));
        self.inner.execute(layer_index, inputs, token_mask)
    }
}

// -----------------------------------------------------------------------
// Regression: a cache hit used to return the stored `LayerExecutionResult`
// verbatim, including the `computation_cost` measured on the ORIGINAL
// (miss) call -- so a replayed layer that performed no computation this
// call was still charged its original cost. `execute_with_mod` sums
// `computation_cost` into `total_computation_cost`, which drives its
// compute-budget cutoff and efficiency score. The two tests below fail
// against that implementation.
// -----------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn execute_layer_cache_hit_reports_zero_computation_cost() {
    let executor = Arc::new(SleepingLayerExecutor {
        inner: ScalingLayerExecutor { layers: 4 },
    });
    let pipeline = MixtureOfDepthsPipeline::new(
        MixtureOfDepthsConfig::default(),
        Arc::new(MockBaseModel),
        Arc::new(MockComplexityAnalyzer),
        Arc::new(MockConfidenceEstimator),
        Arc::new(MockDepthRouter),
    )
    .with_layer_executor(executor);

    let decision = arbitrary_decision(1, Vec::new());
    let inputs = vec![vec![0.5f32, -0.25, 0.75, 0.0]];

    let first = pipeline
        .execute_layer(1, &inputs, &decision, None)
        .await
        .expect("first (miss) execution should succeed");
    assert!(
        first.computation_cost > 0.0,
        "the real (miss) call slept for 5ms, so its measured cost must be \
             positive; this is the baseline the cache-hit assertion below is \
             contrasted against"
    );

    let second = pipeline
        .execute_layer(1, &inputs, &decision, None)
        .await
        .expect("second (cached) execution should succeed");
    assert_eq!(
        second.computation_cost, 0.0,
        "a cache hit performs no computation, so its honest cost is 0.0 -- \
             not the original call's ~5ms carried over verbatim, which is what \
             the pre-fix code did"
    );
    assert_eq!(
        second.token_outputs, first.token_outputs,
        "the cache hit's outputs must still match the original computation"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn execute_with_mod_does_not_double_charge_replayed_layers() {
    // Integration-level counterpart to the test above: proves the fix
    // actually reaches `MoDExecutionResult::total_computation_cost`,
    // not just `execute_layer`'s own return value in isolation.
    let executor = Arc::new(SleepingLayerExecutor {
        inner: ScalingLayerExecutor { layers: 6 },
    });
    let config = MixtureOfDepthsConfig {
        total_layers: 6,
        min_layers: 6,
        // Large enough that the (real, ~30ms) first call never trips the
        // compute-budget early-exit; isolates this test to the
        // cost-accounting question, not the cutoff logic.
        compute_budget: 10_000.0,
        ..Default::default()
    };
    let pipeline = MixtureOfDepthsPipeline::new(
        config,
        Arc::new(MockBaseModel),
        Arc::new(MockComplexityAnalyzer),
        Arc::new(MockConfidenceEstimator),
        Arc::new(MockDepthRouter),
    )
    .with_embedder(Arc::new(HashingEmbedder { dim: 16 }))
    .with_layer_executor(executor);

    let input = vec!["climate".to_string(), "change".to_string()];

    let first = pipeline.execute_with_mod(&input).await.expect("first run should succeed");
    assert!(
        !first.executed_layers.is_empty(),
        "min_layers == total_layers == 6 guarantees every layer executes \
             on the first (cold-cache) run"
    );
    assert!(
        first.total_computation_cost > 0.0,
        "6 real layers each sleeping 5ms must sum to a measurably positive \
             total on the cold-cache run"
    );

    let second = pipeline.execute_with_mod(&input).await.expect("second run should succeed");
    assert_eq!(
        second.executed_layers, first.executed_layers,
        "identical input against the same (deterministic) mocks must route \
             to the same layers both times"
    );
    assert_eq!(
        second.total_computation_cost, 0.0,
        "every layer on the second run is served from cache, so the honest \
             total is 0.0 -- the pre-fix code would have reported (approximately) \
             the same total as the first run, double-charging every replayed layer"
    );
}

/// Attach the real test embedder and layer executor to `pipeline`.
fn wire_real_components(pipeline: MixtureOfDepthsPipeline) -> MixtureOfDepthsPipeline {
    let layers = pipeline.config.total_layers.max(1);
    pipeline
        .with_embedder(Arc::new(HashingEmbedder { dim: 16 }))
        .with_layer_executor(Arc::new(ScalingLayerExecutor { layers }))
}

#[tokio::test]
async fn embeddings_without_an_embedder_are_refused() {
    let pipeline = MixtureOfDepthsPipeline::new(
        MixtureOfDepthsConfig::default(),
        Arc::new(MockBaseModel),
        Arc::new(MockComplexityAnalyzer),
        Arc::new(MockConfidenceEstimator),
        Arc::new(MockDepthRouter),
    );
    let result = pipeline.initialize_embeddings(&["hello".to_string()]).await;
    assert!(
        result.is_err(),
        "routing needs real embeddings; a constant vector must not be substituted"
    );
}

#[tokio::test]
async fn real_embeddings_differ_between_inputs() {
    let embedder = HashingEmbedder { dim: 16 };
    let pipeline = MixtureOfDepthsPipeline::new(
        MixtureOfDepthsConfig::default(),
        Arc::new(MockBaseModel),
        Arc::new(MockComplexityAnalyzer),
        Arc::new(MockConfidenceEstimator),
        Arc::new(MockDepthRouter),
    )
    .with_embedder(Arc::new(embedder));

    let embeddings = pipeline
        .initialize_embeddings(&["alpha".to_string(), "omega".to_string()])
        .await
        .expect("a real embedder should answer");
    assert_eq!(embeddings.len(), 2);
    assert!(
        embeddings[0]
            .iter()
            .zip(embeddings[1].iter())
            .any(|(a, b)| (a - b).abs() > 1e-6),
        "distinct inputs must produce distinct embeddings"
    );
    assert!(
        embeddings[0].iter().any(|v| (v - 0.1).abs() > 1e-6),
        "the constant-0.1 embedding is gone"
    );
}

#[tokio::test]
async fn layer_execution_requires_an_executor() {
    let pipeline = MixtureOfDepthsPipeline::new(
        MixtureOfDepthsConfig::default(),
        Arc::new(MockBaseModel),
        Arc::new(MockComplexityAnalyzer),
        Arc::new(MockConfidenceEstimator),
        Arc::new(MockDepthRouter),
    );
    let decision = RoutingDecision {
        layer_index: 0,
        should_execute: true,
        confidence_score: 0.5,
        complexity_score: 0.5,
        routing_reason: RoutingReason::ConfidenceThreshold,
        token_routing: Vec::new(),
    };
    let inputs = vec![vec![0.5f32; 4]];
    assert!(
        pipeline.execute_layer(0, &inputs, &decision, None).await.is_err(),
        "without an executor there is no layer to run"
    );
}

#[tokio::test]
async fn executed_layer_transforms_its_input() {
    let pipeline = MixtureOfDepthsPipeline::new(
        MixtureOfDepthsConfig::default(),
        Arc::new(MockBaseModel),
        Arc::new(MockComplexityAnalyzer),
        Arc::new(MockConfidenceEstimator),
        Arc::new(MockDepthRouter),
    )
    .with_layer_executor(Arc::new(ScalingLayerExecutor { layers: 4 }));

    let decision = RoutingDecision {
        layer_index: 1,
        should_execute: true,
        confidence_score: 0.5,
        complexity_score: 0.5,
        routing_reason: RoutingReason::ConfidenceThreshold,
        token_routing: Vec::new(),
    };
    let inputs = vec![vec![0.5f32, -0.25, 0.75, 0.0]];
    let result = pipeline
        .execute_layer(1, &inputs, &decision, None)
        .await
        .expect("a real executor should answer");

    assert!(result.was_executed);
    assert!(
        result
            .token_outputs
            .iter()
            .zip(inputs.iter())
            .any(|(o, i)| o.iter().zip(i.iter()).any(|(a, b)| (a - b).abs() > 1e-6)),
        "the layer must actually transform its input"
    );
    assert!(
        result.attention_weights.is_none(),
        "uniform stand-in attention is gone"
    );
    assert!(result.computation_cost >= 0.0);
}

// Mock base model for testing
struct MockBaseModel;

impl Pipeline for MockBaseModel {
    type Input = String;
    type Output = PipelineOutput;

    fn __call__(&self, _input: Self::Input) -> TrustformersResult<Self::Output> {
        Ok(PipelineOutput::Text("Mock output".to_string()))
    }
}

#[test]
fn test_base_model_accessor_returns_the_constructed_pipeline() {
    let pipeline = create_mixture_of_depths_pipeline(
        MixtureOfDepthsConfig::default(),
        Arc::new(MockBaseModel),
    );
    let output = pipeline
        .base_model()
        .__call__("hello".to_string())
        .expect("the base model must be callable through the accessor");
    assert!(
        matches!(output, PipelineOutput::Text(t) if t == "Mock output"),
        "base_model() must return the same pipeline the constructor was given"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_cached_layer_count_starts_empty() {
    let pipeline = create_mixture_of_depths_pipeline(
        MixtureOfDepthsConfig::default(),
        Arc::new(MockBaseModel),
    );
    assert_eq!(
        pipeline.cached_layer_count().await,
        0,
        "a freshly constructed pipeline has an empty layer cache"
    );
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
    let pipeline = wire_real_components(create_mixture_of_depths_pipeline(config, mock_base_model));
    let input = PipelineInput::Text("skip all layers".to_string());
    let result = pipeline.__call__(input);
    assert!(result.is_ok(), "skipping all layers should not crash");
}

// ── Depth reduction vs accuracy trade-off ─────────────────────────────────

#[test]
fn test_efficiency_score_with_fewer_executed_layers() {
    let config = MixtureOfDepthsConfig::default();
    let mock_base_model = Arc::new(MockBaseModel);
    let pipeline = wire_real_components(create_mixture_of_depths_pipeline(
        config.clone(),
        mock_base_model,
    ));
    // Fewer executed layers → higher depth_efficiency
    let score_few = pipeline.calculate_efficiency_score(&[0, 1], 2.0, 0.9);
    let score_many = pipeline.calculate_efficiency_score(&(0..20).collect::<Vec<_>>(), 20.0, 0.9);
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
    let mod_pipeline =
        wire_real_components(create_mixture_of_depths_pipeline(config, mock_base_model));
    let input = PipelineInput::Text("This is a test sentence for mixture of depths".to_string());
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
    let mod_pipeline =
        wire_real_components(create_mixture_of_depths_pipeline(config, mock_base_model));
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
    let pipeline = wire_real_components(create_mixture_of_depths_pipeline(config, mock_base_model));
    let input = PipelineInput::Text("budget test".to_string());
    let result = pipeline.__call__(input);
    assert!(result.is_ok());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_efficiency_optimized_factory() {
    let mock_base_model = Arc::new(MockBaseModel);
    let pipeline = wire_real_components(create_efficiency_optimized_mod_pipeline(mock_base_model));
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
    let pipeline = wire_real_components(create_quality_focused_mod_pipeline(mock_base_model));
    let input = PipelineInput::Text("quality test".to_string());
    let result = pipeline.__call__(input);
    assert!(result.is_ok(), "quality-focused pipeline should succeed");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_non_text_input_rejected() {
    let config = MixtureOfDepthsConfig::default();
    let mock_base_model = Arc::new(MockBaseModel);
    let pipeline = wire_real_components(create_mixture_of_depths_pipeline(config, mock_base_model));
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
    let pipeline = wire_real_components(create_mixture_of_depths_pipeline(config, mock_base_model));
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
