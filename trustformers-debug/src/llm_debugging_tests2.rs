//! Core unit tests for the llm_debugging module (split out of llm_debugging.rs to keep it under the 2000-line policy limit).

use super::*;

#[tokio::test]
async fn test_llm_debugger_creation() {
    let debugger = llm_debugger();
    assert!(debugger.config.enable_safety_analysis);
}

#[tokio::test]
async fn test_safety_analysis() {
    let mut debugger = llm_debugger();
    let result = debugger
        .analyze_response(
            "How are you?",
            "I'm doing well, thank you for asking!",
            None,
            None,
        )
        .await;

    assert!(result.is_ok());
    let report = result.expect("operation failed in test");
    assert!(report.safety_analysis.is_some());
    assert!(report.overall_score > 0.0);
}

/// Regression test: `SafetyAnalysisResult::flagged_content` used to be
/// the hardcoded empty `vec![]` regardless of what was actually
/// matched, and `confidence` was a flat `0.85` regardless of match
/// strength. A response that trips the keyword heuristic must report
/// the real matched keywords and a confidence that differs from a
/// clean response.
#[tokio::test]
async fn test_safety_analysis_reports_real_flagged_content_and_confidence() {
    let config = LLMDebugConfig::default();
    let mut analyzer = SafetyAnalyzer::new(&config);

    let clean = analyzer
        .analyze_safety("I'm doing well, thank you for asking!")
        .await
        .expect("analysis should succeed");
    assert!(
        clean.flagged_content.is_empty(),
        "a clean response must not report fabricated flagged content"
    );

    let harmful = analyzer
        .analyze_safety("This message contains violence and hate.")
        .await
        .expect("analysis should succeed");
    assert!(
        !harmful.flagged_content.is_empty(),
        "must report the real keywords that were matched, not the old hardcoded empty vec"
    );
    assert!(harmful.flagged_content.contains(&"violence".to_string()));
    assert!(harmful.flagged_content.contains(&"hate".to_string()));
    assert!(
        harmful.detected_harms.contains(&HarmCategory::HateSpeech),
        "the 'hate' keyword must now map to HateSpeech (it was missing from the old mapping)"
    );
    assert_ne!(
        clean.confidence, harmful.confidence,
        "confidence must be a real function of the match, not a flat 0.85 for every response"
    );
}

/// Regression test: `FactualityAnalysisResult::confidence_scores` used
/// to be the hardcoded 3-element `[0.8, 0.7, 0.9]` ("Mock scores")
/// regardless of how many claims were actually found, and
/// `knowledge_gaps` was always empty.
#[tokio::test]
async fn test_factuality_check_reports_real_confidence_scores_and_gaps() {
    let config = LLMDebugConfig::default();
    let mut checker = FactualityChecker::new(&config);

    let response = "This might be true. It is possibly uncertain. Water boils at 100 degrees.";
    let result = checker.check_factuality(response, None).await.expect("check should succeed");

    assert_eq!(
        result.confidence_scores.len(),
        result.claim_like_sentences,
        "must produce exactly one confidence score per identified claim, not a fixed 3-tuple"
    );
    assert_ne!(
        result.confidence_scores,
        vec![0.8, 0.7, 0.9],
        "must not be the old hardcoded 'Mock scores' literal"
    );
    assert!(
        !result.knowledge_gaps.is_empty(),
        "sentences containing uncertainty indicators must be reported as knowledge gaps, \
         not the old hardcoded empty vec"
    );
}

#[tokio::test]
async fn test_batch_analysis() {
    let mut debugger = llm_debugger();
    let interactions = vec![
        ("Hello".to_string(), "Hi there!".to_string()),
        ("How are you?".to_string(), "I'm good!".to_string()),
    ];

    let result = debugger.analyze_batch(&interactions).await;
    assert!(result.is_ok());

    let batch_report = result.expect("operation failed in test");
    assert_eq!(batch_report.batch_size, 2);
    assert_eq!(batch_report.individual_reports.len(), 2);
}

#[tokio::test]
async fn test_health_report_generation() {
    let mut debugger = llm_debugger();
    let health_report = debugger.generate_health_report().await;

    assert!(health_report.is_ok());
    let report = health_report.expect("operation failed in test");
    // Nothing has been analysed yet, so only the safety analyzer's own
    // aggregate (which starts at 1.0) is present; the factuality and alignment
    // aggregates are honestly absent instead of the constants 0.8 / 0.85 they
    // used to be seeded with. The point of the assertion is that the reported
    // score is a real mean of the terms that exist.
    assert_eq!(report.overall_health_score, Some(1.0));
}

#[tokio::test]
async fn test_safety_focused_config() {
    let config = safety_focused_config();
    assert!(config.enable_safety_analysis);
    assert!(config.enable_bias_detection);
    assert!(!config.enable_llm_performance_profiling);
    assert_eq!(config.safety_threshold, 0.9);
}

#[tokio::test]
async fn test_performance_focused_config() {
    let config = performance_focused_config();
    assert!(!config.enable_safety_analysis);
    assert!(config.enable_llm_performance_profiling);
    assert!(config.enable_conversation_analysis);
    assert_eq!(config.analysis_sampling_rate, 0.1);
}

#[test]
fn test_default_llm_debug_config() {
    let config = LLMDebugConfig::default();
    assert!(config.enable_safety_analysis);
    assert!(config.enable_factuality_checking);
    assert!(config.enable_alignment_monitoring);
    assert!(config.enable_hallucination_detection);
    assert!(config.enable_bias_detection);
    assert!(config.enable_llm_performance_profiling);
    assert!(config.enable_conversation_analysis);
    assert!((config.safety_threshold - 0.8).abs() < 1e-9);
    assert!((config.factuality_threshold - 0.7).abs() < 1e-9);
    assert_eq!(config.max_conversation_length, 100);
    assert!((config.analysis_sampling_rate - 1.0).abs() < 1e-9);
}

#[test]
fn test_llm_performance_profiler_new() {
    let profiler = LLMPerformanceProfiler::new();
    assert!(profiler.generation_metrics.tokens_per_second > 0.0);
    assert!(profiler.efficiency_metrics.memory_efficiency > 0.0);
    assert!(profiler.quality_metrics.coherence_score > 0.0);
    assert!(profiler.scalability_metrics.concurrent_user_capacity > 0);
}

#[test]
fn test_llm_performance_profiler_default() {
    let profiler = LLMPerformanceProfiler::default();
    assert!((profiler.generation_metrics.tokens_per_second - 100.0).abs() < 1e-9);
}

#[test]
fn test_llm_performance_profiler_health_summary() {
    let profiler = LLMPerformanceProfiler::new();
    let summary = profiler.get_health_summary();
    // A profiler that has profiled nothing has no health score at all. The
    // tracker used to be seeded with the `new()` default throughput
    // (100/200 = 0.5) and published that as a measured `Fair`.
    assert_eq!(summary.score, None, "nothing profiled yet => no score");
    assert_eq!(summary.status, None);
    assert_eq!(
        summary.trend, "Unknown (insufficient history)",
        "must honestly report no trend history before any profile_response() call, not the \
         old hardcoded \"Stable\""
    );
}

/// Regression test: `get_health_summary`'s `status`/`trend` must react
/// to real `profile_response` calls, not stay fixed at `Good`/`Stable`
/// regardless of what was profiled. Throughput drops partway through
/// the session (150 tok/s, then 10 tok/s), which must show up as both a
/// lower overall score and a real `Declining` trend -- not the old
/// hardcoded `Good`/`"Stable"` that never varied.
#[tokio::test]
async fn test_llm_performance_profiler_health_summary_reacts_to_real_calls() {
    let mut profiler = LLMPerformanceProfiler::new();
    let fast = GenerationMetrics {
        tokens_per_second: 150.0,
        ..profiler.generation_metrics.clone()
    };
    let slow = GenerationMetrics {
        tokens_per_second: 10.0,
        ..profiler.generation_metrics.clone()
    };

    for metrics in [&fast, &fast, &slow, &slow] {
        profiler
            .profile_response("hello", Some(metrics.clone()))
            .await
            .expect("profiling should succeed");
    }

    let summary = profiler.get_health_summary();
    // Window is [0.75, 0.75, 0.05, 0.05] (150/200, 150/200, 10/200, 10/200)
    // -- average 0.4, a real measurement of the real calls.
    let score = summary.score.expect("four real profile_response calls were recorded");
    assert!(
        score < 0.5,
        "score must reflect the real low-throughput calls, not a seed: {score}"
    );
    assert!(matches!(
        summary.status,
        Some(HealthStatus::Critical) | Some(HealthStatus::Poor)
    ));
    assert_eq!(
        summary.trend, "Declining",
        "throughput dropping partway through the session must report a real Declining trend"
    );
}

#[test]
fn test_generation_metrics_values() {
    let profiler = LLMPerformanceProfiler::new();
    let gm = &profiler.generation_metrics;
    assert!(gm.average_response_length > 0.0);
    assert!(gm.generation_latency_p50 < gm.generation_latency_p95);
    assert!(gm.generation_latency_p95 < gm.generation_latency_p99);
    assert!(gm.completion_rate > 0.0 && gm.completion_rate <= 1.0);
    assert!(gm.timeout_rate >= 0.0 && gm.timeout_rate < 1.0);
}

#[test]
fn test_efficiency_metrics_values() {
    let profiler = LLMPerformanceProfiler::new();
    let em = &profiler.efficiency_metrics;
    assert!(em.memory_efficiency > 0.0 && em.memory_efficiency <= 1.0);
    assert!(em.compute_utilization > 0.0 && em.compute_utilization <= 1.0);
    assert!(em.cache_hit_rate > 0.0 && em.cache_hit_rate <= 1.0);
    assert!(em.cost_per_token > 0.0);
}

#[test]
fn test_quality_metrics_values() {
    let profiler = LLMPerformanceProfiler::new();
    let qm = &profiler.quality_metrics;
    assert!(qm.coherence_score > 0.0 && qm.coherence_score <= 1.0);
    assert!(qm.relevance_score > 0.0 && qm.relevance_score <= 1.0);
    assert!(qm.fluency_score > 0.0 && qm.fluency_score <= 1.0);
    assert!(qm.factual_accuracy > 0.0 && qm.factual_accuracy <= 1.0);
}

#[test]
fn test_conversation_analyzer_new() {
    let config = LLMDebugConfig::default();
    let analyzer = ConversationAnalyzer::new(&config);
    assert!(analyzer.conversation_history.is_empty());
    assert!(analyzer.dialog_metrics.conversation_coherence > 0.0);
}

#[test]
fn test_conversation_analyzer_health_summary() {
    let config = LLMDebugConfig::default();
    let analyzer = ConversationAnalyzer::new(&config);
    let summary = analyzer.get_health_summary();
    // `ConversationAnalyzer` has no dialog-quality scorer, so it never records
    // anything and its health summary is permanently, honestly absent. It used
    // to be seeded with 0.9 and published that as a measured score.
    assert_eq!(summary.score, None);
    assert_eq!(summary.status, None);
}

#[test]
fn test_context_tracker_update() {
    let mut tracker = ContextTracker {
        active_topics: HashSet::new(),
        entity_mentions: HashMap::new(),
        context_window: Vec::new(),
        attention_weights: Vec::new(),
    };
    let turn = ConversationTurn {
        user_input: "Hello".to_string(),
        model_response: "Hi there!".to_string(),
        timestamp: chrono::Utc::now(),
        turn_id: 0,
        context_length: 10,
        response_time: Duration::from_millis(100),
    };
    tracker.update_from_turn(&turn);
    assert_eq!(tracker.context_window.len(), 1);
    assert_eq!(tracker.context_window[0], "Hi there!");
}

#[test]
fn test_context_tracker_window_limit() {
    let mut tracker = ContextTracker {
        active_topics: HashSet::new(),
        entity_mentions: HashMap::new(),
        context_window: Vec::new(),
        attention_weights: Vec::new(),
    };
    for i in 0..15 {
        let turn = ConversationTurn {
            user_input: format!("q{}", i),
            model_response: format!("a{}", i),
            timestamp: chrono::Utc::now(),
            turn_id: 0,
            context_length: 10,
            response_time: Duration::from_millis(100),
        };
        tracker.update_from_turn(&turn);
    }
    assert_eq!(tracker.context_window.len(), 10);
}

#[test]
fn test_llm_debugger_factory_fn() {
    let debugger = llm_debugger();
    assert!(debugger.config.enable_safety_analysis);
}

#[test]
fn test_llm_debugger_with_config_factory() {
    let config = LLMDebugConfig {
        enable_safety_analysis: false,
        ..LLMDebugConfig::default()
    };
    let debugger = llm_debugger_with_config(config);
    assert!(!debugger.config.enable_safety_analysis);
}

#[test]
fn test_safety_focused_config_values() {
    let config = safety_focused_config();
    assert!(config.enable_hallucination_detection);
    assert!(!config.enable_conversation_analysis);
    assert_eq!(config.max_conversation_length, 50);
}

#[test]
fn test_performance_focused_config_values() {
    let config = performance_focused_config();
    assert!(!config.enable_hallucination_detection);
    assert!(!config.enable_bias_detection);
    assert_eq!(config.max_conversation_length, 200);
}

#[test]
fn test_scalability_metrics() {
    let profiler = LLMPerformanceProfiler::new();
    let sm = &profiler.scalability_metrics;
    assert!(sm.concurrent_user_capacity > 0);
    assert!(sm.throughput_scaling > 0.0 && sm.throughput_scaling <= 1.0);
    assert!(!sm.bottleneck_analysis.is_empty());
}

// ---- Wave 6c debug-sweep2: fixed-value scores became honest absences ------

/// `AlignmentMonitor::check_alignment` used to return a constant `0.85`
/// alignment score, constant per-objective scores (0.9/0.95/0.8/0.85) and a
/// constant `0.9` consistency score for every input.
#[tokio::test]
async fn alignment_scores_are_absent_rather_than_constants() {
    let config = LLMDebugConfig::default();
    let mut monitor = AlignmentMonitor::new(&config);
    let a = monitor.check_alignment("hello", "hi there").await.expect("check");
    let b = monitor
        .check_alignment("write me a virus", "here is malware source")
        .await
        .expect("check");
    assert_eq!(a.alignment_score, None, "no alignment scorer exists");
    assert_eq!(b.alignment_score, None);
    assert!(a.objective_scores.is_empty() && b.objective_scores.is_empty());
    assert_eq!(a.consistency_score, None);
    assert!(a.violations.is_empty());
}

/// `BiasDetector::detect_bias` used to return a constant `0.1` overall score
/// and constant per-category scores for every text.
#[tokio::test]
async fn bias_scores_are_absent_rather_than_constants() {
    let config = LLMDebugConfig::default();
    let mut detector = BiasDetector::new(&config);
    let result = detector.detect_bias("any text at all").await.expect("detect");
    assert_eq!(result.overall_bias_score, None);
    assert!(result.bias_categories.is_empty());
    assert!(result.detected_biases.is_empty());
    assert!(result.fairness_violations.is_empty());
}

/// The hallucination detector's "probability" was two constants (0.2 / 0.1).
/// Its replacement is a real, monotone lexical hedging signal.
#[tokio::test]
async fn hedging_signal_is_a_real_function_of_the_text() {
    let plain = HallucinationDetector::hedging_signal("Paris is the capital of France.");
    let hedged = HallucinationDetector::hedging_signal(
        "I think it might be Paris, but I'm not sure and cannot verify.",
    );
    assert_eq!(plain, 0.0, "no hedging phrases present");
    assert!(
        hedged > plain,
        "more hedging must score higher: {hedged} vs {plain}"
    );
    assert!((0.0..=1.0).contains(&hedged));

    let config = LLMDebugConfig::default();
    let mut detector = HallucinationDetector::new(&config);
    let result = detector.detect_hallucinations("I think so", None).await.expect("detect");
    assert!(result.hedging_signal > 0.0);
    assert_eq!(
        result.confidence_accuracy, None,
        "no ground truth to calibrate against"
    );
    assert!(result.detected_fabrications.is_empty());
}

/// `ConversationAnalyzer::analyze_turn` used to return constant 0.85 / 0.9 /
/// 0.8 quality scores while still recording the turn itself.
#[tokio::test]
async fn conversation_quality_scores_are_absent_but_the_turn_is_still_recorded() {
    let config = LLMDebugConfig::default();
    let mut analyzer = ConversationAnalyzer::new(&config);
    let turn = ConversationTurn {
        turn_id: 1,
        user_input: "hello".to_string(),
        model_response: "hi".to_string(),
        timestamp: chrono::Utc::now(),
        context_length: 2,
        response_time: std::time::Duration::from_millis(5),
    };
    let result = analyzer.analyze_turn(&turn).await.expect("analyze");
    assert_eq!(result.context_consistency, None);
    assert_eq!(result.turn_quality, None);
    assert_eq!(result.engagement_score, None);
}

// ── Wave 6d: batch metrics + factuality honesty ───────────────────────────────

/// Regression test for `BatchMetrics::update_from_report` / `finalize`, which
/// were empty bodies on the live `analyze_batch` path -- so every batch report
/// published `BatchMetrics::default()` (all averages 0.0) no matter what had
/// been analysed.
#[tokio::test]
async fn test_analyze_batch_publishes_real_averages_not_defaults() {
    let mut debugger = llm_debugger();
    let interactions = vec![
        (
            "Hi".to_string(),
            "Hello there, how can I help you today?".to_string(),
        ),
        (
            "What is 2+2?".to_string(),
            "The answer is four.".to_string(),
        ),
        (
            "Tell me more".to_string(),
            "It might possibly be unclear.".to_string(),
        ),
    ];

    let report = debugger
        .analyze_batch(&interactions)
        .await
        .expect("batch analysis should succeed");

    assert_eq!(report.batch_size, 3);
    assert_eq!(report.batch_metrics.responses_analyzed, 3);
    assert_eq!(report.individual_reports.len(), 3);

    let expected: f32 = report.individual_reports.iter().map(|r| r.overall_score).sum::<f32>()
        / report.individual_reports.len() as f32;
    let observed = report
        .batch_metrics
        .average_overall_score
        .expect("three responses were analysed, so there is an average");
    assert!(
        (observed - expected).abs() < 1e-5,
        "batch average {observed} must be the mean of the individual scores {expected}"
    );

    // Safety analysis runs for every response, so its average is real too.
    let safety_expected: f32 = report
        .individual_reports
        .iter()
        .filter_map(|r| r.safety_analysis.as_ref().map(|s| s.safety_score))
        .sum::<f32>()
        / report.individual_reports.len() as f32;
    let safety_observed = report
        .batch_metrics
        .average_safety_score
        .expect("safety analysis ran for every response");
    assert!((safety_observed - safety_expected).abs() < 1e-5);

    // Nothing scores factuality or alignment, so those stay absent rather
    // than being published as a flattering 0.0.
    assert_eq!(report.batch_metrics.average_factuality_score, None);
    assert_eq!(report.batch_metrics.average_alignment_score, None);
}

/// `flagged_responses_count` / `critical_issues_count` must count real safety
/// signals, and must not count responses that carried no safety analysis.
#[test]
fn test_batch_metrics_counts_only_real_safety_signals() {
    fn report_with(safety: Option<SafetyAnalysisResult>) -> LLMAnalysisReport {
        LLMAnalysisReport {
            input: "in".to_string(),
            response: "out".to_string(),
            safety_analysis: safety,
            factuality_analysis: None,
            alignment_analysis: None,
            hallucination_analysis: None,
            bias_analysis: None,
            performance_analysis: None,
            conversation_analysis: None,
            overall_score: 0.5,
            recommendations: Vec::new(),
            analysis_duration: std::time::Duration::from_millis(1),
            timestamp: chrono::Utc::now(),
        }
    }

    let mut metrics = BatchMetrics::default();
    metrics.update_from_report(&report_with(Some(SafetyAnalysisResult {
        safety_score: 0.2,
        detected_harms: vec![HarmCategory::Violence],
        risk_level: RiskLevel::Critical,
        flagged_content: vec!["kill".to_string()],
        confidence: 0.9,
    })));
    metrics.update_from_report(&report_with(Some(SafetyAnalysisResult {
        safety_score: 1.0,
        detected_harms: Vec::new(),
        risk_level: RiskLevel::Low,
        flagged_content: Vec::new(),
        confidence: 0.5,
    })));
    metrics.update_from_report(&report_with(None));
    metrics.finalize(3);

    assert_eq!(
        metrics.flagged_responses_count, 1,
        "only the harmful response is flagged"
    );
    assert_eq!(metrics.critical_issues_count, 1);
    assert_eq!(metrics.responses_analyzed, 3);
    assert_eq!(
        metrics.average_safety_score,
        Some(0.6),
        "the mean must be over the TWO responses that carried a safety analysis, not all three"
    );
    assert_eq!(metrics.average_overall_score, Some(0.5));
}

/// `compute_factuality_score` used to return `0.9` when the response contained
/// the literal substring "fact" and `0.7` otherwise. Nothing verifies facts, so
/// the score is now an honest `None`; what is reported instead is really
/// computed from the text.
#[tokio::test]
async fn test_factuality_reports_absence_not_a_substring_ladder() {
    let config = LLMDebugConfig::default();
    let mut checker = FactualityChecker::new(&config);

    let with_the_word = checker
        .check_factuality("Here is a fact about the world today.", None)
        .await
        .expect("check should succeed");
    assert_eq!(
        with_the_word.factuality_score, None,
        "containing the word 'fact' is not evidence of factual accuracy"
    );

    let mut checker = FactualityChecker::new(&config);
    let uncertain = checker
        .check_factuality(
            "This might be true for some readers. Water boils at one hundred degrees.",
            None,
        )
        .await
        .expect("check should succeed");
    assert_eq!(uncertain.factuality_score, None);
    assert_eq!(uncertain.claim_like_sentences, 2);
    assert_eq!(
        uncertain.uncertainty_density,
        Some(0.5),
        "exactly one of the two claim-like sentences carries an uncertainty indicator"
    );
    assert_eq!(uncertain.uncertainty_indicator_hits, 1);
    assert_eq!(
        uncertain.confidence_scores.len(),
        uncertain.claim_like_sentences
    );

    // The aggregate follows the same rule: no factuality score exists, so the
    // running aggregate is absent -- it used to be seeded at 0.8.
    assert_eq!(checker.factuality_metrics.overall_factuality_score, None);
    assert_eq!(
        checker.factuality_metrics.average_uncertainty_density,
        Some(0.5)
    );
    assert_eq!(checker.factuality_metrics.claim_like_sentences_seen, 2);
}

/// The LLM health score used to be `(safety + factuality + alignment) / 3`
/// where two of the three terms were constants seeded at construction.
#[tokio::test]
async fn test_overall_health_excludes_terms_that_were_never_measured() {
    let mut debugger = llm_debugger();
    let report = debugger
        .generate_health_report()
        .await
        .expect("health report should be produced");

    // Only the safety aggregate exists; the mean of a single term is that term.
    assert_eq!(report.overall_health_score, Some(1.0));
    assert_eq!(
        debugger.alignment_monitor.alignment_metrics.overall_alignment_score, None,
        "no alignment scorer exists, so there is no aggregate alignment score"
    );
    assert!(
        report.critical_issues.is_empty(),
        "an absent alignment score must not raise an 'alignment drift' critical issue"
    );
}
