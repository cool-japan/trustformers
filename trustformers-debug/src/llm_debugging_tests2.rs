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
        result.verified_claims,
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
    assert!(report.overall_health_score > 0.0);
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
    assert!(summary.score > 0.0 && summary.score <= 1.0);
    // A fresh profiler's initial score is 100 tokens/s against a 200
    // tokens/s reference ceiling -- 0.5, a real mid-range score, which
    // `health_status_from_score` correctly buckets as `Fair`. The old
    // implementation asserted `Good` here only because `status` was
    // hardcoded to `HealthStatus::Good` unconditionally, regardless of
    // `score`.
    assert!(matches!(summary.status, HealthStatus::Fair));
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
    // Window is [0.75, 0.75, 0.05, 0.05] (150/200, 150/200, 10/200,
    // 10/200) -- average 0.4, well under the `new()` default of 0.5.
    assert!(
        summary.score < 0.5,
        "score must move toward the real low-throughput calls, not stay frozen at 0.5: {}",
        summary.score
    );
    assert!(matches!(
        summary.status,
        HealthStatus::Critical | HealthStatus::Poor
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
    assert!(summary.score > 0.0);
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
