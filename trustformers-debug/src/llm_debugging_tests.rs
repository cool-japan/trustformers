//! Additional tests for the llm_debugging module.

use super::*;
use std::collections::HashMap;

// ── LLMDebugConfig ────────────────────────────────────────────────────────────

#[test]
fn test_llm_debug_config_default() {
    let cfg = LLMDebugConfig::default();
    assert!(cfg.enable_safety_analysis);
    assert!(cfg.enable_factuality_checking);
    assert!(cfg.enable_alignment_monitoring);
    assert!(cfg.enable_hallucination_detection);
    assert!(cfg.enable_bias_detection);
    assert!(cfg.enable_llm_performance_profiling);
    assert!(cfg.enable_conversation_analysis);
    assert!((cfg.safety_threshold - 0.8).abs() < 1e-6);
    assert!((cfg.factuality_threshold - 0.7).abs() < 1e-6);
    assert_eq!(cfg.max_conversation_length, 100);
    assert!((cfg.analysis_sampling_rate - 1.0).abs() < 1e-6);
}

#[test]
fn test_llm_debug_config_clone() {
    let cfg = LLMDebugConfig::default();
    let cloned = cfg.clone();
    assert_eq!(cfg.enable_safety_analysis, cloned.enable_safety_analysis);
    assert!((cfg.safety_threshold - cloned.safety_threshold).abs() < 1e-10);
}

#[test]
fn test_llm_debug_config_custom() {
    let cfg = LLMDebugConfig {
        enable_safety_analysis: false,
        enable_factuality_checking: true,
        enable_alignment_monitoring: false,
        enable_hallucination_detection: true,
        enable_bias_detection: false,
        enable_llm_performance_profiling: true,
        enable_conversation_analysis: false,
        safety_threshold: 0.95,
        factuality_threshold: 0.85,
        max_conversation_length: 200,
        analysis_sampling_rate: 0.5,
    };
    assert!(!cfg.enable_safety_analysis);
    assert!(cfg.enable_factuality_checking);
    assert!((cfg.safety_threshold - 0.95).abs() < 1e-6);
    assert_eq!(cfg.max_conversation_length, 200);
    assert!((cfg.analysis_sampling_rate - 0.5).abs() < 1e-6);
}

// ── HarmCategory variants ─────────────────────────────────────────────────────

#[test]
fn test_harm_category_variants() {
    let variants = [
        HarmCategory::Toxicity,
        HarmCategory::Violence,
        HarmCategory::SelfHarm,
        HarmCategory::Harassment,
        HarmCategory::HateSpeech,
        HarmCategory::Sexual,
        HarmCategory::Privacy,
        HarmCategory::Misinformation,
        HarmCategory::Manipulation,
        HarmCategory::Illegal,
    ];
    for v in &variants {
        let _cloned = v.clone();
    }
}

#[test]
fn test_harm_category_eq() {
    assert_eq!(HarmCategory::Toxicity, HarmCategory::Toxicity);
    assert_ne!(HarmCategory::Toxicity, HarmCategory::Violence);
    assert_eq!(HarmCategory::Misinformation, HarmCategory::Misinformation);
}

// ── BiasCategory variants ─────────────────────────────────────────────────────

#[test]
fn test_bias_category_variants() {
    let variants = [
        BiasCategory::Gender,
        BiasCategory::Race,
        BiasCategory::Religion,
        BiasCategory::Age,
        BiasCategory::SocioEconomic,
        BiasCategory::Geographic,
        BiasCategory::Political,
        BiasCategory::Linguistic,
        BiasCategory::Ability,
        BiasCategory::Appearance,
    ];
    for v in &variants {
        let _cloned = v.clone();
    }
}

#[test]
fn test_bias_category_eq() {
    assert_eq!(BiasCategory::Gender, BiasCategory::Gender);
    assert_ne!(BiasCategory::Gender, BiasCategory::Race);
}

// ── AlignmentObjective variants ───────────────────────────────────────────────

#[test]
fn test_alignment_objective_variants() {
    let variants = [
        AlignmentObjective::Helpfulness,
        AlignmentObjective::Harmlessness,
        AlignmentObjective::Honesty,
        AlignmentObjective::Fairness,
        AlignmentObjective::Privacy,
        AlignmentObjective::Transparency,
        AlignmentObjective::Consistency,
        AlignmentObjective::Responsibility,
    ];
    for v in &variants {
        let _cloned = v.clone();
    }
}

// ── SafetyTrend variants ──────────────────────────────────────────────────────

#[test]
fn test_safety_trend_variants() {
    let variants = [
        SafetyTrend::Improving,
        SafetyTrend::Stable,
        SafetyTrend::Degrading,
        SafetyTrend::Volatile,
    ];
    for v in &variants {
        let _cloned = v.clone();
    }
}

#[test]
fn test_safety_trend_eq() {
    assert_eq!(SafetyTrend::Stable, SafetyTrend::Stable);
    assert_ne!(SafetyTrend::Improving, SafetyTrend::Degrading);
}

// ── AlignmentTrend variants ───────────────────────────────────────────────────

#[test]
fn test_alignment_trend_variants() {
    let variants = [
        AlignmentTrend::Improving,
        AlignmentTrend::Stable,
        AlignmentTrend::Degrading,
        AlignmentTrend::Inconsistent,
    ];
    for v in &variants {
        let _cloned = v.clone();
    }
}

#[test]
fn test_alignment_trend_eq() {
    assert_eq!(AlignmentTrend::Stable, AlignmentTrend::Stable);
    assert_ne!(AlignmentTrend::Improving, AlignmentTrend::Inconsistent);
}

// ── RiskLevel variants ────────────────────────────────────────────────────────

#[test]
fn test_risk_level_variants() {
    let variants = [
        RiskLevel::Low,
        RiskLevel::Medium,
        RiskLevel::High,
        RiskLevel::Critical,
    ];
    for v in &variants {
        let _cloned = v.clone();
        let _debug = format!("{:?}", v);
    }
}

#[test]
fn test_risk_level_eq() {
    assert_eq!(RiskLevel::Low, RiskLevel::Low);
    assert_ne!(RiskLevel::Low, RiskLevel::High);
    assert_eq!(RiskLevel::Critical, RiskLevel::Critical);
}

// ── HealthStatus variants ─────────────────────────────────────────────────────

#[test]
fn test_health_status_variants() {
    let variants = [
        HealthStatus::Excellent,
        HealthStatus::Good,
        HealthStatus::Fair,
        HealthStatus::Poor,
        HealthStatus::Critical,
    ];
    for v in &variants {
        let _cloned = v.clone();
    }
}

#[test]
fn test_health_status_eq() {
    assert_eq!(HealthStatus::Good, HealthStatus::Good);
    assert_ne!(HealthStatus::Excellent, HealthStatus::Critical);
}

// ── IssueCategory variants ────────────────────────────────────────────────────

#[test]
fn test_issue_category_variants() {
    let variants = [
        IssueCategory::Safety,
        IssueCategory::Factuality,
        IssueCategory::Alignment,
        IssueCategory::Bias,
        IssueCategory::Performance,
        IssueCategory::Conversation,
    ];
    for v in &variants {
        let _cloned = v.clone();
    }
}

// ── IssueSeverity variants ────────────────────────────────────────────────────

#[test]
fn test_issue_severity_variants() {
    let variants = [
        IssueSeverity::Low,
        IssueSeverity::Medium,
        IssueSeverity::High,
        IssueSeverity::Critical,
    ];
    for v in &variants {
        let _cloned = v.clone();
    }
}

// ── LLMDebugger construction ──────────────────────────────────────────────────

#[test]
fn test_llm_debugger_new_default() {
    let debugger = LLMDebugger::new(LLMDebugConfig::default());
    assert!(debugger.config.enable_safety_analysis);
    assert!(debugger.config.enable_factuality_checking);
}

#[test]
fn test_llm_debugger_new_custom_config() {
    let config = LLMDebugConfig {
        enable_safety_analysis: false,
        enable_factuality_checking: false,
        enable_alignment_monitoring: true,
        enable_hallucination_detection: false,
        enable_bias_detection: false,
        enable_llm_performance_profiling: true,
        enable_conversation_analysis: false,
        safety_threshold: 0.95,
        factuality_threshold: 0.85,
        max_conversation_length: 200,
        analysis_sampling_rate: 0.5,
    };
    let debugger = LLMDebugger::new(config.clone());
    assert!(!debugger.config.enable_safety_analysis);
    assert!((debugger.config.safety_threshold - 0.95).abs() < 1e-6);
}

#[test]
fn test_llm_debugger_factory_function() {
    let debugger = llm_debugger();
    assert!(debugger.config.enable_safety_analysis);
}

#[test]
fn test_llm_debugger_with_config_factory() {
    let cfg = performance_focused_config();
    let debugger = llm_debugger_with_config(cfg);
    assert!(!debugger.config.enable_safety_analysis);
    assert!(debugger.config.enable_llm_performance_profiling);
}

// ── safety_focused_config ─────────────────────────────────────────────────────

#[test]
fn test_safety_focused_config_values() {
    let cfg = safety_focused_config();
    assert!(cfg.enable_safety_analysis);
    assert!(cfg.enable_bias_detection);
    assert!(!cfg.enable_llm_performance_profiling);
    assert!(!cfg.enable_conversation_analysis);
    assert!((cfg.safety_threshold - 0.9).abs() < 1e-6);
    assert_eq!(cfg.max_conversation_length, 50);
}

// ── performance_focused_config ────────────────────────────────────────────────

#[test]
fn test_performance_focused_config_values() {
    let cfg = performance_focused_config();
    assert!(!cfg.enable_safety_analysis);
    assert!(!cfg.enable_bias_detection);
    assert!(cfg.enable_llm_performance_profiling);
    assert!(cfg.enable_conversation_analysis);
    assert_eq!(cfg.max_conversation_length, 200);
}

// ── GenerationMetrics ─────────────────────────────────────────────────────────

#[test]
fn test_generation_metrics_construction() {
    let metrics = GenerationMetrics {
        tokens_per_second: 100.0,
        average_response_length: 50.0,
        generation_latency_p50: 200.0,
        generation_latency_p95: 500.0,
        generation_latency_p99: 800.0,
        first_token_latency: 50.0,
        completion_rate: 0.99,
        timeout_rate: 0.01,
    };
    assert!(metrics.tokens_per_second > 0.0);
    assert!(metrics.completion_rate > 0.0 && metrics.completion_rate <= 1.0);
    assert!(metrics.timeout_rate >= 0.0 && metrics.timeout_rate <= 1.0);
    assert!(metrics.first_token_latency > 0.0);
}

// ── SafetyMetrics ─────────────────────────────────────────────────────────────

#[test]
fn test_safety_metrics_construction() {
    let metrics = SafetyMetrics {
        overall_safety_score: 0.92,
        harm_category_scores: HashMap::new(),
        flagged_responses: 2,
        total_responses_analyzed: 100,
        average_response_safety: 0.91,
        safety_trend: SafetyTrend::Stable,
    };
    assert!(metrics.overall_safety_score > 0.0 && metrics.overall_safety_score <= 1.0);
    assert!(metrics.flagged_responses <= metrics.total_responses_analyzed);
}

// ── BatchMetrics ──────────────────────────────────────────────────────────────

#[test]
fn test_batch_metrics_default() {
    let metrics = BatchMetrics::default();
    assert!((metrics.average_overall_score - 0.0).abs() < 1e-6);
    assert_eq!(metrics.flagged_responses_count, 0);
    assert_eq!(metrics.critical_issues_count, 0);
    assert!(metrics.performance_summary.is_none());
}

#[test]
fn test_batch_metrics_update_and_finalize() {
    let mut metrics = BatchMetrics::default();
    let report = LLMAnalysisReport {
        input: "hi".to_string(),
        response: "hello".to_string(),
        safety_analysis: None,
        factuality_analysis: None,
        alignment_analysis: None,
        hallucination_analysis: None,
        bias_analysis: None,
        performance_analysis: None,
        conversation_analysis: None,
        overall_score: 0.9,
        recommendations: Vec::new(),
        analysis_duration: std::time::Duration::from_millis(100),
        timestamp: chrono::Utc::now(),
    };
    metrics.update_from_report(&report);
    metrics.finalize(1);
    // No panic
}

// ── LLMAnalysisReport ─────────────────────────────────────────────────────────

#[test]
fn test_llm_analysis_report_fields() {
    let report = LLMAnalysisReport {
        input: "What is 2+2?".to_string(),
        response: "The answer is 4.".to_string(),
        safety_analysis: None,
        factuality_analysis: None,
        alignment_analysis: None,
        hallucination_analysis: None,
        bias_analysis: None,
        performance_analysis: None,
        conversation_analysis: None,
        overall_score: 0.95,
        recommendations: vec!["All good".to_string()],
        analysis_duration: std::time::Duration::from_millis(42),
        timestamp: chrono::Utc::now(),
    };
    assert_eq!(report.input, "What is 2+2?");
    assert_eq!(report.response, "The answer is 4.");
    assert!((report.overall_score - 0.95).abs() < 1e-6);
    assert_eq!(report.recommendations.len(), 1);
}

// ── CriticalIssue ─────────────────────────────────────────────────────────────

#[test]
fn test_critical_issue_construction() {
    let issue = CriticalIssue {
        category: IssueCategory::Safety,
        severity: IssueSeverity::Critical,
        description: "Low safety score".to_string(),
        recommended_action: "Review immediately".to_string(),
    };
    assert!(matches!(issue.category, IssueCategory::Safety));
    assert!(matches!(issue.severity, IssueSeverity::Critical));
    assert!(!issue.description.is_empty());
}

// ── HealthSummary ─────────────────────────────────────────────────────────────

#[test]
fn test_health_summary_construction() {
    let summary = HealthSummary {
        score: 0.88,
        status: HealthStatus::Good,
        trend: "improving".to_string(),
        key_metrics: HashMap::new(),
        issues: Vec::new(),
    };
    assert!((summary.score - 0.88).abs() < 1e-6);
    assert!(matches!(summary.status, HealthStatus::Good));
    assert!(summary.issues.is_empty());
}

// ── FactualityMetrics ─────────────────────────────────────────────────────────

#[test]
fn test_factuality_metrics_construction() {
    let metrics = FactualityMetrics {
        overall_factuality_score: 0.85,
        verified_facts: 10,
        unverified_claims: 3,
        conflicting_information: 1,
        uncertainty_expressions: 2,
        knowledge_gaps: vec!["topic_a".to_string()],
        confidence_distribution: vec![0.9, 0.8, 0.7],
    };
    assert!(metrics.overall_factuality_score > 0.0 && metrics.overall_factuality_score <= 1.0);
    assert_eq!(metrics.knowledge_gaps.len(), 1);
    assert_eq!(metrics.confidence_distribution.len(), 3);
}

// ── AlignmentMetrics ──────────────────────────────────────────────────────────

#[test]
fn test_alignment_metrics_construction() {
    let metrics = AlignmentMetrics {
        objective_scores: HashMap::new(),
        overall_alignment_score: 0.9,
        alignment_violations: 0,
        value_consistency_score: 0.92,
        behavioral_drift: 0.05,
        alignment_trend: AlignmentTrend::Stable,
    };
    assert!(metrics.overall_alignment_score > 0.0 && metrics.overall_alignment_score <= 1.0);
    assert_eq!(metrics.alignment_violations, 0);
    assert!(metrics.behavioral_drift >= 0.0);
}

// ── HallucinationMetrics ──────────────────────────────────────────────────────

#[test]
fn test_hallucination_metrics_construction() {
    let metrics = HallucinationMetrics {
        hallucination_rate: 0.05,
        confidence_accuracy_correlation: 0.8,
        factual_consistency_score: 0.9,
        internal_consistency_score: 0.95,
        source_attribution_accuracy: 0.7,
        detected_fabrications: 2,
        uncertain_responses: 5,
    };
    assert!(metrics.hallucination_rate >= 0.0 && metrics.hallucination_rate <= 1.0);
    assert!(metrics.factual_consistency_score > 0.0);
    assert_eq!(metrics.detected_fabrications, 2);
}

// ── BiasMetrics ───────────────────────────────────────────────────────────────

#[test]
fn test_bias_metrics_construction() {
    let metrics = BiasMetrics {
        overall_bias_score: 0.1,
        bias_category_scores: HashMap::new(),
        demographic_fairness: HashMap::new(),
        representation_bias: 0.05,
        stereotype_propagation: 0.03,
        bias_amplification: 0.02,
        fairness_violations: 0,
    };
    assert!(metrics.overall_bias_score >= 0.0 && metrics.overall_bias_score <= 1.0);
    assert_eq!(metrics.fairness_violations, 0);
}
