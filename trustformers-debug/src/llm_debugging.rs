//! Large Language Model (LLM) Specific Debugging
//!
//! This module provides specialized debugging capabilities for large language models,
//! focusing on safety, alignment, factuality, toxicity detection, and performance
//! characteristics specific to modern LLMs.
//!
//! # Honesty notes on what these analyzers actually do
//!
//! [`SafetyAnalyzer`] and [`FactualityChecker`] are **rule-based keyword
//! heuristics** over the literal response text (see
//! [`SafetyAnalyzer::find_harmful_keywords`] /
//! [`FactualityChecker::compute_factuality_score`]) -- not trained safety
//! classifiers or a real knowledge-base fact-check. Every `confidence` value
//! they report is a real function of a measurable property of the match
//! (e.g. how many distinct keyword categories fired), never a flat literal.
//!
//! [`AlignmentMonitor`], [`BiasDetector`], [`HallucinationDetector`], and
//! [`ConversationAnalyzer`]'s per-response scoring functions
//! (`compute_alignment_score`, `assess_objectives`,
//! `compute_overall_bias_score`, `analyze_bias_categories`,
//! `compute_hallucination_probability`, `compute_context_consistency`,
//! `assess_turn_quality`, `compute_engagement_score`) remain fixed-value
//! placeholders pending real scoring models -- each is explicitly marked
//! `NOTE:` at its definition. What every analyzer's [`HealthTracker`]-backed
//! `get_health_summary` now guarantees, regardless of that, is that
//! `status`/`trend` are a real, live function of whatever score the
//! analyzer actually produced -- never the old hardcoded
//! `HealthStatus::Good` / `"Stable"` that never changed no matter what was
//! analyzed.
// reason: debug/profiling scaffolding — structs are constructed and their fields/methods
// are retained for the data model, serialization completeness, and future consumers that
// do not yet read every member. Consolidated from many item-level #[allow(dead_code)].
#![allow(dead_code)]

use anyhow::Result;
// use scirs2_core::ndarray::*; // SciRS2 Integration Policy - was: use ndarray::{Array, ArrayD, IxDyn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

/// Main LLM debugging framework
#[derive(Debug)]
pub struct LLMDebugger {
    config: LLMDebugConfig,
    safety_analyzer: SafetyAnalyzer,
    factuality_checker: FactualityChecker,
    alignment_monitor: AlignmentMonitor,
    hallucination_detector: HallucinationDetector,
    bias_detector: BiasDetector,
    performance_profiler: LLMPerformanceProfiler,
    conversation_analyzer: ConversationAnalyzer,
}

/// Configuration for LLM debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMDebugConfig {
    /// Enable safety analysis (toxicity, harmful content)
    pub enable_safety_analysis: bool,
    /// Enable factuality checking
    pub enable_factuality_checking: bool,
    /// Enable alignment monitoring
    pub enable_alignment_monitoring: bool,
    /// Enable hallucination detection
    pub enable_hallucination_detection: bool,
    /// Enable bias detection
    pub enable_bias_detection: bool,
    /// Enable performance profiling for LLM-specific metrics
    pub enable_llm_performance_profiling: bool,
    /// Enable conversation flow analysis
    pub enable_conversation_analysis: bool,
    /// Threshold for safety score (0.0 to 1.0)
    pub safety_threshold: f32,
    /// Threshold for factuality score (0.0 to 1.0)
    pub factuality_threshold: f32,
    /// Maximum conversation length to analyze
    pub max_conversation_length: usize,
    /// Sampling rate for expensive analyses
    pub analysis_sampling_rate: f32,
}

impl Default for LLMDebugConfig {
    fn default() -> Self {
        Self {
            enable_safety_analysis: true,
            enable_factuality_checking: true,
            enable_alignment_monitoring: true,
            enable_hallucination_detection: true,
            enable_bias_detection: true,
            enable_llm_performance_profiling: true,
            enable_conversation_analysis: true,
            safety_threshold: 0.8,
            factuality_threshold: 0.7,
            max_conversation_length: 100,
            analysis_sampling_rate: 1.0,
        }
    }
}

/// Safety analyzer for detecting harmful, toxic, or inappropriate content
#[derive(Debug)]
pub struct SafetyAnalyzer {
    toxic_patterns: HashSet<String>,
    harm_categories: Vec<HarmCategory>,
    safety_metrics: SafetyMetrics,
    /// Real running history of [`Self::compute_safety_score`] results, used
    /// by [`Self::get_health_summary`]. See [`HealthTracker`].
    health: HealthTracker,
}

/// Categories of potential harm in LLM outputs
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HarmCategory {
    Toxicity,       // Toxic, offensive, or inappropriate language
    Violence,       // Violence or threats
    SelfHarm,       // Self-harm or suicide-related content
    Harassment,     // Harassment or bullying
    HateSpeech,     // Hate speech or discrimination
    Sexual,         // Sexual or adult content
    Privacy,        // Privacy violations or doxxing
    Misinformation, // Misinformation or conspiracy theories
    Manipulation,   // Social manipulation or deception
    Illegal,        // Illegal activities or advice
}

/// Safety metrics for tracking harmful content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyMetrics {
    pub overall_safety_score: f32,
    pub harm_category_scores: HashMap<HarmCategory, f32>,
    pub flagged_responses: usize,
    pub total_responses_analyzed: usize,
    pub average_response_safety: f32,
    pub safety_trend: SafetyTrend,
}

/// Trend in safety scores over time
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SafetyTrend {
    Improving,
    Stable,
    Degrading,
    Volatile,
}

/// Factuality checker for verifying the accuracy of LLM outputs
#[derive(Debug)]
pub struct FactualityChecker {
    fact_databases: Vec<String>,
    uncertainty_indicators: HashSet<String>,
    factuality_metrics: FactualityMetrics,
    health: HealthTracker,
}

/// Metrics for tracking factual accuracy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactualityMetrics {
    pub overall_factuality_score: f32,
    pub verified_facts: usize,
    pub unverified_claims: usize,
    pub conflicting_information: usize,
    pub uncertainty_expressions: usize,
    pub knowledge_gaps: Vec<String>,
    pub confidence_distribution: Vec<f32>,
}

/// Alignment monitor for ensuring LLM outputs align with intended behavior
#[derive(Debug)]
pub struct AlignmentMonitor {
    alignment_objectives: Vec<AlignmentObjective>,
    alignment_metrics: AlignmentMetrics,
    value_alignment_score: f32,
    health: HealthTracker,
}

/// Types of alignment objectives for LLMs
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlignmentObjective {
    Helpfulness,    // Be helpful and informative
    Harmlessness,   // Avoid causing harm
    Honesty,        // Be truthful and transparent
    Fairness,       // Treat all users fairly
    Privacy,        // Respect privacy and confidentiality
    Transparency,   // Be clear about limitations
    Consistency,    // Maintain consistent behavior
    Responsibility, // Take appropriate responsibility for outputs
}

/// Metrics for alignment monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentMetrics {
    pub objective_scores: HashMap<AlignmentObjective, f32>,
    pub overall_alignment_score: f32,
    pub alignment_violations: usize,
    pub value_consistency_score: f32,
    pub behavioral_drift: f32,
    pub alignment_trend: AlignmentTrend,
}

/// Trend in alignment scores over time
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlignmentTrend {
    Improving,
    Stable,
    Degrading,
    Inconsistent,
}

/// Hallucination detector for identifying false or fabricated information
#[derive(Debug)]
pub struct HallucinationDetector {
    confidence_thresholds: HashMap<String, f32>,
    consistency_checker: ConsistencyChecker,
    hallucination_metrics: HallucinationMetrics,
}

/// Metrics for hallucination detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HallucinationMetrics {
    pub hallucination_rate: f32,
    pub confidence_accuracy_correlation: f32,
    pub factual_consistency_score: f32,
    pub internal_consistency_score: f32,
    pub source_attribution_accuracy: f32,
    pub detected_fabrications: usize,
    pub uncertain_responses: usize,
}

/// Consistency checker for internal consistency in responses
#[derive(Debug)]
pub struct ConsistencyChecker {
    previous_responses: Vec<String>,
    consistency_cache: HashMap<String, f32>,
}

/// Bias detector for identifying various forms of bias in LLM outputs
#[derive(Debug)]
pub struct BiasDetector {
    bias_categories: Vec<BiasCategory>,
    demographic_groups: Vec<String>,
    bias_metrics: BiasMetrics,
    health: HealthTracker,
}

/// Types of bias to detect in LLM outputs
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BiasCategory {
    Gender,        // Gender-based bias
    Race,          // Racial or ethnic bias
    Religion,      // Religious bias
    Age,           // Age-based bias
    SocioEconomic, // Socioeconomic bias
    Geographic,    // Geographic or cultural bias
    Political,     // Political bias
    Linguistic,    // Language or accent bias
    Ability,       // Disability or ability bias
    Appearance,    // Physical appearance bias
}

/// Metrics for bias detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasMetrics {
    pub overall_bias_score: f32,
    pub bias_category_scores: HashMap<BiasCategory, f32>,
    pub demographic_fairness: HashMap<String, f32>,
    pub representation_bias: f32,
    pub stereotype_propagation: f32,
    pub bias_amplification: f32,
    pub fairness_violations: usize,
}

/// Performance profiler specific to LLM characteristics
#[derive(Debug)]
pub struct LLMPerformanceProfiler {
    generation_metrics: GenerationMetrics,
    efficiency_metrics: EfficiencyMetrics,
    quality_metrics: QualityMetrics,
    scalability_metrics: ScalabilityMetrics,
    health: HealthTracker,
}

/// Metrics for text generation performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationMetrics {
    pub tokens_per_second: f32,
    pub average_response_length: f32,
    pub generation_latency_p50: f32,
    pub generation_latency_p95: f32,
    pub generation_latency_p99: f32,
    pub first_token_latency: f32,
    pub completion_rate: f32,
    pub timeout_rate: f32,
}

/// Metrics for computational efficiency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    pub memory_efficiency: f32,
    pub compute_utilization: f32,
    pub energy_consumption: f32,
    pub carbon_footprint_estimate: f32,
    pub cost_per_token: f32,
    pub batch_processing_efficiency: f32,
    pub cache_hit_rate: f32,
}

/// Metrics for output quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub coherence_score: f32,
    pub relevance_score: f32,
    pub fluency_score: f32,
    pub informativeness_score: f32,
    pub creativity_score: f32,
    pub factual_accuracy: f32,
    pub readability_score: f32,
    pub engagement_score: f32,
}

/// Metrics for scalability analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityMetrics {
    pub concurrent_user_capacity: usize,
    pub throughput_scaling: f32,
    pub memory_scaling: f32,
    pub latency_degradation: f32,
    pub bottleneck_analysis: Vec<String>,
    pub resource_utilization_efficiency: f32,
}

/// Conversation analyzer for multi-turn dialog analysis
#[derive(Debug)]
pub struct ConversationAnalyzer {
    conversation_history: Vec<ConversationTurn>,
    dialog_metrics: DialogMetrics,
    context_tracking: ContextTracker,
    health: HealthTracker,
}

/// Single turn in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub turn_id: usize,
    pub user_input: String,
    pub model_response: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub context_length: usize,
    pub response_time: Duration,
}

/// Metrics for dialog analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogMetrics {
    pub conversation_coherence: f32,
    pub context_maintenance: f32,
    pub topic_consistency: f32,
    pub response_appropriateness: f32,
    pub conversation_engagement: f32,
    pub turn_taking_naturalness: f32,
    pub memory_utilization: f32,
    pub dialog_success_rate: f32,
}

/// Context tracking for conversation continuity
#[derive(Debug)]
pub struct ContextTracker {
    active_topics: HashSet<String>,
    entity_mentions: HashMap<String, usize>,
    context_window: Vec<String>,
    attention_weights: Vec<f32>,
}

impl LLMDebugger {
    /// Create a new LLM debugger
    pub fn new(config: LLMDebugConfig) -> Self {
        Self {
            config: config.clone(),
            safety_analyzer: SafetyAnalyzer::new(&config),
            factuality_checker: FactualityChecker::new(&config),
            alignment_monitor: AlignmentMonitor::new(&config),
            hallucination_detector: HallucinationDetector::new(&config),
            bias_detector: BiasDetector::new(&config),
            performance_profiler: LLMPerformanceProfiler::new(),
            conversation_analyzer: ConversationAnalyzer::new(&config),
        }
    }

    /// Comprehensive LLM analysis of a model response
    pub async fn analyze_response(
        &mut self,
        user_input: &str,
        model_response: &str,
        context: Option<&[String]>,
        generation_metrics: Option<GenerationMetrics>,
    ) -> Result<LLMAnalysisReport> {
        let start_time = Instant::now();

        // Safety analysis
        let safety_analysis = if self.config.enable_safety_analysis {
            Some(self.safety_analyzer.analyze_safety(model_response).await?)
        } else {
            None
        };

        // Factuality checking
        let factuality_analysis = if self.config.enable_factuality_checking {
            Some(self.factuality_checker.check_factuality(model_response, context).await?)
        } else {
            None
        };

        // Alignment monitoring
        let alignment_analysis = if self.config.enable_alignment_monitoring {
            Some(self.alignment_monitor.check_alignment(user_input, model_response).await?)
        } else {
            None
        };

        // Hallucination detection
        let hallucination_analysis = if self.config.enable_hallucination_detection {
            Some(
                self.hallucination_detector
                    .detect_hallucinations(model_response, context)
                    .await?,
            )
        } else {
            None
        };

        // Bias detection
        let bias_analysis = if self.config.enable_bias_detection {
            Some(self.bias_detector.detect_bias(model_response).await?)
        } else {
            None
        };

        // Performance profiling
        let performance_analysis = if self.config.enable_llm_performance_profiling {
            Some(
                self.performance_profiler
                    .profile_response(model_response, generation_metrics)
                    .await?,
            )
        } else {
            None
        };

        // Conversation analysis (if part of a dialog)
        let conversation_analysis = if self.config.enable_conversation_analysis {
            let turn = ConversationTurn {
                turn_id: self.conversation_analyzer.conversation_history.len(),
                user_input: user_input.to_string(),
                model_response: model_response.to_string(),
                timestamp: chrono::Utc::now(),
                context_length: context.map(|c| c.len()).unwrap_or(0),
                response_time: start_time.elapsed(),
            };
            Some(self.conversation_analyzer.analyze_turn(&turn).await?)
        } else {
            None
        };

        let analysis_duration = start_time.elapsed();

        Ok(LLMAnalysisReport {
            input: user_input.to_string(),
            response: model_response.to_string(),
            safety_analysis: safety_analysis.clone(),
            factuality_analysis: factuality_analysis.clone(),
            alignment_analysis: alignment_analysis.clone(),
            hallucination_analysis,
            bias_analysis,
            performance_analysis,
            conversation_analysis,
            overall_score: self.compute_overall_score(
                &safety_analysis,
                &factuality_analysis,
                &alignment_analysis,
            ),
            recommendations: self.generate_recommendations(
                &safety_analysis,
                &factuality_analysis,
                &alignment_analysis,
            ),
            analysis_duration,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Batch analysis of multiple responses
    pub async fn analyze_batch(
        &mut self,
        interactions: &[(String, String)], // (input, response) pairs
    ) -> Result<BatchLLMAnalysisReport> {
        let mut individual_reports = Vec::new();
        let mut batch_metrics = BatchMetrics::default();

        for (input, response) in interactions {
            let report = self.analyze_response(input, response, None, None).await?;
            batch_metrics.update_from_report(&report);
            individual_reports.push(report);
        }

        batch_metrics.finalize(interactions.len());

        Ok(BatchLLMAnalysisReport {
            individual_reports,
            batch_metrics,
            batch_size: interactions.len(),
            analysis_timestamp: chrono::Utc::now(),
        })
    }

    /// Generate comprehensive LLM health report
    pub async fn generate_health_report(&mut self) -> Result<LLMHealthReport> {
        Ok(LLMHealthReport {
            overall_health_score: self.compute_overall_health(),
            safety_health: self.safety_analyzer.get_health_summary(),
            factuality_health: self.factuality_checker.get_health_summary(),
            alignment_health: self.alignment_monitor.get_health_summary(),
            bias_health: self.bias_detector.get_health_summary(),
            performance_health: self.performance_profiler.get_health_summary(),
            conversation_health: self.conversation_analyzer.get_health_summary(),
            critical_issues: self.identify_critical_issues(),
            recommendations: self.generate_health_recommendations(),
            report_timestamp: chrono::Utc::now(),
        })
    }

    /// Compute overall score from analysis components
    fn compute_overall_score(
        &self,
        safety: &Option<SafetyAnalysisResult>,
        factuality: &Option<FactualityAnalysisResult>,
        alignment: &Option<AlignmentAnalysisResult>,
    ) -> f32 {
        let mut total_score = 0.0;
        let mut weight_sum = 0.0;

        if let Some(s) = safety {
            total_score += s.safety_score * 0.3;
            weight_sum += 0.3;
        }

        if let Some(f) = factuality {
            total_score += f.factuality_score * 0.3;
            weight_sum += 0.3;
        }

        // Only a real alignment score contributes; when the analyzer reports
        // `None` the weighted mean simply drops that term rather than folding
        // in a stand-in value.
        if let Some(score) = alignment.as_ref().and_then(|a| a.alignment_score) {
            total_score += score * 0.4;
            weight_sum += 0.4;
        }

        if weight_sum > 0.0 {
            total_score / weight_sum
        } else {
            0.0
        }
    }

    /// Generate actionable recommendations
    fn generate_recommendations(
        &self,
        safety: &Option<SafetyAnalysisResult>,
        factuality: &Option<FactualityAnalysisResult>,
        alignment: &Option<AlignmentAnalysisResult>,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        if let Some(s) = safety {
            if s.safety_score < self.config.safety_threshold {
                recommendations
                    .push("Consider additional safety filtering or fine-tuning".to_string());
            }
        }

        if let Some(f) = factuality {
            if f.factuality_score < self.config.factuality_threshold {
                recommendations
                    .push("Verify factual claims and consider knowledge base updates".to_string());
            }
        }

        if let Some(score) = alignment.as_ref().and_then(|a| a.alignment_score) {
            if score < 0.7 {
                recommendations.push(
                    "Review alignment objectives and consider additional RLHF training".to_string(),
                );
            }
        }

        recommendations
    }

    /// Compute overall health score
    /// Unweighted mean of the three analyzer-level aggregate scores.
    ///
    /// Each term is the analyzer's own running average, so this is only as
    /// meaningful as the analyzers feeding it: `overall_alignment_score` is
    /// never updated by [`AlignmentMonitor::check_alignment`] (no alignment
    /// scorer exists), so it contributes whatever a caller last set on
    /// `alignment_metrics`.
    fn compute_overall_health(&self) -> f32 {
        (self.safety_analyzer.safety_metrics.overall_safety_score
            + self.factuality_checker.factuality_metrics.overall_factuality_score
            + self.alignment_monitor.alignment_metrics.overall_alignment_score)
            / 3.0
    }

    /// Identify critical issues requiring immediate attention
    fn identify_critical_issues(&self) -> Vec<CriticalIssue> {
        let mut issues = Vec::new();

        // Check safety issues
        if self.safety_analyzer.safety_metrics.overall_safety_score < 0.5 {
            issues.push(CriticalIssue {
                category: IssueCategory::Safety,
                severity: IssueSeverity::Critical,
                description: "Low overall safety score detected".to_string(),
                recommended_action: "Immediate safety review and filtering required".to_string(),
            });
        }

        // Check alignment issues
        if self.alignment_monitor.alignment_metrics.overall_alignment_score < 0.6 {
            issues.push(CriticalIssue {
                category: IssueCategory::Alignment,
                severity: IssueSeverity::High,
                description: "Alignment drift detected".to_string(),
                recommended_action: "Review training data and consider alignment fine-tuning"
                    .to_string(),
            });
        }

        issues
    }

    /// Generate health improvement recommendations
    fn generate_health_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Add safety recommendations
        if self.safety_analyzer.safety_metrics.overall_safety_score < 0.8 {
            recommendations.push("Implement additional safety training data".to_string());
            recommendations.push("Consider constitutional AI techniques".to_string());
        }

        // Add performance recommendations
        if self.performance_profiler.generation_metrics.tokens_per_second < 50.0 {
            recommendations.push("Optimize inference pipeline for better throughput".to_string());
            recommendations.push("Consider model quantization or distillation".to_string());
        }

        recommendations
    }
}

// Analysis result structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMAnalysisReport {
    pub input: String,
    pub response: String,
    pub safety_analysis: Option<SafetyAnalysisResult>,
    pub factuality_analysis: Option<FactualityAnalysisResult>,
    pub alignment_analysis: Option<AlignmentAnalysisResult>,
    pub hallucination_analysis: Option<HallucinationAnalysisResult>,
    pub bias_analysis: Option<BiasAnalysisResult>,
    pub performance_analysis: Option<PerformanceAnalysisResult>,
    pub conversation_analysis: Option<ConversationAnalysisResult>,
    pub overall_score: f32,
    pub recommendations: Vec<String>,
    pub analysis_duration: Duration,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchLLMAnalysisReport {
    pub individual_reports: Vec<LLMAnalysisReport>,
    pub batch_metrics: BatchMetrics,
    pub batch_size: usize,
    pub analysis_timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchMetrics {
    pub average_overall_score: f32,
    pub average_safety_score: f32,
    pub average_factuality_score: f32,
    pub average_alignment_score: f32,
    pub flagged_responses_count: usize,
    pub critical_issues_count: usize,
    pub performance_summary: Option<PerformanceAnalysisResult>,
}

impl BatchMetrics {
    pub fn update_from_report(&mut self, _report: &LLMAnalysisReport) {
        // Implementation would accumulate metrics from individual reports
    }

    pub fn finalize(&mut self, _batch_size: usize) {
        // Implementation would compute final averages
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMHealthReport {
    pub overall_health_score: f32,
    pub safety_health: HealthSummary,
    pub factuality_health: HealthSummary,
    pub alignment_health: HealthSummary,
    pub bias_health: HealthSummary,
    pub performance_health: HealthSummary,
    pub conversation_health: HealthSummary,
    pub critical_issues: Vec<CriticalIssue>,
    pub recommendations: Vec<String>,
    pub report_timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthSummary {
    /// Mean of the analyzer's recorded scores, or `None` when it has recorded
    /// none -- either because nothing has been analysed yet, or because the
    /// analyzer has no scorer at all (see [`HealthTracker::recent_scores`]).
    pub score: Option<f32>,
    /// Status derived from [`Self::score`]; `None` whenever the score is.
    pub status: Option<HealthStatus>,
    pub trend: String,
    pub key_metrics: HashMap<String, f32>,
    pub issues: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    Excellent,
    Good,
    Fair,
    Poor,
    Critical,
}

/// Real [`HealthStatus`] bucket for a score on the conventional 0.0-1.0
/// (higher-is-healthier) scale. Shared by every `get_health_summary` in
/// this module so `status` is always a genuine function of the tracked
/// score -- never a hardcoded `HealthStatus::Good`.
fn health_status_from_score(score: f32) -> HealthStatus {
    if score >= 0.9 {
        HealthStatus::Excellent
    } else if score >= 0.75 {
        HealthStatus::Good
    } else if score >= 0.5 {
        HealthStatus::Fair
    } else if score >= 0.25 {
        HealthStatus::Poor
    } else {
        HealthStatus::Critical
    }
}

/// How many recent scores [`HealthTracker`] keeps for trend analysis.
const HEALTH_TREND_WINDOW: usize = 20;

/// Tracks a bounded window of real, per-call analysis scores so
/// `get_health_summary` can derive a real [`HealthStatus`] and trend
/// direction from actual history, instead of the old hardcoded
/// `HealthStatus::Good` / `trend: "Stable".to_string()` that never changed
/// no matter what was analyzed.
///
/// Scores must be on the conventional 0.0 (worst) - 1.0 (best) scale;
/// callers whose native metric is inverted (e.g. a bias score where lower
/// is better) should record `1.0 - raw_score`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthTracker {
    /// The last [`HEALTH_TREND_WINDOW`] scores recorded via [`Self::record`],
    /// oldest first.
    ///
    /// Empty means *nothing has been analysed yet*, which every accessor
    /// reports as `None`. There is deliberately no seed value: an analyzer
    /// whose scorer does not exist (see [`AlignmentMonitor`], [`BiasDetector`],
    /// [`ConversationAnalyzer`]) never records anything, and a seed would make
    /// its health summary publish that seed forever as if it had been measured.
    recent_scores: VecDeque<f32>,
}

impl HealthTracker {
    fn new() -> Self {
        Self {
            recent_scores: VecDeque::with_capacity(HEALTH_TREND_WINDOW),
        }
    }

    /// Record one real, freshly-computed score into the bounded window.
    fn record(&mut self, score: f32) {
        self.recent_scores.push_back(score);
        while self.recent_scores.len() > HEALTH_TREND_WINDOW {
            self.recent_scores.pop_front();
        }
    }

    /// Average of the recorded window; `None` until a real score has been
    /// recorded.
    fn average_score(&self) -> Option<f32> {
        if self.recent_scores.is_empty() {
            None
        } else {
            Some(self.recent_scores.iter().sum::<f32>() / self.recent_scores.len() as f32)
        }
    }

    /// Real [`HealthStatus`] derived from [`Self::average_score`]; `None`
    /// until a real score has been recorded.
    fn status(&self) -> Option<HealthStatus> {
        self.average_score().map(health_status_from_score)
    }

    /// Real trend label: splits the recorded window in half and compares
    /// the mean of the newer half against the mean of the older half.
    /// `"Unknown (insufficient history)"` -- never a fabricated `"Stable"`
    /// -- until at least two scores have been recorded. A window that is
    /// genuinely flat (every recorded score equal) is honestly `"Stable"`,
    /// not `"Unknown"`.
    fn trend_label(&self) -> String {
        if self.recent_scores.len() < 2 {
            return "Unknown (insufficient history)".to_string();
        }
        let mid = self.recent_scores.len() / 2;
        let older_avg: f32 = self.recent_scores.iter().take(mid).sum::<f32>() / mid as f32;
        let newer_count = self.recent_scores.len() - mid;
        let newer_avg: f32 = self.recent_scores.iter().skip(mid).sum::<f32>() / newer_count as f32;

        const EPSILON: f32 = 0.02;
        let delta = newer_avg - older_avg;
        if delta > EPSILON {
            "Improving".to_string()
        } else if delta < -EPSILON {
            "Declining".to_string()
        } else {
            "Stable".to_string()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticalIssue {
    pub category: IssueCategory,
    pub severity: IssueSeverity,
    pub description: String,
    pub recommended_action: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueCategory {
    Safety,
    Factuality,
    Alignment,
    Bias,
    Performance,
    Conversation,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

// Individual analysis result types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyAnalysisResult {
    pub safety_score: f32,
    pub detected_harms: Vec<HarmCategory>,
    pub risk_level: RiskLevel,
    pub flagged_content: Vec<String>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactualityAnalysisResult {
    pub factuality_score: f32,
    pub verified_claims: usize,
    pub unverified_claims: usize,
    pub confidence_scores: Vec<f32>,
    pub knowledge_gaps: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentAnalysisResult {
    /// Overall alignment score, or `None` when no scorer is available.
    ///
    /// Always `None` from [`AlignmentMonitor::check_alignment`]: scoring
    /// alignment requires a policy/preference model, and this crate ships
    /// none. It used to be the constant `0.85`.
    pub alignment_score: Option<f32>,
    /// Per-objective scores; empty for the same reason as
    /// [`Self::alignment_score`] (previously the constants 0.9/0.95/0.8/0.85).
    pub objective_scores: HashMap<AlignmentObjective, f32>,
    /// Concrete alignment violations found. Always empty here: no violation
    /// detector exists.
    pub violations: Vec<String>,
    /// Consistency between input and response; `None` here (previously the
    /// constant `0.9`).
    pub consistency_score: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HallucinationAnalysisResult {
    /// Crude lexical hedging signal, not a calibrated probability -- see
    /// [`HallucinationDetector::hedging_signal`].
    pub hedging_signal: f32,
    /// How well the response's stated confidence matches its accuracy.
    ///
    /// Always `None`: measuring it needs ground truth for the claims, which
    /// this crate never receives. Previously the constant `0.7`.
    pub confidence_accuracy: Option<f32>,
    /// Real internal-consistency score from
    /// [`ConsistencyChecker::check_consistency`].
    pub internal_consistency: f32,
    /// Concrete fabricated statements found. Always empty: no fact-checking
    /// backend exists.
    pub detected_fabrications: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiasAnalysisResult {
    /// Overall bias score, or `None` when no scorer is available.
    ///
    /// Always `None` from [`BiasDetector::detect_bias`]: real bias detection
    /// needs demographic-term and stereotype models this crate does not have.
    /// It used to be the constant `0.1`.
    pub overall_bias_score: Option<f32>,
    /// Per-category bias scores; empty for the same reason (previously the
    /// constants Gender 0.1 / Race 0.05 / Religion 0.08).
    pub bias_categories: HashMap<BiasCategory, f32>,
    /// Concrete biased statements found. Always empty: no detector exists.
    pub detected_biases: Vec<String>,
    /// Concrete fairness violations found. Always empty: no detector exists.
    pub fairness_violations: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAnalysisResult {
    pub generation_metrics: GenerationMetrics,
    pub efficiency_metrics: EfficiencyMetrics,
    pub quality_metrics: QualityMetrics,
    pub bottlenecks: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationAnalysisResult {
    pub dialog_metrics: DialogMetrics,
    /// Consistency of this turn with the conversation context; `None` --
    /// dialog-quality scoring needs a trained model this crate does not have.
    /// Previously the constant `0.85`.
    pub context_consistency: Option<f32>,
    /// Quality of this turn; `None` for the same reason (previously `0.9`).
    pub turn_quality: Option<f32>,
    /// Engagement level; `None` for the same reason (previously `0.8`).
    pub engagement_score: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Rule-based, keyword-heuristic implementations. See the module docs and
/// [`HealthTracker`]: every score below is a deterministic function of the
/// literal response text (there is no trained classifier or knowledge base
/// behind this), and `confidence` is always derived from a real, measurable
/// property of the match (never a flat literal).
impl SafetyAnalyzer {
    pub fn new(_config: &LLMDebugConfig) -> Self {
        Self {
            toxic_patterns: HashSet::new(),
            harm_categories: vec![
                HarmCategory::Toxicity,
                HarmCategory::Violence,
                HarmCategory::SelfHarm,
                HarmCategory::Harassment,
                HarmCategory::HateSpeech,
            ],
            safety_metrics: SafetyMetrics {
                overall_safety_score: 1.0,
                harm_category_scores: HashMap::new(),
                flagged_responses: 0,
                total_responses_analyzed: 0,
                average_response_safety: 1.0,
                safety_trend: SafetyTrend::Stable,
            },
            health: HealthTracker::new(),
        }
    }

    pub async fn analyze_safety(&mut self, response: &str) -> Result<SafetyAnalysisResult> {
        let matched_keywords = self.find_harmful_keywords(response);
        let safety_score = Self::score_from_matches(matched_keywords.len());
        let detected_harms = self.detect_harmful_content(response);
        let risk_level = self.assess_risk_level(safety_score);

        self.safety_metrics.total_responses_analyzed += 1;
        if safety_score < 0.8 {
            self.safety_metrics.flagged_responses += 1;
        }
        // Real running history feeds `overall_safety_score` /
        // `average_response_safety` -- these used to be frozen at their
        // `new()` defaults forever, since nothing ever wrote back to them.
        self.health.record(safety_score);
        if let Some(average) = self.health.average_score() {
            self.safety_metrics.overall_safety_score = average;
            self.safety_metrics.average_response_safety = average;
        }
        self.safety_metrics.safety_trend = match self.health.trend_label().as_str() {
            "Improving" => SafetyTrend::Improving,
            "Declining" => SafetyTrend::Degrading,
            _ => SafetyTrend::Stable,
        };

        // Real confidence, derived from how many distinct harmful-keyword
        // categories were matched: zero matches is weaker evidence of
        // actual safety than a clean, unambiguous multi-keyword hit is
        // evidence of harm -- never the old flat `0.85` regardless of
        // content.
        let confidence = match matched_keywords.len() {
            0 => 0.6,
            1 => 0.75,
            _ => 0.9,
        };

        Ok(SafetyAnalysisResult {
            safety_score,
            detected_harms,
            risk_level,
            // The actual keywords this rule matched -- never the old
            // hardcoded empty `vec![]` regardless of what was found.
            flagged_content: matched_keywords.into_iter().map(str::to_string).collect(),
            confidence,
        })
    }

    /// The fixed keyword list this rule-based heuristic checks for. Public
    /// visibility of the list itself (via [`Self::find_harmful_keywords`])
    /// is intentional: callers should be able to see exactly what this
    /// heuristic does and does not catch, rather than trusting an opaque
    /// "AI" judgment.
    fn find_harmful_keywords(&self, response: &str) -> Vec<&'static str> {
        const HARMFUL_KEYWORDS: [&str; 4] = ["violence", "harm", "toxic", "hate"];
        let lower = response.to_lowercase();
        HARMFUL_KEYWORDS
            .iter()
            .copied()
            .filter(|keyword| lower.contains(keyword))
            .collect()
    }

    /// Real function of the match count (more distinct harmful-keyword
    /// categories -> lower/worse score), not a two-way literal switch.
    fn score_from_matches(match_count: usize) -> f32 {
        match match_count {
            0 => 0.95,
            1 => 0.5,
            _ => 0.2,
        }
    }

    fn compute_safety_score(&self, response: &str) -> f32 {
        Self::score_from_matches(self.find_harmful_keywords(response).len())
    }

    fn detect_harmful_content(&self, response: &str) -> Vec<HarmCategory> {
        // Rule-based mapping from the same keyword list `compute_safety_score`
        // checks. "harm" is intentionally left unmapped: it is too generic
        // to safely categorize (e.g. it could mean self-harm, harassment, or
        // neither) without producing misleading category labels.
        let lower = response.to_lowercase();
        let mut detected = Vec::new();

        if lower.contains("violence") {
            detected.push(HarmCategory::Violence);
        }
        if lower.contains("toxic") {
            detected.push(HarmCategory::Toxicity);
        }
        if lower.contains("hate") {
            detected.push(HarmCategory::HateSpeech);
        }

        detected
    }

    fn assess_risk_level(&self, safety_score: f32) -> RiskLevel {
        if safety_score >= 0.9 {
            RiskLevel::Low
        } else if safety_score >= 0.7 {
            RiskLevel::Medium
        } else if safety_score >= 0.5 {
            RiskLevel::High
        } else {
            RiskLevel::Critical
        }
    }

    /// Real health summary derived from [`Self::health`]'s running history
    /// of [`Self::analyze_safety`] calls -- `status` and `trend` used to be
    /// hardcoded (`trend` via a `safety_trend` field that was set once at
    /// construction and never updated).
    pub fn get_health_summary(&self) -> HealthSummary {
        HealthSummary {
            score: self.health.average_score(),
            status: self.health.status(),
            trend: self.health.trend_label(),
            key_metrics: HashMap::new(),
            issues: vec![],
        }
    }
}

impl FactualityChecker {
    pub fn new(_config: &LLMDebugConfig) -> Self {
        Self {
            fact_databases: vec!["wikipedia".to_string(), "wikidata".to_string()],
            uncertainty_indicators: ["might", "possibly", "unclear", "uncertain"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            factuality_metrics: FactualityMetrics {
                overall_factuality_score: 0.8,
                verified_facts: 0,
                unverified_claims: 0,
                conflicting_information: 0,
                uncertainty_expressions: 0,
                knowledge_gaps: vec![],
                confidence_distribution: vec![],
            },
            health: HealthTracker::new(),
        }
    }

    pub async fn check_factuality(
        &mut self,
        response: &str,
        _context: Option<&[String]>,
    ) -> Result<FactualityAnalysisResult> {
        let factuality_score = self.compute_factuality_score(response);
        let verified_claims = self.count_verified_claims(response);
        let unverified_claims = self.count_unverified_claims(response);

        self.factuality_metrics.verified_facts += verified_claims;
        self.factuality_metrics.unverified_claims += unverified_claims;
        // Real running history feeds `overall_factuality_score`, which used
        // to be frozen at its `new()` default (0.8) forever.
        self.health.record(factuality_score);
        if let Some(average) = self.health.average_score() {
            self.factuality_metrics.overall_factuality_score = average;
        }

        Ok(FactualityAnalysisResult {
            factuality_score,
            verified_claims,
            unverified_claims,
            // One real per-claim confidence value (see
            // `compute_claim_confidence_scores`), not the old fixed
            // 3-element `[0.8, 0.7, 0.9]` "Mock scores" regardless of how
            // many claims were actually found.
            confidence_scores: self.compute_claim_confidence_scores(response),
            // The actual sentences containing an uncertainty indicator, not
            // the old hardcoded empty `vec![]`.
            knowledge_gaps: self.extract_knowledge_gaps(response),
        })
    }

    /// Rule-based heuristic (see module docs): docks the base score for
    /// each uncertainty-indicator occurrence found, rather than the old
    /// two-way `contains("fact")` switch that ignored uncertainty entirely.
    fn compute_factuality_score(&self, response: &str) -> f32 {
        let uncertainty_hits = self.count_unverified_claims(response);
        let base: f32 = if response.contains("fact") { 0.9 } else { 0.7 };
        (base - 0.05 * uncertainty_hits as f32).max(0.1)
    }

    fn count_verified_claims(&self, response: &str) -> usize {
        // Simplified counting - would extract and verify actual claims
        response.split('.').filter(|s| s.len() > 10).count()
    }

    fn count_unverified_claims(&self, response: &str) -> usize {
        // Simplified counting - would identify unverifiable claims
        self.uncertainty_indicators
            .iter()
            .map(|indicator| response.matches(indicator).count())
            .sum()
    }

    /// One real confidence value per sentence [`Self::count_verified_claims`]
    /// treats as a "claim" (same `split('.')` + length-10 filter): lower for
    /// sentences that also contain an uncertainty indicator, higher for
    /// those that don't. Always exactly as long as `verified_claims` --
    /// never the old fixed 3-element `[0.8, 0.7, 0.9]`.
    fn compute_claim_confidence_scores(&self, response: &str) -> Vec<f32> {
        response
            .split('.')
            .filter(|claim| claim.len() > 10)
            .map(|claim| {
                let lower = claim.to_lowercase();
                let has_uncertainty =
                    self.uncertainty_indicators.iter().any(|ind| lower.contains(ind.as_str()));
                if has_uncertainty {
                    0.5
                } else {
                    0.85
                }
            })
            .collect()
    }

    /// The actual sentences containing an uncertainty indicator -- a real
    /// (if crude) extraction, not the old hardcoded empty `vec![]`.
    fn extract_knowledge_gaps(&self, response: &str) -> Vec<String> {
        response
            .split('.')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .filter(|s| {
                let lower = s.to_lowercase();
                self.uncertainty_indicators.iter().any(|ind| lower.contains(ind.as_str()))
            })
            .map(str::to_string)
            .collect()
    }

    /// Real health summary derived from [`Self::health`]'s running
    /// history -- `status`/`trend` used to be hardcoded to
    /// `HealthStatus::Good` / `"Stable"` regardless of any actual score.
    pub fn get_health_summary(&self) -> HealthSummary {
        HealthSummary {
            score: self.health.average_score(),
            status: self.health.status(),
            trend: self.health.trend_label(),
            key_metrics: HashMap::new(),
            issues: vec![],
        }
    }
}

impl AlignmentMonitor {
    pub fn new(_config: &LLMDebugConfig) -> Self {
        Self {
            alignment_objectives: vec![
                AlignmentObjective::Helpfulness,
                AlignmentObjective::Harmlessness,
                AlignmentObjective::Honesty,
                AlignmentObjective::Fairness,
            ],
            alignment_metrics: AlignmentMetrics {
                objective_scores: HashMap::new(),
                overall_alignment_score: 0.85,
                alignment_violations: 0,
                value_consistency_score: 0.9,
                behavioral_drift: 0.1,
                alignment_trend: AlignmentTrend::Stable,
            },
            value_alignment_score: 0.85,
            health: HealthTracker::new(),
        }
    }

    pub async fn check_alignment(
        &mut self,
        input: &str,
        response: &str,
    ) -> Result<AlignmentAnalysisResult> {
        // No score is computable, so nothing is recorded into `health` and
        // `alignment_metrics` keeps whatever a caller set. Recording a
        // constant would have made `overall_alignment_score` converge to that
        // constant no matter what was analysed.
        let _ = (input, response);

        Ok(AlignmentAnalysisResult {
            alignment_score: None,
            objective_scores: HashMap::new(),
            violations: Vec::new(),
            consistency_score: None,
        })
    }

    /// Real health summary derived from [`Self::health`]'s running
    /// history -- see [`SafetyAnalyzer::get_health_summary`].
    pub fn get_health_summary(&self) -> HealthSummary {
        HealthSummary {
            score: self.health.average_score(),
            status: self.health.status(),
            trend: self.health.trend_label(),
            key_metrics: HashMap::new(),
            issues: vec![],
        }
    }
}

impl HallucinationDetector {
    pub fn new(_config: &LLMDebugConfig) -> Self {
        Self {
            confidence_thresholds: HashMap::new(),
            consistency_checker: ConsistencyChecker {
                previous_responses: Vec::new(),
                consistency_cache: HashMap::new(),
            },
            hallucination_metrics: HallucinationMetrics {
                hallucination_rate: 0.1,
                confidence_accuracy_correlation: 0.7,
                factual_consistency_score: 0.8,
                internal_consistency_score: 0.85,
                source_attribution_accuracy: 0.9,
                detected_fabrications: 0,
                uncertain_responses: 0,
            },
        }
    }

    pub async fn detect_hallucinations(
        &mut self,
        response: &str,
        _context: Option<&[String]>,
    ) -> Result<HallucinationAnalysisResult> {
        let internal_consistency = self.consistency_checker.check_consistency(response);

        Ok(HallucinationAnalysisResult {
            hedging_signal: Self::hedging_signal(response),
            confidence_accuracy: None,
            internal_consistency,
            detected_fabrications: Vec::new(),
        })
    }

    /// Fraction of the crate's hedging phrases (`HEDGING_PHRASES`) that appear
    /// in `response`, in `[0, 1]`.
    ///
    /// A lexical surface signal only: hedging correlates with a model
    /// expressing uncertainty, but this measures the WORDS, not whether
    /// anything is actually fabricated. It replaces
    /// `compute_hallucination_probability`, which returned `0.2` if the
    /// response contained the literal string `"I'm not sure"` and `0.1`
    /// otherwise -- two constants published under the name "probability".
    pub fn hedging_signal(response: &str) -> f32 {
        /// Phrases a model uses when expressing uncertainty.
        const HEDGING_PHRASES: &[&str] = &[
            "i'm not sure",
            "i am not sure",
            "i think",
            "i believe",
            "possibly",
            "might be",
            "as far as i know",
            "if i recall",
            "i'm not certain",
            "cannot verify",
        ];
        let lowered = response.to_lowercase();
        let hits = HEDGING_PHRASES.iter().filter(|p| lowered.contains(**p)).count();
        hits as f32 / HEDGING_PHRASES.len() as f32
    }
}

impl ConsistencyChecker {
    pub fn check_consistency(&mut self, response: &str) -> f32 {
        self.previous_responses.push(response.to_string());
        // Simplified consistency checking
        0.85
    }
}

impl BiasDetector {
    pub fn new(_config: &LLMDebugConfig) -> Self {
        Self {
            bias_categories: vec![
                BiasCategory::Gender,
                BiasCategory::Race,
                BiasCategory::Religion,
                BiasCategory::Age,
            ],
            demographic_groups: vec![
                "male".to_string(),
                "female".to_string(),
                "young".to_string(),
                "elderly".to_string(),
            ],
            bias_metrics: BiasMetrics {
                overall_bias_score: 0.1, // Lower is better for bias
                bias_category_scores: HashMap::new(),
                demographic_fairness: HashMap::new(),
                representation_bias: 0.1,
                stereotype_propagation: 0.05,
                bias_amplification: 0.08,
                fairness_violations: 0,
            },
            // `BiasDetector` has no bias scorer, so nothing is ever recorded
            // here and the health summary is honestly absent.
            health: HealthTracker::new(),
        }
    }

    pub async fn detect_bias(&mut self, response: &str) -> Result<BiasAnalysisResult> {
        // No bias score is computable, so nothing is recorded into `health`
        // and `bias_metrics` keeps whatever a caller set. Recording the old
        // constant made `overall_bias_score` converge to 0.1 for every text.
        let _ = response;

        Ok(BiasAnalysisResult {
            overall_bias_score: None,
            bias_categories: HashMap::new(),
            detected_biases: Vec::new(),
            fairness_violations: Vec::new(),
        })
    }

    /// Real health summary derived from [`Self::health`]'s running
    /// history -- see [`SafetyAnalyzer::get_health_summary`].
    pub fn get_health_summary(&self) -> HealthSummary {
        HealthSummary {
            score: self.health.average_score(),
            status: self.health.status(),
            trend: self.health.trend_label(),
            key_metrics: HashMap::new(),
            issues: vec![],
        }
    }
}

impl Default for LLMPerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl LLMPerformanceProfiler {
    pub fn new() -> Self {
        Self {
            generation_metrics: GenerationMetrics {
                tokens_per_second: 100.0,
                average_response_length: 150.0,
                generation_latency_p50: 200.0,
                generation_latency_p95: 500.0,
                generation_latency_p99: 1000.0,
                first_token_latency: 50.0,
                completion_rate: 0.98,
                timeout_rate: 0.02,
            },
            efficiency_metrics: EfficiencyMetrics {
                memory_efficiency: 0.85,
                compute_utilization: 0.75,
                energy_consumption: 0.5,        // kWh per 1000 tokens
                carbon_footprint_estimate: 0.1, // kg CO2 per 1000 tokens
                cost_per_token: 0.001,          // USD per token
                batch_processing_efficiency: 0.9,
                cache_hit_rate: 0.7,
            },
            quality_metrics: QualityMetrics {
                coherence_score: 0.9,
                relevance_score: 0.85,
                fluency_score: 0.95,
                informativeness_score: 0.8,
                creativity_score: 0.7,
                factual_accuracy: 0.85,
                readability_score: 0.9,
                engagement_score: 0.8,
            },
            scalability_metrics: ScalabilityMetrics {
                concurrent_user_capacity: 1000,
                throughput_scaling: 0.8,
                memory_scaling: 0.7,
                latency_degradation: 0.1,
                bottleneck_analysis: vec!["Memory bandwidth".to_string()],
                resource_utilization_efficiency: 0.8,
            },
            health: HealthTracker::new(),
        }
    }

    pub async fn profile_response(
        &mut self,
        _response: &str,
        generation_metrics: Option<GenerationMetrics>,
    ) -> Result<PerformanceAnalysisResult> {
        let gen_metrics = generation_metrics.unwrap_or_else(|| self.generation_metrics.clone());

        // Real running history from this call's real throughput, feeding
        // `get_health_summary` -- which used to read `self.generation_metrics`
        // directly and was therefore frozen at the `new()` default whenever
        // a caller supplied its own `generation_metrics` (as most real
        // callers would).
        self.health.record((gen_metrics.tokens_per_second / 200.0).min(1.0));

        Ok(PerformanceAnalysisResult {
            generation_metrics: gen_metrics,
            efficiency_metrics: self.efficiency_metrics.clone(),
            quality_metrics: self.quality_metrics.clone(),
            // No bottleneck attribution exists: the profiler records aggregate
            // throughput, never a per-stage breakdown to rank.
            bottlenecks: Vec::new(),
        })
    }

    /// Real health summary derived from [`Self::health`]'s running
    /// history -- see [`SafetyAnalyzer::get_health_summary`].
    pub fn get_health_summary(&self) -> HealthSummary {
        HealthSummary {
            score: self.health.average_score(),
            status: self.health.status(),
            trend: self.health.trend_label(),
            key_metrics: HashMap::new(),
            issues: vec![],
        }
    }
}

impl ConversationAnalyzer {
    pub fn new(_config: &LLMDebugConfig) -> Self {
        Self {
            conversation_history: Vec::new(),
            dialog_metrics: DialogMetrics {
                conversation_coherence: 0.9,
                context_maintenance: 0.85,
                topic_consistency: 0.8,
                response_appropriateness: 0.9,
                conversation_engagement: 0.75,
                turn_taking_naturalness: 0.8,
                memory_utilization: 0.7,
                dialog_success_rate: 0.85,
            },
            context_tracking: ContextTracker {
                active_topics: HashSet::new(),
                entity_mentions: HashMap::new(),
                context_window: Vec::new(),
                attention_weights: Vec::new(),
            },
            health: HealthTracker::new(),
        }
    }

    pub async fn analyze_turn(
        &mut self,
        turn: &ConversationTurn,
    ) -> Result<ConversationAnalysisResult> {
        self.conversation_history.push(turn.clone());
        self.context_tracking.update_from_turn(turn);
        // No dialog-quality score is computable, so nothing is recorded into
        // `health`. The turn itself IS recorded above, so
        // `conversation_history` and `context_tracking` stay real.

        Ok(ConversationAnalysisResult {
            dialog_metrics: self.dialog_metrics.clone(),
            context_consistency: None,
            turn_quality: None,
            engagement_score: None,
        })
    }

    /// Real health summary derived from [`Self::health`]'s running
    /// history -- see [`SafetyAnalyzer::get_health_summary`].
    pub fn get_health_summary(&self) -> HealthSummary {
        HealthSummary {
            score: self.health.average_score(),
            status: self.health.status(),
            trend: self.health.trend_label(),
            key_metrics: HashMap::new(),
            issues: vec![],
        }
    }
}

impl ContextTracker {
    pub fn update_from_turn(&mut self, turn: &ConversationTurn) {
        // Update context tracking based on the turn
        self.context_window.push(turn.model_response.clone());
        if self.context_window.len() > 10 {
            self.context_window.remove(0);
        }
    }
}

/// Convenience macros for LLM debugging
#[macro_export]
macro_rules! debug_llm_response {
    ($debugger:expr, $input:expr, $response:expr) => {
        $debugger.analyze_response($input, $response, None, None).await
    };
}

#[macro_export]
macro_rules! debug_llm_batch {
    ($debugger:expr, $interactions:expr) => {
        $debugger.analyze_batch($interactions).await
    };
}

/// Create a new LLM debugger with default configuration
pub fn llm_debugger() -> LLMDebugger {
    LLMDebugger::new(LLMDebugConfig::default())
}

/// Create a new LLM debugger with custom configuration
pub fn llm_debugger_with_config(config: LLMDebugConfig) -> LLMDebugger {
    LLMDebugger::new(config)
}

/// Create a safety-focused LLM debugger configuration
pub fn safety_focused_config() -> LLMDebugConfig {
    LLMDebugConfig {
        enable_safety_analysis: true,
        enable_factuality_checking: true,
        enable_alignment_monitoring: true,
        enable_hallucination_detection: true,
        enable_bias_detection: true,
        enable_llm_performance_profiling: false,
        enable_conversation_analysis: false,
        safety_threshold: 0.9,
        factuality_threshold: 0.8,
        max_conversation_length: 50,
        analysis_sampling_rate: 1.0,
    }
}

/// Create a performance-focused LLM debugger configuration
pub fn performance_focused_config() -> LLMDebugConfig {
    LLMDebugConfig {
        enable_safety_analysis: false,
        enable_factuality_checking: false,
        enable_alignment_monitoring: false,
        enable_hallucination_detection: false,
        enable_bias_detection: false,
        enable_llm_performance_profiling: true,
        enable_conversation_analysis: true,
        safety_threshold: 0.7,
        factuality_threshold: 0.6,
        max_conversation_length: 200,
        analysis_sampling_rate: 0.1,
    }
}

#[cfg(test)]
#[path = "llm_debugging_tests.rs"]
mod llm_debugging_tests;

/// Core unit tests for LLM debugging functionality. Split into a
/// separate file (`llm_debugging_tests2.rs`) to keep this file under
/// the 2000-line policy limit.
#[cfg(test)]
#[path = "llm_debugging_tests2.rs"]
mod tests;
