//! Advanced Vocabulary Intelligence System for TrustformeRS Tokenizers
//!
//! This module provides sophisticated analysis and intelligence capabilities
//! for tokenizer vocabularies including semantic analysis, efficiency optimization,
//! and advanced pattern recognition.

use crate::vocab_analyzer::VocabAnalysisResult;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};

/// Configuration for advanced vocabulary intelligence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabIntelligenceConfig {
    /// Enable semantic similarity analysis
    pub enable_semantic_analysis: bool,
    /// Enable compression efficiency analysis
    pub enable_compression_analysis: bool,
    /// Enable cross-lingual analysis
    pub enable_cross_lingual_analysis: bool,
    /// Enable domain adaptation analysis
    pub enable_domain_analysis: bool,
    /// Enable vocabulary evolution tracking
    pub enable_evolution_tracking: bool,
    /// Enable subword efficiency analysis
    pub enable_subword_efficiency: bool,
    /// Enable token frequency prediction
    pub enable_frequency_prediction: bool,
    /// Enable vocabulary optimization suggestions
    pub enable_optimization_suggestions: bool,
    /// Minimum similarity threshold for clustering
    pub similarity_threshold: f32,
    /// Languages for cross-lingual analysis
    pub target_languages: Vec<String>,
    /// Domains for domain-specific analysis
    pub target_domains: Vec<String>,
    /// History length for evolution tracking
    pub evolution_history_length: usize,
}

impl Default for VocabIntelligenceConfig {
    fn default() -> Self {
        Self {
            enable_semantic_analysis: true,
            enable_compression_analysis: true,
            enable_cross_lingual_analysis: true,
            enable_domain_analysis: true,
            enable_evolution_tracking: true,
            enable_subword_efficiency: true,
            enable_frequency_prediction: true,
            enable_optimization_suggestions: true,
            similarity_threshold: 0.8,
            target_languages: vec![
                "en".to_string(),
                "es".to_string(),
                "fr".to_string(),
                "de".to_string(),
            ],
            target_domains: vec![
                "general".to_string(),
                "scientific".to_string(),
                "technical".to_string(),
            ],
            evolution_history_length: 100,
        }
    }
}

/// Semantic similarity analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticAnalysis {
    /// Token clusters based on semantic similarity
    pub semantic_clusters: Vec<SemanticCluster>,
    /// Redundant tokens that could be merged
    pub redundant_tokens: Vec<RedundantTokenGroup>,
    /// Tokens with high semantic diversity
    pub diverse_tokens: Vec<String>,
    /// Semantic coverage score (0-100)
    pub semantic_coverage_score: f32,
    /// Average inter-cluster distance
    pub average_cluster_distance: f32,
    /// Vocabulary coherence score
    pub coherence_score: f32,
}

/// Semantic cluster of related tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCluster {
    /// Cluster identifier
    pub cluster_id: String,
    /// Tokens in this cluster
    pub tokens: Vec<String>,
    /// Cluster centroid (representative concept)
    pub centroid: String,
    /// Intra-cluster similarity score
    pub cohesion_score: f32,
    /// Semantic theme/category
    pub semantic_theme: String,
    /// Frequency weight of cluster
    pub frequency_weight: f32,
}

/// Group of redundant tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedundantTokenGroup {
    /// Primary token (most frequent or representative)
    pub primary_token: String,
    /// Alternative tokens that could be merged
    pub alternative_tokens: Vec<String>,
    /// Similarity scores with primary token
    pub similarity_scores: Vec<f32>,
    /// Estimated compression benefit
    pub compression_benefit: f32,
    /// Risk assessment for merging
    pub merge_risk: MergeRisk,
}

/// Risk assessment for token merging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MergeRisk {
    Low,      // Safe to merge
    Medium,   // Moderate risk, review needed
    High,     // High risk, careful consideration needed
    Critical, // Should not merge
}

/// Compression efficiency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionAnalysis {
    /// Current compression ratio
    pub current_compression_ratio: f32,
    /// Theoretical optimal compression
    pub optimal_compression_ratio: f32,
    /// Compression efficiency score (0-100)
    pub efficiency_score: f32,
    /// Subword decomposition efficiency
    pub subword_efficiency: SubwordEfficiency,
    /// Token length distribution analysis
    pub length_distribution: LengthDistributionAnalysis,
    /// Frequency-based compression opportunities
    pub frequency_opportunities: Vec<CompressionOpportunity>,
}

/// Subword decomposition efficiency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubwordEfficiency {
    /// Average subword decomposition length
    pub average_decomposition_length: f32,
    /// Percentage of efficient decompositions
    pub efficient_decompositions_percent: f32,
    /// Over-segmented tokens (too many subwords)
    pub over_segmented_tokens: Vec<String>,
    /// Under-segmented tokens (could be split more)
    pub under_segmented_tokens: Vec<String>,
    /// Optimal subword length distribution
    pub optimal_length_distribution: BTreeMap<usize, f32>,
}

/// Token length distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LengthDistributionAnalysis {
    /// Current length distribution
    pub current_distribution: BTreeMap<usize, usize>,
    /// Optimal length distribution
    pub optimal_distribution: BTreeMap<usize, usize>,
    /// Length efficiency score
    pub efficiency_score: f32,
    /// Tokens that are too long
    pub overlong_tokens: Vec<String>,
    /// Tokens that are too short
    pub underlong_tokens: Vec<String>,
}

/// Compression opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionOpportunity {
    /// Type of opportunity
    pub opportunity_type: CompressionOpportunityType,
    /// Affected tokens
    pub affected_tokens: Vec<String>,
    /// Estimated compression improvement
    pub compression_improvement_percent: f32,
    /// Implementation difficulty
    pub implementation_difficulty: ImplementationDifficulty,
    /// Description
    pub description: String,
}

/// Type of compression opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionOpportunityType {
    TokenMerging,
    SubwordOptimization,
    FrequencyRebalancing,
    LengthOptimization,
    RedundancyElimination,
    PrefixOptimization,
    SuffixOptimization,
}

/// Implementation difficulty levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationDifficulty {
    Trivial, // Automatic optimization
    Easy,    // Simple configuration change
    Medium,  // Requires retraining
    Hard,    // Significant restructuring
    Expert,  // Requires domain expertise
}

/// Cross-lingual vocabulary analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossLingualAnalysis {
    /// Language coverage analysis
    pub language_coverage: HashMap<String, LanguageCoverage>,
    /// Cross-lingual token overlaps
    pub cross_lingual_overlaps: Vec<CrossLingualOverlap>,
    /// Language-specific efficiency scores
    pub language_efficiency_scores: HashMap<String, f32>,
    /// Multilingual optimization opportunities
    pub multilingual_opportunities: Vec<MultilingualOpportunity>,
    /// Language diversity score
    pub diversity_score: f32,
}

/// Coverage analysis for a specific language
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageCoverage {
    /// Language code
    pub language_code: String,
    /// Percentage of vocabulary dedicated to this language
    pub vocabulary_percentage: f32,
    /// Coverage efficiency for this language
    pub coverage_efficiency: f32,
    /// Common words missing from vocabulary
    pub missing_common_words: Vec<String>,
    /// Over-represented rare words
    pub over_represented_words: Vec<String>,
    /// Language-specific subword patterns
    pub subword_patterns: HashMap<String, f32>,
}

/// Cross-lingual token overlap analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossLingualOverlap {
    /// Languages involved in overlap
    pub languages: Vec<String>,
    /// Overlapping tokens
    pub overlapping_tokens: Vec<String>,
    /// Overlap efficiency score
    pub efficiency_score: f32,
    /// Optimization potential
    pub optimization_potential: f32,
}

/// Multilingual optimization opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultilingualOpportunity {
    /// Opportunity type
    pub opportunity_type: MultilingualOpportunityType,
    /// Affected languages
    pub affected_languages: Vec<String>,
    /// Potential improvement
    pub improvement_description: String,
    /// Expected efficiency gain
    pub efficiency_gain_percent: f32,
}

/// Type of multilingual optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MultilingualOpportunityType {
    SharedTokenOptimization,
    LanguageSpecificTuning,
    ScriptOptimization,
    CommonPrefixSharing,
    TransliterationNormalization,
}

/// Domain adaptation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainAnalysis {
    /// Domain-specific token distributions
    pub domain_distributions: HashMap<String, DomainDistribution>,
    /// Cross-domain token adaptability
    pub adaptability_scores: HashMap<String, f32>,
    /// Domain-specific optimization suggestions
    pub domain_optimizations: Vec<DomainOptimization>,
    /// Vocabulary domain coverage
    pub domain_coverage: f32,
}

/// Token distribution for a specific domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainDistribution {
    /// Domain identifier
    pub domain: String,
    /// Token frequency distribution in this domain
    pub token_frequencies: HashMap<String, f32>,
    /// Domain-specific efficiency score
    pub efficiency_score: f32,
    /// Important tokens missing for this domain
    pub missing_domain_tokens: Vec<String>,
    /// Overrepresented general tokens
    pub overrepresented_general_tokens: Vec<String>,
}

/// Domain-specific optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainOptimization {
    /// Target domain
    pub domain: String,
    /// Optimization type
    pub optimization_type: DomainOptimizationType,
    /// Description
    pub description: String,
    /// Expected improvement
    pub expected_improvement_percent: f32,
    /// Required tokens to add
    pub tokens_to_add: Vec<String>,
    /// Tokens to remove or reduce
    pub tokens_to_reduce: Vec<String>,
}

/// Type of domain optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DomainOptimizationType {
    DomainSpecificExpansion,
    GeneralTokenReduction,
    TerminologyOptimization,
    AbbreviationHandling,
    TechnicalJargonIntegration,
}

/// Vocabulary evolution tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionAnalysis {
    /// Evolution timeline
    pub evolution_timeline: Vec<EvolutionSnapshot>,
    /// Vocabulary stability score
    pub stability_score: f32,
    /// Trending tokens (gaining importance)
    pub trending_tokens: Vec<TrendingToken>,
    /// Declining tokens (losing importance)
    pub declining_tokens: Vec<DeclineToken>,
    /// Evolution predictions
    pub predictions: Vec<EvolutionPrediction>,
}

/// Snapshot of vocabulary state at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionSnapshot {
    /// Timestamp
    pub timestamp: u64,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Efficiency metrics at this time
    pub efficiency_metrics: EfficiencyMetrics,
    /// Major changes since last snapshot
    pub changes: Vec<VocabularyChange>,
}

/// Efficiency metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    /// Compression ratio
    pub compression_ratio: f32,
    /// Coverage efficiency
    pub coverage_efficiency: f32,
    /// Subword efficiency
    pub subword_efficiency: f32,
    /// Cross-lingual efficiency
    pub cross_lingual_efficiency: f32,
}

/// Vocabulary change event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabularyChange {
    /// Type of change
    pub change_type: ChangeType,
    /// Affected tokens
    pub affected_tokens: Vec<String>,
    /// Impact description
    pub impact_description: String,
    /// Performance impact
    pub performance_impact: f32,
}

/// Type of vocabulary change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    TokenAdded,
    TokenRemoved,
    TokenMerged,
    TokenSplit,
    FrequencyUpdated,
    DomainExpansion,
    LanguageAdded,
}

/// Token showing trending behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendingToken {
    /// Token text
    pub token: String,
    /// Trend strength (0-100)
    pub trend_strength: f32,
    /// Frequency change rate
    pub frequency_change_rate: f32,
    /// Predicted future importance
    pub predicted_importance: f32,
    /// Trend category
    pub trend_category: TrendCategory,
}

/// Token showing declining usage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeclineToken {
    /// Token text
    pub token: String,
    /// Decline rate
    pub decline_rate: f32,
    /// Predicted obsolescence time
    pub predicted_obsolescence_days: Option<u32>,
    /// Decline reason
    pub decline_reason: DeclineReason,
}

/// Category of trending behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendCategory {
    EmergingTechnology,
    PopularCulture,
    NewDomain,
    LanguageEvolution,
    SeasonalTrend,
    Unknown,
}

/// Reason for token decline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeclineReason {
    Obsolete,
    Superseded,
    DomainShift,
    LanguageChange,
    OverSegmentation,
    Unknown,
}

/// Evolution prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionPrediction {
    /// Prediction type
    pub prediction_type: PredictionType,
    /// Time horizon (days)
    pub time_horizon_days: u32,
    /// Confidence score (0-100)
    pub confidence_score: f32,
    /// Prediction description
    pub description: String,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
}

/// Type of evolution prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PredictionType {
    VocabularyGrowth,
    DomainExpansion,
    LanguageShift,
    EfficiencyDecline,
    CompressionOpportunity,
    OptimizationNeed,
}

/// Comprehensive vocabulary intelligence results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabIntelligenceResult {
    /// Basic analysis results
    pub basic_analysis: VocabAnalysisResult,
    /// Semantic analysis
    pub semantic_analysis: Option<SemanticAnalysis>,
    /// Compression analysis
    pub compression_analysis: Option<CompressionAnalysis>,
    /// Cross-lingual analysis
    pub cross_lingual_analysis: Option<CrossLingualAnalysis>,
    /// Domain analysis
    pub domain_analysis: Option<DomainAnalysis>,
    /// Evolution analysis
    pub evolution_analysis: Option<EvolutionAnalysis>,
    /// Overall intelligence score (0-100)
    pub intelligence_score: f32,
    /// Actionable recommendations
    pub actionable_recommendations: Vec<ActionableRecommendation>,
    /// Risk assessment
    pub risk_assessment: RiskAssessment,
}

/// Actionable recommendation for vocabulary improvement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionableRecommendation {
    /// Recommendation category
    pub category: RecommendationCategory,
    /// Priority level
    pub priority: RecommendationPriority,
    /// Title
    pub title: String,
    /// Detailed description
    pub description: String,
    /// Expected benefits
    pub expected_benefits: Vec<String>,
    /// Implementation steps
    pub implementation_steps: Vec<String>,
    /// Estimated effort
    pub effort_estimate: EffortEstimate,
    /// Risk factors
    pub risk_factors: Vec<String>,
    /// Success metrics
    pub success_metrics: Vec<String>,
}

/// Category of recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationCategory {
    Performance,
    Efficiency,
    Coverage,
    Maintenance,
    Expansion,
    Optimization,
    Modernization,
}

/// Priority level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    Critical,
    High,
    Medium,
    Low,
    Optional,
}

/// Effort estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffortEstimate {
    /// Estimated hours
    pub estimated_hours: f32,
    /// Complexity level
    pub complexity: ComplexityLevel,
    /// Required expertise
    pub required_expertise: Vec<ExpertiseArea>,
    /// Dependencies
    pub dependencies: Vec<String>,
}

/// Complexity level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Trivial,
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Area of expertise required
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpertiseArea {
    Tokenization,
    MachineLearning,
    Linguistics,
    DataScience,
    SoftwareEngineering,
    DomainExpertise,
}

/// Risk assessment for vocabulary changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskAssessment {
    /// Overall risk level
    pub overall_risk: RiskLevel,
    /// Specific risk factors
    pub risk_factors: Vec<RiskFactor>,
    /// Mitigation strategies
    pub mitigation_strategies: Vec<String>,
    /// Rollback plan complexity
    pub rollback_complexity: ComplexityLevel,
}

/// Risk level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    VeryLow,
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Specific risk factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    /// Risk type
    pub risk_type: RiskType,
    /// Probability (0-100)
    pub probability: f32,
    /// Impact (0-100)
    pub impact: f32,
    /// Description
    pub description: String,
}

/// Type of risk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskType {
    PerformanceDegradation,
    AccuracyLoss,
    CompatibilityBreaking,
    MaintenanceOverhead,
    CoverageReduction,
    ImplementationComplexity,
}

/// Advanced vocabulary intelligence analyzer
pub struct VocabIntelligenceAnalyzer {
    pub(super) config: VocabIntelligenceConfig,
    #[allow(dead_code)]
    pub(super) similarity_cache: HashMap<(String, String), f32>,
    pub(super) evolution_history: Vec<EvolutionSnapshot>,
}
