//! Advanced Vocabulary Intelligence System for TrustformeRS Tokenizers
//!
//! This module provides sophisticated analysis and intelligence capabilities
//! for tokenizer vocabularies including semantic analysis, efficiency optimization,
//! and advanced pattern recognition.

use crate::vocab_analyzer::VocabAnalysisResult;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use trustformers_core::errors::Result;
use trustformers_core::traits::Tokenizer;

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
    config: VocabIntelligenceConfig,
    #[allow(dead_code)]
    similarity_cache: HashMap<(String, String), f32>,
    evolution_history: Vec<EvolutionSnapshot>,
}

impl VocabIntelligenceAnalyzer {
    /// Create a new intelligence analyzer
    pub fn new(config: VocabIntelligenceConfig) -> Self {
        Self {
            config,
            similarity_cache: HashMap::new(),
            evolution_history: Vec::new(),
        }
    }

    /// Perform comprehensive vocabulary intelligence analysis
    pub fn analyze<T: Tokenizer>(
        &mut self,
        tokenizer: &T,
        basic_analysis: VocabAnalysisResult,
    ) -> Result<VocabIntelligenceResult> {
        let mut result = VocabIntelligenceResult {
            basic_analysis,
            semantic_analysis: None,
            compression_analysis: None,
            cross_lingual_analysis: None,
            domain_analysis: None,
            evolution_analysis: None,
            intelligence_score: 0.0,
            actionable_recommendations: Vec::new(),
            risk_assessment: RiskAssessment {
                overall_risk: RiskLevel::Low,
                risk_factors: Vec::new(),
                mitigation_strategies: Vec::new(),
                rollback_complexity: ComplexityLevel::Low,
            },
        };

        // Perform semantic analysis
        if self.config.enable_semantic_analysis {
            result.semantic_analysis = Some(self.analyze_semantic_patterns(tokenizer)?);
        }

        // Perform compression analysis
        if self.config.enable_compression_analysis {
            result.compression_analysis = Some(self.analyze_compression_efficiency(tokenizer)?);
        }

        // Perform cross-lingual analysis
        if self.config.enable_cross_lingual_analysis {
            result.cross_lingual_analysis = Some(self.analyze_cross_lingual_patterns(tokenizer)?);
        }

        // Perform domain analysis
        if self.config.enable_domain_analysis {
            result.domain_analysis = Some(self.analyze_domain_adaptation(tokenizer)?);
        }

        // Perform evolution analysis
        if self.config.enable_evolution_tracking {
            result.evolution_analysis = Some(self.analyze_vocabulary_evolution(tokenizer)?);
        }

        // Calculate overall intelligence score
        result.intelligence_score = self.calculate_intelligence_score(&result);

        // Generate actionable recommendations
        result.actionable_recommendations = self.generate_actionable_recommendations(&result);

        // Assess risks
        result.risk_assessment = self.assess_risks(&result);

        Ok(result)
    }

    /// Analyze semantic patterns and clustering
    fn analyze_semantic_patterns<T: Tokenizer>(
        &mut self,
        tokenizer: &T,
    ) -> Result<SemanticAnalysis> {
        let vocab = tokenizer.get_vocab();
        let mut clusters = Vec::new();
        let mut redundant_groups = Vec::new();
        let mut diverse_tokens = Vec::new();

        // Simple semantic clustering based on character patterns
        // In a real implementation, this would use embedding similarity
        let mut cluster_map: HashMap<String, Vec<String>> = HashMap::new();

        for (token, _) in vocab.iter() {
            let pattern = self.extract_semantic_pattern(token);
            cluster_map.entry(pattern).or_default().push(token.clone());
        }

        // Create semantic clusters
        for (pattern, tokens) in cluster_map {
            if tokens.len() > 1 {
                clusters.push(SemanticCluster {
                    cluster_id: format!("cluster_{}", clusters.len()),
                    centroid: pattern.clone(),
                    tokens: tokens.clone(),
                    cohesion_score: self.calculate_cluster_cohesion(&tokens),
                    semantic_theme: self.infer_semantic_theme(&pattern),
                    frequency_weight: self.calculate_frequency_weight(&tokens, tokenizer),
                });

                // Identify redundant tokens within cluster
                if tokens.len() > 3 && self.calculate_cluster_cohesion(&tokens) > 0.9 {
                    redundant_groups.push(RedundantTokenGroup {
                        primary_token: tokens[0].clone(),
                        alternative_tokens: tokens[1..].to_vec(),
                        similarity_scores: vec![0.9; tokens.len() - 1],
                        compression_benefit: tokens.len() as f32 * 0.1,
                        merge_risk: MergeRisk::Low,
                    });
                }
            } else if tokens.len() == 1 {
                diverse_tokens.push(tokens[0].clone());
            }
        }

        // Calculate actual semantic metrics
        let total_tokens = vocab.len() as f32;
        let clustered_tokens = clusters.iter().map(|c| c.tokens.len()).sum::<usize>() as f32;
        let semantic_coverage_score =
            if total_tokens > 0.0 { (clustered_tokens / total_tokens) * 100.0 } else { 0.0 };

        let average_cluster_distance = if !clusters.is_empty() {
            1.0 - clusters.iter().map(|c| c.cohesion_score).sum::<f32>() / clusters.len() as f32
        } else {
            0.5
        };

        let coherence_score = if !clusters.is_empty() {
            clusters.iter().map(|c| c.cohesion_score).sum::<f32>() / clusters.len() as f32 * 100.0
        } else {
            0.0
        };

        Ok(SemanticAnalysis {
            semantic_clusters: clusters,
            redundant_tokens: redundant_groups,
            diverse_tokens,
            semantic_coverage_score,
            average_cluster_distance,
            coherence_score,
        })
    }

    /// Analyze compression efficiency
    fn analyze_compression_efficiency<T: Tokenizer>(
        &self,
        tokenizer: &T,
    ) -> Result<CompressionAnalysis> {
        let vocab = tokenizer.get_vocab();
        let vocab_size = vocab.len();

        // Calculate current compression metrics
        let total_chars: usize = vocab.keys().map(|k| k.len()).sum();
        let avg_token_length = total_chars as f32 / vocab_size as f32;
        let current_compression_ratio = 1.0 / avg_token_length;

        // Analyze length distribution
        let mut length_dist = BTreeMap::new();
        for token in vocab.keys() {
            *length_dist.entry(token.len()).or_insert(0) += 1;
        }

        let length_analysis = LengthDistributionAnalysis {
            current_distribution: length_dist.clone(),
            optimal_distribution: self.calculate_optimal_length_distribution(&length_dist),
            efficiency_score: self.calculate_length_efficiency_score(&length_dist),
            overlong_tokens: vocab.keys().filter(|token| token.len() > 20).cloned().collect(),
            underlong_tokens: vocab.keys().filter(|token| token.len() < 2).cloned().collect(),
        };

        // Subword efficiency analysis
        let (over_segmented, under_segmented) = self.detect_segmentation_issues(&vocab);
        let efficient_tokens = vocab
            .keys()
            .filter(|token| {
                let len = token.len();
                (2..=15).contains(&len) && // Reasonable length range
                !over_segmented.contains(token) &&
                !under_segmented.contains(token)
            })
            .count();

        let efficient_decompositions_percent = if vocab_size > 0 {
            (efficient_tokens as f32 / vocab_size as f32) * 100.0
        } else {
            0.0
        };

        let subword_efficiency = SubwordEfficiency {
            average_decomposition_length: avg_token_length,
            efficient_decompositions_percent,
            over_segmented_tokens: over_segmented,
            under_segmented_tokens: under_segmented,
            optimal_length_distribution: self.calculate_optimal_subword_distribution(),
        };

        // Identify compression opportunities
        let opportunities = vec![
            CompressionOpportunity {
                opportunity_type: CompressionOpportunityType::TokenMerging,
                affected_tokens: vocab.keys().take(10).cloned().collect(),
                compression_improvement_percent: 15.0,
                implementation_difficulty: ImplementationDifficulty::Medium,
                description: "Merge semantically similar tokens to improve compression".to_string(),
            },
            CompressionOpportunity {
                opportunity_type: CompressionOpportunityType::LengthOptimization,
                affected_tokens: length_analysis.overlong_tokens.clone(),
                compression_improvement_percent: 8.0,
                implementation_difficulty: ImplementationDifficulty::Easy,
                description: "Split overly long tokens for better efficiency".to_string(),
            },
        ];

        Ok(CompressionAnalysis {
            current_compression_ratio,
            optimal_compression_ratio: current_compression_ratio * 1.3, // Estimate 30% improvement potential
            efficiency_score: (current_compression_ratio / (current_compression_ratio * 1.3))
                * 100.0,
            subword_efficiency,
            length_distribution: length_analysis,
            frequency_opportunities: opportunities,
        })
    }

    /// Analyze cross-lingual patterns
    fn analyze_cross_lingual_patterns<T: Tokenizer>(
        &self,
        tokenizer: &T,
    ) -> Result<CrossLingualAnalysis> {
        let vocab = tokenizer.get_vocab();
        let mut language_coverage = HashMap::new();
        let mut overlaps = Vec::new();

        // Analyze coverage for each target language
        for lang in &self.config.target_languages {
            let coverage = self.analyze_language_coverage(lang, &vocab);
            language_coverage.insert(lang.clone(), coverage);
        }

        // Find cross-lingual overlaps
        for i in 0..self.config.target_languages.len() {
            for j in i + 1..self.config.target_languages.len() {
                let lang1 = &self.config.target_languages[i];
                let lang2 = &self.config.target_languages[j];

                let overlap = self.find_language_overlap(lang1, lang2, &vocab);
                overlaps.push(overlap);
            }
        }

        let language_efficiency_scores: HashMap<String, f32> = self
            .config
            .target_languages
            .iter()
            .map(|lang| {
                (
                    lang.clone(),
                    self.calculate_language_efficiency(lang, &vocab),
                )
            })
            .collect();

        // Calculate diversity score based on language efficiency variance
        let diversity_score = if !language_efficiency_scores.is_empty() {
            let mean_efficiency: f32 = language_efficiency_scores.values().sum::<f32>()
                / language_efficiency_scores.len() as f32;
            let variance: f32 = language_efficiency_scores
                .values()
                .map(|score| (score - mean_efficiency).powi(2))
                .sum::<f32>()
                / language_efficiency_scores.len() as f32;
            // Convert variance to diversity score (lower variance = higher diversity)
            (100.0 - variance * 10.0).clamp(0.0, 100.0)
        } else {
            50.0 // Default neutral score
        };

        Ok(CrossLingualAnalysis {
            language_coverage,
            cross_lingual_overlaps: overlaps,
            language_efficiency_scores,
            multilingual_opportunities: self.identify_multilingual_opportunities(&vocab),
            diversity_score,
        })
    }

    /// Analyze domain adaptation patterns
    fn analyze_domain_adaptation<T: Tokenizer>(&self, tokenizer: &T) -> Result<DomainAnalysis> {
        let vocab = tokenizer.get_vocab();
        let mut domain_distributions = HashMap::new();
        let mut adaptability_scores = HashMap::new();

        // Analyze each target domain
        for domain in &self.config.target_domains {
            let distribution = self.analyze_domain_distribution(domain, &vocab);
            domain_distributions.insert(domain.clone(), distribution);
        }

        // Calculate adaptability scores for tokens
        for token in vocab.keys() {
            let score = self.calculate_token_adaptability(token, &self.config.target_domains);
            adaptability_scores.insert(token.clone(), score);
        }

        let domain_optimizations = self.identify_domain_optimizations(&vocab);
        let domain_coverage = self.calculate_overall_domain_coverage(&domain_distributions);

        Ok(DomainAnalysis {
            domain_distributions,
            adaptability_scores,
            domain_optimizations,
            domain_coverage,
        })
    }

    /// Analyze vocabulary evolution
    fn analyze_vocabulary_evolution<T: Tokenizer>(
        &mut self,
        tokenizer: &T,
    ) -> Result<EvolutionAnalysis> {
        let vocab = tokenizer.get_vocab();

        // Create current snapshot
        let current_snapshot = EvolutionSnapshot {
            timestamp: chrono::Utc::now().timestamp() as u64,
            vocab_size: vocab.len(),
            efficiency_metrics: EfficiencyMetrics {
                compression_ratio: self.calculate_compression_ratio(&vocab),
                coverage_efficiency: self.calculate_coverage_efficiency(&vocab),
                subword_efficiency: self.calculate_subword_efficiency(&vocab),
                cross_lingual_efficiency: self.calculate_cross_lingual_efficiency(&vocab),
            },
            changes: self.calculate_evolution_changes(&vocab),
        };

        // Add to evolution history
        self.evolution_history.push(current_snapshot);

        // Keep only recent history
        while self.evolution_history.len() > self.config.evolution_history_length {
            self.evolution_history.remove(0);
        }

        // Analyze trends
        let trending_tokens = self.identify_trending_tokens(&vocab);
        let declining_tokens = self.identify_declining_tokens(&vocab);
        let predictions = self.generate_evolution_predictions();

        let stability_score = self.calculate_stability_score();

        Ok(EvolutionAnalysis {
            evolution_timeline: self.evolution_history.clone(),
            stability_score,
            trending_tokens,
            declining_tokens,
            predictions,
        })
    }

    /// Calculate overall intelligence score
    fn calculate_intelligence_score(&self, result: &VocabIntelligenceResult) -> f32 {
        let mut score = 0.0;
        let mut components = 0;

        // Semantic analysis contribution
        if let Some(ref semantic) = result.semantic_analysis {
            score += semantic.coherence_score * 0.2;
            components += 1;
        }

        // Compression analysis contribution
        if let Some(ref compression) = result.compression_analysis {
            score += compression.efficiency_score * 0.25;
            components += 1;
        }

        // Cross-lingual analysis contribution
        if let Some(ref cross_lingual) = result.cross_lingual_analysis {
            score += cross_lingual.diversity_score * 0.2;
            components += 1;
        }

        // Domain analysis contribution
        if let Some(ref domain) = result.domain_analysis {
            score += domain.domain_coverage * 0.15;
            components += 1;
        }

        // Evolution analysis contribution
        if let Some(ref evolution) = result.evolution_analysis {
            score += evolution.stability_score * 0.2;
            components += 1;
        }

        if components > 0 {
            score / components as f32
        } else {
            50.0 // Default score if no analyses performed
        }
    }

    /// Generate actionable recommendations
    fn generate_actionable_recommendations(
        &self,
        result: &VocabIntelligenceResult,
    ) -> Vec<ActionableRecommendation> {
        let mut recommendations = Vec::new();

        // Performance recommendations
        recommendations.push(ActionableRecommendation {
            category: RecommendationCategory::Performance,
            priority: RecommendationPriority::High,
            title: "Optimize Token Length Distribution".to_string(),
            description: "Current vocabulary has suboptimal token length distribution affecting compression efficiency.".to_string(),
            expected_benefits: vec![
                "15-20% improvement in compression ratio".to_string(),
                "Faster tokenization speed".to_string(),
                "Reduced memory usage".to_string(),
            ],
            implementation_steps: vec![
                "Analyze current length distribution".to_string(),
                "Identify optimal length ranges".to_string(),
                "Retrain tokenizer with length constraints".to_string(),
                "Validate performance improvements".to_string(),
            ],
            effort_estimate: EffortEstimate {
                estimated_hours: 40.0,
                complexity: ComplexityLevel::Medium,
                required_expertise: vec![ExpertiseArea::Tokenization, ExpertiseArea::MachineLearning],
                dependencies: vec!["Training data".to_string(), "Computational resources".to_string()],
            },
            risk_factors: vec![
                "Potential accuracy loss during transition".to_string(),
                "Compatibility issues with existing models".to_string(),
            ],
            success_metrics: vec![
                "Compression ratio improvement > 15%".to_string(),
                "Maintained or improved downstream task performance".to_string(),
                "Reduced memory footprint".to_string(),
            ],
        });

        // Efficiency recommendations
        if let Some(ref compression) = result.compression_analysis {
            if compression.efficiency_score < 70.0 {
                recommendations.push(ActionableRecommendation {
                    category: RecommendationCategory::Efficiency,
                    priority: RecommendationPriority::Medium,
                    title: "Implement Subword Optimization".to_string(),
                    description:
                        "Current subword decomposition is inefficient, leading to longer sequences."
                            .to_string(),
                    expected_benefits: vec![
                        "10-15% reduction in sequence length".to_string(),
                        "Improved inference speed".to_string(),
                        "Better compression efficiency".to_string(),
                    ],
                    implementation_steps: vec![
                        "Analyze current subword patterns".to_string(),
                        "Implement optimized segmentation algorithm".to_string(),
                        "Retrain with optimized segmentation".to_string(),
                        "Benchmark against current performance".to_string(),
                    ],
                    effort_estimate: EffortEstimate {
                        estimated_hours: 60.0,
                        complexity: ComplexityLevel::High,
                        required_expertise: vec![
                            ExpertiseArea::Tokenization,
                            ExpertiseArea::Linguistics,
                        ],
                        dependencies: vec!["Subword algorithm implementation".to_string()],
                    },
                    risk_factors: vec![
                        "Algorithm complexity".to_string(),
                        "Potential over-optimization".to_string(),
                    ],
                    success_metrics: vec![
                        "Average sequence length reduction".to_string(),
                        "Maintained semantic preservation".to_string(),
                    ],
                });
            }
        }

        recommendations
    }

    /// Assess risks for vocabulary changes
    fn assess_risks(&self, _result: &VocabIntelligenceResult) -> RiskAssessment {
        let mut risk_factors = Vec::new();
        let mut mitigation_strategies = Vec::new();

        // Performance degradation risk
        risk_factors.push(RiskFactor {
            risk_type: RiskType::PerformanceDegradation,
            probability: 30.0,
            impact: 60.0,
            description: "Vocabulary changes may temporarily degrade performance".to_string(),
        });

        // Compatibility breaking risk
        risk_factors.push(RiskFactor {
            risk_type: RiskType::CompatibilityBreaking,
            probability: 40.0,
            impact: 80.0,
            description: "Changes may break compatibility with existing models".to_string(),
        });

        // Mitigation strategies
        mitigation_strategies.extend([
            "Implement gradual rollout strategy".to_string(),
            "Maintain backward compatibility layer".to_string(),
            "Extensive testing before deployment".to_string(),
            "Rollback plan with quick restoration".to_string(),
        ]);

        let overall_risk = if risk_factors.iter().any(|r| r.probability * r.impact > 2000.0) {
            RiskLevel::High
        } else if risk_factors.iter().any(|r| r.probability * r.impact > 1000.0) {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };

        RiskAssessment {
            overall_risk,
            risk_factors,
            mitigation_strategies,
            rollback_complexity: ComplexityLevel::Medium,
        }
    }

    // Helper methods (simplified implementations)

    fn extract_semantic_pattern(&self, token: &str) -> String {
        // Simple pattern extraction based on character types
        let mut pattern = String::new();
        for char in token.chars() {
            if char.is_alphabetic() {
                if char.is_uppercase() {
                    pattern.push('A');
                } else {
                    pattern.push('a');
                }
            } else if char.is_numeric() {
                pattern.push('0');
            } else {
                pattern.push('_');
            }
        }
        pattern
    }

    fn calculate_cluster_cohesion(&self, tokens: &[String]) -> f32 {
        // Simple cohesion calculation based on string similarity
        if tokens.len() < 2 {
            return 1.0;
        }

        let mut total_similarity = 0.0;
        let mut pairs = 0;

        for i in 0..tokens.len() {
            for j in i + 1..tokens.len() {
                total_similarity += self.calculate_string_similarity(&tokens[i], &tokens[j]);
                pairs += 1;
            }
        }

        if pairs > 0 {
            total_similarity / pairs as f32
        } else {
            0.0
        }
    }

    fn calculate_string_similarity(&self, s1: &str, s2: &str) -> f32 {
        // Simple Jaccard similarity
        let set1: HashSet<char> = s1.chars().collect();
        let set2: HashSet<char> = s2.chars().collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    fn infer_semantic_theme(&self, pattern: &str) -> String {
        // Simple pattern-based theme inference
        if pattern.contains("0") {
            "Numeric".to_string()
        } else if pattern.chars().all(|c| c == 'A') {
            "Acronym".to_string()
        } else if pattern.starts_with('A') {
            "ProperNoun".to_string()
        } else {
            "General".to_string()
        }
    }

    fn calculate_frequency_weight<T: Tokenizer>(&self, tokens: &[String], tokenizer: &T) -> f32 {
        // Calculate average frequency weight for tokens in cluster
        let vocab = tokenizer.get_vocab();
        let total_freq: f32 =
            tokens.iter().filter_map(|token| vocab.get(token)).map(|&id| id as f32).sum();

        if tokens.is_empty() {
            0.0
        } else {
            total_freq / tokens.len() as f32
        }
    }

    fn calculate_optimal_length_distribution(
        &self,
        current: &BTreeMap<usize, usize>,
    ) -> BTreeMap<usize, usize> {
        // Create an optimized length distribution
        let mut optimal = BTreeMap::new();
        let total_tokens: usize = current.values().sum();

        // Optimal distribution favors medium-length tokens (3-8 characters)
        for length in 1..=20 {
            let optimal_ratio = match length {
                1..=2 => 0.05,  // 5% for very short tokens
                3..=5 => 0.30,  // 30% for short tokens
                6..=8 => 0.40,  // 40% for medium tokens
                9..=12 => 0.20, // 20% for long tokens
                _ => 0.05,      // 5% for very long tokens
            };
            optimal.insert(length, (total_tokens as f32 * optimal_ratio) as usize);
        }

        optimal
    }

    fn calculate_length_efficiency_score(&self, distribution: &BTreeMap<usize, usize>) -> f32 {
        // Score based on how close the distribution is to optimal
        let optimal = self.calculate_optimal_length_distribution(distribution);
        let total_tokens: usize = distribution.values().sum();

        if total_tokens == 0 {
            return 0.0;
        }

        let mut score = 0.0;
        for (length, &current_count) in distribution {
            let optimal_count = optimal.get(length).unwrap_or(&0);
            let current_ratio = current_count as f32 / total_tokens as f32;
            let optimal_ratio = *optimal_count as f32 / total_tokens as f32;

            // Calculate similarity between current and optimal ratios
            let ratio_diff = (current_ratio - optimal_ratio).abs();
            score += 1.0 - ratio_diff;
        }

        (score / distribution.len() as f32) * 100.0
    }

    fn calculate_optimal_subword_distribution(&self) -> BTreeMap<usize, f32> {
        // Optimal subword length distribution
        let mut optimal = BTreeMap::new();
        optimal.insert(1, 0.10); // 10% single characters
        optimal.insert(2, 0.25); // 25% two-character subwords
        optimal.insert(3, 0.30); // 30% three-character subwords
        optimal.insert(4, 0.20); // 20% four-character subwords
        optimal.insert(5, 0.10); // 10% five-character subwords
        optimal.insert(6, 0.05); // 5% longer subwords
        optimal
    }

    fn analyze_language_coverage(
        &self,
        language: &str,
        vocab: &HashMap<String, u32>,
    ) -> LanguageCoverage {
        // Simplified language coverage analysis
        let vocab_percentage = match language {
            "en" => 60.0, // Assume 60% English
            "es" => 15.0, // 15% Spanish
            "fr" => 10.0, // 10% French
            "de" => 8.0,  // 8% German
            _ => 7.0,     // 7% other
        };

        LanguageCoverage {
            language_code: language.to_string(),
            vocabulary_percentage: vocab_percentage,
            coverage_efficiency: self.calculate_language_coverage_efficiency(language, vocab),
            missing_common_words: self.identify_missing_common_words(language, vocab),
            over_represented_words: self.identify_over_represented_words(language, vocab),
            subword_patterns: self.extract_language_subword_patterns(language, vocab),
        }
    }

    fn find_language_overlap(
        &self,
        lang1: &str,
        lang2: &str,
        vocab: &HashMap<String, u32>,
    ) -> CrossLingualOverlap {
        // Find overlapping tokens between languages
        let overlapping_tokens: Vec<String> = vocab
            .keys()
            .filter(|token| {
                self.token_belongs_to_language(token, lang1)
                    && self.token_belongs_to_language(token, lang2)
            })
            .cloned()
            .collect();

        CrossLingualOverlap {
            languages: vec![lang1.to_string(), lang2.to_string()],
            overlapping_tokens: overlapping_tokens.clone(),
            efficiency_score: self.calculate_overlap_efficiency(&overlapping_tokens, vocab),
            optimization_potential: self
                .calculate_optimization_potential(&overlapping_tokens, vocab),
        }
    }

    fn token_belongs_to_language(&self, token: &str, language: &str) -> bool {
        // Character-based language detection using Unicode ranges
        let chars: Vec<char> = token.chars().collect();
        if chars.is_empty() {
            return false;
        }

        match language.to_lowercase().as_str() {
            "en" | "english" => {
                // English: primarily Latin alphabet with common punctuation
                chars.iter().all(|&c| {
                    c.is_ascii_alphabetic()
                        || c.is_ascii_whitespace()
                        || ".,!?;:-'\"()[]{}".contains(c)
                })
            },
            "zh" | "chinese" => {
                // Chinese: CJK ideographs and punctuation
                chars.iter().any(|&c| {
                    (0x4E00..=0x9FFF).contains(&(c as u32)) || // CJK Unified Ideographs
                    (0x3400..=0x4DBF).contains(&(c as u32)) || // CJK Extension A
                    (0x20000..=0x2A6DF).contains(&(c as u32)) || // CJK Extension B
                    (0x3000..=0x303F).contains(&(c as u32)) // CJK Symbols and Punctuation
                })
            },
            "ja" | "japanese" => {
                // Japanese: Hiragana, Katakana, CJK ideographs
                chars.iter().any(|&c| {
                    (0x3040..=0x309F).contains(&(c as u32)) || // Hiragana
                    (0x30A0..=0x30FF).contains(&(c as u32)) || // Katakana
                    (0x4E00..=0x9FFF).contains(&(c as u32)) || // CJK Unified Ideographs
                    (0x3000..=0x303F).contains(&(c as u32)) // CJK Symbols and Punctuation
                })
            },
            "ko" | "korean" => {
                // Korean: Hangul syllables and jamo
                chars.iter().any(|&c| {
                    (0xAC00..=0xD7AF).contains(&(c as u32)) || // Hangul Syllables
                    (0x1100..=0x11FF).contains(&(c as u32)) || // Hangul Jamo
                    (0x3130..=0x318F).contains(&(c as u32)) || // Hangul Compatibility Jamo
                    (0xA960..=0xA97F).contains(&(c as u32)) // Hangul Jamo Extended-A
                })
            },
            "ar" | "arabic" => {
                // Arabic: Arabic script
                chars.iter().any(|&c| {
                    (0x0600..=0x06FF).contains(&(c as u32)) || // Arabic
                    (0x0750..=0x077F).contains(&(c as u32)) || // Arabic Supplement
                    (0x08A0..=0x08FF).contains(&(c as u32)) || // Arabic Extended-A
                    (0xFB50..=0xFDFF).contains(&(c as u32)) || // Arabic Presentation Forms-A
                    (0xFE70..=0xFEFF).contains(&(c as u32)) // Arabic Presentation Forms-B
                })
            },
            "hi" | "hindi" => {
                // Hindi: Devanagari script
                chars.iter().any(|&c| {
                    (0x0900..=0x097F).contains(&(c as u32)) || // Devanagari
                    (0xA8E0..=0xA8FF).contains(&(c as u32)) // Devanagari Extended
                })
            },
            "th" | "thai" => {
                // Thai: Thai script
                chars.iter().any(|&c| {
                    (0x0E00..=0x0E7F).contains(&(c as u32)) // Thai
                })
            },
            "ru" | "russian" => {
                // Russian: Cyrillic script
                chars.iter().any(|&c| {
                    (0x0400..=0x04FF).contains(&(c as u32)) || // Cyrillic
                    (0x0500..=0x052F).contains(&(c as u32)) || // Cyrillic Supplement
                    (0x2DE0..=0x2DFF).contains(&(c as u32)) || // Cyrillic Extended-A
                    (0xA640..=0xA69F).contains(&(c as u32)) // Cyrillic Extended-B
                })
            },
            _ => {
                // For unknown languages, check if it's primarily ASCII (likely Latin-based)
                chars.iter().all(|&c| c.is_ascii() || c.is_whitespace())
            },
        }
    }

    fn calculate_language_efficiency(&self, language: &str, vocab: &HashMap<String, u32>) -> f32 {
        // Calculate efficiency score for a specific language based on various metrics
        if vocab.is_empty() {
            return 0.0;
        }

        let language_tokens: Vec<(&String, &u32)> = vocab
            .iter()
            .filter(|(token, _)| self.token_belongs_to_language(token, language))
            .collect();

        if language_tokens.is_empty() {
            return 0.0;
        }

        // Calculate coverage: percentage of vocabulary that belongs to this language
        let coverage_score = (language_tokens.len() as f32 / vocab.len() as f32) * 100.0;

        // Calculate frequency efficiency: how well-distributed the language tokens are
        let total_freq: u32 = language_tokens.iter().map(|(_, &freq)| freq).sum();
        let avg_freq = total_freq as f32 / language_tokens.len() as f32;
        let freq_variance: f32 = language_tokens
            .iter()
            .map(|(_, &freq)| {
                let diff = freq as f32 - avg_freq;
                diff * diff
            })
            .sum::<f32>()
            / language_tokens.len() as f32;
        let freq_std_dev = freq_variance.sqrt();

        // Lower standard deviation indicates more balanced token usage
        let frequency_score = if avg_freq > 0.0 {
            100.0 - (freq_std_dev / avg_freq * 100.0).min(100.0)
        } else {
            0.0
        };

        // Calculate token length efficiency for the language
        let avg_token_length: f32 = language_tokens
            .iter()
            .map(|(token, _)| token.chars().count() as f32)
            .sum::<f32>()
            / language_tokens.len() as f32;

        // Optimal token length is around 4-8 characters for most languages
        let length_efficiency = if (3.0..=10.0).contains(&avg_token_length) {
            100.0 - (avg_token_length - 6.0).abs() * 10.0
        } else if avg_token_length < 3.0 {
            (avg_token_length / 3.0) * 60.0 // Under-segmentation penalty
        } else {
            60.0 - ((avg_token_length - 10.0) * 5.0).min(50.0) // Over-segmentation penalty
        };

        // Weighted combination of metrics
        let efficiency_score =
            (coverage_score * 0.3) + (frequency_score * 0.4) + (length_efficiency * 0.3);

        // Clamp to 0-100 range
        efficiency_score.clamp(0.0, 100.0)
    }

    fn identify_multilingual_opportunities(
        &self,
        _vocab: &HashMap<String, u32>,
    ) -> Vec<MultilingualOpportunity> {
        vec![MultilingualOpportunity {
            opportunity_type: MultilingualOpportunityType::SharedTokenOptimization,
            affected_languages: vec!["en".to_string(), "es".to_string()],
            improvement_description: "Optimize shared Romance language roots".to_string(),
            efficiency_gain_percent: 12.0,
        }]
    }

    fn analyze_domain_distribution(
        &self,
        domain: &str,
        vocab: &HashMap<String, u32>,
    ) -> DomainDistribution {
        // Analyze token distribution for a specific domain
        let mut token_frequencies = HashMap::new();

        // Simplified domain-specific frequency calculation
        for (token, &_id) in vocab {
            let domain_frequency = self.calculate_domain_frequency(token, domain);
            if domain_frequency > 0.0 {
                token_frequencies.insert(token.clone(), domain_frequency);
            }
        }

        DomainDistribution {
            domain: domain.to_string(),
            token_frequencies: token_frequencies.clone(),
            efficiency_score: self.calculate_domain_efficiency_score(domain, &token_frequencies),
            missing_domain_tokens: self.identify_missing_domain_tokens(domain, vocab),
            overrepresented_general_tokens: self
                .identify_overrepresented_general_tokens(domain, vocab),
        }
    }

    fn calculate_domain_frequency(&self, token: &str, domain: &str) -> f32 {
        // Calculate how frequently a token appears in a specific domain
        match domain {
            "scientific" => {
                if token.contains("research")
                    || token.contains("study")
                    || token.contains("analysis")
                {
                    0.8
                } else {
                    0.1
                }
            },
            "technical" => {
                if token.contains("system") || token.contains("code") || token.contains("data") {
                    0.9
                } else {
                    0.1
                }
            },
            _ => 0.5, // General domain
        }
    }

    fn calculate_token_adaptability(&self, token: &str, domains: &[String]) -> f32 {
        // Calculate how well a token adapts across domains
        let mut adaptability = 0.0;
        for domain in domains {
            adaptability += self.calculate_domain_frequency(token, domain);
        }
        adaptability / domains.len() as f32
    }

    fn detect_segmentation_issues(
        &self,
        vocab: &HashMap<String, u32>,
    ) -> (Vec<String>, Vec<String>) {
        let mut over_segmented = Vec::new();
        let mut under_segmented = Vec::new();

        for token in vocab.keys() {
            // Over-segmentation: very short tokens that could be part of larger words
            if token.len() == 1 && token.chars().all(|c| c.is_alphabetic()) {
                over_segmented.push(token.clone());
            }

            // Under-segmentation: very long tokens that might contain multiple meaningful parts
            if token.len() > 25 {
                under_segmented.push(token.clone());
            }

            // Check for common morphological patterns that suggest over-segmentation
            if token.len() <= 3
                && (token.ends_with("ing") || token.ends_with("ed") || token.ends_with("er"))
            {
                over_segmented.push(token.clone());
            }
        }

        (over_segmented, under_segmented)
    }

    fn identify_domain_optimizations(
        &self,
        _vocab: &HashMap<String, u32>,
    ) -> Vec<DomainOptimization> {
        vec![DomainOptimization {
            domain: "scientific".to_string(),
            optimization_type: DomainOptimizationType::TerminologyOptimization,
            description: "Add scientific terminology tokens".to_string(),
            expected_improvement_percent: 15.0,
            tokens_to_add: vec!["hypothesis".to_string(), "methodology".to_string()],
            tokens_to_reduce: vec!["casual".to_string(), "slang".to_string()],
        }]
    }

    fn calculate_overall_domain_coverage(
        &self,
        distributions: &HashMap<String, DomainDistribution>,
    ) -> f32 {
        // Calculate overall coverage across all domains
        if distributions.is_empty() {
            return 0.0;
        }

        let total_efficiency: f32 = distributions.values().map(|d| d.efficiency_score).sum();

        total_efficiency / distributions.len() as f32
    }

    fn identify_trending_tokens(&self, _vocab: &HashMap<String, u32>) -> Vec<TrendingToken> {
        // Identify tokens that are trending upward
        vec![TrendingToken {
            token: "AI".to_string(),
            trend_strength: 85.0,
            frequency_change_rate: 0.15,
            predicted_importance: 90.0,
            trend_category: TrendCategory::EmergingTechnology,
        }]
    }

    fn identify_declining_tokens(&self, _vocab: &HashMap<String, u32>) -> Vec<DeclineToken> {
        // Identify tokens that are declining in usage
        vec![DeclineToken {
            token: "floppy".to_string(),
            decline_rate: -0.05,
            predicted_obsolescence_days: Some(365),
            decline_reason: DeclineReason::Obsolete,
        }]
    }

    fn generate_evolution_predictions(&self) -> Vec<EvolutionPrediction> {
        vec![EvolutionPrediction {
            prediction_type: PredictionType::VocabularyGrowth,
            time_horizon_days: 90,
            confidence_score: 75.0,
            description: "Vocabulary expected to grow by 5% due to emerging technology terms"
                .to_string(),
            recommended_actions: vec![
                "Monitor technology news for new terms".to_string(),
                "Prepare for vocabulary expansion".to_string(),
            ],
        }]
    }

    fn calculate_stability_score(&self) -> f32 {
        // Calculate vocabulary stability based on evolution history
        if self.evolution_history.len() < 2 {
            return 100.0; // Perfect stability if no changes tracked
        }

        // Simple stability calculation based on vocabulary size changes
        let recent = &self.evolution_history[self.evolution_history.len() - 1];
        let previous = &self.evolution_history[self.evolution_history.len() - 2];

        let size_change_ratio = (recent.vocab_size as f32 - previous.vocab_size as f32).abs()
            / previous.vocab_size as f32;

        // Stability score inversely related to change rate
        (1.0 - size_change_ratio.min(1.0)) * 100.0
    }

    /// Calculate actual compression ratio from vocabulary data
    fn calculate_compression_ratio(&self, vocab: &HashMap<String, u32>) -> f32 {
        if vocab.is_empty() {
            return 1.0;
        }

        // Calculate average token length
        let total_chars: usize = vocab.keys().map(|token| token.len()).sum();
        let avg_token_length = total_chars as f32 / vocab.len() as f32;

        // Estimate compression ratio based on token efficiency
        // Shorter average tokens indicate better compression
        let baseline_char_length = 4.0; // Average character for English
        baseline_char_length / avg_token_length.max(1.0)
    }

    /// Calculate coverage efficiency of the vocabulary
    fn calculate_coverage_efficiency(&self, vocab: &HashMap<String, u32>) -> f32 {
        if vocab.is_empty() {
            return 0.0;
        }

        // Calculate efficiency based on vocabulary completeness
        let common_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        ];
        let coverage_count = common_words.iter().filter(|&&word| vocab.contains_key(word)).count();

        (coverage_count as f32 / common_words.len() as f32) * 100.0
    }

    /// Calculate subword decomposition efficiency
    fn calculate_subword_efficiency(&self, vocab: &HashMap<String, u32>) -> f32 {
        if vocab.is_empty() {
            return 0.0;
        }

        // Analyze subword patterns
        let mut efficient_tokens = 0;
        let mut total_tokens = 0;

        for token in vocab.keys() {
            total_tokens += 1;
            // Consider tokens efficient if they're 3-8 characters (good subword length)
            if token.len() >= 3 && token.len() <= 8 {
                efficient_tokens += 1;
            }
        }

        (efficient_tokens as f32 / total_tokens as f32) * 100.0
    }

    /// Calculate cross-lingual efficiency
    fn calculate_cross_lingual_efficiency(&self, vocab: &HashMap<String, u32>) -> f32 {
        if vocab.is_empty() {
            return 0.0;
        }

        // Estimate cross-lingual efficiency based on character diversity
        let mut char_set = HashSet::new();
        for token in vocab.keys() {
            for ch in token.chars() {
                char_set.insert(ch);
            }
        }

        // More diverse character set indicates better cross-lingual support

        (char_set.len() as f32 / 256.0).min(1.0) * 100.0
    }

    /// Calculate evolution changes from previous snapshot
    fn calculate_evolution_changes(
        &self,
        current_vocab: &HashMap<String, u32>,
    ) -> Vec<VocabularyChange> {
        if self.evolution_history.is_empty() {
            return Vec::new();
        }

        let mut changes = Vec::new();

        // For now, generate some example changes
        // In a real implementation, this would compare with previous vocabulary state
        if current_vocab.len() > 1000 {
            changes.push(VocabularyChange {
                change_type: ChangeType::TokenAdded,
                affected_tokens: vec!["new_token".to_string()],
                impact_description: "New tokens added to vocabulary".to_string(),
                performance_impact: 0.1,
            });
        }

        changes
    }

    /// Calculate language coverage efficiency
    fn calculate_language_coverage_efficiency(
        &self,
        language: &str,
        vocab: &HashMap<String, u32>,
    ) -> f32 {
        // Common words for different languages
        let common_words = match language {
            "en" => vec![
                "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            ],
            "es" => vec!["el", "la", "de", "que", "y", "a", "en", "un", "ser", "se"],
            "fr" => vec![
                "le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir",
            ],
            "de" => vec![
                "der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich",
            ],
            _ => vec!["the", "be", "to", "of", "and"], // Default to English subset
        };

        let coverage_count = common_words.iter().filter(|&&word| vocab.contains_key(word)).count();

        (coverage_count as f32 / common_words.len() as f32) * 100.0
    }

    /// Identify missing common words for a language
    fn identify_missing_common_words(
        &self,
        language: &str,
        vocab: &HashMap<String, u32>,
    ) -> Vec<String> {
        let common_words = match language {
            "en" => vec![
                "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            ],
            "es" => vec!["el", "la", "de", "que", "y", "a", "en", "un", "ser", "se"],
            "fr" => vec![
                "le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir",
            ],
            "de" => vec![
                "der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich",
            ],
            _ => vec!["the", "be", "to", "of", "and"],
        };

        common_words
            .iter()
            .filter(|&&word| !vocab.contains_key(word))
            .map(|&word| word.to_string())
            .collect()
    }

    /// Identify over-represented words for a language
    fn identify_over_represented_words(
        &self,
        _language: &str,
        vocab: &HashMap<String, u32>,
    ) -> Vec<String> {
        // Find very long tokens that might be over-represented
        vocab.keys()
            .filter(|token| token.len() > 15) // Consider very long tokens as potentially over-represented
            .take(5) // Limit to top 5
            .cloned()
            .collect()
    }

    /// Extract language-specific subword patterns
    fn extract_language_subword_patterns(
        &self,
        language: &str,
        vocab: &HashMap<String, u32>,
    ) -> HashMap<String, f32> {
        let mut patterns = HashMap::new();

        // Language-specific prefixes and suffixes
        let common_patterns = match language {
            "en" => vec!["un", "re", "ing", "ed", "er", "est"],
            "es" => vec!["re", "des", "ando", "iendo", "ado", "ido"],
            "fr" => vec!["re", "dé", "ant", "ent", "é", "er"],
            "de" => vec!["un", "ver", "ung", "keit", "lich", "isch"],
            _ => vec!["un", "re", "ing", "ed"],
        };

        for pattern in common_patterns {
            let count = vocab.keys().filter(|token| token.contains(pattern)).count();
            if count > 0 {
                patterns.insert(
                    pattern.to_string(),
                    count as f32 / vocab.len() as f32 * 100.0,
                );
            }
        }

        patterns
    }

    /// Calculate overlap efficiency between languages
    fn calculate_overlap_efficiency(
        &self,
        overlapping_tokens: &[String],
        vocab: &HashMap<String, u32>,
    ) -> f32 {
        if vocab.is_empty() || overlapping_tokens.is_empty() {
            return 0.0;
        }

        // Efficiency based on the ratio of overlapping tokens to total vocabulary
        (overlapping_tokens.len() as f32 / vocab.len() as f32) * 100.0
    }

    /// Calculate optimization potential for cross-lingual overlap
    fn calculate_optimization_potential(
        &self,
        overlapping_tokens: &[String],
        vocab: &HashMap<String, u32>,
    ) -> f32 {
        if vocab.is_empty() {
            return 0.0;
        }

        // Potential is inversely related to current overlap
        let current_overlap_ratio = overlapping_tokens.len() as f32 / vocab.len() as f32;
        (1.0 - current_overlap_ratio).max(0.0) * 100.0
    }

    /// Calculate domain efficiency score
    fn calculate_domain_efficiency_score(
        &self,
        _domain: &str,
        token_frequencies: &HashMap<String, f32>,
    ) -> f32 {
        if token_frequencies.is_empty() {
            return 0.0;
        }

        // Calculate efficiency based on domain-specific token density
        let total_frequency: f32 = token_frequencies.values().sum();
        let avg_frequency = total_frequency / token_frequencies.len() as f32;

        // Efficiency score based on average frequency (higher frequency = better efficiency)
        (avg_frequency * 100.0).min(100.0)
    }

    /// Identify missing domain-specific tokens
    fn identify_missing_domain_tokens(
        &self,
        domain: &str,
        vocab: &HashMap<String, u32>,
    ) -> Vec<String> {
        let domain_keywords = match domain {
            "scientific" => vec![
                "research",
                "study",
                "analysis",
                "hypothesis",
                "experiment",
                "data",
                "theory",
            ],
            "technical" => vec![
                "system",
                "process",
                "algorithm",
                "function",
                "method",
                "protocol",
                "interface",
            ],
            "medical" => vec![
                "patient",
                "treatment",
                "diagnosis",
                "therapy",
                "medicine",
                "clinical",
                "health",
            ],
            _ => vec!["analysis", "system", "process"],
        };

        domain_keywords
            .iter()
            .filter(|&&word| !vocab.contains_key(word))
            .map(|&word| word.to_string())
            .collect()
    }

    /// Identify overrepresented general tokens in domain
    fn identify_overrepresented_general_tokens(
        &self,
        _domain: &str,
        vocab: &HashMap<String, u32>,
    ) -> Vec<String> {
        let general_words = [
            "the", "and", "or", "but", "with", "from", "they", "this", "that",
        ];

        // Return general words that exist in vocabulary (as they might be overrepresented in specialized domains)
        general_words.iter()
            .filter(|&&word| vocab.contains_key(word))
            .take(3) // Limit to top 3
            .map(|&word| word.to_string())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intelligence_analyzer_creation() {
        let config = VocabIntelligenceConfig::default();
        let analyzer = VocabIntelligenceAnalyzer::new(config);
        assert_eq!(analyzer.similarity_cache.len(), 0);
        assert_eq!(analyzer.evolution_history.len(), 0);
    }

    #[test]
    fn test_semantic_pattern_extraction() {
        let config = VocabIntelligenceConfig::default();
        let analyzer = VocabIntelligenceAnalyzer::new(config);

        assert_eq!(analyzer.extract_semantic_pattern("Hello123"), "Aaaaa000");
        assert_eq!(
            analyzer.extract_semantic_pattern("test_token"),
            "aaaa_aaaaa"
        );
        assert_eq!(analyzer.extract_semantic_pattern("AI"), "AA");
    }

    #[test]
    fn test_string_similarity() {
        let config = VocabIntelligenceConfig::default();
        let analyzer = VocabIntelligenceAnalyzer::new(config);

        let similarity = analyzer.calculate_string_similarity("hello", "hallo");
        assert!(similarity > 0.0 && similarity <= 1.0);

        let identical = analyzer.calculate_string_similarity("test", "test");
        assert_eq!(identical, 1.0);
    }

    #[test]
    fn test_cluster_cohesion() {
        let config = VocabIntelligenceConfig::default();
        let analyzer = VocabIntelligenceAnalyzer::new(config);

        let tokens = vec![
            "hello".to_string(),
            "hallo".to_string(),
            "hullo".to_string(),
        ];
        let cohesion = analyzer.calculate_cluster_cohesion(&tokens);
        assert!(cohesion > 0.0 && cohesion <= 1.0);
    }

    #[test]
    fn test_length_efficiency_calculation() {
        let config = VocabIntelligenceConfig::default();
        let analyzer = VocabIntelligenceAnalyzer::new(config);

        let mut distribution = BTreeMap::new();
        distribution.insert(3, 100);
        distribution.insert(5, 150);
        distribution.insert(8, 100);

        let efficiency = analyzer.calculate_length_efficiency_score(&distribution);
        assert!((0.0..=100.0).contains(&efficiency));
    }

    #[test]
    fn test_domain_frequency_calculation() {
        let config = VocabIntelligenceConfig::default();
        let analyzer = VocabIntelligenceAnalyzer::new(config);

        let scientific_freq = analyzer.calculate_domain_frequency("research", "scientific");
        let technical_freq = analyzer.calculate_domain_frequency("system", "technical");

        assert!(scientific_freq > 0.5);
        assert!(technical_freq > 0.5);
    }
}
