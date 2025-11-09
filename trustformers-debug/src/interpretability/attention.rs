//! Attention analysis for transformer models
//!
//! This module provides comprehensive attention analysis capabilities including
//! head specialization analysis, attention flow analysis, and pattern detection.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Attention analysis result for transformer models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionAnalysisResult {
    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,
    /// Attention weights by layer and head
    pub attention_weights: HashMap<String, AttentionLayerResult>,
    /// Token-to-token attention patterns
    pub attention_patterns: AttentionPatterns,
    /// Head specialization analysis
    pub head_specialization: HeadSpecializationAnalysis,
    /// Attention flow analysis
    pub attention_flow: AttentionFlowAnalysis,
    /// Key attention statistics
    pub attention_stats: AttentionStatistics,
}

/// Attention analysis for a single layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionLayerResult {
    /// Layer index
    pub layer_index: usize,
    /// Attention heads results
    pub heads: HashMap<usize, AttentionHeadResult>,
    /// Layer-level attention patterns
    pub layer_patterns: LayerAttentionPatterns,
    /// Layer attention statistics
    pub layer_stats: LayerAttentionStats,
}

/// Attention analysis for a single head
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionHeadResult {
    /// Head index
    pub head_index: usize,
    /// Attention matrix (token x token)
    pub attention_matrix: Vec<Vec<f64>>,
    /// Token attention scores
    pub token_scores: Vec<TokenAttentionScore>,
    /// Head specialization type
    pub specialization_type: HeadSpecializationType,
    /// Attention entropy
    pub entropy: f64,
    /// Maximum attention value
    pub max_attention: f64,
    /// Attention sparsity
    pub sparsity: f64,
}

/// Token attention information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenAttentionScore {
    /// Token text
    pub token: String,
    /// Token position
    pub position: usize,
    /// Attention received (sum of incoming attention)
    pub attention_received: f64,
    /// Attention given (sum of outgoing attention)
    pub attention_given: f64,
    /// Self-attention score
    pub self_attention: f64,
    /// Most attended tokens
    pub most_attended: Vec<(String, f64)>,
}

/// Types of attention head specialization
#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum HeadSpecializationType {
    /// Focuses on local dependencies
    Local,
    /// Focuses on long-range dependencies
    Global,
    /// Focuses on syntactic relationships
    Syntactic,
    /// Focuses on semantic relationships
    Semantic,
    /// Focuses on positional patterns
    Positional,
    /// Mixed or unclear specialization
    Mixed,
    /// Attention on special tokens
    SpecialToken,
}

/// Overall attention patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionPatterns {
    /// Diagonal attention patterns (local dependencies)
    pub diagonal_patterns: Vec<DiagonalPattern>,
    /// Vertical attention patterns (token dependencies)
    pub vertical_patterns: Vec<VerticalPattern>,
    /// Block attention patterns
    pub block_patterns: Vec<BlockPattern>,
    /// Repetitive attention patterns
    pub repetitive_patterns: Vec<RepetitivePattern>,
}

/// Diagonal attention pattern (local dependencies)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagonalPattern {
    /// Layer and head identifier
    pub layer_head: (usize, usize),
    /// Diagonal offset
    pub offset: i32,
    /// Pattern strength
    pub strength: f64,
    /// Pattern coverage (how much of the diagonal)
    pub coverage: f64,
}

/// Vertical attention pattern (specific token focus)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerticalPattern {
    /// Layer and head identifier
    pub layer_head: (usize, usize),
    /// Target token position
    pub target_position: usize,
    /// Pattern strength
    pub strength: f64,
    /// Number of tokens attending to target
    pub attending_tokens: usize,
}

/// Block attention pattern (sequence segments)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockPattern {
    /// Layer and head identifier
    pub layer_head: (usize, usize),
    /// Block start position
    pub start_position: usize,
    /// Block end position
    pub end_position: usize,
    /// Internal attention strength
    pub internal_strength: f64,
    /// External attention (leakage)
    pub external_attention: f64,
}

/// Repetitive attention pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepetitivePattern {
    /// Layer and head identifier
    pub layer_head: (usize, usize),
    /// Pattern period
    pub period: usize,
    /// Pattern strength
    pub strength: f64,
    /// Number of repetitions
    pub repetitions: usize,
}

/// Layer-level attention patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAttentionPatterns {
    /// Average attention distance
    pub avg_attention_distance: f64,
    /// Attention concentration (how focused)
    pub concentration: f64,
    /// Inter-head similarity
    pub inter_head_similarity: f64,
    /// Layer attention diversity
    pub diversity: f64,
}

/// Layer attention statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAttentionStats {
    /// Mean attention value
    pub mean_attention: f64,
    /// Attention variance
    pub attention_variance: f64,
    /// Attention entropy
    pub entropy: f64,
    /// Number of significant attention connections
    pub significant_connections: usize,
    /// Attention sparsity ratio
    pub sparsity_ratio: f64,
}

/// Head specialization analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadSpecializationAnalysis {
    /// Specialization by layer
    pub layer_specialization: HashMap<usize, Vec<HeadSpecializationType>>,
    /// Head clustering results
    pub head_clusters: Vec<HeadCluster>,
    /// Specialization evolution across layers
    pub specialization_evolution: SpecializationEvolution,
    /// Head redundancy analysis
    pub redundancy_analysis: HeadRedundancyAnalysis,
}

/// Cluster of similar attention heads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadCluster {
    /// Cluster ID
    pub cluster_id: usize,
    /// Heads in cluster (layer, head)
    pub heads: Vec<(usize, usize)>,
    /// Cluster centroid pattern
    pub centroid_pattern: Vec<f64>,
    /// Cluster cohesion
    pub cohesion: f64,
    /// Cluster specialization type
    pub specialization: HeadSpecializationType,
}

/// Evolution of specialization across layers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecializationEvolution {
    /// Specialization distribution by layer
    pub layer_distribution: HashMap<usize, HashMap<HeadSpecializationType, usize>>,
    /// Specialization transitions
    pub transitions: Vec<SpecializationTransition>,
    /// Overall specialization trend
    pub trend: SpecializationTrend,
}

/// Transition between specialization types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecializationTransition {
    /// From layer
    pub from_layer: usize,
    /// To layer
    pub to_layer: usize,
    /// From specialization
    pub from_specialization: HeadSpecializationType,
    /// To specialization
    pub to_specialization: HeadSpecializationType,
    /// Transition strength
    pub strength: f64,
}

/// Overall specialization trend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpecializationTrend {
    /// Increasing specialization with depth
    Increasing,
    /// Decreasing specialization with depth
    Decreasing,
    /// Stable specialization
    Stable,
    /// Mixed pattern
    Mixed,
}

/// Head redundancy analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadRedundancyAnalysis {
    /// Redundant head pairs
    pub redundant_pairs: Vec<RedundantHeadPair>,
    /// Overall redundancy score
    pub redundancy_score: f64,
    /// Pruning recommendations
    pub pruning_recommendations: Vec<PruningRecommendation>,
    /// Essential heads (cannot be pruned)
    pub essential_heads: Vec<(usize, usize)>,
}

/// Pair of redundant attention heads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedundantHeadPair {
    /// First head (layer, head)
    pub head1: (usize, usize),
    /// Second head (layer, head)
    pub head2: (usize, usize),
    /// Similarity score
    pub similarity: f64,
    /// Redundancy type
    pub redundancy_type: RedundancyType,
}

/// Type of redundancy between heads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RedundancyType {
    /// Identical attention patterns
    Identical,
    /// Highly correlated patterns
    Correlated,
    /// Functionally equivalent
    Functional,
    /// Partially overlapping
    Partial,
}

/// Recommendation for head pruning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningRecommendation {
    /// Head to prune (layer, head)
    pub head_to_prune: (usize, usize),
    /// Confidence in pruning recommendation
    pub confidence: f64,
    /// Expected impact of pruning
    pub expected_impact: PruningImpact,
    /// Reason for recommendation
    pub reason: String,
}

/// Expected impact of pruning a head
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningImpact {
    /// Expected performance drop
    pub performance_drop: f64,
    /// Memory savings
    pub memory_savings: f64,
    /// Computational savings
    pub computational_savings: f64,
    /// Risk level
    pub risk_level: RiskLevel,
}

/// Risk level for pruning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Low risk
    Low,
    /// Medium risk
    Medium,
    /// High risk
    High,
    /// Critical risk
    Critical,
}

/// Attention flow analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionFlowAnalysis {
    /// Flow paths through layers
    pub flow_paths: Vec<AttentionFlowPath>,
    /// Flow bottlenecks
    pub bottlenecks: Vec<AttentionBottleneck>,
    /// Flow efficiency metrics
    pub efficiency_metrics: FlowEfficiencyMetrics,
    /// Layer flow statistics
    pub layer_flow_stats: HashMap<usize, LayerFlowStats>,
}

/// Attention flow path
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionFlowPath {
    /// Path ID
    pub path_id: String,
    /// Starting token position
    pub start_position: usize,
    /// Ending token position
    pub end_position: usize,
    /// Flow steps through layers
    pub flow_steps: Vec<LayerFlowStep>,
    /// Total flow strength
    pub total_strength: f64,
    /// Path length
    pub path_length: usize,
}

/// Flow step through a layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerFlowStep {
    /// Layer index
    pub layer_index: usize,
    /// Head index
    pub head_index: usize,
    /// Flow strength at this step
    pub strength: f64,
    /// Flow transformation type
    pub transformation: FlowTransformation,
    /// Tokens involved in this step
    pub involved_tokens: Vec<usize>,
}

/// Type of flow transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FlowTransformation {
    /// Direct attention flow
    Direct,
    /// Attention diffusion
    Diffusion,
    /// Attention concentration
    Concentration,
    /// Attention routing
    Routing,
}

/// Attention bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionBottleneck {
    /// Bottleneck location (layer, head)
    pub location: (usize, usize),
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,
    /// Severity of bottleneck
    pub severity: f64,
    /// Affected flow paths
    pub affected_paths: Vec<String>,
}

/// Type of attention bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    /// Information bottleneck
    Information,
    /// Attention bottleneck
    Attention,
    /// Capacity bottleneck
    Capacity,
    /// Flow bottleneck
    Flow,
}

/// Flow efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowEfficiencyMetrics {
    /// Overall flow efficiency
    pub overall_efficiency: f64,
    /// Information preservation ratio
    pub information_preservation: f64,
    /// Flow redundancy
    pub flow_redundancy: f64,
    /// Bottleneck impact
    pub bottleneck_impact: f64,
}

/// Layer flow statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerFlowStats {
    /// Incoming flow strength
    pub incoming_flow: f64,
    /// Outgoing flow strength
    pub outgoing_flow: f64,
    /// Flow retention ratio
    pub retention_ratio: f64,
    /// Flow transformation ratio
    pub transformation_ratio: f64,
}

/// Attention statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionStatistics {
    /// Average attention entropy across all heads
    pub avg_entropy: f64,
    /// Attention concentration distribution
    pub concentration_distribution: HashMap<String, f64>,
    /// Sparsity distribution
    pub sparsity_distribution: SparsityDistribution,
    /// Key insights from attention analysis
    pub insights: Vec<AttentionInsight>,
}

/// Sparsity distribution across layers and heads
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparsityDistribution {
    /// Sparsity by layer
    pub by_layer: HashMap<usize, f64>,
    /// Sparsity by head
    pub by_head: HashMap<(usize, usize), f64>,
    /// Overall sparsity
    pub overall_sparsity: f64,
    /// Sparsity variance
    pub sparsity_variance: f64,
}

/// Attention insight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionInsight {
    /// Insight type
    pub insight_type: InsightType,
    /// Insight description
    pub description: String,
    /// Confidence level
    pub confidence: f64,
    /// Supporting evidence
    pub evidence: Vec<String>,
}

/// Type of attention insight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InsightType {
    /// Head specialization pattern
    HeadSpecialization,
    /// Attention pattern discovery
    PatternDiscovery,
    /// Flow analysis finding
    FlowAnalysis,
    /// Redundancy detection
    RedundancyDetection,
    /// Performance optimization opportunity
    OptimizationOpportunity,
    /// Model behavior explanation
    BehaviorExplanation,
}