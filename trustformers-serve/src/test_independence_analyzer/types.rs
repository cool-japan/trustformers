//! Core Types and Definitions for Test Independence Analysis
//!
//! This module contains all the fundamental types, enums, and configuration structures
//! used throughout the test independence analysis system.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

use crate::test_parallelization::TestDependency;

// ================================================================================================
// Core Analysis Types
// ================================================================================================

/// Analysis result type alias
pub type AnalysisResult<T> = Result<T, AnalysisError>;

/// Comprehensive error types for test independence analysis
#[derive(Debug, thiserror::Error)]
pub enum AnalysisError {
    /// Cache operation failed
    #[error("Cache operation failed: {message}")]
    CacheError { message: String },

    /// Dependency analysis failed
    #[error("Dependency analysis failed for test '{test_id}': {message}")]
    DependencyAnalysisError { test_id: String, message: String },

    /// Conflict detection failed
    #[error("Conflict detection failed: {message}")]
    ConflictDetectionError { message: String },

    /// Resource database error
    #[error("Resource database error: {message}")]
    ResourceDatabaseError { message: String },

    /// Graph operation failed
    #[error("Graph operation failed: {message}")]
    GraphError { message: String },

    /// Test grouping failed
    #[error("Test grouping failed: {message}")]
    GroupingError { message: String },

    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },

    /// Validation error
    #[error("Validation error: {message}")]
    ValidationError { message: String },

    /// Internal system error
    #[error("Internal system error: {message}")]
    InternalError { message: String },

    /// Invalid usage record
    #[error("Invalid usage record: {reason}")]
    InvalidUsageRecord { reason: String },

    /// Invalid allocation event
    #[error("Invalid allocation event: {reason}")]
    InvalidAllocationEvent { reason: String },

    /// Analysis timeout
    #[error("Analysis timeout during {operation} (timeout: {timeout:?})")]
    AnalysisTimeout {
        operation: String,
        timeout: Duration,
    },

    /// Time conversion error
    #[error("Time conversion error: {source}")]
    TimeConversionError {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Strategy not found
    #[error("Strategy not found: {message}")]
    StrategyNotFound { message: String },

    /// Resource type already exists
    #[error("Resource type already exists: {type_id}")]
    ResourceTypeAlreadyExists { type_id: String },

    /// Invalid grouping
    #[error("Invalid grouping: {message}")]
    InvalidGrouping { message: String },
}

// ================================================================================================
// Cache Types
// ================================================================================================

/// Cached dependency analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedDependencyAnalysis {
    /// Test identifier
    pub test_id: String,

    /// Dependencies found
    pub dependencies: Vec<TestDependency>,

    /// Analysis timestamp
    #[serde(skip)]

    /// Analysis version
    pub version: u64,

    /// Analysis confidence score (0.0 to 1.0)
    pub confidence: f32,

    /// Cache metadata
    pub metadata: CacheMetadata,
}

/// Cached conflict analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedConflictAnalysis {
    /// Test pairs analyzed
    pub test_pairs: Vec<(String, String)>,

    /// Conflicts found
    pub conflicts: Vec<ResourceConflict>,

    /// Analysis timestamp
    #[serde(skip)]

    /// Analysis confidence score (0.0 to 1.0)
    pub confidence: f32,

    /// Cache metadata
    pub metadata: CacheMetadata,
}

/// Cached grouping analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedGroupingAnalysis {
    /// Test groups
    pub groups: Vec<TestGroup>,

    /// Grouping strategy used
    pub strategy: GroupingStrategy,

    /// Analysis timestamp
    #[serde(skip)]

    /// Group formation quality score (0.0 to 1.0)
    pub quality_score: f32,

    /// Cache metadata
    pub metadata: CacheMetadata,
}

/// Cache metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetadata {
    /// Entry creation time
    pub created_at: DateTime<Utc>,

    /// Last access time
    pub last_accessed: DateTime<Utc>,

    /// Access count
    pub access_count: u64,

    /// Entry size in bytes
    pub size_bytes: u64,

    /// Cache key
    pub cache_key: String,

    /// Cache tags for categorization
    pub tags: Vec<String>,
}

/// Cache statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct CacheStatistics {
    /// Cache hits
    pub hits: u64,

    /// Cache misses
    pub misses: u64,

    /// Cache evictions
    pub evictions: u64,

    /// Cache memory usage (bytes)
    pub memory_usage: u64,

    /// Hit ratio (0.0 to 1.0)
    pub hit_ratio: f32,

    /// Average access time
    pub average_access_time: Duration,

    /// Cache entries by type
    pub entries_by_type: HashMap<String, u64>,
}

impl CacheStatistics {
    /// Update statistics after a cache operation
    pub fn update_after_access(&mut self, hit: bool, access_time: Duration) {
        if hit {
            self.hits += 1;
        } else {
            self.misses += 1;
        }

        // Update hit ratio
        let total_accesses = self.hits + self.misses;
        if total_accesses > 0 {
            self.hit_ratio = self.hits as f32 / total_accesses as f32;
        }

        // Update average access time
        let total_time = self.average_access_time.as_nanos() as f64 * (total_accesses - 1) as f64;
        let new_average = (total_time + access_time.as_nanos() as f64) / total_accesses as f64;
        self.average_access_time = Duration::from_nanos(new_average as u64);
    }
}

// ================================================================================================
// Dependency Graph Types
// ================================================================================================

/// Dependency edge in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyEdge {
    /// Target test
    pub target: String,

    /// Dependency information
    pub dependency: TestDependency,

    /// Edge weight for algorithms (0.0 to 1.0)
    pub weight: f32,

    /// Edge metadata
    pub metadata: EdgeMetadata,
}

/// Edge metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeMetadata {
    /// Edge creation time
    pub created_at: DateTime<Utc>,

    /// Last validation time
    pub last_validated: DateTime<Utc>,

    /// Validation confidence (0.0 to 1.0)
    pub confidence: f32,

    /// Edge tags
    pub tags: Vec<String>,

    /// Additional properties
    pub properties: HashMap<String, String>,
}

/// Graph metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadata {
    /// Number of nodes (tests)
    pub node_count: usize,

    /// Number of edges (dependencies)
    pub edge_count: usize,

    /// Graph density (0.0 to 1.0)
    pub density: f32,

    /// Strongly connected components
    pub strongly_connected_components: Vec<Vec<String>>,

    /// Topological ordering (if DAG)
    pub topological_order: Option<Vec<String>>,

    /// Graph analysis timestamp
    pub last_analysis: DateTime<Utc>,

    /// Graph properties
    pub properties: GraphProperties,
}

/// Graph properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphProperties {
    /// Is the graph a DAG (Directed Acyclic Graph)
    pub is_dag: bool,

    /// Has cycles
    pub has_cycles: bool,

    /// Maximum path length
    pub max_path_length: usize,

    /// Average degree
    pub average_degree: f32,

    /// Clustering coefficient
    pub clustering_coefficient: f32,
}

impl Default for GraphMetadata {
    fn default() -> Self {
        Self {
            node_count: 0,
            edge_count: 0,
            density: 0.0,
            strongly_connected_components: Vec::new(),
            topological_order: None,
            last_analysis: Utc::now(),
            properties: GraphProperties {
                is_dag: true,
                has_cycles: false,
                max_path_length: 0,
                average_degree: 0.0,
                clustering_coefficient: 0.0,
            },
        }
    }
}

// ================================================================================================
// Resource Usage Types
// ================================================================================================

/// Resource usage record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageRecord {
    /// Test identifier
    pub test_id: String,

    /// Resource type
    pub resource_type: String,

    /// Resource identifier
    pub resource_id: String,

    /// Usage start time
    #[serde(skip)]

    /// Usage duration
    pub duration: Duration,

    /// Usage amount/intensity
    pub usage_amount: f64,

    /// Usage efficiency (0.0 to 1.0)
    pub efficiency: f32,

    /// Concurrent users
    pub concurrent_users: usize,

    /// Usage metadata
    pub metadata: UsageMetadata,
}

/// Usage metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageMetadata {
    /// Usage category
    pub category: UsageCategory,

    /// Usage priority
    pub priority: UsagePriority,

    /// Usage tags
    pub tags: Vec<String>,

    /// Additional properties
    pub properties: HashMap<String, String>,
}

/// Usage categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UsageCategory {
    /// CPU usage
    Cpu,
    /// Memory usage
    Memory,
    /// Disk I/O usage
    DiskIo,
    /// Network I/O usage
    NetworkIo,
    /// Database usage
    Database,
    /// File system usage
    FileSystem,
    /// Custom resource usage
    Custom(String),
}

/// Usage priority levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum UsagePriority {
    /// Low priority
    Low = 0,
    /// Normal priority
    Normal = 1,
    /// High priority
    High = 2,
    /// Critical priority
    Critical = 3,
}

impl Default for UsagePriority {
    fn default() -> Self {
        UsagePriority::Normal
    }
}

/// Resource type definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceTypeDefinition {
    /// Resource type name
    pub name: String,

    /// Resource description
    pub description: String,

    /// Maximum concurrent users
    pub max_concurrent_users: Option<usize>,

    /// Sharing capabilities
    pub sharing_capabilities: ResourceSharingSpec,

    /// Conflict detection rules
    pub conflict_rules: Vec<ConflictRule>,

    /// Performance characteristics
    pub performance_characteristics: PerformanceCharacteristics,

    /// Resource configuration
    pub config: ResourceConfig,
}

/// Resource sharing specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSharingSpec {
    /// Can be shared among tests
    pub shareable: bool,

    /// Maximum sharing level
    pub max_sharing_level: Option<usize>,

    /// Sharing overhead (0.0 to 1.0)
    pub sharing_overhead: f32,

    /// Sharing constraints
    pub constraints: Vec<SharingConstraint>,
}

/// Sharing constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharingConstraint {
    /// Constraint type
    pub constraint_type: SharingConstraintType,

    /// Constraint value
    pub value: String,

    /// Constraint severity
    pub severity: ConstraintSeverity,

    /// Constraint description
    pub description: String,
}

/// Sharing constraint types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SharingConstraintType {
    /// Memory limit constraint
    MemoryLimit,
    /// CPU usage constraint
    CpuUsage,
    /// Network bandwidth constraint
    NetworkBandwidth,
    /// Filesystem access constraint
    FilesystemAccess,
    /// Database connection constraint
    DatabaseConnection,
    /// Port range constraint
    PortRange,
    /// Custom constraint
    Custom(String),
}

/// Constraint severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ConstraintSeverity {
    /// Soft constraint (preference)
    Soft = 0,
    /// Hard constraint (requirement)
    Hard = 1,
    /// Critical constraint (violation causes failure)
    Critical = 2,
}

/// Performance characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceCharacteristics {
    /// Throughput characteristics
    pub throughput: ThroughputCharacteristics,

    /// Latency characteristics
    pub latency: LatencyCharacteristics,

    /// Scalability characteristics
    pub scalability: ScalabilityCharacteristics,
}

/// Throughput characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputCharacteristics {
    /// Base throughput
    pub base_throughput: f64,

    /// Peak throughput
    pub peak_throughput: f64,

    /// Throughput unit (e.g., "requests/second", "MB/s")
    pub unit: String,
}

/// Latency characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyCharacteristics {
    /// Base latency
    pub base_latency: Duration,

    /// Latency under load
    pub load_latency: Duration,

    /// Latency variance
    pub variance: Duration,
}

/// Scalability characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityCharacteristics {
    /// Maximum scale factor
    pub max_scale_factor: f32,

    /// Scale efficiency (0.0 to 1.0)
    pub scale_efficiency: f32,

    /// Bottleneck points
    pub bottlenecks: Vec<String>,
}

/// Resource configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    /// Configuration parameters
    pub parameters: HashMap<String, String>,

    /// Configuration version
    pub version: String,

    /// Configuration validation rules
    pub validation_rules: Vec<ValidationRule>,
}

/// Validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,

    /// Rule description
    pub description: String,

    /// Rule expression
    pub expression: String,

    /// Rule severity
    pub severity: ValidationSeverity,
}

/// Validation severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidationSeverity {
    /// Warning level
    Warning = 0,
    /// Error level
    Error = 1,
    /// Critical level
    Critical = 2,
}

// ================================================================================================
// Conflict Types
// ================================================================================================

/// Resource conflict information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConflict {
    /// Conflict identifier
    pub id: String,

    /// First test in conflict
    pub test1: String,

    /// Second test in conflict
    pub test2: String,

    /// Resource type involved
    pub resource_type: String,

    /// Resource identifier
    pub resource_id: String,

    /// Conflict type
    pub conflict_type: ConflictType,

    /// Conflict severity
    pub severity: ConflictSeverity,

    /// Conflict description
    pub description: String,

    /// Resolution strategies
    pub resolution_strategies: Vec<ResolutionStrategy>,

    /// Conflict metadata
    pub metadata: ConflictMetadata,
}

/// Types of resource conflicts
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ConflictType {
    /// Exclusive access conflict
    ExclusiveAccess,
    /// Capacity limit conflict
    CapacityLimit,
    /// Performance degradation conflict
    PerformanceDegradation,
    /// Data corruption risk
    DataCorruption,
    /// Race condition risk
    RaceCondition,
    /// Deadlock risk
    DeadlockRisk,
    /// Custom conflict type
    Custom(String),
}

/// Conflict severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ConflictSeverity {
    /// Low severity (minor performance impact)
    Low = 0,
    /// Medium severity (noticeable impact)
    Medium = 1,
    /// High severity (significant impact)
    High = 2,
    /// Critical severity (prevents execution)
    Critical = 3,
}

/// Conflict metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictMetadata {
    /// Detection timestamp
    pub detected_at: DateTime<Utc>,

    /// Detection method
    pub detection_method: String,

    /// Detection confidence (0.0 to 1.0)
    pub confidence: f32,

    /// Historical occurrences
    pub historical_occurrences: u64,

    /// Last occurrence
    pub last_occurrence: Option<DateTime<Utc>>,
}

/// Conflict resolution strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionStrategy {
    /// Strategy name
    pub name: String,

    /// Strategy description
    pub description: String,

    /// Implementation difficulty
    pub difficulty: ResolutionDifficulty,

    /// Expected effectiveness (0.0 to 1.0)
    pub effectiveness: f32,

    /// Implementation cost
    pub cost: ResolutionCost,

    /// Strategy parameters
    pub parameters: HashMap<String, String>,

    /// Strategy metadata
    pub metadata: StrategyMetadata,
}

/// Resolution difficulty levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResolutionDifficulty {
    /// Easy (configuration change)
    Easy = 0,
    /// Medium (code modification)
    Medium = 1,
    /// Hard (significant refactoring)
    Hard = 2,
    /// Very Hard (architectural change)
    VeryHard = 3,
}

/// Resolution cost levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResolutionCost {
    /// Low cost
    Low = 0,
    /// Medium cost
    Medium = 1,
    /// High cost
    High = 2,
    /// Very High cost
    VeryHigh = 3,
}

/// Strategy metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyMetadata {
    /// Strategy version
    pub version: String,

    /// Author information
    pub author: String,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Success rate in previous applications (0.0 to 1.0)
    pub success_rate: f32,

    /// Strategy tags
    pub tags: Vec<String>,
}

// ================================================================================================
// Test Grouping Types
// ================================================================================================

/// Test group for parallel execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestGroup {
    /// Group identifier
    pub id: String,

    /// Group name
    pub name: String,

    /// Tests in group
    pub tests: Vec<String>,

    /// Group characteristics
    pub characteristics: GroupCharacteristics,

    /// Execution requirements
    pub requirements: GroupRequirements,

    /// Group priority (0.0 to 1.0)
    pub priority: f32,

    /// Estimated execution time
    pub estimated_duration: Duration,

    /// Group tags
    pub tags: Vec<String>,

    /// Group metadata
    pub metadata: GroupMetadata,
}

/// Group characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupCharacteristics {
    /// Group homogeneity score (0.0 to 1.0)
    pub homogeneity: f32,

    /// Resource compatibility score (0.0 to 1.0)
    pub resource_compatibility: f32,

    /// Performance predictability (0.0 to 1.0)
    pub performance_predictability: f32,

    /// Isolation level
    pub isolation_level: IsolationLevel,

    /// Parallelization efficiency (0.0 to 1.0)
    pub parallelization_efficiency: f32,
}

/// Isolation levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum IsolationLevel {
    /// No isolation required
    None = 0,
    /// Minimal isolation (shared resources OK)
    Minimal = 1,
    /// Moderate isolation (some resource separation)
    Moderate = 2,
    /// High isolation (strong resource separation)
    High = 3,
    /// Complete isolation (no shared resources)
    Complete = 4,
}

/// Group execution requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupRequirements {
    /// Resource requirements
    pub resource_requirements: Vec<ResourceRequirement>,

    /// Environment requirements
    pub environment_requirements: EnvironmentRequirements,

    /// Timing constraints
    pub timing_constraints: TimingConstraints,

    /// Dependency constraints
    pub dependency_constraints: DependencyConstraints,
}

/// Resource requirement
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResourceRequirement {
    /// Resource type
    pub resource_type: String,

    /// Minimum required amount
    pub min_amount: f64,

    /// Maximum required amount
    pub max_amount: f64,

    /// Requirement priority
    pub priority: UsagePriority,

    /// Requirement flexibility
    pub flexibility: RequirementFlexibility,
}

/// Requirement flexibility levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RequirementFlexibility {
    /// Strict requirement (must be met exactly)
    Strict,
    /// Flexible requirement (can be adjusted)
    Flexible,
    /// Optional requirement (nice to have)
    Optional,
}

impl Default for RequirementFlexibility {
    fn default() -> Self {
        RequirementFlexibility::Flexible
    }
}

/// Environment requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentRequirements {
    /// Required environment variables
    pub environment_variables: HashMap<String, String>,

    /// Required system properties
    pub system_properties: HashMap<String, String>,

    /// Required services
    pub required_services: Vec<String>,

    /// Incompatible services
    pub incompatible_services: Vec<String>,
}

/// Timing constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingConstraints {
    /// Maximum execution time
    pub max_execution_time: Duration,

    /// Preferred execution time
    pub preferred_execution_time: Duration,

    /// Setup time requirements
    pub setup_time: Duration,

    /// Teardown time requirements
    pub teardown_time: Duration,
}

/// Dependency constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyConstraints {
    /// Tests that must run before this group
    pub prerequisites: Vec<String>,

    /// Tests that cannot run concurrently with this group
    pub exclusions: Vec<String>,

    /// Tests that should run after this group
    pub dependents: Vec<String>,
}

/// Group metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupMetadata {
    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Creation method
    pub created_by: String,

    /// Group version
    pub version: u64,

    /// Quality metrics
    pub quality_metrics: GroupQualityMetrics,

    /// Historical performance
    pub historical_performance: Vec<GroupPerformanceRecord>,
}

/// Group quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupQualityMetrics {
    /// Cohesion score (0.0 to 1.0)
    pub cohesion: f32,

    /// Coupling score (0.0 to 1.0, lower is better)
    pub coupling: f32,

    /// Stability score (0.0 to 1.0)
    pub stability: f32,

    /// Maintainability score (0.0 to 1.0)
    pub maintainability: f32,
}

/// Group performance record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupPerformanceRecord {
    /// Execution timestamp
    pub executed_at: DateTime<Utc>,

    /// Actual execution time
    pub execution_time: Duration,

    /// Success rate (0.0 to 1.0)
    pub success_rate: f32,

    /// Resource utilization
    pub resource_utilization: HashMap<String, f32>,

    /// Performance notes
    pub notes: String,
}

// ================================================================================================
// Configuration and Management Types
// ================================================================================================

/// Grouping strategy enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GroupingStrategy {
    /// Group by test category
    ByCategory,
    /// Group by resource usage patterns
    ByResourceUsage,
    /// Group by execution time
    ByExecutionTime,
    /// Group by dependencies
    ByDependencies,
    /// Optimal grouping using ML algorithms
    OptimalMl,
    /// Custom grouping strategy
    Custom(String),
}

/// Conflict detection rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictRule {
    /// Rule name
    pub name: String,

    /// Rule description
    pub description: String,

    /// Rule pattern
    pub pattern: ConflictPattern,

    /// Rule action
    pub action: ConflictAction,

    /// Rule confidence (0.0 to 1.0)
    pub confidence: f32,

    /// Rule metadata
    pub metadata: RuleMetadata,
}

/// Conflict pattern specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictPattern {
    /// Resource ID collision
    ResourceIdCollision,
    /// Port range overlap
    PortRangeOverlap,
    /// File path overlap
    FilePathOverlap,
    /// Database table conflict
    DatabaseTableConflict,
    /// Memory region conflict
    MemoryRegionConflict,
    /// Custom pattern
    Custom(String),
}

/// Conflict action specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictAction {
    /// Log the conflict
    Log,
    /// Warn about the conflict
    Warn,
    /// Prevent concurrent execution
    Prevent,
    /// Suggest resolution
    Suggest,
    /// Auto-resolve if possible
    AutoResolve,
    /// Custom action
    Custom(String),
}

/// Rule metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleMetadata {
    /// Rule version
    pub version: String,

    /// Rule author
    pub author: String,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last modified timestamp
    pub modified_at: DateTime<Utc>,

    /// Rule effectiveness (0.0 to 1.0)
    pub effectiveness: f32,

    /// Usage statistics
    pub usage_stats: RuleUsageStats,
}

/// Rule usage statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RuleUsageStats {
    /// Number of times rule was triggered
    pub trigger_count: u64,

    /// Number of true positives
    pub true_positives: u64,

    /// Number of false positives
    pub false_positives: u64,

    /// Number of true negatives
    pub true_negatives: u64,

    /// Number of false negatives
    pub false_negatives: u64,

    /// Precision score (0.0 to 1.0)
    pub precision: f32,

    /// Recall score (0.0 to 1.0)
    pub recall: f32,
}

impl RuleUsageStats {
    /// Update statistics after rule execution
    pub fn update(&mut self, true_positive: bool, false_positive: bool) {
        self.trigger_count += 1;

        if true_positive {
            self.true_positives += 1;
        }

        if false_positive {
            self.false_positives += 1;
        }

        // Update precision and recall
        if self.true_positives + self.false_positives > 0 {
            self.precision =
                self.true_positives as f32 / (self.true_positives + self.false_positives) as f32;
        }

        if self.true_positives + self.false_negatives > 0 {
            self.recall =
                self.true_positives as f32 / (self.true_positives + self.false_negatives) as f32;
        }
    }
}

// ================================================================================================
// Analysis Statistics Types
// ================================================================================================

/// Analysis statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct AnalysisStatistics {
    /// Total analyses performed
    pub total_analyses: u64,

    /// Successful analyses
    pub successful_analyses: u64,

    /// Failed analyses
    pub failed_analyses: u64,

    /// Average analysis time
    pub average_analysis_time: Duration,

    /// Analysis performance by type
    pub performance_by_type: HashMap<String, AnalysisPerformanceMetrics>,

    /// Cache performance
    pub cache_statistics: CacheStatistics,
}

/// Analysis performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisPerformanceMetrics {
    /// Total executions
    pub total_executions: u64,

    /// Average execution time
    pub average_execution_time: Duration,

    /// Minimum execution time
    pub min_execution_time: Duration,

    /// Maximum execution time
    pub max_execution_time: Duration,

    /// Success rate (0.0 to 1.0)
    pub success_rate: f32,

    /// Confidence distribution
    pub confidence_distribution: ConfidenceDistribution,
}

/// Confidence distribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceDistribution {
    /// Very low confidence (0.0-0.2)
    pub very_low: u64,

    /// Low confidence (0.2-0.4)
    pub low: u64,

    /// Medium confidence (0.4-0.6)
    pub medium: u64,

    /// High confidence (0.6-0.8)
    pub high: u64,

    /// Very high confidence (0.8-1.0)
    pub very_high: u64,
}

impl ConfidenceDistribution {
    /// Add a confidence score to the distribution
    pub fn add_confidence(&mut self, confidence: f32) {
        match confidence {
            c if c < 0.2 => self.very_low += 1,
            c if c < 0.4 => self.low += 1,
            c if c < 0.6 => self.medium += 1,
            c if c < 0.8 => self.high += 1,
            _ => self.very_high += 1,
        }
    }
}

// ============================================================================
// Additional Test Independence Analyzer Types
// ============================================================================

/// Database statistics for resource database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseStatistics {
    /// Total resources tracked
    pub total_resources: u64,
    /// Active resources
    pub active_resources: u32,
    /// Database size in bytes
    pub database_size: u64,
    /// Query count
    pub query_count: u64,
    /// Average query time
    pub avg_query_time: Duration,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

impl Default for DatabaseStatistics {
    fn default() -> Self {
        Self {
            total_resources: 0,
            active_resources: 0,
            database_size: 0,
            query_count: 0,
            avg_query_time: Duration::from_secs(0),
            cache_hit_rate: 0.0,
        }
    }
}

/// Comparison operator for test grouping rules
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOperator {
    /// Equal to
    Equal,
    /// Not equal to
    NotEqual,
    /// Greater than
    GreaterThan,
    /// Less than
    LessThan,
    /// Greater than or equal
    GreaterThanOrEqual,
    /// Less than or equal
    LessThanOrEqual,
    /// Contains
    Contains,
    /// Not contains
    NotContains,
    /// Matches regex
    Matches,
}
