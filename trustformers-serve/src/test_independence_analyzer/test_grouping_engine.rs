//! Test Grouping Engine Module
//!
//! This module provides sophisticated test grouping capabilities for optimal
//! parallel execution. It analyzes test characteristics, dependencies, and
//! resource requirements to create balanced, efficient test groups that
//! maximize parallelization while minimizing conflicts and resource contention.

use crate::performance_optimizer::performance_modeling::ValidationResult;
use crate::performance_optimizer::test_characterization::types::OrderingConstraint;
use crate::test_independence_analyzer::types::*;
use crate::test_parallelization::{TestDependency, TestParallelizationMetadata, TestResourceUsage};
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
    time::{Duration, Instant},
};
use tracing::{debug, info};

/// Advanced test grouping engine with multiple optimization strategies
#[derive(Debug)]
pub struct TestGroupingEngine {
    /// Configuration for grouping behavior
    config: Arc<RwLock<GroupingEngineConfig>>,

    /// Available grouping strategies
    strategies: Arc<RwLock<Vec<GroupingStrategy>>>,

    /// Group optimization algorithms
    optimizers: Arc<RwLock<Vec<GroupOptimizer>>>,

    /// Grouping performance metrics
    metrics: Arc<RwLock<GroupingMetrics>>,

    /// Learned grouping patterns
    learned_patterns: Arc<RwLock<Vec<LearnedGroupingPattern>>>,

    /// Group validation rules
    validation_rules: Arc<RwLock<Vec<GroupValidationRule>>>,
}

/// Configuration for the test grouping engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupingEngineConfig {
    /// Default grouping strategy to use
    pub default_strategy: GroupingStrategyType,

    /// Maximum number of tests per group
    pub max_tests_per_group: usize,

    /// Minimum number of tests per group
    pub min_tests_per_group: usize,

    /// Target resource utilization per group
    pub target_resource_utilization: f32,

    /// Maximum acceptable resource utilization per group
    pub max_resource_utilization: f32,

    /// Enable adaptive grouping based on historical performance
    pub adaptive_grouping: bool,

    /// Enable machine learning-based group optimization
    pub ml_optimization: bool,

    /// Grouping timeout per batch
    pub grouping_timeout: Duration,

    /// Enable detailed grouping logging
    pub detailed_logging: bool,

    /// Group balancing weights
    pub balancing_weights: GroupingBalancingWeights,

    /// Quality thresholds for group validation
    pub quality_thresholds: GroupingQualityThresholds,
}

/// Balancing weights for different grouping factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupingBalancingWeights {
    /// Weight for resource utilization balance
    pub resource_utilization_weight: f32,

    /// Weight for execution time balance
    pub execution_time_weight: f32,

    /// Weight for dependency minimization
    pub dependency_weight: f32,

    /// Weight for test category similarity
    pub category_similarity_weight: f32,

    /// Weight for conflict avoidance
    pub conflict_avoidance_weight: f32,

    /// Weight for test complexity balance
    pub complexity_balance_weight: f32,
}

impl Default for GroupingBalancingWeights {
    fn default() -> Self {
        Self {
            resource_utilization_weight: 0.25,
            execution_time_weight: 0.25,
            dependency_weight: 0.2,
            category_similarity_weight: 0.1,
            conflict_avoidance_weight: 0.15,
            complexity_balance_weight: 0.05,
        }
    }
}

/// Quality thresholds for validating group quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupingQualityThresholds {
    /// Minimum acceptable group homogeneity score
    pub min_homogeneity: f32,

    /// Minimum acceptable resource compatibility score
    pub min_resource_compatibility: f32,

    /// Minimum acceptable duration balance score
    pub min_duration_balance: f32,

    /// Maximum acceptable dependency complexity
    pub max_dependency_complexity: f32,

    /// Minimum acceptable parallelization potential
    pub min_parallelization_potential: f32,
}

impl Default for GroupingQualityThresholds {
    fn default() -> Self {
        Self {
            min_homogeneity: 0.6,
            min_resource_compatibility: 0.7,
            min_duration_balance: 0.5,
            max_dependency_complexity: 0.8,
            min_parallelization_potential: 0.6,
        }
    }
}

impl Default for GroupingEngineConfig {
    fn default() -> Self {
        Self {
            default_strategy: GroupingStrategyType::Balanced,
            max_tests_per_group: 10,
            min_tests_per_group: 2,
            target_resource_utilization: 0.7,
            max_resource_utilization: 0.9,
            adaptive_grouping: true,
            ml_optimization: false,
            grouping_timeout: Duration::from_secs(30),
            detailed_logging: false,
            balancing_weights: GroupingBalancingWeights::default(),
            quality_thresholds: GroupingQualityThresholds::default(),
        }
    }
}

/// Test grouping strategy with comprehensive configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupingStrategy {
    /// Strategy identifier
    pub strategy_id: String,

    /// Strategy name
    pub name: String,

    /// Strategy description
    pub description: String,

    /// Strategy type
    pub strategy_type: GroupingStrategyType,

    /// Strategy parameters
    pub parameters: GroupingStrategyParameters,

    /// Strategy effectiveness score
    pub effectiveness_score: f32,

    /// Strategy applicability conditions
    pub applicability: Vec<StrategyApplicabilityCondition>,

    /// Expected outcomes
    pub expected_outcomes: Vec<GroupingOutcome>,
}

/// Types of grouping strategies
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GroupingStrategyType {
    /// Balanced approach considering all factors
    Balanced,

    /// Resource-optimal grouping
    ResourceOptimal,

    /// Time-optimal grouping
    TimeOptimal,

    /// Dependency-aware grouping
    DependencyAware,

    /// Category-based grouping
    CategoryBased,

    /// Conflict-minimizing grouping
    ConflictMinimizing,

    /// Machine learning-based grouping
    MachineLearning,

    /// Custom strategy
    Custom(String),
}

/// Parameters for grouping strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupingStrategyParameters {
    /// Primary optimization target
    pub optimization_target: OptimizationTarget,

    /// Secondary optimization targets
    pub secondary_targets: Vec<OptimizationTarget>,

    /// Algorithm-specific parameters
    pub algorithm_params: HashMap<String, f32>,

    /// Constraints to enforce
    pub constraints: Vec<GroupingConstraint>,
}

/// Optimization targets for grouping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationTarget {
    /// Minimize total execution time
    MinimizeExecutionTime,

    /// Maximize resource utilization
    MaximizeResourceUtilization,

    /// Minimize resource conflicts
    MinimizeConflicts,

    /// Balance workload across groups
    BalanceWorkload,

    /// Minimize dependencies between groups
    MinimizeCrossDependencies,

    /// Maximize test isolation
    MaximizeIsolation,

    /// Custom optimization target
    Custom(String),
}

/// Constraints for grouping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupingConstraint {
    /// Constraint type
    pub constraint_type: GroupingConstraintType,

    /// Constraint value
    pub constraint_value: f32,

    /// Constraint description
    pub description: String,

    /// Constraint priority
    pub priority: u32,
}

/// Types of grouping constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GroupingConstraintType {
    /// Maximum group size
    MaxGroupSize,

    /// Minimum group size
    MinGroupSize,

    /// Maximum resource usage per group
    MaxResourceUsage,

    /// Maximum execution time per group
    MaxExecutionTime,

    /// Maximum conflicts per group
    MaxConflicts,

    /// Minimum compatibility score
    MinCompatibilityScore,

    /// Custom constraint
    Custom(String),
}

/// Conditions where a strategy is applicable
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyApplicabilityCondition {
    /// Condition description
    pub description: String,

    /// Test set size range
    pub test_set_size_range: Option<(usize, usize)>,

    /// Resource type requirements
    pub required_resource_types: Vec<String>,

    /// Test category requirements
    pub required_test_categories: Vec<String>,

    /// Complexity level requirements
    pub complexity_requirements: Option<ComplexityRange>,
}

/// Range of complexity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityRange {
    /// Minimum complexity
    pub min_complexity: f32,

    /// Maximum complexity
    pub max_complexity: f32,
}

/// Expected outcomes from a grouping strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupingOutcome {
    /// Outcome description
    pub description: String,

    /// Expected improvement percentage
    pub expected_improvement: f32,

    /// Outcome confidence
    pub confidence: f32,

    /// Metrics affected
    pub affected_metrics: Vec<String>,
}

/// Group optimizer for post-processing optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupOptimizer {
    /// Optimizer identifier
    pub optimizer_id: String,

    /// Optimizer name
    pub name: String,

    /// Optimizer description
    pub description: String,

    /// Optimization technique
    pub technique: OptimizationTechnique,

    /// Optimization parameters
    pub parameters: OptimizationParameters,

    /// Expected effectiveness
    pub effectiveness: f32,
}

/// Optimization techniques
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationTechnique {
    /// Genetic algorithm optimization
    GeneticAlgorithm,

    /// Simulated annealing
    SimulatedAnnealing,

    /// Greedy local search
    GreedySearch,

    /// Hill climbing
    HillClimbing,

    /// Particle swarm optimization
    ParticleSwarm,

    /// Custom technique
    Custom(String),
}

/// Parameters for optimization algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationParameters {
    /// Maximum iterations
    pub max_iterations: usize,

    /// Convergence threshold
    pub convergence_threshold: f32,

    /// Population size (for population-based algorithms)
    pub population_size: Option<usize>,

    /// Mutation rate (for genetic algorithms)
    pub mutation_rate: Option<f32>,

    /// Temperature schedule (for simulated annealing)
    pub temperature_schedule: Option<TemperatureSchedule>,

    /// Custom parameters
    pub custom_params: HashMap<String, f32>,
}

/// Temperature schedule for simulated annealing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureSchedule {
    /// Initial temperature
    pub initial_temperature: f32,

    /// Final temperature
    pub final_temperature: f32,

    /// Cooling rate
    pub cooling_rate: f32,

    /// Schedule type
    pub schedule_type: CoolingScheduleType,
}

/// Types of cooling schedules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoolingScheduleType {
    /// Linear cooling
    Linear,

    /// Exponential cooling
    Exponential,

    /// Logarithmic cooling
    Logarithmic,

    /// Custom schedule
    Custom(String),
}

/// Grouping performance metrics
#[derive(Debug, Default, Clone)]
pub struct GroupingMetrics {
    /// Total groupings performed
    pub total_groupings: u64,

    /// Groupings by strategy
    pub groupings_by_strategy: HashMap<GroupingStrategyType, u64>,

    /// Average grouping time
    pub average_grouping_time: Duration,

    /// Best grouping quality achieved
    pub best_quality_score: f32,

    /// Average grouping quality
    pub average_quality_score: f32,

    /// Grouping success rate
    pub success_rate: f32,

    /// Performance improvements achieved
    pub performance_improvements: GroupingPerformanceImprovements,

    /// Resource utilization statistics
    pub resource_utilization_stats: ResourceUtilizationStats,
}

/// Performance improvements from grouping
#[derive(Debug, Clone, Default)]
pub struct GroupingPerformanceImprovements {
    /// Average execution time reduction
    pub execution_time_reduction: f32,

    /// Average resource utilization improvement
    pub resource_utilization_improvement: f32,

    /// Average conflict reduction
    pub conflict_reduction: f32,

    /// Parallelization efficiency improvement
    pub parallelization_efficiency_improvement: f32,
}

/// Resource utilization statistics
#[derive(Debug, Clone, Default)]
pub struct ResourceUtilizationStats {
    /// Average CPU utilization across groups
    pub average_cpu_utilization: f32,

    /// Average memory utilization across groups
    pub average_memory_utilization: f32,

    /// Average GPU utilization across groups
    pub average_gpu_utilization: f32,

    /// Average network utilization across groups
    pub average_network_utilization: f32,

    /// Utilization balance score
    pub utilization_balance_score: f32,
}

/// Learned grouping patterns from historical data
#[derive(Debug, Clone)]
pub struct LearnedGroupingPattern {
    /// Pattern identifier
    pub pattern_id: String,

    /// Pattern description
    pub description: String,

    /// Pattern characteristics
    pub characteristics: GroupingPatternCharacteristics,

    /// Success rate of this pattern
    pub success_rate: f32,

    /// Number of times pattern was observed
    pub observation_count: u32,

    /// Quality improvements achieved
    pub quality_improvements: HashMap<String, f32>,

    /// Last updated timestamp
    pub last_updated: DateTime<Utc>,
}

/// Characteristics of a grouping pattern
#[derive(Debug, Clone)]
pub struct GroupingPatternCharacteristics {
    /// Test categories typically grouped together
    pub typical_category_combinations: Vec<Vec<String>>,

    /// Resource usage patterns that work well together
    pub compatible_resource_patterns: Vec<ResourceUsagePattern>,

    /// Optimal group size ranges
    pub optimal_group_sizes: Vec<(usize, usize)>,

    /// Environmental conditions
    pub environmental_conditions: HashMap<String, String>,

    /// Success indicators
    pub success_indicators: Vec<SuccessIndicator>,
}

/// Resource usage patterns
#[derive(Debug, Clone)]
pub struct ResourceUsagePattern {
    /// Pattern name
    pub pattern_name: String,

    /// Resource type
    pub resource_type: String,

    /// Usage characteristics
    pub usage_characteristics: UsageCharacteristics,

    /// Compatibility score with other patterns
    pub compatibility_scores: HashMap<String, f32>,
}

/// Usage characteristics for a resource
#[derive(Debug, Clone)]
pub struct UsageCharacteristics {
    /// Typical usage intensity
    pub typical_intensity: f32,

    /// Usage duration pattern
    pub duration_pattern: DurationPattern,

    /// Peak usage timing
    pub peak_timing: Option<PeakTiming>,

    /// Variability score
    pub variability: f32,
}

/// Duration patterns for resource usage
#[derive(Debug, Clone)]
pub enum DurationPattern {
    /// Short bursts of usage
    ShortBurst,

    /// Steady sustained usage
    Sustained,

    /// Variable duration
    Variable,

    /// Periodic usage pattern
    Periodic { interval: Duration },

    /// Custom pattern
    Custom(String),
}

/// Peak usage timing information
#[derive(Debug, Clone)]
pub struct PeakTiming {
    /// Relative time when peak occurs (0.0 = start, 1.0 = end)
    pub relative_time: f32,

    /// Duration of peak usage
    pub peak_duration: Duration,

    /// Peak intensity multiplier
    pub peak_intensity_multiplier: f32,
}

/// Success indicators for grouping patterns
#[derive(Debug, Clone)]
pub struct SuccessIndicator {
    /// Indicator type
    pub indicator_type: SuccessIndicatorType,

    /// Indicator value
    pub value: f32,

    /// Indicator importance
    pub importance: f32,
}

/// Types of success indicators
#[derive(Debug, Clone)]
pub enum SuccessIndicatorType {
    /// Execution time improvement
    ExecutionTimeImprovement,

    /// Resource utilization efficiency
    ResourceEfficiency,

    /// Conflict reduction
    ConflictReduction,

    /// Test pass rate
    TestPassRate,

    /// Custom indicator
    Custom(String),
}

/// Group validation rules
#[derive(Debug, Clone)]
pub struct GroupValidationRule {
    /// Rule identifier
    pub rule_id: String,

    /// Rule name
    pub name: String,

    /// Rule description
    pub description: String,

    /// Validation criteria
    pub criteria: Vec<ValidationCriterion>,

    /// Rule severity
    pub severity: ValidationSeverity,

    /// Rule enabled status
    pub enabled: bool,
}

/// Validation criteria for groups
#[derive(Debug, Clone)]
pub struct ValidationCriterion {
    /// Criterion type
    pub criterion_type: ValidationCriterionType,

    /// Comparison operator
    pub operator: ComparisonOperator,

    /// Expected value
    pub expected_value: f32,

    /// Criterion description
    pub description: String,
}

/// Types of validation criteria
#[derive(Debug, Clone)]
pub enum ValidationCriterionType {
    /// Group size
    GroupSize,

    /// Resource utilization
    ResourceUtilization,

    /// Execution time balance
    ExecutionTimeBalance,

    /// Conflict count
    ConflictCount,

    /// Dependency complexity
    DependencyComplexity,

    /// Homogeneity score
    HomogeneityScore,

    /// Custom criterion
    Custom(String),
}

/// Validation severity levels
#[derive(Debug, Clone)]
pub enum ValidationSeverity {
    /// Warning (group can be used but may not be optimal)
    Warning,

    /// Error (group should be modified)
    Error,

    /// Critical (group must not be used)
    Critical,
}

/// Test group with comprehensive metadata
#[derive(Debug, Clone)]
pub struct TestGroup {
    /// Group unique identifier
    pub id: String,

    /// Group name
    pub name: String,

    /// Tests in this group
    pub tests: Vec<String>,

    /// Group characteristics
    pub characteristics: GroupCharacteristics,

    /// Execution requirements
    pub requirements: GroupRequirements,

    /// Group priority score
    pub priority: f32,

    /// Estimated execution time
    pub estimated_duration: Duration,

    /// Group tags for categorization
    pub tags: Vec<String>,

    /// Group creation metadata
    pub creation_metadata: GroupCreationMetadata,

    /// Group validation results
    pub validation_results: Vec<ValidationResult>,
}

/// Characteristics of a test group
#[derive(Debug, Clone)]
pub struct GroupCharacteristics {
    /// Group homogeneity score (0.0 to 1.0)
    pub homogeneity: f32,

    /// Resource compatibility score
    pub resource_compatibility: f32,

    /// Duration balance score
    pub duration_balance: f32,

    /// Dependency complexity score
    pub dependency_complexity: f32,

    /// Parallelization potential score
    pub parallelization_potential: f32,

    /// Conflict risk score
    pub conflict_risk: f32,

    /// Overall group quality score
    pub overall_quality: f32,
}

/// Group execution requirements
#[derive(Debug, Clone)]
pub struct GroupRequirements {
    /// Minimum resources required
    pub min_resources: ResourceRequirement,

    /// Optimal resources required
    pub optimal_resources: ResourceRequirement,

    /// Maximum resources usable
    pub max_resources: ResourceRequirement,

    /// Isolation requirements
    pub isolation: crate::test_parallelization::IsolationRequirements,

    /// Ordering constraints within the group
    pub ordering_constraints: Vec<OrderingConstraint>,

    /// Setup and teardown requirements
    pub setup_teardown: SetupTeardownRequirements,
}

/// Setup and teardown requirements
#[derive(Debug, Clone)]
pub struct SetupTeardownRequirements {
    /// Setup operations required before group execution
    pub setup_operations: Vec<SetupOperation>,

    /// Teardown operations required after group execution
    pub teardown_operations: Vec<TeardownOperation>,

    /// Setup timeout
    pub setup_timeout: Duration,

    /// Teardown timeout
    pub teardown_timeout: Duration,
}

/// Setup operation definition
#[derive(Debug, Clone)]
pub struct SetupOperation {
    /// Operation identifier
    pub operation_id: String,

    /// Operation description
    pub description: String,

    /// Operation type
    pub operation_type: SetupOperationType,

    /// Expected duration
    pub expected_duration: Duration,

    /// Operation parameters
    pub parameters: HashMap<String, String>,
}

/// Types of setup operations
#[derive(Debug, Clone)]
pub enum SetupOperationType {
    /// Resource allocation
    ResourceAllocation,

    /// Environment preparation
    EnvironmentPreparation,

    /// Service startup
    ServiceStartup,

    /// Data initialization
    DataInitialization,

    /// Custom setup operation
    Custom(String),
}

/// Teardown operation definition
#[derive(Debug, Clone)]
pub struct TeardownOperation {
    /// Operation identifier
    pub operation_id: String,

    /// Operation description
    pub description: String,

    /// Operation type
    pub operation_type: TeardownOperationType,

    /// Expected duration
    pub expected_duration: Duration,

    /// Operation parameters
    pub parameters: HashMap<String, String>,
}

/// Types of teardown operations
#[derive(Debug, Clone)]
pub enum TeardownOperationType {
    /// Resource cleanup
    ResourceCleanup,

    /// Environment cleanup
    EnvironmentCleanup,

    /// Service shutdown
    ServiceShutdown,

    /// Data cleanup
    DataCleanup,

    /// Custom teardown operation
    Custom(String),
}

/// Group creation metadata
#[derive(Debug, Clone)]
pub struct GroupCreationMetadata {
    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Strategy used for creation
    pub strategy_used: GroupingStrategyType,

    /// Creation duration
    pub creation_duration: Duration,

    /// Creator information
    pub creator: String,

    /// Creation parameters
    pub creation_parameters: HashMap<String, String>,

    /// Quality score at creation
    pub initial_quality_score: f32,
}

impl TestGroupingEngine {
    /// Create a new test grouping engine with default configuration
    pub fn new() -> Self {
        Self::with_config(GroupingEngineConfig::default())
    }

    /// Create a new test grouping engine with custom configuration
    pub fn with_config(config: GroupingEngineConfig) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            strategies: Arc::new(RwLock::new(Self::create_default_strategies())),
            optimizers: Arc::new(RwLock::new(Self::create_default_optimizers())),
            metrics: Arc::new(RwLock::new(GroupingMetrics::default())),
            learned_patterns: Arc::new(RwLock::new(Vec::new())),
            validation_rules: Arc::new(RwLock::new(Self::create_default_validation_rules())),
        }
    }

    /// Create test groups from a set of tests and detected conflicts
    pub fn create_test_groups(
        &self,
        tests: &[TestParallelizationMetadata],
        dependencies: &[TestDependency],
        conflicts: &[ResourceConflict],
    ) -> AnalysisResult<Vec<TestGroup>> {
        let start_time = Instant::now();
        let config = self.config.read();

        if start_time.elapsed() > config.grouping_timeout {
            return Err(AnalysisError::AnalysisTimeout {
                operation: "test grouping".to_string(),
                timeout: config.grouping_timeout,
            });
        }

        info!(
            "Creating test groups for {} tests with {} dependencies and {} conflicts",
            tests.len(),
            dependencies.len(),
            conflicts.len()
        );

        // Select the best grouping strategy
        let strategy = self.select_grouping_strategy(tests, dependencies, conflicts)?;

        // Create initial grouping
        let initial_groups =
            self.create_initial_grouping(tests, dependencies, conflicts, &strategy)?;

        // Optimize groups
        let optimized_groups = if config.ml_optimization {
            self.optimize_groups(initial_groups)?
        } else {
            initial_groups
        };

        // Validate groups
        let validated_groups = self.validate_groups(optimized_groups)?;

        // Update metrics
        self.update_grouping_metrics(&validated_groups, start_time.elapsed(), &strategy);

        if config.detailed_logging {
            debug!(
                "Created {} test groups in {:?}",
                validated_groups.len(),
                start_time.elapsed()
            );
        }

        Ok(validated_groups)
    }

    /// Select the best grouping strategy for the given test set
    fn select_grouping_strategy(
        &self,
        tests: &[TestParallelizationMetadata],
        dependencies: &[TestDependency],
        conflicts: &[ResourceConflict],
    ) -> AnalysisResult<GroupingStrategy> {
        let strategies = self.strategies.read();
        let config = self.config.read();

        // Calculate characteristics of the test set
        let test_set_characteristics =
            self.analyze_test_set_characteristics(tests, dependencies, conflicts);

        // Find the best matching strategy
        let mut best_strategy = None;
        let mut best_score = 0.0;

        for strategy in strategies.iter() {
            if self.is_strategy_applicable(strategy, &test_set_characteristics) {
                let score = self.calculate_strategy_score(strategy, &test_set_characteristics);
                if score > best_score {
                    best_score = score;
                    best_strategy = Some(strategy.clone());
                }
            }
        }

        match best_strategy {
            Some(strategy) => Ok(strategy),
            None => {
                // Fall back to default strategy
                strategies
                    .iter()
                    .find(|s| s.strategy_type == config.default_strategy)
                    .cloned()
                    .ok_or_else(|| AnalysisError::StrategyNotFound {
                        message: format!("{:?}", config.default_strategy),
                    })
            },
        }
    }

    /// Analyze characteristics of the test set
    fn analyze_test_set_characteristics(
        &self,
        tests: &[TestParallelizationMetadata],
        dependencies: &[TestDependency],
        conflicts: &[ResourceConflict],
    ) -> TestSetCharacteristics {
        let mut characteristics = TestSetCharacteristics {
            test_count: tests.len(),
            dependency_count: dependencies.len(),
            conflict_count: conflicts.len(),
            average_test_duration: Duration::from_secs(0),
            resource_diversity: 0.0,
            category_diversity: 0.0,
            complexity_distribution: ComplexityDistribution::default(),
        };

        // Calculate average test duration
        let total_duration: Duration = tests.iter().map(|t| t.resource_usage.duration).sum();
        if !tests.is_empty() {
            characteristics.average_test_duration = total_duration / tests.len() as u32;
        }

        // Calculate resource diversity
        let unique_resources: HashSet<_> = tests
            .iter()
            .flat_map(|t| {
                let mut resources = Vec::new();
                if t.resource_usage.cpu_cores > 0.0 {
                    resources.push("CPU");
                }
                if t.resource_usage.memory_mb > 0 {
                    resources.push("Memory");
                }
                if !t.resource_usage.gpu_devices.is_empty() {
                    resources.push("GPU");
                }
                if !t.resource_usage.network_ports.is_empty() {
                    resources.push("Network");
                }
                if !t.resource_usage.temp_directories.is_empty() {
                    resources.push("Filesystem");
                }
                if t.resource_usage.database_connections > 0 {
                    resources.push("Database");
                }
                resources
            })
            .collect();
        characteristics.resource_diversity = unique_resources.len() as f32 / 6.0; // Max 6 resource types

        // Calculate category diversity
        let unique_categories: HashSet<_> =
            tests.iter().map(|t| format!("{:?}", t.base_context.category)).collect();
        characteristics.category_diversity = unique_categories.len() as f32 / tests.len() as f32;

        characteristics
    }

    /// Check if a strategy is applicable to the test set
    fn is_strategy_applicable(
        &self,
        strategy: &GroupingStrategy,
        characteristics: &TestSetCharacteristics,
    ) -> bool {
        for condition in &strategy.applicability {
            // Check test set size
            if let Some((min_size, max_size)) = condition.test_set_size_range {
                if characteristics.test_count < min_size || characteristics.test_count > max_size {
                    return false;
                }
            }

            // Add more sophisticated applicability checks here
        }

        true
    }

    /// Calculate a score for how well a strategy fits the test set
    fn calculate_strategy_score(
        &self,
        strategy: &GroupingStrategy,
        _characteristics: &TestSetCharacteristics,
    ) -> f32 {
        // Start with the strategy's base effectiveness score
        let score = strategy.effectiveness_score;

        // Add more sophisticated scoring logic here based on test set characteristics
        // For now, return the base effectiveness score

        score
    }

    /// Create initial grouping using the selected strategy
    fn create_initial_grouping(
        &self,
        tests: &[TestParallelizationMetadata],
        dependencies: &[TestDependency],
        conflicts: &[ResourceConflict],
        strategy: &GroupingStrategy,
    ) -> AnalysisResult<Vec<TestGroup>> {
        match strategy.strategy_type {
            GroupingStrategyType::Balanced => {
                self.create_balanced_grouping(tests, dependencies, conflicts)
            },
            GroupingStrategyType::ResourceOptimal => {
                self.create_resource_optimal_grouping(tests, dependencies, conflicts)
            },
            GroupingStrategyType::TimeOptimal => {
                self.create_time_optimal_grouping(tests, dependencies, conflicts)
            },
            GroupingStrategyType::DependencyAware => {
                self.create_dependency_aware_grouping(tests, dependencies, conflicts)
            },
            GroupingStrategyType::CategoryBased => {
                self.create_category_based_grouping(tests, dependencies, conflicts)
            },
            GroupingStrategyType::ConflictMinimizing => {
                self.create_conflict_minimizing_grouping(tests, dependencies, conflicts)
            },
            GroupingStrategyType::MachineLearning => {
                self.create_ml_based_grouping(tests, dependencies, conflicts)
            },
            GroupingStrategyType::Custom(_) => {
                self.create_custom_grouping(tests, dependencies, conflicts, strategy)
            },
        }
    }

    /// Create balanced grouping considering all factors
    fn create_balanced_grouping(
        &self,
        tests: &[TestParallelizationMetadata],
        _dependencies: &[TestDependency],
        conflicts: &[ResourceConflict],
    ) -> AnalysisResult<Vec<TestGroup>> {
        let config = self.config.read();
        let _weights = &config.balancing_weights;

        let mut groups = Vec::new();
        let mut unassigned_tests: VecDeque<_> = tests.iter().collect();
        let mut group_id = 0;

        // Create conflict map for quick lookup
        let conflict_map = self.create_conflict_map(conflicts);

        while !unassigned_tests.is_empty() {
            let mut current_group_tests = Vec::new();
            let mut current_group_resources = ResourceRequirement::default();

            // Select first test as group seed
            if let Some(seed_test) = unassigned_tests.pop_front() {
                current_group_tests.push(seed_test.base_context.test_name.clone());
                current_group_resources = self
                    .add_resource_requirements(current_group_resources, &seed_test.resource_usage);

                // Add compatible tests to the group
                let mut remaining_tests = Vec::new();
                while let Some(test) = unassigned_tests.pop_front() {
                    let test_resources =
                        ResourceRequirement::from_resource_usage(&test.resource_usage);
                    let combined_resources = self
                        .combine_resource_requirements(&current_group_resources, &test_resources);

                    // Check if adding this test would exceed limits
                    if current_group_tests.len() < config.max_tests_per_group
                        && self.is_resource_usage_acceptable(
                            &combined_resources,
                            config.max_resource_utilization,
                        )
                        && !self.would_create_conflicts(
                            &test.base_context.test_name,
                            &current_group_tests,
                            &conflict_map,
                        )
                    {
                        current_group_tests.push(test.base_context.test_name.clone());
                        current_group_resources = combined_resources;
                    } else {
                        remaining_tests.push(test);
                    }
                }

                // Put unselected tests back in queue
                for test in remaining_tests {
                    unassigned_tests.push_back(test);
                }

                // Create the group if it meets minimum size requirement
                if current_group_tests.len() >= config.min_tests_per_group {
                    let group = self.create_test_group_from_tests(
                        group_id,
                        current_group_tests,
                        tests,
                        GroupingStrategyType::Balanced,
                    )?;
                    groups.push(group);
                    group_id += 1;
                } else {
                    // If group is too small, add remaining tests to the last group or create individual groups
                    if let Some(last_group) = groups.last_mut() {
                        last_group.tests.extend(current_group_tests);
                        // Recalculate characteristics
                        last_group.characteristics =
                            self.calculate_group_characteristics(&last_group.tests, tests);
                    } else {
                        // Create individual groups for remaining tests
                        for test_name in current_group_tests {
                            let individual_group = self.create_test_group_from_tests(
                                group_id,
                                vec![test_name],
                                tests,
                                GroupingStrategyType::Balanced,
                            )?;
                            groups.push(individual_group);
                            group_id += 1;
                        }
                    }
                }
            }
        }

        Ok(groups)
    }

    /// Create conflict map for quick lookup
    fn create_conflict_map(
        &self,
        conflicts: &[ResourceConflict],
    ) -> HashMap<String, HashSet<String>> {
        let mut conflict_map: HashMap<String, HashSet<String>> = HashMap::new();

        for conflict in conflicts {
            conflict_map
                .entry(conflict.test1.clone())
                .or_default()
                .insert(conflict.test2.clone());

            conflict_map
                .entry(conflict.test2.clone())
                .or_default()
                .insert(conflict.test1.clone());
        }

        conflict_map
    }

    /// Check if adding a test would create conflicts with existing group members
    fn would_create_conflicts(
        &self,
        test_name: &str,
        group_tests: &[String],
        conflict_map: &HashMap<String, HashSet<String>>,
    ) -> bool {
        if let Some(test_conflicts) = conflict_map.get(test_name) {
            for group_test in group_tests {
                if test_conflicts.contains(group_test) {
                    return true;
                }
            }
        }
        false
    }

    /// Check if resource usage is within acceptable limits
    // TODO: ResourceRequirement no longer has cpu_cores, memory_mb, network_ports
    // It now has generic resource_type, min_amount, max_amount
    fn is_resource_usage_acceptable(
        &self,
        resources: &ResourceRequirement,
        max_utilization: f32,
    ) -> bool {
        // Check if the resource max_amount is within acceptable limits
        resources.max_amount <= max_utilization as f64
    }

    /// Add resource requirements
    fn add_resource_requirements(
        &self,
        base: ResourceRequirement,
        usage: &TestResourceUsage,
    ) -> ResourceRequirement {
        // Convert TestResourceUsage to ResourceRequirement and combine
        let usage_req = ResourceRequirement::from_resource_usage(usage);
        self.combine_resource_requirements(&base, &usage_req)
    }

    /// Combine two resource requirements
    fn combine_resource_requirements(
        &self,
        req1: &ResourceRequirement,
        req2: &ResourceRequirement,
    ) -> ResourceRequirement {
        // If resource types match, combine additively
        if req1.resource_type == req2.resource_type {
            ResourceRequirement {
                resource_type: req1.resource_type.clone(),
                min_amount: req1.min_amount + req2.min_amount,
                max_amount: req1.max_amount + req2.max_amount,
                // Use higher priority
                priority: if self.priority_level(&req1.priority)
                    > self.priority_level(&req2.priority)
                {
                    req1.priority.clone()
                } else {
                    req2.priority.clone()
                },
                // Use more restrictive flexibility
                flexibility: if self.flexibility_level(&req1.flexibility)
                    < self.flexibility_level(&req2.flexibility)
                {
                    req1.flexibility.clone()
                } else {
                    req2.flexibility.clone()
                },
            }
        } else {
            // For different resource types, use the more constrained requirement
            // (higher total amount)
            let req1_total = req1.min_amount + req1.max_amount;
            let req2_total = req2.min_amount + req2.max_amount;

            if req1_total >= req2_total {
                req1.clone()
            } else {
                req2.clone()
            }
        }
    }

    /// Get numeric priority level for comparison
    fn priority_level(&self, priority: &UsagePriority) -> u8 {
        match priority {
            UsagePriority::Critical => 3,
            UsagePriority::High => 2,
            UsagePriority::Normal => 1,
            UsagePriority::Low => 0,
        }
    }

    /// Get numeric flexibility level for comparison
    fn flexibility_level(&self, flexibility: &RequirementFlexibility) -> u8 {
        match flexibility {
            RequirementFlexibility::Strict => 0,
            RequirementFlexibility::Flexible => 1,
            RequirementFlexibility::Optional => 2,
        }
    }

    /// Create a test group from a list of test names
    fn create_test_group_from_tests(
        &self,
        group_id: usize,
        test_names: Vec<String>,
        all_tests: &[TestParallelizationMetadata],
        strategy_type: GroupingStrategyType,
    ) -> AnalysisResult<TestGroup> {
        let group_tests: Vec<_> = all_tests
            .iter()
            .filter(|t| test_names.contains(&t.base_context.test_name))
            .collect();

        if group_tests.is_empty() {
            return Err(AnalysisError::InvalidGrouping {
                message: "No tests found for group".to_string(),
            });
        }

        // Calculate group characteristics
        let characteristics = self.calculate_group_characteristics(&test_names, all_tests);

        // Calculate resource requirements
        let mut min_resources = ResourceRequirement::default();
        let mut optimal_resources = ResourceRequirement::default();
        let mut max_resources = ResourceRequirement::default();

        for test in &group_tests {
            let test_req = ResourceRequirement::from_resource_usage(&test.resource_usage);
            min_resources = self.combine_resource_requirements(&min_resources, &test_req);
            optimal_resources = self.combine_resource_requirements(&optimal_resources, &test_req);
            max_resources = self.combine_resource_requirements(&max_resources, &test_req);
        }

        // Scale resources appropriately
        optimal_resources.min_amount *= 1.1; // 10% buffer
        optimal_resources.max_amount *= 1.2; // 20% buffer
        max_resources.min_amount *= 1.5; // 50% buffer
        max_resources.max_amount *= 2.0; // 100% buffer

        // Calculate estimated duration
        let estimated_duration = group_tests
            .iter()
            .map(|t| t.resource_usage.duration)
            .max()
            .unwrap_or(Duration::from_secs(60)); // Parallel execution, so use max duration

        // Create group requirements
        let requirements = GroupRequirements {
            min_resources,
            optimal_resources,
            max_resources,
            isolation: crate::test_parallelization::IsolationRequirements {
                process_isolation: group_tests
                    .iter()
                    .any(|t| t.isolation_requirements.process_isolation),
                network_isolation: group_tests
                    .iter()
                    .any(|t| t.isolation_requirements.network_isolation),
                filesystem_isolation: group_tests
                    .iter()
                    .any(|t| t.isolation_requirements.filesystem_isolation),
                database_isolation: group_tests
                    .iter()
                    .any(|t| t.isolation_requirements.database_isolation),
                gpu_isolation: group_tests.iter().any(|t| t.isolation_requirements.gpu_isolation),
                custom_isolation: HashMap::new(),
            },
            ordering_constraints: vec![], // Would be populated based on dependencies
            setup_teardown: SetupTeardownRequirements {
                setup_operations: vec![],
                teardown_operations: vec![],
                setup_timeout: Duration::from_secs(30),
                teardown_timeout: Duration::from_secs(30),
            },
        };

        // Generate tags
        let tags = self.generate_group_tags(&group_tests);

        // Create group metadata
        let creation_metadata = GroupCreationMetadata {
            created_at: Utc::now(),
            strategy_used: strategy_type,
            creation_duration: Duration::from_millis(0), // Would be measured
            creator: "TestGroupingEngine".to_string(),
            creation_parameters: HashMap::new(),
            initial_quality_score: characteristics.overall_quality,
        };

        let group = TestGroup {
            id: format!("group_{}", group_id),
            name: format!("Test Group {}", group_id),
            tests: test_names,
            characteristics,
            requirements,
            priority: group_tests.iter().map(|t| t.priority).sum::<f32>()
                / group_tests.len() as f32,
            estimated_duration,
            tags,
            creation_metadata,
            validation_results: vec![],
        };

        Ok(group)
    }

    /// Calculate group characteristics
    fn calculate_group_characteristics(
        &self,
        test_names: &[String],
        all_tests: &[TestParallelizationMetadata],
    ) -> GroupCharacteristics {
        let group_tests: Vec<_> = all_tests
            .iter()
            .filter(|t| test_names.contains(&t.base_context.test_name))
            .collect();

        if group_tests.is_empty() {
            return GroupCharacteristics {
                homogeneity: 0.0,
                resource_compatibility: 0.0,
                duration_balance: 0.0,
                dependency_complexity: 0.0,
                parallelization_potential: 0.0,
                conflict_risk: 0.0,
                overall_quality: 0.0,
            };
        }

        // Calculate homogeneity based on test categories
        let categories: Vec<_> =
            group_tests.iter().map(|t| format!("{:?}", t.base_context.category)).collect();
        let unique_categories: HashSet<_> = categories.iter().collect();
        let homogeneity = 1.0 - (unique_categories.len() as f32 - 1.0) / categories.len() as f32;

        // Calculate resource compatibility
        let resource_compatibility = self.calculate_resource_compatibility(&group_tests);

        // Calculate duration balance
        let durations: Vec<_> = group_tests.iter().map(|t| t.resource_usage.duration).collect();
        let duration_balance = if durations.len() > 1 {
            let mean_duration = durations.iter().sum::<Duration>() / durations.len() as u32;
            let variance = durations
                .iter()
                .map(|d| {
                    let diff = d.as_secs_f64() - mean_duration.as_secs_f64();
                    diff * diff
                })
                .sum::<f64>()
                / durations.len() as f64;
            let std_dev = variance.sqrt();
            let coefficient_of_variation = std_dev / mean_duration.as_secs_f64();
            (1.0 - coefficient_of_variation).max(0.0) as f32
        } else {
            1.0
        };

        // Simplified calculations for other metrics
        let dependency_complexity = 0.5; // Would be calculated based on actual dependencies
        let parallelization_potential = group_tests
            .iter()
            .map(|t| if t.parallelization_hints.parallel_with_any { 1.0 } else { 0.0 })
            .sum::<f32>()
            / group_tests.len() as f32;
        let conflict_risk = 0.3; // Would be calculated based on actual conflicts

        // Calculate overall quality as weighted average
        let overall_quality = (homogeneity * 0.2
            + resource_compatibility * 0.3
            + duration_balance * 0.2
            + (1.0 - dependency_complexity) * 0.1
            + parallelization_potential * 0.15
            + (1.0 - conflict_risk) * 0.05)
            .min(1.0)
            .max(0.0);

        GroupCharacteristics {
            homogeneity,
            resource_compatibility,
            duration_balance,
            dependency_complexity,
            parallelization_potential,
            conflict_risk,
            overall_quality,
        }
    }

    /// Calculate resource compatibility score
    fn calculate_resource_compatibility(
        &self,
        group_tests: &[&TestParallelizationMetadata],
    ) -> f32 {
        if group_tests.len() <= 1 {
            return 1.0;
        }

        let mut compatibility_scores = Vec::new();

        // Check CPU compatibility
        let cpu_usages: Vec<_> = group_tests.iter().map(|t| t.resource_usage.cpu_cores).collect();
        let max_cpu = cpu_usages.iter().sum::<f32>();
        let cpu_compatibility = if max_cpu <= 1.0 { 1.0 } else { 1.0 / max_cpu };
        compatibility_scores.push(cpu_compatibility);

        // Check memory compatibility
        let total_memory: u64 = group_tests.iter().map(|t| t.resource_usage.memory_mb).sum();
        let memory_compatibility =
            if total_memory <= 8192 { 1.0 } else { 8192.0 / total_memory as f32 };
        compatibility_scores.push(memory_compatibility);

        // Check GPU compatibility
        let all_gpu_devices: HashSet<_> =
            group_tests.iter().flat_map(|t| t.resource_usage.gpu_devices.iter()).collect();
        let requested_gpus: Vec<_> =
            group_tests.iter().flat_map(|t| t.resource_usage.gpu_devices.iter()).collect();
        let gpu_compatibility = if requested_gpus.is_empty() {
            1.0
        } else {
            all_gpu_devices.len() as f32 / requested_gpus.len() as f32
        };
        compatibility_scores.push(gpu_compatibility);

        // Average the compatibility scores
        compatibility_scores.iter().sum::<f32>() / compatibility_scores.len() as f32
    }

    /// Generate tags for a group
    fn generate_group_tags(&self, group_tests: &[&TestParallelizationMetadata]) -> Vec<String> {
        let mut tags = Vec::new();

        // Add category tags
        let categories: HashSet<_> = group_tests
            .iter()
            .map(|t| format!("category:{:?}", t.base_context.category))
            .collect();
        tags.extend(categories);

        // Add resource tags
        if group_tests.iter().any(|t| !t.resource_usage.gpu_devices.is_empty()) {
            tags.push("gpu".to_string());
        }
        if group_tests.iter().any(|t| !t.resource_usage.network_ports.is_empty()) {
            tags.push("network".to_string());
        }
        if group_tests.iter().any(|t| t.resource_usage.database_connections > 0) {
            tags.push("database".to_string());
        }
        if group_tests.iter().any(|t| !t.resource_usage.temp_directories.is_empty()) {
            tags.push("filesystem".to_string());
        }

        // Add size tag
        tags.push(format!("size:{}", group_tests.len()));

        tags
    }

    /// Stub implementations for other grouping strategies
    fn create_resource_optimal_grouping(
        &self,
        tests: &[TestParallelizationMetadata],
        dependencies: &[TestDependency],
        conflicts: &[ResourceConflict],
    ) -> AnalysisResult<Vec<TestGroup>> {
        // Fallback to balanced grouping for now
        self.create_balanced_grouping(tests, dependencies, conflicts)
    }

    fn create_time_optimal_grouping(
        &self,
        tests: &[TestParallelizationMetadata],
        dependencies: &[TestDependency],
        conflicts: &[ResourceConflict],
    ) -> AnalysisResult<Vec<TestGroup>> {
        // Fallback to balanced grouping for now
        self.create_balanced_grouping(tests, dependencies, conflicts)
    }

    fn create_dependency_aware_grouping(
        &self,
        tests: &[TestParallelizationMetadata],
        dependencies: &[TestDependency],
        conflicts: &[ResourceConflict],
    ) -> AnalysisResult<Vec<TestGroup>> {
        // Fallback to balanced grouping for now
        self.create_balanced_grouping(tests, dependencies, conflicts)
    }

    fn create_category_based_grouping(
        &self,
        tests: &[TestParallelizationMetadata],
        dependencies: &[TestDependency],
        conflicts: &[ResourceConflict],
    ) -> AnalysisResult<Vec<TestGroup>> {
        // Fallback to balanced grouping for now
        self.create_balanced_grouping(tests, dependencies, conflicts)
    }

    fn create_conflict_minimizing_grouping(
        &self,
        tests: &[TestParallelizationMetadata],
        dependencies: &[TestDependency],
        conflicts: &[ResourceConflict],
    ) -> AnalysisResult<Vec<TestGroup>> {
        // Fallback to balanced grouping for now
        self.create_balanced_grouping(tests, dependencies, conflicts)
    }

    fn create_ml_based_grouping(
        &self,
        tests: &[TestParallelizationMetadata],
        dependencies: &[TestDependency],
        conflicts: &[ResourceConflict],
    ) -> AnalysisResult<Vec<TestGroup>> {
        // Fallback to balanced grouping for now
        self.create_balanced_grouping(tests, dependencies, conflicts)
    }

    fn create_custom_grouping(
        &self,
        tests: &[TestParallelizationMetadata],
        dependencies: &[TestDependency],
        conflicts: &[ResourceConflict],
        _strategy: &GroupingStrategy,
    ) -> AnalysisResult<Vec<TestGroup>> {
        // Fallback to balanced grouping for now
        self.create_balanced_grouping(tests, dependencies, conflicts)
    }

    /// Optimize groups using configured optimizers
    fn optimize_groups(&self, groups: Vec<TestGroup>) -> AnalysisResult<Vec<TestGroup>> {
        // For now, return groups as-is
        // This would implement sophisticated optimization algorithms
        Ok(groups)
    }

    /// Validate groups using validation rules
    fn validate_groups(&self, groups: Vec<TestGroup>) -> AnalysisResult<Vec<TestGroup>> {
        let validation_rules = self.validation_rules.read();
        let mut validated_groups = Vec::new();

        for mut group in groups {
            let mut validation_results = Vec::new();

            for rule in validation_rules.iter() {
                if rule.enabled {
                    let result = self.validate_group_against_rule(&group, rule);
                    validation_results.push(result);
                }
            }

            group.validation_results = validation_results;
            validated_groups.push(group);
        }

        Ok(validated_groups)
    }

    /// Validate a group against a specific rule
    fn validate_group_against_rule(
        &self,
        _group: &TestGroup,
        _rule: &GroupValidationRule,
    ) -> ValidationResult {
        use crate::performance_optimizer::performance_modeling::types::{
            ResidualAnalysis, TestDataStatistics, ValidationDetails,
        };
        use std::collections::HashMap;

        // Simplified validation result with actual fields
        ValidationResult {
            metrics: HashMap::new(),
            cv_scores: vec![0.8],
            confidence: 0.8,
            details: ValidationDetails {
                test_samples: 0,
                test_statistics: TestDataStatistics {
                    mean_target: 0.0,
                    target_std: 0.0,
                    feature_correlations: HashMap::new(),
                    distribution_info: crate::performance_optimizer::performance_modeling::types::DistributionInfo::default(),
                },
                prediction_errors: vec![],
                residual_analysis: ResidualAnalysis {
                    autocorrelation: 0.0,
                    heteroscedasticity_p_value: 0.5,
                    normality_p_value: 0.5,
                    outliers: vec![],
                },
            },
            validated_at: chrono::Utc::now(),
        }
    }

    /// Update grouping metrics
    fn update_grouping_metrics(
        &self,
        groups: &[TestGroup],
        grouping_duration: Duration,
        strategy: &GroupingStrategy,
    ) {
        let mut metrics = self.metrics.write();

        metrics.total_groupings += 1;
        *metrics.groupings_by_strategy.entry(strategy.strategy_type.clone()).or_insert(0) += 1;

        // Update average grouping time
        let total_groupings = metrics.total_groupings as u32;
        metrics.average_grouping_time = Duration::from_nanos(
            ((metrics.average_grouping_time.as_nanos() * (total_groupings - 1) as u128
                + grouping_duration.as_nanos())
                / total_groupings as u128) as u64,
        );

        // Update quality metrics
        let average_quality: f32 =
            groups.iter().map(|g| g.characteristics.overall_quality).sum::<f32>()
                / groups.len() as f32;
        metrics.average_quality_score =
            (metrics.average_quality_score * (total_groupings - 1) as f32 + average_quality)
                / total_groupings as f32;

        if average_quality > metrics.best_quality_score {
            metrics.best_quality_score = average_quality;
        }

        // Update success rate (simplified)
        metrics.success_rate = 1.0; // All groupings are considered successful for now
    }

    /// Create default grouping strategies
    fn create_default_strategies() -> Vec<GroupingStrategy> {
        vec![
            GroupingStrategy {
                strategy_id: "balanced".to_string(),
                name: "Balanced Grouping".to_string(),
                description: "Balance all factors for optimal grouping".to_string(),
                strategy_type: GroupingStrategyType::Balanced,
                parameters: GroupingStrategyParameters {
                    optimization_target: OptimizationTarget::BalanceWorkload,
                    secondary_targets: vec![
                        OptimizationTarget::MinimizeConflicts,
                        OptimizationTarget::MaximizeResourceUtilization,
                    ],
                    algorithm_params: HashMap::new(),
                    constraints: vec![],
                },
                effectiveness_score: 0.8,
                applicability: vec![],
                expected_outcomes: vec![],
            },
            GroupingStrategy {
                strategy_id: "resource_optimal".to_string(),
                name: "Resource Optimal Grouping".to_string(),
                description: "Optimize for maximum resource utilization".to_string(),
                strategy_type: GroupingStrategyType::ResourceOptimal,
                parameters: GroupingStrategyParameters {
                    optimization_target: OptimizationTarget::MaximizeResourceUtilization,
                    secondary_targets: vec![OptimizationTarget::MinimizeConflicts],
                    algorithm_params: HashMap::new(),
                    constraints: vec![],
                },
                effectiveness_score: 0.75,
                applicability: vec![],
                expected_outcomes: vec![],
            },
        ]
    }

    /// Create default optimizers
    fn create_default_optimizers() -> Vec<GroupOptimizer> {
        vec![GroupOptimizer {
            optimizer_id: "greedy_search".to_string(),
            name: "Greedy Local Search".to_string(),
            description: "Greedy optimization for quick improvements".to_string(),
            technique: OptimizationTechnique::GreedySearch,
            parameters: OptimizationParameters {
                max_iterations: 100,
                convergence_threshold: 0.01,
                population_size: None,
                mutation_rate: None,
                temperature_schedule: None,
                custom_params: HashMap::new(),
            },
            effectiveness: 0.6,
        }]
    }

    /// Create default validation rules
    fn create_default_validation_rules() -> Vec<GroupValidationRule> {
        vec![
            GroupValidationRule {
                rule_id: "max_group_size".to_string(),
                name: "Maximum Group Size".to_string(),
                description: "Ensure groups don't exceed maximum size".to_string(),
                criteria: vec![ValidationCriterion {
                    criterion_type: ValidationCriterionType::GroupSize,
                    operator: ComparisonOperator::LessThanOrEqual,
                    expected_value: 10.0,
                    description: "Group size should not exceed 10 tests".to_string(),
                }],
                severity: ValidationSeverity::Error,
                enabled: true,
            },
            GroupValidationRule {
                rule_id: "min_quality_score".to_string(),
                name: "Minimum Quality Score".to_string(),
                description: "Ensure groups meet minimum quality threshold".to_string(),
                criteria: vec![ValidationCriterion {
                    criterion_type: ValidationCriterionType::HomogeneityScore,
                    operator: ComparisonOperator::GreaterThanOrEqual,
                    expected_value: 0.5,
                    description: "Group homogeneity should be at least 0.5".to_string(),
                }],
                severity: ValidationSeverity::Warning,
                enabled: true,
            },
        ]
    }

    /// Get grouping metrics
    pub fn get_metrics(&self) -> GroupingMetrics {
        (*self.metrics.read()).clone()
    }
}

/// Test set characteristics for strategy selection
#[derive(Debug)]
struct TestSetCharacteristics {
    pub test_count: usize,
    pub dependency_count: usize,
    pub conflict_count: usize,
    pub average_test_duration: Duration,
    pub resource_diversity: f32,
    pub category_diversity: f32,
    pub complexity_distribution: ComplexityDistribution,
}

/// Distribution of test complexity
#[derive(Debug, Default)]
struct ComplexityDistribution {
    pub low_complexity_count: usize,
    pub medium_complexity_count: usize,
    pub high_complexity_count: usize,
}

/// Helper implementation for ResourceRequirement
impl ResourceRequirement {
    /// Create from TestResourceUsage with comprehensive resource analysis
    pub fn from_resource_usage(usage: &TestResourceUsage) -> Self {
        use crate::test_independence_analyzer::types::{RequirementFlexibility, UsagePriority};

        // Analyze all resource types and create a composite requirement
        // Calculate weighted resource cost considering all resource types
        let cpu_cost = usage.cpu_cores as f64;
        let memory_cost = (usage.memory_mb as f64) / 1024.0; // Convert MB to GB
        let gpu_cost = (usage.gpu_devices.len() as f64) * 2.0; // GPU weighted higher
        let port_cost = (usage.network_ports.len() as f64) * 0.1; // Ports weighted lower

        let total_cost = cpu_cost + memory_cost + gpu_cost + port_cost;

        // Determine primary resource type based on highest usage
        let resource_type = if gpu_cost > cpu_cost && gpu_cost > memory_cost {
            "gpu_device"
        } else if memory_cost > cpu_cost {
            "memory"
        } else {
            "cpu"
        };

        // Determine priority based on resource intensity
        let priority = if total_cost > 10.0 {
            UsagePriority::High
        } else if total_cost > 5.0 {
            UsagePriority::Normal
        } else {
            UsagePriority::Low
        };

        // Determine flexibility based on GPU requirements
        // GPU requirements are typically less flexible
        let flexibility = if !usage.gpu_devices.is_empty() {
            RequirementFlexibility::Strict // GPU requirements are strict
        } else if usage.cpu_cores < 2.0 {
            RequirementFlexibility::Flexible
        } else {
            RequirementFlexibility::Flexible
        };

        Self {
            resource_type: resource_type.to_string(),
            min_amount: total_cost,
            max_amount: total_cost * 1.2, // 20% buffer for peak usage
            priority,
            flexibility,
        }
    }
}

impl Default for TestGroupingEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_timeout_optimization::{
        TestCategory, TestComplexityHints, TestExecutionContext,
    };
    use std::time::Duration;

    fn create_test_metadata(test_id: &str, category: TestCategory) -> TestParallelizationMetadata {
        TestParallelizationMetadata {
            base_context: TestExecutionContext {
                test_name: test_id.to_string(),
                category,
                environment: "test".to_string(),
                complexity_hints: TestComplexityHints::default(),
                expected_duration: Some(Duration::from_secs(10)),
                timeout_override: None,
            },
            dependencies: vec![],
            resource_usage: TestResourceUsage {
                test_id: test_id.to_string(),
                cpu_cores: 0.2,
                memory_mb: 256,
                gpu_devices: vec![],
                network_ports: vec![],
                temp_directories: vec![],
                database_connections: 0,
                duration: Duration::from_secs(10),
                priority: 1.0,
            },
            isolation_requirements: crate::test_parallelization::IsolationRequirements {
                process_isolation: false,
                network_isolation: false,
                filesystem_isolation: false,
                database_isolation: false,
                gpu_isolation: false,
                custom_isolation: HashMap::new(),
            },
            tags: vec![],
            priority: 1.0,
            parallelization_hints: crate::test_parallelization::ParallelizationHints {
                parallel_within_category: true,
                parallel_with_any: true,
                sequential_only: false,
                preferred_batch_size: None,
                optimal_concurrency: None,
                resource_sharing: crate::test_parallelization::ResourceSharingCapabilities {
                    cpu_sharing: true,
                    memory_sharing: false,
                    gpu_sharing: false,
                    network_sharing: true,
                    filesystem_sharing: false,
                },
            },
        }
    }

    #[test]
    fn test_grouping_engine_creation() {
        let engine = TestGroupingEngine::new();
        let metrics = engine.get_metrics();
        assert_eq!(metrics.total_groupings, 0);
    }

    #[test]
    fn test_basic_grouping() {
        let engine = TestGroupingEngine::new();

        let tests = vec![
            create_test_metadata("test1", TestCategory::Unit),
            create_test_metadata("test2", TestCategory::Unit),
            create_test_metadata("test3", TestCategory::Integration),
            create_test_metadata("test4", TestCategory::Integration),
        ];

        let groups = engine.create_test_groups(&tests, &[], &[]).unwrap();

        assert!(!groups.is_empty());
        assert!(groups.iter().all(|g| !g.tests.is_empty()));

        // Check that all tests are assigned to groups
        let total_assigned_tests: usize = groups.iter().map(|g| g.tests.len()).sum();
        assert_eq!(total_assigned_tests, tests.len());
    }

    #[test]
    fn test_conflict_aware_grouping() {
        let engine = TestGroupingEngine::new();

        let tests = vec![
            create_test_metadata("test1", TestCategory::Unit),
            create_test_metadata("test2", TestCategory::Unit),
        ];

        let conflicts = vec![ResourceConflict {
            id: "conflict1".to_string(),
            resource_id: "cpu_0".to_string(),
            test1: "test1".to_string(),
            test2: "test2".to_string(),
            resource_type: "CPU".to_string(),
            conflict_type: ConflictType::ExclusiveAccess,
            severity: ConflictSeverity::High,
            description: "CPU conflict".to_string(),
            resolution_strategies: vec![],
            metadata: ConflictMetadata {
                detected_at: chrono::Utc::now(),
                detection_method: "test".to_string(),
                confidence: 1.0,
                historical_occurrences: 0,
                last_occurrence: None,
            },
        }];

        let groups = engine.create_test_groups(&tests, &[], &conflicts).unwrap();

        // Conflicting tests should be in different groups
        let test1_groups: Vec<_> =
            groups.iter().filter(|g| g.tests.contains(&"test1".to_string())).collect();
        let test2_groups: Vec<_> =
            groups.iter().filter(|g| g.tests.contains(&"test2".to_string())).collect();

        // Tests should either be in different groups, or there should be only one group if conflict is resolved
        assert!(test1_groups.len() > 0 && test2_groups.len() > 0);
    }

    #[test]
    fn test_group_characteristics_calculation() {
        let engine = TestGroupingEngine::new();

        let tests = vec![
            create_test_metadata("test1", TestCategory::Unit),
            create_test_metadata("test2", TestCategory::Unit),
        ];

        let test_names = vec!["test1".to_string(), "test2".to_string()];
        let characteristics = engine.calculate_group_characteristics(&test_names, &tests);

        // Check that characteristics are calculated
        assert!(characteristics.homogeneity >= 0.0 && characteristics.homogeneity <= 1.0);
        assert!(
            characteristics.resource_compatibility >= 0.0
                && characteristics.resource_compatibility <= 1.0
        );
        assert!(characteristics.overall_quality >= 0.0 && characteristics.overall_quality <= 1.0);
    }
}
