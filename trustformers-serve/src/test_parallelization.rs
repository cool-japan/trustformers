//! Test Parallelization Framework for TrustformeRS
//!
//! This module provides comprehensive test parallelization capabilities including
//! test independence analysis, smart scheduling, resource management, and performance
//! optimization for concurrent test execution in the TrustformeRS ecosystem.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fmt, time::Duration};

use crate::test_timeout_optimization::{TestExecutionContext, TestExecutionResult};

/// Test execution status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestExecutionStatus {
    Success,
    Failed,
    Timeout,
    Cancelled,
}

/// Test parallelization framework configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestParallelizationConfig {
    /// Enable test parallelization
    pub enabled: bool,

    /// Maximum number of tests to run concurrently
    pub max_concurrent_tests: usize,

    /// Resource-specific concurrency limits
    pub resource_limits: ResourceLimits,

    /// Test independence analysis configuration
    pub independence_analysis: IndependenceAnalysisConfig,

    /// Smart scheduling configuration
    pub scheduling: SchedulingConfig,

    /// Performance optimization configuration
    pub performance_optimization: PerformanceOptimizationConfig,

    /// Resource management configuration
    pub resource_management: ResourceManagementConfig,

    /// Test suite organization configuration
    pub test_suite_organization: TestSuiteOrganizationConfig,

    /// Environment-specific overrides
    pub environment_overrides: HashMap<String, EnvironmentParallelizationConfig>,
}

/// Resource-specific concurrency limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum CPU-intensive tests running concurrently
    pub cpu_intensive_tests: usize,

    /// Maximum memory-intensive tests running concurrently
    pub memory_intensive_tests: usize,

    /// Maximum GPU tests running concurrently
    pub gpu_tests: usize,

    /// Maximum network tests running concurrently
    pub network_tests: usize,

    /// Maximum filesystem tests running concurrently
    pub filesystem_tests: usize,

    /// Maximum database tests running concurrently
    pub database_tests: usize,

    /// Total memory limit for all tests (MB)
    pub total_memory_limit_mb: Option<u64>,

    /// CPU usage limit for all tests (percentage)
    pub total_cpu_limit_percent: Option<f32>,
}

/// Test independence analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndependenceAnalysisConfig {
    /// Enable automatic dependency detection
    pub auto_dependency_detection: bool,

    /// Enable resource conflict detection
    pub resource_conflict_detection: bool,

    /// Enable test ordering optimization
    pub test_ordering_optimization: bool,

    /// Dependency analysis depth
    pub dependency_analysis_depth: usize,

    /// Cache dependency analysis results
    pub cache_analysis_results: bool,

    /// Analysis cache TTL
    pub analysis_cache_ttl: Duration,
}

/// Smart scheduling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulingConfig {
    /// Scheduling strategy
    pub strategy: SchedulingStrategy,

    /// Load balancing configuration
    pub load_balancing: LoadBalancingConfig,

    /// Priority-based scheduling
    pub priority_scheduling: PrioritySchedulingConfig,

    /// Adaptive scheduling based on historical performance
    pub adaptive_scheduling: bool,

    /// Test failure handling strategy
    pub failure_handling: FailureHandlingStrategy,

    /// Early termination strategy
    pub early_termination: EarlyTerminationStrategy,
}

/// Scheduling strategy enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SchedulingStrategy {
    /// First In, First Out
    Fifo,
    /// Shortest Job First
    ShortestJobFirst,
    /// Priority-based scheduling
    Priority,
    /// Resource-aware scheduling
    ResourceAware,
    /// Adaptive scheduling based on historical data
    Adaptive,
    /// Custom scheduling algorithm
    Custom(String),
}

impl Default for SchedulingStrategy {
    fn default() -> Self {
        SchedulingStrategy::ResourceAware
    }
}

/// Load balancing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingConfig {
    /// Enable dynamic load balancing
    pub enabled: bool,

    /// Load balancing strategy
    pub strategy: LoadBalancingStrategy,

    /// Rebalancing interval
    pub rebalancing_interval: Duration,

    /// Load threshold for rebalancing
    pub load_threshold: f32,

    /// Enable work stealing
    pub work_stealing: bool,
}

/// Load balancing strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin assignment
    RoundRobin,
    /// Least loaded worker first
    LeastLoaded,
    /// Resource-based balancing
    ResourceBased,
    /// Performance-based balancing
    PerformanceBased,
}

/// Priority scheduling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrioritySchedulingConfig {
    /// Enable priority-based scheduling
    pub enabled: bool,

    /// Priority calculation factors
    pub priority_factors: PriorityFactors,

    /// Dynamic priority adjustment
    pub dynamic_adjustment: bool,

    /// Priority boost for failed tests
    pub failure_priority_boost: f32,
}

/// Priority calculation factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorityFactors {
    /// Test category weight
    pub category_weight: f32,

    /// Expected duration weight
    pub duration_weight: f32,

    /// Resource usage weight
    pub resource_weight: f32,

    /// Historical success rate weight
    pub success_rate_weight: f32,

    /// Dependency chain length weight
    pub dependency_weight: f32,
}

/// Failure handling strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FailureHandlingStrategy {
    /// Continue running all tests regardless of failures
    ContinueAll,
    /// Stop dependent tests when a dependency fails
    StopDependent,
    /// Stop entire test suite on first failure
    FailFast,
    /// Stop test group on failure within group
    FailGroup,
}

/// Early termination strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EarlyTerminationStrategy {
    /// No early termination
    None,
    /// Terminate on error rate threshold
    ErrorRateThreshold(f32),
    /// Terminate on time budget exceeded
    TimeBudget(Duration),
    /// Terminate on resource exhaustion
    ResourceExhaustion,
    /// Custom termination condition
    Custom(String),
}

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceOptimizationConfig {
    /// Enable adaptive parallelism
    pub adaptive_parallelism: bool,

    /// CPU core scaling configuration
    pub cpu_scaling: CpuScalingConfig,

    /// Memory optimization configuration
    pub memory_optimization: MemoryOptimizationConfig,

    /// Warmup optimization
    pub warmup_optimization: WarmupOptimizationConfig,

    /// Test batching optimization
    pub test_batching: TestBatchingConfig,

    /// Performance monitoring
    pub performance_monitoring: ParallelPerformanceMonitoringConfig,
}

/// CPU scaling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuScalingConfig {
    /// Enable automatic CPU scaling
    pub enabled: bool,

    /// Minimum CPU utilization target
    pub min_cpu_utilization: f32,

    /// Maximum CPU utilization target
    pub max_cpu_utilization: f32,

    /// CPU scaling factor
    pub scaling_factor: f32,

    /// Scaling adjustment interval
    pub adjustment_interval: Duration,
}

/// Memory optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryOptimizationConfig {
    /// Enable memory-aware scheduling
    pub memory_aware_scheduling: bool,

    /// Memory usage threshold for warnings
    pub memory_warning_threshold: f32,

    /// Memory usage threshold for throttling
    pub memory_throttling_threshold: f32,

    /// Enable garbage collection hints
    pub gc_hints: bool,

    /// Memory cleanup between tests
    pub cleanup_between_tests: bool,
}

/// Warmup optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupOptimizationConfig {
    /// Enable test warmup
    pub enabled: bool,

    /// Number of warmup iterations
    pub warmup_iterations: usize,

    /// Warmup timeout
    pub warmup_timeout: Duration,

    /// Cache warmup results
    pub cache_warmup: bool,

    /// Parallel warmup execution
    pub parallel_warmup: bool,
}

/// Test batching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestBatchingConfig {
    /// Enable test batching
    pub enabled: bool,

    /// Optimal batch size
    pub optimal_batch_size: usize,

    /// Maximum batch size
    pub max_batch_size: usize,

    /// Batching strategy
    pub batching_strategy: BatchingStrategy,

    /// Batch timeout
    pub batch_timeout: Duration,
}

/// Test batching strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BatchingStrategy {
    /// Group by test category
    ByCategory,
    /// Group by resource usage
    ByResource,
    /// Group by expected duration
    ByDuration,
    /// Group by dependency chain
    ByDependency,
    /// Custom batching logic
    Custom(String),
}

/// Parallel performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelPerformanceMonitoringConfig {
    /// Enable performance monitoring
    pub enabled: bool,

    /// Metrics collection interval
    pub collection_interval: Duration,

    /// Performance regression detection
    pub regression_detection: bool,

    /// Bottleneck detection
    pub bottleneck_detection: bool,

    /// Resource utilization tracking
    pub resource_utilization_tracking: bool,

    /// Parallel efficiency metrics
    pub parallel_efficiency_metrics: bool,
}

/// Resource management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceManagementConfig {
    /// Resource pool configuration
    pub resource_pools: ResourcePoolConfig,

    /// Resource conflict resolution
    pub conflict_resolution: ConflictResolutionConfig,

    /// Resource monitoring
    pub resource_monitoring: ResourceMonitoringConfig,

    /// Resource cleanup
    pub resource_cleanup: ResourceCleanupConfig,
}

/// Resource pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePoolConfig {
    /// Enable resource pooling
    pub enabled: bool,

    /// Network port pool
    pub network_port_pool: PortPoolConfig,

    /// Temporary directory pool
    pub temp_directory_pool: TempDirPoolConfig,

    /// GPU device pool
    pub gpu_device_pool: GpuPoolConfig,

    /// Database connection pool
    pub database_pool: DatabasePoolConfig,
}

/// Network port pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortPoolConfig {
    /// Port range start
    pub start_port: u16,

    /// Port range end
    pub end_port: u16,

    /// Reserved ports to avoid
    pub reserved_ports: Vec<u16>,

    /// Port allocation timeout
    pub allocation_timeout: Duration,
}

/// Temporary directory pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TempDirPoolConfig {
    /// Base temporary directory
    pub base_dir: String,

    /// Maximum directories in pool
    pub max_directories: usize,

    /// Directory cleanup strategy
    pub cleanup_strategy: TempDirCleanupStrategy,

    /// Directory size limit (MB)
    pub size_limit_mb: Option<u64>,
}

/// Temporary directory cleanup strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TempDirCleanupStrategy {
    /// Clean up immediately after test
    Immediate,
    /// Clean up at end of test suite
    AtEnd,
    /// Keep directories for debugging
    Keep,
    /// Custom cleanup logic
    Custom(String),
}

/// GPU device pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuPoolConfig {
    /// Available GPU device IDs
    pub device_ids: Vec<usize>,

    /// GPU memory limit per test (MB)
    pub memory_limit_mb: Option<u64>,

    /// GPU utilization limit per test
    pub utilization_limit: Option<f32>,

    /// GPU sharing strategy
    pub sharing_strategy: GpuSharingStrategy,
}

/// GPU sharing strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GpuSharingStrategy {
    /// Exclusive access per test
    Exclusive,
    /// Shared access with memory limits
    SharedMemory,
    /// Shared access with time slicing
    TimeSliced,
    /// No GPU sharing restrictions
    Unrestricted,
}

/// Database connection pool configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabasePoolConfig {
    /// Maximum database connections
    pub max_connections: usize,

    /// Connection timeout
    pub connection_timeout: Duration,

    /// Database isolation strategy
    pub isolation_strategy: DatabaseIsolationStrategy,

    /// Test database cleanup
    pub cleanup_strategy: DatabaseCleanupStrategy,
}

/// Database isolation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatabaseIsolationStrategy {
    /// Separate database per test
    PerTest,
    /// Separate schema per test
    PerSchema,
    /// Shared database with transactions
    SharedTransaction,
    /// No isolation
    None,
}

/// Database cleanup strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatabaseCleanupStrategy {
    /// Truncate tables after each test
    Truncate,
    /// Drop and recreate schema
    DropRecreate,
    /// Rollback transactions
    Rollback,
    /// No cleanup
    None,
}

/// Resource conflict resolution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictResolutionConfig {
    /// Conflict detection strategy
    pub detection_strategy: ConflictDetectionStrategy,

    /// Resolution strategy
    pub resolution_strategy: ConflictResolutionStrategy,

    /// Conflict timeout
    pub conflict_timeout: Duration,

    /// Maximum resolution attempts
    pub max_resolution_attempts: usize,
}

/// Conflict detection strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictDetectionStrategy {
    /// Static analysis of test metadata
    Static,
    /// Runtime monitoring for conflicts
    Runtime,
    /// Hybrid static and runtime detection
    Hybrid,
}

/// Conflict resolution strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolutionStrategy {
    /// Queue conflicting tests
    Queue,
    /// Retry with backoff
    RetryWithBackoff,
    /// Skip conflicting tests
    Skip,
    /// Allocate alternative resources
    AllocateAlternative,
}

/// Resource monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMonitoringConfig {
    /// Enable resource monitoring
    pub enabled: bool,

    /// Monitoring interval
    pub monitoring_interval: Duration,

    /// Resource usage thresholds
    pub usage_thresholds: ResourceUsageThresholds,

    /// Alert configuration
    pub alerts: ResourceAlertConfig,
}

/// Resource usage thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsageThresholds {
    /// CPU usage warning threshold
    pub cpu_warning: f32,

    /// CPU usage critical threshold
    pub cpu_critical: f32,

    /// Memory usage warning threshold
    pub memory_warning: f32,

    /// Memory usage critical threshold
    pub memory_critical: f32,

    /// GPU usage warning threshold
    pub gpu_warning: Option<f32>,

    /// GPU usage critical threshold
    pub gpu_critical: Option<f32>,
}

/// Resource alert configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAlertConfig {
    /// Enable alerts
    pub enabled: bool,

    /// Alert cooldown period
    pub cooldown_period: Duration,

    /// Alert escalation levels
    pub escalation_levels: Vec<AlertLevel>,
}

/// Alert level configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertLevel {
    /// Alert level name
    pub level: String,

    /// Threshold for this level
    pub threshold: f32,

    /// Action to take
    pub action: AlertAction,
}

/// Alert action enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertAction {
    /// Log warning message
    Log,
    /// Throttle test execution
    Throttle,
    /// Pause test execution
    Pause,
    /// Terminate test execution
    Terminate,
    /// Custom action
    Custom(String),
}

/// Resource cleanup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceCleanupConfig {
    /// Enable automatic cleanup
    pub enabled: bool,

    /// Cleanup interval
    pub cleanup_interval: Duration,

    /// Cleanup strategies for different resources
    pub cleanup_strategies: HashMap<String, CleanupStrategy>,

    /// Force cleanup on shutdown
    pub force_cleanup_on_shutdown: bool,
}

/// Cleanup strategy enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CleanupStrategy {
    /// Immediate cleanup
    Immediate,
    /// Deferred cleanup
    Deferred(Duration),
    /// Lazy cleanup
    Lazy,
    /// No cleanup
    None,
    /// Custom cleanup logic
    Custom(String),
}

/// Test suite organization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuiteOrganizationConfig {
    /// Test suite definition
    pub suite_definition: SuiteDefinitionConfig,

    /// Test grouping configuration
    pub test_grouping: TestGroupingConfig,

    /// Suite execution order
    pub execution_order: SuiteExecutionOrder,

    /// Suite dependencies
    pub suite_dependencies: bool,
}

/// Suite definition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteDefinitionConfig {
    /// Auto-discovery of test suites
    pub auto_discovery: bool,

    /// Manual suite definitions
    pub manual_suites: HashMap<String, TestSuiteDefinition>,

    /// Suite naming convention
    pub naming_convention: SuiteNamingConvention,

    /// Suite metadata requirements
    pub metadata_requirements: Vec<String>,
}

/// Test suite definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSuiteDefinition {
    /// Suite name
    pub name: String,

    /// Suite description
    pub description: String,

    /// Test patterns to include
    pub include_patterns: Vec<String>,

    /// Test patterns to exclude
    pub exclude_patterns: Vec<String>,

    /// Suite tags
    pub tags: Vec<String>,

    /// Suite priority
    pub priority: i32,

    /// Suite timeout
    pub timeout: Option<Duration>,

    /// Suite resource requirements
    pub resource_requirements: SuiteResourceRequirements,
}

/// Suite resource requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteResourceRequirements {
    /// CPU cores required
    pub cpu_cores: Option<usize>,

    /// Memory required (MB)
    pub memory_mb: Option<u64>,

    /// GPU devices required
    pub gpu_devices: Option<usize>,

    /// Network ports required
    pub network_ports: Option<usize>,

    /// Temporary directories required
    pub temp_directories: Option<usize>,

    /// Database connections required
    pub database_connections: Option<usize>,
}

/// Suite naming convention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuiteNamingConvention {
    /// Use directory structure
    DirectoryBased,
    /// Use test name prefixes
    PrefixBased,
    /// Use test tags
    TagBased,
    /// Custom naming logic
    Custom(String),
}

/// Test grouping configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestGroupingConfig {
    /// Grouping strategy
    pub strategy: TestGroupingStrategy,

    /// Maximum tests per group
    pub max_tests_per_group: usize,

    /// Group formation criteria
    pub formation_criteria: GroupFormationCriteria,

    /// Dynamic regrouping
    pub dynamic_regrouping: bool,
}

/// Test grouping strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestGroupingStrategy {
    /// Group by test category
    ByCategory,
    /// Group by resource usage
    ByResource,
    /// Group by expected duration
    ByDuration,
    /// Group by dependencies
    ByDependency,
    /// Group by priority
    ByPriority,
    /// No grouping
    None,
    /// Custom grouping logic
    Custom(String),
}

/// Group formation criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupFormationCriteria {
    /// Similarity threshold for grouping
    pub similarity_threshold: f32,

    /// Resource compatibility requirement
    pub resource_compatibility: bool,

    /// Duration balance requirement
    pub duration_balance: bool,

    /// Priority clustering
    pub priority_clustering: bool,
}

/// Suite execution order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuiteExecutionOrder {
    /// Execute in definition order
    Definition,
    /// Execute by priority
    Priority,
    /// Execute by estimated duration (shortest first)
    Duration,
    /// Execute by dependency order
    Dependency,
    /// Execute in random order
    Random,
    /// Custom execution order
    Custom(String),
}

/// Environment-specific parallelization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentParallelizationConfig {
    /// Concurrency multiplier for this environment
    pub concurrency_multiplier: f32,

    /// Resource limit overrides
    pub resource_limit_overrides: ResourceLimits,

    /// Disabled features for this environment
    pub disabled_features: Vec<String>,

    /// Environment-specific scheduling strategy
    pub scheduling_strategy_override: Option<SchedulingStrategy>,
}

impl Default for TestParallelizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_concurrent_tests: num_cpus::get().max(4),
            resource_limits: ResourceLimits::default(),
            independence_analysis: IndependenceAnalysisConfig::default(),
            scheduling: SchedulingConfig::default(),
            performance_optimization: PerformanceOptimizationConfig::default(),
            resource_management: ResourceManagementConfig::default(),
            test_suite_organization: TestSuiteOrganizationConfig::default(),
            environment_overrides: HashMap::new(),
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        let cpu_count = num_cpus::get();
        Self {
            cpu_intensive_tests: cpu_count / 2,
            memory_intensive_tests: 2,
            gpu_tests: 1,
            network_tests: 4,
            filesystem_tests: 8,
            database_tests: 2,
            total_memory_limit_mb: Some(8192), // 8GB
            total_cpu_limit_percent: Some(80.0),
        }
    }
}

impl Default for IndependenceAnalysisConfig {
    fn default() -> Self {
        Self {
            auto_dependency_detection: true,
            resource_conflict_detection: true,
            test_ordering_optimization: true,
            dependency_analysis_depth: 5,
            cache_analysis_results: true,
            analysis_cache_ttl: Duration::from_secs(3600), // 1 hour
        }
    }
}

impl Default for SchedulingConfig {
    fn default() -> Self {
        Self {
            strategy: SchedulingStrategy::ResourceAware,
            load_balancing: LoadBalancingConfig::default(),
            priority_scheduling: PrioritySchedulingConfig::default(),
            adaptive_scheduling: true,
            failure_handling: FailureHandlingStrategy::StopDependent,
            early_termination: EarlyTerminationStrategy::ErrorRateThreshold(0.2), // 20% error rate
        }
    }
}

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strategy: LoadBalancingStrategy::ResourceBased,
            rebalancing_interval: Duration::from_secs(10),
            load_threshold: 0.8,
            work_stealing: true,
        }
    }
}

impl Default for PrioritySchedulingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            priority_factors: PriorityFactors::default(),
            dynamic_adjustment: true,
            failure_priority_boost: 1.5,
        }
    }
}

impl Default for PriorityFactors {
    fn default() -> Self {
        Self {
            category_weight: 0.3,
            duration_weight: 0.2,
            resource_weight: 0.25,
            success_rate_weight: 0.15,
            dependency_weight: 0.1,
        }
    }
}

impl Default for PerformanceOptimizationConfig {
    fn default() -> Self {
        Self {
            adaptive_parallelism: true,
            cpu_scaling: CpuScalingConfig::default(),
            memory_optimization: MemoryOptimizationConfig::default(),
            warmup_optimization: WarmupOptimizationConfig::default(),
            test_batching: TestBatchingConfig::default(),
            performance_monitoring: ParallelPerformanceMonitoringConfig::default(),
        }
    }
}

impl Default for CpuScalingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_cpu_utilization: 0.4,
            max_cpu_utilization: 0.8,
            scaling_factor: 1.2,
            adjustment_interval: Duration::from_secs(30),
        }
    }
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            memory_aware_scheduling: true,
            memory_warning_threshold: 0.7,
            memory_throttling_threshold: 0.85,
            gc_hints: true,
            cleanup_between_tests: true,
        }
    }
}

impl Default for WarmupOptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            warmup_iterations: 3,
            warmup_timeout: Duration::from_secs(30),
            cache_warmup: true,
            parallel_warmup: true,
        }
    }
}

impl Default for TestBatchingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            optimal_batch_size: 4,
            max_batch_size: 8,
            batching_strategy: BatchingStrategy::ByResource,
            batch_timeout: Duration::from_secs(300),
        }
    }
}

impl Default for ParallelPerformanceMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval: Duration::from_millis(500),
            regression_detection: true,
            bottleneck_detection: true,
            resource_utilization_tracking: true,
            parallel_efficiency_metrics: true,
        }
    }
}

impl Default for ResourceManagementConfig {
    fn default() -> Self {
        Self {
            resource_pools: ResourcePoolConfig::default(),
            conflict_resolution: ConflictResolutionConfig::default(),
            resource_monitoring: ResourceMonitoringConfig::default(),
            resource_cleanup: ResourceCleanupConfig::default(),
        }
    }
}

impl Default for ResourcePoolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            network_port_pool: PortPoolConfig::default(),
            temp_directory_pool: TempDirPoolConfig::default(),
            gpu_device_pool: GpuPoolConfig::default(),
            database_pool: DatabasePoolConfig::default(),
        }
    }
}

impl Default for PortPoolConfig {
    fn default() -> Self {
        Self {
            start_port: 10000,
            end_port: 20000,
            reserved_ports: vec![],
            allocation_timeout: Duration::from_secs(5),
        }
    }
}

impl Default for TempDirPoolConfig {
    fn default() -> Self {
        Self {
            base_dir: "/tmp/trustformers_tests".to_string(),
            max_directories: 100,
            cleanup_strategy: TempDirCleanupStrategy::AtEnd,
            size_limit_mb: Some(1024), // 1GB per directory
        }
    }
}

impl Default for GpuPoolConfig {
    fn default() -> Self {
        Self {
            device_ids: vec![0],         // Default to first GPU
            memory_limit_mb: Some(2048), // 2GB per test
            utilization_limit: Some(0.8),
            sharing_strategy: GpuSharingStrategy::SharedMemory,
        }
    }
}

impl Default for DatabasePoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 10,
            connection_timeout: Duration::from_secs(30),
            isolation_strategy: DatabaseIsolationStrategy::PerSchema,
            cleanup_strategy: DatabaseCleanupStrategy::Truncate,
        }
    }
}

impl Default for ConflictResolutionConfig {
    fn default() -> Self {
        Self {
            detection_strategy: ConflictDetectionStrategy::Hybrid,
            resolution_strategy: ConflictResolutionStrategy::Queue,
            conflict_timeout: Duration::from_secs(60),
            max_resolution_attempts: 3,
        }
    }
}

impl Default for ResourceMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            monitoring_interval: Duration::from_secs(1),
            usage_thresholds: ResourceUsageThresholds::default(),
            alerts: ResourceAlertConfig::default(),
        }
    }
}

impl Default for ResourceUsageThresholds {
    fn default() -> Self {
        Self {
            cpu_warning: 0.7,
            cpu_critical: 0.9,
            memory_warning: 0.8,
            memory_critical: 0.95,
            gpu_warning: Some(0.8),
            gpu_critical: Some(0.95),
        }
    }
}

impl Default for ResourceAlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cooldown_period: Duration::from_secs(60),
            escalation_levels: vec![
                AlertLevel {
                    level: "warning".to_string(),
                    threshold: 0.7,
                    action: AlertAction::Log,
                },
                AlertLevel {
                    level: "critical".to_string(),
                    threshold: 0.9,
                    action: AlertAction::Throttle,
                },
            ],
        }
    }
}

impl Default for ResourceCleanupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cleanup_interval: Duration::from_secs(300), // 5 minutes
            cleanup_strategies: HashMap::from([
                ("network_ports".to_string(), CleanupStrategy::Immediate),
                (
                    "temp_directories".to_string(),
                    CleanupStrategy::Deferred(Duration::from_secs(600)),
                ),
                ("gpu_memory".to_string(), CleanupStrategy::Immediate),
                (
                    "database_connections".to_string(),
                    CleanupStrategy::Immediate,
                ),
            ]),
            force_cleanup_on_shutdown: true,
        }
    }
}

impl Default for TestSuiteOrganizationConfig {
    fn default() -> Self {
        Self {
            suite_definition: SuiteDefinitionConfig::default(),
            test_grouping: TestGroupingConfig::default(),
            execution_order: SuiteExecutionOrder::Priority,
            suite_dependencies: true,
        }
    }
}

impl Default for SuiteDefinitionConfig {
    fn default() -> Self {
        Self {
            auto_discovery: true,
            manual_suites: HashMap::new(),
            naming_convention: SuiteNamingConvention::DirectoryBased,
            metadata_requirements: vec![],
        }
    }
}

impl Default for TestGroupingConfig {
    fn default() -> Self {
        Self {
            strategy: TestGroupingStrategy::ByResource,
            max_tests_per_group: 10,
            formation_criteria: GroupFormationCriteria::default(),
            dynamic_regrouping: true,
        }
    }
}

impl Default for GroupFormationCriteria {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.7,
            resource_compatibility: true,
            duration_balance: true,
            priority_clustering: true,
        }
    }
}

/// Test dependency information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TestDependency {
    /// Test that depends on another
    pub dependent_test: String,

    /// Test that is depended upon
    pub dependency_test: String,

    /// Type of dependency
    pub dependency_type: DependencyType,

    /// Strength of dependency (0.0 to 1.0)
    pub strength: f32,

    /// Reason for dependency
    pub reason: String,
}

/// Types of test dependencies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyType {
    /// Hard dependency - dependent test cannot run without dependency
    Hard,
    /// Soft dependency - dependent test can run but may behave differently
    Soft,
    /// Resource conflict - tests cannot run simultaneously
    Conflict,
    /// Ordering dependency - tests should run in specific order
    Ordering,
    /// Setup dependency - dependency sets up environment for dependent test
    Setup,
}

/// Resource usage information for a test
#[derive(Debug, Clone)]
pub struct TestResourceUsage {
    /// Test identifier
    pub test_id: String,

    /// CPU usage (cores)
    pub cpu_cores: f32,

    /// Memory usage (MB)
    pub memory_mb: u64,

    /// GPU device IDs used
    pub gpu_devices: Vec<usize>,

    /// Network ports used
    pub network_ports: Vec<u16>,

    /// Temporary directories used
    pub temp_directories: Vec<String>,

    /// Database connections used
    pub database_connections: usize,

    /// Resource usage duration
    pub duration: Duration,

    /// Resource usage priority
    pub priority: f32,
}

/// Test execution metadata for parallelization
#[derive(Debug, Clone)]
pub struct TestParallelizationMetadata {
    /// Base test execution context
    pub base_context: TestExecutionContext,

    /// Test dependencies
    pub dependencies: Vec<TestDependency>,

    /// Resource usage information
    pub resource_usage: TestResourceUsage,

    /// Test isolation requirements
    pub isolation_requirements: IsolationRequirements,

    /// Test tags for grouping and filtering
    pub tags: Vec<String>,

    /// Test priority for scheduling
    pub priority: f32,

    /// Test parallelization hints
    pub parallelization_hints: ParallelizationHints,
}

/// Test isolation requirements
#[derive(Debug, Clone)]
pub struct IsolationRequirements {
    /// Requires process isolation
    pub process_isolation: bool,

    /// Requires network isolation
    pub network_isolation: bool,

    /// Requires filesystem isolation
    pub filesystem_isolation: bool,

    /// Requires database isolation
    pub database_isolation: bool,

    /// Requires GPU isolation
    pub gpu_isolation: bool,

    /// Custom isolation requirements
    pub custom_isolation: HashMap<String, String>,
}

/// Test parallelization hints
#[derive(Debug, Clone)]
pub struct ParallelizationHints {
    /// Can run in parallel with tests of same category
    pub parallel_within_category: bool,

    /// Can run in parallel with any test
    pub parallel_with_any: bool,

    /// Cannot run in parallel (must be sequential)
    pub sequential_only: bool,

    /// Preferred execution batch size
    pub preferred_batch_size: Option<usize>,

    /// Optimal concurrency level for this test
    pub optimal_concurrency: Option<usize>,

    /// Resource sharing capabilities
    pub resource_sharing: ResourceSharingCapabilities,
}

/// Resource sharing capabilities
#[derive(Debug, Clone)]
pub struct ResourceSharingCapabilities {
    /// Can share CPU with other tests
    pub cpu_sharing: bool,

    /// Can share memory with other tests
    pub memory_sharing: bool,

    /// Can share GPU with other tests
    pub gpu_sharing: bool,

    /// Can share network resources with other tests
    pub network_sharing: bool,

    /// Can share filesystem with other tests
    pub filesystem_sharing: bool,
}

impl Default for IsolationRequirements {
    fn default() -> Self {
        Self {
            process_isolation: false,
            network_isolation: false,
            filesystem_isolation: false,
            database_isolation: false,
            gpu_isolation: false,
            custom_isolation: HashMap::new(),
        }
    }
}

impl Default for ParallelizationHints {
    fn default() -> Self {
        Self {
            parallel_within_category: true,
            parallel_with_any: false,
            sequential_only: false,
            preferred_batch_size: None,
            optimal_concurrency: None,
            resource_sharing: ResourceSharingCapabilities::default(),
        }
    }
}

impl Default for ResourceSharingCapabilities {
    fn default() -> Self {
        Self {
            cpu_sharing: true,
            memory_sharing: false,
            gpu_sharing: false,
            network_sharing: true,
            filesystem_sharing: true,
        }
    }
}

/// Test parallelization execution result
#[derive(Debug, Clone)]
pub struct TestParallelizationResult {
    /// Base test execution result
    pub base_result: TestExecutionResult,

    /// Parallelization metrics
    pub parallelization_metrics: ParallelizationMetrics,

    /// Resource utilization during execution
    pub resource_utilization: ResourceUtilizationMetrics,

    /// Scheduling information
    pub scheduling_info: SchedulingInfo,

    /// Performance analysis
    pub performance_analysis: PerformanceAnalysis,
}

/// Parallelization-specific metrics
#[derive(Debug, Clone)]
pub struct ParallelizationMetrics {
    /// Number of tests running concurrently
    pub concurrent_tests: usize,

    /// Parallel efficiency (0.0 to 1.0)
    pub parallel_efficiency: f32,

    /// Resource contention detected
    pub resource_contention: bool,

    /// Load balancing effectiveness
    pub load_balancing_effectiveness: f32,

    /// Scheduling overhead
    pub scheduling_overhead: Duration,

    /// Total parallelization overhead
    pub total_overhead: Duration,

    /// Speedup achieved vs sequential execution
    pub speedup_factor: f32,

    /// Scalability metrics
    pub scalability_metrics: ScalabilityMetrics,
}

/// Scalability metrics
#[derive(Debug, Clone)]
pub struct ScalabilityMetrics {
    /// Optimal concurrency level found
    pub optimal_concurrency: usize,

    /// Scalability efficiency curve
    pub efficiency_curve: Vec<(usize, f32)>,

    /// Bottleneck resources identified
    pub bottleneck_resources: Vec<String>,

    /// Scalability score (0.0 to 1.0)
    pub scalability_score: f32,
}

/// Resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceUtilizationMetrics {
    /// CPU utilization statistics
    pub cpu_utilization: UtilizationStats,

    /// Memory utilization statistics
    pub memory_utilization: UtilizationStats,

    /// GPU utilization statistics
    pub gpu_utilization: Option<UtilizationStats>,

    /// Network utilization statistics
    pub network_utilization: UtilizationStats,

    /// Filesystem utilization statistics
    pub filesystem_utilization: UtilizationStats,

    /// Overall resource efficiency
    pub overall_efficiency: f32,
}

/// Resource utilization statistics
#[derive(Debug, Clone)]
pub struct UtilizationStats {
    /// Average utilization percentage
    pub average: f32,

    /// Peak utilization percentage
    pub peak: f32,

    /// Minimum utilization percentage
    pub minimum: f32,

    /// Standard deviation of utilization
    pub std_deviation: f32,

    /// Utilization timeline
    pub timeline: Vec<(Duration, f32)>,

    /// Efficiency score
    pub efficiency_score: f32,
}

/// Test scheduling information
#[derive(Debug, Clone)]
pub struct SchedulingInfo {
    /// Time test was scheduled
    pub scheduled_at: DateTime<Utc>,

    /// Time test started execution
    pub started_at: DateTime<Utc>,

    /// Time test completed
    pub completed_at: DateTime<Utc>,

    /// Scheduling strategy used
    pub strategy_used: SchedulingStrategy,

    /// Queue position when scheduled
    pub queue_position: usize,

    /// Wait time in queue
    pub queue_wait_time: Duration,

    /// Scheduling decisions made
    pub scheduling_decisions: Vec<SchedulingDecision>,

    /// Resource allocations
    pub resource_allocations: Vec<ResourceAllocation>,
}

/// Scheduling decision information
#[derive(Debug, Clone)]
pub struct SchedulingDecision {
    /// Decision timestamp
    pub timestamp: DateTime<Utc>,

    /// Decision type
    pub decision_type: SchedulingDecisionType,

    /// Reason for decision
    pub reason: String,

    /// Alternative options considered
    pub alternatives_considered: Vec<String>,

    /// Decision confidence score
    pub confidence: f32,
}

/// Types of scheduling decisions
#[derive(Debug, Clone)]
pub enum SchedulingDecisionType {
    /// Test was scheduled for immediate execution
    ScheduleImmediate,

    /// Test was queued for later execution
    Queue,

    /// Test was delayed due to resource constraints
    Delay,

    /// Test was batched with other tests
    Batch,

    /// Test was rescheduled
    Reschedule,

    /// Test was cancelled
    Cancel,
}

/// Resource allocation information
#[derive(Debug, Clone)]
pub struct ResourceAllocation {
    /// Resource type
    pub resource_type: String,

    /// Resource identifier
    pub resource_id: String,

    /// Allocation timestamp
    pub allocated_at: DateTime<Utc>,

    /// Deallocation timestamp
    pub deallocated_at: Option<DateTime<Utc>>,

    /// Allocation duration
    pub duration: Duration,

    /// Resource utilization during allocation
    pub utilization: f32,

    /// Allocation efficiency
    pub efficiency: f32,
}

/// Performance analysis results
#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    /// Performance compared to sequential execution
    pub sequential_comparison: SequentialComparison,

    /// Performance compared to historical data
    pub historical_comparison: HistoricalComparison,

    /// Bottleneck analysis
    pub bottleneck_analysis: BottleneckAnalysis,

    /// Optimization recommendations
    pub optimization_recommendations: Vec<OptimizationRecommendation>,

    /// Performance score (0.0 to 1.0)
    pub performance_score: f32,
}

/// Sequential execution comparison
#[derive(Debug, Clone)]
pub struct SequentialComparison {
    /// Estimated sequential execution time
    pub estimated_sequential_time: Duration,

    /// Actual parallel execution time
    pub actual_parallel_time: Duration,

    /// Time savings achieved
    pub time_savings: Duration,

    /// Speedup factor
    pub speedup_factor: f32,

    /// Efficiency percentage
    pub efficiency_percentage: f32,
}

/// Historical performance comparison
#[derive(Debug, Clone)]
pub struct HistoricalComparison {
    /// Previous execution times
    pub previous_times: Vec<Duration>,

    /// Performance trend
    pub trend: PerformanceTrend,

    /// Regression detected
    pub regression_detected: bool,

    /// Performance improvement percentage
    pub improvement_percentage: f32,
}

/// Performance trend analysis
#[derive(Debug, Clone)]
pub enum PerformanceTrend {
    /// Performance is improving
    Improving(f32),

    /// Performance is stable
    Stable,

    /// Performance is degrading
    Degrading(f32),

    /// Insufficient data for trend analysis
    InsufficientData,
}

/// Bottleneck analysis results
#[derive(Debug, Clone)]
pub struct BottleneckAnalysis {
    /// Primary bottleneck identified
    pub primary_bottleneck: Option<BottleneckInfo>,

    /// Secondary bottlenecks
    pub secondary_bottlenecks: Vec<BottleneckInfo>,

    /// Bottleneck impact on performance
    pub impact_analysis: BottleneckImpactAnalysis,

    /// Bottleneck mitigation suggestions
    pub mitigation_suggestions: Vec<String>,
}

/// Bottleneck information
#[derive(Debug, Clone)]
pub struct BottleneckInfo {
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,

    /// Bottleneck severity (0.0 to 1.0)
    pub severity: f32,

    /// Resource utilization at bottleneck
    pub resource_utilization: f32,

    /// Time spent in bottleneck
    pub time_spent: Duration,

    /// Bottleneck description
    pub description: String,
}

/// Types of bottlenecks
#[derive(Debug, Clone)]
pub enum BottleneckType {
    /// CPU bottleneck
    Cpu,

    /// Memory bottleneck
    Memory,

    /// GPU bottleneck
    Gpu,

    /// Network I/O bottleneck
    NetworkIo,

    /// Filesystem I/O bottleneck
    FilesystemIo,

    /// Database bottleneck
    Database,

    /// Synchronization bottleneck
    Synchronization,

    /// Resource contention bottleneck
    ResourceContention,
}

/// Bottleneck impact analysis
#[derive(Debug, Clone)]
pub struct BottleneckImpactAnalysis {
    /// Overall performance impact
    pub overall_impact: f32,

    /// Parallelization impact
    pub parallelization_impact: f32,

    /// Resource efficiency impact
    pub resource_efficiency_impact: f32,

    /// Scalability impact
    pub scalability_impact: f32,
}

/// Optimization recommendation
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Recommendation type
    pub recommendation_type: OptimizationType,

    /// Recommendation description
    pub description: String,

    /// Expected impact
    pub expected_impact: f32,

    /// Implementation complexity
    pub implementation_complexity: ComplexityLevel,

    /// Priority for implementation
    pub priority: RecommendationPriority,

    /// Specific parameters for the optimization
    pub parameters: HashMap<String, String>,
}

/// Types of optimizations
#[derive(Debug, Clone)]
pub enum OptimizationType {
    /// Increase parallelism
    IncreaseParallelism,

    /// Decrease parallelism
    DecreaseParallelism,

    /// Adjust resource allocation
    AdjustResourceAllocation,

    /// Change scheduling strategy
    ChangeSchedulingStrategy,

    /// Optimize test batching
    OptimizeTestBatching,

    /// Improve resource sharing
    ImproveResourceSharing,

    /// Add caching
    AddCaching,

    /// Optimize test order
    OptimizeTestOrder,

    /// Custom optimization
    Custom(String),
}

/// Implementation complexity levels
#[derive(Debug, Clone)]
pub enum ComplexityLevel {
    /// Low complexity (configuration change)
    Low,

    /// Medium complexity (code changes required)
    Medium,

    /// High complexity (significant refactoring required)
    High,

    /// Very high complexity (architectural changes required)
    VeryHigh,
}

/// Recommendation priority levels
#[derive(Debug, Clone)]
pub enum RecommendationPriority {
    /// Critical priority (immediate action required)
    Critical,

    /// High priority (implement soon)
    High,

    /// Medium priority (implement when convenient)
    Medium,

    /// Low priority (nice to have)
    Low,
}

// Display implementations for better debugging and logging
impl fmt::Display for SchedulingStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchedulingStrategy::Fifo => write!(f, "FIFO"),
            SchedulingStrategy::ShortestJobFirst => write!(f, "Shortest Job First"),
            SchedulingStrategy::Priority => write!(f, "Priority"),
            SchedulingStrategy::ResourceAware => write!(f, "Resource Aware"),
            SchedulingStrategy::Adaptive => write!(f, "Adaptive"),
            SchedulingStrategy::Custom(name) => write!(f, "Custom: {}", name),
        }
    }
}

impl fmt::Display for DependencyType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DependencyType::Hard => write!(f, "Hard"),
            DependencyType::Soft => write!(f, "Soft"),
            DependencyType::Conflict => write!(f, "Conflict"),
            DependencyType::Ordering => write!(f, "Ordering"),
            DependencyType::Setup => write!(f, "Setup"),
        }
    }
}

impl fmt::Display for BottleneckType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BottleneckType::Cpu => write!(f, "CPU"),
            BottleneckType::Memory => write!(f, "Memory"),
            BottleneckType::Gpu => write!(f, "GPU"),
            BottleneckType::NetworkIo => write!(f, "Network I/O"),
            BottleneckType::FilesystemIo => write!(f, "Filesystem I/O"),
            BottleneckType::Database => write!(f, "Database"),
            BottleneckType::Synchronization => write!(f, "Synchronization"),
            BottleneckType::ResourceContention => write!(f, "Resource Contention"),
        }
    }
}

impl fmt::Display for OptimizationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizationType::IncreaseParallelism => write!(f, "Increase Parallelism"),
            OptimizationType::DecreaseParallelism => write!(f, "Decrease Parallelism"),
            OptimizationType::AdjustResourceAllocation => write!(f, "Adjust Resource Allocation"),
            OptimizationType::ChangeSchedulingStrategy => write!(f, "Change Scheduling Strategy"),
            OptimizationType::OptimizeTestBatching => write!(f, "Optimize Test Batching"),
            OptimizationType::ImproveResourceSharing => write!(f, "Improve Resource Sharing"),
            OptimizationType::AddCaching => write!(f, "Add Caching"),
            OptimizationType::OptimizeTestOrder => write!(f, "Optimize Test Order"),
            OptimizationType::Custom(name) => write!(f, "Custom: {}", name),
        }
    }
}

impl fmt::Display for ComplexityLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComplexityLevel::Low => write!(f, "Low"),
            ComplexityLevel::Medium => write!(f, "Medium"),
            ComplexityLevel::High => write!(f, "High"),
            ComplexityLevel::VeryHigh => write!(f, "Very High"),
        }
    }
}

impl fmt::Display for RecommendationPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RecommendationPriority::Critical => write!(f, "Critical"),
            RecommendationPriority::High => write!(f, "High"),
            RecommendationPriority::Medium => write!(f, "Medium"),
            RecommendationPriority::Low => write!(f, "Low"),
        }
    }
}
