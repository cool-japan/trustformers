//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

// 0.2.1: these came from `crate::resource_manager`, the placeholder tree deleted
// in favour of `crate::resource_management` (see the re-export comment in
// `lib.rs`). The types are the same shape.
use crate::resource_management::{
    AlertSystem, DistributionEvent, ExecutionPerformanceMetrics, ExecutionState, HealthChecker,
    LoadMetrics, WorkerPool,
};
use crate::test_independence_analyzer::TestIndependenceAnalysis;
use crate::test_parallelization::{
    EarlyTerminationStrategy, FailureHandlingStrategy, LoadBalancingStrategy, ResourceAllocation,
    SchedulingStrategy, TestParallelizationMetadata,
};
use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

/// Rebalancing configuration
#[derive(Debug, Clone)]
pub struct RebalancingConfig {
    /// Enable automatic rebalancing
    pub enabled: bool,
    /// Rebalancing interval
    pub interval: Duration,
    /// Load imbalance threshold
    pub imbalance_threshold: f32,
    /// Rebalancing aggressiveness
    pub aggressiveness: f32,
    /// Work stealing configuration
    pub work_stealing: WorkStealingConfig,
}
/// Load balancer for distributing tests across available resources
pub struct LoadBalancer {
    /// Load balancing configuration
    _config: Arc<RwLock<LoadBalancingConfig>>,
    /// Worker pool
    _worker_pool: Arc<WorkerPool>,
    /// Load metrics
    _load_metrics: Arc<Mutex<LoadMetrics>>,
    /// Work distribution history
    _distribution_history: Arc<Mutex<Vec<DistributionEvent>>>,
}
impl LoadBalancer {
    pub(super) async fn new(config: LoadBalancingConfig) -> Result<Self> {
        Ok(Self {
            _config: Arc::new(RwLock::new(config)),
            _worker_pool: Arc::new(WorkerPool::default()),
            _load_metrics: Arc::new(Mutex::new(LoadMetrics::default())),
            _distribution_history: Arc::new(Mutex::new(Vec::new())),
        })
    }
}
/// Capacity the parallel execution engine is allowed to hand out.
///
/// 0.2.1: this used to be a `Default`-constructed all-zero struct that nothing
/// ever read. It now carries measured/configured capacity — see
/// [`AvailableResources::detect`] — and [`ResourceManager::can_allocate`](super::resources::ResourceManager::can_allocate)
/// actually checks against it. Every field is either measured from the running
/// host or taken from the operator-supplied [`ResourcePoolConfig`]; none is
/// invented.
///
/// The GPU and temporary-directory fields deliberately hold *identifiers and
/// slot counts* rather than device/directory descriptors: this engine has no
/// GPU enumerator and no filesystem prober, so it cannot honestly report a
/// device's name, memory size or a directory's free space.
///
/// [`ResourcePoolConfig`]: crate::test_parallelization::ResourcePoolConfig
#[derive(Debug, Default, Clone)]
pub struct AvailableResources {
    /// CPU cores available (logical cores reported by the host)
    pub cpu_cores: f32,
    /// Memory available (MB), as reported by the host at detection time
    pub memory_mb: u64,
    /// GPU device IDs this engine may allocate from (operator-configured pool
    /// membership, not a hardware enumeration)
    pub gpu_device_ids: Vec<usize>,
    /// Network ports available for allocation
    pub network_ports: Vec<u16>,
    /// Number of temporary-directory slots the pool may hand out
    pub temp_directory_slots: usize,
    /// Database connections available
    pub database_connections: usize,
    /// Custom resources
    pub custom_resources: HashMap<String, f64>,
}

impl AvailableResources {
    /// Build the engine's capacity from the running host plus the configured
    /// resource pools.
    ///
    /// CPU cores come from [`std::thread::available_parallelism`] and memory
    /// from `sysinfo`'s available-memory reading; ports, GPU device IDs,
    /// temp-directory slots and database connections come from `pools`. If the
    /// host refuses to report parallelism the core count falls back to 1 — the
    /// only value that is certainly true — rather than to a guess.
    pub fn detect(pools: &crate::test_parallelization::ResourcePoolConfig) -> Self {
        let cpu_cores = std::thread::available_parallelism().map(|n| n.get() as f32).unwrap_or(1.0);

        let mut system = sysinfo::System::new();
        system.refresh_memory();
        let memory_mb = system.available_memory() / (1024 * 1024);

        let port_cfg = &pools.network_port_pool;
        let network_ports: Vec<u16> = if port_cfg.start_port > port_cfg.end_port {
            Vec::new()
        } else {
            (port_cfg.start_port..=port_cfg.end_port)
                .filter(|p| !port_cfg.reserved_ports.contains(p))
                .collect()
        };

        Self {
            cpu_cores,
            memory_mb,
            gpu_device_ids: pools.gpu_device_pool.device_ids.clone(),
            network_ports,
            temp_directory_slots: pools.temp_directory_pool.max_directories,
            database_connections: pools.database_pool.max_connections,
            custom_resources: HashMap::new(),
        }
    }
}
/// Worker scaling configuration
#[derive(Debug, Clone)]
pub struct WorkerScalingConfig {
    /// Enable auto-scaling
    pub enabled: bool,
    /// Scale-up threshold
    pub scale_up_threshold: f32,
    /// Scale-down threshold
    pub scale_down_threshold: f32,
    /// Scaling cooldown
    pub cooldown_period: Duration,
    /// Scaling factor
    pub scaling_factor: f32,
}
/// Work stealing configuration
#[derive(Debug, Clone)]
pub struct WorkStealingConfig {
    /// Enable work stealing
    pub enabled: bool,
    /// Steal threshold
    pub steal_threshold: f32,
    /// Maximum steals per interval
    pub max_steals_per_interval: usize,
    /// Steal attempt timeout
    pub steal_timeout: Duration,
}
/// Execution constraint types
#[derive(Debug, Clone)]
pub enum ExecutionConstraintType {
    /// Must execute before
    Before,
    /// Must execute after
    After,
    /// Cannot execute with
    CannotExecuteWith,
    /// Requires resource
    RequiresResource,
    /// Custom constraint
    Custom(String),
}
/// Dependency metrics for tracking performance
#[derive(Debug, Default)]
pub struct DependencyMetrics {
    /// Total dependencies analyzed
    pub total_analyzed: u64,
    /// Resolution time
    pub resolution_time: Duration,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}
#[derive(Debug, Clone)]
pub struct LoadBalancingConfig {
    /// Load balancing strategy
    pub strategy: LoadBalancingStrategy,
    /// Rebalancing configuration
    pub rebalancing: RebalancingConfig,
    /// Worker configuration
    pub worker_config: WorkerConfig,
    /// Performance thresholds
    pub thresholds: LoadBalancingThresholds,
}
/// Pool item states
#[derive(Debug, Clone)]
pub enum PoolItemState {
    /// Available for allocation
    Available,
    /// Currently allocated
    Allocated,
    /// Under maintenance
    Maintenance,
    /// Failed/unusable
    Failed,
}
/// Scheduled test with priority and metadata
#[derive(Debug, Clone)]
pub struct ScheduledTest {
    /// Test metadata
    pub metadata: TestParallelizationMetadata,
    /// Calculated priority
    pub priority: f32,
    /// Scheduling timestamp
    pub scheduled_at: DateTime<Utc>,
    /// Estimated start time
    pub estimated_start: Option<DateTime<Utc>>,
    /// Resource requirements
    pub resource_requirements: ResourceRequirement,
    /// Scheduling constraints
    pub constraints: Vec<SchedulingConstraint>,
    /// Retry count
    pub retry_count: usize,
    /// Scheduling metadata
    pub scheduling_metadata: HashMap<String, String>,
}
/// Alert levels
#[derive(Debug, Clone)]
pub enum AlertLevel {
    /// Info level
    Info,
    /// Warning level
    Warning,
    /// Error level
    Error,
    /// Critical level
    Critical,
}

pub struct ExecutionMonitor {
    /// Monitoring configuration
    _config: Arc<RwLock<MonitoringConfig>>,
    /// Active executions
    _active_executions: Arc<Mutex<HashMap<String, ExecutionState>>>,
    /// Performance metrics
    _performance_metrics: Arc<Mutex<ExecutionPerformanceMetrics>>,
    /// Health checks
    _health_checker: Arc<HealthChecker>,
    /// Alert system
    _alert_system: Arc<AlertSystem>,
}
impl ExecutionMonitor {
    pub(super) async fn new(config: MonitoringConfig) -> Result<Self> {
        Ok(Self {
            _config: Arc::new(RwLock::new(config)),
            _active_executions: Arc::new(Mutex::new(HashMap::new())),
            _performance_metrics: Arc::new(Mutex::new(ExecutionPerformanceMetrics::default())),
            _health_checker: Arc::new(HealthChecker::new()),
            _alert_system: Arc::new(AlertSystem::new()),
        })
    }
}
/// Worker specialization configuration
#[derive(Debug, Clone)]
pub struct WorkerSpecializationConfig {
    /// Enable worker specialization
    pub enabled: bool,
    /// Specialization by test category
    pub by_category: bool,
    /// Specialization by resource type
    pub by_resource: bool,
    /// Specialization by performance characteristics
    pub by_performance: bool,
}
/// Resource pool item
#[derive(Debug, Clone)]
pub struct PoolItem {
    /// Item identifier
    pub id: String,
    /// Item value/resource
    pub resource: String,
    /// Item metadata
    pub metadata: HashMap<String, String>,
    /// Item state
    pub state: PoolItemState,
    /// Last used timestamp
    pub last_used: Option<DateTime<Utc>>,
}
/// Alert destination
#[derive(Debug, Clone)]
pub struct AlertDestination {
    /// Destination type
    pub destination_type: AlertDestinationType,
    /// Destination configuration
    pub config: HashMap<String, String>,
    /// Alert levels for this destination
    pub alert_levels: Vec<AlertLevel>,
}
/// Queue statistics, all measured by [`PriorityQueue`](super::scheduling::PriorityQueue) from its own traffic.
#[derive(Debug, Default, Clone)]
pub struct QueueStatistics {
    /// Total items enqueued
    pub total_enqueued: u64,
    /// Total items dequeued
    pub total_dequeued: u64,
    /// Current queue size
    pub current_size: usize,
    /// Peak queue size
    pub peak_size: usize,
    /// Average wait time
    pub average_wait_time: Duration,
    /// Queue throughput
    pub throughput: f64,
}
/// Directory permissions
#[derive(Debug, Clone)]
pub struct DirectoryPermissions {
    /// Read permission
    pub read: bool,
    /// Write permission
    pub write: bool,
    /// Execute permission
    pub execute: bool,
    /// Owner
    pub owner: String,
    /// Group
    pub group: String,
}
/// Execution session state
#[derive(Debug, Clone)]
pub enum ExecutionSessionState {
    /// Session initializing
    Initializing,
    /// Session running
    Running,
    /// Session paused
    Paused,
    /// Session completing
    Completing,
    /// Session completed
    Completed,
    /// Session failed
    Failed(String),
}
/// Dependency node types
#[derive(Debug, Clone)]
pub enum DependencyNodeType {
    /// Regular test
    Test,
    /// Setup task
    Setup,
    /// Teardown task
    Teardown,
    /// Resource initialization
    ResourceInit,
    /// Custom node type
    Custom(String),
}
/// Execution queue for managing test execution order
#[derive(Debug)]
pub struct ExecutionQueue {
    /// Queued tests
    pub queued_tests: VecDeque<ScheduledTest>,
    /// Queue metadata
    pub metadata: ExecutionQueueMetadata,
}
impl ExecutionQueue {
    pub(super) fn new() -> Self {
        Self {
            queued_tests: VecDeque::new(),
            metadata: ExecutionQueueMetadata::default(),
        }
    }
}
/// Performance tracking configuration
#[derive(Debug, Clone)]
pub struct PerformanceTrackingConfig {
    /// Enable detailed tracking
    pub detailed_tracking: bool,
    /// Metrics collection interval
    pub collection_interval: Duration,
    /// Metrics retention period
    pub retention_period: Duration,
    /// Performance analysis interval
    pub analysis_interval: Duration,
    /// Regression detection
    pub regression_detection: bool,
}
/// Alert thresholds
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// High error rate threshold
    pub high_error_rate: f32,
    /// High latency threshold
    pub high_latency: Duration,
    /// Resource exhaustion threshold
    pub resource_exhaustion: f32,
    /// Queue backup threshold
    pub queue_backup: usize,
    /// Worker failure threshold
    pub worker_failure: usize,
}
/// Alert destination types
#[derive(Debug, Clone)]
pub enum AlertDestinationType {
    /// Log file
    Log,
    /// Console output
    Console,
    /// Email notification
    Email,
    /// Webhook
    Webhook,
    /// Custom destination
    Custom(String),
}
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Performance tracking
    pub performance_tracking: PerformanceTrackingConfig,
    /// Health check configuration
    pub health_checks: HealthCheckConfig,
    /// Alert configuration
    pub alerts: AlertConfig,
}
/// Dependency graph statistics
#[derive(Debug, Default)]
pub struct DependencyGraphStatistics {
    /// Number of nodes
    pub node_count: usize,
    /// Number of edges
    pub edge_count: usize,
    /// Graph density
    pub density: f32,
    /// Longest dependency chain
    pub longest_chain: usize,
    /// Circular dependencies
    pub circular_dependencies: Vec<Vec<String>>,
}
/// Execution constraint
#[derive(Debug, Clone)]
pub struct ExecutionConstraint {
    /// Constraint type
    pub constraint_type: ExecutionConstraintType,
    /// Constraint value
    pub value: String,
    /// Constraint priority
    pub priority: f32,
}
/// Resource allocation state
#[derive(Debug, Clone)]
pub struct ResourceAllocationState {
    /// Allocated resources
    pub allocated: ResourceAllocation,
    /// The requirement this allocation reserved capacity for. Kept so the
    /// manager can subtract the right amounts again on release.
    pub requirement: ResourceRequirement,
    /// Allocation timestamp
    pub allocated_at: DateTime<Utc>,
    /// Expected deallocation time
    pub expected_deallocation: Option<DateTime<Utc>>,
    /// Measured allocation efficiency, or `None` when nothing measured it —
    /// which is every allocation this engine produces today.
    pub efficiency: Option<f32>,
    /// Allocation metadata
    pub metadata: HashMap<String, String>,
}
/// Resource requirement specification (moved from analyzer to avoid circular deps)
#[derive(Debug, Clone)]
pub struct ResourceRequirement {
    /// Resource type identifier (e.g. "network_port", "gpu_device", "temp_directory")
    pub resource_type: String,
    /// Minimum amount required
    pub min_amount: f64,
    /// CPU cores required
    pub cpu_cores: f32,
    /// Memory required (MB)
    pub memory_mb: u64,
    /// GPU devices required
    pub gpu_devices: Vec<usize>,
    /// Network ports required
    pub network_ports: usize,
    /// Temporary directories required
    pub temp_directories: usize,
    /// Database connections required
    pub database_connections: usize,
    /// Custom resources required
    pub custom_resources: HashMap<String, f64>,
}
/// Dependency node metadata
#[derive(Debug, Clone)]
pub struct DependencyNodeMetadata {
    /// Node type
    pub node_type: DependencyNodeType,
    /// Priority level
    pub priority: f32,
    /// Resource requirements
    pub resource_requirements: ResourceRequirement,
    /// Execution constraints
    pub constraints: Vec<ExecutionConstraint>,
}
/// Execution session state
#[derive(Debug, Clone)]
pub struct ExecutionSession {
    /// Session ID
    pub id: String,
    /// Session start time
    pub start_time: DateTime<Utc>,
    /// Test analysis
    pub analysis: TestIndependenceAnalysis,
    /// Session configuration
    pub config: ExecutionSessionConfig,
    /// Session state
    pub state: ExecutionSessionState,
}
impl ExecutionSession {
    pub(super) fn new(id: String, analysis: TestIndependenceAnalysis) -> Self {
        Self {
            id,
            start_time: Utc::now(),
            analysis,
            config: ExecutionSessionConfig::default(),
            state: ExecutionSessionState::Initializing,
        }
    }
}
/// Translate the parallelization crate's monitoring configuration into the
/// `resource_management` one.
///
/// Only the three fields with an exact counterpart are carried across; the
/// remainder keep `resource_management`'s own defaults rather than being
/// invented from unrelated values.
pub(super) fn monitoring_config_from(
    config: &crate::test_parallelization::ResourceMonitoringConfig,
) -> crate::resource_management::ResourceMonitoringConfig {
    crate::resource_management::ResourceMonitoringConfig {
        enable_real_time: config.enabled,
        monitoring_interval_secs: config.monitoring_interval.as_secs(),
        enable_alerts: config.alerts.enabled,
        ..Default::default()
    }
}

/// Load balancing thresholds
#[derive(Debug, Clone)]
pub struct LoadBalancingThresholds {
    /// CPU utilization threshold
    pub cpu_threshold: f32,
    /// Memory utilization threshold
    pub memory_threshold: f32,
    /// Queue length threshold
    pub queue_threshold: usize,
    /// Response time threshold
    pub response_time_threshold: Duration,
    /// Error rate threshold
    pub error_rate_threshold: f32,
}
/// Scheduling event details
#[derive(Debug, Clone)]
pub struct SchedulingEventDetails {
    /// Priority at time of event
    pub priority: f32,
    /// Queue position at time of event
    pub queue_position: Option<usize>,
    /// Resource allocation at time of event
    pub resource_allocation: Option<String>,
    /// Worker assignment
    pub worker_id: Option<String>,
    /// Wait time
    pub wait_time: Option<Duration>,
    /// Additional details
    pub additional_details: HashMap<String, String>,
}
/// Adaptive scheduling parameters
#[derive(Debug, Clone)]
pub struct AdaptiveSchedulingParams {
    /// Learning rate for adaptation
    pub learning_rate: f32,
    /// Adaptation interval
    pub adaptation_interval: Duration,
    /// Performance history window
    pub history_window: usize,
    /// Minimum confidence for adaptation
    pub min_confidence: f32,
    /// Maximum adaptation rate
    pub max_adaptation_rate: f32,
}
/// Scheduling event types
#[derive(Debug, Clone)]
pub enum SchedulingEventType {
    /// Test queued
    Queued,
    /// Test scheduled
    Scheduled,
    /// Test started
    Started,
    /// Test completed
    Completed,
    /// Test failed
    Failed,
    /// Test cancelled
    Cancelled,
    /// Test rescheduled
    Rescheduled,
    /// Priority adjusted
    PriorityAdjusted,
}

/// Dependency tracker for managing test dependencies
#[derive(Debug)]
pub struct DependencyTracker {
    /// Dependency graph
    pub(crate) _dependency_graph: Arc<RwLock<DependencyGraph>>,
    /// Dependency resolution cache
    pub(crate) _resolution_cache: Arc<Mutex<HashMap<String, DependencyResolution>>>,
    /// Blocked tests
    pub(crate) _blocked_tests: Arc<Mutex<HashMap<String, Vec<String>>>>,
    /// Dependency metrics
    pub(crate) _metrics: Arc<Mutex<DependencyMetrics>>,
}
/// Dependency resolution result
#[derive(Debug, Clone)]
pub struct DependencyResolution {
    /// Resolved dependencies
    pub resolved: Vec<String>,
    /// Resolution timestamp
    pub timestamp: DateTime<Utc>,
    /// Resolution status
    pub status: ResolutionStatus,
}
/// Dependency graph representation
#[derive(Debug, Default)]
pub struct DependencyGraph {
    /// Adjacency list
    _adjacency_list: HashMap<String, Vec<String>>,
    /// Reverse adjacency list
    _reverse_adjacency_list: HashMap<String, Vec<String>>,
    /// Node metadata
    _node_metadata: HashMap<String, DependencyNodeMetadata>,
    /// Graph statistics
    _statistics: DependencyGraphStatistics,
}
/// Resource pool configuration
#[derive(Debug, Clone)]
pub struct ResourcePoolConfig {
    /// Minimum pool size
    pub min_size: usize,
    /// Maximum pool size
    pub max_size: usize,
    /// Growth strategy
    pub growth_strategy: PoolGrowthStrategy,
    /// Cleanup interval
    pub cleanup_interval: Duration,
    /// Item timeout
    pub item_timeout: Duration,
}
/// Cleanup policy for temporary directories
#[derive(Debug, Clone)]
pub enum CleanupPolicy {
    /// Immediate cleanup
    Immediate,
    /// Cleanup after delay
    Delayed(Duration),
    /// Manual cleanup
    Manual,
    /// Custom cleanup
    Custom(String),
}
/// Alert configuration
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Enable alerts
    pub enabled: bool,
    /// Alert cooldown period
    pub cooldown_period: Duration,
    /// Alert thresholds
    pub thresholds: AlertThresholds,
    /// Alert destinations
    pub destinations: Vec<AlertDestination>,
}
/// Pool growth strategies
#[derive(Debug, Clone)]
pub enum PoolGrowthStrategy {
    /// Fixed size pool
    Fixed,
    /// Grow on demand
    OnDemand,
    /// Preemptive growth
    Preemptive,
    /// Custom strategy
    Custom(String),
}
/// Scheduling constraint
#[derive(Debug, Clone)]
pub struct SchedulingConstraint {
    /// Constraint type
    pub constraint_type: SchedulingConstraintType,
    /// Constraint value
    pub value: String,
    /// Constraint priority
    pub priority: f32,
    /// Constraint deadline
    pub deadline: Option<DateTime<Utc>>,
}
/// Execution session configuration
#[derive(Debug, Clone)]
pub struct ExecutionSessionConfig {
    /// Maximum concurrent tests
    pub max_concurrent_tests: usize,
    /// Session timeout
    pub session_timeout: Duration,
    /// Failure handling strategy
    pub failure_handling: FailureHandlingStrategy,
    /// Early termination strategy
    pub early_termination: EarlyTerminationStrategy,
}
/// Engine-wide statistics, accumulated from real executions.
///
/// 0.2.1: this used to carry a `resource_efficiency` counter that nothing ever
/// wrote or read; it is gone rather than left reporting a permanent zero.
/// `average_parallelism` is now derived on demand from the concurrency actually
/// observed at each test's start, instead of being a second stored counter.
#[derive(Debug)]
pub struct EngineStatistics {
    /// Test bodies that ran to completion.
    pub total_tests_executed: AtomicU64,
    /// Summed wall-clock execution time of those bodies, in microseconds.
    pub total_execution_micros: AtomicU64,
    /// Summed observed concurrency, the numerator of the parallelism average.
    pub summed_observed_concurrency: AtomicU64,
    /// Highest concurrency observed across this engine's lifetime.
    pub peak_parallelism: AtomicU64,
    /// Engine uptime
    pub uptime_start: Instant,
}
impl EngineStatistics {
    pub(super) fn new() -> Self {
        Self {
            total_tests_executed: AtomicU64::new(0),
            total_execution_micros: AtomicU64::new(0),
            summed_observed_concurrency: AtomicU64::new(0),
            peak_parallelism: AtomicU64::new(0),
            uptime_start: Instant::now(),
        }
    }

    /// Fold one completed execution into the totals.
    pub(super) fn record(&self, execution_time: Duration, observed_concurrency: usize) {
        self.total_tests_executed.fetch_add(1, Ordering::SeqCst);
        self.total_execution_micros.fetch_add(
            execution_time.as_micros().min(u128::from(u64::MAX)) as u64,
            Ordering::SeqCst,
        );
        let concurrency = observed_concurrency as u64;
        self.summed_observed_concurrency.fetch_add(concurrency, Ordering::SeqCst);
        self.peak_parallelism.fetch_max(concurrency, Ordering::SeqCst);
    }

    /// Mean concurrency observed at test start, or `None` before any test ran.
    pub fn average_parallelism(&self) -> Option<f64> {
        let executed = self.total_tests_executed.load(Ordering::SeqCst);
        if executed == 0 {
            return None;
        }
        Some(self.summed_observed_concurrency.load(Ordering::SeqCst) as f64 / executed as f64)
    }

    /// Mean execution time, or `None` before any test ran.
    pub fn average_execution_time(&self) -> Option<Duration> {
        let executed = self.total_tests_executed.load(Ordering::SeqCst);
        if executed == 0 {
            return None;
        }
        Some(Duration::from_micros(
            self.total_execution_micros.load(Ordering::SeqCst) / executed,
        ))
    }
}
/// Worker configuration
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Initial worker count
    pub initial_worker_count: usize,
    /// Minimum workers
    pub min_workers: usize,
    /// Maximum workers
    pub max_workers: usize,
    /// Worker scaling configuration
    pub scaling: WorkerScalingConfig,
    /// Worker specialization
    pub specialization: WorkerSpecializationConfig,
}
/// Queue configuration
#[derive(Debug)]
pub struct QueueConfig {
    /// Maximum queue size; 0 means unbounded.
    pub max_size: usize,
    /// Whether [`PriorityQueue::pop`](super::scheduling::PriorityQueue::pop) honours priority (otherwise plain FIFO).
    pub priority_enabled: bool,
}
impl Default for QueueConfig {
    /// Unbounded and priority-ordered -- the behaviour the type name promises.
    ///
    /// 0.2.1: this was `#[derive(Default)]`, which produced `priority_enabled:
    /// false`, i.e. a "PriorityQueue" that ignored priority. Nothing noticed
    /// because nothing read the field at the time.
    fn default() -> Self {
        Self {
            max_size: 0,
            priority_enabled: true,
        }
    }
}
/// Execution queue metadata
#[derive(Debug, Default)]
pub struct ExecutionQueueMetadata {
    /// Total tests queued
    pub total_queued: u64,
    /// Total tests dequeued
    pub total_dequeued: u64,
    /// Average queue time
    pub average_queue_time: Duration,
    /// Queue efficiency
    pub efficiency: f32,
}
/// Queue configuration
/// Scheduling event for history tracking
#[derive(Debug, Clone)]
pub struct SchedulingEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event type
    pub event_type: SchedulingEventType,
    /// Test identifier
    pub test_id: String,
    /// Event details
    pub details: SchedulingEventDetails,
    /// Event metadata
    pub metadata: HashMap<String, String>,
}
/// Scheduler performance metrics.
///
/// The first two fields are counted and timed by [`TestScheduler`](super::scheduling::TestScheduler) on every
/// real decision. The remaining four are `Option` and stay `None`: judging
/// scheduling *accuracy*, queue *efficiency* or priority *effectiveness*
/// requires an outcome oracle this crate does not have, and nothing here
/// resolves dependencies, so reporting `0.0` for them would read as a measured
/// zero rather than an unmeasured field.
#[derive(Debug, Default, Clone)]
pub struct SchedulerMetrics {
    /// Scheduling decisions made (queue pushes and pops).
    pub decisions_made: u64,
    /// Mean wall-clock time one scheduling decision took.
    pub average_decision_time: Duration,
    /// Not observed: no oracle compares scheduled order against an optimum.
    pub scheduling_accuracy: Option<f32>,
    /// Not observed: no baseline exists to divide realised throughput by.
    pub queue_efficiency: Option<f32>,
    /// Not observed: priorities are honoured but never scored after the fact.
    pub priority_effectiveness: Option<f32>,
    /// Not observed: this scheduler does not resolve dependencies.
    pub dependency_resolution_time: Option<Duration>,
}
/// Health check configuration
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// Health check interval
    pub interval: Duration,
    /// Health check timeout
    pub timeout: Duration,
    /// Failure threshold
    pub failure_threshold: usize,
    /// Recovery check interval
    pub recovery_interval: Duration,
    /// Enable deep health checks
    pub deep_checks: bool,
}
#[derive(Debug, Clone)]
pub enum ResolutionStatus {
    Success,
    Failed(String),
    Partial,
}
/// Resource pool for managing shared resources
#[derive(Debug)]
pub struct ResourcePool {
    /// Pool name
    pub name: String,
    /// Resource type
    pub resource_type: String,
    /// Available items
    pub available: Vec<PoolItem>,
    /// Allocated items
    pub allocated: HashMap<String, PoolItem>,
    /// Pool configuration
    pub config: ResourcePoolConfig,
    /// Pool statistics
    pub stats: ResourcePoolStats,
}
/// Priority weights for scheduling decisions
#[derive(Debug, Clone)]
pub struct PriorityWeights {
    /// Test category weight
    pub category_weight: f32,
    /// Estimated duration weight
    pub duration_weight: f32,
    /// Resource requirements weight
    pub resource_weight: f32,
    /// Dependency chain weight
    pub dependency_weight: f32,
    /// Historical performance weight
    pub performance_weight: f32,
    /// Failure rate weight
    pub failure_rate_weight: f32,
}
/// Scheduling constraint types
#[derive(Debug, Clone)]
pub enum SchedulingConstraintType {
    /// Time window constraint
    TimeWindow,
    /// Resource availability constraint
    ResourceAvailability,
    /// Dependency constraint
    Dependency,
    /// Priority constraint
    Priority,
    /// Custom constraint
    Custom(String),
}
/// Resource pool statistics
#[derive(Debug, Default)]
pub struct ResourcePoolStats {
    /// Total allocations
    pub total_allocations: u64,
    /// Current allocations
    pub current_allocations: usize,
    /// Peak allocations
    pub peak_allocations: usize,
    /// Allocation failures
    pub allocation_failures: u64,
    /// Average allocation time
    pub average_allocation_time: Duration,
    /// Pool utilization
    pub utilization: f32,
}
/// Configuration structures
#[derive(Debug, Clone, Default)]
pub struct SchedulingConfig {
    /// Primary scheduling strategy
    pub strategy: SchedulingStrategy,
    /// Priority weights
    pub priority_weights: PriorityWeights,
    /// Queue management
    pub queue_management: QueueManagementConfig,
    /// Adaptive scheduling parameters
    pub adaptive_params: AdaptiveSchedulingParams,
}
/// Queue management configuration
#[derive(Debug, Clone)]
pub struct QueueManagementConfig {
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Queue timeout
    pub queue_timeout: Duration,
    /// Priority boost interval
    pub priority_boost_interval: Duration,
    /// Starvation prevention
    pub starvation_prevention: bool,
    /// Queue compaction interval
    pub compaction_interval: Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0
        }
        fn next_f64(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn next_f32(&mut self) -> f32 {
            self.next_f64() as f32
        }
        fn next_usize(&mut self, bound: usize) -> usize {
            (self.next_u64() as usize) % bound.max(1)
        }
    }

    // ---- Enum variant tests ----
    #[test]
    fn test_execution_constraint_type_variants() {
        let types = [
            ExecutionConstraintType::Before,
            ExecutionConstraintType::After,
            ExecutionConstraintType::CannotExecuteWith,
            ExecutionConstraintType::RequiresResource,
            ExecutionConstraintType::Custom("test".to_string()),
        ];
        assert_eq!(types.len(), 5);
    }

    #[test]
    fn test_pool_item_state_variants() {
        let states = [
            PoolItemState::Available,
            PoolItemState::Allocated,
            PoolItemState::Maintenance,
            PoolItemState::Failed,
        ];
        assert_eq!(states.len(), 4);
    }

    #[test]
    fn test_alert_level_variants() {
        let levels = [
            AlertLevel::Info,
            AlertLevel::Warning,
            AlertLevel::Error,
            AlertLevel::Critical,
        ];
        assert_eq!(levels.len(), 4);
    }

    #[test]
    fn test_execution_session_state_variants() {
        let states = [
            ExecutionSessionState::Initializing,
            ExecutionSessionState::Running,
            ExecutionSessionState::Paused,
            ExecutionSessionState::Completing,
            ExecutionSessionState::Completed,
            ExecutionSessionState::Failed("error".to_string()),
        ];
        assert_eq!(states.len(), 6);
    }

    #[test]
    fn test_dependency_node_type_variants() {
        let types = [
            DependencyNodeType::Test,
            DependencyNodeType::Setup,
            DependencyNodeType::Teardown,
            DependencyNodeType::ResourceInit,
            DependencyNodeType::Custom("custom".to_string()),
        ];
        assert_eq!(types.len(), 5);
    }

    #[test]
    fn test_scheduling_event_type_variants() {
        let types = [
            SchedulingEventType::Queued,
            SchedulingEventType::Scheduled,
            SchedulingEventType::Started,
            SchedulingEventType::Completed,
            SchedulingEventType::Failed,
            SchedulingEventType::Cancelled,
            SchedulingEventType::Rescheduled,
            SchedulingEventType::PriorityAdjusted,
        ];
        assert_eq!(types.len(), 8);
    }

    #[test]
    fn test_cleanup_policy_variants() {
        let policies = [
            CleanupPolicy::Immediate,
            CleanupPolicy::Delayed(Duration::from_secs(60)),
            CleanupPolicy::Manual,
            CleanupPolicy::Custom("rotate".to_string()),
        ];
        assert_eq!(policies.len(), 4);
    }

    #[test]
    fn test_pool_growth_strategy_variants() {
        let strategies = [
            PoolGrowthStrategy::Fixed,
            PoolGrowthStrategy::OnDemand,
            PoolGrowthStrategy::Preemptive,
            PoolGrowthStrategy::Custom("adaptive".to_string()),
        ];
        assert_eq!(strategies.len(), 4);
    }

    #[test]
    fn test_scheduling_constraint_type_variants() {
        let types = [
            SchedulingConstraintType::TimeWindow,
            SchedulingConstraintType::ResourceAvailability,
            SchedulingConstraintType::Dependency,
            SchedulingConstraintType::Priority,
            SchedulingConstraintType::Custom("custom_constraint".to_string()),
        ];
        assert_eq!(types.len(), 5);
    }

    #[test]
    fn test_resolution_status_variants() {
        let statuses = [
            ResolutionStatus::Success,
            ResolutionStatus::Failed("timeout".to_string()),
            ResolutionStatus::Partial,
        ];
        assert_eq!(statuses.len(), 3);
    }

    // ---- Default impl tests ----
    #[test]
    fn test_dependency_metrics_default() {
        let m = DependencyMetrics::default();
        assert_eq!(m.total_analyzed, 0);
        assert!((m.cache_hit_rate - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_queue_statistics_default() {
        let s = QueueStatistics::default();
        assert_eq!(s.total_enqueued, 0);
        assert_eq!(s.total_dequeued, 0);
        assert_eq!(s.current_size, 0);
    }

    #[test]
    fn test_dependency_graph_default() {
        let g = DependencyGraph::default();
        let formatted = format!("{:?}", g);
        assert!(formatted.contains("DependencyGraph"));
    }

    #[test]
    fn test_dependency_graph_statistics_default() {
        let s = DependencyGraphStatistics::default();
        assert_eq!(s.node_count, 0);
        assert_eq!(s.edge_count, 0);
        assert!(s.circular_dependencies.is_empty());
    }

    #[test]
    fn test_scheduler_metrics_default() {
        let m = SchedulerMetrics::default();
        assert_eq!(m.decisions_made, 0);
        assert_eq!(m.average_decision_time, Duration::ZERO);
        // The four unmeasured fields are structurally absent, not zero.
        assert!(m.scheduling_accuracy.is_none());
        assert!(m.queue_efficiency.is_none());
        assert!(m.priority_effectiveness.is_none());
        assert!(m.dependency_resolution_time.is_none());
    }

    #[test]
    fn test_resource_pool_stats_default() {
        let s = ResourcePoolStats::default();
        assert_eq!(s.total_allocations, 0);
        assert_eq!(s.current_allocations, 0);
        assert_eq!(s.peak_allocations, 0);
    }

    // ---- Struct construction tests ----
    #[test]
    fn test_rebalancing_config_construction() {
        let c = RebalancingConfig {
            enabled: true,
            interval: Duration::from_secs(30),
            imbalance_threshold: 0.2,
            aggressiveness: 0.5,
            work_stealing: WorkStealingConfig {
                enabled: true,
                steal_threshold: 0.7,
                max_steals_per_interval: 10,
                steal_timeout: Duration::from_millis(100),
            },
        };
        assert!(c.enabled);
        assert!(c.work_stealing.enabled);
    }

    #[test]
    fn test_resource_pool_config_construction() {
        let c = ResourcePoolConfig {
            min_size: 5,
            max_size: 50,
            growth_strategy: PoolGrowthStrategy::OnDemand,
            cleanup_interval: Duration::from_secs(60),
            item_timeout: Duration::from_secs(300),
        };
        assert!(c.max_size > c.min_size);
    }

    #[test]
    fn test_adaptive_scheduling_params_construction() {
        let p = AdaptiveSchedulingParams {
            learning_rate: 0.01,
            adaptation_interval: Duration::from_secs(10),
            history_window: 100,
            min_confidence: 0.5,
            max_adaptation_rate: 0.1,
        };
        assert!((p.learning_rate - 0.01).abs() < f32::EPSILON);
    }

    #[test]
    fn test_priority_weights_construction() {
        let w = PriorityWeights {
            category_weight: 1.0,
            duration_weight: 0.8,
            resource_weight: 0.6,
            dependency_weight: 1.2,
            performance_weight: 0.5,
            failure_rate_weight: 1.5,
        };
        assert!(w.failure_rate_weight > w.category_weight);
    }

    #[test]
    fn test_resource_requirement_construction() {
        let r = ResourceRequirement {
            resource_type: "gpu_device".to_string(),
            min_amount: 1.0,
            cpu_cores: 4.0,
            memory_mb: 8192,
            gpu_devices: vec![0],
            network_ports: 2,
            temp_directories: 1,
            database_connections: 0,
            custom_resources: HashMap::new(),
        };
        assert_eq!(r.resource_type, "gpu_device");
        assert_eq!(r.gpu_devices.len(), 1);
    }

    #[test]
    fn test_execution_constraint_construction() {
        let c = ExecutionConstraint {
            constraint_type: ExecutionConstraintType::RequiresResource,
            value: "gpu_0".to_string(),
            priority: 1.0,
        };
        assert!((c.priority - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_available_resources_default() {
        let r = AvailableResources::default();
        assert!((r.cpu_cores - 0.0).abs() < f32::EPSILON);
        assert_eq!(r.memory_mb, 0);
        assert!(r.gpu_device_ids.is_empty());
        assert_eq!(r.temp_directory_slots, 0);
    }

    #[test]
    fn test_worker_scaling_config_construction() {
        let c = WorkerScalingConfig {
            enabled: true,
            scale_up_threshold: 0.8,
            scale_down_threshold: 0.3,
            cooldown_period: Duration::from_secs(60),
            scaling_factor: 2.0,
        };
        assert!(c.scale_up_threshold > c.scale_down_threshold);
    }

    // ---- LCG-driven tests ----
    #[test]
    fn test_lcg_generates_priority_values() {
        let mut rng = Lcg::new(42);
        for _ in 0..50 {
            let priority = rng.next_f32();
            assert!((0.0..1.0).contains(&priority));
        }
    }

    #[test]
    fn test_lcg_generates_queue_sizes() {
        let mut rng = Lcg::new(999);
        for _ in 0..30 {
            let size = rng.next_usize(1000);
            assert!(size < 1000);
        }
    }

    #[test]
    fn test_lcg_selects_alert_levels() {
        let mut rng = Lcg::new(1234);
        let levels = [
            AlertLevel::Info,
            AlertLevel::Warning,
            AlertLevel::Error,
            AlertLevel::Critical,
        ];
        for _ in 0..20 {
            let idx = rng.next_usize(levels.len());
            let formatted = format!("{:?}", levels[idx]);
            assert!(!formatted.is_empty());
        }
    }
}
