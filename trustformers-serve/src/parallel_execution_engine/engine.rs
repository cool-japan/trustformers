//! The parallel test execution engine itself.
//!
//! Split out of `types.rs` in 0.2.1 to keep every file in this module under the
//! 2000-line limit. This file owns the execution loop, the caller-supplied
//! [`TestRunner`] seam and the measured [`ParallelExecutionOutcome`].

use super::resources::{AllocationFeasibility, ResourceManager};
use super::scheduling::TestScheduler;
use super::types::*;
use crate::test_independence_analyzer::TestIndependenceAnalysis;
use crate::test_parallelization::{
    DependencyType, LoadBalancingStrategy, ResourceAllocation, SchedulingStrategy, TestDependency,
    TestParallelizationConfig, TestParallelizationMetadata,
};
use crate::test_timeout_optimization::{TestExecutionResult, TestTimeoutFramework};
use anyhow::Result;
use chrono::{DateTime, Utc};
use log::{debug, error, info, warn};
use parking_lot::{Mutex, RwLock};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::task::{JoinHandle, JoinSet};

/// The body of one test, as supplied by the caller.
///
/// The engine schedules tests by name; it has no way to turn a name into
/// something runnable on its own. A [`TestRunner`] provides that mapping, and
/// the future it hands back is what [`TestTimeoutFramework::execute_test`]
/// actually runs and times.
pub type TestBodyFuture = std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send>>;
/// A callable test body. Receives the framework's progress tracker so a test
/// can report progress for early-termination and stall detection.
pub type TestBody = Box<
    dyn FnOnce(Arc<crate::test_timeout_optimization::TestProgressTracker>) -> TestBodyFuture
        + Send
        + 'static,
>;
/// Supplies runnable bodies for scheduled tests.
///
/// 0.2.1: without this seam the engine had no executor at all, and its
/// `start_test_execution` invented a `TestExecutionResult` (a 100 ms sleep, then
/// hardcoded CPU/memory/task counts) for every test. There is deliberately no
/// default implementation: an engine with no runner refuses to execute rather
/// than manufacturing results.
pub trait TestRunner: Send + Sync {
    /// Return the body for `test_name`, or `None` when this runner does not
    /// know the test. `None` makes the engine report the test as unrunnable
    /// instead of inventing an outcome for it.
    fn test_body(&self, test_name: &str) -> Option<TestBody>;
}
/// Everything the engine actually observed about one executed test.
///
/// 0.2.1: this replaces the engine's old `TestParallelizationResult` return
/// value, every numeric field of which was a constant written into the source
/// (`parallel_efficiency: 0.85`, `speedup_factor: 2.0`, CPU 75/85, a
/// `"CPU bound operations"` bottleneck string, and so on) regardless of what
/// ran. Each field below is measured: `result` comes from
/// [`TestTimeoutFramework::execute_test`], the timestamps are taken around the
/// real execution, `queue_wait` is the gap between scheduling and start, and
/// `observed_concurrency` is the in-flight count sampled when this test began.
#[derive(Debug, Clone)]
pub struct ParallelExecutionOutcome {
    /// The real execution result, produced and timed by the timeout framework.
    pub result: TestExecutionResult,
    /// The reservation this test ran under, with its real elapsed duration.
    pub allocation: ResourceAllocation,
    /// When the test entered the scheduler queue.
    pub scheduled_at: DateTime<Utc>,
    /// When its body actually started running.
    pub started_at: DateTime<Utc>,
    /// When its body finished.
    pub completed_at: DateTime<Utc>,
    /// Measured time spent waiting in the queue.
    pub queue_wait: Duration,
    /// Number of test bodies in flight at the moment this one started,
    /// counting this one.
    pub observed_concurrency: usize,
    /// The scheduling strategy in force for this run.
    pub strategy: SchedulingStrategy,
}
/// Parallel test execution engine
pub struct ParallelExecutionEngine {
    /// Configuration
    config: Arc<RwLock<TestParallelizationConfig>>,
    /// Test timeout framework: the real executor behind every test body.
    timeout_framework: Arc<TestTimeoutFramework>,
    /// Test scheduler
    scheduler: Arc<TestScheduler>,
    /// Resource manager
    resource_manager: Arc<ResourceManager>,
    /// Load balancer
    _load_balancer: Arc<LoadBalancer>,
    /// Execution monitor
    _execution_monitor: Arc<ExecutionMonitor>,
    /// Active execution sessions
    active_sessions: Arc<Mutex<HashMap<String, ExecutionSession>>>,
    /// Execution queue
    _execution_queue: Arc<Mutex<ExecutionQueue>>,
    /// Engine statistics
    engine_stats: Arc<EngineStatistics>,
    /// Shutdown signal, honoured by the execution loop.
    shutdown: Arc<AtomicBool>,
    /// Caller-supplied source of runnable test bodies.
    test_runner: Option<Arc<dyn TestRunner>>,
    /// Test bodies currently running, used to observe real concurrency.
    in_flight: Arc<AtomicUsize>,
    /// Background tasks
    _background_tasks: Vec<JoinHandle<()>>,
}
impl ParallelExecutionEngine {
    /// Create a new parallel execution engine
    pub async fn new(
        config: TestParallelizationConfig,
        timeout_framework: Arc<TestTimeoutFramework>,
    ) -> Result<Self> {
        let scheduler = Arc::new(TestScheduler::new(config.scheduling.clone()).await?);
        let resource_manager =
            Arc::new(ResourceManager::new(config.resource_management.clone()).await?);
        let load_balancer = Arc::new(
            LoadBalancer::new(LoadBalancingConfig {
                strategy: LoadBalancingStrategy::RoundRobin,
                rebalancing: RebalancingConfig {
                    enabled: true,
                    interval: std::time::Duration::from_secs(30),
                    imbalance_threshold: 0.8,
                    aggressiveness: 0.5,
                    work_stealing: WorkStealingConfig {
                        enabled: true,
                        steal_threshold: 0.7,
                        max_steals_per_interval: 10,
                        steal_timeout: std::time::Duration::from_millis(100),
                    },
                },
                worker_config: WorkerConfig {
                    initial_worker_count: 4,
                    min_workers: 1,
                    max_workers: 16,
                    scaling: WorkerScalingConfig {
                        enabled: true,
                        scale_up_threshold: 0.8,
                        scale_down_threshold: 0.3,
                        cooldown_period: std::time::Duration::from_secs(60),
                        scaling_factor: 1.5,
                    },
                    specialization: WorkerSpecializationConfig {
                        enabled: false,
                        by_category: false,
                        by_resource: false,
                        by_performance: false,
                    },
                },
                thresholds: LoadBalancingThresholds {
                    cpu_threshold: 0.8,
                    memory_threshold: 0.8,
                    queue_threshold: 100,
                    response_time_threshold: std::time::Duration::from_millis(100),
                    error_rate_threshold: 0.05,
                },
            })
            .await?,
        );
        let monitor_config = MonitoringConfig {
            monitoring_interval: std::time::Duration::from_secs(1),
            performance_tracking: PerformanceTrackingConfig {
                detailed_tracking: true,
                collection_interval: std::time::Duration::from_secs(10),
                retention_period: std::time::Duration::from_secs(3600),
                analysis_interval: std::time::Duration::from_secs(60),
                regression_detection: true,
            },
            health_checks: HealthCheckConfig {
                interval: std::time::Duration::from_secs(30),
                timeout: std::time::Duration::from_secs(10),
                failure_threshold: 3,
                recovery_interval: std::time::Duration::from_secs(60),
                deep_checks: false,
            },
            alerts: AlertConfig {
                enabled: true,
                cooldown_period: std::time::Duration::from_secs(60),
                thresholds: AlertThresholds {
                    high_error_rate: 0.1,
                    high_latency: std::time::Duration::from_secs(5),
                    resource_exhaustion: 0.9,
                    queue_backup: 100,
                    worker_failure: 3,
                },
                destinations: vec![AlertDestination {
                    destination_type: AlertDestinationType::Log,
                    config: std::collections::HashMap::new(),
                    alert_levels: vec![AlertLevel::Error, AlertLevel::Warning],
                }],
            },
        };
        let execution_monitor = Arc::new(ExecutionMonitor::new(monitor_config).await?);
        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            timeout_framework,
            scheduler,
            resource_manager,
            _load_balancer: load_balancer,
            _execution_monitor: execution_monitor,
            active_sessions: Arc::new(Mutex::new(HashMap::new())),
            _execution_queue: Arc::new(Mutex::new(ExecutionQueue::new())),
            engine_stats: Arc::new(EngineStatistics::new()),
            shutdown: Arc::new(AtomicBool::new(false)),
            test_runner: None,
            in_flight: Arc::new(AtomicUsize::new(0)),
            _background_tasks: Vec::new(),
        })
    }

    /// Install the source of runnable test bodies.
    ///
    /// Until this is called the engine has no executor, and
    /// [`Self::execute_parallel`] refuses to run scheduled tests rather than
    /// fabricating results for them.
    pub fn set_test_runner(&mut self, runner: Arc<dyn TestRunner>) {
        self.test_runner = Some(runner);
    }

    /// Builder form of [`Self::set_test_runner`].
    #[must_use]
    pub fn with_test_runner(mut self, runner: Arc<dyn TestRunner>) -> Self {
        self.test_runner = Some(runner);
        self
    }

    /// Whether a test runner has been installed.
    pub fn has_test_runner(&self) -> bool {
        self.test_runner.is_some()
    }

    /// Ask the execution loop to stop after the tests already in flight finish.
    pub fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }

    /// Statistics accumulated across this engine's executions.
    pub fn statistics(&self) -> &Arc<EngineStatistics> {
        &self.engine_stats
    }

    /// The scheduler this engine queues tests through.
    pub fn scheduler(&self) -> &Arc<TestScheduler> {
        &self.scheduler
    }

    /// The resource manager backing this engine's capacity accounting.
    pub fn resource_manager(&self) -> &Arc<ResourceManager> {
        &self.resource_manager
    }
    /// Execute the analysed tests in parallel, honouring priority order and
    /// the engine's real resource capacity.
    ///
    /// Returns one [`ParallelExecutionOutcome`] per test that actually ran.
    ///
    /// # Errors
    ///
    /// Returns an error when tests are scheduled but no [`TestRunner`] has been
    /// installed (see [`Self::set_test_runner`]) -- the engine cannot invoke a
    /// test from its name alone, and says so instead of returning invented
    /// results -- or when a scheduled test's requirements exceed the engine's
    /// total capacity and could therefore never be granted.
    pub async fn execute_parallel(
        &mut self,
        analysis: TestIndependenceAnalysis,
    ) -> Result<Vec<ParallelExecutionOutcome>> {
        info!(
            "Starting parallel execution of {} tests",
            analysis.tests.len()
        );
        if !analysis.tests.is_empty() && self.test_runner.is_none() {
            return Err(anyhow::anyhow!(
                "ParallelExecutionEngine has no TestRunner installed, so it cannot execute the \
                 {} scheduled test(s); call set_test_runner() with a runner that can supply \
                 their bodies",
                analysis.tests.len()
            ));
        }
        self.start_background_tasks().await?;
        let session_id = self.create_execution_session(&analysis).await?;
        self.schedule_tests(&analysis).await?;
        let results = self.execute_scheduled_tests(&session_id).await?;
        self.cleanup_execution_session(&session_id).await?;
        self.stop_background_tasks().await?;
        info!("Parallel execution completed. {} results", results.len());
        Ok(results)
    }
    /// Create an execution session
    async fn create_execution_session(
        &self,
        analysis: &TestIndependenceAnalysis,
    ) -> Result<String> {
        let session_id = uuid::Uuid::new_v4().to_string();
        let session = ExecutionSession::new(session_id.clone(), analysis.clone());
        {
            let mut sessions = self.active_sessions.lock();
            sessions.insert(session_id.clone(), session);
        }
        debug!("Created execution session: {}", session_id);
        Ok(session_id)
    }
    /// Schedule tests for execution
    async fn schedule_tests(&self, analysis: &TestIndependenceAnalysis) -> Result<()> {
        for test_metadata in &analysis.tests {
            let scheduled_test =
                self.create_scheduled_test(test_metadata, &analysis.dependencies).await?;
            self.scheduler.schedule_test(scheduled_test).await?;
        }
        Ok(())
    }
    /// Create a scheduled test from metadata
    async fn create_scheduled_test(
        &self,
        metadata: &TestParallelizationMetadata,
        dependencies: &[TestDependency],
    ) -> Result<ScheduledTest> {
        let priority = self.calculate_test_priority(metadata, dependencies).await?;
        let resource_requirements = self.calculate_resource_requirements(metadata).await?;
        let constraints = self.extract_scheduling_constraints(metadata, dependencies).await?;
        Ok(ScheduledTest {
            metadata: metadata.clone(),
            priority,
            scheduled_at: Utc::now(),
            estimated_start: None,
            resource_requirements,
            constraints,
            retry_count: 0,
            scheduling_metadata: HashMap::new(),
        })
    }
    /// Calculate test priority for scheduling
    async fn calculate_test_priority(
        &self,
        metadata: &TestParallelizationMetadata,
        _dependencies: &[TestDependency],
    ) -> Result<f32> {
        let base_priority = metadata.priority;
        let category_adjustment = match metadata.base_context.category {
            crate::test_timeout_optimization::TestCategory::Unit => 1.0,
            crate::test_timeout_optimization::TestCategory::Integration => 0.8,
            crate::test_timeout_optimization::TestCategory::Property => 0.7,
            crate::test_timeout_optimization::TestCategory::Stress => 0.4,
            crate::test_timeout_optimization::TestCategory::Chaos => 0.5,
            _ => 0.6,
        };
        Ok(base_priority * category_adjustment)
    }
    /// Calculate resource requirements for a test
    async fn calculate_resource_requirements(
        &self,
        metadata: &TestParallelizationMetadata,
    ) -> Result<ResourceRequirement> {
        Ok(ResourceRequirement {
            resource_type: "mixed".to_string(),
            min_amount: 1.0,
            cpu_cores: metadata.resource_usage.cpu_cores,
            memory_mb: metadata.resource_usage.memory_mb,
            gpu_devices: metadata.resource_usage.gpu_devices.clone(),
            network_ports: metadata.resource_usage.network_ports.len(),
            temp_directories: metadata.resource_usage.temp_directories.len(),
            database_connections: metadata.resource_usage.database_connections,
            custom_resources: HashMap::new(),
        })
    }
    /// Extract scheduling constraints from metadata and dependencies
    async fn extract_scheduling_constraints(
        &self,
        _metadata: &TestParallelizationMetadata,
        dependencies: &[TestDependency],
    ) -> Result<Vec<SchedulingConstraint>> {
        let mut constraints = Vec::new();
        for dependency in dependencies {
            match dependency.dependency_type {
                DependencyType::Hard | DependencyType::Setup => {
                    constraints.push(SchedulingConstraint {
                        constraint_type: SchedulingConstraintType::Dependency,
                        value: dependency.dependency_test.clone(),
                        priority: dependency.strength,
                        deadline: None,
                    });
                },
                DependencyType::Conflict => {
                    constraints.push(SchedulingConstraint {
                        constraint_type: SchedulingConstraintType::ResourceAvailability,
                        value: format!("avoid_concurrent:{}", dependency.dependency_test),
                        priority: dependency.strength,
                        deadline: None,
                    });
                },
                _ => {},
            }
        }
        Ok(constraints)
    }
    /// Run every scheduled test, respecting priority order, the configured
    /// concurrency limit and real resource capacity.
    ///
    /// 0.2.1: the two helpers this loop used to call --
    /// `start_test_execution` and `process_execution_result` -- did not execute
    /// anything. The first slept 100 ms and returned a `TestExecutionResult`
    /// whose CPU, memory, task and checkpoint counts were constants in the
    /// source; the second built a whole `TestParallelizationResult` out of
    /// further constants (efficiency 0.85, speedup 1.0/2.0, CPU 75/85, a
    /// "CPU bound operations" bottleneck, "Consider CPU optimization"). Both are
    /// deleted. Tests are now run through
    /// [`TestTimeoutFramework::execute_test`], which times them for real and
    /// derives their outcome from what the body returned.
    async fn execute_scheduled_tests(
        &self,
        session_id: &str,
    ) -> Result<Vec<ParallelExecutionOutcome>> {
        let mut results = Vec::new();
        let mut active_executions: JoinSet<Result<ParallelExecutionOutcome>> = JoinSet::new();
        // Every reservation this session handed out, so a task that panics or
        // is cancelled cannot leak its capacity: the sweep below reclaims
        // whatever is still held when the loop finishes.
        let mut spawned_allocations: Vec<String> = Vec::new();
        loop {
            if self.should_stop_execution() {
                break;
            }
            while self.can_start_new_test(active_executions.len()) {
                if let Some(scheduled_test) = self.scheduler.get_next_test().await? {
                    let test_id = scheduled_test.metadata.resource_usage.test_id.clone();
                    match self.resource_manager.feasibility(&scheduled_test.resource_requirements) {
                        AllocationFeasibility::Grantable => {
                            let allocation = self
                                .resource_manager
                                .allocate_resources(&scheduled_test.resource_requirements, &test_id)
                                .await?;
                            let allocation_id = allocation.resource_id.clone();
                            match self.build_execution_task(scheduled_test, allocation) {
                                Ok(task) => {
                                    spawned_allocations.push(allocation_id);
                                    active_executions.spawn(task);
                                },
                                Err(e) => {
                                    // The runner does not know this test. Hand
                                    // the reservation straight back rather than
                                    // leaking it, and surface the reason.
                                    self.resource_manager.release_allocation(&allocation_id).await;
                                    return Err(e);
                                },
                            }
                        },
                        AllocationFeasibility::WaitForCapacity(reason) => {
                            // Live allocations hold the capacity this test needs;
                            // requeue and retry once one of them is released.
                            debug!("Requeueing test {test_id}: {reason}");
                            self.scheduler.requeue_test(scheduled_test).await?;
                            break;
                        },
                        AllocationFeasibility::ExceedsCapacity(reason) => {
                            // No amount of waiting can satisfy this request, so
                            // report it instead of spinning on the queue forever.
                            return Err(anyhow::anyhow!(
                                "test {test_id} can never be scheduled by this engine: {reason}"
                            ));
                        },
                    }
                } else {
                    break;
                }
            }
            if let Ok(Some(joined)) =
                tokio::time::timeout(Duration::from_millis(100), active_executions.join_next())
                    .await
            {
                self.absorb_completion(joined, &mut results).await;
            }
            if active_executions.is_empty() && self.scheduler.is_queue_empty().await {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        while let Some(joined) = active_executions.join_next().await {
            self.absorb_completion(joined, &mut results).await;
        }

        // Reclaim anything a panicked or cancelled task left reserved. A task
        // that never returned an outcome still holds its slot, and without this
        // the manager would refuse forever once enough tasks had died.
        for allocation_id in spawned_allocations {
            if let Some(leaked) = self.resource_manager.release_allocation(&allocation_id).await {
                warn!(
                    "Reclaimed allocation {} left held by a test that produced no outcome",
                    leaked.resource_id
                );
            }
        }

        debug!(
            "Session {session_id} produced {} execution outcome(s)",
            results.len()
        );
        Ok(results)
    }

    /// Fold one joined task into `results`, releasing its reservation.
    ///
    /// A task that panicked or was cancelled carries no outcome, so nothing is
    /// pushed for it -- the engine reports fewer outcomes than tests rather
    /// than substituting a synthetic one.
    async fn absorb_completion(
        &self,
        joined: std::result::Result<Result<ParallelExecutionOutcome>, tokio::task::JoinError>,
        results: &mut Vec<ParallelExecutionOutcome>,
    ) {
        match joined {
            Ok(Ok(mut outcome)) => {
                // Release by allocation id, not by test name: the two are
                // filed under different fields and need not agree.
                if let Some(allocation) =
                    self.resource_manager.release_allocation(&outcome.allocation.resource_id).await
                {
                    outcome.allocation = allocation;
                }
                self.engine_stats
                    .record(outcome.result.execution_time, outcome.observed_concurrency);
                results.push(outcome);
            },
            Ok(Err(e)) => {
                error!("Test execution returned an error: {e:#}");
            },
            Err(e) => {
                // A panicked/cancelled task carries no test identity, so its
                // reservation can only be reclaimed by the session teardown.
                error!("Test execution task failed to join: {e:?}");
            },
        }
    }

    /// Whether the caller has asked the loop to wind down.
    fn should_stop_execution(&self) -> bool {
        self.shutdown.load(Ordering::SeqCst)
    }

    /// Whether another test may start, given how many are already in flight.
    ///
    /// 0.2.1: this used to return `true` unconditionally, so
    /// `max_concurrent_tests` was configured but never enforced.
    fn can_start_new_test(&self, in_flight: usize) -> bool {
        let limit = self.config.read().max_concurrent_tests;
        limit == 0 || in_flight < limit
    }

    /// Build the future that really runs one scheduled test.
    ///
    /// # Errors
    ///
    /// Returns an error when no runner is installed, or when the installed
    /// runner does not know this test -- either way the engine declines to
    /// produce an outcome for it.
    fn build_execution_task(
        &self,
        test: ScheduledTest,
        allocation: ResourceAllocation,
    ) -> Result<impl std::future::Future<Output = Result<ParallelExecutionOutcome>> + Send + 'static>
    {
        let context = test.metadata.base_context.clone();
        let test_name = context.test_name.clone();
        let runner = self.test_runner.clone().ok_or_else(|| {
            anyhow::anyhow!("no TestRunner installed; cannot execute test {test_name}")
        })?;
        let body = runner.test_body(&test_name).ok_or_else(|| {
            anyhow::anyhow!(
                "the installed TestRunner supplies no body for test {test_name}; it cannot be run"
            )
        })?;
        let framework = Arc::clone(&self.timeout_framework);
        let in_flight = Arc::clone(&self.in_flight);
        let scheduled_at = test.scheduled_at;
        let strategy = self.config.read().scheduling.strategy.clone();
        Ok(async move {
            let observed_concurrency = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            let started_at = Utc::now();
            let result = framework.execute_test(context, body).await;
            let completed_at = Utc::now();
            in_flight.fetch_sub(1, Ordering::SeqCst);
            let result = result?;
            let queue_wait = (started_at - scheduled_at).to_std().unwrap_or(Duration::ZERO);
            Ok(ParallelExecutionOutcome {
                result,
                allocation,
                scheduled_at,
                started_at,
                completed_at,
                queue_wait,
                observed_concurrency,
                strategy,
            })
        })
    }

    async fn cleanup_execution_session(&self, _session_id: &str) -> Result<()> {
        Ok(())
    }
    async fn start_background_tasks(&mut self) -> Result<()> {
        Ok(())
    }
    async fn stop_background_tasks(&mut self) -> Result<()> {
        Ok(())
    }
}
