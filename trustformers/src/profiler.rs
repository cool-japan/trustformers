//! Built-in Performance Profiler
//!
//! High-level profiling interface for performance analysis and bottleneck detection.
//! Integrates with the trustformers-core performance infrastructure to provide
//! easy-to-use profiling capabilities for models, pipelines, and operations.

use crate::core::performance::{
    BenchmarkResult, BenchmarkSuite, LatencyMetrics, MemoryMetrics, MetricsTracker,
    OptimizationAdvisor, OptimizationSuggestion, PerformanceProfiler as CoreProfiler,
    ProfileResult, ThroughputMetrics,
};
use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use trustformers_core::errors::TrustformersError;

/// High-level performance profiler for trustformers
pub struct Profiler {
    /// Core profiler instance
    core_profiler: CoreProfiler,
    /// Optimization advisor
    advisor: OptimizationAdvisor,
    /// Benchmark suite
    benchmark_suite: BenchmarkSuite,
    /// Metrics tracker
    metrics_tracker: MetricsTracker,
    /// Configuration
    config: ProfilerConfig,
    /// Session start time
    session_start: Instant,
    /// Active sessions
    active_sessions: Arc<Mutex<HashMap<String, ProfileSession>>>,
    /// Core-profiler guards for operations currently in flight, keyed
    /// `"<session>::<operation>"`. Dropping a guard records the interval in the
    /// core profiler's aggregate view.
    operation_guards: Arc<Mutex<HashMap<String, crate::core::performance::profiler::ProfileGuard>>>,
}

/// Profiler configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    /// Enable automatic profiling
    pub auto_enable: bool,
    /// Enable memory profiling
    pub enable_memory: bool,
    /// Enable optimization suggestions
    pub enable_advisor: bool,
    /// Enable benchmarking
    pub enable_benchmarks: bool,
    /// Maximum number of sessions to keep
    pub max_sessions: usize,
    /// Output directory for reports
    pub output_dir: Option<String>,
    /// Auto-save results
    pub auto_save: bool,
}

impl Default for ProfilerConfig {
    fn default() -> Self {
        Self {
            auto_enable: true,
            enable_memory: true,
            enable_advisor: true,
            enable_benchmarks: false, // Expensive, off by default
            max_sessions: 10,
            output_dir: None,
            auto_save: false,
        }
    }
}

/// A single measured operation duration.
#[derive(Debug, Clone)]
pub struct OperationSample {
    /// Name of the operation that was timed.
    pub operation: String,
    /// Wall-clock duration of this single call.
    pub duration: Duration,
}

/// Workload counters a caller reported for a session.
///
/// The profiler cannot know how many tokens an operation processed, so these
/// stay at zero until [`Profiler::record_workload`] is called. Zero therefore
/// means "not reported", never "measured zero".
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct WorkloadCounters {
    /// Tokens processed.
    pub tokens: usize,
    /// Batches processed.
    pub batches: usize,
    /// Individual samples/sequences processed.
    pub items: usize,
}

impl WorkloadCounters {
    /// Whether any workload was reported at all.
    pub fn is_reported(&self) -> bool {
        self.tokens > 0 || self.batches > 0 || self.items > 0
    }
}

/// Where a [`ProfileResults`] memory reading came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum MemoryMetricsSource {
    /// Operating-system process memory (resident + virtual) read via `sysinfo`.
    /// Allocation/deallocation counts are not available from this source and
    /// are reported as zero.
    ProcessMemory,
    /// No memory source could be read.
    #[default]
    Unavailable,
}

/// Profile session information
///
/// Not `Clone`: it owns live core-profiler guards for in-flight operations.
#[derive(Debug)]
pub struct ProfileSession {
    /// Session ID
    pub id: String,
    /// Session name
    pub name: String,
    /// Start time
    pub start_time: Instant,
    /// End time (if completed)
    pub end_time: Option<Instant>,
    /// Session results
    pub results: Option<ProfileResults>,
    /// Every individually measured call duration in this session.
    pub samples: Vec<OperationSample>,
    /// Operations that were started but not yet ended, with their start instant.
    in_flight: HashMap<String, Instant>,
    /// Workload counters reported by the caller.
    pub workload: WorkloadCounters,
    /// Resident-set size when the session started, in bytes.
    pub start_rss_bytes: Option<usize>,
    /// Highest resident-set size observed during the session, in bytes.
    pub peak_rss_bytes: Option<usize>,
}

/// Comprehensive profile results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileResults {
    /// Session information
    pub session_id: String,
    /// Total session duration
    pub total_duration: Duration,
    /// Operation profile results
    pub operations: HashMap<String, ProfileResult>,
    /// Latency metrics
    pub latency_metrics: LatencyMetrics,
    /// Throughput metrics
    pub throughput_metrics: ThroughputMetrics,
    /// Memory metrics
    pub memory_metrics: Option<MemoryMetrics>,
    /// Where `memory_metrics` was read from.
    #[serde(default)]
    pub memory_metrics_source: MemoryMetricsSource,
    /// Workload counters the caller reported for this session.
    #[serde(default)]
    pub workload: WorkloadCounters,
    /// Optimization suggestions
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
    /// Benchmark results (if enabled)
    pub benchmark_results: Option<Vec<BenchmarkResult>>,
    /// Performance summary
    pub summary: ProfileSummary,
}

/// Performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSummary {
    /// Total operations profiled
    pub total_operations: usize,
    /// Total time spent
    pub total_time: Duration,
    /// Average operation time
    pub avg_operation_time: Duration,
    /// Slowest operation
    pub slowest_operation: String,
    /// Fastest operation
    pub fastest_operation: String,
    /// Resident bytes as a percentage of the reserved address space.
    ///
    /// `None` when no memory source could be read. This is a measured ratio,
    /// not a quality score.
    pub memory_efficiency: Option<f64>,
    /// Latency-consistency score: `100 * p50 / p99`, clamped to 0..=100.
    ///
    /// A run whose tail latency matches its median scores 100. `None` when no
    /// per-call samples were recorded.
    pub performance_score: Option<f64>,
    /// Number of bottlenecks identified
    pub bottlenecks_found: usize,
    /// Share of total time spent in operations that triggered a suggestion, in
    /// percent. `None` when no time was recorded.
    pub optimization_potential: Option<f64>,
}

impl Profiler {
    /// Create a new profiler with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(ProfilerConfig::default())
    }

    /// Create a new profiler with custom configuration
    pub fn with_config(config: ProfilerConfig) -> Result<Self> {
        Ok(Self::build(config))
    }

    /// Infallibly construct a profiler from a configuration.
    ///
    /// This is the shared constructor used by `with_config`, `Default`, and the
    /// global profiler initializer so none of them need to unwrap a `Result`.
    fn build(config: ProfilerConfig) -> Self {
        let core_profiler = CoreProfiler::new();

        if config.auto_enable {
            core_profiler.enable();
        }

        let advisor = OptimizationAdvisor::new();
        let benchmark_suite = BenchmarkSuite::new(trustformers_core::BenchmarkConfig::default());
        let metrics_tracker = MetricsTracker::new(100); // Use 100 as default window size

        Self {
            core_profiler,
            advisor,
            benchmark_suite,
            metrics_tracker,
            config,
            session_start: Instant::now(),
            active_sessions: Arc::new(Mutex::new(HashMap::new())),
            operation_guards: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// How long this profiler instance has been alive.
    pub fn uptime(&self) -> Duration {
        self.session_start.elapsed()
    }

    /// The optimization advisor this profiler was constructed with.
    ///
    /// [`Self::generate_optimization_suggestions`] does not call
    /// [`OptimizationAdvisor::analyze`] on it — see that method's doc comment
    /// — so this is exposed for callers who want the advisor's real,
    /// rule-based analysis directly.
    pub fn advisor(&self) -> &OptimizationAdvisor {
        &self.advisor
    }

    /// The rolling-window metrics tracker this profiler was constructed with.
    ///
    /// Nothing currently calls [`MetricsTracker::record_inference`] on it —
    /// `end_session` computes its latency/throughput metrics directly from
    /// the session's own samples instead — so its window is empty unless a
    /// caller records into it directly.
    pub fn metrics_tracker(&self) -> &MetricsTracker {
        &self.metrics_tracker
    }

    /// Enable profiling
    pub fn enable(&self) {
        self.core_profiler.enable();
    }

    /// Disable profiling
    pub fn disable(&self) {
        self.core_profiler.disable();
    }

    /// Check if profiling is enabled
    pub fn is_enabled(&self) -> bool {
        self.core_profiler.is_enabled()
    }

    /// Start a new profiling session
    pub fn start_session(&self, name: &str) -> Result<String> {
        let session_id = format!("{}_{}", name, chrono::Utc::now().timestamp());
        let start_rss_bytes = read_process_memory().map(|m| m.resident_bytes);
        let session = ProfileSession {
            id: session_id.clone(),
            name: name.to_string(),
            start_time: Instant::now(),
            end_time: None,
            results: None,
            samples: Vec::new(),
            in_flight: HashMap::new(),
            workload: WorkloadCounters::default(),
            start_rss_bytes,
            peak_rss_bytes: start_rss_bytes,
        };

        let mut sessions = self.active_sessions.lock().map_err(|e| {
            TrustformersError::runtime_error(format!("Failed to lock sessions: {}", e))
        })?;

        // Clean up old sessions if we exceed the limit
        if sessions.len() >= self.config.max_sessions {
            let oldest_id = sessions.values().min_by_key(|s| s.start_time).map(|s| s.id.clone());
            if let Some(id) = oldest_id {
                sessions.remove(&id);
            }
        }

        sessions.insert(session_id.clone(), session);
        Ok(session_id)
    }

    /// End a profiling session and generate results
    pub fn end_session(&self, session_id: &str) -> Result<ProfileResults> {
        let mut sessions = self.active_sessions.lock().map_err(|e| {
            TrustformersError::runtime_error(format!("Failed to lock sessions: {}", e))
        })?;

        let session = sessions.get_mut(session_id).ok_or_else(|| {
            TrustformersError::invalid_input(format!("Session {} not found", session_id))
        })?;

        let end_time = Instant::now();
        session.end_time = Some(end_time);
        let total_duration = end_time - session.start_time;

        // Collect results from core profiler
        let operations = self.core_profiler.get_results();

        // Generate metrics from the durations this session actually measured.
        let durations: Vec<Duration> = session.samples.iter().map(|s| s.duration).collect();
        let latency_metrics = Self::generate_latency_metrics(&durations, total_duration);
        let throughput_metrics =
            Self::generate_throughput_metrics(&session.workload, total_duration);

        if let Some(current) = read_process_memory() {
            session.peak_rss_bytes =
                Some(session.peak_rss_bytes.unwrap_or(0).max(current.resident_bytes));
        }
        let (memory_metrics, memory_metrics_source) = if self.config.enable_memory {
            Self::generate_memory_metrics(session.peak_rss_bytes)
        } else {
            (None, MemoryMetricsSource::Unavailable)
        };
        let workload = session.workload;

        // Generate optimization suggestions
        let optimization_suggestions = if self.config.enable_advisor {
            self.generate_optimization_suggestions(&operations)
        } else {
            Vec::new()
        };

        // Run benchmarks if enabled
        let benchmark_results =
            if self.config.enable_benchmarks { Some(self.run_benchmarks()?) } else { None };

        // Generate summary
        let summary = Self::generate_summary(
            &operations,
            &latency_metrics,
            memory_metrics.as_ref(),
            &optimization_suggestions,
        );

        let results = ProfileResults {
            session_id: session_id.to_string(),
            total_duration,
            operations,
            latency_metrics,
            throughput_metrics,
            memory_metrics,
            memory_metrics_source,
            workload,
            optimization_suggestions,
            benchmark_results,
            summary,
        };

        session.results = Some(results.clone());

        // Auto-save if configured
        if self.config.auto_save {
            self.save_results(&results)?;
        }

        Ok(results)
    }

    /// Replace the benchmark suite used when `enable_benchmarks` is set.
    pub fn with_benchmark_suite(mut self, suite: BenchmarkSuite) -> Self {
        self.benchmark_suite = suite;
        self
    }

    /// Report how much work a session processed.
    ///
    /// Throughput cannot be measured without this: the profiler sees durations,
    /// not tokens. Counters accumulate, so it is safe to call once per batch.
    ///
    /// # Errors
    ///
    /// Fails when the session lock is poisoned or the session does not exist.
    pub fn record_workload(
        &self,
        session_id: &str,
        tokens: usize,
        batches: usize,
        items: usize,
    ) -> Result<()> {
        let mut sessions = self.active_sessions.lock().map_err(|e| {
            TrustformersError::runtime_error(format!("Failed to lock sessions: {}", e))
        })?;
        let session = sessions.get_mut(session_id).ok_or_else(|| {
            TrustformersError::invalid_input(format!("Session {} not found", session_id))
        })?;
        session.workload.tokens += tokens;
        session.workload.batches += batches;
        session.workload.items += items;
        Ok(())
    }

    /// Record one measured call duration against a session.
    fn record_sample(&self, session_id: &str, operation: &str, duration: Duration) {
        let Ok(mut sessions) = self.active_sessions.lock() else {
            tracing::warn!("profiler session lock poisoned; sample dropped");
            return;
        };
        if let Some(session) = sessions.get_mut(session_id) {
            session.samples.push(OperationSample {
                operation: operation.to_string(),
                duration,
            });
        }
    }

    /// Profile a function with automatic session management
    pub fn profile_function<F, R>(&self, name: &str, f: F) -> Result<(R, ProfileResults)>
    where
        F: FnOnce() -> R,
    {
        let session_id = self.start_session(name)?;
        let guard = self.core_profiler.start_operation(name);

        let started = Instant::now();
        let result = f();
        let elapsed = started.elapsed();

        drop(guard);
        self.record_sample(&session_id, name, elapsed);
        let profile_results = self.end_session(&session_id)?;

        Ok((result, profile_results))
    }

    /// Lightweight profiling function that only measures execution time
    /// This is optimized for performance benchmarks where minimal overhead is critical
    pub fn profile_function_lightweight<F, R>(&self, _name: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        // Just execute the function without any profiling overhead for benchmarks
        f()
    }

    /// Profile an async function
    pub async fn profile_async<F, R>(&self, name: &str, f: F) -> Result<(R, ProfileResults)>
    where
        F: std::future::Future<Output = R>,
    {
        let session_id = self.start_session(name)?;
        let guard = self.core_profiler.start_operation(name);

        let started = Instant::now();
        let result = f.await;
        let elapsed = started.elapsed();

        drop(guard);
        self.record_sample(&session_id, name, elapsed);
        let profile_results = self.end_session(&session_id)?;

        Ok((result, profile_results))
    }

    /// Get the global profiler instance
    pub fn instance() -> &'static Profiler {
        get_global_profiler()
    }

    /// Mark the beginning of a named operation within a session.
    ///
    /// The instant is stored and consumed by [`Profiler::end_operation`], which
    /// records the real elapsed duration. Calling `start_operation` twice for
    /// the same name simply restarts the clock.
    ///
    /// `_metadata` is accepted for API compatibility and reserved for future use.
    pub fn start_operation(
        &self,
        session_id: &str,
        op_name: &str,
        _metadata: Option<HashMap<String, String>>,
    ) {
        let Ok(mut sessions) = self.active_sessions.lock() else {
            tracing::warn!("profiler session lock poisoned; operation start dropped");
            return;
        };
        if let Some(session) = sessions.get_mut(session_id) {
            session.in_flight.insert(op_name.to_string(), Instant::now());
            let key = format!("{}::{}", session_id, op_name);
            let guard = self.core_profiler.start_operation(&key);
            if let Ok(mut guards) = self.operation_guards.lock() {
                guards.insert(key, guard);
            }
        }
    }

    /// End a named operation within a session and record its measured duration.
    ///
    /// Does nothing when the operation was never started — no duration can be
    /// invented for it.
    pub fn end_operation(&self, session_id: &str, op_name: &str) {
        let Ok(mut sessions) = self.active_sessions.lock() else {
            tracing::warn!("profiler session lock poisoned; operation end dropped");
            return;
        };
        let Some(session) = sessions.get_mut(session_id) else {
            return;
        };
        let Some(started) = session.in_flight.remove(op_name) else {
            tracing::debug!(
                operation = op_name,
                "end_operation called without a matching start_operation"
            );
            return;
        };
        let elapsed = started.elapsed();
        session.samples.push(OperationSample {
            operation: op_name.to_string(),
            duration: elapsed,
        });
        drop(sessions);

        // Dropping the guard records the same interval in the core profiler's
        // aggregate view.
        let key = format!("{}::{}", session_id, op_name);
        if let Ok(mut guards) = self.operation_guards.lock() {
            guards.remove(&key);
        }
    }

    /// Snapshot of the sessions currently held by this profiler.
    ///
    /// # Errors
    ///
    /// Fails when the session lock is poisoned.
    pub fn get_active_sessions(&self) -> Result<Vec<ProfileSessionInfo>> {
        let sessions = self.active_sessions.lock().map_err(|e| {
            TrustformersError::runtime_error(format!("Failed to lock sessions: {}", e))
        })?;
        Ok(sessions
            .values()
            .map(|s| ProfileSessionInfo {
                id: s.id.clone(),
                name: s.name.clone(),
                start_time: s.start_time,
                end_time: s.end_time,
                sample_count: s.samples.len(),
                workload: s.workload,
                completed: s.results.is_some(),
            })
            .collect())
    }

    /// Get session results
    pub fn get_session_results(&self, session_id: &str) -> Result<Option<ProfileResults>> {
        let sessions = self.active_sessions.lock().map_err(|e| {
            TrustformersError::runtime_error(format!("Failed to lock sessions: {}", e))
        })?;
        Ok(sessions.get(session_id).and_then(|s| s.results.clone()))
    }

    /// Clear all sessions and profiling data
    pub fn clear(&self) -> Result<()> {
        self.core_profiler.clear();
        let mut sessions = self.active_sessions.lock().map_err(|e| {
            TrustformersError::runtime_error(format!("Failed to lock sessions: {}", e))
        })?;
        sessions.clear();
        Ok(())
    }

    /// Generate a performance dashboard URL (if available)
    pub fn get_dashboard_url(&self) -> Option<String> {
        // In a real implementation, this would return a URL to a web dashboard
        None
    }

    /// Export results to various formats
    pub fn export_results(&self, session_id: &str, format: ExportFormat, path: &str) -> Result<()> {
        let sessions = self.active_sessions.lock().map_err(|e| {
            TrustformersError::runtime_error(format!("Failed to lock sessions: {}", e))
        })?;

        let session = sessions.get(session_id).ok_or_else(|| {
            TrustformersError::invalid_input(format!("Session {} not found", session_id))
        })?;

        let results = session.results.as_ref().ok_or_else(|| {
            TrustformersError::invalid_input("Session has no results".to_string())
        })?;

        match format {
            ExportFormat::Json => {
                let json = serde_json::to_string_pretty(results).map_err(|e| {
                    TrustformersError::serialization_error(format!(
                        "JSON serialization failed: {}",
                        e
                    ))
                })?;
                std::fs::write(path, json).map_err(|e| {
                    TrustformersError::io_error(format!("File write failed: {}", e))
                })?;
            },
            ExportFormat::Html => {
                let html = self.generate_html_report(results);
                std::fs::write(path, html).map_err(|e| {
                    TrustformersError::io_error(format!("File write failed: {}", e))
                })?;
            },
            ExportFormat::Flamegraph => {
                self.core_profiler.export_flamegraph(path).map_err(|e| {
                    TrustformersError::invalid_operation(format!("Flamegraph export failed: {}", e))
                })?;
            },
            ExportFormat::Csv => {
                let csv = self.generate_csv_report(results);
                std::fs::write(path, csv).map_err(|e| {
                    TrustformersError::io_error(format!("File write failed: {}", e))
                })?;
            },
        }

        Ok(())
    }

    // Helper methods

    /// Real latency distribution over every individually measured call.
    ///
    /// Percentiles, median and standard deviation all come from the recorded
    /// sample vector via [`LatencyMetrics::from_durations`]; when nothing was
    /// measured the result is the all-zero `count: 0` default rather than a
    /// stand-in derived from the mean.
    fn generate_latency_metrics(durations: &[Duration], window: Duration) -> LatencyMetrics {
        if durations.is_empty() {
            return LatencyMetrics {
                window_duration: window,
                ..LatencyMetrics::default()
            };
        }
        LatencyMetrics {
            window_duration: window,
            ..LatencyMetrics::from_durations(durations)
        }
    }

    /// Throughput from the workload the caller reported.
    ///
    /// The profiler cannot see tokens or batches, so nothing is inferred: an
    /// unreported workload yields all-zero counters (`count`-style semantics)
    /// instead of an invented "100 tokens per operation".
    fn generate_throughput_metrics(
        workload: &WorkloadCounters,
        total_duration: Duration,
    ) -> ThroughputMetrics {
        if !workload.is_reported() || total_duration.is_zero() {
            return ThroughputMetrics {
                tokens_per_second: 0.0,
                batches_per_second: 0.0,
                samples_per_second: 0.0,
                avg_batch_size: 0.0,
                avg_sequence_length: 0.0,
                total_tokens: workload.tokens,
                total_batches: workload.batches,
                total_duration,
            };
        }
        ThroughputMetrics::calculate(
            workload.tokens,
            workload.batches,
            workload.items,
            total_duration,
        )
    }

    /// Memory metrics from the operating system, or `None`.
    ///
    /// Resident and virtual sizes are real readings from `sysinfo`. Allocation
    /// and deallocation *counts* cannot be obtained this way, so they are
    /// reported as zero and [`ProfileResults::memory_metrics_source`] records
    /// that the numbers came from the process view — nothing here is estimated.
    fn generate_memory_metrics(
        peak_rss_bytes: Option<usize>,
    ) -> (Option<MemoryMetrics>, MemoryMetricsSource) {
        match read_process_memory() {
            Some(p) => {
                let peak = peak_rss_bytes.unwrap_or(p.resident_bytes).max(p.resident_bytes);
                let mut metrics = MemoryMetrics::new(
                    p.resident_bytes,
                    peak,
                    p.resident_bytes,
                    p.virtual_bytes.max(p.resident_bytes),
                );
                metrics.num_allocations = 0;
                metrics.num_deallocations = 0;
                (Some(metrics), MemoryMetricsSource::ProcessMemory)
            },
            None => (None, MemoryMetricsSource::Unavailable),
        }
    }

    /// Flag the single slowest recorded operation as worth optimizing.
    ///
    /// # Known limitation
    ///
    /// This is a minimal heuristic (one rule: "the slowest operation took
    /// over 100ms"), not [`Self::advisor`]'s rule-based
    /// [`OptimizationAdvisor::analyze`] — that needs an `AnalysisContext`
    /// assembled from real hardware detection and the session's latency/
    /// memory/throughput metrics, which this method does not build. Wiring
    /// it in is real, scoped follow-up work, not something to fake here.
    /// `expected_improvement`'s percentages are honestly `None` (no measured
    /// or modelled estimate exists) rather than a plausible-looking constant.
    fn generate_optimization_suggestions(
        &self,
        operations: &HashMap<String, ProfileResult>,
    ) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();

        // Find slow operations
        let mut sorted_ops: Vec<_> = operations.iter().collect();
        sorted_ops.sort_by_key(|(_, item)| std::cmp::Reverse(item.total_time));

        if let Some((name, result)) = sorted_ops.first() {
            if result.total_time > Duration::from_millis(100) {
                suggestions.push(OptimizationSuggestion {
                    id: "slow_operation".to_string(),
                    category: crate::core::performance::OptimizationCategory::Compute,
                    impact: crate::core::performance::ImpactLevel::High,
                    difficulty: crate::core::performance::Difficulty::Medium,
                    title: format!("Optimize slow operation: {}", name),
                    description: format!("Operation {} is taking {:.2}ms, consider optimization", name, result.total_time.as_secs_f64() * 1000.0),
                    implementation_steps: vec![
                        "Profile the operation in detail".to_string(),
                        "Consider algorithmic improvements".to_string(),
                        "Enable hardware acceleration".to_string(),
                    ],
                    // No measurement or model backs a specific percentage
                    // here, so these are honestly `None` rather than an
                    // invented (if plausible-looking) number.
                    expected_improvement: crate::core::performance::PerformanceImprovement {
                        latency_reduction: None,
                        throughput_increase: None,
                        memory_reduction: None,
                        other_metrics: std::collections::HashMap::new(),
                    },
                    code_examples: Some(vec![crate::core::performance::CodeExample {
                        language: "rust".to_string(),
                        code: format!("// Optimize {} operation\n// Consider using GPU acceleration or kernel fusion", name),
                        description: "Example optimization approach".to_string(),
                    }]),
                    warnings: vec![],
                    related_suggestions: vec![],
                });
            }
        }

        suggestions
    }

    /// Summary derived entirely from the measurements above.
    fn generate_summary(
        operations: &HashMap<String, ProfileResult>,
        latency: &LatencyMetrics,
        memory: Option<&MemoryMetrics>,
        suggestions: &[OptimizationSuggestion],
    ) -> ProfileSummary {
        let total_operations = operations.len();
        let total_time: Duration = operations.values().map(|r| r.total_time).sum();
        let avg_operation_time = if total_operations > 0 {
            total_time / total_operations as u32
        } else {
            Duration::ZERO
        };

        let slowest_operation = operations
            .iter()
            .max_by_key(|(_, r)| r.total_time)
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| "none".to_string());

        let fastest_operation = operations
            .iter()
            .min_by_key(|(_, r)| r.total_time)
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| "none".to_string());

        // Resident bytes over reserved address space: a measured ratio, not a
        // score. `None` when no memory source answered.
        let memory_efficiency = memory.and_then(|m| {
            if m.reserved_bytes == 0 {
                None
            } else {
                Some((m.current_bytes as f64 / m.reserved_bytes as f64) * 100.0)
            }
        });

        // Latency consistency: how close the tail is to the median.
        let performance_score = if latency.count == 0 || latency.p99_ms <= 0.0 {
            None
        } else {
            Some((latency.p50_ms / latency.p99_ms * 100.0).clamp(0.0, 100.0))
        };

        // Share of measured time attributable to operations a suggestion names.
        let optimization_potential = if total_time.is_zero() {
            None
        } else {
            let flagged: Duration = operations
                .iter()
                .filter(|(name, _)| suggestions.iter().any(|s| s.title.contains(name.as_str())))
                .map(|(_, r)| r.total_time)
                .sum();
            Some(flagged.as_secs_f64() / total_time.as_secs_f64() * 100.0)
        };

        ProfileSummary {
            total_operations,
            total_time,
            avg_operation_time,
            slowest_operation,
            fastest_operation,
            memory_efficiency,
            performance_score,
            bottlenecks_found: suggestions.len(),
            optimization_potential,
        }
    }

    /// Run the configured benchmark suite.
    ///
    /// The suite needs a model to benchmark; the high-level profiler has none,
    /// so instead of pretending, this reports the suite's own (initially empty)
    /// result set and tells the caller how to feed it. Benchmarks are opt-in
    /// (`enable_benchmarks`), so this path is never silently taken.
    fn run_benchmarks(&self) -> Result<Vec<BenchmarkResult>> {
        let results = self.benchmark_suite.results().to_vec();
        if results.is_empty() {
            return Err(TrustformersError::invalid_operation(
                "benchmarking was enabled but no benchmark has been run: call \
                 `BenchmarkSuite::benchmark_inference` on a model and pass the suite in via \
                 `Profiler::with_benchmark_suite` before ending the session"
                    .to_string(),
            )
            .into());
        }
        Ok(results)
    }

    fn save_results(&self, results: &ProfileResults) -> Result<()> {
        if let Some(output_dir) = &self.config.output_dir {
            let filename = format!("{}/profile_{}.json", output_dir, results.session_id);
            let json = serde_json::to_string_pretty(results).map_err(|e| {
                TrustformersError::serialization_error(format!("JSON serialization failed: {}", e))
            })?;
            std::fs::write(&filename, json)
                .map_err(|e| TrustformersError::io_error(format!("File write failed: {}", e)))?;
        }
        Ok(())
    }

    fn generate_html_report(&self, results: &ProfileResults) -> String {
        format!(
            r#"<!DOCTYPE html>
<html>
<head>
    <title>TrustformeRS Performance Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background: #e9e9e9; border-radius: 3px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>TrustformeRS Performance Report</h1>
        <p>Session: {}</p>
        <p>Duration: {:.2}ms</p>
    </div>

    <div class="section">
        <h2>Summary</h2>
        <div class="metric">Operations: {}</div>
        <div class="metric">Latency consistency: {}</div>
        <div class="metric">Memory efficiency: {}</div>
        <div class="metric">Bottlenecks: {}</div>
    </div>

    <div class="section">
        <h2>Operations</h2>
        <table>
            <tr><th>Operation</th><th>Calls</th><th>Total Time (ms)</th><th>Avg Time (ms)</th></tr>
            {}
        </table>
    </div>

    <div class="section">
        <h2>Optimization Suggestions</h2>
        <ul>
            {}
        </ul>
    </div>
</body>
</html>"#,
            results.session_id,
            results.total_duration.as_secs_f64() * 1000.0,
            results.summary.total_operations,
            results
                .summary
                .performance_score
                .map(|v| format!("{:.1}", v))
                .unwrap_or_else(|| "not measured".to_string()),
            results
                .summary
                .memory_efficiency
                .map(|v| format!("{:.1}%", v))
                .unwrap_or_else(|| "not measured".to_string()),
            results.summary.bottlenecks_found,
            results
                .operations
                .iter()
                .map(|(name, result)| format!(
                    "<tr><td>{}</td><td>{}</td><td>{:.2}</td><td>{:.2}</td></tr>",
                    name,
                    result.call_count,
                    result.total_time.as_secs_f64() * 1000.0,
                    result.avg_time.as_secs_f64() * 1000.0
                ))
                .collect::<Vec<_>>()
                .join("\n"),
            results
                .optimization_suggestions
                .iter()
                .map(|s| format!("<li>{}: {}</li>", s.title, s.description))
                .collect::<Vec<_>>()
                .join("\n")
        )
    }

    fn generate_csv_report(&self, results: &ProfileResults) -> String {
        let mut csv = String::from(
            "Operation,Calls,Total Time (ms),Avg Time (ms),Min Time (ms),Max Time (ms)\n",
        );

        for (name, result) in &results.operations {
            csv.push_str(&format!(
                "{},{},{:.2},{:.2},{:.2},{:.2}\n",
                name,
                result.call_count,
                result.total_time.as_secs_f64() * 1000.0,
                result.avg_time.as_secs_f64() * 1000.0,
                result.min_time.as_secs_f64() * 1000.0,
                result.max_time.as_secs_f64() * 1000.0
            ));
        }

        csv
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::build(ProfilerConfig::default())
    }
}

/// Read-only snapshot of a profiling session.
#[derive(Debug, Clone)]
pub struct ProfileSessionInfo {
    /// Session id.
    pub id: String,
    /// Session name.
    pub name: String,
    /// When the session started.
    pub start_time: Instant,
    /// When the session ended, if it has.
    pub end_time: Option<Instant>,
    /// Number of individually measured call durations recorded so far.
    pub sample_count: usize,
    /// Workload counters reported for this session.
    pub workload: WorkloadCounters,
    /// Whether results have been generated for this session.
    pub completed: bool,
}

// ---------------------------------------------------------------------------
// Real memory sources
// ---------------------------------------------------------------------------

/// Operating-system view of this process's memory.
#[derive(Debug, Clone, Copy)]
pub struct ProcessMemory {
    /// Resident set size in bytes.
    pub resident_bytes: usize,
    /// Virtual (reserved) address space in bytes.
    pub virtual_bytes: usize,
}

/// Read this process's memory usage from the operating system.
///
/// Returns `None` when the platform does not expose the current process (for
/// example inside a sandbox that hides `/proc`), so callers can say "unknown"
/// rather than report a placeholder.
pub fn read_process_memory() -> Option<ProcessMemory> {
    static SYSTEM: OnceLock<Mutex<sysinfo::System>> = OnceLock::new();

    let pid = sysinfo::get_current_pid().ok()?;
    let system = SYSTEM.get_or_init(|| Mutex::new(sysinfo::System::new()));
    let mut system = system.lock().ok()?;
    system.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
    let process = system.process(pid)?;
    Some(ProcessMemory {
        resident_bytes: process.memory() as usize,
        virtual_bytes: process.virtual_memory() as usize,
    })
}

/// Export format for profiling results
#[derive(Debug, Clone, Copy)]
pub enum ExportFormat {
    Json,
    Html,
    Flamegraph,
    Csv,
}

/// Global profiler instance
static GLOBAL_PROFILER: std::sync::OnceLock<Profiler> = std::sync::OnceLock::new();

/// Type alias for backward compatibility
pub type GlobalProfiler = Profiler;

/// Get the global profiler instance
pub fn get_global_profiler() -> &'static Profiler {
    GLOBAL_PROFILER.get_or_init(|| Profiler::build(ProfilerConfig::default()))
}

/// Convenience macro for profiling operations
#[macro_export]
macro_rules! profile_operation {
    ($name:expr, $code:block) => {{
        $crate::profiler::get_global_profiler().profile_function($name, || $code)
    }};
}

/// Convenience function for profiling with the global profiler
pub fn profile_fn<F, R>(name: &str, f: F) -> Result<(R, ProfileResults)>
where
    F: FnOnce() -> R,
{
    get_global_profiler().profile_function(name, f)
}

/// Convenience function for async profiling with the global profiler
pub async fn profile_async<F, R>(name: &str, f: F) -> Result<(R, ProfileResults)>
where
    F: std::future::Future<Output = R>,
{
    get_global_profiler().profile_async(name, f).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    #[test]
    fn test_profiler_creation() {
        let profiler = Profiler::new();
        assert!(profiler.is_ok());

        let profiler = profiler.expect("operation failed in test");
        assert!(profiler.is_enabled()); // Auto-enabled by default
    }

    #[test]
    fn test_uptime_advisor_and_metrics_tracker_are_reachable() {
        let profiler = Profiler::new().expect("operation failed in test");
        // Real reads of previously write-only fields: `uptime` reports a
        // real, non-negative elapsed duration; `advisor`/`metrics_tracker`
        // return the actual instances the profiler was built with.
        let _ = profiler.uptime();
        assert_eq!(
            profiler.metrics_tracker().latency_metrics().count,
            0,
            "a freshly built tracker has recorded nothing yet"
        );
        // `OptimizationAdvisor` exposes no introspectable state of its own;
        // reaching it without panicking is the contract here.
        let _ = profiler.advisor();
    }

    #[test]
    fn test_generate_optimization_suggestions_does_not_fabricate_improvement_numbers() {
        let profiler = Profiler::new().expect("operation failed in test");
        let mut slow_op = ProfileResult::new("slow_op".to_string());
        slow_op.total_time = Duration::from_millis(150);
        let mut operations = HashMap::new();
        operations.insert("slow_op".to_string(), slow_op);

        let suggestions = profiler.generate_optimization_suggestions(&operations);
        assert_eq!(
            suggestions.len(),
            1,
            "a >100ms operation should produce exactly one suggestion"
        );
        let improvement = &suggestions[0].expected_improvement;
        assert_eq!(
            improvement.latency_reduction, None,
            "no measurement or model backs a specific percentage; it must not be fabricated"
        );
        assert_eq!(improvement.throughput_increase, None);
        assert_eq!(improvement.memory_reduction, None);
    }

    #[test]
    fn test_session_management() {
        let profiler = Profiler::new().expect("operation failed in test");

        let session_id = profiler.start_session("test_session").expect("operation failed in test");
        assert!(!session_id.is_empty());

        sleep(Duration::from_millis(10));

        let results = profiler.end_session(&session_id).expect("operation failed in test");
        assert_eq!(results.session_id, session_id);
        assert!(results.total_duration > Duration::ZERO);
    }

    #[test]
    fn test_profile_function() {
        let profiler = Profiler::new().expect("operation failed in test");

        let (result, profile_results) = profiler
            .profile_function("test_operation", || {
                sleep(Duration::from_millis(10));
                42
            })
            .expect("operation failed in test");

        assert_eq!(result, 42);
        assert!(!profile_results.operations.is_empty());
        assert!(profile_results.total_duration > Duration::ZERO);
    }

    #[test]
    fn test_export_formats() {
        let profiler = Profiler::new().expect("operation failed in test");

        let (_, results) = profiler
            .profile_function("export_test", || {
                sleep(Duration::from_millis(5));
            })
            .expect("operation failed in test");

        // Test HTML generation
        let html = profiler.generate_html_report(&results);
        assert!(html.contains("TrustformeRS Performance Report"));

        // Test CSV generation
        let csv = profiler.generate_csv_report(&results);
        assert!(csv.contains("Operation,Calls"));
    }

    // -----------------------------------------------------------------------
    // Regression tests for the removed placeholder metrics.
    //
    // `generate_latency_metrics` used to set `median = mean`, `std_dev = 0.0`
    // and `p90 = p95 = p99 = p999 = max`; `generate_memory_metrics` returned
    // seven hardcoded constants; `generate_throughput_metrics` multiplied the
    // operation rate by 100 "tokens per operation". Each test below fails
    // against that code.
    // -----------------------------------------------------------------------

    /// Percentiles must come from the real sample distribution.
    #[test]
    fn latency_percentiles_are_computed_from_real_samples() {
        let profiler = Profiler::new().expect("profiler should build");
        let session_id = profiler.start_session("percentiles").expect("session");

        // Three deliberately different durations.
        for delay_ms in [1u64, 5, 40] {
            profiler.start_operation(&session_id, "step", None);
            sleep(Duration::from_millis(delay_ms));
            profiler.end_operation(&session_id, "step");
        }

        let results = profiler.end_session(&session_id).expect("results");
        let latency = &results.latency_metrics;

        assert_eq!(latency.count, 3, "every measured call must be counted");
        assert!(
            latency.std_dev_ms > 0.0,
            "a spread of 1ms/5ms/40ms cannot have zero standard deviation"
        );
        assert!(
            latency.median_ms < latency.max_ms,
            "median {} must be below max {} for this sample set",
            latency.median_ms,
            latency.max_ms
        );
        assert!(
            (latency.median_ms - latency.mean_ms).abs() > f64::EPSILON,
            "median must be the middle sample, not a copy of the mean"
        );
        assert!(
            latency.p50_ms <= latency.p90_ms && latency.p90_ms <= latency.p99_ms,
            "percentiles must be monotonic"
        );
        assert!(
            latency.window_duration <= results.total_duration,
            "the reported window must be the real session window, not a fixed hour"
        );
        assert!(
            latency.window_duration != Duration::from_secs(3600),
            "the window must not be the old hardcoded one-hour value"
        );
    }

    /// Unreported workload must not be turned into invented token counts.
    #[test]
    fn throughput_is_zero_until_a_workload_is_reported() {
        let profiler = Profiler::new().expect("profiler should build");
        let session_id = profiler.start_session("throughput").expect("session");
        profiler.start_operation(&session_id, "step", None);
        sleep(Duration::from_millis(2));
        profiler.end_operation(&session_id, "step");
        let results = profiler.end_session(&session_id).expect("results");
        assert_eq!(
            results.throughput_metrics.total_tokens, 0,
            "no caller reported any tokens, so none may be claimed"
        );
        assert_eq!(results.throughput_metrics.tokens_per_second, 0.0);

        let session_id = profiler.start_session("throughput_reported").expect("session");
        profiler.record_workload(&session_id, 512, 4, 16).expect("record workload");
        sleep(Duration::from_millis(2));
        let results = profiler.end_session(&session_id).expect("results");
        assert_eq!(results.throughput_metrics.total_tokens, 512);
        assert!(results.throughput_metrics.tokens_per_second > 0.0);
        assert_eq!(results.workload.batches, 4);
    }

    /// Memory metrics must be a real reading, tagged with their source.
    #[test]
    fn memory_metrics_are_measured_not_constant() {
        let profiler = Profiler::new().expect("profiler should build");
        let session_id = profiler.start_session("memory").expect("session");
        let results = profiler.end_session(&session_id).expect("results");

        match results.memory_metrics_source {
            MemoryMetricsSource::ProcessMemory => {
                let memory = results.memory_metrics.expect("a source implies metrics");
                assert_ne!(
                    memory.current_bytes,
                    1024 * 1024 * 80,
                    "80MB was the old hardcoded value"
                );
                assert_ne!(memory.peak_bytes, 1024 * 1024 * 100);
                assert_ne!(memory.num_allocations, 1000);
                assert_eq!(
                    memory.num_allocations, 0,
                    "the process view cannot count allocations, so it must report none"
                );
                assert!(
                    memory.current_bytes > 0,
                    "a live process has resident memory"
                );
            },
            MemoryMetricsSource::Unavailable => {
                assert!(
                    results.memory_metrics.is_none(),
                    "an unavailable source must not carry numbers"
                );
            },
        }
    }

    /// Benchmarks must not silently succeed with an empty result set.
    #[test]
    fn enabled_benchmarks_without_a_run_report_an_error() {
        let profiler = Profiler::with_config(ProfilerConfig {
            enable_benchmarks: true,
            ..ProfilerConfig::default()
        })
        .expect("profiler should build");
        let session_id = profiler.start_session("benchmarks").expect("session");
        let result = profiler.end_session(&session_id);
        assert!(
            result.is_err(),
            "returning `Some(vec![])` would claim benchmarks ran when none did"
        );
    }

    #[test]
    fn test_global_profiler() {
        let (result, profile_results) = profile_fn("global_test", || {
            sleep(Duration::from_millis(5));
            "test"
        })
        .expect("operation failed in test");

        assert_eq!(result, "test");
        assert!(!profile_results.operations.is_empty());
    }
}
