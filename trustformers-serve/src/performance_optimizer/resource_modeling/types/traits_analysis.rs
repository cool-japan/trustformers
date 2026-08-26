//! Analysis and validation types for resource modeling traits
//!
//! Benchmark suite, cache analysis, processing engine, validation,
//! quality assurance, and performance profiler report types.

use super::*;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

/// Read the host's CPU cache hierarchy as `(levels, total bytes)`.
///
/// Linux: every cache visible to CPU 0 is published under
/// `/sys/devices/system/cpu/cpu0/cache/index*/`, with `level` and a `size`
/// written as e.g. `32K` or `8192K`. Distinct levels are counted once, and the
/// sizes are summed.
///
/// macOS: `sysctl` reports `hw.l1dcachesize`, `hw.l2cachesize` and
/// `hw.l3cachesize` in bytes; a level is present when its size is non-zero.
///
/// Anything else gets [`MeasurementUnavailable`] rather than a guess.
fn detect_host_cache_hierarchy() -> anyhow::Result<(u8, usize)> {
    #[cfg(target_os = "linux")]
    {
        use std::collections::BTreeSet;

        let base = std::path::Path::new("/sys/devices/system/cpu/cpu0/cache");
        let entries = std::fs::read_dir(base).map_err(|error| {
            MeasurementUnavailable::raise(
                "the CPU cache hierarchy",
                "the kernel does not publish /sys/devices/system/cpu/cpu0/cache",
            )
            .context(error.to_string())
        })?;

        let mut levels = BTreeSet::new();
        let mut total = 0usize;
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(level) = std::fs::read_to_string(path.join("level")) else {
                continue;
            };
            let Ok(level) = level.trim().parse::<u8>() else {
                continue;
            };
            let Ok(size) = std::fs::read_to_string(path.join("size")) else {
                continue;
            };
            let Some(bytes) = parse_sysfs_cache_size(size.trim()) else {
                continue;
            };
            levels.insert(level);
            total += bytes;
        }

        if levels.is_empty() {
            return Err(MeasurementUnavailable::raise(
                "the CPU cache hierarchy",
                "the kernel published no readable cache index for CPU 0",
            ));
        }
        return Ok((levels.len() as u8, total));
    }

    #[cfg(target_os = "macos")]
    {
        let mut levels = 0u8;
        let mut total = 0usize;
        for key in ["hw.l1dcachesize", "hw.l2cachesize", "hw.l3cachesize"] {
            let output = std::process::Command::new("sysctl").args(["-n", key]).output();
            let Ok(output) = output else { continue };
            if !output.status.success() {
                continue;
            }
            let Ok(text) = String::from_utf8(output.stdout) else {
                continue;
            };
            let Ok(bytes) = text.trim().parse::<usize>() else {
                continue;
            };
            if bytes > 0 {
                levels += 1;
                total += bytes;
            }
        }

        if levels == 0 {
            return Err(MeasurementUnavailable::raise(
                "the CPU cache hierarchy",
                "sysctl reported no non-zero hw.l*cachesize value",
            ));
        }
        return Ok((levels, total));
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        Err(MeasurementUnavailable::raise(
            "the CPU cache hierarchy",
            "this platform exposes no cache-topology interface known to this crate",
        ))
    }
}

/// Parse a Linux sysfs cache `size` value such as `32K`, `1M` or `512` into
/// bytes.
///
/// Compiled on Linux, where the detector calls it, and under `cfg(test)`
/// everywhere else so the parser stays under test on other platforms.
#[cfg(any(target_os = "linux", test))]
pub(crate) fn parse_sysfs_cache_size(value: &str) -> Option<usize> {
    let value = value.trim();
    let (digits, multiplier) = match value.chars().last()? {
        'K' | 'k' => (&value[..value.len() - 1], 1024),
        'M' | 'm' => (&value[..value.len() - 1], 1024 * 1024),
        'G' | 'g' => (&value[..value.len() - 1], 1024 * 1024 * 1024),
        _ => (value, 1),
    };
    digits.trim().parse::<usize>().ok().map(|n| n * multiplier)
}

/// Median of `values`, or `0.0` for an empty slice.
pub(crate) fn median_of(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

/// Raised when a measurement has no backend on this build.
///
/// Several analyses in this module describe hardware behaviour — cache latency,
/// synthetic benchmark scores, workload characterisation — that cannot be
/// obtained without either a platform-specific counter API or an actual
/// benchmark harness. `trustformers-serve` carries neither. Each such analysis
/// returns this error rather than the constant it used to return.
#[derive(Debug, thiserror::Error)]
#[error("{measurement} cannot be measured on this build: {reason}")]
pub struct MeasurementUnavailable {
    /// What was asked for.
    pub measurement: &'static str,
    /// Why no value can be produced.
    pub reason: &'static str,
}

impl MeasurementUnavailable {
    /// Build the boxed error for `measurement`.
    ///
    /// Returns `anyhow::Error` rather than `Self` because every caller
    /// immediately wraps it in `Err(..)`; naming it `new` would suggest a
    /// constructor, so it is `raise`.
    pub(crate) fn raise(measurement: &'static str, reason: &'static str) -> anyhow::Error {
        anyhow::Error::new(Self {
            measurement,
            reason,
        })
    }
}

// ============================================================================
// Benchmark Suite Types
// ============================================================================

/// Synthetic benchmark suite for comprehensive performance testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticBenchmarkSuite {
    /// CPU benchmarks
    pub cpu_benchmarks: CpuBenchmarkSuite,
    /// Memory benchmarks
    pub memory_bandwidth: f64,
    /// Storage benchmarks
    pub storage_iops: u64,
    /// Network benchmarks
    pub network_bandwidth: f64,
}

impl Default for SyntheticBenchmarkSuite {
    fn default() -> Self {
        Self {
            cpu_benchmarks: CpuBenchmarkSuite::default(),
            memory_bandwidth: 0.0,
            storage_iops: 0,
            network_bandwidth: 0.0,
        }
    }
}

impl SyntheticBenchmarkSuite {
    /// Create a new SyntheticBenchmarkSuite with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Execute synthetic benchmark suite with given configuration
    pub async fn execute_suite(
        &self,
        _config: &HashMap<String, String>,
    ) -> anyhow::Result<HashMap<String, f64>> {
        // The scores this used to return — a constant 100.0 CPU score plus the
        // struct's own default-initialised fields echoed back — were never
        // produced by running anything.
        Err(MeasurementUnavailable::raise(
            "the synthetic benchmark suite",
            "no benchmark harness is linked into trustformers-serve",
        ))
    }
}

// ============================================================================
// Benchmark and Analysis Engine Types
// ============================================================================

/// Real workload analyzer for analyzing actual workload patterns
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RealWorkloadAnalyzer {
    /// Workload patterns analyzed
    pub patterns_analyzed: u64,
    /// Analysis accuracy
    pub accuracy: f64,
}

impl RealWorkloadAnalyzer {
    /// Create a new RealWorkloadAnalyzer with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze workload patterns with given configuration
    pub async fn analyze_workloads(
        &mut self,
        _config: &HashMap<String, String>,
    ) -> anyhow::Result<HashMap<String, f64>> {
        // `workload_efficiency` was the literal 85.0 and `patterns_found` was
        // a count of how many times this method had been called.
        Err(MeasurementUnavailable::raise(
            "workload pattern analysis",
            "no workload tracing source is wired into this analyzer",
        ))
    }
}

/// Micro-benchmark engine for detailed performance testing
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MicroBenchmarkEngine {
    /// Benchmarks executed
    pub benchmarks_executed: u64,
    /// Total execution time
    pub total_time: std::time::Duration,
}

impl MicroBenchmarkEngine {
    /// Create a new MicroBenchmarkEngine with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Execute micro-benchmarks with given configuration
    ///
    /// Returns [`MeasurementUnavailable`]: the `avg_latency_ns` of `100.0` and
    /// `throughput_ops_sec` of `1_000_000.0` this used to report were literals
    /// that no timing loop produced, and `benchmarks_run` counted calls to
    /// this method rather than benchmarks. Nothing was executed, so nothing is
    /// reported.
    pub async fn execute_micro_benchmarks(
        &mut self,
        _config: &HashMap<String, String>,
    ) -> anyhow::Result<HashMap<String, f64>> {
        Err(MeasurementUnavailable::raise(
            "micro-benchmark timings",
            "no micro-benchmark harness is linked into trustformers-serve",
        ))
    }
}

/// Benchmark orchestrator for coordinating benchmark execution
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkOrchestrator {
    /// Active benchmarks
    pub active_benchmarks: u32,
    /// Completed benchmarks
    pub completed_benchmarks: u64,
}

impl BenchmarkOrchestrator {
    /// Create a new BenchmarkOrchestrator with default values
    pub fn new() -> Self {
        Self::default()
    }
}

/// Benchmark execution state
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct BenchmarkExecutionState {
    /// Execution running
    pub running: bool,
    /// Start time
    #[serde(skip)]
    pub start_time: std::time::Instant,
    /// Progress percentage
    pub progress: f64,
}

impl Default for BenchmarkExecutionState {
    fn default() -> Self {
        Self {
            running: false,
            start_time: std::time::Instant::now(),
            progress: 0.0,
        }
    }
}

impl BenchmarkExecutionState {
    /// Create a new BenchmarkExecutionState with default values
    pub fn new() -> Self {
        Self::default()
    }
}

/// Benchmark result variants for different benchmark types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BenchmarkResult {
    /// Synthetic benchmark results
    Synthetic(HashMap<String, f64>),
    /// Real workload analysis results
    Workload(HashMap<String, f64>),
    /// Micro-benchmark results
    Micro(HashMap<String, f64>),
}

impl Default for BenchmarkResult {
    fn default() -> Self {
        Self::Synthetic(HashMap::new())
    }
}

/// Detailed benchmark result for a single benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedBenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Score
    pub score: f64,
    /// Execution time
    pub execution_time: std::time::Duration,
    /// Pass/fail status
    pub passed: bool,
}

impl Default for DetailedBenchmarkResult {
    fn default() -> Self {
        Self {
            name: String::new(),
            score: 0.0,
            execution_time: std::time::Duration::from_secs(0),
            passed: false,
        }
    }
}

// ============================================================================
// Cache Analysis Types
// ============================================================================

/// Cache detection engine for identifying cache hierarchy
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CacheDetectionEngine {
    /// Detected cache levels
    pub cache_levels: u8,
    /// Total cache size
    pub total_cache_size: usize,
}

impl CacheDetectionEngine {
    /// Create a new CacheDetectionEngine with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Detect the CPU cache hierarchy, returning `(levels, total bytes)`.
    ///
    /// Read from the operating system: Linux publishes every cache index under
    /// `/sys/devices/system/cpu/cpu0/cache/`, and macOS answers
    /// `hw.l1dcachesize` / `hw.l2cachesize` / `hw.l3cachesize` through `sysctl`.
    /// Platforms that expose neither get [`MeasurementUnavailable`].
    ///
    /// Before 0.2.1 this assigned `3` levels and `8 MiB` unconditionally and
    /// reported them as a detection result.
    pub fn detect_cache_hierarchy(&mut self) -> anyhow::Result<(u8, usize)> {
        let (levels, total) = detect_host_cache_hierarchy()?;
        self.cache_levels = levels;
        self.total_cache_size = total;
        Ok((levels, total))
    }
}

/// Cache performance tester
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CachePerformanceTester {
    /// Hit rate
    pub hit_rate: f64,
    /// Miss penalty (cycles)
    pub miss_penalty: f64,
}

impl CachePerformanceTester {
    /// Create a new CachePerformanceTester with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Test all cache levels
    pub fn test_all_cache_levels(&mut self) -> anyhow::Result<(f64, f64)> {
        // A 95% hit rate and a 100-cycle miss penalty were written here by
        // hand; nothing sampled a performance counter.
        Err(MeasurementUnavailable::raise(
            "cache hit rate and miss penalty",
            "reading them needs hardware performance counters (perf_event / \
             kperf), which this crate does not open",
        ))
    }

    /// Test L1 cache performance
    pub async fn test_l1_cache_performance(&mut self) -> anyhow::Result<f64> {
        Err(MeasurementUnavailable::raise(
            "L1 cache latency",
            "measuring it needs a pointer-chase benchmark sized to the cache, \
             which this crate does not run",
        ))
    }

    /// Test L2 cache performance
    pub async fn test_l2_cache_performance(&mut self) -> anyhow::Result<f64> {
        Err(MeasurementUnavailable::raise(
            "L2 cache latency",
            "measuring it needs a pointer-chase benchmark sized to the cache, \
             which this crate does not run",
        ))
    }

    /// Test L3 cache performance
    pub async fn test_l3_cache_performance(&mut self) -> anyhow::Result<f64> {
        Err(MeasurementUnavailable::raise(
            "L3 cache latency",
            "measuring it needs a pointer-chase benchmark sized to the cache, \
             which this crate does not run",
        ))
    }
}

/// Cache optimization analyzer
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CacheOptimizationAnalyzer {
    /// Optimization opportunities found
    pub opportunities: u32,
    /// Estimated improvement
    pub estimated_improvement: f64,
}

impl CacheOptimizationAnalyzer {
    /// Create a new CacheOptimizationAnalyzer with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze optimization opportunities
    pub fn analyze_optimization_opportunities(&mut self) -> anyhow::Result<(u32, f64)> {
        // "5 opportunities, 15% improvement" was written here as a literal.
        Err(MeasurementUnavailable::raise(
            "cache optimization opportunities",
            "there is no cache access trace to analyse",
        ))
    }
}

/// Cache modeling engine
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CacheModelingEngine {
    /// Model accuracy
    pub accuracy: f64,
    /// Prediction confidence
    pub confidence: f64,
}

impl CacheModelingEngine {
    /// Create a new CacheModelingEngine with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Model cache behavior
    pub fn model_cache_behavior(&mut self) -> anyhow::Result<(f64, f64)> {
        // 92.5% accuracy and 88% confidence for a model that was never fitted.
        Err(MeasurementUnavailable::raise(
            "cache behaviour modelling",
            "no cache model is fitted, so it has no accuracy to report",
        ))
    }
}

/// Cache analysis state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheAnalysisState {
    /// Analysis active
    pub active: bool,
    /// Samples collected
    pub samples_collected: u64,
}

impl Default for CacheAnalysisState {
    fn default() -> Self {
        Self {
            active: false,
            samples_collected: 0,
        }
    }
}

impl CacheAnalysisState {
    /// Create a new CacheAnalysisState with default values
    pub fn new() -> Self {
        Self::default()
    }
}

// ============================================================================
// Processing and Analysis Engine Types
// ============================================================================

/// Statistical analysis engine
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StatisticalAnalysisEngine {
    /// Analyses performed
    pub analyses_performed: u64,
    /// Analysis accuracy
    pub accuracy: f64,
}

impl StatisticalAnalysisEngine {
    /// Create a new StatisticalAnalysisEngine with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze results with statistical methods
    ///
    /// Returns [`MeasurementUnavailable`]. The `mean: 100.0`, `stddev: 10.0`,
    /// `median: 98.0` and `confidence_interval: 95.0` this used to return were
    /// the same four literals for every input — the `results` argument was
    /// never read. Summary statistics also have nothing to summarise here:
    /// `ProfileResult` is an enum of per-subsystem profile structs, not a
    /// numeric sample, so there is no population to take a mean over even in
    /// principle. Producing statistics from it needs a defined metric
    /// extraction that this crate does not have.
    pub async fn analyze_results(
        &mut self,
        _results: &HashMap<
            String,
            crate::performance_optimizer::resource_modeling::performance_profiler::ProfileResult,
        >,
    ) -> anyhow::Result<HashMap<String, f64>> {
        Err(MeasurementUnavailable::raise(
            "summary statistics over profiling results",
            "profile results carry no numeric sample to summarise",
        ))
    }
}

/// Trend analysis engine
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrendAnalysisEngine {
    /// Trends detected
    pub trends_detected: u32,
    /// Prediction accuracy
    pub prediction_accuracy: f64,
}

impl TrendAnalysisEngine {
    /// Create a new TrendAnalysisEngine with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze performance trends from results
    ///
    /// Returns [`MeasurementUnavailable`]. A trend needs a time series, and
    /// this is handed a single snapshot: the previous implementation reported
    /// `trend_direction: 1.0` ("improving") and `trend_strength: 0.8` on every
    /// call without reading `results` at all, so it declared improvement it
    /// had no way to observe.
    pub async fn analyze_performance_trends(
        &mut self,
        _results: &HashMap<
            String,
            crate::performance_optimizer::resource_modeling::performance_profiler::ProfileResult,
        >,
    ) -> anyhow::Result<HashMap<String, f64>> {
        Err(MeasurementUnavailable::raise(
            "performance trends",
            "a single profiling snapshot contains no time series to trend",
        ))
    }
}

/// Optimization recommender
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OptimizationRecommender {
    /// Recommendations generated
    pub recommendations_generated: u64,
    /// Success rate
    pub success_rate: f64,
}

impl OptimizationRecommender {
    /// Create a new OptimizationRecommender with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate optimization recommendations based on results
    ///
    /// Returns [`MeasurementUnavailable`]. The three sentences this used to
    /// return ("Consider increasing thread pool size", ...) were a fixed list
    /// emitted regardless of `results`, carried `estimated_impact` figures of
    /// `0.8 / 0.6 / 0.4` that no experiment produced, and would have been
    /// identical on a machine where every one of them was the wrong advice.
    /// Advice presented as derived from measurements has to be derived from
    /// measurements.
    pub async fn generate_recommendations(
        &mut self,
        _results: &HashMap<String, f64>,
    ) -> anyhow::Result<OptimizationRecommendations> {
        Err(MeasurementUnavailable::raise(
            "optimization recommendations",
            "no rule set maps profiling statistics to recommendations on this build",
        ))
    }
}

/// Report generator
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReportGenerator {
    /// Reports generated
    pub reports_generated: u64,
    /// Report format
    pub format: String,
}

impl ReportGenerator {
    /// Create a new ReportGenerator with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Render the supplied profiling statistics as a report.
    ///
    /// The report now contains the caller's data and nothing else. It used to
    /// ignore `data` entirely and emit the fixed lines "System performance
    /// metrics analyzed" and "Recommendations: See optimization section" —
    /// a report that read the same whether the machine was healthy or on fire,
    /// and that pointed at an "optimization section" it did not produce.
    ///
    /// Keys are emitted in sorted order so that two reports over the same
    /// statistics compare equal.
    pub async fn generate_detailed_report(
        &mut self,
        data: &HashMap<String, f64>,
    ) -> anyhow::Result<String> {
        self.reports_generated += 1;
        let format = if self.format.is_empty() { "text" } else { &self.format };
        let mut entries: Vec<(&String, &f64)> = data.iter().collect();
        entries.sort_by(|a, b| a.0.cmp(b.0));

        let mut report = format!(
            "Performance Profiling Report #{}\nFormat: {}\nMetrics: {}\n",
            self.reports_generated,
            format,
            entries.len()
        );
        if entries.is_empty() {
            report.push_str("No statistics were supplied for this report.\n");
        } else {
            for (key, value) in entries {
                report.push_str(&format!("  {key}: {value}\n"));
            }
        }
        Ok(report)
    }
}

/// Processing state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingState {
    /// Processing active
    pub active: bool,
    /// Items processed
    pub items_processed: u64,
}

impl Default for ProcessingState {
    fn default() -> Self {
        Self {
            active: false,
            items_processed: 0,
        }
    }
}

impl ProcessingState {
    /// Create a new ProcessingState with default values
    pub fn new() -> Self {
        Self::default()
    }
}

/// Bottleneck type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BottleneckType {
    /// CPU bottleneck
    Cpu,
    /// Memory bottleneck
    Memory,
    /// I/O bottleneck
    Io,
    /// Network bottleneck
    Network,
    /// GPU bottleneck
    Gpu,
}

// ============================================================================
// Validation and Quality Assurance Types
// ============================================================================

/// Result validation engine
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResultValidationEngine {
    /// Validations performed
    pub validations_performed: u64,
    /// Pass rate
    pub pass_rate: f64,
}

impl ResultValidationEngine {
    /// Create a new ResultValidationEngine with default values
    pub fn new() -> Self {
        Self::default()
    }
}

/// Consistency checker
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConsistencyChecker {
    /// Checks performed
    pub checks_performed: u64,
    /// Inconsistencies found
    pub inconsistencies_found: u32,
}

impl ConsistencyChecker {
    /// Create a new ConsistencyChecker with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Check the supplied metrics for values that cannot be a measurement.
    ///
    /// A metric is inconsistent when it is `NaN` or infinite: no profiler can
    /// have measured either, so its presence means the value came from a
    /// division by zero, an uninitialised field, or an arithmetic overflow
    /// upstream. The score is the fraction of entries that survive that check,
    /// and every failing key is named in `inconsistencies`.
    ///
    /// This used to report `is_consistent: true` and `consistency_score: 0.95`
    /// unconditionally, without reading `results` — it declared data sound
    /// before looking at it, and could never have flagged a problem.
    /// `inconsistencies_found` now counts real findings rather than staying at
    /// zero forever.
    pub async fn check_result_consistency(
        &mut self,
        results: &HashMap<String, f64>,
    ) -> anyhow::Result<ConsistencyResults> {
        self.checks_performed += 1;

        let mut inconsistencies: Vec<String> = results
            .iter()
            .filter(|(_, value)| !value.is_finite())
            .map(|(key, value)| {
                let kind = if value.is_nan() { "NaN" } else { "infinite" };
                format!("{key} is {kind}, which is not a measurable value")
            })
            .collect();
        inconsistencies.sort();

        self.inconsistencies_found = self
            .inconsistencies_found
            .saturating_add(u32::try_from(inconsistencies.len()).unwrap_or(u32::MAX));

        // An empty input is vacuously consistent: there is nothing in it that
        // could contradict anything else.
        let score = if results.is_empty() {
            1.0
        } else {
            (results.len() - inconsistencies.len()) as f64 / results.len() as f64
        };

        Ok(ConsistencyResults {
            is_consistent: inconsistencies.is_empty(),
            consistency_score: score,
            inconsistencies,
            overall_consistency_score: score,
        })
    }
}

/// Outlier detector
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OutlierDetector {
    /// Outliers detected
    pub outliers_detected: u32,
    /// Detection sensitivity
    pub sensitivity: f64,
}

impl OutlierDetector {
    /// Create a new OutlierDetector with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Detect outliers in `data` with a modified z-score test.
    pub async fn detect_outliers(&mut self, _data: &[f64]) -> anyhow::Result<OutlierResults> {
        // Modified z-score against the median absolute deviation: robust to the
        // outliers it is looking for, unlike a mean/stddev rule. A point is an
        // outlier when its score exceeds `sensitivity` (default 3.5, the
        // conventional Iglewicz-Hoaglin cutoff).
        //
        // This used to increment a counter and return "0 outliers" regardless of
        // the data, so a run of wild samples reported a clean data set.
        if _data.is_empty() {
            return Ok(OutlierResults::default());
        }

        let threshold = if self.sensitivity > 0.0 { self.sensitivity } else { 3.5 };
        let median = median_of(_data);
        let deviations: Vec<f64> = _data.iter().map(|x| (x - median).abs()).collect();
        let mad = median_of(&deviations);

        // With a zero MAD every deviation is either exactly zero or an outlier
        // by any robust measure; fall back to a strict equality test.
        let scores: Vec<f64> = _data
            .iter()
            .map(|x| {
                if mad > 0.0 {
                    0.6745 * (x - median).abs() / mad
                } else if (x - median).abs() > 0.0 {
                    f64::INFINITY
                } else {
                    0.0
                }
            })
            .collect();

        let outlier_indices: Vec<usize> = scores
            .iter()
            .enumerate()
            .filter(|(_, score)| **score > threshold)
            .map(|(index, _)| index)
            .collect();

        self.outliers_detected = outlier_indices.len() as u32;

        let mut outlier_metrics = HashMap::new();
        outlier_metrics.insert("median".to_string(), median);
        outlier_metrics.insert("median_absolute_deviation".to_string(), mad);
        outlier_metrics.insert("threshold".to_string(), threshold);

        Ok(OutlierResults {
            outliers_detected: outlier_indices.len() as u64,
            outlier_percentage: outlier_indices.len() as f64 / _data.len() as f64 * 100.0,
            outlier_indices,
            outlier_scores: scores,
            outlier_metrics,
        })
    }
}

/// Quality assurance engine
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualityAssuranceEngine {
    /// Quality checks performed
    pub checks_performed: u64,
    /// Overall quality score
    pub quality_score: f64,
}

impl QualityAssuranceEngine {
    /// Create a new QualityAssuranceEngine with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Assess the supplied metrics against the checks this build can actually
    /// run.
    ///
    /// Two properties are checkable without knowing what each metric means:
    /// whether the set is empty (nothing was collected), and whether every
    /// value is finite. `data_completeness` is `0.0` for an empty set and
    /// `1.0` otherwise; `consistency_score` is the finite fraction; the
    /// overall score is their product; and each non-finite metric becomes a
    /// named issue.
    ///
    /// `reliability_score` / `data_reliability` are equal to the consistency
    /// score rather than the `0.90` literal they used to carry: reliability
    /// over repeated runs would need more than one run, and this method is
    /// given one.
    ///
    /// Previously every field was a constant — `overall_quality: 0.95`,
    /// `data_completeness: 1.0`, no issues — returned without reading `data`,
    /// so a report over an empty metric set claimed complete, high-quality
    /// data.
    pub async fn perform_quality_checks(
        &mut self,
        data: &HashMap<String, f64>,
    ) -> anyhow::Result<QualityAssessmentReport> {
        self.checks_performed += 1;

        let mut issues: Vec<String> = data
            .iter()
            .filter(|(_, value)| !value.is_finite())
            .map(|(key, value)| {
                let kind = if value.is_nan() { "NaN" } else { "infinite" };
                format!("{key} is {kind}")
            })
            .collect();
        issues.sort();

        let data_completeness = if data.is_empty() { 0.0 } else { 1.0 };
        let consistency_score = if data.is_empty() {
            0.0
        } else {
            (data.len() - issues.len()) as f64 / data.len() as f64
        };
        let overall = data_completeness * consistency_score;
        self.quality_score = overall;

        let recommendations = if issues.is_empty() && !data.is_empty() {
            Vec::new()
        } else {
            vec![QualityRecommendation {
                recommendation: if data.is_empty() {
                    "No metrics were collected; check that profiling ran before assessing quality"
                        .to_string()
                } else {
                    "Re-run profiling: some metrics are not finite values".to_string()
                },
                priority: "High".to_string(),
                expected_improvement: 0.0,
                action: "Re-run profiling".to_string(),
                implementation_difficulty: "Low".to_string(),
            }]
        };

        Ok(QualityAssessmentReport {
            overall_quality: overall,
            data_completeness,
            consistency_score,
            reliability_score: consistency_score,
            issues: issues.clone(),
            quality_score: overall,
            quality_issues: issues,
            data_reliability: consistency_score,
            recommendations,
            assessment_timestamp: chrono::Utc::now(),
        })
    }
}

/// Validation state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationState {
    /// Validation active
    pub active: bool,
    /// Items validated
    pub items_validated: u64,
}

impl Default for ValidationState {
    fn default() -> Self {
        Self {
            active: false,
            items_validated: 0,
        }
    }
}

impl ValidationState {
    /// Create a new ValidationState with default values
    pub fn new() -> Self {
        Self::default()
    }
}

/// Quality issue type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityIssueType {
    /// Data quality issue
    DataQuality,
    /// Performance issue
    Performance,
    /// Consistency issue
    Consistency,
    /// Accuracy issue
    Accuracy,
}

/// Issue severity enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}

/// Recommendation priority enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationPriority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Difficulty level enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DifficultyLevel {
    /// Easy
    Easy,
    /// Medium
    Medium,
    /// Hard
    Hard,
    /// Very hard
    VeryHard,
}

// ============================================================================
// Performance Profiler Types
// ============================================================================

/// Performance analysis report
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceAnalysisReport {
    pub summary: String,
    pub cpu_analysis: String,
    pub memory_analysis: String,
    pub io_analysis: String,
    pub network_analysis: String,
    pub gpu_analysis: String,
    pub recommendations: Vec<String>,
    pub executive_summary: ExecutiveSummary,
    pub detailed_analysis: String,
    pub optimization_recommendations: Vec<String>,
    pub performance_score: f64,
    pub analysis_timestamp: DateTime<Utc>,
}

/// Optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OptimizationRecommendations {
    pub recommendations: Vec<String>,
    pub priority_order: Vec<String>,
    pub estimated_impact: Vec<f64>,
}

/// Benchmark suite definition
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BenchmarkSuiteDefinition {
    pub suite_name: String,
    pub benchmarks: Vec<String>,
    pub configuration: std::collections::HashMap<String, String>,
    pub synthetic_config: HashMap<String, String>,
    pub workload_config: HashMap<String, String>,
    pub micro_config: HashMap<String, String>,
}

/// Benchmark suite results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSuiteResults {
    pub suite_definition: BenchmarkSuiteDefinition,
    pub results: HashMap<String, BenchmarkResult>,
    pub execution_duration: Duration,
    pub timestamp: DateTime<Utc>,
}

impl Default for BenchmarkSuiteResults {
    fn default() -> Self {
        Self {
            suite_definition: BenchmarkSuiteDefinition::default(),
            results: HashMap::new(),
            execution_duration: Duration::from_secs(0),
            timestamp: Utc::now(),
        }
    }
}

/// Quality assessment report
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualityAssessmentReport {
    pub overall_quality: f64,
    pub data_completeness: f64,
    pub consistency_score: f64,
    pub reliability_score: f64,
    pub issues: Vec<String>,
    pub quality_score: f64,
    pub quality_issues: Vec<String>,
    pub data_reliability: f64,
    pub recommendations: Vec<QualityRecommendation>,
    pub assessment_timestamp: DateTime<Utc>,
}

/// Sequential I/O performance metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SequentialIoPerformance {
    pub read_throughput_mbps: f64,
    pub write_throughput_mbps: f64,
    pub read_latency_ms: f64,
    pub write_latency_ms: f64,
    pub results: Vec<SequentialIoResult>,
}

/// Random I/O performance metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RandomIoPerformance {
    pub read_iops: f64,
    pub write_iops: f64,
    pub read_latency_us: f64,
    pub write_latency_us: f64,
    pub results: Vec<RandomIoResult>,
}

/// Filesystem performance metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FilesystemPerformanceMetrics {
    pub metadata_ops_per_sec: f64,
    pub file_creation_rate: f64,
    pub directory_traversal_time_ms: f64,
    pub file_deletion_rate: f64,
    pub directory_traversal_rate: f64,
    pub metadata_operation_latency: Duration,
}

/// Packet loss characteristics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PacketLossCharacteristics {
    pub loss_rate: f64,
    pub burst_loss_rate: f64,
    pub recovery_time_ms: f64,
    pub loss_by_packet_size: HashMap<String, f64>,
    pub baseline_loss_rate: f64,
    pub recovery_time: Duration,
}

/// Connection overhead analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConnectionOverheadAnalysis {
    pub connection_setup_time_ms: f64,
    pub teardown_time_ms: f64,
    pub overhead_percentage: f64,
    pub tcp_handshake_overhead: Duration,
    pub udp_setup_overhead: Duration,
    pub ssl_handshake_overhead: Duration,
    pub connection_reuse_benefit: f64,
}

/// Protocol performance analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProtocolPerformanceAnalysis {
    pub protocol_name: String,
    pub throughput_mbps: f64,
    pub latency_ms: f64,
    pub efficiency: f64,
    pub tcp_performance: ProtocolPerformanceMetrics,
    pub udp_performance: ProtocolPerformanceMetrics,
    pub http_performance: ProtocolPerformanceMetrics,
    pub websocket_performance: ProtocolPerformanceMetrics,
}

/// Protocol performance metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProtocolPerformanceMetrics {
    pub protocol: String,
    pub throughput: f64,
    pub latency: f64,
    pub packet_loss: f64,
    pub throughput_mbps: f64,
    pub cpu_utilization: f64,
    pub memory_overhead_kb: u64,
}

/// GPU thermal analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuThermalAnalysis {
    pub temperature_celsius: f64,
    pub hotspot_temp_celsius: f64,
    pub thermal_throttling: bool,
    pub cooling_effectiveness: f64,
    pub idle_temperature: f64,
    pub load_temperature: f64,
    pub throttling_threshold: f64,
    pub power_consumption_idle: f64,
    pub power_consumption_load: f64,
    pub cooling_efficiency: f64,
}

/// Compute utilization analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ComputeUtilizationAnalysis {
    pub compute_utilization: f64,
    pub memory_utilization: f64,
    pub efficiency_score: f64,
    pub shader_utilization: f64,
    pub memory_controller_utilization: f64,
    pub tensor_core_utilization: f64,
    pub rt_core_utilization: f64,
    pub optimal_workload_size: usize,
}

/// GPU capability information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuCapabilityInfo {
    pub vendor: GpuVendor,
    pub model: String,
    pub compute_capability: String,
    pub cuda_cores: u32,
    pub memory_bandwidth_gbps: f64,
    pub max_clock_mhz: u32,
    pub features: Vec<String>,
}

/// Comprehensive cache analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ComprehensiveCacheAnalysis {
    pub l1_hit_rate: f64,
    pub l2_hit_rate: f64,
    pub l3_hit_rate: f64,
    pub cache_miss_penalty_ns: f64,
    pub cache_hierarchy: Vec<CpuCacheAnalysis>,
    pub performance_results: HashMap<String, f64>,
    pub optimization_analysis: String,
    pub cache_model: String,
    pub analysis_duration: Duration,
    pub timestamp: DateTime<Utc>,
}

/// CPU cache analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CpuCacheAnalysis {
    pub cache_level: u8,
    pub hit_rate: f64,
    pub miss_rate: f64,
    pub latency_ns: f64,
    pub hierarchy: Vec<String>,
    pub l1_performance: HashMap<String, f64>,
    pub l2_performance: HashMap<String, f64>,
    pub l3_performance: HashMap<String, f64>,
    pub coherency_analysis: CacheCoherencyAnalysis,
    pub prefetcher_analysis: PrefetcherAnalysis,
}

/// Cache coherency analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CacheCoherencyAnalysis {
    pub coherency_protocol: String,
    pub invalidations_per_sec: f64,
    pub coherency_traffic_mbps: f64,
    pub protocol: String,
    pub coherency_overhead: f64,
    pub false_sharing_impact: f64,
    pub coherency_traffic_percentage: f64,
}

/// Prefetcher analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PrefetcherAnalysis {
    pub prefetch_accuracy: f64,
    pub useful_prefetches: u64,
    pub wasted_prefetches: u64,
    pub l1_prefetcher_hit_rate: f64,
    pub l2_prefetcher_hit_rate: f64,
    pub prefetch_coverage: f64,
    pub prefetch_timeliness: f64,
}

/// Processed profiling results
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProcessedResults {
    pub results: std::collections::HashMap<String, Vec<f64>>,
    pub metadata: std::collections::HashMap<String, String>,
    pub timestamp: DateTime<Utc>,
    pub statistics: HashMap<String, f64>,
    pub trends: Vec<String>,
    pub correlations: HashMap<String, f64>,
    pub bottlenecks: Vec<String>,
    pub processing_duration: Duration,
}

/// Executive summary for reports
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExecutiveSummary {
    pub key_findings: Vec<String>,
    pub performance_score: f64,
    pub critical_issues: Vec<String>,
    pub overall_performance_rating: String,
    pub critical_recommendations: Vec<String>,
}

/// Validation results
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValidationResults {
    pub is_valid: bool,
    pub validation_errors: Vec<String>,
    pub validation_warnings: Vec<String>,
    pub confidence_score: f64,
    pub consistency_results: ConsistencyResults,
    pub outlier_results: OutlierResults,
    pub qa_results: QualityAssessmentReport,
    pub confidence_scores: HashMap<String, f64>,
    pub overall_validity: bool,
    pub validation_duration: Duration,
    pub timestamp: DateTime<Utc>,
}

/// Consistency results
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConsistencyResults {
    pub is_consistent: bool,
    pub consistency_score: f64,
    pub inconsistencies: Vec<String>,
    pub overall_consistency_score: f64,
}

/// Outlier results
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OutlierResults {
    pub outliers_detected: u64,
    pub outlier_indices: Vec<usize>,
    pub outlier_scores: Vec<f64>,
    pub outlier_percentage: f64,
    pub outlier_metrics: HashMap<String, f64>,
}

/// Quality results
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualityResults {
    pub quality_score: f64,
    pub quality_metrics: std::collections::HashMap<String, f64>,
    pub quality_issues: Vec<String>,
    pub overall_quality_score: f64,
}

/// Confidence scores
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConfidenceScores {
    pub overall_confidence: f64,
    pub metric_confidence: std::collections::HashMap<String, f64>,
    pub consistency_confidence: f64,
    pub outlier_confidence: f64,
    pub quality_confidence: f64,
}

/// Quality issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    pub issue_type: String,
    pub severity: String,
    pub description: String,
    pub affected_metrics: Vec<String>,
}

/// Quality recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRecommendation {
    pub recommendation: String,
    pub priority: String,
    pub expected_improvement: f64,
    pub action: String,
    pub implementation_difficulty: String,
}

/// Performance correlations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceCorrelations {
    pub correlations: std::collections::HashMap<String, f64>,
    pub strong_correlations: Vec<(String, String, f64)>,
    pub cpu_memory_correlation: f64,
    pub memory_io_correlation: f64,
    pub network_cpu_correlation: f64,
    pub gpu_memory_correlation: f64,
    pub cross_subsystem_dependencies: HashMap<String, Vec<String>>,
}

/// Bottleneck analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BottleneckAnalysis {
    pub primary_bottleneck: String,
    pub bottleneck_severity: f64,
    pub contributing_factors: Vec<String>,
    pub identified_bottlenecks: Vec<PerformanceBottleneck>,
    pub bottleneck_interaction_matrix: BottleneckInteractionMatrix,
}

/// Bottleneck interaction matrix
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BottleneckInteractionMatrix {
    pub interactions: std::collections::HashMap<String, std::collections::HashMap<String, f64>>,
    pub interaction_coefficients: HashMap<String, f64>,
}

/// Storage analysis results
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StorageAnalysisResults {
    pub sequential_performance: SequentialIoPerformance,
    pub random_performance: RandomIoPerformance,
    pub filesystem_metrics: FilesystemPerformanceMetrics,
}

/// Queue depth optimization results
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueueDepthOptimizationResults {
    pub optimal_queue_depth: usize,
    pub throughput_at_optimal: f64,
    pub latency_at_optimal: f64,
}

/// I/O latency analysis results
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IoLatencyAnalysisResults {
    pub avg_latency_us: f64,
    pub p50_latency_us: f64,
    pub p99_latency_us: f64,
    pub max_latency_us: f64,
}

/// I/O pattern analysis results
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IoPatternAnalysisResults {
    pub sequential_ratio: f64,
    pub random_ratio: f64,
    pub read_write_ratio: f64,
}

/// Network interface analysis results
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkInterfaceAnalysisResults {
    pub interface_name: String,
    pub bandwidth_mbps: f64,
    pub packet_rate_pps: f64,
    pub error_rate: f64,
    pub max_bandwidth_bps: u64,
    pub mtu_size: u32,
}

/// Network bandwidth analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkBandwidthAnalysis {
    pub peak_bandwidth_mbps: f64,
    pub average_bandwidth_mbps: f64,
    pub utilization_percentage: f64,
}

/// Network latency analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkLatencyAnalysis {
    pub min_latency_ms: f64,
    pub avg_latency_ms: f64,
    pub max_latency_ms: f64,
    pub jitter_ms: f64,
}

/// MTU optimization results
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MtuOptimizationResults {
    pub optimal_mtu: usize,
    pub throughput_improvement: f64,
    pub latency_impact: f64,
}

/// GPU compute performance
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuComputePerformance {
    pub compute_throughput_gflops: f64,
    pub memory_throughput_gbps: f64,
    pub efficiency: f64,
    pub peak_gflops: f64,
}

/// GPU memory performance
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuMemoryPerformance {
    pub bandwidth_gbps: f64,
    pub latency_ns: f64,
    pub utilization: f64,
    pub peak_bandwidth_gbps: f64,
    pub transfer_overhead_ns: f64,
}

/// GPU kernel analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuKernelAnalysis {
    pub kernel_name: String,
    pub execution_time_ms: f64,
    pub occupancy: f64,
    pub memory_efficiency: f64,
    pub average_launch_overhead_ns: f64,
    pub context_switch_overhead_ns: f64,
}

// ============================================================================
// Hardware Detector Types
// ============================================================================

// CpuPerformanceCharacteristics, StorageDevice, NetworkInterface, GpuDeviceModel,
// GpuUtilizationCharacteristics are imported from super::super::types (lines 34-35)

// ============================================================================
// Temperature Monitor Types
// ============================================================================

/// Fan controller
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FanController {
    pub fan_id: String,
    pub current_speed_rpm: u32,
    pub target_speed_rpm: u32,
    pub control_mode: String,
}

/// Cooling curve
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CoolingCurve {
    pub temperature_points: Vec<f64>,
    pub fan_speed_points: Vec<u32>,
    pub curve_type: String,
}

impl Default for QualityIssue {
    fn default() -> Self {
        Self {
            issue_type: String::new(),
            severity: "medium".to_string(),
            description: String::new(),
            affected_metrics: Vec::new(),
        }
    }
}

impl Default for QualityRecommendation {
    fn default() -> Self {
        Self {
            recommendation: String::new(),
            priority: "medium".to_string(),
            expected_improvement: 0.0,
            action: "none".to_string(),
            implementation_difficulty: "low".to_string(),
        }
    }
}

// =============================================================================
// I/O PROFILING RESULT TYPES
// =============================================================================

/// Sequential I/O test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequentialIoResult {
    pub throughput: f64,
    pub latency: Duration,
    pub block_size: usize,
    pub total_bytes: usize,
    pub operation_type: String,
    pub test_size: usize,
    pub read_mbps: f64,
    pub read_latency: Duration,
    pub write_mbps: f64,
    pub write_latency: Duration,
}

impl Default for SequentialIoResult {
    fn default() -> Self {
        Self {
            throughput: 0.0,
            latency: Duration::from_secs(0),
            block_size: 0,
            total_bytes: 0,
            operation_type: String::from("read"),
            test_size: 0,
            read_mbps: 0.0,
            read_latency: Duration::from_secs(0),
            write_mbps: 0.0,
            write_latency: Duration::from_secs(0),
        }
    }
}

/// Random I/O test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomIoResult {
    pub iops: f64,
    pub latency: Duration,
    pub queue_depth: usize,
    pub total_operations: usize,
    pub operation_type: String,
    pub block_size: usize,
    pub read_iops: f64,
    pub write_iops: f64,
    pub mixed_workload_iops: f64,
}

impl Default for RandomIoResult {
    fn default() -> Self {
        Self {
            iops: 0.0,
            latency: Duration::from_secs(0),
            queue_depth: 1,
            total_operations: 0,
            operation_type: String::from("read"),
            block_size: 4096,
            read_iops: 0.0,
            write_iops: 0.0,
            mixed_workload_iops: 0.0,
        }
    }
}

#[cfg(test)]
mod measurement_tests {
    use super::*;

    #[test]
    fn sysfs_cache_sizes_parse_with_their_units() {
        assert_eq!(parse_sysfs_cache_size("32K"), Some(32 * 1024));
        assert_eq!(parse_sysfs_cache_size("8192K"), Some(8192 * 1024));
        assert_eq!(parse_sysfs_cache_size("1M"), Some(1024 * 1024));
        assert_eq!(parse_sysfs_cache_size("512"), Some(512));
        assert_eq!(parse_sysfs_cache_size(""), None);
        assert_eq!(parse_sysfs_cache_size("wat"), None);
    }

    #[test]
    fn median_handles_both_parities_and_the_empty_case() {
        assert_eq!(median_of(&[]), 0.0);
        assert_eq!(median_of(&[5.0]), 5.0);
        assert_eq!(median_of(&[1.0, 3.0, 2.0]), 2.0);
        assert_eq!(median_of(&[1.0, 2.0, 3.0, 4.0]), 2.5);
    }

    #[tokio::test]
    async fn outlier_detection_finds_the_outlier_it_is_given() {
        let mut detector = OutlierDetector::default();
        // Nine tightly clustered samples and one far away.
        let data = [10.0, 10.2, 9.8, 10.1, 9.9, 10.3, 9.7, 10.0, 10.1, 250.0];

        let results = detector.detect_outliers(&data).await.expect("detect");

        assert_eq!(
            results.outliers_detected, 1,
            "the sample at 250.0 is an outlier among values near 10"
        );
        assert_eq!(results.outlier_indices, vec![9]);
        assert!(results.outlier_percentage > 9.0 && results.outlier_percentage < 11.0);
        assert_eq!(results.outlier_scores.len(), data.len());
    }

    #[tokio::test]
    async fn outlier_detection_reports_none_for_clean_data() {
        let mut detector = OutlierDetector::default();
        let data = [10.0, 10.2, 9.8, 10.1, 9.9, 10.3, 9.7, 10.0];

        let results = detector.detect_outliers(&data).await.expect("detect");

        assert_eq!(results.outliers_detected, 0);
        assert!(results.outlier_indices.is_empty());
        assert_eq!(results.outlier_percentage, 0.0);
    }

    #[tokio::test]
    async fn cache_latency_probes_report_their_absence() {
        let mut tester = CachePerformanceTester::default();

        let error = tester
            .test_l1_cache_performance()
            .await
            .expect_err("no counter API is open, so no latency can be reported");
        assert!(
            error.to_string().contains("cannot be measured"),
            "unexpected error: {error}"
        );

        let error = tester
            .test_all_cache_levels()
            .expect_err("hit rate and miss penalty need performance counters");
        assert!(
            error.to_string().contains("cannot be measured"),
            "unexpected error: {error}"
        );
    }

    #[tokio::test]
    async fn synthetic_benchmarks_report_their_absence() {
        let suite = SyntheticBenchmarkSuite::default();
        let error = suite
            .execute_suite(&HashMap::new())
            .await
            .expect_err("no benchmark harness is linked, so there are no scores");
        assert!(
            error.to_string().contains("cannot be measured"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn cache_hierarchy_detection_is_read_from_the_host_or_refused() {
        let mut engine = CacheDetectionEngine::default();
        match engine.detect_cache_hierarchy() {
            Ok((levels, total)) => {
                assert!(levels >= 1, "a detected hierarchy has at least one level");
                assert!(total > 0, "a detected hierarchy has a non-zero size");
            },
            Err(error) => assert!(
                error.to_string().contains("cannot be measured"),
                "a failure must say why, not fall back to a constant: {error}"
            ),
        }
    }

    /// Regression: `check_result_consistency` used to report
    /// `is_consistent: true` with a score of `0.95` without reading its input,
    /// so it could never flag anything.
    #[tokio::test]
    async fn consistency_check_flags_values_that_cannot_be_measurements() {
        let mut checker = ConsistencyChecker::new();
        let mut results = HashMap::new();
        results.insert("throughput".to_string(), 120.0);
        results.insert("latency".to_string(), f64::NAN);
        results.insert("bandwidth".to_string(), f64::INFINITY);

        let outcome = checker
            .check_result_consistency(&results)
            .await
            .expect("the check itself must succeed");

        assert!(!outcome.is_consistent);
        assert_eq!(outcome.inconsistencies.len(), 2);
        assert!((outcome.consistency_score - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(checker.inconsistencies_found, 2);
        assert!(outcome.inconsistencies.iter().any(|i| i.contains("latency")));
        assert!(outcome.inconsistencies.iter().any(|i| i.contains("bandwidth")));
    }

    /// Clean input is reported clean, and the score is exactly 1.0 rather than
    /// the 0.95 the old constant returned.
    #[tokio::test]
    async fn consistency_check_passes_finite_metrics() {
        let mut checker = ConsistencyChecker::new();
        let mut results = HashMap::new();
        results.insert("throughput".to_string(), 120.0);
        results.insert("latency".to_string(), 3.5);

        let outcome = checker
            .check_result_consistency(&results)
            .await
            .expect("the check itself must succeed");

        assert!(outcome.is_consistent);
        assert!(outcome.inconsistencies.is_empty());
        assert!((outcome.consistency_score - 1.0).abs() < 1e-9);
    }

    /// Regression: `perform_quality_checks` used to report
    /// `data_completeness: 1.0` and `overall_quality: 0.95` for an empty
    /// metric set — high-quality, complete data that did not exist.
    #[tokio::test]
    async fn quality_assessment_reports_an_empty_metric_set_as_incomplete() {
        let mut engine = QualityAssuranceEngine::new();
        let report = engine
            .perform_quality_checks(&HashMap::new())
            .await
            .expect("the assessment itself must succeed");

        assert!((report.data_completeness - 0.0).abs() < 1e-9);
        assert!((report.overall_quality - 0.0).abs() < 1e-9);
        assert_eq!(report.recommendations.len(), 1);
    }

    /// A complete, finite metric set scores 1.0 and raises no issues.
    #[tokio::test]
    async fn quality_assessment_scores_clean_metrics() {
        let mut engine = QualityAssuranceEngine::new();
        let mut data = HashMap::new();
        data.insert("a".to_string(), 1.0);
        data.insert("b".to_string(), 2.0);

        let report = engine
            .perform_quality_checks(&data)
            .await
            .expect("the assessment itself must succeed");

        assert!((report.overall_quality - 1.0).abs() < 1e-9);
        assert!(report.quality_issues.is_empty());
        assert!(report.recommendations.is_empty());
        assert!((engine.quality_score - 1.0).abs() < 1e-9);
    }

    /// Regression: `generate_detailed_report` ignored its input and emitted the
    /// fixed line "Summary: System performance metrics analyzed".
    #[tokio::test]
    async fn detailed_report_contains_the_statistics_it_was_given() {
        let mut generator = ReportGenerator::new();
        let mut data = HashMap::new();
        data.insert("total_profiles".to_string(), 3.0);
        data.insert("cache_hits".to_string(), 17.0);

        let report =
            generator.generate_detailed_report(&data).await.expect("rendering must succeed");

        assert!(report.contains("cache_hits: 17"), "{report}");
        assert!(report.contains("total_profiles: 3"), "{report}");
        assert!(report.contains("Metrics: 2"), "{report}");
        assert!(
            !report.contains("System performance metrics analyzed"),
            "{report}"
        );
        // Keys render in sorted order, so the same statistics always render
        // identically.
        assert!(
            report.find("cache_hits").unwrap_or(0) < report.find("total_profiles").unwrap_or(0)
        );
    }

    /// Regression: `analyze_results` returned mean 100 / stddev 10 / median 98
    /// for every input, including an empty one.
    #[tokio::test]
    async fn summary_statistics_report_that_they_have_no_sample() {
        let mut engine = StatisticalAnalysisEngine::new();
        let err = engine
            .analyze_results(&HashMap::new())
            .await
            .expect_err("profile results are not a numeric sample");
        assert!(err.to_string().contains("summary statistics"), "{err}");
    }
}
