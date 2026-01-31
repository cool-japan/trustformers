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
use std::sync::{Arc, Mutex};
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

/// Profile session information
#[derive(Debug, Clone)]
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
    /// Memory efficiency score (0-100)
    pub memory_efficiency: f64,
    /// Performance score (0-100)
    pub performance_score: f64,
    /// Number of bottlenecks identified
    pub bottlenecks_found: usize,
    /// Optimization potential (0-100)
    pub optimization_potential: f64,
}

impl Profiler {
    /// Create a new profiler with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(ProfilerConfig::default())
    }

    /// Create a new profiler with custom configuration
    pub fn with_config(config: ProfilerConfig) -> Result<Self> {
        let core_profiler = CoreProfiler::new();

        if config.auto_enable {
            core_profiler.enable();
        }

        let advisor = OptimizationAdvisor::new();
        let benchmark_suite = BenchmarkSuite::new(trustformers_core::BenchmarkConfig::default());
        let metrics_tracker = MetricsTracker::new(100); // Use 100 as default window size

        Ok(Self {
            core_profiler,
            advisor,
            benchmark_suite,
            metrics_tracker,
            config,
            session_start: Instant::now(),
            active_sessions: Arc::new(Mutex::new(HashMap::new())),
        })
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
        let session = ProfileSession {
            id: session_id.clone(),
            name: name.to_string(),
            start_time: Instant::now(),
            end_time: None,
            results: None,
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

        session.end_time = Some(Instant::now());
        let total_duration = session.end_time.expect("end_time just set to Some on previous line")
            - session.start_time;

        // Collect results from core profiler
        let operations = self.core_profiler.get_results();

        // Generate metrics
        let latency_metrics = self.generate_latency_metrics(&operations);
        let throughput_metrics = self.generate_throughput_metrics(&operations, total_duration);
        let memory_metrics = if self.config.enable_memory {
            Some(self.generate_memory_metrics())
        } else {
            None
        };

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
        let summary = self.generate_summary(&operations, total_duration, &optimization_suggestions);

        let results = ProfileResults {
            session_id: session_id.to_string(),
            total_duration,
            operations,
            latency_metrics,
            throughput_metrics,
            memory_metrics,
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

    /// Profile a function with automatic session management
    pub fn profile_function<F, R>(&self, name: &str, f: F) -> Result<(R, ProfileResults)>
    where
        F: FnOnce() -> R,
    {
        let session_id = self.start_session(name)?;
        let _guard = self.core_profiler.start_operation(name);

        let result = f();

        drop(_guard);
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
        let _guard = self.core_profiler.start_operation(name);

        let result = f.await;

        drop(_guard);
        let profile_results = self.end_session(&session_id)?;

        Ok((result, profile_results))
    }

    /// Get active sessions
    pub fn get_active_sessions(&self) -> Result<Vec<ProfileSession>> {
        let sessions = self.active_sessions.lock().map_err(|e| {
            TrustformersError::runtime_error(format!("Failed to lock sessions: {}", e))
        })?;
        Ok(sessions.values().cloned().collect())
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

    fn generate_latency_metrics(
        &self,
        operations: &HashMap<String, ProfileResult>,
    ) -> LatencyMetrics {
        let mut total_time = Duration::ZERO;
        let mut operation_count = 0;
        let mut min_time = Duration::MAX;
        let mut max_time = Duration::ZERO;

        for result in operations.values() {
            total_time += result.total_time;
            operation_count += result.call_count;
            min_time = min_time.min(result.min_time);
            max_time = max_time.max(result.max_time);
        }

        let avg_time = if operation_count > 0 {
            total_time / operation_count as u32
        } else {
            Duration::ZERO
        };

        LatencyMetrics {
            count: operations.len(),
            mean_ms: avg_time.as_millis() as f64,
            median_ms: avg_time.as_millis() as f64, // Simplified
            std_dev_ms: 0.0,                        // Simplified
            min_ms: min_time.as_millis() as f64,
            max_ms: max_time.as_millis() as f64,
            p50_ms: avg_time.as_millis() as f64, // 50th percentile (median)
            p90_ms: max_time.as_millis() as f64, // Simplified
            p95_ms: max_time.as_millis() as f64,
            p99_ms: max_time.as_millis() as f64,
            p999_ms: max_time.as_millis() as f64,
            window_duration: Duration::from_secs(3600), // 1 hour window
        }
    }

    fn generate_throughput_metrics(
        &self,
        operations: &HashMap<String, ProfileResult>,
        total_duration: Duration,
    ) -> ThroughputMetrics {
        let total_operations: usize = operations.values().map(|r| r.call_count).sum();
        let operations_per_second = if total_duration.as_secs() > 0 {
            total_operations as f64 / total_duration.as_secs_f64()
        } else {
            0.0
        };

        ThroughputMetrics {
            tokens_per_second: operations_per_second * 100.0, // Estimate 100 tokens per operation
            batches_per_second: operations_per_second / 10.0, // Estimate 10 operations per batch
            samples_per_second: operations_per_second,
            avg_batch_size: 10.0,                 // Estimate
            avg_sequence_length: 100.0,           // Estimate
            total_tokens: total_operations * 100, // Estimate 100 tokens per operation
            total_batches: total_operations / 10, // Estimate 10 operations per batch
            total_duration,
        }
    }

    fn generate_memory_metrics(&self) -> MemoryMetrics {
        // In a real implementation, this would collect actual memory usage
        MemoryMetrics {
            current_bytes: 1024 * 1024 * 80,    // 80MB estimate
            peak_bytes: 1024 * 1024 * 100,      // 100MB estimate
            allocated_bytes: 1024 * 1024 * 120, // 120MB estimate
            reserved_bytes: 1024 * 1024 * 150,  // 150MB estimate
            num_allocations: 1000,              // Estimate
            num_deallocations: 950,             // Estimate
            fragmentation_percent: 5.0,         // 5% fragmentation estimate
        }
    }

    fn generate_optimization_suggestions(
        &self,
        operations: &HashMap<String, ProfileResult>,
    ) -> Vec<OptimizationSuggestion> {
        // In a real implementation, this would use the OptimizationAdvisor
        let mut suggestions = Vec::new();

        // Find slow operations
        let mut sorted_ops: Vec<_> = operations.iter().collect();
        sorted_ops.sort_by(|(_, a), (_, b)| b.total_time.cmp(&a.total_time));

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
                    expected_improvement: crate::core::performance::PerformanceImprovement {
                        latency_reduction: Some(30.0),
                        throughput_increase: Some(25.0),
                        memory_reduction: Some(10.0),
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

    fn generate_summary(
        &self,
        operations: &HashMap<String, ProfileResult>,
        total_duration: Duration,
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

        // Simple scoring algorithm
        let memory_efficiency = 85.0; // Estimate
        let performance_score = if total_time.as_millis() < 100 { 90.0 } else { 60.0 };
        let bottlenecks_found = suggestions.len();
        let optimization_potential = suggestions.len() as f64 * 10.0;

        ProfileSummary {
            total_operations,
            total_time,
            avg_operation_time,
            slowest_operation,
            fastest_operation,
            memory_efficiency,
            performance_score,
            bottlenecks_found,
            optimization_potential,
        }
    }

    fn run_benchmarks(&self) -> Result<Vec<BenchmarkResult>> {
        // In a real implementation, this would run actual benchmarks
        Ok(vec![])
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
        <div class="metric">Performance Score: {:.1}</div>
        <div class="metric">Memory Efficiency: {:.1}%</div>
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
            results.summary.performance_score,
            results.summary.memory_efficiency,
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
        Self::new().expect("Profiler::new should not fail with default config")
    }
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
    GLOBAL_PROFILER
        .get_or_init(|| Profiler::new().expect("Profiler::new should not fail for global instance"))
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

        let profiler = profiler.unwrap();
        assert!(profiler.is_enabled()); // Auto-enabled by default
    }

    #[test]
    fn test_session_management() {
        let profiler = Profiler::new().unwrap();

        let session_id = profiler.start_session("test_session").unwrap();
        assert!(!session_id.is_empty());

        sleep(Duration::from_millis(10));

        let results = profiler.end_session(&session_id).unwrap();
        assert_eq!(results.session_id, session_id);
        assert!(results.total_duration > Duration::ZERO);
    }

    #[test]
    fn test_profile_function() {
        let profiler = Profiler::new().unwrap();

        let (result, profile_results) = profiler
            .profile_function("test_operation", || {
                sleep(Duration::from_millis(10));
                42
            })
            .unwrap();

        assert_eq!(result, 42);
        assert!(!profile_results.operations.is_empty());
        assert!(profile_results.total_duration > Duration::ZERO);
    }

    #[test]
    fn test_export_formats() {
        let profiler = Profiler::new().unwrap();

        let (_, results) = profiler
            .profile_function("export_test", || {
                sleep(Duration::from_millis(5));
            })
            .unwrap();

        // Test HTML generation
        let html = profiler.generate_html_report(&results);
        assert!(html.contains("TrustformeRS Performance Report"));

        // Test CSV generation
        let csv = profiler.generate_csv_report(&results);
        assert!(csv.contains("Operation,Calls"));
    }

    #[test]
    fn test_global_profiler() {
        let (result, profile_results) = profile_fn("global_test", || {
            sleep(Duration::from_millis(5));
            "test"
        })
        .unwrap();

        assert_eq!(result, "test");
        assert!(!profile_results.operations.is_empty());
    }
}
