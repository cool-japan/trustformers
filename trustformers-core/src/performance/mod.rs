//! Performance benchmarking and profiling infrastructure

pub mod benchmark;
pub mod comparison;
pub mod continuous;
pub mod custom_benchmarks;
pub mod memory_profiler;
pub mod metrics;
pub mod optimization_advisor;
pub mod profiler;
pub mod regression_detector;

pub use benchmark::{BenchmarkConfig, BenchmarkResult, BenchmarkSuite};
pub use comparison::{
    ComparisonResult, Framework, HuggingFaceBenchmark, ModelComparison, PytorchBenchmark,
};
pub use continuous::{ContinuousBenchmark, ContinuousBenchmarkConfig, PerformanceRegression};
pub use custom_benchmarks::{
    BenchmarkBuilder, BenchmarkCategory, BenchmarkDSL, BenchmarkMetadata, BenchmarkRegistry,
    BenchmarkReport, BenchmarkRunner, BenchmarkRunnerBuilder, BenchmarkSpec, CustomBenchmark,
    ReportFormat, Reporter, RunConfig, RunMode,
};
pub use memory_profiler::{MemoryProfiler, MemorySnapshot, MemoryTracker};
pub use metrics::{LatencyMetrics, MemoryMetrics, MetricsTracker, ThroughputMetrics};
pub use optimization_advisor::{
    AnalysisContext, CodeExample, Difficulty, HardwareInfo, ImpactLevel, OptimizationAdvisor,
    OptimizationCategory, OptimizationReport, OptimizationRule, OptimizationSuggestion,
    OptimizationSummary, PerformanceImprovement,
};
pub use profiler::{PerformanceProfiler, ProfileResult};
pub use regression_detector::{
    HardwareConfig, PerformanceBaseline, PerformanceMeasurement, RegressionConfig,
    RegressionDetector, RegressionResult,
};
