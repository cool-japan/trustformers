//! Type definitions for testing results and measurements

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Test results for numerical parity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalParityResults {
    /// Whether all tests passed
    pub all_passed: bool,
    /// Individual test results
    pub test_results: Vec<TestResult>,
    /// Overall statistics
    pub statistics: TestStatistics,
    /// Timing information
    pub timing: TimingInfo,
}

/// Individual test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// Test name
    pub name: String,
    /// Whether test passed
    pub passed: bool,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Numerical differences found
    pub numerical_differences: Option<NumericalDifferences>,
    /// Execution time
    pub execution_time: Duration,
}

/// Numerical differences between expected and actual outputs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalDifferences {
    /// Maximum absolute difference
    pub max_abs_diff: f32,
    /// Mean absolute difference
    pub mean_abs_diff: f32,
    /// Root mean square difference
    pub rms_diff: f32,
    /// Percentage of values within tolerance
    pub within_tolerance_percent: f32,
}

/// Overall test statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestStatistics {
    /// Total tests run
    pub total_tests: usize,
    /// Tests passed
    pub passed_tests: usize,
    /// Tests failed
    pub failed_tests: usize,
    /// Pass rate percentage
    pub pass_rate: f32,
}

/// Timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingInfo {
    /// Total test execution time
    pub total_time: Duration,
    /// Average time per test
    pub average_time: Duration,
    /// Fastest test time
    pub fastest_time: Duration,
    /// Slowest test time
    pub slowest_time: Duration,
}

/// Performance profiling results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceResults {
    /// Layer-wise performance breakdown
    pub layer_performance: Vec<LayerPerformance>,
    /// Overall model performance
    pub overall_performance: OverallPerformance,
    /// Memory usage analysis
    pub memory_analysis: MemoryAnalysis,
    /// Throughput measurements
    pub throughput: ThroughputMeasurements,
}

/// Measured performance of one profiled unit of work.
///
/// Without layer-level instrumentation the profiler measures whole forward
/// passes, so `layer_name` identifies the profiled input and `layer_type` says
/// what was measured. Fields with no measurement source are `None`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerPerformance {
    /// Layer name/identifier
    pub layer_name: String,
    /// Layer type (attention, mlp, etc.)
    pub layer_type: String,
    /// Measured wall-clock time of the forward pass
    pub forward_time: Duration,
    /// Resident-set growth observed across the measurement, in MB.
    ///
    /// `None` when the platform cannot report process memory. This is a *delta*
    /// of the process RSS, not an allocation trace, so it can legitimately be 0.
    pub memory_usage_mb: Option<f64>,
    /// FLOPS (floating point operations per second). `None`: not instrumented.
    pub flops: Option<f64>,
    /// Utilization percentage. `None`: not instrumented.
    pub utilization_percent: Option<f32>,
}

/// Overall model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallPerformance {
    /// Total measured inference time
    pub total_inference_time: Duration,
    /// Tokens processed per second, derived from the measured time and the real
    /// token count of the profiled inputs.
    pub tokens_per_second: f32,
    /// Total FLOPS. `None`: not instrumented.
    pub total_flops: Option<f64>,
    /// Highest process RSS observed during profiling, in MB (`None` when the
    /// platform cannot report it).
    pub peak_memory_mb: Option<f64>,
    /// Mean process RSS observed during profiling, in MB.
    pub average_memory_mb: Option<f64>,
}

/// Memory usage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAnalysis {
    /// Memory usage by layer type
    pub by_layer_type: HashMap<String, f64>,
    /// Memory usage by tensor type
    pub by_tensor_type: HashMap<String, f64>,
    /// Memory efficiency score (0-100). `None`: needs allocator instrumentation
    /// this profiler does not have.
    pub efficiency_score: Option<f32>,
    /// Memory fragmentation percentage. `None`: needs allocator instrumentation.
    pub fragmentation_percent: Option<f32>,
}

/// Throughput measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMeasurements {
    /// Batch size used for measurements
    pub batch_size: usize,
    /// Sequence length used
    pub sequence_length: usize,
    /// Throughput in tokens/second
    pub tokens_per_second: f32,
    /// Throughput in samples/second
    pub samples_per_second: f32,
    /// Latency per token (milliseconds)
    pub latency_per_token_ms: f32,
}
