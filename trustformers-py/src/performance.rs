use numpy::{PyArray, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use scirs2_core::ndarray::ArrayD;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Performance profiler for tensor operations
#[pyclass]
pub struct PerformanceProfiler {
    measurements: HashMap<String, Vec<Duration>>,
    memory_usage: HashMap<String, usize>,
    start_times: HashMap<String, Instant>,
}

#[pymethods]
impl PerformanceProfiler {
    #[new]
    pub fn new() -> Self {
        Self {
            measurements: HashMap::new(),
            memory_usage: HashMap::new(),
            start_times: HashMap::new(),
        }
    }

    /// Start timing an operation
    pub fn start_timer(&mut self, operation: String) {
        self.start_times.insert(operation, Instant::now());
    }

    /// Stop timing an operation and record the result
    pub fn stop_timer(&mut self, operation: String) -> PyResult<f64> {
        if let Some(start_time) = self.start_times.remove(&operation) {
            let duration = start_time.elapsed();
            let duration_ms = duration.as_secs_f64() * 1000.0;

            self.measurements.entry(operation).or_insert_with(Vec::new).push(duration);

            Ok(duration_ms)
        } else {
            Err(PyValueError::new_err(format!(
                "No timer started for operation: {}",
                operation
            )))
        }
    }

    /// Record memory usage for an operation
    pub fn record_memory(&mut self, operation: String, bytes: usize) {
        self.memory_usage.insert(operation, bytes);
    }

    /// Get timing statistics for an operation
    pub fn get_stats(&self, py: Python<'_>, operation: String) -> PyResult<PyObject> {
        if let Some(measurements) = self.measurements.get(&operation) {
            if measurements.is_empty() {
                return Ok(py.None());
            }

            let durations_ms: Vec<f64> =
                measurements.iter().map(|d| d.as_secs_f64() * 1000.0).collect();

            let count = durations_ms.len();
            let sum = durations_ms.iter().sum::<f64>();
            let mean = sum / count as f64;

            let mut sorted = durations_ms.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let median = if count % 2 == 0 {
                (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
            } else {
                sorted[count / 2]
            };

            let min = sorted[0];
            let max = sorted[count - 1];

            let variance =
                durations_ms.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / count as f64;
            let std_dev = variance.sqrt();

            let stats = PyDict::new(py);
            stats.set_item("operation", &operation)?;
            stats.set_item("count", count)?;
            stats.set_item("mean_ms", mean)?;
            stats.set_item("median_ms", median)?;
            stats.set_item("min_ms", min)?;
            stats.set_item("max_ms", max)?;
            stats.set_item("std_dev_ms", std_dev)?;
            stats.set_item("total_ms", sum)?;

            if let Some(&memory_bytes) = self.memory_usage.get(&operation) {
                stats.set_item("memory_bytes", memory_bytes)?;
                stats.set_item("memory_mb", memory_bytes as f64 / (1024.0 * 1024.0))?;
            }

            Ok(stats.into_any().unbind())
        } else {
            Ok(py.None())
        }
    }

    /// Get all recorded statistics
    pub fn get_all_stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let all_stats = PyDict::new(py);

        for operation in self.measurements.keys() {
            if let Ok(stats) = self.get_stats(py, operation.clone()) {
                if !stats.is_none(py) {
                    all_stats.set_item(operation, stats)?;
                }
            }
        }

        Ok(all_stats.into_any().unbind())
    }

    /// Clear all measurements
    pub fn clear(&mut self) {
        self.measurements.clear();
        self.memory_usage.clear();
        self.start_times.clear();
    }

    /// Benchmark a tensor operation
    #[staticmethod]
    pub fn benchmark_operation(
        py: Python<'_>,
        operation_name: String,
        tensor: &Bound<'_, PyArray<f32, scirs2_core::ndarray::IxDyn>>,
        iterations: usize,
    ) -> PyResult<PyObject> {
        let array = tensor.try_readonly()?.as_array().to_owned();
        let mut durations = Vec::with_capacity(iterations);

        // Warm up
        for _ in 0..std::cmp::min(iterations / 10, 10) {
            let _ = Self::perform_benchmark_operation(&operation_name, &array)?;
        }

        // Actual benchmarking
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = Self::perform_benchmark_operation(&operation_name, &array)?;
            durations.push(start.elapsed());
        }

        let durations_ms: Vec<f64> = durations.iter().map(|d| d.as_secs_f64() * 1000.0).collect();

        let count = durations_ms.len();
        let sum = durations_ms.iter().sum::<f64>();
        let mean = sum / count as f64;

        let mut sorted = durations_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if count % 2 == 0 {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            sorted[count / 2]
        };

        let p95_idx = (count as f64 * 0.95).ceil() as usize - 1;
        let p95 = sorted[p95_idx];

        let p99_idx = (count as f64 * 0.99).ceil() as usize - 1;
        let p99 = sorted[p99_idx];

        // Calculate throughput
        let elements = array.len();
        let throughput_elements_per_sec = elements as f64 / (mean / 1000.0);

        let result = PyDict::new(py);
        result.set_item("operation", operation_name)?;
        result.set_item("iterations", iterations)?;
        result.set_item("tensor_shape", array.shape().to_vec())?;
        result.set_item("elements", elements)?;
        result.set_item("mean_ms", mean)?;
        result.set_item("median_ms", median)?;
        result.set_item("min_ms", sorted[0])?;
        result.set_item("max_ms", sorted[count - 1])?;
        result.set_item("p95_ms", p95)?;
        result.set_item("p99_ms", p99)?;
        result.set_item("throughput_elements_per_sec", throughput_elements_per_sec)?;

        Ok(result.into_any().unbind())
    }

    /// Compare performance of multiple operations
    #[staticmethod]
    pub fn compare_operations(
        py: Python<'_>,
        operations: Vec<String>,
        tensor: &Bound<'_, PyArray<f32, scirs2_core::ndarray::IxDyn>>,
        iterations: usize,
    ) -> PyResult<PyObject> {
        let mut results = Vec::new();

        for operation in operations {
            let benchmark_result =
                Self::benchmark_operation(py, operation.clone(), tensor, iterations)?;
            results.push(benchmark_result);
        }

        let result_list = PyList::new(py, &results)?;
        Ok(result_list.into_any().unbind())
    }
}

impl PerformanceProfiler {
    fn perform_benchmark_operation(
        operation_name: &str,
        array: &ArrayD<f32>,
    ) -> PyResult<ArrayD<f32>> {
        match operation_name {
            "softmax" => {
                // Simplified softmax for benchmarking
                let max_val = array.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_vals = array.mapv(|x| (x - max_val).exp());
                let sum = exp_vals.sum();
                Ok(exp_vals.mapv(|x| x / sum))
            },
            "relu" => Ok(array.mapv(|x| x.max(0.0))),
            "gelu" => Ok(array
                .mapv(|x| 0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh()))),
            "exp" => Ok(array.mapv(|x| x.exp())),
            "log" => Ok(array.mapv(|x| x.ln())),
            "sqrt" => Ok(array.mapv(|x| x.sqrt())),
            "sin" => Ok(array.mapv(|x| x.sin())),
            "cos" => Ok(array.mapv(|x| x.cos())),
            "tanh" => Ok(array.mapv(|x| x.tanh())),
            "sigmoid" => Ok(array.mapv(|x| 1.0 / (1.0 + (-x).exp()))),
            _ => Ok(array.clone()), // Default: no-op
        }
    }
}

/// Memory tracker for monitoring memory usage
#[pyclass]
pub struct MemoryTracker {
    peak_usage: usize,
    current_usage: usize,
    allocations: Vec<(String, usize, Instant)>,
}

#[pymethods]
impl MemoryTracker {
    #[new]
    pub fn new() -> Self {
        Self {
            peak_usage: 0,
            current_usage: 0,
            allocations: Vec::new(),
        }
    }

    /// Record a memory allocation
    pub fn allocate(&mut self, name: String, bytes: usize) {
        self.current_usage += bytes;
        self.peak_usage = self.peak_usage.max(self.current_usage);
        self.allocations.push((name, bytes, Instant::now()));
    }

    /// Record a memory deallocation
    pub fn deallocate(&mut self, bytes: usize) {
        self.current_usage = self.current_usage.saturating_sub(bytes);
    }

    /// Get current memory usage
    pub fn current_usage_mb(&self) -> f64 {
        self.current_usage as f64 / (1024.0 * 1024.0)
    }

    /// Get peak memory usage
    pub fn peak_usage_mb(&self) -> f64 {
        self.peak_usage as f64 / (1024.0 * 1024.0)
    }

    /// Get memory usage summary
    pub fn get_summary(&self, py: Python<'_>) -> PyResult<PyObject> {
        let summary = PyDict::new(py);
        summary.set_item("current_usage_bytes", self.current_usage)?;
        summary.set_item("current_usage_mb", self.current_usage_mb())?;
        summary.set_item("peak_usage_bytes", self.peak_usage)?;
        summary.set_item("peak_usage_mb", self.peak_usage_mb())?;
        summary.set_item("total_allocations", self.allocations.len())?;

        Ok(summary.into_any().unbind())
    }

    /// Reset all tracking
    pub fn reset(&mut self) {
        self.peak_usage = 0;
        self.current_usage = 0;
        self.allocations.clear();
    }
}

/// Context manager for automatic performance profiling
#[pyclass]
pub struct ProfilerContext {
    profiler: Py<PerformanceProfiler>,
    operation: String,
}

#[pymethods]
impl ProfilerContext {
    #[new]
    pub fn new(profiler: Py<PerformanceProfiler>, operation: String) -> Self {
        Self {
            profiler,
            operation,
        }
    }

    pub fn __enter__(&mut self, py: Python<'_>) -> PyResult<()> {
        self.profiler.borrow_mut(py).start_timer(self.operation.clone());
        Ok(())
    }

    pub fn __exit__(
        &mut self,
        py: Python<'_>,
        _exc_type: Option<PyObject>,
        _exc_value: Option<PyObject>,
        _traceback: Option<PyObject>,
    ) -> PyResult<bool> {
        let _ = self.profiler.borrow_mut(py).stop_timer(self.operation.clone());
        Ok(false) // Don't suppress exceptions
    }
}

/// Utility functions for performance analysis
#[pyclass]
pub struct PerformanceUtils;

#[pymethods]
impl PerformanceUtils {
    /// Estimate tensor memory usage
    #[staticmethod]
    pub fn estimate_memory_usage(shape: Vec<usize>, dtype: Option<String>) -> PyResult<usize> {
        let elements: usize = shape.iter().product();
        let dtype_str = dtype.unwrap_or_else(|| "float32".to_string());

        let bytes_per_element = match dtype_str.as_str() {
            "float32" | "f32" => 4,
            "float64" | "f64" => 8,
            "float16" | "f16" => 2,
            "int32" | "i32" => 4,
            "int64" | "i64" => 8,
            "int16" | "i16" => 2,
            "int8" | "i8" => 1,
            "uint8" | "u8" => 1,
            _ => 4, // Default to f32
        };

        Ok(elements * bytes_per_element)
    }

    /// Calculate theoretical FLOPS for an operation
    #[staticmethod]
    pub fn calculate_flops(operation: String, shape: Vec<usize>) -> PyResult<usize> {
        let elements: usize = shape.iter().product();

        let flops = match operation.as_str() {
            "matmul" => {
                if shape.len() >= 2 {
                    // For matrix multiplication: 2 * m * n * k (assuming last two dims are m, k and k, n)
                    let m = shape[shape.len() - 2];
                    let n = shape[shape.len() - 1];
                    let k = shape[shape.len() - 1]; // Simplified assumption
                    2 * m * n * k
                } else {
                    elements
                }
            },
            "conv2d" => {
                // Simplified: assume 3x3 kernel
                elements * 9 * 2 // 9 multiplications + 9 additions per output element
            },
            "softmax" => {
                elements * 4 // exp, sum, divide, subtract
            },
            "layernorm" => {
                elements * 6 // mean, variance, normalize
            },
            "gelu" | "silu" | "mish" => {
                elements * 8 // Multiple transcendental operations
            },
            "relu" | "sigmoid" | "tanh" => {
                elements * 2 // Simple operations
            },
            _ => elements, // Default: one operation per element
        };

        Ok(flops)
    }

    /// Estimate optimal batch size for memory constraints
    #[staticmethod]
    pub fn estimate_optimal_batch_size(
        tensor_shape: Vec<usize>,
        memory_limit_mb: f64,
        safety_factor: Option<f64>,
    ) -> PyResult<usize> {
        let safety = safety_factor.unwrap_or(0.8); // Use 80% of available memory by default
        let memory_limit_bytes = (memory_limit_mb * 1024.0 * 1024.0 * safety) as usize;

        let single_tensor_size = Self::estimate_memory_usage(tensor_shape.clone(), None)?;

        if single_tensor_size == 0 {
            return Ok(1);
        }

        let max_batch_size = memory_limit_bytes / single_tensor_size;
        Ok(max_batch_size.max(1))
    }
}
