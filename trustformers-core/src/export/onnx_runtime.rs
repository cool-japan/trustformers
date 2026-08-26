//! ONNX inference sessions backed by this crate's own CPU interpreter.
//!
//! There is no binding to Microsoft's ONNX Runtime here — that would be a C++
//! dependency, which the pure-Rust policy forbids. What this module provides is
//! real: [`ONNXRuntimeBackend::load_model`] parses the actual `.onnx` protobuf with
//! [`super::onnx_proto`], and [`ONNXRuntimeSession::run`] executes the graph with
//! [`super::onnx_cpu`]. Operators the interpreter does not implement produce a
//! structured
//! [`ErrorKind::UnsupportedOperation`](crate::errors::ErrorKind::UnsupportedOperation)
//! error naming the operator.
//!
//! An earlier revision of this module returned `thread_rng()` numbers shaped by
//! filename heuristics (`model_path.contains("gpt2") => 50257`) and timed that RNG
//! as "ONNX Runtime performance". Nothing of the sort remains: `run` computes,
//! `benchmark` times real computation, and the optimizer transforms the graph or
//! fails.

use super::onnx::ONNXExporter;
use super::onnx_cpu::{CpuTensor, OnnxGraphExecutor};
use super::onnx_proto::{decode_model, encode_model};
use crate::errors::unsupported_operation;
use crate::tensor::Tensor;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::path::Path;

pub use super::onnx_optimize::{quantize_onnx_model, GraphOptimizationStats, QuantizationStats};

/// A loaded ONNX model ready for CPU execution.
#[derive(Debug)]
pub struct ONNXRuntimeSession {
    session_config: ONNXRuntimeConfig,
    executor: OnnxGraphExecutor,
    providers: Vec<ExecutionProvider>,
    model_path: String,
    initializer_bytes: usize,
}

/// Configuration for a session.
#[derive(Debug, Clone)]
pub struct ONNXRuntimeConfig {
    pub inter_op_num_threads: Option<usize>,
    pub intra_op_num_threads: Option<usize>,
    pub enable_cpu_mem_arena: bool,
    pub enable_mem_pattern: bool,
    pub execution_mode: ExecutionMode,
    pub graph_optimization_level: GraphOptimizationLevel,
    pub log_severity_level: LogLevel,
}

impl Default for ONNXRuntimeConfig {
    fn default() -> Self {
        Self {
            inter_op_num_threads: None,
            intra_op_num_threads: None,
            enable_cpu_mem_arena: true,
            enable_mem_pattern: true,
            execution_mode: ExecutionMode::Sequential,
            graph_optimization_level: GraphOptimizationLevel::All,
            log_severity_level: LogLevel::Warning,
        }
    }
}

/// Execution providers. Only [`ExecutionProvider::CPU`] can actually run a graph
/// in this build; the others exist so callers can describe a target and be told
/// clearly that it is unavailable.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExecutionProvider {
    CPU,
    CUDA { device_id: Option<i32> },
    TensorRT { device_id: Option<i32> },
    OpenVINO,
    DirectML,
    CoreML,
}

/// Execution modes.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ExecutionMode {
    Sequential,
    Parallel,
}

/// Graph optimization levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GraphOptimizationLevel {
    None,
    Basic,
    Extended,
    All,
}

/// Logging levels.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum LogLevel {
    Verbose,
    Info,
    Warning,
    Error,
    Fatal,
}

/// Loads ONNX models into [`ONNXRuntimeSession`]s.
pub struct ONNXRuntimeBackend {
    config: ONNXRuntimeConfig,
}

impl Default for ONNXRuntimeBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl ONNXRuntimeBackend {
    /// Create a backend with the default configuration.
    pub fn new() -> Self {
        Self {
            config: ONNXRuntimeConfig::default(),
        }
    }

    /// Create a backend with an explicit configuration.
    pub fn with_config(config: ONNXRuntimeConfig) -> Self {
        Self { config }
    }

    /// The configuration sessions will be created with.
    pub fn config(&self) -> &ONNXRuntimeConfig {
        &self.config
    }

    /// Parse an `.onnx` file and prepare it for execution.
    ///
    /// The file must be a real binary `onnx.ModelProto`; anything else fails here
    /// rather than at inference time.
    pub fn load_model<P: AsRef<Path>>(&self, model_path: P) -> Result<ONNXRuntimeSession> {
        let model_path = model_path.as_ref();

        if !model_path.exists() {
            return Err(anyhow!("ONNX model file not found: {:?}", model_path));
        }

        let bytes = std::fs::read(model_path)?;
        let model = decode_model(&bytes).map_err(|e| {
            anyhow!(
                "{:?} is not a parsable ONNX model: {e}. TrustformeRS reads binary \
                 `onnx.ModelProto` files only.",
                model_path
            )
        })?;

        let initializer_bytes =
            model.graph.initializers.iter().map(|tensor| tensor.raw_data.len()).sum();
        let executor = OnnxGraphExecutor::new(model)?;

        let missing = executor.unsupported_operators();
        if !missing.is_empty() {
            log::warn!(
                "ONNX model {:?} uses operators this build cannot execute: {}",
                model_path,
                missing.join(", ")
            );
        }

        Ok(ONNXRuntimeSession {
            session_config: self.config.clone(),
            executor,
            providers: self.get_available_providers(),
            model_path: model_path.to_string_lossy().to_string(),
            initializer_bytes,
        })
    }

    /// Execution providers this build can actually use.
    ///
    /// Only the CPU provider is listed, because only the CPU interpreter exists.
    /// Reporting CUDA because `/usr/local/cuda` happens to be present, as an
    /// earlier revision did, would promise an accelerator that never runs.
    pub fn get_available_providers(&self) -> Vec<ExecutionProvider> {
        vec![ExecutionProvider::CPU]
    }

    /// Build session options from this backend's configuration.
    pub fn create_session_options(&self) -> ONNXSessionOptions {
        ONNXSessionOptions {
            execution_providers: self.get_available_providers(),
            inter_op_num_threads: self.config.inter_op_num_threads,
            intra_op_num_threads: self.config.intra_op_num_threads,
            enable_cpu_mem_arena: self.config.enable_cpu_mem_arena,
            enable_mem_pattern: self.config.enable_mem_pattern,
            execution_mode: self.config.execution_mode.clone(),
            graph_optimization_level: self.config.graph_optimization_level,
            log_severity_level: self.config.log_severity_level.clone(),
        }
    }
}

/// Session options.
#[derive(Debug, Clone)]
pub struct ONNXSessionOptions {
    pub execution_providers: Vec<ExecutionProvider>,
    pub inter_op_num_threads: Option<usize>,
    pub intra_op_num_threads: Option<usize>,
    pub enable_cpu_mem_arena: bool,
    pub enable_mem_pattern: bool,
    pub execution_mode: ExecutionMode,
    pub graph_optimization_level: GraphOptimizationLevel,
    pub log_severity_level: LogLevel,
}

impl ONNXRuntimeSession {
    /// Run inference.
    ///
    /// Every value returned is computed from the inputs by the graph's operators.
    pub fn run(&self, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        let mut cpu_inputs = HashMap::with_capacity(inputs.len());
        for (name, tensor) in inputs {
            cpu_inputs.insert(name, CpuTensor::from_tensor(&tensor)?);
        }

        let cpu_outputs = self.executor.run(cpu_inputs)?;

        let mut outputs = HashMap::with_capacity(cpu_outputs.len());
        for (name, value) in cpu_outputs {
            outputs.insert(name, value.into_tensor()?);
        }
        Ok(outputs)
    }

    /// The session's configuration.
    pub fn config(&self) -> &ONNXRuntimeConfig {
        &self.session_config
    }

    /// Names the caller must supply to [`Self::run`].
    pub fn input_names(&self) -> &[String] {
        self.executor.input_names()
    }

    /// Names [`Self::run`] returns.
    pub fn output_names(&self) -> &[String] {
        self.executor.output_names()
    }

    /// Path the model was loaded from.
    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    /// Execution providers available to this session.
    pub fn execution_providers(&self) -> &[ExecutionProvider] {
        &self.providers
    }

    /// The parsed graph.
    pub fn executor(&self) -> &OnnxGraphExecutor {
        &self.executor
    }

    /// Operators in this graph that the CPU interpreter cannot execute.
    pub fn unsupported_operators(&self) -> Vec<String> {
        self.executor.unsupported_operators()
    }

    /// Run inference on a named execution provider.
    ///
    /// Only [`ExecutionProvider::CPU`] runs; anything else fails with a structured
    /// error rather than silently falling back and reporting accelerator numbers.
    pub fn run_with_provider(
        &self,
        inputs: HashMap<String, Tensor>,
        provider: ExecutionProvider,
    ) -> Result<HashMap<String, Tensor>> {
        match provider {
            ExecutionProvider::CPU => self.run(inputs),
            other => Err(unsupported_operation(
                format!("ONNX inference on the {other:?} execution provider"),
                "this build executes ONNX graphs with a pure-Rust CPU interpreter; no \
                 accelerator execution provider is linked in",
            )
            .into()),
        }
    }

    /// Measure real end-to-end latency of [`Self::run`].
    pub fn benchmark(
        &self,
        inputs: HashMap<String, Tensor>,
        num_runs: usize,
    ) -> Result<BenchmarkResults> {
        if num_runs == 0 {
            return Err(anyhow!("benchmark needs at least one run"));
        }

        let mut latencies = Vec::with_capacity(num_runs);
        for _ in 0..num_runs {
            let start = std::time::Instant::now();
            let outputs = self.run(inputs.clone())?;
            let duration = start.elapsed();
            // Touch the result so the work cannot be optimized away.
            debug_assert!(!outputs.is_empty());
            latencies.push(duration.as_secs_f64() * 1000.0);
        }

        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let percentile = |fraction: f64| -> f64 {
            let index = ((latencies.len() as f64 * fraction) as usize).min(latencies.len() - 1);
            latencies[index]
        };

        Ok(BenchmarkResults {
            num_runs,
            mean_latency_ms: latencies.iter().sum::<f64>() / latencies.len() as f64,
            median_latency_ms: percentile(0.5),
            p90_latency_ms: percentile(0.9),
            p95_latency_ms: percentile(0.95),
            p99_latency_ms: percentile(0.99),
            min_latency_ms: latencies[0],
            max_latency_ms: latencies[latencies.len() - 1],
        })
    }

    /// Real memory figures: the model's own weight bytes plus the host's memory.
    pub fn get_memory_info(&self) -> Result<MemoryInfo> {
        let mut system = sysinfo::System::new();
        system.refresh_memory();

        Ok(MemoryInfo {
            total_memory_bytes: system.total_memory() as usize,
            available_memory_bytes: system.available_memory() as usize,
            model_memory_bytes: self.initializer_bytes,
        })
    }
}

/// Latency statistics from [`ONNXRuntimeSession::benchmark`].
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub num_runs: usize,
    pub mean_latency_ms: f64,
    pub median_latency_ms: f64,
    pub p90_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
}

impl BenchmarkResults {
    /// Print a human-readable summary.
    pub fn print_summary(&self) {
        println!("TrustformeRS CPU ONNX interpreter — benchmark");
        println!("=============================================");
        println!("Number of runs: {}", self.num_runs);
        println!("Mean latency: {:.3} ms", self.mean_latency_ms);
        println!("Median latency: {:.3} ms", self.median_latency_ms);
        println!("P90 latency: {:.3} ms", self.p90_latency_ms);
        println!("P95 latency: {:.3} ms", self.p95_latency_ms);
        println!("P99 latency: {:.3} ms", self.p99_latency_ms);
        println!("Min latency: {:.3} ms", self.min_latency_ms);
        println!("Max latency: {:.3} ms", self.max_latency_ms);
    }
}

/// Memory figures reported by [`ONNXRuntimeSession::get_memory_info`].
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Physical memory installed on the host, as reported by the OS.
    pub total_memory_bytes: usize,
    /// Physical memory the OS reports as available.
    pub available_memory_bytes: usize,
    /// Bytes of tensor data carried by the model's initializers.
    pub model_memory_bytes: usize,
}

/// Graph-level transformations on `.onnx` files.
pub struct ONNXOptimizer;

impl ONNXOptimizer {
    /// Apply real graph optimizations and write the result.
    ///
    /// See [`Self::optimize_model_with_stats`] for what each level does. The
    /// output is always a re-serialised, valid `onnx.ModelProto`; it is never a
    /// byte copy of the input dressed up as an optimization.
    pub fn optimize_model<P: AsRef<Path>, Q: AsRef<Path>>(
        input_path: P,
        output_path: Q,
        optimization_level: GraphOptimizationLevel,
    ) -> Result<()> {
        Self::optimize_model_with_stats(input_path, output_path, optimization_level).map(|_| ())
    }

    /// Apply real graph optimizations and report what changed.
    ///
    /// * [`GraphOptimizationLevel::None`] — parse and re-serialise only. Nothing is
    ///   removed; the report shows zero changes.
    /// * [`GraphOptimizationLevel::Basic`] — eliminate `Identity` nodes by rewiring
    ///   their consumers, then drop initializers no node reads any more.
    /// * [`GraphOptimizationLevel::Extended`] / [`GraphOptimizationLevel::All`] —
    ///   the above plus constant folding: any node whose inputs are all constants
    ///   is evaluated with the CPU interpreter and replaced by an initializer.
    pub fn optimize_model_with_stats<P: AsRef<Path>, Q: AsRef<Path>>(
        input_path: P,
        output_path: Q,
        optimization_level: GraphOptimizationLevel,
    ) -> Result<GraphOptimizationStats> {
        let input_path = input_path.as_ref();
        let output_path = output_path.as_ref();

        if !input_path.exists() {
            return Err(anyhow!("Input ONNX model not found: {:?}", input_path));
        }

        let bytes = std::fs::read(input_path)?;
        let mut model = decode_model(&bytes)
            .map_err(|e| anyhow!("{:?} is not a parsable ONNX model: {e}", input_path))?;

        let stats = super::onnx_optimize::optimize_graph(
            &mut model.graph,
            optimization_level,
            bytes.len(),
        )?;

        let optimized = encode_model(&model);
        std::fs::write(output_path, &optimized)?;

        Ok(GraphOptimizationStats {
            bytes_after: optimized.len(),
            ..stats
        })
    }

    /// Quantize an ONNX model's weights to int8 and write the result.
    ///
    /// [`QuantizationMode::Dynamic`] performs real weight-only int8 quantization:
    /// each large float initializer is replaced by an int8 tensor plus a
    /// `DequantizeLinear` node that restores the original tensor name, so the graph
    /// still computes the same function to within quantization error and the file
    /// really does shrink.
    ///
    /// [`QuantizationMode::Static`] additionally requires activation ranges
    /// measured on calibration data. This function is given no calibration data, so
    /// it returns a structured error instead of pretending.
    pub fn quantize_model<P: AsRef<Path>, Q: AsRef<Path>>(
        input_path: P,
        output_path: Q,
        quantization_mode: QuantizationMode,
    ) -> Result<()> {
        Self::quantize_model_with_stats(input_path, output_path, quantization_mode).map(|_| ())
    }

    /// Quantize an ONNX model's weights and report what changed.
    pub fn quantize_model_with_stats<P: AsRef<Path>, Q: AsRef<Path>>(
        input_path: P,
        output_path: Q,
        quantization_mode: QuantizationMode,
    ) -> Result<QuantizationStats> {
        let input_path = input_path.as_ref();
        let output_path = output_path.as_ref();

        if !input_path.exists() {
            return Err(anyhow!("Input ONNX model not found: {:?}", input_path));
        }

        match quantization_mode {
            QuantizationMode::Dynamic => {},
            QuantizationMode::Static => {
                return Err(unsupported_operation(
                    "static ONNX quantization",
                    "static quantization needs activation ranges measured on calibration data; \
                     none was supplied. Use QuantizationMode::Dynamic for weight-only int8 \
                     quantization, which needs no calibration set.",
                )
                .into())
            },
        }

        let bytes = std::fs::read(input_path)?;
        let mut model = decode_model(&bytes)
            .map_err(|e| anyhow!("{:?} is not a parsable ONNX model: {e}", input_path))?;

        let stats = quantize_onnx_model(&mut model.graph, bytes.len())?;

        // Weight-only int8 quantization needs DequantizeLinear, which entered the
        // default domain at opset 10.
        for opset in &mut model.opset_imports {
            if opset.domain.is_empty() && opset.version < 10 {
                opset.version = 10;
            }
        }

        let encoded = ONNXExporter::new().encode_graph(&model)?;
        std::fs::write(output_path, &encoded)?;

        Ok(QuantizationStats {
            bytes_after: encoded.len(),
            ..stats
        })
    }
}

/// Quantization modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMode {
    Static,
    Dynamic,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::onnx::{
        ONNXDataType, ONNXDimension, ONNXGraph, ONNXNode, ONNXTensor, ONNXTensorShape,
        ONNXTensorType, ONNXTypeInfo, ONNXValueInfo,
    };
    use crate::export::ExportConfig;

    fn temp_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(name);
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    fn value_info(name: &str, dims: &[i64]) -> ONNXValueInfo {
        ONNXValueInfo {
            name: name.to_string(),
            type_info: ONNXTypeInfo {
                tensor_type: ONNXTensorType {
                    elem_type: ONNXDataType::Float,
                    shape: ONNXTensorShape {
                        dims: dims.iter().map(|&d| ONNXDimension::Value(d)).collect(),
                    },
                },
            },
        }
    }

    fn initializer(name: &str, dims: &[i64], values: &[f32]) -> ONNXTensor {
        ONNXTensor {
            name: name.to_string(),
            data_type: ONNXDataType::Float,
            dims: dims.to_vec(),
            raw_data: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        }
    }

    fn node(op: &str, name: &str, inputs: &[&str], outputs: &[&str]) -> ONNXNode {
        ONNXNode {
            op_type: op.to_string(),
            inputs: inputs.iter().map(|s| s.to_string()).collect(),
            outputs: outputs.iter().map(|s| s.to_string()).collect(),
            attributes: HashMap::new(),
            name: name.to_string(),
        }
    }

    /// A small MLP: y = Relu(x @ w1) @ w2
    fn write_test_model(path: &Path) {
        let w1: Vec<f32> = (0..64 * 8).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let w2: Vec<f32> = (0..8 * 4).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();

        let graph = ONNXGraph {
            nodes: vec![
                node("MatMul", "mm1", &["x", "w1"], &["h"]),
                node("Relu", "relu", &["h"], &["a"]),
                node("MatMul", "mm2", &["a", "w2"], &["y"]),
            ],
            inputs: vec![value_info("x", &[1, 64])],
            outputs: vec![value_info("y", &[1, 4])],
            initializers: vec![
                initializer("w1", &[64, 8], &w1),
                initializer("w2", &[8, 4], &w2),
            ],
            name: "mlp".to_string(),
        };

        let exporter = ONNXExporter::new().with_opset_version(17);
        let model = exporter.wrap_graph(graph, &ExportConfig::default());
        exporter.export_graph(&model, path).expect("write model");
    }

    fn sample_input() -> HashMap<String, Tensor> {
        let values: Vec<f32> = (0..64).map(|i| (i as f32 % 7.0) - 3.0).collect();
        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            Tensor::from_vec(values, &[1, 64]).expect("input tensor"),
        );
        inputs
    }

    #[test]
    fn test_onnx_runtime_backend_creation() {
        let backend = ONNXRuntimeBackend::new();
        assert!(backend.config().enable_cpu_mem_arena);
    }

    #[test]
    fn test_onnx_runtime_config() {
        let config = ONNXRuntimeConfig {
            inter_op_num_threads: Some(4),
            intra_op_num_threads: Some(2),
            enable_cpu_mem_arena: false,
            enable_mem_pattern: false,
            execution_mode: ExecutionMode::Parallel,
            graph_optimization_level: GraphOptimizationLevel::Basic,
            log_severity_level: LogLevel::Error,
        };
        let backend = ONNXRuntimeBackend::with_config(config);
        assert_eq!(backend.config().inter_op_num_threads, Some(4));
        assert!(!backend.config().enable_cpu_mem_arena);
    }

    #[test]
    fn only_the_cpu_provider_is_advertised() {
        let providers = ONNXRuntimeBackend::new().get_available_providers();
        assert_eq!(providers, vec![ExecutionProvider::CPU]);
    }

    #[test]
    fn test_session_options() {
        let options = ONNXRuntimeBackend::new().create_session_options();
        assert!(!options.execution_providers.is_empty());
        assert!(options.enable_cpu_mem_arena);
    }

    #[test]
    fn test_load_nonexistent_model() {
        let result = ONNXRuntimeBackend::new().load_model("nonexistent.onnx");
        assert!(result.expect_err("must fail").to_string().contains("not found"));
    }

    /// Regression test: the old loader "extracted metadata" by scanning the file
    /// for substrings and happily accepted arbitrary bytes.
    #[test]
    fn load_model_rejects_files_that_are_not_onnx() {
        let dir = temp_dir("trustformers_onnxrt_bad_file");
        let path = dir.join("model.onnx");
        std::fs::write(&path, "dummy onnx content").expect("write");

        let err = ONNXRuntimeBackend::new()
            .load_model(&path)
            .expect_err("a text file is not a model");
        assert!(
            err.to_string().contains("not a parsable ONNX model"),
            "{err}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn input_and_output_names_come_from_the_graph() {
        let dir = temp_dir("trustformers_onnxrt_names");
        let path = dir.join("mlp.onnx");
        write_test_model(&path);

        let session = ONNXRuntimeBackend::new().load_model(&path).expect("load");
        assert_eq!(session.input_names(), &["x".to_string()]);
        assert_eq!(session.output_names(), &["y".to_string()]);
        assert!(session.unsupported_operators().is_empty());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Regression test for `simulate_inference`: the output must be a function of
    /// the input, and must be reproducible.
    #[test]
    fn run_computes_a_deterministic_function_of_the_input() {
        let dir = temp_dir("trustformers_onnxrt_run");
        let path = dir.join("mlp.onnx");
        write_test_model(&path);

        let session = ONNXRuntimeBackend::new().load_model(&path).expect("load");

        let first = session.run(sample_input()).expect("run");
        let second = session.run(sample_input()).expect("run");
        assert_eq!(
            first["y"].to_vec_f32().expect("f32"),
            second["y"].to_vec_f32().expect("f32"),
            "the same input must give the same output"
        );

        let mut other = sample_input();
        other.insert(
            "x".to_string(),
            Tensor::from_vec(vec![1.0; 64], &[1, 64]).expect("tensor"),
        );
        let different = session.run(other).expect("run");
        assert_ne!(
            first["y"].to_vec_f32().expect("f32"),
            different["y"].to_vec_f32().expect("f32"),
            "a different input must give a different output"
        );

        assert_eq!(first["y"].shape(), vec![1, 4]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The interpreter's answer must equal a naive reference computation.
    #[test]
    fn run_matches_a_naive_reference_implementation() {
        let dir = temp_dir("trustformers_onnxrt_reference");
        let path = dir.join("mlp.onnx");
        write_test_model(&path);

        let session = ONNXRuntimeBackend::new().load_model(&path).expect("load");
        let inputs = sample_input();
        let x = inputs["x"].to_vec_f32().expect("f32");
        let outputs = session.run(inputs).expect("run");

        let w1: Vec<f32> = (0..64 * 8).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
        let w2: Vec<f32> = (0..8 * 4).map(|i| ((i % 11) as f32 - 5.0) * 0.1).collect();

        let mut hidden = [0.0f32; 8];
        for (column, cell) in hidden.iter_mut().enumerate() {
            let mut accumulator = 0.0f32;
            for (row, &value) in x.iter().enumerate() {
                accumulator += value * w1[row * 8 + column];
            }
            *cell = accumulator.max(0.0);
        }
        let mut expected = [0.0f32; 4];
        for (column, cell) in expected.iter_mut().enumerate() {
            let mut accumulator = 0.0f32;
            for (row, &value) in hidden.iter().enumerate() {
                accumulator += value * w2[row * 4 + column];
            }
            *cell = accumulator;
        }

        for (actual, expected) in
            outputs["y"].to_vec_f32().expect("f32").iter().zip(expected.iter())
        {
            assert!((actual - expected).abs() < 1e-4, "{actual} vs {expected}");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_with_a_non_cpu_provider_is_refused() {
        let dir = temp_dir("trustformers_onnxrt_provider");
        let path = dir.join("mlp.onnx");
        write_test_model(&path);

        let session = ONNXRuntimeBackend::new().load_model(&path).expect("load");
        let err = session
            .run_with_provider(
                sample_input(),
                ExecutionProvider::CUDA { device_id: Some(0) },
            )
            .expect_err("no CUDA provider exists here");
        assert!(err.to_string().contains("Unsupported operation"), "{err}");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn benchmark_times_real_execution() {
        let dir = temp_dir("trustformers_onnxrt_benchmark");
        let path = dir.join("mlp.onnx");
        write_test_model(&path);

        let session = ONNXRuntimeBackend::new().load_model(&path).expect("load");
        let results = session.benchmark(sample_input(), 5).expect("benchmark");

        assert_eq!(results.num_runs, 5);
        assert!(results.min_latency_ms <= results.median_latency_ms);
        assert!(results.median_latency_ms <= results.max_latency_ms);
        assert!(results.mean_latency_ms > 0.0);

        assert!(session.benchmark(sample_input(), 0).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn benchmark_propagates_execution_errors() {
        let dir = temp_dir("trustformers_onnxrt_benchmark_err");
        let path = dir.join("mlp.onnx");
        write_test_model(&path);

        let session = ONNXRuntimeBackend::new().load_model(&path).expect("load");
        // A missing input must fail rather than time an empty loop.
        assert!(session.benchmark(HashMap::new(), 3).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn memory_info_reports_the_models_real_weight_bytes() {
        let dir = temp_dir("trustformers_onnxrt_memory");
        let path = dir.join("mlp.onnx");
        write_test_model(&path);

        let session = ONNXRuntimeBackend::new().load_model(&path).expect("load");
        let info = session.get_memory_info().expect("memory info");

        // 64*8 + 8*4 floats of initializer data.
        assert_eq!(info.model_memory_bytes, (64 * 8 + 8 * 4) * 4);
        assert!(info.total_memory_bytes > 0);
        assert!(info.available_memory_bytes <= info.total_memory_bytes);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_quantization_modes() {
        assert_ne!(QuantizationMode::Static, QuantizationMode::Dynamic);
    }

    /// Regression test for the optimizer that `std::fs::copy`'d its input.
    #[test]
    fn optimize_model_removes_identity_nodes() {
        let dir = temp_dir("trustformers_onnxrt_optimize");
        let input = dir.join("model.onnx");
        let output = dir.join("optimized.onnx");

        let graph = ONNXGraph {
            nodes: vec![
                node("Identity", "id", &["x"], &["x1"]),
                node("Add", "add", &["x1", "b"], &["y"]),
            ],
            inputs: vec![value_info("x", &[2])],
            outputs: vec![value_info("y", &[2])],
            initializers: vec![
                initializer("b", &[2], &[1.0, 2.0]),
                initializer("unused", &[2], &[9.0, 9.0]),
            ],
            name: "g".to_string(),
        };
        let exporter = ONNXExporter::new();
        let model = exporter.wrap_graph(graph, &ExportConfig::default());
        exporter.export_graph(&model, &input).expect("write");

        let stats = ONNXOptimizer::optimize_model_with_stats(
            &input,
            &output,
            GraphOptimizationLevel::Basic,
        )
        .expect("optimize");
        assert_eq!(stats.identity_nodes_removed, 1);
        assert_eq!(stats.initializers_removed, 1);

        let original = std::fs::read(&input).expect("read input");
        let optimized = std::fs::read(&output).expect("read output");
        assert_ne!(
            original, optimized,
            "an optimized model must not be a byte copy"
        );

        let reloaded = ONNXRuntimeBackend::new().load_model(&output).expect("load");
        assert_eq!(reloaded.executor().graph().nodes.len(), 1);

        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            Tensor::from_vec(vec![10.0, 20.0], &[2]).expect("tensor"),
        );
        let outputs = reloaded.run(inputs).expect("run");
        assert_eq!(outputs["y"].to_vec_f32().expect("f32"), vec![11.0, 22.0]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn optimize_model_folds_constants_at_the_higher_levels() {
        let dir = temp_dir("trustformers_onnxrt_fold");
        let input = dir.join("model.onnx");
        let output = dir.join("folded.onnx");

        let graph = ONNXGraph {
            nodes: vec![
                node("Add", "const_add", &["a", "b"], &["c"]),
                node("Mul", "scale", &["x", "c"], &["y"]),
            ],
            inputs: vec![value_info("x", &[2])],
            outputs: vec![value_info("y", &[2])],
            initializers: vec![
                initializer("a", &[2], &[1.0, 2.0]),
                initializer("b", &[2], &[3.0, 4.0]),
            ],
            name: "g".to_string(),
        };
        let exporter = ONNXExporter::new();
        let model = exporter.wrap_graph(graph, &ExportConfig::default());
        exporter.export_graph(&model, &input).expect("write");

        let stats =
            ONNXOptimizer::optimize_model_with_stats(&input, &output, GraphOptimizationLevel::All)
                .expect("optimize");
        assert_eq!(stats.constant_folded_nodes, 1);

        let reloaded = ONNXRuntimeBackend::new().load_model(&output).expect("load");
        assert_eq!(reloaded.executor().graph().nodes.len(), 1);

        let mut inputs = HashMap::new();
        inputs.insert(
            "x".to_string(),
            Tensor::from_vec(vec![1.0, 1.0], &[2]).expect("tensor"),
        );
        let outputs = reloaded.run(inputs).expect("run");
        assert_eq!(outputs["y"].to_vec_f32().expect("f32"), vec![4.0, 6.0]);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn optimize_model_at_level_none_reports_no_changes() {
        let dir = temp_dir("trustformers_onnxrt_optimize_none");
        let input = dir.join("mlp.onnx");
        let output = dir.join("same.onnx");
        write_test_model(&input);

        let stats =
            ONNXOptimizer::optimize_model_with_stats(&input, &output, GraphOptimizationLevel::None)
                .expect("optimize");
        assert_eq!(stats.identity_nodes_removed, 0);
        assert_eq!(stats.constant_folded_nodes, 0);
        assert_eq!(stats.initializers_removed, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Regression test for the "quantizer" that copied the file.
    #[test]
    fn quantize_model_really_shrinks_and_still_computes() {
        let dir = temp_dir("trustformers_onnxrt_quantize");
        let input = dir.join("mlp.onnx");
        let output = dir.join("quantized.onnx");
        write_test_model(&input);

        let reference = ONNXRuntimeBackend::new().load_model(&input).expect("load");
        let expected = reference.run(sample_input()).expect("run")["y"].to_vec_f32().expect("f32");

        let stats =
            ONNXOptimizer::quantize_model_with_stats(&input, &output, QuantizationMode::Dynamic)
                .expect("quantize");
        assert_eq!(stats.tensors_quantized, 2);
        assert!(
            stats.bytes_after < stats.bytes_before,
            "quantized model must be smaller: {} -> {}",
            stats.bytes_before,
            stats.bytes_after
        );

        let original = std::fs::read(&input).expect("read");
        let quantized = std::fs::read(&output).expect("read");
        assert_ne!(original, quantized, "must not be a byte copy");

        let session = ONNXRuntimeBackend::new().load_model(&output).expect("load quantized");
        let actual = session.run(sample_input()).expect("run")["y"].to_vec_f32().expect("f32");

        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert!(
                (actual - expected).abs() < 0.2,
                "quantized output drifted too far: {actual} vs {expected}"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn static_quantization_is_refused_without_calibration_data() {
        let dir = temp_dir("trustformers_onnxrt_static_quant");
        let input = dir.join("mlp.onnx");
        let output = dir.join("static.onnx");
        write_test_model(&input);

        let err = ONNXOptimizer::quantize_model(&input, &output, QuantizationMode::Static)
            .expect_err("no calibration data was supplied");
        assert!(err.to_string().contains("Unsupported operation"), "{err}");
        assert!(!output.exists(), "no output file may be produced");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn optimizer_rejects_non_onnx_inputs() {
        let dir = temp_dir("trustformers_onnxrt_optimize_bad");
        let input = dir.join("input.onnx");
        let output = dir.join("output.onnx");
        std::fs::write(&input, "dummy onnx content").expect("write");

        assert!(
            ONNXOptimizer::optimize_model(&input, &output, GraphOptimizationLevel::All).is_err()
        );
        assert!(ONNXOptimizer::quantize_model(&input, &output, QuantizationMode::Dynamic).is_err());
        assert!(!output.exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_benchmark_results() {
        let results = BenchmarkResults {
            num_runs: 100,
            mean_latency_ms: 15.5,
            median_latency_ms: 14.2,
            p90_latency_ms: 18.7,
            p95_latency_ms: 20.1,
            p99_latency_ms: 25.3,
            min_latency_ms: 12.1,
            max_latency_ms: 28.9,
        };
        assert_eq!(results.num_runs, 100);
        assert!((results.mean_latency_ms - 15.5).abs() < 1e-6);
    }
}
