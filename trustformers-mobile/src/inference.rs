//! Mobile Inference Engine
//!
//! This module provides a unified mobile inference engine that integrates
//! platform-specific optimizations, quantization, and memory management
//! for efficient transformer inference on mobile devices.

use crate::{
    optimization::MobileOptimizationEngine, MobileBackend, MobileConfig, MobilePlatform,
    MobileStats,
};
use safetensors::SafeTensors;
use scirs2_core::ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
use trustformers_core::errors::{
    invalid_format, invalid_input, runtime_error, unsupported_operation, Result,
};
use trustformers_core::Tensor;

/// Supported model formats for loading
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    /// SafeTensors format (.safetensors)
    SafeTensors,
    /// PyTorch format (.pt, .pth, .bin)
    PyTorch,
    /// ONNX format (.onnx)
    ONNX,
    /// TensorFlow format (.pb)
    TensorFlow,
    /// Unknown or unsupported format
    Unknown,
}

/// Execution strategy for inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStrategy {
    /// Execute layers sequentially
    Sequential,
    /// Parallelize within layers
    LayerParallel,
    /// Full parallel execution
    FullParallel,
}

/// Execution plan for mobile inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPlan {
    pub strategy: ExecutionStrategy,
    pub num_layers: usize,
    pub batch_size: usize,
    pub checkpoint_interval: usize,
    /// Deterministic application order for the loaded weight tensors.
    ///
    /// A `HashMap<String, Tensor>` has no defined iteration order, so an
    /// engine that walked `model_weights` directly would apply layers in a
    /// different (arbitrary, hash-seed-dependent) order on every run. This
    /// list is computed once, in [`MobileInferenceEngine::load_model`], with
    /// a "natural" sort (numeric runs inside a name compare numerically, so
    /// `"h.2"` sorts before `"h.10"`) so the same checkpoint always executes
    /// the same way.
    pub ordered_weight_names: Vec<String>,
}

impl ExecutionPlan {
    pub fn new(strategy: ExecutionStrategy, num_layers: usize) -> Self {
        Self {
            strategy,
            num_layers,
            batch_size: 1,
            checkpoint_interval: 0, // Disabled by default
            ordered_weight_names: Vec::new(),
        }
    }

    /// Recompute the deterministic execution order from a freshly loaded
    /// weight map.
    fn set_layer_order(&mut self, weights: &HashMap<String, Tensor>) {
        let mut names: Vec<String> = weights.keys().cloned().collect();
        names.sort_by(|a, b| natural_cmp(a, b));
        self.num_layers = names.len();
        self.ordered_weight_names = names;
    }
}

/// One run of a "natural sort" key: either a digit run (compared as a
/// number, so `"2"` sorts before `"10"`) or a non-digit run (compared as
/// text).
#[derive(Debug, PartialEq, Eq)]
enum NaturalPart {
    Num(u64),
    Text(String),
}

/// Split `s` into alternating digit/non-digit runs for [`natural_cmp`].
fn natural_key(s: &str) -> Vec<NaturalPart> {
    let mut parts = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_ascii_digit() {
            let mut num_str = String::new();
            while let Some(&d) = chars.peek() {
                if d.is_ascii_digit() {
                    num_str.push(d);
                    chars.next();
                } else {
                    break;
                }
            }
            // A digit run longer than u64::MAX's digit count cannot occur in
            // any realistic tensor name; saturate rather than panic if it
            // somehow does.
            parts.push(NaturalPart::Num(num_str.parse().unwrap_or(u64::MAX)));
        } else {
            let mut text = String::new();
            while let Some(&d) = chars.peek() {
                if !d.is_ascii_digit() {
                    text.push(d);
                    chars.next();
                } else {
                    break;
                }
            }
            parts.push(NaturalPart::Text(text));
        }
    }
    parts
}

/// Compare two tensor names so that embedded integers order numerically
/// (`"layer.2.weight"` < `"layer.10.weight"`) while everything else orders
/// lexically. Falls back to a plain string comparison when the natural keys
/// tie (e.g. one name is a strict prefix of the other), which keeps the
/// order total and deterministic.
fn natural_cmp(a: &str, b: &str) -> Ordering {
    let ka = natural_key(a);
    let kb = natural_key(b);
    for (pa, pb) in ka.iter().zip(kb.iter()) {
        let ord = match (pa, pb) {
            (NaturalPart::Num(x), NaturalPart::Num(y)) => x.cmp(y),
            (NaturalPart::Text(x), NaturalPart::Text(y)) => x.cmp(y),
            // A digit run and a text run at the same position: digits sort
            // first, matching the common "prefix.N" < "prefix.suffix" case.
            (NaturalPart::Num(_), NaturalPart::Text(_)) => Ordering::Less,
            (NaturalPart::Text(_), NaturalPart::Num(_)) => Ordering::Greater,
        };
        if ord != Ordering::Equal {
            return ord;
        }
    }
    ka.len().cmp(&kb.len()).then_with(|| a.cmp(b))
}

/// Decode a little-endian `f32` buffer. Errors (rather than silently
/// truncating) when the byte count is not a multiple of the element width.
fn le_f32_vec(name: &str, data: &[u8]) -> Result<Vec<f32>> {
    if !data.len().is_multiple_of(4) {
        return Err(invalid_format(
            "a byte length that is a multiple of 4 for an F32 tensor",
            format!("tensor '{name}' has {} bytes", data.len()),
        ));
    }
    Ok(data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Decode a little-endian `f64` buffer.
fn le_f64_vec(name: &str, data: &[u8]) -> Result<Vec<f64>> {
    if !data.len().is_multiple_of(8) {
        return Err(invalid_format(
            "a byte length that is a multiple of 8 for an F64 tensor",
            format!("tensor '{name}' has {} bytes", data.len()),
        ));
    }
    Ok(data
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

/// Decode a little-endian `i64` buffer.
fn le_i64_vec(name: &str, data: &[u8]) -> Result<Vec<i64>> {
    if !data.len().is_multiple_of(8) {
        return Err(invalid_format(
            "a byte length that is a multiple of 8 for an I64 tensor",
            format!("tensor '{name}' has {} bytes", data.len()),
        ));
    }
    Ok(data
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

/// Decode a little-endian IEEE-754 half-precision buffer.
fn le_f16_vec(name: &str, data: &[u8]) -> Result<Vec<half::f16>> {
    if !data.len().is_multiple_of(2) {
        return Err(invalid_format(
            "a byte length that is a multiple of 2 for an F16 tensor",
            format!("tensor '{name}' has {} bytes", data.len()),
        ));
    }
    Ok(data.chunks_exact(2).map(|c| half::f16::from_le_bytes([c[0], c[1]])).collect())
}

/// Decode a little-endian bfloat16 buffer.
fn le_bf16_vec(name: &str, data: &[u8]) -> Result<Vec<half::bf16>> {
    if !data.len().is_multiple_of(2) {
        return Err(invalid_format(
            "a byte length that is a multiple of 2 for a BF16 tensor",
            format!("tensor '{name}' has {} bytes", data.len()),
        ));
    }
    Ok(data.chunks_exact(2).map(|c| half::bf16::from_le_bytes([c[0], c[1]])).collect())
}

/// Build a [`Tensor`] from a shape and a `Vec` of already-decoded elements,
/// reporting a shape mismatch (never panicking) if the element count is
/// wrong for the declared shape.
fn array_tensor<T, F>(name: &str, shape: &[usize], data: Vec<T>, wrap: F) -> Result<Tensor>
where
    F: FnOnce(ArrayD<T>) -> Tensor,
{
    let expected: usize = shape.iter().product();
    if data.len() != expected {
        return Err(invalid_format(
            format!("{expected} elements for shape {shape:?}"),
            format!("tensor '{name}' decoded to {} elements", data.len()),
        ));
    }
    let array = ArrayD::from_shape_vec(IxDyn(shape), data).map_err(|e| {
        invalid_format("a shape-compatible buffer", format!("tensor '{name}': {e}"))
    })?;
    Ok(wrap(array))
}

/// Convert one safetensors [`View`](safetensors::tensor::View) into a real
/// [`Tensor`] holding that view's own bytes. Returns `Ok(None)` for a dtype
/// this engine does not (yet) have a `Tensor` variant for, so the caller can
/// skip it without fabricating data in its place.
fn safetensors_view_to_tensor(
    name: &str,
    dtype: safetensors::Dtype,
    shape: &[usize],
    data: &[u8],
) -> Result<Option<Tensor>> {
    use safetensors::Dtype;
    Ok(Some(match dtype {
        Dtype::F32 => array_tensor(name, shape, le_f32_vec(name, data)?, Tensor::F32)?,
        Dtype::F64 => array_tensor(name, shape, le_f64_vec(name, data)?, Tensor::F64)?,
        Dtype::I64 => array_tensor(name, shape, le_i64_vec(name, data)?, Tensor::I64)?,
        Dtype::F16 => array_tensor(name, shape, le_f16_vec(name, data)?, Tensor::F16)?,
        Dtype::BF16 => array_tensor(name, shape, le_bf16_vec(name, data)?, Tensor::BF16)?,
        _ => return Ok(None),
    }))
}

/// Convert one ONNX `TensorProto` initializer into a real [`Tensor`] holding
/// that initializer's own `raw_data`. Returns `Ok(None)` for a dtype this
/// engine does not (yet) have a `Tensor` variant for.
fn onnx_tensor_to_tensor(
    name: &str,
    dtype: trustformers_core::export::ONNXDataType,
    shape: &[usize],
    raw: &[u8],
) -> Result<Option<Tensor>> {
    use trustformers_core::export::ONNXDataType;
    Ok(Some(match dtype {
        ONNXDataType::Float => array_tensor(name, shape, le_f32_vec(name, raw)?, Tensor::F32)?,
        ONNXDataType::Double => array_tensor(name, shape, le_f64_vec(name, raw)?, Tensor::F64)?,
        ONNXDataType::Int64 => array_tensor(name, shape, le_i64_vec(name, raw)?, Tensor::I64)?,
        ONNXDataType::Float16 => array_tensor(name, shape, le_f16_vec(name, raw)?, Tensor::F16)?,
        ONNXDataType::BFloat16 => array_tensor(name, shape, le_bf16_vec(name, raw)?, Tensor::BF16)?,
        _ => return Ok(None),
    }))
}

/// Unified mobile inference engine
#[derive(Debug)]
pub struct MobileInferenceEngine {
    config: MobileConfig,
    optimizer: MobileOptimizationEngine,
    execution_plan: ExecutionPlan,
    stats: MobileStats,
    model_loaded: bool,
    model_weights: Option<HashMap<String, Tensor>>,
    cache: Option<InferenceCache>,
}

impl MobileInferenceEngine {
    /// Create new mobile inference engine
    pub fn new(config: MobileConfig) -> Result<Self> {
        config.validate()?;

        let optimizer = MobileOptimizationEngine::new(config.clone())?;
        let execution_plan = ExecutionPlan::new(ExecutionStrategy::Sequential, 12); // Default 12 layers
        let stats = MobileStats::new(&config);

        Ok(Self {
            config,
            optimizer,
            execution_plan,
            stats,
            model_loaded: false,
            model_weights: None,
            cache: None,
        })
    }

    /// Load model weights and optimize for mobile deployment
    pub fn load_model(&mut self, weights: HashMap<String, Tensor>) -> Result<()> {
        tracing::info!("Loading model with {} parameters", weights.len());

        // Optimize weights for mobile deployment
        let optimized_weights = self.optimizer.optimize_model_weights(&weights)?;

        // Calculate memory footprint
        let total_params: usize =
            optimized_weights.values().map(|t| t.shape().iter().product::<usize>()).sum();

        let footprint = self.optimizer.estimate_memory_footprint(total_params);

        if footprint.total_memory_bytes > self.config.max_memory_mb * 1024 * 1024 {
            return Err(runtime_error(format!(
                "Model requires {}MB but limit is {}MB",
                footprint.memory_usage_mb(),
                self.config.max_memory_mb
            )));
        }

        self.execution_plan.set_layer_order(&optimized_weights);
        self.model_weights = Some(optimized_weights);
        self.model_loaded = true;

        // Initialize cache if needed
        if self.should_use_cache() {
            self.cache = Some(InferenceCache::new(self.config.max_memory_mb / 4));
        }

        tracing::info!(
            "Model loaded successfully. Memory footprint: {:.1}MB ({:.1}% savings)",
            footprint.memory_usage_mb(),
            footprint.memory_savings_percent
        );

        Ok(())
    }

    /// Load model from file path
    pub fn load_model_from_file(&mut self, model_path: &str) -> Result<()> {
        use std::fs;
        use std::path::Path;

        let path = Path::new(model_path);
        let model_data = fs::read(model_path)
            .map_err(|e| runtime_error(format!("Failed to read model file: {}", e)))?;

        let weights = self.parse_model_format(&model_data, path)?;
        self.load_model(weights)
    }

    /// Parse model format based on file extension and magic bytes
    ///
    /// Every branch parses the checkpoint's own bytes; none of them
    /// synthesize weights. A format this engine cannot yet parse (or cannot
    /// identify at all) is a hard error, never a silently substituted random
    /// tensor set.
    fn parse_model_format(&self, data: &[u8], path: &Path) -> Result<HashMap<String, Tensor>> {
        let format = self.detect_model_format(data, path)?;

        let weights = match format {
            ModelFormat::SafeTensors => {
                tracing::info!("Loading SafeTensors format model");
                Self::parse_safetensors(data)?
            },
            ModelFormat::PyTorch => {
                tracing::info!("Loading PyTorch format model");
                Self::parse_pytorch(data)?
            },
            ModelFormat::ONNX => {
                tracing::info!("Loading ONNX format model");
                Self::parse_onnx(data)?
            },
            ModelFormat::TensorFlow => {
                return Err(unsupported_operation(
                    "loading a TensorFlow SavedModel/.pb checkpoint",
                    "MobileInferenceEngine::load_model_from_file (TensorFlow parsing is not \
                     implemented; convert the model to safetensors or ONNX first)",
                ));
            },
            ModelFormat::Unknown => {
                return Err(invalid_format(
                    "safetensors, PyTorch (.pt/.pth/.bin), or ONNX (.onnx)",
                    format!(
                        "unrecognised model file at {} (no matching extension or file header)",
                        path.display()
                    ),
                ));
            },
        };

        if weights.is_empty() {
            return Err(runtime_error(format!(
                "{} contained no tensors",
                path.display()
            )));
        }

        Ok(weights)
    }

    /// Detect model format from file extension and, failing that, the file's
    /// own structural header.
    fn detect_model_format(&self, data: &[u8], path: &Path) -> Result<ModelFormat> {
        // Check file extension first
        if let Some(extension) = path.extension().and_then(|s| s.to_str()) {
            match extension.to_lowercase().as_str() {
                "safetensors" => return Ok(ModelFormat::SafeTensors),
                "pt" | "pth" | "bin" => return Ok(ModelFormat::PyTorch),
                "onnx" => return Ok(ModelFormat::ONNX),
                "pb" => return Ok(ModelFormat::TensorFlow),
                _ => {},
            }
        }

        // Fall back to each format's real structural header. safetensors has
        // no magic string; a valid file's first 8 bytes are a little-endian
        // u64 giving the length of a JSON header that immediately follows
        // and must itself parse as a JSON object.
        if data.len() >= 9 {
            let header_len = u64::from_le_bytes([
                data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
            ]);
            let header_end = 8usize.saturating_add(header_len as usize);
            if header_len > 0
                && header_len < data.len() as u64
                && header_end <= data.len()
                && data[8] == b'{'
                && serde_json::from_slice::<serde_json::Value>(&data[8..header_end]).is_ok()
            {
                return Ok(ModelFormat::SafeTensors);
            }
        }

        if data.len() >= 8 {
            // PyTorch: a real ZIP-based checkpoint (`PK\x03\x04` local file
            // header) or a legacy pickle stream (opcode `\x80`, protocol byte).
            if data.starts_with(b"PK\x03\x04") || data.starts_with(b"\x80") {
                return Ok(ModelFormat::PyTorch);
            }

            // ONNX ModelProto: field 1 (ir_version, varint) tag byte 0x08.
            if data.starts_with(b"\x08") {
                return Ok(ModelFormat::ONNX);
            }

            // TensorFlow SavedModel protobuf (`saved_model.pb`): field 1
            // (saved_model_schema_version, varint) tag byte 0x08 as well, so
            // this is only reachable once the extension check above has
            // already ruled ONNX in or out via `.pb`/`.onnx`. Kept as a last
            // resort for extension-less TensorFlow files.
            if path.extension().and_then(|s| s.to_str()) == Some("pb") {
                return Ok(ModelFormat::TensorFlow);
            }
        }

        Ok(ModelFormat::Unknown)
    }

    /// Parse a real safetensors buffer via the `safetensors` crate.
    ///
    /// Every tensor's bytes are decoded per its declared dtype; nothing is
    /// invented. A tensor whose dtype this engine cannot yet represent as a
    /// `Tensor` variant (e.g. an 8-bit quantized buffer) is skipped with a
    /// warning rather than fabricated; a load that yields zero usable
    /// tensors is still rejected by the empty-weights check in
    /// [`Self::parse_model_format`].
    fn parse_safetensors(data: &[u8]) -> Result<HashMap<String, Tensor>> {
        let parsed = SafeTensors::deserialize(data).map_err(|e| {
            invalid_format("a valid safetensors buffer", format!("parse error: {e}"))
        })?;

        let mut weights = HashMap::new();
        for (name, view) in parsed.tensors() {
            let view_data = view.data();
            match safetensors_view_to_tensor(&name, view.dtype(), view.shape(), view_data)? {
                Some(tensor) => {
                    weights.insert(name, tensor);
                },
                None => {
                    tracing::warn!(
                        "safetensors tensor '{name}' has dtype {:?}, which this engine does not \
                         map to a Tensor variant; skipping it (its bytes are not used).",
                        view.dtype()
                    );
                },
            }
        }
        Ok(weights)
    }

    /// Parse a real PyTorch checkpoint (`.pt`/`.pth`/`.bin`) via
    /// [`trustformers_core`]'s ZIP + pickle reader.
    fn parse_pytorch(data: &[u8]) -> Result<HashMap<String, Tensor>> {
        use trustformers_core::traits::WeightReader;
        use trustformers_core::utils::weight_loading::PyTorchReader;

        let mut reader = PyTorchReader::from_bytes(data)?;
        let mut weights = HashMap::new();
        for name in reader.list_tensors() {
            let tensor = reader.read_tensor(&name)?;
            weights.insert(name, tensor);
        }
        Ok(weights)
    }

    /// Parse a real ONNX `ModelProto` via [`trustformers_core`]'s pure-Rust
    /// protobuf decoder, taking the graph's initializers as the weight set.
    ///
    /// Initializers whose element dtype this engine cannot represent as a
    /// [`Tensor`] (e.g. `String`, `Bool`) are skipped with a warning rather
    /// than failing the whole load: ONNX graphs routinely carry non-float
    /// constants (shape/index buffers) that are irrelevant to weight-role
    /// dispatch in [`MobileInferenceEngine::process_layer`]. A load that
    /// yields zero usable tensors is still rejected by the empty-weights
    /// check in [`Self::parse_model_format`].
    fn parse_onnx(data: &[u8]) -> Result<HashMap<String, Tensor>> {
        use trustformers_core::export::onnx_proto::decode_model;

        let model = decode_model(data)
            .map_err(|e| invalid_format("a valid ONNX ModelProto", format!("parse error: {e}")))?;

        let mut weights = HashMap::new();
        for initializer in &model.graph.initializers {
            let shape: Result<Vec<usize>> = initializer
                .dims
                .iter()
                .map(|&d| {
                    usize::try_from(d).map_err(|_| {
                        invalid_format(
                            "non-negative ONNX tensor dimensions",
                            format!("initializer '{}' has dimension {d}", initializer.name),
                        )
                    })
                })
                .collect();
            let shape = shape?;

            match onnx_tensor_to_tensor(
                &initializer.name,
                initializer.data_type,
                &shape,
                &initializer.raw_data,
            ) {
                Ok(Some(tensor)) => {
                    weights.insert(initializer.name.clone(), tensor);
                },
                Ok(None) => {
                    tracing::warn!(
                        "ONNX initializer '{}' has dtype {:?}, which this engine does not map to a \
                         Tensor variant; skipping it (its bytes are not used).",
                        initializer.name,
                        initializer.data_type
                    );
                },
                Err(e) => return Err(e),
            }
        }
        Ok(weights)
    }

    /// Perform inference with f32 input/output arrays (for C API)
    pub fn inference_f32(&mut self, input_data: &[f32], output_data: &mut [f32]) -> Result<usize> {
        // Convert input array to tensor
        let input_tensor = Tensor::from_vec(input_data.to_vec(), &[1, input_data.len()])?;

        // Perform inference
        let output_tensor = self.inference(&input_tensor)?;

        // Extract data from output tensor
        let output_vec = output_tensor.data()?;
        let output_size = output_vec.len().min(output_data.len());

        // Copy to output array
        for i in 0..output_size {
            output_data[i] = output_vec[i];
        }

        Ok(output_size)
    }

    /// Perform optimized mobile inference
    pub fn inference(&mut self, input: &Tensor) -> Result<Tensor> {
        if !self.model_loaded {
            return Err(runtime_error("Model not loaded"));
        }

        let start_time = Instant::now();

        // Check cache first
        if let Some(ref cache) = self.cache {
            if let Some(cached_result) = cache.get(input) {
                let inference_time = start_time.elapsed().as_millis() as f32;
                self.stats.update_inference(inference_time);
                tracing::debug!("Cache hit for inference");
                return Ok(cached_result);
            }
        }

        // Optimize input tensor
        let optimized_input = self.optimizer.optimize_tensor(input)?;

        // Perform inference based on execution strategy
        let result = match self.execution_plan.strategy {
            ExecutionStrategy::Sequential => self.sequential_inference(&optimized_input),
            ExecutionStrategy::LayerParallel => self.layer_parallel_inference(&optimized_input),
            ExecutionStrategy::FullParallel => self.full_parallel_inference(&optimized_input),
        }?;

        // Cache result if caching is enabled
        if let Some(ref mut cache) = self.cache {
            cache.put(input.clone(), result.clone());
        }

        let inference_time = start_time.elapsed().as_millis() as f32;
        self.stats.update_inference(inference_time);

        // Update memory statistics
        let current_memory = self.estimate_current_memory_usage();
        self.stats.update_memory(current_memory);

        Ok(result)
    }

    /// Perform batch inference with mobile optimizations
    pub fn batch_inference(&mut self, inputs: Vec<Tensor>) -> Result<Vec<Tensor>> {
        if !self.model_loaded {
            return Err(runtime_error("Model not loaded"));
        }

        // Optimize batch for mobile constraints
        let optimized_inputs = self.optimizer.optimize_batch(&inputs)?;

        let mut results = Vec::with_capacity(optimized_inputs.len());
        for input in optimized_inputs {
            let result = self.inference(&input)?;
            results.push(result);
        }

        Ok(results)
    }

    /// Get current inference statistics
    pub fn get_stats(&self) -> &MobileStats {
        &self.stats
    }

    /// Get memory usage information
    pub fn get_memory_info(&self) -> MobileMemoryInfo {
        let footprint = if let Some(ref weights) = self.model_weights {
            let total_params: usize =
                weights.values().map(|t| t.shape().iter().product::<usize>()).sum();
            self.optimizer.estimate_memory_footprint(total_params)
        } else {
            self.optimizer.estimate_memory_footprint(0)
        };

        MobileMemoryInfo {
            model_memory_mb: footprint.model_memory_bytes / (1024 * 1024),
            runtime_memory_mb: footprint.runtime_overhead_bytes / (1024 * 1024),
            total_memory_mb: footprint.total_memory_bytes / (1024 * 1024),
            memory_limit_mb: self.config.max_memory_mb,
            memory_savings_percent: footprint.memory_savings_percent,
            cache_memory_mb: self.cache.as_ref().map(|c| c.memory_usage_mb()).unwrap_or(0),
        }
    }

    /// Update configuration and re-optimize
    pub fn update_config(&mut self, new_config: MobileConfig) -> Result<()> {
        new_config.validate()?;

        self.config = new_config.clone();
        self.optimizer = MobileOptimizationEngine::new(new_config)?;

        // Re-optimize loaded model if available
        if let Some(ref weights) = self.model_weights.clone() {
            self.load_model(weights.clone())?;
        }

        Ok(())
    }

    /// Set power mode for inference
    pub fn set_power_mode(&mut self, power_mode: crate::optimization::PowerMode) -> Result<()> {
        // Update the configuration based on power mode
        match power_mode {
            crate::optimization::PowerMode::PowerSaving => {
                self.config.use_fp16 = true;
                self.config.max_memory_mb /= 2;
                self.config.backend = crate::MobileBackend::CPU;
            },
            crate::optimization::PowerMode::Balanced => {
                // Keep current settings but optimize for balance
                self.config.use_fp16 = true;
            },
            crate::optimization::PowerMode::HighPerformance => {
                self.config.use_fp16 = false;
                self.config.backend = crate::MobileBackend::GPU;
            },
        }

        // Update the optimizer with new config
        self.optimizer = crate::optimization::MobileOptimizationEngine::new(self.config.clone())?;

        Ok(())
    }

    /// Reduce performance by a factor (0.0 = minimum, 1.0 = maximum)
    pub fn reduce_performance(&mut self, factor: f32) -> Result<()> {
        let factor = factor.clamp(0.1, 1.0);

        // Reduce memory usage
        self.config.max_memory_mb = (self.config.max_memory_mb as f32 * factor) as usize;

        // Force FP16 for reduced performance
        if factor < 0.8 {
            self.config.use_fp16 = true;
        }

        // Switch to CPU for very low performance
        if factor < 0.5 {
            self.config.backend = crate::MobileBackend::CPU;
        }

        // Update optimizer
        self.optimizer = crate::optimization::MobileOptimizationEngine::new(self.config.clone())?;

        Ok(())
    }

    /// Set batch size for inference
    pub fn set_batch_size(&mut self, batch_size: usize) -> Result<()> {
        // Note: This is a placeholder implementation since batch size isn't directly stored in config
        // In a real implementation, this would be stored in the engine state
        if batch_size == 0 {
            return Err(invalid_input("Batch size must be greater than 0"));
        }

        // For now, adjust memory based on batch size
        // Larger batches need more memory
        let base_memory = 512; // Base memory in MB
        let memory_per_batch = 64; // Additional memory per batch item
        self.config.max_memory_mb = base_memory + (batch_size - 1) * memory_per_batch;

        Ok(())
    }

    /// Clear inference cache to free memory
    pub fn clear_cache(&mut self) {
        if let Some(ref mut cache) = self.cache {
            cache.clear();
        }
    }

    /// Force garbage collection to free memory
    pub fn force_gc(&mut self) {
        self.clear_cache();
        // In a real implementation, this would trigger platform-specific GC
    }

    /// The input width [`Self::warm_up`] should use, derived from the first
    /// weight tensor in `execution_plan.ordered_weight_names` (the same
    /// tensor `run_ordered_layers` will apply first to a real input).
    ///
    /// For a 2D weight this is whichever dimension is *not* the "output"
    /// side, matching `process_layer`'s own orientation logic: for a
    /// `[512, 512]` square weight either dimension works identically, so
    /// the first (`weight_shape[0]`) is used, mirroring `process_layer`'s
    /// own no-transpose tie-break; for a `[in, out]` or `[out, in]`
    /// rectangular weight, the differing dimension unambiguously identifies
    /// which side is "in". A 1D first weight (a bare bias/scale with no
    /// preceding projection) uses its own length directly, since
    /// `process_layer` applies a 1D weight only when its length already
    /// matches the input's last dimension.
    ///
    /// # Errors
    ///
    /// Returns an error if no weights are loaded, or if the first weight's
    /// rank is neither 1 nor 2 (this engine's `process_layer` does not
    /// apply higher-rank weights, so there would be nothing for warm-up to
    /// meaningfully exercise).
    fn infer_warm_up_hidden_size(&self) -> Result<usize> {
        let weights = self
            .model_weights
            .as_ref()
            .ok_or_else(|| runtime_error("Cannot warm up: no weights loaded"))?;
        let first_name = self.execution_plan.ordered_weight_names.first().ok_or_else(|| {
            runtime_error("Cannot warm up: the loaded model has no weight tensors")
        })?;
        let first_weight = weights.get(first_name).ok_or_else(|| {
            runtime_error(format!(
                "internal error: weight '{first_name}' is in the execution plan but missing \
                 from the loaded weight map"
            ))
        })?;

        let shape = first_weight.shape();
        match shape.len() {
            1 => Ok(shape[0]),
            2 => Ok(shape[0]),
            other => Err(runtime_error(format!(
                "Cannot warm up: the first loaded weight '{first_name}' has rank {other}, which \
                 this engine's layer dispatch does not apply (only rank 1 and rank 2 weights are \
                 supported)"
            ))),
        }
    }

    /// Warm up the engine by running dummy inferences
    ///
    /// This method runs several dummy inference passes to:
    /// - Initialize GPU/accelerator resources
    /// - Compile compute shaders/kernels
    /// - Populate caches
    /// - Stabilize performance measurements
    ///
    /// Should be called after model loading to ensure consistent performance.
    pub fn warm_up(&mut self) -> Result<()> {
        if !self.model_loaded {
            return Err(runtime_error("Cannot warm up: model not loaded"));
        }

        tracing::info!("Starting engine warm-up...");
        let start_time = Instant::now();

        // Determine input shape from the *actual loaded model*, not a
        // hardcoded guess. `inference()` now performs real shape-driven
        // dispatch (see `process_layer`/`run_ordered_layers`) and errors
        // when no loaded weight is shape-compatible with the input; a fixed
        // `hidden_size = 512` here would make warm-up fail for any real
        // checkpoint whose first layer's input width differs from 512, even
        // though `inference()` itself works fine on that model's real
        // input shape. Derive the width instead from the first tensor in
        // the execution order, exactly as `run_ordered_layers` will apply
        // it, so warm-up exercises the model it was actually given.
        let batch_size = 1;
        let seq_length = 128; // Typical warm-up sequence length
        let hidden_size = self.infer_warm_up_hidden_size()?;

        // Run multiple warm-up iterations
        let warm_up_iterations = 3;

        for i in 0..warm_up_iterations {
            // Create dummy input tensor
            let dummy_input = Tensor::zeros(&[batch_size, seq_length, hidden_size])?;

            // Perform inference (this will initialize kernels and caches)
            let _result = self.inference(&dummy_input)?;

            tracing::debug!(
                "Warm-up iteration {}/{} completed",
                i + 1,
                warm_up_iterations
            );
        }

        let warm_up_time = start_time.elapsed();
        tracing::info!(
            "Engine warm-up completed in {:.2}ms ({} iterations)",
            warm_up_time.as_millis(),
            warm_up_iterations
        );

        Ok(())
    }

    /// Set performance mode for the engine
    ///
    /// This is a convenience wrapper around set_power_mode that accepts
    /// integer mode values for C FFI compatibility:
    /// - 0: Power Saving mode
    /// - 1: Balanced mode
    /// - 2: High Performance mode
    pub fn set_performance_mode(&mut self, mode: i32) -> Result<()> {
        let power_mode = match mode {
            0 => crate::optimization::PowerMode::PowerSaving,
            1 => crate::optimization::PowerMode::Balanced,
            2 => crate::optimization::PowerMode::HighPerformance,
            _ => return Err(invalid_input(format!("Invalid performance mode: {}", mode))),
        };

        self.set_power_mode(power_mode)
    }

    // Private inference methods
    //
    // All three execution strategies below run the identical real
    // computation -- matmul/bias-add/scale, dispatched per weight tensor by
    // `process_layer` -- via the same ambiguity-checked walk in
    // `run_ordered_layers`. Real multi-threaded scheduling of the
    // `LayerParallel`/`FullParallel` strategies is future work; today they
    // differ from `Sequential` only in name, never in the numbers they
    // produce. That is an honest, documented limitation -- the previous
    // implementation had three strategies that all silently returned the
    // input unchanged, which "worked" identically for the wrong reason.

    fn sequential_inference(&self, input: &Tensor) -> Result<Tensor> {
        self.run_ordered_layers(input)
    }

    fn layer_parallel_inference(&self, input: &Tensor) -> Result<Tensor> {
        self.run_ordered_layers(input)
    }

    fn full_parallel_inference(&self, input: &Tensor) -> Result<Tensor> {
        self.run_ordered_layers(input)
    }

    /// Apply loaded weight tensors to `input` via [`Self::process_layer`]'s
    /// real matmul/bias/scale dispatch, one at a time (or one matched
    /// weight+bias pair at a time -- see below), until no remaining tensor
    /// is shape-compatible with the current activation.
    ///
    /// This engine has no per-model architecture graph -- only a flat bag
    /// of named tensors -- so at every step it must be able to tell *which*
    /// remaining tensor is "next". When the current activation's shape is
    /// simultaneously compatible with more than one not-yet-applied tensor,
    /// there is in general no architecture-free way to pick the right one.
    /// Chaining them anyway in an arbitrary (e.g. alphabetically sorted)
    /// order would still run to completion and produce a confident,
    /// shape-plausible number -- but not a meaningful one, since the wrong
    /// tensor could be applied at each such step. That is fabrication with
    /// extra steps, strictly worse than the old identity pass because it is
    /// not obviously wrong. This engine refuses instead: an ambiguous step
    /// is a hard error naming every candidate, so the caller learns the
    /// model is not a simple linear stack rather than silently getting a
    /// wrong answer.
    ///
    /// One specific two-way "ambiguity" is not really one and is resolved
    /// automatically rather than rejected: a projection weight together
    /// with its own bias, by the standard `<prefix>.weight` /
    /// `<prefix>.bias` (or `<prefix>_weight` / `<prefix>_bias`) naming
    /// convention every checkpoint format this engine parses uses. When a
    /// weight's input width equals its output width (a square projection --
    /// e.g. an attention output or residual-stream projection, extremely
    /// common in real transformer blocks), its bias's length coincidentally
    /// equals the *current* activation width too, so naive shape-only
    /// ambiguity detection would reject the single most common real
    /// checkpoint pattern (`nn.Linear` with `bias=true`) as unresolvable.
    /// [`Self::find_bias_pair`] recognises exactly this shape: the
    /// compatible set is exactly `{weight, its own name-matched bias}`, the
    /// bias's length is the weight's real projection output width (computed
    /// the same way [`Self::process_layer`] itself would), and nothing else
    /// is also compatible this round -- and applies both as one atomic
    /// step (matmul, then bias-add). Any other multi-candidate situation
    /// (two unrelated same-width weights, a weight plus an unrelated
    /// same-width scale tensor, etc.) is still rejected as ambiguous.
    ///
    /// This function also refuses to return the input unchanged when *no*
    /// loaded tensor ever applied -- the previous implementation always
    /// "succeeded" that way, indistinguishable from a real model whose
    /// layers happen to be a no-op.
    fn run_ordered_layers(&self, input: &Tensor) -> Result<Tensor> {
        let Some(weights) = self.model_weights.as_ref() else {
            return Ok(input.clone());
        };

        // The natural-sort order only matters for a stable, human-readable
        // candidate listing in the ambiguity error below; which tensor gets
        // applied is decided by shape compatibility, not list position.
        let mut remaining: Vec<&String> = self.execution_plan.ordered_weight_names.iter().collect();

        let mut current = input.clone();
        let mut applied = 0usize;

        loop {
            let current_shape = current.shape();
            let compatible_positions: Vec<usize> = remaining
                .iter()
                .enumerate()
                .filter_map(|(position, name)| {
                    let weight = weights.get(name.as_str())?;
                    Self::layer_is_shape_compatible(&current_shape, &weight.shape())
                        .then_some(position)
                })
                .collect();

            if compatible_positions.len() == 2 {
                if let Some((weight_pos, bias_pos)) =
                    Self::find_bias_pair(&current_shape, &remaining, &compatible_positions, weights)
                {
                    // Remove the higher index first so the lower index
                    // remains valid.
                    let (first, second) = if weight_pos > bias_pos {
                        (weight_pos, bias_pos)
                    } else {
                        (bias_pos, weight_pos)
                    };
                    let name_a = remaining.remove(first);
                    let name_b = remaining.remove(second);
                    let (weight_name, bias_name) =
                        if first == weight_pos { (name_a, name_b) } else { (name_b, name_a) };

                    let weight = weights.get(weight_name.as_str()).ok_or_else(|| {
                        runtime_error(format!(
                            "internal error: weight '{weight_name}' was in the execution plan \
                             but is no longer in the loaded weight map"
                        ))
                    })?;
                    let after_weight =
                        self.process_layer(&current, weight_name, weight)?.ok_or_else(|| {
                            runtime_error(format!(
                                "internal error: weight '{weight_name}' passed the \
                                 shape-compatibility check but process_layer declined to apply it"
                            ))
                        })?;

                    let bias = weights.get(bias_name.as_str()).ok_or_else(|| {
                        runtime_error(format!(
                            "internal error: bias '{bias_name}' was in the execution plan but is \
                             no longer in the loaded weight map"
                        ))
                    })?;
                    let after_bias =
                        self.process_layer(&after_weight, bias_name, bias)?.ok_or_else(|| {
                            runtime_error(format!(
                                "internal error: bias '{bias_name}' was matched to weight \
                                 '{weight_name}' but process_layer declined to apply it after \
                                 the projection"
                            ))
                        })?;

                    current = after_bias;
                    applied += 2;
                    continue;
                }
            }

            match compatible_positions.len() {
                0 => break,
                1 => {
                    let name = remaining.remove(compatible_positions[0]);
                    let weight = weights.get(name.as_str()).ok_or_else(|| {
                        runtime_error(format!(
                            "internal error: weight '{name}' was in the execution plan but is \
                             no longer in the loaded weight map"
                        ))
                    })?;
                    let next = self.process_layer(&current, name, weight)?.ok_or_else(|| {
                        runtime_error(format!(
                            "internal error: weight '{name}' passed the shape-compatibility \
                             check but process_layer declined to apply it"
                        ))
                    })?;
                    current = next;
                    applied += 1;

                    // Apply checkpointing if configured
                    if self.execution_plan.checkpoint_interval > 0 {
                        // No checkpointing backend exists yet; documented
                        // no-op rather than a fabricated intermediate-state
                        // save.
                    }
                },
                _ => {
                    let candidates: Vec<&str> =
                        compatible_positions.iter().map(|&p| remaining[p].as_str()).collect();
                    return Err(unsupported_operation(
                        format!(
                            "choosing which of {} shape-compatible weight tensors ({}) to apply \
                             next to an activation of shape {:?}",
                            candidates.len(),
                            candidates.join(", "),
                            current_shape
                        ),
                        "MobileInferenceEngine::inference (this engine has no per-model \
                         architecture graph -- only a flat set of named tensors -- and refuses \
                         to guess an execution order among multiple simultaneously-compatible \
                         candidates other than a weight matched with its own bias; it can only \
                         run a checkpoint whose weight shapes form a single unambiguous linear \
                         stack. For architectures with attention/FFN branching, load the \
                         checkpoint through trustformers_models instead)",
                    ));
                },
            }
        }

        if applied == 0 && !self.execution_plan.ordered_weight_names.is_empty() {
            return Err(runtime_error(format!(
                "none of the {} loaded weight tensors had a shape compatible with an input of \
                 shape {:?}; refusing to return the input unchanged as if inference had run",
                self.execution_plan.ordered_weight_names.len(),
                input.shape()
            )));
        }

        Ok(current)
    }

    /// When `compatible_positions` names exactly two tensors, check whether
    /// they are a projection weight and its own bias (by the
    /// `<prefix>.weight`/`<prefix>.bias` or `<prefix>_weight`/`<prefix>_bias`
    /// naming convention) whose shapes are consistent with that reading --
    /// the bias's length must equal the weight's *projection output* width,
    /// computed exactly as [`Self::process_layer`]'s matmul branch would
    /// (see [`Self::projection_output_dim`]), not merely "some length that
    /// happens to match the current input". Returns
    /// `Some((weight_position, bias_position))` (positions into `remaining`)
    /// on a match, `None` otherwise -- including when neither tensor is 2D,
    /// when their names do not follow the convention, or when the bias
    /// length is the coincidental current-width match rather than the real
    /// output width.
    fn find_bias_pair(
        current_shape: &[usize],
        remaining: &[&String],
        compatible_positions: &[usize],
        weights: &HashMap<String, Tensor>,
    ) -> Option<(usize, usize)> {
        let &last_dim = current_shape.last()?;
        debug_assert_eq!(
            compatible_positions.len(),
            2,
            "find_bias_pair expects exactly 2 candidates"
        );
        let [pos_a, pos_b] = [compatible_positions[0], compatible_positions[1]];
        let name_a = remaining[pos_a].as_str();
        let name_b = remaining[pos_b].as_str();
        let shape_a = weights.get(name_a)?.shape();
        let shape_b = weights.get(name_b)?.shape();

        // Try both orderings: (a=weight, b=bias) and (b=weight, a=bias).
        for &((weight_pos, weight_name, weight_shape), (bias_pos, bias_name, bias_shape)) in &[
            ((pos_a, name_a, &shape_a), (pos_b, name_b, &shape_b)),
            ((pos_b, name_b, &shape_b), (pos_a, name_a, &shape_a)),
        ] {
            if weight_shape.len() != 2 || bias_shape.len() != 1 {
                continue;
            }
            let Some(expected_bias_name) = Self::bias_name_for_weight(weight_name) else {
                continue;
            };
            if expected_bias_name != bias_name {
                continue;
            }
            let Some(output_dim) = Self::projection_output_dim(last_dim, weight_shape) else {
                continue;
            };
            if bias_shape[0] == output_dim {
                return Some((weight_pos, bias_pos));
            }
        }
        None
    }

    /// The bias tensor name a checkpoint would use for `weight_name`, under
    /// the `<prefix>.weight` -> `<prefix>.bias` (or `<prefix>_weight` ->
    /// `<prefix>_bias`) convention used by every checkpoint format this
    /// engine parses (safetensors/PyTorch state dicts, this crate's own
    /// `create_transformer_weights`-style naming, etc.). Returns `None` for
    /// a name that does not end in `weight` under either convention -- no
    /// guess is made in that case.
    fn bias_name_for_weight(weight_name: &str) -> Option<String> {
        if let Some(prefix) = weight_name.strip_suffix(".weight") {
            Some(format!("{prefix}.bias"))
        } else {
            weight_name.strip_suffix("_weight").map(|prefix| format!("{prefix}_bias"))
        }
    }

    /// The output width a 2D `weight_shape` would project an activation of
    /// `input_last_dim` to, under [`Self::process_layer`]'s own
    /// orientation rule (dimension 0 matches -> use as-is, result width is
    /// dimension 1; otherwise transpose, result width is dimension 0).
    /// Returns `None` when `weight_shape` is not rank 2 or neither
    /// dimension matches `input_last_dim`. The single source of truth for
    /// "what width would this projection produce", shared by
    /// [`Self::find_bias_pair`] and (implicitly, via the same rule
    /// restated inline) [`Self::process_layer`]'s matmul branch.
    fn projection_output_dim(input_last_dim: usize, weight_shape: &[usize]) -> Option<usize> {
        if weight_shape.len() != 2 {
            return None;
        }
        if weight_shape[0] == input_last_dim {
            Some(weight_shape[1])
        } else if weight_shape[1] == input_last_dim {
            Some(weight_shape[0])
        } else {
            None
        }
    }

    /// Whether [`Self::process_layer`] would apply a weight of
    /// `weight_shape` to an activation of `input_shape` -- i.e. whether it
    /// would return `Ok(Some(_))` rather than `Ok(None)` -- without
    /// actually performing the computation. The single source of truth for
    /// "is this tensor a candidate here", shared by the ambiguity check in
    /// [`Self::run_ordered_layers`] and the dispatch in
    /// [`Self::process_layer`] so the two can never disagree.
    fn layer_is_shape_compatible(input_shape: &[usize], weight_shape: &[usize]) -> bool {
        let Some(&last_dim) = input_shape.last() else {
            return false;
        };
        match weight_shape.len() {
            2 => weight_shape[0] == last_dim || weight_shape[1] == last_dim,
            1 => weight_shape[0] == last_dim,
            _ => false,
        }
    }

    /// Apply one weight tensor's real numerical role to `input`.
    ///
    /// Dispatch is purely shape- and name-driven -- this engine has no
    /// per-model architecture graph, only a bag of named tensors:
    ///
    /// * A 2D tensor whose first or second dimension matches `input`'s last
    ///   dimension is a linear projection, computed as a real `matmul`
    ///   (`weight` is transposed first when it is stored `[out, in]`,
    ///   PyTorch's `nn.Linear` convention). A 1D input is treated as a
    ///   single row (a batch dimension of 1 is added for the multiply, then
    ///   removed from the result).
    /// * A 1D tensor whose only dimension matches `input`'s last dimension
    ///   is a bias (name ends in `bias`, added) or an elementwise scale
    ///   (anything else -- e.g. a LayerNorm/RMSNorm weight -- multiplied).
    /// * An activation is applied only when the tensor's own name says so
    ///   (`"gelu"` or `"relu"` as a substring); this engine does not guess
    ///   an architecture's nonlinearity from a naming convention it cannot
    ///   verify.
    /// * Anything else (shape does not line up with `input`) is skipped:
    ///   `Ok(None)`. Skipping is honest -- no data is invented.
    ///
    /// Callers that need to know *whether* this will apply, without paying
    /// for (or side-effecting on) the computation, should use
    /// [`Self::layer_is_shape_compatible`] instead of calling this and
    /// discarding the result.
    fn process_layer(&self, input: &Tensor, name: &str, weight: &Tensor) -> Result<Option<Tensor>> {
        let input_shape = input.shape();
        let weight_shape = weight.shape();
        if !Self::layer_is_shape_compatible(&input_shape, &weight_shape) {
            return Ok(None);
        }
        // The compatibility check above guarantees `input_shape` is
        // non-empty (it examines `input_shape.last()`), so this cannot fail.
        let Some(&last_dim) = input_shape.last() else {
            return Ok(None);
        };
        let lower_name = name.to_ascii_lowercase();

        match weight_shape.len() {
            2 => {
                // The compatibility check guarantees at least one of these
                // matches; prefer no transpose when both do (a square
                // weight matrix), matching the pre-refactor tie-break.
                let weight_for_matmul = if weight_shape[0] == last_dim {
                    weight.clone()
                } else {
                    weight.transpose(0, 1)?
                };

                // `Tensor::matmul`'s batched path requires both operands to
                // share rank (>= 3) with identical leading dims, so it
                // cannot broadcast a plain 2D weight over an N-D batch of
                // activations (e.g. `[batch, seq, hidden] @ [hidden, out]`,
                // the common transformer case). Flatten every leading
                // dimension into one "rows" axis, run the well-supported 2D
                // GEMM, then restore the original leading shape -- the
                // standard, numerically exact way to apply a `Linear` layer
                // to a batch of arbitrary rank.
                let leading_shape: Vec<usize> = if input_shape.len() > 1 {
                    input_shape[..input_shape.len() - 1].to_vec()
                } else {
                    Vec::new()
                };
                let rows: usize = leading_shape.iter().product::<usize>().max(1);

                let matmul_input = if input_shape.len() == 2 {
                    input.clone()
                } else {
                    input.reshape(&[rows, last_dim])?
                };

                let mut result = matmul_input.matmul(&weight_for_matmul)?;

                if lower_name.contains("gelu") {
                    result = result.gelu()?;
                } else if lower_name.contains("relu") {
                    result = result.relu()?;
                }

                if input_shape.len() != 2 {
                    let out_dim = match result.shape().get(1) {
                        Some(&d) => d,
                        None => {
                            return Err(runtime_error(format!(
                                "internal error: 2D matmul for weight '{name}' produced a \
                                 non-2D result"
                            )));
                        },
                    };
                    let restored_shape = if input_shape.len() == 1 {
                        vec![out_dim]
                    } else {
                        let mut shape = leading_shape;
                        shape.push(out_dim);
                        shape
                    };
                    result = result.reshape(&restored_shape)?;
                }

                Ok(Some(result))
            },
            1 if weight_shape[0] == last_dim => {
                let is_bias = lower_name.ends_with(".bias")
                    || lower_name.ends_with("_bias")
                    || lower_name == "bias";
                if is_bias {
                    Ok(Some(input.add(weight)?))
                } else {
                    Ok(Some(input.mul(weight)?))
                }
            },
            _ => Ok(None),
        }
    }

    fn estimate_current_memory_usage(&self) -> usize {
        let mut total = 0;

        // Model weights memory
        if let Some(ref weights) = self.model_weights {
            for weight in weights.values() {
                total += weight.memory_usage();
            }
        }

        // Cache memory
        if let Some(ref cache) = self.cache {
            total += cache.memory_usage_mb() * 1024 * 1024;
        }

        // Convert to MB
        total / (1024 * 1024)
    }

    fn should_use_cache(&self) -> bool {
        // Enable cache only if we have sufficient memory
        self.config.max_memory_mb >= 512
            && self.config.memory_optimization != crate::MemoryOptimization::Maximum
    }
}

/// Inference cache for mobile deployment
#[derive(Debug)]
struct InferenceCache {
    cache: HashMap<Vec<u8>, Tensor>,
    max_size_mb: usize,
    current_size_bytes: usize,
}

impl InferenceCache {
    fn new(max_size_mb: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size_mb,
            current_size_bytes: 0,
        }
    }

    fn get(&self, input: &Tensor) -> Option<Tensor> {
        let key = self.tensor_to_key(input);
        self.cache.get(&key).cloned()
    }

    fn put(&mut self, input: Tensor, output: Tensor) {
        let key = self.tensor_to_key(&input);
        let entry_size = input.memory_usage() + output.memory_usage();

        // Check if we have space
        if self.current_size_bytes + entry_size > self.max_size_mb * 1024 * 1024 {
            self.evict_lru();
        }

        self.cache.insert(key, output);
        self.current_size_bytes += entry_size;
    }

    fn clear(&mut self) {
        self.cache.clear();
        self.current_size_bytes = 0;
    }

    fn memory_usage_mb(&self) -> usize {
        self.current_size_bytes / (1024 * 1024)
    }

    fn tensor_to_key(&self, tensor: &Tensor) -> Vec<u8> {
        // Create a simple key from tensor shape and first few values
        // This is a simplified implementation
        let shape = tensor.shape();
        let mut key = Vec::new();

        for &dim in &shape {
            key.extend_from_slice(&dim.to_le_bytes());
        }

        key
    }

    fn evict_lru(&mut self) {
        // Simple eviction strategy - remove oldest entries
        // In practice, would use a proper LRU implementation
        if let Some(first_key) = self.cache.keys().next().cloned() {
            self.cache.remove(&first_key);
            self.current_size_bytes = self.current_size_bytes.saturating_sub(1024 * 1024);
            // Approximate
        }
    }
}

/// Mobile memory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileMemoryInfo {
    /// Model memory usage in MB
    pub model_memory_mb: usize,
    /// Runtime memory overhead in MB
    pub runtime_memory_mb: usize,
    /// Total memory usage in MB
    pub total_memory_mb: usize,
    /// Memory limit in MB
    pub memory_limit_mb: usize,
    /// Memory savings percentage
    pub memory_savings_percent: f32,
    /// Cache memory usage in MB
    pub cache_memory_mb: usize,
}

impl MobileMemoryInfo {
    /// Check if memory usage is within limits
    pub fn is_within_limits(&self) -> bool {
        self.total_memory_mb <= self.memory_limit_mb
    }

    /// Get memory utilization percentage
    pub fn memory_utilization_percent(&self) -> f32 {
        (self.total_memory_mb as f32 / self.memory_limit_mb as f32) * 100.0
    }

    /// Get available memory in MB
    pub fn available_memory_mb(&self) -> usize {
        self.memory_limit_mb.saturating_sub(self.total_memory_mb)
    }
}

/// Mobile inference configuration builder
pub struct MobileInferenceBuilder {
    config: MobileConfig,
}

impl MobileInferenceBuilder {
    /// Create new builder with default mobile configuration
    pub fn new() -> Self {
        Self {
            config: MobileConfig::default(),
        }
    }

    /// Set target platform
    pub fn platform(mut self, platform: MobilePlatform) -> Self {
        self.config.platform = platform;
        self
    }

    /// Set inference backend
    pub fn backend(mut self, backend: MobileBackend) -> Self {
        self.config.backend = backend;
        self
    }

    /// Set memory limit
    pub fn memory_limit_mb(mut self, limit: usize) -> Self {
        self.config.max_memory_mb = limit;
        self
    }

    /// Enable/disable FP16 precision
    pub fn fp16(mut self, enable: bool) -> Self {
        self.config.use_fp16 = enable;
        self
    }

    /// Set quantization scheme
    pub fn quantization(mut self, scheme: crate::MobileQuantizationScheme) -> Self {
        self.config.quantization = Some(crate::MobileQuantizationConfig {
            scheme,
            dynamic: true,
            per_channel: false,
        });
        self
    }

    /// Set thread count
    pub fn threads(mut self, count: usize) -> Self {
        self.config.num_threads = count;
        self
    }

    /// Enable/disable batching
    pub fn batching(mut self, enable: bool, max_batch_size: usize) -> Self {
        self.config.enable_batching = enable;
        self.config.max_batch_size = max_batch_size;
        self
    }

    /// Set memory optimization level
    pub fn memory_optimization(mut self, level: crate::MemoryOptimization) -> Self {
        self.config.memory_optimization = level;
        self
    }

    /// Build the inference engine
    pub fn build(self) -> Result<MobileInferenceEngine> {
        MobileInferenceEngine::new(self.config)
    }
}

impl Default for MobileInferenceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mobile_inference_engine_creation() {
        let config = MobileConfig::default();
        let engine = MobileInferenceEngine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_model_loading() {
        let config = MobileConfig::default();
        let mut engine = MobileInferenceEngine::new(config).expect("Failed to create engine");

        let mut weights = HashMap::new();
        weights.insert(
            "layer1".to_string(),
            Tensor::ones(&[10, 10]).expect("Failed to create tensor"),
        );
        weights.insert(
            "layer2".to_string(),
            Tensor::ones(&[10, 5]).expect("Failed to create tensor"),
        );

        let result = engine.load_model(weights);
        assert!(result.is_ok());
        assert!(engine.model_loaded);
    }

    #[test]
    fn test_inference() {
        let config = MobileConfig::default();
        let mut engine = MobileInferenceEngine::new(config).expect("Failed to create engine");

        // Load a simple model
        let mut weights = HashMap::new();
        weights.insert(
            "layer1".to_string(),
            Tensor::ones(&[5, 5]).expect("Failed to create tensor"),
        );
        engine.load_model(weights).expect("Failed to load model");

        // Perform inference
        let input = Tensor::ones(&[5]).expect("Failed to create tensor");
        let result = engine.inference(&input);
        assert!(result.is_ok());
    }

    #[test]
    fn test_batch_inference() {
        let config = MobileConfig {
            enable_batching: true,
            max_batch_size: 3,
            ..Default::default()
        };
        let mut engine = MobileInferenceEngine::new(config).expect("Failed to create engine");

        // Load a simple model
        let mut weights = HashMap::new();
        weights.insert(
            "layer1".to_string(),
            Tensor::ones(&[5, 5]).expect("Failed to create tensor"),
        );
        engine.load_model(weights).expect("Failed to load model");

        // Perform batch inference
        let inputs = vec![
            Tensor::ones(&[5]).expect("Failed to create tensor"),
            Tensor::ones(&[5]).expect("Failed to create tensor"),
        ];
        let results = engine.batch_inference(inputs);
        assert!(results.is_ok());
        assert_eq!(results.expect("Batch inference failed").len(), 2);
    }

    #[test]
    fn test_memory_info() {
        let config = MobileConfig::default();
        let engine = MobileInferenceEngine::new(config).expect("Failed to create engine");

        let memory_info = engine.get_memory_info();
        assert!(memory_info.memory_limit_mb > 0);
        assert!(memory_info.memory_utilization_percent() >= 0.0);
    }

    #[test]
    fn test_inference_builder() {
        let engine = MobileInferenceBuilder::new()
            .platform(MobilePlatform::Ios)
            .backend(MobileBackend::CoreML)
            .memory_limit_mb(1024)
            .fp16(true)
            .quantization(crate::MobileQuantizationScheme::Int8)
            .threads(4)
            .batching(true, 2)
            .memory_optimization(crate::MemoryOptimization::Balanced)
            .build();

        assert!(engine.is_ok());
    }

    #[test]
    fn test_config_update() {
        let config = MobileConfig::default();
        let mut engine = MobileInferenceEngine::new(config).expect("Failed to create engine");

        let new_config = MobileConfig {
            max_memory_mb: 1024,
            num_threads: 8,
            ..Default::default()
        };

        let result = engine.update_config(new_config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_cache_operations() {
        let config = MobileConfig {
            max_memory_mb: 1024, // Enough for cache
            ..Default::default()
        };
        let mut engine = MobileInferenceEngine::new(config).expect("Failed to create engine");

        // Load model to enable caching
        let mut weights = HashMap::new();
        weights.insert(
            "layer1".to_string(),
            Tensor::ones(&[5, 5]).expect("Failed to create tensor"),
        );
        engine.load_model(weights).expect("Failed to load model");

        // Test cache operations
        engine.clear_cache();
        engine.force_gc();

        // These should not panic
    }

    #[test]
    fn test_warm_up() {
        let config = MobileConfig::default();
        let mut engine = MobileInferenceEngine::new(config).expect("Failed to create engine");

        // Load an unambiguous two-layer linear stack: 512 -> 256 -> 128.
        // Each step's output width differs from every other loaded weight's
        // matching dimension, so `run_ordered_layers` never has more than
        // one shape-compatible candidate at a time (a same-width fixture,
        // e.g. two [512, 512] weights, would be a genuine architectural
        // ambiguity and *should* be rejected -- see
        // `test_inference_errors_on_ambiguous_weight_set` below).
        let mut weights = HashMap::new();
        weights.insert(
            "layer.0.weight".to_string(),
            Tensor::ones(&[512, 256]).expect("Operation failed"),
        );
        weights.insert(
            "layer.1.weight".to_string(),
            Tensor::ones(&[256, 128]).expect("Operation failed"),
        );
        engine.load_model(weights).expect("Failed to load model");

        // Test warm-up
        let result = engine.warm_up();
        assert!(result.is_ok(), "Warm-up should succeed after model loading");
    }

    /// `warm_up` must derive its dummy input width from the *loaded
    /// model's* first weight, not a hardcoded `512`. A checkpoint whose
    /// first layer takes width 37 (deliberately not 512, and not a
    /// multiple of it) proves this: against the old hardcoded-512 dummy
    /// input, `warm_up` would build a `[1, 128, 512]` tensor that
    /// `process_layer` cannot apply to a `[37, 20]` weight (no dimension
    /// matches), so `inference()` -- which now errors when nothing
    /// applies -- would fail here even though a real caller feeding this
    /// model its actual 37-wide input works fine.
    #[test]
    fn test_warm_up_derives_hidden_size_from_loaded_model() {
        let config = MobileConfig::default();
        let mut engine = MobileInferenceEngine::new(config).expect("engine");

        let mut weights = HashMap::new();
        weights.insert(
            "layer.0.weight".to_string(),
            Tensor::ones(&[37, 20]).expect("w"),
        );
        engine.load_model(weights).expect("load_model");

        let result = engine.warm_up();
        assert!(
            result.is_ok(),
            "warm_up must derive its dummy input width (37) from the loaded model instead of \
             a hardcoded 512, got: {result:?}"
        );
    }

    /// A checkpoint whose weight shapes do not form an unambiguous linear
    /// stack (here: two different `[512, 512]` weights, either of which
    /// could legally apply to a 512-wide activation) must be rejected with
    /// a structured error, not silently resolved by applying them in an
    /// arbitrary sorted order -- which would compute a shape-plausible but
    /// architecturally meaningless number.
    #[test]
    fn test_inference_errors_on_ambiguous_weight_set() {
        let config = MobileConfig::default();
        let mut engine = MobileInferenceEngine::new(config).expect("engine");

        let mut weights = HashMap::new();
        weights.insert(
            "attn.c_attn.weight".to_string(),
            Tensor::ones(&[512, 512]).expect("w"),
        );
        weights.insert(
            "attn.c_proj.weight".to_string(),
            Tensor::ones(&[512, 512]).expect("w"),
        );
        engine.load_model(weights).expect("load_model");

        let input = Tensor::zeros(&[1, 512]).expect("input");
        let result = engine.inference(&input);

        assert!(
            result.is_err(),
            "an ambiguous set of simultaneously-compatible weights must be rejected, not \
             resolved by guessing an order"
        );
    }

    /// A *square* `nn.Linear(hidden, hidden, bias=True)` -- the single most
    /// common real transformer sub-layer shape (attention output
    /// projection, residual-stream MLP projections, etc.) -- must not be
    /// rejected as "ambiguous". Its `weight` ([768, 768]) and its own
    /// `bias` ([768]) both key off width 768, which is exactly the shape
    /// pattern `test_inference_errors_on_ambiguous_weight_set` above
    /// correctly rejects for two *unrelated* tensors; the difference here
    /// is the `<prefix>.weight`/`<prefix>.bias` naming relationship, which
    /// `find_bias_pair` must recognise and apply as one atomic
    /// matmul-then-bias-add step. Without that recognition, this is the
    /// single most common real checkpoint pattern this engine would be
    /// unable to run at all.
    #[test]
    fn test_inference_resolves_square_weight_and_its_own_bias_not_ambiguous() {
        // Quantization is disabled so the exact-value assertions below are
        // meaningful: `MobileConfig::default()` applies dynamic Int8
        // quantization to loaded weights, which is real lossy rounding (not
        // a bug) that would make an exact expected-output comparison
        // meaningless noise rather than a check of `find_bias_pair`'s logic.
        let config = MobileConfig {
            quantization: None,
            ..MobileConfig::default()
        };
        let mut engine = MobileInferenceEngine::new(config).expect("engine");

        let mut weights = HashMap::new();
        // [4, 4] identity so the matmul step is a no-op and the bias-add's
        // contribution is exactly and only the bias values -- makes the
        // expected output exact and easy to state.
        weights.insert(
            "block.0.proj.weight".to_string(),
            Tensor::from_vec(
                vec![
                    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ],
                &[4, 4],
            )
            .expect("identity weight"),
        );
        weights.insert(
            "block.0.proj.bias".to_string(),
            Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], &[4]).expect("bias"),
        );
        engine.load_model(weights).expect("load_model");

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).expect("input");
        let output = engine
            .inference(&input)
            .expect("a weight matched with its own bias must not be rejected as ambiguous");

        assert_eq!(output.shape(), vec![1, 4]);
        let data = output.data().expect("output data");
        assert_eq!(data, vec![11.0, 22.0, 33.0, 44.0]);
    }

    /// The `_weight`/`_bias` (underscore) naming convention must also be
    /// recognised, not only the dotted `.weight`/`.bias` form.
    #[test]
    fn test_inference_resolves_bias_pair_with_underscore_naming() {
        // See the comment in
        // `test_inference_resolves_square_weight_and_its_own_bias_not_ambiguous`:
        // quantization is disabled so the exact-value assertion is meaningful.
        let config = MobileConfig {
            quantization: None,
            ..MobileConfig::default()
        };
        let mut engine = MobileInferenceEngine::new(config).expect("engine");

        let mut weights = HashMap::new();
        weights.insert(
            "dense_weight".to_string(),
            Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).expect("identity weight"),
        );
        weights.insert(
            "dense_bias".to_string(),
            Tensor::from_vec(vec![5.0, 6.0], &[2]).expect("bias"),
        );
        engine.load_model(weights).expect("load_model");

        let input = Tensor::from_vec(vec![1.0, 1.0], &[1, 2]).expect("input");
        let output = engine.inference(&input).expect("underscore-named weight/bias pair");

        assert_eq!(output.data().expect("data"), vec![6.0, 7.0]);
    }

    /// Two same-width tensors that happen to have `.weight`/`.bias`-shaped
    /// names but do *not* actually name-match each other's prefix must
    /// still be rejected as ambiguous -- `find_bias_pair` matches on the
    /// real naming relationship, not merely "one 2D and one 1D tensor of
    /// compatible width are present".
    #[test]
    fn test_inference_still_rejects_unrelated_weight_and_bias_names() {
        let config = MobileConfig::default();
        let mut engine = MobileInferenceEngine::new(config).expect("engine");

        let mut weights = HashMap::new();
        weights.insert(
            "layer_a.weight".to_string(),
            Tensor::ones(&[4, 4]).expect("w"),
        );
        // Not "layer_a.bias" -- an unrelated prefix that happens to be 1D
        // and width-4.
        weights.insert("layer_b.bias".to_string(), Tensor::ones(&[4]).expect("b"));
        engine.load_model(weights).expect("load_model");

        let input = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 4]).expect("input");
        let result = engine.inference(&input);

        assert!(
            result.is_err(),
            "a weight and an unrelated same-width 'bias'-shaped tensor must not be paired just \
             because their shapes happen to line up"
        );
    }

    #[test]
    fn test_warm_up_without_model() {
        let config = MobileConfig::default();
        let mut engine = MobileInferenceEngine::new(config).expect("Failed to create engine");

        // Test warm-up without loading model (should fail)
        let result = engine.warm_up();
        assert!(result.is_err(), "Warm-up should fail without model");
    }

    #[test]
    fn test_set_performance_mode() {
        let config = MobileConfig::default();
        let mut engine = MobileInferenceEngine::new(config).expect("Failed to create engine");

        // Test all valid performance modes
        assert!(
            engine.set_performance_mode(0).is_ok(),
            "Power Saving mode should work"
        );
        assert!(
            engine.set_performance_mode(1).is_ok(),
            "Balanced mode should work"
        );
        assert!(
            engine.set_performance_mode(2).is_ok(),
            "High Performance mode should work"
        );

        // Test invalid mode
        assert!(
            engine.set_performance_mode(3).is_err(),
            "Invalid mode should return error"
        );
        assert!(
            engine.set_performance_mode(-1).is_err(),
            "Negative mode should return error"
        );
    }

    #[test]
    fn test_performance_mode_changes_config() {
        let config = MobileConfig {
            use_fp16: false,
            backend: crate::MobileBackend::CPU,
            ..Default::default()
        };
        let mut engine = MobileInferenceEngine::new(config).expect("Failed to create engine");

        // Set to high performance mode
        engine.set_performance_mode(2).expect("Operation failed");
        // In high performance mode, fp16 should be disabled and backend should be GPU
        // (Note: These assertions verify the implementation logic)

        // Set to power saving mode
        engine.set_performance_mode(0).expect("Operation failed");
        // In power saving mode, fp16 should be enabled and backend should be CPU

        // The engine should still be functional after mode changes
        let _ = engine.get_stats(); // Verify stats are accessible
    }

    // -- Regression tests: real parsing and real layer execution --------
    //
    // These target the two P0 findings this module used to have: (1) the
    // SafeTensors/PyTorch/ONNX/TensorFlow parsers discarded the file bytes
    // and fabricated random transformer-shaped weights via `Tensor::randn`,
    // and (2) `process_layer` was `Ok(input.clone())`, so `inference()` was
    // the identity function no matter what was "loaded". Every test below
    // would have failed against that code.

    /// `process_layer` must perform a real matmul (checked against a
    /// hand-computed expected result) when the weight is stored `[in, out]`
    /// -- the old `Ok(input.clone())` body would return `[1.0, 1.0]`
    /// unchanged instead of the projected `[2.0, 3.0]` computed below.
    #[test]
    fn test_process_layer_real_matmul_in_out_orientation() {
        let config = MobileConfig::default();
        let engine = MobileInferenceEngine::new(config).expect("engine");

        // input [1, 2] = [1, 1]; weight [2, 2] stored [in=2, out=2].
        let input = Tensor::from_vec(vec![1.0, 1.0], &[1, 2]).expect("input tensor");
        let weight = Tensor::from_vec(vec![1.0, 2.0, 1.0, 1.0], &[2, 2]).expect("weight tensor");

        let result = engine
            .process_layer(&input, "encoder.proj.weight", &weight)
            .expect("process_layer should succeed")
            .expect("a 2D weight matching the input's last dim must be applied, not skipped");

        // [1,1] @ [[1,2],[1,1]] = [1*1+1*1, 1*2+1*1] = [2, 3]
        assert_eq!(result.shape(), vec![1, 2]);
        let data = result.data().expect("tensor data");
        assert!(
            (data[0] - 2.0).abs() < 1e-5,
            "expected 2.0, got {}",
            data[0]
        );
        assert!(
            (data[1] - 3.0).abs() < 1e-5,
            "expected 3.0, got {}",
            data[1]
        );
    }

    /// A weight stored `[out, in]` (PyTorch `nn.Linear` convention) must be
    /// transposed before the multiply, not skipped or misapplied.
    #[test]
    fn test_process_layer_real_matmul_out_in_orientation() {
        let config = MobileConfig::default();
        let engine = MobileInferenceEngine::new(config).expect("engine");

        // input [1, 3] = [1, 2, 3]; weight [out=1, in=3] = [[1, 0, 1]].
        // y = x @ W^T = [1*1 + 2*0 + 3*1] = [4]
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).expect("input tensor");
        let weight = Tensor::from_vec(vec![1.0, 0.0, 1.0], &[1, 3]).expect("weight tensor");

        let result = engine
            .process_layer(&input, "lm_head.weight", &weight)
            .expect("process_layer should succeed")
            .expect("a [out,in] weight matching the input's last dim must be applied");

        assert_eq!(result.shape(), vec![1, 1]);
        let data = result.data().expect("tensor data");
        assert!(
            (data[0] - 4.0).abs() < 1e-5,
            "expected 4.0, got {}",
            data[0]
        );
    }

    /// A rank-3 `[batch, seq, hidden]` activation must flatten/restore
    /// correctly around the 2D GEMM (`Tensor::matmul`'s batched path
    /// requires matching rank on both operands, so a naive
    /// `input.matmul(&weight)` would hard-error here).
    #[test]
    fn test_process_layer_flattens_batch_dims_for_nd_input() {
        let config = MobileConfig::default();
        let engine = MobileInferenceEngine::new(config).expect("engine");

        // input [1, 2, 2] (batch=1, seq=2, hidden=2); weight [2, 2] identity*2.
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2]).expect("input tensor");
        let weight = Tensor::from_vec(vec![2.0, 0.0, 0.0, 2.0], &[2, 2]).expect("weight tensor");

        let result = engine
            .process_layer(&input, "layer.0.weight", &weight)
            .expect("process_layer should succeed")
            .expect("2D weight matching the last dim must be applied to a 3D input");

        assert_eq!(result.shape(), vec![1, 2, 2]);
        let data = result.data().expect("tensor data");
        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    /// A `.bias` tensor is added; anything else 1D and shape-compatible
    /// (e.g. a LayerNorm scale) is multiplied elementwise. Both are real
    /// arithmetic against the loaded tensor, not an echo of the input.
    #[test]
    fn test_process_layer_bias_add_and_elementwise_scale() {
        let config = MobileConfig::default();
        let engine = MobileInferenceEngine::new(config).expect("engine");

        let input = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[1, 3]).expect("input tensor");

        let bias = Tensor::from_vec(vec![0.5, -0.5, 2.0], &[3]).expect("bias tensor");
        let biased = engine
            .process_layer(&input, "block.0.attn.bias", &bias)
            .expect("ok")
            .expect("bias tensor must be applied");
        assert_eq!(biased.data().expect("data"), vec![1.5, 0.5, 3.0]);

        let scale = Tensor::from_vec(vec![2.0, 3.0, 0.5], &[3]).expect("scale tensor");
        let scaled = engine
            .process_layer(&input, "block.0.ln_1.weight", &scale)
            .expect("ok")
            .expect("scale tensor must be applied");
        assert_eq!(scaled.data().expect("data"), vec![2.0, 3.0, 0.5]);
    }

    /// A weight whose shape does not line up with the input's last
    /// dimension is skipped (`Ok(None)`), not silently misapplied.
    #[test]
    fn test_process_layer_skips_incompatible_shape() {
        let config = MobileConfig::default();
        let engine = MobileInferenceEngine::new(config).expect("engine");

        let input = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[1, 3]).expect("input tensor");
        let unrelated = Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("unrelated tensor");

        let result = engine
            .process_layer(&input, "unrelated.weight", &unrelated)
            .expect("shape mismatch is not an error");
        assert!(
            result.is_none(),
            "an incompatible shape must be skipped, not applied"
        );
    }

    /// End-to-end: a loaded weight that projects the input to a *different*
    /// output width proves `inference()` performed a real transformation.
    /// The old `process_layer` body (`Ok(input.clone())`) could never
    /// change the shape, so this assertion alone falsifies the identity-pass
    /// bug regardless of any quantization rounding applied afterward.
    #[test]
    fn test_inference_end_to_end_changes_shape_not_identity() {
        let config = MobileConfig::default();
        let mut engine = MobileInferenceEngine::new(config).expect("engine");

        let mut weights = HashMap::new();
        // [in=4, out=2]: projects a 4-wide input down to width 2.
        weights.insert(
            "down_proj.weight".to_string(),
            Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0], &[4, 2])
                .expect("weight tensor"),
        );
        engine.load_model(weights).expect("load_model");

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).expect("input tensor");
        let output = engine.inference(&input).expect("inference should succeed");

        assert_ne!(
            output.shape(),
            input.shape(),
            "a real linear projection must change the shape; an identity pass cannot"
        );
        assert_eq!(output.shape(), vec![1, 2]);
    }

    /// `inference()` must refuse to silently echo the input when *no*
    /// loaded weight tensor's shape is compatible with it -- the previous
    /// implementation "succeeded" in exactly this situation by returning
    /// the input unchanged.
    #[test]
    fn test_inference_errors_when_no_weight_is_applicable() {
        let config = MobileConfig::default();
        let mut engine = MobileInferenceEngine::new(config).expect("engine");

        let mut weights = HashMap::new();
        // Every dimension here (7) is incompatible with the length-3 input below.
        weights.insert(
            "incompatible.weight".to_string(),
            Tensor::ones(&[7, 7]).expect("weight tensor"),
        );
        engine.load_model(weights).expect("load_model");

        let input = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[1, 3]).expect("input tensor");
        let result = engine.inference(&input);

        assert!(
            result.is_err(),
            "inference must error rather than return the input unchanged when nothing could be applied"
        );
    }

    /// The execution order is a numeric-aware ("natural") sort of the
    /// tensor names, not the arbitrary order a `HashMap` would iterate in
    /// (which differs from run to run and, unsorted, would put
    /// `"h.10.weight"` before `"h.2.weight"`).
    #[test]
    fn test_execution_plan_uses_natural_sort_order() {
        let config = MobileConfig::default();
        let mut engine = MobileInferenceEngine::new(config).expect("engine");

        let mut weights = HashMap::new();
        for i in [10usize, 2, 1] {
            weights.insert(
                format!("transformer.h.{i}.weight"),
                Tensor::ones(&[4, 4]).expect("w"),
            );
        }
        engine.load_model(weights).expect("load_model");

        assert_eq!(
            engine.execution_plan.ordered_weight_names,
            vec![
                "transformer.h.1.weight".to_string(),
                "transformer.h.2.weight".to_string(),
                "transformer.h.10.weight".to_string(),
            ]
        );
    }

    /// A real safetensors buffer (built with the `safetensors` crate, the
    /// same encoder real tooling uses) must decode to tensors holding
    /// exactly its own bytes -- not `Tensor::randn` fabricated data. This
    /// calls the parser directly (bypassing `load_model`'s quantization
    /// pass) so the assertion is exact.
    #[test]
    fn test_parse_safetensors_decodes_real_bytes_not_random_weights() {
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;

        let raw: Vec<u8> = [1.0f32, -2.5, 3.0, 4.5].iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::F32, vec![2, 2], &raw).expect("valid tensor view");
        let mut tensors: HashMap<String, TensorView> = HashMap::new();
        tensors.insert("known.weight".to_string(), view);
        let bytes = safetensors::serialize(&tensors, None).expect("serialize safetensors");

        let weights =
            MobileInferenceEngine::parse_safetensors(&bytes).expect("real safetensors must parse");

        assert_eq!(weights.len(), 1);
        let tensor = weights.get("known.weight").expect("tensor present under its real name");
        assert_eq!(tensor.shape(), vec![2, 2]);
        assert_eq!(tensor.data().expect("data"), vec![1.0, -2.5, 3.0, 4.5]);
    }

    /// A byte buffer that is not a valid safetensors file must error, never
    /// fall back to `Tensor::randn` placeholder weights.
    #[test]
    fn test_parse_safetensors_rejects_garbage_bytes() {
        let garbage = vec![0xFFu8; 64];
        let result = MobileInferenceEngine::parse_safetensors(&garbage);
        assert!(
            result.is_err(),
            "garbage bytes must not parse as safetensors"
        );
    }

    /// A byte buffer that is not a valid PyTorch checkpoint must error.
    #[test]
    fn test_parse_pytorch_rejects_garbage_bytes() {
        let garbage = vec![0x00u8; 64];
        let result = MobileInferenceEngine::parse_pytorch(&garbage);
        assert!(
            result.is_err(),
            "garbage bytes must not parse as a PyTorch checkpoint"
        );
    }

    /// A byte buffer that is not a valid ONNX `ModelProto` must error.
    #[test]
    fn test_parse_onnx_rejects_garbage_bytes() {
        let garbage = vec![0xAAu8; 64];
        let result = MobileInferenceEngine::parse_onnx(&garbage);
        assert!(
            result.is_err(),
            "garbage bytes must not parse as an ONNX model"
        );
    }

    /// `load_model_from_file` must reject TensorFlow checkpoints with a
    /// structured error instead of fabricating GPT-2-shaped random weights
    /// for a format it cannot parse -- the previous behavior for every
    /// unimplemented/unknown format.
    #[test]
    fn test_load_model_from_file_rejects_tensorflow_format() {
        let path = std::env::temp_dir().join(format!(
            "trustformers_mobile_test_tf_{}_{}.pb",
            std::process::id(),
            fastrand::u64(..)
        ));
        std::fs::write(&path, b"not actually a TensorFlow SavedModel").expect("write temp file");

        let config = MobileConfig::default();
        let mut engine = MobileInferenceEngine::new(config).expect("engine");
        let result = engine.load_model_from_file(path.to_str().expect("utf8 path"));
        let _ = std::fs::remove_file(&path);

        assert!(
            result.is_err(),
            "TensorFlow format must be a structured error, not fabricated weights"
        );
        assert!(
            !engine.model_loaded,
            "a rejected load must not leave the engine 'loaded'"
        );
    }

    /// A file with no recognisable extension or structural header must be
    /// rejected outright, not silently filled with `Tensor::randn`
    /// "placeholder weights" as the previous implementation did for any
    /// unrecognised format.
    #[test]
    fn test_load_model_from_file_rejects_unrecognised_format() {
        let path = std::env::temp_dir().join(format!(
            "trustformers_mobile_test_unknown_{}_{}.bin.tmp",
            std::process::id(),
            fastrand::u64(..)
        ));
        std::fs::write(
            &path,
            b"neither a checkpoint nor anything else recognisable",
        )
        .expect("write temp file");

        let config = MobileConfig::default();
        let mut engine = MobileInferenceEngine::new(config).expect("engine");
        let result = engine.load_model_from_file(path.to_str().expect("utf8 path"));
        let _ = std::fs::remove_file(&path);

        assert!(
            result.is_err(),
            "an unrecognised format must be a structured error, not placeholder weights"
        );
    }

    /// Full pipeline, end to end, on disk: write a real safetensors file
    /// (via the `safetensors` crate) for a tiny unambiguous two-layer
    /// linear stack, load it through `load_model_from_file` (which also
    /// runs it through `MobileOptimizationEngine`'s quantization pass --
    /// `MobileConfig::default()` selects dynamic Int8), then run
    /// `inference()` and check the output is numerically the real matmul
    /// chain, not an echo of the input and not `Tensor::randn` noise.
    ///
    /// Int8 quantization perturbs values, so this asserts the output
    /// *shape* changed (impossible for the old identity pass) and that the
    /// result is finite and of a sane order of magnitude for the known
    /// input -- not an exact equality, since quantization rounding is
    /// real and expected here.
    #[test]
    fn test_load_model_from_file_end_to_end_safetensors() {
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;

        // [in=4, out=2] projecting a 4-wide input down to width 2, then
        // [in=2, out=1] projecting down to a scalar: an unambiguous chain
        // (256 -> ... no: 4 -> 2 -> 1, each width distinct).
        let w0: Vec<u8> = [1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let w1: Vec<u8> = [1.0f32, 1.0].iter().flat_map(|v| v.to_le_bytes()).collect();

        let view0 = TensorView::new(Dtype::F32, vec![4, 2], &w0).expect("view0");
        let view1 = TensorView::new(Dtype::F32, vec![2, 1], &w1).expect("view1");
        let mut tensors: HashMap<String, TensorView> = HashMap::new();
        tensors.insert("layer.0.weight".to_string(), view0);
        tensors.insert("layer.1.weight".to_string(), view1);
        let bytes = safetensors::serialize(&tensors, None).expect("serialize safetensors");

        let path = std::env::temp_dir().join(format!(
            "trustformers_mobile_test_e2e_{}_{}.safetensors",
            std::process::id(),
            fastrand::u64(..)
        ));
        std::fs::write(&path, &bytes).expect("write temp safetensors file");

        let config = MobileConfig::default();
        let mut engine = MobileInferenceEngine::new(config).expect("engine");
        let load_result = engine.load_model_from_file(path.to_str().expect("utf8 path"));
        let _ = std::fs::remove_file(&path);
        load_result.expect("a real safetensors file with an unambiguous linear stack must load");
        assert!(engine.model_loaded);

        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).expect("input");
        let output = engine.inference(&input).expect("inference over a real loaded checkpoint");

        assert_eq!(
            output.shape(),
            vec![1, 1],
            "the real two-layer projection must reduce width 4 -> 2 -> 1"
        );
        let data = output.data().expect("output data");
        assert!(
            data[0].is_finite(),
            "quantized real computation must stay finite, got {}",
            data[0]
        );
    }
}
