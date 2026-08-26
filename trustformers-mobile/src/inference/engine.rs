//! The `MobileInferenceEngine`: model loading (SafeTensors/ONNX/TFLite/CoreML), execution planning and inference.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::{optimization::MobileOptimizationEngine, MobileConfig, MobileStats};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;
use trustformers_core::errors::{
    invalid_format, invalid_input, runtime_error, unsupported_operation, Result,
};
use trustformers_core::Tensor;

use super::cache::InferenceCache;
use super::formats::{ExecutionPlan, ExecutionStrategy, ModelFormat};
use super::memory_info::MobileMemoryInfo;
use super::tensor_conversion::{onnx_tensor_to_tensor, safetensors_view_to_tensor};

/// Unified mobile inference engine
#[derive(Debug)]
pub struct MobileInferenceEngine {
    pub(super) config: MobileConfig,
    pub(super) optimizer: MobileOptimizationEngine,
    pub(super) execution_plan: ExecutionPlan,
    pub(super) stats: MobileStats,
    pub(super) model_loaded: bool,
    pub(super) model_weights: Option<HashMap<String, Tensor>>,
    pub(super) cache: Option<InferenceCache>,
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
    pub(super) fn parse_model_format(
        &self,
        data: &[u8],
        path: &Path,
    ) -> Result<HashMap<String, Tensor>> {
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
    pub(super) fn detect_model_format(&self, data: &[u8], path: &Path) -> Result<ModelFormat> {
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
    /// `pub(crate)`: reused directly by `wasm::WasmMobileEngine::parse_model_weights`
    /// so the WASM bridge parses real safetensors bytes through this exact,
    /// already-tested decoder rather than a second, divergent copy of it.
    pub(crate) fn parse_safetensors(data: &[u8]) -> Result<HashMap<String, Tensor>> {
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
    pub(super) fn parse_pytorch(data: &[u8]) -> Result<HashMap<String, Tensor>> {
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
    pub(super) fn parse_onnx(data: &[u8]) -> Result<HashMap<String, Tensor>> {
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

    /// Whether a model is currently loaded (weights present and
    /// [`Self::inference`] would not immediately error with "Model not
    /// loaded"). Exposes the private `model_loaded` flag other modules in
    /// this crate (e.g. `react_native`'s bridge, via its own differently-
    /// shaped `is_model_loaded(&self, model_id: &str)`) need to answer that
    /// question honestly instead of hardcoding `true`. Named
    /// `has_loaded_model` rather than `is_model_loaded` specifically to
    /// avoid colliding with that bridge method's inherent-impl name (same
    /// type, same crate, different arity -- Rust does not allow overloading
    /// by arity for inherent methods).
    pub fn has_loaded_model(&self) -> bool {
        self.model_loaded
    }

    /// Unload the currently loaded model, freeing its weights and any
    /// cached inference results.
    ///
    /// This engine holds one active model at a time (see `model_weights:
    /// Option<HashMap<String, Tensor>>` above); callers that need
    /// multi-model bookkeeping (tracking several model IDs and which one is
    /// currently active) layer that on top, e.g. `react_native`'s
    /// `ModelManager`. After this call, [`Self::inference`] returns the
    /// same "Model not loaded" error it would for a freshly constructed
    /// engine, and [`Self::load_model`]/[`Self::load_model_from_file`] must
    /// be called again before running inference. Named `clear_loaded_model`
    /// rather than `unload_model` for the same arity-collision reason as
    /// [`Self::has_loaded_model`] above.
    pub fn clear_loaded_model(&mut self) {
        self.model_weights = None;
        self.model_loaded = false;
        self.execution_plan.ordered_weight_names.clear();
        self.cache = None;
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
    pub(super) fn infer_warm_up_hidden_size(&self) -> Result<usize> {
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

    pub(super) fn sequential_inference(&self, input: &Tensor) -> Result<Tensor> {
        self.run_ordered_layers(input)
    }

    pub(super) fn layer_parallel_inference(&self, input: &Tensor) -> Result<Tensor> {
        self.run_ordered_layers(input)
    }

    pub(super) fn full_parallel_inference(&self, input: &Tensor) -> Result<Tensor> {
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
    pub(super) fn run_ordered_layers(&self, input: &Tensor) -> Result<Tensor> {
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
    pub(super) fn find_bias_pair(
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
    pub(super) fn bias_name_for_weight(weight_name: &str) -> Option<String> {
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
    pub(super) fn projection_output_dim(
        input_last_dim: usize,
        weight_shape: &[usize],
    ) -> Option<usize> {
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
    pub(super) fn layer_is_shape_compatible(input_shape: &[usize], weight_shape: &[usize]) -> bool {
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
    pub(super) fn process_layer(
        &self,
        input: &Tensor,
        name: &str,
        weight: &Tensor,
    ) -> Result<Option<Tensor>> {
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

    pub(super) fn estimate_current_memory_usage(&self) -> usize {
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

    pub(super) fn should_use_cache(&self) -> bool {
        // Enable cache only if we have sufficient memory
        self.config.max_memory_mb >= 512
            && self.config.memory_optimization != crate::MemoryOptimization::Maximum
    }
}
