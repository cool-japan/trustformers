//! C API for TrustformeRS - High-performance transformer library
//!
//! This crate provides C-compatible bindings for the TrustformeRS library,
//! enabling integration with other programming languages through FFI.

#![allow(clippy::not_unsafe_ptr_arg_deref)]
#![allow(clippy::upper_case_acronyms)] // Model names like BERT, GPT2, etc. are standard
#![allow(clippy::type_complexity)] // Complex types are necessary for C API compatibility
#![allow(clippy::too_many_arguments)] // C APIs often need many parameters
#![allow(clippy::manual_clamp)] // Manual clamp is clearer in some contexts
#![allow(clippy::wrong_self_convention)] // C API naming conventions differ
#![allow(clippy::cognitive_complexity)] // Complex C API functions are expected
#![allow(clippy::blocks_in_conditions)] // Generated code may have this pattern
#![allow(clippy::collapsible_if)] // Clarity over brevity for C API
#![allow(clippy::too_deeply_nested)] // Nested blocks in complex C API logic
#![allow(clippy::vec_init_then_push)] // Clear initialization patterns
#![allow(clippy::unnecessary_lazy_evaluations)] // Functional style preferred
#![allow(clippy::useless_format)] // Format used for consistency
#![allow(clippy::single_char_add_str)] // String operations for clarity
#![allow(clippy::manual_strip)] // Explicit prefix stripping
#![allow(clippy::or_insert_with_default)] // Closure style preferred
#![allow(clippy::derivable_impls)] // Custom implementations may be needed
#![allow(clippy::redundant_closure)] // Closures for clarity
#![allow(clippy::manual_slice_size_calculation)] // Explicit calculations
#![allow(clippy::crate_in_macro_def)] // Macro patterns
#![allow(clippy::match_like_matches_macro)] // Match for clarity
#![allow(clippy::manual_range_contains)] // Explicit range checks
#![allow(clippy::to_string_trait_impl)] // ToString implementations
#![allow(clippy::unnecessary_cast)] // Explicit type conversions
#![allow(clippy::needless_lifetimes)] // Explicit lifetimes for FFI
#![allow(clippy::collapsible_else_if)] // Clarity in control flow
#![allow(clippy::explicit_counter_loop)] // Counter loops for clarity
#![allow(clippy::manual_div_ceil)] // Explicit division
#![allow(clippy::doc_lazy_continuation)] // Doc comment style
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_macros)]
#![allow(non_camel_case_types)]
#![allow(unused_variables)]
#![allow(unused_assignments)]
#![allow(unused_mut)]
#![allow(private_interfaces)]
#![allow(deprecated)]
#![allow(ambiguous_glob_reexports)]
#![allow(static_mut_refs)]
#![allow(unused_unsafe)]

use anyhow::{anyhow, Result};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::sync::Arc;

// Core TrustformeRS imports
use trustformers::pipeline::Pipeline;
use trustformers::{AutoModel, AutoTokenizer};
use trustformers_core::tensor::Tensor;

mod abi_stability;
mod asic;
mod async_api;
mod cloud;
mod containers;
mod debug_utilities;
mod distributed;
mod error;
mod fuzzing;
mod intel_ai;
mod memory_pool;
mod memory_safety;
mod memory_safety_enhanced;
mod model;
mod model_optimization;
mod modern_architectures;
mod multimodal;
mod performance;
mod performance_analytics;
mod pipeline;
mod platform;
mod plugin;
mod profiling_tools;
mod quantization;
mod streaming;
mod tensor;
mod thread_pool;
mod tokenizer;
mod tpu;
mod utils;
mod zero_copy;

#[cfg(feature = "codegen")]
pub mod codegen;

#[cfg(all(feature = "cuda", any(target_os = "linux", target_os = "windows")))]
mod cuda;

#[cfg(target_os = "macos")]
mod metal;

#[cfg(feature = "rocm")]
mod rocm;

#[cfg(feature = "serving")]
mod http_server;

#[cfg(feature = "serving")]
mod batch_processor;

#[cfg(feature = "serving")]
mod model_cache;

#[cfg(feature = "serving")]
mod grpc_server;

// Tests are now in a separate module for better organization
#[cfg(test)]
mod tests;

pub use async_api::{
    trustformers_async_cancel, trustformers_async_cancel_all, trustformers_async_free,
    trustformers_async_get_active_count, trustformers_async_get_error,
    trustformers_async_get_status, trustformers_async_load_model, trustformers_async_wait,
    TrustformersAsyncCallback, TrustformersAsyncHandle, TrustformersAsyncProgressCallback,
    TrustformersAsyncStatus,
};
pub use cloud::*;
pub use containers::*;
pub use error::*;
pub use memory_safety::*;
pub use memory_safety_enhanced::{
    trustformers_enhanced_memory_verify, trustformers_predict_memory_leaks,
};
pub use model::*;
pub use model_optimization::{
    trustformers_nas_config_create, trustformers_nas_config_free, trustformers_nas_manager_create,
    trustformers_nas_manager_free, trustformers_nas_result_free, trustformers_nas_run_search,
    trustformers_nas_set_algorithm, trustformers_nas_set_hardware_constraints,
    trustformers_nas_set_objectives, trustformers_nas_set_search_params,
    trustformers_nas_start_search, NASManager, TrustformersNASConfig, TrustformersNASResult,
};
pub use modern_architectures::{
    trustformers_compare_architectures, trustformers_configure_attention,
    trustformers_configure_mamba, trustformers_configure_moe, trustformers_create_mamba_model,
    trustformers_mamba_inference, trustformers_modern_arch_config_create,
    trustformers_modern_arch_config_free, trustformers_modern_arch_free,
    trustformers_modern_arch_get_metrics, GeneralModelConfig, MambaConfig, MoEConfig,
    ModernArchitectureType, ModernAttentionConfig, TrustformersModernArchitecture,
};
pub use multimodal::{
    trustformers_compute_similarity, trustformers_create_multimodal_model,
    trustformers_multimodal_config_create, trustformers_multimodal_config_free,
    trustformers_multimodal_free, trustformers_multimodal_inference,
    trustformers_multimodal_output_free, trustformers_vision_language_task, AudioEncoderConfig,
    MultiModalConfig, MultiModalOutput, TrustformersMultiModalModel, VisionEncoderConfig,
    VisionLanguageTask,
};
pub use performance::{
    trustformers_create_performance_optimizer, trustformers_get_performance_stats, PerformanceStats,
};
pub use performance_analytics::{
    trustformers_advanced_analytics_create, trustformers_analytics_apply_optimizations,
    trustformers_analytics_get_insights, trustformers_analytics_start_monitoring,
    AdvancedPerformanceAnalytics,
};
pub use pipeline::{
    trustformers_pipeline_create, trustformers_pipeline_destroy, trustformers_pipeline_infer,
    TrustformersPipeline, TrustformersPipelineConfig,
};
// Platform module exports
// pub use platform::{...};

pub use memory_pool::{
    trustformers_memory_pool_alloc, trustformers_memory_pool_create,
    trustformers_memory_pool_destroy, trustformers_memory_pool_free, trustformers_memory_pool_gc,
    trustformers_memory_pool_get_stats, trustformers_memory_pool_reset_stats,
    TrustformersMemoryPoolConfig, TrustformersMemoryPoolHandle, TrustformersMemoryPoolStats,
};
pub use plugin::{
    trustformers_plugin_add_search_path, trustformers_plugin_execute_operation,
    trustformers_plugin_find_by_name, trustformers_plugin_get_capabilities,
    trustformers_plugin_get_metadata, trustformers_plugin_initialize, trustformers_plugin_list,
    trustformers_plugin_register, trustformers_plugin_register_operation,
    trustformers_plugin_unregister, PluginCapabilities, PluginOperationFn, PluginType,
    TrustformersPluginHandle,
};
pub use quantization::{
    trustformers_advanced_quantization_create, trustformers_advanced_quantization_report,
    trustformers_nas_quantization_optimize, trustformers_quantization_create_engine,
    trustformers_quantization_quantize_model, AdvancedQuantizationConfig,
    AdvancedQuantizationEngine,
};
pub use streaming::{
    trustformers_streaming_create, trustformers_streaming_free, trustformers_streaming_get_text,
    trustformers_streaming_get_token_count, trustformers_streaming_is_active,
    trustformers_streaming_start, trustformers_streaming_stop, trustformers_streaming_wait,
    TrustformersChunkCallback, TrustformersProgressCallback, TrustformersStreamHandle,
    TrustformersStreamingConfig, TrustformersTokenCallback,
};
pub use tensor::{
    trustformers_tensor_add, trustformers_tensor_clone, trustformers_tensor_copy_data,
    trustformers_tensor_create_from_data, trustformers_tensor_div, trustformers_tensor_free,
    trustformers_tensor_gelu, trustformers_tensor_get_data_ptr, trustformers_tensor_matmul,
    trustformers_tensor_mul, trustformers_tensor_numel, trustformers_tensor_ones,
    trustformers_tensor_permute, trustformers_tensor_print_info, trustformers_tensor_rand,
    trustformers_tensor_randn, trustformers_tensor_reduce, trustformers_tensor_relu,
    trustformers_tensor_reshape, trustformers_tensor_shape, trustformers_tensor_softmax,
    trustformers_tensor_sub, trustformers_tensor_transpose, trustformers_tensor_zeros,
    TrustformersDType, TrustformersInterpolationMode, TrustformersReduceOp, TrustformersTensor,
};
pub use tokenizer::{
    trustformers_tokenizer_decode, trustformers_tokenizer_destroy, trustformers_tokenizer_encode,
    trustformers_tokenizer_from_pretrained, TrustformersTokenizer, TrustformersTokenizerConfig,
};
pub use zero_copy::{
    trustformers_shared_memory_create, trustformers_shared_memory_free,
    trustformers_shared_memory_get_ptr, trustformers_shared_memory_open, trustformers_tensor_mmap,
    trustformers_tensor_zero_copy_view, trustformers_zero_copy_get_stats,
    TrustformersMemoryMapMode, TrustformersSharedMemHandle, TrustformersZeroCopyConfig,
    TrustformersZeroCopyStats,
};

// Re-export utility functions for use throughout the C API
pub use utils::{c_str_to_string, result_to_error, string_to_c_str};

// Utility function to convert core Result<T> to (TrustformersError, Option<T>)
pub fn core_result_to_error<T>(result: Result<T>) -> (TrustformersError, Option<T>) {
    match result {
        Ok(value) => (TrustformersError::Success, Some(value)),
        Err(e) => {
            // Map anyhow errors to C API error codes based on message content
            let error_str = e.to_string();
            let error_code = if error_str.contains("model") {
                TrustformersError::ModelLoadError
            } else if error_str.contains("tokenizer") {
                TrustformersError::TokenizerError
            } else if error_str.contains("pipeline") {
                TrustformersError::PipelineError
            } else if error_str.contains("tensor") {
                TrustformersError::TensorError
            } else if error_str.contains("file") || error_str.contains("not found") {
                TrustformersError::FileNotFound
            } else if error_str.contains("memory") {
                TrustformersError::OutOfMemory
            } else if error_str.contains("config") {
                TrustformersError::ConfigError
            } else {
                TrustformersError::RuntimeError
            };
            (error_code, None)
        },
    }
}

// Helper function to convert trustformers error Result<T> to (TrustformersError, Option<T>)
pub fn trustformers_result_to_error<T>(
    result: std::result::Result<T, trustformers::error::TrustformersError>,
) -> (TrustformersError, Option<T>) {
    match result {
        Ok(value) => (TrustformersError::Success, Some(value)),
        Err(e) => {
            // Convert trustformers::error::TrustformersError to C API error code
            // The trustformers error uses structured enum variants
            let error_str = e.to_string();
            let error_code = if error_str.contains("Model") {
                TrustformersError::ModelLoadError
            } else if error_str.contains("Tokenizer") || error_str.contains("tokenization") {
                TrustformersError::TokenizerError
            } else if error_str.contains("Pipeline") {
                TrustformersError::PipelineError
            } else if error_str.contains("Hub") || error_str.contains("Hub Error") {
                TrustformersError::FileNotFound
            } else if error_str.contains("Auto Configuration") || error_str.contains("Config") {
                TrustformersError::ConfigError
            } else {
                TrustformersError::RuntimeError
            };
            (error_code, None)
        },
    }
}

// Helper function to convert core error Result<T> to (TrustformersError, Option<T>)
pub fn core_tensor_result_to_error<T>(
    result: std::result::Result<T, trustformers_core::errors::TrustformersError>,
) -> (TrustformersError, Option<T>) {
    match result {
        Ok(value) => (TrustformersError::Success, Some(value)),
        Err(e) => {
            // trustformers_core::errors::TrustformersError is a complex struct
            // Map based on the error message
            let error_str = e.to_string();
            let error_code = if error_str.contains("null") || error_str.contains("pointer") {
                TrustformersError::NullPointer
            } else if error_str.contains("memory") || error_str.contains("allocation") {
                TrustformersError::OutOfMemory
            } else if error_str.contains("tensor") {
                TrustformersError::TensorError
            } else if error_str.contains("config") {
                TrustformersError::ConfigError
            } else if error_str.contains("serialization") || error_str.contains("serialize") {
                TrustformersError::SerializationError
            } else if error_str.contains("runtime") {
                TrustformersError::RuntimeError
            } else if error_str.contains("resource") {
                TrustformersError::ResourceLimitExceeded
            } else {
                TrustformersError::TensorError
            };
            (error_code, None)
        },
    }
}

// Global configuration and state management
static GLOBAL_CONFIG: Lazy<RwLock<Option<TrustformersConfig>>> = Lazy::new(|| RwLock::new(None));
static RESOURCE_REGISTRY: Lazy<RwLock<ResourceRegistry>> =
    Lazy::new(|| RwLock::new(ResourceRegistry::new()));

/// Main configuration structure for TrustformeRS
#[derive(Debug, Clone)]
#[repr(C)]
pub struct TrustformersConfig {
    /// Maximum number of models that can be loaded simultaneously
    pub max_models: c_int,
    /// Memory limit in megabytes
    pub memory_limit_mb: c_int,
    /// Thread pool size for inference
    pub thread_pool_size: c_int,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Log level (0=Error, 1=Warn, 2=Info, 3=Debug)
    pub log_level: c_int,
    /// Path to configuration file
    pub config_file_path: *const c_char,
    /// Enable GPU acceleration if available
    pub enable_gpu: bool,
    /// GPU device index to use
    pub gpu_device_id: c_int,
    /// Enable mixed precision
    pub enable_mixed_precision: bool,
    /// Cache directory for models
    pub cache_dir: *const c_char,
}

// SAFETY: TrustformersConfig is used in a global static with RwLock synchronization.
// The raw pointers are set once during initialization and never mutated.
// Access is synchronized through RwLock, ensuring thread-safe reads and writes.
unsafe impl Send for TrustformersConfig {}
unsafe impl Sync for TrustformersConfig {}

impl Default for TrustformersConfig {
    fn default() -> Self {
        Self {
            max_models: 10,
            memory_limit_mb: 4096,
            thread_pool_size: 4,
            enable_monitoring: false,
            log_level: 2, // Info
            config_file_path: ptr::null(),
            enable_gpu: true,
            gpu_device_id: 0,
            enable_mixed_precision: true,
            cache_dir: ptr::null(),
        }
    }
}

/// Device types for model execution
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub enum DeviceType {
    CPU = 0,
    GPU = 1,
    TPU = 2,
    ASIC = 3,
}

/// Precision types for model execution
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub enum Precision {
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
    Int8 = 3,
    Int4 = 4,
}

/// Model types supported by TrustformeRS
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub enum ModelType {
    BERT = 0,
    GPT2 = 1,
    T5 = 2,
    RoBERTa = 3,
    ELECTRA = 4,
    DistilBERT = 5,
    ALBERT = 6,
    DeBERTa = 7,
    Llama = 8,
    Mistral = 9,
    Phi3 = 10,
    Gemma = 11,
    Qwen = 12,
    Mamba = 13,
    Custom = 999,
}

/// Task types for pipeline execution
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub enum TaskType {
    TextClassification = 0,
    TokenClassification = 1,
    QuestionAnswering = 2,
    TextGeneration = 3,
    Summarization = 4,
    Translation = 5,
    FillMask = 6,
    FeatureExtraction = 7,
    SentenceSimilarity = 8,
    ZeroShotClassification = 9,
}

/// Model configuration structure
#[derive(Debug, Clone)]
#[repr(C)]
pub struct ModelConfig {
    pub model_path: *const c_char,
    pub model_type: ModelType,
    pub precision: Precision,
    pub device: DeviceType,
    pub batch_size: c_int,
    pub max_sequence_length: c_int,
    pub custom_config: *const c_char,
}

/// Resource registry for managing loaded models and tokenizers
#[derive(Default)]
pub struct ResourceRegistry {
    models: HashMap<usize, Arc<AutoModel>>,
    tokenizers: HashMap<usize, Arc<AutoTokenizer>>,
    pipelines: HashMap<
        usize,
        Arc<dyn Pipeline<Input = String, Output = trustformers::pipeline::PipelineOutput>>,
    >,
    tensors: HashMap<usize, Arc<Tensor>>,
    next_handle: usize,
}

impl ResourceRegistry {
    fn new() -> Self {
        Self {
            models: HashMap::new(),
            tokenizers: HashMap::new(),
            pipelines: HashMap::new(),
            tensors: HashMap::new(),
            next_handle: 1,
        }
    }

    fn register_model(&mut self, model: AutoModel) -> usize {
        let handle = self.next_handle;
        self.next_handle += 1;
        self.models.insert(handle, Arc::new(model));
        handle
    }

    fn get_model(&self, handle: usize) -> Option<&Arc<AutoModel>> {
        self.models.get(&handle)
    }

    fn remove_model(&mut self, handle: usize) -> bool {
        self.models.remove(&handle).is_some()
    }

    fn register_tokenizer(&mut self, tokenizer: AutoTokenizer) -> usize {
        let handle = self.next_handle;
        self.next_handle += 1;
        self.tokenizers.insert(handle, Arc::new(tokenizer));
        handle
    }

    fn get_tokenizer(&self, handle: usize) -> Option<&Arc<AutoTokenizer>> {
        self.tokenizers.get(&handle)
    }

    fn remove_tokenizer(&mut self, handle: usize) -> bool {
        self.tokenizers.remove(&handle).is_some()
    }

    fn register_pipeline(
        &mut self,
        pipeline: Box<
            dyn Pipeline<Input = String, Output = trustformers::pipeline::PipelineOutput>,
        >,
    ) -> usize {
        let handle = self.next_handle;
        self.next_handle += 1;
        self.pipelines.insert(handle, pipeline.into());
        handle
    }

    fn get_pipeline(
        &self,
        handle: usize,
    ) -> Option<&Arc<dyn Pipeline<Input = String, Output = trustformers::pipeline::PipelineOutput>>>
    {
        self.pipelines.get(&handle)
    }

    fn remove_pipeline(&mut self, handle: usize) -> bool {
        self.pipelines.remove(&handle).is_some()
    }

    fn register_tensor(&mut self, tensor: Tensor) -> usize {
        let handle = self.next_handle;
        self.next_handle += 1;
        self.tensors.insert(handle, Arc::new(tensor));
        handle
    }

    fn get_tensor(&self, handle: usize) -> Option<&Arc<Tensor>> {
        self.tensors.get(&handle)
    }

    fn unregister_tensor(&mut self, handle: usize) -> bool {
        self.tensors.remove(&handle).is_some()
    }

    fn unregister_model(&mut self, handle: usize) -> bool {
        self.remove_model(handle)
    }

    fn clear(&mut self) {
        self.models.clear();
        self.tokenizers.clear();
        self.pipelines.clear();
        self.tensors.clear();
        self.next_handle = 1;
    }
}

/// Performance tracking for optimization hints
#[derive(Debug, Default)]
pub struct PerformanceTracker {
    operation_times: HashMap<String, Vec<f64>>,
    memory_allocations: Vec<u64>,
    cache_hits: HashMap<String, usize>,
    cache_misses: HashMap<String, usize>,
}

impl PerformanceTracker {
    fn new() -> Self {
        Self::default()
    }

    fn record_operation(&mut self, operation: &str, duration_ms: f64) {
        self.operation_times.entry(operation.to_string()).or_default().push(duration_ms);
    }

    fn record_memory_allocation(&mut self, bytes: u64) {
        self.memory_allocations.push(bytes);
    }

    fn record_cache_hit(&mut self, cache_name: &str) {
        *self.cache_hits.entry(cache_name.to_string()).or_insert(0) += 1;
    }

    fn record_cache_miss(&mut self, cache_name: &str) {
        *self.cache_misses.entry(cache_name.to_string()).or_insert(0) += 1;
    }

    fn get_stats(&self) -> PerformanceStats {
        let total_hits: u64 = self.cache_hits.values().sum::<usize>() as u64;
        let total_ops: u64 = self.operation_times.values().map(|v| v.len() as u64).sum();

        PerformanceStats {
            simd_enabled: false,             // Would need to query from system
            dynamic_batching_enabled: false, // Would need to query from pipeline config
            kernel_fusion_enabled: false,    // Would need to query from optimization settings
            average_batch_size: 1.0,         // Simplified - not tracked here
            throughput_per_second: if self.calculate_average_operation_time() > 0.0 {
                1000.0 / self.calculate_average_operation_time() // ops per second
            } else {
                0.0
            },
            fusion_cache_hits: total_hits,
            fusion_cache_size: self.cache_hits.len() + self.cache_misses.len(),
        }
    }

    fn calculate_average_operation_time(&self) -> f64 {
        let total_time: f64 = self.operation_times.values().flat_map(|times| times.iter()).sum();
        let total_ops: usize = self.operation_times.values().map(|times| times.len()).sum();

        if total_ops > 0 {
            total_time / total_ops as f64
        } else {
            0.0
        }
    }

    fn calculate_cache_hit_rate(&self) -> f64 {
        let total_hits: usize = self.cache_hits.values().sum();
        let total_misses: usize = self.cache_misses.values().sum();
        let total = total_hits + total_misses;

        if total > 0 {
            total_hits as f64 / total as f64
        } else {
            1.0
        }
    }

    fn get_performance_summary(&self) -> PerformanceSummary {
        let mut hints = Vec::new();

        // Check for slow operations
        for (operation, times) in &self.operation_times {
            let avg_time: f64 = times.iter().sum::<f64>() / times.len() as f64;
            if avg_time > 1000.0 && times.len() >= 10 {
                // Over 1 second average with at least 10 samples
                hints.push(OptimizationHint {
                    hint_type: OptimizationType::ModelOptimization,
                    message: format!("Operation '{}' is averaging {:.1}ms", operation, avg_time),
                    priority: if avg_time > 5000.0 { 3 } else { 2 },
                });
            }
        }

        // Check cache hit rates
        for cache_name in self.cache_hits.keys() {
            let hits = *self.cache_hits.get(cache_name).unwrap_or(&0);
            let misses = *self.cache_misses.get(cache_name).unwrap_or(&0);
            let total = hits + misses;

            if total >= 100 {
                let hit_rate = hits as f64 / total as f64;
                if hit_rate < 0.5 {
                    hints.push(OptimizationHint {
                        hint_type: OptimizationType::CachingStrategy,
                        message: format!(
                            "Cache '{}' has low hit rate: {:.1}%",
                            cache_name,
                            hit_rate * 100.0
                        ),
                        priority: if hit_rate < 0.2 { 3 } else { 2 },
                    });
                }
            }
        }

        PerformanceSummary {
            optimization_hints: hints,
            stats: self.get_stats(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OptimizationHint {
    pub hint_type: OptimizationType,
    pub message: String,
    pub priority: u8, // 1=Low, 2=Medium, 3=High
}

#[derive(Debug, Clone)]
pub enum OptimizationType {
    ModelOptimization,
    CachingStrategy,
    MemoryManagement,
    BatchingOptimization,
}

#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    pub optimization_hints: Vec<OptimizationHint>,
    pub stats: PerformanceStats,
}

// C API Functions

/// Initialize TrustformeRS with the given configuration
#[no_mangle]
pub extern "C" fn trustformers_init(config: *const TrustformersConfig) -> *const c_char {
    if config.is_null() {
        return string_to_c_str("Configuration pointer cannot be null".to_string());
    }

    let config = unsafe { &*config };

    // Validate configuration
    if config.max_models <= 0 || config.memory_limit_mb <= 0 || config.thread_pool_size <= 0 {
        return string_to_c_str("Invalid configuration parameters".to_string());
    }

    // Store global configuration
    {
        let mut global_config = GLOBAL_CONFIG.write();
        *global_config = Some(config.clone());
    }

    // Initialize logging if needed
    if !config.config_file_path.is_null() {
        // In a real implementation, we would initialize logging here
    }

    ptr::null()
}

/// Clean up TrustformeRS resources
#[no_mangle]
pub extern "C" fn trustformers_cleanup() {
    // Clear global configuration
    {
        let mut global_config = GLOBAL_CONFIG.write();
        *global_config = None;
    }

    // Clear resource registry
    {
        let mut registry = RESOURCE_REGISTRY.write();
        registry.clear();
    }
}

// NOTE: trustformers_get_version is now defined in utils_impl::c_api
// This duplicate definition has been removed to avoid symbol conflicts
// /// Get TrustformeRS version string
// #[no_mangle]
// pub extern "C" fn trustformers_get_version() -> *const c_char {
//     string_to_c_str(env!("CARGO_PKG_VERSION").to_string())
// }

/// Get number of available compute devices
#[no_mangle]
pub extern "C" fn trustformers_get_device_count() -> c_int {
    // In a real implementation, this would detect actual devices
    // For now, return 1 (CPU always available)
    1
}

/// Get information about a specific device
#[no_mangle]
pub extern "C" fn trustformers_get_device_info(device_index: c_int) -> *const c_char {
    if device_index < 0 {
        return ptr::null();
    }

    match device_index {
        0 => string_to_c_str("CPU Device".to_string()),
        _ => ptr::null(),
    }
}

/// Enable or disable performance monitoring
#[no_mangle]
pub extern "C" fn trustformers_enable_performance_monitoring(enable: bool) -> c_int {
    // Update global configuration
    if let Some(config) = GLOBAL_CONFIG.write().as_mut() {
        config.enable_monitoring = enable;
        0 // Success
    } else {
        -1 // Error: not initialized
    }
}

// NOTE: trustformers_free_string is now defined in utils_impl::c_api
// This duplicate definition has been removed to avoid symbol conflicts
// /// Free a C string allocated by TrustformeRS
// #[no_mangle]
// pub extern "C" fn trustformers_free_string(str_ptr: *mut c_char) {
//     if !str_ptr.is_null() {
//         unsafe {
//             let _ = CString::from_raw(str_ptr);
//         }
//     }
// }

// Helper functions

impl TrustformersConfig {
    /// Convert C string pointers to Rust strings safely
    fn get_config_file_path(&self) -> Result<String> {
        if self.config_file_path.is_null() {
            return Ok(String::new());
        }

        unsafe { CStr::from_ptr(self.config_file_path) }
            .to_str()
            .map_err(|e| anyhow!("Invalid UTF-8 string: {}", e))
            .map(|s| s.to_string())
    }

    /// Convert C string pointers to Rust strings safely
    fn get_cache_dir(&self) -> Result<String> {
        if self.cache_dir.is_null() {
            return Ok(String::new());
        }

        unsafe { CStr::from_ptr(self.cache_dir) }
            .to_str()
            .map_err(|e| anyhow!("Invalid UTF-8 string: {}", e))
            .map(|s| s.to_string())
    }
}
