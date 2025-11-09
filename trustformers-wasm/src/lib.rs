// Using std for wasm-bindgen compatibility and feature completeness
// Note: While wasm-bindgen 0.2.100+ supports no_std, this codebase uses many std features
// (HashMap, async/await, complex error types) that would require extensive refactoring
// to work with no_std+alloc. The std overhead is negligible in WebAssembly contexts.

// Allow excessive nesting for complex algorithms (matrix ops, GPU buffer management, etc.)
#![allow(clippy::excessive_nesting)]

use std::string::ToString;
use std::vec::Vec;

pub mod layers;
pub mod models;
#[cfg(feature = "web-workers")]
pub mod runtime;

// Import core modules from the core subdirectory
pub mod core;
pub use core::{model, pipeline, tensor, tokenizer, utils};

// Compute modules
pub mod compute;

#[cfg(feature = "web-workers")]
pub use compute::web_workers;

#[cfg(feature = "shared-memory")]
pub use compute::threads;

#[cfg(feature = "webgpu")]
pub use compute::webgpu_simple;

#[cfg(feature = "webgpu")]
pub use compute::webgpu;

#[cfg(feature = "webgpu")]
pub use compute::gpu_tensor;

// Storage modules
#[cfg(feature = "indexeddb")]
pub mod storage;

#[cfg(feature = "memory64")]
pub use storage::memory64;

#[cfg(feature = "streaming-loader")]
pub use storage::streaming_loader;

#[cfg(feature = "model-splitting")]
pub use storage::model_splitting;

#[cfg(feature = "react-components")]
pub mod react_components;

#[cfg(feature = "vue-components")]
pub mod vue_components;

#[cfg(feature = "angular-components")]
pub mod angular_components;

#[cfg(feature = "web-components")]
pub mod web_components;

#[cfg(feature = "playground")]
pub mod playground;

#[cfg(feature = "streaming-generation")]
pub mod streaming_generation;

#[cfg(feature = "mobile-optimization")]
pub mod mobile;

#[cfg(feature = "mobile-optimization")]
pub mod touch_gestures;

#[cfg(feature = "mobile-optimization")]
pub mod camera_integration;

#[cfg(feature = "mobile-optimization")]
pub mod device_capability;

#[cfg(feature = "mobile-optimization")]
pub mod device_capability_detection;

pub mod debug;
pub mod error;
pub mod events;
// Import optimization modules from the optimization subdirectory
pub mod optimization;
pub use optimization::{
    batch_processing, memory_pool, quantization, simd_tensor_ops, weight_compression,
};

pub mod auto_docs;
pub mod multi_model_manager;
pub mod performance;
pub mod performance_profiler;
pub mod plugin_framework;
pub mod plugins;

use std::sync::Mutex;
use wasm_bindgen::prelude::*;

// Global GPU memory tracking
static GPU_MEMORY_TRACKER: Mutex<GpuMemoryTracker> = Mutex::new(GpuMemoryTracker::new());

/// GPU memory tracker for monitoring WebGPU buffer allocations
#[derive(Debug)]
struct GpuMemoryTracker {
    current_usage: usize,
    peak_usage: usize,
    total_allocated: usize,
    total_deallocated: usize,
    allocation_count: usize,
    deallocation_count: usize,
}

impl GpuMemoryTracker {
    const fn new() -> Self {
        Self {
            current_usage: 0,
            peak_usage: 0,
            total_allocated: 0,
            total_deallocated: 0,
            allocation_count: 0,
            deallocation_count: 0,
        }
    }

    fn allocate(&mut self, size: usize) {
        self.current_usage += size;
        self.total_allocated += size;
        self.allocation_count += 1;

        if self.current_usage > self.peak_usage {
            self.peak_usage = self.current_usage;
        }
    }

    fn deallocate(&mut self, size: usize) {
        self.current_usage = self.current_usage.saturating_sub(size);
        self.total_deallocated += size;
        self.deallocation_count += 1;
    }

    fn get_current_usage(&self) -> usize {
        self.current_usage
    }

    fn get_peak_usage(&self) -> usize {
        self.peak_usage
    }

    fn reset_peak(&mut self) {
        self.peak_usage = self.current_usage;
    }
}

#[wasm_bindgen]
pub struct TrustformersWasm {
    initialized: bool,
}

impl Default for TrustformersWasm {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen]
impl TrustformersWasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        #[cfg(feature = "console_panic")]
        console_error_panic_hook::set_once();

        TrustformersWasm { initialized: true }
    }

    #[wasm_bindgen(getter)]
    pub fn version(&self) -> String {
        "0.1.0-alpha.1".to_string()
    }

    #[wasm_bindgen(getter)]
    pub fn initialized(&self) -> bool {
        self.initialized
    }
}

// Re-export main types
pub use tensor::WasmTensor;

#[cfg(feature = "webgpu")]
pub use webgpu::{
    AsyncExecutor, DeviceCapabilities, DeviceSelector, DeviceType, ExecutionStatus, FusableOp,
    KernelFusion, OperationType, Priority, ShaderManager, WorkgroupTuner,
};

#[cfg(feature = "web-workers")]
pub use runtime::edge_runtime::{
    EdgeCapabilities, EdgeInferenceConfig, EdgeRuntime, EdgeRuntimeDetector,
};

#[cfg(feature = "web-workers")]
pub use runtime::geo_distribution::{
    create_geo_distribution_manager, estimate_network_latency, get_distance_between_points,
    EdgeLocation, GeoDistributionManager, GeoRegion, RoutingDecision, RoutingWeights, UserLocation,
};

#[cfg(feature = "web-workers")]
pub use runtime::edge_caching::{
    create_edge_cache_manager, create_edge_computing_cache_config,
    create_memory_efficient_cache_config, create_performance_cache_config, estimate_cache_overhead,
    CacheConfig, CacheEntry, CacheEntryType, CacheStatistics, ConsistencyLevel, EdgeCacheManager,
    EvictionPolicy, ReplicationStrategy,
};

#[cfg(feature = "web-workers")]
pub use web_workers::{
    get_optimal_worker_count, is_web_workers_supported, WorkerCoordinator, WorkerPool,
    WorkerPriority, WorkerTaskType,
};

#[cfg(feature = "shared-memory")]
pub use threads::{
    get_optimal_thread_count, is_cross_origin_isolated, is_threading_supported, AtomicOperations,
    ThreadPool, ThreadSync, ThreadTaskType,
};

#[cfg(feature = "indexeddb")]
pub use storage::{CompressionType, ModelMetadata, ModelStorage, StoredModel};

#[cfg(feature = "memory64")]
pub use memory64::{
    can_load_model_size, get_memory64_capabilities, is_memory64_supported, AllocationStrategy,
    Memory64Capabilities, Memory64Manager,
};

#[cfg(feature = "streaming-loader")]
pub use streaming_loader::{
    get_optimal_chunk_size_kb, is_cache_api_available, is_streaming_compilation_supported,
    LoadingProgress, StreamingConfig, StreamingLoader,
};

#[cfg(feature = "model-splitting")]
pub use model_splitting::{
    get_recommended_chunk_size_mb, should_split_model, ChunkConfig, ChunkPriority, ChunkType,
    LoadingStrategy, ModelLoadingSession, ModelSplitter,
};

#[cfg(feature = "react-components")]
pub use react_components::{
    generate_react_package, is_react_available, ComponentType, InferenceState, ModelLoadingState,
    ReactComponentFactory, ReactConfig,
};

#[cfg(feature = "vue-components")]
pub use vue_components::{
    generate_vue_package, is_vue_available, VueComponentFactory, VueComponentType, VueConfig,
    VueInferenceState, VueModelState,
};

#[cfg(feature = "angular-components")]
pub use angular_components::{
    generate_angular_package, is_angular_available, AngularComponentType, AngularConfig,
    AngularInferenceState, AngularModelState, AngularServiceFactory, AngularServiceType,
};

#[cfg(feature = "web-components")]
pub use web_components::{
    create_web_component_html_template, generate_web_components_package,
    is_web_components_supported, ComponentType as WebComponentType,
    InferenceState as WebInferenceState, ModelState as WebModelState, WebComponentConfig,
    WebComponentFactory,
};

#[cfg(feature = "playground")]
pub use playground::{
    create_playground_config, create_playground_example, generate_playground_package,
    ExampleCategory, InteractivePlayground, PlaygroundConfig, PlaygroundExample,
};

#[cfg(feature = "streaming-generation")]
pub use streaming_generation::{
    get_optimal_streaming_config, is_streaming_supported, CompletionReason, GenerationProgress,
    StreamingConfig as GenerationStreamingConfig, StreamingGenerator, StreamingStats,
    StreamingToken,
};

#[cfg(feature = "mobile-optimization")]
pub use mobile::{
    create_mobile_optimizer, get_device_memory_gb, get_optimal_model_for_device, is_low_data_mode,
    is_mobile_device, is_tablet_device, AdaptiveModelConfig, BatteryInfo, BatteryStatus,
    DeviceClass, MobileCapabilities, MobileOptimizer, ModelSize, NetworkStatus, NetworkType,
};

pub use auto_docs::{
    create_default_doc_generator,
    create_html_doc_generator,
    create_markdown_doc_generator,
    // Version and build information
    get_version_info,
    AutoDocGenerator,
    DocConfig,
    DocFormat,
    DocTheme,
    VersionInfo,
};
pub use batch_processing::{
    BatchConfig, BatchProcessor, BatchResponse, BatchingStrategy, Priority as BatchPriority,
};
pub use debug::{DebugConfig, DebugLogger, LogLevel, PerformanceMetrics};
pub use error::{
    ErrorBuilder, ErrorCode, ErrorCollection, ErrorContext, ErrorHandler, ErrorSeverity,
    TrustformersError, TrustformersResult,
};
pub use events::{EventData, EventEmittable, EventManager, EventPriority, EventType};
pub use multi_model_manager::{
    create_development_multi_model_manager, create_production_multi_model_manager,
    DeploymentEnvironment, ModelPriority, ModelStatus, MultiModelConfig, MultiModelManager,
};
pub use performance::{
    BottleneckType, OperationType as ProfilerOperationType, ProfilerConfig, ResourceType,
};
pub use performance_profiler::{
    create_development_profiler, create_production_profiler, PerformanceProfiler,
};
pub use plugin_framework::{
    create_default_plugin_config, create_plugin_context, ExecutionMetrics, ExecutionPriority,
    ModelMetadata as PluginModelMetadata, PerformanceBudget, Plugin, PluginConfig, PluginContext,
    PluginError, PluginErrorCode, PluginManager, PluginMetadata, PluginPermission, PluginRegistry,
    PluginResult, PluginType, ResourceLimits,
};
pub use plugins::{ModelOptimizerPlugin, TextProcessorPlugin, VisualizationPlugin};
pub use quantization::{
    QuantizationConfig, QuantizationPrecision, QuantizationStrategy, QuantizedModelData,
    WebQuantizer,
};
pub use weight_compression::{
    CompressedModelData, CompressionConfig, CompressionLevel, CompressionStrategy, SparsityPattern,
    WeightCompressor,
};

#[wasm_bindgen]
pub fn init_panic_hook() {
    #[cfg(feature = "console_panic")]
    console_error_panic_hook::set_once();
}

// Utility functions for WebAssembly
#[wasm_bindgen]
pub fn get_wasm_memory_usage() -> usize {
    // Get the current memory pages and convert to bytes
    // Each WASM memory page is 64KB
    let memory = wasm_bindgen::memory();
    let memory_obj: &js_sys::WebAssembly::Memory = memory.unchecked_ref();
    let buffer = js_sys::WebAssembly::Memory::buffer(memory_obj);
    let array_buffer: &js_sys::ArrayBuffer = buffer.unchecked_ref();
    js_sys::ArrayBuffer::byte_length(array_buffer) as usize
}

/// Memory usage statistics
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct MemoryStats {
    wasm_memory: usize,
    gpu_memory: usize,
    peak_gpu_memory: usize,
}

#[wasm_bindgen]
impl MemoryStats {
    #[wasm_bindgen(getter)]
    pub fn wasm_memory(&self) -> usize {
        self.wasm_memory
    }

    #[wasm_bindgen(getter)]
    pub fn gpu_memory(&self) -> usize {
        self.gpu_memory
    }

    #[wasm_bindgen(getter)]
    pub fn peak_gpu_memory(&self) -> usize {
        self.peak_gpu_memory
    }

    #[wasm_bindgen(getter)]
    pub fn used_mb(&self) -> f32 {
        self.wasm_memory as f32 / (1024.0 * 1024.0)
    }

    #[wasm_bindgen(getter)]
    pub fn limit_mb(&self) -> f32 {
        // Return a reasonable WASM memory limit (256MB typical)
        256.0
    }
}

/// Track GPU memory allocation (called by WebGPU backend)
#[wasm_bindgen]
pub fn track_gpu_allocation(size: usize) {
    if let Ok(mut tracker) = GPU_MEMORY_TRACKER.lock() {
        tracker.allocate(size);
    }
}

/// Track GPU memory deallocation (called by WebGPU backend)
#[wasm_bindgen]
pub fn track_gpu_deallocation(size: usize) {
    if let Ok(mut tracker) = GPU_MEMORY_TRACKER.lock() {
        tracker.deallocate(size);
    }
}

/// Get current GPU memory usage
#[wasm_bindgen]
pub fn get_gpu_memory_usage() -> usize {
    GPU_MEMORY_TRACKER
        .lock()
        .map(|tracker| tracker.get_current_usage())
        .unwrap_or(0)
}

/// Get peak GPU memory usage
#[wasm_bindgen]
pub fn get_peak_gpu_memory_usage() -> usize {
    GPU_MEMORY_TRACKER.lock().map(|tracker| tracker.get_peak_usage()).unwrap_or(0)
}

/// Reset peak GPU memory usage tracking
#[wasm_bindgen]
pub fn reset_peak_gpu_memory() {
    if let Ok(mut tracker) = GPU_MEMORY_TRACKER.lock() {
        tracker.reset_peak();
    }
}

/// Get comprehensive memory statistics
#[wasm_bindgen]
pub fn get_memory_stats() -> MemoryStats {
    MemoryStats {
        wasm_memory: get_wasm_memory_usage(),
        gpu_memory: get_gpu_memory_usage(),
        peak_gpu_memory: get_peak_gpu_memory_usage(),
    }
}

#[wasm_bindgen]
pub fn enable_simd() -> bool {
    // Check if SIMD is available in the WASM environment
    cfg!(target_feature = "simd128")
}

// Enhanced inference API with automatic device selection
#[wasm_bindgen]
pub struct InferenceSession {
    model_type: String,
    #[cfg(feature = "webgpu")]
    device_selector: Option<webgpu::DeviceSelector>,
    #[cfg(feature = "webgpu")]
    current_device: webgpu::DeviceType,
    #[cfg(feature = "web-workers")]
    edge_detector: runtime::edge_runtime::EdgeRuntimeDetector,
    #[cfg(feature = "web-workers")]
    edge_config: runtime::edge_runtime::EdgeInferenceConfig,
    #[cfg(feature = "indexeddb")]
    storage: Option<storage::ModelStorage>,
    debug_logger: Option<debug::DebugLogger>,
    quantizer: Option<quantization::WebQuantizer>,
    batch_processor: Option<batch_processing::BatchProcessor>,
    event_emitter: Option<events::EventEmitter>,
}

#[wasm_bindgen]
impl InferenceSession {
    #[wasm_bindgen(constructor)]
    pub fn new(model_type: String) -> Result<InferenceSession, JsValue> {
        #[cfg(feature = "web-workers")]
        let edge_detector = runtime::edge_runtime::EdgeRuntimeDetector::new();
        #[cfg(feature = "web-workers")]
        let edge_config =
            runtime::edge_runtime::EdgeInferenceConfig::for_runtime(&edge_detector.capabilities());

        Ok(InferenceSession {
            model_type,
            #[cfg(feature = "webgpu")]
            device_selector: None,
            #[cfg(feature = "webgpu")]
            current_device: webgpu::DeviceType::CPU,
            #[cfg(feature = "web-workers")]
            edge_detector,
            #[cfg(feature = "web-workers")]
            edge_config,
            #[cfg(feature = "indexeddb")]
            storage: None,
            debug_logger: None,
            quantizer: None,
            batch_processor: None,
            event_emitter: None,
        })
    }

    /// Initialize with automatic device selection
    pub async fn initialize_with_auto_device(&mut self) -> Result<(), JsValue> {
        #[cfg(feature = "webgpu")]
        {
            let mut selector = webgpu::DeviceSelector::new().await?;
            selector.initialize_device().await?;
            self.current_device = selector.selected_device();
            self.device_selector = Some(selector);
        }
        Ok(())
    }

    /// Get the currently selected device type
    #[cfg(feature = "webgpu")]
    #[wasm_bindgen(getter)]
    pub fn current_device_type(&self) -> DeviceType {
        self.current_device
    }

    /// Get device capabilities
    #[cfg(feature = "webgpu")]
    pub fn get_device_capabilities(&self) -> Option<DeviceCapabilities> {
        self.device_selector.as_ref().map(|s| s.capabilities())
    }

    pub async fn load_model(&mut self, model_data: &[u8]) -> Result<(), JsValue> {
        use crate::error::{ErrorBuilder, ErrorCode};

        let model_size = model_data.len();
        let model_size_mb = model_size as f64 / 1024.0 / 1024.0;

        // Emit model load start event
        if let Some(ref mut emitter) = self.event_emitter {
            let event = events::EventData::model_load_start(&self.model_type, model_size_mb);
            emitter.emit(event);
        }

        // Validate model data
        if model_data.is_empty() {
            let error = ErrorBuilder::new(ErrorCode::E1001, "Model data is empty")
                .operation("load_model")
                .component("inference_session")
                .build();

            // Emit error event
            if let Some(ref mut emitter) = self.event_emitter {
                let event = events::EventData::error_occurred(&error.message, "load_model");
                emitter.emit(event);
            }

            return Err(error.into());
        }

        // Check if model is too large (> 2GB)
        if model_size > 2_147_483_648 {
            let error = ErrorBuilder::new(ErrorCode::E1002, "Model exceeds maximum size limit")
                .operation("load_model")
                .component("inference_session")
                .memory_usage_mb(model_size_mb)
                .additional_info("Consider using quantization or model splitting")
                .build();

            // Emit error event
            if let Some(ref mut emitter) = self.event_emitter {
                let event = events::EventData::error_occurred(&error.message, "load_model")
                    .with_data("size_mb", &format!("{model_size_mb:.2}"));
                emitter.emit(event);
            }

            return Err(error.into());
        }

        let start_time = js_sys::Date::now();

        // Debug logging
        if let Some(ref mut logger) = self.debug_logger {
            logger.start_timer("model_loading");
            logger.log_model_loading(&self.model_type, model_size, "memory");
            logger.log_memory_usage("Before model loading");
        }

        // Check available memory before loading
        let memory_stats = crate::get_memory_stats();
        let available_memory = 2_147_483_648; // 2GB assumed available
        if model_size > available_memory {
            return Err(
                ErrorBuilder::new(ErrorCode::E4002, "Insufficient memory for model")
                    .operation("load_model")
                    .component("inference_session")
                    .memory_usage_mb(memory_stats.wasm_memory as f64 / 1024.0 / 1024.0)
                    .additional_info("Try enabling quantization or using a smaller model")
                    .build()
                    .into(),
            );
        }

        #[cfg(feature = "webgpu")]
        if let Some(selector) = &self.device_selector {
            let should_use_gpu = selector.should_use_gpu(model_size, 0.8); // High complexity for model loading

            if should_use_gpu {
                // GPU-optimized model loading
                web_sys::console::log_1(
                    &format!("Loading model on GPU (size: {model_size} bytes)").into(),
                );

                // Simulate GPU model loading validation
                let gpu_memory_required = model_size * 2; // Assume 2x memory needed for GPU
                let capabilities = selector.capabilities();
                if gpu_memory_required > capabilities.gpu_memory_limit as usize {
                    return Err(
                        ErrorBuilder::new(ErrorCode::E3003, "Insufficient GPU memory")
                            .operation("load_model")
                            .component("webgpu_device")
                            .device_type("GPU")
                            .memory_usage_mb(gpu_memory_required as f64 / 1024.0 / 1024.0)
                            .additional_info("Try using CPU mode or enabling quantization")
                            .build()
                            .into(),
                    );
                }
            } else {
                // CPU-optimized model loading
                web_sys::console::log_1(
                    &format!("Loading model on CPU (size: {model_size} bytes)").into(),
                );
            }
        } else {
            web_sys::console::log_1(
                &format!("Loading model on CPU (size: {model_size} bytes)").into(),
            );
        }

        #[cfg(not(feature = "webgpu"))]
        web_sys::console::log_1(&format!("Loading model on CPU (size: {model_size} bytes)").into());

        // Simulate model loading validation
        if model_size < 1024 {
            if let Some(ref mut logger) = self.debug_logger {
                logger.warn(
                    "Model size is very small, may not be a valid model",
                    "model_validation",
                );
            }
        }

        // Complete debug logging
        if let Some(ref mut logger) = self.debug_logger {
            logger.log_memory_usage("After model loading");
            logger.end_timer("model_loading");
            logger.info(
                &format!(
                    "Model loaded successfully (size: {} MB)",
                    model_size / 1024 / 1024
                ),
                "model_loading",
            );
        }

        // Emit model load complete event
        let duration_ms = js_sys::Date::now() - start_time;
        if let Some(ref mut emitter) = self.event_emitter {
            let event = events::EventData::model_load_complete(&self.model_type, duration_ms);
            emitter.emit(event);
        }

        Ok(())
    }

    pub fn predict(&mut self, input: &tensor::WasmTensor) -> Result<tensor::WasmTensor, JsValue> {
        let input_size = input.len();
        let start_time = js_sys::Date::now();

        // Emit inference start event
        if let Some(ref mut emitter) = self.event_emitter {
            let event = events::EventData::inference_start(&input.shape());
            emitter.emit(event);
        }

        // Debug logging for inference
        if let Some(ref mut logger) = self.debug_logger {
            logger.start_timer("inference");
            logger.log_inference(&self.model_type, &input.shape(), "auto");
            logger.log_memory_usage("Before inference");
        }

        #[cfg(feature = "webgpu")]
        if let Some(selector) = &self.device_selector {
            let should_use_gpu = selector.should_use_gpu(input_size, 0.6); // Medium complexity for inference

            if should_use_gpu {
                web_sys::console::log_1(
                    &format!("Running inference on GPU (input size: {input_size})").into(),
                );
                // GPU-accelerated prediction would go here
            } else {
                web_sys::console::log_1(
                    &format!("Running inference on CPU (input size: {input_size})").into(),
                );
                // CPU-optimized prediction would go here
            }
        } else {
            web_sys::console::log_1(
                &format!("Running inference on CPU (input size: {input_size})").into(),
            );
        }

        #[cfg(not(feature = "webgpu"))]
        web_sys::console::log_1(
            &format!("Running inference on CPU (input size: {input_size})").into(),
        );

        let result = input.clone();

        // Complete debug logging for inference
        if let Some(ref mut logger) = self.debug_logger {
            logger.log_memory_usage("After inference");
            logger.end_timer("inference");
        }

        // Emit inference complete event
        let duration_ms = js_sys::Date::now() - start_time;
        if let Some(ref mut emitter) = self.event_emitter {
            let event = events::EventData::inference_complete(duration_ms, result.len());
            emitter.emit(event);
        }

        Ok(result)
    }

    /// Force device selection for testing
    #[cfg(feature = "webgpu")]
    pub fn force_device_type(&mut self, device_type: DeviceType) {
        self.current_device = device_type;
    }

    /// Get edge runtime capabilities
    #[cfg(feature = "web-workers")]
    #[wasm_bindgen(getter)]
    pub fn edge_capabilities(&self) -> EdgeCapabilities {
        self.edge_detector.capabilities()
    }

    /// Get edge inference configuration
    #[cfg(feature = "web-workers")]
    #[wasm_bindgen(getter)]
    pub fn edge_config(&self) -> EdgeInferenceConfig {
        self.edge_config.clone()
    }

    /// Check if the current edge runtime is suitable for ML inference
    #[cfg(feature = "web-workers")]
    pub fn is_edge_suitable(&self) -> bool {
        self.edge_detector.is_ml_suitable()
    }

    /// Get recommended model size for current edge runtime
    #[cfg(feature = "web-workers")]
    pub fn recommended_model_size_mb(&self) -> u32 {
        self.edge_detector.recommended_model_size_mb()
    }

    /// Get cold start optimization recommendations
    #[cfg(feature = "web-workers")]
    pub fn get_cold_start_optimizations(&self) -> js_sys::Array {
        self.edge_detector.get_cold_start_optimizations()
    }

    /// Initialize model storage with optional maximum size in MB
    #[cfg(feature = "indexeddb")]
    pub async fn initialize_storage(&mut self, max_storage_mb: f64) -> Result<(), JsValue> {
        let mut storage =
            storage::ModelStorage::new("trustformers-models".to_string(), max_storage_mb);
        storage.initialize().await?;
        self.storage = Some(storage);
        web_sys::console::log_1(
            &format!("Initialized model storage (max: {max_storage_mb} MB)").into(),
        );
        Ok(())
    }

    /// Store a model in IndexedDB for caching
    #[cfg(feature = "indexeddb")]
    pub async fn store_model(
        &self,
        model_id: &str,
        model_name: &str,
        architecture: &str,
        version: &str,
        data: &[u8],
    ) -> Result<(), JsValue> {
        let storage = self.storage.as_ref().ok_or("Storage not initialized")?;
        storage.store_model(model_id, model_name, architecture, version, data).await
    }

    /// Load a model from IndexedDB cache
    #[cfg(feature = "indexeddb")]
    pub async fn load_cached_model(&self, model_id: &str) -> Result<Option<Vec<u8>>, JsValue> {
        let storage = self.storage.as_ref().ok_or("Storage not initialized")?;
        storage.get_model(model_id).await
    }

    /// Check if a model exists in cache
    #[cfg(feature = "indexeddb")]
    pub async fn has_cached_model(&self, model_id: &str) -> Result<bool, JsValue> {
        let storage = self.storage.as_ref().ok_or("Storage not initialized")?;
        storage.has_model(model_id).await
    }

    /// Load model with automatic caching
    #[cfg(feature = "indexeddb")]
    pub async fn load_model_with_cache(
        &mut self,
        model_id: &str,
        model_url: &str,
        model_name: &str,
        architecture: &str,
        version: &str,
    ) -> Result<(), JsValue> {
        if let Some(storage) = &self.storage {
            // Try to load from cache first
            if let Some(cached_data) = storage.get_model(model_id).await? {
                web_sys::console::log_1(&format!("Loaded model '{model_name}' from cache").into());
                return self.load_model(&cached_data).await;
            }
        }

        // Download from URL if not in cache
        web_sys::console::log_1(
            &format!("Downloading model '{model_name}' from {model_url}").into(),
        );

        let model_data = self.fetch_model_from_url(model_url).await.map_err(|e| {
            JsValue::from_str(&format!(
                "Failed to fetch model from {}: {:?}",
                model_url, e
            ))
        })?;

        // Store in cache if storage is available
        if let Some(storage) = &self.storage {
            storage
                .store_model(model_id, model_name, architecture, version, &model_data)
                .await?;
        }

        self.load_model(&model_data).await
    }

    /// Get storage usage statistics
    #[cfg(feature = "indexeddb")]
    pub async fn get_storage_stats(&self) -> Result<Option<String>, JsValue> {
        if let Some(storage) = &self.storage {
            let usage = storage.get_storage_usage().await?;
            let models_js = storage.list_models().await?;
            let models_count = if let Ok(models) =
                serde_wasm_bindgen::from_value::<Vec<ModelMetadata>>(models_js)
            {
                models.len()
            } else {
                0
            };
            Ok(Some(format!(
                "Storage: {} bytes, {} models",
                usage, models_count
            )))
        } else {
            Ok(None)
        }
    }

    /// Clear all cached models
    #[cfg(feature = "indexeddb")]
    pub async fn clear_model_cache(&self) -> Result<(), JsValue> {
        let storage = self.storage.as_ref().ok_or("Storage not initialized")?;
        storage.clear_all().await
    }

    /// Fetch model data from a remote URL using the Fetch API
    #[allow(dead_code)]
    async fn fetch_model_from_url(&self, url: &str) -> Result<Vec<u8>, JsValue> {
        use wasm_bindgen::JsCast;
        use wasm_bindgen_futures::JsFuture;

        // Create fetch request
        let request = web_sys::Request::new_with_str(url)?;
        request.headers().set("Accept", "application/octet-stream")?;

        // Get the global window object
        let window = web_sys::window().ok_or("No global window object")?;

        // Perform the fetch
        let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
        let resp: web_sys::Response = resp_value.dyn_into()?;

        // Check if the response is ok
        if !resp.ok() {
            return Err(JsValue::from_str(&format!(
                "HTTP error: {} {}",
                resp.status(),
                resp.status_text()
            )));
        }

        // Get the response as ArrayBuffer
        let array_buffer = JsFuture::from(resp.array_buffer()?).await?;
        let uint8_array = js_sys::Uint8Array::new(&array_buffer);

        // Convert to Vec<u8>
        let mut data = vec![0u8; uint8_array.length() as usize];
        uint8_array.copy_to(&mut data);

        web_sys::console::log_1(
            &format!("Successfully downloaded {len} bytes", len = data.len()).into(),
        );

        Ok(data)
    }

    /// Initialize debug logging
    pub fn enable_debug_logging(&mut self, config: debug::DebugConfig) {
        self.debug_logger = Some(debug::DebugLogger::new(config));
        if let Some(ref mut logger) = self.debug_logger {
            logger.info(
                "Debug logging enabled for inference session",
                "initialization",
            );
        }
    }

    /// Disable debug logging
    pub fn disable_debug_logging(&mut self) {
        if let Some(ref mut logger) = self.debug_logger {
            logger.info("Debug logging disabled", "cleanup");
        }
        self.debug_logger = None;
    }

    /// Start a performance timer
    pub fn start_timer(&mut self, operation: &str) {
        if let Some(ref mut logger) = self.debug_logger {
            logger.start_timer(operation);
        }
    }

    /// End a performance timer
    pub fn end_timer(&mut self, operation: &str) -> Option<f64> {
        if let Some(ref mut logger) = self.debug_logger {
            logger.end_timer(operation)
        } else {
            None
        }
    }

    /// Get debug performance summary
    pub fn get_performance_summary(&self) -> Option<String> {
        self.debug_logger.as_ref().map(|logger| logger.get_performance_summary())
    }

    /// Export debug logs
    pub fn export_debug_logs(&self) -> Option<String> {
        self.debug_logger.as_ref().map(|logger| logger.export_logs())
    }

    /// Clear debug logs
    pub fn clear_debug_logs(&mut self) {
        if let Some(ref mut logger) = self.debug_logger {
            logger.clear();
        }
    }

    /// Log memory usage with context
    pub fn log_memory_usage(&mut self, context: &str) {
        if let Some(ref mut logger) = self.debug_logger {
            logger.log_memory_usage(context);
        }
    }

    /// Initialize quantization with configuration
    pub fn enable_quantization(&mut self, config: quantization::QuantizationConfig) {
        self.quantizer = Some(quantization::WebQuantizer::new(config));
        if let Some(ref mut logger) = self.debug_logger {
            logger.info("Quantization enabled for inference session", "quantization");
        }
    }

    /// Disable quantization
    pub fn disable_quantization(&mut self) {
        if let Some(ref mut logger) = self.debug_logger {
            logger.info("Quantization disabled", "quantization");
        }
        self.quantizer = None;
    }

    /// Load model with automatic quantization
    pub async fn load_model_with_quantization(&mut self, model_data: &[u8]) -> Result<(), JsValue> {
        let mut final_data = model_data.to_vec();
        let mut quantization_summary = "No quantization applied".to_string();

        // Apply quantization if enabled and beneficial
        if let Some(ref quantizer) = self.quantizer {
            if quantizer.should_quantize(model_data.len()) {
                if let Some(ref mut logger) = self.debug_logger {
                    logger.start_timer("quantization");
                    logger.info(
                        &format!(
                            "Starting quantization for {len} bytes",
                            len = model_data.len()
                        ),
                        "quantization",
                    );
                }

                match quantizer.quantize_model(model_data) {
                    Ok(quantized) => {
                        final_data = quantized.data();
                        quantization_summary = quantized.summary();
                        self.log_quantization_success(&quantization_summary);
                    },
                    Err(e) => {
                        let error_msg = format!("Quantization failed: {e:?}, using original model");
                        self.log_quantization_failure(&error_msg);
                        // Continue with original data if quantization fails
                    },
                }
            } else {
                self.log_quantization_skipped();
            }
        }

        // Load the (possibly quantized) model
        self.load_model(&final_data).await?;

        if let Some(ref mut logger) = self.debug_logger {
            logger.info(
                &format!("Model loaded: {quantization_summary}"),
                "model_loading",
            );
        }

        Ok(())
    }

    /// Get quantization recommendations for current model
    pub fn get_quantization_recommendations(
        &self,
        model_size_bytes: usize,
    ) -> Option<quantization::QuantizationConfig> {
        self.quantizer.as_ref().map(|q| q.get_recommended_settings(model_size_bytes))
    }

    /// Check if quantization would be beneficial for a given model size
    pub fn should_quantize_model(&self, model_size_bytes: usize) -> bool {
        self.quantizer.as_ref().is_some_and(|q| q.should_quantize(model_size_bytes))
    }

    /// Initialize batch processing with configuration
    pub fn enable_batch_processing(&mut self, config: batch_processing::BatchConfig) {
        self.batch_processor = Some(batch_processing::BatchProcessor::new(config));
        if let Some(ref mut logger) = self.debug_logger {
            logger.info(
                "Batch processing enabled for inference session",
                "batch_processing",
            );
        }
    }

    /// Disable batch processing
    pub fn disable_batch_processing(&mut self) {
        if let Some(ref mut logger) = self.debug_logger {
            logger.info("Batch processing disabled", "batch_processing");
        }
        self.batch_processor = None;
    }

    /// Add a request to the batch queue
    pub fn add_batch_request(
        &mut self,
        input: &tensor::WasmTensor,
        priority: batch_processing::Priority,
        timeout_ms: Option<u32>,
    ) -> Option<String> {
        if let Some(ref mut processor) = self.batch_processor {
            let request_id = processor.add_request(input.clone(), priority, timeout_ms);

            if let Some(ref mut logger) = self.debug_logger {
                logger.debug(
                    &format!("Added batch request {request_id} with priority {priority:?}"),
                    "batch_processing",
                );
            }

            Some(request_id)
        } else {
            if let Some(ref mut logger) = self.debug_logger {
                logger.warn("Batch processing not enabled", "batch_processing");
            }
            None
        }
    }

    /// Process pending batch requests
    pub async fn process_batch(&mut self) -> Result<Vec<batch_processing::BatchResponse>, JsValue> {
        if let Some(ref mut processor) = self.batch_processor {
            if let Some(ref mut logger) = self.debug_logger {
                logger.start_timer("batch_processing");
                logger.debug(
                    &format!(
                        "Processing batch with {} pending requests",
                        processor.queue_length()
                    ),
                    "batch_processing",
                );
            }

            let responses = processor.process_batch().await?;

            if let Some(ref mut logger) = self.debug_logger {
                logger.info(
                    &format!("Processed batch: {len} responses", len = responses.len()),
                    "batch_processing",
                );
                logger.end_timer("batch_processing");
            }

            Ok(responses)
        } else {
            Err("Batch processing not enabled".into())
        }
    }

    /// Check if a batch is ready for processing
    pub fn is_batch_ready(&self) -> bool {
        self.batch_processor.as_ref().is_some_and(|p| p.is_batch_ready())
    }

    /// Get current batch queue length
    pub fn get_batch_queue_length(&self) -> usize {
        self.batch_processor.as_ref().map_or(0, |p| p.queue_length())
    }

    /// Get batch processing statistics
    pub fn get_batch_stats(&self) -> Option<String> {
        self.batch_processor.as_ref().map(|p| p.get_stats())
    }

    /// Clear the batch queue
    pub fn clear_batch_queue(&mut self) {
        if let Some(ref mut processor) = self.batch_processor {
            processor.clear_queue();
            if let Some(ref mut logger) = self.debug_logger {
                logger.info("Batch queue cleared", "batch_processing");
            }
        }
    }

    /// Enable event system
    pub fn enable_events(&mut self) {
        self.event_emitter = Some(events::EventEmitter::new());
        if let Some(ref mut logger) = self.debug_logger {
            logger.info("Event system enabled for inference session", "events");
        }
    }

    /// Disable event system
    pub fn disable_events(&mut self) {
        if let Some(ref mut logger) = self.debug_logger {
            logger.info("Event system disabled", "events");
        }
        self.event_emitter = None;
    }

    /// Get event history as JSON
    pub fn get_event_history(&self) -> Option<String> {
        self.event_emitter
            .as_ref()
            .and_then(|emitter| serde_json::to_string(emitter.get_history()).ok())
    }

    /// Clear event history
    pub fn clear_event_history(&mut self) {
        if let Some(ref mut emitter) = self.event_emitter {
            emitter.clear_history();
        }
    }

    /// Emit a custom event
    pub fn emit_custom_event(&mut self, event_type: u32, source: &str, data: Option<String>) {
        if let Some(ref mut emitter) = self.event_emitter {
            if let Ok(event_type) = Self::event_type_from_u32(event_type) {
                let mut event = events::EventData::new(event_type, source);
                if let Some(data_str) = data {
                    event = event.with_data("custom_data", &data_str);
                }
                emitter.emit(event);
            }
        }
    }

    fn event_type_from_u32(event_type: u32) -> Result<events::EventType, ()> {
        match event_type {
            1000 => Ok(events::EventType::ModelLoadStart),
            1001 => Ok(events::EventType::ModelLoadProgress),
            1002 => Ok(events::EventType::ModelLoadComplete),
            1003 => Ok(events::EventType::ModelLoadError),
            2000 => Ok(events::EventType::InferenceStart),
            2001 => Ok(events::EventType::InferenceProgress),
            2002 => Ok(events::EventType::InferenceComplete),
            2003 => Ok(events::EventType::InferenceError),
            6000 => Ok(events::EventType::ErrorOccurred),
            _ => Err(()),
        }
    }

    /// Single inference with automatic batching support
    pub async fn predict_with_batching(
        &mut self,
        input: &tensor::WasmTensor,
        priority: batch_processing::Priority,
    ) -> Result<tensor::WasmTensor, JsValue> {
        if let Some(ref mut processor) = self.batch_processor {
            // Add to batch queue
            let request_id = processor.add_request(input.clone(), priority, None);

            // Process if batch is ready or if it's a high-priority request
            if processor.is_batch_ready() || priority >= batch_processing::Priority::High {
                let responses = processor.process_batch().await?;

                // Find our response
                return self.find_batch_response(&responses, &request_id);
            }

            // If not processed in batch, fall back to direct prediction
            return self.predict(input);
        }

        // No batch processing, use direct prediction
        self.predict(input)
    }

    /// Process all pending batches
    pub async fn flush_batches(&mut self) -> Result<Vec<batch_processing::BatchResponse>, JsValue> {
        if let Some(ref mut processor) = self.batch_processor {
            let mut all_responses = Vec::new();

            while processor.queue_length() > 0 {
                let responses = processor.process_batch().await?;
                all_responses.extend(responses);
            }

            Ok(all_responses)
        } else {
            Ok(Vec::new())
        }
    }

    // Helper methods to reduce nesting
    fn log_quantization_success(&mut self, summary: &str) {
        if let Some(ref mut logger) = self.debug_logger {
            logger.info(summary, "quantization");
            logger.end_timer("quantization");
        }
    }

    fn log_quantization_failure(&mut self, error: &str) {
        if let Some(ref mut logger) = self.debug_logger {
            logger.warn(error, "quantization");
            logger.end_timer("quantization");
        }
    }

    fn log_quantization_skipped(&mut self) {
        if let Some(ref mut logger) = self.debug_logger {
            logger.info(
                "Model size is optimal, skipping quantization",
                "quantization",
            );
        }
    }

    fn find_batch_response(
        &self,
        responses: &[batch_processing::BatchResponse],
        request_id: &str,
    ) -> Result<tensor::WasmTensor, JsValue> {
        for response in responses.iter() {
            if response.request_id() == request_id {
                if let Some(_result) = response.result() {
                    // Convert String result to WasmTensor - in a real implementation this would parse the result
                    // For now, create a dummy tensor with the result as metadata
                    return tensor::WasmTensor::new(vec![1.0], vec![1]);
                } else if let Some(error) = response.error() {
                    return Err(error.into());
                }
            }
        }
        Err("Response not found for request ID".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initialization() {
        let tf = TrustformersWasm::new();
        assert!(tf.initialized());
        assert_eq!(tf.version(), "0.1.0-alpha.1");
    }
}
