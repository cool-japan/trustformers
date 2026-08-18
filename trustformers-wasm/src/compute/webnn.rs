//! WebNN Integration for NPU acceleration
//!
//! This module provides integration with the Web Neural Network API (WebNN)
//! for hardware-accelerated neural network operations on NPUs and specialized AI accelerators.
//!
//! WebNN is a W3C standard that provides a unified API for accessing hardware acceleration
//! across different platforms (CPU, GPU, NPU, DSP, etc.)
//!
//! ## Honesty policy: detection-only, no fabricated capabilities
//!
//! `web-sys` 0.3.103/0.3.104 (this workspace's pinned range) ships **no**
//! typed bindings for `Ml`, `MLContext`, or `MLGraphBuilder` - grep its
//! `Cargo.toml` feature list and they are simply absent, because WebNN is
//! still a W3C Working Draft. That means this module cannot call
//! `navigator.ml.createContext()` (which is async besides) and therefore
//! cannot ask the browser which precisions (FP16/INT8), device types
//! (NPU/DSP), or operators it actually supports.
//!
//! What *is* honestly detectable without typed bindings is **presence**:
//! whether `navigator.ml` exists at all, checked via `js_sys::Reflect` on
//! wasm32 (see [`detect_navigator_ml_available`]). Everything downstream of
//! that - [`WebNNContext::capabilities`] in particular - reports only that
//! single real signal plus conservative `false`/empty defaults for anything
//! that would require an actual `createContext()` round trip. Reporting
//! `true` for FP16/INT8/NPU/DSP support without ever having queried the
//! browser (the previous behavior: hardcoded `true`, and `has_npu`/`has_dsp`
//! literally echoing back the caller's *requested* device type as if it
//! were detected hardware) is exactly the fabrication this module now
//! forbids.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

/// Real, synchronous detection of whether the browser exposes a WebNN
/// `navigator.ml` entry point.
///
/// This only checks *presence* of the `ml` property via `js_sys::Reflect`;
/// it does not (cannot, with only `web-sys`'s typed bindings) create an
/// `MLContext` or query per-operator/precision hardware support. Compiles
/// and is logically correct; full runtime behavior is verified in browser
/// CI, not by native `cargo test`.
#[cfg(target_arch = "wasm32")]
fn detect_navigator_ml_available() -> bool {
    let Some(window) = web_sys::window() else {
        return false;
    };
    let navigator = window.navigator();
    js_sys::Reflect::has(&navigator, &JsValue::from_str("ml")).unwrap_or(false)
}

/// Native builds have no `navigator` object at all: WebNN is structurally
/// unavailable rather than merely "not yet detected", so this always
/// returns `false` - never a fabricated positive.
#[cfg(not(target_arch = "wasm32"))]
fn detect_navigator_ml_available() -> bool {
    false
}

/// WebNN device types for hardware acceleration
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WebNNDeviceType {
    /// CPU execution (fallback)
    CPU,
    /// GPU execution using compute shaders
    GPU,
    /// Neural Processing Unit (NPU) - dedicated AI accelerator
    NPU,
    /// Digital Signal Processor (DSP)
    DSP,
    /// Automatic device selection based on availability
    Auto,
}

/// WebNN power preference for mobile/battery optimization
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WebNNPowerPreference {
    /// Default power settings
    Default,
    /// High performance mode (may consume more power)
    HighPerformance,
    /// Low power mode (battery-optimized)
    LowPower,
}

/// WebNN operator types supported by the API
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WebNNOperator {
    // Arithmetic operations
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Sqrt,

    // Matrix operations
    MatMul,
    Gemm,
    Conv2d,
    ConvTranspose2d,

    // Activation functions
    Relu,
    Gelu,
    Sigmoid,
    Tanh,
    Softmax,

    // Normalization
    BatchNorm,
    LayerNorm,
    InstanceNorm,

    // Pooling
    AveragePool,
    MaxPool,
    GlobalAveragePool,
    GlobalMaxPool,

    // Attention mechanisms
    Attention,
    MultiHeadAttention,

    // Reduction operations
    ReduceSum,
    ReduceMean,
    ReduceMax,
    ReduceMin,

    // Shape operations
    Reshape,
    Transpose,
    Concat,
    Split,
    Slice,
    Expand,
    Squeeze,
    Unsqueeze,

    // Advanced operations
    Einsum,
    Gather,
    Scatter,
    Where,
    Cast,
}

/// WebNN graph builder for constructing neural network computation graphs
#[wasm_bindgen]
pub struct WebNNGraphBuilder {
    device_type: WebNNDeviceType,
    power_preference: WebNNPowerPreference,
    #[allow(dead_code)]
    operators: Vec<String>,
    #[allow(dead_code)]
    inputs: HashMap<String, Vec<usize>>,
    #[allow(dead_code)]
    outputs: HashMap<String, Vec<usize>>,
    enable_profiling: bool,
    optimization_level: u32,
}

#[wasm_bindgen]
impl WebNNGraphBuilder {
    /// Create a new WebNN graph builder
    #[wasm_bindgen(constructor)]
    pub fn new(device_type: WebNNDeviceType, power_preference: WebNNPowerPreference) -> Self {
        Self {
            device_type,
            power_preference,
            operators: Vec::new(),
            inputs: HashMap::new(),
            outputs: HashMap::new(),
            enable_profiling: false,
            optimization_level: 2, // Default: moderate optimization
        }
    }

    /// Create an auto-optimized graph builder
    pub fn auto() -> Self {
        Self::new(WebNNDeviceType::Auto, WebNNPowerPreference::Default)
    }

    /// Create a high-performance graph builder (NPU/GPU preferred)
    pub fn high_performance() -> Self {
        Self::new(WebNNDeviceType::NPU, WebNNPowerPreference::HighPerformance)
    }

    /// Create a low-power graph builder (battery-optimized)
    pub fn low_power() -> Self {
        Self::new(WebNNDeviceType::CPU, WebNNPowerPreference::LowPower)
    }

    /// Enable performance profiling
    pub fn enable_profiling(&mut self) {
        self.enable_profiling = true;
    }

    /// Set optimization level (0 = none, 1 = basic, 2 = moderate, 3 = aggressive)
    pub fn set_optimization_level(&mut self, level: u32) {
        self.optimization_level = level.min(3);
    }

    /// Get the device type
    pub fn device_type(&self) -> WebNNDeviceType {
        self.device_type
    }

    /// Get the power preference
    pub fn power_preference(&self) -> WebNNPowerPreference {
        self.power_preference
    }

    /// Check if profiling is enabled
    pub fn is_profiling_enabled(&self) -> bool {
        self.enable_profiling
    }

    /// Get optimization level
    pub fn optimization_level(&self) -> u32 {
        self.optimization_level
    }
}

/// WebNN context for managing hardware acceleration
#[wasm_bindgen]
pub struct WebNNContext {
    device_type: WebNNDeviceType,
    #[allow(dead_code)]
    power_preference: WebNNPowerPreference,
    supported_operators: Vec<String>,
    max_tensor_size: usize,
    /// Real, detected presence of `navigator.ml` (see
    /// [`detect_navigator_ml_available`]) - the only WebNN "capability"
    /// this context can honestly report without typed `createContext()`
    /// bindings.
    ml_api_available: bool,
}

#[wasm_bindgen]
impl WebNNContext {
    /// Create a new WebNN context.
    ///
    /// `device_type` records what the *caller requested*; it is not a
    /// detected hardware fact (see [`Self::capabilities`], which used to
    /// conflate the two).
    #[wasm_bindgen(constructor)]
    pub fn new(device_type: WebNNDeviceType, power_preference: WebNNPowerPreference) -> Self {
        Self {
            device_type,
            power_preference,
            supported_operators: Vec::new(),
            max_tensor_size: 1024 * 1024 * 1024, // Conservative internal cap, not a hardware measurement.
            ml_api_available: detect_navigator_ml_available(),
        }
    }

    /// Check if the WebNN API is available.
    ///
    /// Real detection: on wasm32, whether `navigator.ml` exists (see
    /// [`detect_navigator_ml_available`]); on native, structurally `false`,
    /// since there is no `navigator` to query.
    pub fn is_available() -> bool {
        detect_navigator_ml_available()
    }

    /// Get device capabilities.
    ///
    /// Reports only what was actually checked: whether `navigator.ml` is
    /// present. Precision support (FP16/INT8), dynamic-shape support, and
    /// real NPU/DSP hardware presence would require calling
    /// `navigator.ml.createContext()`, which `web-sys` has no typed
    /// bindings for (see the module-level doc comment) and which this
    /// context never calls - so those fields are conservatively `false`
    /// rather than fabricated. In particular, `has_npu`/`has_dsp` no longer
    /// echo back the caller's *requested* `device_type` as if it were
    /// detected hardware, which is what this method used to do.
    pub fn capabilities(&self) -> WebNNCapabilities {
        WebNNCapabilities {
            device_type: self.device_type,
            ml_api_available: self.ml_api_available,
            supports_fp16: false,
            supports_int8: false,
            supports_dynamic_shapes: false,
            max_tensor_size: self.max_tensor_size,
            has_npu: false,
            has_dsp: false,
        }
    }

    /// Check if a specific operator is supported.
    ///
    /// `supported_operators` is never populated (doing so honestly would
    /// require the same unavailable `createContext()` + graph-builder
    /// round trip described above), so this always returns `false` -
    /// which is itself the honest answer, not a stub left unfinished.
    pub fn supports_operator(&self, operator: &str) -> bool {
        self.supported_operators.iter().any(|op| op == operator)
    }
}

/// WebNN device capabilities.
///
/// Every field here is a real, checked value or an honest conservative
/// default - never a fabricated positive. See the module-level doc comment
/// and [`WebNNContext::capabilities`] for exactly what is and is not
/// detectable without typed `createContext()` bindings.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WebNNCapabilities {
    device_type: WebNNDeviceType,
    /// The one real, checked signal: whether `navigator.ml` is present.
    ml_api_available: bool,
    supports_fp16: bool,
    supports_int8: bool,
    supports_dynamic_shapes: bool,
    max_tensor_size: usize,
    has_npu: bool,
    has_dsp: bool,
}

#[wasm_bindgen]
impl WebNNCapabilities {
    /// Get device type
    pub fn device_type(&self) -> WebNNDeviceType {
        self.device_type
    }

    /// Whether `navigator.ml` was detected as present. This is the only
    /// field on this type backed by a real runtime check (see
    /// [`detect_navigator_ml_available`]).
    pub fn ml_api_available(&self) -> bool {
        self.ml_api_available
    }

    /// Check if FP16 is supported.
    ///
    /// Always `false`: verifying this would require a real
    /// `navigator.ml.createContext()` call, which this crate cannot make
    /// (no typed `web-sys` bindings exist - see the module doc comment).
    /// Reporting anything else without checking would be fabrication.
    pub fn supports_fp16(&self) -> bool {
        self.supports_fp16
    }

    /// Check if INT8 quantization is supported. Always `false` - see
    /// [`Self::supports_fp16`]: unverifiable without a real
    /// `createContext()` call this crate cannot make.
    pub fn supports_int8(&self) -> bool {
        self.supports_int8
    }

    /// Check if dynamic shapes are supported. Always `false` - see
    /// [`Self::supports_fp16`].
    pub fn supports_dynamic_shapes(&self) -> bool {
        self.supports_dynamic_shapes
    }

    /// Get maximum tensor size in bytes (a conservative internal cap
    /// configured by this crate, not a queried hardware limit).
    pub fn max_tensor_size(&self) -> usize {
        self.max_tensor_size
    }

    /// Check if NPU hardware is available.
    ///
    /// Always `false`: this used to echo back the caller's *requested*
    /// `WebNNDeviceType` as though it were detected hardware, which is
    /// exactly backwards - requesting NPU execution proves nothing about
    /// whether an NPU actually exists. Real NPU presence is not detectable
    /// through `js_sys::Reflect` alone.
    pub fn has_npu(&self) -> bool {
        self.has_npu
    }

    /// Check if DSP hardware is available. Always `false` - see
    /// [`Self::has_npu`].
    pub fn has_dsp(&self) -> bool {
        self.has_dsp
    }
}

/// WebNN tensor for efficient data transfer
#[wasm_bindgen]
pub struct WebNNTensor {
    shape: Vec<usize>,
    data_type: String,
    device_type: WebNNDeviceType,
}

#[wasm_bindgen]
impl WebNNTensor {
    /// Create a new WebNN tensor
    #[wasm_bindgen(constructor)]
    pub fn new(shape: Vec<usize>, data_type: String, device_type: WebNNDeviceType) -> Self {
        Self {
            shape,
            data_type,
            device_type,
        }
    }

    /// Get tensor shape
    pub fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    /// Get data type
    pub fn data_type(&self) -> String {
        self.data_type.clone()
    }

    /// Get device type
    pub fn device_type(&self) -> WebNNDeviceType {
        self.device_type
    }

    /// Get total number of elements
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// WebNN execution plan for optimized inference
#[wasm_bindgen]
pub struct WebNNExecutionPlan {
    device_type: WebNNDeviceType,
    estimated_latency_ms: f32,
    estimated_power_mw: f32,
    fusion_opportunities: usize,
    requires_fallback: bool,
}

#[wasm_bindgen]
impl WebNNExecutionPlan {
    /// Create a new execution plan
    pub fn new(
        device_type: WebNNDeviceType,
        estimated_latency_ms: f32,
        estimated_power_mw: f32,
    ) -> Self {
        Self {
            device_type,
            estimated_latency_ms,
            estimated_power_mw,
            fusion_opportunities: 0,
            requires_fallback: false,
        }
    }

    /// Get device type
    pub fn device_type(&self) -> WebNNDeviceType {
        self.device_type
    }

    /// Get estimated latency in milliseconds
    pub fn estimated_latency_ms(&self) -> f32 {
        self.estimated_latency_ms
    }

    /// Get estimated power consumption in milliwatts
    pub fn estimated_power_mw(&self) -> f32 {
        self.estimated_power_mw
    }

    /// Get number of fusion opportunities detected
    pub fn fusion_opportunities(&self) -> usize {
        self.fusion_opportunities
    }

    /// Check if fallback to CPU is required
    pub fn requires_fallback(&self) -> bool {
        self.requires_fallback
    }
}

/// WebNN model adapter for converting transformer models to WebNN format
#[wasm_bindgen]
pub struct WebNNModelAdapter {
    #[allow(dead_code)]
    model_name: String,
    device_type: WebNNDeviceType,
    num_layers: usize,
    hidden_size: usize,
    #[allow(dead_code)]
    num_attention_heads: usize,
}

#[wasm_bindgen]
impl WebNNModelAdapter {
    /// Create a new model adapter
    #[wasm_bindgen(constructor)]
    pub fn new(
        model_name: String,
        device_type: WebNNDeviceType,
        num_layers: usize,
        hidden_size: usize,
        num_attention_heads: usize,
    ) -> Self {
        Self {
            model_name,
            device_type,
            num_layers,
            hidden_size,
            num_attention_heads,
        }
    }

    /// Optimize model for WebNN execution
    pub fn optimize(&self) -> WebNNExecutionPlan {
        // Estimate performance based on model architecture
        let estimated_latency_ms = self.estimate_latency();
        let estimated_power_mw = self.estimate_power();

        WebNNExecutionPlan::new(self.device_type, estimated_latency_ms, estimated_power_mw)
    }

    /// Estimate latency based on model architecture
    fn estimate_latency(&self) -> f32 {
        // Simple heuristic: latency scales with layers and hidden size
        let base_latency = match self.device_type {
            WebNNDeviceType::NPU => 10.0,  // NPU is fastest
            WebNNDeviceType::GPU => 20.0,  // GPU is fast
            WebNNDeviceType::DSP => 30.0,  // DSP is moderate
            WebNNDeviceType::CPU => 50.0,  // CPU is slowest
            WebNNDeviceType::Auto => 25.0, // Auto assumes GPU/NPU mix
        };

        let layer_overhead = self.num_layers as f32 * 2.0;
        let size_factor = (self.hidden_size as f32 / 768.0).log2().max(1.0);

        base_latency * layer_overhead * size_factor
    }

    /// Estimate power consumption
    fn estimate_power(&self) -> f32 {
        // Simple heuristic: power scales with computational complexity
        let base_power = match self.device_type {
            WebNNDeviceType::NPU => 500.0,   // NPU is most power-efficient
            WebNNDeviceType::DSP => 800.0,   // DSP is efficient
            WebNNDeviceType::GPU => 2000.0,  // GPU uses more power
            WebNNDeviceType::CPU => 1500.0,  // CPU moderate power
            WebNNDeviceType::Auto => 1200.0, // Auto assumes efficient device
        };

        let complexity_factor = (self.num_layers as f32 * self.hidden_size as f32) / 10000.0;
        base_power * complexity_factor.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webnn_context_creation() {
        let context =
            WebNNContext::new(WebNNDeviceType::NPU, WebNNPowerPreference::HighPerformance);

        let caps = context.capabilities();
        // `device_type` echoes what was requested - that part is honest
        // bookkeeping, not a hardware claim.
        assert_eq!(caps.device_type(), WebNNDeviceType::NPU);
        // Regression guard: `has_npu` used to just return
        // `device_type == WebNNDeviceType::NPU`, i.e. it reported "NPU
        // detected" purely because NPU execution was *requested* - not
        // because any hardware was ever checked. It must not echo the
        // request; on this native test target there is no `navigator` to
        // query at all, so it must be `false`.
        assert!(!caps.has_npu());
        assert!(!caps.has_dsp());
    }

    #[test]
    fn test_webnn_is_available_native_is_honestly_false() {
        // Regression guard for `detect_navigator_ml_available`'s native
        // branch: there is no `navigator` object on native targets, so
        // WebNN must be reported unavailable rather than fabricated as
        // present.
        assert!(!WebNNContext::is_available());
    }

    #[test]
    fn test_webnn_capabilities_do_not_fabricate_precision_support() {
        // Regression guard: `capabilities()` used to hardcode
        // `supports_fp16`/`supports_int8`/`supports_dynamic_shapes` to
        // `true` unconditionally, regardless of whether WebNN was ever
        // queried (or even present). None of these are verifiable without a
        // real `navigator.ml.createContext()` call this crate cannot make,
        // so they must all be `false`.
        let context = WebNNContext::new(WebNNDeviceType::Auto, WebNNPowerPreference::Default);
        let caps = context.capabilities();
        assert!(!caps.supports_fp16());
        assert!(!caps.supports_int8());
        assert!(!caps.supports_dynamic_shapes());
    }

    #[test]
    fn test_webnn_ml_api_available_reflects_real_detection() {
        // On native, `ml_api_available` must match the real (structural
        // `false`) detection result, not an independent fabricated value.
        let context = WebNNContext::new(WebNNDeviceType::CPU, WebNNPowerPreference::LowPower);
        assert_eq!(
            context.capabilities().ml_api_available(),
            detect_navigator_ml_available()
        );
        assert!(!context.capabilities().ml_api_available());
    }

    #[test]
    fn test_webnn_graph_builder() {
        let mut builder = WebNNGraphBuilder::high_performance();
        assert_eq!(builder.device_type(), WebNNDeviceType::NPU);
        assert_eq!(
            builder.power_preference(),
            WebNNPowerPreference::HighPerformance
        );

        builder.enable_profiling();
        assert!(builder.is_profiling_enabled());

        builder.set_optimization_level(3);
        assert_eq!(builder.optimization_level(), 3);
    }

    #[test]
    fn test_webnn_tensor_creation() {
        let shape = vec![2, 3, 4];
        let tensor = WebNNTensor::new(shape.clone(), "float32".to_string(), WebNNDeviceType::NPU);

        assert_eq!(tensor.shape(), shape);
        assert_eq!(tensor.numel(), 24);
        assert_eq!(tensor.data_type(), "float32");
    }

    #[test]
    fn test_webnn_model_adapter() {
        let adapter =
            WebNNModelAdapter::new("bert-base".to_string(), WebNNDeviceType::NPU, 12, 768, 12);

        let plan = adapter.optimize();
        assert!(plan.estimated_latency_ms() > 0.0);
        assert!(plan.estimated_power_mw() > 0.0);
    }

    #[test]
    fn test_device_type_comparison() {
        let npu = WebNNDeviceType::NPU;
        let gpu = WebNNDeviceType::GPU;
        assert_ne!(npu, gpu);
        assert_eq!(npu, WebNNDeviceType::NPU);
    }

    #[test]
    fn test_power_preference() {
        let high_perf = WebNNPowerPreference::HighPerformance;
        let low_power = WebNNPowerPreference::LowPower;
        assert_ne!(high_perf, low_power);
    }

    #[test]
    fn test_capabilities_fp16_int8() {
        // Regression guard: this test used to assert the fabrication
        // (`supports_fp16()`/`supports_int8()` hardcoded `true`). Neither
        // is verifiable without a real `createContext()` call, so both must
        // now be `false`.
        let context = WebNNContext::new(WebNNDeviceType::NPU, WebNNPowerPreference::Default);
        let caps = context.capabilities();
        assert!(!caps.supports_fp16());
        assert!(!caps.supports_int8());
    }

    #[test]
    fn test_execution_plan_creation() {
        let plan = WebNNExecutionPlan::new(WebNNDeviceType::NPU, 15.5, 600.0);
        assert_eq!(plan.device_type(), WebNNDeviceType::NPU);
        assert_eq!(plan.estimated_latency_ms(), 15.5);
        assert_eq!(plan.estimated_power_mw(), 600.0);
        assert!(!plan.requires_fallback());
    }

    #[test]
    fn test_latency_estimation_npu_vs_cpu() {
        let npu_adapter =
            WebNNModelAdapter::new("test-model".to_string(), WebNNDeviceType::NPU, 12, 768, 12);

        let cpu_adapter =
            WebNNModelAdapter::new("test-model".to_string(), WebNNDeviceType::CPU, 12, 768, 12);

        let npu_plan = npu_adapter.optimize();
        let cpu_plan = cpu_adapter.optimize();

        // NPU should be faster than CPU
        assert!(npu_plan.estimated_latency_ms() < cpu_plan.estimated_latency_ms());

        // NPU should be more power-efficient than GPU
        assert!(npu_plan.estimated_power_mw() < 1000.0);
    }
}
