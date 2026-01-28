//! React Native Turbo Module Implementation for TrustformeRS Mobile
//!
//! This module provides Turbo Module support for React Native 0.68+ with improved
//! performance, type safety, and modern React Native architecture integration.

use crate::react_native::{
    InferenceRequest, InferenceResponse, ReactNativeConfig, TrustformersReactNative,
};
use crate::MobileConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use trustformers_core::error::Result;

/// Turbo Module specification for TrustformeRS
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboModuleSpec {
    /// Module name
    pub name: String,
    /// Module version
    pub version: String,
    /// Supported methods
    pub methods: Vec<TurboMethodSpec>,
    /// Type definitions
    pub types: HashMap<String, TurboTypeSpec>,
}

/// Turbo Module method specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboMethodSpec {
    /// Method name
    pub name: String,
    /// Method type (sync/async/promise)
    pub method_type: TurboMethodType,
    /// Parameters
    pub parameters: Vec<TurboParameter>,
    /// Return type
    pub return_type: String,
    /// Whether method is optional
    pub optional: bool,
}

/// Turbo Module method types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TurboMethodType {
    /// Synchronous method
    Sync,
    /// Asynchronous method with callback
    Async,
    /// Promise-based method
    Promise,
}

/// Turbo Module parameter specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: String,
    /// Whether parameter is optional
    pub optional: bool,
    /// Default value
    pub default: Option<serde_json::Value>,
}

/// Turbo Module type specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboTypeSpec {
    /// Type name
    pub name: String,
    /// Type kind (object, array, primitive)
    pub kind: TurboTypeKind,
    /// Properties for object types
    pub properties: Option<HashMap<String, TurboProperty>>,
    /// Element type for array types
    pub element_type: Option<String>,
}

/// Turbo Module type kinds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TurboTypeKind {
    Object,
    Array,
    Primitive,
    Union,
}

/// Turbo Module property specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboProperty {
    /// Property type
    pub prop_type: String,
    /// Whether property is optional
    pub optional: bool,
    /// Property description
    pub description: Option<String>,
}

/// Turbo Module configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboModuleConfig {
    /// Enable new architecture features
    pub enable_new_architecture: bool,
    /// Enable Fabric renderer integration
    pub enable_fabric: bool,
    /// Enable Hermes optimizations
    pub enable_hermes_optimizations: bool,
    /// Enable concurrent features
    pub enable_concurrent_features: bool,
    /// JavaScript interface configuration
    pub js_interface: TurboJSInterfaceConfig,
    /// Performance configuration
    pub performance: TurboPerformanceConfig,
}

/// Turbo Module JavaScript interface configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboJSInterfaceConfig {
    /// Enable automatic type generation
    pub enable_auto_type_generation: bool,
    /// Enable runtime type checking
    pub enable_runtime_type_checking: bool,
    /// Type definition output path
    pub type_definition_path: Option<String>,
    /// Use strict TypeScript mode
    pub strict_typescript: bool,
}

/// Turbo Module performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboPerformanceConfig {
    /// Enable lazy loading
    pub enable_lazy_loading: bool,
    /// Enable method call caching
    pub enable_method_caching: bool,
    /// Cache size limit
    pub cache_size_limit: usize,
    /// Enable async execution
    pub enable_async_execution: bool,
    /// Maximum concurrent operations
    pub max_concurrent_ops: usize,
}

/// Turbo Module method result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboMethodResult {
    /// Success flag
    pub success: bool,
    /// Result data
    pub data: Option<serde_json::Value>,
    /// Error information
    pub error: Option<TurboError>,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Method metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Turbo Module error information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Error stack trace
    pub stack: Option<String>,
    /// Additional error data
    pub data: Option<serde_json::Value>,
}

/// TrustformeRS Turbo Module implementation
pub struct TrustformersTurboModule {
    config: TurboModuleConfig,
    base_module: Arc<Mutex<TrustformersReactNative>>,
    method_cache: Arc<Mutex<HashMap<String, TurboMethodResult>>>,
    spec: TurboModuleSpec,
    performance_monitor: Arc<Mutex<TurboPerformanceMonitor>>,
}

/// Performance monitoring for Turbo Module
#[derive(Debug, Clone)]
struct TurboPerformanceMonitor {
    method_call_counts: HashMap<String, usize>,
    method_execution_times: HashMap<String, Vec<f64>>,
    cache_hit_ratio: f64,
    total_operations: usize,
    successful_operations: usize,
}

/// Global Turbo Module instance
static TURBO_MODULE: OnceLock<Arc<Mutex<TrustformersTurboModule>>> = OnceLock::new();

impl TrustformersTurboModule {
    /// Create new Turbo Module
    pub fn new(
        config: TurboModuleConfig,
        rn_config: ReactNativeConfig,
        mobile_config: MobileConfig,
    ) -> Result<Self> {
        let base_module = Arc::new(Mutex::new(TrustformersReactNative::new(
            rn_config,
            mobile_config,
        )?));

        let method_cache = Arc::new(Mutex::new(HashMap::new()));
        let spec = Self::generate_turbo_spec();
        let performance_monitor = Arc::new(Mutex::new(TurboPerformanceMonitor::new()));

        Ok(Self {
            config,
            base_module,
            method_cache,
            spec,
            performance_monitor,
        })
    }

    /// Initialize Turbo Module
    pub async fn initialize(&self) -> Result<TurboMethodResult> {
        let start_time = std::time::Instant::now();

        let init_result = {
            let base = self.base_module.lock().expect("Failed to acquire lock");
            base.initialize()
        };

        let execution_time = start_time.elapsed().as_millis() as f64;

        match init_result {
            Ok(data) => {
                self.update_performance_stats("initialize", execution_time, true);

                let mut metadata = HashMap::new();
                metadata.insert(
                    "turbo_module_version".to_string(),
                    serde_json::Value::String(env!("CARGO_PKG_VERSION").to_string()),
                );
                metadata.insert(
                    "new_architecture_enabled".to_string(),
                    serde_json::Value::Bool(self.config.enable_new_architecture),
                );

                Ok(TurboMethodResult {
                    success: true,
                    data: Some(serde_json::from_str(&data)?),
                    error: None,
                    execution_time_ms: execution_time,
                    metadata,
                })
            },
            Err(e) => {
                self.update_performance_stats("initialize", execution_time, false);

                Ok(TurboMethodResult {
                    success: false,
                    data: None,
                    error: Some(TurboError {
                        code: "INIT_ERROR".to_string(),
                        message: e.to_string(),
                        stack: None,
                        data: None,
                    }),
                    execution_time_ms: execution_time,
                    metadata: HashMap::new(),
                })
            },
        }
    }

    /// Perform inference with Turbo Module optimizations
    pub async fn inference(&self, request: &InferenceRequest) -> Result<TurboMethodResult> {
        let start_time = std::time::Instant::now();
        let method_name = "inference";

        // Check method cache if enabled
        if self.config.performance.enable_method_caching {
            if let Some(cached_result) = self.check_method_cache(method_name, request) {
                self.update_cache_stats(true);
                return Ok(cached_result);
            }
        }

        self.update_cache_stats(false);

        // Perform inference
        let inference_result = {
            let base = self.base_module.lock().expect("Failed to acquire lock");
            let request_json = serde_json::to_string(request)?;
            base.inference(&request_json).await
        };

        let execution_time = start_time.elapsed().as_millis() as f64;

        match inference_result {
            Ok(response_json) => {
                let response: InferenceResponse = serde_json::from_str(&response_json)?;

                let mut metadata = HashMap::new();
                metadata.insert(
                    "cache_enabled".to_string(),
                    serde_json::Value::Bool(self.config.performance.enable_method_caching),
                );
                metadata.insert(
                    "async_execution".to_string(),
                    serde_json::Value::Bool(self.config.performance.enable_async_execution),
                );
                metadata.insert(
                    "inference_engine_time_ms".to_string(),
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(response.inference_time_ms)
                            .expect("Failed to get value"),
                    ),
                );

                let result = TurboMethodResult {
                    success: response.success,
                    data: Some(serde_json::to_value(&response)?),
                    error: response.error_message.map(|msg| TurboError {
                        code: "INFERENCE_ERROR".to_string(),
                        message: msg,
                        stack: None,
                        data: None,
                    }),
                    execution_time_ms: execution_time,
                    metadata,
                };

                // Cache successful results
                if self.config.performance.enable_method_caching && response.success {
                    self.cache_method_result(method_name, request, &result);
                }

                self.update_performance_stats(method_name, execution_time, response.success);
                Ok(result)
            },
            Err(e) => {
                self.update_performance_stats(method_name, execution_time, false);

                Ok(TurboMethodResult {
                    success: false,
                    data: None,
                    error: Some(TurboError {
                        code: "INFERENCE_ERROR".to_string(),
                        message: e.to_string(),
                        stack: None,
                        data: None,
                    }),
                    execution_time_ms: execution_time,
                    metadata: HashMap::new(),
                })
            },
        }
    }

    /// Get device capabilities with Turbo Module enhancements
    pub async fn get_device_capabilities(&self) -> Result<TurboMethodResult> {
        let start_time = std::time::Instant::now();

        let capabilities_result = {
            let base = self.base_module.lock().expect("Failed to acquire lock");
            base.get_device_capabilities()
        };

        let execution_time = start_time.elapsed().as_millis() as f64;

        match capabilities_result {
            Ok(capabilities_json) => {
                let capabilities: serde_json::Value = serde_json::from_str(&capabilities_json)?;

                let mut enhanced_capabilities =
                    capabilities.as_object().expect("Failed to get value").clone();
                enhanced_capabilities.insert(
                    "turbo_module_support".to_string(),
                    serde_json::Value::Bool(true),
                );
                enhanced_capabilities.insert(
                    "new_architecture_support".to_string(),
                    serde_json::Value::Bool(self.config.enable_new_architecture),
                );
                enhanced_capabilities.insert(
                    "fabric_support".to_string(),
                    serde_json::Value::Bool(self.config.enable_fabric),
                );

                let mut metadata = HashMap::new();
                metadata.insert(
                    "turbo_module_version".to_string(),
                    serde_json::Value::String(self.spec.version.clone()),
                );

                self.update_performance_stats("get_device_capabilities", execution_time, true);

                Ok(TurboMethodResult {
                    success: true,
                    data: Some(serde_json::Value::Object(enhanced_capabilities)),
                    error: None,
                    execution_time_ms: execution_time,
                    metadata,
                })
            },
            Err(e) => {
                self.update_performance_stats("get_device_capabilities", execution_time, false);

                Ok(TurboMethodResult {
                    success: false,
                    data: None,
                    error: Some(TurboError {
                        code: "CAPABILITIES_ERROR".to_string(),
                        message: e.to_string(),
                        stack: None,
                        data: None,
                    }),
                    execution_time_ms: execution_time,
                    metadata: HashMap::new(),
                })
            },
        }
    }

    /// Get Turbo Module specification
    pub fn get_spec(&self) -> &TurboModuleSpec {
        &self.spec
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> Result<TurboMethodResult> {
        let start_time = std::time::Instant::now();

        let monitor = self.performance_monitor.lock().expect("Failed to acquire lock");
        let stats = serde_json::json!({
            "method_call_counts": monitor.method_call_counts,
            "average_execution_times": monitor.get_average_execution_times(),
            "cache_hit_ratio": monitor.cache_hit_ratio,
            "total_operations": monitor.total_operations,
            "successful_operations": monitor.successful_operations,
            "success_rate": if monitor.total_operations > 0 {
                monitor.successful_operations as f64 / monitor.total_operations as f64
            } else { 0.0 }
        });

        let execution_time = start_time.elapsed().as_millis() as f64;

        Ok(TurboMethodResult {
            success: true,
            data: Some(stats),
            error: None,
            execution_time_ms: execution_time,
            metadata: HashMap::new(),
        })
    }

    /// Generate TypeScript definitions
    pub fn generate_typescript_definitions(&self) -> Result<String> {
        let mut ts_def = String::new();

        ts_def
            .push_str("// Auto-generated TypeScript definitions for TrustformeRS Turbo Module\n\n");
        ts_def.push_str("export interface TrustformersTurboModule {\n");

        for method in &self.spec.methods {
            let params: Vec<String> = method
                .parameters
                .iter()
                .map(|p| {
                    format!(
                        "{}{}: {}",
                        p.name,
                        if p.optional { "?" } else { "" },
                        p.param_type
                    )
                })
                .collect();

            let return_type = match method.method_type {
                TurboMethodType::Sync => method.return_type.clone(),
                TurboMethodType::Async => format!("Promise<{}>", method.return_type),
                TurboMethodType::Promise => format!("Promise<{}>", method.return_type),
            };

            ts_def.push_str(&format!(
                "  {}({}): {};\n",
                method.name,
                params.join(", "),
                return_type
            ));
        }

        ts_def.push_str("}\n\n");

        // Add type definitions
        for (name, type_spec) in &self.spec.types {
            ts_def.push_str(&self.generate_typescript_type(name, type_spec));
        }

        Ok(ts_def)
    }

    // Private helper methods

    fn generate_turbo_spec() -> TurboModuleSpec {
        let methods = vec![
            TurboMethodSpec {
                name: "initialize".to_string(),
                method_type: TurboMethodType::Promise,
                parameters: vec![TurboParameter {
                    name: "config".to_string(),
                    param_type: "TurboModuleConfig".to_string(),
                    optional: true,
                    default: None,
                }],
                return_type: "TurboMethodResult".to_string(),
                optional: false,
            },
            TurboMethodSpec {
                name: "inference".to_string(),
                method_type: TurboMethodType::Promise,
                parameters: vec![TurboParameter {
                    name: "request".to_string(),
                    param_type: "InferenceRequest".to_string(),
                    optional: false,
                    default: None,
                }],
                return_type: "TurboMethodResult".to_string(),
                optional: false,
            },
            TurboMethodSpec {
                name: "getDeviceCapabilities".to_string(),
                method_type: TurboMethodType::Promise,
                parameters: vec![],
                return_type: "TurboMethodResult".to_string(),
                optional: false,
            },
            TurboMethodSpec {
                name: "getPerformanceStats".to_string(),
                method_type: TurboMethodType::Sync,
                parameters: vec![],
                return_type: "TurboMethodResult".to_string(),
                optional: false,
            },
        ];

        let mut types = HashMap::new();

        // Add core type definitions
        types.insert(
            "TurboMethodResult".to_string(),
            TurboTypeSpec {
                name: "TurboMethodResult".to_string(),
                kind: TurboTypeKind::Object,
                properties: Some(
                    [
                        (
                            "success".to_string(),
                            TurboProperty {
                                prop_type: "boolean".to_string(),
                                optional: false,
                                description: Some("Operation success flag".to_string()),
                            },
                        ),
                        (
                            "data".to_string(),
                            TurboProperty {
                                prop_type: "any".to_string(),
                                optional: true,
                                description: Some("Result data".to_string()),
                            },
                        ),
                        (
                            "error".to_string(),
                            TurboProperty {
                                prop_type: "TurboError".to_string(),
                                optional: true,
                                description: Some("Error information".to_string()),
                            },
                        ),
                        (
                            "executionTimeMs".to_string(),
                            TurboProperty {
                                prop_type: "number".to_string(),
                                optional: false,
                                description: Some("Execution time in milliseconds".to_string()),
                            },
                        ),
                    ]
                    .into(),
                ),
                element_type: None,
            },
        );

        TurboModuleSpec {
            name: "TrustformersTurboModule".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            methods,
            types,
        }
    }

    fn check_method_cache(
        &self,
        method_name: &str,
        request: &InferenceRequest,
    ) -> Option<TurboMethodResult> {
        let cache = self.method_cache.lock().expect("Failed to acquire lock");
        let cache_key = format!("{}_{}", method_name, self.generate_cache_key(request));
        cache.get(&cache_key).cloned()
    }

    fn cache_method_result(
        &self,
        method_name: &str,
        request: &InferenceRequest,
        result: &TurboMethodResult,
    ) {
        let mut cache = self.method_cache.lock().expect("Failed to acquire lock");

        // Simple cache eviction if size limit exceeded
        if cache.len() >= self.config.performance.cache_size_limit {
            cache.clear();
        }

        let cache_key = format!("{}_{}", method_name, self.generate_cache_key(request));
        cache.insert(cache_key, result.clone());
    }

    fn generate_cache_key(&self, request: &InferenceRequest) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        request.model_id.hash(&mut hasher);
        request.input_shape.hash(&mut hasher);
        request.input_data.len().hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    fn update_cache_stats(&self, cache_hit: bool) {
        let mut monitor = self.performance_monitor.lock().expect("Failed to acquire lock");
        let total_cache_ops = monitor.method_call_counts.values().sum::<usize>() as f64;

        if total_cache_ops > 0.0 {
            let cache_hits = if cache_hit { 1.0 } else { 0.0 };
            monitor.cache_hit_ratio =
                (monitor.cache_hit_ratio * (total_cache_ops - 1.0) + cache_hits) / total_cache_ops;
        }
    }

    fn update_performance_stats(&self, method_name: &str, execution_time: f64, success: bool) {
        let mut monitor = self.performance_monitor.lock().expect("Failed to acquire lock");

        *monitor.method_call_counts.entry(method_name.to_string()).or_insert(0) += 1;
        monitor
            .method_execution_times
            .entry(method_name.to_string())
            .or_default()
            .push(execution_time);

        monitor.total_operations += 1;
        if success {
            monitor.successful_operations += 1;
        }
    }

    fn generate_typescript_type(&self, name: &str, type_spec: &TurboTypeSpec) -> String {
        match type_spec.kind {
            TurboTypeKind::Object => {
                let mut ts_type = format!("export interface {} {{\n", name);

                if let Some(properties) = &type_spec.properties {
                    for (prop_name, prop) in properties {
                        ts_type.push_str(&format!(
                            "  {}{}: {};\n",
                            prop_name,
                            if prop.optional { "?" } else { "" },
                            prop.prop_type
                        ));
                    }
                }

                ts_type.push_str("}\n\n");
                ts_type
            },
            TurboTypeKind::Array => {
                if let Some(element_type) = &type_spec.element_type {
                    format!("export type {} = {}[];\n\n", name, element_type)
                } else {
                    format!("export type {} = any[];\n\n", name)
                }
            },
            TurboTypeKind::Primitive => {
                format!("export type {} = {};\n\n", name, type_spec.name)
            },
            TurboTypeKind::Union => {
                // Simplified union type generation
                format!("export type {} = string | number | boolean;\n\n", name)
            },
        }
    }
}

impl TurboPerformanceMonitor {
    fn new() -> Self {
        Self {
            method_call_counts: HashMap::new(),
            method_execution_times: HashMap::new(),
            cache_hit_ratio: 0.0,
            total_operations: 0,
            successful_operations: 0,
        }
    }

    fn get_average_execution_times(&self) -> HashMap<String, f64> {
        self.method_execution_times
            .iter()
            .map(|(method, times)| {
                let avg = times.iter().sum::<f64>() / times.len() as f64;
                (method.clone(), avg)
            })
            .collect()
    }
}

impl Default for TurboModuleConfig {
    fn default() -> Self {
        Self {
            enable_new_architecture: true,
            enable_fabric: true,
            enable_hermes_optimizations: true,
            enable_concurrent_features: true,
            js_interface: TurboJSInterfaceConfig {
                enable_auto_type_generation: true,
                enable_runtime_type_checking: false,
                type_definition_path: Some("src/types/TrustformersTurboModule.ts".to_string()),
                strict_typescript: true,
            },
            performance: TurboPerformanceConfig {
                enable_lazy_loading: true,
                enable_method_caching: true,
                cache_size_limit: 100,
                enable_async_execution: true,
                max_concurrent_ops: 4,
            },
        }
    }
}

/// Export functions for React Native Turbo Module bridge
pub mod turbo_module_exports {
    use super::*;
    use std::ffi::{CStr, CString};
    use std::os::raw::c_char;

    /// Initialize TrustformeRS Turbo Module
    #[no_mangle]
    pub extern "C" fn trustformers_turbo_initialize(config_json: *const c_char) -> *mut c_char {
        let result = std::panic::catch_unwind(|| {
            unsafe {
                let config_str = CStr::from_ptr(config_json).to_str().unwrap_or("{}");

                let turbo_config: TurboModuleConfig =
                    serde_json::from_str(config_str).unwrap_or_default();
                let rn_config = ReactNativeConfig::default();
                let mobile_config = MobileConfig::default();

                match TrustformersTurboModule::new(turbo_config, rn_config, mobile_config) {
                    Ok(module) => {
                        let module_arc = Arc::new(Mutex::new(module));
                        let _ = TURBO_MODULE.set(module_arc.clone());

                        // Initialize the module
                        let runtime = tokio::runtime::Runtime::new().expect("Failed to get value");
                        let init_result = runtime.block_on(async {
                            let module_lock = module_arc.lock().expect("Failed to acquire lock");
                            module_lock.initialize().await
                        });

                        match init_result {
                            Ok(result) => serde_json::to_string(&result).unwrap_or_default(),
                            Err(e) => {
                                let error = TurboMethodResult {
                                    success: false,
                                    data: None,
                                    error: Some(TurboError {
                                        code: "INIT_ERROR".to_string(),
                                        message: e.to_string(),
                                        stack: None,
                                        data: None,
                                    }),
                                    execution_time_ms: 0.0,
                                    metadata: HashMap::new(),
                                };
                                serde_json::to_string(&error).unwrap_or_default()
                            },
                        }
                    },
                    Err(e) => {
                        let error = TurboMethodResult {
                            success: false,
                            data: None,
                            error: Some(TurboError {
                                code: "MODULE_CREATION_ERROR".to_string(),
                                message: e.to_string(),
                                stack: None,
                                data: None,
                            }),
                            execution_time_ms: 0.0,
                            metadata: HashMap::new(),
                        };
                        serde_json::to_string(&error).unwrap_or_default()
                    },
                }
            }
        });

        match result {
            Ok(json_str) => CString::new(json_str).expect("Failed to get value").into_raw(),
            Err(_) => {
                let error = TurboMethodResult {
                    success: false,
                    data: None,
                    error: Some(TurboError {
                        code: "PANIC_ERROR".to_string(),
                        message: "Module initialization panicked".to_string(),
                        stack: None,
                        data: None,
                    }),
                    execution_time_ms: 0.0,
                    metadata: HashMap::new(),
                };
                let error_json = serde_json::to_string(&error).unwrap_or_default();
                CString::new(error_json).expect("Failed to get value").into_raw()
            },
        }
    }

    /// Perform inference through Turbo Module
    #[no_mangle]
    pub extern "C" fn trustformers_turbo_inference(request_json: *const c_char) -> *mut c_char {
        let result = std::panic::catch_unwind(|| unsafe {
            if let Some(module_arc) = TURBO_MODULE.get() {
                let request_str = CStr::from_ptr(request_json).to_str().unwrap_or("{}");

                match serde_json::from_str::<InferenceRequest>(request_str) {
                    Ok(request) => {
                        let runtime = tokio::runtime::Runtime::new().expect("Failed to get value");
                        let inference_result = runtime.block_on(async {
                            let module = module_arc.lock().expect("Failed to acquire lock");
                            module.inference(&request).await
                        });

                        match inference_result {
                            Ok(result) => serde_json::to_string(&result).unwrap_or_default(),
                            Err(e) => {
                                let error_result = TurboMethodResult {
                                    success: false,
                                    data: None,
                                    error: Some(TurboError {
                                        code: "INFERENCE_ERROR".to_string(),
                                        message: e.to_string(),
                                        stack: None,
                                        data: None,
                                    }),
                                    execution_time_ms: 0.0,
                                    metadata: HashMap::new(),
                                };
                                serde_json::to_string(&error_result).unwrap_or_default()
                            },
                        }
                    },
                    Err(e) => {
                        let error_result = TurboMethodResult {
                            success: false,
                            data: None,
                            error: Some(TurboError {
                                code: "PARSE_ERROR".to_string(),
                                message: format!("Failed to parse request: {}", e),
                                stack: None,
                                data: None,
                            }),
                            execution_time_ms: 0.0,
                            metadata: HashMap::new(),
                        };
                        serde_json::to_string(&error_result).unwrap_or_default()
                    },
                }
            } else {
                let error_result = TurboMethodResult {
                    success: false,
                    data: None,
                    error: Some(TurboError {
                        code: "MODULE_NOT_INITIALIZED".to_string(),
                        message: "Turbo Module not initialized".to_string(),
                        stack: None,
                        data: None,
                    }),
                    execution_time_ms: 0.0,
                    metadata: HashMap::new(),
                };
                serde_json::to_string(&error_result).unwrap_or_default()
            }
        });

        match result {
            Ok(json_str) => CString::new(json_str).expect("Failed to get value").into_raw(),
            Err(_) => {
                let error_result = TurboMethodResult {
                    success: false,
                    data: None,
                    error: Some(TurboError {
                        code: "PANIC_ERROR".to_string(),
                        message: "Inference panicked".to_string(),
                        stack: None,
                        data: None,
                    }),
                    execution_time_ms: 0.0,
                    metadata: HashMap::new(),
                };
                let error_json = serde_json::to_string(&error_result).unwrap_or_default();
                CString::new(error_json).expect("Failed to get value").into_raw()
            },
        }
    }

    /// Get TypeScript definitions
    #[no_mangle]
    pub extern "C" fn trustformers_turbo_get_typescript_definitions() -> *mut c_char {
        if let Some(module_arc) = TURBO_MODULE.get() {
            let module = module_arc.lock().expect("Failed to acquire lock");
            let ts_definitions = module.generate_typescript_definitions().unwrap_or_default();
            CString::new(ts_definitions).expect("Failed to get value").into_raw()
        } else {
            CString::new("// Module not initialized")
                .expect("Failed to get value")
                .into_raw()
        }
    }

    /// Free string allocated by Rust
    #[no_mangle]
    pub extern "C" fn trustformers_turbo_free_string(ptr: *mut c_char) {
        if !ptr.is_null() {
            unsafe {
                let _ = CString::from_raw(ptr);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_turbo_module_creation() {
        let turbo_config = TurboModuleConfig::default();
        let rn_config = ReactNativeConfig::default();
        let mobile_config = MobileConfig::default();

        let result = TrustformersTurboModule::new(turbo_config, rn_config, mobile_config);
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_turbo_module_initialization() {
        let turbo_config = TurboModuleConfig::default();
        let rn_config = ReactNativeConfig::default();
        let mobile_config = MobileConfig::default();

        let module = TrustformersTurboModule::new(turbo_config, rn_config, mobile_config)
            .expect("Failed to get value");
        let result = module.initialize().await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_typescript_definition_generation() {
        let turbo_config = TurboModuleConfig::default();
        let rn_config = ReactNativeConfig::default();
        let mobile_config = MobileConfig::default();

        let module = TrustformersTurboModule::new(turbo_config, rn_config, mobile_config)
            .expect("Failed to get value");
        let ts_definitions = module.generate_typescript_definitions();
        assert!(ts_definitions.is_ok());

        let definitions = ts_definitions.expect("Failed to get value");
        assert!(definitions.contains("TrustformersTurboModule"));
        assert!(definitions.contains("initialize"));
        assert!(definitions.contains("inference"));
    }

    #[test]
    fn test_turbo_module_spec() {
        let spec = TrustformersTurboModule::generate_turbo_spec();
        assert_eq!(spec.name, "TrustformersTurboModule");
        assert!(!spec.methods.is_empty());
        assert!(!spec.types.is_empty());
    }

    #[test]
    fn test_turbo_module_config_default() {
        let config = TurboModuleConfig::default();
        assert!(config.enable_new_architecture);
        assert!(config.enable_fabric);
        assert!(config.enable_hermes_optimizations);
        assert!(config.performance.enable_method_caching);
    }
}
