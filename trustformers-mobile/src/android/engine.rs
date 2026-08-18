//! Android Inference Engine Implementation
//!
//! This module contains the main AndroidInferenceEngine that orchestrates
//! NNAPI, GPU, and CPU inference backends for Android devices.

use crate::android::device_info::{AndroidDeviceInfo, NNAPIDeviceInfo, NNAPIHardwareDevice};
use crate::android::gpu::{AndroidGPUBackend, AndroidGPUComputeState};
use crate::android::nnapi::{
    ANeuralNetworksPerformanceInfo, ANeuralNetworks_getDevice, ANeuralNetworks_getDeviceCount,
    ANeuralNetworks_getDeviceFeatureLevel, ANeuralNetworks_getDeviceName,
    ANeuralNetworks_getDevicePerformanceInfo, ANeuralNetworks_getDeviceType, NNAPIModel,
    ANEURALNETWORKS_DEVICE_ACCELERATOR, ANEURALNETWORKS_DEVICE_CPU, ANEURALNETWORKS_DEVICE_GPU,
    ANEURALNETWORKS_NO_ERROR,
};
use crate::{MobileBackend, MobileConfig, MobilePlatform, MobileStats};
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use trustformers_core::error::{CoreError, Result};
use trustformers_core::{Tensor, TrustformersError};

#[cfg(target_os = "android")]
use jni::JavaVM;

/// Android-specific inference engine
pub struct AndroidInferenceEngine {
    config: MobileConfig,
    stats: MobileStats,
    model_loaded: bool,
    /// The real inference engine backing [`Self::cpu_inference`]. Populated
    /// by [`Self::load_cpu_model`], which parses the checkpoint at
    /// `model_path` through the same real safetensors/PyTorch/ONNX parsers
    /// [`crate::inference::MobileInferenceEngine`] uses everywhere else in
    /// this crate -- no separate, second "Android CPU" weight loader exists,
    /// and none should: a checkpoint's bytes mean the same thing regardless
    /// of which backend enum variant selected the CPU path.
    cpu_engine: Option<crate::inference::MobileInferenceEngine>,
    #[cfg(target_os = "android")]
    nnapi_model: Option<NNAPIModel>,
    #[cfg(target_os = "android")]
    jvm: Option<JavaVM>,
    #[cfg(target_os = "android")]
    gpu_state: Option<AndroidGPUComputeState>,
}

impl AndroidInferenceEngine {
    /// Create new Android inference engine
    pub fn new(config: MobileConfig) -> Result<Self> {
        if config.platform != MobilePlatform::Android {
            return Err(CoreError::ConfigError {
                message: "Android inference engine requires Android platform configuration"
                    .to_string(),
                context: trustformers_core::error::ErrorContext::new(
                    trustformers_core::error::ErrorCode::E4001,
                    "new".to_string(),
                ),
            });
        }

        let stats = MobileStats::new(&config);

        Ok(Self {
            config,
            stats,
            model_loaded: false,
            cpu_engine: None,
            #[cfg(target_os = "android")]
            nnapi_model: None,
            #[cfg(target_os = "android")]
            jvm: None,
            #[cfg(target_os = "android")]
            gpu_state: None,
        })
    }

    /// Initialize with JVM reference for JNI integration
    #[cfg(target_os = "android")]
    pub fn init_jvm(&mut self, jvm: JavaVM) {
        self.jvm = Some(jvm);
    }

    /// Load model for Android inference
    pub fn load_model(&mut self, model_path: &str) -> Result<()> {
        match self.config.backend {
            MobileBackend::NNAPI => self.load_nnapi_model(model_path),
            MobileBackend::CPU => self.load_cpu_model(model_path),
            MobileBackend::GPU => self.load_gpu_model(model_path),
            _ => Err(TrustformersError::runtime_error(format!(
                "Backend {:?} not supported on Android",
                self.config.backend
            ))
            .into()),
        }
    }

    /// Perform inference using Android optimizations
    pub fn inference(&mut self, input: &Tensor) -> Result<Tensor> {
        if !self.model_loaded {
            return Err(TrustformersError::runtime_error("Model not loaded".into()).into());
        }

        let start_time = std::time::Instant::now();

        let result = match self.config.backend {
            MobileBackend::NNAPI => self.nnapi_inference(input),
            MobileBackend::CPU => self.cpu_inference(input),
            MobileBackend::GPU => self.gpu_inference(input),
            _ => Err(TrustformersError::runtime_error("Unsupported backend".into()).into()),
        };

        let inference_time = start_time.elapsed().as_millis() as f32;
        self.stats.update_inference(inference_time);

        result
    }

    /// Get current performance statistics
    pub fn get_stats(&self) -> &MobileStats {
        &self.stats
    }

    /// Update configuration
    pub fn update_config(&mut self, config: MobileConfig) -> Result<()> {
        if config.platform != MobilePlatform::Android {
            return Err(CoreError::ConfigError {
                message: "Android inference engine requires Android platform configuration"
                    .to_string(),
                context: trustformers_core::error::ErrorContext::new(
                    trustformers_core::error::ErrorCode::E4001,
                    "update_config".to_string(),
                ),
            });
        }

        self.config = config;
        self.stats = MobileStats::new(&self.config);
        Ok(())
    }

    /// Check Android device capabilities
    pub fn check_device_capabilities() -> AndroidDeviceInfo {
        AndroidDeviceInfo::detect()
    }

    /// Detect available NNAPI hardware acceleration devices
    pub fn detect_nnapi_devices() -> Vec<NNAPIDeviceInfo> {
        #[cfg(target_os = "android")]
        {
            Self::detect_nnapi_devices_impl()
        }

        #[cfg(not(target_os = "android"))]
        {
            Vec::new()
        }
    }

    #[cfg(target_os = "android")]
    fn detect_nnapi_devices_impl() -> Vec<NNAPIDeviceInfo> {
        let mut devices = Vec::new();
        let mut device_count: u32 = 0;

        // Get number of available NNAPI devices
        let result = unsafe { ANeuralNetworks_getDeviceCount(&mut device_count) };
        if result != ANEURALNETWORKS_NO_ERROR {
            tracing::warn!("Failed to get NNAPI device count: {}", result);
            return devices;
        }

        tracing::info!("Found {} NNAPI devices", device_count);

        // Query each device
        for device_index in 0..device_count {
            if let Some(device_info) = Self::query_nnapi_device(device_index) {
                devices.push(device_info);
            }
        }

        devices
    }

    #[cfg(target_os = "android")]
    fn query_nnapi_device(device_index: u32) -> Option<NNAPIDeviceInfo> {
        let mut device_ptr: *mut c_void = std::ptr::null_mut();
        let result = unsafe { ANeuralNetworks_getDevice(device_index, &mut device_ptr) };

        if result != ANEURALNETWORKS_NO_ERROR || device_ptr.is_null() {
            tracing::warn!("Failed to get NNAPI device {}: {}", device_index, result);
            return None;
        }

        // Get device name
        let name = {
            let mut name_ptr: *const c_char = std::ptr::null();
            let result = unsafe { ANeuralNetworks_getDeviceName(device_ptr, &mut name_ptr) };
            if result == ANEURALNETWORKS_NO_ERROR && !name_ptr.is_null() {
                unsafe { CStr::from_ptr(name_ptr) }.to_string_lossy().into_owned()
            } else {
                format!("Device {}", device_index)
            }
        };

        // Get device type
        let mut device_type: i32 = 0;
        let result = unsafe { ANeuralNetworks_getDeviceType(device_ptr, &mut device_type) };
        if result != ANEURALNETWORKS_NO_ERROR {
            device_type = ANEURALNETWORKS_DEVICE_CPU;
        }

        // Get feature level
        let mut feature_level: i32 = 27;
        let result =
            unsafe { ANeuralNetworks_getDeviceFeatureLevel(device_ptr, &mut feature_level) };
        if result != ANEURALNETWORKS_NO_ERROR {
            feature_level = 27; // Default to API level 27
        }

        // Get performance info
        let mut performance_info = ANeuralNetworksPerformanceInfo {
            exec_time: 1.0,
            power_usage: 1.0,
        };
        let result =
            unsafe { ANeuralNetworks_getDevicePerformanceInfo(device_ptr, &mut performance_info) };
        if result != ANEURALNETWORKS_NO_ERROR {
            performance_info = ANeuralNetworksPerformanceInfo {
                exec_time: 1.0,
                power_usage: 1.0,
            };
        }

        Some(NNAPIDeviceInfo {
            index: device_index,
            device_ptr,
            name,
            device_type,
            feature_level,
            performance_info,
            vendor_extensions: Vec::new(), // Would query extensions in practice
        })
    }

    /// Get best NNAPI device for inference
    pub fn get_best_nnapi_device() -> Option<NNAPIDeviceInfo> {
        let devices = Self::detect_nnapi_devices();
        if devices.is_empty() {
            return None;
        }

        // Prefer GPU/Accelerator over CPU, and lower execution time
        let best_device = devices.into_iter().min_by(|a, b| {
            // First compare by device type (GPU/Accelerator preferred)
            let type_order_a = match a.device_type {
                t if t == ANEURALNETWORKS_DEVICE_ACCELERATOR => 0,
                t if t == ANEURALNETWORKS_DEVICE_GPU => 1,
                _ => 2,
            };
            let type_order_b = match b.device_type {
                t if t == ANEURALNETWORKS_DEVICE_ACCELERATOR => 0,
                t if t == ANEURALNETWORKS_DEVICE_GPU => 1,
                _ => 2,
            };

            type_order_a.cmp(&type_order_b).then_with(|| {
                a.performance_info
                    .exec_time
                    .partial_cmp(&b.performance_info.exec_time)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });

        best_device
    }

    /// Convert device type to string
    pub fn device_type_to_string(device_type: i32) -> &'static str {
        match device_type {
            t if t == ANEURALNETWORKS_DEVICE_CPU => "CPU",
            t if t == ANEURALNETWORKS_DEVICE_GPU => "GPU",
            t if t == ANEURALNETWORKS_DEVICE_ACCELERATOR => "Accelerator",
            _ => "Unknown",
        }
    }

    // Backend-specific implementations
    fn load_nnapi_model(&mut self, model_path: &str) -> Result<()> {
        #[cfg(target_os = "android")]
        {
            let model = NNAPIModel::new()?;
            self.nnapi_model = Some(model);
            self.model_loaded = true;
            tracing::info!("Loaded NNAPI model from: {}", model_path);
            Ok(())
        }

        #[cfg(not(target_os = "android"))]
        {
            tracing::warn!("NNAPI not available on non-Android platforms");
            Err(TrustformersError::runtime_error("NNAPI not available".into()).into())
        }
    }

    /// Load `model_path` (safetensors, PyTorch `.bin`/`.pt`/`.pth`, or ONNX
    /// -- auto-detected from the file's own bytes/extension) through the
    /// real [`crate::inference::MobileInferenceEngine`], the same real
    /// parsers used everywhere else in this crate. A checkpoint that fails
    /// to parse, or that parses to zero usable tensors, is a hard error --
    /// `model_loaded` is only ever set once real weights are actually held.
    fn load_cpu_model(&mut self, model_path: &str) -> Result<()> {
        let mut engine = crate::inference::MobileInferenceEngine::new(self.config.clone())
            .map_err(|e| {
                TrustformersError::runtime_error(format!(
                    "failed to construct the CPU inference engine: {e}"
                ))
            })?;
        engine.load_model_from_file(model_path).map_err(|e| {
            TrustformersError::runtime_error(format!(
                "failed to load CPU model from '{model_path}': {e}"
            ))
        })?;

        self.cpu_engine = Some(engine);
        self.model_loaded = true;
        tracing::info!("Loaded CPU model from: {}", model_path);
        Ok(())
    }

    fn load_gpu_model(&mut self, model_path: &str) -> Result<()> {
        #[cfg(target_os = "android")]
        {
            let gpu_state = AndroidGPUComputeState::new(AndroidGPUBackend::Vulkan)?;
            self.gpu_state = Some(gpu_state);
            self.model_loaded = true;
            tracing::info!("Loaded GPU model from: {}", model_path);
            Ok(())
        }

        #[cfg(not(target_os = "android"))]
        {
            tracing::warn!("GPU compute not available on non-Android platforms");
            Err(TrustformersError::runtime_error("GPU compute not available".into()).into())
        }
    }

    /// NNAPI inference: real execution requires building an
    /// `ANeuralNetworksModel` operand graph from the loaded model's tensor
    /// ops, feeding it through `ANeuralNetworksExecution`, and reading the
    /// result back -- the full graph-compilation half of the NNAPI C API
    /// [`crate::nnapi::NNAPIEngine`]'s `execute` also does not yet implement
    /// (its `create_nnapi_model`/`compile_model`/`execute_inference` are all
    /// placeholder handles returning `Ok(1)`/`Ok(())`, see that module).
    /// Building that graph compiler is real, substantial work this pass
    /// does not fabricate a shortcut for; what changes here is that a call
    /// with no such compiler wired in reports it honestly -- a structured
    /// `NotImplemented` error -- rather than returning a `[1, 1000]` tensor
    /// of `0.5`s that looks like a completed accelerator inference.
    fn nnapi_inference(&mut self, _input: &Tensor) -> Result<Tensor> {
        #[cfg(target_os = "android")]
        {
            if self.nnapi_model.is_none() {
                return Err(
                    TrustformersError::runtime_error("NNAPI model not loaded".into()).into(),
                );
            }
            tracing::warn!(
                "NNAPI inference was requested but this engine has no operand-graph compiler \
                 (ANeuralNetworksModel construction from a loaded tensor set is not yet \
                 implemented); refusing to fabricate a result -- use MobileBackend::CPU instead"
            );
            Err(TrustformersError::not_implemented(
                "NNAPI operand-graph compilation and execution".to_string(),
            )
            .into())
        }

        #[cfg(not(target_os = "android"))]
        {
            Err(TrustformersError::runtime_error(
                "NNAPI not available: this is not an Android build".into(),
            )
            .into())
        }
    }

    /// Run `input` through the real weights [`Self::load_cpu_model`] loaded
    /// -- real matmul/activation dispatch via
    /// [`crate::inference::MobileInferenceEngine::inference`], not an
    /// elementwise placeholder formula.
    fn cpu_inference(&mut self, input: &Tensor) -> Result<Tensor> {
        tracing::debug!("Performing CPU inference");
        let engine = self.cpu_engine.as_mut().ok_or_else(|| {
            TrustformersError::runtime_error(
                "CPU model not loaded (load_cpu_model must succeed before cpu_inference runs)"
                    .into(),
            )
        })?;
        engine.inference(input).map_err(|e| {
            TrustformersError::runtime_error(format!("CPU inference failed: {e}")).into()
        })
    }

    /// GPU (Vulkan/OpenGL ES compute) inference: real execution requires
    /// real shader/compute-pipeline dispatch against `self.gpu_state`'s
    /// `VkDevice`/`VkCommandBuffer` (or the EGL compute-shader equivalent),
    /// which is not yet implemented -- `AndroidGPUComputeState` (see
    /// `android::gpu`) currently only tracks handles, with no kernel
    /// upload/dispatch/readback path behind them. As with
    /// [`Self::nnapi_inference`], a per-element formula standing in for that
    /// dispatch would be indistinguishable from real accelerator output
    /// while computing nothing accelerator-specific at all; this reports
    /// the gap honestly instead.
    fn gpu_inference(&mut self, _input: &Tensor) -> Result<Tensor> {
        #[cfg(target_os = "android")]
        {
            if self.gpu_state.is_none() {
                return Err(
                    TrustformersError::runtime_error("GPU state not initialized".into()).into(),
                );
            }
            tracing::warn!(
                "GPU inference was requested but this engine has no compute-shader dispatch \
                 path (Vulkan/OpenGL ES kernel upload and execution against \
                 AndroidGPUComputeState is not yet implemented); refusing to fabricate a \
                 result -- use MobileBackend::CPU instead"
            );
            Err(
                TrustformersError::not_implemented("Android GPU compute dispatch".to_string())
                    .into(),
            )
        }

        #[cfg(not(target_os = "android"))]
        {
            Err(TrustformersError::runtime_error(
                "GPU compute not available: this is not an Android build".into(),
            )
            .into())
        }
    }

    /// Check if model is loaded
    pub fn is_model_loaded(&self) -> bool {
        self.model_loaded
    }

    /// Get configuration
    pub fn get_config(&self) -> &MobileConfig {
        &self.config
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = MobileStats::new(&self.config);
    }
}

impl Drop for AndroidInferenceEngine {
    fn drop(&mut self) {
        #[cfg(target_os = "android")]
        if let Some(ref _model) = self.nnapi_model {
            tracing::debug!("Cleaning up NNAPI model resources");
        }

        #[cfg(target_os = "android")]
        if let Some(ref mut gpu_state) = self.gpu_state {
            gpu_state.cleanup();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_android_inference_engine_creation() {
        let config = MobileConfig::android_optimized();
        let engine = AndroidInferenceEngine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_wrong_platform_config() {
        let mut config = MobileConfig::android_optimized();
        config.platform = MobilePlatform::iOS;
        let engine = AndroidInferenceEngine::new(config);
        assert!(engine.is_err());
    }

    #[test]
    fn test_device_type_string_conversion() {
        assert_eq!(
            AndroidInferenceEngine::device_type_to_string(ANEURALNETWORKS_DEVICE_CPU),
            "CPU"
        );
        assert_eq!(
            AndroidInferenceEngine::device_type_to_string(ANEURALNETWORKS_DEVICE_GPU),
            "GPU"
        );
        assert_eq!(
            AndroidInferenceEngine::device_type_to_string(ANEURALNETWORKS_DEVICE_ACCELERATOR),
            "Accelerator"
        );
        assert_eq!(AndroidInferenceEngine::device_type_to_string(-1), "Unknown");
    }

    #[test]
    fn test_engine_state_management() {
        let config = MobileConfig::android_optimized();
        let mut engine = AndroidInferenceEngine::new(config).expect("Operation failed");

        assert!(!engine.is_model_loaded());
        assert_eq!(engine.get_config().platform, MobilePlatform::Android);

        // Test stats reset
        engine.reset_stats();
        assert!(engine.get_stats().total_inferences == 0);
    }

    /// Write a minimal, real, single-linear-layer safetensors checkpoint
    /// (a `[4, 4]` `"linear.weight"` -- deliberately square so
    /// `MobileInferenceEngine::process_layer`'s orientation tie-break is
    /// unambiguous) to a fresh file under `std::env::temp_dir()` and return
    /// its path.
    fn write_test_safetensors_checkpoint() -> std::path::PathBuf {
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;

        // 4x4 identity matrix: `input @ identity == input`, so the expected
        // inference output is exactly known without hand-deriving a matmul.
        let mut identity = vec![0.0f32; 16];
        for i in 0..4 {
            identity[i * 4 + i] = 1.0;
        }
        let raw: Vec<u8> = identity.iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = TensorView::new(Dtype::F32, vec![4, 4], &raw).expect("valid tensor view");
        let mut tensors: HashMap<String, TensorView> = HashMap::new();
        tensors.insert("linear.weight".to_string(), view);
        let bytes = safetensors::serialize(&tensors, None).expect("serialize safetensors");

        let path = std::env::temp_dir().join(format!(
            "trustformers_android_engine_test_{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, &bytes).expect("write test checkpoint");
        path
    }

    /// Regression test for the P0 finding: `load_cpu_model` used to accept
    /// any path string (even `"test_model.tflite"`, which is never read)
    /// and unconditionally set `model_loaded = true`. A checkpoint that
    /// does not exist must now fail to load.
    #[test]
    fn test_cpu_model_loading_rejects_missing_file() {
        let config = MobileConfig::android_optimized();
        let mut engine = AndroidInferenceEngine::new(config).expect("Operation failed");

        let result = engine.load_cpu_model("this_file_does_not_exist.safetensors");
        assert!(
            result.is_err(),
            "a nonexistent checkpoint path must fail to load, not silently report success"
        );
        assert!(!engine.is_model_loaded());
    }

    #[test]
    fn test_cpu_model_loading_real_checkpoint() {
        let config = MobileConfig::android_optimized();
        let mut engine = AndroidInferenceEngine::new(config).expect("Operation failed");
        let path = write_test_safetensors_checkpoint();

        let result = engine.load_cpu_model(path.to_str().expect("utf8 path"));
        let _ = std::fs::remove_file(&path);

        assert!(result.is_ok(), "real checkpoint must load: {result:?}");
        assert!(engine.is_model_loaded());
    }

    /// Regression test for the P0 finding: `cpu_inference` used to compute
    /// `input * 0.5` regardless of what (if anything) was "loaded" -- no
    /// model weights were ever consulted. Loading a real `[4, 4]` identity
    /// matrix and running inference must now reproduce the input exactly
    /// (an identity matmul), which the old `x * 0.5` formula would not.
    #[test]
    fn test_cpu_inference_runs_real_matmul_against_loaded_weights() {
        let config = MobileConfig::android_optimized();
        let mut engine = AndroidInferenceEngine::new(config).expect("Operation failed");
        let path = write_test_safetensors_checkpoint();
        engine
            .load_cpu_model(path.to_str().expect("utf8 path"))
            .expect("load real checkpoint");
        let _ = std::fs::remove_file(&path);

        let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input_tensor = Tensor::from_vec(input_data.clone(), &[4]).expect("Operation failed");

        let result = engine.cpu_inference(&input_tensor);
        assert!(result.is_ok(), "cpu_inference failed: {result:?}");

        let output = result.expect("Operation failed");
        let output_data = output.data().expect("output tensor data");

        // `input @ identity == input` -- the old `x * 0.5` fake would have
        // produced [0.5, 1.0, 1.5, 2.0] instead.
        assert_eq!(output_data, input_data);
    }

    /// `cpu_inference` before any model is loaded must error, not run a
    /// formula against uninitialised state.
    #[test]
    fn test_cpu_inference_without_loaded_model_errors() {
        let config = MobileConfig::android_optimized();
        let mut engine = AndroidInferenceEngine::new(config).expect("Operation failed");

        let input_tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).expect("input");
        let result = engine.cpu_inference(&input_tensor);
        assert!(result.is_err());
    }

    #[test]
    fn test_config_update() {
        let config = MobileConfig::android_optimized();
        let mut engine = AndroidInferenceEngine::new(config).expect("Operation failed");

        let mut new_config = MobileConfig::android_optimized();
        new_config.max_memory_mb = 2048;

        let result = engine.update_config(new_config);
        assert!(result.is_ok());
        assert_eq!(engine.get_config().max_memory_mb, 2048);
    }

    #[test]
    fn test_inference_without_model() {
        let config = MobileConfig::android_optimized();
        let mut engine = AndroidInferenceEngine::new(config).expect("Operation failed");

        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input_tensor = Tensor::from_vec(input_data, &[4]).expect("Operation failed");

        let result = engine.inference(&input_tensor);
        assert!(result.is_err());
    }
}
// scope-probe: 1787021521
