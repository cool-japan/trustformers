//! JNI Integration for Android Applications
//!
//! This module provides Java Native Interface (JNI) bindings for integrating
//! TrustformeRS with Android applications written in Java/Kotlin.

use crate::MobileConfig;
use trustformers_core::{Tensor};
use trustformers_core::error::{CoreError, Result};

#[cfg(target_os = "android")]
use jni::{
    objects::{JByteArray, JClass, JObject, JString},
    sys::{jboolean, jbyteArray, jlong, jstring},
    JNIEnv, JavaVM,
};

use super::{engine::AndroidInferenceEngine, types::AndroidDeviceInfo};

/// JNI function to create Android inference engine
#[cfg(target_os = "android")]
#[no_mangle]
pub extern "system" fn Java_com_trustformers_TrustformersEngine_createEngine(
    env: JNIEnv,
    _class: JClass,
    config_json: JString,
) -> jlong {
    let config_str: String = match env.get_string(config_json) {
        Ok(s) => s.into(),
        Err(e) => {
            tracing::error!("Failed to get config string: {:?}", e);
            return 0;
        }
    };

    let config: MobileConfig = match serde_json::from_str(&config_str) {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("Failed to parse config JSON: {:?}", e);
            return 0;
        }
    };

    match AndroidInferenceEngine::new(config) {
        Ok(mut engine) => {
            // Get JVM reference for later use
            if let Ok(jvm) = env.get_java_vm() {
                engine.init_jvm(jvm);
            }
            Box::into_raw(Box::new(engine)) as jlong
        }
        Err(e) => {
            tracing::error!("Failed to create inference engine: {:?}", e);
            0
        }
    }
}

/// JNI function to load model
#[cfg(target_os = "android")]
#[no_mangle]
pub extern "system" fn Java_com_trustformers_TrustformersEngine_loadModel(
    env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
    model_path: JString,
) -> jboolean {
    if engine_ptr == 0 {
        tracing::error!("Engine pointer is null");
        return false as jboolean;
    }

    let engine = unsafe { &mut *(engine_ptr as *mut AndroidInferenceEngine) };
    let path_str: String = match env.get_string(model_path) {
        Ok(s) => s.into(),
        Err(e) => {
            tracing::error!("Failed to get model path string: {:?}", e);
            return false as jboolean;
        }
    };

    match engine.load_model(&path_str) {
        Ok(_) => {
            tracing::info!("Model loaded successfully: {}", path_str);
            true as jboolean
        }
        Err(e) => {
            tracing::error!("Failed to load model: {:?}", e);
            false as jboolean
        }
    }
}

/// JNI function to perform inference
#[cfg(target_os = "android")]
#[no_mangle]
pub extern "system" fn Java_com_trustformers_TrustformersEngine_inference(
    env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
    input_data: jbyteArray,
) -> jbyteArray {
    if engine_ptr == 0 {
        tracing::error!("Engine pointer is null");
        return JObject::null().into_inner();
    }

    let engine = unsafe { &mut *(engine_ptr as *mut AndroidInferenceEngine) };

    // Convert JByteArray to Rust Vec<u8>
    let input_bytes = match env.convert_byte_array(input_data) {
        Ok(bytes) => bytes,
        Err(e) => {
            tracing::error!("Failed to convert input byte array: {:?}", e);
            return JObject::null().into_inner();
        }
    };

    // Convert bytes to tensor (assuming f32 data)
    let input_floats: Vec<f32> = input_bytes
        .chunks(4)
        .map(|chunk| {
            if chunk.len() == 4 {
                let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                f32::from_le_bytes(bytes)
            } else {
                0.0 // Handle incomplete chunks
            }
        })
        .collect();

    let input_tensor = match Tensor::from_vec(input_floats, &[input_bytes.len() / 4]) {
        Ok(t) => t,
        Err(e) => {
            tracing::error!("Failed to create input tensor: {:?}", e);
            return JObject::null().into_inner();
        }
    };

    // Perform inference
    let output_tensor = match engine.inference(&input_tensor) {
        Ok(t) => t,
        Err(e) => {
            tracing::error!("Inference failed: {:?}", e);
            return JObject::null().into_inner();
        }
    };

    // Convert output tensor back to bytes
    let output_data = match output_tensor.as_slice::<f32>() {
        Ok(data) => data,
        Err(e) => {
            tracing::error!("Failed to get output tensor data: {:?}", e);
            return JObject::null().into_inner();
        }
    };

    let output_bytes: Vec<u8> = output_data
        .iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();

    match env.byte_array_from_slice(&output_bytes) {
        Ok(array) => array,
        Err(e) => {
            tracing::error!("Failed to create output byte array: {:?}", e);
            JObject::null().into_inner()
        }
    }
}

/// JNI function to get device information
#[cfg(target_os = "android")]
#[no_mangle]
pub extern "system" fn Java_com_trustformers_TrustformersEngine_getDeviceInfo(
    env: JNIEnv,
    _class: JClass,
) -> jstring {
    let device_info = AndroidDeviceInfo::detect();
    let json = match serde_json::to_string(&device_info) {
        Ok(j) => j,
        Err(e) => {
            tracing::error!("Failed to serialize device info: {:?}", e);
            return JObject::null().into_inner();
        }
    };

    match env.new_string(json) {
        Ok(jstr) => jstr.into_inner(),
        Err(e) => {
            tracing::error!("Failed to create Java string: {:?}", e);
            JObject::null().into_inner()
        }
    }
}

/// JNI function to get inference statistics
#[cfg(target_os = "android")]
#[no_mangle]
pub extern "system" fn Java_com_trustformers_TrustformersEngine_getStats(
    env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
) -> jstring {
    if engine_ptr == 0 {
        tracing::error!("Engine pointer is null");
        return JObject::null().into_inner();
    }

    let engine = unsafe { &*(engine_ptr as *const AndroidInferenceEngine) };
    let stats = engine.get_stats();

    let json = match serde_json::to_string(stats) {
        Ok(j) => j,
        Err(e) => {
            tracing::error!("Failed to serialize stats: {:?}", e);
            return JObject::null().into_inner();
        }
    };

    match env.new_string(json) {
        Ok(jstr) => jstr.into_inner(),
        Err(e) => {
            tracing::error!("Failed to create Java string: {:?}", e);
            JObject::null().into_inner()
        }
    }
}

/// JNI function to check hardware capabilities
#[cfg(target_os = "android")]
#[no_mangle]
pub extern "system" fn Java_com_trustformers_TrustformersEngine_checkCapabilities(
    env: JNIEnv,
    _class: JClass,
) -> jstring {
    let device_info = AndroidDeviceInfo::detect();

    let capabilities = serde_json::json!({
        "nnapi_available": device_info.nnapi_info.is_some(),
        "vulkan_supported": device_info.gpu_info.vulkan_supported,
        "opengl_es_version": device_info.gpu_info.opengl_es_version,
        "total_memory_mb": device_info.total_memory_mb,
        "cpu_cores": device_info.cpu_cores,
        "performance_class": device_info.performance_class,
        "is_flagship": device_info.is_flagship_device(),
        "is_ml_capable": device_info.is_ml_capable(),
        "hardware_acceleration": AndroidInferenceEngine::has_hardware_acceleration(),
    });

    let json = match serde_json::to_string(&capabilities) {
        Ok(j) => j,
        Err(e) => {
            tracing::error!("Failed to serialize capabilities: {:?}", e);
            return JObject::null().into_inner();
        }
    };

    match env.new_string(json) {
        Ok(jstr) => jstr.into_inner(),
        Err(e) => {
            tracing::error!("Failed to create Java string: {:?}", e);
            JObject::null().into_inner()
        }
    }
}

/// JNI function to release engine resources
#[cfg(target_os = "android")]
#[no_mangle]
pub extern "system" fn Java_com_trustformers_TrustformersEngine_releaseEngine(
    _env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
) {
    if engine_ptr != 0 {
        unsafe {
            let _engine = Box::from_raw(engine_ptr as *mut AndroidInferenceEngine);
            // Engine will be dropped automatically
        }
        tracing::info!("Engine resources released");
    }
}

/// Helper function to create Java exception
#[cfg(target_os = "android")]
fn throw_runtime_exception(env: &JNIEnv, message: &str) {
    if let Err(e) = env.throw_new("java/lang/RuntimeException", message) {
        tracing::error!("Failed to throw Java exception: {:?}", e);
    }
}

/// JNI function to set log level
#[cfg(target_os = "android")]
#[no_mangle]
pub extern "system" fn Java_com_trustformers_TrustformersEngine_setLogLevel(
    env: JNIEnv,
    _class: JClass,
    level: jlong,
) {
    let log_level = match level {
        0 => tracing::Level::ERROR,
        1 => tracing::Level::WARN,
        2 => tracing::Level::INFO,
        3 => tracing::Level::DEBUG,
        4 => tracing::Level::TRACE,
        _ => tracing::Level::INFO,
    };

    // In practice, would configure tracing subscriber
    tracing::info!("Log level set to: {:?}", log_level);
}

/// Utility functions for JNI integration
pub mod utils {
    use super::*;

    /// Convert Rust Result to JNI boolean result
    pub fn result_to_jboolean(result: Result<()>) -> jboolean {
        match result {
            Ok(_) => true as jboolean,
            Err(e) => {
                tracing::error!("Operation failed: {:?}", e);
                false as jboolean
            }
        }
    }

    /// Convert tensor to byte array for JNI
    pub fn tensor_to_bytes(tensor: &Tensor) -> Result<Vec<u8>> {
        let data = tensor.as_slice::<f32>()?;
        let bytes: Vec<u8> = data
            .iter()
            .flat_map(|&f| f.to_le_bytes().to_vec())
            .collect();
        Ok(bytes)
    }

    /// Convert byte array to tensor
    pub fn bytes_to_tensor(bytes: &[u8], shape: &[usize]) -> Result<Tensor> {
        let floats: Vec<f32> = bytes
            .chunks(4)
            .map(|chunk| {
                if chunk.len() == 4 {
                    let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    f32::from_le_bytes(bytes)
                } else {
                    0.0
                }
            })
            .collect();

        Tensor::from_vec(floats, shape)
    }
}

// Provide stub implementations for non-Android platforms
#[cfg(not(target_os = "android"))]
pub mod stubs {
    use super::*;

    pub fn initialize_jni() -> Result<()> {
        Err(TrustformersError::runtime_error(
            "JNI is only available on Android".into(),
        ))
    }

    pub fn create_engine_from_java(_config_json: &str) -> Result<*mut AndroidInferenceEngine> {
        Err(TrustformersError::runtime_error(
            "JNI is only available on Android".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_conversion() {
        let original_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let bytes = utils::tensor_to_bytes(&Tensor::from_vec(original_data.clone(), &[4]).unwrap()).unwrap();
        let reconstructed = utils::bytes_to_tensor(&bytes, &[4]).unwrap();
        let reconstructed_data = reconstructed.as_slice::<f32>().unwrap();

        assert_eq!(original_data, reconstructed_data);
    }

    #[test]
    fn test_result_conversion() {
        assert_eq!(utils::result_to_jboolean(Ok(())), true as jboolean);
        assert_eq!(
            utils::result_to_jboolean(Err(TrustformersError::runtime_error("test".into()).into())),
            false as jboolean
        );
    }

    #[cfg(not(target_os = "android"))]
    #[test]
    fn test_stubs() {
        assert!(stubs::initialize_jni().is_err().into());
        assert!(stubs::create_engine_from_java("{}").is_err());
    }
}
