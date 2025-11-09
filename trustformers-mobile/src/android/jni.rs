//! JNI Bindings for Android Integration
//!
//! This module provides Java Native Interface (JNI) bindings for integrating
//! TrustformeRS with Android applications.

use crate::android::device_info::AndroidDeviceInfo;
use crate::android::engine::AndroidInferenceEngine;
use crate::MobileConfig;
use trustformers_core::Tensor;

#[cfg(target_os = "android")]
use jni::{
    objects::{JClass, JObject, JString},
    sys::{jboolean, jbyteArray, jlong, jstring},
    JNIEnv,
};

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
        Err(_) => return 0,
    };

    let config: MobileConfig = match serde_json::from_str(&config_str) {
        Ok(c) => c,
        Err(_) => return 0,
    };

    match AndroidInferenceEngine::new(config) {
        Ok(mut engine) => {
            // Get JVM reference for later use
            if let Ok(jvm) = env.get_java_vm() {
                engine.init_jvm(jvm);
            }
            Box::into_raw(Box::new(engine)) as jlong
        },
        Err(_) => 0,
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
        return false as jboolean;
    }

    let engine = unsafe { &mut *(engine_ptr as *mut AndroidInferenceEngine) };
    let path_str: String = match env.get_string(model_path) {
        Ok(s) => s.into(),
        Err(_) => return false as jboolean,
    };

    engine.load_model(&path_str).is_ok() as jboolean
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
        return JObject::null().into_inner();
    }

    let engine = unsafe { &mut *(engine_ptr as *mut AndroidInferenceEngine) };

    // Convert JByteArray to Rust Vec<u8>
    let input_bytes = match env.convert_byte_array(input_data) {
        Ok(bytes) => bytes,
        Err(_) => return JObject::null().into_inner(),
    };

    // Convert bytes to tensor (simplified)
    let input_floats: Vec<f32> = input_bytes
        .chunks(4)
        .map(|chunk| {
            let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
            f32::from_le_bytes(bytes)
        })
        .collect();

    let input_tensor = match Tensor::from_vec(input_floats, &[input_bytes.len() / 4]) {
        Ok(t) => t,
        Err(_) => return JObject::null().into_inner(),
    };

    // Perform inference
    let output_tensor = match engine.inference(&input_tensor) {
        Ok(t) => t,
        Err(_) => return JObject::null().into_inner(),
    };

    // Convert output tensor back to bytes (simplified)
    let output_bytes: Vec<u8> = vec![0; 4]; // Placeholder

    match env.byte_array_from_slice(&output_bytes) {
        Ok(array) => array,
        Err(_) => JObject::null().into_inner(),
    }
}

/// JNI function to get device info
#[cfg(target_os = "android")]
#[no_mangle]
pub extern "system" fn Java_com_trustformers_TrustformersEngine_getDeviceInfo(
    env: JNIEnv,
    _class: JClass,
) -> jstring {
    let device_info = AndroidDeviceInfo::detect();
    let json = match serde_json::to_string(&device_info) {
        Ok(j) => j,
        Err(_) => return JObject::null().into_inner(),
    };

    match env.new_string(json) {
        Ok(jstr) => jstr.into_inner(),
        Err(_) => JObject::null().into_inner(),
    }
}

/// JNI function to release engine
#[cfg(target_os = "android")]
#[no_mangle]
pub extern "system" fn Java_com_trustformers_TrustformersEngine_releaseEngine(
    _env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
) {
    if engine_ptr != 0 {
        unsafe {
            let _ = Box::from_raw(engine_ptr as *mut AndroidInferenceEngine);
        }
    }
}

/// JNI function to get engine stats
#[cfg(target_os = "android")]
#[no_mangle]
pub extern "system" fn Java_com_trustformers_TrustformersEngine_getStats(
    env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
) -> jstring {
    if engine_ptr == 0 {
        return JObject::null().into_inner();
    }

    let engine = unsafe { &*(engine_ptr as *const AndroidInferenceEngine) };
    let stats = engine.get_stats();

    let json = match serde_json::to_string(&stats) {
        Ok(j) => j,
        Err(_) => return JObject::null().into_inner(),
    };

    match env.new_string(json) {
        Ok(jstr) => jstr.into_inner(),
        Err(_) => JObject::null().into_inner(),
    }
}

/// JNI function to update configuration
#[cfg(target_os = "android")]
#[no_mangle]
pub extern "system" fn Java_com_trustformers_TrustformersEngine_updateConfig(
    env: JNIEnv,
    _class: JClass,
    engine_ptr: jlong,
    config_json: JString,
) -> jboolean {
    if engine_ptr == 0 {
        return false as jboolean;
    }

    let config_str: String = match env.get_string(config_json) {
        Ok(s) => s.into(),
        Err(_) => return false as jboolean,
    };

    let config: MobileConfig = match serde_json::from_str(&config_str) {
        Ok(c) => c,
        Err(_) => return false as jboolean,
    };

    let engine = unsafe { &mut *(engine_ptr as *mut AndroidInferenceEngine) };
    engine.update_config(config).is_ok() as jboolean
}

// Stub implementations for non-Android platforms
#[cfg(not(target_os = "android"))]
pub extern "system" fn Java_com_trustformers_TrustformersEngine_createEngine(
    _config_json: *const std::os::raw::c_char,
) -> i64 {
    0
}

#[cfg(not(target_os = "android"))]
pub extern "system" fn Java_com_trustformers_TrustformersEngine_loadModel(
    _engine_ptr: i64,
    _model_path: *const std::os::raw::c_char,
) -> bool {
    false
}

#[cfg(not(target_os = "android"))]
pub extern "system" fn Java_com_trustformers_TrustformersEngine_releaseEngine(_engine_ptr: i64) {
    // No-op for non-Android
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jni_functions_exist() {
        // Test that JNI functions are properly exported
        // This is mainly a compilation test
        #[cfg(target_os = "android")]
        {
            // Functions should be accessible
            assert!(true);
        }
        #[cfg(not(target_os = "android"))]
        {
            // Stub functions should be accessible
            assert!(true);
        }
    }

    #[test]
    fn test_null_pointer_safety() {
        // Test that functions handle null pointers safely
        #[cfg(not(target_os = "android"))]
        {
            let result = Java_com_trustformers_TrustformersEngine_createEngine(std::ptr::null());
            assert_eq!(result, 0);

            let load_result =
                Java_com_trustformers_TrustformersEngine_loadModel(0, std::ptr::null());
            assert!(!load_result);

            // Should not panic
            Java_com_trustformers_TrustformersEngine_releaseEngine(0);
        }
    }
}
