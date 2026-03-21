//! Device capability utility functions

use super::detector::DeviceCapabilityDetector;
use super::structs::DeviceCapabilitiesWasm;
use wasm_bindgen::prelude::*;
use web_sys::window;

// Factory functions for easy creation
#[wasm_bindgen]
pub async fn detect_device_capabilities() -> Result<DeviceCapabilitiesWasm, JsValue> {
    let mut detector = DeviceCapabilityDetector::new();
    detector.detect_all_capabilities().await
}

#[wasm_bindgen]
pub async fn create_capability_detector() -> Result<DeviceCapabilityDetector, JsValue> {
    let mut detector = DeviceCapabilityDetector::new();
    detector.detect_all_capabilities().await?;
    Ok(detector)
}

// Quick capability checks
#[wasm_bindgen]
pub fn is_high_performance_device() -> bool {
    // Quick synchronous check
    if let Some(window) = window() {
        let navigator = window.navigator();
        let cores = navigator.hardware_concurrency() as u32;
        let memory = js_sys::Reflect::get(&navigator, &"deviceMemory".into())
            .unwrap_or_default()
            .as_f64()
            .unwrap_or(4.0);

        cores >= 8 && memory >= 6.0
    } else {
        false
    }
}

#[wasm_bindgen]
pub fn is_mobile_device() -> bool {
    if let Some(window) = window() {
        let navigator = window.navigator();
        let user_agent = navigator.user_agent().unwrap_or_default();
        user_agent.contains("Mobile") || user_agent.contains("Android")
    } else {
        false
    }
}

#[wasm_bindgen]
pub fn supports_advanced_features() -> bool {
    if let Some(window) = window() {
        let navigator = window.navigator();

        let webgpu = js_sys::Reflect::has(&navigator, &"gpu".into()).unwrap_or(false);
        let wasm = js_sys::Reflect::has(&window, &"WebAssembly".into()).unwrap_or(false);
        let shared_array_buffer =
            js_sys::Reflect::has(&window, &"SharedArrayBuffer".into()).unwrap_or(false);

        webgpu && wasm && shared_array_buffer
    } else {
        false
    }
}
