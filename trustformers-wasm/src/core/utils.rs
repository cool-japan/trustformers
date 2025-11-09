use std::format;
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;

#[cfg(feature = "webgpu")]
use web_sys::console;

/// Get current time in milliseconds
#[wasm_bindgen]
pub fn get_current_time_ms() -> f64 {
    js_sys::Date::now()
}

// Logging utilities
#[wasm_bindgen]
pub fn log(_msg: &str) {
    #[cfg(all(feature = "console_panic", feature = "webgpu"))]
    console::log_1(&_msg.into());
}

#[wasm_bindgen]
pub fn log_error(_msg: &str) {
    #[cfg(all(feature = "console_panic", feature = "webgpu"))]
    console::error_1(&_msg.into());
}

#[wasm_bindgen]
pub fn log_warn(_msg: &str) {
    #[cfg(all(feature = "console_panic", feature = "webgpu"))]
    console::warn_1(&_msg.into());
}

// Performance measurement
#[wasm_bindgen]
pub struct Timer {
    #[allow(dead_code)]
    start_time: f64,
    name: String,
}

#[wasm_bindgen]
impl Timer {
    #[wasm_bindgen(constructor)]
    pub fn new(name: String) -> Timer {
        #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
        let start_time =
            web_sys::window().and_then(|w| w.performance()).map(|p| p.now()).unwrap_or(0.0);

        #[cfg(not(all(target_arch = "wasm32", feature = "webgpu")))]
        let start_time = 0.0;

        Timer { start_time, name }
    }

    pub fn elapsed(&self) -> f64 {
        #[cfg(all(target_arch = "wasm32", feature = "webgpu"))]
        {
            web_sys::window()
                .and_then(|w| w.performance())
                .map(|p| p.now() - self.start_time)
                .unwrap_or(0.0)
        }

        #[cfg(not(all(target_arch = "wasm32", feature = "webgpu")))]
        0.0
    }

    pub fn log_elapsed(&self) {
        let elapsed = self.elapsed();
        log(&format!("{name}: {elapsed:.2}ms", name = self.name));
    }
}

// Memory utilities - use main MemoryStats from lib.rs

// Model format utilities
#[wasm_bindgen]
pub enum ModelFormat {
    Onnx,
    Gguf,
    SafeTensors,
    Json,
}

#[wasm_bindgen]
pub struct ModelLoader;

#[wasm_bindgen]
impl ModelLoader {
    #[wasm_bindgen(constructor)]
    pub fn new() -> ModelLoader {
        ModelLoader
    }

    pub async fn load_from_url(
        &self,
        _url: &str,
        #[allow(unused_variables)] format: ModelFormat,
    ) -> Result<Vec<u8>, JsValue> {
        #[cfg(feature = "webgpu")]
        {
            use web_sys::{Request, RequestInit, RequestMode, Response};

            let window = web_sys::window().ok_or("No window")?;

            let opts = RequestInit::new();
            opts.set_method("GET");
            opts.set_mode(RequestMode::Cors);

            let request = Request::new_with_str_and_init(_url, &opts)?;

            let resp_value =
                wasm_bindgen_futures::JsFuture::from(window.fetch_with_request(&request)).await?;
            let resp: Response = resp_value.dyn_into()?;

            let array_buffer = wasm_bindgen_futures::JsFuture::from(resp.array_buffer()?).await?;
            let uint8_array = js_sys::Uint8Array::new(&array_buffer);

            Ok(uint8_array.to_vec())
        }

        #[cfg(not(feature = "webgpu"))]
        Err(JsValue::from_str("WebGPU feature not enabled"))
    }

    pub fn parse_model(&self, data: &[u8], format: ModelFormat) -> Result<JsValue, JsValue> {
        match format {
            ModelFormat::Json => {
                let json_str = core::str::from_utf8(data)
                    .map_err(|e| JsValue::from_str(&format!("Invalid UTF-8: {e:?}")))?;
                Ok(JsValue::from_str(json_str))
            },
            _ => Err(JsValue::from_str("Model format not yet supported")),
        }
    }
}

impl Default for ModelLoader {
    fn default() -> Self {
        Self::new()
    }
}

// WebGPU utilities and device detection
pub fn has_webgpu() -> bool {
    web_sys::window()
        .and_then(|w| {
            js_sys::Reflect::get(&w.navigator(), &JsValue::from_str("gpu"))
                .ok()
                .filter(|v| !v.is_undefined())
        })
        .is_some()
}

pub fn has_webgl() -> bool {
    let window = web_sys::window();
    if let Some(w) = window {
        if let Some(document) = w.document() {
            if let Ok(canvas) = document.create_element("canvas") {
                if let Ok(canvas) = canvas.dyn_into::<web_sys::HtmlCanvasElement>() {
                    return canvas.get_context("webgl").is_ok()
                        || canvas.get_context("webgl2").is_ok();
                }
            }
        }
    }
    false
}

pub fn has_simd() -> bool {
    // Check if SIMD is available in the WASM environment
    cfg!(target_feature = "simd128")
}

pub fn has_threads() -> bool {
    // Check if SharedArrayBuffer is available (indicates thread support)
    js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("SharedArrayBuffer"))
        .map(|v| !v.is_undefined())
        .unwrap_or(false)
}

pub fn get_memory_mb() -> usize {
    // Get available memory in MB
    // Since we can't reliably detect actual memory, use a conservative estimate
    // based on typical browser limits for WASM
    if cfg!(target_arch = "wasm32") {
        // In WASM, we typically have access to up to 4GB, but browsers limit it
        // Use a conservative estimate of 1GB
        1024
    } else {
        256 // Default fallback for non-WASM environments
    }
}

// Legacy WebGPU check (kept for backward compatibility)
#[cfg(feature = "webgpu")]
pub fn check_webgpu_available() -> bool {
    has_webgpu()
}

// Export utility functions
#[wasm_bindgen]
pub fn version() -> String {
    format!(
        "trustformers-wasm v{version}",
        version = env!("CARGO_PKG_VERSION")
    )
}

#[wasm_bindgen]
pub fn features() -> Vec<JsValue> {
    let mut features = vec![JsValue::from_str("core")];

    #[cfg(feature = "console_panic")]
    features.push(JsValue::from_str("console_panic"));

    #[cfg(feature = "webgpu")]
    features.push(JsValue::from_str("webgpu"));

    #[cfg(target_feature = "simd128")]
    features.push(JsValue::from_str("simd128"));

    features
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let v = version();
        assert!(v.contains("trustformers-wasm"));
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn test_features() {
        let f = features();
        assert!(f.len() > 0);
    }
}
