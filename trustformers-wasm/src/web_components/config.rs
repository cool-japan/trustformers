use serde::{Deserialize, Serialize};
use std::string::String;
use std::vec::Vec;
use wasm_bindgen::prelude::*;
use web_sys::Document;

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ComponentType {
    InferenceEngine,
    ModelLoader,
    TensorVisualization,
    PerformanceMonitor,
    BatchProcessor,
    ModelRegistry,
    QuantizationControl,
    DebugConsole,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InferenceState {
    Idle,
    Loading,
    Ready,
    Processing,
    Complete,
    Error,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelState {
    Unloaded,
    Downloading,
    Loading,
    Loaded,
    Failed,
    Cached,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebComponentConfig {
    enable_shadow_dom: bool,
    theme: String,
    custom_styles: String,
    enable_debug: bool,
    auto_resize: bool,
    event_delegation: bool,
}

impl Default for WebComponentConfig {
    fn default() -> Self {
        Self {
            enable_shadow_dom: true,
            theme: "default".to_string(),
            custom_styles: String::new(),
            enable_debug: false,
            auto_resize: true,
            event_delegation: true,
        }
    }
}

#[wasm_bindgen]
impl WebComponentConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    #[wasm_bindgen(getter)]
    pub fn enable_shadow_dom(&self) -> bool {
        self.enable_shadow_dom
    }

    #[wasm_bindgen(setter)]
    pub fn set_enable_shadow_dom(&mut self, value: bool) {
        self.enable_shadow_dom = value;
    }

    #[wasm_bindgen(getter)]
    pub fn theme(&self) -> String {
        self.theme.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_theme(&mut self, theme: String) {
        self.theme = theme;
    }

    #[wasm_bindgen(getter)]
    pub fn custom_styles(&self) -> String {
        self.custom_styles.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_custom_styles(&mut self, styles: String) {
        self.custom_styles = styles;
    }

    #[wasm_bindgen(getter)]
    pub fn enable_debug(&self) -> bool {
        self.enable_debug
    }

    #[wasm_bindgen(setter)]
    pub fn set_enable_debug(&mut self, value: bool) {
        self.enable_debug = value;
    }

    #[wasm_bindgen(getter)]
    pub fn auto_resize(&self) -> bool {
        self.auto_resize
    }

    #[wasm_bindgen(setter)]
    pub fn set_auto_resize(&mut self, value: bool) {
        self.auto_resize = value;
    }

    #[wasm_bindgen(getter)]
    pub fn event_delegation(&self) -> bool {
        self.event_delegation
    }

    #[wasm_bindgen(setter)]
    pub fn set_event_delegation(&mut self, value: bool) {
        self.event_delegation = value;
    }
}

#[wasm_bindgen]
pub struct WebComponentFactory {
    pub(crate) config: WebComponentConfig,
    pub(crate) document: Document,
    pub(crate) registered_components: Vec<String>,
}
