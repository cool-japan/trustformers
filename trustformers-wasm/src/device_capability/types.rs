//! Device capability type definitions and enums

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DeviceType {
    Mobile,
    Tablet,
    Desktop,
    TV,
    Watch,
    Unknown,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum OperatingSystem {
    iOS,
    Android,
    Windows,
    MacOS,
    Linux,
    ChromeOS,
    Unknown,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Browser {
    Chrome,
    Firefox,
    Safari,
    Edge,
    Opera,
    Samsung,
    Unknown,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GPUVendor {
    Apple,
    Qualcomm,
    Mali,
    Adreno,
    NVIDIA,
    AMD,
    Intel,
    PowerVR,
    Unknown,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum CPUArchitecture {
    ARM64,
    ARM32,
    x86_64,
    x86,
    Unknown,
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ModelComplexity {
    Minimal,    // < 10M parameters
    Small,      // 10M - 100M parameters
    Medium,     // 100M - 1B parameters
    Large,      // 1B - 10B parameters
    ExtraLarge, // > 10B parameters
}

#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum InferenceSpeed {
    RealTime, // < 100ms
    Fast,     // 100ms - 500ms
    Medium,   // 500ms - 2s
    Slow,     // 2s - 10s
    VerySlow, // > 10s
}
