//! WebAssembly-compatible model loading and inference.
//!
//! Split into submodules for maintainability (see individual module docs
//! for what changed from the earlier placeholder implementation):
//! - [`config`]: architecture/format enums and default per-architecture sizing.
//! - [`formats`]: checkpoint format detection/parsing (real SafeTensors reader;
//!   TensorRT/CoreML/TensorFlowLite are detection+metadata only).
//! - [`weights`]: named weight storage and lookup.
//! - [`forward`]: the actual transformer math (pure Rust, no `wasm_bindgen`).
//! - [`wasm_model`]: [`WasmModel`]/[`QuantizedModel`], wiring the above together.

pub mod config;
pub mod formats;
pub mod forward;
pub mod wasm_model;
pub mod weights;

pub use config::{ModelArchitecture, ModelConfig, ModelFormat};
pub use wasm_model::{QuantizationType, QuantizedModel, WasmModel};
