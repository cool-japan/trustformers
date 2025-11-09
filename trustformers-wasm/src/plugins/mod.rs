// Plugin implementations module
// Contains example and reference plugins for the TrustformeRS WASM plugin framework

pub mod example_plugin;

pub use example_plugin::{ModelOptimizerPlugin, TextProcessorPlugin, VisualizationPlugin};

// Re-export for convenience
pub use crate::plugin_framework::*;
