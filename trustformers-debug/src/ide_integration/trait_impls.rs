//! # IDEPluginConfig - Trait Implementations
//!
//! This module contains trait implementations for `IDEPluginConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Default`
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::path::PathBuf;

use super::types::IDEPluginConfig;
use super::types_3::{IDEPluginManager, JupyterWidgetManager};

impl Default for IDEPluginConfig {
    fn default() -> Self {
        Self {
            enable_syntax_highlighting: true,
            enable_code_completion: true,
            enable_inline_debugging: true,
            enable_tensor_visualization: true,
            enable_real_time_metrics: true,
            auto_open_debugger: false,
            visualization_format: "png".to_string(),
            debug_port: 8899,
            max_variable_display_length: 1000,
            refresh_interval_ms: 1000,
            workspace_root: PathBuf::from("."),
            log_level: "info".to_string(),
        }
    }
}

impl Default for IDEPluginManager {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for JupyterWidgetManager {
    fn default() -> Self {
        Self::new()
    }
}
