//! Debug utilities for TrustformeRS C API
//!
//! This module provides comprehensive debugging and profiling capabilities including:
//! - Model introspection and analysis
//! - Performance profiling and bottleneck detection
//! - Memory usage tracking and leak detection
//! - Interactive debugging console
//! - Visualization and reporting tools

pub mod c_api;
pub mod console;
pub mod debug_manager;
pub mod types;

// Re-export main types and functionality
pub use c_api::*;
pub use console::{start_interactive_console, InteractiveDebugConsole};
pub use debug_manager::DebugUtilities;
pub use types::*;

// Re-export console functions
pub use console::{trustformers_debug_console_available, trustformers_debug_console_start};
