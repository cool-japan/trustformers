//! Device capability detection modules
//!
//! This module provides comprehensive device capability detection
//! split into logical components for better maintainability.

pub mod detector;
pub mod structs;
pub mod types;
pub mod utils;

// Re-export main types for convenience
pub use detector::DeviceCapabilityDetector;
pub use structs::*;
pub use types::*;
pub use utils::*;
