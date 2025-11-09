//! Model Diagnostics Module Organization
//!
//! This module provides comprehensive model diagnostics and analysis capabilities
//! including performance monitoring, architecture analysis, training dynamics,
//! layer-level analysis, alert systems, auto-debugging, and advanced analytics.

/// Core data structures and type definitions
pub mod types;

/// Performance metrics and analysis
pub mod performance;

/// Model architecture analysis
pub mod architecture;

/// Training dynamics and convergence analysis
pub mod training;

/// Layer-level analysis and activations
pub mod layers;

/// Alert system and diagnostics
pub mod alerts;

/// Auto-debugging and recommendations
pub mod auto_debug;

/// Advanced analytics (clustering, temporal, stability)
pub mod analytics;

// Re-export all public types for backward compatibility
pub use self::alerts::*;
pub use self::analytics::*;
pub use self::architecture::*;
pub use self::auto_debug::*;
pub use self::layers::*;
pub use self::performance::*;
pub use self::training::*;
pub use self::types::*;
