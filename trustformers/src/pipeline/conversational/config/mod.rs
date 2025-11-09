//! Configuration management modules for conversational AI pipeline.
//!
//! This module is organized into several sub-modules for better maintainability:
//!
//! - [`manager`]: Central configuration manager with environment support
//! - [`builder`]: Fluent builder patterns for creating configurations
//! - [`presets`]: Pre-defined configuration presets for common use cases
//! - [`validation`]: Configuration validation rules and validators
//! - [`merging`]: Configuration merging and conflict resolution
//! - [`utils`]: Utility functions for configuration management

pub mod builder;
pub mod manager;
pub mod merging;
pub mod presets;
pub mod utils;
pub mod validation;

// Re-export main types for convenience
pub use builder::ConversationalConfigBuilder;
pub use manager::ConfigurationManager;
pub use merging::ConfigurationMerger;
pub use presets::{ConfigurationPreset, ConfigurationPresets};
pub use utils::utils as ConfigurationUtils;
pub use validation::{ConfigurationValidator, ValidationRules};
