//! Design patterns and standardized utilities for TrustformeRS Core
//!
//! This module provides standardized design patterns that should be used
//! consistently across the entire codebase for better maintainability
//! and developer experience.

pub mod builder;
pub mod config_unification;

pub use builder::{
    Buildable, Builder, BuilderError, BuilderResult, ConfigBuilder, ConfigBuilderImpl,
    ConfigSerializable, StandardBuilder, StandardConfig, ValidatedBuilder,
};

pub use config_unification::{
    merge_unified_configs, AccessControlConfig, AccessControlModel, AuthenticationConfig,
    AuthenticationMethod, BenchmarkConfig, BenchmarkFrequency, CacheConfig, CacheEvictionPolicy,
    ConfigManager, ConfigMetadata, ConfigSource, CpuLimits, DebugConfig, DebugLevel,
    DebugOutputFormat, EncryptionConfig, EnvironmentConfig, EnvironmentType, GpuLimits,
    KeyManagementConfig, KeyStorageLocation, LogFormat, LogLevel, LogOutput, LogRotation,
    LoggingConfig, MemoryLimits, NetworkLimits, OptimizationConfig, OptimizationLevel,
    PerformanceConfig, PrecisionConfig, PrecisionType, ProfilingLevel, ResourceConfig,
    SecurityConfig, SecurityLevel, StorageLimits, TimeoutConfig, UnifiedConfig,
};

/// Re-export macros for convenience
pub use crate::{builder_methods, quick_builder};

/// Standard result type for pattern operations
pub type PatternResult<T> = std::result::Result<T, PatternError>;

/// Errors that can occur in pattern implementations
#[derive(Debug, thiserror::Error)]
pub enum PatternError {
    #[error("Builder error: {0}")]
    Builder(#[from] BuilderError),
    #[error("Validation error: {reason}")]
    Validation { reason: String },
    #[error("Configuration error: {reason}")]
    Configuration { reason: String },
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Common validation functions that can be used across builders
pub mod validators {

    use crate::errors::Result;

    /// Validate that a string is not empty
    pub fn non_empty_string(value: &str, field_name: &str) -> Result<()> {
        if value.trim().is_empty() {
            return Err(crate::errors::TrustformersError::invalid_config(format!(
                "Field '{}' cannot be empty",
                field_name
            )));
        }
        Ok(())
    }

    /// Validate that a numeric value is within a range
    pub fn numeric_range<T: PartialOrd + std::fmt::Display>(
        value: T,
        min: T,
        max: T,
        field_name: &str,
    ) -> Result<()> {
        if value < min || value > max {
            return Err(crate::errors::TrustformersError::invalid_config(format!(
                "Field '{}' value {} must be between {} and {}",
                field_name, value, min, max
            )));
        }
        Ok(())
    }

    /// Validate that a value is positive
    pub fn positive<T: PartialOrd + Default + std::fmt::Display>(
        value: T,
        field_name: &str,
    ) -> Result<()> {
        if value <= T::default() {
            return Err(crate::errors::TrustformersError::invalid_config(format!(
                "Field '{}' value {} must be positive",
                field_name, value
            )));
        }
        Ok(())
    }

    /// Validate that a collection is not empty
    pub fn non_empty_collection<T>(collection: &[T], field_name: &str) -> Result<()> {
        if collection.is_empty() {
            return Err(crate::errors::TrustformersError::invalid_config(format!(
                "Field '{}' cannot be empty",
                field_name
            )));
        }
        Ok(())
    }

    /// Validate that a path exists
    pub fn path_exists(path: &std::path::Path, field_name: &str) -> Result<()> {
        if !path.exists() {
            return Err(crate::errors::TrustformersError::invalid_config(format!(
                "Path '{}' for field '{}' does not exist",
                path.display(),
                field_name
            )));
        }
        Ok(())
    }
}

/// Common configuration patterns
pub mod config_patterns {
    use super::*;
    use serde::{Deserialize, Serialize};

    /// Standard configuration base that all configs should inherit from
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BaseConfig {
        /// Configuration name/identifier
        pub name: Option<String>,
        /// Human-readable description
        pub description: Option<String>,
        /// Version of the configuration format
        pub version: String,
        /// Tags for categorization
        pub tags: Vec<String>,
        /// Whether this configuration is enabled
        pub enabled: bool,
        /// Timestamp when configuration was created
        #[serde(default = "chrono::Utc::now")]
        pub created_at: chrono::DateTime<chrono::Utc>,
        /// Timestamp when configuration was last modified
        #[serde(default = "chrono::Utc::now")]
        pub modified_at: chrono::DateTime<chrono::Utc>,
    }

    impl Default for BaseConfig {
        fn default() -> Self {
            let now = chrono::Utc::now();
            Self {
                name: None,
                description: None,
                version: "1.0.0".to_string(),
                tags: Vec::new(),
                enabled: true,
                created_at: now,
                modified_at: now,
            }
        }
    }

    impl StandardConfig for BaseConfig {
        fn validate(&self) -> crate::errors::Result<()> {
            if let Some(name) = &self.name {
                validators::non_empty_string(name, "name")?;
            }
            Ok(())
        }
    }

    /// Resource limits configuration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ResourceLimits {
        /// Maximum memory usage in bytes
        pub max_memory_bytes: Option<usize>,
        /// Maximum CPU percentage (0-100)
        pub max_cpu_percent: Option<f64>,
        /// Maximum GPU memory in bytes
        pub max_gpu_memory_bytes: Option<usize>,
        /// Timeout in milliseconds
        pub timeout_ms: Option<u64>,
        /// Maximum concurrent operations
        pub max_concurrent: Option<usize>,
    }

    impl Default for ResourceLimits {
        fn default() -> Self {
            Self {
                max_memory_bytes: None,
                max_cpu_percent: Some(80.0),
                max_gpu_memory_bytes: None,
                timeout_ms: Some(300_000), // 5 minutes
                max_concurrent: Some(4),
            }
        }
    }

    impl StandardConfig for ResourceLimits {
        fn validate(&self) -> crate::errors::Result<()> {
            if let Some(cpu) = self.max_cpu_percent {
                validators::numeric_range(cpu, 0.0, 100.0, "max_cpu_percent")?;
            }
            if let Some(concurrent) = self.max_concurrent {
                validators::positive(concurrent, "max_concurrent")?;
            }
            Ok(())
        }
    }

    /// Logging configuration
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct LoggingConfig {
        /// Log level
        pub level: String,
        /// Log format (json, text, etc.)
        pub format: String,
        /// Log output destination
        pub output: String,
        /// Whether to include timestamps
        pub include_timestamps: bool,
        /// Whether to include source file information
        pub include_source: bool,
        /// Maximum log file size in bytes
        pub max_file_size_bytes: Option<usize>,
        /// Maximum number of log files to keep
        pub max_files: Option<usize>,
    }

    impl Default for LoggingConfig {
        fn default() -> Self {
            Self {
                level: "info".to_string(),
                format: "text".to_string(),
                output: "stdout".to_string(),
                include_timestamps: true,
                include_source: false,
                max_file_size_bytes: Some(10 * 1024 * 1024), // 10 MB
                max_files: Some(5),
            }
        }
    }

    impl StandardConfig for LoggingConfig {
        fn validate(&self) -> crate::errors::Result<()> {
            let valid_levels = ["trace", "debug", "info", "warn", "error"];
            if !valid_levels.contains(&self.level.as_str()) {
                return Err(crate::errors::TrustformersError::invalid_config(format!(
                    "Invalid log level '{}'. Must be one of: {}",
                    self.level,
                    valid_levels.join(", ")
                )));
            }

            let valid_formats = ["json", "text", "pretty"];
            if !valid_formats.contains(&self.format.as_str()) {
                return Err(crate::errors::TrustformersError::invalid_config(format!(
                    "Invalid log format '{}'. Must be one of: {}",
                    self.format,
                    valid_formats.join(", ")
                )));
            }

            Ok(())
        }
    }
}

/// Example usage and documentation
pub mod examples {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Default, Serialize, Deserialize)]
    pub struct ExampleConfig {
        pub base: config_patterns::BaseConfig,
        pub resources: config_patterns::ResourceLimits,
        pub logging: config_patterns::LoggingConfig,
        pub custom_setting: String,
    }

    impl StandardConfig for ExampleConfig {
        fn validate(&self) -> crate::errors::Result<()> {
            self.base.validate()?;
            self.resources.validate()?;
            self.logging.validate()?;
            validators::non_empty_string(&self.custom_setting, "custom_setting")?;
            Ok(())
        }
    }

    /// Example of creating a builder for the ExampleConfig
    pub fn example_config_builder() -> ConfigBuilderImpl<ExampleConfig, ExampleConfig> {
        ConfigBuilderImpl::new()
    }

    /// Example usage function showing how to use the standardized patterns
    #[allow(dead_code)]
    pub fn example_usage() -> crate::errors::Result<ExampleConfig> {
        let config = ExampleConfig {
            custom_setting: "example_value".to_string(),
            ..Default::default()
        };

        let builder = example_config_builder()
            .config(config)
            .name("example_configuration")
            .description("An example of using standardized builder patterns")
            .tag("example")
            .tag("documentation");

        builder.build()
    }
}
