//! Configuration Unification for TrustformeRS Core
//!
//! This module provides unified configuration structures and patterns
//! that standardize configuration management across all modules.

use super::{validators, ConfigSerializable, StandardConfig};
use crate::errors::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Unified configuration base that all specific configs should inherit from
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UnifiedConfig {
    /// Configuration metadata
    pub metadata: ConfigMetadata,
    /// Resource limitations
    pub resources: ResourceConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
    /// Performance settings
    pub performance: PerformanceConfig,
    /// Security settings
    pub security: SecurityConfig,
    /// Environment-specific settings
    pub environment: EnvironmentConfig,
}

impl StandardConfig for UnifiedConfig {
    fn validate(&self) -> Result<()> {
        self.metadata.validate()?;
        self.resources.validate()?;
        self.logging.validate()?;
        self.performance.validate()?;
        self.security.validate()?;
        self.environment.validate()?;
        Ok(())
    }
}

/// Configuration metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigMetadata {
    /// Configuration name/identifier
    pub name: String,
    /// Human-readable description
    pub description: Option<String>,
    /// Configuration version
    pub version: String,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Whether this configuration is enabled
    pub enabled: bool,
    /// Creation timestamp
    #[serde(default = "chrono::Utc::now")]
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last modification timestamp
    #[serde(default = "chrono::Utc::now")]
    pub modified_at: chrono::DateTime<chrono::Utc>,
    /// Author/creator information
    pub author: Option<String>,
    /// Configuration source (file, environment, default, etc.)
    pub source: ConfigSource,
    /// Configuration priority (higher priority configs override lower)
    pub priority: u32,
}

impl Default for ConfigMetadata {
    fn default() -> Self {
        let now = chrono::Utc::now();
        Self {
            name: "default".to_string(),
            description: None,
            version: "1.0.0".to_string(),
            tags: Vec::new(),
            enabled: true,
            created_at: now,
            modified_at: now,
            author: None,
            source: ConfigSource::Default,
            priority: 100,
        }
    }
}

impl StandardConfig for ConfigMetadata {
    fn validate(&self) -> Result<()> {
        validators::non_empty_string(&self.name, "name")?;
        validators::non_empty_string(&self.version, "version")?;
        Ok(())
    }
}

/// Source of configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConfigSource {
    Default,
    File,
    Environment,
    Database,
    Remote,
    CommandLine,
    Code,
}

/// Unified resource configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceConfig {
    /// Memory limits
    pub memory: MemoryLimits,
    /// CPU limits
    pub cpu: CpuLimits,
    /// GPU limits
    pub gpu: GpuLimits,
    /// Storage limits
    pub storage: StorageLimits,
    /// Network limits
    pub network: NetworkLimits,
    /// Timeout settings
    pub timeouts: TimeoutConfig,
}

impl StandardConfig for ResourceConfig {
    fn validate(&self) -> Result<()> {
        self.memory.validate()?;
        self.cpu.validate()?;
        self.gpu.validate()?;
        self.storage.validate()?;
        self.network.validate()?;
        self.timeouts.validate()?;
        Ok(())
    }
}

/// Memory configuration limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLimits {
    /// Maximum heap memory in bytes
    pub max_heap_bytes: Option<u64>,
    /// Maximum GPU memory in bytes
    pub max_gpu_bytes: Option<u64>,
    /// Maximum shared memory in bytes
    pub max_shared_bytes: Option<u64>,
    /// Memory warning threshold (percentage)
    pub warning_threshold_percent: f64,
    /// Enable memory monitoring
    pub monitoring_enabled: bool,
    /// Memory pressure detection
    pub pressure_detection: bool,
}

impl Default for MemoryLimits {
    fn default() -> Self {
        Self {
            max_heap_bytes: None,
            max_gpu_bytes: None,
            max_shared_bytes: None,
            warning_threshold_percent: 80.0,
            monitoring_enabled: true,
            pressure_detection: true,
        }
    }
}

impl StandardConfig for MemoryLimits {
    fn validate(&self) -> Result<()> {
        validators::numeric_range(
            self.warning_threshold_percent,
            0.0,
            100.0,
            "warning_threshold_percent",
        )?;
        Ok(())
    }
}

/// CPU configuration limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuLimits {
    /// Maximum CPU cores to use
    pub max_cores: Option<usize>,
    /// CPU usage limit (percentage)
    pub max_usage_percent: Option<f64>,
    /// Thread pool size
    pub thread_pool_size: Option<usize>,
    /// CPU affinity settings
    pub affinity: Vec<usize>,
    /// NUMA node preferences
    pub numa_nodes: Vec<usize>,
}

impl Default for CpuLimits {
    fn default() -> Self {
        Self {
            max_cores: None,
            max_usage_percent: Some(80.0),
            thread_pool_size: None,
            affinity: Vec::new(),
            numa_nodes: Vec::new(),
        }
    }
}

impl StandardConfig for CpuLimits {
    fn validate(&self) -> Result<()> {
        if let Some(usage) = self.max_usage_percent {
            validators::numeric_range(usage, 0.0, 100.0, "max_usage_percent")?;
        }
        if let Some(cores) = self.max_cores {
            validators::positive(cores, "max_cores")?;
        }
        Ok(())
    }
}

/// GPU configuration limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuLimits {
    /// Maximum number of GPUs to use
    pub max_devices: Option<usize>,
    /// Specific GPU device IDs to use
    pub device_ids: Vec<usize>,
    /// GPU memory limit per device in bytes
    pub memory_per_device_bytes: Option<u64>,
    /// GPU usage limit per device (percentage)
    pub max_usage_percent: Option<f64>,
    /// Enable GPU monitoring
    pub monitoring_enabled: bool,
}

impl Default for GpuLimits {
    fn default() -> Self {
        Self {
            max_devices: None,
            device_ids: Vec::new(),
            memory_per_device_bytes: None,
            max_usage_percent: Some(90.0),
            monitoring_enabled: true,
        }
    }
}

impl StandardConfig for GpuLimits {
    fn validate(&self) -> Result<()> {
        if let Some(usage) = self.max_usage_percent {
            validators::numeric_range(usage, 0.0, 100.0, "max_usage_percent")?;
        }
        Ok(())
    }
}

/// Storage configuration limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageLimits {
    /// Maximum disk space in bytes
    pub max_disk_bytes: Option<u64>,
    /// Temporary directory path
    pub temp_dir: Option<PathBuf>,
    /// Cache directory path
    pub cache_dir: Option<PathBuf>,
    /// Maximum cache size in bytes
    pub max_cache_bytes: Option<u64>,
    /// Disk usage warning threshold (percentage)
    pub warning_threshold_percent: f64,
}

impl Default for StorageLimits {
    fn default() -> Self {
        Self {
            max_disk_bytes: None,
            temp_dir: Some(std::env::temp_dir()),
            cache_dir: None,
            max_cache_bytes: Some(1024 * 1024 * 1024), // 1GB
            warning_threshold_percent: 85.0,
        }
    }
}

impl StandardConfig for StorageLimits {
    fn validate(&self) -> Result<()> {
        validators::numeric_range(
            self.warning_threshold_percent,
            0.0,
            100.0,
            "warning_threshold_percent",
        )?;

        if let Some(temp_dir) = &self.temp_dir {
            if !temp_dir.exists() {
                return Err(anyhow::anyhow!(
                    "Temp directory does not exist: {}",
                    temp_dir.display()
                )
                .into());
            }
        }

        Ok(())
    }
}

/// Network configuration limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkLimits {
    /// Maximum bandwidth in bytes per second
    pub max_bandwidth_bps: Option<u64>,
    /// Connection timeout in milliseconds
    pub connection_timeout_ms: u64,
    /// Request timeout in milliseconds
    pub request_timeout_ms: u64,
    /// Maximum concurrent connections
    pub max_connections: Option<usize>,
    /// Retry attempts
    pub max_retries: u32,
}

impl Default for NetworkLimits {
    fn default() -> Self {
        Self {
            max_bandwidth_bps: None,
            connection_timeout_ms: 30_000, // 30 seconds
            request_timeout_ms: 300_000,   // 5 minutes
            max_connections: Some(100),
            max_retries: 3,
        }
    }
}

impl StandardConfig for NetworkLimits {
    fn validate(&self) -> Result<()> {
        validators::positive(self.connection_timeout_ms, "connection_timeout_ms")?;
        validators::positive(self.request_timeout_ms, "request_timeout_ms")?;
        Ok(())
    }
}

/// Timeout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    /// Default operation timeout in milliseconds
    pub default_timeout_ms: u64,
    /// Training timeout in milliseconds
    pub training_timeout_ms: Option<u64>,
    /// Inference timeout in milliseconds
    pub inference_timeout_ms: Option<u64>,
    /// Export timeout in milliseconds
    pub export_timeout_ms: Option<u64>,
    /// Model loading timeout in milliseconds
    pub loading_timeout_ms: Option<u64>,
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            default_timeout_ms: 300_000, // 5 minutes
            training_timeout_ms: None,
            inference_timeout_ms: Some(30_000), // 30 seconds
            export_timeout_ms: Some(1_800_000), // 30 minutes
            loading_timeout_ms: Some(600_000),  // 10 minutes
        }
    }
}

impl StandardConfig for TimeoutConfig {
    fn validate(&self) -> Result<()> {
        validators::positive(self.default_timeout_ms, "default_timeout_ms")?;
        Ok(())
    }
}

/// Unified logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: LogLevel,
    /// Log format
    pub format: LogFormat,
    /// Log outputs
    pub outputs: Vec<LogOutput>,
    /// Include timestamps
    pub include_timestamps: bool,
    /// Include source file information
    pub include_source: bool,
    /// Include thread information
    pub include_thread: bool,
    /// Maximum log file size in bytes
    pub max_file_size_bytes: Option<u64>,
    /// Maximum number of log files to keep
    pub max_files: Option<usize>,
    /// Log rotation strategy
    pub rotation: LogRotation,
    /// Structured logging fields
    pub structured_fields: HashMap<String, String>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            format: LogFormat::Text,
            outputs: vec![LogOutput::Stdout],
            include_timestamps: true,
            include_source: false,
            include_thread: false,
            max_file_size_bytes: Some(10 * 1024 * 1024), // 10 MB
            max_files: Some(5),
            rotation: LogRotation::Size,
            structured_fields: HashMap::new(),
        }
    }
}

impl StandardConfig for LoggingConfig {}

/// Log levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

/// Log formats
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LogFormat {
    Text,
    Json,
    Pretty,
    Compact,
}

/// Log outputs
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LogOutput {
    Stdout,
    Stderr,
    File(PathBuf),
    Syslog,
    Network(String),
}

/// Log rotation strategies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LogRotation {
    None,
    Size,
    Time,
    Daily,
    Weekly,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable performance monitoring
    pub monitoring_enabled: bool,
    /// Profiling level
    pub profiling_level: ProfilingLevel,
    /// Benchmark settings
    pub benchmarking: BenchmarkConfig,
    /// Optimization settings
    pub optimization: OptimizationConfig,
    /// Cache settings
    pub caching: CacheConfig,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            monitoring_enabled: true,
            profiling_level: ProfilingLevel::Basic,
            benchmarking: BenchmarkConfig::default(),
            optimization: OptimizationConfig::default(),
            caching: CacheConfig::default(),
        }
    }
}

impl StandardConfig for PerformanceConfig {}

/// Profiling levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProfilingLevel {
    None,
    Basic,
    Detailed,
    Comprehensive,
}

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Enable automatic benchmarking
    pub auto_benchmark: bool,
    /// Benchmark frequency
    pub frequency: BenchmarkFrequency,
    /// Warmup iterations
    pub warmup_iterations: usize,
    /// Measurement iterations
    pub measurement_iterations: usize,
    /// Statistical confidence level
    pub confidence_level: f64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            auto_benchmark: false,
            frequency: BenchmarkFrequency::Never,
            warmup_iterations: 3,
            measurement_iterations: 10,
            confidence_level: 0.95,
        }
    }
}

/// Benchmark frequency
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum BenchmarkFrequency {
    Never,
    OnStartup,
    Daily,
    Weekly,
    OnConfigChange,
    OnDemand,
}

/// Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Optimization level
    pub level: OptimizationLevel,
    /// Enable specific optimizations
    pub optimizations: HashMap<String, bool>,
    /// Target hardware
    pub target_hardware: Option<String>,
    /// Precision settings
    pub precision: PrecisionConfig,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        let mut optimizations = HashMap::new();
        optimizations.insert("simd".to_string(), true);
        optimizations.insert("vectorization".to_string(), true);
        optimizations.insert("loop_unrolling".to_string(), true);
        optimizations.insert("kernel_fusion".to_string(), true);

        Self {
            level: OptimizationLevel::Balanced,
            optimizations,
            target_hardware: None,
            precision: PrecisionConfig::default(),
        }
    }
}

/// Optimization levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OptimizationLevel {
    None,
    Basic,
    Balanced,
    Aggressive,
    Maximum,
}

/// Precision configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrecisionConfig {
    /// Default precision
    pub default_precision: PrecisionType,
    /// Mixed precision enabled
    pub mixed_precision: bool,
    /// Automatic precision selection
    pub auto_precision: bool,
    /// Precision loss threshold
    pub loss_threshold: f64,
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self {
            default_precision: PrecisionType::FP32,
            mixed_precision: false,
            auto_precision: false,
            loss_threshold: 1e-6,
        }
    }
}

/// Precision types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum PrecisionType {
    FP16,
    BF16,
    FP32,
    FP64,
    INT8,
    INT16,
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable caching
    pub enabled: bool,
    /// Cache size in bytes
    pub size_bytes: Option<u64>,
    /// Cache eviction policy
    pub eviction_policy: CacheEvictionPolicy,
    /// Cache TTL in seconds
    pub ttl_seconds: Option<u64>,
    /// Cache directory
    pub cache_dir: Option<PathBuf>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            size_bytes: Some(1024 * 1024 * 1024), // 1GB
            eviction_policy: CacheEvictionPolicy::LRU,
            ttl_seconds: Some(3600), // 1 hour
            cache_dir: None,
        }
    }
}

/// Cache eviction policies
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CacheEvictionPolicy {
    LRU,
    LFU,
    FIFO,
    Random,
    TTL,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable security features
    pub enabled: bool,
    /// Security level
    pub level: SecurityLevel,
    /// Encryption settings
    pub encryption: EncryptionConfig,
    /// Authentication settings
    pub authentication: AuthenticationConfig,
    /// Access control settings
    pub access_control: AccessControlConfig,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            level: SecurityLevel::Standard,
            encryption: EncryptionConfig::default(),
            authentication: AuthenticationConfig::default(),
            access_control: AccessControlConfig::default(),
        }
    }
}

impl StandardConfig for SecurityConfig {}

/// Security levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SecurityLevel {
    Minimal,
    Standard,
    High,
    Maximum,
}

/// Encryption configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionConfig {
    /// Enable encryption at rest
    pub at_rest: bool,
    /// Enable encryption in transit
    pub in_transit: bool,
    /// Encryption algorithm
    pub algorithm: Option<String>,
    /// Key management
    pub key_management: KeyManagementConfig,
}

impl Default for EncryptionConfig {
    fn default() -> Self {
        Self {
            at_rest: false,
            in_transit: true,
            algorithm: Some("AES-256-GCM".to_string()),
            key_management: KeyManagementConfig::default(),
        }
    }
}

/// Key management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyManagementConfig {
    /// Key rotation enabled
    pub rotation_enabled: bool,
    /// Key rotation interval in days
    pub rotation_interval_days: Option<u32>,
    /// Key storage location
    pub storage_location: KeyStorageLocation,
}

impl Default for KeyManagementConfig {
    fn default() -> Self {
        Self {
            rotation_enabled: false,
            rotation_interval_days: Some(90),
            storage_location: KeyStorageLocation::Local,
        }
    }
}

/// Key storage locations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum KeyStorageLocation {
    Local,
    Environment,
    Vault,
    HSM,
    Cloud,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationConfig {
    /// Authentication method
    pub method: AuthenticationMethod,
    /// Session timeout in seconds
    pub session_timeout_seconds: Option<u64>,
    /// Multi-factor authentication
    pub mfa_enabled: bool,
}

impl Default for AuthenticationConfig {
    fn default() -> Self {
        Self {
            method: AuthenticationMethod::None,
            session_timeout_seconds: Some(3600), // 1 hour
            mfa_enabled: false,
        }
    }
}

/// Authentication methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AuthenticationMethod {
    None,
    ApiKey,
    JWT,
    OAuth2,
    LDAP,
    SAML,
}

/// Access control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlConfig {
    /// Access control model
    pub model: AccessControlModel,
    /// Default permissions
    pub default_permissions: Vec<String>,
    /// Permission inheritance
    pub inheritance_enabled: bool,
}

impl Default for AccessControlConfig {
    fn default() -> Self {
        Self {
            model: AccessControlModel::None,
            default_permissions: vec!["read".to_string()],
            inheritance_enabled: true,
        }
    }
}

/// Access control models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AccessControlModel {
    None,
    RBAC, // Role-Based Access Control
    ABAC, // Attribute-Based Access Control
    DAC,  // Discretionary Access Control
    MAC,  // Mandatory Access Control
}

/// Environment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentConfig {
    /// Environment type
    pub environment_type: EnvironmentType,
    /// Environment variables
    pub variables: HashMap<String, String>,
    /// Feature flags
    pub feature_flags: HashMap<String, bool>,
    /// Debug settings
    pub debug: DebugConfig,
}

impl Default for EnvironmentConfig {
    fn default() -> Self {
        Self {
            environment_type: EnvironmentType::Development,
            variables: HashMap::new(),
            feature_flags: HashMap::new(),
            debug: DebugConfig::default(),
        }
    }
}

impl StandardConfig for EnvironmentConfig {}

/// Environment types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EnvironmentType {
    Development,
    Testing,
    Staging,
    Production,
}

/// Debug configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugConfig {
    /// Enable debug mode
    pub enabled: bool,
    /// Debug level
    pub level: DebugLevel,
    /// Enable verbose output
    pub verbose: bool,
    /// Enable stack traces
    pub stack_traces: bool,
    /// Debug output format
    pub output_format: DebugOutputFormat,
}

impl Default for DebugConfig {
    fn default() -> Self {
        Self {
            enabled: cfg!(debug_assertions),
            level: DebugLevel::Info,
            verbose: false,
            stack_traces: cfg!(debug_assertions),
            output_format: DebugOutputFormat::Pretty,
        }
    }
}

/// Debug levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DebugLevel {
    None,
    Basic,
    Info,
    Verbose,
    Trace,
}

/// Debug output formats
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum DebugOutputFormat {
    Plain,
    Pretty,
    Json,
    Structured,
}

/// Configuration manager for handling unified configurations
pub struct ConfigManager {
    configs: HashMap<String, UnifiedConfig>,
    active_config: Option<String>,
}

impl ConfigManager {
    /// Create a new configuration manager
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
            active_config: None,
        }
    }

    /// Add a configuration
    pub fn add_config(&mut self, name: String, config: UnifiedConfig) -> Result<()> {
        config.validate()?;
        self.configs.insert(name, config);
        Ok(())
    }

    /// Get a configuration by name
    pub fn get_config(&self, name: &str) -> Option<&UnifiedConfig> {
        self.configs.get(name)
    }

    /// Set active configuration
    pub fn set_active(&mut self, name: String) -> Result<()> {
        if !self.configs.contains_key(&name) {
            return Err(anyhow::anyhow!("Configuration '{}' not found", name).into());
        }
        self.active_config = Some(name);
        Ok(())
    }

    /// Get active configuration
    pub fn get_active(&self) -> Option<&UnifiedConfig> {
        self.active_config.as_ref().and_then(|name| self.configs.get(name))
    }

    /// Merge configurations (later configs override earlier ones)
    pub fn merge_configs(&self, names: &[String]) -> Result<UnifiedConfig> {
        if names.is_empty() {
            return Ok(UnifiedConfig::default());
        }

        let mut result = self
            .get_config(&names[0])
            .ok_or_else(|| anyhow::anyhow!("Configuration '{}' not found", names[0]))?
            .clone();

        for name in &names[1..] {
            let config = self
                .get_config(name)
                .ok_or_else(|| anyhow::anyhow!("Configuration '{}' not found", name))?;
            result = merge_unified_configs(result, config.clone())?;
        }

        Ok(result)
    }

    /// Load configuration from file
    pub fn load_from_file(&mut self, name: String, path: &std::path::Path) -> Result<()> {
        let config = UnifiedConfig::load_from_file(path)?;
        self.add_config(name, config)
    }

    /// Save configuration to file
    pub fn save_to_file(&self, name: &str, path: &std::path::Path) -> Result<()> {
        let config = self
            .get_config(name)
            .ok_or_else(|| anyhow::anyhow!("Configuration '{}' not found", name))?;
        config.save_to_file(path)
    }
}

impl Default for ConfigManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Merge a *partial* override, expressed as JSON, on top of a base config.
///
/// This is the unambiguous merge: only the keys present in `overrides` change,
/// nested objects merge recursively, and arrays are replaced wholesale (a list
/// in an override is a complete specification of that list).
///
/// ```
/// use trustformers_core::patterns::{merge_unified_config_with_overrides, UnifiedConfig};
///
/// let mut base = UnifiedConfig::default();
/// base.metadata.name = "service".to_string();
///
/// let merged = merge_unified_config_with_overrides(
///     base,
///     serde_json::json!({"logging": {"level": "Warn"}}),
/// )
/// .expect("valid override");
///
/// // The override only mentioned the log level, so the name survives.
/// assert_eq!(merged.metadata.name, "service");
/// ```
pub fn merge_unified_config_with_overrides(
    base: UnifiedConfig,
    overrides: serde_json::Value,
) -> Result<UnifiedConfig> {
    let mut merged = serde_json::to_value(&base).map_err(|error| {
        crate::errors::TrustformersError::serialization_error(format!(
            "failed to serialize base config: {error}"
        ))
    })?;

    merge_json_values(&mut merged, overrides);

    serde_json::from_value(merged).map_err(|error| {
        crate::errors::TrustformersError::serialization_error(format!(
            "merged configuration is not a valid UnifiedConfig: {error}"
        ))
    })
}

/// Merge two fully-populated unified configurations.
///
/// `UnifiedConfig` has no optional fields, so an override struct always carries
/// a value for *every* setting; there is no "unset" to fall back on. This
/// function therefore treats a field whose override value equals
/// `UnifiedConfig::default()`'s value as "not specified" and keeps `base`'s
/// value for it; every other field takes the override.
///
/// The consequence, stated plainly: an override that deliberately sets a field
/// *back to the type default* cannot be distinguished from one that never
/// touched it. When that matters, use
/// [`merge_unified_config_with_overrides`], which takes a partial JSON
/// override and has no such ambiguity.
pub fn merge_unified_configs(
    base: UnifiedConfig,
    override_config: UnifiedConfig,
) -> Result<UnifiedConfig> {
    let serialize = |config: &UnifiedConfig, what: &str| -> Result<serde_json::Value> {
        serde_json::to_value(config).map_err(|error| {
            crate::errors::TrustformersError::serialization_error(format!(
                "failed to serialize {what} config: {error}"
            ))
        })
    };

    let mut merged = serialize(&base, "base")?;
    let overrides = serialize(&override_config, "override")?;
    let defaults = serialize(&UnifiedConfig::default(), "default")?;

    merge_json_against_defaults(&mut merged, &overrides, &defaults);

    serde_json::from_value(merged).map_err(|error| {
        crate::errors::TrustformersError::serialization_error(format!(
            "merged configuration is not a valid UnifiedConfig: {error}"
        ))
    })
}

/// Recursively merge `overrides` into `target`.
fn merge_json_values(target: &mut serde_json::Value, overrides: serde_json::Value) {
    match (target, overrides) {
        (serde_json::Value::Object(target_map), serde_json::Value::Object(override_map)) => {
            for (key, value) in override_map {
                match target_map.get_mut(&key) {
                    Some(existing) => merge_json_values(existing, value),
                    None => {
                        target_map.insert(key, value);
                    },
                }
            }
        },
        (target_slot, value) => *target_slot = value,
    }
}

/// Recursively merge `overrides` into `target`, skipping override values that
/// are indistinguishable from the type's default.
fn merge_json_against_defaults(
    target: &mut serde_json::Value,
    overrides: &serde_json::Value,
    defaults: &serde_json::Value,
) {
    match (target, overrides, defaults) {
        (
            serde_json::Value::Object(target_map),
            serde_json::Value::Object(override_map),
            serde_json::Value::Object(default_map),
        ) => {
            for (key, value) in override_map {
                let default_value = default_map.get(key);
                match target_map.get_mut(key) {
                    Some(existing) => match default_value {
                        Some(default_value) => {
                            merge_json_against_defaults(existing, value, default_value)
                        },
                        // No default to compare against: the override wins.
                        None => *existing = value.clone(),
                    },
                    None => {
                        target_map.insert(key.clone(), value.clone());
                    },
                }
            }
        },
        (target_slot, value, default_value) => {
            // A leaf that still holds the type default carries no information.
            if value != default_value {
                *target_slot = value.clone();
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: `merge_unified_configs` ignored `base` entirely and
    /// returned `override_config`, so layering a partial override wiped every
    /// unrelated section.
    #[test]
    fn test_merge_keeps_base_fields_the_override_does_not_change() -> Result<()> {
        let mut base = UnifiedConfig::default();
        base.metadata.name = "base-name".to_string();
        base.metadata.version = "1.2.3".to_string();
        base.resources.memory.max_heap_bytes = Some(4096);
        base.logging.level = LogLevel::Debug;

        // The override only cares about the logging level.
        let mut override_config = UnifiedConfig::default();
        override_config.logging.level = LogLevel::Warn;

        let merged = merge_unified_configs(base.clone(), override_config)?;

        assert_eq!(
            merged.logging.level,
            LogLevel::Warn,
            "the override must win"
        );
        assert_eq!(
            merged.metadata.name, "base-name",
            "a field the override never mentioned must survive"
        );
        assert_eq!(merged.metadata.version, "1.2.3");
        assert_eq!(
            merged.resources.memory.max_heap_bytes,
            Some(4096),
            "an untouched section must not be reset to its default"
        );

        Ok(())
    }

    /// The unambiguous partial-override API keeps everything it does not name.
    #[test]
    fn test_partial_json_override_keeps_the_rest_of_the_base() -> Result<()> {
        let mut base = UnifiedConfig::default();
        base.metadata.name = "base-name".to_string();
        base.resources.memory.max_heap_bytes = Some(4096);

        let merged = merge_unified_config_with_overrides(
            base,
            serde_json::json!({"logging": {"level": "Warn"}}),
        )?;

        assert_eq!(merged.logging.level, LogLevel::Warn);
        assert_eq!(merged.metadata.name, "base-name");
        assert_eq!(merged.resources.memory.max_heap_bytes, Some(4096));
        Ok(())
    }

    #[test]
    fn test_merge_json_values_is_recursive() {
        let mut target = serde_json::json!({
            "a": {"x": 1, "y": 2},
            "b": [1, 2, 3],
            "c": "keep",
        });
        merge_json_values(
            &mut target,
            serde_json::json!({"a": {"y": 20, "z": 30}, "b": [9]}),
        );

        assert_eq!(target["a"]["x"], 1, "untouched nested field survives");
        assert_eq!(target["a"]["y"], 20, "overridden nested field wins");
        assert_eq!(target["a"]["z"], 30, "new nested field is added");
        assert_eq!(target["b"], serde_json::json!([9]), "arrays are replaced");
        assert_eq!(target["c"], "keep");
    }
    use tempfile::tempdir;

    #[test]
    fn test_unified_config_creation() {
        let config = UnifiedConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.metadata.name, "default");
        assert!(config.metadata.enabled);
    }

    #[test]
    fn test_config_serialization() {
        let config = UnifiedConfig::default();
        let json = config.to_json().expect("operation failed in test");
        let deserialized = UnifiedConfig::from_json(&json).expect("operation failed in test");

        assert_eq!(config.metadata.name, deserialized.metadata.name);
        assert_eq!(config.metadata.enabled, deserialized.metadata.enabled);
    }

    #[test]
    fn test_config_manager() {
        let mut manager = ConfigManager::new();
        let config = UnifiedConfig::default();

        manager.add_config("test".to_string(), config).expect("add operation failed");
        manager.set_active("test".to_string()).expect("operation failed in test");

        let active = manager.get_active().expect("operation failed in test");
        assert_eq!(active.metadata.name, "default");
    }

    #[test]
    fn test_config_file_operations() {
        let temp_dir = tempdir().expect("temp file creation failed");
        let config_path = temp_dir.path().join("test_config.json");

        let config = UnifiedConfig::default();
        config.save_to_file(&config_path).expect("operation failed in test");

        let loaded_config =
            UnifiedConfig::load_from_file(&config_path).expect("operation failed in test");
        assert_eq!(config.metadata.name, loaded_config.metadata.name);
    }

    #[test]
    fn test_validation() {
        let mut config = UnifiedConfig::default();
        config.metadata.name = "".to_string(); // Invalid empty name

        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_resource_limits_validation() {
        let limits = MemoryLimits {
            warning_threshold_percent: 150.0, // Invalid percentage
            ..Default::default()
        };

        let result = limits.validate();
        assert!(result.is_err());
    }
}
