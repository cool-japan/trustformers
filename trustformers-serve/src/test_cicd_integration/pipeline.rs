//! Pipeline integration configuration

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

/// Pipeline integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineIntegrationConfig {
    /// Integration enabled
    pub enabled: bool,

    /// Integration types
    pub integration_types: Vec<PipelineIntegrationType>,

    /// Integration settings
    pub settings: IntegrationSettings,

    /// Rate limiting
    pub rate_limiting: RateLimitConfig,

    /// Error handling
    pub error_handling: ErrorHandlingConfig,

    /// Hooks configuration
    pub hooks: HookConfig,

    /// Artifact management
    pub artifact_management: ArtifactManagementConfig,

    /// Notification configuration
    pub notifications: super::environment::NotificationConfig,
}

/// Pipeline integration types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PipelineIntegrationType {
    /// Webhook integration
    Webhook { url: String, secret: Option<String> },

    /// GitHub Actions integration
    GitHubActions { repository: String, token: String },

    /// GitLab CI integration
    GitLabCi { project_id: String, token: String },

    /// Jenkins integration
    Jenkins {
        url: String,
        username: String,
        token: String,
    },

    /// Azure DevOps integration
    AzureDevOps {
        organization: String,
        project: String,
        token: String,
    },

    /// CircleCI integration
    CircleCi { project: String, token: String },

    /// Travis CI integration
    TravisCi { repository: String, token: String },

    /// Buildkite integration
    Buildkite {
        organization: String,
        pipeline: String,
        token: String,
    },

    /// TeamCity integration
    TeamCity {
        server: String,
        username: String,
        password: String,
    },

    /// Custom integration
    Custom {
        name: String,
        config: HashMap<String, String>,
    },
}

/// Integration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationSettings {
    /// Connection timeout
    pub connection_timeout: Duration,

    /// Request timeout
    pub request_timeout: Duration,

    /// Maximum concurrent connections
    pub max_concurrent_connections: usize,

    /// Retry configuration
    pub retry_config: super::environment::RetryConfig,

    /// Authentication configuration
    pub auth_config: super::environment::AuthConfig,

    /// Custom headers
    pub custom_headers: HashMap<String, String>,
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Rate limiting enabled
    pub enabled: bool,

    /// Rate limit strategy
    pub strategy: RateLimitStrategy,

    /// Requests per minute
    pub requests_per_minute: u64,

    /// Burst capacity
    pub burst_capacity: u64,

    /// Rate limit window
    pub window_duration: Duration,
}

/// Rate limiting strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateLimitStrategy {
    /// Token bucket strategy
    TokenBucket,

    /// Fixed window strategy
    FixedWindow,

    /// Sliding window strategy
    SlidingWindow,

    /// Leaky bucket strategy
    LeakyBucket,

    /// Custom strategy
    Custom(String),
}

/// Error handling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorHandlingConfig {
    /// Error handling enabled
    pub enabled: bool,

    /// Maximum retry attempts
    pub max_retry_attempts: usize,

    /// Retry delay
    pub retry_delay: Duration,

    /// Fallback actions
    pub fallback_actions: Vec<FallbackAction>,

    /// Circuit breaker enabled
    pub circuit_breaker_enabled: bool,

    /// Circuit breaker threshold
    pub circuit_breaker_threshold: u32,

    /// Circuit breaker timeout
    pub circuit_breaker_timeout: Duration,
}

/// Fallback actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FallbackAction {
    /// Log error and continue
    LogAndContinue,

    /// Send notification
    SendNotification { channel: String },

    /// Use default configuration
    UseDefaultConfig,

    /// Skip integration
    SkipIntegration,

    /// Fail fast
    FailFast,

    /// Custom action
    Custom(String),
}

/// Hook configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookConfig {
    /// Hooks enabled
    pub enabled: bool,

    /// Pre-test hooks
    pub pre_test_hooks: Vec<HookType>,

    /// Post-test hooks
    pub post_test_hooks: Vec<HookType>,

    /// Pre-optimization hooks
    pub pre_optimization_hooks: Vec<HookType>,

    /// Post-optimization hooks
    pub post_optimization_hooks: Vec<HookType>,

    /// Error hooks
    pub error_hooks: Vec<HookType>,

    /// Hook timeout
    pub hook_timeout: Duration,
}

/// Hook types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HookType {
    /// Shell command hook
    ShellCommand { command: String, args: Vec<String> },

    /// HTTP webhook
    HttpWebhook {
        url: String,
        method: String,
        headers: HashMap<String, String>,
    },

    /// Script hook
    Script { language: String, script: String },

    /// Function hook
    Function {
        name: String,
        parameters: HashMap<String, serde_json::Value>,
    },

    /// Plugin hook
    Plugin {
        plugin_name: String,
        config: HashMap<String, String>,
    },

    /// Custom hook
    Custom(String),
}

/// Hook triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HookTrigger {
    /// Before test execution
    BeforeTest,

    /// After test execution
    AfterTest,

    /// Before optimization
    BeforeOptimization,

    /// After optimization
    AfterOptimization,

    /// On test success
    OnTestSuccess,

    /// On test failure
    OnTestFailure,

    /// On optimization success
    OnOptimizationSuccess,

    /// On optimization failure
    OnOptimizationFailure,

    /// On error
    OnError,

    /// Custom trigger
    Custom(String),
}

/// Artifact management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactManagementConfig {
    /// Artifact management enabled
    pub enabled: bool,

    /// Storage configuration
    pub storage: ArtifactStorageConfig,

    /// Artifact types
    pub artifact_types: HashMap<String, ArtifactTypeConfig>,

    /// Retention policy
    pub retention: RetentionPolicy,

    /// Compression settings
    pub compression: CompressionSettings,

    /// Upload settings
    pub upload: UploadSettings,

    /// Encryption settings
    pub encryption: EncryptionSettings,
}

/// Artifact storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactStorageConfig {
    /// Storage type
    pub storage_type: ArtifactStorageType,

    /// Base path
    pub base_path: String,

    /// Maximum storage size
    pub max_storage_size: Option<u64>,

    /// Storage quota per project
    pub quota_per_project: Option<u64>,
}

/// Artifact storage types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactStorageType {
    /// Local file system
    Local { path: String },

    /// AWS S3
    AwsS3 { bucket: String, region: String },

    /// Azure Blob Storage
    AzureBlob { account: String, container: String },

    /// Google Cloud Storage
    GoogleCloudStorage { bucket: String, project: String },

    /// FTP/SFTP
    Ftp {
        host: String,
        username: String,
        password: String,
    },

    /// HTTP/HTTPS
    Http { base_url: String },

    /// Custom storage
    Custom(String),
}

/// Artifact type configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactTypeConfig {
    /// Artifact name pattern
    pub name_pattern: String,

    /// File extensions
    pub file_extensions: Vec<String>,

    /// Maximum file size
    pub max_file_size: Option<u64>,

    /// Compression enabled
    pub compression_enabled: bool,

    /// Encryption required
    pub encryption_required: bool,

    /// Artifact priority
    pub priority: ArtifactPriority,

    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

/// Artifact priority levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Default retention period
    pub default_retention_days: u32,

    /// Retention rules
    pub rules: Vec<RetentionRule>,

    /// Auto-cleanup enabled
    pub auto_cleanup: bool,

    /// Cleanup schedule
    pub cleanup_schedule: String,
}

/// Retention rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionRule {
    /// Rule pattern
    pub pattern: String,

    /// Retention period (days)
    pub retention_days: u32,

    /// Action on expiration
    pub action: RetentionAction,
}

/// Retention actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RetentionAction {
    /// Delete artifact
    Delete,

    /// Archive artifact
    Archive,

    /// Move to different storage
    Move { destination: String },

    /// Compress artifact
    Compress,

    /// Custom action
    Custom(String),
}

/// Compression settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionSettings {
    /// Compression enabled
    pub enabled: bool,

    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,

    /// Compression level (0-9)
    pub level: u8,

    /// Minimum file size for compression
    pub min_file_size: u64,
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Gzip,
    Zstd,
    Lz4,
    Bzip2,
    Xz,
}

/// Upload settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadSettings {
    /// Parallel uploads enabled
    pub parallel_uploads: bool,

    /// Maximum concurrent uploads
    pub max_concurrent_uploads: usize,

    /// Upload chunk size
    pub chunk_size: u64,

    /// Upload timeout
    pub timeout: Duration,

    /// Upload validation
    pub validation: UploadValidation,
}

/// Upload validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadValidation {
    /// Validation enabled
    pub enabled: bool,

    /// Checksum validation
    pub checksum_validation: bool,

    /// Size validation
    pub size_validation: bool,

    /// Validation rules
    pub rules: Vec<ValidationRule>,
}

/// Validation rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    /// Rule name
    pub name: String,

    /// Rule condition
    pub condition: String,

    /// Validation action
    pub action: ValidationAction,
}

/// Validation actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationAction {
    /// Accept upload
    Accept,

    /// Reject upload
    Reject,

    /// Quarantine upload
    Quarantine,

    /// Transform upload
    Transform { transformation: String },

    /// Custom action
    Custom(String),
}

/// Encryption settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionSettings {
    /// Encryption enabled
    pub enabled: bool,

    /// Encryption algorithm
    pub algorithm: EncryptionAlgorithm,

    /// Key management
    pub key_management: KeyManagement,

    /// Encryption scope
    pub scope: Vec<String>,
}

/// Encryption algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    Aes256,
    ChaCha20,
    Aes128,
}

/// Key management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyManagement {
    /// Static key
    Static { key: String },

    /// Key derivation
    KeyDerivation { passphrase: String },

    /// External key management
    External { service: String, key_id: String },
}
