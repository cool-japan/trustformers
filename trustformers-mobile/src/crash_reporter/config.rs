//! Configuration types for mobile crash reporting: privacy, storage, analysis, reporting, recovery and platform-specific (iOS/Android) settings.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Android-specific crash configuration
#[cfg(target_os = "android")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AndroidCrashConfig {
    /// Enable native crash handling
    pub native_crashes: bool,
    /// Enable Java crash handling
    pub java_crashes: bool,
    /// Enable ANR (Application Not Responding) detection
    pub anr_detection: bool,
    /// Include Google Play compliance
    pub play_store_compliance: bool,
    /// Tombstone integration
    pub tombstone_integration: bool,
}
#[cfg(target_os = "android")]
impl Default for AndroidCrashConfig {
    fn default() -> Self {
        Self {
            native_crashes: true,
            java_crashes: true,
            anr_detection: true,
            play_store_compliance: true,
            tombstone_integration: true,
        }
    }
}
/// Authentication types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthType {
    ApiKey,
    BearerToken,
    Custom,
    None,
}
/// Analysis configuration for crash reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrashAnalysisConfig {
    /// Enable automatic crash analysis
    pub auto_analyze: bool,
    /// Enable pattern detection
    pub pattern_detection: bool,
    /// Enable similarity analysis
    pub similarity_analysis: bool,
    /// Enable recovery suggestions
    pub recovery_suggestions: bool,
    /// Analysis timeout (seconds)
    pub analysis_timeout_secs: u64,
    /// Enable ML-based analysis
    pub ml_analysis: bool,
}
impl Default for CrashAnalysisConfig {
    fn default() -> Self {
        Self {
            auto_analyze: true,
            pattern_detection: true,
            similarity_analysis: true,
            recovery_suggestions: true,
            analysis_timeout_secs: 30,
            ml_analysis: false,
        }
    }
}
/// Privacy configuration for crash data collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrashPrivacyConfig {
    /// Enable automatic crash reporting
    pub auto_report: bool,
    /// Collect user consent before reporting
    pub require_consent: bool,
    /// Include user data in reports
    pub include_user_data: bool,
    /// Include system information
    pub include_system_info: bool,
    /// Include stack traces
    pub include_stack_traces: bool,
    /// Include memory dumps
    pub include_memory_dumps: bool,
    /// Anonymize sensitive data
    pub anonymize_data: bool,
    /// Data retention policy
    pub retention_days: u32,
}
impl Default for CrashPrivacyConfig {
    fn default() -> Self {
        Self {
            auto_report: false,
            require_consent: true,
            include_user_data: false,
            include_system_info: true,
            include_stack_traces: true,
            include_memory_dumps: false,
            anonymize_data: true,
            retention_days: 30,
        }
    }
}
/// Recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrashRecoveryConfig {
    /// Enable automatic recovery
    pub auto_recovery: bool,
    /// Recovery strategies
    pub recovery_strategies: Vec<RecoveryStrategy>,
    /// Recovery timeout (seconds)
    pub recovery_timeout_secs: u64,
    /// Enable state restoration
    pub state_restoration: bool,
    /// Safe mode configuration
    pub safe_mode_config: SafeModeConfig,
}
impl Default for CrashRecoveryConfig {
    fn default() -> Self {
        Self {
            auto_recovery: true,
            recovery_strategies: vec![
                RecoveryStrategy::ClearCache,
                RecoveryStrategy::ResetModel,
                RecoveryStrategy::SafeMode,
            ],
            recovery_timeout_secs: 60,
            state_restoration: true,
            safe_mode_config: SafeModeConfig::default(),
        }
    }
}
/// Configuration for crash reporting system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrashReporterConfig {
    /// Enable crash reporting
    pub enabled: bool,
    /// Privacy mode configuration
    pub privacy_config: CrashPrivacyConfig,
    /// Storage configuration
    pub storage_config: CrashStorageConfig,
    /// Analysis configuration
    pub analysis_config: CrashAnalysisConfig,
    /// Reporting configuration
    pub reporting_config: CrashReportingConfig,
    /// Recovery configuration
    pub recovery_config: CrashRecoveryConfig,
    /// Platform-specific settings
    pub platform_config: PlatformCrashConfig,
}
impl Default for CrashReporterConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            privacy_config: CrashPrivacyConfig::default(),
            storage_config: CrashStorageConfig::default(),
            analysis_config: CrashAnalysisConfig::default(),
            reporting_config: CrashReportingConfig::default(),
            recovery_config: CrashRecoveryConfig::default(),
            platform_config: PlatformCrashConfig::default(),
        }
    }
}
/// Reporting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrashReportingConfig {
    /// Enable remote reporting
    pub remote_reporting: bool,
    /// Remote endpoint URL
    pub remote_endpoint: Option<String>,
    /// Authentication configuration
    pub auth_config: Option<ReportingAuthConfig>,
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// Batch reporting
    pub batch_reporting: bool,
    /// Batch size
    pub batch_size: usize,
}
impl Default for CrashReportingConfig {
    fn default() -> Self {
        Self {
            remote_reporting: false,
            remote_endpoint: None,
            auth_config: None,
            retry_config: RetryConfig::default(),
            batch_reporting: true,
            batch_size: 10,
        }
    }
}
/// Storage configuration for crash reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrashStorageConfig {
    /// Local storage directory
    pub storage_directory: PathBuf,
    /// Maximum crash reports to store locally
    pub max_local_reports: usize,
    /// Maximum storage size (MB)
    pub max_storage_size_mb: usize,
    /// Compress stored reports
    pub compress_reports: bool,
    /// Enable encryption
    pub encrypt_reports: bool,
    /// Encryption key source
    pub encryption_key_source: EncryptionKeySource,
}
impl Default for CrashStorageConfig {
    fn default() -> Self {
        Self {
            storage_directory: PathBuf::from("crash_reports"),
            max_local_reports: 100,
            max_storage_size_mb: 50,
            compress_reports: true,
            encrypt_reports: true,
            encryption_key_source: EncryptionKeySource::Keychain,
        }
    }
}
/// Encryption key source
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncryptionKeySource {
    /// Hardware-based key
    Hardware,
    /// Keychain/Keystore
    Keychain,
    /// Generated key
    Generated,
    /// No encryption
    None,
}
/// iOS-specific crash configuration
#[cfg(target_os = "ios")]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IOSCrashConfig {
    /// Enable Mach exception handling
    pub mach_exceptions: bool,
    /// Enable BSD signal handling
    pub bsd_signals: bool,
    /// Enable C++ exception handling
    pub cpp_exceptions: bool,
    /// Include app store compliance features
    pub app_store_compliance: bool,
    /// Privacy manifest integration
    pub privacy_manifest: bool,
}
#[cfg(target_os = "ios")]
impl Default for IOSCrashConfig {
    fn default() -> Self {
        Self {
            mach_exceptions: true,
            bsd_signals: true,
            cpp_exceptions: true,
            app_store_compliance: true,
            privacy_manifest: true,
        }
    }
}
/// Platform-specific crash configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlatformCrashConfig {
    /// iOS-specific configuration
    #[cfg(target_os = "ios")]
    pub ios_config: IOSCrashConfig,
    /// Android-specific configuration
    #[cfg(target_os = "android")]
    pub android_config: AndroidCrashConfig,
    /// Signal handling configuration
    pub signal_config: SignalHandlingConfig,
}
/// Recovery strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Restart application
    RestartApp,
    /// Reset model state
    ResetModel,
    /// Clear cache
    ClearCache,
    /// Safe mode
    SafeMode,
    /// Reduced functionality
    ReducedFunctionality,
    /// Manual intervention
    ManualIntervention,
}
/// Authentication configuration for remote reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportingAuthConfig {
    /// Authentication type
    pub auth_type: AuthType,
    /// API key
    pub api_key: Option<String>,
    /// Bearer token
    pub bearer_token: Option<String>,
    /// Custom headers
    pub custom_headers: HashMap<String, String>,
}
/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_attempts: usize,
    /// Base delay (ms)
    pub base_delay_ms: u64,
    /// Maximum delay (ms)
    pub max_delay_ms: u64,
    /// Exponential backoff multiplier
    pub backoff_multiplier: f64,
    /// Jitter factor
    pub jitter_factor: f64,
}
impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay_ms: 1000,
            max_delay_ms: 30000,
            backoff_multiplier: 2.0,
            jitter_factor: 0.1,
        }
    }
}
/// Safe mode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafeModeConfig {
    /// Enable safe mode
    pub enabled: bool,
    /// Disable GPU acceleration
    pub disable_gpu: bool,
    /// Reduce memory usage
    pub reduce_memory: bool,
    /// Disable advanced features
    pub disable_advanced_features: bool,
    /// Safe mode timeout (seconds)
    pub timeout_secs: u64,
}
impl Default for SafeModeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            disable_gpu: true,
            reduce_memory: true,
            disable_advanced_features: true,
            timeout_secs: 300,
        }
    }
}
/// Signal handling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalHandlingConfig {
    /// Signals to handle
    pub handled_signals: Vec<i32>,
    /// Stack trace depth
    pub stack_trace_depth: usize,
    /// Signal handler timeout (ms)
    pub handler_timeout_ms: u64,
    /// Enable async-safe handling
    pub async_safe: bool,
}
impl Default for SignalHandlingConfig {
    fn default() -> Self {
        Self {
            handled_signals: vec![
                libc::SIGSEGV,
                libc::SIGABRT,
                libc::SIGBUS,
                libc::SIGFPE,
                libc::SIGILL,
                libc::SIGPIPE,
            ],
            stack_trace_depth: 50,
            handler_timeout_ms: 5000,
            async_safe: true,
        }
    }
}
