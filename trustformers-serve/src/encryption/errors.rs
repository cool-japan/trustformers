//! Error Types for Encryption System
//!
//! This module contains all error types used throughout the encryption system,
//! providing comprehensive error handling and diagnostics.

use thiserror::Error;

/// Comprehensive error type for the encryption system
#[derive(Error, Debug, Clone)]
pub enum EncryptionError {
    /// Configuration-related errors
    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },

    /// Key management errors
    #[error("Key not found: {key_id}")]
    KeyNotFound { key_id: String },

    #[error("Key expired: {key_id}")]
    KeyExpired { key_id: String },

    #[error("Key generation failed: {message}")]
    KeyGenerationFailed { message: String },

    #[error("Key derivation failed: {message}")]
    KeyDerivationFailed { message: String },

    #[error("Key validation failed: {message}")]
    KeyValidationFailed { message: String },

    /// Encryption/Decryption errors
    #[error("Encryption failed: {message}")]
    EncryptionFailed { message: String },

    #[error("Decryption failed: {message}")]
    DecryptionFailed { message: String },

    #[error("Authentication tag verification failed")]
    AuthenticationFailed,

    #[error("Invalid nonce/IV: {message}")]
    InvalidNonce { message: String },

    #[error("Unsupported algorithm: {algorithm}")]
    UnsupportedAlgorithm { algorithm: String },

    /// Key rotation and lifecycle errors
    #[error("Key rotation failed: {message}")]
    KeyRotationFailed { message: String },

    #[error("Key backup failed: {message}")]
    KeyBackupFailed { message: String },

    #[error("Key restore failed: {message}")]
    KeyRestoreFailed { message: String },

    #[error("Key lifecycle violation: {message}")]
    KeyLifecycleViolation { message: String },

    /// External KMS errors
    #[error("KMS error: {message}")]
    KMSError { message: String },

    #[error("AWS KMS error: {message}")]
    AWSKMSError { message: String },

    #[error("GCP KMS error: {message}")]
    GCPKMSError { message: String },

    #[error("Azure Key Vault error: {message}")]
    AzureKeyVaultError { message: String },

    #[error("HashiCorp Vault error: {message}")]
    VaultError { message: String },

    /// HSM errors
    #[error("HSM error: {message}")]
    HSMError { message: String },

    #[error("HSM connection failed: {message}")]
    HSMConnectionFailed { message: String },

    #[error("HSM authentication failed: {message}")]
    HSMAuthenticationFailed { message: String },

    #[error("PKCS#11 error: {message}")]
    PKCS11Error { message: String },

    /// Database encryption errors
    #[error("Database encryption error: {message}")]
    DatabaseEncryptionError { message: String },

    #[error("Column encryption failed: {column} - {message}")]
    ColumnEncryptionFailed { column: String, message: String },

    #[error("Table encryption failed: {table} - {message}")]
    TableEncryptionFailed { table: String, message: String },

    #[error("Sensitive data detection failed: {message}")]
    SensitiveDataDetectionFailed { message: String },

    /// Filesystem encryption errors
    #[error("Filesystem encryption error: {message}")]
    FilesystemEncryptionError { message: String },

    #[error("File encryption failed: {path} - {message}")]
    FileEncryptionFailed { path: String, message: String },

    #[error("File decryption failed: {path} - {message}")]
    FileDecryptionFailed { path: String, message: String },

    #[error("Path validation failed: {path} - {message}")]
    PathValidationFailed { path: String, message: String },

    /// Memory encryption errors
    #[error("Memory encryption error: {message}")]
    MemoryEncryptionError { message: String },

    #[error("Memory wiping failed: {message}")]
    MemoryWipingFailed { message: String },

    #[error("Memory protection failed: {message}")]
    MemoryProtectionFailed { message: String },

    #[error("Memory access violation: {message}")]
    MemoryAccessViolation { message: String },

    /// Compliance and audit errors
    #[error("Compliance violation: {standard} - {message}")]
    ComplianceViolation { standard: String, message: String },

    #[error("Audit log failed: {message}")]
    AuditLogFailed { message: String },

    #[error("Compliance check failed: {message}")]
    ComplianceCheckFailed { message: String },

    #[error("Data classification failed: {message}")]
    DataClassificationFailed { message: String },

    /// Performance and hardware errors
    #[error("Hardware acceleration error: {message}")]
    HardwareAccelerationError { message: String },

    #[error("AES-NI not available")]
    AESNINotAvailable,

    #[error("Performance optimization failed: {message}")]
    PerformanceOptimizationFailed { message: String },

    #[error("Cache error: {message}")]
    CacheError { message: String },

    /// I/O and system errors
    #[error("I/O error: {message}")]
    IOError { message: String },

    #[error("Network error: {message}")]
    NetworkError { message: String },

    #[error("Timeout error: {operation} - {timeout_ms}ms")]
    TimeoutError { operation: String, timeout_ms: u64 },

    #[error("Resource unavailable: {resource}")]
    ResourceUnavailable { resource: String },

    /// Validation and format errors
    #[error("Invalid input format: {message}")]
    InvalidInputFormat { message: String },

    #[error("Data corruption detected: {message}")]
    DataCorruption { message: String },

    #[error("Checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: String, actual: String },

    #[error("Invalid parameter: {parameter} - {message}")]
    InvalidParameter { parameter: String, message: String },

    /// Concurrency and synchronization errors
    #[error("Lock acquisition failed: {message}")]
    LockAcquisitionFailed { message: String },

    #[error("Deadlock detected: {message}")]
    DeadlockDetected { message: String },

    #[error("Race condition detected: {message}")]
    RaceConditionDetected { message: String },

    /// Metrics and monitoring errors
    #[error("Metrics collection failed: {message}")]
    MetricsCollectionFailed { message: String },

    #[error("Monitoring system error: {message}")]
    MonitoringSystemError { message: String },

    /// Generic system errors
    #[error("System error: {message}")]
    SystemError { message: String },

    #[error("Internal error: {message}")]
    InternalError { message: String },

    #[error("Operation not supported: {operation}")]
    OperationNotSupported { operation: String },

    #[error("Service unavailable: {service}")]
    ServiceUnavailable { service: String },
}

impl EncryptionError {
    /// Create a configuration error
    pub fn config_error(message: impl Into<String>) -> Self {
        Self::ConfigurationError {
            message: message.into(),
        }
    }

    /// Create a key not found error
    pub fn key_not_found(key_id: impl Into<String>) -> Self {
        Self::KeyNotFound {
            key_id: key_id.into(),
        }
    }

    /// Create an encryption failed error
    pub fn encryption_failed(message: impl Into<String>) -> Self {
        Self::EncryptionFailed {
            message: message.into(),
        }
    }

    /// Create a decryption failed error
    pub fn decryption_failed(message: impl Into<String>) -> Self {
        Self::DecryptionFailed {
            message: message.into(),
        }
    }

    /// Create a KMS error
    pub fn kms_error(message: impl Into<String>) -> Self {
        Self::KMSError {
            message: message.into(),
        }
    }

    /// Create an HSM error
    pub fn hsm_error(message: impl Into<String>) -> Self {
        Self::HSMError {
            message: message.into(),
        }
    }

    /// Create a system error
    pub fn system_error(message: impl Into<String>) -> Self {
        Self::SystemError {
            message: message.into(),
        }
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            // Recoverable errors (transient issues)
            Self::TimeoutError { .. }
            | Self::NetworkError { .. }
            | Self::ResourceUnavailable { .. }
            | Self::ServiceUnavailable { .. }
            | Self::LockAcquisitionFailed { .. }
            | Self::CacheError { .. } => true,

            // Non-recoverable errors (permanent issues)
            Self::KeyExpired { .. }
            | Self::AuthenticationFailed
            | Self::UnsupportedAlgorithm { .. }
            | Self::DataCorruption { .. }
            | Self::ComplianceViolation { .. }
            | Self::OperationNotSupported { .. } => false,

            // Context-dependent errors
            _ => false,
        }
    }

    /// Get the error category
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::ConfigurationError { .. } => ErrorCategory::Configuration,

            Self::KeyNotFound { .. }
            | Self::KeyExpired { .. }
            | Self::KeyGenerationFailed { .. }
            | Self::KeyDerivationFailed { .. }
            | Self::KeyValidationFailed { .. }
            | Self::KeyRotationFailed { .. }
            | Self::KeyBackupFailed { .. }
            | Self::KeyRestoreFailed { .. }
            | Self::KeyLifecycleViolation { .. } => ErrorCategory::KeyManagement,

            Self::EncryptionFailed { .. }
            | Self::DecryptionFailed { .. }
            | Self::AuthenticationFailed
            | Self::InvalidNonce { .. }
            | Self::UnsupportedAlgorithm { .. } => ErrorCategory::Cryptography,

            Self::KMSError { .. }
            | Self::AWSKMSError { .. }
            | Self::GCPKMSError { .. }
            | Self::AzureKeyVaultError { .. }
            | Self::VaultError { .. } => ErrorCategory::KMS,

            Self::HSMError { .. }
            | Self::HSMConnectionFailed { .. }
            | Self::HSMAuthenticationFailed { .. }
            | Self::PKCS11Error { .. } => ErrorCategory::HSM,

            Self::DatabaseEncryptionError { .. }
            | Self::ColumnEncryptionFailed { .. }
            | Self::TableEncryptionFailed { .. }
            | Self::SensitiveDataDetectionFailed { .. } => ErrorCategory::Database,

            Self::FilesystemEncryptionError { .. }
            | Self::FileEncryptionFailed { .. }
            | Self::FileDecryptionFailed { .. }
            | Self::PathValidationFailed { .. } => ErrorCategory::Filesystem,

            Self::MemoryEncryptionError { .. }
            | Self::MemoryWipingFailed { .. }
            | Self::MemoryProtectionFailed { .. }
            | Self::MemoryAccessViolation { .. } => ErrorCategory::Memory,

            Self::ComplianceViolation { .. }
            | Self::AuditLogFailed { .. }
            | Self::ComplianceCheckFailed { .. }
            | Self::DataClassificationFailed { .. } => ErrorCategory::Compliance,

            Self::HardwareAccelerationError { .. }
            | Self::AESNINotAvailable
            | Self::PerformanceOptimizationFailed { .. }
            | Self::CacheError { .. } => ErrorCategory::Performance,

            Self::IOError { .. }
            | Self::NetworkError { .. }
            | Self::TimeoutError { .. }
            | Self::ResourceUnavailable { .. } => ErrorCategory::IO,

            Self::InvalidInputFormat { .. }
            | Self::DataCorruption { .. }
            | Self::ChecksumMismatch { .. }
            | Self::InvalidParameter { .. } => ErrorCategory::Validation,

            Self::LockAcquisitionFailed { .. }
            | Self::DeadlockDetected { .. }
            | Self::RaceConditionDetected { .. } => ErrorCategory::Concurrency,

            Self::MetricsCollectionFailed { .. } | Self::MonitoringSystemError { .. } => {
                ErrorCategory::Monitoring
            },

            Self::SystemError { .. }
            | Self::InternalError { .. }
            | Self::OperationNotSupported { .. }
            | Self::ServiceUnavailable { .. } => ErrorCategory::System,
        }
    }

    /// Get a user-friendly error message
    pub fn user_message(&self) -> String {
        match self {
            Self::AuthenticationFailed => {
                "Authentication failed. Please check your credentials.".to_string()
            },
            Self::KeyExpired { key_id } => format!(
                "The encryption key '{}' has expired and needs to be rotated.",
                key_id
            ),
            Self::UnsupportedAlgorithm { algorithm } => {
                format!("The encryption algorithm '{}' is not supported.", algorithm)
            },
            Self::ComplianceViolation { standard, .. } => {
                format!("Operation violates {} compliance requirements.", standard)
            },
            _ => self.to_string(),
        }
    }
}

/// Error categories for classification and handling
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorCategory {
    Configuration,
    KeyManagement,
    Cryptography,
    KMS,
    HSM,
    Database,
    Filesystem,
    Memory,
    Compliance,
    Performance,
    IO,
    Validation,
    Concurrency,
    Monitoring,
    System,
}

/// Result type alias for encryption operations
pub type EncryptionResult<T> = Result<T, EncryptionError>;

/// Convert common error types to EncryptionError
impl From<std::io::Error> for EncryptionError {
    fn from(err: std::io::Error) -> Self {
        Self::IOError {
            message: err.to_string(),
        }
    }
}

impl From<serde_json::Error> for EncryptionError {
    fn from(err: serde_json::Error) -> Self {
        Self::InvalidInputFormat {
            message: format!("JSON error: {}", err),
        }
    }
}

impl From<tokio::time::error::Elapsed> for EncryptionError {
    fn from(_err: tokio::time::error::Elapsed) -> Self {
        Self::TimeoutError {
            operation: "Unknown".to_string(),
            timeout_ms: 0, // Will be set by caller
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = EncryptionError::key_not_found("test-key-123");
        assert!(matches!(error, EncryptionError::KeyNotFound { .. }));
        assert_eq!(error.category(), ErrorCategory::KeyManagement);
    }

    #[test]
    fn test_error_recoverability() {
        let recoverable = EncryptionError::TimeoutError {
            operation: "encryption".to_string(),
            timeout_ms: 5000,
        };
        assert!(recoverable.is_recoverable());

        let non_recoverable = EncryptionError::AuthenticationFailed;
        assert!(!non_recoverable.is_recoverable());
    }

    #[test]
    fn test_error_categories() {
        let config_error = EncryptionError::config_error("Invalid setting");
        assert_eq!(config_error.category(), ErrorCategory::Configuration);

        let crypto_error = EncryptionError::encryption_failed("Bad key");
        assert_eq!(crypto_error.category(), ErrorCategory::Cryptography);

        let kms_error = EncryptionError::kms_error("Connection failed");
        assert_eq!(kms_error.category(), ErrorCategory::KMS);
    }

    #[test]
    fn test_user_messages() {
        let auth_error = EncryptionError::AuthenticationFailed;
        let user_msg = auth_error.user_message();
        assert!(user_msg.contains("Authentication failed"));

        let expired_error = EncryptionError::KeyExpired {
            key_id: "test-key".to_string(),
        };
        let user_msg = expired_error.user_message();
        assert!(user_msg.contains("expired"));
        assert!(user_msg.contains("test-key"));
    }

    #[test]
    fn test_error_conversions() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
        let encryption_error: EncryptionError = io_error.into();
        assert!(matches!(encryption_error, EncryptionError::IOError { .. }));
    }
}
