//! Encryption System Modules
//!
//! This module provides a comprehensive encryption system organized into focused
//! sub-modules for better maintainability and clarity.

// Core modules
pub mod errors;
pub mod service;
pub mod types;

// Re-export core types and errors for convenience
pub use errors::{EncryptionError, EncryptionResult, ErrorCategory};

// Legacy re-exports for backward compatibility
pub use types::{
    BackupLocation, ColumnEncryptionConfig, ComplianceConfig, ComplianceStandard, DEKCachingConfig,
    DEKConfig, DEKGenerationMethod, DEKLifecycleConfig, DatabaseEncryptionConfig,
    DatabaseEncryptionScope, DetectionAction, EncryptionAlgorithm, EncryptionConfig, EscrowAgent,
    EvictionPolicy, FilesystemEncryptionConfig, HSMConfig, HSMType, HardwareAcceleration,
    KeyBackupConfig, KeyDerivationConfig, KeyDerivationFunction, KeyEscrowConfig,
    KeyGenerationMethod, KeyManagementConfig, KeyManagementSystem, KeyRotationConfig,
    MasterKeyConfig, MemoryEncryptionConfig, MemoryWipingConfig, PerformanceConfig,
    RotationSchedule, RotationTrigger, SaltConfig, SaltGenerationMethod, SaltStorage,
    SensitiveDataDetection, SensitiveDataPattern, TableEncryptionConfig, WipingMethod,
};

/// Initialize the encryption system with default configuration
pub fn init_encryption_system() -> EncryptionResult<service::EncryptionService> {
    let config = EncryptionConfig::default();
    service::EncryptionService::new(config)
}

/// Validate encryption configuration
pub fn validate_encryption_config(config: &EncryptionConfig) -> EncryptionResult<()> {
    if !config.enabled {
        return Ok(());
    }

    if config.key_management.master_key.key_size < 128 {
        return Err(EncryptionError::config_error(
            "Master key size must be at least 128 bits",
        ));
    }

    Ok(())
}
