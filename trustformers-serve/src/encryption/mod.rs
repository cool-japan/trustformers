//! Encryption System Modules
//!
//! This module provides a comprehensive encryption system organized into focused
//! sub-modules for better maintainability and clarity.
//!
//! # What is real here
//!
//! [`service`] is the genuine AEAD service: RustCrypto AES-GCM and
//! ChaCha20-Poly1305 with OS-CSPRNG key material and real KDFs. [`cipher`] holds
//! the block/stream primitives, [`weight_encryption`] the model-weight envelope
//! format, and [`types`] the configuration surface those consume.
//!
//! # Removed in 0.2.1: the unsound orphan key-management cluster
//!
//! Roughly 9,000 lines of encryption code — `key_management.rs`,
//! `key_rotation.rs`, `filesystem_encryption.rs`, `secure_session.rs`, and the
//! `compliance/`, `database_encryption/`, `memory_encryption/` and
//! `performance/` subtrees — were deleted rather than wired in.
//!
//! They were never declared as modules, so they had never been compiled, were
//! unreachable from any public construction path, and had drifted out of sync
//! with the types they imported (`key_management.rs` no longer compiled at all).
//! More importantly they were not merely incomplete but actively dangerous:
//! every key-derivation and key-generation routine returned `vec![0u8; n]` —
//! an all-zero key — and the filesystem envelope used an all-zero IV and an
//! all-zero authentication tag. Mounting that code would have shipped
//! encryption that produces ciphertext anyone can decrypt, behind an API that
//! reports success. Shipping it dormant in a published crate was only slightly
//! better: it was dead weight that read like a supported feature.
//!
//! The real AEAD service in [`service`] already covers the working subset of
//! what that cluster claimed to do. Anything it does not cover — HSM-backed
//! master keys, scheduled key rotation, per-column database encryption — is
//! genuinely not implemented, and this crate now says so by not offering an API
//! for it, rather than by offering one backed by zero keys.

// Core modules
pub mod cipher;
pub mod errors;
pub mod service;
pub mod types;
pub mod weight_encryption;

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
