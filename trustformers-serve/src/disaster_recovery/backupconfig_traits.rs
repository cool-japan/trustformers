//! # BackupConfig - Trait Implementations
//!
//! This module contains trait implementations for `BackupConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{
    BackupConfig, BackupStrategy, BackupTarget, CompressionAlgorithm, CompressionConfig,
    EncryptionAlgorithm, EncryptionConfig, RetentionPolicy, StorageType, VerificationConfig,
};

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strategy: BackupStrategy::Incremental {
                full_backup_interval: Duration::from_secs(7 * 24 * 3600),
                incremental_interval: Duration::from_secs(3600),
            },
            targets: vec![BackupTarget {
                target_id: "s3-backup".to_string(),
                storage_type: StorageType::S3 {
                    bucket: "disaster-recovery-backups".to_string(),
                    region: "us-west-2".to_string(),
                    access_key_id: "backup-access-key".to_string(),
                    secret_access_key: "backup-secret-key".to_string(),
                },
                path: "/backups/trustformers-serve".to_string(),
                priority: 1,
            }],
            retention: RetentionPolicy {
                daily_backups: 30,
                weekly_backups: 12,
                monthly_backups: 12,
                yearly_backups: 5,
            },
            compression: CompressionConfig {
                enabled: true,
                algorithm: CompressionAlgorithm::Gzip,
                level: 6,
            },
            encryption: EncryptionConfig {
                enabled: true,
                algorithm: EncryptionAlgorithm::AES256,
                key_id: "backup-encryption-key".to_string(),
            },
            verification: VerificationConfig {
                enabled: true,
                verify_after_backup: true,
                periodic_verification: true,
                verification_interval: Duration::from_secs(24 * 3600),
            },
        }
    }
}
