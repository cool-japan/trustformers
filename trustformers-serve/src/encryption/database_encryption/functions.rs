//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{DatabaseEncryptionManager, SensitiveDataDetector, TableEncryptionManager, TableEncryptionStatus};

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_database_encryption_manager_creation() {
        let config = DatabaseEncryptionConfig::default();
        let master_key_manager = Arc::new(
            crate::encryption::key_management::MasterKeyManager::new(
                crate::encryption::types::MasterKeyConfig::default(),
                Arc::new(crate::encryption::key_management::InMemoryKMS::new()),
                None,
            ),
        );
        let dek_manager = Arc::new(
            crate::encryption::key_management::DataEncryptionKeyManager::new(
                crate::encryption::types::DEKConfig::default(),
                Arc::clone(&master_key_manager),
                Arc::new(
                    crate::encryption::key_management::KeyDerivationManager::new(
                        crate::encryption::types::KeyDerivationConfig::default(),
                        Arc::new(
                            crate::encryption::key_management::InMemorySaltStorage::new(),
                        ),
                    ),
                ),
            ),
        );
        let db_encryption_manager = DatabaseEncryptionManager::new(config, dek_manager);
        assert!(db_encryption_manager.config.enabled);
    }
    #[tokio::test]
    async fn test_sensitive_data_detector() {
        let config = SensitiveDataDetection::default();
        let detector = SensitiveDataDetector::new(config);
        detector.start().await.expect("async operation should succeed in test");
        let result = detector
            .scan_data("Contact me at john@example.com", "test_context")
            .await;
        assert!(result.is_ok());
    }
    #[tokio::test]
    async fn test_table_encryption_status() {
        let config = TableEncryptionConfig::default();
        let master_key_manager = Arc::new(
            crate::encryption::key_management::MasterKeyManager::new(
                crate::encryption::types::MasterKeyConfig::default(),
                Arc::new(crate::encryption::key_management::InMemoryKMS::new()),
                None,
            ),
        );
        let dek_manager = Arc::new(
            crate::encryption::key_management::DataEncryptionKeyManager::new(
                crate::encryption::types::DEKConfig::default(),
                Arc::clone(&master_key_manager),
                Arc::new(
                    crate::encryption::key_management::KeyDerivationManager::new(
                        crate::encryption::types::KeyDerivationConfig::default(),
                        Arc::new(
                            crate::encryption::key_management::InMemorySaltStorage::new(),
                        ),
                    ),
                ),
            ),
        );
        let table_manager = TableEncryptionManager::new(config, dek_manager);
        table_manager.start().await.expect("async operation should succeed in test");
        let result = table_manager
            .set_table_status("test_table", TableEncryptionStatus::Encrypted)
            .await;
        assert!(result.is_ok());
    }
}
