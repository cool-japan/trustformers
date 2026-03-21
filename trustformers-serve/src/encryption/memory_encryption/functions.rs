//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::types::{DefaultHardwareProtection, HardwareProtectionHandle, MemoryEncryptionManager, SecureMemoryManager, WipingPatterns};

/// Hardware protection interface
pub trait HardwareProtectionInterface {
    /// Enable hardware protection for memory region
    async fn enable_protection(
        &self,
        address: *mut u8,
        size: usize,
        level: ProtectionLevel,
    ) -> Result<HardwareProtectionHandle>;
    /// Disable hardware protection
    async fn disable_protection(&self, handle: HardwareProtectionHandle) -> Result<()>;
    /// Check if hardware protection is available
    fn is_available(&self) -> bool;
    /// Get supported protection levels
    fn supported_levels(&self) -> Vec<ProtectionLevel>;
}
#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn test_memory_encryption_manager_creation() {
        let config = MemoryEncryptionConfig::default();
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
        let hardware_protection = Arc::new(DefaultHardwareProtection);
        let memory_encryption_manager = MemoryEncryptionManager::new(
            config,
            dek_manager,
            hardware_protection,
        );
        assert!(memory_encryption_manager.config.enabled);
    }
    #[tokio::test]
    async fn test_secure_buffer_allocation() {
        let dek_manager = Arc::new(
            crate::encryption::key_management::DataEncryptionKeyManager::new(
                crate::encryption::types::DEKConfig::default(),
                Arc::new(
                    crate::encryption::key_management::MasterKeyManager::new(
                        crate::encryption::types::MasterKeyConfig::default(),
                        Arc::new(crate::encryption::key_management::InMemoryKMS::new()),
                        None,
                    ),
                ),
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
        let secure_memory_manager = SecureMemoryManager::new(dek_manager);
        secure_memory_manager.start().await.expect("async operation should succeed in test");
        let buffer = secure_memory_manager
            .allocate_secure(1024, ProtectionLevel::Basic)
            .await;
        assert!(buffer.is_ok());
    }
    #[tokio::test]
    async fn test_wiping_patterns() {
        let patterns = WipingPatterns::new();
        let random_pattern = patterns.get_random_pattern(1024);
        assert_eq!(random_pattern.len(), 1024);
    }
}
