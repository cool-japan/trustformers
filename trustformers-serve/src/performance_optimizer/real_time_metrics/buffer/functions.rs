//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::types::StorageStats;

/// Storage backend trait for different persistence strategies
#[async_trait::async_trait]
pub trait StorageBackend<T>: Send + Sync {
    /// Store data to the backend
    async fn store(&self, key: &str, data: &[T]) -> Result<()>;
    /// Retrieve data from the backend
    async fn retrieve(&self, key: &str) -> Result<Vec<T>>;
    /// Delete data from the backend
    async fn delete(&self, key: &str) -> Result<()>;
    /// List available keys
    async fn list_keys(&self) -> Result<Vec<String>>;
    /// Get storage statistics
    fn get_stats(&self) -> StorageStats;
    /// Perform cleanup operations
    async fn cleanup(&self) -> Result<()>;
}
mod uuid {}
