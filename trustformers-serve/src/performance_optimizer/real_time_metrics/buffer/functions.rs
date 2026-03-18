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
mod uuid {
    use std::fmt;
    pub struct Uuid(u128);
    impl Uuid {
        pub fn new_v4() -> Self {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            use std::time::{SystemTime, UNIX_EPOCH};
            let mut hasher = DefaultHasher::new();
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_else(|_| std::time::Duration::from_secs(0))
                .as_nanos()
                .hash(&mut hasher);
            Uuid(hasher.finish() as u128)
        }
    }
    impl fmt::Display for Uuid {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{:032x}", self.0)
        }
    }
}
