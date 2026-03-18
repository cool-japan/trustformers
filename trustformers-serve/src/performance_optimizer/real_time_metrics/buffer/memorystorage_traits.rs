//! # MemoryStorage - Trait Implementations
//!
//! This module contains trait implementations for `MemoryStorage`.
//!
//! ## Implemented Traits
//!
//! - `StorageBackend`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::performance_optimizer::real_time_metrics::types::common::AtomicF32;
use anyhow::Result;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use super::functions::StorageBackend;
use super::types::{MemoryStorage, StorageStats};

#[async_trait::async_trait]
impl<T> StorageBackend<T> for MemoryStorage<T>
where
    T: Clone + Send + Sync + 'static,
{
    async fn store(&self, key: &str, data: &[T]) -> Result<()> {
        let start_time = Instant::now();
        let estimated_size = std::mem::size_of::<T>() * data.len();
        let current_memory = self.current_memory_bytes.load(Ordering::Relaxed);
        if current_memory + estimated_size > self.max_memory_bytes {
            self.stats.failed_operations.fetch_add(1, Ordering::Relaxed);
            return Err(anyhow::anyhow!("Memory limit exceeded"));
        }
        {
            let mut storage = self.data.write();
            storage.insert(key.to_string(), data.to_vec());
        }
        self.current_memory_bytes.fetch_add(estimated_size, Ordering::Relaxed);
        self.stats.store_operations.fetch_add(1, Ordering::Relaxed);
        self.stats.bytes_stored.fetch_add(estimated_size as u64, Ordering::Relaxed);
        self.update_latency(start_time.elapsed());
        self.update_utilization();
        Ok(())
    }
    async fn retrieve(&self, key: &str) -> Result<Vec<T>> {
        let start_time = Instant::now();
        let data = {
            let storage = self.data.read();
            storage.get(key).cloned()
        };
        let result = data.ok_or_else(|| anyhow::anyhow!("Key not found: {}", key))?;
        let retrieved_size = std::mem::size_of::<T>() * result.len();
        self.stats.retrieve_operations.fetch_add(1, Ordering::Relaxed);
        self.stats.bytes_retrieved.fetch_add(retrieved_size as u64, Ordering::Relaxed);
        self.update_latency(start_time.elapsed());
        Ok(result)
    }
    async fn delete(&self, key: &str) -> Result<()> {
        let start_time = Instant::now();
        let removed_size = {
            let mut storage = self.data.write();
            if let Some(data) = storage.remove(key) {
                std::mem::size_of::<T>() * data.len()
            } else {
                return Err(anyhow::anyhow!("Key not found: {}", key));
            }
        };
        self.current_memory_bytes.fetch_sub(removed_size, Ordering::Relaxed);
        self.stats.delete_operations.fetch_add(1, Ordering::Relaxed);
        self.update_latency(start_time.elapsed());
        self.update_utilization();
        Ok(())
    }
    async fn list_keys(&self) -> Result<Vec<String>> {
        let storage = self.data.read();
        Ok(storage.keys().cloned().collect())
    }
    fn get_stats(&self) -> StorageStats {
        StorageStats {
            store_operations: AtomicU64::new(self.stats.store_operations.load(Ordering::Relaxed)),
            retrieve_operations: AtomicU64::new(
                self.stats.retrieve_operations.load(Ordering::Relaxed),
            ),
            delete_operations: AtomicU64::new(self.stats.delete_operations.load(Ordering::Relaxed)),
            bytes_stored: AtomicU64::new(self.stats.bytes_stored.load(Ordering::Relaxed)),
            bytes_retrieved: AtomicU64::new(self.stats.bytes_retrieved.load(Ordering::Relaxed)),
            avg_latency_ns: AtomicU64::new(self.stats.avg_latency_ns.load(Ordering::Relaxed)),
            failed_operations: AtomicU64::new(self.stats.failed_operations.load(Ordering::Relaxed)),
            utilization_percent: AtomicF32::new(
                self.stats.utilization_percent.load(Ordering::Relaxed),
            ),
        }
    }
    async fn cleanup(&self) -> Result<()> {
        let mut storage = self.data.write();
        storage.clear();
        self.current_memory_bytes.store(0, Ordering::Relaxed);
        self.update_utilization();
        Ok(())
    }
}
