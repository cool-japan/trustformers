//! Inference result caching.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use trustformers_core::Tensor;

/// Inference cache for mobile deployment
#[derive(Debug)]
pub(super) struct InferenceCache {
    pub(super) cache: HashMap<Vec<u8>, Tensor>,
    pub(super) max_size_mb: usize,
    pub(super) current_size_bytes: usize,
}

impl InferenceCache {
    pub(super) fn new(max_size_mb: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size_mb,
            current_size_bytes: 0,
        }
    }

    pub(super) fn get(&self, input: &Tensor) -> Option<Tensor> {
        let key = self.tensor_to_key(input);
        self.cache.get(&key).cloned()
    }

    pub(super) fn put(&mut self, input: Tensor, output: Tensor) {
        let key = self.tensor_to_key(&input);
        let entry_size = input.memory_usage() + output.memory_usage();

        // Check if we have space
        if self.current_size_bytes + entry_size > self.max_size_mb * 1024 * 1024 {
            self.evict_lru();
        }

        self.cache.insert(key, output);
        self.current_size_bytes += entry_size;
    }

    pub(super) fn clear(&mut self) {
        self.cache.clear();
        self.current_size_bytes = 0;
    }

    pub(super) fn memory_usage_mb(&self) -> usize {
        self.current_size_bytes / (1024 * 1024)
    }

    pub(super) fn tensor_to_key(&self, tensor: &Tensor) -> Vec<u8> {
        // Create a simple key from tensor shape and first few values
        // This is a simplified implementation
        let shape = tensor.shape();
        let mut key = Vec::new();

        for &dim in &shape {
            key.extend_from_slice(&dim.to_le_bytes());
        }

        key
    }

    pub(super) fn evict_lru(&mut self) {
        // Simple eviction strategy - remove oldest entries
        // In practice, would use a proper LRU implementation
        if let Some(first_key) = self.cache.keys().next().cloned() {
            self.cache.remove(&first_key);
            self.current_size_bytes = self.current_size_bytes.saturating_sub(1024 * 1024);
            // Approximate
        }
    }
}
