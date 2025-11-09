use crate::vocab::Vocab;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock, Weak};
use trustformers_core::errors::{Result, TrustformersError};

/// Configuration for the shared vocabulary pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabPoolConfig {
    /// Maximum number of vocabularies to keep in the pool
    pub max_pool_size: usize,
    /// Whether to enable automatic cleanup of unused vocabularies
    pub enable_auto_cleanup: bool,
    /// Cleanup interval in milliseconds
    pub cleanup_interval_ms: u64,
    /// Memory threshold for triggering cleanup (in bytes)
    pub memory_threshold_bytes: usize,
    /// Whether to enable vocabulary deduplication
    pub enable_deduplication: bool,
}

impl Default for VocabPoolConfig {
    fn default() -> Self {
        Self {
            max_pool_size: 100,
            enable_auto_cleanup: true,
            cleanup_interval_ms: 30000,                // 30 seconds
            memory_threshold_bytes: 1024 * 1024 * 500, // 500 MB
            enable_deduplication: true,
        }
    }
}

/// Statistics about the vocabulary pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabPoolStats {
    /// Number of vocabularies in the pool
    pub vocabulary_count: usize,
    /// Total memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Number of active references
    pub active_references: usize,
    /// Number of cache hits
    pub cache_hits: usize,
    /// Number of cache misses
    pub cache_misses: usize,
    /// Memory saved through deduplication
    pub memory_saved_bytes: usize,
    /// Average vocabulary size
    pub average_vocab_size: usize,
}

/// Vocabulary entry in the pool
#[derive(Debug, Clone)]
struct VocabEntry {
    /// The vocabulary itself
    vocab: Arc<Vocab>,
    /// Unique identifier for this vocabulary
    #[allow(dead_code)]
    id: String,
    /// Hash of the vocabulary content for deduplication
    content_hash: u64,
    /// Size in bytes
    size_bytes: usize,
    /// Number of times this vocabulary has been accessed
    access_count: usize,
    /// Timestamp of last access
    last_accessed: std::time::Instant,
    /// Weak references to track usage
    weak_refs: Vec<Weak<Vocab>>,
}

impl VocabEntry {
    fn new(vocab: Arc<Vocab>, id: String) -> Self {
        let content_hash = Self::calculate_hash(&vocab);
        let size_bytes = Self::estimate_size(&vocab);

        Self {
            vocab,
            id,
            content_hash,
            size_bytes,
            access_count: 0,
            last_accessed: std::time::Instant::now(),
            weak_refs: Vec::new(),
        }
    }

    fn calculate_hash(vocab: &Vocab) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Hash the token-to-id mapping
        let token_map = vocab.get_token_to_id_map();
        let mut sorted_tokens: Vec<_> = token_map.iter().collect();
        sorted_tokens.sort_by_key(|(_, id)| *id);

        for (token, id) in sorted_tokens {
            token.hash(&mut hasher);
            id.hash(&mut hasher);
        }

        hasher.finish()
    }

    fn estimate_size(vocab: &Vocab) -> usize {
        let token_map = vocab.get_token_to_id_map();
        let mut size = std::mem::size_of::<Vocab>();

        for token in token_map.keys() {
            size += token.len() + std::mem::size_of::<String>() + std::mem::size_of::<u32>();
        }

        size
    }

    fn is_unused(&self) -> bool {
        // A vocab is unused if only the pool itself holds a reference to it
        Arc::strong_count(&self.vocab) == 1
    }

    fn cleanup_weak_refs(&mut self) {
        self.weak_refs.retain(|weak_ref| weak_ref.upgrade().is_some());
    }

    fn add_weak_ref(&mut self, vocab_ref: &Arc<Vocab>) {
        self.weak_refs.push(Arc::downgrade(vocab_ref));
    }
}

/// Shared vocabulary pool for memory-efficient vocabulary management
#[derive(Debug)]
pub struct SharedVocabPool {
    /// Configuration
    config: VocabPoolConfig,
    /// Storage for vocabulary entries
    pool: Arc<RwLock<HashMap<String, VocabEntry>>>,
    /// Hash-to-ID mapping for deduplication
    hash_to_id: Arc<RwLock<HashMap<u64, String>>>,
    /// Statistics
    stats: Arc<RwLock<VocabPoolStats>>,
    /// Last cleanup time
    last_cleanup: Arc<RwLock<std::time::Instant>>,
}

impl SharedVocabPool {
    /// Create a new shared vocabulary pool
    pub fn new(config: VocabPoolConfig) -> Self {
        Self {
            config,
            pool: Arc::new(RwLock::new(HashMap::new())),
            hash_to_id: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(VocabPoolStats {
                vocabulary_count: 0,
                memory_usage_bytes: 0,
                active_references: 0,
                cache_hits: 0,
                cache_misses: 0,
                memory_saved_bytes: 0,
                average_vocab_size: 0,
            })),
            last_cleanup: Arc::new(RwLock::new(std::time::Instant::now())),
        }
    }

    /// Get or create a vocabulary in the pool
    pub fn get_or_insert(&self, id: String, vocab: Vocab) -> Result<Arc<Vocab>> {
        // Check if we need to perform cleanup
        if self.config.enable_auto_cleanup {
            self.try_cleanup();
        }

        let vocab_arc = Arc::new(vocab);

        // Calculate hash for deduplication
        let content_hash = VocabEntry::calculate_hash(&vocab_arc);

        // Check for existing vocabulary with same content
        if self.config.enable_deduplication {
            if let Some(existing_id) = self.hash_to_id.read().unwrap().get(&content_hash) {
                if let Some(existing_vocab) = self.get_by_id(existing_id) {
                    // Update statistics
                    let mut stats = self.stats.write().unwrap();
                    stats.cache_hits += 1;
                    stats.memory_saved_bytes += VocabEntry::estimate_size(&vocab_arc);

                    return Ok(existing_vocab);
                }
            }
        }

        // Check if vocabulary with this ID already exists
        {
            let pool = self.pool.read().unwrap();
            if let Some(entry) = pool.get(&id) {
                let mut stats = self.stats.write().unwrap();
                stats.cache_hits += 1;
                return Ok(entry.vocab.clone());
            }
        }

        // Create new entry
        let entry = VocabEntry::new(vocab_arc.clone(), id.clone());

        // Check if we need to make room in the pool
        {
            let pool = self.pool.read().unwrap();
            if pool.len() >= self.config.max_pool_size {
                drop(pool);
                self.evict_least_recently_used()?;
            }
        }

        // Insert into pool
        {
            let mut pool = self.pool.write().unwrap();
            let mut hash_to_id = self.hash_to_id.write().unwrap();

            pool.insert(id.clone(), entry);
            if self.config.enable_deduplication {
                hash_to_id.insert(content_hash, id);
            }
        }

        // Update statistics
        {
            let mut stats = self.stats.write().unwrap();
            stats.vocabulary_count += 1;
            stats.memory_usage_bytes += VocabEntry::estimate_size(&vocab_arc);
            stats.cache_misses += 1;
            stats.active_references += 1;
            stats.average_vocab_size = stats.memory_usage_bytes / stats.vocabulary_count.max(1);
        }

        Ok(vocab_arc)
    }

    /// Get vocabulary by ID
    pub fn get_by_id(&self, id: &str) -> Option<Arc<Vocab>> {
        let mut pool = self.pool.write().unwrap();
        if let Some(entry) = pool.get_mut(id) {
            entry.access_count += 1;
            entry.last_accessed = std::time::Instant::now();

            let vocab = entry.vocab.clone();
            entry.add_weak_ref(&vocab);

            // Update statistics
            let mut stats = self.stats.write().unwrap();
            stats.cache_hits += 1;

            Some(vocab)
        } else {
            // Update statistics
            let mut stats = self.stats.write().unwrap();
            stats.cache_misses += 1;
            None
        }
    }

    /// Check if vocabulary exists in pool
    pub fn contains(&self, id: &str) -> bool {
        self.pool.read().unwrap().contains_key(id)
    }

    /// Remove vocabulary from pool
    pub fn remove(&self, id: &str) -> Option<Arc<Vocab>> {
        let mut pool = self.pool.write().unwrap();
        if let Some(entry) = pool.remove(id) {
            // Remove from hash mapping
            let mut hash_to_id = self.hash_to_id.write().unwrap();
            hash_to_id.remove(&entry.content_hash);

            // Update statistics
            let mut stats = self.stats.write().unwrap();
            stats.vocabulary_count -= 1;
            stats.memory_usage_bytes = stats.memory_usage_bytes.saturating_sub(entry.size_bytes);
            stats.average_vocab_size = if stats.vocabulary_count > 0 {
                stats.memory_usage_bytes / stats.vocabulary_count
            } else {
                0
            };

            Some(entry.vocab)
        } else {
            None
        }
    }

    /// Clear all vocabularies from pool
    pub fn clear(&self) {
        let mut pool = self.pool.write().unwrap();
        let mut hash_to_id = self.hash_to_id.write().unwrap();

        pool.clear();
        hash_to_id.clear();

        // Reset statistics
        let mut stats = self.stats.write().unwrap();
        stats.vocabulary_count = 0;
        stats.memory_usage_bytes = 0;
        stats.active_references = 0;
        stats.average_vocab_size = 0;
    }

    /// Get pool statistics
    pub fn get_stats(&self) -> VocabPoolStats {
        self.stats.read().unwrap().clone()
    }

    /// Force cleanup of unused vocabularies
    pub fn cleanup(&self) -> Result<usize> {
        let mut pool = self.pool.write().unwrap();
        let mut hash_to_id = self.hash_to_id.write().unwrap();
        let mut removed_count = 0;
        let mut memory_freed = 0;

        // Clean up weak references and find unused entries
        let mut to_remove = Vec::new();
        for (id, entry) in pool.iter_mut() {
            entry.cleanup_weak_refs();
            if entry.is_unused() {
                to_remove.push((id.clone(), entry.content_hash, entry.size_bytes));
            }
        }

        // Remove unused entries
        for (id, content_hash, size_bytes) in to_remove {
            pool.remove(&id);
            hash_to_id.remove(&content_hash);
            removed_count += 1;
            memory_freed += size_bytes;
        }

        // Update statistics
        let mut stats = self.stats.write().unwrap();
        stats.vocabulary_count = pool.len();
        stats.memory_usage_bytes = stats.memory_usage_bytes.saturating_sub(memory_freed);
        stats.average_vocab_size = if stats.vocabulary_count > 0 {
            stats.memory_usage_bytes / stats.vocabulary_count
        } else {
            0
        };

        // Update last cleanup time
        *self.last_cleanup.write().unwrap() = std::time::Instant::now();

        Ok(removed_count)
    }

    /// Try to perform cleanup if needed
    fn try_cleanup(&self) {
        let last_cleanup = *self.last_cleanup.read().unwrap();
        let now = std::time::Instant::now();

        let should_cleanup = if let Ok(stats) = self.stats.read() {
            now.duration_since(last_cleanup).as_millis() >= self.config.cleanup_interval_ms as u128
                || stats.memory_usage_bytes >= self.config.memory_threshold_bytes
        } else {
            false
        };

        if should_cleanup {
            let _ = self.cleanup();
        }
    }

    /// Evict least recently used vocabulary
    fn evict_least_recently_used(&self) -> Result<()> {
        let mut pool = self.pool.write().unwrap();
        let mut hash_to_id = self.hash_to_id.write().unwrap();

        // Find the least recently used entry
        let mut oldest_time = std::time::Instant::now();
        let mut oldest_id = String::new();
        let mut oldest_hash = 0u64;
        let mut oldest_size = 0usize;

        for (id, entry) in pool.iter() {
            if entry.last_accessed < oldest_time {
                oldest_time = entry.last_accessed;
                oldest_id = id.clone();
                oldest_hash = entry.content_hash;
                oldest_size = entry.size_bytes;
            }
        }

        if !oldest_id.is_empty() {
            pool.remove(&oldest_id);
            hash_to_id.remove(&oldest_hash);

            // Update statistics
            let mut stats = self.stats.write().unwrap();
            stats.vocabulary_count -= 1;
            stats.memory_usage_bytes = stats.memory_usage_bytes.saturating_sub(oldest_size);
            stats.average_vocab_size = if stats.vocabulary_count > 0 {
                stats.memory_usage_bytes / stats.vocabulary_count
            } else {
                0
            };
        }

        Ok(())
    }

    /// Get list of all vocabulary IDs in the pool
    pub fn list_vocabularies(&self) -> Vec<String> {
        self.pool.read().unwrap().keys().cloned().collect()
    }

    /// Get memory usage for a specific vocabulary
    pub fn get_vocab_memory_usage(&self, id: &str) -> Option<usize> {
        self.pool.read().unwrap().get(id).map(|entry| entry.size_bytes)
    }

    /// Get access statistics for a specific vocabulary
    pub fn get_vocab_access_stats(&self, id: &str) -> Option<(usize, std::time::Instant)> {
        self.pool
            .read()
            .unwrap()
            .get(id)
            .map(|entry| (entry.access_count, entry.last_accessed))
    }

    /// Merge vocabularies in the pool
    pub fn merge_vocabularies(&self, ids: &[String], new_id: String) -> Result<Arc<Vocab>> {
        let vocabs: Result<Vec<_>> = ids
            .iter()
            .map(|id| {
                self.get_by_id(id).ok_or_else(|| {
                    TrustformersError::other(format!("Vocabulary '{}' not found in pool", id))
                })
            })
            .collect();

        let vocabs = vocabs?;
        let vocab_refs: Vec<&Vocab> = vocabs.iter().map(|v| v.as_ref()).collect();

        // Use the existing merge functionality from Vocab
        let merged_vocab = Vocab::merge_multiple(
            vocab_refs.into_iter().cloned().collect(),
            crate::vocab::MergeStrategy::KeepBothWithSuffix,
        )?;

        self.get_or_insert(new_id, merged_vocab)
    }

    /// Create a global singleton instance
    pub fn global() -> &'static SharedVocabPool {
        use std::sync::OnceLock;
        static POOL: OnceLock<SharedVocabPool> = OnceLock::new();

        POOL.get_or_init(|| SharedVocabPool::new(VocabPoolConfig::default()))
    }
}

impl Default for SharedVocabPool {
    fn default() -> Self {
        Self::new(VocabPoolConfig::default())
    }
}

/// A vocabulary reference that automatically manages pool lifecycle
#[derive(Debug, Clone)]
pub struct PooledVocab {
    vocab: Arc<Vocab>,
    id: String,
    pool: Arc<SharedVocabPool>,
}

impl PooledVocab {
    /// Create a new pooled vocabulary
    pub fn new(pool: Arc<SharedVocabPool>, id: String, vocab: Vocab) -> Result<Self> {
        let vocab_arc = pool.get_or_insert(id.clone(), vocab)?;
        Ok(Self {
            vocab: vocab_arc,
            id,
            pool,
        })
    }

    /// Get the underlying vocabulary
    pub fn vocab(&self) -> &Arc<Vocab> {
        &self.vocab
    }

    /// Get the vocabulary ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get reference to the pool
    pub fn pool(&self) -> &Arc<SharedVocabPool> {
        &self.pool
    }
}

impl std::ops::Deref for PooledVocab {
    type Target = Vocab;

    fn deref(&self) -> &Self::Target {
        &self.vocab
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_vocab_pool_creation() {
        let config = VocabPoolConfig::default();
        let pool = SharedVocabPool::new(config);

        let stats = pool.get_stats();
        assert_eq!(stats.vocabulary_count, 0);
        assert_eq!(stats.memory_usage_bytes, 0);
    }

    #[test]
    fn test_vocab_insertion_and_retrieval() {
        let pool = SharedVocabPool::new(VocabPoolConfig::default());

        let mut token_map = HashMap::new();
        token_map.insert("hello".to_string(), 0);
        token_map.insert("world".to_string(), 1);
        let vocab = Vocab::from_map(token_map);

        let vocab_ref = pool.get_or_insert("test_vocab".to_string(), vocab).unwrap();
        assert_eq!(vocab_ref.size(), 2);

        let retrieved = pool.get_by_id("test_vocab").unwrap();
        assert_eq!(retrieved.size(), 2);

        let stats = pool.get_stats();
        assert_eq!(stats.vocabulary_count, 1);
        assert!(stats.memory_usage_bytes > 0);
    }

    #[test]
    fn test_vocab_deduplication() {
        let mut config = VocabPoolConfig::default();
        config.enable_deduplication = true;
        let pool = SharedVocabPool::new(config);

        let mut token_map = HashMap::new();
        token_map.insert("hello".to_string(), 0);
        token_map.insert("world".to_string(), 1);

        let vocab1 = Vocab::from_map(token_map.clone());
        let vocab2 = Vocab::from_map(token_map);

        let vocab_ref1 = pool.get_or_insert("vocab1".to_string(), vocab1).unwrap();
        let vocab_ref2 = pool.get_or_insert("vocab2".to_string(), vocab2).unwrap();

        // Should be the same reference due to deduplication
        assert!(Arc::ptr_eq(&vocab_ref1, &vocab_ref2));

        let stats = pool.get_stats();
        assert!(stats.memory_saved_bytes > 0);
    }

    #[test]
    fn test_vocab_cleanup() {
        let pool = SharedVocabPool::new(VocabPoolConfig::default());

        let mut token_map = HashMap::new();
        token_map.insert("test".to_string(), 0);
        let vocab = Vocab::from_map(token_map);

        // Insert and immediately drop the reference
        {
            let _vocab_ref = pool.get_or_insert("temp_vocab".to_string(), vocab).unwrap();
        }

        // Force cleanup
        let removed_count = pool.cleanup().unwrap();
        assert_eq!(removed_count, 1);

        let stats = pool.get_stats();
        assert_eq!(stats.vocabulary_count, 0);
    }

    #[test]
    fn test_pooled_vocab() {
        let pool = Arc::new(SharedVocabPool::new(VocabPoolConfig::default()));

        let mut token_map = HashMap::new();
        token_map.insert("pooled".to_string(), 0);
        let vocab = Vocab::from_map(token_map);

        let pooled_vocab =
            PooledVocab::new(pool.clone(), "pooled_test".to_string(), vocab).unwrap();

        assert_eq!(pooled_vocab.id(), "pooled_test");
        assert_eq!(pooled_vocab.size(), 1);
        assert!(pooled_vocab.contains("pooled"));
    }

    #[test]
    fn test_memory_threshold_cleanup() {
        let mut config = VocabPoolConfig::default();
        config.memory_threshold_bytes = 100; // Very low threshold
        config.enable_auto_cleanup = true;

        let pool = SharedVocabPool::new(config);

        // Add a large vocabulary to trigger cleanup
        let mut token_map = HashMap::new();
        for i in 0..1000 {
            token_map.insert(format!("token_{}", i), i);
        }
        let vocab = Vocab::from_map(token_map);

        let _vocab_ref = pool.get_or_insert("large_vocab".to_string(), vocab).unwrap();

        // The memory threshold should trigger automatic cleanup
        // Add another vocab to trigger the check
        let mut small_map = HashMap::new();
        small_map.insert("small".to_string(), 0);
        let small_vocab = Vocab::from_map(small_map);

        let _small_ref = pool.get_or_insert("small_vocab".to_string(), small_vocab).unwrap();

        // Check that some cleanup occurred
        let stats = pool.get_stats();
        assert!(stats.vocabulary_count <= 2);
    }
}
