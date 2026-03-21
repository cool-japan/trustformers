use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::cache_key::CacheKey;

/// Trait for cache eviction policies
pub trait EvictionPolicy: Send + Sync {
    /// Called when an entry is accessed
    fn on_access(&mut self, key: &CacheKey);

    /// Called when a new entry is added
    fn on_insert(&mut self, key: &CacheKey, size: usize);

    /// Called when an entry is removed
    fn on_remove(&mut self, key: &CacheKey);

    /// Get the next key to evict, if any
    fn next_eviction(&mut self) -> Option<CacheKey>;

    /// Check if eviction is needed based on current state
    fn should_evict(&self) -> bool;

    /// Get current memory usage (for size-based policies)
    fn current_size(&self) -> usize {
        0
    }

    /// Get maximum allowed size (for size-based policies)
    fn max_size(&self) -> usize {
        usize::MAX
    }
}

/// Least Recently Used (LRU) eviction policy
pub struct LRUEviction {
    capacity: usize,
    access_order: VecDeque<CacheKey>,
    key_positions: HashMap<CacheKey, usize>,
}

impl LRUEviction {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            access_order: VecDeque::with_capacity(capacity),
            key_positions: HashMap::new(),
        }
    }

    fn move_to_back(&mut self, key: &CacheKey) {
        // Remove from current position if exists
        if self.key_positions.contains_key(key) {
            // Find and remove the key from access_order
            self.access_order.retain(|k| k != key);
            // Rebuild positions map
            self.key_positions.clear();
            for (idx, k) in self.access_order.iter().enumerate() {
                self.key_positions.insert(k.clone(), idx);
            }
        }

        // Add to back
        self.access_order.push_back(key.clone());
        self.key_positions.insert(key.clone(), self.access_order.len() - 1);
    }
}

impl EvictionPolicy for LRUEviction {
    fn on_access(&mut self, key: &CacheKey) {
        self.move_to_back(key);
    }

    fn on_insert(&mut self, key: &CacheKey, _size: usize) {
        self.access_order.push_back(key.clone());
        self.key_positions.insert(key.clone(), self.access_order.len() - 1);
    }

    fn on_remove(&mut self, key: &CacheKey) {
        if self.key_positions.contains_key(key) {
            // Remove the key from access_order
            self.access_order.retain(|k| k != key);
            // Rebuild positions map
            self.key_positions.clear();
            for (idx, k) in self.access_order.iter().enumerate() {
                self.key_positions.insert(k.clone(), idx);
            }
        }
    }

    fn next_eviction(&mut self) -> Option<CacheKey> {
        self.access_order.pop_front().map(|key| {
            // Rebuild positions map
            self.key_positions.clear();
            for (idx, k) in self.access_order.iter().enumerate() {
                self.key_positions.insert(k.clone(), idx);
            }
            key
        })
    }

    fn should_evict(&self) -> bool {
        self.access_order.len() > self.capacity
    }
}

/// Time-To-Live (TTL) based eviction policy
pub struct TTLEviction {
    ttl: Duration,
    expiry_times: HashMap<CacheKey, Instant>,
    expiry_queue: VecDeque<(Instant, CacheKey)>,
}

impl TTLEviction {
    pub fn new(ttl: Duration) -> Self {
        Self {
            ttl,
            expiry_times: HashMap::new(),
            expiry_queue: VecDeque::new(),
        }
    }

    fn cleanup_expired(&mut self) {
        let now = Instant::now();
        while let Some(&(expiry, _)) = self.expiry_queue.front() {
            if expiry <= now {
                if let Some((_, key)) = self.expiry_queue.pop_front() {
                    self.expiry_times.remove(&key);
                }
            } else {
                break;
            }
        }
    }
}

impl EvictionPolicy for TTLEviction {
    fn on_access(&mut self, key: &CacheKey) {
        // Update expiry time on access (optional, for sliding window TTL)
        let new_expiry = Instant::now() + self.ttl;
        self.expiry_times.insert(key.clone(), new_expiry);

        // Note: In production, we'd want to update the queue position too
        // but for simplicity, we'll just rely on cleanup
    }

    fn on_insert(&mut self, key: &CacheKey, _size: usize) {
        let expiry = Instant::now() + self.ttl;
        self.expiry_times.insert(key.clone(), expiry);
        self.expiry_queue.push_back((expiry, key.clone()));
    }

    fn on_remove(&mut self, key: &CacheKey) {
        self.expiry_times.remove(key);
        // Note: We don't remove from queue for efficiency
        // Cleanup will handle it when we reach that entry
    }

    fn next_eviction(&mut self) -> Option<CacheKey> {
        self.cleanup_expired();

        // Check if the oldest entry has expired
        let now = Instant::now();
        if let Some(&(expiry, _)) = self.expiry_queue.front() {
            if expiry <= now {
                if let Some((_, key)) = self.expiry_queue.pop_front() {
                    self.expiry_times.remove(&key);
                    return Some(key);
                }
            }
        }

        None
    }

    fn should_evict(&self) -> bool {
        if let Some(&(expiry, _)) = self.expiry_queue.front() {
            expiry <= Instant::now()
        } else {
            false
        }
    }
}

/// Size-based eviction policy
pub struct SizeBasedEviction {
    max_memory: usize,
    current_memory: usize,
    entry_sizes: HashMap<CacheKey, usize>,
    access_order: VecDeque<CacheKey>, // For tie-breaking (LRU within size)
}

impl SizeBasedEviction {
    pub fn new(max_memory_bytes: usize) -> Self {
        Self {
            max_memory: max_memory_bytes,
            current_memory: 0,
            entry_sizes: HashMap::new(),
            access_order: VecDeque::new(),
        }
    }
}

impl EvictionPolicy for SizeBasedEviction {
    fn on_access(&mut self, key: &CacheKey) {
        // Move to back of access order for LRU tie-breaking
        if let Some(pos) = self.access_order.iter().position(|k| k == key) {
            self.access_order.remove(pos);
            self.access_order.push_back(key.clone());
        }
    }

    fn on_insert(&mut self, key: &CacheKey, size: usize) {
        self.entry_sizes.insert(key.clone(), size);
        self.current_memory += size;
        self.access_order.push_back(key.clone());
    }

    fn on_remove(&mut self, key: &CacheKey) {
        if let Some(size) = self.entry_sizes.remove(key) {
            self.current_memory = self.current_memory.saturating_sub(size);
        }
        self.access_order.retain(|k| k != key);
    }

    fn next_eviction(&mut self) -> Option<CacheKey> {
        // Evict least recently used entry
        self.access_order.pop_front().map(|key| {
            if let Some(size) = self.entry_sizes.remove(&key) {
                self.current_memory = self.current_memory.saturating_sub(size);
            }
            key
        })
    }

    fn should_evict(&self) -> bool {
        self.current_memory > self.max_memory
    }

    fn current_size(&self) -> usize {
        self.current_memory
    }

    fn max_size(&self) -> usize {
        self.max_memory
    }
}

/// Composite eviction policy that combines multiple policies
pub struct CompositeEviction {
    policies: Vec<Arc<parking_lot::Mutex<Box<dyn EvictionPolicy>>>>,
}

impl CompositeEviction {
    pub fn new(policies: Vec<Box<dyn EvictionPolicy>>) -> Self {
        Self {
            policies: policies.into_iter().map(|p| Arc::new(parking_lot::Mutex::new(p))).collect(),
        }
    }

    /// Helper method to notify all policies about a key removal
    fn notify_removal(&self, key: &CacheKey) {
        for policy in &self.policies {
            policy.lock().on_remove(key);
        }
    }
}

impl EvictionPolicy for CompositeEviction {
    fn on_access(&mut self, key: &CacheKey) {
        for policy in &self.policies {
            policy.lock().on_access(key);
        }
    }

    fn on_insert(&mut self, key: &CacheKey, size: usize) {
        for policy in &self.policies {
            policy.lock().on_insert(key, size);
        }
    }

    fn on_remove(&mut self, key: &CacheKey) {
        for policy in &self.policies {
            policy.lock().on_remove(key);
        }
    }

    fn next_eviction(&mut self) -> Option<CacheKey> {
        // Return the first eviction from any policy
        for policy in &self.policies {
            let mut p = policy.lock();
            if !p.should_evict() {
                continue;
            }

            if let Some(key) = p.next_eviction() {
                // Notify other policies about the removal
                drop(p); // Release lock
                self.notify_removal(&key);
                return Some(key);
            }
        }
        None
    }

    fn should_evict(&self) -> bool {
        self.policies.iter().any(|p| p.lock().should_evict())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_eviction() {
        let mut lru = LRUEviction::new(3);

        let key1 = CacheKey::new(1, "model".to_string(), "task".to_string(), 0);
        let key2 = CacheKey::new(2, "model".to_string(), "task".to_string(), 0);
        let key3 = CacheKey::new(3, "model".to_string(), "task".to_string(), 0);
        let key4 = CacheKey::new(4, "model".to_string(), "task".to_string(), 0);

        lru.on_insert(&key1, 100);
        lru.on_insert(&key2, 200);
        lru.on_insert(&key3, 300);

        assert!(!lru.should_evict());

        lru.on_access(&key1); // Move key1 to end
        lru.on_insert(&key4, 400);

        // key2 should be evicted (least recently used)
        let evicted = lru.next_eviction();
        assert_eq!(evicted, Some(key2));
    }

    #[test]
    fn test_size_based_eviction() {
        let mut size_evict = SizeBasedEviction::new(1000);

        let key1 = CacheKey::new(1, "model".to_string(), "task".to_string(), 0);
        let key2 = CacheKey::new(2, "model".to_string(), "task".to_string(), 0);
        let key3 = CacheKey::new(3, "model".to_string(), "task".to_string(), 0);

        size_evict.on_insert(&key1, 400);
        assert!(!size_evict.should_evict());

        size_evict.on_insert(&key2, 400);
        assert!(!size_evict.should_evict());

        size_evict.on_insert(&key3, 300);
        assert!(size_evict.should_evict());
        assert_eq!(size_evict.current_size(), 1100);

        let evicted = size_evict.next_eviction();
        assert_eq!(evicted, Some(key1)); // First inserted
        assert_eq!(size_evict.current_size(), 700);
    }
}
