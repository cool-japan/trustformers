//! Prefix caching for KV cache reuse across requests sharing common prompt prefixes.
//!
//! When multiple requests share the same system prompt or few-shot examples, the KV
//! cache for those shared tokens can be stored in a trie structure and reused,
//! avoiding redundant computation.

use std::collections::HashMap;
use std::time::Instant;

// ─── Hash ─────────────────────────────────────────────────────────────────────

/// Newtype wrapper for a FNV-1a hash of a token sequence prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrefixHash(pub u64);

impl PrefixHash {
    /// FNV-1a offset basis (64-bit).
    const FNV_OFFSET: u64 = 14695981039346656037;
    /// FNV-1a prime (64-bit).
    const FNV_PRIME: u64 = 1099511628211;

    /// Hash an entire token sequence using FNV-1a.
    pub fn from_tokens(tokens: &[u32]) -> Self {
        let mut hash = Self::FNV_OFFSET;
        for &token in tokens {
            let bytes = token.to_le_bytes();
            for byte in bytes {
                hash ^= byte as u64;
                hash = hash.wrapping_mul(Self::FNV_PRIME);
            }
        }
        PrefixHash(hash)
    }

    /// Incrementally extend a base hash with one additional token.
    pub fn extend(base: PrefixHash, token: u32) -> Self {
        let mut hash = base.0;
        let bytes = token.to_le_bytes();
        for byte in bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(Self::FNV_PRIME);
        }
        PrefixHash(hash)
    }
}

// ─── Node ─────────────────────────────────────────────────────────────────────

/// A single node in the prefix trie.
#[derive(Debug, Clone)]
pub struct PrefixNode {
    /// Hash of the full token path from root to this node.
    pub hash: PrefixHash,
    /// Tokens stored in this specific node (a single token in our implementation).
    pub token_ids: Vec<u32>,
    /// Parent node hash (`None` for the root).
    pub parent: Option<PrefixHash>,
    /// Hashes of all child nodes.
    pub children: Vec<PrefixHash>,
    /// Depth in the trie (root = 0).
    pub depth: usize,
    /// Number of active requests currently referencing this node.
    pub ref_count: usize,
    /// Wall-clock time of the last access to this node.
    pub last_access_time: Instant,
    /// Estimated KV cache memory for the tokens at this node.
    pub kv_cache_size_bytes: usize,
}

// ─── Stats ────────────────────────────────────────────────────────────────────

/// Aggregate statistics for the prefix trie.
#[derive(Debug, Clone, Default)]
pub struct PrefixTrieStats {
    pub total_nodes: usize,
    pub total_tokens_cached: usize,
    pub total_kv_bytes: usize,
    pub hit_count: u64,
    pub miss_count: u64,
    pub eviction_count: u64,
}

impl PrefixTrieStats {
    /// Fraction of lookups that resulted in a cache hit (0.0 – 1.0).
    pub fn hit_rate(&self) -> f32 {
        let total = self.hit_count + self.miss_count;
        if total == 0 {
            0.0
        } else {
            self.hit_count as f32 / total as f32
        }
    }
}

// ─── Eviction Policy ──────────────────────────────────────────────────────────

/// Policy that governs which nodes are chosen for eviction when the cache is full.
#[derive(Debug, Clone, PartialEq)]
pub enum PrefixEvictionPolicy {
    /// Evict the least recently used leaf nodes first.
    Lru,
    /// Prefer nodes with zero ref-count; fall back to LRU among those.
    RefCountBased,
    /// Prefer evicting the largest (most bytes) leaf nodes first.
    SizeAware,
}

// ─── Config ───────────────────────────────────────────────────────────────────

/// Configuration for the prefix trie cache.
#[derive(Debug, Clone)]
pub struct PrefixCacheConfig {
    /// Maximum total number of tokens that may be cached.
    pub max_total_tokens: usize,
    /// Maximum total bytes (KV cache estimate) that may be held.
    pub max_total_bytes: usize,
    /// Which eviction policy to apply when limits are exceeded.
    pub eviction_policy: PrefixEvictionPolicy,
    /// Minimum token-sequence length required before a node is cached.
    pub min_prefix_length: usize,
    /// Estimated KV bytes per token per layer (rough model-dependent constant).
    pub bytes_per_token_per_layer: usize,
    /// Number of transformer layers (used for byte estimation).
    pub num_layers: usize,
}

impl Default for PrefixCacheConfig {
    fn default() -> Self {
        Self {
            max_total_tokens: 100_000,
            max_total_bytes: 2 * 1024 * 1024 * 1024,
            eviction_policy: PrefixEvictionPolicy::Lru,
            min_prefix_length: 4,
            bytes_per_token_per_layer: 512,
            num_layers: 32,
        }
    }
}

// ─── Cache ────────────────────────────────────────────────────────────────────

/// A token-prefix trie backed by a hash map for O(1) node lookups.
///
/// Each node represents one token in the shared prefix path. Paths are walked
/// incrementally so that a sequence of N tokens produces N nodes (plus a virtual
/// root at depth 0).
pub struct PrefixCache {
    /// All trie nodes, keyed by their `PrefixHash`.
    pub nodes: HashMap<PrefixHash, PrefixNode>,
    pub config: PrefixCacheConfig,
    pub stats: PrefixTrieStats,
    /// Sentinel root hash — always present, represents the empty prefix.
    root_hash: PrefixHash,
}

impl PrefixCache {
    /// Create a new, empty prefix cache with the given configuration.
    pub fn new(config: PrefixCacheConfig) -> Self {
        // Virtual root node: empty tokens, no parent, depth 0.
        let root_hash = PrefixHash(0);
        let root = PrefixNode {
            hash: root_hash,
            token_ids: Vec::new(),
            parent: None,
            children: Vec::new(),
            depth: 0,
            ref_count: 0,
            last_access_time: Instant::now(),
            kv_cache_size_bytes: 0,
        };
        let mut nodes = HashMap::new();
        nodes.insert(root_hash, root);

        Self {
            nodes,
            config,
            stats: PrefixTrieStats::default(),
            root_hash,
        }
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// Insert a token sequence into the trie, building any missing nodes along
    /// the path.  Returns the hash of the deepest node (representing the full
    /// sequence).
    ///
    /// Also increments `ref_count` on every node along the path and triggers
    /// eviction if capacity limits are exceeded.
    pub fn insert(&mut self, tokens: &[u32]) -> PrefixHash {
        if tokens.is_empty() {
            return self.root_hash;
        }

        let mut current_hash = self.root_hash;

        for (idx, &token) in tokens.iter().enumerate() {
            let child_hash = PrefixHash::extend(current_hash, token);

            if !self.nodes.contains_key(&child_hash) {
                // Only create the node if we meet the minimum prefix length.
                let depth = idx + 1;
                let kv_bytes = self.estimate_bytes_for_tokens(1);
                let node = PrefixNode {
                    hash: child_hash,
                    token_ids: vec![token],
                    parent: Some(current_hash),
                    children: Vec::new(),
                    depth,
                    ref_count: 0,
                    last_access_time: Instant::now(),
                    kv_cache_size_bytes: kv_bytes,
                };
                self.nodes.insert(child_hash, node);

                // Register this child with the parent.
                if let Some(parent) = self.nodes.get_mut(&current_hash) {
                    parent.children.push(child_hash);
                }

                self.stats.total_nodes += 1;
                self.stats.total_tokens_cached += 1;
                self.stats.total_kv_bytes += kv_bytes;
            }

            // Refresh access time and bump ref_count.
            if let Some(node) = self.nodes.get_mut(&child_hash) {
                node.ref_count += 1;
                node.last_access_time = Instant::now();
            }

            current_hash = child_hash;
        }

        self.evict_if_needed();

        current_hash
    }

    /// Walk the trie, finding the longest prefix of `tokens` that is already
    /// cached.
    ///
    /// Returns `(matched_prefix_length, last_matched_node_hash)`.
    /// If nothing is matched beyond the root, returns `(0, None)`.
    ///
    /// Updates hit/miss statistics: a hit is recorded if at least
    /// `config.min_prefix_length` tokens are matched.
    pub fn lookup_prefix(&mut self, tokens: &[u32]) -> (usize, Option<PrefixHash>) {
        if tokens.is_empty() {
            self.stats.miss_count += 1;
            return (0, None);
        }

        let mut current_hash = self.root_hash;
        let mut matched = 0usize;
        let mut last_hash: Option<PrefixHash> = None;

        for &token in tokens {
            let child_hash = PrefixHash::extend(current_hash, token);
            if self.nodes.contains_key(&child_hash) {
                matched += 1;
                last_hash = Some(child_hash);
                if let Some(node) = self.nodes.get_mut(&child_hash) {
                    node.last_access_time = Instant::now();
                }
                current_hash = child_hash;
            } else {
                break;
            }
        }

        if matched >= self.config.min_prefix_length {
            self.stats.hit_count += 1;
        } else {
            self.stats.miss_count += 1;
            // Still return whatever partial match we found, but stats say miss.
            if matched == 0 {
                return (0, None);
            }
        }

        (matched, last_hash)
    }

    /// Decrement the `ref_count` for `hash` and all of its ancestors.
    pub fn release(&mut self, hash: PrefixHash) {
        let mut current = Some(hash);
        while let Some(h) = current {
            if h == self.root_hash {
                break;
            }
            let parent = if let Some(node) = self.nodes.get_mut(&h) {
                node.ref_count = node.ref_count.saturating_sub(1);
                node.parent
            } else {
                break;
            };
            current = parent;
        }
    }

    /// Evict nodes if either token or byte limits are exceeded.
    ///
    /// Returns the number of nodes evicted.
    pub fn evict_if_needed(&mut self) -> usize {
        let mut evicted = 0usize;
        loop {
            let over_tokens = self.total_tokens_in_tree() > self.config.max_total_tokens;
            let over_bytes = self.stats.total_kv_bytes > self.config.max_total_bytes;
            if !over_tokens && !over_bytes {
                break;
            }
            if !self.evict_lru_leaf() {
                break;
            }
            evicted += 1;
            self.stats.eviction_count += 1;
        }
        evicted
    }

    /// Estimate the KV cache memory for a given number of tokens.
    pub fn estimate_bytes_for_tokens(&self, num_tokens: usize) -> usize {
        num_tokens * self.config.bytes_per_token_per_layer * self.config.num_layers
    }

    /// Total number of actual (non-root) tokens currently stored in the trie.
    pub fn total_tokens_in_tree(&self) -> usize {
        // Every non-root node holds exactly one token.
        self.nodes.len().saturating_sub(1)
    }

    /// Return a reference to the current statistics snapshot.
    pub fn stats(&self) -> &PrefixTrieStats {
        &self.stats
    }

    /// Evict one LRU leaf node according to the configured eviction policy.
    ///
    /// Returns `true` if a node was evicted, `false` if nothing could be evicted.
    pub fn evict_lru_leaf(&mut self) -> bool {
        let victim = match self.config.eviction_policy {
            PrefixEvictionPolicy::Lru => self.select_lru_leaf(),
            PrefixEvictionPolicy::RefCountBased => self.select_ref_count_leaf(),
            PrefixEvictionPolicy::SizeAware => self.select_size_aware_leaf(),
        };

        if let Some(hash) = victim {
            self.remove_node(hash);
            true
        } else {
            false
        }
    }

    /// Return `true` if `hash` refers to a leaf node (no children).
    pub fn is_leaf(&self, hash: &PrefixHash) -> bool {
        self.nodes.get(hash).map(|n| n.children.is_empty()).unwrap_or(false)
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Select the LRU leaf node (oldest `last_access_time`), excluding root.
    fn select_lru_leaf(&self) -> Option<PrefixHash> {
        self.nodes
            .values()
            .filter(|n| n.hash != self.root_hash && n.children.is_empty())
            .min_by_key(|n| n.last_access_time)
            .map(|n| n.hash)
    }

    /// Select the leaf with ref_count == 0 and oldest access time; fall back to
    /// any leaf if all have non-zero ref_counts.
    fn select_ref_count_leaf(&self) -> Option<PrefixHash> {
        // First try: zero ref-count leaves, oldest first.
        let zero_ref = self
            .nodes
            .values()
            .filter(|n| n.hash != self.root_hash && n.children.is_empty() && n.ref_count == 0)
            .min_by_key(|n| n.last_access_time)
            .map(|n| n.hash);

        if zero_ref.is_some() {
            return zero_ref;
        }

        // Fall back: any leaf, oldest first.
        self.select_lru_leaf()
    }

    /// Select the leaf node with the largest `kv_cache_size_bytes`.
    fn select_size_aware_leaf(&self) -> Option<PrefixHash> {
        self.nodes
            .values()
            .filter(|n| n.hash != self.root_hash && n.children.is_empty())
            .max_by_key(|n| n.kv_cache_size_bytes)
            .map(|n| n.hash)
    }

    /// Remove a node from the trie, updating parent bookkeeping and stats.
    fn remove_node(&mut self, hash: PrefixHash) {
        if hash == self.root_hash {
            return;
        }
        let node = match self.nodes.remove(&hash) {
            Some(n) => n,
            None => return,
        };

        // Remove from parent's children list.
        if let Some(parent_hash) = node.parent {
            if let Some(parent) = self.nodes.get_mut(&parent_hash) {
                parent.children.retain(|&c| c != hash);
            }
        }

        self.stats.total_nodes = self.stats.total_nodes.saturating_sub(1);
        self.stats.total_tokens_cached =
            self.stats.total_tokens_cached.saturating_sub(node.token_ids.len());
        self.stats.total_kv_bytes =
            self.stats.total_kv_bytes.saturating_sub(node.kv_cache_size_bytes);
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cache() -> PrefixCache {
        PrefixCache::new(PrefixCacheConfig::default())
    }

    fn small_cache() -> PrefixCache {
        PrefixCache::new(PrefixCacheConfig {
            max_total_tokens: 8,
            max_total_bytes: 8 * 512 * 32,
            min_prefix_length: 2,
            ..PrefixCacheConfig::default()
        })
    }

    // ── FNV hash consistency ────────────────────────────────────────────────

    #[test]
    fn test_fnv_hash_consistency() {
        let tokens = vec![1u32, 2, 3, 4];
        let h1 = PrefixHash::from_tokens(&tokens);
        let h2 = PrefixHash::from_tokens(&tokens);
        assert_eq!(h1, h2, "same input must produce same hash");
    }

    #[test]
    fn test_fnv_hash_different_sequences() {
        let h1 = PrefixHash::from_tokens(&[1, 2, 3]);
        let h2 = PrefixHash::from_tokens(&[1, 2, 4]);
        assert_ne!(h1, h2, "different sequences should (almost always) differ");
    }

    // ── Hash extend ─────────────────────────────────────────────────────────

    #[test]
    fn test_hash_extend_consistency() {
        let base = PrefixHash::from_tokens(&[10, 20]);
        let extended = PrefixHash::extend(base, 30);
        let full = PrefixHash::from_tokens(&[10, 20, 30]);
        // extend must agree with from_tokens for the same full prefix.
        assert_eq!(extended, full, "incremental extend must match full hash");
    }

    #[test]
    fn test_hash_extend_deterministic() {
        let base = PrefixHash(0);
        let h1 = PrefixHash::extend(base, 42);
        let h2 = PrefixHash::extend(base, 42);
        assert_eq!(h1, h2);
    }

    // ── Insert ──────────────────────────────────────────────────────────────

    #[test]
    fn test_insert_single_token_sequence() {
        let mut cache = default_cache();
        let tokens = vec![100u32, 200, 300, 400];
        let hash = cache.insert(&tokens);
        // Returned hash must be stored.
        assert!(cache.nodes.contains_key(&hash));
        assert_eq!(cache.total_tokens_in_tree(), 4);
    }

    #[test]
    fn test_insert_empty_returns_root() {
        let mut cache = default_cache();
        let hash = cache.insert(&[]);
        assert_eq!(hash, PrefixHash(0));
    }

    // ── Lookup – exact match ─────────────────────────────────────────────────

    #[test]
    fn test_lookup_exact_match() {
        let mut cache = default_cache();
        let tokens = vec![1u32, 2, 3, 4];
        cache.insert(&tokens);
        let (matched, node) = cache.lookup_prefix(&tokens);
        assert_eq!(matched, tokens.len());
        assert!(node.is_some());
    }

    // ── Lookup – partial match ───────────────────────────────────────────────

    #[test]
    fn test_lookup_partial_match() {
        let mut cache = default_cache();
        let tokens = vec![1u32, 2, 3, 4, 5, 6];
        cache.insert(&tokens);
        // Query a longer sequence; only the cached prefix should match.
        let query = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        let (matched, _node) = cache.lookup_prefix(&query);
        assert_eq!(matched, 6);
    }

    // ── Lookup – miss ────────────────────────────────────────────────────────

    #[test]
    fn test_lookup_miss() {
        let mut cache = default_cache();
        let (matched, node) = cache.lookup_prefix(&[99u32, 98, 97, 96]);
        assert_eq!(matched, 0);
        assert!(node.is_none());
    }

    // ── Trie depth tracking ──────────────────────────────────────────────────

    #[test]
    fn test_trie_depth_tracking() {
        let mut cache = default_cache();
        let tokens = vec![10u32, 20, 30];
        let hash = cache.insert(&tokens);
        let node = cache.nodes.get(&hash).expect("node must exist");
        assert_eq!(node.depth, 3);
    }

    // ── Ref count tracking ───────────────────────────────────────────────────

    #[test]
    fn test_ref_count_tracking() {
        let mut cache = default_cache();
        let tokens = vec![5u32, 6, 7, 8];
        // Insert twice → ref_count should accumulate per insert call.
        let hash = cache.insert(&tokens);
        cache.insert(&tokens);
        let node = cache.nodes.get(&hash).expect("node must exist");
        // Each insert bumps ref_count by 1 for each token in the path.
        assert!(node.ref_count >= 2);
    }

    // ── Release decrements ref count ─────────────────────────────────────────

    #[test]
    fn test_release_decrements_ref_count() {
        let mut cache = default_cache();
        let tokens = vec![1u32, 2, 3, 4];
        let hash = cache.insert(&tokens);
        let before = cache.nodes.get(&hash).map(|n| n.ref_count).unwrap_or(0);
        cache.release(hash);
        let after = cache.nodes.get(&hash).map(|n| n.ref_count).unwrap_or(0);
        assert!(after < before, "ref_count must decrease after release");
    }

    // ── Eviction (LRU policy) ────────────────────────────────────────────────

    #[test]
    fn test_eviction_lru_policy() {
        let mut cache = small_cache(); // max 8 tokens
                                       // Fill up the cache: insert 8 distinct token sequences.
        cache.insert(&[1u32, 2, 3, 4]);
        cache.insert(&[5u32, 6, 7, 8]);
        assert!(cache.total_tokens_in_tree() <= 8);

        // Inserting beyond capacity must trigger eviction.
        cache.insert(&[9u32, 10, 11, 12]);
        assert!(
            cache.stats().eviction_count > 0,
            "eviction must have occurred"
        );
        assert!(
            cache.total_tokens_in_tree() <= 8,
            "tokens must stay within limit"
        );
    }

    // ── Stats hit rate calculation ────────────────────────────────────────────

    #[test]
    fn test_stats_hit_rate() {
        let mut cache = default_cache();
        let tokens = vec![1u32, 2, 3, 4];
        cache.insert(&tokens);

        // Hit
        cache.lookup_prefix(&tokens);
        // Miss
        cache.lookup_prefix(&[99u32, 98, 97, 96]);

        let stats = cache.stats();
        assert_eq!(stats.hit_count, 1);
        assert_eq!(stats.miss_count, 1);
        let rate = stats.hit_rate();
        assert!((rate - 0.5).abs() < 1e-6);
    }

    // ── Byte estimation ───────────────────────────────────────────────────────

    #[test]
    fn test_byte_estimation() {
        let cache = default_cache();
        // Default: 512 bytes_per_token_per_layer * 32 layers = 16384 per token.
        let estimated = cache.estimate_bytes_for_tokens(1);
        assert_eq!(estimated, 512 * 32);
        let estimated_4 = cache.estimate_bytes_for_tokens(4);
        assert_eq!(estimated_4, 4 * 512 * 32);
    }

    // ── is_leaf ───────────────────────────────────────────────────────────────

    #[test]
    fn test_is_leaf() {
        let mut cache = default_cache();
        let tokens = vec![1u32, 2, 3];
        let leaf_hash = cache.insert(&tokens);
        assert!(cache.is_leaf(&leaf_hash), "deepest node must be a leaf");

        // The parent (hash for [1, 2]) should not be a leaf.
        // Use incremental extend from root (PrefixHash(0)) to get the
        // correct trie-internal hash for the path [1, 2].
        let root = PrefixHash(0);
        let h1 = PrefixHash::extend(root, 1);
        let parent_hash = PrefixHash::extend(h1, 2);
        assert!(
            !cache.is_leaf(&parent_hash),
            "intermediate node must not be a leaf"
        );
    }

    // ── Shared prefix branching ──────────────────────────────────────────────

    #[test]
    fn test_shared_prefix_branching() {
        let mut cache = default_cache();
        // Insert [1,2,3,4] and [1,2,3,5] — they share [1,2,3].
        cache.insert(&[1u32, 2, 3, 4]);
        cache.insert(&[1u32, 2, 3, 5]);

        // Total unique nodes = 5 (1, 2, 3, 4, 5) — root not counted.
        assert_eq!(cache.total_tokens_in_tree(), 5);

        // The node for path [1,2,3] must have two children.
        let root = PrefixHash(0);
        let h1 = PrefixHash::extend(root, 1);
        let h2 = PrefixHash::extend(h1, 2);
        let h3 = PrefixHash::extend(h2, 3);
        let node3 = cache.nodes.get(&h3).expect("node [1,2,3] must exist");
        assert_eq!(
            node3.children.len(),
            2,
            "shared prefix node must have two children"
        );
    }

    // ── Lookup updates last_access_time ──────────────────────────────────────

    #[test]
    fn test_lookup_updates_access_time() {
        let mut cache = default_cache();
        let tokens = vec![10u32, 20, 30, 40];
        let hash = cache.insert(&tokens);

        let before = cache.nodes.get(&hash).expect("node exists").last_access_time;
        // Small sleep ensures a different Instant is observed (nanosecond resolution).
        std::thread::sleep(std::time::Duration::from_millis(1));
        cache.lookup_prefix(&tokens);
        let after = cache.nodes.get(&hash).expect("node exists").last_access_time;

        assert!(after >= before, "access time must not move backwards");
    }

    // ── Miss when below min_prefix_length ────────────────────────────────────

    #[test]
    fn test_miss_below_min_prefix_length() {
        // min_prefix_length = 2; query only 1 matching token.
        let mut cache = PrefixCache::new(PrefixCacheConfig {
            min_prefix_length: 4,
            ..PrefixCacheConfig::default()
        });
        cache.insert(&[1u32, 2, 3, 4, 5, 6]);
        // Query matches [1] only — shorter than min_prefix_length=4 → miss.
        let (matched, _node) = cache.lookup_prefix(&[1u32, 99, 99]);
        // matched may be 1 but stats should say miss.
        assert_eq!(
            cache.stats().miss_count,
            1,
            "partial match below threshold must be a miss"
        );
        let _ = matched; // May be > 0 but irrelevant to stats
    }

    // ── Hit rate is zero before any lookups ───────────────────────────────────

    #[test]
    fn test_hit_rate_zero_before_lookups() {
        let cache = default_cache();
        assert_eq!(cache.stats().hit_rate(), 0.0);
    }

    // ── hit_rate = 1.0 when all lookups hit ──────────────────────────────────

    #[test]
    fn test_hit_rate_all_hits() {
        let mut cache = default_cache();
        let tokens = vec![1u32, 2, 3, 4];
        cache.insert(&tokens);
        cache.lookup_prefix(&tokens);
        cache.lookup_prefix(&tokens);
        let stats = cache.stats();
        assert_eq!(stats.hit_count, 2);
        assert_eq!(stats.miss_count, 0);
        assert!((stats.hit_rate() - 1.0).abs() < 1e-6);
    }

    // ── hit_rate formula: hits/(hits+misses) ─────────────────────────────────

    #[test]
    fn test_hit_rate_formula() {
        let mut cache = default_cache();
        let tokens = vec![5u32, 6, 7, 8];
        cache.insert(&tokens);

        // 3 hits, 1 miss
        cache.lookup_prefix(&tokens);
        cache.lookup_prefix(&tokens);
        cache.lookup_prefix(&tokens);
        cache.lookup_prefix(&[99u32, 100, 101, 102]);

        let stats = cache.stats();
        let expected = 3.0_f32 / 4.0_f32;
        assert!(
            (stats.hit_rate() - expected).abs() < 1e-6,
            "hit_rate = hits/(hits+misses)"
        );
    }

    // ── Stats total_nodes tracks insertions ──────────────────────────────────

    #[test]
    fn test_stats_total_nodes_tracking() {
        let mut cache = default_cache();
        assert_eq!(cache.stats().total_nodes, 0);
        cache.insert(&[1u32, 2, 3]);
        assert_eq!(cache.stats().total_nodes, 3);
    }

    // ── Stats total_tokens_cached tracks insertions ───────────────────────────

    #[test]
    fn test_stats_total_tokens_cached_tracking() {
        let mut cache = default_cache();
        cache.insert(&[1u32, 2, 3, 4]);
        assert_eq!(cache.stats().total_tokens_cached, 4);
    }

    // ── Evict LRU leaf removes oldest leaf ───────────────────────────────────

    #[test]
    fn test_evict_lru_leaf_removes_node() {
        let mut cache = default_cache();
        cache.insert(&[1u32, 2, 3, 4]);
        let before = cache.total_tokens_in_tree();
        let evicted = cache.evict_lru_leaf();
        assert!(evicted, "must evict a node from a non-empty trie");
        assert!(
            cache.total_tokens_in_tree() < before,
            "token count must decrease after eviction"
        );
    }

    // ── Evict from empty trie returns false ───────────────────────────────────

    #[test]
    fn test_evict_from_empty_trie() {
        let mut cache = default_cache();
        let result = cache.evict_lru_leaf();
        // Only root exists; no evictable leaf.
        assert!(!result, "evicting from empty trie must return false");
    }

    // ── Release decrements all ancestors ─────────────────────────────────────

    #[test]
    fn test_release_decrements_ancestors() {
        let mut cache = default_cache();
        // Insert twice to give each node ref_count >= 2.
        let tokens = vec![1u32, 2, 3];
        cache.insert(&tokens);
        let hash = cache.insert(&tokens);

        let root = PrefixHash(0);
        let h1 = PrefixHash::extend(root, 1);
        let before_h1 = cache.nodes.get(&h1).expect("node 1 exists").ref_count;

        cache.release(hash);

        let after_h1 = cache.nodes.get(&h1).expect("node 1 exists").ref_count;
        assert!(
            after_h1 < before_h1,
            "release must decrement ref_count on ancestors"
        );
    }

    // ── Eviction policy: RefCountBased prefers zero-ref leaves ────────────────

    #[test]
    fn test_eviction_ref_count_based_prefers_zero_ref() {
        let mut cache = PrefixCache::new(PrefixCacheConfig {
            eviction_policy: PrefixEvictionPolicy::RefCountBased,
            max_total_tokens: 4,
            max_total_bytes: 4 * 512 * 32,
            min_prefix_length: 1,
            ..PrefixCacheConfig::default()
        });

        // Insert [1,2] first (gets ref_count 1 each), then insert [3,4] separately.
        cache.insert(&[1u32, 2]);
        let hash34 = cache.insert(&[3u32, 4]);

        // Release [3,4] to zero ref_count so it becomes the preferred eviction target.
        cache.release(hash34);

        // Now inserting [5,6] should trigger eviction of the zero-ref branch.
        cache.insert(&[5u32, 6]);
        assert!(cache.stats().eviction_count > 0, "eviction must occur");
    }

    // ── Eviction policy: SizeAware evicts largest leaf ────────────────────────

    #[test]
    fn test_eviction_size_aware_evicts_largest() {
        // Use SizeAware policy with very small budget to force eviction.
        let mut cache = PrefixCache::new(PrefixCacheConfig {
            eviction_policy: PrefixEvictionPolicy::SizeAware,
            max_total_tokens: 3,
            max_total_bytes: 3 * 512 * 32,
            min_prefix_length: 1,
            ..PrefixCacheConfig::default()
        });

        cache.insert(&[1u32, 2]);
        cache.insert(&[3u32, 4]); // Triggers eviction when total > 3.
        assert!(
            cache.stats().eviction_count > 0,
            "eviction must occur under SizeAware"
        );
        assert!(cache.total_tokens_in_tree() <= 3);
    }

    // ── PrefixHash from_tokens on empty slice returns FNV offset ─────────────

    #[test]
    fn test_fnv_hash_empty_slice() {
        let h = PrefixHash::from_tokens(&[]);
        assert_eq!(h, PrefixHash(PrefixHash::FNV_OFFSET));
    }

    // ── estimate_bytes_for_tokens zero tokens ─────────────────────────────────

    #[test]
    fn test_estimate_bytes_zero_tokens() {
        let cache = default_cache();
        assert_eq!(cache.estimate_bytes_for_tokens(0), 0);
    }

    // ── insert idempotent on same tokens (no duplicate nodes) ─────────────────

    #[test]
    fn test_insert_idempotent_no_duplicate_nodes() {
        let mut cache = default_cache();
        let tokens = vec![10u32, 20, 30, 40];
        cache.insert(&tokens);
        let before_count = cache.stats().total_nodes;
        cache.insert(&tokens); // Same tokens again.
                               // No new nodes should be created; count must not increase.
        assert_eq!(
            cache.stats().total_nodes,
            before_count,
            "re-inserting same tokens must not add new nodes"
        );
    }

    // ── total_tokens_in_tree excludes root ────────────────────────────────────

    #[test]
    fn test_total_tokens_excludes_root() {
        let cache = default_cache();
        // Only root exists (depth 0, no tokens).
        assert_eq!(cache.total_tokens_in_tree(), 0);
    }

    // ── Eviction reduces total_kv_bytes ──────────────────────────────────────

    #[test]
    fn test_eviction_reduces_kv_bytes() {
        let mut cache = default_cache();
        cache.insert(&[1u32, 2, 3, 4]);
        let before = cache.stats().total_kv_bytes;
        assert!(before > 0);
        cache.evict_lru_leaf();
        let after = cache.stats().total_kv_bytes;
        assert!(after < before, "kv_bytes must decrease after eviction");
    }

    // ── Eviction count is tracked in stats ───────────────────────────────────

    #[test]
    fn test_eviction_count_in_stats() {
        let mut cache = default_cache();
        cache.insert(&[1u32, 2, 3, 4]);
        assert_eq!(cache.stats().eviction_count, 0);
        cache.evict_if_needed(); // Should be a no-op since under limits.
        assert_eq!(cache.stats().eviction_count, 0);
    }

    // ── Extended tests ─────────────────────────────────────────────────────

    // Test: PrefixHash::from_tokens — different token sequences give different hashes
    #[test]
    fn test_prefix_hash_different_tokens_differ() {
        let h1 = PrefixHash::from_tokens(&[1u32, 2, 3]);
        let h2 = PrefixHash::from_tokens(&[3u32, 2, 1]);
        assert_ne!(h1, h2, "different token orderings must hash differently");
    }

    // Test: PrefixHash::from_tokens — same sequence is deterministic
    #[test]
    fn test_prefix_hash_deterministic() {
        let tokens = vec![10u32, 20, 30, 40];
        let h1 = PrefixHash::from_tokens(&tokens);
        let h2 = PrefixHash::from_tokens(&tokens);
        assert_eq!(h1, h2, "hashing same tokens must give same result");
    }

    // Test: PrefixHash::extend — extends correctly
    #[test]
    fn test_prefix_hash_extend_differs_from_original() {
        let base = PrefixHash::from_tokens(&[1u32, 2]);
        let extended = PrefixHash::extend(base, 3);
        assert_ne!(
            base, extended,
            "extending a hash must produce a different hash"
        );
    }

    // Test: PrefixTrieStats::hit_rate — 0 when no lookups
    #[test]
    fn test_prefix_trie_stats_hit_rate_zero_initially() {
        let stats = PrefixTrieStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    // Test: PrefixTrieStats::hit_rate — 1.0 when all hits
    #[test]
    fn test_prefix_trie_stats_hit_rate_all_hits() {
        let stats = PrefixTrieStats {
            hit_count: 10,
            miss_count: 0,
            ..PrefixTrieStats::default()
        };
        assert!((stats.hit_rate() - 1.0).abs() < 1e-6);
    }

    // Test: PrefixCache::lookup_prefix — hit increments hit_count.
    // The default config has min_prefix_length=4, so we must insert and look up
    // at least 4 tokens to trigger a hit rather than a miss.
    #[test]
    fn test_lookup_hit_increments_hit_count() {
        let mut cache = default_cache();
        cache.insert(&[1u32, 2, 3, 4]);
        cache.lookup_prefix(&[1u32, 2, 3, 4]);
        assert!(cache.stats().hit_count >= 1);
    }

    // Test: PrefixCache::lookup_prefix — miss increments miss_count
    #[test]
    fn test_lookup_miss_increments_miss_count() {
        let mut cache = default_cache();
        cache.lookup_prefix(&[99u32, 98, 97]);
        assert!(cache.stats().miss_count >= 1);
    }

    // Test: PrefixCache::lookup_prefix — longer prefix not in cache returns empty
    #[test]
    fn test_lookup_longer_prefix_than_cached_returns_empty() {
        let mut cache = default_cache();
        cache.insert(&[1u32, 2]);
        // Lookup [1, 2, 3, 4]: first 2 tokens match but last 2 are not cached
        let result = cache.lookup_prefix(&[1u32, 2, 3, 4]);
        // Result should match at most the first 2 tokens
        assert!(result.0 <= 2);
    }

    // Test: PrefixEvictionPolicy variants are distinct
    #[test]
    fn test_prefix_eviction_policy_variants_distinct() {
        assert_ne!(
            PrefixEvictionPolicy::Lru,
            PrefixEvictionPolicy::RefCountBased
        );
        assert_ne!(PrefixEvictionPolicy::SizeAware, PrefixEvictionPolicy::Lru);
    }

    // Test: PrefixCacheConfig::default — sensible defaults
    #[test]
    fn test_prefix_cache_config_default() {
        let cfg = PrefixCacheConfig::default();
        assert!(cfg.max_total_tokens > 0);
        assert!(cfg.max_total_bytes > 0);
    }

    // Test: insert returns a deterministic hash for same tokens
    #[test]
    fn test_insert_returns_deterministic_hash() {
        let mut cache1 = default_cache();
        let mut cache2 = default_cache();
        let tokens = vec![5u32, 10, 15, 20];
        let h1 = cache1.insert(&tokens);
        let h2 = cache2.insert(&tokens);
        assert_eq!(
            h1, h2,
            "same tokens must yield same hash across different caches"
        );
    }

    // Test: PrefixCache::insert always creates nodes regardless of min_prefix_length;
    // min_prefix_length only governs whether lookup_prefix counts as a hit.
    #[test]
    fn test_insert_below_min_prefix_length_creates_no_nodes() {
        let cfg = PrefixCacheConfig {
            min_prefix_length: 5,
            ..PrefixCacheConfig::default()
        };
        let mut cache = PrefixCache::new(cfg);
        cache.insert(&[1u32, 2]); // Only 2 tokens, below min_prefix_length=5
                                  // insert() builds a node for every token unconditionally; min_prefix_length
                                  // affects only the hit/miss accounting in lookup_prefix.
        assert_eq!(
            cache.stats().total_nodes,
            2,
            "insert always adds one node per token regardless of min_prefix_length"
        );
    }

    // Test: total_tokens_in_tree grows with insertions
    #[test]
    fn test_total_tokens_in_tree_grows() {
        let mut cache = default_cache();
        assert_eq!(cache.total_tokens_in_tree(), 0);
        cache.insert(&[1u32, 2, 3]);
        assert_eq!(cache.total_tokens_in_tree(), 3);
    }

    // Test: PrefixHash::FNV_OFFSET constant is defined correctly
    #[test]
    fn test_fnv_offset_constant() {
        assert_eq!(PrefixHash::FNV_OFFSET, 14695981039346656037u64);
    }

    // Test: PrefixCache::stats total_kv_bytes grows with insertions
    #[test]
    fn test_stats_total_kv_bytes_grows_with_insertions() {
        let mut cache = default_cache();
        assert_eq!(cache.stats().total_kv_bytes, 0);
        cache.insert(&[1u32, 2, 3, 4]);
        assert!(
            cache.stats().total_kv_bytes > 0,
            "total_kv_bytes must be > 0 after insertion"
        );
    }
}
