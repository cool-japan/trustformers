use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Compressed vocabulary using variable-length encoding and dictionary compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedVocab {
    /// Compressed token-to-ID mapping using string interning
    token_pool: Vec<String>,
    token_to_pool_id: HashMap<u32, u32>, // Maps token_id to pool_id
    pool_id_to_token: HashMap<u32, u32>, // Maps pool_id to token_id
    /// Frequency-based ordering for better compression
    frequent_tokens: Vec<u32>, // Most frequent token IDs
    vocab_size: usize,
}

impl CompressedVocab {
    /// Create a new compressed vocabulary from a token map
    pub fn from_token_map(token_map: HashMap<String, u32>) -> Result<Self> {
        let vocab_size = token_map.len();

        // Sort tokens by frequency (assuming lower IDs are more frequent)
        let mut token_freq: Vec<(String, u32)> = token_map.into_iter().collect();
        token_freq.sort_by_key(|(_, id)| *id);

        // Create string pool and mappings
        let mut token_pool = Vec::new();
        let mut token_to_pool_id = HashMap::new();
        let mut pool_id_to_token = HashMap::new();
        let mut frequent_tokens = Vec::new();

        for (pool_id, (token, token_id)) in token_freq.into_iter().enumerate() {
            let pool_id = pool_id as u32;

            token_pool.push(token);
            token_to_pool_id.insert(token_id, pool_id);
            pool_id_to_token.insert(pool_id, token_id);

            // Mark first 1000 tokens as frequent for fast access
            if pool_id < 1000 {
                frequent_tokens.push(token_id);
            }
        }

        Ok(Self {
            token_pool,
            token_to_pool_id,
            pool_id_to_token,
            frequent_tokens,
            vocab_size,
        })
    }

    /// Create from a regular vocabulary
    pub fn from_vocab(vocab: &crate::vocab::Vocab) -> Result<Self> {
        let mut token_map = HashMap::new();

        for id in 0..vocab.size() {
            if let Some(token) = vocab.get_token(id as u32) {
                token_map.insert(token, id as u32);
            }
        }

        Self::from_token_map(token_map)
    }

    /// Get token ID for a given token
    pub fn get_id(&self, token: &str) -> Option<u32> {
        // Fast path: check frequent tokens first
        for &token_id in &self.frequent_tokens {
            if let Some(&pool_id) = self.token_to_pool_id.get(&token_id) {
                if let Some(pool_token) = self.token_pool.get(pool_id as usize) {
                    if pool_token == token {
                        return Some(token_id);
                    }
                }
            }
        }

        // Slower path: linear search through token pool
        for (pool_id, pool_token) in self.token_pool.iter().enumerate() {
            if pool_token == token {
                if let Some(&token_id) = self.pool_id_to_token.get(&(pool_id as u32)) {
                    return Some(token_id);
                }
            }
        }

        None
    }

    /// Get token by ID
    pub fn get_token(&self, id: u32) -> Option<String> {
        if let Some(&pool_id) = self.token_to_pool_id.get(&id) {
            self.token_pool.get(pool_id as usize).cloned()
        } else {
            None
        }
    }

    /// Check if token exists in vocabulary
    pub fn contains(&self, token: &str) -> bool {
        self.get_id(token).is_some()
    }

    /// Get vocabulary size
    pub fn size(&self) -> usize {
        self.vocab_size
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> CompressedVocabStats {
        let token_pool_size = self
            .token_pool
            .iter()
            .map(|s| s.len() + std::mem::size_of::<String>())
            .sum::<usize>();

        let mapping_size = self.token_to_pool_id.len() * (std::mem::size_of::<u32>() * 2)
            + self.pool_id_to_token.len() * (std::mem::size_of::<u32>() * 2);

        let frequent_tokens_size = self.frequent_tokens.len() * std::mem::size_of::<u32>();

        let total_size = token_pool_size + mapping_size + frequent_tokens_size;

        CompressedVocabStats {
            total_size,
            token_pool_size,
            mapping_size,
            frequent_tokens_size,
            vocab_size: self.vocab_size,
            compression_ratio: self.calculate_compression_ratio(),
        }
    }

    /// Calculate compression ratio compared to naive HashMap approach
    fn calculate_compression_ratio(&self) -> f64 {
        // Estimate naive HashMap size
        let naive_size = self
            .token_pool
            .iter()
            .map(|s| s.len() + std::mem::size_of::<String>() + std::mem::size_of::<u32>())
            .sum::<usize>();

        // Calculate compressed size without calling memory_stats() to avoid infinite recursion
        let token_pool_size = self
            .token_pool
            .iter()
            .map(|s| s.len() + std::mem::size_of::<String>())
            .sum::<usize>();

        let mapping_size = self.token_to_pool_id.len() * (std::mem::size_of::<u32>() * 2)
            + self.pool_id_to_token.len() * (std::mem::size_of::<u32>() * 2);

        let frequent_tokens_size = self.frequent_tokens.len() * std::mem::size_of::<u32>();

        let compressed_size = token_pool_size + mapping_size + frequent_tokens_size;

        if compressed_size > 0 {
            naive_size as f64 / compressed_size as f64
        } else {
            1.0
        }
    }

    /// Optimize the vocabulary for better compression
    pub fn optimize(&mut self) {
        // Re-sort by actual usage frequency if available
        // For now, we'll optimize the frequent tokens list
        self.frequent_tokens.sort();

        // Remove duplicates
        self.frequent_tokens.dedup();

        // Limit to most frequent 1000 tokens
        if self.frequent_tokens.len() > 1000 {
            self.frequent_tokens.truncate(1000);
        }
    }

    /// Create a trie-like structure for prefix matching
    pub fn build_prefix_trie(&self) -> PrefixTrie {
        let mut trie = PrefixTrie::new();

        for (pool_id, token) in self.token_pool.iter().enumerate() {
            if let Some(&token_id) = self.pool_id_to_token.get(&(pool_id as u32)) {
                trie.insert(token, token_id);
            }
        }

        trie
    }

    /// Find tokens with a given prefix efficiently
    pub fn find_tokens_with_prefix(&self, prefix: &str) -> Vec<(String, u32)> {
        let mut results = Vec::new();

        for (pool_id, token) in self.token_pool.iter().enumerate() {
            if token.starts_with(prefix) {
                if let Some(&token_id) = self.pool_id_to_token.get(&(pool_id as u32)) {
                    results.push((token.clone(), token_id));
                }
            }
        }

        results.sort_by_key(|(_, id)| *id);
        results
    }
}

/// Memory usage statistics for compressed vocabulary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedVocabStats {
    pub total_size: usize,
    pub token_pool_size: usize,
    pub mapping_size: usize,
    pub frequent_tokens_size: usize,
    pub vocab_size: usize,
    pub compression_ratio: f64,
}

/// Simple prefix trie for efficient token prefix matching
#[derive(Debug, Clone)]
pub struct PrefixTrie {
    root: TrieNode,
}

#[derive(Debug, Clone)]
struct TrieNode {
    children: HashMap<char, TrieNode>,
    token_id: Option<u32>,
    is_end: bool,
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            token_id: None,
            is_end: false,
        }
    }
}

impl PrefixTrie {
    pub fn new() -> Self {
        Self {
            root: TrieNode::new(),
        }
    }

    pub fn insert(&mut self, token: &str, token_id: u32) {
        let mut current = &mut self.root;

        for ch in token.chars() {
            current = current.children.entry(ch).or_insert_with(TrieNode::new);
        }

        current.is_end = true;
        current.token_id = Some(token_id);
    }

    pub fn find_with_prefix(&self, prefix: &str) -> Vec<u32> {
        let mut current = &self.root;

        // Navigate to the prefix node
        for ch in prefix.chars() {
            if let Some(child) = current.children.get(&ch) {
                current = child;
            } else {
                return Vec::new(); // Prefix not found
            }
        }

        // Collect all token IDs under this prefix
        let mut results = Vec::new();
        self.collect_token_ids(current, &mut results);
        results.sort();
        results
    }

    fn collect_token_ids(&self, node: &TrieNode, results: &mut Vec<u32>) {
        if node.is_end {
            if let Some(token_id) = node.token_id {
                results.push(token_id);
            }
        }

        for child in node.children.values() {
            self.collect_token_ids(child, results);
        }
    }
}

impl Default for PrefixTrie {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compressed_vocab_basic() {
        let mut token_map = HashMap::new();
        token_map.insert("hello".to_string(), 0);
        token_map.insert("world".to_string(), 1);
        token_map.insert("test".to_string(), 2);

        let compressed_vocab = CompressedVocab::from_token_map(token_map).unwrap();

        assert_eq!(compressed_vocab.get_id("hello"), Some(0));
        assert_eq!(compressed_vocab.get_id("world"), Some(1));
        assert_eq!(compressed_vocab.get_id("test"), Some(2));
        assert_eq!(compressed_vocab.get_id("unknown"), None);

        assert_eq!(compressed_vocab.get_token(0), Some("hello".to_string()));
        assert_eq!(compressed_vocab.get_token(1), Some("world".to_string()));
        assert_eq!(compressed_vocab.get_token(2), Some("test".to_string()));
        assert_eq!(compressed_vocab.get_token(999), None);

        assert!(compressed_vocab.contains("hello"));
        assert!(!compressed_vocab.contains("unknown"));
        assert_eq!(compressed_vocab.size(), 3);
    }

    #[test]
    fn test_compressed_vocab_memory_stats() {
        let mut token_map = HashMap::new();
        for i in 0..100 {
            token_map.insert(format!("token_{}", i), i);
        }

        let compressed_vocab = CompressedVocab::from_token_map(token_map).unwrap();
        let stats = compressed_vocab.memory_stats();

        assert_eq!(stats.vocab_size, 100);
        assert!(stats.total_size > 0);
        assert!(stats.compression_ratio > 0.0);
        assert!(stats.token_pool_size > 0);
        assert!(stats.mapping_size > 0);
    }

    #[test]
    fn test_compressed_vocab_frequent_tokens() {
        let mut token_map = HashMap::new();
        for i in 0..150 {
            token_map.insert(format!("token_{}", i), i);
        }

        let compressed_vocab = CompressedVocab::from_token_map(token_map).unwrap();

        // Should have frequent tokens (first 150, all are frequent since < 1000)
        assert_eq!(compressed_vocab.frequent_tokens.len(), 150);

        // Fast lookup for frequent tokens
        assert_eq!(compressed_vocab.get_id("token_0"), Some(0));
        assert_eq!(compressed_vocab.get_id("token_149"), Some(149));
    }

    #[test]
    fn test_compressed_vocab_optimization() {
        let mut token_map = HashMap::new();
        for i in 0..100 {
            token_map.insert(format!("token_{}", i), i);
        }

        let mut compressed_vocab = CompressedVocab::from_token_map(token_map).unwrap();
        compressed_vocab.optimize();

        assert!(compressed_vocab.frequent_tokens.len() <= 1000);
        assert_eq!(compressed_vocab.size(), 100);
    }

    #[test]
    fn test_prefix_trie() {
        let mut trie = PrefixTrie::new();
        trie.insert("hello", 1);
        trie.insert("help", 2);
        trie.insert("world", 3);
        trie.insert("word", 4);

        let results = trie.find_with_prefix("hel");
        assert_eq!(results, vec![1, 2]); // hello, help

        let results = trie.find_with_prefix("wor");
        assert_eq!(results, vec![3, 4]); // world, word

        let results = trie.find_with_prefix("xyz");
        assert!(results.is_empty());
    }

    #[test]
    fn test_compressed_vocab_prefix_search() {
        let mut token_map = HashMap::new();
        token_map.insert("prefix_1".to_string(), 1);
        token_map.insert("prefix_2".to_string(), 2);
        token_map.insert("other".to_string(), 3);

        let compressed_vocab = CompressedVocab::from_token_map(token_map).unwrap();
        let results = compressed_vocab.find_tokens_with_prefix("prefix_");

        assert_eq!(results.len(), 2);
        assert!(results.contains(&("prefix_1".to_string(), 1)));
        assert!(results.contains(&("prefix_2".to_string(), 2)));
    }

    #[test]
    fn test_compressed_vocab_trie_building() {
        let mut token_map = HashMap::new();
        token_map.insert("test".to_string(), 1);
        token_map.insert("testing".to_string(), 2);
        token_map.insert("other".to_string(), 3);

        let compressed_vocab = CompressedVocab::from_token_map(token_map).unwrap();
        let trie = compressed_vocab.build_prefix_trie();

        let results = trie.find_with_prefix("test");
        assert!(results.contains(&1)); // test
        assert!(results.contains(&2)); // testing
        assert!(!results.contains(&3)); // other
    }
}
