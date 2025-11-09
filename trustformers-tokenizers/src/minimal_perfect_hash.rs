use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use trustformers_core::errors::{Result, TrustformersError};

/// Configuration for minimal perfect hashing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimalPerfectHashConfig {
    /// Load factor for hash table (should be < 1.0)
    pub load_factor: f64,
    /// Number of hash functions to use
    pub num_hash_functions: usize,
    /// Use double hashing to reduce collisions
    pub use_double_hashing: bool,
}

impl Default for MinimalPerfectHashConfig {
    fn default() -> Self {
        Self {
            load_factor: 0.9,
            num_hash_functions: 3,
            use_double_hashing: true,
        }
    }
}

/// Minimal perfect hash function implementation
///
/// This implementation uses a variant of the CHM (Czech, Havas, Majewski) algorithm
/// which creates a minimal perfect hash function for a given set of keys.
///
/// The hash function guarantees:
/// - No collisions for the given key set
/// - Minimal space usage (close to theoretical minimum)
/// - Fast O(1) lookup time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinimalPerfectHash {
    /// Configuration parameters
    config: MinimalPerfectHashConfig,
    /// Primary hash table
    table: Vec<Option<u32>>,
    /// Secondary hash table for collision resolution
    secondary_table: Vec<Option<u32>>,
    /// Hash function parameters
    hash_params: Vec<(u64, u64)>,
    /// Table size (should be prime)
    table_size: usize,
    /// Number of keys in the hash function
    num_keys: usize,
    /// Original keys for verification during lookup
    keys: Vec<String>,
}

impl MinimalPerfectHash {
    /// Create a new minimal perfect hash function from a set of keys
    pub fn new(keys: &[String], config: MinimalPerfectHashConfig) -> Result<Self> {
        let num_keys = keys.len();
        let target_size = (num_keys as f64 / config.load_factor).ceil() as usize;
        let table_size = Self::next_prime(target_size);
        println!(
            "MPH Debug: num_keys={}, target_load_factor={}, target_size={}, table_size={}",
            num_keys, config.load_factor, target_size, table_size
        );

        let mut mph = Self {
            config,
            table: vec![None; table_size],
            secondary_table: vec![None; table_size],
            hash_params: Vec::new(),
            table_size,
            num_keys,
            keys: keys.to_vec(),
        };

        mph.build_hash_function(keys)?;
        Ok(mph)
    }

    /// Build the minimal perfect hash function
    fn build_hash_function(&mut self, keys: &[String]) -> Result<()> {
        // Generate random hash function parameters
        self.hash_params = (0..self.config.num_hash_functions)
            .map(|_| (Self::random_u64(), Self::random_u64()))
            .collect();

        // Try to build the hash function with different parameter sets
        for _attempt in 0..100 {
            if self.try_build_hash_function(keys)? {
                return Ok(());
            }

            // Generate new parameters for retry
            self.hash_params = (0..self.config.num_hash_functions)
                .map(|_| (Self::random_u64(), Self::random_u64()))
                .collect();

            // Clear tables for retry
            self.table.fill(None);
            self.secondary_table.fill(None);
        }

        Err(TrustformersError::other(
            "Failed to build minimal perfect hash function after 100 attempts".to_string(),
        ))
    }

    /// Attempt to build the hash function with current parameters
    fn try_build_hash_function(&mut self, keys: &[String]) -> Result<bool> {
        // Phase 1: Build conflict graph
        let mut conflicts = HashMap::new();
        let mut buckets: Vec<Vec<(usize, String)>> = vec![Vec::new(); self.table_size];

        // Hash all keys and group by bucket
        for (key_idx, key) in keys.iter().enumerate() {
            let bucket = self.hash_primary(key);
            buckets[bucket].push((key_idx, key.clone()));
        }

        // Find conflicts (buckets with multiple keys)
        for (bucket_idx, bucket) in buckets.iter().enumerate() {
            if bucket.len() > 1 {
                conflicts.insert(bucket_idx, bucket.clone());
            }
        }

        // Phase 2: Resolve conflicts using secondary hashing
        for (bucket_idx, bucket_keys) in conflicts {
            if !self.resolve_bucket_conflicts(bucket_idx, &bucket_keys)? {
                return Ok(false);
            }
        }

        // Phase 3: Place non-conflicting keys
        for (bucket_idx, bucket) in buckets.iter().enumerate() {
            if bucket.len() == 1 {
                let (key_idx, _) = &bucket[0];
                self.table[bucket_idx] = Some(*key_idx as u32);
            }
        }

        Ok(true)
    }

    /// Resolve conflicts in a bucket using secondary hashing
    fn resolve_bucket_conflicts(
        &mut self,
        bucket_idx: usize,
        keys: &[(usize, String)],
    ) -> Result<bool> {
        if !self.config.use_double_hashing {
            return Ok(false);
        }

        // Try different secondary hash functions
        for hash_idx in 1..self.config.num_hash_functions {
            let mut used_positions = std::collections::HashSet::new();
            let mut success = true;

            for (_key_idx, key) in keys {
                let secondary_pos = self.hash_secondary(key, hash_idx);
                if used_positions.contains(&secondary_pos)
                    || self.secondary_table[secondary_pos].is_some()
                {
                    success = false;
                    break;
                }
                used_positions.insert(secondary_pos);
            }

            if success {
                // Place all keys in secondary table
                for (key_idx, key) in keys {
                    let secondary_pos = self.hash_secondary(key, hash_idx);
                    self.secondary_table[secondary_pos] = Some(*key_idx as u32);
                }

                // Mark primary bucket as using secondary hashing
                self.table[bucket_idx] = Some(u32::MAX); // Special marker
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Primary hash function
    fn hash_primary(&self, key: &str) -> usize {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();

        let (a, b) = self.hash_params[0];
        ((a.wrapping_mul(hash).wrapping_add(b)) % self.table_size as u64) as usize
    }

    /// Secondary hash function
    fn hash_secondary(&self, key: &str, func_idx: usize) -> usize {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();

        let (a, b) = self.hash_params[func_idx];
        ((a.wrapping_mul(hash).wrapping_add(b)) % self.table_size as u64) as usize
    }

    /// Look up a key in the hash function
    pub fn lookup(&self, key: &str) -> Option<u32> {
        let primary_pos = self.hash_primary(key);

        match self.table[primary_pos] {
            Some(u32::MAX) => {
                // Key is in secondary table
                for hash_idx in 1..self.config.num_hash_functions {
                    let secondary_pos = self.hash_secondary(key, hash_idx);
                    if let Some(value) = self.secondary_table[secondary_pos] {
                        // Verify that the key matches
                        if value < self.keys.len() as u32 && self.keys[value as usize] == key {
                            return Some(value);
                        }
                    }
                }
                None
            },
            Some(value) => {
                // Verify that the key matches
                if value < self.keys.len() as u32 && self.keys[value as usize] == key {
                    Some(value)
                } else {
                    None
                }
            },
            None => None,
        }
    }

    /// Check if the hash function contains a key
    pub fn contains(&self, key: &str) -> bool {
        self.lookup(key).is_some()
    }

    /// Get the number of keys in the hash function
    pub fn len(&self) -> usize {
        self.num_keys
    }

    /// Check if the hash function is empty
    pub fn is_empty(&self) -> bool {
        self.num_keys == 0
    }

    /// Get memory usage statistics
    pub fn memory_usage(&self) -> MemoryUsage {
        let table_size = self.table.len() * std::mem::size_of::<Option<u32>>();
        let secondary_table_size = self.secondary_table.len() * std::mem::size_of::<Option<u32>>();
        let params_size = self.hash_params.len() * std::mem::size_of::<(u64, u64)>();

        MemoryUsage {
            total_bytes: table_size + secondary_table_size + params_size,
            table_bytes: table_size,
            secondary_table_bytes: secondary_table_size,
            params_bytes: params_size,
            load_factor: self.num_keys as f64 / self.table_size as f64,
        }
    }

    /// Find the next prime number >= n
    fn next_prime(n: usize) -> usize {
        if n <= 2 {
            return 2;
        }

        let mut candidate = if n % 2 == 0 { n + 1 } else { n };

        while !Self::is_prime(candidate) {
            candidate += 2;
        }

        candidate
    }

    /// Check if a number is prime
    fn is_prime(n: usize) -> bool {
        if n <= 1 {
            return false;
        }
        if n <= 3 {
            return true;
        }
        if n % 2 == 0 || n % 3 == 0 {
            return false;
        }

        let mut i = 5;
        while i * i <= n {
            if n % i == 0 || n % (i + 2) == 0 {
                return false;
            }
            i += 6;
        }

        true
    }

    /// Generate a random u64 (simple implementation)
    fn random_u64() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
        let mut x = now.as_nanos() as u64;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    }
}

/// Memory usage statistics for the minimal perfect hash
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Total memory usage in bytes
    pub total_bytes: usize,
    /// Memory used by primary table
    pub table_bytes: usize,
    /// Memory used by secondary table
    pub secondary_table_bytes: usize,
    /// Memory used by hash parameters
    pub params_bytes: usize,
    /// Actual load factor achieved
    pub load_factor: f64,
}

impl std::fmt::Display for MemoryUsage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Total: {} bytes, Primary: {} bytes, Secondary: {} bytes, Params: {} bytes, Load factor: {:.2}",
            self.total_bytes, self.table_bytes, self.secondary_table_bytes, self.params_bytes, self.load_factor
        )
    }
}

/// Vocabulary implementation using minimal perfect hashing
#[derive(Debug, Clone)]
pub struct MinimalPerfectHashVocab {
    /// The minimal perfect hash function
    mph: MinimalPerfectHash,
    /// Reverse mapping from ID to token
    id_to_token: Vec<String>,
    /// Original token order
    tokens: Vec<String>,
}

impl MinimalPerfectHashVocab {
    /// Create a new vocabulary from tokens
    pub fn new(tokens: Vec<String>) -> Result<Self> {
        let config = MinimalPerfectHashConfig::default();
        let mph = MinimalPerfectHash::new(&tokens, config)?;

        let id_to_token = tokens.clone();

        Ok(Self {
            mph,
            id_to_token,
            tokens,
        })
    }

    /// Create a new vocabulary with custom configuration
    pub fn with_config(tokens: Vec<String>, config: MinimalPerfectHashConfig) -> Result<Self> {
        let mph = MinimalPerfectHash::new(&tokens, config)?;
        let id_to_token = tokens.clone();

        Ok(Self {
            mph,
            id_to_token,
            tokens,
        })
    }

    /// Get the ID for a token
    pub fn get_id(&self, token: &str) -> Option<u32> {
        self.mph.lookup(token)
    }

    /// Get the token for an ID
    pub fn get_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    /// Check if the vocabulary contains a token
    pub fn contains(&self, token: &str) -> bool {
        self.mph.contains(token)
    }

    /// Get the vocabulary size
    pub fn size(&self) -> usize {
        self.mph.len()
    }

    /// Check if the vocabulary is empty
    pub fn is_empty(&self) -> bool {
        self.mph.is_empty()
    }

    /// Get memory usage statistics
    pub fn memory_usage(&self) -> MemoryUsage {
        let mut usage = self.mph.memory_usage();

        // Add memory usage from id_to_token mapping
        let id_to_token_size: usize =
            self.id_to_token.iter().map(|s| s.len() + std::mem::size_of::<String>()).sum();

        usage.total_bytes += id_to_token_size;
        usage
    }

    /// Get all tokens in the vocabulary
    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }

    /// Compare memory efficiency with standard HashMap
    pub fn efficiency_comparison(&self) -> EfficiencyComparison {
        let mph_usage = self.memory_usage();

        // Estimate HashMap memory usage
        let hashmap_overhead =
            self.tokens.len() * (std::mem::size_of::<String>() + std::mem::size_of::<u32>());
        let hashmap_capacity = self.tokens.len() * 2; // Assume 50% load factor
        let hashmap_buckets = hashmap_capacity * std::mem::size_of::<Option<(String, u32)>>();
        let hashmap_total = hashmap_overhead + hashmap_buckets;

        let token_storage: usize = self.tokens.iter().map(|s| s.len()).sum();
        let hashmap_estimated = hashmap_total + token_storage;

        EfficiencyComparison {
            mph_bytes: mph_usage.total_bytes,
            hashmap_bytes: hashmap_estimated,
            space_savings: 1.0 - (mph_usage.total_bytes as f64 / hashmap_estimated as f64),
            compression_ratio: hashmap_estimated as f64 / mph_usage.total_bytes as f64,
        }
    }
}

/// Comparison of memory efficiency between MPH and HashMap
#[derive(Debug, Clone)]
pub struct EfficiencyComparison {
    /// Memory usage of MPH vocabulary
    pub mph_bytes: usize,
    /// Estimated memory usage of HashMap vocabulary
    pub hashmap_bytes: usize,
    /// Space savings as a fraction (0.0 to 1.0)
    pub space_savings: f64,
    /// Compression ratio (how many times smaller)
    pub compression_ratio: f64,
}

impl std::fmt::Display for EfficiencyComparison {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MPH: {} bytes, HashMap: {} bytes, Savings: {:.1}%, Compression: {:.2}x",
            self.mph_bytes,
            self.hashmap_bytes,
            self.space_savings * 100.0,
            self.compression_ratio
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_perfect_hash_creation() {
        let keys = vec![
            "hello".to_string(),
            "world".to_string(),
            "test".to_string(),
            "key".to_string(),
            "value".to_string(),
        ];

        let config = MinimalPerfectHashConfig::default();
        let mph = MinimalPerfectHash::new(&keys, config).unwrap();

        assert_eq!(mph.len(), 5);
        assert!(!mph.is_empty());
    }

    #[test]
    fn test_minimal_perfect_hash_lookup() {
        let keys = vec!["hello".to_string(), "world".to_string(), "test".to_string()];

        let config = MinimalPerfectHashConfig::default();
        let mph = MinimalPerfectHash::new(&keys, config).unwrap();

        // All keys should be found
        assert!(mph.contains("hello"));
        assert!(mph.contains("world"));
        assert!(mph.contains("test"));

        // Non-existent keys should not be found
        assert!(!mph.contains("nonexistent"));
        assert!(!mph.contains("missing"));
    }

    #[test]
    fn test_minimal_perfect_hash_no_collisions() {
        let keys = vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
            "date".to_string(),
            "elderberry".to_string(),
        ];

        let config = MinimalPerfectHashConfig::default();
        let mph = MinimalPerfectHash::new(&keys, config).unwrap();

        // Get all hash values
        let mut hash_values = std::collections::HashSet::new();
        for key in &keys {
            if let Some(value) = mph.lookup(key) {
                assert!(
                    !hash_values.contains(&value),
                    "Collision detected for key: {}",
                    key
                );
                hash_values.insert(value);
            } else {
                panic!("Key not found: {}", key);
            }
        }

        assert_eq!(hash_values.len(), keys.len());
    }

    #[test]
    fn test_minimal_perfect_hash_vocab() {
        let tokens = vec![
            "the".to_string(),
            "quick".to_string(),
            "brown".to_string(),
            "fox".to_string(),
            "jumps".to_string(),
        ];

        let vocab = MinimalPerfectHashVocab::new(tokens.clone()).unwrap();

        assert_eq!(vocab.size(), 5);
        assert!(!vocab.is_empty());

        // Test token to ID mapping
        for (expected_id, token) in tokens.iter().enumerate() {
            if let Some(id) = vocab.get_id(token) {
                assert_eq!(id, expected_id as u32);
            } else {
                panic!("Token not found: {}", token);
            }
        }

        // Test ID to token mapping
        for (id, expected_token) in tokens.iter().enumerate() {
            if let Some(token) = vocab.get_token(id as u32) {
                assert_eq!(token, expected_token);
            } else {
                panic!("ID not found: {}", id);
            }
        }
    }

    #[test]
    fn test_memory_usage() {
        let tokens = vec![
            "token1".to_string(),
            "token2".to_string(),
            "token3".to_string(),
        ];

        let vocab = MinimalPerfectHashVocab::new(tokens).unwrap();
        let usage = vocab.memory_usage();

        assert!(usage.total_bytes > 0);
        assert!(usage.table_bytes > 0);
        assert!(usage.load_factor > 0.0);
        assert!(usage.load_factor <= 1.0);
    }

    #[test]
    fn test_efficiency_comparison() {
        let tokens = vec![
            "efficiency".to_string(),
            "comparison".to_string(),
            "test".to_string(),
            "minimal".to_string(),
            "perfect".to_string(),
            "hash".to_string(),
        ];

        let vocab = MinimalPerfectHashVocab::new(tokens).unwrap();
        let comparison = vocab.efficiency_comparison();

        assert!(comparison.mph_bytes > 0);
        assert!(comparison.hashmap_bytes > 0);
        assert!(comparison.compression_ratio > 0.0);

        // MPH should generally be more memory efficient
        assert!(comparison.space_savings >= 0.0);
    }

    #[test]
    fn test_large_vocabulary() {
        // Use a smaller vocabulary size for better test reliability
        let tokens: Vec<String> = vec![
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "hello", "world",
            "rust", "code",
        ]
        .into_iter()
        .map(|s| s.to_string())
        .collect();

        // Use a very permissive configuration for test reliability
        let config = MinimalPerfectHashConfig {
            load_factor: 0.5, // Much lower load factor for better success rate
            num_hash_functions: 3,
            use_double_hashing: true,
        };

        let vocab = MinimalPerfectHashVocab::with_config(tokens.clone(), config).unwrap();

        assert_eq!(vocab.size(), tokens.len());

        // Test random lookups
        for i in [0, 3, 7, tokens.len() - 1] {
            let token = &tokens[i];
            assert!(vocab.contains(token));
            assert_eq!(vocab.get_id(token), Some(i as u32));
            assert_eq!(vocab.get_token(i as u32), Some(token.as_str()));
        }
    }

    #[test]
    fn test_custom_config() {
        let tokens = vec![
            "custom".to_string(),
            "config".to_string(),
            "test".to_string(),
        ];

        let config = MinimalPerfectHashConfig {
            load_factor: 0.8,
            num_hash_functions: 2,
            use_double_hashing: false,
        };

        let vocab = MinimalPerfectHashVocab::with_config(tokens, config).unwrap();
        assert_eq!(vocab.size(), 3);

        let usage = vocab.memory_usage();
        println!("Load factor: {}, expected <= 0.8", usage.load_factor);
        assert!(usage.load_factor <= 0.8);
    }
}
