use anyhow::Result;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Strategy for merging vocabularies when there are conflicts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Keep tokens from the first vocabulary, ignore conflicting tokens from the second
    PreferFirst,
    /// Replace tokens from the first vocabulary with tokens from the second
    PreferSecond,
    /// Keep both tokens, adding a suffix to conflicting tokens from the second vocabulary
    KeepBothWithSuffix,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocab {
    token_to_id: HashMap<String, u32>,
    id_to_token: Vec<String>,
}

impl Vocab {
    pub fn new() -> Self {
        Self {
            token_to_id: HashMap::new(),
            id_to_token: Vec::new(),
        }
    }

    pub fn from_map(map: HashMap<String, u32>) -> Self {
        let mut id_to_token = vec![String::new(); map.len()];
        for (token, &id) in &map {
            if (id as usize) < id_to_token.len() {
                id_to_token[id as usize] = token.clone();
            }
        }

        Self {
            token_to_id: map,
            id_to_token,
        }
    }

    pub fn add_token(&mut self, token: String) -> u32 {
        if let Some(&id) = self.token_to_id.get(&token) {
            return id;
        }

        // Try to find the first available empty slot (reuse freed IDs)
        if let Some((index, _)) = self.id_to_token.iter().enumerate().find(|(_, s)| s.is_empty()) {
            let id = index as u32;
            self.token_to_id.insert(token.clone(), id);
            self.id_to_token[index] = token;
            return id;
        }

        // No empty slots, append at the end
        let id = self.id_to_token.len() as u32;
        self.token_to_id.insert(token.clone(), id);
        self.id_to_token.push(token);
        id
    }

    pub fn get_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    pub fn get_token(&self, id: u32) -> Option<String> {
        self.id_to_token.get(id as usize).cloned()
    }

    pub fn contains(&self, token: &str) -> bool {
        self.token_to_id.contains_key(token)
    }

    pub fn size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn get_vocab(&self) -> &HashMap<String, u32> {
        &self.token_to_id
    }

    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &u32)> {
        self.token_to_id.iter()
    }

    pub fn get(&self, token: &str) -> Option<&u32> {
        self.token_to_id.get(token)
    }

    pub fn get_token_to_id_map(&self) -> &HashMap<String, u32> {
        &self.token_to_id
    }

    /// Merge another vocabulary into this one
    pub fn merge_with(&mut self, other: &Vocab, strategy: MergeStrategy) -> Result<()> {
        match strategy {
            MergeStrategy::PreferFirst => {
                // Collect tokens from other vocabulary sorted by their original IDs
                let mut other_tokens: Vec<(String, u32)> =
                    other.token_to_id.iter().map(|(token, &id)| (token.clone(), id)).collect();
                other_tokens.sort_by_key(|(_, id)| *id);

                // Only add tokens that don't exist in the current vocabulary
                for (token, _) in other_tokens {
                    if !self.contains(&token) {
                        self.add_token(token);
                    }
                }
            },
            MergeStrategy::PreferSecond => {
                // Collect tokens from other vocabulary sorted by their original IDs
                let mut other_tokens: Vec<(String, u32)> =
                    other.token_to_id.iter().map(|(token, &id)| (token.clone(), id)).collect();
                other_tokens.sort_by_key(|(_, id)| *id);

                // Identify conflicting tokens and remove them
                let mut conflicting_tokens = std::collections::HashSet::new();
                for (token, _) in &other_tokens {
                    if self.contains(token) {
                        conflicting_tokens.insert(token.clone());
                        if let Some(old_id) = self.token_to_id.remove(token) {
                            if (old_id as usize) < self.id_to_token.len() {
                                self.id_to_token[old_id as usize] = String::new();
                            }
                        }
                    }
                }

                // Add non-conflicting tokens first (they can reuse freed IDs)
                for (token, _) in &other_tokens {
                    if !conflicting_tokens.contains(token) {
                        self.add_token(token.clone());
                    }
                }

                // Add conflicting tokens last (they get new IDs)
                for (token, _) in &other_tokens {
                    if conflicting_tokens.contains(token) {
                        // Force new ID by adding at the end
                        let id = self.id_to_token.len() as u32;
                        self.token_to_id.insert(token.clone(), id);
                        self.id_to_token.push(token.clone());
                    }
                }
            },
            MergeStrategy::KeepBothWithSuffix => {
                // Collect tokens from other vocabulary sorted by their original IDs
                let mut other_tokens: Vec<(String, u32)> =
                    other.token_to_id.iter().map(|(token, &id)| (token.clone(), id)).collect();
                other_tokens.sort_by_key(|(_, id)| *id);

                // Add tokens with suffix for conflicts
                for (token, _) in other_tokens {
                    if self.contains(&token) {
                        let mut suffix = 1;
                        let mut new_token = format!("{}_{}", token, suffix);
                        while self.contains(&new_token) {
                            suffix += 1;
                            new_token = format!("{}_{}", token, suffix);
                        }
                        self.add_token(new_token);
                    } else {
                        self.add_token(token.clone());
                    }
                }
            },
        }
        Ok(())
    }

    /// Merge multiple vocabularies using the specified strategy
    pub fn merge_multiple(vocabs: Vec<Vocab>, strategy: MergeStrategy) -> Result<Vocab> {
        if vocabs.is_empty() {
            return Ok(Vocab::new());
        }

        let mut result = vocabs[0].clone();
        for vocab in vocabs.iter().skip(1) {
            result.merge_with(vocab, strategy)?;
        }
        Ok(result)
    }

    /// Get all tokens in the vocabulary
    pub fn get_all_tokens(&self) -> Vec<String> {
        self.id_to_token.clone()
    }

    /// Create a vocabulary from a list of tokens
    pub fn from_tokens(tokens: Vec<String>) -> Self {
        let mut vocab = Vocab::new();
        for token in tokens {
            vocab.add_token(token);
        }
        vocab
    }

    /// Remove a token from the vocabulary
    pub fn remove_token(&mut self, token: &str) -> Option<u32> {
        if let Some(id) = self.token_to_id.remove(token) {
            if (id as usize) < self.id_to_token.len() {
                self.id_to_token[id as usize] = String::new();
            }
            Some(id)
        } else {
            None
        }
    }

    /// Compact the vocabulary by removing empty slots
    pub fn compact(&mut self) {
        let mut new_token_to_id = HashMap::new();
        let mut new_id_to_token = Vec::new();

        for token in self.id_to_token.iter() {
            if !token.is_empty() {
                let new_id = new_id_to_token.len() as u32;
                new_token_to_id.insert(token.clone(), new_id);
                new_id_to_token.push(token.clone());
            }
        }

        self.token_to_id = new_token_to_id;
        self.id_to_token = new_id_to_token;
    }
}

impl Default for Vocab {
    fn default() -> Self {
        Self::new()
    }
}

/// Lazy vocabulary that loads from file only when first accessed
pub struct LazyVocab {
    vocab: OnceCell<Arc<RwLock<Vocab>>>,
    loader: Box<dyn Fn() -> Result<Vocab> + Send + Sync>,
}

impl std::fmt::Debug for LazyVocab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyVocab")
            .field("vocab", &self.vocab)
            .field("loader", &"<closure>")
            .finish()
    }
}

impl LazyVocab {
    /// Create a new lazy vocabulary with a custom loader function
    pub fn new<F>(loader: F) -> Self
    where
        F: Fn() -> Result<Vocab> + Send + Sync + 'static,
    {
        Self {
            vocab: OnceCell::new(),
            loader: Box::new(loader),
        }
    }

    /// Create a lazy vocabulary that loads from a JSON file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Self {
        let path = path.as_ref().to_owned();
        Self::new(move || {
            let content = std::fs::read_to_string(&path)?;
            let vocab: Vocab = serde_json::from_str(&content)?;
            Ok(vocab)
        })
    }

    /// Create a lazy vocabulary from a tokenizer.json file
    pub fn from_tokenizer_json<P: AsRef<Path>>(path: P) -> Self {
        let path = path.as_ref().to_owned();
        Self::new(move || {
            let content = std::fs::read_to_string(&path)?;
            let value: serde_json::Value = serde_json::from_str(&content)?;

            if let Some(vocab_obj) = value.get("model").and_then(|m| m.get("vocab")) {
                let mut token_to_id = HashMap::new();
                if let Some(vocab_map) = vocab_obj.as_object() {
                    for (token, id_value) in vocab_map {
                        if let Some(id) = id_value.as_u64() {
                            token_to_id.insert(token.clone(), id as u32);
                        }
                    }
                }
                Ok(Vocab::from_map(token_to_id))
            } else {
                anyhow::bail!("No vocab found in tokenizer.json")
            }
        })
    }

    /// Get the vocabulary, loading it if necessary
    fn get_vocab(&self) -> Result<Arc<RwLock<Vocab>>> {
        if let Some(vocab) = self.vocab.get() {
            Ok(vocab.clone())
        } else {
            let vocab = (self.loader)()?;
            let vocab_arc = Arc::new(RwLock::new(vocab));
            self.vocab
                .set(vocab_arc.clone())
                .map_err(|_| anyhow::anyhow!("Failed to set vocab"))?;
            Ok(vocab_arc)
        }
    }

    /// Get token ID, loading vocabulary if necessary
    pub fn get_id(&self, token: &str) -> Result<Option<u32>> {
        let vocab = self.get_vocab()?;
        let vocab_guard = vocab.read().expect("lock should not be poisoned");
        Ok(vocab_guard.get_id(token))
    }

    /// Get token by ID, loading vocabulary if necessary
    pub fn get_token(&self, id: u32) -> Result<Option<String>> {
        let vocab = self.get_vocab()?;
        let vocab_guard = vocab.read().expect("lock should not be poisoned");
        Ok(vocab_guard.get_token(id))
    }

    /// Check if token exists, loading vocabulary if necessary
    pub fn contains(&self, token: &str) -> Result<bool> {
        let vocab = self.get_vocab()?;
        let vocab_guard = vocab.read().expect("lock should not be poisoned");
        Ok(vocab_guard.contains(token))
    }

    /// Get vocabulary size, loading vocabulary if necessary
    pub fn size(&self) -> Result<usize> {
        let vocab = self.get_vocab()?;
        let vocab_guard = vocab.read().expect("lock should not be poisoned");
        Ok(vocab_guard.size())
    }

    /// Check if vocabulary is loaded
    pub fn is_loaded(&self) -> bool {
        self.vocab.get().is_some()
    }

    /// Force load the vocabulary
    pub fn load(&self) -> Result<()> {
        self.get_vocab()?;
        Ok(())
    }

    /// Get a reference to the loaded vocabulary (if loaded)
    pub fn try_get_vocab(&self) -> Option<Arc<RwLock<Vocab>>> {
        self.vocab.get().cloned()
    }
}

/// Vocabulary that supports both in-memory and lazy loading
#[derive(Debug)]
pub enum FlexibleVocab {
    Immediate(Vocab),
    Lazy(LazyVocab),
}

impl FlexibleVocab {
    /// Create an immediate vocabulary
    pub fn immediate(vocab: Vocab) -> Self {
        Self::Immediate(vocab)
    }

    /// Create a lazy vocabulary
    pub fn lazy(lazy_vocab: LazyVocab) -> Self {
        Self::Lazy(lazy_vocab)
    }

    /// Create a lazy vocabulary from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Self {
        Self::Lazy(LazyVocab::from_file(path))
    }

    /// Get token ID
    pub fn get_id(&self, token: &str) -> Result<Option<u32>> {
        match self {
            Self::Immediate(vocab) => Ok(vocab.get_id(token)),
            Self::Lazy(lazy_vocab) => lazy_vocab.get_id(token),
        }
    }

    /// Get token by ID
    pub fn get_token(&self, id: u32) -> Result<Option<String>> {
        match self {
            Self::Immediate(vocab) => Ok(vocab.get_token(id)),
            Self::Lazy(lazy_vocab) => lazy_vocab.get_token(id),
        }
    }

    /// Check if token exists
    pub fn contains(&self, token: &str) -> Result<bool> {
        match self {
            Self::Immediate(vocab) => Ok(vocab.contains(token)),
            Self::Lazy(lazy_vocab) => lazy_vocab.contains(token),
        }
    }

    /// Get vocabulary size
    pub fn size(&self) -> Result<usize> {
        match self {
            Self::Immediate(vocab) => Ok(vocab.size()),
            Self::Lazy(lazy_vocab) => lazy_vocab.size(),
        }
    }

    /// Check if vocabulary is loaded (always true for immediate)
    pub fn is_loaded(&self) -> bool {
        match self {
            Self::Immediate(_) => true,
            Self::Lazy(lazy_vocab) => lazy_vocab.is_loaded(),
        }
    }
}

/// Configuration for dynamic vocabulary adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationConfig {
    /// Maximum vocabulary size
    pub max_vocab_size: usize,
    /// Minimum frequency threshold for keeping tokens
    pub min_frequency: usize,
    /// Time window for frequency calculation (in seconds)
    pub time_window: u64,
    /// Whether to enable automatic pruning
    pub auto_prune: bool,
    /// Pruning frequency (how often to prune, in number of additions)
    pub prune_frequency: usize,
    /// Factor for exponential decay of token frequencies
    pub decay_factor: f64,
}

impl Default for AdaptationConfig {
    fn default() -> Self {
        Self {
            max_vocab_size: 50000,
            min_frequency: 5,
            time_window: 3600, // 1 hour
            auto_prune: true,
            prune_frequency: 1000,
            decay_factor: 0.995,
        }
    }
}

/// Token statistics for dynamic adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenStats {
    pub frequency: usize,
    pub last_seen: u64,
    pub first_seen: u64,
    pub contexts: Vec<String>,
}

impl Default for TokenStats {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenStats {
    pub fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime is before UNIX_EPOCH")
            .as_secs();
        Self {
            frequency: 1,
            last_seen: now,
            first_seen: now,
            contexts: Vec::new(),
        }
    }

    pub fn update(&mut self, context: Option<String>) {
        self.frequency += 1;
        self.last_seen = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime is before UNIX_EPOCH")
            .as_secs();

        if let Some(ctx) = context {
            self.contexts.push(ctx);
            // Keep only the last 5 contexts to save memory
            if self.contexts.len() > 5 {
                self.contexts.remove(0);
            }
        }
    }

    pub fn apply_decay(&mut self, factor: f64) {
        self.frequency = (self.frequency as f64 * factor).max(1.0) as usize;
    }

    pub fn age(&self) -> u64 {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime is before UNIX_EPOCH")
            .as_secs();
        now - self.first_seen
    }
}

/// Dynamic vocabulary that can adapt based on usage patterns
pub struct DynamicVocab {
    vocab: Vocab,
    token_stats: HashMap<String, TokenStats>,
    config: AdaptationConfig,
    additions_since_prune: usize,
    adaptation_history: Vec<(u64, usize)>, // (timestamp, vocab_size)
}

impl DynamicVocab {
    pub fn new(config: AdaptationConfig) -> Self {
        Self {
            vocab: Vocab::new(),
            token_stats: HashMap::new(),
            config,
            additions_since_prune: 0,
            adaptation_history: Vec::new(),
        }
    }

    pub fn from_vocab(vocab: Vocab, config: AdaptationConfig) -> Self {
        let mut token_stats = HashMap::new();

        // Initialize stats for existing tokens
        for token in vocab.get_all_tokens() {
            if !token.is_empty() {
                token_stats.insert(token, TokenStats::new());
            }
        }

        Self {
            vocab,
            token_stats,
            config,
            additions_since_prune: 0,
            adaptation_history: Vec::new(),
        }
    }

    /// Add a new token or update existing token statistics
    pub fn add_or_update_token(&mut self, token: String, context: Option<String>) -> u32 {
        let id = if let Some(existing_id) = self.vocab.get_id(&token) {
            // Update existing token
            if let Some(stats) = self.token_stats.get_mut(&token) {
                stats.update(context);
            }
            existing_id
        } else {
            // Add new token
            let id = self.vocab.add_token(token.clone());
            let mut stats = TokenStats::new();
            if let Some(ctx) = context {
                stats.contexts.push(ctx);
            }
            self.token_stats.insert(token, stats);
            self.additions_since_prune += 1;
            id
        };

        // Auto-prune if needed
        if self.config.auto_prune && self.additions_since_prune >= self.config.prune_frequency {
            self.prune_vocabulary();
        }

        id
    }

    /// Prune the vocabulary based on frequency and age
    pub fn prune_vocabulary(&mut self) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("SystemTime is before UNIX_EPOCH")
            .as_secs();

        // Apply decay to all token frequencies
        for stats in self.token_stats.values_mut() {
            stats.apply_decay(self.config.decay_factor);
        }

        // Find tokens to remove
        let mut tokens_to_remove = Vec::new();
        for (token, stats) in &self.token_stats {
            // Remove tokens that are too old or too infrequent
            if stats.frequency < self.config.min_frequency
                || (now - stats.last_seen) > self.config.time_window
            {
                tokens_to_remove.push(token.clone());
            }
        }

        // Remove tokens
        for token in tokens_to_remove {
            self.vocab.remove_token(&token);
            self.token_stats.remove(&token);
        }

        // If still too large, remove least frequent tokens
        if self.vocab.size() > self.config.max_vocab_size {
            let mut sorted_tokens: Vec<_> = self
                .token_stats
                .iter()
                .map(|(token, stats)| (token.clone(), stats.frequency))
                .collect();
            sorted_tokens.sort_by_key(|(_, freq)| *freq);

            // Only remove tokens if we have more than max_vocab_size
            if sorted_tokens.len() > self.config.max_vocab_size {
                let tokens_to_remove = sorted_tokens.len() - self.config.max_vocab_size;
                for (token, _) in sorted_tokens.iter().take(tokens_to_remove) {
                    self.vocab.remove_token(token);
                    self.token_stats.remove(token);
                }
            }
        }

        // Compact the vocabulary
        self.vocab.compact();

        // Record adaptation history
        self.adaptation_history.push((now, self.vocab.size()));

        // Keep only the last 100 history entries
        if self.adaptation_history.len() > 100 {
            self.adaptation_history.remove(0);
        }

        self.additions_since_prune = 0;
    }

    /// Get token statistics
    pub fn get_token_stats(&self, token: &str) -> Option<&TokenStats> {
        self.token_stats.get(token)
    }

    /// Get all token statistics
    pub fn get_all_stats(&self) -> &HashMap<String, TokenStats> {
        &self.token_stats
    }

    /// Get vocabulary adaptation history
    pub fn get_adaptation_history(&self) -> &[(u64, usize)] {
        &self.adaptation_history
    }

    /// Get the most frequent tokens
    pub fn get_most_frequent_tokens(&self, limit: usize) -> Vec<(String, usize)> {
        let mut sorted_tokens: Vec<_> = self
            .token_stats
            .iter()
            .map(|(token, stats)| (token.clone(), stats.frequency))
            .collect();
        sorted_tokens.sort_by_key(|(_, freq)| std::cmp::Reverse(*freq));
        sorted_tokens.into_iter().take(limit).collect()
    }

    /// Get the least frequent tokens
    pub fn get_least_frequent_tokens(&self, limit: usize) -> Vec<(String, usize)> {
        let mut sorted_tokens: Vec<_> = self
            .token_stats
            .iter()
            .map(|(token, stats)| (token.clone(), stats.frequency))
            .collect();
        sorted_tokens.sort_by_key(|(_, freq)| *freq);
        sorted_tokens.into_iter().take(limit).collect()
    }

    /// Get tokens by age (oldest first)
    pub fn get_oldest_tokens(&self, limit: usize) -> Vec<(String, u64)> {
        let mut sorted_tokens: Vec<_> = self
            .token_stats
            .iter()
            .map(|(token, stats)| (token.clone(), stats.age()))
            .collect();
        sorted_tokens.sort_by_key(|(_, age)| std::cmp::Reverse(*age));
        sorted_tokens.into_iter().take(limit).collect()
    }

    /// Adapt vocabulary based on a batch of texts
    pub fn adapt_from_texts(&mut self, texts: &[String]) {
        for text in texts {
            let tokens = text.split_whitespace();
            for token in tokens {
                self.add_or_update_token(token.to_string(), Some(text.clone()));
            }
        }
    }

    /// Get adaptation statistics
    pub fn get_adaptation_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();

        stats.insert("vocab_size".to_string(), self.vocab.size() as f64);
        stats.insert(
            "additions_since_prune".to_string(),
            self.additions_since_prune as f64,
        );

        if !self.token_stats.is_empty() {
            let avg_frequency =
                self.token_stats.values().map(|stats| stats.frequency as f64).sum::<f64>()
                    / self.token_stats.len() as f64;
            stats.insert("avg_frequency".to_string(), avg_frequency);

            let max_frequency =
                self.token_stats.values().map(|stats| stats.frequency).max().unwrap_or(0) as f64;
            stats.insert("max_frequency".to_string(), max_frequency);

            let min_frequency =
                self.token_stats.values().map(|stats| stats.frequency).min().unwrap_or(0) as f64;
            stats.insert("min_frequency".to_string(), min_frequency);
        }

        stats
    }

    /// Export vocabulary with statistics
    pub fn export_with_stats(&self) -> HashMap<String, (u32, TokenStats)> {
        let mut result = HashMap::new();

        for (token, stats) in &self.token_stats {
            if let Some(id) = self.vocab.get_id(token) {
                result.insert(token.clone(), (id, stats.clone()));
            }
        }

        result
    }

    /// Set configuration
    pub fn set_config(&mut self, config: AdaptationConfig) {
        self.config = config;
    }

    /// Get configuration
    pub fn get_config(&self) -> &AdaptationConfig {
        &self.config
    }

    /// Get the underlying vocabulary
    pub fn get_vocab(&self) -> &Vocab {
        &self.vocab
    }

    /// Get mutable reference to the underlying vocabulary
    pub fn get_vocab_mut(&mut self) -> &mut Vocab {
        &mut self.vocab
    }
}

impl std::ops::Deref for DynamicVocab {
    type Target = Vocab;

    fn deref(&self) -> &Self::Target {
        &self.vocab
    }
}
