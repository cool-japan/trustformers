use anyhow::Result;
use once_cell::sync::OnceCell;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
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

/// Context-aware vocabulary adaptation
pub struct ContextualVocab {
    contexts: HashMap<String, DynamicVocab>,
    global_vocab: DynamicVocab,
    current_context: Option<String>,
    config: AdaptationConfig,
}

impl ContextualVocab {
    pub fn new(config: AdaptationConfig) -> Self {
        Self {
            contexts: HashMap::new(),
            global_vocab: DynamicVocab::new(config.clone()),
            current_context: None,
            config,
        }
    }

    /// Set the current context
    pub fn set_context(&mut self, context: String) {
        self.current_context = Some(context);
    }

    /// Clear the current context
    pub fn clear_context(&mut self) {
        self.current_context = None;
    }

    /// Add a token in the current context
    pub fn add_token(&mut self, token: String) -> u32 {
        // Add to global vocabulary
        let global_id = self
            .global_vocab
            .add_or_update_token(token.clone(), self.current_context.clone());

        // Add to context-specific vocabulary if context is set
        if let Some(ref context) = self.current_context {
            let context_vocab = self
                .contexts
                .entry(context.clone())
                .or_insert_with(|| DynamicVocab::new(self.config.clone()));
            context_vocab.add_or_update_token(token, Some(context.clone()));
        }

        global_id
    }

    /// Get token ID from current context or global vocabulary
    pub fn get_id(&self, token: &str) -> Option<u32> {
        if let Some(ref context) = self.current_context {
            if let Some(context_vocab) = self.contexts.get(context) {
                if let Some(id) = context_vocab.get_id(token) {
                    return Some(id);
                }
            }
        }

        self.global_vocab.get_id(token)
    }

    /// Get all contexts
    pub fn get_contexts(&self) -> Vec<String> {
        self.contexts.keys().cloned().collect()
    }

    /// Get vocabulary for a specific context
    pub fn get_context_vocab(&self, context: &str) -> Option<&DynamicVocab> {
        self.contexts.get(context)
    }

    /// Get global vocabulary
    pub fn get_global_vocab(&self) -> &DynamicVocab {
        &self.global_vocab
    }

    /// Merge contexts into global vocabulary
    pub fn merge_contexts_to_global(&mut self) {
        for (context, context_vocab) in &self.contexts {
            for token in context_vocab.get_all_stats().keys() {
                self.global_vocab.add_or_update_token(token.clone(), Some(context.clone()));
            }
        }
    }

    /// Get adaptation statistics for all contexts
    pub fn get_all_adaptation_stats(&self) -> HashMap<String, HashMap<String, f64>> {
        let mut all_stats = HashMap::new();

        all_stats.insert(
            "global".to_string(),
            self.global_vocab.get_adaptation_stats(),
        );

        for (context, vocab) in &self.contexts {
            all_stats.insert(context.clone(), vocab.get_adaptation_stats());
        }

        all_stats
    }
}

/// Multi-vocabulary system for handling multiple vocabularies with different configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VocabSelectionStrategy {
    /// Use the first vocabulary that contains the token
    FirstMatch,
    /// Use the vocabulary with the highest priority
    Priority,
    /// Use the vocabulary with the highest frequency for the token
    HighestFrequency,
    /// Use the vocabulary based on specified context
    Context,
    /// Use the vocabulary based on language detection
    Language,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabConfig {
    pub name: String,
    pub priority: u32,
    pub contexts: Vec<String>,
    pub languages: Vec<String>,
    pub adaptation_config: AdaptationConfig,
}

impl VocabConfig {
    pub fn new(name: String) -> Self {
        Self {
            name,
            priority: 1,
            contexts: Vec::new(),
            languages: Vec::new(),
            adaptation_config: AdaptationConfig::default(),
        }
    }

    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }

    pub fn with_contexts(mut self, contexts: Vec<String>) -> Self {
        self.contexts = contexts;
        self
    }

    pub fn with_languages(mut self, languages: Vec<String>) -> Self {
        self.languages = languages;
        self
    }

    pub fn with_adaptation_config(mut self, config: AdaptationConfig) -> Self {
        self.adaptation_config = config;
        self
    }
}

pub struct MultiVocabulary {
    vocabularies: HashMap<String, DynamicVocab>,
    configs: HashMap<String, VocabConfig>,
    selection_strategy: VocabSelectionStrategy,
    default_vocab: Option<String>,
    current_context: Option<String>,
    current_language: Option<String>,
    fallback_chain: Vec<String>,
}

impl MultiVocabulary {
    pub fn new(selection_strategy: VocabSelectionStrategy) -> Self {
        Self {
            vocabularies: HashMap::new(),
            configs: HashMap::new(),
            selection_strategy,
            default_vocab: None,
            current_context: None,
            current_language: None,
            fallback_chain: Vec::new(),
        }
    }

    /// Add a vocabulary with configuration
    pub fn add_vocabulary(&mut self, config: VocabConfig) -> Result<()> {
        let name = config.name.clone();
        let vocab = DynamicVocab::new(config.adaptation_config.clone());

        self.vocabularies.insert(name.clone(), vocab);
        self.configs.insert(name.clone(), config);

        // Set as default if it's the first vocabulary
        if self.default_vocab.is_none() {
            self.default_vocab = Some(name);
        }

        Ok(())
    }

    /// Add an existing vocabulary with configuration
    pub fn add_existing_vocabulary(
        &mut self,
        config: VocabConfig,
        vocab: DynamicVocab,
    ) -> Result<()> {
        let name = config.name.clone();

        self.vocabularies.insert(name.clone(), vocab);
        self.configs.insert(name.clone(), config);

        // Set as default if it's the first vocabulary
        if self.default_vocab.is_none() {
            self.default_vocab = Some(name);
        }

        Ok(())
    }

    /// Remove a vocabulary
    pub fn remove_vocabulary(&mut self, name: &str) -> Option<DynamicVocab> {
        self.configs.remove(name);
        let vocab = self.vocabularies.remove(name);

        // Update default if it was removed
        if self.default_vocab.as_ref() == Some(&name.to_string()) {
            self.default_vocab = self.vocabularies.keys().next().cloned();
        }

        vocab
    }

    /// Set current context
    pub fn set_context(&mut self, context: String) {
        self.current_context = Some(context);
    }

    /// Set current language
    pub fn set_language(&mut self, language: String) {
        self.current_language = Some(language);
    }

    /// Set fallback chain
    pub fn set_fallback_chain(&mut self, chain: Vec<String>) {
        self.fallback_chain = chain;
    }

    /// Set default vocabulary
    pub fn set_default_vocabulary(&mut self, name: String) -> Result<()> {
        if self.vocabularies.contains_key(&name) {
            self.default_vocab = Some(name);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Vocabulary '{}' not found", name))
        }
    }

    /// Select vocabulary based on strategy
    fn select_vocabulary(&self, token: &str) -> Option<&str> {
        match self.selection_strategy {
            VocabSelectionStrategy::FirstMatch => {
                for (name, vocab) in &self.vocabularies {
                    if vocab.contains(token) {
                        return Some(name);
                    }
                }
                self.default_vocab.as_deref()
            },
            VocabSelectionStrategy::Priority => {
                let mut best_vocab = None;
                let mut best_priority = 0;

                for (name, config) in &self.configs {
                    if let Some(vocab) = self.vocabularies.get(name) {
                        if vocab.contains(token) && config.priority > best_priority {
                            best_priority = config.priority;
                            best_vocab = Some(name.as_str());
                        }
                    }
                }

                best_vocab.or(self.default_vocab.as_deref())
            },
            VocabSelectionStrategy::HighestFrequency => {
                let mut best_vocab = None;
                let mut best_frequency = 0;

                for (name, vocab) in &self.vocabularies {
                    if let Some(stats) = vocab.get_token_stats(token) {
                        if stats.frequency > best_frequency {
                            best_frequency = stats.frequency;
                            best_vocab = Some(name.as_str());
                        }
                    }
                }

                best_vocab.or(self.default_vocab.as_deref())
            },
            VocabSelectionStrategy::Context => {
                if let Some(ref context) = self.current_context {
                    for (name, config) in &self.configs {
                        if config.contexts.contains(context) {
                            if let Some(vocab) = self.vocabularies.get(name) {
                                if vocab.contains(token) {
                                    return Some(name);
                                }
                            }
                        }
                    }
                }
                self.default_vocab.as_deref()
            },
            VocabSelectionStrategy::Language => {
                if let Some(ref language) = self.current_language {
                    for (name, config) in &self.configs {
                        if config.languages.contains(language) {
                            if let Some(vocab) = self.vocabularies.get(name) {
                                if vocab.contains(token) {
                                    return Some(name);
                                }
                            }
                        }
                    }
                }
                self.default_vocab.as_deref()
            },
        }
    }

    /// Get token ID from the appropriate vocabulary
    pub fn get_id(&self, token: &str) -> Option<u32> {
        // Try selected vocabulary first
        if let Some(vocab_name) = self.select_vocabulary(token) {
            if let Some(vocab) = self.vocabularies.get(vocab_name) {
                if let Some(id) = vocab.get_id(token) {
                    return Some(id);
                }
            }
        }

        // Try fallback chain
        for vocab_name in &self.fallback_chain {
            if let Some(vocab) = self.vocabularies.get(vocab_name) {
                if let Some(id) = vocab.get_id(token) {
                    return Some(id);
                }
            }
        }

        None
    }

    /// Add token to the appropriate vocabulary
    pub fn add_token(&mut self, token: String) -> Option<u32> {
        let vocab_name = self.select_vocabulary(&token)?.to_string();

        if let Some(vocab) = self.vocabularies.get_mut(&vocab_name) {
            let context = self.current_context.clone();
            Some(vocab.add_or_update_token(token, context))
        } else {
            None
        }
    }

    /// Add token to a specific vocabulary
    pub fn add_token_to_vocab(&mut self, token: String, vocab_name: &str) -> Option<u32> {
        if let Some(vocab) = self.vocabularies.get_mut(vocab_name) {
            let context = self.current_context.clone();
            Some(vocab.add_or_update_token(token, context))
        } else {
            None
        }
    }

    /// Get token from the appropriate vocabulary
    pub fn get_token(&self, id: u32) -> Option<String> {
        // Try all vocabularies to find the token
        for vocab in self.vocabularies.values() {
            if let Some(token) = vocab.get_token(id) {
                return Some(token);
            }
        }
        None
    }

    /// Get vocabulary names
    pub fn get_vocabulary_names(&self) -> Vec<String> {
        self.vocabularies.keys().cloned().collect()
    }

    /// Get vocabulary by name
    pub fn get_vocabulary(&self, name: &str) -> Option<&DynamicVocab> {
        self.vocabularies.get(name)
    }

    /// Get mutable vocabulary by name
    pub fn get_vocabulary_mut(&mut self, name: &str) -> Option<&mut DynamicVocab> {
        self.vocabularies.get_mut(name)
    }

    /// Get vocabulary configuration
    pub fn get_config(&self, name: &str) -> Option<&VocabConfig> {
        self.configs.get(name)
    }

    /// Update vocabulary configuration
    pub fn update_config(&mut self, name: &str, config: VocabConfig) -> Result<()> {
        if self.vocabularies.contains_key(name) {
            self.configs.insert(name.to_string(), config);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Vocabulary '{}' not found", name))
        }
    }

    /// Get total vocabulary size across all vocabularies
    pub fn total_vocab_size(&self) -> usize {
        self.vocabularies.values().map(|v| v.size()).sum()
    }

    /// Get unique tokens across all vocabularies
    pub fn unique_tokens(&self) -> HashSet<String> {
        let mut unique = HashSet::new();
        for vocab in self.vocabularies.values() {
            for token in vocab.get_all_tokens() {
                if !token.is_empty() {
                    unique.insert(token);
                }
            }
        }
        unique
    }

    /// Merge vocabularies based on strategy
    pub fn merge_vocabularies(
        &mut self,
        target_name: &str,
        source_names: &[&str],
        strategy: MergeStrategy,
    ) -> Result<()> {
        // First, collect all source vocabularies to avoid borrowing issues
        let mut source_vocabs = Vec::new();
        for source_name in source_names {
            let source_vocab = self
                .vocabularies
                .get(*source_name)
                .ok_or_else(|| anyhow::anyhow!("Source vocabulary '{}' not found", source_name))?;
            source_vocabs.push(source_vocab.get_vocab().clone());
        }

        // Now merge all source vocabularies into target
        let target_vocab = self
            .vocabularies
            .get_mut(target_name)
            .ok_or_else(|| anyhow::anyhow!("Target vocabulary '{}' not found", target_name))?;

        for source_regular_vocab in source_vocabs {
            target_vocab.get_vocab_mut().merge_with(&source_regular_vocab, strategy)?;
        }

        Ok(())
    }

    /// Get statistics for all vocabularies
    pub fn get_all_stats(&self) -> HashMap<String, HashMap<String, f64>> {
        let mut all_stats = HashMap::new();

        for (name, vocab) in &self.vocabularies {
            all_stats.insert(name.clone(), vocab.get_adaptation_stats());
        }

        all_stats
    }

    /// Export all vocabularies with their statistics
    pub fn export_all_vocabularies(&self) -> HashMap<String, HashMap<String, (u32, TokenStats)>> {
        let mut all_exports = HashMap::new();

        for (name, vocab) in &self.vocabularies {
            all_exports.insert(name.clone(), vocab.export_with_stats());
        }

        all_exports
    }

    /// Synchronize token between vocabularies
    pub fn sync_token(
        &mut self,
        token: &str,
        source_vocab: &str,
        target_vocabs: &[&str],
    ) -> Result<()> {
        // First, get the token stats from source vocabulary
        let token_stats = {
            let source = self
                .vocabularies
                .get(source_vocab)
                .ok_or_else(|| anyhow::anyhow!("Source vocabulary '{}' not found", source_vocab))?;

            source
                .get_token_stats(token)
                .ok_or_else(|| anyhow::anyhow!("Token '{}' not found in source vocabulary", token))?
                .clone()
        };

        // Now update target vocabularies
        for target_name in target_vocabs {
            if let Some(target_vocab) = self.vocabularies.get_mut(*target_name) {
                // Add token with same context information
                if token_stats.contexts.is_empty() {
                    // If no contexts, add token without context
                    target_vocab.add_or_update_token(token.to_string(), None);
                } else {
                    // Add token with each context
                    for context in &token_stats.contexts {
                        target_vocab.add_or_update_token(token.to_string(), Some(context.clone()));
                    }
                }
            }
        }

        Ok(())
    }

    /// Get current selection strategy
    pub fn get_selection_strategy(&self) -> VocabSelectionStrategy {
        self.selection_strategy
    }

    /// Set selection strategy
    pub fn set_selection_strategy(&mut self, strategy: VocabSelectionStrategy) {
        self.selection_strategy = strategy;
    }

    /// Get current context
    pub fn get_current_context(&self) -> Option<&String> {
        self.current_context.as_ref()
    }

    /// Get current language
    pub fn get_current_language(&self) -> Option<&String> {
        self.current_language.as_ref()
    }

    /// Clear current context
    pub fn clear_context(&mut self) {
        self.current_context = None;
    }

    /// Clear current language
    pub fn clear_language(&mut self) {
        self.current_language = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::NamedTempFile;

    #[test]
    fn test_lazy_vocab_basic() {
        let mut vocab = Vocab::new();
        vocab.add_token("hello".to_string());
        vocab.add_token("world".to_string());

        let lazy_vocab = LazyVocab::new(move || Ok(vocab.clone()));

        // Should not be loaded initially
        assert!(!lazy_vocab.is_loaded());

        // First access should load the vocab
        assert_eq!(lazy_vocab.get_id("hello").unwrap(), Some(0));
        assert!(lazy_vocab.is_loaded());

        // Subsequent accesses should use the loaded vocab
        assert_eq!(lazy_vocab.get_id("world").unwrap(), Some(1));
        assert_eq!(lazy_vocab.get_token(0).unwrap(), Some("hello".to_string()));
        assert_eq!(lazy_vocab.size().unwrap(), 2);
    }

    #[test]
    fn test_lazy_vocab_from_file() {
        let mut vocab = Vocab::new();
        vocab.add_token("test".to_string());
        vocab.add_token("vocab".to_string());

        // Create a temporary file
        let temp_file = NamedTempFile::new().unwrap();
        let json_content = serde_json::to_string(&vocab).unwrap();
        fs::write(temp_file.path(), json_content).unwrap();

        let lazy_vocab = LazyVocab::from_file(temp_file.path());

        assert!(!lazy_vocab.is_loaded());
        assert_eq!(lazy_vocab.get_id("test").unwrap(), Some(0));
        assert!(lazy_vocab.is_loaded());
        assert_eq!(lazy_vocab.get_token(1).unwrap(), Some("vocab".to_string()));
    }

    #[test]
    fn test_flexible_vocab_immediate() {
        let mut vocab = Vocab::new();
        vocab.add_token("immediate".to_string());

        let flex_vocab = FlexibleVocab::immediate(vocab);

        assert!(flex_vocab.is_loaded());
        assert_eq!(flex_vocab.get_id("immediate").unwrap(), Some(0));
        assert_eq!(flex_vocab.size().unwrap(), 1);
    }

    #[test]
    fn test_flexible_vocab_lazy() {
        let mut vocab = Vocab::new();
        vocab.add_token("lazy".to_string());

        let lazy_vocab = LazyVocab::new(move || Ok(vocab.clone()));
        let flex_vocab = FlexibleVocab::lazy(lazy_vocab);

        assert!(!flex_vocab.is_loaded());
        assert_eq!(flex_vocab.get_id("lazy").unwrap(), Some(0));
        assert!(flex_vocab.is_loaded());
    }

    #[test]
    fn test_lazy_vocab_error_handling() {
        let lazy_vocab = LazyVocab::new(|| anyhow::bail!("Test error"));

        assert!(lazy_vocab.get_id("test").is_err());
        assert!(!lazy_vocab.is_loaded());
    }

    #[test]
    fn test_lazy_vocab_tokenizer_json() {
        let tokenizer_json = r#"{
            "model": {
                "vocab": {
                    "hello": 0,
                    "world": 1,
                    "test": 2
                }
            }
        }"#;

        let temp_file = NamedTempFile::new().unwrap();
        fs::write(temp_file.path(), tokenizer_json).unwrap();

        let lazy_vocab = LazyVocab::from_tokenizer_json(temp_file.path());

        assert!(!lazy_vocab.is_loaded());
        assert_eq!(lazy_vocab.get_id("hello").unwrap(), Some(0));
        assert_eq!(lazy_vocab.get_id("world").unwrap(), Some(1));
        assert_eq!(lazy_vocab.get_id("test").unwrap(), Some(2));
        assert!(lazy_vocab.is_loaded());
        assert_eq!(lazy_vocab.size().unwrap(), 3);
    }

    #[test]
    fn test_vocab_merge_prefer_first() {
        let mut vocab1 = Vocab::new();
        vocab1.add_token("hello".to_string());
        vocab1.add_token("world".to_string());

        let mut vocab2 = Vocab::new();
        vocab2.add_token("world".to_string()); // Conflict
        vocab2.add_token("test".to_string());

        vocab1.merge_with(&vocab2, MergeStrategy::PreferFirst).unwrap();

        assert_eq!(vocab1.size(), 3);
        assert_eq!(vocab1.get_id("hello"), Some(0));
        assert_eq!(vocab1.get_id("world"), Some(1)); // Original ID preserved
        assert_eq!(vocab1.get_id("test"), Some(2));
    }

    #[test]
    fn test_vocab_merge_prefer_second() {
        let mut vocab1 = Vocab::new();
        vocab1.add_token("hello".to_string());
        vocab1.add_token("world".to_string());

        let mut vocab2 = Vocab::new();
        vocab2.add_token("world".to_string()); // Conflict
        vocab2.add_token("test".to_string());

        vocab1.merge_with(&vocab2, MergeStrategy::PreferSecond).unwrap();

        assert_eq!(vocab1.size(), 3);
        assert_eq!(vocab1.get_id("hello"), Some(0));
        assert_eq!(vocab1.get_id("world"), Some(2)); // New ID from second vocab
        assert_eq!(vocab1.get_id("test"), Some(1));
    }

    #[test]
    fn test_vocab_merge_keep_both_with_suffix() {
        let mut vocab1 = Vocab::new();
        vocab1.add_token("hello".to_string());
        vocab1.add_token("world".to_string());

        let mut vocab2 = Vocab::new();
        vocab2.add_token("world".to_string()); // Conflict
        vocab2.add_token("test".to_string());

        vocab1.merge_with(&vocab2, MergeStrategy::KeepBothWithSuffix).unwrap();

        assert_eq!(vocab1.size(), 4);
        assert_eq!(vocab1.get_id("hello"), Some(0));
        assert_eq!(vocab1.get_id("world"), Some(1)); // Original token
        assert_eq!(vocab1.get_id("world_1"), Some(2)); // Suffixed token
        assert_eq!(vocab1.get_id("test"), Some(3));
    }

    #[test]
    fn test_vocab_merge_multiple() {
        let vocab1 = Vocab::from_tokens(vec!["a".to_string(), "b".to_string()]);
        let vocab2 = Vocab::from_tokens(vec!["c".to_string(), "d".to_string()]);
        let vocab3 = Vocab::from_tokens(vec!["e".to_string(), "f".to_string()]);

        let merged =
            Vocab::merge_multiple(vec![vocab1, vocab2, vocab3], MergeStrategy::PreferFirst)
                .unwrap();

        assert_eq!(merged.size(), 6);
        assert_eq!(merged.get_id("a"), Some(0));
        assert_eq!(merged.get_id("b"), Some(1));
        assert_eq!(merged.get_id("c"), Some(2));
        assert_eq!(merged.get_id("d"), Some(3));
        assert_eq!(merged.get_id("e"), Some(4));
        assert_eq!(merged.get_id("f"), Some(5));
    }

    #[test]
    fn test_vocab_from_tokens() {
        let tokens = vec!["hello".to_string(), "world".to_string(), "test".to_string()];
        let vocab = Vocab::from_tokens(tokens);

        assert_eq!(vocab.size(), 3);
        assert_eq!(vocab.get_id("hello"), Some(0));
        assert_eq!(vocab.get_id("world"), Some(1));
        assert_eq!(vocab.get_id("test"), Some(2));
    }

    #[test]
    fn test_vocab_remove_token() {
        let mut vocab = Vocab::new();
        vocab.add_token("hello".to_string());
        vocab.add_token("world".to_string());
        vocab.add_token("test".to_string());

        assert_eq!(vocab.size(), 3);
        assert_eq!(vocab.remove_token("world"), Some(1));
        assert_eq!(vocab.size(), 3); // Size doesn't change until compact
        assert_eq!(vocab.get_id("world"), None);
        assert_eq!(vocab.get_token(1), Some(String::new())); // Empty slot
    }

    #[test]
    fn test_vocab_compact() {
        let mut vocab = Vocab::new();
        vocab.add_token("hello".to_string());
        vocab.add_token("world".to_string());
        vocab.add_token("test".to_string());

        vocab.remove_token("world");
        vocab.compact();

        assert_eq!(vocab.size(), 2);
        assert_eq!(vocab.get_id("hello"), Some(0));
        assert_eq!(vocab.get_id("test"), Some(1));
        assert_eq!(vocab.get_token(0), Some("hello".to_string()));
        assert_eq!(vocab.get_token(1), Some("test".to_string()));
    }

    #[test]
    fn test_vocab_get_all_tokens() {
        let mut vocab = Vocab::new();
        vocab.add_token("hello".to_string());
        vocab.add_token("world".to_string());

        let tokens = vocab.get_all_tokens();
        assert_eq!(tokens, vec!["hello".to_string(), "world".to_string()]);
    }

    #[test]
    fn test_merge_strategy_serialization() {
        let strategy = MergeStrategy::PreferFirst;
        let serialized = serde_json::to_string(&strategy).unwrap();
        let deserialized: MergeStrategy = serde_json::from_str(&serialized).unwrap();
        assert_eq!(strategy, deserialized);
    }

    #[test]
    fn test_adaptation_config() {
        let config = AdaptationConfig::default();
        assert_eq!(config.max_vocab_size, 50000);
        assert_eq!(config.min_frequency, 5);
        assert_eq!(config.time_window, 3600);
        assert!(config.auto_prune);
        assert_eq!(config.prune_frequency, 1000);
        assert_eq!(config.decay_factor, 0.995);
    }

    #[test]
    fn test_token_stats() {
        let mut stats = TokenStats::new();
        assert_eq!(stats.frequency, 1);
        assert!(stats.age() < 10); // Age should be very small for newly created stats

        stats.update(Some("context1".to_string()));
        assert_eq!(stats.frequency, 2);
        assert_eq!(stats.contexts.len(), 1);
        assert_eq!(stats.contexts[0], "context1");

        stats.apply_decay(0.5);
        assert_eq!(stats.frequency, 1); // max(2 * 0.5, 1.0) = 1
    }

    #[test]
    fn test_dynamic_vocab_basic() {
        let config = AdaptationConfig::default();
        let mut vocab = DynamicVocab::new(config);

        let id1 = vocab.add_or_update_token("hello".to_string(), None);
        let id2 = vocab.add_or_update_token("world".to_string(), None);
        let id3 = vocab.add_or_update_token("hello".to_string(), None); // Same token

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id1, id3); // Should return same ID

        assert_eq!(vocab.size(), 2);
        assert_eq!(vocab.get_token_stats("hello").unwrap().frequency, 2);
        assert_eq!(vocab.get_token_stats("world").unwrap().frequency, 1);
    }

    #[test]
    fn test_dynamic_vocab_adaptation() {
        let config = AdaptationConfig {
            max_vocab_size: 3,
            min_frequency: 2,
            auto_prune: false,
            decay_factor: 1.0, // No decay for test
            ..Default::default()
        };
        let mut vocab = DynamicVocab::new(config);

        // Add tokens
        vocab.add_or_update_token("hello".to_string(), None);
        vocab.add_or_update_token("world".to_string(), None);
        vocab.add_or_update_token("test".to_string(), None);
        vocab.add_or_update_token("hello".to_string(), None); // Increase frequency
        vocab.add_or_update_token("foo".to_string(), None);

        assert_eq!(vocab.size(), 4);

        // Manual pruning
        vocab.prune_vocabulary();

        // Should keep tokens with frequency >= 2
        assert_eq!(vocab.size(), 1); // Only "hello" has frequency >= 2
        assert!(vocab.get_id("hello").is_some());
        assert!(vocab.get_id("world").is_none());
        assert!(vocab.get_id("test").is_none());
        assert!(vocab.get_id("foo").is_none());
    }

    #[test]
    fn test_dynamic_vocab_from_texts() {
        let config = AdaptationConfig::default();
        let mut vocab = DynamicVocab::new(config);

        let texts = vec![
            "hello world".to_string(),
            "hello test".to_string(),
            "world test".to_string(),
        ];

        vocab.adapt_from_texts(&texts);

        assert_eq!(vocab.size(), 3);
        assert_eq!(vocab.get_token_stats("hello").unwrap().frequency, 2);
        assert_eq!(vocab.get_token_stats("world").unwrap().frequency, 2);
        assert_eq!(vocab.get_token_stats("test").unwrap().frequency, 2);
    }

    #[test]
    fn test_dynamic_vocab_statistics() {
        let config = AdaptationConfig::default();
        let mut vocab = DynamicVocab::new(config);

        vocab.add_or_update_token("hello".to_string(), None);
        vocab.add_or_update_token("world".to_string(), None);
        vocab.add_or_update_token("hello".to_string(), None); // Increase frequency

        let stats = vocab.get_adaptation_stats();
        assert_eq!(stats.get("vocab_size").unwrap(), &2.0);
        assert_eq!(stats.get("avg_frequency").unwrap(), &1.5); // (2 + 1) / 2
        assert_eq!(stats.get("max_frequency").unwrap(), &2.0);
        assert_eq!(stats.get("min_frequency").unwrap(), &1.0);

        let most_frequent = vocab.get_most_frequent_tokens(1);
        assert_eq!(most_frequent[0].0, "hello");
        assert_eq!(most_frequent[0].1, 2);

        let least_frequent = vocab.get_least_frequent_tokens(1);
        assert_eq!(least_frequent[0].0, "world");
        assert_eq!(least_frequent[0].1, 1);
    }

    #[test]
    fn test_contextual_vocab() {
        let config = AdaptationConfig::default();
        let mut vocab = ContextualVocab::new(config);

        // Add tokens in different contexts
        vocab.set_context("context1".to_string());
        vocab.add_token("hello".to_string());
        vocab.add_token("world".to_string());

        vocab.set_context("context2".to_string());
        vocab.add_token("hello".to_string()); // Same token, different context
        vocab.add_token("test".to_string());

        vocab.clear_context();
        vocab.add_token("global".to_string());

        assert_eq!(vocab.get_contexts().len(), 2);
        assert!(vocab.get_contexts().contains(&"context1".to_string()));
        assert!(vocab.get_contexts().contains(&"context2".to_string()));

        let global_vocab = vocab.get_global_vocab();
        assert_eq!(global_vocab.size(), 4); // hello, world, test, global

        let context1_vocab = vocab.get_context_vocab("context1").unwrap();
        assert_eq!(context1_vocab.size(), 2); // hello, world

        let context2_vocab = vocab.get_context_vocab("context2").unwrap();
        assert_eq!(context2_vocab.size(), 2); // hello, test
    }

    #[test]
    fn test_contextual_vocab_lookup() {
        let config = AdaptationConfig::default();
        let mut vocab = ContextualVocab::new(config);

        vocab.set_context("context1".to_string());
        vocab.add_token("hello".to_string());

        // Should find token in current context
        assert!(vocab.get_id("hello").is_some());

        vocab.set_context("context2".to_string());
        // Should find token in global vocab even if not in current context
        assert!(vocab.get_id("hello").is_some());

        // Should not find non-existent token
        assert!(vocab.get_id("nonexistent").is_none());
    }

    #[test]
    fn test_dynamic_vocab_export() {
        let config = AdaptationConfig::default();
        let mut vocab = DynamicVocab::new(config);

        vocab.add_or_update_token("hello".to_string(), Some("context1".to_string()));
        vocab.add_or_update_token("world".to_string(), Some("context2".to_string()));

        let export = vocab.export_with_stats();
        assert_eq!(export.len(), 2);

        let (hello_id, hello_stats) = export.get("hello").unwrap();
        assert_eq!(*hello_id, 0);
        assert_eq!(hello_stats.frequency, 1);
        assert_eq!(hello_stats.contexts.len(), 1);
        assert_eq!(hello_stats.contexts[0], "context1");

        let (world_id, world_stats) = export.get("world").unwrap();
        assert_eq!(*world_id, 1);
        assert_eq!(world_stats.frequency, 1);
        assert_eq!(world_stats.contexts.len(), 1);
        assert_eq!(world_stats.contexts[0], "context2");
    }

    #[test]
    fn test_adaptation_config_serialization() {
        let config = AdaptationConfig {
            max_vocab_size: 10000,
            min_frequency: 3,
            time_window: 7200,
            auto_prune: false,
            prune_frequency: 500,
            decay_factor: 0.99,
        };

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: AdaptationConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.max_vocab_size, deserialized.max_vocab_size);
        assert_eq!(config.min_frequency, deserialized.min_frequency);
        assert_eq!(config.time_window, deserialized.time_window);
        assert_eq!(config.auto_prune, deserialized.auto_prune);
        assert_eq!(config.prune_frequency, deserialized.prune_frequency);
        assert_eq!(config.decay_factor, deserialized.decay_factor);
    }

    #[test]
    fn test_vocab_config() {
        let config = VocabConfig::new("test_vocab".to_string())
            .with_priority(5)
            .with_contexts(vec!["context1".to_string(), "context2".to_string()])
            .with_languages(vec!["en".to_string(), "es".to_string()]);

        assert_eq!(config.name, "test_vocab");
        assert_eq!(config.priority, 5);
        assert_eq!(config.contexts, vec!["context1", "context2"]);
        assert_eq!(config.languages, vec!["en", "es"]);
    }

    #[test]
    fn test_multi_vocabulary_basic() {
        let mut multi_vocab = MultiVocabulary::new(VocabSelectionStrategy::FirstMatch);

        // Add vocabularies
        let config1 = VocabConfig::new("vocab1".to_string()).with_priority(1);
        let config2 = VocabConfig::new("vocab2".to_string()).with_priority(2);

        multi_vocab.add_vocabulary(config1).unwrap();
        multi_vocab.add_vocabulary(config2).unwrap();

        assert_eq!(multi_vocab.get_vocabulary_names().len(), 2);
        assert!(multi_vocab.get_vocabulary_names().contains(&"vocab1".to_string()));
        assert!(multi_vocab.get_vocabulary_names().contains(&"vocab2".to_string()));
    }

    #[test]
    fn test_multi_vocabulary_token_operations() {
        let mut multi_vocab = MultiVocabulary::new(VocabSelectionStrategy::FirstMatch);

        let config1 = VocabConfig::new("vocab1".to_string());
        multi_vocab.add_vocabulary(config1).unwrap();

        // Add tokens
        let id1 = multi_vocab.add_token("hello".to_string()).unwrap();
        let id2 = multi_vocab.add_token("world".to_string()).unwrap();

        // Test retrieval
        assert_eq!(multi_vocab.get_id("hello"), Some(id1));
        assert_eq!(multi_vocab.get_id("world"), Some(id2));
        assert_eq!(multi_vocab.get_token(id1), Some("hello".to_string()));
        assert_eq!(multi_vocab.get_token(id2), Some("world".to_string()));

        // Test non-existent token
        assert_eq!(multi_vocab.get_id("nonexistent"), None);
    }

    #[test]
    fn test_multi_vocabulary_priority_strategy() {
        let mut multi_vocab = MultiVocabulary::new(VocabSelectionStrategy::Priority);

        let config1 = VocabConfig::new("vocab1".to_string()).with_priority(1);
        let config2 = VocabConfig::new("vocab2".to_string()).with_priority(2);

        multi_vocab.add_vocabulary(config1).unwrap();
        multi_vocab.add_vocabulary(config2).unwrap();

        // Add same token to both vocabularies
        multi_vocab.add_token_to_vocab("hello".to_string(), "vocab1").unwrap();
        multi_vocab.add_token_to_vocab("hello".to_string(), "vocab2").unwrap();

        // Should select vocab2 due to higher priority
        let selected_vocab = multi_vocab.select_vocabulary("hello").unwrap();
        assert_eq!(selected_vocab, "vocab2");
    }

    #[test]
    fn test_multi_vocabulary_context_strategy() {
        let mut multi_vocab = MultiVocabulary::new(VocabSelectionStrategy::Context);

        let config1 =
            VocabConfig::new("vocab1".to_string()).with_contexts(vec!["context1".to_string()]);
        let config2 =
            VocabConfig::new("vocab2".to_string()).with_contexts(vec!["context2".to_string()]);

        multi_vocab.add_vocabulary(config1).unwrap();
        multi_vocab.add_vocabulary(config2).unwrap();

        // Add token to both vocabularies
        multi_vocab.add_token_to_vocab("hello".to_string(), "vocab1").unwrap();
        multi_vocab.add_token_to_vocab("hello".to_string(), "vocab2").unwrap();

        // Set context and test selection
        multi_vocab.set_context("context1".to_string());
        let selected_vocab = multi_vocab.select_vocabulary("hello").unwrap();
        assert_eq!(selected_vocab, "vocab1");

        multi_vocab.set_context("context2".to_string());
        let selected_vocab = multi_vocab.select_vocabulary("hello").unwrap();
        assert_eq!(selected_vocab, "vocab2");
    }

    #[test]
    fn test_multi_vocabulary_language_strategy() {
        let mut multi_vocab = MultiVocabulary::new(VocabSelectionStrategy::Language);

        let config1 =
            VocabConfig::new("english".to_string()).with_languages(vec!["en".to_string()]);
        let config2 =
            VocabConfig::new("spanish".to_string()).with_languages(vec!["es".to_string()]);

        multi_vocab.add_vocabulary(config1).unwrap();
        multi_vocab.add_vocabulary(config2).unwrap();

        // Add token to both vocabularies
        multi_vocab.add_token_to_vocab("hello".to_string(), "english").unwrap();
        multi_vocab.add_token_to_vocab("hola".to_string(), "spanish").unwrap();

        // Set language and test selection
        multi_vocab.set_language("en".to_string());
        let selected_vocab = multi_vocab.select_vocabulary("hello").unwrap();
        assert_eq!(selected_vocab, "english");

        multi_vocab.set_language("es".to_string());
        let selected_vocab = multi_vocab.select_vocabulary("hola").unwrap();
        assert_eq!(selected_vocab, "spanish");
    }

    #[test]
    fn test_multi_vocabulary_fallback_chain() {
        let mut multi_vocab = MultiVocabulary::new(VocabSelectionStrategy::FirstMatch);

        let config1 = VocabConfig::new("vocab1".to_string());
        let config2 = VocabConfig::new("vocab2".to_string());

        multi_vocab.add_vocabulary(config1).unwrap();
        multi_vocab.add_vocabulary(config2).unwrap();

        // Add token only to vocab2
        multi_vocab.add_token_to_vocab("hello".to_string(), "vocab2").unwrap();

        // Set fallback chain
        multi_vocab.set_fallback_chain(vec!["vocab1".to_string(), "vocab2".to_string()]);

        // Should find token in vocab2 via fallback
        assert!(multi_vocab.get_id("hello").is_some());
    }

    #[test]
    fn test_multi_vocabulary_merge() {
        let mut multi_vocab = MultiVocabulary::new(VocabSelectionStrategy::FirstMatch);

        let config1 = VocabConfig::new("vocab1".to_string());
        let config2 = VocabConfig::new("vocab2".to_string());

        multi_vocab.add_vocabulary(config1).unwrap();
        multi_vocab.add_vocabulary(config2).unwrap();

        // Add tokens to different vocabularies
        multi_vocab.add_token_to_vocab("hello".to_string(), "vocab1").unwrap();
        multi_vocab.add_token_to_vocab("world".to_string(), "vocab2").unwrap();

        // Get sizes before merge
        let vocab1_size_before = multi_vocab.get_vocabulary("vocab1").unwrap().size();
        let _vocab2_size_before = multi_vocab.get_vocabulary("vocab2").unwrap().size();

        // Merge vocab2 into vocab1
        multi_vocab
            .merge_vocabularies("vocab1", &["vocab2"], MergeStrategy::PreferFirst)
            .unwrap();

        // Check that vocab1 now contains both tokens
        let vocab1_size_after = multi_vocab.get_vocabulary("vocab1").unwrap().size();
        assert!(vocab1_size_after > vocab1_size_before);

        // Both tokens should be accessible from vocab1
        assert!(multi_vocab.get_vocabulary("vocab1").unwrap().contains("hello"));
        assert!(multi_vocab.get_vocabulary("vocab1").unwrap().contains("world"));
    }

    #[test]
    fn test_multi_vocabulary_sync_token() {
        let mut multi_vocab = MultiVocabulary::new(VocabSelectionStrategy::FirstMatch);

        let config1 = VocabConfig::new("vocab1".to_string());
        let config2 = VocabConfig::new("vocab2".to_string());

        multi_vocab.add_vocabulary(config1).unwrap();
        multi_vocab.add_vocabulary(config2).unwrap();

        // Add token to vocab1
        multi_vocab.add_token_to_vocab("hello".to_string(), "vocab1").unwrap();

        // Sync token to vocab2
        multi_vocab.sync_token("hello", "vocab1", &["vocab2"]).unwrap();

        // Both vocabularies should contain the token
        assert!(multi_vocab.get_vocabulary("vocab1").unwrap().contains("hello"));
        assert!(multi_vocab.get_vocabulary("vocab2").unwrap().contains("hello"));
    }

    #[test]
    fn test_multi_vocabulary_statistics() {
        let mut multi_vocab = MultiVocabulary::new(VocabSelectionStrategy::FirstMatch);

        let config1 = VocabConfig::new("vocab1".to_string());
        let config2 = VocabConfig::new("vocab2".to_string());

        multi_vocab.add_vocabulary(config1).unwrap();
        multi_vocab.add_vocabulary(config2).unwrap();

        // Add tokens
        multi_vocab.add_token_to_vocab("hello".to_string(), "vocab1").unwrap();
        multi_vocab.add_token_to_vocab("world".to_string(), "vocab1").unwrap();
        multi_vocab.add_token_to_vocab("test".to_string(), "vocab2").unwrap();

        // Test statistics
        assert_eq!(multi_vocab.total_vocab_size(), 3);

        let unique_tokens = multi_vocab.unique_tokens();
        assert_eq!(unique_tokens.len(), 3);
        assert!(unique_tokens.contains("hello"));
        assert!(unique_tokens.contains("world"));
        assert!(unique_tokens.contains("test"));

        let all_stats = multi_vocab.get_all_stats();
        assert_eq!(all_stats.len(), 2);
        assert!(all_stats.contains_key("vocab1"));
        assert!(all_stats.contains_key("vocab2"));
    }

    #[test]
    fn test_multi_vocabulary_remove() {
        let mut multi_vocab = MultiVocabulary::new(VocabSelectionStrategy::FirstMatch);

        let config1 = VocabConfig::new("vocab1".to_string());
        let config2 = VocabConfig::new("vocab2".to_string());

        multi_vocab.add_vocabulary(config1).unwrap();
        multi_vocab.add_vocabulary(config2).unwrap();

        assert_eq!(multi_vocab.get_vocabulary_names().len(), 2);

        // Remove vocabulary
        let removed_vocab = multi_vocab.remove_vocabulary("vocab1");
        assert!(removed_vocab.is_some());
        assert_eq!(multi_vocab.get_vocabulary_names().len(), 1);
        assert!(!multi_vocab.get_vocabulary_names().contains(&"vocab1".to_string()));
    }

    #[test]
    fn test_vocabulary_selection_strategy_serialization() {
        let strategies = [
            VocabSelectionStrategy::FirstMatch,
            VocabSelectionStrategy::Priority,
            VocabSelectionStrategy::HighestFrequency,
            VocabSelectionStrategy::Context,
            VocabSelectionStrategy::Language,
        ];

        for strategy in strategies {
            let serialized = serde_json::to_string(&strategy).unwrap();
            let deserialized: VocabSelectionStrategy = serde_json::from_str(&serialized).unwrap();
            assert_eq!(strategy, deserialized);
        }
    }

    #[test]
    fn test_vocab_config_serialization() {
        let config = VocabConfig::new("test".to_string())
            .with_priority(5)
            .with_contexts(vec!["ctx1".to_string(), "ctx2".to_string()])
            .with_languages(vec!["en".to_string(), "es".to_string()]);

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: VocabConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.name, deserialized.name);
        assert_eq!(config.priority, deserialized.priority);
        assert_eq!(config.contexts, deserialized.contexts);
        assert_eq!(config.languages, deserialized.languages);
    }
}
