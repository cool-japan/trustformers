use std::sync::Arc;

use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;

/// Key-Value cache for efficient generation.
///
/// The cache stores one key tensor and one value tensor **per transformer
/// layer**; `seq_len` tracks how many token positions those tensors currently
/// cover.  Layer registration and sequence growth are therefore separate
/// operations - see [`KVCache::push_layer`] and [`KVCache::advance`].
#[derive(Debug, Clone, Default)]
pub struct KVCache {
    pub keys: Vec<Tensor>,
    pub values: Vec<Tensor>,
    pub seq_len: usize,
}

impl KVCache {
    pub fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
            seq_len: 0,
        }
    }

    /// Register the key/value tensors of one more layer.
    ///
    /// This does **not** change `seq_len`: adding a layer does not add a token
    /// position.  Use [`KVCache::advance`] once per decoded token.
    pub fn push_layer(&mut self, key: Tensor, value: Tensor) -> Result<()> {
        if self.keys.len() != self.values.len() {
            return Err(TrustformersError::invalid_input(
                "Key-value cache size mismatch".to_string(),
            ));
        }

        self.keys.push(key);
        self.values.push(value);
        Ok(())
    }

    /// Replace the cached key/value tensors of an existing layer.
    pub fn set_layer(&mut self, layer_idx: usize, key: Tensor, value: Tensor) -> Result<()> {
        if layer_idx >= self.keys.len() || layer_idx >= self.values.len() {
            return Err(TrustformersError::invalid_input(format!(
                "layer {layer_idx} is not present in a cache with {} layers",
                self.keys.len()
            )));
        }
        self.keys[layer_idx] = key;
        self.values[layer_idx] = value;
        Ok(())
    }

    /// Record that `tokens` more positions are now covered by the cache.
    pub fn advance(&mut self, tokens: usize) {
        self.seq_len += tokens;
    }

    /// Number of layers currently held.
    pub fn num_layers(&self) -> usize {
        self.keys.len()
    }

    pub fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
        self.seq_len = 0;
    }

    pub fn get_layer(&self, layer_idx: usize) -> Option<(&Tensor, &Tensor)> {
        if layer_idx < self.keys.len() {
            Some((&self.keys[layer_idx], &self.values[layer_idx]))
        } else {
            None
        }
    }
}

/// Beam for beam search.
///
/// The key/value cache is held behind an [`Arc`] so that expanding a beam is
/// `O(1)`: [`Beam::extend`] shares the parent's cache instead of deep-copying
/// every cached key and value tensor of every layer.  A beam that needs to
/// mutate its own cache calls [`Beam::cache_mut`], which clones lazily
/// (copy-on-write) and only when the cache is actually shared.
#[derive(Debug, Clone)]
pub struct Beam {
    pub tokens: Vec<usize>,
    pub score: f32,
    pub finished: bool,
    pub cache: Option<Arc<KVCache>>,
}

impl Beam {
    pub fn new(tokens: Vec<usize>, score: f32) -> Self {
        Self {
            tokens,
            score,
            finished: false,
            cache: None,
        }
    }

    /// Attach a cache to this beam.
    pub fn with_cache(mut self, cache: KVCache) -> Self {
        self.cache = Some(Arc::new(cache));
        self
    }

    /// Extend the beam with one more token.
    ///
    /// The returned beam shares the parent's cache; no tensor is copied.
    pub fn extend(&self, token: usize, score: f32) -> Self {
        let mut new_tokens = Vec::with_capacity(self.tokens.len() + 1);
        new_tokens.extend_from_slice(&self.tokens);
        new_tokens.push(token);

        Self {
            tokens: new_tokens,
            score: self.score + score,
            finished: false,
            cache: self.cache.clone(),
        }
    }

    /// Mutable access to this beam's cache, cloning it only if another beam
    /// still refers to the same allocation.
    pub fn cache_mut(&mut self) -> Option<&mut KVCache> {
        self.cache.as_mut().map(Arc::make_mut)
    }

    pub fn finalize(&mut self) {
        self.finished = true;
    }

    pub fn get_normalized_score(&self) -> f32 {
        if self.tokens.is_empty() {
            0.0
        } else {
            self.score / self.tokens.len() as f32
        }
    }

    /// Length-normalised score with an explicit length penalty
    /// (`score / len^length_penalty`), matching the beam-search ranking rule.
    pub fn length_normalized_score(&self, length_penalty: f32) -> f32 {
        let length = self.tokens.len().max(1) as f32;
        self.score / length.powf(length_penalty)
    }
}
