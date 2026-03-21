use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;

/// Key-Value cache for efficient generation
#[derive(Debug, Clone)]
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

    pub fn append(&mut self, key: Tensor, value: Tensor) -> Result<()> {
        if self.keys.len() != self.values.len() {
            return Err(TrustformersError::invalid_input(
                "Key-value cache size mismatch".to_string(),
            ));
        }

        self.keys.push(key);
        self.values.push(value);
        self.seq_len += 1;
        Ok(())
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

impl Default for KVCache {
    fn default() -> Self {
        Self::new()
    }
}

/// Beam for beam search
#[derive(Debug, Clone)]
pub struct Beam {
    pub tokens: Vec<usize>,
    pub score: f32,
    pub finished: bool,
    pub cache: Option<KVCache>,
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

    pub fn extend(&self, token: usize, score: f32) -> Self {
        let mut new_tokens = self.tokens.clone();
        new_tokens.push(token);

        Self {
            tokens: new_tokens,
            score: self.score + score,
            finished: false,
            cache: self.cache.clone(),
        }
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
}
