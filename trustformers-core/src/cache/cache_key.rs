use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::fmt;
use std::hash::{Hash, Hasher};

/// A cache key for inference requests
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    /// Hash of the input text or tokens
    pub input_hash: u64,
    /// Model identifier
    pub model_id: String,
    /// Task type (e.g., "text-classification", "text-generation")
    pub task: String,
    /// Additional parameters that affect the output
    pub params_hash: u64,
}

impl fmt::Display for CacheKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}-{}-{}-{}",
            self.model_id, self.task, self.input_hash, self.params_hash
        )
    }
}

impl CacheKey {
    pub fn new(input_hash: u64, model_id: String, task: String, params_hash: u64) -> Self {
        Self {
            input_hash,
            model_id,
            task,
            params_hash,
        }
    }
}

/// Builder for creating cache keys with proper hashing
pub struct CacheKeyBuilder {
    model_id: String,
    task: String,
    input_hasher: DefaultHasher,
    params_hasher: DefaultHasher,
}

impl CacheKeyBuilder {
    pub fn new(model_id: impl Into<String>, task: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            task: task.into(),
            input_hasher: DefaultHasher::new(),
            params_hasher: DefaultHasher::new(),
        }
    }

    /// Add text input to the key
    pub fn with_text(mut self, text: &str) -> Self {
        text.hash(&mut self.input_hasher);
        self
    }

    /// Add tokenized input to the key
    pub fn with_tokens(mut self, tokens: &[u32]) -> Self {
        tokens.hash(&mut self.input_hasher);
        self
    }

    /// Add a parameter that affects output
    pub fn with_param<T: Hash>(mut self, name: &str, value: &T) -> Self {
        name.hash(&mut self.params_hasher);
        value.hash(&mut self.params_hasher);
        self
    }

    /// Add generation-specific parameters
    pub fn with_generation_params(
        mut self,
        max_length: Option<usize>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<usize>,
        do_sample: bool,
        num_beams: Option<usize>,
    ) -> Self {
        if let Some(v) = max_length {
            "max_length".hash(&mut self.params_hasher);
            v.hash(&mut self.params_hasher);
        }
        if let Some(v) = temperature {
            "temperature".hash(&mut self.params_hasher);
            v.to_bits().hash(&mut self.params_hasher);
        }
        if let Some(v) = top_p {
            "top_p".hash(&mut self.params_hasher);
            v.to_bits().hash(&mut self.params_hasher);
        }
        if let Some(v) = top_k {
            "top_k".hash(&mut self.params_hasher);
            v.hash(&mut self.params_hasher);
        }
        "do_sample".hash(&mut self.params_hasher);
        do_sample.hash(&mut self.params_hasher);
        if let Some(v) = num_beams {
            "num_beams".hash(&mut self.params_hasher);
            v.hash(&mut self.params_hasher);
        }
        self
    }

    /// Build the final cache key
    pub fn build(self) -> CacheKey {
        CacheKey::new(
            self.input_hasher.finish(),
            self.model_id,
            self.task,
            self.params_hasher.finish(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_key_builder() {
        let key1 = CacheKeyBuilder::new("bert-base", "text-classification")
            .with_text("Hello world")
            .with_param("max_length", &512)
            .build();

        let key2 = CacheKeyBuilder::new("bert-base", "text-classification")
            .with_text("Hello world")
            .with_param("max_length", &512)
            .build();

        let key3 = CacheKeyBuilder::new("bert-base", "text-classification")
            .with_text("Hello world!")
            .with_param("max_length", &512)
            .build();

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_generation_params_hashing() {
        let key1 = CacheKeyBuilder::new("gpt2", "text-generation")
            .with_text("Once upon a time")
            .with_generation_params(Some(100), Some(0.8), Some(0.9), Some(50), true, Some(4))
            .build();

        let key2 = CacheKeyBuilder::new("gpt2", "text-generation")
            .with_text("Once upon a time")
            .with_generation_params(Some(100), Some(0.8), Some(0.9), Some(50), true, Some(4))
            .build();

        let key3 = CacheKeyBuilder::new("gpt2", "text-generation")
            .with_text("Once upon a time")
            .with_generation_params(Some(100), Some(0.9), Some(0.9), Some(50), true, Some(4))
            .build();

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }
}
