//! # TextPreprocessingConfig - Trait Implementations
//!
//! This module contains trait implementations for `TextPreprocessingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::TextPreprocessingConfig;

impl Default for TextPreprocessingConfig {
    fn default() -> Self {
        Self {
            normalize_whitespace: true,
            to_lowercase: false,
            remove_special_chars: false,
            remove_numbers: false,
            remove_stop_words: false,
            stop_words_language: "english".to_string(),
            enable_stemming: false,
            enable_lemmatization: false,
            min_word_length: 1,
            max_word_length: 50,
            spell_check: false,
            spell_check_language: "en".to_string(),
        }
    }
}
