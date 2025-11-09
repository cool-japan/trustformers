use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::Result;

/// Represents a word in the original text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Word {
    /// The word text
    pub text: String,
    /// Start position in the original text
    pub start: usize,
    /// End position in the original text
    pub end: usize,
    /// Index of the word in the sequence
    pub word_index: usize,
}

/// Represents the alignment between tokens and words
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenAlignment {
    /// Token index in the tokenized sequence
    pub token_index: usize,
    /// Word index that this token belongs to
    pub word_index: Option<usize>,
    /// Character start position in the original text
    pub char_start: usize,
    /// Character end position in the original text
    pub char_end: usize,
    /// Whether this token is a special token
    pub is_special: bool,
    /// Whether this token starts a word
    pub starts_word: bool,
    /// Whether this token ends a word
    pub ends_word: bool,
}

/// Represents a span in the text with word-level alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignedSpan {
    /// Start position in the original text
    pub start: usize,
    /// End position in the original text
    pub end: usize,
    /// Word indices that this span covers
    pub word_indices: Vec<usize>,
    /// Token indices that this span covers
    pub token_indices: Vec<usize>,
    /// The text content of the span
    pub text: String,
}

/// Configuration for word alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentConfig {
    /// Language-specific word boundary detection
    pub language: Option<String>,
    /// Whether to preserve entity boundaries
    pub preserve_entities: bool,
    /// Custom word separators
    pub word_separators: Vec<String>,
    /// Whether to handle contractions as single words
    pub handle_contractions: bool,
    /// Whether to split hyphenated words
    pub split_hyphenated: bool,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            language: None,
            preserve_entities: false,
            word_separators: vec![" ".to_string(), "\t".to_string(), "\n".to_string()],
            handle_contractions: true,
            split_hyphenated: false,
        }
    }
}

/// Token-to-word alignment engine
#[derive(Debug, Clone)]
pub struct AlignmentEngine {
    config: AlignmentConfig,
    /// Cached word boundaries for efficient lookup
    word_boundary_cache: HashMap<String, Vec<(usize, usize)>>,
}

impl AlignmentEngine {
    pub fn new(config: AlignmentConfig) -> Self {
        Self {
            config,
            word_boundary_cache: HashMap::new(),
        }
    }

    /// Extract words from text with their positions
    pub fn extract_words(&mut self, text: &str) -> Vec<Word> {
        if let Some(cached) = self.word_boundary_cache.get(text) {
            return cached
                .iter()
                .enumerate()
                .map(|(i, (start, end))| Word {
                    text: text[*start..*end].to_string(),
                    start: *start,
                    end: *end,
                    word_index: i,
                })
                .collect();
        }

        let word_boundaries = self.find_word_boundaries(text);
        let words = word_boundaries
            .iter()
            .enumerate()
            .map(|(i, (start, end))| Word {
                text: text[*start..*end].to_string(),
                start: *start,
                end: *end,
                word_index: i,
            })
            .collect();

        self.word_boundary_cache.insert(text.to_string(), word_boundaries);
        words
    }

    /// Find word boundaries in text
    fn find_word_boundaries(&self, text: &str) -> Vec<(usize, usize)> {
        let mut boundaries = Vec::new();
        let mut current_start = 0;
        let mut in_word = false;
        let chars = text.char_indices().peekable();

        for (i, ch) in chars {
            let is_separator = self.is_word_separator(ch);

            if !in_word && !is_separator {
                // Starting a new word
                current_start = i;
                in_word = true;
            } else if in_word && is_separator {
                // Ending a word
                boundaries.push((current_start, i));
                in_word = false;
            }
        }

        // Handle word at end of text
        if in_word {
            boundaries.push((current_start, text.len()));
        }

        // Handle contractions and hyphenated words
        if self.config.handle_contractions {
            boundaries = self.handle_contractions(text, boundaries);
        }

        if self.config.split_hyphenated {
            boundaries = self.split_hyphenated_words(text, boundaries);
        }

        boundaries
    }

    /// Check if a character is a word separator
    fn is_word_separator(&self, ch: char) -> bool {
        // Standard separators
        if ch.is_whitespace() {
            return true;
        }

        // Punctuation that separates words
        if ch.is_ascii_punctuation() {
            // Special handling for contractions and hyphenated words
            if self.config.handle_contractions && ch == '\'' {
                return false;
            }
            if !self.config.split_hyphenated && ch == '-' {
                return false;
            }
            return true;
        }

        // Custom separators
        self.config.word_separators.iter().any(|sep| sep.chars().any(|c| c == ch))
    }

    /// Handle contractions as single words
    fn handle_contractions(
        &self,
        text: &str,
        boundaries: Vec<(usize, usize)>,
    ) -> Vec<(usize, usize)> {
        let mut new_boundaries = Vec::new();
        let mut i = 0;

        while i < boundaries.len() {
            let (start, end) = boundaries[i];
            let _word_text = &text[start..end];

            // Check if this word is followed by an apostrophe + word
            if i + 1 < boundaries.len() {
                let next_start = boundaries[i + 1].0;
                let between_text = &text[end..next_start];

                if between_text.contains('\'') {
                    // Merge this word with the next one
                    let (_, next_end) = boundaries[i + 1];
                    new_boundaries.push((start, next_end));
                    i += 2; // Skip the next word
                    continue;
                }
            }

            new_boundaries.push((start, end));
            i += 1;
        }

        new_boundaries
    }

    /// Split hyphenated words
    fn split_hyphenated_words(
        &self,
        text: &str,
        boundaries: Vec<(usize, usize)>,
    ) -> Vec<(usize, usize)> {
        let mut new_boundaries = Vec::new();

        for (start, end) in boundaries {
            let word_text = &text[start..end];
            if word_text.contains('-') {
                // Split on hyphens
                let mut current_start = start;
                for (i, ch) in word_text.char_indices() {
                    if ch == '-' {
                        if current_start < start + i {
                            new_boundaries.push((current_start, start + i));
                        }
                        current_start = start + i + 1;
                    }
                }
                if current_start < end {
                    new_boundaries.push((current_start, end));
                }
            } else {
                new_boundaries.push((start, end));
            }
        }

        new_boundaries
    }

    /// Align tokens to words
    pub fn align_tokens_to_words(
        &mut self,
        text: &str,
        token_offsets: &[(usize, usize)],
        special_tokens_mask: Option<&[u8]>,
    ) -> Result<Vec<TokenAlignment>> {
        let words = self.extract_words(text);
        let mut alignments = Vec::new();

        for (token_index, (token_start, token_end)) in token_offsets.iter().enumerate() {
            let is_special = special_tokens_mask
                .map(|mask| mask.get(token_index).copied().unwrap_or(0) == 1)
                .unwrap_or(false);

            if is_special {
                // Special tokens don't align to words
                alignments.push(TokenAlignment {
                    token_index,
                    word_index: None,
                    char_start: *token_start,
                    char_end: *token_end,
                    is_special: true,
                    starts_word: false,
                    ends_word: false,
                });
                continue;
            }

            // Find which word this token belongs to
            let word_index = self.find_word_for_token(&words, *token_start, *token_end);

            // Determine if this token starts or ends a word
            let (starts_word, ends_word) = if let Some(word_idx) = word_index {
                let word = &words[word_idx];
                let starts = *token_start == word.start;
                let ends = *token_end == word.end;
                (starts, ends)
            } else {
                (false, false)
            };

            alignments.push(TokenAlignment {
                token_index,
                word_index,
                char_start: *token_start,
                char_end: *token_end,
                is_special,
                starts_word,
                ends_word,
            });
        }

        Ok(alignments)
    }

    /// Find which word a token belongs to
    fn find_word_for_token(
        &self,
        words: &[Word],
        token_start: usize,
        token_end: usize,
    ) -> Option<usize> {
        // Find the word that contains this token
        for (i, word) in words.iter().enumerate() {
            if token_start >= word.start && token_end <= word.end {
                return Some(i);
            }
            // Handle partial overlaps (subword tokens)
            if token_start < word.end && token_end > word.start {
                return Some(i);
            }
        }
        None
    }

    /// Extract spans with word-level alignment
    pub fn extract_spans(
        &mut self,
        text: &str,
        alignments: &[TokenAlignment],
        spans: &[(usize, usize)],
    ) -> Result<Vec<AlignedSpan>> {
        let words = self.extract_words(text);
        let mut aligned_spans = Vec::new();

        for (span_start, span_end) in spans {
            let mut word_indices = Vec::new();
            let mut token_indices = Vec::new();

            // Find words covered by this span
            for word in &words {
                if word.start < *span_end && word.end > *span_start {
                    word_indices.push(word.word_index);
                }
            }

            // Find tokens covered by this span
            for alignment in alignments {
                if alignment.char_start < *span_end && alignment.char_end > *span_start {
                    token_indices.push(alignment.token_index);
                }
            }

            let span_text = text[*span_start..*span_end].to_string();

            aligned_spans.push(AlignedSpan {
                start: *span_start,
                end: *span_end,
                word_indices,
                token_indices,
                text: span_text,
            });
        }

        Ok(aligned_spans)
    }

    /// Get word boundaries for a specific token
    pub fn get_word_boundaries_for_token(
        &self,
        alignments: &[TokenAlignment],
        token_index: usize,
    ) -> Option<(usize, usize)> {
        if let Some(alignment) = alignments.get(token_index) {
            if let Some(word_idx) = alignment.word_index {
                // Find the full word span
                let word_start = alignments
                    .iter()
                    .filter(|a| a.word_index == Some(word_idx))
                    .map(|a| a.char_start)
                    .min()
                    .unwrap_or(alignment.char_start);

                let word_end = alignments
                    .iter()
                    .filter(|a| a.word_index == Some(word_idx))
                    .map(|a| a.char_end)
                    .max()
                    .unwrap_or(alignment.char_end);

                return Some((word_start, word_end));
            }
        }
        None
    }

    /// Check if tokens form a complete word
    pub fn tokens_form_complete_word(
        &self,
        alignments: &[TokenAlignment],
        token_indices: &[usize],
    ) -> bool {
        if token_indices.is_empty() {
            return false;
        }

        // Get the word indices for these tokens
        let mut word_indices = std::collections::HashSet::new();
        for &token_idx in token_indices {
            if let Some(alignment) = alignments.get(token_idx) {
                if let Some(word_idx) = alignment.word_index {
                    word_indices.insert(word_idx);
                }
            }
        }

        // Check if we have exactly one word
        if word_indices.len() != 1 {
            return false;
        }

        let word_idx = *word_indices.iter().next().unwrap();

        // Check if these tokens cover the entire word
        let word_tokens: Vec<usize> = alignments
            .iter()
            .filter(|a| a.word_index == Some(word_idx))
            .map(|a| a.token_index)
            .collect();

        let mut token_indices_sorted = token_indices.to_vec();
        token_indices_sorted.sort();
        let mut word_tokens_sorted = word_tokens;
        word_tokens_sorted.sort();

        token_indices_sorted == word_tokens_sorted
    }

    /// Preserve entity boundaries during alignment
    pub fn preserve_entities(
        &mut self,
        text: &str,
        alignments: &[TokenAlignment],
        entities: &[(usize, usize, String)], // (start, end, label)
    ) -> Result<Vec<AlignedSpan>> {
        let mut entity_spans = Vec::new();

        for (start, end, _label) in entities {
            let mut word_indices = Vec::new();
            let mut token_indices = Vec::new();

            // Find words and tokens within this entity
            for alignment in alignments {
                if alignment.char_start >= *start && alignment.char_end <= *end {
                    token_indices.push(alignment.token_index);
                    if let Some(word_idx) = alignment.word_index {
                        if !word_indices.contains(&word_idx) {
                            word_indices.push(word_idx);
                        }
                    }
                }
            }

            let entity_text = text[*start..*end].to_string();

            entity_spans.push(AlignedSpan {
                start: *start,
                end: *end,
                word_indices,
                token_indices,
                text: entity_text,
            });
        }

        Ok(entity_spans)
    }
}

/// Utility functions for common alignment tasks
impl AlignmentEngine {
    /// Get all tokens that belong to a specific word
    pub fn get_tokens_for_word(
        &self,
        alignments: &[TokenAlignment],
        word_index: usize,
    ) -> Vec<usize> {
        alignments
            .iter()
            .filter(|a| a.word_index == Some(word_index))
            .map(|a| a.token_index)
            .collect()
    }

    /// Get the word index for a token
    pub fn get_word_for_token(
        &self,
        alignments: &[TokenAlignment],
        token_index: usize,
    ) -> Option<usize> {
        alignments.get(token_index).and_then(|a| a.word_index)
    }

    /// Check if a token starts a word
    pub fn token_starts_word(&self, alignments: &[TokenAlignment], token_index: usize) -> bool {
        alignments.get(token_index).map(|a| a.starts_word).unwrap_or(false)
    }

    /// Check if a token ends a word
    pub fn token_ends_word(&self, alignments: &[TokenAlignment], token_index: usize) -> bool {
        alignments.get(token_index).map(|a| a.ends_word).unwrap_or(false)
    }

    /// Get statistics about the alignment
    pub fn get_alignment_stats(&self, alignments: &[TokenAlignment]) -> AlignmentStats {
        let total_tokens = alignments.len();
        let special_tokens = alignments.iter().filter(|a| a.is_special).count();
        let aligned_tokens = alignments.iter().filter(|a| a.word_index.is_some()).count();

        let unique_words = alignments
            .iter()
            .filter_map(|a| a.word_index)
            .collect::<std::collections::HashSet<_>>()
            .len();

        AlignmentStats {
            total_tokens,
            special_tokens,
            aligned_tokens,
            unique_words,
            alignment_ratio: aligned_tokens as f64 / total_tokens as f64,
        }
    }
}

/// Statistics about token-to-word alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentStats {
    pub total_tokens: usize,
    pub special_tokens: usize,
    pub aligned_tokens: usize,
    pub unique_words: usize,
    pub alignment_ratio: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_extraction() {
        let mut engine = AlignmentEngine::new(AlignmentConfig::default());
        let text = "Hello, world! This is a test.";
        let words = engine.extract_words(text);

        assert_eq!(words.len(), 6);
        assert_eq!(words[0].text, "Hello");
        assert_eq!(words[1].text, "world");
        assert_eq!(words[2].text, "This");
        assert_eq!(words[3].text, "is");
        assert_eq!(words[4].text, "a");
        assert_eq!(words[5].text, "test");
    }

    #[test]
    fn test_contractions() {
        let mut config = AlignmentConfig::default();
        config.handle_contractions = true;
        let mut engine = AlignmentEngine::new(config);

        let text = "I'm can't won't";
        let words = engine.extract_words(text);

        assert_eq!(words.len(), 3);
        assert_eq!(words[0].text, "I'm");
        assert_eq!(words[1].text, "can't");
        assert_eq!(words[2].text, "won't");
    }

    #[test]
    fn test_hyphenated_words() {
        let mut config = AlignmentConfig::default();
        config.split_hyphenated = true;
        let mut engine = AlignmentEngine::new(config);

        let text = "state-of-the-art";
        let words = engine.extract_words(text);

        assert_eq!(words.len(), 4);
        assert_eq!(words[0].text, "state");
        assert_eq!(words[1].text, "of");
        assert_eq!(words[2].text, "the");
        assert_eq!(words[3].text, "art");
    }

    #[test]
    fn test_token_alignment() {
        let mut engine = AlignmentEngine::new(AlignmentConfig::default());
        let text = "Hello world";
        let token_offsets = vec![(0, 5), (6, 11)]; // "Hello", "world"

        let alignments = engine.align_tokens_to_words(text, &token_offsets, None).unwrap();

        assert_eq!(alignments.len(), 2);
        assert_eq!(alignments[0].word_index, Some(0));
        assert_eq!(alignments[1].word_index, Some(1));
        assert!(alignments[0].starts_word);
        assert!(alignments[0].ends_word);
        assert!(alignments[1].starts_word);
        assert!(alignments[1].ends_word);
    }

    #[test]
    fn test_subword_alignment() {
        let mut engine = AlignmentEngine::new(AlignmentConfig::default());
        let text = "Hello world";
        let token_offsets = vec![(0, 3), (3, 5), (6, 11)]; // "Hel", "lo", "world"

        let alignments = engine.align_tokens_to_words(text, &token_offsets, None).unwrap();

        assert_eq!(alignments.len(), 3);
        assert_eq!(alignments[0].word_index, Some(0));
        assert_eq!(alignments[1].word_index, Some(0));
        assert_eq!(alignments[2].word_index, Some(1));
        assert!(alignments[0].starts_word);
        assert!(!alignments[0].ends_word);
        assert!(!alignments[1].starts_word);
        assert!(alignments[1].ends_word);
    }

    #[test]
    fn test_alignment_stats() {
        let engine = AlignmentEngine::new(AlignmentConfig::default());
        let alignments = vec![
            TokenAlignment {
                token_index: 0,
                word_index: Some(0),
                char_start: 0,
                char_end: 5,
                is_special: false,
                starts_word: true,
                ends_word: true,
            },
            TokenAlignment {
                token_index: 1,
                word_index: None,
                char_start: 0,
                char_end: 0,
                is_special: true,
                starts_word: false,
                ends_word: false,
            },
        ];

        let stats = engine.get_alignment_stats(&alignments);
        assert_eq!(stats.total_tokens, 2);
        assert_eq!(stats.special_tokens, 1);
        assert_eq!(stats.aligned_tokens, 1);
        assert_eq!(stats.unique_words, 1);
        assert_eq!(stats.alignment_ratio, 0.5);
    }
}
