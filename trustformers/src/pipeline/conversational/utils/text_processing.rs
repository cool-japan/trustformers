//! Text processing utilities for conversational AI.
//!
//! This module provides comprehensive text processing capabilities including
//! token estimation, text cleaning, normalization, and analysis utilities.

use regex::Regex;

use crate::core::error::Result;
use crate::core::traits::Tokenizer;

/// Text processing utilities for conversation handling
pub struct TextProcessor;

impl TextProcessor {
    /// Estimate token count for text using simple heuristics
    pub fn estimate_token_count(text: &str) -> usize {
        // Rough estimation: 1 token per 4 characters on average
        // This is a fallback when tokenizer is not available
        text.len() / 4
    }

    /// Estimate token count using actual tokenizer if available
    pub fn estimate_token_count_with_tokenizer<T: Tokenizer>(
        text: &str,
        tokenizer: &T,
    ) -> Result<usize> {
        match tokenizer.encode(text) {
            Ok(tokenized) => Ok(tokenized.input_ids.len()),
            Err(_) => Ok(Self::estimate_token_count(text)), // Fallback to estimation
        }
    }

    /// Clean and normalize text for processing
    pub fn clean_text(text: &str) -> String {
        let mut cleaned = text.trim().to_string();

        // Remove excessive whitespace
        cleaned = cleaned.replace("  ", " ");
        cleaned = cleaned.replace("\t", " ");
        cleaned = cleaned.replace("\r\n", "\n");
        cleaned = cleaned.replace("\r", "\n");

        // Normalize line breaks
        cleaned = cleaned.replace("\n\n\n", "\n\n");

        cleaned
    }

    /// Clean generated response text
    pub fn clean_generated_response(response: &str) -> String {
        let mut cleaned = response.trim().to_string();

        // Remove common generation artifacts
        cleaned = cleaned.replace("<|endoftext|>", "");
        cleaned = cleaned.replace("<|end|>", "");
        cleaned = cleaned.replace("<eos>", "");
        cleaned = cleaned.replace("<|assistant|>", "");
        cleaned = cleaned.replace("<|user|>", "");
        cleaned = cleaned.replace("<|system|>", "");

        // Normalize multiple newlines
        cleaned = cleaned.replace("\n\n\n", "\n\n");
        cleaned = cleaned.replace("\n\n", "\n");

        // Ensure proper sentence ending
        if !cleaned.ends_with(['.', '!', '?', '\n']) && !cleaned.is_empty() {
            cleaned.push('.');
        }

        // Truncate if too long (safety limit)
        if cleaned.len() > 2000 {
            cleaned.truncate(1997);
            cleaned.push_str("...");
        }

        cleaned.trim().to_string()
    }

    /// Enhanced response cleaning with more comprehensive artifact removal
    pub fn clean_generated_response_enhanced(response: &str) -> String {
        let mut cleaned = response.trim().to_string();

        // Remove common generation artifacts
        let artifacts = [
            "<|endoftext|>",
            "<|end|>",
            "<eos>",
            "<|assistant|>",
            "<|user|>",
            "<|system|>",
            "</s>",
            "<s>",
            "<pad>",
            "<unk>",
            "<mask>",
            "[CLS]",
            "[SEP]",
            "[PAD]",
            "[UNK]",
            "[MASK]",
        ];

        for artifact in artifacts {
            cleaned = cleaned.replace(artifact, "");
        }

        // Remove excessive whitespace and normalize
        cleaned = cleaned.replace("\n\n\n", "\n\n");
        cleaned = cleaned.replace("\n\n", "\n");
        cleaned = cleaned.replace("  ", " ");

        // Ensure proper sentence ending
        if !cleaned.ends_with(['.', '!', '?']) && !cleaned.is_empty() {
            cleaned.push('.');
        }

        // Truncate if too long (safety limit)
        if cleaned.len() > 2000 {
            cleaned.truncate(1997);
            cleaned.push_str("...");
        }

        cleaned.trim().to_string()
    }

    /// Normalize text for comparison and analysis
    pub fn normalize_for_comparison(text: &str) -> String {
        text.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Extract sentences from text
    pub fn extract_sentences(text: &str) -> Vec<String> {
        // Simple sentence splitting on common sentence endings
        let sentence_regex = Regex::new(r"[.!?]+\s+").expect("static regex pattern is valid");
        sentence_regex
            .split(text)
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().to_string())
            .collect()
    }

    /// Count words in text
    pub fn count_words(text: &str) -> usize {
        text.split_whitespace().count()
    }

    /// Calculate reading time estimate (words per minute)
    pub fn estimate_reading_time(text: &str, wpm: usize) -> std::time::Duration {
        let word_count = Self::count_words(text);
        let minutes = (word_count as f64 / wpm as f64).ceil() as u64;
        std::time::Duration::from_secs(minutes * 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_token_count_empty() {
        let count = TextProcessor::estimate_token_count("");
        assert_eq!(count, 0);
    }

    #[test]
    fn test_estimate_token_count_basic() {
        let text = "Hello, world! This is a test."; // 29 chars => 7
        let count = TextProcessor::estimate_token_count(text);
        assert_eq!(count, text.len() / 4);
    }

    #[test]
    fn test_clean_text_trims_whitespace() {
        let result = TextProcessor::clean_text("  hello world  ");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_clean_text_collapses_double_spaces() {
        let result = TextProcessor::clean_text("hello  world");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_clean_text_normalizes_tabs() {
        let result = TextProcessor::clean_text("hello\tworld");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_clean_text_normalizes_crlf() {
        let result = TextProcessor::clean_text("line1\r\nline2");
        assert_eq!(result, "line1\nline2");
    }

    #[test]
    fn test_clean_generated_response_removes_eos_tokens() {
        let result = TextProcessor::clean_generated_response("Hello<|endoftext|>");
        assert!(!result.contains("<|endoftext|>"));
    }

    #[test]
    fn test_clean_generated_response_removes_end_token() {
        let result = TextProcessor::clean_generated_response("Hello<|end|> World");
        assert!(!result.contains("<|end|>"));
    }

    #[test]
    fn test_clean_generated_response_adds_period() {
        let result = TextProcessor::clean_generated_response("Hello world");
        assert!(result.ends_with('.'));
    }

    #[test]
    fn test_clean_generated_response_no_double_period() {
        let result = TextProcessor::clean_generated_response("Hello world.");
        assert!(result.ends_with('.'));
        assert!(!result.ends_with(".."));
    }

    #[test]
    fn test_clean_generated_response_truncates_long_text() {
        let long_text = "a".repeat(2500);
        let result = TextProcessor::clean_generated_response(&long_text);
        assert!(result.len() <= 2000);
        assert!(result.ends_with("..."));
    }

    #[test]
    fn test_clean_generated_response_enhanced_removes_bert_tokens() {
        let result = TextProcessor::clean_generated_response_enhanced("[CLS] Hello [SEP]");
        assert!(!result.contains("[CLS]"));
        assert!(!result.contains("[SEP]"));
    }

    #[test]
    fn test_clean_generated_response_enhanced_removes_pad_tokens() {
        let result = TextProcessor::clean_generated_response_enhanced("text[PAD][UNK]");
        assert!(!result.contains("[PAD]"));
        assert!(!result.contains("[UNK]"));
    }

    #[test]
    fn test_normalize_for_comparison_lowercases() {
        let result = TextProcessor::normalize_for_comparison("Hello World");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_normalize_for_comparison_removes_punctuation() {
        let result = TextProcessor::normalize_for_comparison("hello, world!");
        assert!(!result.contains(','));
        assert!(!result.contains('!'));
    }

    #[test]
    fn test_normalize_for_comparison_collapses_whitespace() {
        let result = TextProcessor::normalize_for_comparison("hello   world");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_extract_sentences_basic() {
        let sentences = TextProcessor::extract_sentences("Hello world. How are you? I am fine.");
        assert!(!sentences.is_empty());
    }

    #[test]
    fn test_count_words_empty() {
        assert_eq!(TextProcessor::count_words(""), 0);
    }

    #[test]
    fn test_count_words_single() {
        assert_eq!(TextProcessor::count_words("hello"), 1);
    }

    #[test]
    fn test_count_words_multiple() {
        assert_eq!(TextProcessor::count_words("one two three four"), 4);
    }

    #[test]
    fn test_estimate_reading_time_basic() {
        // 200 words at 200 wpm = 1 minute
        let text = (0..200).map(|_| "word").collect::<Vec<_>>().join(" ");
        let duration = TextProcessor::estimate_reading_time(&text, 200);
        assert_eq!(duration.as_secs(), 60);
    }

    #[test]
    fn test_estimate_reading_time_zero_wpm_division_safe() {
        // With wpm=1 and 60 words => 60 minutes
        let text = (0..60).map(|_| "word").collect::<Vec<_>>().join(" ");
        let duration = TextProcessor::estimate_reading_time(&text, 1);
        assert_eq!(duration.as_secs(), 3600);
    }
}
