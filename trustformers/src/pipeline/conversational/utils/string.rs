//! String manipulation and validation utilities.
//!
//! This module provides utilities for string processing, keyword extraction,
//! similarity calculation, and pattern matching for conversational AI.

use std::collections::HashSet;

/// String manipulation and validation utilities
pub struct StringUtils;

impl StringUtils {
    /// Check if string contains any of the patterns
    pub fn contains_any(text: &str, patterns: &[&str]) -> bool {
        let text_lower = text.to_lowercase();
        patterns.iter().any(|pattern| text_lower.contains(&pattern.to_lowercase()))
    }

    /// Check if string matches any of the patterns (case-insensitive)
    pub fn matches_any(text: &str, patterns: &[&str]) -> bool {
        let text_lower = text.to_lowercase();
        patterns.iter().any(|pattern| text_lower == pattern.to_lowercase())
    }

    /// Extract keywords from text
    pub fn extract_keywords(text: &str, min_length: usize) -> Vec<String> {
        let stop_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does",
            "did", "will", "would", "could", "should", "may", "might", "can", "this", "that",
            "these", "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
            "us", "them",
        ];

        text.split_whitespace()
            .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|word| word.len() >= min_length && !stop_words.contains(&word.as_str()))
            .collect()
    }

    /// Calculate string similarity (simple implementation)
    pub fn string_similarity(s1: &str, s2: &str) -> f32 {
        if s1 == s2 {
            return 1.0;
        }

        let s1_lower = s1.to_lowercase();
        let s2_lower = s2.to_lowercase();

        if s1_lower == s2_lower {
            return 0.9;
        }

        // Simple word-based similarity
        let words1: HashSet<&str> = s1_lower.split_whitespace().collect();
        let words2: HashSet<&str> = s2_lower.split_whitespace().collect();

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            0.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Validate if text is meaningful (not just noise)
    pub fn is_meaningful(text: &str) -> bool {
        let cleaned = text.trim();

        // Check minimum length
        if cleaned.len() < 2 {
            return false;
        }

        // Check if it's not just repeated characters
        let unique_chars: HashSet<char> = cleaned.chars().collect();
        if unique_chars.len() == 1 {
            return false;
        }

        // Check for at least one alphabetic character
        cleaned.chars().any(|c| c.is_alphabetic())
    }

    /// Remove excessive whitespace and normalize text
    pub fn normalize_whitespace(text: &str) -> String {
        text.split_whitespace().collect::<Vec<&str>>().join(" ")
    }

    /// Truncate text while preserving word boundaries
    pub fn truncate_words(text: &str, max_length: usize) -> String {
        if text.len() <= max_length {
            return text.to_string();
        }

        let mut truncated = String::new();
        let mut current_len = 0;

        for word in text.split_whitespace() {
            if current_len + word.len() + 1 > max_length {
                break;
            }

            if !truncated.is_empty() {
                truncated.push(' ');
                current_len += 1;
            }

            truncated.push_str(word);
            current_len += word.len();
        }

        if current_len < text.len() {
            truncated.push_str("...");
        }

        truncated
    }
}
