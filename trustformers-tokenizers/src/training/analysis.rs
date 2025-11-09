//! Advanced analysis tools for tokenizer training and evaluation.
//!
//! This module provides comprehensive analysis capabilities including:
//! - Coverage analysis for evaluating tokenizer effectiveness
//! - Language detection based on character and n-gram frequencies
//! - Token distribution analysis for vocabulary optimization
//! - Statistical analysis of tokenization patterns

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Detailed coverage analysis results for tokenizer evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageAnalysis {
    /// Character-level coverage rate (0.0 to 1.0)
    pub char_coverage_rate: f64,
    /// Word-level coverage rate (0.0 to 1.0)
    pub word_coverage_rate: f64,
    /// Compression ratio (tokens per character)
    pub compression_ratio: f64,
    /// Total characters in the test corpus
    pub total_chars: usize,
    /// Total words in the test corpus
    pub total_words: usize,
    /// Total tokens produced by tokenization
    pub total_tokens: usize,
    /// Number of characters covered by the vocabulary
    pub covered_chars: usize,
    /// Number of words covered by the vocabulary
    pub covered_words: usize,
    /// Distribution of token lengths
    pub length_distribution: HashMap<usize, u32>,
    /// List of out-of-vocabulary tokens encountered
    pub oov_tokens: Vec<String>,
    /// Size of the tokenizer vocabulary
    pub vocab_size: usize,
}

impl CoverageAnalysis {
    /// Generate a comprehensive summary report of the coverage analysis.
    pub fn summary(&self) -> String {
        format!(
            "Coverage Analysis Summary:\n\
             - Character Coverage: {:.2}% ({}/{})\n\
             - Word Coverage: {:.2}% ({}/{})\n\
             - Compression Ratio: {:.3}\n\
             - Vocabulary Size: {}\n\
             - OOV Tokens: {}\n\
             - Average Token Length: {:.2}",
            self.char_coverage_rate * 100.0,
            self.covered_chars,
            self.total_chars,
            self.word_coverage_rate * 100.0,
            self.covered_words,
            self.total_words,
            self.compression_ratio,
            self.vocab_size,
            self.oov_tokens.len(),
            self.average_token_length()
        )
    }

    /// Calculate average token length based on the length distribution.
    pub fn average_token_length(&self) -> f64 {
        let total_tokens: u32 = self.length_distribution.values().sum();
        if total_tokens == 0 {
            return 0.0;
        }

        let weighted_sum: u32 = self
            .length_distribution
            .iter()
            .map(|(&length, &count)| length as u32 * count)
            .sum();

        weighted_sum as f64 / total_tokens as f64
    }

    /// Get the most common token lengths with their frequencies.
    pub fn top_token_lengths(&self, n: usize) -> Vec<(usize, u32)> {
        let mut lengths: Vec<_> = self.length_distribution.iter().collect();
        lengths.sort_by(|a, b| b.1.cmp(a.1));
        lengths.into_iter().take(n).map(|(&len, &count)| (len, count)).collect()
    }

    /// Calculate efficiency score combining coverage and compression.
    pub fn efficiency_score(&self) -> f64 {
        // Weighted combination of character coverage and inverse compression ratio
        let coverage_score = 0.6 * self.char_coverage_rate + 0.4 * self.word_coverage_rate;
        let compression_score = 1.0 / (1.0 + self.compression_ratio);
        0.7 * coverage_score + 0.3 * compression_score
    }
}

/// Language detection based on character frequency analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageDetector {
    /// Character frequency profiles for different languages
    language_profiles: HashMap<String, HashMap<char, f64>>,
    /// N-gram profiles for enhanced detection accuracy
    ngram_profiles: HashMap<String, HashMap<String, f64>>,
    /// List of supported languages
    supported_languages: Vec<String>,
}

impl LanguageDetector {
    /// Create a new language detector with built-in language profiles.
    pub fn new() -> Self {
        let mut detector = Self {
            language_profiles: HashMap::new(),
            ngram_profiles: HashMap::new(),
            supported_languages: Vec::new(),
        };

        detector.initialize_built_in_profiles();
        detector
    }

    /// Initialize built-in language profiles for common languages.
    fn initialize_built_in_profiles(&mut self) {
        // English character frequencies (approximated from corpora)
        let mut english_chars = HashMap::new();
        english_chars.insert('e', 12.7);
        english_chars.insert('t', 9.1);
        english_chars.insert('a', 8.2);
        english_chars.insert('o', 7.5);
        english_chars.insert('i', 7.0);
        english_chars.insert('n', 6.7);
        english_chars.insert('s', 6.3);
        english_chars.insert('h', 6.1);
        english_chars.insert('r', 6.0);

        let mut english_ngrams = HashMap::new();
        english_ngrams.insert("th".to_string(), 2.7);
        english_ngrams.insert("he".to_string(), 2.3);
        english_ngrams.insert("in".to_string(), 2.0);
        english_ngrams.insert("er".to_string(), 1.8);
        english_ngrams.insert("an".to_string(), 1.6);

        self.language_profiles.insert("en".to_string(), english_chars);
        self.ngram_profiles.insert("en".to_string(), english_ngrams);

        // Spanish character frequencies
        let mut spanish_chars = HashMap::new();
        spanish_chars.insert('e', 13.7);
        spanish_chars.insert('a', 11.5);
        spanish_chars.insert('o', 8.7);
        spanish_chars.insert('s', 8.0);
        spanish_chars.insert('r', 6.9);
        spanish_chars.insert('n', 6.7);
        spanish_chars.insert('i', 6.2);
        spanish_chars.insert('d', 5.9);
        spanish_chars.insert('l', 5.0);

        let mut spanish_ngrams = HashMap::new();
        spanish_ngrams.insert("de".to_string(), 2.8);
        spanish_ngrams.insert("la".to_string(), 2.5);
        spanish_ngrams.insert("es".to_string(), 2.1);
        spanish_ngrams.insert("en".to_string(), 1.9);
        spanish_ngrams.insert("el".to_string(), 1.7);

        self.language_profiles.insert("es".to_string(), spanish_chars);
        self.ngram_profiles.insert("es".to_string(), spanish_ngrams);

        // French character frequencies
        let mut french_chars = HashMap::new();
        french_chars.insert('e', 14.7);
        french_chars.insert('s', 7.9);
        french_chars.insert('a', 7.6);
        french_chars.insert('i', 7.5);
        french_chars.insert('t', 7.2);
        french_chars.insert('n', 7.1);
        french_chars.insert('r', 6.6);
        french_chars.insert('u', 6.3);
        french_chars.insert('l', 5.5);

        let mut french_ngrams = HashMap::new();
        french_ngrams.insert("de".to_string(), 3.0);
        french_ngrams.insert("le".to_string(), 2.4);
        french_ngrams.insert("es".to_string(), 2.1);
        french_ngrams.insert("re".to_string(), 1.8);
        french_ngrams.insert("nt".to_string(), 1.6);

        self.language_profiles.insert("fr".to_string(), french_chars);
        self.ngram_profiles.insert("fr".to_string(), french_ngrams);

        // German character frequencies
        let mut german_chars = HashMap::new();
        german_chars.insert('e', 17.4);
        german_chars.insert('n', 9.8);
        german_chars.insert('i', 7.6);
        german_chars.insert('s', 7.3);
        german_chars.insert('r', 7.0);
        german_chars.insert('a', 6.5);
        german_chars.insert('t', 6.2);
        german_chars.insert('d', 5.1);
        german_chars.insert('h', 4.8);

        let mut german_ngrams = HashMap::new();
        german_ngrams.insert("er".to_string(), 3.9);
        german_ngrams.insert("en".to_string(), 3.6);
        german_ngrams.insert("ch".to_string(), 2.4);
        german_ngrams.insert("de".to_string(), 2.1);
        german_ngrams.insert("ei".to_string(), 1.8);

        self.language_profiles.insert("de".to_string(), german_chars);
        self.ngram_profiles.insert("de".to_string(), german_ngrams);

        self.supported_languages = vec![
            "en".to_string(),
            "es".to_string(),
            "fr".to_string(),
            "de".to_string(),
        ];
    }

    /// Detect the language of a given text.
    ///
    /// Returns a `LanguageDetectionResult` with the detected language,
    /// confidence score, and scores for all supported languages.
    pub fn detect_language(&self, text: &str) -> LanguageDetectionResult {
        if text.trim().is_empty() {
            return LanguageDetectionResult {
                detected_language: "unknown".to_string(),
                confidence: 0.0,
                scores: HashMap::new(),
            };
        }

        let text_lower = text.to_lowercase();
        let char_freq = self.calculate_char_frequency(&text_lower);
        let ngram_freq = self.calculate_ngram_frequency(&text_lower, 2);

        let mut scores = HashMap::new();

        for lang in &self.supported_languages {
            let char_score = self.calculate_char_similarity(&char_freq, lang);
            let ngram_score = self.calculate_ngram_similarity(&ngram_freq, lang);

            // Weighted combination of character and n-gram scores
            let combined_score = 0.6 * char_score + 0.4 * ngram_score;
            scores.insert(lang.clone(), combined_score);
        }

        let (detected_language, confidence) = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(lang, score)| (lang.clone(), *score))
            .unwrap_or(("unknown".to_string(), 0.0));

        LanguageDetectionResult {
            detected_language,
            confidence,
            scores,
        }
    }

    /// Calculate character frequency distribution in text.
    fn calculate_char_frequency(&self, text: &str) -> HashMap<char, f64> {
        let mut freq = HashMap::new();
        let total_chars = text.chars().filter(|c| c.is_alphabetic()).count() as f64;

        if total_chars == 0.0 {
            return freq;
        }

        for ch in text.chars() {
            if ch.is_alphabetic() {
                *freq.entry(ch).or_insert(0.0) += 1.0;
            }
        }

        // Convert to percentages
        for value in freq.values_mut() {
            *value = (*value / total_chars) * 100.0;
        }

        freq
    }

    /// Calculate n-gram frequency distribution in text.
    fn calculate_ngram_frequency(&self, text: &str, n: usize) -> HashMap<String, f64> {
        let mut freq = HashMap::new();
        let chars: Vec<char> = text.chars().filter(|c| c.is_alphabetic()).collect();
        let total_ngrams = chars.len().saturating_sub(n - 1) as f64;

        if total_ngrams == 0.0 {
            return freq;
        }

        for window in chars.windows(n) {
            let ngram: String = window.iter().collect();
            *freq.entry(ngram).or_insert(0.0) += 1.0;
        }

        // Convert to percentages
        for value in freq.values_mut() {
            *value = (*value / total_ngrams) * 100.0;
        }

        freq
    }

    /// Calculate similarity between text character frequency and language profile.
    fn calculate_char_similarity(&self, text_freq: &HashMap<char, f64>, language: &str) -> f64 {
        let profile = match self.language_profiles.get(language) {
            Some(p) => p,
            None => return 0.0,
        };

        let mut similarity = 0.0;
        let mut total_chars = 0;

        for (ch, expected_freq) in profile {
            let actual_freq = text_freq.get(ch).unwrap_or(&0.0);
            similarity += 1.0 / (1.0 + (expected_freq - actual_freq).abs());
            total_chars += 1;
        }

        if total_chars > 0 {
            similarity / total_chars as f64
        } else {
            0.0
        }
    }

    /// Calculate similarity between text n-gram frequency and language profile.
    fn calculate_ngram_similarity(&self, text_freq: &HashMap<String, f64>, language: &str) -> f64 {
        let profile = match self.ngram_profiles.get(language) {
            Some(p) => p,
            None => return 0.0,
        };

        let mut similarity = 0.0;
        let mut total_ngrams = 0;

        for (ngram, expected_freq) in profile {
            let actual_freq = text_freq.get(ngram).unwrap_or(&0.0);
            similarity += 1.0 / (1.0 + (expected_freq - actual_freq).abs());
            total_ngrams += 1;
        }

        if total_ngrams > 0 {
            similarity / total_ngrams as f64
        } else {
            0.0
        }
    }

    /// Get list of supported languages.
    pub fn supported_languages(&self) -> &[String] {
        &self.supported_languages
    }

    /// Add a custom language profile.
    pub fn add_language_profile(
        &mut self,
        language: String,
        char_profile: HashMap<char, f64>,
        ngram_profile: HashMap<String, f64>,
    ) {
        self.language_profiles.insert(language.clone(), char_profile);
        self.ngram_profiles.insert(language.clone(), ngram_profile);
        if !self.supported_languages.contains(&language) {
            self.supported_languages.push(language);
        }
    }
}

impl Default for LanguageDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Language detection result containing detected language and confidence scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageDetectionResult {
    /// The detected language code (e.g., "en", "es", "fr", "de")
    pub detected_language: String,
    /// Confidence score for the detected language (0.0 to 1.0)
    pub confidence: f64,
    /// Scores for all supported languages
    pub scores: HashMap<String, f64>,
}

impl LanguageDetectionResult {
    /// Get the top N language candidates with their scores.
    pub fn top_candidates(&self, n: usize) -> Vec<(String, f64)> {
        let mut sorted_scores: Vec<_> = self.scores.iter().collect();
        sorted_scores.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted_scores
            .into_iter()
            .take(n)
            .map(|(lang, score)| (lang.clone(), *score))
            .collect()
    }

    /// Check if the detection confidence is above a threshold.
    pub fn is_confident(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }
}

/// Enhanced token distribution analysis for vocabulary optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenDistributionAnalyzer {
    /// Length distribution bins for analysis
    length_bins: Vec<usize>,
    /// Frequency distribution bins for analysis
    frequency_bins: Vec<usize>,
}

impl TokenDistributionAnalyzer {
    /// Create a new token distribution analyzer with default bins.
    pub fn new() -> Self {
        Self {
            length_bins: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50],
            frequency_bins: vec![1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
        }
    }

    /// Create a custom analyzer with specified bins.
    pub fn with_bins(length_bins: Vec<usize>, frequency_bins: Vec<usize>) -> Self {
        Self {
            length_bins,
            frequency_bins,
        }
    }

    /// Analyze token distribution from a vocabulary mapping.
    pub fn analyze_distribution(&self, vocab: &HashMap<String, u32>) -> TokenDistributionResult {
        let mut length_distribution = HashMap::new();
        let mut frequency_distribution = HashMap::new();
        let mut character_usage = HashMap::new();
        let mut prefix_analysis = HashMap::new();
        let mut suffix_analysis = HashMap::new();

        // Initialize bins
        for &bin in &self.length_bins {
            length_distribution.insert(bin, 0);
        }
        for &bin in &self.frequency_bins {
            frequency_distribution.insert(bin, 0);
        }

        let mut total_tokens = 0;
        let mut total_length = 0;
        let mut max_length = 0;
        let mut min_length = usize::MAX;

        for (token, &freq) in vocab {
            let length = token.chars().count();
            total_tokens += 1;
            total_length += length;
            max_length = max_length.max(length);
            min_length = min_length.min(length);

            // Length distribution
            for &bin in &self.length_bins {
                if length <= bin {
                    *length_distribution.entry(bin).or_insert(0) += 1;
                    break;
                }
            }

            // Frequency distribution (using token frequency)
            let token_freq = freq as usize;
            for &bin in &self.frequency_bins {
                if token_freq <= bin {
                    *frequency_distribution.entry(bin).or_insert(0) += 1;
                    break;
                }
            }

            // Character usage analysis
            for ch in token.chars() {
                *character_usage.entry(ch).or_insert(0) += 1;
            }

            // Prefix/suffix analysis (2-3 character prefixes/suffixes)
            if length >= 2 {
                let prefix2: String = token.chars().take(2).collect();
                *prefix_analysis.entry(prefix2).or_insert(0) += 1;

                let suffix2: String =
                    token.chars().rev().take(2).collect::<String>().chars().rev().collect();
                *suffix_analysis.entry(suffix2).or_insert(0) += 1;
            }
            if length >= 3 {
                let prefix3: String = token.chars().take(3).collect();
                *prefix_analysis.entry(prefix3).or_insert(0) += 1;

                let suffix3: String =
                    token.chars().rev().take(3).collect::<String>().chars().rev().collect();
                *suffix_analysis.entry(suffix3).or_insert(0) += 1;
            }
        }

        let average_length =
            if total_tokens > 0 { total_length as f64 / total_tokens as f64 } else { 0.0 };

        // Sort character usage by frequency
        let mut char_frequency: Vec<_> = character_usage.into_iter().collect();
        char_frequency.sort_by(|a, b| b.1.cmp(&a.1));

        // Get top prefixes and suffixes
        let mut prefix_frequency: Vec<_> = prefix_analysis.into_iter().collect();
        prefix_frequency.sort_by(|a, b| b.1.cmp(&a.1));
        prefix_frequency.truncate(20); // Top 20

        let mut suffix_frequency: Vec<_> = suffix_analysis.into_iter().collect();
        suffix_frequency.sort_by(|a, b| b.1.cmp(&a.1));
        suffix_frequency.truncate(20); // Top 20

        TokenDistributionResult {
            total_tokens,
            average_length,
            max_length,
            min_length: if min_length == usize::MAX { 0 } else { min_length },
            length_distribution,
            frequency_distribution,
            character_frequency: char_frequency.into_iter().collect(),
            prefix_frequency: prefix_frequency.into_iter().collect(),
            suffix_frequency: suffix_frequency.into_iter().collect(),
        }
    }
}

impl Default for TokenDistributionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Token distribution analysis result with comprehensive statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenDistributionResult {
    /// Total number of tokens analyzed
    pub total_tokens: usize,
    /// Average token length in characters
    pub average_length: f64,
    /// Maximum token length
    pub max_length: usize,
    /// Minimum token length
    pub min_length: usize,
    /// Distribution of tokens by length bins
    pub length_distribution: HashMap<usize, usize>,
    /// Distribution of tokens by frequency bins
    pub frequency_distribution: HashMap<usize, usize>,
    /// Character usage frequency
    pub character_frequency: HashMap<char, usize>,
    /// Prefix frequency analysis
    pub prefix_frequency: HashMap<String, usize>,
    /// Suffix frequency analysis
    pub suffix_frequency: HashMap<String, usize>,
}

impl TokenDistributionResult {
    /// Generate a human-readable analysis report.
    pub fn generate_report(&self) -> String {
        format!(
            "Token Distribution Analysis Report\n\
             ===================================\n\
             Total Tokens: {}\n\
             Average Token Length: {:.2}\n\
             Min/Max Length: {}/{}\n\
             \n\
             Length Distribution:\n\
             {}\n\
             \n\
             Top 10 Characters by Frequency:\n\
             {}\n\
             \n\
             Top 10 Prefixes:\n\
             {}\n\
             \n\
             Top 10 Suffixes:\n\
             {}",
            self.total_tokens,
            self.average_length,
            self.min_length,
            self.max_length,
            self.format_length_distribution(),
            self.format_character_frequency(),
            self.format_prefix_frequency(),
            self.format_suffix_frequency()
        )
    }

    /// Format length distribution for display.
    fn format_length_distribution(&self) -> String {
        let mut sorted: Vec<_> = self.length_distribution.iter().collect();
        sorted.sort_by_key(|(len, _)| *len);

        sorted
            .iter()
            .map(|(len, count)| format!("  Length ≤{}: {} tokens", len, count))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Format character frequency for display.
    fn format_character_frequency(&self) -> String {
        self.character_frequency
            .iter()
            .take(10)
            .map(|(ch, count)| format!("  '{}': {} occurrences", ch, count))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Format prefix frequency for display.
    fn format_prefix_frequency(&self) -> String {
        self.prefix_frequency
            .iter()
            .take(10)
            .map(|(prefix, count)| format!("  '{}': {} tokens", prefix, count))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Format suffix frequency for display.
    fn format_suffix_frequency(&self) -> String {
        self.suffix_frequency
            .iter()
            .take(10)
            .map(|(suffix, count)| format!("  '{}': {} tokens", suffix, count))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Calculate vocabulary diversity score.
    pub fn diversity_score(&self) -> f64 {
        if self.total_tokens == 0 {
            return 0.0;
        }

        // Calculate entropy-based diversity score
        let mut entropy = 0.0;
        let total_chars: usize = self.character_frequency.values().sum();

        if total_chars > 0 {
            for &freq in self.character_frequency.values() {
                let prob = freq as f64 / total_chars as f64;
                if prob > 0.0 {
                    entropy -= prob * prob.log2();
                }
            }
        }

        entropy
    }

    /// Get optimal length range based on the distribution.
    pub fn optimal_length_range(&self) -> (usize, usize) {
        let mut cumulative = 0;
        let target_coverage = (self.total_tokens as f64 * 0.8) as usize; // 80% coverage

        let mut sorted: Vec<_> = self.length_distribution.iter().collect();
        sorted.sort_by_key(|(len, _)| *len);

        let min_optimal = self.min_length;
        let mut max_optimal = self.max_length;

        for (len, count) in sorted {
            cumulative += count;
            if cumulative >= target_coverage {
                max_optimal = *len;
                break;
            }
        }

        (min_optimal, max_optimal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coverage_analysis() {
        let analysis = CoverageAnalysis {
            char_coverage_rate: 0.95,
            word_coverage_rate: 0.88,
            compression_ratio: 0.6,
            total_chars: 1000,
            total_words: 200,
            total_tokens: 600,
            covered_chars: 950,
            covered_words: 176,
            length_distribution: {
                let mut dist = HashMap::new();
                dist.insert(1, 100);
                dist.insert(2, 200);
                dist.insert(3, 150);
                dist.insert(4, 100);
                dist.insert(5, 50);
                dist
            },
            oov_tokens: vec!["[UNK]".to_string()],
            vocab_size: 5000,
        };

        assert!((analysis.average_token_length() - 2.6666666666666665).abs() < 1e-10);
        assert!(analysis.efficiency_score() > 0.0);

        let summary = analysis.summary();
        assert!(summary.contains("95.00%"));
        assert!(summary.contains("88.00%"));

        let top_lengths = analysis.top_token_lengths(3);
        assert_eq!(top_lengths[0], (2, 200));
    }

    #[test]
    fn test_language_detector() {
        let detector = LanguageDetector::new();

        assert_eq!(detector.supported_languages().len(), 4);
        assert!(detector.supported_languages().contains(&"en".to_string()));

        let result = detector.detect_language("Hello world, this is a test in English");
        assert_eq!(result.detected_language, "en");
        assert!(result.confidence > 0.0);

        let top_candidates = result.top_candidates(2);
        assert_eq!(top_candidates[0].0, "en");

        assert!(result.is_confident(0.1));
    }

    #[test]
    fn test_language_detector_empty_text() {
        let detector = LanguageDetector::new();
        let result = detector.detect_language("");
        assert_eq!(result.detected_language, "unknown");
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_token_distribution_analyzer() {
        let analyzer = TokenDistributionAnalyzer::new();

        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), 1);
        vocab.insert("the".to_string(), 2);
        vocab.insert("hello".to_string(), 3);
        vocab.insert("world".to_string(), 4);
        vocab.insert("test".to_string(), 5);

        let result = analyzer.analyze_distribution(&vocab);

        assert_eq!(result.total_tokens, 5);
        assert!(result.average_length > 0.0);
        assert_eq!(result.min_length, 1);
        assert_eq!(result.max_length, 5);

        let report = result.generate_report();
        assert!(report.contains("Token Distribution Analysis Report"));
        assert!(report.contains("Total Tokens: 5"));

        assert!(result.diversity_score() > 0.0);

        let (min_opt, max_opt) = result.optimal_length_range();
        assert!(min_opt <= max_opt);
    }

    #[test]
    fn test_custom_language_profile() {
        let mut detector = LanguageDetector::new();

        let mut custom_chars = HashMap::new();
        custom_chars.insert('x', 50.0);
        custom_chars.insert('y', 40.0);
        custom_chars.insert('z', 30.0);

        let mut custom_ngrams = HashMap::new();
        custom_ngrams.insert("xy".to_string(), 20.0);
        custom_ngrams.insert("yz".to_string(), 15.0);
        custom_ngrams.insert("xz".to_string(), 10.0);

        detector.add_language_profile("custom".to_string(), custom_chars, custom_ngrams);

        assert!(detector.supported_languages().contains(&"custom".to_string()));

        // Use a text with very high density of custom characters
        let result = detector.detect_language("xyxyxyxyxyzyzyzyzxzxzxzxz");
        assert_eq!(result.detected_language, "custom");
    }

    #[test]
    fn test_distribution_analyzer_custom_bins() {
        let analyzer = TokenDistributionAnalyzer::with_bins(vec![1, 3, 5], vec![1, 10, 100]);

        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), 1);
        vocab.insert("abc".to_string(), 15);
        vocab.insert("abcde".to_string(), 150);

        let result = analyzer.analyze_distribution(&vocab);
        assert_eq!(result.total_tokens, 3);
    }
}
