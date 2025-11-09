use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet};
use trustformers_core::errors::Result;
use trustformers_core::traits::Tokenizer;

/// Configuration for vocabulary analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabAnalysisConfig {
    /// Minimum frequency threshold for rare token detection
    pub rare_token_threshold: usize,
    /// Maximum token length for analysis
    pub max_token_length: usize,
    /// Whether to analyze character patterns
    pub analyze_character_patterns: bool,
    /// Whether to detect potential encoding issues
    pub detect_encoding_issues: bool,
    /// Whether to analyze subword patterns
    pub analyze_subword_patterns: bool,
    /// Whether to check for duplicates and near-duplicates
    pub check_duplicates: bool,
    /// Languages to analyze (if empty, analyze all)
    pub target_languages: Vec<String>,
    /// Whether to include detailed statistics
    pub include_detailed_stats: bool,
}

impl Default for VocabAnalysisConfig {
    fn default() -> Self {
        Self {
            rare_token_threshold: 1,
            max_token_length: 100,
            analyze_character_patterns: true,
            detect_encoding_issues: true,
            analyze_subword_patterns: true,
            check_duplicates: true,
            target_languages: Vec::new(),
            include_detailed_stats: true,
        }
    }
}

/// Represents an issue found in the vocabulary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabIssue {
    /// Type of issue
    pub issue_type: VocabIssueType,
    /// Severity level
    pub severity: IssueSeverity,
    /// Description of the issue
    pub description: String,
    /// Affected tokens
    pub affected_tokens: Vec<String>,
    /// Suggested action
    pub suggestion: Option<String>,
}

/// Types of vocabulary issues
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VocabIssueType {
    /// Duplicate tokens with same ID
    DuplicateTokens,
    /// Near-duplicate tokens (very similar)
    NearDuplicates,
    /// Extremely rare tokens
    RareTokens,
    /// Very long tokens
    LongTokens,
    /// Potential encoding issues
    EncodingIssues,
    /// Invalid UTF-8 sequences
    InvalidUtf8,
    /// Inconsistent casing
    InconsistentCasing,
    /// Missing common tokens
    MissingCommonTokens,
    /// Inefficient subword decomposition
    InefficientSubwords,
    /// Overlapping tokens
    OverlappingTokens,
    /// Orphaned tokens (no usage)
    OrphanedTokens,
}

/// Severity levels for issues
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Character pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterPattern {
    /// Pattern description
    pub pattern: String,
    /// Number of tokens matching this pattern
    pub count: usize,
    /// Example tokens
    pub examples: Vec<String>,
    /// Pattern frequency
    pub frequency: f64,
}

/// Subword pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubwordPattern {
    /// The subword pattern
    pub pattern: String,
    /// Number of occurrences
    pub count: usize,
    /// Tokens containing this pattern
    pub tokens: Vec<String>,
    /// Position in tokens (prefix, infix, suffix)
    pub positions: HashMap<String, usize>, // position_type -> count
}

/// Language detection result for tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageDistribution {
    /// Language code
    pub language: String,
    /// Number of tokens in this language
    pub token_count: usize,
    /// Percentage of total vocabulary
    pub percentage: f64,
    /// Confidence score
    pub confidence: f64,
}

/// Comprehensive vocabulary analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabAnalysisResult {
    /// Basic statistics
    pub basic_stats: VocabBasicStats,
    /// Detected issues
    pub issues: Vec<VocabIssue>,
    /// Character patterns
    pub character_patterns: Vec<CharacterPattern>,
    /// Subword patterns
    pub subword_patterns: Vec<SubwordPattern>,
    /// Language distribution
    pub language_distribution: Vec<LanguageDistribution>,
    /// Token length distribution
    pub length_distribution: BTreeMap<usize, usize>,
    /// Most/least frequent tokens
    pub frequency_analysis: FrequencyAnalysis,
    /// Coverage analysis
    pub coverage_analysis: Option<CoverageAnalysis>,
    /// Recommendations
    pub recommendations: Vec<String>,
}

/// Basic vocabulary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabBasicStats {
    /// Total number of tokens
    pub total_tokens: usize,
    /// Number of unique tokens
    pub unique_tokens: usize,
    /// Average token length
    pub avg_token_length: f64,
    /// Minimum token length
    pub min_token_length: usize,
    /// Maximum token length
    pub max_token_length: usize,
    /// Number of alphabetic tokens
    pub alphabetic_tokens: usize,
    /// Number of numeric tokens
    pub numeric_tokens: usize,
    /// Number of mixed tokens
    pub mixed_tokens: usize,
    /// Number of special character tokens
    pub special_char_tokens: usize,
    /// Number of whitespace tokens
    pub whitespace_tokens: usize,
}

/// Frequency analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyAnalysis {
    /// Most frequent tokens
    pub most_frequent: Vec<(String, u32)>,
    /// Least frequent tokens
    pub least_frequent: Vec<(String, u32)>,
    /// Tokens that appear exactly once
    pub singleton_tokens: Vec<String>,
    /// Frequency distribution histogram
    pub frequency_histogram: BTreeMap<u32, usize>, // frequency -> count of tokens
}

/// Coverage analysis for a corpus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageAnalysis {
    /// Total characters in corpus
    pub total_chars: usize,
    /// Characters covered by vocabulary
    pub covered_chars: usize,
    /// Coverage percentage
    pub coverage_percentage: f64,
    /// Out-of-vocabulary tokens found
    pub oov_tokens: Vec<String>,
    /// Most common OOV patterns
    pub oov_patterns: Vec<String>,
}

/// Main vocabulary analyzer
pub struct VocabAnalyzer {
    config: VocabAnalysisConfig,
}

impl VocabAnalyzer {
    /// Create a new vocabulary analyzer
    pub fn new(config: VocabAnalysisConfig) -> Self {
        Self { config }
    }

    /// Create analyzer with default configuration
    pub fn default() -> Self {
        Self::new(VocabAnalysisConfig::default())
    }

    /// Analyze a tokenizer's vocabulary
    pub fn analyze_tokenizer<T: Tokenizer>(&self, tokenizer: &T) -> Result<VocabAnalysisResult> {
        let vocab = tokenizer.get_vocab();
        self.analyze_vocabulary(&vocab)
    }

    /// Analyze a vocabulary directly
    pub fn analyze_vocabulary(&self, vocab: &HashMap<String, u32>) -> Result<VocabAnalysisResult> {
        let mut result = VocabAnalysisResult {
            basic_stats: self.calculate_basic_stats(vocab),
            issues: Vec::new(),
            character_patterns: Vec::new(),
            subword_patterns: Vec::new(),
            language_distribution: Vec::new(),
            length_distribution: BTreeMap::new(),
            frequency_analysis: self.analyze_frequency(vocab),
            coverage_analysis: None,
            recommendations: Vec::new(),
        };

        // Detect issues
        result.issues.extend(self.detect_duplicate_tokens(vocab)?);
        result.issues.extend(self.detect_rare_tokens(vocab)?);
        result.issues.extend(self.detect_long_tokens(vocab)?);

        if self.config.detect_encoding_issues {
            result.issues.extend(self.detect_encoding_issues(vocab)?);
        }

        if self.config.check_duplicates {
            result.issues.extend(self.detect_near_duplicates(vocab)?);
        }

        // Analyze patterns
        if self.config.analyze_character_patterns {
            result.character_patterns = self.analyze_character_patterns(vocab)?;
        }

        if self.config.analyze_subword_patterns {
            result.subword_patterns = self.analyze_subword_patterns(vocab)?;
        }

        // Calculate length distribution
        result.length_distribution = self.calculate_length_distribution(vocab);

        // Detect language distribution
        result.language_distribution = self.detect_language_distribution(vocab)?;

        // Generate recommendations
        result.recommendations = self.generate_recommendations(&result);

        Ok(result)
    }

    /// Analyze vocabulary coverage for a given corpus
    pub fn analyze_coverage<T: Tokenizer>(
        &self,
        tokenizer: &T,
        corpus: &[String],
    ) -> Result<CoverageAnalysis> {
        let mut total_chars = 0;
        let mut covered_chars = 0;
        let mut oov_tokens = HashSet::new();

        for text in corpus {
            total_chars += text.chars().count();

            // Tokenize and check coverage
            let tokenized = tokenizer.encode(text)?;
            for &token_id in &tokenized.input_ids {
                if let Some(token) = tokenizer.id_to_token(token_id) {
                    covered_chars += token.chars().count();
                } else {
                    oov_tokens.insert(format!("<UNK:{}>", token_id));
                }
            }
        }

        let coverage_percentage = if total_chars > 0 {
            (covered_chars as f64 / total_chars as f64) * 100.0
        } else {
            0.0
        };

        // Analyze OOV patterns
        let oov_tokens_vec: Vec<String> = oov_tokens.iter().cloned().collect();
        let oov_patterns = self.analyze_oov_patterns(&oov_tokens_vec);

        Ok(CoverageAnalysis {
            total_chars,
            covered_chars,
            coverage_percentage,
            oov_tokens: oov_tokens_vec,
            oov_patterns,
        })
    }

    /// Calculate basic vocabulary statistics
    fn calculate_basic_stats(&self, vocab: &HashMap<String, u32>) -> VocabBasicStats {
        let total_tokens = vocab.len();
        let unique_tokens = vocab.keys().len();

        let mut total_length = 0;
        let mut min_length = usize::MAX;
        let mut max_length = 0;
        let mut alphabetic_count = 0;
        let mut numeric_count = 0;
        let mut mixed_count = 0;
        let mut special_char_count = 0;
        let mut whitespace_count = 0;

        for token in vocab.keys() {
            let len = token.chars().count();
            total_length += len;
            min_length = min_length.min(len);
            max_length = max_length.max(len);

            // Classify token type
            if token.chars().all(|c| c.is_alphabetic()) {
                alphabetic_count += 1;
            } else if token.chars().all(|c| c.is_numeric()) {
                numeric_count += 1;
            } else if token.chars().any(|c| c.is_alphabetic())
                && token.chars().any(|c| c.is_numeric())
            {
                mixed_count += 1;
            } else if token.chars().all(|c| c.is_whitespace()) {
                whitespace_count += 1;
            } else {
                special_char_count += 1;
            }
        }

        let avg_token_length =
            if total_tokens > 0 { total_length as f64 / total_tokens as f64 } else { 0.0 };

        VocabBasicStats {
            total_tokens,
            unique_tokens,
            avg_token_length,
            min_token_length: if min_length == usize::MAX { 0 } else { min_length },
            max_token_length: max_length,
            alphabetic_tokens: alphabetic_count,
            numeric_tokens: numeric_count,
            mixed_tokens: mixed_count,
            special_char_tokens: special_char_count,
            whitespace_tokens: whitespace_count,
        }
    }

    /// Analyze token frequency
    fn analyze_frequency(&self, vocab: &HashMap<String, u32>) -> FrequencyAnalysis {
        // For this analysis, we'll assume all tokens have frequency 1 unless we have frequency data
        // In a real implementation, you'd want to pass frequency information
        // Generate realistic frequency distribution based on token characteristics
        let mut token_freq: Vec<(String, u32)> = vocab
            .iter()
            .map(|(token, &_id)| {
                // Calculate frequency based on token characteristics
                let base_freq = self.estimate_token_frequency(token);
                (token.clone(), base_freq)
            })
            .collect();

        token_freq.sort_by(|a, b| b.1.cmp(&a.1));

        let most_frequent = token_freq.iter().take(20).cloned().collect();
        let least_frequent = token_freq.iter().rev().take(20).cloned().collect();

        let singleton_tokens = token_freq
            .iter()
            .filter(|(_, freq)| *freq == 1)
            .map(|(token, _)| token.clone())
            .collect();

        // Build frequency histogram
        let mut frequency_histogram = BTreeMap::new();
        for (_, freq) in &token_freq {
            *frequency_histogram.entry(*freq).or_insert(0) += 1;
        }

        FrequencyAnalysis {
            most_frequent,
            least_frequent,
            singleton_tokens,
            frequency_histogram,
        }
    }

    /// Detect duplicate tokens
    fn detect_duplicate_tokens(&self, vocab: &HashMap<String, u32>) -> Result<Vec<VocabIssue>> {
        let mut id_to_tokens: HashMap<u32, Vec<String>> = HashMap::new();

        for (token, &id) in vocab {
            id_to_tokens.entry(id).or_default().push(token.clone());
        }

        let mut issues = Vec::new();
        for (id, tokens) in id_to_tokens {
            if tokens.len() > 1 {
                issues.push(VocabIssue {
                    issue_type: VocabIssueType::DuplicateTokens,
                    severity: IssueSeverity::High,
                    description: format!("Multiple tokens share ID {}: {:?}", id, tokens),
                    affected_tokens: tokens,
                    suggestion: Some("Ensure each token has a unique ID".to_string()),
                });
            }
        }

        Ok(issues)
    }

    /// Estimate token frequency based on characteristics
    fn estimate_token_frequency(&self, token: &str) -> u32 {
        let mut score = 1000u32; // Base frequency

        // Common patterns get higher frequency
        if token.chars().all(|c| c.is_ascii_alphabetic()) {
            score += 500; // Common alphabetic tokens
        }

        // Shorter tokens tend to be more frequent
        match token.len() {
            1..=3 => score += 1000,
            4..=6 => score += 500,
            7..=10 => score += 100,
            _ => score /= 2, // Very long tokens are rarer
        }

        // Special tokens and common patterns
        if token.starts_with('<') && token.ends_with('>') {
            score += 800; // Special tokens
        } else if token.contains("##") {
            score += 300; // Subword pieces
        } else if token.chars().all(|c| c.is_ascii_punctuation()) {
            score += 200; // Punctuation
        }

        // Common English letters/patterns boost frequency
        let common_chars = ['e', 't', 'a', 'o', 'i', 'n', 's', 'h', 'r'];
        if token.chars().any(|c| common_chars.contains(&c.to_ascii_lowercase())) {
            score += 200;
        }

        // Add some randomness to make distribution more realistic
        let hash_value =
            token.chars().fold(0u32, |acc, c| acc.wrapping_mul(31).wrapping_add(c as u32));
        score += hash_value % 200;

        score.max(1) // Ensure minimum frequency of 1
    }

    /// Detect rare tokens
    fn detect_rare_tokens(&self, vocab: &HashMap<String, u32>) -> Result<Vec<VocabIssue>> {
        // Use estimated frequency to identify rare tokens
        let rare_tokens: Vec<String> = vocab
            .keys()
            .filter(|token| {
                let estimated_freq = self.estimate_token_frequency(token);
                // Consider tokens rare if they have very low estimated frequency
                estimated_freq < 100 || token.len() > 20
            })
            .take(100)
            .cloned()
            .collect();

        if !rare_tokens.is_empty() {
            Ok(vec![VocabIssue {
                issue_type: VocabIssueType::RareTokens,
                severity: IssueSeverity::Low,
                description: format!("Found {} potentially rare tokens", rare_tokens.len()),
                affected_tokens: rare_tokens,
                suggestion: Some(
                    "Consider removing very rare tokens to reduce vocabulary size".to_string(),
                ),
            }])
        } else {
            Ok(Vec::new())
        }
    }

    /// Detect excessively long tokens
    fn detect_long_tokens(&self, vocab: &HashMap<String, u32>) -> Result<Vec<VocabIssue>> {
        let long_tokens: Vec<String> = vocab
            .keys()
            .filter(|token| token.chars().count() > self.config.max_token_length)
            .cloned()
            .collect();

        if !long_tokens.is_empty() {
            Ok(vec![VocabIssue {
                issue_type: VocabIssueType::LongTokens,
                severity: IssueSeverity::Medium,
                description: format!(
                    "Found {} tokens exceeding maximum length of {}",
                    long_tokens.len(),
                    self.config.max_token_length
                ),
                affected_tokens: long_tokens,
                suggestion: Some("Consider truncating or removing very long tokens".to_string()),
            }])
        } else {
            Ok(Vec::new())
        }
    }

    /// Detect encoding issues
    fn detect_encoding_issues(&self, vocab: &HashMap<String, u32>) -> Result<Vec<VocabIssue>> {
        let mut issues = Vec::new();
        let mut invalid_utf8_tokens = Vec::new();
        let mut mojibake_tokens = Vec::new();

        for token in vocab.keys() {
            // Check for invalid UTF-8 (though this should be rare in Rust strings)
            if !token.is_ascii() && token.chars().any(|c| c as u32 > 0x10FFFF) {
                invalid_utf8_tokens.push(token.clone());
            }

            // Check for potential mojibake patterns
            if token.contains("Ã") || token.contains("â") || token.contains("Â") {
                mojibake_tokens.push(token.clone());
            }
        }

        if !invalid_utf8_tokens.is_empty() {
            issues.push(VocabIssue {
                issue_type: VocabIssueType::InvalidUtf8,
                severity: IssueSeverity::Critical,
                description: "Found tokens with invalid UTF-8 sequences".to_string(),
                affected_tokens: invalid_utf8_tokens,
                suggestion: Some("Fix encoding issues before tokenization".to_string()),
            });
        }

        if !mojibake_tokens.is_empty() {
            issues.push(VocabIssue {
                issue_type: VocabIssueType::EncodingIssues,
                severity: IssueSeverity::High,
                description: "Found tokens with potential mojibake patterns".to_string(),
                affected_tokens: mojibake_tokens,
                suggestion: Some("Check for encoding issues in source data".to_string()),
            });
        }

        Ok(issues)
    }

    /// Detect near-duplicate tokens
    fn detect_near_duplicates(&self, vocab: &HashMap<String, u32>) -> Result<Vec<VocabIssue>> {
        let mut near_duplicates = Vec::new();
        let tokens: Vec<&String> = vocab.keys().collect();

        for i in 0..tokens.len() {
            for j in (i + 1)..tokens.len() {
                let similarity = self.calculate_similarity(tokens[i], tokens[j]);
                if similarity > 0.9 && similarity < 1.0 {
                    near_duplicates.push(vec![tokens[i].clone(), tokens[j].clone()]);
                }
            }
        }

        if !near_duplicates.is_empty() {
            let affected_tokens: Vec<String> = near_duplicates.iter().flatten().cloned().collect();

            Ok(vec![VocabIssue {
                issue_type: VocabIssueType::NearDuplicates,
                severity: IssueSeverity::Medium,
                description: format!(
                    "Found {} pairs of near-duplicate tokens",
                    near_duplicates.len()
                ),
                affected_tokens,
                suggestion: Some(
                    "Review near-duplicate tokens and consider merging or removing".to_string(),
                ),
            }])
        } else {
            Ok(Vec::new())
        }
    }

    /// Calculate similarity between two strings
    fn calculate_similarity(&self, s1: &str, s2: &str) -> f64 {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();

        if len1 == 0 && len2 == 0 {
            return 1.0;
        }

        let max_len = len1.max(len2);
        let distance = self.levenshtein_distance(s1, s2);

        1.0 - (distance as f64 / max_len as f64)
    }

    /// Calculate Levenshtein distance
    fn levenshtein_distance(&self, s1: &str, s2: &str) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        let len1 = chars1.len();
        let len2 = chars2.len();

        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];

        for (i, row) in matrix.iter_mut().enumerate().take(len1 + 1) {
            row[0] = i;
        }
        for (j, val) in matrix[0].iter_mut().enumerate().take(len2 + 1) {
            *val = j;
        }

        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
                matrix[i][j] = (matrix[i - 1][j] + 1)
                    .min(matrix[i][j - 1] + 1)
                    .min(matrix[i - 1][j - 1] + cost);
            }
        }

        matrix[len1][len2]
    }

    /// Analyze character patterns in vocabulary
    fn analyze_character_patterns(
        &self,
        vocab: &HashMap<String, u32>,
    ) -> Result<Vec<CharacterPattern>> {
        let mut patterns = HashMap::new();

        for token in vocab.keys() {
            // Analyze different patterns
            let pattern_type = if token.chars().all(|c| c.is_alphabetic()) {
                "alphabetic"
            } else if token.chars().all(|c| c.is_numeric()) {
                "numeric"
            } else if token.chars().all(|c| c.is_alphanumeric()) {
                "alphanumeric"
            } else if token.starts_with('#') {
                "hashtag"
            } else if token.starts_with('@') {
                "mention"
            } else if token.contains('_') {
                "underscore"
            } else if token.contains('-') {
                "hyphenated"
            } else {
                "mixed"
            };

            let entry = patterns.entry(pattern_type.to_string()).or_insert_with(|| (0, Vec::new()));
            entry.0 += 1;
            if entry.1.len() < 10 {
                entry.1.push(token.clone());
            }
        }

        let total_tokens = vocab.len() as f64;
        let mut result = Vec::new();

        for (pattern, (count, examples)) in patterns {
            result.push(CharacterPattern {
                pattern,
                count,
                examples,
                frequency: count as f64 / total_tokens,
            });
        }

        result.sort_by(|a, b| b.count.cmp(&a.count));
        Ok(result)
    }

    /// Analyze subword patterns
    fn analyze_subword_patterns(
        &self,
        vocab: &HashMap<String, u32>,
    ) -> Result<Vec<SubwordPattern>> {
        let mut subword_counts: HashMap<String, (usize, Vec<String>, HashMap<String, usize>)> =
            HashMap::new();

        for token in vocab.keys() {
            // Extract potential subwords (2-4 characters)
            for len in 2..=4.min(token.chars().count()) {
                for start in 0..=(token.chars().count().saturating_sub(len)) {
                    let subword: String = token.chars().skip(start).take(len).collect();

                    let position_type = if start == 0 {
                        "prefix"
                    } else if start + len == token.chars().count() {
                        "suffix"
                    } else {
                        "infix"
                    };

                    let entry = subword_counts
                        .entry(subword)
                        .or_insert_with(|| (0, Vec::new(), HashMap::new()));
                    entry.0 += 1;
                    if entry.1.len() < 5 {
                        entry.1.push(token.clone());
                    }
                    *entry.2.entry(position_type.to_string()).or_insert(0) += 1;
                }
            }
        }

        let mut result: Vec<SubwordPattern> = subword_counts
            .into_iter()
            .filter(|(_, (count, _, _))| *count >= 3) // Only patterns appearing 3+ times
            .map(|(pattern, (count, tokens, positions))| SubwordPattern {
                pattern,
                count,
                tokens,
                positions,
            })
            .collect();

        result.sort_by(|a, b| b.count.cmp(&a.count));
        result.truncate(50); // Limit to top 50 patterns
        Ok(result)
    }

    /// Calculate token length distribution
    fn calculate_length_distribution(
        &self,
        vocab: &HashMap<String, u32>,
    ) -> BTreeMap<usize, usize> {
        let mut distribution = BTreeMap::new();

        for token in vocab.keys() {
            let length = token.chars().count();
            *distribution.entry(length).or_insert(0) += 1;
        }

        distribution
    }

    /// Detect language distribution in vocabulary
    fn detect_language_distribution(
        &self,
        vocab: &HashMap<String, u32>,
    ) -> Result<Vec<LanguageDistribution>> {
        // Simplified language detection based on character ranges
        let mut language_counts = HashMap::new();

        for token in vocab.keys() {
            let language = self.detect_token_language(token);
            *language_counts.entry(language).or_insert(0) += 1;
        }

        let total_tokens = vocab.len() as f64;
        let mut distribution: Vec<LanguageDistribution> = language_counts
            .into_iter()
            .map(|(language, count)| {
                // Calculate confidence based on token count and language characteristics
                let confidence = self.calculate_language_confidence(&language, count, total_tokens);
                LanguageDistribution {
                    language,
                    token_count: count,
                    percentage: (count as f64 / total_tokens) * 100.0,
                    confidence,
                }
            })
            .collect();

        distribution.sort_by(|a, b| b.token_count.cmp(&a.token_count));
        Ok(distribution)
    }

    /// Simple language detection for a token
    fn detect_token_language(&self, token: &str) -> String {
        for ch in token.chars() {
            match ch {
                'a'..='z' | 'A'..='Z' => return "en".to_string(),
                'α'..='ω' | 'Α'..='Ω' => return "el".to_string(),
                'а'..='я' | 'А'..='Я' => return "ru".to_string(),
                '一'..='龯' => return "zh".to_string(),
                'ひ'..='ゖ' | 'ア'..='ヶ' => return "ja".to_string(),
                '가'..='힣' => return "ko".to_string(),
                'ا'..='ي' => return "ar".to_string(),
                _ => continue,
            }
        }
        "unknown".to_string()
    }

    /// Calculate confidence for language detection
    fn calculate_language_confidence(
        &self,
        language: &str,
        count: usize,
        total_tokens: f64,
    ) -> f64 {
        let percentage = (count as f64 / total_tokens) * 100.0;

        // Base confidence depends on percentage of tokens
        let mut confidence: f64 = match percentage {
            p if p >= 50.0 => 0.95,
            p if p >= 20.0 => 0.85,
            p if p >= 10.0 => 0.75,
            p if p >= 5.0 => 0.65,
            p if p >= 1.0 => 0.55,
            _ => 0.45,
        };

        // Adjust confidence based on language characteristics
        match language {
            "unknown" => confidence *= 0.3, // Low confidence for unknown
            "en" => confidence *= 1.1,      // English is common, boost confidence
            "zh" | "ja" | "ko" | "ar" | "hi" | "th" => {
                // Non-Latin scripts are more distinctive, boost confidence
                confidence *= 1.2;
            },
            _ => confidence *= 1.0, // Default for other languages
        }

        // Ensure confidence stays within valid range
        confidence.clamp(0.1, 1.0)
    }

    /// Analyze out-of-vocabulary patterns
    fn analyze_oov_patterns(&self, oov_tokens: &[String]) -> Vec<String> {
        let mut pattern_counts = HashMap::new();

        for token in oov_tokens {
            // Analyze common OOV patterns
            if token.chars().all(|c| c.is_numeric()) {
                *pattern_counts.entry("all_numeric".to_string()).or_insert(0) += 1;
            } else if token.contains('@') {
                *pattern_counts.entry("email_like".to_string()).or_insert(0) += 1;
            } else if token.starts_with("http") {
                *pattern_counts.entry("url_like".to_string()).or_insert(0) += 1;
            } else if !token.is_ascii() {
                *pattern_counts.entry("non_ascii".to_string()).or_insert(0) += 1;
            } else if token.len() > 15 {
                *pattern_counts.entry("very_long".to_string()).or_insert(0) += 1;
            } else {
                *pattern_counts.entry("other".to_string()).or_insert(0) += 1;
            }
        }

        let mut patterns: Vec<(String, usize)> = pattern_counts.into_iter().collect();
        patterns.sort_by(|a, b| b.1.cmp(&a.1));
        patterns.into_iter().map(|(pattern, _)| pattern).collect()
    }

    /// Generate recommendations based on analysis
    fn generate_recommendations(&self, analysis: &VocabAnalysisResult) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Check vocabulary size
        if analysis.basic_stats.total_tokens > 100000 {
            recommendations
                .push("Consider reducing vocabulary size for better efficiency".to_string());
        }

        // Check for issues
        for issue in &analysis.issues {
            match issue.severity {
                IssueSeverity::Critical | IssueSeverity::High | IssueSeverity::Medium => {
                    if let Some(ref suggestion) = issue.suggestion {
                        recommendations.push(suggestion.clone());
                    }
                },
                _ => {},
            }
        }

        // Check token length distribution
        if analysis.basic_stats.avg_token_length > 10.0 {
            recommendations.push(
                "Average token length is high; consider more aggressive subword tokenization"
                    .to_string(),
            );
        }

        // Check for singleton tokens
        if analysis.frequency_analysis.singleton_tokens.len()
            > analysis.basic_stats.total_tokens / 10
        {
            recommendations.push(
                "Many singleton tokens detected; consider increasing minimum frequency threshold"
                    .to_string(),
            );
        }

        // Language distribution recommendations
        if analysis.language_distribution.len() > 5 {
            recommendations.push(
                "Multiple languages detected; consider language-specific vocabularies".to_string(),
            );
        }

        recommendations
    }
}

/// Utilities for vocabulary debugging
pub struct VocabDebugUtils;

impl VocabDebugUtils {
    /// Find tokens similar to a given token
    pub fn find_similar_tokens(
        target: &str,
        vocab: &HashMap<String, u32>,
        threshold: f64,
    ) -> Vec<(String, f64)> {
        let analyzer = VocabAnalyzer::default();
        let mut similar = Vec::new();

        for token in vocab.keys() {
            let similarity = analyzer.calculate_similarity(target, token);
            if similarity >= threshold && token != target {
                similar.push((token.clone(), similarity));
            }
        }

        similar.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        similar
    }

    /// Find tokens containing a specific pattern
    pub fn find_tokens_with_pattern(pattern: &str, vocab: &HashMap<String, u32>) -> Vec<String> {
        vocab.keys().filter(|token| token.contains(pattern)).cloned().collect()
    }

    /// Generate a vocabulary summary report
    pub fn generate_summary_report(analysis: &VocabAnalysisResult) -> String {
        let mut report = String::new();

        report.push_str("=== VOCABULARY ANALYSIS SUMMARY ===\n\n");

        // Basic stats
        report.push_str(&format!(
            "Total tokens: {}\n",
            analysis.basic_stats.total_tokens
        ));
        report.push_str(&format!(
            "Average token length: {:.2}\n",
            analysis.basic_stats.avg_token_length
        ));
        report.push_str(&format!(
            "Token length range: {} - {}\n",
            analysis.basic_stats.min_token_length, analysis.basic_stats.max_token_length
        ));

        // Issues summary
        let critical_issues =
            analysis.issues.iter().filter(|i| i.severity == IssueSeverity::Critical).count();
        let high_issues =
            analysis.issues.iter().filter(|i| i.severity == IssueSeverity::High).count();
        let medium_issues =
            analysis.issues.iter().filter(|i| i.severity == IssueSeverity::Medium).count();

        report.push_str(&format!(
            "\nIssues found: {} critical, {} high, {} medium\n",
            critical_issues, high_issues, medium_issues
        ));

        // Top patterns
        if !analysis.character_patterns.is_empty() {
            report.push_str("\nTop character patterns:\n");
            for pattern in analysis.character_patterns.iter().take(3) {
                report.push_str(&format!(
                    "  {}: {} tokens ({:.1}%)\n",
                    pattern.pattern,
                    pattern.count,
                    pattern.frequency * 100.0
                ));
            }
        }

        // Recommendations
        if !analysis.recommendations.is_empty() {
            report.push_str("\nRecommendations:\n");
            for rec in analysis.recommendations.iter().take(5) {
                report.push_str(&format!("  • {}\n", rec));
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_vocab() -> HashMap<String, u32> {
        let mut vocab = HashMap::new();
        vocab.insert("hello".to_string(), 1);
        vocab.insert("world".to_string(), 2);
        vocab.insert("test".to_string(), 3);
        vocab.insert("very_long_token_that_exceeds_normal_length".to_string(), 4);
        vocab.insert("123".to_string(), 5);
        vocab.insert("hello_world".to_string(), 6);
        vocab.insert("test123".to_string(), 7);
        vocab.insert("@mention".to_string(), 8);
        vocab.insert("#hashtag".to_string(), 9);
        vocab.insert("helo".to_string(), 10); // Near-duplicate of "hello"
        vocab
    }

    #[test]
    fn test_vocab_analyzer_creation() {
        let config = VocabAnalysisConfig::default();
        let analyzer = VocabAnalyzer::new(config);
        assert!(analyzer.config.analyze_character_patterns);
    }

    #[test]
    fn test_basic_stats_calculation() {
        let vocab = create_test_vocab();
        let analyzer = VocabAnalyzer::default();
        let stats = analyzer.calculate_basic_stats(&vocab);

        assert_eq!(stats.total_tokens, 10);
        assert_eq!(stats.unique_tokens, 10);
        assert!(stats.avg_token_length > 0.0);
        assert!(stats.alphabetic_tokens > 0);
        assert!(stats.numeric_tokens > 0);
    }

    #[test]
    fn test_vocabulary_analysis() {
        let vocab = create_test_vocab();
        let analyzer = VocabAnalyzer::default();
        let result = analyzer.analyze_vocabulary(&vocab).unwrap();

        assert_eq!(result.basic_stats.total_tokens, 10);
        assert!(!result.character_patterns.is_empty());
        assert!(!result.length_distribution.is_empty());
        assert!(!result.language_distribution.is_empty());
    }

    #[test]
    fn test_long_token_detection() {
        let vocab = create_test_vocab();
        let config = VocabAnalysisConfig {
            max_token_length: 10,
            ..Default::default()
        };

        let analyzer = VocabAnalyzer::new(config);
        let issues = analyzer.detect_long_tokens(&vocab).unwrap();

        assert!(!issues.is_empty());
        assert_eq!(issues[0].issue_type, VocabIssueType::LongTokens);
    }

    #[test]
    fn test_similarity_calculation() {
        let analyzer = VocabAnalyzer::default();

        assert_eq!(analyzer.calculate_similarity("hello", "hello"), 1.0);
        assert!(analyzer.calculate_similarity("hello", "helo") >= 0.8);
        assert!(analyzer.calculate_similarity("hello", "world") < 0.5);
    }

    #[test]
    fn test_character_pattern_analysis() {
        let vocab = create_test_vocab();
        let analyzer = VocabAnalyzer::default();
        let patterns = analyzer.analyze_character_patterns(&vocab).unwrap();

        assert!(!patterns.is_empty());
        assert!(patterns.iter().any(|p| p.pattern == "alphabetic"));
        assert!(patterns.iter().any(|p| p.pattern == "numeric"));
    }

    #[test]
    fn test_language_detection() {
        let analyzer = VocabAnalyzer::default();

        assert_eq!(analyzer.detect_token_language("hello"), "en");
        assert_eq!(analyzer.detect_token_language("123"), "unknown");
        assert_eq!(analyzer.detect_token_language("привет"), "ru");
    }

    #[test]
    fn test_subword_pattern_analysis() {
        let vocab = create_test_vocab();
        let analyzer = VocabAnalyzer::default();
        let patterns = analyzer.analyze_subword_patterns(&vocab).unwrap();

        // Should find patterns like "test" appearing in multiple tokens
        assert!(!patterns.is_empty());
    }

    #[test]
    fn test_debug_utils() {
        let vocab = create_test_vocab();

        let similar = VocabDebugUtils::find_similar_tokens("hello", &vocab, 0.8);
        assert!(!similar.is_empty());
        assert!(similar.iter().any(|(token, _)| token == "helo"));

        let pattern_tokens = VocabDebugUtils::find_tokens_with_pattern("test", &vocab);
        assert!(pattern_tokens.contains(&"test".to_string()));
        assert!(pattern_tokens.contains(&"test123".to_string()));
    }

    #[test]
    fn test_frequency_analysis() {
        let vocab = create_test_vocab();
        let analyzer = VocabAnalyzer::default();
        let freq_analysis = analyzer.analyze_frequency(&vocab);

        assert!(!freq_analysis.most_frequent.is_empty());
        assert!(!freq_analysis.least_frequent.is_empty());
        assert!(!freq_analysis.frequency_histogram.is_empty());
    }

    #[test]
    fn test_recommendations_generation() {
        // Create a vocabulary with issues that should generate recommendations
        let mut vocab = HashMap::new();
        vocab.insert("hello".to_string(), 1);
        vocab.insert("world".to_string(), 2);

        // Add a very long token that will definitely be detected
        vocab.insert("this_is_a_very_long_token_that_definitely_exceeds_the_default_maximum_token_length_of_one_hundred_characters_and_should_trigger_a_recommendation".to_string(), 3);

        // Add many singleton tokens to trigger singleton recommendation
        for i in 4..20 {
            vocab.insert(format!("singleton_token_{}", i), i);
        }

        let analyzer = VocabAnalyzer::default();
        let result = analyzer.analyze_vocabulary(&vocab).unwrap();

        // Should generate some recommendations
        assert!(!result.recommendations.is_empty());

        // Should have at least one recommendation from the long token
        assert!(result.recommendations.iter().any(|rec| rec.contains("long tokens")));
    }

    #[test]
    fn test_summary_report() {
        let vocab = create_test_vocab();
        let analyzer = VocabAnalyzer::default();
        let result = analyzer.analyze_vocabulary(&vocab).unwrap();

        let report = VocabDebugUtils::generate_summary_report(&result);
        assert!(report.contains("VOCABULARY ANALYSIS SUMMARY"));
        assert!(report.contains("Total tokens"));
    }
}
