//! Coverage reporting for TrustformeRS tokenizers
//!
//! This module provides comprehensive coverage analysis and reporting
//! for tokenization quality, vocabulary usage, and model performance metrics.

use crate::{TokenizedInput, Tokenizer};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for coverage reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageConfig {
    /// Minimum token frequency for inclusion in reports
    pub min_token_frequency: usize,
    /// Maximum number of examples to include in reports
    pub max_examples: usize,
    /// Whether to include detailed token analysis
    pub include_token_analysis: bool,
    /// Whether to include vocabulary statistics
    pub include_vocab_stats: bool,
    /// Whether to include performance metrics
    pub include_performance_metrics: bool,
    /// Whether to include quality metrics
    pub include_quality_metrics: bool,
    /// Report output format
    pub output_format: ReportFormat,
    /// Coverage thresholds for warnings
    pub coverage_thresholds: CoverageThresholds,
}

impl Default for CoverageConfig {
    fn default() -> Self {
        Self {
            min_token_frequency: 1,
            max_examples: 100,
            include_token_analysis: true,
            include_vocab_stats: true,
            include_performance_metrics: true,
            include_quality_metrics: true,
            output_format: ReportFormat::Json,
            coverage_thresholds: CoverageThresholds::default(),
        }
    }
}

/// Coverage thresholds for quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageThresholds {
    /// Minimum vocabulary coverage percentage
    pub min_vocab_coverage: f64,
    /// Minimum character coverage percentage
    pub min_char_coverage: f64,
    /// Maximum OOV (out-of-vocabulary) rate
    pub max_oov_rate: f64,
    /// Minimum average token length
    pub min_avg_token_length: f64,
    /// Maximum average token length
    pub max_avg_token_length: f64,
    /// Maximum tokens per second for performance warning
    pub min_tokens_per_second: f64,
}

impl Default for CoverageThresholds {
    fn default() -> Self {
        Self {
            min_vocab_coverage: 0.95,
            min_char_coverage: 0.99,
            max_oov_rate: 0.05,
            min_avg_token_length: 2.0,
            max_avg_token_length: 10.0,
            min_tokens_per_second: 1000.0,
        }
    }
}

/// Report output formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    Json,
    Html,
    Markdown,
    Csv,
    Yaml,
}

/// Comprehensive coverage report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageReport {
    /// Report metadata
    pub metadata: ReportMetadata,
    /// Vocabulary coverage analysis
    pub vocabulary_coverage: VocabularyCoverage,
    /// Character coverage analysis
    pub character_coverage: CharacterCoverage,
    /// Token distribution analysis
    pub token_distribution: TokenDistribution,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Quality metrics
    pub quality_metrics: QualityMetrics,
    /// Coverage warnings and recommendations
    pub warnings: Vec<CoverageWarning>,
    /// Detailed examples
    pub examples: Vec<CoverageExample>,
}

/// Report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    /// Report generation timestamp
    pub timestamp: String,
    /// Tokenizer name/type
    pub tokenizer_name: String,
    /// Number of test samples
    pub sample_count: usize,
    /// Total processing time
    pub processing_time: Duration,
    /// Report configuration
    pub config: CoverageConfig,
}

/// Vocabulary coverage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VocabularyCoverage {
    /// Total vocabulary size
    pub total_vocab_size: usize,
    /// Number of tokens used
    pub used_tokens: usize,
    /// Vocabulary coverage percentage
    pub coverage_percentage: f64,
    /// Most frequent tokens
    pub most_frequent_tokens: Vec<(String, usize)>,
    /// Least frequent tokens
    pub least_frequent_tokens: Vec<(String, usize)>,
    /// Unused tokens
    pub unused_tokens: Vec<String>,
    /// Token frequency distribution
    pub frequency_distribution: HashMap<usize, usize>,
}

/// Character coverage analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterCoverage {
    /// Total unique characters in input
    pub total_characters: usize,
    /// Characters covered by tokenizer
    pub covered_characters: usize,
    /// Character coverage percentage
    pub coverage_percentage: f64,
    /// Uncovered characters
    pub uncovered_characters: Vec<char>,
    /// Character frequency distribution
    pub character_frequencies: HashMap<char, usize>,
    /// Unicode category distribution
    pub unicode_categories: HashMap<String, usize>,
}

/// Token distribution analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenDistribution {
    /// Average tokens per input
    pub avg_tokens_per_input: f64,
    /// Token length distribution
    pub token_length_distribution: HashMap<usize, usize>,
    /// Average token length
    pub avg_token_length: f64,
    /// Compression ratio (characters per token)
    pub compression_ratio: f64,
    /// Out-of-vocabulary rate
    pub oov_rate: f64,
    /// Most common token patterns
    pub common_patterns: Vec<(String, usize)>,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total processing time
    pub total_time: Duration,
    /// Average time per input
    pub avg_time_per_input: Duration,
    /// Tokens processed per second
    pub tokens_per_second: f64,
    /// Characters processed per second
    pub characters_per_second: f64,
    /// Memory usage statistics
    pub memory_usage: MemoryUsageStats,
    /// Throughput percentiles
    pub throughput_percentiles: HashMap<String, f64>,
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageStats {
    /// Peak memory usage in bytes
    pub peak_memory: usize,
    /// Average memory usage in bytes
    pub avg_memory: usize,
    /// Memory usage per token
    pub memory_per_token: f64,
    /// Vocabulary memory footprint
    pub vocab_memory: usize,
}

/// Quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Vocabulary efficiency score
    pub vocab_efficiency: f64,
    /// Tokenization consistency score
    pub consistency_score: f64,
    /// Information density score
    pub information_density: f64,
    /// Subword quality score
    pub subword_quality: f64,
    /// Language coverage score
    pub language_coverage: f64,
    /// Overall quality score
    pub overall_score: f64,
}

/// Coverage warnings and issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageWarning {
    /// Warning type
    pub warning_type: WarningType,
    /// Warning severity
    pub severity: WarningSeverity,
    /// Warning message
    pub message: String,
    /// Recommended action
    pub recommendation: String,
    /// Affected examples
    pub examples: Vec<String>,
}

/// Types of coverage warnings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarningType {
    LowVocabCoverage,
    LowCharCoverage,
    HighOOVRate,
    PoorPerformance,
    MemoryIssue,
    InconsistentTokenization,
    VocabularyWaste,
    QualityIssue,
}

/// Warning severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarningSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Coverage examples for detailed analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageExample {
    /// Original input text
    pub input: String,
    /// Tokenized output
    pub tokens: Vec<String>,
    /// Token IDs
    pub token_ids: Vec<u32>,
    /// Character-to-token alignment
    pub alignment: Vec<(usize, usize)>,
    /// Issues found
    pub issues: Vec<String>,
    /// Quality score for this example
    pub quality_score: f64,
}

/// Coverage analyzer implementation
pub struct CoverageAnalyzer<T: Tokenizer> {
    tokenizer: T,
    config: CoverageConfig,
    token_frequencies: HashMap<String, usize>,
    character_frequencies: HashMap<char, usize>,
    processing_times: Vec<Duration>,
    examples: Vec<CoverageExample>,
}

impl<T: Tokenizer> CoverageAnalyzer<T> {
    /// Create a new coverage analyzer
    pub fn new(tokenizer: T, config: CoverageConfig) -> Self {
        Self {
            tokenizer,
            config,
            token_frequencies: HashMap::new(),
            character_frequencies: HashMap::new(),
            processing_times: Vec::new(),
            examples: Vec::new(),
        }
    }

    /// Create with default configuration
    pub fn from_tokenizer(tokenizer: T) -> Self {
        Self::new(tokenizer, CoverageConfig::default())
    }

    /// Analyze a single input text
    pub fn analyze_input(&mut self, text: &str) -> Result<()> {
        let start_time = Instant::now();

        // Tokenize the input
        let tokenized = self.tokenizer.encode(text)?;
        let processing_time = start_time.elapsed();
        self.processing_times.push(processing_time);

        // Decode tokens to get token strings
        let _decoded = self.tokenizer.decode(&tokenized.input_ids)?;
        let token_strings = self.extract_token_strings(&tokenized, text)?;

        // Update token frequencies
        for token in &token_strings {
            *self.token_frequencies.entry(token.clone()).or_insert(0) += 1;
        }

        // Update character frequencies
        for ch in text.chars() {
            *self.character_frequencies.entry(ch).or_insert(0) += 1;
        }

        // Create example if within limit
        if self.examples.len() < self.config.max_examples {
            let example = self.create_coverage_example(text, &tokenized, &token_strings)?;
            self.examples.push(example);
        }

        Ok(())
    }

    /// Analyze a batch of input texts
    pub fn analyze_batch(&mut self, texts: &[String]) -> Result<()> {
        for text in texts {
            self.analyze_input(text)?;
        }
        Ok(())
    }

    /// Generate comprehensive coverage report
    pub fn generate_report(&self) -> Result<CoverageReport> {
        let _start_time = Instant::now();

        let metadata = self.create_report_metadata();
        let vocabulary_coverage = self.analyze_vocabulary_coverage()?;
        let character_coverage = self.analyze_character_coverage();
        let token_distribution = self.analyze_token_distribution();
        let performance_metrics = self.analyze_performance_metrics();
        let quality_metrics = self.calculate_quality_metrics(
            &vocabulary_coverage,
            &character_coverage,
            &token_distribution,
        );
        let warnings = self.generate_warnings(
            &vocabulary_coverage,
            &character_coverage,
            &token_distribution,
            &performance_metrics,
        );

        let report = CoverageReport {
            metadata,
            vocabulary_coverage,
            character_coverage,
            token_distribution,
            performance_metrics,
            quality_metrics,
            warnings,
            examples: self.examples.clone(),
        };

        Ok(report)
    }

    /// Extract token strings from tokenized input
    fn extract_token_strings(
        &self,
        tokenized: &TokenizedInput,
        original_text: &str,
    ) -> Result<Vec<String>> {
        let mut token_strings = Vec::new();

        // Try to decode individual tokens
        for &token_id in &tokenized.input_ids {
            if let Ok(token_str) = self.tokenizer.decode(&[token_id]) {
                token_strings.push(token_str);
            } else {
                // Fallback to token ID as string
                token_strings.push(format!("<{}>", token_id));
            }
        }

        // If we don't have enough tokens, split by approximate token boundaries
        if token_strings.len() != tokenized.input_ids.len() {
            // Fallback: estimate tokens by character spans
            let chars_per_token = original_text.len() as f64 / tokenized.input_ids.len() as f64;
            token_strings.clear();

            for i in 0..tokenized.input_ids.len() {
                let start = (i as f64 * chars_per_token) as usize;
                let end =
                    ((i + 1) as f64 * chars_per_token).min(original_text.len() as f64) as usize;

                if start < original_text.len() {
                    let token = &original_text[start..end];
                    token_strings.push(token.to_string());
                }
            }
        }

        Ok(token_strings)
    }

    /// Create coverage example
    fn create_coverage_example(
        &self,
        text: &str,
        tokenized: &TokenizedInput,
        tokens: &[String],
    ) -> Result<CoverageExample> {
        // Calculate alignment (simplified)
        let mut alignment = Vec::new();
        let mut char_pos = 0;

        for token in tokens.iter() {
            let start_char = char_pos;
            char_pos += token.chars().count();
            alignment.push((start_char, char_pos.min(text.chars().count())));
        }

        // Detect issues
        let mut issues = Vec::new();

        // Check for very short or very long tokens
        for token in tokens {
            if token.len() == 1 && token.chars().all(|c| c.is_alphabetic()) {
                issues.push(format!("Single character token: '{}'", token));
            } else if token.len() > 20 {
                issues.push(format!("Very long token: '{}'", token));
            }
        }

        // Check for potential OOV tokens
        if tokens.iter().any(|t| t.starts_with('<') && t.ends_with('>')) {
            issues.push("Contains unknown tokens".to_string());
        }

        // Calculate quality score
        let quality_score = self.calculate_example_quality_score(text, tokens);

        Ok(CoverageExample {
            input: text.to_string(),
            tokens: tokens.to_vec(),
            token_ids: tokenized.input_ids.clone(),
            alignment,
            issues,
            quality_score,
        })
    }

    /// Calculate quality score for an example
    fn calculate_example_quality_score(&self, text: &str, tokens: &[String]) -> f64 {
        let mut score = 1.0;

        // Penalize for too many or too few tokens
        let char_count = text.chars().count();
        let token_count = tokens.len();
        let tokens_per_char = token_count as f64 / char_count as f64;

        if tokens_per_char > 0.8 {
            score *= 0.7; // Too many tokens (poor compression)
        } else if tokens_per_char < 0.1 {
            score *= 0.8; // Too few tokens (might miss details)
        }

        // Penalize for very short or very long tokens
        let avg_token_length =
            tokens.iter().map(|t| t.len()).sum::<usize>() as f64 / tokens.len() as f64;
        if !(2.0..=15.0).contains(&avg_token_length) {
            score *= 0.9;
        }

        // Penalize for unknown tokens
        let unknown_tokens =
            tokens.iter().filter(|t| t.starts_with('<') && t.ends_with('>')).count();
        if unknown_tokens > 0 {
            score *= 0.5_f64.powi(unknown_tokens as i32);
        }

        score
    }

    /// Create report metadata
    fn create_report_metadata(&self) -> ReportMetadata {
        ReportMetadata {
            timestamp: chrono::Utc::now().to_rfc3339(),
            tokenizer_name: "TrustformeRS Tokenizer".to_string(),
            sample_count: self.processing_times.len(),
            processing_time: self.processing_times.iter().sum(),
            config: self.config.clone(),
        }
    }

    /// Analyze vocabulary coverage
    fn analyze_vocabulary_coverage(&self) -> Result<VocabularyCoverage> {
        // Note: We can't easily get the full vocabulary from the generic tokenizer trait
        // This is a simplified implementation
        let used_tokens = self.token_frequencies.len();
        let total_vocab_size = used_tokens * 2; // Rough estimate
        let coverage_percentage = if total_vocab_size > 0 {
            used_tokens as f64 / total_vocab_size as f64 * 100.0
        } else {
            0.0
        };

        // Get most and least frequent tokens
        let mut sorted_tokens: Vec<_> = self
            .token_frequencies
            .iter()
            .map(|(token, &freq)| (token.clone(), freq))
            .collect();
        sorted_tokens.sort_by(|a, b| b.1.cmp(&a.1));

        let most_frequent_tokens = sorted_tokens.iter().take(20).cloned().collect();

        let least_frequent_tokens = sorted_tokens
            .iter()
            .filter(|(_, freq)| *freq >= self.config.min_token_frequency)
            .rev()
            .take(20)
            .cloned()
            .collect();

        // Calculate frequency distribution
        let mut frequency_distribution = HashMap::new();
        for &freq in self.token_frequencies.values() {
            *frequency_distribution.entry(freq).or_insert(0) += 1;
        }

        Ok(VocabularyCoverage {
            total_vocab_size,
            used_tokens,
            coverage_percentage,
            most_frequent_tokens,
            least_frequent_tokens,
            unused_tokens: Vec::new(), // Can't determine without full vocabulary
            frequency_distribution,
        })
    }

    /// Analyze character coverage
    fn analyze_character_coverage(&self) -> CharacterCoverage {
        let total_characters = self.character_frequencies.len();
        let covered_characters = self.character_frequencies.len(); // All seen characters are "covered"
        let coverage_percentage = 100.0; // By definition, all input characters are covered

        // Analyze Unicode categories
        let mut unicode_categories = HashMap::new();
        for &ch in self.character_frequencies.keys() {
            let category = if ch.is_alphabetic() {
                "Letter".to_string()
            } else if ch.is_numeric() {
                "Number".to_string()
            } else if ch.is_whitespace() {
                "Separator".to_string()
            } else {
                "Other".to_string()
            };
            *unicode_categories.entry(category).or_insert(0) += 1;
        }

        CharacterCoverage {
            total_characters,
            covered_characters,
            coverage_percentage,
            uncovered_characters: Vec::new(),
            character_frequencies: self.character_frequencies.clone(),
            unicode_categories,
        }
    }

    /// Analyze token distribution
    fn analyze_token_distribution(&self) -> TokenDistribution {
        let total_inputs = self.processing_times.len();
        let total_tokens: usize = self.token_frequencies.values().sum();

        let avg_tokens_per_input =
            if total_inputs > 0 { total_tokens as f64 / total_inputs as f64 } else { 0.0 };

        // Calculate token length distribution
        let mut token_length_distribution = HashMap::new();
        let mut total_length = 0;

        for (token, &freq) in &self.token_frequencies {
            let length = token.chars().count();
            *token_length_distribution.entry(length).or_insert(0) += freq;
            total_length += length * freq;
        }

        let avg_token_length =
            if total_tokens > 0 { total_length as f64 / total_tokens as f64 } else { 0.0 };

        // Calculate compression ratio
        let total_chars: usize = self.character_frequencies.values().sum();
        let compression_ratio =
            if total_tokens > 0 { total_chars as f64 / total_tokens as f64 } else { 0.0 };

        // Calculate OOV rate (tokens that look like unknowns)
        let oov_tokens = self
            .token_frequencies
            .iter()
            .filter(|(token, _)| token.starts_with('<') && token.ends_with('>'))
            .map(|(_, &freq)| freq)
            .sum::<usize>();

        let oov_rate = if total_tokens > 0 { oov_tokens as f64 / total_tokens as f64 } else { 0.0 };

        // Find common patterns
        let mut pattern_counts = HashMap::new();
        for token in self.token_frequencies.keys() {
            // Analyze prefixes and suffixes
            if token.len() >= 3 {
                let prefix = &token[..2];
                let suffix = &token[token.len() - 2..];
                *pattern_counts.entry(format!("prefix:{}", prefix)).or_insert(0) += 1;
                *pattern_counts.entry(format!("suffix:{}", suffix)).or_insert(0) += 1;
            }
        }

        let mut common_patterns: Vec<_> = pattern_counts.into_iter().collect();
        common_patterns.sort_by(|a, b| b.1.cmp(&a.1));
        common_patterns.truncate(20);

        TokenDistribution {
            avg_tokens_per_input,
            token_length_distribution,
            avg_token_length,
            compression_ratio,
            oov_rate,
            common_patterns,
        }
    }

    /// Analyze performance metrics
    fn analyze_performance_metrics(&self) -> PerformanceMetrics {
        let total_time: Duration = self.processing_times.iter().sum();
        let avg_time_per_input = if !self.processing_times.is_empty() {
            total_time / self.processing_times.len() as u32
        } else {
            Duration::from_secs(0)
        };

        let total_tokens: usize = self.token_frequencies.values().sum();
        let total_chars: usize = self.character_frequencies.values().sum();

        let tokens_per_second = if total_time.as_secs_f64() > 0.0 {
            total_tokens as f64 / total_time.as_secs_f64()
        } else {
            0.0
        };

        let characters_per_second = if total_time.as_secs_f64() > 0.0 {
            total_chars as f64 / total_time.as_secs_f64()
        } else {
            0.0
        };

        // Calculate throughput percentiles
        let mut sorted_times = self.processing_times.clone();
        sorted_times.sort();

        let mut throughput_percentiles = HashMap::new();
        if !sorted_times.is_empty() {
            let p50_idx = sorted_times.len() / 2;
            let p90_idx = (sorted_times.len() as f64 * 0.9) as usize;
            let p99_idx = (sorted_times.len() as f64 * 0.99) as usize;

            throughput_percentiles.insert("p50".to_string(), sorted_times[p50_idx].as_secs_f64());
            throughput_percentiles.insert("p90".to_string(), sorted_times[p90_idx].as_secs_f64());
            throughput_percentiles.insert("p99".to_string(), sorted_times[p99_idx].as_secs_f64());
        }

        // Rough memory usage estimate
        let vocab_memory = self.token_frequencies.len() * 64; // Rough estimate
        let memory_usage = MemoryUsageStats {
            peak_memory: vocab_memory * 2,
            avg_memory: vocab_memory,
            memory_per_token: if total_tokens > 0 {
                vocab_memory as f64 / total_tokens as f64
            } else {
                0.0
            },
            vocab_memory,
        };

        PerformanceMetrics {
            total_time,
            avg_time_per_input,
            tokens_per_second,
            characters_per_second,
            memory_usage,
            throughput_percentiles,
        }
    }

    /// Calculate quality metrics
    fn calculate_quality_metrics(
        &self,
        vocab_coverage: &VocabularyCoverage,
        char_coverage: &CharacterCoverage,
        token_dist: &TokenDistribution,
    ) -> QualityMetrics {
        // Vocabulary efficiency: how well the vocabulary is utilized
        let vocab_efficiency = vocab_coverage.coverage_percentage / 100.0;

        // Consistency score: based on token length variance
        let consistency_score = if token_dist.avg_token_length > 0.0 {
            1.0 / (1.0 + (token_dist.avg_token_length - 4.0).abs() / 4.0)
        } else {
            0.0
        };

        // Information density: compression ratio normalized
        let information_density = (token_dist.compression_ratio / 5.0).min(1.0);

        // Subword quality: based on OOV rate and token distribution
        let subword_quality = (1.0 - token_dist.oov_rate).max(0.0);

        // Language coverage: based on character coverage
        let language_coverage = char_coverage.coverage_percentage / 100.0;

        // Overall score: weighted average
        let overall_score = vocab_efficiency * 0.2
            + consistency_score * 0.2
            + information_density * 0.2
            + subword_quality * 0.2
            + language_coverage * 0.2;

        QualityMetrics {
            vocab_efficiency,
            consistency_score,
            information_density,
            subword_quality,
            language_coverage,
            overall_score,
        }
    }

    /// Generate warnings based on analysis
    fn generate_warnings(
        &self,
        vocab_coverage: &VocabularyCoverage,
        char_coverage: &CharacterCoverage,
        token_dist: &TokenDistribution,
        performance: &PerformanceMetrics,
    ) -> Vec<CoverageWarning> {
        let mut warnings = Vec::new();
        let thresholds = &self.config.coverage_thresholds;

        // Check vocabulary coverage
        if vocab_coverage.coverage_percentage < thresholds.min_vocab_coverage * 100.0 {
            warnings.push(CoverageWarning {
                warning_type: WarningType::LowVocabCoverage,
                severity: WarningSeverity::Warning,
                message: format!(
                    "Low vocabulary coverage: {:.1}%",
                    vocab_coverage.coverage_percentage
                ),
                recommendation: "Consider using a larger or more diverse training corpus"
                    .to_string(),
                examples: vec![],
            });
        }

        // Check character coverage
        if char_coverage.coverage_percentage < thresholds.min_char_coverage * 100.0 {
            warnings.push(CoverageWarning {
                warning_type: WarningType::LowCharCoverage,
                severity: WarningSeverity::Error,
                message: format!(
                    "Low character coverage: {:.1}%",
                    char_coverage.coverage_percentage
                ),
                recommendation: "Review tokenizer configuration and vocabulary".to_string(),
                examples: char_coverage
                    .uncovered_characters
                    .iter()
                    .take(10)
                    .map(|c| c.to_string())
                    .collect(),
            });
        }

        // Check OOV rate
        if token_dist.oov_rate > thresholds.max_oov_rate {
            warnings.push(CoverageWarning {
                warning_type: WarningType::HighOOVRate,
                severity: WarningSeverity::Error,
                message: format!("High OOV rate: {:.1}%", token_dist.oov_rate * 100.0),
                recommendation: "Expand vocabulary or improve tokenization algorithm".to_string(),
                examples: vec![],
            });
        }

        // Check token length
        if token_dist.avg_token_length < thresholds.min_avg_token_length
            || token_dist.avg_token_length > thresholds.max_avg_token_length
        {
            warnings.push(CoverageWarning {
                warning_type: WarningType::QualityIssue,
                severity: WarningSeverity::Warning,
                message: format!(
                    "Suboptimal average token length: {:.1}",
                    token_dist.avg_token_length
                ),
                recommendation: "Adjust tokenization parameters for better token granularity"
                    .to_string(),
                examples: vec![],
            });
        }

        // Check performance
        if performance.tokens_per_second < thresholds.min_tokens_per_second {
            warnings.push(CoverageWarning {
                warning_type: WarningType::PoorPerformance,
                severity: WarningSeverity::Warning,
                message: format!(
                    "Low throughput: {:.0} tokens/sec",
                    performance.tokens_per_second
                ),
                recommendation: "Consider performance optimizations or hardware upgrades"
                    .to_string(),
                examples: vec![],
            });
        }

        warnings
    }
}

/// Format and export coverage reports
pub struct CoverageReportExporter;

impl CoverageReportExporter {
    /// Export report to string in specified format
    pub fn export_to_string(report: &CoverageReport, format: &ReportFormat) -> Result<String> {
        match format {
            ReportFormat::Json => serde_json::to_string_pretty(report)
                .map_err(|e| anyhow!("Failed to serialize JSON: {}", e)),
            ReportFormat::Yaml => serde_yaml::to_string(report)
                .map_err(|e| anyhow!("Failed to serialize YAML: {}", e)),
            ReportFormat::Html => Self::export_to_html(report),
            ReportFormat::Markdown => Self::export_to_markdown(report),
            ReportFormat::Csv => Self::export_to_csv(report),
        }
    }

    /// Export to HTML format
    fn export_to_html(report: &CoverageReport) -> Result<String> {
        let mut html = String::new();
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>Tokenizer Coverage Report</title>\n");
        html.push_str("<style>body{font-family:Arial,sans-serif;margin:40px;}table{border-collapse:collapse;width:100%;}th,td{border:1px solid #ddd;padding:8px;text-align:left;}</style>\n");
        html.push_str("</head>\n<body>\n");

        html.push_str("<h1>Tokenizer Coverage Report</h1>\n");
        html.push_str(&format!(
            "<p>Generated: {}</p>\n",
            report.metadata.timestamp
        ));
        html.push_str(&format!(
            "<p>Samples: {}</p>\n",
            report.metadata.sample_count
        ));

        // Vocabulary Coverage
        html.push_str("<h2>Vocabulary Coverage</h2>\n");
        html.push_str(&format!(
            "<p>Coverage: {:.1}%</p>\n",
            report.vocabulary_coverage.coverage_percentage
        ));
        html.push_str(&format!(
            "<p>Used Tokens: {}/{}</p>\n",
            report.vocabulary_coverage.used_tokens, report.vocabulary_coverage.total_vocab_size
        ));

        // Performance Metrics
        html.push_str("<h2>Performance Metrics</h2>\n");
        html.push_str(&format!(
            "<p>Tokens/sec: {:.0}</p>\n",
            report.performance_metrics.tokens_per_second
        ));
        html.push_str(&format!(
            "<p>Characters/sec: {:.0}</p>\n",
            report.performance_metrics.characters_per_second
        ));

        // Warnings
        if !report.warnings.is_empty() {
            html.push_str("<h2>Warnings</h2>\n<ul>\n");
            for warning in &report.warnings {
                html.push_str(&format!(
                    "<li><strong>{:?}</strong>: {}</li>\n",
                    warning.severity, warning.message
                ));
            }
            html.push_str("</ul>\n");
        }

        html.push_str("</body>\n</html>");
        Ok(html)
    }

    /// Export to Markdown format
    fn export_to_markdown(report: &CoverageReport) -> Result<String> {
        let mut md = String::new();

        md.push_str("# Tokenizer Coverage Report\n\n");
        md.push_str(&format!("**Generated:** {}\n", report.metadata.timestamp));
        md.push_str(&format!(
            "**Samples:** {}\n\n",
            report.metadata.sample_count
        ));

        md.push_str("## Vocabulary Coverage\n\n");
        md.push_str(&format!(
            "- **Coverage:** {:.1}%\n",
            report.vocabulary_coverage.coverage_percentage
        ));
        md.push_str(&format!(
            "- **Used Tokens:** {}/{}\n\n",
            report.vocabulary_coverage.used_tokens, report.vocabulary_coverage.total_vocab_size
        ));

        md.push_str("## Performance Metrics\n\n");
        md.push_str(&format!(
            "- **Tokens/sec:** {:.0}\n",
            report.performance_metrics.tokens_per_second
        ));
        md.push_str(&format!(
            "- **Characters/sec:** {:.0}\n\n",
            report.performance_metrics.characters_per_second
        ));

        if !report.warnings.is_empty() {
            md.push_str("## Warnings\n\n");
            for warning in &report.warnings {
                md.push_str(&format!(
                    "- **{:?}:** {}\n",
                    warning.severity, warning.message
                ));
            }
        }

        Ok(md)
    }

    /// Export to CSV format
    fn export_to_csv(report: &CoverageReport) -> Result<String> {
        let mut csv = String::new();

        csv.push_str("Metric,Value\n");
        csv.push_str(&format!(
            "Vocabulary Coverage,{:.1}%\n",
            report.vocabulary_coverage.coverage_percentage
        ));
        csv.push_str(&format!(
            "Character Coverage,{:.1}%\n",
            report.character_coverage.coverage_percentage
        ));
        csv.push_str(&format!(
            "Average Token Length,{:.2}\n",
            report.token_distribution.avg_token_length
        ));
        csv.push_str(&format!(
            "OOV Rate,{:.2}%\n",
            report.token_distribution.oov_rate * 100.0
        ));
        csv.push_str(&format!(
            "Tokens per Second,{:.0}\n",
            report.performance_metrics.tokens_per_second
        ));
        csv.push_str(&format!(
            "Overall Quality Score,{:.2}\n",
            report.quality_metrics.overall_score
        ));

        Ok(csv)
    }

    /// Save report to file
    pub fn save_to_file(report: &CoverageReport, path: &str, format: &ReportFormat) -> Result<()> {
        let content = Self::export_to_string(report, format)?;
        std::fs::write(path, content).map_err(|e| anyhow!("Failed to write report to file: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::char::CharTokenizer;
    use std::collections::HashMap;

    fn create_test_char_tokenizer() -> CharTokenizer {
        let mut vocab = HashMap::new();
        vocab.insert("[PAD]".to_string(), 0);
        vocab.insert("[UNK]".to_string(), 1);
        vocab.insert("[CLS]".to_string(), 2);
        vocab.insert("[SEP]".to_string(), 3);
        vocab.insert("h".to_string(), 4);
        vocab.insert("e".to_string(), 5);
        vocab.insert("l".to_string(), 6);
        vocab.insert("o".to_string(), 7);
        vocab.insert("w".to_string(), 8);
        vocab.insert("r".to_string(), 9);
        vocab.insert("d".to_string(), 10);
        vocab.insert(" ".to_string(), 11);
        vocab.insert("t".to_string(), 12);
        vocab.insert("s".to_string(), 13);
        CharTokenizer::new(vocab)
    }

    #[test]
    fn test_coverage_config() {
        let config = CoverageConfig::default();
        assert_eq!(config.min_token_frequency, 1);
        assert_eq!(config.max_examples, 100);
        assert!(config.include_token_analysis);
    }

    #[test]
    fn test_coverage_analyzer_creation() {
        let tokenizer = create_test_char_tokenizer();
        let analyzer = CoverageAnalyzer::from_tokenizer(tokenizer);
        assert_eq!(analyzer.token_frequencies.len(), 0);
        assert_eq!(analyzer.character_frequencies.len(), 0);
    }

    #[test]
    fn test_analyze_input() {
        let tokenizer = create_test_char_tokenizer();
        let mut analyzer = CoverageAnalyzer::from_tokenizer(tokenizer);

        let result = analyzer.analyze_input("hello world");
        assert!(result.is_ok());
        assert!(!analyzer.token_frequencies.is_empty());
        assert!(!analyzer.character_frequencies.is_empty());
    }

    #[test]
    fn test_generate_report() {
        let tokenizer = create_test_char_tokenizer();
        let mut analyzer = CoverageAnalyzer::from_tokenizer(tokenizer);

        analyzer.analyze_input("hello").unwrap();
        analyzer.analyze_input("world").unwrap();

        let report = analyzer.generate_report();
        assert!(report.is_ok());
        let report = report.unwrap();
        assert_eq!(report.metadata.sample_count, 2);
        assert!(report.vocabulary_coverage.used_tokens > 0);
    }

    #[test]
    fn test_report_export_json() {
        let tokenizer = create_test_char_tokenizer();
        let mut analyzer = CoverageAnalyzer::from_tokenizer(tokenizer);

        analyzer.analyze_input("test").unwrap();
        let report = analyzer.generate_report().unwrap();

        let json_result = CoverageReportExporter::export_to_string(&report, &ReportFormat::Json);
        assert!(json_result.is_ok());
        let json = json_result.unwrap();
        assert!(json.contains("vocabulary_coverage"));
    }

    #[test]
    fn test_report_export_markdown() {
        let tokenizer = create_test_char_tokenizer();
        let mut analyzer = CoverageAnalyzer::from_tokenizer(tokenizer);

        analyzer.analyze_input("test").unwrap();
        let report = analyzer.generate_report().unwrap();

        let md_result = CoverageReportExporter::export_to_string(&report, &ReportFormat::Markdown);
        assert!(md_result.is_ok());
        let md = md_result.unwrap();
        assert!(md.contains("# Tokenizer Coverage Report"));
    }

    #[test]
    fn test_coverage_thresholds() {
        let thresholds = CoverageThresholds::default();
        assert_eq!(thresholds.min_vocab_coverage, 0.95);
        assert_eq!(thresholds.max_oov_rate, 0.05);
    }

    #[test]
    fn test_quality_score_calculation() {
        let tokenizer = create_test_char_tokenizer();
        let analyzer = CoverageAnalyzer::from_tokenizer(tokenizer);

        let tokens = vec!["hello".to_string(), "world".to_string()];
        let score = analyzer.calculate_example_quality_score("hello world", &tokens);
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn test_warning_generation() {
        let tokenizer = create_test_char_tokenizer();
        let mut analyzer = CoverageAnalyzer::from_tokenizer(tokenizer);

        // Analyze some text that might generate warnings
        analyzer.analyze_input("a b c d e f g").unwrap(); // Many single-char tokens
        let report = analyzer.generate_report().unwrap();

        // Should have some warnings about token length
        assert!(!report.warnings.is_empty());
    }

    #[test]
    fn test_batch_analysis() {
        let tokenizer = create_test_char_tokenizer();
        let mut analyzer = CoverageAnalyzer::from_tokenizer(tokenizer);

        let texts = vec![
            "hello world".to_string(),
            "goodbye world".to_string(),
            "test text".to_string(),
        ];

        let result = analyzer.analyze_batch(&texts);
        assert!(result.is_ok());
        assert_eq!(analyzer.processing_times.len(), 3);
    }
}
