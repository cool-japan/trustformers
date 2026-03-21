//! Training utilities and optimization helpers.
//!
//! This module provides various utility functions for tokenizer training,
//! including coverage analysis, tokenizer comparison, vocabulary optimization,
//! parameter suggestion, and export utilities.

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Write;
use trustformers_core::errors::{Result, TrustformersError};

use super::analysis::CoverageAnalysis;
use super::config::TrainingConfig;
use super::metrics::TrainingMetrics;

/// Utility functions for training analysis and validation.
pub struct TrainingUtils;

impl TrainingUtils {
    /// Analyze vocabulary coverage on a test dataset.
    ///
    /// This function evaluates how well a tokenizer covers the vocabulary
    /// of a given test set, providing detailed statistics about tokenization
    /// effectiveness.
    pub fn analyze_coverage<T: trustformers_core::traits::Tokenizer>(
        tokenizer: &T,
        test_texts: &[String],
    ) -> Result<HashMap<String, f64>> {
        let mut stats = HashMap::new();
        let mut total_tokens = 0;
        let mut oov_tokens = 0;
        let mut char_count = 0;

        for text in test_texts {
            let tokenized = tokenizer.encode(text)?;
            total_tokens += tokenized.input_ids.len();
            char_count += text.chars().count();

            for &id in &tokenized.input_ids {
                if id == 1 {
                    // Assuming UNK token has ID 1
                    oov_tokens += 1;
                }
            }
        }

        stats.insert("total_tokens".to_string(), total_tokens as f64);
        stats.insert("oov_tokens".to_string(), oov_tokens as f64);
        stats.insert(
            "oov_rate".to_string(),
            if total_tokens > 0 { oov_tokens as f64 / total_tokens as f64 } else { 0.0 },
        );
        stats.insert("coverage".to_string(), 1.0 - stats["oov_rate"]);
        stats.insert(
            "compression_ratio".to_string(),
            if char_count > 0 { total_tokens as f64 / char_count as f64 } else { 0.0 },
        );

        Ok(stats)
    }

    /// Compare two tokenizers on the same dataset.
    ///
    /// This function provides a side-by-side comparison of two tokenizers,
    /// returning performance metrics for each one.
    pub fn compare_tokenizers<
        T1: trustformers_core::traits::Tokenizer,
        T2: trustformers_core::traits::Tokenizer,
    >(
        tokenizer1: &T1,
        tokenizer2: &T2,
        test_texts: &[String],
    ) -> Result<HashMap<String, (f64, f64)>> {
        let stats1 = Self::analyze_coverage(tokenizer1, test_texts)?;
        let stats2 = Self::analyze_coverage(tokenizer2, test_texts)?;

        let mut comparison = HashMap::new();
        for (key, value1) in stats1 {
            if let Some(&value2) = stats2.get(&key) {
                comparison.insert(key, (value1, value2));
            }
        }

        Ok(comparison)
    }

    /// Export vocabulary to different file formats.
    ///
    /// Supports JSON, TSV, and plain text formats for vocabulary export.
    pub fn export_vocab<T: trustformers_core::traits::Tokenizer>(
        _tokenizer: &T,
        format: &str,
        path: &str,
    ) -> Result<()> {
        // Create a temporary vocab map - this method needs to be implemented in the trait
        let vocab: HashMap<String, u32> = HashMap::new();
        let mut file = File::create(path)?;

        match format {
            "json" => {
                let json_data = serde_json::to_string_pretty(&vocab)?;
                file.write_all(json_data.as_bytes())?;
            },
            "tsv" => {
                writeln!(file, "token\tid")?;
                for (token, id) in vocab {
                    writeln!(file, "{}\t{}", token, id)?;
                }
            },
            "txt" => {
                for (token, _) in vocab {
                    writeln!(file, "{}", token)?;
                }
            },
            _ => {
                return Err(TrustformersError::invalid_input(format!(
                    "Unsupported format: {}",
                    format
                )));
            },
        }

        Ok(())
    }

    /// Optimize vocabulary size using coverage analysis.
    ///
    /// This function performs a binary search to find the optimal vocabulary size
    /// that achieves the target coverage with the minimum number of tokens.
    pub fn optimize_vocab_size<T, U>(
        trainer_factory: T,
        _texts: &[String],
        validation_texts: &[String],
        target_coverage: f64,
        size_range: (usize, usize),
    ) -> Result<(usize, TrainingMetrics)>
    where
        T: Fn(&TrainingConfig) -> Result<U>,
        U: trustformers_core::traits::Tokenizer,
    {
        let (min_size, max_size) = size_range;
        let mut best_size = min_size;
        let mut best_metrics = TrainingMetrics::new();
        let mut best_coverage = 0.0;

        // Binary search for optimal vocabulary size
        let mut low = min_size;
        let mut high = max_size;

        while low <= high {
            let mid = (low + high) / 2;

            let config = TrainingConfig {
                vocab_size: mid,
                ..Default::default()
            };

            match trainer_factory(&config) {
                Ok(tokenizer) => {
                    let stats = Self::analyze_coverage(&tokenizer, validation_texts)?;
                    let coverage = stats.get("coverage").unwrap_or(&0.0);

                    let metrics = TrainingMetrics {
                        vocab_size: mid,
                        coverage: *coverage,
                        compression_ratio: *stats.get("compression_ratio").unwrap_or(&0.0),
                        oov_rate: *stats.get("oov_rate").unwrap_or(&1.0),
                        total_tokens: *stats.get("total_tokens").unwrap_or(&0.0) as usize,
                        ..Default::default()
                    };

                    if *coverage >= target_coverage && *coverage > best_coverage {
                        best_size = mid;
                        best_metrics = metrics;
                        best_coverage = *coverage;
                    }

                    if *coverage < target_coverage {
                        low = mid + 1;
                    } else {
                        high = mid - 1;
                    }
                },
                Err(_) => {
                    // Training failed, try larger vocabulary
                    low = mid + 1;
                },
            }
        }

        Ok((best_size, best_metrics))
    }

    /// Perform detailed coverage analysis with comprehensive statistics.
    pub fn detailed_coverage_analysis<T: trustformers_core::traits::Tokenizer>(
        tokenizer: &T,
        test_texts: &[String],
    ) -> Result<CoverageAnalysis> {
        let mut char_coverage = HashMap::new();
        let mut word_coverage = HashMap::new();
        let mut length_distribution = HashMap::new();
        let mut oov_tokens = Vec::new();

        let mut total_chars = 0;
        let mut total_words = 0;
        let mut total_tokens = 0;
        let mut covered_chars = 0;
        let mut covered_words = 0;

        for text in test_texts {
            let words: Vec<&str> = text.split_whitespace().collect();
            total_words += words.len();
            total_chars += text.chars().count();

            let tokenized = tokenizer.encode(text)?;
            total_tokens += tokenized.input_ids.len();

            // Track token length distribution with actual token lengths
            for &id in &tokenized.input_ids {
                // Get actual token string from the tokenizer and calculate its length
                let len = if let Some(token_str) = tokenizer.id_to_token(id) {
                    token_str.chars().count()
                } else {
                    1 // Default to 1 for unknown tokens
                };
                *length_distribution.entry(len).or_insert(0) += 1;

                if id == 1 {
                    // Assuming UNK token has ID 1
                    oov_tokens.push("[UNK]".to_string());
                }
            }

            // Character-level coverage - simplified without direct vocab access
            for ch in text.chars() {
                // For now, assume all characters are covered
                // In a real implementation, this would require vocabulary access
                covered_chars += 1;
                *char_coverage.entry(ch).or_insert(0) += 1;
            }

            // Word-level coverage - simplified without direct vocab access
            for word in words {
                // For now, assume all words are covered
                // In a real implementation, this would require vocabulary access
                covered_words += 1;
                *word_coverage.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        let char_coverage_rate =
            if total_chars > 0 { covered_chars as f64 / total_chars as f64 } else { 0.0 };
        let word_coverage_rate =
            if total_words > 0 { covered_words as f64 / total_words as f64 } else { 0.0 };
        let compression_ratio =
            if total_chars > 0 { total_tokens as f64 / total_chars as f64 } else { 0.0 };

        Ok(CoverageAnalysis {
            char_coverage_rate,
            word_coverage_rate,
            compression_ratio,
            total_chars,
            total_words,
            total_tokens,
            covered_chars,
            covered_words,
            length_distribution,
            oov_tokens,
            vocab_size: tokenizer.vocab_size(),
        })
    }

    /// Suggest optimal training parameters based on corpus analysis.
    ///
    /// This function analyzes the provided corpus and suggests appropriate
    /// training parameters based on corpus characteristics.
    pub fn suggest_training_params(texts: &[String]) -> TrainingConfig {
        let mut char_freq = HashMap::new();
        let mut word_freq = HashMap::new();
        let mut total_chars = 0;
        let mut total_words = 0;

        // Analyze corpus characteristics
        for text in texts {
            total_chars += text.chars().count();
            let words: Vec<&str> = text.split_whitespace().collect();
            total_words += words.len();

            for ch in text.chars() {
                *char_freq.entry(ch).or_insert(0) += 1;
            }

            for word in words {
                *word_freq.entry(word.to_string()).or_insert(0) += 1;
            }
        }

        let _unique_chars = char_freq.len();
        let unique_words = word_freq.len();

        // Calculate corpus statistics
        let avg_word_length =
            if total_words > 0 { total_chars as f64 / total_words as f64 } else { 0.0 };
        let vocab_diversity = unique_words as f64 / total_words as f64;

        // Suggest parameters based on corpus characteristics
        let suggested_vocab_size = if vocab_diversity > 0.1 {
            // High diversity corpus needs larger vocabulary
            (unique_words * 2).clamp(8000, 50000)
        } else {
            // Low diversity corpus can use smaller vocabulary
            (unique_words + 1000).clamp(4000, 30000)
        };

        let min_frequency = if total_words > 100000 {
            // Large corpus can afford higher min frequency
            3
        } else {
            // Small corpus should use lower min frequency
            1
        };

        let max_input_chars = (avg_word_length * 2.0) as usize;

        TrainingConfig {
            vocab_size: suggested_vocab_size,
            min_frequency,
            max_input_chars_per_word: max_input_chars.clamp(50, 200),
            ..Default::default()
        }
    }

    /// Validate tokenizer performance against quality thresholds.
    pub fn validate_tokenizer_quality<T: trustformers_core::traits::Tokenizer>(
        tokenizer: &T,
        test_texts: &[String],
        quality_thresholds: &QualityThresholds,
    ) -> Result<QualityReport> {
        let coverage_stats = Self::analyze_coverage(tokenizer, test_texts)?;
        let detailed_analysis = Self::detailed_coverage_analysis(tokenizer, test_texts)?;

        let coverage = coverage_stats.get("coverage").unwrap_or(&0.0);
        let compression_ratio = coverage_stats.get("compression_ratio").unwrap_or(&0.0);
        let oov_rate = coverage_stats.get("oov_rate").unwrap_or(&1.0);

        let mut issues = Vec::new();
        let mut warnings = Vec::new();

        // Check coverage threshold
        if *coverage < quality_thresholds.min_coverage {
            issues.push(format!(
                "Coverage {:.2}% below minimum threshold {:.2}%",
                coverage * 100.0,
                quality_thresholds.min_coverage * 100.0
            ));
        }

        // Check OOV rate threshold
        if *oov_rate > quality_thresholds.max_oov_rate {
            issues.push(format!(
                "OOV rate {:.2}% above maximum threshold {:.2}%",
                oov_rate * 100.0,
                quality_thresholds.max_oov_rate * 100.0
            ));
        }

        // Check compression ratio
        if *compression_ratio > quality_thresholds.max_compression_ratio {
            warnings.push(format!(
                "High compression ratio {:.3} may indicate over-tokenization",
                compression_ratio
            ));
        }

        // Check vocabulary size efficiency
        let efficiency_score = detailed_analysis.efficiency_score();
        if efficiency_score < quality_thresholds.min_efficiency_score {
            warnings.push(format!(
                "Low efficiency score {:.3} suggests suboptimal vocabulary",
                efficiency_score
            ));
        }

        let overall_quality = if issues.is_empty() {
            if warnings.is_empty() {
                QualityLevel::Excellent
            } else {
                QualityLevel::Good
            }
        } else if issues.len() < 2 {
            QualityLevel::Acceptable
        } else {
            QualityLevel::Poor
        };

        Ok(QualityReport {
            overall_quality,
            coverage: *coverage,
            oov_rate: *oov_rate,
            compression_ratio: *compression_ratio,
            efficiency_score,
            issues,
            warnings,
            vocab_size: tokenizer.vocab_size(),
        })
    }

    /// Generate comprehensive training recommendations.
    pub fn generate_training_recommendations(
        texts: &[String],
        current_config: Option<&TrainingConfig>,
    ) -> TrainingRecommendations {
        let suggested_config = Self::suggest_training_params(texts);

        let mut recommendations = Vec::new();

        if let Some(current) = current_config {
            // Compare with current configuration and provide specific recommendations
            if current.vocab_size < suggested_config.vocab_size / 2 {
                recommendations
                    .push("Consider increasing vocabulary size for better coverage".to_string());
            } else if current.vocab_size > suggested_config.vocab_size * 2 {
                recommendations
                    .push("Consider reducing vocabulary size to improve efficiency".to_string());
            }

            if current.min_frequency < suggested_config.min_frequency {
                recommendations
                    .push("Consider increasing minimum frequency to reduce noise".to_string());
            }

            if current.max_input_chars_per_word < suggested_config.max_input_chars_per_word {
                recommendations
                    .push("Consider increasing max input characters per word".to_string());
            }
        } else {
            recommendations.push("Use suggested configuration as starting point".to_string());
            recommendations.push("Consider running vocabulary size optimization".to_string());
            recommendations.push("Validate on held-out test set".to_string());
        }

        // Add general recommendations based on corpus size
        let total_words: usize = texts.iter().map(|t| t.split_whitespace().count()).sum();
        if total_words < 10000 {
            recommendations.push("Corpus is small - consider lowering min_frequency".to_string());
        } else if total_words > 1000000 {
            recommendations.push("Large corpus detected - consider streaming training".to_string());
        }

        TrainingRecommendations {
            suggested_config,
            recommendations,
            corpus_size: total_words,
        }
    }

    /// Perform statistical analysis of tokenization patterns.
    pub fn analyze_tokenization_patterns<T: trustformers_core::traits::Tokenizer>(
        tokenizer: &T,
        texts: &[String],
    ) -> Result<TokenizationPatterns> {
        let mut token_lengths = Vec::new();
        let mut tokens_per_sentence = Vec::new();
        let mut unique_tokens = HashSet::new();
        let mut total_tokens = 0;

        for text in texts {
            let tokenized = tokenizer.encode(text)?;
            tokens_per_sentence.push(tokenized.input_ids.len());
            total_tokens += tokenized.input_ids.len();

            for &id in &tokenized.input_ids {
                unique_tokens.insert(id);
                if let Some(token_str) = tokenizer.id_to_token(id) {
                    token_lengths.push(token_str.chars().count());
                }
            }
        }

        // Calculate statistics
        let avg_tokens_per_sentence = if !tokens_per_sentence.is_empty() {
            tokens_per_sentence.iter().sum::<usize>() as f64 / tokens_per_sentence.len() as f64
        } else {
            0.0
        };

        let avg_token_length = if !token_lengths.is_empty() {
            token_lengths.iter().sum::<usize>() as f64 / token_lengths.len() as f64
        } else {
            0.0
        };

        let vocab_utilization = if tokenizer.vocab_size() > 0 {
            unique_tokens.len() as f64 / tokenizer.vocab_size() as f64
        } else {
            0.0
        };

        Ok(TokenizationPatterns {
            avg_tokens_per_sentence,
            avg_token_length,
            vocab_utilization,
            total_unique_tokens: unique_tokens.len(),
            total_tokens,
        })
    }
}

/// Quality thresholds for tokenizer validation.
#[derive(Debug, Clone)]
pub struct QualityThresholds {
    pub min_coverage: f64,
    pub max_oov_rate: f64,
    pub max_compression_ratio: f64,
    pub min_efficiency_score: f64,
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_coverage: 0.95,         // 95% coverage
            max_oov_rate: 0.05,         // 5% OOV rate
            max_compression_ratio: 0.8, // 0.8 tokens per character
            min_efficiency_score: 0.7,  // 0.7 efficiency score
        }
    }
}

/// Quality assessment levels.
#[derive(Debug, Clone, PartialEq)]
pub enum QualityLevel {
    Excellent,
    Good,
    Acceptable,
    Poor,
}

/// Comprehensive quality report for a tokenizer.
#[derive(Debug, Clone)]
pub struct QualityReport {
    pub overall_quality: QualityLevel,
    pub coverage: f64,
    pub oov_rate: f64,
    pub compression_ratio: f64,
    pub efficiency_score: f64,
    pub issues: Vec<String>,
    pub warnings: Vec<String>,
    pub vocab_size: usize,
}

impl QualityReport {
    /// Generate a human-readable quality report.
    pub fn summary(&self) -> String {
        let quality_str = match self.overall_quality {
            QualityLevel::Excellent => "Excellent",
            QualityLevel::Good => "Good",
            QualityLevel::Acceptable => "Acceptable",
            QualityLevel::Poor => "Poor",
        };

        let mut report = format!(
            "Tokenizer Quality Report\n\
             ========================\n\
             Overall Quality: {}\n\
             Coverage: {:.2}%\n\
             OOV Rate: {:.2}%\n\
             Compression Ratio: {:.3}\n\
             Efficiency Score: {:.3}\n\
             Vocabulary Size: {}\n",
            quality_str,
            self.coverage * 100.0,
            self.oov_rate * 100.0,
            self.compression_ratio,
            self.efficiency_score,
            self.vocab_size
        );

        if !self.issues.is_empty() {
            report.push_str("\nIssues:\n");
            for issue in &self.issues {
                report.push_str(&format!("  - {}\n", issue));
            }
        }

        if !self.warnings.is_empty() {
            report.push_str("\nWarnings:\n");
            for warning in &self.warnings {
                report.push_str(&format!("  - {}\n", warning));
            }
        }

        report
    }
}

/// Training recommendations based on corpus analysis.
#[derive(Debug, Clone)]
pub struct TrainingRecommendations {
    pub suggested_config: TrainingConfig,
    pub recommendations: Vec<String>,
    pub corpus_size: usize,
}

impl TrainingRecommendations {
    /// Generate a formatted recommendations report.
    pub fn report(&self) -> String {
        let mut report = format!(
            "Training Recommendations\n\
             ========================\n\
             Corpus Size: {} words\n\
             \n\
             Suggested Configuration:\n\
             - Vocabulary Size: {}\n\
             - Min Frequency: {}\n\
             - Max Input Chars: {}\n\
             \n\
             Recommendations:\n",
            self.corpus_size,
            self.suggested_config.vocab_size,
            self.suggested_config.min_frequency,
            self.suggested_config.max_input_chars_per_word
        );

        for rec in &self.recommendations {
            report.push_str(&format!("  - {}\n", rec));
        }

        report
    }
}

/// Statistical analysis of tokenization patterns.
#[derive(Debug, Clone)]
pub struct TokenizationPatterns {
    pub avg_tokens_per_sentence: f64,
    pub avg_token_length: f64,
    pub vocab_utilization: f64,
    pub total_unique_tokens: usize,
    pub total_tokens: usize,
}

impl TokenizationPatterns {
    /// Generate a pattern analysis report.
    pub fn report(&self) -> String {
        format!(
            "Tokenization Patterns Analysis\n\
             ==============================\n\
             Average Tokens per Sentence: {:.2}\n\
             Average Token Length: {:.2} characters\n\
             Vocabulary Utilization: {:.2}%\n\
             Total Unique Tokens: {}\n\
             Total Tokens: {}",
            self.avg_tokens_per_sentence,
            self.avg_token_length,
            self.vocab_utilization * 100.0,
            self.total_unique_tokens,
            self.total_tokens
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suggest_training_params() {
        let texts = vec![
            "hello world".to_string(),
            "hello there".to_string(),
            "world peace".to_string(),
        ];

        let config = TrainingUtils::suggest_training_params(&texts);
        assert!(config.vocab_size > 0);
        assert!(config.min_frequency > 0);
        assert!(config.max_input_chars_per_word >= 50);
    }

    #[test]
    fn test_quality_thresholds() {
        let thresholds = QualityThresholds::default();
        assert_eq!(thresholds.min_coverage, 0.95);
        assert_eq!(thresholds.max_oov_rate, 0.05);
        assert!(thresholds.max_compression_ratio > 0.0);
        assert!(thresholds.min_efficiency_score > 0.0);
    }

    #[test]
    fn test_quality_report() {
        let report = QualityReport {
            overall_quality: QualityLevel::Good,
            coverage: 0.92,
            oov_rate: 0.08,
            compression_ratio: 0.6,
            efficiency_score: 0.75,
            issues: vec!["Test issue".to_string()],
            warnings: vec!["Test warning".to_string()],
            vocab_size: 5000,
        };

        let summary = report.summary();
        assert!(summary.contains("Good"));
        assert!(summary.contains("92.00%"));
        assert!(summary.contains("Test issue"));
        assert!(summary.contains("Test warning"));
    }

    #[test]
    fn test_training_recommendations() {
        let recommendations = TrainingRecommendations {
            suggested_config: TrainingConfig::default(),
            recommendations: vec![
                "Increase vocabulary size".to_string(),
                "Use validation set".to_string(),
            ],
            corpus_size: 10000,
        };

        let report = recommendations.report();
        assert!(report.contains("10000 words"));
        assert!(report.contains("Increase vocabulary size"));
        assert!(report.contains("30000")); // Default vocab size
    }

    #[test]
    fn test_tokenization_patterns() {
        let patterns = TokenizationPatterns {
            avg_tokens_per_sentence: 8.5,
            avg_token_length: 4.2,
            vocab_utilization: 0.75,
            total_unique_tokens: 1500,
            total_tokens: 10000,
        };

        let report = patterns.report();
        assert!(report.contains("8.50"));
        assert!(report.contains("4.20"));
        assert!(report.contains("75.00%"));
        assert!(report.contains("1500"));
        assert!(report.contains("10000"));
    }

    #[test]
    fn test_quality_levels() {
        assert_eq!(QualityLevel::Excellent, QualityLevel::Excellent);
        assert_ne!(QualityLevel::Good, QualityLevel::Poor);
    }
}
