use crate::tokenizer::TokenizerWrapper;
use crate::visualization::{TokenVisualizer, VisualizationConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::Tokenizer;

/// Comprehensive tokenization debugger for analyzing tokenization behavior
pub struct TokenizationDebugger {
    tokenizers: HashMap<String, TokenizerWrapper>,
    history: Vec<DebugSession>,
    config: DebuggerConfig,
}

/// Configuration for the tokenization debugger
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebuggerConfig {
    /// Maximum number of sessions to keep in history
    pub max_history_size: usize,

    /// Whether to automatically analyze common issues
    pub auto_analyze_issues: bool,

    /// Whether to show detailed character-level information
    pub show_character_details: bool,

    /// Whether to compare with reference tokenizers
    pub enable_comparison: bool,

    /// Maximum text length to debug (for performance)
    pub max_text_length: usize,
}

impl Default for DebuggerConfig {
    fn default() -> Self {
        Self {
            max_history_size: 100,
            auto_analyze_issues: true,
            show_character_details: true,
            enable_comparison: true,
            max_text_length: 10000,
        }
    }
}

/// A debugging session containing input text and analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugSession {
    pub id: String,
    pub input_text: String,
    pub timestamp: u64,
    pub tokenizer_results: HashMap<String, TokenizationResult>,
    pub analysis: DebugAnalysis,
    pub issues: Vec<DetectedIssue>,
}

/// Results from tokenizing with a specific tokenizer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizationResult {
    pub tokenizer_name: String,
    pub tokens: Vec<String>,
    pub token_ids: Vec<u32>,
    pub token_count: usize,
    pub character_count: usize,
    pub compression_ratio: f64,
    pub processing_time_ms: f64,
    pub character_offsets: Option<Vec<(usize, usize)>>,
    pub oov_tokens: Vec<String>,
    pub special_tokens: Vec<String>,
}

/// Analysis of tokenization behavior across tokenizers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugAnalysis {
    pub total_tokenizers: usize,
    pub consensus_tokens: Vec<String>,
    pub disagreement_tokens: Vec<String>,
    pub compression_stats: CompressionStats,
    pub performance_stats: PerformanceStats,
    pub character_analysis: CharacterAnalysis,
    pub pattern_analysis: PatternAnalysis,
}

/// Statistics about compression ratios across tokenizers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    pub min_ratio: f64,
    pub max_ratio: f64,
    pub avg_ratio: f64,
    pub std_deviation: f64,
    pub best_tokenizer: String,
    pub worst_tokenizer: String,
}

/// Performance statistics across tokenizers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub min_time_ms: f64,
    pub max_time_ms: f64,
    pub avg_time_ms: f64,
    pub fastest_tokenizer: String,
    pub slowest_tokenizer: String,
}

/// Analysis of character-level behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterAnalysis {
    pub total_characters: usize,
    pub unique_characters: usize,
    pub character_frequency: HashMap<char, usize>,
    pub problematic_characters: Vec<char>,
    pub unicode_categories: HashMap<String, usize>,
}

/// Analysis of tokenization patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternAnalysis {
    pub common_prefixes: Vec<(String, usize)>,
    pub common_suffixes: Vec<(String, usize)>,
    pub token_length_distribution: HashMap<usize, usize>,
    pub subword_patterns: Vec<(String, usize)>,
}

/// Types of issues that can be detected during tokenization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueType {
    /// High variance in token count across tokenizers
    HighTokenCountVariance,

    /// Many OOV (out-of-vocabulary) tokens
    HighOOVRate,

    /// Poor compression ratio
    PoorCompression,

    /// Slow tokenization performance
    SlowPerformance,

    /// Inconsistent tokenization across similar texts
    InconsistentTokenization,

    /// Problematic Unicode handling
    UnicodeIssues,

    /// Unexpected special token behavior
    SpecialTokenIssues,

    /// Token boundary issues
    BoundaryIssues,
}

/// A detected issue with suggested solutions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedIssue {
    pub issue_type: IssueType,
    pub severity: IssueSeverity,
    pub description: String,
    pub affected_tokenizers: Vec<String>,
    pub suggestions: Vec<String>,
    pub examples: Vec<String>,
}

/// Severity levels for detected issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl Default for TokenizationDebugger {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenizationDebugger {
    /// Create a new tokenization debugger
    pub fn new() -> Self {
        Self {
            tokenizers: HashMap::new(),
            history: Vec::new(),
            config: DebuggerConfig::default(),
        }
    }

    /// Create debugger with custom configuration
    pub fn with_config(config: DebuggerConfig) -> Self {
        Self {
            tokenizers: HashMap::new(),
            history: Vec::new(),
            config,
        }
    }

    /// Add a tokenizer to the debugger
    pub fn add_tokenizer(&mut self, name: String, tokenizer: TokenizerWrapper) {
        self.tokenizers.insert(name, tokenizer);
    }

    /// Remove a tokenizer from the debugger
    pub fn remove_tokenizer(&mut self, name: &str) -> Option<TokenizerWrapper> {
        self.tokenizers.remove(name)
    }

    /// List all available tokenizers
    pub fn list_tokenizers(&self) -> Vec<String> {
        self.tokenizers.keys().cloned().collect()
    }

    /// Debug tokenization of input text with all registered tokenizers
    pub fn debug_text(&mut self, text: &str) -> Result<DebugSession> {
        if text.len() > self.config.max_text_length {
            return Err(TrustformersError::invalid_input(format!(
                "Text too long: {} characters (max: {})",
                text.len(),
                self.config.max_text_length
            )));
        }

        let session_id = format!("debug_{}", chrono::Utc::now().timestamp());
        let mut tokenizer_results = HashMap::new();

        // Tokenize with each registered tokenizer
        for (name, tokenizer) in &self.tokenizers {
            let start_time = std::time::Instant::now();

            match tokenizer.encode(text) {
                Ok(result) => {
                    let processing_time = start_time.elapsed().as_secs_f64() * 1000.0;

                    let compression_ratio = if !text.is_empty() {
                        result.input_ids.len() as f64 / text.len() as f64
                    } else {
                        0.0
                    };

                    // Analyze OOV tokens (simplified - would need tokenizer vocab access)
                    let tokens: Vec<String> = result
                        .input_ids
                        .iter()
                        .filter_map(|&id| tokenizer.id_to_token(id))
                        .collect();
                    let oov_tokens = self.find_oov_tokens(&tokens, tokenizer);
                    let special_tokens = self.find_special_tokens(&tokens, tokenizer);

                    let tokenization_result = TokenizationResult {
                        tokenizer_name: name.clone(),
                        tokens: result
                            .input_ids
                            .iter()
                            .filter_map(|&id| tokenizer.id_to_token(id))
                            .collect(),
                        token_ids: result.input_ids.clone(),
                        token_count: result.input_ids.len(),
                        character_count: text.len(),
                        compression_ratio,
                        processing_time_ms: processing_time,
                        character_offsets: None, // TokenizedInput doesn't have offsets
                        oov_tokens,
                        special_tokens,
                    };

                    tokenizer_results.insert(name.clone(), tokenization_result);
                },
                Err(e) => {
                    // Create error result
                    let tokenization_result = TokenizationResult {
                        tokenizer_name: name.clone(),
                        tokens: vec![format!("ERROR: {}", e)],
                        token_ids: vec![],
                        token_count: 0,
                        character_count: text.len(),
                        compression_ratio: 0.0,
                        processing_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                        character_offsets: None,
                        oov_tokens: vec![],
                        special_tokens: vec![],
                    };

                    tokenizer_results.insert(name.clone(), tokenization_result);
                },
            }
        }

        // Perform analysis
        let analysis = self.analyze_results(&tokenizer_results, text);
        let issues = if self.config.auto_analyze_issues {
            self.detect_issues(&tokenizer_results, &analysis)
        } else {
            vec![]
        };

        let session = DebugSession {
            id: session_id,
            input_text: text.to_string(),
            timestamp: chrono::Utc::now().timestamp() as u64,
            tokenizer_results,
            analysis,
            issues,
        };

        // Add to history
        self.history.push(session.clone());
        if self.history.len() > self.config.max_history_size {
            self.history.remove(0);
        }

        Ok(session)
    }

    /// Compare tokenization across multiple texts
    pub fn compare_texts(&mut self, texts: &[String]) -> Result<Vec<DebugSession>> {
        let mut sessions = Vec::new();

        for text in texts {
            let session = self.debug_text(text)?;
            sessions.push(session);
        }

        Ok(sessions)
    }

    /// Get debugging history
    pub fn get_history(&self) -> &[DebugSession] {
        &self.history
    }

    /// Get a specific session by ID
    pub fn get_session(&self, session_id: &str) -> Option<&DebugSession> {
        self.history.iter().find(|s| s.id == session_id)
    }

    /// Generate a detailed debug report
    pub fn generate_report(&self, session_id: &str) -> Result<String> {
        let session = self.get_session(session_id).ok_or_else(|| {
            TrustformersError::invalid_input(format!("Session not found: {}", session_id))
        })?;

        let mut report = String::new();

        report.push_str("# Tokenization Debug Report\n");
        report.push_str(&format!("Session ID: {}\n", session.id));
        report.push_str(&format!("Timestamp: {}\n", session.timestamp));
        report.push_str(&format!(
            "Input Length: {} characters\n\n",
            session.input_text.len()
        ));

        report.push_str("## Input Text\n");
        report.push_str(&format!("```\n{}\n```\n\n", session.input_text));

        report.push_str("## Tokenizer Results\n");
        for (name, result) in &session.tokenizer_results {
            report.push_str(&format!("### {}\n", name));
            report.push_str(&format!("- Tokens: {}\n", result.token_count));
            report.push_str(&format!("- Compression: {:.3}\n", result.compression_ratio));
            report.push_str(&format!("- Time: {:.2}ms\n", result.processing_time_ms));
            report.push_str(&format!("- OOV Tokens: {}\n", result.oov_tokens.len()));
            report.push_str(&format!(
                "- Special Tokens: {}\n",
                result.special_tokens.len()
            ));
            report.push('\n');
        }

        report.push_str("## Analysis\n");
        let analysis = &session.analysis;
        report.push_str(&format!(
            "- Total Tokenizers: {}\n",
            analysis.total_tokenizers
        ));
        report.push_str(&format!(
            "- Consensus Tokens: {}\n",
            analysis.consensus_tokens.len()
        ));
        report.push_str(&format!(
            "- Disagreement Tokens: {}\n",
            analysis.disagreement_tokens.len()
        ));
        report.push_str(&format!(
            "- Best Compression: {} ({:.3})\n",
            analysis.compression_stats.best_tokenizer, analysis.compression_stats.min_ratio
        ));
        report.push_str(&format!(
            "- Fastest: {} ({:.2}ms)\n",
            analysis.performance_stats.fastest_tokenizer, analysis.performance_stats.min_time_ms
        ));
        report.push('\n');

        if !session.issues.is_empty() {
            report.push_str("## Detected Issues\n");
            for (i, issue) in session.issues.iter().enumerate() {
                report.push_str(&format!("### Issue {}: {:?}\n", i + 1, issue.issue_type));
                report.push_str(&format!("**Severity:** {:?}\n", issue.severity));
                report.push_str(&format!("**Description:** {}\n", issue.description));
                report.push_str(&format!(
                    "**Affected Tokenizers:** {}\n",
                    issue.affected_tokenizers.join(", ")
                ));
                report.push_str("**Suggestions:**\n");
                for suggestion in &issue.suggestions {
                    report.push_str(&format!("- {}\n", suggestion));
                }
                report.push('\n');
            }
        }

        Ok(report)
    }

    /// Generate HTML visualization of tokenization
    pub fn generate_html_visualization(&self, session_id: &str) -> Result<String> {
        let session = self.get_session(session_id).ok_or_else(|| {
            TrustformersError::invalid_input(format!("Session not found: {}", session_id))
        })?;

        // Use the existing visualization module
        let config = VisualizationConfig::default();
        let _visualizer = TokenVisualizer::new(config);

        // Generate visualization for each tokenizer
        let mut html = String::new();
        html.push_str("<!DOCTYPE html><html><head><title>Tokenization Debug</title>");
        html.push_str("<style>body{font-family:Arial,sans-serif;margin:20px;}");
        html.push_str(".tokenizer{margin-bottom:30px;border:1px solid #ccc;padding:15px;}");
        html.push_str(".token{display:inline-block;margin:2px;padding:4px 8px;border:1px solid #999;background:#f0f0f0;}");
        html.push_str("</style></head><body>");

        html.push_str("<h1>Tokenization Debug Report</h1>");
        html.push_str(&format!(
            "<p><strong>Input:</strong> {}</p>",
            session.input_text
        ));

        for (name, result) in &session.tokenizer_results {
            html.push_str(&format!("<div class='tokenizer'><h2>{}</h2>", name));
            html.push_str(&format!(
                "<p>Tokens: {} | Compression: {:.3} | Time: {:.2}ms</p>",
                result.token_count, result.compression_ratio, result.processing_time_ms
            ));

            html.push_str("<div>");
            for token in &result.tokens {
                html.push_str(&format!(
                    "<span class='token'>{}</span>",
                    html_escape(token)
                ));
            }
            html.push_str("</div></div>");
        }

        html.push_str("</body></html>");
        Ok(html)
    }

    /// Analyze tokenization results and generate statistics
    fn analyze_results(
        &self,
        results: &HashMap<String, TokenizationResult>,
        text: &str,
    ) -> DebugAnalysis {
        let total_tokenizers = results.len();

        // Find consensus and disagreement tokens
        let mut token_agreement = HashMap::new();
        for result in results.values() {
            for token in &result.tokens {
                *token_agreement.entry(token.clone()).or_insert(0) += 1;
            }
        }

        let consensus_threshold = (total_tokenizers as f64 * 0.7) as usize;
        let consensus_tokens: Vec<String> = token_agreement
            .iter()
            .filter(|(_, &count)| count >= consensus_threshold)
            .map(|(token, _)| token.clone())
            .collect();

        let disagreement_tokens: Vec<String> = token_agreement
            .iter()
            .filter(|(_, &count)| count < consensus_threshold)
            .map(|(token, _)| token.clone())
            .collect();

        // Compression statistics
        let compression_ratios: Vec<f64> = results.values().map(|r| r.compression_ratio).collect();
        let compression_stats = self.calculate_compression_stats(&compression_ratios, results);

        // Performance statistics
        let performance_times: Vec<f64> = results.values().map(|r| r.processing_time_ms).collect();
        let performance_stats = self.calculate_performance_stats(&performance_times, results);

        // Character analysis
        let character_analysis = self.analyze_characters(text);

        // Pattern analysis
        let pattern_analysis = self.analyze_patterns(results);

        DebugAnalysis {
            total_tokenizers,
            consensus_tokens,
            disagreement_tokens,
            compression_stats,
            performance_stats,
            character_analysis,
            pattern_analysis,
        }
    }

    fn calculate_compression_stats(
        &self,
        ratios: &[f64],
        results: &HashMap<String, TokenizationResult>,
    ) -> CompressionStats {
        if ratios.is_empty() {
            return CompressionStats {
                min_ratio: 0.0,
                max_ratio: 0.0,
                avg_ratio: 0.0,
                std_deviation: 0.0,
                best_tokenizer: "None".to_string(),
                worst_tokenizer: "None".to_string(),
            };
        }

        let min_ratio = ratios.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_ratio = ratios.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg_ratio = ratios.iter().sum::<f64>() / ratios.len() as f64;

        let variance =
            ratios.iter().map(|r| (r - avg_ratio).powi(2)).sum::<f64>() / ratios.len() as f64;
        let std_deviation = variance.sqrt();

        let best_tokenizer = results
            .iter()
            .min_by(|a, b| a.1.compression_ratio.partial_cmp(&b.1.compression_ratio).unwrap())
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| "Unknown".to_string());

        let worst_tokenizer = results
            .iter()
            .max_by(|a, b| a.1.compression_ratio.partial_cmp(&b.1.compression_ratio).unwrap())
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| "Unknown".to_string());

        CompressionStats {
            min_ratio,
            max_ratio,
            avg_ratio,
            std_deviation,
            best_tokenizer,
            worst_tokenizer,
        }
    }

    fn calculate_performance_stats(
        &self,
        times: &[f64],
        results: &HashMap<String, TokenizationResult>,
    ) -> PerformanceStats {
        if times.is_empty() {
            return PerformanceStats {
                min_time_ms: 0.0,
                max_time_ms: 0.0,
                avg_time_ms: 0.0,
                fastest_tokenizer: "None".to_string(),
                slowest_tokenizer: "None".to_string(),
            };
        }

        let min_time_ms = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_time_ms = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let avg_time_ms = times.iter().sum::<f64>() / times.len() as f64;

        let fastest_tokenizer = results
            .iter()
            .min_by(|a, b| a.1.processing_time_ms.partial_cmp(&b.1.processing_time_ms).unwrap())
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| "Unknown".to_string());

        let slowest_tokenizer = results
            .iter()
            .max_by(|a, b| a.1.processing_time_ms.partial_cmp(&b.1.processing_time_ms).unwrap())
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| "Unknown".to_string());

        PerformanceStats {
            min_time_ms,
            max_time_ms,
            avg_time_ms,
            fastest_tokenizer,
            slowest_tokenizer,
        }
    }

    fn analyze_characters(&self, text: &str) -> CharacterAnalysis {
        let mut character_frequency = HashMap::new();
        let mut unicode_categories = HashMap::new();
        let mut problematic_characters = Vec::new();

        for ch in text.chars() {
            *character_frequency.entry(ch).or_insert(0) += 1;

            // Categorize by Unicode general category
            let category = match ch {
                c if c.is_ascii_alphabetic() => "ASCII Letter",
                c if c.is_ascii_digit() => "ASCII Digit",
                c if c.is_ascii_punctuation() => "ASCII Punctuation",
                c if c.is_ascii_whitespace() => "ASCII Whitespace",
                c if c.is_alphabetic() => "Unicode Letter",
                c if c.is_numeric() => "Unicode Number",
                c if c.is_whitespace() => "Unicode Whitespace",
                _ => "Other",
            };
            *unicode_categories.entry(category.to_string()).or_insert(0) += 1;

            // Detect potentially problematic characters
            if (ch.is_control() || (ch as u32) > 0x10000) && !problematic_characters.contains(&ch) {
                problematic_characters.push(ch);
            }
        }

        CharacterAnalysis {
            total_characters: text.len(),
            unique_characters: character_frequency.len(),
            character_frequency,
            problematic_characters,
            unicode_categories,
        }
    }

    fn analyze_patterns(&self, results: &HashMap<String, TokenizationResult>) -> PatternAnalysis {
        let mut all_tokens = Vec::new();
        for result in results.values() {
            all_tokens.extend(result.tokens.iter().cloned());
        }

        // Analyze prefixes and suffixes
        let mut prefix_counts = HashMap::new();
        let mut suffix_counts = HashMap::new();
        let mut length_distribution = HashMap::new();

        for token in &all_tokens {
            *length_distribution.entry(token.len()).or_insert(0) += 1;

            if token.len() >= 2 {
                let prefix = &token[..2];
                let suffix = &token[token.len() - 2..];
                *prefix_counts.entry(prefix.to_string()).or_insert(0) += 1;
                *suffix_counts.entry(suffix.to_string()).or_insert(0) += 1;
            }
        }

        let mut common_prefixes: Vec<_> = prefix_counts.into_iter().collect();
        common_prefixes.sort_by(|a, b| b.1.cmp(&a.1));
        common_prefixes.truncate(10);

        let mut common_suffixes: Vec<_> = suffix_counts.into_iter().collect();
        common_suffixes.sort_by(|a, b| b.1.cmp(&a.1));
        common_suffixes.truncate(10);

        // Simple subword pattern detection
        let mut subword_patterns = HashMap::new();
        for token in &all_tokens {
            if token.starts_with("##") || token.starts_with("▁") || token.ends_with("@@") {
                *subword_patterns.entry(token.clone()).or_insert(0) += 1;
            }
        }

        let mut subword_patterns: Vec<_> = subword_patterns.into_iter().collect();
        subword_patterns.sort_by(|a, b| b.1.cmp(&a.1));
        subword_patterns.truncate(20);

        PatternAnalysis {
            common_prefixes,
            common_suffixes,
            token_length_distribution: length_distribution,
            subword_patterns,
        }
    }

    fn detect_issues(
        &self,
        results: &HashMap<String, TokenizationResult>,
        analysis: &DebugAnalysis,
    ) -> Vec<DetectedIssue> {
        let mut issues = Vec::new();

        // Check for high token count variance
        let token_counts: Vec<usize> = results.values().map(|r| r.token_count).collect();
        if let (Some(&min_tokens), Some(&max_tokens)) =
            (token_counts.iter().min(), token_counts.iter().max())
        {
            let variance_ratio = max_tokens as f64 / min_tokens.max(1) as f64;
            if variance_ratio > 2.0 {
                issues.push(DetectedIssue {
                    issue_type: IssueType::HighTokenCountVariance,
                    severity: IssueSeverity::Medium,
                    description: format!(
                        "High variance in token count: {} to {} tokens",
                        min_tokens, max_tokens
                    ),
                    affected_tokenizers: results.keys().cloned().collect(),
                    suggestions: vec![
                        "Consider using tokenizers with similar vocabularies".to_string(),
                        "Check if different tokenizers are appropriate for the same use case"
                            .to_string(),
                    ],
                    examples: vec![],
                });
            }
        }

        // Check for poor compression
        if analysis.compression_stats.avg_ratio > 0.8 {
            issues.push(DetectedIssue {
                issue_type: IssueType::PoorCompression,
                severity: IssueSeverity::Medium,
                description: format!(
                    "Poor compression ratio: {:.3} (higher is worse)",
                    analysis.compression_stats.avg_ratio
                ),
                affected_tokenizers: results.keys().cloned().collect(),
                suggestions: vec![
                    "Consider using subword tokenizers (BPE, WordPiece, Unigram)".to_string(),
                    "Increase vocabulary size if using limited vocabularies".to_string(),
                    "Check if the text domain matches the tokenizer training data".to_string(),
                ],
                examples: vec![],
            });
        }

        // Check for slow performance
        if analysis.performance_stats.avg_time_ms > 100.0 {
            issues.push(DetectedIssue {
                issue_type: IssueType::SlowPerformance,
                severity: IssueSeverity::Low,
                description: format!(
                    "Slow tokenization: {:.2}ms average",
                    analysis.performance_stats.avg_time_ms
                ),
                affected_tokenizers: vec![analysis.performance_stats.slowest_tokenizer.clone()],
                suggestions: vec![
                    "Consider using faster tokenizers for real-time applications".to_string(),
                    "Check if vocabulary loading can be optimized".to_string(),
                    "Consider caching tokenization results".to_string(),
                ],
                examples: vec![],
            });
        }

        // Check for Unicode issues
        if !analysis.character_analysis.problematic_characters.is_empty() {
            issues.push(DetectedIssue {
                issue_type: IssueType::UnicodeIssues,
                severity: IssueSeverity::High,
                description: format!(
                    "Found {} potentially problematic Unicode characters",
                    analysis.character_analysis.problematic_characters.len()
                ),
                affected_tokenizers: results.keys().cloned().collect(),
                suggestions: vec![
                    "Ensure tokenizers properly handle Unicode normalization".to_string(),
                    "Consider preprocessing to handle control characters".to_string(),
                    "Verify tokenizer training data included diverse Unicode content".to_string(),
                ],
                examples: analysis
                    .character_analysis
                    .problematic_characters
                    .iter()
                    .take(5)
                    .map(|c| format!("'{}'", c))
                    .collect(),
            });
        }

        issues
    }

    fn find_oov_tokens(&self, tokens: &[String], _tokenizer: &TokenizerWrapper) -> Vec<String> {
        // Simplified OOV detection - would need access to tokenizer vocabulary
        // For now, detect common patterns that might indicate OOV tokens
        tokens
            .iter()
            .filter(|token| {
                token.contains("[UNK]") || token.contains("<unk>") || token.contains("�")
            })
            .cloned()
            .collect()
    }

    fn find_special_tokens(&self, tokens: &[String], _tokenizer: &TokenizerWrapper) -> Vec<String> {
        // Detect common special token patterns
        tokens
            .iter()
            .filter(|token| {
                token.starts_with('[') && token.ends_with(']')
                    || token.starts_with('<') && token.ends_with('>')
                    || token.starts_with("▁")
                    || token.starts_with("##")
            })
            .cloned()
            .collect()
    }
}

/// Helper function to escape HTML characters
fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debugger_creation() {
        let debugger = TokenizationDebugger::new();
        assert_eq!(debugger.list_tokenizers().len(), 0);
        assert_eq!(debugger.get_history().len(), 0);
    }

    #[test]
    fn test_config_default() {
        let config = DebuggerConfig::default();
        assert_eq!(config.max_history_size, 100);
        assert!(config.auto_analyze_issues);
        assert!(config.show_character_details);
    }

    #[test]
    fn test_html_escape() {
        assert_eq!(html_escape("<test>"), "&lt;test&gt;");
        assert_eq!(html_escape("&amp;"), "&amp;amp;");
        assert_eq!(html_escape("\"quote\""), "&quot;quote&quot;");
    }
}
