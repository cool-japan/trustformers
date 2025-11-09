use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::{TokenizedInput, Tokenizer};

/// Configuration for token visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationConfig {
    pub show_token_ids: bool,
    pub show_attention_mask: bool,
    pub show_special_tokens: bool,
    pub show_position_info: bool,
    pub use_colors: bool,
    pub max_display_length: Option<usize>,
    pub highlight_patterns: Vec<String>,
    pub custom_token_colors: HashMap<String, String>,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            show_token_ids: true,
            show_attention_mask: false,
            show_special_tokens: true,
            show_position_info: false,
            use_colors: true,
            max_display_length: Some(100),
            highlight_patterns: Vec::new(),
            custom_token_colors: HashMap::new(),
        }
    }
}

/// Statistics about tokenization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizationStats {
    pub total_tokens: usize,
    pub unique_tokens: usize,
    pub special_tokens_count: usize,
    pub average_token_length: f64,
    pub compression_ratio: f64,
    pub oov_count: usize,
    pub token_type_distribution: HashMap<String, usize>,
    pub longest_token: Option<String>,
    pub shortest_token: Option<String>,
}

/// Detailed token information for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenInfo {
    pub token: String,
    pub token_id: u32,
    pub position: usize,
    pub start_char: Option<usize>,
    pub end_char: Option<usize>,
    pub is_special: bool,
    pub attention_value: u8,
    pub token_type: Option<String>,
    pub frequency: Option<f64>,
}

/// Visualization of tokenized input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenVisualization {
    pub original_text: String,
    pub tokens: Vec<TokenInfo>,
    pub statistics: TokenizationStats,
    pub config: VisualizationConfig,
}

/// Comparison between different tokenizers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerComparison {
    pub original_text: String,
    pub tokenizations: HashMap<String, TokenVisualization>,
    pub comparison_stats: ComparisonStats,
}

/// Statistics comparing multiple tokenizers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonStats {
    pub token_count_variance: f64,
    pub common_tokens: Vec<String>,
    pub unique_tokens_by_tokenizer: HashMap<String, Vec<String>>,
    pub compression_ratio_comparison: HashMap<String, f64>,
    pub similarity_scores: HashMap<String, HashMap<String, f64>>,
}

/// Token visualizer implementation
pub struct TokenVisualizer {
    config: VisualizationConfig,
    special_tokens: HashMap<String, u32>,
}

impl TokenVisualizer {
    /// Create a new token visualizer
    pub fn new(config: VisualizationConfig) -> Self {
        Self {
            config,
            special_tokens: HashMap::new(),
        }
    }

    /// Create visualizer with default configuration
    pub fn default() -> Self {
        Self::new(VisualizationConfig::default())
    }

    /// Add special tokens for recognition
    pub fn with_special_tokens(mut self, special_tokens: HashMap<String, u32>) -> Self {
        self.special_tokens = special_tokens;
        self
    }

    /// Visualize tokenization from any tokenizer
    pub fn visualize<T: Tokenizer>(&self, tokenizer: &T, text: &str) -> Result<TokenVisualization> {
        let tokenized = tokenizer.encode(text)?;
        let tokens = self.extract_token_info(tokenizer, text, &tokenized)?;
        let statistics = self.calculate_statistics(text, &tokens);

        Ok(TokenVisualization {
            original_text: text.to_string(),
            tokens,
            statistics,
            config: self.config.clone(),
        })
    }

    /// Extract detailed token information
    fn extract_token_info<T: Tokenizer>(
        &self,
        tokenizer: &T,
        _original_text: &str,
        tokenized: &TokenizedInput,
    ) -> Result<Vec<TokenInfo>> {
        let mut tokens = Vec::new();

        for (i, &token_id) in tokenized.input_ids.iter().enumerate() {
            // Try to decode individual token
            let token_text = match tokenizer.decode(&[token_id]) {
                Ok(text) => text,
                Err(_) => format!("[UNK:{}]", token_id),
            };

            let is_special = self.special_tokens.values().any(|&id| id == token_id)
                || token_text.starts_with('[') && token_text.ends_with(']');

            let attention_value = tokenized.attention_mask.get(i).copied().unwrap_or(0);

            tokens.push(TokenInfo {
                token: token_text,
                token_id,
                position: i,
                start_char: None, // Would need offset mapping from tokenizer
                end_char: None,
                is_special,
                attention_value,
                token_type: None, // Could be enhanced with token type classification
                frequency: None,  // Could be enhanced with frequency data
            });
        }

        Ok(tokens)
    }

    /// Calculate tokenization statistics
    fn calculate_statistics(&self, original_text: &str, tokens: &[TokenInfo]) -> TokenizationStats {
        let total_tokens = tokens.len();
        let unique_tokens =
            tokens.iter().map(|t| &t.token).collect::<std::collections::HashSet<_>>().len();

        let special_tokens_count = tokens.iter().filter(|t| t.is_special).count();

        let total_char_length: usize = tokens.iter().map(|t| t.token.len()).sum();

        let average_token_length = if total_tokens > 0 {
            total_char_length as f64 / total_tokens as f64
        } else {
            0.0
        };

        let compression_ratio = if !original_text.is_empty() {
            total_tokens as f64 / original_text.len() as f64
        } else {
            0.0
        };

        let oov_count =
            tokens.iter().filter(|t| t.token.contains("[UNK") || t.token == "[UNK]").count();

        let mut token_type_distribution = HashMap::new();
        for token in tokens {
            let token_type = self.classify_token(&token.token);
            *token_type_distribution.entry(token_type).or_insert(0) += 1;
        }

        let longest_token = tokens.iter().max_by_key(|t| t.token.len()).map(|t| t.token.clone());

        let shortest_token = tokens
            .iter()
            .filter(|t| !t.is_special)
            .min_by_key(|t| t.token.len())
            .map(|t| t.token.clone());

        TokenizationStats {
            total_tokens,
            unique_tokens,
            special_tokens_count,
            average_token_length,
            compression_ratio,
            oov_count,
            token_type_distribution,
            longest_token,
            shortest_token,
        }
    }

    /// Classify token type for statistics
    fn classify_token(&self, token: &str) -> String {
        if token.starts_with('[') && token.ends_with(']') {
            "special".to_string()
        } else if token.chars().all(|c| c.is_numeric()) {
            "numeric".to_string()
        } else if token.chars().all(|c| c.is_alphabetic()) {
            "alphabetic".to_string()
        } else if token.chars().all(|c| c.is_alphanumeric()) {
            "alphanumeric".to_string()
        } else if token.chars().all(|c| c.is_whitespace()) {
            "whitespace".to_string()
        } else if token.chars().all(|c| c.is_ascii_punctuation()) {
            "punctuation".to_string()
        } else {
            "mixed".to_string()
        }
    }

    /// Compare multiple tokenizers
    pub fn compare_tokenizers<T: Tokenizer>(
        &self,
        tokenizers: HashMap<String, &T>,
        text: &str,
    ) -> Result<TokenizerComparison> {
        let mut tokenizations = HashMap::new();

        for (name, tokenizer) in tokenizers {
            let visualization = self.visualize(tokenizer, text)?;
            tokenizations.insert(name, visualization);
        }

        let comparison_stats = self.calculate_comparison_stats(&tokenizations);

        Ok(TokenizerComparison {
            original_text: text.to_string(),
            tokenizations,
            comparison_stats,
        })
    }

    /// Calculate comparison statistics
    fn calculate_comparison_stats(
        &self,
        tokenizations: &HashMap<String, TokenVisualization>,
    ) -> ComparisonStats {
        let token_counts: Vec<usize> =
            tokenizations.values().map(|t| t.statistics.total_tokens).collect();

        let token_count_variance = if token_counts.len() > 1 {
            let mean = token_counts.iter().sum::<usize>() as f64 / token_counts.len() as f64;
            let variance_sum: f64 =
                token_counts.iter().map(|&count| (count as f64 - mean).powi(2)).sum();
            variance_sum / token_counts.len() as f64
        } else {
            0.0
        };

        // Find common tokens across all tokenizers
        let all_tokens: Vec<Vec<String>> = tokenizations
            .values()
            .map(|t| t.tokens.iter().map(|token| token.token.clone()).collect())
            .collect();

        let mut common_tokens = Vec::new();
        if !all_tokens.is_empty() {
            let first_tokens: std::collections::HashSet<String> =
                all_tokens[0].iter().cloned().collect();
            common_tokens = first_tokens
                .into_iter()
                .filter(|token| all_tokens.iter().skip(1).all(|tokens| tokens.contains(token)))
                .collect();
        }

        // Find unique tokens by tokenizer
        let mut unique_tokens_by_tokenizer = HashMap::new();
        for (name, visualization) in tokenizations {
            let tokens: std::collections::HashSet<String> =
                visualization.tokens.iter().map(|t| t.token.clone()).collect();

            let unique: Vec<String> = tokens
                .into_iter()
                .filter(|token| {
                    tokenizations
                        .iter()
                        .filter(|(other_name, _)| *other_name != name)
                        .all(|(_, other_viz)| !other_viz.tokens.iter().any(|t| &t.token == token))
                })
                .collect();

            unique_tokens_by_tokenizer.insert(name.clone(), unique);
        }

        // Compression ratio comparison
        let compression_ratio_comparison: HashMap<String, f64> = tokenizations
            .iter()
            .map(|(name, viz)| (name.clone(), viz.statistics.compression_ratio))
            .collect();

        // Calculate similarity scores (Jaccard similarity)
        let mut similarity_scores = HashMap::new();
        for (name1, viz1) in tokenizations {
            let mut scores = HashMap::new();
            let tokens1: std::collections::HashSet<String> =
                viz1.tokens.iter().map(|t| t.token.clone()).collect();

            for (name2, viz2) in tokenizations {
                if name1 != name2 {
                    let tokens2: std::collections::HashSet<String> =
                        viz2.tokens.iter().map(|t| t.token.clone()).collect();

                    let intersection = tokens1.intersection(&tokens2).count();
                    let union = tokens1.union(&tokens2).count();
                    let similarity =
                        if union > 0 { intersection as f64 / union as f64 } else { 0.0 };

                    scores.insert(name2.clone(), similarity);
                }
            }
            similarity_scores.insert(name1.clone(), scores);
        }

        ComparisonStats {
            token_count_variance,
            common_tokens,
            unique_tokens_by_tokenizer,
            compression_ratio_comparison,
            similarity_scores,
        }
    }

    /// Generate HTML visualization
    pub fn to_html(&self, visualization: &TokenVisualization) -> String {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>Token Visualization</title>\n");
        html.push_str("<style>\n");
        html.push_str(Self::get_css());
        html.push_str("</style>\n</head>\n<body>\n");

        html.push_str("<h1>Token Visualization</h1>\n");

        // Original text
        html.push_str("<div class='section'>\n");
        html.push_str("<h2>Original Text</h2>\n");
        html.push_str(&format!(
            "<div class='original-text'>{}</div>\n",
            html_escape(&visualization.original_text)
        ));
        html.push_str("</div>\n");

        // Tokens
        html.push_str("<div class='section'>\n");
        html.push_str("<h2>Tokens</h2>\n");
        html.push_str("<div class='tokens'>\n");

        for (i, token) in visualization.tokens.iter().enumerate() {
            let class = if token.is_special { "token special" } else { "token" };
            let color = self.get_token_color(token);

            html.push_str(&format!(
                "<span class='{}' style='background-color: {}' title='ID: {}, Pos: {}'>",
                class, color, token.token_id, token.position
            ));
            html.push_str(&html_escape(&token.token));

            if self.config.show_token_ids {
                html.push_str(&format!("<sub>{}</sub>", token.token_id));
            }

            html.push_str("</span>");

            if i < visualization.tokens.len() - 1 {
                html.push(' ');
            }
        }

        html.push_str("</div>\n</div>\n");

        // Statistics
        html.push_str("<div class='section'>\n");
        html.push_str("<h2>Statistics</h2>\n");
        html.push_str("<table class='stats-table'>\n");

        let stats = &visualization.statistics;
        html.push_str(&format!(
            "<tr><td>Total Tokens</td><td>{}</td></tr>\n",
            stats.total_tokens
        ));
        html.push_str(&format!(
            "<tr><td>Unique Tokens</td><td>{}</td></tr>\n",
            stats.unique_tokens
        ));
        html.push_str(&format!(
            "<tr><td>Special Tokens</td><td>{}</td></tr>\n",
            stats.special_tokens_count
        ));
        html.push_str(&format!(
            "<tr><td>Average Token Length</td><td>{:.2}</td></tr>\n",
            stats.average_token_length
        ));
        html.push_str(&format!(
            "<tr><td>Compression Ratio</td><td>{:.4}</td></tr>\n",
            stats.compression_ratio
        ));
        html.push_str(&format!(
            "<tr><td>OOV Count</td><td>{}</td></tr>\n",
            stats.oov_count
        ));

        if let Some(longest) = &stats.longest_token {
            html.push_str(&format!(
                "<tr><td>Longest Token</td><td>{}</td></tr>\n",
                html_escape(longest)
            ));
        }

        if let Some(shortest) = &stats.shortest_token {
            html.push_str(&format!(
                "<tr><td>Shortest Token</td><td>{}</td></tr>\n",
                html_escape(shortest)
            ));
        }

        html.push_str("</table>\n</div>\n");

        // Token type distribution
        html.push_str("<div class='section'>\n");
        html.push_str("<h2>Token Type Distribution</h2>\n");
        html.push_str("<table class='stats-table'>\n");

        for (token_type, count) in &stats.token_type_distribution {
            html.push_str(&format!(
                "<tr><td>{}</td><td>{}</td></tr>\n",
                token_type, count
            ));
        }

        html.push_str("</table>\n</div>\n");

        html.push_str("</body>\n</html>");
        html
    }

    /// Get CSS styles for HTML visualization
    fn get_css() -> &'static str {
        r#"
body {
    font-family: Arial, sans-serif;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f5f5f5;
}

.section {
    background: white;
    margin: 20px 0;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

h1, h2 {
    color: #333;
}

.original-text {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 4px;
    font-family: monospace;
    border-left: 4px solid #007bff;
}

.tokens {
    font-family: monospace;
    line-height: 2;
    word-wrap: break-word;
}

.token {
    display: inline-block;
    padding: 2px 4px;
    margin: 1px;
    border-radius: 3px;
    border: 1px solid #ddd;
    background-color: #e9ecef;
    position: relative;
}

.token.special {
    background-color: #fff3cd;
    border-color: #ffeaa7;
    font-weight: bold;
}

.token:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    z-index: 10;
}

.stats-table {
    width: 100%;
    border-collapse: collapse;
}

.stats-table td {
    padding: 8px 12px;
    border-bottom: 1px solid #eee;
}

.stats-table td:first-child {
    font-weight: bold;
    color: #555;
}

sub {
    font-size: 0.7em;
    color: #666;
}
"#
    }

    /// Get color for a token
    fn get_token_color(&self, token: &TokenInfo) -> String {
        if let Some(color) = self.config.custom_token_colors.get(&token.token) {
            return color.clone();
        }

        if token.is_special {
            return "#fff3cd".to_string();
        }

        // Generate color based on token type or content
        match token.token_type.as_deref() {
            Some("numeric") => "#d1ecf1".to_string(),
            Some("alphabetic") => "#d4edda".to_string(),
            Some("punctuation") => "#f8d7da".to_string(),
            Some("whitespace") => "#f1f3f4".to_string(),
            _ => "#e9ecef".to_string(),
        }
    }

    /// Generate plain text visualization
    pub fn to_text(&self, visualization: &TokenVisualization) -> String {
        let mut text = String::new();

        text.push_str("=== Token Visualization ===\n\n");

        text.push_str("Original Text:\n");
        text.push_str(&visualization.original_text);
        text.push_str("\n\n");

        text.push_str("Tokens:\n");
        for (i, token) in visualization.tokens.iter().enumerate() {
            text.push_str(&format!("{:3}: ", i));
            if self.config.show_token_ids {
                text.push_str(&format!("[{}] ", token.token_id));
            }
            text.push_str(&format!("\"{}\"", token.token));
            if token.is_special {
                text.push_str(" (SPECIAL)");
            }
            text.push('\n');
        }

        text.push_str("\nStatistics:\n");
        let stats = &visualization.statistics;
        text.push_str(&format!("  Total Tokens: {}\n", stats.total_tokens));
        text.push_str(&format!("  Unique Tokens: {}\n", stats.unique_tokens));
        text.push_str(&format!(
            "  Special Tokens: {}\n",
            stats.special_tokens_count
        ));
        text.push_str(&format!(
            "  Average Token Length: {:.2}\n",
            stats.average_token_length
        ));
        text.push_str(&format!(
            "  Compression Ratio: {:.4}\n",
            stats.compression_ratio
        ));
        text.push_str(&format!("  OOV Count: {}\n", stats.oov_count));

        if !stats.token_type_distribution.is_empty() {
            text.push_str("\nToken Type Distribution:\n");
            for (token_type, count) in &stats.token_type_distribution {
                text.push_str(&format!("  {}: {}\n", token_type, count));
            }
        }

        text
    }

    /// Export visualization to JSON
    pub fn to_json(&self, visualization: &TokenVisualization) -> Result<String> {
        serde_json::to_string_pretty(visualization).map_err(|e| {
            TrustformersError::other(
                anyhow::anyhow!("Failed to serialize to JSON: {}", e).to_string(),
            )
        })
    }

    /// Generate comparison report
    pub fn comparison_report(&self, comparison: &TokenizerComparison) -> String {
        let mut report = String::new();

        report.push_str("=== Tokenizer Comparison Report ===\n\n");

        report.push_str("Original Text:\n");
        report.push_str(&comparison.original_text);
        report.push_str("\n\n");

        report.push_str("Tokenization Results:\n");
        for (name, viz) in &comparison.tokenizations {
            report.push_str(&format!(
                "\n{} ({} tokens):\n",
                name, viz.statistics.total_tokens
            ));
            for token in &viz.tokens {
                report.push_str(&format!("  \"{}\"", token.token));
            }
            report.push('\n');
        }

        report.push_str("\nComparison Statistics:\n");
        let stats = &comparison.comparison_stats;
        report.push_str(&format!(
            "  Token Count Variance: {:.2}\n",
            stats.token_count_variance
        ));
        report.push_str(&format!("  Common Tokens: {}\n", stats.common_tokens.len()));

        if !stats.common_tokens.is_empty() {
            report.push_str("    ");
            for (i, token) in stats.common_tokens.iter().enumerate() {
                if i > 0 {
                    report.push_str(", ");
                }
                report.push_str(&format!("\"{}\"", token));
                if i >= 10 {
                    report.push_str("...");
                    break;
                }
            }
            report.push('\n');
        }

        report.push_str("\nSimilarity Scores (Jaccard):\n");
        for (name1, scores) in &stats.similarity_scores {
            for (name2, score) in scores {
                report.push_str(&format!("  {} vs {}: {:.3}\n", name1, name2, score));
            }
        }

        report
    }
}

/// HTML escape function
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
    use crate::char::CharTokenizer;

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
    fn test_visualization_creation() {
        let tokenizer = create_test_char_tokenizer();
        let visualizer = TokenVisualizer::default();

        let result = visualizer.visualize(&tokenizer, "Hello world!").unwrap();

        assert_eq!(result.original_text, "Hello world!");
        assert!(!result.tokens.is_empty());
        assert!(result.statistics.total_tokens > 0);
    }

    #[test]
    fn test_html_generation() {
        let tokenizer = create_test_char_tokenizer();
        let visualizer = TokenVisualizer::default();

        let visualization = visualizer.visualize(&tokenizer, "Hello").unwrap();
        let html = visualizer.to_html(&visualization);

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("Token Visualization"));
        assert!(html.contains("Hello"));
    }

    #[test]
    fn test_text_generation() {
        let tokenizer = create_test_char_tokenizer();
        let visualizer = TokenVisualizer::default();

        let visualization = visualizer.visualize(&tokenizer, "Hello").unwrap();
        let text = visualizer.to_text(&visualization);

        assert!(text.contains("=== Token Visualization ==="));
        assert!(text.contains("Hello"));
        assert!(text.contains("Statistics:"));
    }

    #[test]
    fn test_json_export() {
        let tokenizer = create_test_char_tokenizer();
        let visualizer = TokenVisualizer::default();

        let visualization = visualizer.visualize(&tokenizer, "Hi").unwrap();
        let json = visualizer.to_json(&visualization).unwrap();

        assert!(json.contains("original_text"));
        assert!(json.contains("tokens"));
        assert!(json.contains("statistics"));
    }

    #[test]
    fn test_tokenizer_comparison() {
        let char_tokenizer = create_test_char_tokenizer();
        let tokenizer2 = create_test_char_tokenizer();

        let mut tokenizers = HashMap::new();
        tokenizers.insert("char1".to_string(), &char_tokenizer);
        tokenizers.insert("char2".to_string(), &tokenizer2);

        let visualizer = TokenVisualizer::default();
        let comparison = visualizer.compare_tokenizers(tokenizers, "Hello").unwrap();

        assert_eq!(comparison.original_text, "Hello");
        assert_eq!(comparison.tokenizations.len(), 2);
        assert!(comparison.comparison_stats.similarity_scores.contains_key("char1"));
    }
}
