//! Attention pattern visualization for transformer models
//!
//! This module provides tools to visualize and analyze attention patterns in transformer
//! models, including multi-head attention, cross-attention, and self-attention mechanisms.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// Note: ndarray types available for advanced array operations if needed

/// Attention pattern visualizer for transformer models
#[derive(Debug)]
pub struct AttentionVisualizer {
    /// Stored attention weights by layer and head
    attention_weights: HashMap<String, AttentionWeights>,
    /// Token vocabularies for labeling
    token_vocab: Option<Vec<String>>,
    /// Configuration
    config: AttentionVisualizerConfig,
}

/// Configuration for attention visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionVisualizerConfig {
    /// Whether to normalize attention weights
    pub normalize: bool,
    /// Minimum attention weight to display (for filtering)
    pub min_weight: f64,
    /// Maximum number of tokens to visualize
    pub max_tokens: usize,
    /// Color scheme for visualization
    pub color_scheme: ColorScheme,
}

/// Color scheme options for visualization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorScheme {
    /// Blue to red gradient
    BlueRed,
    /// Grayscale
    Grayscale,
    /// Viridis color map
    Viridis,
    /// Plasma color map
    Plasma,
}

impl Default for AttentionVisualizerConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            min_weight: 0.01,
            max_tokens: 512,
            color_scheme: ColorScheme::BlueRed,
        }
    }
}

/// Attention weights for a specific layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionWeights {
    /// Layer name
    pub layer_name: String,
    /// Number of attention heads
    pub num_heads: usize,
    /// Attention weights [num_heads, seq_len, seq_len]
    pub weights: Vec<Vec<Vec<f64>>>,
    /// Source tokens (query tokens)
    pub source_tokens: Vec<String>,
    /// Target tokens (key tokens)
    pub target_tokens: Vec<String>,
    /// Attention type
    pub attention_type: AttentionType,
}

/// Type of attention mechanism
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttentionType {
    /// Self-attention (source and target are the same)
    SelfAttention,
    /// Cross-attention (source and target are different)
    CrossAttention,
    /// Encoder-decoder attention
    EncoderDecoderAttention,
}

/// Attention pattern analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionAnalysis {
    /// Layer name
    pub layer_name: String,
    /// Average attention entropy per head
    pub entropy_per_head: Vec<f64>,
    /// Average attention sparsity per head
    pub sparsity_per_head: Vec<f64>,
    /// Most attended tokens (across all heads)
    pub most_attended_tokens: Vec<(usize, f64)>,
    /// Attention flow patterns
    pub flow_patterns: Vec<AttentionFlow>,
}

/// Attention flow between token positions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionFlow {
    /// Source position
    pub from: usize,
    /// Target position
    pub to: usize,
    /// Attention weight
    pub weight: f64,
    /// Head index
    pub head: usize,
}

/// Heatmap data for attention visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionHeatmap {
    /// Layer name
    pub layer_name: String,
    /// Head index
    pub head: usize,
    /// Attention weights matrix
    pub weights: Vec<Vec<f64>>,
    /// Row labels (query tokens)
    pub row_labels: Vec<String>,
    /// Column labels (key tokens)
    pub col_labels: Vec<String>,
}

impl AttentionVisualizer {
    /// Create a new attention visualizer
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_debug::AttentionVisualizer;
    ///
    /// let visualizer = AttentionVisualizer::new();
    /// ```
    pub fn new() -> Self {
        Self {
            attention_weights: HashMap::new(),
            token_vocab: None,
            config: AttentionVisualizerConfig::default(),
        }
    }

    /// Create a new attention visualizer with custom configuration
    pub fn with_config(config: AttentionVisualizerConfig) -> Self {
        Self {
            attention_weights: HashMap::new(),
            token_vocab: None,
            config,
        }
    }

    /// Set token vocabulary for labeling
    pub fn set_token_vocab(&mut self, tokens: Vec<String>) {
        self.token_vocab = Some(tokens);
    }

    /// Register attention weights for a layer
    ///
    /// # Arguments
    ///
    /// * `layer_name` - Name of the layer
    /// * `weights` - Attention weights [num_heads, seq_len, seq_len]
    /// * `source_tokens` - Source (query) tokens
    /// * `target_tokens` - Target (key) tokens
    /// * `attention_type` - Type of attention mechanism
    ///
    /// # Example
    ///
    /// ```
    /// # use trustformers_debug::{AttentionVisualizer, AttentionType};
    /// # let mut visualizer = AttentionVisualizer::new();
    /// let weights = vec![
    ///     vec![vec![0.5, 0.3, 0.2], vec![0.1, 0.6, 0.3], vec![0.2, 0.3, 0.5]]
    /// ];
    /// let tokens = vec!["Hello".to_string(), "world".to_string(), "!".to_string()];
    ///
    /// visualizer.register(
    ///     "layer.0.attention",
    ///     weights,
    ///     tokens.clone(),
    ///     tokens.clone(),
    ///     AttentionType::SelfAttention,
    /// ).unwrap();
    /// ```
    pub fn register(
        &mut self,
        layer_name: &str,
        weights: Vec<Vec<Vec<f64>>>,
        source_tokens: Vec<String>,
        target_tokens: Vec<String>,
        attention_type: AttentionType,
    ) -> Result<()> {
        let num_heads = weights.len();

        let attention_weights = AttentionWeights {
            layer_name: layer_name.to_string(),
            num_heads,
            weights,
            source_tokens,
            target_tokens,
            attention_type,
        };

        self.attention_weights.insert(layer_name.to_string(), attention_weights);

        Ok(())
    }

    /// Get attention weights for a specific layer
    pub fn get_attention(&self, layer_name: &str) -> Option<&AttentionWeights> {
        self.attention_weights.get(layer_name)
    }

    /// Create a heatmap for a specific head in a layer
    pub fn create_heatmap(&self, layer_name: &str, head: usize) -> Result<AttentionHeatmap> {
        let attention = self
            .attention_weights
            .get(layer_name)
            .ok_or_else(|| anyhow::anyhow!("Layer {} not found", layer_name))?;

        if head >= attention.num_heads {
            anyhow::bail!(
                "Head {} out of range (max: {})",
                head,
                attention.num_heads - 1
            );
        }

        let weights = &attention.weights[head];

        Ok(AttentionHeatmap {
            layer_name: layer_name.to_string(),
            head,
            weights: weights.clone(),
            row_labels: attention.source_tokens.clone(),
            col_labels: attention.target_tokens.clone(),
        })
    }

    /// Analyze attention patterns for a layer
    pub fn analyze(&self, layer_name: &str) -> Result<AttentionAnalysis> {
        let attention = self
            .attention_weights
            .get(layer_name)
            .ok_or_else(|| anyhow::anyhow!("Layer {} not found", layer_name))?;

        let entropy_per_head = attention
            .weights
            .iter()
            .map(|head_weights| compute_entropy(head_weights))
            .collect();

        let sparsity_per_head = attention
            .weights
            .iter()
            .map(|head_weights| compute_sparsity(head_weights, self.config.min_weight))
            .collect();

        let most_attended_tokens = find_most_attended_tokens(&attention.weights);

        let flow_patterns = extract_attention_flows(&attention.weights, self.config.min_weight);

        Ok(AttentionAnalysis {
            layer_name: layer_name.to_string(),
            entropy_per_head,
            sparsity_per_head,
            most_attended_tokens,
            flow_patterns,
        })
    }

    /// Plot attention heatmap as ASCII art
    pub fn plot_heatmap_ascii(&self, layer_name: &str, head: usize) -> Result<String> {
        let heatmap = self.create_heatmap(layer_name, head)?;

        let mut output = String::new();
        output.push_str(&format!(
            "Attention Heatmap: {} (Head {})\n",
            layer_name, head
        ));
        output.push_str(&"=".repeat(60));
        output.push('\n');

        // Limit display size for readability
        let max_display = 20;
        let display_rows = heatmap.row_labels.len().min(max_display);
        let display_cols = heatmap.col_labels.len().min(max_display);

        // Column headers
        output.push_str("        ");
        for col in 0..display_cols {
            output.push_str(&format!(
                "{:4}",
                heatmap.col_labels[col].chars().next().unwrap_or('?')
            ));
        }
        output.push('\n');

        // Rows with values
        for row in 0..display_rows {
            let label = &heatmap.row_labels[row];
            output.push_str(&format!(
                "{:6}  ",
                label.chars().take(6).collect::<String>()
            ));

            for col in 0..display_cols {
                let weight = heatmap.weights[row][col];
                let symbol = weight_to_symbol(weight);
                output.push_str(&format!("{:4}", symbol));
            }
            output.push('\n');
        }

        if display_rows < heatmap.row_labels.len() || display_cols < heatmap.col_labels.len() {
            output.push_str(&format!(
                "\n(Showing {}/{} rows, {}/{} cols)\n",
                display_rows,
                heatmap.row_labels.len(),
                display_cols,
                heatmap.col_labels.len()
            ));
        }

        Ok(output)
    }

    /// Export attention weights to JSON
    pub fn export_to_json(&self, layer_name: &str, output_path: &Path) -> Result<()> {
        let attention = self
            .attention_weights
            .get(layer_name)
            .ok_or_else(|| anyhow::anyhow!("Layer {} not found", layer_name))?;

        let json = serde_json::to_string_pretty(attention)?;
        std::fs::write(output_path, json)?;

        Ok(())
    }

    /// Export to BertViz-compatible format (HTML)
    pub fn export_to_bertviz(&self, layer_name: &str, output_path: &Path) -> Result<()> {
        let attention = self
            .attention_weights
            .get(layer_name)
            .ok_or_else(|| anyhow::anyhow!("Layer {} not found", layer_name))?;

        let mut html =
            String::from("<html><head><title>Attention Visualization</title></head><body>");
        html.push_str(&format!("<h1>{}</h1>", layer_name));

        for head in 0..attention.num_heads {
            html.push_str(&format!("<h2>Head {}</h2>", head));
            html.push_str("<table border='1'><tr><th></th>");

            // Column headers
            for token in &attention.target_tokens {
                html.push_str(&format!("<th>{}</th>", html_escape(token)));
            }
            html.push_str("</tr>");

            // Rows
            for (row_idx, source_token) in attention.source_tokens.iter().enumerate() {
                html.push_str(&format!("<tr><th>{}</th>", html_escape(source_token)));

                for col_idx in 0..attention.target_tokens.len() {
                    let weight = attention.weights[head][row_idx][col_idx];
                    let color = weight_to_color(weight);
                    html.push_str(&format!(
                        "<td style='background-color: {}'>{:.3}</td>",
                        color, weight
                    ));
                }
                html.push_str("</tr>");
            }

            html.push_str("</table>");
        }

        html.push_str("</body></html>");
        std::fs::write(output_path, html)?;

        Ok(())
    }

    /// Get summary statistics for all layers
    pub fn summary(&self) -> String {
        let mut output = String::new();
        output.push_str("Attention Summary\n");
        output.push_str(&"=".repeat(80));
        output.push('\n');

        for (layer_name, attention) in &self.attention_weights {
            output.push_str(&format!("\nLayer: {}\n", layer_name));
            output.push_str(&format!("  Num Heads: {}\n", attention.num_heads));
            output.push_str(&format!(
                "  Seq Length: {}\n",
                attention.source_tokens.len()
            ));
            output.push_str(&format!(
                "  Attention Type: {:?}\n",
                attention.attention_type
            ));

            if let Ok(analysis) = self.analyze(layer_name) {
                output.push_str(&format!(
                    "  Avg Entropy: {:.4}\n",
                    analysis.entropy_per_head.iter().sum::<f64>()
                        / analysis.entropy_per_head.len() as f64
                ));
                output.push_str(&format!(
                    "  Avg Sparsity: {:.4}\n",
                    analysis.sparsity_per_head.iter().sum::<f64>()
                        / analysis.sparsity_per_head.len() as f64
                ));
            }
        }

        output
    }

    /// Clear all stored attention weights
    pub fn clear(&mut self) {
        self.attention_weights.clear();
    }

    /// Get number of stored layers
    pub fn num_layers(&self) -> usize {
        self.attention_weights.len()
    }
}

impl Default for AttentionVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions

/// Compute entropy of attention distribution
fn compute_entropy(weights: &[Vec<f64>]) -> f64 {
    let mut total_entropy = 0.0;
    let mut count = 0;

    for row in weights {
        let sum: f64 = row.iter().sum();
        if sum > 0.0 {
            let entropy: f64 = row
                .iter()
                .filter(|&&w| w > 0.0)
                .map(|&w| {
                    let p = w / sum;
                    -p * p.log2()
                })
                .sum();
            total_entropy += entropy;
            count += 1;
        }
    }

    if count > 0 {
        total_entropy / count as f64
    } else {
        0.0
    }
}

/// Compute sparsity (fraction of weights below threshold)
fn compute_sparsity(weights: &[Vec<f64>], threshold: f64) -> f64 {
    let total_weights: usize = weights.iter().map(|row| row.len()).sum();
    let sparse_weights: usize =
        weights.iter().map(|row| row.iter().filter(|&&w| w < threshold).count()).sum();

    if total_weights > 0 {
        sparse_weights as f64 / total_weights as f64
    } else {
        0.0
    }
}

/// Find most attended token positions
fn find_most_attended_tokens(weights: &[Vec<Vec<f64>>]) -> Vec<(usize, f64)> {
    let seq_len = if !weights.is_empty() && !weights[0].is_empty() {
        weights[0][0].len()
    } else {
        return Vec::new();
    };

    let mut token_attention = vec![0.0; seq_len];

    for head_weights in weights {
        for row in head_weights {
            for (i, &weight) in row.iter().enumerate() {
                token_attention[i] += weight;
            }
        }
    }

    let mut indexed: Vec<_> = token_attention.iter().enumerate().map(|(i, &w)| (i, w)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    indexed.into_iter().take(10).collect()
}

/// Extract significant attention flows
fn extract_attention_flows(weights: &[Vec<Vec<f64>>], threshold: f64) -> Vec<AttentionFlow> {
    let mut flows = Vec::new();

    for (head, head_weights) in weights.iter().enumerate() {
        for (from, row) in head_weights.iter().enumerate() {
            for (to, &weight) in row.iter().enumerate() {
                if weight >= threshold {
                    flows.push(AttentionFlow {
                        from,
                        to,
                        weight,
                        head,
                    });
                }
            }
        }
    }

    flows.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal));
    flows.into_iter().take(100).collect()
}

/// Convert attention weight to ASCII symbol
fn weight_to_symbol(weight: f64) -> &'static str {
    if weight > 0.8 {
        "█"
    } else if weight > 0.6 {
        "▓"
    } else if weight > 0.4 {
        "▒"
    } else if weight > 0.2 {
        "░"
    } else {
        " "
    }
}

/// Convert attention weight to HTML color
fn weight_to_color(weight: f64) -> String {
    let intensity = (weight * 255.0) as u8;
    format!("rgb(255, {}, {})", 255 - intensity, 255 - intensity)
}

/// Escape HTML special characters
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_visualizer_creation() {
        let visualizer = AttentionVisualizer::new();
        assert_eq!(visualizer.num_layers(), 0);
    }

    #[test]
    fn test_register_attention() {
        let mut visualizer = AttentionVisualizer::new();

        let weights = vec![vec![
            vec![0.5, 0.3, 0.2],
            vec![0.1, 0.6, 0.3],
            vec![0.2, 0.3, 0.5],
        ]];

        let tokens = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        visualizer
            .register(
                "layer.0",
                weights,
                tokens.clone(),
                tokens,
                AttentionType::SelfAttention,
            )
            .unwrap();

        assert_eq!(visualizer.num_layers(), 1);
    }

    #[test]
    fn test_create_heatmap() {
        let mut visualizer = AttentionVisualizer::new();

        let weights = vec![vec![
            vec![0.5, 0.3, 0.2],
            vec![0.1, 0.6, 0.3],
            vec![0.2, 0.3, 0.5],
        ]];

        let tokens = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        visualizer
            .register(
                "layer.0",
                weights,
                tokens.clone(),
                tokens,
                AttentionType::SelfAttention,
            )
            .unwrap();

        let heatmap = visualizer.create_heatmap("layer.0", 0).unwrap();
        assert_eq!(heatmap.layer_name, "layer.0");
        assert_eq!(heatmap.head, 0);
        assert_eq!(heatmap.weights.len(), 3);
    }

    #[test]
    fn test_analyze_attention() {
        let mut visualizer = AttentionVisualizer::new();

        let weights = vec![vec![
            vec![0.7, 0.2, 0.1],
            vec![0.1, 0.8, 0.1],
            vec![0.1, 0.1, 0.8],
        ]];

        let tokens = vec!["A".to_string(), "B".to_string(), "C".to_string()];

        visualizer
            .register(
                "layer.0",
                weights,
                tokens.clone(),
                tokens,
                AttentionType::SelfAttention,
            )
            .unwrap();

        let analysis = visualizer.analyze("layer.0").unwrap();
        assert_eq!(analysis.entropy_per_head.len(), 1);
        assert_eq!(analysis.sparsity_per_head.len(), 1);
        assert!(!analysis.most_attended_tokens.is_empty());
    }

    #[test]
    fn test_export_to_json() {
        use std::env;

        let temp_dir = env::temp_dir();
        let output_path = temp_dir.join("attention.json");

        let mut visualizer = AttentionVisualizer::new();
        let weights = vec![vec![vec![1.0]]];
        let tokens = vec!["A".to_string()];

        visualizer
            .register(
                "layer.0",
                weights,
                tokens.clone(),
                tokens,
                AttentionType::SelfAttention,
            )
            .unwrap();

        visualizer.export_to_json("layer.0", &output_path).unwrap();
        assert!(output_path.exists());

        // Clean up
        let _ = std::fs::remove_file(output_path);
    }

    #[test]
    fn test_compute_entropy() {
        let weights = vec![vec![0.5, 0.3, 0.2], vec![1.0, 0.0, 0.0]];

        let entropy = compute_entropy(&weights);
        assert!(entropy > 0.0);
    }

    #[test]
    fn test_compute_sparsity() {
        let weights = vec![vec![0.9, 0.05, 0.05], vec![0.01, 0.01, 0.98]];

        let sparsity = compute_sparsity(&weights, 0.1);
        assert!(sparsity > 0.0);
        assert!(sparsity <= 1.0);
    }
}
