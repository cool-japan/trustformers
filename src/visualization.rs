use tch::{Tensor, Kind, Device};
use std::collections::HashMap;

pub struct AttentionVisualizer {
    attention_weights: Vec<Vec<Tensor>>, // [layer][head] -> attention weights
    layer_names: Vec<String>,
    head_names: Vec<Vec<String>>,
}

impl AttentionVisualizer {
    pub fn new() -> Self {
        Self {
            attention_weights: Vec::new(),
            layer_names: Vec::new(),
            head_names: Vec::new(),
        }
    }

    pub fn add_layer(&mut self, layer_name: String, num_heads: usize) {
        self.layer_names.push(layer_name);
        self.attention_weights.push(vec![Tensor::new(); num_heads]);

        let head_names: Vec<String> = (0..num_heads)
            .map(|i| format!("head_{}", i))
            .collect();
        self.head_names.push(head_names);
    }

    pub fn record_attention(&mut self, layer_idx: usize, head_idx: usize, attention_weights: &Tensor) {
        if layer_idx < self.attention_weights.len() && head_idx < self.attention_weights[layer_idx].len() {
            self.attention_weights[layer_idx][head_idx] = attention_weights.detach();
        }
    }

    pub fn get_attention_patterns(&self) -> &Vec<Vec<Tensor>> {
        &self.attention_weights
    }

    pub fn compute_attention_statistics(&self) -> AttentionStatistics {
        let mut layer_stats = Vec::new();

        for (layer_idx, layer_weights) in self.attention_weights.iter().enumerate() {
            let mut head_stats = Vec::new();

            for (head_idx, head_weights) in layer_weights.iter().enumerate() {
                if head_weights.numel() > 0 {
                    let stats = HeadStatistics {
                        mean_attention: head_weights.mean(Kind::Float).double_value(&[]),
                        max_attention: head_weights.max().double_value(&[]),
                        min_attention: head_weights.min().double_value(&[]),
                        entropy: self.compute_entropy(head_weights),
                        sparsity: self.compute_sparsity(head_weights, 0.01),
                    };
                    head_stats.push(stats);
                }
            }

            layer_stats.push(LayerStatistics {
                layer_name: self.layer_names[layer_idx].clone(),
                head_statistics: head_stats,
            });
        }

        AttentionStatistics { layer_statistics: layer_stats }
    }

    fn compute_entropy(&self, attention_weights: &Tensor) -> f64 {
        // Compute entropy across the last dimension (attention over sequence)
        let log_probs = attention_weights.log();
        let entropy = -(attention_weights * log_probs).sum_dim_intlist(&[-1], false, Kind::Float);
        entropy.mean(Kind::Float).double_value(&[])
    }

    fn compute_sparsity(&self, attention_weights: &Tensor, threshold: f64) -> f64 {
        let total_elements = attention_weights.numel() as f64;
        let sparse_elements = attention_weights.lt(threshold).sum(Kind::Float).double_value(&[]);
        sparse_elements / total_elements
    }

    pub fn extract_attention_patterns(&self) -> Vec<AttentionPattern> {
        let mut patterns = Vec::new();

        for (layer_idx, layer_weights) in self.attention_weights.iter().enumerate() {
            for (head_idx, head_weights) in layer_weights.iter().enumerate() {
                if head_weights.numel() > 0 {
                    let pattern = self.analyze_pattern(head_weights);
                    patterns.push(AttentionPattern {
                        layer_idx,
                        head_idx,
                        pattern_type: pattern,
                        strength: self.compute_pattern_strength(head_weights, &pattern),
                    });
                }
            }
        }

        patterns
    }

    fn analyze_pattern(&self, attention_weights: &Tensor) -> PatternType {
        let seq_len = attention_weights.size()[-1];

        // Check for diagonal pattern (local attention)
        let diagonal_strength = self.compute_diagonal_strength(attention_weights);

        // Check for uniform pattern (global attention)
        let uniformity = self.compute_uniformity(attention_weights);

        // Check for beginning/end bias
        let beginning_bias = self.compute_position_bias(attention_weights, 0.2); // First 20%
        let end_bias = self.compute_position_bias(attention_weights, 0.8); // Last 20%

        // Determine pattern type based on strengths
        if diagonal_strength > 0.6 {
            PatternType::Local
        } else if uniformity > 0.8 {
            PatternType::Global
        } else if beginning_bias > 0.5 {
            PatternType::BeginningFocused
        } else if end_bias > 0.5 {
            PatternType::EndFocused
        } else {
            PatternType::Mixed
        }
    }

    fn compute_diagonal_strength(&self, attention_weights: &Tensor) -> f64 {
        let dims = attention_weights.size();
        if dims.len() < 2 {
            return 0.0;
        }

        let seq_len_1 = dims[dims.len() - 2];
        let seq_len_2 = dims[dims.len() - 1];
        let min_seq_len = std::cmp::min(seq_len_1, seq_len_2);

        // Extract diagonal elements
        let mut diagonal_sum = 0.0;
        let total_sum = attention_weights.sum(Kind::Float).double_value(&[]);

        // Create a diagonal mask to extract diagonal elements
        let diagonal_mask = Tensor::eye(min_seq_len, (Kind::Bool, attention_weights.device()));

        // Expand mask to match attention tensor dimensions if needed
        let mask = if dims.len() > 2 {
            // Handle batch dimensions by expanding the diagonal mask
            let mut expanded_shape = dims.clone();
            expanded_shape[expanded_shape.len() - 2] = min_seq_len;
            expanded_shape[expanded_shape.len() - 1] = min_seq_len;
            diagonal_mask.expand(&expanded_shape, false)
        } else {
            diagonal_mask
        };

        // Extract diagonal values by masking
        let diagonal_elements = attention_weights.narrow(-2, 0, min_seq_len)
                                                .narrow(-1, 0, min_seq_len)
                                                .masked_select(&mask);

        if diagonal_elements.numel() > 0 {
            diagonal_sum = diagonal_elements.sum(Kind::Float).double_value(&[]);

            // Return ratio of diagonal attention to total attention
            if total_sum > 0.0 {
                diagonal_sum / total_sum
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    fn compute_uniformity(&self, attention_weights: &Tensor) -> f64 {
        // Compute how uniform the attention distribution is
        let mean_attention = attention_weights.mean_dim(&[-1], true, Kind::Float);
        let variance = attention_weights.var_dim(&[-1], false, true, Kind::Float);
        let cv = variance.sqrt() / mean_attention; // Coefficient of variation
        let uniformity = 1.0 / (1.0 + cv.mean(Kind::Float).double_value(&[]));
        uniformity
    }

    fn compute_position_bias(&self, attention_weights: &Tensor, position_ratio: f64) -> f64 {
        let seq_len = attention_weights.size()[-1];
        let position_end = (seq_len as f64 * position_ratio) as i64;

        if position_ratio < 0.5 {
            // Beginning bias
            let beginning_attention = attention_weights.narrow(-1, 0, position_end);
            beginning_attention.mean(Kind::Float).double_value(&[])
        } else {
            // End bias
            let end_start = (seq_len as f64 * position_ratio) as i64;
            let end_attention = attention_weights.narrow(-1, end_start, seq_len - end_start);
            end_attention.mean(Kind::Float).double_value(&[])
        }
    }

    fn compute_pattern_strength(&self, attention_weights: &Tensor, pattern: &PatternType) -> f64 {
        match pattern {
            PatternType::Local => self.compute_diagonal_strength(attention_weights),
            PatternType::Global => self.compute_uniformity(attention_weights),
            PatternType::BeginningFocused => self.compute_position_bias(attention_weights, 0.2),
            PatternType::EndFocused => self.compute_position_bias(attention_weights, 0.8),
            PatternType::Mixed => 0.5, // Default strength for mixed patterns
        }
    }

    pub fn generate_heatmap_data(&self, layer_idx: usize, head_idx: usize) -> Option<Vec<Vec<f64>>> {
        if layer_idx < self.attention_weights.len() && head_idx < self.attention_weights[layer_idx].len() {
            let attention_weights = &self.attention_weights[layer_idx][head_idx];

            if attention_weights.numel() > 0 {
                // Convert tensor to 2D vector for heatmap visualization
                let dims = attention_weights.size();
                if dims.len() >= 2 {
                    let height = dims[dims.len() - 2] as usize;
                    let width = dims[dims.len() - 1] as usize;

                    let mut heatmap_data = vec![vec![0.0; width]; height];

                    // Extract values from tensor - handle different tensor shapes
                    let flattened = if dims.len() > 2 {
                        // Handle batched tensors by taking the first batch
                        attention_weights.narrow(0, 0, 1).squeeze_dim(0)
                    } else {
                        attention_weights.shallow_clone()
                    };

                    // Convert tensor to Vec<f64> for easier indexing
                    let values: Vec<f64> = flattened.to_kind(Kind::Double)
                                                   .flatten(0, -1)
                                                   .into();

                    // Fill heatmap data with actual tensor values
                    for i in 0..height {
                        for j in 0..width {
                            let idx = i * width + j;
                            if idx < values.len() {
                                heatmap_data[i][j] = values[idx];
                            } else {
                                heatmap_data[i][j] = 0.0; // Fallback for out-of-bounds
                            }
                        }
                    }

                    return Some(heatmap_data);
                }
            }
        }
        None
    }

    pub fn clear(&mut self) {
        self.attention_weights.clear();
        self.layer_names.clear();
        self.head_names.clear();
    }
}

#[derive(Debug, Clone)]
pub struct AttentionStatistics {
    pub layer_statistics: Vec<LayerStatistics>,
}

#[derive(Debug, Clone)]
pub struct LayerStatistics {
    pub layer_name: String,
    pub head_statistics: Vec<HeadStatistics>,
}

#[derive(Debug, Clone)]
pub struct HeadStatistics {
    pub mean_attention: f64,
    pub max_attention: f64,
    pub min_attention: f64,
    pub entropy: f64,
    pub sparsity: f64,
}

#[derive(Debug, Clone)]
pub struct AttentionPattern {
    pub layer_idx: usize,
    pub head_idx: usize,
    pub pattern_type: PatternType,
    pub strength: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PatternType {
    Local,           // Attention focused on nearby positions
    Global,          // Uniform attention across all positions
    BeginningFocused, // Strong attention to beginning of sequence
    EndFocused,      // Strong attention to end of sequence
    Mixed,           // No clear pattern
}

pub struct TokenAttentionAnalyzer {
    token_attention_maps: HashMap<String, Vec<f64>>,
}

impl TokenAttentionAnalyzer {
    pub fn new() -> Self {
        Self {
            token_attention_maps: HashMap::new(),
        }
    }

    pub fn analyze_token_attention(&mut self,
                                  tokens: &[String],
                                  attention_weights: &Tensor,
                                  layer_idx: usize,
                                  head_idx: usize) {
        if tokens.len() != attention_weights.size()[-1] as usize {
            return; // Dimension mismatch
        }

        for (token_idx, token) in tokens.iter().enumerate() {
            let key = format!("{}_{}_{}_{}", layer_idx, head_idx, token_idx, token);

            // Extract attention values for this token from the attention weights tensor
            let attention_values = if attention_weights.numel() > 0 {
                // Handle different tensor shapes (batched vs unbatched)
                let processed_tensor = if attention_weights.size().len() > 2 {
                    // For batched tensors [batch, seq_len, seq_len], take first batch
                    attention_weights.narrow(0, 0, 1).squeeze_dim(0)
                } else {
                    attention_weights.shallow_clone()
                };

                // Extract the row corresponding to this token
                if token_idx < processed_tensor.size()[0] as usize {
                    let token_attention_row = processed_tensor.narrow(0, token_idx as i64, 1).squeeze_dim(0);
                    let values: Vec<f64> = token_attention_row.to_kind(Kind::Double).into();
                    values
                } else {
                    // Fallback if token_idx is out of bounds
                    vec![0.0; tokens.len()]
                }
            } else {
                // Fallback for empty tensors
                vec![0.0; tokens.len()]
            };

            self.token_attention_maps.insert(key, attention_values);
        }
    }

    pub fn get_token_attention_summary(&self, token: &str) -> Vec<f64> {
        let mut summary = Vec::new();

        for (key, values) in &self.token_attention_maps {
            if key.contains(token) {
                let avg_attention = values.iter().sum::<f64>() / values.len() as f64;
                summary.push(avg_attention);
            }
        }

        summary
    }

    pub fn find_most_attended_tokens(&self, top_k: usize) -> Vec<(String, f64)> {
        let mut token_scores: HashMap<String, f64> = HashMap::new();

        for (key, values) in &self.token_attention_maps {
            let parts: Vec<&str> = key.split('_').collect();
            if parts.len() >= 4 {
                let token = parts[3];
                let avg_attention = values.iter().sum::<f64>() / values.len() as f64;

                *token_scores.entry(token.to_string()).or_insert(0.0) += avg_attention;
            }
        }

        let mut sorted_tokens: Vec<(String, f64)> = token_scores.into_iter().collect();
        sorted_tokens.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        sorted_tokens.truncate(top_k);

        sorted_tokens
    }
}

pub fn create_attention_hooks() -> AttentionHooks {
    AttentionHooks::new()
}

pub struct AttentionHooks {
    visualizer: AttentionVisualizer,
    enabled: bool,
}

impl AttentionHooks {
    pub fn new() -> Self {
        Self {
            visualizer: AttentionVisualizer::new(),
            enabled: false,
        }
    }

    pub fn enable(&mut self) {
        self.enabled = true;
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn hook_attention(&mut self, layer_name: &str, attention_weights: &Tensor) {
        if !self.enabled {
            return;
        }

        // Extract layer and head information from the attention weights
        let num_heads = attention_weights.size()[1] as usize;

        // Add layer if not exists
        if !self.visualizer.layer_names.contains(&layer_name.to_string()) {
            self.visualizer.add_layer(layer_name.to_string(), num_heads);
        }

        // Find layer index
        if let Some(layer_idx) = self.visualizer.layer_names.iter().position(|name| name == layer_name) {
            // Record attention for each head
            for head_idx in 0..num_heads {
                let head_attention = attention_weights.narrow(1, head_idx as i64, 1).squeeze_dim(1);
                self.visualizer.record_attention(layer_idx, head_idx, &head_attention);
            }
        }
    }

    pub fn get_visualizer(&self) -> &AttentionVisualizer {
        &self.visualizer
    }

    pub fn get_visualizer_mut(&mut self) -> &mut AttentionVisualizer {
        &mut self.visualizer
    }

    pub fn reset(&mut self) {
        self.visualizer.clear();
    }
}

impl Default for AttentionVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for TokenAttentionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for AttentionHooks {
    fn default() -> Self {
        Self::new()
    }
}