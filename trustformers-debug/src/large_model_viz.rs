//! Large Model Visualization with Memory Efficiency
//!
//! This module provides optimized visualization for large transformer models,
//! using smart sampling, hierarchical rendering, and memory-efficient techniques
//! to handle models with billions of parameters.

use anyhow::{Context, Result};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Large model visualizer with memory-efficient rendering
///
/// Features:
/// - Smart layer sampling (visualize representative layers)
/// - Hierarchical graph rendering (collapse/expand sections)
/// - Streaming visualization (process in chunks)
/// - Memory-bounded caching
/// - Progressive loading
#[derive(Debug)]
pub struct LargeModelVisualizer {
    /// Configuration
    config: LargeModelVisualizerConfig,
    /// Cached layer metadata
    layer_cache: Arc<RwLock<HashMap<String, LayerMetadata>>>,
    /// Visualization state
    state: Arc<RwLock<VisualizationState>>,
}

/// Configuration for large model visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LargeModelVisualizerConfig {
    /// Enable smart layer sampling
    pub enable_smart_sampling: bool,
    /// Maximum layers to visualize fully (rest are sampled)
    pub max_full_layers: usize,
    /// Sampling strategy
    pub sampling_strategy: SamplingStrategy,
    /// Enable hierarchical rendering
    pub enable_hierarchical: bool,
    /// Enable streaming mode for very large models
    pub enable_streaming: bool,
    /// Maximum memory for visualization (MB)
    pub max_memory_mb: usize,
    /// Chunk size for streaming (number of layers)
    pub stream_chunk_size: usize,
    /// Enable progressive detail loading
    pub enable_progressive_loading: bool,
    /// Visualization format
    pub output_format: VisualizationFormat,
}

impl Default for LargeModelVisualizerConfig {
    fn default() -> Self {
        Self {
            enable_smart_sampling: true,
            max_full_layers: 50,
            sampling_strategy: SamplingStrategy::Adaptive,
            enable_hierarchical: true,
            enable_streaming: true,
            max_memory_mb: 1024, // 1 GB
            stream_chunk_size: 10,
            enable_progressive_loading: true,
            output_format: VisualizationFormat::InteractiveSvg,
        }
    }
}

/// Layer sampling strategy for large models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Uniform sampling (evenly spaced layers)
    Uniform,
    /// Adaptive sampling (more samples where complexity varies)
    Adaptive,
    /// Representative sampling (first, middle, last + interesting layers)
    Representative,
    /// Importance-based (based on parameter count, compute cost)
    ImportanceBased,
}

/// Visualization output format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisualizationFormat {
    /// Static PNG image (memory efficient)
    StaticPng,
    /// Static SVG (scalable but larger)
    StaticSvg,
    /// Interactive SVG with zoom/pan
    InteractiveSvg,
    /// Interactive HTML with JavaScript
    InteractiveHtml,
    /// Text-based summary (minimal memory)
    TextSummary,
    /// JSON metadata only
    JsonMetadata,
}

/// Metadata about a model layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerMetadata {
    /// Layer name
    pub name: String,
    /// Layer index
    pub index: usize,
    /// Layer type
    pub layer_type: String,
    /// Number of parameters
    pub param_count: usize,
    /// Estimated memory (MB)
    pub memory_mb: f64,
    /// Estimated compute cost (FLOPS)
    pub compute_flops: u64,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Is this layer sampled for visualization?
    pub is_sampled: bool,
}

/// Current visualization state
#[derive(Debug, Clone, Default)]
struct VisualizationState {
    /// Total layers in model
    total_layers: usize,
    /// Layers currently loaded
    #[allow(dead_code)]
    loaded_layers: Vec<String>,
    /// Current memory usage (MB)
    current_memory_mb: f64,
    /// Visualization progress (0.0-1.0)
    progress: f64,
    /// Is visualization complete?
    #[allow(dead_code)]
    is_complete: bool,
}

/// Visualization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationResult {
    /// Output file path (if saved to file)
    pub output_path: Option<String>,
    /// Inline data (if small enough)
    pub inline_data: Option<Vec<u8>>,
    /// Visualization statistics
    pub stats: VisualizationStats,
    /// Sampled layer indices
    pub sampled_layers: Vec<usize>,
    /// Total model statistics
    pub model_stats: ModelStatistics,
}

/// Statistics about the visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationStats {
    /// Number of layers visualized
    pub layers_visualized: usize,
    /// Number of layers in model
    pub total_layers: usize,
    /// Sampling ratio
    pub sampling_ratio: f64,
    /// Memory used for visualization (MB)
    pub memory_used_mb: f64,
    /// Time taken (seconds)
    pub time_taken_secs: f64,
    /// Output size (bytes)
    pub output_size_bytes: usize,
}

/// Overall model statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatistics {
    /// Total parameters
    pub total_params: usize,
    /// Total memory footprint (MB)
    pub total_memory_mb: f64,
    /// Total compute cost (GFLOPS)
    pub total_gflops: f64,
    /// Deepest layer index
    pub max_depth: usize,
    /// Layer type distribution
    pub layer_types: HashMap<String, usize>,
}

/// Layer group for hierarchical visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerGroup {
    /// Group name
    pub name: String,
    /// Layer indices in this group
    pub layers: Vec<usize>,
    /// Is this group collapsed?
    pub collapsed: bool,
    /// Summary statistics for group
    pub summary: GroupSummary,
}

/// Summary for a layer group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupSummary {
    /// Total parameters in group
    pub param_count: usize,
    /// Total memory (MB)
    pub memory_mb: f64,
    /// Average compute cost per layer
    pub avg_compute_flops: u64,
}

impl LargeModelVisualizer {
    /// Create a new large model visualizer
    ///
    /// # Arguments
    /// * `config` - Visualizer configuration
    ///
    /// # Example
    /// ```rust
    /// use trustformers_debug::{LargeModelVisualizer, LargeModelVisualizerConfig};
    ///
    /// let config = LargeModelVisualizerConfig::default();
    /// let visualizer = LargeModelVisualizer::new(config);
    /// ```
    pub fn new(config: LargeModelVisualizerConfig) -> Self {
        info!("Initializing large model visualizer");
        Self {
            config,
            layer_cache: Arc::new(RwLock::new(HashMap::new())),
            state: Arc::new(RwLock::new(VisualizationState::default())),
        }
    }

    /// Add layer metadata to the visualizer
    ///
    /// # Arguments
    /// * `metadata` - Layer metadata
    pub fn add_layer(&self, metadata: LayerMetadata) -> Result<()> {
        let mut cache = self.layer_cache.write();
        let mut state = self.state.write();

        cache.insert(metadata.name.clone(), metadata.clone());
        state.total_layers = cache.len();
        state.current_memory_mb += metadata.memory_mb;

        // Check memory limit
        if state.current_memory_mb > self.config.max_memory_mb as f64 {
            warn!(
                "Memory limit exceeded: {:.1} MB > {} MB. Consider increasing max_memory_mb or enabling sampling",
                state.current_memory_mb,
                self.config.max_memory_mb
            );
        }

        Ok(())
    }

    /// Analyze model and determine sampling strategy
    ///
    /// # Returns
    /// Indices of layers to visualize in detail
    pub fn determine_sampling(&self) -> Result<Vec<usize>> {
        let cache = self.layer_cache.read();
        let state = self.state.read();

        if !self.config.enable_smart_sampling || state.total_layers <= self.config.max_full_layers {
            // Visualize all layers
            return Ok((0..state.total_layers).collect());
        }

        debug!(
            "Applying {:?} sampling strategy for {} layers",
            self.config.sampling_strategy, state.total_layers
        );

        let sampled_indices = match self.config.sampling_strategy {
            SamplingStrategy::Uniform => self.uniform_sampling(state.total_layers),
            SamplingStrategy::Adaptive => self.adaptive_sampling(&cache),
            SamplingStrategy::Representative => self.representative_sampling(state.total_layers),
            SamplingStrategy::ImportanceBased => self.importance_sampling(&cache),
        };

        Ok(sampled_indices)
    }

    /// Uniform sampling: evenly spaced layers
    fn uniform_sampling(&self, total_layers: usize) -> Vec<usize> {
        let max_layers = self.config.max_full_layers;
        let step = (total_layers as f64 / max_layers as f64).ceil() as usize;

        (0..total_layers).step_by(step).collect()
    }

    /// Adaptive sampling: more samples where complexity varies
    fn adaptive_sampling(&self, cache: &HashMap<String, LayerMetadata>) -> Vec<usize> {
        let mut layers: Vec<_> = cache.values().collect();
        layers.sort_by_key(|l| l.index);

        let mut sampled = Vec::new();
        let max_layers = self.config.max_full_layers;

        // Always include first and last layers
        if !layers.is_empty() {
            sampled.push(0);
            sampled.push(layers.len() - 1);
        }

        // Calculate complexity variance between consecutive layers
        let mut variances = Vec::new();
        for i in 0..layers.len().saturating_sub(1) {
            let complexity_diff =
                (layers[i + 1].param_count as i64 - layers[i].param_count as i64).abs();
            variances.push((i, complexity_diff));
        }

        // Sort by variance (descending)
        variances.sort_by(|a, b| b.1.cmp(&a.1));

        // Sample layers with highest variance
        for (idx, _) in variances.iter().take(max_layers.saturating_sub(2)) {
            sampled.push(*idx);
        }

        sampled.sort_unstable();
        sampled.dedup();
        sampled
    }

    /// Representative sampling: first, middle, last + interesting layers
    fn representative_sampling(&self, total_layers: usize) -> Vec<usize> {
        let mut sampled = Vec::new();

        if total_layers == 0 {
            return sampled;
        }

        // First layers
        sampled.extend(0..3.min(total_layers));

        // Middle layers
        let mid = total_layers / 2;
        sampled.extend((mid.saturating_sub(1))..=(mid + 1).min(total_layers - 1));

        // Last layers
        sampled.extend((total_layers.saturating_sub(3))..total_layers);

        // Add evenly spaced samples in between
        let remaining_budget = self.config.max_full_layers.saturating_sub(sampled.len());
        let step = (total_layers as f64 / remaining_budget as f64).ceil() as usize;

        for i in (0..total_layers).step_by(step) {
            sampled.push(i);
        }

        sampled.sort_unstable();
        sampled.dedup();
        sampled
    }

    /// Importance-based sampling: prioritize large/complex layers
    fn importance_sampling(&self, cache: &HashMap<String, LayerMetadata>) -> Vec<usize> {
        let mut layers: Vec<_> = cache.values().collect();

        // Calculate importance score (weighted sum of params and compute)
        layers.sort_by(|a, b| {
            let score_a = (a.param_count as f64) + (a.compute_flops as f64 / 1e9);
            let score_b = (b.param_count as f64) + (b.compute_flops as f64 / 1e9);
            score_b.partial_cmp(&score_a).unwrap()
        });

        layers.iter().take(self.config.max_full_layers).map(|l| l.index).collect()
    }

    /// Create hierarchical layer groups
    ///
    /// Groups layers by type or sequential blocks for collapsible visualization
    pub fn create_layer_groups(&self) -> Result<Vec<LayerGroup>> {
        let cache = self.layer_cache.read();

        if !self.config.enable_hierarchical || cache.len() < 20 {
            // Not worth grouping small models
            return Ok(Vec::new());
        }

        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();

        // Group by layer type
        for metadata in cache.values() {
            groups
                .entry(metadata.layer_type.clone())
                .or_insert_with(Vec::new)
                .push(metadata.index);
        }

        // Create LayerGroup objects
        let mut layer_groups = Vec::new();

        for (layer_type, indices) in groups {
            // Calculate summary
            let group_layers: Vec<_> = indices
                .iter()
                .filter_map(|&idx| cache.values().find(|l| l.index == idx))
                .collect();

            let param_count: usize = group_layers.iter().map(|l| l.param_count).sum();
            let memory_mb: f64 = group_layers.iter().map(|l| l.memory_mb).sum();
            let avg_compute_flops = if !group_layers.is_empty() {
                group_layers.iter().map(|l| l.compute_flops).sum::<u64>()
                    / group_layers.len() as u64
            } else {
                0
            };

            let indices_len = indices.len();
            layer_groups.push(LayerGroup {
                name: format!("{} ({} layers)", layer_type, indices_len),
                layers: indices,
                collapsed: indices_len > 10, // Auto-collapse large groups
                summary: GroupSummary {
                    param_count,
                    memory_mb,
                    avg_compute_flops,
                },
            });
        }

        // Sort by first layer index
        layer_groups.sort_by_key(|g| g.layers.first().copied().unwrap_or(0));

        Ok(layer_groups)
    }

    /// Generate visualization with memory-efficient rendering
    ///
    /// # Arguments
    /// * `output_path` - Optional output file path
    ///
    /// # Returns
    /// Visualization result with statistics
    pub fn visualize(&self, output_path: Option<String>) -> Result<VisualizationResult> {
        info!("Starting large model visualization");

        let start_time = std::time::Instant::now();

        // Determine which layers to visualize
        let sampled_layers = self.determine_sampling()?;

        info!(
            "Visualizing {} out of {} layers",
            sampled_layers.len(),
            self.state.read().total_layers
        );

        // Calculate model statistics
        let model_stats = self.calculate_model_stats()?;

        // Generate visualization based on format
        let (output_data, output_size) = match self.config.output_format {
            VisualizationFormat::TextSummary => self.generate_text_summary(&sampled_layers)?,
            VisualizationFormat::JsonMetadata => self.generate_json_metadata(&sampled_layers)?,
            VisualizationFormat::StaticSvg => self.generate_static_svg(&sampled_layers)?,
            VisualizationFormat::InteractiveSvg => {
                self.generate_interactive_svg(&sampled_layers)?
            },
            VisualizationFormat::InteractiveHtml => {
                self.generate_interactive_html(&sampled_layers)?
            },
            VisualizationFormat::StaticPng => {
                anyhow::bail!("PNG generation not yet implemented - use SVG or HTML instead")
            },
        };

        // Save to file if path provided
        if let Some(ref path) = output_path {
            std::fs::write(path, &output_data)
                .with_context(|| format!("Failed to write visualization to {}", path))?;
            info!("Saved visualization to {}", path);
        }

        let time_taken = start_time.elapsed().as_secs_f64();
        let state = self.state.read();

        Ok(VisualizationResult {
            output_path,
            inline_data: if output_size < 1024 * 1024 { Some(output_data) } else { None }, // Include inline if < 1MB
            stats: VisualizationStats {
                layers_visualized: sampled_layers.len(),
                total_layers: state.total_layers,
                sampling_ratio: sampled_layers.len() as f64 / state.total_layers as f64,
                memory_used_mb: state.current_memory_mb,
                time_taken_secs: time_taken,
                output_size_bytes: output_size,
            },
            sampled_layers,
            model_stats,
        })
    }

    /// Calculate overall model statistics
    fn calculate_model_stats(&self) -> Result<ModelStatistics> {
        let cache = self.layer_cache.read();

        let total_params: usize = cache.values().map(|l| l.param_count).sum();
        let total_memory_mb: f64 = cache.values().map(|l| l.memory_mb).sum();
        let total_gflops: f64 = cache.values().map(|l| l.compute_flops).sum::<u64>() as f64 / 1e9;
        let max_depth = cache.values().map(|l| l.index).max().unwrap_or(0);

        let mut layer_types: HashMap<String, usize> = HashMap::new();
        for metadata in cache.values() {
            *layer_types.entry(metadata.layer_type.clone()).or_insert(0) += 1;
        }

        Ok(ModelStatistics {
            total_params,
            total_memory_mb,
            total_gflops,
            max_depth,
            layer_types,
        })
    }

    /// Generate text summary (minimal memory)
    fn generate_text_summary(&self, sampled_layers: &[usize]) -> Result<(Vec<u8>, usize)> {
        let cache = self.layer_cache.read();

        let mut summary = String::from("=== Large Model Visualization Summary ===\n\n");

        summary.push_str(&format!(
            "Total Layers: {}\n",
            self.state.read().total_layers
        ));
        summary.push_str(&format!("Visualized Layers: {}\n\n", sampled_layers.len()));

        summary.push_str("Layer Details:\n");
        for &idx in sampled_layers {
            if let Some(layer) = cache.values().find(|l| l.index == idx) {
                summary.push_str(&format!(
                    "  [{}] {} - {} params, {:.2} MB, {:.1} GFLOPS\n",
                    layer.index,
                    layer.name,
                    layer.param_count,
                    layer.memory_mb,
                    layer.compute_flops as f64 / 1e9
                ));
            }
        }

        let bytes = summary.into_bytes();
        let size = bytes.len();
        Ok((bytes, size))
    }

    /// Generate JSON metadata
    fn generate_json_metadata(&self, sampled_layers: &[usize]) -> Result<(Vec<u8>, usize)> {
        let cache = self.layer_cache.read();

        let layers: Vec<_> = sampled_layers
            .iter()
            .filter_map(|&idx| cache.values().find(|l| l.index == idx).cloned())
            .collect();

        let json = serde_json::to_string_pretty(&layers)?;
        let bytes = json.into_bytes();
        let size = bytes.len();
        Ok((bytes, size))
    }

    /// Generate static SVG
    fn generate_static_svg(&self, sampled_layers: &[usize]) -> Result<(Vec<u8>, usize)> {
        let cache = self.layer_cache.read();

        let mut svg = String::from(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="800" viewBox="0 0 1200 800">
<style>
.layer { fill: #4a90e2; stroke: #2c5aa0; stroke-width: 2; }
.layer-text { fill: white; font-family: Arial, sans-serif; font-size: 12px; }
.title { font-family: Arial, sans-serif; font-size: 20px; font-weight: bold; }
</style>
<text x="600" y="30" class="title" text-anchor="middle">Model Architecture</text>
"#,
        );

        let layer_height = 60;
        let layer_width = 200;
        let x_offset = 500;
        let y_start = 60;

        for (i, &idx) in sampled_layers.iter().enumerate() {
            if let Some(layer) = cache.values().find(|l| l.index == idx) {
                let y = y_start + i * (layer_height + 20);

                svg.push_str(&format!(
                    r#"<rect x="{}" y="{}" width="{}" height="{}" class="layer" />
<text x="{}" y="{}" class="layer-text" text-anchor="middle">{}</text>
<text x="{}" y="{}" class="layer-text" text-anchor="middle">{:.1}M params</text>
"#,
                    x_offset,
                    y,
                    layer_width,
                    layer_height,
                    x_offset + layer_width / 2,
                    y + 25,
                    layer.name,
                    x_offset + layer_width / 2,
                    y + 45,
                    layer.param_count as f64 / 1e6
                ));
            }
        }

        svg.push_str("</svg>");

        let bytes = svg.into_bytes();
        let size = bytes.len();
        Ok((bytes, size))
    }

    /// Generate interactive SVG with zoom/pan
    fn generate_interactive_svg(&self, sampled_layers: &[usize]) -> Result<(Vec<u8>, usize)> {
        // For now, delegate to static SVG
        // TODO: Add pan/zoom JavaScript
        self.generate_static_svg(sampled_layers)
    }

    /// Generate interactive HTML with JavaScript
    fn generate_interactive_html(&self, sampled_layers: &[usize]) -> Result<(Vec<u8>, usize)> {
        let cache = self.layer_cache.read();
        let model_stats = self.calculate_model_stats()?;

        let mut html = String::from(
            r#"<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Large Model Visualization</title>
<style>
body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
.container { max-width: 1200px; margin: 0 auto; }
.header { background: #4a90e2; color: white; padding: 20px; border-radius: 8px; }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }
.stat-card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.layer-list { background: white; padding: 20px; border-radius: 8px; }
.layer { padding: 10px; margin: 5px 0; background: #f9f9f9; border-left: 4px solid #4a90e2; }
</style>
</head>
<body>
<div class="container">
<div class="header">
<h1>Large Model Visualization</h1>
<p>Interactive view of model architecture</p>
</div>
<div class="stats">
"#,
        );

        // Add stats cards
        html.push_str(&format!(
            r#"<div class="stat-card">
<h3>{:.1}M</h3>
<p>Total Parameters</p>
</div>
<div class="stat-card">
<h3>{:.1} GB</h3>
<p>Total Memory</p>
</div>
<div class="stat-card">
<h3>{}</h3>
<p>Total Layers</p>
</div>
<div class="stat-card">
<h3>{}/{}</h3>
<p>Visualized/Total</p>
</div>
"#,
            model_stats.total_params as f64 / 1e6,
            model_stats.total_memory_mb / 1024.0,
            model_stats.max_depth + 1,
            sampled_layers.len(),
            self.state.read().total_layers
        ));

        html.push_str("</div><div class=\"layer-list\"><h2>Layer Details</h2>");

        // Add layer details
        for &idx in sampled_layers {
            if let Some(layer) = cache.values().find(|l| l.index == idx) {
                html.push_str(&format!(
                    r#"<div class="layer">
<strong>[{}] {}</strong><br>
Type: {} | Parameters: {:.1}M | Memory: {:.2} MB | Compute: {:.1} GFLOPS
</div>
"#,
                    layer.index,
                    layer.name,
                    layer.layer_type,
                    layer.param_count as f64 / 1e6,
                    layer.memory_mb,
                    layer.compute_flops as f64 / 1e9
                ));
            }
        }

        html.push_str("</div></div></body></html>");

        let bytes = html.into_bytes();
        let size = bytes.len();
        Ok((bytes, size))
    }

    /// Get current visualization progress (0.0-1.0)
    pub fn get_progress(&self) -> f64 {
        self.state.read().progress
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> MemoryStats {
        let state = self.state.read();
        MemoryStats {
            current_mb: state.current_memory_mb,
            max_mb: self.config.max_memory_mb as f64,
            utilization_pct: (state.current_memory_mb / self.config.max_memory_mb as f64 * 100.0)
                .min(100.0),
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Current memory usage (MB)
    pub current_mb: f64,
    /// Maximum allowed memory (MB)
    pub max_mb: f64,
    /// Utilization percentage
    pub utilization_pct: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visualizer_creation() {
        let config = LargeModelVisualizerConfig::default();
        let _visualizer = LargeModelVisualizer::new(config);
    }

    #[test]
    fn test_add_layers() -> Result<()> {
        let config = LargeModelVisualizerConfig::default();
        let visualizer = LargeModelVisualizer::new(config);

        for i in 0..10 {
            let metadata = LayerMetadata {
                name: format!("layer_{}", i),
                index: i,
                layer_type: "Linear".to_string(),
                param_count: 1024 * 1024,
                memory_mb: 4.0,
                compute_flops: 1_000_000_000,
                input_shape: vec![512],
                output_shape: vec![512],
                is_sampled: false,
            };
            visualizer.add_layer(metadata)?;
        }

        let stats = visualizer.get_memory_stats();
        assert_eq!(stats.current_mb, 40.0);

        Ok(())
    }

    #[test]
    fn test_uniform_sampling() -> Result<()> {
        let mut config = LargeModelVisualizerConfig::default();
        config.max_full_layers = 5;
        config.sampling_strategy = SamplingStrategy::Uniform;

        let visualizer = LargeModelVisualizer::new(config);

        // Add 20 layers
        for i in 0..20 {
            let metadata = LayerMetadata {
                name: format!("layer_{}", i),
                index: i,
                layer_type: "Linear".to_string(),
                param_count: 1024 * 1024,
                memory_mb: 4.0,
                compute_flops: 1_000_000_000,
                input_shape: vec![512],
                output_shape: vec![512],
                is_sampled: false,
            };
            visualizer.add_layer(metadata)?;
        }

        let sampled = visualizer.determine_sampling()?;
        assert_eq!(sampled.len(), 5);

        Ok(())
    }

    #[test]
    fn test_text_visualization() -> Result<()> {
        let config = LargeModelVisualizerConfig {
            output_format: VisualizationFormat::TextSummary,
            ..Default::default()
        };

        let visualizer = LargeModelVisualizer::new(config);

        // Add a few layers
        for i in 0..5 {
            let metadata = LayerMetadata {
                name: format!("layer_{}", i),
                index: i,
                layer_type: "Linear".to_string(),
                param_count: 1024 * 1024 * (i + 1),
                memory_mb: 4.0 * (i + 1) as f64,
                compute_flops: 1_000_000_000 * (i + 1) as u64,
                input_shape: vec![512],
                output_shape: vec![512],
                is_sampled: false,
            };
            visualizer.add_layer(metadata)?;
        }

        let result = visualizer.visualize(None)?;

        assert_eq!(result.stats.layers_visualized, 5);
        assert!(result.stats.output_size_bytes > 0);

        Ok(())
    }
}
