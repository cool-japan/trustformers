//! Custom Visualization Plugin System
//!
//! This module provides an extensible plugin system for custom visualizations,
//! allowing users to create and register their own visualization tools.

use anyhow::Result;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Visualization data that can be passed to plugins
#[derive(Debug, Clone)]
pub enum VisualizationData {
    /// 1D array data
    Array1D(Vec<f64>),
    /// 2D array data
    Array2D(Vec<Vec<f64>>),
    /// Tensor data with shape information
    Tensor { data: Vec<f64>, shape: Vec<usize> },
    /// Key-value pairs (for metadata, metrics, etc.)
    KeyValue(HashMap<String, String>),
    /// Time series data
    TimeSeries {
        timestamps: Vec<f64>,
        values: Vec<f64>,
        labels: Vec<String>,
    },
    /// Custom JSON data
    Json(serde_json::Value),
}

/// Output format for visualizations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputFormat {
    /// PNG image
    Png,
    /// SVG vector graphics
    Svg,
    /// HTML interactive visualization
    Html,
    /// Plain text/ASCII
    Text,
    /// JSON data
    Json,
    /// CSV data
    Csv,
}

/// Plugin capabilities and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin description
    pub description: String,
    /// Author
    pub author: String,
    /// Supported input data types
    pub supported_inputs: Vec<String>,
    /// Supported output formats
    pub supported_outputs: Vec<OutputFormat>,
    /// Tags/categories
    pub tags: Vec<String>,
}

/// Configuration for visualization plugins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Output format
    pub output_format: OutputFormat,
    /// Output path (file or directory)
    pub output_path: Option<String>,
    /// Width in pixels (for image outputs)
    pub width: usize,
    /// Height in pixels (for image outputs)
    pub height: usize,
    /// Color scheme
    pub color_scheme: String,
    /// Additional custom parameters
    pub custom_params: HashMap<String, serde_json::Value>,
}

impl Default for PluginConfig {
    fn default() -> Self {
        Self {
            output_format: OutputFormat::Png,
            output_path: None,
            width: 800,
            height: 600,
            color_scheme: "viridis".to_string(),
            custom_params: HashMap::new(),
        }
    }
}

/// Result of plugin execution
#[derive(Debug)]
pub struct PluginResult {
    /// Success status
    pub success: bool,
    /// Output file path (if file was created)
    pub output_path: Option<String>,
    /// Raw output data (if applicable)
    pub output_data: Option<Vec<u8>>,
    /// Metadata about the visualization
    pub metadata: HashMap<String, String>,
    /// Error message (if failed)
    pub error: Option<String>,
}

/// Trait for visualization plugins
///
/// Implement this trait to create custom visualization plugins.
pub trait VisualizationPlugin: Send + Sync {
    /// Get plugin metadata
    fn metadata(&self) -> PluginMetadata;

    /// Execute the visualization
    ///
    /// # Arguments
    /// * `data` - Input data to visualize
    /// * `config` - Configuration for the visualization
    ///
    /// # Returns
    /// Result containing the plugin output
    fn execute(&self, data: VisualizationData, config: PluginConfig) -> Result<PluginResult>;

    /// Validate input data
    ///
    /// # Arguments
    /// * `data` - Data to validate
    ///
    /// # Returns
    /// True if data is valid for this plugin
    fn validate(&self, data: &VisualizationData) -> bool {
        // Default implementation: accept all data
        let _ = data;
        true
    }

    /// Get configuration schema (for UI generation)
    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {}
        })
    }
}

/// Plugin manager for registering and executing visualization plugins
pub struct PluginManager {
    /// Registered plugins
    plugins: Arc<RwLock<HashMap<String, Box<dyn VisualizationPlugin>>>>,
    /// Plugin execution history
    history: Arc<RwLock<Vec<PluginExecution>>>,
}

/// Record of plugin execution
#[derive(Debug, Clone)]
struct PluginExecution {
    plugin_name: String,
    timestamp: std::time::SystemTime,
    success: bool,
    duration_ms: u128,
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginManager {
    /// Create a new plugin manager
    pub fn new() -> Self {
        let manager = Self {
            plugins: Arc::new(RwLock::new(HashMap::new())),
            history: Arc::new(RwLock::new(Vec::new())),
        };

        // Register built-in plugins
        manager.register_builtin_plugins();

        manager
    }

    /// Register built-in plugins
    fn register_builtin_plugins(&self) {
        // Register histogram plugin
        self.register_plugin(Box::new(HistogramPlugin)).ok();

        // Register heatmap plugin
        self.register_plugin(Box::new(HeatmapPlugin)).ok();

        // Register line plot plugin
        self.register_plugin(Box::new(LinePlotPlugin)).ok();

        // Register scatter plot plugin
        self.register_plugin(Box::new(ScatterPlotPlugin)).ok();
    }

    /// Register a new plugin
    ///
    /// # Arguments
    /// * `plugin` - Plugin to register
    pub fn register_plugin(&self, plugin: Box<dyn VisualizationPlugin>) -> Result<()> {
        let name = plugin.metadata().name.clone();

        self.plugins.write().insert(name.clone(), plugin);

        tracing::info!(plugin_name = %name, "Registered visualization plugin");

        Ok(())
    }

    /// Unregister a plugin
    ///
    /// # Arguments
    /// * `name` - Name of plugin to unregister
    pub fn unregister_plugin(&self, name: &str) -> Result<()> {
        self.plugins.write().remove(name);

        tracing::info!(plugin_name = %name, "Unregistered visualization plugin");

        Ok(())
    }

    /// Get list of registered plugins
    pub fn list_plugins(&self) -> Vec<PluginMetadata> {
        self.plugins.read().values().map(|p| p.metadata()).collect()
    }

    /// Execute a plugin
    ///
    /// # Arguments
    /// * `plugin_name` - Name of plugin to execute
    /// * `data` - Input data
    /// * `config` - Configuration
    pub fn execute(
        &self,
        plugin_name: &str,
        data: VisualizationData,
        config: PluginConfig,
    ) -> Result<PluginResult> {
        let start_time = std::time::Instant::now();

        let result = {
            let plugins = self.plugins.read();
            let plugin = plugins
                .get(plugin_name)
                .ok_or_else(|| anyhow::anyhow!("Plugin not found: {}", plugin_name))?;

            // Validate data
            if !plugin.validate(&data) {
                anyhow::bail!("Invalid data for plugin: {}", plugin_name);
            }

            plugin.execute(data, config)?
        };

        let duration = start_time.elapsed().as_millis();

        // Record execution
        self.history.write().push(PluginExecution {
            plugin_name: plugin_name.to_string(),
            timestamp: std::time::SystemTime::now(),
            success: result.success,
            duration_ms: duration,
        });

        Ok(result)
    }

    /// Get plugin by name
    pub fn get_plugin(&self, name: &str) -> Option<PluginMetadata> {
        self.plugins.read().get(name).map(|p| p.metadata())
    }

    /// Get execution history
    pub fn get_history(&self) -> Vec<String> {
        self.history
            .read()
            .iter()
            .map(|e| {
                format!(
                    "{}: {} ({}ms) - {}",
                    e.timestamp.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
                    e.plugin_name,
                    e.duration_ms,
                    if e.success { "success" } else { "failed" }
                )
            })
            .collect()
    }
}

// ============================================================================
// Built-in Plugins
// ============================================================================

/// Histogram visualization plugin
struct HistogramPlugin;

impl VisualizationPlugin for HistogramPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "histogram".to_string(),
            version: "1.0.0".to_string(),
            description: "Generates histogram visualizations".to_string(),
            author: "TrustformeRS".to_string(),
            supported_inputs: vec!["Array1D".to_string(), "Tensor".to_string()],
            supported_outputs: vec![OutputFormat::Png, OutputFormat::Svg, OutputFormat::Text],
            tags: vec!["distribution".to_string(), "statistics".to_string()],
        }
    }

    fn execute(&self, data: VisualizationData, config: PluginConfig) -> Result<PluginResult> {
        let values = match data {
            VisualizationData::Array1D(v) => v,
            VisualizationData::Tensor { data, .. } => data,
            _ => anyhow::bail!("Unsupported data type for histogram"),
        };

        // Calculate histogram bins
        let bins = config.custom_params.get("bins").and_then(|v| v.as_u64()).unwrap_or(20) as usize;

        let min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let bin_width = (max - min) / bins as f64;

        let mut counts = vec![0; bins];
        for &value in &values {
            let bin_idx = ((value - min) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(bins - 1);
            counts[bin_idx] += 1;
        }

        // Generate output
        let output_text = format!(
            "Histogram (bins={}):\nMin={:.4}, Max={:.4}\nBin counts: {:?}",
            bins, min, max, counts
        );

        Ok(PluginResult {
            success: true,
            output_path: None,
            output_data: Some(output_text.into_bytes()),
            metadata: {
                let mut m = HashMap::new();
                m.insert("bins".to_string(), bins.to_string());
                m.insert("min".to_string(), min.to_string());
                m.insert("max".to_string(), max.to_string());
                m
            },
            error: None,
        })
    }

    fn validate(&self, data: &VisualizationData) -> bool {
        matches!(
            data,
            VisualizationData::Array1D(_) | VisualizationData::Tensor { .. }
        )
    }
}

/// Heatmap visualization plugin
struct HeatmapPlugin;

impl VisualizationPlugin for HeatmapPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "heatmap".to_string(),
            version: "1.0.0".to_string(),
            description: "Generates heatmap visualizations for 2D data".to_string(),
            author: "TrustformeRS".to_string(),
            supported_inputs: vec!["Array2D".to_string(), "Tensor".to_string()],
            supported_outputs: vec![OutputFormat::Png, OutputFormat::Html],
            tags: vec!["matrix".to_string(), "2d".to_string()],
        }
    }

    fn execute(&self, data: VisualizationData, _config: PluginConfig) -> Result<PluginResult> {
        let (rows, cols) = match &data {
            VisualizationData::Array2D(v) => (v.len(), v.first().map(|r| r.len()).unwrap_or(0)),
            VisualizationData::Tensor { shape, .. } if shape.len() == 2 => (shape[0], shape[1]),
            _ => anyhow::bail!("Heatmap requires 2D data"),
        };

        Ok(PluginResult {
            success: true,
            output_path: None,
            output_data: Some(format!("Heatmap {}x{}", rows, cols).into_bytes()),
            metadata: {
                let mut m = HashMap::new();
                m.insert("rows".to_string(), rows.to_string());
                m.insert("cols".to_string(), cols.to_string());
                m
            },
            error: None,
        })
    }

    fn validate(&self, data: &VisualizationData) -> bool {
        match data {
            VisualizationData::Array2D(_) => true,
            VisualizationData::Tensor { shape, .. } => shape.len() == 2,
            _ => false,
        }
    }
}

/// Line plot visualization plugin
struct LinePlotPlugin;

impl VisualizationPlugin for LinePlotPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "lineplot".to_string(),
            version: "1.0.0".to_string(),
            description: "Generates line plots for time series data".to_string(),
            author: "TrustformeRS".to_string(),
            supported_inputs: vec!["TimeSeries".to_string(), "Array1D".to_string()],
            supported_outputs: vec![OutputFormat::Png, OutputFormat::Svg],
            tags: vec!["timeseries".to_string(), "trend".to_string()],
        }
    }

    fn execute(&self, data: VisualizationData, _config: PluginConfig) -> Result<PluginResult> {
        let points = match &data {
            VisualizationData::TimeSeries { values, .. } => values.len(),
            VisualizationData::Array1D(v) => v.len(),
            _ => anyhow::bail!("Line plot requires time series or 1D array data"),
        };

        Ok(PluginResult {
            success: true,
            output_path: None,
            output_data: Some(format!("Line plot with {} points", points).into_bytes()),
            metadata: {
                let mut m = HashMap::new();
                m.insert("points".to_string(), points.to_string());
                m
            },
            error: None,
        })
    }
}

/// Scatter plot visualization plugin
struct ScatterPlotPlugin;

impl VisualizationPlugin for ScatterPlotPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "scatterplot".to_string(),
            version: "1.0.0".to_string(),
            description: "Generates scatter plots for 2D point data".to_string(),
            author: "TrustformeRS".to_string(),
            supported_inputs: vec!["Array2D".to_string()],
            supported_outputs: vec![OutputFormat::Png, OutputFormat::Html],
            tags: vec!["correlation".to_string(), "distribution".to_string()],
        }
    }

    fn execute(&self, data: VisualizationData, _config: PluginConfig) -> Result<PluginResult> {
        let points = match &data {
            VisualizationData::Array2D(v) => v.len(),
            _ => anyhow::bail!("Scatter plot requires 2D array data"),
        };

        Ok(PluginResult {
            success: true,
            output_path: None,
            output_data: Some(format!("Scatter plot with {} points", points).into_bytes()),
            metadata: {
                let mut m = HashMap::new();
                m.insert("points".to_string(), points.to_string());
                m
            },
            error: None,
        })
    }

    fn validate(&self, data: &VisualizationData) -> bool {
        matches!(data, VisualizationData::Array2D(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_manager_creation() {
        let manager = PluginManager::new();
        let plugins = manager.list_plugins();

        // Should have built-in plugins
        assert!(!plugins.is_empty());
    }

    #[test]
    fn test_histogram_plugin() {
        let manager = PluginManager::new();

        let data = VisualizationData::Array1D(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let config = PluginConfig::default();

        let result = manager.execute("histogram", data, config).unwrap();

        assert!(result.success);
        assert!(result.output_data.is_some());
    }

    #[test]
    fn test_plugin_validation() {
        let manager = PluginManager::new();

        // Valid data for histogram
        let data = VisualizationData::Array1D(vec![1.0, 2.0, 3.0]);
        let config = PluginConfig::default();

        assert!(manager.execute("histogram", data, config.clone()).is_ok());

        // Invalid data for heatmap (needs 2D)
        let data = VisualizationData::Array1D(vec![1.0, 2.0, 3.0]);

        assert!(manager.execute("heatmap", data, config).is_err());
    }

    #[test]
    fn test_custom_plugin_registration() {
        let manager = PluginManager::new();

        // Count plugins before
        let count_before = manager.list_plugins().len();

        // Register histogram again (should replace)
        manager.register_plugin(Box::new(HistogramPlugin)).unwrap();

        let count_after = manager.list_plugins().len();

        // Should be same count (replacement)
        assert_eq!(count_before, count_after);
    }
}
