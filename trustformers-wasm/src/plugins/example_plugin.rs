// Example plugin implementation for TrustformeRS WASM
// Demonstrates how to create community plugins

#![allow(clippy::missing_enforced_import_renames)]

use crate::plugin_framework::{
    ExecutionMetrics, Plugin, PluginConfig, PluginContext, PluginError, PluginErrorCode,
    PluginMetadata, PluginPermission, PluginResult, PluginType,
};
use std::collections::HashMap;

/// Example text processor plugin
pub struct TextProcessorPlugin {
    enabled: bool,
    settings: HashMap<String, String>,
}

impl Default for TextProcessorPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl TextProcessorPlugin {
    pub fn new() -> Self {
        Self {
            enabled: false,
            settings: HashMap::new(),
        }
    }
}

impl Plugin for TextProcessorPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "Text Processor".to_string(),
            version: "1.0.0".to_string(),
            author: "TrustformeRS Community".to_string(),
            description: "Example plugin for text preprocessing".to_string(),
            plugin_type: PluginType::Preprocessor,
            dependencies: vec![],
            permissions: vec![
                PluginPermission::ReadModelData,
                PluginPermission::DebugAccess,
            ],
        }
    }

    fn initialize(&mut self, config: PluginConfig) -> Result<(), PluginError> {
        self.settings = config.settings;
        self.enabled = true;

        // Validate required settings
        if !self.settings.contains_key("processing_mode") {
            return Err(PluginError {
                code: PluginErrorCode::InvalidConfiguration,
                message: "Missing 'processing_mode' setting".to_string(),
                details: Some("Valid modes: 'lowercase', 'uppercase', 'trim'".to_string()),
            });
        }

        Ok(())
    }

    fn execute(&self, context: &PluginContext) -> Result<PluginResult, PluginError> {
        if !self.enabled {
            return Err(PluginError {
                code: PluginErrorCode::ExecutionFailed,
                message: "Plugin not initialized".to_string(),
                details: None,
            });
        }

        let start_time = std::time::Instant::now();
        let mut result_data = HashMap::new();

        // Get input text from context
        let input_text = context.request_data.get("input_text").cloned().unwrap_or_default();

        // Process text based on settings
        let default_mode = "trim".to_string();
        let processing_mode = self.settings.get("processing_mode").unwrap_or(&default_mode);

        let processed_text = match processing_mode.as_str() {
            "lowercase" => input_text.to_lowercase(),
            "uppercase" => input_text.to_uppercase(),
            "trim" => input_text.trim().to_string(),
            _ => input_text.clone(), // No processing
        };

        result_data.insert("processed_text".to_string(), processed_text.clone());
        result_data.insert("original_length".to_string(), input_text.len().to_string());
        result_data.insert(
            "processed_length".to_string(),
            processed_text.len().to_string(),
        );
        result_data.insert("processing_mode".to_string(), processing_mode.clone());

        let execution_time = start_time.elapsed().as_millis() as f64;

        Ok(PluginResult {
            success: true,
            data: result_data,
            metrics: ExecutionMetrics {
                execution_time_ms: execution_time,
                memory_used_mb: 0.1, // Minimal memory usage
                cpu_usage_percent: 5.0,
                gpu_memory_used_mb: None,
            },
            messages: vec![format!(
                "Processed {} characters using {} mode",
                processed_text.len(),
                processing_mode
            )],
        })
    }

    fn cleanup(&mut self) {
        self.enabled = false;
        self.settings.clear();
    }
}

/// Example model optimizer plugin
pub struct ModelOptimizerPlugin {
    enabled: bool,
    optimization_level: u8,
}

impl Default for ModelOptimizerPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelOptimizerPlugin {
    pub fn new() -> Self {
        Self {
            enabled: false,
            optimization_level: 1,
        }
    }
}

impl Plugin for ModelOptimizerPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "Model Optimizer".to_string(),
            version: "1.2.0".to_string(),
            author: "TrustformeRS Team".to_string(),
            description: "Example plugin for model optimization".to_string(),
            plugin_type: PluginType::Optimizer,
            dependencies: vec![],
            permissions: vec![
                PluginPermission::ReadModelData,
                PluginPermission::WriteModelData,
                PluginPermission::GpuAccess,
                PluginPermission::ProfilingAccess,
            ],
        }
    }

    fn initialize(&mut self, config: PluginConfig) -> Result<(), PluginError> {
        // Parse optimization level from config
        if let Some(level_str) = config.settings.get("optimization_level") {
            match level_str.parse::<u8>() {
                Ok(level) if level <= 3 => self.optimization_level = level,
                _ => {
                    return Err(PluginError {
                        code: PluginErrorCode::InvalidConfiguration,
                        message: "Invalid optimization level (must be 0-3)".to_string(),
                        details: None,
                    })
                },
            }
        }

        self.enabled = true;
        Ok(())
    }

    fn execute(&self, context: &PluginContext) -> Result<PluginResult, PluginError> {
        if !self.enabled {
            return Err(PluginError {
                code: PluginErrorCode::ExecutionFailed,
                message: "Plugin not initialized".to_string(),
                details: None,
            });
        }

        let start_time = std::time::Instant::now();
        let mut result_data = HashMap::new();

        // Simulate model optimization based on level
        let optimization_applied = match self.optimization_level {
            0 => "None",
            1 => "Basic quantization",
            2 => "Advanced quantization + pruning",
            3 => "Full optimization suite",
            _ => "Unknown",
        };

        // Simulate optimization metrics
        let size_reduction = match self.optimization_level {
            0 => 0.0,
            1 => 15.0,
            2 => 35.0,
            3 => 55.0,
            _ => 0.0,
        };

        let speed_improvement = match self.optimization_level {
            0 => 0.0,
            1 => 10.0,
            2 => 25.0,
            3 => 40.0,
            _ => 0.0,
        };

        result_data.insert(
            "optimization_applied".to_string(),
            optimization_applied.to_string(),
        );
        result_data.insert(
            "size_reduction_percent".to_string(),
            size_reduction.to_string(),
        );
        result_data.insert(
            "speed_improvement_percent".to_string(),
            speed_improvement.to_string(),
        );
        result_data.insert(
            "optimization_level".to_string(),
            self.optimization_level.to_string(),
        );

        // Add model metadata if available
        if let Some(ref metadata) = context.model_metadata {
            result_data.insert(
                "original_model_size_mb".to_string(),
                metadata.size_mb.to_string(),
            );
            result_data.insert("model_type".to_string(), metadata.model_type.clone());
        }

        let execution_time = start_time.elapsed().as_millis() as f64;

        Ok(PluginResult {
            success: true,
            data: result_data,
            metrics: ExecutionMetrics {
                execution_time_ms: execution_time,
                memory_used_mb: 5.0,
                cpu_usage_percent: 15.0,
                gpu_memory_used_mb: Some(2.0),
            },
            messages: vec![
                format!("Applied {optimization_applied} optimization"),
                format!(
                    "Achieved {:.1}% size reduction and {:.1}% speed improvement",
                    size_reduction, speed_improvement
                ),
            ],
        })
    }

    fn cleanup(&mut self) {
        self.enabled = false;
        self.optimization_level = 1;
    }
}

/// Example visualization plugin
pub struct VisualizationPlugin {
    enabled: bool,
    chart_type: String,
}

impl Default for VisualizationPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl VisualizationPlugin {
    pub fn new() -> Self {
        Self {
            enabled: false,
            chart_type: "bar".to_string(),
        }
    }
}

impl Plugin for VisualizationPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "Data Visualizer".to_string(),
            version: "0.9.0".to_string(),
            author: "Community Developer".to_string(),
            description: "Create charts and graphs from model data".to_string(),
            plugin_type: PluginType::Visualizer,
            dependencies: vec![],
            permissions: vec![PluginPermission::ReadModelData, PluginPermission::UiAccess],
        }
    }

    fn initialize(&mut self, config: PluginConfig) -> Result<(), PluginError> {
        if let Some(chart_type) = config.settings.get("chart_type") {
            let valid_types = ["bar", "line", "pie", "scatter"];
            if valid_types.contains(&chart_type.as_str()) {
                self.chart_type = chart_type.clone();
            } else {
                return Err(PluginError {
                    code: PluginErrorCode::InvalidConfiguration,
                    message: format!("Invalid chart type: {chart_type}"),
                    details: Some("Valid types: bar, line, pie, scatter".to_string()),
                });
            }
        }

        self.enabled = true;
        Ok(())
    }

    fn execute(&self, _context: &PluginContext) -> Result<PluginResult, PluginError> {
        if !self.enabled {
            return Err(PluginError {
                code: PluginErrorCode::ExecutionFailed,
                message: "Plugin not initialized".to_string(),
                details: None,
            });
        }

        let start_time = std::time::Instant::now();
        let mut result_data = HashMap::new();

        // Generate mock chart data based on chart type
        let chart_data = match self.chart_type.as_str() {
            "bar" => {
                r#"{"type":"bar","data":{"labels":["Q1","Q2","Q3","Q4"],"values":[10,20,15,25]}}"#
            },
            "line" => r#"{"type":"line","data":{"x":[1,2,3,4,5],"y":[2,4,3,5,4]}}"#,
            "pie" => r#"{"type":"pie","data":{"labels":["A","B","C"],"values":[30,40,30]}}"#,
            "scatter" => r#"{"type":"scatter","data":{"points":[[1,2],[2,4],[3,3],[4,5]]}}"#,
            _ => r#"{"type":"unknown","data":{}}"#,
        };

        result_data.insert("chart_type".to_string(), self.chart_type.clone());
        result_data.insert("chart_data".to_string(), chart_data.to_string());
        result_data.insert("width".to_string(), "800".to_string());
        result_data.insert("height".to_string(), "600".to_string());

        let execution_time = start_time.elapsed().as_millis() as f64;

        Ok(PluginResult {
            success: true,
            data: result_data,
            metrics: ExecutionMetrics {
                execution_time_ms: execution_time,
                memory_used_mb: 1.0,
                cpu_usage_percent: 8.0,
                gpu_memory_used_mb: None,
            },
            messages: vec![
                format!("Generated {chart_type} chart", chart_type = self.chart_type),
                "Chart data ready for rendering".to_string(),
            ],
        })
    }

    fn cleanup(&mut self) {
        self.enabled = false;
        self.chart_type = "bar".to_string();
    }
}
