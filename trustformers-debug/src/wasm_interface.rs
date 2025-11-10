//! WebAssembly Interface for TrustformeRS Debugging
//!
//! This module provides a WebAssembly-compatible interface for running TrustformeRS
//! debugging tools in web browsers and Node.js environments.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

// Import the `console.log` function from the `console` module for web debugging
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

// Macro for console logging in WASM
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// Configuration for WASM debugging interface
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmDebugConfig {
    /// Enable browser-specific optimizations
    pub browser_optimizations: bool,
    /// Enable Node.js-specific features
    pub nodejs_features: bool,
    /// Maximum memory usage (in MB)
    pub max_memory_mb: usize,
    /// Enable streaming for large datasets
    pub enable_streaming: bool,
    /// Chunk size for streaming operations
    pub streaming_chunk_size: usize,
    /// Enable WebGL acceleration
    pub enable_webgl: bool,
    /// Enable Web Workers for parallel processing
    pub enable_web_workers: bool,
}

impl Default for WasmDebugConfig {
    fn default() -> Self {
        Self {
            browser_optimizations: true,
            nodejs_features: false,
            max_memory_mb: 256,
            enable_streaming: true,
            streaming_chunk_size: 1024,
            enable_webgl: false,
            enable_web_workers: true,
        }
    }
}

/// WebAssembly-compatible tensor data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct WasmTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
    name: String,
}

/// WebAssembly debugging result
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct WasmDebugResult {
    layer_name: String,
    metrics: String, // JSON-serialized metrics
    success: bool,
    error_message: Option<String>,
}

/// WebAssembly debugging session
#[derive(Debug)]
#[wasm_bindgen]
pub struct WasmDebugSession {
    config: WasmDebugConfig,
    tensors: HashMap<String, WasmTensor>,
    analysis_results: HashMap<String, WasmDebugResult>,
    session_id: String,
    is_initialized: bool,
}

#[wasm_bindgen]
impl WasmDebugSession {
    /// Create new WASM debugging session
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmDebugSession {
        console_log!("Creating new TrustformeRS Debug WASM session");

        Self {
            config: WasmDebugConfig::default(),
            tensors: HashMap::new(),
            analysis_results: HashMap::new(),
            session_id: generate_session_id(),
            is_initialized: false,
        }
    }

    /// Initialize the debugging session with configuration
    #[wasm_bindgen]
    pub fn initialize(&mut self, config_json: &str) -> bool {
        console_log!(
            "Initializing WASM debug session with config: {}",
            config_json
        );

        match serde_json::from_str::<WasmDebugConfig>(config_json) {
            Ok(config) => {
                self.config = config;
                self.is_initialized = true;
                console_log!("WASM debug session initialized successfully");
                true
            },
            Err(e) => {
                console_log!("Failed to initialize session: {}", e);
                false
            },
        }
    }

    /// Add tensor for debugging analysis
    #[wasm_bindgen]
    pub fn add_tensor(&mut self, name: &str, data: &[f32], shape: &[usize]) -> bool {
        console_log!("Adding tensor '{}' with shape {:?}", name, shape);

        // Validate tensor data
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            console_log!(
                "Tensor size mismatch: expected {}, got {}",
                expected_size,
                data.len()
            );
            return false;
        }

        // Check memory constraints
        let tensor_size_mb = (data.len() * 4) / (1024 * 1024); // 4 bytes per f32
        if tensor_size_mb > self.config.max_memory_mb / 2 {
            console_log!("Tensor too large: {} MB exceeds limit", tensor_size_mb);
            return false;
        }

        let tensor = WasmTensor {
            data: data.to_vec(),
            shape: shape.to_vec(),
            name: name.to_string(),
        };

        self.tensors.insert(name.to_string(), tensor);
        console_log!("Tensor '{}' added successfully", name);
        true
    }

    /// Perform basic tensor analysis
    #[wasm_bindgen]
    pub fn analyze_tensor(&mut self, name: &str) -> String {
        console_log!("Analyzing tensor '{}'", name);

        if !self.is_initialized {
            let error = "Session not initialized".to_string();
            console_log!("Error: {}", error);
            return serde_json::to_string(&WasmDebugResult {
                layer_name: name.to_string(),
                metrics: "{}".to_string(),
                success: false,
                error_message: Some(error),
            })
            .unwrap_or_else(|_| "{}".to_string());
        }

        match self.tensors.get(name) {
            Some(tensor) => {
                let analysis = self.perform_tensor_analysis(tensor);
                let metrics_json =
                    serde_json::to_string(&analysis).unwrap_or_else(|_| "{}".to_string());

                let result = WasmDebugResult {
                    layer_name: name.to_string(),
                    metrics: metrics_json,
                    success: true,
                    error_message: None,
                };

                self.analysis_results.insert(name.to_string(), result.clone());
                console_log!("Analysis completed for tensor '{}'", name);

                serde_json::to_string(&result).unwrap_or_else(|_| "{}".to_string())
            },
            None => {
                let error = format!("Tensor '{}' not found", name);
                console_log!("Error: {}", error);
                serde_json::to_string(&WasmDebugResult {
                    layer_name: name.to_string(),
                    metrics: "{}".to_string(),
                    success: false,
                    error_message: Some(error),
                })
                .unwrap_or_else(|_| "{}".to_string())
            },
        }
    }

    /// Get comprehensive analysis of all tensors
    #[wasm_bindgen]
    pub fn analyze_all_tensors(&mut self) -> String {
        console_log!("Analyzing all {} tensors", self.tensors.len());

        let mut all_results = HashMap::new();

        // Collect tensor names first to avoid borrowing conflicts
        let tensor_names: Vec<String> = self.tensors.keys().cloned().collect();

        for tensor_name in tensor_names {
            let result = self.analyze_tensor(&tensor_name);
            if let Ok(parsed_result) = serde_json::from_str::<WasmDebugResult>(&result) {
                all_results.insert(tensor_name, parsed_result);
            }
        }

        serde_json::to_string(&all_results).unwrap_or_else(|_| "{}".to_string())
    }

    /// Detect anomalies in tensor data
    #[wasm_bindgen]
    pub fn detect_anomalies(&self, name: &str, threshold: f32) -> String {
        console_log!(
            "Detecting anomalies in tensor '{}' with threshold {}",
            name,
            threshold
        );

        match self.tensors.get(name) {
            Some(tensor) => {
                let anomalies = self.find_tensor_anomalies(tensor, threshold);
                serde_json::to_string(&anomalies).unwrap_or_else(|_| "{}".to_string())
            },
            None => {
                console_log!("Tensor '{}' not found for anomaly detection", name);
                "{}".to_string()
            },
        }
    }

    /// Generate visualization data for tensor
    #[wasm_bindgen]
    pub fn generate_visualization_data(&self, name: &str) -> String {
        console_log!("Generating visualization data for tensor '{}'", name);

        match self.tensors.get(name) {
            Some(tensor) => {
                let viz_data = self.create_visualization_data(tensor);
                serde_json::to_string(&viz_data).unwrap_or_else(|_| "{}".to_string())
            },
            None => {
                console_log!("Tensor '{}' not found for visualization", name);
                "{}".to_string()
            },
        }
    }

    /// Export analysis results
    #[wasm_bindgen]
    pub fn export_results(&self, format: &str) -> String {
        console_log!("Exporting results in format: {}", format);

        match format {
            "json" => {
                serde_json::to_string(&self.analysis_results).unwrap_or_else(|_| "{}".to_string())
            },
            "csv" => self.export_to_csv(),
            "html" => self.export_to_html(),
            _ => {
                console_log!("Unsupported export format: {}", format);
                "{}".to_string()
            },
        }
    }

    /// Get session statistics
    #[wasm_bindgen]
    pub fn get_session_stats(&self) -> String {
        let stats = WasmSessionStats {
            session_id: self.session_id.clone(),
            tensor_count: self.tensors.len(),
            analysis_count: self.analysis_results.len(),
            memory_usage_mb: self.estimate_memory_usage(),
            is_initialized: self.is_initialized,
        };

        serde_json::to_string(&stats).unwrap_or_else(|_| "{}".to_string())
    }

    /// Clear all data and reset session
    #[wasm_bindgen]
    pub fn clear(&mut self) {
        console_log!("Clearing WASM debug session");
        self.tensors.clear();
        self.analysis_results.clear();
        console_log!("Session cleared");
    }

    /// Get list of available tensors
    #[wasm_bindgen]
    pub fn get_tensor_list(&self) -> String {
        let tensor_names: Vec<String> = self.tensors.keys().cloned().collect();
        serde_json::to_string(&tensor_names).unwrap_or_else(|_| "[]".to_string())
    }

    /// Remove a tensor from the session
    #[wasm_bindgen]
    pub fn remove_tensor(&mut self, name: &str) -> bool {
        console_log!("Removing tensor '{}'", name);
        self.tensors.remove(name).is_some()
    }
}

impl WasmDebugSession {
    /// Perform comprehensive tensor analysis
    fn perform_tensor_analysis(&self, tensor: &WasmTensor) -> TensorAnalysisResult {
        let data = &tensor.data;
        let total_elements = data.len();

        // Basic statistics
        let sum: f32 = data.iter().sum();
        let mean = sum / total_elements as f32;
        let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        // Variance and standard deviation
        let variance: f32 =
            data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / total_elements as f32;
        let std_dev = variance.sqrt();

        // Count special values
        let nan_count = data.iter().filter(|x| x.is_nan()).count();
        let inf_count = data.iter().filter(|x| x.is_infinite()).count();
        let zero_count = data.iter().filter(|&x| *x == 0.0).count();

        // Sparsity
        let sparsity = zero_count as f32 / total_elements as f32;

        // Range
        let range = max - min;

        // L1 and L2 norms
        let l1_norm: f32 = data.iter().map(|x| x.abs()).sum();
        let l2_norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Histogram data (simplified)
        let histogram = self.calculate_histogram(data, 20);

        TensorAnalysisResult {
            name: tensor.name.clone(),
            shape: tensor.shape.clone(),
            total_elements,
            mean,
            std_dev,
            min,
            max,
            variance,
            nan_count,
            inf_count,
            zero_count,
            sparsity,
            range,
            l1_norm,
            l2_norm,
            histogram,
        }
    }

    /// Calculate histogram for tensor data
    fn calculate_histogram(&self, data: &[f32], bins: usize) -> Vec<usize> {
        let min_val = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        if min_val == max_val {
            return vec![data.len()];
        }

        let bin_width = (max_val - min_val) / bins as f32;
        let mut histogram = vec![0; bins];

        for &value in data {
            if value.is_finite() {
                let bin_index = ((value - min_val) / bin_width) as usize;
                let bin_index = bin_index.min(bins - 1);
                histogram[bin_index] += 1;
            }
        }

        histogram
    }

    /// Find anomalies in tensor data
    fn find_tensor_anomalies(&self, tensor: &WasmTensor, threshold: f32) -> AnomalyDetectionResult {
        let data = &tensor.data;
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let variance: f32 =
            data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std_dev = variance.sqrt();

        let mut outlier_indices = Vec::new();
        let mut outlier_values = Vec::new();

        for (i, &value) in data.iter().enumerate() {
            let z_score = (value - mean).abs() / std_dev;
            if z_score > threshold {
                outlier_indices.push(i);
                outlier_values.push(value);
            }
        }

        let outlier_percentage = outlier_indices.len() as f32 / data.len() as f32 * 100.0;

        AnomalyDetectionResult {
            tensor_name: tensor.name.clone(),
            outlier_count: outlier_indices.len(),
            outlier_percentage,
            outlier_indices,
            outlier_values,
            threshold_used: threshold,
            mean,
            std_dev,
        }
    }

    /// Create visualization data for tensor
    fn create_visualization_data(&self, tensor: &WasmTensor) -> VisualizationData {
        let data = &tensor.data;
        let shape = &tensor.shape;

        // Sample data for visualization to avoid overwhelming browser
        let sample_size = 1000.min(data.len());
        let step = data.len() / sample_size;
        let sampled_data: Vec<f32> =
            data.iter().step_by(step.max(1)).take(sample_size).cloned().collect();

        // Create histogram
        let histogram = self.calculate_histogram(data, 50);

        // For 2D data, create heatmap data
        let heatmap_data = if shape.len() == 2 {
            Some(self.create_2d_heatmap_data(data, shape[0], shape[1]))
        } else {
            None
        };

        VisualizationData {
            tensor_name: tensor.name.clone(),
            sampled_data,
            histogram,
            heatmap_data,
            shape: shape.clone(),
        }
    }

    /// Create 2D heatmap data
    fn create_2d_heatmap_data(&self, data: &[f32], rows: usize, cols: usize) -> Vec<Vec<f32>> {
        let mut heatmap = Vec::new();

        for i in 0..rows {
            let mut row = Vec::new();
            for j in 0..cols {
                let idx = i * cols + j;
                if idx < data.len() {
                    row.push(data[idx]);
                } else {
                    row.push(0.0);
                }
            }
            heatmap.push(row);
        }

        heatmap
    }

    /// Export results to CSV format
    fn export_to_csv(&self) -> String {
        let mut csv = String::from("tensor_name,metric,value\n");

        for (name, result) in &self.analysis_results {
            if let Ok(metrics) = serde_json::from_str::<TensorAnalysisResult>(&result.metrics) {
                csv.push_str(&format!("{},mean,{}\n", name, metrics.mean));
                csv.push_str(&format!("{},std_dev,{}\n", name, metrics.std_dev));
                csv.push_str(&format!("{},min,{}\n", name, metrics.min));
                csv.push_str(&format!("{},max,{}\n", name, metrics.max));
                csv.push_str(&format!("{},sparsity,{}\n", name, metrics.sparsity));
            }
        }

        csv
    }

    /// Export results to HTML format
    fn export_to_html(&self) -> String {
        let mut html = String::from(
            "<!DOCTYPE html><html><head><title>TrustformeRS Debug Results</title></head><body>",
        );
        html.push_str("<h1>TrustformeRS Debug Analysis Results</h1>");

        for (name, result) in &self.analysis_results {
            html.push_str(&format!("<h2>Tensor: {}</h2>", name));
            if let Ok(metrics) = serde_json::from_str::<TensorAnalysisResult>(&result.metrics) {
                html.push_str("<table border='1'>");
                html.push_str(&format!(
                    "<tr><td>Mean</td><td>{:.6}</td></tr>",
                    metrics.mean
                ));
                html.push_str(&format!(
                    "<tr><td>Std Dev</td><td>{:.6}</td></tr>",
                    metrics.std_dev
                ));
                html.push_str(&format!("<tr><td>Min</td><td>{:.6}</td></tr>", metrics.min));
                html.push_str(&format!("<tr><td>Max</td><td>{:.6}</td></tr>", metrics.max));
                html.push_str(&format!(
                    "<tr><td>Sparsity</td><td>{:.2}%</td></tr>",
                    metrics.sparsity * 100.0
                ));
                html.push_str("</table>");
            }
        }

        html.push_str("</body></html>");
        html
    }

    /// Estimate current memory usage
    fn estimate_memory_usage(&self) -> usize {
        let mut total_bytes = 0;

        for tensor in self.tensors.values() {
            total_bytes += tensor.data.len() * 4; // 4 bytes per f32
        }

        total_bytes / (1024 * 1024) // Convert to MB
    }
}

/// Tensor analysis result structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorAnalysisResult {
    pub name: String,
    pub shape: Vec<usize>,
    pub total_elements: usize,
    pub mean: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub variance: f32,
    pub nan_count: usize,
    pub inf_count: usize,
    pub zero_count: usize,
    pub sparsity: f32,
    pub range: f32,
    pub l1_norm: f32,
    pub l2_norm: f32,
    pub histogram: Vec<usize>,
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionResult {
    pub tensor_name: String,
    pub outlier_count: usize,
    pub outlier_percentage: f32,
    pub outlier_indices: Vec<usize>,
    pub outlier_values: Vec<f32>,
    pub threshold_used: f32,
    pub mean: f32,
    pub std_dev: f32,
}

/// Visualization data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationData {
    pub tensor_name: String,
    pub sampled_data: Vec<f32>,
    pub histogram: Vec<usize>,
    pub heatmap_data: Option<Vec<Vec<f32>>>,
    pub shape: Vec<usize>,
}

/// Session statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmSessionStats {
    pub session_id: String,
    pub tensor_count: usize,
    pub analysis_count: usize,
    pub memory_usage_mb: usize,
    pub is_initialized: bool,
}

/// Generate a unique session ID
fn generate_session_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

    format!("wasm_session_{}", timestamp)
}

/// JavaScript-callable utility functions
#[wasm_bindgen]
pub struct WasmUtils;

#[wasm_bindgen]
impl WasmUtils {
    /// Get library version
    #[wasm_bindgen]
    pub fn get_version() -> String {
        "0.1.0-alpha.1".to_string()
    }

    /// Check WebAssembly capabilities
    #[wasm_bindgen]
    pub fn check_capabilities() -> String {
        let capabilities = WasmCapabilities {
            threads_supported: cfg!(feature = "atomics"),
            simd_supported: cfg!(target_feature = "simd128"),
            memory_64_supported: cfg!(target_pointer_width = "64"),
            bulk_memory_supported: true,
            reference_types_supported: true,
        };

        serde_json::to_string(&capabilities).unwrap_or_else(|_| "{}".to_string())
    }

    /// Memory usage information
    #[wasm_bindgen]
    pub fn get_memory_info() -> String {
        #[cfg(target_arch = "wasm32")]
        {
            let memory_info = WasmMemoryInfo {
                pages: wasm_bindgen::memory().buffer().byte_length() / 65536,
                bytes: wasm_bindgen::memory().buffer().byte_length(),
            };
            serde_json::to_string(&memory_info).unwrap_or_else(|_| "{}".to_string())
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            "{}".to_string()
        }
    }
}

/// WebAssembly capabilities information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmCapabilities {
    pub threads_supported: bool,
    pub simd_supported: bool,
    pub memory_64_supported: bool,
    pub bulk_memory_supported: bool,
    pub reference_types_supported: bool,
}

/// WebAssembly memory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmMemoryInfo {
    pub pages: usize,
    pub bytes: usize,
}

// Module-level tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_wasm_session_creation() {
        let session = WasmDebugSession::new();
        assert!(!session.is_initialized);
        assert_eq!(session.tensors.len(), 0);
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_tensor_addition() {
        let mut session = WasmDebugSession::new();
        let _ = session.initialize(r#"{"browser_optimizations": true}"#);

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];

        assert!(session.add_tensor("test_tensor", &data, &shape));
        assert_eq!(session.tensors.len(), 1);
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_tensor_analysis() {
        let mut session = WasmDebugSession::new();
        let _ = session.initialize(r#"{"browser_optimizations": true}"#);

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];

        session.add_tensor("test_tensor", &data, &shape);
        let result = session.analyze_tensor("test_tensor");

        assert!(!result.is_empty());
        assert!(result.contains("test_tensor"));
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_anomaly_detection() {
        let mut session = WasmDebugSession::new();
        let _ = session.initialize(r#"{"browser_optimizations": true}"#);

        let data = vec![1.0, 1.0, 1.0, 100.0]; // One outlier
        let shape = vec![4];

        session.add_tensor("anomaly_test", &data, &shape);
        let result = session.detect_anomalies("anomaly_test", 2.0);

        assert!(!result.is_empty());
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_export_functionality() {
        let mut session = WasmDebugSession::new();
        let _ = session.initialize(r#"{"browser_optimizations": true}"#);

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];

        session.add_tensor("export_test", &data, &shape);
        let _ = session.analyze_tensor("export_test");

        let json_export = session.export_results("json");
        assert!(!json_export.is_empty());

        let csv_export = session.export_results("csv");
        assert!(!csv_export.is_empty());

        let html_export = session.export_results("html");
        assert!(html_export.contains("<!DOCTYPE html>"));
    }

    #[test]
    fn test_utilities() {
        let version = WasmUtils::get_version();
        assert!(!version.is_empty());

        let capabilities = WasmUtils::check_capabilities();
        assert!(!capabilities.is_empty());
    }
}
