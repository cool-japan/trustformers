//! Mobile model debugging and analysis for TrustformeRS
//!
//! This module provides comprehensive model debugging capabilities specifically designed for mobile
//! environments, including inference debugging, model validation, and performance analysis.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use trustformers_core::error::{CoreError, Result};
use trustformers_core::Tensor;
use trustformers_core::TrustformersError;

/// Model debugging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDebuggerConfig {
    /// Enable model debugging
    pub enabled: bool,
    /// Maximum number of inference traces to keep
    pub max_inference_traces: usize,
    /// Enable detailed tensor analysis
    pub enable_tensor_analysis: bool,
    /// Enable gradient analysis (for training models)
    pub enable_gradient_analysis: bool,
    /// Enable model structure validation
    pub enable_structure_validation: bool,
    /// Enable performance profiling
    pub enable_performance_profiling: bool,
    /// Maximum execution time before flagging as slow
    pub slow_inference_threshold: Duration,
    /// Enable automatic anomaly detection
    pub enable_anomaly_detection: bool,
    /// Tensor value ranges for anomaly detection
    pub normal_value_range: (f32, f32),
    /// Enable debugging output generation
    pub enable_debug_output: bool,
}

/// Model debugging information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDebugInfo {
    pub model_id: String,
    pub model_type: String,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shapes: Vec<Vec<usize>>,
    pub parameter_count: usize,
    pub memory_usage: usize,
    pub inference_count: usize,
    pub average_inference_time: Duration,
    pub last_inference_time: Duration,
    pub validation_status: ValidationStatus,
    pub anomalies_detected: Vec<ModelAnomaly>,
    pub performance_profile: PerformanceProfile,
}

/// Model validation status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationStatus {
    Valid,
    Warning(Vec<String>),
    Error(Vec<String>),
    NotValidated,
}

/// Detected model anomaly
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelAnomaly {
    pub anomaly_type: AnomalyType,
    pub severity: AnomalySeverity,
    pub description: String,
    pub location: String,
    pub timestamp: u64,
    pub suggested_action: String,
    pub tensor_info: Option<TensorDebugInfo>,
}

/// Types of model anomalies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnomalyType {
    NaNValues,
    InfiniteValues,
    ExtremeValues,
    UnexpectedShapes,
    PerformanceDegradation,
    MemorySpike,
    GradientExplosion,
    GradientVanishing,
    ModelDivergence,
    InvalidOperation,
}

/// Severity levels for anomalies
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AnomalySeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Tensor debugging information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDebugInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub min_value: f32,
    pub max_value: f32,
    pub mean_value: f32,
    pub std_dev: f32,
    pub zero_count: usize,
    pub nan_count: usize,
    pub inf_count: usize,
    pub memory_size: usize,
    pub sparsity: f32,
}

/// Performance profiling information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceProfile {
    pub total_inference_time: Duration,
    pub preprocessing_time: Duration,
    pub model_execution_time: Duration,
    pub postprocessing_time: Duration,
    pub memory_allocation_time: Duration,
    pub memory_peak: usize,
    pub cpu_usage: f32,
    pub gpu_usage: f32,
    pub bottlenecks: Vec<PerformanceBottleneck>,
}

/// Performance bottleneck information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBottleneck {
    pub operation_name: String,
    pub execution_time: Duration,
    pub percentage_of_total: f32,
    pub optimization_suggestions: Vec<String>,
}

/// Inference trace for debugging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceTrace {
    pub trace_id: String,
    pub timestamp: u64,
    pub input_tensors: Vec<TensorDebugInfo>,
    pub output_tensors: Vec<TensorDebugInfo>,
    pub intermediate_tensors: Vec<TensorDebugInfo>,
    pub execution_time: Duration,
    pub memory_usage: usize,
    pub anomalies: Vec<ModelAnomaly>,
    pub operations: Vec<OperationTrace>,
}

/// Individual operation trace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationTrace {
    pub operation_name: String,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shapes: Vec<Vec<usize>>,
    pub execution_time: Duration,
    pub memory_delta: i64,
    pub parameters: HashMap<String, String>,
}

/// Mobile model debugger
pub struct MobileModelDebugger {
    config: ModelDebuggerConfig,
    model_info: Arc<Mutex<HashMap<String, ModelDebugInfo>>>,
    inference_traces: Arc<Mutex<VecDeque<InferenceTrace>>>,
    anomaly_detectors: Arc<Mutex<HashMap<String, AnomalyDetector>>>,
    performance_history: Arc<Mutex<VecDeque<PerformanceProfile>>>,
}

/// Anomaly detection state
#[derive(Debug)]
struct AnomalyDetector {
    model_id: String,
    baseline_stats: TensorStats,
    recent_stats: VecDeque<TensorStats>,
    anomaly_threshold: f32,
    consecutive_anomalies: usize,
}

/// Statistical information for tensors
#[derive(Debug, Clone)]
struct TensorStats {
    mean: f32,
    std_dev: f32,
    min_val: f32,
    max_val: f32,
    nan_rate: f32,
    inf_rate: f32,
}

impl Default for ModelDebuggerConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_inference_traces: 100,
            enable_tensor_analysis: true,
            enable_gradient_analysis: false, // Disabled by default for inference-only models
            enable_structure_validation: true,
            enable_performance_profiling: true,
            slow_inference_threshold: Duration::from_millis(100),
            enable_anomaly_detection: true,
            normal_value_range: (-10.0, 10.0),
            enable_debug_output: false,
        }
    }
}

impl MobileModelDebugger {
    /// Create a new model debugger
    pub fn new(config: ModelDebuggerConfig) -> Self {
        Self {
            config,
            model_info: Arc::new(Mutex::new(HashMap::new())),
            inference_traces: Arc::new(Mutex::new(VecDeque::new())),
            anomaly_detectors: Arc::new(Mutex::new(HashMap::new())),
            performance_history: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Register a model for debugging
    pub fn register_model(
        &self,
        model_id: String,
        model_type: String,
        input_shapes: Vec<Vec<usize>>,
        output_shapes: Vec<Vec<usize>>,
        parameter_count: usize,
    ) -> Result<()> {
        if !self.config.enabled {
            return Ok(());
        }

        let mut model_info = self
            .model_info
            .lock()
            .map_err(|_| TrustformersError::runtime_error("Failed to acquire lock".into()))?;

        let debug_info = ModelDebugInfo {
            model_id: model_id.clone(),
            model_type,
            input_shapes,
            output_shapes,
            parameter_count,
            memory_usage: 0,
            inference_count: 0,
            average_inference_time: Duration::from_millis(0),
            last_inference_time: Duration::from_millis(0),
            validation_status: ValidationStatus::NotValidated,
            anomalies_detected: Vec::new(),
            performance_profile: PerformanceProfile::default(),
        };

        model_info.insert(model_id.clone(), debug_info);

        // Initialize anomaly detector
        if self.config.enable_anomaly_detection {
            let mut detectors = self
                .anomaly_detectors
                .lock()
                .map_err(|_| TrustformersError::runtime_error("Failed to acquire lock".into()))?;

            detectors.insert(
                model_id.clone(),
                AnomalyDetector {
                    model_id,
                    baseline_stats: TensorStats::default(),
                    recent_stats: VecDeque::new(),
                    anomaly_threshold: 2.0, // 2 standard deviations
                    consecutive_anomalies: 0,
                },
            );
        }

        Ok(())
    }

    /// Debug an inference operation
    pub fn debug_inference(
        &self,
        model_id: &str,
        input_tensors: &[Tensor],
        output_tensors: &[Tensor],
        execution_time: Duration,
        memory_usage: usize,
    ) -> Result<InferenceTrace> {
        if !self.config.enabled {
            return Err(TrustformersError::runtime_error("Debugger not enabled".into()).into());
        }

        let trace_id = format!(
            "trace_{}_{}",
            model_id,
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos()
        );

        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

        // Analyze input tensors
        let input_debug_info = if self.config.enable_tensor_analysis {
            input_tensors
                .iter()
                .enumerate()
                .map(|(i, tensor)| self.analyze_tensor(tensor, &format!("input_{}", i)))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };

        // Analyze output tensors
        let output_debug_info = if self.config.enable_tensor_analysis {
            output_tensors
                .iter()
                .enumerate()
                .map(|(i, tensor)| self.analyze_tensor(tensor, &format!("output_{}", i)))
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };

        // Detect anomalies
        let mut anomalies = Vec::new();
        if self.config.enable_anomaly_detection {
            anomalies.extend(self.detect_tensor_anomalies(model_id, &input_debug_info)?);
            anomalies.extend(self.detect_tensor_anomalies(model_id, &output_debug_info)?);
            anomalies.extend(self.detect_performance_anomalies(
                model_id,
                execution_time,
                memory_usage,
            )?);
        }

        // Create inference trace
        let trace = InferenceTrace {
            trace_id: trace_id.clone(),
            timestamp,
            input_tensors: input_debug_info,
            output_tensors: output_debug_info,
            intermediate_tensors: Vec::new(), // Would be populated by model internals
            execution_time,
            memory_usage,
            anomalies: anomalies.clone(),
            operations: Vec::new(), // Would be populated by operation tracing
        };

        // Store trace
        let mut traces = self
            .inference_traces
            .lock()
            .map_err(|_| TrustformersError::runtime_error("Failed to acquire lock".into()))?;

        traces.push_back(trace.clone());

        // Keep only recent traces
        while traces.len() > self.config.max_inference_traces {
            traces.pop_front();
        }

        // Update model info
        self.update_model_info(model_id, execution_time, memory_usage, anomalies)?;

        Ok(trace)
    }

    /// Validate a model structure
    pub fn validate_model(&self, model_id: &str) -> Result<ValidationStatus> {
        if !self.config.enable_structure_validation {
            return Ok(ValidationStatus::NotValidated);
        }

        let model_info = self
            .model_info
            .lock()
            .map_err(|_| TrustformersError::runtime_error("Failed to acquire lock".into()))?;

        let info = model_info
            .get(model_id)
            .ok_or_else(|| TrustformersError::runtime_error("Model not registered".into()))?;

        let mut warnings = Vec::new();
        let errors = Vec::new();

        // Check input/output shape consistency
        if info.input_shapes.is_empty() {
            warnings.push("No input shapes defined".to_string());
        }

        if info.output_shapes.is_empty() {
            warnings.push("No output shapes defined".to_string());
        }

        // Check parameter count
        if info.parameter_count == 0 {
            warnings.push("Model has no parameters".to_string());
        }

        // Check for extremely large models (mobile constraint)
        let estimated_memory = info.parameter_count * 4; // Assume FP32
        if estimated_memory > 100 * 1024 * 1024 {
            // 100MB
            warnings.push("Model may be too large for mobile deployment".to_string());
        }

        // Check inference performance
        if info.average_inference_time > Duration::from_millis(500) {
            warnings.push("Average inference time exceeds mobile-friendly threshold".to_string());
        }

        // Return validation status
        if !errors.is_empty() {
            Ok(ValidationStatus::Error(errors))
        } else if !warnings.is_empty() {
            Ok(ValidationStatus::Warning(warnings))
        } else {
            Ok(ValidationStatus::Valid)
        }
    }

    /// Get debugging information for a model
    pub fn get_model_debug_info(&self, model_id: &str) -> Result<ModelDebugInfo> {
        let model_info = self
            .model_info
            .lock()
            .map_err(|_| TrustformersError::runtime_error("Failed to acquire lock".into()))?;

        model_info
            .get(model_id)
            .cloned()
            .ok_or_else(|| TrustformersError::runtime_error("Model not registered".into()).into())
    }

    /// Get recent inference traces
    pub fn get_inference_traces(
        &self,
        model_id: Option<&str>,
        limit: Option<usize>,
    ) -> Result<Vec<InferenceTrace>> {
        let traces = self
            .inference_traces
            .lock()
            .map_err(|_| TrustformersError::runtime_error("Failed to acquire lock".into()))?;

        let filtered_traces: Vec<InferenceTrace> = traces
            .iter()
            .filter(|trace| {
                if let Some(id) = model_id {
                    // Extract model ID from trace ID (simplified)
                    trace.trace_id.contains(id)
                } else {
                    true
                }
            })
            .cloned()
            .collect();

        let result = if let Some(limit) = limit {
            filtered_traces.into_iter().rev().take(limit).collect()
        } else {
            filtered_traces
        };

        Ok(result)
    }

    /// Get detected anomalies
    pub fn get_anomalies(
        &self,
        model_id: Option<&str>,
        severity: Option<AnomalySeverity>,
    ) -> Result<Vec<ModelAnomaly>> {
        let model_info = self
            .model_info
            .lock()
            .map_err(|_| TrustformersError::runtime_error("Failed to acquire lock".into()))?;

        let mut all_anomalies = Vec::new();

        for (id, info) in model_info.iter() {
            if let Some(target_id) = model_id {
                if id != target_id {
                    continue;
                }
            }

            for anomaly in &info.anomalies_detected {
                if let Some(target_severity) = severity {
                    if anomaly.severity != target_severity {
                        continue;
                    }
                }
                all_anomalies.push(anomaly.clone());
            }
        }

        // Sort by timestamp (most recent first)
        all_anomalies.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));

        Ok(all_anomalies)
    }

    /// Generate debugging report
    pub fn generate_debug_report(&self, model_id: &str) -> Result<String> {
        let info = self.get_model_debug_info(model_id)?;
        let traces = self.get_inference_traces(Some(model_id), Some(10))?;
        let anomalies = self.get_anomalies(Some(model_id), None)?;

        let mut report = String::new();

        report.push_str(&format!("# Model Debug Report: {}\n\n", model_id));

        // Model information
        report.push_str("## Model Information\n");
        report.push_str(&format!("- Type: {}\n", info.model_type));
        report.push_str(&format!("- Parameters: {}\n", info.parameter_count));
        report.push_str(&format!(
            "- Memory Usage: {} MB\n",
            info.memory_usage / (1024 * 1024)
        ));
        report.push_str(&format!("- Inference Count: {}\n", info.inference_count));
        report.push_str(&format!(
            "- Average Inference Time: {:?}\n",
            info.average_inference_time
        ));
        report.push('\n');

        // Input/Output shapes
        report.push_str("## Model Structure\n");
        report.push_str("### Input Shapes\n");
        for (i, shape) in info.input_shapes.iter().enumerate() {
            report.push_str(&format!("- Input {}: {:?}\n", i, shape));
        }
        report.push_str("### Output Shapes\n");
        for (i, shape) in info.output_shapes.iter().enumerate() {
            report.push_str(&format!("- Output {}: {:?}\n", i, shape));
        }
        report.push('\n');

        // Validation status
        report.push_str("## Validation Status\n");
        match &info.validation_status {
            ValidationStatus::Valid => report.push_str("✅ Model validation passed\n"),
            ValidationStatus::Warning(warnings) => {
                report.push_str("⚠️ Model validation warnings:\n");
                for warning in warnings {
                    report.push_str(&format!("  - {}\n", warning));
                }
            },
            ValidationStatus::Error(errors) => {
                report.push_str("❌ Model validation errors:\n");
                for error in errors {
                    report.push_str(&format!("  - {}\n", error));
                }
            },
            ValidationStatus::NotValidated => report.push_str("❓ Model not validated\n"),
        }
        report.push('\n');

        // Anomalies
        if !anomalies.is_empty() {
            report.push_str("## Detected Anomalies\n");
            for anomaly in anomalies.iter().take(5) {
                let severity_emoji = match anomaly.severity {
                    AnomalySeverity::Info => "ℹ️",
                    AnomalySeverity::Warning => "⚠️",
                    AnomalySeverity::Error => "❌",
                    AnomalySeverity::Critical => "🔥",
                };
                report.push_str(&format!(
                    "{} {:?}: {}\n",
                    severity_emoji, anomaly.anomaly_type, anomaly.description
                ));
                report.push_str(&format!(
                    "   Suggested action: {}\n",
                    anomaly.suggested_action
                ));
            }
            report.push('\n');
        }

        // Recent traces
        if !traces.is_empty() {
            report.push_str("## Recent Inference Traces\n");
            for trace in traces.iter().take(5) {
                report.push_str(&format!(
                    "- Trace {}: {:?} execution time, {} MB memory\n",
                    trace.trace_id,
                    trace.execution_time,
                    trace.memory_usage / (1024 * 1024)
                ));
                if !trace.anomalies.is_empty() {
                    report.push_str(&format!("  Anomalies: {}\n", trace.anomalies.len()));
                }
            }
        }

        Ok(report)
    }

    /// Clear debugging data
    pub fn clear_debug_data(&self, model_id: Option<&str>) -> Result<()> {
        if let Some(id) = model_id {
            // Clear data for specific model
            let mut traces = self
                .inference_traces
                .lock()
                .map_err(|_| TrustformersError::runtime_error("Failed to acquire lock".into()))?;

            traces.retain(|trace| !trace.trace_id.contains(id));

            let mut model_info = self
                .model_info
                .lock()
                .map_err(|_| TrustformersError::runtime_error("Failed to acquire lock".into()))?;

            if let Some(info) = model_info.get_mut(id) {
                info.anomalies_detected.clear();
                info.inference_count = 0;
            }
        } else {
            // Clear all data
            let mut traces = self
                .inference_traces
                .lock()
                .map_err(|_| TrustformersError::runtime_error("Failed to acquire lock".into()))?;
            traces.clear();

            let mut model_info = self
                .model_info
                .lock()
                .map_err(|_| TrustformersError::runtime_error("Failed to acquire lock".into()))?;

            for info in model_info.values_mut() {
                info.anomalies_detected.clear();
                info.inference_count = 0;
            }
        }

        Ok(())
    }

    // Private helper methods

    fn analyze_tensor(&self, tensor: &Tensor, name: &str) -> Result<TensorDebugInfo> {
        let data = tensor.data()?;
        let shape = tensor.shape().to_vec();

        if data.is_empty() {
            return Ok(TensorDebugInfo {
                name: name.to_string(),
                shape,
                dtype: "f32".to_string(),
                min_value: 0.0,
                max_value: 0.0,
                mean_value: 0.0,
                std_dev: 0.0,
                zero_count: 0,
                nan_count: 0,
                inf_count: 0,
                memory_size: 0,
                sparsity: 0.0,
            });
        }

        let min_value = data.iter().fold(f32::INFINITY, |acc, &val| acc.min(val));
        let max_value = data.iter().fold(f32::NEG_INFINITY, |acc, &val| acc.max(val));
        let mean_value = data.iter().sum::<f32>() / data.len() as f32;

        let variance =
            data.iter().map(|&val| (val - mean_value).powi(2)).sum::<f32>() / data.len() as f32;
        let std_dev = variance.sqrt();

        let zero_count = data.iter().filter(|&&val| val == 0.0).count();
        let nan_count = data.iter().filter(|&&val| val.is_nan()).count();
        let inf_count = data.iter().filter(|&&val| val.is_infinite()).count();

        let memory_size = data.len() * std::mem::size_of::<f32>();
        let sparsity = zero_count as f32 / data.len() as f32;

        Ok(TensorDebugInfo {
            name: name.to_string(),
            shape,
            dtype: "f32".to_string(),
            min_value,
            max_value,
            mean_value,
            std_dev,
            zero_count,
            nan_count,
            inf_count,
            memory_size,
            sparsity,
        })
    }

    fn detect_tensor_anomalies(
        &self,
        model_id: &str,
        tensor_infos: &[TensorDebugInfo],
    ) -> Result<Vec<ModelAnomaly>> {
        let mut anomalies = Vec::new();
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

        for tensor_info in tensor_infos {
            // Check for NaN values
            if tensor_info.nan_count > 0 {
                anomalies.push(ModelAnomaly {
                    anomaly_type: AnomalyType::NaNValues,
                    severity: AnomalySeverity::Error,
                    description: format!(
                        "Tensor '{}' contains {} NaN values",
                        tensor_info.name, tensor_info.nan_count
                    ),
                    location: format!("model:{}, tensor:{}", model_id, tensor_info.name),
                    timestamp,
                    suggested_action: "Check for division by zero or invalid operations"
                        .to_string(),
                    tensor_info: Some(tensor_info.clone()),
                });
            }

            // Check for infinite values
            if tensor_info.inf_count > 0 {
                anomalies.push(ModelAnomaly {
                    anomaly_type: AnomalyType::InfiniteValues,
                    severity: AnomalySeverity::Error,
                    description: format!(
                        "Tensor '{}' contains {} infinite values",
                        tensor_info.name, tensor_info.inf_count
                    ),
                    location: format!("model:{}, tensor:{}", model_id, tensor_info.name),
                    timestamp,
                    suggested_action: "Check for overflow or invalid mathematical operations"
                        .to_string(),
                    tensor_info: Some(tensor_info.clone()),
                });
            }

            // Check for extreme values
            let (min_normal, max_normal) = self.config.normal_value_range;
            if tensor_info.min_value < min_normal || tensor_info.max_value > max_normal {
                anomalies.push(ModelAnomaly {
                    anomaly_type: AnomalyType::ExtremeValues,
                    severity: AnomalySeverity::Warning,
                    description: format!(
                        "Tensor '{}' has extreme values: min={}, max={}",
                        tensor_info.name, tensor_info.min_value, tensor_info.max_value
                    ),
                    location: format!("model:{}, tensor:{}", model_id, tensor_info.name),
                    timestamp,
                    suggested_action: "Consider value normalization or clipping".to_string(),
                    tensor_info: Some(tensor_info.clone()),
                });
            }
        }

        Ok(anomalies)
    }

    fn detect_performance_anomalies(
        &self,
        model_id: &str,
        execution_time: Duration,
        memory_usage: usize,
    ) -> Result<Vec<ModelAnomaly>> {
        let mut anomalies = Vec::new();
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();

        // Check for slow inference
        if execution_time > self.config.slow_inference_threshold {
            anomalies.push(ModelAnomaly {
                anomaly_type: AnomalyType::PerformanceDegradation,
                severity: AnomalySeverity::Warning,
                description: format!("Slow inference detected: {:?}", execution_time),
                location: format!("model:{}", model_id),
                timestamp,
                suggested_action: "Consider model optimization or hardware acceleration"
                    .to_string(),
                tensor_info: None,
            });
        }

        // Check for high memory usage (simplified threshold)
        let high_memory_threshold = 50 * 1024 * 1024; // 50MB
        if memory_usage > high_memory_threshold {
            anomalies.push(ModelAnomaly {
                anomaly_type: AnomalyType::MemorySpike,
                severity: AnomalySeverity::Warning,
                description: format!(
                    "High memory usage detected: {} MB",
                    memory_usage / (1024 * 1024)
                ),
                location: format!("model:{}", model_id),
                timestamp,
                suggested_action: "Consider memory optimization or model compression".to_string(),
                tensor_info: None,
            });
        }

        Ok(anomalies)
    }

    fn update_model_info(
        &self,
        model_id: &str,
        execution_time: Duration,
        memory_usage: usize,
        anomalies: Vec<ModelAnomaly>,
    ) -> Result<()> {
        let mut model_info = self
            .model_info
            .lock()
            .map_err(|_| TrustformersError::runtime_error("Failed to acquire lock".into()))?;

        if let Some(info) = model_info.get_mut(model_id) {
            info.inference_count += 1;
            info.last_inference_time = execution_time;

            // Update running average
            let alpha = 0.1; // Smoothing factor
            if info.inference_count == 1 {
                info.average_inference_time = execution_time;
            } else {
                let current_avg_ms = info.average_inference_time.as_millis() as f64;
                let new_time_ms = execution_time.as_millis() as f64;
                let new_avg_ms = alpha * new_time_ms + (1.0 - alpha) * current_avg_ms;
                info.average_inference_time = Duration::from_millis(new_avg_ms as u64);
            }

            info.memory_usage = memory_usage;
            info.anomalies_detected.extend(anomalies);

            // Keep only recent anomalies (last 100)
            if info.anomalies_detected.len() > 100 {
                info.anomalies_detected.drain(0..info.anomalies_detected.len() - 100);
            }
        }

        Ok(())
    }
}

impl Default for PerformanceProfile {
    fn default() -> Self {
        Self {
            total_inference_time: Duration::from_millis(0),
            preprocessing_time: Duration::from_millis(0),
            model_execution_time: Duration::from_millis(0),
            postprocessing_time: Duration::from_millis(0),
            memory_allocation_time: Duration::from_millis(0),
            memory_peak: 0,
            cpu_usage: 0.0,
            gpu_usage: 0.0,
            bottlenecks: Vec::new(),
        }
    }
}

impl Default for TensorStats {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 0.0,
            min_val: 0.0,
            max_val: 0.0,
            nan_rate: 0.0,
            inf_rate: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_debugger_creation() {
        let config = ModelDebuggerConfig::default();
        let debugger = MobileModelDebugger::new(config);
        assert!(debugger.config.enabled);
    }

    #[test]
    fn test_model_registration() {
        let config = ModelDebuggerConfig::default();
        let debugger = MobileModelDebugger::new(config);

        let result = debugger.register_model(
            "test_model".to_string(),
            "transformer".to_string(),
            vec![vec![1, 224, 224, 3]],
            vec![vec![1, 1000]],
            1000000,
        );

        assert!(result.is_ok());

        let info = debugger.get_model_debug_info("test_model").expect("Operation failed");
        assert_eq!(info.model_id, "test_model");
        assert_eq!(info.model_type, "transformer");
        assert_eq!(info.parameter_count, 1000000);
    }

    #[test]
    fn test_tensor_analysis() {
        let config = ModelDebuggerConfig::default();
        let debugger = MobileModelDebugger::new(config);

        // Create test tensor with some problematic values
        let data = vec![1.0, 2.0, f32::NAN, 4.0, f32::INFINITY, 0.0];
        let tensor = Tensor::from_vec(data, &[6]).expect("Operation failed");

        let analysis = debugger.analyze_tensor(&tensor, "test_tensor").expect("Operation failed");

        assert_eq!(analysis.name, "test_tensor");
        assert_eq!(analysis.nan_count, 1);
        assert_eq!(analysis.inf_count, 1);
        assert_eq!(analysis.zero_count, 1);
    }

    #[test]
    fn test_anomaly_detection() {
        let config = ModelDebuggerConfig::default();
        let debugger = MobileModelDebugger::new(config);

        // Register model
        debugger
            .register_model(
                "test_model".to_string(),
                "test".to_string(),
                vec![vec![1, 4]],
                vec![vec![1, 1]],
                100,
            )
            .expect("Operation failed");

        // Create tensors with anomalies
        let input_data = vec![1.0, 2.0, f32::NAN, 4.0];
        let output_data = vec![f32::INFINITY];

        let input_tensor = Tensor::from_vec(input_data, &[1, 4]).expect("Operation failed");
        let output_tensor = Tensor::from_vec(output_data, &[1, 1]).expect("Operation failed");

        let trace = debugger
            .debug_inference(
                "test_model",
                &[input_tensor],
                &[output_tensor],
                Duration::from_millis(50),
                1024 * 1024,
            )
            .expect("Operation failed");

        // Should detect NaN in input and infinity in output
        assert!(!trace.anomalies.is_empty());
        assert!(trace.anomalies.iter().any(|a| a.anomaly_type == AnomalyType::NaNValues));
        assert!(trace.anomalies.iter().any(|a| a.anomaly_type == AnomalyType::InfiniteValues));
    }

    #[test]
    fn test_model_validation() {
        let config = ModelDebuggerConfig::default();
        let debugger = MobileModelDebugger::new(config);

        // Register a large model that should trigger warnings
        debugger
            .register_model(
                "large_model".to_string(),
                "large".to_string(),
                vec![vec![1, 1000, 1000, 3]],
                vec![vec![1, 10000]],
                100_000_000, // 100M parameters
            )
            .expect("Operation failed");

        let validation = debugger.validate_model("large_model").expect("Operation failed");

        match validation {
            ValidationStatus::Warning(warnings) => {
                assert!(warnings.iter().any(|w| w.contains("too large for mobile")));
            },
            status => panic!(
                "Expected ValidationStatus::Warning for large model, got {:?}",
                status
            ),
        }
    }
}
