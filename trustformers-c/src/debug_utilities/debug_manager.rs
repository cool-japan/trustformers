//! Main debug utilities implementation
//!
//! This module contains the core debugging and profiling functionality.

use super::types::*;
use crate::error::{TrustformersError, TrustformersResult};
use crate::memory_safety::MemorySafetyVerifier;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Main debug utilities manager
pub struct DebugUtilities {
    pub sessions: HashMap<String, DebugSession>,
    pub model_introspections: HashMap<String, ModelIntrospection>,
    pub global_profiling: bool,
}

impl DebugUtilities {
    /// Create a new debug utilities manager
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
            model_introspections: HashMap::new(),
            global_profiling: false,
        }
    }

    /// Start a new debug session
    pub fn start_debug_session(&mut self, session_id: &str) -> TrustformersResult<()> {
        if self.sessions.contains_key(session_id) {
            return Err(TrustformersError::InvalidParameter);
        }

        let session = DebugSession {
            session_id: session_id.to_string(),
            start_time: Instant::now(),
            profiling_data: Arc::new(Mutex::new(ProfilingData {
                session_id: session_id.to_string(),
                total_duration_ms: 0.0,
                layer_timings: Vec::new(),
                memory_snapshots: Vec::new(),
                tensor_operations: Vec::new(),
                bottlenecks: Vec::new(),
            })),
            memory_tracker: MemorySafetyVerifier::new(
                crate::memory_safety::MemorySafetyConfig::default(),
            ),
            is_active: true,
        };

        self.sessions.insert(session_id.to_string(), session);
        Ok(())
    }

    /// Stop a debug session and return profiling data
    pub fn stop_debug_session(&mut self, session_id: &str) -> TrustformersResult<ProfilingData> {
        let session =
            self.sessions.remove(session_id).ok_or(TrustformersError::InvalidParameter)?;

        let elapsed = session.start_time.elapsed();
        let mut profiling_data = session.profiling_data.lock().unwrap();
        profiling_data.total_duration_ms = elapsed.as_secs_f64() * 1000.0;

        // Analyze bottlenecks
        self.analyze_bottlenecks(&mut profiling_data);

        Ok(profiling_data.clone())
    }

    /// Introspect a model and return detailed information
    pub fn introspect_model(
        &mut self,
        model_id: &str,
        model_ptr: *const std::ffi::c_void,
    ) -> TrustformersResult<ModelIntrospection> {
        // In a real implementation, this would analyze the model structure
        // For demonstration, we'll create a mock introspection

        let introspection = ModelIntrospection {
            model_id: model_id.to_string(),
            model_name: format!("Model_{}", model_id),
            model_type: "Transformer".to_string(),
            parameters_count: 175_000_000, // Example: GPT-3 scale
            layers_count: 96,
            input_shape: vec![1, 512], // [batch_size, sequence_length]
            output_shape: vec![1, 512, 50257], // [batch_size, sequence_length, vocab_size]
            memory_usage_bytes: 350_000_000, // ~350MB
            created_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::from_secs(0))
                .as_secs()
                .to_string(),
            last_used: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or(Duration::from_secs(0))
                .as_secs()
                .to_string(),
            usage_count: 0,
            avg_inference_time_ms: 0.0,
            layer_info: self.generate_layer_info(),
            quantization_info: Some(QuantizationInfo {
                is_quantized: false,
                quantization_bits: 32,
                quantization_method: "None".to_string(),
                compression_ratio: 1.0,
                accuracy_loss: None,
            }),
        };

        self.model_introspections.insert(model_id.to_string(), introspection.clone());
        Ok(introspection)
    }

    /// Generate model architecture visualization
    pub fn generate_model_visualization(
        &self,
        model_id: &str,
    ) -> TrustformersResult<ModelVisualization> {
        let introspection = self
            .model_introspections
            .get(model_id)
            .ok_or(TrustformersError::InvalidParameter)?;

        let mut nodes = Vec::new();
        let mut edges = Vec::new();

        // Create input node
        nodes.push(VisualizationNode {
            id: "input".to_string(),
            label: "Input".to_string(),
            node_type: "input".to_string(),
            position: Some((0.0, 0.0)),
            size: (100.0, 50.0),
            color: "#4CAF50".to_string(),
            metadata: HashMap::new(),
        });

        // Create layer nodes
        for (i, layer) in introspection.layer_info.iter().enumerate() {
            let node_id = format!("layer_{}", i);
            nodes.push(VisualizationNode {
                id: node_id.clone(),
                label: layer.name.clone(),
                node_type: layer.layer_type.clone(),
                position: Some((0.0, (i + 1) as f64 * 100.0)),
                size: (150.0, 75.0),
                color: self.get_layer_color(&layer.layer_type),
                metadata: {
                    let mut meta = HashMap::new();
                    meta.insert("parameters".to_string(), layer.parameters_count.to_string());
                    meta.insert(
                        "memory_mb".to_string(),
                        (layer.memory_usage_bytes / 1_000_000).to_string(),
                    );
                    meta
                },
            });

            // Create edge from previous layer
            let prev_id = if i == 0 { "input".to_string() } else { format!("layer_{}", i - 1) };
            edges.push(VisualizationEdge {
                from: prev_id,
                to: node_id,
                label: Some(format!("{}x{}", layer.input_shape[0], layer.input_shape[1])),
                edge_type: "dataflow".to_string(),
                weight: 1.0,
                color: "#666666".to_string(),
            });
        }

        // Create output node
        nodes.push(VisualizationNode {
            id: "output".to_string(),
            label: "Output".to_string(),
            node_type: "output".to_string(),
            position: Some((0.0, (introspection.layers_count + 1) as f64 * 100.0)),
            size: (100.0, 50.0),
            color: "#F44336".to_string(),
            metadata: HashMap::new(),
        });

        // Edge from last layer to output
        if !introspection.layer_info.is_empty() {
            edges.push(VisualizationEdge {
                from: format!("layer_{}", introspection.layers_count - 1),
                to: "output".to_string(),
                label: None,
                edge_type: "dataflow".to_string(),
                weight: 1.0,
                color: "#666666".to_string(),
            });
        }

        let mut layout_hints = HashMap::new();
        layout_hints.insert("algorithm".to_string(), "hierarchical".to_string());
        layout_hints.insert("direction".to_string(), "top_to_bottom".to_string());

        Ok(ModelVisualization {
            nodes,
            edges,
            layout_hints,
        })
    }

    /// Record tensor operation for profiling
    pub fn record_tensor_operation(
        &mut self,
        session_id: &str,
        operation_type: &str,
        input_shapes: Vec<Vec<usize>>,
        output_shape: Vec<usize>,
        duration: Duration,
        memory_delta: i64,
    ) -> TrustformersResult<()> {
        let session = self.sessions.get(session_id).ok_or(TrustformersError::InvalidParameter)?;

        let flops = self.estimate_flops(operation_type, &input_shapes, &output_shape);
        let operation = TensorOperation {
            operation_type: operation_type.to_string(),
            input_shapes,
            output_shape,
            duration_ms: duration.as_secs_f64() * 1000.0,
            memory_delta_bytes: memory_delta,
            flops,
        };

        session.profiling_data.lock().unwrap().tensor_operations.push(operation);
        Ok(())
    }

    /// Take a memory snapshot
    pub fn take_memory_snapshot(&mut self, session_id: &str) -> TrustformersResult<()> {
        let session = self.sessions.get(session_id).ok_or(TrustformersError::InvalidParameter)?;

        let memory_tracker = &session.memory_tracker;
        let stats = memory_tracker.get_statistics();
        let breakdown = memory_tracker.get_resource_breakdown();
        let snapshot = MemorySnapshot {
            timestamp_ms: session.start_time.elapsed().as_secs_f64() * 1000.0,
            total_allocated_bytes: breakdown.total_memory as u64,
            peak_allocated_bytes: breakdown.total_memory as u64, // Use same value as approximation
            gpu_memory_bytes: None, // Would be filled with actual GPU memory info
            tensor_count: stats.current_allocations,
            fragmentation_ratio: 0.0, // Would be calculated if available
        };

        session.profiling_data.lock().unwrap().memory_snapshots.push(snapshot);
        Ok(())
    }

    /// Export debug data to various formats
    pub fn export_debug_data(&self, session_id: &str, format: &str) -> TrustformersResult<String> {
        let session = self.sessions.get(session_id).ok_or(TrustformersError::InvalidParameter)?;

        let profiling_data = session.profiling_data.lock().unwrap();

        match format.to_lowercase().as_str() {
            "json" => serde_json::to_string_pretty(&*profiling_data)
                .map_err(|_| TrustformersError::SerializationError),
            "csv" => self.export_to_csv(&profiling_data),
            "html" => self.export_to_html(&profiling_data),
            _ => Err(TrustformersError::InvalidParameter),
        }
    }

    /// Generate performance report
    pub fn generate_performance_report(&self, session_id: &str) -> TrustformersResult<String> {
        let session = self.sessions.get(session_id).ok_or(TrustformersError::InvalidParameter)?;

        let profiling_data = session.profiling_data.lock().unwrap();

        let mut report = String::new();
        report.push_str("# Performance Analysis Report\n\n");

        // Summary
        report.push_str("## Summary\n");
        report.push_str(&format!("- Session ID: {}\n", profiling_data.session_id));
        report.push_str(&format!(
            "- Total Duration: {:.2} ms\n",
            profiling_data.total_duration_ms
        ));
        report.push_str(&format!(
            "- Tensor Operations: {}\n",
            profiling_data.tensor_operations.len()
        ));
        report.push_str(&format!(
            "- Memory Snapshots: {}\n",
            profiling_data.memory_snapshots.len()
        ));
        report.push_str(&format!(
            "- Bottlenecks Found: {}\n\n",
            profiling_data.bottlenecks.len()
        ));

        // Layer timings
        if !profiling_data.layer_timings.is_empty() {
            report.push_str("## Layer Performance\n");
            for timing in &profiling_data.layer_timings {
                report.push_str(&format!(
                    "- {}: {:.2} ms (utilization: {:.1}%, cache hit: {:.1}%)\n",
                    timing.layer_name,
                    timing.forward_pass_ms,
                    timing.compute_utilization * 100.0,
                    timing.cache_hit_rate * 100.0
                ));
            }
            report.push('\n');
        }

        // Memory analysis
        if let (Some(first), Some(last)) = (
            profiling_data.memory_snapshots.first(),
            profiling_data.memory_snapshots.last(),
        ) {
            report.push_str("## Memory Analysis\n");
            report.push_str(&format!(
                "- Initial Memory: {:.2} MB\n",
                first.total_allocated_bytes as f64 / 1_000_000.0
            ));
            report.push_str(&format!(
                "- Final Memory: {:.2} MB\n",
                last.total_allocated_bytes as f64 / 1_000_000.0
            ));
            report.push_str(&format!(
                "- Peak Memory: {:.2} MB\n",
                last.peak_allocated_bytes as f64 / 1_000_000.0
            ));
            report.push_str(&format!(
                "- Final Fragmentation: {:.1}%\n\n",
                last.fragmentation_ratio * 100.0
            ));
        }

        // Bottlenecks
        if !profiling_data.bottlenecks.is_empty() {
            report.push_str("## Performance Bottlenecks\n");
            for bottleneck in &profiling_data.bottlenecks {
                report.push_str(&format!(
                    "### {} (Severity: {:.1}%)\n",
                    bottleneck.location,
                    bottleneck.severity * 100.0
                ));
                report.push_str(&format!("- Type: {}\n", bottleneck.bottleneck_type));
                report.push_str(&format!("- Description: {}\n", bottleneck.description));
                report.push_str(&format!(
                    "- Suggested Fix: {}\n\n",
                    bottleneck.suggested_fix
                ));
            }
        }

        // Recommendations
        report.push_str("## Recommendations\n");
        report.push_str(&self.generate_recommendations(&profiling_data));

        Ok(report)
    }

    // Helper methods

    fn generate_layer_info(&self) -> Vec<LayerInfo> {
        // Mock layer information for demonstration
        vec![
            LayerInfo {
                name: "embedding".to_string(),
                layer_type: "Embedding".to_string(),
                input_shape: vec![1, 512],
                output_shape: vec![1, 512, 768],
                parameters_count: 38_597_376,
                memory_usage_bytes: 154_389_504,
                activation_function: None,
                is_trainable: true,
            },
            LayerInfo {
                name: "transformer_block_0".to_string(),
                layer_type: "TransformerBlock".to_string(),
                input_shape: vec![1, 512, 768],
                output_shape: vec![1, 512, 768],
                parameters_count: 7_087_872,
                memory_usage_bytes: 28_351_488,
                activation_function: Some("GELU".to_string()),
                is_trainable: true,
            },
            LayerInfo {
                name: "layer_norm".to_string(),
                layer_type: "LayerNorm".to_string(),
                input_shape: vec![1, 512, 768],
                output_shape: vec![1, 512, 768],
                parameters_count: 1536,
                memory_usage_bytes: 6144,
                activation_function: None,
                is_trainable: true,
            },
            LayerInfo {
                name: "output_projection".to_string(),
                layer_type: "Linear".to_string(),
                input_shape: vec![1, 512, 768],
                output_shape: vec![1, 512, 50257],
                parameters_count: 38_597_376,
                memory_usage_bytes: 154_389_504,
                activation_function: None,
                is_trainable: true,
            },
        ]
    }

    fn get_layer_color(&self, layer_type: &str) -> String {
        match layer_type {
            "Embedding" => "#9C27B0".to_string(),
            "TransformerBlock" => "#2196F3".to_string(),
            "LayerNorm" => "#FF9800".to_string(),
            "Linear" => "#795548".to_string(),
            "Attention" => "#E91E63".to_string(),
            _ => "#9E9E9E".to_string(),
        }
    }

    fn estimate_flops(
        &self,
        operation_type: &str,
        input_shapes: &[Vec<usize>],
        output_shape: &[usize],
    ) -> Option<u64> {
        // Simplified FLOPS estimation
        match operation_type {
            "matmul" => {
                if input_shapes.len() >= 2 {
                    let m =
                        input_shapes[0].iter().take(input_shapes[0].len() - 1).product::<usize>();
                    let k = input_shapes[0].last().unwrap_or(&1);
                    let n = input_shapes[1].last().unwrap_or(&1);
                    Some((2 * m * k * n) as u64)
                } else {
                    None
                }
            },
            "conv2d" => {
                let output_elements: usize = output_shape.iter().product();
                Some((output_elements * 2) as u64) // Simplified
            },
            _ => None,
        }
    }

    fn analyze_bottlenecks(&self, profiling_data: &mut ProfilingData) {
        profiling_data.bottlenecks.clear();

        // Analyze layer timings for slow layers
        for timing in &profiling_data.layer_timings {
            if timing.forward_pass_ms > 100.0 {
                profiling_data.bottlenecks.push(PerformanceBottleneck {
                    location: timing.layer_name.clone(),
                    bottleneck_type: "Slow Layer".to_string(),
                    severity: (timing.forward_pass_ms / 1000.0).min(1.0),
                    description: format!("Layer takes {:.2} ms to execute", timing.forward_pass_ms),
                    suggested_fix: "Consider quantization or layer pruning".to_string(),
                });
            }

            if timing.cache_hit_rate < 0.5 {
                profiling_data.bottlenecks.push(PerformanceBottleneck {
                    location: timing.layer_name.clone(),
                    bottleneck_type: "Poor Cache Performance".to_string(),
                    severity: 1.0 - timing.cache_hit_rate,
                    description: format!(
                        "Cache hit rate is only {:.1}%",
                        timing.cache_hit_rate * 100.0
                    ),
                    suggested_fix: "Optimize memory access patterns or increase cache size"
                        .to_string(),
                });
            }
        }

        // Analyze memory growth
        if profiling_data.memory_snapshots.len() >= 2 {
            let first = &profiling_data.memory_snapshots[0];
            let last = &profiling_data.memory_snapshots[profiling_data.memory_snapshots.len() - 1];

            let memory_growth = (last.total_allocated_bytes as f64
                - first.total_allocated_bytes as f64)
                / first.total_allocated_bytes as f64;

            if memory_growth > 0.5 {
                profiling_data.bottlenecks.push(PerformanceBottleneck {
                    location: "Memory System".to_string(),
                    bottleneck_type: "Memory Growth".to_string(),
                    severity: memory_growth.min(1.0),
                    description: format!("Memory usage grew by {:.1}%", memory_growth * 100.0),
                    suggested_fix: "Check for memory leaks or optimize tensor lifecycles"
                        .to_string(),
                });
            }

            if last.fragmentation_ratio > 0.3 {
                profiling_data.bottlenecks.push(PerformanceBottleneck {
                    location: "Memory System".to_string(),
                    bottleneck_type: "Memory Fragmentation".to_string(),
                    severity: last.fragmentation_ratio,
                    description: format!(
                        "Memory fragmentation is {:.1}%",
                        last.fragmentation_ratio * 100.0
                    ),
                    suggested_fix: "Use memory pooling or adjust allocation patterns".to_string(),
                });
            }
        }
    }

    fn export_to_csv(&self, profiling_data: &ProfilingData) -> TrustformersResult<String> {
        let mut csv = String::new();

        // Layer timings CSV
        csv.push_str(
            "layer_name,forward_pass_ms,memory_allocation_ms,compute_utilization,cache_hit_rate\n",
        );
        for timing in &profiling_data.layer_timings {
            csv.push_str(&format!(
                "{},{},{},{},{}\n",
                timing.layer_name,
                timing.forward_pass_ms,
                timing.memory_allocation_ms,
                timing.compute_utilization,
                timing.cache_hit_rate
            ));
        }

        Ok(csv)
    }

    fn export_to_html(&self, profiling_data: &ProfilingData) -> TrustformersResult<String> {
        let mut html = String::new();
        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str("<title>TrustformeRS Debug Report</title>\n");
        html.push_str("<style>\n");
        html.push_str("body { font-family: Arial, sans-serif; margin: 40px; }\n");
        html.push_str("table { border-collapse: collapse; width: 100%; }\n");
        html.push_str("th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n");
        html.push_str("th { background-color: #f2f2f2; }\n");
        html.push_str("</style>\n</head>\n<body>\n");

        html.push_str(&format!(
            "<h1>Debug Report - {}</h1>\n",
            profiling_data.session_id
        ));
        html.push_str(&format!(
            "<p>Total Duration: {:.2} ms</p>\n",
            profiling_data.total_duration_ms
        ));

        // Layer timings table
        if !profiling_data.layer_timings.is_empty() {
            html.push_str("<h2>Layer Performance</h2>\n");
            html.push_str("<table>\n");
            html.push_str("<tr><th>Layer</th><th>Time (ms)</th><th>Utilization</th><th>Cache Hit Rate</th></tr>\n");
            for timing in &profiling_data.layer_timings {
                html.push_str(&format!(
                    "<tr><td>{}</td><td>{:.2}</td><td>{:.1}%</td><td>{:.1}%</td></tr>\n",
                    timing.layer_name,
                    timing.forward_pass_ms,
                    timing.compute_utilization * 100.0,
                    timing.cache_hit_rate * 100.0
                ));
            }
            html.push_str("</table>\n");
        }

        html.push_str("</body>\n</html>");
        Ok(html)
    }

    fn generate_recommendations(&self, profiling_data: &ProfilingData) -> String {
        let mut recommendations = String::new();

        if profiling_data.total_duration_ms > 1000.0 {
            recommendations.push_str("- Consider model quantization to reduce inference time\n");
        }

        if profiling_data
            .memory_snapshots
            .last()
            .is_some_and(|s| s.fragmentation_ratio > 0.2)
        {
            recommendations.push_str("- Implement memory pooling to reduce fragmentation\n");
        }

        if profiling_data.layer_timings.iter().any(|t| t.cache_hit_rate < 0.6) {
            recommendations.push_str("- Optimize data locality to improve cache performance\n");
        }

        if recommendations.is_empty() {
            recommendations.push_str("- No major performance issues detected\n");
        }

        recommendations
    }
}
