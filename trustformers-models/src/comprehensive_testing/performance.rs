//! Performance profiling for models.
//!
//! Every number this profiler reports is measured: wall-clock time comes from
//! [`Instant`], memory from the operating system through the same `sysinfo`
//! reader the memory profiler uses. Quantities that would need instrumentation
//! this profiler does not have (per-layer timings, FLOP counts, fragmentation)
//! are `None` rather than plausible constants.

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Model;

use super::config::{TestDataType, TestInputConfig, ValidationConfig};
use super::types::{
    LayerPerformance, MemoryAnalysis, OverallPerformance, PerformanceResults,
    ThroughputMeasurements,
};

/// Performance profiler for models
pub struct PerformanceProfiler {
    config: ValidationConfig,
}

impl PerformanceProfiler {
    /// Create a new performance profiler
    pub fn new() -> Self {
        Self {
            config: ValidationConfig::default(),
        }
    }

    /// Create profiler with custom configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Profile model performance by running the configured test inputs.
    ///
    /// For every configured input the profiler runs one real forward pass,
    /// measures its wall-clock duration and the process's resident-set size
    /// before and after, and derives throughput from the measured time and the
    /// real token count of the inputs.
    ///
    /// # Errors
    ///
    /// Returns an error when no test inputs are configured (there would be
    /// nothing to measure) or when a forward pass fails.
    pub fn profile_model<M: Model<Input = Tensor, Output = Tensor>>(
        &self,
        model: &M,
    ) -> Result<PerformanceResults> {
        if self.config.test_inputs.is_empty() {
            return Err(anyhow!(
                "performance profiling needs at least one configured test input"
            ));
        }

        let mut layer_performance = Vec::with_capacity(self.config.test_inputs.len());
        let mut total_time = Duration::ZERO;
        let mut total_tokens: u64 = 0;
        let mut total_samples: u64 = 0;
        let mut memory_samples: Vec<f64> = Vec::new();

        // Profile each test input
        for test_input in &self.config.test_inputs {
            let input = self.create_test_input(test_input)?;
            let shape = input.shape();
            let elements: usize = shape.iter().product();
            let batch_size = shape.first().copied().unwrap_or(1);
            let tokens_in_input = elements as u64;

            let memory_before = current_resident_mb();
            let start_time = Instant::now();
            let _output = model.forward(input)?;
            let inference_time = start_time.elapsed();
            let memory_after = current_resident_mb();

            total_time += inference_time;
            total_tokens += tokens_in_input;
            total_samples += batch_size as u64;

            if let Some(after) = memory_after {
                memory_samples.push(after);
            }

            layer_performance.push(LayerPerformance {
                layer_name: format!("forward_pass[{}]", test_input.name),
                layer_type: "full_model_forward".to_string(),
                forward_time: inference_time,
                memory_usage_mb: match (memory_before, memory_after) {
                    (Some(before), Some(after)) => Some(after - before),
                    _ => None,
                },
                // No layer-level hooks exist, so these stay unmeasured.
                flops: None,
                utilization_percent: None,
            });
        }

        let elapsed_seconds = total_time.as_secs_f64();
        let tokens_per_second =
            if elapsed_seconds > 0.0 { total_tokens as f64 / elapsed_seconds } else { 0.0 };
        let samples_per_second =
            if elapsed_seconds > 0.0 { total_samples as f64 / elapsed_seconds } else { 0.0 };
        let latency_per_token_ms = if total_tokens > 0 {
            total_time.as_secs_f64() * 1000.0 / total_tokens as f64
        } else {
            0.0
        };

        let peak_memory_mb =
            memory_samples.iter().copied().fold(None::<f64>, |acc, sample| match acc {
                Some(current) => Some(current.max(sample)),
                None => Some(sample),
            });
        let average_memory_mb = if memory_samples.is_empty() {
            None
        } else {
            Some(memory_samples.iter().sum::<f64>() / memory_samples.len() as f64)
        };

        let overall_performance = OverallPerformance {
            total_inference_time: total_time,
            tokens_per_second: tokens_per_second as f32,
            total_flops: None,
            peak_memory_mb,
            average_memory_mb,
        };

        let memory_analysis = MemoryAnalysis {
            by_layer_type: HashMap::new(),
            by_tensor_type: HashMap::new(),
            // Both need allocator-level instrumentation, which this profiler
            // deliberately does not fake.
            efficiency_score: None,
            fragmentation_percent: None,
        };

        // Report the shape the throughput was actually measured on.
        let last_input = self.config.test_inputs.last().ok_or_else(|| {
            anyhow!("performance profiling needs at least one configured test input")
        })?;
        let throughput = ThroughputMeasurements {
            batch_size: last_input.dimensions.first().copied().unwrap_or(1),
            sequence_length: last_input.dimensions.get(1).copied().unwrap_or(1),
            tokens_per_second: tokens_per_second as f32,
            samples_per_second: samples_per_second as f32,
            latency_per_token_ms: latency_per_token_ms as f32,
        };

        Ok(PerformanceResults {
            layer_performance,
            overall_performance,
            memory_analysis,
            throughput,
        })
    }

    /// Create test input (helper method)
    fn create_test_input(&self, config: &TestInputConfig) -> Result<Tensor> {
        if config.dimensions.is_empty() || config.dimensions.contains(&0) {
            return Err(anyhow!(
                "test input `{}` has an unusable shape {:?}",
                config.name,
                config.dimensions
            ));
        }

        match config.data_type {
            TestDataType::I32 | TestDataType::I64 => {
                // Create token IDs for language models
                let count = config.dimensions.iter().product::<usize>();
                let input_ids: Vec<f32> = (0..count).map(|i| ((i % 1000) + 1) as f32).collect();
                Ok(Tensor::from_vec(input_ids, &config.dimensions)?)
            },
            TestDataType::F32 | TestDataType::F16 => {
                // Create floating point input
                Ok(Tensor::randn(&config.dimensions)?)
            },
        }
    }

    /// Get the model name being profiled
    pub fn get_model_name(&self) -> &str {
        "Unknown"
    }
}

/// Current process resident-set size in MB, or `None` when the platform cannot
/// report it.
fn current_resident_mb() -> Option<f64> {
    crate::memory_profiling::MemoryProfiler::get_process_memory_info()
        .ok()
        .map(|info| info.resident_mb)
}

impl Default for PerformanceProfiler {
    fn default() -> Self {
        Self::new()
    }
}
