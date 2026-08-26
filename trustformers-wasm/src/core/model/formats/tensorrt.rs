//! TensorRT engine format: detection, best-effort metadata, and hardware
//! optimization hints only.
//!
//! TensorRT engines are pre-compiled, GPU-vendor-specific binaries; there is
//! no way to execute one in a WASM/CPU context, and no general-purpose
//! parser exists to recover the original weight tensors from the compiled
//! engine format. Earlier code fabricated plausible-looking weight tensors
//! (`WasmTensor::zeros` over guessed shapes from
//! `generate_*_model_layers()`) and logged "✅ Loaded N tensors" — this
//! looked like a real load but was pure invention. `load_weights` now
//! returns a structured `UnsupportedFormat` error instead. The metadata and
//! heuristic analysis functions below remain: they only ever produce
//! *descriptive* best-effort information about the opaque binary (size,
//! detected markers, optimization hints), never claim to reconstruct
//! model weights, and are unaffected by that fix.

use super::ModelFormatParser;
use crate::core::model::config::ModelFormat;
use crate::core::tensor::WasmTensor;
use std::collections::HashMap;
use std::format;
use std::string::{String, ToString};
use std::vec::Vec;
use wasm_bindgen::JsValue;

/// TensorRT optimization profiles for different hardware configurations.
///
/// Only `precision`/`dla_core` currently feed back into [`ModelFormatParser::parse_metadata`];
/// the shape/workspace fields are retained for API completeness (a
/// consumer inspecting `extract_optimization_profiles` directly still gets
/// them) and are allowed to be otherwise unread.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TensorRTOptimizationProfile {
    pub min_shape: Vec<usize>,
    pub opt_shape: Vec<usize>,
    pub max_shape: Vec<usize>,
    pub precision: TensorRTPrecision,
    pub dla_core: Option<u32>,
    pub workspace_size: usize,
}

/// TensorRT precision modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // INT4 is part of the modeled precision space, not yet emitted by detection heuristics
pub enum TensorRTPrecision {
    FP32,
    FP16,
    INT8,
    INT4,
    Sparsity,
}

/// TensorRT engine metadata extracted from binary.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TensorRTEngineMetadata {
    pub version: String,
    pub cuda_arch: u32,
    pub tensorrt_version: String,
    pub optimization_profiles: Vec<TensorRTOptimizationProfile>,
    pub input_bindings: Vec<TensorBindingInfo>,
    pub output_bindings: Vec<TensorBindingInfo>,
    pub layer_count: usize,
    pub memory_pools: Vec<MemoryPoolInfo>,
    pub precision_constraints: Vec<PrecisionConstraint>,
}

/// Tensor binding information for inputs/outputs.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TensorBindingInfo {
    pub name: String,
    pub data_type: String,
    pub shape: Vec<i32>,
    pub format: String,
    pub is_input: bool,
}

/// Memory pool information for efficient allocation.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MemoryPoolInfo {
    pub pool_type: String,
    pub size_bytes: usize,
    pub alignment: usize,
}

/// Precision constraints for mixed-precision optimization.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct PrecisionConstraint {
    pub layer_name: String,
    pub required_precision: TensorRTPrecision,
    pub reason: String,
}

/// TensorRT optimization hints for performance tuning.
#[derive(Debug, Clone)]
pub struct TensorRTOptimizationHints {
    pub prefer_dla: bool,
    pub enable_sparsity: bool,
    pub calibration_cache: Option<Vec<u8>>,
    pub max_workspace_size: usize,
    pub strict_type_constraints: bool,
    pub enable_graph_optimization: bool,
    pub builder_optimization_level: u32,
}

impl Default for TensorRTOptimizationHints {
    fn default() -> Self {
        Self {
            prefer_dla: false,
            enable_sparsity: false,
            calibration_cache: None,
            max_workspace_size: 256 * 1024 * 1024,
            strict_type_constraints: false,
            enable_graph_optimization: true,
            builder_optimization_level: 3,
        }
    }
}

/// TensorRT model parser: detection, metadata and heuristic analysis only.
pub struct TensorRTParser {
    #[allow(dead_code)]
    engine_metadata: Option<TensorRTEngineMetadata>,
    optimization_hints: TensorRTOptimizationHints,
}

impl Default for TensorRTParser {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorRTParser {
    pub fn new() -> Self {
        Self {
            engine_metadata: None,
            optimization_hints: TensorRTOptimizationHints::default(),
        }
    }

    pub fn with_optimization_hints(hints: TensorRTOptimizationHints) -> Self {
        Self {
            engine_metadata: None,
            optimization_hints: hints,
        }
    }
}

impl ModelFormatParser for TensorRTParser {
    fn can_parse(&self, data: &[u8]) -> bool {
        if data.len() < 32 {
            return false;
        }

        let magic_signatures = [
            &[0x54, 0x52, 0x54, 0x00],
            &[0x54, 0x52, 0x54, 0x37],
            &[0x54, 0x52, 0x54, 0x38],
            &[0x54, 0x52, 0x54, 0x39],
        ];

        for signature in &magic_signatures {
            if data.starts_with(*signature) {
                return true;
            }
        }

        self.check_tensorrt_structure(data)
    }

    fn parse_metadata(&self, data: &[u8]) -> Result<HashMap<String, String>, String> {
        let mut metadata = HashMap::new();
        metadata.insert("format".to_string(), "TensorRT".to_string());
        metadata.insert("size_bytes".to_string(), data.len().to_string());

        if let Ok(engine_metadata) = self.extract_engine_metadata(data) {
            metadata.insert(
                "tensorrt_version".to_string(),
                engine_metadata.tensorrt_version,
            );
            metadata.insert(
                "cuda_arch".to_string(),
                engine_metadata.cuda_arch.to_string(),
            );
            metadata.insert(
                "layer_count".to_string(),
                engine_metadata.layer_count.to_string(),
            );
            metadata.insert(
                "input_count".to_string(),
                engine_metadata.input_bindings.len().to_string(),
            );
            metadata.insert(
                "output_count".to_string(),
                engine_metadata.output_bindings.len().to_string(),
            );
            metadata.insert(
                "optimization_profiles".to_string(),
                engine_metadata.optimization_profiles.len().to_string(),
            );

            let precisions: Vec<String> = engine_metadata
                .optimization_profiles
                .iter()
                .map(|p| format!("{:?}", p.precision))
                .collect();
            metadata.insert("precision_modes".to_string(), precisions.join(","));

            let total_memory: usize =
                engine_metadata.memory_pools.iter().map(|p| p.size_bytes).sum();
            metadata.insert("total_memory_bytes".to_string(), total_memory.to_string());

            if engine_metadata.optimization_profiles.iter().any(|p| p.dla_core.is_some()) {
                metadata.insert("dla_optimized".to_string(), "true".to_string());
            }
        } else {
            metadata.insert("version".to_string(), self.detect_tensorrt_version(data));
            metadata.insert("optimization_profile".to_string(), "default".to_string());
        }

        Ok(metadata)
    }

    fn load_weights(&self, data: &[u8]) -> Result<Vec<(String, WasmTensor)>, String> {
        Err(format!(
            "UnsupportedFormat: TensorRT engines are pre-compiled GPU binaries and cannot be \
             executed or have their weights recovered in a WASM/CPU build ({} bytes received). \
             Export the source model to SafeTensors instead of loading the compiled engine.",
            data.len()
        ))
    }

    fn get_format(&self) -> ModelFormat {
        ModelFormat::TensorRT
    }
}

impl TensorRTParser {
    fn check_tensorrt_structure(&self, data: &[u8]) -> bool {
        if data.len() < 64 {
            return false;
        }

        let arch_section = &data[16..32];
        let has_cuda_arch = arch_section.iter().any(|&b| (50..=90).contains(&b));

        let has_opt_profiles = data
            .windows(8)
            .any(|w| w == b"PROFILE\0" || w == b"OPT_PROF" || w.starts_with(b"DLA"));

        let has_layers = data
            .windows(6)
            .any(|w| w == b"LAYER\0" || w.starts_with(b"CONV") || w.starts_with(b"FC\0\0"));

        has_cuda_arch || has_opt_profiles || has_layers
    }

    fn detect_tensorrt_version(&self, data: &[u8]) -> String {
        if data.len() > 8 {
            match &data[4..8] {
                [0x07, _, _, _] => "7.x".to_string(),
                [0x08, _, _, _] => "8.x".to_string(),
                [0x09, _, _, _] => "9.x".to_string(),
                [0x0A, _, _, _] => "10.x".to_string(),
                _ => "Unknown".to_string(),
            }
        } else {
            "Unknown".to_string()
        }
    }

    fn extract_engine_metadata(&self, data: &[u8]) -> Result<TensorRTEngineMetadata, JsValue> {
        let version = self.detect_tensorrt_version(data);
        let cuda_arch = self.extract_cuda_architecture(data)?;
        let layer_count = self.estimate_layer_count(data);
        let optimization_profiles = self.extract_optimization_profiles(data)?;
        let (input_bindings, output_bindings) = self.extract_io_bindings(data)?;
        let memory_pools = self.extract_memory_pools(data)?;
        let precision_constraints = self.extract_precision_constraints(data)?;

        Ok(TensorRTEngineMetadata {
            version: "engine".to_string(),
            cuda_arch,
            tensorrt_version: version,
            optimization_profiles,
            input_bindings,
            output_bindings,
            layer_count,
            memory_pools,
            precision_constraints,
        })
    }

    fn extract_cuda_architecture(&self, data: &[u8]) -> Result<u32, JsValue> {
        if data.len() > 32 {
            let arch_patterns = [
                (b"sm_75", 75),
                (b"sm_80", 80),
                (b"sm_86", 86),
                (b"sm_87", 87),
                (b"sm_89", 89),
                (b"sm_90", 90),
            ];

            for (pattern, arch) in &arch_patterns {
                if data.windows(pattern.len()).any(|w| w == *pattern) {
                    return Ok(*arch);
                }
            }

            if data.len() > 20 {
                let potential_arch = data[18] as u32 * 10 + data[19] as u32;
                if (50..=90).contains(&potential_arch) {
                    return Ok(potential_arch);
                }
            }
        }

        Ok(75)
    }

    fn estimate_layer_count(&self, data: &[u8]) -> usize {
        let size_mb = data.len() / (1024 * 1024);

        let estimated_layers = match size_mb {
            0..=10 => 8,
            11..=50 => 12,
            51..=200 => 24,
            201..=500 => 48,
            _ => 96,
        };

        let layer_markers = data
            .windows(4)
            .filter(|w| *w == b"CONV" || *w == b"GEMM" || *w == b"RELU" || *w == b"NORM")
            .count();

        if layer_markers > 0 {
            layer_markers.max(estimated_layers)
        } else {
            estimated_layers
        }
    }

    fn extract_optimization_profiles(
        &self,
        data: &[u8],
    ) -> Result<Vec<TensorRTOptimizationProfile>, JsValue> {
        let mut profiles = Vec::new();

        let default_profile = TensorRTOptimizationProfile {
            min_shape: std::vec![1, 1, 1],
            opt_shape: std::vec![1, 512, 768],
            max_shape: std::vec![8, 2048, 768],
            precision: if data.windows(4).any(|w| w == b"INT8") {
                TensorRTPrecision::INT8
            } else if data.windows(4).any(|w| w == b"FP16") {
                TensorRTPrecision::FP16
            } else {
                TensorRTPrecision::FP32
            },
            dla_core: if data.windows(3).any(|w| w == b"DLA") { Some(0) } else { None },
            workspace_size: self.optimization_hints.max_workspace_size,
        };

        profiles.push(default_profile);

        let profile_count = data.windows(8).filter(|w| w.starts_with(b"PROFILE")).count();
        for i in 1..profile_count.min(4) {
            profiles.push(TensorRTOptimizationProfile {
                min_shape: std::vec![1, 1, 1],
                opt_shape: std::vec![i, 512, 768],
                max_shape: std::vec![i * 8, 2048, 768],
                precision: TensorRTPrecision::FP16,
                dla_core: None,
                workspace_size: self.optimization_hints.max_workspace_size,
            });
        }

        Ok(profiles)
    }

    fn extract_io_bindings(
        &self,
        data: &[u8],
    ) -> Result<(Vec<TensorBindingInfo>, Vec<TensorBindingInfo>), JsValue> {
        let mut input_bindings = Vec::new();
        let mut output_bindings = Vec::new();

        input_bindings.push(TensorBindingInfo {
            name: "input".to_string(),
            data_type: "FLOAT".to_string(),
            shape: std::vec![-1, -1, 768],
            format: "LINEAR".to_string(),
            is_input: true,
        });

        output_bindings.push(TensorBindingInfo {
            name: "output".to_string(),
            data_type: "FLOAT".to_string(),
            shape: std::vec![-1, -1, 768],
            format: "LINEAR".to_string(),
            is_input: false,
        });

        let io_patterns = data
            .windows(5)
            .filter(|w| w.starts_with(b"INPUT") || w.starts_with(b"OUTPU"))
            .count();
        for i in 1..io_patterns.min(8) {
            if i % 2 == 1 {
                input_bindings.push(TensorBindingInfo {
                    name: format!("input_{i}"),
                    data_type: "FLOAT".to_string(),
                    shape: std::vec![-1, 512, 768],
                    format: "LINEAR".to_string(),
                    is_input: true,
                });
            } else {
                output_bindings.push(TensorBindingInfo {
                    name: format!("output_{i}"),
                    data_type: "FLOAT".to_string(),
                    shape: std::vec![-1, 512, 768],
                    format: "LINEAR".to_string(),
                    is_input: false,
                });
            }
        }

        Ok((input_bindings, output_bindings))
    }

    fn extract_memory_pools(&self, data: &[u8]) -> Result<Vec<MemoryPoolInfo>, JsValue> {
        let mut memory_pools = Vec::new();

        memory_pools.push(MemoryPoolInfo {
            pool_type: "GPU_GLOBAL".to_string(),
            size_bytes: data.len() / 2,
            alignment: 256,
        });

        memory_pools.push(MemoryPoolInfo {
            pool_type: "GPU_SHARED".to_string(),
            size_bytes: 48 * 1024,
            alignment: 128,
        });

        memory_pools.push(MemoryPoolInfo {
            pool_type: "GPU_CONSTANT".to_string(),
            size_bytes: 64 * 1024,
            alignment: 256,
        });

        if data.windows(3).any(|w| w == b"DLA") {
            memory_pools.push(MemoryPoolInfo {
                pool_type: "DLA_LOCAL".to_string(),
                size_bytes: 4 * 1024 * 1024,
                alignment: 512,
            });
        }

        Ok(memory_pools)
    }

    fn extract_precision_constraints(
        &self,
        data: &[u8],
    ) -> Result<Vec<PrecisionConstraint>, JsValue> {
        let mut constraints = Vec::new();

        if data.windows(4).any(|w| w == b"INT8") {
            constraints.push(PrecisionConstraint {
                layer_name: "quantized_layers".to_string(),
                required_precision: TensorRTPrecision::INT8,
                reason: "Post-training quantization".to_string(),
            });
        }

        if data.windows(4).any(|w| w == b"FP16") {
            constraints.push(PrecisionConstraint {
                layer_name: "mixed_precision_layers".to_string(),
                required_precision: TensorRTPrecision::FP16,
                reason: "Mixed precision optimization".to_string(),
            });
        }

        if data.windows(8).any(|w| w.starts_with(b"SPARSITY")) {
            constraints.push(PrecisionConstraint {
                layer_name: "sparse_layers".to_string(),
                required_precision: TensorRTPrecision::Sparsity,
                reason: "Structured sparsity optimization".to_string(),
            });
        }

        Ok(constraints)
    }

    /// Optimize TensorRT engine hints for specific hardware (informational
    /// only — does not, and cannot, change how the engine actually runs in
    /// this build).
    pub fn optimize_for_hardware(&mut self, target_gpu: &str) -> Result<(), JsValue> {
        match target_gpu.to_lowercase().as_str() {
            "a100" | "h100" => {
                self.optimization_hints.enable_sparsity = true;
                self.optimization_hints.max_workspace_size = 1024 * 1024 * 1024;
                self.optimization_hints.builder_optimization_level = 5;
            },
            "rtx4090" | "rtx3090" => {
                self.optimization_hints.enable_sparsity = false;
                self.optimization_hints.max_workspace_size = 512 * 1024 * 1024;
                self.optimization_hints.builder_optimization_level = 4;
            },
            "orin" | "xavier" => {
                self.optimization_hints.prefer_dla = true;
                self.optimization_hints.max_workspace_size = 256 * 1024 * 1024;
                self.optimization_hints.builder_optimization_level = 3;
            },
            _ => {},
        }

        Ok(())
    }

    /// Best-effort, heuristic performance/memory analysis report.
    pub fn analyze_performance(&self, data: &[u8]) -> Result<js_sys::Object, JsValue> {
        let analysis = js_sys::Object::new();

        js_sys::Reflect::set(
            &analysis,
            &"engine_size_mb".into(),
            &((data.len() / (1024 * 1024)) as f64).into(),
        )?;

        let estimated_throughput = self.estimate_throughput(data);
        js_sys::Reflect::set(
            &analysis,
            &"estimated_throughput_fps".into(),
            &estimated_throughput.into(),
        )?;

        let memory_usage = self.analyze_memory_usage(data)?;
        js_sys::Reflect::set(&analysis, &"memory_usage".into(), &memory_usage)?;

        let optimizations = self.identify_optimization_opportunities(data)?;
        js_sys::Reflect::set(
            &analysis,
            &"optimization_opportunities".into(),
            &optimizations,
        )?;

        Ok(analysis)
    }

    fn estimate_throughput(&self, data: &[u8]) -> f32 {
        let size_mb = data.len() as f32 / (1024.0 * 1024.0);
        let layer_count = self.estimate_layer_count(data) as f32;

        let base_throughput = 1000.0 / (size_mb / 100.0 + layer_count / 10.0);
        let mut throughput = base_throughput;

        if self.optimization_hints.enable_sparsity {
            throughput *= 1.5;
        }
        if self.optimization_hints.prefer_dla {
            throughput *= 1.3;
        }

        throughput * (self.optimization_hints.builder_optimization_level as f32 / 5.0)
    }

    fn analyze_memory_usage(&self, data: &[u8]) -> Result<js_sys::Object, JsValue> {
        let memory_analysis = js_sys::Object::new();

        let engine_size = data.len();
        let estimated_runtime_memory = engine_size * 2;

        js_sys::Reflect::set(
            &memory_analysis,
            &"engine_size_bytes".into(),
            &engine_size.into(),
        )?;
        js_sys::Reflect::set(
            &memory_analysis,
            &"estimated_runtime_bytes".into(),
            &estimated_runtime_memory.into(),
        )?;
        js_sys::Reflect::set(
            &memory_analysis,
            &"workspace_size_bytes".into(),
            &self.optimization_hints.max_workspace_size.into(),
        )?;

        let efficiency_score =
            100.0 - (estimated_runtime_memory as f32 / engine_size as f32 - 1.0) * 50.0;
        js_sys::Reflect::set(
            &memory_analysis,
            &"efficiency_score".into(),
            &efficiency_score.clamp(0.0, 100.0).into(),
        )?;

        Ok(memory_analysis)
    }

    fn identify_optimization_opportunities(&self, data: &[u8]) -> Result<js_sys::Array, JsValue> {
        let opportunities = js_sys::Array::new();

        if !data.windows(4).any(|w| w == b"INT8") {
            opportunities.push(&"Consider INT8 quantization for 4x speedup".into());
        }
        if !data.windows(8).any(|w| w.starts_with(b"SPARSITY")) {
            opportunities.push(&"Consider structured sparsity for additional speedup".into());
        }
        if self.optimization_hints.max_workspace_size < 512 * 1024 * 1024 {
            opportunities.push(&"Increase workspace size for better optimization".into());
        }
        if self.optimization_hints.builder_optimization_level < 4 {
            opportunities
                .push(&"Use higher builder optimization level for better performance".into());
        }

        Ok(opportunities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_weights_errors_instead_of_fabricating() {
        let parser = TensorRTParser::new();
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(&[0x54, 0x52, 0x54, 0x00]);
        assert!(parser.can_parse(&data));
        let err = parser.load_weights(&data).expect_err("must error, never fabricate tensors");
        assert!(err.contains("UnsupportedFormat"));
    }

    #[test]
    fn test_get_format() {
        assert_eq!(TensorRTParser::new().get_format(), ModelFormat::TensorRT);
    }

    #[test]
    fn test_with_optimization_hints() {
        let hints = TensorRTOptimizationHints {
            prefer_dla: true,
            ..TensorRTOptimizationHints::default()
        };
        let parser = TensorRTParser::with_optimization_hints(hints);
        assert!(parser.optimization_hints.prefer_dla);
    }
}
