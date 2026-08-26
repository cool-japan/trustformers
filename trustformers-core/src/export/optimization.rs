//! Optimization passes applied to a model before export.
//!
//! # What can and cannot be optimized through the `Model` trait
//!
//! Constant folding, dead-code elimination, operator fusion and layout selection
//! all rewrite a *computation graph*. The [`Model`] trait exposes parameters
//! (through [`Model::named_tensors`]) but no graph, so those four passes cannot be
//! carried out here. They report themselves as not applicable and, if invoked
//! directly, return a structured
//! [`ErrorKind::UnsupportedOperation`](crate::errors::ErrorKind::UnsupportedOperation)
//! naming the limitation. Real graph optimization on an exported ONNX model lives
//! in [`ONNXOptimizer`](super::onnx_runtime::ONNXOptimizer).
//!
//! [`WeightCompressionPass`] *is* implemented: it works on parameters, which the
//! trait does expose, and reports byte counts it actually measured.
//!
//! An earlier revision of this module returned hard-coded statistics from every
//! pass — "removed 15 constant operations", "1.25x speedup on GPU", "10 MB
//! saved" — regardless of the model. Those numbers were never measured and are
//! gone.

use crate::errors::{unsupported_operation, Result};
use crate::tensor::Tensor;
use crate::traits::Model;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Explanation attached to every graph-level pass that cannot run through `Model`.
pub const GRAPH_PASS_UNSUPPORTED_REASON: &str =
    "this pass rewrites the model's computation graph, which the `Model` trait does \
     not expose (`named_tensors` yields parameters only). Export to ONNX and use \
     `ONNXOptimizer::optimize_model_with_stats`, which performs identity \
     elimination, constant folding and dead-initializer removal on a real graph.";

/// Statistics with nothing to report, used when a pass is switched off.
fn no_change() -> OptimizationStats {
    OptimizationStats {
        operations_removed: 0,
        operations_modified: 0,
        size_reduction_bytes: 0,
        speedup_factor: 1.0,
        precision_preserved: true,
    }
}

/// Configuration for export optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Enable constant folding optimization
    pub constant_folding: bool,
    /// Enable dead code elimination
    pub dead_code_elimination: bool,
    /// Enable operator fusion
    pub operator_fusion: bool,
    /// Enable layout optimization
    pub layout_optimization: bool,
    /// Enable weight compression
    pub weight_compression: bool,
    /// Target hardware for optimization
    pub target_hardware: TargetHardware,
    /// Optimization level (0-3)
    pub optimization_level: u8,
    /// Whether to preserve numerical precision
    pub preserve_precision: bool,
}

/// Target hardware for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TargetHardware {
    CPU,
    GPU,
    Mobile,
    Edge,
    WebAssembly,
}

/// Optimization pass trait
pub trait OptimizationPass: Send + Sync {
    /// Name of the optimization pass
    fn name(&self) -> &str;

    /// Description of what this pass does
    fn description(&self) -> &str;

    /// Apply the optimization pass to a model
    fn apply<M: Model>(
        &self,
        model: &mut M,
        config: &OptimizationConfig,
    ) -> Result<OptimizationStats>;

    /// Check if this pass is applicable for the given configuration
    fn is_applicable(&self, config: &OptimizationConfig) -> bool;

    /// Get the expected impact of this optimization
    fn expected_impact(&self) -> OptimizationImpact;
}

/// Concrete enum holding all optimization pass types for dyn compatibility
#[derive(Clone)]
pub enum ConcreteOptimizationPass {
    ConstantFolding(ConstantFoldingPass),
    DeadCodeElimination(DeadCodeEliminationPass),
    OperatorFusion(OperatorFusionPass),
    LayoutOptimization(LayoutOptimizationPass),
    WeightCompression(WeightCompressionPass),
}

impl OptimizationPass for ConcreteOptimizationPass {
    fn name(&self) -> &str {
        match self {
            ConcreteOptimizationPass::ConstantFolding(pass) => pass.name(),
            ConcreteOptimizationPass::DeadCodeElimination(pass) => pass.name(),
            ConcreteOptimizationPass::OperatorFusion(pass) => pass.name(),
            ConcreteOptimizationPass::LayoutOptimization(pass) => pass.name(),
            ConcreteOptimizationPass::WeightCompression(pass) => pass.name(),
        }
    }

    fn description(&self) -> &str {
        match self {
            ConcreteOptimizationPass::ConstantFolding(pass) => pass.description(),
            ConcreteOptimizationPass::DeadCodeElimination(pass) => pass.description(),
            ConcreteOptimizationPass::OperatorFusion(pass) => pass.description(),
            ConcreteOptimizationPass::LayoutOptimization(pass) => pass.description(),
            ConcreteOptimizationPass::WeightCompression(pass) => pass.description(),
        }
    }

    fn apply<M: Model>(
        &self,
        model: &mut M,
        config: &OptimizationConfig,
    ) -> Result<OptimizationStats> {
        match self {
            ConcreteOptimizationPass::ConstantFolding(pass) => pass.apply(model, config),
            ConcreteOptimizationPass::DeadCodeElimination(pass) => pass.apply(model, config),
            ConcreteOptimizationPass::OperatorFusion(pass) => pass.apply(model, config),
            ConcreteOptimizationPass::LayoutOptimization(pass) => pass.apply(model, config),
            ConcreteOptimizationPass::WeightCompression(pass) => pass.apply(model, config),
        }
    }

    fn is_applicable(&self, config: &OptimizationConfig) -> bool {
        match self {
            ConcreteOptimizationPass::ConstantFolding(pass) => pass.is_applicable(config),
            ConcreteOptimizationPass::DeadCodeElimination(pass) => pass.is_applicable(config),
            ConcreteOptimizationPass::OperatorFusion(pass) => pass.is_applicable(config),
            ConcreteOptimizationPass::LayoutOptimization(pass) => pass.is_applicable(config),
            ConcreteOptimizationPass::WeightCompression(pass) => pass.is_applicable(config),
        }
    }

    fn expected_impact(&self) -> OptimizationImpact {
        match self {
            ConcreteOptimizationPass::ConstantFolding(pass) => pass.expected_impact(),
            ConcreteOptimizationPass::DeadCodeElimination(pass) => pass.expected_impact(),
            ConcreteOptimizationPass::OperatorFusion(pass) => pass.expected_impact(),
            ConcreteOptimizationPass::LayoutOptimization(pass) => pass.expected_impact(),
            ConcreteOptimizationPass::WeightCompression(pass) => pass.expected_impact(),
        }
    }
}

/// Statistics about applied optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStats {
    /// Number of operations removed
    pub operations_removed: usize,
    /// Number of operations modified
    pub operations_modified: usize,
    /// Size reduction in bytes, measured by the pass that produced these stats.
    pub size_reduction_bytes: u64,
    /// Measured speedup factor, or `1.0` when the pass did not benchmark anything.
    ///
    /// No pass in this crate predicts a speedup: predicting one without running the
    /// model would be a guess dressed up as a measurement.
    pub speedup_factor: f64,
    /// Whether precision was preserved
    pub precision_preserved: bool,
}

/// Expected impact of an optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationImpact {
    Low,
    Medium,
    High,
    Critical,
}

/// Optimization pipeline manager
pub struct OptimizationPipeline {
    passes: Vec<ConcreteOptimizationPass>,
    config: OptimizationConfig,
}

impl OptimizationPipeline {
    /// Create a new optimization pipeline
    pub fn new(config: OptimizationConfig) -> Self {
        let mut pipeline = Self {
            passes: Vec::new(),
            config,
        };

        pipeline.register_default_passes();
        pipeline
    }

    /// Register default optimization passes
    fn register_default_passes(&mut self) {
        self.add_pass(ConcreteOptimizationPass::ConstantFolding(
            ConstantFoldingPass::new(),
        ));
        self.add_pass(ConcreteOptimizationPass::DeadCodeElimination(
            DeadCodeEliminationPass::new(),
        ));
        self.add_pass(ConcreteOptimizationPass::OperatorFusion(
            OperatorFusionPass::new(),
        ));
        self.add_pass(ConcreteOptimizationPass::LayoutOptimization(
            LayoutOptimizationPass::new(),
        ));
        self.add_pass(ConcreteOptimizationPass::WeightCompression(
            WeightCompressionPass::new(),
        ));
    }

    /// Add an optimization pass to the pipeline
    pub fn add_pass(&mut self, pass: ConcreteOptimizationPass) {
        self.passes.push(pass);
    }

    /// Apply all applicable optimization passes
    pub fn apply_optimizations<M: Model>(&self, model: &mut M) -> Result<PipelineStats> {
        let mut total_stats = PipelineStats::new();
        let mut applied_passes = Vec::new();

        for pass in &self.passes {
            if pass.is_applicable(&self.config) {
                tracing::info!("Applying optimization pass: {}", pass.name());

                let stats = pass.apply(model, &self.config)?;
                total_stats.add_pass_stats(pass.name().to_string(), stats);
                applied_passes.push(pass.name().to_string());
            }
        }

        total_stats.applied_passes = applied_passes;
        Ok(total_stats)
    }

    /// Get a list of applicable passes for the current configuration
    pub fn get_applicable_passes(&self) -> Vec<&str> {
        self.passes
            .iter()
            .filter(|pass| pass.is_applicable(&self.config))
            .map(|pass| pass.name())
            .collect()
    }
}

/// Statistics for the entire optimization pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStats {
    /// Applied optimization passes
    pub applied_passes: Vec<String>,
    /// Total operations removed
    pub total_operations_removed: usize,
    /// Total operations modified
    pub total_operations_modified: usize,
    /// Total size reduction in bytes
    pub total_size_reduction_bytes: u64,
    /// Overall speedup factor
    pub overall_speedup_factor: f64,
    /// Stats per pass
    pub pass_stats: HashMap<String, OptimizationStats>,
}

impl PipelineStats {
    fn new() -> Self {
        Self {
            applied_passes: Vec::new(),
            total_operations_removed: 0,
            total_operations_modified: 0,
            total_size_reduction_bytes: 0,
            overall_speedup_factor: 1.0,
            pass_stats: HashMap::new(),
        }
    }

    fn add_pass_stats(&mut self, pass_name: String, stats: OptimizationStats) {
        self.total_operations_removed += stats.operations_removed;
        self.total_operations_modified += stats.operations_modified;
        self.total_size_reduction_bytes += stats.size_reduction_bytes;
        self.overall_speedup_factor *= stats.speedup_factor;
        self.pass_stats.insert(pass_name, stats);
    }
}

// Specific optimization pass implementations

/// Constant folding optimization pass
#[derive(Clone)]
pub struct ConstantFoldingPass;

impl ConstantFoldingPass {
    fn new() -> Self {
        Self
    }
}

impl OptimizationPass for ConstantFoldingPass {
    fn name(&self) -> &str {
        "constant_folding"
    }

    fn description(&self) -> &str {
        "Folds constant expressions at compile time to reduce runtime computation"
    }

    fn apply<M: Model>(
        &self,
        model: &mut M,
        config: &OptimizationConfig,
    ) -> Result<OptimizationStats> {
        if !config.constant_folding {
            return Ok(no_change());
        }
        let _ = model;
        Err(unsupported_operation(
            "constant folding on a `Model`",
            GRAPH_PASS_UNSUPPORTED_REASON,
        ))
    }

    /// Never applicable: see [`GRAPH_PASS_UNSUPPORTED_REASON`].
    fn is_applicable(&self, _config: &OptimizationConfig) -> bool {
        false
    }

    fn expected_impact(&self) -> OptimizationImpact {
        OptimizationImpact::Low
    }
}

/// Dead code elimination pass
#[derive(Clone)]
pub struct DeadCodeEliminationPass;

impl DeadCodeEliminationPass {
    fn new() -> Self {
        Self
    }
}

impl OptimizationPass for DeadCodeEliminationPass {
    fn name(&self) -> &str {
        "dead_code_elimination"
    }

    fn description(&self) -> &str {
        "Removes unused operations and parameters from the model"
    }

    fn apply<M: Model>(
        &self,
        model: &mut M,
        config: &OptimizationConfig,
    ) -> Result<OptimizationStats> {
        if !config.dead_code_elimination {
            return Ok(no_change());
        }
        let _ = model;
        Err(unsupported_operation(
            "dead code elimination on a `Model`",
            GRAPH_PASS_UNSUPPORTED_REASON,
        ))
    }

    /// Never applicable: see [`GRAPH_PASS_UNSUPPORTED_REASON`].
    fn is_applicable(&self, _config: &OptimizationConfig) -> bool {
        false
    }

    fn expected_impact(&self) -> OptimizationImpact {
        OptimizationImpact::Medium
    }
}

/// Operator fusion pass
#[derive(Clone)]
pub struct OperatorFusionPass;

impl OperatorFusionPass {
    fn new() -> Self {
        Self
    }
}

impl OptimizationPass for OperatorFusionPass {
    fn name(&self) -> &str {
        "operator_fusion"
    }

    fn description(&self) -> &str {
        "Fuses compatible operations to reduce memory bandwidth and improve cache efficiency"
    }

    fn apply<M: Model>(
        &self,
        model: &mut M,
        config: &OptimizationConfig,
    ) -> Result<OptimizationStats> {
        if !config.operator_fusion {
            return Ok(no_change());
        }
        let _ = model;
        Err(unsupported_operation(
            "operator fusion on a `Model`",
            GRAPH_PASS_UNSUPPORTED_REASON,
        ))
    }

    /// Never applicable: see [`GRAPH_PASS_UNSUPPORTED_REASON`].
    fn is_applicable(&self, _config: &OptimizationConfig) -> bool {
        false
    }

    fn expected_impact(&self) -> OptimizationImpact {
        OptimizationImpact::High
    }
}

/// Layout optimization pass
#[derive(Clone)]
pub struct LayoutOptimizationPass;

impl LayoutOptimizationPass {
    fn new() -> Self {
        Self
    }
}

impl OptimizationPass for LayoutOptimizationPass {
    fn name(&self) -> &str {
        "layout_optimization"
    }

    fn description(&self) -> &str {
        "Optimizes tensor layouts for better memory access patterns"
    }

    fn apply<M: Model>(
        &self,
        model: &mut M,
        config: &OptimizationConfig,
    ) -> Result<OptimizationStats> {
        if !config.layout_optimization {
            return Ok(no_change());
        }
        let _ = model;
        Err(unsupported_operation(
            "tensor layout optimization on a `Model`",
            GRAPH_PASS_UNSUPPORTED_REASON,
        ))
    }

    /// Never applicable: see [`GRAPH_PASS_UNSUPPORTED_REASON`].
    fn is_applicable(&self, _config: &OptimizationConfig) -> bool {
        false
    }

    fn expected_impact(&self) -> OptimizationImpact {
        OptimizationImpact::High
    }
}

/// Weight compression pass
#[derive(Clone)]
pub struct WeightCompressionPass;

impl WeightCompressionPass {
    fn new() -> Self {
        Self
    }
}

impl OptimizationPass for WeightCompressionPass {
    fn name(&self) -> &str {
        "weight_compression"
    }

    fn description(&self) -> &str {
        "Quantizes float parameters to per-tensor int8 and writes the recovered values back"
    }

    /// Apply real per-tensor int8 quantization to every float parameter.
    ///
    /// For each parameter the pass computes `scale = max|w| / 127`, quantizes to
    /// int8 and writes the dequantized values back, so the model afterwards holds
    /// exactly the values an int8 export would produce. The reported byte figures
    /// are the difference between the f32 footprint and the int8-plus-scale
    /// footprint of the parameters it actually touched — measured, not predicted.
    ///
    /// # Errors
    ///
    /// Fails when the model exposes no parameters through
    /// [`Model::named_tensors_mut`], since there is then nothing to compress and a
    /// zero-change success would be misleading.
    fn apply<M: Model>(
        &self,
        model: &mut M,
        config: &OptimizationConfig,
    ) -> Result<OptimizationStats> {
        if !config.weight_compression {
            return Ok(no_change());
        }
        if config.preserve_precision {
            // int8 quantization is lossy by construction.
            return Ok(no_change());
        }

        let mut parameters = model.named_tensors_mut();
        if parameters.is_empty() {
            return Err(unsupported_operation(
                "weight compression",
                "the model exposes no parameters through `Model::named_tensors_mut`, so there \
                 is nothing to compress",
            ));
        }

        let mut tensors_modified = 0usize;
        let mut bytes_before = 0u64;
        let mut bytes_after = 0u64;

        for (_name, parameter) in parameters.iter_mut() {
            let Tensor::F32(array) = &**parameter else {
                // Only float parameters are quantizable; integer buffers are left alone.
                continue;
            };
            let shape = array.shape().to_vec();
            let values: Vec<f32> = array.iter().copied().collect();
            if values.is_empty() {
                continue;
            }

            let amax = values.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
            if amax == 0.0 || !amax.is_finite() {
                continue;
            }
            let scale = amax / 127.0;

            let compressed: Vec<f32> = values
                .iter()
                .map(|&v| ((v / scale).round().clamp(-127.0, 127.0)) * scale)
                .collect();

            **parameter = Tensor::from_vec(compressed, &shape)?;

            tensors_modified += 1;
            bytes_before += (values.len() * std::mem::size_of::<f32>()) as u64;
            // int8 payload plus one f32 scale per tensor.
            bytes_after += values.len() as u64 + std::mem::size_of::<f32>() as u64;
        }

        Ok(OptimizationStats {
            operations_removed: 0,
            operations_modified: tensors_modified,
            size_reduction_bytes: bytes_before.saturating_sub(bytes_after),
            // This pass does not benchmark, so it claims no speedup.
            speedup_factor: 1.0,
            precision_preserved: tensors_modified == 0,
        })
    }

    fn is_applicable(&self, config: &OptimizationConfig) -> bool {
        config.weight_compression && config.optimization_level > 0 && !config.preserve_precision
    }

    fn expected_impact(&self) -> OptimizationImpact {
        OptimizationImpact::Critical
    }
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            constant_folding: true,
            dead_code_elimination: true,
            operator_fusion: true,
            layout_optimization: true,
            weight_compression: false, // Conservative default
            target_hardware: TargetHardware::CPU,
            optimization_level: 2,
            preserve_precision: true,
        }
    }
}

/// Preset optimization configurations for common scenarios
impl OptimizationConfig {
    /// Configuration optimized for fast inference on CPU
    pub fn for_cpu_inference() -> Self {
        Self {
            constant_folding: true,
            dead_code_elimination: true,
            operator_fusion: true,
            layout_optimization: true,
            weight_compression: true,
            target_hardware: TargetHardware::CPU,
            optimization_level: 2,
            preserve_precision: true,
        }
    }

    /// Configuration optimized for GPU inference
    pub fn for_gpu_inference() -> Self {
        Self {
            constant_folding: true,
            dead_code_elimination: true,
            operator_fusion: true,
            layout_optimization: true,
            weight_compression: false, // GPU has more memory
            target_hardware: TargetHardware::GPU,
            optimization_level: 3,
            preserve_precision: true,
        }
    }

    /// Configuration optimized for mobile deployment
    pub fn for_mobile() -> Self {
        Self {
            constant_folding: true,
            dead_code_elimination: true,
            operator_fusion: true,
            layout_optimization: true,
            weight_compression: true,
            target_hardware: TargetHardware::Mobile,
            optimization_level: 3,
            preserve_precision: false, // Accept some precision loss for size
        }
    }

    /// Configuration for edge devices with limited resources
    pub fn for_edge() -> Self {
        Self {
            constant_folding: true,
            dead_code_elimination: true,
            operator_fusion: true,
            layout_optimization: false, // Simpler layouts for edge
            weight_compression: true,
            target_hardware: TargetHardware::Edge,
            optimization_level: 3,
            preserve_precision: false,
        }
    }

    /// Configuration for WebAssembly deployment
    pub fn for_webassembly() -> Self {
        Self {
            constant_folding: true,
            dead_code_elimination: true,
            operator_fusion: false, // Simpler for WASM
            layout_optimization: false,
            weight_compression: true,
            target_hardware: TargetHardware::WebAssembly,
            optimization_level: 2,
            preserve_precision: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::export::test_support::TestModel;

    #[test]
    fn test_optimization_config_presets() {
        let cpu_config = OptimizationConfig::for_cpu_inference();
        assert_eq!(cpu_config.target_hardware, TargetHardware::CPU);
        assert!(cpu_config.preserve_precision);

        let mobile_config = OptimizationConfig::for_mobile();
        assert_eq!(mobile_config.target_hardware, TargetHardware::Mobile);
        assert!(!mobile_config.preserve_precision); // Mobile accepts precision loss
    }

    /// Every pass is switched on, yet only the one that works on parameters can
    /// report itself applicable: the other four need a computation graph, which the
    /// `Model` trait does not expose.
    #[test]
    fn only_the_weight_pass_can_apply_through_the_model_trait() {
        let config = OptimizationConfig {
            constant_folding: true,
            dead_code_elimination: true,
            operator_fusion: true,
            layout_optimization: true,
            weight_compression: true,
            preserve_precision: false,
            ..OptimizationConfig::default()
        };
        let pipeline = OptimizationPipeline::new(config);
        assert_eq!(pipeline.get_applicable_passes(), vec!["weight_compression"]);
    }

    /// Regression test for the pipeline entry point. It used to aggregate hard-coded
    /// per-pass statistics — "15 operations removed", "1.25x speedup" — from all five
    /// passes regardless of the model. Now only the pass that can actually run
    /// contributes, and every number it reports was measured.
    #[test]
    fn apply_optimizations_aggregates_only_measured_statistics() {
        let config = OptimizationConfig {
            weight_compression: true,
            optimization_level: 3,
            preserve_precision: false,
            ..OptimizationConfig::default()
        };
        let pipeline = OptimizationPipeline::new(config);

        let mut model = TestModel::with_seed(1.0);
        let before: Vec<Vec<f32>> = model
            .named_tensors()
            .iter()
            .map(|(_, tensor)| tensor.to_vec_f32().expect("f32"))
            .collect();
        let element_count: u64 = before.iter().map(|values| values.len() as u64).sum();

        let stats = pipeline.apply_optimizations(&mut model).expect("pipeline");

        assert_eq!(stats.applied_passes, vec!["weight_compression"]);
        assert_eq!(stats.pass_stats.len(), 1);
        assert!(stats.pass_stats.contains_key("weight_compression"));

        // The four graph passes never ran, so nothing may be claimed on their behalf.
        assert_eq!(
            stats.total_operations_removed, 0,
            "no pass here removes graph operations"
        );
        assert_eq!(
            stats.overall_speedup_factor, 1.0,
            "no pass benchmarked anything, so no speedup may be claimed"
        );

        // The one pass that ran reports bytes it actually measured: 4 -> 1 byte per
        // element, minus one retained f32 scale per tensor.
        assert_eq!(stats.total_operations_modified, 3);
        assert_eq!(stats.total_size_reduction_bytes, element_count * 3 - 3 * 4);

        let after: Vec<Vec<f32>> = model
            .named_tensors()
            .iter()
            .map(|(_, tensor)| tensor.to_vec_f32().expect("f32"))
            .collect();
        assert_ne!(
            before, after,
            "the pipeline must really rewrite the weights"
        );
    }

    /// With nothing enabled the pipeline must report an honest zero rather than a
    /// plausible-looking summary.
    #[test]
    fn apply_optimizations_reports_nothing_when_no_pass_applies() {
        let pipeline = OptimizationPipeline::new(OptimizationConfig::default());
        let mut model = TestModel::with_seed(1.0);
        let before = model.named_tensors()[0].1.to_vec_f32().expect("f32");

        let stats = pipeline.apply_optimizations(&mut model).expect("pipeline");

        assert!(stats.applied_passes.is_empty());
        assert!(stats.pass_stats.is_empty());
        assert_eq!(stats.total_operations_removed, 0);
        assert_eq!(stats.total_operations_modified, 0);
        assert_eq!(stats.total_size_reduction_bytes, 0);
        assert_eq!(stats.overall_speedup_factor, 1.0);
        assert_eq!(
            model.named_tensors()[0].1.to_vec_f32().expect("f32"),
            before
        );
    }

    /// Regression test: the graph passes used to return invented statistics such as
    /// "removed 15 constant operations" for any model at all.
    #[test]
    fn graph_passes_refuse_rather_than_invent_statistics() {
        let config = OptimizationConfig::default();
        let mut model = TestModel::with_seed(1.0);

        for (name, result) in [
            (
                "constant_folding",
                ConstantFoldingPass::new().apply(&mut model, &config),
            ),
            (
                "dead_code_elimination",
                DeadCodeEliminationPass::new().apply(&mut model, &config),
            ),
            (
                "operator_fusion",
                OperatorFusionPass::new().apply(&mut model, &config),
            ),
            (
                "layout_optimization",
                LayoutOptimizationPass::new().apply(&mut model, &config),
            ),
        ] {
            let err = result.expect_err("{name} must not report invented statistics");
            assert!(
                err.to_string().contains("Unsupported operation"),
                "{name}: {err}"
            );
        }
    }

    #[test]
    fn a_disabled_graph_pass_reports_no_change_instead_of_failing() {
        let config = OptimizationConfig {
            constant_folding: false,
            ..OptimizationConfig::default()
        };
        let mut model = TestModel::with_seed(1.0);
        let stats = ConstantFoldingPass::new()
            .apply(&mut model, &config)
            .expect("a disabled pass simply does nothing");
        assert_eq!(stats.operations_removed, 0);
        assert_eq!(stats.size_reduction_bytes, 0);
        assert_eq!(stats.speedup_factor, 1.0);
    }

    /// The weight pass must really change the model's values and report measured bytes.
    #[test]
    fn weight_compression_quantizes_the_real_parameters() {
        let config = OptimizationConfig {
            weight_compression: true,
            optimization_level: 3,
            preserve_precision: false,
            ..OptimizationConfig::default()
        };

        let mut model = TestModel::with_seed(1.0);
        let before: Vec<Vec<f32>> = model
            .named_tensors()
            .iter()
            .map(|(_, tensor)| tensor.to_vec_f32().expect("f32"))
            .collect();

        let stats = WeightCompressionPass::new().apply(&mut model, &config).expect("compression");

        let after: Vec<Vec<f32>> = model
            .named_tensors()
            .iter()
            .map(|(_, tensor)| tensor.to_vec_f32().expect("f32"))
            .collect();

        assert_eq!(
            stats.operations_modified, 3,
            "all three parameters are float"
        );
        assert!(!stats.precision_preserved);
        assert_eq!(
            stats.speedup_factor, 1.0,
            "no benchmark was run, so no speedup is claimed"
        );

        // 4 bytes -> 1 byte per element, minus one f32 scale per tensor.
        let element_count: u64 = before.iter().map(|values| values.len() as u64).sum();
        assert_eq!(stats.size_reduction_bytes, element_count * 3 - 3 * 4);

        assert_ne!(before, after, "quantization must change the stored values");
        for (before_values, after_values) in before.iter().zip(after.iter()) {
            let amax = before_values.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
            let step = amax / 127.0;
            for (original, quantized) in before_values.iter().zip(after_values.iter()) {
                assert!(
                    (original - quantized).abs() <= step,
                    "{original} -> {quantized} exceeds one quantization step of {step}"
                );
            }
        }
    }

    #[test]
    fn weight_compression_refuses_a_model_without_parameters() {
        let config = OptimizationConfig {
            weight_compression: true,
            preserve_precision: false,
            ..OptimizationConfig::default()
        };
        let mut model = TestModel::empty();
        let err = WeightCompressionPass::new()
            .apply(&mut model, &config)
            .expect_err("nothing to compress");
        assert!(err.to_string().contains("named_tensors_mut"), "{err}");
    }

    #[test]
    fn precision_preserving_configurations_skip_lossy_compression() {
        let config = OptimizationConfig {
            weight_compression: true,
            preserve_precision: true,
            ..OptimizationConfig::default()
        };
        let mut model = TestModel::with_seed(1.0);
        let before = model.named_tensors()[0].1.to_vec_f32().expect("f32");

        let stats = WeightCompressionPass::new().apply(&mut model, &config).expect("no-op");
        assert_eq!(stats.operations_modified, 0);
        assert_eq!(
            model.named_tensors()[0].1.to_vec_f32().expect("f32"),
            before
        );
    }

    #[test]
    fn test_optimization_stats() {
        let stats = OptimizationStats {
            operations_removed: 10,
            operations_modified: 20,
            size_reduction_bytes: 1024,
            speedup_factor: 1.5,
            precision_preserved: true,
        };

        assert_eq!(stats.operations_removed, 10);
        assert_eq!(stats.speedup_factor, 1.5);
        assert!(stats.precision_preserved);
    }

    #[test]
    fn test_pipeline_stats() {
        let mut stats = PipelineStats::new();

        let pass_stats = OptimizationStats {
            operations_removed: 5,
            operations_modified: 10,
            size_reduction_bytes: 512,
            speedup_factor: 1.2,
            precision_preserved: true,
        };

        stats.add_pass_stats("test_pass".to_string(), pass_stats);

        assert_eq!(stats.total_operations_removed, 5);
        assert_eq!(stats.overall_speedup_factor, 1.2);
        assert!(stats.pass_stats.contains_key("test_pass"));
    }

    #[test]
    fn test_optimization_impact() {
        let pass = WeightCompressionPass::new();
        assert_eq!(pass.expected_impact(), OptimizationImpact::Critical);

        let pass = ConstantFoldingPass::new();
        assert_eq!(pass.expected_impact(), OptimizationImpact::Low);
    }
}
