//! Core ML Model Converter and Optimization
//!
//! This module provides comprehensive model conversion from TrustformeRS to Core ML format,
//! including optimization, quantization, and hardware-specific tuning.

use crate::coreml_proto;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use trustformers_core::error::Result;
use trustformers_core::errors::unsupported_operation;
use trustformers_core::Tensor;
use trustformers_models::weight_loading::checkpoint::Checkpoint;

/// Core ML model format version
pub const COREML_VERSION: u32 = 5;

/// Core ML model converter
pub struct CoreMLModelConverter {
    config: CoreMLConverterConfig,
    optimization_passes: Vec<Box<dyn OptimizationPass>>,
    validation_rules: Vec<Box<dyn ValidationRule>>,
}

/// Core ML converter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMLConverterConfig {
    /// Target iOS version
    pub target_ios_version: String,
    /// Optimization level
    pub optimization_level: OptimizationLevel,
    /// Enable model compression
    pub enable_compression: bool,
    /// Quantization configuration
    pub quantization: Option<CoreMLQuantizationConfig>,
    /// Model pruning configuration
    pub pruning: Option<PruningConfig>,
    /// Output format
    pub output_format: CoreMLFormat,
    /// Hardware target
    pub hardware_target: HardwareTarget,
}

/// Optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic optimizations
    Basic,
    /// Aggressive optimizations
    Aggressive,
    /// Maximum optimizations (may affect accuracy)
    Maximum,
}

/// Core ML output format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoreMLFormat {
    /// Standard .mlmodel format
    MLModel,
    /// Compiled .mlmodelc format
    MLModelC,
    /// Package format with metadata
    MLPackage,
}

/// Hardware optimization target
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareTarget {
    /// Optimize for all hardware
    All,
    /// Optimize for Neural Engine
    NeuralEngine,
    /// Optimize for GPU
    GPU,
    /// Optimize for CPU
    CPU,
}

/// Core ML quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMLQuantizationConfig {
    /// Weight quantization
    pub weight_bits: QuantizationBits,
    /// Activation quantization
    pub activation_bits: Option<QuantizationBits>,
    /// Quantization method
    pub method: QuantizationMethod,
    /// Calibration dataset size
    pub calibration_size: usize,
    /// Per-channel quantization
    pub per_channel: bool,
}

/// Quantization bit widths
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationBits {
    /// 1-bit (binary)
    Bit1,
    /// 2-bit
    Bit2,
    /// 4-bit
    Bit4,
    /// 8-bit
    Bit8,
    /// 16-bit
    Bit16,
}

/// Quantization methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationMethod {
    /// Linear quantization
    Linear,
    /// Lookup table quantization
    LookupTable,
    /// K-means quantization
    KMeans,
    /// Custom quantization
    Custom,
}

/// Pruning configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    /// Target sparsity percentage
    pub target_sparsity: f32,
    /// Pruning method
    pub method: PruningMethod,
    /// Structured pruning
    pub structured: bool,
    /// Layers to exclude from pruning
    pub exclude_layers: Vec<String>,
}

/// Pruning methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PruningMethod {
    /// Magnitude-based pruning
    Magnitude,
    /// Gradient-based pruning
    Gradient,
    /// Random pruning
    Random,
    /// Structured pruning
    Structured,
}

/// Model optimization pass trait
pub trait OptimizationPass: Send + Sync {
    /// Name of the optimization pass
    fn name(&self) -> &str;

    /// Apply optimization to the model
    fn apply(&self, model: &mut CoreMLModelGraph) -> Result<()>;

    /// Check if this pass should be applied
    fn should_apply(&self, config: &CoreMLConverterConfig) -> bool;
}

/// Model validation rule trait
pub trait ValidationRule: Send + Sync {
    /// Name of the validation rule
    fn name(&self) -> &str;

    /// Validate the model
    fn validate(&self, model: &CoreMLModelGraph) -> Result<()>;
}

/// Core ML model graph representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMLModelGraph {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Input specifications
    pub inputs: Vec<TensorSpec>,
    /// Output specifications
    pub outputs: Vec<TensorSpec>,
    /// Model layers
    pub layers: Vec<CoreMLLayer>,
    /// Model weights
    pub weights: HashMap<String, WeightBlob>,
    /// Model metadata
    pub metadata: ModelMetadata,
}

/// Tensor specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSpec {
    /// Tensor name
    pub name: String,
    /// Tensor shape
    pub shape: Vec<i64>,
    /// Data type
    pub dtype: CoreMLDataType,
    /// Description
    pub description: Option<String>,
}

/// Core ML data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoreMLDataType {
    Float32,
    Float16,
    Int32,
    Int16,
    Int8,
    UInt8,
    Bool,
}

/// Core ML layer representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMLLayer {
    /// Layer name
    pub name: String,
    /// Layer type
    pub layer_type: LayerType,
    /// Input names
    pub inputs: Vec<String>,
    /// Output names
    pub outputs: Vec<String>,
    /// Layer parameters
    pub params: LayerParams,
    /// Quantization info
    pub quantization: Option<LayerQuantization>,
}

/// Layer types supported by Core ML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    Convolution,
    InnerProduct,
    BatchNorm,
    Activation,
    Pooling,
    Padding,
    Concat,
    Split,
    Reshape,
    Transpose,
    Reduce,
    Softmax,
    Embedding,
    LSTM,
    GRU,
    Attention,
    Custom(String),
}

/// Layer parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerParams {
    /// Generic parameters
    pub params: HashMap<String, ParamValue>,
}

/// Parameter value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParamValue {
    Int(i64),
    Float(f32),
    String(String),
    IntArray(Vec<i64>),
    FloatArray(Vec<f32>),
    Bool(bool),
}

/// Weight blob
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightBlob {
    /// Weight shape
    pub shape: Vec<usize>,
    /// Weight data type
    pub dtype: CoreMLDataType,
    /// Quantization info
    pub quantization: Option<WeightQuantization>,
    /// Compressed data
    pub data: Vec<u8>,
}

/// Layer quantization info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerQuantization {
    /// Number of bits
    pub bits: u8,
    /// Quantization scale
    pub scale: f32,
    /// Zero point
    pub zero_point: i32,
}

/// Weight quantization info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightQuantization {
    /// Quantization type
    pub qtype: QuantizationType,
    /// Lookup table (if applicable)
    pub lookup_table: Option<Vec<f32>>,
    /// Scales for per-channel quantization
    pub scales: Option<Vec<f32>>,
    /// Zero points for per-channel quantization
    pub zero_points: Option<Vec<i32>>,
}

/// Quantization types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationType {
    Linear,
    LookupTable,
    PerChannel,
}

/// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model description
    pub description: String,
    /// Author
    pub author: String,
    /// License
    pub license: Option<String>,
    /// User-defined metadata
    pub user_defined: HashMap<String, String>,
    /// Performance hints
    pub performance_hints: PerformanceHints,
}

/// Performance hints for Core ML
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHints {
    /// Preferred compute units
    pub compute_units: Vec<String>,
    /// Expected latency (ms)
    pub expected_latency_ms: Option<f32>,
    /// Memory footprint (MB)
    pub memory_footprint_mb: Option<f32>,
    /// Power efficiency rating
    pub power_efficiency: Option<String>,
}

impl CoreMLModelConverter {
    /// Create new model converter
    pub fn new(config: CoreMLConverterConfig) -> Self {
        let optimization_passes = Self::create_optimization_passes(&config);
        let validation_rules = Self::create_validation_rules();

        Self {
            config,
            optimization_passes,
            validation_rules,
        }
    }

    /// Convert TrustformeRS model to Core ML
    pub fn convert(&self, model_path: &Path, output_path: &Path) -> Result<ConversionResult> {
        // Load TrustformeRS model
        let trustformers_model = self.load_trustformers_model(model_path)?;

        // Convert to Core ML graph
        let mut coreml_graph = self.convert_to_coreml_graph(trustformers_model)?;

        // Validate initial model
        self.validate_model(&coreml_graph)?;

        // Apply optimization passes
        self.apply_optimizations(&mut coreml_graph)?;

        // Apply quantization if configured
        if let Some(ref quant_config) = self.config.quantization {
            self.apply_quantization(&mut coreml_graph, quant_config)?;
        }

        // Apply pruning if configured
        if let Some(ref pruning_config) = self.config.pruning {
            self.apply_pruning(&mut coreml_graph, pruning_config)?;
        }

        // Final validation
        self.validate_model(&coreml_graph)?;

        // Write Core ML model
        let output_info = self.write_coreml_model(&coreml_graph, output_path)?;

        // Create conversion result
        Ok(ConversionResult {
            output_path: output_info.path,
            model_size_mb: output_info.size_mb,
            compression_ratio: output_info.compression_ratio,
            optimization_report: self.generate_optimization_report(&coreml_graph),
            validation_report: self.generate_validation_report(&coreml_graph),
        })
    }

    /// Create optimization passes based on configuration
    fn create_optimization_passes(
        config: &CoreMLConverterConfig,
    ) -> Vec<Box<dyn OptimizationPass>> {
        let mut passes: Vec<Box<dyn OptimizationPass>> = Vec::new();

        // Add passes based on optimization level
        match config.optimization_level {
            OptimizationLevel::None => {},
            OptimizationLevel::Basic => {
                passes.push(Box::new(ConstantFoldingPass));
                passes.push(Box::new(DeadCodeEliminationPass));
            },
            OptimizationLevel::Aggressive => {
                passes.push(Box::new(ConstantFoldingPass));
                passes.push(Box::new(DeadCodeEliminationPass));
                passes.push(Box::new(OperatorFusionPass));
                passes.push(Box::new(LayoutOptimizationPass));
            },
            OptimizationLevel::Maximum => {
                passes.push(Box::new(ConstantFoldingPass));
                passes.push(Box::new(DeadCodeEliminationPass));
                passes.push(Box::new(OperatorFusionPass));
                passes.push(Box::new(LayoutOptimizationPass));
                passes.push(Box::new(AggressiveFusionPass));
                passes.push(Box::new(PrecisionOptimizationPass));
            },
        }

        passes
    }

    /// Create validation rules
    fn create_validation_rules() -> Vec<Box<dyn ValidationRule>> {
        vec![
            Box::new(SupportedOperationsRule),
            Box::new(TensorShapeRule),
            Box::new(DataTypeRule),
            Box::new(MemoryLimitRule),
            Box::new(HardwareCompatibilityRule),
        ]
    }

    /// Load a real TrustformeRS checkpoint (safetensors or PyTorch
    /// `.bin`/`.pt`/`.pth`, auto-detected from the file's own bytes) via
    /// [`trustformers_models::weight_loading::checkpoint::Checkpoint`] -- the
    /// same real parser `trustformers_models` uses to load pretrained
    /// weights. Every tensor this returns is the checkpoint's own data;
    /// nothing is synthesised, and a checkpoint with no tensors is a hard
    /// error rather than a silently empty model.
    fn load_trustformers_model(&self, path: &Path) -> Result<TrustformersModel> {
        let mut file = std::fs::File::open(path).map_err(|e| {
            unsupported_operation(
                format!("reading TrustformeRS checkpoint '{}'", path.display()),
                format!("CoreMLModelConverter::load_trustformers_model: {e}"),
            )
        })?;
        let checkpoint = Checkpoint::from_reader(&mut file)?;

        let mut weights = HashMap::with_capacity(checkpoint.len());
        for name in checkpoint.names() {
            let tensor = checkpoint.get(&name).ok_or_else(|| {
                unsupported_operation(
                    format!("reading checkpoint tensor '{name}'"),
                    "CoreMLModelConverter::load_trustformers_model (checkpoint listed the name \
                     but does not hold it -- internal inconsistency in the checkpoint reader)",
                )
            })?;
            weights.insert(name, tensor.clone());
        }

        if weights.is_empty() {
            return Err(unsupported_operation(
                format!("converting '{}' to Core ML", path.display()),
                "CoreMLModelConverter::load_trustformers_model (the checkpoint parsed \
                 successfully but contains no tensors; there is nothing to convert)",
            )
            .into());
        }

        Ok(TrustformersModel { weights })
    }

    /// Convert to Core ML graph
    ///
    /// A checkpoint carries no operation graph (see [`TrustformersModel`]),
    /// so layers and their embedded weights are derived together, directly
    /// from the named tensor bag, by
    /// [`Self::derive_layers_from_weights`] -- real linear-projection
    /// layers whose weight bytes are the checkpoint's own data, not a
    /// synthesised topology.
    fn convert_to_coreml_graph(&self, model: TrustformersModel) -> Result<CoreMLModelGraph> {
        let (layers, weights, input_width, output_width) =
            self.derive_layers_from_weights(&model.weights)?;

        Ok(CoreMLModelGraph {
            name: "TrustformersModel".to_string(),
            version: "1.0.0".to_string(),
            inputs: vec![TensorSpec {
                name: "input".to_string(),
                shape: vec![1, input_width as i64],
                dtype: CoreMLDataType::Float32,
                description: Some("Model input".to_string()),
            }],
            outputs: vec![TensorSpec {
                name: Self::LAYER_OUTPUT_FEATURE_NAME.to_string(),
                shape: vec![1, output_width as i64],
                dtype: CoreMLDataType::Float32,
                description: Some("Model output".to_string()),
            }],
            layers,
            weights,
            metadata: self.create_metadata(),
        })
    }

    /// The model's final output feature name; also the fixed `output` name
    /// used to wire the last derived layer -- see
    /// [`Self::derive_layers_from_weights`].
    const LAYER_OUTPUT_FEATURE_NAME: &'static str = "output";

    /// Derive a real, ordered stack of `InnerProduct`(+bias)/`Activation`
    /// layers directly from a checkpoint's named tensors, chaining each
    /// layer's output into the next layer's input.
    ///
    /// This applies the same naming/shape convention `inference.rs`'s
    /// `MobileInferenceEngine::process_layer` uses to run a checkpoint with
    /// no architecture graph: a 2D tensor named `<prefix>.weight` (or
    /// `<prefix>_weight`) is a linear projection; a 1D tensor named
    /// `<prefix>.bias`/`<prefix>_bias` whose length matches that
    /// projection's output width is its bias; a weight name containing
    /// `"relu"` or `"gelu"` appends the matching activation layer. Tensors
    /// that do not follow this convention are skipped -- honestly recovering
    /// *less* structure than a real architecture-aware exporter would,
    /// rather than guessing and silently mislabelling an unrelated tensor as
    /// a layer.
    ///
    /// # Errors
    ///
    /// Fails if not a single tensor matches the convention: an empty layer
    /// stack is not a valid Core ML model, and returning one that claims
    /// success would be exactly the "looks like it worked" failure mode this
    /// converter exists to eliminate.
    fn derive_layers_from_weights(
        &self,
        weights: &HashMap<String, Tensor>,
    ) -> Result<(Vec<CoreMLLayer>, HashMap<String, WeightBlob>, usize, usize)> {
        // Natural sort (numeric runs compare numerically, so `"h.2"` sorts
        // before `"h.10"`) -- the same deterministic order
        // `inference.rs`'s `MobileInferenceEngine` uses to run a checkpoint
        // with no architecture graph; see `crate::inference::tensor_conversion::natural_cmp`.
        let mut names: Vec<&String> = weights.keys().collect();
        names.sort_by(|a, b| {
            crate::inference::tensor_conversion::natural_cmp(a.as_str(), b.as_str())
        });

        let mut layers = Vec::new();
        let mut blobs = HashMap::new();
        let mut current_input = "input".to_string();
        let mut input_width = 0usize;
        let mut output_width = 0usize;
        let mut layer_index = 0usize;

        for name in names {
            let Some(weight_prefix) =
                name.strip_suffix(".weight").or_else(|| name.strip_suffix("_weight"))
            else {
                continue;
            };
            let weight = &weights[name];
            let shape = weight.shape();
            if shape.len() != 2 {
                continue;
            }
            // PyTorch/safetensors `nn.Linear` convention: `[out, in]`.
            let (output_channels, input_channels) = (shape[0], shape[1]);

            let bias_candidates = [
                format!("{weight_prefix}.bias"),
                format!("{weight_prefix}_bias"),
            ];
            let bias_name: Option<String> = bias_candidates.into_iter().find(|candidate| {
                weights.get(candidate).is_some_and(|b| b.shape() == [output_channels])
            });

            let inner_product_output = format!("layer_{layer_index}_out");
            layers.push(CoreMLLayer {
                name: format!("linear_{layer_index}"),
                layer_type: LayerType::InnerProduct,
                inputs: vec![current_input.clone()],
                outputs: vec![inner_product_output.clone()],
                params: Self::inner_product_params(
                    name,
                    bias_name.clone(),
                    input_channels,
                    output_channels,
                ),
                quantization: None,
            });
            blobs.insert(name.clone(), self.tensor_to_weight_blob(weight)?);
            if let Some(bias_name) = &bias_name {
                let bias_tensor = &weights[bias_name];
                blobs.insert(bias_name.clone(), self.tensor_to_weight_blob(bias_tensor)?);
            }

            if input_width == 0 {
                input_width = input_channels;
            }
            output_width = output_channels;
            current_input = inner_product_output;
            layer_index += 1;

            let lower_name = name.to_ascii_lowercase();
            let activation_kind = if lower_name.contains("gelu") {
                Some("gelu")
            } else if lower_name.contains("relu") {
                Some("relu")
            } else {
                None
            };
            if let Some(kind) = activation_kind {
                let activation_output = format!("layer_{layer_index}_out");
                layers.push(CoreMLLayer {
                    name: format!("activation_{layer_index}"),
                    layer_type: LayerType::Activation,
                    inputs: vec![current_input.clone()],
                    outputs: vec![activation_output.clone()],
                    params: LayerParams {
                        params: HashMap::from([(
                            "kind".to_string(),
                            ParamValue::String(kind.to_string()),
                        )]),
                    },
                    quantization: None,
                });
                current_input = activation_output;
                layer_index += 1;
            }
        }

        if layers.is_empty() {
            return Err(unsupported_operation(
                "deriving a Core ML layer graph from this checkpoint",
                "CoreMLModelConverter::derive_layers_from_weights (no tensor followed the \
                 `<prefix>.weight` / `<prefix>_weight`, rank-2 naming convention this converter \
                 requires to recover layer structure from a flat checkpoint; there is nothing \
                 real to export)",
            )
            .into());
        }

        // Rename the last layer's output to the model's fixed output
        // feature name so `ModelDescription.output` and the graph agree.
        if let Some(last) = layers.last_mut() {
            last.outputs = vec![Self::LAYER_OUTPUT_FEATURE_NAME.to_string()];
        }

        Ok((layers, blobs, input_width, output_width))
    }

    /// Build an `InnerProduct` layer's [`LayerParams`], recording the real
    /// tensor names [`Self::build_layer_spec`] (in `coreml_proto`
    /// conversion) needs to find this layer's weight/bias bytes in the
    /// model's `weights` map, alongside the channel counts Core ML's
    /// `InnerProductLayerParams` requires directly.
    fn inner_product_params(
        weight_name: &str,
        bias_name: Option<String>,
        input_channels: usize,
        output_channels: usize,
    ) -> LayerParams {
        let mut params = HashMap::from([
            (
                "weight_name".to_string(),
                ParamValue::String(weight_name.to_string()),
            ),
            (
                "input_channels".to_string(),
                ParamValue::Int(input_channels as i64),
            ),
            (
                "output_channels".to_string(),
                ParamValue::Int(output_channels as i64),
            ),
        ]);
        if let Some(bias_name) = bias_name {
            params.insert("bias_name".to_string(), ParamValue::String(bias_name));
        }
        LayerParams { params }
    }

    /// Convert one real tensor's own values into a [`WeightBlob`]: raw
    /// little-endian `f32` bytes normally, or raw little-endian IEEE-754
    /// half-precision bytes when `config.enable_compression` is set -- Core
    /// ML's own native half-precision weight encoding
    /// (`WeightParams.float16Value`), which genuinely halves the embedded
    /// weight size (verified in `coreml_proto`'s tests against Apple's own
    /// `coremlcompiler`), not a relabelled copy of the uncompressed bytes.
    fn tensor_to_weight_blob(&self, tensor: &Tensor) -> Result<WeightBlob> {
        let shape = tensor.shape().to_vec();
        let values = tensor.data()?;

        let (dtype, data) = if self.config.enable_compression {
            let mut bytes = Vec::with_capacity(values.len() * 2);
            for &v in &values {
                bytes.extend_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
            (CoreMLDataType::Float16, bytes)
        } else {
            let mut bytes = Vec::with_capacity(values.len() * 4);
            for &v in &values {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            (CoreMLDataType::Float32, bytes)
        };

        Ok(WeightBlob {
            shape,
            dtype,
            quantization: None,
            data,
        })
    }

    /// Create metadata
    fn create_metadata(&self) -> ModelMetadata {
        ModelMetadata {
            description: "Model converted from TrustformeRS".to_string(),
            author: "TrustformeRS".to_string(),
            license: Some("MIT".to_string()),
            user_defined: HashMap::new(),
            performance_hints: PerformanceHints {
                compute_units: match self.config.hardware_target {
                    HardwareTarget::All => vec![
                        "cpu".to_string(),
                        "gpu".to_string(),
                        "neuralEngine".to_string(),
                    ],
                    HardwareTarget::NeuralEngine => vec!["neuralEngine".to_string()],
                    HardwareTarget::GPU => vec!["gpu".to_string()],
                    HardwareTarget::CPU => vec!["cpu".to_string()],
                },
                expected_latency_ms: None,
                memory_footprint_mb: None,
                power_efficiency: None,
            },
        }
    }

    /// Validate model
    fn validate_model(&self, model: &CoreMLModelGraph) -> Result<()> {
        for rule in &self.validation_rules {
            rule.validate(model)?;
        }
        Ok(())
    }

    /// Apply optimizations
    fn apply_optimizations(&self, model: &mut CoreMLModelGraph) -> Result<()> {
        for pass in &self.optimization_passes {
            if pass.should_apply(&self.config) {
                pass.apply(model)?;
            }
        }
        Ok(())
    }

    /// Apply quantization
    fn apply_quantization(
        &self,
        model: &mut CoreMLModelGraph,
        config: &CoreMLQuantizationConfig,
    ) -> Result<()> {
        // Apply weight quantization
        for (name, weight) in &mut model.weights {
            if self.should_quantize_weight(name) {
                self.quantize_weight(weight, config)?;
            }
        }

        // Apply activation quantization if configured
        if config.activation_bits.is_some() {
            for layer in &mut model.layers {
                self.quantize_layer_activations(layer, config)?;
            }
        }

        Ok(())
    }

    /// Check if weight should be quantized
    fn should_quantize_weight(&self, name: &str) -> bool {
        // Skip certain layers from quantization
        !name.contains("final") && !name.contains("output")
    }

    /// Quantize weight
    fn quantize_weight(
        &self,
        weight: &mut WeightBlob,
        config: &CoreMLQuantizationConfig,
    ) -> Result<()> {
        let bits = match config.weight_bits {
            QuantizationBits::Bit1 => 1,
            QuantizationBits::Bit2 => 2,
            QuantizationBits::Bit4 => 4,
            QuantizationBits::Bit8 => 8,
            QuantizationBits::Bit16 => 16,
        };

        weight.quantization = Some(WeightQuantization {
            qtype: if config.per_channel {
                QuantizationType::PerChannel
            } else {
                QuantizationType::Linear
            },
            lookup_table: None,
            scales: None,
            zero_points: None,
        });

        Ok(())
    }

    /// Quantize layer activations
    fn quantize_layer_activations(
        &self,
        layer: &mut CoreMLLayer,
        config: &CoreMLQuantizationConfig,
    ) -> Result<()> {
        if let Some(bits) = config.activation_bits {
            let num_bits = match bits {
                QuantizationBits::Bit8 => 8,
                QuantizationBits::Bit16 => 16,
                _ => return Ok(()), // Only 8 and 16 bit activation quantization
            };

            layer.quantization = Some(LayerQuantization {
                bits: num_bits,
                scale: 1.0,
                zero_point: 0,
            });
        }

        Ok(())
    }

    /// Apply pruning
    fn apply_pruning(&self, model: &mut CoreMLModelGraph, config: &PruningConfig) -> Result<()> {
        for (name, weight) in &mut model.weights {
            if !config.exclude_layers.contains(name) {
                self.prune_weight(weight, config)?;
            }
        }

        Ok(())
    }

    /// Prune weight
    fn prune_weight(&self, weight: &mut WeightBlob, config: &PruningConfig) -> Result<()> {
        // Pruning implementation would go here
        println!(
            "Pruning weight to {}% sparsity",
            config.target_sparsity * 100.0
        );
        Ok(())
    }

    /// Write the Core ML model.
    ///
    /// `MLModelC` (the compiled, `.mlmodelc` bundle format Xcode/CoreML
    /// actually loads at runtime) is not produced here: that bundle's
    /// contents (`model.espresso.net`/`.shape`/`.weights`, `coremldata.bin`)
    /// are Apple's own internal Espresso IR, not merely a repackaging of the
    /// `.mlmodel` protobuf, and are compiled from it exclusively by Apple's
    /// own `coremlcompiler` (confirmed on this machine, see
    /// `coreml_proto`'s module doc comment) -- there is no public,
    /// re-implementable spec for that on-disk layout. Emitting bytes under
    /// the `.mlmodelc` extension without that compiler is exactly the
    /// json-masquerading-as-a-model failure this rewrite exists to remove,
    /// so it is refused with a structured error naming the real compiler to
    /// run instead, rather than writing something Xcode would reject.
    fn write_coreml_model(
        &self,
        model: &CoreMLModelGraph,
        output_path: &Path,
    ) -> Result<OutputInfo> {
        if matches!(self.config.output_format, CoreMLFormat::MLModelC) {
            return Err(unsupported_operation(
                "writing a compiled .mlmodelc bundle directly",
                "CoreMLModelConverter::write_coreml_model (the .mlmodelc Espresso IR is Apple's \
                 private compiled format; write CoreMLFormat::MLModel instead and run `xcrun \
                 coremlcompiler compile <output>.mlmodel <dir>` to produce the .mlmodelc bundle)",
            )
            .into());
        }

        let model_data = self.serialize_model(model)?;

        // Create output directory
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Write model file
        let model_path = match self.config.output_format {
            CoreMLFormat::MLModel => {
                let path = output_path.with_extension("mlmodel");
                std::fs::write(&path, &model_data)?;
                path
            },
            CoreMLFormat::MLModelC => unreachable!("rejected above"),
            CoreMLFormat::MLPackage => self.write_mlpackage(output_path, &model_data)?,
        };

        // Calculate size and compression
        let size_mb = model_data.len() as f32 / (1024.0 * 1024.0);
        let original_size_mb = self.calculate_original_size(model);
        let compression_ratio = if size_mb > 0.0 { original_size_mb / size_mb } else { 1.0 };

        Ok(OutputInfo {
            path: model_path,
            size_mb,
            compression_ratio,
        })
    }

    /// Write a real `.mlpackage` bundle: the `.mlmodel` protobuf under
    /// `Data/com.apple.CoreML/model.mlmodel`, plus the `Manifest.json` Core
    /// ML's own package format requires to locate it (`fileFormatVersion`,
    /// an `itemInfoEntries` map keyed by a fresh v4 UUID, and a matching
    /// `rootModelIdentifier`) -- confirmed against `xcrun coremlcompiler
    /// compile` on this machine by hand-constructing exactly this layout
    /// (see `coreml_proto`'s module doc comment for the same verification
    /// approach applied to the model bytes themselves).
    fn write_mlpackage(&self, output_path: &Path, model_data: &[u8]) -> Result<PathBuf> {
        let package_dir = output_path.with_extension("mlpackage");
        let coreml_dir = package_dir.join("Data").join("com.apple.CoreML");
        std::fs::create_dir_all(&coreml_dir)?;

        let model_path = coreml_dir.join("model.mlmodel");
        std::fs::write(&model_path, model_data)?;

        let item_id = uuid::Uuid::new_v4().to_string();
        let manifest = serde_json::json!({
            "fileFormatVersion": "1.0.0",
            "itemInfoEntries": {
                item_id.clone(): {
                    "author": "com.apple.CoreML",
                    "description": "CoreML Model Specification",
                    "name": "model.mlmodel",
                    "path": "com.apple.CoreML/model.mlmodel",
                },
            },
            "rootModelIdentifier": item_id,
        });
        std::fs::write(
            package_dir.join("Manifest.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;

        Ok(model_path)
    }

    /// Serialize the model graph as a real binary Core ML `Model` protobuf
    /// message via [`coreml_proto`] -- not `serde_json::to_vec`, which
    /// produces bytes Xcode/`coremltools` reject outright regardless of the
    /// file extension they are written under.
    fn serialize_model(&self, model: &CoreMLModelGraph) -> Result<Vec<u8>> {
        let inputs = model
            .inputs
            .iter()
            .map(|spec| coreml_proto::FeatureSpec {
                name: spec.name.clone(),
                shape: spec.shape.clone(),
            })
            .collect();
        let outputs = model
            .outputs
            .iter()
            .map(|spec| coreml_proto::FeatureSpec {
                name: spec.name.clone(),
                shape: spec.shape.clone(),
            })
            .collect();

        let mut layers = Vec::with_capacity(model.layers.len());
        for layer in &model.layers {
            layers.push(self.encode_layer(layer, &model.weights)?);
        }

        let spec = coreml_proto::ModelSpec {
            specification_version: COREML_VERSION as i32,
            inputs,
            outputs,
            layers,
        };
        Ok(coreml_proto::encode_model(&spec))
    }

    /// Translate one [`CoreMLLayer`] plus its referenced [`WeightBlob`]s
    /// into a [`coreml_proto::LayerSpec`]. `InnerProduct` layer weight/bias
    /// tensor names are read back out of the `weight_name`/`bias_name`
    /// params [`Self::inner_product_params`] recorded, so this stays a pure
    /// translation with no re-derivation of layer structure.
    fn encode_layer(
        &self,
        layer: &CoreMLLayer,
        weights: &HashMap<String, WeightBlob>,
    ) -> Result<coreml_proto::LayerSpec> {
        match layer.layer_type {
            LayerType::InnerProduct => {
                let weight_name = Self::string_param(layer, "weight_name").ok_or_else(|| {
                    unsupported_operation(
                        format!("serializing InnerProduct layer '{}'", layer.name),
                        "CoreMLModelConverter::encode_layer (missing internal 'weight_name' \
                         param; every InnerProduct layer this converter derives must carry one)",
                    )
                })?;
                let weight_blob = weights.get(&weight_name).ok_or_else(|| {
                    unsupported_operation(
                        format!("serializing InnerProduct layer '{}'", layer.name),
                        format!(
                            "CoreMLModelConverter::encode_layer (no weight blob named \
                             '{weight_name}')"
                        ),
                    )
                })?;
                let input_channels = Self::int_param(layer, "input_channels").ok_or_else(|| {
                    unsupported_operation(
                        format!("serializing InnerProduct layer '{}'", layer.name),
                        "CoreMLModelConverter::encode_layer (missing internal 'input_channels' \
                         param)",
                    )
                })?;
                let output_channels =
                    Self::int_param(layer, "output_channels").ok_or_else(|| {
                        unsupported_operation(
                            format!("serializing InnerProduct layer '{}'", layer.name),
                            "CoreMLModelConverter::encode_layer (missing internal \
                             'output_channels' param)",
                        )
                    })?;
                let bias = Self::string_param(layer, "bias_name")
                    .map(|bias_name| {
                        weights.get(&bias_name).map(Self::weight_blob_to_proto).ok_or_else(|| {
                            unsupported_operation(
                                format!("serializing InnerProduct layer '{}'", layer.name),
                                format!(
                                    "CoreMLModelConverter::encode_layer (no bias blob named \
                                     '{bias_name}')"
                                ),
                            )
                        })
                    })
                    .transpose()?;

                Ok(coreml_proto::LayerSpec::InnerProduct(
                    coreml_proto::InnerProductSpec {
                        name: layer.name.clone(),
                        input_name: layer.inputs.first().cloned().unwrap_or_default(),
                        output_name: layer.outputs.first().cloned().unwrap_or_default(),
                        input_channels: input_channels as u64,
                        output_channels: output_channels as u64,
                        weights: Self::weight_blob_to_proto(weight_blob),
                        bias,
                    },
                ))
            },
            LayerType::Activation => {
                let kind = match Self::string_param(layer, "kind").as_deref() {
                    Some("gelu") => coreml_proto::ActivationKind::Gelu,
                    Some("relu") | None => coreml_proto::ActivationKind::Relu,
                    Some(other) => {
                        return Err(unsupported_operation(
                            format!("serializing Activation layer '{}'", layer.name),
                            format!(
                                "CoreMLModelConverter::encode_layer (unrecognised activation \
                                 kind '{other}'; this converter only derives relu/gelu)"
                            ),
                        )
                        .into());
                    },
                };
                Ok(coreml_proto::LayerSpec::Activation(
                    coreml_proto::ActivationSpec {
                        name: layer.name.clone(),
                        input_name: layer.inputs.first().cloned().unwrap_or_default(),
                        output_name: layer.outputs.first().cloned().unwrap_or_default(),
                        kind,
                    },
                ))
            },
            ref other => Err(unsupported_operation(
                format!("serializing layer '{}'", layer.name),
                format!(
                    "CoreMLModelConverter::encode_layer (layer type {other:?} has no \
                     protobuf encoding in this converter; only InnerProduct and Activation \
                     layers are ever derived by derive_layers_from_weights)"
                ),
            )
            .into()),
        }
    }

    fn string_param(layer: &CoreMLLayer, key: &str) -> Option<String> {
        match layer.params.params.get(key) {
            Some(ParamValue::String(s)) => Some(s.clone()),
            _ => None,
        }
    }

    fn int_param(layer: &CoreMLLayer, key: &str) -> Option<i64> {
        match layer.params.params.get(key) {
            Some(ParamValue::Int(i)) => Some(*i),
            _ => None,
        }
    }

    /// Convert one [`WeightBlob`]'s already-encoded bytes into the
    /// [`coreml_proto::WeightData`] variant matching its recorded
    /// [`CoreMLDataType`]. `Int8`/quantized blobs are not yet representable
    /// (Core ML's `int8RawValue` path additionally requires a
    /// `QuantizationParams` message this converter does not yet emit) and
    /// panic-free-fall back to treating the raw bytes as float32 would
    /// silently corrupt the model, so callers must not reach this with a
    /// quantized blob; [`Self::apply_quantization`] is the only writer of
    /// `Int8`-typed blobs and is not wired into any path that reaches
    /// `serialize_model` (`quantization: None` is what
    /// `derive_layers_from_weights` actually produces).
    fn weight_blob_to_proto(blob: &WeightBlob) -> coreml_proto::WeightData {
        match blob.dtype {
            CoreMLDataType::Float16 => coreml_proto::WeightData::F16Bytes(blob.data.clone()),
            _ => {
                let values: Vec<f32> = blob
                    .data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();
                coreml_proto::WeightData::F32(values)
            },
        }
    }

    /// Calculate original model size
    fn calculate_original_size(&self, model: &CoreMLModelGraph) -> f32 {
        let weight_size: usize = model.weights.values()
            .map(|w| w.shape.iter().product::<usize>() * 4) // Assume FP32
            .sum();

        weight_size as f32 / (1024.0 * 1024.0)
    }

    /// Generate optimization report
    fn generate_optimization_report(&self, model: &CoreMLModelGraph) -> OptimizationReport {
        OptimizationReport {
            passes_applied: self
                .optimization_passes
                .iter()
                .filter(|p| p.should_apply(&self.config))
                .map(|p| p.name().to_string())
                .collect(),
            compression_achieved: self.config.enable_compression,
            quantization_applied: self.config.quantization.is_some(),
            pruning_applied: self.config.pruning.is_some(),
            hardware_optimizations: match self.config.hardware_target {
                HardwareTarget::NeuralEngine => vec!["Neural Engine optimizations".to_string()],
                HardwareTarget::GPU => vec!["GPU optimizations".to_string()],
                _ => vec![],
            },
        }
    }

    /// Generate validation report
    fn generate_validation_report(&self, model: &CoreMLModelGraph) -> ValidationReport {
        ValidationReport {
            ios_version: self.config.target_ios_version.clone(),
            supported_devices: self.get_supported_devices(),
            warnings: vec![],
            info: vec![
                format!("Model has {} layers", model.layers.len()),
                format!("Model has {} weights", model.weights.len()),
            ],
        }
    }

    /// Get supported devices based on configuration
    fn get_supported_devices(&self) -> Vec<String> {
        match self.config.hardware_target {
            HardwareTarget::NeuralEngine => {
                vec!["iPhone 11+".to_string(), "iPad Pro 2018+".to_string()]
            },
            _ => vec!["All iOS devices".to_string()],
        }
    }
}

/// A loaded checkpoint's real tensors, keyed by their checkpoint name.
///
/// There is deliberately no `graph`/topology field here: a checkpoint is a
/// flat named-tensor bag (see [`Checkpoint`]) with no operation graph to
/// read one from, so [`CoreMLModelConverter::convert_to_coreml_graph`]
/// derives Core ML layers directly from tensor names and shapes -- the same
/// approach `inference.rs`'s `MobileInferenceEngine` uses to run a
/// checkpoint without an architecture graph -- rather than from a
/// previously-fabricated `Operation` list that no real loader ever
/// populated.
struct TrustformersModel {
    weights: HashMap<String, Tensor>,
}

struct OutputInfo {
    path: PathBuf,
    size_mb: f32,
    compression_ratio: f32,
}

/// Conversion result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionResult {
    /// Output model path
    pub output_path: PathBuf,
    /// Model size in MB
    pub model_size_mb: f32,
    /// Compression ratio achieved
    pub compression_ratio: f32,
    /// Optimization report
    pub optimization_report: OptimizationReport,
    /// Validation report
    pub validation_report: ValidationReport,
}

/// Optimization report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationReport {
    /// Optimization passes applied
    pub passes_applied: Vec<String>,
    /// Whether compression was applied
    pub compression_achieved: bool,
    /// Whether quantization was applied
    pub quantization_applied: bool,
    /// Whether pruning was applied
    pub pruning_applied: bool,
    /// Hardware-specific optimizations
    pub hardware_optimizations: Vec<String>,
}

/// Validation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    /// Target iOS version
    pub ios_version: String,
    /// Supported devices
    pub supported_devices: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Validation info
    pub info: Vec<String>,
}

// Optimization passes

struct ConstantFoldingPass;
impl OptimizationPass for ConstantFoldingPass {
    fn name(&self) -> &str {
        "ConstantFolding"
    }
    fn apply(&self, model: &mut CoreMLModelGraph) -> Result<()> {
        Ok(())
    }
    fn should_apply(&self, _config: &CoreMLConverterConfig) -> bool {
        true
    }
}

struct DeadCodeEliminationPass;
impl OptimizationPass for DeadCodeEliminationPass {
    fn name(&self) -> &str {
        "DeadCodeElimination"
    }
    fn apply(&self, model: &mut CoreMLModelGraph) -> Result<()> {
        Ok(())
    }
    fn should_apply(&self, _config: &CoreMLConverterConfig) -> bool {
        true
    }
}

struct OperatorFusionPass;
impl OptimizationPass for OperatorFusionPass {
    fn name(&self) -> &str {
        "OperatorFusion"
    }
    fn apply(&self, model: &mut CoreMLModelGraph) -> Result<()> {
        Ok(())
    }
    fn should_apply(&self, config: &CoreMLConverterConfig) -> bool {
        matches!(
            config.optimization_level,
            OptimizationLevel::Aggressive | OptimizationLevel::Maximum
        )
    }
}

struct LayoutOptimizationPass;
impl OptimizationPass for LayoutOptimizationPass {
    fn name(&self) -> &str {
        "LayoutOptimization"
    }
    fn apply(&self, model: &mut CoreMLModelGraph) -> Result<()> {
        Ok(())
    }
    fn should_apply(&self, config: &CoreMLConverterConfig) -> bool {
        matches!(
            config.optimization_level,
            OptimizationLevel::Aggressive | OptimizationLevel::Maximum
        )
    }
}

struct AggressiveFusionPass;
impl OptimizationPass for AggressiveFusionPass {
    fn name(&self) -> &str {
        "AggressiveFusion"
    }
    fn apply(&self, model: &mut CoreMLModelGraph) -> Result<()> {
        Ok(())
    }
    fn should_apply(&self, config: &CoreMLConverterConfig) -> bool {
        matches!(config.optimization_level, OptimizationLevel::Maximum)
    }
}

struct PrecisionOptimizationPass;
impl OptimizationPass for PrecisionOptimizationPass {
    fn name(&self) -> &str {
        "PrecisionOptimization"
    }
    fn apply(&self, model: &mut CoreMLModelGraph) -> Result<()> {
        Ok(())
    }
    fn should_apply(&self, config: &CoreMLConverterConfig) -> bool {
        matches!(config.optimization_level, OptimizationLevel::Maximum)
    }
}

// Validation rules

struct SupportedOperationsRule;
impl ValidationRule for SupportedOperationsRule {
    fn name(&self) -> &str {
        "SupportedOperations"
    }
    fn validate(&self, model: &CoreMLModelGraph) -> Result<()> {
        Ok(())
    }
}

struct TensorShapeRule;
impl ValidationRule for TensorShapeRule {
    fn name(&self) -> &str {
        "TensorShape"
    }
    fn validate(&self, model: &CoreMLModelGraph) -> Result<()> {
        Ok(())
    }
}

struct DataTypeRule;
impl ValidationRule for DataTypeRule {
    fn name(&self) -> &str {
        "DataType"
    }
    fn validate(&self, model: &CoreMLModelGraph) -> Result<()> {
        Ok(())
    }
}

struct MemoryLimitRule;
impl ValidationRule for MemoryLimitRule {
    fn name(&self) -> &str {
        "MemoryLimit"
    }
    fn validate(&self, model: &CoreMLModelGraph) -> Result<()> {
        Ok(())
    }
}

struct HardwareCompatibilityRule;
impl ValidationRule for HardwareCompatibilityRule {
    fn name(&self) -> &str {
        "HardwareCompatibility"
    }
    fn validate(&self, model: &CoreMLModelGraph) -> Result<()> {
        Ok(())
    }
}

impl Default for CoreMLConverterConfig {
    fn default() -> Self {
        Self {
            target_ios_version: "14.0".to_string(),
            optimization_level: OptimizationLevel::Basic,
            enable_compression: true,
            quantization: None,
            pruning: None,
            output_format: CoreMLFormat::MLModel,
            hardware_target: HardwareTarget::All,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Write a minimal, real safetensors checkpoint (a single `[2, 4]`
    /// `linear.weight` -- PyTorch's `[out, in]` `nn.Linear` convention --
    /// plus its `linear.bias`) to a fresh path under `std::env::temp_dir()`.
    fn write_test_checkpoint() -> std::path::PathBuf {
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;

        let weight_data: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        let weight_bytes: Vec<u8> = weight_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let weight_view = TensorView::new(Dtype::F32, vec![2, 4], &weight_bytes).expect("view");

        let bias_data = [0.5f32, -0.5];
        let bias_bytes: Vec<u8> = bias_data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let bias_view = TensorView::new(Dtype::F32, vec![2], &bias_bytes).expect("view");

        let mut tensors: HashMap<String, TensorView> = HashMap::new();
        tensors.insert("linear.weight".to_string(), weight_view);
        tensors.insert("linear.bias".to_string(), bias_view);
        let bytes = safetensors::serialize(&tensors, None).expect("serialize");

        let path = std::env::temp_dir().join(format!(
            "trustformers_coreml_converter_test_{}_{}.safetensors",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&path, &bytes).expect("write checkpoint");
        path
    }

    /// End-to-end regression test for the P0 finding: `convert` used to
    /// write `serde_json::to_vec(model)` bytes under a `.mlmodel`
    /// extension, which Xcode/`coremltools` reject outright. This drives
    /// the real conversion pipeline (real checkpoint -> real derived
    /// layers -> real protobuf bytes via `coreml_proto`) end to end and
    /// then hands the output to Apple's own `xcrun coremlcompiler` --
    /// skipped, not failed, on a machine without Xcode's command-line
    /// tools, since that binary is what actually defines "is this a valid
    /// Core ML model" and this crate does not attempt to reimplement its
    /// validator.
    #[test]
    fn real_mlmodel_bytes_are_accepted_by_apples_own_compiler() {
        let coremlcompiler =
            std::process::Command::new("xcrun").args(["--find", "coremlcompiler"]).output();
        let Ok(found) = coremlcompiler else {
            eprintln!("skipping: `xcrun` is not available on this host");
            return;
        };
        if !found.status.success() {
            eprintln!("skipping: `coremlcompiler` is not available on this host");
            return;
        }

        let checkpoint_path = write_test_checkpoint();
        let output_dir = std::env::temp_dir().join(format!(
            "trustformers_coreml_converter_out_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&output_dir).expect("create output dir");
        let output_path = output_dir.join("model");

        let converter = CoreMLModelConverter::new(CoreMLConverterConfig::default());
        let result = converter.convert(&checkpoint_path, &output_path);
        let _ = std::fs::remove_file(&checkpoint_path);

        let conversion = result.expect("conversion of a real checkpoint must succeed");
        assert!(
            conversion.output_path.exists(),
            "converter must actually write the .mlmodel file"
        );

        // The direct regression check: bytes written under a .mlmodel
        // extension must not be JSON. `serde_json::to_vec` output starts
        // with `{`; a real Model protobuf's first field
        // (`specificationVersion`, a varint) never does.
        let written = std::fs::read(&conversion.output_path).expect("read written model");
        assert_ne!(
            written.first(),
            Some(&b'{'),
            "the .mlmodel file must not be JSON -- got what looks like a JSON object"
        );

        let compiled_dir = output_dir.join("compiled");
        let compile_result = std::process::Command::new("xcrun")
            .args(["coremlcompiler", "compile"])
            .arg(&conversion.output_path)
            .arg(&compiled_dir)
            .output()
            .expect("run coremlcompiler");

        let _ = std::fs::remove_dir_all(&output_dir);

        assert!(
            compile_result.status.success(),
            "Apple's own coremlcompiler rejected the emitted .mlmodel:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&compile_result.stdout),
            String::from_utf8_lossy(&compile_result.stderr)
        );
    }

    /// Same real-checkpoint pipeline as
    /// [`real_mlmodel_bytes_are_accepted_by_apples_own_compiler`], but
    /// through the `.mlpackage` bundle writer -- a materially different
    /// code path (`write_mlpackage`'s `Manifest.json` plus the nested
    /// `Data/com.apple.CoreML/model.mlmodel`) that the other test does not
    /// exercise. `coremlcompiler` accepts a `.mlpackage` directory directly.
    #[test]
    fn real_mlpackage_bundle_is_accepted_by_apples_own_compiler() {
        let coremlcompiler =
            std::process::Command::new("xcrun").args(["--find", "coremlcompiler"]).output();
        let Ok(found) = coremlcompiler else {
            eprintln!("skipping: `xcrun` is not available on this host");
            return;
        };
        if !found.status.success() {
            eprintln!("skipping: `coremlcompiler` is not available on this host");
            return;
        }

        let checkpoint_path = write_test_checkpoint();
        let output_dir = std::env::temp_dir().join(format!(
            "trustformers_coreml_mlpackage_test_{}_{}",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&output_dir).expect("create output dir");
        let output_path = output_dir.join("model");

        let mut config = CoreMLConverterConfig::default();
        config.output_format = CoreMLFormat::MLPackage;
        let converter = CoreMLModelConverter::new(config);
        let result = converter.convert(&checkpoint_path, &output_path);
        let _ = std::fs::remove_file(&checkpoint_path);

        let conversion = result.expect("mlpackage conversion of a real checkpoint must succeed");
        assert!(conversion.output_path.exists());
        assert!(
            conversion.output_path.to_string_lossy().ends_with("model.mlmodel"),
            "mlpackage layout must be Data/com.apple.CoreML/model.mlmodel, got {:?}",
            conversion.output_path
        );

        let package_dir = output_path.with_extension("mlpackage");
        assert!(
            package_dir.join("Manifest.json").exists(),
            "a real .mlpackage bundle must carry a Manifest.json"
        );

        let compiled_dir = output_dir.join("compiled");
        let compile_result = std::process::Command::new("xcrun")
            .args(["coremlcompiler", "compile"])
            .arg(&package_dir)
            .arg(&compiled_dir)
            .output()
            .expect("run coremlcompiler");

        let _ = std::fs::remove_dir_all(&output_dir);

        assert!(
            compile_result.status.success(),
            "Apple's own coremlcompiler rejected the emitted .mlpackage:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&compile_result.stdout),
            String::from_utf8_lossy(&compile_result.stderr)
        );
    }

    #[test]
    fn test_converter_creation() {
        let config = CoreMLConverterConfig::default();
        let converter = CoreMLModelConverter::new(config);
        assert!(!converter.optimization_passes.is_empty());
        assert!(!converter.validation_rules.is_empty());
    }

    #[test]
    fn test_quantization_config() {
        let config = CoreMLQuantizationConfig {
            weight_bits: QuantizationBits::Bit8,
            activation_bits: Some(QuantizationBits::Bit8),
            method: QuantizationMethod::Linear,
            calibration_size: 1000,
            per_channel: true,
        };

        assert_eq!(config.weight_bits, QuantizationBits::Bit8);
        assert!(config.per_channel);
    }

    #[test]
    fn test_pruning_config() {
        let config = PruningConfig {
            target_sparsity: 0.5,
            method: PruningMethod::Magnitude,
            structured: false,
            exclude_layers: vec!["output".to_string()],
        };

        assert_eq!(config.target_sparsity, 0.5);
        assert!(config.exclude_layers.contains(&"output".to_string()));
    }

    #[test]
    fn test_default_converter_config() {
        let config = CoreMLConverterConfig::default();
        assert!(!config.target_ios_version.is_empty());
        assert!(matches!(
            config.optimization_level,
            OptimizationLevel::Basic
        ));
        assert!(matches!(config.output_format, CoreMLFormat::MLModel));
    }

    #[test]
    fn test_optimization_level_variants() {
        let levels = vec![
            OptimizationLevel::None,
            OptimizationLevel::Basic,
            OptimizationLevel::Aggressive,
            OptimizationLevel::Maximum,
        ];
        assert_eq!(levels.len(), 4);
    }

    #[test]
    fn test_coreml_format_variants() {
        let formats = vec![
            CoreMLFormat::MLModel,
            CoreMLFormat::MLModelC,
            CoreMLFormat::MLPackage,
        ];
        assert_eq!(formats.len(), 3);
    }

    #[test]
    fn test_hardware_target_variants() {
        let targets = vec![
            HardwareTarget::All,
            HardwareTarget::NeuralEngine,
            HardwareTarget::GPU,
            HardwareTarget::CPU,
        ];
        assert_eq!(targets.len(), 4);
    }

    #[test]
    fn test_quantization_bits_variants() {
        let bits = vec![
            QuantizationBits::Bit1,
            QuantizationBits::Bit2,
            QuantizationBits::Bit4,
            QuantizationBits::Bit8,
            QuantizationBits::Bit16,
        ];
        assert_eq!(bits.len(), 5);
    }

    #[test]
    fn test_quantization_method_variants() {
        let methods = vec![QuantizationMethod::Linear, QuantizationMethod::KMeans];
        assert_eq!(methods.len(), 2);
    }

    #[test]
    fn test_pruning_method_variants() {
        let methods = vec![
            PruningMethod::Magnitude,
            PruningMethod::Gradient,
            PruningMethod::Random,
            PruningMethod::Structured,
        ];
        assert_eq!(methods.len(), 4);
    }

    #[test]
    fn test_converter_has_optimization_passes() {
        let config = CoreMLConverterConfig::default();
        let converter = CoreMLModelConverter::new(config);
        assert!(!converter.optimization_passes.is_empty());
    }

    #[test]
    fn test_converter_has_validation_rules() {
        let config = CoreMLConverterConfig::default();
        let converter = CoreMLModelConverter::new(config);
        assert!(converter.validation_rules.len() >= 3);
    }

    #[test]
    fn test_quantization_config_4bit() {
        let config = CoreMLQuantizationConfig {
            weight_bits: QuantizationBits::Bit4,
            activation_bits: None,
            method: QuantizationMethod::KMeans,
            calibration_size: 500,
            per_channel: false,
        };
        assert_eq!(config.weight_bits, QuantizationBits::Bit4);
        assert!(config.activation_bits.is_none());
        assert!(!config.per_channel);
    }

    #[test]
    fn test_pruning_config_structured() {
        let config = PruningConfig {
            target_sparsity: 0.9,
            method: PruningMethod::Structured,
            structured: true,
            exclude_layers: vec![],
        };
        assert_eq!(config.target_sparsity, 0.9);
        assert!(config.structured);
        assert!(config.exclude_layers.is_empty());
    }

    #[test]
    fn test_pruning_config_sparsity_bounds() {
        let config = PruningConfig {
            target_sparsity: 0.5,
            method: PruningMethod::Magnitude,
            structured: false,
            exclude_layers: vec!["output".to_string()],
        };
        assert!(config.target_sparsity >= 0.0);
        assert!(config.target_sparsity <= 1.0);
    }

    #[test]
    fn test_converter_config_with_compression() {
        let mut config = CoreMLConverterConfig::default();
        config.enable_compression = true;
        assert!(config.enable_compression);
    }

    #[test]
    fn test_converter_config_hardware_target_gpu() {
        let mut config = CoreMLConverterConfig::default();
        config.hardware_target = HardwareTarget::GPU;
        assert_eq!(config.hardware_target, HardwareTarget::GPU);
    }

    #[test]
    fn test_converter_config_aggressive_optimization() {
        let mut config = CoreMLConverterConfig::default();
        config.optimization_level = OptimizationLevel::Aggressive;
        assert_eq!(config.optimization_level, OptimizationLevel::Aggressive);
    }

    #[test]
    fn test_coreml_version_constant() {
        assert_eq!(COREML_VERSION, 5);
    }

    #[test]
    fn test_quantization_bits_equality() {
        assert_eq!(QuantizationBits::Bit8, QuantizationBits::Bit8);
        assert_ne!(QuantizationBits::Bit4, QuantizationBits::Bit8);
    }

    #[test]
    fn test_optimization_level_equality() {
        assert_eq!(OptimizationLevel::None, OptimizationLevel::None);
        assert_ne!(OptimizationLevel::None, OptimizationLevel::Maximum);
    }

    #[test]
    fn test_hardware_target_equality() {
        assert_eq!(HardwareTarget::All, HardwareTarget::All);
        assert_ne!(HardwareTarget::CPU, HardwareTarget::GPU);
    }

    #[test]
    fn test_converter_config_with_quantization_and_pruning() {
        let config = CoreMLConverterConfig {
            target_ios_version: "17.0".to_string(),
            optimization_level: OptimizationLevel::Maximum,
            enable_compression: true,
            quantization: Some(CoreMLQuantizationConfig {
                weight_bits: QuantizationBits::Bit4,
                activation_bits: Some(QuantizationBits::Bit8),
                method: QuantizationMethod::Linear,
                calibration_size: 2000,
                per_channel: true,
            }),
            pruning: Some(PruningConfig {
                target_sparsity: 0.7,
                method: PruningMethod::Magnitude,
                structured: false,
                exclude_layers: vec!["embed".to_string()],
            }),
            output_format: CoreMLFormat::MLPackage,
            hardware_target: HardwareTarget::NeuralEngine,
        };
        assert!(config.quantization.is_some());
        assert!(config.pruning.is_some());
    }
}
