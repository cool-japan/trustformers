//! LoRA: Low-Rank Adaptation for Large Language Models
//!
//! Implements the LoRA (Hu et al., 2021) parameter-efficient fine-tuning method.
//! LoRA adds low-rank decomposition matrices (A and B) alongside frozen base weights.
//! During training only A and B are updated; at inference the delta can be merged.
//!
//! # Reference
//!
//! Hu et al. 2021: "LoRA: Low-Rank Adaptation of Large Language Models"
//! <https://arxiv.org/abs/2106.09685>
//!
//! # RSLoRA Reference
//!
//! Kalajdzievski 2023: "A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA"
//! <https://arxiv.org/abs/2312.03732>

use std::collections::HashMap;
use std::fmt;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors that can occur in LoRA operations.
#[derive(Debug, Clone, PartialEq)]
pub enum LoraError {
    /// A layer with this name was not found in the LoRA model.
    LayerNotFound(String),
    /// The base weight tensor does not match the expected LoRA delta shape.
    ShapeMismatch {
        /// Name of the layer with the mismatch.
        layer: String,
        /// Expected number of elements.
        expected: usize,
        /// Actual number of elements in the provided weight tensor.
        actual: usize,
    },
    /// Attempted to merge when already merged (or unmerge when not merged).
    InvalidMergeState(String),
    /// A configuration value is out of range or otherwise invalid.
    InvalidConfig(String),
    /// The batch size is inconsistent with the input length.
    BatchSizeMismatch {
        /// Input length in elements.
        input_len: usize,
        /// Batch size provided.
        batch_size: usize,
        /// Expected in_features per token/row.
        in_features: usize,
    },
}

impl fmt::Display for LoraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoraError::LayerNotFound(name) => {
                write!(f, "LoRA layer '{}' not found", name)
            }
            LoraError::ShapeMismatch { layer, expected, actual } => {
                write!(
                    f,
                    "Shape mismatch for layer '{}': expected {} elements, got {}",
                    layer, expected, actual
                )
            }
            LoraError::InvalidMergeState(msg) => {
                write!(f, "Invalid merge state: {}", msg)
            }
            LoraError::InvalidConfig(msg) => {
                write!(f, "Invalid LoRA configuration: {}", msg)
            }
            LoraError::BatchSizeMismatch { input_len, batch_size, in_features } => {
                write!(
                    f,
                    "Batch size mismatch: input length {} with batch_size {} and \
                     in_features {} is inconsistent (expected {} elements)",
                    input_len, batch_size, in_features, batch_size * in_features
                )
            }
        }
    }
}

impl std::error::Error for LoraError {}

// ─── Enums ───────────────────────────────────────────────────────────────────

/// Which bias parameters to include when applying LoRA.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoraBias {
    /// No bias parameters are trainable.
    None,
    /// All bias parameters (base + LoRA layers) are trainable.
    All,
    /// Only the LoRA-layer bias parameters are trainable.
    LoraOnly,
}

impl Default for LoraBias {
    fn default() -> Self {
        LoraBias::None
    }
}

/// The downstream task type that LoRA is being applied to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoraTaskType {
    /// Causal language modelling (auto-regressive generation).
    CausalLm,
    /// Sequence classification.
    SeqCls,
    /// Token classification (NER, POS, etc.).
    TokenCls,
    /// Feature extraction (embedding models, encoders).
    FeatureExtraction,
}

impl Default for LoraTaskType {
    fn default() -> Self {
        LoraTaskType::CausalLm
    }
}

/// Initialisation method for the LoRA A matrix.
///
/// The B matrix is always initialised to zeros so that the merged model
/// is identical to the base model at the start of training.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoraInitMethod {
    /// Sample A from N(0, σ²) with σ = 1/√in_features (standard Gaussian).
    Gaussian,
    /// Kaiming uniform initialisation: A ~ Uniform(-√(1/fan_in), √(1/fan_in)).
    KaimingUniform,
    /// Initialise A to zeros (used for ablation studies).
    Zero,
}

impl Default for LoraInitMethod {
    fn default() -> Self {
        LoraInitMethod::KaimingUniform
    }
}

// ─── LoraConfig ───────────────────────────────────────────────────────────────

/// Configuration for LoRA fine-tuning.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// LoRA rank — controls the expressiveness of the low-rank update.
    pub rank: usize,
    /// LoRA scaling factor α.  Effective scale = α / rank (or α / √rank for rsLoRA).
    pub alpha: f32,
    /// Dropout probability applied to the input before the LoRA B projection.
    pub dropout: f32,
    /// Names of the modules (layers) to which LoRA adaptors are attached.
    pub target_modules: Vec<String>,
    /// Which bias parameters to make trainable.
    pub bias: LoraBias,
    /// Downstream task type.
    pub task_type: LoraTaskType,
    /// Initialisation method for the A matrix.
    pub init_lora_weights: LoraInitMethod,
    /// When `true`, use rank-stabilized LoRA: scale = α / √rank instead of α / rank.
    pub use_rslora: bool,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            dropout: 0.0,
            target_modules: vec!["q_proj".to_string(), "v_proj".to_string()],
            bias: LoraBias::default(),
            task_type: LoraTaskType::default(),
            init_lora_weights: LoraInitMethod::default(),
            use_rslora: false,
        }
    }
}

impl LoraConfig {
    /// Compute the effective scaling factor based on the current configuration.
    ///
    /// - Standard LoRA: `alpha / rank`
    /// - rsLoRA: `alpha / sqrt(rank)`
    pub fn effective_scaling(&self) -> f32 {
        if self.use_rslora {
            self.alpha / (self.rank as f32).sqrt()
        } else {
            self.alpha / self.rank as f32
        }
    }
}

// ─── Pseudo-random helpers (FNV-1a based) ────────────────────────────────────

/// FNV-1a 64-bit hash of a u64 value — used as a cheap deterministic PRNG.
#[inline]
fn fnv1a_u64(value: u64) -> u64 {
    const FNV_OFFSET: u64 = 14_695_981_039_346_656_037;
    const FNV_PRIME: u64 = 1_099_511_628_211;
    let mut hash = FNV_OFFSET;
    let bytes = value.to_le_bytes();
    for &byte in &bytes {
        hash ^= u64(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Expand a single u64 to a `u64` via FNV — avoids the trait issue.
#[inline]
fn u64(x: u8) -> u64 {
    x as u64
}

/// Generate a deterministic pseudo-random f32 in (-1, 1) from a seed and index.
#[inline]
fn pseudo_rand_f32(seed: u64, index: usize) -> f32 {
    let h = fnv1a_u64(seed ^ fnv1a_u64(index as u64));
    // Map to [0, 1)
    let unit = (h >> 11) as f32 / (1u64 << 53) as f32;
    // Map to (-1, 1)
    unit * 2.0 - 1.0
}

/// Generate a deterministic pseudo-random f32 in [0, 1) from a seed and index.
///
/// Used in tests to verify range properties of the PRNG.
#[allow(dead_code)]
#[inline]
fn pseudo_rand_unit(seed: u64, index: usize) -> f32 {
    let h = fnv1a_u64(seed ^ fnv1a_u64(index as u64 + 0xDEAD_BEEF));
    (h >> 11) as f32 / (1u64 << 53) as f32
}

// ─── LoraLayer ────────────────────────────────────────────────────────────────

/// A single LoRA adaptor layer.
///
/// Stores:
/// - `lora_a`: shape `[in_features, rank]` (row-major, row = input dimension)
/// - `lora_b`: shape `[rank, out_features]` (row-major, row = rank dimension)
///
/// The delta applied to the base output is:  `(x @ lora_a) @ lora_b * scaling`
#[derive(Debug, Clone)]
pub struct LoraLayer {
    /// A matrix, stored row-major with shape [in_features × rank].
    pub lora_a: Vec<f32>,
    /// B matrix, stored row-major with shape [rank × out_features].
    pub lora_b: Vec<f32>,
    /// Effective scaling factor (alpha/rank or alpha/sqrt(rank)).
    pub scaling: f32,
    /// Input feature dimension.
    pub in_features: usize,
    /// Output feature dimension.
    pub out_features: usize,
    /// LoRA rank.
    pub rank: usize,
}

impl LoraLayer {
    /// Create a new LoRA layer with the given dimensions and configuration.
    ///
    /// A is initialised according to `config.init_lora_weights`;
    /// B is always initialised to zeros.
    /// The seed for deterministic initialisation is derived from the dimensions.
    pub fn new(in_features: usize, out_features: usize, config: &LoraConfig) -> Self {
        let rank = config.rank;
        let scaling = config.effective_scaling();

        // Deterministic seed based on dimensions and rank.
        let seed: u64 = (in_features as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (out_features as u64).wrapping_mul(0x6C62_272E_07BB_0142)
            ^ (rank as u64).wrapping_mul(0xBEA2_25AF_9267_9A57);

        let a_size = in_features * rank;
        let lora_a = match config.init_lora_weights {
            LoraInitMethod::Gaussian => {
                // σ = 1 / √in_features  (lecun normal style)
                let std_dev = 1.0 / (in_features as f32).sqrt();
                (0..a_size)
                    .map(|i| pseudo_rand_f32(seed, i) * std_dev)
                    .collect()
            }
            LoraInitMethod::KaimingUniform => {
                // Kaiming uniform: bound = √(1 / fan_in)
                let bound = 1.0 / (in_features as f32).sqrt();
                (0..a_size)
                    .map(|i| pseudo_rand_f32(seed, i) * bound)
                    .collect()
            }
            LoraInitMethod::Zero => vec![0.0f32; a_size],
        };

        let lora_b = vec![0.0f32; rank * out_features];

        Self {
            lora_a,
            lora_b,
            scaling,
            in_features,
            out_features,
            rank,
        }
    }

    /// Compute the LoRA delta for a batch of inputs.
    ///
    /// # Arguments
    ///
    /// * `x` — Input activations, shape `[batch_size, in_features]` row-major.
    /// * `batch_size` — Number of rows in `x`.
    ///
    /// # Returns
    ///
    /// Delta activations to add to the base layer output,
    /// shape `[batch_size, out_features]`.
    pub fn forward(&self, x: &[f32], batch_size: usize) -> Vec<f32> {
        // x: [batch_size, in_features]
        // lora_a: [in_features, rank]
        // intermediate: [batch_size, rank]
        // lora_b: [rank, out_features]
        // output: [batch_size, out_features]

        let intermediate_size = batch_size * self.rank;
        let mut intermediate = vec![0.0f32; intermediate_size];

        // x @ lora_a  →  [batch_size, rank]
        for b in 0..batch_size {
            for r in 0..self.rank {
                let mut acc = 0.0f32;
                for k in 0..self.in_features {
                    acc += x[b * self.in_features + k] * self.lora_a[k * self.rank + r];
                }
                intermediate[b * self.rank + r] = acc;
            }
        }

        // intermediate @ lora_b  →  [batch_size, out_features]
        let mut output = vec![0.0f32; batch_size * self.out_features];
        for b in 0..batch_size {
            for o in 0..self.out_features {
                let mut acc = 0.0f32;
                for r in 0..self.rank {
                    acc += intermediate[b * self.rank + r] * self.lora_b[r * self.out_features + o];
                }
                output[b * self.out_features + o] = acc * self.scaling;
            }
        }

        output
    }

    /// Compute the merged weight delta: `lora_a @ lora_b * scaling`.
    ///
    /// # Returns
    ///
    /// Weight delta of shape `[in_features, out_features]`, row-major.
    pub fn get_delta_weight(&self) -> Vec<f32> {
        // lora_a: [in_features, rank]
        // lora_b: [rank, out_features]
        // result: [in_features, out_features]
        let mut delta = vec![0.0f32; self.in_features * self.out_features];
        for i in 0..self.in_features {
            for o in 0..self.out_features {
                let mut acc = 0.0f32;
                for r in 0..self.rank {
                    acc += self.lora_a[i * self.rank + r] * self.lora_b[r * self.out_features + o];
                }
                delta[i * self.out_features + o] = acc * self.scaling;
            }
        }
        delta
    }

    /// Add the LoRA delta weight into a base weight tensor in place.
    ///
    /// `base_weight` must have length `in_features * out_features`.
    pub fn merge_into_weight(&self, base_weight: &mut [f32]) {
        let delta = self.get_delta_weight();
        for (w, d) in base_weight.iter_mut().zip(delta.iter()) {
            *w += d;
        }
    }

    /// Subtract the LoRA delta weight from a base weight tensor in place.
    ///
    /// `base_weight` must have length `in_features * out_features`.
    pub fn unmerge_from_weight(&self, base_weight: &mut [f32]) {
        let delta = self.get_delta_weight();
        for (w, d) in base_weight.iter_mut().zip(delta.iter()) {
            *w -= d;
        }
    }

    /// Number of trainable parameters in this layer: rank*in + rank*out.
    pub fn trainable_params(&self) -> usize {
        self.rank * self.in_features + self.rank * self.out_features
    }
}

// ─── LoraModel ────────────────────────────────────────────────────────────────

/// Tracks all LoRA adaptors for a model and orchestrates merge/unmerge.
#[derive(Debug)]
pub struct LoraModel {
    /// Map from layer name to the LoRA adaptor for that layer.
    pub layer_configs: HashMap<String, LoraLayer>,
    /// Global LoRA configuration.
    pub config: LoraConfig,
    /// Whether the LoRA deltas are currently merged into the base weights.
    pub is_merged: bool,
}

impl LoraModel {
    /// Create a new, empty LoRA model with the given configuration.
    pub fn new(config: LoraConfig) -> Self {
        Self {
            layer_configs: HashMap::new(),
            config,
            is_merged: false,
        }
    }

    /// Register a new LoRA adaptor for the named layer.
    ///
    /// If the layer name is not in `config.target_modules`, it is silently skipped
    /// so callers can iterate all layer names without filtering themselves.
    pub fn add_layer(&mut self, name: String, in_features: usize, out_features: usize) {
        if self.config.target_modules.contains(&name) {
            let layer = LoraLayer::new(in_features, out_features, &self.config);
            self.layer_configs.insert(name, layer);
        }
    }

    /// Compute `base_output + lora_delta` for the named layer.
    ///
    /// If `name` has no registered LoRA adaptor, returns a clone of `base_output`.
    pub fn forward_with_lora(
        &self,
        name: &str,
        base_output: &[f32],
        x: &[f32],
        batch_size: usize,
    ) -> Vec<f32> {
        match self.layer_configs.get(name) {
            Some(layer) => {
                let delta = layer.forward(x, batch_size);
                base_output
                    .iter()
                    .zip(delta.iter())
                    .map(|(b, d)| b + d)
                    .collect()
            }
            None => base_output.to_vec(),
        }
    }

    /// Merge all LoRA deltas into the provided base weight map.
    ///
    /// # Errors
    ///
    /// Returns [`LoraError::InvalidMergeState`] if already merged.
    /// Returns [`LoraError::LayerNotFound`] if a LoRA layer has no corresponding base weight.
    /// Returns [`LoraError::ShapeMismatch`] if the base weight shape is unexpected.
    pub fn merge_weights(
        &mut self,
        base_weights: &mut HashMap<String, Vec<f32>>,
    ) -> Result<(), LoraError> {
        if self.is_merged {
            return Err(LoraError::InvalidMergeState(
                "weights are already merged; call unmerge_weights first".to_string(),
            ));
        }

        for (name, layer) in &self.layer_configs {
            let base = base_weights
                .get_mut(name)
                .ok_or_else(|| LoraError::LayerNotFound(name.clone()))?;

            let expected = layer.in_features * layer.out_features;
            if base.len() != expected {
                return Err(LoraError::ShapeMismatch {
                    layer: name.clone(),
                    expected,
                    actual: base.len(),
                });
            }

            layer.merge_into_weight(base);
        }

        self.is_merged = true;
        Ok(())
    }

    /// Unmerge all LoRA deltas from the provided base weight map.
    ///
    /// # Errors
    ///
    /// Returns [`LoraError::InvalidMergeState`] if not currently merged.
    /// Returns [`LoraError::LayerNotFound`] if a LoRA layer has no corresponding base weight.
    /// Returns [`LoraError::ShapeMismatch`] if the base weight shape is unexpected.
    pub fn unmerge_weights(
        &mut self,
        base_weights: &mut HashMap<String, Vec<f32>>,
    ) -> Result<(), LoraError> {
        if !self.is_merged {
            return Err(LoraError::InvalidMergeState(
                "weights are not merged; call merge_weights first".to_string(),
            ));
        }

        for (name, layer) in &self.layer_configs {
            let base = base_weights
                .get_mut(name)
                .ok_or_else(|| LoraError::LayerNotFound(name.clone()))?;

            let expected = layer.in_features * layer.out_features;
            if base.len() != expected {
                return Err(LoraError::ShapeMismatch {
                    layer: name.clone(),
                    expected,
                    actual: base.len(),
                });
            }

            layer.unmerge_from_weight(base);
        }

        self.is_merged = false;
        Ok(())
    }

    /// Count the total number of trainable parameters across all LoRA adaptors.
    ///
    /// Each adaptor contributes `rank * in_features + rank * out_features`.
    pub fn trainable_parameter_count(&self) -> usize {
        self.layer_configs.values().map(|l| l.trainable_params()).sum()
    }

    /// Count the total number of base model parameters from the provided dimension list.
    ///
    /// `base_sizes` is a slice of `(in_features, out_features)` pairs, one per base layer.
    pub fn total_base_parameter_count(&self, base_sizes: &[(usize, usize)]) -> usize {
        base_sizes.iter().map(|(i, o)| i * o).sum()
    }
}

// ─── LoRA+ ────────────────────────────────────────────────────────────────────

/// Configuration for LoRA+ (Hayou et al., 2024), which uses different learning
/// rates for the A and B matrices.
///
/// The key insight: the B matrix update should use a higher learning rate than
/// the A matrix.  Empirically, `lr_B = lr_B_multiplier * lr_A` works well with
/// the default multiplier of 16.0.
///
/// Reference: "LoRA+: Efficient Low-Rank Adaptation of Large Models"
/// <https://arxiv.org/abs/2402.12354>
#[derive(Debug, Clone)]
pub struct LoraPlusConfig {
    /// LoRA rank.
    pub rank: usize,
    /// LoRA alpha scaling factor.
    pub alpha: f32,
    /// Dropout probability.
    pub dropout: f32,
    /// Multiplier applied to the base learning rate to obtain the B-matrix lr.
    /// Default: 16.0 (as recommended in the LoRA+ paper).
    pub learning_rate_b_multiplier: f32,
}

impl Default for LoraPlusConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            dropout: 0.0,
            learning_rate_b_multiplier: 16.0,
        }
    }
}

impl LoraPlusConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<(), LoraError> {
        if self.rank == 0 {
            return Err(LoraError::InvalidConfig("rank must be > 0".to_string()));
        }
        if self.learning_rate_b_multiplier <= 0.0 {
            return Err(LoraError::InvalidConfig(
                "learning_rate_b_multiplier must be > 0".to_string(),
            ));
        }
        if self.dropout < 0.0 || self.dropout >= 1.0 {
            return Err(LoraError::InvalidConfig(
                "dropout must be in [0, 1)".to_string(),
            ));
        }
        Ok(())
    }
}

/// Compute LoRA+ effective learning rates for the A and B matrices.
///
/// Returns `(lr_A, lr_B)` where `lr_B = lr_B_multiplier * base_lr` and
/// `lr_A = base_lr`.
///
/// The asymmetric learning rate is the core idea of LoRA+: the B matrix
/// should converge faster (higher lr) because it directly modulates the
/// output magnitude, while A acts more like a feature extractor.
///
/// # Errors
///
/// Returns `LoraError::InvalidConfig` if the configuration is invalid.
pub fn lora_plus_lr_schedule(
    base_lr: f32,
    config: &LoraPlusConfig,
) -> Result<(f32, f32), LoraError> {
    config.validate()?;
    if base_lr <= 0.0 {
        return Err(LoraError::InvalidConfig(format!(
            "base_lr must be > 0, got {}",
            base_lr
        )));
    }
    let lr_a = base_lr;
    let lr_b = base_lr * config.learning_rate_b_multiplier;
    Ok((lr_a, lr_b))
}

// ─── DoRA ─────────────────────────────────────────────────────────────────────

/// DoRA: Weight-Decomposed Low-Rank Adaptation (Liu et al., 2024).
///
/// DoRA decomposes a weight matrix `W` into a magnitude component (column-wise
/// L2 norms) and a directional component (column-wise normalized matrix), then
/// applies a LoRA delta to the directional component.
///
/// Forward pass:
/// ```text
/// V_new = (V + ΔW) / ||V + ΔW||_c   (column-wise normalised)
/// y = m ⊙ V_new @ x
/// ```
///
/// where `m` is the learnable magnitude vector, `ΔW = B @ A` is the LoRA delta
/// (shape `[d_out, d_in]`), and `||·||_c` denotes column-wise L2 normalisation.
///
/// Reference: "DoRA: Weight-Decomposed Low-Rank Adaptation"
/// <https://arxiv.org/abs/2402.09353>
#[derive(Debug, Clone)]
pub struct DoraLayer {
    /// Magnitude vector: column-wise L2 norm of the original weight `W`.
    /// Shape: `[d_out]` (one scalar per output neuron).
    pub magnitude: Vec<f32>,
    /// Directional component: `W / ||W||_c`, stored row-major `[d_out, d_in]`.
    pub direction: Vec<f32>,
    /// LoRA A matrix, shape `[rank, d_in]` (row-major).
    pub lora_a: Vec<f32>,
    /// LoRA B matrix, shape `[d_out, rank]` (row-major).
    pub lora_b: Vec<f32>,
    /// LoRA rank.
    pub rank: usize,
    /// Input dimension.
    pub d_in: usize,
    /// Output dimension.
    pub d_out: usize,
    /// LoRA scaling factor (alpha / rank).
    pub scaling: f32,
}

impl DoraLayer {
    /// Initialise a DoRA layer from a pre-trained weight matrix.
    ///
    /// `weight` is expected in row-major order with shape `[d_out, d_in]`.
    /// The LoRA B matrix is initialised to zero so the initial DoRA forward
    /// pass is identical to the base model.
    ///
    /// # Errors
    ///
    /// Returns `LoraError::ShapeMismatch` if `weight.len() != d_out * d_in`.
    /// Returns `LoraError::InvalidConfig` if `rank == 0` or dimensions are zero.
    pub fn from_pretrained(
        weight: &[f32],
        d_out: usize,
        d_in: usize,
        rank: usize,
        alpha: f32,
    ) -> Result<Self, LoraError> {
        if rank == 0 {
            return Err(LoraError::InvalidConfig("rank must be > 0".to_string()));
        }
        if d_in == 0 || d_out == 0 {
            return Err(LoraError::InvalidConfig(
                "d_in and d_out must be > 0".to_string(),
            ));
        }
        let expected = d_out * d_in;
        if weight.len() != expected {
            return Err(LoraError::ShapeMismatch {
                layer: "dora".to_string(),
                expected,
                actual: weight.len(),
            });
        }

        // Compute column-wise L2 norms (each "column" = row in row-major d_out×d_in)
        // Here we treat columns as the d_out dimension: for each output neuron i,
        // norm_i = ||W[i, :]||_2 (L2 norm across all d_in inputs for that output).
        let magnitude: Vec<f32> = (0..d_out)
            .map(|i| {
                let row_sq: f32 = (0..d_in)
                    .map(|j| {
                        let v = weight[i * d_in + j];
                        v * v
                    })
                    .sum();
                row_sq.sqrt().max(1e-8)
            })
            .collect();

        // Normalise each row by its L2 norm to get the directional component
        let direction: Vec<f32> = (0..d_out)
            .flat_map(|i| {
                let m = magnitude[i];
                (0..d_in).map(move |j| weight[i * d_in + j] / m)
            })
            .collect();

        // A: Kaiming-uniform init, shape [rank, d_in]
        let a_seed: u64 = (d_in as u64).wrapping_mul(0x9E37_79B9)
            ^ (d_out as u64).wrapping_mul(0x6C62_272E)
            ^ (rank as u64).wrapping_mul(0xBEA2_25AF);
        let bound = 1.0 / (d_in as f32).sqrt();
        let lora_a: Vec<f32> = (0..rank * d_in)
            .map(|idx| pseudo_rand_f32(a_seed, idx) * bound)
            .collect();

        // B: zeros, shape [d_out, rank]
        let lora_b = vec![0.0f32; d_out * rank];

        let scaling = alpha / rank as f32;

        Ok(Self {
            magnitude,
            direction,
            lora_a,
            lora_b,
            rank,
            d_in,
            d_out,
            scaling,
        })
    }

    /// Compute `ΔW = scaling * lora_b @ lora_a`, shape `[d_out, d_in]`.
    fn delta_weight(&self) -> Vec<f32> {
        // lora_b: [d_out, rank], lora_a: [rank, d_in]
        // result: [d_out, d_in]
        let mut dw = vec![0.0f32; self.d_out * self.d_in];
        for i in 0..self.d_out {
            for j in 0..self.d_in {
                let mut acc = 0.0f32;
                for r in 0..self.rank {
                    acc += self.lora_b[i * self.rank + r] * self.lora_a[r * self.d_in + j];
                }
                dw[i * self.d_in + j] = acc * self.scaling;
            }
        }
        dw
    }

    /// Column-wise normalise a `[d_out, d_in]` matrix (normalise each row).
    fn col_normalise(mat: &[f32], d_out: usize, d_in: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; d_out * d_in];
        for i in 0..d_out {
            let row_sq: f32 = (0..d_in).map(|j| mat[i * d_in + j].powi(2)).sum();
            let norm = row_sq.sqrt().max(1e-8);
            for j in 0..d_in {
                out[i * d_in + j] = mat[i * d_in + j] / norm;
            }
        }
        out
    }

    /// Forward pass for a single input vector `x` of length `d_in`.
    ///
    /// Computes:
    /// 1. `V_hat = direction + ΔW`         (additive update in direction space)
    /// 2. `V_norm = V_hat / ||V_hat||_c`   (re-normalise columns)
    /// 3. `y[i] = magnitude[i] * (V_norm[i, :] · x)`
    ///
    /// # Errors
    ///
    /// Returns `LoraError::BatchSizeMismatch` if `input.len() != d_in`.
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>, LoraError> {
        if input.len() != self.d_in {
            return Err(LoraError::BatchSizeMismatch {
                input_len: input.len(),
                batch_size: 1,
                in_features: self.d_in,
            });
        }

        let dw = self.delta_weight();

        // V_hat = direction + ΔW
        let v_hat: Vec<f32> = self.direction.iter().zip(dw.iter())
            .map(|(&d, &delta)| d + delta)
            .collect();

        // V_norm = column-normalise V_hat
        let v_norm = Self::col_normalise(&v_hat, self.d_out, self.d_in);

        // y[i] = magnitude[i] * dot(V_norm[i, :], input)
        let mut output = vec![0.0f32; self.d_out];
        for i in 0..self.d_out {
            let dot: f32 = (0..self.d_in)
                .map(|j| v_norm[i * self.d_in + j] * input[j])
                .sum();
            output[i] = self.magnitude[i] * dot;
        }

        Ok(output)
    }

    /// Merge the DoRA layer back into a full weight matrix.
    ///
    /// Returns the effective weight `W_eff = magnitude ⊙ V_norm` where
    /// `V_norm = (direction + ΔW) / ||direction + ΔW||_c`.
    ///
    /// Shape: `[d_out, d_in]` row-major.
    pub fn merge_weights(&self) -> Result<Vec<f32>, LoraError> {
        let dw = self.delta_weight();

        // V_hat = direction + ΔW
        let v_hat: Vec<f32> = self.direction.iter().zip(dw.iter())
            .map(|(&d, &delta)| d + delta)
            .collect();

        // V_norm = column-normalise
        let v_norm = Self::col_normalise(&v_hat, self.d_out, self.d_in);

        // W_eff[i, j] = magnitude[i] * v_norm[i, j]
        let mut w_eff = vec![0.0f32; self.d_out * self.d_in];
        for i in 0..self.d_out {
            let m = self.magnitude[i];
            for j in 0..self.d_in {
                w_eff[i * self.d_in + j] = m * v_norm[i * self.d_in + j];
            }
        }

        Ok(w_eff)
    }

    /// Total number of trainable parameters in this DoRA layer.
    ///
    /// = `d_out` (magnitude) + `rank * d_in` (lora_a) + `d_out * rank` (lora_b)
    pub fn trainable_params(&self) -> usize {
        self.d_out + self.rank * self.d_in + self.d_out * self.rank
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1. Config defaults ────────────────────────────────────────────────────
    #[test]
    fn test_config_defaults() {
        let cfg = LoraConfig::default();
        assert_eq!(cfg.rank, 16);
        assert!((cfg.alpha - 32.0).abs() < 1e-6);
        assert!((cfg.dropout - 0.0).abs() < 1e-6);
        assert_eq!(cfg.target_modules, vec!["q_proj", "v_proj"]);
        assert_eq!(cfg.bias, LoraBias::None);
        assert_eq!(cfg.task_type, LoraTaskType::CausalLm);
        assert_eq!(cfg.init_lora_weights, LoraInitMethod::KaimingUniform);
        assert!(!cfg.use_rslora);
    }

    // ── 2. Standard scaling = alpha / rank ────────────────────────────────────
    #[test]
    fn test_standard_scaling() {
        let mut cfg = LoraConfig::default();
        cfg.rank = 16;
        cfg.alpha = 32.0;
        cfg.use_rslora = false;
        // scale = 32 / 16 = 2.0
        assert!((cfg.effective_scaling() - 2.0).abs() < 1e-6);
    }

    // ── 3. rsLoRA scaling = alpha / sqrt(rank) ────────────────────────────────
    #[test]
    fn test_rslora_scaling() {
        let mut cfg = LoraConfig::default();
        cfg.rank = 16;
        cfg.alpha = 32.0;
        cfg.use_rslora = true;
        // scale = 32 / sqrt(16) = 32 / 4 = 8.0
        assert!((cfg.effective_scaling() - 8.0).abs() < 1e-6);
    }

    // ── 4. LoraLayer: check shape (forward output length) ────────────────────
    #[test]
    fn test_lora_layer_forward_shape() {
        let cfg = LoraConfig { rank: 4, alpha: 8.0, ..Default::default() };
        let layer = LoraLayer::new(8, 16, &cfg);
        let x = vec![0.5f32; 3 * 8]; // batch=3, in=8
        let out = layer.forward(&x, 3);
        assert_eq!(out.len(), 3 * 16, "output should be [batch, out_features]");
    }

    // ── 5. Delta weight shape ─────────────────────────────────────────────────
    #[test]
    fn test_delta_weight_shape() {
        let cfg = LoraConfig { rank: 4, alpha: 8.0, ..Default::default() };
        let layer = LoraLayer::new(12, 24, &cfg);
        let delta = layer.get_delta_weight();
        assert_eq!(delta.len(), 12 * 24, "delta shape should be [in_features, out_features]");
    }

    // ── 6. lora_b = 0 → forward output is all zeros at init ──────────────────
    #[test]
    fn test_forward_zero_at_init() {
        let cfg = LoraConfig {
            rank: 4,
            alpha: 8.0,
            init_lora_weights: LoraInitMethod::KaimingUniform,
            ..Default::default()
        };
        let layer = LoraLayer::new(8, 16, &cfg);
        // lora_b is zeros, so regardless of lora_a, output is zero
        let x = vec![1.0f32; 2 * 8];
        let out = layer.forward(&x, 2);
        for &v in &out {
            assert!(v.abs() < 1e-7, "Initial forward pass must be zero (lora_b=0)");
        }
    }

    // ── 7. Merge / unmerge round-trip ─────────────────────────────────────────
    #[test]
    fn test_merge_unmerge_round_trip() {
        let mut cfg = LoraConfig::default();
        cfg.rank = 2;
        cfg.alpha = 4.0;
        cfg.target_modules = vec!["linear".to_string()];
        cfg.init_lora_weights = LoraInitMethod::Gaussian;

        let in_f = 4usize;
        let out_f = 4usize;

        // Give lora_b non-zero values to get a non-trivial delta
        let mut layer = LoraLayer::new(in_f, out_f, &cfg);
        for (i, v) in layer.lora_b.iter_mut().enumerate() {
            *v = 0.1 * (i as f32 + 1.0);
        }

        let original_weights: Vec<f32> = (0..in_f * out_f).map(|i| i as f32 * 0.5).collect();
        let mut base_weights: HashMap<String, Vec<f32>> = HashMap::new();
        base_weights.insert("linear".to_string(), original_weights.clone());

        let mut model = LoraModel {
            layer_configs: {
                let mut m = HashMap::new();
                m.insert("linear".to_string(), layer);
                m
            },
            config: cfg,
            is_merged: false,
        };

        // Merge
        model.merge_weights(&mut base_weights).expect("merge should succeed");
        assert!(model.is_merged);

        // Weights should differ after merge
        let merged = base_weights["linear"].clone();
        let any_diff = merged.iter().zip(original_weights.iter()).any(|(a, b)| (a - b).abs() > 1e-7);
        assert!(any_diff, "merged weights should differ from originals");

        // Unmerge
        model.unmerge_weights(&mut base_weights).expect("unmerge should succeed");
        assert!(!model.is_merged);

        // Weights should be restored to original
        let restored = &base_weights["linear"];
        for (r, o) in restored.iter().zip(original_weights.iter()) {
            assert!((r - o).abs() < 1e-5, "unmerge should restore original weights, got {} expected {}", r, o);
        }
    }

    // ── 8. Trainable parameter count ──────────────────────────────────────────
    #[test]
    fn test_trainable_parameter_count() {
        let mut cfg = LoraConfig::default();
        cfg.rank = 4;
        cfg.target_modules = vec!["q_proj".to_string(), "v_proj".to_string()];

        let mut model = LoraModel::new(cfg);
        model.add_layer("q_proj".to_string(), 64, 64);
        model.add_layer("v_proj".to_string(), 64, 32);

        // q_proj: rank*(in+out) = 4*(64+64) = 512
        // v_proj: rank*(in+out) = 4*(64+32) = 384
        let expected = 512 + 384;
        assert_eq!(model.trainable_parameter_count(), expected);
    }

    // ── 9. Target module filtering ────────────────────────────────────────────
    #[test]
    fn test_target_module_filtering() {
        let mut cfg = LoraConfig::default();
        cfg.target_modules = vec!["q_proj".to_string()]; // only q_proj

        let mut model = LoraModel::new(cfg);
        model.add_layer("q_proj".to_string(), 8, 8);
        model.add_layer("k_proj".to_string(), 8, 8); // not in target_modules
        model.add_layer("v_proj".to_string(), 8, 8); // not in target_modules

        assert_eq!(model.layer_configs.len(), 1);
        assert!(model.layer_configs.contains_key("q_proj"));
        assert!(!model.layer_configs.contains_key("k_proj"));
        assert!(!model.layer_configs.contains_key("v_proj"));
    }

    // ── 10. Gaussian vs KaimingUniform vs Zero init ───────────────────────────
    #[test]
    fn test_init_methods() {
        let mut cfg = LoraConfig {
            rank: 4,
            alpha: 4.0,
            ..Default::default()
        };

        // Gaussian: lora_a should be non-zero
        cfg.init_lora_weights = LoraInitMethod::Gaussian;
        let gaussian_layer = LoraLayer::new(16, 16, &cfg);
        assert!(gaussian_layer.lora_a.iter().any(|&v| v.abs() > 1e-7));
        // lora_b always zero
        assert!(gaussian_layer.lora_b.iter().all(|&v| v == 0.0));

        // KaimingUniform: lora_a should be non-zero
        cfg.init_lora_weights = LoraInitMethod::KaimingUniform;
        let kaiming_layer = LoraLayer::new(16, 16, &cfg);
        assert!(kaiming_layer.lora_a.iter().any(|&v| v.abs() > 1e-7));

        // Zero: lora_a should be all zero
        cfg.init_lora_weights = LoraInitMethod::Zero;
        let zero_layer = LoraLayer::new(16, 16, &cfg);
        assert!(zero_layer.lora_a.iter().all(|&v| v == 0.0));
    }

    // ── 11. Batch forward: batch dimension correctly handled ──────────────────
    #[test]
    fn test_batch_forward() {
        let mut cfg = LoraConfig { rank: 2, alpha: 2.0, ..Default::default() };
        cfg.init_lora_weights = LoraInitMethod::Zero;

        let mut layer = LoraLayer::new(4, 6, &cfg);
        // Set non-trivial lora_a and lora_b
        layer.lora_a = vec![1.0; 4 * 2]; // all ones: [4, 2]
        layer.lora_b = vec![1.0; 2 * 6]; // all ones: [2, 6]

        // Input: 2 samples, each [1,2,3,4] → shape [2, 4]
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
        let out = layer.forward(&x, 2);

        // x @ lora_a: each row [1,2,3,4] × [[1,1],[1,1],[1,1],[1,1]] → [10,10]
        // [10,10] @ lora_b: [10,10] × [[1,1,1,1,1,1],[1,1,1,1,1,1]] → [20,20,20,20,20,20]
        // scale = alpha/rank = 2/2 = 1.0
        // Expected: each batch row = 20.0 for all 6 outputs
        assert_eq!(out.len(), 2 * 6);
        for &v in &out {
            assert!((v - 20.0).abs() < 1e-5, "expected 20.0 got {}", v);
        }
    }

    // ── 12. Error: double merge ────────────────────────────────────────────────
    #[test]
    fn test_error_double_merge() {
        let mut cfg = LoraConfig::default();
        cfg.target_modules = vec!["w".to_string()];
        cfg.rank = 2;

        let mut model = LoraModel::new(cfg);
        model.add_layer("w".to_string(), 4, 4);

        let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
        weights.insert("w".to_string(), vec![0.0f32; 16]);

        model.merge_weights(&mut weights).expect("first merge ok");
        let err = model.merge_weights(&mut weights).expect_err("double merge should error");
        assert!(matches!(err, LoraError::InvalidMergeState(_)));
    }

    // ── 13. Error: unmerge when not merged ────────────────────────────────────
    #[test]
    fn test_error_unmerge_not_merged() {
        let mut cfg = LoraConfig::default();
        cfg.target_modules = vec!["w".to_string()];
        cfg.rank = 2;

        let mut model = LoraModel::new(cfg);
        model.add_layer("w".to_string(), 4, 4);

        let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
        weights.insert("w".to_string(), vec![0.0f32; 16]);

        let err = model.unmerge_weights(&mut weights).expect_err("unmerge without merge should error");
        assert!(matches!(err, LoraError::InvalidMergeState(_)));
    }

    // ── 14. Error: shape mismatch on merge ────────────────────────────────────
    #[test]
    fn test_error_shape_mismatch_on_merge() {
        let mut cfg = LoraConfig::default();
        cfg.target_modules = vec!["w".to_string()];
        cfg.rank = 2;

        let mut model = LoraModel::new(cfg);
        model.add_layer("w".to_string(), 4, 4); // expects 4*4=16 elements

        let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
        weights.insert("w".to_string(), vec![0.0f32; 8]); // wrong size

        let err = model.merge_weights(&mut weights).expect_err("shape mismatch should error");
        assert!(matches!(err, LoraError::ShapeMismatch { .. }));
    }

    // ── 15. total_base_parameter_count ────────────────────────────────────────
    #[test]
    fn test_total_base_parameter_count() {
        let model = LoraModel::new(LoraConfig::default());
        let sizes = [(64usize, 64usize), (64, 32), (32, 16)];
        let total = model.total_base_parameter_count(&sizes);
        assert_eq!(total, 64 * 64 + 64 * 32 + 32 * 16);
    }

    // ── 16. forward_with_lora: unknown layer returns base unchanged ────────────
    #[test]
    fn test_forward_with_lora_unknown_layer() {
        let model = LoraModel::new(LoraConfig::default());
        let base_out = vec![1.0f32, 2.0, 3.0];
        let x = vec![0.5f32; 4];
        let result = model.forward_with_lora("nonexistent", &base_out, &x, 1);
        assert_eq!(result, base_out);
    }

    // ── 17. pseudo_rand_f32 determinism ───────────────────────────────────────
    #[test]
    fn test_pseudo_rand_determinism() {
        let a = pseudo_rand_f32(42, 100);
        let b = pseudo_rand_f32(42, 100);
        assert_eq!(a, b, "pseudo_rand_f32 must be deterministic");

        let c = pseudo_rand_f32(42, 101);
        // Different index should (almost certainly) give different value
        // (not strictly guaranteed but FNV is avalanche-strong)
        assert_ne!(a, c);
    }

    // ── 18. pseudo_rand_unit in [0,1) ────────────────────────────────────────
    #[test]
    fn test_pseudo_rand_unit_range() {
        for i in 0..1000usize {
            let v = pseudo_rand_unit(0xABCD_EF01, i);
            assert!(v >= 0.0 && v < 1.0, "pseudo_rand_unit out of range: {}", v);
        }
    }

    // ─── LoRA+ tests ──────────────────────────────────────────────────────────

    // 19. lora_plus_lr_schedule: lr_B = multiplier * lr_A
    #[test]
    fn test_lora_plus_lr_ratio() {
        let cfg = LoraPlusConfig { learning_rate_b_multiplier: 16.0, ..Default::default() };
        let (lr_a, lr_b) = lora_plus_lr_schedule(1e-4, &cfg).expect("ok");
        assert!((lr_a - 1e-4).abs() < 1e-10, "lr_A should equal base_lr");
        let expected_lr_b = 1e-4 * 16.0;
        assert!((lr_b - expected_lr_b).abs() < 1e-10, "lr_B = 16 * base_lr");
    }

    // 20. lora_plus_lr_schedule: lr_A < lr_B always
    #[test]
    fn test_lora_plus_lr_b_greater_than_a() {
        let cfg = LoraPlusConfig::default();
        let (lr_a, lr_b) = lora_plus_lr_schedule(3e-5, &cfg).expect("ok");
        assert!(lr_b > lr_a, "lr_B must be greater than lr_A in LoRA+");
    }

    // 21. lora_plus_lr_schedule: rejects zero base_lr
    #[test]
    fn test_lora_plus_zero_lr_error() {
        let cfg = LoraPlusConfig::default();
        let err = lora_plus_lr_schedule(0.0, &cfg).unwrap_err();
        assert!(matches!(err, LoraError::InvalidConfig(_)));
    }

    // 22. lora_plus_lr_schedule: rejects negative base_lr
    #[test]
    fn test_lora_plus_negative_lr_error() {
        let cfg = LoraPlusConfig::default();
        let err = lora_plus_lr_schedule(-1e-4, &cfg).unwrap_err();
        assert!(matches!(err, LoraError::InvalidConfig(_)));
    }

    // 23. LoraPlusConfig validate: rejects rank=0
    #[test]
    fn test_lora_plus_config_rank_zero() {
        let cfg = LoraPlusConfig { rank: 0, ..Default::default() };
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, LoraError::InvalidConfig(_)));
    }

    // 24. LoraPlusConfig validate: rejects multiplier <= 0
    #[test]
    fn test_lora_plus_config_bad_multiplier() {
        let cfg = LoraPlusConfig { learning_rate_b_multiplier: -1.0, ..Default::default() };
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, LoraError::InvalidConfig(_)));
    }

    // 25. LoraPlusConfig validate: rejects dropout >= 1.0
    #[test]
    fn test_lora_plus_config_bad_dropout() {
        let cfg = LoraPlusConfig { dropout: 1.0, ..Default::default() };
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, LoraError::InvalidConfig(_)));
    }

    // ─── DoRA tests ───────────────────────────────────────────────────────────

    // 26. DoraLayer from_pretrained: correct shapes
    #[test]
    fn test_dora_layer_from_pretrained_shapes() {
        let d_out = 4usize;
        let d_in = 8usize;
        let rank = 2usize;
        let weight: Vec<f32> = (0..d_out * d_in).map(|i| i as f32 * 0.1 + 0.1).collect();
        let layer = DoraLayer::from_pretrained(&weight, d_out, d_in, rank, 4.0).expect("ok");
        assert_eq!(layer.magnitude.len(), d_out, "magnitude shape");
        assert_eq!(layer.direction.len(), d_out * d_in, "direction shape");
        assert_eq!(layer.lora_a.len(), rank * d_in, "lora_a shape");
        assert_eq!(layer.lora_b.len(), d_out * rank, "lora_b shape");
    }

    // 27. DoraLayer from_pretrained: direction is column-normalised (each row has L2 norm=1)
    #[test]
    fn test_dora_direction_is_unit_norm() {
        let d_out = 3usize;
        let d_in = 4usize;
        let weight: Vec<f32> = (0..d_out * d_in).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let layer = DoraLayer::from_pretrained(&weight, d_out, d_in, 2, 4.0).expect("ok");
        for i in 0..d_out {
            let norm_sq: f32 = (0..d_in).map(|j| layer.direction[i * d_in + j].powi(2)).sum();
            let norm = norm_sq.sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "row {} should be unit norm, got {}", i, norm);
        }
    }

    // 28. DoraLayer from_pretrained: lora_b is all zeros
    #[test]
    fn test_dora_lora_b_init_zero() {
        let weight: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let layer = DoraLayer::from_pretrained(&weight, 2, 2, 2, 4.0).expect("ok");
        assert!(layer.lora_b.iter().all(|&v| v == 0.0), "lora_b should init to zero");
    }

    // 29. DoraLayer forward: at init (lora_b=0), ΔW=0 and output = magnitude * direction @ x
    #[test]
    fn test_dora_forward_at_init() {
        // Simple 2x2 identity-like weight
        let weight = vec![1.0f32, 0.0, 0.0, 1.0]; // [[1,0],[0,1]]
        let layer = DoraLayer::from_pretrained(&weight, 2, 2, 1, 1.0).expect("ok");
        // At init, ΔW = 0, direction = normalised weight rows
        // For [[1,0],[0,1]], magnitude = [1, 1], direction = [[1,0],[0,1]]
        let input = vec![3.0f32, 5.0];
        let out = layer.forward(&input).expect("forward ok");
        // Expected: magnitude ⊙ (direction @ input) = [1*3, 1*5] = [3, 5]
        assert_eq!(out.len(), 2);
        assert!((out[0] - 3.0).abs() < 1e-5, "expected 3.0 got {}", out[0]);
        assert!((out[1] - 5.0).abs() < 1e-5, "expected 5.0 got {}", out[1]);
    }

    // 30. DoraLayer forward: dimension mismatch returns error
    #[test]
    fn test_dora_forward_dim_mismatch() {
        let weight = vec![1.0f32, 0.0, 0.0, 1.0];
        let layer = DoraLayer::from_pretrained(&weight, 2, 2, 1, 1.0).expect("ok");
        let wrong_input = vec![1.0f32, 2.0, 3.0]; // d_in=2 but giving 3
        let err = layer.forward(&wrong_input).unwrap_err();
        assert!(matches!(err, LoraError::BatchSizeMismatch { .. }));
    }

    // 31. DoraLayer merge_weights: at init, merged weight recovers original weight
    #[test]
    fn test_dora_merge_weights_at_init_recovers_original() {
        let d_out = 3usize;
        let d_in = 4usize;
        let weight: Vec<f32> = (0..d_out * d_in).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let layer = DoraLayer::from_pretrained(&weight, d_out, d_in, 2, 4.0).expect("ok");
        // At init, ΔW=0, so W_eff = magnitude ⊙ direction
        // Since direction = weight / magnitude, W_eff[i,j] = magnitude[i] * (weight[i,j] / magnitude[i]) = weight[i,j]
        let merged = layer.merge_weights().expect("merge ok");
        for (&m, &w) in merged.iter().zip(weight.iter()) {
            assert!((m - w).abs() < 1e-4, "merged should recover original: {} vs {}", m, w);
        }
    }

    // 32. DoraLayer from_pretrained: shape mismatch returns error
    #[test]
    fn test_dora_from_pretrained_shape_mismatch() {
        let weight = vec![1.0f32, 2.0]; // only 2 elements, but 2*3=6 expected
        let err = DoraLayer::from_pretrained(&weight, 2, 3, 1, 1.0).unwrap_err();
        assert!(matches!(err, LoraError::ShapeMismatch { .. }));
    }

    // 33. DoraLayer trainable_params count
    #[test]
    fn test_dora_trainable_params_count() {
        let weight: Vec<f32> = vec![1.0f32; 4 * 8]; // d_out=4, d_in=8
        let rank = 2;
        let layer = DoraLayer::from_pretrained(&weight, 4, 8, rank, 4.0).expect("ok");
        // magnitude: d_out=4, lora_a: rank*d_in=2*8=16, lora_b: d_out*rank=4*2=8
        let expected = 4 + 16 + 8; // = 28
        assert_eq!(layer.trainable_params(), expected);
    }

    // 34. DoraLayer magnitude is positive (all L2 norms > 0)
    #[test]
    fn test_dora_magnitude_positive() {
        let weight: Vec<f32> = (0..3 * 5).map(|i| (i as f32 + 1.0) * 0.3).collect();
        let layer = DoraLayer::from_pretrained(&weight, 3, 5, 2, 4.0).expect("ok");
        for &m in &layer.magnitude {
            assert!(m > 0.0, "magnitude must be positive, got {}", m);
        }
    }

    // 35. DoraLayer from_pretrained: rank=0 returns error
    #[test]
    fn test_dora_from_pretrained_rank_zero_error() {
        let weight = vec![1.0f32, 2.0, 3.0, 4.0];
        let err = DoraLayer::from_pretrained(&weight, 2, 2, 0, 1.0).unwrap_err();
        assert!(matches!(err, LoraError::InvalidConfig(_)));
    }
}
