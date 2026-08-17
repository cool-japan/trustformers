//! Real component modules and fusion operators for
//! [`crate::hybrid_architectures::HybridArchitecture`].
//!
//! Every module here performs an actual computation on its input. Modules are
//! built lazily from the observed feature width, so a component always matches
//! the tensor that reaches it and a sequential pipeline keeps its shape.
//!
//! Component coverage:
//!
//! | Component | Implementation |
//! |---|---|
//! | `Transformer` | pre-norm blocks: multi-head self-attention + feed-forward |
//! | `Attention` | multi-head attention (self or causal, per `AttentionType`) |
//! | `RNN` | Elman / GRU / LSTM recurrence over the sequence axis |
//! | `StateSpace` | diagonal state-space scan `h_t = a⊙h_{t-1} + b⊙x_t` |
//! | `CNN` | 1-D convolution over the sequence axis with "same" padding |
//! | `Memory` | content-addressed key/value memory read |
//! | `GNN` | not implemented — no graph structure reaches the forward pass |
//! | `Custom` | not implemented — the configuration carries no executable body |

use trustformers_core::{
    errors::{not_implemented, Result, TrustformersError},
    layers::{FeedForward, LayerNorm, Linear, MultiHeadAttention},
    tensor::Tensor,
    traits::Layer,
};

use super::{
    ArchitecturalComponent, AttentionType, EnsembleMethod, ParallelFusionMethod, RNNCellType,
    StateSpaceType,
};

/// Receptive field of the 1-D convolutional component.
const CONV_KERNEL: usize = 3;

/// Split a `[.., features]` tensor into `(rows, features)` plus its data.
fn rows_and_features(tensor: &Tensor) -> Result<(Vec<f32>, usize, usize, Vec<usize>)> {
    let shape = tensor.shape();
    let features = *shape.last().ok_or_else(|| {
        TrustformersError::shape_error("component input has no dimensions".to_string())
    })?;
    if features == 0 {
        return Err(TrustformersError::shape_error(
            "component input has a zero-width feature dimension".to_string(),
        ));
    }
    let data = tensor.to_vec_f32()?;
    let rows = data.len() / features;
    Ok((data, rows, features, shape))
}

/// Reshape a `[batch, seq, features]` (or `[seq, features]`) tensor into the
/// canonical 3-D layout used by the sequence components.
fn as_sequence(tensor: &Tensor) -> Result<(Tensor, Vec<usize>)> {
    let shape = tensor.shape();
    match shape.len() {
        2 => Ok((tensor.reshape(&[1, shape[0], shape[1]])?, shape)),
        3 => Ok((tensor.contiguous()?, shape)),
        _ => Err(TrustformersError::shape_error(format!(
            "sequence components expect [seq, features] or [batch, seq, features], got {:?}",
            shape
        ))),
    }
}

/// One pre-norm transformer block: self-attention then feed-forward, each
/// wrapped in a residual connection.
#[derive(Debug)]
pub struct TransformerBlock {
    norm_attention: LayerNorm,
    attention: MultiHeadAttention,
    norm_feed_forward: LayerNorm,
    feed_forward: FeedForward,
}

impl TransformerBlock {
    fn new(width: usize, num_heads: usize) -> Result<Self> {
        let heads = usable_heads(width, num_heads);
        Ok(Self {
            norm_attention: LayerNorm::new(vec![width], 1e-12)?,
            attention: MultiHeadAttention::new(width, heads, 0.0, true)?,
            norm_feed_forward: LayerNorm::new(vec![width], 1e-12)?,
            feed_forward: FeedForward::new(width, width * 4, 0.0)?,
        })
    }

    fn forward(&self, input: &Tensor, causal: bool) -> Result<Tensor> {
        let normed = self.norm_attention.forward(input.clone())?;
        let attended = self.attention.forward_self_attention(&normed, None, causal)?;
        let residual = input.add(&attended)?;

        let normed = self.norm_feed_forward.forward(residual.clone())?;
        let projected = self.feed_forward.forward(normed)?;
        residual.add(&projected)
    }

    fn parameter_count(&self) -> usize {
        self.attention.parameter_count()
            + self.feed_forward.parameter_count()
            + self.norm_attention.parameter_count()
            + self.norm_feed_forward.parameter_count()
    }
}

/// Largest head count that divides `width` and does not exceed `requested`.
fn usable_heads(width: usize, requested: usize) -> usize {
    let requested = requested.max(1).min(width);
    for heads in (1..=requested).rev() {
        if width.is_multiple_of(heads) {
            return heads;
        }
    }
    1
}

/// A recurrent cell over the sequence axis.
#[derive(Debug)]
pub struct RecurrentCell {
    cell_type: RNNCellType,
    /// Gate projections; the layout depends on the cell type.
    input_projection: Linear,
    hidden_projection: Linear,
    width: usize,
}

impl RecurrentCell {
    fn new(width: usize, cell_type: RNNCellType) -> Result<Self> {
        let gates =
            match cell_type {
                RNNCellType::LSTM => 4,
                RNNCellType::GRU => 3,
                RNNCellType::RNN | RNNCellType::IndRNN => 1,
                RNNCellType::ConvLSTM => return Err(not_implemented(
                    "hybrid RNN component: ConvLSTM needs spatial inputs the forward pass does \
                     not carry",
                )),
            };
        Ok(Self {
            cell_type,
            input_projection: Linear::new(width, width * gates, true),
            hidden_projection: Linear::new(width, width * gates, false),
            width,
        })
    }

    /// Run the recurrence over `[batch, seq, width]`, returning the same shape.
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        let (batch, seq_len, width) = (shape[0], shape[1], shape[2]);
        if width != self.width {
            return Err(TrustformersError::shape_error(format!(
                "recurrent cell built for width {} received {}",
                self.width, width
            )));
        }

        let mut hidden = Tensor::zeros(&[batch, width])?;
        let mut cell = Tensor::zeros(&[batch, width])?;
        let mut outputs = Vec::with_capacity(seq_len);

        for step in 0..seq_len {
            let step_input = input.slice(1, step, step + 1)?.squeeze(1)?.contiguous()?;
            let gates = self
                .input_projection
                .forward(step_input)?
                .add(&self.hidden_projection.forward(hidden.clone())?)?;

            match self.cell_type {
                RNNCellType::RNN | RNNCellType::IndRNN => {
                    hidden = gates.tanh()?;
                },
                RNNCellType::GRU => {
                    let reset = gates.slice(1, 0, width)?.contiguous()?.sigmoid()?;
                    let update = gates.slice(1, width, 2 * width)?.contiguous()?.sigmoid()?;
                    let candidate =
                        gates.slice(1, 2 * width, 3 * width)?.contiguous()?.mul(&reset)?.tanh()?;
                    let keep = Tensor::ones_like(&update)?.sub(&update)?;
                    hidden = update.mul(&hidden)?.add(&keep.mul(&candidate)?)?;
                },
                RNNCellType::LSTM => {
                    let input_gate = gates.slice(1, 0, width)?.contiguous()?.sigmoid()?;
                    let forget_gate = gates.slice(1, width, 2 * width)?.contiguous()?.sigmoid()?;
                    let candidate = gates.slice(1, 2 * width, 3 * width)?.contiguous()?.tanh()?;
                    let output_gate =
                        gates.slice(1, 3 * width, 4 * width)?.contiguous()?.sigmoid()?;
                    cell = forget_gate.mul(&cell)?.add(&input_gate.mul(&candidate)?)?;
                    hidden = output_gate.mul(&cell.tanh()?)?;
                },
                RNNCellType::ConvLSTM => unreachable!("rejected at construction"),
            }

            outputs.push(hidden.clone().unsqueeze(1)?);
        }

        Tensor::concat(&outputs, 1)?.contiguous()
    }

    fn parameter_count(&self) -> usize {
        self.input_projection.parameter_count() + self.hidden_projection.parameter_count()
    }
}

/// A diagonal state-space scan `h_t = a ⊙ h_{t-1} + b ⊙ x_t`, `y = c ⊙ h + d ⊙ x`.
#[derive(Debug)]
pub struct StateSpaceModule {
    log_decay: Vec<f32>,
    input_gain: Vec<f32>,
    output_gain: Vec<f32>,
    skip: Vec<f32>,
    width: usize,
}

impl StateSpaceModule {
    fn new(width: usize, model_type: &StateSpaceType) -> Result<Self> {
        // Deterministic HiPPO-style geometric decay spectrum: channel i keeps a
        // memory horizon that grows with i.
        let base = match model_type {
            StateSpaceType::Mamba | StateSpaceType::S5 => 0.5,
            _ => 0.25,
        };
        let log_decay = (0..width)
            .map(|index| -(base + index as f32 / width.max(1) as f32))
            .collect::<Vec<_>>();
        Ok(Self {
            log_decay,
            input_gain: vec![1.0; width],
            output_gain: vec![1.0; width],
            skip: vec![0.1; width],
            width,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        let (batch, seq_len, width) = (shape[0], shape[1], shape[2]);
        if width != self.width {
            return Err(TrustformersError::shape_error(format!(
                "state-space module built for width {} received {}",
                self.width, width
            )));
        }

        let values = input.to_vec_f32()?;
        let mut output = vec![0.0f32; values.len()];

        for b in 0..batch {
            let mut state = vec![0.0f32; width];
            for t in 0..seq_len {
                let offset = (b * seq_len + t) * width;
                for channel in 0..width {
                    let decay = self.log_decay[channel].exp();
                    state[channel] = decay * state[channel]
                        + self.input_gain[channel] * values[offset + channel];
                    output[offset + channel] = self.output_gain[channel] * state[channel]
                        + self.skip[channel] * values[offset + channel];
                }
            }
        }

        Tensor::from_vec(output, &shape)
    }

    fn parameter_count(&self) -> usize {
        self.width * 4
    }
}

/// A 1-D convolution over the sequence axis with "same" zero padding.
#[derive(Debug)]
pub struct ConvolutionModule {
    taps: Vec<Linear>,
    width: usize,
}

impl ConvolutionModule {
    fn new(width: usize, layers: usize) -> Result<Self> {
        let _ = layers;
        Ok(Self {
            taps: (0..CONV_KERNEL).map(|_| Linear::new(width, width, false)).collect(),
            width,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        let (batch, seq_len, width) = (shape[0], shape[1], shape[2]);
        if width != self.width {
            return Err(TrustformersError::shape_error(format!(
                "convolution built for width {} received {}",
                self.width, width
            )));
        }

        let radius = (CONV_KERNEL / 2) as isize;
        let mut accumulator: Option<Tensor> = None;

        for (tap_index, tap) in self.taps.iter().enumerate() {
            let offset = tap_index as isize - radius;
            let shifted = shift_sequence(input, offset, batch, seq_len, width)?;
            let contribution = tap.forward(shifted)?;
            accumulator = Some(match accumulator {
                Some(sum) => sum.add(&contribution)?,
                None => contribution,
            });
        }

        accumulator.ok_or_else(|| {
            TrustformersError::shape_error("convolution produced no taps".to_string())
        })
    }

    fn parameter_count(&self) -> usize {
        self.taps.iter().map(|t| t.parameter_count()).sum()
    }
}

/// Shift a `[batch, seq, width]` tensor along the sequence axis, zero-padding.
fn shift_sequence(
    input: &Tensor,
    offset: isize,
    batch: usize,
    seq_len: usize,
    width: usize,
) -> Result<Tensor> {
    if offset == 0 {
        return input.contiguous();
    }
    let values = input.to_vec_f32()?;
    let mut shifted = vec![0.0f32; values.len()];
    for b in 0..batch {
        for t in 0..seq_len {
            let source = t as isize + offset;
            if source < 0 || source >= seq_len as isize {
                continue;
            }
            let source = source as usize;
            let from = (b * seq_len + source) * width;
            let to = (b * seq_len + t) * width;
            shifted[to..to + width].copy_from_slice(&values[from..from + width]);
        }
    }
    Tensor::from_vec(shifted, &[batch, seq_len, width])
}

/// A content-addressed key/value memory read.
#[derive(Debug)]
pub struct MemoryModule {
    query: Linear,
    output: Linear,
    keys: Tensor,
    values: Tensor,
    width: usize,
}

impl MemoryModule {
    fn new(width: usize, memory_size: usize) -> Result<Self> {
        let slots = memory_size.max(1);
        Ok(Self {
            query: Linear::new(width, width, false),
            output: Linear::new(width, width, false),
            keys: Tensor::randn(&[slots, width])?.scalar_mul(0.02)?,
            values: Tensor::randn(&[slots, width])?.scalar_mul(0.02)?,
            width,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (data, rows, features, shape) = rows_and_features(input)?;
        if features != self.width {
            return Err(TrustformersError::shape_error(format!(
                "memory module built for width {} received {}",
                self.width, features
            )));
        }

        let flat = Tensor::from_vec(data, &[rows, features])?;
        let queries = self.query.forward(flat)?;
        let scores = queries.matmul(&self.keys.transpose(0, 1)?)?;
        let weights = scores.softmax(1)?;
        let read = weights.matmul(&self.values)?;
        let projected = self.output.forward(read)?;
        projected.reshape(&shape)
    }

    fn parameter_count(&self) -> usize {
        self.query.parameter_count()
            + self.output.parameter_count()
            + self.keys.shape().iter().product::<usize>()
            + self.values.shape().iter().product::<usize>()
    }
}

/// An instantiated component, built for a concrete feature width.
#[derive(Debug)]
pub enum ComponentModule {
    /// Stack of pre-norm transformer blocks
    Transformer {
        /// Blocks, applied in order
        blocks: Vec<TransformerBlock>,
        /// Whether attention is causally masked
        causal: bool,
    },
    /// A single multi-head attention layer with a residual norm
    Attention {
        /// Attention layer
        attention: MultiHeadAttention,
        /// Pre-attention normalization
        norm: LayerNorm,
        /// Whether attention is causally masked
        causal: bool,
    },
    /// Recurrent stack
    Recurrent {
        /// Cells, applied in order
        cells: Vec<RecurrentCell>,
    },
    /// Diagonal state-space stack
    StateSpace {
        /// Scans, applied in order
        scans: Vec<StateSpaceModule>,
    },
    /// 1-D convolutional stack
    Convolution {
        /// Convolutions, applied in order
        convolutions: Vec<ConvolutionModule>,
    },
    /// Content-addressed memory
    Memory {
        /// Memory read module
        memory: MemoryModule,
    },
}

impl ComponentModule {
    /// Instantiate the module for a component type and an observed feature width.
    pub fn build(component: &ArchitecturalComponent, width: usize) -> Result<Self> {
        if width == 0 {
            return Err(TrustformersError::shape_error(
                "hybrid components need a non-zero feature width".to_string(),
            ));
        }

        match component {
            ArchitecturalComponent::Transformer {
                layers,
                num_heads,
                variant,
                ..
            } => {
                let count = (*layers).clamp(1, 16);
                let mut blocks = Vec::with_capacity(count);
                for _ in 0..count {
                    blocks.push(TransformerBlock::new(width, *num_heads)?);
                }
                Ok(Self::Transformer {
                    blocks,
                    causal: matches!(variant, super::TransformerVariant::GPT),
                })
            },
            ArchitecturalComponent::Attention {
                attention_type,
                num_heads,
                ..
            } => {
                let causal = matches!(attention_type, AttentionType::LocalAttention);
                Ok(Self::Attention {
                    attention: MultiHeadAttention::new(
                        width,
                        usable_heads(width, *num_heads),
                        0.0,
                        true,
                    )?,
                    norm: LayerNorm::new(vec![width], 1e-12)?,
                    causal,
                })
            },
            ArchitecturalComponent::RNN {
                layers, cell_type, ..
            } => {
                let count = (*layers).clamp(1, 8);
                let mut cells = Vec::with_capacity(count);
                for _ in 0..count {
                    cells.push(RecurrentCell::new(width, cell_type.clone())?);
                }
                Ok(Self::Recurrent { cells })
            },
            ArchitecturalComponent::StateSpace {
                layers, model_type, ..
            } => {
                let count = (*layers).clamp(1, 8);
                let mut scans = Vec::with_capacity(count);
                for _ in 0..count {
                    scans.push(StateSpaceModule::new(width, model_type)?);
                }
                Ok(Self::StateSpace { scans })
            },
            ArchitecturalComponent::CNN { layers, .. } => {
                let count = (*layers).clamp(1, 8);
                let mut convolutions = Vec::with_capacity(count);
                for _ in 0..count {
                    convolutions.push(ConvolutionModule::new(width, count)?);
                }
                Ok(Self::Convolution { convolutions })
            },
            ArchitecturalComponent::Memory { memory_size, .. } => Ok(Self::Memory {
                memory: MemoryModule::new(width, *memory_size)?,
            }),
            ArchitecturalComponent::GNN { .. } => Err(not_implemented(
                "hybrid GNN component: message passing needs a graph structure, and the hybrid \
                 forward pass receives only dense tensors",
            )),
            ArchitecturalComponent::Custom { name, .. } => Err(not_implemented(format!(
                "hybrid Custom component '{}': the configuration carries no executable body",
                name
            ))),
        }
    }

    /// Run the module on `input`, preserving its shape.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match self {
            Self::Transformer { blocks, causal } => {
                let (sequence, original) = as_sequence(input)?;
                let mut current = sequence;
                for block in blocks {
                    current = block.forward(&current, *causal)?;
                }
                current.reshape(&original)
            },
            Self::Attention {
                attention,
                norm,
                causal,
            } => {
                let (sequence, original) = as_sequence(input)?;
                let normed = norm.forward(sequence.clone())?;
                let attended = attention.forward_self_attention(&normed, None, *causal)?;
                sequence.add(&attended)?.reshape(&original)
            },
            Self::Recurrent { cells } => {
                let (sequence, original) = as_sequence(input)?;
                let mut current = sequence;
                for cell in cells {
                    current = cell.forward(&current)?;
                }
                current.reshape(&original)
            },
            Self::StateSpace { scans } => {
                let (sequence, original) = as_sequence(input)?;
                let mut current = sequence;
                for scan in scans {
                    current = scan.forward(&current)?;
                }
                current.reshape(&original)
            },
            Self::Convolution { convolutions } => {
                let (sequence, original) = as_sequence(input)?;
                let mut current = sequence;
                for convolution in convolutions {
                    current = convolution.forward(&current)?;
                }
                current.reshape(&original)
            },
            Self::Memory { memory } => memory.forward(input),
        }
    }

    /// Number of trainable scalars in the module.
    pub fn parameter_count(&self) -> usize {
        match self {
            Self::Transformer { blocks, .. } => blocks.iter().map(|b| b.parameter_count()).sum(),
            Self::Attention {
                attention, norm, ..
            } => attention.parameter_count() + norm.parameter_count(),
            Self::Recurrent { cells } => cells.iter().map(|c| c.parameter_count()).sum(),
            Self::StateSpace { scans } => scans.iter().map(|s| s.parameter_count()).sum(),
            Self::Convolution { convolutions } => {
                convolutions.iter().map(|c| c.parameter_count()).sum()
            },
            Self::Memory { memory } => memory.parameter_count(),
        }
    }
}

/// Elementwise mean of a set of equally shaped tensors.
fn mean_of(outputs: &[Tensor]) -> Result<Tensor> {
    let mut sum = outputs
        .first()
        .cloned()
        .ok_or_else(|| TrustformersError::shape_error("no outputs to combine".to_string()))?;
    for output in &outputs[1..] {
        sum = sum.add(output)?;
    }
    sum.scalar_div(outputs.len() as f32)
}

/// Fuse a set of component outputs.
///
/// All fusion methods are real tensor operations; none of them returns
/// `outputs[0]` unchanged (except in the degenerate single-input case, where
/// that *is* the answer).
pub fn fuse(outputs: &[Tensor], method: &ParallelFusionMethod) -> Result<Tensor> {
    if outputs.is_empty() {
        return Err(TrustformersError::shape_error(
            "fusion needs at least one output".to_string(),
        ));
    }
    if outputs.len() == 1 {
        return Ok(outputs[0].clone());
    }

    let reference = outputs[0].shape();
    let same_shape = outputs.iter().all(|o| o.shape() == reference);

    match method {
        ParallelFusionMethod::Concatenation => {
            // Concatenate along the feature axis, then project back to the
            // original width so the pipeline shape is preserved.
            let axis = reference.len().saturating_sub(1);
            let concatenated = Tensor::concat(outputs, axis)?.contiguous()?;
            let width = *reference.last().ok_or_else(|| {
                TrustformersError::shape_error("fusion inputs have no features".to_string())
            })?;
            let projection = Linear::new(width * outputs.len(), width, false);
            projection.forward(concatenated)
        },
        ParallelFusionMethod::Addition => {
            if !same_shape {
                return Err(TrustformersError::shape_error(
                    "additive fusion needs identically shaped outputs".to_string(),
                ));
            }
            mean_of(outputs)
        },
        ParallelFusionMethod::Multiplication => {
            if !same_shape {
                return Err(TrustformersError::shape_error(
                    "multiplicative fusion needs identically shaped outputs".to_string(),
                ));
            }
            let mut product = outputs[0].clone();
            for output in &outputs[1..] {
                product = product.mul(output)?;
            }
            Ok(product)
        },
        ParallelFusionMethod::Gating => {
            if !same_shape {
                return Err(TrustformersError::shape_error(
                    "gated fusion needs identically shaped outputs".to_string(),
                ));
            }
            // g = sigmoid(W [a; b]);  out = g * a + (1 - g) * b, folded left.
            let width = *reference.last().ok_or_else(|| {
                TrustformersError::shape_error("fusion inputs have no features".to_string())
            })?;
            let axis = reference.len().saturating_sub(1);
            let gate_projection = Linear::new(width * 2, width, true);

            let mut fused = outputs[0].clone();
            for output in &outputs[1..] {
                let pair = Tensor::concat(&[fused.clone(), output.clone()], axis)?.contiguous()?;
                let gate = gate_projection.forward(pair)?.sigmoid()?;
                let complement = Tensor::ones_like(&gate)?.sub(&gate)?;
                fused = gate.mul(&fused)?.add(&complement.mul(output)?)?;
            }
            Ok(fused)
        },
        ParallelFusionMethod::CrossAttention => {
            if !same_shape {
                return Err(TrustformersError::shape_error(
                    "cross-attention fusion needs identically shaped outputs".to_string(),
                ));
            }
            let width = *reference.last().ok_or_else(|| {
                TrustformersError::shape_error("fusion inputs have no features".to_string())
            })?;
            let attention = MultiHeadAttention::new(width, usable_heads(width, 4), 0.0, true)?;

            // Each later output attends over the running fusion result.
            let (mut fused, original) = as_sequence(&outputs[0])?;
            for output in &outputs[1..] {
                let (context, _) = as_sequence(output)?;
                let attended =
                    attention.forward_attention(&fused, &context, &context, None, false)?;
                fused = fused.add(&attended)?;
            }
            fused.reshape(&original)
        },
        ParallelFusionMethod::MultiModal => {
            if !same_shape {
                return Err(TrustformersError::shape_error(
                    "multimodal fusion needs identically shaped outputs".to_string(),
                ));
            }
            // Modality-weighted sum with weights learned from the modality
            // means, normalised with a softmax so the fusion is a convex
            // combination that still depends on every input.
            let mut energies = Vec::with_capacity(outputs.len());
            for output in outputs {
                let values = output.to_vec_f32()?;
                let mean = values.iter().sum::<f32>() / values.len().max(1) as f32;
                energies.push(mean);
            }
            let max = energies.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exponentials: Vec<f32> = energies.iter().map(|e| (e - max).exp()).collect();
            let total: f32 = exponentials.iter().sum();

            let mut fused: Option<Tensor> = None;
            for (output, weight) in outputs.iter().zip(exponentials.iter()) {
                let scaled = output.mul_scalar(weight / total.max(f32::EPSILON))?;
                fused = Some(match fused {
                    Some(sum) => sum.add(&scaled)?,
                    None => scaled,
                });
            }
            fused.ok_or_else(|| {
                TrustformersError::shape_error("multimodal fusion produced nothing".to_string())
            })
        },
    }
}

/// Combine ensemble member outputs.
///
/// `weights` are the members' recorded performance scores; they drive the
/// weighted, boosting and dynamic-selection combinators.
pub fn combine_ensemble(
    outputs: &[Tensor],
    weights: &[f32],
    method: &EnsembleMethod,
) -> Result<Tensor> {
    if outputs.is_empty() {
        return Err(TrustformersError::shape_error(
            "ensemble needs at least one member".to_string(),
        ));
    }
    if outputs.len() == 1 {
        return Ok(outputs[0].clone());
    }

    let reference = outputs[0].shape();
    if outputs.iter().any(|o| o.shape() != reference) {
        return Err(TrustformersError::shape_error(
            "ensemble members must produce identically shaped outputs".to_string(),
        ));
    }

    match method {
        EnsembleMethod::MajorityVoting => {
            // Each member votes for its argmax class per row; the winning class
            // is returned as a one-hot distribution.
            let (_, rows, features, shape) = rows_and_features(&outputs[0])?;
            let member_data: Vec<Vec<f32>> =
                outputs.iter().map(|o| o.to_vec_f32()).collect::<Result<_>>()?;

            let mut result = vec![0.0f32; rows * features];
            for row in 0..rows {
                let mut votes = vec![0usize; features];
                for data in &member_data {
                    let slice = &data[row * features..(row + 1) * features];
                    votes[argmax(slice)] += 1;
                }
                let winner = votes
                    .iter()
                    .enumerate()
                    .max_by_key(|(_, count)| **count)
                    .map(|(index, _)| index)
                    .unwrap_or(0);
                result[row * features + winner] = 1.0;
            }
            Tensor::from_vec(result, &shape)
        },
        EnsembleMethod::WeightedAveraging | EnsembleMethod::Boosting => {
            // Boosting differs only in that the weights are the members'
            // sequentially accumulated scores, which the caller supplies.
            let normalised = normalised_weights(weights, outputs.len());
            let mut sum: Option<Tensor> = None;
            for (output, weight) in outputs.iter().zip(normalised.iter()) {
                let scaled = output.mul_scalar(*weight)?;
                sum = Some(match sum {
                    Some(total) => total.add(&scaled)?,
                    None => scaled,
                });
            }
            sum.ok_or_else(|| {
                TrustformersError::shape_error("ensemble produced nothing".to_string())
            })
        },
        EnsembleMethod::Bagging => mean_of(outputs),
        EnsembleMethod::Stacking => {
            // A linear meta-learner over the concatenated member outputs.
            let axis = reference.len().saturating_sub(1);
            let width = *reference.last().ok_or_else(|| {
                TrustformersError::shape_error("ensemble outputs have no features".to_string())
            })?;
            let stacked = Tensor::concat(outputs, axis)?.contiguous()?;
            let meta_learner = Linear::new(width * outputs.len(), width, false);
            meta_learner.forward(stacked)
        },
        EnsembleMethod::DynamicSelection => {
            // Pick the member with the highest recorded score; ties go to the
            // first member.
            let normalised = normalised_weights(weights, outputs.len());
            let best = normalised
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(index, _)| index)
                .unwrap_or(0);
            Ok(outputs[best].clone())
        },
    }
}

fn normalised_weights(weights: &[f32], count: usize) -> Vec<f32> {
    let mut values: Vec<f32> = (0..count)
        .map(|index| weights.get(index).copied().unwrap_or(1.0).max(0.0))
        .collect();
    let total: f32 = values.iter().sum();
    if total <= f32::EPSILON {
        let uniform = 1.0 / count.max(1) as f32;
        return vec![uniform; count];
    }
    for value in values.iter_mut() {
        *value /= total;
    }
    values
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

/// Mean activation energy of a tensor, used by the adaptive router as a real
/// (measured) signal instead of a hardcoded score.
pub fn activation_energy(tensor: &Tensor) -> Result<f32> {
    let values = tensor.to_vec_f32()?;
    if values.is_empty() {
        return Ok(0.0);
    }
    Ok(values.iter().map(|v| v.abs()).sum::<f32>() / values.len() as f32)
}

/// Normalised prediction confidence: the softmax margin between the best and
/// second-best feature of each row, averaged over rows.
pub fn prediction_confidence(tensor: &Tensor) -> Result<f32> {
    let (data, rows, features, _) = rows_and_features(tensor)?;
    if features < 2 || rows == 0 {
        return Ok(0.0);
    }

    let mut total = 0.0f32;
    for row in 0..rows {
        let slice = &data[row * features..(row + 1) * features];
        let max = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exponentials: Vec<f32> = slice.iter().map(|v| (v - max).exp()).collect();
        let sum: f32 = exponentials.iter().sum();
        let mut sorted = exponentials;
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        total += (sorted[0] - sorted[1]) / sum.max(f32::EPSILON);
    }

    Ok(total / rows as f32)
}
