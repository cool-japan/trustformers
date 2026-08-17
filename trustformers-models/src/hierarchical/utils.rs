use super::config::{AggregationMethod, ReductionMethod, TreeConstruction};
use trustformers_core::{
    errors::{invalid_input, not_implemented, tensor_op_error, Result},
    tensor::Tensor,
};

/// Decompose a 3-D `[batch, seq, hidden]` tensor into flat row-major data plus dims.
///
/// Every pooling / resampling routine in this module works on the sequence axis of
/// a `[batch, seq, hidden]` activation tensor, so the shape contract is validated
/// once here rather than being silently assumed by each routine.
fn as_batch_seq_hidden(tensor: &Tensor) -> Result<(Vec<f32>, usize, usize, usize)> {
    let shape = tensor.shape();
    if shape.len() != 3 {
        return Err(tensor_op_error(
            "hierarchical_pooling",
            format!("expected a 3-D [batch, seq, hidden] tensor, got shape {shape:?}"),
        ));
    }
    let data = tensor.data()?;
    let (batch, seq, hidden) = (shape[0], shape[1], shape[2]);
    if data.len() != batch * seq * hidden {
        return Err(tensor_op_error(
            "hierarchical_pooling",
            format!(
                "data length {} is inconsistent with shape {shape:?}",
                data.len()
            ),
        ));
    }
    Ok((data, batch, seq, hidden))
}

/// Reject a zero window size before it turns into a division by zero.
fn check_reduction_factor(reduction_factor: usize) -> Result<()> {
    if reduction_factor == 0 {
        return Err(invalid_input(
            "hierarchical reduction_factor must be >= 1, got 0",
        ));
    }
    Ok(())
}

/// Linearly resample a 3-D buffer along `axis` (0 = batch, 1 = seq, 2 = hidden).
///
/// Uses `align_corners`-style linear interpolation: output index `j` samples the
/// input at `j · (n_in - 1) / (n_out - 1)`, so the first and last elements are
/// preserved exactly and a same-length request is the identity. Works in both
/// directions (up- and down-sampling).
fn resample_axis(data: &[f32], dims: [usize; 3], axis: usize, new_len: usize) -> Vec<f32> {
    let outer: usize = dims[..axis].iter().product();
    let inner: usize = dims[axis + 1..].iter().product();
    let n_in = dims[axis];

    let mut out = vec![0.0f32; outer * new_len * inner];
    if n_in == 0 || new_len == 0 || inner == 0 {
        return out;
    }

    for o in 0..outer {
        for j in 0..new_len {
            let pos = if new_len == 1 || n_in == 1 {
                0.0f32
            } else {
                j as f32 * (n_in - 1) as f32 / (new_len - 1) as f32
            };
            let lo = pos.floor().max(0.0) as usize;
            let lo = lo.min(n_in - 1);
            let hi = (lo + 1).min(n_in - 1);
            let frac = pos - lo as f32;

            let lo_base = (o * n_in + lo) * inner;
            let hi_base = (o * n_in + hi) * inner;
            let out_base = (o * new_len + j) * inner;
            for k in 0..inner {
                let a = data[lo_base + k];
                let b = data[hi_base + k];
                out[out_base + k] = a + (b - a) * frac;
            }
        }
    }

    out
}

/// Numerically stable softmax over a slice, written in place.
fn softmax_in_place(scores: &mut [f32]) {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        let uniform = 1.0 / scores.len().max(1) as f32;
        scores.iter_mut().for_each(|s| *s = uniform);
        return;
    }
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max).exp();
        sum += *s;
    }
    if sum > 0.0 {
        scores.iter_mut().for_each(|s| *s /= sum);
    } else {
        let uniform = 1.0 / scores.len().max(1) as f32;
        scores.iter_mut().for_each(|s| *s = uniform);
    }
}

/// Output structure for hierarchical transformers
#[derive(Debug, Clone)]
pub struct HierarchicalOutput {
    /// Final output tensor
    pub output: Tensor,
    /// Outputs from each hierarchical level
    pub level_outputs: Vec<Tensor>,
    /// Attention weights for each level (optional)
    pub attention_weights: Option<Vec<Tensor>>,
    /// Hierarchical positions
    pub hierarchical_positions: Option<Vec<Vec<usize>>>,
}

/// Build hierarchical representation from input sequence
pub fn build_hierarchy(
    input: Tensor,
    num_levels: usize,
    reduction_factor: usize,
    reduction_method: ReductionMethod,
) -> Result<Vec<Tensor>> {
    let mut hierarchy = Vec::new();
    let mut current_tensor = input;

    for level in 0..num_levels {
        hierarchy.push(current_tensor.clone());

        if level < num_levels - 1 {
            // Reduce sequence length for next level
            current_tensor =
                reduce_sequence_length(current_tensor, reduction_factor, &reduction_method)?;
        }
    }

    Ok(hierarchy)
}

/// Reduce sequence length using specified method
fn reduce_sequence_length(
    tensor: Tensor,
    reduction_factor: usize,
    method: &ReductionMethod,
) -> Result<Tensor> {
    match method {
        ReductionMethod::AveragePooling => average_pool_sequence(tensor, reduction_factor),
        ReductionMethod::MaxPooling => max_pool_sequence(tensor, reduction_factor),
        ReductionMethod::LearnablePooling => Err(not_implemented(
            "ReductionMethod::LearnablePooling needs learned pooling weights, which this \
             weight-free utility API cannot provide; use AveragePooling, MaxPooling, \
             AttentionPooling or TokenMerging",
        )),
        ReductionMethod::StridedConvolution => Err(not_implemented(
            "ReductionMethod::StridedConvolution needs a learned convolution kernel, which this \
             weight-free utility API cannot provide; use AveragePooling, MaxPooling, \
             AttentionPooling or TokenMerging",
        )),
        ReductionMethod::AttentionPooling => attention_pool_sequence(tensor, reduction_factor),
        ReductionMethod::TokenMerging => token_merge_sequence(tensor, reduction_factor),
    }
}

/// Average pooling over the sequence dimension.
///
/// Non-overlapping windows of `reduction_factor` timesteps are replaced by their
/// arithmetic mean, per batch element and per hidden channel. The trailing window
/// may be shorter than `reduction_factor` and is averaged over its true length.
fn average_pool_sequence(tensor: Tensor, reduction_factor: usize) -> Result<Tensor> {
    check_reduction_factor(reduction_factor)?;
    let (data, batch_size, seq_len, hidden_size) = as_batch_seq_hidden(&tensor)?;

    let new_seq_len = seq_len.div_ceil(reduction_factor);
    let mut pooled_data = vec![0.0f32; batch_size * new_seq_len * hidden_size];

    for b in 0..batch_size {
        for s in 0..new_seq_len {
            let start = s * reduction_factor;
            let end = (start + reduction_factor).min(seq_len);
            let count = (end - start) as f32;
            let out_base = (b * new_seq_len + s) * hidden_size;

            for i in start..end {
                let in_base = (b * seq_len + i) * hidden_size;
                for h in 0..hidden_size {
                    pooled_data[out_base + h] += data[in_base + h];
                }
            }
            if count > 0.0 {
                for h in 0..hidden_size {
                    pooled_data[out_base + h] /= count;
                }
            }
        }
    }

    Tensor::from_vec(pooled_data, &[batch_size, new_seq_len, hidden_size])
}

/// Max pooling over the sequence dimension.
///
/// Each non-overlapping window of `reduction_factor` timesteps is replaced by the
/// per-channel maximum over that window.
fn max_pool_sequence(tensor: Tensor, reduction_factor: usize) -> Result<Tensor> {
    check_reduction_factor(reduction_factor)?;
    let (data, batch_size, seq_len, hidden_size) = as_batch_seq_hidden(&tensor)?;

    let new_seq_len = seq_len.div_ceil(reduction_factor);
    let mut pooled_data = vec![f32::NEG_INFINITY; batch_size * new_seq_len * hidden_size];

    for b in 0..batch_size {
        for s in 0..new_seq_len {
            let start = s * reduction_factor;
            let end = (start + reduction_factor).min(seq_len);
            let out_base = (b * new_seq_len + s) * hidden_size;

            for i in start..end {
                let in_base = (b * seq_len + i) * hidden_size;
                for h in 0..hidden_size {
                    let v = data[in_base + h];
                    if v > pooled_data[out_base + h] {
                        pooled_data[out_base + h] = v;
                    }
                }
            }
        }
    }

    Tensor::from_vec(pooled_data, &[batch_size, new_seq_len, hidden_size])
}

/// Parameter-free attention pooling over the sequence dimension.
///
/// Within each window the mean vector acts as the query; every token is scored by
/// its scaled dot product with that query, the scores are softmax-normalised over
/// the window, and the pooled token is the resulting convex combination:
///
/// ```text
/// q_w   = mean_{i∈w} x_i
/// α_i   = softmax_i( ⟨x_i, q_w⟩ / sqrt(hidden) )
/// out_w = Σ_{i∈w} α_i · x_i
/// ```
///
/// This is the standard weight-free attention-pooling reduction: it degenerates to
/// average pooling when all tokens in a window are identical, and emphasises tokens
/// aligned with the window's dominant direction otherwise.
fn attention_pool_sequence(tensor: Tensor, reduction_factor: usize) -> Result<Tensor> {
    check_reduction_factor(reduction_factor)?;
    let (data, batch_size, seq_len, hidden_size) = as_batch_seq_hidden(&tensor)?;

    let new_seq_len = seq_len.div_ceil(reduction_factor);
    let mut pooled_data = vec![0.0f32; batch_size * new_seq_len * hidden_size];
    let scale = 1.0 / (hidden_size.max(1) as f32).sqrt();

    for b in 0..batch_size {
        for s in 0..new_seq_len {
            let start = s * reduction_factor;
            let end = (start + reduction_factor).min(seq_len);
            let window = end - start;
            let out_base = (b * new_seq_len + s) * hidden_size;
            if window == 0 {
                continue;
            }

            // Query = window mean.
            let mut query = vec![0.0f32; hidden_size];
            for i in start..end {
                let in_base = (b * seq_len + i) * hidden_size;
                for h in 0..hidden_size {
                    query[h] += data[in_base + h];
                }
            }
            for q in query.iter_mut() {
                *q /= window as f32;
            }

            // Scaled dot-product scores, then softmax over the window.
            let mut scores = vec![0.0f32; window];
            for (slot, i) in (start..end).enumerate() {
                let in_base = (b * seq_len + i) * hidden_size;
                let mut dot = 0.0f32;
                for h in 0..hidden_size {
                    dot += data[in_base + h] * query[h];
                }
                scores[slot] = dot * scale;
            }
            softmax_in_place(&mut scores);

            for (slot, i) in (start..end).enumerate() {
                let in_base = (b * seq_len + i) * hidden_size;
                let w = scores[slot];
                for h in 0..hidden_size {
                    pooled_data[out_base + h] += w * data[in_base + h];
                }
            }
        }
    }

    Tensor::from_vec(pooled_data, &[batch_size, new_seq_len, hidden_size])
}

/// Token merging (ToMe-style agglomerative merge) over the sequence dimension.
///
/// Within each window the two most cosine-similar *adjacent* tokens are repeatedly
/// merged into their size-weighted mean until a single token remains. Unlike plain
/// average pooling this keeps the merge order data-dependent: near-duplicate tokens
/// collapse first, so an outlier token retains a larger share of the merged vector.
///
/// Reference: Bolya et al., "Token Merging: Your ViT But Faster" (2023).
fn token_merge_sequence(tensor: Tensor, reduction_factor: usize) -> Result<Tensor> {
    check_reduction_factor(reduction_factor)?;
    let (data, batch_size, seq_len, hidden_size) = as_batch_seq_hidden(&tensor)?;

    let new_seq_len = seq_len.div_ceil(reduction_factor);
    let mut pooled_data = vec![0.0f32; batch_size * new_seq_len * hidden_size];

    for b in 0..batch_size {
        for s in 0..new_seq_len {
            let start = s * reduction_factor;
            let end = (start + reduction_factor).min(seq_len);
            let out_base = (b * new_seq_len + s) * hidden_size;
            if start >= end {
                continue;
            }

            // (vector, accumulated token count) for every survivor in this window.
            let mut tokens: Vec<(Vec<f32>, f32)> = (start..end)
                .map(|i| {
                    let in_base = (b * seq_len + i) * hidden_size;
                    (data[in_base..in_base + hidden_size].to_vec(), 1.0f32)
                })
                .collect();

            while tokens.len() > 1 {
                let mut best_idx = 0usize;
                let mut best_sim = f32::NEG_INFINITY;
                for i in 0..tokens.len() - 1 {
                    let sim = cosine_similarity(&tokens[i].0, &tokens[i + 1].0);
                    if sim > best_sim {
                        best_sim = sim;
                        best_idx = i;
                    }
                }

                let (right_vec, right_size) = tokens.remove(best_idx + 1);
                let (left_vec, left_size) = &mut tokens[best_idx];
                let total = *left_size + right_size;
                for h in 0..hidden_size {
                    left_vec[h] = (left_vec[h] * *left_size + right_vec[h] * right_size) / total;
                }
                *left_size = total;
            }

            let merged = &tokens[0].0;
            pooled_data[out_base..out_base + hidden_size].copy_from_slice(merged);
        }
    }

    Tensor::from_vec(pooled_data, &[batch_size, new_seq_len, hidden_size])
}

/// Cosine similarity of two equal-length vectors; `0.0` when either is degenerate.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom > 0.0 {
        dot / denom
    } else {
        0.0
    }
}

/// Hierarchical pooling with different strategies
pub fn hierarchical_pooling(
    tensors: Vec<Tensor>,
    method: &ReductionMethod,
    reduction_factors: Vec<usize>,
) -> Result<Vec<Tensor>> {
    let mut pooled_tensors = Vec::new();

    for (i, tensor) in tensors.iter().enumerate() {
        if i < reduction_factors.len() {
            let pooled = reduce_sequence_length(tensor.clone(), reduction_factors[i], method)?;
            pooled_tensors.push(pooled);
        } else {
            pooled_tensors.push(tensor.clone());
        }
    }

    Ok(pooled_tensors)
}

/// Hierarchical upsampling
pub fn hierarchical_upsampling(
    tensors: Vec<Tensor>,
    target_lengths: Vec<usize>,
) -> Result<Vec<Tensor>> {
    let mut upsampled_tensors = Vec::new();

    for (i, tensor) in tensors.iter().enumerate() {
        if i < target_lengths.len() {
            let upsampled = upsample_sequence(tensor.clone(), target_lengths[i])?;
            upsampled_tensors.push(upsampled);
        } else {
            upsampled_tensors.push(tensor.clone());
        }
    }

    Ok(upsampled_tensors)
}

/// Resample a `[batch, seq, hidden]` sequence to `target_length` timesteps.
///
/// Uses `align_corners` linear interpolation along the sequence axis, so the first
/// and last timesteps are reproduced exactly and an unchanged length is the
/// identity. Works in both directions (the name is historical: shortening is a
/// valid request from `hierarchical_upsampling` when a level is already longer
/// than its target).
fn upsample_sequence(tensor: Tensor, target_length: usize) -> Result<Tensor> {
    let (data, batch_size, seq_len, hidden_size) = as_batch_seq_hidden(&tensor)?;

    if seq_len == target_length {
        return Ok(tensor);
    }

    let resampled = resample_axis(&data, [batch_size, seq_len, hidden_size], 1, target_length);
    Tensor::from_vec(resampled, &[batch_size, target_length, hidden_size])
}

/// Compute hierarchical positions for each level
pub fn compute_hierarchical_positions(
    seq_len: usize,
    num_levels: usize,
    reduction_factor: usize,
) -> Result<Vec<Vec<usize>>> {
    let mut positions = Vec::new();

    for level in 0..num_levels {
        let level_reduction = reduction_factor.pow(level as u32);
        let level_seq_len = seq_len.div_ceil(level_reduction);

        let level_positions: Vec<usize> = (0..level_seq_len).map(|i| i * level_reduction).collect();

        positions.push(level_positions);
    }

    Ok(positions)
}

/// Create attention mask for tree-structured attention
pub fn create_tree_mask(
    seq_len: usize,
    branching_factor: usize,
    tree_construction: &TreeConstruction,
) -> Result<Tensor> {
    match tree_construction {
        TreeConstruction::Binary => create_binary_tree_mask(seq_len),
        TreeConstruction::Balanced => create_balanced_tree_mask(seq_len, branching_factor),
        TreeConstruction::Learned => Err(not_implemented(
            "TreeConstruction::Learned requires a learned structure predictor; no learned tree \
             parameters exist in this weight-free utility API",
        )),
        TreeConstruction::SyntaxGuided => Err(not_implemented(
            "TreeConstruction::SyntaxGuided requires an external syntactic parse of the input; \
             supply the parse and build the mask explicitly",
        )),
    }
}

/// Create binary tree attention mask
fn create_binary_tree_mask(seq_len: usize) -> Result<Tensor> {
    let mut mask = vec![vec![f32::NEG_INFINITY; seq_len]; seq_len];

    // Build binary tree structure
    for i in 0..seq_len {
        // Each node can attend to its parent and children
        let parent = if i > 0 { (i - 1) / 2 } else { 0 };
        let left_child = 2 * i + 1;
        let right_child = 2 * i + 2;

        // Self-attention
        mask[i][i] = 0.0;

        // Parent connection
        if parent < seq_len {
            mask[i][parent] = 0.0;
        }

        // Child connections
        if left_child < seq_len {
            mask[i][left_child] = 0.0;
        }
        if right_child < seq_len {
            mask[i][right_child] = 0.0;
        }
    }

    let flattened_mask: Vec<f32> = mask.into_iter().flatten().collect();
    Tensor::from_vec(flattened_mask, &[seq_len, seq_len])
}

/// Create balanced k-ary tree attention mask
fn create_balanced_tree_mask(seq_len: usize, branching_factor: usize) -> Result<Tensor> {
    let mut mask = vec![vec![f32::NEG_INFINITY; seq_len]; seq_len];

    // Build k-ary tree structure
    for i in 0..seq_len {
        // Each node can attend to its parent and children
        let parent = if i > 0 { (i - 1) / branching_factor } else { 0 };

        // Self-attention
        mask[i][i] = 0.0;

        // Parent connection
        if parent < seq_len {
            mask[i][parent] = 0.0;
        }

        // Child connections
        for j in 0..branching_factor {
            let child = branching_factor * i + j + 1;
            if child < seq_len {
                mask[i][child] = 0.0;
            }
        }
    }

    let flattened_mask: Vec<f32> = mask.into_iter().flatten().collect();
    Tensor::from_vec(flattened_mask, &[seq_len, seq_len])
}

/// Aggregate features across hierarchical levels
pub fn aggregate_hierarchical_features(
    level_outputs: Vec<Tensor>,
    method: &AggregationMethod,
    target_shape: &[usize],
) -> Result<Tensor> {
    if level_outputs.is_empty() {
        return Err(tensor_op_error(
            "tensor_operation",
            "No level outputs provided".to_string(),
        ));
    }

    match method {
        AggregationMethod::Sum => aggregate_sum(level_outputs, target_shape),
        AggregationMethod::Concatenation => aggregate_concatenation(level_outputs, target_shape),
        AggregationMethod::WeightedSum => aggregate_weighted_sum(level_outputs, target_shape),
        AggregationMethod::AttentionAggregation => aggregate_attention(level_outputs, target_shape),
        AggregationMethod::GatedAggregation => aggregate_gated(level_outputs, target_shape),
    }
}

/// Sum aggregation across levels
fn aggregate_sum(level_outputs: Vec<Tensor>, target_shape: &[usize]) -> Result<Tensor> {
    // Every level — including the first — is aligned to the target shape before the
    // sum; the coarser levels have shorter sequences after hierarchical reduction.
    let mut result = upsample_to_shape(level_outputs[0].clone(), target_shape)?;

    for level_output in level_outputs.iter().skip(1) {
        let upsampled = upsample_to_shape(level_output.clone(), target_shape)?;
        result = result.add(&upsampled)?;
    }

    Ok(result)
}

/// Concatenation aggregation
fn aggregate_concatenation(level_outputs: Vec<Tensor>, target_shape: &[usize]) -> Result<Tensor> {
    let mut aligned_outputs = Vec::new();

    for output in level_outputs {
        let aligned = upsample_to_shape(output, target_shape)?;
        aligned_outputs.push(aligned);
    }

    let last_dim = target_shape.len() - 1;
    Tensor::concat(&aligned_outputs, last_dim)
}

/// Weighted sum aggregation with uniform level weights
fn aggregate_weighted_sum(level_outputs: Vec<Tensor>, target_shape: &[usize]) -> Result<Tensor> {
    let num_levels = level_outputs.len();
    let weight = 1.0 / num_levels as f32;

    let mut result = Tensor::zeros(target_shape)?;

    for output in level_outputs.iter() {
        let upsampled = upsample_to_shape(output.clone(), target_shape)?;
        let weighted = upsampled.mul_scalar(weight)?;
        result = result.add(&weighted)?;
    }

    Ok(result)
}

/// Attention-based aggregation across hierarchical levels.
///
/// Each level is aligned to the target shape and summarised by its mean activation
/// vector. The levels are then scored against the mean of those summaries with a
/// scaled dot product and softmax-normalised, so levels whose representation aligns
/// with the consensus direction contribute more:
///
/// ```text
/// s_l   = mean over (batch, seq) of level l          (a hidden-dim vector)
/// q     = mean_l s_l
/// α_l   = softmax_l( ⟨s_l, q⟩ / sqrt(hidden) )
/// out   = Σ_l α_l · level_l
/// ```
fn aggregate_attention(level_outputs: Vec<Tensor>, target_shape: &[usize]) -> Result<Tensor> {
    if target_shape.len() != 3 {
        return Err(tensor_op_error(
            "hierarchical_aggregate",
            format!("target shape must be 3-D [batch, seq, hidden], got {target_shape:?}"),
        ));
    }
    let hidden = target_shape[2];

    let mut aligned = Vec::with_capacity(level_outputs.len());
    let mut summaries = Vec::with_capacity(level_outputs.len());
    for output in level_outputs {
        let level = upsample_to_shape(output, target_shape)?;
        let data = level.data()?;
        let tokens = (data.len() / hidden.max(1)).max(1);
        let mut summary = vec![0.0f32; hidden];
        for (i, v) in data.iter().enumerate() {
            summary[i % hidden] += v;
        }
        for value in summary.iter_mut() {
            *value /= tokens as f32;
        }
        summaries.push(summary);
        aligned.push(level);
    }

    // Consensus query: the mean of the per-level summaries.
    let mut query = vec![0.0f32; hidden];
    for summary in &summaries {
        for (q, s) in query.iter_mut().zip(summary.iter()) {
            *q += s;
        }
    }
    for q in query.iter_mut() {
        *q /= summaries.len() as f32;
    }

    let scale = 1.0 / (hidden.max(1) as f32).sqrt();
    let mut scores: Vec<f32> = summaries
        .iter()
        .map(|summary| summary.iter().zip(query.iter()).map(|(s, q)| s * q).sum::<f32>() * scale)
        .collect();
    softmax_in_place(&mut scores);

    let mut result = Tensor::zeros(target_shape)?;
    for (level, weight) in aligned.iter().zip(scores.iter()) {
        result = result.add(&level.mul_scalar(*weight)?)?;
    }

    Ok(result)
}

/// Gated aggregation across hierarchical levels.
///
/// A parameter-free sigmoid gate is derived from each level's own activations —
/// `g_l = σ(mean(level_l))`, broadcast over the level — and the gated levels are
/// combined and renormalised by the total gate mass, so a level whose activations
/// are strongly negative is suppressed rather than averaged in blindly.
fn aggregate_gated(level_outputs: Vec<Tensor>, target_shape: &[usize]) -> Result<Tensor> {
    let mut gates = Vec::with_capacity(level_outputs.len());
    let mut aligned = Vec::with_capacity(level_outputs.len());

    for output in level_outputs {
        let level = upsample_to_shape(output, target_shape)?;
        let data = level.data()?;
        let mean = if data.is_empty() { 0.0 } else { data.iter().sum::<f32>() / data.len() as f32 };
        gates.push(1.0 / (1.0 + (-mean).exp()));
        aligned.push(level);
    }

    let gate_sum: f32 = gates.iter().sum();
    let mut result = Tensor::zeros(target_shape)?;
    if gate_sum <= 0.0 {
        return Ok(result);
    }

    for (level, gate) in aligned.iter().zip(gates.iter()) {
        result = result.add(&level.mul_scalar(gate / gate_sum)?)?;
    }

    Ok(result)
}

/// Resample a `[batch, seq, hidden]` tensor to `target_shape`.
///
/// The sequence axis and, when a pyramid level narrows the hidden width, the hidden
/// axis are linearly interpolated. A batch of 1 is broadcast to the target batch;
/// any other batch mismatch is an error rather than a silent reinterpretation.
fn upsample_to_shape(tensor: Tensor, target_shape: &[usize]) -> Result<Tensor> {
    let current_shape = tensor.shape();

    if current_shape == target_shape {
        return Ok(tensor);
    }

    if target_shape.len() != 3 {
        return Err(tensor_op_error(
            "hierarchical_upsample",
            format!("target shape must be 3-D [batch, seq, hidden], got {target_shape:?}"),
        ));
    }

    let (data, batch_size, seq_len, hidden_size) = as_batch_seq_hidden(&tensor)?;
    let (target_batch, target_seq, target_hidden) =
        (target_shape[0], target_shape[1], target_shape[2]);

    let mut dims = [batch_size, seq_len, hidden_size];
    let mut buffer = data;

    if seq_len != target_seq {
        buffer = resample_axis(&buffer, dims, 1, target_seq);
        dims[1] = target_seq;
    }
    if hidden_size != target_hidden {
        buffer = resample_axis(&buffer, dims, 2, target_hidden);
        dims[2] = target_hidden;
    }
    if batch_size != target_batch {
        if batch_size != 1 {
            return Err(tensor_op_error(
                "hierarchical_upsample",
                format!(
                    "cannot align batch {batch_size} to target batch {target_batch}: only a \
                     batch of 1 can be broadcast"
                ),
            ));
        }
        let per_item = dims[1] * dims[2];
        let mut broadcast = Vec::with_capacity(target_batch * per_item);
        for _ in 0..target_batch {
            broadcast.extend_from_slice(&buffer[..per_item]);
        }
        buffer = broadcast;
    }

    Tensor::from_vec(buffer, target_shape)
}

/// Compute hierarchical attention patterns
///
/// `positions` must hold one row per level output — the rows produced by
/// [`compute_hierarchical_positions`] for the same hierarchy. A short `positions`
/// slice is reported rather than indexed past its end.
pub fn compute_hierarchical_attention_patterns(
    level_outputs: &[Tensor],
    positions: &[Vec<usize>],
) -> Result<Vec<AttentionPattern>> {
    if positions.len() < level_outputs.len() {
        return Err(invalid_input(format!(
            "compute_hierarchical_attention_patterns needs one positions row per level output, \
             got {} rows for {} levels",
            positions.len(),
            level_outputs.len()
        )));
    }

    let mut patterns = Vec::new();

    for (level, output) in level_outputs.iter().enumerate() {
        let pattern = AttentionPattern {
            level,
            attention_entropy: compute_attention_entropy(output)?,
            attention_sparsity: compute_attention_sparsity(output)?,
            dominant_positions: positions[level].clone(),
        };
        patterns.push(pattern);
    }

    Ok(patterns)
}

/// Attention pattern analysis
#[derive(Debug, Clone)]
pub struct AttentionPattern {
    pub level: usize,
    pub attention_entropy: f32,
    pub attention_sparsity: f32,
    pub dominant_positions: Vec<usize>,
}

/// Normalised Shannon entropy of the row-wise softmax distribution.
///
/// Each row along the last axis is turned into a distribution with a numerically
/// stable softmax, its entropy `H = -Σ p·ln p` is divided by `ln(n)` so it lands in
/// `[0, 1]`, and the per-row values are averaged. `1.0` means a perfectly flat
/// (maximally uncertain) distribution, `0.0` means one position dominates.
fn compute_attention_entropy(tensor: &Tensor) -> Result<f32> {
    let shape = tensor.shape();
    let last_dim = shape.last().copied().unwrap_or(0);
    let data = tensor.data()?;

    if last_dim <= 1 || data.is_empty() {
        return Ok(0.0);
    }

    let ln_n = (last_dim as f32).ln();
    let mut total = 0.0f32;
    let mut rows = 0usize;

    for row in data.chunks(last_dim) {
        let mut probabilities = row.to_vec();
        softmax_in_place(&mut probabilities);
        let entropy: f32 = probabilities.iter().filter(|&&p| p > 1e-10).map(|&p| -p * p.ln()).sum();
        total += (entropy / ln_n).clamp(0.0, 1.0);
        rows += 1;
    }

    if rows == 0 {
        return Ok(0.0);
    }
    Ok(total / rows as f32)
}

/// Fraction of entries that are negligible relative to the largest magnitude.
///
/// An entry counts as sparse when `|x| <= 1e-2 · max|x|`; an all-zero tensor is
/// fully sparse (`1.0`) by definition.
fn compute_attention_sparsity(tensor: &Tensor) -> Result<f32> {
    let data = tensor.data()?;
    if data.is_empty() {
        return Ok(0.0);
    }

    let max_magnitude = data.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
    if max_magnitude == 0.0 {
        return Ok(1.0);
    }

    let threshold = 1e-2 * max_magnitude;
    let near_zero = data.iter().filter(|v| v.abs() <= threshold).count();
    Ok(near_zero as f32 / data.len() as f32)
}

/// Build hierarchical tree structure
pub fn build_hierarchical_tree(
    seq_len: usize,
    branching_factor: usize,
    max_depth: usize,
) -> Result<HierarchicalTree> {
    let mut tree = HierarchicalTree::new(seq_len, branching_factor, max_depth);

    // Build tree structure
    for depth in 0..max_depth {
        let nodes_at_level = branching_factor.pow(depth as u32);
        for i in 0..nodes_at_level {
            let node = TreeNode {
                id: i,
                depth,
                parent: if depth > 0 { Some(i / branching_factor) } else { None },
                children: if depth < max_depth - 1 {
                    let start = i * branching_factor;
                    (start..start + branching_factor).collect()
                } else {
                    Vec::new()
                },
                position: i,
            };
            tree.add_node(node);
        }
    }

    Ok(tree)
}

/// Hierarchical tree structure
#[derive(Debug, Clone)]
pub struct HierarchicalTree {
    pub nodes: Vec<TreeNode>,
    pub seq_len: usize,
    pub branching_factor: usize,
    pub max_depth: usize,
}

/// Tree node
#[derive(Debug, Clone)]
pub struct TreeNode {
    pub id: usize,
    pub depth: usize,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub position: usize,
}

impl HierarchicalTree {
    pub fn new(seq_len: usize, branching_factor: usize, max_depth: usize) -> Self {
        Self {
            nodes: Vec::new(),
            seq_len,
            branching_factor,
            max_depth,
        }
    }

    pub fn add_node(&mut self, node: TreeNode) {
        self.nodes.push(node);
    }

    pub fn get_node(&self, id: usize) -> Option<&TreeNode> {
        self.nodes.get(id)
    }

    pub fn get_nodes_at_depth(&self, depth: usize) -> Vec<&TreeNode> {
        self.nodes.iter().filter(|node| node.depth == depth).collect()
    }
}

#[cfg(test)]
mod pooling_tests {
    use super::*;
    use crate::hierarchical::config::TreeConstruction;

    /// `[1, 4, 2]` tensor with distinct, hand-checkable values.
    fn ramp_tensor() -> Tensor {
        // t0 = [1, 2], t1 = [3, 4], t2 = [5, 6], t3 = [7, 8]
        Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[1, 4, 2])
            .expect("tensor construction")
    }

    /// Average pooling must return the real window means, not zeros.
    ///
    /// This is the regression test for the pooling stub whose window loop only
    /// incremented a counter and pushed an all-zero `sum` vector.
    #[test]
    fn test_average_pool_matches_hand_computed_means() {
        let pooled = average_pool_sequence(ramp_tensor(), 2).expect("average pool");
        assert_eq!(pooled.shape(), vec![1, 2, 2]);
        let data = pooled.data().expect("data");
        // mean([1,2],[3,4]) = [2,3]; mean([5,6],[7,8]) = [6,7]
        assert_eq!(data, vec![2.0, 3.0, 6.0, 7.0]);
    }

    /// A ragged trailing window is averaged over its true length.
    #[test]
    fn test_average_pool_ragged_trailing_window() {
        let tensor =
            Tensor::from_vec(vec![1.0, 1.0, 3.0, 3.0, 10.0, 20.0], &[1, 3, 2]).expect("tensor");
        let pooled = average_pool_sequence(tensor, 2).expect("average pool");
        let data = pooled.data().expect("data");
        // window 0 = mean([1,1],[3,3]) = [2,2]; window 1 = [10,20] alone
        assert_eq!(data, vec![2.0, 2.0, 10.0, 20.0]);
    }

    /// Max pooling must return real per-channel maxima, not a zero buffer.
    #[test]
    fn test_max_pool_matches_hand_computed_maxima() {
        let tensor = Tensor::from_vec(
            vec![1.0, -5.0, 3.0, -1.0, -9.0, 2.0, 0.5, -0.25],
            &[1, 4, 2],
        )
        .expect("tensor");
        let pooled = max_pool_sequence(tensor, 2).expect("max pool");
        assert_eq!(pooled.shape(), vec![1, 2, 2]);
        let data = pooled.data().expect("data");
        // max([1,-5],[3,-1]) = [3,-1]; max([-9,2],[0.5,-0.25]) = [0.5, 2]
        assert_eq!(data, vec![3.0, -1.0, 0.5, 2.0]);
    }

    /// Pooling output must depend on the input: two different inputs of the same
    /// shape must not produce the same pooled tensor.
    #[test]
    fn test_pooling_output_varies_with_input() {
        for method in [
            ReductionMethod::AveragePooling,
            ReductionMethod::MaxPooling,
            ReductionMethod::AttentionPooling,
            ReductionMethod::TokenMerging,
        ] {
            let a = reduce_sequence_length(ramp_tensor(), 2, &method).expect("reduce a");
            let scaled =
                Tensor::from_vec(vec![-1.0, 4.0, 0.5, -2.0, 9.0, -3.0, 2.0, 2.5], &[1, 4, 2])
                    .expect("tensor");
            let b = reduce_sequence_length(scaled, 2, &method).expect("reduce b");
            assert_ne!(
                a.data().expect("a data"),
                b.data().expect("b data"),
                "{method:?} pooling ignored its input"
            );
            assert!(
                a.data().expect("a data").iter().any(|v| *v != 0.0),
                "{method:?} pooling returned an all-zero tensor"
            );
        }
    }

    /// Attention pooling degenerates to the identity when a window is constant.
    #[test]
    fn test_attention_pooling_constant_window_is_that_vector() {
        let tensor = Tensor::from_vec(vec![2.0, -3.0, 2.0, -3.0], &[1, 2, 2]).expect("tensor");
        let pooled = attention_pool_sequence(tensor, 2).expect("attention pool");
        let data = pooled.data().expect("data");
        assert!((data[0] - 2.0).abs() < 1e-5, "got {data:?}");
        assert!((data[1] + 3.0).abs() < 1e-5, "got {data:?}");
    }

    /// Attention pooling produces a convex combination of the window tokens, so
    /// every channel stays inside the window's min/max range.
    #[test]
    fn test_attention_pooling_is_convex_combination() {
        let pooled = attention_pool_sequence(ramp_tensor(), 4).expect("attention pool");
        let data = pooled.data().expect("data");
        assert_eq!(pooled.shape(), vec![1, 1, 2]);
        assert!(
            data[0] >= 1.0 && data[0] <= 7.0,
            "channel 0 out of hull: {data:?}"
        );
        assert!(
            data[1] >= 2.0 && data[1] <= 8.0,
            "channel 1 out of hull: {data:?}"
        );
    }

    /// Token merging of a constant window reproduces the token exactly.
    #[test]
    fn test_token_merging_constant_window() {
        let tensor =
            Tensor::from_vec(vec![4.0, -1.0, 4.0, -1.0, 4.0, -1.0], &[1, 3, 2]).expect("tensor");
        let merged = token_merge_sequence(tensor, 3).expect("token merge");
        let data = merged.data().expect("data");
        assert!((data[0] - 4.0).abs() < 1e-5, "got {data:?}");
        assert!((data[1] + 1.0).abs() < 1e-5, "got {data:?}");
    }

    /// Methods that genuinely need learned weights must report that honestly
    /// instead of silently aliasing to average pooling.
    #[test]
    fn test_weight_dependent_reductions_report_not_implemented() {
        for method in [
            ReductionMethod::LearnablePooling,
            ReductionMethod::StridedConvolution,
        ] {
            let err = reduce_sequence_length(ramp_tensor(), 2, &method);
            assert!(err.is_err(), "{method:?} must not silently fake a result");
        }
    }

    /// A zero reduction factor is rejected instead of dividing by zero.
    #[test]
    fn test_zero_reduction_factor_rejected() {
        assert!(average_pool_sequence(ramp_tensor(), 0).is_err());
        assert!(max_pool_sequence(ramp_tensor(), 0).is_err());
    }

    /// Non-3-D inputs are rejected with a shape error.
    #[test]
    fn test_pooling_rejects_non_3d_input() {
        let flat = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).expect("tensor");
        assert!(average_pool_sequence(flat, 2).is_err());
    }

    /// `build_hierarchy` must carry real pooled values into every coarser level.
    #[test]
    fn test_build_hierarchy_levels_hold_real_values() {
        let hierarchy =
            build_hierarchy(ramp_tensor(), 3, 2, ReductionMethod::AveragePooling).expect("build");
        assert_eq!(hierarchy.len(), 3);
        assert_eq!(hierarchy[0].shape(), vec![1, 4, 2]);
        assert_eq!(hierarchy[1].shape(), vec![1, 2, 2]);
        assert_eq!(hierarchy[2].shape(), vec![1, 1, 2]);
        assert_eq!(hierarchy[1].data().expect("l1"), vec![2.0, 3.0, 6.0, 7.0]);
        // Level 2 pools level 1: mean([2,3],[6,7]) = [4,5]
        assert_eq!(hierarchy[2].data().expect("l2"), vec![4.0, 5.0]);
    }

    /// Linear upsampling reproduces the endpoints and interpolates the middle.
    #[test]
    fn test_upsample_sequence_linear_interpolation() {
        let tensor = Tensor::from_vec(vec![0.0, 10.0, 4.0, 20.0], &[1, 2, 2]).expect("tensor");
        let up = upsample_sequence(tensor, 3).expect("upsample");
        assert_eq!(up.shape(), vec![1, 3, 2]);
        let data = up.data().expect("data");
        // endpoints preserved, midpoint is the average
        assert!((data[0] - 0.0).abs() < 1e-6, "{data:?}");
        assert!((data[1] - 10.0).abs() < 1e-6, "{data:?}");
        assert!((data[2] - 2.0).abs() < 1e-6, "{data:?}");
        assert!((data[3] - 15.0).abs() < 1e-6, "{data:?}");
        assert!((data[4] - 4.0).abs() < 1e-6, "{data:?}");
        assert!((data[5] - 20.0).abs() < 1e-6, "{data:?}");
    }

    /// `upsample_to_shape` must carry real data, not a zero buffer.
    #[test]
    fn test_upsample_to_shape_preserves_signal() {
        let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 2, 2]).expect("tensor");
        let up = upsample_to_shape(tensor, &[1, 4, 2]).expect("upsample");
        assert_eq!(up.shape(), vec![1, 4, 2]);
        let data = up.data().expect("data");
        assert!(data.iter().any(|v| *v != 0.0), "upsampling returned zeros");
        assert!((data[0] - 1.0).abs() < 1e-6, "{data:?}");
        assert!((data[7] - 4.0).abs() < 1e-6, "{data:?}");
    }

    /// Aggregation must combine real level values rather than zeros.
    #[test]
    fn test_aggregate_methods_produce_nonzero_output() {
        let level0 = ramp_tensor();
        let level1 = average_pool_sequence(ramp_tensor(), 2).expect("pool");
        let target = vec![1usize, 4, 2];

        for method in [
            AggregationMethod::Sum,
            AggregationMethod::WeightedSum,
            AggregationMethod::AttentionAggregation,
            AggregationMethod::GatedAggregation,
            AggregationMethod::Concatenation,
        ] {
            let out = aggregate_hierarchical_features(
                vec![level0.clone(), level1.clone()],
                &method,
                &target,
            )
            .expect("aggregate");
            let data = out.data().expect("data");
            assert!(
                data.iter().any(|v| v.abs() > 1e-6),
                "{method:?} aggregation produced an all-zero tensor"
            );
        }
    }

    /// Sum aggregation of a single level equals that level (aligned to target).
    #[test]
    fn test_aggregate_sum_single_level_is_identity() {
        let out = aggregate_hierarchical_features(
            vec![ramp_tensor()],
            &AggregationMethod::Sum,
            &[1, 4, 2],
        )
        .expect("aggregate");
        assert_eq!(
            out.data().expect("data"),
            ramp_tensor().data().expect("ref")
        );
    }

    /// Entropy must respond to the shape of the distribution rather than being a
    /// hardcoded `0.5`.
    #[test]
    fn test_attention_entropy_is_data_dependent() {
        let uniform = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 4]).expect("tensor");
        let peaked = Tensor::from_vec(vec![50.0, 0.0, 0.0, 0.0], &[1, 4]).expect("tensor");

        let uniform_entropy = compute_attention_entropy(&uniform).expect("entropy");
        let peaked_entropy = compute_attention_entropy(&peaked).expect("entropy");

        assert!(
            (uniform_entropy - 1.0).abs() < 1e-4,
            "uniform rows must have maximal normalised entropy, got {uniform_entropy}"
        );
        assert!(
            peaked_entropy < 0.05,
            "a peaked row must have near-zero entropy, got {peaked_entropy}"
        );
        assert!(uniform_entropy > peaked_entropy);
    }

    /// Sparsity must count the actual near-zero entries.
    #[test]
    fn test_attention_sparsity_is_data_dependent() {
        let half = Tensor::from_vec(vec![0.0, 1.0, 0.0, 1.0], &[1, 4]).expect("tensor");
        let dense = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 4]).expect("tensor");

        let half_sparsity = compute_attention_sparsity(&half).expect("sparsity");
        let dense_sparsity = compute_attention_sparsity(&dense).expect("sparsity");

        assert!((half_sparsity - 0.5).abs() < 1e-6, "got {half_sparsity}");
        assert!((dense_sparsity - 0.0).abs() < 1e-6, "got {dense_sparsity}");
    }

    /// The patterns API must surface the real per-level metrics.
    #[test]
    fn test_hierarchical_attention_patterns_use_real_metrics() {
        let uniform = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0], &[1, 4]).expect("tensor");
        let peaked = Tensor::from_vec(vec![50.0, 0.0, 0.0, 0.0], &[1, 4]).expect("tensor");
        let positions = vec![vec![0usize, 1, 2, 3], vec![0usize, 2]];

        let patterns =
            compute_hierarchical_attention_patterns(&[uniform, peaked], &positions).expect("pat");
        assert_eq!(patterns.len(), 2);
        assert!(
            patterns[0].attention_entropy > patterns[1].attention_entropy,
            "entropies must differ per level: {patterns:?}"
        );
        assert!(patterns[1].attention_sparsity > patterns[0].attention_sparsity);
    }

    /// A short `positions` slice must be reported, not indexed past its end.
    ///
    /// The previous implementation did `positions[level].clone()` while iterating
    /// `level_outputs`, so this call panicked instead of returning an error.
    #[test]
    fn test_attention_patterns_reject_missing_position_rows() {
        let first = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).expect("tensor");
        let second = Tensor::from_vec(vec![4.0, 3.0, 2.0, 1.0], &[1, 4]).expect("tensor");
        let positions = vec![vec![0usize, 1, 2, 3]]; // one row for two levels

        assert!(
            compute_hierarchical_attention_patterns(&[first, second], &positions).is_err(),
            "a positions slice shorter than the level list must be an error"
        );
    }

    /// Tree constructions that need external structure must report it honestly.
    #[test]
    fn test_unimplemented_tree_constructions_error() {
        assert!(create_tree_mask(8, 2, &TreeConstruction::Binary).is_ok());
        assert!(create_tree_mask(8, 2, &TreeConstruction::Balanced).is_ok());
        assert!(create_tree_mask(8, 2, &TreeConstruction::Learned).is_err());
        assert!(create_tree_mask(8, 2, &TreeConstruction::SyntaxGuided).is_err());
    }

    /// `hierarchical_pooling` applies the per-level reduction factors.
    #[test]
    fn test_hierarchical_pooling_applies_factors() {
        let pooled = hierarchical_pooling(
            vec![ramp_tensor(), ramp_tensor()],
            &ReductionMethod::AveragePooling,
            vec![2, 4],
        )
        .expect("pool");
        assert_eq!(pooled[0].shape(), vec![1, 2, 2]);
        assert_eq!(pooled[1].shape(), vec![1, 1, 2]);
        assert_eq!(pooled[1].data().expect("data"), vec![4.0, 5.0]);
    }

    /// `hierarchical_upsampling` resamples to the requested lengths with real data.
    #[test]
    fn test_hierarchical_upsampling_lengths_and_values() {
        let coarse = average_pool_sequence(ramp_tensor(), 2).expect("pool");
        let up = hierarchical_upsampling(vec![coarse], vec![4]).expect("upsample");
        assert_eq!(up[0].shape(), vec![1, 4, 2]);
        let data = up[0].data().expect("data");
        assert!(data.iter().any(|v| *v != 0.0));
        assert!((data[0] - 2.0).abs() < 1e-6, "{data:?}");
        assert!((data[7] - 7.0).abs() < 1e-6, "{data:?}");
    }
}
