//! Real weight transformations used by the compression pipeline.
//!
//! Every function here **mutates or measures actual tensor data**. There are no
//! metadata-only operations: if a routine reports a sparsity, a scale or a
//! compressed size, that number was produced by reading (and usually rewriting)
//! the weights.

use std::collections::{BinaryHeap, HashMap};

use trustformers_core::{
    errors::{Result, TrustformersError},
    tensor::Tensor,
};

/// Statistics produced by a pruning pass over one tensor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PruneStats {
    /// Number of weights in the tensor.
    pub total: usize,
    /// Number of weights that are exactly zero after pruning.
    pub zeroed: usize,
    /// Magnitude threshold below which weights were removed.
    pub threshold: f32,
}

impl PruneStats {
    /// Fraction of the tensor that is zero.
    pub fn sparsity(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.zeroed as f32 / self.total as f32
        }
    }
}

/// Affine quantization parameters actually used to round a tensor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct QuantizationParameters {
    /// Step size between representable values.
    pub scale: f32,
    /// Integer value that maps to real zero.
    pub zero_point: i32,
    /// Lowest representable integer.
    pub qmin: i32,
    /// Highest representable integer.
    pub qmax: i32,
}

impl QuantizationParameters {
    /// Quantize one value to its integer level.
    pub fn quantize(&self, value: f32) -> i32 {
        if self.scale <= 0.0 {
            return self.zero_point;
        }
        let level = (value / self.scale).round() as i64 + self.zero_point as i64;
        level.clamp(self.qmin as i64, self.qmax as i64) as i32
    }

    /// Reconstruct the real value represented by an integer level.
    pub fn dequantize(&self, level: i32) -> f32 {
        (level - self.zero_point) as f32 * self.scale
    }
}

/// Read a tensor's values, failing loudly for unsupported element types.
pub fn tensor_values(name: &str, tensor: &Tensor) -> Result<Vec<f32>> {
    tensor.data().map_err(|e| {
        TrustformersError::invalid_operation(format!("cannot read parameter `{name}`: {e}"))
    })
}

/// Write values back into a tensor, preserving its shape.
pub fn write_tensor(name: &str, tensor: &mut Tensor, values: &[f32]) -> Result<()> {
    let shape = tensor.shape();
    let expected: usize = shape.iter().product();
    if values.len() != expected {
        return Err(TrustformersError::shape_error(format!(
            "cannot write {} values into parameter `{name}` of shape {shape:?}",
            values.len()
        )));
    }
    *tensor = Tensor::from_slice(values, &shape)?;
    Ok(())
}

/// Derive affine quantization parameters from a tensor's real value range.
///
/// * `symmetric` pins zero to the middle of the range (`zero_point` = 0 for
///   signed grids), which is what most inference kernels expect for weights.
/// * `signed` selects a `[-2^(b-1), 2^(b-1)-1]` grid instead of `[0, 2^b-1]`.
pub fn derive_quantization_parameters(
    values: &[f32],
    bits: u8,
    signed: bool,
    symmetric: bool,
) -> Result<QuantizationParameters> {
    if !(1..=32).contains(&bits) {
        return Err(TrustformersError::invalid_config(format!(
            "quantization needs between 1 and 32 bits, got {bits}"
        )));
    }
    if values.is_empty() {
        return Err(TrustformersError::invalid_operation(
            "cannot derive quantization parameters from an empty tensor".to_string(),
        ));
    }

    let (qmin, qmax) = if signed {
        let half = 1i64 << (bits as i64 - 1);
        ((-half) as i32, (half - 1) as i32)
    } else {
        (0, ((1i64 << bits as i64) - 1) as i32)
    };

    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !min.is_finite() || !max.is_finite() {
        return Err(TrustformersError::invalid_operation(
            "cannot quantize a tensor that contains non-finite values".to_string(),
        ));
    }

    if symmetric {
        let bound = min.abs().max(max.abs());
        let levels = qmax.max(1) as f32;
        let scale = if bound > 0.0 { bound / levels } else { 1.0 };
        Ok(QuantizationParameters {
            scale,
            zero_point: 0,
            qmin,
            qmax,
        })
    } else {
        let span = max - min;
        let levels = (qmax - qmin).max(1) as f32;
        let scale = if span > 0.0 { span / levels } else { 1.0 };
        let zero_point = (qmin as f32 - min / scale).round() as i32;
        Ok(QuantizationParameters {
            scale,
            zero_point: zero_point.clamp(qmin, qmax),
            qmin,
            qmax,
        })
    }
}

/// Quantize a tensor to `bits` and write the reconstructed values back.
///
/// This is *fake quantization*: the tensor keeps its f32 storage but every value
/// is snapped onto the integer grid, so the numerical effect of quantization is
/// really present in the weights (and measurable) rather than merely recorded.
///
/// Returns the parameters used and the integer levels, so a caller can serialise
/// the true low-precision representation.
pub fn quantize_tensor_in_place(
    name: &str,
    tensor: &mut Tensor,
    bits: u8,
    signed: bool,
    symmetric: bool,
) -> Result<(QuantizationParameters, Vec<i32>)> {
    let values = tensor_values(name, tensor)?;
    let params = derive_quantization_parameters(&values, bits, signed, symmetric)?;

    let mut levels = Vec::with_capacity(values.len());
    let mut reconstructed = Vec::with_capacity(values.len());
    for value in &values {
        let level = params.quantize(*value);
        levels.push(level);
        reconstructed.push(params.dequantize(level));
    }

    write_tensor(name, tensor, &reconstructed)?;
    Ok((params, levels))
}

/// Zero the smallest-magnitude weights of a tensor until `sparsity` is reached.
///
/// `sparsity` is the fraction of weights to remove, in `[0, 1)`. Weights already
/// at zero count towards the target.
pub fn magnitude_prune_in_place(
    name: &str,
    tensor: &mut Tensor,
    sparsity: f32,
) -> Result<PruneStats> {
    let threshold = magnitude_threshold_for(&tensor_values(name, tensor)?, sparsity)?;
    prune_below_threshold(name, tensor, threshold)
}

/// Compute the magnitude threshold that removes `sparsity` of the values.
pub fn magnitude_threshold_for(values: &[f32], sparsity: f32) -> Result<f32> {
    if !(0.0..1.0).contains(&sparsity) {
        return Err(TrustformersError::invalid_config(format!(
            "sparsity must lie in [0, 1), got {sparsity}"
        )));
    }
    if values.is_empty() {
        return Ok(0.0);
    }

    let mut magnitudes: Vec<f32> = values.iter().map(|v| v.abs()).collect();
    magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let cut = ((magnitudes.len() as f32) * sparsity).round() as usize;
    if cut == 0 {
        return Ok(0.0);
    }
    let index = cut.min(magnitudes.len()) - 1;
    // Anything with |w| <= threshold is removed.
    Ok(magnitudes[index])
}

/// Zero every weight whose magnitude is at or below `threshold`.
pub fn prune_below_threshold(
    name: &str,
    tensor: &mut Tensor,
    threshold: f32,
) -> Result<PruneStats> {
    let mut values = tensor_values(name, tensor)?;
    let total = values.len();
    let mut zeroed = 0usize;

    for value in &mut values {
        if value.abs() <= threshold {
            *value = 0.0;
        }
        if *value == 0.0 {
            zeroed += 1;
        }
    }

    write_tensor(name, tensor, &values)?;
    Ok(PruneStats {
        total,
        zeroed,
        threshold,
    })
}

/// Zero a random `sparsity` fraction of the weights (the pruning baseline).
///
/// The RNG is seeded by the caller so results are reproducible.
pub fn random_prune_in_place(
    name: &str,
    tensor: &mut Tensor,
    sparsity: f32,
    rng: &mut impl FnMut() -> f32,
) -> Result<PruneStats> {
    if !(0.0..1.0).contains(&sparsity) {
        return Err(TrustformersError::invalid_config(format!(
            "sparsity must lie in [0, 1), got {sparsity}"
        )));
    }

    let mut values = tensor_values(name, tensor)?;
    let total = values.len();
    let target = ((total as f32) * sparsity).round() as usize;

    // Random selection without replacement: partial Fisher-Yates over indices.
    let mut indices: Vec<usize> = (0..total).collect();
    for i in 0..target.min(total) {
        let remaining = total - i;
        let pick = i + ((rng() * remaining as f32) as usize).min(remaining - 1);
        indices.swap(i, pick);
        values[indices[i]] = 0.0;
    }

    let zeroed = values.iter().filter(|v| **v == 0.0).count();
    write_tensor(name, tensor, &values)?;
    Ok(PruneStats {
        total,
        zeroed,
        threshold: 0.0,
    })
}

/// Which axis of a 2-D weight matrix a structured pruning pass removes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructureAxis {
    /// Remove output units (rows of a `[out, in]` weight).
    Rows,
    /// Remove input units (columns of a `[out, in]` weight).
    Columns,
}

/// Zero whole rows or columns of a 2-D weight matrix, choosing the least
/// important ones by their L1 or L2 norm.
///
/// Returns the indices that were removed.
pub fn structured_prune_in_place(
    name: &str,
    tensor: &mut Tensor,
    ratio: f32,
    axis: StructureAxis,
    use_l1: bool,
) -> Result<Vec<usize>> {
    if !(0.0..1.0).contains(&ratio) {
        return Err(TrustformersError::invalid_config(format!(
            "structured pruning ratio must lie in [0, 1), got {ratio}"
        )));
    }

    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(TrustformersError::shape_error(format!(
            "structured pruning needs a rank-2 weight, but `{name}` has shape {shape:?}"
        )));
    }
    let (rows, columns) = (shape[0], shape[1]);
    let mut values = tensor_values(name, tensor)?;

    let count = match axis {
        StructureAxis::Rows => rows,
        StructureAxis::Columns => columns,
    };

    // Importance of each structure.
    let mut importance: Vec<(usize, f32)> = Vec::with_capacity(count);
    for index in 0..count {
        let mut score = 0.0f32;
        match axis {
            StructureAxis::Rows => {
                for column in 0..columns {
                    let value = values[index * columns + column];
                    score += if use_l1 { value.abs() } else { value * value };
                }
            },
            StructureAxis::Columns => {
                for row in 0..rows {
                    let value = values[row * columns + index];
                    score += if use_l1 { value.abs() } else { value * value };
                }
            },
        }
        importance.push((index, if use_l1 { score } else { score.sqrt() }));
    }

    importance.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let remove_count = ((count as f32) * ratio).round() as usize;
    let removed: Vec<usize> = importance
        .iter()
        .take(remove_count.min(count))
        .map(|(index, _)| *index)
        .collect();

    for &index in &removed {
        match axis {
            StructureAxis::Rows => {
                for column in 0..columns {
                    values[index * columns + column] = 0.0;
                }
            },
            StructureAxis::Columns => {
                for row in 0..rows {
                    values[row * columns + index] = 0.0;
                }
            },
        }
    }

    write_tensor(name, tensor, &values)?;
    Ok(removed)
}

/// Replace every weight by the nearest of `k` centroids found with Lloyd's
/// algorithm (1-D k-means), initialised deterministically on the value range.
///
/// Returns the centroids and, for each weight, the index of its centroid — the
/// pair that a real clustered-weight format stores.
pub fn cluster_weights_in_place(
    name: &str,
    tensor: &mut Tensor,
    clusters: usize,
    iterations: usize,
) -> Result<(Vec<f32>, Vec<usize>)> {
    if clusters == 0 {
        return Err(TrustformersError::invalid_config(
            "weight clustering needs at least one cluster".to_string(),
        ));
    }

    let mut values = tensor_values(name, tensor)?;
    if values.is_empty() {
        return Err(TrustformersError::invalid_operation(format!(
            "cannot cluster the empty parameter `{name}`"
        )));
    }

    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !min.is_finite() || !max.is_finite() {
        return Err(TrustformersError::invalid_operation(format!(
            "parameter `{name}` contains non-finite values"
        )));
    }

    let effective_clusters = clusters.min(values.len());
    // Deterministic linear initialisation across the observed range.
    let mut centroids: Vec<f32> = (0..effective_clusters)
        .map(|i| {
            if effective_clusters == 1 {
                (min + max) / 2.0
            } else {
                min + (max - min) * i as f32 / (effective_clusters - 1) as f32
            }
        })
        .collect();

    let mut assignments = vec![0usize; values.len()];
    for _ in 0..iterations.max(1) {
        let mut changed = false;

        // Assignment step.
        for (index, value) in values.iter().enumerate() {
            let mut best = 0usize;
            let mut best_distance = f32::INFINITY;
            for (cluster, centroid) in centroids.iter().enumerate() {
                let distance = (value - centroid).abs();
                if distance < best_distance {
                    best_distance = distance;
                    best = cluster;
                }
            }
            if assignments[index] != best {
                assignments[index] = best;
                changed = true;
            }
        }

        // Update step.
        let mut sums = vec![0.0f64; effective_clusters];
        let mut counts = vec![0usize; effective_clusters];
        for (index, value) in values.iter().enumerate() {
            sums[assignments[index]] += f64::from(*value);
            counts[assignments[index]] += 1;
        }
        for cluster in 0..effective_clusters {
            if counts[cluster] > 0 {
                centroids[cluster] = (sums[cluster] / counts[cluster] as f64) as f32;
            }
        }

        if !changed {
            break;
        }
    }

    for (index, value) in values.iter_mut().enumerate() {
        *value = centroids[assignments[index]];
    }
    write_tensor(name, tensor, &values)?;

    Ok((centroids, assignments))
}

/// Replace a 2-D weight with its best rank-`rank` approximation.
///
/// Uses a randomized range finder: `Y = W Ω` for a fixed pseudo-random `Ω`,
/// orthonormalised with modified Gram-Schmidt, followed by `W ≈ Q (Qᵀ W)`. Power
/// iterations sharpen the range estimate for slowly decaying spectra.
///
/// Returns the relative Frobenius error of the approximation — a real measure of
/// what the decomposition cost.
pub fn low_rank_approximate_in_place(
    name: &str,
    tensor: &mut Tensor,
    rank: usize,
    power_iterations: usize,
) -> Result<f32> {
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(TrustformersError::shape_error(format!(
            "low-rank decomposition needs a rank-2 weight, but `{name}` has shape {shape:?}"
        )));
    }
    let (rows, columns) = (shape[0], shape[1]);
    let target_rank = rank.clamp(1, rows.min(columns));

    let original = tensor_values(name, tensor)?;

    // Deterministic pseudo-random test matrix (columns x target_rank).
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 * 2.0 - 1.0
    };
    let mut omega = vec![0.0f32; columns * target_rank];
    for value in &mut omega {
        *value = next();
    }

    // Y = W * Omega  (rows x target_rank)
    let mut y = matmul(&original, rows, columns, &omega, columns, target_rank);
    orthonormalize_columns(&mut y, rows, target_rank);

    for _ in 0..power_iterations {
        // Z = W^T Y  (columns x k), then Y = W Z
        let z = matmul_transpose_a(&original, rows, columns, &y, rows, target_rank);
        y = matmul(&original, rows, columns, &z, columns, target_rank);
        orthonormalize_columns(&mut y, rows, target_rank);
    }

    // B = Q^T W  (k x columns); W_k = Q B
    let b = matmul_transpose_a(&y, rows, target_rank, &original, rows, columns);
    let approximation = matmul(&y, rows, target_rank, &b, target_rank, columns);

    let mut error_norm = 0.0f64;
    let mut original_norm = 0.0f64;
    for (a, b) in original.iter().zip(approximation.iter()) {
        let difference = f64::from(a - b);
        error_norm += difference * difference;
        original_norm += f64::from(*a) * f64::from(*a);
    }

    write_tensor(name, tensor, &approximation)?;

    Ok(if original_norm > 0.0 {
        (error_norm.sqrt() / original_norm.sqrt()) as f32
    } else {
        0.0
    })
}

/// Dense row-major matrix product `A (m x k) * B (k x n)`.
fn matmul(a: &[f32], m: usize, k: usize, b: &[f32], k2: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(k, k2);
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for inner in 0..k {
            let a_value = a[row * k + inner];
            if a_value == 0.0 {
                continue;
            }
            for column in 0..n {
                out[row * n + column] += a_value * b[inner * n + column];
            }
        }
    }
    out
}

/// `Aᵀ (k x m) * B (k x n)` for row-major `A (k x m)` and `B (k x n)`.
fn matmul_transpose_a(a: &[f32], k: usize, m: usize, b: &[f32], k2: usize, n: usize) -> Vec<f32> {
    debug_assert_eq!(k, k2);
    let mut out = vec![0.0f32; m * n];
    for inner in 0..k {
        for row in 0..m {
            let a_value = a[inner * m + row];
            if a_value == 0.0 {
                continue;
            }
            for column in 0..n {
                out[row * n + column] += a_value * b[inner * n + column];
            }
        }
    }
    out
}

/// Modified Gram-Schmidt orthonormalisation of the columns of a row-major matrix.
fn orthonormalize_columns(matrix: &mut [f32], rows: usize, columns: usize) {
    for column in 0..columns {
        // Subtract the projection onto every previously orthonormalised column.
        for previous in 0..column {
            let mut dot = 0.0f32;
            for row in 0..rows {
                dot += matrix[row * columns + previous] * matrix[row * columns + column];
            }
            for row in 0..rows {
                matrix[row * columns + column] -= dot * matrix[row * columns + previous];
            }
        }

        let mut norm = 0.0f32;
        for row in 0..rows {
            let value = matrix[row * columns + column];
            norm += value * value;
        }
        let norm = norm.sqrt();
        if norm > 1e-12 {
            for row in 0..rows {
                matrix[row * columns + column] /= norm;
            }
        }
    }
}

/// A canonical Huffman code plus the bitstream it produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HuffmanEncoded {
    /// `(symbol, code length in bits)` pairs, sorted canonically.
    pub code_lengths: Vec<(u8, u8)>,
    /// Packed bitstream, most-significant bit first.
    pub bitstream: Vec<u8>,
    /// Number of meaningful bits in `bitstream`.
    pub bit_length: usize,
    /// Number of symbols encoded.
    pub symbol_count: usize,
}

impl HuffmanEncoded {
    /// Size of the encoded payload in bytes (excluding the code table).
    pub fn payload_bytes(&self) -> usize {
        self.bit_length.div_ceil(8)
    }

    /// Size of the serialised code table in bytes (`symbol` + `length` per entry).
    pub fn table_bytes(&self) -> usize {
        self.code_lengths.len() * 2
    }

    /// Total compressed size in bytes, table included.
    pub fn total_bytes(&self) -> usize {
        self.payload_bytes() + self.table_bytes()
    }
}

/// Node of the Huffman tree used while building the code.
#[derive(Debug, Eq, PartialEq)]
struct HuffmanNode {
    frequency: usize,
    /// Smallest symbol under this node — makes the tie-break deterministic.
    tie_break: u8,
    symbols: Vec<u8>,
}

impl Ord for HuffmanNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is a max-heap; invert so the smallest frequency pops first.
        other
            .frequency
            .cmp(&self.frequency)
            .then_with(|| other.tie_break.cmp(&self.tie_break))
    }
}

impl PartialOrd for HuffmanNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Huffman-encode a symbol stream with a canonical code.
///
/// # Errors
///
/// Returns an error for an empty input: there is no code to build and no
/// compression ratio to report.
pub fn huffman_encode(symbols: &[u8]) -> Result<HuffmanEncoded> {
    if symbols.is_empty() {
        return Err(TrustformersError::invalid_operation(
            "cannot Huffman-encode an empty symbol stream".to_string(),
        ));
    }

    let mut frequencies = [0usize; 256];
    for &symbol in symbols {
        frequencies[symbol as usize] += 1;
    }

    let distinct: Vec<u8> = (0..=255u8).filter(|s| frequencies[*s as usize] > 0).collect();

    // Code lengths from the Huffman tree: merge the two lightest nodes, adding a
    // bit to every symbol underneath them.
    let mut lengths: HashMap<u8, u8> = HashMap::new();
    if distinct.len() == 1 {
        // Degenerate alphabet: a single one-bit code.
        lengths.insert(distinct[0], 1);
    } else {
        let mut heap: BinaryHeap<HuffmanNode> = distinct
            .iter()
            .map(|&symbol| HuffmanNode {
                frequency: frequencies[symbol as usize],
                tie_break: symbol,
                symbols: vec![symbol],
            })
            .collect();

        for &symbol in &distinct {
            lengths.insert(symbol, 0);
        }

        while heap.len() > 1 {
            let (Some(left), Some(right)) = (heap.pop(), heap.pop()) else {
                break;
            };
            for symbol in left.symbols.iter().chain(right.symbols.iter()) {
                if let Some(length) = lengths.get_mut(symbol) {
                    *length = length.saturating_add(1);
                }
            }
            let mut symbols = left.symbols;
            symbols.extend(right.symbols);
            heap.push(HuffmanNode {
                frequency: left.frequency + right.frequency,
                tie_break: left.tie_break.min(right.tie_break),
                symbols,
            });
        }
    }

    // Canonical code assignment: sort by (length, symbol) and count up.
    let mut code_lengths: Vec<(u8, u8)> =
        lengths.into_iter().map(|(symbol, length)| (symbol, length.max(1))).collect();
    code_lengths.sort_by_key(|(symbol, length)| (*length, *symbol));

    let mut codes: HashMap<u8, (u32, u8)> = HashMap::new();
    let mut code: u32 = 0;
    let mut previous_length = 0u8;
    for &(symbol, length) in &code_lengths {
        if previous_length == 0 {
            previous_length = length;
        } else if length > previous_length {
            code <<= length - previous_length;
            previous_length = length;
        }
        codes.insert(symbol, (code, length));
        code += 1;
    }

    // Pack the bitstream, most-significant bit first.
    let mut bitstream = Vec::with_capacity(symbols.len());
    let mut current = 0u8;
    let mut used = 0u8;
    let mut bit_length = 0usize;
    for &symbol in symbols {
        let (code, length) = codes.get(&symbol).copied().ok_or_else(|| {
            TrustformersError::invalid_operation(format!("symbol {symbol} has no Huffman code"))
        })?;
        for bit_index in (0..length).rev() {
            let bit = ((code >> bit_index) & 1) as u8;
            current = (current << 1) | bit;
            used += 1;
            bit_length += 1;
            if used == 8 {
                bitstream.push(current);
                current = 0;
                used = 0;
            }
        }
    }
    if used > 0 {
        bitstream.push(current << (8 - used));
    }

    Ok(HuffmanEncoded {
        code_lengths,
        bitstream,
        bit_length,
        symbol_count: symbols.len(),
    })
}

/// Decode a canonical-Huffman bitstream back into its symbols.
pub fn huffman_decode(encoded: &HuffmanEncoded) -> Result<Vec<u8>> {
    if encoded.code_lengths.is_empty() {
        return Err(TrustformersError::invalid_operation(
            "the Huffman code table is empty".to_string(),
        ));
    }

    // Rebuild the canonical codes exactly as the encoder assigned them.
    let mut lookup: HashMap<(u8, u32), u8> = HashMap::new();
    let mut code: u32 = 0;
    let mut previous_length = 0u8;
    for &(symbol, length) in &encoded.code_lengths {
        if previous_length == 0 {
            previous_length = length;
        } else if length > previous_length {
            code <<= length - previous_length;
            previous_length = length;
        }
        lookup.insert((length, code), symbol);
        code += 1;
    }

    let mut decoded = Vec::with_capacity(encoded.symbol_count);
    let mut current_code: u32 = 0;
    let mut current_length: u8 = 0;

    for bit_index in 0..encoded.bit_length {
        let byte = encoded.bitstream.get(bit_index / 8).copied().ok_or_else(|| {
            TrustformersError::invalid_operation(
                "the Huffman bitstream is shorter than its declared bit length".to_string(),
            )
        })?;
        let bit = (byte >> (7 - (bit_index % 8))) & 1;
        current_code = (current_code << 1) | u32::from(bit);
        current_length = current_length.saturating_add(1);

        if let Some(&symbol) = lookup.get(&(current_length, current_code)) {
            decoded.push(symbol);
            current_code = 0;
            current_length = 0;
            if decoded.len() == encoded.symbol_count {
                break;
            }
        }

        if current_length > 32 {
            return Err(TrustformersError::invalid_operation(
                "the Huffman bitstream contains a code longer than 32 bits".to_string(),
            ));
        }
    }

    if decoded.len() != encoded.symbol_count {
        return Err(TrustformersError::invalid_operation(format!(
            "the Huffman bitstream decoded to {} symbols but {} were encoded",
            decoded.len(),
            encoded.symbol_count
        )));
    }

    Ok(decoded)
}

/// Map quantization levels onto byte symbols for entropy coding.
///
/// Only grids of at most 256 levels can be byte-coded; anything wider is an
/// error rather than a silently truncated stream.
pub fn levels_to_symbols(levels: &[i32], params: &QuantizationParameters) -> Result<Vec<u8>> {
    let span = i64::from(params.qmax) - i64::from(params.qmin) + 1;
    if span > 256 {
        return Err(TrustformersError::invalid_config(format!(
            "byte-level entropy coding supports at most 256 quantization levels, got {span}"
        )));
    }

    levels
        .iter()
        .map(|level| {
            let offset = i64::from(*level) - i64::from(params.qmin);
            u8::try_from(offset).map_err(|_| {
                TrustformersError::invalid_operation(format!(
                    "quantization level {level} is outside the coded range"
                ))
            })
        })
        .collect()
}

#[cfg(test)]
#[path = "weight_ops_tests.rs"]
mod tests;
