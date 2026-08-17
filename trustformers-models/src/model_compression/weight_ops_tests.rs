//! Tests for the real weight transformations.

use super::*;

fn tensor_from(values: &[f32], shape: &[usize]) -> Tensor {
    Tensor::from_slice(values, shape).expect("tensor creation must succeed")
}

#[test]
fn test_magnitude_pruning_zeroes_the_smallest_weights() {
    let mut tensor = tensor_from(&[0.1, -0.9, 0.2, 0.8, -0.05, 0.6], &[2, 3]);

    let stats = magnitude_prune_in_place("w", &mut tensor, 0.5).expect("pruning must succeed");

    assert_eq!(stats.total, 6);
    assert_eq!(stats.zeroed, 3, "half of six weights must be removed");
    assert!((stats.sparsity() - 0.5).abs() < 1e-6);

    let values = tensor.data().expect("weights");
    // The three smallest magnitudes (0.05, 0.1, 0.2) are gone, the rest survive.
    assert_eq!(values[0], 0.0);
    assert_eq!(values[2], 0.0);
    assert_eq!(values[4], 0.0);
    assert!((values[1] + 0.9).abs() < 1e-6);
    assert!((values[3] - 0.8).abs() < 1e-6);
    assert!((values[5] - 0.6).abs() < 1e-6);
}

#[test]
fn test_magnitude_pruning_with_zero_sparsity_changes_nothing() {
    let original = [0.1f32, -0.9, 0.2];
    let mut tensor = tensor_from(&original, &[3]);
    let stats = magnitude_prune_in_place("w", &mut tensor, 0.0).expect("pruning");
    assert_eq!(stats.zeroed, 0);
    assert_eq!(tensor.data().expect("weights"), original.to_vec());
}

#[test]
fn test_pruning_rejects_impossible_sparsity() {
    let mut tensor = tensor_from(&[1.0, 2.0], &[2]);
    assert!(magnitude_prune_in_place("w", &mut tensor, 1.0).is_err());
    assert!(magnitude_prune_in_place("w", &mut tensor, -0.1).is_err());
}

#[test]
fn test_random_pruning_removes_the_requested_fraction() {
    let mut tensor = tensor_from(&[1.0; 10], &[10]);
    let mut state = 12345u64;
    let mut rng = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as f32) / (u32::MAX as f32 / 2.0)
    };

    let stats = random_prune_in_place("w", &mut tensor, 0.3, &mut rng).expect("pruning");
    assert_eq!(stats.total, 10);
    assert_eq!(stats.zeroed, 3);
    assert_eq!(
        tensor.data().expect("weights").iter().filter(|v| **v == 0.0).count(),
        3
    );
}

#[test]
fn test_quantization_snaps_weights_onto_the_grid() {
    let mut tensor = tensor_from(&[-1.0, -0.5, 0.0, 0.25, 1.0], &[5]);
    let original = tensor.data().expect("weights");

    let (params, levels) =
        quantize_tensor_in_place("w", &mut tensor, 4, true, true).expect("quantization");

    assert_eq!(params.qmin, -8);
    assert_eq!(params.qmax, 7);
    assert_eq!(levels.len(), 5);

    let quantized = tensor.data().expect("weights");
    assert_ne!(quantized, original, "quantization must change the weights");

    // Every value lands exactly on a grid point and stays close to the original.
    for (value, level) in quantized.iter().zip(levels.iter()) {
        let expected = params.dequantize(*level);
        assert!((value - expected).abs() < 1e-6);
    }
    for (before, after) in original.iter().zip(quantized.iter()) {
        assert!(
            (before - after).abs() <= params.scale,
            "quantization error {} exceeds the step size {}",
            (before - after).abs(),
            params.scale
        );
    }
}

#[test]
fn test_quantization_error_shrinks_with_more_bits() {
    let values: Vec<f32> = (0..64).map(|i| (i as f32 / 63.0) * 2.0 - 1.0).collect();

    let mut coarse = tensor_from(&values, &[64]);
    quantize_tensor_in_place("w", &mut coarse, 2, true, true).expect("quantization");
    let coarse_error: f32 = coarse
        .data()
        .expect("weights")
        .iter()
        .zip(values.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    let mut fine = tensor_from(&values, &[64]);
    quantize_tensor_in_place("w", &mut fine, 8, true, true).expect("quantization");
    let fine_error: f32 = fine
        .data()
        .expect("weights")
        .iter()
        .zip(values.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(
        fine_error < coarse_error,
        "8-bit quantization ({fine_error}) must beat 2-bit ({coarse_error})"
    );
}

#[test]
fn test_asymmetric_quantization_covers_a_shifted_range() {
    let values = [10.0f32, 11.0, 12.0, 13.0];
    let mut tensor = tensor_from(&values, &[4]);
    quantize_tensor_in_place("w", &mut tensor, 8, false, false).expect("quantization");

    let quantized = tensor.data().expect("weights");
    for (before, after) in values.iter().zip(quantized.iter()) {
        assert!(
            (before - after).abs() < 0.05,
            "8-bit asymmetric quantization of {before} produced {after}"
        );
    }
}

#[test]
fn test_quantization_rejects_impossible_grids() {
    let mut tensor = tensor_from(&[1.0, 2.0], &[2]);
    assert!(quantize_tensor_in_place("w", &mut tensor, 0, true, true).is_err());
    assert!(quantize_tensor_in_place("w", &mut tensor, 64, true, true).is_err());

    let mut broken = tensor_from(&[f32::NAN, 1.0], &[2]);
    assert!(quantize_tensor_in_place("w", &mut broken, 8, true, true).is_err());
}

#[test]
fn test_structured_pruning_removes_whole_rows() {
    // Row 1 is tiny, so it is the one that must go.
    let mut tensor = tensor_from(&[1.0, 1.0, 0.01, 0.01, 2.0, 2.0], &[3, 2]);

    let removed = structured_prune_in_place("w", &mut tensor, 0.34, StructureAxis::Rows, false)
        .expect("structured pruning");

    assert_eq!(removed, vec![1]);
    let values = tensor.data().expect("weights");
    assert_eq!(&values[2..4], &[0.0, 0.0]);
    assert!(values[0] != 0.0 && values[4] != 0.0);
}

#[test]
fn test_structured_pruning_removes_whole_columns() {
    let mut tensor = tensor_from(&[1.0, 0.01, 2.0, 0.02], &[2, 2]);
    let removed = structured_prune_in_place("w", &mut tensor, 0.5, StructureAxis::Columns, true)
        .expect("structured pruning");

    assert_eq!(removed, vec![1]);
    let values = tensor.data().expect("weights");
    assert_eq!(values[1], 0.0);
    assert_eq!(values[3], 0.0);
    assert!(values[0] != 0.0 && values[2] != 0.0);
}

#[test]
fn test_structured_pruning_requires_a_matrix() {
    let mut tensor = tensor_from(&[1.0, 2.0, 3.0], &[3]);
    assert!(structured_prune_in_place("w", &mut tensor, 0.5, StructureAxis::Rows, false).is_err());
}

#[test]
fn test_weight_clustering_collapses_values_onto_centroids() {
    let mut tensor = tensor_from(&[0.0, 0.05, 1.0, 0.95, -1.0, -0.9], &[6]);

    let (centroids, assignments) =
        cluster_weights_in_place("w", &mut tensor, 3, 20).expect("clustering");

    assert_eq!(centroids.len(), 3);
    assert_eq!(assignments.len(), 6);

    let values = tensor.data().expect("weights");
    let distinct: std::collections::BTreeSet<u32> = values.iter().map(|v| v.to_bits()).collect();
    assert!(
        distinct.len() <= 3,
        "clustered weights must take at most 3 distinct values, got {}",
        distinct.len()
    );

    // Nearby weights end up sharing a centroid.
    assert_eq!(assignments[0], assignments[1]);
    assert_eq!(assignments[2], assignments[3]);
    assert_eq!(assignments[4], assignments[5]);
}

#[test]
fn test_low_rank_approximation_is_exact_for_a_rank_one_matrix() {
    // Outer product => exactly rank 1.
    let rows = [1.0f32, 2.0, 3.0];
    let columns = [1.0f32, -1.0, 2.0, 0.5];
    let mut values = Vec::new();
    for r in rows.iter() {
        for c in columns.iter() {
            values.push(r * c);
        }
    }
    let mut tensor = tensor_from(&values, &[3, 4]);

    let error = low_rank_approximate_in_place("w", &mut tensor, 1, 2).expect("decomposition");

    assert!(
        error < 1e-4,
        "a rank-1 matrix must be reconstructed exactly, relative error {error}"
    );
    let reconstructed = tensor.data().expect("weights");
    for (before, after) in values.iter().zip(reconstructed.iter()) {
        assert!((before - after).abs() < 1e-3, "{before} vs {after}");
    }
}

#[test]
fn test_low_rank_approximation_loses_information_below_full_rank() {
    // Identity-like matrix has full rank: a rank-1 approximation must be lossy.
    let values = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let mut tensor = tensor_from(&values, &[3, 3]);

    let error = low_rank_approximate_in_place("w", &mut tensor, 1, 3).expect("decomposition");
    assert!(
        error > 0.1,
        "a rank-1 approximation of the identity must be visibly lossy, got {error}"
    );

    let mut full = tensor_from(&values, &[3, 3]);
    let full_error = low_rank_approximate_in_place("w", &mut full, 3, 3).expect("decomposition");
    assert!(
        full_error < error,
        "a higher rank must approximate better: {full_error} vs {error}"
    );
}

#[test]
fn test_huffman_round_trip() {
    let symbols: Vec<u8> = "the quick brown fox jumps over the lazy dog"
        .bytes()
        .chain(std::iter::repeat_n(b'e', 40))
        .collect();

    let encoded = huffman_encode(&symbols).expect("encoding");
    let decoded = huffman_decode(&encoded).expect("decoding");

    assert_eq!(decoded, symbols, "Huffman coding must round-trip exactly");
    assert_eq!(encoded.symbol_count, symbols.len());
    assert!(
        encoded.payload_bytes() < symbols.len(),
        "a skewed distribution must compress: {} bytes vs {}",
        encoded.payload_bytes(),
        symbols.len()
    );
}

#[test]
fn test_huffman_round_trip_for_a_single_symbol() {
    let symbols = vec![7u8; 16];
    let encoded = huffman_encode(&symbols).expect("encoding");
    assert_eq!(
        encoded.bit_length, 16,
        "a one-symbol alphabet needs one bit each"
    );
    assert_eq!(huffman_decode(&encoded).expect("decoding"), symbols);
}

#[test]
fn test_huffman_round_trip_for_uniform_data() {
    let symbols: Vec<u8> = (0..=255u8).collect();
    let encoded = huffman_encode(&symbols).expect("encoding");
    assert_eq!(huffman_decode(&encoded).expect("decoding"), symbols);
    // A flat distribution over 256 symbols cannot be compressed below 8 bits each.
    assert!(encoded.bit_length >= symbols.len() * 8);
}

#[test]
fn test_huffman_rejects_empty_input() {
    assert!(huffman_encode(&[]).is_err());
}

#[test]
fn test_quantized_weights_can_be_entropy_coded() {
    // Weights concentrated near zero quantize to a skewed symbol distribution.
    let values: Vec<f32> = (0..512).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
    let mut tensor = tensor_from(&values, &[512]);

    let (params, levels) =
        quantize_tensor_in_place("w", &mut tensor, 8, true, true).expect("quantization");
    let symbols = levels_to_symbols(&levels, &params).expect("symbol mapping");
    let encoded = huffman_encode(&symbols).expect("encoding");

    assert_eq!(huffman_decode(&encoded).expect("decoding"), symbols);
    assert!(
        encoded.total_bytes() < symbols.len(),
        "entropy coding must shrink a skewed stream: {} vs {}",
        encoded.total_bytes(),
        symbols.len()
    );
}

#[test]
fn test_levels_to_symbols_rejects_wide_grids() {
    let params = QuantizationParameters {
        scale: 1.0,
        zero_point: 0,
        qmin: -32768,
        qmax: 32767,
    };
    assert!(levels_to_symbols(&[0, 1], &params).is_err());
}
