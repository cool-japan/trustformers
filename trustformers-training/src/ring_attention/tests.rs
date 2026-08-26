//! Tests for ring attention: KV rotation, block-sparse attention and the
//! communication-volume accounting.

use super::*;

#[test]
fn test_ring_attention_config() {
    let config = RingAttentionConfig::default();
    assert_eq!(config.num_devices, 8);
    assert_eq!(config.chunk_size, 4096);
    assert!(config.bidirectional);
}

#[test]
fn test_ring_attention_manager_creation() {
    let config = RingAttentionConfig::default();
    let sequence_length = 32768;
    let manager =
        RingAttentionManager::new(config, sequence_length).expect("operation failed in test");

    assert_eq!(manager.devices.len(), 8);
    assert_eq!(manager.global_sequence_length, sequence_length);

    // Check device chunk assignments
    for (i, device) in manager.devices.iter().enumerate() {
        assert_eq!(device.device_rank, i);
        let expected_start = i * 4096;
        assert_eq!(device.sequence_chunk.0, expected_start);
    }
}

#[test]
fn test_optimal_device_calculation() {
    let devices = utils::calculate_optimal_devices(1_000_000, 4096);
    assert!(devices > 0);
    assert!(devices <= 128);

    // Should prefer power-of-2 device counts
    assert!([1, 2, 4, 8, 16, 32, 64, 128].contains(&devices));
}

#[test]
fn test_speedup_estimation() {
    let speedup = utils::estimate_speedup(1_000_000, 32, 900.0);
    assert!(speedup > 1.0);
    assert!(speedup <= 32.0); // Can't exceed number of devices
}

#[test]
fn test_preset_configs() {
    let presets = utils::create_preset_configs();
    assert!(presets.contains_key("small_scale"));
    assert!(presets.contains_key("medium_scale"));
    assert!(presets.contains_key("large_scale"));
    assert!(presets.contains_key("ultra_scale"));

    let ultra_config = &presets["ultra_scale"];
    assert_eq!(ultra_config.num_devices, 128);
    assert!(ultra_config.compression_enabled);
}

#[test]
fn test_model_params_memory_estimation() {
    let params = ModelParams {
        num_heads: 32,
        head_dim: 128,
        hidden_dim: 4096,
        num_layers: 24,
        causal: true,
    };

    let memory = params.estimate_memory_usage();
    assert!(memory > 0);
    // Should be reasonable for a large model (several GB)
    assert!(memory > 1_000_000_000); // > 1GB
}

#[test]
fn test_ring_attention_stats() {
    let mut stats = RingAttentionStats {
        total_attention_ops: 1000,
        computation_time_ms: 100.0,
        communication_time_ms: 20.0,
        ..RingAttentionStats::default()
    };

    // Compute efficiency: computation / total time
    let total_time = stats.computation_time_ms + stats.communication_time_ms;
    stats.efficiency_score = (stats.computation_time_ms / total_time) as f32;

    let expected_efficiency = 100.0 / 120.0;
    assert!((stats.efficiency_score - expected_efficiency as f32).abs() < 0.01);
}

#[test]
fn test_optimized_config_creation() {
    let model_params = ModelParams {
        num_heads: 32,
        head_dim: 128,
        hidden_dim: 4096,
        num_layers: 24,
        causal: true,
    };

    let config = RingAttentionManager::create_optimized_config(
        2_000_000, // 2M tokens
        16,        // 16 devices
        model_params,
    );

    assert_eq!(config.num_devices, 16);
    assert!(config.compression_enabled); // Should enable for 2M tokens
    assert!(config.chunk_size > 0);
}

/// Dense scaled dot-product attention, written the obvious way, used as the
/// reference for the tiled implementation.
fn naive_attention(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    seq_len: usize,
    hidden: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_len * hidden];
    for i in 0..seq_len {
        let last = if causal { i + 1 } else { seq_len };
        let mut scores = Vec::with_capacity(last);
        for j in 0..last {
            let dot: f32 =
                (0..hidden).map(|d| queries[i * hidden + d] * keys[j * hidden + d]).sum::<f32>()
                    * scale;
            scores.push(dot);
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exponentials: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let total: f32 = exponentials.iter().sum();
        for (j, weight) in exponentials.iter().enumerate() {
            for d in 0..hidden {
                output[i * hidden + d] += (weight / total) * values[j * hidden + d];
            }
        }
    }
    output
}

fn block_sparse_manager(causal: bool, hidden: usize) -> RingAttentionManager {
    let config = RingAttentionConfig {
        num_devices: 1,
        chunk_size: 8,
        head_dim: hidden,
        causal,
        ..RingAttentionConfig::default()
    };
    RingAttentionManager::new(config, 8).expect("manager must build in test")
}

/// Regression: the block-sparse path used to accumulate into a discarded
/// temporary and return the freshly zeroed output tensor, so it produced
/// all-zero attention for every input. It must now match dense attention.
#[test]
fn block_sparse_attention_matches_dense_reference() {
    let seq_len = 8usize;
    let hidden = 4usize;
    let make = |seed: f32| -> Vec<f32> {
        (0..seq_len * hidden)
            .map(|i| ((i as f32 * 0.37 + seed).sin() * 0.9) + seed * 0.1)
            .collect()
    };
    let queries = make(0.2);
    let keys = make(1.1);
    let values = make(2.3);
    let scale = 1.0 / (hidden as f32).sqrt();

    for causal in [false, true] {
        let mut manager = block_sparse_manager(causal, hidden);
        let q = Tensor::from_vec(queries.clone(), &[1, seq_len, hidden])
            .expect("tensor must build in test");
        let k = Tensor::from_vec(keys.clone(), &[1, seq_len, hidden])
            .expect("tensor must build in test");
        let v = Tensor::from_vec(values.clone(), &[1, seq_len, hidden])
            .expect("tensor must build in test");

        let expected = naive_attention(&queries, &keys, &values, seq_len, hidden, scale, causal);

        // Every tile size must give the same answer as the dense reference.
        for block_size in [1usize, 3, 8, 32] {
            let actual = manager
                .compute_block_sparse_attention(&q, &k, &v, block_size)
                .expect("block-sparse attention must succeed in test")
                .to_vec_f32()
                .expect("tensor read must succeed in test");

            assert_eq!(actual.len(), expected.len());
            assert!(
                actual.iter().any(|value| value.abs() > 1e-6),
                "output must not be all zeros (causal={causal}, block={block_size})"
            );
            for (index, (got, want)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    (got - want).abs() < 1e-4,
                    "causal={causal} block={block_size} index={index}: {got} != {want}"
                );
            }
        }
    }
}

#[test]
fn block_sparse_attention_output_depends_on_values() {
    let seq_len = 4usize;
    let hidden = 2usize;
    let mut manager = block_sparse_manager(false, hidden);

    let queries: Vec<f32> = (0..seq_len * hidden).map(|i| i as f32 * 0.1).collect();
    let keys: Vec<f32> = (0..seq_len * hidden).map(|i| (i as f32 * 0.2).cos()).collect();
    let q = Tensor::from_vec(queries, &[1, seq_len, hidden]).expect("tensor builds in test");
    let k = Tensor::from_vec(keys, &[1, seq_len, hidden]).expect("tensor builds in test");

    let first = manager
        .compute_block_sparse_attention(
            &q,
            &k,
            &Tensor::from_vec(vec![1.0f32; seq_len * hidden], &[1, seq_len, hidden])
                .expect("tensor builds in test"),
            2,
        )
        .expect("attention must succeed in test")
        .to_vec_f32()
        .expect("tensor read must succeed in test");

    let second = manager
        .compute_block_sparse_attention(
            &q,
            &k,
            &Tensor::from_vec(
                (0..seq_len * hidden).map(|i| i as f32).collect::<Vec<f32>>(),
                &[1, seq_len, hidden],
            )
            .expect("tensor builds in test"),
            2,
        )
        .expect("attention must succeed in test")
        .to_vec_f32()
        .expect("tensor read must succeed in test");

    assert_ne!(first, second, "the output must depend on the value tensor");
    // A constant value tensor is reproduced exactly by any convex
    // combination of its rows.
    for value in &first {
        assert!((value - 1.0).abs() < 1e-5, "expected 1.0, got {value}");
    }
}

#[test]
fn block_sparse_attention_rejects_mismatched_shapes() {
    let mut manager = block_sparse_manager(false, 2);
    let q = Tensor::from_vec(vec![0.0f32; 8], &[1, 4, 2]).expect("tensor builds in test");
    let k = Tensor::from_vec(vec![0.0f32; 4], &[1, 2, 2]).expect("tensor builds in test");
    assert!(manager.compute_block_sparse_attention(&q, &k, &q, 2).is_err());
}

// ── Ring KV rotation ─────────────────────────────────────────────────
//
// The previous implementation ignored every device's stored K/V and pushed
// a freshly synthesised `sin`/`cos` buffer around the ring. These tests
// pin the rotation to the real data: what arrives at device `i + 1` must be
// exactly what device `i` holds.

fn ring_manager(num_devices: usize, chunk: usize, head_dim: usize) -> RingAttentionManager {
    let config = RingAttentionConfig {
        num_devices,
        chunk_size: chunk,
        head_dim,
        compression_enabled: false,
        bidirectional: false,
        ..Default::default()
    };
    RingAttentionManager::new(config, chunk * num_devices).expect("ring manager must build in test")
}

/// Device-specific K/V that no synthetic generator would reproduce.
fn device_kv(rank: usize, len: usize) -> (Vec<f32>, Vec<f32>) {
    let keys: Vec<f32> = (0..len).map(|i| 100.0 * rank as f32 + i as f32).collect();
    let values: Vec<f32> = (0..len).map(|i| -(100.0 * rank as f32 + i as f32)).collect();
    (keys, values)
}

#[test]
fn rotation_delivers_each_device_s_own_kv_to_its_successor() {
    let devices = 4;
    let len = 6;
    let mut manager = ring_manager(devices, 3, 2);
    manager.communication_pattern = RingCommunicationPattern::Unidirectional;

    for rank in 0..devices {
        let (keys, values) = device_kv(rank, len);
        manager.set_local_kv(rank, keys, values).expect("kv must load in test");
    }

    manager.rotate_kv_pairs().expect("rotation must succeed in test");

    for source in 0..devices {
        let destination = (source + 1) % devices;
        let received = &manager.devices[destination].received_kv;
        assert_eq!(
            received.len(),
            1,
            "device {destination} must get exactly one hop"
        );

        let pair = &received[0];
        assert_eq!(pair.source_rank, source);

        let (expected_keys, expected_values) = device_kv(source, len);
        assert_eq!(
            pair.keys, expected_keys,
            "device {destination} must receive device {source}'s real keys"
        );
        assert_eq!(pair.values, expected_values);
        assert_eq!(
            pair.position_range, manager.devices[source].sequence_chunk,
            "the chunk range must travel with the data"
        );
    }
}

#[test]
fn rotation_without_loaded_kv_errors_instead_of_synthesising() {
    let mut manager = ring_manager(2, 3, 2);
    manager.communication_pattern = RingCommunicationPattern::Unidirectional;

    let error = manager.rotate_kv_pairs().expect_err("rotating empty devices must fail in test");
    assert!(error.to_string().contains("set_local_kv"), "{error}");
}

#[test]
fn bidirectional_rotation_delivers_both_neighbours_real_kv() {
    let devices = 3;
    let len = 4;
    let mut manager = ring_manager(devices, 2, 2);
    manager.communication_pattern = RingCommunicationPattern::Bidirectional;

    for rank in 0..devices {
        let (keys, values) = device_kv(rank, len);
        manager.set_local_kv(rank, keys, values).expect("kv must load in test");
    }

    manager.rotate_kv_pairs().expect("rotation must succeed in test");

    for destination in 0..devices {
        let mut sources: Vec<usize> = manager.devices[destination]
            .received_kv
            .iter()
            .map(|pair| pair.source_rank)
            .collect();
        sources.sort_unstable();

        let forward = (destination + devices - 1) % devices;
        let backward = (destination + 1) % devices;
        let mut expected = vec![forward, backward];
        expected.sort_unstable();
        assert_eq!(sources, expected, "device {destination} neighbours");

        for pair in &manager.devices[destination].received_kv {
            let (expected_keys, _) = device_kv(pair.source_rank, len);
            assert_eq!(pair.keys, expected_keys);
        }
    }
}

#[test]
fn rotation_accounts_for_the_bytes_it_actually_moved() {
    let devices = 2;
    let len = 8;
    let mut manager = ring_manager(devices, 4, 2);
    manager.communication_pattern = RingCommunicationPattern::Unidirectional;

    for rank in 0..devices {
        let (keys, values) = device_kv(rank, len);
        manager.set_local_kv(rank, keys, values).expect("kv must load in test");
    }
    manager.rotate_kv_pairs().expect("rotation must succeed in test");

    let expected_bytes = (2 * len * std::mem::size_of::<f32>()) as u64;
    for rank in 0..devices {
        assert_eq!(
            manager.devices[rank].attention_stats.communication_volume, expected_bytes,
            "device {rank} must report the bytes it really sent"
        );
    }
}

#[test]
fn kv_compression_shrinks_the_payload_and_stays_close_to_the_original() {
    let devices = 2;
    let len = 64;

    let mut manager = ring_manager(devices, 8, 8);
    manager.communication_pattern = RingCommunicationPattern::Unidirectional;
    manager.config.compression_enabled = true;
    manager.config.compression_ratio = 0.25; // 8-bit payload

    for rank in 0..devices {
        let (keys, values) = device_kv(rank, len);
        manager.set_local_kv(rank, keys, values).expect("kv must load in test");
    }
    manager.rotate_kv_pairs().expect("rotation must succeed in test");

    let uncompressed = RingAttentionManager::uncompressed_kv_bytes(len) as u64;
    for rank in 0..devices {
        let reported = manager.devices[rank].attention_stats.communication_volume;
        assert!(
            reported < uncompressed,
            "compression must report fewer bytes than {uncompressed}, got {reported}"
        );
    }

    // The reconstruction must still track the source data: the previous
    // implementation replaced the tail with a geometric decay of the last
    // kept sample, which drifts arbitrarily far from the original.
    let received = &manager.devices[1].received_kv[0];
    let (expected_keys, _) = device_kv(0, len);
    let span = expected_keys.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        - expected_keys.iter().cloned().fold(f32::INFINITY, f32::min);
    let tolerance = span / 255.0; // one 8-bit step
    for (got, want) in received.keys.iter().zip(&expected_keys) {
        assert!(
            (got - want).abs() <= tolerance + 1e-4,
            "quantized {got} is more than one step from {want}"
        );
    }
}

#[test]
fn kv_compression_survives_out_of_range_ratios() {
    // `compression_ratio` values that used to panic: > 1.0 produced
    // `step_by(0)`, and a very small ratio indexed an empty vector.
    for ratio in [0.0f32, 0.001, 1.0, 4.0, f32::NAN] {
        let mut manager = ring_manager(2, 4, 4);
        manager.communication_pattern = RingCommunicationPattern::Unidirectional;
        manager.config.compression_enabled = true;
        manager.config.compression_ratio = ratio;

        for rank in 0..2 {
            let (keys, values) = device_kv(rank, 16);
            manager.set_local_kv(rank, keys, values).expect("kv must load in test");
        }
        manager
            .rotate_kv_pairs()
            .unwrap_or_else(|error| panic!("ratio {ratio} must not fail: {error}"));

        assert_eq!(manager.devices[1].received_kv[0].keys.len(), 16);
    }
}

#[test]
fn quantization_bits_map_the_ratio_and_stay_in_range() {
    assert_eq!(RingAttentionManager::quantization_bits(0.25), Some(8));
    assert_eq!(RingAttentionManager::quantization_bits(0.5), Some(16));
    assert_eq!(RingAttentionManager::quantization_bits(0.0), Some(2));
    // A ratio of 1.0 or more asks for no reduction at all.
    assert_eq!(RingAttentionManager::quantization_bits(1.0), None);
    assert_eq!(RingAttentionManager::quantization_bits(10.0), None);
    assert_eq!(RingAttentionManager::quantization_bits(f32::NAN), None);
}

#[test]
fn a_ratio_of_one_transmits_the_data_verbatim_and_claims_no_saving() {
    let devices = 2;
    let len = 32;

    for ratio in [1.0f32, 4.0, f32::NAN] {
        let mut manager = ring_manager(devices, 8, 4);
        manager.communication_pattern = RingCommunicationPattern::Unidirectional;
        manager.config.compression_enabled = true;
        manager.config.compression_ratio = ratio;

        for rank in 0..devices {
            let (keys, values) = device_kv(rank, len);
            manager.set_local_kv(rank, keys, values).expect("kv must load in test");
        }
        manager.rotate_kv_pairs().expect("rotation must succeed in test");

        let uncompressed = RingAttentionManager::uncompressed_kv_bytes(len) as u64;
        for rank in 0..devices {
            assert_eq!(
                manager.devices[rank].attention_stats.communication_volume, uncompressed,
                "ratio {ratio} must not claim a saving it did not make"
            );
        }

        let (expected_keys, expected_values) = device_kv(0, len);
        let received = &manager.devices[1].received_kv[0];
        assert_eq!(
            received.keys, expected_keys,
            "ratio {ratio} must be lossless"
        );
        assert_eq!(
            received.values, expected_values,
            "ratio {ratio} must be lossless"
        );
    }
}

#[test]
fn quantization_is_exact_for_a_constant_buffer() {
    let mut buffer = vec![0.375f32; 8];
    RingAttentionManager::quantize_in_place(&mut buffer, 4);
    assert_eq!(buffer, vec![0.375f32; 8]);
}
