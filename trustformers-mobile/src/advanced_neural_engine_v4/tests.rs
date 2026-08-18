//! Tests for the Neural Engine v4 optimizer.
//!
//! Most of these are regression tests against the module's previous behaviour:
//! it did not compile at all, attention returned `Tensor::zeros(&[1, 1])`, and
//! every reported analytic was a hardcoded constant.

use super::*;

fn device() -> NeuralEngineDeviceInfo {
    NeuralEngineDeviceInfo {
        device_name: "iPhone 15 Pro".to_string(),
        chip_name: "A17 Pro".to_string(),
        neural_engine_version: "v4".to_string(),
        memory_gb: 8,
        gpu_cores: 6,
        cpu_cores: 6,
    }
}

fn engine() -> AdvancedNeuralEngineV4 {
    AdvancedNeuralEngineV4::new(NeuralEngineV4Config::default(), device()).expect("engine")
}

/// Reference implementation of single-head scaled dot-product attention,
/// written as directly as possible from the definition so that it is an
/// independent check on the optimized path.
fn reference_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    mask: Option<&[f32]>,
) -> Vec<f32> {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_len * head_dim];
    for i in 0..seq_len {
        let mut scores = vec![0.0f32; seq_len];
        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[j] = dot * scale;
            if let Some(m) = mask {
                scores[j] += m[i * seq_len + j];
            }
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for score in &mut scores {
            *score = if score.is_finite() { (*score - max).exp() } else { 0.0 };
            sum += *score;
        }
        for d in 0..head_dim {
            let mut acc = 0.0f32;
            for j in 0..seq_len {
                acc += scores[j] / sum * v[j * head_dim + d];
            }
            output[i * head_dim + d] = acc;
        }
    }
    output
}

/// Regression: `execute_optimized_attention` was
/// `let _ = (query, key, value, attention_mask); Ok(Tensor::zeros(&[1, 1])?)`.
#[test]
fn attention_returns_the_right_shape_and_is_not_zeros() {
    let engine = engine();
    let seq_len = 4;
    let hidden = 8;
    let values: Vec<f32> = (0..seq_len * hidden).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let q = Tensor::from_vec(values.clone(), &[seq_len, hidden]).expect("q");
    let k = Tensor::from_vec(values.clone(), &[seq_len, hidden]).expect("k");
    let v = Tensor::from_vec(values, &[seq_len, hidden]).expect("v");

    let output = engine.execute_optimized_attention(&q, &k, &v, None, 2).expect("attention");

    // The old code returned a 1x1 tensor.
    assert_eq!(output.shape(), vec![seq_len, hidden]);
    let data = output.to_vec_f32().expect("vec");
    assert_eq!(data.len(), seq_len * hidden);
    assert!(
        data.iter().any(|x| *x != 0.0),
        "attention output is all zeros"
    );
    assert!(
        data.iter().all(|x| x.is_finite()),
        "attention produced non-finite values"
    );
}

/// Single-head attention must match an independent reference implementation.
#[test]
fn attention_matches_the_reference_implementation() {
    let engine = engine();
    let seq_len = 5;
    let head_dim = 4;
    let q: Vec<f32> = (0..seq_len * head_dim).map(|i| ((i * 7) % 13) as f32 * 0.1).collect();
    let k: Vec<f32> = (0..seq_len * head_dim).map(|i| ((i * 5) % 11) as f32 * 0.1).collect();
    let v: Vec<f32> = (0..seq_len * head_dim).map(|i| ((i * 3) % 7) as f32 * 0.1).collect();

    let output = engine
        .execute_optimized_attention(
            &Tensor::from_vec(q.clone(), &[seq_len, head_dim]).expect("q"),
            &Tensor::from_vec(k.clone(), &[seq_len, head_dim]).expect("k"),
            &Tensor::from_vec(v.clone(), &[seq_len, head_dim]).expect("v"),
            None,
            1,
        )
        .expect("attention")
        .to_vec_f32()
        .expect("vec");

    let expected = reference_attention(&q, &k, &v, seq_len, head_dim, None);
    for (i, (actual, want)) in output.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - want).abs() < 1e-5,
            "element {i}: {actual} != {want}"
        );
    }
}

/// Each head must attend independently: computing two heads at once must give
/// the same answer as computing each separately.
#[test]
fn multi_head_attention_matches_per_head_computation() {
    let engine = engine();
    let seq_len = 4;
    let head_dim = 3;
    let num_heads = 2;
    let hidden = head_dim * num_heads;

    let make = |offset: usize| -> Vec<f32> {
        (0..seq_len * hidden)
            .map(|i| (((i + offset) * 11) % 17) as f32 * 0.05)
            .collect()
    };
    let (q, k, v) = (make(0), make(3), make(7));

    let combined = engine
        .execute_optimized_attention(
            &Tensor::from_vec(q.clone(), &[seq_len, hidden]).expect("q"),
            &Tensor::from_vec(k.clone(), &[seq_len, hidden]).expect("k"),
            &Tensor::from_vec(v.clone(), &[seq_len, hidden]).expect("v"),
            None,
            num_heads,
        )
        .expect("attention")
        .to_vec_f32()
        .expect("vec");

    for head in 0..num_heads {
        // Slice this head's columns out of the interleaved layout.
        let slice = |source: &[f32]| -> Vec<f32> {
            let mut out = Vec::with_capacity(seq_len * head_dim);
            for row in 0..seq_len {
                for d in 0..head_dim {
                    out.push(source[row * hidden + head * head_dim + d]);
                }
            }
            out
        };
        let expected =
            reference_attention(&slice(&q), &slice(&k), &slice(&v), seq_len, head_dim, None);
        for row in 0..seq_len {
            for d in 0..head_dim {
                let actual = combined[row * hidden + head * head_dim + d];
                let want = expected[row * head_dim + d];
                assert!(
                    (actual - want).abs() < 1e-5,
                    "head {head} row {row} dim {d}: {actual} != {want}"
                );
            }
        }
    }
}

/// Attention weights must sum to one: with an all-equal value matrix, every
/// output row equals that value.
#[test]
fn attention_weights_form_a_probability_distribution() {
    let engine = engine();
    let seq_len = 6;
    let hidden = 4;
    let q: Vec<f32> = (0..seq_len * hidden).map(|i| (i % 5) as f32).collect();
    let k: Vec<f32> = (0..seq_len * hidden).map(|i| (i % 3) as f32).collect();
    // Every value row is the constant 2.5, so any convex combination is 2.5.
    let v = vec![2.5f32; seq_len * hidden];

    let output = engine
        .execute_optimized_attention(
            &Tensor::from_vec(q, &[seq_len, hidden]).expect("q"),
            &Tensor::from_vec(k, &[seq_len, hidden]).expect("k"),
            &Tensor::from_vec(v, &[seq_len, hidden]).expect("v"),
            None,
            1,
        )
        .expect("attention")
        .to_vec_f32()
        .expect("vec");

    for (i, value) in output.iter().enumerate() {
        assert!(
            (value - 2.5).abs() < 1e-5,
            "element {i}: {value} should be 2.5 (weights must sum to 1)"
        );
    }
}

/// A causal mask must prevent any position from attending to the future: the
/// first output row must depend only on the first value row.
#[test]
fn causal_mask_is_respected() {
    let engine = engine();
    let seq_len = 4;
    let hidden = 2;
    let q: Vec<f32> = (0..seq_len * hidden).map(|i| i as f32 * 0.3).collect();
    let k = q.clone();
    // Distinct value rows so the effect of masking is visible.
    let mut v = vec![0.0f32; seq_len * hidden];
    for row in 0..seq_len {
        for d in 0..hidden {
            v[row * hidden + d] = (row as f32 + 1.0) * 10.0;
        }
    }

    let mut mask = vec![0.0f32; seq_len * seq_len];
    for row in 0..seq_len {
        for col in (row + 1)..seq_len {
            mask[row * seq_len + col] = f32::NEG_INFINITY;
        }
    }

    let output = engine
        .execute_optimized_attention(
            &Tensor::from_vec(q, &[seq_len, hidden]).expect("q"),
            &Tensor::from_vec(k, &[seq_len, hidden]).expect("k"),
            &Tensor::from_vec(v, &[seq_len, hidden]).expect("v"),
            Some(&Tensor::from_vec(mask, &[seq_len, seq_len]).expect("mask")),
            1,
        )
        .expect("attention")
        .to_vec_f32()
        .expect("vec");

    // Row 0 can only see value row 0, which is 10.0 everywhere.
    for d in 0..hidden {
        assert!(
            (output[d] - 10.0).abs() < 1e-4,
            "row 0 dim {d}: {} should be exactly the first value row (10.0)",
            output[d]
        );
    }
    // Later rows see more of the (larger) values, so they must exceed row 0.
    assert!(
        output[(seq_len - 1) * hidden] > output[0],
        "the last row should attend to larger values than the first"
    );
}

#[test]
fn attention_rejects_malformed_input() {
    let engine = engine();
    let good = Tensor::from_vec(vec![1.0; 8], &[4, 2]).expect("t");

    // Wrong rank.
    let three_d = Tensor::from_vec(vec![1.0; 8], &[2, 2, 2]).expect("t");
    assert!(engine
        .execute_optimized_attention(&three_d, &three_d, &three_d, None, 1)
        .is_err());

    // Mismatched shapes.
    let other = Tensor::from_vec(vec![1.0; 6], &[3, 2]).expect("t");
    assert!(engine.execute_optimized_attention(&good, &other, &good, None, 1).is_err());

    // num_heads does not divide the hidden size.
    assert!(engine.execute_optimized_attention(&good, &good, &good, None, 3).is_err());

    // Zero heads.
    assert!(engine.execute_optimized_attention(&good, &good, &good, None, 0).is_err());

    // Wrong mask shape.
    let bad_mask = Tensor::from_vec(vec![0.0; 4], &[2, 2]).expect("mask");
    assert!(engine
        .execute_optimized_attention(&good, &good, &good, Some(&bad_mask), 1)
        .is_err());
}

/// Regression: `get_compilation_statistics` returned
/// `total_compilations: 100, successful_compilations: 98, cache_hit_rate: 0.85`
/// unconditionally. With nothing compiled it must say so.
#[test]
fn compilation_statistics_are_measured_or_unavailable() {
    let engine = engine();
    match engine.compilation_statistics() {
        Availability::NotAvailable { reason } => {
            assert!(reason.contains("no graph compilations"), "{reason}");
        },
        Availability::Available(stats) => {
            panic!("must not report statistics before anything compiled: {stats:?}")
        },
    }

    engine.record_compilation(true, Duration::from_millis(100), false);
    engine.record_compilation(true, Duration::from_millis(200), true);
    engine.record_compilation(false, Duration::from_millis(300), false);

    let stats = engine
        .compilation_statistics()
        .measured()
        .cloned()
        .expect("statistics after recording");
    assert_eq!(stats.total_compilations, 3);
    assert_eq!(stats.successful_compilations, 2);
    assert_eq!(stats.cache_hits, 1);
    assert_eq!(stats.average_compilation_time, Duration::from_millis(200));
    assert!((stats.cache_hit_rate - 1.0 / 3.0).abs() < 1e-12);

    // The old hardcoded values must not appear.
    assert_ne!(stats.total_compilations, 100);
    assert!((stats.cache_hit_rate - 0.85).abs() > 1e-6);
}

/// Regression: execution analytics were invented. They must reflect real runs.
#[test]
fn execution_statistics_are_measured_or_unavailable() {
    let engine = engine();
    match engine.execution_statistics() {
        Availability::NotAvailable { reason } => {
            assert!(reason.contains("no operations"), "{reason}");
        },
        Availability::Available(stats) => {
            panic!("must not report statistics before anything ran: {stats:?}")
        },
    }

    let t = Tensor::from_vec(vec![0.5; 16], &[4, 4]).expect("t");
    engine.execute_optimized_attention(&t, &t, &t, None, 2).expect("run 1");
    engine.execute_optimized_attention(&t, &t, &t, None, 2).expect("run 2");

    let stats = engine
        .execution_statistics()
        .measured()
        .cloned()
        .expect("statistics after execution");
    assert_eq!(stats.execution_count, 2);
    assert_eq!(stats.total_elements_produced, 32);
    // These are real wall-clock measurements, so they must be non-zero and
    // ordered, but their magnitudes are not asserted (that would be a claim
    // about the machine, not the code).
    assert!(stats.average_latency > Duration::ZERO);
    assert!(stats.fastest <= stats.average_latency);
    assert!(stats.slowest >= stats.average_latency);
    assert!(stats.elements_per_second > 0.0);
    assert!(stats.backend == "cpu" || stats.backend == "metal");
}

/// Every recorded execution must carry a genuine measurement.
#[test]
fn execution_records_capture_real_measurements() {
    let engine = engine();
    assert!(engine.execution_records().is_empty());

    let t = Tensor::from_vec(vec![1.0; 8], &[2, 4]).expect("t");
    engine.execute_optimized_attention(&t, &t, &t, None, 1).expect("run");

    let records = engine.execution_records();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].operation, "attention");
    assert_eq!(records[0].element_count, 8);
    assert_eq!(records[0].thermal_state, NeuralEngineThermalState::Nominal);
    assert!(records[0].elapsed > Duration::ZERO);
}

/// Regression: `get_memory_statistics` returned fixed byte counts. It must now
/// report the real process footprint, or say it cannot.
#[test]
fn memory_statistics_are_real() {
    let engine = engine();
    match engine.memory_statistics() {
        Availability::Available(stats) => {
            // A running test process always uses some memory.
            assert!(
                stats.resident_bytes > 0,
                "resident set size should be positive"
            );
            // The old code always reported exactly 128 MiB.
            assert_ne!(stats.resident_bytes, 128 * 1024 * 1024);
        },
        Availability::NotAvailable { reason } => {
            // Acceptable on a platform that does not report it, as long as it
            // says so rather than inventing a number.
            assert!(reason.contains("did not report"), "{reason}");
        },
    }
}

/// Thermal state is caller-reported and defaults to Nominal; it is never
/// invented.
#[test]
fn thermal_state_is_caller_reported() {
    let engine = engine();
    assert_eq!(engine.thermal_state(), NeuralEngineThermalState::Nominal);
    assert!((engine.performance_scale() - 1.0).abs() < 1e-6);

    engine.set_thermal_state(NeuralEngineThermalState::Serious);
    assert_eq!(engine.thermal_state(), NeuralEngineThermalState::Serious);
    assert!((engine.performance_scale() - 0.6).abs() < 1e-6);

    engine.set_thermal_state(NeuralEngineThermalState::Critical);
    assert!(engine.performance_scale() < 0.6);
}

#[test]
fn thermal_scales_are_monotone() {
    let states = [
        NeuralEngineThermalState::Nominal,
        NeuralEngineThermalState::Fair,
        NeuralEngineThermalState::Serious,
        NeuralEngineThermalState::Critical,
    ];
    for pair in states.windows(2) {
        assert!(
            pair[0].performance_scale() > pair[1].performance_scale(),
            "{:?} should allow more performance than {:?}",
            pair[0],
            pair[1]
        );
    }
}

/// Core detection must be honest about what it knows.
#[test]
fn core_detection_distinguishes_known_from_unknown_chips() {
    assert_eq!(
        AdvancedNeuralEngineV4::known_neural_engine_cores("A17 Pro"),
        Some(16)
    );
    assert_eq!(
        AdvancedNeuralEngineV4::known_neural_engine_cores("M3 Pro"),
        Some(16)
    );
    assert_eq!(
        AdvancedNeuralEngineV4::known_neural_engine_cores("A13 Bionic"),
        Some(8)
    );
    // An unknown chip yields None rather than a confident wrong answer.
    assert_eq!(
        AdvancedNeuralEngineV4::known_neural_engine_cores("A16"),
        None
    );
    assert_eq!(
        AdvancedNeuralEngineV4::known_neural_engine_cores("Snapdragon"),
        None
    );

    // The fallback is the documented conservative floor.
    let mut unknown = device();
    unknown.chip_name = "Totally New Chip".to_string();
    assert_eq!(
        AdvancedNeuralEngineV4::detect_neural_engine_cores(&unknown),
        8
    );
}

#[test]
fn explicit_core_count_is_honoured_and_zero_is_refused() {
    let mut config = NeuralEngineV4Config::default();
    config.num_cores = Some(4);
    let engine = AdvancedNeuralEngineV4::new(config, device()).expect("engine");
    assert_eq!(engine.num_cores(), 4);

    let mut bad = NeuralEngineV4Config::default();
    bad.num_cores = Some(0);
    assert!(AdvancedNeuralEngineV4::new(bad, device()).is_err());
}

/// The full analytics snapshot must contain no fabricated field.
#[test]
fn performance_analytics_contain_no_invented_numbers() {
    let engine = engine();
    let analytics = engine.performance_analytics();

    // Before anything runs, execution and compilation must be unavailable.
    assert!(!analytics.execution.is_available());
    assert!(!analytics.compilation.is_available());
    assert_eq!(analytics.num_cores, 16);
    assert_eq!(analytics.thermal_state, NeuralEngineThermalState::Nominal);
    assert!(analytics.backend == "cpu" || analytics.backend == "metal");

    // After a real run, execution becomes available.
    let t = Tensor::from_vec(vec![0.25; 4], &[2, 2]).expect("t");
    engine.execute_optimized_attention(&t, &t, &t, None, 1).expect("run");
    let analytics = engine.performance_analytics();
    assert!(analytics.execution.is_available());
    // ...but compilation stays unavailable, because none happened.
    assert!(!analytics.compilation.is_available());
}

#[test]
fn availability_accessors_behave() {
    let available: Availability<u32> = Availability::Available(7);
    assert!(available.is_available());
    assert_eq!(available.measured(), Some(&7));

    let missing: Availability<u32> = Availability::NotAvailable {
        reason: "nothing measured".to_string(),
    };
    assert!(!missing.is_available());
    assert_eq!(missing.measured(), None);
}

#[test]
fn config_defaults_are_sane() {
    let config = NeuralEngineV4Config::default();
    assert!(config.enable_multi_core);
    assert!(config.num_cores.is_none());
    assert!(config.dynamic_recompilation.enabled);
    assert!(config.memory_optimization.enable_prefetching);
    assert!(config.attention_config.enable_flash_attention);
    assert!(matches!(
        config.precision_config.default_precision,
        NeuralEnginePrecision::FP16
    ));
    assert_eq!(
        config.thermal_config.target_thermal_state,
        NeuralEngineThermalState::Fair
    );
}

/// Records accrue one per execution, up to the retention cap.
#[test]
fn execution_records_accrue_per_run() {
    let engine = engine();
    let t = Tensor::from_vec(vec![1.0; 2], &[1, 2]).expect("t");
    for _ in 0..50 {
        engine.execute_optimized_attention(&t, &t, &t, None, 1).expect("run");
    }
    assert_eq!(engine.execution_records().len(), 50);
}

/// The history buffer must actually evict once it reaches its cap, rather than
/// growing without bound. Exercised directly against the ring buffer, because
/// running MAX_EXECUTION_HISTORY real attentions would be needlessly slow.
#[test]
fn execution_history_evicts_at_the_cap() {
    let engine = engine();
    let record = |index: usize| ExecutionRecord {
        operation: format!("op-{index}"),
        elapsed: Duration::from_nanos(1),
        element_count: index,
        thermal_state: NeuralEngineThermalState::Nominal,
        backend: "cpu",
    };

    for index in 0..(MAX_EXECUTION_HISTORY + 25) {
        engine.record_execution_for_test(record(index));
    }

    let records = engine.execution_records();
    assert_eq!(
        records.len(),
        MAX_EXECUTION_HISTORY,
        "history must be capped"
    );
    // The oldest 25 were evicted, so the first retained record is index 25.
    assert_eq!(records[0].element_count, 25);
    assert_eq!(
        records[records.len() - 1].element_count,
        MAX_EXECUTION_HISTORY + 24
    );
}

/// The Metal kernel applies a causal mask unconditionally, so it may only be
/// used when the caller asked for exactly causal attention. This test pins the
/// predicate that decides that.
///
/// Regression: the GPU path was originally gated on `mask.is_none()`, so an
/// *unmasked* request silently received causal results under the `metal`
/// feature.
#[test]
fn causal_mask_detection_is_exact() {
    let seq_len = 4;
    let neg = f32::NEG_INFINITY;

    // The canonical causal mask.
    let mut causal = vec![0.0f32; seq_len * seq_len];
    for row in 0..seq_len {
        for col in (row + 1)..seq_len {
            causal[row * seq_len + col] = neg;
        }
    }
    assert!(is_causal_mask(&causal, seq_len));

    // An all-zero mask is *not* causal: it permits attending to the future.
    assert!(!is_causal_mask(&vec![0.0f32; seq_len * seq_len], seq_len));

    // A fully-masked matrix is not causal either.
    assert!(!is_causal_mask(&vec![neg; seq_len * seq_len], seq_len));

    // A causal mask with one extra masked position is not the causal mask.
    let mut extra = causal.clone();
    extra[seq_len] = neg; // (1, 0), below the diagonal
    assert!(!is_causal_mask(&extra, seq_len));

    // A causal mask with a finite penalty instead of -inf is not causal.
    let mut soft = causal.clone();
    soft[1] = -1e9;
    assert!(!is_causal_mask(&soft, seq_len));

    // A non-zero value on the diagonal is not causal (it is a bias).
    let mut biased = causal.clone();
    biased[0] = 0.5;
    assert!(!is_causal_mask(&biased, seq_len));

    // Wrong length is rejected rather than indexing out of bounds.
    assert!(!is_causal_mask(&causal, seq_len + 1));
    assert!(!is_causal_mask(&[], seq_len));
}

/// Unmasked attention must attend to the whole sequence on every backend.
///
/// With a strictly increasing value sequence, unmasked attention at position 0
/// must see later (larger) values, whereas causal attention at position 0 sees
/// only the first. This distinguishes the two functions behaviourally, so it
/// fails if a causal kernel is ever used to serve an unmasked request.
#[test]
fn unmasked_attention_attends_to_the_whole_sequence() {
    let engine = engine();
    let seq_len = 4;
    let hidden = 2;

    // Uniform q/k so every attention weight is equal: the output is then the
    // plain mean of the value rows over whatever positions are visible.
    let q = vec![0.0f32; seq_len * hidden];
    let k = vec![0.0f32; seq_len * hidden];
    let mut v = vec![0.0f32; seq_len * hidden];
    for row in 0..seq_len {
        for d in 0..hidden {
            v[row * hidden + d] = (row as f32 + 1.0) * 10.0;
        }
    }

    let output = engine
        .execute_optimized_attention(
            &Tensor::from_vec(q, &[seq_len, hidden]).expect("q"),
            &Tensor::from_vec(k, &[seq_len, hidden]).expect("k"),
            &Tensor::from_vec(v, &[seq_len, hidden]).expect("v"),
            None,
            1,
        )
        .expect("attention")
        .to_vec_f32()
        .expect("vec");

    // Row 0 sees all four value rows equally: mean(10, 20, 30, 40) = 25.
    // A causal kernel would return 10 here.
    assert!(
        (output[0] - 25.0).abs() < 1e-4,
        "unmasked row 0 should average the whole sequence (25.0), got {} — a value of 10.0 \
         means a causal kernel served an unmasked request",
        output[0]
    );

    // Every row sees the same thing, so every row is 25.
    for row in 0..seq_len {
        assert!(
            (output[row * hidden] - 25.0).abs() < 1e-4,
            "row {row}: {} should be 25.0",
            output[row * hidden]
        );
    }
}
