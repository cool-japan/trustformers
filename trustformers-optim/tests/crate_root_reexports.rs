//! Smoke test locking in the crate-root re-exports for `fsdp`, `optimizer_surgery`,
//! and `per_layer_quant`.
//!
//! These three modules were already `pub mod`-declared but their types were only
//! reachable via `trustformers_optim::fsdp::FsdpConfig` (etc.), not directly at
//! `trustformers_optim::FsdpConfig`. This test imports every one of the 21 publicly
//! re-exported types by name straight from the crate root: if any of them is ever
//! dropped from `lib.rs`'s `pub use` blocks, this file simply stops compiling.

use std::collections::HashMap;
// All 21 types below are re-exported straight from the crate root — 7 from `fsdp`,
// 6 from `optimizer_surgery`, and 8 from `per_layer_quant` (alphabetized by rustfmt,
// so the three groups are interleaved here rather than listed module-by-module).
use trustformers_optim::{
    BitWidth, BitWidthStrategy, FsdpConfig, FsdpError, FsdpMemoryAnalyzer, FsdpState, FsdpUnit,
    LayerBitWidthAssignment, LayerSensitivity, MigrationReport, OptimizerKind, OptimizerSurgeon,
    ParamStateSnapshot, PerLayerQuantSelector, QuantSelectionError, QuantizationPolicy,
    QuantizationSummary, ShardingStrategy, SurgeryConfig, SurgeryError, WrappingPolicy,
};

#[test]
fn fsdp_types_reachable_at_crate_root() {
    let config = FsdpConfig {
        world_size: 4,
        local_rank: 1,
        wrapping_policy: WrappingPolicy::TransformerLayerWrap { min_params: 1_000 },
        ..Default::default()
    };
    let mut state = FsdpState::new(config.clone());

    let mut values: HashMap<String, Vec<f64>> = HashMap::new();
    values.insert("layer.weight".to_string(), vec![1.0_f64; 64]);
    let unit_id = state
        .wrap_unit(vec!["layer.weight".to_string()], values)
        .expect("wrap_unit should succeed for a fresh, non-empty unit");

    assert_eq!(state.unit_count(), 1);
    assert_eq!(state.total_params(), 64);

    let unit = FsdpUnit::new(
        unit_id,
        vec!["layer.weight".to_string()],
        64,
        config.world_size,
    );
    assert_eq!(unit.local_params(), unit.shard_size);

    let peak_memory = FsdpMemoryAnalyzer::peak_memory(&config, 64);
    assert!(peak_memory > 0, "peak memory estimate should be positive");

    // ShardingStrategy is independent data used alongside FsdpConfig in real FSDP
    // setups — confirm it round-trips through equality on its own.
    assert_eq!(ShardingStrategy::FullShard, ShardingStrategy::FullShard);
    assert_ne!(ShardingStrategy::FullShard, ShardingStrategy::NoShard);

    // FsdpError is reachable and produced by a real failure path: re-registering
    // an already-wrapped parameter name.
    let dup_values: HashMap<String, Vec<f64>> =
        HashMap::from([("layer.weight".to_string(), vec![2.0_f64; 64])]);
    let err = state
        .wrap_unit(vec!["layer.weight".to_string()], dup_values)
        .expect_err("re-registering the same param name must fail");
    assert!(matches!(err, FsdpError::AlreadyRegistered(_)));
}

#[test]
fn optimizer_surgery_types_reachable_at_crate_root() {
    let config = SurgeryConfig {
        from: OptimizerKind::Adam,
        to: OptimizerKind::SGD,
        transfer_momentum: true,
        reset_step: false,
        momentum_scale: 1.0,
    };
    let surgeon = OptimizerSurgeon::new(config);

    let mut snapshot = ParamStateSnapshot::new("layer.weight");
    snapshot.first_moment = Some(vec![0.1, 0.2, 0.3]);
    let mut states = HashMap::new();
    states.insert(snapshot.param_name.clone(), snapshot);

    surgeon
        .validate(&states)
        .expect("Adam->SGD with a first_moment present must validate");

    let migrated = surgeon.migrate(&states).expect("migrate should succeed");
    let dst = migrated.get("layer.weight").expect("migrated state for layer.weight");
    assert!(
        dst.momentum.is_some(),
        "Adam's first_moment should become SGD's momentum"
    );

    let report: MigrationReport = surgeon.migration_report(&states, &migrated);
    assert_eq!(report.params_migrated, 1);

    // SurgeryError is reachable and directly constructible.
    let err = SurgeryError::EmptyStates;
    assert!(format!("{err}").contains("Empty"));
}

#[test]
fn per_layer_quant_types_reachable_at_crate_root() {
    let policy = QuantizationPolicy {
        strategy: BitWidthStrategy::Uniform(BitWidth::Int8),
        budget_bytes: None,
        min_bit_width: BitWidth::Int2,
        max_bit_width: BitWidth::Fp32,
    };
    let selector = PerLayerQuantSelector::new(policy);

    let layers = vec![LayerSensitivity {
        layer_name: "embed".to_string(),
        gradient_norm: 1.0,
        weight_variance: 1.0,
        activation_range: 1.0,
        output_sensitivity: 1.0,
        is_embedding: true,
        is_final_layer: false,
    }];
    let counts = vec![10_000_usize];

    let assignments: Vec<LayerBitWidthAssignment> = selector
        .assign_bit_widths(&layers, &counts)
        .expect("uniform Int8 assignment should succeed");
    assert_eq!(assignments[0].bit_width, BitWidth::Int8);

    let summary: QuantizationSummary = PerLayerQuantSelector::summary_report(&assignments);
    assert!(summary.total_params > 0);

    // QuantSelectionError is reachable and produced by a real failure path.
    let err = selector
        .assign_bit_widths(&[], &[])
        .expect_err("empty layer list must be rejected");
    assert!(matches!(err, QuantSelectionError::EmptyLayers));
}
