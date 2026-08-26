//! Tests for sequence parallelism.

use super::*;
use crate::distributed::SimulatedProcessGroup;
use std::sync::Arc;

#[test]
fn test_sequence_parallelism_config() {
    let config = SequenceParallelismConfig::default();
    assert_eq!(config.sequence_parallel_size, 1);
    assert_eq!(config.max_sequence_length_per_device, 2048);
    assert_eq!(config.overlap_size, 128);
}

#[test]
fn test_sequence_parallelism_creation() {
    let config = SequenceParallelismConfig {
        sequence_parallel_size: 4,
        max_sequence_length_per_device: 1024,
        overlap_size: 64,
        ..Default::default()
    };

    let process_group = Arc::new(SimulatedProcessGroup::new(0, 4));
    let sequence_parallelism = SequenceParallelism::new(config, 0, 4, process_group);

    assert!(sequence_parallelism.is_ok());
}

#[test]
#[ignore] // Memory-intensive test causes SIGKILL in constrained environments
fn test_equal_chunks_splitting() {
    let config = SequenceParallelismConfig {
        sequence_parallel_size: 2,
        max_sequence_length_per_device: 1000,
        overlap_size: 100,
        ..Default::default()
    };

    let process_group = Arc::new(SimulatedProcessGroup::new(0, 2));
    let mut sequence_parallelism =
        SequenceParallelism::new(config, 0, 2, process_group).expect("operation failed in test");

    let chunks = sequence_parallelism.split_sequence(1800).expect("operation failed in test");
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].start_position, 0);
    assert_eq!(chunks[0].end_position, 1000);
    assert_eq!(chunks[1].start_position, 900); // 1000 - 100 overlap
}

#[test]
#[ignore] // Memory-intensive test causes SIGKILL in constrained environments
fn test_chunk_processing() {
    let config = SequenceParallelismConfig {
        sequence_parallel_size: 2,
        max_sequence_length_per_device: 1000,
        overlap_size: 100,
        ..Default::default()
    };

    let process_group = Arc::new(SimulatedProcessGroup::new(0, 2));
    let mut sequence_parallelism =
        SequenceParallelism::new(config, 0, 2, process_group).expect("operation failed in test");

    let _chunks = sequence_parallelism.split_sequence(1800).expect("operation failed in test");

    let input = Tensor::zeros(&[1000, 768]).expect("tensor operation failed");
    let result = sequence_parallelism.forward_chunk(0, &input, None);
    assert!(result.is_ok());
}

#[test]
fn test_gradient_synchronization() {
    let config = SequenceParallelismConfig {
        sync_gradients: true,
        ..Default::default()
    };

    let process_group = Arc::new(SimulatedProcessGroup::new(0, 1));
    let sequence_parallelism =
        SequenceParallelism::new(config, 0, 1, process_group).expect("operation failed in test");

    let mut gradients = HashMap::new();
    gradients.insert(
        "test_param".to_string(),
        Tensor::ones(&[10, 10]).expect("tensor operation failed"),
    );

    let result = sequence_parallelism.synchronize_gradients(&mut gradients);
    assert!(result.is_ok());
}

#[test]
fn test_memory_usage_update() {
    let config = SequenceParallelismConfig::default();
    let process_group = Arc::new(SimulatedProcessGroup::new(0, 1));
    let sequence_parallelism =
        SequenceParallelism::new(config, 0, 1, process_group).expect("operation failed in test");

    let result = sequence_parallelism.update_memory_usage(0, 1024 * 1024 * 1024); // 1GB
    assert!(result.is_ok());

    let stats = sequence_parallelism.get_statistics();
    assert_eq!(stats.peak_memory_usage, 1024 * 1024 * 1024);
}

#[test]
fn test_optimal_sequence_config_calculation() {
    let config = utils::calculate_optimal_sequence_config(
        10000,                  // total sequence length
        8 * 1024 * 1024 * 1024, // 8GB memory per device
        1024,                   // 1KB per token
        4,                      // world size
    )
    .expect("operation failed in test");

    assert!(config.sequence_parallel_size <= 4);
    assert!(config.max_sequence_length_per_device > 0);
}

#[test]
fn test_communication_cost_estimation() {
    let config = SequenceParallelismConfig::default();
    let cost = utils::estimate_communication_cost(&config, 768, 12);
    assert!(cost > 0.0);
}

#[test]
fn test_memory_savings_calculation() {
    let savings = utils::calculate_memory_savings(10000, 4, 768);
    assert!(savings > 0.0 && savings < 1.0);
}

// ── Partitioning strategies ───────────────────────────────────────────────
//
// Every test below would pass trivially against the previous implementation,
// where `SemanticBoundaries`, `Dynamic` and `ComplexityBased` all silently
// delegated to `split_equal_chunks` (and the attention "analysis" drew its
// scores from `fastrand`). They assert that the chosen strategy actually
// changes the partition, and that it does so deterministically.

fn coordinator(
    strategy: SequenceSplittingStrategy,
    parallel_size: usize,
    chunk_size: usize,
) -> SequenceParallelism {
    let config = SequenceParallelismConfig {
        sequence_parallel_size: parallel_size,
        max_sequence_length_per_device: chunk_size,
        overlap_size: 0,
        splitting_strategy: strategy,
        attention_communication_opt: false,
        ..Default::default()
    };
    let group = Arc::new(SimulatedProcessGroup::new(0, parallel_size));
    SequenceParallelism::new(config, 0, parallel_size, group)
        .expect("coordinator must build in test")
}

#[test]
fn semantic_splitting_without_boundaries_is_an_error_not_a_silent_fallback() {
    let mut parallelism = coordinator(SequenceSplittingStrategy::SemanticBoundaries, 4, 64);
    let error = parallelism
        .split_sequence(200)
        .expect_err("semantic splitting must refuse to guess boundaries in test");
    assert!(
        error.to_string().contains("set_segment_boundaries"),
        "{error}"
    );
}

#[test]
fn semantic_splitting_cuts_only_at_registered_boundaries() {
    let mut parallelism = coordinator(SequenceSplittingStrategy::SemanticBoundaries, 4, 64);
    let boundaries = vec![13, 27, 44, 61, 79, 96];
    parallelism
        .set_segment_boundaries(boundaries.clone())
        .expect("boundaries must register in test");

    let chunks = parallelism.split_sequence(100).expect("split must succeed in test");
    assert_eq!(chunks.len(), 4, "one chunk per sequence-parallel rank");

    assert_eq!(chunks[0].start_position, 0);
    assert_eq!(chunks[chunks.len() - 1].end_position, 100);

    // Every interior cut is one of the registered offsets, and the chunks tile
    // the sequence without gaps.
    for window in chunks.windows(2) {
        assert_eq!(window[0].end_position, window[1].start_position);
        assert!(
            boundaries.contains(&window[0].end_position),
            "cut at {} is not a registered boundary",
            window[0].end_position
        );
    }

    // A run against uniform chunking would have cut at 25/50/75, none of which
    // is a registered boundary.
    let uniform = [25usize, 50, 75];
    for chunk in chunks.windows(2) {
        assert!(!uniform.contains(&chunk[0].end_position));
    }
}

#[test]
fn semantic_splitting_is_deterministic_across_runs() {
    let cuts: Vec<Vec<usize>> = (0..3)
        .map(|_| {
            let mut parallelism =
                coordinator(SequenceSplittingStrategy::SemanticBoundaries, 3, 128);
            parallelism
                .set_segment_boundaries(vec![9, 21, 40, 55, 73, 90])
                .expect("boundaries must register in test");
            parallelism
                .split_sequence(100)
                .expect("split must succeed in test")
                .iter()
                .map(|chunk| chunk.end_position)
                .collect()
        })
        .collect();

    assert_eq!(cuts[0], cuts[1]);
    assert_eq!(cuts[1], cuts[2]);
}

#[test]
fn complexity_splitting_without_costs_is_an_error() {
    let mut parallelism = coordinator(SequenceSplittingStrategy::ComplexityBased, 2, 128);
    let error = parallelism
        .split_sequence(100)
        .expect_err("complexity splitting must refuse to invent costs in test");
    assert!(error.to_string().contains("set_position_costs"), "{error}");
}

#[test]
fn complexity_splitting_balances_the_registered_cost() {
    // The first quarter of the sequence carries 9x the work of the rest, so a
    // cost-balanced partition must give it a much shorter chunk than uniform
    // chunking would.
    let mut costs = vec![1.0f32; 100];
    for cost in costs.iter_mut().take(25) {
        *cost = 9.0;
    }

    let mut parallelism = coordinator(SequenceSplittingStrategy::ComplexityBased, 2, 128);
    parallelism
        .set_position_costs(costs.clone())
        .expect("costs must register in test");

    let chunks = parallelism.split_sequence(100).expect("split must succeed in test");
    assert_eq!(chunks.len(), 2);

    let boundary = chunks[0].end_position;
    assert!(
        boundary < 50,
        "the expensive prefix must get a shorter chunk than uniform chunking's 50, got {boundary}"
    );

    // Both halves must carry comparable total cost.
    let first: f32 = costs[..boundary].iter().sum();
    let second: f32 = costs[boundary..].iter().sum();
    let imbalance = (first - second).abs() / (first + second);
    assert!(imbalance < 0.15, "cost imbalance {imbalance} is too large");
}

#[test]
fn complexity_splitting_rejects_a_short_cost_vector() {
    let mut parallelism = coordinator(SequenceSplittingStrategy::ComplexityBased, 2, 128);
    parallelism
        .set_position_costs(vec![1.0; 10])
        .expect("costs must register in test");
    let error = parallelism
        .split_sequence(100)
        .expect_err("a short cost vector must be rejected in test");
    assert!(error.to_string().contains("cover"), "{error}");
}

#[test]
fn position_costs_reject_invalid_values() {
    let parallelism = coordinator(SequenceSplittingStrategy::ComplexityBased, 2, 128);
    assert!(parallelism.set_position_costs(Vec::new()).is_err());
    assert!(parallelism.set_position_costs(vec![1.0, -1.0]).is_err());
    assert!(parallelism.set_position_costs(vec![f32::NAN]).is_err());
    assert!(parallelism.set_position_costs(vec![0.0, 0.0]).is_err());
}

#[test]
fn dynamic_splitting_shrinks_chunks_under_memory_pressure() {
    let relaxed = {
        let mut parallelism = coordinator(SequenceSplittingStrategy::Dynamic, 4, 64);
        parallelism.split_sequence(256).expect("split must succeed in test")
    };

    let pressured = {
        let mut parallelism = coordinator(SequenceSplittingStrategy::Dynamic, 4, 64);
        // 15 GiB against the 16 GiB budget => pressure > 0.8 => half chunks.
        parallelism
            .update_memory_usage(0, 15 * 1024 * 1024 * 1024)
            .expect("memory update must succeed in test");
        parallelism.split_sequence(256).expect("split must succeed in test")
    };

    assert_eq!(relaxed[0].end_position - relaxed[0].start_position, 64);
    assert_eq!(
        pressured[0].end_position - pressured[0].start_position,
        32,
        "high memory pressure must halve the chunk size, not be computed and discarded"
    );
    assert!(
        pressured.len() > relaxed.len(),
        "smaller chunks means more of them: {} vs {}",
        pressured.len(),
        relaxed.len()
    );
}

#[test]
fn attention_based_splitting_is_reproducible_for_the_same_input() {
    // The previous implementation drew its "content complexity" from
    // `fastrand`, so identical inputs produced different partitions.
    let cuts: Vec<Vec<usize>> = (0..3)
        .map(|_| {
            let mut parallelism = coordinator(SequenceSplittingStrategy::AttentionBased, 2, 64);
            parallelism
                .split_sequence(100)
                .expect("split must succeed in test")
                .iter()
                .map(|chunk| chunk.end_position)
                .collect()
        })
        .collect();

    assert_eq!(cuts[0], cuts[1]);
    assert_eq!(cuts[1], cuts[2]);
}
