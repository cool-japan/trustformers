//! Unit tests for the tensor-parallelism module.
//!
//! Kept in its own file so `tensor_parallelism.rs` stays well under the
//! 2000-line ceiling.

use super::*;
use crate::distributed::SimulatedProcessGroup;
use std::sync::Arc;

#[test]
fn test_tensor_parallelism_config() {
    let config = TensorParallelismConfig::default();
    assert_eq!(config.tensor_parallel_size, 1);
    assert!(config.column_parallel);
    assert!(config.row_parallel);
}

#[test]
fn test_tensor_parallelism_creation() {
    let config = TensorParallelismConfig {
        tensor_parallel_size: 4,
        ..Default::default()
    };

    let process_group = Arc::new(SimulatedProcessGroup::new(0, 4));
    let tensor_parallelism = TensorParallelism::new(config, 0, 4, process_group);

    assert!(tensor_parallelism.is_ok());
}

/// Regression: `TensorPartitioningStrategy::Custom` used to silently fall
/// back to column-wise partitioning, so a caller's layout was replaced with
/// a different one without any signal.
#[test]
fn custom_partitioning_requires_a_registered_partitioner() {
    let config = TensorParallelismConfig {
        tensor_parallel_size: 2,
        partitioning_strategy: TensorPartitioningStrategy::Custom,
        ..Default::default()
    };
    let process_group = Arc::new(SimulatedProcessGroup::new(0, 2));
    let mut tensor_parallelism =
        TensorParallelism::new(config, 0, 2, process_group).expect("construction in test");

    let error = tensor_parallelism
        .partition_tensor("weights", &[4, 8], None)
        .expect_err("custom strategy without a partitioner must fail");
    assert!(
        error.to_string().contains("set_custom_partitioner"),
        "error must name the fix, got: {error}"
    );
}

#[test]
fn custom_partitioner_is_used_and_validated() {
    let config = TensorParallelismConfig {
        tensor_parallel_size: 2,
        partitioning_strategy: TensorPartitioningStrategy::Custom,
        ..Default::default()
    };
    let process_group = Arc::new(SimulatedProcessGroup::new(0, 2));
    let mut tensor_parallelism =
        TensorParallelism::new(config, 0, 2, process_group).expect("construction in test");

    // A layout that is deliberately *not* the column-wise fallback: an
    // uneven 1/3 vs 2/3 split along the rows.
    tensor_parallelism.set_custom_partitioner(Arc::new(|name, shape, _size| {
        let rows = shape[0];
        let first = rows / 3;
        Ok(vec![
            TensorPartition {
                partition_id: 0,
                device_rank: 0,
                tensor_name: name.to_string(),
                shape: vec![first, shape[1]],
                offset: vec![0, 0],
                needs_communication: false,
                dependencies: Vec::new(),
            },
            TensorPartition {
                partition_id: 1,
                device_rank: 1,
                tensor_name: name.to_string(),
                shape: vec![rows - first, shape[1]],
                offset: vec![first, 0],
                needs_communication: false,
                dependencies: Vec::new(),
            },
        ])
    }));

    let partitions = tensor_parallelism
        .partition_tensor("weights", &[6, 4], None)
        .expect("custom partitioning must succeed in test");
    assert_eq!(partitions.len(), 2);
    assert_eq!(partitions[0].shape, vec![2, 4]);
    assert_eq!(partitions[1].shape, vec![4, 4]);
}

#[test]
fn custom_partitioner_must_tile_the_tensor() {
    let config = TensorParallelismConfig {
        tensor_parallel_size: 2,
        partitioning_strategy: TensorPartitioningStrategy::Custom,
        ..Default::default()
    };
    let process_group = Arc::new(SimulatedProcessGroup::new(0, 2));
    let mut tensor_parallelism =
        TensorParallelism::new(config, 0, 2, process_group).expect("construction in test");

    tensor_parallelism.set_custom_partitioner(Arc::new(|name, shape, _size| {
        Ok(vec![TensorPartition {
            partition_id: 0,
            device_rank: 0,
            tensor_name: name.to_string(),
            shape: vec![1, shape[1]],
            offset: vec![0, 0],
            needs_communication: false,
            dependencies: Vec::new(),
        }])
    }));

    assert!(tensor_parallelism.partition_tensor("weights", &[6, 4], None).is_err());
}

#[test]
fn test_column_wise_partitioning() {
    let config = TensorParallelismConfig {
        tensor_parallel_size: 2,
        ..Default::default()
    };

    let process_group = Arc::new(SimulatedProcessGroup::new(0, 2));
    let mut tensor_parallelism =
        TensorParallelism::new(config, 0, 2, process_group).expect("tensor operation failed");

    let partitions = tensor_parallelism
        .partition_tensor("test", &[100, 200], None)
        .expect("tensor operation failed");
    assert_eq!(partitions.len(), 2);
    assert_eq!(partitions[0].shape, vec![100, 100]);
    assert_eq!(partitions[1].shape, vec![100, 100]);
}

#[test]
fn test_row_wise_partitioning() {
    let config = TensorParallelismConfig {
        tensor_parallel_size: 2,
        partitioning_strategy: TensorPartitioningStrategy::RowWise,
        ..Default::default()
    };

    let process_group = Arc::new(SimulatedProcessGroup::new(0, 2));
    let mut tensor_parallelism =
        TensorParallelism::new(config, 0, 2, process_group).expect("tensor operation failed");

    let partitions = tensor_parallelism
        .partition_tensor("test", &[100, 200], None)
        .expect("tensor operation failed");
    assert_eq!(partitions.len(), 2);
    assert_eq!(partitions[0].shape, vec![50, 200]);
    assert_eq!(partitions[1].shape, vec![50, 200]);
}

#[test]
fn test_batch_wise_partitioning() {
    let config = TensorParallelismConfig {
        tensor_parallel_size: 2,
        partitioning_strategy: TensorPartitioningStrategy::BatchWise,
        ..Default::default()
    };

    let process_group = Arc::new(SimulatedProcessGroup::new(0, 2));
    let mut tensor_parallelism =
        TensorParallelism::new(config, 0, 2, process_group).expect("tensor operation failed");

    let partitions = tensor_parallelism
        .partition_tensor("test", &[64, 100, 200], None)
        .expect("tensor operation failed");
    assert_eq!(partitions.len(), 2);
    assert_eq!(partitions[0].shape, vec![32, 100, 200]);
    assert_eq!(partitions[1].shape, vec![32, 100, 200]);
}

#[test]
fn test_tensor_operation_execution() {
    let config = TensorParallelismConfig::default();
    let process_group = Arc::new(SimulatedProcessGroup::new(0, 1));
    let tensor_parallelism =
        TensorParallelism::new(config, 0, 1, process_group).expect("tensor operation failed");

    let operation = TensorOperation {
        operation_id: 0,
        operation_type: TensorOperationType::Add,
        input_partitions: vec![0, 1],
        output_partitions: vec![0],
        communication_requirements: vec![],
        memory_requirements: 1024,
    };

    let mut inputs = HashMap::new();
    inputs.insert(
        "A".to_string(),
        Tensor::ones(&[10, 10]).expect("tensor operation failed"),
    );
    inputs.insert(
        "B".to_string(),
        Tensor::ones(&[10, 10]).expect("tensor operation failed"),
    );

    let result = tensor_parallelism.execute_operation(&operation, &inputs);
    assert!(result.is_ok());
}

/// Naive single-head scaled dot-product attention reference.
fn reference_attention(q: &[f32], k: &[f32], v: &[f32], seq: usize, dim: usize) -> Vec<f32> {
    let scale = 1.0 / (dim as f32).sqrt();
    let mut out = vec![0.0f32; seq * dim];
    for row in 0..seq {
        let mut scores = vec![0.0f32; seq];
        for (col, score) in scores.iter_mut().enumerate() {
            let mut dot = 0.0f32;
            for f in 0..dim {
                dot += q[row * dim + f] * k[col * dim + f];
            }
            *score = dot * scale;
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
        let sum: f32 = exp.iter().sum();
        for (col, weight) in exp.iter().enumerate() {
            let weight = weight / sum;
            for f in 0..dim {
                out[row * dim + f] += weight * v[col * dim + f];
            }
        }
    }
    out
}

#[test]
fn attention_is_not_the_identity_and_matches_the_reference() {
    let config = TensorParallelismConfig::default();
    let process_group = Arc::new(SimulatedProcessGroup::new(0, 1));
    let tensor_parallelism =
        TensorParallelism::new(config, 0, 1, process_group).expect("must build in test");

    let seq = 3;
    let dim = 2;
    let q = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
    let k = vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
    let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let mut inputs = HashMap::new();
    inputs.insert(
        "query".to_string(),
        Tensor::from_slice(&q, &[seq, dim]).expect("tensor must build in test"),
    );
    inputs.insert(
        "key".to_string(),
        Tensor::from_slice(&k, &[seq, dim]).expect("tensor must build in test"),
    );
    inputs.insert(
        "value".to_string(),
        Tensor::from_slice(&v, &[seq, dim]).expect("tensor must build in test"),
    );

    let operation = TensorOperation {
        operation_id: 1,
        operation_type: TensorOperationType::Attention,
        input_partitions: vec![],
        output_partitions: vec![],
        communication_requirements: vec![],
        memory_requirements: 0,
    };

    let outputs = tensor_parallelism
        .execute_operation(&operation, &inputs)
        .expect("attention must execute in test");
    let produced = outputs
        .get("output")
        .expect("attention must produce an output")
        .to_vec_f32()
        .expect("tensor read must succeed in test");

    let expected = reference_attention(&q, &k, &v, seq, dim);
    for (got, want) in produced.iter().zip(&expected) {
        approx::assert_relative_eq!(got, want, epsilon = 1e-5);
    }

    // Regression: the previous implementation returned `inputs["input"]`
    // unchanged, so assert the output genuinely differs from the query.
    assert_ne!(produced, q, "attention must not be the identity");
}

#[test]
fn attention_errors_on_missing_inputs() {
    let config = TensorParallelismConfig::default();
    let process_group = Arc::new(SimulatedProcessGroup::new(0, 1));
    let tensor_parallelism =
        TensorParallelism::new(config, 0, 1, process_group).expect("must build in test");

    let operation = TensorOperation {
        operation_id: 1,
        operation_type: TensorOperationType::Attention,
        input_partitions: vec![],
        output_partitions: vec![],
        communication_requirements: vec![],
        memory_requirements: 0,
    };

    // Empty input map used to yield an empty output map instead of an error.
    assert!(tensor_parallelism.execute_operation(&operation, &HashMap::new()).is_err());
}

#[test]
fn matmul_and_linear_error_on_missing_inputs() {
    let config = TensorParallelismConfig::default();
    let process_group = Arc::new(SimulatedProcessGroup::new(0, 1));
    let tensor_parallelism =
        TensorParallelism::new(config, 0, 1, process_group).expect("must build in test");

    for operation_type in [
        TensorOperationType::MatMul,
        TensorOperationType::Linear,
        TensorOperationType::Add,
    ] {
        let operation = TensorOperation {
            operation_id: 0,
            operation_type,
            input_partitions: vec![],
            output_partitions: vec![],
            communication_requirements: vec![],
            memory_requirements: 0,
        };
        assert!(tensor_parallelism.execute_operation(&operation, &HashMap::new()).is_err());
    }
}

#[test]
fn layernorm_normalises_and_activation_is_real() {
    let config = TensorParallelismConfig::default();
    let process_group = Arc::new(SimulatedProcessGroup::new(0, 1));
    let tensor_parallelism =
        TensorParallelism::new(config, 0, 1, process_group).expect("must build in test");

    let mut inputs = HashMap::new();
    inputs.insert(
        "input".to_string(),
        Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("tensor must build in test"),
    );

    let layernorm = TensorOperation {
        operation_id: 0,
        operation_type: TensorOperationType::LayerNorm,
        input_partitions: vec![],
        output_partitions: vec![],
        communication_requirements: vec![],
        memory_requirements: 0,
    };
    let normalized = tensor_parallelism
        .execute_operation(&layernorm, &inputs)
        .expect("layernorm must execute in test")
        .get("output")
        .expect("layernorm must produce an output")
        .to_vec_f32()
        .expect("tensor read must succeed in test");

    // Each row has zero mean and unit variance (up to epsilon).
    for row in normalized.chunks(2) {
        approx::assert_relative_eq!(row[0] + row[1], 0.0f32, epsilon = 1e-4);
        approx::assert_relative_eq!(row[0], -1.0f32, epsilon = 1e-3);
    }

    inputs.insert(
        "input".to_string(),
        Tensor::from_slice(&[-2.0, -0.5, 0.5, 2.0], &[4]).expect("tensor must build in test"),
    );
    let relu = TensorOperation {
        operation_id: 1,
        operation_type: TensorOperationType::Activation,
        input_partitions: vec![],
        output_partitions: vec![],
        communication_requirements: vec![],
        memory_requirements: 0,
    };
    let activated = tensor_parallelism
        .execute_operation(&relu, &inputs)
        .expect("activation must execute in test")
        .get("output")
        .expect("activation must produce an output")
        .to_vec_f32()
        .expect("tensor read must succeed in test");
    assert_eq!(activated, vec![0.0, 0.0, 0.5, 2.0]);
}

#[test]
fn all_reduce_requirement_moves_real_partition_data() {
    use crate::distributed_collective::InProcessProcessGroup;

    let world_size = 2;
    let groups = InProcessProcessGroup::group(world_size).expect("groups must build in test");

    let handles: Vec<_> = groups
        .into_iter()
        .enumerate()
        .map(|(rank, group)| {
            std::thread::spawn(move || {
                let config = TensorParallelismConfig {
                    tensor_parallel_size: world_size,
                    column_parallel: false,
                    ..Default::default()
                };
                let group: Arc<dyn ProcessGroup> = Arc::new(group);
                let mut tp = TensorParallelism::new(config, rank, world_size, group)
                    .expect("must build in test");
                let partitions = tp
                    .partition_tensor("w", &[4, 4], None)
                    .expect("partitioning must succeed in test");
                let partition_id = partitions[0].partition_id;

                tp.store_partition_data(
                    partition_id,
                    Tensor::from_slice(&[rank as f32 + 1.0; 4], &[4])
                        .expect("tensor must build in test"),
                )
                .expect("store must succeed in test");

                let operation = TensorOperation {
                    operation_id: 0,
                    operation_type: TensorOperationType::Add,
                    input_partitions: vec![],
                    output_partitions: vec![],
                    communication_requirements: vec![CommunicationRequirement {
                        source_partition: partition_id,
                        target_partition: partition_id,
                        communication_type: TensorCommunicationPattern::AllReduce,
                        data_size: 16,
                    }],
                    memory_requirements: 0,
                };

                let mut inputs = HashMap::new();
                inputs.insert(
                    "A".to_string(),
                    Tensor::ones(&[2, 2]).expect("tensor must build in test"),
                );
                inputs.insert(
                    "B".to_string(),
                    Tensor::ones(&[2, 2]).expect("tensor must build in test"),
                );
                tp.execute_operation(&operation, &inputs)
                    .expect("operation must execute in test");

                tp.partition_data(partition_id)
                    .expect("partition data must survive")
                    .to_vec_f32()
                    .expect("tensor read must succeed in test")
            })
        })
        .collect();

    for handle in handles {
        let values = handle.join().expect("rank thread must not panic in test");
        // 1 + 2 summed across the two ranks.
        assert_eq!(values, vec![3.0f32; 4]);
    }
}

#[test]
fn all_reduce_requirement_errors_without_registered_data() {
    let config = TensorParallelismConfig {
        tensor_parallel_size: 2,
        ..Default::default()
    };
    let process_group = Arc::new(SimulatedProcessGroup::new(0, 2));
    let mut tp = TensorParallelism::new(config, 0, 2, process_group).expect("must build in test");
    let partitions = tp
        .partition_tensor("w", &[4, 4], None)
        .expect("partitioning must succeed in test");

    let requirement = CommunicationRequirement {
        source_partition: partitions[0].partition_id,
        target_partition: partitions[0].partition_id,
        communication_type: TensorCommunicationPattern::AllReduce,
        data_size: 16,
    };
    assert!(
        tp.handle_communication_requirements(&[requirement]).is_err(),
        "an all-reduce with no registered tensor must fail loudly"
    );
}

#[test]
fn test_optimal_tensor_config_calculation() {
    // Use 10B parameters (40GB memory) with 8GB per device
    // This requires at least 5 devices, so tensor_parallel_size > 1
    let config = utils::calculate_optimal_tensor_config(
        10_000_000_000,         // 10B parameters (40GB memory)
        8 * 1024 * 1024 * 1024, // 8GB memory per device
        8,                      // world size
    )
    .expect("operation failed in test");

    assert!(
        config.tensor_parallel_size > 1,
        "Expected tensor_parallel_size > 1, got {}",
        config.tensor_parallel_size
    );
}

#[test]
fn test_communication_overhead_estimation() {
    let config = TensorParallelismConfig::default();
    let overhead = utils::estimate_communication_overhead(&config, 1024 * 1024, 100);
    assert!(overhead > 0.0);
}

#[test]
fn test_memory_savings_calculation() {
    let savings = utils::calculate_memory_savings(1_000_000_000, 4);
    assert!(savings > 0.0 && savings < 1.0);
}
