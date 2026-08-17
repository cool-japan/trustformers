//! Multi-rank tests for ZeRO stage-1 optimizer-state sharding.
//!
//! Every test here spawns real ranks as threads over
//! [`crate::distributed_collective::InProcessProcessGroup`], so the collectives
//! move real bytes. A test that passed against a no-op or value-scaling
//! "collective" would be worthless, which is exactly what these assertions rule
//! out: the sharded step must land on the *same* parameters as an unsharded
//! reference step computed with the true mean gradient.

use super::*;
use crate::distributed::SimulatedProcessGroup;
use crate::distributed_collective::run_in_process;
use trustformers_optim::SGD;

/// Learning rate small enough that the closed-form reference below stays exact
/// in f32 for the tiny tensors used here.
const LR: f32 = 0.125;

fn sgd() -> SGD {
    SGD::new(LR, 0.0, 0.0, false)
}

fn parameter_set() -> Result<HashMap<String, Tensor>> {
    Ok(HashMap::from([
        ("layer0.weight".to_string(), Tensor::zeros(&[8])?),
        ("layer0.bias".to_string(), Tensor::zeros(&[2])?),
        ("layer1.weight".to_string(), Tensor::zeros(&[4])?),
        ("layer1.bias".to_string(), Tensor::zeros(&[1])?),
    ]))
}

/// Gradient contributed by `rank`: every element of `name` gets a value that
/// depends on both the rank and the parameter, so a collective that drops or
/// rescales data cannot accidentally produce the right answer.
fn rank_gradients(
    rank: usize,
    parameters: &HashMap<String, Tensor>,
) -> Result<HashMap<String, Tensor>> {
    let mut gradients = HashMap::with_capacity(parameters.len());
    for (index, name) in sorted_keys(parameters).into_iter().enumerate() {
        let length =
            parameters.get(&name).ok_or_else(|| anyhow!("missing parameter {name}"))?.len();
        let value = (rank as f32 + 1.0) * (index as f32 + 1.0);
        gradients.insert(name, Tensor::from_slice(&vec![value; length], &[length])?);
    }
    Ok(gradients)
}

fn sorted_keys(map: &HashMap<String, Tensor>) -> Vec<String> {
    let mut names: Vec<String> = map.keys().cloned().collect();
    names.sort();
    names
}

#[test]
fn stage2_and_stage3_are_refused_instead_of_silently_running_stage1() {
    let group: Arc<dyn ProcessGroup> = Arc::new(SimulatedProcessGroup::new(0, 1));

    for stage in [ZeroStage::Stage2, ZeroStage::Stage3] {
        let Err(error) = ZeroStage1Optimizer::new(Arc::clone(&group), sgd(), stage) else {
            panic!("stage 2/3 must be refused in test");
        };
        assert!(
            error.to_string().contains("ZeRO stage 2/3"),
            "unexpected error: {error}"
        );
    }

    let Err(error) = ZeroStage1Optimizer::new(Arc::clone(&group), sgd(), ZeroStage::Disabled)
    else {
        panic!("Disabled must be refused in test");
    };
    assert!(error.to_string().contains("DataParallelTrainer"), "{error}");
}

#[test]
fn step_before_register_parameters_is_an_error() {
    let group: Arc<dyn ProcessGroup> = Arc::new(SimulatedProcessGroup::new(0, 1));
    let mut zero = ZeroStage1Optimizer::new(group, sgd(), ZeroStage::Stage1)
        .expect("stage 1 must build in test");

    let mut parameters = parameter_set().expect("parameters must build in test");
    let mut gradients = rank_gradients(0, &parameters).expect("gradients must build in test");

    let error = zero
        .step(&mut parameters, &mut gradients)
        .expect_err("stepping without registration must fail in test");
    assert!(error.to_string().contains("register_parameters"), "{error}");
}

#[test]
fn balanced_assignment_is_deterministic_and_partitions_every_parameter() {
    let parameters = parameter_set().expect("parameters must build in test");
    let world_size = 4;

    let mut per_rank_owned: Vec<Vec<String>> = Vec::new();
    let mut owner_maps: Vec<HashMap<String, usize>> = Vec::new();

    for rank in 0..world_size {
        let group: Arc<dyn ProcessGroup> = Arc::new(SimulatedProcessGroup::new(rank, world_size));
        let mut zero = ZeroStage1Optimizer::new(group, sgd(), ZeroStage::Stage1)
            .expect("stage 1 must build in test");
        zero.register_parameters(&parameters)
            .expect("registration must succeed in test");
        per_rank_owned.push(zero.owned_parameters().to_vec());
        owner_maps.push(
            sorted_keys(&parameters)
                .into_iter()
                .filter_map(|name| zero.owner_of(&name).map(|owner| (name, owner)))
                .collect(),
        );
    }

    // Every rank derives the identical owner map without communicating.
    for map in &owner_maps {
        assert_eq!(
            map, &owner_maps[0],
            "the shard map must not depend on the rank"
        );
    }

    // The shards form a partition: disjoint, and together the whole model.
    let mut all_owned: Vec<String> = per_rank_owned.iter().flatten().cloned().collect();
    all_owned.sort();
    assert_eq!(all_owned, sorted_keys(&parameters));

    // Largest-first greedy balancing must put the two largest tensors (8 and 4
    // elements) on different ranks.
    let owners = &owner_maps[0];
    assert_ne!(
        owners.get("layer0.weight"),
        owners.get("layer1.weight"),
        "the two largest tensors must not land on the same rank"
    );
}

#[test]
fn round_robin_assignment_walks_the_sorted_names() {
    let parameters = parameter_set().expect("parameters must build in test");
    let group: Arc<dyn ProcessGroup> = Arc::new(SimulatedProcessGroup::new(0, 2));
    let mut zero = ZeroStage1Optimizer::new(group, sgd(), ZeroStage::Stage1)
        .expect("stage 1 must build in test")
        .with_assignment(ShardAssignment::RoundRobin);
    zero.register_parameters(&parameters)
        .expect("registration must succeed in test");

    for (index, name) in sorted_keys(&parameters).into_iter().enumerate() {
        assert_eq!(
            zero.owner_of(&name),
            Some(index % 2),
            "round robin on `{name}`"
        );
    }
}

#[test]
fn memory_report_measures_the_real_partition() {
    let parameters = parameter_set().expect("parameters must build in test");
    let world_size = 2;
    let mut owned_total = 0usize;

    for rank in 0..world_size {
        let group: Arc<dyn ProcessGroup> = Arc::new(SimulatedProcessGroup::new(rank, world_size));
        let mut zero = ZeroStage1Optimizer::new(group, sgd(), ZeroStage::Stage1)
            .expect("stage 1 must build in test");
        zero.register_parameters(&parameters)
            .expect("registration must succeed in test");

        let report = zero.memory_report();
        assert_eq!(report.total_elements, 8 + 2 + 4 + 1);
        assert_eq!(report.total_parameters, 4);
        assert!(report.owned_fraction() > 0.0 && report.owned_fraction() < 1.0);
        owned_total += report.owned_elements;
    }

    assert_eq!(owned_total, 15, "the shards must sum to the whole model");
}

#[test]
fn sharded_step_matches_the_unsharded_reference_on_every_rank() {
    const WORLD: usize = 4;

    let per_rank = run_in_process(WORLD, |rank, group| -> Result<Vec<(String, Vec<f32>)>> {
        let group: Arc<dyn ProcessGroup> = group;
        let mut zero = ZeroStage1Optimizer::new(group, sgd(), ZeroStage::Stage1)?;

        let mut parameters = parameter_set()?;
        zero.register_parameters(&parameters)?;

        let mut gradients = rank_gradients(rank, &parameters)?;
        zero.step(&mut parameters, &mut gradients)?;

        let mut out = Vec::new();
        for name in sorted_keys(&parameters) {
            let tensor =
                parameters.get(&name).ok_or_else(|| anyhow!("parameter {name} vanished"))?;
            out.push((name, tensor.to_vec_f32()?));
        }
        Ok(out)
    })
    .expect("in-process run must succeed in test");

    // Reference: plain SGD from zero with the true mean gradient.
    // grad(rank, index) = (rank + 1) * (index + 1); mean over ranks 0..3 is
    // 2.5 * (index + 1). One SGD step from zero gives -lr * mean.
    let reference: Vec<f32> = (0..4).map(|index| -LR * 2.5 * (index as f32 + 1.0)).collect();

    for (rank, values) in per_rank.into_iter().enumerate() {
        let values = values.expect("each rank must complete its step in test");
        assert_eq!(values.len(), 4, "rank {rank}");
        for (index, (name, elements)) in values.into_iter().enumerate() {
            for element in elements {
                approx::assert_relative_eq!(element, reference[index], epsilon = 1e-6);
            }
            assert!(!name.is_empty());
        }
    }
}

#[test]
fn every_rank_leaves_the_step_with_identical_parameters() {
    const WORLD: usize = 3;

    let per_rank = run_in_process(WORLD, |rank, group| -> Result<Vec<f32>> {
        let group: Arc<dyn ProcessGroup> = group;
        let mut zero = ZeroStage1Optimizer::new(group, sgd(), ZeroStage::Stage1)?;

        let mut parameters = parameter_set()?;
        zero.register_parameters(&parameters)?;
        let mut gradients = rank_gradients(rank, &parameters)?;
        zero.step(&mut parameters, &mut gradients)?;

        // Flatten in sorted order so the comparison across ranks is aligned.
        let mut flat = Vec::new();
        for name in sorted_keys(&parameters) {
            let tensor =
                parameters.get(&name).ok_or_else(|| anyhow!("parameter {name} vanished"))?;
            flat.extend(tensor.to_vec_f32()?);
        }
        Ok(flat)
    })
    .expect("in-process run must succeed in test");

    let mut flattened = Vec::new();
    for values in per_rank {
        flattened.push(values.expect("each rank must complete its step in test"));
    }
    for values in &flattened {
        assert_eq!(
            values, &flattened[0],
            "ZeRO must broadcast every owner's update, so all ranks agree bit-exactly"
        );
    }
    // And the step actually moved the parameters away from zero.
    assert!(flattened[0].iter().any(|value| value.abs() > 1e-6));
}

#[test]
fn all_reduce_gradients_produces_the_true_mean() {
    const WORLD: usize = 4;

    let per_rank = run_in_process(WORLD, |rank, group| -> Result<Vec<f32>> {
        let group: Arc<dyn ProcessGroup> = group;
        let zero = ZeroStage1Optimizer::new(group, sgd(), ZeroStage::Stage1)?;

        let mut gradients = HashMap::from([(
            "g".to_string(),
            Tensor::from_slice(&[rank as f32, 2.0 * rank as f32], &[2])?,
        )]);
        zero.all_reduce_gradients(&mut gradients)?;
        Ok(gradients.get("g").ok_or_else(|| anyhow!("gradient vanished"))?.to_vec_f32()?)
    })
    .expect("in-process run must succeed in test");

    // mean(0,1,2,3) = 1.5 and mean(0,2,4,6) = 3.0
    for values in per_rank {
        let values = values.expect("each rank must complete the all-reduce in test");
        approx::assert_relative_eq!(values[0], 1.5f32, epsilon = 1e-6);
        approx::assert_relative_eq!(values[1], 3.0f32, epsilon = 1e-6);
    }
}

#[test]
fn a_gradient_without_a_registered_parameter_is_rejected() {
    let group: Arc<dyn ProcessGroup> = Arc::new(SimulatedProcessGroup::new(0, 1));
    let mut zero = ZeroStage1Optimizer::new(group, sgd(), ZeroStage::Stage1)
        .expect("stage 1 must build in test");

    let mut parameters = parameter_set().expect("parameters must build in test");
    zero.register_parameters(&parameters)
        .expect("registration must succeed in test");

    let mut gradients = rank_gradients(0, &parameters).expect("gradients must build in test");
    gradients.insert(
        "ghost".to_string(),
        Tensor::zeros(&[2]).expect("tensor must build in test"),
    );

    let error = zero
        .step(&mut parameters, &mut gradients)
        .expect_err("an unregistered gradient must fail in test");
    assert!(error.to_string().contains("ghost"), "{error}");
}
