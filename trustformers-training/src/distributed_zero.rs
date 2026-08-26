//! ZeRO stage-1: optimizer-state sharding across data-parallel ranks.
//!
//! # What ZeRO stage 1 does
//!
//! In plain data-parallel training every rank keeps a full copy of the
//! optimizer state (for Adam: two moment tensors per parameter, i.e. twice the
//! model size in fp32). ZeRO stage 1 partitions the *parameters* into disjoint
//! shards, one per rank, and lets each rank instantiate optimizer state for its
//! own shard only. A step then runs:
//!
//! 1. **All-reduce** the gradients and divide by the world size, so every rank
//!    holds the mean gradient (identical on all ranks).
//! 2. **Local update** of the parameters this rank owns, using this rank's
//!    optimizer — the only place its state lives.
//! 3. **Broadcast** each updated parameter from its owner, so every rank leaves
//!    the step with the identical, complete parameter set.
//!
//! The result is mathematically identical to non-sharded data-parallel training
//! with the same optimizer; only the memory footprint of the optimizer state
//! changes (roughly `1/world_size` of it per rank).
//!
//! Stages 2 (gradient sharding) and 3 (parameter sharding) are **not**
//! implemented here; [`ZeroStage`] names them and
//! [`ZeroStage1Optimizer::new`] refuses them with a clear error rather than
//! silently running stage 1 under a stage-3 label. `trustformers-optim`'s
//! `zero` module carries the fuller machinery.
//!
//! # Communication
//!
//! Every collective goes through [`ProcessGroup`], so the backend is whatever
//! the caller built: [`crate::distributed_collective::InProcessProcessGroup`]
//! for threads, [`crate::distributed_collective::TcpProcessGroup`] for
//! processes. No collective is simulated.
//!
//! # Example
//!
//! ```
//! use std::collections::HashMap;
//! use std::sync::Arc;
//! use trustformers_core::tensor::Tensor;
//! use trustformers_training::distributed::ProcessGroup;
//! use trustformers_training::distributed_collective::run_in_process;
//! use trustformers_training::distributed_zero::{ZeroStage, ZeroStage1Optimizer};
//!
//! # fn main() -> anyhow::Result<()> {
//! let per_rank = run_in_process(4, |rank, group| -> anyhow::Result<Vec<f32>> {
//!     let group: Arc<dyn ProcessGroup> = group;
//!     let mut zero = ZeroStage1Optimizer::new(
//!         group,
//!         trustformers_optim::SGD::new(0.1, 0.0, 0.0, false),
//!         ZeroStage::Stage1,
//!     )?;
//!
//!     let mut parameters = HashMap::from([
//!         ("w".to_string(), Tensor::zeros(&[4])?),
//!         ("b".to_string(), Tensor::zeros(&[2])?),
//!     ]);
//!     zero.register_parameters(&parameters)?;
//!
//!     // Every rank contributes a different gradient; ZeRO averages them.
//!     let mut gradients = HashMap::from([
//!         ("w".to_string(), Tensor::from_slice(&[rank as f32; 4], &[4])?),
//!         ("b".to_string(), Tensor::from_slice(&[1.0f32; 2], &[2])?),
//!     ]);
//!     zero.step(&mut parameters, &mut gradients)?;
//!
//!     Ok(parameters["w"].to_vec_f32()?)
//! })?;
//!
//! // mean gradient = (0+1+2+3)/4 = 1.5, so w = -lr * 1.5 = -0.15 on every rank.
//! for values in per_rank {
//!     assert!((values?[0] + 0.15).abs() < 1e-5);
//! }
//! # Ok(())
//! # }
//! ```

use crate::distributed::{DistributedError, ProcessGroup};
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// ZeRO partitioning stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroStage {
    /// No sharding: every rank keeps full optimizer state.
    Disabled,
    /// Optimizer state is sharded; gradients and parameters are replicated.
    Stage1,
    /// Optimizer state **and** gradients are sharded. Not implemented by this
    /// type.
    Stage2,
    /// Optimizer state, gradients **and** parameters are sharded. Not
    /// implemented by this type.
    Stage3,
}

/// How a parameter shard is assigned to a rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShardAssignment {
    /// Greedy balancing by element count: parameters are visited largest-first
    /// (ties broken by name) and each goes to the rank holding the fewest
    /// elements so far. Deterministic, so every rank derives the same map.
    #[default]
    BalancedBySize,
    /// Round-robin over the sorted parameter names. Simple and also
    /// deterministic, but ignores tensor sizes.
    RoundRobin,
}

/// Per-rank memory accounting for the sharded optimizer state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZeroMemoryReport {
    /// Elements this rank owns.
    pub owned_elements: usize,
    /// Elements in the full model.
    pub total_elements: usize,
    /// Parameters this rank owns.
    pub owned_parameters: usize,
    /// Parameters in the full model.
    pub total_parameters: usize,
}

impl ZeroMemoryReport {
    /// Fraction of the model this rank keeps optimizer state for, in `[0, 1]`.
    ///
    /// Multiply by the optimizer's per-element state size to get the actual
    /// saving; this type deliberately reports the *measured* partition rather
    /// than an estimated byte count for an optimizer it cannot inspect.
    pub fn owned_fraction(&self) -> f32 {
        if self.total_elements == 0 {
            0.0
        } else {
            self.owned_elements as f32 / self.total_elements as f32
        }
    }
}

/// Data-parallel optimizer with ZeRO stage-1 optimizer-state sharding.
///
/// See the [module documentation](self) for the algorithm.
pub struct ZeroStage1Optimizer<O: Optimizer> {
    process_group: Arc<dyn ProcessGroup>,
    optimizer: O,
    assignment: ShardAssignment,
    /// `parameter name -> owning rank`, identical on every rank.
    owners: HashMap<String, usize>,
    /// Names this rank owns, in sorted order.
    owned: Vec<String>,
    /// Element counts used to build the assignment, kept for reporting.
    sizes: HashMap<String, usize>,
}

impl<O: Optimizer> ZeroStage1Optimizer<O> {
    /// Build a stage-1 sharded optimizer over `process_group`.
    ///
    /// # Errors
    ///
    /// * [`ZeroStage::Stage2`] / [`ZeroStage::Stage3`] are rejected: this type
    ///   implements stage 1 only, and running stage 1 while reporting stage 3
    ///   would misstate the memory characteristics of the job.
    /// * [`ZeroStage::Disabled`] is rejected too — use
    ///   [`crate::distributed::DataParallelTrainer`] for unsharded training.
    pub fn new(
        process_group: Arc<dyn ProcessGroup>,
        optimizer: O,
        stage: ZeroStage,
    ) -> Result<Self> {
        match stage {
            ZeroStage::Stage1 => {},
            ZeroStage::Disabled => {
                return Err(DistributedError::InvalidConfig(
                    "ZeroStage::Disabled requests no sharding; use DataParallelTrainer instead of \
                     ZeroStage1Optimizer"
                        .to_string(),
                )
                .into())
            },
            ZeroStage::Stage2 | ZeroStage::Stage3 => {
                return Err(DistributedError::UnsupportedOperation {
                    operation: "ZeRO stage 2/3",
                    reason: "ZeroStage1Optimizer shards optimizer state only. Gradient and \
                             parameter sharding are not implemented here; see the `zero` module \
                             of trustformers-optim",
                }
                .into())
            },
        }

        if process_group.world_size() == 0 {
            return Err(
                DistributedError::InvalidConfig("world_size must be >= 1".to_string()).into(),
            );
        }

        Ok(Self {
            process_group,
            optimizer,
            assignment: ShardAssignment::default(),
            owners: HashMap::new(),
            owned: Vec::new(),
            sizes: HashMap::new(),
        })
    }

    /// Choose how parameters are assigned to ranks. Must be set identically on
    /// every rank before [`ZeroStage1Optimizer::register_parameters`].
    pub fn with_assignment(mut self, assignment: ShardAssignment) -> Self {
        self.assignment = assignment;
        self
    }

    /// Partition `parameters` across the ranks of the process group.
    ///
    /// The partition is derived purely from the parameter names and their
    /// element counts, so every rank computes the same map without
    /// communicating. Call this once, before stepping, and again whenever the
    /// parameter set changes.
    pub fn register_parameters(&mut self, parameters: &HashMap<String, Tensor>) -> Result<()> {
        let world_size = self.process_group.world_size().max(1);
        let rank = self.process_group.rank();

        let mut names: Vec<String> = parameters.keys().cloned().collect();
        names.sort();

        self.sizes = names
            .iter()
            .map(|name| {
                let length = parameters.get(name).map(|tensor| tensor.len()).unwrap_or(0);
                (name.clone(), length)
            })
            .collect();

        self.owners = match self.assignment {
            ShardAssignment::RoundRobin => names
                .iter()
                .enumerate()
                .map(|(index, name)| (name.clone(), index % world_size))
                .collect(),
            ShardAssignment::BalancedBySize => {
                // Largest-first greedy bin packing. Sorting by (size desc,
                // name asc) makes the order total and therefore identical on
                // every rank.
                let mut ordered: Vec<&String> = names.iter().collect();
                ordered.sort_by(|left, right| {
                    let left_size = self.sizes.get(*left).copied().unwrap_or(0);
                    let right_size = self.sizes.get(*right).copied().unwrap_or(0);
                    right_size.cmp(&left_size).then_with(|| left.cmp(right))
                });

                let mut load = vec![0usize; world_size];
                let mut owners = HashMap::with_capacity(ordered.len());
                for name in ordered {
                    let (target, _) = load
                        .iter()
                        .enumerate()
                        .min_by_key(|(index, elements)| (**elements, *index))
                        .ok_or_else(|| anyhow!("world size {world_size} has no ranks"))?;
                    load[target] += self.sizes.get(name).copied().unwrap_or(0);
                    owners.insert(name.clone(), target);
                }
                owners
            },
        };

        self.owned = names
            .into_iter()
            .filter(|name| self.owners.get(name).copied() == Some(rank))
            .collect();

        Ok(())
    }

    /// The rank that owns `name`'s optimizer state, if it is registered.
    pub fn owner_of(&self, name: &str) -> Option<usize> {
        self.owners.get(name).copied()
    }

    /// Names this rank owns, in sorted order.
    pub fn owned_parameters(&self) -> &[String] {
        &self.owned
    }

    /// Measured partition of the model held by this rank.
    pub fn memory_report(&self) -> ZeroMemoryReport {
        let owned_elements =
            self.owned.iter().map(|name| self.sizes.get(name).copied().unwrap_or(0)).sum();
        ZeroMemoryReport {
            owned_elements,
            total_elements: self.sizes.values().copied().sum(),
            owned_parameters: self.owned.len(),
            total_parameters: self.sizes.len(),
        }
    }

    /// Borrow the wrapped optimizer.
    pub fn optimizer(&self) -> &O {
        &self.optimizer
    }

    /// Mutably borrow the wrapped optimizer.
    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.optimizer
    }

    /// Average `gradients` across the ranks with a real all-reduce.
    ///
    /// Names are visited in sorted order so every rank issues the identical
    /// collective sequence.
    pub fn all_reduce_gradients(&self, gradients: &mut HashMap<String, Tensor>) -> Result<()> {
        if gradients.is_empty() {
            return Ok(());
        }
        let world_size = self.process_group.world_size().max(1);

        let mut names: Vec<String> = gradients.keys().cloned().collect();
        names.sort();

        let mut tensors: Vec<Tensor> =
            names.iter().filter_map(|name| gradients.get(name).cloned()).collect();
        if tensors.len() != names.len() {
            return Err(anyhow!(
                "a gradient vanished while preparing the all-reduce"
            ));
        }

        self.process_group.all_reduce(&mut tensors)?;

        for (name, reduced) in names.iter().zip(tensors) {
            let averaged = if world_size > 1 {
                reduced.scalar_mul(1.0 / world_size as f32)?
            } else {
                reduced
            };
            if let Some(slot) = gradients.get_mut(name) {
                *slot = averaged;
            }
        }
        Ok(())
    }

    /// Run one sharded optimizer step.
    ///
    /// `gradients` are all-reduced and averaged in place; `parameters` are
    /// updated on their owning rank and then broadcast, so all ranks return
    /// with identical values.
    ///
    /// # Errors
    ///
    /// Fails when a registered parameter has no gradient, when a gradient has
    /// no parameter, or when the parameter set was never registered.
    pub fn step(
        &mut self,
        parameters: &mut HashMap<String, Tensor>,
        gradients: &mut HashMap<String, Tensor>,
    ) -> Result<()> {
        if self.owners.is_empty() {
            return Err(DistributedError::InvalidConfig(
                "ZeroStage1Optimizer::step called before register_parameters".to_string(),
            )
            .into());
        }

        for name in gradients.keys() {
            if !self.owners.contains_key(name) {
                return Err(anyhow!(
                    "gradient `{name}` has no registered parameter; call register_parameters again \
                     after changing the parameter set"
                ));
            }
        }

        self.all_reduce_gradients(gradients)?;

        // Update only this rank's shard: its optimizer state exists for these
        // parameters and no others, which is the entire point of stage 1.
        for name in &self.owned {
            let gradient = gradients
                .get(name)
                .ok_or_else(|| anyhow!("no gradient for owned parameter `{name}`"))?;
            let parameter = parameters
                .get_mut(name)
                .ok_or_else(|| anyhow!("no parameter tensor for owned parameter `{name}`"))?;
            self.optimizer.update(parameter, gradient)?;
        }
        self.optimizer.step();

        self.synchronize_parameters(parameters)?;
        Ok(())
    }

    /// Broadcast every registered parameter from its owner so all ranks agree.
    pub fn synchronize_parameters(&self, parameters: &mut HashMap<String, Tensor>) -> Result<()> {
        if self.process_group.world_size() <= 1 {
            return Ok(());
        }

        let mut names: Vec<String> = self.owners.keys().cloned().collect();
        names.sort();

        for name in names {
            let owner = self
                .owners
                .get(&name)
                .copied()
                .ok_or_else(|| anyhow!("parameter `{name}` lost its owner"))?;
            let tensor = parameters
                .get_mut(&name)
                .ok_or_else(|| anyhow!("no parameter tensor named `{name}` to synchronize"))?;
            self.process_group.broadcast(tensor, owner)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
