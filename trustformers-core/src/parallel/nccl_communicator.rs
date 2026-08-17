// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! NCCL-shaped communicator implementation for multi-GPU collective operations
//!
//! There is no real NVIDIA NCCL binding here: this crate is pure Rust with
//! no C/C++ FFI in its default build, and no `-sys` crate wraps NCCL. What
//! this module does provide, behind the `nccl` feature, is the same
//! `Communicator` surface (`all_reduce`, `all_gather`, `broadcast`,
//! `reduce_scatter`, `send`/`recv`) backed by real in-process, rank-indexed
//! shared buffers (see [`super::local_communicator`]) rather than the no-ops
//! and zero-filled tensors this module used to return while still claiming
//! `Ok`. Every rank named here must live in the same OS process (e.g. one
//! thread per rank via [`NcclCommunicator::init_all`]); real cross-node
//! execution requires `trustformers-training`'s TCP backend.

use super::local_communicator::InProcessGroup;
use super::Communicator;
use crate::errors::TrustformersError;
use crate::tensor::Tensor;
use std::sync::Arc;

type Result<T> = std::result::Result<T, TrustformersError>;

/// NCCL-shaped communicator for GPU-to-GPU communication.
///
/// Despite the name (kept for API continuity with callers that select
/// `CommunicationBackend::Nccl`), this performs real collective math over
/// an in-process [`InProcessGroup`] rather than talking to a GPU or a real
/// NCCL runtime - see the module docs.
#[cfg(feature = "nccl")]
pub struct NcclCommunicator {
    /// Rank of this process in the communication group
    rank: usize,
    /// Total number of processes in the communication group
    world_size: usize,
    /// Shared rank-indexed state for the group this communicator belongs
    /// to (see [`InProcessGroup`]).
    group: Arc<InProcessGroup>,
    /// Device ID for this communicator (bookkeeping only - no CUDA context
    /// is opened).
    device_id: i32,
}

#[cfg(feature = "nccl")]
impl NcclCommunicator {
    /// Create a standalone, single-member communicator for `device_id`.
    ///
    /// Only valid for `world_size == 1`: a lone `NcclCommunicator` has no
    /// way to rendezvous with `world_size - 1` other ranks it knows nothing
    /// about. For real multi-rank groups, build every rank together with
    /// [`NcclCommunicator::init_all`], which shares one group across all of
    /// them.
    pub fn new(rank: usize, world_size: usize, device_id: i32) -> Result<Self> {
        if rank >= world_size {
            return Err(TrustformersError::invalid_input(format!(
                "Rank {} must be less than world_size {}",
                rank, world_size
            )));
        }
        if world_size != 1 {
            return Err(TrustformersError::invalid_input(format!(
                "NcclCommunicator::new only supports world_size == 1 (got {}); a lone \
                 communicator cannot rendezvous with the other {} rank(s) it would need to \
                 share a group with. Use NcclCommunicator::init_all to build every rank \
                 together.",
                world_size,
                world_size.saturating_sub(1)
            )));
        }

        Ok(Self {
            rank,
            world_size,
            group: InProcessGroup::new(world_size),
            device_id,
        })
    }

    /// Initialize NCCL-shaped communicators for all processes.
    ///
    /// Builds `world_size` communicators that share one real
    /// [`InProcessGroup`], with ranks `0..world_size` - e.g. hand each
    /// returned communicator to its own thread to exercise real multi-rank
    /// collectives in-process.
    pub fn init_all(world_size: usize, device_ids: &[i32]) -> Result<Vec<Self>> {
        if device_ids.len() != world_size {
            return Err(TrustformersError::invalid_input(
                "Number of device IDs must match world size".to_string(),
            ));
        }

        let group = InProcessGroup::new(world_size);
        Ok(device_ids
            .iter()
            .enumerate()
            .map(|(rank, &device_id)| Self {
                rank,
                world_size,
                group: Arc::clone(&group),
                device_id,
            })
            .collect())
    }

    /// Get the rank of this communicator
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get the world size
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Get the device ID
    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    fn elementwise_sum(tensors: &[Tensor]) -> Result<Tensor> {
        let mut iter = tensors.iter();
        let first = iter.next().ok_or_else(|| {
            TrustformersError::invalid_input("elementwise_sum: input list is empty".to_string())
        })?;
        let mut acc = first.clone();
        for t in iter {
            acc = acc.add(t)?;
        }
        Ok(acc)
    }

    /// Perform all-reduce (sum) across the group.
    fn nccl_all_reduce(&self, tensor: &mut Tensor) -> Result<()> {
        if tensor.shape().is_empty() {
            return Err(TrustformersError::invalid_input(
                "Cannot perform all-reduce on empty tensor".to_string(),
            ));
        }
        let summed = self.group.collective(self.rank, tensor.clone(), Self::elementwise_sum)?;
        *tensor = summed;
        Ok(())
    }

    /// Perform all-gather across the group: every rank's tensor is stacked
    /// along a new leading axis, so the result has shape
    /// `[world_size, ...input_shape]` and really contains every rank's
    /// data (not zeros).
    fn nccl_all_gather(&self, tensor: &Tensor) -> Result<Tensor> {
        self.group.collective(self.rank, tensor.clone(), |values| {
            let mut expanded = Vec::with_capacity(values.len());
            for t in values {
                let mut shape = t.shape();
                shape.insert(0, 1);
                expanded.push(t.reshape(&shape)?);
            }
            if expanded.len() == 1 {
                return expanded.into_iter().next().ok_or_else(|| {
                    TrustformersError::runtime_error(
                        "nccl_all_gather: unreachable empty vec".to_string(),
                    )
                });
            }
            Tensor::concat(&expanded, 0)
        })
    }

    /// Perform broadcast from `root` across the group.
    fn nccl_broadcast(&self, tensor: &mut Tensor, root: usize) -> Result<()> {
        if root >= self.world_size {
            return Err(TrustformersError::invalid_input(format!(
                "Root rank {} must be less than world_size {}",
                root, self.world_size
            )));
        }
        let broadcasted = self.group.collective(self.rank, tensor.clone(), move |values| {
            values.get(root).cloned().ok_or_else(|| {
                TrustformersError::runtime_error(
                    "nccl_broadcast: root rank missing from deposited values".to_string(),
                )
            })
        })?;
        *tensor = broadcasted;
        Ok(())
    }

    /// Synchronize all processes in the communicator.
    ///
    /// There is no separate GPU stream to synchronize (see the module
    /// docs), so this rendezvous is itself the synchronization: every rank
    /// must reach this call before any of them proceeds.
    pub fn barrier(&self) -> Result<()> {
        let mut placeholder = Tensor::zeros(&[1])?;
        // Reuse the collective machinery purely for its barrier semantics;
        // the value itself is discarded.
        self.nccl_all_reduce(&mut placeholder)?;
        Ok(())
    }

    /// Destroy the NCCL communicator and clean up resources.
    ///
    /// There is no external handle to release (see the module docs), so
    /// this is a documented no-op rather than a call into a nonexistent
    /// runtime.
    pub fn destroy(&mut self) -> Result<()> {
        Ok(())
    }
}

#[cfg(feature = "nccl")]
impl Communicator for NcclCommunicator {
    fn all_reduce(&self, tensor: &mut Tensor) -> Result<()> {
        self.nccl_all_reduce(tensor)
    }

    fn all_gather(&self, tensor: &Tensor, _split_dim: usize) -> Result<Tensor> {
        self.nccl_all_gather(tensor)
    }

    fn reduce_scatter(&self, tensor: &Tensor, split_dim: usize) -> Result<Tensor> {
        let summed = self.group.collective(self.rank, tensor.clone(), Self::elementwise_sum)?;
        let shape = summed.shape();
        if split_dim >= shape.len() {
            return Err(TrustformersError::invalid_input(format!(
                "reduce_scatter: split_dim {} out of bounds for shape {:?}",
                split_dim, shape
            )));
        }
        let dim_size = shape[split_dim];
        if self.world_size == 0 || dim_size % self.world_size != 0 {
            return Err(TrustformersError::invalid_input(format!(
                "reduce_scatter: dimension {} size {} is not evenly divisible by world_size {}",
                split_dim, dim_size, self.world_size
            )));
        }
        let chunk = dim_size / self.world_size;
        let start = self.rank * chunk;
        summed.slice(split_dim, start, start + chunk)
    }

    fn broadcast(&self, tensor: &mut Tensor, root: usize) -> Result<()> {
        self.nccl_broadcast(tensor, root)
    }

    fn send(&self, tensor: &Tensor, dest: usize) -> Result<()> {
        if dest >= self.world_size {
            return Err(TrustformersError::invalid_input(format!(
                "send: destination rank {} is out of bounds for world_size {}",
                dest, self.world_size
            )));
        }
        self.group.post(self.rank, dest, tensor.clone());
        Ok(())
    }

    fn recv(&self, shape: &[usize], src: usize) -> Result<Tensor> {
        if src >= self.world_size {
            return Err(TrustformersError::invalid_input(format!(
                "recv: source rank {} is out of bounds for world_size {}",
                src, self.world_size
            )));
        }
        let tensor = self.group.take(src, self.rank)?;
        if tensor.shape() != shape {
            return Err(TrustformersError::invalid_input(format!(
                "recv: received tensor with shape {:?}, expected {:?}",
                tensor.shape(),
                shape
            )));
        }
        Ok(tensor)
    }
}

#[cfg(feature = "nccl")]
impl Drop for NcclCommunicator {
    fn drop(&mut self) {
        if let Err(e) = self.destroy() {
            log::error!("Failed to destroy NCCL communicator: {}", e);
        }
    }
}

// Provide factory function for creating NCCL communicators
#[cfg(feature = "nccl")]
pub fn create_nccl_communicator(
    rank: usize,
    world_size: usize,
    device_id: i32,
) -> Result<Arc<dyn Communicator>> {
    let comm = NcclCommunicator::new(rank, world_size, device_id)?;
    Ok(Arc::new(comm))
}

#[cfg(not(feature = "nccl"))]
pub fn create_nccl_communicator(
    _rank: usize,
    _world_size: usize,
    _device_id: i32,
) -> Result<Arc<dyn Communicator>> {
    Err(TrustformersError::invalid_config(
        "NCCL feature not enabled. Compile with --features nccl to use NCCL communicator"
            .to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "nccl")]
    fn test_nccl_communicator_creation() {
        let comm = NcclCommunicator::new(0, 1, 0).expect("operation failed in test");
        assert_eq!(comm.rank(), 0);
        assert_eq!(comm.world_size(), 1);
        assert_eq!(comm.device_id(), 0);
    }

    #[test]
    #[cfg(feature = "nccl")]
    fn test_invalid_rank() {
        let result = NcclCommunicator::new(2, 2, 0);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "nccl")]
    fn test_standalone_new_rejects_multi_rank() {
        // A lone `new()` call cannot rendezvous with peers it knows
        // nothing about; only `init_all` can build a real multi-rank group.
        let result = NcclCommunicator::new(0, 2, 0);
        assert!(result.is_err());
    }

    #[test]
    #[cfg(feature = "nccl")]
    fn test_single_rank_operations() -> Result<()> {
        let comm = NcclCommunicator::new(0, 1, 0)?;

        let mut tensor = Tensor::ones(&[4, 4])?;
        comm.all_reduce(&mut tensor)?;
        assert_eq!(tensor.data()?, vec![1.0f32; 16]);

        let gathered = comm.all_gather(&tensor, 0)?;
        assert_eq!(gathered.shape()[0], 1);

        comm.broadcast(&mut tensor, 0)?;
        comm.barrier()?;

        Ok(())
    }

    /// Regression test: before this fix, `nccl_all_reduce` validated the
    /// tensor and then returned `Ok(())` without ever modifying it - a
    /// no-op disguised as a completed reduction. Three ranks built via
    /// `init_all` (so they share one real group) each contribute a
    /// distinct tensor; a correct sum-all-reduce combines all three.
    #[test]
    #[cfg(feature = "nccl")]
    fn test_all_reduce_combines_every_ranks_real_data() {
        let comms = NcclCommunicator::init_all(3, &[0, 1, 2]).expect("init_all");
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                std::thread::spawn(move || {
                    let rank = comm.rank();
                    let mut tensor =
                        Tensor::from_vec(vec![(rank + 1) as f32], &[1]).expect("tensor");
                    comm.all_reduce(&mut tensor).expect("all_reduce");
                    tensor.data().expect("data")
                })
            })
            .collect();

        for h in handles {
            let data = h.join().expect("thread");
            // Ranks contribute 1, 2, 3 -> sum = 6.
            assert_eq!(
                data,
                vec![6.0f32],
                "all_reduce must sum every rank's real data"
            );
        }
    }

    /// Regression test: before this fix, `nccl_all_gather` always returned
    /// `Tensor::zeros(&output_shape)`, discarding every rank's actual data.
    #[test]
    #[cfg(feature = "nccl")]
    fn test_all_gather_contains_every_ranks_real_data() {
        let comms = NcclCommunicator::init_all(3, &[0, 1, 2]).expect("init_all");
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                std::thread::spawn(move || {
                    let rank = comm.rank();
                    let tensor = Tensor::from_vec(vec![(rank * 10) as f32], &[1]).expect("tensor");
                    let gathered = comm.all_gather(&tensor, 0).expect("all_gather");
                    gathered.data().expect("data")
                })
            })
            .collect();

        for h in handles {
            let data = h.join().expect("thread");
            assert_eq!(
                data,
                vec![0.0f32, 10.0, 20.0],
                "all_gather must not be all-zeros and must preserve per-rank data by rank order"
            );
        }
    }

    #[test]
    fn test_create_nccl_communicator_factory() {
        let result = create_nccl_communicator(0, 1, 0);

        #[cfg(feature = "nccl")]
        assert!(result.is_ok());

        #[cfg(not(feature = "nccl"))]
        assert!(result.is_err());
    }
}
