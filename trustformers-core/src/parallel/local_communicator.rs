// Copyright (c) 2025-2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

//! In-process, rank-indexed shared-buffer communicator.
//!
//! Real multi-node collectives need a network transport - see the
//! `trustformers-training` crate's TCP backend for that. Within a single OS
//! process, though, multiple "ranks" can share state directly: every
//! participating thread holds an `InProcessGroup` (via `Arc`) plus its own
//! rank index, and collectives rendezvous through a pair of reusable
//! barriers so that data really moves between ranks instead of being echoed
//! back to its own sender or replaced with zeros.
//!
//! This backs [`LocalCommunicator`] (the always-available `Communicator`
//! used by [`super::model_parallel::ModelParallelContext`] outside the
//! `nccl` feature) and, when the `nccl` feature is enabled,
//! `nccl_communicator::NcclCommunicator`.
//!
//! # Usage requirement
//!
//! Exactly like real NCCL/MPI, every rank in a group must call collectives
//! in the same order. A rank that skips a collective call (or calls a
//! different one) will hang the whole group on the barrier - this is not a
//! shortcut unique to this implementation, it is how every real collective
//! library behaves under the same misuse.

use crate::errors::{timeout_error, Result, TrustformersError};
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::{Arc, Barrier, Condvar, Mutex};
use std::time::Duration;

/// How long a point-to-point `recv` waits for its matching `send` before
/// giving up. Generous, but bounded, so a mismatched rank in a test times
/// out with a clear error instead of hanging forever.
const MAILBOX_TIMEOUT: Duration = Duration::from_secs(30);

/// Rank-indexed shared state for a group of in-process communicators.
///
/// `collective` implements the reduce-style operations (all-reduce,
/// all-gather, broadcast, reduce-scatter's underlying sum) generically:
/// every rank deposits its contribution, one rank (`Barrier`'s designated
/// leader) combines every rank's contribution once all have arrived, and
/// every rank reads the same combined result. `send`/`recv` implement
/// point-to-point exchange through per-`(src, dst)` mailboxes so ring-style
/// patterns (all-to-all) can pair up without a full-group barrier per hop.
pub(crate) struct InProcessGroup {
    world_size: usize,
    slots: Vec<Mutex<Option<Tensor>>>,
    result: Mutex<Option<Tensor>>,
    error: Mutex<Option<String>>,
    entry_barrier: Barrier,
    exit_barrier: Barrier,
    mailboxes: Mutex<HashMap<(usize, usize), Tensor>>,
    mailbox_cv: Condvar,
}

impl InProcessGroup {
    pub(crate) fn new(world_size: usize) -> Arc<Self> {
        let barrier_size = world_size.max(1);
        Arc::new(Self {
            world_size,
            slots: (0..world_size).map(|_| Mutex::new(None)).collect(),
            result: Mutex::new(None),
            error: Mutex::new(None),
            entry_barrier: Barrier::new(barrier_size),
            exit_barrier: Barrier::new(barrier_size),
            mailboxes: Mutex::new(HashMap::new()),
            mailbox_cv: Condvar::new(),
        })
    }

    pub(crate) fn world_size(&self) -> usize {
        self.world_size
    }

    /// Run one collective operation for `rank`.
    ///
    /// Deposits `input` into this rank's slot, rendezvous with every other
    /// rank via `entry_barrier`, lets exactly one rank compute `reduce`
    /// over every rank's deposited value once all have arrived, publishes
    /// it, then rendezvous again via `exit_barrier` before any rank reads
    /// the published result. Because every call site's ranks call this in
    /// the same order (see module docs), a later call's writes cannot
    /// overlap an earlier call's reads: the next `entry_barrier.wait()`
    /// cannot release any rank until every rank has already returned from
    /// (and therefore finished reading the result of) the previous call.
    pub(crate) fn collective<F>(&self, rank: usize, input: Tensor, reduce: F) -> Result<Tensor>
    where
        F: FnOnce(&[Tensor]) -> Result<Tensor>,
    {
        if self.world_size <= 1 {
            // Single-member group: a collective over one participant is
            // its own input, computed the same way a real one would be
            // (via `reduce`), not shortcut to a raw clone.
            return reduce(std::slice::from_ref(&input));
        }

        *self.slots[rank].lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(input);

        let wait = self.entry_barrier.wait();
        if wait.is_leader() {
            let inputs: Result<Vec<Tensor>> = self
                .slots
                .iter()
                .enumerate()
                .map(|(r, slot)| {
                    slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone().ok_or_else(
                        || {
                            TrustformersError::runtime_error(format!(
                                "in-process collective: rank {r} did not deposit data before \
                                 the entry barrier released (every rank in the group must call \
                                 collectives in the same order)"
                            ))
                        },
                    )
                })
                .collect();

            let mut result_slot =
                self.result.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut error_slot = self.error.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            *result_slot = None;
            *error_slot = None;
            match inputs.and_then(|values| reduce(&values)) {
                Ok(computed) => *result_slot = Some(computed),
                Err(e) => *error_slot = Some(e.to_string()),
            }
        }

        self.exit_barrier.wait();

        if let Some(msg) =
            self.error.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).clone()
        {
            return Err(TrustformersError::runtime_error(msg));
        }
        self.result
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
            .ok_or_else(|| {
                TrustformersError::runtime_error(
                    "in-process collective: no result was published".to_string(),
                )
            })
    }

    /// Post `tensor` for rank `to` to pick up via a matching `recv(from,
    /// to)`. Does not block: overwrites (rather than queues) any prior
    /// undelivered message for the same `(from, to)` pair, so callers must
    /// not post a second message to the same peer before the first is
    /// received.
    pub(crate) fn post(&self, from: usize, to: usize, tensor: Tensor) {
        let mut mailboxes = self.mailboxes.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        mailboxes.insert((from, to), tensor);
        self.mailbox_cv.notify_all();
    }

    /// Block until rank `to` has a message from rank `from`, then return
    /// it. Bounded by [`MAILBOX_TIMEOUT`] so a mismatched pairing fails
    /// loudly instead of hanging.
    pub(crate) fn take(&self, from: usize, to: usize) -> Result<Tensor> {
        let mailboxes = self.mailboxes.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let (mut mailboxes, wait_result) = self
            .mailbox_cv
            .wait_timeout_while(mailboxes, MAILBOX_TIMEOUT, |m| !m.contains_key(&(from, to)))
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if wait_result.timed_out() {
            return Err(timeout_error(
                format!("in-process recv: rank {to} waiting on a message from rank {from}"),
                MAILBOX_TIMEOUT.as_millis() as u64,
            ));
        }
        mailboxes.remove(&(from, to)).ok_or_else(|| {
            TrustformersError::runtime_error(
                "in-process recv: mailbox entry vanished after wait completed".to_string(),
            )
        })
    }
}

/// Stack same-shape tensors along a new leading axis, producing shape
/// `[tensors.len(), ...tensors[0].shape()]`.
fn stack_tensors(tensors: &[Tensor]) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(TrustformersError::invalid_input(
            "stack_tensors: input list is empty".to_string(),
        ));
    }
    let mut expanded = Vec::with_capacity(tensors.len());
    for t in tensors {
        let mut shape = t.shape();
        shape.insert(0, 1);
        expanded.push(t.reshape(&shape)?);
    }
    if expanded.len() == 1 {
        let only = expanded.into_iter().next().ok_or_else(|| {
            TrustformersError::runtime_error(
                "stack_tensors: unreachable empty vec after length check".to_string(),
            )
        })?;
        return Ok(only);
    }
    Tensor::concat(&expanded, 0)
}

/// Elementwise sum of same-shape tensors (the shared core of all-reduce and
/// reduce-scatter).
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

/// A [`super::Communicator`] backed by `InProcessGroup`: every rank in the
/// group lives in this process (possibly on a different thread), and
/// collectives move real per-rank data through shared, rank-indexed buffers
/// instead of faking success. See the module docs for what this does and
/// does not provide.
#[derive(Clone)]
pub struct LocalCommunicator {
    rank: usize,
    group: Arc<InProcessGroup>,
}

impl LocalCommunicator {
    /// Build a single-member group - the common single-process default.
    /// Collectives on the result are correct identities (there is exactly
    /// one participant), not a fake shortcut: an all-reduce over one rank
    /// is its own input, an all-gather of one rank is a length-1 gather.
    pub fn single() -> Self {
        Self {
            rank: 0,
            group: InProcessGroup::new(1),
        }
    }

    /// Build `world_size` communicators sharing one real group, with ranks
    /// `0..world_size`. Intended for single-process multi-rank exercises -
    /// e.g. spawn one OS thread per returned communicator, each driving the
    /// same sequence of collective/point-to-point calls. Real cross-node
    /// execution needs `trustformers-training`'s TCP backend.
    pub fn group(world_size: usize) -> Vec<Self> {
        if world_size == 0 {
            return Vec::new();
        }
        let group = InProcessGroup::new(world_size);
        (0..world_size)
            .map(|rank| Self {
                rank,
                group: Arc::clone(&group),
            })
            .collect()
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world_size(&self) -> usize {
        self.group.world_size()
    }
}

impl super::Communicator for LocalCommunicator {
    fn all_gather(&self, tensor: &Tensor, _split_dim: usize) -> Result<Tensor> {
        self.group.collective(self.rank, tensor.clone(), stack_tensors)
    }

    fn reduce_scatter(&self, tensor: &Tensor, split_dim: usize) -> Result<Tensor> {
        let summed = self.group.collective(self.rank, tensor.clone(), elementwise_sum)?;
        let shape = summed.shape();
        if split_dim >= shape.len() {
            return Err(TrustformersError::invalid_input(format!(
                "reduce_scatter: split_dim {} out of bounds for shape {:?}",
                split_dim, shape
            )));
        }
        let world_size = self.group.world_size();
        let dim_size = shape[split_dim];
        if world_size == 0 || dim_size % world_size != 0 {
            return Err(TrustformersError::invalid_input(format!(
                "reduce_scatter: dimension {} size {} is not evenly divisible by world_size {}",
                split_dim, dim_size, world_size
            )));
        }
        let chunk = dim_size / world_size;
        let start = self.rank * chunk;
        summed.slice(split_dim, start, start + chunk)
    }

    fn all_reduce(&self, tensor: &mut Tensor) -> Result<()> {
        let summed = self.group.collective(self.rank, tensor.clone(), elementwise_sum)?;
        *tensor = summed;
        Ok(())
    }

    fn send(&self, tensor: &Tensor, dest: usize) -> Result<()> {
        if dest >= self.group.world_size() {
            return Err(TrustformersError::invalid_input(format!(
                "send: destination rank {} is out of bounds for world_size {}",
                dest,
                self.group.world_size()
            )));
        }
        self.group.post(self.rank, dest, tensor.clone());
        Ok(())
    }

    fn recv(&self, shape: &[usize], src: usize) -> Result<Tensor> {
        if src >= self.group.world_size() {
            return Err(TrustformersError::invalid_input(format!(
                "recv: source rank {} is out of bounds for world_size {}",
                src,
                self.group.world_size()
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

    fn broadcast(&self, tensor: &mut Tensor, root: usize) -> Result<()> {
        if root >= self.group.world_size() {
            return Err(TrustformersError::invalid_input(format!(
                "broadcast: root rank {} is out of bounds for world_size {}",
                root,
                self.group.world_size()
            )));
        }
        let broadcasted = self.group.collective(self.rank, tensor.clone(), move |values| {
            values.get(root).cloned().ok_or_else(|| {
                TrustformersError::runtime_error(
                    "broadcast: root rank missing from deposited values".to_string(),
                )
            })
        })?;
        *tensor = broadcasted;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::Communicator;
    use super::*;
    use std::thread;

    #[test]
    fn test_single_all_reduce_is_identity() {
        let comm = LocalCommunicator::single();
        let mut tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor");
        comm.all_reduce(&mut tensor).expect("all_reduce");
        assert_eq!(tensor.data().expect("data"), vec![1.0, 2.0, 3.0]);
    }

    /// Regression test: before this fix, `NcclCommunicator::all_reduce`
    /// (which this module's `collective` now also backs) left the tensor
    /// untouched - a no-op disguised as success. Three ranks each
    /// contribute a distinct tensor; a correct sum-all-reduce must combine
    /// all three, not echo any single rank's input.
    #[test]
    fn test_all_reduce_sums_real_data_from_every_rank() {
        let comms = LocalCommunicator::group(3);
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                thread::spawn(move || {
                    let rank = comm.rank();
                    let mut tensor =
                        Tensor::from_vec(vec![(rank + 1) as f32, (rank + 1) as f32 * 10.0], &[2])
                            .expect("tensor");
                    comm.all_reduce(&mut tensor).expect("all_reduce");
                    tensor.data().expect("data")
                })
            })
            .collect();

        let results: Vec<Vec<f32>> =
            handles.into_iter().map(|h| h.join().expect("thread")).collect();
        // Ranks contribute [1,10], [2,20], [3,30] -> sum = [6, 60].
        for data in &results {
            assert_eq!(data, &vec![6.0f32, 60.0]);
        }
    }

    /// Regression test: before this fix, `NcclCommunicator::all_gather`
    /// always returned `Tensor::zeros`. Each rank's real contribution must
    /// appear in the gathered output, indexed by rank.
    #[test]
    fn test_all_gather_collects_every_ranks_real_data() {
        let comms = LocalCommunicator::group(4);
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                thread::spawn(move || {
                    let rank = comm.rank();
                    let tensor = Tensor::from_vec(vec![rank as f32], &[1]).expect("tensor");
                    let gathered = comm.all_gather(&tensor, 0).expect("all_gather");
                    gathered.data().expect("data")
                })
            })
            .collect();

        let results: Vec<Vec<f32>> =
            handles.into_iter().map(|h| h.join().expect("thread")).collect();
        for data in &results {
            assert_eq!(
                data,
                &vec![0.0f32, 1.0, 2.0, 3.0],
                "must contain every rank's own value in order"
            );
        }
    }

    /// Regression test: before this fix, point-to-point exchange
    /// (`tensor_parallel::simulate_point_to_point_exchange`) returned the
    /// sender's own data untouched. This exercises the transport
    /// underneath that fix directly: rank 0 sends data that only rank 1
    /// could have produced (a value rank 0 never had), and rank 1 must
    /// receive exactly that.
    #[test]
    fn test_send_recv_moves_real_data_between_ranks() {
        let mut comms = LocalCommunicator::group(2);
        let comm1 = comms.pop().expect("rank 1");
        let comm0 = comms.pop().expect("rank 0");

        let sender = thread::spawn(move || {
            let payload = Tensor::from_vec(vec![42.0, 43.0], &[2]).expect("tensor");
            comm0.send(&payload, 1).expect("send");
        });
        let receiver = thread::spawn(move || comm1.recv(&[2], 0).expect("recv"));

        sender.join().expect("sender thread");
        let received = receiver.join().expect("receiver thread");
        assert_eq!(received.data().expect("data"), vec![42.0, 43.0]);
    }

    #[test]
    fn test_recv_wrong_shape_errors() {
        let mut comms = LocalCommunicator::group(2);
        let comm1 = comms.pop().expect("rank 1");
        let comm0 = comms.pop().expect("rank 0");

        let sender = thread::spawn(move || {
            let payload = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor");
            comm0.send(&payload, 1).expect("send");
        });
        let receiver = thread::spawn(move || comm1.recv(&[2], 0));

        sender.join().expect("sender thread");
        let result = receiver.join().expect("receiver thread");
        assert!(result.is_err(), "receiving a mismatched shape must error");
    }

    #[test]
    fn test_broadcast_from_non_zero_root() {
        let comms = LocalCommunicator::group(3);
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                thread::spawn(move || {
                    let rank = comm.rank();
                    let mut tensor = Tensor::from_vec(vec![rank as f32], &[1]).expect("tensor");
                    comm.broadcast(&mut tensor, 2).expect("broadcast");
                    tensor.data().expect("data")
                })
            })
            .collect();

        for h in handles {
            let data = h.join().expect("thread");
            assert_eq!(
                data,
                vec![2.0f32],
                "every rank must observe root rank 2's value"
            );
        }
    }

    #[test]
    fn test_reduce_scatter_gives_each_rank_its_own_summed_chunk() {
        let comms = LocalCommunicator::group(2);
        let handles: Vec<_> = comms
            .into_iter()
            .map(|comm| {
                thread::spawn(move || {
                    let rank = comm.rank();
                    // rank 0: [1,2,3,4], rank 1: [10,20,30,40]
                    let base = if rank == 0 {
                        vec![1.0, 2.0, 3.0, 4.0]
                    } else {
                        vec![10.0, 20.0, 30.0, 40.0]
                    };
                    let tensor = Tensor::from_vec(base, &[4]).expect("tensor");
                    let scattered = comm.reduce_scatter(&tensor, 0).expect("reduce_scatter");
                    (rank, scattered.data().expect("data"))
                })
            })
            .collect();

        let mut results: Vec<(usize, Vec<f32>)> =
            handles.into_iter().map(|h| h.join().expect("thread")).collect();
        results.sort_by_key(|(rank, _)| *rank);
        // sum = [11, 22, 33, 44]; rank 0 gets the first half, rank 1 the second.
        assert_eq!(results[0].1, vec![11.0, 22.0]);
        assert_eq!(results[1].1, vec![33.0, 44.0]);
    }
}
