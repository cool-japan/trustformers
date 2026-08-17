//! Real collective communication algorithms over a [`Transport`].
//!
//! Everything here moves actual bytes and performs actual reduction
//! arithmetic. Nothing is simulated, scaled "to indicate that a collective
//! occurred", or slept through.
//!
//! Algorithms implemented:
//!
//! | Collective       | Algorithm                                    |
//! |------------------|----------------------------------------------|
//! | `all_reduce`     | ring reduce-scatter + ring all-gather        |
//! | `reduce_scatter` | ring reduce-scatter                          |
//! | `all_gather`     | ring all-gather                              |
//! | `broadcast`      | binomial tree                                |
//! | `reduce`         | binomial tree                                |
//! | `barrier`        | gather-to-root + release                     |
//!
//! # Calling contract
//!
//! Collectives are SPMD: every rank must issue the same sequence of calls with
//! matching shapes. Message tags are derived from a per-rank operation counter,
//! so a rank that skips or reorders a collective will fail with a transport
//! timeout rather than silently corrupting data. A per-instance lock serialises
//! calls, so one [`Collective`] must be driven by one logical thread of control.

use super::transport::{InProcessSession, InProcessTransport, Transport};
use anyhow::Result;
use std::sync::Mutex;

/// Reduction operator applied element-wise during a collective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Element-wise sum.
    Sum,
    /// Element-wise sum divided by the world size.
    Mean,
    /// Element-wise maximum.
    Max,
    /// Element-wise minimum.
    Min,
    /// Element-wise product.
    Prod,
}

impl ReduceOp {
    #[inline]
    fn combine(self, accumulator: f32, incoming: f32) -> f32 {
        match self {
            // `Mean` accumulates as a sum and is scaled once at the end.
            ReduceOp::Sum | ReduceOp::Mean => accumulator + incoming,
            ReduceOp::Max => accumulator.max(incoming),
            ReduceOp::Min => accumulator.min(incoming),
            ReduceOp::Prod => accumulator * incoming,
        }
    }
}

/// Errors raised by collective algorithms.
#[derive(Debug, thiserror::Error)]
pub enum CollectiveError {
    /// A peer contributed a buffer of a different length.
    #[error(
        "collective length mismatch on rank {rank}: expected {expected} elements from rank \
         {peer}, received {actual}"
    )]
    LengthMismatch {
        /// Local rank.
        rank: usize,
        /// Sending rank.
        peer: usize,
        /// Expected element count.
        expected: usize,
        /// Received element count.
        actual: usize,
    },

    /// A payload was not a whole number of `f32` values.
    #[error("payload of {bytes} bytes is not a whole number of f32 values")]
    UnalignedPayload {
        /// Payload length in bytes.
        bytes: usize,
    },

    /// The root/destination rank does not exist.
    #[error("root rank {root} is out of range for world size {world_size}")]
    RootOutOfRange {
        /// Requested root.
        root: usize,
        /// Configured world size.
        world_size: usize,
    },

    /// `reduce_scatter` requires the input length to be divisible by the world
    /// size, matching the NCCL/MPI contract.
    #[error("reduce_scatter input length {length} is not divisible by world size {world_size}")]
    IndivisibleInput {
        /// Input length.
        length: usize,
        /// Configured world size.
        world_size: usize,
    },

    /// The world size cannot be represented by the tag encoding.
    #[error("world size {world_size} exceeds the maximum supported by the tag encoding ({max})")]
    WorldSizeTooLarge {
        /// Configured world size.
        world_size: usize,
        /// Largest supported world size.
        max: usize,
    },
}

/// Number of low bits of a tag reserved for the step index within one
/// collective. A collective performs at most `2 * (world_size - 1)` steps.
const STEP_BITS: u32 = 16;
const MAX_STEPS: u64 = 1 << STEP_BITS;

/// Tag bit reserved for caller-chosen point-to-point tags.
///
/// Collective tags are `op_id * 2^16 + step`; reaching this bit would require
/// more than `2^47` collectives on a single rank, so the two spaces are
/// disjoint in practice.
pub const USER_TAG_BIT: u64 = 1 << 63;

#[inline]
fn encode_f32(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[inline]
fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        return Err(CollectiveError::UnalignedPayload { bytes: bytes.len() }.into());
    }
    let mut values = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        let mut word = [0u8; 4];
        word.copy_from_slice(chunk);
        values.push(f32::from_le_bytes(word));
    }
    Ok(values)
}

/// Balanced partition of `len` elements into `parts` contiguous ranges.
///
/// The first `len % parts` ranges receive one extra element, so the partition
/// is identical on every rank and tolerates `len < parts` (producing empty
/// ranges).
pub fn chunk_ranges(len: usize, parts: usize) -> Vec<(usize, usize)> {
    let mut ranges = Vec::with_capacity(parts);
    if parts == 0 {
        return ranges;
    }
    let base = len / parts;
    let remainder = len % parts;
    let mut start = 0usize;
    for index in 0..parts {
        let size = base + usize::from(index < remainder);
        ranges.push((start, start + size));
        start += size;
    }
    ranges
}

/// Collective operations executed over a point-to-point [`Transport`].
#[derive(Debug)]
pub struct Collective<T: Transport> {
    transport: T,
    /// Monotonic operation counter *and* the per-rank serialisation lock.
    op_counter: Mutex<u64>,
}

impl<T: Transport> Collective<T> {
    /// Wrap `transport` in a collective executor.
    pub fn new(transport: T) -> Result<Self> {
        let world_size = transport.world_size();
        let max = (u64::MAX / MAX_STEPS) as usize;
        if world_size > max {
            return Err(CollectiveError::WorldSizeTooLarge { world_size, max }.into());
        }
        Ok(Self {
            transport,
            op_counter: Mutex::new(0),
        })
    }

    /// The underlying transport.
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// Rank of this participant.
    pub fn rank(&self) -> usize {
        self.transport.rank()
    }

    /// Total number of participants.
    pub fn world_size(&self) -> usize {
        self.transport.world_size()
    }

    /// Reserve the next operation id, locking out concurrent collectives on
    /// this rank for the duration of the returned guard.
    fn begin(&self) -> (std::sync::MutexGuard<'_, u64>, u64) {
        let mut guard = self.op_counter.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let op_id = *guard;
        *guard += 1;
        (guard, op_id)
    }

    #[inline]
    fn tag(op_id: u64, step: u64) -> u64 {
        op_id * MAX_STEPS + step
    }

    fn check_root(&self, root: usize) -> Result<()> {
        if root >= self.world_size() {
            return Err(CollectiveError::RootOutOfRange {
                root,
                world_size: self.world_size(),
            }
            .into());
        }
        Ok(())
    }

    fn recv_values(&self, peer: usize, tag: u64, expected: usize) -> Result<Vec<f32>> {
        let values = decode_f32(&self.transport.recv(peer, tag)?)?;
        if values.len() != expected {
            return Err(CollectiveError::LengthMismatch {
                rank: self.rank(),
                peer,
                expected,
                actual: values.len(),
            }
            .into());
        }
        Ok(values)
    }

    /// Synchronise all ranks: no rank returns before every rank has entered.
    pub fn barrier(&self) -> Result<()> {
        let world_size = self.world_size();
        if world_size <= 1 {
            return Ok(());
        }

        let (_guard, op_id) = self.begin();
        let arrive = Self::tag(op_id, 0);
        let release = Self::tag(op_id, 1);
        let rank = self.rank();

        if rank == 0 {
            for peer in 1..world_size {
                self.transport.recv(peer, arrive)?;
            }
            for peer in 1..world_size {
                self.transport.send(peer, release, &[])?;
            }
        } else {
            self.transport.send(0, arrive, &[])?;
            self.transport.recv(0, release)?;
        }

        Ok(())
    }

    /// Broadcast `buffer` from `root` to every rank, using a binomial tree.
    ///
    /// On non-root ranks `buffer` is replaced wholesale by the root's contents
    /// (including its length), so the result is bit-identical to the source.
    pub fn broadcast(&self, buffer: &mut Vec<f32>, root: usize) -> Result<()> {
        self.check_root(root)?;
        let world_size = self.world_size();
        if world_size <= 1 {
            return Ok(());
        }

        let (_guard, op_id) = self.begin();
        let tag = Self::tag(op_id, 0);
        let rank = self.rank();
        let virtual_rank = (rank + world_size - root) % world_size;
        let to_real = |virtual_peer: usize| (virtual_peer + root) % world_size;

        // Receive phase: wait for the parent that owns our highest-order bit.
        let mut mask = 1usize;
        while mask < world_size {
            if virtual_rank & mask != 0 {
                let parent = to_real(virtual_rank & !mask);
                *buffer = decode_f32(&self.transport.recv(parent, tag)?)?;
                break;
            }
            mask <<= 1;
        }

        // Send phase: forward to every child below our lowest set bit.
        let payload = encode_f32(buffer);
        mask >>= 1;
        while mask > 0 {
            let child = virtual_rank | mask;
            if child < world_size {
                self.transport.send(to_real(child), tag, &payload)?;
            }
            mask >>= 1;
        }

        Ok(())
    }

    /// Reduce `buffer` from all ranks into `root`, using a binomial tree.
    ///
    /// On non-root ranks the buffer is restored to its original contents, so a
    /// failed assumption about who owns the result cannot silently poison a
    /// gradient.
    pub fn reduce(&self, buffer: &mut [f32], root: usize, op: ReduceOp) -> Result<()> {
        self.check_root(root)?;
        let world_size = self.world_size();
        if world_size <= 1 {
            if op == ReduceOp::Mean {
                // Dividing by one is the identity; kept explicit for clarity.
            }
            return Ok(());
        }

        let (_guard, op_id) = self.begin();
        let tag = Self::tag(op_id, 0);
        let rank = self.rank();
        let length = buffer.len();
        let virtual_rank = (rank + world_size - root) % world_size;
        let to_real = |virtual_peer: usize| (virtual_peer + root) % world_size;
        let original: Option<Vec<f32>> = if rank == root { None } else { Some(buffer.to_vec()) };

        let mut mask = 1usize;
        while mask < world_size {
            if virtual_rank & mask == 0 {
                let child = virtual_rank | mask;
                if child < world_size {
                    let peer = to_real(child);
                    let incoming = self.recv_values(peer, tag, length)?;
                    for (slot, value) in buffer.iter_mut().zip(incoming) {
                        *slot = op.combine(*slot, value);
                    }
                }
            } else {
                let parent = to_real(virtual_rank & !mask);
                self.transport.send(parent, tag, &encode_f32(buffer))?;
                break;
            }
            mask <<= 1;
        }

        if let Some(original) = original {
            buffer.copy_from_slice(&original);
        } else if op == ReduceOp::Mean {
            let scale = 1.0 / world_size as f32;
            for slot in buffer.iter_mut() {
                *slot *= scale;
            }
        }

        Ok(())
    }

    /// All-reduce `buffer` in place with the bandwidth-optimal ring algorithm
    /// (reduce-scatter followed by all-gather).
    ///
    /// Every rank ends with the identical reduction of every rank's input.
    pub fn all_reduce(&self, buffer: &mut [f32], op: ReduceOp) -> Result<()> {
        let world_size = self.world_size();
        if world_size <= 1 {
            return Ok(());
        }

        let (_guard, op_id) = self.begin();
        let ranges = chunk_ranges(buffer.len(), world_size);

        self.ring_reduce_scatter_in_place(buffer, &ranges, op, op_id, 0)?;
        self.ring_all_gather_in_place(buffer, &ranges, op_id, (world_size - 1) as u64, 1)?;

        if op == ReduceOp::Mean {
            let scale = 1.0 / world_size as f32;
            for slot in buffer.iter_mut() {
                *slot *= scale;
            }
        }

        Ok(())
    }

    /// Reduce `input` across all ranks and return this rank's contiguous slice
    /// of the result.
    ///
    /// `input.len()` must be divisible by the world size, matching the
    /// NCCL/MPI contract.
    pub fn reduce_scatter(&self, input: &[f32], op: ReduceOp) -> Result<Vec<f32>> {
        let world_size = self.world_size();
        if world_size <= 1 {
            return Ok(input.to_vec());
        }
        if !input.len().is_multiple_of(world_size) {
            return Err(CollectiveError::IndivisibleInput {
                length: input.len(),
                world_size,
            }
            .into());
        }

        let (_guard, op_id) = self.begin();
        let mut buffer = input.to_vec();
        let ranges = chunk_ranges(buffer.len(), world_size);
        self.ring_reduce_scatter_in_place(&mut buffer, &ranges, op, op_id, 0)?;

        // After a ring reduce-scatter, rank r owns chunk (r + 1) % world_size.
        let owned = (self.rank() + 1) % world_size;
        let (start, end) = ranges[owned];
        let mut result = buffer[start..end].to_vec();

        if op == ReduceOp::Mean {
            let scale = 1.0 / world_size as f32;
            for slot in result.iter_mut() {
                *slot *= scale;
            }
        }

        Ok(result)
    }

    /// Which chunk index of a `reduce_scatter` result belongs to `rank`.
    pub fn reduce_scatter_chunk_index(&self, rank: usize) -> usize {
        (rank + 1) % self.world_size()
    }

    /// Gather every rank's `local` buffer, returning them indexed by rank.
    ///
    /// All ranks must contribute the same number of elements.
    pub fn all_gather(&self, local: &[f32]) -> Result<Vec<Vec<f32>>> {
        let world_size = self.world_size();
        if world_size <= 1 {
            return Ok(vec![local.to_vec()]);
        }

        let (_guard, op_id) = self.begin();
        let count = local.len();
        let mut buffer = vec![0.0f32; count * world_size];
        let rank = self.rank();
        buffer[rank * count..(rank + 1) * count].copy_from_slice(local);

        let ranges = chunk_ranges(buffer.len(), world_size);
        self.ring_all_gather_in_place(&mut buffer, &ranges, op_id, 0, 0)?;

        Ok(ranges.iter().map(|&(start, end)| buffer[start..end].to_vec()).collect())
    }

    /// Ring reduce-scatter: after `world_size - 1` steps, rank `r` holds the
    /// fully reduced chunk `(r + 1) % world_size`.
    fn ring_reduce_scatter_in_place(
        &self,
        buffer: &mut [f32],
        ranges: &[(usize, usize)],
        op: ReduceOp,
        op_id: u64,
        step_base: u64,
    ) -> Result<()> {
        let world_size = self.world_size();
        let rank = self.rank();
        let next = (rank + 1) % world_size;
        let previous = (rank + world_size - 1) % world_size;

        for step in 0..world_size - 1 {
            let send_chunk = (rank + world_size - step) % world_size;
            let recv_chunk = (rank + world_size - step - 1) % world_size;
            let tag = Self::tag(op_id, step_base + step as u64);

            let (send_start, send_end) = ranges[send_chunk];
            self.transport.send(next, tag, &encode_f32(&buffer[send_start..send_end]))?;

            let (recv_start, recv_end) = ranges[recv_chunk];
            let incoming = self.recv_values(previous, tag, recv_end - recv_start)?;
            for (slot, value) in buffer[recv_start..recv_end].iter_mut().zip(incoming) {
                *slot = op.combine(*slot, value);
            }
        }

        Ok(())
    }

    /// Ring all-gather: propagates each rank's owned chunk around the ring
    /// until every rank holds every chunk.
    ///
    /// `owned_offset` states which chunk each rank starts out owning: rank `r`
    /// owns chunk `(r + owned_offset) % world_size`. It is `1` after a ring
    /// reduce-scatter (which leaves rank `r` holding chunk `r + 1`) and `0` for
    /// a standalone all-gather (where rank `r` contributes chunk `r`).
    fn ring_all_gather_in_place(
        &self,
        buffer: &mut [f32],
        ranges: &[(usize, usize)],
        op_id: u64,
        step_base: u64,
        owned_offset: usize,
    ) -> Result<()> {
        let world_size = self.world_size();
        let rank = self.rank();
        let next = (rank + 1) % world_size;
        let previous = (rank + world_size - 1) % world_size;

        for step in 0..world_size - 1 {
            let send_chunk = (rank + owned_offset + world_size - step) % world_size;
            let recv_chunk = (rank + owned_offset + world_size - step - 1) % world_size;
            let tag = Self::tag(op_id, step_base + step as u64);

            let (send_start, send_end) = ranges[send_chunk];
            self.transport.send(next, tag, &encode_f32(&buffer[send_start..send_end]))?;

            let (recv_start, recv_end) = ranges[recv_chunk];
            let incoming = self.recv_values(previous, tag, recv_end - recv_start)?;
            buffer[recv_start..recv_end].copy_from_slice(&incoming);
        }

        Ok(())
    }

    /// Symmetric full-buffer exchange with `peer`.
    ///
    /// Both partners must call this at the same point in their collective
    /// sequence; the shared tag makes the exchange order-independent, so no
    /// send/recv interleaving convention is required. This is the primitive
    /// behind recursive-doubling (butterfly) all-reduce.
    pub fn exchange(&self, peer: usize, values: &[f32]) -> Result<Vec<f32>> {
        if peer >= self.world_size() {
            return Err(CollectiveError::RootOutOfRange {
                root: peer,
                world_size: self.world_size(),
            }
            .into());
        }

        let (_guard, op_id) = self.begin();
        let tag = Self::tag(op_id, 0);
        self.transport.send(peer, tag, &encode_f32(values))?;
        self.recv_values(peer, tag, values.len())
    }

    /// Send `values` to `peer` as a standalone point-to-point message.
    ///
    /// Uses the same tag counter as the collectives, so a matching
    /// [`Collective::recv_from`] must be issued at the same point in every
    /// rank's call sequence.
    pub fn send_to(&self, peer: usize, values: &[f32]) -> Result<()> {
        let (_guard, op_id) = self.begin();
        self.transport.send(peer, Self::tag(op_id, 0), &encode_f32(values))
    }

    /// Receive a point-to-point message from `peer`.
    pub fn recv_from(&self, peer: usize) -> Result<Vec<f32>> {
        let (_guard, op_id) = self.begin();
        decode_f32(&self.transport.recv(peer, Self::tag(op_id, 0))?)
    }

    /// Send `values` to `peer` under a caller-chosen `tag`.
    ///
    /// Unlike [`Collective::send_to`], the tag does not come from the operation
    /// counter, so sender and receiver need only agree on the tag — not on
    /// their positions in a shared call sequence. That is what makes
    /// asymmetric protocols (pipeline stages, hierarchical reductions, where
    /// ranks execute different numbers of operations) expressible.
    ///
    /// User tags live in a reserved half of the tag space
    /// ([`USER_TAG_BIT`]) and can never collide with collective tags.
    pub fn send_tagged(&self, peer: usize, tag: u64, values: &[f32]) -> Result<()> {
        self.transport.send(
            peer,
            USER_TAG_BIT | (tag & !USER_TAG_BIT),
            &encode_f32(values),
        )
    }

    /// Receive a message from `peer` under a caller-chosen `tag`.
    pub fn recv_tagged(&self, peer: usize, tag: u64) -> Result<Vec<f32>> {
        decode_f32(&self.transport.recv(peer, USER_TAG_BIT | (tag & !USER_TAG_BIT))?)
    }
}

/// Build one [`Collective`] per rank, all sharing a fresh in-process session.
///
/// This is the ergonomic entry point for multi-threaded (single process)
/// multi-rank execution and for tests.
pub fn in_process_collectives(world_size: usize) -> Result<Vec<Collective<InProcessTransport>>> {
    let session = InProcessSession::new(world_size)?;
    (0..world_size).map(|rank| Collective::new(session.transport(rank)?)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Run `body` on every rank concurrently and collect the results in rank
    /// order.
    fn spmd<R, F>(world_size: usize, body: F) -> Vec<R>
    where
        R: Send + 'static,
        F: Fn(usize, &Collective<InProcessTransport>) -> R + Send + Sync + 'static,
    {
        let collectives =
            in_process_collectives(world_size).expect("in-process collectives must build in test");
        let body = Arc::new(body);

        let handles: Vec<_> = collectives
            .into_iter()
            .enumerate()
            .map(|(rank, collective)| {
                let body = Arc::clone(&body);
                std::thread::spawn(move || body(rank, &collective))
            })
            .collect();

        handles
            .into_iter()
            .map(|handle| handle.join().expect("rank thread must not panic in test"))
            .collect()
    }

    #[test]
    fn all_reduce_sum_matches_naive_reference() {
        let world_size = 4;
        let length = 11; // deliberately not divisible by the world size
        let results = spmd(world_size, move |rank, collective| {
            let mut buffer: Vec<f32> =
                (0..length).map(|i| (rank * 100 + i) as f32 * 0.25).collect();
            collective.all_reduce(&mut buffer, ReduceOp::Sum).expect("all_reduce in test");
            buffer
        });

        let expected: Vec<f32> = (0..length)
            .map(|i| (0..world_size).map(|rank| (rank * 100 + i) as f32 * 0.25).sum::<f32>())
            .collect();

        for (rank, actual) in results.iter().enumerate() {
            assert_eq!(actual.len(), length, "rank {rank} length");
            for (index, (got, want)) in actual.iter().zip(&expected).enumerate() {
                approx::assert_relative_eq!(got, want, epsilon = 1e-4);
                let _ = index;
            }
        }
    }

    #[test]
    fn all_reduce_mean_divides_by_world_size() {
        let world_size = 3;
        let results = spmd(world_size, |rank, collective| {
            let mut buffer = vec![(rank + 1) as f32; 6];
            collective.all_reduce(&mut buffer, ReduceOp::Mean).expect("all_reduce in test");
            buffer
        });

        // (1 + 2 + 3) / 3 == 2
        for buffer in &results {
            for value in buffer {
                approx::assert_relative_eq!(*value, 2.0f32, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn all_reduce_max_and_min_are_real_reductions() {
        let world_size = 4;
        let max_results = spmd(world_size, |rank, collective| {
            let mut buffer = vec![rank as f32, -(rank as f32)];
            collective.all_reduce(&mut buffer, ReduceOp::Max).expect("all_reduce in test");
            buffer
        });
        for buffer in &max_results {
            assert_eq!(buffer, &vec![3.0, 0.0]);
        }

        let min_results = spmd(world_size, |rank, collective| {
            let mut buffer = vec![rank as f32, -(rank as f32)];
            collective.all_reduce(&mut buffer, ReduceOp::Min).expect("all_reduce in test");
            buffer
        });
        for buffer in &min_results {
            assert_eq!(buffer, &vec![0.0, -3.0]);
        }
    }

    #[test]
    fn all_reduce_is_not_the_identity() {
        // The old fake implementation multiplied by 1.0 and returned. This
        // asserts the output genuinely depends on the *other* ranks' data.
        let world_size = 2;
        let results = spmd(world_size, |rank, collective| {
            let mut buffer = vec![if rank == 0 { 1.0f32 } else { 10.0f32 }; 4];
            collective.all_reduce(&mut buffer, ReduceOp::Sum).expect("all_reduce in test");
            buffer
        });
        for buffer in &results {
            assert_eq!(buffer, &vec![11.0f32; 4]);
        }
    }

    #[test]
    fn broadcast_is_bit_exact_and_never_scales() {
        let world_size = 4;
        let root = 2;
        let source: Vec<f32> = (0..7).map(|i| 0.1234_f32 * (i as f32 + 1.0)).collect();
        let source_for_ranks = source.clone();

        let results = spmd(world_size, move |rank, collective| {
            let mut buffer = if rank == root {
                source_for_ranks.clone()
            } else {
                // Deliberately different length and contents.
                vec![-999.0f32; 3]
            };
            collective.broadcast(&mut buffer, root).expect("broadcast in test");
            buffer
        });

        for (rank, buffer) in results.iter().enumerate() {
            assert_eq!(buffer, &source, "rank {rank} must receive the exact source");
        }
    }

    #[test]
    fn all_gather_preserves_rank_order() {
        let world_size = 4;
        let results = spmd(world_size, |rank, collective| {
            let local = vec![rank as f32, rank as f32 + 0.5];
            collective.all_gather(&local).expect("all_gather in test")
        });

        for (rank, gathered) in results.iter().enumerate() {
            assert_eq!(gathered.len(), world_size, "rank {rank}");
            for (peer, chunk) in gathered.iter().enumerate() {
                assert_eq!(
                    chunk,
                    &vec![peer as f32, peer as f32 + 0.5],
                    "rank {rank} slot {peer}"
                );
            }
        }
    }

    #[test]
    fn reduce_scatter_returns_the_owning_chunk() {
        let world_size = 3;
        let length = 9;
        let results = spmd(world_size, move |rank, collective| {
            let input: Vec<f32> = (0..length).map(|i| (rank + 1) as f32 * (i as f32)).collect();
            let chunk = collective
                .reduce_scatter(&input, ReduceOp::Sum)
                .expect("reduce_scatter in test");
            (collective.reduce_scatter_chunk_index(rank), chunk)
        });

        // Sum over ranks of (rank+1)*i == 6*i for world_size 3.
        let full: Vec<f32> = (0..length).map(|i| 6.0 * i as f32).collect();
        let ranges = chunk_ranges(length, world_size);

        for (chunk_index, chunk) in results {
            let (start, end) = ranges[chunk_index];
            assert_eq!(chunk, full[start..end].to_vec());
        }
    }

    #[test]
    fn reduce_scatter_rejects_indivisible_input() {
        let results = spmd(2, |_rank, collective| {
            collective.reduce_scatter(&[1.0, 2.0, 3.0], ReduceOp::Sum).is_err()
        });
        assert!(results.iter().all(|failed| *failed));
    }

    #[test]
    fn reduce_leaves_non_root_buffers_untouched() {
        let world_size = 4;
        let root = 1;
        let results = spmd(world_size, move |rank, collective| {
            let mut buffer = vec![(rank + 1) as f32, (rank + 1) as f32 * 2.0];
            collective.reduce(&mut buffer, root, ReduceOp::Sum).expect("reduce in test");
            buffer
        });

        for (rank, buffer) in results.iter().enumerate() {
            if rank == root {
                assert_eq!(buffer, &vec![10.0, 20.0]); // 1+2+3+4 and its double
            } else {
                assert_eq!(buffer, &vec![(rank + 1) as f32, (rank + 1) as f32 * 2.0]);
            }
        }
    }

    #[test]
    fn barrier_synchronises_all_ranks() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let world_size = 4;
        let arrived = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&arrived);

        let observed = spmd(world_size, move |_rank, collective| {
            counter.fetch_add(1, Ordering::SeqCst);
            collective.barrier().expect("barrier in test");
            counter.load(Ordering::SeqCst)
        });

        for count in observed {
            assert_eq!(count, world_size, "every rank must see all arrivals");
        }
    }

    #[test]
    fn repeated_collectives_stay_in_lockstep() {
        let world_size = 3;
        let results = spmd(world_size, move |rank, collective| {
            let mut buffer = vec![rank as f32 + 1.0; 5];
            for _ in 0..8 {
                collective.all_reduce(&mut buffer, ReduceOp::Sum).expect("all_reduce in test");
                collective.barrier().expect("barrier in test");
                for slot in buffer.iter_mut() {
                    *slot /= world_size as f32;
                }
            }
            buffer
        });

        // Mean of {1,2,3} is 2, and repeated averaging is a fixed point.
        for buffer in &results {
            for value in buffer {
                approx::assert_relative_eq!(*value, 2.0f32, epsilon = 1e-5);
            }
        }
    }

    #[test]
    fn single_rank_collectives_are_the_identity() {
        let collectives =
            in_process_collectives(1).expect("in-process collectives must build in test");
        let collective = &collectives[0];

        let mut buffer = vec![1.5f32, -2.5, 3.0];
        collective.all_reduce(&mut buffer, ReduceOp::Sum).expect("all_reduce in test");
        assert_eq!(buffer, vec![1.5f32, -2.5, 3.0]);

        collective.broadcast(&mut buffer, 0).expect("broadcast in test");
        assert_eq!(buffer, vec![1.5f32, -2.5, 3.0]);

        collective.barrier().expect("barrier in test");
        assert_eq!(
            collective.all_gather(&buffer).expect("all_gather in test").len(),
            1
        );
    }

    #[test]
    fn chunk_ranges_partition_exactly() {
        let ranges = chunk_ranges(11, 4);
        assert_eq!(ranges, vec![(0, 3), (3, 6), (6, 9), (9, 11)]);

        let short = chunk_ranges(2, 4);
        assert_eq!(short, vec![(0, 1), (1, 2), (2, 2), (2, 2)]);
    }
}
