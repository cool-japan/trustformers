//! [`ProcessGroup`] implementations backed by real collective communication.
//!
//! Both groups delegate to [`trustformers_optim::collective::Collective`],
//! which implements ring all-reduce / all-gather / reduce-scatter and binomial
//! broadcast / reduce over a pluggable point-to-point transport. Nothing here
//! simulates, scales, or sleeps.
//!
//! * [`InProcessProcessGroup`] — ranks are threads of one process, rendezvousing
//!   through shared memory. This is what the test-suite exercises.
//! * [`TcpProcessGroup`] — ranks are separate processes (optionally on separate
//!   hosts) connected by a TCP mesh.
//!
//! # Calling contract
//!
//! Collectives are SPMD. Every rank must issue the same sequence of calls, with
//! the same tensor shapes, in the same order. A rank that diverges fails with a
//! transport timeout instead of silently corrupting the reduction.

use crate::distributed::{DistributedError, ProcessGroup};
use anyhow::Result;
use std::net::SocketAddr;
use std::sync::Arc;
use trustformers_core::tensor::Tensor;
use trustformers_optim::collective::{Collective, ReduceOp};
use trustformers_optim::transport::{
    join_in_process_session, InProcessSession, InProcessTransport, TcpTransport, Transport,
};

/// A [`ProcessGroup`] over any [`Transport`].
///
/// This is the shared implementation of the tensor plumbing: extract `f32`
/// values, run the collective, rebuild the tensor with its original shape.
#[derive(Debug)]
pub struct CollectiveProcessGroup {
    collective: Collective<Arc<dyn Transport>>,
}

impl CollectiveProcessGroup {
    /// Wrap `transport` as a process group.
    pub fn new(transport: Arc<dyn Transport>) -> Result<Self> {
        Ok(Self {
            collective: Collective::new(transport)?,
        })
    }

    /// The underlying collective executor, for operations outside the
    /// [`ProcessGroup`] trait (all-gather, reduce-scatter, point-to-point).
    pub fn collective(&self) -> &Collective<Arc<dyn Transport>> {
        &self.collective
    }

    /// All-gather `tensor`, returning one tensor per rank in rank order.
    ///
    /// Every rank must contribute the same shape.
    pub fn all_gather(&self, tensor: &Tensor) -> Result<Vec<Tensor>> {
        let shape = tensor.shape();
        let gathered = self.collective.all_gather(&tensor.to_vec_f32()?)?;
        gathered
            .iter()
            .map(|values| Tensor::from_slice(values, &shape).map_err(anyhow::Error::from))
            .collect()
    }

    /// Reduce `tensor` across all ranks and return this rank's slice of the
    /// result. The element count must be divisible by the world size.
    pub fn reduce_scatter(&self, tensor: &Tensor, op: ReduceOp) -> Result<Tensor> {
        let chunk = self.collective.reduce_scatter(&tensor.to_vec_f32()?, op)?;
        let length = chunk.len();
        Ok(Tensor::from_slice(&chunk, &[length])?)
    }

    /// All-reduce with an explicit reduction operator (the [`ProcessGroup`]
    /// trait method always sums).
    pub fn all_reduce_with(&self, tensors: &mut [Tensor], op: ReduceOp) -> Result<()> {
        for tensor in tensors.iter_mut() {
            let shape = tensor.shape();
            let mut values = tensor.to_vec_f32()?;
            self.collective.all_reduce(&mut values, op)?;
            *tensor = Tensor::from_slice(&values, &shape)?;
        }
        Ok(())
    }

    /// Send a tensor to `peer`. The peer must issue a matching
    /// [`CollectiveProcessGroup::recv_ordered`] at the same point in its call
    /// sequence.
    pub fn send_ordered(&self, peer: usize, tensor: &Tensor) -> Result<()> {
        self.collective.send_to(peer, &tensor.to_vec_f32()?)
    }

    /// Receive a tensor from `peer` and give it `shape`.
    pub fn recv_ordered(&self, peer: usize, shape: &[usize]) -> Result<Tensor> {
        let values = self.collective.recv_from(peer)?;
        self.reshape_received(peer, values, shape)
    }

    fn reshape_received(&self, peer: usize, values: Vec<f32>, shape: &[usize]) -> Result<Tensor> {
        let expected: usize = shape.iter().product();
        if values.len() != expected {
            return Err(DistributedError::InvalidConfig(format!(
                "rank {} received {} values from rank {peer} but shape {shape:?} needs {expected}",
                self.collective.rank(),
                values.len()
            ))
            .into());
        }
        Ok(Tensor::from_slice(&values, shape)?)
    }
}

impl ProcessGroup for CollectiveProcessGroup {
    fn all_reduce(&self, tensors: &mut [Tensor]) -> Result<()> {
        self.all_reduce_with(tensors, ReduceOp::Sum)
    }

    fn broadcast(&self, tensor: &mut Tensor, src_rank: usize) -> Result<()> {
        // The root's shape is authoritative: non-root ranks adopt it, so the
        // result is bit-identical to the source. `Collective::broadcast`
        // replaces the whole buffer, so the dimension count travels with it.
        const MAX_EXACT_DIM: usize = 1 << 24; // f32 integers are exact below this
        let local_shape = tensor.shape();
        if local_shape.iter().any(|dim| *dim > MAX_EXACT_DIM) {
            return Err(DistributedError::InvalidConfig(format!(
                "broadcast cannot encode a dimension larger than {MAX_EXACT_DIM}: {local_shape:?}"
            ))
            .into());
        }

        let mut shape_dims: Vec<f32> = local_shape.iter().map(|dim| *dim as f32).collect();
        self.collective.broadcast(&mut shape_dims, src_rank)?;
        let shape: Vec<usize> = shape_dims.iter().map(|dim| *dim as usize).collect();

        let mut values = tensor.to_vec_f32()?;
        self.collective.broadcast(&mut values, src_rank)?;
        *tensor = Tensor::from_slice(&values, &shape)?;
        Ok(())
    }

    fn reduce(&self, tensor: &mut Tensor, dst_rank: usize) -> Result<()> {
        let shape = tensor.shape();
        let mut values = tensor.to_vec_f32()?;
        self.collective.reduce(&mut values, dst_rank, ReduceOp::Sum)?;
        *tensor = Tensor::from_slice(&values, &shape)?;
        Ok(())
    }

    fn barrier(&self) -> Result<()> {
        self.collective.barrier()
    }

    fn rank(&self) -> usize {
        self.collective.rank()
    }

    fn world_size(&self) -> usize {
        self.collective.world_size()
    }

    fn all_gather(&self, tensor: &Tensor) -> Result<Vec<Tensor>> {
        CollectiveProcessGroup::all_gather(self, tensor)
    }

    fn reduce_scatter(&self, tensor: &Tensor) -> Result<Tensor> {
        CollectiveProcessGroup::reduce_scatter(self, tensor, ReduceOp::Sum)
    }

    fn send(&self, peer: usize, tag: u64, tensor: &Tensor) -> Result<()> {
        self.collective.send_tagged(peer, tag, &tensor.to_vec_f32()?)
    }

    fn recv(&self, peer: usize, tag: u64, shape: &[usize]) -> Result<Tensor> {
        let values = self.collective.recv_tagged(peer, tag)?;
        self.reshape_received(peer, values, shape)
    }

    fn supports_point_to_point(&self) -> bool {
        true
    }
}

/// Process group whose ranks are threads of a single process.
#[derive(Debug)]
pub struct InProcessProcessGroup {
    inner: CollectiveProcessGroup,
}

impl InProcessProcessGroup {
    /// Join (or create) the named shared-memory session as `rank`.
    ///
    /// Every rank of a session must pass the same `session` key and
    /// `world_size`, and a distinct `rank`.
    pub fn join(session: &str, rank: usize, world_size: usize) -> Result<Self> {
        let session = join_in_process_session(session, world_size)?;
        let transport: Arc<dyn Transport> = Arc::new(session.transport(rank)?);
        Ok(Self {
            inner: CollectiveProcessGroup::new(transport)?,
        })
    }

    /// Build one group per rank over a fresh anonymous session.
    ///
    /// This is the ergonomic entry point for spawning `world_size` worker
    /// threads in one process.
    pub fn group(world_size: usize) -> Result<Vec<Self>> {
        let session = InProcessSession::new(world_size)?;
        (0..world_size)
            .map(|rank| {
                let transport: Arc<dyn Transport> = Arc::new(session.transport(rank)?);
                Ok(Self {
                    inner: CollectiveProcessGroup::new(transport)?,
                })
            })
            .collect()
    }

    /// Build a group from an already-claimed transport.
    pub fn from_transport(transport: InProcessTransport) -> Result<Self> {
        let transport: Arc<dyn Transport> = Arc::new(transport);
        Ok(Self {
            inner: CollectiveProcessGroup::new(transport)?,
        })
    }

    /// The shared collective implementation.
    pub fn inner(&self) -> &CollectiveProcessGroup {
        &self.inner
    }
}

impl ProcessGroup for InProcessProcessGroup {
    fn all_reduce(&self, tensors: &mut [Tensor]) -> Result<()> {
        self.inner.all_reduce(tensors)
    }
    fn broadcast(&self, tensor: &mut Tensor, src_rank: usize) -> Result<()> {
        self.inner.broadcast(tensor, src_rank)
    }
    fn reduce(&self, tensor: &mut Tensor, dst_rank: usize) -> Result<()> {
        self.inner.reduce(tensor, dst_rank)
    }
    fn barrier(&self) -> Result<()> {
        self.inner.barrier()
    }
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn world_size(&self) -> usize {
        self.inner.world_size()
    }
    fn all_gather(&self, tensor: &Tensor) -> Result<Vec<Tensor>> {
        ProcessGroup::all_gather(&self.inner, tensor)
    }
    fn reduce_scatter(&self, tensor: &Tensor) -> Result<Tensor> {
        ProcessGroup::reduce_scatter(&self.inner, tensor)
    }
    fn send(&self, peer: usize, tag: u64, tensor: &Tensor) -> Result<()> {
        ProcessGroup::send(&self.inner, peer, tag, tensor)
    }
    fn recv(&self, peer: usize, tag: u64, shape: &[usize]) -> Result<Tensor> {
        ProcessGroup::recv(&self.inner, peer, tag, shape)
    }
    fn supports_point_to_point(&self) -> bool {
        true
    }
}

/// Process group whose ranks are separate processes connected over TCP.
#[derive(Debug)]
pub struct TcpProcessGroup {
    inner: CollectiveProcessGroup,
}

impl TcpProcessGroup {
    /// Bind `master_port + rank` and connect to the remaining ranks at
    /// `master_addr:(master_port + peer)`.
    pub fn connect(
        rank: usize,
        world_size: usize,
        master_addr: &str,
        master_port: u16,
    ) -> Result<Self> {
        let transport = TcpTransport::from_master(rank, world_size, master_addr, master_port)?;
        Self::from_transport(transport)
    }

    /// Connect using an explicit peer table where `peers[i]` is rank `i`'s
    /// address. Use this for multi-host jobs.
    pub fn with_peers(rank: usize, peers: Vec<SocketAddr>) -> Result<Self> {
        Self::from_transport(TcpTransport::bind(rank, peers)?)
    }

    /// Build a group from an already-bound transport.
    pub fn from_transport(transport: TcpTransport) -> Result<Self> {
        let transport: Arc<dyn Transport> = Arc::new(transport);
        Ok(Self {
            inner: CollectiveProcessGroup::new(transport)?,
        })
    }

    /// The shared collective implementation.
    pub fn inner(&self) -> &CollectiveProcessGroup {
        &self.inner
    }
}

impl ProcessGroup for TcpProcessGroup {
    fn all_reduce(&self, tensors: &mut [Tensor]) -> Result<()> {
        self.inner.all_reduce(tensors)
    }
    fn broadcast(&self, tensor: &mut Tensor, src_rank: usize) -> Result<()> {
        self.inner.broadcast(tensor, src_rank)
    }
    fn reduce(&self, tensor: &mut Tensor, dst_rank: usize) -> Result<()> {
        self.inner.reduce(tensor, dst_rank)
    }
    fn barrier(&self) -> Result<()> {
        self.inner.barrier()
    }
    fn rank(&self) -> usize {
        self.inner.rank()
    }
    fn world_size(&self) -> usize {
        self.inner.world_size()
    }
    fn all_gather(&self, tensor: &Tensor) -> Result<Vec<Tensor>> {
        ProcessGroup::all_gather(&self.inner, tensor)
    }
    fn reduce_scatter(&self, tensor: &Tensor) -> Result<Tensor> {
        ProcessGroup::reduce_scatter(&self.inner, tensor)
    }
    fn send(&self, peer: usize, tag: u64, tensor: &Tensor) -> Result<()> {
        ProcessGroup::send(&self.inner, peer, tag, tensor)
    }
    fn recv(&self, peer: usize, tag: u64, shape: &[usize]) -> Result<Tensor> {
        ProcessGroup::recv(&self.inner, peer, tag, shape)
    }
    fn supports_point_to_point(&self) -> bool {
        true
    }
}

/// Spawn `world_size` threads, each holding its own real process group, and
/// collect their results in rank order.
///
/// This is the standard way to drive multi-rank code in one process, and the
/// harness the distributed tests use.
pub fn run_in_process<R, F>(world_size: usize, body: F) -> Result<Vec<R>>
where
    R: Send + 'static,
    F: Fn(usize, &InProcessProcessGroup) -> R + Send + Sync + 'static,
{
    let groups = InProcessProcessGroup::group(world_size)?;
    let body = Arc::new(body);

    let handles: Vec<_> = groups
        .into_iter()
        .enumerate()
        .map(|(rank, group)| {
            let body = Arc::clone(&body);
            std::thread::spawn(move || body(rank, &group))
        })
        .collect();

    let mut results = Vec::with_capacity(world_size);
    for handle in handles {
        results.push(
            handle
                .join()
                .map_err(|_| anyhow::anyhow!("a rank thread panicked during the collective"))?,
        );
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_reduce_sums_across_ranks() {
        let results = run_in_process(4, |rank, group| {
            let mut tensors = vec![Tensor::from_slice(
                &[rank as f32, rank as f32 + 1.0, rank as f32 + 2.0],
                &[3],
            )
            .expect("tensor must build in test")];
            group.all_reduce(&mut tensors).expect("all_reduce must succeed in test");
            tensors[0].to_vec_f32().expect("tensor read must succeed in test")
        })
        .expect("in-process run must succeed in test");

        // 0+1+2+3 = 6 for the first element, then 6+4 and 6+8.
        for values in &results {
            assert_eq!(values.len(), 3);
            approx::assert_relative_eq!(values[0], 6.0f32, epsilon = 1e-5);
            approx::assert_relative_eq!(values[1], 10.0f32, epsilon = 1e-5);
            approx::assert_relative_eq!(values[2], 14.0f32, epsilon = 1e-5);
        }
    }

    #[test]
    fn broadcast_copies_the_root_tensor_bit_exactly() {
        let source: Vec<f32> = (0..6).map(|i| 0.0625_f32 * (i as f32 + 1.0)).collect();
        let expected = source.clone();

        let results = run_in_process(3, move |rank, group| {
            let mut tensor = if rank == 1 {
                Tensor::from_slice(&source, &[2, 3]).expect("tensor must build in test")
            } else {
                Tensor::from_slice(&[-1.0f32; 2], &[2]).expect("tensor must build in test")
            };
            group.broadcast(&mut tensor, 1).expect("broadcast must succeed in test");
            (
                tensor.shape(),
                tensor.to_vec_f32().expect("tensor read must succeed in test"),
            )
        })
        .expect("in-process run must succeed in test");

        for (shape, values) in &results {
            assert_eq!(shape, &vec![2, 3], "the root's shape must win");
            assert_eq!(values, &expected, "broadcast must be bit-exact");
        }
    }

    #[test]
    fn all_gather_preserves_rank_order() {
        let results = run_in_process(3, |rank, group| {
            let local = Tensor::from_slice(&[rank as f32, -(rank as f32)], &[2])
                .expect("tensor must build in test");
            group
                .inner()
                .all_gather(&local)
                .expect("all_gather must succeed in test")
                .iter()
                .map(|tensor| tensor.to_vec_f32().expect("tensor read must succeed in test"))
                .collect::<Vec<_>>()
        })
        .expect("in-process run must succeed in test");

        for gathered in &results {
            assert_eq!(gathered.len(), 3);
            for (peer, values) in gathered.iter().enumerate() {
                assert_eq!(values, &vec![peer as f32, -(peer as f32)]);
            }
        }
    }

    #[test]
    fn reduce_scatter_returns_the_owning_chunk() {
        let results = run_in_process(2, |rank, group| {
            let tensor = Tensor::from_slice(&[rank as f32 + 1.0; 4], &[4])
                .expect("tensor must build in test");
            let chunk = group
                .inner()
                .reduce_scatter(&tensor, ReduceOp::Sum)
                .expect("reduce_scatter must succeed in test");
            chunk.to_vec_f32().expect("tensor read must succeed in test")
        })
        .expect("in-process run must succeed in test");

        // Sum over both ranks is 3 everywhere; each rank owns two elements.
        for chunk in &results {
            assert_eq!(chunk, &vec![3.0f32, 3.0]);
        }
    }

    #[test]
    fn reduce_delivers_the_sum_only_to_the_destination() {
        let results = run_in_process(4, |rank, group| {
            let mut tensor = Tensor::from_slice(&[rank as f32 + 1.0, 2.0], &[2])
                .expect("tensor must build in test");
            group.reduce(&mut tensor, 0).expect("reduce must succeed in test");
            tensor.to_vec_f32().expect("tensor read must succeed in test")
        })
        .expect("in-process run must succeed in test");

        assert_eq!(results[0], vec![10.0f32, 8.0]); // 1+2+3+4 and 2*4
        for (rank, values) in results.iter().enumerate().skip(1) {
            assert_eq!(
                values,
                &vec![rank as f32 + 1.0, 2.0],
                "rank {rank} untouched"
            );
        }
    }

    #[test]
    fn barrier_synchronises_ranks() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let arrived = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&arrived);
        let results = run_in_process(4, move |_rank, group| {
            counter.fetch_add(1, Ordering::SeqCst);
            group.barrier().expect("barrier must succeed in test");
            counter.load(Ordering::SeqCst)
        })
        .expect("in-process run must succeed in test");

        for count in results {
            assert_eq!(count, 4);
        }
    }

    #[test]
    fn point_to_point_moves_a_real_tensor() {
        let results = run_in_process(2, |rank, group| {
            if rank == 0 {
                let tensor =
                    Tensor::from_slice(&[1.5, -2.5, 3.5], &[3]).expect("tensor must build in test");
                group.inner().send_ordered(1, &tensor).expect("send must succeed in test");
                Vec::new()
            } else {
                group
                    .inner()
                    .recv_ordered(0, &[3])
                    .expect("recv must succeed in test")
                    .to_vec_f32()
                    .expect("tensor read must succeed in test")
            }
        })
        .expect("in-process run must succeed in test");

        assert_eq!(results[1], vec![1.5f32, -2.5, 3.5]);
    }

    #[test]
    fn named_sessions_rendezvous_across_threads() {
        let handles: Vec<_> = (0..2)
            .map(|rank| {
                std::thread::spawn(move || {
                    let group = InProcessProcessGroup::join("collective-test-session", rank, 2)
                        .expect("join must succeed in test");
                    let mut tensors = vec![Tensor::from_slice(&[rank as f32 + 1.0], &[1])
                        .expect("tensor must build in test")];
                    group.all_reduce(&mut tensors).expect("all_reduce must succeed in test");
                    tensors[0].to_vec_f32().expect("tensor read must succeed in test")
                })
            })
            .collect();

        for handle in handles {
            let values = handle.join().expect("rank thread must not panic in test");
            assert_eq!(values, vec![3.0f32]);
        }
    }
}
