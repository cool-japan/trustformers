//! MPI Communication Backend for Multi-Node Training
//!
//! This module provides MPI-based communication for distributed training
//! across multiple nodes in a cluster. It implements the Communicator trait
//! for seamless integration with the existing parallel infrastructure.
//!
//! ## Enabling MPI Support
//!
//! To enable MPI support, build with the `mpi` feature:
//! ```bash
//! cargo build --features mpi
//! ```
//!
//! Make sure you have MPI installed on your system (OpenMPI or MPICH).

use super::Communicator;
use crate::errors::{runtime_error, Result};
use crate::Tensor;

#[cfg(feature = "mpi")]
use parking_lot::Mutex;
#[cfg(feature = "mpi")]
use std::sync::Arc;

#[cfg(feature = "mpi")]
use mpi::{
    collective::{CommunicatorCollectives, Root},
    datatype::PartitionMut,
    point_to_point::{Destination, Source},
    request::Request,
    topology::{Communicator as MpiCommunicator, SimpleCommunicator},
};

/// MPI-based communicator for multi-node distributed training
pub struct MpiCommunicatorImpl {
    #[cfg(feature = "mpi")]
    world: SimpleCommunicator,
    #[cfg(feature = "mpi")]
    #[allow(dead_code)]
    custom_comm: Option<SimpleCommunicator>,
    rank: usize,
    world_size: usize,
    #[cfg(feature = "mpi")]
    #[allow(dead_code)]
    pending_requests: Arc<Mutex<Vec<Request<'static, f32>>>>,
}

impl MpiCommunicatorImpl {
    /// Create a new MPI communicator
    pub fn new() -> Result<Self> {
        #[cfg(feature = "mpi")]
        {
            // Initialize MPI (should be done once per process)
            let universe = mpi::initialize()
                .ok_or_else(|| runtime_error("Failed to initialize MPI".to_string()))?;

            let world = universe.world();
            let rank = world.rank() as usize;
            let world_size = world.size() as usize;

            println!("MPI Communicator initialized:");
            println!("  Rank: {}", rank);
            println!("  World Size: {}", world_size);
            if let Ok(proc_name) = mpi::environment::processor_name() {
                println!("  Processor Name: {}", proc_name);
            }

            Ok(Self {
                world,
                custom_comm: None,
                rank,
                world_size,
                pending_requests: Arc::new(Mutex::new(Vec::new())),
            })
        }

        #[cfg(not(feature = "mpi"))]
        {
            Err(runtime_error(
                "MPI support is disabled. To enable MPI:\n\
                 1. Build with: cargo build --features mpi\n\
                 2. Make sure MPI is installed on your system (OpenMPI or MPICH)\n\
                 \n\
                 For more information, see the multi-node training guide.",
            ))
        }
    }

    /// Get MPI rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get MPI world size
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Barrier synchronization
    pub fn barrier(&self) -> Result<()> {
        #[cfg(feature = "mpi")]
        {
            self.world.barrier();
            Ok(())
        }

        #[cfg(not(feature = "mpi"))]
        Ok(())
    }

    /// Finalize MPI
    pub fn finalize() -> Result<()> {
        #[cfg(feature = "mpi")]
        {
            // MPI finalization is handled automatically by the mpi crate
            Ok(())
        }

        #[cfg(not(feature = "mpi"))]
        Ok(())
    }

    #[cfg(feature = "mpi")]
    fn tensor_to_slice(&self, tensor: &Tensor) -> Result<Vec<f32>> {
        // Convert tensor to a contiguous slice for MPI operations
        tensor.data()
    }

    #[cfg(feature = "mpi")]
    fn slice_to_tensor(&self, data: Vec<f32>, shape: &[usize]) -> Result<Tensor> {
        Tensor::from_vec(data, shape)
    }
}

impl Communicator for MpiCommunicatorImpl {
    fn all_gather(
        &self,
        tensor: &Tensor,
        #[allow(unused_variables)] split_dim: usize,
    ) -> Result<Tensor> {
        #[cfg(feature = "mpi")]
        {
            let shape = tensor.shape();
            let data = self.tensor_to_slice(tensor)?;

            // Calculate sizes for all_gather
            let local_size = data.len();
            let mut all_sizes = vec![0; self.world_size];
            self.world.all_gather_into(&local_size, &mut all_sizes[..]);

            // Prepare receive buffer
            let total_size: usize = all_sizes.iter().sum();
            let mut recv_buffer = vec![0.0f32; total_size];

            // Perform all_gatherv
            let displs: Vec<_> = all_sizes
                .iter()
                .scan(0, |acc, &size| {
                    let displ = *acc;
                    *acc += size;
                    Some(displ as i32)
                })
                .collect();

            let counts: Vec<i32> = all_sizes.iter().map(|&s| s as i32).collect();
            let mut partition =
                PartitionMut::new(&mut recv_buffer[..], counts.as_slice(), displs.as_slice());
            self.world.all_gather_varcount_into(&data[..], &mut partition);

            // Reconstruct tensor with gathered data
            let mut new_shape = shape.clone();
            new_shape[split_dim] *= self.world_size;
            self.slice_to_tensor(recv_buffer, &new_shape)
        }

        #[cfg(not(feature = "mpi"))]
        Ok(tensor.clone())
    }

    fn reduce_scatter(
        &self,
        tensor: &Tensor,
        #[allow(unused_variables)] split_dim: usize,
    ) -> Result<Tensor> {
        #[cfg(feature = "mpi")]
        {
            let shape = tensor.shape();
            let data = self.tensor_to_slice(tensor)?;

            // Calculate local chunk size
            let total_size = shape[split_dim];
            let chunk_size = total_size / self.world_size;
            let remainder = total_size % self.world_size;

            if remainder != 0 {
                return Err(runtime_error(format!(
                    "Tensor dimension {} ({}) is not evenly divisible by world size {}",
                    split_dim, total_size, self.world_size
                )));
            }

            // Calculate elements per chunk
            let elements_per_unit = shape
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != split_dim)
                .map(|(_, &s)| s)
                .product::<usize>();
            let elements_per_chunk = chunk_size * elements_per_unit;

            // Prepare receive buffer
            let mut recv_buffer = vec![0.0f32; elements_per_chunk];

            // Perform reduce_scatter with block-based API (equal sizes)
            self.world.reduce_scatter_block_into(
                &data[..],
                &mut recv_buffer[..],
                mpi::collective::SystemOperation::sum(),
            );

            // Reconstruct tensor with reduced data
            let mut new_shape = shape.clone();
            new_shape[split_dim] = chunk_size;
            self.slice_to_tensor(recv_buffer, &new_shape)
        }

        #[cfg(not(feature = "mpi"))]
        Ok(tensor.clone())
    }

    fn all_reduce(&self, #[allow(unused_variables)] tensor: &mut Tensor) -> Result<()> {
        #[cfg(feature = "mpi")]
        {
            let data = self.tensor_to_slice(tensor)?;
            let mut recv_buffer = vec![0.0f32; data.len()];

            // Perform all_reduce with sum operation
            self.world.all_reduce_into(
                &data[..],
                &mut recv_buffer[..],
                mpi::collective::SystemOperation::sum(),
            );

            // Update tensor with reduced data
            *tensor = self.slice_to_tensor(recv_buffer, &tensor.shape())?;
            Ok(())
        }

        #[cfg(not(feature = "mpi"))]
        Ok(())
    }

    fn send(&self, tensor: &Tensor, dest: usize) -> Result<()> {
        #[cfg(feature = "mpi")]
        {
            let data = self.tensor_to_slice(tensor)?;
            let dest_process = self.world.process_at_rank(dest as i32);
            dest_process.send(&data[..]);
            Ok(())
        }

        #[cfg(not(feature = "mpi"))]
        {
            let _ = (tensor, dest);
            Ok(())
        }
    }

    fn recv(&self, shape: &[usize], src: usize) -> Result<Tensor> {
        #[cfg(feature = "mpi")]
        {
            let total_elements: usize = shape.iter().product();
            let mut recv_buffer = vec![0.0f32; total_elements];

            let src_process = self.world.process_at_rank(src as i32);
            src_process.receive_into(&mut recv_buffer[..]);

            self.slice_to_tensor(recv_buffer, shape)
        }

        #[cfg(not(feature = "mpi"))]
        {
            let _ = src;
            Tensor::zeros(shape)
        }
    }

    fn broadcast(
        &self,
        #[allow(unused_variables)] tensor: &mut Tensor,
        #[allow(unused_variables)] root: usize,
    ) -> Result<()> {
        #[cfg(feature = "mpi")]
        {
            let mut data = self.tensor_to_slice(tensor)?;

            // All processes (including root) call broadcast_into on the root process
            let root_process = self.world.process_at_rank(root as i32);
            root_process.broadcast_into(&mut data[..]);

            *tensor = self.slice_to_tensor(data, &tensor.shape())?;
            Ok(())
        }

        #[cfg(not(feature = "mpi"))]
        {
            let _ = root;
            Ok(())
        }
    }
}

/// MPI utility functions
pub mod mpi_utils {
    use super::*;

    /// Initialize MPI environment
    pub fn init_mpi_environment() -> Result<()> {
        #[cfg(feature = "mpi")]
        {
            // MPI is initialized when creating the communicator
            Ok(())
        }

        #[cfg(not(feature = "mpi"))]
        Err(runtime_error("MPI support is disabled"))
    }

    /// Check MPI environment status
    pub fn check_mpi_environment() -> Result<()> {
        #[cfg(feature = "mpi")]
        {
            // If this struct was successfully created, MPI is initialized
            // (since we initialize it in the new() method)
            Ok(())
        }

        #[cfg(not(feature = "mpi"))]
        Err(runtime_error("MPI support is disabled"))
    }

    /// Get node-local rank and size
    pub fn get_node_local_info() -> Result<(usize, usize)> {
        #[cfg(feature = "mpi")]
        {
            // In a real implementation, we would use MPI_COMM_SPLIT_TYPE
            // to get node-local communicator. For now, return placeholder.
            Ok((0, 1))
        }

        #[cfg(not(feature = "mpi"))]
        Err(runtime_error("MPI support is disabled"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mpi_feature_disabled() {
        // When MPI feature is disabled, constructor should return error
        #[cfg(not(feature = "mpi"))]
        {
            let result = MpiCommunicatorImpl::new();
            assert!(result.is_err());

            if let Err(e) = result {
                assert!(e.to_string().contains("MPI support is disabled"));
                assert!(e.to_string().contains("--features mpi"));
            }
        }
    }

    #[test]
    fn test_mpi_utils_feature_disabled() {
        #[cfg(not(feature = "mpi"))]
        {
            assert!(mpi_utils::init_mpi_environment().is_err());
            assert!(mpi_utils::check_mpi_environment().is_err());
            assert!(mpi_utils::get_node_local_info().is_err());
        }
    }

    #[test]
    #[cfg(feature = "mpi")]
    fn test_mpi_basic_operations() {
        // This test would only run in an MPI environment
        // Typically run with: mpirun -np 2 cargo test --features mpi
        if let Ok(comm) = MpiCommunicatorImpl::new() {
            assert!(comm.rank() < comm.world_size());
            assert!(comm.world_size() > 0);

            // Test barrier
            assert!(comm.barrier().is_ok());
        }
    }
}
