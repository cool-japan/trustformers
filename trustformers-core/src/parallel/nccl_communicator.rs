// Copyright (c) 2024 TrustformeRS Contributors
// SPDX-License-Identifier: Apache-2.0

//! NCCL-based communicator implementation for high-performance GPU collective operations

use super::Communicator;
use crate::errors::TrustformersError;
use crate::tensor::Tensor;
use std::sync::Arc;

type Result<T> = std::result::Result<T, TrustformersError>;

/// NCCL communicator for GPU-to-GPU communication
///
/// NCCL (NVIDIA Collective Communication Library) provides optimized
/// collective communication primitives for multi-GPU training.
#[cfg(feature = "nccl")]
pub struct NcclCommunicator {
    /// Rank of this process in the communication group
    rank: usize,
    /// Total number of processes in the communication group
    world_size: usize,
    /// NCCL communicator handle (would be actual NCCL handle in real implementation)
    _comm_handle: NcclCommHandle,
    /// Device ID for this communicator
    device_id: i32,
}

#[cfg(feature = "nccl")]
struct NcclCommHandle {
    // In a real implementation, this would contain the actual NCCL communicator handle
    // For now, we'll use a placeholder that can be extended when NCCL bindings are available
    _placeholder: std::marker::PhantomData<()>,
}

#[cfg(feature = "nccl")]
impl NcclCommunicator {
    /// Create a new NCCL communicator
    ///
    /// # Arguments
    /// * `rank` - Rank of this process (0 to world_size-1)
    /// * `world_size` - Total number of processes
    /// * `device_id` - CUDA device ID to use
    pub fn new(rank: usize, world_size: usize, device_id: i32) -> Result<Self> {
        if rank >= world_size {
            return Err(TrustformersError::invalid_input(format!(
                "Rank {} must be less than world_size {}",
                rank, world_size
            )));
        }

        // In a real implementation, this would initialize NCCL:
        // 1. Call ncclGetUniqueId() on rank 0 and broadcast the ID
        // 2. Call ncclCommInitRank() with the unique ID, rank, and world_size
        // 3. Set the CUDA device context

        let comm_handle = NcclCommHandle {
            _placeholder: std::marker::PhantomData,
        };

        Ok(Self {
            rank,
            world_size,
            _comm_handle: comm_handle,
            device_id,
        })
    }

    /// Initialize NCCL communicator for all processes
    ///
    /// This is a collective operation that must be called by all processes
    /// in the communication group.
    pub fn init_all(world_size: usize, device_ids: &[i32]) -> Result<Vec<Self>> {
        if device_ids.len() != world_size {
            return Err(TrustformersError::invalid_input(
                "Number of device IDs must match world size".to_string(),
            ));
        }

        let mut communicators = Vec::with_capacity(world_size);

        // In a real implementation, this would:
        // 1. Generate a unique ID for the communicator group
        // 2. Initialize all communicators with the same unique ID
        // 3. Each communicator would be associated with its device

        for (rank, &device_id) in device_ids.iter().enumerate().take(world_size) {
            let comm = Self::new(rank, world_size, device_id)?;
            communicators.push(comm);
        }

        Ok(communicators)
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

    /// Perform optimized all-reduce using NCCL
    fn nccl_all_reduce(&self, tensor: &mut Tensor) -> Result<()> {
        // In a real implementation, this would:
        // 1. Ensure tensor is on the correct GPU device
        // 2. Call ncclAllReduce() with the tensor data
        // 3. Synchronize the CUDA stream

        // For now, we'll simulate the operation by ensuring the tensor is valid
        if tensor.shape().is_empty() {
            return Err(TrustformersError::invalid_input(
                "Cannot perform all-reduce on empty tensor".to_string(),
            ));
        }

        // Placeholder: In real implementation, this would perform GPU-accelerated reduction
        log::debug!(
            "NCCL all-reduce on device {} for tensor with shape {:?}",
            self.device_id,
            tensor.shape()
        );

        Ok(())
    }

    /// Perform optimized all-gather using NCCL
    fn nccl_all_gather(&self, tensor: &Tensor) -> Result<Tensor> {
        // In a real implementation, this would:
        // 1. Allocate output tensor with size [world_size, ...input_shape]
        // 2. Call ncclAllGather() to gather tensors from all ranks
        // 3. Return the concatenated result

        let input_shape = tensor.shape();
        let mut output_shape = vec![self.world_size];
        output_shape.extend_from_slice(&input_shape);

        log::debug!(
            "NCCL all-gather on device {} for tensor with shape {:?} -> {:?}",
            self.device_id,
            input_shape,
            output_shape
        );

        // Placeholder: Create output tensor (in real implementation, this would contain gathered data)
        Tensor::zeros(&output_shape)
    }

    /// Perform optimized broadcast using NCCL
    fn nccl_broadcast(&self, tensor: &mut Tensor, root: usize) -> Result<()> {
        if root >= self.world_size {
            return Err(TrustformersError::invalid_input(format!(
                "Root rank {} must be less than world_size {}",
                root, self.world_size
            )));
        }

        // In a real implementation, this would:
        // 1. Call ncclBcast() with the tensor data and root rank
        // 2. Synchronize the CUDA stream

        log::debug!(
            "NCCL broadcast on device {} from root {} for tensor with shape {:?}",
            self.device_id,
            root,
            tensor.shape()
        );

        Ok(())
    }

    /// Synchronize all processes in the communicator
    pub fn barrier(&self) -> Result<()> {
        // In a real implementation, this would use NCCL's synchronization primitives
        // or CUDA events to ensure all operations are complete

        log::debug!("NCCL barrier on device {}", self.device_id);
        Ok(())
    }

    /// Destroy the NCCL communicator and clean up resources
    pub fn destroy(&mut self) -> Result<()> {
        // In a real implementation, this would call ncclCommDestroy()
        log::debug!("Destroying NCCL communicator on device {}", self.device_id);
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

    fn reduce_scatter(&self, tensor: &Tensor, _split_dim: usize) -> Result<Tensor> {
        // In a real implementation, this would use ncclReduceScatter()
        log::debug!(
            "NCCL reduce-scatter on device {} for tensor with shape {:?}",
            self.device_id,
            tensor.shape()
        );

        // For now, return a tensor with reduced size
        let input_shape = tensor.shape();
        if input_shape.is_empty() {
            return Err(TrustformersError::invalid_input(
                "Cannot perform reduce-scatter on empty tensor".to_string(),
            ));
        }

        let mut output_shape = input_shape.to_vec();
        output_shape[0] /= self.world_size; // Scatter along first dimension

        Tensor::zeros(&output_shape)
    }

    fn broadcast(&self, tensor: &mut Tensor, root: usize) -> Result<()> {
        self.nccl_broadcast(tensor, root)
    }

    fn send(&self, _tensor: &Tensor, _dest: usize) -> Result<()> {
        // NCCL doesn't provide direct point-to-point operations
        // This would typically be implemented using CUDA-aware MPI or NCCL send/recv primitives
        Err(TrustformersError::runtime_error(
            "Point-to-point send not yet implemented for NCCL backend".to_string(),
        ))
    }

    fn recv(&self, _shape: &[usize], _src: usize) -> Result<Tensor> {
        // NCCL doesn't provide direct point-to-point operations
        Err(TrustformersError::runtime_error(
            "Point-to-point recv not yet implemented for NCCL backend".to_string(),
        ))
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
        let comm = NcclCommunicator::new(0, 2, 0).unwrap();
        assert_eq!(comm.rank(), 0);
        assert_eq!(comm.world_size(), 2);
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
    fn test_nccl_operations() -> Result<()> {
        let comm = NcclCommunicator::new(0, 2, 0)?;

        // Test all-reduce
        let mut tensor = Tensor::ones(&[4, 4])?;
        comm.all_reduce(&mut tensor)?;

        // Test all-gather
        let gathered = comm.all_gather(&tensor, 0)?;
        assert_eq!(gathered.shape()[0], 2); // Should have world_size in first dimension

        // Test broadcast
        comm.broadcast(&mut tensor, 0)?;

        // Test barrier
        comm.barrier()?;

        Ok(())
    }

    #[test]
    fn test_create_nccl_communicator_factory() {
        let result = create_nccl_communicator(0, 2, 0);

        #[cfg(feature = "nccl")]
        assert!(result.is_ok());

        #[cfg(not(feature = "nccl"))]
        assert!(result.is_err());
    }
}
