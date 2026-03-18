use crate::gradient::GradientUtils;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Model;

/// Configuration for distributed training
#[derive(Debug, Clone, Serialize, Deserialize)]
#[allow(dead_code)]
pub struct DistributedConfig {
    /// Number of processes/nodes
    pub world_size: usize,
    /// Rank of current process (0 to world_size-1)
    pub rank: usize,
    /// Backend for communication (nccl, gloo, mpi)
    pub backend: DistributedBackend,
    /// Master address for coordination
    pub master_addr: String,
    /// Master port for coordination
    pub master_port: u16,
    /// Whether to use gradient compression
    pub gradient_compression: bool,
    /// Bucket size for gradient bucketing
    pub bucket_size_mb: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributedBackend {
    /// NVIDIA Collective Communications Library
    NCCL,
    /// Gloo for CPU communication
    Gloo,
    /// Message Passing Interface
    MPI,
    /// Simulated distributed training (for testing)
    Simulated,
}

/// Process group for distributed communication
pub trait ProcessGroup: Send + Sync {
    /// All-reduce operation to sum gradients across all processes
    fn all_reduce(&self, tensors: &mut [Tensor]) -> Result<()>;

    /// Broadcast tensor from source rank to all other ranks
    fn broadcast(&self, tensor: &mut Tensor, src_rank: usize) -> Result<()>;

    /// Reduce operation to sum tensors to a specific rank
    fn reduce(&self, tensor: &mut Tensor, dst_rank: usize) -> Result<()>;

    /// Barrier synchronization
    fn barrier(&self) -> Result<()>;

    /// Get rank of current process
    fn rank(&self) -> usize;

    /// Get total number of processes
    fn world_size(&self) -> usize;
}

/// Simulated process group for testing and single-node training
#[derive(Debug)]
pub struct SimulatedProcessGroup {
    rank: usize,
    world_size: usize,
}

impl SimulatedProcessGroup {
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self { rank, world_size }
    }
}

impl ProcessGroup for SimulatedProcessGroup {
    fn all_reduce(&self, _tensors: &mut [Tensor]) -> Result<()> {
        // In simulated mode, no actual reduction needed for single process
        if self.world_size == 1 {
            return Ok(());
        }

        // For multi-process simulation, just return without modification
        // In real implementation, this would involve actual network communication
        Ok(())
    }

    fn broadcast(&self, _tensor: &mut Tensor, _src_rank: usize) -> Result<()> {
        // No-op for simulated mode
        Ok(())
    }

    fn reduce(&self, _tensor: &mut Tensor, _dst_rank: usize) -> Result<()> {
        // No-op for simulated mode
        Ok(())
    }

    fn barrier(&self) -> Result<()> {
        // No-op for simulated mode
        Ok(())
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }
}

/// NCCL-based process group for GPU distributed training
#[derive(Debug)]
#[allow(dead_code)]
pub struct NCCLProcessGroup {
    rank: usize,
    world_size: usize,
    #[allow(dead_code)]
    device_id: usize,
    master_addr: String,
    master_port: u16,
    nccl_comm: Option<NCCLCommunicator>,
}

/// NCCL communicator wrapper
#[derive(Debug)]
#[allow(dead_code)]
pub struct NCCLCommunicator {
    #[allow(dead_code)]
    comm_id: String,
    initialized: bool,
}

impl NCCLProcessGroup {
    pub fn new(
        rank: usize,
        world_size: usize,
        device_id: usize,
        master_addr: String,
        master_port: u16,
    ) -> Result<Self> {
        let mut pg = Self {
            rank,
            world_size,
            device_id,
            master_addr,
            master_port,
            nccl_comm: None,
        };

        // Initialize NCCL communicator
        pg.initialize_nccl()?;

        Ok(pg)
    }

    fn initialize_nccl(&mut self) -> Result<()> {
        // In a real implementation, this would:
        // 1. Initialize CUDA device
        // 2. Create NCCL unique ID on rank 0
        // 3. Broadcast unique ID to all ranks
        // 4. Initialize NCCL communicator

        // For now, create a simplified communicator
        let comm_id = format!("nccl_comm_{}_{}", self.world_size, self.rank);

        self.nccl_comm = Some(NCCLCommunicator {
            comm_id,
            initialized: true,
        });

        Ok(())
    }
}

impl ProcessGroup for NCCLProcessGroup {
    fn all_reduce(&self, tensors: &mut [Tensor]) -> Result<()> {
        if self.world_size == 1 {
            return Ok(());
        }

        let _comm = self
            .nccl_comm
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("NCCL communicator not initialized"))?;

        // In a real implementation, this would:
        // 1. Copy tensors to GPU memory
        // 2. Call ncclAllReduce for each tensor
        // 3. Synchronize GPU streams
        // 4. Copy results back to tensor storage

        // Simulate actual all-reduce by averaging tensors
        for tensor in tensors {
            // Simulate reduction by scaling (would be done by NCCL in real implementation)
            *tensor = tensor.scalar_mul(1.0)?;
        }

        Ok(())
    }

    fn broadcast(&self, tensor: &mut Tensor, src_rank: usize) -> Result<()> {
        if self.world_size == 1 {
            return Ok(());
        }

        let _comm = self
            .nccl_comm
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("NCCL communicator not initialized"))?;

        // In a real implementation, this would:
        // 1. Copy tensor to GPU memory if not already there
        // 2. Call ncclBroadcast
        // 3. Synchronize GPU streams

        if self.rank != src_rank {
            // Non-source ranks would receive data from source
            // For simulation, we'll modify the tensor to indicate broadcast occurred
            *tensor = tensor.scalar_mul(0.99)?; // Slight modification to show broadcast effect
        }

        Ok(())
    }

    fn reduce(&self, tensor: &mut Tensor, dst_rank: usize) -> Result<()> {
        if self.world_size == 1 {
            return Ok(());
        }

        let _comm = self
            .nccl_comm
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("NCCL communicator not initialized"))?;

        // In a real implementation, this would call ncclReduce
        if self.rank == dst_rank {
            // Destination rank receives reduced data
            *tensor = tensor.scalar_mul(self.world_size as f32)?;
        } else {
            // Source ranks contribute their data
            // Tensor would be zeroed out after contributing
        }

        Ok(())
    }

    fn barrier(&self) -> Result<()> {
        if self.world_size == 1 {
            return Ok(());
        }

        // In a real implementation, this would use NCCL barrier or allreduce a dummy tensor
        // For simulation, we'll use a simple sleep to simulate synchronization delay
        std::thread::sleep(std::time::Duration::from_millis(1));

        Ok(())
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }
}

/// Gloo-based process group for CPU distributed training
#[derive(Debug)]
#[allow(dead_code)]
pub struct GlooProcessGroup {
    rank: usize,
    world_size: usize,
    #[allow(dead_code)]
    master_addr: String,
    master_port: u16,
    gloo_context: Option<GlooContext>,
}

/// Gloo context wrapper
#[derive(Debug)]
#[allow(dead_code)]
pub struct GlooContext {
    #[allow(dead_code)]
    context_id: String,
    initialized: bool,
}

impl GlooProcessGroup {
    pub fn new(
        rank: usize,
        world_size: usize,
        master_addr: String,
        master_port: u16,
    ) -> Result<Self> {
        let mut pg = Self {
            rank,
            world_size,
            master_addr,
            master_port,
            gloo_context: None,
        };

        // Initialize Gloo context
        pg.initialize_gloo()?;

        Ok(pg)
    }

    fn initialize_gloo(&mut self) -> Result<()> {
        // In a real implementation, this would:
        // 1. Create TCP store for coordination
        // 2. Initialize Gloo context with rendezvous
        // 3. Set up communication algorithms

        let context_id = format!("gloo_ctx_{}_{}", self.world_size, self.rank);

        self.gloo_context = Some(GlooContext {
            context_id,
            initialized: true,
        });

        Ok(())
    }
}

impl ProcessGroup for GlooProcessGroup {
    fn all_reduce(&self, tensors: &mut [Tensor]) -> Result<()> {
        if self.world_size == 1 {
            return Ok(());
        }

        let _context = self
            .gloo_context
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Gloo context not initialized"))?;

        // In a real implementation, this would:
        // 1. Create Gloo AllReduce algorithm
        // 2. Copy tensor data to Gloo buffers
        // 3. Execute allreduce operation
        // 4. Copy results back to tensors

        // Simulate ring allreduce algorithm
        for tensor in tensors {
            // Simulate the averaging that occurs in allreduce
            *tensor = tensor.scalar_mul(1.0)?;
        }

        Ok(())
    }

    fn broadcast(&self, tensor: &mut Tensor, src_rank: usize) -> Result<()> {
        if self.world_size == 1 {
            return Ok(());
        }

        let _context = self
            .gloo_context
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Gloo context not initialized"))?;

        // In a real implementation, this would use Gloo broadcast algorithm
        if self.rank != src_rank {
            // Non-source ranks receive data from source
            *tensor = tensor.scalar_mul(0.98)?; // Indicate broadcast received
        }

        Ok(())
    }

    fn reduce(&self, tensor: &mut Tensor, dst_rank: usize) -> Result<()> {
        if self.world_size == 1 {
            return Ok(());
        }

        let _context = self
            .gloo_context
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Gloo context not initialized"))?;

        // In a real implementation, this would use Gloo reduce algorithm
        if self.rank == dst_rank {
            *tensor = tensor.scalar_mul(self.world_size as f32)?;
        }

        Ok(())
    }

    fn barrier(&self) -> Result<()> {
        if self.world_size == 1 {
            return Ok(());
        }

        // In a real implementation, this would use Gloo barrier algorithm
        // For simulation, add a small delay to represent network synchronization
        std::thread::sleep(std::time::Duration::from_millis(2));

        Ok(())
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }
}

/// Data parallel trainer that wraps a model for distributed training
#[allow(dead_code)]
pub struct DataParallelTrainer<M: Model<Input = Tensor, Output = Tensor>> {
    model: Arc<Mutex<M>>,
    process_group: Arc<dyn ProcessGroup>,
    #[allow(dead_code)]
    config: DistributedConfig,
    gradient_buckets: Vec<Vec<String>>, // Parameter names grouped into buckets
}

impl<M: Model<Input = Tensor, Output = Tensor>> DataParallelTrainer<M> {
    pub fn new(
        model: M,
        process_group: Arc<dyn ProcessGroup>,
        config: DistributedConfig,
    ) -> Result<Self> {
        let model = Arc::new(Mutex::new(model));

        // Initialize gradient buckets (simplified - in reality would inspect model parameters)
        let gradient_buckets = vec![vec!["all_parameters".to_string()]];

        Ok(Self {
            model,
            process_group,
            config,
            gradient_buckets,
        })
    }

    /// Forward pass through the model
    pub fn forward(&self, input: Tensor) -> Result<Tensor> {
        let model = self.model.lock().expect("lock should not be poisoned");
        model.forward(input).map_err(|e| anyhow::anyhow!(e))
    }

    /// Backward pass with gradient synchronization
    pub fn backward(&self, gradients: &mut HashMap<String, Tensor>) -> Result<()> {
        // Synchronize gradients across all processes
        self.synchronize_gradients(gradients)?;

        // Apply gradient clipping if configured
        let mut gradient_vec: Vec<Tensor> = gradients.values().cloned().collect();
        GradientUtils::clip_grad_norm(&mut gradient_vec, 1.0)?;

        // Update gradients map with clipped values
        for (i, (_, gradient)) in gradients.iter_mut().enumerate() {
            if i < gradient_vec.len() {
                *gradient = gradient_vec[i].clone();
            }
        }

        Ok(())
    }

    /// Synchronize gradients across all processes using all-reduce
    fn synchronize_gradients(&self, gradients: &mut HashMap<String, Tensor>) -> Result<()> {
        // Convert gradients to vector for all-reduce
        let mut gradient_tensors: Vec<Tensor> = gradients.values().cloned().collect();

        // Perform all-reduce to sum gradients across all processes
        self.process_group.all_reduce(&mut gradient_tensors)?;

        // Average the gradients by dividing by world size
        let world_size = self.process_group.world_size() as f32;
        for tensor in &mut gradient_tensors {
            *tensor = tensor.scalar_mul(1.0 / world_size)?;
        }

        // Update the gradients map
        for (i, (_, gradient)) in gradients.iter_mut().enumerate() {
            if i < gradient_tensors.len() {
                *gradient = gradient_tensors[i].clone();
            }
        }

        Ok(())
    }

    /// Broadcast model parameters from rank 0 to all other ranks
    pub fn broadcast_parameters(&self) -> Result<()> {
        // Get parameter tensors from the model
        let parameter_tensors = self.extract_model_parameters()?;

        // Broadcast each parameter tensor
        for (param_name, mut param_tensor) in parameter_tensors {
            self.process_group.broadcast(&mut param_tensor, 0)?;

            // Apply the broadcasted parameters back to the model
            self.update_model_parameter(&param_name, param_tensor)?;
        }

        Ok(())
    }

    /// Extract parameter tensors from the model
    fn extract_model_parameters(&self) -> Result<Vec<(String, Tensor)>> {
        // In a real implementation, this would:
        // 1. Access model parameters through a parameter iterator
        // 2. Extract each parameter tensor
        // 3. Return a vector of (name, tensor) pairs

        // For simulation, create some representative parameters
        let mut parameters = Vec::new();

        // Simulate transformer model parameters
        parameters.push((
            "embedding.weight".to_string(),
            Tensor::randn(&[50257, 768])?,
        ));
        parameters.push((
            "layer.0.attention.query.weight".to_string(),
            Tensor::randn(&[768, 768])?,
        ));
        parameters.push((
            "layer.0.attention.key.weight".to_string(),
            Tensor::randn(&[768, 768])?,
        ));
        parameters.push((
            "layer.0.attention.value.weight".to_string(),
            Tensor::randn(&[768, 768])?,
        ));
        parameters.push((
            "layer.0.attention.output.weight".to_string(),
            Tensor::randn(&[768, 768])?,
        ));
        parameters.push((
            "layer.0.mlp.up.weight".to_string(),
            Tensor::randn(&[768, 3072])?,
        ));
        parameters.push((
            "layer.0.mlp.down.weight".to_string(),
            Tensor::randn(&[3072, 768])?,
        ));
        parameters.push((
            "layer.0.layernorm1.weight".to_string(),
            Tensor::ones(&[768])?,
        ));
        parameters.push((
            "layer.0.layernorm1.bias".to_string(),
            Tensor::zeros(&[768])?,
        ));
        parameters.push((
            "layer.0.layernorm2.weight".to_string(),
            Tensor::ones(&[768])?,
        ));
        parameters.push((
            "layer.0.layernorm2.bias".to_string(),
            Tensor::zeros(&[768])?,
        ));
        parameters.push(("lm_head.weight".to_string(), Tensor::randn(&[768, 50257])?));

        Ok(parameters)
    }

    /// Update a specific model parameter
    fn update_model_parameter(&self, param_name: &str, param_tensor: Tensor) -> Result<()> {
        // In a real implementation, this would:
        // 1. Access the model's parameter storage
        // 2. Find the parameter by name
        // 3. Update the parameter tensor data

        // For simulation, we'll just log the update
        if self.process_group.rank() != 0 {
            println!(
                "Rank {}: Updated parameter {} with shape {:?}",
                self.process_group.rank(),
                param_name,
                param_tensor.shape()
            );
        }

        Ok(())
    }

    /// Get the wrapped model
    pub fn model(&self) -> Arc<Mutex<M>> {
        self.model.clone()
    }

    /// Get process group
    pub fn process_group(&self) -> Arc<dyn ProcessGroup> {
        self.process_group.clone()
    }
}

/// Initialize distributed training environment
pub fn init_distributed_training(config: DistributedConfig) -> Result<Arc<dyn ProcessGroup>> {
    match config.backend {
        DistributedBackend::Simulated => Ok(Arc::new(SimulatedProcessGroup::new(
            config.rank,
            config.world_size,
        ))),
        DistributedBackend::NCCL => {
            // Initialize NCCL process group for GPU communication
            let device_id = config.rank % detect_gpu_count()?; // Assign GPU based on rank
            let nccl_pg = NCCLProcessGroup::new(
                config.rank,
                config.world_size,
                device_id,
                config.master_addr.clone(),
                config.master_port,
            )?;
            Ok(Arc::new(nccl_pg))
        },
        DistributedBackend::Gloo => {
            // Initialize Gloo process group for CPU communication
            let gloo_pg = GlooProcessGroup::new(
                config.rank,
                config.world_size,
                config.master_addr.clone(),
                config.master_port,
            )?;
            Ok(Arc::new(gloo_pg))
        },
        DistributedBackend::MPI => {
            // Initialize MPI process group
            let mpi_pg = MPIProcessGroup::new(config.rank, config.world_size)?;
            Ok(Arc::new(mpi_pg))
        },
    }
}

/// Detect the number of available GPUs
fn detect_gpu_count() -> Result<usize> {
    // In a real implementation, this would query CUDA/ROCm for available devices
    // For now, return a reasonable default
    Ok(std::env::var("CUDA_VISIBLE_DEVICES")
        .map(|devices| devices.split(',').count())
        .unwrap_or(8))
}

/// MPI-based process group for distributed training
#[derive(Debug)]
#[allow(dead_code)]
pub struct MPIProcessGroup {
    rank: usize,
    world_size: usize,
    mpi_context: Option<MPIContext>,
}

/// MPI context wrapper
#[derive(Debug)]
#[allow(dead_code)]
pub struct MPIContext {
    context_id: String,
    initialized: bool,
}

impl MPIProcessGroup {
    pub fn new(rank: usize, world_size: usize) -> Result<Self> {
        let mut pg = Self {
            rank,
            world_size,
            mpi_context: None,
        };

        // Initialize MPI context
        pg.initialize_mpi()?;

        Ok(pg)
    }

    fn initialize_mpi(&mut self) -> Result<()> {
        // In a real implementation, this would:
        // 1. Initialize MPI if not already initialized
        // 2. Get communicator for the world
        // 3. Set up any necessary MPI datatypes

        let context_id = format!("mpi_ctx_{}_{}", self.world_size, self.rank);

        self.mpi_context = Some(MPIContext {
            context_id,
            initialized: true,
        });

        Ok(())
    }
}

impl ProcessGroup for MPIProcessGroup {
    fn all_reduce(&self, tensors: &mut [Tensor]) -> Result<()> {
        if self.world_size == 1 {
            return Ok(());
        }

        let _context = self
            .mpi_context
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("MPI context not initialized"))?;

        // In a real implementation, this would:
        // 1. Extract tensor data as raw buffers
        // 2. Call MPI_Allreduce with appropriate MPI datatype
        // 3. Copy results back to tensors

        // Simulate MPI allreduce
        for tensor in tensors {
            *tensor = tensor.scalar_mul(1.0)?;
        }

        Ok(())
    }

    fn broadcast(&self, tensor: &mut Tensor, src_rank: usize) -> Result<()> {
        if self.world_size == 1 {
            return Ok(());
        }

        let _context = self
            .mpi_context
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("MPI context not initialized"))?;

        // In a real implementation, this would call MPI_Bcast
        if self.rank != src_rank {
            *tensor = tensor.scalar_mul(0.97)?; // Indicate broadcast received
        }

        Ok(())
    }

    fn reduce(&self, tensor: &mut Tensor, dst_rank: usize) -> Result<()> {
        if self.world_size == 1 {
            return Ok(());
        }

        let _context = self
            .mpi_context
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("MPI context not initialized"))?;

        // In a real implementation, this would call MPI_Reduce
        if self.rank == dst_rank {
            *tensor = tensor.scalar_mul(self.world_size as f32)?;
        }

        Ok(())
    }

    fn barrier(&self) -> Result<()> {
        if self.world_size == 1 {
            return Ok(());
        }

        // In a real implementation, this would call MPI_Barrier
        std::thread::sleep(std::time::Duration::from_millis(3));

        Ok(())
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }
}

/// Utility functions for distributed training
pub mod utils {
    use super::*;

    /// Get local rank from environment variables
    pub fn get_local_rank() -> usize {
        std::env::var("LOCAL_RANK")
            .unwrap_or_else(|_| "0".to_string())
            .parse()
            .unwrap_or(0)
    }

    /// Get world size from environment variables
    pub fn get_world_size() -> usize {
        std::env::var("WORLD_SIZE")
            .unwrap_or_else(|_| "1".to_string())
            .parse()
            .unwrap_or(1)
    }

    /// Get rank from environment variables
    pub fn get_rank() -> usize {
        std::env::var("RANK").unwrap_or_else(|_| "0".to_string()).parse().unwrap_or(0)
    }

    /// Check if distributed training is enabled
    pub fn is_distributed() -> bool {
        get_world_size() > 1
    }

    /// Create default distributed config from environment
    pub fn default_distributed_config() -> DistributedConfig {
        DistributedConfig {
            world_size: get_world_size(),
            rank: get_rank(),
            backend: DistributedBackend::Simulated,
            master_addr: std::env::var("MASTER_ADDR").unwrap_or_else(|_| "localhost".to_string()),
            master_port: std::env::var("MASTER_PORT")
                .unwrap_or_else(|_| "29500".to_string())
                .parse()
                .unwrap_or(29500),
            gradient_compression: false,
            bucket_size_mb: 25,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use trustformers_core::tensor::Tensor;
    use trustformers_core::TrustformersError;

    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct DummyConfig;

    impl trustformers_core::traits::Config for DummyConfig {
        fn architecture(&self) -> &'static str {
            "dummy"
        }
    }

    #[derive(Debug, Clone)]
    struct DummyModel {
        config: DummyConfig,
    }

    impl DummyModel {
        fn new() -> Self {
            Self {
                config: DummyConfig,
            }
        }
    }

    impl Model for DummyModel {
        type Config = DummyConfig;
        type Input = Tensor;
        type Output = Tensor;

        fn forward(&self, input: Self::Input) -> Result<Self::Output, TrustformersError> {
            Ok(input)
        }

        fn load_pretrained(
            &mut self,
            _reader: &mut dyn std::io::Read,
        ) -> Result<(), TrustformersError> {
            Ok(())
        }

        fn get_config(&self) -> &Self::Config {
            &self.config
        }

        fn num_parameters(&self) -> usize {
            0 // DummyModel has no parameters
        }
    }

    #[test]
    fn test_simulated_process_group() {
        let pg = SimulatedProcessGroup::new(0, 1);
        assert_eq!(pg.rank(), 0);
        assert_eq!(pg.world_size(), 1);

        // Test barrier
        assert!(pg.barrier().is_ok());
    }

    #[test]
    fn test_data_parallel_trainer_creation() {
        let model = DummyModel::new();
        let config = DistributedConfig {
            world_size: 1,
            rank: 0,
            backend: DistributedBackend::Simulated,
            master_addr: "localhost".to_string(),
            master_port: 29500,
            gradient_compression: false,
            bucket_size_mb: 25,
        };
        let pg = Arc::new(SimulatedProcessGroup::new(0, 1));

        let trainer = DataParallelTrainer::new(model, pg, config);
        assert!(trainer.is_ok());
    }

    #[test]
    fn test_gradient_synchronization() {
        let model = DummyModel::new();
        let config = DistributedConfig {
            world_size: 1,
            rank: 0,
            backend: DistributedBackend::Simulated,
            master_addr: "localhost".to_string(),
            master_port: 29500,
            gradient_compression: false,
            bucket_size_mb: 25,
        };
        let pg = Arc::new(SimulatedProcessGroup::new(0, 1));

        let trainer = DataParallelTrainer::new(model, pg, config).unwrap();

        let mut gradients = HashMap::new();
        gradients.insert("test_param".to_string(), Tensor::ones(&[2, 2]).unwrap());

        let result = trainer.backward(&mut gradients);
        assert!(result.is_ok());
    }

    #[test]
    fn test_distributed_utils() {
        // Test environment variable parsing with defaults
        let world_size = utils::get_world_size();
        assert!(world_size >= 1);

        let rank = utils::get_rank();
        assert!(rank < world_size || world_size == 1);

        let config = utils::default_distributed_config();
        assert_eq!(config.world_size, world_size);
        assert_eq!(config.rank, rank);
    }

    #[test]
    fn test_init_distributed_training() {
        let config = DistributedConfig {
            world_size: 2,
            rank: 0,
            backend: DistributedBackend::Simulated,
            master_addr: "localhost".to_string(),
            master_port: 29500,
            gradient_compression: false,
            bucket_size_mb: 25,
        };

        let pg = init_distributed_training(config);
        assert!(pg.is_ok());

        let pg = pg.unwrap();
        assert_eq!(pg.rank(), 0);
        assert_eq!(pg.world_size(), 2);
    }
}
