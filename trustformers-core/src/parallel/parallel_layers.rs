//! Parallel implementations of neural network layers for model parallelism
//!
//! This module provides distributed versions of common layers that can be
//! split across multiple devices for model parallel training.

#![allow(unused_variables)] // Parallel layer implementation

use super::model_parallel::{DistributedTensor, ModelParallelContext, TensorPartition};
use crate::errors::{tensor_op_error, Result};
use crate::Tensor;
use std::sync::Arc;

/// Column-parallel linear layer
///
/// Splits the weight matrix by columns (output dimension) across devices.
/// This is typically used for the first linear layer in MLP blocks.
pub struct ColumnParallelLinear {
    /// Local weight shard [in_features, out_features_per_device]
    weight: Tensor,
    /// Bias (only on rank 0 to avoid duplication)
    bias: Option<Tensor>,
    /// Model parallel context
    mp_context: Arc<ModelParallelContext>,
    /// Total input features
    #[allow(dead_code)]
    in_features: usize,
    /// Total output features (across all devices)
    out_features: usize,
}

impl ColumnParallelLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
        mp_context: Arc<ModelParallelContext>,
    ) -> Result<Self> {
        let world_size = mp_context.world_size();
        let rank = mp_context.rank();

        // Calculate local output features
        let out_features_per_device = out_features.div_ceil(world_size);
        let local_out_start = rank * out_features_per_device;
        let local_out_end = ((rank + 1) * out_features_per_device).min(out_features);
        let local_out_features = local_out_end - local_out_start;

        // Initialize local weight shard
        let weight = Tensor::randn(&[in_features, local_out_features])?;

        // Bias only on rank 0 to avoid duplication during all-reduce
        let bias = if bias && rank == 0 { Some(Tensor::zeros(&[out_features])?) } else { None };

        Ok(Self {
            weight,
            bias,
            mp_context,
            in_features,
            out_features,
        })
    }

    pub fn forward(&self, input: &Tensor) -> Result<DistributedTensor> {
        // Input: [batch_size, seq_len, in_features]
        // Weight: [in_features, local_out_features]
        // Output: [batch_size, seq_len, local_out_features]

        let output = input.matmul(&self.weight)?;

        // Add bias if present (only on rank 0)
        let output = if let Some(ref bias) = self.bias {
            // Slice bias to match local output features
            let rank = self.mp_context.rank();
            let world_size = self.mp_context.world_size();
            let out_features_per_device = self.out_features.div_ceil(world_size);
            let local_out_start = rank * out_features_per_device;
            let local_out_end = ((rank + 1) * out_features_per_device).min(self.out_features);

            let local_bias = bias.slice(0, local_out_start, local_out_end)?;
            output.add(&local_bias)?
        } else {
            output
        };

        // Create distributed tensor
        let mut global_shape = input.shape().to_vec();
        let last_dim = global_shape.len() - 1;
        global_shape[last_dim] = self.out_features;

        let partition = TensorPartition {
            split_dim: global_shape.len() - 1,
            start_idx: self.mp_context.rank() * self.out_features / self.mp_context.world_size(),
            end_idx: ((self.mp_context.rank() + 1) * self.out_features
                / self.mp_context.world_size())
            .min(self.out_features),
            num_partitions: self.mp_context.world_size(),
            partition_rank: self.mp_context.rank(),
        };

        Ok(DistributedTensor::new(
            output,
            global_shape,
            partition,
            self.mp_context.rank(),
        ))
    }

    pub fn backward(&mut self, grad_output: &DistributedTensor, input: &Tensor) -> Result<Tensor> {
        // Gradient w.r.t weight: input^T @ grad_output
        let input_ndim = input.shape().len();
        let grad_weight = input
            .transpose(input_ndim.saturating_sub(2), input_ndim.saturating_sub(1))?
            .matmul(&grad_output.local_shard)?;

        // Gradient w.r.t input: grad_output @ weight^T
        // This needs all-reduce since each device computes partial gradient
        let weight_ndim = self.weight.shape().len();
        let mut grad_input = grad_output.local_shard.matmul(
            &self
                .weight
                .transpose(weight_ndim.saturating_sub(2), weight_ndim.saturating_sub(1))?,
        )?;

        // All-reduce grad_input across devices
        self.mp_context.all_reduce(&mut grad_input)?;

        // Update local weight shard
        // In practice, this would be handled by the optimizer
        // self.weight = self.weight - learning_rate * grad_weight;

        Ok(grad_input)
    }
}

/// Row-parallel linear layer
///
/// Splits the weight matrix by rows (input dimension) across devices.
/// This is typically used for the second linear layer in MLP blocks.
pub struct RowParallelLinear {
    /// Local weight shard [in_features_per_device, out_features]
    weight: Tensor,
    /// Bias (replicated on all devices)
    bias: Option<Tensor>,
    /// Model parallel context
    mp_context: Arc<ModelParallelContext>,
    /// Total input features (across all devices)
    #[allow(dead_code)]
    in_features: usize,
    /// Output features
    _out_features: usize,
}

impl RowParallelLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        bias: bool,
        mp_context: Arc<ModelParallelContext>,
    ) -> Result<Self> {
        let world_size = mp_context.world_size();
        let rank = mp_context.rank();

        // Calculate local input features
        let in_features_per_device = in_features.div_ceil(world_size);
        let local_in_start = rank * in_features_per_device;
        let local_in_end = ((rank + 1) * in_features_per_device).min(in_features);
        let local_in_features = local_in_end - local_in_start;

        // Initialize local weight shard
        let weight = Tensor::randn(&[local_in_features, out_features])?;

        // Bias is replicated on all devices
        let bias = if bias { Some(Tensor::zeros(&[out_features])?) } else { None };

        Ok(Self {
            weight,
            bias,
            mp_context,
            in_features,
            _out_features: out_features,
        })
    }

    pub fn forward(&self, input: &DistributedTensor) -> Result<Tensor> {
        // Input is distributed: [batch_size, seq_len, local_in_features]
        // Weight: [local_in_features, out_features]
        // Local output: [batch_size, seq_len, out_features]

        let local_output = input.local_shard.matmul(&self.weight)?;

        // All-reduce to sum contributions from all devices
        let mut output = local_output;
        self.mp_context.all_reduce(&mut output)?;

        // Add bias (same on all devices)
        if let Some(bias) = &self.bias {
            output = output.add(bias)?;
        }

        Ok(output)
    }

    pub fn backward(
        &mut self,
        grad_output: &Tensor,
        input: &DistributedTensor,
    ) -> Result<DistributedTensor> {
        // Gradient w.r.t weight: input^T @ grad_output
        let input_ndim = input.local_shard.shape().len();
        let grad_weight = input
            .local_shard
            .transpose(input_ndim.saturating_sub(2), input_ndim.saturating_sub(1))?
            .matmul(grad_output)?;

        // Gradient w.r.t input: grad_output @ weight^T
        let weight_ndim = self.weight.shape().len();
        let grad_input_local = grad_output.matmul(
            &self
                .weight
                .transpose(weight_ndim.saturating_sub(2), weight_ndim.saturating_sub(1))?,
        )?;

        // Create distributed gradient tensor
        let partition = input.partition.clone();
        Ok(DistributedTensor::new(
            grad_input_local,
            input.global_shape.clone(),
            partition,
            self.mp_context.rank(),
        ))
    }
}

/// Parallel multi-head attention layer
///
/// Distributes attention heads across devices for model parallelism.
pub struct ParallelMultiHeadAttention {
    /// Number of attention heads per device
    #[allow(dead_code)]
    num_heads_per_device: usize,
    /// Total number of heads
    num_heads: usize,
    /// Head dimension
    head_dim: usize,
    /// Hidden size
    hidden_size: usize,
    /// Query projection (column parallel)
    q_proj: ColumnParallelLinear,
    /// Key projection (column parallel)
    k_proj: ColumnParallelLinear,
    /// Value projection (column parallel)
    v_proj: ColumnParallelLinear,
    /// Output projection (row parallel)
    o_proj: RowParallelLinear,
    /// Model parallel context
    mp_context: Arc<ModelParallelContext>,
}

impl ParallelMultiHeadAttention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        mp_context: Arc<ModelParallelContext>,
    ) -> Result<Self> {
        let world_size = mp_context.world_size();

        if num_heads % world_size != 0 {
            return Err(tensor_op_error(
                "ParallelMultiHeadAttention::new",
                format!(
                    "Number of heads {} must be divisible by world size {}",
                    num_heads, world_size
                ),
            ));
        }

        let num_heads_per_device = num_heads / world_size;
        let head_dim = hidden_size / num_heads;

        // Create parallel linear layers
        let q_proj =
            ColumnParallelLinear::new(hidden_size, hidden_size, false, mp_context.clone())?;

        let k_proj =
            ColumnParallelLinear::new(hidden_size, hidden_size, false, mp_context.clone())?;

        let v_proj =
            ColumnParallelLinear::new(hidden_size, hidden_size, false, mp_context.clone())?;

        let o_proj = RowParallelLinear::new(hidden_size, hidden_size, false, mp_context.clone())?;

        Ok(Self {
            num_heads_per_device,
            num_heads,
            head_dim,
            hidden_size,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            mp_context,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let batch_size = hidden_states.shape()[0];
        let seq_len = hidden_states.shape()[1];

        // Project to Q, K, V (distributed across devices)
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Get local tensor shards for processing
        let q = q.local_shard.clone();
        let k = k.local_shard.clone();
        let v = v.local_shard.clone();

        // Reshape for multi-head attention: [batch, seq_len, hidden_size] -> [batch, seq_len, num_heads_local, head_dim]
        let num_heads_local = self.num_heads / self.mp_context.world_size();
        let q = q.reshape(&[batch_size, seq_len, num_heads_local, self.head_dim])?;
        let k = k.reshape(&[batch_size, seq_len, num_heads_local, self.head_dim])?;
        let v = v.reshape(&[batch_size, seq_len, num_heads_local, self.head_dim])?;

        // Transpose for attention: [batch, seq_len, num_heads_local, head_dim] -> [batch, num_heads_local, seq_len, head_dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Compute attention scores
        let k_ndim = k.shape().len();
        let scores = q.matmul(&k.transpose(k_ndim.saturating_sub(2), k_ndim.saturating_sub(1))?)?;
        let scores = scores.scalar_mul(1.0 / (self.head_dim as f32).sqrt())?;

        // Apply attention mask if provided
        let scores = if let Some(mask) = attention_mask { scores.add(mask)? } else { scores };

        // Softmax over last dimension
        let scores_ndim = scores.shape().len();
        let attn_probs = scores.softmax((scores_ndim as i32) - 1)?;

        // Apply attention to values
        let attn_output = attn_probs.matmul(&v)?;

        // Transpose back: [batch, num_heads_local, seq_len, head_dim] -> [batch, seq_len, num_heads_local, head_dim]
        let attn_output = attn_output.transpose(1, 2)?;

        // Reshape back to original format: [batch, seq_len, num_heads_local, head_dim] -> [batch, seq_len, hidden_size_local]
        let hidden_size_local = num_heads_local * self.head_dim;
        let attn_output = attn_output.reshape(&[batch_size, seq_len, hidden_size_local])?;

        // Create distributed tensor for row-parallel output projection
        let attn_distributed = DistributedTensor::new(
            attn_output,
            vec![batch_size, seq_len, self.hidden_size],
            TensorPartition {
                split_dim: 2,
                start_idx: self.mp_context.rank() * self.hidden_size / self.mp_context.world_size(),
                end_idx: ((self.mp_context.rank() + 1) * self.hidden_size
                    / self.mp_context.world_size())
                .min(self.hidden_size),
                num_partitions: self.mp_context.world_size(),
                partition_rank: self.mp_context.rank(),
            },
            self.mp_context.rank(),
        );

        // Output projection with all-reduce
        self.o_proj.forward(&attn_distributed)
    }
}

/// Parallel MLP/FFN layer
///
/// Implements the feed-forward network with model parallelism.
pub struct ParallelMLP {
    /// First linear layer (column parallel)
    fc1: ColumnParallelLinear,
    /// Second linear layer (row parallel)
    fc2: RowParallelLinear,
    /// Activation function
    activation: ActivationType,
    /// Model parallel context
    #[allow(dead_code)]
    mp_context: Arc<ModelParallelContext>,
}

#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    Relu,
    Gelu,
    GeluNew,
    Swiglu,
}

impl ParallelMLP {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        activation: ActivationType,
        mp_context: Arc<ModelParallelContext>,
    ) -> Result<Self> {
        let fc1 =
            ColumnParallelLinear::new(hidden_size, intermediate_size, false, mp_context.clone())?;

        let fc2 =
            RowParallelLinear::new(intermediate_size, hidden_size, false, mp_context.clone())?;

        Ok(Self {
            fc1,
            fc2,
            activation,
            mp_context,
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // First linear layer (column parallel)
        let hidden = self.fc1.forward(hidden_states)?;

        // Apply activation function to the local shard
        let activated = self.apply_activation(&hidden.local_shard)?;

        // Create distributed tensor for row parallel layer
        let hidden_distributed = DistributedTensor::new(
            activated,
            hidden.global_shape.clone(),
            hidden.partition.clone(),
            hidden.device_id,
        );

        // Second linear layer (row parallel)
        self.fc2.forward(&hidden_distributed)
    }

    /// Apply activation function to tensor
    fn apply_activation(&self, tensor: &Tensor) -> Result<Tensor> {
        use crate::ops::activations::{gelu, gelu_new, relu, swiglu};

        match self.activation {
            ActivationType::Relu => Ok(relu(tensor)?),
            ActivationType::Gelu => Ok(gelu(tensor)?),
            ActivationType::GeluNew => Ok(gelu_new(tensor)?),
            ActivationType::Swiglu => {
                // SwiGLU requires splitting the input tensor and applying activation
                // For SwiGLU: SwiGLU(x) = Swish(Wx) ⊙ Vx where Swish(x) = x * sigmoid(x)
                let shape = tensor.shape();
                if shape[shape.len() - 1] % 2 != 0 {
                    return Err(tensor_op_error(
                        "ParallelMLP::apply_activation",
                        "SwiGLU requires even dimension for splitting",
                    ));
                }

                let split_size = shape[shape.len() - 1] / 2;
                let mut new_shape = shape.to_vec();
                let last_idx = new_shape.len() - 1;
                new_shape[last_idx] = split_size;

                // Split tensor into two halves along the last axis
                let last_axis = shape.len() - 1;
                let gate_tensor = tensor.slice(last_axis, 0, split_size)?;
                let up_tensor = tensor.slice(last_axis, split_size, shape[last_axis])?;

                // Apply swish to gate and element-wise multiply
                Ok(swiglu(&gate_tensor, &up_tensor)?)
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::model_parallel::{CommunicationBackend, ModelParallelConfig};
    use super::*;

    #[test]
    fn test_column_parallel_linear() {
        let config = ModelParallelConfig {
            num_devices: 2,
            device_ids: vec![0, 1],
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };

        let mp_context = Arc::new(ModelParallelContext::new(config).unwrap());
        let layer = ColumnParallelLinear::new(512, 2048, true, mp_context).unwrap();

        // Check weight dimensions
        assert_eq!(layer.weight.shape(), &[512, 1024]); // 2048 / 2 = 1024
    }

    #[test]
    fn test_parallel_attention_heads() {
        let config = ModelParallelConfig {
            num_devices: 4,
            device_ids: vec![0, 1, 2, 3],
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };

        let mp_context = Arc::new(ModelParallelContext::new(config).unwrap());
        let attn = ParallelMultiHeadAttention::new(768, 12, mp_context).unwrap();

        assert_eq!(attn.num_heads_per_device, 3); // 12 / 4 = 3
        assert_eq!(attn.head_dim, 64); // 768 / 12 = 64
    }
}
