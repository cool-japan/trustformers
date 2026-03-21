/// Mixture of Experts (MoE) infrastructure for efficient scaling
///
/// This module provides reusable MoE components that can be integrated into
/// various transformer architectures like Mixtral, GLaM, Switch Transformer, etc.
use std::collections::HashMap;
use trustformers_core::{errors::Result, layers::Linear, tensor::Tensor, traits::Layer};

/// Configuration for MoE layers
#[derive(Debug, Clone)]
pub struct MoEConfig {
    pub hidden_size: usize,
    pub num_experts: usize,
    pub num_experts_per_token: usize,
    pub expert_capacity: Option<usize>, // For capacity-based routing
    pub load_balancing_loss_coeff: f32,
    pub router_z_loss_coeff: f32,
    pub use_auxiliary_loss: bool,
    pub jitter_noise: f32, // Noise for load balancing
}

impl Default for MoEConfig {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            num_experts: 8,
            num_experts_per_token: 2,
            expert_capacity: None,
            load_balancing_loss_coeff: 0.01,
            router_z_loss_coeff: 0.001,
            use_auxiliary_loss: true,
            jitter_noise: 1e-2,
        }
    }
}

/// Expert routing statistics for load balancing
#[derive(Debug, Clone)]
pub struct RoutingStats {
    pub expert_counts: Vec<f32>,
    pub expert_weights: Vec<f32>,
    pub load_balancing_loss: f32,
    pub router_z_loss: f32,
}

/// Generic expert trait that can wrap different layer types
pub trait Expert: Layer<Input = Tensor, Output = Tensor> + Send + Sync {
    fn expert_id(&self) -> usize;
    fn capacity(&self) -> Option<usize> {
        None
    }
}

/// Basic MLP expert implementation
pub struct MLPExpert {
    id: usize,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    activation: String,
}

impl MLPExpert {
    pub fn new(
        id: usize,
        hidden_size: usize,
        intermediate_size: usize,
        activation: String,
    ) -> Result<Self> {
        let gate_proj = Linear::new(hidden_size, intermediate_size, false);
        let up_proj = Linear::new(hidden_size, intermediate_size, false);
        let down_proj = Linear::new(intermediate_size, hidden_size, false);

        Ok(Self {
            id,
            gate_proj,
            up_proj,
            down_proj,
            activation,
        })
    }

    fn apply_activation(&self, x: &Tensor) -> Result<Tensor> {
        match self.activation.as_str() {
            "silu" | "swish" => x.silu(),
            "gelu" => x.gelu(),
            "relu" => x.relu(),
            _ => Ok(x.clone()),
        }
    }
}

impl Layer for MLPExpert {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let gate = self.gate_proj.forward(input.clone())?;
        let gate_activated = self.apply_activation(&gate)?;

        let up = self.up_proj.forward(input)?;
        let gated = gate_activated.mul(&up)?;

        self.down_proj.forward(gated)
    }
}

impl Expert for MLPExpert {
    fn expert_id(&self) -> usize {
        self.id
    }
}

/// Top-K router for expert selection
pub struct TopKRouter {
    gate: Linear,
    config: MoEConfig,
}

impl TopKRouter {
    pub fn new(config: MoEConfig) -> Result<Self> {
        let gate = Linear::new(config.hidden_size, config.num_experts, false);
        Ok(Self { gate, config })
    }

    /// Route tokens to top-k experts
    pub fn route(&self, hidden_states: &Tensor) -> Result<RouterOutput> {
        let batch_size = hidden_states.shape()[0];
        let seq_len = hidden_states.shape()[1];
        let hidden_size = hidden_states.shape()[2];

        // Flatten for per-token routing
        let flattened = hidden_states.reshape(&[batch_size * seq_len, hidden_size])?;

        // Compute router logits
        let router_logits = self.gate.forward(flattened)?;

        // Add jitter noise for load balancing during training
        let router_logits = if self.config.jitter_noise > 0.0 {
            let noise = Tensor::randn_like(&router_logits)?.mul_scalar(self.config.jitter_noise)?;
            router_logits.add(&noise)?
        } else {
            router_logits
        };

        // Apply softmax to get probabilities
        let router_probs = router_logits.softmax(-1)?;

        // Select top-k experts
        let (top_k_weights, top_k_indices) = self.select_top_k(&router_probs)?;

        // Compute auxiliary losses
        let stats = self.compute_routing_stats(&router_probs, &top_k_weights, &top_k_indices)?;

        Ok(RouterOutput {
            top_k_weights,
            top_k_indices,
            router_probs,
            stats,
        })
    }

    fn select_top_k(&self, router_probs: &Tensor) -> Result<(Tensor, Tensor)> {
        let num_tokens = router_probs.shape()[0];
        let num_experts = router_probs.shape()[1];

        let mut all_weights = Vec::new();
        let mut all_indices = Vec::new();

        for token_idx in 0..num_tokens {
            // Get probabilities for this token
            let mut expert_probs: Vec<(f32, usize)> = Vec::new();
            for expert_idx in 0..num_experts {
                let prob = router_probs.get_scalar(&[token_idx, expert_idx])?;
                expert_probs.push((prob, expert_idx));
            }

            // Sort and select top-k
            expert_probs.sort_by(|a, b| b.0.partial_cmp(&a.0).expect("operation failed"));
            expert_probs.truncate(self.config.num_experts_per_token);

            // Renormalize selected probabilities
            let sum: f32 = expert_probs.iter().map(|(p, _)| p).sum();
            let norm_factor = if sum > 0.0 { 1.0 / sum } else { 1.0 };

            for (prob, expert_idx) in expert_probs {
                all_weights.push(prob * norm_factor);
                all_indices.push(expert_idx as f32);
            }
        }

        let weights_tensor = Tensor::from_vec(
            all_weights,
            &[num_tokens, self.config.num_experts_per_token],
        )?;
        let indices_tensor = Tensor::from_vec(
            all_indices,
            &[num_tokens, self.config.num_experts_per_token],
        )?;

        Ok((weights_tensor, indices_tensor))
    }

    fn compute_routing_stats(
        &self,
        router_probs: &Tensor,
        top_k_weights: &Tensor,
        top_k_indices: &Tensor,
    ) -> Result<RoutingStats> {
        let num_tokens = router_probs.shape()[0];
        let num_experts = self.config.num_experts;

        // Count tokens routed to each expert
        let mut expert_counts = vec![0.0; num_experts];
        let mut expert_weights = vec![0.0; num_experts];

        for token_idx in 0..num_tokens {
            for k in 0..self.config.num_experts_per_token {
                let expert_idx = top_k_indices.get_scalar(&[token_idx, k])? as usize;
                let weight = top_k_weights.get_scalar(&[token_idx, k])?;

                expert_counts[expert_idx] += 1.0;
                expert_weights[expert_idx] += weight;
            }
        }

        // Normalize counts and weights
        let total_tokens = num_tokens as f32;
        expert_counts.iter_mut().for_each(|c| *c /= total_tokens);
        expert_weights.iter_mut().for_each(|w| *w /= total_tokens);

        // Compute load balancing loss (encourages uniform expert usage)
        let _mean_count = 1.0 / num_experts as f32;
        let load_balancing_loss: f32 = expert_counts
            .iter()
            .zip(expert_weights.iter())
            .map(|(count, weight)| count * weight)
            .sum::<f32>()
            * num_experts as f32
            - 1.0;

        // Compute router z-loss (encourages sparsity)
        let router_z_loss = router_probs
            .pow(2.0)?
            .sum(Some(vec![router_probs.shape().len() - 1]), false)?
            .mean()?
            .get_scalar(&[])?;

        Ok(RoutingStats {
            expert_counts,
            expert_weights,
            load_balancing_loss,
            router_z_loss,
        })
    }
}

/// Output from the router
pub struct RouterOutput {
    pub top_k_weights: Tensor,
    pub top_k_indices: Tensor,
    pub router_probs: Tensor,
    pub stats: RoutingStats,
}

/// Sparse Mixture of Experts layer
pub struct SparseMoE<E: Expert> {
    experts: Vec<E>,
    router: TopKRouter,
    config: MoEConfig,
}

impl<E: Expert> SparseMoE<E> {
    pub fn new(experts: Vec<E>, config: MoEConfig) -> Result<Self> {
        let router = TopKRouter::new(config.clone())?;
        Ok(Self {
            experts,
            router,
            config,
        })
    }

    /// Get the number of experts
    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }

    /// Get routing statistics from the last forward pass
    pub fn last_routing_stats(&self) -> Option<&RoutingStats> {
        // This would need to be stored from the last forward pass
        None
    }
}

impl<E: Expert> Layer for SparseMoE<E> {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        let hidden_size = input.shape()[2];

        // Route tokens to experts
        let router_output = self.router.route(&input)?;

        // Flatten input for processing
        let flattened_input = input.reshape(&[batch_size * seq_len, hidden_size])?;
        let num_tokens = flattened_input.shape()[0];

        // Initialize output
        let mut output = Tensor::zeros(&[num_tokens, hidden_size])?;

        // Process each token
        for token_idx in 0..num_tokens {
            // Proper tensor slicing: select single token input
            let token_input =
                flattened_input.slice_multi(&[(token_idx, token_idx + 1), (0, hidden_size)])?;

            // Combine outputs from selected experts
            let mut token_output = Tensor::zeros(&[1, hidden_size])?;
            for k in 0..self.config.num_experts_per_token {
                let expert_idx = router_output.top_k_indices.get_scalar(&[token_idx, k])? as usize;
                let weight = router_output.top_k_weights.get_scalar(&[token_idx, k])?;

                // Get expert output
                let expert_output = self.experts[expert_idx].forward(token_input.clone())?;
                let weighted_output = expert_output.mul_scalar(weight)?;

                // Accumulate expert outputs for this token
                token_output = token_output.add(&weighted_output)?;
            }

            // Set the token output in the final output tensor
            // Use slice and add for proper accumulation per token
            let token_output_slice =
                output.slice_multi(&[(token_idx, token_idx + 1), (0, hidden_size)])?;
            let updated_slice = token_output_slice.add(&token_output)?;

            // For now, we'll use a workaround since set_slice is not available
            // This approach maintains per-token processing but requires reconstruction
            if token_idx == 0 {
                output = updated_slice.clone();
            } else {
                // Concatenate along the first dimension
                let current_tokens = output.slice_multi(&[(0, token_idx), (0, hidden_size)])?;
                let remaining_shape = if token_idx + 1 < num_tokens {
                    Some(output.slice_multi(&[(token_idx + 1, num_tokens), (0, hidden_size)])?)
                } else {
                    None
                };

                // Reconstruct output tensor with updated token
                output = if let Some(remaining) = remaining_shape {
                    Tensor::concat(&[current_tokens, updated_slice, remaining], 0)?
                } else {
                    Tensor::concat(&[current_tokens, updated_slice], 0)?
                };
            }
        }

        // Reshape back to original dimensions
        output.reshape(&[batch_size, seq_len, hidden_size])
    }
}

/// Switch Transformer style MoE (uses only top-1 expert per token)
pub type SwitchMoE<E> = SparseMoE<E>;

/// Helper function to create Switch Transformer configuration
pub fn switch_config(hidden_size: usize, num_experts: usize) -> MoEConfig {
    MoEConfig {
        hidden_size,
        num_experts,
        num_experts_per_token: 1, // Switch uses top-1
        ..Default::default()
    }
}

/// Helper function to create GLaM-style configuration
pub fn glam_config(hidden_size: usize, num_experts: usize) -> MoEConfig {
    MoEConfig {
        hidden_size,
        num_experts,
        num_experts_per_token: 2, // GLaM uses top-2
        ..Default::default()
    }
}

/// Expert parallel processing for distributed training
pub struct ExpertParallel<E: Expert> {
    local_experts: Vec<E>,
    expert_mapping: HashMap<usize, usize>, // global_id -> local_id
    #[allow(dead_code)]
    rank: usize,
    #[allow(dead_code)]
    world_size: usize,
}

impl<E: Expert> ExpertParallel<E> {
    pub fn new(experts: Vec<E>, rank: usize, world_size: usize) -> Self {
        let mut expert_mapping = HashMap::new();
        for (local_id, expert) in experts.iter().enumerate() {
            expert_mapping.insert(expert.expert_id(), local_id);
        }

        Self {
            local_experts: experts,
            expert_mapping,
            rank,
            world_size,
        }
    }

    /// Check if an expert is available locally
    pub fn has_expert(&self, expert_id: usize) -> bool {
        self.expert_mapping.contains_key(&expert_id)
    }

    /// Forward pass for local experts only
    pub fn forward_local(&self, expert_id: usize, input: &Tensor) -> Result<Option<Tensor>> {
        if let Some(&local_id) = self.expert_mapping.get(&expert_id) {
            let output = self.local_experts[local_id].forward(input.clone())?;
            Ok(Some(output))
        } else {
            Ok(None)
        }
    }
}
