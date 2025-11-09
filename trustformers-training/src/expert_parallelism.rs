use crate::distributed::ProcessGroup;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use trustformers_core::tensor::Tensor;

/// Expert Parallelism Configuration for Mixture of Experts (MoE) models
///
/// Expert parallelism distributes experts across different devices/processes,
/// enabling scaling of MoE models with efficient expert routing and load balancing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertParallelismConfig {
    /// Number of experts in the MoE layer
    pub num_experts: usize,
    /// Number of experts per device/process
    pub experts_per_device: usize,
    /// Number of devices/processes for expert parallelism
    pub expert_parallel_size: usize,
    /// Top-k routing for expert selection
    pub top_k: usize,
    /// Load balancing strategy
    pub load_balancing: LoadBalancingStrategy,
    /// Expert routing strategy
    pub routing_strategy: ExpertRoutingStrategy,
    /// Whether to use expert capacity limiting
    pub capacity_factor: f32,
    /// Drop tokens when capacity is exceeded
    pub drop_tokens: bool,
    /// Use auxiliary load balancing loss
    pub use_auxiliary_loss: bool,
    /// Auxiliary loss weight
    pub auxiliary_loss_weight: f32,
    /// Expert communication pattern
    pub communication_pattern: ExpertCommunicationPattern,
}

impl Default for ExpertParallelismConfig {
    fn default() -> Self {
        Self {
            num_experts: 8,
            experts_per_device: 2,
            expert_parallel_size: 4,
            top_k: 2,
            load_balancing: LoadBalancingStrategy::TokenChoiceBased,
            routing_strategy: ExpertRoutingStrategy::LearnedGating,
            capacity_factor: 1.25,
            drop_tokens: false,
            use_auxiliary_loss: true,
            auxiliary_loss_weight: 0.01,
            communication_pattern: ExpertCommunicationPattern::AllToAll,
        }
    }
}

/// Load balancing strategies for expert utilization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Balance load based on token choice
    TokenChoiceBased,
    /// Balance load based on expert choice
    ExpertChoiceBased,
    /// Dynamic load balancing
    Dynamic,
    /// Round-robin assignment
    RoundRobin,
    /// Load-aware routing
    LoadAware,
}

/// Expert routing strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpertRoutingStrategy {
    /// Learned gating network
    LearnedGating,
    /// Hash-based routing
    HashBased,
    /// Random routing
    Random,
    /// Load-based routing
    LoadBased,
    /// Similarity-based routing
    SimilarityBased,
}

/// Communication patterns for expert parallelism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExpertCommunicationPattern {
    /// All-to-all communication
    AllToAll,
    /// Point-to-point communication
    PointToPoint,
    /// Hierarchical communication
    Hierarchical,
    /// Ring-based communication
    Ring,
}

/// Expert assignment and routing information
#[derive(Debug, Clone)]
pub struct ExpertAssignment {
    /// Expert ID
    pub expert_id: usize,
    /// Device/process rank where expert is located
    pub device_rank: usize,
    /// Local expert index on the device
    pub local_expert_id: usize,
    /// Load weight for this expert
    pub load_weight: f32,
}

/// Token routing information
#[derive(Debug, Clone)]
pub struct TokenRouting {
    /// Token indices
    pub token_indices: Vec<usize>,
    /// Expert assignments for each token
    pub expert_assignments: Vec<Vec<(usize, f32)>>, // (expert_id, weight)
    /// Communication destinations
    pub destinations: HashMap<usize, Vec<usize>>, // device_rank -> token_indices
    /// Capacity constraints
    pub capacity_usage: HashMap<usize, usize>, // expert_id -> current_tokens
}

/// Expert parallelism coordinator
#[allow(dead_code)]
pub struct ExpertParallelism {
    config: ExpertParallelismConfig,
    #[allow(dead_code)]
    global_rank: usize,
    world_size: usize,

    // Expert assignment mapping
    expert_assignments: Vec<ExpertAssignment>,
    local_experts: Vec<usize>, // Expert IDs local to this device

    // Process groups
    expert_group: Arc<dyn ProcessGroup>,

    // Load balancing state
    load_balancing_state: Arc<RwLock<LoadBalancingState>>,

    // Communication statistics
    communication_stats: Arc<Mutex<ExpertCommunicationStats>>,

    // Routing cache for efficiency
    routing_cache: Arc<Mutex<HashMap<String, TokenRouting>>>,
}

/// Load balancing state tracking
#[derive(Debug, Default)]
#[allow(dead_code)]
struct LoadBalancingState {
    expert_loads: HashMap<usize, f32>,
    #[allow(dead_code)]
    expert_utilization: HashMap<usize, f32>,
    token_distribution: HashMap<usize, usize>,
    imbalance_score: f32,
    last_rebalance_time: Option<Instant>,
}

/// Communication statistics for expert parallelism
#[derive(Debug, Default)]
#[allow(dead_code)]
struct ExpertCommunicationStats {
    all_to_all_time: Duration,
    #[allow(dead_code)]
    point_to_point_time: Duration,
    total_tokens_routed: u64,
    expert_load_variance: f32,
    communication_efficiency: f32,
    routing_overhead: Duration,
}

impl ExpertParallelism {
    /// Create a new expert parallelism coordinator
    pub fn new(
        config: ExpertParallelismConfig,
        global_rank: usize,
        world_size: usize,
        expert_group: Arc<dyn ProcessGroup>,
    ) -> Result<Self> {
        // Validate configuration
        if config.num_experts % config.expert_parallel_size != 0 {
            return Err(anyhow!(
                "Number of experts ({}) must be divisible by expert parallel size ({})",
                config.num_experts,
                config.expert_parallel_size
            ));
        }

        if config.experts_per_device * config.expert_parallel_size != config.num_experts {
            return Err(anyhow!(
                "Expert assignment mismatch: experts_per_device ({}) * expert_parallel_size ({}) != num_experts ({})",
                config.experts_per_device, config.expert_parallel_size, config.num_experts
            ));
        }

        // Create expert assignments
        let expert_assignments = Self::create_expert_assignments(&config, world_size)?;
        let local_experts = Self::get_local_experts(&expert_assignments, global_rank);

        Ok(Self {
            config,
            global_rank,
            world_size,
            expert_assignments,
            local_experts,
            expert_group,
            load_balancing_state: Arc::new(RwLock::new(LoadBalancingState::default())),
            communication_stats: Arc::new(Mutex::new(ExpertCommunicationStats::default())),
            routing_cache: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Create expert assignments across devices
    fn create_expert_assignments(
        config: &ExpertParallelismConfig,
        _world_size: usize,
    ) -> Result<Vec<ExpertAssignment>> {
        let mut assignments = Vec::new();

        for expert_id in 0..config.num_experts {
            let device_rank = expert_id / config.experts_per_device;
            let local_expert_id = expert_id % config.experts_per_device;

            assignments.push(ExpertAssignment {
                expert_id,
                device_rank,
                local_expert_id,
                load_weight: 1.0, // Initialize with equal weights
            });
        }

        Ok(assignments)
    }

    /// Get local expert IDs for a given device rank
    fn get_local_experts(assignments: &[ExpertAssignment], device_rank: usize) -> Vec<usize> {
        assignments
            .iter()
            .filter(|assignment| assignment.device_rank == device_rank)
            .map(|assignment| assignment.expert_id)
            .collect()
    }

    /// Route tokens to experts based on gating scores
    pub fn route_tokens(&self, tokens: &Tensor, gating_scores: &Tensor) -> Result<TokenRouting> {
        let start_time = Instant::now();

        // Implement token routing logic based on strategy
        let routing = match self.config.routing_strategy {
            ExpertRoutingStrategy::LearnedGating => {
                self.learned_gating_routing(tokens, gating_scores)?
            },
            ExpertRoutingStrategy::HashBased => self.hash_based_routing(tokens)?,
            ExpertRoutingStrategy::Random => self.random_routing(tokens)?,
            ExpertRoutingStrategy::LoadBased => self.load_based_routing(tokens, gating_scores)?,
            ExpertRoutingStrategy::SimilarityBased => {
                self.similarity_based_routing(tokens, gating_scores)?
            },
        };

        // Update statistics
        {
            let mut stats = self.communication_stats.lock().unwrap();
            stats.routing_overhead += start_time.elapsed();
            stats.total_tokens_routed += tokens.shape()[0] as u64;
        }

        Ok(routing)
    }

    /// Learned gating-based token routing
    fn learned_gating_routing(
        &self,
        tokens: &Tensor,
        _gating_scores: &Tensor,
    ) -> Result<TokenRouting> {
        let batch_size = tokens.shape()[0];
        let num_tokens = batch_size;

        let mut token_routing = TokenRouting {
            token_indices: (0..num_tokens).collect(),
            expert_assignments: Vec::new(),
            destinations: HashMap::new(),
            capacity_usage: HashMap::new(),
        };

        // Get top-k experts for each token
        for token_idx in 0..num_tokens {
            let mut expert_scores = Vec::new();

            // Extract scores for this token (simplified - in practice would use tensor operations)
            for expert_id in 0..self.config.num_experts {
                let score = 1.0 / (expert_id + 1) as f32; // Placeholder score calculation
                expert_scores.push((expert_id, score));
            }

            // Sort by score and take top-k
            expert_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            expert_scores.truncate(self.config.top_k);

            // Normalize weights
            let total_weight: f32 = expert_scores.iter().map(|(_, w)| w).sum();
            let normalized_assignments: Vec<(usize, f32)> = expert_scores
                .iter()
                .map(|(expert_id, weight)| (*expert_id, weight / total_weight))
                .collect();

            token_routing.expert_assignments.push(normalized_assignments.clone());

            // Update destinations mapping
            for (expert_id, _) in normalized_assignments {
                let device_rank = self.expert_assignments[expert_id].device_rank;
                token_routing.destinations.entry(device_rank).or_default().push(token_idx);
            }
        }

        Ok(token_routing)
    }

    /// Hash-based token routing for deterministic assignment
    fn hash_based_routing(&self, tokens: &Tensor) -> Result<TokenRouting> {
        let batch_size = tokens.shape()[0];
        let mut token_routing = TokenRouting {
            token_indices: (0..batch_size).collect(),
            expert_assignments: Vec::new(),
            destinations: HashMap::new(),
            capacity_usage: HashMap::new(),
        };

        for token_idx in 0..batch_size {
            // Simple hash-based assignment (in practice, would use token content)
            let expert_id = token_idx % self.config.num_experts;
            let device_rank = self.expert_assignments[expert_id].device_rank;

            token_routing.expert_assignments.push(vec![(expert_id, 1.0)]);
            token_routing.destinations.entry(device_rank).or_default().push(token_idx);
        }

        Ok(token_routing)
    }

    /// Random token routing
    fn random_routing(&self, tokens: &Tensor) -> Result<TokenRouting> {
        let batch_size = tokens.shape()[0];
        let mut token_routing = TokenRouting {
            token_indices: (0..batch_size).collect(),
            expert_assignments: Vec::new(),
            destinations: HashMap::new(),
            capacity_usage: HashMap::new(),
        };

        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        for token_idx in 0..batch_size {
            let mut hasher = DefaultHasher::new();
            token_idx.hash(&mut hasher);
            let expert_id = (hasher.finish() as usize) % self.config.num_experts;
            let device_rank = self.expert_assignments[expert_id].device_rank;

            token_routing.expert_assignments.push(vec![(expert_id, 1.0)]);
            token_routing.destinations.entry(device_rank).or_default().push(token_idx);
        }

        Ok(token_routing)
    }

    /// Load-based token routing
    fn load_based_routing(&self, tokens: &Tensor, _gating_scores: &Tensor) -> Result<TokenRouting> {
        let batch_size = tokens.shape()[0];
        let mut token_routing = TokenRouting {
            token_indices: (0..batch_size).collect(),
            expert_assignments: Vec::new(),
            destinations: HashMap::new(),
            capacity_usage: HashMap::new(),
        };

        let load_state = self.load_balancing_state.read().unwrap();

        for token_idx in 0..batch_size {
            // Find least loaded expert
            let mut min_load = f32::INFINITY;
            let mut selected_expert = 0;

            for expert_id in 0..self.config.num_experts {
                let load = load_state.expert_loads.get(&expert_id).unwrap_or(&0.0);
                if *load < min_load {
                    min_load = *load;
                    selected_expert = expert_id;
                }
            }

            let device_rank = self.expert_assignments[selected_expert].device_rank;
            token_routing.expert_assignments.push(vec![(selected_expert, 1.0)]);
            token_routing.destinations.entry(device_rank).or_default().push(token_idx);
        }

        Ok(token_routing)
    }

    /// Similarity-based token routing
    fn similarity_based_routing(
        &self,
        tokens: &Tensor,
        gating_scores: &Tensor,
    ) -> Result<TokenRouting> {
        // For now, fall back to learned gating (similarity requires embedding analysis)
        self.learned_gating_routing(tokens, gating_scores)
    }

    /// Perform all-to-all communication for expert parallelism
    pub fn all_to_all_communication(
        &self,
        local_tokens: &Tensor,
        routing: &TokenRouting,
    ) -> Result<HashMap<usize, Tensor>> {
        let start_time = Instant::now();

        // Simulate all-to-all communication
        // In practice, this would involve actual tensor communication
        let mut expert_inputs = HashMap::new();

        for expert_id in &self.local_experts {
            // Collect tokens assigned to this expert
            let mut expert_tokens = Vec::new();

            for (token_idx, assignments) in routing.expert_assignments.iter().enumerate() {
                for (assigned_expert_id, weight) in assignments {
                    if *assigned_expert_id == *expert_id && *weight > 0.0 {
                        // In practice, would extract actual token data
                        expert_tokens.push(token_idx);
                    }
                }
            }

            // Create tensor for this expert (simplified)
            if !expert_tokens.is_empty() {
                let expert_tensor = Tensor::zeros(&[expert_tokens.len(), local_tokens.shape()[1]])?;
                expert_inputs.insert(*expert_id, expert_tensor);
            }
        }

        // Update communication statistics
        {
            let mut stats = self.communication_stats.lock().unwrap();
            stats.all_to_all_time += start_time.elapsed();
        }

        Ok(expert_inputs)
    }

    /// Update load balancing state
    pub fn update_load_balancing(&self, expert_outputs: &HashMap<usize, Tensor>) -> Result<()> {
        let mut load_state = self.load_balancing_state.write().unwrap();

        // Update expert loads based on output sizes
        for (expert_id, output) in expert_outputs {
            let load = output.shape()[0] as f32; // Number of tokens processed
            load_state.expert_loads.insert(*expert_id, load);
        }

        // Calculate utilization and imbalance
        let total_load: f32 = load_state.expert_loads.values().sum();
        let avg_load = total_load / self.config.num_experts as f32;

        let mut variance = 0.0;
        for load in load_state.expert_loads.values() {
            variance += (load - avg_load).powi(2);
        }
        variance /= self.config.num_experts as f32;

        load_state.imbalance_score = variance.sqrt() / avg_load.max(1e-6);
        load_state.last_rebalance_time = Some(Instant::now());

        Ok(())
    }

    /// Get load balancing statistics
    pub fn get_load_balancing_stats(&self) -> LoadBalancingStats {
        let load_state = self.load_balancing_state.read().unwrap();
        let comm_stats = self.communication_stats.lock().unwrap();

        LoadBalancingStats {
            expert_loads: load_state.expert_loads.clone(),
            imbalance_score: load_state.imbalance_score,
            total_tokens_routed: comm_stats.total_tokens_routed,
            communication_efficiency: comm_stats.communication_efficiency,
            routing_overhead: comm_stats.routing_overhead,
        }
    }

    /// Get local expert IDs
    pub fn local_experts(&self) -> &[usize] {
        &self.local_experts
    }

    /// Get expert assignment for a given expert ID
    pub fn get_expert_assignment(&self, expert_id: usize) -> Option<&ExpertAssignment> {
        self.expert_assignments.get(expert_id)
    }

    /// Get configuration
    pub fn config(&self) -> &ExpertParallelismConfig {
        &self.config
    }
}

/// Load balancing statistics
#[derive(Debug, Clone)]
pub struct LoadBalancingStats {
    pub expert_loads: HashMap<usize, f32>,
    pub imbalance_score: f32,
    pub total_tokens_routed: u64,
    pub communication_efficiency: f32,
    pub routing_overhead: Duration,
}

/// Expert parallelism utilities
pub mod utils {
    use super::*;

    /// Calculate optimal expert parallelism configuration
    pub fn calculate_optimal_expert_config(
        num_experts: usize,
        world_size: usize,
        memory_per_expert_mb: usize,
        available_memory_mb: usize,
    ) -> Result<ExpertParallelismConfig> {
        let experts_per_device = std::cmp::min(
            available_memory_mb / memory_per_expert_mb,
            num_experts / world_size,
        );

        if experts_per_device == 0 {
            return Err(anyhow!("Insufficient memory for expert parallelism"));
        }

        let expert_parallel_size = (num_experts + experts_per_device - 1) / experts_per_device;

        Ok(ExpertParallelismConfig {
            num_experts,
            experts_per_device,
            expert_parallel_size,
            ..Default::default()
        })
    }

    /// Estimate communication cost for expert parallelism
    pub fn estimate_communication_cost(
        config: &ExpertParallelismConfig,
        batch_size: usize,
        sequence_length: usize,
        hidden_size: usize,
    ) -> f32 {
        let total_tokens = batch_size * sequence_length;
        let tokens_per_expert = total_tokens / config.num_experts;
        let communication_volume = tokens_per_expert * hidden_size * 4; // 4 bytes per float

        // Simplified cost model
        communication_volume as f32 / (1024.0 * 1024.0) // Convert to MB
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::SimulatedProcessGroup;
    use std::sync::Arc;

    #[test]
    fn test_expert_parallelism_config() {
        let config = ExpertParallelismConfig::default();
        assert_eq!(config.num_experts, 8);
        assert_eq!(config.expert_parallel_size, 4);
        assert_eq!(config.experts_per_device, 2);
    }

    #[test]
    fn test_expert_assignment_creation() {
        let config = ExpertParallelismConfig {
            num_experts: 8,
            experts_per_device: 2,
            expert_parallel_size: 4,
            ..Default::default()
        };

        let assignments = ExpertParallelism::create_expert_assignments(&config, 4).unwrap();
        assert_eq!(assignments.len(), 8);

        // Check that experts are distributed correctly
        for (i, assignment) in assignments.iter().enumerate() {
            assert_eq!(assignment.expert_id, i);
            assert_eq!(assignment.device_rank, i / 2);
            assert_eq!(assignment.local_expert_id, i % 2);
        }
    }

    #[test]
    fn test_local_experts() {
        let config = ExpertParallelismConfig {
            num_experts: 8,
            experts_per_device: 2,
            expert_parallel_size: 4,
            ..Default::default()
        };

        let assignments = ExpertParallelism::create_expert_assignments(&config, 4).unwrap();
        let local_experts = ExpertParallelism::get_local_experts(&assignments, 1);

        assert_eq!(local_experts, vec![2, 3]);
    }

    #[test]
    fn test_expert_parallelism_creation() {
        let config = ExpertParallelismConfig {
            num_experts: 8,
            experts_per_device: 2,
            expert_parallel_size: 4,
            ..Default::default()
        };

        let process_group = Arc::new(SimulatedProcessGroup::new(0, 4));
        let expert_parallelism = ExpertParallelism::new(config, 0, 4, process_group);

        assert!(expert_parallelism.is_ok());
        let ep = expert_parallelism.unwrap();
        assert_eq!(ep.local_experts(), &[0, 1]);
    }

    #[test]
    fn test_hash_based_routing() {
        let config = ExpertParallelismConfig {
            num_experts: 4,
            experts_per_device: 1,
            expert_parallel_size: 4,
            routing_strategy: ExpertRoutingStrategy::HashBased,
            ..Default::default()
        };

        let process_group = Arc::new(SimulatedProcessGroup::new(0, 4));
        let expert_parallelism = ExpertParallelism::new(config, 0, 4, process_group).unwrap();

        let tokens = Tensor::zeros(&[8, 16]).unwrap();
        let routing = expert_parallelism.hash_based_routing(&tokens).unwrap();

        assert_eq!(routing.token_indices.len(), 8);
        assert_eq!(routing.expert_assignments.len(), 8);
    }

    #[test]
    fn test_load_balancing_update() {
        let config = ExpertParallelismConfig::default();
        let process_group = Arc::new(SimulatedProcessGroup::new(0, 4));
        let expert_parallelism = ExpertParallelism::new(config, 0, 4, process_group).unwrap();

        let mut expert_outputs = HashMap::new();
        expert_outputs.insert(0, Tensor::zeros(&[10, 16]).unwrap());
        expert_outputs.insert(1, Tensor::zeros(&[15, 16]).unwrap());

        let result = expert_parallelism.update_load_balancing(&expert_outputs);
        assert!(result.is_ok());

        let stats = expert_parallelism.get_load_balancing_stats();
        assert!(stats.expert_loads.contains_key(&0));
        assert!(stats.expert_loads.contains_key(&1));
    }

    #[test]
    fn test_optimal_expert_config_calculation() {
        let config = utils::calculate_optimal_expert_config(16, 8, 1000, 4000).unwrap();
        assert!(config.experts_per_device <= 4);
        assert!(config.expert_parallel_size > 0);
    }

    #[test]
    fn test_communication_cost_estimation() {
        let config = ExpertParallelismConfig::default();
        let cost = utils::estimate_communication_cost(&config, 32, 512, 768);
        assert!(cost > 0.0);
    }
}
