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

/// Cosine similarity between two equally sized vectors.
///
/// Returns `0.0` when either vector has zero norm, which keeps a degenerate
/// prototype from dominating the ranking.
fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;
    for (a, b) in left.iter().zip(right) {
        dot += a * b;
        left_norm += a * a;
        right_norm += b * b;
    }
    let denominator = left_norm.sqrt() * right_norm.sqrt();
    if denominator > 0.0 {
        (dot / denominator).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Expert parallelism coordinator
pub struct ExpertParallelism {
    config: ExpertParallelismConfig,
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
struct LoadBalancingState {
    expert_loads: HashMap<usize, f32>,
    expert_utilization: HashMap<usize, f32>,
    token_distribution: HashMap<usize, usize>,
    imbalance_score: f32,
    last_rebalance_time: Option<Instant>,
}

/// Communication statistics for expert parallelism
#[derive(Debug, Default)]
struct ExpertCommunicationStats {
    all_to_all_time: Duration,
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
        if !config.num_experts.is_multiple_of(config.expert_parallel_size) {
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
            let mut stats =
                self.communication_stats.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            stats.routing_overhead += start_time.elapsed();
            stats.total_tokens_routed += tokens.shape()[0] as u64;
        }

        Ok(routing)
    }

    /// Read the `[num_tokens, num_experts]` gating logits out of `gating_scores`.
    ///
    /// The tensor must have `num_experts` as its last dimension; the leading
    /// dimensions are flattened into the token axis. The token count implied by
    /// the gating tensor must match `tokens`.
    fn gating_logits(&self, tokens: &Tensor, gating_scores: &Tensor) -> Result<Vec<Vec<f32>>> {
        let num_experts = self.config.num_experts;
        let gating_shape = gating_scores.shape();

        let last_dim = *gating_shape.last().ok_or_else(|| {
            anyhow!("gating_scores must have at least one dimension, got a scalar")
        })?;
        if last_dim != num_experts {
            return Err(anyhow!(
                "gating_scores last dimension is {} but the layer has {} experts",
                last_dim,
                num_experts
            ));
        }

        let values = gating_scores.to_vec_f32()?;
        if !values.len().is_multiple_of(num_experts) {
            return Err(anyhow!(
                "gating_scores holds {} values, which is not a multiple of {} experts",
                values.len(),
                num_experts
            ));
        }
        let num_gating_rows = values.len() / num_experts;

        let token_shape = tokens.shape();
        let num_tokens = *token_shape
            .first()
            .ok_or_else(|| anyhow!("tokens must have at least one dimension, got a scalar"))?;
        if num_gating_rows != num_tokens {
            return Err(anyhow!(
                "gating_scores describes {} tokens but the token tensor has {}",
                num_gating_rows,
                num_tokens
            ));
        }

        Ok(values.chunks_exact(num_experts).map(<[f32]>::to_vec).collect())
    }

    /// Numerically stable softmax over one token's expert logits.
    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !max_logit.is_finite() {
            // All-infinite/NaN row: fall back to a uniform distribution rather
            // than propagating NaN into the router.
            let uniform = 1.0 / logits.len().max(1) as f32;
            return vec![uniform; logits.len()];
        }
        let mut exponentials: Vec<f32> =
            logits.iter().map(|logit| (logit - max_logit).exp()).collect();
        let total: f32 = exponentials.iter().sum();
        if total > 0.0 {
            for value in exponentials.iter_mut() {
                *value /= total;
            }
        } else {
            let uniform = 1.0 / logits.len().max(1) as f32;
            exponentials.iter_mut().for_each(|value| *value = uniform);
        }
        exponentials
    }

    /// Select the `top_k` highest-probability experts for one token and
    /// renormalize their probabilities so they sum to one.
    ///
    /// Ties are broken by ascending expert id, making the routing
    /// deterministic.
    fn top_k_from_probabilities(&self, probabilities: &[f32]) -> Vec<(usize, f32)> {
        let top_k = self.config.top_k.clamp(1, probabilities.len().max(1));

        let mut ranked: Vec<(usize, f32)> = probabilities.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
        });
        ranked.truncate(top_k);

        let total: f32 = ranked.iter().map(|(_, weight)| *weight).sum();
        if total > 0.0 {
            ranked.iter().map(|(expert, weight)| (*expert, weight / total)).collect()
        } else {
            // Degenerate gating row: split the token evenly across the selected
            // experts instead of dropping it.
            let uniform = 1.0 / ranked.len().max(1) as f32;
            ranked.iter().map(|(expert, _)| (*expert, uniform)).collect()
        }
    }

    /// Learned-gating token routing: softmax over the per-token expert logits,
    /// then top-k selection with renormalized weights.
    ///
    /// This is the standard Switch/GShard router. The gating tensor is read for
    /// real; two tokens with different gating rows receive different expert
    /// assignments.
    fn learned_gating_routing(
        &self,
        tokens: &Tensor,
        gating_scores: &Tensor,
    ) -> Result<TokenRouting> {
        let logits = self.gating_logits(tokens, gating_scores)?;
        let num_tokens = logits.len();

        let mut token_routing = TokenRouting {
            token_indices: (0..num_tokens).collect(),
            expert_assignments: Vec::with_capacity(num_tokens),
            destinations: HashMap::new(),
            capacity_usage: HashMap::new(),
        };

        let capacity = self.expert_capacity(num_tokens);

        for (token_idx, row) in logits.iter().enumerate() {
            let probabilities = Self::softmax(row);
            let selected = self.top_k_from_probabilities(&probabilities);

            let mut accepted: Vec<(usize, f32)> = Vec::with_capacity(selected.len());
            for (expert_id, weight) in selected {
                let used = token_routing.capacity_usage.entry(expert_id).or_insert(0);
                if self.config.drop_tokens && *used >= capacity {
                    // Expert is at capacity and dropping is enabled: this token
                    // is not routed to it.
                    continue;
                }
                *used += 1;
                accepted.push((expert_id, weight));
            }

            // Renormalize after any capacity-driven drops so the surviving
            // weights still form a convex combination.
            let total: f32 = accepted.iter().map(|(_, weight)| *weight).sum();
            if total > 0.0 {
                for (_, weight) in accepted.iter_mut() {
                    *weight /= total;
                }
            }

            for (expert_id, _) in &accepted {
                let device_rank = self.expert_assignments[*expert_id].device_rank;
                token_routing.destinations.entry(device_rank).or_default().push(token_idx);
            }
            token_routing.expert_assignments.push(accepted);
        }

        Ok(token_routing)
    }

    /// Per-expert token capacity implied by `capacity_factor`.
    fn expert_capacity(&self, num_tokens: usize) -> usize {
        let ideal =
            (num_tokens as f32 * self.config.top_k as f32) / self.config.num_experts.max(1) as f32;
        ((ideal * self.config.capacity_factor).ceil() as usize).max(1)
    }

    /// Flatten `tokens` into one feature vector per token.
    ///
    /// The leading dimension is the token axis; every trailing dimension is
    /// flattened into the feature axis.
    fn token_embeddings(&self, tokens: &Tensor) -> Result<Vec<Vec<f32>>> {
        let shape = tokens.shape();
        let num_tokens = *shape
            .first()
            .ok_or_else(|| anyhow!("tokens must have at least one dimension, got a scalar"))?;
        if num_tokens == 0 {
            return Ok(Vec::new());
        }

        let values = tokens.to_vec_f32()?;
        if !values.len().is_multiple_of(num_tokens) {
            return Err(anyhow!(
                "token tensor holds {} values which is not a multiple of {} tokens",
                values.len(),
                num_tokens
            ));
        }
        let features = values.len() / num_tokens;
        Ok(values.chunks_exact(features).map(<[f32]>::to_vec).collect())
    }

    /// Hash-based token routing.
    ///
    /// The expert is derived from the token's **content** (the bit patterns of
    /// its features), so identical tokens always land on the same expert and
    /// different tokens are spread deterministically. This is the routing used
    /// when reproducibility matters more than gating quality.
    fn hash_based_routing(&self, tokens: &Tensor) -> Result<TokenRouting> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let embeddings = self.token_embeddings(tokens)?;
        let mut token_routing = TokenRouting {
            token_indices: (0..embeddings.len()).collect(),
            expert_assignments: Vec::with_capacity(embeddings.len()),
            destinations: HashMap::new(),
            capacity_usage: HashMap::new(),
        };

        for (token_idx, embedding) in embeddings.iter().enumerate() {
            let mut hasher = DefaultHasher::new();
            for value in embedding {
                // Hash the canonical bit pattern so -0.0 and 0.0 agree.
                let canonical = if *value == 0.0 { 0.0f32 } else { *value };
                canonical.to_bits().hash(&mut hasher);
            }
            let expert_id = (hasher.finish() as usize) % self.config.num_experts;
            let device_rank = self.expert_assignments[expert_id].device_rank;

            *token_routing.capacity_usage.entry(expert_id).or_insert(0) += 1;
            token_routing.expert_assignments.push(vec![(expert_id, 1.0)]);
            token_routing.destinations.entry(device_rank).or_default().push(token_idx);
        }

        Ok(token_routing)
    }

    /// Pseudo-random token routing.
    ///
    /// The assignment is drawn from a deterministic hash of the token index, so
    /// it is uniform across experts yet reproducible across runs and ranks —
    /// a genuine requirement for SPMD training, where every rank must derive the
    /// same routing.
    fn random_routing(&self, tokens: &Tensor) -> Result<TokenRouting> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let shape = tokens.shape();
        let batch_size = *shape
            .first()
            .ok_or_else(|| anyhow!("tokens must have at least one dimension, got a scalar"))?;
        let mut token_routing = TokenRouting {
            token_indices: (0..batch_size).collect(),
            expert_assignments: Vec::with_capacity(batch_size),
            destinations: HashMap::new(),
            capacity_usage: HashMap::new(),
        };

        for token_idx in 0..batch_size {
            let mut hasher = DefaultHasher::new();
            // Mixing in a fixed salt decorrelates this from `hash_based_routing`.
            0x9E37_79B9_7F4A_7C15u64.hash(&mut hasher);
            token_idx.hash(&mut hasher);
            let expert_id = (hasher.finish() as usize) % self.config.num_experts;
            let device_rank = self.expert_assignments[expert_id].device_rank;

            *token_routing.capacity_usage.entry(expert_id).or_insert(0) += 1;
            token_routing.expert_assignments.push(vec![(expert_id, 1.0)]);
            token_routing.destinations.entry(device_rank).or_default().push(token_idx);
        }

        Ok(token_routing)
    }

    /// Load-aware token routing.
    ///
    /// Each token's gating probabilities are discounted by the expert's current
    /// load — the historical load recorded in [`LoadBalancingState`] plus the
    /// tokens already assigned in this batch — so a highly-rated but saturated
    /// expert loses to a slightly worse but idle one. With all loads equal this
    /// degenerates to plain top-k gating.
    fn load_based_routing(&self, tokens: &Tensor, gating_scores: &Tensor) -> Result<TokenRouting> {
        let logits = self.gating_logits(tokens, gating_scores)?;
        let num_tokens = logits.len();

        let mut token_routing = TokenRouting {
            token_indices: (0..num_tokens).collect(),
            expert_assignments: Vec::with_capacity(num_tokens),
            destinations: HashMap::new(),
            capacity_usage: HashMap::new(),
        };

        let historical_loads: Vec<f32> = {
            let load_state = self
                .load_balancing_state
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            (0..self.config.num_experts)
                .map(|expert_id| *load_state.expert_loads.get(&expert_id).unwrap_or(&0.0))
                .collect()
        };

        // Running load accumulated while routing this batch.
        let mut batch_loads = vec![0.0f32; self.config.num_experts];
        let top_k = self.config.top_k.clamp(1, self.config.num_experts);

        for (token_idx, row) in logits.iter().enumerate() {
            let probabilities = Self::softmax(row);

            let mut ranked: Vec<(usize, f32)> = (0..self.config.num_experts)
                .map(|expert_id| {
                    let load = historical_loads[expert_id] + batch_loads[expert_id];
                    (expert_id, probabilities[expert_id] / (1.0 + load))
                })
                .collect();
            ranked.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
            });
            ranked.truncate(top_k);

            // Weights come from the *gating* distribution, not the discounted
            // score: load balancing decides who gets the token, gating decides
            // how much it counts.
            let mut assignments: Vec<(usize, f32)> = ranked
                .iter()
                .map(|(expert_id, _)| (*expert_id, probabilities[*expert_id]))
                .collect();
            let total: f32 = assignments.iter().map(|(_, weight)| *weight).sum();
            if total > 0.0 {
                for (_, weight) in assignments.iter_mut() {
                    *weight /= total;
                }
            } else {
                let uniform = 1.0 / assignments.len().max(1) as f32;
                for (_, weight) in assignments.iter_mut() {
                    *weight = uniform;
                }
            }

            for (expert_id, _) in &assignments {
                batch_loads[*expert_id] += 1.0;
                *token_routing.capacity_usage.entry(*expert_id).or_insert(0) += 1;
                let device_rank = self.expert_assignments[*expert_id].device_rank;
                token_routing.destinations.entry(device_rank).or_default().push(token_idx);
            }
            token_routing.expert_assignments.push(assignments);
        }

        Ok(token_routing)
    }

    /// Similarity-based token routing.
    ///
    /// Expert prototypes are formed as the gating-weighted mean of the batch's
    /// token embeddings; each token is then routed to the `top_k` experts whose
    /// prototype has the highest cosine similarity to it. Semantically similar
    /// tokens therefore share experts even when their raw gating logits differ
    /// slightly.
    fn similarity_based_routing(
        &self,
        tokens: &Tensor,
        gating_scores: &Tensor,
    ) -> Result<TokenRouting> {
        let logits = self.gating_logits(tokens, gating_scores)?;
        let embeddings = self.token_embeddings(tokens)?;
        if embeddings.len() != logits.len() {
            return Err(anyhow!(
                "token tensor describes {} tokens but gating_scores describes {}",
                embeddings.len(),
                logits.len()
            ));
        }
        let num_experts = self.config.num_experts;
        let features = embeddings.first().map(Vec::len).unwrap_or(0);

        // Prototype[e] = sum_t softmax(logits_t)[e] * x_t
        let mut prototypes = vec![vec![0.0f32; features]; num_experts];
        let mut prototype_mass = vec![0.0f32; num_experts];
        let mut probabilities_per_token = Vec::with_capacity(logits.len());
        for (embedding, row) in embeddings.iter().zip(&logits) {
            let probabilities = Self::softmax(row);
            for (expert_id, probability) in probabilities.iter().enumerate() {
                prototype_mass[expert_id] += probability;
                for (slot, value) in prototypes[expert_id].iter_mut().zip(embedding) {
                    *slot += probability * value;
                }
            }
            probabilities_per_token.push(probabilities);
        }
        for (prototype, mass) in prototypes.iter_mut().zip(&prototype_mass) {
            if *mass > 0.0 {
                let inverse = 1.0 / mass;
                for slot in prototype.iter_mut() {
                    *slot *= inverse;
                }
            }
        }

        let mut token_routing = TokenRouting {
            token_indices: (0..embeddings.len()).collect(),
            expert_assignments: Vec::with_capacity(embeddings.len()),
            destinations: HashMap::new(),
            capacity_usage: HashMap::new(),
        };
        let top_k = self.config.top_k.clamp(1, num_experts);

        for (token_idx, embedding) in embeddings.iter().enumerate() {
            let mut ranked: Vec<(usize, f32)> = prototypes
                .iter()
                .enumerate()
                .map(|(expert_id, prototype)| (expert_id, cosine_similarity(embedding, prototype)))
                .collect();
            ranked.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
            });
            ranked.truncate(top_k);

            // Map similarities in [-1, 1] onto non-negative mixing weights.
            let mut assignments: Vec<(usize, f32)> = ranked
                .iter()
                .map(|(expert_id, similarity)| (*expert_id, (similarity + 1.0) * 0.5))
                .collect();
            let total: f32 = assignments.iter().map(|(_, weight)| *weight).sum();
            if total > 0.0 {
                for (_, weight) in assignments.iter_mut() {
                    *weight /= total;
                }
            } else {
                let uniform = 1.0 / assignments.len().max(1) as f32;
                for (_, weight) in assignments.iter_mut() {
                    *weight = uniform;
                }
            }

            for (expert_id, _) in &assignments {
                *token_routing.capacity_usage.entry(*expert_id).or_insert(0) += 1;
                let device_rank = self.expert_assignments[*expert_id].device_rank;
                token_routing.destinations.entry(device_rank).or_default().push(token_idx);
            }
            token_routing.expert_assignments.push(assignments);
        }

        Ok(token_routing)
    }

    /// Perform all-to-all communication for expert parallelism
    pub fn all_to_all_communication(
        &self,
        local_tokens: &Tensor,
        routing: &TokenRouting,
    ) -> Result<HashMap<usize, Tensor>> {
        let start_time = Instant::now();

        // Gather the *actual* rows of `local_tokens` that each local expert has
        // been assigned. Token order within an expert's batch follows the token
        // index, which every rank derives identically from the routing table.
        let embeddings = self.token_embeddings(local_tokens)?;
        let features = embeddings.first().map(Vec::len).unwrap_or(0);
        let mut expert_inputs = HashMap::new();
        let mut bytes_moved = 0u64;

        for expert_id in &self.local_experts {
            let mut gathered: Vec<f32> = Vec::new();
            let mut rows = 0usize;

            for (token_idx, assignments) in routing.expert_assignments.iter().enumerate() {
                let assigned = assignments.iter().any(|(assigned_expert_id, weight)| {
                    assigned_expert_id == expert_id && *weight > 0.0
                });
                if !assigned {
                    continue;
                }
                let embedding = embeddings.get(token_idx).ok_or_else(|| {
                    anyhow!(
                        "routing references token {} but only {} tokens were supplied",
                        token_idx,
                        embeddings.len()
                    )
                })?;
                gathered.extend_from_slice(embedding);
                rows += 1;
            }

            if rows > 0 {
                bytes_moved += (gathered.len() * std::mem::size_of::<f32>()) as u64;
                expert_inputs.insert(
                    *expert_id,
                    Tensor::from_slice(&gathered, &[rows, features])?,
                );
            }
        }

        // Update communication statistics
        {
            let mut stats =
                self.communication_stats.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            stats.all_to_all_time += start_time.elapsed();
            stats.communication_efficiency = if bytes_moved > 0 {
                bytes_moved as f32 / (local_tokens.len() * std::mem::size_of::<f32>()).max(1) as f32
            } else {
                0.0
            };
        }

        Ok(expert_inputs)
    }

    /// Update load balancing state
    pub fn update_load_balancing(&self, expert_outputs: &HashMap<usize, Tensor>) -> Result<()> {
        let mut load_state = self
            .load_balancing_state
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

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
        let load_state = self
            .load_balancing_state
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let comm_stats =
            self.communication_stats.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

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

        let expert_parallel_size = num_experts.div_ceil(experts_per_device);

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

        let assignments = ExpertParallelism::create_expert_assignments(&config, 4)
            .expect("operation failed in test");
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

        let assignments = ExpertParallelism::create_expert_assignments(&config, 4)
            .expect("operation failed in test");
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
        let ep = expert_parallelism.expect("operation failed in test");
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
        let expert_parallelism =
            ExpertParallelism::new(config, 0, 4, process_group).expect("operation failed in test");

        let tokens = Tensor::zeros(&[8, 16]).expect("tensor operation failed");
        let routing = expert_parallelism
            .hash_based_routing(&tokens)
            .expect("operation failed in test");

        assert_eq!(routing.token_indices.len(), 8);
        assert_eq!(routing.expert_assignments.len(), 8);
    }

    #[test]
    fn test_load_balancing_update() {
        let config = ExpertParallelismConfig::default();
        let process_group = Arc::new(SimulatedProcessGroup::new(0, 4));
        let expert_parallelism =
            ExpertParallelism::new(config, 0, 4, process_group).expect("operation failed in test");

        let mut expert_outputs = HashMap::new();
        expert_outputs.insert(
            0,
            Tensor::zeros(&[10, 16]).expect("tensor operation failed"),
        );
        expert_outputs.insert(
            1,
            Tensor::zeros(&[15, 16]).expect("tensor operation failed"),
        );

        let result = expert_parallelism.update_load_balancing(&expert_outputs);
        assert!(result.is_ok());

        let stats = expert_parallelism.get_load_balancing_stats();
        assert!(stats.expert_loads.contains_key(&0));
        assert!(stats.expert_loads.contains_key(&1));
    }

    fn four_expert_parallelism(strategy: ExpertRoutingStrategy, top_k: usize) -> ExpertParallelism {
        let config = ExpertParallelismConfig {
            num_experts: 4,
            experts_per_device: 1,
            expert_parallel_size: 4,
            top_k,
            routing_strategy: strategy,
            drop_tokens: false,
            ..Default::default()
        };
        let process_group = Arc::new(SimulatedProcessGroup::new(0, 1));
        ExpertParallelism::new(config, 0, 4, process_group)
            .expect("expert parallelism must build in test")
    }

    #[test]
    fn learned_gating_follows_the_gating_scores() {
        // Four tokens, each with a different argmax expert. The old
        // implementation ignored `gating_scores` entirely and routed every
        // token to experts 0..top_k.
        let expert_parallelism = four_expert_parallelism(ExpertRoutingStrategy::LearnedGating, 1);

        let tokens = Tensor::zeros(&[4, 3]).expect("tensor must build in test");
        let gating = Tensor::from_slice(
            &[
                5.0, 0.0, 0.0, 0.0, // token 0 -> expert 0
                0.0, 5.0, 0.0, 0.0, // token 1 -> expert 1
                0.0, 0.0, 5.0, 0.0, // token 2 -> expert 2
                0.0, 0.0, 0.0, 5.0, // token 3 -> expert 3
            ],
            &[4, 4],
        )
        .expect("tensor must build in test");

        let routing = expert_parallelism
            .route_tokens(&tokens, &gating)
            .expect("routing must succeed in test");

        assert_eq!(routing.expert_assignments.len(), 4);
        for (token_idx, assignments) in routing.expert_assignments.iter().enumerate() {
            assert_eq!(assignments.len(), 1, "top_k = 1");
            assert_eq!(
                assignments[0].0, token_idx,
                "token {token_idx} must go to expert {token_idx}"
            );
            approx::assert_relative_eq!(assignments[0].1, 1.0f32, epsilon = 1e-6);
        }

        // Every expert received exactly one token: the routing is not collapsed
        // onto expert 0.
        let mut experts_used: Vec<usize> = routing
            .expert_assignments
            .iter()
            .flat_map(|assignments| assignments.iter().map(|(expert, _)| *expert))
            .collect();
        experts_used.sort_unstable();
        assert_eq!(experts_used, vec![0, 1, 2, 3]);
    }

    #[test]
    fn learned_gating_top_k_weights_are_softmax_normalised() {
        let expert_parallelism = four_expert_parallelism(ExpertRoutingStrategy::LearnedGating, 2);

        let tokens = Tensor::zeros(&[1, 2]).expect("tensor must build in test");
        // logits [2, 1, 0, 0] -> softmax picks experts 0 and 1.
        let gating =
            Tensor::from_slice(&[2.0, 1.0, 0.0, 0.0], &[1, 4]).expect("tensor must build in test");

        let routing = expert_parallelism
            .route_tokens(&tokens, &gating)
            .expect("routing must succeed in test");

        let assignments = &routing.expert_assignments[0];
        assert_eq!(assignments.len(), 2);
        assert_eq!(assignments[0].0, 0);
        assert_eq!(assignments[1].0, 1);

        // Renormalized softmax over the two retained logits: e^2 / (e^2 + e^1).
        let expected_first = 1.0f32 / (1.0 + (-1.0f32).exp());
        approx::assert_relative_eq!(assignments[0].1, expected_first, epsilon = 1e-5);
        approx::assert_relative_eq!(assignments[0].1 + assignments[1].1, 1.0f32, epsilon = 1e-6);
    }

    #[test]
    fn different_gating_rows_produce_different_assignments() {
        let expert_parallelism = four_expert_parallelism(ExpertRoutingStrategy::LearnedGating, 1);

        let tokens = Tensor::zeros(&[2, 2]).expect("tensor must build in test");
        let gating = Tensor::from_slice(&[0.0, 0.0, 9.0, 0.0, 9.0, 0.0, 0.0, 0.0], &[2, 4])
            .expect("tensor must build in test");

        let routing = expert_parallelism
            .route_tokens(&tokens, &gating)
            .expect("routing must succeed in test");

        assert_ne!(
            routing.expert_assignments[0][0].0, routing.expert_assignments[1][0].0,
            "tokens with different gating rows must not share an expert"
        );
        assert_eq!(routing.expert_assignments[0][0].0, 2);
        assert_eq!(routing.expert_assignments[1][0].0, 0);
    }

    #[test]
    fn routing_rejects_a_mismatched_gating_tensor() {
        let expert_parallelism = four_expert_parallelism(ExpertRoutingStrategy::LearnedGating, 1);
        let tokens = Tensor::zeros(&[4, 3]).expect("tensor must build in test");

        // Wrong expert axis.
        let bad_experts = Tensor::zeros(&[4, 3]).expect("tensor must build in test");
        assert!(expert_parallelism.route_tokens(&tokens, &bad_experts).is_err());

        // Wrong token count.
        let bad_tokens = Tensor::zeros(&[2, 4]).expect("tensor must build in test");
        assert!(expert_parallelism.route_tokens(&tokens, &bad_tokens).is_err());
    }

    #[test]
    fn load_based_routing_spreads_tokens_when_gating_is_uniform() {
        let expert_parallelism = four_expert_parallelism(ExpertRoutingStrategy::LoadBased, 1);

        let tokens = Tensor::zeros(&[8, 2]).expect("tensor must build in test");
        let gating =
            Tensor::from_slice(&[0.0f32; 8 * 4], &[8, 4]).expect("tensor must build in test");

        let routing = expert_parallelism
            .route_tokens(&tokens, &gating)
            .expect("routing must succeed in test");

        // Eight tokens, four experts, uniform gating: perfectly balanced.
        for expert_id in 0..4 {
            assert_eq!(
                routing.capacity_usage.get(&expert_id).copied().unwrap_or(0),
                2,
                "expert {expert_id} load"
            );
        }
    }

    #[test]
    fn similarity_routing_groups_similar_tokens() {
        let expert_parallelism = four_expert_parallelism(ExpertRoutingStrategy::SimilarityBased, 1);

        // Two clusters: tokens 0,1 point along +x; tokens 2,3 along +y.
        let tokens = Tensor::from_slice(&[1.0, 0.0, 0.9, 0.1, 0.0, 1.0, 0.1, 0.9], &[4, 2])
            .expect("tensor must build in test");
        let gating = Tensor::from_slice(
            &[
                4.0, 0.0, 0.0, 0.0, //
                4.0, 0.0, 0.0, 0.0, //
                0.0, 4.0, 0.0, 0.0, //
                0.0, 4.0, 0.0, 0.0, //
            ],
            &[4, 4],
        )
        .expect("tensor must build in test");

        let routing = expert_parallelism
            .route_tokens(&tokens, &gating)
            .expect("routing must succeed in test");

        assert_eq!(
            routing.expert_assignments[0][0].0, routing.expert_assignments[1][0].0,
            "the +x cluster must share an expert"
        );
        assert_eq!(
            routing.expert_assignments[2][0].0, routing.expert_assignments[3][0].0,
            "the +y cluster must share an expert"
        );
        assert_ne!(
            routing.expert_assignments[0][0].0, routing.expert_assignments[2][0].0,
            "the two clusters must land on different experts"
        );
    }

    #[test]
    fn hash_routing_depends_on_token_content() {
        let expert_parallelism = four_expert_parallelism(ExpertRoutingStrategy::HashBased, 1);

        let identical =
            Tensor::from_slice(&[1.0, 2.0, 1.0, 2.0], &[2, 2]).expect("tensor must build in test");
        let routing = expert_parallelism
            .hash_based_routing(&identical)
            .expect("routing must succeed in test");
        assert_eq!(
            routing.expert_assignments[0][0].0, routing.expert_assignments[1][0].0,
            "identical tokens must hash to the same expert"
        );

        let distinct =
            Tensor::from_slice(&[1.0, 2.0, 7.0, -3.0], &[2, 2]).expect("tensor must build in test");
        let other = expert_parallelism
            .hash_based_routing(&distinct)
            .expect("routing must succeed in test");
        assert_eq!(other.expert_assignments.len(), 2);
    }

    #[test]
    fn all_to_all_moves_the_real_token_rows() {
        let expert_parallelism = four_expert_parallelism(ExpertRoutingStrategy::LearnedGating, 1);

        let tokens = Tensor::from_slice(
            &[
                1.0, 2.0, // token 0
                3.0, 4.0, // token 1
                5.0, 6.0, // token 2
                7.0, 8.0, // token 3
            ],
            &[4, 2],
        )
        .expect("tensor must build in test");
        let gating = Tensor::from_slice(
            &[
                9.0, 0.0, 0.0, 0.0, //
                0.0, 9.0, 0.0, 0.0, //
                9.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 9.0, 0.0, //
            ],
            &[4, 4],
        )
        .expect("tensor must build in test");

        let routing = expert_parallelism
            .route_tokens(&tokens, &gating)
            .expect("routing must succeed in test");
        let expert_inputs = expert_parallelism
            .all_to_all_communication(&tokens, &routing)
            .expect("all-to-all must succeed in test");

        // Rank 0 owns expert 0, which received tokens 0 and 2.
        let expert_zero = expert_inputs.get(&0).expect("expert 0 must receive tokens");
        assert_eq!(expert_zero.shape(), vec![2, 2]);
        assert_eq!(
            expert_zero.to_vec_f32().expect("tensor read must succeed in test"),
            vec![1.0, 2.0, 5.0, 6.0],
            "the expert batch must contain the real token rows, not zeros"
        );
    }

    #[test]
    fn test_optimal_expert_config_calculation() {
        let config = utils::calculate_optimal_expert_config(16, 8, 1000, 4000)
            .expect("operation failed in test");
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
