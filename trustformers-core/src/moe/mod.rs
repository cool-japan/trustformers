//! Mixture of Experts (MoE) routing primitives
//!
//! Provides top-k gating, load-balance statistics, auxiliary losses, and
//! expert-capacity enforcement for MoE transformer layers.
//!
//! ## Sub-modules
//! - [`routing`]: Advanced routing strategies (Expert Choice, Hash, Switch Transformer, Random).
//! - [`expert_parallel`]: Expert parallelism all-to-all communication simulator.

pub mod routing;
pub mod expert_parallel;

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error types
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur in MoE routing operations.
#[derive(Debug, Clone, PartialEq)]
pub enum MoeError {
    /// `num_experts` must be ≥ 1.
    InvalidNumExperts(usize),
    /// `top_k` must be ≤ `num_experts`.
    InvalidTopK {
        top_k: usize,
        num_experts: usize,
    },
    /// Batch of token logits was empty.
    EmptyBatch,
    /// Logits vector length did not match `num_experts`.
    LogitsDimensionMismatch {
        expected: usize,
        got: usize,
    },
}

impl fmt::Display for MoeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MoeError::InvalidNumExperts(n) => {
                write!(f, "invalid num_experts: {n} (must be ≥ 1)")
            }
            MoeError::InvalidTopK { top_k, num_experts } => {
                write!(
                    f,
                    "invalid top_k: {top_k} > num_experts: {num_experts}"
                )
            }
            MoeError::EmptyBatch => write!(f, "batch of token logits is empty"),
            MoeError::LogitsDimensionMismatch { expected, got } => {
                write!(
                    f,
                    "logits dimension mismatch: expected {expected}, got {got}"
                )
            }
        }
    }
}

impl std::error::Error for MoeError {}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically stable softmax over a slice.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();
    if sum == 0.0 {
        // Degenerate: return uniform distribution
        let n = logits.len();
        return vec![1.0 / n as f32; n];
    }
    exp_vals.iter().map(|&e| e / sum).collect()
}

/// Return the indices that would sort `vals` in descending order (top-k first).
///
/// Uses a simple selection sort variant that is O(n*k) — acceptable for
/// the small `num_experts` values seen in practice.
fn top_k_indices(vals: &[f32], k: usize) -> Vec<usize> {
    let n = vals.len();
    let k = k.min(n);
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let mut best = i;
        for j in (i + 1)..n {
            if vals[indices[j]] > vals[indices[best]] {
                best = j;
            }
        }
        indices.swap(i, best);
    }
    indices[..k].to_vec()
}

// ─────────────────────────────────────────────────────────────────────────────
// RouterOutput
// ─────────────────────────────────────────────────────────────────────────────

/// Routing decision for a single token.
#[derive(Debug, Clone)]
pub struct RouterOutput {
    /// Indices of the selected experts (top-k, in descending gate-weight order).
    pub expert_indices: Vec<usize>,
    /// Normalized gate weights for each selected expert (sum to 1.0).
    pub gate_weights: Vec<f32>,
    /// Raw logits over all experts before softmax.
    pub raw_logits: Vec<f32>,
    /// Shannon entropy of the gate-weight distribution.
    pub entropy: f32,
}

impl RouterOutput {
    /// Compute Shannon entropy: `H = -∑ w·ln(w)` (nats).
    ///
    /// Weights near zero are skipped to avoid `log(0)`.
    pub fn compute_entropy(weights: &[f32]) -> f32 {
        weights
            .iter()
            .filter(|&&w| w > 1e-10)
            .map(|&w| -w * w.ln())
            .sum()
    }

    /// Return `true` if a single expert received more than `threshold` of the
    /// total gate weight, indicating a degenerate (collapsed) routing.
    pub fn is_degenerate(&self, threshold: f32) -> bool {
        self.gate_weights.iter().any(|&w| w > threshold)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TopKRouter
// ─────────────────────────────────────────────────────────────────────────────

/// Routes each token to its top-k experts based on gate logits.
pub struct TopKRouter {
    /// Total number of experts.
    pub num_experts: usize,
    /// Number of experts to select per token.
    pub top_k: usize,
    /// If `true`, renormalize selected gate weights to sum to 1.
    pub normalize_gates: bool,
    /// Standard deviation of optional jitter noise added to logits.
    /// Set to `0.0` for deterministic routing.
    pub noise_std: f32,
}

impl TopKRouter {
    /// Create a router with default settings (normalize gates, no noise).
    pub fn new(num_experts: usize, top_k: usize) -> Self {
        Self {
            num_experts,
            top_k,
            normalize_gates: true,
            noise_std: 0.0,
        }
    }

    /// Route a single token given pre-computed logits.
    ///
    /// `logits` must have length == `num_experts`.
    pub fn route(&self, logits: &[f32]) -> Result<RouterOutput, MoeError> {
        if self.num_experts == 0 {
            return Err(MoeError::InvalidNumExperts(0));
        }
        if self.top_k > self.num_experts {
            return Err(MoeError::InvalidTopK {
                top_k: self.top_k,
                num_experts: self.num_experts,
            });
        }
        if logits.len() != self.num_experts {
            return Err(MoeError::LogitsDimensionMismatch {
                expected: self.num_experts,
                got: logits.len(),
            });
        }

        // Optional deterministic noise (index-based perturbation)
        let effective_logits: Vec<f32> = if self.noise_std > 0.0 {
            logits
                .iter()
                .enumerate()
                .map(|(i, &l)| {
                    // Deterministic pseudo-noise: sin-based perturbation
                    let noise = self.noise_std * ((i as f32 + 1.0).sin());
                    l + noise
                })
                .collect()
        } else {
            logits.to_vec()
        };

        let raw_logits = effective_logits.clone();

        // Softmax over all experts
        let all_probs = softmax(&effective_logits);

        // Top-k selection
        let selected_indices = top_k_indices(&all_probs, self.top_k);

        // Gather weights for selected experts
        let mut selected_weights: Vec<f32> =
            selected_indices.iter().map(|&i| all_probs[i]).collect();

        // Optionally renormalize
        if self.normalize_gates {
            let total: f32 = selected_weights.iter().sum();
            if total > 1e-10 {
                for w in &mut selected_weights {
                    *w /= total;
                }
            }
        }

        let entropy = RouterOutput::compute_entropy(&selected_weights);

        Ok(RouterOutput {
            expert_indices: selected_indices,
            gate_weights: selected_weights,
            raw_logits,
            entropy,
        })
    }

    /// Route a batch of tokens.
    ///
    /// `batch_logits[i]` must have length == `num_experts`.
    pub fn route_batch(
        &self,
        batch_logits: &[Vec<f32>],
    ) -> Result<Vec<RouterOutput>, MoeError> {
        if batch_logits.is_empty() {
            return Err(MoeError::EmptyBatch);
        }
        batch_logits
            .iter()
            .map(|logits| self.route(logits))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LoadBalanceStats
// ─────────────────────────────────────────────────────────────────────────────

/// Load distribution statistics across experts for a batch of router outputs.
#[derive(Debug, Clone)]
pub struct LoadBalanceStats {
    /// Fraction of token-expert assignments going to each expert.
    /// `expert_fractions[i] = count_i / (num_tokens * top_k)`.
    pub expert_fractions: Vec<f32>,
    /// Imbalance ratio: `max_fraction / ideal_fraction`.
    /// A perfectly balanced router gives `1.0`.
    pub imbalance_ratio: f32,
    /// Coefficient of variation (std / mean) of expert fractions.
    pub cv: f32,
    /// Gini coefficient: `0.0` = perfect balance, `1.0` = all on one expert.
    pub gini: f32,
    /// Number of tokens in the batch.
    pub num_tokens: usize,
    /// Number of experts.
    pub num_experts: usize,
}

impl LoadBalanceStats {
    /// Compute statistics from a batch of router outputs.
    pub fn from_batch(outputs: &[RouterOutput], num_experts: usize) -> Self {
        let num_tokens = outputs.len();
        let mut counts = vec![0usize; num_experts];

        for out in outputs {
            for &idx in &out.expert_indices {
                if idx < num_experts {
                    counts[idx] += 1;
                }
            }
        }

        let top_k = outputs.first().map_or(1, |o| o.expert_indices.len().max(1));
        let total_assignments = num_tokens * top_k;

        let fractions: Vec<f32> = counts
            .iter()
            .map(|&c| {
                if total_assignments == 0 {
                    0.0
                } else {
                    c as f32 / total_assignments as f32
                }
            })
            .collect();

        let ideal = if num_experts == 0 {
            0.0
        } else {
            1.0 / num_experts as f32
        };

        let max_frac = fractions.iter().cloned().fold(0.0_f32, f32::max);
        let imbalance_ratio = if ideal > 0.0 { max_frac / ideal } else { 0.0 };

        // Mean and std for CV
        let mean = fractions.iter().sum::<f32>() / fractions.len().max(1) as f32;
        let variance = fractions
            .iter()
            .map(|&f| (f - mean).powi(2))
            .sum::<f32>()
            / fractions.len().max(1) as f32;
        let std_dev = variance.sqrt();
        let cv = if mean > 1e-10 { std_dev / mean } else { 0.0 };

        // Gini coefficient
        let mut sorted = fractions.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len();
        let gini = if n == 0 || mean < 1e-10 {
            0.0
        } else {
            let numerator: f32 = sorted
                .iter()
                .enumerate()
                .map(|(i, &f)| {
                    let coeff = (2 * (i + 1)) as i64 - n as i64 - 1;
                    coeff as f32 * f
                })
                .sum::<f32>()
                .abs();
            numerator / (n as f32 * sorted.iter().sum::<f32>())
        };

        LoadBalanceStats {
            expert_fractions: fractions,
            imbalance_ratio,
            cv,
            gini,
            num_tokens,
            num_experts,
        }
    }

    /// Return `true` if the imbalance ratio is below `threshold`.
    pub fn is_balanced(&self, threshold: f32) -> bool {
        self.imbalance_ratio < threshold
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LoadBalanceLoss
// ─────────────────────────────────────────────────────────────────────────────

/// Auxiliary loss functions that encourage balanced expert utilization.
#[derive(Debug, Clone)]
pub struct LoadBalanceLoss {
    /// Coefficient multiplied with the auxiliary loss.
    pub alpha: f32,
}

impl LoadBalanceLoss {
    /// Create a new loss helper with the given coefficient.
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }

    /// Switch Transformer auxiliary loss.
    ///
    /// `loss = alpha * num_experts * ∑ f_i * P_i`
    ///
    /// where:
    /// - `f_i` = fraction of tokens routed to expert `i`
    /// - `P_i` = mean gate probability assigned to expert `i` across the batch
    pub fn compute_switch(
        &self,
        outputs: &[RouterOutput],
        num_experts: usize,
    ) -> f32 {
        if outputs.is_empty() || num_experts == 0 {
            return 0.0;
        }

        let num_tokens = outputs.len();
        let top_k = outputs[0].expert_indices.len().max(1);
        let total_assignments = (num_tokens * top_k) as f32;

        // f_i: fraction of tokens assigned to each expert
        let mut counts = vec![0usize; num_experts];
        for out in outputs {
            for &idx in &out.expert_indices {
                if idx < num_experts {
                    counts[idx] += 1;
                }
            }
        }
        let fractions: Vec<f32> = counts
            .iter()
            .map(|&c| c as f32 / total_assignments)
            .collect();

        // P_i: mean gate probability per expert
        // For each token, the gate weight for expert i is the weight at the
        // position of expert i in that token's top-k list (0 if not selected).
        let mut prob_sums = vec![0.0_f32; num_experts];
        for out in outputs {
            for (&idx, &w) in out.expert_indices.iter().zip(out.gate_weights.iter()) {
                if idx < num_experts {
                    prob_sums[idx] += w;
                }
            }
        }
        let mean_probs: Vec<f32> = prob_sums
            .iter()
            .map(|&s| s / num_tokens as f32)
            .collect();

        let dot: f32 = fractions
            .iter()
            .zip(mean_probs.iter())
            .map(|(&f, &p)| f * p)
            .sum();

        self.alpha * num_experts as f32 * dot
    }

    /// Z-loss to prevent router logits from growing too large.
    ///
    /// `Z_loss = alpha * mean_over_batch( log_sum_exp(logits)^2 )`
    pub fn compute_z_loss(&self, batch_logits: &[Vec<f32>]) -> f32 {
        if batch_logits.is_empty() {
            return 0.0;
        }
        let sum_sq: f32 = batch_logits
            .iter()
            .map(|logits| {
                // log_sum_exp(logits) = log( sum( exp(logit_i) ) )
                let max = logits
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max);
                let lse = logits
                    .iter()
                    .map(|&x| (x - max).exp())
                    .sum::<f32>()
                    .ln()
                    + max;
                lse * lse
            })
            .sum();
        self.alpha * sum_sq / batch_logits.len() as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ExpertCapacity
// ─────────────────────────────────────────────────────────────────────────────

/// Capacity constraints for expert parallelism.
///
/// Limits how many tokens each expert can process in a single forward pass,
/// enabling efficient batch-parallel execution with fixed-size expert buffers.
#[derive(Debug, Clone)]
pub struct ExpertCapacity {
    /// Total number of experts.
    pub num_experts: usize,
    /// Number of experts selected per token.
    pub top_k: usize,
    /// Capacity factor: `1.0` = exact, `1.25` = 25 % overflow headroom.
    pub capacity_factor: f32,
}

impl ExpertCapacity {
    /// Create a new capacity constraint.
    pub fn new(num_experts: usize, top_k: usize, capacity_factor: f32) -> Self {
        Self {
            num_experts,
            top_k,
            capacity_factor,
        }
    }

    /// Maximum tokens a single expert may process given a batch of `batch_size`.
    ///
    /// `capacity = ceil(capacity_factor * batch_size * top_k / num_experts)`
    pub fn tokens_per_expert(&self, batch_size: usize) -> usize {
        if self.num_experts == 0 {
            return 0;
        }
        let exact =
            self.capacity_factor * batch_size as f32 * self.top_k as f32
                / self.num_experts as f32;
        exact.ceil() as usize
    }

    /// Apply the capacity constraint to a batch of routing decisions.
    ///
    /// Tokens are processed in order.  When an expert's allocation exceeds the
    /// computed capacity, subsequent tokens are "dropped" from that expert:
    /// their gate weight for the overflowing expert is zeroed out and
    /// redistributed uniformly across **all** experts.
    ///
    /// Returns `(accepted_outputs, total_dropped_assignments)`.
    pub fn apply_capacity(
        &self,
        outputs: &[RouterOutput],
        batch_size: usize,
    ) -> (Vec<RouterOutput>, usize) {
        let capacity = self.tokens_per_expert(batch_size);
        let mut expert_counts = vec![0usize; self.num_experts];
        let mut total_dropped = 0usize;

        let mut result = Vec::with_capacity(outputs.len());

        for out in outputs {
            let mut new_indices: Vec<usize> = Vec::with_capacity(out.expert_indices.len());
            let mut new_weights: Vec<f32> = Vec::with_capacity(out.gate_weights.len());
            let mut dropped_weight = 0.0_f32;

            for (&idx, &w) in out.expert_indices.iter().zip(out.gate_weights.iter()) {
                if idx < self.num_experts && expert_counts[idx] < capacity {
                    expert_counts[idx] += 1;
                    new_indices.push(idx);
                    new_weights.push(w);
                } else {
                    dropped_weight += w;
                    total_dropped += 1;
                }
            }

            // Redistribute dropped weight uniformly across all experts
            // (simplified: add a uniform fallback entry if weight was lost)
            if dropped_weight > 1e-10 && self.num_experts > 0 {
                let uniform_w = dropped_weight / self.num_experts as f32;
                for expert_id in 0..self.num_experts {
                    new_indices.push(expert_id);
                    new_weights.push(uniform_w);
                }
            }

            let entropy = RouterOutput::compute_entropy(&new_weights);
            result.push(RouterOutput {
                expert_indices: new_indices,
                gate_weights: new_weights,
                raw_logits: out.raw_logits.clone(),
                entropy,
            });
        }

        (result, total_dropped)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ExpertLoadBalancer
// ─────────────────────────────────────────────────────────────────────────────

/// Tracks per-expert utilization and computes auxiliary load-balance losses.
///
/// The auxiliary loss from the Switch Transformer paper is:
/// `loss = num_experts * Σ_i f_i * P_i`
/// where
/// - `f_i` = fraction of tokens dispatched to expert `i`
/// - `P_i` = mean router probability for expert `i` across the batch
#[derive(Debug, Clone)]
pub struct ExpertLoadBalancer {
    /// Number of experts
    pub num_experts: usize,
    /// Cumulative token counts per expert (updated via `record_routing`)
    pub expert_counts: Vec<u64>,
    /// Maximum tokens allowed per expert per batch (used by `is_overloaded`)
    pub expert_capacity: usize,
    /// Capacity factor: capacity = capacity_factor * (batch_tokens / num_experts)
    pub capacity_factor: f32,
}

impl ExpertLoadBalancer {
    /// Create a new load balancer.
    ///
    /// `capacity_factor = 1.0` means no slack; `1.25` gives 25 % overflow headroom.
    pub fn new(num_experts: usize, capacity_factor: f32) -> Self {
        Self {
            num_experts,
            expert_counts: vec![0u64; num_experts],
            expert_capacity: 0,
            capacity_factor,
        }
    }

    /// Record a batch of expert assignments (one entry per token).
    ///
    /// Each element of `expert_assignments` is an expert index in `0..num_experts`.
    /// Invalid indices are silently ignored.
    pub fn record_routing(&mut self, expert_assignments: &[usize]) {
        let batch_size = expert_assignments.len();
        // Update per-expert counts
        for &idx in expert_assignments {
            if idx < self.num_experts {
                self.expert_counts[idx] = self.expert_counts[idx].saturating_add(1);
            }
        }
        // Recompute capacity for this batch
        if self.num_experts > 0 {
            let mean_tokens = batch_size as f32 / self.num_experts as f32;
            self.expert_capacity = (self.capacity_factor * mean_tokens).ceil() as usize;
        }
    }

    /// Compute the Switch Transformer auxiliary load-balance loss for a batch.
    ///
    /// `router_probs[token_idx][expert_idx]` = softmax router probability.
    /// Returns `num_experts * Σ_i f_i * P_i`.
    ///
    /// Returns `0.0` on empty input.
    pub fn load_balance_loss(&self, router_probs: &[Vec<f32>]) -> f32 {
        let num_tokens = router_probs.len();
        if num_tokens == 0 || self.num_experts == 0 {
            return 0.0;
        }

        // f_i: fraction of tokens with highest probability at expert i
        let mut dispatch_counts = vec![0usize; self.num_experts];
        for probs in router_probs {
            if probs.len() != self.num_experts {
                continue;
            }
            // find argmax
            let best = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i);
            if let Some(idx) = best {
                dispatch_counts[idx] += 1;
            }
        }
        let fractions: Vec<f32> = dispatch_counts
            .iter()
            .map(|&c| c as f32 / num_tokens as f32)
            .collect();

        // P_i: mean router probability per expert
        let mut prob_sums = vec![0.0_f32; self.num_experts];
        for probs in router_probs {
            if probs.len() == self.num_experts {
                for (i, &p) in probs.iter().enumerate() {
                    prob_sums[i] += p;
                }
            }
        }
        let mean_probs: Vec<f32> = prob_sums
            .iter()
            .map(|&s| s / num_tokens as f32)
            .collect();

        let dot: f32 = fractions
            .iter()
            .zip(mean_probs.iter())
            .map(|(&f, &p)| f * p)
            .sum();

        self.num_experts as f32 * dot
    }

    /// Compute utilization per expert relative to the mean.
    ///
    /// Returns `expert_counts[i] / mean_count` for each expert.
    /// Returns all zeros if the total count is zero.
    pub fn expert_utilization(&self) -> Vec<f32> {
        let total: u64 = self.expert_counts.iter().sum();
        if total == 0 || self.num_experts == 0 {
            return vec![0.0_f32; self.num_experts];
        }
        let mean = total as f32 / self.num_experts as f32;
        self.expert_counts
            .iter()
            .map(|&c| {
                if mean < 1e-10 {
                    0.0
                } else {
                    c as f32 / mean
                }
            })
            .collect()
    }

    /// Return `true` if the cumulative token count for `expert_idx` exceeds
    /// `expert_capacity`.
    pub fn is_overloaded(&self, expert_idx: usize) -> bool {
        if expert_idx >= self.num_experts || self.expert_capacity == 0 {
            return false;
        }
        self.expert_counts[expert_idx] as usize > self.expert_capacity
    }

    /// Count how many tokens in `expert_assignments` overflow the capacity.
    ///
    /// Processes assignments in order: once an expert's running count in *this
    /// call* reaches `expert_capacity`, subsequent tokens to that expert are
    /// counted as overflow.  Returns the total number of overflowed tokens.
    pub fn overflow_tokens(&self, expert_assignments: &[usize]) -> usize {
        if self.expert_capacity == 0 || self.num_experts == 0 {
            return 0;
        }
        let mut per_expert = vec![0usize; self.num_experts];
        let mut overflow = 0usize;
        for &idx in expert_assignments {
            if idx < self.num_experts {
                per_expert[idx] += 1;
                if per_expert[idx] > self.expert_capacity {
                    overflow += 1;
                }
            }
        }
        overflow
    }

    /// Reset cumulative expert counts to zero.
    pub fn reset(&mut self) {
        for c in &mut self.expert_counts {
            *c = 0;
        }
        self.expert_capacity = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Expert dropout
// ─────────────────────────────────────────────────────────────────────────────

/// Randomly drop expert computations during training for regularization.
///
/// Each expert output vector is independently zeroed with probability
/// `drop_prob`.  During inference (`training = false`) the function is a
/// no-op and returns a clone.
///
/// Uses a deterministic LCG seeded with `seed` so results are reproducible
/// without external randomness.
///
/// # Arguments
/// * `expert_outputs` — one flat `Vec<f32>` per token
/// * `drop_prob` — probability of zeroing each vector (clamped to `[0, 1]`)
/// * `training` — if `false`, return input unchanged
/// * `seed` — LCG seed for reproducibility (no stdlib rand/ndarray dependency)
pub fn expert_dropout(
    expert_outputs: &[Vec<f32>],
    drop_prob: f32,
    training: bool,
    seed: u64,
) -> Vec<Vec<f32>> {
    if !training || drop_prob <= 0.0 || expert_outputs.is_empty() {
        return expert_outputs.to_vec();
    }
    let clamped = drop_prob.clamp(0.0, 1.0);
    let threshold = (clamped * u32::MAX as f32) as u64;

    // LCG parameters from Knuth (MMIX)
    const A: u64 = 6_364_136_223_846_793_005;
    const C: u64 = 1_442_695_040_888_963_407;

    let mut state = seed;
    expert_outputs
        .iter()
        .map(|vec| {
            state = state.wrapping_mul(A).wrapping_add(C);
            let sample = state >> 32; // upper 32 bits
            if sample < threshold {
                // Drop: zero out this expert's output
                vec![0.0_f32; vec.len()]
            } else {
                vec.clone()
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Sparse upcycling
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for sparse upcycling (dense-to-MoE conversion).
#[derive(Debug, Clone)]
pub struct UpcyclingConfig {
    /// Number of experts to create
    pub num_experts: usize,
    /// Size of each expert's weight vector (e.g. `intermediate * hidden`)
    pub expert_size: usize,
    /// If `true`, copy dense FFN weights to all experts; otherwise only copy
    /// to expert 0 and zero-init the rest before perturbation.
    pub copy_dense_to_all: bool,
    /// Standard deviation of Gaussian-like perturbation added to each expert.
    /// Small values (e.g. 0.01) break symmetry without destroying the pre-trained
    /// representation.
    pub perturbation_std: f32,
}

impl UpcyclingConfig {
    /// Sensible defaults for upcycling (copy to all, small perturbation).
    pub fn new(num_experts: usize, expert_size: usize) -> Self {
        Self {
            num_experts,
            expert_size,
            copy_dense_to_all: true,
            perturbation_std: 0.01,
        }
    }
}

/// Copy dense FFN weights to all experts, adding small deterministic perturbations
/// to break weight-tying symmetry.
///
/// The perturbation uses a per-expert, per-element LCG so each expert gets a
/// unique but deterministic noise pattern.
///
/// # Arguments
/// * `dense_ffn` — flat weight slice of size `expert_size` (e.g. one weight matrix)
/// * `config` — upcycling configuration
///
/// # Returns
/// A `Vec<Vec<f32>>` of length `num_experts`, each of size `expert_size`.
pub fn upcycle_dense_to_moe(dense_ffn: &[f32], config: &UpcyclingConfig) -> Vec<Vec<f32>> {
    let n = config.num_experts;
    let sz = config.expert_size;

    // LCG parameters
    const A: u64 = 6_364_136_223_846_793_005;
    const C: u64 = 1_442_695_040_888_963_407;

    (0..n)
        .map(|expert_idx| {
            // Each expert starts from a different LCG seed based on its index
            let mut state: u64 = (expert_idx as u64).wrapping_mul(2_654_435_761)
                .wrapping_add(1_013_904_223);

            let base: &[f32] = if config.copy_dense_to_all || expert_idx == 0 {
                // Use dense weights as base (or zero-pad if dense is shorter)
                dense_ffn
            } else {
                // Zero base for experts > 0 when copy_dense_to_all is false
                &[]
            };

            (0..sz)
                .map(|i| {
                    let base_val = if i < base.len() { base[i] } else { 0.0 };
                    // LCG step
                    state = state.wrapping_mul(A).wrapping_add(C);
                    // Convert to f32 in range [-1, 1]
                    let u = (state >> 32) as f32 / u32::MAX as f32 * 2.0 - 1.0;
                    base_val + config.perturbation_std * u
                })
                .collect()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── RouterOutput ────────────────────────────────────────────────────────

    #[test]
    fn test_router_output_entropy_uniform() {
        // For uniform weights the entropy should be ln(k) in nats.
        let k = 4;
        let weights: Vec<f32> = vec![0.25; k];
        let h = RouterOutput::compute_entropy(&weights);
        let expected = (k as f32).ln(); // ln(4) ≈ 1.386
        assert!((h - expected).abs() < 1e-5, "h={h}, expected={expected}");
    }

    #[test]
    fn test_router_output_entropy_concentrated() {
        // All weight on one expert → entropy = 0
        let weights = vec![1.0_f32, 0.0, 0.0, 0.0];
        let h = RouterOutput::compute_entropy(&weights);
        assert!(h < 1e-6, "entropy should be near 0, got {h}");
    }

    #[test]
    fn test_router_output_is_degenerate() {
        let out = RouterOutput {
            expert_indices: vec![0],
            gate_weights: vec![0.95],
            raw_logits: vec![3.0, 0.1],
            entropy: 0.1,
        };
        assert!(out.is_degenerate(0.9));
        assert!(!out.is_degenerate(0.99));
    }

    // ── softmax helper ──────────────────────────────────────────────────────

    #[test]
    fn test_softmax_normalization() {
        let logits = vec![1.0_f32, 2.0, 3.0];
        let probs = softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
        for &p in &probs {
            assert!(p >= 0.0);
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Very large logits should not produce NaN
        let logits = vec![1e30_f32, 1e30, 1e30];
        let probs = softmax(&logits);
        for &p in &probs {
            assert!(p.is_finite(), "softmax should be finite even for large inputs");
        }
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // ── TopKRouter ──────────────────────────────────────────────────────────

    #[test]
    fn test_top_k_router_new() {
        let router = TopKRouter::new(8, 2);
        assert_eq!(router.num_experts, 8);
        assert_eq!(router.top_k, 2);
        assert!(router.normalize_gates);
        assert_eq!(router.noise_std, 0.0);
    }

    #[test]
    fn test_top_k_router_single_token() {
        let router = TopKRouter::new(4, 2);
        let logits = vec![1.0_f32, 4.0, 2.0, 3.0];
        let out = router.route(&logits).expect("route should succeed");
        // Top-2 by logit should be indices 1 (4.0) and 3 (3.0)
        assert_eq!(out.expert_indices.len(), 2);
        assert!(out.expert_indices.contains(&1));
        assert!(out.expert_indices.contains(&3));
        let weight_sum: f32 = out.gate_weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-5, "weights should sum to 1.0");
    }

    #[test]
    fn test_top_k_router_top1() {
        let router = TopKRouter::new(3, 1);
        let logits = vec![0.1_f32, 5.0, 0.5];
        let out = router.route(&logits).expect("route ok");
        assert_eq!(out.expert_indices.len(), 1);
        assert_eq!(out.expert_indices[0], 1); // highest logit
        assert!((out.gate_weights[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_top_k_router_top2() {
        let router = TopKRouter::new(5, 2);
        let logits = vec![0.1_f32, 0.2, 5.0, 3.0, 0.3];
        let out = router.route(&logits).expect("route ok");
        assert_eq!(out.expert_indices.len(), 2);
        // Top-2 should be expert 2 and 3
        assert!(out.expert_indices.contains(&2));
        assert!(out.expert_indices.contains(&3));
    }

    #[test]
    fn test_top_k_router_normalize_gates() {
        let mut router = TopKRouter::new(4, 2);
        router.normalize_gates = true;
        let logits = vec![1.0_f32, 2.0, 3.0, 0.5];
        let out = router.route(&logits).expect("route ok");
        let sum: f32 = out.gate_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_top_k_router_batch() {
        let router = TopKRouter::new(4, 1);
        let batch = vec![
            vec![1.0_f32, 2.0, 0.5, 0.1],
            vec![0.1_f32, 0.2, 5.0, 0.3],
        ];
        let outs = router.route_batch(&batch).expect("batch route ok");
        assert_eq!(outs.len(), 2);
        assert_eq!(outs[0].expert_indices[0], 1); // highest in first token
        assert_eq!(outs[1].expert_indices[0], 2); // highest in second token
    }

    // ── LoadBalanceStats ─────────────────────────────────────────────────────

    fn make_uniform_outputs(num_tokens: usize, num_experts: usize) -> Vec<RouterOutput> {
        // Each token selects one expert in round-robin order
        (0..num_tokens)
            .map(|i| RouterOutput {
                expert_indices: vec![i % num_experts],
                gate_weights: vec![1.0],
                raw_logits: vec![1.0; num_experts],
                entropy: 0.0,
            })
            .collect()
    }

    #[test]
    fn test_load_balance_stats_uniform() {
        let num_experts = 4;
        let outputs = make_uniform_outputs(8, num_experts);
        let stats = LoadBalanceStats::from_batch(&outputs, num_experts);
        // Each expert gets 2/8 = 0.25
        for &f in &stats.expert_fractions {
            assert!((f - 0.25).abs() < 1e-5, "fraction={f}");
        }
        // Imbalance ratio should be 1.0 (ideal)
        assert!((stats.imbalance_ratio - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_load_balance_stats_imbalanced() {
        let num_experts = 4;
        // All 8 tokens go to expert 0
        let outputs: Vec<RouterOutput> = (0..8)
            .map(|_| RouterOutput {
                expert_indices: vec![0],
                gate_weights: vec![1.0],
                raw_logits: vec![1.0; num_experts],
                entropy: 0.0,
            })
            .collect();
        let stats = LoadBalanceStats::from_batch(&outputs, num_experts);
        // Expert 0 gets 1.0, others get 0.0 → imbalance = 1.0 / 0.25 = 4.0
        assert!((stats.imbalance_ratio - 4.0).abs() < 1e-4);
    }

    #[test]
    fn test_load_balance_stats_cv() {
        // Uniform distribution → std = 0, cv = 0
        let num_experts = 4;
        let outputs = make_uniform_outputs(8, num_experts);
        let stats = LoadBalanceStats::from_batch(&outputs, num_experts);
        assert!(stats.cv < 1e-5, "cv should be 0 for uniform, got {}", stats.cv);
    }

    #[test]
    fn test_load_balance_stats_is_balanced() {
        let num_experts = 4;
        let uniform_outputs = make_uniform_outputs(8, num_experts);
        let stats = LoadBalanceStats::from_batch(&uniform_outputs, num_experts);
        // Imbalance ratio == 1.0; threshold of 1.5 should pass
        assert!(stats.is_balanced(1.5));
        // threshold of 0.5 should fail
        assert!(!stats.is_balanced(0.5));
    }

    // ── LoadBalanceLoss ──────────────────────────────────────────────────────

    #[test]
    fn test_load_balance_loss_switch() {
        let loss_fn = LoadBalanceLoss::new(1e-2);
        // Perfectly balanced: 4 tokens, 2 experts, top-1 each
        // Expert 0 gets tokens 0,2; expert 1 gets tokens 1,3 → fractions = [0.5, 0.5]
        let outputs: Vec<RouterOutput> = (0..4)
            .map(|i| RouterOutput {
                expert_indices: vec![i % 2],
                gate_weights: vec![1.0],
                raw_logits: vec![1.0, 1.0],
                entropy: 0.0,
            })
            .collect();
        let l = loss_fn.compute_switch(&outputs, 2);
        // fractions = [0.5, 0.5], mean_probs = [0.5, 0.5]
        // loss = 1e-2 * 2 * (0.5*0.5 + 0.5*0.5) = 1e-2 * 2 * 0.5 = 0.01
        assert!(l > 0.0, "switch loss should be positive");
        assert!(l < 1.0, "switch loss should be small for balanced routing");
    }

    #[test]
    fn test_load_balance_loss_z_loss() {
        let loss_fn = LoadBalanceLoss::new(1e-4);
        let batch_logits = vec![
            vec![1.0_f32, 2.0, 3.0],
            vec![0.5_f32, 0.5, 0.5],
        ];
        let z = loss_fn.compute_z_loss(&batch_logits);
        assert!(z > 0.0, "z-loss should be positive");
        assert!(z.is_finite(), "z-loss should be finite");
    }

    // ── ExpertCapacity ───────────────────────────────────────────────────────

    #[test]
    fn test_expert_capacity_tokens_per_expert() {
        // capacity = ceil(1.25 * 8 * 2 / 4) = ceil(5.0) = 5
        let ec = ExpertCapacity::new(4, 2, 1.25);
        assert_eq!(ec.tokens_per_expert(8), 5);
    }

    #[test]
    fn test_expert_capacity_apply_no_overflow() {
        // 4 tokens, 2 experts, top-1 each, capacity = 2 per expert → no overflow
        let ec = ExpertCapacity::new(2, 1, 1.0);
        let outputs: Vec<RouterOutput> = vec![
            RouterOutput { expert_indices: vec![0], gate_weights: vec![1.0], raw_logits: vec![1.0, 0.0], entropy: 0.0 },
            RouterOutput { expert_indices: vec![1], gate_weights: vec![1.0], raw_logits: vec![0.0, 1.0], entropy: 0.0 },
            RouterOutput { expert_indices: vec![0], gate_weights: vec![1.0], raw_logits: vec![1.0, 0.0], entropy: 0.0 },
            RouterOutput { expert_indices: vec![1], gate_weights: vec![1.0], raw_logits: vec![0.0, 1.0], entropy: 0.0 },
        ];
        let (result, dropped) = ec.apply_capacity(&outputs, 4);
        assert_eq!(dropped, 0, "no tokens should be dropped");
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_expert_capacity_apply_with_overflow() {
        // 4 tokens, 2 experts, top-1 each, capacity = 1 per expert
        // Tokens 0,1,2,3 → experts 0,0,1,1 → experts 0 and 1 overflow on 2nd assignment each
        let ec = ExpertCapacity::new(2, 1, 0.5); // capacity = ceil(0.5 * 4 * 1 / 2) = 1
        let outputs: Vec<RouterOutput> = vec![
            RouterOutput { expert_indices: vec![0], gate_weights: vec![1.0], raw_logits: vec![1.0, 0.0], entropy: 0.0 },
            RouterOutput { expert_indices: vec![0], gate_weights: vec![1.0], raw_logits: vec![1.0, 0.0], entropy: 0.0 },
            RouterOutput { expert_indices: vec![1], gate_weights: vec![1.0], raw_logits: vec![0.0, 1.0], entropy: 0.0 },
            RouterOutput { expert_indices: vec![1], gate_weights: vec![1.0], raw_logits: vec![0.0, 1.0], entropy: 0.0 },
        ];
        let (_result, dropped) = ec.apply_capacity(&outputs, 4);
        assert!(dropped > 0, "some tokens should be dropped due to overflow");
    }

    // ── MoeError Display ─────────────────────────────────────────────────────

    #[test]
    fn test_moe_error_display() {
        let e1 = MoeError::InvalidNumExperts(0);
        assert!(e1.to_string().contains('0'));

        let e2 = MoeError::InvalidTopK {
            top_k: 5,
            num_experts: 3,
        };
        let s2 = e2.to_string();
        assert!(s2.contains('5'));
        assert!(s2.contains('3'));

        let e3 = MoeError::EmptyBatch;
        assert!(e3.to_string().contains("empty"));

        let e4 = MoeError::LogitsDimensionMismatch {
            expected: 8,
            got: 4,
        };
        let s4 = e4.to_string();
        assert!(s4.contains('8'));
        assert!(s4.contains('4'));
    }

    // ── ExpertLoadBalancer ───────────────────────────────────────────────────

    #[test]
    fn test_expert_load_balancer_new() {
        let balancer = ExpertLoadBalancer::new(8, 1.25);
        assert_eq!(balancer.num_experts, 8);
        assert_eq!(balancer.expert_counts.len(), 8);
        assert!(balancer.expert_counts.iter().all(|&c| c == 0));
        assert_eq!(balancer.capacity_factor, 1.25);
    }

    #[test]
    fn test_expert_load_balancer_record_routing_counts() {
        let mut balancer = ExpertLoadBalancer::new(4, 1.0);
        // Send 3 tokens to expert 0, 1 to expert 2
        let assignments = vec![0, 0, 2, 0];
        balancer.record_routing(&assignments);
        assert_eq!(balancer.expert_counts[0], 3);
        assert_eq!(balancer.expert_counts[1], 0);
        assert_eq!(balancer.expert_counts[2], 1);
        assert_eq!(balancer.expert_counts[3], 0);
    }

    #[test]
    fn test_expert_load_balancer_record_routing_sets_capacity() {
        let mut balancer = ExpertLoadBalancer::new(4, 1.0);
        // 8 tokens, 4 experts → mean = 2.0, capacity = ceil(1.0 * 2.0) = 2
        balancer.record_routing(&[0, 1, 2, 3, 0, 1, 2, 3]);
        assert_eq!(balancer.expert_capacity, 2);
    }

    #[test]
    fn test_expert_load_balancer_utilization_uniform() {
        let mut balancer = ExpertLoadBalancer::new(4, 1.0);
        // Perfect balance: 2 tokens to each of 4 experts
        balancer.record_routing(&[0, 1, 2, 3, 0, 1, 2, 3]);
        let util = balancer.expert_utilization();
        for u in &util {
            assert!(
                (u - 1.0).abs() < 1e-5,
                "uniform routing should give utilization=1.0, got {u}"
            );
        }
    }

    #[test]
    fn test_expert_load_balancer_utilization_skewed() {
        let mut balancer = ExpertLoadBalancer::new(2, 1.0);
        // All 4 tokens go to expert 0 → util[0]=2.0, util[1]=0.0
        balancer.record_routing(&[0, 0, 0, 0]);
        let util = balancer.expert_utilization();
        assert_eq!(util.len(), 2);
        assert!(util[0] > 1.0, "overloaded expert should have util > 1.0");
        assert!(util[1] < 1e-5, "idle expert should have util ≈ 0.0");
    }

    #[test]
    fn test_expert_load_balancer_is_overloaded() {
        let mut balancer = ExpertLoadBalancer::new(2, 0.5);
        // 4 tokens, 2 experts → mean = 2.0, capacity = ceil(0.5 * 2.0) = 1
        balancer.record_routing(&[0, 0, 1, 1]);
        // expert_counts = [2, 2], capacity = 1 → both should be overloaded
        assert!(
            balancer.is_overloaded(0),
            "expert 0 with count 2 > capacity 1 should be overloaded"
        );
        assert!(
            balancer.is_overloaded(1),
            "expert 1 with count 2 > capacity 1 should be overloaded"
        );
    }

    #[test]
    fn test_expert_load_balancer_not_overloaded() {
        let mut balancer = ExpertLoadBalancer::new(2, 2.0);
        // 4 tokens, 2 experts → mean = 2.0, capacity = ceil(2.0 * 2.0) = 4
        balancer.record_routing(&[0, 1, 0, 1]);
        assert!(!balancer.is_overloaded(0), "expert 0 should not be overloaded");
        assert!(!balancer.is_overloaded(1), "expert 1 should not be overloaded");
    }

    #[test]
    fn test_expert_load_balancer_overflow_tokens_none() {
        let mut balancer = ExpertLoadBalancer::new(4, 2.0);
        // 8 tokens, capacity = 4 → no overflow for uniform routing
        balancer.record_routing(&[0, 1, 2, 3, 0, 1, 2, 3]);
        let overflow = balancer.overflow_tokens(&[0, 1, 2, 3, 0, 1, 2, 3]);
        assert_eq!(overflow, 0, "no overflow expected for balanced routing");
    }

    #[test]
    fn test_expert_load_balancer_overflow_tokens_some() {
        let mut balancer = ExpertLoadBalancer::new(2, 0.5);
        // 4 tokens, capacity = 1 → 2 overflows if all go to expert 0
        balancer.record_routing(&[0, 0, 0, 0]);
        // Now test overflow: 4 tokens to expert 0, capacity 1 → 3 overflows
        let overflow = balancer.overflow_tokens(&[0, 0, 0, 0]);
        assert_eq!(overflow, 3, "3 tokens should overflow capacity=1");
    }

    #[test]
    fn test_expert_load_balancer_load_balance_loss_uniform() {
        let balancer = ExpertLoadBalancer::new(4, 1.0);
        // Uniform router: each token assigns equal probability to all 4 experts
        // With 4 tokens and uniform probabilities:
        // f_i ≈ 0.25, P_i = 0.25 for all i
        // loss = 4 * (4 * 0.25 * 0.25) = 4 * 0.25 = 1.0
        let uniform: Vec<Vec<f32>> = (0..4)
            .map(|_| vec![0.25_f32; 4])
            .collect();
        let loss = balancer.load_balance_loss(&uniform);
        assert!(loss > 0.0, "load balance loss must be positive");
        assert!(loss.is_finite(), "loss must be finite");
    }

    #[test]
    fn test_expert_load_balancer_load_balance_loss_empty() {
        let balancer = ExpertLoadBalancer::new(4, 1.0);
        let loss = balancer.load_balance_loss(&[]);
        assert_eq!(loss, 0.0, "empty batch should give zero loss");
    }

    #[test]
    fn test_expert_load_balancer_reset() {
        let mut balancer = ExpertLoadBalancer::new(4, 1.0);
        balancer.record_routing(&[0, 1, 2, 3]);
        assert!(balancer.expert_counts.iter().any(|&c| c > 0));
        balancer.reset();
        assert!(
            balancer.expert_counts.iter().all(|&c| c == 0),
            "after reset all counts must be 0"
        );
        assert_eq!(balancer.expert_capacity, 0, "capacity must be reset to 0");
    }

    // ── expert_dropout ───────────────────────────────────────────────────────

    #[test]
    fn test_expert_dropout_inference_noop() {
        let outputs = vec![vec![1.0_f32, 2.0], vec![3.0, 4.0]];
        let result = expert_dropout(&outputs, 0.9, false, 42);
        assert_eq!(result, outputs, "dropout during inference must be a no-op");
    }

    #[test]
    fn test_expert_dropout_zero_prob_noop() {
        let outputs = vec![vec![1.0_f32, 2.0], vec![3.0, 4.0]];
        let result = expert_dropout(&outputs, 0.0, true, 42);
        assert_eq!(result, outputs, "zero drop probability must be a no-op");
    }

    #[test]
    fn test_expert_dropout_full_drop() {
        let outputs: Vec<Vec<f32>> = (0..10).map(|_| vec![1.0; 4]).collect();
        // With prob=1.0 every output should be zeroed
        let result = expert_dropout(&outputs, 1.0, true, 123);
        for row in &result {
            assert!(
                row.iter().all(|&x| x == 0.0),
                "100% dropout must zero all outputs"
            );
        }
    }

    #[test]
    fn test_expert_dropout_preserves_shape() {
        let outputs: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32; 8]).collect();
        let result = expert_dropout(&outputs, 0.5, true, 7);
        assert_eq!(result.len(), outputs.len(), "output count must be preserved");
        for (orig, dropped) in outputs.iter().zip(result.iter()) {
            assert_eq!(orig.len(), dropped.len(), "vector length must be preserved");
        }
    }

    #[test]
    fn test_expert_dropout_deterministic() {
        let outputs: Vec<Vec<f32>> = (0..8).map(|i| vec![i as f32; 4]).collect();
        let r1 = expert_dropout(&outputs, 0.5, true, 99);
        let r2 = expert_dropout(&outputs, 0.5, true, 99);
        assert_eq!(r1, r2, "same seed must produce identical results");
    }

    // ── upcycle_dense_to_moe ─────────────────────────────────────────────────

    #[test]
    fn test_upcycle_dense_to_moe_shape() {
        let dense = vec![0.1_f32; 16];
        let config = UpcyclingConfig::new(4, 16);
        let experts = upcycle_dense_to_moe(&dense, &config);
        assert_eq!(experts.len(), 4, "must produce num_experts expert weight vectors");
        for (i, e) in experts.iter().enumerate() {
            assert_eq!(e.len(), 16, "expert {} must have size expert_size", i);
        }
    }

    #[test]
    fn test_upcycle_dense_to_moe_experts_differ() {
        let dense = vec![0.5_f32; 8];
        let config = UpcyclingConfig {
            num_experts: 3,
            expert_size: 8,
            copy_dense_to_all: true,
            perturbation_std: 0.1,
        };
        let experts = upcycle_dense_to_moe(&dense, &config);
        // All experts start from the same dense weights but perturbation should
        // cause them to differ
        assert_ne!(
            experts[0], experts[1],
            "different experts must have different weights after perturbation"
        );
        assert_ne!(experts[1], experts[2]);
    }

    #[test]
    fn test_upcycle_dense_to_moe_close_to_base() {
        let dense = vec![1.0_f32; 8];
        let config = UpcyclingConfig {
            num_experts: 2,
            expert_size: 8,
            copy_dense_to_all: true,
            perturbation_std: 0.001,
        };
        let experts = upcycle_dense_to_moe(&dense, &config);
        for expert in &experts {
            for (&w, &base) in expert.iter().zip(dense.iter()) {
                assert!(
                    (w - base).abs() < 0.1,
                    "with small perturbation expert weight {} should be close to base {}",
                    w, base
                );
            }
        }
    }

    #[test]
    fn test_upcycle_no_copy_zeros_non_first() {
        let dense = vec![1.0_f32; 6];
        let config = UpcyclingConfig {
            num_experts: 3,
            expert_size: 6,
            copy_dense_to_all: false,
            perturbation_std: 0.0,
        };
        let experts = upcycle_dense_to_moe(&dense, &config);
        // expert 0 should be close to dense
        for (&w, &base) in experts[0].iter().zip(dense.iter()) {
            assert!((w - base).abs() < 1e-6, "expert 0 should match dense");
        }
        // experts 1 and 2 should be ~0 (zero base, no perturbation)
        for (expert_idx, expert) in experts.iter().enumerate().skip(1) {
            for &w in expert {
                assert!(
                    w.abs() < 1e-6,
                    "expert {} with no copy and zero perturbation should be 0.0, got {w}",
                    expert_idx
                );
            }
        }
    }
}
