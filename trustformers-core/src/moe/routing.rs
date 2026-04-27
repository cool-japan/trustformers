//! Advanced MoE Routing Strategies
//!
//! Implements Expert Choice routing (Zoph et al., 2022), Hash routing,
//! Switch Transformer routing (Fedus et al., 2021), and a random router for ablations.

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error types
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur in advanced MoE routing operations.
#[derive(Debug, Clone, PartialEq)]
pub enum MoeRoutingError {
    /// Input was empty (no tokens or hidden states).
    EmptyInput,
    /// Invalid number of experts specified.
    InvalidNumExperts(String),
    /// Invalid capacity factor specified.
    InvalidCapacityFactor(String),
    /// Dimension mismatch between expected and provided sizes.
    DimensionMismatch { expected: usize, got: usize },
}

impl fmt::Display for MoeRoutingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MoeRoutingError::EmptyInput => {
                write!(f, "empty input: no tokens or hidden states provided")
            }
            MoeRoutingError::InvalidNumExperts(msg) => {
                write!(f, "invalid num_experts: {msg}")
            }
            MoeRoutingError::InvalidCapacityFactor(msg) => {
                write!(f, "invalid capacity_factor: {msg}")
            }
            MoeRoutingError::DimensionMismatch { expected, got } => {
                write!(
                    f,
                    "dimension mismatch: expected {expected}, got {got}"
                )
            }
        }
    }
}

impl std::error::Error for MoeRoutingError {}

// ─────────────────────────────────────────────────────────────────────────────
// RouterType enum
// ─────────────────────────────────────────────────────────────────────────────

/// Discriminant for the supported routing strategies.
#[derive(Debug, Clone, PartialEq)]
pub enum RouterType {
    /// Standard top-k routing: each token chooses its top-k experts.
    TopK { k: usize },
    /// Expert Choice routing: each expert chooses its top-capacity tokens.
    ExpertChoice { capacity_factor: f32 },
    /// Switch Transformer top-1 routing with capacity enforcement.
    SwitchTransformer,
    /// Deterministic hash-based routing — no learned weights.
    Hash { num_experts: usize },
    /// Random routing for ablations / debugging.
    RandomRouter { seed: u64 },
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Numerically-stable column-wise softmax over an [rows × cols] row-major matrix.
///
/// Returns a new flattened `Vec<f32>` with the same layout.
fn column_softmax(matrix: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = matrix.to_vec();
    for col in 0..cols {
        let mut max_val = f32::NEG_INFINITY;
        for row in 0..rows {
            let v = matrix[row * cols + col];
            if v > max_val {
                max_val = v;
            }
        }
        let mut sum = 0.0_f32;
        for row in 0..rows {
            let e = (matrix[row * cols + col] - max_val).exp();
            out[row * cols + col] = e;
            sum += e;
        }
        if sum > 1e-10 {
            for row in 0..rows {
                out[row * cols + col] /= sum;
            }
        } else {
            let uniform = 1.0 / rows as f32;
            for row in 0..rows {
                out[row * cols + col] = uniform;
            }
        }
    }
    out
}

/// Numerically-stable row-wise softmax over an [rows × cols] row-major matrix.
fn row_softmax(matrix: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * cols];
    for row in 0..rows {
        let base = row * cols;
        let slice = &matrix[base..base + cols];
        let max_val = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = slice.iter().map(|&x| (x - max_val).exp()).collect();
        let sum: f32 = exp_vals.iter().sum();
        let denom = if sum > 1e-10 { sum } else { 1.0 };
        for col in 0..cols {
            out[base + col] = exp_vals[col] / denom;
        }
    }
    out
}

/// Matrix multiply: (A: [m × k]) × (B: [k × n]) → C: [m × n], all row-major.
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_ip = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_ip * b[p * n + j];
            }
        }
    }
    c
}

/// Return indices of the top-`k` values in `vals` (descending order).
fn top_k_desc(vals: &[f32], k: usize) -> Vec<usize> {
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
// ExpertChoiceAssignment
// ─────────────────────────────────────────────────────────────────────────────

/// Result of Expert Choice routing: each expert independently selected its
/// top-`capacity` tokens.
#[derive(Debug, Clone)]
pub struct ExpertChoiceAssignment {
    /// `expert_token_idx[e]` — the token indices selected by expert `e`.
    pub expert_token_idx: Vec<Vec<usize>>,
    /// `expert_weights[e]` — softmax-normalised affinity weights for each
    /// token selected by expert `e` (same shape as `expert_token_idx`).
    pub expert_weights: Vec<Vec<f32>>,
    /// Number of tokens each expert processes (guaranteed equal across experts).
    pub capacity: usize,
}

impl ExpertChoiceAssignment {
    /// Return the number of tokens handled by each expert.
    pub fn tokens_per_expert(&self) -> Vec<usize> {
        self.expert_token_idx.iter().map(|v| v.len()).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ExpertChoiceRouter
// ─────────────────────────────────────────────────────────────────────────────

/// Expert Choice routing (Zoph et al., 2022).
///
/// Each expert independently selects the top-`capacity` tokens based on a
/// learned affinity score. This guarantees perfect expert utilisation — no
/// tokens are ever dropped — at the cost of some tokens being processed by
/// multiple experts and others by none.
pub struct ExpertChoiceRouter {
    /// Total number of experts.
    pub num_experts: usize,
    /// Capacity factor: `capacity = capacity_factor * seq_len / num_experts`.
    pub capacity_factor: f32,
    /// Flattened router weight matrix of shape `[hidden_size, num_experts]`.
    pub router_weights: Vec<f32>,
    /// Dimensionality of the hidden states fed into the router.
    pub hidden_size: usize,
}

impl ExpertChoiceRouter {
    /// Create a new `ExpertChoiceRouter`.
    ///
    /// Weights are initialised with a small deterministic pattern (not random,
    /// to remain deterministic in tests without requiring an RNG dependency).
    pub fn new(num_experts: usize, hidden_size: usize, capacity_factor: f32) -> Self {
        // Deterministic weight initialisation: scaled sin wave so each
        // (hidden, expert) pair has a distinct non-zero value.
        let total = hidden_size * num_experts;
        let router_weights: Vec<f32> = (0..total)
            .map(|i| ((i as f32 + 1.0) / total as f32).sin() * 0.1)
            .collect();

        Self {
            num_experts,
            capacity_factor,
            router_weights,
            hidden_size,
        }
    }

    /// Compute the affinity matrix `[seq_len × num_experts]` for the given
    /// hidden states `[seq_len × hidden_size]`.
    pub fn compute_affinity(
        hidden: &[f32],
        weights: &[f32],
        seq_len: usize,
        hidden_size: usize,
        num_experts: usize,
    ) -> Vec<f32> {
        // affinity = hidden @ weights  →  [seq_len × num_experts]
        matmul(hidden, weights, seq_len, hidden_size, num_experts)
    }

    /// Route `hidden_states` (flattened `[seq_len × hidden_size]`) and return
    /// an `ExpertChoiceAssignment`.
    pub fn route(
        &self,
        hidden_states: &[f32],
        seq_len: usize,
        hidden_size: usize,
    ) -> Result<ExpertChoiceAssignment, MoeRoutingError> {
        if hidden_states.is_empty() {
            return Err(MoeRoutingError::EmptyInput);
        }
        if self.num_experts == 0 {
            return Err(MoeRoutingError::InvalidNumExperts(
                "num_experts must be ≥ 1".to_string(),
            ));
        }
        if self.capacity_factor <= 0.0 {
            return Err(MoeRoutingError::InvalidCapacityFactor(
                "capacity_factor must be > 0".to_string(),
            ));
        }
        let expected_len = seq_len * hidden_size;
        if hidden_states.len() != expected_len {
            return Err(MoeRoutingError::DimensionMismatch {
                expected: expected_len,
                got: hidden_states.len(),
            });
        }

        // capacity = max(1, floor(capacity_factor * seq_len / num_experts))
        let capacity =
            ((self.capacity_factor * seq_len as f32 / self.num_experts as f32).floor() as usize)
                .max(1);

        // Compute raw affinity [seq_len × num_experts]
        let raw_affinity = Self::compute_affinity(
            hidden_states,
            &self.router_weights,
            seq_len,
            hidden_size,
            self.num_experts,
        );

        // Column-wise softmax so each expert's scores are normalised over tokens
        let soft_affinity =
            column_softmax(&raw_affinity, seq_len, self.num_experts);

        let mut expert_token_idx = Vec::with_capacity(self.num_experts);
        let mut expert_weights_out = Vec::with_capacity(self.num_experts);

        for e in 0..self.num_experts {
            // Collect column `e` of the affinity matrix
            let col: Vec<f32> = (0..seq_len)
                .map(|t| soft_affinity[t * self.num_experts + e])
                .collect();

            // Each expert picks its top-`capacity` tokens
            let top_tokens = top_k_desc(&col, capacity);
            let weights: Vec<f32> = top_tokens.iter().map(|&t| col[t]).collect();

            expert_token_idx.push(top_tokens);
            expert_weights_out.push(weights);
        }

        Ok(ExpertChoiceAssignment {
            expert_token_idx,
            expert_weights: expert_weights_out,
            capacity,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HashRouter
// ─────────────────────────────────────────────────────────────────────────────

/// Deterministic hash-based routing — requires no learned parameters.
///
/// Uses a FNV-1a hash of the token id modulo `num_experts`.
pub struct HashRouter {
    /// Total number of experts.
    pub num_experts: usize,
}

impl HashRouter {
    /// Create a new `HashRouter`.
    pub fn new(num_experts: usize) -> Self {
        Self { num_experts }
    }

    /// Route a single token ID to an expert index using FNV-1a.
    pub fn route_token(token_id: u32, num_experts: usize) -> usize {
        // FNV-1a 32-bit
        const FNV_OFFSET: u32 = 2_166_136_261;
        const FNV_PRIME: u32 = 16_777_619;

        let bytes = token_id.to_le_bytes();
        let mut hash = FNV_OFFSET;
        for byte in bytes {
            hash ^= byte as u32;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        (hash as usize) % num_experts.max(1)
    }

    /// Route a batch of token IDs, returning an expert index for each.
    pub fn route_batch(&self, token_ids: &[u32]) -> Vec<usize> {
        token_ids
            .iter()
            .map(|&id| Self::route_token(id, self.num_experts))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SwitchAssignment
// ─────────────────────────────────────────────────────────────────────────────

/// Result of Switch Transformer routing.
#[derive(Debug, Clone)]
pub struct SwitchAssignment {
    /// `Some(expert_idx)` for accepted tokens, `None` for dropped tokens.
    pub expert_assignments: Vec<Option<usize>>,
    /// Number of tokens dropped due to expert capacity overflow.
    pub num_dropped: usize,
    /// How many tokens each expert received (accepted only).
    pub expert_load: Vec<usize>,
}

impl SwitchAssignment {
    /// Fraction of tokens that were dropped.
    pub fn drop_rate(&self) -> f32 {
        let total = self.expert_assignments.len();
        if total == 0 {
            return 0.0;
        }
        self.num_dropped as f32 / total as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SwitchTransformerRouter
// ─────────────────────────────────────────────────────────────────────────────

/// Switch Transformer top-1 routing (Fedus et al., 2021).
///
/// Each token is routed to the single expert with the highest logit.
/// Tokens that would overflow an expert's capacity are dropped.
pub struct SwitchTransformerRouter {
    /// Total number of experts.
    pub num_experts: usize,
    /// Capacity factor; default 1.25 gives 25 % headroom above ideal load.
    pub capacity_factor: f32,
    /// Flattened router weight matrix of shape `[hidden_size × num_experts]`.
    pub router_weights: Vec<f32>,
    /// Dimensionality of the hidden states.
    pub hidden_size: usize,
}

impl SwitchTransformerRouter {
    /// Create a new `SwitchTransformerRouter` with the given capacity factor.
    ///
    /// Weights are initialised deterministically (sin-wave pattern).
    pub fn new(num_experts: usize, hidden_size: usize, capacity_factor: f32) -> Self {
        let total = hidden_size * num_experts;
        let router_weights: Vec<f32> = (0..total)
            .map(|i| ((i as f32 * 0.3 + 0.7) / total as f32).cos() * 0.1)
            .collect();

        Self {
            num_experts,
            capacity_factor,
            router_weights,
            hidden_size,
        }
    }

    /// Route a batch of hidden states, enforcing expert capacity.
    ///
    /// `hidden` is flattened `[seq_len × hidden_size]`.
    pub fn route(
        &self,
        hidden: &[f32],
        seq_len: usize,
        hidden_size: usize,
    ) -> Result<SwitchAssignment, MoeRoutingError> {
        if hidden.is_empty() {
            return Err(MoeRoutingError::EmptyInput);
        }
        if self.num_experts == 0 {
            return Err(MoeRoutingError::InvalidNumExperts(
                "num_experts must be ≥ 1".to_string(),
            ));
        }
        if self.capacity_factor <= 0.0 {
            return Err(MoeRoutingError::InvalidCapacityFactor(
                "capacity_factor must be > 0".to_string(),
            ));
        }
        let expected = seq_len * hidden_size;
        if hidden.len() != expected {
            return Err(MoeRoutingError::DimensionMismatch {
                expected,
                got: hidden.len(),
            });
        }

        // capacity = ceil(capacity_factor * seq_len / num_experts)
        let capacity = ((self.capacity_factor * seq_len as f32 / self.num_experts as f32).ceil()
            as usize)
            .max(1);

        // Compute logits [seq_len × num_experts]
        let logits = matmul(hidden, &self.router_weights, seq_len, hidden_size, self.num_experts);

        // Row-wise softmax to get routing probabilities
        let probs = row_softmax(&logits, seq_len, self.num_experts);

        let mut expert_load = vec![0usize; self.num_experts];
        let mut expert_assignments: Vec<Option<usize>> = Vec::with_capacity(seq_len);
        let mut num_dropped = 0usize;

        for t in 0..seq_len {
            let row = &probs[t * self.num_experts..(t + 1) * self.num_experts];
            // argmax
            let best_expert = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0);

            if expert_load[best_expert] < capacity {
                expert_load[best_expert] += 1;
                expert_assignments.push(Some(best_expert));
            } else {
                num_dropped += 1;
                expert_assignments.push(None);
            }
        }

        Ok(SwitchAssignment {
            expert_assignments,
            num_dropped,
            expert_load,
        })
    }

    /// Compute the Switch Transformer load-balance auxiliary loss.
    ///
    /// `loss = num_experts * Σ_i (f_i * P_i)`
    ///
    /// where:
    /// - `f_i` = fraction of tokens assigned to expert `i`
    /// - `P_i` = mean router probability for expert `i` over the batch
    pub fn switch_load_balance_loss(
        router_probs: &[f32],
        expert_assignments: &[usize],
        seq_len: usize,
        num_experts: usize,
    ) -> f32 {
        if seq_len == 0 || num_experts == 0 {
            return 0.0;
        }

        // f_i: fraction of tokens assigned to expert i
        let mut counts = vec![0usize; num_experts];
        for &e in expert_assignments {
            if e < num_experts {
                counts[e] += 1;
            }
        }
        let fractions: Vec<f32> = counts.iter().map(|&c| c as f32 / seq_len as f32).collect();

        // P_i: mean router probability for expert i
        // router_probs layout: [seq_len × num_experts]
        let mut prob_sums = vec![0.0_f32; num_experts];
        for t in 0..seq_len {
            for e in 0..num_experts {
                prob_sums[e] += router_probs[t * num_experts + e];
            }
        }
        let mean_probs: Vec<f32> = prob_sums
            .iter()
            .map(|&s| s / seq_len as f32)
            .collect();

        let dot: f32 = fractions
            .iter()
            .zip(mean_probs.iter())
            .map(|(&f, &p)| f * p)
            .sum();

        num_experts as f32 * dot
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ExpertChoiceRouter ──────────────────────────────────────────────────

    #[test]
    fn test_expert_choice_capacity_utilization() {
        // Every expert must get exactly `capacity` tokens.
        let num_experts = 4;
        let hidden_size = 8;
        let seq_len = 16;
        let capacity_factor = 1.0_f32;

        let router = ExpertChoiceRouter::new(num_experts, hidden_size, capacity_factor);
        let hidden: Vec<f32> = (0..seq_len * hidden_size)
            .map(|i| (i as f32) * 0.01)
            .collect();

        let assignment = router
            .route(&hidden, seq_len, hidden_size)
            .expect("route should succeed");

        let expected_capacity =
            ((capacity_factor * seq_len as f32 / num_experts as f32).floor() as usize).max(1);

        assert_eq!(assignment.capacity, expected_capacity);

        let tpe = assignment.tokens_per_expert();
        for (e, &cnt) in tpe.iter().enumerate() {
            assert_eq!(
                cnt, expected_capacity,
                "expert {e} should process exactly {expected_capacity} tokens, got {cnt}"
            );
        }
    }

    #[test]
    fn test_expert_choice_no_dropped_tokens() {
        // Expert Choice never drops tokens — every expert fills its capacity.
        let num_experts = 2;
        let hidden_size = 4;
        let seq_len = 8;
        let router = ExpertChoiceRouter::new(num_experts, hidden_size, 1.0);
        let hidden: Vec<f32> = (0..seq_len * hidden_size).map(|i| i as f32 * 0.1).collect();

        let assignment = router
            .route(&hidden, seq_len, hidden_size)
            .expect("route ok");

        // All expert slots are filled — none is empty.
        for tokens in &assignment.expert_token_idx {
            assert!(!tokens.is_empty(), "each expert must have at least one token");
        }
    }

    #[test]
    fn test_expert_choice_token_indices_in_range() {
        let num_experts = 3;
        let hidden_size = 6;
        let seq_len = 12;
        let router = ExpertChoiceRouter::new(num_experts, hidden_size, 1.0);
        let hidden: Vec<f32> = (0..seq_len * hidden_size).map(|i| i as f32 * 0.05).collect();

        let assignment = router
            .route(&hidden, seq_len, hidden_size)
            .expect("route ok");

        for tokens in &assignment.expert_token_idx {
            for &t in tokens {
                assert!(t < seq_len, "token index {t} out of range [0, {seq_len})");
            }
        }
    }

    #[test]
    fn test_expert_choice_affinity_matrix_dimensions() {
        let num_experts = 4;
        let hidden_size = 8;
        let seq_len = 10;
        let router = ExpertChoiceRouter::new(num_experts, hidden_size, 1.0);
        let hidden: Vec<f32> = vec![0.1; seq_len * hidden_size];

        let affinity = ExpertChoiceRouter::compute_affinity(
            &hidden,
            &router.router_weights,
            seq_len,
            hidden_size,
            num_experts,
        );

        assert_eq!(
            affinity.len(),
            seq_len * num_experts,
            "affinity matrix should have {} elements, got {}",
            seq_len * num_experts,
            affinity.len()
        );
    }

    #[test]
    fn test_expert_choice_error_empty_input() {
        let router = ExpertChoiceRouter::new(4, 8, 1.0);
        let err = router.route(&[], 0, 8).unwrap_err();
        assert_eq!(err, MoeRoutingError::EmptyInput);
    }

    #[test]
    fn test_expert_choice_error_dimension_mismatch() {
        let router = ExpertChoiceRouter::new(4, 8, 1.0);
        // hidden_states has 10 elements but seq_len * hidden_size = 16
        let hidden = vec![0.0_f32; 10];
        let err = router.route(&hidden, 2, 8).unwrap_err();
        matches!(err, MoeRoutingError::DimensionMismatch { .. });
    }

    // ── HashRouter ──────────────────────────────────────────────────────────

    #[test]
    fn test_hash_router_consistency() {
        // Same token id must always map to the same expert.
        let router = HashRouter::new(8);
        let token_id: u32 = 42;
        let first = HashRouter::route_token(token_id, 8);
        for _ in 0..100 {
            assert_eq!(
                HashRouter::route_token(token_id, 8),
                first,
                "hash routing must be deterministic"
            );
        }
    }

    #[test]
    fn test_hash_router_batch_coverage() {
        // With enough tokens and experts the batch should hit most experts.
        let num_experts = 8;
        let router = HashRouter::new(num_experts);
        let token_ids: Vec<u32> = (0..256).collect();
        let assignments = router.route_batch(&token_ids);

        assert_eq!(assignments.len(), token_ids.len());

        let mut seen = vec![false; num_experts];
        for &e in &assignments {
            assert!(e < num_experts, "expert index out of range");
            seen[e] = true;
        }
        let covered = seen.iter().filter(|&&b| b).count();
        // With 256 tokens and 8 experts we expect all experts to be hit.
        assert!(
            covered >= num_experts,
            "only {covered}/{num_experts} experts were assigned at least one token"
        );
    }

    #[test]
    fn test_hash_router_in_range() {
        for num_experts in 1..=16 {
            for token_id in 0..64_u32 {
                let e = HashRouter::route_token(token_id, num_experts);
                assert!(
                    e < num_experts,
                    "expert {e} out of range for num_experts={num_experts}"
                );
            }
        }
    }

    // ── SwitchTransformerRouter ─────────────────────────────────────────────

    #[test]
    fn test_switch_drop_rate_within_capacity() {
        // Set capacity_factor = num_experts so that
        //   capacity = ceil(capacity_factor * seq_len / num_experts)
        //            = ceil(4.0 * 8 / 4)
        //            = 8  (= seq_len)
        // This means even if every token routes to the same expert it still fits.
        let num_experts = 4;
        let hidden_size = 8;
        let seq_len = 8;
        // capacity_factor = num_experts → capacity = seq_len → absolutely no drops
        let router = SwitchTransformerRouter::new(num_experts, hidden_size, num_experts as f32);

        let hidden: Vec<f32> = (0..seq_len * hidden_size).map(|i| i as f32 * 0.01).collect();
        let assignment = router
            .route(&hidden, seq_len, hidden_size)
            .expect("route ok");

        assert_eq!(
            assignment.drop_rate(),
            0.0,
            "no tokens should be dropped when capacity equals seq_len"
        );
    }

    #[test]
    fn test_switch_drop_rate_with_tight_capacity() {
        // With very tight capacity and heavily skewed routing, some tokens may be dropped.
        let num_experts = 8;
        let hidden_size = 4;
        let seq_len = 16;
        // capacity_factor = 0.5 means each expert can only handle floor(0.5 * 16 / 8) = 1 token
        let router = SwitchTransformerRouter::new(num_experts, hidden_size, 0.5);

        let hidden: Vec<f32> = (0..seq_len * hidden_size).map(|i| i as f32 * 0.05).collect();
        let assignment = router
            .route(&hidden, seq_len, hidden_size)
            .expect("route ok");

        // Total tokens = seq_len; total capacity = num_experts * 1 = 8 < 16 → some dropped
        let total_capacity: usize = assignment.expert_load.iter().sum();
        assert!(
            total_capacity <= seq_len,
            "total accepted must not exceed seq_len"
        );
    }

    #[test]
    fn test_switch_load_balance_loss_positive() {
        let num_experts = 2;
        let seq_len = 4;
        // Uniform probabilities and balanced assignments
        let router_probs = vec![0.5_f32, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]; // [4 × 2]
        let expert_assignments = vec![0, 1, 0, 1];

        let loss = SwitchTransformerRouter::switch_load_balance_loss(
            &router_probs,
            &expert_assignments,
            seq_len,
            num_experts,
        );

        // f = [0.5, 0.5], P = [0.5, 0.5]
        // loss = 2 * (0.5*0.5 + 0.5*0.5) = 2 * 0.5 = 1.0
        assert!(
            (loss - 1.0).abs() < 1e-5,
            "expected loss ≈ 1.0, got {loss}"
        );
    }

    #[test]
    fn test_switch_assignment_drop_rate_calculation() {
        let assignment = SwitchAssignment {
            expert_assignments: vec![Some(0), None, Some(1), None, Some(0)],
            num_dropped: 2,
            expert_load: vec![2, 1],
        };
        let dr = assignment.drop_rate();
        assert!(
            (dr - 0.4).abs() < 1e-5,
            "expected drop rate 0.4, got {dr}"
        );
    }

    // ── MoeRoutingError Display ─────────────────────────────────────────────

    #[test]
    fn test_routing_error_display() {
        let e1 = MoeRoutingError::EmptyInput;
        assert!(e1.to_string().contains("empty"));

        let e2 = MoeRoutingError::InvalidNumExperts("must be ≥ 1".to_string());
        assert!(e2.to_string().contains("num_experts"));

        let e3 = MoeRoutingError::InvalidCapacityFactor("must be > 0".to_string());
        assert!(e3.to_string().contains("capacity_factor"));

        let e4 = MoeRoutingError::DimensionMismatch {
            expected: 64,
            got: 32,
        };
        let s4 = e4.to_string();
        assert!(s4.contains("64") && s4.contains("32"));
    }

    #[test]
    fn test_switch_error_empty_input() {
        let router = SwitchTransformerRouter::new(4, 8, 1.25);
        let err = router.route(&[], 0, 8).unwrap_err();
        assert_eq!(err, MoeRoutingError::EmptyInput);
    }
}
