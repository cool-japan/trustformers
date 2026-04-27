//! Expert Parallelism Simulator
//!
//! Simulates the all-to-all communication patterns that arise in expert
//! parallelism (EP), allowing users to reason about data flow, load balance,
//! and communication volume without a real distributed runtime.

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error types
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur in expert-parallel scheduling.
#[derive(Debug, Clone, PartialEq)]
pub enum EpError {
    /// `num_experts` must be divisible by `num_ep_ranks`.
    ExpertsDivisibility {
        num_experts: usize,
        num_ep_ranks: usize,
    },
    /// `num_ep_ranks` must be ≥ 1.
    InvalidNumRanks(String),
    /// `num_experts` must be ≥ 1.
    InvalidNumExperts(String),
    /// `hidden_size` must be ≥ 1.
    InvalidHiddenSize(String),
    /// The provided token assignments contained an expert index that is out of range.
    ExpertIndexOutOfRange { index: usize, num_experts: usize },
    /// Hidden states slice length did not match `seq_len * hidden_size`.
    DimensionMismatch { expected: usize, got: usize },
}

impl fmt::Display for EpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EpError::ExpertsDivisibility {
                num_experts,
                num_ep_ranks,
            } => write!(
                f,
                "num_experts ({num_experts}) must be divisible by num_ep_ranks ({num_ep_ranks})"
            ),
            EpError::InvalidNumRanks(msg) => write!(f, "invalid num_ep_ranks: {msg}"),
            EpError::InvalidNumExperts(msg) => write!(f, "invalid num_experts: {msg}"),
            EpError::InvalidHiddenSize(msg) => write!(f, "invalid hidden_size: {msg}"),
            EpError::ExpertIndexOutOfRange { index, num_experts } => write!(
                f,
                "expert index {index} is out of range [0, {num_experts})"
            ),
            EpError::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
        }
    }
}

impl std::error::Error for EpError {}

// ─────────────────────────────────────────────────────────────────────────────
// ExpertParallelConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for an expert-parallel deployment.
#[derive(Debug, Clone)]
pub struct ExpertParallelConfig {
    /// Total number of experts.
    pub num_experts: usize,
    /// Number of expert-parallel ranks (GPUs / processes).
    pub num_ep_ranks: usize,
    /// Number of experts assigned to each rank = `num_experts / num_ep_ranks`.
    pub experts_per_rank: usize,
    /// Dimensionality of the hidden states.
    pub hidden_size: usize,
    /// Maximum tokens a single expert may process; default 64.
    pub max_tokens_per_expert: usize,
}

impl ExpertParallelConfig {
    /// Create a new `ExpertParallelConfig` with default `max_tokens_per_expert = 64`.
    pub fn new(num_experts: usize, num_ep_ranks: usize, hidden_size: usize) -> Self {
        let experts_per_rank = if num_ep_ranks > 0 {
            num_experts / num_ep_ranks
        } else {
            0
        };
        Self {
            num_experts,
            num_ep_ranks,
            experts_per_rank,
            hidden_size,
            max_tokens_per_expert: 64,
        }
    }

    /// Validate the configuration, returning an error if any invariant is violated.
    pub fn validate(&self) -> Result<(), EpError> {
        if self.num_ep_ranks == 0 {
            return Err(EpError::InvalidNumRanks(
                "num_ep_ranks must be ≥ 1".to_string(),
            ));
        }
        if self.num_experts == 0 {
            return Err(EpError::InvalidNumExperts(
                "num_experts must be ≥ 1".to_string(),
            ));
        }
        if self.hidden_size == 0 {
            return Err(EpError::InvalidHiddenSize(
                "hidden_size must be ≥ 1".to_string(),
            ));
        }
        if self.num_experts % self.num_ep_ranks != 0 {
            return Err(EpError::ExpertsDivisibility {
                num_experts: self.num_experts,
                num_ep_ranks: self.num_ep_ranks,
            });
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AllToAllPlan
// ─────────────────────────────────────────────────────────────────────────────

/// A planned all-to-all communication pattern.
///
/// `send_counts[src][dst]` is the number of tokens that rank `src` sends to rank `dst`.
#[derive(Debug, Clone)]
pub struct AllToAllPlan {
    /// `send_counts[src][dst]` = tokens sent from rank `src` to rank `dst`.
    pub send_counts: Vec<Vec<usize>>,
    /// `recv_counts[dst][src]` = tokens received at rank `dst` from rank `src`.
    pub recv_counts: Vec<Vec<usize>>,
}

impl AllToAllPlan {
    /// Total number of tokens moved across all rank pairs.
    pub fn total_tokens_moved(&self) -> usize {
        self.send_counts
            .iter()
            .enumerate()
            .flat_map(|(src, row)| {
                row.iter()
                    .enumerate()
                    .filter(move |&(dst, _)| dst != src)
                    .map(|(_, &cnt)| cnt)
            })
            .sum()
    }

    /// `true` if the maximum send count is at most `2 × mean send count`.
    pub fn is_balanced(&self) -> bool {
        self.max_imbalance_ratio() <= 2.0
    }

    /// Ratio of maximum send volume to mean send volume across all ranks.
    ///
    /// Returns `1.0` when all ranks send exactly the same amount.
    pub fn max_imbalance_ratio(&self) -> f32 {
        let num_ranks = self.send_counts.len();
        if num_ranks == 0 {
            return 1.0;
        }

        // Per-rank total outgoing token count
        let rank_totals: Vec<usize> = self
            .send_counts
            .iter()
            .enumerate()
            .map(|(src, row)| {
                row.iter()
                    .enumerate()
                    .filter(|&(dst, _)| dst != src)
                    .map(|(_, &c)| c)
                    .sum()
            })
            .collect();

        let total: usize = rank_totals.iter().sum();
        if total == 0 {
            return 1.0;
        }

        let mean = total as f32 / num_ranks as f32;
        let max = *rank_totals.iter().max().unwrap_or(&0) as f32;

        if mean < 1e-10 {
            1.0
        } else {
            max / mean
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CommunicationVolume
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate communication volume statistics for a planned all-to-all exchange.
#[derive(Debug, Clone)]
pub struct CommunicationVolume {
    /// Total bytes transferred across all rank pairs (both directions).
    pub total_bytes: usize,
    /// Maximum bytes sent or received by any single rank.
    pub max_bytes_any_rank: usize,
    /// Mean bytes per rank.
    pub mean_bytes_per_rank: f32,
    /// `total / (num_ranks * max)` — measures how evenly bandwidth is used.
    pub bandwidth_utilization: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// AllToAllCommunication
// ─────────────────────────────────────────────────────────────────────────────

/// Utilities for planning and analysing all-to-all communication patterns.
pub struct AllToAllCommunication;

impl AllToAllCommunication {
    /// Build an `AllToAllPlan` from a flat slice of expert assignments.
    ///
    /// `assignments[i]` is the expert index that token `i` should be sent to.
    /// Tokens are partitioned across `num_ranks` ranks that own contiguous
    /// blocks of experts (`experts_per_rank = num_experts / num_ranks`).
    /// Rank 0 owns experts `[0, experts_per_rank)`, rank 1 owns
    /// `[experts_per_rank, 2*experts_per_rank)`, etc.
    ///
    /// Assumes the calling token stream originates from rank 0 for simulation
    /// purposes.
    pub fn plan_all_to_all(
        assignments: &[usize],
        num_ranks: usize,
        num_experts: usize,
    ) -> AllToAllPlan {
        let experts_per_rank = if num_ranks > 0 {
            (num_experts + num_ranks - 1) / num_ranks // ceiling division
        } else {
            1
        };

        let mut send_counts = vec![vec![0usize; num_ranks]; num_ranks];

        for &expert in assignments {
            let dst_rank = if experts_per_rank > 0 {
                (expert / experts_per_rank).min(num_ranks.saturating_sub(1))
            } else {
                0
            };
            // Tokens originate at rank 0 in this simulation
            send_counts[0][dst_rank] += 1;
        }

        // recv_counts is the transpose of send_counts
        let mut recv_counts = vec![vec![0usize; num_ranks]; num_ranks];
        for src in 0..num_ranks {
            for dst in 0..num_ranks {
                recv_counts[dst][src] = send_counts[src][dst];
            }
        }

        AllToAllPlan {
            send_counts,
            recv_counts,
        }
    }

    /// Compute communication volume statistics from a plan.
    ///
    /// `bytes_per_token = hidden_size * size_of(f32)` (assumed 4 bytes).
    pub fn simulate_communication_volume(
        plan: &AllToAllPlan,
        hidden_size: usize,
    ) -> CommunicationVolume {
        let bytes_per_token = hidden_size * 4; // f32 = 4 bytes
        let num_ranks = plan.send_counts.len();

        // Per-rank outgoing bytes
        let rank_send_bytes: Vec<usize> = plan
            .send_counts
            .iter()
            .enumerate()
            .map(|(src, row)| {
                row.iter()
                    .enumerate()
                    .filter(|&(dst, _)| dst != src)
                    .map(|(_, &c)| c * bytes_per_token)
                    .sum()
            })
            .collect();

        let total_bytes: usize = rank_send_bytes.iter().sum();
        let max_bytes_any_rank = *rank_send_bytes.iter().max().unwrap_or(&0);
        let mean_bytes_per_rank = if num_ranks > 0 {
            total_bytes as f32 / num_ranks as f32
        } else {
            0.0
        };

        let bandwidth_utilization = if num_ranks > 0 && max_bytes_any_rank > 0 {
            total_bytes as f32 / (num_ranks as f32 * max_bytes_any_rank as f32)
        } else {
            0.0
        };

        CommunicationVolume {
            total_bytes,
            max_bytes_any_rank,
            mean_bytes_per_rank,
            bandwidth_utilization,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// EpScheduleResult
// ─────────────────────────────────────────────────────────────────────────────

/// Result of an expert-parallel schedule simulation.
#[derive(Debug, Clone)]
pub struct EpScheduleResult {
    /// The all-to-all communication plan.
    pub all_to_all_plan: AllToAllPlan,
    /// Aggregate communication volume.
    pub communication_volume: CommunicationVolume,
    /// Number of tokens processed by each rank's local experts.
    pub local_token_counts: Vec<usize>,
    /// `true` if the load balance criterion is met.
    pub is_load_balanced: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// ExpertParallelScheduler
// ─────────────────────────────────────────────────────────────────────────────

/// Simulates expert-parallel scheduling for a given configuration.
pub struct ExpertParallelScheduler {
    /// Configuration for this scheduler.
    pub config: ExpertParallelConfig,
}

impl ExpertParallelScheduler {
    /// Create a new scheduler, validating the configuration.
    pub fn new(config: ExpertParallelConfig) -> Result<Self, EpError> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Return the expert indices assigned to the given `rank`.
    ///
    /// Rank `r` handles experts `[r * experts_per_rank, (r+1) * experts_per_rank)`.
    pub fn local_expert_indices(&self, rank: usize) -> Vec<usize> {
        let start = rank * self.config.experts_per_rank;
        let end = start + self.config.experts_per_rank;
        (start..end.min(self.config.num_experts)).collect()
    }

    /// Simulate a full scheduling pass for the given expert `assignments`.
    ///
    /// `hidden` is a flattened `[seq_len × hidden_size]` matrix (used only for
    /// size validation; no computation is performed on the values).
    pub fn schedule(
        &self,
        assignments: &[usize],
        hidden: &[f32],
        seq_len: usize,
    ) -> Result<EpScheduleResult, EpError> {
        // Validate hidden states dimensions
        let expected_hidden_len = seq_len * self.config.hidden_size;
        if hidden.len() != expected_hidden_len {
            return Err(EpError::DimensionMismatch {
                expected: expected_hidden_len,
                got: hidden.len(),
            });
        }

        // Validate assignments
        for &e in assignments {
            if e >= self.config.num_experts {
                return Err(EpError::ExpertIndexOutOfRange {
                    index: e,
                    num_experts: self.config.num_experts,
                });
            }
        }

        // Build the all-to-all plan
        let all_to_all_plan = AllToAllCommunication::plan_all_to_all(
            assignments,
            self.config.num_ep_ranks,
            self.config.num_experts,
        );

        // Simulate communication volume
        let communication_volume = AllToAllCommunication::simulate_communication_volume(
            &all_to_all_plan,
            self.config.hidden_size,
        );

        // Count how many tokens end up at each rank
        let mut local_token_counts = vec![0usize; self.config.num_ep_ranks];
        for &e in assignments {
            let rank = e / self.config.experts_per_rank;
            if rank < self.config.num_ep_ranks {
                local_token_counts[rank] += 1;
            }
        }

        let is_load_balanced = all_to_all_plan.is_balanced();

        Ok(EpScheduleResult {
            all_to_all_plan,
            communication_volume,
            local_token_counts,
            is_load_balanced,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(
        num_experts: usize,
        num_ep_ranks: usize,
        hidden_size: usize,
    ) -> ExpertParallelConfig {
        ExpertParallelConfig::new(num_experts, num_ep_ranks, hidden_size)
    }

    // ── AllToAllPlan ─────────────────────────────────────────────────────────

    #[test]
    fn test_plan_all_to_all_uniform_routing() {
        // 8 tokens routed uniformly to 4 experts on 2 ranks (2 experts/rank)
        // experts 0,1 → rank 0; experts 2,3 → rank 1
        let assignments = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let plan = AllToAllCommunication::plan_all_to_all(&assignments, 2, 4);

        // 4 tokens (experts 0,1) go to rank 0 (local), 4 tokens (experts 2,3) to rank 1
        assert_eq!(plan.send_counts[0][0], 4, "4 tokens stay local at rank 0");
        assert_eq!(plan.send_counts[0][1], 4, "4 tokens sent to rank 1");
    }

    #[test]
    fn test_plan_all_to_all_unbalanced_routing() {
        // All 8 tokens assigned to expert 0 (rank 0)
        let assignments = vec![0usize; 8];
        let plan = AllToAllCommunication::plan_all_to_all(&assignments, 2, 4);

        assert_eq!(plan.send_counts[0][0], 8, "all tokens stay at rank 0");
        assert_eq!(plan.send_counts[0][1], 0, "nothing sent to rank 1");
    }

    #[test]
    fn test_total_tokens_moved_uniform() {
        // 8 tokens: 4 stay at rank 0, 4 cross to rank 1
        let assignments = vec![0, 0, 0, 0, 3, 3, 3, 3]; // 4 to rank 0, 4 to rank 1
        let plan = AllToAllCommunication::plan_all_to_all(&assignments, 2, 4);
        let moved = plan.total_tokens_moved();
        // Only cross-rank tokens count (0→1)
        assert_eq!(moved, 4, "4 tokens cross rank boundaries, got {moved}");
    }

    #[test]
    fn test_total_tokens_moved_all_local() {
        // All tokens stay on rank 0
        let assignments = vec![0usize; 16];
        let plan = AllToAllCommunication::plan_all_to_all(&assignments, 4, 8);
        assert_eq!(
            plan.total_tokens_moved(),
            0,
            "no cross-rank movement when all tokens are local"
        );
    }

    // ── CommunicationVolume ──────────────────────────────────────────────────

    #[test]
    fn test_communication_volume_total_bytes() {
        // 4 tokens cross from rank 0 to rank 1, hidden_size = 8 → 4 * 8 * 4 = 128 bytes
        let assignments = vec![2, 2, 2, 2]; // all to expert 2 → rank 1
        let plan = AllToAllCommunication::plan_all_to_all(&assignments, 2, 4);
        let vol = AllToAllCommunication::simulate_communication_volume(&plan, 8);

        assert_eq!(vol.total_bytes, 4 * 8 * 4, "expected 128 bytes, got {}", vol.total_bytes);
    }

    #[test]
    fn test_communication_volume_bandwidth_utilization_range() {
        let assignments = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let plan = AllToAllCommunication::plan_all_to_all(&assignments, 2, 4);
        let vol = AllToAllCommunication::simulate_communication_volume(&plan, 16);

        assert!(
            vol.bandwidth_utilization >= 0.0 && vol.bandwidth_utilization <= 1.0,
            "bandwidth utilization must be in [0, 1], got {}",
            vol.bandwidth_utilization
        );
    }

    // ── ExpertParallelConfig / Validate ─────────────────────────────────────

    #[test]
    fn test_ep_config_valid() {
        let cfg = make_config(8, 4, 512);
        assert!(cfg.validate().is_ok());
        assert_eq!(cfg.experts_per_rank, 2);
    }

    #[test]
    fn test_ep_config_invalid_divisibility() {
        let cfg = make_config(7, 4, 512);
        let err = cfg.validate().unwrap_err();
        matches!(err, EpError::ExpertsDivisibility { .. });
    }

    #[test]
    fn test_ep_config_invalid_zero_ranks() {
        let cfg = make_config(8, 0, 512);
        let err = cfg.validate().unwrap_err();
        matches!(err, EpError::InvalidNumRanks(_));
    }

    // ── ExpertParallelScheduler ──────────────────────────────────────────────

    #[test]
    fn test_local_expert_indices() {
        let cfg = make_config(8, 4, 512);
        let scheduler = ExpertParallelScheduler::new(cfg).expect("config should be valid");

        assert_eq!(scheduler.local_expert_indices(0), vec![0, 1]);
        assert_eq!(scheduler.local_expert_indices(1), vec![2, 3]);
        assert_eq!(scheduler.local_expert_indices(2), vec![4, 5]);
        assert_eq!(scheduler.local_expert_indices(3), vec![6, 7]);
    }

    #[test]
    fn test_schedule_uniform_routing() {
        let cfg = make_config(4, 2, 8);
        let scheduler = ExpertParallelScheduler::new(cfg).expect("valid config");

        let seq_len = 8;
        let assignments = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let hidden = vec![0.0_f32; seq_len * 8];

        let result = scheduler
            .schedule(&assignments, &hidden, seq_len)
            .expect("schedule should succeed");

        assert_eq!(result.local_token_counts.len(), 2);
        // 4 tokens to experts 0,1 (rank 0) and 4 to experts 2,3 (rank 1)
        assert_eq!(result.local_token_counts[0], 4);
        assert_eq!(result.local_token_counts[1], 4);
    }

    // ── AllToAllPlan imbalance ───────────────────────────────────────────────

    #[test]
    fn test_imbalance_ratio_balanced() {
        // Perfectly balanced: rank 0 sends 4 tokens to rank 1, and that's it.
        // But with only 1 sending rank the ratio is trivially 1.0
        let assignments = vec![2, 2, 2, 2];
        let plan = AllToAllCommunication::plan_all_to_all(&assignments, 2, 4);
        let ratio = plan.max_imbalance_ratio();
        // rank 0 sends 4 cross-rank; rank 1 sends 0 → max=4, mean=2 → ratio=2.0
        assert!(ratio >= 1.0, "ratio must be ≥ 1.0, got {ratio}");
    }

    #[test]
    fn test_imbalance_ratio_all_local() {
        let assignments = vec![0usize; 8];
        let plan = AllToAllCommunication::plan_all_to_all(&assignments, 2, 4);
        // No cross-rank traffic → total_moved = 0 → ratio = 1.0 by convention
        let ratio = plan.max_imbalance_ratio();
        assert_eq!(ratio, 1.0, "all-local routing should yield ratio 1.0, got {ratio}");
    }
}
