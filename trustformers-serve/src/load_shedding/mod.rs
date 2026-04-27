//! Load Shedding — proactive request rejection/degradation under high load.
//!
//! When system load exceeds capacity, accepting additional requests worsens the
//! situation for *all* clients (cascading failure).  Load shedding rejects or
//! degrades low-priority requests before the system becomes saturated.

mod load_shedding_extra_tests;

use std::time::Instant;

// ─────────────────────────────────────────────────────────────────────────────
// LoadSheddingPolicy
// ─────────────────────────────────────────────────────────────────────────────

/// Determines the algorithm used to decide whether a request should be shed.
#[derive(Debug, Clone, PartialEq)]
pub enum LoadSheddingPolicy {
    /// Never shed — all requests are accepted.
    None,
    /// Shed requests randomly with the given probability.
    ProbabilisticDrop { drop_probability: f32 },
    /// Shed requests when the estimated queue wait exceeds a threshold.
    LatencyBased { max_queue_latency_ms: f64 },
    /// Shed requests when CPU or memory utilisation exceeds a threshold.
    ResourceBased {
        cpu_threshold: f32,
        memory_threshold: f32,
    },
    /// Degrade service quality instead of dropping outright.
    GradualDegradation { min_quality: f32 },
}

// ─────────────────────────────────────────────────────────────────────────────
// LoadSheddingConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a [`LoadShedder`].
#[derive(Debug, Clone)]
pub struct LoadSheddingConfig {
    /// Which algorithm to use for shedding decisions.
    pub policy: LoadSheddingPolicy,
    /// Number of priority levels (default 3, numbered 0 = lowest … n-1 = highest).
    pub priority_levels: usize,
    /// Load fraction at which each priority level starts being shed.
    /// Must have exactly `priority_levels` entries.
    /// E.g. [0.3, 0.7, 1.0] — priority 0 is shed at 30 % load, priority 1 at
    /// 70 %, priority 2 only when fully saturated.
    pub priority_thresholds: Vec<f32>,
    /// Minimum time between two consecutive shedding decisions (milliseconds).
    pub cooldown_ms: u64,
    /// Exponential back-off multiplier applied to the adaptive drop probability.
    pub backoff_factor: f32,
}

impl Default for LoadSheddingConfig {
    fn default() -> Self {
        LoadSheddingConfig {
            policy: LoadSheddingPolicy::None,
            priority_levels: 3,
            priority_thresholds: vec![0.3, 0.7, 1.0],
            cooldown_ms: 100,
            backoff_factor: 1.5,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SystemLoad
// ─────────────────────────────────────────────────────────────────────────────

/// A snapshot of the current system load, fed into shedding decisions.
#[derive(Debug, Clone)]
pub struct SystemLoad {
    /// CPU utilisation fraction [0.0, 1.0].
    pub cpu_utilization: f32,
    /// Memory utilisation fraction [0.0, 1.0].
    pub memory_utilization: f32,
    /// Number of requests currently being processed.
    pub active_requests: usize,
    /// Number of requests waiting in the queue.
    pub queue_depth: usize,
    /// Estimated time a newly-arriving request would wait in the queue (ms).
    pub estimated_queue_latency_ms: f64,
}

impl SystemLoad {
    /// Returns `true` when either CPU or memory exceeds the given thresholds.
    pub fn is_overloaded(&self, cpu_threshold: f32, mem_threshold: f32) -> bool {
        self.cpu_utilization > cpu_threshold || self.memory_utilization > mem_threshold
    }

    /// A single [0.0, 1.0] score representing overall load pressure.
    ///
    /// Computed as `max(cpu, memory) * (1 + log2(1 + queue_depth) / 10)`.
    pub fn overall_load_score(&self) -> f32 {
        let resource_load = self.cpu_utilization.max(self.memory_utilization);
        let queue_pressure = (1.0 + self.queue_depth as f32).log2() / 10.0;
        (resource_load * (1.0 + queue_pressure)).min(1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SheddingReason / SheddingDecision
// ─────────────────────────────────────────────────────────────────────────────

/// Describes why a request was shed.
#[derive(Debug, Clone, PartialEq)]
pub enum SheddingReason {
    HighCpuLoad { cpu: f32 },
    HighMemoryLoad { mem: f32 },
    QueueTooLong { estimated_ms: f64 },
    LowPriority { priority: u8 },
    ProbabilisticDrop,
}

/// The outcome of a shedding decision for a single request.
#[derive(Debug, Clone, PartialEq)]
pub enum SheddingDecision {
    /// Process the request normally.
    Accept,
    /// Reject the request entirely.
    Drop { reason: SheddingReason },
    /// Process at a reduced quality level (value in [0.0, 1.0]).
    Degrade { quality_level: f32 },
    /// Ask the client to retry after the suggested delay.
    Defer { suggested_retry_ms: u64 },
}

// ─────────────────────────────────────────────────────────────────────────────
// SheddingStats
// ─────────────────────────────────────────────────────────────────────────────

/// Counters tracking the overall behaviour of a [`LoadShedder`].
#[derive(Debug, Clone, Default)]
pub struct SheddingStats {
    pub total_requests: u64,
    pub total_shed: u64,
    pub total_degraded: u64,
    pub total_deferred: u64,
}

impl SheddingStats {
    /// Fraction of requests that were shed (dropped), in [0.0, 1.0].
    pub fn shed_rate(&self) -> f32 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.total_shed as f32 / self.total_requests as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LoadShedder
// ─────────────────────────────────────────────────────────────────────────────

/// Applies a configured shedding policy to incoming requests.
pub struct LoadShedder {
    pub config: LoadSheddingConfig,
    pub shedding_stats: SheddingStats,
    /// Adaptive drop probability updated by [`LoadShedder::feedback`].
    pub current_drop_probability: f32,
    /// When the most recent shed decision was made.
    pub last_shed_time: Option<Instant>,
    /// Internal monotonic counter for deterministic drop seeds (no rand).
    request_counter: u64,
}

impl LoadShedder {
    /// Create a new `LoadShedder` with the given configuration.
    pub fn new(config: LoadSheddingConfig) -> Self {
        let initial_prob = match &config.policy {
            LoadSheddingPolicy::ProbabilisticDrop { drop_probability } => *drop_probability,
            _ => 0.0,
        };
        LoadShedder {
            config,
            shedding_stats: SheddingStats::default(),
            current_drop_probability: initial_prob,
            last_shed_time: None,
            request_counter: 0,
        }
    }

    /// Decide whether to accept, drop, degrade, or defer a request.
    ///
    /// `request_priority` is in `0..priority_levels`, where 0 is the *lowest*
    /// priority.
    pub fn should_shed(&mut self, request_priority: u8, load: &SystemLoad) -> SheddingDecision {
        self.shedding_stats.total_requests += 1;
        self.request_counter = self.request_counter.wrapping_add(1);

        // ── 1. Check cooldown ─────────────────────────────────────────────
        if let Some(last) = self.last_shed_time {
            let elapsed_ms = last.elapsed().as_millis() as u64;
            if elapsed_ms < self.config.cooldown_ms {
                // Within cooldown window — accept unconditionally.
                return SheddingDecision::Accept;
            }
        }

        // ── 2. Apply policy ──────────────────────────────────────────────
        let decision = match &self.config.policy.clone() {
            LoadSheddingPolicy::None => SheddingDecision::Accept,

            LoadSheddingPolicy::ProbabilisticDrop { drop_probability } => {
                let effective_prob = self.current_drop_probability.max(*drop_probability);
                if Self::deterministic_should_drop(self.request_counter, effective_prob) {
                    SheddingDecision::Drop {
                        reason: SheddingReason::ProbabilisticDrop,
                    }
                } else {
                    SheddingDecision::Accept
                }
            },

            LoadSheddingPolicy::LatencyBased {
                max_queue_latency_ms,
            } => {
                if load.estimated_queue_latency_ms > *max_queue_latency_ms {
                    SheddingDecision::Drop {
                        reason: SheddingReason::QueueTooLong {
                            estimated_ms: load.estimated_queue_latency_ms,
                        },
                    }
                } else {
                    SheddingDecision::Accept
                }
            },

            LoadSheddingPolicy::ResourceBased {
                cpu_threshold,
                memory_threshold,
            } => {
                if load.cpu_utilization > *cpu_threshold {
                    SheddingDecision::Drop {
                        reason: SheddingReason::HighCpuLoad {
                            cpu: load.cpu_utilization,
                        },
                    }
                } else if load.memory_utilization > *memory_threshold {
                    SheddingDecision::Drop {
                        reason: SheddingReason::HighMemoryLoad {
                            mem: load.memory_utilization,
                        },
                    }
                } else {
                    SheddingDecision::Accept
                }
            },

            LoadSheddingPolicy::GradualDegradation { min_quality } => {
                let load_score = load.overall_load_score();
                if load_score > 0.8 {
                    let quality = (1.0 - (load_score - 0.8) / 0.2).max(*min_quality).min(1.0);
                    SheddingDecision::Degrade {
                        quality_level: quality,
                    }
                } else {
                    SheddingDecision::Accept
                }
            },
        };

        // ── 3. Priority-based override ───────────────────────────────────
        // Lower-priority requests are shed at lower load thresholds.
        // Priority overrides are not applied when the policy is None.
        let decision = if matches!(self.config.policy, LoadSheddingPolicy::None) {
            decision
        } else {
            self.apply_priority_override(request_priority, load, decision)
        };

        // ── 4. Update shed time & stats ──────────────────────────────────
        match &decision {
            SheddingDecision::Drop { .. } => {
                self.shedding_stats.total_shed += 1;
                self.last_shed_time = Some(Instant::now());
            },
            SheddingDecision::Degrade { .. } => {
                self.shedding_stats.total_degraded += 1;
            },
            SheddingDecision::Defer { .. } => {
                self.shedding_stats.total_deferred += 1;
                self.last_shed_time = Some(Instant::now());
            },
            SheddingDecision::Accept => {},
        }

        decision
    }

    /// Apply priority thresholds as an override on top of the base policy
    /// decision.  If the overall load score exceeds the threshold for this
    /// priority level and the base policy already accepted the request, we
    /// still drop it.
    fn apply_priority_override(
        &self,
        request_priority: u8,
        load: &SystemLoad,
        base: SheddingDecision,
    ) -> SheddingDecision {
        // Only apply priority shedding when the base already shed OR when the
        // priority threshold for this level is exceeded.
        let priority_index = (request_priority as usize)
            .min(self.config.priority_thresholds.len().saturating_sub(1));
        let threshold = self.config.priority_thresholds.get(priority_index).copied().unwrap_or(1.0);

        let load_score = load.overall_load_score();

        if matches!(base, SheddingDecision::Accept) && load_score >= threshold {
            SheddingDecision::Drop {
                reason: SheddingReason::LowPriority {
                    priority: request_priority,
                },
            }
        } else {
            base
        }
    }

    /// Update the adaptive drop probability based on feedback.
    ///
    /// * `shed` — whether a request was shed at the last decision point.
    /// * `actual_latency_ms` — the observed latency (if the request was processed).
    ///   Pass `None` when the request was shed.
    ///
    /// When shedding was correct (latency was high), increase the probability.
    /// When shedding was wrong (latency was fine), decrease it.
    pub fn feedback(&mut self, shed: bool, actual_latency_ms: Option<f64>) {
        const HIGH_LATENCY_THRESHOLD_MS: f64 = 1_000.0;
        const DECAY: f32 = 0.9; // factor to reduce probability when unnecessary

        if shed {
            // We shed a request — was that correct?
            // Without actual latency we assume it was correct (conservative).
            let latency_high =
                actual_latency_ms.map(|ms| ms >= HIGH_LATENCY_THRESHOLD_MS).unwrap_or(true);

            if latency_high {
                // Correct shed — increase probability.
                self.current_drop_probability =
                    (self.current_drop_probability * self.config.backoff_factor).min(1.0);
            } else {
                // Wrong shed — system was fine, back off.
                self.current_drop_probability *= DECAY;
            }
        } else {
            // We accepted the request.
            match actual_latency_ms {
                Some(ms) if ms >= HIGH_LATENCY_THRESHOLD_MS => {
                    // We should have shed it — increase probability.
                    self.current_drop_probability =
                        (self.current_drop_probability * self.config.backoff_factor).min(1.0);
                },
                Some(_) => {
                    // Latency fine — slowly decay the drop probability.
                    self.current_drop_probability *= DECAY;
                },
                None => {
                    // No latency data — leave probability unchanged.
                },
            }
        }
    }

    /// Deterministically decide whether to drop a request without using the
    /// `rand` crate.
    ///
    /// Uses FNV-1a of `seed` mapped to [0.0, 1.0) and compared against
    /// `probability`.
    pub fn deterministic_should_drop(seed: u64, probability: f32) -> bool {
        // FNV-1a 64-bit hash of the seed bytes.
        const FNV_OFFSET: u64 = 14695981039346656037;
        const FNV_PRIME: u64 = 1099511628211;
        let mut hash = FNV_OFFSET;
        for &b in &seed.to_le_bytes() {
            hash ^= b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        // Map to [0.0, 1.0).
        let normalised = (hash as f64 / u64::MAX as f64) as f32;
        normalised < probability
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ShedDecider trait
// ─────────────────────────────────────────────────────────────────────────────

/// Common interface for any shedding decision-maker.
pub trait ShedDecider {
    /// Returns `true` if the request with this hash should be dropped.
    fn should_shed_request(&self, request_hash: u64) -> bool;
}

// ─────────────────────────────────────────────────────────────────────────────
// AdaptiveLoadShedder
// ─────────────────────────────────────────────────────────────────────────────

/// Adjusts the drop probability proportionally based on queue depth relative to
/// a target depth.
///
/// Control law: `dp ← clamp(dp + k * (queue_depth − target_depth) / target_depth, 0, 1)`
pub struct AdaptiveLoadShedder {
    /// Current drop probability in [0, 1].
    pub drop_probability: f32,
    /// Proportional control gain (default 0.1).
    pub k: f32,
    /// Target queue depth at which no correction is applied.
    pub target_depth: usize,
}

impl AdaptiveLoadShedder {
    /// Create a new adaptive shedder with the given target depth and control gain.
    pub fn new(target_depth: usize, k: f32) -> Self {
        Self {
            drop_probability: 0.0,
            k,
            target_depth,
        }
    }

    /// Update the drop probability based on the current observed queue depth.
    ///
    /// Proportional control: error = (queue_depth − target_depth) / target_depth.
    /// When target_depth is 0 the probability is left unchanged.
    pub fn update_drop_probability(&mut self, queue_depth: usize, target_depth: usize) {
        if target_depth == 0 {
            return;
        }
        let error = (queue_depth as f32 - target_depth as f32) / target_depth as f32;
        self.drop_probability = (self.drop_probability + self.k * error).clamp(0.0, 1.0);
        self.target_depth = target_depth;
    }
}

impl ShedDecider for AdaptiveLoadShedder {
    fn should_shed_request(&self, request_hash: u64) -> bool {
        LoadShedder::deterministic_should_drop(request_hash, self.drop_probability)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PriorityAwareShedder
// ─────────────────────────────────────────────────────────────────────────────

/// Applies different drop rates per priority level.
///
/// Higher priority (higher `priority` value) → lower drop rate.
/// `drop_rates[0]` applies to priority 0 (lowest), and so on.
pub struct PriorityAwareShedder {
    /// Per-priority drop rates; `drop_rates[i]` is the drop probability for priority `i`.
    pub drop_rates: Vec<f32>,
}

impl PriorityAwareShedder {
    /// Create a new shedder with the given per-priority drop rates.
    pub fn new(drop_rates: Vec<f32>) -> Self {
        Self { drop_rates }
    }

    /// Decide whether to shed a request of the given `priority`.
    ///
    /// `seed` provides deterministic pseudo-randomness (no external RNG).
    pub fn shed_by_priority(&self, priority: u8, seed: u64) -> bool {
        let rate = self.drop_rates.get(priority as usize).copied().unwrap_or(0.0);
        LoadShedder::deterministic_should_drop(seed, rate)
    }
}

impl ShedDecider for PriorityAwareShedder {
    fn should_shed_request(&self, request_hash: u64) -> bool {
        // Default: use priority 0 (lowest) as the baseline.
        self.shed_by_priority(0, request_hash)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ShedderComposition
// ─────────────────────────────────────────────────────────────────────────────

/// Combines multiple `ShedDecider` implementations: shed if ANY shedder says drop.
pub struct ShedderComposition;

impl ShedderComposition {
    /// Return `true` if any shedder in the slice decides to shed the request.
    pub fn any_shed(shedders: &[&dyn ShedDecider], request_hash: u64) -> bool {
        shedders.iter().any(|s| s.should_shed_request(request_hash))
    }

    /// Return `true` only if ALL shedders in the slice decide to shed the request.
    pub fn all_shed(shedders: &[&dyn ShedDecider], request_hash: u64) -> bool {
        if shedders.is_empty() {
            return false;
        }
        shedders.iter().all(|s| s.should_shed_request(request_hash))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_load() -> SystemLoad {
        SystemLoad {
            cpu_utilization: 0.0,
            memory_utilization: 0.0,
            active_requests: 0,
            queue_depth: 0,
            estimated_queue_latency_ms: 0.0,
        }
    }

    fn high_load(cpu: f32, mem: f32) -> SystemLoad {
        SystemLoad {
            cpu_utilization: cpu,
            memory_utilization: mem,
            active_requests: 100,
            queue_depth: 50,
            estimated_queue_latency_ms: 2_000.0,
        }
    }

    // 1. None policy always accepts
    #[test]
    fn test_none_policy_always_accepts() {
        let mut shedder = LoadShedder::new(LoadSheddingConfig::default());
        for _ in 0..10 {
            let decision = shedder.should_shed(0, &high_load(0.99, 0.99));
            assert_eq!(decision, SheddingDecision::Accept);
        }
    }

    // 2. ProbabilisticDrop with p=1.0 always drops (after cooldown resets)
    #[test]
    fn test_probabilistic_drop_p1_always_drops() {
        let config = LoadSheddingConfig {
            policy: LoadSheddingPolicy::ProbabilisticDrop {
                drop_probability: 1.0,
            },
            priority_thresholds: vec![1.0, 1.0, 1.0], // disable priority override
            cooldown_ms: 0,
            ..LoadSheddingConfig::default()
        };
        let mut shedder = LoadShedder::new(config);
        // With p=1.0 every request must be dropped (except those blocked by cooldown,
        // but cooldown_ms=0 means no cooldown).
        let decision = shedder.should_shed(2, &zero_load());
        assert!(
            matches!(decision, SheddingDecision::Drop { .. }),
            "p=1.0 should always drop"
        );
    }

    // 3. LatencyBased drops when over threshold
    #[test]
    fn test_latency_based_drops_when_over_threshold() {
        let config = LoadSheddingConfig {
            policy: LoadSheddingPolicy::LatencyBased {
                max_queue_latency_ms: 500.0,
            },
            priority_thresholds: vec![1.0, 1.0, 1.0],
            cooldown_ms: 0,
            ..LoadSheddingConfig::default()
        };
        let mut shedder = LoadShedder::new(config);

        let over_threshold = SystemLoad {
            estimated_queue_latency_ms: 1_000.0,
            ..zero_load()
        };
        let under_threshold = SystemLoad {
            estimated_queue_latency_ms: 200.0,
            ..zero_load()
        };

        assert!(matches!(
            shedder.should_shed(2, &over_threshold),
            SheddingDecision::Drop { .. }
        ));
        // Reset shed time so cooldown doesn't interfere.
        shedder.last_shed_time = None;
        assert_eq!(
            shedder.should_shed(2, &under_threshold),
            SheddingDecision::Accept
        );
    }

    // 4. ResourceBased triggers on CPU threshold
    #[test]
    fn test_resource_based_cpu_trigger() {
        let config = LoadSheddingConfig {
            policy: LoadSheddingPolicy::ResourceBased {
                cpu_threshold: 0.7,
                memory_threshold: 0.9,
            },
            priority_thresholds: vec![1.0, 1.0, 1.0],
            cooldown_ms: 0,
            ..LoadSheddingConfig::default()
        };
        let mut shedder = LoadShedder::new(config);

        let high_cpu = SystemLoad {
            cpu_utilization: 0.85,
            ..zero_load()
        };
        assert!(matches!(
            shedder.should_shed(2, &high_cpu),
            SheddingDecision::Drop {
                reason: SheddingReason::HighCpuLoad { .. }
            }
        ));
    }

    // 5. ResourceBased triggers on memory threshold
    #[test]
    fn test_resource_based_memory_trigger() {
        let config = LoadSheddingConfig {
            policy: LoadSheddingPolicy::ResourceBased {
                cpu_threshold: 0.9,
                memory_threshold: 0.7,
            },
            priority_thresholds: vec![1.0, 1.0, 1.0],
            cooldown_ms: 0,
            ..LoadSheddingConfig::default()
        };
        let mut shedder = LoadShedder::new(config);

        let high_mem = SystemLoad {
            memory_utilization: 0.85,
            ..zero_load()
        };
        assert!(matches!(
            shedder.should_shed(2, &high_mem),
            SheddingDecision::Drop {
                reason: SheddingReason::HighMemoryLoad { .. }
            }
        ));
    }

    // 6. Low-priority requests shed at lower load thresholds
    #[test]
    fn test_low_priority_shed_at_lower_load() {
        let config = LoadSheddingConfig {
            // Use LatencyBased so the priority override mechanism is active.
            policy: LoadSheddingPolicy::LatencyBased {
                max_queue_latency_ms: 9_999_999.0, // won't trigger from latency alone
            },
            priority_levels: 3,
            priority_thresholds: vec![0.3, 0.7, 1.0],
            cooldown_ms: 0,
            ..LoadSheddingConfig::default()
        };
        let mut shedder = LoadShedder::new(config);

        // Load score ~0.5 — priority 0 (threshold 0.3) should be shed,
        // priority 2 (threshold 1.0) should not.
        let moderate_load = SystemLoad {
            cpu_utilization: 0.5,
            memory_utilization: 0.0,
            active_requests: 10,
            queue_depth: 0,
            estimated_queue_latency_ms: 0.0,
        };

        // priority=0 threshold=0.3: 0.5 >= 0.3 → should drop
        let low_prio_decision = shedder.should_shed(0, &moderate_load);
        assert!(
            matches!(low_prio_decision, SheddingDecision::Drop { .. }),
            "low priority should be shed at moderate load"
        );

        // Reset state between calls
        shedder.last_shed_time = None;

        // priority=2 threshold=1.0: 0.5 < 1.0 → should accept
        let high_prio_decision = shedder.should_shed(2, &moderate_load);
        assert_eq!(
            high_prio_decision,
            SheddingDecision::Accept,
            "high priority should not be shed at moderate load"
        );
    }

    // 7. Priority thresholds are respected
    #[test]
    fn test_priority_thresholds() {
        let config = LoadSheddingConfig {
            // Use LatencyBased so the priority override mechanism is active.
            policy: LoadSheddingPolicy::LatencyBased {
                max_queue_latency_ms: 9_999_999.0, // won't trigger from latency alone
            },
            priority_levels: 3,
            priority_thresholds: vec![0.1, 0.5, 1.0],
            cooldown_ms: 0,
            ..LoadSheddingConfig::default()
        };
        let mut shedder = LoadShedder::new(config);

        let load = SystemLoad {
            cpu_utilization: 0.6,
            ..zero_load()
        };

        // Priority 0: threshold 0.1, load 0.6 → drop
        assert!(matches!(
            shedder.should_shed(0, &load),
            SheddingDecision::Drop { .. }
        ));
        shedder.last_shed_time = None;

        // Priority 1: threshold 0.5, load 0.6 → drop (0.6 >= 0.5)
        assert!(matches!(
            shedder.should_shed(1, &load),
            SheddingDecision::Drop { .. }
        ));
        shedder.last_shed_time = None;

        // Priority 2: threshold 1.0, load 0.6 → accept (0.6 < 1.0)
        assert_eq!(shedder.should_shed(2, &load), SheddingDecision::Accept);
    }

    // 8. Cooldown: rapid calls don't shed due to cooldown
    #[test]
    fn test_cooldown_prevents_consecutive_shedding() {
        let config = LoadSheddingConfig {
            policy: LoadSheddingPolicy::LatencyBased {
                max_queue_latency_ms: 10.0,
            },
            priority_thresholds: vec![1.0, 1.0, 1.0],
            cooldown_ms: 100_000, // 100 seconds — effectively infinite for test
            ..LoadSheddingConfig::default()
        };
        let mut shedder = LoadShedder::new(config);

        let high_lat = SystemLoad {
            estimated_queue_latency_ms: 9999.0,
            ..zero_load()
        };

        // First call: no cooldown active, should shed.
        let first = shedder.should_shed(2, &high_lat);
        assert!(
            matches!(first, SheddingDecision::Drop { .. }),
            "first call should shed"
        );

        // Second call immediately after: still within cooldown, must accept.
        let second = shedder.should_shed(2, &high_lat);
        assert_eq!(
            second,
            SheddingDecision::Accept,
            "within cooldown should accept"
        );
    }

    // 9. deterministic_should_drop is consistent
    #[test]
    fn test_deterministic_should_drop_consistency() {
        // Same seed + probability must always return the same result.
        let a = LoadShedder::deterministic_should_drop(42, 0.5);
        let b = LoadShedder::deterministic_should_drop(42, 0.5);
        assert_eq!(a, b);

        // p=0.0 must never drop, p=1.0 must always drop.
        assert!(!LoadShedder::deterministic_should_drop(12345, 0.0));
        assert!(LoadShedder::deterministic_should_drop(12345, 1.0));
    }

    // 10. SheddingStats shed_rate
    #[test]
    fn test_shedding_stats_shed_rate() {
        let mut stats = SheddingStats::default();
        assert_eq!(stats.shed_rate(), 0.0);

        stats.total_requests = 10;
        stats.total_shed = 3;
        let rate = stats.shed_rate();
        assert!((rate - 0.3).abs() < 1e-5, "shed rate should be 0.3");
    }

    // 11. feedback adjusts drop probability upward on high-latency correct shed
    #[test]
    fn test_feedback_increases_probability_on_correct_shed() {
        let config = LoadSheddingConfig {
            policy: LoadSheddingPolicy::ProbabilisticDrop {
                drop_probability: 0.1,
            },
            cooldown_ms: 0,
            backoff_factor: 2.0,
            ..LoadSheddingConfig::default()
        };
        let mut shedder = LoadShedder::new(config);
        shedder.current_drop_probability = 0.2;

        shedder.feedback(true, Some(2_000.0)); // shed was correct
        assert!(
            shedder.current_drop_probability > 0.2,
            "probability should increase after correct shed"
        );
    }

    // 12. system load overloaded check
    #[test]
    fn test_system_load_is_overloaded() {
        let load = SystemLoad {
            cpu_utilization: 0.9,
            memory_utilization: 0.4,
            active_requests: 5,
            queue_depth: 0,
            estimated_queue_latency_ms: 0.0,
        };

        assert!(
            load.is_overloaded(0.8, 0.9),
            "high CPU should trigger overload"
        );
        assert!(!load.is_overloaded(0.95, 0.95), "both below threshold");
    }

    // 13. Degrade variant is returned by GradualDegradation at very high load
    #[test]
    fn test_degrade_variant_at_high_load() {
        let config = LoadSheddingConfig {
            policy: LoadSheddingPolicy::GradualDegradation { min_quality: 0.1 },
            priority_thresholds: vec![1.0, 1.0, 1.0],
            cooldown_ms: 0,
            ..LoadSheddingConfig::default()
        };
        let mut shedder = LoadShedder::new(config);

        // overall_load_score > 0.8 requires cpu or memory > ~0.8.
        let very_high_load = SystemLoad {
            cpu_utilization: 0.95,
            memory_utilization: 0.0,
            active_requests: 0,
            queue_depth: 0,
            estimated_queue_latency_ms: 0.0,
        };
        let decision = shedder.should_shed(2, &very_high_load);
        assert!(
            matches!(decision, SheddingDecision::Degrade { .. }),
            "GradualDegradation at very high load must return Degrade variant, got {:?}",
            decision
        );
    }

    // ── AdaptiveLoadShedder tests ─────────────────────────────────────────────

    // 14. Adaptive shedder: drop_probability starts at 0
    #[test]
    fn test_adaptive_shedder_initial_probability_zero() {
        let shedder = AdaptiveLoadShedder::new(10, 0.1);
        assert!((shedder.drop_probability - 0.0).abs() < f32::EPSILON);
    }

    // 15. Adaptive shedder: queue over target increases probability
    #[test]
    fn test_adaptive_shedder_queue_over_target_increases_prob() {
        let mut shedder = AdaptiveLoadShedder::new(10, 0.1);
        shedder.update_drop_probability(20, 10); // 100% over target
        assert!(
            shedder.drop_probability > 0.0,
            "over-target queue should increase drop probability: {}",
            shedder.drop_probability
        );
    }

    // 16. Adaptive shedder: queue under target decreases probability
    #[test]
    fn test_adaptive_shedder_queue_under_target_decreases_prob() {
        let mut shedder = AdaptiveLoadShedder::new(10, 0.1);
        shedder.drop_probability = 0.5; // artificially high
        shedder.update_drop_probability(5, 10); // 50% under target → error < 0
        assert!(
            shedder.drop_probability < 0.5,
            "under-target queue should decrease drop probability: {}",
            shedder.drop_probability
        );
    }

    // 17. Adaptive shedder: probability is clamped to [0, 1]
    #[test]
    fn test_adaptive_shedder_probability_clamped() {
        let mut shedder = AdaptiveLoadShedder::new(1, 10.0); // very high gain
        shedder.update_drop_probability(1000, 1); // massively over target
        assert!(
            shedder.drop_probability <= 1.0,
            "probability must not exceed 1.0: {}",
            shedder.drop_probability
        );
        shedder.drop_probability = 0.0;
        shedder.update_drop_probability(0, 1000); // massively under target
        assert!(
            shedder.drop_probability >= 0.0,
            "probability must not go below 0.0: {}",
            shedder.drop_probability
        );
    }

    // 18. Adaptive shedder: zero target_depth leaves probability unchanged
    #[test]
    fn test_adaptive_shedder_zero_target_depth_no_change() {
        let mut shedder = AdaptiveLoadShedder::new(10, 0.1);
        shedder.drop_probability = 0.3;
        shedder.update_drop_probability(100, 0); // target=0 → no update
        assert!(
            (shedder.drop_probability - 0.3).abs() < f32::EPSILON,
            "zero target_depth should not change probability"
        );
    }

    // 19. Adaptive shedder: at-target queue leaves probability unchanged
    #[test]
    fn test_adaptive_shedder_at_target_no_change() {
        let mut shedder = AdaptiveLoadShedder::new(10, 0.1);
        shedder.drop_probability = 0.3;
        shedder.update_drop_probability(10, 10); // error = 0
        assert!(
            (shedder.drop_probability - 0.3).abs() < f32::EPSILON,
            "queue at target should not change probability"
        );
    }

    // ── PriorityAwareShedder tests ────────────────────────────────────────────

    // 20. Priority-aware shedder: high-priority never shed with 0.0 rate
    #[test]
    fn test_priority_aware_high_priority_never_shed() {
        let shedder = PriorityAwareShedder::new(vec![1.0, 0.5, 0.0]); // priority 2 = 0% drop
        for seed in 0..100u64 {
            assert!(
                !shedder.shed_by_priority(2, seed),
                "priority 2 with 0% rate should never be shed (seed {seed})"
            );
        }
    }

    // 21. Priority-aware shedder: lowest priority always shed with 1.0 rate
    #[test]
    fn test_priority_aware_low_priority_always_shed() {
        let shedder = PriorityAwareShedder::new(vec![1.0, 0.5, 0.0]);
        let decision = shedder.shed_by_priority(0, 42);
        assert!(decision, "priority 0 with 100% rate should be shed");
    }

    // 22. Priority-aware shedder: different priorities get different rates
    #[test]
    fn test_priority_aware_different_rates_per_level() {
        let shedder = PriorityAwareShedder::new(vec![1.0, 0.0]); // 0=always drop, 1=never drop
        assert!(shedder.shed_by_priority(0, 99), "priority 0 should drop");
        assert!(
            !shedder.shed_by_priority(1, 99),
            "priority 1 should not drop"
        );
    }

    // 23. Priority-aware shedder: out-of-bounds priority uses 0.0 rate (no shed)
    #[test]
    fn test_priority_aware_out_of_bounds_never_sheds() {
        let shedder = PriorityAwareShedder::new(vec![1.0, 1.0]);
        // Priority 99 is out of bounds → default rate 0.0 → never shed
        for seed in 0..20u64 {
            assert!(
                !shedder.shed_by_priority(99, seed),
                "out-of-bounds priority should not be shed"
            );
        }
    }

    // ── ShedderComposition tests ──────────────────────────────────────────────

    // 24. any_shed — returns false when no shedders
    #[test]
    fn test_any_shed_no_shedders_returns_false() {
        let empty: Vec<&dyn ShedDecider> = vec![];
        assert!(!ShedderComposition::any_shed(&empty, 42));
    }

    // 25. any_shed — returns true if one of two shedders sheds
    #[test]
    fn test_any_shed_one_sheds() {
        // Shedder 1: always drop (rate 1.0), Shedder 2: never drop (rate 0.0)
        let s1 = PriorityAwareShedder::new(vec![1.0]);
        let s2 = PriorityAwareShedder::new(vec![0.0]);
        let shedders: Vec<&dyn ShedDecider> = vec![&s1, &s2];
        assert!(
            ShedderComposition::any_shed(&shedders, 42),
            "any_shed should return true when at least one shedder drops"
        );
    }

    // 26. any_shed — returns false if no shedder sheds
    #[test]
    fn test_any_shed_none_sheds() {
        let s1 = PriorityAwareShedder::new(vec![0.0]);
        let s2 = PriorityAwareShedder::new(vec![0.0]);
        let shedders: Vec<&dyn ShedDecider> = vec![&s1, &s2];
        assert!(
            !ShedderComposition::any_shed(&shedders, 42),
            "any_shed should return false when no shedder drops"
        );
    }

    // 27. all_shed — returns false when no shedders
    #[test]
    fn test_all_shed_no_shedders_returns_false() {
        let empty: Vec<&dyn ShedDecider> = vec![];
        assert!(!ShedderComposition::all_shed(&empty, 42));
    }

    // 28. all_shed — returns true only when all shedders agree
    #[test]
    fn test_all_shed_all_agree() {
        let s1 = PriorityAwareShedder::new(vec![1.0]);
        let s2 = PriorityAwareShedder::new(vec![1.0]);
        let shedders: Vec<&dyn ShedDecider> = vec![&s1, &s2];
        assert!(
            ShedderComposition::all_shed(&shedders, 42),
            "all_shed should return true when all shedders drop"
        );
    }

    // ── Extended tests ─────────────────────────────────────────────────────

    // Test 29: LoadSheddingPolicy::None != ProbabilisticDrop
    #[test]
    fn test_load_shedding_policy_none_ne_probabilistic() {
        let a = LoadSheddingPolicy::None;
        let b = LoadSheddingPolicy::ProbabilisticDrop {
            drop_probability: 0.5,
        };
        assert_ne!(a, b);
    }

    // Test 30: LoadSheddingConfig::default — backoff_factor is 1.5
    #[test]
    fn test_load_shedding_config_default_backoff_factor() {
        let cfg = LoadSheddingConfig::default();
        assert!((cfg.backoff_factor - 1.5).abs() < 1e-6);
    }

    // Test 31: LoadSheddingConfig::default — cooldown_ms is 100
    #[test]
    fn test_load_shedding_config_default_cooldown_ms() {
        let cfg = LoadSheddingConfig::default();
        assert_eq!(cfg.cooldown_ms, 100);
    }

    // Test 32: SystemLoad::is_overloaded — false when both below thresholds
    #[test]
    fn test_system_load_not_overloaded() {
        let load = zero_load();
        assert!(!load.is_overloaded(0.5, 0.5));
    }

    // Test 33: SystemLoad::is_overloaded — true when CPU exceeds threshold
    #[test]
    fn test_system_load_cpu_overloaded() {
        let load = SystemLoad {
            cpu_utilization: 0.9,
            ..zero_load()
        };
        assert!(load.is_overloaded(0.8, 0.9));
    }

    // Test 34: SystemLoad::is_overloaded — true when memory exceeds threshold
    #[test]
    fn test_system_load_mem_overloaded() {
        let load = SystemLoad {
            memory_utilization: 0.95,
            ..zero_load()
        };
        assert!(load.is_overloaded(0.99, 0.9));
    }

    // Test 35: SystemLoad::overall_load_score — deep queue raises score
    #[test]
    fn test_overall_load_score_deep_queue_raises_score() {
        let baseline = SystemLoad {
            cpu_utilization: 0.5,
            ..zero_load()
        };
        let deep_queue = SystemLoad {
            cpu_utilization: 0.5,
            queue_depth: 1000,
            ..zero_load()
        };
        assert!(
            deep_queue.overall_load_score() > baseline.overall_load_score(),
            "deep queue must raise overall_load_score"
        );
    }

    // Test 36: SheddingDecision::Accept == Accept (PartialEq)
    #[test]
    fn test_shedding_decision_accept_eq() {
        assert_eq!(SheddingDecision::Accept, SheddingDecision::Accept);
    }

    // Test 37: ResourceBased accepts below thresholds
    #[test]
    fn test_resource_based_accepts_below_threshold() {
        let config = LoadSheddingConfig {
            policy: LoadSheddingPolicy::ResourceBased {
                cpu_threshold: 0.8,
                memory_threshold: 0.8,
            },
            cooldown_ms: 0,
            ..LoadSheddingConfig::default()
        };
        let mut shedder = LoadShedder::new(config);
        let low_load = SystemLoad {
            cpu_utilization: 0.3,
            memory_utilization: 0.3,
            ..zero_load()
        };
        let d = shedder.should_shed(2, &low_load);
        assert_eq!(d, SheddingDecision::Accept, "below threshold must Accept");
    }

    // Test 38: LoadShedder::shedding_stats starts at zero
    #[test]
    fn test_load_shedder_stats_start_zero() {
        let shedder = LoadShedder::new(LoadSheddingConfig::default());
        assert_eq!(shedder.shedding_stats.total_shed, 0);
        assert_eq!(shedder.shedding_stats.total_requests, 0);
    }

    // Test 39: LoadShedder::shedding_stats total_requests increments on each call
    #[test]
    fn test_load_shedder_stats_requests_increments() {
        let mut shedder = LoadShedder::new(LoadSheddingConfig::default());
        shedder.should_shed(2, &zero_load()); // None policy → Accept
        assert!(
            shedder.shedding_stats.total_requests >= 1,
            "total_requests must be >= 1 after a check"
        );
    }

    // Test 40: PriorityAwareShedder::new with empty thresholds
    #[test]
    fn test_priority_aware_shedder_empty_thresholds() {
        let shedder = PriorityAwareShedder::new(vec![]);
        // Should not panic; any priority should be shed (no thresholds = no protection)
        let _ = shedder.should_shed_request(0);
    }

    // Test 41: LoadSheddingPolicy::LatencyBased ne ResourceBased
    #[test]
    fn test_load_shedding_policy_latency_ne_resource() {
        let a = LoadSheddingPolicy::LatencyBased {
            max_queue_latency_ms: 100.0,
        };
        let b = LoadSheddingPolicy::ResourceBased {
            cpu_threshold: 0.5,
            memory_threshold: 0.5,
        };
        assert_ne!(a, b);
    }

    // Test 42: ProbabilisticDrop with p=0.0 never drops
    #[test]
    fn test_probabilistic_drop_p0_never_drops() {
        let config = LoadSheddingConfig {
            policy: LoadSheddingPolicy::ProbabilisticDrop {
                drop_probability: 0.0,
            },
            cooldown_ms: 0,
            ..LoadSheddingConfig::default()
        };
        let mut shedder = LoadShedder::new(config);
        for _ in 0..20 {
            let d = shedder.should_shed(2, &zero_load());
            assert_eq!(d, SheddingDecision::Accept, "p=0.0 should never drop");
        }
    }
}
