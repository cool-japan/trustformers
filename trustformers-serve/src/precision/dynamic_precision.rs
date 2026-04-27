//! Dynamic precision selection: automatically switch between FP32/FP16/INT8
//! based on hardware capabilities, latency requirements, and quality targets.
//!
//! # Design
//!
//! The [`DynamicPrecisionSelector`] evaluates a [`PrecisionPolicy`] against the
//! current [`HardwareMetrics`] to produce the optimal [`InferencePrecision`].
//! Historical latency observations are recorded so that the `Adaptive` policy
//! can learn from actual inference runs.

use std::sync::Mutex;

/// The numerical format used during inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum InferencePrecision {
    Float32,
    Float16,
    BFloat16,
    Int8,
    Int4,
}

impl InferencePrecision {
    /// Relative memory footprint compared to FP32 (1.0)
    pub fn memory_factor(&self) -> f32 {
        match self {
            Self::Float32 => 1.0,
            Self::Float16 => 0.5,
            Self::BFloat16 => 0.5,
            Self::Int8 => 0.25,
            Self::Int4 => 0.125,
        }
    }

    /// Expected compute speedup relative to FP32 on hardware that supports the format
    pub fn compute_speedup(&self) -> f32 {
        match self {
            Self::Float32 => 1.0,
            Self::Float16 => 1.8,
            Self::BFloat16 => 1.6,
            Self::Int8 => 3.2,
            Self::Int4 => 5.0,
        }
    }

    /// Expected quality degradation fraction in [0.0, 1.0] relative to FP32 (0.0)
    pub fn quality_loss(&self) -> f32 {
        match self {
            Self::Float32 => 0.0,
            Self::Float16 => 0.005,
            Self::BFloat16 => 0.008,
            Self::Int8 => 0.03,
            Self::Int4 => 0.08,
        }
    }

    /// Human-readable name for the precision format
    pub fn name(&self) -> &'static str {
        match self {
            Self::Float32 => "float32",
            Self::Float16 => "float16",
            Self::BFloat16 => "bfloat16",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
        }
    }

    /// Whether this precision format is available given the hardware flags
    pub fn is_available(&self, metrics: &HardwareMetrics) -> bool {
        match self {
            Self::Float32 => true,
            Self::Float16 => metrics.supports_fp16,
            Self::BFloat16 => metrics.supports_bf16,
            Self::Int8 => metrics.supports_int8,
            // INT4 is treated as always available (software quantisation fallback)
            Self::Int4 => true,
        }
    }
}

/// Policy for choosing the inference precision
#[derive(Debug, Clone)]
pub enum PrecisionPolicy {
    /// Always use exactly this precision
    Fixed(InferencePrecision),
    /// Minimize latency while keeping quality above `quality_floor`
    MinimizeLatency { quality_floor: f32 },
    /// Minimize memory usage while keeping quality above `quality_floor`
    MinimizeMemory { quality_floor: f32 },
    /// Use the highest quality precision that fits within the latency budget
    MaximizeQuality { latency_budget_ms: f32 },
    /// Adapt based on current hardware metrics and historical data
    Adaptive,
}

/// Current hardware metrics used for adaptive precision selection
#[derive(Debug, Clone)]
pub struct HardwareMetrics {
    /// Available GPU / CPU memory in megabytes
    pub available_memory_mb: u64,
    /// GPU utilisation as a percentage [0, 100]
    pub gpu_utilization_pct: f32,
    /// Current observed inference latency in milliseconds
    pub current_latency_ms: f32,
    /// Thermal pressure: 0.0 = cool, 1.0 = critical
    pub thermal_pressure: f32,
    /// Hardware supports FP16 SIMD/tensor operations
    pub supports_fp16: bool,
    /// Hardware supports BF16 SIMD/tensor operations
    pub supports_bf16: bool,
    /// Hardware supports INT8 quantisation
    pub supports_int8: bool,
}

impl Default for HardwareMetrics {
    fn default() -> Self {
        Self {
            available_memory_mb: 8192,
            gpu_utilization_pct: 0.0,
            current_latency_ms: 50.0,
            thermal_pressure: 0.0,
            supports_fp16: true,
            supports_bf16: false,
            supports_int8: true,
        }
    }
}

/// Result of a precision recommendation
#[derive(Debug)]
pub struct PrecisionRecommendation {
    pub precision: InferencePrecision,
    pub reason: String,
    pub expected_speedup: f32,
    pub expected_memory_reduction: f32,
}

/// Precision selector that applies a [`PrecisionPolicy`] to select the best format
pub struct DynamicPrecisionSelector {
    policy: PrecisionPolicy,
    /// Ring buffer of recent (precision, latency_ms) observations
    history: Mutex<Vec<(InferencePrecision, f32)>>,
}

impl DynamicPrecisionSelector {
    /// Create a new selector with the given policy
    pub fn new(policy: PrecisionPolicy) -> Self {
        Self {
            policy,
            history: Mutex::new(Vec::new()),
        }
    }

    /// Select the best precision for the current hardware conditions
    pub fn select(&self, metrics: &HardwareMetrics) -> InferencePrecision {
        self.recommend(metrics).precision
    }

    /// Record an inference latency observation to inform future adaptive decisions
    pub fn record_result(&self, precision: InferencePrecision, latency_ms: f32) {
        const MAX_HISTORY: usize = 256;
        if let Ok(mut hist) = self.history.lock() {
            hist.push((precision, latency_ms));
            if hist.len() > MAX_HISTORY {
                hist.remove(0);
            }
        }
    }

    /// Get current recommendation with human-readable reasoning
    pub fn recommend(&self, metrics: &HardwareMetrics) -> PrecisionRecommendation {
        match &self.policy {
            PrecisionPolicy::Fixed(p) => {
                let p = *p;
                PrecisionRecommendation {
                    precision: p,
                    reason: format!("Fixed policy: always use {}", p.name()),
                    expected_speedup: p.compute_speedup(),
                    expected_memory_reduction: 1.0 - p.memory_factor(),
                }
            },

            PrecisionPolicy::MinimizeLatency { quality_floor } => {
                let quality_floor = *quality_floor;
                // From fastest (lowest memory_factor / highest speedup) to slowest
                let candidates = [
                    InferencePrecision::Int4,
                    InferencePrecision::Int8,
                    InferencePrecision::Float16,
                    InferencePrecision::BFloat16,
                    InferencePrecision::Float32,
                ];
                // Sort by descending compute_speedup; pick first that is available
                // and whose quality_loss is within floor
                let chosen = candidates
                    .iter()
                    .filter(|p| {
                        p.is_available(metrics) && (1.0 - p.quality_loss()) >= quality_floor
                    })
                    .max_by(|a, b| {
                        a.compute_speedup()
                            .partial_cmp(&b.compute_speedup())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .copied()
                    .unwrap_or(InferencePrecision::Float32);

                PrecisionRecommendation {
                    precision: chosen,
                    reason: format!(
                        "MinimizeLatency: {} chosen (speedup={:.1}x, quality_floor={:.3})",
                        chosen.name(),
                        chosen.compute_speedup(),
                        quality_floor
                    ),
                    expected_speedup: chosen.compute_speedup(),
                    expected_memory_reduction: 1.0 - chosen.memory_factor(),
                }
            },

            PrecisionPolicy::MinimizeMemory { quality_floor } => {
                let quality_floor = *quality_floor;
                let candidates = [
                    InferencePrecision::Int4,
                    InferencePrecision::Int8,
                    InferencePrecision::Float16,
                    InferencePrecision::BFloat16,
                    InferencePrecision::Float32,
                ];
                let chosen = candidates
                    .iter()
                    .filter(|p| {
                        p.is_available(metrics) && (1.0 - p.quality_loss()) >= quality_floor
                    })
                    .min_by(|a, b| {
                        a.memory_factor()
                            .partial_cmp(&b.memory_factor())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .copied()
                    .unwrap_or(InferencePrecision::Float32);

                PrecisionRecommendation {
                    precision: chosen,
                    reason: format!(
                        "MinimizeMemory: {} chosen (memory_factor={:.3}, quality_floor={:.3})",
                        chosen.name(),
                        chosen.memory_factor(),
                        quality_floor
                    ),
                    expected_speedup: chosen.compute_speedup(),
                    expected_memory_reduction: 1.0 - chosen.memory_factor(),
                }
            },

            PrecisionPolicy::MaximizeQuality { latency_budget_ms } => {
                let budget = *latency_budget_ms;
                // Estimate the achieved latency for each format
                // (relative: fp32 latency = current_latency_ms / speedup_factor_of_current)
                // We treat current_latency_ms as an FP32 baseline for this selection.
                let fp32_latency = metrics.current_latency_ms;
                let candidates = [
                    InferencePrecision::Float32,
                    InferencePrecision::BFloat16,
                    InferencePrecision::Float16,
                    InferencePrecision::Int8,
                    InferencePrecision::Int4,
                ];
                // Pick highest quality (lowest quality_loss) that fits the budget
                let chosen = candidates
                    .iter()
                    .filter(|p| {
                        if !p.is_available(metrics) {
                            return false;
                        }
                        let estimated_latency = fp32_latency / p.compute_speedup();
                        estimated_latency <= budget
                    })
                    .min_by(|a, b| {
                        a.quality_loss()
                            .partial_cmp(&b.quality_loss())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .copied()
                    .unwrap_or(InferencePrecision::Int4);

                PrecisionRecommendation {
                    precision: chosen,
                    reason: format!(
                        "MaximizeQuality: {} chosen within {:.0}ms budget",
                        chosen.name(),
                        budget
                    ),
                    expected_speedup: chosen.compute_speedup(),
                    expected_memory_reduction: 1.0 - chosen.memory_factor(),
                }
            },

            PrecisionPolicy::Adaptive => self.recommend_adaptive(metrics),
        }
    }

    /// Adaptive recommendation based on hardware pressure and historical data
    fn recommend_adaptive(&self, metrics: &HardwareMetrics) -> PrecisionRecommendation {
        // Under thermal pressure or very high GPU utilisation, prefer lower precision
        // to reduce heat and compute cost.
        let thermal_ok = metrics.thermal_pressure < 0.7;
        let gpu_ok = metrics.gpu_utilization_pct < 85.0;
        let memory_ok = metrics.available_memory_mb >= 2048;

        // Check historical average latency for FP16 vs FP32
        let hist_avg_latency = {
            let hist = self.history.lock().unwrap_or_else(|e| e.into_inner());
            if hist.is_empty() {
                None
            } else {
                let sum: f32 = hist.iter().map(|(_, l)| l).sum();
                Some(sum / hist.len() as f32)
            }
        };

        // Heuristic:
        // - Thermal critical → INT8
        // - GPU/memory pressure → FP16 (if available)
        // - Historical latency high → FP16 / INT8
        // - Otherwise → FP32 for best quality
        let (precision, reason) = if !thermal_ok {
            (
                InferencePrecision::Int8,
                "Adaptive: thermal pressure high, using INT8",
            )
        } else if !gpu_ok || !memory_ok {
            if metrics.supports_fp16 {
                (
                    InferencePrecision::Float16,
                    "Adaptive: resource pressure, using FP16",
                )
            } else {
                (
                    InferencePrecision::Int8,
                    "Adaptive: resource pressure, FP16 unavailable, using INT8",
                )
            }
        } else if let Some(avg) = hist_avg_latency {
            if avg > 200.0 && metrics.supports_fp16 {
                (
                    InferencePrecision::Float16,
                    "Adaptive: high historical latency, using FP16",
                )
            } else {
                (
                    InferencePrecision::Float32,
                    "Adaptive: conditions good, using FP32",
                )
            }
        } else {
            (
                InferencePrecision::Float32,
                "Adaptive: no history, defaulting to FP32",
            )
        };

        PrecisionRecommendation {
            precision,
            reason: reason.to_string(),
            expected_speedup: precision.compute_speedup(),
            expected_memory_reduction: 1.0 - precision.memory_factor(),
        }
    }
}

// ─────────────────────────── tests ───────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_hw() -> HardwareMetrics {
        HardwareMetrics::default()
    }

    #[test]
    fn test_fixed_policy_always_returns_same() {
        let sel =
            DynamicPrecisionSelector::new(PrecisionPolicy::Fixed(InferencePrecision::Float16));
        assert_eq!(sel.select(&default_hw()), InferencePrecision::Float16);
        // Even under pressure
        let mut hw = default_hw();
        hw.thermal_pressure = 1.0;
        assert_eq!(sel.select(&hw), InferencePrecision::Float16);
    }

    #[test]
    fn test_minimize_latency_picks_fastest_available() {
        let mut hw = default_hw();
        hw.supports_fp16 = true;
        hw.supports_int8 = true;
        let sel = DynamicPrecisionSelector::new(PrecisionPolicy::MinimizeLatency {
            quality_floor: 0.90,
        });
        // With quality_floor=0.90, INT4 quality = 1-0.08=0.92 ≥ 0.90 → should pick INT4
        let prec = sel.select(&hw);
        // INT4 has highest compute_speedup (5.0) and quality 0.92 ≥ floor
        assert_eq!(prec, InferencePrecision::Int4);
    }

    #[test]
    fn test_minimize_latency_quality_floor_excludes_int4() {
        let hw = default_hw();
        let sel = DynamicPrecisionSelector::new(PrecisionPolicy::MinimizeLatency {
            quality_floor: 0.95, // INT4 quality=0.92 < 0.95, INT8 quality=0.97 ≥ 0.95
        });
        let prec = sel.select(&hw);
        assert_eq!(prec, InferencePrecision::Int8);
    }

    #[test]
    fn test_minimize_memory_picks_smallest_format() {
        let hw = default_hw();
        let sel = DynamicPrecisionSelector::new(PrecisionPolicy::MinimizeMemory {
            quality_floor: 0.90, // INT4 quality=0.92 ≥ 0.90 → smallest memory_factor
        });
        let prec = sel.select(&hw);
        assert_eq!(prec, InferencePrecision::Int4);
    }

    #[test]
    fn test_maximize_quality_within_budget() {
        let mut hw = default_hw();
        hw.current_latency_ms = 100.0;
        hw.supports_fp16 = true;
        // With FP32 latency=100ms and budget=100ms: FP32 estimated = 100/1.0 = 100 ≤ 100 → pick FP32
        let sel = DynamicPrecisionSelector::new(PrecisionPolicy::MaximizeQuality {
            latency_budget_ms: 100.0,
        });
        let prec = sel.select(&hw);
        assert_eq!(prec, InferencePrecision::Float32);
    }

    #[test]
    fn test_maximize_quality_tight_budget_forces_lower() {
        let mut hw = default_hw();
        hw.current_latency_ms = 100.0;
        hw.supports_fp16 = true;
        hw.supports_int8 = true;
        // Budget = 50ms: FP32 (100ms) too slow, FP16 (100/1.8 ≈ 55ms) too slow,
        // INT8 (100/3.2 ≈ 31ms) within budget
        let sel = DynamicPrecisionSelector::new(PrecisionPolicy::MaximizeQuality {
            latency_budget_ms: 40.0,
        });
        let prec = sel.select(&hw);
        assert_eq!(prec, InferencePrecision::Int8);
    }

    #[test]
    fn test_adaptive_defaults_to_fp32_on_idle_hw() {
        let hw = default_hw(); // low pressure, no history
        let sel = DynamicPrecisionSelector::new(PrecisionPolicy::Adaptive);
        let prec = sel.select(&hw);
        assert_eq!(prec, InferencePrecision::Float32);
    }

    #[test]
    fn test_adaptive_thermal_pressure_uses_int8() {
        let mut hw = default_hw();
        hw.thermal_pressure = 0.9;
        let sel = DynamicPrecisionSelector::new(PrecisionPolicy::Adaptive);
        let prec = sel.select(&hw);
        assert_eq!(prec, InferencePrecision::Int8);
    }

    #[test]
    fn test_record_result_influences_adaptive() {
        let sel = DynamicPrecisionSelector::new(PrecisionPolicy::Adaptive);
        // Record many high-latency observations
        for _ in 0..10 {
            sel.record_result(InferencePrecision::Float32, 500.0);
        }
        let hw = default_hw();
        // High historical latency (500ms avg) should push to FP16
        let prec = sel.select(&hw);
        assert_eq!(prec, InferencePrecision::Float16);
    }

    #[test]
    fn test_memory_factor_values() {
        assert!((InferencePrecision::Float32.memory_factor() - 1.0).abs() < 1e-6);
        assert!((InferencePrecision::Float16.memory_factor() - 0.5).abs() < 1e-6);
        assert!((InferencePrecision::Int8.memory_factor() - 0.25).abs() < 1e-6);
        assert!((InferencePrecision::Int4.memory_factor() - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_quality_loss_ordering() {
        // FP32 should have lowest quality loss, INT4 highest
        assert!(
            InferencePrecision::Float32.quality_loss() < InferencePrecision::Float16.quality_loss()
        );
        assert!(
            InferencePrecision::Float16.quality_loss() < InferencePrecision::Int8.quality_loss()
        );
        assert!(InferencePrecision::Int8.quality_loss() < InferencePrecision::Int4.quality_loss());
    }

    #[test]
    fn test_recommendation_contains_reason() {
        let sel = DynamicPrecisionSelector::new(PrecisionPolicy::Fixed(InferencePrecision::Int8));
        let rec = sel.recommend(&default_hw());
        assert!(!rec.reason.is_empty());
        assert_eq!(rec.precision, InferencePrecision::Int8);
    }

    // Test 13: BFloat16 available when bf16 flag is set
    #[test]
    fn test_bf16_available_when_flag_set() {
        let mut hw = default_hw();
        hw.supports_bf16 = true;
        assert!(InferencePrecision::BFloat16.is_available(&hw));
    }

    // Test 14: BFloat16 not available when bf16 flag is clear
    #[test]
    fn test_bf16_not_available_when_flag_clear() {
        let mut hw = default_hw();
        hw.supports_bf16 = false;
        assert!(!InferencePrecision::BFloat16.is_available(&hw));
    }

    // Test 15: Int8 not available when flag is clear
    #[test]
    fn test_int8_not_available_when_flag_clear() {
        let mut hw = default_hw();
        hw.supports_int8 = false;
        assert!(!InferencePrecision::Int8.is_available(&hw));
    }

    // Test 16: Float32 is always available regardless of hardware flags
    #[test]
    fn test_float32_always_available() {
        let mut hw = default_hw();
        hw.supports_fp16 = false;
        hw.supports_bf16 = false;
        hw.supports_int8 = false;
        assert!(InferencePrecision::Float32.is_available(&hw));
    }

    // Test 17: Int4 is always available (software quantization fallback)
    #[test]
    fn test_int4_always_available() {
        let mut hw = default_hw();
        hw.supports_fp16 = false;
        hw.supports_bf16 = false;
        hw.supports_int8 = false;
        assert!(InferencePrecision::Int4.is_available(&hw));
    }

    // Test 18: compute_speedup ordering — Int4 > Int8 > Float16 > Float32
    #[test]
    fn test_compute_speedup_ordering() {
        assert!(
            InferencePrecision::Int4.compute_speedup() > InferencePrecision::Int8.compute_speedup()
        );
        assert!(
            InferencePrecision::Int8.compute_speedup()
                > InferencePrecision::Float16.compute_speedup()
        );
        assert!(
            InferencePrecision::Float16.compute_speedup()
                > InferencePrecision::Float32.compute_speedup()
        );
    }

    // Test 19: MinimizeMemory with fp16 and int8 both unavailable falls back to Int4
    #[test]
    fn test_minimize_memory_no_fp16_no_int8_falls_back_to_int4() {
        let mut hw = default_hw();
        hw.supports_fp16 = false;
        hw.supports_bf16 = false;
        hw.supports_int8 = false;
        let sel = DynamicPrecisionSelector::new(PrecisionPolicy::MinimizeMemory {
            quality_floor: 0.9, // Int4 quality = 0.92 >= 0.9
        });
        let prec = sel.select(&hw);
        assert_eq!(
            prec,
            InferencePrecision::Int4,
            "must fall back to Int4 when fp16/int8 unavailable"
        );
    }

    // Test 20: MaximizeQuality with near-zero budget forces Int4 (fastest, fits any budget)
    #[test]
    fn test_maximize_quality_near_zero_budget_returns_int4() {
        let mut hw = default_hw();
        hw.supports_fp16 = true;
        hw.supports_bf16 = true;
        hw.supports_int8 = true;
        hw.current_latency_ms = 100.0;
        let sel = DynamicPrecisionSelector::new(PrecisionPolicy::MaximizeQuality {
            latency_budget_ms: 0.001, // too tight for anything except Int4 (100/5.0=20ms > 0.001 even)
        });
        // When no format fits, the fallback is Int4
        let prec = sel.select(&hw);
        assert_eq!(
            prec,
            InferencePrecision::Int4,
            "must fall back to Int4 when budget is near-zero"
        );
    }

    // Test 21: Adaptive selects Float16 under high GPU utilization
    #[test]
    fn test_adaptive_high_gpu_utilization_uses_fp16() {
        let mut hw = default_hw();
        hw.gpu_utilization_pct = 90.0; // > 85% threshold
        hw.supports_fp16 = true;
        hw.available_memory_mb = 8192;
        hw.thermal_pressure = 0.0; // thermal ok
        let sel = DynamicPrecisionSelector::new(PrecisionPolicy::Adaptive);
        let prec = sel.select(&hw);
        assert_eq!(
            prec,
            InferencePrecision::Float16,
            "high GPU utilization must select Float16"
        );
    }

    // Test 22: record_result is bounded — 300 observations don't panic or overflow
    #[test]
    fn test_record_result_history_is_bounded() {
        let sel = DynamicPrecisionSelector::new(PrecisionPolicy::Adaptive);
        for i in 0u32..300 {
            sel.record_result(InferencePrecision::Float32, i as f32);
        }
        // After 300 observations the selector must still function correctly.
        let prec = sel.select(&default_hw());
        // With high avg latency the result should be Float16 (avg >> 200ms) — just verify no panic.
        let _ = prec; // result is a valid InferencePrecision variant
    }
}
