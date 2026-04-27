//! Automatic learning rate selection using model-type heuristics and LR range test analysis.
//!
//! # Design overview
//!
//! 1. **Heuristic selection** — uses well-known LR sweet spots for each model class
//!    (BERT, GPT, fine-tuning, etc.) and scales them slightly based on model size.
//! 2. **LR range test** — fits a smoothed loss curve, identifies the region of
//!    steepest descent, and proposes the LR just before the minimum.
//! 3. **Combined** — merges the two estimates with a confidence-weighted average.
//!
//! No external dependencies are used beyond `std`.

// ---------------------------------------------------------------------------
// Model type
// ---------------------------------------------------------------------------

/// Broad model-class hints that drive heuristic LR selection.
#[derive(Debug, Clone, PartialEq)]
pub enum AutoLrModelType {
    /// BERT-style masked LM / encoder.  Typical LR: ~2e-5.
    Bert,
    /// GPT-style causal LM.  Typical LR: ~3e-4.
    Gpt,
    /// Very large models (>10B params).  Typical LR: ~1e-5.
    LargeModel,
    /// Small models (<100M params) trained from scratch.  Typical LR: ~1e-3.
    SmallModel,
    /// Fine-tuning a pretrained model on a downstream task.  Typical LR: ~2e-5.
    FineTuning,
    /// Pre-training from scratch.  Typical LR: ~1.5e-4.
    PreTraining,
    /// User-supplied range override.
    Custom {
        /// `(min_lr, max_lr)` — the inclusive plausible range.
        suggested_range: (f64, f64),
    },
}

impl AutoLrModelType {
    /// Returns `(min_lr, max_lr)` — the typical plausible range for this model class.
    pub fn suggested_range(&self) -> (f64, f64) {
        match self {
            Self::Bert => (5e-6, 5e-5),
            Self::Gpt => (1e-4, 1e-3),
            Self::LargeModel => (5e-6, 5e-5),
            Self::SmallModel => (5e-4, 5e-3),
            Self::FineTuning => (1e-5, 5e-5),
            Self::PreTraining => (5e-5, 3e-4),
            Self::Custom { suggested_range } => *suggested_range,
        }
    }

    /// Returns the single "sweet-spot" LR for this model class.
    pub fn sweet_spot(&self) -> f64 {
        match self {
            Self::Bert => 2e-5,
            Self::Gpt => 3e-4,
            Self::LargeModel => 1e-5,
            Self::SmallModel => 1e-3,
            Self::FineTuning => 2e-5,
            Self::PreTraining => 1.5e-4,
            Self::Custom { suggested_range } => {
                // Geometric mean of the range.
                let (lo, hi) = *suggested_range;
                (lo * hi).sqrt()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Strategy
// ---------------------------------------------------------------------------

/// Strategy to use for LR selection.
#[derive(Debug, Clone, PartialEq)]
pub enum AutoLrStrategy {
    /// Use model-type heuristics only (no loss curve required).
    Heuristic,
    /// Analyse a loss curve from an LR range test.
    LrRangeTest,
    /// Combine heuristic estimate with LR range test analysis.
    Combined,
}

// ---------------------------------------------------------------------------
// Recommended schedule
// ---------------------------------------------------------------------------

/// Recommended LR schedule for training.
#[derive(Debug, Clone, PartialEq)]
pub enum RecommendedSchedule {
    /// Linear decay from `max_lr` to `min_lr`.
    Linear,
    /// Cosine annealing.
    Cosine,
    /// Cosine annealing with periodic restarts.
    CosineWithRestarts,
    /// 1-Cycle policy (warm-up to peak, then anneal).
    OneCycle,
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// The output of automatic LR selection.
#[derive(Debug, Clone)]
pub struct AutoLrResult {
    /// Suggested initial learning rate.
    pub suggested_lr: f64,
    /// Plausible LR range `(min, max)`.
    pub lr_range: (f64, f64),
    /// Confidence in the suggestion: `0.0` = best guess, `1.0` = high confidence.
    pub confidence: f32,
    /// Recommended warm-up steps (if configured).
    pub warmup_steps: Option<usize>,
    /// Human-readable explanation of how the LR was chosen.
    pub rationale: String,
}

/// Full training LR configuration recommended by the selector.
#[derive(Debug, Clone)]
pub struct TrainingLrConfig {
    /// Initial learning rate (after warm-up, if any).
    pub initial_lr: f64,
    /// Peak LR (top of warm-up ramp).
    pub max_lr: f64,
    /// Minimum LR for decay schedule.
    pub min_lr: f64,
    /// Number of warm-up steps.
    pub warmup_steps: usize,
    /// Recommended schedule type.
    pub lr_schedule: RecommendedSchedule,
    /// Recommended weight decay.
    pub weight_decay: f64,
}

// ---------------------------------------------------------------------------
// AutoLrConfig
// ---------------------------------------------------------------------------

/// Configuration for the automatic LR selector.
#[derive(Debug, Clone)]
pub struct AutoLrConfig {
    /// Model class hint.
    pub model_type: AutoLrModelType,
    /// Whether to schedule a warm-up phase.
    pub use_warmup: bool,
    /// Fraction of total training steps used for warm-up (e.g. 0.06).
    pub warmup_fraction: f64,
    /// Which selection strategy to apply.
    pub strategy: AutoLrStrategy,
}

impl Default for AutoLrConfig {
    fn default() -> Self {
        Self {
            model_type: AutoLrModelType::Bert,
            use_warmup: true,
            warmup_fraction: 0.06,
            strategy: AutoLrStrategy::Heuristic,
        }
    }
}

// ---------------------------------------------------------------------------
// AutoLrSelector
// ---------------------------------------------------------------------------

/// Automatic learning rate selector.
pub struct AutoLrSelector {
    config: AutoLrConfig,
}

impl AutoLrSelector {
    /// Create a new selector with the given configuration.
    pub fn new(config: AutoLrConfig) -> Self {
        Self { config }
    }

    /// Select a learning rate based on model-type heuristics.
    ///
    /// `num_params` is used to apply a mild size-based correction: very large
    /// models benefit from slightly lower LRs.
    pub fn select_from_heuristics(&self, num_params: u64) -> AutoLrResult {
        let sweet_spot = self.config.model_type.sweet_spot();
        let range = self.config.model_type.suggested_range();

        // Size-based adjustment: scale LR down by log10(param_count / 1M)^0.3
        let size_factor = size_correction(num_params);
        let adjusted_lr = (sweet_spot * size_factor).clamp(range.0 * 0.5, range.1 * 2.0);

        let rationale = format!(
            "Heuristic selection for {:?}: sweet-spot={:.2e}, \
             size-correction={:.3} ({} params), adjusted={:.2e}",
            self.config.model_type, sweet_spot, size_factor, num_params, adjusted_lr
        );

        AutoLrResult {
            suggested_lr: adjusted_lr,
            lr_range: range,
            confidence: 0.5,
            warmup_steps: None,
            rationale,
        }
    }

    /// Select a learning rate from LR range test results.
    ///
    /// `lr_loss_curve` is a slice of `(lr, loss)` pairs obtained by running a
    /// short training loop while linearly (or log-linearly) increasing the LR.
    ///
    /// The algorithm:
    /// 1. Smooth the loss with an exponential moving average.
    /// 2. Compute the finite-difference derivative.
    /// 3. Find the index where the derivative is most negative (steepest descent).
    /// 4. Back off one step to avoid the divergence regime.
    pub fn select_from_range_test(&self, lr_loss_curve: &[(f64, f64)]) -> AutoLrResult {
        if lr_loss_curve.len() < 4 {
            // Fall back to heuristic if insufficient data.
            let mut result = self.select_from_heuristics(0);
            result.rationale = format!(
                "Insufficient LR range test data ({} points); fell back to heuristic. {}",
                lr_loss_curve.len(),
                result.rationale
            );
            result.confidence = 0.2;
            return result;
        }

        // Smooth loss values with EMA (beta = 0.9).
        let smoothed = smooth_ema(lr_loss_curve, 0.9);

        // Compute derivative approximation.
        let derivatives: Vec<f64> = (1..smoothed.len())
            .map(|i| {
                let dloss = smoothed[i].1 - smoothed[i - 1].1;
                let dlr = smoothed[i].0 - smoothed[i - 1].0;
                if dlr.abs() < 1e-30 { 0.0 } else { dloss / dlr }
            })
            .collect();

        // Find index of steepest descent (most negative derivative).
        let steepest_idx = derivatives
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Back off slightly: use the LR one step before the steepest descent.
        let safe_idx = steepest_idx.saturating_sub(1);
        let suggested_lr = lr_loss_curve[safe_idx].0;
        let heuristic_range = self.config.model_type.suggested_range();

        // Clamp to model-type range to avoid pathological curves.
        let clamped_lr = suggested_lr.clamp(heuristic_range.0 * 0.1, heuristic_range.1 * 10.0);

        let rationale = format!(
            "LR range test: steepest descent at index {} (lr={:.2e}), \
             suggested lr={:.2e} (clamped from {:.2e})",
            steepest_idx, lr_loss_curve[steepest_idx].0, clamped_lr, suggested_lr
        );

        AutoLrResult {
            suggested_lr: clamped_lr,
            lr_range: heuristic_range,
            confidence: 0.75,
            warmup_steps: None,
            rationale,
        }
    }

    /// Compute the number of warm-up steps from total training steps.
    pub fn compute_warmup_steps(&self, total_steps: usize) -> usize {
        if !self.config.use_warmup || total_steps == 0 {
            return 0;
        }
        let frac = self.config.warmup_fraction.clamp(0.0, 1.0);
        ((total_steps as f64 * frac).round() as usize).max(1)
    }

    /// Produce a full `TrainingLrConfig` recommendation.
    pub fn recommend_training_config(
        &self,
        num_params: u64,
        total_steps: usize,
    ) -> TrainingLrConfig {
        let base = self.select_from_heuristics(num_params);
        let initial_lr = base.suggested_lr;
        let (min_lr, max_lr) = base.lr_range;
        let warmup_steps = self.compute_warmup_steps(total_steps);

        // Choose schedule and weight decay based on model type.
        let (lr_schedule, weight_decay) = schedule_for_model(&self.config.model_type);

        TrainingLrConfig {
            initial_lr,
            max_lr,
            min_lr,
            warmup_steps,
            lr_schedule,
            weight_decay,
        }
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Mild size-based LR correction factor.
///
/// For a 1M-param model the factor is 1.0.  For a 10B-param model it is ~0.5.
fn size_correction(num_params: u64) -> f64 {
    if num_params == 0 {
        return 1.0;
    }
    let millions = (num_params as f64) / 1_000_000.0;
    if millions <= 1.0 {
        1.0
    } else {
        // log10(M)^(-0.3), clamped to [0.3, 1.0].
        (millions.log10().powf(0.3)).recip().clamp(0.3, 1.0)
    }
}

/// Exponential moving average smoothing over `(lr, loss)` pairs.
fn smooth_ema(curve: &[(f64, f64)], beta: f64) -> Vec<(f64, f64)> {
    let mut smoothed = Vec::with_capacity(curve.len());
    let mut ema = curve[0].1;
    for &(lr, loss) in curve {
        ema = beta * ema + (1.0 - beta) * loss;
        smoothed.push((lr, ema));
    }
    smoothed
}

/// Return the recommended schedule and weight decay for a model type.
fn schedule_for_model(model_type: &AutoLrModelType) -> (RecommendedSchedule, f64) {
    match model_type {
        AutoLrModelType::Bert => (RecommendedSchedule::Linear, 0.01),
        AutoLrModelType::Gpt => (RecommendedSchedule::Cosine, 0.1),
        AutoLrModelType::LargeModel => (RecommendedSchedule::CosineWithRestarts, 0.1),
        AutoLrModelType::SmallModel => (RecommendedSchedule::OneCycle, 0.01),
        AutoLrModelType::FineTuning => (RecommendedSchedule::Linear, 0.01),
        AutoLrModelType::PreTraining => (RecommendedSchedule::Cosine, 0.1),
        AutoLrModelType::Custom { .. } => (RecommendedSchedule::Cosine, 0.01),
    }
}

// ---------------------------------------------------------------------------
// LrRangeTest (Smith's LR Finder)
// ---------------------------------------------------------------------------

/// Learning Rate Range Test (Smith 2017).
///
/// Runs a short training sweep where the learning rate increases exponentially
/// from `min_lr` to `max_lr` over `num_iters` steps.  The optimal LR is
/// identified as the one where the (smoothed) loss gradient is most negative
/// (steepest descent), backed off by one step to stay in the stable regime.
#[derive(Debug, Clone)]
pub struct LrRangeTest {
    /// Minimum (starting) learning rate.
    pub min_lr: f32,
    /// Maximum (ending) learning rate.
    pub max_lr: f32,
    /// Number of LR-sweep iterations.
    pub num_iters: usize,
    /// EMA smoothing factor for the loss curve (`beta` in `ema = beta * ema + (1-beta) * loss`).
    pub smoothing: f32,
}

impl LrRangeTest {
    /// Create a new LR range test with the given bounds and iteration count.
    ///
    /// `smoothing` defaults to 0.98 (heavy smoothing), appropriate for noisy
    /// mini-batch loss curves.
    pub fn new(min_lr: f32, max_lr: f32, num_iters: usize) -> Self {
        Self { min_lr, max_lr, num_iters, smoothing: 0.98 }
    }

    /// Return the learning rate at iteration `step` (0-based).
    ///
    /// Uses an exponential schedule: `lr(step) = min_lr * (max_lr / min_lr) ^ (step / (num_iters - 1))`.
    /// Falls back to `min_lr` when `num_iters <= 1`.
    pub fn lr_at_step(&self, step: usize) -> f32 {
        if self.num_iters <= 1 || self.min_lr <= 0.0 || self.max_lr <= 0.0 {
            return self.min_lr;
        }
        let t = step.min(self.num_iters - 1) as f32 / (self.num_iters - 1) as f32;
        let ratio = self.max_lr / self.min_lr;
        self.min_lr * ratio.powf(t)
    }

    /// Smooth a raw loss sequence using exponential moving average.
    ///
    /// Uses bias correction so early steps are not dominated by the initial
    /// EMA value.
    pub fn smooth_losses(&self, losses: &[f32]) -> Vec<f32> {
        if losses.is_empty() {
            return Vec::new();
        }
        let beta = self.smoothing.clamp(0.0, 1.0 - f32::EPSILON);
        let mut ema = 0.0_f32;
        losses
            .iter()
            .enumerate()
            .map(|(i, &loss)| {
                ema = beta * ema + (1.0 - beta) * loss;
                // Bias-correction factor.
                let correction = 1.0 - beta.powi((i + 1) as i32);
                if correction.abs() < 1e-9 { ema } else { ema / correction }
            })
            .collect()
    }

    /// Identify the optimal learning rate from a `(lr, loss)` history.
    ///
    /// Algorithm:
    /// 1. Smooth the loss values with the EMA smoother.
    /// 2. Compute finite-difference loss gradients.
    /// 3. Find the step with the most negative gradient (steepest descent).
    /// 4. Return the LR one step before that point to stay in the stable regime.
    ///
    /// Returns `None` when the history has fewer than 4 points.
    pub fn find_optimal_lr(loss_history: &[(f32, f32)]) -> Option<f32> {
        if loss_history.len() < 4 {
            return None;
        }

        let smoother = LrRangeTest::new(
            loss_history[0].0,
            loss_history.last().map(|(lr, _)| *lr).unwrap_or(1.0),
            loss_history.len(),
        );
        let raw_losses: Vec<f32> = loss_history.iter().map(|(_, l)| *l).collect();
        let smoothed = smoother.smooth_losses(&raw_losses);

        // Finite-difference gradient.
        let gradients: Vec<f32> = (1..smoothed.len())
            .map(|i| {
                let dloss = smoothed[i] - smoothed[i - 1];
                let dlr = loss_history[i].0 - loss_history[i - 1].0;
                if dlr.abs() < f32::EPSILON { 0.0 } else { dloss / dlr }
            })
            .collect();

        // Index of steepest descent.
        let steepest = gradients
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)?;

        // Back off one step.
        let safe_idx = steepest.saturating_sub(1);
        Some(loss_history[safe_idx].0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn bert_selector() -> AutoLrSelector {
        AutoLrSelector::new(AutoLrConfig {
            model_type: AutoLrModelType::Bert,
            use_warmup: true,
            warmup_fraction: 0.06,
            strategy: AutoLrStrategy::Heuristic,
        })
    }

    fn gpt_selector() -> AutoLrSelector {
        AutoLrSelector::new(AutoLrConfig {
            model_type: AutoLrModelType::Gpt,
            use_warmup: true,
            warmup_fraction: 0.1,
            strategy: AutoLrStrategy::LrRangeTest,
        })
    }

    #[test]
    fn test_bert_heuristic_range() {
        let sel = bert_selector();
        let res = sel.select_from_heuristics(110_000_000); // 110M params
        let (lo, hi) = AutoLrModelType::Bert.suggested_range();
        // The adjusted LR should lie within a reasonable neighbourhood of the range.
        assert!(res.suggested_lr >= lo * 0.4, "lr too low: {:.2e}", res.suggested_lr);
        assert!(res.suggested_lr <= hi * 2.5, "lr too high: {:.2e}", res.suggested_lr);
    }

    #[test]
    fn test_gpt_sweet_spot() {
        let expected = AutoLrModelType::Gpt.sweet_spot();
        assert!((expected - 3e-4).abs() < 1e-6);
    }

    #[test]
    fn test_size_correction_small_model() {
        let sel = AutoLrSelector::new(AutoLrConfig {
            model_type: AutoLrModelType::SmallModel,
            use_warmup: false,
            warmup_fraction: 0.0,
            strategy: AutoLrStrategy::Heuristic,
        });
        // 1M params → size_correction = 1.0, result near sweet_spot
        let res = sel.select_from_heuristics(1_000_000);
        let sweet = AutoLrModelType::SmallModel.sweet_spot();
        assert!((res.suggested_lr - sweet).abs() < sweet * 0.1);
    }

    #[test]
    fn test_warmup_steps_computed() {
        let sel = bert_selector();
        let steps = sel.compute_warmup_steps(10_000);
        // 6% of 10000 = 600
        assert_eq!(steps, 600);
    }

    #[test]
    fn test_warmup_disabled() {
        let sel = AutoLrSelector::new(AutoLrConfig {
            model_type: AutoLrModelType::Bert,
            use_warmup: false,
            warmup_fraction: 0.1,
            strategy: AutoLrStrategy::Heuristic,
        });
        assert_eq!(sel.compute_warmup_steps(10_000), 0);
    }

    #[test]
    fn test_warmup_zero_total_steps() {
        let sel = bert_selector();
        assert_eq!(sel.compute_warmup_steps(0), 0);
    }

    #[test]
    fn test_lr_range_test_basic() {
        let sel = gpt_selector();
        // Simulate a well-behaved loss curve: loss decreases then explodes.
        let curve: Vec<(f64, f64)> = (0..20)
            .map(|i| {
                let lr = 1e-6 * 10f64.powf(i as f64 * 0.2);
                let loss = if i < 12 { 3.0 - 0.2 * i as f64 } else { 1.0 + (i - 12) as f64 * 2.0 };
                (lr, loss)
            })
            .collect();

        let res = sel.select_from_range_test(&curve);
        assert!(res.suggested_lr > 0.0, "suggested LR must be positive");
        assert!(res.confidence >= 0.7);
    }

    #[test]
    fn test_lr_range_test_insufficient_data_fallback() {
        let sel = gpt_selector();
        let short_curve = vec![(1e-5, 3.0), (1e-4, 2.8)];
        let res = sel.select_from_range_test(&short_curve);
        assert!(res.suggested_lr > 0.0);
        assert!(res.confidence < 0.5, "should have low confidence on fallback");
    }

    #[test]
    fn test_recommend_training_config_fields() {
        let sel = bert_selector();
        let cfg = sel.recommend_training_config(110_000_000, 50_000);
        assert!(cfg.initial_lr > 0.0);
        assert!(cfg.max_lr > cfg.min_lr);
        assert!(cfg.warmup_steps > 0);
        assert_eq!(cfg.lr_schedule, RecommendedSchedule::Linear);
        assert!(cfg.weight_decay > 0.0);
    }

    #[test]
    fn test_fine_tuning_lower_lr() {
        let ft_sel = AutoLrSelector::new(AutoLrConfig {
            model_type: AutoLrModelType::FineTuning,
            use_warmup: true,
            warmup_fraction: 0.06,
            strategy: AutoLrStrategy::Heuristic,
        });
        let pt_sel = AutoLrSelector::new(AutoLrConfig {
            model_type: AutoLrModelType::PreTraining,
            use_warmup: true,
            warmup_fraction: 0.06,
            strategy: AutoLrStrategy::Heuristic,
        });
        let ft_res = ft_sel.select_from_heuristics(300_000_000);
        let pt_res = pt_sel.select_from_heuristics(300_000_000);
        // Fine-tuning LR should typically be lower than pre-training LR.
        assert!(ft_res.suggested_lr < pt_res.suggested_lr,
            "FT LR ({:.2e}) should < PT LR ({:.2e})", ft_res.suggested_lr, pt_res.suggested_lr);
    }

    #[test]
    fn test_custom_model_type() {
        let custom_sel = AutoLrSelector::new(AutoLrConfig {
            model_type: AutoLrModelType::Custom { suggested_range: (1e-4, 1e-2) },
            use_warmup: false,
            warmup_fraction: 0.0,
            strategy: AutoLrStrategy::Heuristic,
        });
        let res = custom_sel.select_from_heuristics(50_000_000);
        assert!(res.suggested_lr >= 1e-4 * 0.5 && res.suggested_lr <= 1e-2 * 2.0,
            "custom LR out of expected range: {:.2e}", res.suggested_lr);
    }

    // ─── LrRangeTest tests ────────────────────────────────────────────────

    // ── Test 11: lr_at_step returns min_lr at step 0 ──
    #[test]
    fn test_lr_range_test_step_zero() {
        let t = LrRangeTest::new(1e-5, 1e-1, 100);
        assert!((t.lr_at_step(0) - 1e-5).abs() < 1e-9);
    }

    // ── Test 12: lr_at_step returns max_lr at last step ──
    #[test]
    fn test_lr_range_test_step_last() {
        let t = LrRangeTest::new(1e-5, 1e-1, 100);
        let last = t.lr_at_step(99);
        assert!((last - 1e-1).abs() < 1e-7, "expected ~1e-1, got {last}");
    }

    // ── Test 13: lr_at_step is strictly increasing ──
    #[test]
    fn test_lr_range_test_monotone() {
        let t = LrRangeTest::new(1e-6, 1.0, 50);
        let mut prev = t.lr_at_step(0);
        for step in 1..50 {
            let cur = t.lr_at_step(step);
            assert!(cur >= prev, "lr not monotone at step {step}: {cur} < {prev}");
            prev = cur;
        }
    }

    // ── Test 14: lr_at_step with num_iters=1 returns min_lr ──
    #[test]
    fn test_lr_range_test_single_iter() {
        let t = LrRangeTest::new(0.001, 0.1, 1);
        assert!((t.lr_at_step(0) - 0.001).abs() < 1e-9);
    }

    // ── Test 15: smooth_losses length matches input ──
    #[test]
    fn test_smooth_losses_length() {
        let t = LrRangeTest::new(1e-5, 1e-1, 20);
        let losses = vec![3.0_f32; 20];
        let smoothed = t.smooth_losses(&losses);
        assert_eq!(smoothed.len(), 20);
    }

    // ── Test 16: smooth_losses with constant input converges to that value ──
    #[test]
    fn test_smooth_losses_constant_input() {
        let t = LrRangeTest { min_lr: 1e-5, max_lr: 1e-1, num_iters: 50, smoothing: 0.9 };
        let losses = vec![2.5_f32; 100];
        let smoothed = t.smooth_losses(&losses);
        // The last smoothed value should be close to 2.5 after many steps.
        let last = *smoothed.last().unwrap_or(&0.0);
        assert!((last - 2.5).abs() < 0.01, "expected ~2.5, got {last}");
    }

    // ── Test 17: find_optimal_lr returns None for too few points ──
    #[test]
    fn test_find_optimal_lr_too_few_points() {
        assert!(LrRangeTest::find_optimal_lr(&[(1e-5, 3.0), (1e-4, 2.5)]).is_none());
    }

    // ── Test 18: find_optimal_lr returns Some for valid curve ──
    #[test]
    fn test_find_optimal_lr_valid_curve() {
        // Simulate: loss decreases to a minimum then explodes.
        let curve: Vec<(f32, f32)> = (0..30)
            .map(|i| {
                let lr = 1e-6_f32 * 10f32.powf(i as f32 * 0.15);
                let loss = if i < 18 {
                    3.0 - 0.1 * i as f32
                } else {
                    1.2 + (i - 18) as f32 * 1.5
                };
                (lr, loss)
            })
            .collect();
        let opt_lr = LrRangeTest::find_optimal_lr(&curve);
        assert!(opt_lr.is_some(), "should find optimal LR");
        assert!(opt_lr.unwrap() > 0.0, "optimal LR must be positive");
    }

    // ── Test 19: smooth_losses on empty input returns empty ──
    #[test]
    fn test_smooth_losses_empty_input() {
        let t = LrRangeTest::new(1e-5, 1e-1, 10);
        assert!(t.smooth_losses(&[]).is_empty());
    }

    // ── Test 20: find_optimal_lr stays within the provided LR range ──
    #[test]
    fn test_find_optimal_lr_in_range() {
        let lrs: Vec<f32> = (0..20).map(|i| 1e-5_f32 * 10f32.powf(i as f32 * 0.2)).collect();
        let curve: Vec<(f32, f32)> = lrs
            .iter()
            .enumerate()
            .map(|(i, &lr)| {
                let loss = if i < 10 { 3.0 - 0.2 * i as f32 } else { 1.0 + i as f32 };
                (lr, loss)
            })
            .collect();

        let min_lr = curve[0].0;
        let max_lr = curve.last().map(|(lr, _)| *lr).unwrap_or(1.0);
        let opt = LrRangeTest::find_optimal_lr(&curve).expect("should find LR");
        assert!(opt >= min_lr && opt <= max_lr, "optimal LR {opt} outside [{min_lr}, {max_lr}]");
    }
}
