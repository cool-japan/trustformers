//! Learning Rate Range Test (LR Finder) for automatic LR selection.
//!
//! The LR Finder runs a mini training loop with exponentially increasing LR,
//! records the loss at each step, and identifies the optimal LR as the point
//! just before the loss diverges.
//!
//! Reference: Smith (2015) "Cyclical Learning Rates for Training Neural Networks"
//!
//! # Algorithm
//! 1. Set initial LR to `lr_min`, target final LR to `lr_max`
//! 2. For each step: compute loss, record (lr, loss), multiply lr by `lr_max/lr_min^(1/num_steps)`
//! 3. Stop if loss > best_loss * diverge_threshold
//! 4. Plot smoothed loss vs log(lr) to find optimal LR

use trustformers_core::errors::{Result, TrustformersError};

/// Configuration for the LR Finder
#[derive(Debug, Clone)]
pub struct LrFinderConfig {
    /// Starting (minimum) learning rate
    pub lr_min: f64,
    /// Ending (maximum) learning rate
    pub lr_max: f64,
    /// Number of steps to run
    pub num_steps: usize,
    /// Stop if loss exceeds best_loss * diverge_threshold
    pub diverge_threshold: f64,
    /// Exponential moving average smoothing beta
    pub smooth_beta: f64,
}

impl Default for LrFinderConfig {
    fn default() -> Self {
        Self {
            lr_min: 1e-7,
            lr_max: 10.0,
            num_steps: 100,
            diverge_threshold: 5.0,
            smooth_beta: 0.98,
        }
    }
}

/// A single data point recorded during LR finding
#[derive(Debug, Clone)]
pub struct LrFinderPoint {
    /// Step index (0-based)
    pub step: usize,
    /// Learning rate at this step
    pub lr: f64,
    /// Raw loss at this step
    pub loss: f64,
    /// EMA-smoothed loss at this step
    pub smoothed_loss: f64,
}

/// Reason the LR finder stopped early
#[derive(Debug, Clone, PartialEq)]
pub enum LrStopReason {
    /// Loss exceeded best_loss * diverge_threshold
    Diverged,
    /// Reached the maximum configured number of steps
    MaxStepsReached,
}

/// Action returned by `LrFinder::record_loss`
#[derive(Debug, Clone, PartialEq)]
pub enum LrFinderAction {
    /// Continue running more steps
    Continue,
    /// Stop for the given reason
    Stop(LrStopReason),
}

/// Result of the LR finder run
#[derive(Debug, Clone)]
pub struct LrFinderResult {
    /// Full history of (step, lr, loss, smoothed_loss)
    pub history: Vec<LrFinderPoint>,
    /// Suggested LR (one decade below min-loss LR)
    pub suggested_lr: f64,
    /// LR at the point of minimum smoothed loss
    pub min_loss_lr: f64,
    /// Step at which the run diverged, if applicable
    pub diverged_at_step: Option<usize>,
}

impl LrFinderResult {
    /// The LR at the point of steepest loss descent (maximum negative slope in smoothed loss)
    pub fn steepest_descent_lr(&self) -> f64 {
        if self.history.len() < 2 {
            return self.suggested_lr;
        }
        let mut steepest_slope = 0.0_f64;
        let mut steepest_lr = self.history[0].lr;
        for window in self.history.windows(2) {
            let slope = window[1].smoothed_loss - window[0].smoothed_loss;
            if slope < steepest_slope {
                steepest_slope = slope;
                steepest_lr = window[0].lr;
            }
        }
        steepest_lr
    }

    /// The suggested LR (one decade below min-loss LR)
    pub fn suggested_lr(&self) -> f64 {
        self.suggested_lr
    }

    /// Export history as CSV string
    pub fn to_csv(&self) -> String {
        let mut out = String::from("step,lr,loss,smoothed_loss\n");
        for pt in &self.history {
            out.push_str(&format!(
                "{},{:.10e},{:.10e},{:.10e}\n",
                pt.step, pt.lr, pt.loss, pt.smoothed_loss
            ));
        }
        out
    }

    /// Export history as a JSON string
    pub fn to_json(&self) -> Result<String> {
        // Hand-craft JSON to avoid needing serde here (history can be large)
        let mut out = String::from("{\n  \"history\": [\n");
        for (i, pt) in self.history.iter().enumerate() {
            let comma = if i + 1 < self.history.len() { "," } else { "" };
            out.push_str(&format!(
                "    {{\"step\":{},\"lr\":{:.10e},\"loss\":{:.10e},\"smoothed_loss\":{:.10e}}}{}",
                pt.step, pt.lr, pt.loss, pt.smoothed_loss, comma
            ));
            out.push('\n');
        }
        out.push_str("  ],\n");
        out.push_str(&format!(
            "  \"suggested_lr\":{:.10e},\n  \"min_loss_lr\":{:.10e},\n",
            self.suggested_lr, self.min_loss_lr
        ));
        match self.diverged_at_step {
            Some(s) => out.push_str(&format!("  \"diverged_at_step\":{}\n", s)),
            None => out.push_str("  \"diverged_at_step\":null\n"),
        }
        out.push('}');
        Ok(out)
    }

    /// Render an ASCII plot of log(lr) vs smoothed_loss
    ///
    /// `width` is the number of columns, `height` is the number of rows.
    pub fn to_ascii_plot(&self, width: usize, height: usize) -> String {
        if self.history.is_empty() || width < 4 || height < 4 {
            return String::from("[empty plot]");
        }
        let log_lrs: Vec<f64> = self.history.iter().map(|p| p.lr.ln()).collect();
        let losses: Vec<f64> = self.history.iter().map(|p| p.smoothed_loss).collect();

        let x_min = log_lrs.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = log_lrs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let y_min = losses.iter().cloned().fold(f64::INFINITY, f64::min);
        let y_max = losses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let x_range = if (x_max - x_min).abs() < f64::EPSILON { 1.0 } else { x_max - x_min };
        let y_range = if (y_max - y_min).abs() < f64::EPSILON { 1.0 } else { y_max - y_min };

        let mut grid = vec![vec![' '; width]; height];

        for (x_val, y_val) in log_lrs.iter().zip(losses.iter()) {
            let col = (((x_val - x_min) / x_range) * (width as f64 - 1.0)) as usize;
            let row_f = ((y_val - y_min) / y_range) * (height as f64 - 1.0);
            // Invert y-axis so higher loss is at the top
            let row = (height - 1).saturating_sub(row_f as usize);
            let col = col.min(width - 1);
            let row = row.min(height - 1);
            grid[row][col] = '*';
        }

        let mut out = String::new();
        out.push_str(&format!(
            "  Smoothed Loss vs log(LR)  [min_lr={:.2e}, max_lr={:.2e}]\n",
            self.history.first().map(|p| p.lr).unwrap_or(0.0),
            self.history.last().map(|p| p.lr).unwrap_or(0.0),
        ));
        let border: String = "-".repeat(width + 2);
        out.push_str(&format!("+{}+\n", border));
        for row in &grid {
            let line: String = row.iter().collect();
            out.push_str(&format!("|  {}  |\n", line));
        }
        out.push_str(&format!("+{}+\n", border));
        out
    }
}

/// LR Finder state machine
///
/// Usage pattern:
/// ```text
/// let mut finder = LrFinder::new(config)?;
/// while !finder.is_done() {
///     let lr = finder.current_lr();
///     // run one training step with this LR, obtain loss
///     let action = finder.record_loss(loss);
///     if action == LrFinderAction::Stop(_) { break; }
/// }
/// let result = finder.result();
/// ```
pub struct LrFinder {
    config: LrFinderConfig,
    history: Vec<LrFinderPoint>,
    current_step: usize,
    current_lr: f64,
    lr_multiplier: f64,
    best_loss: f64,
    smoothed_loss: f64,
    avg_loss: f64,
    done: bool,
    diverged_at_step: Option<usize>,
}

impl LrFinder {
    /// Create a new LR finder from the given config.
    ///
    /// Returns an error if the config is invalid (e.g., `lr_min >= lr_max`, `num_steps == 0`).
    pub fn new(config: LrFinderConfig) -> Result<Self> {
        if config.lr_min <= 0.0 {
            return Err(TrustformersError::config_error(
                "lr_min must be positive",
                "LrFinder::new",
            ));
        }
        if config.lr_max <= config.lr_min {
            return Err(TrustformersError::config_error(
                "lr_max must be greater than lr_min",
                "LrFinder::new",
            ));
        }
        if config.num_steps == 0 {
            return Err(TrustformersError::config_error(
                "num_steps must be > 0",
                "LrFinder::new",
            ));
        }
        if config.diverge_threshold <= 1.0 {
            return Err(TrustformersError::config_error(
                "diverge_threshold must be > 1.0",
                "LrFinder::new",
            ));
        }
        if !(0.0..1.0).contains(&config.smooth_beta) {
            return Err(TrustformersError::config_error(
                "smooth_beta must be in [0, 1)",
                "LrFinder::new",
            ));
        }

        // lr_multiplier = (lr_max / lr_min) ^ (1 / (num_steps - 1))
        // Handle num_steps == 1 as a degenerate case (single measurement at lr_min)
        let lr_multiplier = if config.num_steps > 1 {
            (config.lr_max / config.lr_min).powf(1.0 / (config.num_steps as f64 - 1.0))
        } else {
            1.0
        };

        Ok(Self {
            current_lr: config.lr_min,
            lr_multiplier,
            config,
            history: Vec::new(),
            current_step: 0,
            best_loss: f64::INFINITY,
            smoothed_loss: 0.0,
            avg_loss: 0.0,
            done: false,
            diverged_at_step: None,
        })
    }

    /// Current learning rate to use for this step.
    pub fn current_lr(&self) -> f64 {
        self.current_lr
    }

    /// Current step number (0-indexed; incremented after each `record_loss` call).
    pub fn step(&self) -> usize {
        self.current_step
    }

    /// Whether the finder has finished (diverged or max steps reached).
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// Record the loss obtained at the current LR.
    ///
    /// Returns `LrFinderAction::Continue` if more steps should be taken, or
    /// `LrFinderAction::Stop(reason)` if the run should terminate.
    pub fn record_loss(&mut self, loss: f64) -> LrFinderAction {
        if self.done {
            return LrFinderAction::Stop(LrStopReason::MaxStepsReached);
        }

        let step = self.current_step;
        let lr = self.current_lr;

        // Compute EMA-smoothed loss (bias-corrected)
        self.avg_loss =
            self.config.smooth_beta * self.avg_loss + (1.0 - self.config.smooth_beta) * loss;
        let bias_correction = 1.0 - self.config.smooth_beta.powi(step as i32 + 1);
        let smoothed = self.avg_loss / bias_correction;
        self.smoothed_loss = smoothed;

        // Track best smoothed loss
        if smoothed < self.best_loss {
            self.best_loss = smoothed;
        }

        self.history.push(LrFinderPoint {
            step,
            lr,
            loss,
            smoothed_loss: smoothed,
        });

        self.current_step += 1;
        self.current_lr *= self.lr_multiplier;

        // Check divergence
        if smoothed > self.best_loss * self.config.diverge_threshold {
            self.done = true;
            self.diverged_at_step = Some(step);
            return LrFinderAction::Stop(LrStopReason::Diverged);
        }

        // Check max steps
        if self.current_step >= self.config.num_steps {
            self.done = true;
            return LrFinderAction::Stop(LrStopReason::MaxStepsReached);
        }

        LrFinderAction::Continue
    }

    /// Compute and return the final `LrFinderResult`.
    pub fn result(self) -> LrFinderResult {
        let (suggested_lr, min_loss_lr) = compute_suggested_lr(&self.history);
        LrFinderResult {
            history: self.history,
            suggested_lr,
            min_loss_lr,
            diverged_at_step: self.diverged_at_step,
        }
    }
}

/// Compute the suggested LR and the min-loss LR from a completed history.
fn compute_suggested_lr(history: &[LrFinderPoint]) -> (f64, f64) {
    if history.is_empty() {
        return (1e-3, 1e-3);
    }

    // Find min-loss LR
    let min_pt = history.iter().min_by(|a, b| {
        a.smoothed_loss
            .partial_cmp(&b.smoothed_loss)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let min_loss_lr = min_pt.map(|p| p.lr).unwrap_or(history[0].lr);

    // Suggested LR = one decade below min-loss LR (as in fastai)
    let suggested_lr = min_loss_lr / 10.0;

    (suggested_lr, min_loss_lr)
}

/// Find the suggested optimal LR from a list of `(lr, loss)` data points.
///
/// Strategy: locate the point where the smoothed loss rate-of-decrease is maximum
/// (steepest descent), then return the LR at that point.
pub fn find_optimal_lr(history: &[(f64, f64)]) -> f64 {
    if history.is_empty() {
        return 1e-3;
    }
    if history.len() == 1 {
        return history[0].0;
    }

    // Compute EMA smoothing
    let beta = 0.98_f64;
    let mut avg = 0.0_f64;
    let mut smoothed: Vec<f64> = Vec::with_capacity(history.len());
    for (i, &(_, loss)) in history.iter().enumerate() {
        avg = beta * avg + (1.0 - beta) * loss;
        let bias_correction = 1.0 - beta.powi(i as i32 + 1);
        smoothed.push(avg / bias_correction);
    }

    // Find steepest descent (maximum negative slope over consecutive smoothed losses)
    let mut steepest_slope = 0.0_f64;
    let mut steepest_idx = 0_usize;
    for i in 1..smoothed.len() {
        let slope = smoothed[i] - smoothed[i - 1];
        if slope < steepest_slope {
            steepest_slope = slope;
            steepest_idx = i - 1;
        }
    }

    history[steepest_idx].0
}

// ─────────────────────────────────────────── tests ──────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lr_finder_config_default() {
        let cfg = LrFinderConfig::default();
        assert!(cfg.lr_min > 0.0);
        assert!(cfg.lr_max > cfg.lr_min);
        assert!(cfg.num_steps > 0);
        assert!(cfg.diverge_threshold > 1.0);
        assert!((0.0..1.0).contains(&cfg.smooth_beta));
    }

    #[test]
    fn test_lr_finder_multiplier_computation() {
        let cfg = LrFinderConfig {
            lr_min: 1e-7,
            lr_max: 1e-1,
            num_steps: 7,
            ..Default::default()
        };
        let finder = LrFinder::new(cfg.clone()).expect("should construct");
        // After num_steps-1 multiplications, lr_min * mult^(n-1) == lr_max
        let final_lr = cfg.lr_min * finder.lr_multiplier.powi((cfg.num_steps as i32) - 1);
        let rel_err = ((final_lr - cfg.lr_max) / cfg.lr_max).abs();
        assert!(
            rel_err < 1e-9,
            "Expected final lr ≈ lr_max, got rel_err={}",
            rel_err
        );
    }

    #[test]
    fn test_lr_finder_records_history() {
        let cfg = LrFinderConfig {
            num_steps: 5,
            diverge_threshold: 100.0,
            ..Default::default()
        };
        let mut finder = LrFinder::new(cfg).expect("construct");
        for _ in 0..5 {
            finder.record_loss(1.0);
        }
        assert_eq!(finder.history.len(), 5);
    }

    #[test]
    fn test_lr_finder_detects_divergence() {
        let cfg = LrFinderConfig {
            num_steps: 20,
            diverge_threshold: 3.0,
            smooth_beta: 0.0, // no smoothing so raw == smoothed
            ..Default::default()
        };
        let mut finder = LrFinder::new(cfg).expect("construct");
        // First loss seeds best_loss
        let action = finder.record_loss(1.0);
        assert_eq!(action, LrFinderAction::Continue);
        // Now supply a loss that clearly triggers divergence
        let action = finder.record_loss(10.0);
        assert_eq!(action, LrFinderAction::Stop(LrStopReason::Diverged));
        assert!(finder.is_done());
    }

    #[test]
    fn test_lr_finder_max_steps() {
        let cfg = LrFinderConfig {
            num_steps: 3,
            diverge_threshold: 1000.0,
            ..Default::default()
        };
        let mut finder = LrFinder::new(cfg).expect("construct");
        finder.record_loss(1.0);
        finder.record_loss(1.0);
        let action = finder.record_loss(1.0);
        assert_eq!(action, LrFinderAction::Stop(LrStopReason::MaxStepsReached));
        assert!(finder.is_done());
    }

    #[test]
    fn test_lr_finder_result_steepest_descent() {
        // Build a hand-crafted result with an unambiguous steepest descent.
        // Losses: 4.0, 3.5, 2.0, 1.8, 3.0
        //         Δ = -0.5, -1.5, -0.2, +1.2
        // Steepest drop is between index 1 (lr=1e-4) and index 2 (lr=1e-3): slope -1.5
        let mut history = Vec::new();
        let lrs = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1];
        let losses = [4.0, 3.5, 2.0, 1.8, 3.0];
        for (i, (&lr, &loss)) in lrs.iter().zip(losses.iter()).enumerate() {
            history.push(LrFinderPoint {
                step: i,
                lr,
                loss,
                smoothed_loss: loss,
            });
        }
        let result = LrFinderResult {
            history,
            suggested_lr: 1e-4,
            min_loss_lr: 1e-3,
            diverged_at_step: None,
        };
        let sd_lr = result.steepest_descent_lr();
        // Steepest descent starts at lr=1e-4 (slope=-1.5 going to step 2)
        assert_eq!(sd_lr, 1e-4);
    }

    #[test]
    fn test_lr_finder_to_csv() {
        let cfg = LrFinderConfig {
            num_steps: 3,
            diverge_threshold: 1000.0,
            smooth_beta: 0.0,
            ..Default::default()
        };
        let mut finder = LrFinder::new(cfg).expect("construct");
        finder.record_loss(1.0);
        finder.record_loss(0.8);
        finder.record_loss(0.6);
        let result = finder.result();
        let csv = result.to_csv();
        assert!(csv.starts_with("step,lr,loss,smoothed_loss\n"));
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 4); // header + 3 data rows
    }

    #[test]
    fn test_lr_finder_to_json() {
        let cfg = LrFinderConfig {
            num_steps: 2,
            diverge_threshold: 1000.0,
            smooth_beta: 0.0,
            ..Default::default()
        };
        let mut finder = LrFinder::new(cfg).expect("construct");
        finder.record_loss(1.0);
        finder.record_loss(0.5);
        let result = finder.result();
        let json = result.to_json().expect("to_json");
        assert!(json.contains("\"history\""));
        assert!(json.contains("\"suggested_lr\""));
        assert!(json.contains("\"min_loss_lr\""));
    }

    #[test]
    fn test_lr_finder_to_ascii_plot() {
        let cfg = LrFinderConfig {
            num_steps: 10,
            diverge_threshold: 1000.0,
            ..Default::default()
        };
        let mut finder = LrFinder::new(cfg).expect("construct");
        for i in 0..10 {
            finder.record_loss(10.0 - i as f64);
        }
        let result = finder.result();
        let plot = result.to_ascii_plot(40, 10);
        assert!(plot.contains('*'));
    }

    #[test]
    fn test_find_optimal_lr_known_case() {
        // Manually built (lr, loss) pairs with a clear minimum descent region
        let history: Vec<(f64, f64)> = vec![
            (1e-6, 4.0),
            (1e-5, 3.5),
            (1e-4, 2.0),
            (1e-3, 1.0), // steepest descent toward here
            (1e-2, 1.2),
            (1e-1, 2.5),
        ];
        let optimal = find_optimal_lr(&history);
        // The steepest descent should be before 1e-3
        assert!(
            optimal < 1e-2,
            "Expected optimal lr < 1e-2, got {}",
            optimal
        );
    }

    #[test]
    fn test_lr_finder_smooth_loss() {
        let cfg = LrFinderConfig {
            num_steps: 10,
            diverge_threshold: 1000.0,
            smooth_beta: 0.9,
            ..Default::default()
        };
        let mut finder = LrFinder::new(cfg).expect("construct");
        finder.record_loss(1.0);
        let first_smoothed = finder.history[0].smoothed_loss;
        // With beta=0.9 and step=0, smoothed should be close to the raw loss
        // bias_correction = 1 - 0.9^1 = 0.1; avg = 0.1 * 1.0 = 0.1; smoothed = 0.1/0.1 = 1.0
        assert!((first_smoothed - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lr_finder_invalid_config_lr_min_zero() {
        let cfg = LrFinderConfig {
            lr_min: 0.0,
            ..Default::default()
        };
        assert!(LrFinder::new(cfg).is_err());
    }

    #[test]
    fn test_lr_finder_invalid_config_lr_max_lte_min() {
        let cfg = LrFinderConfig {
            lr_min: 0.01,
            lr_max: 0.001,
            ..Default::default()
        };
        assert!(LrFinder::new(cfg).is_err());
    }

    #[test]
    fn test_lr_finder_single_step() {
        let cfg = LrFinderConfig {
            num_steps: 1,
            diverge_threshold: 2.0,
            smooth_beta: 0.0,
            ..Default::default()
        };
        let mut finder = LrFinder::new(cfg).expect("construct");
        let action = finder.record_loss(0.5);
        assert_eq!(action, LrFinderAction::Stop(LrStopReason::MaxStepsReached));
        let result = finder.result();
        assert_eq!(result.history.len(), 1);
    }

    #[test]
    fn test_lr_finder_result_min_loss_lr() {
        let cfg = LrFinderConfig {
            num_steps: 5,
            diverge_threshold: 1000.0,
            smooth_beta: 0.0,
            ..Default::default()
        };
        let mut finder = LrFinder::new(cfg).expect("construct");
        // Feed losses: high, lower, lowest, higher, highest
        finder.record_loss(3.0);
        finder.record_loss(2.0);
        finder.record_loss(1.0);
        finder.record_loss(2.5);
        finder.record_loss(3.5);
        let result = finder.result();
        // min_loss_lr should be at step 2
        assert_eq!(result.min_loss_lr, result.history[2].lr);
        // suggested_lr is min_loss_lr / 10
        let expected_suggested = result.min_loss_lr / 10.0;
        assert!((result.suggested_lr - expected_suggested).abs() < 1e-12);
    }
}
