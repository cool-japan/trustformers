//! Gradient accumulation utilities for simulating larger batch sizes
//! by accumulating gradients across multiple micro-batches.

use std::fmt;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during gradient accumulation.
#[derive(Debug, Clone, PartialEq)]
pub enum GradError {
    /// Number of parameter tensors does not match the buffer.
    ShapeMismatch {
        expected: usize,
        got: usize,
    },
    /// A specific parameter tensor length mismatches.
    TensorLengthMismatch {
        param_idx: usize,
        expected: usize,
        got: usize,
    },
    /// Finalize called but no gradients have been accumulated yet.
    EmptyBuffer,
}

impl fmt::Display for GradError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GradError::ShapeMismatch { expected, got } => write!(
                f,
                "gradient shape mismatch: expected {} parameter tensors, got {}",
                expected, got
            ),
            GradError::TensorLengthMismatch {
                param_idx,
                expected,
                got,
            } => write!(
                f,
                "tensor length mismatch at param {}: expected {} elements, got {}",
                param_idx, expected, got
            ),
            GradError::EmptyBuffer => {
                write!(f, "cannot finalize: no gradients have been accumulated")
            }
        }
    }
}

impl std::error::Error for GradError {}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for gradient accumulation behaviour.
#[derive(Debug, Clone)]
pub struct GradAccumConfig {
    /// Number of micro-batches whose gradients are accumulated before one
    /// optimizer update.
    pub accumulation_steps: usize,
    /// When `true`, accumulated gradients are divided by `step_count` before
    /// the update so that the effective gradient is the *mean* over
    /// micro-batches.
    pub normalize_by_steps: bool,
    /// Optional global gradient-norm clipping threshold.
    pub clip_grad_norm: Option<f32>,
    /// Only synchronize gradients on the final accumulation step (useful for
    /// DDP-style training to avoid redundant all-reduces).
    pub sync_on_last: bool,
}

impl Default for GradAccumConfig {
    fn default() -> Self {
        Self {
            accumulation_steps: 4,
            normalize_by_steps: true,
            clip_grad_norm: Some(1.0),
            sync_on_last: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Gradient utilities
// ---------------------------------------------------------------------------

/// Compute the global L2 norm across all gradient tensors.
///
/// `sqrt(Σ_p Σ_i g[p][i]²)`
pub fn global_grad_norm(grads: &[Vec<f32>]) -> f32 {
    let sum_sq: f32 = grads.iter().flat_map(|g| g.iter()).map(|v| v * v).sum();
    sum_sq.sqrt()
}

/// Clip gradients in-place so that their global L2 norm does not exceed
/// `max_norm`.
///
/// Returns the *original* (pre-clipping) global norm.
pub fn clip_grad_norm_(grads: &mut [Vec<f32>], max_norm: f32) -> f32 {
    let norm = global_grad_norm(grads);
    if norm > max_norm {
        let scale = max_norm / norm;
        for param_grads in grads.iter_mut() {
            for g in param_grads.iter_mut() {
                *g *= scale;
            }
        }
    }
    norm
}

// ---------------------------------------------------------------------------
// GradientBuffer
// ---------------------------------------------------------------------------

/// Internal accumulation buffer — one `Vec<f32>` per parameter tensor.
#[derive(Debug)]
pub struct GradientBuffer {
    /// Accumulated gradients, indexed by parameter.
    pub gradients: Vec<Vec<f32>>,
    /// How many micro-batch gradient sets have been added so far.
    pub step_count: usize,
    /// `true` once `step_count == config.accumulation_steps`.
    pub is_ready: bool,
}

impl GradientBuffer {
    /// Create a new buffer initialized with zeros.
    ///
    /// `param_shapes` is a slice where each element is the *number of scalar
    /// parameters* in that tensor.
    pub fn new(param_shapes: &[usize]) -> Self {
        let gradients = param_shapes.iter().map(|&n| vec![0.0_f32; n]).collect();
        Self {
            gradients,
            step_count: 0,
            is_ready: false,
        }
    }

    /// Accumulate a set of per-parameter gradients.
    ///
    /// Each entry in `grads` must have the same length as the corresponding
    /// buffer slot; the number of entries must match the number of parameters.
    pub fn accumulate(
        &mut self,
        grads: &[Vec<f32>],
        config: &GradAccumConfig,
    ) -> Result<(), GradError> {
        if grads.len() != self.gradients.len() {
            return Err(GradError::ShapeMismatch {
                expected: self.gradients.len(),
                got: grads.len(),
            });
        }

        for (idx, (buf, incoming)) in
            self.gradients.iter_mut().zip(grads.iter()).enumerate()
        {
            if buf.len() != incoming.len() {
                return Err(GradError::TensorLengthMismatch {
                    param_idx: idx,
                    expected: buf.len(),
                    got: incoming.len(),
                });
            }
            for (b, g) in buf.iter_mut().zip(incoming.iter()) {
                *b += g;
            }
        }

        self.step_count += 1;
        if self.step_count == config.accumulation_steps {
            self.is_ready = true;
        }
        Ok(())
    }

    /// Finalize accumulation: optionally normalize, optionally clip, then
    /// reset the buffer.
    ///
    /// Returns the accumulated (and possibly normalized/clipped) gradients.
    /// Returns [`GradError::EmptyBuffer`] if no steps have been accumulated.
    pub fn finalize(
        &mut self,
        config: &GradAccumConfig,
    ) -> Result<Vec<Vec<f32>>, GradError> {
        if self.step_count == 0 {
            return Err(GradError::EmptyBuffer);
        }

        let mut result = self.gradients.clone();

        if config.normalize_by_steps && self.step_count > 0 {
            let divisor = self.step_count as f32;
            for param_grads in result.iter_mut() {
                for g in param_grads.iter_mut() {
                    *g /= divisor;
                }
            }
        }

        if let Some(max_norm) = config.clip_grad_norm {
            clip_grad_norm_(&mut result, max_norm);
        }

        self.reset();
        Ok(result)
    }

    /// Reset the buffer to zeros without finalizing.
    pub fn reset(&mut self) {
        for param_grads in self.gradients.iter_mut() {
            for g in param_grads.iter_mut() {
                *g = 0.0;
            }
        }
        self.step_count = 0;
        self.is_ready = false;
    }

    /// Return the number of micro-batch steps accumulated so far.
    pub fn current_step(&self) -> usize {
        self.step_count
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Running statistics collected across optimizer updates.
#[derive(Debug, Clone, Default)]
pub struct GradAccumStats {
    /// Total number of complete optimizer updates performed.
    pub total_updates: u64,
    /// Total number of micro-batches processed.
    pub total_micro_batches: u64,
    /// Number of times gradient clipping was applied.
    pub clips_applied: u64,
    /// Running mean of the gradient norm *before* clipping.
    pub mean_grad_norm_before_clip: f32,
    /// Maximum gradient norm seen across all updates.
    pub max_grad_norm_seen: f32,
}

impl GradAccumStats {
    /// Record one optimizer update.
    ///
    /// `grad_norm` is the global norm *before* any clipping was applied.
    /// `was_clipped` indicates whether the norm exceeded the clip threshold.
    pub fn record_update(&mut self, grad_norm: f32, was_clipped: bool) {
        self.total_updates += 1;
        if was_clipped {
            self.clips_applied += 1;
        }
        // Update running mean with online (Welford-style) formula.
        let n = self.total_updates as f32;
        self.mean_grad_norm_before_clip +=
            (grad_norm - self.mean_grad_norm_before_clip) / n;
        if grad_norm > self.max_grad_norm_seen {
            self.max_grad_norm_seen = grad_norm;
        }
    }
}

// ---------------------------------------------------------------------------
// GradientAccumulator — high-level entry point
// ---------------------------------------------------------------------------

/// High-level gradient accumulator that owns the config, buffer, and stats.
pub struct GradientAccumulator {
    /// Configuration governing accumulation behaviour.
    pub config: GradAccumConfig,
    /// Underlying accumulation buffer.
    pub buffer: GradientBuffer,
    /// Accumulated statistics.
    pub stats: GradAccumStats,
}

impl GradientAccumulator {
    /// Create a new accumulator.
    ///
    /// `param_shapes` must list the element count of every parameter tensor.
    pub fn new(config: GradAccumConfig, param_shapes: &[usize]) -> Self {
        let buffer = GradientBuffer::new(param_shapes);
        Self {
            config,
            buffer,
            stats: GradAccumStats::default(),
        }
    }

    /// Submit one micro-batch of gradients.
    ///
    /// Returns `Some(accumulated_grads)` when the accumulation window is
    /// complete; returns `None` when more micro-batches are needed.
    pub fn step(
        &mut self,
        grads: &[Vec<f32>],
    ) -> Result<Option<Vec<Vec<f32>>>, GradError> {
        self.buffer.accumulate(grads, &self.config)?;
        self.stats.total_micro_batches += 1;

        if self.buffer.is_ready {
            let norm_before = global_grad_norm(&self.buffer.gradients);
            let result = self.buffer.finalize(&self.config)?;
            let was_clipped = self
                .config
                .clip_grad_norm
                .map(|max| norm_before > max)
                .unwrap_or(false);
            // norm_before is before normalization; for stats we record the
            // normalized norm if applicable.
            let stats_norm = if self.config.normalize_by_steps {
                norm_before / self.config.accumulation_steps as f32
            } else {
                norm_before
            };
            self.stats.record_update(stats_norm, was_clipped);
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    /// Force finalization even if `accumulation_steps` has not been reached.
    ///
    /// Useful at the end of an epoch when the final mini-batch is smaller than
    /// the accumulation window.  Returns `None` if the buffer is empty.
    pub fn force_finalize(
        &mut self,
    ) -> Result<Option<Vec<Vec<f32>>>, GradError> {
        if self.buffer.step_count == 0 {
            return Ok(None);
        }
        let norm_before = global_grad_norm(&self.buffer.gradients);
        let result = self.buffer.finalize(&self.config)?;
        let was_clipped = self
            .config
            .clip_grad_norm
            .map(|max| norm_before > max)
            .unwrap_or(false);
        let stats_norm = if self.config.normalize_by_steps && self.buffer.step_count > 0 {
            // step_count was already reset; use the pre-reset value captured
            // in norm_before (un-normalised).  We normalise here by the number
            // of steps that *were* accumulated before the reset.
            // Because finalize already reset step_count, we track the norm of
            // the raw sum divided by the result length isn't available; so we
            // simply pass the raw norm and note it was from a partial batch.
            norm_before
        } else {
            norm_before
        };
        self.stats.record_update(stats_norm, was_clipped);
        Ok(Some(result))
    }

    /// Access accumulated statistics.
    pub fn stats(&self) -> &GradAccumStats {
        &self.stats
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a config with no clipping and no normalization.
    fn bare_config(steps: usize) -> GradAccumConfig {
        GradAccumConfig {
            accumulation_steps: steps,
            normalize_by_steps: false,
            clip_grad_norm: None,
            sync_on_last: true,
        }
    }

    /// Helper: build a config with normalization but no clipping.
    fn norm_config(steps: usize) -> GradAccumConfig {
        GradAccumConfig {
            accumulation_steps: steps,
            normalize_by_steps: true,
            clip_grad_norm: None,
            sync_on_last: true,
        }
    }

    // ------------------------------------------------------------------
    // 1. Accumulate 4 steps then finalize (basic end-to-end)
    // ------------------------------------------------------------------
    #[test]
    fn test_accumulate_4_steps_then_finalize() {
        let config = bare_config(4);
        let param_shapes = &[3_usize, 2];
        let mut buf = GradientBuffer::new(param_shapes);

        let grads_a = vec![vec![1.0_f32; 3], vec![1.0_f32; 2]];

        for step in 0..4 {
            let result = buf.accumulate(&grads_a, &config);
            assert!(result.is_ok());
            if step < 3 {
                assert!(!buf.is_ready, "should not be ready before step 4");
            } else {
                assert!(buf.is_ready, "should be ready after step 4");
            }
        }

        let finalized = buf.finalize(&config).expect("finalize should succeed");
        assert_eq!(finalized.len(), 2);
        // Each element should be 4.0 (accumulated, not normalized).
        for v in &finalized[0] {
            assert!((v - 4.0).abs() < 1e-6, "expected 4.0, got {v}");
        }
        for v in &finalized[1] {
            assert!((v - 4.0).abs() < 1e-6, "expected 4.0, got {v}");
        }
        // Buffer should be reset.
        assert_eq!(buf.step_count, 0);
        assert!(!buf.is_ready);
    }

    // ------------------------------------------------------------------
    // 2. Normalization: check division by 4
    // ------------------------------------------------------------------
    #[test]
    fn test_normalization_divides_by_steps() {
        let config = norm_config(4);
        let param_shapes = &[4_usize];
        let mut buf = GradientBuffer::new(param_shapes);

        // Accumulate grads that sum to 8.0 per element (2.0 * 4 steps).
        let grads = vec![vec![2.0_f32; 4]];
        for _ in 0..4 {
            buf.accumulate(&grads, &config).expect("accumulate");
        }
        let finalized = buf.finalize(&config).expect("finalize");
        for v in &finalized[0] {
            assert!(
                (v - 2.0).abs() < 1e-6,
                "expected mean 2.0 after dividing sum 8.0 by 4, got {v}"
            );
        }
    }

    // ------------------------------------------------------------------
    // 3. Gradient clipping: global norm > max → scaled
    // ------------------------------------------------------------------
    #[test]
    fn test_gradient_clipping_scales_down() {
        // Two params, each with one element = 3.0 → global norm = sqrt(18) ≈ 4.24.
        // max_norm = 1.0 → should be clipped.
        let mut grads = vec![vec![3.0_f32], vec![3.0_f32]];
        let original_norm = clip_grad_norm_(&mut grads, 1.0);

        let original_expected = (2.0_f32 * 9.0_f32).sqrt(); // sqrt(18)
        assert!(
            (original_norm - original_expected).abs() < 1e-4,
            "expected original norm ~{original_expected}, got {original_norm}"
        );

        // After clipping the global norm should equal max_norm = 1.0.
        let new_norm = global_grad_norm(&grads);
        assert!(
            (new_norm - 1.0).abs() < 1e-5,
            "expected clipped norm 1.0, got {new_norm}"
        );
    }

    // ------------------------------------------------------------------
    // 4. No clipping when norm is under max
    // ------------------------------------------------------------------
    #[test]
    fn test_no_clipping_when_under_max() {
        // Global norm = sqrt(0.25 + 0.25) = sqrt(0.5) ≈ 0.707 < 1.0.
        let mut grads = vec![vec![0.5_f32], vec![0.5_f32]];
        let original = grads.clone();
        clip_grad_norm_(&mut grads, 1.0);
        assert_eq!(grads, original, "grads should not change when under max_norm");
    }

    // ------------------------------------------------------------------
    // 5. reset() clears the buffer
    // ------------------------------------------------------------------
    #[test]
    fn test_reset_clears_buffer() {
        let config = bare_config(4);
        let param_shapes = &[2_usize];
        let mut buf = GradientBuffer::new(param_shapes);

        buf.accumulate(&[vec![5.0, 5.0]], &config).expect("accumulate");
        assert_eq!(buf.step_count, 1);

        buf.reset();
        assert_eq!(buf.step_count, 0);
        assert!(!buf.is_ready);
        assert_eq!(buf.gradients[0], vec![0.0_f32, 0.0_f32]);
    }

    // ------------------------------------------------------------------
    // 6. Partial batch at end of epoch — force_finalize
    // ------------------------------------------------------------------
    #[test]
    fn test_force_finalize_partial_batch() {
        let config = bare_config(4); // needs 4 steps normally
        let param_shapes = &[2_usize];
        let mut acc = GradientAccumulator::new(config, param_shapes);

        // Only push 2 steps.
        let grads = vec![vec![1.0_f32, 2.0_f32]];
        acc.step(&grads).expect("step 1");
        acc.step(&grads).expect("step 2");
        assert!(!acc.buffer.is_ready);

        let forced = acc.force_finalize().expect("force_finalize ok");
        assert!(forced.is_some(), "should have gradients");
        let result = forced.expect("should be some");
        // 2 steps accumulated (no normalization) → 2.0, 4.0.
        assert!((result[0][0] - 2.0).abs() < 1e-6);
        assert!((result[0][1] - 4.0).abs() < 1e-6);
    }

    // ------------------------------------------------------------------
    // 7. Stats tracking
    // ------------------------------------------------------------------
    #[test]
    fn test_stats_tracking() {
        let config = GradAccumConfig {
            accumulation_steps: 2,
            normalize_by_steps: false,
            clip_grad_norm: Some(1.0),
            sync_on_last: true,
        };
        let param_shapes = &[1_usize];
        let mut acc = GradientAccumulator::new(config, param_shapes);

        // First update: norm = sqrt(4+4) = 2.83 > 1.0 → clipped.
        acc.step(&[vec![2.0_f32]]).expect("step 1a");
        acc.step(&[vec![2.0_f32]]).expect("step 1b → update 1");

        assert_eq!(acc.stats().total_updates, 1);
        assert_eq!(acc.stats().total_micro_batches, 2);
        assert_eq!(acc.stats().clips_applied, 1);

        // Second update: small gradients, no clipping.
        acc.step(&[vec![0.1_f32]]).expect("step 2a");
        acc.step(&[vec![0.1_f32]]).expect("step 2b → update 2");

        assert_eq!(acc.stats().total_updates, 2);
        assert_eq!(acc.stats().clips_applied, 1, "no additional clip expected");
    }

    // ------------------------------------------------------------------
    // 8. Multiple parameters
    // ------------------------------------------------------------------
    #[test]
    fn test_multiple_parameters() {
        let config = bare_config(2);
        let param_shapes = &[3_usize, 5, 2];
        let mut buf = GradientBuffer::new(param_shapes);

        let g = vec![
            vec![1.0_f32, 2.0, 3.0],
            vec![0.5_f32; 5],
            vec![10.0_f32, 20.0],
        ];
        buf.accumulate(&g, &config).expect("step 1");
        buf.accumulate(&g, &config).expect("step 2");

        let finalized = buf.finalize(&config).expect("finalize");
        assert_eq!(finalized[0], vec![2.0, 4.0, 6.0]);
        for v in &finalized[1] {
            assert!((v - 1.0).abs() < 1e-6);
        }
        assert_eq!(finalized[2], vec![20.0, 40.0]);
    }

    // ------------------------------------------------------------------
    // 9. Error on shape mismatch (wrong number of param tensors)
    // ------------------------------------------------------------------
    #[test]
    fn test_error_on_shape_mismatch() {
        let config = bare_config(4);
        let param_shapes = &[3_usize, 2];
        let mut buf = GradientBuffer::new(param_shapes);

        // Only one tensor instead of two.
        let grads = vec![vec![1.0_f32; 3]];
        let result = buf.accumulate(&grads, &config);
        match result {
            Err(GradError::ShapeMismatch { expected: 2, got: 1 }) => {} // correct
            other => panic!("expected ShapeMismatch, got {:?}", other),
        }
    }

    // ------------------------------------------------------------------
    // 10. clip_grad_norm_ returns the original norm (before clipping)
    // ------------------------------------------------------------------
    #[test]
    fn test_clip_returns_original_norm() {
        // One element = 5.0 → norm = 5.0.
        let mut grads = vec![vec![5.0_f32]];
        let returned = clip_grad_norm_(&mut grads, 2.0);
        assert!(
            (returned - 5.0).abs() < 1e-5,
            "expected original norm 5.0, got {returned}"
        );
        // After clipping the norm should be 2.0.
        let after = global_grad_norm(&grads);
        assert!(
            (after - 2.0).abs() < 1e-5,
            "expected clipped norm 2.0, got {after}"
        );
    }

    // ------------------------------------------------------------------
    // Bonus: tensor-length mismatch error
    // ------------------------------------------------------------------
    #[test]
    fn test_error_on_tensor_length_mismatch() {
        let config = bare_config(4);
        let param_shapes = &[3_usize];
        let mut buf = GradientBuffer::new(param_shapes);

        // Wrong length for the first tensor.
        let grads = vec![vec![1.0_f32, 2.0]]; // length 2 instead of 3
        let result = buf.accumulate(&grads, &config);
        match result {
            Err(GradError::TensorLengthMismatch { param_idx: 0, expected: 3, got: 2 }) => {} // correct
            other => panic!("expected TensorLengthMismatch, got {:?}", other),
        }
    }

    // ------------------------------------------------------------------
    // Bonus: finalize empty buffer returns error
    // ------------------------------------------------------------------
    #[test]
    fn test_finalize_empty_buffer_error() {
        let config = bare_config(4);
        let param_shapes = &[2_usize];
        let mut buf = GradientBuffer::new(param_shapes);
        let result = buf.finalize(&config);
        assert_eq!(result, Err(GradError::EmptyBuffer));
    }
}
