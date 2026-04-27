//! Core curriculum scheduler with multiple ordering strategies.
//!
//! Curriculum learning trains models by progressively exposing them to harder
//! examples. This module implements:
//!
//! - [`CurriculumStrategy`]: Four strategies (length, difficulty, self-paced, competence)
//! - [`CurriculumScheduler`]: Drives strategy execution step-by-step
//! - Utility functions: [`sort_by_length`], [`sort_by_perplexity`], [`anti_curriculum_indices`]

use std::fmt;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Error kind for curriculum operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CurriculumErrorKind {
    InvalidConfig,
    EmptyDataset,
    StepOutOfRange,
}

impl fmt::Display for CurriculumErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig => write!(f, "InvalidConfig"),
            Self::EmptyDataset => write!(f, "EmptyDataset"),
            Self::StepOutOfRange => write!(f, "StepOutOfRange"),
        }
    }
}

/// Error type for curriculum operations.
#[derive(Debug)]
pub struct CurriculumError {
    pub kind: CurriculumErrorKind,
    pub message: String,
}

impl CurriculumError {
    pub fn new(kind: CurriculumErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }

    pub fn invalid_config(message: impl Into<String>) -> Self {
        Self::new(CurriculumErrorKind::InvalidConfig, message)
    }

    pub fn empty_dataset(message: impl Into<String>) -> Self {
        Self::new(CurriculumErrorKind::EmptyDataset, message)
    }

    pub fn step_out_of_range(message: impl Into<String>) -> Self {
        Self::new(CurriculumErrorKind::StepOutOfRange, message)
    }
}

impl fmt::Display for CurriculumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CurriculumError({}: {})", self.kind, self.message)
    }
}

impl std::error::Error for CurriculumError {}

// ---------------------------------------------------------------------------
// Strategy definitions
// ---------------------------------------------------------------------------

/// Strategy for ordering training examples in the curriculum.
#[derive(Debug, Clone)]
pub enum CurriculumStrategy {
    /// Order by increasing sequence length.
    ByLength {
        /// Maximum sequence length at the start of training.
        initial_max_len: usize,
        /// Maximum sequence length at the end of training.
        final_max_len: usize,
        /// Number of steps over which to ramp from initial to final length.
        ramp_steps: usize,
    },
    /// Order by difficulty score (lower = easier).
    ByDifficulty {
        /// Pre-computed difficulty score for each sample.
        difficulty_scores: Vec<f32>,
        /// Fraction of easiest samples accessible at the start (e.g. 0.1 = 10%).
        initial_percentile: f32,
        /// Fraction accessible at the end (e.g. 1.0 = all samples).
        final_percentile: f32,
        /// Number of steps over which to ramp from initial to final percentile.
        ramp_steps: usize,
    },
    /// Self-paced: scheduler includes examples the model is most uncertain about
    /// (uncertainty > threshold), and gradually lowers the threshold.
    SelfPaced {
        /// Initial uncertainty threshold — only high-uncertainty samples are kept.
        uncertainty_threshold: f32,
        /// How much the threshold shrinks per step (multiply by 1 - ramp_rate).
        ramp_rate: f32,
    },
    /// Competence-based: a competence score grows with steps, and the fraction
    /// of the dataset accessible equals the competence score.
    CompetenceBased {
        /// Competence at step 0 (fraction of dataset accessible, 0..1).
        initial_competence: f32,
        /// Rate at which competence grows per step (added each step, clamped to 1).
        growth_rate: f32,
    },
}

// ---------------------------------------------------------------------------
// Curriculum window
// ---------------------------------------------------------------------------

/// Snapshot of the current curriculum state after a scheduler step.
#[derive(Debug, Clone)]
pub struct CurriculumWindow {
    /// Current step index (0-based).
    pub step: usize,
    /// Fraction of difficulty percentile accessible (0..1).
    pub percentile: f32,
    /// Current maximum sequence length (only set for `ByLength` strategy).
    pub max_length: Option<usize>,
    /// Fraction of the total dataset currently accessible.
    pub active_fraction: f32,
}

// ---------------------------------------------------------------------------
// Curriculum scheduler
// ---------------------------------------------------------------------------

/// Drives a curriculum strategy over the course of training.
pub struct CurriculumScheduler {
    strategy: CurriculumStrategy,
    /// Steps completed so far.
    current_step: usize,
    /// Total number of samples in the dataset.
    total_samples: usize,
}

impl CurriculumScheduler {
    /// Create a new curriculum scheduler.
    ///
    /// Returns an error when the configuration is logically invalid (e.g. empty
    /// dataset, initial percentile out of [0,1]).
    pub fn new(
        strategy: CurriculumStrategy,
        total_samples: usize,
    ) -> Result<Self, CurriculumError> {
        if total_samples == 0 {
            return Err(CurriculumError::empty_dataset("total_samples must be > 0"));
        }

        // Validate strategy-specific constraints.
        match &strategy {
            CurriculumStrategy::ByLength { initial_max_len, final_max_len, .. } => {
                if *initial_max_len == 0 {
                    return Err(CurriculumError::invalid_config(
                        "initial_max_len must be > 0",
                    ));
                }
                if *final_max_len < *initial_max_len {
                    return Err(CurriculumError::invalid_config(
                        "final_max_len must be >= initial_max_len",
                    ));
                }
            }
            CurriculumStrategy::ByDifficulty {
                initial_percentile,
                final_percentile,
                difficulty_scores,
                ..
            } => {
                if *initial_percentile < 0.0 || *initial_percentile > 1.0 {
                    return Err(CurriculumError::invalid_config(
                        "initial_percentile must be in [0, 1]",
                    ));
                }
                if *final_percentile < 0.0 || *final_percentile > 1.0 {
                    return Err(CurriculumError::invalid_config(
                        "final_percentile must be in [0, 1]",
                    ));
                }
                if difficulty_scores.is_empty() {
                    return Err(CurriculumError::empty_dataset(
                        "difficulty_scores must not be empty",
                    ));
                }
            }
            CurriculumStrategy::SelfPaced { ramp_rate, .. } => {
                if *ramp_rate < 0.0 || *ramp_rate > 1.0 {
                    return Err(CurriculumError::invalid_config(
                        "ramp_rate must be in [0, 1]",
                    ));
                }
            }
            CurriculumStrategy::CompetenceBased {
                initial_competence,
                growth_rate,
            } => {
                if *initial_competence < 0.0 || *initial_competence > 1.0 {
                    return Err(CurriculumError::invalid_config(
                        "initial_competence must be in [0, 1]",
                    ));
                }
                if *growth_rate < 0.0 {
                    return Err(CurriculumError::invalid_config(
                        "growth_rate must be >= 0",
                    ));
                }
            }
        }

        Ok(Self { strategy, current_step: 0, total_samples })
    }

    /// Advance one training step and return the new curriculum window.
    pub fn step(&mut self) -> CurriculumWindow {
        let window = self.build_window();
        self.current_step += 1;
        window
    }

    /// Build a window snapshot for the current step without advancing.
    fn build_window(&self) -> CurriculumWindow {
        let percentile = self.current_percentile();
        let max_length = self.current_max_length();
        let active_fraction = self.active_fraction_for_percentile(percentile);

        CurriculumWindow {
            step: self.current_step,
            percentile,
            max_length,
            active_fraction,
        }
    }

    /// Get the current difficulty percentile (fraction of dataset accessible, 0..1).
    pub fn current_percentile(&self) -> f32 {
        match &self.strategy {
            CurriculumStrategy::ByDifficulty {
                initial_percentile,
                final_percentile,
                ramp_steps,
                ..
            } => {
                let t = linear_ramp_t(self.current_step, *ramp_steps);
                (initial_percentile + (final_percentile - initial_percentile) * t).clamp(0.0, 1.0)
            }
            CurriculumStrategy::CompetenceBased { .. } => self.competence(),
            CurriculumStrategy::SelfPaced { uncertainty_threshold, ramp_rate } => {
                // Percentile = fraction of samples whose uncertainty *exceeds* the
                // shrinking threshold. Since threshold shrinks, more samples are included.
                let threshold = uncertainty_threshold * (1.0 - ramp_rate).powi(self.current_step as i32);
                // Return the fraction as a proxy for "how open" the curriculum is.
                1.0 - threshold.clamp(0.0, 1.0)
            }
            CurriculumStrategy::ByLength { initial_max_len, final_max_len, ramp_steps } => {
                // Map current max length to a percentile in [0, 1].
                let cur_len = self.interpolate_length(*initial_max_len, *final_max_len, *ramp_steps);
                if *final_max_len == *initial_max_len {
                    1.0
                } else {
                    ((cur_len - initial_max_len) as f32
                        / (final_max_len - initial_max_len) as f32)
                        .clamp(0.0, 1.0)
                }
            }
        }
    }

    /// Get the current maximum sequence length for `ByLength` strategy.
    /// Returns `None` for other strategies.
    pub fn current_max_length(&self) -> Option<usize> {
        match &self.strategy {
            CurriculumStrategy::ByLength { initial_max_len, final_max_len, ramp_steps } => {
                Some(self.interpolate_length(*initial_max_len, *final_max_len, *ramp_steps))
            }
            _ => None,
        }
    }

    fn interpolate_length(
        &self,
        initial_max_len: usize,
        final_max_len: usize,
        ramp_steps: usize,
    ) -> usize {
        let t = linear_ramp_t(self.current_step, ramp_steps);
        let len =
            initial_max_len as f32 + (final_max_len as f32 - initial_max_len as f32) * t;
        (len.round() as usize).clamp(initial_max_len, final_max_len)
    }

    /// Get indices of samples currently included in the curriculum.
    ///
    /// For `ByDifficulty`, `all_difficulties` is used to rank samples. For
    /// other strategies, an empty slice is acceptable (treated as uniform).
    pub fn active_indices(&self, all_difficulties: &[f32]) -> Vec<usize> {
        let percentile = self.current_percentile();
        match &self.strategy {
            CurriculumStrategy::ByLength { initial_max_len, final_max_len, ramp_steps } => {
                let max_len = self.interpolate_length(*initial_max_len, *final_max_len, *ramp_steps);
                // Without actual length information, return indices proportional to max_len.
                // When difficulties represent lengths, filter by <= max_len.
                all_difficulties
                    .iter()
                    .enumerate()
                    .filter(|(_, &d)| (d as usize) <= max_len)
                    .map(|(i, _)| i)
                    .collect()
            }
            CurriculumStrategy::ByDifficulty { difficulty_scores, .. } => {
                let scores = if all_difficulties.is_empty() {
                    difficulty_scores.as_slice()
                } else {
                    all_difficulties
                };
                indices_below_percentile(scores, percentile)
            }
            CurriculumStrategy::SelfPaced { uncertainty_threshold, ramp_rate } => {
                let threshold =
                    uncertainty_threshold * (1.0 - ramp_rate).powi(self.current_step as i32);
                all_difficulties
                    .iter()
                    .enumerate()
                    .filter(|(_, &d)| d >= threshold)
                    .map(|(i, _)| i)
                    .collect()
            }
            CurriculumStrategy::CompetenceBased { .. } => {
                let n = ((self.total_samples as f32 * self.competence()) as usize)
                    .max(1)
                    .min(self.total_samples);
                // Sort by difficulty if available, otherwise take first n.
                if all_difficulties.is_empty() {
                    (0..n).collect()
                } else {
                    indices_below_percentile(all_difficulties, self.competence())
                        .into_iter()
                        .take(n)
                        .collect()
                }
            }
        }
    }

    /// Compute competence from the current step.
    ///
    /// Returns a value in [0, 1] for `CompetenceBased` strategies; returns 1.0
    /// for all other strategies.
    pub fn competence(&self) -> f32 {
        match &self.strategy {
            CurriculumStrategy::CompetenceBased { initial_competence, growth_rate } => {
                (initial_competence + growth_rate * self.current_step as f32).clamp(0.0, 1.0)
            }
            _ => 1.0,
        }
    }

    /// Return the total number of samples.
    pub fn total_samples(&self) -> usize {
        self.total_samples
    }

    /// Return the current step index.
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    fn active_fraction_for_percentile(&self, percentile: f32) -> f32 {
        percentile.clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Utility: linear ramp interpolation
// ---------------------------------------------------------------------------

/// Returns a value in [0, 1] representing how far along `step` is in `ramp_steps`.
fn linear_ramp_t(step: usize, ramp_steps: usize) -> f32 {
    if ramp_steps == 0 {
        return 1.0;
    }
    (step as f32 / ramp_steps as f32).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Utility: percentile-based index selection
// ---------------------------------------------------------------------------

/// Return indices of samples whose difficulty score is at or below the given
/// percentile (i.e. among the easiest `percentile * 100`% of samples).
fn indices_below_percentile(scores: &[f32], percentile: f32) -> Vec<usize> {
    if scores.is_empty() {
        return Vec::new();
    }
    let n = scores.len();
    // Sort indices by ascending difficulty.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        scores[a].partial_cmp(&scores[b]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let keep = ((n as f32 * percentile).ceil() as usize).max(1).min(n);
    order[..keep].to_vec()
}

// ---------------------------------------------------------------------------
// Sorting functions
// ---------------------------------------------------------------------------

/// Sort sample indices by length-based difficulty (ascending — shortest first).
///
/// `lengths[i]` is the sequence length of sample `i`. Returns indices sorted
/// from shortest to longest.
pub fn sort_by_length(lengths: &[usize]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..lengths.len()).collect();
    indices.sort_by_key(|&i| lengths[i]);
    indices
}

/// Sort sample indices by perplexity-based difficulty (ascending — lowest
/// perplexity / easiest first).
///
/// Higher perplexity means the model is more confused ⇒ harder sample.
pub fn sort_by_perplexity(perplexities: &[f32]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..perplexities.len()).collect();
    indices.sort_by(|&a, &b| {
        perplexities[a]
            .partial_cmp(&perplexities[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
}

/// Anti-curriculum: train on the hardest examples first and progressively
/// include easier ones.
///
/// At step 0 only the hardest `1/total_steps` fraction is accessible; by
/// `total_steps - 1` all samples are accessible.
///
/// `difficulty_scores[i]` is the difficulty of sample `i` (higher = harder).
pub fn anti_curriculum_indices(
    difficulty_scores: &[f32],
    step: usize,
    total_steps: usize,
) -> Vec<usize> {
    if difficulty_scores.is_empty() {
        return Vec::new();
    }
    let n = difficulty_scores.len();
    // Sort descending: hardest first.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        difficulty_scores[b]
            .partial_cmp(&difficulty_scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // How many samples to include grows linearly from (n / total_steps) to n.
    let fraction = if total_steps == 0 {
        1.0_f32
    } else {
        ((step + 1) as f32 / total_steps as f32).clamp(0.0, 1.0)
    };
    let keep = ((n as f32 * fraction).ceil() as usize).max(1).min(n);
    order[..keep].to_vec()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- sort_by_length -----------------------------------------------------

    #[test]
    fn test_sort_by_length_ascending() {
        let lengths = [10, 3, 7, 1, 5];
        let sorted = sort_by_length(&lengths);
        // Expect indices in ascending length order: 3(1), 1(3), 4(5), 2(7), 0(10)
        assert_eq!(sorted, vec![3, 1, 4, 2, 0]);
    }

    #[test]
    fn test_sort_by_length_empty() {
        let sorted = sort_by_length(&[]);
        assert!(sorted.is_empty());
    }

    #[test]
    fn test_sort_by_length_ties() {
        let lengths = [5, 5, 5];
        let sorted = sort_by_length(&lengths);
        assert_eq!(sorted.len(), 3);
    }

    // --- sort_by_perplexity -------------------------------------------------

    #[test]
    fn test_sort_by_perplexity_ascending() {
        let perplexities = [100.0_f32, 10.0, 50.0, 1.0];
        let sorted = sort_by_perplexity(&perplexities);
        // Expected: 3(1.0), 1(10.0), 2(50.0), 0(100.0)
        assert_eq!(sorted, vec![3, 1, 2, 0]);
    }

    #[test]
    fn test_sort_by_perplexity_empty() {
        let sorted = sort_by_perplexity(&[]);
        assert!(sorted.is_empty());
    }

    // --- anti_curriculum_indices --------------------------------------------

    #[test]
    fn test_anti_curriculum_hardest_first() {
        let scores = [0.1_f32, 0.9, 0.5, 0.3];
        // step=0, total_steps=4 → keep 25% = 1 sample → hardest = index 1
        let indices = anti_curriculum_indices(&scores, 0, 4);
        assert_eq!(indices, vec![1]);
    }

    #[test]
    fn test_anti_curriculum_all_at_last_step() {
        let scores = [0.2_f32, 0.8, 0.5];
        let indices = anti_curriculum_indices(&scores, 2, 3);
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn test_anti_curriculum_empty() {
        let indices = anti_curriculum_indices(&[], 0, 10);
        assert!(indices.is_empty());
    }

    // --- CurriculumScheduler — ByLength -------------------------------------

    #[test]
    fn test_by_length_grows_over_steps() {
        let strategy = CurriculumStrategy::ByLength {
            initial_max_len: 10,
            final_max_len: 100,
            ramp_steps: 10,
        };
        let mut sched = CurriculumScheduler::new(strategy, 1000).expect("valid config");
        let w0 = sched.step();
        let w5 = {
            for _ in 0..4 {
                sched.step();
            }
            sched.step()
        };
        let len0 = w0.max_length.expect("ByLength should have max_length");
        let len5 = w5.max_length.expect("ByLength should have max_length");
        assert!(len5 > len0, "Length should grow: {} vs {}", len5, len0);
    }

    #[test]
    fn test_by_length_clamps_at_final() {
        let strategy = CurriculumStrategy::ByLength {
            initial_max_len: 10,
            final_max_len: 50,
            ramp_steps: 5,
        };
        let mut sched = CurriculumScheduler::new(strategy, 100).expect("valid config");
        // Advance past the ramp.
        for _ in 0..20 {
            sched.step();
        }
        let w = sched.step();
        assert_eq!(w.max_length, Some(50));
    }

    #[test]
    fn test_by_length_invalid_config() {
        let strategy = CurriculumStrategy::ByLength {
            initial_max_len: 100,
            final_max_len: 10, // final < initial — invalid
            ramp_steps: 10,
        };
        assert!(CurriculumScheduler::new(strategy, 100).is_err());
    }

    // --- CurriculumScheduler — ByDifficulty ---------------------------------

    #[test]
    fn test_by_difficulty_percentile_ramp() {
        let scores: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let strategy = CurriculumStrategy::ByDifficulty {
            difficulty_scores: scores,
            initial_percentile: 0.1,
            final_percentile: 1.0,
            ramp_steps: 10,
        };
        let mut sched = CurriculumScheduler::new(strategy, 100).expect("valid config");
        let p0 = sched.step().percentile;
        for _ in 0..9 {
            sched.step();
        }
        let p10 = sched.step().percentile;
        assert!(p10 > p0, "Percentile should increase: {} vs {}", p10, p0);
    }

    #[test]
    fn test_by_difficulty_active_indices_subset() {
        let scores = vec![0.1_f32, 0.9, 0.5, 0.3, 0.7];
        let strategy = CurriculumStrategy::ByDifficulty {
            difficulty_scores: scores.clone(),
            initial_percentile: 0.4,
            final_percentile: 1.0,
            ramp_steps: 100,
        };
        let sched = CurriculumScheduler::new(strategy, 5).expect("valid config");
        let indices = sched.active_indices(&scores);
        // At step 0, percentile ≈ 0.4 → ⌈5 * 0.4⌉ = 2 easiest samples.
        assert!(indices.len() <= 3, "Expected ≤3 active indices, got {}", indices.len());
        assert!(!indices.is_empty());
    }

    // --- CurriculumScheduler — SelfPaced ------------------------------------

    #[test]
    fn test_self_paced_threshold_decreases() {
        let strategy = CurriculumStrategy::SelfPaced {
            uncertainty_threshold: 0.8,
            ramp_rate: 0.1,
        };
        let mut sched = CurriculumScheduler::new(strategy, 200).expect("valid config");
        let p0 = sched.step().percentile;
        for _ in 0..19 {
            sched.step();
        }
        let p20 = sched.step().percentile;
        // As threshold shrinks, percentile (1 - threshold) should grow.
        assert!(p20 > p0, "Self-paced percentile should grow: {} vs {}", p20, p0);
    }

    #[test]
    fn test_self_paced_invalid_ramp_rate() {
        let strategy = CurriculumStrategy::SelfPaced {
            uncertainty_threshold: 0.5,
            ramp_rate: 1.5, // invalid: > 1
        };
        assert!(CurriculumScheduler::new(strategy, 100).is_err());
    }

    // --- CurriculumScheduler — CompetenceBased ------------------------------

    #[test]
    fn test_competence_grows_linearly() {
        let strategy = CurriculumStrategy::CompetenceBased {
            initial_competence: 0.1,
            growth_rate: 0.05,
        };
        let mut sched = CurriculumScheduler::new(strategy, 100).expect("valid config");
        // At step 0: competence = 0.1
        assert!((sched.competence() - 0.1).abs() < 1e-5);
        sched.step(); // advances to step 1
        // Now current_step = 1 → competence = 0.1 + 0.05 * 1 = 0.15
        assert!(
            (sched.competence() - 0.15).abs() < 1e-5,
            "competence at step 1 should be 0.15, got {}",
            sched.competence()
        );
    }

    #[test]
    fn test_competence_clamps_at_one() {
        let strategy = CurriculumStrategy::CompetenceBased {
            initial_competence: 0.9,
            growth_rate: 0.5,
        };
        let mut sched = CurriculumScheduler::new(strategy, 100).expect("valid config");
        for _ in 0..10 {
            sched.step();
        }
        assert!(sched.competence() <= 1.0, "Competence must not exceed 1.0");
    }

    #[test]
    fn test_competence_active_fraction() {
        let strategy = CurriculumStrategy::CompetenceBased {
            initial_competence: 0.2,
            growth_rate: 0.0, // frozen
        };
        let sched = CurriculumScheduler::new(strategy, 100).expect("valid config");
        let w = sched.build_window();
        // active_fraction ≈ 0.2
        assert!(
            (w.active_fraction - 0.2).abs() < 1e-5,
            "active_fraction should ≈ 0.2, got {}",
            w.active_fraction
        );
    }

    // --- Error cases --------------------------------------------------------

    #[test]
    fn test_empty_dataset_error() {
        let strategy = CurriculumStrategy::CompetenceBased {
            initial_competence: 0.1,
            growth_rate: 0.01,
        };
        let result = CurriculumScheduler::new(strategy, 0);
        assert!(result.is_err());
        assert_eq!(result.err().map(|e| e.kind), Some(CurriculumErrorKind::EmptyDataset));
    }

    #[test]
    fn test_by_difficulty_empty_scores_error() {
        let strategy = CurriculumStrategy::ByDifficulty {
            difficulty_scores: vec![],
            initial_percentile: 0.1,
            final_percentile: 1.0,
            ramp_steps: 10,
        };
        let result = CurriculumScheduler::new(strategy, 100);
        assert!(result.is_err());
    }

    // --- Window step tracking -----------------------------------------------

    #[test]
    fn test_window_step_increments() {
        let strategy = CurriculumStrategy::CompetenceBased {
            initial_competence: 0.5,
            growth_rate: 0.0,
        };
        let mut sched = CurriculumScheduler::new(strategy, 50).expect("valid config");
        let w0 = sched.step();
        let w1 = sched.step();
        let w2 = sched.step();
        assert_eq!(w0.step, 0);
        assert_eq!(w1.step, 1);
        assert_eq!(w2.step, 2);
    }

    // --- CurriculumError display --------------------------------------------

    #[test]
    fn test_error_display() {
        let err = CurriculumError::invalid_config("bad ramp");
        let s = format!("{}", err);
        assert!(s.contains("InvalidConfig"));
        assert!(s.contains("bad ramp"));
    }

    // --- linear_ramp_t helper -----------------------------------------------

    #[test]
    fn test_linear_ramp_t_boundaries() {
        assert!((linear_ramp_t(0, 10) - 0.0).abs() < 1e-5);
        assert!((linear_ramp_t(10, 10) - 1.0).abs() < 1e-5);
        assert!((linear_ramp_t(5, 10) - 0.5).abs() < 1e-5);
        // Zero ramp_steps returns 1.0 immediately.
        assert!((linear_ramp_t(0, 0) - 1.0).abs() < 1e-5);
    }

    // --- sort_by_perplexity — single element --------------------------------

    #[test]
    fn test_sort_by_perplexity_single() {
        let sorted = sort_by_perplexity(&[42.0_f32]);
        assert_eq!(sorted, vec![0]);
    }

    // --- sort_by_length — single element ------------------------------------

    #[test]
    fn test_sort_by_length_single() {
        let sorted = sort_by_length(&[7]);
        assert_eq!(sorted, vec![0]);
    }

    // --- anti_curriculum — progressive inclusion ----------------------------

    /// Verify that successive steps include progressively more samples.
    #[test]
    fn test_anti_curriculum_progressive_inclusion() {
        let scores: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let total_steps = 10;
        let counts: Vec<usize> = (0..total_steps)
            .map(|step| anti_curriculum_indices(&scores, step, total_steps).len())
            .collect();

        for pair in counts.windows(2) {
            assert!(
                pair[1] >= pair[0],
                "anti-curriculum should include at least as many samples at later steps: {:?}",
                counts
            );
        }
    }

    /// With total_steps=0 the function should return all samples immediately.
    #[test]
    fn test_anti_curriculum_zero_total_steps() {
        let scores = vec![0.3_f32, 0.7, 0.5];
        let indices = anti_curriculum_indices(&scores, 0, 0);
        assert_eq!(indices.len(), 3, "zero total_steps should return all samples");
    }

    // --- ByDifficulty — full-dataset at final percentile -------------------

    #[test]
    fn test_by_difficulty_full_at_final_percentile() {
        let scores: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let strategy = CurriculumStrategy::ByDifficulty {
            difficulty_scores: scores.clone(),
            initial_percentile: 1.0, // full access from start
            final_percentile: 1.0,
            ramp_steps: 5,
        };
        let sched = CurriculumScheduler::new(strategy, 10).expect("valid config");
        let indices = sched.active_indices(&scores);
        assert_eq!(indices.len(), 10, "100% percentile should include all samples");
    }

    // --- CompetenceBased — active_indices grows with steps -----------------

    #[test]
    fn test_competence_active_indices_grows() {
        let strategy = CurriculumStrategy::CompetenceBased {
            initial_competence: 0.1,
            growth_rate: 0.1,
        };
        let mut sched = CurriculumScheduler::new(strategy, 100).expect("valid config");

        // Step 0 → competence = 0.1 → ~10 samples.
        let idx0 = sched.active_indices(&[]).len();
        sched.step(); // advance to step 1

        // Step 1 → competence = 0.2 → ~20 samples.
        let idx1 = sched.active_indices(&[]).len();
        assert!(idx1 >= idx0, "active indices should grow with competence: {} vs {}", idx1, idx0);
    }

    // --- SelfPaced — active_indices expands as threshold decreases ----------

    #[test]
    fn test_self_paced_active_indices_expand() {
        // Uncertainties uniformly spread in [0, 1].
        let uncertainties: Vec<f32> = (0..20).map(|i| i as f32 / 19.0).collect();
        let strategy = CurriculumStrategy::SelfPaced {
            uncertainty_threshold: 0.8, // initially only high-uncertainty samples
            ramp_rate: 0.3,
        };
        let mut sched = CurriculumScheduler::new(strategy, 20).expect("valid config");

        let count_step0 = sched.active_indices(&uncertainties).len();
        for _ in 0..5 { sched.step(); }
        let count_step5 = sched.active_indices(&uncertainties).len();

        assert!(
            count_step5 >= count_step0,
            "self-paced should include more samples as threshold drops: {count_step0} → {count_step5}"
        );
    }

    // --- CurriculumWindow — active_fraction clamps to [0, 1] ---------------

    #[test]
    fn test_active_fraction_stays_in_range() {
        let strategy = CurriculumStrategy::CompetenceBased {
            initial_competence: 0.5,
            growth_rate: 0.2,
        };
        let mut sched = CurriculumScheduler::new(strategy, 50).expect("valid config");
        for _ in 0..20 {
            let w = sched.step();
            assert!(
                w.active_fraction >= 0.0 && w.active_fraction <= 1.0,
                "active_fraction out of [0,1]: {}",
                w.active_fraction
            );
        }
    }

    // --- ByLength — active_indices filtered by length ----------------------

    #[test]
    fn test_by_length_active_indices_filtered() {
        // Sample lengths: 5, 10, 20, 30, 50.
        let lengths = vec![5.0_f32, 10.0, 20.0, 30.0, 50.0];
        let strategy = CurriculumStrategy::ByLength {
            initial_max_len: 15,
            final_max_len: 50,
            ramp_steps: 100,
        };
        let sched = CurriculumScheduler::new(strategy, 5).expect("valid config");
        // At step 0, max_len == initial_max_len = 15 → only lengths ≤ 15 included.
        let active = sched.active_indices(&lengths);
        for &i in &active {
            assert!(
                lengths[i] <= 15.0,
                "sample with length {} should not be active at max_len=15",
                lengths[i]
            );
        }
        assert!(!active.is_empty(), "some samples must be active");
    }

    // --- Error: initial_competence out of range -----------------------------

    #[test]
    fn test_competence_invalid_initial() {
        let strategy = CurriculumStrategy::CompetenceBased {
            initial_competence: 1.5, // > 1 — invalid
            growth_rate: 0.01,
        };
        assert!(CurriculumScheduler::new(strategy, 100).is_err());
    }

    // --- Error: negative growth_rate ----------------------------------------

    #[test]
    fn test_competence_negative_growth_rate() {
        let strategy = CurriculumStrategy::CompetenceBased {
            initial_competence: 0.5,
            growth_rate: -0.01, // invalid
        };
        assert!(CurriculumScheduler::new(strategy, 100).is_err());
    }

    // --- Error: ByDifficulty percentile out of range -----------------------

    #[test]
    fn test_by_difficulty_invalid_percentile() {
        let strategy = CurriculumStrategy::ByDifficulty {
            difficulty_scores: vec![1.0_f32, 2.0],
            initial_percentile: -0.1, // invalid
            final_percentile: 1.0,
            ramp_steps: 10,
        };
        assert!(CurriculumScheduler::new(strategy, 2).is_err());
    }

    // --- indices_below_percentile at 0 percentile --------------------------

    #[test]
    fn test_indices_below_percentile_minimum() {
        // Calling active_indices with percentile=0 should still return at least 1 sample.
        let scores: Vec<f32> = (1..=5).map(|i| i as f32).collect();
        let strategy = CurriculumStrategy::ByDifficulty {
            difficulty_scores: scores.clone(),
            initial_percentile: 0.0, // effectively 0 but clamped to 1
            final_percentile: 1.0,
            ramp_steps: 1000,
        };
        let sched = CurriculumScheduler::new(strategy, 5).expect("valid config");
        let active = sched.active_indices(&scores);
        assert!(!active.is_empty(), "at least one sample should always be active");
    }
}
