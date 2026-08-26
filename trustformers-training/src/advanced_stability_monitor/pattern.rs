//! Instability patterns and the detector that evaluates them.
//!
//! Split out of the parent module to keep both files under the workspace's
//! 2000-line limit; the public path is unchanged because [`super`] re-exports
//! every item defined here.

use super::{RiskLevel, TrainingDynamics, TrendDirection};
use anyhow::Result;
use std::collections::HashMap;

/// Tolerance used by the `equals` indicator condition.
const PATTERN_EQUALITY_EPSILON: f32 = 1e-6;

/// Pattern detector for complex training dynamics.
///
/// Holds a library of named instability [`Pattern`]s and evaluates every one of them against a
/// [`TrainingDynamics`] snapshot. [`PatternDetector::new`] seeds the library with the built-in
/// patterns listed in [`PatternDetector::builtin_patterns`]; callers may add their own with
/// [`PatternDetector::register_pattern`].
pub struct PatternDetector {
    pattern_library: HashMap<String, Pattern>,
    /// Risk level reported when the matching pattern fires.
    severities: HashMap<String, RiskLevel>,
}

impl Default for PatternDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl PatternDetector {
    /// A detector preloaded with the built-in instability patterns.
    pub fn new() -> Self {
        let mut detector = Self {
            pattern_library: HashMap::new(),
            severities: HashMap::new(),
        };
        for (pattern, severity) in Self::builtin_patterns() {
            // The built-in patterns only use metric and condition names that
            // `evaluate_indicator` knows, so registration cannot fail here.
            detector
                .register_pattern(pattern, severity)
                .expect("built-in stability patterns must use known metrics and conditions");
        }
        detector
    }

    /// A detector with an empty library, for callers that want only their own patterns.
    pub fn empty() -> Self {
        Self {
            pattern_library: HashMap::new(),
            severities: HashMap::new(),
        }
    }

    /// Names of the metrics an indicator may reference.
    ///
    /// Scalars read straight off [`TrainingDynamics`]:
    /// `lr_effectiveness`, `convergence_velocity`, `oscillation_frequency`.
    ///
    /// Trend predicates, exposed as `1.0`/`0.0` so they compare like any other metric:
    /// `loss_trend_{decreasing,increasing,stable,oscillating,diverging}` and the same five
    /// for `gradient_trend_*`.
    ///
    /// Statistics derived from `phase_trajectory` (`(loss, gradient_norm)` pairs):
    /// `loss_spike_ratio` (max loss over median loss), `max_gradient_norm`,
    /// `min_gradient_norm`, `gradient_norm_collapse_ratio` (min over max) and
    /// `trajectory_length`.
    pub fn known_metrics() -> &'static [&'static str] {
        &[
            "lr_effectiveness",
            "convergence_velocity",
            "oscillation_frequency",
            "loss_trend_decreasing",
            "loss_trend_increasing",
            "loss_trend_stable",
            "loss_trend_oscillating",
            "loss_trend_diverging",
            "gradient_trend_decreasing",
            "gradient_trend_increasing",
            "gradient_trend_stable",
            "gradient_trend_oscillating",
            "gradient_trend_diverging",
            "loss_spike_ratio",
            "max_gradient_norm",
            "min_gradient_norm",
            "gradient_norm_collapse_ratio",
            "trajectory_length",
        ]
    }

    /// Comparison operators an indicator may use.
    pub fn known_conditions() -> &'static [&'static str] {
        &[
            "greater_than",
            "greater_or_equal",
            "less_than",
            "less_or_equal",
            "equals",
        ]
    }

    /// Add a pattern to the library, replacing any pattern of the same name.
    ///
    /// # Errors
    ///
    /// The pattern has no indicators, `min_indicators` is zero or larger than the number of
    /// indicators (the rule could never fire), or an indicator names a metric outside
    /// [`PatternDetector::known_metrics`] or a condition outside
    /// [`PatternDetector::known_conditions`]. Validating here means
    /// [`PatternDetector::detect_patterns`] can never silently ignore a typo'd rule.
    pub fn register_pattern(&mut self, pattern: Pattern, severity: RiskLevel) -> Result<()> {
        if pattern.indicators.is_empty() {
            return Err(anyhow::anyhow!(
                "pattern '{}' has no indicators, so it could never be evaluated",
                pattern.name
            ));
        }
        if pattern.min_indicators == 0 || pattern.min_indicators > pattern.indicators.len() {
            return Err(anyhow::anyhow!(
                "pattern '{}' requires {} of its {} indicators; min_indicators must be in \
                 1..={}",
                pattern.name,
                pattern.min_indicators,
                pattern.indicators.len(),
                pattern.indicators.len()
            ));
        }
        for indicator in &pattern.indicators {
            if !Self::known_metrics().contains(&indicator.metric.as_str()) {
                return Err(anyhow::anyhow!(
                    "pattern '{}' references unknown metric '{}'; known metrics are {:?}",
                    pattern.name,
                    indicator.metric,
                    Self::known_metrics()
                ));
            }
            if !Self::known_conditions().contains(&indicator.condition.as_str()) {
                return Err(anyhow::anyhow!(
                    "pattern '{}' uses unknown condition '{}'; known conditions are {:?}",
                    pattern.name,
                    indicator.condition,
                    Self::known_conditions()
                ));
            }
        }
        self.severities.insert(pattern.name.clone(), severity);
        self.pattern_library.insert(pattern.name.clone(), pattern);
        Ok(())
    }

    /// Number of patterns currently in the library.
    pub fn pattern_count(&self) -> usize {
        self.pattern_library.len()
    }

    /// Evaluate every library pattern against `dynamics`.
    ///
    /// A pattern is reported when at least [`Pattern::min_indicators`] of its indicators hold —
    /// for the built-in library that means *all* of them, because each built-in pattern is a
    /// conjunction. `confidence` is the exact fraction that held, so a fully-matched pattern
    /// reports `1.0`. Results are ordered by descending confidence, then by name, so the
    /// output is stable across runs (the library is a `HashMap`).
    ///
    /// This used to be `Vec::new()` against an empty library — no pattern could ever fire.
    pub fn detect_patterns(&self, dynamics: &TrainingDynamics) -> Vec<DetectedPattern> {
        let stats = TrajectoryStats::from(dynamics);
        let mut detected = Vec::new();

        for pattern in self.pattern_library.values() {
            let total = pattern.indicators.len();
            if total == 0 {
                continue;
            }
            let satisfied = pattern
                .indicators
                .iter()
                .filter(|indicator| Self::evaluate_indicator(indicator, dynamics, &stats))
                .count();
            let confidence = satisfied as f32 / total as f32;
            if satisfied >= pattern.min_indicators {
                detected.push(DetectedPattern {
                    pattern: pattern.clone(),
                    confidence,
                    severity: self
                        .severities
                        .get(&pattern.name)
                        .cloned()
                        .unwrap_or(RiskLevel::Medium),
                });
            }
        }

        detected.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.pattern.name.cmp(&b.pattern.name))
        });
        detected
    }

    /// Does this indicator hold for the given dynamics?
    fn evaluate_indicator(
        indicator: &PatternIndicator,
        dynamics: &TrainingDynamics,
        stats: &TrajectoryStats,
    ) -> bool {
        let Some(value) = Self::metric_value(&indicator.metric, dynamics, stats) else {
            // Unreachable for registered patterns — `register_pattern` rejects unknown
            // metrics — but an unknown name must never count as "satisfied".
            return false;
        };
        let threshold = indicator.threshold;
        match indicator.condition.as_str() {
            "greater_than" => value > threshold,
            "greater_or_equal" => value >= threshold,
            "less_than" => value < threshold,
            "less_or_equal" => value <= threshold,
            "equals" => (value - threshold).abs() <= PATTERN_EQUALITY_EPSILON,
            _ => false,
        }
    }

    /// Numeric value of a named metric, or `None` when the name is unknown.
    fn metric_value(
        metric: &str,
        dynamics: &TrainingDynamics,
        stats: &TrajectoryStats,
    ) -> Option<f32> {
        let flag = |b: bool| if b { 1.0 } else { 0.0 };
        Some(match metric {
            "lr_effectiveness" => dynamics.lr_effectiveness,
            "convergence_velocity" => dynamics.convergence_velocity,
            "oscillation_frequency" => dynamics.oscillation_frequency,
            "loss_trend_decreasing" => {
                flag(matches!(dynamics.loss_trend, TrendDirection::Decreasing))
            },
            "loss_trend_increasing" => {
                flag(matches!(dynamics.loss_trend, TrendDirection::Increasing))
            },
            "loss_trend_stable" => flag(matches!(dynamics.loss_trend, TrendDirection::Stable)),
            "loss_trend_oscillating" => {
                flag(matches!(dynamics.loss_trend, TrendDirection::Oscillating))
            },
            "loss_trend_diverging" => {
                flag(matches!(dynamics.loss_trend, TrendDirection::Diverging))
            },
            "gradient_trend_decreasing" => flag(matches!(
                dynamics.gradient_trend,
                TrendDirection::Decreasing
            )),
            "gradient_trend_increasing" => flag(matches!(
                dynamics.gradient_trend,
                TrendDirection::Increasing
            )),
            "gradient_trend_stable" => {
                flag(matches!(dynamics.gradient_trend, TrendDirection::Stable))
            },
            "gradient_trend_oscillating" => flag(matches!(
                dynamics.gradient_trend,
                TrendDirection::Oscillating
            )),
            "gradient_trend_diverging" => {
                flag(matches!(dynamics.gradient_trend, TrendDirection::Diverging))
            },
            "loss_spike_ratio" => stats.loss_spike_ratio,
            "max_gradient_norm" => stats.max_gradient_norm,
            "min_gradient_norm" => stats.min_gradient_norm,
            "gradient_norm_collapse_ratio" => stats.gradient_norm_collapse_ratio,
            "trajectory_length" => stats.trajectory_length,
            _ => return None,
        })
    }

    /// The instability patterns every detector starts with.
    ///
    /// Each is a rule an experienced practitioner would apply by eye to a loss/gradient trace:
    /// a diverging loss, a loss spike, exploding or collapsing gradient norms, an oscillating
    /// loss, a learning rate that has stopped buying progress, and a stalled run.
    pub fn builtin_patterns() -> Vec<(Pattern, RiskLevel)> {
        let indicator = PatternIndicator::new;
        vec![
            (
                Pattern::all_of(
                    "loss_divergence",
                    "Loss is diverging while gradients grow — the run is failing",
                    vec![
                        indicator("loss_trend_diverging", "greater_or_equal", 1.0),
                        indicator("gradient_trend_increasing", "greater_or_equal", 1.0),
                    ],
                ),
                RiskLevel::Critical,
            ),
            (
                Pattern::all_of(
                    "loss_spike",
                    "A loss value far above the trajectory median",
                    vec![indicator("loss_spike_ratio", "greater_than", 3.0)],
                ),
                RiskLevel::High,
            ),
            (
                Pattern::all_of(
                    "gradient_explosion",
                    "Gradient norms have grown past a safe magnitude",
                    vec![
                        indicator("max_gradient_norm", "greater_than", 100.0),
                        indicator("gradient_trend_increasing", "greater_or_equal", 1.0),
                    ],
                ),
                RiskLevel::Critical,
            ),
            (
                Pattern::all_of(
                    "gradient_norm_collapse",
                    "Gradient norms have collapsed towards zero (vanishing signal)",
                    vec![
                        indicator("max_gradient_norm", "less_than", 1e-4),
                        indicator("gradient_trend_decreasing", "greater_or_equal", 1.0),
                    ],
                ),
                RiskLevel::High,
            ),
            (
                Pattern::all_of(
                    "loss_oscillation",
                    "Loss oscillates instead of descending — usually too high an LR",
                    vec![
                        indicator("loss_trend_oscillating", "greater_or_equal", 1.0),
                        indicator("oscillation_frequency", "greater_than", 0.3),
                    ],
                ),
                RiskLevel::Medium,
            ),
            (
                Pattern::all_of(
                    "learning_rate_divergence",
                    "The learning rate has stopped buying loss reduction",
                    vec![
                        indicator("lr_effectiveness", "less_than", 0.1),
                        indicator("loss_trend_increasing", "greater_or_equal", 1.0),
                    ],
                ),
                RiskLevel::High,
            ),
            (
                Pattern::all_of(
                    "training_stagnation",
                    "Loss is flat and convergence has effectively stopped",
                    vec![
                        indicator("loss_trend_stable", "greater_or_equal", 1.0),
                        indicator("convergence_velocity", "less_than", 1e-3),
                    ],
                ),
                RiskLevel::Low,
            ),
        ]
    }
}

/// Scalar statistics derived from a [`TrainingDynamics::phase_trajectory`].
struct TrajectoryStats {
    /// `max(loss) / median(loss)`; `1.0` when the trajectory is empty or the median is zero.
    loss_spike_ratio: f32,
    max_gradient_norm: f32,
    min_gradient_norm: f32,
    /// `min / max` of the gradient norms; `1.0` when there is nothing to compare.
    gradient_norm_collapse_ratio: f32,
    trajectory_length: f32,
}

impl TrajectoryStats {
    fn from(dynamics: &TrainingDynamics) -> Self {
        let trajectory = &dynamics.phase_trajectory;
        if trajectory.is_empty() {
            return Self {
                loss_spike_ratio: 1.0,
                max_gradient_norm: 0.0,
                min_gradient_norm: 0.0,
                gradient_norm_collapse_ratio: 1.0,
                trajectory_length: 0.0,
            };
        }

        let mut losses: Vec<f32> = trajectory.iter().map(|(loss, _)| *loss).collect();
        losses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = losses.len() / 2;
        let median = if losses.len().is_multiple_of(2) {
            (losses[mid - 1] + losses[mid]) / 2.0
        } else {
            losses[mid]
        };
        let max_loss = losses.last().copied().unwrap_or(0.0);
        let loss_spike_ratio = if median.abs() > f32::EPSILON { max_loss / median } else { 1.0 };

        let mut max_gradient_norm = f32::NEG_INFINITY;
        let mut min_gradient_norm = f32::INFINITY;
        for (_, norm) in trajectory {
            max_gradient_norm = max_gradient_norm.max(*norm);
            min_gradient_norm = min_gradient_norm.min(*norm);
        }
        let gradient_norm_collapse_ratio = if max_gradient_norm.abs() > f32::EPSILON {
            min_gradient_norm / max_gradient_norm
        } else {
            1.0
        };

        Self {
            loss_spike_ratio,
            max_gradient_norm,
            min_gradient_norm,
            gradient_norm_collapse_ratio,
            trajectory_length: trajectory.len() as f32,
        }
    }
}

/// A named instability rule: a set of [`PatternIndicator`]s plus how many of them have to
/// hold before the pattern is reported.
///
/// `min_indicators` is what makes the rule mean something. Every built-in pattern is a
/// *conjunction* ("gradients collapsed **and** the gradient trend is decreasing"), so
/// requiring only a fraction of the indicators would report `gradient_norm_collapse` on any
/// healthy run whose gradients happen to be shrinking. Build patterns with
/// [`Pattern::all_of`] (conjunction), [`Pattern::any_of`] (disjunction) or
/// [`Pattern::at_least`] rather than by struct literal, so the field is never left
/// inconsistent with `indicators`.
#[derive(Debug, Clone)]
pub struct Pattern {
    pub name: String,
    pub description: String,
    pub indicators: Vec<PatternIndicator>,
    /// How many indicators must hold. Validated by
    /// [`PatternDetector::register_pattern`] to be in `1..=indicators.len()`.
    pub min_indicators: usize,
}

impl Pattern {
    /// A pattern that fires only when **every** indicator holds.
    pub fn all_of(
        name: impl Into<String>,
        description: impl Into<String>,
        indicators: Vec<PatternIndicator>,
    ) -> Self {
        let min_indicators = indicators.len().max(1);
        Self {
            name: name.into(),
            description: description.into(),
            indicators,
            min_indicators,
        }
    }

    /// A pattern that fires as soon as **any** indicator holds.
    pub fn any_of(
        name: impl Into<String>,
        description: impl Into<String>,
        indicators: Vec<PatternIndicator>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            indicators,
            min_indicators: 1,
        }
    }

    /// A pattern that fires once at least `min_indicators` of its indicators hold.
    pub fn at_least(
        name: impl Into<String>,
        description: impl Into<String>,
        indicators: Vec<PatternIndicator>,
        min_indicators: usize,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            indicators,
            min_indicators,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PatternIndicator {
    pub metric: String,
    pub condition: String,
    pub threshold: f32,
}

impl PatternIndicator {
    /// Build an indicator; see [`PatternDetector::known_metrics`] and
    /// [`PatternDetector::known_conditions`] for the accepted names.
    pub fn new(metric: impl Into<String>, condition: impl Into<String>, threshold: f32) -> Self {
        Self {
            metric: metric.into(),
            condition: condition.into(),
            threshold,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DetectedPattern {
    pub pattern: Pattern,
    /// Fraction of the pattern's indicators that held — always
    /// `>= min_indicators / indicators.len()`, and `1.0` for a fully matched pattern.
    pub confidence: f32,
    pub severity: RiskLevel,
}
