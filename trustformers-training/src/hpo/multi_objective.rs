//! Multi-objective hyperparameter optimization (MOHO).
//!
//! Uses NSGA-II style Pareto front selection to find configurations that
//! optimize multiple objectives simultaneously.
//!
//! # Example
//!
//! ```rust
//! use trustformers_training::hpo::multi_objective::{
//!     MultiObjectiveHpo, MultiObjectiveHpoConfig, HpSearchSpace, ObjectiveDirection,
//!     MultiObjectiveResult, HpConfig,
//! };
//! use std::collections::HashMap;
//!
//! let mut search_space = HashMap::new();
//! search_space.insert("lr".to_string(), HpSearchSpace::Float { min: 1e-5, max: 1e-2, log_scale: true });
//!
//! let config = MultiObjectiveHpoConfig {
//!     search_space,
//!     objectives: vec![
//!         ("accuracy".to_string(), ObjectiveDirection::Maximize),
//!         ("latency_ms".to_string(), ObjectiveDirection::Minimize),
//!     ],
//!     n_trials: 20,
//!     seed: 42,
//!     use_pareto_front: true,
//! };
//!
//! let mut hpo = MultiObjectiveHpo::new(config).unwrap();
//! let cfg = hpo.suggest();
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::error_handling::{ErrorContext, ErrorSeverity, ErrorType, SystemInfo, TrainingError};

// ---------------------------------------------------------------------------
// Core value and config types
// ---------------------------------------------------------------------------

/// A single hyperparameter value (supports multiple primitive types).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum HpValue {
    /// Floating-point value
    Float(f64),
    /// Integer value
    Int(i64),
    /// Boolean value
    Bool(bool),
    /// Plain string value
    String(String),
    /// Categorical choice
    Choice(String),
}

/// A single hyperparameter configuration produced by the search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HpConfig {
    /// Mapping from parameter name to its value.
    pub params: HashMap<String, HpValue>,
    /// Unique trial identifier (monotonically increasing).
    pub trial_id: usize,
}

/// An evaluation result associating a configuration with its multi-objective outcomes.
#[derive(Debug, Clone)]
pub struct MultiObjectiveResult {
    /// The configuration that was evaluated.
    pub config: HpConfig,
    /// One scalar per objective, in the same order as `MultiObjectiveHpoConfig::objectives`.
    pub objectives: Vec<f64>,
    /// Optional numeric metadata (e.g., wall-clock time, memory usage).
    pub metadata: HashMap<String, f64>,
}

// ---------------------------------------------------------------------------
// Objective direction
// ---------------------------------------------------------------------------

/// Whether an objective should be minimized or maximized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObjectiveDirection {
    /// Lower values are better (e.g., loss, latency).
    Minimize,
    /// Higher values are better (e.g., accuracy, F1).
    Maximize,
}

// ---------------------------------------------------------------------------
// Search space
// ---------------------------------------------------------------------------

/// Definition of a single hyperparameter's search space.
#[derive(Debug, Clone)]
pub enum HpSearchSpace {
    /// Continuous floating-point range, optionally sampled in log-space.
    Float {
        /// Minimum value (inclusive).
        min: f64,
        /// Maximum value (inclusive).
        max: f64,
        /// If true, sample uniformly in log-space.
        log_scale: bool,
    },
    /// Discrete integer range `[min, max]` (inclusive on both ends).
    Int {
        /// Minimum value.
        min: i64,
        /// Maximum value.
        max: i64,
    },
    /// Unordered set of choices.
    Categorical {
        /// Available options.
        choices: Vec<HpValue>,
    },
    /// Boolean flag.
    Bool,
}

// ---------------------------------------------------------------------------
// Pareto front
// ---------------------------------------------------------------------------

/// The set of non-dominated (Pareto-optimal) solutions found so far.
#[derive(Debug, Clone, Default)]
pub struct ParetoFront {
    /// All currently non-dominated solutions.
    pub solutions: Vec<MultiObjectiveResult>,
}

impl ParetoFront {
    /// Create an empty Pareto front.
    pub fn new() -> Self {
        Self { solutions: Vec::new() }
    }

    /// Update the front with a new candidate result.
    ///
    /// If the new result is non-dominated (no existing solution dominates it),
    /// it is added and any solutions it dominates are removed.
    pub fn update(
        &mut self,
        result: MultiObjectiveResult,
        directions: &[(String, ObjectiveDirection)],
    ) {
        // Check whether any existing solution dominates the new result.
        let dominated_by_existing = self
            .solutions
            .iter()
            .any(|s| Self::dominates(&s.objectives, &result.objectives, directions));

        if dominated_by_existing {
            return;
        }

        // Remove solutions dominated by the new result.
        self.solutions
            .retain(|s| !Self::dominates(&result.objectives, &s.objectives, directions));

        self.solutions.push(result);
    }

    /// Returns `true` if objective vector `a` strictly dominates vector `b`.
    ///
    /// Dominance means `a` is **no worse** than `b` in every objective **and
    /// strictly better** in at least one.
    pub fn dominates(
        a: &[f64],
        b: &[f64],
        directions: &[(String, ObjectiveDirection)],
    ) -> bool {
        if a.len() != b.len() || a.len() != directions.len() {
            return false;
        }

        let mut at_least_one_better = false;

        for (i, (_, dir)) in directions.iter().enumerate() {
            let better = match dir {
                ObjectiveDirection::Minimize => a[i] < b[i],
                ObjectiveDirection::Maximize => a[i] > b[i],
            };
            let worse = match dir {
                ObjectiveDirection::Minimize => a[i] > b[i],
                ObjectiveDirection::Maximize => a[i] < b[i],
            };

            if worse {
                return false;
            }
            if better {
                at_least_one_better = true;
            }
        }

        at_least_one_better
    }

    /// Number of non-dominated solutions currently in the front.
    pub fn len(&self) -> usize {
        self.solutions.len()
    }

    /// Returns `true` if the Pareto front contains no solutions.
    pub fn is_empty(&self) -> bool {
        self.solutions.is_empty()
    }

    /// Return the best solution for objective at `idx` according to `direction`.
    pub fn best_for_objective(
        &self,
        idx: usize,
        direction: &ObjectiveDirection,
    ) -> Option<&MultiObjectiveResult> {
        self.solutions.iter().reduce(|best, cur| {
            let cur_val = cur.objectives.get(idx).copied().unwrap_or(f64::NAN);
            let best_val = best.objectives.get(idx).copied().unwrap_or(f64::NAN);
            let cur_is_better = match direction {
                ObjectiveDirection::Minimize => cur_val < best_val,
                ObjectiveDirection::Maximize => cur_val > best_val,
            };
            if cur_is_better { cur } else { best }
        })
    }

    /// Human-readable summary of the Pareto front.
    pub fn summary(&self) -> String {
        if self.solutions.is_empty() {
            return "ParetoFront: empty".to_string();
        }
        let mut lines = vec![format!("ParetoFront ({} solutions):", self.solutions.len())];
        for (i, sol) in self.solutions.iter().enumerate() {
            let obj_str: Vec<String> =
                sol.objectives.iter().map(|v| format!("{:.4}", v)).collect();
            lines.push(format!(
                "  [{}] trial={} objectives=[{}]",
                i,
                sol.config.trial_id,
                obj_str.join(", ")
            ));
        }
        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// HPO configuration
// ---------------------------------------------------------------------------

/// Full configuration for a multi-objective HPO run.
#[derive(Debug, Clone)]
pub struct MultiObjectiveHpoConfig {
    /// Named hyperparameters and their search spaces.
    pub search_space: HashMap<String, HpSearchSpace>,
    /// Named objectives and their optimisation directions.
    pub objectives: Vec<(String, ObjectiveDirection)>,
    /// Total number of trials to run.
    pub n_trials: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
    /// When `true`, maintain and update a Pareto front after each trial.
    pub use_pareto_front: bool,
}

// ---------------------------------------------------------------------------
// PRNG (xorshift64 — no external dependency)
// ---------------------------------------------------------------------------

/// Simple xorshift64 PRNG for reproducible sampling without the `rand` crate.
fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Sample a `f64` in `[0, 1)` from the PRNG state.
fn rng_f64(state: &mut u64) -> f64 {
    (xorshift64(state) >> 11) as f64 / (1u64 << 53) as f64
}

/// Sample a `usize` in `[0, n)`.
fn rng_usize(state: &mut u64, n: usize) -> usize {
    (xorshift64(state) as usize) % n
}

// ---------------------------------------------------------------------------
// Multi-objective HPO runner
// ---------------------------------------------------------------------------

/// Multi-objective HPO runner.
///
/// Manages trial suggestion (currently random search), result recording, and
/// Pareto front maintenance. The architecture supports extension to NSGA-II or
/// any Pareto-aware acquisition strategy.
pub struct MultiObjectiveHpo {
    config: MultiObjectiveHpoConfig,
    results: Vec<MultiObjectiveResult>,
    pareto_front: ParetoFront,
    rng_state: u64,
    trial_counter: usize,
}

impl MultiObjectiveHpo {
    /// Create a new multi-objective HPO runner with the given configuration.
    ///
    /// Returns an error when the configuration is invalid (e.g., no objectives,
    /// no search space parameters).
    pub fn new(config: MultiObjectiveHpoConfig) -> Result<Self, TrainingError> {
        if config.objectives.is_empty() {
            return Err(make_config_error("MultiObjectiveHpo requires at least one objective"));
        }
        if config.search_space.is_empty() {
            return Err(make_config_error(
                "MultiObjectiveHpo requires at least one search space parameter",
            ));
        }
        if config.n_trials == 0 {
            return Err(make_config_error("n_trials must be > 0"));
        }

        // Ensure seed is non-zero (xorshift64 state must not be 0).
        let rng_state = if config.seed == 0 { 0xdeadbeef_cafebabe } else { config.seed };

        Ok(Self {
            config,
            results: Vec::new(),
            pareto_front: ParetoFront::new(),
            rng_state,
            trial_counter: 0,
        })
    }

    /// Sample the next hyperparameter configuration to evaluate.
    ///
    /// Uses random search (uniform / log-uniform depending on the space definition).
    /// Future extensions can replace this with NSGA-II tournament selection or a
    /// Gaussian-process-based acquisition function.
    pub fn suggest(&mut self) -> HpConfig {
        let trial_id = self.trial_counter;
        self.trial_counter += 1;

        // Collect the search space entries first to avoid a simultaneous
        // mutable (rng_state) + immutable (config.search_space) borrow.
        let entries: Vec<(String, HpSearchSpace)> = self
            .config
            .search_space
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let mut params: HashMap<String, HpValue> = HashMap::new();
        for (name, space) in &entries {
            let value = self.sample_from_space(space);
            params.insert(name.clone(), value);
        }

        HpConfig { params, trial_id }
    }

    /// Record the objective values for a completed trial.
    ///
    /// The number of objectives in `result.objectives` must match the length of
    /// `config.objectives`.  If Pareto-front tracking is enabled the front is
    /// updated automatically.
    pub fn record(&mut self, result: MultiObjectiveResult) {
        if self.config.use_pareto_front {
            self.pareto_front.update(result.clone(), &self.config.objectives);
        }
        self.results.push(result);
    }

    /// Return a reference to the current Pareto front.
    pub fn pareto_front(&self) -> &ParetoFront {
        &self.pareto_front
    }

    /// Return all recorded trial results.
    pub fn results(&self) -> &[MultiObjectiveResult] {
        &self.results
    }

    /// Find the best result under a weighted-sum scalarisation of objectives.
    ///
    /// `weights` must have the same length as `config.objectives`. Objectives
    /// are normalised to `[0, 1]` before weighting so that different scales do
    /// not dominate the selection.
    pub fn best_weighted(&self, weights: &[f64]) -> Option<&MultiObjectiveResult> {
        if self.results.is_empty() || weights.len() != self.config.objectives.len() {
            return None;
        }

        // Compute min/max per objective for normalisation.
        let n_obj = self.config.objectives.len();
        let mut obj_min = vec![f64::INFINITY; n_obj];
        let mut obj_max = vec![f64::NEG_INFINITY; n_obj];
        for r in &self.results {
            for (i, &v) in r.objectives.iter().enumerate() {
                if i < n_obj {
                    if v < obj_min[i] {
                        obj_min[i] = v;
                    }
                    if v > obj_max[i] {
                        obj_max[i] = v;
                    }
                }
            }
        }

        self.results.iter().max_by(|a, b| {
            let score_a = weighted_score(a, weights, &obj_min, &obj_max, &self.config.objectives);
            let score_b = weighted_score(b, weights, &obj_min, &obj_max, &self.config.objectives);
            score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Export all trial results as a CSV string.
    pub fn to_csv(&self) -> String {
        if self.results.is_empty() {
            return String::new();
        }

        // Collect all parameter names in sorted order.
        let mut param_names: Vec<String> = self
            .config
            .search_space
            .keys()
            .cloned()
            .collect();
        param_names.sort();

        let obj_names: Vec<String> =
            self.config.objectives.iter().map(|(n, _)| n.clone()).collect();

        let mut lines = Vec::new();
        let header = format!(
            "trial_id,{},{}",
            param_names.join(","),
            obj_names.join(",")
        );
        lines.push(header);

        for r in &self.results {
            let params: Vec<String> = param_names
                .iter()
                .map(|k| match r.config.params.get(k) {
                    Some(HpValue::Float(v)) => format!("{:.6}", v),
                    Some(HpValue::Int(v)) => v.to_string(),
                    Some(HpValue::Bool(v)) => v.to_string(),
                    Some(HpValue::String(v)) | Some(HpValue::Choice(v)) => v.clone(),
                    None => String::new(),
                })
                .collect();
            let objs: Vec<String> = r.objectives.iter().map(|v| format!("{:.6}", v)).collect();
            lines.push(format!("{},{},{}", r.config.trial_id, params.join(","), objs.join(",")));
        }

        lines.join("\n")
    }

    /// Serialise all trial results to a JSON string.
    pub fn to_json(&self) -> Result<String, TrainingError> {
        // Build a JSON-friendly representation without using serde on the whole struct.
        let entries: Vec<serde_json::Value> = self
            .results
            .iter()
            .map(|r| {
                let params_json: serde_json::Map<String, serde_json::Value> = r
                    .config
                    .params
                    .iter()
                    .map(|(k, v)| (k.clone(), hp_value_to_json(v)))
                    .collect();
                serde_json::json!({
                    "trial_id": r.config.trial_id,
                    "params": params_json,
                    "objectives": r.objectives,
                    "metadata": r.metadata,
                })
            })
            .collect();

        serde_json::to_string_pretty(&serde_json::json!({ "trials": entries })).map_err(|e| {
            make_config_error(&format!("JSON serialisation failed: {}", e))
        })
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    fn sample_from_space(&mut self, space: &HpSearchSpace) -> HpValue {
        match space {
            HpSearchSpace::Float { min, max, log_scale } => {
                let u = rng_f64(&mut self.rng_state);
                let v = if *log_scale {
                    let log_min = min.ln();
                    let log_max = max.ln();
                    (log_min + u * (log_max - log_min)).exp()
                } else {
                    min + u * (max - min)
                };
                HpValue::Float(v)
            }
            HpSearchSpace::Int { min, max } => {
                let range = (max - min + 1) as usize;
                let v = *min + rng_usize(&mut self.rng_state, range) as i64;
                HpValue::Int(v)
            }
            HpSearchSpace::Categorical { choices } => {
                if choices.is_empty() {
                    return HpValue::String(String::new());
                }
                let idx = rng_usize(&mut self.rng_state, choices.len());
                choices[idx].clone()
            }
            HpSearchSpace::Bool => {
                let bit = xorshift64(&mut self.rng_state) & 1;
                HpValue::Bool(bit == 1)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn weighted_score(
    result: &MultiObjectiveResult,
    weights: &[f64],
    obj_min: &[f64],
    obj_max: &[f64],
    objectives: &[(String, ObjectiveDirection)],
) -> f64 {
    result
        .objectives
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            if i >= weights.len() || i >= objectives.len() {
                return 0.0;
            }
            let range = obj_max[i] - obj_min[i];
            let norm = if range.abs() < 1e-12 {
                0.5
            } else {
                (v - obj_min[i]) / range
            };
            // Flip minimised objectives so that "higher normalised score = better".
            let oriented = match &objectives[i].1 {
                ObjectiveDirection::Minimize => 1.0 - norm,
                ObjectiveDirection::Maximize => norm,
            };
            weights[i] * oriented
        })
        .sum()
}

fn hp_value_to_json(v: &HpValue) -> serde_json::Value {
    match v {
        HpValue::Float(f) => serde_json::Value::Number(
            serde_json::Number::from_f64(*f).unwrap_or(serde_json::Number::from(0)),
        ),
        HpValue::Int(i) => serde_json::json!(i),
        HpValue::Bool(b) => serde_json::json!(b),
        HpValue::String(s) | HpValue::Choice(s) => serde_json::json!(s),
    }
}

// ---------------------------------------------------------------------------
// Pareto front computation (standalone functions)
// ---------------------------------------------------------------------------

/// Compute the indices of non-dominated (Pareto-optimal) points.
///
/// `points` is a slice of `(params, objectives)` pairs; only the `objectives`
/// part is used for dominance checks.  Lower values are assumed to be better
/// for every objective (minimisation).  To handle maximisation objectives,
/// negate those dimensions before calling.
///
/// A point `p` dominates `q` if it is **no worse** in every objective and
/// **strictly better** in at least one.
pub fn compute_pareto_front(points: &[(Vec<f32>, Vec<f32>)]) -> Vec<usize> {
    let n = points.len();
    let mut dominated = vec![false; n];

    for i in 0..n {
        if dominated[i] {
            continue;
        }
        for j in 0..n {
            if i == j || dominated[j] {
                continue;
            }
            let objs_i = &points[i].1;
            let objs_j = &points[j].1;
            if strictly_dominates_f32(objs_i, objs_j) {
                dominated[j] = true;
            }
        }
    }

    (0..n).filter(|&i| !dominated[i]).collect()
}

/// Returns `true` when `a` strictly dominates `b` under minimisation.
fn strictly_dominates_f32(a: &[f32], b: &[f32]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut strictly_better = false;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        if ai > bi {
            // `a` is worse in this dimension → does not dominate.
            return false;
        }
        if ai < bi {
            strictly_better = true;
        }
    }
    strictly_better
}

// ---------------------------------------------------------------------------
// Hypervolume indicator
// ---------------------------------------------------------------------------

/// Compute the hypervolume dominated by `pareto_front` relative to
/// `reference_point`.
///
/// Assumes **minimisation** (all objectives should be smaller than the
/// reference point for the point to contribute positive volume).
///
/// Uses the exact sweep algorithm for 2-D fronts and a Monte-Carlo fallback
/// for higher dimensions (using 10 000 random samples with a seeded LCG).
///
/// Returns `0.0` if the front is empty or no point is dominated by the
/// reference.
pub fn hypervolume_indicator(pareto_front: &[Vec<f32>], reference_point: &[f32]) -> f32 {
    if pareto_front.is_empty() || reference_point.is_empty() {
        return 0.0;
    }

    let n_obj = reference_point.len();

    // Fast exact path for 2-D.
    if n_obj == 2 {
        return hypervolume_2d(pareto_front, reference_point);
    }

    // General case: Monte-Carlo approximation.
    hypervolume_monte_carlo(pareto_front, reference_point, 10_000)
}

/// Exact 2-D hypervolume by sweeping.
fn hypervolume_2d(front: &[Vec<f32>], reference: &[f32]) -> f32 {
    // Filter points that are dominated by the reference (i.e., strictly below it).
    let mut pts: Vec<(f32, f32)> = front
        .iter()
        .filter(|p| p.len() >= 2 && p[0] < reference[0] && p[1] < reference[1])
        .map(|p| (p[0], p[1]))
        .collect();

    if pts.is_empty() {
        return 0.0;
    }

    // Sort by first objective ascending.
    pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut hv = 0.0_f32;
    let mut prev_x = pts[0].0;
    let mut min_y = pts[0].1;

    for i in 1..pts.len() {
        let (x, y) = pts[i];
        // Width × height contribution of the previous step.
        hv += (x - prev_x) * (reference[1] - min_y);
        prev_x = x;
        min_y = min_y.min(y);
    }
    // Final segment to reference.
    hv += (reference[0] - prev_x) * (reference[1] - min_y);
    hv.max(0.0)
}

/// Monte-Carlo hypervolume estimate: sample random points in the bounding box
/// and count those dominated by at least one Pareto front point.
fn hypervolume_monte_carlo(front: &[Vec<f32>], reference: &[f32], n_samples: u64) -> f32 {
    let n_obj = reference.len();

    // Compute bounding box minimum per objective.
    let mut lower: Vec<f32> = vec![f32::INFINITY; n_obj];
    for pt in front {
        for (d, &v) in pt.iter().enumerate().take(n_obj) {
            if v < lower[d] {
                lower[d] = v;
            }
        }
    }
    // Replace inf with 0 if no points are valid.
    for v in &mut lower {
        if !v.is_finite() {
            *v = 0.0;
        }
    }

    let box_vol: f64 = reference
        .iter()
        .zip(lower.iter())
        .map(|(&r, &l)| (r - l).max(0.0) as f64)
        .product();

    if box_vol <= 0.0 {
        return 0.0;
    }

    // xorshift64 PRNG.
    let mut rng_state: u64 = 0xdeadbeef_12345678;
    let mut dominated_count: u64 = 0;

    for _ in 0..n_samples {
        // Sample a random point in the bounding box.
        let sample: Vec<f32> = (0..n_obj)
            .map(|d| {
                let u = {
                    rng_state ^= rng_state << 13;
                    rng_state ^= rng_state >> 7;
                    rng_state ^= rng_state << 17;
                    (rng_state >> 11) as f64 / (1u64 << 53) as f64
                };
                lower[d] + u as f32 * (reference[d] - lower[d]).max(0.0)
            })
            .collect();

        // Check if dominated by any Pareto point.
        let is_dominated = front.iter().any(|pt| {
            pt.iter()
                .zip(sample.iter())
                .all(|(&pv, &sv)| pv <= sv)
        });
        if is_dominated {
            dominated_count += 1;
        }
    }

    let fraction = dominated_count as f64 / n_samples as f64;
    (fraction * box_vol) as f32
}

// ---------------------------------------------------------------------------
// NSGA-II non-domination sorting
// ---------------------------------------------------------------------------

/// NSGA-II non-domination sort with crowding distance tie-breaking.
///
/// `objectives` is a slice of objective vectors (one per individual).
/// All objectives are assumed to be **minimised**.  To handle maximisation,
/// negate those dimensions before calling.
///
/// Returns indices sorted from best (rank 1, high crowding distance) to worst.
pub fn non_domination_sort(objectives: &[Vec<f32>]) -> Vec<usize> {
    let n = objectives.len();
    if n == 0 {
        return Vec::new();
    }

    // ── Step 1: compute domination counts and dominated sets ──────────────
    let mut domination_count: Vec<usize> = vec![0; n];
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            if strictly_dominates_f32(&objectives[i], &objectives[j]) {
                dominated_by[i].push(j);
                domination_count[j] += 1;
            }
        }
    }

    // ── Step 2: build Pareto fronts ───────────────────────────────────────
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut current_front: Vec<usize> = (0..n)
        .filter(|&i| domination_count[i] == 0)
        .collect();

    while !current_front.is_empty() {
        fronts.push(current_front.clone());
        let mut next_front = Vec::new();
        for &i in &current_front {
            for &j in &dominated_by[i] {
                domination_count[j] = domination_count[j].saturating_sub(1);
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        current_front = next_front;
    }

    // ── Step 3: sort each front by crowding distance (descending) ─────────
    let mut result = Vec::with_capacity(n);
    for front in &fronts {
        let crowding = crowding_distances(front, objectives);
        // Pair each index with its crowding distance, sort descending.
        let mut front_with_cd: Vec<(usize, f32)> = front
            .iter()
            .zip(crowding.iter())
            .map(|(&idx, &cd)| (idx, cd))
            .collect();
        front_with_cd.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        result.extend(front_with_cd.iter().map(|(idx, _)| *idx));
    }
    result
}

/// Compute crowding distances for a set of indices within a front.
fn crowding_distances(front: &[usize], objectives: &[Vec<f32>]) -> Vec<f32> {
    let m = front.len();
    if m == 0 {
        return Vec::new();
    }
    if m <= 2 {
        return vec![f32::INFINITY; m];
    }

    let n_obj = objectives[front[0]].len();
    let mut distances = vec![0.0_f32; m];

    for obj_idx in 0..n_obj {
        // Sort front members by this objective.
        let mut order: Vec<usize> = (0..m).collect();
        order.sort_by(|&a, &b| {
            let va = objectives[front[a]].get(obj_idx).copied().unwrap_or(0.0);
            let vb = objectives[front[b]].get(obj_idx).copied().unwrap_or(0.0);
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let min_val = objectives[front[order[0]]].get(obj_idx).copied().unwrap_or(0.0);
        let max_val = objectives[front[order[m - 1]]].get(obj_idx).copied().unwrap_or(0.0);
        let range = max_val - min_val;

        // Boundary solutions get infinite distance.
        distances[order[0]] = f32::INFINITY;
        distances[order[m - 1]] = f32::INFINITY;

        if range.abs() < f32::EPSILON {
            continue;
        }

        for k in 1..(m - 1) {
            let v_next = objectives[front[order[k + 1]]].get(obj_idx).copied().unwrap_or(0.0);
            let v_prev = objectives[front[order[k - 1]]].get(obj_idx).copied().unwrap_or(0.0);
            distances[order[k]] += (v_next - v_prev) / range;
        }
    }

    distances
}

fn make_config_error(msg: &str) -> TrainingError {
    TrainingError {
        error_type: ErrorType::Configuration,
        message: msg.to_string(),
        error_code: "MOHO_CONFIG_ERR".to_string(),
        severity: ErrorSeverity::High,
        context: ErrorContext {
            component: "MultiObjectiveHpo".to_string(),
            operation: "new".to_string(),
            epoch: None,
            step: None,
            batch_size: None,
            learning_rate: None,
            model_state: None,
            system_info: SystemInfo {
                memory_usage: None,
                gpu_memory_usage: None,
                cpu_usage: None,
                disk_space: None,
                network_status: None,
            },
            additional_data: HashMap::new(),
        },
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        recovery_suggestions: Vec::new(),
        related_errors: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_config(n_trials: usize) -> MultiObjectiveHpoConfig {
        let mut search_space = HashMap::new();
        search_space.insert(
            "lr".to_string(),
            HpSearchSpace::Float { min: 1e-5, max: 1e-2, log_scale: true },
        );
        search_space.insert("batch".to_string(), HpSearchSpace::Int { min: 8, max: 128 });
        search_space.insert(
            "dropout".to_string(),
            HpSearchSpace::Float { min: 0.0, max: 0.5, log_scale: false },
        );

        MultiObjectiveHpoConfig {
            search_space,
            objectives: vec![
                ("accuracy".to_string(), ObjectiveDirection::Maximize),
                ("latency_ms".to_string(), ObjectiveDirection::Minimize),
            ],
            n_trials,
            seed: 42,
            use_pareto_front: true,
        }
    }

    fn make_result(config: HpConfig, acc: f64, lat: f64) -> MultiObjectiveResult {
        MultiObjectiveResult {
            config,
            objectives: vec![acc, lat],
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_new_valid_config() {
        let config = simple_config(10);
        assert!(MultiObjectiveHpo::new(config).is_ok());
    }

    #[test]
    fn test_new_empty_objectives_fails() {
        let mut config = simple_config(10);
        config.objectives = vec![];
        assert!(MultiObjectiveHpo::new(config).is_err());
    }

    #[test]
    fn test_new_empty_search_space_fails() {
        let mut config = simple_config(10);
        config.search_space = HashMap::new();
        assert!(MultiObjectiveHpo::new(config).is_err());
    }

    #[test]
    fn test_new_zero_trials_fails() {
        let mut config = simple_config(0);
        config.n_trials = 0;
        assert!(MultiObjectiveHpo::new(config).is_err());
    }

    #[test]
    fn test_suggest_produces_correct_param_names() {
        let config = simple_config(5);
        let mut hpo = MultiObjectiveHpo::new(config).expect("valid config");
        let suggestion = hpo.suggest();

        assert!(suggestion.params.contains_key("lr"));
        assert!(suggestion.params.contains_key("batch"));
        assert!(suggestion.params.contains_key("dropout"));
    }

    #[test]
    fn test_suggest_increments_trial_id() {
        let config = simple_config(10);
        let mut hpo = MultiObjectiveHpo::new(config).expect("valid config");
        let s0 = hpo.suggest();
        let s1 = hpo.suggest();
        assert_eq!(s0.trial_id, 0);
        assert_eq!(s1.trial_id, 1);
    }

    #[test]
    fn test_float_range_sampled_in_bounds() {
        let mut space = HashMap::new();
        space.insert(
            "lr".to_string(),
            HpSearchSpace::Float { min: 0.001, max: 0.1, log_scale: false },
        );
        let config = MultiObjectiveHpoConfig {
            search_space: space,
            objectives: vec![("loss".to_string(), ObjectiveDirection::Minimize)],
            n_trials: 100,
            seed: 7,
            use_pareto_front: false,
        };
        let mut hpo = MultiObjectiveHpo::new(config).expect("valid");
        for _ in 0..50 {
            let s = hpo.suggest();
            if let Some(HpValue::Float(v)) = s.params.get("lr") {
                assert!(*v >= 0.001 && *v <= 0.1, "lr={} out of bounds", v);
            } else {
                panic!("Expected Float for 'lr'");
            }
        }
    }

    #[test]
    fn test_int_range_sampled_in_bounds() {
        let mut space = HashMap::new();
        space.insert("batch".to_string(), HpSearchSpace::Int { min: 1, max: 64 });
        let config = MultiObjectiveHpoConfig {
            search_space: space,
            objectives: vec![("loss".to_string(), ObjectiveDirection::Minimize)],
            n_trials: 100,
            seed: 13,
            use_pareto_front: false,
        };
        let mut hpo = MultiObjectiveHpo::new(config).expect("valid");
        for _ in 0..50 {
            let s = hpo.suggest();
            if let Some(HpValue::Int(v)) = s.params.get("batch") {
                assert!(*v >= 1 && *v <= 64, "batch={} out of bounds", v);
            } else {
                panic!("Expected Int for 'batch'");
            }
        }
    }

    #[test]
    fn test_pareto_dominance_basic() {
        let dirs = vec![
            ("acc".to_string(), ObjectiveDirection::Maximize),
            ("lat".to_string(), ObjectiveDirection::Minimize),
        ];
        // a dominates b: a is better in both
        assert!(ParetoFront::dominates(&[0.9, 10.0], &[0.8, 20.0], &dirs));
        // b dominates a: b is better in both
        assert!(!ParetoFront::dominates(&[0.8, 20.0], &[0.9, 10.0], &dirs));
        // non-dominated: a better in acc, worse in lat
        assert!(!ParetoFront::dominates(&[0.9, 20.0], &[0.8, 10.0], &dirs));
    }

    #[test]
    fn test_pareto_front_update() {
        let dirs = vec![
            ("acc".to_string(), ObjectiveDirection::Maximize),
            ("lat".to_string(), ObjectiveDirection::Minimize),
        ];
        let mut front = ParetoFront::new();

        let cfg_a = HpConfig { params: HashMap::new(), trial_id: 0 };
        let cfg_b = HpConfig { params: HashMap::new(), trial_id: 1 };
        let cfg_c = HpConfig { params: HashMap::new(), trial_id: 2 };

        // Insert A (0.9, 10)
        front.update(make_result(cfg_a, 0.9, 10.0), &dirs);
        assert_eq!(front.len(), 1);

        // Insert B (0.8, 5) — non-dominated (better lat, worse acc)
        front.update(make_result(cfg_b, 0.8, 5.0), &dirs);
        assert_eq!(front.len(), 2);

        // Insert C (0.95, 4) — dominates both A and B
        front.update(make_result(cfg_c, 0.95, 4.0), &dirs);
        assert_eq!(front.len(), 1);
        assert_eq!(front.solutions[0].config.trial_id, 2);
    }

    #[test]
    fn test_record_and_pareto_tracked() {
        let config = simple_config(20);
        let mut hpo = MultiObjectiveHpo::new(config).expect("valid");

        for i in 0..5 {
            let cfg = hpo.suggest();
            let result = make_result(cfg, 0.7 + 0.05 * i as f64, 100.0 - 10.0 * i as f64);
            hpo.record(result);
        }

        assert_eq!(hpo.results().len(), 5);
        assert!(!hpo.pareto_front().is_empty());
    }

    #[test]
    fn test_best_weighted() {
        let config = simple_config(10);
        let mut hpo = MultiObjectiveHpo::new(config).expect("valid");

        let r1 = make_result(HpConfig { params: HashMap::new(), trial_id: 0 }, 0.9, 50.0);
        let r2 = make_result(HpConfig { params: HashMap::new(), trial_id: 1 }, 0.6, 10.0);
        hpo.record(r1);
        hpo.record(r2);

        // Weight accuracy 1.0, latency 0.0 → pick highest accuracy = r1
        let best = hpo.best_weighted(&[1.0, 0.0]).expect("best exists");
        assert_eq!(best.config.trial_id, 0);

        // Weight accuracy 0.0, latency 1.0 → pick lowest latency = r2
        let best = hpo.best_weighted(&[0.0, 1.0]).expect("best exists");
        assert_eq!(best.config.trial_id, 1);
    }

    #[test]
    fn test_to_csv_structure() {
        let config = simple_config(5);
        let mut hpo = MultiObjectiveHpo::new(config).expect("valid");
        let cfg = hpo.suggest();
        hpo.record(make_result(cfg, 0.85, 30.0));

        let csv = hpo.to_csv();
        let lines: Vec<&str> = csv.lines().collect();
        assert!(lines.len() >= 2, "CSV should have header + at least one row");
        assert!(lines[0].contains("trial_id"));
        assert!(lines[0].contains("accuracy"));
        assert!(lines[0].contains("latency_ms"));
    }

    #[test]
    fn test_to_json_valid() {
        let config = simple_config(5);
        let mut hpo = MultiObjectiveHpo::new(config).expect("valid");
        let cfg = hpo.suggest();
        hpo.record(make_result(cfg, 0.85, 30.0));

        let json = hpo.to_json().expect("serialisation ok");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
        assert!(parsed["trials"].is_array());
        assert_eq!(parsed["trials"].as_array().map(|a| a.len()), Some(1));
    }

    #[test]
    fn test_categorical_sampling() {
        let mut space = HashMap::new();
        space.insert(
            "optimiser".to_string(),
            HpSearchSpace::Categorical {
                choices: vec![
                    HpValue::Choice("adam".to_string()),
                    HpValue::Choice("sgd".to_string()),
                    HpValue::Choice("adamw".to_string()),
                ],
            },
        );
        let config = MultiObjectiveHpoConfig {
            search_space: space,
            objectives: vec![("loss".to_string(), ObjectiveDirection::Minimize)],
            n_trials: 30,
            seed: 99,
            use_pareto_front: false,
        };
        let mut hpo = MultiObjectiveHpo::new(config).expect("valid");
        for _ in 0..20 {
            let s = hpo.suggest();
            match s.params.get("optimiser") {
                Some(HpValue::Choice(v)) => {
                    assert!(["adam", "sgd", "adamw"].contains(&v.as_str()), "unexpected: {}", v);
                }
                other => panic!("Expected Choice, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_pareto_front_summary_non_empty() {
        let dirs = vec![("acc".to_string(), ObjectiveDirection::Maximize)];
        let mut front = ParetoFront::new();
        let cfg = HpConfig { params: HashMap::new(), trial_id: 0 };
        front.update(MultiObjectiveResult { config: cfg, objectives: vec![0.9], metadata: HashMap::new() }, &dirs);
        let summary = front.summary();
        assert!(summary.contains("ParetoFront"));
        assert!(summary.contains("solutions"));
    }

    // ─── compute_pareto_front tests ───────────────────────────────────────

    // ── Test 16: single point is its own Pareto front ──
    #[test]
    fn test_compute_pareto_front_single_point() {
        let points = vec![(vec![1.0_f32], vec![0.5_f32])];
        let front = compute_pareto_front(&points);
        assert_eq!(front, vec![0]);
    }

    // ── Test 17: dominated point excluded ──
    #[test]
    fn test_compute_pareto_front_dominated_excluded() {
        // Point A (0.1, 0.1) dominates B (0.5, 0.5) under minimisation.
        let points = vec![
            (vec![], vec![0.1_f32, 0.1_f32]),
            (vec![], vec![0.5_f32, 0.5_f32]),
        ];
        let front = compute_pareto_front(&points);
        assert!(front.contains(&0), "A should be in front");
        assert!(!front.contains(&1), "B should be dominated");
    }

    // ── Test 18: non-dominated tradeoff set ──
    #[test]
    fn test_compute_pareto_front_tradeoff() {
        // A: (0.1, 0.9) and B: (0.9, 0.1) — neither dominates the other.
        let points = vec![
            (vec![], vec![0.1_f32, 0.9_f32]),
            (vec![], vec![0.9_f32, 0.1_f32]),
        ];
        let front = compute_pareto_front(&points);
        assert_eq!(front.len(), 2, "both should be on the front");
    }

    // ── Test 19: empty input returns empty front ──
    #[test]
    fn test_compute_pareto_front_empty() {
        let front = compute_pareto_front(&[]);
        assert!(front.is_empty());
    }

    // ── Test 20: 3-objective Pareto front ──
    #[test]
    fn test_compute_pareto_front_3d() {
        let pts = vec![
            (vec![], vec![0.1_f32, 0.2_f32, 0.3_f32]), // best in all three → dominates others
            (vec![], vec![0.5_f32, 0.5_f32, 0.5_f32]),
            (vec![], vec![0.9_f32, 0.9_f32, 0.9_f32]),
        ];
        let front = compute_pareto_front(&pts);
        assert!(front.contains(&0), "index 0 should be on front");
        assert_eq!(front.len(), 1, "only one non-dominated point");
    }

    // ─── hypervolume_indicator tests ──────────────────────────────────────

    // ── Test 21: empty front gives 0 ──
    #[test]
    fn test_hypervolume_empty_front() {
        assert!((hypervolume_indicator(&[], &[1.0, 1.0])).abs() < 1e-6);
    }

    // ── Test 22: known 2-D hypervolume ──
    #[test]
    fn test_hypervolume_2d_known_value() {
        // Single point (0, 0), reference (1, 1) → volume = 1.
        let front = vec![vec![0.0_f32, 0.0_f32]];
        let hv = hypervolume_indicator(&front, &[1.0_f32, 1.0_f32]);
        assert!((hv - 1.0).abs() < 1e-4, "expected HV ≈ 1.0, got {hv}");
    }

    // ── Test 23: point beyond reference contributes nothing ──
    #[test]
    fn test_hypervolume_point_beyond_reference() {
        // Point (2.0, 2.0) with reference (1.0, 1.0) → no volume contributed.
        let front = vec![vec![2.0_f32, 2.0_f32]];
        let hv = hypervolume_indicator(&front, &[1.0_f32, 1.0_f32]);
        assert!(hv == 0.0, "point beyond reference should give 0, got {hv}");
    }

    // ── Test 24: 2-D front with multiple points ──
    #[test]
    fn test_hypervolume_2d_multiple_points() {
        // Two-point front at (0.0, 0.5) and (0.5, 0.0), reference (1.0, 1.0).
        // Expected HV = 0.5*0.5 + 0.5*1.0 = ... use sweep for exact value.
        let front = vec![
            vec![0.0_f32, 0.5_f32],
            vec![0.5_f32, 0.0_f32],
        ];
        let hv = hypervolume_indicator(&front, &[1.0_f32, 1.0_f32]);
        assert!(hv > 0.0, "hypervolume should be positive, got {hv}");
        assert!(hv <= 1.0, "hypervolume should be at most reference volume, got {hv}");
    }

    // ─── non_domination_sort tests ────────────────────────────────────────

    // ── Test 25: single point returns it at index 0 ──
    #[test]
    fn test_non_domination_sort_single() {
        let result = non_domination_sort(&[vec![0.5_f32, 0.5_f32]]);
        assert_eq!(result, vec![0]);
    }

    // ── Test 26: best point appears first ──
    #[test]
    fn test_non_domination_sort_order() {
        // A dominates B and C.
        let objs = vec![
            vec![0.1_f32, 0.1_f32], // index 0 — best
            vec![0.5_f32, 0.5_f32], // index 1
            vec![0.9_f32, 0.9_f32], // index 2 — worst
        ];
        let sorted = non_domination_sort(&objs);
        assert_eq!(sorted[0], 0, "best point should be first");
    }

    // ── Test 27: empty objectives returns empty ──
    #[test]
    fn test_non_domination_sort_empty() {
        assert!(non_domination_sort(&[]).is_empty());
    }

    // ── Test 28: non-dominated front contains all tradeoff points ──
    #[test]
    fn test_non_domination_sort_tradeoff() {
        // All on the Pareto front: no one dominates another.
        let objs = vec![
            vec![0.1_f32, 0.9_f32],
            vec![0.5_f32, 0.5_f32],
            vec![0.9_f32, 0.1_f32],
        ];
        let sorted = non_domination_sort(&objs);
        // All should appear in the output.
        assert_eq!(sorted.len(), 3);
    }

    // ── Test 29: result contains all indices exactly once ──
    #[test]
    fn test_non_domination_sort_all_indices_once() {
        let objs: Vec<Vec<f32>> = (0..10)
            .map(|i| vec![i as f32 * 0.1, (10 - i) as f32 * 0.1])
            .collect();
        let sorted = non_domination_sort(&objs);
        assert_eq!(sorted.len(), 10);
        let mut seen = sorted.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), 10, "all indices should appear exactly once");
    }
}
