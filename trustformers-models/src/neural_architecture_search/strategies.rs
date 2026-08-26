//! Search strategies that consume real evaluation results.
//!
//! Each strategy here does something different with the measurements: the
//! REINFORCE controller updates a policy from the reward, the Bayesian optimizer
//! fits a Gaussian process surrogate and maximises Expected Improvement, and
//! NSGA-II performs genuine non-dominated sorting with crowding distance. None
//! of them is random sampling with a different log line.

use std::collections::HashMap;

use scirs2_core::random::*;

use super::{Architecture, ArchitectureEvaluation, SearchSpace};

/// A categorical policy over one search-space dimension or choice.
#[derive(Debug, Clone)]
struct CategoricalPolicy {
    /// Unnormalised log-probabilities, one per option.
    logits: Vec<f32>,
}

impl CategoricalPolicy {
    fn new(options: usize) -> Self {
        Self {
            logits: vec![0.0; options.max(1)],
        }
    }

    fn probabilities(&self) -> Vec<f32> {
        let max = self.logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !max.is_finite() {
            return vec![1.0 / self.logits.len() as f32; self.logits.len()];
        }
        let exps: Vec<f32> = self.logits.iter().map(|l| (l - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum <= 0.0 {
            return vec![1.0 / self.logits.len() as f32; self.logits.len()];
        }
        exps.into_iter().map(|e| e / sum).collect()
    }

    fn sample(&self, rng: &mut impl Rng) -> usize {
        let probabilities = self.probabilities();
        let draw: f32 = rng.random::<f32>();
        let mut cumulative = 0.0;
        for (index, probability) in probabilities.iter().enumerate() {
            cumulative += probability;
            if draw <= cumulative {
                return index;
            }
        }
        probabilities.len().saturating_sub(1)
    }

    /// REINFORCE update: `logits += lr * advantage * (onehot(action) - p)`.
    fn reinforce(&mut self, action: usize, advantage: f32, learning_rate: f32) {
        let probabilities = self.probabilities();
        for (index, logit) in self.logits.iter_mut().enumerate() {
            let indicator = if index == action { 1.0 } else { 0.0 };
            *logit += learning_rate * advantage * (indicator - probabilities[index]);
        }
    }
}

/// A REINFORCE controller over the search space.
///
/// The controller keeps one categorical policy per dimension and per categorical
/// choice, samples architectures from it, and moves probability mass towards the
/// actions that earned an above-average reward. Because the update uses the
/// measured fitness, the strategy provably differs from random sampling.
#[derive(Debug, Clone)]
pub struct ReinforceController {
    dimension_policies: HashMap<String, CategoricalPolicy>,
    choice_policies: HashMap<String, CategoricalPolicy>,
    /// Exponential moving average of the reward, used as the baseline.
    baseline: f32,
    baseline_initialised: bool,
    learning_rate: f32,
    baseline_decay: f32,
}

impl ReinforceController {
    /// Build a controller for `search_space`.
    pub fn new(search_space: &SearchSpace, learning_rate: f32) -> Self {
        let mut dimension_policies = HashMap::new();
        for (name, range) in &search_space.dimensions {
            let options = (((range.max - range.min) / range.step.max(1)) + 1).max(1) as usize;
            dimension_policies.insert(name.clone(), CategoricalPolicy::new(options));
        }

        let mut choice_policies = HashMap::new();
        for (name, options) in &search_space.choices {
            choice_policies.insert(name.clone(), CategoricalPolicy::new(options.len()));
        }

        Self {
            dimension_policies,
            choice_policies,
            baseline: 0.0,
            baseline_initialised: false,
            learning_rate,
            baseline_decay: 0.9,
        }
    }

    /// Sample an architecture from the current policy, returning the actions
    /// taken so they can be reinforced.
    pub fn sample(
        &self,
        search_space: &SearchSpace,
        rng: &mut impl Rng,
    ) -> (Architecture, SampledActions) {
        let mut architecture = Architecture::new();
        let mut dimension_actions = HashMap::new();
        let mut choice_actions = HashMap::new();

        for (name, range) in &search_space.dimensions {
            let Some(policy) = self.dimension_policies.get(name) else {
                continue;
            };
            let action = policy.sample(rng);
            let value = range.min + (action as i32) * range.step.max(1);
            architecture.dimensions.insert(name.clone(), value.min(range.max));
            dimension_actions.insert(name.clone(), action);
        }

        for (name, options) in &search_space.choices {
            if options.is_empty() {
                continue;
            }
            let Some(policy) = self.choice_policies.get(name) else {
                continue;
            };
            let action = policy.sample(rng).min(options.len() - 1);
            architecture.choices.insert(name.clone(), options[action].clone());
            choice_actions.insert(name.clone(), action);
        }

        (
            architecture,
            SampledActions {
                dimension_actions,
                choice_actions,
            },
        )
    }

    /// Update the policy from a measured reward.
    pub fn update(&mut self, actions: &SampledActions, reward: f32) {
        if !self.baseline_initialised {
            self.baseline = reward;
            self.baseline_initialised = true;
        }
        let advantage = reward - self.baseline;
        self.baseline = self.baseline_decay * self.baseline + (1.0 - self.baseline_decay) * reward;

        for (name, action) in &actions.dimension_actions {
            if let Some(policy) = self.dimension_policies.get_mut(name) {
                policy.reinforce(*action, advantage, self.learning_rate);
            }
        }
        for (name, action) in &actions.choice_actions {
            if let Some(policy) = self.choice_policies.get_mut(name) {
                policy.reinforce(*action, advantage, self.learning_rate);
            }
        }
    }

    /// Current reward baseline (the controller's running expectation).
    pub fn baseline(&self) -> f32 {
        self.baseline
    }

    /// Probability the controller currently assigns to one dimension's options.
    pub fn dimension_probabilities(&self, dimension: &str) -> Option<Vec<f32>> {
        self.dimension_policies.get(dimension).map(CategoricalPolicy::probabilities)
    }
}

/// The actions a controller took while sampling an architecture.
#[derive(Debug, Clone, Default)]
pub struct SampledActions {
    dimension_actions: HashMap<String, usize>,
    choice_actions: HashMap<String, usize>,
}

/// Encode an architecture as a normalised feature vector for the surrogate model.
pub fn encode_architecture(architecture: &Architecture, search_space: &SearchSpace) -> Vec<f32> {
    let mut names: Vec<&String> = search_space.dimensions.keys().collect();
    names.sort();

    let mut features = Vec::with_capacity(names.len() + search_space.choices.len());
    for name in names {
        let Some(range) = search_space.dimensions.get(name) else {
            continue;
        };
        let value = architecture.dimensions.get(name).copied().unwrap_or(range.min) as f32;
        let span = (range.max - range.min).max(1) as f32;
        features.push(((value - range.min as f32) / span).clamp(0.0, 1.0));
    }

    let mut choice_names: Vec<&String> = search_space.choices.keys().collect();
    choice_names.sort();
    for name in choice_names {
        let Some(options) = search_space.choices.get(name) else {
            continue;
        };
        if options.is_empty() {
            continue;
        }
        let index = architecture
            .choices
            .get(name)
            .and_then(|value| options.iter().position(|option| option == value))
            .unwrap_or(0);
        features.push(index as f32 / options.len().max(1) as f32);
    }

    features
}

/// A Gaussian-process surrogate with an RBF kernel.
///
/// Small by design: NAS runs evaluate hundreds, not millions, of candidates, so
/// an exact `O(n^3)` Cholesky solve is the right tool.
pub struct GaussianProcess {
    inputs: Vec<Vec<f32>>,
    targets: Vec<f32>,
    length_scale: f32,
    noise: f32,
    /// Lower-triangular Cholesky factor of the kernel matrix.
    cholesky: Vec<Vec<f32>>,
    /// Solution of `K alpha = y`.
    alpha: Vec<f32>,
    mean: f32,
}

impl GaussianProcess {
    /// Fit a GP to the observed `(architecture encoding, fitness)` pairs.
    pub fn fit(inputs: Vec<Vec<f32>>, targets: Vec<f32>, length_scale: f32, noise: f32) -> Self {
        let n = inputs.len();
        let mean = if n == 0 { 0.0 } else { targets.iter().sum::<f32>() / n as f32 };

        let mut kernel = vec![vec![0.0f32; n]; n];
        for i in 0..n {
            for j in 0..n {
                kernel[i][j] = rbf(&inputs[i], &inputs[j], length_scale);
                if i == j {
                    kernel[i][j] += noise;
                }
            }
        }

        let cholesky = cholesky_decompose(&kernel);
        let centered: Vec<f32> = targets.iter().map(|t| t - mean).collect();
        let alpha = cholesky_solve(&cholesky, &centered);

        Self {
            inputs,
            targets,
            length_scale,
            noise,
            cholesky,
            alpha,
            mean,
        }
    }

    /// Posterior mean and standard deviation at `x`.
    pub fn predict(&self, x: &[f32]) -> (f32, f32) {
        if self.inputs.is_empty() {
            return (self.mean, 1.0);
        }

        let k: Vec<f32> =
            self.inputs.iter().map(|input| rbf(input, x, self.length_scale)).collect();
        let mean = self.mean + k.iter().zip(self.alpha.iter()).map(|(a, b)| a * b).sum::<f32>();

        // v = L^-1 k ; variance = k(x,x) - v^T v
        let v = forward_substitute(&self.cholesky, &k);
        let variance =
            (1.0 + self.noise - v.iter().map(|value| value * value).sum::<f32>()).max(1e-9);

        (mean, variance.sqrt())
    }

    /// Expected improvement over the best observed target.
    pub fn expected_improvement(&self, x: &[f32]) -> f32 {
        let best = self.targets.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !best.is_finite() {
            return 1.0;
        }
        let (mean, sigma) = self.predict(x);
        if sigma <= 1e-9 {
            return (mean - best).max(0.0);
        }
        let z = (mean - best) / sigma;
        let cdf = 0.5 * (1.0 + erf(f64::from(z) / std::f64::consts::SQRT_2) as f32);
        let pdf = (-0.5 * z * z).exp() / (2.0 * std::f32::consts::PI).sqrt();
        (mean - best) * cdf + sigma * pdf
    }

    /// Number of observations the surrogate was fitted on.
    pub fn observations(&self) -> usize {
        self.inputs.len()
    }
}

/// Squared-exponential kernel.
fn rbf(a: &[f32], b: &[f32], length_scale: f32) -> f32 {
    let mut squared_distance = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let difference = x - y;
        squared_distance += difference * difference;
    }
    (-squared_distance / (2.0 * length_scale * length_scale)).exp()
}

/// Cholesky decomposition with a jitter fallback for near-singular matrices.
fn cholesky_decompose(matrix: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = matrix.len();
    let mut l = vec![vec![0.0f32; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                l[i][j] = sum.max(1e-9).sqrt();
            } else {
                l[i][j] = if l[j][j].abs() > 1e-12 { sum / l[j][j] } else { 0.0 };
            }
        }
    }

    l
}

/// Solve `L y = b` for lower-triangular `L`.
fn forward_substitute(l: &[Vec<f32>], b: &[f32]) -> Vec<f32> {
    let n = l.len();
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let mut sum = b.get(i).copied().unwrap_or(0.0);
        for k in 0..i {
            sum -= l[i][k] * y[k];
        }
        y[i] = if l[i][i].abs() > 1e-12 { sum / l[i][i] } else { 0.0 };
    }
    y
}

/// Solve `L Lᵀ x = b`.
fn cholesky_solve(l: &[Vec<f32>], b: &[f32]) -> Vec<f32> {
    let n = l.len();
    let y = forward_substitute(l, b);

    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[k][i] * x[k];
        }
        x[i] = if l[i][i].abs() > 1e-12 { sum / l[i][i] } else { 0.0 };
    }
    x
}

/// Error function (Abramowitz & Stegun 7.1.26), sufficient for the acquisition.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    sign * y
}

/// Objective vector of one evaluation, in the order the objectives were declared.
fn objective_vector(evaluation: &ArchitectureEvaluation, objectives: &[String]) -> Vec<f32> {
    objectives
        .iter()
        .map(|name| evaluation.metrics.get(name).copied().unwrap_or(0.0))
        .collect()
}

/// Does `a` dominate `b`? (All objectives are maximised.)
fn dominates(a: &[f32], b: &[f32]) -> bool {
    let mut strictly_better = false;
    for (x, y) in a.iter().zip(b.iter()) {
        if x < y {
            return false;
        }
        if x > y {
            strictly_better = true;
        }
    }
    strictly_better
}

/// Fast non-dominated sorting (Deb et al., NSGA-II).
///
/// Returns the fronts as index lists, best front first.
pub fn non_dominated_fronts(
    population: &[ArchitectureEvaluation],
    objectives: &[String],
) -> Vec<Vec<usize>> {
    let vectors: Vec<Vec<f32>> = population
        .iter()
        .map(|evaluation| objective_vector(evaluation, objectives))
        .collect();

    let n = population.len();
    let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut domination_count = vec![0usize; n];
    let mut fronts: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();

    for p in 0..n {
        for q in 0..n {
            if p == q {
                continue;
            }
            if dominates(&vectors[p], &vectors[q]) {
                dominated_by[p].push(q);
            } else if dominates(&vectors[q], &vectors[p]) {
                domination_count[p] += 1;
            }
        }
        if domination_count[p] == 0 {
            current.push(p);
        }
    }

    while !current.is_empty() {
        let mut next = Vec::new();
        for &p in &current {
            for &q in &dominated_by[p] {
                domination_count[q] -= 1;
                if domination_count[q] == 0 {
                    next.push(q);
                }
            }
        }
        fronts.push(std::mem::take(&mut current));
        current = next;
    }

    fronts
}

/// Crowding distance of every member of a front (Deb et al., NSGA-II).
pub fn crowding_distances(
    population: &[ArchitectureEvaluation],
    front: &[usize],
    objectives: &[String],
) -> Vec<f32> {
    let mut distances = vec![0.0f32; front.len()];
    if front.len() <= 2 {
        return vec![f32::INFINITY; front.len()];
    }

    for (objective_index, objective) in objectives.iter().enumerate() {
        let mut order: Vec<usize> = (0..front.len()).collect();
        order.sort_by(|a, b| {
            let va = population[front[*a]].metrics.get(objective).copied().unwrap_or(0.0);
            let vb = population[front[*b]].metrics.get(objective).copied().unwrap_or(0.0);
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let first = order[0];
        let last = order[order.len() - 1];
        distances[first] = f32::INFINITY;
        distances[last] = f32::INFINITY;

        let min = population[front[first]].metrics.get(objective).copied().unwrap_or(0.0);
        let max = population[front[last]].metrics.get(objective).copied().unwrap_or(0.0);
        let span = (max - min).abs().max(1e-9);

        for position in 1..order.len() - 1 {
            let index = order[position];
            if distances[index].is_infinite() {
                continue;
            }
            let previous = population[front[order[position - 1]]]
                .metrics
                .get(objective)
                .copied()
                .unwrap_or(0.0);
            let next = population[front[order[position + 1]]]
                .metrics
                .get(objective)
                .copied()
                .unwrap_or(0.0);
            distances[index] += (next - previous).abs() / span;
        }

        let _ = objective_index;
    }

    distances
}
