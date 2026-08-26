use anyhow::Result;
use scirs2_core::ndarray::Array1; // SciRS2 Integration Policy
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Strategies for preventing catastrophic forgetting
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum CatastrophicPreventionStrategy {
    /// Elastic Weight Consolidation
    #[default]
    EWC,
    /// Progressive Neural Networks
    Progressive,
    /// Memory Replay
    MemoryReplay,
    /// Learning without Forgetting (LwF)
    LwF,
    /// Gradient Episodic Memory (GEM)
    GEM,
    /// Average Gradient Episodic Memory (A-GEM)
    AGEM,
    /// Packnet
    PackNet,
    /// Synaptic Intelligence
    SynapticIntelligence,
    /// Meta-Experience Replay (MER)
    MER,
    /// Combined approach using multiple strategies
    Combined(Vec<CatastrophicPreventionStrategy>),
}

/// Regularization methods for catastrophic forgetting prevention
pub trait RegularizationMethod {
    /// Compute regularization penalty for current parameters
    fn compute_penalty(&self, current_params: &HashMap<String, Array1<f32>>) -> f32;

    /// Update method with new task information
    fn update(&mut self, task_id: &str, params: &HashMap<String, Array1<f32>>) -> Result<()>;

    /// Get method name
    fn name(&self) -> &str;

    /// Reset method state
    fn reset(&mut self);
}

/// Elastic Weight Consolidation regularization
#[derive(Debug)]
pub struct EWCRegularization {
    lambda: f32,
    fisher_information: HashMap<String, Array1<f32>>,
    optimal_params: HashMap<String, Array1<f32>>,
}

impl EWCRegularization {
    pub fn new(lambda: f32) -> Self {
        Self {
            lambda,
            fisher_information: HashMap::new(),
            optimal_params: HashMap::new(),
        }
    }
}

impl RegularizationMethod for EWCRegularization {
    fn compute_penalty(&self, current_params: &HashMap<String, Array1<f32>>) -> f32 {
        let mut penalty = 0.0;

        for (param_name, current_param) in current_params {
            if let (Some(fisher), Some(optimal)) = (
                self.fisher_information.get(param_name),
                self.optimal_params.get(param_name),
            ) {
                let diff = current_param - optimal;
                penalty += (fisher * &diff * &diff).sum() * 0.5;
            }
        }

        penalty * self.lambda
    }

    fn update(&mut self, _task_id: &str, params: &HashMap<String, Array1<f32>>) -> Result<()> {
        for (param_name, param_values) in params {
            self.optimal_params.insert(param_name.clone(), param_values.clone());
            // Fisher information would be computed during training
            if !self.fisher_information.contains_key(param_name) {
                self.fisher_information
                    .insert(param_name.clone(), Array1::ones(param_values.len()));
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "EWC"
    }

    fn reset(&mut self) {
        self.fisher_information.clear();
        self.optimal_params.clear();
    }
}

/// Learning without Forgetting (Li & Hoiem, 2016) regularization.
///
/// LwF is an *output-space* method: at a task boundary it snapshots the model's logits on a
/// set of anchor inputs, and while the next task trains it penalises the divergence between
/// the current logits on those same anchors and the snapshot.
///
/// # Contract
///
/// Because [`RegularizationMethod`] is expressed over named `Array1<f32>` vectors, LwF
/// interprets those vectors as **logits on the anchor inputs**, not as parameters:
///
/// * [`RegularizationMethod::update`] stores the supplied vectors as the previous task's
///   soft targets (one entry per anchor key).
/// * [`RegularizationMethod::compute_penalty`] expects the *current* logits under the same
///   keys and returns `alpha · Σ_k T² · KL(p_old‖p_new)`.
///
/// Keys with no stored counterpart are ignored, so before the first task boundary the penalty
/// is legitimately zero — after one, it is not.
#[derive(Debug)]
pub struct LwFRegularization {
    alpha: f32,
    temperature: f32,
    old_outputs: HashMap<String, Array1<f32>>,
    /// Id of the task whose outputs are currently stored.
    previous_task: Option<String>,
}

impl LwFRegularization {
    pub fn new(alpha: f32, temperature: f32) -> Self {
        Self {
            alpha,
            temperature,
            old_outputs: HashMap::new(),
            previous_task: None,
        }
    }

    /// Temperature-scaled softmax, numerically stabilised.
    fn soft_targets(logits: &Array1<f32>, temperature: f32) -> Array1<f32> {
        let t = if temperature.abs() < f32::EPSILON { 1.0 } else { temperature };
        let scaled: Vec<f32> = logits.iter().map(|&x| x / t).collect();
        let max = scaled.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut exps: Vec<f32> = scaled.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        if sum > 0.0 {
            for e in exps.iter_mut() {
                *e /= sum;
            }
        }
        Array1::from_vec(exps)
    }

    /// Temperature-scaled knowledge-distillation divergence between two logit vectors.
    ///
    /// `T² · KL(softmax(old/T) ‖ softmax(new/T))` — the standard Hinton et al. formulation,
    /// with the `T²` factor that keeps the gradient magnitude comparable across temperatures.
    /// The result is `0` exactly when the two distributions coincide and strictly positive
    /// otherwise.
    pub fn distillation_loss(&self, new_outputs: &Array1<f32>, old_outputs: &Array1<f32>) -> f32 {
        if new_outputs.len() != old_outputs.len() || new_outputs.is_empty() {
            return 0.0;
        }
        let p_old = Self::soft_targets(old_outputs, self.temperature);
        let p_new = Self::soft_targets(new_outputs, self.temperature);
        let mut kl = 0.0f32;
        for (&p, &q) in p_old.iter().zip(p_new.iter()) {
            if p > 0.0 {
                kl += p * (p.max(f32::MIN_POSITIVE).ln() - q.max(f32::MIN_POSITIVE).ln());
            }
        }
        let t = if self.temperature.abs() < f32::EPSILON { 1.0 } else { self.temperature };
        kl.max(0.0) * t * t
    }

    /// The task whose outputs are currently held as soft targets.
    pub fn previous_task(&self) -> Option<&str> {
        self.previous_task.as_deref()
    }

    /// Number of stored anchor outputs.
    pub fn num_stored_outputs(&self) -> usize {
        self.old_outputs.len()
    }
}

impl RegularizationMethod for LwFRegularization {
    fn compute_penalty(&self, current_params: &HashMap<String, Array1<f32>>) -> f32 {
        let mut penalty = 0.0f32;
        for (key, current) in current_params {
            if let Some(old) = self.old_outputs.get(key) {
                penalty += self.distillation_loss(current, old);
            }
        }
        penalty * self.alpha
    }

    fn update(&mut self, task_id: &str, params: &HashMap<String, Array1<f32>>) -> Result<()> {
        // Snapshot the finishing task's outputs on the anchor inputs; these become the soft
        // targets that the next task is distilled against.
        self.old_outputs.clear();
        for (key, values) in params {
            self.old_outputs.insert(key.clone(), values.clone());
        }
        self.previous_task = Some(task_id.to_string());
        Ok(())
    }

    fn name(&self) -> &str {
        "LwF"
    }

    fn reset(&mut self) {
        self.old_outputs.clear();
        self.previous_task = None;
    }
}

/// Synaptic Intelligence regularization
#[derive(Debug)]
pub struct SynapticIntelligenceRegularization {
    c: f32,
    xi: f32,
    omega: HashMap<String, Array1<f32>>,
    importance: HashMap<String, Array1<f32>>,
}

impl SynapticIntelligenceRegularization {
    pub fn new(c: f32, xi: f32) -> Self {
        Self {
            c,
            xi,
            omega: HashMap::new(),
            importance: HashMap::new(),
        }
    }

    /// Update importance estimates
    pub fn update_importance(
        &mut self,
        param_name: &str,
        param_change: &Array1<f32>,
        gradient: &Array1<f32>,
    ) {
        let importance_update = param_change * gradient;

        if let Some(current_importance) = self.importance.get_mut(param_name) {
            *current_importance = &*current_importance + &importance_update;
        } else {
            self.importance.insert(param_name.to_string(), importance_update);
        }
    }
}

impl RegularizationMethod for SynapticIntelligenceRegularization {
    fn compute_penalty(&self, current_params: &HashMap<String, Array1<f32>>) -> f32 {
        let mut penalty = 0.0;

        for (param_name, current_param) in current_params {
            if let Some(omega) = self.omega.get(param_name) {
                let param_diff = current_param; // This should be difference from initialization
                penalty += (omega * param_diff * param_diff).sum();
            }
        }

        penalty * self.c
    }

    fn update(&mut self, _task_id: &str, params: &HashMap<String, Array1<f32>>) -> Result<()> {
        // Update omega values based on importance
        for param_name in params.keys() {
            if let Some(importance) = self.importance.get(param_name) {
                let omega_update = importance / (importance.mapv(|x| x * x).sum() + self.xi);

                if let Some(current_omega) = self.omega.get_mut(param_name) {
                    *current_omega = &*current_omega + &omega_update;
                } else {
                    self.omega.insert(param_name.clone(), omega_update);
                }
            }
        }

        // Reset importance for next task
        self.importance.clear();
        Ok(())
    }

    fn name(&self) -> &str {
        "SI"
    }

    fn reset(&mut self) {
        self.omega.clear();
        self.importance.clear();
    }
}

/// Memory-based regularization for GEM/A-GEM
#[derive(Debug)]
pub struct MemoryRegularization {
    memory_size: usize,
    margin: f32,
    episodic_memory: Vec<(Array1<f32>, Array1<f32>)>, // (input, target) pairs
}

impl MemoryRegularization {
    pub fn new(memory_size: usize, margin: f32) -> Self {
        Self {
            memory_size,
            margin,
            episodic_memory: Vec::new(),
        }
    }

    /// Add example to episodic memory
    pub fn add_memory(&mut self, input: Array1<f32>, target: Array1<f32>) {
        if self.episodic_memory.len() >= self.memory_size {
            // Simple replacement strategy - remove oldest
            self.episodic_memory.remove(0);
        }
        self.episodic_memory.push((input, target));
    }

    /// Compute gradient violation constraint
    pub fn compute_gradient_violation(
        &self,
        current_gradient: &Array1<f32>,
        memory_gradients: &[Array1<f32>],
    ) -> f32 {
        let mut max_violation: f32 = 0.0;

        for memory_grad in memory_gradients {
            let dot_product = current_gradient.dot(memory_grad);
            if dot_product < 0.0 {
                max_violation = max_violation.max(-dot_product);
            }
        }

        max_violation
    }
}

impl RegularizationMethod for MemoryRegularization {
    /// Squared-hinge penalty on the distance from the stored episodic targets.
    ///
    /// GEM/A-GEM express their constraint on gradients rather than on parameters, but the
    /// stored `(input, target)` pairs still define a measurable violation: any current vector
    /// that has drifted further than `margin` from the memorised target for the same anchor
    /// contributes `(distance − margin)²`. Anchors are matched positionally against the
    /// sorted keys of `current_params`, so the penalty is deterministic.
    fn compute_penalty(&self, current_params: &HashMap<String, Array1<f32>>) -> f32 {
        if self.episodic_memory.is_empty() || current_params.is_empty() {
            return 0.0;
        }
        let mut keys: Vec<&String> = current_params.keys().collect();
        keys.sort();

        let mut penalty = 0.0f32;
        for (idx, (_input, target)) in self.episodic_memory.iter().enumerate() {
            let Some(key) = keys.get(idx % keys.len()) else {
                continue;
            };
            let Some(current) = current_params.get(*key) else {
                continue;
            };
            if current.len() != target.len() {
                continue;
            }
            let diff = current - target;
            let distance = diff.dot(&diff).sqrt();
            if distance > self.margin {
                let excess = distance - self.margin;
                penalty += excess * excess;
            }
        }
        penalty
    }

    /// Record the task's parameter vectors as episodic memory anchors.
    ///
    /// The vector is stored as both the "input" and the "target" of the anchor: GEM's memory
    /// exists to remember where the model was at the task boundary, which is exactly this
    /// snapshot. Insertion honours `memory_size` through [`MemoryRegularization::add_memory`].
    fn update(&mut self, _task_id: &str, params: &HashMap<String, Array1<f32>>) -> Result<()> {
        let mut keys: Vec<&String> = params.keys().collect();
        keys.sort();
        for key in keys {
            if let Some(values) = params.get(key) {
                self.add_memory(values.clone(), values.clone());
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "Memory"
    }

    fn reset(&mut self) {
        self.episodic_memory.clear();
    }
}

/// Combined regularization using multiple methods
pub struct CombinedRegularization {
    methods: Vec<Box<dyn RegularizationMethod>>,
    weights: Vec<f32>,
}

impl Default for CombinedRegularization {
    fn default() -> Self {
        Self::new()
    }
}

impl CombinedRegularization {
    pub fn new() -> Self {
        Self {
            methods: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// Add regularization method with weight
    pub fn add_method(&mut self, method: Box<dyn RegularizationMethod>, weight: f32) {
        self.methods.push(method);
        self.weights.push(weight);
    }
}

impl RegularizationMethod for CombinedRegularization {
    fn compute_penalty(&self, current_params: &HashMap<String, Array1<f32>>) -> f32 {
        self.methods
            .iter()
            .zip(&self.weights)
            .map(|(method, &weight)| weight * method.compute_penalty(current_params))
            .sum()
    }

    fn update(&mut self, task_id: &str, params: &HashMap<String, Array1<f32>>) -> Result<()> {
        for method in &mut self.methods {
            method.update(task_id, params)?;
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "Combined"
    }

    fn reset(&mut self) {
        for method in &mut self.methods {
            method.reset();
        }
    }
}

/// Factory for creating regularization methods
pub struct RegularizationFactory;

impl RegularizationFactory {
    /// Create regularization method from strategy
    pub fn create_method(
        strategy: &CatastrophicPreventionStrategy,
    ) -> Box<dyn RegularizationMethod> {
        match strategy {
            CatastrophicPreventionStrategy::EWC => Box::new(EWCRegularization::new(0.4)),
            CatastrophicPreventionStrategy::LwF => Box::new(LwFRegularization::new(1.0, 4.0)),
            CatastrophicPreventionStrategy::SynapticIntelligence => {
                Box::new(SynapticIntelligenceRegularization::new(0.1, 0.1))
            },
            CatastrophicPreventionStrategy::GEM | CatastrophicPreventionStrategy::AGEM => {
                Box::new(MemoryRegularization::new(1000, 0.5))
            },
            CatastrophicPreventionStrategy::Combined(strategies) => {
                let mut combined = CombinedRegularization::new();
                for strategy in strategies {
                    combined
                        .add_method(Self::create_method(strategy), 1.0 / strategies.len() as f32);
                }
                Box::new(combined)
            },
            _ => {
                // Default to EWC for other strategies
                Box::new(EWCRegularization::new(0.4))
            },
        }
    }
}

/// Configuration for different prevention strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreventionConfig {
    pub strategy: CatastrophicPreventionStrategy,
    pub ewc_lambda: f32,
    pub lwf_alpha: f32,
    pub lwf_temperature: f32,
    pub si_c: f32,
    pub si_xi: f32,
    pub memory_size: usize,
    pub gem_margin: f32,
}

impl Default for PreventionConfig {
    fn default() -> Self {
        Self {
            strategy: CatastrophicPreventionStrategy::EWC,
            ewc_lambda: 0.4,
            lwf_alpha: 1.0,
            lwf_temperature: 4.0,
            si_c: 0.1,
            si_xi: 0.1,
            memory_size: 1000,
            gem_margin: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_ewc_regularization() {
        let mut ewc = EWCRegularization::new(0.4);

        let mut params = HashMap::new();
        params.insert("weight1".to_string(), Array1::from_vec(vec![1.0, 2.0, 3.0]));

        ewc.update("task1", &params).expect("operation failed in test");

        let mut current_params = HashMap::new();
        current_params.insert("weight1".to_string(), Array1::from_vec(vec![1.1, 2.1, 3.1]));

        let penalty = ewc.compute_penalty(&current_params);
        assert!(penalty > 0.0);
    }

    #[test]
    fn test_regularization_factory() {
        let method = RegularizationFactory::create_method(&CatastrophicPreventionStrategy::EWC);
        assert_eq!(method.name(), "EWC");

        let lwf_method = RegularizationFactory::create_method(&CatastrophicPreventionStrategy::LwF);
        assert_eq!(lwf_method.name(), "LwF");
    }

    #[test]
    fn test_combined_regularization() {
        let mut combined = CombinedRegularization::new();
        combined.add_method(Box::new(EWCRegularization::new(0.4)), 0.5);
        combined.add_method(Box::new(LwFRegularization::new(1.0, 4.0)), 0.5);

        assert_eq!(combined.name(), "Combined");

        let mut params = HashMap::new();
        params.insert("weight1".to_string(), Array1::from_vec(vec![1.0, 2.0]));

        combined.update("task1", &params).expect("operation failed in test");
        let penalty = combined.compute_penalty(&params);
        assert!(penalty >= 0.0);
    }

    // ── LwF regression tests ─────────────────────────────────────────────────

    #[test]
    fn test_lwf_penalty_is_zero_before_a_task_boundary_and_nonzero_after() {
        // Regression: `compute_penalty` used to return a hardcoded 0.0 and `update` stored
        // nothing, so selecting LwF applied no regularization whatsoever.
        let mut lwf = LwFRegularization::new(1.0, 2.0);

        let mut old_outputs = HashMap::new();
        old_outputs.insert("head".to_string(), Array1::from_vec(vec![3.0, 0.0, 0.0]));
        let mut new_outputs = HashMap::new();
        new_outputs.insert("head".to_string(), Array1::from_vec(vec![0.0, 3.0, 0.0]));

        assert_eq!(
            lwf.compute_penalty(&new_outputs),
            0.0,
            "with nothing stored the penalty must be exactly 0"
        );

        lwf.update("task_a", &old_outputs).expect("update failed");
        assert_eq!(lwf.previous_task(), Some("task_a"));
        assert_eq!(lwf.num_stored_outputs(), 1);

        let penalty = lwf.compute_penalty(&new_outputs);
        assert!(
            penalty > 0.0,
            "after a task switch a diverged head must be penalised, got {penalty}"
        );
    }

    #[test]
    fn test_lwf_penalty_vanishes_when_outputs_are_unchanged() {
        let mut lwf = LwFRegularization::new(1.0, 2.0);
        let mut outputs = HashMap::new();
        outputs.insert("head".to_string(), Array1::from_vec(vec![1.0, 2.0, 3.0]));
        lwf.update("task_a", &outputs).expect("update failed");
        let penalty = lwf.compute_penalty(&outputs);
        assert!(
            penalty.abs() < 1e-5,
            "identical outputs must give a zero KD penalty, got {penalty}"
        );
    }

    #[test]
    fn test_lwf_distillation_loss_matches_hand_computed_kl() {
        // T = 1, old logits [0, ln 3] -> p = [0.25, 0.75]; new logits [0, 0] -> q = [0.5, 0.5].
        // KL = 0.25*ln(0.25/0.5) + 0.75*ln(0.75/0.5)
        let lwf = LwFRegularization::new(1.0, 1.0);
        let old = Array1::from_vec(vec![0.0, 3.0f32.ln()]);
        let new = Array1::from_vec(vec![0.0, 0.0]);
        let expected = 0.25f32 * (0.25f32 / 0.5).ln() + 0.75f32 * (0.75f32 / 0.5).ln();
        let got = lwf.distillation_loss(&new, &old);
        assert!(
            (got - expected).abs() < 1e-5,
            "expected {expected}, got {got}"
        );
    }

    #[test]
    fn test_lwf_penalty_scales_with_alpha() {
        let mut outputs_old = HashMap::new();
        outputs_old.insert("head".to_string(), Array1::from_vec(vec![2.0, 0.0]));
        let mut outputs_new = HashMap::new();
        outputs_new.insert("head".to_string(), Array1::from_vec(vec![0.0, 2.0]));

        let mut a = LwFRegularization::new(1.0, 2.0);
        a.update("t", &outputs_old).expect("update");
        let mut b = LwFRegularization::new(3.0, 2.0);
        b.update("t", &outputs_old).expect("update");

        let pa = a.compute_penalty(&outputs_new);
        let pb = b.compute_penalty(&outputs_new);
        assert!((pb - 3.0 * pa).abs() < 1e-4, "alpha must scale the penalty");
    }

    #[test]
    fn test_lwf_reset_clears_the_stored_outputs() {
        let mut lwf = LwFRegularization::new(1.0, 2.0);
        let mut outputs = HashMap::new();
        outputs.insert("head".to_string(), Array1::from_vec(vec![1.0, 0.0]));
        lwf.update("t", &outputs).expect("update");
        lwf.reset();
        assert_eq!(lwf.num_stored_outputs(), 0);
        assert!(lwf.previous_task().is_none());
    }

    #[test]
    fn test_memory_regularization_update_records_anchors_and_penalises_drift() {
        // Regression: `update` was `Ok(())`, so the episodic memory stayed empty forever.
        let mut memory_reg = MemoryRegularization::new(10, 0.5);
        let mut params = HashMap::new();
        params.insert("w".to_string(), Array1::from_vec(vec![0.0, 0.0]));
        memory_reg.update("task_a", &params).expect("update failed");
        assert_eq!(memory_reg.episodic_memory.len(), 1);

        // Unchanged parameters: distance 0 < margin => no penalty.
        assert_eq!(memory_reg.compute_penalty(&params), 0.0);

        // Drift well beyond the margin => positive penalty.
        let mut drifted = HashMap::new();
        drifted.insert("w".to_string(), Array1::from_vec(vec![3.0, 4.0])); // distance 5
        let penalty = memory_reg.compute_penalty(&drifted);
        let expected = (5.0f32 - 0.5).powi(2);
        assert!(
            (penalty - expected).abs() < 1e-4,
            "expected {expected}, got {penalty}"
        );
    }

    #[test]
    fn test_memory_regularization() {
        let mut memory_reg = MemoryRegularization::new(10, 0.5);

        let input = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let target = Array1::from_vec(vec![0.0, 1.0]);

        memory_reg.add_memory(input, target);
        assert_eq!(memory_reg.episodic_memory.len(), 1);

        // Test gradient violation
        let current_grad = Array1::from_vec(vec![1.0, -1.0]);
        let memory_grads = vec![Array1::from_vec(vec![-1.0, 1.0])];

        let violation = memory_reg.compute_gradient_violation(&current_grad, &memory_grads);
        assert!(violation > 0.0);
    }

    #[test]
    fn test_synaptic_intelligence() {
        let mut si = SynapticIntelligenceRegularization::new(0.1, 0.1);

        let param_change = Array1::from_vec(vec![0.1, 0.2]);
        let gradient = Array1::from_vec(vec![1.0, 0.5]);

        si.update_importance("weight1", &param_change, &gradient);

        let mut params = HashMap::new();
        params.insert("weight1".to_string(), Array1::from_vec(vec![1.0, 2.0]));

        si.update("task1", &params).expect("operation failed in test");

        let penalty = si.compute_penalty(&params);
        assert!(penalty >= 0.0);
    }
}
