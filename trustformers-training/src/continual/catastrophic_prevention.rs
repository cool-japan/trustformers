use anyhow::Result;
use scirs2_core::ndarray::Array1; // SciRS2 Integration Policy
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Strategies for preventing catastrophic forgetting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CatastrophicPreventionStrategy {
    /// Elastic Weight Consolidation
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

impl Default for CatastrophicPreventionStrategy {
    fn default() -> Self {
        Self::EWC
    }
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

/// Learning without Forgetting regularization
#[derive(Debug)]
#[allow(dead_code)]
pub struct LwFRegularization {
    #[allow(dead_code)]
    alpha: f32,
    temperature: f32,
    old_outputs: HashMap<String, Array1<f32>>,
}

impl LwFRegularization {
    pub fn new(alpha: f32, temperature: f32) -> Self {
        Self {
            alpha,
            temperature,
            old_outputs: HashMap::new(),
        }
    }

    /// Compute knowledge distillation loss
    pub fn distillation_loss(&self, new_outputs: &Array1<f32>, old_outputs: &Array1<f32>) -> f32 {
        // Simplified distillation loss computation
        let diff = new_outputs - old_outputs;
        (&diff * &diff).sum() / new_outputs.len() as f32
    }
}

impl RegularizationMethod for LwFRegularization {
    fn compute_penalty(&self, _current_params: &HashMap<String, Array1<f32>>) -> f32 {
        // LwF penalty is computed differently - this is a placeholder
        0.0
    }

    fn update(&mut self, _task_id: &str, _params: &HashMap<String, Array1<f32>>) -> Result<()> {
        // Store outputs from previous task for distillation
        Ok(())
    }

    fn name(&self) -> &str {
        "LwF"
    }

    fn reset(&mut self) {
        self.old_outputs.clear();
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
    #[allow(dead_code)]
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
    fn compute_penalty(&self, _current_params: &HashMap<String, Array1<f32>>) -> f32 {
        // Memory-based methods use constraints rather than penalties
        0.0
    }

    fn update(&mut self, _task_id: &str, _params: &HashMap<String, Array1<f32>>) -> Result<()> {
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

        ewc.update("task1", &params).unwrap();

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

        combined.update("task1", &params).unwrap();
        let penalty = combined.compute_penalty(&params);
        assert!(penalty >= 0.0);
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

        si.update("task1", &params).unwrap();

        let penalty = si.compute_penalty(&params);
        assert!(penalty >= 0.0);
    }
}
