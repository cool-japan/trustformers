//! # AdaFisher: Adaptive Second Order Optimization via Fisher Information (Simplified)
//!
//! This is a simplified implementation of AdaFisher that uses basic tensor operations
//! available in the TrustformeRS core. The full implementation would require more
//! advanced tensor operations for true Fisher information computation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::{errors::Result, tensor::Tensor, traits::Optimizer};

use crate::{
    common::StateMemoryStats,
    param_id::{ParamId, ParamRegistry},
    traits::StatefulOptimizer,
};

/// Configuration for AdaFisher optimizer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AdaFisherConfig {
    pub learning_rate: f32,
    pub fisher_decay: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
}

impl Default for AdaFisherConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            fisher_decay: 0.95,
            epsilon: 1e-6,
            weight_decay: 0.01,
        }
    }
}

/// Simplified AdaFisher optimizer state.
#[derive(Clone, Debug)]
pub struct AdaFisherState {
    pub momentum: Tensor,
    pub variance: Tensor,
    pub step: usize,
}

/// Simplified AdaFisher optimizer.
///
/// # Parameter identity
///
/// Per-parameter Fisher state is keyed through [`ParamRegistry`], so the same
/// parameter resolves to the same state entry on every step and across a
/// checkpoint save/load. See [`crate::param_id`] for the identity contract.
#[derive(Clone, Debug)]
pub struct AdaFisher {
    config: AdaFisherConfig,
    states: HashMap<String, AdaFisherState>,
    /// Stable parameter identity. Never serialised: it is rebuilt from the
    /// checkpointed state keys on load.
    params: ParamRegistry,
    step: usize,
    memory_stats: StateMemoryStats,
}

impl AdaFisher {
    pub fn new(learning_rate: f32, fisher_decay: f32, epsilon: f32, _block_size: usize) -> Self {
        Self {
            config: AdaFisherConfig {
                learning_rate,
                fisher_decay,
                epsilon,
                weight_decay: 0.01,
            },
            states: HashMap::new(),
            params: ParamRegistry::new(),
            step: 0,
            memory_stats: StateMemoryStats {
                momentum_elements: 0,
                variance_elements: 0,
                third_moment_elements: 0,
                total_bytes: 0,
                num_parameters: 0,
            },
        }
    }

    pub fn for_language_models() -> Self {
        Self::new(3e-4, 0.99, 1e-8, 128)
    }

    pub fn for_image_classification() -> Self {
        Self::new(1e-3, 0.95, 1e-6, 64)
    }

    /// Per-parameter Fisher statistics: `(element count, local step, mean Fisher diagonal)`.
    ///
    /// The mean Fisher diagonal is the average of the accumulated second-moment
    /// (curvature) buffer — a measured value, not an estimate.
    ///
    /// # Errors
    ///
    /// Returns an error when a stored variance buffer cannot be read as `f32`.
    pub fn fisher_stats(&self) -> Result<HashMap<String, (usize, usize, f32)>> {
        let mut stats = HashMap::new();
        for (key, state) in &self.states {
            let variance = state.variance.data_f32()?;
            let numel = variance.len();
            let mean_fisher =
                if numel == 0 { 0.0 } else { variance.iter().sum::<f32>() / numel as f32 };
            stats.insert(key.clone(), (numel, state.step, mean_fisher));
        }
        Ok(stats)
    }

    /// Bytes actually held by the momentum and Fisher-diagonal buffers.
    pub fn fisher_memory_usage(&self) -> usize {
        self.states
            .values()
            .map(|state| state.momentum.size_bytes() + state.variance.size_bytes())
            .sum()
    }

    /// Updates one parameter that carries a stable caller-supplied name.
    ///
    /// Prefer this over [`Optimizer::update`] whenever the surrounding API knows the
    /// parameter's name: named state keys make checkpoint resume independent of the
    /// order in which parameters are visited.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor dtype has no addressable buffer or when a
    /// tensor operation fails.
    pub fn update_named(
        &mut self,
        name: &str,
        parameter: &mut Tensor,
        gradient: &Tensor,
    ) -> Result<()> {
        let id = self.params.id_for_named_tensor(name, parameter)?;
        self.apply_update(id, parameter, gradient)
    }

    /// Shared update body: both the named and the anonymous path funnel through here.
    fn apply_update(
        &mut self,
        id: ParamId,
        parameter: &mut Tensor,
        gradient: &Tensor,
    ) -> Result<()> {
        let key = self
            .params
            .key(id)
            .map(str::to_string)
            .unwrap_or_else(|| format!("p:{}", id.index()));

        let state = match self.states.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => entry.insert(AdaFisherState {
                momentum: Tensor::zeros_like(parameter)?,
                variance: Tensor::zeros_like(parameter)?,
                step: 0,
            }),
        };

        state.step += 1;

        // Adam-like update with Fisher information approximation
        state.momentum = state
            .momentum
            .mul_scalar(self.config.fisher_decay)?
            .add(&gradient.mul_scalar(1.0 - self.config.fisher_decay)?)?;

        state.variance = state
            .variance
            .mul_scalar(self.config.fisher_decay)?
            .add(&gradient.pow_scalar(2.0)?.mul_scalar(1.0 - self.config.fisher_decay)?)?;

        // Bias correction
        let bias_correction1 = 1.0 - self.config.fisher_decay.powi(state.step as i32);
        let bias_correction2 = 1.0 - self.config.fisher_decay.powi(state.step as i32);

        let corrected_momentum = state.momentum.div_scalar(bias_correction1)?;
        let corrected_variance = state.variance.div_scalar(bias_correction2)?;

        // Update parameter
        let denominator = corrected_variance.sqrt()?.add_scalar(self.config.epsilon)?;
        let update = corrected_momentum.div(&denominator)?.mul_scalar(self.config.learning_rate)?;

        *parameter = parameter.sub(&update)?;

        // `sub` allocates a fresh buffer, so the registry's address cache must follow
        // the parameter to its new home.
        self.params.rebind(id, parameter)?;
        self.refresh_memory_stats();

        Ok(())
    }

    /// Recomputes the reported memory statistics from the live buffers.
    fn refresh_memory_stats(&mut self) {
        let momentum_elements: usize = self.states.values().map(|s| s.momentum.len()).sum();
        let variance_elements: usize = self.states.values().map(|s| s.variance.len()).sum();
        self.memory_stats = StateMemoryStats {
            momentum_elements,
            variance_elements,
            third_moment_elements: 0,
            total_bytes: self.fisher_memory_usage(),
            num_parameters: self.states.len(),
        };
    }
}

impl Optimizer for AdaFisher {
    fn update(&mut self, parameter: &mut Tensor, gradient: &Tensor) -> Result<()> {
        let id = self.params.id_for_tensor(parameter)?;
        self.apply_update(id, parameter, gradient)
    }

    fn zero_grad(&mut self) {
        // Nothing to clear for AdaFisher
    }

    fn step(&mut self) {
        self.step += 1;
    }

    fn get_lr(&self) -> f32 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }
}

impl StatefulOptimizer for AdaFisher {
    type Config = AdaFisherConfig;
    type State = StateMemoryStats;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn state(&self) -> &Self::State {
        &self.memory_stats
    }

    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.memory_stats
    }

    /// Serialises the full per-parameter state under the stable identity keys.
    ///
    /// Entries are `momentum:<key>`, `variance:<key>` and `pstep:<key>`, where `<key>`
    /// is the canonical [`ParamRegistry`] key (`n:<name>` or `p:<index>`). Because the
    /// identity is embedded in the entry name, [`Self::load_state_dict`] can rebuild
    /// the registry from the checkpoint alone.
    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut state_dict = HashMap::new();
        state_dict.insert("step".to_string(), Tensor::scalar(self.step as f32)?);
        for (key, state) in &self.states {
            state_dict.insert(format!("momentum:{key}"), state.momentum.clone());
            state_dict.insert(format!("variance:{key}"), state.variance.clone());
            state_dict.insert(format!("pstep:{key}"), Tensor::scalar(state.step as f32)?);
        }
        Ok(state_dict)
    }

    /// Restores the per-parameter state written by [`Self::state_dict`].
    ///
    /// # Errors
    ///
    /// Returns an error when a checkpoint entry carries an unrecognised identity
    /// prefix or a momentum entry has no matching variance entry.
    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        self.states.clear();
        self.params.clear();

        if let Some(step_tensor) = state.get("step") {
            self.step = step_tensor.to_scalar()? as usize;
        }

        // Rebuild in key order so anonymous slots keep their registration indices.
        let mut keys: Vec<&str> =
            state.keys().filter_map(|k| k.strip_prefix("momentum:")).collect();
        keys.sort_unstable();

        for key in keys {
            let momentum = state.get(&format!("momentum:{key}")).ok_or_else(|| {
                trustformers_core::errors::TrustformersError::invalid_input(format!(
                    "AdaFisher checkpoint is missing momentum for '{key}'"
                ))
            })?;
            let variance = state.get(&format!("variance:{key}")).ok_or_else(|| {
                trustformers_core::errors::TrustformersError::invalid_input(format!(
                    "AdaFisher checkpoint is missing variance for '{key}'"
                ))
            })?;
            let local_step = match state.get(&format!("pstep:{key}")) {
                Some(tensor) => tensor.to_scalar()? as usize,
                None => 0,
            };

            self.params.restore_key(key, momentum.len())?;
            self.states.insert(
                key.to_string(),
                AdaFisherState {
                    momentum: momentum.clone(),
                    variance: variance.clone(),
                    step: local_step,
                },
            );
        }

        self.refresh_memory_stats();
        Ok(())
    }

    fn memory_usage(&self) -> StateMemoryStats {
        self.memory_stats.clone()
    }

    fn reset_state(&mut self) {
        self.states.clear();
        self.params.clear();
        self.step = 0;
        self.refresh_memory_stats();
    }

    fn num_parameters(&self) -> usize {
        self.states.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(values: &[f32]) -> Tensor {
        Tensor::from_vec(values.to_vec(), &[values.len()]).expect("tensor")
    }

    /// Regression: state used to be keyed on `states.len()` plus a hash of the
    /// parameter's *current values*, so every call inserted a brand-new entry and the
    /// optimizer was stuck on step 1 forever while the map grew without bound.
    #[test]
    fn repeated_updates_reuse_one_state_entry() {
        let mut optimizer = AdaFisher::new(1e-2, 0.9, 1e-8, 64);
        let mut param = tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let grad = tensor(&[0.5; 6]);

        for _ in 0..5 {
            optimizer.update(&mut param, &grad).expect("update");
        }

        assert_eq!(
            optimizer.states.len(),
            1,
            "state map must not grow per step"
        );
        let step = optimizer.states.values().next().map(|s| s.step).expect("state");
        assert_eq!(step, 5, "the local step counter must advance");
    }

    /// Two distinct parameters must not share one state entry.
    #[test]
    fn distinct_parameters_get_distinct_state() {
        let mut optimizer = AdaFisher::new(1e-2, 0.9, 1e-8, 64);
        let mut first = tensor(&[1.0, 2.0]);
        let mut second = tensor(&[3.0, 4.0, 5.0]);
        let grad_first = tensor(&[1.0, 1.0]);
        let grad_second = tensor(&[1.0, 1.0, 1.0]);

        optimizer.update(&mut first, &grad_first).expect("first");
        optimizer.update(&mut second, &grad_second).expect("second");
        optimizer.update(&mut first, &grad_first).expect("first again");

        assert_eq!(optimizer.states.len(), 2);
    }

    /// A zero gradient after a non-zero one must still move the parameter, because
    /// the accumulated momentum carries over. With the old value-hash keying every
    /// step started from freshly zeroed moments, so the second step was a no-op.
    #[test]
    fn state_is_carried_between_steps() {
        let mut optimizer = AdaFisher::new(1e-2, 0.9, 1e-8, 64);
        let mut param = tensor(&[1.0]);

        let before = param.data_f32().expect("data")[0];
        optimizer.update(&mut param, &tensor(&[1.0])).expect("step 1");
        let after_one = param.data_f32().expect("data")[0];
        assert!(before - after_one > 0.0, "the parameter must move");

        optimizer.update(&mut param, &tensor(&[0.0])).expect("step 2");
        let after_two = param.data_f32().expect("data")[0];
        assert!(
            after_one - after_two > 1e-6,
            "momentum from step 1 must still drive step 2: {after_one} -> {after_two}"
        );
    }

    /// Named identity must survive a checkpoint round trip, unlike a heap address.
    #[test]
    fn state_survives_a_save_load_round_trip() {
        let mut optimizer = AdaFisher::new(1e-2, 0.9, 1e-8, 64);
        let mut param = tensor(&[1.0, 2.0]);
        let grad = tensor(&[0.5, -0.5]);

        optimizer.update_named("w", &mut param, &grad).expect("step 1");
        optimizer.update_named("w", &mut param, &grad).expect("step 2");
        let checkpoint = optimizer.state_dict().expect("state_dict");
        let reference = param.data_f32().expect("data");

        // A fresh optimizer in a fresh "process": different buffers, different addresses.
        let mut resumed = AdaFisher::new(1e-2, 0.9, 1e-8, 64);
        resumed.load_state_dict(checkpoint).expect("load_state_dict");
        assert_eq!(
            resumed.states.len(),
            1,
            "state must be restored, not dropped"
        );

        let mut resumed_param = tensor(&reference);
        resumed.update_named("w", &mut resumed_param, &grad).expect("step 3");

        // The same trajectory continued in the original optimizer must agree.
        let mut continued = param.clone();
        optimizer.update_named("w", &mut continued, &grad).expect("step 3 reference");

        let restored = resumed_param.data_f32().expect("data");
        let expected = continued.data_f32().expect("data");
        for (a, b) in restored.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "resume diverged from the uninterrupted run: {a} vs {b}"
            );
        }
    }

    /// Convergence smoke test on the quadratic bowl `f(x) = Σ x²` (`∇f = 2x`).
    #[test]
    fn descends_a_quadratic_bowl() {
        let mut optimizer = AdaFisher::new(5e-2, 0.9, 1e-8, 64);
        let mut param = tensor(&[3.0, -4.0]);
        let initial: f32 = param.data_f32().expect("data").iter().map(|v| v * v).sum();

        for _ in 0..300 {
            let values = param.data_f32().expect("data");
            let grad = tensor(&values.iter().map(|v| 2.0 * v).collect::<Vec<_>>());
            optimizer.update(&mut param, &grad).expect("step");
        }

        let final_loss: f32 = param.data_f32().expect("data").iter().map(|v| v * v).sum();
        assert!(
            final_loss < initial * 0.1,
            "loss must fall: {initial} -> {final_loss}"
        );
    }

    /// Fisher statistics must be measured, not the old hardcoded `64.0`.
    #[test]
    fn fisher_stats_reflect_the_accumulated_curvature() {
        let mut optimizer = AdaFisher::new(1e-2, 0.9, 1e-8, 64);
        let mut param = tensor(&[1.0, 1.0]);
        let small = tensor(&[0.1, 0.1]);
        optimizer.update_named("small", &mut param, &small).expect("small");

        let mut other = tensor(&[1.0, 1.0]);
        let large = tensor(&[10.0, 10.0]);
        optimizer.update_named("large", &mut other, &large).expect("large");

        let stats = optimizer.fisher_stats().expect("stats");
        let small_fisher = stats.get("n:small").map(|s| s.2).expect("small entry");
        let large_fisher = stats.get("n:large").map(|s| s.2).expect("large entry");
        assert!(
            large_fisher > small_fisher * 100.0,
            "Fisher diagonal must track gradient magnitude: {small_fisher} vs {large_fisher}"
        );
        assert!(optimizer.fisher_memory_usage() > 0);
    }
}
