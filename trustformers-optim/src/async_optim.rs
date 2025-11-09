//! # Asynchronous Optimization Methods
//!
//! This module implements asynchronous optimization algorithms for distributed
//! training where workers can update parameters without strict synchronization.
//!
//! ## Available Methods
//!
//! - **Async SGD**: Asynchronous stochastic gradient descent
//! - **Hogwild!**: Lock-free asynchronous SGD for sparse features
//! - **Delayed Gradient**: Methods that handle stale gradients
//! - **Elastic Averaging SGD**: Combines local and global parameter averaging

use anyhow::Result;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use trustformers_core::tensor::Tensor;

/// Configuration for asynchronous SGD.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncSGDConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Momentum coefficient
    pub momentum: f32,
    /// Weight decay
    pub weight_decay: f32,
    /// Maximum allowed staleness for gradient updates
    pub max_staleness: usize,
    /// Staleness adaptive factor
    pub staleness_factor: f32,
}

impl Default for AsyncSGDConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            momentum: 0.9,
            weight_decay: 0.0,
            max_staleness: 10,
            staleness_factor: 0.9,
        }
    }
}

/// Configuration for Hogwild! optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HogwildConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Sparse update ratio (fraction of parameters updated per step)
    pub sparse_ratio: f32,
    /// Maximum number of concurrent workers
    pub max_workers: usize,
}

impl Default for HogwildConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            sparse_ratio: 0.1,
            max_workers: 4,
        }
    }
}

/// Configuration for delayed gradient methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelayedGradientConfig {
    /// Base learning rate
    pub learning_rate: f32,
    /// Maximum gradient delay (in steps)
    pub max_delay: usize,
    /// Delay compensation method
    pub compensation_method: DelayCompensationMethod,
    /// Compensation factor
    pub compensation_factor: f32,
}

impl Default for DelayedGradientConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            max_delay: 20,
            compensation_method: DelayCompensationMethod::LinearDecay,
            compensation_factor: 0.5,
        }
    }
}

/// Methods for compensating gradient delays.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DelayCompensationMethod {
    /// No compensation
    None,
    /// Linear decay based on delay
    LinearDecay,
    /// Exponential decay based on delay
    ExponentialDecay,
    /// Adaptive compensation
    Adaptive,
}

/// Configuration for Elastic Averaging SGD.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElasticAveragingConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Elastic force coefficient
    pub alpha: f32,
    /// Communication period (steps between synchronization)
    pub tau: usize,
    /// Beta parameter for moving average
    pub beta: f32,
}

impl Default for ElasticAveragingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            alpha: 0.6,
            tau: 10,
            beta: 0.9,
        }
    }
}

/// Shared parameter server for asynchronous optimization.
pub struct ParameterServer {
    /// Global parameters
    parameters: Arc<RwLock<Vec<Tensor>>>,
    /// Global step counter
    global_step: AtomicUsize,
    /// Parameter version counters
    version_counters: Arc<Mutex<Vec<usize>>>,
    /// Worker update timestamps
    worker_timestamps: Arc<Mutex<HashMap<usize, Instant>>>,
}

impl ParameterServer {
    /// Create a new parameter server.
    pub fn new(initial_parameters: Vec<Tensor>) -> Self {
        let param_count = initial_parameters.len();
        Self {
            parameters: Arc::new(RwLock::new(initial_parameters)),
            global_step: AtomicUsize::new(0),
            version_counters: Arc::new(Mutex::new(vec![0; param_count])),
            worker_timestamps: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get current parameters for a worker.
    pub fn get_parameters(&self, worker_id: usize) -> Result<(Vec<Tensor>, Vec<usize>)> {
        let params = self.parameters.read().clone();
        let versions = self.version_counters.lock().clone();

        // Update worker timestamp
        let mut timestamps = self.worker_timestamps.lock();
        timestamps.insert(worker_id, Instant::now());

        Ok((params, versions))
    }

    /// Update parameters with gradients from a worker.
    pub fn update_parameters(
        &self,
        worker_id: usize,
        gradients: Vec<Tensor>,
        param_versions: Vec<usize>,
        learning_rate: f32,
    ) -> Result<()> {
        let _current_step = self.global_step.load(Ordering::SeqCst);

        // Check staleness
        let staleness = self.compute_staleness(worker_id, &param_versions)?;
        if staleness > 10 {
            // Skip very stale updates
            return Ok(());
        }

        // Apply staleness compensation
        let compensated_lr = learning_rate * (1.0 / (1.0 + staleness as f32 * 0.1));

        // Update parameters
        {
            let mut params = self.parameters.write();
            let mut versions = self.version_counters.lock();

            for (i, gradient) in gradients.iter().enumerate() {
                if i < params.len() {
                    let update = gradient.mul_scalar(compensated_lr)?;
                    params[i] = params[i].sub(&update)?;
                    versions[i] += 1;
                }
            }
        }

        self.global_step.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn compute_staleness(&self, _worker_id: usize, param_versions: &[usize]) -> Result<usize> {
        let current_versions = self.version_counters.lock();
        let max_staleness = param_versions
            .iter()
            .zip(current_versions.iter())
            .map(|(old, new)| new.saturating_sub(*old))
            .max()
            .unwrap_or(0);
        Ok(max_staleness)
    }

    /// Get current global step.
    pub fn get_global_step(&self) -> usize {
        self.global_step.load(Ordering::SeqCst)
    }
}

/// Asynchronous SGD optimizer.
pub struct AsyncSGD {
    config: AsyncSGDConfig,
    worker_id: usize,
    parameter_server: Arc<ParameterServer>,
    momentum_buffers: Vec<Tensor>,
    local_parameters: Vec<Tensor>,
    param_versions: Vec<usize>,
    last_sync_step: usize,
}

impl AsyncSGD {
    /// Create a new async SGD optimizer.
    pub fn new(
        config: AsyncSGDConfig,
        worker_id: usize,
        parameter_server: Arc<ParameterServer>,
    ) -> Result<Self> {
        let (params, versions) = parameter_server.get_parameters(worker_id)?;
        let param_count = params.len();

        Ok(Self {
            config,
            worker_id,
            parameter_server,
            momentum_buffers: (0..param_count)
                .map(|i| Tensor::zeros(&params[i].shape()).map_err(anyhow::Error::from))
                .collect::<Result<Vec<_>>>()?,
            local_parameters: params,
            param_versions: versions,
            last_sync_step: 0,
        })
    }

    /// Perform an optimization step.
    pub fn step(&mut self, gradients: &[Tensor]) -> Result<()> {
        // Check if we need to sync with parameter server
        let current_step = self.parameter_server.get_global_step();
        let staleness = current_step - self.last_sync_step;

        if staleness > self.config.max_staleness {
            self.sync_with_server()?;
        }

        // Apply momentum and update local parameters
        for (i, gradient) in gradients.iter().enumerate() {
            if i < self.local_parameters.len() {
                // Apply weight decay
                let effective_grad = if self.config.weight_decay > 0.0 {
                    gradient.add(&self.local_parameters[i].mul_scalar(self.config.weight_decay)?)?
                } else {
                    gradient.clone()
                };

                // Update momentum
                self.momentum_buffers[i] = self.momentum_buffers[i]
                    .mul_scalar(self.config.momentum)?
                    .add(&effective_grad)?;

                // Apply staleness compensation
                let staleness_factor = self.config.staleness_factor.powi(staleness as i32);
                let compensated_lr = self.config.learning_rate * staleness_factor;

                // Update local parameters
                let update = self.momentum_buffers[i].mul_scalar(compensated_lr)?;
                self.local_parameters[i] = self.local_parameters[i].sub(&update)?;
            }
        }

        // Send updates to parameter server periodically
        if current_step % 5 == 0 {
            self.push_to_server(gradients)?;
        }

        Ok(())
    }

    fn sync_with_server(&mut self) -> Result<()> {
        let (params, versions) = self.parameter_server.get_parameters(self.worker_id)?;
        self.local_parameters = params;
        self.param_versions = versions;
        self.last_sync_step = self.parameter_server.get_global_step();
        Ok(())
    }

    fn push_to_server(&self, gradients: &[Tensor]) -> Result<()> {
        self.parameter_server.update_parameters(
            self.worker_id,
            gradients.to_vec(),
            self.param_versions.clone(),
            self.config.learning_rate,
        )
    }

    /// Get current local parameters.
    pub fn get_parameters(&self) -> &[Tensor] {
        &self.local_parameters
    }
}

/// Hogwild! optimizer for sparse features.
pub struct Hogwild {
    config: HogwildConfig,
    #[allow(dead_code)]
    worker_id: usize,
    shared_parameters: Arc<RwLock<Vec<Tensor>>>,
    local_step: usize,
}

impl Hogwild {
    /// Create a new Hogwild! optimizer.
    pub fn new(
        config: HogwildConfig,
        worker_id: usize,
        shared_parameters: Arc<RwLock<Vec<Tensor>>>,
    ) -> Self {
        Self {
            config,
            worker_id,
            shared_parameters,
            local_step: 0,
        }
    }

    /// Perform sparse parameter update.
    pub fn sparse_step(&mut self, sparse_gradients: &[(usize, Tensor)]) -> Result<()> {
        // Lock-free updates for sparse gradients
        // In practice, this would use atomic operations for true lock-free behavior

        for &(param_idx, ref gradient) in sparse_gradients {
            {
                let params = self.shared_parameters.read();
                if param_idx >= params.len() {
                    continue;
                }
            } // Release read lock

            // This is a simplified version - real Hogwild! uses lock-free atomic updates
            let mut params_write = self.shared_parameters.write();
            let update = gradient.mul_scalar(self.config.learning_rate)?;
            params_write[param_idx] = params_write[param_idx].sub(&update)?;
        }

        self.local_step += 1;
        Ok(())
    }

    /// Generate sparse gradient indices based on sparse ratio.
    pub fn select_sparse_indices(&self, total_params: usize) -> Vec<usize> {
        use scirs2_core::random::*; // SciRS2 Integration Policy

        let num_sparse = (total_params as f32 * self.config.sparse_ratio) as usize;
        let mut indices: Vec<usize> = (0..total_params).collect();
        let mut rng = thread_rng();
        indices.shuffle(rng.rng_mut());
        indices.truncate(num_sparse);
        indices
    }
}

/// Delayed gradient optimizer.
pub struct DelayedGradient {
    config: DelayedGradientConfig,
    parameters: Vec<Tensor>,
    gradient_buffer: Vec<(Tensor, usize, Instant)>, // (gradient, delay, timestamp)
    current_step: usize,
}

impl DelayedGradient {
    /// Create a new delayed gradient optimizer.
    pub fn new(config: DelayedGradientConfig, initial_parameters: Vec<Tensor>) -> Self {
        Self {
            config,
            parameters: initial_parameters,
            gradient_buffer: Vec::new(),
            current_step: 0,
        }
    }

    /// Add a delayed gradient to the buffer.
    pub fn add_delayed_gradient(&mut self, gradient: Tensor, delay: usize) {
        self.gradient_buffer.push((gradient, delay, Instant::now()));
    }

    /// Process delayed gradients and update parameters.
    pub fn step(&mut self) -> Result<()> {
        self.current_step += 1;

        // Process gradients that are ready
        let mut i = 0;
        while i < self.gradient_buffer.len() {
            let (ref gradient, delay, timestamp) = &self.gradient_buffer[i];
            let age = timestamp.elapsed();

            if age >= Duration::from_millis((*delay as u64) * 10) {
                // Apply delay compensation
                let compensation = self.compute_delay_compensation(*delay)?;
                let compensated_lr = self.config.learning_rate * compensation;

                // Update parameters
                for (j, param) in self.parameters.iter_mut().enumerate() {
                    if j < 1 {
                        // Assuming single parameter for simplicity
                        let update = gradient.mul_scalar(compensated_lr)?;
                        *param = param.sub(&update)?;
                    }
                }

                self.gradient_buffer.remove(i);
            } else {
                i += 1;
            }
        }

        Ok(())
    }

    fn compute_delay_compensation(&self, delay: usize) -> Result<f32> {
        if delay > self.config.max_delay {
            return Ok(0.0); // Discard very old gradients
        }

        let delay_ratio = delay as f32 / self.config.max_delay as f32;

        let compensation = match self.config.compensation_method {
            DelayCompensationMethod::None => 1.0,
            DelayCompensationMethod::LinearDecay => {
                1.0 - delay_ratio * self.config.compensation_factor
            },
            DelayCompensationMethod::ExponentialDecay => {
                (-delay_ratio * self.config.compensation_factor).exp()
            },
            DelayCompensationMethod::Adaptive => {
                // Simple adaptive scheme
                1.0 / (1.0 + delay_ratio * self.config.compensation_factor)
            },
        };

        Ok(compensation.max(0.1)) // Minimum 10% of original learning rate
    }

    /// Get current parameters.
    pub fn get_parameters(&self) -> &[Tensor] {
        &self.parameters
    }
}

/// Elastic Averaging SGD optimizer.
pub struct ElasticAveraging {
    config: ElasticAveragingConfig,
    #[allow(dead_code)]
    worker_id: usize,
    local_parameters: Vec<Tensor>,
    global_parameters: Arc<RwLock<Vec<Tensor>>>,
    elastic_force: Vec<Tensor>,
    local_step: usize,
    last_communication: usize,
}

impl ElasticAveraging {
    /// Create a new Elastic Averaging SGD optimizer.
    pub fn new(
        config: ElasticAveragingConfig,
        worker_id: usize,
        global_parameters: Arc<RwLock<Vec<Tensor>>>,
    ) -> Result<Self> {
        let global_params = global_parameters.read().clone();
        let param_count = global_params.len();

        Ok(Self {
            config,
            worker_id,
            local_parameters: global_params.clone(),
            global_parameters,
            elastic_force: (0..param_count)
                .map(|i| Tensor::zeros(&global_params[i].shape()).map_err(anyhow::Error::from))
                .collect::<Result<Vec<_>>>()?,
            local_step: 0,
            last_communication: 0,
        })
    }

    /// Perform optimization step with elastic averaging.
    pub fn step(&mut self, gradients: &[Tensor]) -> Result<()> {
        // Update local parameters with gradients
        for (i, gradient) in gradients.iter().enumerate() {
            if i < self.local_parameters.len() {
                let update = gradient.mul_scalar(self.config.learning_rate)?;
                self.local_parameters[i] = self.local_parameters[i].sub(&update)?;
            }
        }

        // Apply elastic force
        let global_params = self.global_parameters.read();
        for i in 0..self.local_parameters.len() {
            let diff = self.local_parameters[i].sub(&global_params[i])?;
            self.elastic_force[i] = diff.mul_scalar(self.config.alpha)?;
            let elastic_update = self.elastic_force[i].mul_scalar(self.config.learning_rate)?;
            self.local_parameters[i] = self.local_parameters[i].sub(&elastic_update)?;
        }
        drop(global_params);

        self.local_step += 1;

        // Communicate with global parameters periodically
        if self.local_step - self.last_communication >= self.config.tau {
            self.communicate_with_global()?;
            self.last_communication = self.local_step;
        }

        Ok(())
    }

    fn communicate_with_global(&mut self) -> Result<()> {
        let mut global_params = self.global_parameters.write();

        // Update global parameters with moving average
        for i in 0..global_params.len() {
            let local_contrib = self.local_parameters[i].mul_scalar(1.0 - self.config.beta)?;
            let global_contrib = global_params[i].mul_scalar(self.config.beta)?;
            global_params[i] = local_contrib.add(&global_contrib)?;
        }

        // Update local parameters from global
        self.local_parameters = global_params.clone();

        Ok(())
    }

    /// Get current local parameters.
    pub fn get_parameters(&self) -> &[Tensor] {
        &self.local_parameters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_async_sgd_config() {
        let config = AsyncSGDConfig::default();
        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.momentum, 0.9);
        assert_eq!(config.max_staleness, 10);
    }

    #[test]
    fn test_hogwild_config() {
        let config = HogwildConfig::default();
        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.sparse_ratio, 0.1);
        assert_eq!(config.max_workers, 4);
    }

    #[test]
    fn test_delayed_gradient_config() {
        let config = DelayedGradientConfig::default();
        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.max_delay, 20);
        assert!(matches!(
            config.compensation_method,
            DelayCompensationMethod::LinearDecay
        ));
    }

    #[test]
    fn test_parameter_server_creation() {
        let params = vec![Tensor::zeros(&[10]).unwrap()];
        let server = ParameterServer::new(params);
        assert_eq!(server.get_global_step(), 0);
    }

    #[test]
    fn test_elastic_averaging_config() {
        let config = ElasticAveragingConfig::default();
        assert_eq!(config.learning_rate, 1e-3);
        assert_eq!(config.alpha, 0.6);
        assert_eq!(config.tau, 10);
        assert_eq!(config.beta, 0.9);
    }
}
