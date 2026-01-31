//! Advanced Optimizer Features
//!
//! This module implements advanced optimization techniques including optimizer fusion,
//! multi-optimizer training, warm-up strategies, and checkpointing optimizations.

use crate::LRScheduler;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::traits::Optimizer;
use trustformers_core::Tensor;

/// Configuration for optimizer fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionConfig {
    /// Whether to enable parameter fusion
    pub fuse_parameters: bool,
    /// Whether to enable gradient fusion
    pub fuse_gradients: bool,
    /// Whether to enable state fusion
    pub fuse_state: bool,
    /// Fusion window size
    pub window_size: usize,
    /// Memory threshold for fusion (in bytes)
    pub memory_threshold: usize,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            fuse_parameters: true,
            fuse_gradients: true,
            fuse_state: true,
            window_size: 32,
            memory_threshold: 1024 * 1024 * 100, // 100MB
        }
    }
}

/// Fused optimizer that combines multiple optimizers for efficiency
pub struct FusedOptimizer {
    optimizers: Vec<Box<dyn Optimizer>>,
    config: FusionConfig,
    fused_parameters: Arc<Mutex<HashMap<String, Tensor>>>,
    fused_gradients: Arc<Mutex<HashMap<String, Tensor>>>,
    fusion_groups: Vec<Vec<usize>>, // Groups of optimizer indices that can be fused
}

impl FusedOptimizer {
    /// Create a new fused optimizer
    pub fn new(optimizers: Vec<Box<dyn Optimizer>>, config: FusionConfig) -> Result<Self> {
        let fusion_groups = Self::compute_fusion_groups(&optimizers, &config);

        Ok(Self {
            optimizers,
            config,
            fused_parameters: Arc::new(Mutex::new(HashMap::new())),
            fused_gradients: Arc::new(Mutex::new(HashMap::new())),
            fusion_groups,
        })
    }

    /// Compute fusion groups based on optimizer compatibility
    fn compute_fusion_groups(
        optimizers: &[Box<dyn Optimizer>],
        config: &FusionConfig,
    ) -> Vec<Vec<usize>> {
        let mut groups = Vec::new();
        let mut used = vec![false; optimizers.len()];

        for i in 0..optimizers.len() {
            if used[i] {
                continue;
            }

            let mut group = vec![i];
            used[i] = true;

            // Find compatible optimizers to fuse with
            for j in (i + 1)..optimizers.len() {
                if used[j] {
                    continue;
                }

                if Self::can_fuse(&optimizers[i], &optimizers[j], config) {
                    group.push(j);
                    used[j] = true;
                }
            }

            groups.push(group);
        }

        groups
    }

    /// Check if two optimizers can be fused
    fn can_fuse(
        _opt1: &Box<dyn Optimizer>,
        _opt2: &Box<dyn Optimizer>,
        _config: &FusionConfig,
    ) -> bool {
        // Simple heuristic: for now, assume all optimizers can be fused
        // In a real implementation, we would check optimizer types and configurations
        true
    }

    /// Fuse parameters across optimizer groups
    fn fuse_parameters(&self, parameters: &mut HashMap<String, Tensor>) -> Result<()> {
        if !self.config.fuse_parameters {
            return Ok(());
        }

        let mut fused_params = self
            .fused_parameters
            .lock()
            .expect("fused_parameters lock should not be poisoned");
        fused_params.clear();

        // Group parameters by fusion groups
        for group in &self.fusion_groups {
            if group.len() > 1 {
                // Create fused parameter tensor for this group
                let group_params: Vec<_> = parameters
                    .iter()
                    .filter(|(name, _)| {
                        // Check if parameter belongs to this group (simplified)
                        group.iter().any(|&i| name.contains(&format!("opt_{}", i)))
                    })
                    .collect();

                if !group_params.is_empty() {
                    // Concatenate parameters
                    let fused_name = format!("fused_group_{}", group[0]);
                    let fused_tensor = self.concatenate_tensors(
                        &group_params.iter().map(|(_, t)| *t).collect::<Vec<_>>(),
                    )?;
                    fused_params.insert(fused_name, fused_tensor);
                }
            }
        }

        Ok(())
    }

    /// Concatenate tensors for fusion
    fn concatenate_tensors(&self, tensors: &[&Tensor]) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(TrustformersError::invalid_argument(
                "Empty tensor list".to_string(),
            ));
        }

        // Flatten all tensors and concatenate
        let mut total_size = 0;
        for tensor in tensors {
            total_size += tensor.len();
        }

        // Create concatenated tensor (simplified implementation)
        Tensor::zeros(&[total_size])
    }

    /// Perform fused optimization step
    pub fn fused_step(&mut self, parameters: &mut HashMap<String, Tensor>) -> Result<()> {
        // Fuse parameters
        self.fuse_parameters(parameters)?;

        // Apply optimization steps to fused groups
        let fusion_groups = self.fusion_groups.clone();
        for group in &fusion_groups {
            if group.len() > 1 {
                // Apply fused optimization
                self.apply_fused_group_optimization(group)?;
            } else {
                // Apply single optimizer
                let optimizer_idx = group[0];
                // Apply optimization for single optimizer (simplified)
                for (name, param) in parameters.iter_mut() {
                    if let Some(grad) = self.get_gradient_for_param(name) {
                        self.optimizers[optimizer_idx].update(param, &grad)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply optimization to a fused group
    fn apply_fused_group_optimization(&mut self, group: &[usize]) -> Result<()> {
        // Use the first optimizer in the group as the representative
        let primary_optimizer_idx = group[0];

        let mut fused_params = self
            .fused_parameters
            .lock()
            .expect("fused_parameters lock should not be poisoned");
        let fused_gradients = self
            .fused_gradients
            .lock()
            .expect("fused_gradients lock should not be poisoned");

        let group_name = format!("fused_group_{}", primary_optimizer_idx);

        if let (Some(param), Some(grad)) = (
            fused_params.get_mut(&group_name),
            fused_gradients.get(&group_name),
        ) {
            self.optimizers[primary_optimizer_idx].update(param, grad)?;
        }

        Ok(())
    }

    /// Get gradient for parameter
    ///
    /// Integrates with the automatic differentiation system to retrieve
    /// accumulated gradients for the specified parameter.
    fn get_gradient_for_param(&self, param_name: &str) -> Option<Tensor> {
        // Check fused gradients first (higher priority)
        {
            let fused_gradients = self.fused_gradients.lock().ok()?;
            if let Some(gradient) = fused_gradients.get(param_name) {
                return Some(gradient.clone());
            }
        }

        // Check individual optimizer gradients by parameter name
        // Parameter names are typically formatted as "optimizer_{idx}_{param_name}"
        for (idx, _optimizer) in self.optimizers.iter().enumerate() {
            let full_param_name = format!("optimizer_{}_{}", idx, param_name);

            // Try to get gradient from fused gradients with full name
            let fused_gradients = self.fused_gradients.lock().ok()?;
            if let Some(gradient) = fused_gradients.get(&full_param_name) {
                return Some(gradient.clone());
            }
            drop(fused_gradients);

            // For individual parameters, we would need access to the parameter
            // registry maintained by the automatic differentiation system.
            // This is typically maintained at the model level rather than optimizer level.
        }

        // Return None if gradient not found in any registry
        None
    }

    /// Register gradient for parameter in the fused gradient registry
    ///
    /// This method allows external automatic differentiation systems to register
    /// computed gradients with the fused optimizer for parameter updates.
    pub fn register_gradient(&self, param_name: &str, gradient: Tensor) -> Result<()> {
        let mut fused_gradients = self.fused_gradients.lock().map_err(|_| {
            TrustformersError::tensor_op_error(
                "Failed to lock fused gradients",
                "register_gradient",
            )
        })?;

        fused_gradients.insert(param_name.to_string(), gradient);
        Ok(())
    }

    /// Clear all registered gradients
    ///
    /// This should be called after each optimization step to clear accumulated gradients.
    pub fn clear_gradients(&self) -> Result<()> {
        let mut fused_gradients = self.fused_gradients.lock().map_err(|_| {
            TrustformersError::tensor_op_error("Failed to lock fused gradients", "clear_gradients")
        })?;

        fused_gradients.clear();
        Ok(())
    }

    /// Get all available gradient parameter names
    ///
    /// Returns a list of parameter names for which gradients are currently available.
    pub fn get_available_gradient_names(&self) -> Result<Vec<String>> {
        let fused_gradients = self.fused_gradients.lock().map_err(|_| {
            TrustformersError::tensor_op_error(
                "Failed to lock fused gradients",
                "get_available_gradient_names",
            )
        })?;

        Ok(fused_gradients.keys().cloned().collect())
    }

    /// Get fusion statistics
    pub fn get_fusion_stats(&self) -> FusionStats {
        let total_optimizers = self.optimizers.len();
        let fused_groups = self.fusion_groups.iter().filter(|group| group.len() > 1).count();
        let unfused_optimizers = self.fusion_groups.iter().filter(|group| group.len() == 1).count();

        FusionStats {
            total_optimizers,
            fused_groups,
            unfused_optimizers,
            fusion_ratio: fused_groups as f64 / total_optimizers as f64,
            memory_saved: self.estimate_memory_savings(),
        }
    }

    /// Estimate memory savings from fusion
    fn estimate_memory_savings(&self) -> usize {
        let fused_params = self
            .fused_parameters
            .lock()
            .expect("fused_parameters lock should not be poisoned");
        let total_fused_size: usize = fused_params.values()
            .map(|t| t.len() * 4) // Assuming f32 tensors
            .sum();

        // Estimate original size
        let estimated_original_size = total_fused_size * 2; // Conservative estimate

        estimated_original_size.saturating_sub(total_fused_size)
    }
}

/// Statistics for optimizer fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionStats {
    pub total_optimizers: usize,
    pub fused_groups: usize,
    pub unfused_optimizers: usize,
    pub fusion_ratio: f64,
    pub memory_saved: usize,
}

/// Multi-optimizer training system
pub struct MultiOptimizerTrainer {
    optimizers: HashMap<String, Box<dyn Optimizer>>,
    parameter_assignments: HashMap<String, String>, // param_name -> optimizer_name
    schedulers: HashMap<String, Box<dyn LRScheduler>>,
    weights: HashMap<String, f64>, // optimizer weights for ensemble
}

impl Default for MultiOptimizerTrainer {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiOptimizerTrainer {
    /// Create a new multi-optimizer trainer
    pub fn new() -> Self {
        Self {
            optimizers: HashMap::new(),
            parameter_assignments: HashMap::new(),
            schedulers: HashMap::new(),
            weights: HashMap::new(),
        }
    }

    /// Add an optimizer with a name
    pub fn add_optimizer(
        &mut self,
        name: String,
        optimizer: Box<dyn Optimizer>,
        weight: f64,
    ) -> Result<()> {
        self.optimizers.insert(name.clone(), optimizer);
        self.weights.insert(name, weight);
        Ok(())
    }

    /// Add a scheduler for an optimizer
    pub fn add_scheduler(
        &mut self,
        optimizer_name: String,
        scheduler: Box<dyn LRScheduler>,
    ) -> Result<()> {
        if !self.optimizers.contains_key(&optimizer_name) {
            return Err(TrustformersError::invalid_argument(format!(
                "Optimizer {} not found",
                optimizer_name
            )));
        }

        self.schedulers.insert(optimizer_name, scheduler);
        Ok(())
    }

    /// Assign parameters to optimizers
    pub fn assign_parameters(&mut self, assignments: HashMap<String, String>) -> Result<()> {
        // Validate that all assigned optimizers exist
        for optimizer_name in assignments.values() {
            if !self.optimizers.contains_key(optimizer_name) {
                return Err(TrustformersError::invalid_argument(format!(
                    "Optimizer {} not found",
                    optimizer_name
                )));
            }
        }

        self.parameter_assignments = assignments;
        Ok(())
    }

    /// Perform multi-optimizer training step
    pub fn step(
        &mut self,
        parameters: &HashMap<String, Tensor>,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<()> {
        // Group parameters by optimizer
        let mut optimizer_params: HashMap<String, Vec<(String, Tensor, Tensor)>> = HashMap::new();

        for (param_name, param) in parameters {
            if let Some(grad) = gradients.get(param_name) {
                let optimizer_name = self
                    .parameter_assignments
                    .get(param_name)
                    .cloned()
                    .unwrap_or_else(|| "default".to_string());

                optimizer_params.entry(optimizer_name).or_default().push((
                    param_name.clone(),
                    param.clone(),
                    grad.clone(),
                ));
            }
        }

        // Apply optimizers
        for (optimizer_name, param_grad_pairs) in optimizer_params {
            if let Some(optimizer) = self.optimizers.get_mut(&optimizer_name) {
                let weight = self.weights.get(&optimizer_name).copied().unwrap_or(1.0);

                for (_, param, grad) in param_grad_pairs {
                    // Scale gradient by optimizer weight
                    let scaled_grad = grad.mul_scalar(weight as f32)?;
                    optimizer.update(&mut param.clone(), &scaled_grad)?;
                }
            }
        }

        Ok(())
    }

    /// Update learning rates using schedulers
    pub fn step_schedulers(&mut self, epoch: usize) -> Result<()> {
        for (optimizer_name, scheduler) in &mut self.schedulers {
            let new_lr = scheduler.get_lr(epoch);

            if let Some(optimizer) = self.optimizers.get_mut(optimizer_name) {
                optimizer.set_lr(new_lr);
            }
        }

        Ok(())
    }

    /// Get training statistics
    pub fn get_stats(&self) -> MultiOptimizerStats {
        MultiOptimizerStats {
            num_optimizers: self.optimizers.len(),
            num_schedulers: self.schedulers.len(),
            num_assigned_params: self.parameter_assignments.len(),
            optimizer_weights: self.weights.clone(),
        }
    }
}

/// Statistics for multi-optimizer training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiOptimizerStats {
    pub num_optimizers: usize,
    pub num_schedulers: usize,
    pub num_assigned_params: usize,
    pub optimizer_weights: HashMap<String, f64>,
}

/// Optimizer warm-up strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WarmupStrategy {
    /// Linear warmup
    Linear { steps: usize },
    /// Exponential warmup
    Exponential { steps: usize, base: f64 },
    /// Cosine warmup
    Cosine { steps: usize },
    /// Custom warmup function
    Custom { steps: usize },
}

/// Warmup optimizer wrapper
pub struct WarmupOptimizer {
    inner: Box<dyn Optimizer>,
    strategy: WarmupStrategy,
    current_step: usize,
    base_lr: f64,
    target_lr: f64,
}

impl WarmupOptimizer {
    /// Create a new warmup optimizer
    pub fn new(
        optimizer: Box<dyn Optimizer>,
        strategy: WarmupStrategy,
        base_lr: f64,
        target_lr: f64,
    ) -> Self {
        Self {
            inner: optimizer,
            strategy,
            current_step: 0,
            base_lr,
            target_lr,
        }
    }

    /// Get current learning rate based on warmup strategy
    fn get_warmup_lr(&self) -> f64 {
        let warmup_steps = match &self.strategy {
            WarmupStrategy::Linear { steps } => *steps,
            WarmupStrategy::Exponential { steps, .. } => *steps,
            WarmupStrategy::Cosine { steps } => *steps,
            WarmupStrategy::Custom { steps } => *steps,
        };

        if self.current_step >= warmup_steps {
            return self.target_lr;
        }

        let progress = self.current_step as f64 / warmup_steps as f64;

        match &self.strategy {
            WarmupStrategy::Linear { .. } => {
                self.base_lr + (self.target_lr - self.base_lr) * progress
            },
            WarmupStrategy::Exponential { base, .. } => {
                self.base_lr + (self.target_lr - self.base_lr) * base.powf(1.0 - progress)
            },
            WarmupStrategy::Cosine { .. } => {
                let cosine_progress = 0.5 * (1.0 - (std::f64::consts::PI * progress).cos());
                self.base_lr + (self.target_lr - self.base_lr) * cosine_progress
            },
            WarmupStrategy::Custom { .. } => {
                // Custom implementation would go here
                self.base_lr + (self.target_lr - self.base_lr) * progress
            },
        }
    }

    /// Check if warmup is complete
    pub fn is_warmup_complete(&self) -> bool {
        let warmup_steps = match &self.strategy {
            WarmupStrategy::Linear { steps } => *steps,
            WarmupStrategy::Exponential { steps, .. } => *steps,
            WarmupStrategy::Cosine { steps } => *steps,
            WarmupStrategy::Custom { steps } => *steps,
        };

        self.current_step >= warmup_steps
    }
}

impl Optimizer for WarmupOptimizer {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        // Update learning rate based on warmup strategy
        let current_lr = self.get_warmup_lr();
        self.inner.set_lr(current_lr as f32);

        // Perform the actual optimization step
        self.inner.update(parameter, grad)
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad()
    }

    fn step(&mut self) {
        self.inner.step();
        self.current_step += 1;
    }

    fn get_lr(&self) -> f32 {
        self.get_warmup_lr() as f32
    }

    fn set_lr(&mut self, lr: f32) {
        self.target_lr = lr as f64;
        self.inner.set_lr(lr);
    }
}

/// Checkpointing optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Save interval (in steps)
    pub save_interval: usize,
    /// Whether to compress checkpoints
    pub compress: bool,
    /// Maximum number of checkpoints to keep
    pub max_checkpoints: usize,
    /// Whether to save only state diffs
    pub incremental: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            save_interval: 1000,
            compress: true,
            max_checkpoints: 5,
            incremental: false,
        }
    }
}

/// Memory-bandwidth co-optimization
pub struct MemoryBandwidthOptimizer {
    inner: Box<dyn Optimizer>,
    memory_threshold: usize,
    bandwidth_threshold: f64,
    adaptive_batch_size: bool,
    current_batch_size: usize,
    base_batch_size: usize,
}

impl MemoryBandwidthOptimizer {
    /// Create a new memory-bandwidth co-optimizer
    pub fn new(
        optimizer: Box<dyn Optimizer>,
        memory_threshold: usize,
        bandwidth_threshold: f64,
        base_batch_size: usize,
    ) -> Self {
        Self {
            inner: optimizer,
            memory_threshold,
            bandwidth_threshold,
            adaptive_batch_size: true,
            current_batch_size: base_batch_size,
            base_batch_size,
        }
    }

    /// Adjust batch size based on memory and bandwidth usage
    pub fn adjust_batch_size(&mut self, memory_usage: usize, bandwidth_usage: f64) -> usize {
        if !self.adaptive_batch_size {
            return self.current_batch_size;
        }

        let memory_pressure = memory_usage as f64 / self.memory_threshold as f64;
        let bandwidth_pressure = bandwidth_usage / self.bandwidth_threshold;

        let pressure = memory_pressure.max(bandwidth_pressure);

        if pressure > 1.1 {
            // High pressure - reduce batch size
            self.current_batch_size = (self.current_batch_size as f64 * 0.9) as usize;
            self.current_batch_size = self.current_batch_size.max(1);
        } else if pressure < 0.8 {
            // Low pressure - increase batch size
            self.current_batch_size = (self.current_batch_size as f64 * 1.1) as usize;
            self.current_batch_size = self.current_batch_size.min(self.base_batch_size * 4);
        }

        self.current_batch_size
    }

    /// Get current resource utilization
    pub fn get_utilization(&self) -> ResourceUtilization {
        ResourceUtilization {
            current_batch_size: self.current_batch_size,
            base_batch_size: self.base_batch_size,
            memory_threshold: self.memory_threshold,
            bandwidth_threshold: self.bandwidth_threshold,
            adaptive_enabled: self.adaptive_batch_size,
        }
    }
}

impl Optimizer for MemoryBandwidthOptimizer {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        self.inner.update(parameter, grad)
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad()
    }

    fn step(&mut self) {
        self.inner.step()
    }

    fn get_lr(&self) -> f32 {
        self.inner.get_lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.inner.set_lr(lr)
    }
}

/// Resource utilization statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    pub current_batch_size: usize,
    pub base_batch_size: usize,
    pub memory_threshold: usize,
    pub bandwidth_threshold: f64,
    pub adaptive_enabled: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Adam;

    #[test]
    fn test_fusion_config_default() {
        let config = FusionConfig::default();
        assert!(config.fuse_parameters);
        assert!(config.fuse_gradients);
        assert!(config.fuse_state);
        assert_eq!(config.window_size, 32);
    }

    #[test]
    fn test_warmup_strategy_linear() {
        let strategy = WarmupStrategy::Linear { steps: 100 };

        let adam = Adam::new(0.001, (0.9, 0.999), 1e-8, 0.0);

        let warmup_optimizer = WarmupOptimizer::new(Box::new(adam), strategy, 0.0, 0.001);

        assert!(!warmup_optimizer.is_warmup_complete());
        assert_eq!(warmup_optimizer.get_warmup_lr(), 0.0);
    }

    #[test]
    fn test_multi_optimizer_trainer_creation() {
        let mut trainer = MultiOptimizerTrainer::new();

        let adam = Adam::new(0.001, (0.9, 0.999), 1e-8, 0.0);
        trainer.add_optimizer("adam".to_string(), Box::new(adam), 1.0).unwrap();

        let stats = trainer.get_stats();
        assert_eq!(stats.num_optimizers, 1);
        assert_eq!(stats.optimizer_weights.get("adam"), Some(&1.0));
    }

    #[test]
    fn test_memory_bandwidth_optimizer() {
        let adam = Adam::new(0.001, (0.9, 0.999), 1e-8, 0.0);
        let mut mb_optimizer = MemoryBandwidthOptimizer::new(
            Box::new(adam),
            1024 * 1024 * 100, // 100MB
            100.0,             // 100 MB/s
            32,
        );

        let utilization = mb_optimizer.get_utilization();
        assert_eq!(utilization.current_batch_size, 32);
        assert_eq!(utilization.base_batch_size, 32);

        // Test batch size adjustment under high memory pressure
        let new_batch_size = mb_optimizer.adjust_batch_size(
            1024 * 1024 * 120, // 120MB (above threshold)
            50.0,
        );
        assert!(new_batch_size < 32);
    }

    #[test]
    fn test_checkpoint_config_default() {
        let config = CheckpointConfig::default();
        assert_eq!(config.save_interval, 1000);
        assert!(config.compress);
        assert_eq!(config.max_checkpoints, 5);
        assert!(!config.incremental);
    }
}
