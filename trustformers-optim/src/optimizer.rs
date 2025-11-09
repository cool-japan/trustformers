use anyhow::Result;
use std::collections::HashMap;
use trustformers_core::tensor::Tensor;

/// Trait for optimizer state management and parameter updates.
pub trait OptimizerState {
    /// Zero out gradients
    fn zero_grad(&mut self) -> Result<()>;

    /// Perform optimization step
    fn step(&mut self, parameters: &mut [Tensor]) -> Result<()>;

    /// Get current learning rate
    fn get_lr(&self) -> f32;

    /// Set learning rate
    fn set_lr(&mut self, lr: f32);

    /// Save optimizer state to dictionary
    fn state_dict(&self) -> Result<HashMap<String, Tensor>>;

    /// Load optimizer state from dictionary
    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()>;
}
