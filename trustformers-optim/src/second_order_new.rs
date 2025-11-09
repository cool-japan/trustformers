/// Second-order optimization methods
///
/// This module has been refactored into separate submodules for better organization
/// and maintainability. Each optimizer now has its own dedicated module.

pub mod lbfgs;

// Import from original file for remaining optimizers that need to be extracted
use anyhow::Result;
use std::collections::HashMap;
use trustformers_core::tensor::Tensor;

// Re-export the LBFGS components
pub use lbfgs::{LBFGS, LineSearchMethod};

// The following structs would be moved to their own modules in a complete refactor:

/// Kronecker-Factored Approximate Curvature (K-FAC) optimizer.
///
/// K-FAC approximates the Fisher Information Matrix using Kronecker products,
/// making second-order optimization feasible for neural networks.
///
/// Note: This implementation would be moved to its own module (kfac.rs) in a complete refactor.
#[derive(Debug)]
pub struct KFAC {
    pub learning_rate: f32,
    pub momentum: f32,
    pub damping: f32,
    pub weight_decay: f32,
    pub update_freq: usize,
    pub eps: f32,

    // Internal state
    pub step: usize,
    pub momentum_buffer: HashMap<String, Vec<f32>>,
    pub cov_ata: HashMap<String, Vec<Vec<f32>>>, // A^T A covariance matrices
    pub cov_ggt: HashMap<String, Vec<Vec<f32>>>, // G G^T covariance matrices
    pub inv_cov_ata: HashMap<String, Vec<Vec<f32>>>, // Inverse of A^T A
    pub inv_cov_ggt: HashMap<String, Vec<Vec<f32>>>, // Inverse of G G^T
}

impl Default for KFAC {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            momentum: 0.9,
            damping: 1e-3,
            weight_decay: 0.0,
            update_freq: 10,
            eps: 1e-10,
            step: 0,
            momentum_buffer: HashMap::new(),
            cov_ata: HashMap::new(),
            cov_ggt: HashMap::new(),
            inv_cov_ata: HashMap::new(),
            inv_cov_ggt: HashMap::new(),
        }
    }
}

impl KFAC {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            ..Default::default()
        }
    }
}

/// Shampoo optimizer with adaptive preconditioning
///
/// Note: This implementation would be moved to its own module (shampoo.rs) in a complete refactor.
#[derive(Debug)]
pub struct Shampoo {
    pub learning_rate: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub momentum: f32,
    pub update_freq: usize,

    // Internal state
    pub step: usize,
    pub momentum_buffer: HashMap<String, Vec<f32>>,
    pub h_matrices: HashMap<String, Vec<Vec<f32>>>,
}

impl Default for Shampoo {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            eps: 1e-4,
            weight_decay: 0.0,
            momentum: 0.0,
            update_freq: 10,
            step: 0,
            momentum_buffer: HashMap::new(),
            h_matrices: HashMap::new(),
        }
    }
}

impl Shampoo {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            ..Default::default()
        }
    }
}

/// Natural Gradient Descent optimizer
///
/// Note: This implementation would be moved to its own module (natural_gradient.rs) in a complete refactor.
#[derive(Debug)]
pub struct NaturalGradient {
    pub learning_rate: f32,
    pub damping: f32,
    pub update_freq: usize,

    // Internal state
    pub step: usize,
    pub fisher_info: HashMap<String, Vec<Vec<f32>>>,
    pub inv_fisher: HashMap<String, Vec<Vec<f32>>>,
}

impl Default for NaturalGradient {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            damping: 1e-3,
            update_freq: 10,
            step: 0,
            fisher_info: HashMap::new(),
            inv_fisher: HashMap::new(),
        }
    }
}

impl NaturalGradient {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            ..Default::default()
        }
    }
}