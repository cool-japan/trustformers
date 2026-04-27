//! Lazy optimizer state: allocate optimizer state only when first gradient is seen.
//!
//! Reduces memory usage for sparse models and large embedding tables where many
//! parameters may never receive a gradient (e.g., tokens that are never used in
//! the current batch).

use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};

// ─────────────────────────────────────────── LazyStateValue ─────────────────

/// A lazily-allocated optimizer state value.
///
/// Starts in the `Uninitialized` variant and transitions to a concrete storage
/// variant upon the first gradient observation.
#[derive(Debug)]
pub enum LazyStateValue {
    /// Not yet allocated — zero memory cost.
    Uninitialized,
    /// A single f32 scalar (e.g., per-parameter step count).
    Scalar(f32),
    /// A heap-allocated f32 vector (e.g., first/second moment buffers).
    Vector(Vec<f32>),
}

impl LazyStateValue {
    /// Returns `true` if the value has been initialized.
    pub fn is_initialized(&self) -> bool {
        !matches!(self, Self::Uninitialized)
    }

    /// Returns the scalar value if this variant is `Scalar`.
    pub fn as_scalar(&self) -> Option<f32> {
        match self {
            Self::Scalar(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns an immutable slice of the vector if this variant is `Vector`.
    pub fn as_vector(&self) -> Option<&[f32]> {
        match self {
            Self::Vector(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Returns a mutable reference to the inner `Vec<f32>` if this variant is `Vector`.
    pub fn as_vector_mut(&mut self) -> Option<&mut Vec<f32>> {
        match self {
            Self::Vector(v) => Some(v),
            _ => None,
        }
    }

    /// Initialize as a scalar.  Panics if already initialized (programming error).
    pub fn initialize_scalar(&mut self, value: f32) {
        debug_assert!(
            matches!(self, Self::Uninitialized),
            "LazyStateValue::initialize_scalar called on an already-initialized value"
        );
        *self = Self::Scalar(value);
    }

    /// Initialize as a vector of `shape` elements all filled with `fill_value`.
    pub fn initialize_vector(&mut self, shape: usize, fill_value: f32) {
        debug_assert!(
            matches!(self, Self::Uninitialized),
            "LazyStateValue::initialize_vector called on an already-initialized value"
        );
        *self = Self::Vector(vec![fill_value; shape]);
    }

    /// Approximate heap memory in bytes.
    pub fn memory_bytes(&self) -> usize {
        match self {
            Self::Uninitialized => 0,
            Self::Scalar(_) => std::mem::size_of::<f32>(),
            Self::Vector(v) => v.len() * std::mem::size_of::<f32>(),
        }
    }
}

// ─────────────────────────────────────────── LazyParamState ─────────────────

/// Lazy optimizer state container for a single parameter tensor.
///
/// State entries are stored in a `HashMap<String, LazyStateValue>` and
/// allocated on the first call to `get_or_init_scalar` / `get_or_init_vector`.
#[derive(Debug, Default)]
pub struct LazyParamState {
    /// Named state entries (e.g., `"exp_avg"`, `"exp_avg_sq"`).
    states: HashMap<String, LazyStateValue>,
    /// How many optimizer updates have been applied to this parameter.
    step: u64,
}

impl LazyParamState {
    /// Create an empty (fully uninitialized) param state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or lazily initialize a scalar state, returning a mutable reference.
    pub fn get_or_init_scalar(&mut self, key: &str, init_value: f32) -> &mut f32 {
        let entry = self.states.entry(key.to_owned()).or_insert(LazyStateValue::Uninitialized);
        if matches!(entry, LazyStateValue::Uninitialized) {
            entry.initialize_scalar(init_value);
        }
        match entry {
            LazyStateValue::Scalar(v) => v,
            _ => unreachable!("entry should be Scalar after initialization"),
        }
    }

    /// Get or lazily initialize a vector state, returning a mutable reference.
    pub fn get_or_init_vector(&mut self, key: &str, size: usize, fill: f32) -> &mut Vec<f32> {
        let entry = self.states.entry(key.to_owned()).or_insert(LazyStateValue::Uninitialized);
        if matches!(entry, LazyStateValue::Uninitialized) {
            entry.initialize_vector(size, fill);
        }
        match entry {
            LazyStateValue::Vector(v) => v,
            _ => unreachable!("entry should be Vector after initialization"),
        }
    }

    /// Read a scalar without mutating state. Returns `None` if not yet initialized.
    pub fn get_scalar(&self, key: &str) -> Option<f32> {
        self.states.get(key)?.as_scalar()
    }

    /// Increment the per-parameter step counter.
    pub fn increment_step(&mut self) {
        self.step += 1;
    }

    /// Current update step count for this parameter (0 = never updated).
    pub fn step(&self) -> u64 {
        self.step
    }

    /// Approximate memory consumed by all initialized state entries (bytes).
    pub fn memory_bytes(&self) -> usize {
        self.states.values().map(|v| v.memory_bytes()).sum()
    }

    /// Number of state entries that have been initialized (i.e., not `Uninitialized`).
    pub fn num_initialized(&self) -> usize {
        self.states.values().filter(|v| v.is_initialized()).count()
    }

    /// Total number of state entries (initialized + uninitialized).
    pub fn num_entries(&self) -> usize {
        self.states.len()
    }
}

// ─────────────────────────────────────────── LazyAdam ────────────────────────

/// Statistics about memory allocation in a `LazyAdam` optimizer.
#[derive(Debug, Clone)]
pub struct LazyOptimizerStats {
    /// Total number of parameter indices that have ever been seen.
    pub total_params: usize,
    /// Parameters whose state has been allocated (at least one gradient received).
    pub allocated_params: usize,
    /// Parameters that have been registered but whose state is still zero-sized.
    pub uninitialized_params: usize,
    /// Total bytes consumed by all parameter states.
    pub memory_bytes: usize,
    /// `allocated_params / total_params` (0.0 if `total_params == 0`).
    pub allocation_ratio: f32,
}

/// Adam optimizer with **lazy state allocation**.
///
/// Moment vectors (`exp_avg`, `exp_avg_sq`) are allocated only the first time a
/// gradient is observed for a given parameter index.  This is beneficial for
/// embedding tables and other sparse workloads where only a small fraction of
/// parameter rows are touched per batch.
///
/// The update rule is identical to AdamW:
/// ```text
/// exp_avg     ← β1 * exp_avg     + (1 − β1) * g
/// exp_avg_sq  ← β2 * exp_avg_sq  + (1 − β2) * g²
/// m̂ = exp_avg     / (1 − β1^t)
/// v̂ = exp_avg_sq  / (1 − β2^t)
/// θ  ← (1 − lr * wd) * θ − lr * m̂ / (√v̂ + ε)
/// ```
pub struct LazyAdam {
    lr: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    /// State per parameter, keyed by parameter index.
    states: HashMap<usize, LazyParamState>,
    /// Global step counter (incremented once per `step_param` call regardless of how many params).
    global_step: u64,
}

impl LazyAdam {
    /// Create a `LazyAdam` with explicit hyperparameters.
    pub fn new(lr: f64, beta1: f64, beta2: f64, epsilon: f64, weight_decay: f64) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            states: HashMap::new(),
            global_step: 0,
        }
    }

    /// Create a `LazyAdam` with a given learning rate and sensible defaults for other parameters.
    pub fn with_lr(lr: f64) -> Self {
        Self::new(lr, 0.9, 0.999, 1e-8, 0.01)
    }

    /// Apply one Adam update to a single parameter slice given its gradient.
    ///
    /// State is lazily allocated on the first call for each `param_idx`.
    /// The `param` slice is updated **in-place**.
    ///
    /// Returns an error if `param` and `grad` have different lengths.
    pub fn step_param(&mut self, param_idx: usize, param: &mut [f32], grad: &[f32]) -> Result<()> {
        if param.len() != grad.len() {
            return Err(TrustformersError::tensor_op_error(
                &format!("param length {} != grad length {}", param.len(), grad.len()),
                "LazyAdam::step_param",
            ));
        }

        let state = self.states.entry(param_idx).or_default();
        state.increment_step();
        let t = state.step() as f64;

        // Bias-correction factors
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        // Lazily allocate moment buffers on first use.
        let n = param.len();
        let exp_avg = state.get_or_init_vector("exp_avg", n, 0.0);
        let lr = self.lr;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let epsilon = self.epsilon;
        let weight_decay = self.weight_decay;

        let beta1_f = beta1 as f32;
        // First pass: update exp_avg (m)
        for (m, g) in exp_avg.iter_mut().zip(grad.iter()) {
            *m = beta1_f * (*m) + (1.0 - beta1_f) * g;
        }

        // We need to read exp_avg again and also work on exp_avg_sq.
        // Temporarily snapshot exp_avg to avoid borrow conflicts.
        let exp_avg_snapshot: Vec<f32> = state.get_or_init_vector("exp_avg", n, 0.0).to_vec();

        let exp_avg_sq = state.get_or_init_vector("exp_avg_sq", n, 0.0);

        // Update exp_avg_sq (v)
        for (v, g) in exp_avg_sq.iter_mut().zip(grad.iter()) {
            *v = beta2 as f32 * (*v) + (1.0 - beta2) as f32 * g * g;
        }

        let exp_avg_sq_snapshot: Vec<f32> = state.get_or_init_vector("exp_avg_sq", n, 0.0).to_vec();

        // Apply update to params
        let lr_f = lr as f32;
        let bc1_f = bc1 as f32;
        let bc2_f = bc2 as f32;
        let eps_f = epsilon as f32;
        let wd_f = weight_decay as f32;

        for i in 0..n {
            let m_hat = exp_avg_snapshot[i] / bc1_f;
            let v_hat = exp_avg_sq_snapshot[i] / bc2_f;
            // Decoupled weight decay
            param[i] = param[i] * (1.0 - lr_f * wd_f) - lr_f * m_hat / (v_hat.sqrt() + eps_f);
        }

        self.global_step += 1;
        Ok(())
    }

    /// Collect memory and allocation statistics.
    pub fn memory_stats(&self) -> LazyOptimizerStats {
        let total_params = self.states.len();
        let allocated_params = self.states.values().filter(|s| s.num_initialized() > 0).count();
        let uninitialized_params = total_params - allocated_params;
        let memory_bytes: usize = self.states.values().map(|s| s.memory_bytes()).sum();
        let allocation_ratio = if total_params == 0 {
            0.0
        } else {
            allocated_params as f32 / total_params as f32
        };
        LazyOptimizerStats {
            total_params,
            allocated_params,
            uninitialized_params,
            memory_bytes,
            allocation_ratio,
        }
    }

    /// Number of parameter indices for which state has been allocated.
    pub fn num_allocated_params(&self) -> usize {
        self.states.values().filter(|s| s.num_initialized() > 0).count()
    }

    /// Reset the optimizer state for a single parameter (useful for fine-tuning
    /// or when a parameter is re-initialized).
    pub fn reset_param(&mut self, param_idx: usize) {
        self.states.remove(&param_idx);
    }

    /// Current global step count.
    pub fn global_step(&self) -> u64 {
        self.global_step
    }
}

// ─────────────────────────────────────────── tests ───────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── LazyStateValue ──

    #[test]
    fn test_lazy_state_value_uninitialized() {
        let v = LazyStateValue::Uninitialized;
        assert!(!v.is_initialized());
        assert!(v.as_scalar().is_none());
        assert!(v.as_vector().is_none());
        assert_eq!(v.memory_bytes(), 0);
    }

    #[test]
    fn test_lazy_state_value_initialize_scalar() {
        let mut v = LazyStateValue::Uninitialized;
        v.initialize_scalar(std::f32::consts::PI);
        assert!(v.is_initialized());
        assert!((v.as_scalar().expect("scalar") - std::f32::consts::PI).abs() < 1e-6);
        assert_eq!(v.memory_bytes(), std::mem::size_of::<f32>());
    }

    #[test]
    fn test_lazy_state_value_initialize_vector() {
        let mut v = LazyStateValue::Uninitialized;
        v.initialize_vector(4, 0.0);
        assert!(v.is_initialized());
        let slice = v.as_vector().expect("vector");
        assert_eq!(slice.len(), 4);
        assert!(slice.iter().all(|&x| x == 0.0));
        assert_eq!(v.memory_bytes(), 4 * std::mem::size_of::<f32>());
    }

    #[test]
    fn test_lazy_state_value_vector_mut() {
        let mut v = LazyStateValue::Uninitialized;
        v.initialize_vector(3, 1.0);
        if let Some(vec) = v.as_vector_mut() {
            vec[0] = 42.0;
        }
        assert_eq!(v.as_vector().expect("vector")[0], 42.0);
    }

    // ── LazyParamState ──

    #[test]
    fn test_lazy_param_state_new_is_empty() {
        let s = LazyParamState::new();
        assert_eq!(s.step(), 0);
        assert_eq!(s.num_initialized(), 0);
        assert_eq!(s.memory_bytes(), 0);
    }

    #[test]
    fn test_lazy_param_state_get_or_init_scalar() {
        let mut s = LazyParamState::new();
        {
            let val = s.get_or_init_scalar("lr", 0.001);
            assert!((*val - 0.001).abs() < 1e-10);
            *val = 0.002;
        }
        // Second call should return the updated value
        let val2 = s.get_or_init_scalar("lr", 0.001);
        assert!((*val2 - 0.002).abs() < 1e-10);
    }

    #[test]
    fn test_lazy_param_state_get_or_init_vector() {
        let mut s = LazyParamState::new();
        {
            let v = s.get_or_init_vector("exp_avg", 5, 0.0);
            assert_eq!(v.len(), 5);
            v[2] = 7.0;
        }
        let v2 = s.get_or_init_vector("exp_avg", 5, 0.0);
        assert_eq!(v2[2], 7.0);
    }

    #[test]
    fn test_lazy_param_state_get_scalar_before_init() {
        let s = LazyParamState::new();
        assert!(s.get_scalar("missing").is_none());
    }

    #[test]
    fn test_lazy_param_state_memory_bytes_grows() {
        let mut s = LazyParamState::new();
        assert_eq!(s.memory_bytes(), 0);
        s.get_or_init_vector("exp_avg", 10, 0.0);
        assert_eq!(s.memory_bytes(), 10 * std::mem::size_of::<f32>());
        s.get_or_init_vector("exp_avg_sq", 10, 0.0);
        assert_eq!(s.memory_bytes(), 20 * std::mem::size_of::<f32>());
    }

    #[test]
    fn test_lazy_param_state_num_initialized() {
        let mut s = LazyParamState::new();
        s.get_or_init_vector("a", 1, 0.0);
        assert_eq!(s.num_initialized(), 1);
        s.get_or_init_scalar("b", 0.0);
        assert_eq!(s.num_initialized(), 2);
    }

    // ── LazyAdam ──

    #[test]
    fn test_lazy_adam_with_lr() {
        let opt = LazyAdam::with_lr(1e-3);
        assert_eq!(opt.num_allocated_params(), 0);
        assert_eq!(opt.global_step(), 0);
    }

    #[test]
    fn test_lazy_adam_step_param_allocates_state() {
        let mut opt = LazyAdam::with_lr(1e-3);
        let mut param = vec![1.0f32; 4];
        let grad = vec![0.1f32; 4];
        opt.step_param(0, &mut param, &grad).expect("step_param");
        assert_eq!(opt.num_allocated_params(), 1);
    }

    #[test]
    fn test_lazy_adam_step_param_updates_params() {
        let mut opt = LazyAdam::with_lr(1e-2);
        let mut param = vec![1.0f32; 3];
        let original = param.clone();
        let grad = vec![1.0f32; 3];
        opt.step_param(0, &mut param, &grad).expect("step_param");
        // Params should have decreased (gradient descent)
        for (orig, updated) in original.iter().zip(param.iter()) {
            assert!(updated < orig, "param should decrease after gradient step");
        }
    }

    #[test]
    fn test_lazy_adam_step_param_length_mismatch_error() {
        let mut opt = LazyAdam::with_lr(1e-3);
        let mut param = vec![1.0f32; 3];
        let grad = vec![0.1f32; 5]; // wrong length
        assert!(opt.step_param(0, &mut param, &grad).is_err());
    }

    #[test]
    fn test_lazy_adam_reset_param() {
        let mut opt = LazyAdam::with_lr(1e-3);
        let mut param = vec![1.0f32; 4];
        let grad = vec![0.1f32; 4];
        opt.step_param(0, &mut param, &grad).expect("step_param");
        assert_eq!(opt.num_allocated_params(), 1);
        opt.reset_param(0);
        assert_eq!(opt.num_allocated_params(), 0);
    }

    #[test]
    fn test_lazy_adam_memory_stats() {
        let mut opt = LazyAdam::with_lr(1e-3);
        let mut p0 = vec![0.0f32; 8];
        let g0 = vec![0.1f32; 8];
        opt.step_param(0, &mut p0, &g0).expect("step_param");
        let stats = opt.memory_stats();
        assert_eq!(stats.total_params, 1);
        assert_eq!(stats.allocated_params, 1);
        assert_eq!(stats.uninitialized_params, 0);
        assert!(stats.memory_bytes > 0);
        assert!((stats.allocation_ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_lazy_adam_sparse_allocation() {
        let mut opt = LazyAdam::with_lr(1e-3);
        // Only update param 5 out of hypothetical 100 params
        let mut param = vec![0.5f32; 4];
        let grad = vec![0.1f32; 4];
        opt.step_param(5, &mut param, &grad).expect("step_param");
        let stats = opt.memory_stats();
        // Only 1 param's state should be allocated
        assert_eq!(stats.allocated_params, 1);
    }

    #[test]
    fn test_lazy_adam_multiple_steps_convergence() {
        let mut opt = LazyAdam::new(0.1, 0.9, 0.999, 1e-8, 0.0);
        // Optimize simple quadratic: f(x) = x^2 → grad = 2x
        let mut param = vec![2.0f32];
        for _ in 0..50 {
            let grad = vec![2.0 * param[0]];
            opt.step_param(0, &mut param, &grad).expect("step_param");
        }
        assert!(
            param[0].abs() < 0.5,
            "Expected convergence toward 0, got {}",
            param[0]
        );
    }
}
