//! EWC: Elastic Weight Consolidation for continual learning
//!
//! Prevents catastrophic forgetting by penalising changes to weights
//! that were important for a previous task.  Importance is measured by
//! the diagonal Fisher Information Matrix (FIM).
//!
//! Reference: Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting
//! in neural networks", PNAS.

use std::fmt;

// ─── NamedParameter ──────────────────────────────────────────────────────────

/// A named parameter with its value (represents one weight tensor, flattened).
#[derive(Debug, Clone)]
pub struct NamedParameter {
    /// Unique name identifying the weight (e.g. `"encoder.layer.0.weight"`).
    pub name: String,
    /// Flat f32 values of the weight tensor.
    pub values: Vec<f32>,
}

impl NamedParameter {
    /// Construct a new named parameter.
    pub fn new(name: &str, values: Vec<f32>) -> Self {
        Self {
            name: name.to_owned(),
            values,
        }
    }

    /// Number of scalar elements.
    pub fn numel(&self) -> usize {
        self.values.len()
    }

    /// L2 norm of the parameter values.
    pub fn norm(&self) -> f32 {
        self.values.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }
}

// ─── FisherInformation ───────────────────────────────────────────────────────

/// Diagonal Fisher Information for one parameter tensor.
#[derive(Debug, Clone)]
pub struct FisherInformation {
    /// Same name as the corresponding `NamedParameter`.
    pub name: String,
    /// Per-element diagonal Fisher: F_i = E[(∂ log p / ∂θ_i)²].
    pub fisher: Vec<f32>,
}

impl FisherInformation {
    /// Mean Fisher information across all elements.
    pub fn mean(&self) -> f32 {
        if self.fisher.is_empty() {
            return 0.0;
        }
        self.fisher.iter().sum::<f32>() / self.fisher.len() as f32
    }

    /// Maximum Fisher information across all elements.
    pub fn max(&self) -> f32 {
        self.fisher.iter().cloned().fold(0.0_f32, f32::max)
    }

    /// Fraction of elements whose Fisher value exceeds `threshold`.
    pub fn important_fraction(&self, threshold: f32) -> f32 {
        if self.fisher.is_empty() {
            return 0.0;
        }
        let count = self.fisher.iter().filter(|&&f| f > threshold).count();
        count as f32 / self.fisher.len() as f32
    }
}

// ─── EwcConfig ───────────────────────────────────────────────────────────────

/// Configuration for EWC regularisation.
#[derive(Debug, Clone)]
pub struct EwcConfig {
    /// Regularisation strength λ.  Larger values anchor weights more tightly.
    pub lambda: f32,
    /// If `true`, use online EWC: accumulate Fisher information across tasks.
    pub online: bool,
    /// Decay factor γ for online EWC.  New Fisher = γ·old_F + new_F.
    pub gamma: f32,
}

impl Default for EwcConfig {
    fn default() -> Self {
        Self {
            lambda: 5000.0,
            online: false,
            gamma: 1.0,
        }
    }
}

// ─── EwcPenaltyResult ────────────────────────────────────────────────────────

/// Output of `compute_ewc_penalty`.
#[derive(Debug, Clone)]
pub struct EwcPenaltyResult {
    /// Summed EWC penalty across all parameters: (λ/2) Σ F_i (θ_i − θ*_i)².
    pub total_penalty: f32,
    /// Per-parameter penalty contribution `(name, penalty)`.
    pub per_param_penalty: Vec<(String, f32)>,
    /// Mean importance-weighted squared deviation (unnormalized by λ).
    pub mean_deviation: f32,
    /// Number of parameter tensors constrained.
    pub num_params: usize,
}

// ─── EwcError ────────────────────────────────────────────────────────────────

/// Errors returned by EWC operations.
#[derive(Debug)]
pub enum EwcError {
    /// At least one parameter slice was empty.
    EmptyParameters,
    /// The number of parameter tensors did not match.
    ParamCountMismatch { expected: usize, got: usize },
    /// Parameter name at position `i` did not match between slices.
    ParamNameMismatch { expected: String, got: String },
    /// A parameter tensor's element count did not match the Fisher vector.
    SizeMismatch { param: String, expected: usize, got: usize },
    /// Online EWC attempted to update Fisher but no anchors are registered.
    NoAnchors,
}

impl fmt::Display for EwcError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EwcError::EmptyParameters => {
                write!(f, "EwcError: parameter slices are empty")
            }
            EwcError::ParamCountMismatch { expected, got } => {
                write!(
                    f,
                    "EwcError: parameter count mismatch — expected {expected}, got {got}"
                )
            }
            EwcError::ParamNameMismatch { expected, got } => {
                write!(
                    f,
                    "EwcError: parameter name mismatch — expected '{expected}', got '{got}'"
                )
            }
            EwcError::SizeMismatch {
                param,
                expected,
                got,
            } => {
                write!(
                    f,
                    "EwcError: size mismatch for '{param}' — expected {expected} elements, got {got}"
                )
            }
            EwcError::NoAnchors => {
                write!(f, "EwcError: no task anchors registered")
            }
        }
    }
}

impl std::error::Error for EwcError {}

// ─── compute_fisher ──────────────────────────────────────────────────────────

/// Compute diagonal Fisher Information from squared gradients.
///
/// In practice the caller accumulates `(∂L/∂θ_i)²` over a mini-batch and
/// passes the mean as `gradients_squared`.  The empirical Fisher is then
/// simply those averaged squared gradients.
pub fn compute_fisher(gradients_squared: &[NamedParameter]) -> Vec<FisherInformation> {
    gradients_squared
        .iter()
        .map(|g| FisherInformation {
            name: g.name.clone(),
            fisher: g.values.clone(),
        })
        .collect()
}

// ─── compute_ewc_penalty ─────────────────────────────────────────────────────

/// Compute the EWC penalty for `current_params` given anchors and Fisher info.
///
/// For each parameter *i*:
/// ```text
/// deviation_i = Σ_j  F_i[j] · (θ_i[j] − θ*_i[j])²
/// penalty_i   = (λ/2) · deviation_i
/// ```
pub fn compute_ewc_penalty(
    current_params: &[NamedParameter],
    anchor_params: &[NamedParameter],
    fisher: &[FisherInformation],
    config: &EwcConfig,
) -> Result<EwcPenaltyResult, EwcError> {
    // ── length validation ────────────────────────────────────────────────────
    if current_params.is_empty() {
        return Err(EwcError::EmptyParameters);
    }
    if current_params.len() != anchor_params.len() {
        return Err(EwcError::ParamCountMismatch {
            expected: anchor_params.len(),
            got: current_params.len(),
        });
    }
    if current_params.len() != fisher.len() {
        return Err(EwcError::ParamCountMismatch {
            expected: fisher.len(),
            got: current_params.len(),
        });
    }

    let mut per_param_penalty: Vec<(String, f32)> = Vec::with_capacity(current_params.len());
    let mut total_penalty = 0.0_f32;
    let mut total_deviation = 0.0_f32;

    for ((cur, anc), fi) in current_params
        .iter()
        .zip(anchor_params.iter())
        .zip(fisher.iter())
    {
        // ── name matching ────────────────────────────────────────────────────
        if cur.name != anc.name {
            return Err(EwcError::ParamNameMismatch {
                expected: anc.name.clone(),
                got: cur.name.clone(),
            });
        }
        if cur.name != fi.name {
            return Err(EwcError::ParamNameMismatch {
                expected: fi.name.clone(),
                got: cur.name.clone(),
            });
        }

        // ── size matching ────────────────────────────────────────────────────
        if cur.values.len() != anc.values.len() {
            return Err(EwcError::SizeMismatch {
                param: cur.name.clone(),
                expected: anc.values.len(),
                got: cur.values.len(),
            });
        }
        if cur.values.len() != fi.fisher.len() {
            return Err(EwcError::SizeMismatch {
                param: cur.name.clone(),
                expected: fi.fisher.len(),
                got: cur.values.len(),
            });
        }

        // ── deviation ────────────────────────────────────────────────────────
        let deviation: f32 = cur
            .values
            .iter()
            .zip(anc.values.iter())
            .zip(fi.fisher.iter())
            .map(|((&c, &a), &f_val)| f_val * (c - a) * (c - a))
            .sum();

        let penalty = 0.5 * config.lambda * deviation;
        total_penalty += penalty;
        total_deviation += deviation;

        per_param_penalty.push((cur.name.clone(), penalty));
    }

    let mean_deviation = total_deviation / current_params.len() as f32;

    Ok(EwcPenaltyResult {
        total_penalty,
        per_param_penalty,
        mean_deviation,
        num_params: current_params.len(),
    })
}

// ─── EwcTaskAnchor ───────────────────────────────────────────────────────────

/// Stores the anchored weights and Fisher information for one completed task.
#[derive(Debug, Clone)]
pub struct EwcTaskAnchor {
    /// Human-readable name for the task (e.g. `"task_1_sst2"`).
    pub task_name: String,
    /// Optimal parameters θ* at the end of this task.
    pub anchor_params: Vec<NamedParameter>,
    /// Diagonal Fisher information estimated on this task.
    pub fisher: Vec<FisherInformation>,
}

impl EwcTaskAnchor {
    /// Construct a new task anchor.
    pub fn new(
        task_name: &str,
        anchor_params: Vec<NamedParameter>,
        fisher: Vec<FisherInformation>,
    ) -> Self {
        Self {
            task_name: task_name.to_owned(),
            anchor_params,
            fisher,
        }
    }

    /// Number of parameter tensors stored.
    pub fn num_params(&self) -> usize {
        self.anchor_params.len()
    }

    /// Total number of scalar elements across all parameters.
    pub fn total_numel(&self) -> usize {
        self.anchor_params.iter().map(|p| p.numel()).sum()
    }
}

// ─── EwcTrainer ──────────────────────────────────────────────────────────────

/// Manages EWC anchors for multiple tasks and computes the combined penalty.
pub struct EwcTrainer {
    /// EWC configuration.
    pub config: EwcConfig,
    anchors: Vec<EwcTaskAnchor>,
    penalty_history: Vec<f32>,
}

impl EwcTrainer {
    /// Create a new `EwcTrainer` with the given configuration.
    pub fn new(config: EwcConfig) -> Self {
        Self {
            config,
            anchors: Vec::new(),
            penalty_history: Vec::new(),
        }
    }

    /// Register a completed task.
    ///
    /// `anchor_params` are the optimal weights θ* at the end of the task.
    /// `gradients_squared` are the mean-squared gradients used to estimate
    /// the Fisher information.
    pub fn register_task(
        &mut self,
        task_name: &str,
        anchor_params: Vec<NamedParameter>,
        gradients_squared: &[NamedParameter],
    ) -> Result<(), EwcError> {
        let fisher = compute_fisher(gradients_squared);

        if self.config.online && !self.anchors.is_empty() {
            // Online EWC: blend new Fisher into the most recent anchor.
            let last = self.anchors.last_mut().ok_or(EwcError::NoAnchors)?;
            for (old_f, new_f) in last.fisher.iter_mut().zip(fisher.iter()) {
                for (o, &n) in old_f.fisher.iter_mut().zip(new_f.fisher.iter()) {
                    *o = self.config.gamma * *o + n;
                }
            }
        } else {
            self.anchors
                .push(EwcTaskAnchor::new(task_name, anchor_params, fisher));
        }
        Ok(())
    }

    /// Compute the total EWC penalty for `current_params` across all tasks.
    ///
    /// The result is appended to the internal penalty history.
    pub fn total_penalty(&mut self, current_params: &[NamedParameter]) -> Result<f32, EwcError> {
        let mut total = 0.0_f32;
        for anchor in &self.anchors {
            let result = compute_ewc_penalty(
                current_params,
                &anchor.anchor_params,
                &anchor.fisher,
                &self.config,
            )?;
            total += result.total_penalty;
        }
        self.penalty_history.push(total);
        Ok(total)
    }

    /// Number of registered task anchors.
    pub fn num_tasks(&self) -> usize {
        self.anchors.len()
    }

    /// Full history of penalty values returned by `total_penalty`.
    pub fn penalty_history(&self) -> &[f32] {
        &self.penalty_history
    }

    /// All registered task anchors.
    pub fn anchors(&self) -> &[EwcTaskAnchor] {
        &self.anchors
    }

    /// Mean penalty over the recorded history.
    pub fn mean_penalty(&self) -> f32 {
        if self.penalty_history.is_empty() {
            return 0.0;
        }
        self.penalty_history.iter().sum::<f32>() / self.penalty_history.len() as f32
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn param(name: &str, v: Vec<f32>) -> NamedParameter {
        NamedParameter::new(name, v)
    }

    fn make_params(n: usize, val: f32) -> Vec<NamedParameter> {
        (0..n)
            .map(|i| param(&format!("p{i}"), vec![val; 4]))
            .collect()
    }

    // ── NamedParameter ───────────────────────────────────────────────────────

    #[test]
    fn test_named_parameter_norm() {
        // [3, 4] → norm = 5
        let p = param("w", vec![3.0, 4.0]);
        let diff = (p.norm() - 5.0_f32).abs();
        assert!(diff < 1e-5, "expected norm ≈ 5, got {}", p.norm());
    }

    #[test]
    fn test_named_parameter_numel() {
        let p = param("w", vec![1.0, 2.0, 3.0]);
        assert_eq!(p.numel(), 3);
    }

    // ── FisherInformation ────────────────────────────────────────────────────

    #[test]
    fn test_fisher_information_mean() {
        let fi = FisherInformation {
            name: "p0".into(),
            fisher: vec![1.0, 2.0, 3.0, 4.0],
        };
        let diff = (fi.mean() - 2.5_f32).abs();
        assert!(diff < 1e-5, "expected mean ≈ 2.5, got {}", fi.mean());
    }

    #[test]
    fn test_fisher_information_max() {
        let fi = FisherInformation {
            name: "p0".into(),
            fisher: vec![0.5, 9.0, 3.0],
        };
        let diff = (fi.max() - 9.0_f32).abs();
        assert!(diff < 1e-5, "expected max ≈ 9.0, got {}", fi.max());
    }

    #[test]
    fn test_fisher_information_important_fraction() {
        let fi = FisherInformation {
            name: "p0".into(),
            fisher: vec![0.1, 0.5, 1.0, 2.0],
        };
        // threshold = 0.4 → 3 out of 4 exceed it
        let frac = fi.important_fraction(0.4);
        let diff = (frac - 0.75_f32).abs();
        assert!(diff < 1e-5, "expected fraction ≈ 0.75, got {frac}");
    }

    #[test]
    fn test_fisher_information_important_fraction_none() {
        let fi = FisherInformation {
            name: "p0".into(),
            fisher: vec![0.1, 0.2],
        };
        assert_eq!(fi.important_fraction(1.0), 0.0);
    }

    // ── compute_fisher ───────────────────────────────────────────────────────

    #[test]
    fn test_compute_fisher_basic() {
        let grads = vec![
            param("a", vec![1.0, 4.0]),
            param("b", vec![9.0, 0.25]),
        ];
        let fisher = compute_fisher(&grads);
        assert_eq!(fisher.len(), 2);
        assert_eq!(fisher[0].name, "a");
        assert_eq!(fisher[0].fisher, vec![1.0, 4.0]);
        assert_eq!(fisher[1].name, "b");
        assert_eq!(fisher[1].fisher, vec![9.0, 0.25]);
    }

    // ── EwcConfig ────────────────────────────────────────────────────────────

    #[test]
    fn test_ewc_config_default() {
        let cfg = EwcConfig::default();
        assert!((cfg.lambda - 5000.0).abs() < 1e-5);
        assert!(!cfg.online);
        assert!((cfg.gamma - 1.0).abs() < 1e-5);
    }

    // ── compute_ewc_penalty ──────────────────────────────────────────────────

    #[test]
    fn test_ewc_penalty_zero_deviation() {
        // current == anchor → penalty must be 0
        let cur = vec![param("w", vec![1.0, 2.0, 3.0])];
        let anc = vec![param("w", vec![1.0, 2.0, 3.0])];
        let fi = compute_fisher(&[param("w", vec![1.0, 1.0, 1.0])]);
        let cfg = EwcConfig::default();
        let res = compute_ewc_penalty(&cur, &anc, &fi, &cfg).expect("penalty");
        assert!(res.total_penalty.abs() < 1e-6);
    }

    #[test]
    fn test_ewc_penalty_nonzero_deviation() {
        // deviation = F * (θ − θ*)²
        // F=[1], θ=[2], θ*=[1] → deviation=1 → penalty = (5000/2)*1 = 2500
        let cur = vec![param("w", vec![2.0])];
        let anc = vec![param("w", vec![1.0])];
        let fi = compute_fisher(&[param("w", vec![1.0])]);
        let cfg = EwcConfig::default();
        let res = compute_ewc_penalty(&cur, &anc, &fi, &cfg).expect("penalty");
        let diff = (res.total_penalty - 2500.0_f32).abs();
        assert!(diff < 1e-2, "expected ≈2500, got {}", res.total_penalty);
    }

    #[test]
    fn test_ewc_penalty_lambda_scaling() {
        let cur = vec![param("w", vec![2.0])];
        let anc = vec![param("w", vec![1.0])];
        let fi = compute_fisher(&[param("w", vec![1.0])]);

        let cfg_a = EwcConfig { lambda: 1.0, ..Default::default() };
        let cfg_b = EwcConfig { lambda: 10.0, ..Default::default() };
        let res_a = compute_ewc_penalty(&cur, &anc, &fi, &cfg_a).expect("a");
        let res_b = compute_ewc_penalty(&cur, &anc, &fi, &cfg_b).expect("b");

        let ratio = res_b.total_penalty / res_a.total_penalty;
        assert!((ratio - 10.0).abs() < 1e-4, "ratio should be 10, got {ratio}");
    }

    #[test]
    fn test_ewc_penalty_fisher_weighting() {
        // higher Fisher → higher penalty for the same deviation
        let cur = vec![param("w", vec![2.0])];
        let anc = vec![param("w", vec![1.0])];
        let fi_low = compute_fisher(&[param("w", vec![1.0])]);
        let fi_high = compute_fisher(&[param("w", vec![10.0])]);

        let cfg = EwcConfig { lambda: 1.0, ..Default::default() };
        let res_low = compute_ewc_penalty(&cur, &anc, &fi_low, &cfg).expect("low");
        let res_high = compute_ewc_penalty(&cur, &anc, &fi_high, &cfg).expect("high");

        assert!(res_high.total_penalty > res_low.total_penalty);
    }

    #[test]
    fn test_ewc_penalty_per_param() {
        let cur = vec![
            param("w1", vec![2.0]),
            param("w2", vec![3.0]),
        ];
        let anc = vec![
            param("w1", vec![1.0]),
            param("w2", vec![1.0]),
        ];
        let fi = compute_fisher(&[param("w1", vec![1.0]), param("w2", vec![1.0])]);
        let cfg = EwcConfig { lambda: 2.0, ..Default::default() };
        let res = compute_ewc_penalty(&cur, &anc, &fi, &cfg).expect("penalty");

        assert_eq!(res.per_param_penalty.len(), 2);
        assert_eq!(res.per_param_penalty[0].0, "w1");
        assert_eq!(res.per_param_penalty[1].0, "w2");
        // w1: (λ/2)*F*(θ−θ*)² = 1.0 * 1.0 * 1.0 = 1.0
        // w2: (λ/2)*F*(θ−θ*)² = 1.0 * 1.0 * 4.0 = 4.0
        assert!((res.per_param_penalty[0].1 - 1.0).abs() < 1e-4);
        assert!((res.per_param_penalty[1].1 - 4.0).abs() < 1e-4);
    }

    // ── EwcTaskAnchor ────────────────────────────────────────────────────────

    #[test]
    fn test_ewc_task_anchor_new() {
        let params = make_params(3, 1.0);
        let fi = compute_fisher(&params);
        let anchor = EwcTaskAnchor::new("task1", params, fi);
        assert_eq!(anchor.task_name, "task1");
        assert_eq!(anchor.num_params(), 3);
        assert_eq!(anchor.total_numel(), 12); // 3 params × 4 elements
    }

    // ── EwcTrainer ───────────────────────────────────────────────────────────

    #[test]
    fn test_ewc_trainer_register_task() {
        let mut trainer = EwcTrainer::new(EwcConfig::default());
        let params = make_params(2, 1.0);
        let grads = make_params(2, 0.5);
        trainer.register_task("t1", params, &grads).expect("register");
        assert_eq!(trainer.num_tasks(), 1);
    }

    #[test]
    fn test_ewc_trainer_total_penalty() {
        let mut trainer = EwcTrainer::new(EwcConfig { lambda: 2.0, ..Default::default() });
        let anchor = make_params(1, 1.0);
        let grads = vec![param("p0", vec![1.0; 4])];
        trainer.register_task("t1", anchor, &grads).expect("register");

        // current params deviate by 1 per element
        let current = vec![param("p0", vec![2.0; 4])];
        let penalty = trainer.total_penalty(&current).expect("penalty");
        // (2/2) * 1 * 4 = 4
        assert!((penalty - 4.0).abs() < 1e-4, "expected 4.0, got {penalty}");
    }

    #[test]
    fn test_ewc_trainer_multi_task() {
        let mut trainer = EwcTrainer::new(EwcConfig { lambda: 1.0, ..Default::default() });
        let grads = vec![param("p0", vec![1.0; 2])];

        trainer
            .register_task("t1", vec![param("p0", vec![0.0; 2])], &grads)
            .expect("t1");
        trainer
            .register_task("t2", vec![param("p0", vec![0.0; 2])], &grads)
            .expect("t2");

        assert_eq!(trainer.num_tasks(), 2);

        // current = [1, 1] → deviation per anchor = 2 × (1−0)² × 1 = 2
        // two tasks → total = 2 × (0.5 × 1 × 2) = 2
        let current = vec![param("p0", vec![1.0; 2])];
        let penalty = trainer.total_penalty(&current).expect("penalty");
        assert!((penalty - 2.0).abs() < 1e-4, "expected 2.0, got {penalty}");
    }

    #[test]
    fn test_ewc_trainer_online_mode() {
        let cfg = EwcConfig {
            lambda: 1.0,
            online: true,
            gamma: 1.0,
        };
        let mut trainer = EwcTrainer::new(cfg);
        let grads1 = vec![param("p0", vec![1.0])];
        let grads2 = vec![param("p0", vec![1.0])];

        trainer
            .register_task("t1", vec![param("p0", vec![0.0])], &grads1)
            .expect("t1");
        // Online mode: second task blends into the first anchor instead of adding a new one
        trainer
            .register_task("t2", vec![param("p0", vec![0.0])], &grads2)
            .expect("t2");

        assert_eq!(trainer.num_tasks(), 1);
        // Fisher should have been doubled (1 + 1)
        assert!((trainer.anchors()[0].fisher[0].fisher[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_ewc_trainer_penalty_history() {
        let mut trainer = EwcTrainer::new(EwcConfig { lambda: 1.0, ..Default::default() });
        let anchor = vec![param("p0", vec![0.0])];
        let grads = vec![param("p0", vec![1.0])];
        trainer.register_task("t1", anchor, &grads).expect("register");

        let cur = vec![param("p0", vec![1.0])];
        let _ = trainer.total_penalty(&cur).expect("p1");
        let _ = trainer.total_penalty(&cur).expect("p2");

        assert_eq!(trainer.penalty_history().len(), 2);
        let mean = trainer.mean_penalty();
        // Both calls yield the same penalty, so mean == that value
        assert!((mean - trainer.penalty_history()[0]).abs() < 1e-5);
    }

    // ── EwcError display ─────────────────────────────────────────────────────

    #[test]
    fn test_ewc_error_display() {
        let e = EwcError::EmptyParameters;
        assert!(e.to_string().contains("empty"));

        let e = EwcError::ParamCountMismatch { expected: 3, got: 2 };
        let s = e.to_string();
        assert!(s.contains("3") && s.contains("2"), "got: {s}");

        let e = EwcError::ParamNameMismatch {
            expected: "foo".into(),
            got: "bar".into(),
        };
        let s = e.to_string();
        assert!(s.contains("foo") && s.contains("bar"), "got: {s}");

        let e = EwcError::SizeMismatch {
            param: "w".into(),
            expected: 4,
            got: 3,
        };
        let s = e.to_string();
        assert!(s.contains("w") && s.contains("4") && s.contains("3"), "got: {s}");

        let e = EwcError::NoAnchors;
        assert!(e.to_string().contains("anchor"));
    }

    // ── Additional tests 21-43 ────────────────────────────────────────────

    // 21. Fisher computation preserves values from squared gradients
    #[test]
    fn test_compute_fisher_preserves_values() {
        let grads = vec![
            param("w0", vec![0.04, 0.09, 0.25]),
            param("w1", vec![1.0, 0.0, 0.01]),
        ];
        let fisher = compute_fisher(&grads);
        assert_eq!(fisher[0].fisher, grads[0].values, "Fisher must equal squared grad values for w0");
        assert_eq!(fisher[1].fisher, grads[1].values, "Fisher must equal squared grad values for w1");
    }

    // 22. EWC penalty formula: Σ F_i * (θ_i - θ*_i)^2 / 2 with known values
    #[test]
    fn test_ewc_penalty_formula_known_values() {
        // F=[2.0, 4.0], θ=[3.0, 5.0], θ*=[1.0, 2.0]
        // deviation = 2*(3-1)^2 + 4*(5-2)^2 = 2*4 + 4*9 = 8 + 36 = 44
        // penalty = (λ/2)*44 = (1.0/2)*44 = 22.0
        let cur = vec![param("w", vec![3.0, 5.0])];
        let anc = vec![param("w", vec![1.0, 2.0])];
        let fi = compute_fisher(&[param("w", vec![2.0, 4.0])]);
        let cfg = EwcConfig { lambda: 1.0, ..Default::default() };
        let res = compute_ewc_penalty(&cur, &anc, &fi, &cfg).expect("penalty");
        assert!((res.total_penalty - 22.0).abs() < 1e-4, "expected 22.0, got {}", res.total_penalty);
    }

    // 23. Online EWC with gamma=0.5: blended Fisher = 0.5 * old + new
    #[test]
    fn test_online_ewc_gamma_decay() {
        let cfg = EwcConfig { lambda: 1.0, online: true, gamma: 0.5 };
        let mut trainer = EwcTrainer::new(cfg);

        // Register first task with Fisher = [4.0]
        let grads1 = vec![param("p0", vec![4.0])];
        trainer.register_task("t1", vec![param("p0", vec![0.0])], &grads1).expect("t1");

        // Register second task with Fisher = [2.0]
        // Online: blended Fisher = 0.5 * 4.0 + 2.0 = 4.0
        let grads2 = vec![param("p0", vec![2.0])];
        trainer.register_task("t2", vec![param("p0", vec![0.0])], &grads2).expect("t2");

        assert_eq!(trainer.num_tasks(), 1, "online mode should keep only 1 anchor");
        let blended = trainer.anchors()[0].fisher[0].fisher[0];
        assert!(
            (blended - 4.0).abs() < 1e-5,
            "blended Fisher = 0.5*4 + 2 = 4.0, got {blended}"
        );
    }

    // 24. Multi-task EWC: separate Fisher per task, total penalty sums contributions
    #[test]
    fn test_multi_task_ewc_separate_anchors() {
        let mut trainer = EwcTrainer::new(EwcConfig { lambda: 2.0, ..Default::default() });
        let grads = vec![param("p0", vec![1.0])]; // Fisher = [1.0]

        trainer.register_task("t1", vec![param("p0", vec![0.0])], &grads).expect("t1");
        trainer.register_task("t2", vec![param("p0", vec![1.0])], &grads).expect("t2");
        trainer.register_task("t3", vec![param("p0", vec![2.0])], &grads).expect("t3");

        assert_eq!(trainer.num_tasks(), 3, "3 tasks should be registered separately");

        // current = [3.0]
        // penalty for t1 anchor(0): (2/2)*1*(3-0)^2 = 9
        // penalty for t2 anchor(1): (2/2)*1*(3-1)^2 = 4
        // penalty for t3 anchor(2): (2/2)*1*(3-2)^2 = 1
        // total = 14
        let current = vec![param("p0", vec![3.0])];
        let total = trainer.total_penalty(&current).expect("penalty");
        assert!((total - 14.0).abs() < 1e-3, "expected 14.0, got {total}");
    }

    // 25. Penalty grows with larger parameter divergence
    #[test]
    fn test_penalty_grows_with_divergence() {
        let mut trainer = EwcTrainer::new(EwcConfig { lambda: 1.0, ..Default::default() });
        let grads = vec![param("p0", vec![1.0])];
        trainer.register_task("t1", vec![param("p0", vec![0.0])], &grads).expect("t1");

        let small_dev = vec![param("p0", vec![0.1])]; // small deviation
        let large_dev = vec![param("p0", vec![5.0])]; // large deviation

        let p_small = trainer.total_penalty(&small_dev).expect("small");
        let p_large = trainer.total_penalty(&large_dev).expect("large");

        assert!(p_small < p_large, "larger deviation should produce larger penalty: {} vs {}", p_small, p_large);
    }

    // 26. Zero penalty when current == anchor (per-param check)
    #[test]
    fn test_zero_penalty_when_current_equals_anchor() {
        let anchor_vals = vec![0.7, -1.2, 3.5, 0.0];
        let cur = vec![param("w", anchor_vals.clone())];
        let anc = vec![param("w", anchor_vals)];
        let fi = compute_fisher(&[param("w", vec![100.0, 100.0, 100.0, 100.0])]);
        let cfg = EwcConfig { lambda: 9999.0, ..Default::default() };
        let res = compute_ewc_penalty(&cur, &anc, &fi, &cfg).expect("penalty");
        assert!(res.total_penalty.abs() < 1e-6, "penalty must be 0 at anchor, got {}", res.total_penalty);
        assert!(res.per_param_penalty[0].1.abs() < 1e-6, "per-param penalty must be 0 at anchor");
    }

    // 27. EwcTrainer with lambda=0 → penalty is always 0
    #[test]
    fn test_zero_lambda_zero_penalty() {
        let mut trainer = EwcTrainer::new(EwcConfig { lambda: 0.0, ..Default::default() });
        let grads = vec![param("p0", vec![1.0, 2.0])];
        trainer.register_task("t1", vec![param("p0", vec![0.0; 2])], &grads).expect("t1");
        let current = vec![param("p0", vec![999.0, 999.0])];
        let penalty = trainer.total_penalty(&current).expect("penalty");
        assert!(penalty.abs() < 1e-6, "lambda=0 should give zero penalty, got {penalty}");
    }

    // 28. compute_fisher with multiple params preserves all names and lengths
    #[test]
    fn test_compute_fisher_multiple_params_names() {
        let grads: Vec<NamedParameter> = (0..5)
            .map(|i| param(&format!("layer.{i}.weight"), vec![0.1 * (i as f32 + 1.0); 3]))
            .collect();
        let fisher = compute_fisher(&grads);
        assert_eq!(fisher.len(), 5);
        for (i, fi) in fisher.iter().enumerate() {
            assert_eq!(fi.name, format!("layer.{i}.weight"));
            assert_eq!(fi.fisher.len(), 3);
        }
    }

    // 29. FisherInformation::mean on empty → 0.0
    #[test]
    fn test_fisher_mean_empty() {
        let fi = FisherInformation { name: "empty".into(), fisher: vec![] };
        assert_eq!(fi.mean(), 0.0, "mean of empty Fisher should be 0.0");
    }

    // 30. FisherInformation::important_fraction on empty → 0.0
    #[test]
    fn test_fisher_important_fraction_empty() {
        let fi = FisherInformation { name: "empty".into(), fisher: vec![] };
        assert_eq!(fi.important_fraction(0.5), 0.0, "fraction on empty should be 0.0");
    }

    // 31. compute_ewc_penalty error: ParamCountMismatch (current != anchor count)
    #[test]
    fn test_ewc_penalty_param_count_mismatch() {
        let cur = vec![param("w0", vec![1.0]), param("w1", vec![1.0])];
        let anc = vec![param("w0", vec![1.0])];
        let fi = compute_fisher(&[param("w0", vec![1.0])]);
        let cfg = EwcConfig::default();
        let err = compute_ewc_penalty(&cur, &anc, &fi, &cfg).expect_err("should fail");
        assert!(matches!(err, EwcError::ParamCountMismatch { .. }));
    }

    // 32. compute_ewc_penalty error: ParamNameMismatch
    #[test]
    fn test_ewc_penalty_param_name_mismatch() {
        let cur = vec![param("weight_a", vec![1.0])];
        let anc = vec![param("weight_b", vec![1.0])];
        let fi = compute_fisher(&[param("weight_a", vec![1.0])]);
        let cfg = EwcConfig::default();
        let err = compute_ewc_penalty(&cur, &anc, &fi, &cfg).expect_err("name mismatch");
        assert!(matches!(err, EwcError::ParamNameMismatch { .. }));
    }

    // 33. compute_ewc_penalty error: SizeMismatch (current element count != fisher)
    #[test]
    fn test_ewc_penalty_size_mismatch() {
        let cur = vec![param("w", vec![1.0, 2.0, 3.0])];
        let anc = vec![param("w", vec![1.0, 2.0, 3.0])];
        // Fisher has only 2 elements but param has 3
        let fi = vec![FisherInformation { name: "w".into(), fisher: vec![1.0, 1.0] }];
        let cfg = EwcConfig::default();
        let err = compute_ewc_penalty(&cur, &anc, &fi, &cfg).expect_err("size mismatch");
        assert!(matches!(err, EwcError::SizeMismatch { .. }));
    }

    // 34. compute_ewc_penalty error: EmptyParameters
    #[test]
    fn test_ewc_penalty_empty_parameters_error() {
        let fi: Vec<FisherInformation> = vec![];
        let cfg = EwcConfig::default();
        let err = compute_ewc_penalty(&[], &[], &fi, &cfg).expect_err("empty params");
        assert!(matches!(err, EwcError::EmptyParameters));
    }

    // 35. EwcTrainer::mean_penalty on empty history → 0.0
    #[test]
    fn test_ewc_trainer_mean_penalty_empty_history() {
        let trainer = EwcTrainer::new(EwcConfig::default());
        assert_eq!(trainer.mean_penalty(), 0.0, "mean penalty with no history should be 0.0");
    }

    // 36. EwcTaskAnchor::total_numel for multi-param anchor
    #[test]
    fn test_ewc_task_anchor_total_numel() {
        let params = vec![
            param("w0", vec![1.0; 10]),
            param("w1", vec![1.0; 5]),
            param("w2", vec![1.0; 3]),
        ];
        let fi = compute_fisher(&params);
        let anchor = EwcTaskAnchor::new("task", params, fi);
        assert_eq!(anchor.total_numel(), 18, "total_numel should be 10+5+3=18");
    }

    // 37. NamedParameter::norm for zero vector → 0.0
    #[test]
    fn test_named_parameter_norm_zero_vector() {
        let p = param("zeros", vec![0.0, 0.0, 0.0, 0.0]);
        assert!((p.norm() - 0.0).abs() < 1e-6, "norm of zero vector should be 0.0");
    }

    // 38. Online EWC with gamma=0.0: new Fisher replaces old
    #[test]
    fn test_online_ewc_gamma_zero_replaces() {
        let cfg = EwcConfig { lambda: 1.0, online: true, gamma: 0.0 };
        let mut trainer = EwcTrainer::new(cfg);

        // First task: Fisher = [8.0]
        trainer.register_task("t1", vec![param("p0", vec![0.0])], &[param("p0", vec![8.0])]).expect("t1");

        // Second task: Fisher = [3.0]; blended = 0.0 * 8.0 + 3.0 = 3.0 (old is zeroed out)
        trainer.register_task("t2", vec![param("p0", vec![0.0])], &[param("p0", vec![3.0])]).expect("t2");

        let blended = trainer.anchors()[0].fisher[0].fisher[0];
        assert!((blended - 3.0).abs() < 1e-5, "gamma=0 → old Fisher discarded, expected 3.0, got {blended}");
    }

    // 39. Non-online EwcTrainer preserves all tasks separately
    #[test]
    fn test_non_online_trainer_keeps_all_tasks() {
        let cfg = EwcConfig { lambda: 1.0, online: false, gamma: 1.0 };
        let mut trainer = EwcTrainer::new(cfg);
        for i in 0..5 {
            let grads = vec![param("p0", vec![1.0])];
            trainer.register_task(&format!("task_{i}"), vec![param("p0", vec![0.0])], &grads)
                .expect("register");
        }
        assert_eq!(trainer.num_tasks(), 5, "5 tasks should be stored separately");
    }

    // 40. EwcPenaltyResult fields: check num_params and mean_deviation
    #[test]
    fn test_ewc_penalty_result_fields() {
        let cur = vec![
            param("w0", vec![2.0, 3.0]),
            param("w1", vec![1.0]),
        ];
        let anc = vec![
            param("w0", vec![0.0, 0.0]),
            param("w1", vec![0.0]),
        ];
        let fi = compute_fisher(&[param("w0", vec![1.0, 1.0]), param("w1", vec![1.0])]);
        let cfg = EwcConfig { lambda: 1.0, ..Default::default() };
        let res = compute_ewc_penalty(&cur, &anc, &fi, &cfg).expect("penalty");

        assert_eq!(res.num_params, 2, "num_params should be 2");
        // deviation_w0 = 1*(2^2) + 1*(3^2) = 4+9 = 13
        // deviation_w1 = 1*(1^2) = 1
        // mean_deviation = (13+1)/2 = 7.0
        assert!((res.mean_deviation - 7.0).abs() < 1e-4, "expected mean_deviation=7.0, got {}", res.mean_deviation);
    }

    // 41. Online EWC with multiple tasks changes fisher sum
    #[test]
    fn test_online_ewc_accumulates_fisher() {
        let cfg = EwcConfig { lambda: 1.0, online: true, gamma: 1.0 };
        let mut trainer = EwcTrainer::new(cfg);

        // Add 3 tasks with Fisher = [1.0] each; accumulated = 3.0
        for i in 0..3 {
            trainer.register_task(
                &format!("t{i}"),
                vec![param("p0", vec![0.0])],
                &[param("p0", vec![1.0])],
            ).expect("register");
        }
        assert_eq!(trainer.num_tasks(), 1, "online mode should keep 1 anchor");
        let accumulated = trainer.anchors()[0].fisher[0].fisher[0];
        assert!(
            (accumulated - 3.0).abs() < 1e-5,
            "accumulated Fisher with gamma=1 should be 3.0, got {accumulated}"
        );
    }

    // 42. EwcConfig gamma default is 1.0 (no decay)
    #[test]
    fn test_ewc_config_gamma_default() {
        let cfg = EwcConfig::default();
        assert!((cfg.gamma - 1.0).abs() < 1e-6, "default gamma should be 1.0");
    }

    // 43. Penalty history grows with each call to total_penalty
    #[test]
    fn test_penalty_history_grows() {
        let mut trainer = EwcTrainer::new(EwcConfig { lambda: 1.0, ..Default::default() });
        let grads = vec![param("p0", vec![1.0])];
        trainer.register_task("t1", vec![param("p0", vec![0.0])], &grads).expect("register");

        let cur = vec![param("p0", vec![1.0])];
        for i in 1..=5 {
            trainer.total_penalty(&cur).expect("penalty");
            assert_eq!(trainer.penalty_history().len(), i, "history length should be {i}");
        }
    }
}
