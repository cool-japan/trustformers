//! Optimizer surgery: change optimizer type mid-training while preserving relevant state.
//!
//! Supports migrating optimizer state between Adam, AdamW, SGD, Lion, and RMSProp,
//! transferring momentum buffers and step counts as appropriate.

use std::collections::HashMap;

// ─────────────────────────────────────────── ParamStateSnapshot ──────────────

/// A snapshot of optimizer state for one parameter.
#[derive(Debug, Clone)]
pub struct ParamStateSnapshot {
    pub param_name: String,
    pub step: u64,
    /// First moment (m) — Adam exponential moving average of gradient.
    pub first_moment: Option<Vec<f64>>,
    /// Second moment (v) — Adam exponential moving average of squared gradient.
    pub second_moment: Option<Vec<f64>>,
    /// SGD momentum buffer.
    pub momentum: Option<Vec<f64>>,
    /// EMA weights for Lion optimizer.
    pub ema: Option<Vec<f64>>,
    /// Last observed gradient norm.
    pub grad_norm: Option<f64>,
}

impl ParamStateSnapshot {
    /// Create a new empty snapshot for the named parameter.
    pub fn new(param_name: impl Into<String>) -> Self {
        Self {
            param_name: param_name.into(),
            step: 0,
            first_moment: None,
            second_moment: None,
            momentum: None,
            ema: None,
            grad_norm: None,
        }
    }
}

// ─────────────────────────────────────────── OptimizerKind ───────────────────

/// Optimizer type identifier used when specifying migration source/target.
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizerKind {
    Adam,
    AdamW,
    SGD,
    Lion,
    RMSProp,
}

impl std::fmt::Display for OptimizerKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Adam => write!(f, "Adam"),
            Self::AdamW => write!(f, "AdamW"),
            Self::SGD => write!(f, "SGD"),
            Self::Lion => write!(f, "Lion"),
            Self::RMSProp => write!(f, "RMSProp"),
        }
    }
}

// ─────────────────────────────────────────── SurgeryConfig ───────────────────

/// Configuration controlling how optimizer state is migrated.
#[derive(Debug, Clone)]
pub struct SurgeryConfig {
    pub from: OptimizerKind,
    pub to: OptimizerKind,
    /// When converting Adam→SGD, use first_moment as momentum buffer (true) or reset (false).
    pub transfer_momentum: bool,
    /// Reset step counter on switch.
    pub reset_step: bool,
    /// Scale transferred momentum by this factor (e.g. 0.9 to dampen).
    pub momentum_scale: f64,
}

impl Default for SurgeryConfig {
    fn default() -> Self {
        Self {
            from: OptimizerKind::Adam,
            to: OptimizerKind::SGD,
            transfer_momentum: true,
            reset_step: false,
            momentum_scale: 1.0,
        }
    }
}

// ─────────────────────────────────────────── MigrationReport ─────────────────

/// Summary report of an optimizer migration operation.
pub struct MigrationReport {
    pub params_migrated: usize,
    pub params_with_transferred_momentum: usize,
    pub params_with_reset_momentum: usize,
    pub steps_preserved: usize,
    pub steps_reset: usize,
    pub from: OptimizerKind,
    pub to: OptimizerKind,
}

impl std::fmt::Display for MigrationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MigrationReport {{ from: {}, to: {}, params_migrated: {}, \
             momentum_transferred: {}, momentum_reset: {}, \
             steps_preserved: {}, steps_reset: {} }}",
            self.from,
            self.to,
            self.params_migrated,
            self.params_with_transferred_momentum,
            self.params_with_reset_momentum,
            self.steps_preserved,
            self.steps_reset,
        )
    }
}

// ─────────────────────────────────────────── SurgeryError ────────────────────

/// Errors that can arise during optimizer surgery.
#[derive(Debug, thiserror::Error)]
pub enum SurgeryError {
    #[error("Invalid migration: {0}")]
    InvalidMigration(String),
    #[error("State incompatible: {0}")]
    IncompatibleState(String),
    #[error("Empty states")]
    EmptyStates,
}

// ─────────────────────────────────────────── OptimizerSurgeon ────────────────

/// Performs optimizer surgery: migrates state from one optimizer type to another.
pub struct OptimizerSurgeon {
    config: SurgeryConfig,
}

impl OptimizerSurgeon {
    /// Create a new surgeon with the given migration configuration.
    pub fn new(config: SurgeryConfig) -> Self {
        Self { config }
    }

    /// Validate that source states are compatible with this migration config.
    pub fn validate(
        &self,
        states: &HashMap<String, ParamStateSnapshot>,
    ) -> Result<(), SurgeryError> {
        if states.is_empty() {
            return Err(SurgeryError::EmptyStates);
        }

        // Check that unsupported migration pairs are rejected.
        match (&self.config.from, &self.config.to) {
            // Explicitly supported migrations.
            (OptimizerKind::Adam, OptimizerKind::SGD)
            | (OptimizerKind::Adam, OptimizerKind::Lion)
            | (OptimizerKind::Adam, OptimizerKind::Adam)
            | (OptimizerKind::Adam, OptimizerKind::AdamW)
            | (OptimizerKind::Adam, OptimizerKind::RMSProp)
            | (OptimizerKind::AdamW, OptimizerKind::Adam)
            | (OptimizerKind::AdamW, OptimizerKind::AdamW)
            | (OptimizerKind::AdamW, OptimizerKind::SGD)
            | (OptimizerKind::AdamW, OptimizerKind::Lion)
            | (OptimizerKind::AdamW, OptimizerKind::RMSProp)
            | (OptimizerKind::SGD, OptimizerKind::Adam)
            | (OptimizerKind::SGD, OptimizerKind::AdamW)
            | (OptimizerKind::SGD, OptimizerKind::SGD)
            | (OptimizerKind::SGD, OptimizerKind::Lion)
            | (OptimizerKind::SGD, OptimizerKind::RMSProp)
            | (OptimizerKind::Lion, OptimizerKind::Adam)
            | (OptimizerKind::Lion, OptimizerKind::AdamW)
            | (OptimizerKind::Lion, OptimizerKind::SGD)
            | (OptimizerKind::Lion, OptimizerKind::Lion)
            | (OptimizerKind::Lion, OptimizerKind::RMSProp)
            | (OptimizerKind::RMSProp, OptimizerKind::Adam)
            | (OptimizerKind::RMSProp, OptimizerKind::AdamW)
            | (OptimizerKind::RMSProp, OptimizerKind::SGD)
            | (OptimizerKind::RMSProp, OptimizerKind::Lion)
            | (OptimizerKind::RMSProp, OptimizerKind::RMSProp) => {},
        }

        // Validate source state compatibility.
        for (name, state) in states.iter() {
            match &self.config.from {
                OptimizerKind::Adam | OptimizerKind::AdamW => {
                    // Adam/AdamW should have first_moment if transfer_momentum is requested.
                    if self.config.transfer_momentum && state.first_moment.is_none() {
                        return Err(SurgeryError::IncompatibleState(format!(
                            "param '{}': transfer_momentum=true but first_moment is None for {} source",
                            name, self.config.from,
                        )));
                    }
                },
                OptimizerKind::SGD => {
                    // SGD should have momentum buffer if transferring.
                    if self.config.transfer_momentum && state.momentum.is_none() {
                        return Err(SurgeryError::IncompatibleState(format!(
                            "param '{}': transfer_momentum=true but momentum is None for SGD source",
                            name,
                        )));
                    }
                },
                OptimizerKind::Lion => {
                    // Lion should have ema if transferring.
                    if self.config.transfer_momentum && state.ema.is_none() {
                        return Err(SurgeryError::IncompatibleState(format!(
                            "param '{}': transfer_momentum=true but ema is None for Lion source",
                            name,
                        )));
                    }
                },
                OptimizerKind::RMSProp => {
                    // RMSProp uses second_moment (v) as its running variance.
                },
            }
        }

        Ok(())
    }

    /// Migrate state from source optimizer type to target optimizer type.
    ///
    /// Returns new state snapshots for use with the target optimizer.
    pub fn migrate(
        &self,
        states: &HashMap<String, ParamStateSnapshot>,
    ) -> Result<HashMap<String, ParamStateSnapshot>, SurgeryError> {
        if states.is_empty() {
            return Err(SurgeryError::EmptyStates);
        }

        let mut result = HashMap::with_capacity(states.len());

        for (name, src) in states.iter() {
            let new_state = self.migrate_single(src)?;
            result.insert(name.clone(), new_state);
        }

        Ok(result)
    }

    /// Migrate a single parameter's state.
    fn migrate_single(&self, src: &ParamStateSnapshot) -> Result<ParamStateSnapshot, SurgeryError> {
        let mut dst = ParamStateSnapshot::new(src.param_name.clone());

        // Step handling.
        dst.step = if self.config.reset_step { 0 } else { src.step };

        // Carry over grad_norm always (informational).
        dst.grad_norm = src.grad_norm;

        match (&self.config.from, &self.config.to) {
            // ── Adam / AdamW → SGD ────────────────────────────────────────────
            (OptimizerKind::Adam | OptimizerKind::AdamW, OptimizerKind::SGD) => {
                if self.config.transfer_momentum {
                    dst.momentum = src
                        .first_moment
                        .as_ref()
                        .map(|m| m.iter().map(|v| v * self.config.momentum_scale).collect());
                }
                // second_moment not used by SGD — discard.
            },

            // ── Adam / AdamW → Lion ───────────────────────────────────────────
            (OptimizerKind::Adam | OptimizerKind::AdamW, OptimizerKind::Lion) => {
                if self.config.transfer_momentum {
                    dst.ema = src
                        .first_moment
                        .as_ref()
                        .map(|m| m.iter().map(|v| v * self.config.momentum_scale).collect());
                }
                // second_moment not used by Lion — discard.
            },

            // ── Adam / AdamW → Adam / AdamW (same family) ─────────────────────
            (
                OptimizerKind::Adam | OptimizerKind::AdamW,
                OptimizerKind::Adam | OptimizerKind::AdamW,
            ) => {
                // Preserve both moments (same structure).
                if self.config.transfer_momentum {
                    dst.first_moment = src
                        .first_moment
                        .as_ref()
                        .map(|m| m.iter().map(|v| v * self.config.momentum_scale).collect());
                }
                dst.second_moment = src.second_moment.clone();
            },

            // ── Adam / AdamW → RMSProp ────────────────────────────────────────
            (OptimizerKind::Adam | OptimizerKind::AdamW, OptimizerKind::RMSProp) => {
                // RMSProp uses second moment (running variance).
                dst.second_moment = src.second_moment.clone();
            },

            // ── SGD → Adam / AdamW ────────────────────────────────────────────
            (OptimizerKind::SGD, OptimizerKind::Adam | OptimizerKind::AdamW) => {
                if self.config.transfer_momentum {
                    dst.first_moment = src
                        .momentum
                        .as_ref()
                        .map(|m| m.iter().map(|v| v * self.config.momentum_scale).collect());
                }
                // second_moment starts from zero (not available in SGD).
                dst.second_moment = None;
            },

            // ── SGD → Lion ────────────────────────────────────────────────────
            (OptimizerKind::SGD, OptimizerKind::Lion) => {
                if self.config.transfer_momentum {
                    dst.ema = src
                        .momentum
                        .as_ref()
                        .map(|m| m.iter().map(|v| v * self.config.momentum_scale).collect());
                }
            },

            // ── SGD → SGD (pass-through) ──────────────────────────────────────
            (OptimizerKind::SGD, OptimizerKind::SGD) => {
                if self.config.transfer_momentum {
                    dst.momentum = src
                        .momentum
                        .as_ref()
                        .map(|m| m.iter().map(|v| v * self.config.momentum_scale).collect());
                }
            },

            // ── SGD → RMSProp ─────────────────────────────────────────────────
            (OptimizerKind::SGD, OptimizerKind::RMSProp) => {
                // No useful state to transfer; second_moment is None.
            },

            // ── Lion → Adam / AdamW ───────────────────────────────────────────
            (OptimizerKind::Lion, OptimizerKind::Adam | OptimizerKind::AdamW) => {
                if self.config.transfer_momentum {
                    dst.first_moment = src
                        .ema
                        .as_ref()
                        .map(|e| e.iter().map(|v| v * self.config.momentum_scale).collect());
                }
                dst.second_moment = None;
            },

            // ── Lion → SGD ────────────────────────────────────────────────────
            (OptimizerKind::Lion, OptimizerKind::SGD) => {
                if self.config.transfer_momentum {
                    dst.momentum = src
                        .ema
                        .as_ref()
                        .map(|e| e.iter().map(|v| v * self.config.momentum_scale).collect());
                }
            },

            // ── Lion → Lion (pass-through) ────────────────────────────────────
            (OptimizerKind::Lion, OptimizerKind::Lion) => {
                if self.config.transfer_momentum {
                    dst.ema = src
                        .ema
                        .as_ref()
                        .map(|e| e.iter().map(|v| v * self.config.momentum_scale).collect());
                }
            },

            // ── Lion → RMSProp ────────────────────────────────────────────────
            (OptimizerKind::Lion, OptimizerKind::RMSProp) => {
                // No useful state to transfer.
            },

            // ── RMSProp → Adam / AdamW ────────────────────────────────────────
            (OptimizerKind::RMSProp, OptimizerKind::Adam | OptimizerKind::AdamW) => {
                // RMSProp's running variance maps to second_moment.
                dst.second_moment = src.second_moment.clone();
            },

            // ── RMSProp → SGD ─────────────────────────────────────────────────
            (OptimizerKind::RMSProp, OptimizerKind::SGD) => {
                // No useful state to transfer.
            },

            // ── RMSProp → Lion ────────────────────────────────────────────────
            (OptimizerKind::RMSProp, OptimizerKind::Lion) => {
                // No useful state to transfer.
            },

            // ── RMSProp → RMSProp (pass-through) ─────────────────────────────
            (OptimizerKind::RMSProp, OptimizerKind::RMSProp) => {
                dst.second_moment = src.second_moment.clone();
            },
        }

        Ok(dst)
    }

    /// Generate a migration summary report comparing before and after snapshots.
    pub fn migration_report(
        &self,
        before: &HashMap<String, ParamStateSnapshot>,
        after: &HashMap<String, ParamStateSnapshot>,
    ) -> MigrationReport {
        let params_migrated = after.len();
        let mut params_with_transferred_momentum = 0usize;
        let mut params_with_reset_momentum = 0usize;
        let mut steps_preserved = 0usize;
        let mut steps_reset = 0usize;

        for (name, dst) in after.iter() {
            let src_step = before.get(name).map(|s| s.step).unwrap_or(0);

            // Count step handling.
            if dst.step == 0 && src_step > 0 {
                steps_reset += 1;
            } else if dst.step == src_step {
                steps_preserved += 1;
            }

            // Count momentum transfer outcome.
            let has_transferred =
                dst.momentum.is_some() || dst.first_moment.is_some() || dst.ema.is_some();

            let src_had_transferable = before
                .get(name)
                .map(|s| s.first_moment.is_some() || s.momentum.is_some() || s.ema.is_some())
                .unwrap_or(false);

            if has_transferred && src_had_transferable {
                params_with_transferred_momentum += 1;
            } else if src_had_transferable && !has_transferred {
                params_with_reset_momentum += 1;
            }
        }

        MigrationReport {
            params_migrated,
            params_with_transferred_momentum,
            params_with_reset_momentum,
            steps_preserved,
            steps_reset,
            from: self.config.from.clone(),
            to: self.config.to.clone(),
        }
    }
}

// ─────────────────────────────────────────── tests ───────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_adam_state(name: &str, step: u64, size: usize) -> ParamStateSnapshot {
        let mut s = ParamStateSnapshot::new(name);
        s.step = step;
        s.first_moment = Some(vec![0.1; size]);
        s.second_moment = Some(vec![0.01; size]);
        s.grad_norm = Some(1.0);
        s
    }

    fn make_sgd_state(name: &str, step: u64, size: usize) -> ParamStateSnapshot {
        let mut s = ParamStateSnapshot::new(name);
        s.step = step;
        s.momentum = Some(vec![0.2; size]);
        s
    }

    fn make_lion_state(name: &str, step: u64, size: usize) -> ParamStateSnapshot {
        let mut s = ParamStateSnapshot::new(name);
        s.step = step;
        s.ema = Some(vec![0.05; size]);
        s
    }

    fn single_map(state: ParamStateSnapshot) -> HashMap<String, ParamStateSnapshot> {
        let mut m = HashMap::new();
        m.insert(state.param_name.clone(), state);
        m
    }

    // ── 1. Adam → SGD with momentum transfer ─────────────────────────────────

    #[test]
    fn test_adam_to_sgd_transfer_momentum() {
        let config = SurgeryConfig {
            from: OptimizerKind::Adam,
            to: OptimizerKind::SGD,
            transfer_momentum: true,
            reset_step: false,
            momentum_scale: 1.0,
        };
        let surgeon = OptimizerSurgeon::new(config);
        let states = single_map(make_adam_state("w", 10, 4));
        let result = surgeon.migrate(&states).expect("migrate");
        let dst = result.get("w").expect("w");
        assert!(dst.momentum.is_some());
        let mom = dst.momentum.as_ref().expect("momentum");
        assert_eq!(mom.len(), 4);
        assert!((mom[0] - 0.1).abs() < 1e-10);
        // second_moment must be cleared
        assert!(dst.second_moment.is_none());
        // first_moment must be cleared
        assert!(dst.first_moment.is_none());
    }

    // ── 2. Adam → SGD without momentum transfer ───────────────────────────────

    #[test]
    fn test_adam_to_sgd_no_transfer() {
        let config = SurgeryConfig {
            from: OptimizerKind::Adam,
            to: OptimizerKind::SGD,
            transfer_momentum: false,
            reset_step: false,
            momentum_scale: 1.0,
        };
        let surgeon = OptimizerSurgeon::new(config);
        let states = single_map(make_adam_state("w", 5, 3));
        let result = surgeon.migrate(&states).expect("migrate");
        let dst = result.get("w").expect("w");
        assert!(dst.momentum.is_none());
        assert!(dst.first_moment.is_none());
        assert!(dst.second_moment.is_none());
    }

    // ── 3. Adam → SGD with momentum_scale ────────────────────────────────────

    #[test]
    fn test_adam_to_sgd_momentum_scale() {
        let config = SurgeryConfig {
            from: OptimizerKind::Adam,
            to: OptimizerKind::SGD,
            transfer_momentum: true,
            reset_step: false,
            momentum_scale: 0.5,
        };
        let surgeon = OptimizerSurgeon::new(config);
        let states = single_map(make_adam_state("w", 10, 2));
        let result = surgeon.migrate(&states).expect("migrate");
        let dst = result.get("w").expect("w");
        let mom = dst.momentum.as_ref().expect("momentum");
        // 0.1 * 0.5 = 0.05
        assert!((mom[0] - 0.05).abs() < 1e-10);
    }

    // ── 4. Adam → Lion ────────────────────────────────────────────────────────

    #[test]
    fn test_adam_to_lion_transfer() {
        let config = SurgeryConfig {
            from: OptimizerKind::Adam,
            to: OptimizerKind::Lion,
            transfer_momentum: true,
            reset_step: false,
            momentum_scale: 1.0,
        };
        let surgeon = OptimizerSurgeon::new(config);
        let states = single_map(make_adam_state("w", 8, 3));
        let result = surgeon.migrate(&states).expect("migrate");
        let dst = result.get("w").expect("w");
        assert!(dst.ema.is_some());
        let ema = dst.ema.as_ref().expect("ema");
        assert!((ema[0] - 0.1).abs() < 1e-10);
        assert!(dst.second_moment.is_none());
        assert!(dst.first_moment.is_none());
    }

    // ── 5. Adam → Lion with momentum_scale ───────────────────────────────────

    #[test]
    fn test_adam_to_lion_momentum_scale() {
        let config = SurgeryConfig {
            from: OptimizerKind::Adam,
            to: OptimizerKind::Lion,
            transfer_momentum: true,
            reset_step: false,
            momentum_scale: 0.9,
        };
        let surgeon = OptimizerSurgeon::new(config);
        let states = single_map(make_adam_state("w", 3, 1));
        let result = surgeon.migrate(&states).expect("migrate");
        let dst = result.get("w").expect("w");
        let ema = dst.ema.as_ref().expect("ema");
        assert!((ema[0] - 0.09).abs() < 1e-10);
    }

    // ── 6. AdamW → Adam ───────────────────────────────────────────────────────

    #[test]
    fn test_adamw_to_adam_preserve_moments() {
        let config = SurgeryConfig {
            from: OptimizerKind::AdamW,
            to: OptimizerKind::Adam,
            transfer_momentum: true,
            reset_step: false,
            momentum_scale: 1.0,
        };
        let surgeon = OptimizerSurgeon::new(config);
        let mut src = make_adam_state("w", 20, 4);
        src.second_moment = Some(vec![0.02; 4]);
        let states = single_map(src);
        let result = surgeon.migrate(&states).expect("migrate");
        let dst = result.get("w").expect("w");
        assert!(dst.first_moment.is_some());
        let fm = dst.first_moment.as_ref().expect("first_moment");
        assert!((fm[0] - 0.1).abs() < 1e-10);
        assert!(dst.second_moment.is_some());
        let sm = dst.second_moment.as_ref().expect("second_moment");
        assert!((sm[0] - 0.02).abs() < 1e-10);
    }

    // ── 7. SGD → Adam ─────────────────────────────────────────────────────────

    #[test]
    fn test_sgd_to_adam_transfer() {
        let config = SurgeryConfig {
            from: OptimizerKind::SGD,
            to: OptimizerKind::Adam,
            transfer_momentum: true,
            reset_step: false,
            momentum_scale: 1.0,
        };
        let surgeon = OptimizerSurgeon::new(config);
        let states = single_map(make_sgd_state("w", 15, 3));
        let result = surgeon.migrate(&states).expect("migrate");
        let dst = result.get("w").expect("w");
        assert!(dst.first_moment.is_some());
        let fm = dst.first_moment.as_ref().expect("first_moment");
        assert!((fm[0] - 0.2).abs() < 1e-10);
        // second_moment must be None (not available in SGD)
        assert!(dst.second_moment.is_none());
    }

    // ── 8. reset_step flag ────────────────────────────────────────────────────

    #[test]
    fn test_reset_step_flag() {
        let config = SurgeryConfig {
            from: OptimizerKind::Adam,
            to: OptimizerKind::SGD,
            transfer_momentum: true,
            reset_step: true,
            momentum_scale: 1.0,
        };
        let surgeon = OptimizerSurgeon::new(config);
        let states = single_map(make_adam_state("w", 100, 2));
        let result = surgeon.migrate(&states).expect("migrate");
        let dst = result.get("w").expect("w");
        assert_eq!(dst.step, 0, "step should be reset to 0");
    }

    // ── 9. preserve step flag ─────────────────────────────────────────────────

    #[test]
    fn test_preserve_step_flag() {
        let config = SurgeryConfig {
            from: OptimizerKind::Adam,
            to: OptimizerKind::SGD,
            transfer_momentum: true,
            reset_step: false,
            momentum_scale: 1.0,
        };
        let surgeon = OptimizerSurgeon::new(config);
        let states = single_map(make_adam_state("w", 42, 2));
        let result = surgeon.migrate(&states).expect("migrate");
        let dst = result.get("w").expect("w");
        assert_eq!(dst.step, 42, "step should be preserved");
    }

    // ── 10. validate: empty states error ──────────────────────────────────────

    #[test]
    fn test_validate_empty_states_error() {
        let config = SurgeryConfig::default();
        let surgeon = OptimizerSurgeon::new(config);
        let empty: HashMap<String, ParamStateSnapshot> = HashMap::new();
        let err = surgeon.validate(&empty).expect_err("should error on empty");
        assert!(matches!(err, SurgeryError::EmptyStates));
    }

    // ── 11. migrate: empty states error ───────────────────────────────────────

    #[test]
    fn test_migrate_empty_states_error() {
        let config = SurgeryConfig::default();
        let surgeon = OptimizerSurgeon::new(config);
        let empty: HashMap<String, ParamStateSnapshot> = HashMap::new();
        let err = surgeon.migrate(&empty).expect_err("should error");
        assert!(matches!(err, SurgeryError::EmptyStates));
    }

    // ── 12. validate: incompatible state (missing first_moment for Adam→SGD) ──

    #[test]
    fn test_validate_incompatible_missing_first_moment() {
        let config = SurgeryConfig {
            from: OptimizerKind::Adam,
            to: OptimizerKind::SGD,
            transfer_momentum: true,
            reset_step: false,
            momentum_scale: 1.0,
        };
        let surgeon = OptimizerSurgeon::new(config);
        // State missing first_moment
        let state = ParamStateSnapshot::new("w"); // no first_moment
        let states = single_map(state);
        let err = surgeon.validate(&states).expect_err("should error");
        assert!(matches!(err, SurgeryError::IncompatibleState(_)));
    }

    // ── 13. validate: ok when transfer_momentum=false even without first_moment

    #[test]
    fn test_validate_ok_no_transfer_no_first_moment() {
        let config = SurgeryConfig {
            from: OptimizerKind::Adam,
            to: OptimizerKind::SGD,
            transfer_momentum: false,
            reset_step: false,
            momentum_scale: 1.0,
        };
        let surgeon = OptimizerSurgeon::new(config);
        let state = ParamStateSnapshot::new("w"); // no first_moment
        let states = single_map(state);
        assert!(surgeon.validate(&states).is_ok());
    }

    // ── 14. migration_report display ──────────────────────────────────────────

    #[test]
    fn test_migration_report_display() {
        let config = SurgeryConfig {
            from: OptimizerKind::Adam,
            to: OptimizerKind::SGD,
            transfer_momentum: true,
            reset_step: false,
            momentum_scale: 1.0,
        };
        let surgeon = OptimizerSurgeon::new(config);
        let before = single_map(make_adam_state("w", 10, 3));
        let after = surgeon.migrate(&before).expect("migrate");
        let report = surgeon.migration_report(&before, &after);
        let s = format!("{}", report);
        assert!(s.contains("Adam"));
        assert!(s.contains("SGD"));
        assert!(s.contains("params_migrated: 1"));
    }

    // ── 15. migration_report stats: steps reset ───────────────────────────────

    #[test]
    fn test_migration_report_steps_reset() {
        let config = SurgeryConfig {
            from: OptimizerKind::Adam,
            to: OptimizerKind::SGD,
            transfer_momentum: true,
            reset_step: true,
            momentum_scale: 1.0,
        };
        let surgeon = OptimizerSurgeon::new(config);
        let before = single_map(make_adam_state("w", 50, 2));
        let after = surgeon.migrate(&before).expect("migrate");
        let report = surgeon.migration_report(&before, &after);
        assert_eq!(report.steps_reset, 1);
        assert_eq!(report.steps_preserved, 0);
    }

    // ── 16. multiple params migrated ─────────────────────────────────────────

    #[test]
    fn test_multiple_params_migrated() {
        let config = SurgeryConfig {
            from: OptimizerKind::Adam,
            to: OptimizerKind::SGD,
            transfer_momentum: true,
            reset_step: false,
            momentum_scale: 1.0,
        };
        let surgeon = OptimizerSurgeon::new(config);
        let mut states = HashMap::new();
        for i in 0..5 {
            let s = make_adam_state(&format!("layer.{}", i), (i as u64) + 1, 4);
            states.insert(s.param_name.clone(), s);
        }
        let result = surgeon.migrate(&states).expect("migrate");
        assert_eq!(result.len(), 5);
        for (_, dst) in result.iter() {
            assert!(dst.momentum.is_some());
        }
    }

    // ── 17. Lion → SGD transfer ───────────────────────────────────────────────

    #[test]
    fn test_lion_to_sgd_transfer() {
        let config = SurgeryConfig {
            from: OptimizerKind::Lion,
            to: OptimizerKind::SGD,
            transfer_momentum: true,
            reset_step: false,
            momentum_scale: 1.0,
        };
        let surgeon = OptimizerSurgeon::new(config);
        let states = single_map(make_lion_state("w", 7, 3));
        let result = surgeon.migrate(&states).expect("migrate");
        let dst = result.get("w").expect("w");
        assert!(dst.momentum.is_some());
        let mom = dst.momentum.as_ref().expect("momentum");
        assert!((mom[0] - 0.05).abs() < 1e-10);
        assert!(dst.ema.is_none());
    }

    // ── 18. AdamW → Adam → AdamW round-trip preserves moments ────────────────

    #[test]
    fn test_adamw_to_adam_round_trip() {
        let config_fwd = SurgeryConfig {
            from: OptimizerKind::AdamW,
            to: OptimizerKind::Adam,
            transfer_momentum: true,
            reset_step: false,
            momentum_scale: 1.0,
        };
        let config_bwd = SurgeryConfig {
            from: OptimizerKind::Adam,
            to: OptimizerKind::AdamW,
            transfer_momentum: true,
            reset_step: false,
            momentum_scale: 1.0,
        };
        let mut src = make_adam_state("w", 20, 3);
        src.second_moment = Some(vec![0.02; 3]);
        let original_fm = src.first_moment.clone();
        let original_sm = src.second_moment.clone();

        let states = single_map(src);
        let mid = OptimizerSurgeon::new(config_fwd).migrate(&states).expect("fwd");
        let end = OptimizerSurgeon::new(config_bwd).migrate(&mid).expect("bwd");

        let dst = end.get("w").expect("w");
        assert_eq!(dst.first_moment, original_fm);
        assert_eq!(dst.second_moment, original_sm);
    }

    // ── 19. SurgeryError display ──────────────────────────────────────────────

    #[test]
    fn test_surgery_error_display() {
        let e1 = SurgeryError::InvalidMigration("reason".to_string());
        assert!(format!("{}", e1).contains("reason"));
        let e2 = SurgeryError::IncompatibleState("msg".to_string());
        assert!(format!("{}", e2).contains("msg"));
        let e3 = SurgeryError::EmptyStates;
        assert!(format!("{}", e3).contains("Empty"));
    }

    // ── 20. grad_norm preserved through migration ─────────────────────────────

    #[test]
    fn test_grad_norm_preserved() {
        let config = SurgeryConfig::default();
        let surgeon = OptimizerSurgeon::new(config);
        let mut src = make_adam_state("w", 5, 2);
        src.grad_norm = Some(std::f64::consts::PI);
        let states = single_map(src);
        let result = surgeon.migrate(&states).expect("migrate");
        let dst = result.get("w").expect("w");
        assert!((dst.grad_norm.expect("grad_norm") - std::f64::consts::PI).abs() < 1e-10);
    }
}
