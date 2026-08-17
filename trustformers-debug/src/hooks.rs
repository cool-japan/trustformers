//! Debugging hooks for automatic tensor and gradient tracking

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Hook trigger conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HookTrigger {
    /// Trigger on every forward pass
    EveryForward,
    /// Trigger on every backward pass
    EveryBackward,
    /// Trigger every N steps
    EveryNSteps(usize),
    /// Trigger when specific conditions are met
    Conditional(HookCondition),
    /// Trigger once and then remove
    Once,
    /// Trigger on specific layers only
    LayerSpecific(Vec<String>),
}

/// Conditions for conditional hooks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HookCondition {
    /// Trigger when loss exceeds threshold
    LossThreshold {
        threshold: f64,
        comparison: Comparison,
    },
    /// Trigger when gradient norm exceeds threshold
    GradientNormThreshold {
        threshold: f64,
        comparison: Comparison,
    },
    /// Trigger when memory usage exceeds threshold
    MemoryThreshold { threshold_mb: f64 },
    /// Trigger on specific training steps
    StepRange { start: usize, end: usize },
    /// Custom condition (placeholder for extensibility)
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Comparison {
    Greater,
    Less,
    Equal,
    GreaterEqual,
    LessEqual,
}

impl Comparison {
    /// Evaluate `value <comparison> threshold`. `Equal` uses a small
    /// relative epsilon rather than exact `==`, since the values being
    /// compared (loss, gradient norm, memory usage) are computed floats
    /// that are never expected to match a configured threshold bit-for-bit.
    fn apply(&self, value: f64, threshold: f64) -> bool {
        match self {
            Comparison::Greater => value > threshold,
            Comparison::Less => value < threshold,
            Comparison::GreaterEqual => value >= threshold,
            Comparison::LessEqual => value <= threshold,
            Comparison::Equal => (value - threshold).abs() <= 1e-9_f64.max(threshold.abs() * 1e-9),
        }
    }
}

/// Hook action types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HookAction {
    /// Inspect tensor values
    InspectTensor,
    /// Track gradient flow
    TrackGradients,
    /// Record layer activations
    RecordActivations,
    /// Save tensor snapshot to file
    SaveSnapshot { path: String },
    /// Generate alert
    Alert {
        message: String,
        severity: AlertSeverity,
    },
    /// Execute custom callback
    CustomCallback { name: String },
    /// Pause training for manual inspection
    PauseTraining,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Hook configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookConfig {
    pub id: Uuid,
    pub name: String,
    pub trigger: HookTrigger,
    pub actions: Vec<HookAction>,
    pub enabled: bool,
    pub max_executions: Option<usize>,
    pub layer_patterns: Vec<String>, // Regex patterns for layer names
}

/// Hook execution context
#[derive(Debug)]
pub struct HookContext {
    pub step: usize,
    pub layer_name: String,
    pub tensor_shape: Vec<usize>,
    pub is_forward: bool,
    pub metadata: HashMap<String, String>,
    /// Current training loss, if the caller has reported one via
    /// [`HookManager::set_loss`]. `None` (rather than a fabricated 0.0 or a
    /// stale value) means no loss has been reported yet for this session --
    /// [`HookCondition::LossThreshold`] never fires on a hook it has no
    /// real data to evaluate.
    pub loss: Option<f64>,
    /// Current gradient norm, if reported via
    /// [`HookManager::set_gradient_norm`]. See `loss` for the `None`
    /// semantics.
    pub grad_norm: Option<f64>,
    /// Current memory usage in MB, if reported via
    /// [`HookManager::set_memory_mb`]. See `loss` for the `None` semantics.
    pub memory_mb: Option<f64>,
}

/// Hook execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookStats {
    pub hook_id: Uuid,
    pub hook_name: String,
    pub total_executions: usize,
    pub last_execution_step: Option<usize>,
    pub total_execution_time_ms: f64,
    pub avg_execution_time_ms: f64,
    pub errors: usize,
}

/// Hook execution result
#[derive(Debug)]
pub enum HookResult {
    Success,
    Error(String),
    Skipped(String),
}

/// Callback function type for custom hooks
pub type HookCallback = Box<dyn Fn(&HookContext, &[u8]) -> Result<()> + Send + Sync>;

/// Hook manager for coordinating debugging hooks
pub struct HookManager {
    hooks: HashMap<Uuid, HookConfig>,
    hook_stats: HashMap<Uuid, HookStats>,
    callbacks: HashMap<String, HookCallback>,
    execution_count: HashMap<Uuid, usize>,
    global_step: usize,
    enabled: bool,
    /// Latest reported loss, memory (MB) and gradient norm -- fed into every
    /// [`HookContext`] built by [`HookManager::execute_hooks`], so
    /// [`HookCondition::LossThreshold`], [`HookCondition::GradientNormThreshold`]
    /// and [`HookCondition::MemoryThreshold`] have real values to compare
    /// against instead of firing unconditionally. See
    /// [`HookManager::set_loss`] / [`HookManager::set_gradient_norm`] /
    /// [`HookManager::set_memory_mb`].
    current_loss: Option<f64>,
    current_grad_norm: Option<f64>,
    current_memory_mb: Option<f64>,
}

impl std::fmt::Debug for HookManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HookManager")
            .field("hooks", &self.hooks)
            .field("hook_stats", &self.hook_stats)
            .field("execution_count", &self.execution_count)
            .field("global_step", &self.global_step)
            .field("enabled", &self.enabled)
            .field("callbacks", &format!("{} callbacks", self.callbacks.len()))
            .field("current_loss", &self.current_loss)
            .field("current_grad_norm", &self.current_grad_norm)
            .field("current_memory_mb", &self.current_memory_mb)
            .finish()
    }
}

impl HookManager {
    /// Create a new hook manager
    pub fn new() -> Self {
        Self {
            hooks: HashMap::new(),
            hook_stats: HashMap::new(),
            callbacks: HashMap::new(),
            execution_count: HashMap::new(),
            global_step: 0,
            enabled: true,
            current_loss: None,
            current_grad_norm: None,
            current_memory_mb: None,
        }
    }

    /// Register a new hook
    pub fn register_hook(&mut self, config: HookConfig) -> Result<Uuid> {
        let hook_id = config.id;

        // Initialize statistics
        self.hook_stats.insert(
            hook_id,
            HookStats {
                hook_id,
                hook_name: config.name.clone(),
                total_executions: 0,
                last_execution_step: None,
                total_execution_time_ms: 0.0,
                avg_execution_time_ms: 0.0,
                errors: 0,
            },
        );

        self.execution_count.insert(hook_id, 0);
        self.hooks.insert(hook_id, config);

        tracing::debug!("Registered hook {}", hook_id);
        Ok(hook_id)
    }

    /// Register a custom callback
    pub fn register_callback(&mut self, name: String, callback: HookCallback) {
        self.callbacks.insert(name, callback);
    }

    /// Remove a hook
    pub fn remove_hook(&mut self, hook_id: Uuid) -> Option<HookConfig> {
        self.hook_stats.remove(&hook_id);
        self.execution_count.remove(&hook_id);
        self.hooks.remove(&hook_id)
    }

    /// Enable/disable a specific hook
    pub fn set_hook_enabled(&mut self, hook_id: Uuid, enabled: bool) -> Result<()> {
        if let Some(hook) = self.hooks.get_mut(&hook_id) {
            hook.enabled = enabled;
            Ok(())
        } else {
            Err(anyhow::anyhow!("Hook {} not found", hook_id))
        }
    }

    /// Enable/disable all hooks
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Update global step counter
    pub fn set_step(&mut self, step: usize) {
        self.global_step = step;
    }

    /// Report the current training loss for [`HookCondition::LossThreshold`]
    /// evaluation. Call this once per step before [`Self::execute_hooks`];
    /// without it, `LossThreshold` conditions never fire (see
    /// [`Self::evaluate_condition`]).
    pub fn set_loss(&mut self, loss: f64) {
        self.current_loss = Some(loss);
    }

    /// Report the current gradient norm for
    /// [`HookCondition::GradientNormThreshold`] evaluation. See
    /// [`Self::set_loss`].
    pub fn set_gradient_norm(&mut self, grad_norm: f64) {
        self.current_grad_norm = Some(grad_norm);
    }

    /// Report current memory usage (in MB) for
    /// [`HookCondition::MemoryThreshold`] evaluation. See [`Self::set_loss`].
    pub fn set_memory_mb(&mut self, memory_mb: f64) {
        self.current_memory_mb = Some(memory_mb);
    }

    /// Execute hooks for a tensor operation
    pub fn execute_hooks<T>(
        &mut self,
        layer_name: &str,
        tensor_data: &[T],
        tensor_shape: &[usize],
        is_forward: bool,
        metadata: Option<HashMap<String, String>>,
    ) -> Vec<(Uuid, HookResult)>
    where
        T: Clone + 'static,
    {
        if !self.enabled {
            return Vec::new();
        }

        let context = HookContext {
            step: self.global_step,
            layer_name: layer_name.to_string(),
            tensor_shape: tensor_shape.to_vec(),
            is_forward,
            metadata: metadata.unwrap_or_default(),
            loss: self.current_loss,
            grad_norm: self.current_grad_norm,
            memory_mb: self.current_memory_mb,
        };

        let mut results = Vec::new();

        // Convert tensor data to bytes for callbacks
        let tensor_bytes = unsafe {
            std::slice::from_raw_parts(
                tensor_data.as_ptr() as *const u8,
                std::mem::size_of_val(tensor_data),
            )
        };

        // Collect hook IDs and configs to avoid borrowing conflicts
        let hooks_to_execute: Vec<(Uuid, HookConfig)> =
            self.hooks.iter().map(|(id, config)| (*id, config.clone())).collect();

        for (hook_id, hook_config) in hooks_to_execute {
            if !hook_config.enabled {
                continue;
            }

            // Check if we should execute this hook
            if let Some(should_execute) = self.should_execute_hook(&hook_config, &context) {
                if !should_execute {
                    results.push((
                        hook_id,
                        HookResult::Skipped("Condition not met".to_string()),
                    ));
                    continue;
                }
            }

            // Check execution count limits
            let current_count = self.execution_count.get(&hook_id).copied().unwrap_or(0);
            if let Some(max_executions) = hook_config.max_executions {
                if current_count >= max_executions {
                    results.push((
                        hook_id,
                        HookResult::Skipped("Max executions reached".to_string()),
                    ));
                    continue;
                }
            }

            // Execute hook
            let start_time = std::time::Instant::now();
            let result = self.execute_single_hook(&hook_config, &context, tensor_bytes);
            let execution_time = start_time.elapsed().as_millis() as f64;

            // Update statistics
            if let Some(stats) = self.hook_stats.get_mut(&hook_id) {
                stats.total_executions += 1;
                stats.last_execution_step = Some(self.global_step);
                stats.total_execution_time_ms += execution_time;
                stats.avg_execution_time_ms =
                    stats.total_execution_time_ms / stats.total_executions as f64;

                if matches!(result, HookResult::Error(_)) {
                    stats.errors += 1;
                }
            }

            // Update execution count
            if let Some(count) = self.execution_count.get_mut(&hook_id) {
                *count += 1;
            }

            results.push((hook_id, result));
        }

        results
    }

    /// Get hook configuration
    pub fn get_hook(&self, hook_id: Uuid) -> Option<&HookConfig> {
        self.hooks.get(&hook_id)
    }

    /// Get all hooks
    pub fn get_all_hooks(&self) -> Vec<&HookConfig> {
        self.hooks.values().collect()
    }

    /// Get hook statistics
    pub fn get_hook_stats(&self, hook_id: Uuid) -> Option<&HookStats> {
        self.hook_stats.get(&hook_id)
    }

    /// Get all hook statistics
    pub fn get_all_stats(&self) -> Vec<&HookStats> {
        self.hook_stats.values().collect()
    }

    /// Clear all hooks
    pub fn clear_hooks(&mut self) {
        self.hooks.clear();
        self.hook_stats.clear();
        self.execution_count.clear();
        self.callbacks.clear();
    }

    /// Create a convenient tensor inspection hook
    pub fn create_tensor_inspection_hook(&mut self, layer_patterns: Vec<String>) -> Result<Uuid> {
        let config = HookConfig {
            id: Uuid::new_v4(),
            name: "Tensor Inspector".to_string(),
            trigger: HookTrigger::EveryForward,
            actions: vec![HookAction::InspectTensor],
            enabled: true,
            max_executions: None,
            layer_patterns,
        };

        self.register_hook(config)
    }

    /// Create a gradient tracking hook
    pub fn create_gradient_tracking_hook(&mut self, layer_patterns: Vec<String>) -> Result<Uuid> {
        let config = HookConfig {
            id: Uuid::new_v4(),
            name: "Gradient Tracker".to_string(),
            trigger: HookTrigger::EveryBackward,
            actions: vec![HookAction::TrackGradients],
            enabled: true,
            max_executions: None,
            layer_patterns,
        };

        self.register_hook(config)
    }

    /// Create a conditional alert hook
    pub fn create_alert_hook(
        &mut self,
        condition: HookCondition,
        message: String,
        severity: AlertSeverity,
    ) -> Result<Uuid> {
        let config = HookConfig {
            id: Uuid::new_v4(),
            name: "Alert Hook".to_string(),
            trigger: HookTrigger::Conditional(condition),
            actions: vec![HookAction::Alert { message, severity }],
            enabled: true,
            max_executions: None,
            layer_patterns: vec![".*".to_string()], // Match all layers
        };

        self.register_hook(config)
    }

    // Private helper methods

    fn should_execute_hook(&self, hook: &HookConfig, context: &HookContext) -> Option<bool> {
        // Check layer pattern matching
        if !hook.layer_patterns.is_empty() {
            let matches_pattern = hook.layer_patterns.iter().any(|pattern| {
                regex::Regex::new(pattern)
                    .map(|re| re.is_match(&context.layer_name))
                    .unwrap_or(false)
            });

            if !matches_pattern {
                return Some(false);
            }
        }

        match &hook.trigger {
            HookTrigger::EveryForward => Some(context.is_forward),
            HookTrigger::EveryBackward => Some(!context.is_forward),
            HookTrigger::EveryNSteps(n) => Some(context.step.is_multiple_of(*n)),
            HookTrigger::Conditional(condition) => {
                Some(self.evaluate_condition(condition, context))
            },
            HookTrigger::Once => {
                let count = self.execution_count.get(&hook.id).copied().unwrap_or(0);
                Some(count == 0)
            },
            HookTrigger::LayerSpecific(layers) => Some(layers.contains(&context.layer_name)),
        }
    }

    /// Evaluate a single [`HookCondition`] against the current context.
    ///
    /// `LossThreshold` / `GradientNormThreshold` / `MemoryThreshold` compare
    /// against real values reported via [`Self::set_loss`] /
    /// [`Self::set_gradient_norm`] / [`Self::set_memory_mb`]. If the caller
    /// never reported that metric for this session, the corresponding
    /// `context` field is `None` and the condition returns `false` -- never
    /// `true` -- since a threshold cannot honestly be judged "met" against
    /// data that was never provided. This intentionally differs from the
    /// old behavior, where every one of these three conditions fired
    /// unconditionally (`_ => true`) regardless of whether any relevant
    /// data existed.
    fn evaluate_condition(&self, condition: &HookCondition, context: &HookContext) -> bool {
        match condition {
            HookCondition::StepRange { start, end } => {
                context.step >= *start && context.step <= *end
            },
            HookCondition::Custom(name) => {
                // For custom conditions, we'd need additional context
                // This is a placeholder implementation
                context.metadata.contains_key(name)
            },
            HookCondition::LossThreshold {
                threshold,
                comparison,
            } => context.loss.map(|loss| comparison.apply(loss, *threshold)).unwrap_or(false),
            HookCondition::GradientNormThreshold {
                threshold,
                comparison,
            } => context
                .grad_norm
                .map(|grad_norm| comparison.apply(grad_norm, *threshold))
                .unwrap_or(false),
            HookCondition::MemoryThreshold { threshold_mb } => {
                context.memory_mb.map(|memory_mb| memory_mb > *threshold_mb).unwrap_or(false)
            },
        }
    }

    fn execute_single_hook(
        &mut self,
        hook: &HookConfig,
        context: &HookContext,
        tensor_data: &[u8],
    ) -> HookResult {
        for action in &hook.actions {
            match self.execute_action(action, context, tensor_data) {
                Ok(()) => continue,
                Err(e) => return HookResult::Error(e.to_string()),
            }
        }
        HookResult::Success
    }

    fn execute_action(
        &mut self,
        action: &HookAction,
        context: &HookContext,
        tensor_data: &[u8],
    ) -> Result<()> {
        match action {
            HookAction::InspectTensor => {
                tracing::debug!(
                    "Inspecting tensor in layer '{}' at step {}",
                    context.layer_name,
                    context.step
                );
                // In practice, this would call the tensor inspector
                Ok(())
            },
            HookAction::TrackGradients => {
                tracing::debug!(
                    "Tracking gradients in layer '{}' at step {}",
                    context.layer_name,
                    context.step
                );
                // In practice, this would call the gradient debugger
                Ok(())
            },
            HookAction::RecordActivations => {
                tracing::debug!(
                    "Recording activations in layer '{}' at step {}",
                    context.layer_name,
                    context.step
                );
                // In practice, this would record activation statistics
                Ok(())
            },
            HookAction::SaveSnapshot { path } => {
                let file_path =
                    format!("{}_{}_step_{}.bin", path, context.layer_name, context.step);
                std::fs::write(&file_path, tensor_data)?;
                tracing::info!("Saved tensor snapshot to {}", file_path);
                Ok(())
            },
            HookAction::Alert { message, severity } => {
                match severity {
                    AlertSeverity::Info => tracing::info!("Hook Alert: {}", message),
                    AlertSeverity::Warning => tracing::warn!("Hook Alert: {}", message),
                    AlertSeverity::Critical => tracing::error!("Hook Alert: {}", message),
                }
                Ok(())
            },
            HookAction::CustomCallback { name } => {
                if let Some(callback) = self.callbacks.get(name) {
                    callback(context, tensor_data)?;
                } else {
                    return Err(anyhow::anyhow!("Callback '{}' not found", name));
                }
                Ok(())
            },
            HookAction::PauseTraining => {
                tracing::warn!(
                    "Training paused by hook at step {} in layer '{}'",
                    context.step,
                    context.layer_name
                );
                // In practice, this would set a flag to pause training
                Ok(())
            },
        }
    }
}

impl Default for HookManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating hook configurations
pub struct HookBuilder {
    config: HookConfig,
}

impl HookBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            config: HookConfig {
                id: Uuid::new_v4(),
                name: name.to_string(),
                trigger: HookTrigger::EveryForward,
                actions: Vec::new(),
                enabled: true,
                max_executions: None,
                layer_patterns: Vec::new(),
            },
        }
    }

    pub fn trigger(mut self, trigger: HookTrigger) -> Self {
        self.config.trigger = trigger;
        self
    }

    pub fn action(mut self, action: HookAction) -> Self {
        self.config.actions.push(action);
        self
    }

    pub fn actions(mut self, actions: Vec<HookAction>) -> Self {
        self.config.actions = actions;
        self
    }

    pub fn max_executions(mut self, max: usize) -> Self {
        self.config.max_executions = Some(max);
        self
    }

    pub fn layer_patterns(mut self, patterns: Vec<String>) -> Self {
        self.config.layer_patterns = patterns;
        self
    }

    pub fn enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    pub fn build(self) -> HookConfig {
        self.config
    }
}

/// Convenience macros for creating hooks
#[macro_export]
macro_rules! tensor_hook {
    ($name:expr, $patterns:expr) => {
        HookBuilder::new($name)
            .trigger(HookTrigger::EveryForward)
            .action(HookAction::InspectTensor)
            .layer_patterns($patterns)
            .build()
    };
}

#[macro_export]
macro_rules! gradient_hook {
    ($name:expr, $patterns:expr) => {
        HookBuilder::new($name)
            .trigger(HookTrigger::EveryBackward)
            .action(HookAction::TrackGradients)
            .layer_patterns($patterns)
            .build()
    };
}

#[macro_export]
macro_rules! alert_hook {
    ($condition:expr, $message:expr, $severity:expr) => {
        HookBuilder::new("Alert Hook")
            .trigger(HookTrigger::Conditional($condition))
            .action(HookAction::Alert {
                message: $message.to_string(),
                severity: $severity,
            })
            .build()
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_hook_config(name: &str, trigger: HookTrigger) -> HookConfig {
        HookConfig {
            id: Uuid::new_v4(),
            name: name.to_string(),
            trigger,
            actions: vec![HookAction::InspectTensor],
            enabled: true,
            max_executions: None,
            layer_patterns: vec![],
        }
    }

    // ── HookManager construction ────────────────────────────────────────────

    #[test]
    fn test_hook_manager_new_defaults() {
        let mgr = HookManager::new();
        assert!(mgr.enabled);
        assert_eq!(mgr.global_step, 0);
        assert!(mgr.get_all_hooks().is_empty());
        assert!(mgr.get_all_stats().is_empty());
    }

    #[test]
    fn test_hook_manager_default_equals_new() {
        let mgr = HookManager::default();
        assert!(mgr.enabled);
    }

    // ── register_hook ──────────────────────────────────────────────────────

    #[test]
    fn test_register_hook_returns_uuid() {
        let mut mgr = HookManager::new();
        let config = make_hook_config("test", HookTrigger::EveryForward);
        let id = config.id;
        let returned = mgr.register_hook(config).expect("register should succeed");
        assert_eq!(returned, id);
    }

    #[test]
    fn test_register_multiple_hooks() {
        let mut mgr = HookManager::new();
        for i in 0..5 {
            let cfg = make_hook_config(&format!("h{}", i), HookTrigger::EveryForward);
            mgr.register_hook(cfg).expect("register should succeed");
        }
        assert_eq!(mgr.get_all_hooks().len(), 5);
    }

    #[test]
    fn test_hook_stats_initialized_on_register() {
        let mut mgr = HookManager::new();
        let cfg = make_hook_config("h0", HookTrigger::EveryForward);
        let id = mgr.register_hook(cfg).expect("register should succeed");
        let stats = mgr.get_hook_stats(id).expect("stats should exist");
        assert_eq!(stats.total_executions, 0);
        assert_eq!(stats.errors, 0);
    }

    // ── remove_hook ────────────────────────────────────────────────────────

    #[test]
    fn test_remove_hook_returns_config() {
        let mut mgr = HookManager::new();
        let cfg = make_hook_config("remove_me", HookTrigger::EveryBackward);
        let id = mgr.register_hook(cfg).expect("register");
        let removed = mgr.remove_hook(id);
        assert!(removed.is_some());
        assert_eq!(removed.expect("should be some").name, "remove_me");
    }

    #[test]
    fn test_remove_nonexistent_hook_returns_none() {
        let mut mgr = HookManager::new();
        let id = Uuid::new_v4();
        assert!(mgr.remove_hook(id).is_none());
    }

    // ── set_hook_enabled ───────────────────────────────────────────────────

    #[test]
    fn test_set_hook_enabled_ok() {
        let mut mgr = HookManager::new();
        let cfg = make_hook_config("h", HookTrigger::EveryForward);
        let id = mgr.register_hook(cfg).expect("register");
        mgr.set_hook_enabled(id, false).expect("should succeed");
        let hook = mgr.get_hook(id).expect("hook should exist");
        assert!(!hook.enabled);
        mgr.set_hook_enabled(id, true).expect("re-enable");
        let hook = mgr.get_hook(id).expect("hook should exist");
        assert!(hook.enabled);
    }

    #[test]
    fn test_set_hook_enabled_nonexistent_errors() {
        let mut mgr = HookManager::new();
        let result = mgr.set_hook_enabled(Uuid::new_v4(), true);
        assert!(result.is_err());
    }

    // ── set_enabled (global) ───────────────────────────────────────────────

    #[test]
    fn test_global_disable_stops_execution() {
        let mut mgr = HookManager::new();
        mgr.set_enabled(false);
        mgr.register_hook(make_hook_config("h", HookTrigger::EveryForward))
            .expect("register");
        let results = mgr.execute_hooks("layer", &[1u8, 2u8], &[2], true, None);
        assert!(
            results.is_empty(),
            "globally disabled manager should execute nothing"
        );
    }

    // ── set_step ───────────────────────────────────────────────────────────

    #[test]
    fn test_set_step_updates_counter() {
        let mut mgr = HookManager::new();
        mgr.set_step(42);
        assert_eq!(mgr.global_step, 42);
    }

    // ── execute_hooks ──────────────────────────────────────────────────────

    #[test]
    fn test_execute_hooks_disabled_hook_skipped() {
        let mut mgr = HookManager::new();
        let mut cfg = make_hook_config("h", HookTrigger::EveryForward);
        cfg.enabled = false;
        mgr.register_hook(cfg).expect("register");
        let results = mgr.execute_hooks("layer", &[0u8], &[1], true, None);
        // Disabled hook → no results (the impl skips it without adding an entry)
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_execute_hooks_every_forward_fires_on_forward() {
        let mut mgr = HookManager::new();
        mgr.register_hook(make_hook_config("h", HookTrigger::EveryForward))
            .expect("register");
        let results = mgr.execute_hooks("layer", &[1u8], &[1], true, None);
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_execute_hooks_every_forward_skipped_on_backward() {
        let mut mgr = HookManager::new();
        // No layer_patterns → pattern check skipped, trigger decides.
        let cfg = make_hook_config("h", HookTrigger::EveryForward);
        mgr.register_hook(cfg).expect("register");
        let results = mgr.execute_hooks("layer", &[1u8], &[1], false, None);
        // is_forward=false → the hook's should_execute returns Some(false) → Skipped
        assert_eq!(results.len(), 1);
        let (_, ref outcome) = results[0];
        assert!(matches!(outcome, HookResult::Skipped(_)));
    }

    #[test]
    fn test_execute_hooks_max_executions_respected() {
        let mut mgr = HookManager::new();
        let mut cfg = make_hook_config("once", HookTrigger::EveryForward);
        cfg.max_executions = Some(1);
        mgr.register_hook(cfg).expect("register");

        // First execution should succeed
        let r1 = mgr.execute_hooks("layer", &[1u8], &[1], true, None);
        assert_eq!(r1.len(), 1);
        assert!(matches!(r1[0].1, HookResult::Success));

        // Second execution should be Skipped
        let r2 = mgr.execute_hooks("layer", &[1u8], &[1], true, None);
        assert_eq!(r2.len(), 1);
        assert!(matches!(r2[0].1, HookResult::Skipped(_)));
    }

    // ── HookCondition metric thresholds ─────────────────────────────────────
    //
    // Regression tests for the `_ => true` bug: LossThreshold /
    // GradientNormThreshold / MemoryThreshold used to fire on every single
    // step regardless of the configured threshold. Each test below would
    // have failed against that old behavior (the "never met" and
    // "no data reported" cases would incorrectly have produced `Success`).

    fn make_conditional_hook_config(condition: HookCondition) -> HookConfig {
        HookConfig {
            id: Uuid::new_v4(),
            name: "conditional".to_string(),
            trigger: HookTrigger::Conditional(condition),
            actions: vec![HookAction::InspectTensor],
            enabled: true,
            max_executions: None,
            layer_patterns: vec![],
        }
    }

    #[test]
    fn test_loss_threshold_does_not_fire_without_reported_loss() {
        let mut mgr = HookManager::new();
        let cond = HookCondition::LossThreshold {
            threshold: 1.0,
            comparison: Comparison::Greater,
        };
        mgr.register_hook(make_conditional_hook_config(cond)).expect("register");

        // No `set_loss` call: the old `_ => true` fallback would fire this
        // unconditionally even though no loss was ever reported.
        let results = mgr.execute_hooks("layer", &[1u8], &[1], true, None);
        assert_eq!(results.len(), 1);
        assert!(
            matches!(results[0].1, HookResult::Skipped(_)),
            "must not fire when no loss has been reported, got {:?}",
            results[0].1
        );
    }

    #[test]
    fn test_loss_threshold_fires_when_exceeded() {
        let mut mgr = HookManager::new();
        let cond = HookCondition::LossThreshold {
            threshold: 1.0,
            comparison: Comparison::Greater,
        };
        mgr.register_hook(make_conditional_hook_config(cond)).expect("register");

        mgr.set_loss(5.0); // loss spiked above threshold
        let results = mgr.execute_hooks("layer", &[1u8], &[1], true, None);
        assert_eq!(results.len(), 1);
        assert!(matches!(results[0].1, HookResult::Success));
    }

    #[test]
    fn test_loss_threshold_does_not_fire_when_below_threshold() {
        let mut mgr = HookManager::new();
        let cond = HookCondition::LossThreshold {
            threshold: 1.0,
            comparison: Comparison::Greater,
        };
        mgr.register_hook(make_conditional_hook_config(cond)).expect("register");

        mgr.set_loss(0.1); // well below threshold
        let results = mgr.execute_hooks("layer", &[1u8], &[1], true, None);
        assert_eq!(results.len(), 1);
        assert!(
            matches!(results[0].1, HookResult::Skipped(_)),
            "must not fire when loss is below the threshold, got {:?}",
            results[0].1
        );
    }

    #[test]
    fn test_gradient_norm_threshold_does_not_fire_without_reported_norm() {
        let mut mgr = HookManager::new();
        let cond = HookCondition::GradientNormThreshold {
            threshold: 10.0,
            comparison: Comparison::Greater,
        };
        mgr.register_hook(make_conditional_hook_config(cond)).expect("register");

        let results = mgr.execute_hooks("layer", &[1u8], &[1], false, None);
        assert_eq!(results.len(), 1);
        assert!(matches!(results[0].1, HookResult::Skipped(_)));
    }

    #[test]
    fn test_gradient_norm_threshold_fires_on_explosion() {
        let mut mgr = HookManager::new();
        let cond = HookCondition::GradientNormThreshold {
            threshold: 10.0,
            comparison: Comparison::Greater,
        };
        mgr.register_hook(make_conditional_hook_config(cond)).expect("register");

        mgr.set_gradient_norm(1000.0); // exploding gradient
        let results = mgr.execute_hooks("layer", &[1u8], &[1], false, None);
        assert_eq!(results.len(), 1);
        assert!(matches!(results[0].1, HookResult::Success));
    }

    #[test]
    fn test_gradient_norm_threshold_respects_less_comparison() {
        let mut mgr = HookManager::new();
        // "fire when gradient vanishes below 1e-6"
        let cond = HookCondition::GradientNormThreshold {
            threshold: 1e-6,
            comparison: Comparison::Less,
        };
        mgr.register_hook(make_conditional_hook_config(cond)).expect("register");

        mgr.set_gradient_norm(0.5); // healthy gradient, should not fire
        let healthy = mgr.execute_hooks("layer", &[1u8], &[1], false, None);
        assert!(matches!(healthy[0].1, HookResult::Skipped(_)));

        mgr.set_gradient_norm(1e-9); // vanished, should fire
        let vanished = mgr.execute_hooks("layer", &[1u8], &[1], false, None);
        assert!(matches!(vanished[0].1, HookResult::Success));
    }

    #[test]
    fn test_memory_threshold_does_not_fire_without_reported_memory() {
        let mut mgr = HookManager::new();
        let cond = HookCondition::MemoryThreshold {
            threshold_mb: 1000.0,
        };
        mgr.register_hook(make_conditional_hook_config(cond)).expect("register");

        let results = mgr.execute_hooks("layer", &[1u8], &[1], true, None);
        assert_eq!(results.len(), 1);
        assert!(matches!(results[0].1, HookResult::Skipped(_)));
    }

    #[test]
    fn test_memory_threshold_fires_over_limit() {
        let mut mgr = HookManager::new();
        let cond = HookCondition::MemoryThreshold {
            threshold_mb: 1000.0,
        };
        mgr.register_hook(make_conditional_hook_config(cond)).expect("register");

        mgr.set_memory_mb(4096.0);
        let results = mgr.execute_hooks("layer", &[1u8], &[1], true, None);
        assert_eq!(results.len(), 1);
        assert!(matches!(results[0].1, HookResult::Success));
    }

    #[test]
    fn test_memory_threshold_does_not_fire_under_limit() {
        let mut mgr = HookManager::new();
        let cond = HookCondition::MemoryThreshold {
            threshold_mb: 1000.0,
        };
        mgr.register_hook(make_conditional_hook_config(cond)).expect("register");

        mgr.set_memory_mb(50.0);
        let results = mgr.execute_hooks("layer", &[1u8], &[1], true, None);
        assert_eq!(results.len(), 1);
        assert!(matches!(results[0].1, HookResult::Skipped(_)));
    }

    #[test]
    fn test_comparison_apply_all_variants() {
        assert!(Comparison::Greater.apply(2.0, 1.0));
        assert!(!Comparison::Greater.apply(1.0, 1.0));
        assert!(Comparison::Less.apply(0.5, 1.0));
        assert!(!Comparison::Less.apply(1.0, 1.0));
        assert!(Comparison::GreaterEqual.apply(1.0, 1.0));
        assert!(Comparison::LessEqual.apply(1.0, 1.0));
        assert!(Comparison::Equal.apply(1.0, 1.0));
        assert!(!Comparison::Equal.apply(1.5, 1.0));
    }

    // ── clear_hooks ────────────────────────────────────────────────────────

    #[test]
    fn test_clear_hooks_empties_everything() {
        let mut mgr = HookManager::new();
        mgr.register_hook(make_hook_config("h0", HookTrigger::EveryForward))
            .expect("register");
        mgr.register_hook(make_hook_config("h1", HookTrigger::EveryBackward))
            .expect("register");
        mgr.clear_hooks();
        assert!(mgr.get_all_hooks().is_empty());
        assert!(mgr.get_all_stats().is_empty());
    }

    // ── convenience builders ────────────────────────────────────────────────

    #[test]
    fn test_create_tensor_inspection_hook() {
        let mut mgr = HookManager::new();
        let id = mgr
            .create_tensor_inspection_hook(vec!["attention.*".to_string()])
            .expect("should succeed");
        assert!(mgr.get_hook(id).is_some());
    }

    #[test]
    fn test_create_gradient_tracking_hook() {
        let mut mgr = HookManager::new();
        let id = mgr
            .create_gradient_tracking_hook(vec!["fc.*".to_string()])
            .expect("should succeed");
        let hook = mgr.get_hook(id).expect("should exist");
        assert!(matches!(hook.trigger, HookTrigger::EveryBackward));
    }

    #[test]
    fn test_create_alert_hook() {
        let mut mgr = HookManager::new();
        let cond = HookCondition::StepRange { start: 0, end: 100 };
        let id = mgr
            .create_alert_hook(cond, "loss exploded".to_string(), AlertSeverity::Critical)
            .expect("should succeed");
        let hook = mgr.get_hook(id).expect("should exist");
        assert!(matches!(hook.trigger, HookTrigger::Conditional(_)));
    }

    // ── HookBuilder ────────────────────────────────────────────────────────

    #[test]
    fn test_hook_builder_basic() {
        let cfg = HookBuilder::new("my_hook")
            .trigger(HookTrigger::EveryNSteps(10))
            .action(HookAction::TrackGradients)
            .max_executions(50)
            .layer_patterns(vec!["norm".to_string()])
            .enabled(true)
            .build();

        assert_eq!(cfg.name, "my_hook");
        assert!(matches!(cfg.trigger, HookTrigger::EveryNSteps(10)));
        assert_eq!(cfg.max_executions, Some(50));
        assert!(cfg.enabled);
    }

    // ── enum variants ──────────────────────────────────────────────────────

    #[test]
    fn test_hook_trigger_variants() {
        let triggers: Vec<String> = vec![
            format!("{:?}", HookTrigger::EveryForward),
            format!("{:?}", HookTrigger::EveryBackward),
            format!("{:?}", HookTrigger::EveryNSteps(5)),
            format!("{:?}", HookTrigger::Once),
            format!("{:?}", HookTrigger::LayerSpecific(vec![])),
        ];
        for t in &triggers {
            assert!(!t.is_empty());
        }
    }

    #[test]
    fn test_hook_action_variants() {
        let actions: Vec<String> = vec![
            format!("{:?}", HookAction::InspectTensor),
            format!("{:?}", HookAction::TrackGradients),
            format!("{:?}", HookAction::RecordActivations),
            format!(
                "{:?}",
                HookAction::SaveSnapshot {
                    path: "/tmp".to_string()
                }
            ),
            format!(
                "{:?}",
                HookAction::Alert {
                    message: "x".to_string(),
                    severity: AlertSeverity::Info
                }
            ),
            format!(
                "{:?}",
                HookAction::CustomCallback {
                    name: "cb".to_string()
                }
            ),
            format!("{:?}", HookAction::PauseTraining),
        ];
        for a in &actions {
            assert!(!a.is_empty());
        }
    }

    #[test]
    fn test_alert_severity_variants() {
        let severities = [
            AlertSeverity::Info,
            AlertSeverity::Warning,
            AlertSeverity::Critical,
        ];
        for s in &severities {
            assert!(!format!("{:?}", s).is_empty());
        }
    }

    #[test]
    fn test_comparison_variants() {
        let comps = [
            Comparison::Greater,
            Comparison::Less,
            Comparison::Equal,
            Comparison::GreaterEqual,
            Comparison::LessEqual,
        ];
        for c in &comps {
            assert!(!format!("{:?}", c).is_empty());
        }
    }

    #[test]
    fn test_hook_stats_fields() {
        let id = Uuid::new_v4();
        let stats = HookStats {
            hook_id: id,
            hook_name: "perf_hook".to_string(),
            total_executions: 100,
            last_execution_step: Some(99),
            total_execution_time_ms: 500.0,
            avg_execution_time_ms: 5.0,
            errors: 2,
        };
        assert_eq!(stats.total_executions, 100);
        assert_eq!(stats.errors, 2);
        assert_eq!(stats.last_execution_step, Some(99));
    }
}
