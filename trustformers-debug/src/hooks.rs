//! Debugging hooks for automatic tensor and gradient tracking

use anyhow::Result;
use scirs2_core::ndarray::{ArrayD, IxDyn};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use uuid::Uuid;

use crate::activation_visualizer::ActivationVisualizer;
use crate::tensor_inspector::TensorInspector;

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
    /// Fires when the named key is present in
    /// [`HookContext::metadata`] -- the caller controls the flag.
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
    /// Real tensor/gradient inspector wired to [`HookAction::InspectTensor`]
    /// and [`HookAction::TrackGradients`]: both actions compute genuine
    /// statistics (mean/std/min/max/NaN & Inf counts, ...) over the tensor
    /// data the hook actually received, retrievable afterwards via
    /// [`Self::tensor_inspector`]. Neither action is a logged no-op.
    tensor_inspector: TensorInspector,
    /// Real activation recorder wired to [`HookAction::RecordActivations`]:
    /// registers the layer's real values with genuine statistics
    /// (mean/std/median/quartiles/sparsity/outliers), retrievable via
    /// [`Self::activation_visualizer`].
    activation_visualizer: ActivationVisualizer,
    /// Last tensor tracked in `tensor_inspector` (via `InspectTensor` or
    /// `RecordActivations`) for each layer name, so a later
    /// `TrackGradients` call on the same layer attaches real gradient
    /// statistics to that same entry via
    /// [`TensorInspector::inspect_gradients`] rather than creating an
    /// unrelated one. See [`Self::execute_action`].
    layer_tensor_ids: HashMap<String, Uuid>,
    /// Shared pause flag set by [`HookAction::PauseTraining`]. See the
    /// contract documented on [`Self::pause_flag`].
    pause_flag: Arc<AtomicBool>,
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
            .field("tensor_inspector", &self.tensor_inspector)
            .field("activation_visualizer", &self.activation_visualizer)
            .field("layer_tensor_ids", &self.layer_tensor_ids)
            .field("paused", &self.pause_flag.load(Ordering::Relaxed))
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
            tensor_inspector: TensorInspector::new(&crate::DebugConfig::default()),
            activation_visualizer: ActivationVisualizer::new(),
            layer_tensor_ids: HashMap::new(),
            pause_flag: Arc::new(AtomicBool::new(false)),
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

    /// Real statistics recorded by [`HookAction::InspectTensor`] and
    /// [`HookAction::TrackGradients`] hooks executed so far -- shapes,
    /// mean/std/min/max, NaN/Inf counts, per-tensor alerts, etc, computed
    /// from the actual tensor data each hook received.
    pub fn tensor_inspector(&self) -> &TensorInspector {
        &self.tensor_inspector
    }

    /// Mutable access to the wired [`TensorInspector`], e.g. to call
    /// [`TensorInspector::clear`] between epochs.
    pub fn tensor_inspector_mut(&mut self) -> &mut TensorInspector {
        &mut self.tensor_inspector
    }

    /// Real per-layer activation statistics recorded by
    /// [`HookAction::RecordActivations`] hooks executed so far.
    pub fn activation_visualizer(&self) -> &ActivationVisualizer {
        &self.activation_visualizer
    }

    /// Mutable access to the wired [`ActivationVisualizer`].
    pub fn activation_visualizer_mut(&mut self) -> &mut ActivationVisualizer {
        &mut self.activation_visualizer
    }

    /// Returns a clone of the shared pause flag that
    /// [`HookAction::PauseTraining`] sets.
    ///
    /// # Contract
    ///
    /// The hook system runs *inside* [`Self::execute_hooks`], called by
    /// whatever training loop is driving it -- it has no independent
    /// thread of control and therefore cannot itself halt that loop.
    /// What [`HookAction::PauseTraining`] truthfully *can* do, and does,
    /// is set this flag to `true` (see [`Self::execute_action`]). For
    /// "pause on hook" behaviour, the training loop must cooperate:
    ///
    ///  1. Once, after constructing the [`HookManager`], clone this flag
    ///     out with `manager.pause_flag()` and keep the `Arc` alongside
    ///     the loop state.
    ///  2. On each step (or at another convenient point), check
    ///     `flag.load(Ordering::SeqCst)` -- equivalently
    ///     [`Self::is_paused`] on the manager, if the loop still has
    ///     access to it -- and if `true`, actually stop advancing (block
    ///     for operator input, yield to a debug console, etc).
    ///  3. Call [`Self::resume_training`] (or `flag.store(false, ...)`
    ///     directly) once ready to continue.
    ///
    /// A training loop that never polls this flag is simply not pausable
    /// by hooks; [`HookAction::PauseTraining`] does not claim otherwise --
    /// it only guarantees the flag itself is set truthfully when it fires.
    pub fn pause_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.pause_flag)
    }

    /// `true` if a [`HookAction::PauseTraining`] hook has fired and
    /// nothing has called [`Self::resume_training`] since. See
    /// [`Self::pause_flag`] for the full contract.
    pub fn is_paused(&self) -> bool {
        self.pause_flag.load(Ordering::SeqCst)
    }

    /// Clears the pause flag set by [`HookAction::PauseTraining`]. See
    /// [`Self::pause_flag`] for the full contract.
    pub fn resume_training(&self) {
        self.pause_flag.store(false, Ordering::SeqCst);
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
        T: Clone + Into<f64> + 'static,
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

        // Convert tensor data to bytes for callbacks / snapshotting, which
        // want the tensor's raw in-memory representation regardless of `T`.
        let tensor_bytes = unsafe {
            std::slice::from_raw_parts(
                tensor_data.as_ptr() as *const u8,
                std::mem::size_of_val(tensor_data),
            )
        };
        // Real numeric values (widened via `T: Into<f64>`), used by the
        // analysis actions (InspectTensor / TrackGradients /
        // RecordActivations) to compute genuine statistics -- see
        // `execute_action`. Computed once here, before `T` is erased.
        let tensor_values: Vec<f64> = tensor_data.iter().cloned().map(Into::into).collect();

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
            let result =
                self.execute_single_hook(&hook_config, &context, tensor_bytes, &tensor_values);
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
                // A custom condition fires when the caller has put `name` into
                // the hook context's metadata. That IS the contract -- the
                // caller decides when the flag is present -- not a stand-in for
                // some richer predicate.
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
        tensor_values: &[f64],
    ) -> HookResult {
        for action in &hook.actions {
            match self.execute_action(action, context, tensor_data, tensor_values) {
                Ok(()) => continue,
                Err(e) => return HookResult::Error(e.to_string()),
            }
        }
        HookResult::Success
    }

    /// Reshape `tensor_values` into an `ArrayD<f64>` using the shape
    /// reported in `context`, for handoff to [`TensorInspector`]. Returns a
    /// structured error (never a fabricated/garbage array) if the element
    /// count does not match the declared shape.
    fn build_array(context: &HookContext, tensor_values: &[f64]) -> Result<ArrayD<f64>> {
        ArrayD::from_shape_vec(IxDyn(&context.tensor_shape), tensor_values.to_vec()).map_err(|e| {
            anyhow::anyhow!(
                "hook tensor shape {:?} does not match {} data element(s) reported for \
                     layer '{}': {}",
                context.tensor_shape,
                tensor_values.len(),
                context.layer_name,
                e
            )
        })
    }

    /// Real implementation shared by [`HookAction::InspectTensor`] and
    /// [`HookAction::RecordActivations`]-as-tensor-tracking: reshapes the
    /// real tensor values and hands them to [`TensorInspector::inspect_tensor`],
    /// then remembers the resulting id for this layer so a later
    /// `TrackGradients` call can attach gradient statistics to the same
    /// entry. Returns the real tensor id on success.
    fn inspect_and_track(
        &mut self,
        context: &HookContext,
        tensor_values: &[f64],
        operation: &str,
    ) -> Result<Uuid> {
        let array = Self::build_array(context, tensor_values)?;
        let id = self.tensor_inspector.inspect_tensor(
            &array,
            &context.layer_name,
            Some(context.layer_name.as_str()),
            Some(operation),
        )?;
        self.layer_tensor_ids.insert(context.layer_name.clone(), id);
        Ok(id)
    }

    fn execute_action(
        &mut self,
        action: &HookAction,
        context: &HookContext,
        tensor_data: &[u8],
        tensor_values: &[f64],
    ) -> Result<()> {
        match action {
            HookAction::InspectTensor => {
                let id = self.inspect_and_track(context, tensor_values, "hook: InspectTensor")?;
                tracing::debug!(
                    "Inspected tensor in layer '{}' at step {} -> tensor id {}",
                    context.layer_name,
                    context.step,
                    id
                );
                Ok(())
            },
            HookAction::TrackGradients => {
                // Attach real gradient statistics to the tensor previously
                // tracked for this layer (via InspectTensor /
                // RecordActivations on an earlier forward pass) when one
                // exists; otherwise this data becomes its own tracked
                // entry so it is never silently dropped.
                let id = if let Some(&existing) = self.layer_tensor_ids.get(&context.layer_name) {
                    let array = Self::build_array(context, tensor_values)?;
                    self.tensor_inspector.inspect_gradients(existing, &array)?;
                    existing
                } else {
                    self.inspect_and_track(
                        context,
                        tensor_values,
                        "hook: TrackGradients (no prior forward tensor tracked for this layer)",
                    )?
                };
                tracing::debug!(
                    "Tracked gradients in layer '{}' at step {} -> tensor id {}",
                    context.layer_name,
                    context.step,
                    id
                );
                Ok(())
            },
            HookAction::RecordActivations => {
                let values_f32: Vec<f32> = tensor_values.iter().map(|&v| v as f32).collect();
                self.activation_visualizer.register(
                    &context.layer_name,
                    values_f32,
                    context.tensor_shape.clone(),
                )?;
                tracing::debug!(
                    "Recorded {} activation value(s) in layer '{}' at step {}",
                    tensor_values.len(),
                    context.layer_name,
                    context.step
                );
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
                // Real action: flips the shared flag a training loop can
                // poll. See `Self::pause_flag` for the full cooperative
                // contract -- the hook system cannot halt the caller's
                // loop directly, only signal it truthfully.
                self.pause_flag.store(true, Ordering::SeqCst);
                tracing::warn!(
                    "Training paused by hook at step {} in layer '{}' -- poll \
                     HookManager::is_paused()/pause_flag() from the training loop to observe \
                     this, and call HookManager::resume_training() to clear it",
                    context.step,
                    context.layer_name
                );
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

    // ── execute_action honesty: InspectTensor / TrackGradients /
    //    RecordActivations / PauseTraining must really act, not just log ──
    //
    // Regression tests for the bug where these four actions logged a debug
    // message and returned `Ok(())` with no other effect. Each test below
    // would fail against that old behavior (no tensor would ever be
    // tracked, no activation ever registered, no flag ever set).

    fn make_action_hook(action: HookAction, trigger: HookTrigger) -> HookConfig {
        HookConfig {
            id: Uuid::new_v4(),
            name: "action_hook".to_string(),
            trigger,
            actions: vec![action],
            enabled: true,
            max_executions: None,
            layer_patterns: vec![],
        }
    }

    #[test]
    fn test_inspect_tensor_computes_real_statistics() {
        let mut mgr = HookManager::new();
        mgr.register_hook(make_action_hook(
            HookAction::InspectTensor,
            HookTrigger::EveryForward,
        ))
        .expect("register");

        let data = [2.0f64, 4.0, 6.0, 8.0];
        let results = mgr.execute_hooks("dense", &data, &[4], true, None);
        assert_eq!(results.len(), 1);
        assert!(
            matches!(results[0].1, HookResult::Success),
            "got {:?}",
            results[0].1
        );

        let tracked = mgr.tensor_inspector().get_all_tensors();
        assert_eq!(
            tracked.len(),
            1,
            "InspectTensor must actually register a tracked tensor"
        );
        let info = tracked[0];
        assert_eq!(info.layer_name.as_deref(), Some("dense"));
        // mean/min/max of [2,4,6,8] -- real, not a placeholder constant.
        assert!((info.stats.mean - 5.0).abs() < 1e-9);
        assert_eq!(info.stats.min, 2.0);
        assert_eq!(info.stats.max, 8.0);
        assert_eq!(info.stats.total_elements, 4);
    }

    #[test]
    fn test_inspect_tensor_flags_real_nan_alert() {
        let mut mgr = HookManager::new();
        mgr.register_hook(make_action_hook(
            HookAction::InspectTensor,
            HookTrigger::EveryForward,
        ))
        .expect("register");

        let data = [1.0f64, f64::NAN, 3.0];
        let results = mgr.execute_hooks("nan_layer", &data, &[3], true, None);
        assert!(matches!(results[0].1, HookResult::Success));

        let alerts = mgr.tensor_inspector().get_alerts();
        assert!(
            alerts.iter().any(|a| a.tensor_name == "nan_layer"
                && matches!(
                    a.alert_type,
                    crate::tensor_inspector::TensorAlertType::NaNValues
                )),
            "a real NaN in the tensor must produce a real NaN alert, got {:?}",
            alerts
        );
    }

    #[test]
    fn test_track_gradients_links_real_stats_to_prior_forward_tensor() {
        let mut mgr = HookManager::new();
        // `execute_hooks` returns results keyed by hook id in HashMap
        // iteration order (unspecified), so capture the ids up front and
        // look results up by id rather than assuming position 0.
        let inspect_id = mgr
            .register_hook(make_action_hook(
                HookAction::InspectTensor,
                HookTrigger::EveryForward,
            ))
            .expect("register forward hook");
        let grad_id = mgr
            .register_hook(make_action_hook(
                HookAction::TrackGradients,
                HookTrigger::EveryBackward,
            ))
            .expect("register backward hook");

        // Forward pass: activations. Only the EveryForward hook should fire.
        let activations = [1.0f64, 2.0, 3.0];
        let fwd = mgr.execute_hooks("linear", &activations, &[3], true, None);
        let fwd_result = fwd.iter().find(|(id, _)| *id == inspect_id).map(|(_, r)| r);
        assert!(
            matches!(fwd_result, Some(HookResult::Success)),
            "got {:?}",
            fwd_result
        );

        // Backward pass on the SAME layer: gradients, deliberately a
        // different distribution from the activations above. Only the
        // EveryBackward hook should fire.
        let gradients = [0.1f64, 0.2, 0.3];
        let bwd = mgr.execute_hooks("linear", &gradients, &[3], false, None);
        let bwd_result = bwd.iter().find(|(id, _)| *id == grad_id).map(|(_, r)| r);
        assert!(
            matches!(bwd_result, Some(HookResult::Success)),
            "got {:?}",
            bwd_result
        );

        let tracked = mgr.tensor_inspector().get_all_tensors();
        assert_eq!(
            tracked.len(),
            1,
            "gradient stats must attach to the existing forward tensor, not spawn a second one"
        );
        let grad_stats = tracked[0]
            .gradient_stats
            .as_ref()
            .expect("TrackGradients must populate gradient_stats with real data");
        assert!(
            (grad_stats.mean - 0.2).abs() < 1e-9,
            "gradient mean must reflect the real gradient values, got {}",
            grad_stats.mean
        );
        // The forward tensor's own stats must be untouched by the gradient call.
        assert!((tracked[0].stats.mean - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_track_gradients_without_prior_forward_still_tracks_real_data() {
        let mut mgr = HookManager::new();
        mgr.register_hook(make_action_hook(
            HookAction::TrackGradients,
            HookTrigger::EveryBackward,
        ))
        .expect("register");

        // No InspectTensor ever ran for "orphan" -- TrackGradients must not
        // silently no-op just because there is nothing to attach to.
        let gradients = [10.0f64, 20.0, 30.0];
        let results = mgr.execute_hooks("orphan", &gradients, &[3], false, None);
        assert!(matches!(results[0].1, HookResult::Success));

        let tracked = mgr.tensor_inspector().get_all_tensors();
        assert_eq!(tracked.len(), 1);
        assert!((tracked[0].stats.mean - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_record_activations_computes_real_statistics() {
        let mut mgr = HookManager::new();
        mgr.register_hook(make_action_hook(
            HookAction::RecordActivations,
            HookTrigger::EveryForward,
        ))
        .expect("register");

        let data = [0.0f64, 1.0, 2.0, 3.0];
        let results = mgr.execute_hooks("relu_1", &data, &[4], true, None);
        assert!(matches!(results[0].1, HookResult::Success));

        let recorded = mgr
            .activation_visualizer()
            .get_activations("relu_1")
            .expect("RecordActivations must register real activation data");
        assert_eq!(recorded.values, vec![0.0f32, 1.0, 2.0, 3.0]);
        assert!((recorded.statistics.mean - 1.5).abs() < 1e-6);
        assert_eq!(recorded.shape, vec![4]);
    }

    #[test]
    fn test_pause_training_sets_shared_flag_and_resume_clears_it() {
        let mut mgr = HookManager::new();
        let flag = mgr.pause_flag();
        assert!(!mgr.is_paused());
        assert!(!flag.load(Ordering::SeqCst));

        mgr.register_hook(make_action_hook(
            HookAction::PauseTraining,
            HookTrigger::EveryForward,
        ))
        .expect("register");

        let results = mgr.execute_hooks("any_layer", &[0.0f64], &[1], true, None);
        assert!(matches!(results[0].1, HookResult::Success));

        // Real, observable side effect: the SAME shared flag handed out
        // before the hook ever ran now reads true, and the manager agrees.
        assert!(
            flag.load(Ordering::SeqCst),
            "PauseTraining must set the real shared pause flag"
        );
        assert!(mgr.is_paused());

        mgr.resume_training();
        assert!(!mgr.is_paused());
        assert!(
            !flag.load(Ordering::SeqCst),
            "resume_training must clear the SAME shared flag"
        );
    }

    #[test]
    fn test_inspect_tensor_shape_mismatch_is_a_structured_error_not_silent_success() {
        let mut mgr = HookManager::new();
        mgr.register_hook(make_action_hook(
            HookAction::InspectTensor,
            HookTrigger::EveryForward,
        ))
        .expect("register");

        // 3 real values, but a shape claiming 10 -- must surface as an
        // error, never silently succeed or fabricate padding.
        let data = [1.0f64, 2.0, 3.0];
        let results = mgr.execute_hooks("mismatched", &data, &[10], true, None);
        assert_eq!(results.len(), 1);
        match &results[0].1 {
            HookResult::Error(msg) => {
                assert!(
                    msg.contains("mismatched"),
                    "error should name the layer: {}",
                    msg
                );
            },
            other => panic!("expected a structured Error, got {:?}", other),
        }
        assert!(
            mgr.tensor_inspector().get_all_tensors().is_empty(),
            "a shape mismatch must not fabricate a tracked tensor"
        );
    }
}
