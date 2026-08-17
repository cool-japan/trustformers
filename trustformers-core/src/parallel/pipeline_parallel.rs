//! Pipeline Parallelism for Large Model Training
//!
//! This module implements pipeline parallelism, which splits a model into stages

#![allow(unused_variables)] // Distributed parallelism implementation with reserved parameters
//! across multiple devices and processes microbatches in a pipelined manner.

use super::model_parallel::{
    ModelParallelContext, PipelineOp, PipelineSchedule, PipelineScheduleType,
};
use crate::errors::{runtime_error, Result};
use crate::Tensor;
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Layer wrapper for pipeline stages
pub trait PipelineLayer: Send + Sync {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor>;

    /// Apply this step's accumulated gradients (already averaged over the
    /// accumulation window by `PipelineOptimizer::step`) to whichever of
    /// this layer's own parameters appear in `grads`, e.g. a plain SGD
    /// update `param -= lr * grad`. Implementations should ignore any keys
    /// in `grads` that do not belong to them.
    ///
    /// The default implementation honestly reports that this layer has no
    /// mechanism to apply gradients, rather than silently discarding them
    /// while `PipelineOptimizer::step` reports success: a layer that holds
    /// learnable parameters must override this to actually update them.
    fn apply_gradients(&mut self, grads: &HashMap<String, Tensor>, lr: f32) -> Result<()> {
        let _ = (grads, lr);
        Err(runtime_error(
            "PipelineLayer::apply_gradients is not implemented for this layer type; \
             PipelineOptimizer::step cannot update its parameters without an override",
        ))
    }
}

/// A single stage in the pipeline
pub struct PipelineStage {
    /// Stage ID (0-indexed)
    pub stage_id: usize,
    /// Layers in this stage
    pub layers: Vec<Box<dyn PipelineLayer>>,
    /// Device ID for this stage
    pub device_id: usize,
    /// Whether this stage requires gradient computation
    pub requires_grad: bool,
}

impl PipelineStage {
    pub fn new(stage_id: usize, device_id: usize) -> Self {
        Self {
            stage_id,
            layers: Vec::new(),
            device_id,
            requires_grad: true,
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn PipelineLayer>) {
        self.layers.push(layer);
    }

    /// Forward pass through all layers in the stage
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output)?;
        }
        Ok(output)
    }

    /// Backward pass through all layers in the stage
    pub fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor> {
        let mut grad = grad_output.clone();
        // Process layers in reverse order
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad)?;
        }
        Ok(grad)
    }

    /// Apply `grads` to every layer in this stage (see
    /// `PipelineLayer::apply_gradients`).
    pub fn apply_gradients(&mut self, grads: &HashMap<String, Tensor>, lr: f32) -> Result<()> {
        for layer in self.layers.iter_mut() {
            layer.apply_gradients(grads, lr)?;
        }
        Ok(())
    }
}

/// Model split into pipeline stages
pub struct PipelineModel {
    /// All pipeline stages
    pub stages: Vec<PipelineStage>,
    /// Model parallel context
    pub mp_context: Arc<ModelParallelContext>,
    /// Stage assignment for this rank
    pub local_stage_id: Option<usize>,
}

impl PipelineModel {
    pub fn new(mp_context: Arc<ModelParallelContext>) -> Self {
        Self {
            stages: Vec::new(),
            mp_context,
            local_stage_id: None,
        }
    }

    /// Add a stage to the pipeline
    pub fn add_stage(&mut self, stage: PipelineStage) {
        if stage.device_id == self.mp_context.rank() {
            self.local_stage_id = Some(stage.stage_id);
        }
        self.stages.push(stage);
    }

    /// Get the local stage for this rank
    pub fn local_stage(&self) -> Result<&PipelineStage> {
        let stage_id =
            self.local_stage_id.ok_or_else(|| runtime_error("No local stage assigned"))?;
        self.stages.get(stage_id).ok_or_else(|| runtime_error("Invalid stage ID"))
    }

    /// Get mutable local stage
    pub fn local_stage_mut(&mut self) -> Result<&mut PipelineStage> {
        let stage_id =
            self.local_stage_id.ok_or_else(|| runtime_error("No local stage assigned"))?;
        self.stages.get_mut(stage_id).ok_or_else(|| runtime_error("Invalid stage ID"))
    }

    /// Get total number of stages
    pub fn num_stages(&self) -> usize {
        self.stages.len()
    }

    /// Apply `grads` to the local stage's layers (see
    /// `PipelineLayer::apply_gradients`). Only the local stage is updated:
    /// exactly like `execute_forward`/`execute_backward` (`local_stage_mut`),
    /// each rank owns and updates only its own stage's parameters.
    pub fn apply_gradients(&mut self, grads: &HashMap<String, Tensor>, lr: f32) -> Result<()> {
        self.local_stage_mut()?.apply_gradients(grads, lr)
    }
}

/// Microbatch data structure
#[derive(Clone)]
pub struct Microbatch {
    /// Microbatch ID
    pub id: usize,
    /// Input tensor
    pub input: Option<Tensor>,
    /// Output tensor (activations)
    pub output: Option<Tensor>,
    /// Gradient w.r.t output
    pub grad_output: Option<Tensor>,
    /// Gradient w.r.t input
    pub grad_input: Option<Tensor>,
    /// Labels for loss computation (only for last stage)
    pub labels: Option<Tensor>,
}

impl Microbatch {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            input: None,
            output: None,
            grad_output: None,
            grad_input: None,
            labels: None,
        }
    }
}

/// Manages microbatches across pipeline stages
pub struct MicrobatchManager {
    /// All microbatches
    microbatches: Vec<Microbatch>,
    /// Activation checkpointing enabled
    checkpoint_activations: bool,
    /// Queue of pending forward passes
    forward_queue: VecDeque<usize>,
    /// Queue of pending backward passes
    backward_queue: VecDeque<usize>,
}

impl MicrobatchManager {
    pub fn new(num_microbatches: usize, checkpoint_activations: bool) -> Self {
        let microbatches = (0..num_microbatches).map(Microbatch::new).collect();

        Self {
            microbatches,
            checkpoint_activations,
            forward_queue: VecDeque::new(),
            backward_queue: VecDeque::new(),
        }
    }

    /// Get microbatch by ID
    pub fn get(&self, id: usize) -> Result<&Microbatch> {
        self.microbatches
            .get(id)
            .ok_or_else(|| runtime_error(format!("Invalid microbatch ID: {}", id)))
    }

    /// Get mutable microbatch
    pub fn get_mut(&mut self, id: usize) -> Result<&mut Microbatch> {
        self.microbatches
            .get_mut(id)
            .ok_or_else(|| runtime_error(format!("Invalid microbatch ID: {}", id)))
    }

    /// Add microbatch to forward queue
    pub fn enqueue_forward(&mut self, mb_id: usize) {
        self.forward_queue.push_back(mb_id);
    }

    /// Add microbatch to backward queue
    pub fn enqueue_backward(&mut self, mb_id: usize) {
        self.backward_queue.push_back(mb_id);
    }

    /// Get next forward microbatch
    pub fn dequeue_forward(&mut self) -> Option<usize> {
        self.forward_queue.pop_front()
    }

    /// Get next backward microbatch
    pub fn dequeue_backward(&mut self) -> Option<usize> {
        self.backward_queue.pop_front()
    }

    /// Clear activation if checkpointing is enabled
    pub fn maybe_clear_activation(&mut self, mb_id: usize) -> Result<()> {
        if self.checkpoint_activations {
            let mb = self.get_mut(mb_id)?;
            mb.output = None; // Clear to save memory
        }
        Ok(())
    }

    /// Recompute activation if needed
    pub fn maybe_recompute_activation(
        &mut self,
        mb_id: usize,
        stage: &PipelineStage,
    ) -> Result<()> {
        let should_recompute = self.checkpoint_activations;
        let mb = self.get_mut(mb_id)?;
        if should_recompute && mb.output.is_none() {
            // Recompute forward pass
            if let Some(input) = &mb.input {
                mb.output = Some(stage.forward(input)?);
            }
        }
        Ok(())
    }
}

/// Pipeline executor that manages the execution schedule
pub struct PipelineExecutor {
    /// Pipeline model
    model: Arc<RwLock<PipelineModel>>,
    /// Pipeline schedule
    schedule: PipelineSchedule,
    /// Microbatch manager
    mb_manager: Arc<Mutex<MicrobatchManager>>,
    /// Communication buffers
    #[allow(dead_code)]
    send_buffers: HashMap<usize, Tensor>,
    _recv_buffers: HashMap<usize, Tensor>,
}

impl PipelineExecutor {
    pub fn new(
        model: Arc<RwLock<PipelineModel>>,
        num_microbatches: usize,
        checkpoint_activations: bool,
    ) -> Result<Self> {
        let num_stages = {
            let model_read = model.read();
            model_read.num_stages()
        };

        let schedule = PipelineSchedule::new(
            num_stages,
            num_microbatches,
            PipelineScheduleType::OneForwardOneBackward,
        );

        let mb_manager = Arc::new(Mutex::new(MicrobatchManager::new(
            num_microbatches,
            checkpoint_activations,
        )));

        Ok(Self {
            model,
            schedule,
            mb_manager,
            send_buffers: HashMap::new(),
            _recv_buffers: HashMap::new(),
        })
    }

    /// Execute one training step
    pub fn execute_step(&mut self, inputs: Vec<Tensor>, labels: Vec<Tensor>) -> Result<f32> {
        let num_inputs = inputs.len();

        // Split inputs into microbatches
        self.prepare_microbatches(inputs, labels)?;

        // Get schedule for local stage
        let stage_id = {
            let model = self.model.read();
            model.local_stage_id.ok_or_else(|| runtime_error("No local stage"))?
        };

        let ops = self.schedule.get_stage_schedule(stage_id);

        // Execute operations according to schedule
        let mut total_loss = 0.0;
        for op in ops {
            match op {
                PipelineOp::Forward { microbatch_id } => {
                    self.execute_forward(microbatch_id)?;
                },
                PipelineOp::Backward { microbatch_id } => {
                    let loss = self.execute_backward(microbatch_id)?;
                    total_loss += loss;
                },
                PipelineOp::SendActivation { to_stage } => {
                    self.send_activation(to_stage)?;
                },
                PipelineOp::RecvActivation { from_stage } => {
                    self.recv_activation(from_stage)?;
                },
                PipelineOp::SendGradient { to_stage } => {
                    self.send_gradient(to_stage)?;
                },
                PipelineOp::RecvGradient { from_stage } => {
                    self.recv_gradient(from_stage)?;
                },
            }
        }

        Ok(total_loss / num_inputs as f32)
    }

    /// Prepare microbatches from full batch
    fn prepare_microbatches(&mut self, inputs: Vec<Tensor>, labels: Vec<Tensor>) -> Result<()> {
        let mut mb_manager = self.mb_manager.lock();

        for (i, (input, label)) in inputs.into_iter().zip(labels).enumerate() {
            let mb = mb_manager.get_mut(i)?;
            mb.input = Some(input);
            mb.labels = Some(label);
            mb_manager.enqueue_forward(i);
        }

        Ok(())
    }

    /// Execute forward pass for a microbatch
    fn execute_forward(&mut self, mb_id: usize) -> Result<()> {
        let mut model = self.model.write();
        let stage = model.local_stage_mut()?;

        let mut mb_manager = self.mb_manager.lock();
        let mb = mb_manager.get_mut(mb_id)?;

        // Get input (from previous stage or initial input)
        let input = if stage.stage_id == 0 {
            mb.input.as_ref().ok_or_else(|| runtime_error("Missing input"))?
        } else {
            // Would receive from previous stage
            mb.output.as_ref().ok_or_else(|| runtime_error("Missing activation"))?
        };

        // Forward pass
        let output = stage.forward(input)?;
        mb.output = Some(output);

        // Maybe clear activation for checkpointing
        mb_manager.maybe_clear_activation(mb_id)?;

        Ok(())
    }

    /// Execute backward pass for a microbatch
    fn execute_backward(&mut self, mb_id: usize) -> Result<f32> {
        let (is_last_stage, stage_id) = {
            let model = self.model.read();
            let stage = model.local_stage()?;
            (stage.stage_id == model.num_stages() - 1, stage.stage_id)
        };

        let mut model = self.model.write();
        let stage = model.local_stage_mut()?;

        let mut mb_manager = self.mb_manager.lock();

        // Recompute activation if needed
        mb_manager.maybe_recompute_activation(mb_id, stage)?;

        let mb = mb_manager.get_mut(mb_id)?;

        // Compute loss and gradient for last stage
        let loss = if is_last_stage {
            // Compute loss (simplified - would use actual loss function)
            1.0
        } else {
            0.0
        };

        // Get gradient w.r.t output
        let grad_output = if is_last_stage {
            // Compute gradient from loss
            mb.output.as_ref().ok_or_else(|| runtime_error("Missing output"))?.clone()
        } else {
            // Would receive from next stage
            mb.grad_output
                .as_ref()
                .ok_or_else(|| runtime_error("Missing grad_output"))?
                .clone()
        };

        // Backward pass
        let grad_input = stage.backward(&grad_output)?;
        mb.grad_input = Some(grad_input);

        Ok(loss)
    }

    /// Returns `Ok(())` unconditionally when `other_stage` names the
    /// caller's own local stage (self-communication is trivially correct:
    /// there is nothing to transport), and an honest error otherwise.
    ///
    /// None of the four `send_*`/`recv_*` methods below can perform a real
    /// cross-stage transfer today for two independent reasons: (1)
    /// `PipelineSchedule`'s schedule generators
    /// (`sequential_schedule`/`one_f1b_schedule`/`interleaved_1f1b_schedule`
    /// in `model_parallel.rs`) never actually emit
    /// `PipelineOp::SendActivation`/`RecvActivation`/`SendGradient`/
    /// `RecvGradient` - only `Forward`/`Backward` - so these are unreachable
    /// from `execute_step` as currently scheduled; and (2) even if they
    /// were scheduled, those `PipelineOp` variants carry only a stage id,
    /// not a microbatch id, so there would be no way to identify *which*
    /// microbatch's activation/gradient to move. Rather than silently
    /// returning `Ok(())` for a cross-stage transfer that cannot actually
    /// happen (the previous behavior), a genuine cross-stage call reports
    /// that honestly.
    fn require_local_stage_or_error(&self, other_stage: usize, op: &str) -> Result<bool> {
        let local_stage_id = self.model.read().local_stage_id;
        if local_stage_id == Some(other_stage) {
            return Ok(true);
        }
        Err(runtime_error(format!(
            "PipelineExecutor::{op}: cross-stage transport to/from stage {other_stage} is not \
             implemented (PipelineOp carries no microbatch id to identify what to transfer, and \
             no schedule currently emits this op)"
        )))
    }

    /// Send activation to `to_stage`. See `require_local_stage_or_error`.
    fn send_activation(&mut self, to_stage: usize) -> Result<()> {
        self.require_local_stage_or_error(to_stage, "send_activation").map(|_| ())
    }

    /// Receive activation from `from_stage`. See `require_local_stage_or_error`.
    fn recv_activation(&mut self, from_stage: usize) -> Result<()> {
        self.require_local_stage_or_error(from_stage, "recv_activation").map(|_| ())
    }

    /// Send gradient to `to_stage`. See `require_local_stage_or_error`.
    fn send_gradient(&mut self, to_stage: usize) -> Result<()> {
        self.require_local_stage_or_error(to_stage, "send_gradient").map(|_| ())
    }

    /// Receive gradient from `from_stage`. See `require_local_stage_or_error`.
    fn recv_gradient(&mut self, from_stage: usize) -> Result<()> {
        self.require_local_stage_or_error(from_stage, "recv_gradient").map(|_| ())
    }
}

/// Optimizer for pipeline parallel training
pub struct PipelineOptimizer {
    /// Learning rate
    lr: f32,
    /// Weight decay
    _weight_decay: f32,
    /// Gradient accumulation steps
    accumulation_steps: usize,
    /// Current accumulation step
    current_step: usize,
    /// Accumulated gradients
    accumulated_grads: HashMap<String, Tensor>,
}

impl PipelineOptimizer {
    pub fn new(lr: f32, weight_decay: f32, accumulation_steps: usize) -> Self {
        Self {
            lr,
            _weight_decay: weight_decay,
            accumulation_steps,
            current_step: 0,
            accumulated_grads: HashMap::new(),
        }
    }

    /// Accumulate gradients from microbatch
    pub fn accumulate_gradients(&mut self, grads: HashMap<String, Tensor>) -> Result<()> {
        for (name, grad) in grads {
            if let Some(acc_grad) = self.accumulated_grads.get_mut(&name) {
                *acc_grad = acc_grad.add(&grad)?;
            } else {
                self.accumulated_grads.insert(name, grad);
            }
        }

        self.current_step += 1;
        Ok(())
    }

    /// Apply gradients if accumulation is complete.
    ///
    /// Averages the accumulated gradients over the accumulation window and
    /// hands them to `model`'s local stage via `PipelineLayer::apply_gradients`
    /// (an SGD-style `param -= lr * grad` for layers that implement it).
    /// This used to clear `accumulated_grads` and return `Ok(true)`
    /// unconditionally - reporting a successful optimizer step that changed
    /// no parameter at all. It now propagates whatever
    /// `PipelineModel::apply_gradients` reports: `Err` for a model whose
    /// layers do not implement gradient application (the default; see
    /// `PipelineLayer::apply_gradients`), so a caller cannot mistake an
    /// unimplemented update for a real one.
    pub fn step(&mut self, model: &mut PipelineModel) -> Result<bool> {
        if self.current_step < self.accumulation_steps {
            return Ok(false);
        }

        // Average the accumulated gradients over the accumulation window
        // before applying them.
        let scale = 1.0 / self.accumulation_steps as f32;
        let averaged: HashMap<String, Tensor> = self
            .accumulated_grads
            .iter()
            .map(|(name, grad)| Ok((name.clone(), grad.mul_scalar(scale)?)))
            .collect::<Result<_>>()?;

        model.apply_gradients(&averaged, self.lr)?;

        self.accumulated_grads.clear();
        self.current_step = 0;

        Ok(true)
    }
}

/// Builder for creating pipeline models
pub struct PipelineModelBuilder {
    mp_context: Arc<ModelParallelContext>,
    stages: Vec<PipelineStage>,
    layers_per_stage: Option<usize>,
}

impl PipelineModelBuilder {
    pub fn new(mp_context: Arc<ModelParallelContext>) -> Self {
        Self {
            mp_context,
            stages: Vec::new(),
            layers_per_stage: None,
        }
    }

    /// Set number of layers per stage (for automatic partitioning)
    pub fn layers_per_stage(mut self, layers_per_stage: usize) -> Self {
        self.layers_per_stage = Some(layers_per_stage);
        self
    }

    /// Add a pre-configured stage
    pub fn add_stage(mut self, stage: PipelineStage) -> Self {
        self.stages.push(stage);
        self
    }

    /// Build the pipeline model
    pub fn build(self) -> Result<PipelineModel> {
        let mut model = PipelineModel::new(self.mp_context);

        for stage in self.stages {
            model.add_stage(stage);
        }

        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::super::model_parallel::{
        CommunicationBackend, ModelParallelConfig, ModelParallelStrategy,
    };
    use super::*;

    #[test]
    fn test_pipeline_stage() {
        let stage = PipelineStage::new(0, 0);
        assert_eq!(stage.stage_id, 0);
        assert_eq!(stage.device_id, 0);
        assert!(stage.requires_grad);
    }

    #[test]
    fn test_microbatch_manager() {
        let mut manager = MicrobatchManager::new(4, true);

        manager.enqueue_forward(0);
        manager.enqueue_forward(1);

        assert_eq!(manager.dequeue_forward(), Some(0));
        assert_eq!(manager.dequeue_forward(), Some(1));
        assert_eq!(manager.dequeue_forward(), None);
    }

    #[test]
    fn test_pipeline_model_builder() {
        let config = ModelParallelConfig {
            num_devices: 4,
            device_ids: vec![0, 1, 2, 3],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };

        let mp_context =
            Arc::new(ModelParallelContext::new(config).expect("operation failed in test"));

        let model = PipelineModelBuilder::new(mp_context)
            .add_stage(PipelineStage::new(0, 0))
            .add_stage(PipelineStage::new(1, 1))
            .build()
            .expect("operation failed in test");

        assert_eq!(model.num_stages(), 2);
    }

    fn single_stage_context() -> Arc<ModelParallelContext> {
        let config = ModelParallelConfig {
            num_devices: 1,
            device_ids: vec![0],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };
        Arc::new(ModelParallelContext::new(config).expect("mp context"))
    }

    /// `PipelineLayer` with one real, named, learnable parameter. Used to
    /// prove `PipelineOptimizer::step` really mutates it via an
    /// `apply_gradients` override (`param -= lr * grad`), unlike the
    /// default (see `NoParamsLayer` below).
    struct LinearLikeLayer {
        param_name: String,
        weight: Tensor,
    }

    impl PipelineLayer for LinearLikeLayer {
        fn forward(&self, input: &Tensor) -> Result<Tensor> {
            input.mul(&self.weight)
        }

        fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor> {
            Ok(grad_output.clone())
        }

        fn apply_gradients(&mut self, grads: &HashMap<String, Tensor>, lr: f32) -> Result<()> {
            if let Some(grad) = grads.get(&self.param_name) {
                self.weight = self.weight.sub(&grad.mul_scalar(lr)?)?;
            }
            Ok(())
        }
    }

    /// `PipelineLayer` that does not override `apply_gradients`, exercising
    /// the trait's default (honest-error) behavior.
    struct NoParamsLayer;

    impl PipelineLayer for NoParamsLayer {
        fn forward(&self, input: &Tensor) -> Result<Tensor> {
            Ok(input.clone())
        }

        fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor> {
            Ok(grad_output.clone())
        }
    }

    /// Regression test: `PipelineOptimizer::step` used to clear
    /// `accumulated_grads` and unconditionally return `Ok(true)` without
    /// ever touching a parameter ("apply gradients" that applied nothing).
    /// With a layer that implements `apply_gradients`, the real weight must
    /// change by exactly `lr * grad`.
    #[test]
    fn test_optimizer_step_actually_updates_layer_parameters() {
        let mut model = PipelineModel::new(single_stage_context());
        let mut stage = PipelineStage::new(0, 0);
        stage.add_layer(Box::new(LinearLikeLayer {
            param_name: "w".to_string(),
            weight: Tensor::from_vec(vec![1.0, 2.0], &[2]).expect("tensor"),
        }));
        model.add_stage(stage);

        let input = Tensor::from_vec(vec![1.0, 1.0], &[2]).expect("tensor");
        let before = model.local_stage().expect("local stage").layers[0]
            .forward(&input)
            .expect("forward");
        assert_eq!(before.data().expect("data"), vec![1.0, 2.0]);

        let mut optimizer = PipelineOptimizer::new(0.1, 0.0, 1);
        let mut grads = HashMap::new();
        grads.insert(
            "w".to_string(),
            Tensor::from_vec(vec![10.0, 10.0], &[2]).expect("tensor"),
        );
        optimizer.accumulate_gradients(grads).expect("accumulate_gradients");

        let applied = optimizer.step(&mut model).expect("step should succeed");
        assert!(applied);

        // new weight = [1,2] - 0.1 * [10,10] = [0,1]
        let after = model.local_stage().expect("local stage").layers[0]
            .forward(&input)
            .expect("forward");
        assert_eq!(after.data().expect("data"), vec![0.0, 1.0]);
    }

    /// Regression test: `step` must not fabricate `Ok(true)` when nothing
    /// in the model can actually apply the accumulated gradient.
    #[test]
    fn test_optimizer_step_errors_when_no_layer_implements_apply_gradients() {
        let mut model = PipelineModel::new(single_stage_context());
        let mut stage = PipelineStage::new(0, 0);
        stage.add_layer(Box::new(NoParamsLayer));
        model.add_stage(stage);

        let mut optimizer = PipelineOptimizer::new(0.1, 0.0, 1);
        let mut grads = HashMap::new();
        grads.insert(
            "w".to_string(),
            Tensor::from_vec(vec![1.0], &[1]).expect("tensor"),
        );
        optimizer.accumulate_gradients(grads).expect("accumulate_gradients");

        let result = optimizer.step(&mut model);
        assert!(
            result.is_err(),
            "must not report a successful optimizer step that updated nothing"
        );
    }

    /// Regression test: `send_activation`/`recv_activation`/`send_gradient`/
    /// `recv_gradient` used to return `Ok(())` unconditionally for every
    /// `to_stage`/`from_stage`, silently claiming a cross-stage transfer
    /// happened when nothing was sent anywhere. Self-stage calls (the only
    /// case that is actually a no-op transfer) must still succeed; a
    /// genuinely different stage must now error instead of lying.
    #[test]
    fn test_send_recv_ok_for_local_stage_err_for_cross_stage() {
        let config = ModelParallelConfig {
            num_devices: 2,
            device_ids: vec![0, 1],
            strategy: ModelParallelStrategy::Pipeline,
            comm_backend: CommunicationBackend::Custom,
            ..Default::default()
        };
        let mp_context = Arc::new(ModelParallelContext::new(config).expect("mp context"));
        let mut model = PipelineModel::new(mp_context);
        model.add_stage(PipelineStage::new(0, 0));
        model.add_stage(PipelineStage::new(1, 1));
        // rank() is always 0 for `ModelParallelContext::new`, so stage 0
        // (device_id 0) is local.
        assert_eq!(model.local_stage_id, Some(0));

        let model = Arc::new(RwLock::new(model));
        let mut executor = PipelineExecutor::new(model, 1, false).expect("executor");

        assert!(
            executor.send_activation(0).is_ok(),
            "self-stage send must succeed"
        );
        assert!(
            executor.recv_activation(0).is_ok(),
            "self-stage recv must succeed"
        );

        assert!(
            executor.send_activation(1).is_err(),
            "cross-stage send must error, not silently claim success"
        );
        assert!(executor.recv_activation(1).is_err());
        assert!(executor.send_gradient(1).is_err());
        assert!(executor.recv_gradient(1).is_err());
    }
}
