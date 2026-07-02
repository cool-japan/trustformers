# trustformers-training

Training infrastructure for TrustformeRS.

## Current State

**Version:** 0.1.4 | **Status:** Alpha | **Updated:** 2026-07-02

This crate provides HuggingFace-`Trainer`-inspired training infrastructure: a core `Trainer`/`TrainingArguments` loop (plus a simpler `SimpleTrainer` builder API), mixed-precision/AMP, quantization-aware training, RLHF (PPO/DPO), few-shot and meta-learning, continual learning, hyperparameter optimization, a large data-pipeline/augmentation/curriculum system, and a family of distributed/parallel-training abstractions (tensor, sequence, 3D, expert and ring-attention parallelism, plus elastic and multi-cloud orchestration).

- **~930 tests passing** (workspace-wide: 18,102 passed / 0 failed, 0 clippy warnings, 0 rustdoc warnings — verified 2026-07-01)
- **1,673 public API items** (`pub fn`/`struct`/`enum`/`trait`, incl. impl-block methods) reachable from `lib.rs`
- **59,720 SLoC** actually compiled into the crate (69 `.rs` files wired into the module tree; the full `src/` tree on disk is 89,914 lines across 102 files — see [Verification Notes](#verification-notes))
- **0 stub/placeholder implementations** (`todo!()`/`unimplemented!()`/TODO/FIXME/HACK/XXX/"placeholder", searched case-insensitively) in the compiled source
- **0 `.unwrap()` calls** in the compiled production source

> **Honest maturity note**: The non-distributed core (the `Trainer`/`SimpleTrainer` loop, losses/metrics, mixed precision, QAT, RLHF, few-shot/meta-learning, continual learning, hyperparameter optimization, and the training-stability/monitoring stack) is genuinely implemented, tested, and free of stubs. The **distributed/multi-node story is weaker than earlier drafts of this document claimed**: there is no ZeRO optimizer (stages 1/2/3) anywhere in the source, and the `NCCL`/`Gloo`/`MPI` backends in `distributed.rs` are in-process simulations of the collective-communication API (`all_reduce`/`broadcast`/`reduce`/`barrier` scale tensors locally; the source comments say explicitly "in a real implementation, this would call `ncclAllReduce`/`MPI_Allreduce`/..."). There is also a substantial amount of code sitting in `src/` that is **not wired into the crate at all** — see [Verification Notes](#verification-notes) and `TODO.md`. This crate is therefore labeled **Alpha** rather than Stable: the tested, reachable surface is solid, but the headline distributed-training claims need to be read with that caveat, and 0.1.x semver means the API can still move.

## Features

### Core Training Infrastructure
- **`Trainer`**: the main training loop (`trainer.rs`) — `Trainer::new(model, args, optimizer, loss_fn, task_type)`, gradient accumulation, checkpointing hooks, `TrainerCallback` trait, `EarlyStoppingCallback`
- **`SimpleTrainer` / `SimpleTrainerBuilder`**: an alternative, simpler builder-style API (`simplified_trainer.rs`) with `CheckpointCallback`, `LoggingCallback`, `MetricsCallback`, `ProgressCallback`
- **`TrainingArguments`**: configuration struct with `EvaluationStrategy`/`SaveStrategy` (`training_args.rs`)
- **Losses**: `CrossEntropyLoss` (with `with_label_smoothing()`), `MSELoss`, the `Loss` trait (`losses.rs`)
- **Metrics**: `Accuracy`, `F1Score`, `Perplexity`, `MetricCollection`, the `Metric` trait (`metrics.rs`)
- **Config validation**: `ConfigValidator`, `ConfigSchema`, `Constraint`, `Validatable` (`config_validation.rs`)
- **Structured error handling**: `TrainingError`, `ErrorManager`, `RecoveryAction`/`RecoveryStrategy`, `ErrorCodeRegistry` with `get_recovery_actions`/`is_critical_error` (`error_codes.rs`, `error_handling.rs`)
- **Training orchestration**: `TrainingOrchestrator`, `JobScheduler`, `TrainingJob`, `CheckpointConfig`/`CheckpointInfo` (`training_orchestration.rs`)

### Training Stability & Monitoring
- **`AdvancedStabilityMonitor`**: loss-landscape analysis, anomaly prediction, `RiskLevel`/`StabilityScore` (`advanced_stability_monitor.rs`)
- **`GradientRecoveryManager`**: gradient-anomaly detection and recovery strategies (`gradient_anomaly_recovery.rs`)
- **`AdaptiveGradientScaler`** / **`AdaptiveLearningRateScheduler`**: dynamic gradient-scale and LR adaptation based on observed training dynamics (`adaptive_gradient_scaling.rs`, `adaptive_learning_rate.rs`)
- **`TrainingDynamicsAnalyzer`** / **`TrainingMonitor`**: convergence/gradient-flow/weight-evolution metrics, health status and anomaly reports (`training_dynamics.rs`, `training_monitor.rs`)

### Distributed & Parallel Training
- **`DataParallelTrainer`** with a `ProcessGroup` trait and `DistributedBackend` (`NCCL`/`Gloo`/`MPI`/`Simulated`) selector (`distributed.rs`) — see the maturity note above regarding the NCCL/Gloo/MPI implementations
- **Tensor parallelism**: `TensorParallelism`, `TensorPartitioningStrategy` (`tensor_parallelism.rs`)
- **Sequence parallelism**: `SequenceParallelism`, `SequenceSplittingStrategy` (`sequence_parallelism.rs`)
- **3D parallelism** (data + tensor + pipeline) with `PipelineSchedule` variants `GPipe`/`PipeDream`/`PipeDream2BW`/`Interleaved1F1B`/`Adaptive` (`parallelism_3d.rs`)
- **Expert parallelism** (MoE-style routing): `ExpertParallelism`, `ExpertRoutingStrategy`, `TokenRouting` (`expert_parallelism.rs`)
- **Ring attention** for long-sequence distributed attention: `RingAttentionBlock`, `RingAttentionManager` (`ring_attention.rs`)
- **Hardware-aware auto-parallelism selection**: `AutoParallelismSelector` picks a strategy from `HardwareConstraints`/`ModelConstraints`/`NetworkTopology` (`auto_parallelism.rs`)
- **Elastic training**: `ElasticTrainingCoordinator` (worker heartbeats, scaling decisions, mid-training checkpoints) (`elastic_training.rs`)
- **Multi-cloud orchestration**: `MultiCloudOrchestrator`, `CloudProvider`, `CloudScheduler`, cost/budget-aware scheduling (`multicloud.rs`)
- **Resource scheduling**: `ResourceScheduler`, `ResourcePool`, cost-aware allocation (`resource_scheduling.rs`)
- *(No ZeRO stage-1/2/3 optimizer exists in this crate today — see the maturity note above.)*

### Mixed Precision & Quantization
- **AMP**: `AMPManager`, `MixedPrecisionConfig`, `LossScaler`, `DynamicBatchingManager` (`mixed_precision.rs`)
- **Quantization-Aware Training (QAT)**: `QATTrainer`, `QATConfig`, `fake_quantize`/`fake_quantize_mixed_bit`, `ActivationQuantizer`, per-tensor and per-channel schemes, `MixedBitQATTrainer` (`qat.rs`)

### RLHF and Alignment (`rlhf` module)
- **PPO**: `PPOTrainer`, `PPOConfig`, `PPOStepResult`, `PolicyModel`, `ValueModel` (`rlhf/ppo.rs`)
- **DPO**: `DPOTrainer`, `DPOConfig`, `DPOLossType::{Sigmoid, Hinge, Ipo, Kto}` (`rlhf/dpo.rs`) — the `Sigmoid` and `Hinge` and `Ipo` variants each implement distinct loss formulas; **the `Kto` variant currently computes the identical formula as `Sigmoid`**, so "KTO" here is not yet a distinct prospect-theory-based loss. `get_batch_logps` also uses a simplified whole-sequence-mean approximation rather than a per-token indexed log-probability gather (noted in the source itself).
- **Reward modeling**: `RewardModel`, `RewardModelConfig`, `RewardPrediction` (`rlhf/reward_model.rs`)
- **Human feedback / preferences**: `HumanFeedback`, `PreferencePair`, `ConstitutionalPrinciple` (`rlhf/feedback.rs`, `rlhf/mod.rs`)

### Few-Shot and Meta-Learning (`few_shot` module)
- **MAML** / **Reptile**: `MAMLTrainer`+`MAMLConfig`, `ReptileTrainer`+`ReptileConfig` (`few_shot/meta_learning.rs`)
- **In-context learning**: `InContextLearner`, `ICLExample` (`few_shot/in_context.rs`)
- **Prompt tuning**: `PromptTuner`, `SoftPrompt` (`few_shot/prompt_tuning.rs`)
- **Cross-task generalization / task adaptation**: `CrossTaskGeneralizer`, `TaskAdapter`, `TaskDescriptor` (`few_shot/cross_task.rs`, `few_shot/task_adaptation.rs`)

### Continual Learning (`continual` module)
- **EWC**: `EWCTrainer`, `EWCConfig`, `FisherInformation` (`continual/ewc.rs`)
- **Progressive Neural Networks**: `ProgressiveNetwork`, `ProgressiveConfig` (`continual/progressive_networks.rs`)
- **Replay**: `MemoryReplay`, `ExperienceBuffer` (`continual/memory_replay.rs`, `continual/replay_buffer.rs`)
- **Task-boundary detection**: `TaskBoundaryDetector`, `TaskTransition` (`continual/task_boundary.rs`)

### Curriculum Learning & Data Pipeline (`data_pipeline` module)
- **Curriculum learning**: `CurriculumLearningManager`, `CurriculumStrategy`, `PacingFunction` — length/difficulty/self-paced style progressions
- **Active learning**: `ActiveLearningManager`, `QueryStrategy`, `UncertaintyMeasure`
- **Augmentation**: image/text/audio/token augmentation strategies with adaptive scheduling
- **Multi-modal handling & validation**: `MultiModalHandler`, `DataValidator`, `StreamingDataset`

### Hyperparameter Optimization (`hyperopt` module)
- **Search strategies**: `GridSearch`, `RandomSearch`, `BayesianOptimization` (with `GPSampler`/`TPESampler`), `Hyperband`/`SuccessiveHalving`, `PopulationBasedTraining` (PBT), `BanditOptimizer`
- **Multi-objective criteria** are supported as an early-stopping/reward composition inside `hyperopt::efficiency` (`EarlyStoppingStrategy::MultiObjective`) rather than as a standalone Pareto-front optimizer
- **Experiment management**: `ExperimentManager`, A/B testing (`ABTestConfig`/`ABTestResults`), data/model lineage and provenance (`experiment_management.rs`)
- **External tracker integrations**: TensorBoard, W&B, Neptune, ClearML, MLflow trackers/configs (`framework_integration.rs`)

### Other Production-Adjacent Modules
- **Model versioning**: `ModelRegistry`, `ModelVersion`, `ModelVersioningManager` (`model_versioning.rs`)
- **Online learning**: `OnlineLearningManager`, `ConceptDrift` detection (`online_learning.rs`)
- **Cost tracking**: `CostTracker`, `Budget`, `CostForecastingModel` (`cost_tracking.rs`)
- **Neural Architecture Search**: `NASController`, `NASAlgorithm`, `SearchSpaceConfig` (`nas_integration.rs`)

## Feature Flags

```toml
[dependencies]
trustformers-training = "0.1.4"
```

- `default = []` — no backend feature is enabled by default.
- `torch` — enables `trustformers-core/torch`. Disabled by default; requires a matching `libtorch` version installed at runtime.
- `candle` — enables `trustformers-core/candle`.
- `full` — enables `candle` plus `trustformers-core/full` (used for full-feature testing; excludes `torch`).

## API Overview

Everything below is re-exported from the crate root (`trustformers_training::*`) unless a submodule path is shown; see `src/lib.rs` for the authoritative list of ~40 top-level modules.

| Area | Key types | Module |
|------|-----------|--------|
| Training loop | `Trainer`, `TrainingArguments`, `TrainerCallback` | `trainer`, `training_args` |
| Simple trainer | `SimpleTrainer`, `SimpleTrainerBuilder` | `simplified_trainer` |
| Losses / metrics | `CrossEntropyLoss`, `MSELoss`, `Accuracy`, `F1Score`, `Perplexity` | `losses`, `metrics` |
| Distributed | `DataParallelTrainer`, `DistributedBackend`, `ProcessGroup` | `distributed` |
| Parallelism | `TensorParallelism`, `SequenceParallelism`, `Parallelism3D`, `ExpertParallelism`, `RingAttentionManager` | `tensor_parallelism`, `sequence_parallelism`, `parallelism_3d`, `expert_parallelism`, `ring_attention` |
| Mixed precision / QAT | `AMPManager`, `MixedPrecisionConfig`, `QATTrainer`, `QATConfig` | `mixed_precision`, `qat` |
| RLHF | `PPOTrainer`, `DPOTrainer`, `RewardModel` | `rlhf` |
| Few-shot / meta-learning | `MAMLTrainer`, `ReptileTrainer`, `InContextLearner`, `PromptTuner` | `few_shot` |
| Continual learning | `EWCTrainer`, `ProgressiveNetwork`, `MemoryReplay` | `continual` |
| Hyperparameter search | `GridSearch`, `BayesianOptimization`, `Hyperband`, `PopulationBasedTraining` | `hyperopt` |
| Data pipeline | `DataPipeline`, `CurriculumLearningManager`, `ActiveLearningManager` | `data_pipeline` |
| Stability / monitoring | `AdvancedStabilityMonitor`, `GradientRecoveryManager`, `TrainingMonitor` | `advanced_stability_monitor`, `gradient_anomaly_recovery`, `training_monitor` |

## Quick Start

This mirrors the doc-tested example in `src/lib.rs` (a minimal stand-in `Model` is used here in place of a real architecture from `trustformers-models`):

```rust
use trustformers_training::{Trainer, TrainingArguments, MSELoss};
use trustformers_training::trainer::TaskType;
use trustformers_optim::Adam;

// Configure training (see `TrainingArguments` for the full set of options).
let args = TrainingArguments::default();

// Choose an optimizer (from trustformers-optim) and a loss function.
let optimizer = Box::new(Adam::new(1e-4, (0.9, 0.999), 1e-8, 0.0));
let loss_fn = Box::new(MSELoss::new());

// Build the trainer; `trainer.train(..)` then runs the loop over your datasets.
let trainer = Trainer::new(model, args, optimizer, loss_fn, TaskType::Classification)?;
```

### Distributed Training (simulated backend)

```rust
use trustformers_training::distributed::{
    DataParallelTrainer, DistributedBackend, DistributedConfig, SimulatedProcessGroup,
};
use std::sync::Arc;

// The simulated backend needs no real cluster, so this runs anywhere.
// Swap in `DistributedBackend::NCCL`/`Gloo`/`MPI` for the corresponding process-group
// type, keeping in mind that today those also perform in-process simulation of the
// collective operations rather than real cross-process networking.
let config = DistributedConfig {
    world_size: 1,
    rank: 0,
    backend: DistributedBackend::Simulated,
    master_addr: "localhost".to_string(),
    master_port: 29500,
    gradient_compression: false,
    bucket_size_mb: 25,
};

let process_group = Arc::new(SimulatedProcessGroup::new(0, 1));
let trainer = DataParallelTrainer::new(model, process_group, config)?;
```

### DPO (Direct Preference Optimization)

```rust
use trustformers_training::rlhf::{DPOConfig, DPOTrainer};

let dpo_config = DPOConfig {
    beta: 0.1,
    ..Default::default()
};

// `ref_model: None` uses the reference-free variant; pass `Some(ref_model)` otherwise.
let trainer = DPOTrainer::new(model, None, dpo_config);
```

## Architecture

The real module layout (from `src/lib.rs`), grouped by area — this crate has ~40 top-level modules under `src/`, so only the largest/most relevant are shown:

```
trustformers-training/
├── src/
│   ├── trainer.rs, training_args.rs, simplified_trainer.rs   # Core training loop(s)
│   ├── losses.rs, metrics.rs, gradient.rs
│   ├── distributed.rs           # DataParallelTrainer + NCCL/Gloo/MPI/Simulated ProcessGroups
│   ├── tensor_parallelism.rs, sequence_parallelism.rs
│   ├── parallelism_3d.rs        # Combined data+tensor+pipeline, GPipe/PipeDream/1F1B scheduling
│   ├── expert_parallelism.rs, ring_attention.rs, auto_parallelism.rs
│   ├── elastic_training.rs, multicloud.rs, resource_scheduling.rs
│   ├── mixed_precision.rs, qat.rs
│   ├── rlhf/                    # ppo.rs, dpo.rs, reward_model.rs, feedback.rs, config.rs, trainer.rs
│   ├── few_shot/                # meta_learning.rs (MAML/Reptile), in_context.rs, prompt_tuning.rs, ...
│   ├── continual/                # ewc.rs, progressive_networks.rs, memory_replay.rs, task_boundary.rs
│   ├── hyperopt/                 # tuner.rs, sampler.rs, strategies.rs, surrogate_models.rs, ...
│   ├── data_pipeline.rs          # Curriculum, active learning, augmentation, multi-modal, validation
│   ├── experiment_management.rs, framework_integration.rs   # A/B testing, W&B/MLflow/ClearML/Neptune/TensorBoard
│   ├── advanced_stability_monitor.rs, gradient_anomaly_recovery.rs
│   ├── adaptive_gradient_scaling.rs, adaptive_learning_rate.rs
│   ├── training_dynamics.rs, training_monitor.rs, training_orchestration.rs
│   ├── config_validation.rs, error_codes.rs, error_handling.rs
│   ├── model_versioning.rs, online_learning.rs, cost_tracking.rs, nas_integration.rs
│   └── lib.rs
```

## Testing

- **~930 tests passing** (workspace-wide: 18,102 passed / 0 failed, 0 clippy warnings, 0 rustdoc warnings — verified 2026-07-01)
- Covers the training loop, distributed abstractions, mixed precision/QAT, RLHF (PPO/DPO), few-shot/continual learning, hyperparameter search, data pipeline, and the stability/monitoring stack
- `examples/` contains illustrative programs, but at least one (`examples/basic_training/simple_classification.rs`) references types (`TrainerConfig`, `TrainingArgs`, `MetricResult`) that no longer match the current public API — see `TODO.md`

## Verification Notes

While documenting this crate we found a meaningful amount of code under `src/` that exists on disk but is **not** referenced by any `mod` declaration in `lib.rs`, and is therefore not compiled into the crate: 21 top-level directories (`dpo/`, `ppo/`, `kto/`, `lora/`, `ewc/`, `curriculum/`, `hpo/`, `orpo/`, `simpo/`, `ipo/`, `spin/`, `grpo/`, `raft/`, `reinforce/`, `distillation/`, `model_merging/`, `constitutional_ai/`, `contrastive_search/`, `token_dpo/`, `online_dpo/`, `reward_modeling/`) plus 6 top-level files (`async_checkpoint.rs`, `distributed_overlap.rs`, `losses_tests.rs`, `metrics_tests.rs`, `training_args_tests.rs`, `mod.rs`) — roughly 30,000 lines across 33 files. Their functionality generally overlaps with (and appears superseded by) modules that *are* wired in, e.g. `rlhf::ppo`/`rlhf::dpo` vs. the orphaned top-level `ppo/`/`dpo/`, or `data_pipeline`'s curriculum types vs. the orphaned top-level `curriculum/`. All statistics in this README (SLoC, public API count, test coverage) describe only the reachable, compiled 69-file tree. See `TODO.md` for details.

## License

Apache-2.0
