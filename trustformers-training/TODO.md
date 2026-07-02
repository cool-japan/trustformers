# trustformers-training TODO List

**Version:** 0.1.4 | **Last reviewed:** 2026-07-02

## Overview

The `trustformers-training` crate provides training infrastructure for the TrustformeRS ecosystem: a
core `Trainer`/`SimpleTrainer` loop, mixed precision/QAT, RLHF (PPO/DPO), few-shot and continual learning,
hyperparameter optimization, a large data-pipeline/augmentation system, training-stability monitoring, and
a family of distributed/parallel-training abstractions (tensor/sequence/3D/expert/ring-attention parallelism
plus elastic and multi-cloud orchestration).

---

## Current Status (verified 2026-07-01)

- **~930 tests passing** (workspace-wide: 18,102 passed / 0 failed, 0 clippy warnings, 0 rustdoc warnings)
- **1,673 public API items** reachable from `lib.rs` (69 compiled `.rs` files, 59,720 SLoC; the full `src/` tree on disk is 89,914 lines across 102 files, ~30,194 of which are orphaned/unwired)
- **0 stub/placeholder implementations** (`todo!()`/`unimplemented!()`/TODO/FIXME/HACK/XXX/"placeholder") in compiled code
- **0 `.unwrap()` calls** in compiled production code
- No file in the compiled tree exceeds the workspace's 2000-line refactor threshold (largest: `auto_parallelism.rs` at 1,610 lines)
- Status: **Alpha** — see "Known Issues" below for why this crate is not labeled Stable despite the test count

This file replaces the previous TODO.md's checklist with one re-verified against the actual source
(`grep`/module-tree audit), rather than carrying forward unverified checkmarks.

---

## Known Issues (found during 2026-07-01 documentation pass)

These are genuine findings from auditing `src/` against `lib.rs`'s module tree — not present in earlier
drafts of this document.

- [ ] **~30,000 lines of orphaned/unwired code.** 21 top-level directories (`dpo/`, `ppo/`, `kto/`, `lora/`,
  `ewc/`, `curriculum/`, `hpo/`, `orpo/`, `simpo/`, `ipo/`, `spin/`, `grpo/`, `raft/`, `reinforce/`,
  `distillation/`, `model_merging/`, `constitutional_ai/`, `contrastive_search/`, `token_dpo/`, `online_dpo/`,
  `reward_modeling/`) plus 6 top-level files (`async_checkpoint.rs`, `distributed_overlap.rs`,
  `losses_tests.rs`, `metrics_tests.rs`, `training_args_tests.rs`, `mod.rs`) are not referenced by any `mod`
  declaration in `lib.rs` and are therefore not compiled into the crate at all. Their functionality generally
  looks superseded by wired-in equivalents (`rlhf::ppo`/`rlhf::dpo` vs. the orphaned `ppo/`/`dpo/`;
  `data_pipeline`'s curriculum types vs. the orphaned `curriculum/`). Action needed: either wire the useful
  ones in (e.g. `grpo/`, which looks like it might add real GRPO support not otherwise present) or delete the
  rest so the tree reflects what's actually shipped.
- [ ] **No ZeRO optimizer.** Earlier documentation for this crate advertised "ZeRO stages 1/2/3" as a
  flagship feature; there is no `ZeroStage`/sharded-optimizer implementation anywhere in the source. If ZeRO
  is wanted, it needs to be implemented from scratch.
- [ ] **`NCCL`/`Gloo`/`MPI` process groups are in-process simulations, not real backends.** In
  `src/distributed.rs`, `NCCLProcessGroup`/`GlooProcessGroup`/`MPIProcessGroup::{all_reduce, broadcast,
  reduce, barrier}` all scale/mutate tensors locally (e.g. `tensor.scalar_mul(1.0)`) with source comments
  reading "In a real implementation, this would call `ncclAllReduce`/`MPI_Allreduce`/...". There is no real
  networking (no `TcpStream`/sockets) and no FFI dependency on libnccl/OpenMPI/gloo in `Cargo.toml`. Only
  `SimulatedProcessGroup` is honestly named; the other three should either get real bindings or be renamed/
  documented as simulations to avoid misleading users planning real multi-node training.
- [ ] **`DPOLossType::Kto` is currently identical to `DPOLossType::Sigmoid`** (`rlhf/dpo.rs`): both compute
  `-log(sigmoid(logits))`. This is not yet a distinct prospect-theory-based KTO loss.
- [ ] **`DPOTrainer::get_batch_logps` uses a simplified approximation**: per its own source comment, it
  currently averages log-probabilities over the whole tensor rather than gathering per-token log-probs
  indexed by `labels`, because tensor indexing for this case isn't wired up yet.
- [ ] **`examples/basic_training/simple_classification.rs` references a stale API**: it imports
  `trainer::TrainerConfig`, `training_args::TrainingArgs`, and `metrics::MetricResult`, none of which exist
  in the current crate (the real names are `Trainer`, `TrainingArguments`, `MetricCollection`/`Metric`).
  The example likely predates a `Trainer`/`TrainingArguments` API rename and needs updating; it is not
  covered by `cargo nextest` since examples aren't compiled by a plain `--all-features` test run.
- [ ] **5 stray `*.rs.prelude_fix` backup files** (`gradient_anomaly_recovery.rs.prelude_fix`,
  `hyperopt/auto_tuner.rs.prelude_fix`, `hyperopt/search_space.rs.prelude_fix`,
  `continual/memory_replay.rs.prelude_fix`, `hyperopt/sampler.rs.prelude_fix`) are leftover, non-compiled
  debris sitting next to their real counterparts. Low priority, but worth deleting.

---

## Completed Features (verified against source)

### Core Training Infrastructure
- [x] `Trainer` main training loop, `TrainerCallback` trait, `EarlyStoppingCallback` (`trainer.rs`)
- [x] `SimpleTrainer`/`SimpleTrainerBuilder` alternative builder-style API with `CheckpointCallback`,
  `LoggingCallback`, `MetricsCallback`, `ProgressCallback` (`simplified_trainer.rs`)
- [x] `TrainingArguments` with `EvaluationStrategy`/`SaveStrategy` (`training_args.rs`)
- [x] Losses: `CrossEntropyLoss` (incl. `with_label_smoothing()`), `MSELoss` (`losses.rs`)
- [x] Metrics: `Accuracy`, `F1Score`, `Perplexity`, `MetricCollection` (`metrics.rs`)
- [x] Config validation framework: `ConfigValidator`/`ConfigSchema`/`Constraint` (`config_validation.rs`)
- [x] Structured error handling with recovery suggestions: `ErrorManager`, `ErrorCodeRegistry`
  (`error_codes.rs`, `error_handling.rs`)
- [x] Training orchestration / job scheduling: `TrainingOrchestrator`, `JobScheduler`, `TrainingJob`,
  `CheckpointConfig`/`CheckpointInfo` (`training_orchestration.rs`)

### Distributed & Parallel Training
- [x] Data-parallel training abstraction: `DataParallelTrainer`, `ProcessGroup` trait (`distributed.rs`)
- [ ] ZeRO optimizer (stages 1/2/3) — **not implemented** (see Known Issues)
- [~] NCCL/Gloo/MPI backends — process-group types and a `DistributedBackend` selector exist, but the
  collective operations are in-process simulations, not real network/FFI implementations (see Known Issues)
- [x] Tensor parallelism: `TensorParallelism`, `TensorPartitioningStrategy` (`tensor_parallelism.rs`)
- [x] Sequence parallelism: `SequenceParallelism`, `SequenceSplittingStrategy` (`sequence_parallelism.rs`)
- [x] 3D parallelism with pipeline scheduling variants `GPipe`/`PipeDream`/`PipeDream2BW`/
  `Interleaved1F1B`/`Adaptive` (`parallelism_3d.rs`)
- [x] Expert parallelism (MoE-style routing): `ExpertParallelism`, `TokenRouting` (`expert_parallelism.rs`)
- [x] Ring attention for long-sequence distributed attention (`ring_attention.rs`)
- [x] Hardware-aware automatic parallelism strategy selection: `AutoParallelismSelector` from
  `HardwareConstraints`/`ModelConstraints`/`NetworkTopology` (`auto_parallelism.rs`) — note this is strategy
  *selection*, not general hyperparameter tuning from hardware
- [x] Elastic training coordinator: worker heartbeats, scaling decisions, mid-training checkpoints
  (`elastic_training.rs`)
- [x] Multi-cloud orchestration: `MultiCloudOrchestrator`, `CloudScheduler`, cost-aware scheduling
  (`multicloud.rs`)
- [x] Resource scheduling: `ResourceScheduler`, `ResourcePool` (`resource_scheduling.rs`)
- [~] Fault tolerance / checkpoint-on-preemption for spot instances: `multicloud.rs`/`cost_tracking.rs`/
  `resource_scheduling.rs` model spot-instance and preemption concepts at the config/cost level, but an
  end-to-end "detect preemption signal → auto-checkpoint" pipeline is not confirmed
- [~] Worker-failure recovery without a full restart: `elastic_training::ElasticTrainingCoordinator` has
  real worker-monitoring/scaling-decision/checkpoint logic, but true zero-downtime replacement of a failed
  worker is not independently verified

### Mixed Precision & Quantization
- [x] AMP: `AMPManager`, `MixedPrecisionConfig`, `LossScaler`, `DynamicBatchingManager` (`mixed_precision.rs`)
- [x] Quantization-Aware Training: `QATTrainer`, per-tensor/per-channel schemes, `MixedBitQATTrainer`,
  `fake_quantize`/`fake_quantize_mixed_bit` (`qat.rs`)

### RLHF and Alignment (`rlhf` module)
- [x] PPO: `PPOTrainer`, `PPOConfig`, `PPOStepResult`, `PolicyModel`, `ValueModel` (`rlhf/ppo.rs`)
- [x] DPO: `DPOTrainer`, `DPOConfig`, distinct `Sigmoid`/`Hinge`/`Ipo` loss formulas (`rlhf/dpo.rs`)
- [ ] KTO as a *distinct* prospect-theory loss — currently aliases the Sigmoid DPO formula (see Known Issues)
- [~] Faithful per-token log-probability computation in `get_batch_logps` — currently a simplified
  whole-sequence-mean approximation (see Known Issues)
- [x] Reward modeling: `RewardModel`, `RewardModelConfig`, `RewardPrediction` (`rlhf/reward_model.rs`)
- [x] Human feedback / preference data: `HumanFeedback`, `PreferencePair`, `ConstitutionalPrinciple`
  (`rlhf/feedback.rs`, `rlhf/mod.rs`)

### Few-Shot and Meta-Learning (`few_shot` module)
- [x] MAML and Reptile: `MAMLTrainer`/`MAMLConfig`, `ReptileTrainer`/`ReptileConfig` (`few_shot/meta_learning.rs`)
- [x] In-context learning: `InContextLearner`, `ICLExample` (`few_shot/in_context.rs`)
- [x] Prompt tuning: `PromptTuner`, `SoftPrompt` (`few_shot/prompt_tuning.rs`)
- [x] Cross-task generalization / task adaptation (`few_shot/cross_task.rs`, `few_shot/task_adaptation.rs`)

### Continual Learning (`continual` module)
- [x] EWC: `EWCTrainer`, `EWCConfig`, `FisherInformation` (`continual/ewc.rs`)
- [x] Progressive Neural Networks (`continual/progressive_networks.rs`)
- [x] Replay buffers / memory replay (`continual/memory_replay.rs`, `continual/replay_buffer.rs`)
- [x] Task-boundary detection (`continual/task_boundary.rs`)

### Curriculum Learning & Data Pipeline
- [x] Curriculum learning (length/difficulty/self-paced): `CurriculumLearningManager`, `PacingFunction` —
  lives in `data_pipeline.rs`, **not** the orphaned top-level `curriculum/` directory
- [x] Active learning: `ActiveLearningManager`, `QueryStrategy` (`data_pipeline.rs`)
- [x] Augmentation (image/text/audio/token) with adaptive scheduling (`data_pipeline.rs`)
- [x] Multi-modal handling and data validation (`data_pipeline.rs`)

### Hyperparameter Tuning (`hyperopt` module)
- [x] Grid search, random search (`GridSearch`, `RandomSearch`)
- [x] Bayesian optimization with GP and TPE samplers (`BayesianOptimization`, `GPSampler`, `TPESampler`)
- [x] Hyperband / successive halving (`Hyperband`, `SuccessiveHalving`)
- [x] Population-Based Training (`PopulationBasedTraining`, `PBTConfig`)
- [x] Bandit-based optimization (`BanditOptimizer`)
- [~] Multi-objective optimization — supported as an early-stopping/reward-composition criterion
  (`hyperopt::efficiency::EarlyStoppingStrategy::MultiObjective`), not as a standalone Pareto-front
  optimizer; the more fully-featured `hpo::multi_objective` on disk is orphaned/unwired (see Known Issues)

### Experiment Management & Tracking
- [x] Native experiment tracking, A/B testing, data/model lineage and provenance
  (`experiment_management.rs`)
- [x] External tracker integrations: TensorBoard, Weights & Biases, Neptune.ai, ClearML, MLflow
  (`framework_integration.rs`)

### Training Stability & Monitoring
- [x] `AdvancedStabilityMonitor`: loss-landscape analysis, anomaly prediction, risk scoring
  (`advanced_stability_monitor.rs`)
- [x] `GradientRecoveryManager`: gradient-anomaly detection and recovery strategies
  (`gradient_anomaly_recovery.rs`)
- [x] `AdaptiveGradientScaler` / `AdaptiveLearningRateScheduler` (`adaptive_gradient_scaling.rs`,
  `adaptive_learning_rate.rs`)
- [x] `TrainingDynamicsAnalyzer` / `TrainingMonitor`: convergence, gradient flow, weight evolution, health
  status (`training_dynamics.rs`, `training_monitor.rs`)

### Other Production-Adjacent Modules
- [x] Model registry/versioning: `ModelRegistry`, `ModelVersion` (`model_versioning.rs`)
- [x] Online learning with concept-drift detection (`online_learning.rs`)
- [x] Cost tracking / budgeting / forecasting: `CostTracker`, `Budget`, `CostForecastingModel`
  (`cost_tracking.rs`)
- [x] Neural Architecture Search: `NASController`, `NASAlgorithm`, `SearchSpaceConfig`
  (`nas_integration.rs`)

---

## Future Enhancements

### High Priority
- [ ] Decide the fate of the ~30,000 lines of orphaned modules: wire in (`grpo/` in particular looks like
  it could add real GRPO support) or delete
- [ ] Implement a real ZeRO optimizer (stage 1 at minimum) if distributed memory sharding is still a goal
- [ ] Give `NCCLProcessGroup`/`GlooProcessGroup`/`MPIProcessGroup` real backend bindings, or rename/document
  them clearly as simulations until they do
- [ ] Fix `examples/basic_training/simple_classification.rs` to use the current `Trainer`/`TrainingArguments`
  API (and consider adding a `cargo check --examples` step to CI so this doesn't recur silently)
- [ ] Implement a distinct KTO loss (prospect-theory utility, asymmetric loss aversion) rather than aliasing
  Sigmoid DPO
- [ ] Wire up proper per-token indexed log-probability gathering in `DPOTrainer::get_batch_logps`
- [ ] Recent PEFT/alignment techniques not yet present anywhere in the tree: DoRA, GaLore, AdaLoRA (the
  orphaned `lora/` directory has *some* LoRA-family code, but it is unwired and none of these newer variants
  were found in it)

### Performance
- [ ] Verify/benchmark the `gradient_compression` flag on `DistributedConfig` — the field exists but its
  runtime effect wasn't independently verified during this pass
- [ ] Add a `benches/` directory with `criterion` benchmarks; none currently exist for this crate (the
  previous README's benchmark table had no backing harness and has been removed)

### Housekeeping
- [ ] Delete the 5 stray `*.rs.prelude_fix` backup files
- [ ] Delete or wire in the legacy `src/mod.rs` (currently dead; a minimal alternate `lib.rs`-shaped file)

---

## Development Guidelines

### Code Standards
- **Use trustformers-core abstractions only**
- **File size limit:** <2000 lines per file (currently satisfied — largest reachable file is 1,610 lines)
- **Error handling:** Use `Result<T, TrustformersError>` / the crate's own `TrainingError`/`TrainingResult`
- **Testing:** Integration tests for distributed training
- **Naming:** snake_case for all identifiers
- **No `unwrap()` in production code** (currently satisfied in the compiled tree)

### Build & Test Commands

```bash
# Run all tests
cargo nextest run -p trustformers-training --all-features

# Check compilation
cargo check -p trustformers-training --all-features
```

---

**Last Updated:** 2026-07-02 — v0.1.4
**Version:** 0.1.4
**Status:** Alpha — ~930 tests passing, 1,673 reachable public API items, 0 stubs, but see "Known Issues"
for the distributed-training and orphaned-module caveats that keep this crate from being labeled Stable.
