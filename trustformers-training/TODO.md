# trustformers-training TODO List

**Version:** 0.2.1 | **Last reviewed:** 2026-08-24 (file-split and SLoC-total items refreshed; the detailed "72 compiled files / 58,207 SLoC / 1,673 public API items" breakdown below is unchanged since 2026-07-09 and is now stale on the file count specifically — this wave split 2 of those 72 files into 15, see the file-size correction below)

## Overview

The `trustformers-training` crate provides training infrastructure for the TrustformeRS ecosystem: a
core `Trainer`/`SimpleTrainer` loop, mixed precision/QAT, RLHF (PPO/DPO), few-shot and continual learning,
hyperparameter optimization, a large data-pipeline/augmentation system, training-stability monitoring, and
a family of distributed/parallel-training abstractions (tensor/sequence/3D/expert/ring-attention parallelism
plus elastic and multi-cloud orchestration).

---

## Current Status (verified 2026-07-01)

- **~1,010 tests passing** as of 2026-07-09, not independently re-run this pass — see root `TODO.md` for the current workspace-wide baseline (21,370 passed / 41 skipped / 0 failed, 2026-08-26)
- **1,673 public API items** reachable from `lib.rs` as of 2026-07-09, not re-verified — the underlying file count (72 compiled `.rs` files) is stale: this wave split 2 of those files (`data_pipeline.rs`, `auto_parallelism.rs`) into 15 files total, so the true compiled-file count today is higher, not recounted. Whole-crate SLoC (compiled + orphaned, `tokei`, verified 2026-08-24): **83,319** — close to but not identical to the 2026-07-09 "full `src/` tree on disk is 83,317 lines" figure, consistent with real edits since (not itself evidence the orphaned-code inventory below is stale, but it wasn't re-run this pass).
- **0 stub/placeholder implementations** (`todo!()`/`unimplemented!()`/TODO/FIXME/HACK/XXX/"placeholder") in compiled code
- **0 `.unwrap()` calls** in compiled production code
- No file in the compiled tree exceeds the workspace's 2000-line refactor threshold. **Corrected 2026-08-24**: `auto_parallelism.rs` (this line's previous "largest, 1,610 lines" example) grew to 2,030 lines between 2026-07-09 and 2026-08-18 and was split this wave into a 6-file `auto_parallelism/` directory module (largest sub-file: `selector.rs` at 1,355 lines); `data_pipeline.rs` (2,143 lines by 2026-08-18) was likewise split into a 9-file `data_pipeline/` directory. Both directories' largest files stay under the limit — see root `TODO.md` for the current largest-files-in-the-workspace figures.
- Status: **Alpha** — see "Known Issues" below for why this crate is not labeled Stable despite the test count

This file replaces the previous TODO.md's checklist with one re-verified against the actual source
(`grep`/module-tree audit), rather than carrying forward unverified checkmarks.

---

## Known Issues (found during 2026-07-01 documentation pass)

These are genuine findings from auditing `src/` against `lib.rs`'s module tree — not present in earlier
drafts of this document.

- [ ] **~25,110 lines of orphaned/unwired code** (down from ~30,000 as of 2026-07-09; see the `hpo`/`mod.rs`
  note below). 20 top-level directories (`dpo/`, `ppo/`, `kto/`, `lora/`,
  `ewc/`, `curriculum/`, `orpo/`, `simpo/`, `ipo/`, `spin/`, `grpo/`, `raft/`, `reinforce/`,
  `distillation/`, `model_merging/`, `constitutional_ai/`, `contrastive_search/`, `token_dpo/`, `online_dpo/`,
  `reward_modeling/`) plus 5 top-level files (`async_checkpoint.rs`, `distributed_overlap.rs`,
  `losses_tests.rs`, `metrics_tests.rs`, `training_args_tests.rs`) are not referenced by any `mod`
  declaration in `lib.rs` and are therefore not compiled into the crate at all. Their functionality generally
  looks superseded by wired-in equivalents (`rlhf::ppo`/`rlhf::dpo` vs. the orphaned `ppo/`/`dpo/`;
  `data_pipeline`'s curriculum types vs. the orphaned `curriculum/`). Action needed: either wire the useful
  ones in (e.g. `grpo/`, which looks like it might add real GRPO support not otherwise present) or delete the
  rest so the tree reflects what's actually shipped. (`hpo/` was mounted via `pub mod hpo;` in 0.2.0 and is
  no longer on this orphaned list — see the Hyperparameter Tuning section below; the orphaned root-level
  `mod.rs` was deleted in 0.2.0 rather than wired in — see Housekeeping below.)
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
- [~] Implement real, distinct KTO loss (Kahneman-Tversky Optimization) (planned 2026-07-05)
  - Goal: DPOLossType::Kto computes the real KTO loss instead of being byte-for-byte identical to Sigmoid DPO.
  - Prerequisites: the existing paired-batch data model (DPOExample/DPOBatch/PreferencePair) has no per-example desirable/undesirable label the real KTO formula needs — this must be built. Major mitigating factor: a complete, mathematically-correct, already-tested (29 tests) scalar reference implementation already exists, orphaned, at src/kto/mod.rs — the work is adapting its math to a batched Tensor path, not deriving KTO from the paper from scratch.
  - Design: add KtoBatch/KtoExample tensor types + a KtoDataCollator (single-sided, allows unbalanced group sizes unlike paired DPO) mirroring DPOBatch/DPODataCollator's shape. Implement the vectorized loss using existing Tensor primitives (no new Tensor infrastructure needed). Add a bootstrap converter PreferencePair -> [KtoExample; 2] so existing paired data can feed KTO immediately. Redirect DPOLossType::Kto's match arm to the new path AND add a dedicated KtoTrainer.
  - Files: src/rlhf/dpo.rs, src/kto/mod.rs, new src/rlhf/kto.rs, lib.rs, a converter fn near rlhf/feedback.rs.
  - Tests: port kto/mod.rs's 29 scalar tests as a parity oracle; a new collator test; an unbalanced-group-size test; a regression test proving DPOLossType::Kto's output now differs from Sigmoid on an asymmetric-lambda batch.
  - Risk: note for the record — DPOTrainer::get_batch_logps is itself a whole-tensor-mean approximation (real per-token gather isn't wired up yet), so the new KTO path inherits that same approximation until separately fixed; also there's a third, unrelated "DPO" implementation (RLHFTrainer::compute_dpo_update) elsewhere in the crate — don't confuse it with either DPO or the new KTO path.
- [ ] **`DPOTrainer::get_batch_logps` uses a simplified approximation**: per its own source comment, it
  currently averages log-probabilities over the whole tensor rather than gathering per-token log-probs
  indexed by `labels`, because tensor indexing for this case isn't wired up yet.
- [~] Fix stale API in examples/basic_training/simple_classification.rs (planned 2026-07-05)
  - Goal: the example compiles and runs against the current API (currently comprehensively stale: wrong Trainer::new arity, wrong Loss/Model/TrainerCallback trait shapes throughout).
  - Design: full rewrite against the current real API, using the crate's own passing lib.rs doctest as the template.
  - Files: trustformers-training/examples/basic_training/simple_classification.rs only.
  - Tests: add cargo check --examples -p trustformers-training so this can't silently regress again.
  - Risk: low, bounded to one file. (Note for the record, out of scope here: the identical staleness pattern recurs in 3 other example files.)
- [~] Delete 5 stray *.rs.prelude_fix backup files (planned 2026-07-05)
  - Goal/Design: delete src/gradient_anomaly_recovery.rs.prelude_fix, src/continual/memory_replay.rs.prelude_fix, src/hyperopt/auto_tuner.rs.prelude_fix, src/hyperopt/sampler.rs.prelude_fix, src/hyperopt/search_space.rs.prelude_fix — confirmed zero references, dead pre-migration snapshots.
  - Files: the 5 files.
  - Tests: cargo check -p trustformers-training --all-features (no-op diff).
  - Risk: none.

---

## 0.2.0 Release Scope (added 2026-07-06)

Two workspace-wide tracks are in scope for 0.2.0: the **OxiCUDA GPU migration** (scirs2-core `gpu` →
OxiCUDA 0.4.x, driven from trustformers-core — no `trustformers-training` code is gated on GPU features,
so no work lands in this crate for that track) and the **PyTorch (tch) dependency removal**. Decision for
the latter: delete the `tch` dependency and the `torch` feature entirely in 0.2.0 (workspace
`Cargo.toml:82`, the trustformers-core `torch` feature plus ~40 lines of cfg arms, and the forwarder
features in `trustformers`, `trustformers-training`, `trustformers-c`); do **not** adopt ToRSh as a
replacement now — a P2 task records evaluating an optional `torsh-interop` feature in 0.3.x once
torsh 0.2.0 ships on crates.io. Sub-decision on candle: drop the unused `candle-nn` workspace dep now,
keep the `candle` feature/variant through 0.2.0 (it is in every `full` set), and decide
implement-vs-remove in 0.3.x. Rationale (verified): `Tensor::Torch`
(`trustformers-core/src/tensor/mod.rs:220-221`) is never constructed anywhere in the workspace, and the
cfg(torch) "PyTorch validation" is simulated with hardcoded results
(`trustformers-core/src/testing/cross_framework.rs:309-325`), so zero functionality is lost — while
keeping `tch` costs a multi-GB libtorch download plus policy violations (torch-sys pulls `cc` (C++), the
OxiARC-banned `zip 0.6.6`, `ureq`, and duplicate old ndarray/rand/safetensors pins,
`Cargo.lock:10598-10610`). All real PyTorch interop is already pure Rust and is kept (safetensors
non-optional at `trustformers-core/Cargo.toml:34`; `checkpoint/formats.rs`, `utils/weight_loading.rs`,
trustformers-optim `pytorch_compat.rs`).

### PyTorch (tch) dependency removal — this crate's tasks

- [x] **[P0] Remove the `torch` forwarder feature.** DONE 2026-07-06: the `torch` forwarder feature was
  deleted from `trustformers-training/Cargo.toml` and the `full` feature comment updated to drop the torch
  mention; `trustformers-training/README.md` had its `torch` feature-flag documentation line removed and
  the `full` line's parenthetical updated accordingly. Landed atomically with the workspace-wide
  `tch`/`torch` removal (root `Cargo.toml`, `trustformers-core`, `trustformers`, `trustformers-c`).
  Verified green in the session's convergence pass: `cargo check --workspace --all-features` and
  `cargo clippy --workspace --all-features --all-targets` both pass with zero warnings; full workspace
  nextest run (12033/12033 passed, 113 skipped GPU-availability-gated) confirms no regression.
  Evidence: `trustformers-training/Cargo.toml`, `trustformers-training/README.md`.

### Post-0.2.0 (0.3.x)

- [ ] **[P2] Track the workspace-level ToRSh evaluation.** Once torsh 0.2.0 ships on crates.io (its
  crates.io 0.1.3 pins scirs2 0.5.1, type-incompatible with this workspace's scirs2 0.6.0 stack), the
  workspace will evaluate an optional `torsh-interop` feature; if adopted, this crate may regain a thin
  forwarder feature. No action here until that decision lands (~/work/torsh).

### OxiCUDA GPU migration

- No tasks in this crate: GPU backends live in trustformers-core (`cuda`/`metal` features, oxicuda
  0.4.x — see `trustformers-core/Cargo.toml:88-131`, `trustformers-core/src/gpu_ops/cuda.rs`).
  `trustformers-training` has no GPU-feature-gated code.

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
  `HardwareConstraints`/`ModelConstraints`/`NetworkTopology` (`auto_parallelism/`, split from a single `auto_parallelism.rs` this wave) — note this is strategy
  *selection*, not general hyperparameter tuning from hardware
- [x] Elastic training coordinator: worker heartbeats, scaling decisions, mid-training checkpoints
  (`elastic_training.rs`) — **honesty-audited 2026-08-25.** Before this pass, `scale_up`/
  `rebalance_workers`/`recover_from_checkpoint` were log-and-`Ok(())` no-ops that still recorded
  `ScalingEvent { success: true }`, `create_checkpoint` hardcoded `step: 0` instead of the caller's real
  step, and `ResourceMonitor`/`FaultDetector`/`LoadBalancer` were one-bool no-op lifecycle types nothing
  ever read. Fixed honestly rather than left fabricated: added a `WorkerProvisioner` trait (the
  cluster-provisioning callback this workspace has no real substrate for); `scale_up`/`rebalance_workers`/
  `recover_from_checkpoint` now return a structured `ElasticTrainingError::NoProvisioner` (naming exactly
  what's missing) unless one is attached via `with_provisioner`, and `scale_up` also errors
  (`PartialProvisioning`) if the provisioner starts fewer workers than requested rather than reporting the
  full target reached; `execute_scaling` records `ScalingEvent::success` from the real operation result,
  never unconditionally; `create_checkpoint` takes the caller's real `step: usize`; the three no-op
  lifecycle types are deleted (dead weight, zero callers, converting them to "real" would still need the
  cluster substrate that doesn't exist). `scale_down`'s local worker deregistration and the scaling
  *decision* heuristics (`evaluate_scaling_decision`/`should_rebalance`) were already real, local bookkeeping
  and are unchanged in kind, only in ordering (`scale_down` now confirms real termination via the
  provisioner, when one is attached, before forgetting a worker locally — previously it could have forgotten
  a worker whose real process termination failed). `WorkerPerformanceMetrics` gained a `workload` field so
  `update_heartbeat` actually feeds `should_rebalance`'s imbalance check, which used to be structurally
  always-false (workload was set to 0.0 at registration and never updated again). 11 new tests cover the
  honest contract (`cargo nextest run -p trustformers-training elastic_training`).
- [x] Multi-cloud orchestration: `MultiCloudOrchestrator`, `CloudScheduler`, cost-aware scheduling
  (`multicloud.rs`)
- [x] Resource scheduling: `ResourceScheduler`, `ResourcePool` (`resource_scheduling.rs`)
- [~] Fault tolerance / checkpoint-on-preemption for spot instances: `multicloud.rs`/`cost_tracking.rs`/
  `resource_scheduling.rs` model spot-instance and preemption concepts at the config/cost level, but an
  end-to-end "detect preemption signal → auto-checkpoint" pipeline is not confirmed
- [~] Worker-failure recovery without a full restart: `elastic_training::ElasticTrainingCoordinator` has
  real worker-monitoring/scaling-decision/checkpoint logic. As of the 2026-08-25 honesty pass,
  `recover_from_checkpoint` honestly requires a caller-supplied `WorkerProvisioner` (it returns
  `ElasticTrainingError::NoProvisioner` without one, instead of the previous log-and-`Ok(())` that reported
  a worker "recovered" with no state restored); `handle_worker_failure` still deregisters a confirmed-dead
  worker locally even when recovery isn't possible, so the failure path itself cannot get stuck, but true
  zero-downtime replacement is only as real as whatever `WorkerProvisioner` a caller supplies — this crate
  ships no such implementation itself (there is no cluster substrate in this workspace)

### Mixed Precision & Quantization
- [x] AMP: `AMPManager`, `MixedPrecisionConfig`, `LossScaler`, `DynamicBatchingManager` (`mixed_precision.rs`)
- [x] Quantization-Aware Training: `QATTrainer`, per-tensor/per-channel schemes, `MixedBitQATTrainer`,
  `fake_quantize`/`fake_quantize_mixed_bit` (`qat.rs`)

### RLHF and Alignment (`rlhf` module)
- [x] PPO: `PPOTrainer`, `PPOConfig`, `PPOStepResult`, `PolicyModel`, `ValueModel` (`rlhf/ppo.rs`)
- [x] DPO: `DPOTrainer`, `DPOConfig`, distinct `Sigmoid`/`Hinge`/`Ipo` loss formulas (`rlhf/dpo.rs`)
- [~] KTO as a *distinct* prospect-theory loss — currently aliases the Sigmoid DPO formula; real KTO
  implementation planned 2026-07-05 (see "Known Issues" above for the full plan)
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
  lives in `data_pipeline/` (a directory module since this wave's file split; previously `data_pipeline.rs`), **not** the orphaned top-level `curriculum/` directory
- [x] Active learning: `ActiveLearningManager`, `QueryStrategy` (`data_pipeline/`)
- [x] Augmentation (image/text/audio/token) with adaptive scheduling (`data_pipeline/`)
- [x] Multi-modal handling and data validation (`data_pipeline/`)

### Hyperparameter Tuning (`hyperopt` module)
- [x] Grid search, random search (`GridSearch`, `RandomSearch`)
- [x] Bayesian optimization with GP and TPE samplers (`BayesianOptimization`, `GPSampler`, `TPESampler`)
- [x] Hyperband / successive halving (`Hyperband`, `SuccessiveHalving`)
- [x] Population-Based Training (`PopulationBasedTraining`, `PBTConfig`)
- [x] Bandit-based optimization (`BanditOptimizer`)
- [x] Wire in hpo::multi_objective (planned 2026-07-05) — **DONE (2026-07-09):** `pub mod hpo;` added to
  `lib.rs` (line 151), re-exporting `MultiObjectiveHpo`/`ParetoFront`/`compute_pareto_front`/
  `hypervolume_indicator`/`non_domination_sort` (from `hpo::multi_objective`) and `AutoLrSelector`/
  `LrRangeTest` (from `hpo::auto_lr`). The module's 83 pre-written tests (`auto_lr.rs`: 21, `mod.rs`: 32,
  `multi_objective.rs`: 30) now compile and run for the first time. The module's one doctest (in
  `multi_objective.rs`'s doc comment) already used `?`/`Ok::<(), Box<dyn std::error::Error>>(())` rather
  than `.unwrap()`, so no additional no-unwrap fix was needed.
  - Goal: expose the already-complete (1332 lines) NSGA-II Pareto-front hyperparameter-search engine.
  - Design: add `pub mod hpo;` to lib.rs next to the existing `pub mod hyperopt;` — zero missing dependencies, zero name collisions confirmed against the ~90 names hyperopt already exports.
  - Files: trustformers-training/src/lib.rs only.
  - Tests: cargo nextest run -p trustformers-training (its ~30 pre-written tests run for the first time); fix the module doctest's .unwrap() to comply with no-unwrap policy while touching this.
  - Risk: none beyond the doctest unwrap fix.

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
  (`cost_tracking.rs`) — **honesty-audited 2026-08-25.** `EfficiencyMetrics::resource_utilization` and
  `idle_cost_percentage` were previously hardcoded to `0.75`/`15.0`, pinning `efficiency_score` at exactly
  `0.60` and making the resource-rightsizing recommendation (gated on `resource_utilization < 0.6`)
  unreachable. `resource_utilization` is now real: the fraction of the report's time range covered by
  billed entry durations (clamped to `[0, 1]`) — a genuine *temporal* utilization signal computed from data
  the tracker already records, not per-machine hardware utilization (CPU/GPU busy %), which this tracker
  has no way to observe. `idle_cost_percentage` is now `Option<f32>` and stays `None`: this tracker has no
  signal distinguishing busy-vs-idle time *within* a billed entry, so it is left honestly absent rather than
  invented; `efficiency_score` is now `Option<f64>`, `Some` only when `idle_cost_percentage` is `Some`. The
  idle-resource-elimination recommendation is now `Some`-gated and will not fire until a real idle signal
  exists. 4 new tests cover the honest contract, including one proving the previously-unreachable
  rightsizing branch now fires (`cargo nextest run -p trustformers-training cost_tracking`).
- [x] Neural Architecture Search: `NASController`, `NASAlgorithm`, `SearchSpaceConfig`
  (`nas_integration.rs`)

---

## Future Enhancements

### High Priority
- [ ] Decide the fate of the ~25,110 lines of orphaned modules: wire in (`grpo/` in particular looks like
  it could add real GRPO support) or delete
- [ ] Implement a real ZeRO optimizer (stage 1 at minimum) if distributed memory sharding is still a goal
- [ ] Give `NCCLProcessGroup`/`GlooProcessGroup`/`MPIProcessGroup` real backend bindings, or rename/document
  them clearly as simulations until they do
- [~] Fix `examples/basic_training/simple_classification.rs` to use the current `Trainer`/`TrainingArguments`
  API (and consider adding a `cargo check --examples` step to CI so this doesn't recur silently) — planned
  2026-07-05 (see "Known Issues" above for the full plan)
- [~] Implement a distinct KTO loss (prospect-theory utility, asymmetric loss aversion) rather than aliasing
  Sigmoid DPO — planned 2026-07-05 (see "Known Issues" above for the full plan)
- [ ] Wire up proper per-token indexed log-probability gathering in `DPOTrainer::get_batch_logps`
- [ ] Recent PEFT/alignment techniques not yet present anywhere in the tree: DoRA, GaLore, AdaLoRA (the
  orphaned `lora/` directory has *some* LoRA-family code, but it is unwired and none of these newer variants
  were found in it)

### Performance
- [~] Wire real gradient_compression (recharacterized from "benchmark" to "implement") (planned 2026-07-05)
  - Goal: the gradient_compression: bool flag on DistributedConfig actually changes runtime behavior (confirmed 100% inert today — the only reference anywhere is a test asserting the field round-trips its own assigned value; synchronize_gradients never reads it).
  - Design: thread the flag (or a CompressionType choice) through DataParallelTrainer::synchronize_gradients before its all_reduce call, delegating actual compress/decompress to trustformers-optim's existing, real CompressionType machinery (already a normal workspace dependency) — wiring to existing real logic, not inventing a new algorithm.
  - Files: trustformers-training/src/distributed.rs.
  - Tests: a test that the flag now actually changes behavior (not just a round-trip test); a correctness test that compressed+decompressed all-reduce stays within tolerance of the uncompressed sum.
  - Risk: low — the compression logic is borrowed, tested, real code; the only new work is the plumbing and the behavioral test.
- [~] Add benches/ directory with criterion benchmarks (planned 2026-07-05)
  - Goal: cargo bench -p trustformers-training works.
  - Design: criterion.workspace = true dev-dependency + [[bench]] entries, mirroring trustformers-optim's existing bench setup. 3 targets: training-loop micro-step, loss functions, mixed-precision loss scaling.
  - Files: new trustformers-training/benches/*.rs, Cargo.toml.
  - Tests: cargo bench --no-run to confirm compilation, then a real cargo bench run.
  - Risk: known pitfall — this workspace's own root TODO.md documents a prior criterion_main!-nested-inside-mod-benches compile failure (E0601) in trustformers-core's bench; keep criterion_main! at file top level.

### Housekeeping
- [~] Delete the 5 stray `*.rs.prelude_fix` backup files — planned 2026-07-05 (see "Known Issues" above for
  the full plan)
- [x] Delete dead src/mod.rs (planned 2026-07-05) — **DONE (2026-07-09):** `trustformers-training/src/mod.rs`
  is deleted (confirmed absent from the tree); `lib.rs` remains the sole crate root.
  - Goal/Design: delete the legacy src/mod.rs — crate root is lib.rs; everything mod.rs declares already exists, more completely, in lib.rs.
  - Files: trustformers-training/src/mod.rs.
  - Tests: cargo check --all-features (no-op diff).
  - Risk: none.

---

## Development Guidelines

### Code Standards
- **Use trustformers-core abstractions only**
- **File size limit:** <2000 lines per file (currently satisfied — verified 2026-08-24 via a full-workspace `wc -l` sweep; the "1,610 lines" figure this line previously cited for the largest file is stale, see the Current Status section above for what happened to it)
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

**Last Updated:** 2026-07-09 — version bumped to 0.2.1; `hpo` module wiring and `src/mod.rs` deletion
confirmed done and checked off; orphaned-code inventory and file/SLoC/test counts refreshed via `tokei`
and source inspection

**2026-08-25 addendum (production-hardening honesty pass, `elastic_training.rs` + `cost_tracking.rs`
only):** see the updated bullets above for `Elastic training coordinator` and `Cost tracking / budgeting /
forecasting`. Baselines before and after this pass: `cargo check -p trustformers-training --all-targets`
and `cargo clippy -p trustformers-training --all-targets -- -D warnings` both `EXIT=0` throughout;
`cargo nextest run -p trustformers-training --no-fail-fast` went from 2100 passed / 2 skipped / 0 failed to
2115 passed / 2 skipped / 0 failed (+15 new tests, 0 regressions). Nothing else in this crate was in scope
for this pass and nothing else was touched.
**Version:** 0.2.1
**Status:** Alpha — ~1,010 tests passing, 1,673 reachable public API items, 0 stubs, but see "Known Issues"
for the distributed-training and orphaned-module caveats that keep this crate from being labeled Stable.
