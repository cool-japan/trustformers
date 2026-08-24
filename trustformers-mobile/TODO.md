# trustformers-mobile TODO List

## Overview

The `trustformers-mobile` crate provides mobile deployment infrastructure for iOS and Android, enabling on-device inference and training with platform-specific hardware acceleration. It includes framework integrations for React Native, Flutter, and Unity (of varying maturity — see Current Status).

**Key Responsibilities:**
- iOS deployment (Swift package, Core ML, Metal)
- Android deployment (Kotlin/Java, NNAPI, Vulkan)
- Hardware acceleration (Neural Engine, Edge TPU, GPU)
- On-device training and federated learning
- Cross-platform framework integration (React Native, Flutter, Unity)
- Mobile-specific optimizations (battery, thermal, memory)
- Model management (OTA updates, compression, caching)

---

## Current Status

**Version:** 0.2.1 | **Date:** 2026-07-09 | **Status:** Alpha

### Implementation Status
🔵 **ALPHA** - Core infrastructure implemented; API may change
✅ **~742 CRATE TESTS PASSING** - 0 failed (workspace-wide: 18,102 passed / 0 failed / 119 skipped)
✅ **ZERO CLIPPY WARNINGS** - Clean lint run across the workspace
✅ **26 DOCTESTS PASSING** - 0 failed, 2 ignored (22 doctest failures fixed 2026-07-01 in `expo_plugin.rs`, `react_native_fabric.rs`, and the `mobile_performance_profiler` subsystem: `collector.rs`, `profiler/profiler_impl.rs`, `profiler/profiler_types.rs`, `types.rs`)
✅ **~3,860 PUBLIC API ITEMS** - functions/structs/enums/traits across 187 files in `src/`; 0 `todo!()`/`unimplemented!()` macros remain
✅ **IOS IMPLEMENTED** - Swift package (`TrustformersKit`), Core ML, Metal
✅ **ANDROID IMPLEMENTED** - Java/Kotlin AAR, NNAPI, Vulkan
🟡 **FRAMEWORKS PARTIALLY INTEGRATED** - Flutter and Unity ship real packages; React Native has Rust bridge code + a usage example but no packaged npm module in this repo (see Known Limitations)
✅ **ON-DEVICE TRAINING** - Federated learning with differential privacy (feature `on-device-training`)
✅ **MOBILE OPTIMIZATIONS** - Battery, thermal, and network-adaptive handling
⚠️ **EXPERIMENTAL CRYPTO IS SIMPLIFIED** - `advanced_security.rs` (post-quantum KEM, homomorphic encryption, MPC) is mock/reference code, not audited — see Known Limitations

Checkmarks below indicate "the described capability has corresponding implemented, compiling code in `src/`" — not "independently security-audited" or "benchmarked on physical devices." The items flagged ⚠️ throughout this document are known exceptions verified during the 2026-07-01 documentation pass.

### Feature Coverage
- **iOS:** Swift package (`TrustformersKit`), Core ML, Metal, Neural Engine, ARKit
- **Android:** AAR (`trustformers-android`), NNAPI, Vulkan, Edge TPU, Wear OS, Android Auto
- **Cross-Platform:** Model management (OTA, INT4/INT8/FP16 quantization), federated learning
- **Frameworks:** React Native (JSI bridge + example, no shipped package), Flutter (Dart FFI), Unity (C# MonoBehaviour), Expo (config-plugin scaffolding)
- **Optimizations:** Battery-aware, thermal management (incl. predictive throttle model), network-adaptive handling

---

## Completed Features

### iOS Implementation

#### Swift Package

**TrustformersKit (`ios-framework/`)**

- ✅ **Architecture**
  - Swift/Rust bridge using C FFI
  - Objective-C compatibility layer
  - `TFKModelConfig`, `TFKInferenceEngine`, `TFKModel` types
  - Combine integration (`TFKInferenceEngine+Combine.swift`)
  - Modern async/await support

- ✅ **App Extensions**
  - Widget Extension support
  - Siri Shortcuts integration
  - Share Extension for model sharing
  - Background processing tasks (`ios_background.rs`, `ios_app_extensions.rs`)

**Verified example** (see `ios-framework/TrustformersKit/Sources/TFKInferenceEngine.swift` / `TFKModelConfig.swift`):
```swift
import TrustformersKit

let config = TFKModelConfig.optimizedConfig()
let engine = TFKInferenceEngine(config: config)
let model = try engine.loadModel(at: modelPath, config: config)
let result = engine.performInference(model, input: inputTensor)
```

---

#### Core ML Integration

**Hardware-accelerated inference on iOS**

- ✅ **Model Conversion** (`coreml_converter.rs`)
  - TrustformeRS → Core ML format
  - Quantization-aware conversion
  - Support for custom ops
  - Optimization for Neural Engine

- ✅ **Core ML Delegate** (`coreml.rs`, feature `coreml`)
  - Neural Engine utilization
  - Performance shaders
  - Hybrid execution (Core ML + Metal)
  - Automatic fallback to CPU/GPU

- ✅ **ANE (Apple Neural Engine)**
  - ANE-optimized model graph
  - INT8 quantization for ANE
  - Batch size optimization

---

#### Metal Acceleration

**GPU-accelerated compute on iOS**

- ✅ **Metal Compute Shaders** (`ios/metal.rs`)
  - Custom Metal kernels for transformer ops
  - Matrix multiplication (SIMD groups)
  - Attention mechanisms
  - Activation functions

- ✅ **Metal Performance Shaders (MPS)** (`ios/mps.rs`, tested in `ios/mps_tests.rs`)
  - MPS graph integration
  - Convolution operations
  - Normalization layers

- ✅ **Multi-GPU Support**
  - iPad Pro dual GPU utilization
  - Workload distribution

---

### Android Implementation

#### Android Library

**AAR package for Java/Kotlin (`android-lib/`, groupId `com.trustformers`, artifactId `trustformers-android`)**

- ✅ **Package Structure**
  - AAR creation with Gradle (`minSdkVersion 21`, `compileSdk`/`targetSdk 33`)
  - JNI bindings for Java/Kotlin (`src/main/jni/trustformers_jni.cpp`)
  - ProGuard rules for release builds

- ✅ **Kotlin Coroutine Support**
  - `TrustformersKt` coroutine wrapper (`com.trustformers.TrustformersKt`)
  - `trustformersEngine { }` DSL builder
  - Coroutines integration via `suspend fun`

**Verified example** (see `android-lib/src/main/java/com/trustformers/TrustformersEngine.java` and `TrustformersKt.kt`):
```kotlin
import com.trustformers.trustformersEngine
import com.trustformers.TrustformersEngine

val engine = trustformersEngine(context) {
    setBackend(TrustformersEngine.EngineConfig.Backend.NNAPI)
    setUseFP16(true)
}
val model = engine.loadModel(modelPath)
val output = engine.inference(model, inputTensor)
```

---

#### NNAPI Integration

**Android Neural Networks API**

- ✅ **Hardware Acceleration** (`nnapi.rs`, feature `nnapi`)
  - Backend detection (NPU, GPU, DSP)
  - Fallback strategies
- ⚠️ **TensorFlow Lite delegate** (`tflite_nnapi_delegate.rs`, feature `tflite-nnapi`): source-complete with real `#[cfg(feature = "tflite-nnapi")]` gates throughout, but the file has no `pub mod` declaration anywhere in `lib.rs` — orphaned, unreachable from the crate today; the `tflite-nnapi` feature currently gates nothing. Found during the 2026-07-09 documentation pass (see [Known Limitations](#known-limitations)).

- ✅ **Optimization**
  - Model compilation for NNAPI
  - Quantization (INT8, FP16)

---

#### GPU Acceleration

**OpenGL ES and Vulkan compute**

- ✅ **Vulkan Compute**
  - Vulkan compute pipelines
  - Descriptor sets for memory
  - Command buffer optimization

---

### Cross-Platform Features

#### Model Management

**OTA updates and versioning (`model_management.rs`)**

- ✅ **Over-the-Air Updates**
  - Incremental model downloads (`ModelManager::download_model`)
  - Differential updates (`apply_differential_update`)
  - Rollback support

- ✅ **Compression**
  - Model quantization (INT4, INT8, FP16)
  - Weight pruning, knowledge distillation (`optimization/knowledge_distillation.rs` — reference implementation, see ⚠️ note in Known Limitations)

- ✅ **Caching**
  - Storage cleanup (`cleanup_storage`), cancelable downloads (`cancel_download`)
  - `get_model_path` / `list_models` / `get_storage_stats`

**Verified example** (see `src/model_management.rs`):
```rust
use trustformers_mobile::model_management::{ModelManager, ModelManagerConfig};

let mut manager = ModelManager::new(ModelManagerConfig {
    storage_directory: "/data/local/models".into(),
    ..Default::default()
})?;

manager.download_model("gpt2-medium", Some(Box::new(|progress| {
    let pct = progress.downloaded_bytes as f64 / progress.total_bytes as f64 * 100.0;
    println!("Download: {pct:.1}%");
}))).await?;

let model_path = manager.get_model_path("gpt2-medium");
```

---

#### On-Device Training

**Federated learning and incremental training (feature `on-device-training`)**

- ✅ **Federated Learning** (`federated.rs`)
  - `FederatedLearningClient::new/train_local_model/apply_global_update/get_fl_stats`
  - Differential privacy (`DifferentialPrivacyConfig { epsilon, delta, clipping_norm, noise_mechanism, per_layer_budget }`)
  - `SecureAggregator` (threshold-based share aggregation)
  - ✅ **Updated 2026-08-18, verified stale**: the note this line used to carry ("simplified/mock implementations", "Placeholder Kyber encryption") no longer describes `advanced_security.rs`. It now implements real algorithms: Paillier (additively homomorphic, `paillier.rs`), Shamir secret sharing over GF(2^8) (`shamir.rs`), a Schnorr sigma protocol with Fiat-Shamir (`zkp.rs`), and ML-KEM-768/ML-DSA-65/SLH-DSA-SHAKE-128f — FIPS 203/204/205 — (`pqc.rs`), each covered by regression tests. Genuinely unimplemented pieces (full FHE, Classic McEliece, Falcon, circuit proof systems, garbled circuits/BGW/GMW) return a structured `UnsupportedOperation` error instead of a placeholder. Real caveats remain, per the module's own doc comment: the RustCrypto PQC crates state they are not independently audited, and the Paillier/Schnorr implementations use `num-bigint`'s non-constant-time `modpow`, so neither is hardened against a local timing attacker.

- ✅ **Incremental Learning** (`training.rs`)
  - On-device training loop (`OnDeviceTrainer`, `OnDeviceTrainingConfig`)
  - LoRA (Low-Rank Adaptation) / adapter-based fine-tuning

- ⚠️ **Privacy**
  - Local differential privacy and gradient clipping are implemented and real (`differential_privacy.rs`, `federated.rs`'s `DifferentialPrivacyConfig`)
  - Secure multi-party computation (MPC) and homomorphic encryption are present only as simplified reference code in `advanced_security.rs`

**Verified example** (see `src/federated.rs`):
```rust
use trustformers_mobile::federated::{
    FederatedLearningClient, FederatedLearningConfig, DifferentialPrivacyConfig, NoiseMechanism,
};

let fl_config = FederatedLearningConfig {
    enable_differential_privacy: true,
    dp_config: Some(DifferentialPrivacyConfig {
        epsilon: 1.0,
        delta: 1e-5,
        clipping_norm: 1.0,
        noise_mechanism: NoiseMechanism::Gaussian,
        per_layer_budget: false,
    }),
    ..Default::default()
};

let mut client = FederatedLearningClient::new(fl_config, training_config, mobile_config)?;
let result = client.train_local_model(&local_examples)?;
```

---

### Framework Integration

#### React Native

**Native modules for RN apps — bridge code + example, not yet a packaged module**

- ✅ **Rust-side bridge** (`react_native.rs`, `react_native_turbo.rs`, `react_native_fabric.rs`; features `react-native`/`expo`)
  - Turbo Module / JSI plumbing on the Rust side
  - Fabric renderer integration points

- ⚠️ **Packaging gap**: `react-native-example/` in this repository contains only `TrustformersCompleteExample.tsx` (plus a README stating the same) — there is no `package.json` or module source here, so `npm install trustformers-react-native` (or `@trustformers/react-native`, the name actually used by the example's imports) is **not** installable from this repo today. Renamed from `react-native-plugin/` on 2026-08-24 so the directory name no longer reads as a publishable package.

**Verified example** (from `react-native-example/TrustformersCompleteExample.tsx`):
```typescript
import { TrustformersEngine } from '@trustformers/react-native';

const deviceInfo = await TrustformersEngine.getDeviceInfo();
const engine = await TrustformersEngine.initialize({ enablePerformanceMonitoring: true });
const models = await engine.getAvailableModels();
```

---

#### Flutter

**Dart FFI bindings (`flutter-plugin/`, pub package `trustformers_flutter`, currently version `1.0.0`)**

- ✅ **Platform Channels**
  - `MethodChannel('trustformers_flutter')` + `EventChannel` for streaming
  - Platform views support (`trustformers_platform_view.dart`)

- ✅ **Dart FFI**
  - `dart:ffi` bindings via the `ffi` package
  - Async Dart/Rust bridge (`TrustformersEngine.create(...)`)

**Verified example** (see `flutter-plugin/lib/src/trustformers_engine.dart`, `trustformers_inference.dart`):
```dart
import 'package:trustformers_flutter/trustformers_flutter.dart';

final config = TrustformersConfig(engineId: 'main', modelPath: 'gpt2.bin');
final engine = await TrustformersEngine.create(config);
await engine.loadModel(config.modelPath);

final result = await engine.inference(
  TrustformersInferenceRequest.textGeneration(inputIds: tokenIds),
);
```

---

#### Unity

**C# bindings for Unity (`unity-package/`, UPM package `com.trustformers.mobile`, currently version `1.0.0`)**

- ✅ **Unity Package**
  - `TrustformersEngine : MonoBehaviour` component (attach to a GameObject, not a plain POCO)
  - IL2CPP compatibility (`IL2CPPSupport.cs`)
  - AR Foundation integration (`TrustformersARManager.cs`)

- ✅ **Performance**
  - `TrustformersPerformanceOptimizer.cs`

**Verified example** (see `unity-package/Runtime/TrustformersEngine.cs`):
```csharp
using Trustformers;

// TrustformersEngine is a MonoBehaviour — attach it to a GameObject
var engine = gameObject.AddComponent<TrustformersEngine>();
engine.modelPath = "gpt2.bin";
engine.InitializeEngine();
float[] output = engine.Inference(inputTensor);
```

---

### Mobile-Specific Optimizations

#### Battery Management

**Power-aware execution (`battery.rs`, always compiled)**

- ✅ **Battery Monitoring**
  - `MobileBatteryManager::get_current_reading` / `get_current_battery_level`
  - `BatteryMonitor`, `PowerPredictor`

- ✅ **Adaptive Execution**
  - `AdaptiveInferenceScheduler`, `BatteryOptimizer`
  - `predict_power_consumption`, `get_optimization_recommendations`

**Verified example** (see `src/battery.rs`, `src/device_info.rs`):
```rust
use trustformers_mobile::{MobileBatteryManager, BatteryConfig};
use trustformers_mobile::device_info::MobileDeviceDetector;

let device_info = MobileDeviceDetector::detect()?;
let mut battery_mgr = MobileBatteryManager::new(BatteryConfig::default(), &device_info)?;
battery_mgr.start()?;

let level = battery_mgr.get_current_battery_level();
let recommendations = battery_mgr.get_optimization_recommendations();
```

---

#### Thermal Management

**Prevent thermal throttling (`thermal/`, incl. `thermal/predictive.rs`)**

- ✅ **Thermal Monitoring**
  - CPU/GPU temperature tracking
  - Linear-regression-based predictive throttle model (`thermal/predictive.rs`)

- ✅ **Adaptive Optimization**
  - Reduce precision when hot (FP32→FP16→INT8)
  - CPU-only fallback during thermal stress

---

#### Memory Pressure Handling

**Low-memory mode (`optimization/enhanced_memory_manager.rs`, `optimization/memory_pool.rs`)**

- ✅ **Memory Management**
  - Memory pressure monitoring
  - Model unloading strategies, shared memory pools

- ✅ **Optimization**
  - Quantization under memory pressure
  - Emergency OOM handling

---

### Platform-Specific Features

#### iOS-Specific

**ARKit, iCloud, Privacy**

- ✅ **ARKit Integration** (`arkit_integration.rs`, compiled only for `target_os = "ios"`)
  - AR object detection, scene understanding

- ✅ **iCloud Model Sync** (`ios_icloud.rs`)
  - Sync models across devices, CloudKit integration

- ✅ **Privacy**
  - On-device only processing, privacy manifest compliance

---

#### Android-Specific

**Work Manager, Wear OS, Android Auto**

- ✅ **Work Manager** (`android_work_manager.rs`)
  - Background model updates, periodic training jobs

- ✅ **Wear OS** (`wear_os_support.rs`)
  - Wear OS app support, health & fitness integration
  - Note: several supporting types in this module are explicitly marked in-source as scaffolding ("Additional type stubs for completeness (would be fully implemented)")

- ✅ **Android Auto** (`android_auto_support.rs`)
  - Voice assistant integration, in-car inference

- ✅ **Edge TPU** (`edge_tpu_support.rs`, compiled only for `target_os = "android"`)
  - Google Coral support, quantized model compilation

---

### Testing and Debugging

#### Mobile Testing Framework

**Test infrastructure (`mobile_testing/`)**

- ✅ **Device Farm Integration** (`mobile_testing/device_farm.rs`, `providers.rs`)
  - AWS Device Farm, Firebase Test Lab, local device-farm providers

- ✅ **Performance Benchmarks** (`benchmarks/`, `benchmarks/performance_targets.rs`)
  - Latency/memory/battery/thermal targets (`PerformanceTargets::default()`: <100ms latency, <5%/hr battery drain, 90% device coverage, <50MB framework size)

- ✅ **Testing Tools**
  - Mobile performance profiler (`mobile_performance_profiler/`), memory leak detector (`memory_leak_detector.rs`), model debugger (`model_debugger.rs`), inference visualizer (`inference_visualizer.rs`), crash reporter (`crash_reporter.rs`)

---

### Distribution

#### Package Management

**Multi-platform distribution**

- ✅ **iOS Distribution**: CocoaPods (`TrustformersKit.podspec`, v1.0.0), Swift Package Manager, XCFramework
- ✅ **Android Distribution**: `com.trustformers:trustformers-android:1.0.0` (Gradle `maven-publish` block in `android-lib/build.gradle`)
- ✅ **App Store Compliance**: privacy manifest, app thinning, export compliance notes present in `ios-framework/`

---

## Known Limitations

### Resolved 2026-08-24 (mobile_performance_profiler / battery honesty audit)

- `mobile_performance_profiler/` had never been audited. It now reports real measurements or explicit absence:
  - `collector.rs`: one real `sysinfo`-backed collector replaces the `IOSCollector`/`AndroidCollector`/`GenericCollector` split, which returned three different sets of invented constants (iOS "128 MB heap / 30% CPU / 55% GPU", Android "96 MB / 35% / 60%", generic "64 MB / 25% / 20%") with no platform call behind any of them. Network metrics no longer report a fixed 1 MB sent / 45 ms / 25 Mbps reading.
  - `MobileMetricsSnapshot`'s `memory`, `cpu`, `gpu`, `network`, `thermal` and `battery` are now `Option<..>` — **breaking field-type change**. A family with no source, or disabled by configuration, is absent; it is no longer a zeroed struct that downstream code scored as "idle and healthy".
  - `BottleneckDetector`, `AlertManager` and `PerformanceAnalyzer` evaluate real threshold rules against real snapshots. All three previously constructed empty rule lists that nothing populated, so `detect_bottlenecks()` / `get_active_alerts()` could only ever return empty and `get_current_health()` always returned exactly 85.0 for CPU from a branch that was unreachable.
  - The profiler now uses the real `optimization::OptimizationEngine` (rules + ranking) instead of a same-named placeholder; the engine's `LowCacheHitRate` rule reads the tracker's measured hit rate instead of comparing a constant 50.0, and suggestions report the measurement that tripped the rule instead of a predicted "% improvement" derived from a fixed 30.0 base.
  - Deleted, all proved never compiled (no `mod` declaration anywhere — verified with a `compile_error!` probe): `mobile_performance_profiler/{core,bottleneck,realtime,profiler_split}/` (~3,900 lines). The real rule logic in `bottleneck/detector.rs` and `realtime/monitor.rs` was ported into the live components before deletion.
  - Deleted `mobile_performance_profiler/metrics.rs` (871 lines, 25 tests): an unreferenced duplicate of `collector.rs` that re-declared `MobileMetricsCollector`, `MobileMetricsSnapshot`, `ThermalMetrics`, `BatteryMetrics` and `PlatformMetrics`, and whose every collector returned `Default::default()` or an invented iOS/Android constant.
- `battery.rs`: real `power_supply` sysfs reads on Linux/Android, honest `None` everywhere else. `power_consumption_mw` is no longer `Some(2500.0)`/`Some(2200.0)`/`Some(1800.0)`; `estimate_time_remaining` no longer returns `Some(120)`; `get_current_battery_level` returns `Option<f32>` and no longer guesses 0.85/0.75/0.65/0.5 from charging status; `predict_consumption` extrapolates measured readings instead of a fixed 2.5 W base.
- `mobile_testing/`: `framework.rs` no longer estimates power as `450 + random*100` mW or memory as `256 + random*256` MB (real `sysinfo` RSS now), and no longer reports a memory leak on a 10% coin flip (sustained-RSS-growth heuristic over real samples). Its local `mod rand` and `mod num_cpus` shims (the latter shadowing the real crate with a fixed 4 cores) are gone. `device_farm.rs` no longer invents AWS/Firebase device catalogues or a complete cross-device report (`success_rate: 0.95`, `avg_latency_ms: 50.0`, `best_device: "aws-iphone-14"`) for tests that never ran. `device_farm_tests.rs` had no `mod` declaration and had never been compiled; it is now wired in.
- `inference_visualizer.rs`: attention maps render the caller's real weight tensor instead of a uniform 0.5 matrix; `inference_duration` is `None` rather than a simulated 50 ms; the invented thermal forecast ("+5 C in 60 s at 0.7 confidence") and the four fixed-strength performance trends are gone.
- `training.rs`: LoRA `A` is scaled by `1/sqrt(fan_in)` (the reference LoRA initialization). Unscaled unit-variance init made the loss-reduction tests fail on unlucky draws.
- `simd_analytics.rs`: three functions claimed algorithms they did not implement and were renamed to what they compute — `compute_mutual_information` → `gaussian_mutual_information` (exact only under a bivariate-Gaussian assumption), `compute_simd_isolation_scores` → `compute_simd_standardized_deviation_scores` (no isolation forest is built), `compute_simd_lof_scores` → `compute_simd_inverse_local_density` (not the LOF density *ratio*). `AnomalyScore`'s `isolation_scores`/`lof_scores` and `CrossMetricRelationship`'s `mutual_information` fields were renamed to match — **breaking field renames**.
- `inference_visualizer.rs`: `RealTimeVisualizationMonitor::get_current_state` measures frame rate and mean render time from the buffered frames instead of reporting a fixed 30 fps / 16 ms / 0 dropped / 0.9 quality; `RenderPerformance`'s fields are now `Option` so "not observable" is distinguishable from a measured zero.

### Still open in this area

- `mobile_performance_profiler` has two `#[ignore]`d tests (`collector.rs`, `profiler/profiler_components.rs`) marked `FIXME: 60+ second delays (likely thread/deadlock issue)`. Both files were substantially rewritten in this pass and the suspected deadlock was not investigated; the skipped count is unchanged at 4.
- `ProfilingSummary::battery_consumed_mah` is `sum(mW) / 1000`, which is not milliamp-hours. It is always `None` today (no battery source reaches the profiler), so nothing consumes the wrong unit, but the field name and the formula still disagree.
- `mobile_testing/framework.rs` still constructs `MemoryUsageStats`-free results; `BenchmarkResult::accuracy_metrics`/`power_stats` and `MemoryTestResult::memory_stats`/`allocation_success_rate` are now `Option` and always `None`, because this framework evaluates no labelled data, reads no per-component power rail and has no allocator introspection. Making any of them real needs an evaluation harness and platform allocator hooks that do not exist yet.
- `DeviceFarmManager` cannot execute anything: `run_test_on_device` reports the missing device channel. Real farm support needs an AWS Device Farm / Firebase Test Lab client (neither is a dependency) or a local ADB/`xcrun` driver.


- Core ML Neural Engine requires iOS 16+ for latest features
- NNAPI varies significantly across Android devices
- Large models require quantization for mobile deployment
- Federated learning requires network connectivity
- ARKit requires iPhone XS or newer
- Some features iOS 16+/Android 12+ only
- ⚠️ **Updated 2026-08-18**: `advanced_security.rs` implements real post-quantum KEM/signatures (ML-KEM-768/ML-DSA-65/SLH-DSA-SHAKE-128f, FIPS 203/204/205, via `ml-kem`/`ml-dsa`/`slh-dsa`), real Paillier homomorphic encryption, and real Shamir secret sharing — not mock reference code. The remaining caveat is narrower than before: the underlying RustCrypto PQC crates state they haven't been independently audited, and the Paillier/Schnorr code's `num-bigint`-based `modpow` isn't constant-time (a local-timing-attacker concern, not a correctness one).
- ⚠️ `react-native-example/` ships an example only — no installable npm package source is present in this repository
- ⚠️ Flutter/Unity/iOS/Android sub-packages version independently at `1.0.0` and do not track the workspace `0.2.1` release
- ⚠️ **Newly found 2026-07-09**: `tflite_nnapi_delegate.rs` is fully written (real `#[cfg(feature = "tflite-nnapi")]` gates internally) but has no `pub mod` declaration in `lib.rs` — the `tflite-nnapi` Cargo feature currently gates nothing. Not yet triaged; see [Future Enhancements](#future-enhancements).

---

## 0.2.0 Release Scope

Two workspace-wide tracks land in 0.2.0: **OxiCUDA GPU migration** (scirs2-core `gpu` → OxiCUDA, see `~/work/oxicuda`) and **PyTorch (tch) dependency removal**. The tch decision: delete the `tch` dependency and the `torch` feature entirely in 0.2.0 (workspace `Cargo.toml:82`, trustformers-core `torch` feature + ~40 lines of cfg arms, and the forwarder features in `trustformers`, `trustformers-training`, `trustformers-c`); do not adopt ToRSh now — a P2 task (tracked in the root TODO.md) will evaluate an optional `torsh-interop` feature in 0.3.x once torsh 0.2.0 ships on crates.io. Sub-decision on candle: drop the unused `candle-nn` workspace dep now, keep the `candle` feature/variant through 0.2.0 (it is in every `full` set), and decide implement-vs-remove in 0.3.x. `trustformers-mobile` has no direct tch/candle usage; its 0.2.0 items below belong to the OxiCUDA/scirs2 cleanup track.

### OxiCUDA GPU migration (scirs2-core gpu → OxiCUDA)

- [x] **[P1]** Remove the unused `scirs2-linalg` dependency (done 2026-07-06)
  - Deleted `scirs2-linalg.workspace = true` from `trustformers-mobile/Cargo.toml` together with the root `scirs2-linalg` workspace dep removal (landed atomically as part of the wider tch/torch + workspace-dependency-hygiene cleanup this session).
  - Evidence: `trustformers-mobile/Cargo.toml:33`; root `Cargo.toml:288`
  - Verify: workspace-wide convergence (round 3) confirms `cargo check --workspace --all-features` and `cargo clippy --workspace --all-features --all-targets` both green with zero warnings.
- [x] **[P1]** Fix iOS-only imports of nonexistent scirs2-core APIs in `advanced_neural_engine_v4` (done 2026-07-06)
  - Removed the dead `scirs2_core::linalg::LinalgOps` and `scirs2_core::tensor::Tensor as SciTensor` imports at `src/advanced_neural_engine_v4.rs:23-24` (verified unused anywhere in the file via `rg`; all real tensor ops already used `trustformers_core::Tensor`). Also trimmed the now-unused `CoreError` import in the same block.
  - Evidence: `trustformers-mobile/src/advanced_neural_engine_v4.rs` — `rg -n "SciTensor|LinalgOps|scirs2_core"` now returns no matches.
  - Verify: could not run `cargo check -p trustformers-mobile --target aarch64-apple-ios` directly (target not installed; instructed not to install it), but workspace-wide convergence (round 3) confirms `cargo check --workspace --all-features` green, and the removed symbols are confirmed unused by exhaustive grep.

### PyTorch (tch) dependency removal

- No trustformers-mobile tasks — this crate has no `tch`/`torch`/`candle` dependency or cfg arms. The removal work lives in the root `Cargo.toml`, `trustformers-core`, `trustformers`, `trustformers-training`, and `trustformers-c` TODOs; the P2 `torsh-interop` evaluation is deferred to post-0.2.0 (0.3.x).

---

## Future Enhancements

### High Priority
- ✅ **INT4/GGUF quantization** — nibble-packed INT4 per-group quantization + pure-Rust GGUF reader (`quantization/int4.rs`, `quantization/gguf_mobile.rs`)
- ✅ **Predictive thermal management** — linear regression thermal predictor with proactive throttle prevention (`thermal/predictive.rs`)
- ✅ **WebNN integration (IR + export)** — W3C WebNN IR, graph builder, JSON/compact-JSON export, structural validation (`webnn/mod.rs`)
- [ ] Triage orphaned `tflite_nnapi_delegate.rs` (found 2026-07-09)
  - Goal: the `tflite-nnapi` Cargo feature should either gate something real or be removed — right now it does neither.
  - Design: the file itself is source-complete (real `#[cfg(feature = "tflite-nnapi")]` guards throughout, mirroring `nnapi.rs`'s structure), it's simply missing a `pub mod tflite_nnapi_delegate;` (+ matching feature-gated `pub use`) in `lib.rs`. Two options: (a) wire it up the same way `swin`/`deit` were wired into `trustformers-models` in 0.2.0 (add the `pub mod`/`pub use`, verify `cargo build --features tflite-nnapi` on an Android target), or (b) delete it as dead code the same way `android_renderscript.rs` was deleted, if on closer inspection it's superseded/unwanted.
  - Files: trustformers-mobile/src/lib.rs (+ delete trustformers-mobile/src/tflite_nnapi_delegate.rs if option (b)); README.md, TODO.md.
  - Tests: cargo check -p trustformers-mobile --features tflite-nnapi (Android target) before/after; cargo build --all-features must be unaffected by whichever path is chosen.
  - Risk: low either way — confirmed zero references to this module anywhere else in the crate today, so neither path has blast radius beyond this one file.
- [ ] Improved model compression techniques
  - **Refinement needed:** target compression ratio? Which techniques: GPTQ, AWQ, SqueezeLLM?
- [ ] Replace the `advanced_security.rs` placeholder cryptography (post-quantum KEM, homomorphic encryption, MPC) with audited implementations before advertising real confidentiality guarantees
- [x] Delete orphaned federated_learning.rs / federated_learning_v2/ (planned 2026-07-05)
  - Goal: remove dead code; keep the one real, mounted, tested implementation (federated.rs).
  - Design: delete src/federated_learning.rs, src/federated_learning_v2/ (mod.rs, crypto.rs, privacy.rs), and src/federated_core.rs (bonus orphan found, same rot, zero references) — none are mod-declared in lib.rs, confirmed zero references anywhere in the workspace. Remove the dead commented-out `pub mod federated_learning_v2;` line pair in lib.rs.
  - Files: delete the files above; edit lib.rs, README.md, TODO.md.
  - Tests: cargo build --all-features + cargo nextest run --all-features — count unaffected (these files' tests never compiled).
  - Risk: none — no mod statement ever included these files.

### Performance
- [ ] Further memory optimizations
  - **Refinement needed:** target peak memory reduction %, which platform?
- [ ] Improved battery efficiency
  - **Refinement needed:** target battery % per inference, which benchmark device?
- [ ] Better cache strategies
  - **Refinement needed:** LRU vs ARC vs model-aware eviction? Target cache miss rate?
- [ ] Hardware optimization: iOS ANE (Apple Neural Engine) via CoreML integration
- [ ] Hardware optimization: Android NPU via NNAPI/QNN
- [ ] Hardware optimization: Android Hexagon DSP acceleration
- [ ] Hardware optimization: Qualcomm AI Engine Direct (QAI-Hub integration)
- [x] Delete android_renderscript.rs (planned 2026-07-05)
  - Goal: remove a fully-stubbed, unreachable module; the real Android GPU path (android::gpu + android::engine::AndroidInferenceEngine, Vulkan/OpenGL-ES) already exists and is wired.
  - Design: delete src/android_renderscript.rs (every native primitive is a stub returning null/zero/no-op; not reachable through MobileBackend's real dispatch enum at all; RenderScript is deprecated upstream). Remove its mod/pub use lines from lib.rs.
  - Files: delete the file; edit lib.rs, README.md, TODO.md.
  - Tests: cargo build --all-features -p trustformers-mobile.
  - Risk: none — zero call sites found anywhere.

### Features
- [ ] More AR/VR integrations
  - **Refinement needed:** VisionOS? ARCore extensions? Specific spatial AI use case?
- [ ] Enhanced privacy features
  - **Refinement needed:** what delta beyond existing DP/MPC/HE? Specific threat model?
- [ ] Cross-platform: .NET MAUI bindings for C# mobile apps (note: a standalone `csharp-wrapper/` P/Invoke wrapper already exists, separate from `unity-package/`)
- [ ] Cross-platform: Capacitor.js plugin for Ionic/Angular apps
- [ ] Cross-platform: Kotlin Multiplatform Mobile (KMM) module
- [ ] Real-time collaboration
  - **Refinement needed:** protocol (WebRTC? CRDT? operational transform?), transport, use-case definition.
- [ ] Publish a real npm package for the React Native bridge (`react-native-example/` is example-only; see Known Limitations)
- [x] Remove 3 inert Cargo features (ios, android, mobile-optimized) (planned 2026-07-05)
  - Goal: these features either do something or don't exist — currently they gate nothing.
  - Design: remove all 3 from [features]; change `default = ["mobile-optimized"]` to `default = []` (mandatory — Cargo hard-errors on a default pointing at a removed feature). Update example build commands and Known-Limitations/Feature-Flags sections in TODO.md/README.md.
  - Files: trustformers-mobile/Cargo.toml, TODO.md, README.md.
  - Tests: cargo check -p trustformers-mobile and --all-features before/after — must produce identical compiled output.
  - Risk: none — confirmed zero cfg(feature = "ios"|"android"|"mobile-optimized") anywhere in src/.

---

## Development Guidelines

### Code Standards
- **File Size:** <2000 lines per file (use modularization)
- **Testing:** Comprehensive test coverage with device farm
- **Documentation:** Platform-specific guides
- **Safety:** All C FFI marked as `unsafe`

### Build & Test Commands

```bash
# Build for iOS
cargo build --target aarch64-apple-ios --release

# Build for Android
cargo build --target aarch64-linux-android --release

# Run tests (~742 passing, all features, verified 2026-07-01)
cargo nextest run --all-features -p trustformers-mobile

# Run doctests (26 passing, 0 failed, 2 ignored)
cargo test --doc -p trustformers-mobile --all-features

# CLI tools shipped with the crate
cargo run --bin abi-checker
cargo run --bin trustformers-profiler

# Build Swift package
./build-ios.sh

# Build Android AAR
./build-android.sh
```

### Platform-Specific Setup

#### iOS Setup

```bash
# Install Xcode command line tools
xcode-select --install

# Add iOS targets
rustup target add aarch64-apple-ios
rustup target add x86_64-apple-ios  # Simulator

# Build framework
cd trustformers-mobile
./build-ios.sh

# The framework will be at: ios-framework/TrustformersKit.xcframework
```

#### Android Setup

```bash
# Install Android NDK
export ANDROID_NDK_HOME=/path/to/ndk

# Add Android targets
rustup target add aarch64-linux-android
rustup target add armv7-linux-androideabi
rustup target add x86_64-linux-android

# Build AAR
./build-android.sh

# The AAR will be at: android-lib/build/outputs/aar/ (via Gradle)
```

---

## Sample Applications

Real, more complete sample apps live in `examples/`:
- `examples/ios_demo_app/` — SwiftUI app with text classification, object detection, and ARKit tabs (`ContentView.swift`, `ViewModels.swift`)
- `examples/android_demo_app/` — Jetpack Compose / Kotlin activities: `MainActivity`, `CameraTranslationActivity`, `RealTimeTranslationActivity`, `SmartCameraActivity`, `CodeCompletionKeyboard`, `AccessibilityFeaturesActivity`
- `examples/*.rs` — 5 Rust examples: `integration_test_example.rs`, `mobile_inference_demo.rs`, `mobile_optimization_demo.rs`, `on_device_finetuning_demo.rs`, `platform_apis_demo.rs`

Minimal, API-verified snippets:

### iOS (SwiftUI, minimal)

```swift
import SwiftUI
import TrustformersKit

struct ContentView: View {
    @State private var result = ""
    let engine = TFKInferenceEngine(config: .optimizedConfig())

    var body: some View {
        VStack {
            Text(result)
            Button("Run inference") {
                if let model = try? engine.loadModel(at: modelURL.path, config: .optimizedConfig()) {
                    result = "\(engine.performInference(model, input: inputTensor))"
                }
            }
        }
    }
}
```

### Android (Jetpack Compose, minimal)

```kotlin
import com.trustformers.trustformersEngine

@Composable
fun TrustformersDemo() {
    var result by remember { mutableStateOf("") }
    val engine = remember { trustformersEngine(context) { setUseFP16(true) } }

    Column {
        Button(onClick = {
            val model = engine.loadModel(modelPath)
            result = engine.inference(model, inputTensor).toString()
        }) { Text("Run inference") }
        Text(result)
    }
}
```

---

**Last Updated:** 2026-07-09
**Version:** 0.2.1
**Status:** Alpha
**Test Suite:** ~742 crate tests passing · 26 doctests passing (0 failed, 2 ignored)
**SLoC:** ~103,900 (Rust, `src/`) · ~124,000 (full repo incl. Swift/Kotlin/C#/Dart/TS bindings, via tokei)
**Platforms:** iOS 11+ (Neural Engine/Core ML require iOS 14+/16+), Android 5.0+ / API 21+ (NNAPI requires API 27+)
