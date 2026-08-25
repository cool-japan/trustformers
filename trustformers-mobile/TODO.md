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

**Version:** 0.2.1 | **Date:** 2026-08-25 | **Status:** Alpha

### Implementation Status
🔵 **ALPHA** - Core infrastructure implemented; API may change
✅ **1,390 CRATE TESTS PASSING** - `cargo nextest run -p trustformers-mobile --all-features --no-fail-fast`, 2026-08-25: 1,390 passed / 0 failed / 2 skipped in 85.6s (the 2 skips are the unrelated long-running `benchmarks/mod.rs` `#[ignore]`s). Up from the previous `1,355` figure with the `mobile-round3` honesty/split work below (12 new tests; the remaining delta versus other in-flight waves' own counts is tracked in the root `TODO.md`, not re-measured here).
✅ **ZERO CLIPPY WARNINGS** - `cargo clippy -p trustformers-mobile --all-targets --all-features -- -D warnings`, 2026-08-25: exit 0, 0 warnings
✅ **27 DOCTESTS PASSING** - `cargo test --doc -p trustformers-mobile --all-features`, 2026-08-25: 27 passed / 0 failed / 2 ignored (22 doctest failures fixed 2026-07-01 in `expo_plugin.rs`, `react_native_fabric.rs`, and the `mobile_performance_profiler` subsystem: `collector.rs`, `profiler/profiler_impl.rs`, `profiler/profiler_types.rs`, `types.rs`; count rose from 26 to 27 with the battery-integral work below)
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
- ✅ **TensorFlow Lite delegate** (`tflite_nnapi_delegate.rs`, feature `tflite-nnapi`): **resolved**, verified 2026-08-24 — `lib.rs:147` now declares `#[cfg(feature = "tflite-nnapi")] pub mod tflite_nnapi_delegate;` (mounted the same way as `nnapi`/`nnapi_converter` above), matching the `tflite-nnapi = ["nnapi"]` feature already declared in `Cargo.toml`; every item inside was already correctly `#[cfg(all(target_os = "android", feature = "tflite-nnapi"))]`-gated. Confirmed compiling as part of this crate's `--all-features` build (`cargo clippy -p trustformers-mobile --all-targets --all-features -- -D warnings` is clean). The module's Android-specific behavior itself was not exercised on a physical Android target or the `aarch64-linux-android` toolchain (not installed in this environment, consistent with other entries in this file). When this fix landed was not determined — the module was already mounted when this pass started, with no matching diff in this crate's uncommitted changes, so it predates this session; see [Known Limitations](#known-limitations) and [Future Enhancements](#future-enhancements) for the now-corrected history.

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

### Resolved 2026-08-24 (mobile-followups: ignore-test investigation + battery mAh integral)

- The two `#[ignore]`d `mobile_performance_profiler` tests (`collector.rs::test_collection_lifecycle`, `profiler/profiler_components.rs::test_bottleneck_detection`) marked `FIXME: 60+ second delays (likely thread/deadlock issue)` were investigated for real, not just re-labelled. Neither file spawns a background thread anywhere (`grep -n 'thread::spawn' collector.rs` and the `RealTimeMonitor` in `profiler_components.rs` both come back empty of any live spawn — `_monitor_thread` is set at construction and never becomes `Some`), so there is no thread to deadlock and no channel/`Condvar` anyone forgot to signal; the FIXME predates the `mobile_performance_profiler / battery honesty audit` rewrite above, which structurally removed whatever used to hang. Both tests now run un-ignored: verified passing individually (repeated runs, direct binary invocation with `--test-threads=1`) and as part of the full `cargo nextest run -p trustformers-mobile --all-features` suite (completes in ~25s, no hang). Skipped count: 4 → 2 (the two remaining skips are unrelated `#[ignore]`s, not touched by this pass).
- `ProfilingSummary::battery_consumed_mah` was `sum(mW) / 1000` — summed instantaneous power readings with no time axis and no voltage, which is not a charge (mAh) in any unit system. Rather than just rename the field, real battery telemetry now reaches the profiler and the formula is a genuine time integral:
  - `mobile_performance_profiler::types::BatteryMetrics` gained a `voltage_v: Option<f32>` field, and its other four fields (`level_percent`, `is_charging`, `power_consumption_mw`, `estimated_life_minutes`) changed from bare `u8`/`bool`/`f32`/`u32` to `Option<..>` — **breaking field-type change**, matching the `Option`-per-family pattern the rest of this snapshot already uses. `is_charging` is `None` when the platform's charging status is `Unknown` (every non-Android/Linux target, plus real gauges that publish no `status` node) rather than silently collapsing `Unknown` into `false`.
  - `mobile_performance_profiler::collector::SystemMetricsCollector::collect_battery_metrics` no longer unconditionally returns `Ok(None)`. It now calls `battery::read_live_battery_reading` (new `pub(crate)` function) — the exact same `#[cfg(any(target_os = "android", target_os = "linux"))]`-gated `power_supply` sysfs reader `battery.rs`'s own `MobileBatteryManager` uses, reused rather than re-implemented a second time. `supports_metric("battery")` is now a live host check, the same pattern already used for `"thermal"`, instead of a hardcoded `false`.
  - `calculate_profiling_summary` (`profiler/profiler_impl.rs`) now builds a chronological `(timestamp, power_mw, voltage_v)` series from every snapshot that carries both a power *and* a voltage reading, and integrates it with a new pure, independently unit-tested free function `integrate_battery_consumed_mah`: each reading's own current (`power_mw / voltage_v` at that instant) is computed *first*, then consecutive readings' currents are trapezoidally averaged and multiplied by the interval's elapsed hours, summed across all intervals. Fewer than two qualifying snapshots (the case on every host without a `power_supply` battery node — every non-Android/Linux target, and most CI runners) yields `None`, not a fabricated figure derived from a single instant.
  - Net effect: `battery_consumed_mah` is a real mAh figure on Android/Linux hosts with a battery once a profiling session spans at least two samples, and stays honestly `None` everywhere else, including this crate's own dev/test machine (macOS has no `power_supply` sysfs, and the sysfs-reading code isn't even compiled in on that target).
  - **Corrected 2026-08-24 (mobile-followups, second pass)**: the first-pass implementation above computed each interval's average current as `avg(power_mw) / avg(voltage_v)` (trapezoidal power over trapezoidal voltage), which is *not* the same quantity as the trapezoidal average of the per-instant current `power_mw / voltage_v` whenever voltage moves within an interval — division is nonlinear, so averaging the ratio and taking the ratio of the averages disagree in general (e.g. 1000 mW at 5 V then 1000 mW at 10 V over 1 h: correct answer is (200 mA + 100 mA) / 2 = 150 mAh; the old formula gave 1000 mW / 7.5 V = 133.3 mAh). All five original unit tests held voltage constant across every interval, so they could not distinguish the two formulas — every existing assertion still passes unchanged after the fix. `integrate_battery_consumed_mah` now divides power by voltage at each reading before averaging; a new test, `varying_voltage_averages_current_not_power_over_voltage`, pins the correct 150.0 mAh result against the varying-voltage case above and would fail under the old formula.
  - Updated consumers: the three `mobile_performance_profiler` tests that previously hard-asserted "battery telemetry has no source in this build" (`collector.rs::test_snapshot_carries_real_measurements_not_constants`, `collector.rs::test_supports_metric_reports_the_truth`) now check the host-conditional invariant instead (mirroring the pre-existing `thermal` test pattern), plus a new dedicated `collector.rs::test_battery_is_measured_or_absent`; `types_tests.rs::test_battery_metrics_default` now expects every field `None`. `profiler_components.rs::test_unmeasured_families_never_trip_a_rule` needed no change (it exercises `MobileMetricsSnapshot::default()`, which is unaffected).

### Resolved 2026-08-25 (mobile-honesty2: thermal_power.rs / network_optimization.rs / device_info.rs / react_native.rs)

- **`thermal_power.rs` — `ThermalMonitor::read_temperature` fabrication removed.** Android's `read_android_temperature` returned `Ok(50.0) // Placeholder` under a comment claiming a `/sys/class/thermal/thermal_zone*/temp` read that never happened; it now genuinely scans `thermal_zone0..63`, parses each zone's millidegree reading, and reports the hottest physically-plausible (`-40..200`°C) value, erroring only when no zone is readable at all. iOS's `read_ios_temperature` returned `Ok(48.0) // Placeholder`; `ProcessInfo.thermalState` needs Objective-C FFI this crate does not link (COOLJAPAN keeps FFI feature-gated off by default), so it now returns a structured error naming exactly that. The non-Android/non-iOS branch computed `45.0 + hash(elapsed_millis) % 20` under `// Simulate temperature for testing` — a fabricated value with no test gate feeding *every* desktop/CI build's live thermal state; it now reads real hardware sensors via `sysinfo::Components` (the same mechanism `mobile_performance_profiler::collector::hottest_component_celsius` already uses), erroring when the host exposes none (Apple Silicon, most containers). `ThermalMonitor::update()` no longer lets a read failure propagate as an `update()` error (power monitoring/stats still need to run that tick); it logs and sets `current_state = ThermalState::Unknown` instead of leaving a stale prior reading in place. `ThermalMonitor::new()`'s initial state changed from `ThermalState::Nominal` to `ThermalState::Unknown` — a monitor that has taken no reading yet is not "measured cool", it is "unmeasured" — the same epistemic state a read failure now produces. `ThermalState` gained an `Unknown` variant (`device_info.rs`); `evaluate_thermal_throttling` holds the current throttle level rather than resolving `Unknown` to "safe" or "emergency"; `apply_thermal_optimizations` gives `Unknown` the same mild hedge as `Fair`; `can_run_inference_now`'s thermal check deliberately still gates only on a *confirmed* `Emergency` reading (documented in-line), so an unwired sensor does not halt inference outright on every non-Android build. `PowerMonitor::read_power_info`'s three fabricated per-platform readings (`Some(75)`/`Discharging`/`Some(2500.0)` desktop, `Some(80)`/`Some(2000.0)` Android, `Some(85)`/`Some(1800.0)` iOS) are gone; it now calls `battery::read_live_battery_reading` — the same real `power_supply` sysfs reader `battery.rs`'s `MobileBatteryManager` already uses — reused rather than re-implemented, real on Android/Linux, honestly absent (`None`/`ChargingStatus::Unknown`) elsewhere. `read_platform_temperature`/`temperature_to_state` are now `pub(crate)` free functions (moved out of the private `ThermalMonitor` impl) so `device_info::MobileDeviceDetector::get_current_thermal_state` can delegate to the identical real read instead of running an independent, fabricated pipeline (see below). This file's existing "unmeasured battery" policy (`unwrap_or(100)` in `evaluate_power_optimization`/`apply_power_optimization`, `if let Some(battery) = ..` in `can_run_inference_now`) was deliberately left unchanged — a separate policy decision from removing a fabricated thermal reading, and a new dedicated test (`test_unmeasured_battery_is_treated_the_same_way_across_this_file`) locks in that the three sites still agree with each other. Found in the same file while auditing: `PowerAwareScheduler::get_next_ready_inference` (reached from the public `ThermalPowerManager::get_next_inference`) ignored its `config` parameter and scheduled the popped request against `MobileConfig::default() // Would be optimized config` instead — the sibling immediate-run path `schedule_inference` correctly uses `current_config.clone()`; fixed to match. 10 new tests.
- **`network_optimization.rs` — `wifi_only`/`charging_only` download constraints converted to the fail-closed pattern `android_work_manager.rs:1176-1204` already established.** `is_wifi_connected`/`is_device_charging` returned hardcoded `true`/`false` under comments claiming platform-specific detection; they now return `Option<bool>` (`None` — no platform detector is wired up yet, on every target). `check_download_constraints` — reached from the live `start_resumable_download` path — no longer silently lets a `wifi_only` download proceed on cellular (`true` meant the check could never fire) or silently blocks every `charging_only` download forever indistinguishably from "not currently charging" (`false` meant the check always fired, whether or not charging state was even knowable); it now refuses with a structured error naming exactly which constraint could not be verified, only when the caller actually declared that constraint. A request with neither constraint is unaffected — the unverifiable checks are never consulted. Found in the same file while fixing the above: `get_current_time_info` — also reached from the same live `check_download_constraints` path, via `is_time_in_window` — returned a hardcoded `hour: 12, day_of_week: 1` regardless of the actual clock, so a `time_windows`-restricted download (e.g. "only overnight") was gated by whether noon-Monday happened to fall in the configured window, never the real time. Implemented for real via `chrono::Local::now()` (already a workspace dependency, the same crate/pattern `trustformers-serve`'s cache warmer uses for the identical question) rather than left as a placeholder, per this wave's IMPLEMENT POLICY. 5 new tests (4 for the wifi/charging fail-closed conversion, 1 for the real clock).
- **`device_info.rs` — thermal/power capability constants replaced with real platform facts or honest absence, feeding the `TODO.md`-advertised `MobileDeviceDetector::detect()` API.** `get_current_thermal_state` (`ThermalState::Nominal` under `// Platform-specific thermal state detection`, which detected nothing) now delegates to `thermal_power::read_platform_temperature` + `temperature_to_state` — the identical real read `ThermalPowerManager`'s live monitor uses — falling back to `ThermalState::Unknown` only when that read fails. `is_thermal_throttling_supported` (hardcoded `true`) is now a static `cfg!(any(target_os = "android", target_os = "ios"))` platform fact rather than an unconditional assertion that even a plain desktop build supports throttling. `is_power_save_mode_active` (hardcoded `false`) now returns `Option<bool>` = `None`: no pure-Rust binding reaches `PowerManager.isPowerSaveMode()` / `ProcessInfo.isLowPowerModeEnabled` on any target this crate builds for today, so `PowerInfo.power_save_mode` is now `Option<bool>` too (**breaking field-type change**) rather than an asserted `false`. `is_low_power_mode_available` (hardcoded `true`) is now the same `cfg!`-based platform fact as `is_thermal_throttling_supported` — both features have existed on Android/iOS since versions this crate could plausibly target, so `true` there is a real fact, not a guess, while a generic/desktop build correctly reports `false` instead of the previous blanket `true`. `adjust_for_power_state` previously read `power_info.battery_level_percent.unwrap_or(100)`, fabricating "fully charged" on every platform without a real battery reader (every non-Android/Linux target); it now treats `power_save_mode`/`battery_level_percent` as independently-contributing signals that are simply skipped when unknown, rather than defaulting to either "assume charged" (the old bug) or "assume critical" (the opposite fabrication) — verified by two new tests, one confirming no adjustment fires with nothing known, one confirming a genuinely known low battery still triggers the moderate-saving branch unchanged. `TODO.md`'s "Verified example" battery-management block (previously here) is not itself affected by these fixes — `MobileBatteryManager`/`get_current_battery_level` were already real from the earlier battery honesty audit; this pass is about the separate thermal/power *capability-detection* constants `detect_thermal_info`/`detect_power_info` assemble into the same `MobileDeviceInfo` that example constructs. 6 new tests.
- **`react_native.rs` — model metadata crossing the JS bridge is now observed from real inference, not a fabricated ImageNet shape.** `get_available_models`/`get_model_info` returned `input_shape: vec![1, 224, 224, 3]` / `output_shape: vec![1, 1000]` for *every* model regardless of architecture (an ImageNet-classifier shape, with comments admitting "Placeholder" / "Would get from actual model"). Nothing in a loaded checkpoint (safetensors/PyTorch/ONNX weight maps) declares a fixed input signature the way a compiled graph would, so there is no shape to report until a real request has actually run one; `ModelInfo.input_shape`/`output_shape` are now `Option<Vec<usize>>` (**breaking field-type change**), populated from a new `observed_shapes: Arc<Mutex<HashMap<String, ObservedShapes>>>` that `perform_inference_internal` writes into from the real request/response tensors on every successful inference, and read back by `get_model_info`/`get_available_models` — `None` until a model's first successful inference, the real observed shapes after. `InferenceResponse.memory_used_mb` was a hardcoded `50` on success / `0` on failure; both paths now read the engine's real `get_memory_info().total_memory_mb` (parameter-count- and quantization-scheme-derived, via the existing `MobileOptimizationEngine::estimate_memory_footprint`) while still holding the engine lock, so a failed inference (e.g. a shape mismatch) no longer misreports a loaded model as using zero memory. The stale `// In a real implementation, you'd use a runtime like tokio` comment on the FFI export `trustformers_rn_inference` is corrected: `tokio` is a real dependency this crate does use elsewhere (`inference_background`'s `spawn_blocking`, reached from the async `inference()` non-FFI path); this specific `extern "C" fn` calls the synchronous `inference_sync` because a plain C ABI entry point has no executor handle available to it, not because of an unaddressed gap. 2 new tests.

### Resolved 2026-08-25 (stragglers: network_optimization.rs P2P shared-model hash/size)

- **`network_optimization.rs` — `P2PManager::add_shared_model` no longer fabricates `model_hash`/`size_bytes`.** It previously constructed `SharedModel { model_hash: "placeholder_hash".to_string(), size_bytes: 1024 * 1024, .. }` for every model regardless of content, exactly as `mobile-honesty2` found. The real fix `mobile-honesty2` identified as needed — the signature receiving model bytes or a path, since this manager has no `ModelManager` reference to source them from — is what landed: `enable_p2p_sharing`/`add_shared_model` now take a `model_path: &Path` (a path rather than `&[u8]`, deliberately: a shared model can be hundreds of megabytes, and streaming the file through the hasher in fixed-size chunks avoids forcing a full in-memory copy on a memory-constrained device). The hash is real `sha2::Sha256`, hex-encoded via `hex::encode` — both already workspace dependencies of this crate, same crates and idiom as `nnapi_converter.rs::compute_model_hash` (no new dependency added). `size_bytes` is the real byte count read. A missing/unreadable file now returns a structured error (via `std::io::Error`'s existing `TrustformersError` conversion) instead of recording a fabricated entry. `availability_score: 1.0` was left as-is but is now documented as a real fact (this node has just read the full file, so it is 100% locally available), not a network-replication score — nothing yet tracks peer replication, hence `peer_sources` staying empty. 2 new tests: real-hash/real-size plus same-content-same-hash/different-content-different-hash, and a missing-file structured-error regression guard (`network_optimization::tests::test_add_shared_model_computes_real_hash_and_size_not_placeholders`, `test_add_shared_model_errors_on_missing_file_instead_of_fabricating`). `enable_p2p_sharing` had zero external callers repo-wide before this change, so the signature change is not breaking anything downstream today.

### Resolved 2026-08-25 (mobile-round3: network_optimization.rs split / thermal_power.rs sentinel bug / device_info.rs enumerate_thermal_zones / intelligent_config_optimizer.rs battery fabrication / P2P exact-hash tests)

- **`network_optimization.rs` split from 2013 lines (the only workspace `.rs` file over the <2000-line policy) into three files, zero behavior change.** Manual extraction, the same pattern already proven on `trustformers-debug/src/computation_graph.rs`: the ~560 lines of pure `#[derive(..., Serialize, Deserialize)]` config/data types (`NetworkOptimizationConfig` and everything it transitively contains, down through `DownloadStatus`) moved to a new `network_optimization_types.rs`, declared `#[path = "network_optimization_types.rs"] mod types;` + `pub use types::*;` at the top of `network_optimization.rs` so every existing `network_optimization::TypeName` path — including the crate-root re-export list in `lib.rs` — resolves identically to before. The ~270-line `#[cfg(test)] mod tests { .. }` block moved to `network_optimization_tests.rs` the same way `computation_graph_tests.rs` does (`#[cfg(test)] #[path = "network_optimization_tests.rs"] mod tests;`). All private manager structs (`NetworkOptimizationManager`, `DownloadManager`, `P2PManager`, `SharedModel`, ...) and their `impl` blocks stayed in `network_optimization.rs` itself — moving them into the types submodule would have needed `pub(crate)`/`pub(super)` on every field the tests file reads (`shared_models`, `model_hash`, `size_bytes`, `p2p_manager`), which is exactly the visibility trap the extraction was designed to avoid; leaving them in place means the `#[path]` tests module (still a child of `network_optimization`, exactly as before) keeps seeing every private item it already saw. Sizes after (rustfmt-clean): `network_optimization.rs` 1186, `network_optimization_types.rs` 574, `network_optimization_tests.rs` 402 (+ 3 new tests documented below). Verified with `cargo clippy -p trustformers-mobile --all-targets --all-features -- -D warnings` (exit 0) and a full `cargo nextest run -p trustformers-mobile --all-features` (see the crate test count above) — every pre-existing test, including the P2P tests from the `stragglers` entry above, still passes under its original name (`network_optimization::tests::test_add_shared_model_computes_real_hash_and_size_not_placeholders` etc.), confirming the module path did not change.
- **`thermal_power.rs` — the `PLAUSIBLE_MILLIDEGREES` range accepted the two sentinel values its own comment claimed to reject.** `read_android_temperature`'s filter was `-40_000..=200_000`, which *contains* both `0` and `-1` — the kernel's sentinels for an unpopulated `/sys/class/thermal/thermal_zone*/temp` zone — so a zone reporting either passed straight through as a "real" reading. Normally masked by `.max()` (a spurious 0/-1 rarely wins against a real reading), but when *every* zone is a sentinel the function returned a fabricated `Ok(0.0)` instead of the structured error it exists to return, and `temperature_to_state(0.0)` then published a healthy `Nominal` thermal state synthesized from zero real sensors. Fixed by excluding `0` and `-1` *explicitly* (`SENTINEL_MILLIDEGREES`) rather than narrowing the plausible range, which would also have discarded a genuine sub-zero reading (e.g. a modem thermistor at -0.5°C) — trade-off documented in-line: a real zone reporting *exactly* 0 or -1 millidegrees is now indistinguishable from an unpopulated one, accepted deliberately because that is overwhelmingly the more common truth. The parse-and-filter chain was also pulled out of the `#[cfg(target_os = "android")]`-gated function into a new pure, target-independent `hottest_plausible_temperature(impl Iterator<Item = &str>) -> Result<f32>`, so the filtering logic itself is unit-tested on every host (this crate's own dev/CI machines never compile `read_android_temperature`'s body at all) instead of only in theory; `read_android_temperature` is now a thin real-filesystem-scan wrapper around it. 5 new tests: sentinel-only input errors, unreadable/out-of-range-only input errors, a mixed scan reports the hottest genuine reading, a genuine sub-zero reading outside the sentinel band is kept, and empty input errors.
- **`device_info.rs` — `enumerate_thermal_zones` no longer a `vec![]` stub under a comment claiming an enumeration it never performed.** Live at `detect_thermal_info` (populates `ThermalInfo.thermal_zones`, which — checked crate-wide — nothing reads yet; this pass only fixes the honesty of populating it, not the still-unused field itself). Now does a real `/sys/class/thermal/thermal_zone{0..64}/type` scan on `#[cfg(any(target_os = "android", target_os = "linux"))]` (the `type` file holds each zone's kernel name, e.g. `"cpu-thermal"`; sibling convention to the `.../temp` millidegree scan `thermal_power::read_android_temperature` already performs, with the `MAX_THERMAL_ZONES` bound duplicated across the two files rather than sharing visibility, the same trade-off that file's own desktop-temperature doc comment already accepts and justifies for the identical reason). No such sysfs convention exists on iOS or a generic desktop/CI host, so `#[cfg(not(any(target_os = "android", target_os = "linux")))]` honestly returns `Vec::new()` instead of a fabricated name. 3 new tests: one for the honest-empty branch (compiles only off Android/Linux); on Android/Linux, one unconditional check that every returned name is non-empty (never gated behind an `if`, so it cannot silently execute zero assertions on a container without `/sys/class/thermal`) plus one explicitly `if`-guarded check that zones are actually found when `/sys/class/thermal/thermal_zone0` exists on the runner — split into two rather than one combined `if`-wrapped test specifically so the first always runs a real assertion regardless of what the host provides.
- **`optimization/intelligent_config_optimizer.rs` — `DeviceCapabilityAssessment::{battery_level, power_constrained}` no longer re-fabricate a 50% battery.** `assess_device_capabilities` read `device_info.power_info.battery_level.unwrap_or(50)` twice — once to derive a 0.0-1.0 `battery_level`, once to derive `power_constrained` — inventing "half battery, not power-constrained" on every platform without a real battery reader (every non-Android/Linux target today). Both fields are now `Option<f32>`/`Option<bool>` (**breaking field-type change** — `DeviceCapabilityAssessment` is `pub`, reachable as `trustformers_mobile::optimization::DeviceCapabilityAssessment`, though grepped workspace-wide with zero consumers of either field before this change), `None` when the underlying reading is `None`. Deliberately *not* following this same file's `ThermalState::Unknown => 0.5` precedent immediately above it in the same function: that neutral midpoint is a documented, reporting-only "assume typical" choice with no consumer that turns it into a decision — independently re-verified here, not just taken on the in-file comment's word: `DeviceCapabilityAssessment` (which is what `overall_score` lives on) has zero readers of `.overall_score` anywhere, including inside this file itself (`grep -n '\.overall_score\b' intelligent_config_optimizer.rs` finds only the struct-literal write site), and the type itself is never named outside this file workspace-wide, so nothing anywhere can even hold an instance to read the field off. A battery percentage is different in kind, not just in current wiring: a caller reading `Some(0.5)` cannot tell "the device happens to be at 50%" from "nothing was measured," so `None` preserves a distinction `Option<f32>` can express that a neutral constant cannot, independent of whether anything reads it today. 3 new tests: an unmeasured battery yields `None`/`None`; a known 20%-and-not-charging battery yields the exact `Some(0.2)`/`Some(true)`; a known 90%-and-charging battery yields `Some(0.9)`/`Some(false)`.
- **P2P shared-model hash: the exact-digest test the `stragglers` entry above was missing.** Its two tests (still passing, see above) prove `add_shared_model` is real (not a placeholder, reproducible, content-sensitive, errors on a missing file) but never assert a specific expected digest and never go through the public `NetworkOptimizationManager::enable_p2p_sharing` entry point (only the private `P2PManager::add_shared_model` directly). Three new tests close both gaps: `test_enable_p2p_sharing_produces_exact_sha256_and_size_for_known_bytes` writes `b"abc"` to a `std::env::temp_dir()` file and asserts the exact SHA-256 hex `ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad` (the standard FIPS 180-4 example digest, independently cross-checked here against both `shasum -a 256` and `openssl dgst -sha256`, not merely recomputed with the same `sha2` crate the implementation uses) and exact size `3`, reached through `enable_p2p_sharing(...).await` end to end (config gate, `Mutex`-guarded `P2PManager`, real `tracing::info!` path); `test_enable_p2p_sharing_with_flag_disabled_errors_and_records_nothing` locks the negative case (`enable_p2p_sharing: false`, the config default) actually refuses and records nothing, not just "some tests exist that happen to enable it"; `test_add_shared_model_exact_sha256_across_chunk_boundary` uses a 70,000-byte `(i % 256) as u8` payload — larger than the implementation's 65,536-byte read chunk — against digest `0c6c96cc20d3f906e54f1f1296e8878c1ac39262fb587cd56235c3aa9103d837` (same independent cross-validation), so the chunked-accumulation loop is exercised beyond a single read, not just the small single-chunk payloads the existing tests use.

### Resolved 2026-08-26 (wave6e cross-polish: device_info.rs enumerate_temperature_sensors)

- **`device_info.rs` — `enumerate_temperature_sensors` no longer a `vec![]` stub under a `// Enumerate available temperature sensors` comment that performed no enumeration at all -- the verbatim sibling of `enumerate_thermal_zones`'s stub, fixed the same way above (mobile-round3), on every platform including Android/Linux.** Live at `detect_thermal_info` (populates `ThermalInfo.temperature_sensors`, which -- checked crate/workspace-wide -- every other call site only ever *constructs* it (`vec![]`/`Vec::new()` default/mock fixtures in `advanced_quantization.rs`, `intelligent_config_optimizer.rs`, `network_adaptation/stats.rs`, `profiler/tests.rs`, this file's own `Default` impl, and one hand-built non-empty fixture in `tests/device_detection_tests.rs`); nothing anywhere reads it back out, the same unread-but-now-honestly-populated situation `enumerate_thermal_zones` documented above, and this pass does not change that either). Three real platform branches, mirroring `thermal_power::read_platform_temperature`'s own three-way split rather than `enumerate_thermal_zones`'s two-way one, because a real second source (`sysinfo::Components`) is available for the branch that function left empty:
  - `#[cfg(any(target_os = "android", target_os = "linux"))]`: real `/sys/class/thermal/thermal_zone{0..64}/{type,temp}` scan, pairing `enumerate_thermal_zones`'s own `type`-file name scan with a `temp`-file reading per zone, filtered through the same unpopulated-zone-sentinel logic (`0`/`-1` millidegrees excluded, `-40_000..=200_000` plausible range) `thermal_power::hottest_plausible_temperature` established -- both the `MAX_THERMAL_ZONES` bound and the sentinel/range constants are duplicated locally rather than shared across the module boundary, the same trade-off `enumerate_thermal_zones` already accepted for the same reason. No per-zone "safe max" sysfs convention is scanned (trip-point files exist but their count/ordering/labels are not standardized across zones or OEMs), so `max_temperature_celsius` is honestly `None` on this branch, not guessed.
  - `#[cfg(target_os = "ios")]`: honestly empty -- iOS exposes no per-sensor API at all (only `ProcessInfo.thermalState`, unreachable without the Objective-C FFI this crate's default build does not implement; see `thermal_power::read_ios_temperature`).
  - Desktop (macOS/Windows/other Unix): real `sysinfo::Components` readings, the same mechanism `thermal_power::read_desktop_temperature` and `mobile_performance_profiler::collector::hottest_component_celsius` already use for this crate's other thermal telemetry. `critical()` (sysinfo's "highest temperature before the component halts") backs `max_temperature_celsius`; `max()` (highest temperature *observed so far*, a different quantity) is not used. Both readings are filtered to finite values (`f32::NAN` is sysinfo's own documented Linux "failed to retrieve" sentinel) and components with an empty `label()` are skipped. Confirmed live, not just theoretically real: on this crate's own Apple Silicon macOS dev host, this returns ~33 real PMU/NAND/battery-die sensors with genuine, varying Celsius readings; `critical()` is uniformly `None` on that backend (unlike Linux hwmon, where it is sometimes populated) -- an observed platform fact, not a bug.
  - 4 new tests, cfg-split the same way the implementation is: an unconditional iOS-empty check; on Linux/Android, one unconditional non-placeholder-name/plausible-range/no-fabricated-max check (never gated behind an `if`, so it cannot silently execute zero assertions on a container without `/sys/class/thermal`) plus one explicitly `if`-guarded check that sensors are actually found when `/sys/class/thermal/thermal_zone0` exists; on desktop, one non-placeholder/finite check that cannot also assert the list is non-empty (a sandboxed CI container may genuinely expose zero `sysinfo` components, which is the honest answer there, not a bug) -- verified non-vacuous on at least this dev host per the above.
  - Verified: `cargo clippy -p trustformers-mobile --all-targets --all-features -- -D warnings` and `cargo clippy -p trustformers-mobile --all-features --target aarch64-unknown-linux-gnu -- -D warnings` both exit 0 (the second specifically to type-check the Android/Linux branch, which the host's own `--all-targets` run cannot reach on this macOS dev machine); `cargo nextest run -p trustformers-mobile --all-features` 1391/1391 passed, 2 skipped (was 1390/2 before this pass -- the +1 is exactly the desktop-branch test, the only one of the four new tests whose `cfg` compiles in on this host). iOS cross-compilation (`--target aarch64-apple-ios`) was not independently verified this pass: it fails before reaching this crate at all, in `scirs2-core`'s `mobile_ffi.rs` (`libc` unresolved), a pre-existing gap in a transitive dependency unrelated to this change. Both touched `.rs` files are rustfmt-clean for every line this pass added or edited (`rustfmt --check` on each; the one remaining diff in `differential_debugging.rs` from the unrelated debug package is pre-existing, untouched code at a different line).
  - Latency check, since `enumerate_temperature_sensors`'s desktop branch adds a second `sysinfo::Components::new_with_refreshed_list()` call inside `detect_thermal_info` (already one call, via `get_current_thermal_state` -> `thermal_power::read_platform_temperature` -> `read_desktop_temperature`) -- the exact "per-request sysinfo latency" class `mobile-round3`'s HostSampler-motivated fixes elsewhere in this wave exist to avoid. Traced every production caller of `MobileDeviceDetector::detect()` workspace-wide (`android_work_manager.rs`, `arkit_integration.rs`, `edge_tpu_support.rs`, `flutter.rs` x2, `unity_interop.rs`, `c_api.rs`, `lifecycle/mod.rs` via `DeviceMonitor::new`, `mobile_performance_profiler/profiler/profiler_impl.rs`, `mobile_testing/{device_farm,framework}.rs`): every one is either a one-time constructor/initializer (`::new`, `initialize`) called once per object lifetime, or an explicit on-demand FFI/method-channel "get device info" query handler invoked only when the external caller (Unity/Flutter/C host app) asks -- never something this crate itself calls automatically in a loop or per inference/request. The measured ~0.3s desktop cost (this dev host) is therefore a one-time or caller-paced cost, not a hot-path regression.

### Still open in this area

- `mobile_testing/framework.rs` still constructs `MemoryUsageStats`-free results; `BenchmarkResult::accuracy_metrics`/`power_stats` and `MemoryTestResult::memory_stats`/`allocation_success_rate` are now `Option` and always `None`, because this framework evaluates no labelled data, reads no per-component power rail and has no allocator introspection. Making any of them real needs an evaluation harness and platform allocator hooks that do not exist yet.
- `DeviceFarmManager` cannot execute anything: `run_test_on_device` reports the missing device channel. Real farm support needs an AWS Device Farm / Firebase Test Lab client (neither is a dependency) or a local ADB/`xcrun` driver.
- **New, found 2026-08-24 by `mobile-followups` while verifying its three named missions (not one of them, and not attempted this pass) — `src/lifecycle/` has the same class of fabrication `mobile-audit` found and fixed in `mobile_performance_profiler/` last wave, still unfixed here.** `mobile-audit`'s Wave-4 sweep covered `mobile_performance_profiler/`; `lifecycle/` (the separate `LifecycleManager` app-state/background-task subsystem) was not part of that sweep and has not had a dedicated audit. Evidence, all read directly from source today:
  - `LifecycleManager`'s `system_monitors: SystemMonitors` / `execution_context: ExecutionContext` fields are built once from hardcoded constants and never refreshed: `SystemMonitors::new()` (`mod.rs:1151-1169`) sets `CpuMonitor { current_usage_percent: 25.0, core_count: 4, frequency_mhz: 2400.0, temperature_celsius: 35.0 }`, `MemorySystemMonitor { total_memory_mb: 2048, available_memory_mb: 1024, used_memory_mb: 1024, cached_memory_mb: 256 }`, `NetworkMonitor::new()` (`mod.rs:1183-1191`) sets `bandwidth_mbps: 25.0`/`signal_strength: 80`/always `WiFi`; `ExecutionContext::new()` (`mod.rs:1113-1136`) sets `AvailableResources { cpu_percent: 100, memory_mb: 1024, .. }` and `SystemState { battery_level: 100, network_connected: true, .. }`. A grep for reassignment of `current_usage_percent`/`available_memory_mb`/`bandwidth_mbps` anywhere in `mod.rs` returns zero hits, and `SystemMonitors::initialize`/`start_monitoring` (`mod.rs:1171-1179`) are empty `Ok(())` bodies whose comments ("Initialize monitoring systems", "Start background monitoring tasks") claim a behavior neither method has. Everything downstream that reads these fields — `determine_battery_state`, `assess_network_quality`, `calculate_system_health`, `capture_resource_usage_snapshot`, `generate_system_health_report`'s `network_health_score: 100.0, // Simplified` (`mod.rs:1371`) — combines them with real arithmetic but off a permanently-fixed process-start snapshot, not live device state. `mobile_performance_profiler::collector::SystemMetricsCollector` already reads real `sysinfo` data for the equivalent metrics (this wave's earlier fix); reusing it here instead of `lifecycle/`'s own unwired monitor structs is the likely fix shape, not re-deriving a second real collector.
  - `BackgroundCoordinator::execute_task_impl` (`tasks.rs:447-497`) does not execute the scheduled task regardless of its `TaskType` (`ModelUpdate`/`FederatedLearning`/`DataSync`/`CacheCleanup`/`Analytics`/`Backup`/`Precomputation`/`HealthCheck`): it sleeps 100 ms and returns a `TaskResult` with `status: Completed` unconditionally and hardcoded performance/resource numbers (`peak_cpu_percent: 25.0`, `battery_consumption_mah: 5.0`, `throughput_ops_per_second: 10.0`, `resource_efficiency_score: 85.0`, five more) identical on every call. `TaskResult` has no `task_type`/`priority` field to read back, so `LifecycleManager::extract_task_type` (`mod.rs:724-727`) papers over the gap with a hardcoded `Some(TaskType::ModelUpdate)` for every completed task (`BackgroundTask.task_type` — the real value — is dropped when `execute_next_task` consumes the task), and `convert_resource_usage`/its call site (`mod.rs:729-737`, `mod.rs:445,447`) hardcode `avg_execution_time_seconds: 0.0`, `priority: TaskPriority::Normal`, `wait_time_seconds: 0.0` the same way. Separately, `BackgroundCoordinator::pause_non_essential_tasks`/`resume_paused_tasks`/`suspend_all_tasks`/`cancel_all_tasks` (`mod.rs:1213-1231`) are four `// Implementation would ... ` + `Ok(())` bodies that report success while doing nothing.
  - `calculate_time_since_user_interaction`/`calculate_foreground_duration`/`calculate_background_duration` (`mod.rs:739-752`) return hardcoded `60`/`300`/`30` regardless of state. `execution_context.user_context.last_interaction_time: Option<Instant>` exists and is closer to real data than the constant, but nothing ever writes to it after `ExecutionContext::new()` sets it once (`mod.rs:1131`), so wiring it in today would report time-since-launch, not time-since-interaction — an actual interaction callback is a second missing piece.
  - Rough severity signal, not a full read: grepping `mod.rs`/`tasks.rs`/`state.rs`/`stats.rs`/`config.rs` for this cluster's own markers (`Simplified`, `Default implementation`, `would typically`, `Calculate (from|based)`, `Implementation would`, `Extract from result metadata`, `Platform-specific implementation needed`) finds 28 hits in `mod.rs`, 3 in `tasks.rs`, 0 in the other three files (grepped only, not read in full, so absence of a hit is not proof of absence of the problem there). The concentration in `mod.rs`/`tasks.rs` and the shape of the fixes needed (a real device-monitor wiring plus real per-`TaskType` execution semantics, not just formula corrections) makes this a dedicated future audit package's scope, the same size class as `mobile-audit`'s Wave-4 pass over `mobile_performance_profiler/`, not a spot fix — not attempted this pass.


- Core ML Neural Engine requires iOS 16+ for latest features
- NNAPI varies significantly across Android devices
- Large models require quantization for mobile deployment
- Federated learning requires network connectivity
- ARKit requires iPhone XS or newer
- Some features iOS 16+/Android 12+ only
- ⚠️ **Updated 2026-08-18**: `advanced_security.rs` implements real post-quantum KEM/signatures (ML-KEM-768/ML-DSA-65/SLH-DSA-SHAKE-128f, FIPS 203/204/205, via `ml-kem`/`ml-dsa`/`slh-dsa`), real Paillier homomorphic encryption, and real Shamir secret sharing — not mock reference code. The remaining caveat is narrower than before: the underlying RustCrypto PQC crates state they haven't been independently audited, and the Paillier/Schnorr code's `num-bigint`-based `modpow` isn't constant-time (a local-timing-attacker concern, not a correctness one).
- ⚠️ `react-native-example/` ships an example only — no installable npm package source is present in this repository
- ⚠️ Flutter/Unity/iOS/Android sub-packages version independently at `1.0.0` and do not track the workspace `0.2.1` release
- ✅ **Resolved, verified 2026-08-24**: the 2026-07-09 finding below is stale. `tflite_nnapi_delegate.rs` now has a `#[cfg(feature = "tflite-nnapi")] pub mod tflite_nnapi_delegate;` declaration at `lib.rs:147` — the `tflite-nnapi` Cargo feature gates the module for real. See the NNAPI Integration entry above and [Future Enhancements](#future-enhancements) for what's still unverified (behavior on an actual Android target).

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
- [x] Triage orphaned `tflite_nnapi_delegate.rs` (found 2026-07-09; option (a) landed, verified 2026-08-24)
  - Goal: the `tflite-nnapi` Cargo feature should either gate something real or be removed — right now it does neither.
  - Resolution: option (a) — `lib.rs:147` now declares `#[cfg(feature = "tflite-nnapi")] pub mod tflite_nnapi_delegate;`, mirroring `nnapi`/`nnapi_converter`'s mounting immediately above it. Exactly when this landed was not determined (it predates this pass, with no matching diff in this session's changes — see the NNAPI Integration entry above).
  - Verified 2026-08-24: `cargo clippy -p trustformers-mobile --all-targets --all-features -- -D warnings` compiles the module clean (exit 0, 0 warnings) as part of this crate's normal gate run. **Not verified**: the module's actual behavior on `aarch64-linux-android` or a physical device — the Android target isn't installed in this environment (same limitation recorded elsewhere in this file, e.g. the OxiCUDA migration section above), so `cargo build --features tflite-nnapi` on a real Android target is still open if anyone wants that specific proof.
  - Files: trustformers-mobile/src/lib.rs, TODO.md (this entry).
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

# Run tests (1,355 passing / 2 skipped, all features, verified 2026-08-24)
cargo nextest run --all-features -p trustformers-mobile

# Run doctests (27 passing, 0 failed, 2 ignored, verified 2026-08-24)
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

**Last Updated:** 2026-08-24
**Version:** 0.2.1
**Status:** Alpha
**Test Suite:** 1,355 crate tests passing / 2 skipped · 27 doctests passing (0 failed, 2 ignored) — `cargo nextest run`/`cargo test --doc`, both `--all-features`, verified 2026-08-24
**SLoC:** ~104,000 (Rust code in `src/`, `tokei` 12.1.2, 2026-08-24: 186 files / 103,992 lines of code) · ~124,000 (full repo incl. Swift/Kotlin/C#/Dart/TS bindings, not re-measured this pass)
**Platforms:** iOS 11+ (Neural Engine/Core ML require iOS 14+/16+), Android 5.0+ / API 21+ (NNAPI requires API 27+)
