# trustformers-mobile

Mobile deployment infrastructure for running transformer models on iOS and Android devices with hardware acceleration and cross-platform framework support.

**Version:** 0.2.1 | **Status:** Alpha | **Tests:** ~742 passing (crate) · 18,102 passing workspace-wide (0 failed, 119 skipped) | **SLoC:** ~103,900 (Rust, `src/`) | **Last Updated:** 2026-07-09

## Status

**Alpha**: Core mobile infrastructure — device detection, battery/thermal/network-aware adaptation, OTA model management, quantization, and the mobile performance profiler — is implemented and covered by ~742 passing crate-level tests (0 clippy warnings, 26 doctests passing / 0 failed / 2 ignored). iOS (Core ML/Metal via `TrustformersKit`) and Android (JNI/NNAPI via the `trustformers-android` AAR) native bridges are implemented and exercised by real Swift/Java/Kotlin source. Cross-platform framework integration is uneven: Flutter (`trustformers_flutter`) and Unity (`com.trustformers.mobile`) ship real Dart/C# packages backed by this crate's FFI; React Native currently has Rust-side JSI/Turbo Module bridge code and a detailed usage example, but no packaged npm module lives in this repository yet. A small number of advanced/experimental modules (post-quantum and homomorphic-encryption primitives in `advanced_security.rs`) are simplified reference code rather than production-grade — see [Known Limitations](#known-limitations).

Public API surface: **~3,860 public items** (functions, structs, enums, traits — including impl-block methods) across 187 files in `src/`. No `todo!()`/`unimplemented!()` macros remain in the crate; where simplified/placeholder logic does exist (see below), it returns a working value rather than panicking.

## Features

### iOS Support

- **TrustformersKit**: Native Swift package (`ios-framework/`, SwiftPM + CocoaPods) — `TFKModelConfig`, `TFKInferenceEngine`
- **Core ML Integration**: `coreml.rs` / `coreml_converter.rs` (feature `coreml`) — Neural Engine acceleration
- **Metal Acceleration**: `ios/metal.rs`, `ios/mps.rs` — Metal compute shaders and Metal Performance Shaders
- **ARKit Integration**: `arkit_integration.rs` (compiled for `target_os = "ios"` only) — AR object detection
- **iCloud sync / App extensions**: `ios_icloud.rs`, `ios_app_extensions.rs`, `ios_background.rs`
- **Privacy-First**: on-device processing; see privacy caveats under Known Limitations

### Android Support

- **Native Android Library**: `trustformers-android` AAR (`android-lib/`) — Java `TrustformersEngine` + Kotlin `TrustformersKt` coroutine wrapper and DSL builder
- **NNAPI Integration**: `nnapi.rs` / `nnapi_converter.rs` (feature `nnapi`, Android target only)
- **TFLite NNAPI delegate**: `tflite_nnapi_delegate.rs` (feature `tflite-nnapi`) — source is fully written (real `#[cfg(feature = "tflite-nnapi")]` guards throughout) but the file has no `pub mod` declaration anywhere in `lib.rs`, so it does not currently compile in and the feature gates nothing yet (orphaned; see [Known Limitations](#known-limitations))
- **Edge TPU Support**: `edge_tpu_support.rs` (compiled for `target_os = "android"` only) — Google Coral acceleration
- **Work Manager / Doze / Content Provider / Android Auto**: `android_work_manager.rs`, `android_doze_compatibility.rs`, `android_content_provider.rs`, `android_auto_support.rs`

### Cross-Platform Features

- **Model Management**: `model_management.rs` — OTA downloads, differential updates, signature verification, storage cleanup, cancelable downloads
- **Quantization**: `quantization/` (nibble-packed INT4 + pure-Rust GGUF reader), `optimization/` (enhanced INT4, knowledge distillation, size/kernel/cache optimizers)
- **Battery / Thermal / Network Adaptation**: `battery.rs`, `thermal/` (incl. a predictive throttle model), `network_adaptation/`
- **Memory Management**: `optimization/enhanced_memory_manager.rs`, `optimization/memory_pool.rs`
- **On-Device Training**: `training.rs`, `federated.rs` (both behind the `on-device-training` feature)

### Framework Integration

- **React Native**: Rust-side JSI/Turbo Module bridge (`react_native.rs`, `react_native_turbo.rs`, `react_native_fabric.rs`; features `react-native`/`expo`) plus a complete usage example (`react-native-plugin/example/TrustformersCompleteExample.tsx`). No npm package source ships in this repo yet.
- **Flutter**: `trustformers_flutter` pub package (`flutter-plugin/`, Dart FFI + platform channels) backed by `flutter.rs` (feature `flutter`)
- **Unity**: `com.trustformers.mobile` UPM package (`unity-package/`, `TrustformersEngine` MonoBehaviour) backed by `unity_interop.rs` (feature `unity`)
- **Expo**: config-plugin scaffolding backed by `expo_plugin.rs` (feature `expo`, implies `react-native`)
- **.NET / C#**: standalone P/Invoke wrapper in `csharp-wrapper/` (separate from the Unity package)

## API Overview

Key modules (see `cargo doc -p trustformers-mobile --all-features --open` for the full generated reference):

| Module | Purpose |
|---|---|
| `model_management` | OTA model downloads, differential updates, signature verification, storage/cache management |
| `battery`, `thermal`, `network_adaptation` | Battery-, thermal-, and network-aware adaptive inference |
| `device_info` | Device capability detection (`MobileDeviceDetector::detect()`) |
| `quantization`, `optimization` | INT4/INT8/FP16 quantization, GGUF loading, knowledge distillation, memory/kernel/cache optimizers |
| `ios` (+ `ios/`), `android` (+ `android_*` siblings) | Platform FFI/JNI bindings |
| `coreml`, `nnapi`, `tflite_nnapi_delegate` | Hardware-accelerated inference delegates |
| `federated`, `training`, `differential_privacy` | On-device / federated learning (feature `on-device-training`) |
| `mobile_performance_profiler` | Profiling, bottleneck detection, real-time monitoring, export |
| `react_native`, `flutter`, `unity_interop`, `expo_plugin` | Cross-platform framework bridges |
| `webnn`, `mlx_integration`, `hardware` | WebNN IR/export, an MLX-*style* graph engine for Apple Silicon (Metal + CPU; **does not link Apple's MLX framework**), next-gen accelerator abstractions |
| `advanced_security`, `advanced_privacy_mechanisms` | Experimental PQ-crypto / HE / MPC / DP research code (see Known Limitations) |

## Quick Start

All platform bindings below operate at the token/tensor level (`Tensor`/`float[]`/token IDs in, structured results out) — text tokenization and detokenization is expected to happen in the host app or via `trustformers-tokenizers`, not inside these bindings.

### Rust

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

Battery-aware scheduling (`battery.rs`, always compiled):

```rust
use trustformers_mobile::{MobileBatteryManager, BatteryConfig};
use trustformers_mobile::device_info::MobileDeviceDetector;

let device_info = MobileDeviceDetector::detect()?;
let mut battery_mgr = MobileBatteryManager::new(BatteryConfig::default(), &device_info)?;
battery_mgr.start()?;
let level = battery_mgr.get_current_battery_level();
```

Federated learning (requires `--features on-device-training`; real, compiled implementation lives in `federated.rs`):

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

### iOS (Swift)

```swift
import TrustformersKit

let config = TFKModelConfig.optimizedConfig()
let engine = TFKInferenceEngine(config: config)
let model = try engine.loadModel(at: modelPath, config: config)
let result = engine.performInference(model, input: inputTensor)
```

### Android (Kotlin)

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

### Flutter

```dart
import 'package:trustformers_flutter/trustformers_flutter.dart';

final config = TrustformersConfig(engineId: 'main', modelPath: 'gpt2.bin');
final engine = await TrustformersEngine.create(config);
await engine.loadModel(config.modelPath);

final result = await engine.inference(
  TrustformersInferenceRequest.textGeneration(inputIds: tokenIds),
);
```

### React Native (example only — see Known Limitations)

```typescript
// react-native-plugin/example/TrustformersCompleteExample.tsx
import { TrustformersEngine } from '@trustformers/react-native';

const deviceInfo = await TrustformersEngine.getDeviceInfo();
const engine = await TrustformersEngine.initialize({ enablePerformanceMonitoring: true });
const models = await engine.getAvailableModels();
```

### Unity (C#)

```csharp
// TrustformersEngine is a MonoBehaviour — attach it to a GameObject
var engine = gameObject.AddComponent<TrustformersEngine>();
engine.modelPath = "gpt2.bin";
engine.InitializeEngine();
float[] output = engine.Inference(inputTensor);
```

## Installation

Sub-packages currently version independently (`1.0.0`) and do not track the workspace's `0.2.1` release; verify against each package's own manifest before pinning.

### iOS (CocoaPods)

```ruby
pod 'TrustformersKit', '~> 1.0'
```

### iOS (Swift Package Manager)

```swift
dependencies: [
    .package(url: "https://github.com/cool-japan/trustformers", from: "1.0.0")
]
```

### Android (Gradle)

```gradle
dependencies {
    implementation 'com.trustformers:trustformers-android:1.0.0'
}
```

### Flutter

```yaml
dependencies:
  trustformers_flutter: ^1.0.0
```

### Unity

Add `com.trustformers.mobile` via the Unity Package Manager (Git URL or local `unity-package/` path).

### Rust crate

```toml
[dependencies]
trustformers-mobile = { version = "0.2.1", features = ["on-device-training"] }
```

## Architecture

```
trustformers-mobile/
├── ios-framework/           # iOS Swift package (TrustformersKit; SwiftPM + CocoaPods)
├── android-lib/             # Android AAR (Java TrustformersEngine + Kotlin TrustformersKt wrapper)
├── react-native-plugin/     # React Native bridge example (@trustformers/react-native; example-only)
├── flutter-plugin/          # Flutter package `trustformers_flutter` (Dart FFI + platform channels)
├── unity-package/           # Unity UPM package `com.trustformers.mobile` (C# MonoBehaviour)
├── csharp-wrapper/          # Standalone .NET/C# P/Invoke wrapper
├── codegen/                 # Python binding-generation tooling
├── trustformers-dashboard/  # Python performance-monitoring dashboard
├── tutorials/                # Integration guides and production tutorials
├── src/                     # Shared Rust core (187 files, ~103,900 SLoC)
│   ├── ios.rs, ios/          # iOS FFI bindings (engine, metal, mps)
│   ├── android/, android_*.rs # Android JNI/NNAPI/WorkManager/Doze bindings
│   ├── model_management.rs   # OTA model lifecycle management
│   ├── battery.rs             # Battery-aware optimization
│   ├── federated.rs            # Federated learning (feature `on-device-training`)
│   └── mobile_performance_profiler/ # Profiling/collector/analysis subsystem
├── tests/                   # 11 integration test files
└── examples/
    ├── ios_demo_app/        # SwiftUI demo app
    ├── android_demo_app/    # Jetpack Compose / Kotlin demo activities
    └── *.rs                 # 5 Rust examples (inference, optimization, fine-tuning, platform APIs)
```

## Performance (Illustrative Targets)

`src/benchmarks/performance_targets.rs` defines the crate's actual, code-level performance targets (`PerformanceTargets::default()`):

| Target | Value |
|--------|-------|
| Max inference latency | < 100 ms |
| Max battery drain | < 5% / hour |
| Min device coverage | 90% |
| Max framework size | < 50 MB |

The per-model tables below are illustrative reference figures and are **not** yet backed by an automated on-device CI benchmark in this repository — treat them as design targets, not measured results.

### iOS (illustrative)

| Model | Device | Latency | Memory |
|-------|--------|---------|--------|
| GPT-2 | A15 Neural Engine | 8ms/token | 280MB |
| BERT-base | A15 Neural Engine | 12ms | 350MB |
| LLaMA-2-7B (INT4) | A15 Neural Engine | 45ms/token | 1.2GB |

### Android (illustrative)

| Model | Device | Latency | Memory |
|-------|--------|---------|--------|
| GPT-2 | Snapdragon 8 Gen 2 | 10ms/token | 290MB |
| BERT-base | Snapdragon 8 Gen 2 | 15ms | 360MB |
| LLaMA-2-7B (INT4) | Snapdragon 8 Gen 2 | 52ms/token | 1.3GB |

## Hardware Support

### iOS
- **Base framework**: iOS 11+ / macOS 10.13+ / watchOS 4+ / tvOS 11+ (`Package.swift`, podspec); the crate's own `[package.metadata.ios]` targets deployment target 12.0
- **Neural Engine**: A12+ (iPhone XS and newer) via Core ML (feature `coreml`)
- **Metal**: Metal Performance Shaders path recommended on iOS 14+

### Android
- **Base library**: `minSdkVersion 21` (Android 5.0+), `targetSdk`/`compileSdk` 33 (`android-lib/build.gradle`)
- **NNAPI**: Android 8.1+ (API 27+) — features `nnapi` / `tflite-nnapi`
- **Edge TPU**: devices with Google Coral (`target_os = "android"` only)

## Feature Flags

Verified against `Cargo.toml` and `#[cfg(feature = ...)]` usage in `src/`:

- `coreml` — gates `coreml.rs` / `coreml_converter.rs` (also requires `target_os = "ios"`)
- `nnapi` — gates `nnapi.rs` / `nnapi_converter.rs` (also requires `target_os = "android"`)
- `tflite-nnapi` — implies `nnapi`; declared for `tflite_nnapi_delegate.rs`, but that file currently has no `pub mod` declaration in `lib.rs`, so this flag does not yet gate anything (see [Known Limitations](#known-limitations))
- `on-device-training` — gates `training.rs`, `federated.rs`, `differential_privacy.rs`, `advanced_training.rs`, `advanced_privacy_mechanisms.rs`
- `web` — gates `wasm.rs` (also requires `target_arch = "wasm32"`)
- `react-native` — gates `react_native.rs` / `react_native_turbo.rs` / `react_native_fabric.rs` re-exports
- `flutter` — gates `flutter.rs`
- `unity` — gates `unity_interop.rs`
- `expo` — implies `react-native`; gates `expo_plugin.rs`

## Testing

```bash
# Run Rust tests (crate-level, default features)
cargo test -p trustformers-mobile

# Run with all features (matches the ~742 passing count verified 2026-07-01)
cargo nextest run --all-features -p trustformers-mobile

# Run doctests (26 passing, 0 failed, 2 ignored)
cargo test --doc -p trustformers-mobile --all-features

# CLI tools shipped with the crate
cargo run --bin abi-checker
cargo run --bin trustformers-profiler

# Test iOS framework
cd ios-framework && swift test

# Test Android library
cd android-lib && ./gradlew test
```

## Development

### Building iOS Framework

```bash
./build-ios.sh
# Output: ios-framework/TrustformersKit.xcframework
```

### Building Android AAR

```bash
./build-android.sh
# Output (via Gradle): android-lib/build/outputs/aar/
```

## Known Limitations

- Alpha status: API surface may still change before a Stable designation
- `advanced_security.rs` implements post-quantum KEM (Kyber/McEliece stand-ins), homomorphic encryption, and secure multi-party computation as **simplified/mock reference code**, not audited cryptography — do not depend on it for real confidentiality guarantees yet
- `react-native-plugin/` in this repository contains a usage example (`TrustformersCompleteExample.tsx`) only — there is no `package.json` or module source here, so React Native integration is not yet an installable package from this repo
- Flutter, Unity, iOS, and Android sub-packages version independently at `1.0.0` and do not track the workspace's `0.2.1` release
- `tflite_nnapi_delegate.rs` is fully written but has no `pub mod` declaration anywhere in `lib.rs` — the `tflite-nnapi` Cargo feature currently gates nothing (orphaned, same pattern as the now-fixed `swin`/`deit` in `trustformers-models` before this release, or the now-deleted `android_renderscript.rs` here). Not yet triaged: wire it up or delete it.
- Core ML Neural Engine requires iOS 16+ for latest features
- NNAPI performance varies significantly across Android devices
- Large models require quantization for mobile deployment
- Federated learning requires network connectivity

## Future Enhancements

- Enhanced quantization methods (INT2, further GGUF coverage)
- Better thermal management algorithms
- More AR/VR integrations
- Real-time collaboration features
- WebNN integration for future platforms
- Replace the `advanced_security.rs` placeholder cryptography with audited implementations
- Publish an actual npm package for the React Native bridge

## License

Licensed under Apache License, Version 2.0 ([LICENSE](../LICENSE)).

---

**Last Updated:** 2026-07-09
**Version:** 0.2.1
**Status:** Alpha
**Test Suite:** ~742 crate tests passing · 26 doctests passing (0 failed, 2 ignored)
**SLoC:** ~103,900 (Rust, `src/`) · ~124,000 (full repo incl. Swift/Kotlin/C#/Dart/TS bindings, via tokei)
**Platforms:** iOS 11+ (Neural Engine/Core ML require iOS 14+/16+), Android 5.0+ / API 21+ (NNAPI requires API 27+)
