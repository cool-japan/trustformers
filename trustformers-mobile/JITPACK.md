# JitPack Integration for TrustformeRS Mobile

[![](https://jitpack.io/v/trustformers/trustformers-mobile.svg)](https://jitpack.io/#trustformers/trustformers-mobile)

This document provides instructions for using TrustformeRS Mobile through JitPack.io, a package repository for Git repositories.

## Quick Setup

### Step 1: Add JitPack Repository

Add JitPack repository to your root `build.gradle` file:

```gradle
allprojects {
    repositories {
        google()
        mavenCentral()
        maven { url 'https://jitpack.io' }
    }
}
```

### Step 2: Add Dependency

Add the dependency to your app's `build.gradle` file:

```gradle
dependencies {
    implementation 'com.github.trustformers:trustformers-mobile:Tag'
}
```

Replace `Tag` with the latest release tag or commit hash.

## Available Modules

### Android Library
```gradle
implementation 'com.github.trustformers.trustformers-mobile:android-lib:Tag'
```

### iOS Framework (CocoaPods)
```ruby
pod 'TrustformersKit', :git => 'https://github.com/trustformers/trustformers-mobile.git', :tag => 'Tag'
```

### React Native Plugin
```bash
npm install https://github.com/trustformers/trustformers-mobile.git#Tag
```

### Flutter Plugin
```yaml
dependencies:
  trustformers_flutter:
    git:
      url: https://github.com/trustformers/trustformers-mobile.git
      ref: Tag
      path: flutter-plugin
```

## Version Management

JitPack supports multiple ways to specify versions:

### Latest Release
```gradle
implementation 'com.github.trustformers:trustformers-mobile:latest'
```

### Specific Release Tag
```gradle
implementation 'com.github.trustformers:trustformers-mobile:v1.0.0'
```

### Branch-based (Development)
```gradle
implementation 'com.github.trustformers:trustformers-mobile:main-SNAPSHOT'
```

### Specific Commit
```gradle
implementation 'com.github.trustformers:trustformers-mobile:7c53ed2'
```

## Build Configuration

### Proguard/R8 Rules

Add these rules to your `proguard-rules.pro`:

```proguard
# TrustformeRS Mobile
-keep class com.trustformers.** { *; }
-keepclassmembers class com.trustformers.** { *; }

# Native library
-keep class * {
    native <methods>;
}

# Kotlin coroutines
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}
```

### NDK Configuration

Ensure your app's `build.gradle` includes the required NDK ABIs:

```gradle
android {
    defaultConfig {
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a', 'x86', 'x86_64'
        }
    }
}
```

## Usage Examples

### Basic Initialization

```kotlin
import com.trustformers.TrustformersEngine

class MainActivity : AppCompatActivity() {
    private lateinit var engine: TrustformersEngine
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        // Initialize TrustformeRS
        engine = TrustformersEngine.Builder()
            .setContext(this)
            .setOptimizationLevel(OptimizationLevel.BALANCED)
            .setBackend(Backend.NNAPI)
            .build()
    }
}
```

### Model Loading and Inference

```kotlin
// Load model
engine.loadModel("path/to/model.tflite") { result ->
    when (result) {
        is LoadResult.Success -> {
            // Model loaded successfully
            runInference()
        }
        is LoadResult.Error -> {
            Log.e("TrustformeRS", "Failed to load model: ${result.message}")
        }
    }
}

// Run inference
private fun runInference() {
    val inputTensor = FloatArray(224 * 224 * 3) { Random.nextFloat() }
    
    engine.runInference(inputTensor) { result ->
        when (result) {
            is InferenceResult.Success -> {
                val predictions = result.outputTensors[0]
                processResults(predictions)
            }
            is InferenceResult.Error -> {
                Log.e("TrustformeRS", "Inference failed: ${result.message}")
            }
        }
    }
}
```

## Troubleshooting

### Common Issues

1. **Build Failures**: Ensure you have the latest Android SDK and NDK installed
2. **Native Library Not Found**: Check that NDK ABIs match your target devices
3. **ProGuard Issues**: Add the recommended ProGuard rules
4. **Memory Issues**: Configure appropriate heap sizes for your use case

### Getting Help

- Check the [main documentation](README.md)
- Browse [example applications](examples/)
- Report issues on [GitHub Issues](https://github.com/trustformers/trustformers-mobile/issues)

## Build Status

You can check the build status of any version on JitPack:
https://jitpack.io/#trustformers/trustformers-mobile

## License

TrustformeRS Mobile is released under the MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) before submitting pull requests.