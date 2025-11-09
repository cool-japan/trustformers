# TrustformeRS Mobile Tutorial Sample Projects

This directory contains complete, runnable sample projects that accompany our video tutorial series. Each project demonstrates key concepts and provides a starting point for your own development.

## Quick Start Projects

### 1. iOS Sentiment Analysis (`ios-sentiment-demo/`)
**Companion to**: iOS Quick Start Tutorial
**Description**: A SwiftUI app that performs real-time sentiment analysis using TrustformersKit.

**Key Features:**
- ✅ SwiftUI interface with Material Design principles
- ✅ TrustformersKit integration with Core ML acceleration
- ✅ Real-time text classification
- ✅ Performance monitoring and error handling
- ✅ Async/await pattern implementation

**Requirements:**
- iOS 13.0+
- Xcode 14+
- Swift 5.5+

**To Run:**
```bash
cd ios-sentiment-demo
open TrustformersSentimentDemo.xcodeproj
# Build and run (⌘+R)
```

### 2. Android Sentiment Analysis (`android-sentiment-demo/`)
**Companion to**: Android Quick Start Tutorial
**Description**: A Jetpack Compose app with NNAPI hardware acceleration for sentiment analysis.

**Key Features:**
- ✅ Modern Jetpack Compose UI
- ✅ MVVM architecture with ViewModel
- ✅ NNAPI hardware acceleration
- ✅ Material Design 3 theming
- ✅ Coroutines for asynchronous operations

**Requirements:**
- Android API 21+
- Android Studio Hedgehog+
- Kotlin 1.8+

**To Run:**
```bash
cd android-sentiment-demo
# Open in Android Studio, sync, and run
```

## Cross-Platform Projects

### 3. React Native Text Classifier (`react-native-classifier/`)
**Companion to**: React Native Integration Tutorial
**Description**: Cross-platform text classification with TypeScript and Turbo Modules.

**Key Features:**
- ✅ TypeScript integration
- ✅ Turbo Module performance optimization
- ✅ Cross-platform iOS/Android compatibility
- ✅ Native UI components
- ✅ Performance benchmarking

### 4. Flutter AI Assistant (`flutter-ai-assistant/`)
**Companion to**: Flutter Integration Tutorial
**Description**: Multi-functional AI assistant using Flutter and Dart FFI.

**Key Features:**
- ✅ Multiple AI capabilities (text, translation, summarization)
- ✅ Dart FFI for native performance
- ✅ Platform-specific optimizations
- ✅ Material/Cupertino adaptive design
- ✅ State management with Provider

### 5. Unity AR Classifier (`unity-ar-classifier/`)
**Companion to**: Unity Integration Tutorial
**Description**: AR application that classifies real-world objects using Unity and AR Foundation.

**Key Features:**
- ✅ AR Foundation integration
- ✅ Real-time object detection and classification
- ✅ C# bindings for TrustformeRS
- ✅ Cross-platform AR (iOS ARKit, Android ARCore)
- ✅ IL2CPP compatibility

## Advanced Features Projects

### 6. iOS Multimodal App (`ios-multimodal-demo/`)
**Companion to**: Advanced Features Deep Dive Tutorial
**Description**: Demonstrates image captioning, visual Q&A, and multimodal AI capabilities.

**Key Features:**
- ✅ Camera integration for real-time analysis
- ✅ Image captioning with BLIP-2 model
- ✅ Visual question answering
- ✅ Core ML + Metal hybrid execution
- ✅ ARKit integration for spatial understanding

### 7. Android Edge TPU Demo (`android-edge-tpu-demo/`)
**Companion to**: Android Advanced Features Tutorial
**Description**: Showcases Google Edge TPU integration for high-performance inference.

**Key Features:**
- ✅ Edge TPU hardware detection and utilization
- ✅ High-performance computer vision tasks
- ✅ Coral USB Accelerator support
- ✅ Benchmark comparison (CPU vs GPU vs TPU)
- ✅ Camera2 API integration

## Performance Optimization Projects

### 8. iOS Performance Benchmark (`ios-performance-benchmark/`)
**Companion to**: Performance Optimization Tutorial
**Description**: Comprehensive benchmarking suite for iOS performance testing.

**Key Features:**
- ✅ Multi-backend performance comparison
- ✅ Memory usage profiling
- ✅ Battery impact measurement
- ✅ Model quantization comparison
- ✅ Instruments integration

### 9. Android Performance Monitor (`android-performance-monitor/`)
**Companion to**: Performance Optimization Tutorial
**Description**: Real-time performance monitoring and optimization recommendations.

**Key Features:**
- ✅ Real-time performance metrics
- ✅ Hardware utilization monitoring
- ✅ Automatic optimization suggestions
- ✅ Export performance reports
- ✅ Integration with Android Studio Profiler

## Production-Ready Projects

### 10. Enterprise Chat App (`enterprise-chat-app/`)
**Companion to**: Production Deployment Tutorial
**Description**: Production-ready chat application with on-device AI capabilities.

**Key Features:**
- ✅ End-to-end encryption
- ✅ Smart reply suggestions
- ✅ Real-time translation
- ✅ Sentiment analysis
- ✅ MDM integration
- ✅ Privacy compliance (GDPR, CCPA)

## Project Structure Standards

Each sample project follows these conventions:

```
project-name/
├── README.md                 # Project-specific documentation
├── SETUP.md                  # Quick setup instructions
├── PERFORMANCE.md            # Performance benchmarks
├── src/                      # Source code
├── docs/                     # Additional documentation
├── tests/                    # Unit and integration tests
├── benchmarks/               # Performance benchmarks
└── assets/                   # Images, models, resources
```

## Common Setup Requirements

### iOS Projects
```bash
# Install dependencies
brew install cocoapods
pod install

# Or use Swift Package Manager (preferred)
# Dependencies automatically resolved in Xcode
```

### Android Projects
```bash
# Ensure Android SDK is installed
# Open in Android Studio and sync Gradle files
# Dependencies automatically downloaded
```

### React Native Projects
```bash
npm install
# iOS
cd ios && pod install && cd ..
npx react-native run-ios

# Android
npx react-native run-android
```

### Flutter Projects
```bash
flutter pub get
flutter run
```

### Unity Projects
```bash
# Open in Unity Hub
# Import TrustformeRS Unity Package
# Build for target platform
```

## Performance Baselines

### Benchmark Results (Reference Device: iPhone 13 Pro, Pixel 6 Pro)

| Model | Platform | Backend | Latency | Memory | Battery/Hour |
|-------|----------|---------|---------|---------|--------------|
| BERT-Base | iOS | Core ML | 45ms | 120MB | 2% |
| BERT-Base | Android | NNAPI | 52ms | 125MB | 2.5% |
| GPT-2 Small | iOS | Metal | 180ms | 340MB | 8% |
| GPT-2 Small | Android | GPU | 195ms | 350MB | 9% |
| DistilBERT | iOS | Hybrid | 25ms | 85MB | 1% |
| DistilBERT | Android | Edge TPU | 15ms | 90MB | 0.5% |

## Troubleshooting Guide

### Common Issues Across Projects

#### Model Loading Failures
```
Error: Failed to load model from URL
Solution: Check internet connectivity, verify model name
```

#### Memory Issues
```
Error: Out of memory during inference
Solution: Reduce batch size, use INT8 quantization
```

#### Performance Issues
```
Problem: Slow inference on device
Solution: Ensure hardware acceleration is enabled
```

#### Build Errors
```
Error: Missing TrustformeRS dependency
Solution: Verify package manager setup and sync
```

### Platform-Specific Issues

#### iOS
- **Code Signing**: Ensure development team is set
- **Privacy Permissions**: Check Info.plist for required permissions
- **Deployment Target**: Verify iOS 13.0+ minimum

#### Android
- **API Level**: Ensure minSdk 21 or higher
- **Permissions**: Check AndroidManifest.xml
- **Hardware Features**: Verify device capabilities

## Contributing

We welcome contributions to improve these sample projects!

### Guidelines
1. **Follow platform conventions** (SwiftUI for iOS, Jetpack Compose for Android)
2. **Include comprehensive documentation** in README files
3. **Add performance benchmarks** for new features
4. **Test on multiple devices** and API levels
5. **Follow accessibility best practices**

### Submission Process
1. Fork the repository
2. Create a feature branch
3. Test thoroughly on target platforms
4. Submit pull request with detailed description
5. Include benchmark results if applicable

## Support and Community

- **Documentation**: [docs.trustformers.ai](https://docs.trustformers.ai)
- **GitHub Issues**: Report bugs and request features
- **Discord Community**: [discord.gg/trustformers](https://discord.gg/trustformers)
- **Stack Overflow**: Tag questions with `trustformers`

## License

All sample projects are licensed under MIT License. See individual project LICENSE files for details.

## Acknowledgments

These sample projects are created and maintained by the TrustformeRS community. Special thanks to all contributors who have helped improve the examples and documentation.

---

**Next Steps**: Choose a sample project that matches your target platform and use case. Follow the setup instructions and start building amazing AI-powered mobile applications with TrustformeRS!