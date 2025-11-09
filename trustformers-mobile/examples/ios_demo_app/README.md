# TrustformersRS iOS Demo App

A comprehensive iOS demo application showcasing the capabilities of TrustformersRS mobile inference framework.

## Features

### 🧠 Text Classification
- Real-time sentiment analysis
- BERT-based text understanding
- Core ML optimized inference
- Performance metrics tracking

### 👁️ Computer Vision
- Object detection using YOLO models
- Real-time image analysis
- Metal-accelerated processing
- Bounding box visualization

### 🌐 Augmented Reality
- ARKit integration with AI inference
- Real-time object detection in AR
- Pose estimation and tracking
- Plane detection and classification
- World map persistence

### 📊 Performance Monitoring
- Real-time device metrics
- CPU, GPU, and memory usage
- Thermal state monitoring
- Battery impact tracking
- Model performance analysis

## Requirements

- iOS 15.0+
- Xcode 14.0+
- Device with A12 Bionic chip or newer
- ARKit support (iPhone 6s or newer)
- 2GB+ available storage

## Installation

### Option 1: Xcode Project
1. Open `TrustformersDemo.xcodeproj` in Xcode
2. Select your development team
3. Build and run on device

### Option 2: Swift Package Manager
1. Add TrustformersKit as a dependency:
```swift
dependencies: [
    .package(url: "https://github.com/trustformers/trustformers-ios", from: "1.0.0")
]
```

### Option 3: CocoaPods
1. Add to your `Podfile`:
```ruby
pod 'TrustformersKit', '~> 1.0'
```

## Architecture

### Core Components

- **TrustformersKit**: Main inference framework
- **Core ML Integration**: Neural Engine acceleration
- **Metal Compute**: GPU-accelerated operations
- **ARKit Bridge**: AR-based AI inference

### Performance Optimizations

- **Adaptive Quality**: Dynamic quality scaling based on device performance
- **Thermal Management**: Automatic throttling to prevent overheating
- **Memory Pool**: Efficient memory management for large models
- **Batch Processing**: Optimized batch inference for multiple inputs

## Usage Examples

### Text Classification
```swift
let engine = InferenceEngine()
engine.classifyText("This is amazing!") { result in
    print("Sentiment: \(result.predictions.first?.label)")
}
```

### Object Detection
```swift
let detector = ObjectDetector()
detector.detect(in: image) { detections in
    for detection in detections {
        print("Found: \(detection.className) at \(detection.bbox)")
    }
}
```

### ARKit Integration
```swift
let arEngine = ARKitEngine()
arEngine.startSession()
arEngine.processFrame(frame) { result in
    print("Detected \(result.detections.count) objects in AR")
}
```

## Model Information

### Included Models
- **BERT-Base**: Text classification and understanding (500MB)
- **GPT-2**: Text generation (1.2GB)
- **YOLOv5**: Object detection (300MB)
- **PoseNet**: Human pose estimation (200MB)

### Custom Models
The demo supports loading custom Core ML models:
```swift
let config = InferenceConfig()
config.customModelPath = "path/to/your/model.mlmodel"
```

## Performance Benchmarks

### iPhone 14 Pro (A16 Bionic)
- Text Classification: ~15ms
- Object Detection: ~25ms
- AR Processing: ~30ms
- Memory Usage: <200MB

### iPhone 12 (A14 Bionic)
- Text Classification: ~20ms
- Object Detection: ~35ms
- AR Processing: ~45ms
- Memory Usage: <300MB

## Privacy & Security

- All inference runs on-device
- No data is sent to external servers
- Models are encrypted at rest
- User consent required for camera/microphone access

## Troubleshooting

### Common Issues

**App crashes on launch**
- Ensure device has sufficient memory (>1GB free)
- Check iOS version compatibility
- Verify app permissions are granted

**Poor performance**
- Close other apps to free memory
- Enable Low Power Mode optimization
- Reduce model quality in settings

**AR features not working**
- Ensure ARKit is supported on device
- Check camera permissions
- Verify adequate lighting conditions

### Debug Information

Enable debug logging:
```swift
TrustformersKit.setLogLevel(.debug)
```

View performance metrics:
```swift
let stats = PerformanceMonitor.shared.currentStats
print("CPU: \(stats.cpuUsage)%, Memory: \(stats.memoryMB)MB")
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

This demo app is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📧 Email: support@trustformers.ai
- 💬 Discord: [TrustformersRS Community](https://discord.gg/trustformers)
- 📖 Documentation: [docs.trustformers.ai](https://docs.trustformers.ai)
- 🐛 Issues: [GitHub Issues](https://github.com/trustformers/trustformers-mobile/issues)