# iOS Quick Start with TrustformeRS Mobile - Video Script

## Video Details
- **Duration**: 8-10 minutes
- **Target Audience**: iOS developers
- **Prerequisites**: Xcode 14+, iOS development basics, Swift knowledge
- **Learning Objectives**: Set up and run TrustformeRS in an iOS app

## Script

### Introduction (0:00 - 0:30)
**[On-screen: Xcode interface with new project]**

**Narrator**: "Welcome to the iOS Quick Start guide for TrustformeRS Mobile! In this tutorial, we'll create a new iOS project, add TrustformeRS as a dependency, and build a simple text classification app that runs completely on-device."

**[Visual: Final demo app showing text input and AI-powered classification]**

### Prerequisites Check (0:30 - 1:00)
**[On-screen: Requirements checklist]**

**Narrator**: "Before we start, make sure you have:"

**Requirements:**
- ✅ Xcode 14 or later
- ✅ iOS 13.0+ deployment target
- ✅ Swift 5.5+ (for async/await support)
- ✅ An iOS device or Simulator

**[Visual: Xcode About dialog showing version, iOS Simulator running]**

### Creating the Xcode Project (1:00 - 2:00)
**[On-screen: Live Xcode project creation]**

**Narrator**: "Let's start by creating a new iOS project. I'll open Xcode and create a new project with SwiftUI."

**Step-by-step:**
1. Open Xcode → Create a new Xcode project
2. Choose iOS → App
3. Product Name: "TrustformersDemo"
4. Interface: SwiftUI
5. Language: Swift
6. Minimum Deployment: iOS 13.0

**[Visual: Screen recording of actual project creation process]**

### Adding TrustformersKit Dependency (2:00 - 3:30)
**[On-screen: Swift Package Manager interface]**

**Narrator**: "Now we'll add TrustformersKit using Swift Package Manager. This is the easiest way to integrate TrustformeRS into your iOS project."

**Swift Package Manager Setup:**
1. File → Add Package Dependency
2. Enter URL: `https://github.com/trustformers/trustformers-mobile`
3. Dependency Rule: Up to Next Major Version
4. Add to Target: TrustformersDemo

**[Visual: Live demonstration of adding the package]**

**Alternative with CocoaPods:**
```ruby
# Podfile
platform :ios, '13.0'
use_frameworks!

target 'TrustformersDemo' do
  pod 'TrustformersKit', '~> 0.1.0'
end
```

### Basic Model Setup (3:30 - 5:00)
**[On-screen: ContentView.swift file in Xcode]**

**Narrator**: "Let's create our inference engine. We'll start with a simple text classification model that can analyze sentiment."

```swift
import SwiftUI
import TrustformersKit

struct ContentView: View {
    @State private var inputText = ""
    @State private var result = ""
    @State private var isLoading = false
    
    // Initialize TrustformeRS engine
    @StateObject private var engine = TextClassificationEngine()
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                Text("TrustformeRS Text Classifier")
                    .font(.title)
                    .padding()
                
                TextEditor(text: $inputText)
                    .border(Color.gray, width: 1)
                    .frame(height: 100)
                    .padding()
                
                Button("Classify Text") {
                    Task {
                        await classifyText()
                    }
                }
                .disabled(inputText.isEmpty || isLoading)
                .buttonStyle(.borderedProminent)
                
                if isLoading {
                    ProgressView("Processing...")
                }
                
                Text(result)
                    .padding()
                    .background(Color.gray.opacity(0.1))
                    .cornerRadius(8)
            }
            .padding()
        }
    }
    
    func classifyText() async {
        isLoading = true
        defer { isLoading = false }
        
        do {
            let prediction = try await engine.classify(text: inputText)
            result = "Sentiment: \(prediction.label)\nConfidence: \(String(format: "%.2f", prediction.confidence))"
        } catch {
            result = "Error: \(error.localizedDescription)"
        }
    }
}
```

### Creating the Engine Class (5:00 - 6:30)
**[On-screen: New Swift file creation]**

**Narrator**: "Now let's create our TextClassificationEngine class that wraps the TrustformeRS inference engine."

```swift
import Foundation
import TrustformersKit

class TextClassificationEngine: ObservableObject {
    private var inferenceEngine: TFKInferenceEngine?
    
    init() {
        setupEngine()
    }
    
    private func setupEngine() {
        // Configure the model
        let config = TFKModelConfig(
            modelName: "distilbert-base-uncased-finetuned-sst-2-english",
            quantization: .fp16,  // Use FP16 for better mobile performance
            backend: .hybrid,     // Automatic Core ML + Metal selection
            maxSequenceLength: 512
        )
        
        do {
            inferenceEngine = try TFKInferenceEngine(config: config)
        } catch {
            print("Failed to initialize engine: \(error)")
        }
    }
    
    struct ClassificationResult {
        let label: String
        let confidence: Double
    }
    
    func classify(text: String) async throws -> ClassificationResult {
        guard let engine = inferenceEngine else {
            throw NSError(domain: "Engine not initialized", code: -1)
        }
        
        let input = TFKTextInput(text: text)
        let output = try await engine.predict(input: input)
        
        // Process the output logits
        let probabilities = softmax(output.logits)
        let maxIndex = probabilities.enumerated().max(by: { $0.element < $1.element })?.offset ?? 0
        
        let labels = ["Negative", "Positive"]
        let label = labels[maxIndex]
        let confidence = probabilities[maxIndex]
        
        return ClassificationResult(label: label, confidence: confidence)
    }
    
    private func softmax(_ values: [Float]) -> [Double] {
        let maxValue = values.max() ?? 0
        let expValues = values.map { exp(Double($0 - maxValue)) }
        let sumExp = expValues.reduce(0, +)
        return expValues.map { $0 / sumExp }
    }
}
```

### Running the App (6:30 - 7:30)
**[On-screen: App running in iOS Simulator]**

**Narrator**: "Let's build and run our app! I'll demonstrate it working in the iOS Simulator first, then on a real device to show the performance difference."

**Demonstration:**
1. Build and run (Cmd+R)
2. Enter sample text: "I love using TrustformeRS on my iPhone!"
3. Tap "Classify Text"
4. Show results appearing
5. Try different text examples

**Performance Notes:**
- Simulator: ~500ms inference time
- iPhone device: ~50ms with Neural Engine
- Memory usage: ~100MB

### Device Performance Comparison (7:30 - 8:30)
**[On-screen: Side-by-side Simulator vs Device performance]**

**Narrator**: "Notice the significant performance improvement on actual hardware. Let's look at why this happens and how to optimize further."

**Performance Factors:**
- **Neural Engine**: 15.8 TOPS on A15 Bionic
- **Core ML Optimization**: Automatic model compilation
- **Metal GPU**: Parallel processing for large models
- **Memory Bandwidth**: Unified memory architecture

**[Visual: Instruments performance monitoring showing CPU, GPU, and Neural Engine usage]**

### Optimization Tips (8:30 - 9:30)
**[On-screen: TFKModelConfig options]**

**Narrator**: "Here are some quick optimization tips for production apps:"

**Configuration Optimizations:**
```swift
let config = TFKModelConfig(
    modelName: "distilbert-base-uncased-finetuned-sst-2-english",
    quantization: .int8,        // Smaller memory footprint
    backend: .coreML,           // Force Core ML for Neural Engine
    cacheStrategy: .aggressive, // Cache compiled models
    batchSize: 1,              // Optimize for single inference
    memoryPoolSize: 64 * 1024 * 1024  // 64MB memory pool
)
```

**Best Practices:**
- Initialize engines early (app startup)
- Use appropriate quantization for your use case
- Monitor memory usage with Instruments
- Test on older devices for compatibility

### Troubleshooting Common Issues (9:30 - 10:00)
**[On-screen: Common error messages and solutions]**

**Narrator**: "If you encounter issues, here are the most common solutions:"

**Common Problems:**
1. **Model loading errors**: Check internet connectivity for first download
2. **Memory warnings**: Reduce batch size or use INT8 quantization
3. **Slow performance**: Ensure you're testing on device, not simulator
4. **Build errors**: Verify iOS deployment target is 13.0+

### Next Steps (10:00 - 10:30)
**[On-screen: Tutorial series roadmap]**

**Narrator**: "Congratulations! You've successfully integrated TrustformeRS into an iOS app. In our next videos, we'll cover Android development, performance optimization, and advanced features like multimodal AI and ARKit integration."

**Upcoming Tutorials:**
- Android Quick Start
- Performance Optimization Deep Dive
- Multimodal AI with Core ML
- ARKit Integration
- Production Deployment Guide

## Code Repository
**[On-screen: GitHub repository link]**

**Narrator**: "The complete source code for this tutorial is available in our GitHub repository. Feel free to clone it, experiment, and let us know your feedback!"

**Repository**: `https://github.com/trustformers/ios-quickstart-demo`

## Supporting Materials

### Complete Project Files

**ContentView.swift**
```swift
// [Complete working ContentView as shown above]
```

**TextClassificationEngine.swift**
```swift
// [Complete engine class as shown above]
```

**Info.plist additions**
```xml
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsArbitraryLoads</key>
    <true/>
</dict>
```

### Visual Assets Needed

1. **Xcode Interface Screenshots**
   - Project creation steps
   - Package Manager interface
   - Code editor with syntax highlighting

2. **App Running Footage**
   - iOS Simulator demonstration
   - Real device performance
   - Instruments profiling

3. **Performance Comparison Charts**
   - Inference time comparison
   - Memory usage graphs
   - Device capability matrix

### Demo Data
**Test Sentences for Classification:**
- "I absolutely love this new feature!"
- "This is the worst app I've ever used."
- "The weather is nice today."
- "TrustformeRS makes mobile AI development so easy!"

## Accessibility Features
- **Screen Reader Support**: All UI elements properly labeled
- **High Contrast**: Code displayed with accessibility-friendly themes
- **Closed Captions**: Full video transcript provided
- **Keyboard Navigation**: All features accessible via keyboard

## Engagement Elements
- **Follow Along**: Encourage viewers to code alongside
- **Checkpoint Questions**: Pause points for understanding check
- **Community Challenge**: Build a different classification model
- **Show and Tell**: Share creations on Discord

## Success Metrics
- **Completion Rate**: Target 70%+ finish rate
- **Code Repository**: 500+ clones in first month
- **Community Posts**: 50+ sharing their modifications
- **Follow-up Questions**: Active Discord discussion

## Production Notes
- **Code Syntax**: Use Xcode's presentation mode for larger fonts
- **Screen Recording**: 60fps for smooth scrolling and typing
- **Audio**: Clear explanation of each code line
- **Pacing**: Allow time for viewers to read and understand code