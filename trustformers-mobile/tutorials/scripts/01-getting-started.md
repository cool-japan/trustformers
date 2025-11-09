# Tutorial Script: Getting Started with TrustformeRS Mobile

**Duration**: 5 minutes  
**Target Audience**: Mobile developers new to TrustformeRS  
**Prerequisites**: Basic iOS/Android development knowledge  

## Script Overview

This tutorial introduces developers to TrustformeRS mobile deployment, covering installation, basic setup, and running your first on-device inference example.

---

## Opening (0:00 - 0:30)

**[Visual: TrustformeRS logo and mobile devices]**

**Narrator**: 
"Welcome to TrustformeRS Mobile! I'm excited to show you how to bring state-of-the-art transformer models directly to iOS and Android devices. In just 5 minutes, you'll learn how to set up TrustformeRS and run your first on-device AI inference."

**[Visual: Split screen showing iPhone and Android device]**

"TrustformeRS Mobile enables privacy-preserving, lightning-fast inference without internet connectivity. Let's get started!"

---

## Section 1: What is TrustformeRS Mobile? (0:30 - 1:30)

**[Visual: Architecture diagram showing on-device inference]**

**Narrator**: 
"TrustformeRS Mobile is a comprehensive framework that brings transformer models to mobile devices. Unlike cloud-based solutions, everything runs locally on your users' devices."

**[Visual: Feature highlights with icons]**

"Key features include:
- **Native iOS and Android support** with Swift and Kotlin APIs
- **Hardware acceleration** using Core ML, Metal, NNAPI, and Vulkan  
- **Model optimization** with automatic quantization and compression
- **Privacy-first design** - no data leaves the device
- **Production-ready** with memory management and battery optimization"

**[Visual: Performance comparison chart]**

"You get enterprise-grade performance while maintaining complete data privacy."

---

## Section 2: Installation and Setup (1:30 - 2:45)

**[Visual: IDE screen showing project setup]**

**Narrator**: 
"Let's set up TrustformeRS in your mobile project. I'll show you both iOS and Android setup."

### iOS Setup

**[Visual: Xcode with Swift Package Manager]**

"For iOS, open Xcode and add TrustformersKit via Swift Package Manager. In your project settings, click 'Package Dependencies' and add:"

**[Text overlay: https://github.com/trustformers/trustformers-mobile]**

```swift
// Add to your imports
import TrustformersKit

// Initialize the engine
let engine = TFKInferenceEngine()
```

### Android Setup

**[Visual: Android Studio with Gradle files]**

"For Android, add TrustformeRS to your `build.gradle`:"

**[Code display]**
```kotlin
dependencies {
    implementation 'com.trustformers:trustformers-mobile:1.0.0'
}
```

```kotlin
// Initialize in your Activity or Fragment
val engine = TrustformersEngine(this)
```

**[Visual: Dependency sync in progress]**

"Sync your project, and you're ready to go!"

---

## Section 3: Your First Inference (2:45 - 4:15)

**[Visual: Code editor with syntax highlighting]**

**Narrator**: 
"Now let's run your first on-device inference. We'll use a pre-trained BERT model for text classification."

### iOS Example

**[Visual: Swift code being typed]**

```swift
class ViewController: UIViewController {
    let engine = TFKInferenceEngine()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        runInference()
    }
    
    func runInference() {
        // Load a pre-trained model
        engine.loadModel("bert-base-uncased") { result in
            switch result {
            case .success:
                self.classifyText()
            case .failure(let error):
                print("Model loading failed: \(error)")
            }
        }
    }
    
    func classifyText() {
        let text = "TrustformeRS makes mobile AI easy!"
        
        engine.classify(text: text) { result in
            switch result {
            case .success(let classification):
                print("Sentiment: \(classification.label)")
                print("Confidence: \(classification.confidence)")
            case .failure(let error):
                print("Classification failed: \(error)")
            }
        }
    }
}
```

### Android Example

**[Visual: Kotlin code being typed]**

```kotlin
class MainActivity : AppCompatActivity() {
    private lateinit var engine: TrustformersEngine
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        engine = TrustformersEngine(this)
        runInference()
    }
    
    private fun runInference() {
        // Load a pre-trained model
        engine.loadModel("bert-base-uncased") { result ->
            when (result) {
                is Result.Success -> classifyText()
                is Result.Error -> Log.e("AI", "Model loading failed: ${result.error}")
            }
        }
    }
    
    private fun classifyText() {
        val text = "TrustformeRS makes mobile AI easy!"
        
        engine.classify(text) { result ->
            when (result) {
                is Result.Success -> {
                    Log.i("AI", "Sentiment: ${result.data.label}")
                    Log.i("AI", "Confidence: ${result.data.confidence}")
                }
                is Result.Error -> Log.e("AI", "Classification failed: ${result.error}")
            }
        }
    }
}
```

**[Visual: App running on device/simulator]**

"Run your app, and you'll see the classification results in your console. The model runs entirely on-device!"

---

## Section 4: Performance and Optimization (4:15 - 4:45)

**[Visual: Performance monitoring dashboard]**

**Narrator**: 
"TrustformeRS automatically optimizes for your target devices. Let's see what's happening under the hood."

**[Visual: Code showing configuration options]**

```swift
// iOS: Enable hardware acceleration
let config = TFKModelConfig()
config.useNeuralEngine = true
config.useMetal = true
config.precision = .fp16

engine.loadModel("bert-base-uncased", config: config)
```

```kotlin
// Android: Configure hardware acceleration
val config = TrustformersConfig.Builder()
    .enableNNAPI(true)
    .enableGPU(true)
    .setPrecision(Precision.FP16)
    .build()

engine.loadModel("bert-base-uncased", config)
```

**[Visual: Performance metrics overlay]**

"This enables Core ML Neural Engine on iOS and NNAPI on Android for maximum performance while maintaining accuracy."

---

## Conclusion and Next Steps (4:45 - 5:00)

**[Visual: Next tutorial previews]**

**Narrator**: 
"Congratulations! You've just run your first on-device AI inference with TrustformeRS. In our next tutorials, we'll dive deeper into:"

**[Text overlay with bullet points]**
- Advanced model optimization techniques
- Custom model deployment
- Production best practices
- Performance profiling and debugging

**[Visual: TrustformeRS community links]**

"Check out the documentation, join our community Discord, and don't forget to star us on GitHub. Thanks for watching, and happy coding!"

---

## Technical Notes

### Recording Instructions
1. **Screen Setup**: Use 1920x1080 resolution with high-contrast IDE theme
2. **Code Typing**: Type at moderate pace (2-3 characters per second)
3. **Device Recording**: Use device screen recording for demo sections
4. **Audio**: Clear, professional narration with consistent pacing

### Required Assets
- TrustformeRS logo and branding assets
- IDE screenshots and recordings (Xcode, Android Studio)
- Device recordings (iPhone, Android)
- Performance dashboard mockups
- Community resource graphics

### Post-Production
- Add closed captions for accessibility
- Include chapter markers at section breaks
- Add animated callouts for important code sections
- Include download links in video description
- Create thumbnail with clear branding

### Code Repository
All code examples should be available in a companion GitHub repository:
- `/ios-getting-started/` - Complete iOS project
- `/android-getting-started/` - Complete Android project
- `README.md` - Setup instructions and requirements
- `requirements.txt` - Dependencies and versions

### Testing Checklist
- [ ] Code examples compile and run successfully
- [ ] All dependencies are available and version-pinned
- [ ] Performance claims are accurate and measurable
- [ ] Accessibility features work correctly
- [ ] Links and resources are valid and up-to-date

---

*Tutorial Script Version: 1.0*  
*Last Updated: 2025-07-16*  
*Estimated Recording Time: 6-8 hours (including setup and editing)*