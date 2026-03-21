# Tutorial 2: iOS Quick Start with TrustformersKit
**Duration**: 9 minutes  
**Target Audience**: iOS developers  
**Prerequisites**: Xcode 14+, Swift 5.5+, basic iOS development

## Video Script

### Opening (0:00 - 0:30)
**[Screen: Xcode welcome screen]**

**Narrator**: "Welcome back! In this tutorial, we're going to build our first iOS app with TrustformersKit. By the end of this video, you'll have a working app that can classify text, generate responses, and run completely offline on your iPhone or iPad."

**[Screen: Final app preview]**

"This is what we'll build today - a simple but powerful AI assistant that works entirely on-device. Let's dive in!"

### Project Setup (0:30 - 2:00)
**[Screen: Xcode new project creation]**

**Narrator**: "First, let's create a new iOS project. Open Xcode and select 'Create a new Xcode project'. Choose 'iOS' and then 'App'."

**[Screen: Project configuration]**

"I'll name this 'TrustformersDemo', set the interface to SwiftUI, and make sure the language is Swift. Click 'Next' and choose where to save your project."

**[Screen: Project navigator]**

"Great! Now we have our basic project structure. The next step is adding TrustformersKit as a dependency."

### Adding TrustformersKit Dependency (2:00 - 3:30)
**[Screen: Package manager interface]**

**Narrator**: "We'll use Swift Package Manager to add TrustformersKit. Go to File menu, then 'Add Package Dependencies'."

**[Screen: Package URL entry]**

"Enter the TrustformersKit repository URL: `https://github.com/trustformers/trustformers-kit-ios`"

**[Screen: Package options]**

"Choose the latest version - I recommend using 'Up to Next Major Version' for stability. Click 'Add Package'."

**[Screen: Target selection]**

"Make sure TrustformersKit is added to your app target, then click 'Add Package' again."

**[Screen: Package dependencies in navigator]**

"Perfect! You can see TrustformersKit is now listed in our Package Dependencies. Now let's start coding!"

### Basic Model Setup (3:30 - 5:00)
**[Screen: ContentView.swift file]**

**Narrator**: "Let's open ContentView.swift and replace the default code with our AI-powered interface."

**[Screen: Code editor with import statement]**

"First, we need to import TrustformersKit at the top of our file:"

```swift
import SwiftUI
import TrustformersKit
```

**[Screen: State variables being added]**

"Now let's add some state variables to manage our AI engine and user input:"

```swift
struct ContentView: View {
    @State private var engine: TFKInferenceEngine?
    @State private var inputText = ""
    @State private var outputText = "Ready to process your text!"
    @State private var isLoading = false
    
    var body: some View {
        // UI code goes here
    }
}
```

**[Screen: Model configuration code]**

"Next, let's create a function to initialize our AI engine:"

```swift
private func setupEngine() {
    Task {
        do {
            let config = TFKModelConfig(
                modelName: "bert-base-uncased",
                quantization: .int8,
                backend: .auto
            )
            
            await MainActor.run {
                self.isLoading = true
            }
            
            self.engine = try await TFKInferenceEngine(config: config)
            
            await MainActor.run {
                self.isLoading = false
                self.outputText = "Model loaded successfully!"
            }
        } catch {
            await MainActor.run {
                self.isLoading = false
                self.outputText = "Error loading model: \(error.localizedDescription)"
            }
        }
    }
}
```

### Building the UI (5:00 - 6:30)
**[Screen: SwiftUI view code]**

**Narrator**: "Now let's build our user interface. We'll create a simple form with text input and output areas:"

```swift
var body: some View {
    NavigationView {
        VStack(spacing: 20) {
            VStack(alignment: .leading, spacing: 8) {
                Text("Input Text")
                    .font(.headline)
                
                TextEditor(text: $inputText)
                    .frame(height: 100)
                    .padding(8)
                    .background(Color(UIColor.systemGray6))
                    .cornerRadius(8)
            }
            
            Button(action: processText) {
                HStack {
                    if isLoading {
                        ProgressView()
                            .scaleEffect(0.8)
                    }
                    Text(isLoading ? "Processing..." : "Analyze Text")
                }
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color.blue)
                .foregroundColor(.white)
                .cornerRadius(8)
            }
            .disabled(isLoading || inputText.isEmpty || engine == nil)
            
            VStack(alignment: .leading, spacing: 8) {
                Text("Output")
                    .font(.headline)
                
                ScrollView {
                    Text(outputText)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .background(Color(UIColor.systemGray6))
                        .cornerRadius(8)
                }
                .frame(height: 150)
            }
            
            Spacer()
        }
        .padding()
        .navigationTitle("TrustformersKit Demo")
        .onAppear {
            setupEngine()
        }
    }
}
```

### Adding Inference Logic (6:30 - 8:00)
**[Screen: processText function]**

**Narrator**: "Now let's implement the text processing function that will actually run our AI model:"

```swift
private func processText() {
    guard let engine = engine else { return }
    
    Task {
        await MainActor.run {
            self.isLoading = true
        }
        
        do {
            let result = try await engine.classify(text: inputText)
            
            await MainActor.run {
                self.outputText = """
                Classification Results:
                
                Label: \(result.label)
                Confidence: \(String(format: "%.2f%%", result.confidence * 100))
                
                Processing Time: \(result.processingTime)ms
                Model: \(result.modelInfo.name)
                Backend: \(result.modelInfo.backend)
                """
                self.isLoading = false
            }
        } catch {
            await MainActor.run {
                self.outputText = "Error: \(error.localizedDescription)"
                self.isLoading = false
            }
        }
    }
}
```

**[Screen: Complete code review]**

"Let's review our complete implementation. We have model loading, a clean UI, and proper error handling - all the essentials for a production app!"

### Testing the App (8:00 - 8:45)
**[Screen: iOS Simulator launch]**

**Narrator**: "Now for the exciting part - let's test our app! I'll run it on the iOS Simulator first."

**[Screen: App running on simulator]**

"The app is launching, and you can see it's loading the BERT model. This might take a few seconds the first time."

**[Screen: Text input and processing]**

"Let me type in some text: 'This movie was absolutely fantastic!' and tap 'Analyze Text'."

**[Screen: Results displaying]**

"Excellent! The model classified this as positive sentiment with 94% confidence. The inference only took 23 milliseconds - that's the power of on-device processing!"

**[Screen: Testing on physical device]**

"Let me also test this on a real iPhone to show you the performance difference..."

### Wrap-up and Next Steps (8:45 - 9:00)
**[Screen: Summary slide]**

**Narrator**: "Congratulations! You've just built your first iOS app with TrustformersKit. In just a few minutes, we've created an app that can perform AI inference completely offline."

**[Screen: Next tutorial preview]**

"In the next tutorial, we'll explore Android development with TrustformeRS, and then we'll dive into performance optimization techniques. Thanks for watching, and I'll see you in the next video!"

## Supporting Materials

### Complete Code Files

#### ContentView.swift
```swift
import SwiftUI
import TrustformersKit

struct ContentView: View {
    @State private var engine: TFKInferenceEngine?
    @State private var inputText = ""
    @State private var outputText = "Ready to process your text!"
    @State private var isLoading = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Input Text")
                        .font(.headline)
                    
                    TextEditor(text: $inputText)
                        .frame(height: 100)
                        .padding(8)
                        .background(Color(UIColor.systemGray6))
                        .cornerRadius(8)
                }
                
                Button(action: processText) {
                    HStack {
                        if isLoading {
                            ProgressView()
                                .scaleEffect(0.8)
                        }
                        Text(isLoading ? "Processing..." : "Analyze Text")
                    }
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(8)
                }
                .disabled(isLoading || inputText.isEmpty || engine == nil)
                
                VStack(alignment: .leading, spacing: 8) {
                    Text("Output")
                        .font(.headline)
                    
                    ScrollView {
                        Text(outputText)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding()
                            .background(Color(UIColor.systemGray6))
                            .cornerRadius(8)
                    }
                    .frame(height: 150)
                }
                
                Spacer()
            }
            .padding()
            .navigationTitle("TrustformersKit Demo")
            .onAppear {
                setupEngine()
            }
        }
    }
    
    private func setupEngine() {
        Task {
            do {
                let config = TFKModelConfig(
                    modelName: "bert-base-uncased",
                    quantization: .int8,
                    backend: .auto
                )
                
                await MainActor.run {
                    self.isLoading = true
                }
                
                self.engine = try await TFKInferenceEngine(config: config)
                
                await MainActor.run {
                    self.isLoading = false
                    self.outputText = "Model loaded successfully!"
                }
            } catch {
                await MainActor.run {
                    self.isLoading = false
                    self.outputText = "Error loading model: \(error.localizedDescription)"
                }
            }
        }
    }
    
    private func processText() {
        guard let engine = engine else { return }
        
        Task {
            await MainActor.run {
                self.isLoading = true
            }
            
            do {
                let result = try await engine.classify(text: inputText)
                
                await MainActor.run {
                    self.outputText = """
                    Classification Results:
                    
                    Label: \(result.label)
                    Confidence: \(String(format: "%.2f%%", result.confidence * 100))
                    
                    Processing Time: \(result.processingTime)ms
                    Model: \(result.modelInfo.name)
                    Backend: \(result.modelInfo.backend)
                    """
                    self.isLoading = false
                }
            } catch {
                await MainActor.run {
                    self.outputText = "Error: \(error.localizedDescription)"
                    self.isLoading = false
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
```

#### Package.swift addition
```swift
dependencies: [
    .package(url: "https://github.com/trustformers/trustformers-kit-ios", from: "1.0.0")
]
```

### Demo Text Examples
1. **Positive**: "This movie was absolutely fantastic!"
2. **Negative**: "The service was terrible and disappointing."
3. **Neutral**: "The weather is cloudy today."
4. **Technical**: "The API endpoint returns JSON data."
5. **Complex**: "While the film had some flaws, the cinematography was breathtaking."

### Key Teaching Points
1. **Async/await patterns** for model loading
2. **Error handling** in Swift
3. **State management** with @State
4. **Task and MainActor** usage
5. **SwiftUI best practices**
6. **Performance considerations**

### Troubleshooting Guide
- **Model loading slow**: Normal on first load, cached afterwards
- **Memory warnings**: Use quantization or smaller models
- **Build errors**: Check iOS deployment target (14.0+)
- **Simulator performance**: Real device much faster
- **Network errors**: Models download once, cached locally

### Performance Tips
- Use `.auto` backend for best performance
- Enable quantization for memory savings
- Test on real devices for accurate performance
- Monitor memory usage with Instruments
- Consider model size vs accuracy tradeoffs