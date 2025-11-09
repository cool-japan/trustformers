# TrustformeRS Swift Bindings

[![Swift](https://img.shields.io/badge/Swift-5.9+-orange.svg)](https://swift.org)
[![Platform](https://img.shields.io/badge/Platform-iOS%2015.0%2B%20%7C%20macOS%2012.0%2B%20%7C%20watchOS%208.0%2B%20%7C%20tvOS%2015.0%2B-lightgrey.svg)](https://developer.apple.com)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Swift bindings for TrustformeRS, providing native iOS and macOS access to state-of-the-art transformer models with optimized performance for Apple platforms.

## Features

- 🚀 **Native Swift API** - Idiomatic Swift interface with async/await support
- 🍎 **Apple Optimized** - Metal and Core ML backend support for maximum performance
- 🔄 **Async/Concurrent** - Full async support with Swift concurrency
- 📱 **Cross-Platform** - iOS, macOS, watchOS, and tvOS support
- ⚡ **High Performance** - Accelerate framework integration for tensor operations
- 🛡️ **Memory Safe** - Automatic resource management with Swift's ARC
- 🎯 **Type Safe** - Comprehensive type safety with Swift's type system

## Supported Tasks

- **Text Generation** - GPT-style text completion and generation
- **Text Classification** - Sentiment analysis, topic classification
- **Question Answering** - Extractive and generative QA
- **Summarization** - Document and text summarization
- **Translation** - Neural machine translation
- **Token Classification** - Named entity recognition, POS tagging
- **Conversational AI** - Multi-turn dialogue systems

## Requirements

- iOS 15.0+ / macOS 12.0+ / watchOS 8.0+ / tvOS 15.0+
- Xcode 14.0+
- Swift 5.9+

## Installation

### Swift Package Manager

Add TrustformeRS to your project using Xcode:

1. File → Add Package Dependencies
2. Enter the repository URL: `https://github.com/cool-japan/trustformers`
3. Select the Swift bindings package

Or add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/cool-japan/trustformers", from: "0.1.0")
],
targets: [
    .target(
        name: "YourTarget",
        dependencies: ["TrustformeRS"]
    )
]
```

### CocoaPods (Coming Soon)

```ruby
pod 'TrustformeRS', '~> 0.1.0'
```

## Quick Start

### Basic Text Generation

```swift
import TrustformeRS

// Initialize TrustformeRS with default configuration
let trustformers = try TrustformeRS(config: .default)

// Create a text generation pipeline
let pipeline = try await trustformers.createPipelineAsync(
    config: .textGeneration(
        modelId: "gpt2",
        temperature: 0.7,
        maxNewTokens: 100
    )
)

// Generate text
let result = try await pipeline.generateAsync(input: "The future of AI is")
print("Generated: \(result)")
```

### Text Classification

```swift
import TrustformeRS

let trustformers = try TrustformeRS(config: .iOS)

// Create sentiment analysis pipeline
let pipeline = try await trustformers.createPipelineAsync(
    config: .textClassification(modelId: "cardiffnlp/twitter-roberta-base-sentiment-latest")
)

// Classify text
let results = try await pipeline.classifyAsync(input: "I love this new iPhone!")
for result in results {
    print("\(result.label): \(result.score)")
}
```

### Question Answering

```swift
import TrustformeRS

let trustformers = try TrustformeRS(config: .macOS)

let pipeline = try await trustformers.createPipelineAsync(
    config: .questionAnswering(modelId: "distilbert-base-cased-distilled-squad")
)

let context = "TrustformeRS is a high-performance Rust library for transformer models."
let question = "What is TrustformeRS?"

let answer = try await pipeline.answerQuestion(question: question, context: context)
print("Answer: \(answer.answer)")
```

## Advanced Usage

### Apple Platform Optimizations

#### Metal Backend (iOS/macOS)

```swift
let trustformers = try TrustformeRS(config: .iOS)

// Enable Metal for GPU acceleration
if TrustformeRS.isMetalAvailable {
    try trustformers.enableMetalBackend()
    print("Metal backend enabled for GPU acceleration")
}
```

#### Core ML Backend

```swift
let trustformers = try TrustformeRS(config: .macOS)

// Enable Core ML for Neural Engine acceleration
if TrustformeRS.isCoreMLAvailable {
    try trustformers.enableCoreMLBackend()
    print("Core ML backend enabled")
}
```

### Streaming Text Generation

```swift
let pipeline = try await trustformers.createPipelineAsync(
    config: .textGeneration(modelId: "gpt2", maxNewTokens: 200)
)

try await pipeline.generateStreaming(input: "Once upon a time") { chunk in
    print(chunk, terminator: "")
    // Update UI with new text chunk
    DispatchQueue.main.async {
        // Update your UI here
    }
}
```

### Batch Processing

```swift
let inputs = [
    "This movie is great!",
    "I hate this product.",
    "The weather is nice today."
]

let results = try await pipeline.classifyBatch(inputs: inputs)
for (input, result) in zip(inputs, results) {
    print("\"\(input)\" -> \(result.first?.label ?? "unknown")")
}
```

### Working with Tensors

```swift
import TrustformeRS

// Create tensors
let tensor1 = Tensor.from2D([[1, 2], [3, 4]])
let tensor2 = Tensor.ones(shape: [2, 2])

// Tensor operations
let sum = try tensor1 + tensor2
let product = try tensor1.matmul(tensor2)

// Reshape and manipulate
let reshaped = try tensor1.reshape([1, 4])
let transposed = try tensor1.transpose()

print("Original: \(tensor1.tensorShape)")
print("Reshaped: \(reshaped.tensorShape)")
print("Transposed: \(transposed.tensorShape)")
```

### Custom Models

```swift
// Load a local model
let modelConfig = Model.Configuration.local(
    path: "/path/to/your/model",
    tokenizerPath: "/path/to/tokenizer"
)

let model = try trustformers.loadModel(config: modelConfig)
let output = try await model.generateAsync(input: "Hello, world!", maxLength: 50)
```

### Working with Tokenizers

```swift
let tokenizer = try trustformers.loadTokenizer(path: "/path/to/tokenizer")

// Tokenize text
let tokens = try tokenizer.encode(text: "Hello, world!")
print("Token IDs: \(tokens.tokenIds)")

// Decode tokens
let decoded = try tokenizer.decode(tokenIds: tokens.tokenIds)
print("Decoded: \(decoded)")

// Batch tokenization
let texts = ["Hello", "world", "how", "are", "you?"]
let batchResult = try tokenizer.encodeBatch(
    texts: texts,
    padding: true,
    maxLength: 10
)
print("Batch shape: \(batchResult.tokenIds.count) x \(batchResult.maxLength)")
```

## Configuration

### TrustformeRS Configuration

```swift
let config = TrustformeRS.Configuration(
    useGPU: true,
    device: "auto",
    numThreads: 4,
    enableLogging: false,
    cacheDir: "/custom/cache/path"
)

let trustformers = try TrustformeRS(config: config)
```

### Platform-Specific Configurations

```swift
// iOS optimized (fewer threads, Metal preferred)
let iOSConfig = TrustformeRS.Configuration.iOS

// macOS optimized (more threads, full GPU utilization)
let macOSConfig = TrustformeRS.Configuration.macOS

// Default configuration (auto-detects platform)
let defaultConfig = TrustformeRS.Configuration.default
```

## Performance Tips

### 1. Use Appropriate Configurations

```swift
// For iOS apps
let config = TrustformeRS.Configuration.iOS
// Automatically sets optimal thread count and enables Metal

// For macOS apps
let config = TrustformeRS.Configuration.macOS
// Uses all available CPU cores and GPU acceleration
```

### 2. Enable Hardware Acceleration

```swift
let trustformers = try TrustformeRS(config: .default)

#if os(iOS) || os(macOS)
// Enable Metal for GPU acceleration
if TrustformeRS.isMetalAvailable {
    try trustformers.enableMetalBackend()
}

// Enable Core ML for Neural Engine
if TrustformeRS.isCoreMLAvailable {
    try trustformers.enableCoreMLBackend()
}
#endif
```

### 3. Use Batch Processing

```swift
// Instead of processing items one by one
let results = try await pipeline.processBatch(inputs: inputs) { input in
    return try await pipeline.classifyAsync(input: input)
}
```

### 4. Optimize Model Selection

```swift
// Use smaller models for mobile devices
#if os(iOS)
let modelId = "distilbert-base-uncased"  // Smaller, faster
#else
let modelId = "bert-large-uncased"       // Larger, more accurate
#endif
```

## Error Handling

```swift
import TrustformeRS

do {
    let trustformers = try TrustformeRS(config: .default)
    let pipeline = try await trustformers.createPipelineAsync(
        config: .textGeneration(modelId: "gpt2")
    )
    let result = try await pipeline.generateAsync(input: "Hello")
    print(result)
} catch TrustformersError.modelNotFound(let message) {
    print("Model not found: \(message)")
} catch TrustformersError.inferenceError(let message) {
    print("Inference error: \(message)")
} catch {
    print("Unexpected error: \(error)")
}
```

## SwiftUI Integration

```swift
import SwiftUI
import TrustformeRS

struct ContentView: View {
    @State private var inputText = ""
    @State private var outputText = ""
    @State private var isGenerating = false
    
    var body: some View {
        VStack {
            TextField("Enter your prompt", text: $inputText)
                .textFieldStyle(RoundedBorderTextFieldStyle())
            
            Button("Generate") {
                generateText()
            }
            .disabled(isGenerating || inputText.isEmpty)
            
            ScrollView {
                Text(outputText)
                    .padding()
            }
        }
        .padding()
    }
    
    private func generateText() {
        isGenerating = true
        outputText = ""
        
        Task {
            do {
                let trustformers = try TrustformeRS(config: .iOS)
                let pipeline = try await trustformers.createPipelineAsync(
                    config: .textGeneration(modelId: "gpt2", maxNewTokens: 100)
                )
                
                try await pipeline.generateStreaming(input: inputText) { chunk in
                    DispatchQueue.main.async {
                        outputText += chunk
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    outputText = "Error: \(error.localizedDescription)"
                }
            }
            
            DispatchQueue.main.async {
                isGenerating = false
            }
        }
    }
}
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Testing

Run tests using Xcode or Swift Package Manager:

```bash
swift test
```

For platform-specific tests:

```bash
# iOS Simulator
swift test --destination 'platform=iOS Simulator,name=iPhone 14'

# macOS
swift test --destination 'platform=macOS'
```

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure the model ID is correct and accessible
2. **Metal not available**: Check device compatibility and iOS version
3. **Memory issues**: Use smaller models or reduce batch sizes
4. **Slow performance**: Enable GPU acceleration and use appropriate thread counts

### Debug Mode

```swift
let config = TrustformeRS.Configuration(
    useGPU: true,
    enableLogging: true  // Enable debug logging
)
```

## Examples

Check the `Examples/` directory for complete sample projects:

- **iOS Text Generator** - Complete iOS app with streaming text generation
- **macOS Chatbot** - Multi-turn conversational AI application
- **Sentiment Analyzer** - Real-time text sentiment analysis
- **Document Summarizer** - Large document summarization tool

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on top of the powerful TrustformeRS Rust library
- Optimized for Apple platforms using Metal and Core ML
- Inspired by Hugging Face Transformers for Python

## Support

- 📚 [Documentation](https://docs.trustformers.ai/swift)
- 🐛 [Issue Tracker](https://github.com/cool-japan/trustformers/issues)
- 💬 [Discussions](https://github.com/cool-japan/trustformers/discussions)
- 📧 [Email Support](mailto:support@trustformers.ai)