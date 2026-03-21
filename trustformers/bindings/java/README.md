# TrustFormeRS Java Bindings

[![Maven Central](https://img.shields.io/maven-central/v/ai.trustformers/trustformers-java)](https://search.maven.org/artifact/ai.trustformers/trustformers-java)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Java](https://img.shields.io/badge/java-11%2B-orange.svg)](https://adoptopenjdk.net/)

Java bindings for the TrustFormeRS transformer library, providing high-performance machine learning capabilities through JNI.

## Features

- 🚀 **High Performance**: Native implementation with JNI for optimal speed
- 🔧 **Easy to Use**: Simple, intuitive Java API
- 🌐 **Hub Integration**: Direct access to Hugging Face models
- 🎯 **Multiple Tasks**: Text generation, classification, Q&A, summarization, and more
- 💾 **Memory Efficient**: Automatic resource management with try-with-resources
- 🖥️ **Cross Platform**: Support for Windows, macOS, and Linux
- ⚡ **GPU Acceleration**: CUDA and Metal support for hardware acceleration

## Installation

### Maven

Add this dependency to your `pom.xml`:

```xml
<dependency>
    <groupId>ai.trustformers</groupId>
    <artifactId>trustformers-java</artifactId>
    <version>0.1.0</version>
</dependency>
```

### Gradle

Add this to your `build.gradle`:

```gradle
implementation 'ai.trustformers:trustformers-java:0.1.0'
```

### Manual Installation

1. Download the JAR file from the [releases page](https://github.com/cool-japan/trustformers/releases)
2. Add it to your classpath
3. Ensure the native library is available in your system PATH or java.library.path

## Quick Start

```java
import ai.trustformers.*;

public class QuickStart {
    public static void main(String[] args) {
        try {
            // Check system capabilities
            TrustFormeRS.printSystemInfo();
            
            // Create a text generation pipeline
            try (Pipeline pipeline = Pipeline.fromHub(Pipeline.TEXT_GENERATION, "gpt2")) {
                String prompt = "The future of artificial intelligence is";
                String generated = pipeline.generateText(prompt, 50, 0.7);
                System.out.println("Generated: " + generated);
            }
            
        } catch (TrustFormeRSException e) {
            System.err.println("Error: " + e.getUserFriendlyMessage());
        }
    }
}
```

## API Reference

### Core Classes

- **`TrustFormeRS`**: Main entry point with system information and capabilities
- **`Model`**: Direct model access for custom operations
- **`Tokenizer`**: Text tokenization and encoding/decoding
- **`Pipeline`**: High-level task-specific workflows
- **`TrustFormeRSException`**: Exception handling with detailed error information

### TrustFormeRS Class

```java
// System information
String version = TrustFormeRS.getVersion();
boolean cudaAvailable = TrustFormeRS.isCudaAvailable();
boolean metalAvailable = TrustFormeRS.isMetalAvailable();
int deviceCount = TrustFormeRS.getDeviceCount();
Map<Integer, String> devices = TrustFormeRS.getAllDeviceInfo();
```

### Model Class

```java
// Load model from local path
try (Model model = new Model("/path/to/model")) {
    // Use model
}

// Load from Hugging Face Hub
try (Model model = Model.fromHub("gpt2")) {
    // Generate text
    String text = model.generateText("Hello", config);
    
    // Get embeddings
    float[] embeddings = model.getEmbeddings("Hello world");
    
    // Get model info
    String info = model.getModelInfo();
    Map<String, Object> config = model.getModelConfig();
}
```

### Tokenizer Class

```java
try (Tokenizer tokenizer = Tokenizer.fromHub("gpt2")) {
    // Encode text to token IDs
    int[] tokens = tokenizer.encode("Hello, world!", true);
    
    // Decode token IDs to text
    String text = tokenizer.decode(tokens, true);
    
    // Tokenize to string tokens
    String[] tokenStrings = tokenizer.tokenize("Hello, world!");
    
    // Batch operations
    int[] batchTokens = tokenizer.encodeBatch(texts, true, 128);
    String[] batchTexts = tokenizer.decodeBatch(tokenIds, true);
    
    // Get vocabulary size
    int vocabSize = tokenizer.getVocabSize();
}
```

### Pipeline Class

```java
// Create pipeline from Hub
try (Pipeline pipeline = Pipeline.fromHub(Pipeline.TEXT_GENERATION, "gpt2")) {
    // Text generation
    String generated = pipeline.generateText("Once upon a time", 100, 0.8);
    
    // Custom configuration
    Map<String, Object> config = new HashMap<>();
    config.put("max_length", 50);
    config.put("temperature", 0.7);
    Map<String, Object> result = pipeline.process("Hello", config);
    
    // Batch processing
    String[] inputs = {"Text 1", "Text 2", "Text 3"};
    List<Map<String, Object>> results = pipeline.processBatch(inputs);
}

// Text classification
try (Pipeline pipeline = Pipeline.fromHub(Pipeline.TEXT_CLASSIFICATION, "distilbert-base-uncased-finetuned-sst-2-english")) {
    List<Map<String, Object>> results = pipeline.classifyText("I love this!");
    for (Map<String, Object> result : results) {
        System.out.println(result.get("label") + ": " + result.get("score"));
    }
}

// Question answering
try (Pipeline pipeline = Pipeline.fromHub(Pipeline.QUESTION_ANSWERING, "distilbert-base-cased-distilled-squad")) {
    Map<String, Object> answer = pipeline.answerQuestion(
        "What is the capital of France?",
        "France is a country in Europe. Its capital is Paris."
    );
    System.out.println("Answer: " + answer.get("answer"));
    System.out.println("Score: " + answer.get("score"));
}

// Summarization
try (Pipeline pipeline = Pipeline.fromHub(Pipeline.SUMMARIZATION, "t5-small")) {
    String summary = pipeline.summarizeText(longText, 100, 20);
    System.out.println("Summary: " + summary);
}
```

## Supported Pipeline Tasks

- **`TEXT_GENERATION`**: Generate text continuations
- **`TEXT_CLASSIFICATION`**: Classify text into categories
- **`QUESTION_ANSWERING`**: Answer questions based on context
- **`SUMMARIZATION`**: Generate text summaries
- **`TRANSLATION`**: Translate between languages
- **`FILL_MASK`**: Fill masked tokens in text
- **`TOKEN_CLASSIFICATION`**: Classify individual tokens (NER, POS)
- **`CONVERSATIONAL`**: Multi-turn dialogue systems
- **`TEXT_TO_SPEECH`**: Convert text to speech
- **`SPEECH_TO_TEXT`**: Convert speech to text
- **`IMAGE_TO_TEXT`**: Generate captions for images
- **`VISUAL_QUESTION_ANSWERING`**: Answer questions about images

## Advanced Usage

### Custom Configuration

```java
Map<String, Object> config = new HashMap<>();
config.put("max_length", 100);
config.put("temperature", 0.8);
config.put("top_p", 0.9);
config.put("top_k", 50);
config.put("do_sample", true);
config.put("num_return_sequences", 3);

try (Pipeline pipeline = Pipeline.fromHub(Pipeline.TEXT_GENERATION, "gpt2")) {
    Map<String, Object> result = pipeline.process("Hello", config);
}
```

### Batch Processing

```java
String[] inputs = {
    "Positive review: This product is amazing!",
    "Negative review: I hate this product.",
    "Neutral review: It's okay, nothing special."
};

try (Pipeline pipeline = Pipeline.fromHub(Pipeline.TEXT_CLASSIFICATION, "sentiment-model")) {
    List<Map<String, Object>> results = pipeline.processBatch(inputs);
    
    for (int i = 0; i < inputs.length; i++) {
        System.out.println("Input: " + inputs[i]);
        Map<String, Object> result = results.get(i);
        System.out.println("Result: " + result);
    }
}
```

### Error Handling

```java
try (Model model = Model.fromHub("gpt2")) {
    String result = model.generateText("Hello");
    System.out.println(result);
    
} catch (TrustFormeRSException e) {
    System.err.println("Error occurred: " + e.getUserFriendlyMessage());
    
    if (e.hasErrorCode()) {
        System.err.println("Error code: " + e.getErrorCode());
    }
    
    if (e.hasSuggestion()) {
        System.err.println("Suggestion: " + e.getSuggestion());
    }
}
```

### Resource Management

The library uses automatic resource management. Always use try-with-resources:

```java
// ✅ Correct - automatic cleanup
try (Model model = Model.fromHub("gpt2")) {
    // Use model
} // Model is automatically closed here

// ❌ Incorrect - manual cleanup required
Model model = Model.fromHub("gpt2");
// Use model
model.close(); // Must remember to call this
```

## Building from Source

### Prerequisites

- Java 11 or later
- Maven 3.6 or later
- Rust toolchain (for native library)
- Git

### Build Steps

```bash
# Clone the repository
git clone https://github.com/cool-japan/trustformers.git
cd trustformers/trustformers/bindings/java

# Build native library and Java bindings
mvn clean compile

# Run tests
mvn test

# Create JAR
mvn package

# Install to local repository
mvn install
```

### Build Profiles

- **`dev`** (default): Development build with tests
- **`release`**: Production build with signing
- **`skip-native`**: Skip native library build (for CI)

```bash
# Release build
mvn clean package -Prelease

# Skip native build
mvn clean compile -Pskip-native
```

## Platform Support

| Platform | Architecture | Status | Notes |
|----------|-------------|--------|-------|
| Linux | x86_64 | ✅ Supported | Full CUDA support |
| Linux | aarch64 | ✅ Supported | CPU and GPU support |
| macOS | x86_64 | ✅ Supported | Intel Macs |
| macOS | aarch64 | ✅ Supported | Apple Silicon, Metal support |
| Windows | x86_64 | ✅ Supported | CUDA support |

## Performance Tips

1. **Use Batch Processing**: Process multiple inputs together for better throughput
2. **GPU Acceleration**: Enable CUDA or Metal for faster inference
3. **Resource Management**: Use try-with-resources to avoid memory leaks
4. **Model Caching**: Reuse models and pipelines instead of recreating them
5. **Configuration Tuning**: Adjust generation parameters for your use case

## Examples

See the [examples directory](examples/) for complete working examples:

- [Basic Usage](examples/TrustFormeRSExample.java) - Comprehensive API demonstration
- [Text Generation](examples/TextGenerationExample.java) - Advanced text generation
- [Text Classification](examples/ClassificationExample.java) - Sentiment analysis and more
- [Question Answering](examples/QuestionAnsweringExample.java) - QA system implementation
- [Batch Processing](examples/BatchProcessingExample.java) - High-throughput processing

## Troubleshooting

### Native Library Loading Issues

If you encounter `UnsatisfiedLinkError`:

1. Check that the native library is in your PATH or java.library.path
2. Verify your platform is supported
3. Try downloading the library manually from releases

### Memory Issues

- Use try-with-resources for automatic cleanup
- Don't hold references to closed models/pipelines
- Monitor memory usage with Java profiling tools

### Model Loading Issues

- Check network connectivity for Hub models
- Verify model names and revisions
- Check available disk space for model downloads

### Performance Issues

- Enable GPU acceleration if available
- Use batch processing for multiple inputs
- Tune generation parameters (temperature, top_p, etc.)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](../../LICENSE) file for details.

## Contributing

We welcome contributions! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.

## Support

- 📚 [Documentation](https://trustformers.ai/docs)
- 💬 [Discussions](https://github.com/cool-japan/trustformers/discussions)
- 🐛 [Issues](https://github.com/cool-japan/trustformers/issues)
- 📧 Email: support@trustformers.ai