# TrustformeRS Java Bindings

High-performance Java JNI bindings for the TrustformeRS-C library, providing native Java access to advanced transformer models and natural language processing capabilities.

## Features

- **Complete C API Coverage**: Full access to all TrustformeRS-C functionality through JNI
- **Type Safe**: Strong typing with comprehensive exception handling
- **Memory Safe**: Automatic resource management with finalizers and explicit cleanup
- **Performance Optimized**: Efficient JNI bindings with minimal overhead
- **Thread Safe**: Safe for use in multi-threaded applications
- **Cross-Platform**: Support for Windows, macOS, and Linux

## Requirements

- **Java 11+**: Java 11 or later
- **TrustformeRS-C Library**: Build and install the TrustformeRS-C library first
- **Maven 3.6+**: For building the Java bindings

## Installation

### Building from Source

1. **Build the TrustformeRS-C Library**:
   ```bash
   cd /path/to/trustformers-c
   cargo build --release
   ```

2. **Build the Java Bindings**:
   ```bash
   cd java
   mvn clean compile
   ```

3. **Run Tests**:
   ```bash
   mvn test
   ```

4. **Package JAR**:
   ```bash
   mvn package
   ```

### Using as a Dependency

Add to your `pom.xml`:

```xml
<dependency>
    <groupId>com.trustformers</groupId>
    <artifactId>trustformers-java</artifactId>
    <version>0.1.0-alpha.1</version>
</dependency>
```

## Quick Start

### Basic Usage

```java
import com.trustformers.*;

public class Example {
    public static void main(String[] args) {
        try {
            // Initialize TrustformeRS
            TrustformeRS trustformers = new TrustformeRS();
            
            // Display version and features
            System.out.println("Version: " + trustformers.getVersion());
            System.out.println("GPU Support: " + trustformers.hasFeature("gpu"));
            
            // Get memory usage
            TrustformeRS.MemoryUsage memUsage = trustformers.getMemoryUsage();
            System.out.println("Memory Usage: " + memUsage);
            
            // Cleanup
            trustformers.close();
            
        } catch (TrustformersException e) {
            System.err.println("Error: " + e.getMessage());
        }
    }
}
```

### Text Generation

```java
// Load model and tokenizer
Model model = trustformers.loadModelFromHub("gpt2");
Tokenizer tokenizer = trustformers.loadTokenizerFromHub("gpt2");

// Create pipeline
Pipeline pipeline = Pipeline.createTextGeneration(trustformers, model, tokenizer);

// Generate text
String prompt = "The future of AI is";
String generated = pipeline.generateText(prompt);
System.out.println("Generated: " + generated);

// Advanced generation with options
Pipeline.GenerationOptions options = new Pipeline.GenerationOptions()
    .maxLength(100)
    .temperature(0.8)
    .topK(50)
    .doSample(true);

String advancedGenerated = pipeline.generateText(prompt, options);
System.out.println("Advanced: " + advancedGenerated);

// Cleanup
pipeline.close();
tokenizer.close();
model.close();
```

### Text Classification

```java
// Load classification model
Model model = trustformers.loadModelFromHub("distilbert-base-uncased-finetuned-sst-2-english");
Tokenizer tokenizer = trustformers.loadTokenizerFromHub("distilbert-base-uncased-finetuned-sst-2-english");

// Create pipeline
Pipeline pipeline = Pipeline.createTextClassification(trustformers, model, tokenizer);

// Classify text
String text = "I love this product!";
Pipeline.ClassificationResult[] results = pipeline.classifyText(text);

for (Pipeline.ClassificationResult result : results) {
    System.out.println("Label: " + result.label + ", Score: " + result.score);
}

// Batch classification
String[] texts = {"Great!", "Terrible...", "Okay product"};
Pipeline.ClassificationResult[][] batchResults = pipeline.classifyTextBatch(texts);

// Cleanup
pipeline.close();
tokenizer.close();
model.close();
```

### Question Answering

```java
// Load QA model
Model model = trustformers.loadModelFromHub("bert-large-uncased-whole-word-masking-finetuned-squad");
Tokenizer tokenizer = trustformers.loadTokenizerFromHub("bert-large-uncased-whole-word-masking-finetuned-squad");

// Create pipeline
Pipeline pipeline = Pipeline.createQuestionAnswering(trustformers, model, tokenizer);

// Answer question
String context = "TrustformeRS is a high-performance transformer library written in Rust.";
String question = "What language is TrustformeRS written in?";

Pipeline.AnswerResult answer = pipeline.answerQuestion(context, question);
System.out.println("Answer: " + answer.answer);
System.out.println("Score: " + answer.score);

// Cleanup
pipeline.close();
tokenizer.close();
model.close();
```

### Conversational AI

```java
// Load conversational model
Model model = trustformers.loadModelFromHub("microsoft/DialoGPT-medium");
Tokenizer tokenizer = trustformers.loadTokenizerFromHub("microsoft/DialoGPT-medium");

// Create pipeline
Pipeline pipeline = Pipeline.createConversational(trustformers, model, tokenizer);

// Have a conversation
String userInput = "Hello! How are you?";
String botResponse = pipeline.addConversationTurn(userInput);
System.out.println("Bot: " + botResponse);

// Get conversation history
Pipeline.ConversationTurn[] history = pipeline.getConversationHistory();
for (Pipeline.ConversationTurn turn : history) {
    System.out.println("User: " + turn.userInput);
    System.out.println("Bot: " + turn.botResponse);
}

// Clear conversation
pipeline.clearConversation();

// Cleanup
pipeline.close();
tokenizer.close();
model.close();
```

## API Reference

### Core Classes

#### `TrustformeRS`
Main library interface for initialization and global operations.

**Methods:**
- `TrustformeRS()` - Create and initialize new instance
- `getVersion()` - Get library version
- `getBuildInfo()` - Get build information
- `hasFeature(String feature)` - Check feature availability
- `setLogLevel(LogLevel level)` - Set logging level
- `loadModelFromHub(String modelName)` - Load model from Hugging Face Hub
- `loadModelFromPath(String modelPath)` - Load model from local path
- `loadTokenizerFromHub(String modelName)` - Load tokenizer from Hub
- `loadTokenizerFromPath(String path)` - Load tokenizer from local path

#### `Model`
Represents a loaded transformer model.

**Methods:**
- `getInfo()` - Get model information
- `setQuantization(int bits)` - Set quantization level
- `validate()` - Validate model integrity
- `getMetadata()` - Get model metadata
- `close()` - Release resources

#### `Tokenizer`
Text tokenization interface.

**Methods:**
- `encode(String text)` - Encode text to tokens
- `decode(int[] tokens)` - Decode tokens to text
- `encodeBatch(String[] texts)` - Batch encoding
- `decodeBatch(int[][] tokenBatches)` - Batch decoding
- `getVocabSize()` - Get vocabulary size
- `getSpecialTokens()` - Get special tokens
- `close()` - Release resources

#### `Pipeline`
High-level interface for NLP tasks.

**Static Factory Methods:**
- `createTextGeneration(trustformers, model, tokenizer)`
- `createTextClassification(trustformers, model, tokenizer)`
- `createQuestionAnswering(trustformers, model, tokenizer)`
- `createConversational(trustformers, model, tokenizer)`

**Methods:**
- `generateText(String prompt)` - Generate text
- `generateText(String prompt, GenerationOptions options)` - Generate with options
- `classifyText(String text)` - Classify text
- `classifyTextBatch(String[] texts)` - Batch classification
- `answerQuestion(String context, String question)` - Answer question
- `addConversationTurn(String userInput)` - Add conversation turn
- `getConversationHistory()` - Get conversation history
- `clearConversation()` - Clear conversation

### Configuration Classes

#### `OptimizationConfig`
Performance optimization settings.

```java
TrustformeRS.OptimizationConfig config = new TrustformeRS.OptimizationConfig();
config.enableSIMD = true;
config.cacheSizeMB = 512;
config.numThreads = 4;
trustformers.applyOptimizations(config);
```

#### `GenerationOptions`
Text generation configuration.

```java
Pipeline.GenerationOptions options = new Pipeline.GenerationOptions()
    .maxLength(100)
    .temperature(0.8)
    .topK(50)
    .topP(0.9)
    .doSample(true);
```

## Memory Management

The Java bindings provide automatic memory management through finalizers, but explicit cleanup is recommended:

```java
// Automatic cleanup (not recommended for production)
Model model = trustformers.loadModelFromHub("gpt2");
// model will be cleaned up by finalizer

// Explicit cleanup (recommended)
Model model = trustformers.loadModelFromHub("gpt2");
try {
    // use model
} finally {
    model.close(); // immediate cleanup
}

// Or use try-with-resources
try (Model model = trustformers.loadModelFromHub("gpt2")) {
    // use model
} // automatically closed
```

### Memory Monitoring

```java
// Basic memory usage
TrustformeRS.MemoryUsage memUsage = trustformers.getMemoryUsage();
System.out.println("Total: " + memUsage.totalMemoryBytes);

// Advanced memory statistics
JsonNode advancedUsage = trustformers.getAdvancedMemoryUsage();
System.out.println("Advanced: " + advancedUsage);

// Memory leak detection
JsonNode leakReport = trustformers.checkMemoryLeaks();
System.out.println("Leaks: " + leakReport);

// Set memory limits
trustformers.setMemoryLimits(2048, 1536); // 2GB max, warning at 1.5GB

// Force cleanup
trustformers.memoryCleanup();
```

## Performance Optimization

### SIMD and Caching

```java
TrustformeRS.OptimizationConfig config = TrustformeRS.OptimizationConfig.defaultConfig();
config.enableSIMD = true;
config.enableCaching = true;
config.cacheSizeMB = 1024;
trustformers.applyOptimizations(config);
```

### Performance Profiling

```java
// Start profiling
trustformers.startProfiling();

// ... perform operations ...

// Stop and get report
JsonNode report = trustformers.stopProfiling();
System.out.println("Performance Report: " + report);
```

### Batch Processing

```java
// Batch operations are more efficient
String[] texts = {"text1", "text2", "text3"};
Pipeline.ClassificationResult[][] results = pipeline.classifyTextBatch(texts);
```

## Error Handling

The bindings provide comprehensive error handling through `TrustformersException`:

```java
try {
    Model model = trustformers.loadModelFromHub("invalid-model");
} catch (TrustformersException e) {
    switch (e.getErrorCode()) {
        case RUNTIME_ERROR:
            System.err.println("Runtime error: " + e.getMessage());
            break;
        case INVALID_PARAMETER:
            System.err.println("Invalid parameter: " + e.getMessage());
            break;
        default:
            System.err.println("Unknown error: " + e.getMessage());
    }
}
```

## Threading and Concurrency

The Java bindings are thread-safe and can be used in multi-threaded applications:

```java
// Multiple threads can safely use the same TrustformeRS instance
TrustformeRS trustformers = new TrustformeRS();

// Each thread should have its own model/tokenizer/pipeline instances
ExecutorService executor = Executors.newFixedThreadPool(4);

for (String text : texts) {
    executor.submit(() -> {
        try (Model model = trustformers.loadModelFromHub("gpt2");
             Tokenizer tokenizer = trustformers.loadTokenizerFromHub("gpt2");
             Pipeline pipeline = Pipeline.createTextGeneration(trustformers, model, tokenizer)) {
            
            String result = pipeline.generateText(text);
            System.out.println("Result: " + result);
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    });
}
```

## Examples

Complete examples are available in the `src/main/java/examples/` directory:

- `BasicUsageExample.java` - Comprehensive usage demonstration
- `TextGenerationExample.java` - Advanced text generation
- `ClassificationExample.java` - Text classification and analysis
- `QuestionAnsweringExample.java` - Question answering systems
- `ConversationalExample.java` - Chatbot implementation

### Running Examples

```bash
# Compile and run basic example
mvn compile exec:java -Dexec.mainClass="examples.BasicUsageExample"

# Run with specific model
mvn compile exec:java -Dexec.mainClass="examples.TextGenerationExample" -Dexec.args="gpt2"
```

## Building and Deployment

### Development Build

```bash
# Clean and compile
mvn clean compile

# Run tests
mvn test

# Package JAR
mvn package

# Install to local repository
mvn install
```

### Production Build

```bash
# Create distribution with native libraries
mvn clean package -Pproduction

# Create uber JAR with all dependencies
mvn clean package -Puber-jar
```

### Native Library Packaging

The Java bindings automatically load the native library from:

1. System library path (recommended for production)
2. JAR resources (for distribution)
3. Relative paths (for development)

To package native libraries in the JAR:

```bash
# Copy native libraries to resources
mkdir -p src/main/resources/native/linux-x86_64
cp ../target/release/libtrustformers_c.so src/main/resources/native/linux-x86_64/

mkdir -p src/main/resources/native/macos-x86_64
cp ../target/release/libtrustformers_c.dylib src/main/resources/native/macos-x86_64/

mkdir -p src/main/resources/native/windows-x86_64
cp ../target/release/trustformers_c.dll src/main/resources/native/windows-x86_64/

# Build JAR with native libraries
mvn package
```

## Troubleshooting

### Library Loading Issues

```bash
# Set library path environment variable
export TRUSTFORMERS_C_LIB_PATH="/path/to/trustformers-c/target/release"

# Or use Java system property
java -Djava.library.path="/path/to/native/libs" -jar your-app.jar
```

### JNI Compilation Issues

```bash
# Ensure Java headers are available
export JAVA_HOME=/path/to/java

# Generate JNI headers
javac -h target/native-headers src/main/java/com/trustformers/*.java
```

### Performance Issues

1. **Enable optimizations**: Use `OptimizationConfig` to enable SIMD and caching
2. **Batch operations**: Use batch methods for multiple inputs
3. **Memory management**: Call `close()` explicitly on resources
4. **JVM tuning**: Configure appropriate heap size and GC settings

### Memory Issues

```bash
# Increase heap size
java -Xmx4g -jar your-app.jar

# Enable memory monitoring
java -XX:+PrintGCDetails -XX:+PrintMemoryUsage -jar your-app.jar
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all examples work
5. Submit a pull request

## License

This project is licensed under the same terms as the main TrustformeRS project.

## Support

For issues and questions:
- GitHub Issues: [Repository Issues](https://github.com/trustformers/trustformers-c/issues)
- Documentation: [TrustformeRS Docs](https://trustformers.github.io/docs/)
- Java API Docs: Run `mvn javadoc:javadoc` to generate API documentation