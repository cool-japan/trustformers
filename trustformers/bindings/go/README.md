# TrustformeRS Go Bindings

Go bindings for the TrustformeRS machine learning library, providing high-performance transformer model inference in Go applications.

## Features

- **High-Performance Inference**: Leverage Rust's performance from Go
- **Memory Safety**: Automatic resource management with Go finalizers
- **CUDA Support**: Hardware acceleration when available
- **Multiple Tasks**: Text generation, classification, question answering, tokenization
- **Easy Integration**: Simple Go API with comprehensive error handling
- **Cross-Platform**: Works on Linux, macOS, and Windows

## Installation

### Prerequisites

1. **TrustformeRS C Library**: Ensure the TrustformeRS C library is installed and available in your system's library path.

2. **Go 1.19+**: This package requires Go 1.19 or later.

3. **CGO**: CGO must be enabled (it's enabled by default).

### Install

```bash
go get github.com/trustformers/trustformers-go
```

## Quick Start

```go
package main

import (
    "fmt"
    "log"
    
    trustformers "github.com/trustformers/trustformers-go"
)

func main() {
    // Load a model
    model, err := trustformers.NewModel("./models/gpt2")
    if err != nil {
        log.Fatal(err)
    }
    defer model.Close()
    
    // Generate text
    result, err := model.Generate("The future of AI is")
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Println("Generated:", result)
}
```

## API Reference

### Core Types

#### Model

```go
// Load a model
model, err := trustformers.NewModel("path/to/model")
defer model.Close()

// Generate text
text, err := model.Generate("input text")
```

#### Tokenizer

```go
// Load a tokenizer
tokenizer, err := trustformers.NewTokenizer("path/to/tokenizer")
defer tokenizer.Close()

// Encode text to tokens
tokens, err := tokenizer.Encode("Hello, world!")

// Decode tokens to text
text, err := tokenizer.Decode(tokens)
```

#### Pipeline

```go
// Create a pipeline
pipeline, err := trustformers.NewPipeline("text-classification", "model-name")
defer pipeline.Close()

// Make predictions
result, err := pipeline.Predict("input text")
```

### Convenience Functions

```go
// Quick text generation
result, err := trustformers.GenerateText("model/path", "input")

// Quick text classification
result, err := trustformers.ClassifyText("model-name", "text to classify")

// Quick question answering
answer, err := trustformers.AnswerQuestion("qa-model", "context", "question")

// Quick tokenization
tokens, err := trustformers.TokenizeText("tokenizer/path", "text")
```

### Hardware Information

```go
// Check CUDA availability
if trustformers.IsCudaAvailable() {
    fmt.Println("CUDA is available")
    fmt.Printf("CUDA devices: %d\n", trustformers.GetCudaDeviceCount())
}

// Get device information
devices := trustformers.GetDeviceInfo()
for _, device := range devices {
    fmt.Printf("Device %d: %s\n", device.ID, device.Name)
}
```

## Error Handling

The bindings provide typed error handling:

```go
if err != nil {
    if trustformersErr, ok := err.(*trustformers.Error); ok {
        switch trustformersErr.Code {
        case trustformers.ModelLoadFailed:
            // Handle model loading error
        case trustformers.InferenceFailed:
            // Handle inference error
        case trustformers.OutOfMemory:
            // Handle memory error
        case trustformers.CudaError:
            // Handle CUDA error
        }
    }
}
```

## Examples

### Text Generation

```go
func generateText() error {
    model, err := trustformers.NewModel("./models/gpt2")
    if err != nil {
        return err
    }
    defer model.Close()
    
    prompts := []string{
        "The future of technology is",
        "In a world where AI",
        "Once upon a time",
    }
    
    for _, prompt := range prompts {
        result, err := model.Generate(prompt)
        if err != nil {
            log.Printf("Error generating for '%s': %v", prompt, err)
            continue
        }
        fmt.Printf("Prompt: %s\nGenerated: %s\n\n", prompt, result)
    }
    
    return nil
}
```

### Sentiment Analysis

```go
func analyzeSentiment() error {
    pipeline, err := trustformers.NewPipeline("text-classification", "sentiment-analysis")
    if err != nil {
        return err
    }
    defer pipeline.Close()
    
    texts := []string{
        "I love this product!",
        "This is disappointing.",
        "Pretty good overall.",
    }
    
    for _, text := range texts {
        sentiment, err := pipeline.Predict(text)
        if err != nil {
            log.Printf("Error analyzing '%s': %v", text, err)
            continue
        }
        fmt.Printf("Text: %s -> Sentiment: %s\n", text, sentiment)
    }
    
    return nil
}
```

### Question Answering

```go
func answerQuestions() error {
    pipeline, err := trustformers.NewPipeline("question-answering", "distilbert-base-cased-distilled-squad")
    if err != nil {
        return err
    }
    defer pipeline.Close()
    
    context := "TrustformeRS is a machine learning library written in Rust that provides high-performance transformer implementations."
    questions := []string{
        "What is TrustformeRS?",
        "What language is it written in?",
    }
    
    for _, question := range questions {
        input := fmt.Sprintf("context: %s question: %s", context, question)
        answer, err := pipeline.Predict(input)
        if err != nil {
            log.Printf("Error answering '%s': %v", question, err)
            continue
        }
        fmt.Printf("Q: %s\nA: %s\n\n", question, answer)
    }
    
    return nil
}
```

## Configuration

### Environment Variables

- `TRUSTFORMERS_MODEL_PATH`: Default path for models
- `TRUSTFORMERS_TOKENIZER_PATH`: Default path for tokenizers

### Build Configuration

When building applications that use these bindings, you may need to specify the library path:

```bash
# If TrustformeRS is installed in a custom location
export CGO_LDFLAGS="-L/path/to/trustformers/lib -ltrust_transformers_c"
go build
```

## Performance Tips

1. **Reuse Models**: Keep models loaded for multiple inferences rather than loading/unloading repeatedly.

2. **Batch Processing**: Process multiple inputs together when possible.

3. **Resource Management**: Always call `Close()` on models, tokenizers, and pipelines, or use `defer`.

4. **CUDA**: Enable CUDA for GPU acceleration when available.

5. **Memory Monitoring**: Monitor memory usage in long-running applications.

## Thread Safety

- **Models, Tokenizers, Pipelines**: Safe for concurrent read operations
- **Resource Management**: Each instance should be closed only once
- **CUDA**: CUDA operations are thread-safe within the same context

## Troubleshooting

### Common Issues

1. **Library Not Found**:
   ```
   error: ld: library not found for -ltrust_transformers_c
   ```
   Solution: Ensure the TrustformeRS C library is installed and in your library path.

2. **Model Loading Fails**:
   Check that the model path is correct and the model files are accessible.

3. **CUDA Errors**:
   Verify CUDA installation and driver compatibility.

### Debug Information

Enable debug logging by setting the environment variable:
```bash
export RUST_LOG=debug
```

## Building from Source

```bash
# Clone the repository
git clone https://github.com/trustformers/trustformers-go.git
cd trustformers-go

# Run tests
go test ./...

# Build example
cd example
go build -o example main.go
./example
```

## License

This project is licensed under the same license as the TrustformeRS project.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions:
- GitHub Issues: [https://github.com/trustformers/trustformers-go/issues](https://github.com/trustformers/trustformers-go/issues)
- Documentation: [https://docs.trustformers.dev](https://docs.trustformers.dev)