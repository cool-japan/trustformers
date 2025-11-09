# TrustFormeRS Ruby Bindings

[![Ruby](https://img.shields.io/badge/Ruby-3.0+-red.svg)](https://ruby-lang.org)
[![Gem Version](https://badge.fury.io/rb/trustformers.svg)](https://badge.fury.io/rb/trustformers)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Ruby bindings for TrustFormeRS, providing high-performance transformer models with an idiomatic Ruby interface.

## Features

- 🚀 **High Performance** - Built on TrustFormeRS Rust core for maximum speed
- 🎯 **Type Safe** - Comprehensive error handling and type checking
- 🔄 **Async Support** - Non-blocking operations for better concurrency
- 📦 **Easy Integration** - Simple Ruby gem installation
- 🛠️ **Flexible Configuration** - Extensive customization options
- 🔧 **Memory Efficient** - Optimized resource management
- 🌐 **Cross Platform** - Linux, macOS, and Windows support

## Supported Tasks

- **Text Generation** - GPT-style text completion and generation
- **Text Classification** - Sentiment analysis, topic classification
- **Question Answering** - Extractive and generative QA
- **Summarization** - Document and text summarization
- **Translation** - Neural machine translation
- **Token Classification** - Named entity recognition, POS tagging
- **Conversational AI** - Multi-turn dialogue systems
- **Feature Extraction** - Text embeddings and representations

## Requirements

- Ruby 3.0+ (recommended: Ruby 3.2+)
- TrustFormeRS native library
- Linux, macOS, or Windows

## Installation

### Via RubyGems

```ruby
gem install trustformers
```

### Via Bundler

Add to your `Gemfile`:

```ruby
gem 'trustformers', '~> 0.1.0'
```

Then run:

```bash
bundle install
```

### Building from Source

```bash
git clone https://github.com/cool-japan/trustformers.git
cd trustformers/bindings/ruby
bundle install
rake compile
```

## Quick Start

### Basic Text Generation

```ruby
require 'trustformers'

# Initialize TrustFormeRS with default configuration
trustformers = TrustFormeRS.new

# Create a text generation pipeline
pipeline = trustformers.create_pipeline(
  task: :text_generation,
  model_id: "gpt2",
  max_new_tokens: 100,
  temperature: 0.7
)

# Generate text
result = pipeline.generate("The future of AI is")
puts result.generated_text
```

### Text Classification

```ruby
require 'trustformers'

trustformers = TrustFormeRS.new

# Create sentiment analysis pipeline
pipeline = trustformers.create_pipeline(
  task: :text_classification,
  model_id: "cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# Classify text
results = pipeline.classify("I love this new Ruby gem!")
results.each do |result|
  puts "#{result.label}: #{result.score}"
end
```

### Question Answering

```ruby
require 'trustformers'

trustformers = TrustFormeRS.new

pipeline = trustformers.create_pipeline(
  task: :question_answering,
  model_id: "distilbert-base-cased-distilled-squad"
)

context = "TrustFormeRS is a high-performance Rust library for transformer models."
question = "What is TrustFormeRS?"

answer = pipeline.answer_question(question: question, context: context)
puts "Answer: #{answer.answer}"
```

## Configuration

### Basic Configuration

```ruby
# CPU-only configuration
config = TrustFormeRS::Configuration.cpu

# GPU configuration (if available)
config = TrustFormeRS::Configuration.gpu

# Custom configuration
config = TrustFormeRS::Configuration.new(
  use_gpu: true,
  device: "auto",
  num_threads: 4,
  enable_logging: false,
  cache_dir: "/path/to/cache"
)

trustformers = TrustFormeRS.new(config)
```

### Environment-Specific Configurations

```ruby
# Development configuration with logging
config = TrustFormeRS::Configuration.development

# Production configuration with optimizations
config = TrustFormeRS::Configuration.production

# Custom configuration for specific needs
config = TrustFormeRS::Configuration.new(
  use_gpu: TrustFormeRS.gpu_available?,
  device: "auto", 
  num_threads: -1, # Auto-detect
  enable_logging: Rails.env.development?,
  cache_dir: Rails.root.join("tmp", "trustformers_cache")
)
```

## Advanced Usage

### Working with Tensors

```ruby
require 'trustformers'

# Create tensors
tensor1 = TrustFormeRS::Tensor.from_2d([[1, 2], [3, 4]])
tensor2 = TrustFormeRS::Tensor.ones([2, 2])

# Tensor operations
sum = tensor1 + tensor2
product = tensor1.matmul(tensor2)

# Reshape and manipulate
reshaped = tensor1.reshape([1, 4])
transposed = tensor1.transpose

puts "Original: #{tensor1.shape}"
puts "Reshaped: #{reshaped.shape}"
puts "Transposed: #{transposed.shape}"
```

### Custom Models

```ruby
# Load a local model
model = trustformers.load_model("/path/to/your/model")
output = model.generate("Hello, world!", max_new_tokens: 50)

# Load with specific configuration
model = trustformers.load_model(
  "bert-base-uncased",
  device: "gpu",
  precision: "float16"
)
```

### Working with Tokenizers

```ruby
tokenizer = trustformers.load_tokenizer("bert-base-uncased")

# Tokenize text
result = tokenizer.encode("Hello, world!")
puts "Token IDs: #{result.token_ids}"

# Decode tokens
decoded = tokenizer.decode(result.token_ids)
puts "Decoded: #{decoded}"

# Batch tokenization
texts = ["Hello", "world", "how", "are", "you?"]
batch_result = tokenizer.encode_batch(
  texts,
  padding: true,
  max_length: 10
)
puts "Batch shape: #{batch_result.token_ids.size} x #{batch_result.max_length}"
```

### Batch Processing

```ruby
inputs = [
  "This movie is great!",
  "I hate this product.",
  "The weather is nice today."
]

# Process batch with automatic batching
results = pipeline.process_batch(inputs, batch_size: 8)

# Manual batch processing
results = inputs.map { |input| pipeline.classify(input) }
```

### Streaming Generation

```ruby
pipeline = trustformers.create_pipeline(
  task: :text_generation,
  model_id: "gpt2",
  max_new_tokens: 200
)

# Stream generation with block
pipeline.generate_streaming("Once upon a time") do |chunk|
  print chunk
  $stdout.flush
end
```

## Error Handling

```ruby
require 'trustformers'

begin
  trustformers = TrustFormeRS.new
  pipeline = trustformers.create_pipeline(
    task: :text_generation,
    model_id: "gpt2"
  )
  result = pipeline.generate("Hello")
  puts result.generated_text

rescue TrustFormeRS::ModelNotFoundError => e
  puts "Model not found: #{e.message}"
rescue TrustFormeRS::InferenceError => e
  puts "Inference error: #{e.message}"
rescue TrustFormeRS::InitializationError => e
  puts "Initialization failed: #{e.message}"
rescue => e
  puts "Unexpected error: #{e.message}"
end
```

## Rails Integration

### Gemfile

```ruby
gem 'trustformers', '~> 0.1.0'
```

### Configuration

```ruby
# config/initializers/trustformers.rb
TrustFormeRS.configure do |config|
  config.use_gpu = Rails.env.production?
  config.device = "auto"
  config.cache_dir = Rails.root.join("tmp", "trustformers_cache")
  config.enable_logging = Rails.env.development?
end
```

### Service Class

```ruby
# app/services/text_generation_service.rb
class TextGenerationService
  def initialize
    @trustformers = TrustFormeRS.new(
      TrustFormeRS::Configuration.production
    )
    @pipeline = @trustformers.create_pipeline(
      task: :text_generation,
      model_id: "gpt2"
    )
  end

  def generate(prompt, **options)
    result = @pipeline.generate(prompt, **options)
    result.generated_text
  rescue => e
    Rails.logger.error "Text generation failed: #{e.message}"
    nil
  end
end
```

### Controller Usage

```ruby
# app/controllers/api/text_controller.rb
class Api::TextController < ApplicationController
  def generate
    service = TextGenerationService.new
    result = service.generate(params[:prompt])
    
    if result
      render json: { text: result }
    else
      render json: { error: "Generation failed" }, status: 422
    end
  end
end
```

## Performance Tips

### 1. Use Appropriate Configurations

```ruby
# For CPU-intensive applications
config = TrustFormeRS::Configuration.cpu
config.num_threads = 8 # Use all available cores

# For GPU applications
config = TrustFormeRS::Configuration.gpu("auto")
config.num_threads = 4 # Fewer threads when using GPU
```

### 2. Enable Hardware Acceleration

```ruby
trustformers = TrustFormeRS.new

# Enable GPU if available
if TrustFormeRS.gpu_available?
  trustformers.enable_gpu("auto")
  puts "GPU acceleration enabled"
end
```

### 3. Use Batch Processing

```ruby
# Instead of processing items one by one
inputs = ["text1", "text2", "text3", ...]
results = pipeline.process_batch(inputs, batch_size: 16)

# Or use built-in batch methods
results = pipeline.classify(inputs) # Automatically batched
```

### 4. Optimize Model Selection

```ruby
# Choose models based on your requirements
case Rails.env
when 'development'
  model_id = "distilgpt2" # Smaller, faster
when 'production'
  model_id = "gpt2-large" # Larger, more accurate
end
```

### 5. Resource Management

```ruby
# Explicitly close resources when done
trustformers = TrustFormeRS.new
begin
  # Use trustformers...
ensure
  trustformers.close
end

# Or use blocks (automatic cleanup)
TrustFormeRS.new do |trustformers|
  # Use trustformers...
  # Automatically closed at end of block
end
```

## Benchmarking

```ruby
require 'benchmark'
require 'trustformers'

trustformers = TrustFormeRS.new
pipeline = trustformers.create_pipeline(
  task: :text_generation,
  model_id: "gpt2"
)

# Benchmark single generation
time = Benchmark.measure do
  result = pipeline.generate("Hello world")
end
puts "Single generation: #{time.real.round(3)}s"

# Benchmark batch processing
inputs = Array.new(100) { "Test prompt #{rand(1000)}" }

time = Benchmark.measure do
  results = pipeline.process_batch(inputs, batch_size: 16)
end
puts "Batch processing (100 items): #{time.real.round(3)}s"
puts "Average per item: #{(time.real / 100).round(4)}s"
```

## Examples

The `examples/` directory contains complete sample applications:

- **text_generation.rb** - Text generation with various configurations
- **sentiment_analysis.rb** - Sentiment classification
- **question_answering.rb** - Question answering system
- **chatbot.rb** - Simple conversational AI
- **summarization.rb** - Document summarization
- **rails_integration/** - Complete Rails application example

### Running Examples

```bash
cd examples
ruby text_generation.rb
ruby sentiment_analysis.rb
```

## Testing

Run the test suite:

```bash
bundle exec rake test
```

Run specific test files:

```bash
bundle exec ruby test/test_trustformers.rb
bundle exec ruby test/test_configuration.rb
bundle exec ruby test/test_tensor.rb
```

Performance tests:

```bash
bundle exec ruby test/performance_test.rb
```

## Troubleshooting

### Common Issues

1. **Gem installation fails**
   ```bash
   # Make sure you have development tools installed
   # Ubuntu/Debian:
   sudo apt-get install build-essential
   
   # macOS:
   xcode-select --install
   
   # Then retry gem installation
   gem install trustformers
   ```

2. **Native library not found**
   ```bash
   # Set environment variable to library location
   export TRUSTFORMERS_LIB_DIR=/path/to/trustformers/target/release
   gem install trustformers
   ```

3. **Model not found errors**
   ```ruby
   # Ensure models are available
   # Either use Hugging Face model IDs or local paths
   pipeline = trustformers.create_pipeline(
     task: :text_generation,
     model_id: "gpt2" # This will download from Hugging Face
   )
   ```

4. **Memory issues**
   ```ruby
   # Use smaller models or reduce batch sizes
   config = TrustFormeRS::Configuration.new(
     num_threads: 2, # Reduce thread count
     cache_dir: "/tmp" # Use temporary cache
   )
   ```

5. **Performance issues**
   ```ruby
   # Enable GPU acceleration
   if TrustFormeRS.gpu_available?
     trustformers.enable_gpu("auto")
   end
   
   # Use appropriate batch sizes
   results = pipeline.process_batch(inputs, batch_size: 8)
   ```

### Debug Mode

Enable debug logging:

```ruby
config = TrustFormeRS::Configuration.new(enable_logging: true)
trustformers = TrustFormeRS.new(config)
```

Or via environment variable:

```bash
DEBUG=1 ruby your_script.rb
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for your changes
4. Ensure all tests pass (`bundle exec rake test`)
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/cool-japan/trustformers.git
cd trustformers/bindings/ruby
bundle install
bundle exec rake compile
bundle exec rake test
```

## License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

## Acknowledgments

- Built on top of the powerful TrustFormeRS Rust library
- Inspired by Hugging Face Transformers for Python
- Thanks to the Ruby community for excellent tooling and libraries

## Support

- 📚 [Documentation](https://docs.trustformers.ai/ruby)
- 🐛 [Issue Tracker](https://github.com/cool-japan/trustformers/issues)
- 💬 [Discussions](https://github.com/cool-japan/trustformers/discussions)
- 📧 [Email Support](mailto:support@trustformers.ai)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes.

## Roadmap

- [ ] Additional model architectures
- [ ] More pipeline tasks
- [ ] Advanced streaming capabilities
- [ ] Model fine-tuning support
- [ ] Integration with popular Ruby frameworks
- [ ] Performance optimizations
- [ ] Enhanced documentation and examples