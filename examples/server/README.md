# TrustformeRS REST API Server

A production-ready REST API server for serving TrustformeRS transformer models.

## Features

- 🚀 High-performance inference server built with Axum
- 🔄 Dynamic model loading and unloading
- 📦 Batch inference support
- 🎯 Multiple task endpoints (classification, generation, QA, NER)
- 📊 Model management API
- 🔧 Health checks and monitoring
- 🌐 CORS support for web applications
- 📈 Request tracing and logging

## Quick Start

### Building the Server

```bash
cd examples/server
cargo build --release
```

### Running the Server

```bash
# Run with default settings
cargo run --release

# With custom logging
RUST_LOG=debug cargo run --release

# Specify port (default: 8080)
PORT=3000 cargo run --release
```

## API Endpoints

### Health Check

```bash
GET /health
```

Returns server health status and version.

### Model Management

#### List Models
```bash
GET /models
```

Returns a list of all loaded models.

#### Get Model Info
```bash
GET /models/{model_id}
```

Returns detailed information about a specific model.

#### Load Model
```bash
POST /models
Content-Type: application/json

{
  "model_name": "bert-base-uncased",
  "model_type": "bert",
  "cache_dir": "/path/to/cache" // optional
}
```

Loads a new model and returns its ID.

#### Unload Model
```bash
DELETE /models/{model_id}
```

Unloads a model from memory.

### Inference Endpoints

#### Text Classification
```bash
POST /predict/classification
Content-Type: application/json

{
  "model_id": "your-model-id",
  "text": "This movie is fantastic!",
  "candidate_labels": ["positive", "negative"] // optional
}
```

#### Text Generation
```bash
POST /predict/generation
Content-Type: application/json

{
  "model_id": "your-model-id",
  "prompt": "Once upon a time",
  "max_length": 100,
  "temperature": 0.8,
  "top_k": 50,
  "top_p": 0.95
}
```

#### Question Answering
```bash
POST /predict/qa
Content-Type: application/json

{
  "model_id": "your-model-id",
  "question": "What is the capital of France?",
  "context": "France is a country in Europe. Its capital is Paris."
}
```

#### Named Entity Recognition
```bash
POST /predict/ner
Content-Type: application/json

{
  "model_id": "your-model-id",
  "text": "John works at Microsoft in Seattle."
}
```

#### Batch Inference
```bash
POST /predict/batch
Content-Type: application/json

{
  "model_id": "your-model-id",
  "task": "classification",
  "inputs": [
    {"text": "First text to classify"},
    {"text": "Second text to classify"}
  ]
}
```

## Example Usage

### Using cURL

```bash
# Load a model
MODEL_ID=$(curl -s -X POST http://localhost:8080/models \
  -H "Content-Type: application/json" \
  -d '{"model_name": "bert-base-uncased", "model_type": "bert"}' \
  | jq -r .model_id)

# Classify text
curl -X POST http://localhost:8080/predict/classification \
  -H "Content-Type: application/json" \
  -d "{
    \"model_id\": \"$MODEL_ID\",
    \"text\": \"This is amazing!\"
  }"
```

### Using Python

```python
import requests
import json

# Server URL
BASE_URL = "http://localhost:8080"

# Load a model
response = requests.post(f"{BASE_URL}/models", json={
    "model_name": "bert-base-uncased",
    "model_type": "bert"
})
model_id = response.json()["model_id"]

# Classify text
response = requests.post(f"{BASE_URL}/predict/classification", json={
    "model_id": model_id,
    "text": "I love this product!"
})
result = response.json()
print(f"Label: {result['label']}, Score: {result['score']}")

# Generate text
response = requests.post(f"{BASE_URL}/predict/generation", json={
    "model_id": model_id,
    "prompt": "The future of AI is",
    "max_length": 50,
    "temperature": 0.8
})
print(f"Generated: {response.json()['generated_text']}")
```

## Configuration

The server can be configured through environment variables:

- `PORT`: Server port (default: 8080)
- `RUST_LOG`: Logging level (default: info)
- `MODEL_CACHE_DIR`: Directory for model cache (default: ~/.cache/trustformers)
- `MAX_MODELS`: Maximum number of models to keep loaded (default: 10)

## Performance Tuning

### Concurrent Requests

The server handles concurrent requests efficiently using Tokio's async runtime. For optimal performance:

1. Use batch endpoints for multiple inputs
2. Keep models loaded between requests
3. Enable response compression for large outputs

### Memory Management

- Models are loaded on-demand and can be unloaded to free memory
- Use the model management API to control memory usage
- Monitor memory usage through system metrics

## Docker Deployment

```dockerfile
FROM rust:1.70 as builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin trustformers-server

FROM ubuntu:22.04
RUN apt-get update && apt-get install -y \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/trustformers-server /usr/local/bin/
EXPOSE 8080
CMD ["trustformers-server"]
```

Build and run:
```bash
docker build -t trustformers-server .
docker run -p 8080:8080 trustformers-server
```

## Monitoring

The server includes built-in tracing and metrics:

- Request/response logging
- Performance metrics per endpoint
- Model loading/inference times
- Memory usage tracking

## Security

- CORS headers for web application integration
- Request validation and error handling
- Rate limiting (configure through reverse proxy)
- Authentication (implement as middleware)

## Contributing

See the main TrustformeRS repository for contribution guidelines.

## License

MIT OR Apache-2.0