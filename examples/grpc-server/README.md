# TrustformeRS gRPC Server

A high-performance gRPC server for serving TrustformeRS models with support for streaming, batching, and model management.

## Features

- **Multiple Inference Modes**: Single prediction, streaming, and batch processing
- **Model Management**: Dynamic loading/unloading of models
- **Health Checks**: Built-in health monitoring
- **Reflection**: Support for grpcurl/grpcui debugging
- **Metrics**: Latency and throughput tracking
- **Multi-Model Support**: Serve multiple models concurrently

## Building

```bash
cd examples/grpc-server
cargo build --release
```

## Running

```bash
# Start the server on default port (50051)
cargo run --release

# With custom settings
RUST_LOG=info cargo run --release
```

## API Overview

The server implements the following gRPC services:

### InferenceService

- `Predict`: Single inference request
- `StreamPredict`: Streaming inference for real-time responses
- `BatchPredict`: Batch inference for multiple inputs
- `ListModels`: Get available models
- `GetModelInfo`: Get model information
- `LoadModel`: Load a model into memory
- `UnloadModel`: Unload a model from memory

## Client Examples

### Python Client

```python
import grpc
from inference_pb2 import PredictRequest, LoadModelRequest
from inference_pb2_grpc import InferenceServiceStub

# Connect to server
channel = grpc.insecure_channel('localhost:50051')
stub = InferenceServiceStub(channel)

# Load a model
load_request = LoadModelRequest(
    model_id="bert-base-uncased",
    model_path="bert-base-uncased",
    device="cpu",
    use_fp16=False
)
stub.LoadModel(load_request)

# Make prediction
predict_request = PredictRequest(
    model_id="bert-base-uncased",
    text="Hello, world!"
)
response = stub.Predict(predict_request)
print(f"Response: {response.text_output.text}")
```

### Using grpcurl

```bash
# List available services
grpcurl -plaintext localhost:50051 list

# Describe the service
grpcurl -plaintext localhost:50051 describe trustformers.inference.InferenceService

# List loaded models
grpcurl -plaintext localhost:50051 trustformers.inference.InferenceService/ListModels

# Load a model
grpcurl -plaintext -d '{
  "model_id": "bert-base-uncased",
  "model_path": "bert-base-uncased",
  "device": "cpu"
}' localhost:50051 trustformers.inference.InferenceService/LoadModel

# Make a prediction
grpcurl -plaintext -d '{
  "model_id": "bert-base-uncased",
  "text": "The capital of France is [MASK]."
}' localhost:50051 trustformers.inference.InferenceService/Predict
```

## Streaming Example

```python
# Streaming inference
def stream_generator():
    yield StreamPredictRequest(
        start=StreamStartRequest(
            model_id="gpt2",
            initial_text="Once upon a time",
            options=PredictOptions(
                max_new_tokens=50,
                temperature=0.8
            )
        )
    )
    
    yield StreamPredictRequest(
        continue=StreamContinueRequest(
            session_id=session_id,
            text=" in a land far away",
            end_stream=False
        )
    )

responses = stub.StreamPredict(stream_generator())
for response in responses:
    print(f"Generated: {response.text}")
```

## Batch Processing

```python
# Batch inference
batch_request = BatchPredictRequest(
    model_id="bert-base-uncased",
    texts=[
        "The weather is nice today.",
        "I love machine learning.",
        "Transformers are powerful."
    ],
    batch_size=3
)
batch_response = stub.BatchPredict(batch_request)
for i, pred in enumerate(batch_response.predictions):
    print(f"Input {i}: {pred.classification_output.labels[0].label}")
```

## Configuration

Environment variables:
- `RUST_LOG`: Logging level (trace, debug, info, warn, error)
- `GRPC_PORT`: Server port (default: 50051)
- `MAX_MODELS`: Maximum concurrent models (default: 10)
- `DEFAULT_DEVICE`: Default device (cpu, cuda, metal)

## Performance Considerations

1. **Model Loading**: Models are loaded on-demand and cached in memory
2. **Batching**: Use batch inference for better throughput
3. **Streaming**: Use streaming for real-time generation tasks
4. **Connection Pooling**: Reuse gRPC channels for better performance

## Docker Deployment

```dockerfile
FROM rust:latest as builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin trustformers-grpc-server

FROM debian:bookworm-slim
COPY --from=builder /app/target/release/trustformers-grpc-server /usr/local/bin/
EXPOSE 50051
CMD ["trustformers-grpc-server"]
```

## Kubernetes Deployment

See the `k8s/` directory for example Kubernetes manifests including:
- Deployment with resource limits
- Service for load balancing
- HorizontalPodAutoscaler for scaling
- NetworkPolicy for security

## Monitoring

The server exposes Prometheus metrics at `/metrics`:
- Request latency histograms
- Model loading times
- Memory usage per model
- Request counts and error rates

## Security

1. **TLS Support**: Configure with certificates for production
2. **Authentication**: Implement interceptors for auth
3. **Rate Limiting**: Use middleware for request throttling
4. **Input Validation**: All inputs are validated before processing

## Troubleshooting

### Common Issues

1. **Model not loading**: Check model path and available memory
2. **High latency**: Consider using GPU or reducing batch size
3. **Connection errors**: Verify firewall rules and port availability

### Debug Mode

Enable debug logging:
```bash
RUST_LOG=trustformers_grpc_server=debug cargo run
```

Use grpcui for interactive debugging:
```bash
grpcui -plaintext localhost:50051
```