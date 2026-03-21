#!/usr/bin/env python3
"""
Example Python client for TrustformeRS gRPC server.

Requirements:
    pip install grpcio grpcio-tools

Generate Python code from proto:
    python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. proto/inference.proto
"""

import grpc
import time
import argparse
from typing import List, Optional

# Note: These imports assume you've generated the Python code from the proto file
# Run: python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. proto/inference.proto
try:
    import inference_pb2
    import inference_pb2_grpc
except ImportError:
    print("Please generate Python code from proto file first:")
    print("python -m grpc_tools.protoc -I./proto --python_out=. --grpc_python_out=. proto/inference.proto")
    exit(1)


class TrustformeRSClient:
    """Client for TrustformeRS gRPC server."""
    
    def __init__(self, host: str = "localhost", port: int = 50051):
        """Initialize client connection."""
        self.channel = grpc.insecure_channel(f"{host}:{port}")
        self.stub = inference_pb2_grpc.InferenceServiceStub(self.channel)
    
    def load_model(self, model_id: str, model_path: Optional[str] = None, 
                   device: str = "cpu", use_fp16: bool = False) -> bool:
        """Load a model into the server."""
        request = inference_pb2.LoadModelRequest(
            model_id=model_id,
            model_path=model_path or model_id,
            device=device,
            use_fp16=use_fp16,
            compile=False
        )
        
        try:
            response = self.stub.LoadModel(request)
            print(f"Model loaded: {response.model_id} in {response.load_time_ms:.2f}ms")
            return response.success
        except grpc.RpcError as e:
            print(f"Error loading model: {e.details()}")
            return False
    
    def unload_model(self, model_id: str):
        """Unload a model from the server."""
        request = inference_pb2.UnloadModelRequest(model_id=model_id)
        try:
            self.stub.UnloadModel(request)
            print(f"Model unloaded: {model_id}")
        except grpc.RpcError as e:
            print(f"Error unloading model: {e.details()}")
    
    def list_models(self) -> List[str]:
        """List all loaded models."""
        response = self.stub.ListModels(inference_pb2.google_dot_protobuf_dot_empty__pb2.Empty())
        return [(m.model_id, m.status.is_loaded) for m in response.models]
    
    def predict(self, model_id: str, text: str, **options) -> str:
        """Make a single prediction."""
        predict_options = inference_pb2.PredictOptions(**options)
        
        request = inference_pb2.PredictRequest(
            model_id=model_id,
            text=text,
            options=predict_options
        )
        
        try:
            response = self.stub.Predict(request)
            
            # Handle different output types
            if response.HasField("text_output"):
                return response.text_output.text
            elif response.HasField("classification_output"):
                labels = response.classification_output.labels
                return f"{labels[0].label} (score: {labels[0].score:.3f})"
            else:
                return str(response)
                
        except grpc.RpcError as e:
            print(f"Prediction error: {e.details()}")
            return None
    
    def batch_predict(self, model_id: str, texts: List[str], batch_size: int = 8) -> List[str]:
        """Make batch predictions."""
        request = inference_pb2.BatchPredictRequest(
            model_id=model_id,
            texts=texts,
            batch_size=batch_size
        )
        
        try:
            response = self.stub.BatchPredict(request)
            results = []
            
            for pred in response.predictions:
                if pred.HasField("text_output"):
                    results.append(pred.text_output.text)
                elif pred.HasField("classification_output"):
                    labels = pred.classification_output.labels
                    results.append(f"{labels[0].label} ({labels[0].score:.3f})")
                else:
                    results.append(str(pred))
            
            if response.metrics:
                m = response.metrics
                print(f"Batch metrics: {m.total_tokens} tokens in {m.total_latency_ms:.2f}ms "
                      f"({m.tokens_per_second:.0f} tokens/sec)")
            
            return results
            
        except grpc.RpcError as e:
            print(f"Batch prediction error: {e.details()}")
            return []
    
    def stream_predict(self, model_id: str, initial_text: str, 
                      max_tokens: int = 50, temperature: float = 0.8):
        """Stream predictions with real-time generation."""
        
        # Start streaming session
        def request_generator():
            # Initial request
            yield inference_pb2.StreamPredictRequest(
                start=inference_pb2.StreamStartRequest(
                    model_id=model_id,
                    initial_text=initial_text,
                    options=inference_pb2.PredictOptions(
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        do_sample=True
                    )
                )
            )
        
        try:
            responses = self.stub.StreamPredict(request_generator())
            
            full_text = initial_text
            for response in responses:
                print(response.text, end='', flush=True)
                full_text += response.text
                
                if response.is_final:
                    print("\n")
                    if response.metrics:
                        print(f"Latency: {response.metrics.latency_ms:.2f}ms")
                    break
            
            return full_text
            
        except grpc.RpcError as e:
            print(f"\nStreaming error: {e.details()}")
            return None
    
    def close(self):
        """Close the connection."""
        self.channel.close()


def main():
    """Example usage of the client."""
    parser = argparse.ArgumentParser(description="TrustformeRS gRPC Client")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=50051, help="Server port")
    parser.add_argument("--model", default="bert-base-uncased", help="Model ID")
    args = parser.parse_args()
    
    # Create client
    client = TrustformeRSClient(args.host, args.port)
    
    try:
        # List models
        print("Currently loaded models:")
        models = client.list_models()
        if not models:
            print("  No models loaded")
        else:
            for model_id, is_loaded in models:
                print(f"  - {model_id} ({'loaded' if is_loaded else 'not loaded'})")
        print()
        
        # Load a model
        print(f"Loading model: {args.model}")
        if client.load_model(args.model):
            print("Model loaded successfully!\n")
        else:
            print("Failed to load model\n")
            return
        
        # Single prediction
        print("Single prediction:")
        text = "The capital of France is"
        result = client.predict(args.model, text, max_new_tokens=10)
        print(f"Input: {text}")
        print(f"Output: {result}\n")
        
        # Batch prediction
        print("Batch prediction:")
        texts = [
            "The weather today is",
            "Machine learning is",
            "Python is a"
        ]
        results = client.batch_predict(args.model, texts)
        for text, result in zip(texts, results):
            print(f"  {text} -> {result}")
        print()
        
        # Streaming (if supported by model)
        if "gpt" in args.model.lower():
            print("Streaming prediction:")
            client.stream_predict(args.model, "Once upon a time", max_tokens=30)
            print()
        
    finally:
        client.close()


if __name__ == "__main__":
    main()