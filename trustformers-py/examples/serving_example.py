#!/usr/bin/env python3
"""
TrustformeRS Model Serving Example

This example demonstrates how to use the TrustformeRS model serving infrastructure
to deploy models for production inference with features like:

- FastAPI-based REST API
- Dynamic batching for efficiency
- Load balancing across model instances
- A/B testing capabilities
- Model versioning
- Prometheus metrics
- Health monitoring

Requirements:
    pip install fastapi uvicorn prometheus-client

Usage:
    python examples/serving_example.py
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add the project root to the path to import trustformers
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import trustformers as tf
    from trustformers.serving import (
        ModelConfig,
        InferenceRequest,
        ABTestConfig,
        ServingManager,
        create_app,
        serve,
        serve_model,
    )
except ImportError as e:
    print(f"Could not import TrustformeRS serving: {e}")
    print("Make sure you have installed the required dependencies:")
    print("  pip install fastapi uvicorn prometheus-client")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def basic_serving_example():
    """
    Basic example of serving a single model
    """
    print("=== Basic Model Serving Example ===")
    
    try:
        # Serve a model (this uses a mock model since we don't have real model files)
        serving_manager = await serve_model(
            model_path="bert-base-uncased",  # Would normally be a path to model files
            model_id="bert-classifier",
            version="1.0.0",
            max_batch_size=8,
            device="cpu"
        )
        
        print("✓ Model loaded successfully")
        
        # Create an inference request
        request = InferenceRequest(
            model_id="bert-classifier",
            inputs="This is a test sentence for classification.",
            parameters={"max_length": 512}
        )
        
        # Process the request
        print(f"Processing request: {request.inputs}")
        response = await serving_manager.inference(request)
        
        print(f"✓ Inference completed")
        print(f"  Request ID: {response.request_id}")
        print(f"  Model: {response.model_id}:{response.version}")
        print(f"  Output: {response.outputs}")
        print(f"  Timing: {response.timing}")
        print(f"  Usage: {response.usage}")
        
        # Get health status
        health = serving_manager.get_health()
        print(f"✓ Service health: {health.status}")
        print(f"  Active models: {health.system['active_instances']}")
        print(f"  Total requests: {health.system['total_requests']}")
        
        # Cleanup
        await serving_manager.shutdown()
        print("✓ Service shutdown complete")
        
    except Exception as e:
        logger.error(f"Error in basic serving example: {e}")
        raise

async def advanced_serving_example():
    """
    Advanced example with multiple models, load balancing, and A/B testing
    """
    print("\n=== Advanced Model Serving Example ===")
    
    try:
        serving_manager = ServingManager.get_instance()
        
        # Load multiple model versions
        configs = [
            ModelConfig(
                model_id="bert-classifier",
                model_path="bert-base-uncased",
                version="1.0.0",
                max_batch_size=4,
                weight=1.0,
                tags={"env": "production", "model": "bert-base"}
            ),
            ModelConfig(
                model_id="bert-classifier", 
                model_path="bert-large-uncased",
                version="2.0.0",
                max_batch_size=2,
                weight=1.5,
                tags={"env": "production", "model": "bert-large"}
            ),
            ModelConfig(
                model_id="roberta-classifier",
                model_path="roberta-base",
                version="1.0.0",
                max_batch_size=4,
                weight=1.0,
                tags={"env": "production", "model": "roberta-base"}
            )
        ]
        
        # Load all models
        for config in configs:
            success = await serving_manager.load_model(config)
            if success:
                print(f"✓ Loaded {config.model_id}:{config.version}")
            else:
                print(f"✗ Failed to load {config.model_id}:{config.version}")
        
        # Set up A/B testing
        ab_test = ABTestConfig(
            test_name="bert-vs-roberta",
            model_a="bert-classifier",
            model_b="roberta-classifier", 
            traffic_split=0.7,  # 70% to bert, 30% to roberta
            enabled=True,
            metadata={"description": "Compare BERT vs RoBERTa performance"}
        )
        
        success = serving_manager.ab_test_manager.create_test(ab_test)
        if success:
            print("✓ A/B test created: bert-vs-roberta")
        
        # Process multiple requests to demonstrate features
        print("\nProcessing multiple requests...")
        requests = [
            InferenceRequest(
                model_id="bert-classifier",
                inputs=f"Test sentence {i} for classification.",
                parameters={"ab_test": "bert-vs-roberta", "max_length": 512},
                priority=i % 3  # Vary priority
            )
            for i in range(10)
        ]
        
        # Process requests concurrently
        start_time = time.time()
        tasks = [serving_manager.inference(req) for req in requests]
        responses = await asyncio.gather(*tasks)
        processing_time = time.time() - start_time
        
        print(f"✓ Processed {len(responses)} requests in {processing_time:.2f}s")
        print(f"  Average per request: {processing_time/len(responses):.3f}s")
        
        # Analyze A/B test results
        ab_results = serving_manager.ab_test_manager.get_test_results("bert-vs-roberta")
        if ab_results:
            print(f"\n📊 A/B Test Results:")
            print(f"  BERT requests: {ab_results['model_a_requests']}")
            print(f"  RoBERTa requests: {ab_results['model_b_requests']}")
            print(f"  BERT avg time: {ab_results['model_a_avg_time']:.3f}s")
            print(f"  RoBERTa avg time: {ab_results['model_b_avg_time']:.3f}s")
        
        # Show model statistics
        print(f"\n📈 Model Statistics:")
        for key, instance in serving_manager.instances.items():
            print(f"  {key}:")
            print(f"    Status: {instance.status.value}")
            print(f"    Requests: {instance.request_count}")
            print(f"    Avg response time: {instance.avg_response_time:.3f}s")
            print(f"    Load time: {instance.load_time:.2f}s")
        
        # Test version management
        print(f"\n🔄 Version Management:")
        versions = serving_manager.version_manager.list_versions("bert-classifier")
        print(f"  Available versions: {versions}")
        
        active = serving_manager.version_manager.get_active_version("bert-classifier")
        print(f"  Active version: {active}")
        
        # Switch to version 2.0.0
        success = serving_manager.version_manager.set_active_version("bert-classifier", "2.0.0")
        if success:
            print(f"  ✓ Switched to version 2.0.0")
            active = serving_manager.version_manager.get_active_version("bert-classifier")
            print(f"  New active version: {active}")
        
        # Cleanup
        await serving_manager.shutdown()
        print("\n✓ Advanced serving demo complete")
        
    except Exception as e:
        logger.error(f"Error in advanced serving example: {e}")
        raise

def fastapi_server_example():
    """
    Example of running a FastAPI server
    """
    print("\n=== FastAPI Server Example ===")
    print("Starting FastAPI server...")
    print("Once running, you can:")
    print("  - View API docs at: http://localhost:8000/docs")
    print("  - Check health at: http://localhost:8000/v1/health")
    print("  - View metrics at: http://localhost:8000/metrics")
    print("  - Send inference requests to: http://localhost:8000/v1/inference")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        # This will start the FastAPI server
        serve(
            host="127.0.0.1",
            port=8000,
            workers=1,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n✓ Server stopped")
    except ImportError:
        print("FastAPI/uvicorn not available. Install with: pip install fastapi uvicorn")

def client_example():
    """
    Example of using the serving API as a client
    """
    print("\n=== Client Example ===")
    
    import requests
    
    # Example requests you can make to a running server
    examples = {
        "Health Check": {
            "method": "GET",
            "url": "http://localhost:8000/v1/health",
            "data": None
        },
        "Load Model": {
            "method": "POST", 
            "url": "http://localhost:8000/v1/models/load",
            "data": {
                "model_id": "test-model",
                "model_path": "bert-base-uncased",
                "version": "1.0.0",
                "max_batch_size": 8,
                "device": "cpu"
            }
        },
        "Inference Request": {
            "method": "POST",
            "url": "http://localhost:8000/v1/inference", 
            "data": {
                "model_id": "test-model",
                "inputs": "This is a test sentence.",
                "parameters": {"max_length": 512},
                "stream": False
            }
        },
        "List Models": {
            "method": "GET",
            "url": "http://localhost:8000/v1/models",
            "data": None
        },
        "Create A/B Test": {
            "method": "POST",
            "url": "http://localhost:8000/v1/ab-tests",
            "data": {
                "test_name": "model-comparison",
                "model_a": "bert-base",
                "model_b": "roberta-base",
                "traffic_split": 0.5,
                "enabled": True
            }
        }
    }
    
    print("Example API calls (requires running server):")
    for name, example in examples.items():
        print(f"\n{name}:")
        print(f"  {example['method']} {example['url']}")
        if example['data']:
            print(f"  Data: {json.dumps(example['data'], indent=2)}")
        
        # Uncomment to actually make requests (requires running server)
        # try:
        #     if example['method'] == 'GET':
        #         response = requests.get(example['url'])
        #     else:
        #         response = requests.post(example['url'], json=example['data'])
        #     print(f"  Response: {response.status_code}")
        #     if response.status_code < 400:
        #         print(f"  {json.dumps(response.json(), indent=2)}")
        # except requests.exceptions.ConnectionError:
        #     print("  (Server not running)")

async def main():
    """
    Main function to run all examples
    """
    print("🚀 TrustformeRS Model Serving Examples")
    print("=====================================")
    
    try:
        # Run basic example
        await basic_serving_example()
        
        # Run advanced example
        await advanced_serving_example()
        
        # Show client examples
        client_example()
        
        # Ask if user wants to start server
        print(f"\n🌐 FastAPI Server")
        print("Do you want to start the FastAPI server? (y/n): ", end="")
        
        # For demo purposes, we'll skip the interactive part
        print("n")
        print("Skipping server startup (run with 'y' to start server)")
        
        print(f"\n✅ All examples completed successfully!")
        print(f"\nTo start the server manually, run:")
        print(f"  python -c \"from trustformers.serving import serve; serve()\"")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise

if __name__ == "__main__":
    # Check for required dependencies
    missing_deps = []
    
    try:
        import fastapi
    except ImportError:
        missing_deps.append("fastapi")
    
    try:
        import uvicorn
    except ImportError:
        missing_deps.append("uvicorn")
    
    if missing_deps:
        print(f"Missing dependencies: {', '.join(missing_deps)}")
        print(f"Install with: pip install {' '.join(missing_deps)}")
        print("Running examples without server functionality...")
    
    # Run the examples
    asyncio.run(main())