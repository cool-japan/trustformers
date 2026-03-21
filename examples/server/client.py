#!/usr/bin/env python3
"""
TrustformeRS REST API Client Example

This script demonstrates how to interact with the TrustformeRS server API.
"""

import requests
import json
import time
from typing import Dict, List, Optional


class TrustformeRSClient:
    """Client for interacting with TrustformeRS REST API"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def health_check(self) -> Dict:
        """Check server health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def load_model(self, model_name: str, model_type: str, cache_dir: Optional[str] = None) -> str:
        """Load a model and return its ID"""
        payload = {
            "model_name": model_name,
            "model_type": model_type
        }
        if cache_dir:
            payload["cache_dir"] = cache_dir
            
        response = self.session.post(f"{self.base_url}/models", json=payload)
        response.raise_for_status()
        return response.json()["model_id"]
    
    def list_models(self) -> List[Dict]:
        """List all loaded models"""
        response = self.session.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self, model_id: str) -> Dict:
        """Get information about a specific model"""
        response = self.session.get(f"{self.base_url}/models/{model_id}")
        response.raise_for_status()
        return response.json()
    
    def unload_model(self, model_id: str) -> None:
        """Unload a model"""
        response = self.session.delete(f"{self.base_url}/models/{model_id}")
        response.raise_for_status()
    
    def classify_text(
        self, 
        model_id: str, 
        text: str, 
        candidate_labels: Optional[List[str]] = None
    ) -> Dict:
        """Classify text using the specified model"""
        payload = {
            "model_id": model_id,
            "text": text
        }
        if candidate_labels:
            payload["candidate_labels"] = candidate_labels
            
        response = self.session.post(f"{self.base_url}/predict/classification", json=payload)
        response.raise_for_status()
        return response.json()
    
    def generate_text(
        self,
        model_id: str,
        prompt: str,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> Dict:
        """Generate text using the specified model"""
        payload = {
            "model_id": model_id,
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature
        }
        if top_k is not None:
            payload["top_k"] = top_k
        if top_p is not None:
            payload["top_p"] = top_p
            
        response = self.session.post(f"{self.base_url}/predict/generation", json=payload)
        response.raise_for_status()
        return response.json()
    
    def answer_question(self, model_id: str, question: str, context: str) -> Dict:
        """Answer a question based on the given context"""
        payload = {
            "model_id": model_id,
            "question": question,
            "context": context
        }
        response = self.session.post(f"{self.base_url}/predict/qa", json=payload)
        response.raise_for_status()
        return response.json()
    
    def extract_entities(self, model_id: str, text: str) -> Dict:
        """Extract named entities from text"""
        payload = {
            "model_id": model_id,
            "text": text
        }
        response = self.session.post(f"{self.base_url}/predict/ner", json=payload)
        response.raise_for_status()
        return response.json()
    
    def batch_inference(self, model_id: str, task: str, inputs: List[Dict]) -> Dict:
        """Perform batch inference"""
        payload = {
            "model_id": model_id,
            "task": task,
            "inputs": inputs
        }
        response = self.session.post(f"{self.base_url}/predict/batch", json=payload)
        response.raise_for_status()
        return response.json()


def main():
    """Demonstrate API usage"""
    client = TrustformeRSClient()
    
    try:
        # Check server health
        print("🏥 Checking server health...")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Version: {health['version']}")
        print()
        
        # Load a model
        print("📦 Loading BERT model...")
        model_id = client.load_model("bert-base-uncased", "bert")
        print(f"   Model ID: {model_id}")
        print()
        
        # List models
        print("📋 Listing loaded models...")
        models = client.list_models()
        for model in models:
            print(f"   - {model['name']} ({model['id']})")
        print()
        
        # Text classification
        print("🎯 Text Classification Demo")
        texts = [
            "This movie is absolutely fantastic! Best film I've seen all year.",
            "Terrible experience. Would not recommend to anyone.",
            "It was okay, nothing special but not bad either."
        ]
        
        for text in texts:
            result = client.classify_text(model_id, text, ["positive", "negative", "neutral"])
            print(f"   Text: \"{text[:50]}...\"")
            print(f"   Label: {result['label']} (score: {result['score']:.3f})")
            print()
        
        # Question answering
        print("❓ Question Answering Demo")
        context = """
        TrustformeRS is a high-performance transformer library written in Rust.
        It provides a fast and memory-efficient alternative to Python-based libraries.
        The library supports various transformer architectures including BERT, GPT-2, and T5.
        """
        questions = [
            "What is TrustformeRS?",
            "What programming language is it written in?",
            "What models does it support?"
        ]
        
        for question in questions:
            result = client.answer_question(model_id, question, context)
            print(f"   Q: {question}")
            print(f"   A: {result['answer']} (score: {result['score']:.3f})")
            print()
        
        # Named entity recognition
        print("🏷️  Named Entity Recognition Demo")
        ner_text = "John Smith works at Microsoft in Seattle. He will visit Paris next week."
        result = client.extract_entities(model_id, ner_text)
        print(f"   Text: \"{ner_text}\"")
        print("   Entities:")
        for entity in result['entities']:
            print(f"   - {entity['word']}: {entity['entity']} (score: {entity['score']:.3f})")
        print()
        
        # Batch inference
        print("📦 Batch Inference Demo")
        batch_inputs = [
            {"text": "Great product!"},
            {"text": "Not satisfied"},
            {"text": "Average quality"}
        ]
        start_time = time.time()
        result = client.batch_inference(model_id, "classification", batch_inputs)
        end_time = time.time()
        
        print(f"   Processed {len(batch_inputs)} inputs in {result['total_time_ms']}ms")
        print(f"   Client-side time: {(end_time - start_time) * 1000:.1f}ms")
        print()
        
        # Clean up
        print("🧹 Cleaning up...")
        client.unload_model(model_id)
        print("   Model unloaded")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}")
        print("   Make sure the TrustformeRS server is running on http://localhost:8080")


if __name__ == "__main__":
    main()