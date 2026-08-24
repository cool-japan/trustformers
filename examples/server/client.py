#!/usr/bin/env python3
"""TrustformeRS REST API client example.

Demonstrates every endpoint `trustformers-server` exposes today. Not run as
part of this repository's automated checks — see README.md's "What was
verified, and what was not" section.

Unlike an earlier version of this script, one loaded model here serves
exactly one task (`text-classification`, `token-classification`,
`question-answering`, or `text-generation`): the server builds a real
task-specific head at load time (see README.md), so there is no single
"bert-base-uncased" checkpoint that can answer classification, NER, and QA
requests all at once. You need one real local checkpoint directory per task
you want to try (they can all be copies/symlinks of the same architecture and
weights if you just want to exercise the wiring — the classification and NER
heads will simply be randomly initialised on top of it, per the `num_labels`
you choose, unless the directory's `model.safetensors` already carries
fine-tuned `classifier.*` weights).
"""

import time
from typing import Dict, List, Optional

import requests


class TrustformeRSClient:
    """Client for interacting with the TrustformeRS REST API."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()

    def health_check(self) -> Dict:
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def load_model(
        self,
        model_name: str,
        task: str,
        num_labels: Optional[int] = None,
        labels: Optional[List[str]] = None,
    ) -> str:
        """Loads a real local checkpoint for one task and returns its model id.

        `model_name` is a directory (absolute, or relative to the server's
        `MODEL_CACHE_DIR` when it set one) containing `config.json`,
        `model.safetensors`, and that checkpoint's tokenizer files. `task` is
        one of `text-classification`, `token-classification`,
        `question-answering`, `text-generation`. `num_labels` is required for
        the first two.
        """
        payload: Dict = {"model_name": model_name, "task": task}
        if num_labels is not None:
            payload["num_labels"] = num_labels
        if labels is not None:
            payload["labels"] = labels

        response = self.session.post(f"{self.base_url}/models", json=payload)
        response.raise_for_status()
        return response.json()["model_id"]

    def list_models(self) -> List[Dict]:
        response = self.session.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()

    def get_model_info(self, model_id: str) -> Dict:
        response = self.session.get(f"{self.base_url}/models/{model_id}")
        response.raise_for_status()
        return response.json()

    def unload_model(self, model_id: str) -> None:
        response = self.session.delete(f"{self.base_url}/models/{model_id}")
        response.raise_for_status()

    def classify_text(self, model_id: str, text: str) -> Dict:
        """Runs `model_id` (must be loaded for `text-classification`) on `text`."""
        payload = {"model_id": model_id, "text": text}
        response = self.session.post(f"{self.base_url}/predict/classification", json=payload)
        response.raise_for_status()
        return response.json()

    def generate_text(
        self,
        model_id: str,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> Dict:
        """Runs `model_id` (must be loaded for `text-generation`) on `prompt`.

        `max_length` is the *absolute* target sequence length (prompt tokens
        + generated tokens), not a new-tokens-only budget.
        """
        payload: Dict = {"model_id": model_id, "prompt": prompt}
        if max_length is not None:
            payload["max_length"] = max_length
        if temperature is not None:
            payload["temperature"] = temperature
        if top_k is not None:
            payload["top_k"] = top_k
        if top_p is not None:
            payload["top_p"] = top_p

        response = self.session.post(f"{self.base_url}/predict/generation", json=payload)
        response.raise_for_status()
        return response.json()

    def answer_question(self, model_id: str, question: str, context: str) -> Dict:
        """Runs `model_id` (must be loaded for `question-answering`).

        The response's `start_token`/`end_token` are indices into the
        tokenized `question [SEP] context` sequence, not character offsets
        into `context` — the server has no offset mapping to compute those
        honestly (see README.md).
        """
        payload = {"model_id": model_id, "question": question, "context": context}
        response = self.session.post(f"{self.base_url}/predict/qa", json=payload)
        response.raise_for_status()
        return response.json()

    def tag_tokens(self, model_id: str, text: str) -> Dict:
        """Runs `model_id` (must be loaded for `token-classification`) on `text`.

        Each returned entity's `token` is the tokenizer's own (possibly
        subword) text, not a whole word reconstructed from character offsets.
        """
        payload = {"model_id": model_id, "text": text}
        response = self.session.post(f"{self.base_url}/predict/ner", json=payload)
        response.raise_for_status()
        return response.json()

    def batch_inference(self, model_id: str, inputs: List[Dict]) -> Dict:
        """Runs every item in `inputs` through `model_id`'s own task.

        Each item's expected shape depends on that task: `{"text": ...}` for
        text-classification/token-classification, `{"question": ...,
        "context": ...}` for question-answering, `{"prompt": ..., ...}` for
        text-generation. A single bad item fails the whole call.
        """
        payload = {"model_id": model_id, "inputs": inputs}
        response = self.session.post(f"{self.base_url}/predict/batch", json=payload)
        response.raise_for_status()
        return response.json()


def main() -> None:
    """Demonstrates every endpoint. Requires real local checkpoints — see the
    module docstring — and is not run as part of this repository's automated
    checks (see README.md)."""
    client = TrustformeRSClient()

    # EDIT ME: point these at real local checkpoint directories before
    # running this script. Each must contain config.json, model.safetensors,
    # and tokenizer files for the architecture it names.
    classification_checkpoint = "/path/to/a/bert-style/checkpoint"
    ner_checkpoint = "/path/to/a/bert-style/checkpoint"
    qa_checkpoint = "/path/to/a/bert-style/checkpoint"
    generation_checkpoint = "/path/to/a/gpt2-style/checkpoint"

    print("Checking server health...")
    health = client.health_check()
    print(f"  status={health['status']} version={health['version']}\n")

    print("Loading a text-classification model...")
    classifier_id = client.load_model(
        classification_checkpoint,
        task="text-classification",
        num_labels=3,
        labels=["negative", "neutral", "positive"],
    )
    print(f"  model_id={classifier_id}\n")

    print("Text classification:")
    for text in [
        "This movie is absolutely fantastic! Best film I've seen all year.",
        "Terrible experience. Would not recommend to anyone.",
        "It was okay, nothing special but not bad either.",
    ]:
        result = client.classify_text(classifier_id, text)
        print(f"  \"{text[:50]}...\" -> {result['label']} (score={result['score']:.3f})")
    print()

    print("Batch classification:")
    batch_result = client.batch_inference(
        classifier_id,
        [{"text": "Great product!"}, {"text": "Not satisfied"}, {"text": "Average quality"}],
    )
    print(f"  {len(batch_result['results'])} results in {batch_result['total_time_ms']}ms\n")

    client.unload_model(classifier_id)

    print("Loading a token-classification (NER) model...")
    ner_id = client.load_model(
        ner_checkpoint,
        task="token-classification",
        num_labels=9,
        labels=["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"],
    )
    ner_result = client.tag_tokens(ner_id, "John Smith works at Microsoft in Seattle.")
    print("Named entity recognition:")
    for entity in ner_result["entities"]:
        print(f"  {entity['token']!r}: {entity['label']} (score={entity['score']:.3f})")
    print()
    client.unload_model(ner_id)

    print("Loading a question-answering model...")
    qa_id = client.load_model(qa_checkpoint, task="question-answering")
    context = (
        "TrustformeRS is a transformer library written in Rust. "
        "It supports several architectures including BERT, GPT-2, and T5."
    )
    for question in ["What is TrustformeRS?", "What language is it written in?"]:
        qa_result = client.answer_question(qa_id, question, context)
        print(f"  Q: {question}")
        print(f"  A: {qa_result['answer']} (score={qa_result['score']:.3f})")
    print()
    client.unload_model(qa_id)

    print("Loading a text-generation model...")
    generator_id = client.load_model(generation_checkpoint, task="text-generation")
    start = time.time()
    gen_result = client.generate_text(generator_id, "The future of AI is", max_length=40, temperature=0.8)
    elapsed_ms = (time.time() - start) * 1000
    print(f"  Generated: {gen_result['generated_text']!r}")
    print(f"  {gen_result['completion_tokens']} tokens in {elapsed_ms:.0f}ms (client-observed)\n")
    client.unload_model(generator_id)

    print("All models unloaded.")


if __name__ == "__main__":
    main()
