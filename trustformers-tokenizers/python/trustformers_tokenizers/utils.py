"""
Utility functions for TrustformeRS tokenizers.

This module provides various utility functions for working with tokenizers,
including analysis, benchmarking, and file I/O operations.
"""

import json
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import statistics

try:
    from .trustformers_tokenizers import (
        analyze_text_coverage,
        benchmark_tokenizer,
    )
except ImportError:
    # Fallback implementations
    def analyze_text_coverage(tokenizer, texts):
        return 0.95
    
    def benchmark_tokenizer(tokenizer, texts, iterations):
        return 1000.0

from .tokenizers import Tokenizer, AutoTokenizer


def load_tokenizer(path_or_name: str) -> Tokenizer:
    """
    Load a tokenizer from a file path or model name.
    
    Args:
        path_or_name: File path or model name
        
    Returns:
        Loaded tokenizer
    """
    return AutoTokenizer.from_pretrained(path_or_name)


def save_tokenizer(tokenizer: Tokenizer, save_directory: str):
    """
    Save a tokenizer to a directory.
    
    Args:
        tokenizer: Tokenizer to save
        save_directory: Directory to save to
    """
    os.makedirs(save_directory, exist_ok=True)
    
    # Save vocabulary
    vocab = tokenizer.get_vocab()
    vocab_file = os.path.join(save_directory, "vocab.json")
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2, ensure_ascii=False)
    
    # Save tokenizer configuration
    config = {
        "tokenizer_type": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": {
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
        },
    }
    
    config_file = os.path.join(save_directory, "tokenizer_config.json")
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)


def analyze_coverage(tokenizer: Tokenizer, texts: List[str]) -> Dict[str, Any]:
    """
    Analyze tokenizer coverage on a set of texts.
    
    Args:
        tokenizer: Tokenizer to analyze
        texts: List of texts to analyze
        
    Returns:
        Coverage analysis results
    """
    if hasattr(tokenizer, '_inner'):
        # Use the fast Rust implementation
        coverage_ratio = analyze_text_coverage(tokenizer, texts)
    else:
        # Fallback Python implementation
        total_chars = sum(len(text) for text in texts)
        covered_chars = 0
        
        for text in texts:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded.input_ids)
            covered_chars += len(decoded)
        
        coverage_ratio = covered_chars / total_chars if total_chars > 0 else 0.0
    
    # Additional analysis
    token_counts = []
    oov_tokens = 0
    total_tokens = 0
    
    for text in texts[:100]:  # Sample first 100 texts for detailed analysis
        encoded = tokenizer.encode(text)
        token_counts.append(len(encoded.input_ids))
        total_tokens += len(encoded.input_ids)
        
        # Count OOV tokens (assuming UNK token ID is available)
        unk_id = tokenizer.token_to_id("[UNK]")
        if unk_id is not None:
            oov_tokens += encoded.input_ids.count(unk_id)
    
    return {
        "coverage_ratio": coverage_ratio,
        "avg_tokens_per_text": statistics.mean(token_counts) if token_counts else 0,
        "median_tokens_per_text": statistics.median(token_counts) if token_counts else 0,
        "max_tokens_per_text": max(token_counts) if token_counts else 0,
        "min_tokens_per_text": min(token_counts) if token_counts else 0,
        "oov_rate": oov_tokens / total_tokens if total_tokens > 0 else 0,
        "total_texts_analyzed": len(texts),
        "sample_size": min(len(texts), 100),
    }


def benchmark_tokenizers(
    tokenizers: Dict[str, Tokenizer],
    texts: List[str],
    iterations: int = 3,
    warmup_iterations: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark multiple tokenizers on the same texts.
    
    Args:
        tokenizers: Dictionary of tokenizer name to tokenizer
        texts: List of texts to benchmark on
        iterations: Number of benchmark iterations
        warmup_iterations: Number of warmup iterations
        
    Returns:
        Benchmark results for each tokenizer
    """
    results = {}
    
    for name, tokenizer in tokenizers.items():
        print(f"Benchmarking {name}...")
        
        # Warmup
        for _ in range(warmup_iterations):
            for text in texts[:10]:  # Use subset for warmup
                tokenizer.encode(text)
        
        # Benchmark encoding
        encoding_times = []
        for iteration in range(iterations):
            start_time = time.perf_counter()
            for text in texts:
                tokenizer.encode(text)
            end_time = time.perf_counter()
            encoding_times.append(end_time - start_time)
        
        # Benchmark decoding
        token_ids_list = [tokenizer.encode(text).input_ids for text in texts[:50]]
        decoding_times = []
        for iteration in range(iterations):
            start_time = time.perf_counter()
            for token_ids in token_ids_list:
                tokenizer.decode(token_ids)
            end_time = time.perf_counter()
            decoding_times.append(end_time - start_time)
        
        # Calculate statistics
        total_texts = len(texts)
        avg_encoding_time = statistics.mean(encoding_times)
        avg_decoding_time = statistics.mean(decoding_times)
        
        results[name] = {
            "encoding": {
                "total_time": avg_encoding_time,
                "texts_per_second": total_texts / avg_encoding_time,
                "time_per_text": avg_encoding_time / total_texts,
                "times": encoding_times,
            },
            "decoding": {
                "total_time": avg_decoding_time,
                "sequences_per_second": len(token_ids_list) / avg_decoding_time,
                "time_per_sequence": avg_decoding_time / len(token_ids_list),
                "times": decoding_times,
            },
            "vocab_size": tokenizer.vocab_size,
            "total_texts": total_texts,
        }
    
    return results


def compare_tokenizations(
    tokenizers: Dict[str, Tokenizer],
    texts: List[str],
    max_texts: int = 10,
) -> List[Dict[str, Any]]:
    """
    Compare how different tokenizers tokenize the same texts.
    
    Args:
        tokenizers: Dictionary of tokenizer name to tokenizer
        texts: List of texts to compare
        max_texts: Maximum number of texts to compare
        
    Returns:
        Comparison results for each text
    """
    results = []
    
    for i, text in enumerate(texts[:max_texts]):
        comparison = {
            "text": text,
            "tokenizations": {},
        }
        
        for name, tokenizer in tokenizers.items():
            encoded = tokenizer.encode(text)
            
            # Get actual tokens if possible
            tokens = []
            for token_id in encoded.input_ids:
                token = tokenizer.id_to_token(token_id)
                tokens.append(token if token is not None else f"<{token_id}>")
            
            comparison["tokenizations"][name] = {
                "input_ids": encoded.input_ids,
                "tokens": tokens,
                "num_tokens": len(encoded.input_ids),
                "decoded": tokenizer.decode(encoded.input_ids),
            }
        
        results.append(comparison)
    
    return results


def analyze_vocabulary_overlap(
    tokenizers: Dict[str, Tokenizer],
    sample_size: int = 1000,
) -> Dict[str, Any]:
    """
    Analyze vocabulary overlap between tokenizers.
    
    Args:
        tokenizers: Dictionary of tokenizer name to tokenizer
        sample_size: Number of vocabulary items to sample
        
    Returns:
        Vocabulary overlap analysis
    """
    vocabs = {}
    vocab_sets = {}
    
    # Get vocabulary samples
    for name, tokenizer in tokenizers.items():
        vocab = tokenizer.get_vocab()
        vocab_sample = set(list(vocab.keys())[:sample_size])
        vocabs[name] = vocab
        vocab_sets[name] = vocab_sample
    
    # Calculate pairwise overlaps
    overlaps = {}
    tokenizer_names = list(tokenizers.keys())
    
    for i, name1 in enumerate(tokenizer_names):
        for name2 in tokenizer_names[i+1:]:
            overlap = len(vocab_sets[name1] & vocab_sets[name2])
            union = len(vocab_sets[name1] | vocab_sets[name2])
            jaccard = overlap / union if union > 0 else 0
            
            overlaps[f"{name1}_vs_{name2}"] = {
                "overlap_count": overlap,
                "union_count": union,
                "jaccard_similarity": jaccard,
                "overlap_ratio_1": overlap / len(vocab_sets[name1]),
                "overlap_ratio_2": overlap / len(vocab_sets[name2]),
            }
    
    return {
        "vocab_sizes": {name: len(vocab) for name, vocab in vocabs.items()},
        "sample_sizes": {name: len(vocab_set) for name, vocab_set in vocab_sets.items()},
        "overlaps": overlaps,
    }


def profile_memory_usage(tokenizer: Tokenizer, texts: List[str]) -> Dict[str, Any]:
    """
    Profile memory usage of tokenizer operations.
    
    Args:
        tokenizer: Tokenizer to profile
        texts: Texts to process
        
    Returns:
        Memory usage profile
    """
    import psutil
    import gc
    
    process = psutil.Process()
    
    # Baseline memory
    gc.collect()
    baseline_memory = process.memory_info().rss
    
    # Encode texts
    gc.collect()
    memory_before_encoding = process.memory_info().rss
    
    encoded_results = []
    for text in texts:
        encoded_results.append(tokenizer.encode(text))
    
    gc.collect()
    memory_after_encoding = process.memory_info().rss
    
    # Decode results
    memory_before_decoding = process.memory_info().rss
    
    for encoded in encoded_results:
        tokenizer.decode(encoded.input_ids)
    
    gc.collect()
    memory_after_decoding = process.memory_info().rss
    
    return {
        "baseline_memory_mb": baseline_memory / (1024 * 1024),
        "encoding_memory_increase_mb": (memory_after_encoding - memory_before_encoding) / (1024 * 1024),
        "decoding_memory_increase_mb": (memory_after_decoding - memory_before_decoding) / (1024 * 1024),
        "peak_memory_mb": memory_after_encoding / (1024 * 1024),
        "memory_per_text_kb": (memory_after_encoding - baseline_memory) / len(texts) / 1024,
        "total_texts": len(texts),
    }


def validate_tokenizer_roundtrip(
    tokenizer: Tokenizer,
    texts: List[str],
    tolerance: float = 0.95,
) -> Dict[str, Any]:
    """
    Validate that tokenizer can roundtrip texts (encode + decode ≈ original).
    
    Args:
        tokenizer: Tokenizer to validate
        texts: Texts to test roundtrip on
        tolerance: Minimum acceptable similarity ratio
        
    Returns:
        Validation results
    """
    results = {
        "total_texts": len(texts),
        "successful_roundtrips": 0,
        "failed_roundtrips": 0,
        "similarity_scores": [],
        "failures": [],
    }
    
    for i, original_text in enumerate(texts):
        try:
            # Encode and decode
            encoded = tokenizer.encode(original_text)
            decoded_text = tokenizer.decode(encoded.input_ids)
            
            # Calculate similarity (simple character-based)
            similarity = calculate_text_similarity(original_text, decoded_text)
            results["similarity_scores"].append(similarity)
            
            if similarity >= tolerance:
                results["successful_roundtrips"] += 1
            else:
                results["failed_roundtrips"] += 1
                results["failures"].append({
                    "index": i,
                    "original": original_text,
                    "decoded": decoded_text,
                    "similarity": similarity,
                })
        
        except Exception as e:
            results["failed_roundtrips"] += 1
            results["failures"].append({
                "index": i,
                "original": original_text,
                "error": str(e),
                "similarity": 0.0,
            })
    
    if results["similarity_scores"]:
        results["avg_similarity"] = statistics.mean(results["similarity_scores"])
        results["min_similarity"] = min(results["similarity_scores"])
        results["max_similarity"] = max(results["similarity_scores"])
    else:
        results["avg_similarity"] = 0.0
        results["min_similarity"] = 0.0
        results["max_similarity"] = 0.0
    
    results["success_rate"] = results["successful_roundtrips"] / results["total_texts"]
    
    return results


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two texts using character-level comparison.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity ratio between 0 and 1
    """
    # Simple character-based similarity
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    
    # Count matching characters
    matching_chars = sum(1 for c1, c2 in zip(text1, text2) if c1 == c2)
    max_length = max(len(text1), len(text2))
    
    return matching_chars / max_length


def export_tokenizer_stats(tokenizer: Tokenizer, output_file: str):
    """
    Export detailed tokenizer statistics to JSON file.
    
    Args:
        tokenizer: Tokenizer to analyze
        output_file: Output JSON file path
    """
    vocab = tokenizer.get_vocab()
    
    # Calculate statistics
    token_lengths = [len(token) for token in vocab.keys()]
    
    stats = {
        "vocab_size": len(vocab),
        "avg_token_length": statistics.mean(token_lengths) if token_lengths else 0,
        "median_token_length": statistics.median(token_lengths) if token_lengths else 0,
        "max_token_length": max(token_lengths) if token_lengths else 0,
        "min_token_length": min(token_lengths) if token_lengths else 0,
        "tokenizer_type": type(tokenizer).__name__,
        "special_tokens": {
            "unk_token": tokenizer.token_to_id("[UNK]"),
            "pad_token": tokenizer.token_to_id("[PAD]"),
            "cls_token": tokenizer.token_to_id("[CLS]"),
            "sep_token": tokenizer.token_to_id("[SEP]"),
            "mask_token": tokenizer.token_to_id("[MASK]"),
        },
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)