#!/usr/bin/env python3
"""
Example usage of TrustformeRS Tokenizers Python bindings.

This script demonstrates various features of the tokenizers library
including loading pre-trained tokenizers, encoding/decoding text,
training custom tokenizers, and performance analysis.
"""

import os
import sys
import time
from typing import List

# Add the package to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    import trustformers_tokenizers as tt
    from trustformers_tokenizers import (
        AutoTokenizer,
        BPETokenizer,
        WordPieceTokenizer,
        CharTokenizer,
        TokenizerTrainer,
        analyze_coverage,
        benchmark_tokenizers,
        compare_tokenizations,
    )
except ImportError as e:
    print(f"Error importing trustformers_tokenizers: {e}")
    print("Make sure the package is built with: maturin develop --features python")
    sys.exit(1)


def demo_basic_usage():
    """Demonstrate basic tokenizer usage."""
    print("=== Basic Tokenizer Usage ===")
    
    # Create a simple character tokenizer
    tokenizer = CharTokenizer(lowercase=True, max_length=512)
    
    # Test texts
    texts = [
        "Hello, world! How are you today?",
        "TrustformeRS provides fast tokenization.",
        "Machine learning is fascinating! 🤖",
    ]
    
    for text in texts:
        # Encode
        encoded = tokenizer.encode(text)
        print(f"Original: {text}")
        print(f"Tokens: {encoded.input_ids}")
        print(f"Length: {len(encoded)}")
        
        # Decode
        decoded = tokenizer.decode(encoded.input_ids)
        print(f"Decoded: {decoded}")
        print(f"Roundtrip successful: {text.lower() == decoded}")
        print()


def demo_tokenizer_comparison():
    """Compare different tokenizer types."""
    print("=== Tokenizer Comparison ===")
    
    # Create different tokenizers
    tokenizers = {
        "char": CharTokenizer(lowercase=False),
        "char_lower": CharTokenizer(lowercase=True),
    }
    
    # Test text
    test_text = "Hello, TrustformeRS! How are you today?"
    
    # Compare tokenizations
    comparison = compare_tokenizations(tokenizers, [test_text])
    
    for result in comparison:
        print(f"Text: {result['text']}")
        for name, tokenization in result['tokenizations'].items():
            print(f"  {name:12}: {tokenization['num_tokens']} tokens")
            print(f"  {' ' * 12}  {tokenization['tokens']}")
        print()


def demo_coverage_analysis():
    """Demonstrate text coverage analysis."""
    print("=== Coverage Analysis ===")
    
    tokenizer = CharTokenizer(lowercase=True)
    
    # Sample texts for analysis
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require large datasets.",
        "Natural language processing is a subfield of AI.",
        "Tokenization is the first step in text processing.",
        "Python is a popular programming language.",
    ]
    
    # Analyze coverage
    coverage = analyze_coverage(tokenizer, texts)
    
    print(f"Coverage Analysis Results:")
    print(f"  Coverage ratio: {coverage['coverage_ratio']:.3f}")
    print(f"  Average tokens per text: {coverage['avg_tokens_per_text']:.1f}")
    print(f"  Median tokens per text: {coverage['median_tokens_per_text']:.1f}")
    print(f"  Token range: {coverage['min_tokens_per_text']} - {coverage['max_tokens_per_text']}")
    print(f"  OOV rate: {coverage['oov_rate']:.3f}")
    print(f"  Texts analyzed: {coverage['total_texts_analyzed']}")
    print()


def demo_benchmarking():
    """Demonstrate tokenizer benchmarking."""
    print("=== Tokenizer Benchmarking ===")
    
    # Create tokenizers to benchmark
    tokenizers = {
        "char": CharTokenizer(lowercase=False),
        "char_lower": CharTokenizer(lowercase=True),
    }
    
    # Generate test texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming various industries.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models have achieved remarkable results.",
        "Artificial intelligence is becoming increasingly important.",
    ] * 20  # Multiply for meaningful benchmark
    
    print(f"Benchmarking with {len(texts)} texts...")
    
    # Run benchmark
    results = benchmark_tokenizers(tokenizers, texts, iterations=3)
    
    # Display results
    for name, result in results.items():
        print(f"\n{name} Tokenizer:")
        print(f"  Encoding: {result['encoding']['texts_per_second']:.1f} texts/sec")
        print(f"  Decoding: {result['decoding']['sequences_per_second']:.1f} sequences/sec")
        print(f"  Vocab size: {result['vocab_size']}")
        print(f"  Time per text: {result['encoding']['time_per_text']*1000:.2f} ms")


def demo_custom_tokenizer():
    """Demonstrate creating custom tokenizers."""
    print("=== Custom Tokenizer Creation ===")
    
    # Create a simple vocabulary
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "hello": 4,
        "world": 5,
        "the": 6,
        "quick": 7,
        "brown": 8,
        "fox": 9,
    }
    
    # Create WordPiece tokenizer
    tokenizer = WordPieceTokenizer(vocab, unk_token="[UNK]")
    
    # Test the custom tokenizer
    test_texts = [
        "hello world",
        "the quick brown fox",
        "unknown words here",
    ]
    
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.input_ids)
        print(f"Text: {text}")
        print(f"Tokens: {encoded.input_ids}")
        print(f"Decoded: {decoded}")
        print()


def demo_training():
    """Demonstrate tokenizer training (simplified)."""
    print("=== Tokenizer Training Demo ===")
    
    # Sample training data
    training_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing helps computers understand human language.",
        "Deep learning models require large amounts of training data.",
        "Python is widely used for machine learning and data science.",
        "Tokenization is an essential preprocessing step in NLP.",
        "Transformers have revolutionized natural language understanding.",
        "BERT and GPT are popular transformer-based models.",
        "Fine-tuning pre-trained models often yields good results.",
        "Data preprocessing is crucial for model performance.",
    ]
    
    # Create trainer
    trainer = TokenizerTrainer(
        vocab_size=1000,
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"],
        show_progress=True,
    )
    
    try:
        # Train a BPE tokenizer
        print("Training BPE tokenizer...")
        bpe_tokenizer = trainer.train_bpe(texts=training_texts)
        
        # Test the trained tokenizer
        test_text = "Machine learning with transformers is powerful."
        encoded = bpe_tokenizer.encode(test_text)
        decoded = bpe_tokenizer.decode(encoded.input_ids)
        
        print(f"Test text: {test_text}")
        print(f"Encoded: {encoded.input_ids}")
        print(f"Decoded: {decoded}")
        print(f"Vocab size: {bpe_tokenizer.vocab_size}")
        
    except Exception as e:
        print(f"Training demo failed (expected with mock implementation): {e}")


def demo_advanced_features():
    """Demonstrate advanced features."""
    print("=== Advanced Features ===")
    
    tokenizer = CharTokenizer(lowercase=True)
    
    # Batch processing
    texts = [
        "First text for batch processing.",
        "Second text in the batch.",
        "Third and final text.",
    ]
    
    print("Batch encoding:")
    batch_encoded = tokenizer.encode_batch(texts)
    for i, encoded in enumerate(batch_encoded):
        print(f"  Text {i+1}: {len(encoded)} tokens")
    
    # Batch decoding
    token_ids_batch = [encoded.input_ids for encoded in batch_encoded]
    batch_decoded = tokenizer.decode_batch(token_ids_batch)
    
    print("\nBatch decoding:")
    for i, decoded in enumerate(batch_decoded):
        print(f"  Text {i+1}: {decoded}")
    
    # Vocabulary exploration
    print(f"\nVocabulary info:")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    
    # Sample some token-to-id mappings
    sample_tokens = ["a", "e", "t", " ", ".", "!"]
    print(f"  Sample token mappings:")
    for token in sample_tokens:
        token_id = tokenizer.token_to_id(token)
        if token_id is not None:
            back_token = tokenizer.id_to_token(token_id)
            print(f"    '{token}' -> {token_id} -> '{back_token}'")


def main():
    """Run all demos."""
    print("TrustformeRS Tokenizers Python Bindings Demo")
    print("=" * 50)
    
    try:
        demo_basic_usage()
        demo_tokenizer_comparison()
        demo_coverage_analysis()
        demo_benchmarking()
        demo_custom_tokenizer()
        demo_training()
        demo_advanced_features()
        
        print("All demos completed successfully! 🎉")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())