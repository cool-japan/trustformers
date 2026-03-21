"""
TrustformeRS Tokenizers - High-performance tokenizers for transformer models

This package provides fast, memory-efficient tokenizers implemented in Rust
with Python bindings. It supports various tokenization algorithms including
BPE, WordPiece, Unigram, and character-level tokenization.

Features:
- High-performance tokenization algorithms
- Memory-efficient implementations
- Support for multiple tokenizer types
- Training from custom corpora
- Compatibility with HuggingFace tokenizers
- Advanced features like subword regularization
- Comprehensive text analysis tools
"""

__version__ = "0.1.0"
__author__ = "TrustformeRS Team"

# Import the Rust extension module
try:
    from .trustformers_tokenizers import *
except ImportError as e:
    raise ImportError(
        "Failed to import the Rust extension module. "
        "Make sure the package was built with the 'python' feature enabled. "
        f"Error: {e}"
    ) from e

# High-level Python API
from .tokenizers import (
    Tokenizer,
    BPETokenizer,
    WordPieceTokenizer,
    UnigramTokenizer,
    CharTokenizer,
    AutoTokenizer,
)
from .training import TokenizerTrainer
from .utils import (
    load_tokenizer,
    save_tokenizer,
    analyze_coverage,
    benchmark_tokenizers,
)

__all__ = [
    # Core classes
    "Tokenizer",
    "BPETokenizer", 
    "WordPieceTokenizer",
    "UnigramTokenizer",
    "CharTokenizer",
    "AutoTokenizer",
    "TokenizerTrainer",
    # Utility functions
    "load_tokenizer",
    "save_tokenizer",
    "analyze_coverage",
    "benchmark_tokenizers",
    # From Rust module
    "PyTokenizedInput",
    "PyTokenizer",
    "PyBPETokenizer",
    "PyWordPieceTokenizer",
    "PyUnigramTokenizer",
    "PyCharTokenizer",
    "PyTokenizerTrainer",
]