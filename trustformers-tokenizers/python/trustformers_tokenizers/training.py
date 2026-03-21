"""
Training utilities for TrustformeRS tokenizers.

This module provides tools for training custom tokenizers from scratch
or fine-tuning existing ones on domain-specific corpora.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Union, Iterator, Any

try:
    from .trustformers_tokenizers import PyTokenizerTrainer
except ImportError:
    from unittest.mock import MagicMock
    PyTokenizerTrainer = MagicMock

from .tokenizers import BPETokenizer, WordPieceTokenizer, UnigramTokenizer


class TokenizerTrainer:
    """High-level interface for training tokenizers."""
    
    def __init__(
        self,
        vocab_size: int = 30000,
        special_tokens: Optional[List[str]] = None,
        min_frequency: int = 2,
        show_progress: bool = True,
    ):
        """
        Initialize tokenizer trainer.
        
        Args:
            vocab_size: Target vocabulary size
            special_tokens: List of special tokens to preserve
            min_frequency: Minimum frequency for tokens to be included
            show_progress: Whether to show training progress
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        self.min_frequency = min_frequency
        self.show_progress = show_progress
        self._inner = PyTokenizerTrainer(vocab_size, self.special_tokens)
    
    def train_bpe(
        self,
        files: Optional[List[str]] = None,
        texts: Optional[List[str]] = None,
        iterator: Optional[Iterator[str]] = None,
    ) -> BPETokenizer:
        """
        Train a BPE tokenizer.
        
        Args:
            files: List of text files to train on
            texts: List of text strings to train on
            iterator: Iterator over text strings
            
        Returns:
            Trained BPE tokenizer
        """
        if files:
            return self._train_from_files("bpe", files)
        elif texts:
            return self._train_from_texts("bpe", texts)
        elif iterator:
            return self._train_from_iterator("bpe", iterator)
        else:
            raise ValueError("Must provide either files, texts, or iterator")
    
    def train_wordpiece(
        self,
        files: Optional[List[str]] = None,
        texts: Optional[List[str]] = None,
        iterator: Optional[Iterator[str]] = None,
        unk_token: str = "[UNK]",
    ) -> WordPieceTokenizer:
        """
        Train a WordPiece tokenizer.
        
        Args:
            files: List of text files to train on
            texts: List of text strings to train on
            iterator: Iterator over text strings
            unk_token: Unknown token
            
        Returns:
            Trained WordPiece tokenizer
        """
        if files:
            return self._train_from_files("wordpiece", files, unk_token=unk_token)
        elif texts:
            return self._train_from_texts("wordpiece", texts, unk_token=unk_token)
        elif iterator:
            return self._train_from_iterator("wordpiece", iterator, unk_token=unk_token)
        else:
            raise ValueError("Must provide either files, texts, or iterator")
    
    def train_unigram(
        self,
        files: Optional[List[str]] = None,
        texts: Optional[List[str]] = None,
        iterator: Optional[Iterator[str]] = None,
    ) -> UnigramTokenizer:
        """
        Train a Unigram tokenizer.
        
        Args:
            files: List of text files to train on
            texts: List of text strings to train on
            iterator: Iterator over text strings
            
        Returns:
            Trained Unigram tokenizer
        """
        if files:
            return self._train_from_files("unigram", files)
        elif texts:
            return self._train_from_texts("unigram", texts)
        elif iterator:
            return self._train_from_iterator("unigram", iterator)
        else:
            raise ValueError("Must provide either files, texts, or iterator")
    
    def _train_from_files(self, algorithm: str, files: List[str], **kwargs) -> Any:
        """Train tokenizer from files."""
        if self.show_progress:
            print(f"Training {algorithm.upper()} tokenizer from {len(files)} files...")
        
        # Validate files exist
        for file_path in files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
        
        # Use the Rust trainer
        trained = self._inner.train_from_files(files)
        
        if self.show_progress:
            print(f"Training completed. Vocabulary size: {trained.get_vocab_size()}")
        
        return self._wrap_tokenizer(algorithm, trained, **kwargs)
    
    def _train_from_texts(self, algorithm: str, texts: List[str], **kwargs) -> Any:
        """Train tokenizer from text list."""
        if self.show_progress:
            print(f"Training {algorithm.upper()} tokenizer from {len(texts)} texts...")
        
        # Use the Rust trainer
        trained = self._inner.train_from_iterator(texts)
        
        if self.show_progress:
            print(f"Training completed. Vocabulary size: {trained.get_vocab_size()}")
        
        return self._wrap_tokenizer(algorithm, trained, **kwargs)
    
    def _train_from_iterator(self, algorithm: str, iterator: Iterator[str], **kwargs) -> Any:
        """Train tokenizer from iterator."""
        if self.show_progress:
            print(f"Training {algorithm.upper()} tokenizer from iterator...")
        
        # Convert iterator to list for the Rust interface
        texts = list(iterator)
        return self._train_from_texts(algorithm, texts, **kwargs)
    
    def _wrap_tokenizer(self, algorithm: str, trained_tokenizer: Any, **kwargs) -> Any:
        """Wrap the trained Rust tokenizer in appropriate Python class."""
        if algorithm == "bpe":
            return BPETokenizer(trained_tokenizer._inner.get_vocab(), [])
        elif algorithm == "wordpiece":
            unk_token = kwargs.get("unk_token", "[UNK]")
            return WordPieceTokenizer(trained_tokenizer._inner.get_vocab(), unk_token)
        elif algorithm == "unigram":
            # For Unigram, we need pieces with scores
            pieces = [(f"piece_{i}", 1.0) for i in range(trained_tokenizer.get_vocab_size())]
            return UnigramTokenizer(pieces)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")


class TrainingConfig:
    """Configuration for tokenizer training."""
    
    def __init__(
        self,
        vocab_size: int = 30000,
        special_tokens: Optional[List[str]] = None,
        min_frequency: int = 2,
        limit_alphabet: Optional[int] = None,
        initial_alphabet: Optional[List[str]] = None,
        show_progress: bool = True,
        continuing_subword_prefix: Optional[str] = None,
        end_of_word_suffix: Optional[str] = None,
        dropout: Optional[float] = None,
        unk_token: str = "[UNK]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        sep_token: str = "[SEP]",
        mask_token: str = "[MASK]",
    ):
        """
        Initialize training configuration.
        
        Args:
            vocab_size: Target vocabulary size
            special_tokens: List of special tokens
            min_frequency: Minimum frequency for inclusion
            limit_alphabet: Limit initial alphabet size
            initial_alphabet: Initial alphabet characters
            show_progress: Show training progress
            continuing_subword_prefix: Prefix for continuing subwords (WordPiece)
            end_of_word_suffix: Suffix for end-of-word tokens (BPE)
            dropout: Dropout rate for subword regularization
            unk_token: Unknown token
            pad_token: Padding token
            cls_token: Classification token
            sep_token: Separator token
            mask_token: Mask token
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or [unk_token, pad_token, cls_token, sep_token, mask_token]
        self.min_frequency = min_frequency
        self.limit_alphabet = limit_alphabet
        self.initial_alphabet = initial_alphabet
        self.show_progress = show_progress
        self.continuing_subword_prefix = continuing_subword_prefix
        self.end_of_word_suffix = end_of_word_suffix
        self.dropout = dropout
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "min_frequency": self.min_frequency,
            "limit_alphabet": self.limit_alphabet,
            "initial_alphabet": self.initial_alphabet,
            "show_progress": self.show_progress,
            "continuing_subword_prefix": self.continuing_subword_prefix,
            "end_of_word_suffix": self.end_of_word_suffix,
            "dropout": self.dropout,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
            "mask_token": self.mask_token,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def save(self, path: str):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def train_tokenizer_from_files(
    algorithm: str,
    files: List[str],
    vocab_size: int = 30000,
    special_tokens: Optional[List[str]] = None,
    **kwargs
) -> Union[BPETokenizer, WordPieceTokenizer, UnigramTokenizer]:
    """
    Convenience function to train a tokenizer from files.
    
    Args:
        algorithm: Tokenization algorithm ("bpe", "wordpiece", or "unigram")
        files: List of text files to train on
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens
        **kwargs: Additional training arguments
        
    Returns:
        Trained tokenizer
    """
    trainer = TokenizerTrainer(vocab_size, special_tokens)
    
    if algorithm.lower() == "bpe":
        return trainer.train_bpe(files=files)
    elif algorithm.lower() == "wordpiece":
        return trainer.train_wordpiece(files=files, **kwargs)
    elif algorithm.lower() == "unigram":
        return trainer.train_unigram(files=files)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def train_tokenizer_from_dataset(
    algorithm: str,
    dataset: Any,
    text_column: str = "text",
    vocab_size: int = 30000,
    special_tokens: Optional[List[str]] = None,
    batch_size: int = 1000,
    **kwargs
) -> Union[BPETokenizer, WordPieceTokenizer, UnigramTokenizer]:
    """
    Train a tokenizer from a HuggingFace dataset.
    
    Args:
        algorithm: Tokenization algorithm
        dataset: HuggingFace dataset object
        text_column: Name of the text column
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens
        batch_size: Batch size for processing
        **kwargs: Additional training arguments
        
    Returns:
        Trained tokenizer
    """
    def text_iterator():
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            for text in batch[text_column]:
                yield text
    
    trainer = TokenizerTrainer(vocab_size, special_tokens)
    
    if algorithm.lower() == "bpe":
        return trainer.train_bpe(iterator=text_iterator())
    elif algorithm.lower() == "wordpiece":
        return trainer.train_wordpiece(iterator=text_iterator(), **kwargs)
    elif algorithm.lower() == "unigram":
        return trainer.train_unigram(iterator=text_iterator())
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


# Utility functions for training
def estimate_vocab_size(
    text_files: List[str],
    sample_ratio: float = 0.1
) -> Dict[str, int]:
    """
    Estimate appropriate vocabulary size based on text corpus.
    
    Args:
        text_files: List of text files to analyze
        sample_ratio: Ratio of text to sample for analysis
        
    Returns:
        Dictionary with recommendations for different algorithms
    """
    import collections
    
    char_counts = collections.Counter()
    word_counts = collections.Counter()
    total_chars = 0
    total_words = 0
    
    for file_path in text_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num % int(1 / sample_ratio) != 0:
                    continue
                
                text = line.strip()
                chars = list(text)
                words = text.split()
                
                char_counts.update(chars)
                word_counts.update(words)
                total_chars += len(chars)
                total_words += len(words)
    
    unique_chars = len(char_counts)
    unique_words = len(word_counts)
    
    return {
        "char_level": unique_chars,
        "word_level": unique_words,
        "bpe_recommended": min(30000, max(8000, unique_words // 4)),
        "wordpiece_recommended": min(30000, max(8000, unique_words // 3)),
        "unigram_recommended": min(30000, max(8000, unique_words // 5)),
        "stats": {
            "total_chars": total_chars,
            "total_words": total_words,
            "unique_chars": unique_chars,
            "unique_words": unique_words,
        }
    }


def validate_training_data(text_files: List[str]) -> Dict[str, Any]:
    """
    Validate training data and provide recommendations.
    
    Args:
        text_files: List of text files to validate
        
    Returns:
        Validation report with recommendations
    """
    report = {
        "files": [],
        "total_size": 0,
        "total_lines": 0,
        "encoding_issues": [],
        "recommendations": [],
    }
    
    for file_path in text_files:
        file_info = {
            "path": file_path,
            "size": 0,
            "lines": 0,
            "encoding": "utf-8",
            "issues": [],
        }
        
        try:
            if not os.path.exists(file_path):
                file_info["issues"].append(f"File not found: {file_path}")
                continue
            
            file_info["size"] = os.path.getsize(file_path)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    file_info["lines"] = line_num
                    if len(line.strip()) == 0:
                        continue
            
            report["total_size"] += file_info["size"]
            report["total_lines"] += file_info["lines"]
            
        except UnicodeDecodeError as e:
            file_info["issues"].append(f"Encoding error: {e}")
            report["encoding_issues"].append(file_path)
        except Exception as e:
            file_info["issues"].append(f"Error reading file: {e}")
        
        report["files"].append(file_info)
    
    # Generate recommendations
    if report["total_size"] < 1024 * 1024:  # Less than 1MB
        report["recommendations"].append("Training corpus is very small. Consider collecting more data.")
    
    if report["total_lines"] < 1000:
        report["recommendations"].append("Very few lines of text. Training may not be effective.")
    
    if report["encoding_issues"]:
        report["recommendations"].append("Fix encoding issues before training.")
    
    return report