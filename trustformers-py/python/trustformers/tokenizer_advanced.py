"""
Advanced tokenizer features for TrustformeRS

Provides fast tokenizers, training capabilities, offset mapping, and alignment tracking.
"""

from typing import List, Dict, Optional, Union, Tuple, Any, Iterator, Set, NamedTuple
import numpy as np
import warnings
import json
import os
import re
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import time


@dataclass
class OffsetMapping:
    """Stores character offset mappings for tokens."""
    char_to_word: List[Optional[int]]
    word_to_chars: List[Tuple[int, int]]
    char_to_token: List[Optional[int]]
    token_to_chars: List[Tuple[int, int]]


@dataclass
class SpecialTokens:
    """Configuration for special tokens."""
    pad_token: str = "[PAD]"
    unk_token: str = "[UNK]"
    cls_token: str = "[CLS]"
    sep_token: str = "[SEP]"
    mask_token: str = "[MASK]"
    bos_token: str = "<s>"
    eos_token: str = "</s>"
    additional_special_tokens: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Union[str, List[str]]]:
        return {
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
            "mask_token": self.mask_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "additional_special_tokens": self.additional_special_tokens,
        }


class TokenizerTrainingCorpus:
    """Corpus iterator for tokenizer training."""
    
    def __init__(self, files: List[str], batch_size: int = 1000):
        self.files = files
        self.batch_size = batch_size
    
    def __iter__(self) -> Iterator[List[str]]:
        batch = []
        for file_path in self.files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        batch.append(line)
                        if len(batch) >= self.batch_size:
                            yield batch
                            batch = []
        if batch:
            yield batch


class AlignmentTracker:
    """Tracks alignment between different tokenization levels."""
    
    def __init__(self):
        self.word_ids: List[Optional[int]] = []
        self.token_to_word: Dict[int, int] = {}
        self.word_to_tokens: Dict[int, List[int]] = defaultdict(list)
        self.char_to_token: List[Optional[int]] = []
        self.token_to_chars: List[Tuple[int, int]] = []
    
    def add_token(self, token_id: int, word_id: Optional[int], char_start: int, char_end: int):
        """Add a token with its alignment information."""
        self.word_ids.append(word_id)
        if word_id is not None:
            self.token_to_word[token_id] = word_id
            self.word_to_tokens[word_id].append(token_id)
        
        # Update character-to-token mapping
        for i in range(char_start, char_end):
            if i < len(self.char_to_token):
                self.char_to_token[i] = token_id
            else:
                self.char_to_token.extend([None] * (i - len(self.char_to_token)))
                self.char_to_token.append(token_id)
        
        self.token_to_chars.append((char_start, char_end))
    
    def get_word_ids(self) -> List[Optional[int]]:
        """Get word IDs for each token."""
        return self.word_ids.copy()
    
    def get_tokens_for_word(self, word_id: int) -> List[int]:
        """Get all token IDs for a given word."""
        return self.word_to_tokens.get(word_id, [])
    
    def get_word_for_token(self, token_id: int) -> Optional[int]:
        """Get word ID for a given token."""
        return self.token_to_word.get(token_id)
    
    def get_char_span_for_token(self, token_id: int) -> Optional[Tuple[int, int]]:
        """Get character span for a given token."""
        if token_id < len(self.token_to_chars):
            return self.token_to_chars[token_id]
        return None


class TokenizerTrainer:
    """Utilities for training tokenizers from scratch."""
    
    @staticmethod
    def train_wordpiece(
        corpus: TokenizerTrainingCorpus,
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: Optional[SpecialTokens] = None,
        unk_token: str = "[UNK]",
        continuing_subword_prefix: str = "##",
        max_input_chars_per_word: int = 100,
    ) -> Dict[str, int]:
        """Train a WordPiece tokenizer."""
        if special_tokens is None:
            special_tokens = SpecialTokens()
        
        # Initialize vocabulary with special tokens
        vocab = {}
        for token in [special_tokens.pad_token, special_tokens.unk_token,
                     special_tokens.cls_token, special_tokens.sep_token,
                     special_tokens.mask_token] + special_tokens.additional_special_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
        
        # Collect character statistics
        char_counts = Counter()
        word_counts = Counter()
        
        for batch in corpus:
            for text in batch:
                words = text.split()
                for word in words:
                    if len(word) <= max_input_chars_per_word:
                        word_counts[word] += 1
                        for char in word:
                            char_counts[char] += 1
        
        # Add frequent characters to vocabulary
        for char, count in char_counts.most_common():
            if count >= min_frequency and len(vocab) < vocab_size:
                vocab[char] = len(vocab)
        
        # WordPiece algorithm (simplified)
        while len(vocab) < vocab_size:
            pairs = defaultdict(int)
            
            # Count adjacent pairs
            for word, count in word_counts.items():
                if count >= min_frequency:
                    subwords = list(word)
                    for i in range(len(subwords) - 1):
                        pair = (subwords[i], subwords[i + 1])
                        pairs[pair] += count
            
            if not pairs:
                break
            
            # Find best pair to merge
            best_pair = max(pairs, key=pairs.get)
            new_token = best_pair[0] + best_pair[1]
            
            if new_token.startswith(continuing_subword_prefix):
                vocab[new_token] = len(vocab)
            else:
                vocab[continuing_subword_prefix + new_token] = len(vocab)
            
            # Update word representations
            new_word_counts = Counter()
            for word, count in word_counts.items():
                new_word = word.replace(best_pair[0] + best_pair[1], new_token)
                new_word_counts[new_word] = count
            word_counts = new_word_counts
        
        return vocab
    
    @staticmethod
    def train_bpe(
        corpus: TokenizerTrainingCorpus,
        vocab_size: int = 30000,
        min_frequency: int = 2,
        special_tokens: Optional[SpecialTokens] = None,
    ) -> Dict[str, int]:
        """Train a BPE (Byte-Pair Encoding) tokenizer."""
        if special_tokens is None:
            special_tokens = SpecialTokens()
        
        # Initialize vocabulary with special tokens
        vocab = {}
        for token in [special_tokens.pad_token, special_tokens.unk_token,
                     special_tokens.cls_token, special_tokens.sep_token,
                     special_tokens.mask_token] + special_tokens.additional_special_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
        
        # Collect word statistics
        word_counts = Counter()
        for batch in corpus:
            for text in batch:
                words = text.split()
                for word in words:
                    word_counts[word] += 1
        
        # Initialize with characters
        chars = set()
        for word in word_counts:
            chars.update(word)
        
        for char in sorted(chars):
            vocab[char] = len(vocab)
        
        # BPE algorithm
        while len(vocab) < vocab_size:
            pairs = defaultdict(int)
            
            # Count adjacent pairs
            for word, count in word_counts.items():
                if count >= min_frequency:
                    chars = list(word)
                    for i in range(len(chars) - 1):
                        pair = (chars[i], chars[i + 1])
                        pairs[pair] += count
            
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            new_token = best_pair[0] + best_pair[1]
            vocab[new_token] = len(vocab)
            
            # Update word representations
            new_word_counts = Counter()
            for word, count in word_counts.items():
                new_word = word.replace(best_pair[0] + best_pair[1], new_token)
                new_word_counts[new_word] = count
            word_counts = new_word_counts
        
        return vocab


class FastTokenizer:
    """Fast tokenizer implementation with optimized performance."""
    
    def __init__(self, vocab: Dict[str, int], special_tokens: Optional[SpecialTokens] = None):
        self.vocab = vocab
        self.ids_to_tokens = {v: k for k, v in vocab.items()}
        self.special_tokens = special_tokens or SpecialTokens()
        self._special_token_ids = set()
        
        # Initialize special token IDs
        for token in [self.special_tokens.pad_token, self.special_tokens.unk_token,
                     self.special_tokens.cls_token, self.special_tokens.sep_token,
                     self.special_tokens.mask_token, self.special_tokens.bos_token,
                     self.special_tokens.eos_token] + self.special_tokens.additional_special_tokens:
            if token in self.vocab:
                self._special_token_ids.add(self.vocab[token])
        
        # Performance optimizations
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        self._tokenization_cache: Dict[str, List[str]] = {}
        self._cache_lock = threading.Lock()
        self._max_cache_size = 10000
        self.model_max_length = 512
        self.padding_side = "right"
    
    @property
    def pad_token_id(self) -> int:
        return self.vocab.get(self.special_tokens.pad_token, 0)
    
    @property
    def unk_token_id(self) -> int:
        return self.vocab.get(self.special_tokens.unk_token, 1)
    
    @property
    def cls_token_id(self) -> int:
        return self.vocab.get(self.special_tokens.cls_token, 2)
    
    @property
    def sep_token_id(self) -> int:
        return self.vocab.get(self.special_tokens.sep_token, 3)
    
    @property
    def mask_token_id(self) -> int:
        return self.vocab.get(self.special_tokens.mask_token, 4)
    
    def tokenize_with_offsets(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        """Tokenize text and return tokens with character offset mappings."""
        tokens = []
        offsets = []
        
        # Track character position
        char_pos = 0
        words = text.split()
        
        for word in words:
            # Find word position in original text
            word_start = text.find(word, char_pos)
            if word_start == -1:
                word_start = char_pos
            
            # Subword tokenization
            subwords = self._subword_tokenize(word)
            subword_char_pos = word_start
            
            for subword in subwords:
                # Calculate subword length (remove ## prefix for offset calculation)
                actual_subword = subword[2:] if subword.startswith("##") else subword
                subword_end = subword_char_pos + len(actual_subword)
                
                tokens.append(subword)
                offsets.append((subword_char_pos, subword_end))
                
                subword_char_pos = subword_end
            
            char_pos = word_start + len(word)
        
        return tokens, offsets
    
    def tokenize_with_alignment(self, text: str) -> Tuple[List[str], AlignmentTracker]:
        """Tokenize text and return alignment information."""
        tracker = AlignmentTracker()
        tokens = []
        
        words = text.split()
        char_pos = 0
        
        for word_idx, word in enumerate(words):
            # Find word position in original text
            word_start = text.find(word, char_pos)
            if word_start == -1:
                word_start = char_pos
            
            # Subword tokenization
            subwords = self._subword_tokenize(word)
            subword_char_pos = word_start
            
            for token_idx, subword in enumerate(subwords):
                # Calculate subword length
                actual_subword = subword[2:] if subword.startswith("##") else subword
                subword_end = subword_char_pos + len(actual_subword)
                
                # Add to tracker
                tracker.add_token(
                    token_id=len(tokens),
                    word_id=word_idx,
                    char_start=subword_char_pos,
                    char_end=subword_end
                )
                
                tokens.append(subword)
                subword_char_pos = subword_end
            
            char_pos = word_start + len(word)
        
        return tokens, tracker
    
    def _subword_tokenize(self, word: str) -> List[str]:
        """Subword tokenization using the loaded vocabulary."""
        if word in self.vocab:
            return [word]
        
        # Greedy longest-first subword tokenization
        tokens = []
        i = 0
        
        while i < len(word):
            longest_match = None
            longest_length = 0
            
            # Find longest matching subword
            for j in range(i + 1, len(word) + 1):
                subword = word[i:j]
                if i > 0:  # Add continuation prefix for non-first subwords
                    subword = "##" + subword
                
                if subword in self.vocab and len(subword) > longest_length:
                    longest_match = subword
                    longest_length = len(subword)
            
            if longest_match:
                tokens.append(longest_match)
                i += longest_length - (2 if longest_match.startswith("##") else 0)
            else:
                # Unknown character, use UNK token
                tokens.append(self.special_tokens.unk_token)
                i += 1
        
        return tokens
    
    def encode_batch_fast(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_offsets_mapping: bool = False,
        return_attention_mask: bool = True,
        return_tensors: Optional[str] = None,
        num_workers: int = 4
    ) -> Dict[str, Any]:
        """Fast batch encoding using parallel processing."""
        if len(texts) <= 1:
            return self._encode_single(
                texts[0] if texts else "",
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_offsets_mapping=return_offsets_mapping,
                return_attention_mask=return_attention_mask,
                return_tensors=return_tensors
            )
        
        # Process in parallel
        def encode_single(text):
            return self._encode_single(
                text,
                add_special_tokens=add_special_tokens,
                padding=False,  # Handle padding at batch level
                truncation=truncation,
                max_length=max_length,
                return_offsets_mapping=return_offsets_mapping,
                return_attention_mask=return_attention_mask,
                return_tensors=None  # Handle tensors at batch level
            )
        
        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=min(num_workers, len(texts))) as executor:
            results = list(executor.map(encode_single, texts))
        
        # Combine results
        combined = {
            "input_ids": [result["input_ids"] for result in results],
        }
        
        if return_attention_mask:
            combined["attention_mask"] = [result["attention_mask"] for result in results]
        
        if return_offsets_mapping:
            combined["offset_mapping"] = [result["offset_mapping"] for result in results]
        
        # Apply padding to the batch
        if padding:
            max_len = max(len(ids) for ids in combined["input_ids"])
            if max_length:
                max_len = min(max_len, max_length)
            
            for i in range(len(combined["input_ids"])):
                current_len = len(combined["input_ids"][i])
                if current_len < max_len:
                    pad_length = max_len - current_len
                    if self.padding_side == "right":
                        combined["input_ids"][i].extend([self.pad_token_id] * pad_length)
                        if return_attention_mask:
                            combined["attention_mask"][i].extend([0] * pad_length)
                        if return_offsets_mapping:
                            combined["offset_mapping"][i].extend([(0, 0)] * pad_length)
                    else:
                        combined["input_ids"][i] = [self.pad_token_id] * pad_length + combined["input_ids"][i]
                        if return_attention_mask:
                            combined["attention_mask"][i] = [0] * pad_length + combined["attention_mask"][i]
                        if return_offsets_mapping:
                            combined["offset_mapping"][i] = [(0, 0)] * pad_length + combined["offset_mapping"][i]
        
        # Convert to tensors if requested
        if return_tensors == "np":
            for key, value in combined.items():
                combined[key] = np.array(value)
        elif return_tensors == "pt":
            try:
                import torch
                for key, value in combined.items():
                    combined[key] = torch.tensor(value)
            except ImportError:
                warnings.warn("PyTorch not available, returning Python lists")
        
        return combined
    
    def _encode_single(
        self,
        text: str,
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_offsets_mapping: bool = False,
        return_attention_mask: bool = True,
        return_tensors: Optional[str] = None
    ) -> Dict[str, Any]:
        """Encode a single text."""
        # Tokenize with offsets if needed
        if return_offsets_mapping:
            tokens, offsets = self.tokenize_with_offsets(text)
        else:
            tokens = self._fast_tokenize(text)
            offsets = None
        
        # Add special tokens
        if add_special_tokens:
            tokens = [self.special_tokens.cls_token] + tokens + [self.special_tokens.sep_token]
            if offsets:
                # Special tokens have (0, 0) offsets
                offsets = [(0, 0)] + offsets + [(0, 0)]
        
        # Convert to IDs
        input_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        
        # Handle truncation
        if truncation and max_length and len(input_ids) > max_length:
            if add_special_tokens:
                # Keep CLS and SEP tokens
                input_ids = input_ids[:max_length-1] + [input_ids[-1]]
                if offsets:
                    offsets = offsets[:max_length-1] + [offsets[-1]]
            else:
                input_ids = input_ids[:max_length]
                if offsets:
                    offsets = offsets[:max_length]
        
        result = {"input_ids": input_ids}
        
        # Attention mask
        if return_attention_mask:
            result["attention_mask"] = [1] * len(input_ids)
        
        # Offset mapping
        if return_offsets_mapping and offsets:
            result["offset_mapping"] = offsets
        
        # Handle padding
        if padding and max_length and len(input_ids) < max_length:
            pad_length = max_length - len(input_ids)
            if self.padding_side == "right":
                result["input_ids"].extend([self.pad_token_id] * pad_length)
                if return_attention_mask:
                    result["attention_mask"].extend([0] * pad_length)
                if return_offsets_mapping:
                    result["offset_mapping"].extend([(0, 0)] * pad_length)
            else:
                result["input_ids"] = [self.pad_token_id] * pad_length + result["input_ids"]
                if return_attention_mask:
                    result["attention_mask"] = [0] * pad_length + result["attention_mask"]
                if return_offsets_mapping:
                    result["offset_mapping"] = [(0, 0)] * pad_length + result["offset_mapping"]
        
        return result
    
    def _fast_tokenize(self, text: str) -> List[str]:
        """Optimized tokenization implementation."""
        # Check cache first
        with self._cache_lock:
            if text in self._tokenization_cache:
                return self._tokenization_cache[text].copy()
        
        # Pre-process text
        text = text.strip()
        if not text:
            return []
        
        # Split on whitespace and punctuation
        tokens = []
        current_token = ""
        
        for char in text:
            if char.isspace():
                if current_token:
                    tokens.extend(self._subword_tokenize(current_token))
                    current_token = ""
            elif char in ".,!?;:()[]{}\"'":
                if current_token:
                    tokens.extend(self._subword_tokenize(current_token))
                    current_token = ""
                tokens.append(char)
            else:
                current_token += char
        
        if current_token:
            tokens.extend(self._subword_tokenize(current_token))
        
        # Cache result
        with self._cache_lock:
            if len(self._tokenization_cache) < self._max_cache_size:
                self._tokenization_cache[text] = tokens.copy()
            elif len(self._tokenization_cache) >= self._max_cache_size:
                # Remove oldest entry (simple LRU approximation)
                oldest_key = next(iter(self._tokenization_cache))
                del self._tokenization_cache[oldest_key]
                self._tokenization_cache[text] = tokens.copy()
        
        return tokens
    
    def clear_cache(self):
        """Clear the tokenization cache."""
        with self._cache_lock:
            self._tokenization_cache.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._cache_lock:
            return {
                "cache_size": len(self._tokenization_cache),
                "max_cache_size": self._max_cache_size,
                "cache_hit_rate": getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_attempts', 1), 1),
            }
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer to directory."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save tokenizer config
        config_file = os.path.join(save_directory, "tokenizer_config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            config = {
                "model_max_length": self.model_max_length,
                "padding_side": self.padding_side,
                "special_tokens": self.special_tokens.to_dict(),
                "vocab_size": len(self.vocab),
                "tokenizer_class": "FastTokenizer",
            }
            json.dump(config, f, indent=2)
        
        # Save vocabulary
        vocab_file = os.path.join(save_directory, "vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load tokenizer from pretrained."""
        if os.path.isdir(pretrained_model_name_or_path):
            return cls._from_local_directory(pretrained_model_name_or_path, **kwargs)
        
        # Fall back to default instance with basic vocab
        vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
        return cls(vocab=vocab, **kwargs)
    
    @classmethod
    def _from_local_directory(cls, directory: str, **kwargs):
        """Load tokenizer from local directory."""
        config_file = os.path.join(directory, "tokenizer_config.json")
        vocab_file = os.path.join(directory, "vocab.json")
        
        # Load config
        config = {}
        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
        
        # Load vocabulary
        vocab = {}
        if os.path.exists(vocab_file):
            with open(vocab_file, "r", encoding="utf-8") as f:
                vocab = json.load(f)
        
        # Create special tokens object
        special_tokens_dict = config.get("special_tokens", {})
        special_tokens = SpecialTokens(**special_tokens_dict)
        
        # Create tokenizer instance
        tokenizer = cls(vocab=vocab, special_tokens=special_tokens, **kwargs)
        
        # Update properties from config
        for key, value in config.items():
            if hasattr(tokenizer, key) and key not in ["special_tokens", "vocab_size"]:
                setattr(tokenizer, key, value)
        
        return tokenizer


def create_fast_tokenizer_from_files(
    training_files: List[str],
    vocab_size: int = 30000,
    algorithm: str = "wordpiece",
    special_tokens: Optional[SpecialTokens] = None,
    min_frequency: int = 2,
    show_progress: bool = True
) -> FastTokenizer:
    """Create a fast tokenizer by training on files."""
    corpus = TokenizerTrainingCorpus(training_files)
    
    if algorithm == "wordpiece":
        vocab = TokenizerTrainer.train_wordpiece(
            corpus,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens
        )
    elif algorithm == "bpe":
        vocab = TokenizerTrainer.train_bpe(
            corpus,
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Supported: 'wordpiece', 'bpe'")
    
    return FastTokenizer(vocab=vocab, special_tokens=special_tokens)