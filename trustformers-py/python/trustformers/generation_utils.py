"""
Text generation utilities for TrustformeRS models.

Provides advanced text generation capabilities including beam search,
nucleus sampling, top-k sampling, and other generation strategies
compatible with HuggingFace's interface.
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Callable
import warnings
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod

from .utils import logging
from .batch_encoding import BatchEncoding

logger = logging.get_logger(__name__)


@dataclass
class GenerationConfig:
    """
    Configuration class for text generation parameters.
    """
    # Generation parameters
    max_length: int = 50
    max_new_tokens: Optional[int] = None
    min_length: int = 0
    min_new_tokens: Optional[int] = None
    early_stopping: bool = False
    
    # Sampling parameters
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0
    typical_p: float = 1.0
    epsilon_cutoff: float = 0.0
    eta_cutoff: float = 0.0
    
    # Beam search parameters
    num_beams: int = 1
    num_beam_groups: int = 1
    penalty_alpha: Optional[float] = None
    use_cache: bool = True
    
    # Special tokens
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    # Repetition penalty
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    encoder_no_repeat_ngram_size: int = 0
    
    # Length penalty
    length_penalty: float = 1.0
    
    # Diversity
    diversity_penalty: float = 0.0
    
    # Constraints
    forced_bos_token_id: Optional[int] = None
    forced_eos_token_id: Optional[int] = None
    remove_invalid_values: bool = False
    
    # Guidance
    guidance_scale: float = 1.0
    
    # Output control
    output_attentions: bool = False
    output_hidden_states: bool = False
    output_scores: bool = False
    return_dict_in_generate: bool = False
    
    # Stopping criteria
    max_time: Optional[float] = None
    
    def __post_init__(self):
        """Validate and adjust parameters after initialization."""
        # Ensure temperature is positive
        if self.temperature <= 0:
            self.temperature = 1.0
            warnings.warn("Temperature must be positive, setting to 1.0")
        
        # Ensure top_p is valid
        if not 0 < self.top_p <= 1:
            self.top_p = 1.0
            warnings.warn("top_p must be between 0 and 1, setting to 1.0")
        
        # Ensure top_k is positive
        if self.top_k <= 0:
            self.top_k = 50
            warnings.warn("top_k must be positive, setting to 50")


class GenerationMixin:
    """
    Mixin class that provides generation capabilities to models.
    """
    
    def __init__(self):
        self.generation_config = GenerationConfig()
    
    def generate(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Union[np.ndarray, Dict[str, Any]]:
        """
        Generate sequences for the given input.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            generation_config: Generation configuration
            **kwargs: Additional generation parameters
            
        Returns:
            Generated sequences or generation output dict
        """
        # Merge generation config
        config = self._prepare_generation_config(generation_config, **kwargs)
        
        # Prepare inputs
        batch_size = input_ids.shape[0] if len(input_ids.shape) > 1 else 1
        if len(input_ids.shape) == 1:
            input_ids = input_ids.reshape(1, -1)
        
        if attention_mask is not None and len(attention_mask.shape) == 1:
            attention_mask = attention_mask.reshape(1, -1)
        
        # Choose generation strategy
        if config.num_beams > 1:
            return self._beam_search_generate(input_ids, attention_mask, config)
        elif config.do_sample:
            return self._sampling_generate(input_ids, attention_mask, config)
        else:
            return self._greedy_generate(input_ids, attention_mask, config)
    
    def _prepare_generation_config(
        self, 
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> GenerationConfig:
        """Prepare generation configuration."""
        if generation_config is None:
            config = GenerationConfig()
        else:
            config = generation_config
        
        # Override with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Set default special tokens if not provided
        if hasattr(self, 'config'):
            model_config = self.config
            if config.pad_token_id is None and hasattr(model_config, 'pad_token_id'):
                config.pad_token_id = model_config.pad_token_id
            if config.eos_token_id is None and hasattr(model_config, 'eos_token_id'):
                config.eos_token_id = model_config.eos_token_id
            if config.bos_token_id is None and hasattr(model_config, 'bos_token_id'):
                config.bos_token_id = model_config.bos_token_id
        
        return config
    
    def _greedy_generate(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray],
        config: GenerationConfig
    ) -> np.ndarray:
        """Generate using greedy decoding."""
        batch_size, seq_len = input_ids.shape
        
        # Determine max length
        if config.max_new_tokens is not None:
            max_length = seq_len + config.max_new_tokens
        else:
            max_length = config.max_length
        
        generated_ids = input_ids.copy()
        past_key_values = None
        
        for step in range(seq_len, max_length):
            # Forward pass
            if past_key_values is not None and config.use_cache:
                # Use cached states
                model_inputs = {
                    'input_ids': generated_ids[:, -1:],
                    'past_key_values': past_key_values
                }
            else:
                model_inputs = {'input_ids': generated_ids}
                
            if attention_mask is not None:
                # Extend attention mask
                new_attention_mask = np.ones((batch_size, generated_ids.shape[1]))
                new_attention_mask[:, :attention_mask.shape[1]] = attention_mask
                model_inputs['attention_mask'] = new_attention_mask
            
            outputs = self(**model_inputs)
            
            # Get logits for next token
            if isinstance(outputs, dict):
                logits = outputs['logits']
                past_key_values = outputs.get('past_key_values')
            else:
                logits = outputs
            
            next_token_logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated_ids, config.repetition_penalty
                )
            
            # Get next token (greedy)
            next_tokens = np.argmax(next_token_logits, axis=-1)
            
            # Add to sequence
            generated_ids = np.concatenate([
                generated_ids, 
                next_tokens.reshape(-1, 1)
            ], axis=1)
            
            # Check for early stopping
            if config.eos_token_id is not None:
                if np.all(next_tokens == config.eos_token_id):
                    break
        
        return generated_ids
    
    def _sampling_generate(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray],
        config: GenerationConfig
    ) -> np.ndarray:
        """Generate using sampling strategies."""
        batch_size, seq_len = input_ids.shape
        
        # Determine max length
        if config.max_new_tokens is not None:
            max_length = seq_len + config.max_new_tokens
        else:
            max_length = config.max_length
        
        generated_ids = input_ids.copy()
        past_key_values = None
        
        for step in range(seq_len, max_length):
            # Forward pass
            if past_key_values is not None and config.use_cache:
                model_inputs = {
                    'input_ids': generated_ids[:, -1:],
                    'past_key_values': past_key_values
                }
            else:
                model_inputs = {'input_ids': generated_ids}
                
            if attention_mask is not None:
                new_attention_mask = np.ones((batch_size, generated_ids.shape[1]))
                new_attention_mask[:, :attention_mask.shape[1]] = attention_mask
                model_inputs['attention_mask'] = new_attention_mask
            
            outputs = self(**model_inputs)
            
            # Get logits
            if isinstance(outputs, dict):
                logits = outputs['logits']
                past_key_values = outputs.get('past_key_values')
            else:
                logits = outputs
            
            next_token_logits = logits[:, -1, :]
            
            # Apply repetition penalty
            if config.repetition_penalty != 1.0:
                next_token_logits = self._apply_repetition_penalty(
                    next_token_logits, generated_ids, config.repetition_penalty
                )
            
            # Apply temperature
            if config.temperature != 1.0:
                next_token_logits = next_token_logits / config.temperature
            
            # Apply top-k filtering
            if config.top_k > 0:
                next_token_logits = self._top_k_filtering(next_token_logits, config.top_k)
            
            # Apply top-p (nucleus) filtering
            if config.top_p < 1.0:
                next_token_logits = self._top_p_filtering(next_token_logits, config.top_p)
            
            # Convert to probabilities and sample
            probs = self._softmax(next_token_logits)
            next_tokens = self._multinomial_sample(probs)
            
            # Add to sequence
            generated_ids = np.concatenate([
                generated_ids,
                next_tokens.reshape(-1, 1)
            ], axis=1)
            
            # Check for early stopping
            if config.eos_token_id is not None:
                if np.all(next_tokens == config.eos_token_id):
                    break
        
        return generated_ids
    
    def _beam_search_generate(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray],
        config: GenerationConfig
    ) -> np.ndarray:
        """Generate using beam search."""
        batch_size, seq_len = input_ids.shape
        num_beams = config.num_beams
        
        # Determine max length
        if config.max_new_tokens is not None:
            max_length = seq_len + config.max_new_tokens
        else:
            max_length = config.max_length
        
        # Expand input for beam search
        expanded_batch_size = batch_size * num_beams
        beam_input_ids = np.repeat(input_ids, num_beams, axis=0)
        
        if attention_mask is not None:
            beam_attention_mask = np.repeat(attention_mask, num_beams, axis=0)
        else:
            beam_attention_mask = None
        
        # Initialize beam scores
        beam_scores = np.zeros((batch_size, num_beams))
        beam_scores[:, 1:] = -1e9  # Only first beam is active initially
        beam_scores = beam_scores.reshape(-1)
        
        # Keep track of sequences
        generated_sequences = beam_input_ids.copy()
        past_key_values = None
        
        for step in range(seq_len, max_length):
            # Forward pass
            if past_key_values is not None and config.use_cache:
                model_inputs = {
                    'input_ids': generated_sequences[:, -1:],
                    'past_key_values': past_key_values
                }
            else:
                model_inputs = {'input_ids': generated_sequences}
                
            if beam_attention_mask is not None:
                new_attention_mask = np.ones((expanded_batch_size, generated_sequences.shape[1]))
                new_attention_mask[:, :beam_attention_mask.shape[1]] = beam_attention_mask
                model_inputs['attention_mask'] = new_attention_mask
            
            outputs = self(**model_inputs)
            
            # Get logits
            if isinstance(outputs, dict):
                logits = outputs['logits']
                past_key_values = outputs.get('past_key_values')
            else:
                logits = outputs
            
            next_token_logits = logits[:, -1, :]
            
            # Apply length penalty
            if config.length_penalty != 1.0:
                current_length = generated_sequences.shape[1]
                length_penalty = ((5 + current_length) / 6) ** config.length_penalty
                next_token_logits = next_token_logits / length_penalty
            
            # Convert to log probabilities
            log_probs = self._log_softmax(next_token_logits)
            
            # Add beam scores
            vocab_size = log_probs.shape[-1]
            next_scores = beam_scores.reshape(-1, 1) + log_probs
            next_scores = next_scores.reshape(batch_size, num_beams * vocab_size)
            
            # Select top 2*num_beams scores
            next_scores, next_tokens = self._get_top_k(next_scores, 2 * num_beams)
            
            # Convert token indices back to beam and token indices
            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size
            
            # Create new sequences
            new_sequences = []
            new_scores = []
            
            for batch_idx in range(batch_size):
                batch_start = batch_idx * num_beams
                batch_end = (batch_idx + 1) * num_beams
                
                # Get sequences for this batch
                batch_sequences = generated_sequences[batch_start:batch_end]
                batch_scores = next_scores[batch_idx]
                batch_tokens = next_tokens[batch_idx]
                batch_indices = next_indices[batch_idx]
                
                # Select top num_beams
                top_indices = np.argsort(batch_scores)[-num_beams:][::-1]
                
                for i, idx in enumerate(top_indices):
                    beam_idx = batch_indices[idx]
                    token_id = batch_tokens[idx]
                    score = batch_scores[idx]
                    
                    # Get base sequence
                    base_seq = batch_sequences[beam_idx]
                    new_seq = np.concatenate([base_seq, [token_id]])
                    
                    new_sequences.append(new_seq)
                    new_scores.append(score)
            
            # Update sequences and scores
            generated_sequences = np.array(new_sequences)
            beam_scores = np.array(new_scores)
            
            # Check for early stopping
            if config.eos_token_id is not None:
                # This is simplified - proper implementation would track finished beams
                pass
        
        # Return best sequences for each batch
        final_sequences = []
        for batch_idx in range(batch_size):
            batch_start = batch_idx * num_beams
            best_idx = batch_start + np.argmax(beam_scores[batch_start:batch_start + num_beams])
            final_sequences.append(generated_sequences[best_idx])
        
        return np.array(final_sequences)
    
    def _apply_repetition_penalty(
        self, 
        logits: np.ndarray, 
        generated_ids: np.ndarray, 
        penalty: float
    ) -> np.ndarray:
        """Apply repetition penalty to logits."""
        batch_size, vocab_size = logits.shape
        
        for batch_idx in range(batch_size):
            for token_id in set(generated_ids[batch_idx].tolist()):
                if logits[batch_idx, token_id] < 0:
                    logits[batch_idx, token_id] *= penalty
                else:
                    logits[batch_idx, token_id] /= penalty
        
        return logits
    
    def _top_k_filtering(self, logits: np.ndarray, top_k: int) -> np.ndarray:
        """Apply top-k filtering to logits."""
        if top_k <= 0:
            return logits
        
        # Get top-k indices
        top_k = min(top_k, logits.shape[-1])
        indices_to_remove = logits < np.partition(logits, -top_k, axis=-1)[..., -top_k, None]
        logits[indices_to_remove] = -float('inf')
        
        return logits
    
    def _top_p_filtering(self, logits: np.ndarray, top_p: float) -> np.ndarray:
        """Apply top-p (nucleus) filtering to logits."""
        if top_p >= 1.0:
            return logits
        
        # Sort logits in descending order
        sorted_indices = np.argsort(logits, axis=-1)[:, ::-1]
        sorted_logits = np.take_along_axis(logits, sorted_indices, axis=-1)
        
        # Convert to probabilities
        sorted_probs = self._softmax(sorted_logits)
        
        # Calculate cumulative probabilities
        cumulative_probs = np.cumsum(sorted_probs, axis=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].copy()
        sorted_indices_to_remove[:, 0] = False
        
        # Convert back to original indices
        indices_to_remove = np.take_along_axis(sorted_indices_to_remove, np.argsort(sorted_indices, axis=-1), axis=-1)
        logits[indices_to_remove] = -float('inf')
        
        return logits
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities."""
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    def _log_softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute log softmax."""
        return logits - np.max(logits, axis=-1, keepdims=True) - np.log(
            np.sum(np.exp(logits - np.max(logits, axis=-1, keepdims=True)), axis=-1, keepdims=True)
        )
    
    def _multinomial_sample(self, probs: np.ndarray) -> np.ndarray:
        """Sample from multinomial distribution."""
        batch_size = probs.shape[0]
        samples = []
        
        for i in range(batch_size):
            # Use numpy's random choice
            sample = np.random.choice(len(probs[i]), p=probs[i])
            samples.append(sample)
        
        return np.array(samples)
    
    def _get_top_k(self, scores: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get top-k scores and indices."""
        # Use argpartition for efficiency
        indices = np.argpartition(scores, -k, axis=-1)[:, -k:]
        top_scores = np.take_along_axis(scores, indices, axis=-1)
        
        # Sort within the top-k
        sort_indices = np.argsort(top_scores, axis=-1)[:, ::-1]
        sorted_scores = np.take_along_axis(top_scores, sort_indices, axis=-1)
        sorted_indices = np.take_along_axis(indices, sort_indices, axis=-1)
        
        return sorted_scores, sorted_indices


def batch_generate(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    generation_config: Optional[GenerationConfig] = None,
    batch_size: int = 8,
    **kwargs
) -> List[str]:
    """
    Generate text for a batch of prompts.
    
    Args:
        model: Text generation model
        tokenizer: Tokenizer
        prompts: List of input prompts
        generation_config: Generation configuration
        batch_size: Batch size for processing
        **kwargs: Additional generation parameters
        
    Returns:
        List of generated texts
    """
    generated_texts = []
    
    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer.batch_encode_plus(
            batch_prompts,
            padding=True,
            truncation=True,
            return_tensors="np"
        )
        
        # Generate
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if hasattr(model, 'generate'):
                outputs = model.generate(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs.get('attention_mask'),
                    generation_config=generation_config,
                    **kwargs
                )
            else:
                raise AttributeError("Model does not support generation")
        
        # Decode outputs
        for j, output_ids in enumerate(outputs):
            # Remove input tokens from output
            input_length = len(inputs['input_ids'][j])
            generated_ids = output_ids[input_length:]
            
            # Decode
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            generated_texts.append(text)
    
    return generated_texts


# Export main classes and functions
__all__ = [
    'GenerationConfig',
    'GenerationMixin',
    'batch_generate'
]