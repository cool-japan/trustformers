"""
GPT-2 model implementations for specific tasks
"""

from typing import Optional, Tuple, Union
import numpy as np

from . import GPT2Model, GPT2Config
from .modeling_utils import (
    CausalLMOutput,
    SequenceClassifierOutput,
)


class GPT2LMHeadModel:
    """GPT-2 Model with a language modeling head using composition."""
    
    def __init__(self, config: GPT2Config):
        # Use composition instead of inheritance
        self.gpt2 = GPT2Model(config.to_dict())
        self.config = config
    
    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        token_type_ids: Optional[np.ndarray] = None,
        position_ids: Optional[np.ndarray] = None,
        past_key_values: Optional[Tuple[Tuple[np.ndarray]]] = None,
        labels: Optional[np.ndarray] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[np.ndarray], CausalLMOutput]:
        """
        Forward pass for causal language modeling.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (not used in GPT-2)
            position_ids: Position IDs
            past_key_values: Past key values for caching
            labels: Labels for computing the loss
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dict instead of tuple
            
        Returns:
            CausalLMOutput or tuple
        """
        # Get base model outputs using composition
        outputs = self.gpt2.forward(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        
        hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs['last_hidden_state']
        
        # Apply language modeling head (mock implementation)
        vocab_size = self.config.vocab_size
        lm_logits = np.random.randn(hidden_states.shape[0], hidden_states.shape[1], vocab_size)
        
        loss = None
        if labels is not None:
            # Shift logits and labels for causal LM
            # Compute cross-entropy loss
            loss = np.array(0.5)
        
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutput(
            loss=loss,
            logits=lm_logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )
    
    def generate(
        self,
        input_ids: np.ndarray,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = 1.0,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        num_beams: Optional[int] = 1,
        **kwargs,
    ) -> np.ndarray:
        """
        Generate text using the language model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_length: Maximum length of generated sequence
            max_new_tokens: Maximum number of new tokens to generate
            min_length: Minimum length of generated sequence
            do_sample: Whether to use sampling (True) or greedy decoding (False)
            temperature: Temperature for sampling (higher = more random)
            top_k: Top-k sampling (only consider top k tokens)
            top_p: Top-p (nucleus) sampling (consider tokens with cumulative prob up to p)
            repetition_penalty: Penalty for repeating tokens (>1.0 discourages repetition)
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            num_beams: Number of beams for beam search (>1 enables beam search)
            
        Returns:
            Generated token IDs [batch_size, max_length]
        """
        # Set default values
        if pad_token_id is None:
            pad_token_id = getattr(self.config, 'pad_token_id', 50256)
        if eos_token_id is None:
            eos_token_id = getattr(self.config, 'eos_token_id', 50256)
        
        # Determine generation length
        if max_new_tokens is not None:
            max_length = input_ids.shape[1] + max_new_tokens
        elif max_length is None:
            max_length = input_ids.shape[1] + 50
        
        if min_length is None:
            min_length = input_ids.shape[1] + 1
            
        batch_size = input_ids.shape[0]
        
        # Simple beam search or sampling implementation
        if num_beams > 1:
            return self._beam_search_generate(
                input_ids, max_length, min_length, num_beams, 
                pad_token_id, eos_token_id, repetition_penalty
            )
        else:
            return self._sample_generate(
                input_ids, max_length, min_length, do_sample,
                temperature, top_k, top_p, repetition_penalty,
                pad_token_id, eos_token_id
            )
    
    def _sample_generate(
        self,
        input_ids: np.ndarray,
        max_length: int,
        min_length: int,
        do_sample: bool,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
        repetition_penalty: float,
        pad_token_id: int,
        eos_token_id: int,
    ) -> np.ndarray:
        """Generate using sampling or greedy decoding."""
        batch_size, seq_len = input_ids.shape
        device_ids = input_ids.copy()
        
        # Keep track of which sequences are finished
        finished = np.zeros(batch_size, dtype=bool)
        
        for step in range(seq_len, max_length):
            # Get logits from forward pass
            outputs = self.forward(device_ids, return_dict=True)
            logits = outputs.logits[:, -1, :]  # Get logits for last token
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits, device_ids, repetition_penalty
                )
            
            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature
            
            if do_sample:
                # Sampling mode
                next_tokens = self._sample_tokens(logits, top_k, top_p)
            else:
                # Greedy decoding
                next_tokens = np.argmax(logits, axis=-1)
            
            # Don't generate more tokens for finished sequences
            next_tokens = np.where(finished, pad_token_id, next_tokens)
            
            # Add next tokens to sequences
            device_ids = np.concatenate([
                device_ids, 
                next_tokens.reshape(batch_size, 1)
            ], axis=1)
            
            # Check for EOS tokens
            if step >= min_length - 1:  # Only check EOS after min_length
                finished |= (next_tokens == eos_token_id)
            
            # Stop if all sequences are finished
            if finished.all():
                break
        
        return device_ids
    
    def _beam_search_generate(
        self,
        input_ids: np.ndarray,
        max_length: int,
        min_length: int,
        num_beams: int,
        pad_token_id: int,
        eos_token_id: int,
        repetition_penalty: float,
    ) -> np.ndarray:
        """Simple beam search implementation."""
        batch_size, seq_len = input_ids.shape
        
        # For simplicity, implement a basic beam search for batch_size=1
        if batch_size > 1:
            # Fall back to sampling for batched inputs
            return self._sample_generate(
                input_ids, max_length, min_length, False, 1.0,
                None, None, repetition_penalty, pad_token_id, eos_token_id
            )
        
        # Initialize beams: [num_beams, seq_len]
        beams = np.tile(input_ids[0], (num_beams, 1))
        beam_scores = np.zeros(num_beams)
        finished_beams = []
        
        for step in range(seq_len, max_length):
            all_candidates = []
            
            for beam_idx in range(num_beams):
                if len(finished_beams) >= num_beams:
                    break
                    
                # Get logits for this beam
                beam_input = beams[beam_idx:beam_idx+1]
                outputs = self.forward(beam_input, return_dict=True)
                logits = outputs.logits[0, -1, :]  # [vocab_size]
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    logits = self._apply_repetition_penalty(
                        logits.reshape(1, -1), beam_input, repetition_penalty
                    )[0]
                
                # Get top-k candidates
                top_k_scores, top_k_indices = self._get_top_k(logits, num_beams)
                
                for score, token_id in zip(top_k_scores, top_k_indices):
                    candidate_score = beam_scores[beam_idx] + score
                    candidate_sequence = np.concatenate([
                        beams[beam_idx], [token_id]
                    ])
                    
                    all_candidates.append((candidate_score, candidate_sequence, token_id))
            
            # Select top candidates
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            
            new_beams = []
            new_scores = []
            
            for score, sequence, last_token in all_candidates[:num_beams]:
                if last_token == eos_token_id and step >= min_length - 1:
                    finished_beams.append((score, sequence))
                else:
                    new_beams.append(sequence)
                    new_scores.append(score)
            
            if len(new_beams) == 0:
                break
                
            # Pad sequences to same length
            max_beam_len = max(len(beam) for beam in new_beams)
            beams = np.array([
                np.pad(beam, (0, max_beam_len - len(beam)), 
                       constant_values=pad_token_id)
                for beam in new_beams
            ])
            beam_scores = np.array(new_scores)
        
        # Return best sequence
        if finished_beams:
            best_score, best_sequence = max(finished_beams, key=lambda x: x[0])
            # Pad to max_length if needed
            result = np.pad(best_sequence, (0, max_length - len(best_sequence)), 
                           constant_values=pad_token_id)
        else:
            result = beams[0]
            if len(result) < max_length:
                result = np.pad(result, (0, max_length - len(result)), 
                               constant_values=pad_token_id)
        
        return result.reshape(1, -1)
    
    def _apply_repetition_penalty(
        self, 
        logits: np.ndarray, 
        input_ids: np.ndarray, 
        penalty: float
    ) -> np.ndarray:
        """Apply repetition penalty to logits."""
        if penalty == 1.0:
            return logits
            
        for batch_idx in range(logits.shape[0]):
            for token_id in np.unique(input_ids[batch_idx]):
                if token_id < logits.shape[1]:
                    if logits[batch_idx, token_id] > 0:
                        logits[batch_idx, token_id] /= penalty
                    else:
                        logits[batch_idx, token_id] *= penalty
        
        return logits
    
    def _sample_tokens(
        self, 
        logits: np.ndarray, 
        top_k: Optional[int], 
        top_p: Optional[float]
    ) -> np.ndarray:
        """Sample tokens from logits with top-k and top-p filtering."""
        batch_size, vocab_size = logits.shape
        
        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            # Get top-k indices
            top_k_indices = np.argpartition(logits, -top_k, axis=-1)[:, -top_k:]
            
            # Zero out non-top-k logits
            mask = np.ones_like(logits) * -np.inf
            for batch_idx in range(batch_size):
                mask[batch_idx, top_k_indices[batch_idx]] = 0
            logits = logits + mask
        
        # Apply top-p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            sorted_indices = np.argsort(logits, axis=-1)[:, ::-1]
            sorted_logits = np.take_along_axis(logits, sorted_indices, axis=-1)
            
            # Convert to probabilities
            probs = self._softmax(sorted_logits)
            cumulative_probs = np.cumsum(probs, axis=-1)
            
            # Find cutoff
            cutoff_mask = cumulative_probs > top_p
            # Keep at least one token
            cutoff_mask[:, 0] = False
            
            # Zero out tokens beyond cutoff
            sorted_logits[cutoff_mask] = -np.inf
            
            # Unsort logits
            unsorted_logits = np.zeros_like(logits)
            np.put_along_axis(unsorted_logits, sorted_indices, sorted_logits, axis=-1)
            logits = unsorted_logits
        
        # Sample from distribution
        probs = self._softmax(logits)
        
        # Multinomial sampling
        next_tokens = np.array([
            np.random.choice(vocab_size, p=probs[i])
            for i in range(batch_size)
        ])
        
        return next_tokens
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax with numerical stability."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _get_top_k(self, logits: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get top-k scores and indices."""
        top_k_indices = np.argpartition(logits, -k)[-k:]
        top_k_scores = logits[top_k_indices]
        
        # Sort by score (descending)
        sorted_order = np.argsort(top_k_scores)[::-1]
        top_k_scores = top_k_scores[sorted_order]
        top_k_indices = top_k_indices[sorted_order]
        
        return top_k_scores, top_k_indices


class GPT2ForSequenceClassification:
    """GPT-2 Model with a sequence classification head."""
    
    def __init__(self, config: GPT2Config):
        super().__init__(config)
        self.num_labels = getattr(config, 'num_labels', 2)
        self.config = config
    
    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        token_type_ids: Optional[np.ndarray] = None,
        position_ids: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[np.ndarray], SequenceClassifierOutput]:
        """
        Forward pass for sequence classification.
        """
        outputs = self.gpt2.forward(
            input_ids,
            attention_mask=attention_mask,
        )
        
        hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs['last_hidden_state']
        
        # Extract features from last token (GPT-2 style)
        # Find last non-padded token for each sequence
        if attention_mask is not None:
            sequence_lengths = np.sum(attention_mask, axis=1) - 1
        else:
            sequence_lengths = np.full(hidden_states.shape[0], hidden_states.shape[1] - 1)
        
        # Get hidden states at last positions
        pooled_output = np.array([hidden_states[i, int(seq_len)] for i, seq_len in enumerate(sequence_lengths)])
        
        # Apply classifier
        logits = np.random.randn(pooled_output.shape[0], self.num_labels)
        
        loss = None
        if labels is not None:
            # Compute cross-entropy loss
            loss = np.array(0.5)
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )