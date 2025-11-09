"""
BERT model implementations for specific tasks
"""

from typing import Optional, Tuple, Union
import numpy as np

from . import BertModel, BertConfig
from .modeling_utils import (
    BaseModelOutputWithPooling,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    QuestionAnsweringModelOutput,
    MaskedLMOutput,
)


class BertForSequenceClassification:
    """BERT Model with a sequence classification head using composition."""
    
    def __init__(self, config: BertConfig):
        # Use composition instead of inheritance
        self.bert = BertModel(config.to_dict())
        self.num_labels = getattr(config, 'num_labels', 2)
        self.config = config
        
        # Initialize classifier head
        # In practice, this would create a linear layer
        # For now, we'll handle it in forward()
    
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
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            position_ids: Position IDs
            labels: Labels for computing the loss
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states
            return_dict: Whether to return a dict instead of tuple
            
        Returns:
            SequenceClassifierOutput or tuple
        """
        # Get base model outputs using composition
        outputs = self.bert.forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        # Extract pooled output
        if hasattr(outputs, 'pooler_output'):
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs['pooler_output']
        
        # Apply classifier (mock implementation)
        # In practice, this would be a linear layer
        logits = np.random.randn(pooled_output.shape[0], self.num_labels)
        
        loss = None
        if labels is not None:
            # Compute cross-entropy loss
            # Mock implementation
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


class BertForTokenClassification:
    """BERT Model with a token classification head using composition."""
    
    def __init__(self, config: BertConfig):
        # Use composition instead of inheritance
        self.bert = BertModel(config.to_dict())
        self.num_labels = getattr(config, 'num_labels', 9)  # Default for NER
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
    ) -> Union[Tuple[np.ndarray], TokenClassifierOutput]:
        """
        Forward pass for token classification.
        """
        outputs = self.bert.forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        sequence_output = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs['last_hidden_state']
        
        # Apply token classifier (mock implementation)
        logits = np.random.randn(sequence_output.shape[0], sequence_output.shape[1], self.num_labels)
        
        loss = None
        if labels is not None:
            # Compute token classification loss
            loss = np.array(0.5)
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )


class BertForQuestionAnswering:
    """BERT Model with a question answering head using composition."""
    
    def __init__(self, config: BertConfig):
        # Use composition instead of inheritance
        self.bert = BertModel(config.to_dict())
        self.config = config
    
    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: Optional[np.ndarray] = None,
        token_type_ids: Optional[np.ndarray] = None,
        position_ids: Optional[np.ndarray] = None,
        start_positions: Optional[np.ndarray] = None,
        end_positions: Optional[np.ndarray] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[np.ndarray], QuestionAnsweringModelOutput]:
        """
        Forward pass for question answering.
        """
        outputs = self.bert.forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        sequence_output = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs['last_hidden_state']
        
        # Apply QA head (mock implementation)
        logits = np.random.randn(sequence_output.shape[0], sequence_output.shape[1], 2)
        start_logits = logits[:, :, 0]
        end_logits = logits[:, :, 1]
        
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # Compute QA loss
            total_loss = np.array(0.5)
        
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output
        
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )


class BertForMaskedLM:
    """BERT Model with a masked language modeling head."""
    
    def __init__(self, config: BertConfig):
        # Use composition instead of inheritance
        self.bert = BertModel(config.to_dict())
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
    ) -> Union[Tuple[np.ndarray], MaskedLMOutput]:
        """
        Forward pass for masked language modeling.
        """
        outputs = self.bert.forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        sequence_output = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs['last_hidden_state']
        
        # Apply MLM head (mock implementation)
        vocab_size = self.config.vocab_size
        prediction_logits = np.random.randn(sequence_output.shape[0], sequence_output.shape[1], vocab_size)
        
        masked_lm_loss = None
        if labels is not None:
            # Compute MLM loss
            masked_lm_loss = np.array(0.5)
        
        if not return_dict:
            output = (prediction_logits,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
        
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )