"""
Additional pipeline implementations for TrustformeRS
"""

from typing import List, Dict, Any, Optional, Union
import numpy as np

from . import (
    TextGenerationPipeline,
    TextClassificationPipeline,
    AutoModel,
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForTokenClassification,
)


class FillMaskPipeline:
    """
    Pipeline for masked language modeling.
    
    Example:
        >>> fill_mask = pipeline("fill-mask", model="bert-base-uncased")
        >>> fill_mask("The capital of France is [MASK].")
    """
    
    def __init__(self, model=None, tokenizer=None, device="cpu", **kwargs):
        if model is None:
            model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def __call__(
        self,
        inputs: Union[str, List[str]],
        targets: Optional[Union[str, List[str]]] = None,
        top_k: int = 5,
        **kwargs
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """
        Fill masked tokens in the input text.
        
        Args:
            inputs: Text with [MASK] tokens
            targets: Optional target words to score
            top_k: Number of top predictions to return
            
        Returns:
            List of predictions with scores
        """
        # Handle single vs batch inputs
        if isinstance(inputs, str):
            inputs = [inputs]
            single_input = True
        else:
            single_input = False
        
        results = []
        
        for text in inputs:
            # Tokenize
            encoded = self.tokenizer(text, return_tensors="np")
            
            # Find mask token positions
            mask_token_id = self.tokenizer.mask_token_id
            mask_positions = np.where(encoded["input_ids"][0] == mask_token_id)[0]
            
            if len(mask_positions) == 0:
                results.append([])
                continue
            
            # Get model predictions
            outputs = self.model(**encoded)
            predictions = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            # Process each mask position
            text_results = []
            for mask_pos in mask_positions:
                # Get top-k predictions for this position
                logits = predictions[0, mask_pos]
                top_k_indices = np.argsort(logits)[-top_k:][::-1]
                
                # Convert to probabilities
                probs = np.exp(logits) / np.sum(np.exp(logits))
                
                # Create results
                for idx in top_k_indices:
                    token = self.tokenizer.decode([idx])
                    score = float(probs[idx])
                    
                    # Create filled sequence
                    filled_ids = encoded["input_ids"][0].copy()
                    filled_ids[mask_pos] = idx
                    sequence = self.tokenizer.decode(filled_ids, skip_special_tokens=True)
                    
                    text_results.append({
                        "score": score,
                        "token": idx,
                        "token_str": token,
                        "sequence": sequence,
                    })
            
            results.append(text_results[:top_k])
        
        return results[0] if single_input else results


class QuestionAnsweringPipeline:
    """
    Pipeline for question answering.
    
    Example:
        >>> qa = pipeline("question-answering")
        >>> qa(question="What is the capital of France?", context="Paris is the capital of France.")
    """
    
    def __init__(self, model=None, tokenizer=None, device="cpu", **kwargs):
        if model is None:
            model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def __call__(
        self,
        question: Union[str, List[str]],
        context: Union[str, List[str]],
        top_k: int = 1,
        max_answer_len: int = 15,
        **kwargs
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Answer questions based on context.
        
        Args:
            question: Question(s) to answer
            context: Context(s) containing the answer
            top_k: Number of answers to return
            max_answer_len: Maximum answer length
            
        Returns:
            Answer(s) with confidence scores
        """
        # Handle single vs batch inputs
        if isinstance(question, str):
            questions = [question]
            contexts = [context] if isinstance(context, str) else context
            single_input = True
        else:
            questions = question
            contexts = context if isinstance(context, list) else [context] * len(questions)
            single_input = False
        
        results = []
        
        for q, c in zip(questions, contexts):
            # Tokenize
            encoded = self.tokenizer(q, c, return_tensors="np", return_offsets_mapping=True)
            offset_mapping = encoded.pop("offset_mapping")[0]
            
            # Get model predictions
            outputs = self.model(**encoded)
            start_logits = outputs.start_logits if hasattr(outputs, 'start_logits') else outputs[0]
            end_logits = outputs.end_logits if hasattr(outputs, 'end_logits') else outputs[1]
            
            # Get top-k start and end positions
            start_indices = np.argsort(start_logits[0])[-20:][::-1]
            end_indices = np.argsort(end_logits[0])[-20:][::-1]
            
            # Find valid answer spans
            answers = []
            for start_idx in start_indices:
                for end_idx in end_indices:
                    if start_idx <= end_idx < start_idx + max_answer_len:
                        score = float(start_logits[0][start_idx] + end_logits[0][end_idx])
                        
                        # Extract answer text
                        if start_idx < len(offset_mapping) and end_idx < len(offset_mapping):
                            start_char = offset_mapping[start_idx][0]
                            end_char = offset_mapping[end_idx][1]
                            answer_text = c[start_char:end_char]
                        else:
                            answer_text = ""
                        
                        answers.append({
                            "score": score,
                            "start": int(start_idx),
                            "end": int(end_idx),
                            "answer": answer_text,
                        })
            
            # Sort by score and take top-k
            answers.sort(key=lambda x: x["score"], reverse=True)
            results.append(answers[:top_k])
        
        return results[0] if single_input else results


class TokenClassificationPipeline:
    """
    Pipeline for token classification (e.g., NER).
    
    Example:
        >>> ner = pipeline("ner")
        >>> ner("My name is John and I live in New York.")
    """
    
    def __init__(self, model=None, tokenizer=None, device="cpu", **kwargs):
        if model is None:
            model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased")
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Default NER labels
        self.id2label = {
            0: "O",
            1: "B-PER",
            2: "I-PER",
            3: "B-ORG",
            4: "I-ORG",
            5: "B-LOC",
            6: "I-LOC",
            7: "B-MISC",
            8: "I-MISC",
        }
    
    def __call__(
        self,
        inputs: Union[str, List[str]],
        aggregation_strategy: str = "simple",
        **kwargs
    ) -> Union[List[Dict[str, Any]], List[List[Dict[str, Any]]]]:
        """
        Classify tokens in the input text.
        
        Args:
            inputs: Text to classify
            aggregation_strategy: How to aggregate tokens ("simple", "first", "average", "max")
            
        Returns:
            List of entities with labels and scores
        """
        if isinstance(inputs, str):
            inputs = [inputs]
            single_input = True
        else:
            single_input = False
        
        results = []
        
        for text in inputs:
            # Tokenize
            encoded = self.tokenizer(text, return_tensors="np", return_offsets_mapping=True)
            offset_mapping = encoded.pop("offset_mapping")[0]
            
            # Get model predictions
            outputs = self.model(**encoded)
            predictions = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            # Get predicted labels
            predicted_labels = np.argmax(predictions[0], axis=1)
            scores = np.max(predictions[0], axis=1)
            
            # Extract entities
            entities = []
            current_entity = None
            
            for idx, (label_id, score, (start, end)) in enumerate(zip(predicted_labels, scores, offset_mapping)):
                label = self.id2label.get(int(label_id), "O")
                
                if label == "O":
                    if current_entity:
                        entities.append(current_entity)
                        current_entity = None
                elif label.startswith("B-"):
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        "entity": label[2:],
                        "score": float(score),
                        "index": idx,
                        "word": text[start:end],
                        "start": int(start),
                        "end": int(end),
                    }
                elif label.startswith("I-") and current_entity and current_entity["entity"] == label[2:]:
                    # Extend current entity
                    current_entity["word"] = text[current_entity["start"]:end]
                    current_entity["end"] = int(end)
                    current_entity["score"] = max(current_entity["score"], float(score))
            
            if current_entity:
                entities.append(current_entity)
            
            results.append(entities)
        
        return results[0] if single_input else results


class SummarizationPipeline:
    """
    Pipeline for text summarization.
    
    Example:
        >>> summarizer = pipeline("summarization")
        >>> summarizer("Long article text here...")
    """
    
    def __init__(self, model=None, tokenizer=None, device="cpu", **kwargs):
        # Default to T5 for summarization
        model_name = "t5-small"
        if model is None:
            from . import T5ForConditionalGeneration
            model = T5ForConditionalGeneration.from_pretrained(model_name)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def __call__(
        self,
        inputs: Union[str, List[str]],
        max_length: int = 130,
        min_length: int = 30,
        do_sample: bool = False,
        **kwargs
    ) -> Union[List[Dict[str, str]], List[List[Dict[str, str]]]]:
        """
        Summarize input text.
        
        Args:
            inputs: Text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
            do_sample: Whether to use sampling
            
        Returns:
            Summary text
        """
        if isinstance(inputs, str):
            inputs = [inputs]
            single_input = True
        else:
            single_input = False
        
        results = []
        
        for text in inputs:
            # Add task prefix for T5
            text = "summarize: " + text
            
            # Tokenize
            encoded = self.tokenizer(text, return_tensors="np", truncation=True, max_length=512)
            
            # Generate summary
            summary_ids = self.model.generate(
                encoded["input_ids"],
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
            )
            
            # Decode
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            results.append([{"summary_text": summary}])
        
        return results[0] if single_input else results


class TranslationPipeline:
    """
    Pipeline for translation.
    
    Example:
        >>> translator = pipeline("translation_en_to_fr")
        >>> translator("Hello world!")
    """
    
    def __init__(self, model=None, tokenizer=None, device="cpu", src_lang="en", tgt_lang="fr", **kwargs):
        # Default to T5 for translation
        model_name = "t5-small"
        if model is None:
            from . import T5ForConditionalGeneration
            model = T5ForConditionalGeneration.from_pretrained(model_name)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
    
    def __call__(
        self,
        inputs: Union[str, List[str]],
        max_length: int = 128,
        **kwargs
    ) -> Union[List[Dict[str, str]], List[List[Dict[str, str]]]]:
        """
        Translate input text.
        
        Args:
            inputs: Text to translate
            max_length: Maximum translation length
            
        Returns:
            Translated text
        """
        if isinstance(inputs, str):
            inputs = [inputs]
            single_input = True
        else:
            single_input = False
        
        results = []
        
        for text in inputs:
            # Add task prefix for T5
            text = f"translate {self.src_lang} to {self.tgt_lang}: " + text
            
            # Tokenize
            encoded = self.tokenizer(text, return_tensors="np", truncation=True, max_length=512)
            
            # Generate translation
            translation_ids = self.model.generate(
                encoded["input_ids"],
                max_length=max_length,
            )
            
            # Decode
            translation = self.tokenizer.decode(translation_ids[0], skip_special_tokens=True)
            
            results.append([{"translation_text": translation}])
        
        return results[0] if single_input else results