"""
Evaluation metrics for NLP tasks.

Provides a comprehensive set of evaluation metrics compatible with HuggingFace's
evaluate library interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable
import warnings
from collections import defaultdict
import numpy as np
from .utils import logging

try:
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, 
        matthews_corrcoef, roc_auc_score, classification_report
    )
    _sklearn_available = True
except ImportError:
    _sklearn_available = False

try:
    import scipy.stats
    _scipy_available = True
except ImportError:
    _scipy_available = False

logger = logging.get_logger(__name__)


class EvaluationMetric(ABC):
    """Base class for all evaluation metrics."""
    
    def __init__(self):
        self._state = {}
    
    @abstractmethod
    def compute(self, predictions: Any, references: Any, **kwargs) -> Union[float, Dict[str, float]]:
        """Compute the metric."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Get the metric name."""
        pass
    
    def get_name(self) -> str:
        """Get the metric name (compatibility method)."""
        return self.name
    
    def reset(self):
        """Reset metric state."""
        self._state = {}


class AccuracyMetric(EvaluationMetric):
    """Accuracy metric for classification tasks."""
    
    def __init__(self):
        super().__init__()
    
    def compute(self, predictions: Union[List, np.ndarray], references: Union[List, np.ndarray], **kwargs) -> float:
        """
        Compute accuracy.
        
        Args:
            predictions: Predicted labels or logits
            references: True labels
            
        Returns:
            Accuracy score
        """
        if len(predictions) == 0 or len(references) == 0:
            return 0.0
            
        predictions = np.array(predictions)
        references = np.array(references)
        
        # Handle logits vs labels
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=-1)
            
        accuracy = np.mean(predictions == references)
        return float(accuracy)
    
    @property
    def name(self) -> str:
        return "accuracy"


class F1Metric(EvaluationMetric):
    """F1 score metric for classification tasks."""
    
    def __init__(self, average: str = "macro"):
        """
        Initialize F1 metric.
        
        Args:
            average: Averaging strategy ('micro', 'macro', 'weighted', 'binary')
        """
        super().__init__()
        self.average = average
    
    def compute(self, predictions: Union[List, np.ndarray], references: Union[List, np.ndarray], **kwargs) -> float:
        """
        Compute F1 score.
        
        Args:
            predictions: Predicted labels or logits
            references: True labels
            
        Returns:
            F1 score
        """
        if not _sklearn_available:
            raise ImportError("sklearn is required for F1 metric. Install with: pip install scikit-learn")
            
        predictions = np.array(predictions)
        references = np.array(references)
        
        # Handle logits vs labels
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=-1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            references, predictions, average=self.average, zero_division=0
        )
        
        return float(f1)
    
    @property
    def name(self) -> str:
        return "f1"


class PrecisionRecallMetric(EvaluationMetric):
    """Precision and Recall metrics."""
    
    def __init__(self, average: str = "macro"):
        super().__init__()
        self.average = average
    
    def compute(self, predictions: Union[List, np.ndarray], references: Union[List, np.ndarray], **kwargs) -> Dict[str, float]:
        """Compute precision and recall."""
        if not _sklearn_available:
            raise ImportError("sklearn is required for precision/recall metrics. Install with: pip install scikit-learn")
            
        predictions = np.array(predictions)
        references = np.array(references)
        
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=-1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            references, predictions, average=self.average, zero_division=0
        )
        
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }
    
    @property
    def name(self) -> str:
        return "precision_recall"


class MatthewsCorrCoefMetric(EvaluationMetric):
    """Matthews Correlation Coefficient for binary classification."""
    
    def __init__(self):
        super().__init__()
    
    def compute(self, predictions: Union[List, np.ndarray], references: Union[List, np.ndarray], **kwargs) -> float:
        """Compute Matthews correlation coefficient."""
        if not _sklearn_available:
            raise ImportError("sklearn is required for MCC metric. Install with: pip install scikit-learn")
            
        predictions = np.array(predictions)
        references = np.array(references)
        
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=-1)
        
        mcc = matthews_corrcoef(references, predictions)
        return float(mcc)
    
    @property
    def name(self) -> str:
        return "matthews_corrcoef"


class PerplexityMetric(EvaluationMetric):
    """Perplexity metric for language models."""
    
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
    
    def compute(self, predictions: Union[List, np.ndarray], references: Union[List, np.ndarray], **kwargs) -> float:
        """
        Compute perplexity.
        
        Args:
            predictions: Logits or log probabilities
            references: True labels
            
        Returns:
            Dictionary with perplexity
        """
        predictions = np.array(predictions)
        references = np.array(references)
        
        # Calculate cross-entropy loss
        if predictions.ndim == 2:
            # Logits provided
            # Apply softmax to get probabilities
            exp_logits = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
            
            # Get probabilities for true labels
            true_probs = probs[np.arange(len(references)), references]
        else:
            # Assume log probabilities or probabilities
            true_probs = predictions
        
        # Avoid log(0)
        true_probs = np.clip(true_probs, 1e-10, 1.0)
        
        # Calculate cross-entropy loss
        cross_entropy = -np.mean(np.log(true_probs))
        
        # Calculate perplexity
        perplexity = np.exp(cross_entropy)
        
        return float(perplexity)
    
    @property
    def name(self) -> str:
        return "perplexity"


class BLEUMetric(EvaluationMetric):
    """BLEU score for text generation evaluation."""
    
    def __init__(self, weights: List[float] = None):
        super().__init__()
        self.weights = weights or [0.25, 0.25, 0.25, 0.25]
    
    def compute(self, predictions: List[str], references: List[List[str]], **kwargs) -> float:
        """
        Compute BLEU score.
        
        Args:
            predictions: Generated text
            references: Reference texts (list of lists for multiple references)
            
        Returns:
            BLEU score
        """
        try:
            from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
            import nltk
            # Download required NLTK data
            try:
                nltk.download('punkt_tab', quiet=True)
            except:
                nltk.download('punkt', quiet=True)  # fallback
            from nltk.tokenize import word_tokenize
        except ImportError:
            raise ImportError("NLTK is required for BLEU metric. Install with: pip install nltk")
        
        try:
            # Handle pre-tokenized inputs (lists) vs raw text (strings)
            if isinstance(predictions[0], list):
                # Already tokenized
                tokenized_predictions = predictions
                tokenized_references = references
            else:
                # Tokenize predictions and references
                tokenized_predictions = [word_tokenize(pred.lower()) for pred in predictions]
                tokenized_references = [[word_tokenize(ref.lower()) for ref in refs] for refs in references]
        except LookupError:
            # Fallback to simple split tokenization if NLTK data not available
            if isinstance(predictions[0], list):
                tokenized_predictions = predictions
                tokenized_references = references
            else:
                tokenized_predictions = [pred.lower().split() for pred in predictions]
                tokenized_references = [[ref.lower().split() for ref in refs] for refs in references]
        
        # Compute BLEU score with smoothing
        smoothing = SmoothingFunction().method1
        bleu_score = corpus_bleu(
            tokenized_references, 
            tokenized_predictions, 
            weights=self.weights,
            smoothing_function=smoothing
        )
        
        return float(bleu_score)
    
    @property
    def name(self) -> str:
        return "bleu"


class ROUGEMetric(EvaluationMetric):
    """ROUGE score for summarization evaluation."""
    
    def __init__(self, rouge_type: str = "rouge1"):
        super().__init__()
        self.rouge_type = rouge_type
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Union[float, Dict[str, float]]:
        """
        Compute ROUGE scores.
        
        Args:
            predictions: Generated summaries
            references: Reference summaries
            
        Returns:
            ROUGE score (float) for the specified rouge_type, or dict of all scores
        """
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            logger.warning("rouge-score not available. Computing simple ROUGE approximation.")
            result = self._compute_simple_rouge(predictions, references)
            # Return specific rouge type score or full dict
            if self.rouge_type in result:
                return result[self.rouge_type]
            return result.get("rouge1", 0.0)  # Default fallback
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for pred, ref in zip(predictions, references):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        result = {
            "rouge1": float(np.mean(rouge1_scores)),
            "rouge2": float(np.mean(rouge2_scores)),
            "rougeL": float(np.mean(rougeL_scores))
        }
        
        # Return specific rouge type score or full dict
        if self.rouge_type in result:
            return result[self.rouge_type]
        return result["rouge1"]  # Default fallback
    
    def _compute_simple_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Simple ROUGE approximation using word overlap."""
        rouge1_scores = []
        
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            
            if len(ref_words) == 0:
                rouge1_scores.append(0.0)
            else:
                overlap = len(pred_words & ref_words)
                rouge1_scores.append(overlap / len(ref_words))
        
        return {"rouge1": float(np.mean(rouge1_scores))}
    
    @property
    def name(self) -> str:
        return "rouge"


class ExactMatchMetric(EvaluationMetric):
    """Exact match metric for QA tasks."""
    
    def __init__(self, ignore_case: bool = True, normalize_text: bool = True):
        super().__init__()
        self.ignore_case = ignore_case
        self.normalize_text = normalize_text
    
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> float:
        """
        Compute exact match accuracy.
        
        Args:
            predictions: Predicted answers
            references: True answers
            
        Returns:
            Exact match accuracy as float
        """
        exact_matches = []
        
        for pred, ref in zip(predictions, references):
            # Normalize text based on settings
            if self.normalize_text:
                pred_norm = self._normalize_text(pred)
                ref_norm = self._normalize_text(ref)
            else:
                pred_norm = pred
                ref_norm = ref
                
            if self.ignore_case:
                pred_norm = pred_norm.lower()
                ref_norm = ref_norm.lower()
            
            exact_matches.append(pred_norm == ref_norm)
        
        return float(np.mean(exact_matches))
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        import re
        # Remove punctuation and extra whitespace
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
        return text.strip()
    
    @property
    def name(self) -> str:
        return "exact_match"


class SquadMetric(EvaluationMetric):
    """SQuAD-style F1 and Exact Match for QA evaluation."""
    
    def compute(self, predictions: List[Union[str, Dict]], references: List[Union[str, Dict]], **kwargs) -> Dict[str, float]:
        """
        Compute SQuAD F1 and Exact Match.
        
        Args:
            predictions: Predicted answers (strings or dicts with 'prediction_text' key)
            references: True answers (strings or dicts with 'answers' key)
            
        Returns:
            Dictionary with F1 and exact match scores
        """
        f1_scores = []
        exact_matches = []
        
        for pred, ref in zip(predictions, references):
            # Extract text from dict format if needed
            if isinstance(pred, dict):
                pred_text = pred.get('prediction_text', str(pred))
            else:
                pred_text = str(pred)
                
            if isinstance(ref, dict):
                # Handle multiple answer format
                if 'answers' in ref:
                    ref_answers = ref['answers']
                    if isinstance(ref_answers, list) and len(ref_answers) > 0:
                        # Use first answer for now
                        ref_text = str(ref_answers[0].get('text', ref_answers[0]) if isinstance(ref_answers[0], dict) else ref_answers[0])
                    else:
                        ref_text = str(ref_answers)
                else:
                    ref_text = str(ref)
            else:
                ref_text = str(ref)
            
            # Normalize and tokenize
            pred_tokens = self._normalize_answer(pred_text).split()
            ref_tokens = self._normalize_answer(ref_text).split()
            
            # Exact match
            exact_matches.append(pred_tokens == ref_tokens)
            
            # F1 score
            if len(ref_tokens) == 0:
                f1_scores.append(1.0 if len(pred_tokens) == 0 else 0.0)
            else:
                common_tokens = set(pred_tokens) & set(ref_tokens)
                if len(common_tokens) == 0:
                    f1_scores.append(0.0)
                else:
                    precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0.0
                    recall = len(common_tokens) / len(ref_tokens)
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    f1_scores.append(f1)
        
        return {
            "exact_match": float(np.mean(exact_matches)),
            "f1": float(np.mean(f1_scores))
        }
    
    def _normalize_answer(self, text: str) -> str:
        """Normalize answer text for comparison."""
        import re
        import string
        
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        
        def white_space_fix(text):
            return ' '.join(text.split())
        
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        
        def lower(text):
            return text.lower()
        
        return white_space_fix(remove_articles(remove_punc(lower(text))))
    
    @property
    def name(self) -> str:
        return "squad"


class MetricCollection:
    """Collection of multiple metrics."""
    
    def __init__(self, metrics: Dict[str, EvaluationMetric]):
        """
        Initialize metric collection.
        
        Args:
            metrics: Dictionary mapping metric names to metric instances
        """
        self.metrics = metrics
    
    def compute(self, predictions: Any, references: Any, **kwargs) -> Dict[str, float]:
        """
        Compute all metrics in the collection.
        
        Args:
            predictions: Predictions
            references: References
            
        Returns:
            Dictionary with all metric results
        """
        results = {}
        
        for name, metric in self.metrics.items():
            try:
                metric_results = metric.compute(predictions, references, **kwargs)
                # Handle both dict and float/scalar returns
                if isinstance(metric_results, dict):
                    # Store dict results under the metric name
                    results[name] = metric_results
                else:
                    results[name] = float(metric_results)
            except Exception as e:
                logger.warning(f"Failed to compute metric {name}: {e}")
                results[name] = float('nan')
        
        return results
    
    def add_metric(self, name: str, metric: EvaluationMetric):
        """Add a metric to the collection."""
        self.metrics[name] = metric
    
    def remove_metric(self, name: str):
        """Remove a metric from the collection."""
        if name in self.metrics:
            del self.metrics[name]
    
    def reset(self):
        """Reset all metrics in the collection."""
        for metric in self.metrics.values():
            metric.reset()


# Convenience functions for common metric collections
def get_classification_metrics(average: str = "macro") -> MetricCollection:
    """Get common classification metrics."""
    return MetricCollection({
        "accuracy": AccuracyMetric(),
        "f1": F1Metric(average=average),
        "precision_recall": PrecisionRecallMetric(average=average),
        "matthews_corrcoef": MatthewsCorrCoefMetric(),
    })

def get_generation_metrics(include_perplexity: bool = False) -> MetricCollection:
    """Get common text generation metrics."""
    metrics = {
        "bleu": BLEUMetric(),
        "rouge": ROUGEMetric(),
    }
    
    if include_perplexity:
        metrics["perplexity"] = PerplexityMetric()
    
    return MetricCollection(metrics)

def get_qa_metrics() -> MetricCollection:
    """Get common QA metrics."""
    return MetricCollection({
        "squad": SquadMetric(),
        "exact_match": ExactMatchMetric(),
    })

def compute_metric(
    metric_name: str,
    predictions: Any = None,
    references: Any = None,
    labels: Any = None,
    **kwargs
) -> Union[float, Dict[str, float]]:
    """
    Compute a single metric by name.
    
    Args:
        metric_name: Name of the metric to compute
        predictions: Predictions
        references: References (alternative: labels)
        labels: Labels (alternative to references for compatibility)
        **kwargs: Additional arguments for the metric
        
    Returns:
        Metric result (float or dict)
    """
    # Handle both 'references' and 'labels' parameters for compatibility
    if references is None and labels is not None:
        references = labels
    elif references is None and labels is None:
        raise TypeError("Either 'references' or 'labels' must be provided")
    
    # Handle metric collections
    if metric_name == "classification":
        return get_classification_metrics().compute(predictions, references, **kwargs)
    elif metric_name == "generation":
        return get_generation_metrics().compute(predictions, references, **kwargs)
    elif metric_name == "qa":
        return get_qa_metrics().compute(predictions, references, **kwargs)
    
    metric_classes = {
        "accuracy": AccuracyMetric,
        "f1": F1Metric,
        "precision_recall": PrecisionRecallMetric,
        "matthews_corrcoef": MatthewsCorrCoefMetric,
        "perplexity": PerplexityMetric,
        "bleu": BLEUMetric,
        "rouge": ROUGEMetric,
        "exact_match": ExactMatchMetric,
        "squad": SquadMetric,
    }
    
    if metric_name not in metric_classes:
        raise ValueError(f"Unknown metric: {metric_name}. Available: {list(metric_classes.keys())}")
    
    metric_class = metric_classes[metric_name]
    
    # Handle metrics with parameters
    if metric_name in ["f1", "precision_recall"]:
        average = kwargs.pop("average", "macro")
        metric = metric_class(average=average)
    else:
        metric = metric_class()
    
    return metric.compute(predictions, references, **kwargs)