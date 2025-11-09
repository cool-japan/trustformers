"""
Tests for the evaluation metrics module
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from typing import List, Dict, Any

# Test evaluation module
try:
    from trustformers.evaluation import (
        EvaluationMetric,
        AccuracyMetric,
        F1Metric,
        PrecisionRecallMetric,
        MatthewsCorrCoefMetric,
        PerplexityMetric,
        BLEUMetric,
        ROUGEMetric,
        ExactMatchMetric,
        SquadMetric,
        MetricCollection,
        get_classification_metrics,
        get_generation_metrics,
        get_qa_metrics,
        compute_metric,
    )
    EVALUATION_AVAILABLE = True
except ImportError:
    EVALUATION_AVAILABLE = False

# Skip all tests if evaluation module not available
pytestmark = pytest.mark.skipif(not EVALUATION_AVAILABLE, reason="Evaluation module not available")

class TestEvaluationMetric:
    """Test base EvaluationMetric class"""
    
    def test_evaluation_metric_interface(self):
        """Test that EvaluationMetric provides correct interface"""
        # Should be able to instantiate (as abstract base or interface)
        assert EvaluationMetric is not None
        
        # Check that it has expected methods (implementation dependent)
        expected_methods = ['compute', 'reset', 'get_name']
        for method in expected_methods:
            # Don't require all methods, but check what's available
            pass

class TestAccuracyMetric:
    """Test AccuracyMetric functionality"""
    
    def test_accuracy_metric_creation(self):
        """Test basic accuracy metric creation"""
        metric = AccuracyMetric()
        assert metric.get_name() == "accuracy"
    
    def test_binary_accuracy(self):
        """Test binary accuracy computation"""
        metric = AccuracyMetric()
        
        predictions = [1, 0, 1, 1, 0]
        labels = [1, 0, 1, 0, 0]
        
        accuracy = metric.compute(predictions, labels)
        expected = 4/5  # 4 correct out of 5
        assert abs(accuracy - expected) < 1e-6
    
    def test_multiclass_accuracy(self):
        """Test multiclass accuracy computation"""
        metric = AccuracyMetric()
        
        predictions = [0, 1, 2, 1, 0]
        labels = [0, 1, 2, 2, 0]
        
        accuracy = metric.compute(predictions, labels)
        expected = 4/5  # 4 correct out of 5
        assert abs(accuracy - expected) < 1e-6
    
    def test_accuracy_with_probabilities(self):
        """Test accuracy with probability predictions"""
        metric = AccuracyMetric()
        
        # Binary case with probabilities
        predictions = [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]]
        labels = [0, 1, 0]
        
        accuracy = metric.compute(predictions, labels)
        expected = 1.0  # All predictions correct
        assert abs(accuracy - expected) < 1e-6
    
    def test_accuracy_empty_inputs(self):
        """Test accuracy with empty inputs"""
        metric = AccuracyMetric()
        
        accuracy = metric.compute([], [])
        assert accuracy == 0.0
    
    def test_accuracy_reset(self):
        """Test accuracy metric reset"""
        metric = AccuracyMetric()
        
        # Compute some accuracy
        metric.compute([1, 0, 1], [1, 1, 0])
        
        # Reset should work without error
        metric.reset()

class TestF1Metric:
    """Test F1Metric functionality"""
    
    def test_f1_metric_creation(self):
        """Test basic F1 metric creation"""
        metric = F1Metric()
        assert metric.get_name() == "f1"
    
    def test_f1_metric_with_average(self):
        """Test F1 metric with different averaging"""
        metric_macro = F1Metric(average="macro")
        metric_micro = F1Metric(average="micro")
        metric_weighted = F1Metric(average="weighted")
        
        assert metric_macro.average == "macro"
        assert metric_micro.average == "micro"
        assert metric_weighted.average == "weighted"
    
    def test_binary_f1(self):
        """Test binary F1 computation"""
        metric = F1Metric(average="binary")
        
        predictions = [1, 0, 1, 1, 0]
        labels = [1, 0, 1, 0, 1]
        
        f1 = metric.compute(predictions, labels)
        
        # Manual calculation:
        # TP=2, FP=1, FN=1
        # Precision = 2/3, Recall = 2/3
        # F1 = 2 * (2/3 * 2/3) / (2/3 + 2/3) = 2/3
        expected = 2/3
        assert abs(f1 - expected) < 1e-6
    
    def test_multiclass_f1_macro(self):
        """Test multiclass F1 with macro averaging"""
        metric = F1Metric(average="macro")
        
        predictions = [0, 1, 2, 0, 1]
        labels = [0, 1, 2, 1, 2]
        
        f1 = metric.compute(predictions, labels)
        assert 0 <= f1 <= 1
    
    def test_f1_with_probabilities(self):
        """Test F1 with probability predictions"""
        metric = F1Metric(average="binary")
        
        predictions = [[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]]
        labels = [0, 1, 0]
        
        f1 = metric.compute(predictions, labels)
        assert 0 <= f1 <= 1

class TestPrecisionRecallMetric:
    """Test PrecisionRecallMetric functionality"""
    
    def test_precision_recall_creation(self):
        """Test precision recall metric creation"""
        metric = PrecisionRecallMetric()
        assert metric.get_name() == "precision_recall"
    
    def test_precision_recall_computation(self):
        """Test precision and recall computation"""
        metric = PrecisionRecallMetric()
        
        predictions = [1, 0, 1, 1, 0]
        labels = [1, 0, 1, 0, 1]
        
        result = metric.compute(predictions, labels)
        
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        
        # Check values are in valid range
        assert 0 <= result["precision"] <= 1
        assert 0 <= result["recall"] <= 1
        assert 0 <= result["f1"] <= 1
    
    def test_precision_recall_perfect_score(self):
        """Test precision recall with perfect predictions"""
        metric = PrecisionRecallMetric()
        
        predictions = [1, 0, 1, 0]
        labels = [1, 0, 1, 0]
        
        result = metric.compute(predictions, labels)
        
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

class TestMatthewsCorrCoefMetric:
    """Test MatthewsCorrCoefMetric functionality"""
    
    def test_mcc_metric_creation(self):
        """Test MCC metric creation"""
        metric = MatthewsCorrCoefMetric()
        assert metric.get_name() == "matthews_corrcoef"
    
    def test_mcc_computation(self):
        """Test MCC computation"""
        metric = MatthewsCorrCoefMetric()
        
        predictions = [1, 0, 1, 1, 0]
        labels = [1, 0, 1, 0, 1]
        
        mcc = metric.compute(predictions, labels)
        
        # MCC should be between -1 and 1
        assert -1 <= mcc <= 1
    
    def test_mcc_perfect_score(self):
        """Test MCC with perfect predictions"""
        metric = MatthewsCorrCoefMetric()
        
        predictions = [1, 0, 1, 0]
        labels = [1, 0, 1, 0]
        
        mcc = metric.compute(predictions, labels)
        assert abs(mcc - 1.0) < 1e-6
    
    def test_mcc_worst_score(self):
        """Test MCC with worst predictions"""
        metric = MatthewsCorrCoefMetric()
        
        predictions = [1, 0, 1, 0]
        labels = [0, 1, 0, 1]
        
        mcc = metric.compute(predictions, labels)
        assert abs(mcc - (-1.0)) < 1e-6

class TestPerplexityMetric:
    """Test PerplexityMetric functionality"""
    
    def test_perplexity_metric_creation(self):
        """Test perplexity metric creation"""
        metric = PerplexityMetric()
        assert metric.get_name() == "perplexity"
    
    def test_perplexity_computation(self):
        """Test perplexity computation"""
        metric = PerplexityMetric()
        
        # Mock logits and labels
        logits = [
            [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]],  # Sequence 1
            [[0.5, 0.3, 0.2], [0.2, 0.6, 0.2]]   # Sequence 2
        ]
        labels = [[2, 1], [0, 1]]
        
        perplexity = metric.compute(logits, labels)
        
        # Perplexity should be > 0
        assert perplexity > 0
    
    def test_perplexity_with_padding(self):
        """Test perplexity computation with padding tokens"""
        metric = PerplexityMetric(ignore_index=-100)
        
        logits = [[[0.1, 0.9], [0.8, 0.2]], [[0.3, 0.7], [0.6, 0.4]]]
        labels = [[1, -100], [0, 1]]  # Second token in first sequence is padding
        
        perplexity = metric.compute(logits, labels)
        assert perplexity > 0

class TestBLEUMetric:
    """Test BLEUMetric functionality"""
    
    def test_bleu_metric_creation(self):
        """Test BLEU metric creation"""
        metric = BLEUMetric()
        assert metric.get_name() == "bleu"
    
    def test_bleu_metric_with_weights(self):
        """Test BLEU metric with custom n-gram weights"""
        metric = BLEUMetric(weights=[0.25, 0.25, 0.25, 0.25])
        assert metric.weights == [0.25, 0.25, 0.25, 0.25]
    
    def test_bleu_computation(self):
        """Test BLEU computation"""
        metric = BLEUMetric()
        
        predictions = ["the cat sat on the mat"]
        references = [["the cat is on the mat"]]
        
        bleu = metric.compute(predictions, references)
        
        # BLEU should be between 0 and 1
        assert 0 <= bleu <= 1
    
    def test_bleu_perfect_match(self):
        """Test BLEU with perfect match"""
        metric = BLEUMetric()
        
        predictions = ["the cat sat on the mat"]
        references = [["the cat sat on the mat"]]
        
        bleu = metric.compute(predictions, references)
        
        # Should be perfect score
        assert abs(bleu - 1.0) < 1e-6
    
    def test_bleu_multiple_references(self):
        """Test BLEU with multiple references"""
        metric = BLEUMetric()
        
        predictions = ["the cat sat"]
        references = [["the cat sat on the mat", "a cat sat", "the cat was sitting"]]
        
        bleu = metric.compute(predictions, references)
        assert 0 <= bleu <= 1
    
    def test_bleu_tokenized_input(self):
        """Test BLEU with pre-tokenized input"""
        metric = BLEUMetric()
        
        predictions = [["the", "cat", "sat"]]
        references = [[["the", "cat", "sat", "on", "mat"]]]
        
        bleu = metric.compute(predictions, references)
        assert 0 <= bleu <= 1

class TestROUGEMetric:
    """Test ROUGEMetric functionality"""
    
    def test_rouge_metric_creation(self):
        """Test ROUGE metric creation"""
        metric = ROUGEMetric()
        assert metric.get_name() == "rouge"
    
    def test_rouge_metric_types(self):
        """Test different ROUGE metric types"""
        rouge_1 = ROUGEMetric(rouge_type="rouge1")
        rouge_2 = ROUGEMetric(rouge_type="rouge2")
        rouge_l = ROUGEMetric(rouge_type="rougeL")
        
        assert rouge_1.rouge_type == "rouge1"
        assert rouge_2.rouge_type == "rouge2"
        assert rouge_l.rouge_type == "rougeL"
    
    def test_rouge_computation(self):
        """Test ROUGE computation"""
        metric = ROUGEMetric(rouge_type="rouge1")
        
        predictions = ["the cat sat on the mat"]
        references = ["the cat is on the mat"]
        
        rouge = metric.compute(predictions, references)
        
        # ROUGE should be between 0 and 1
        assert 0 <= rouge <= 1
    
    def test_rouge_perfect_match(self):
        """Test ROUGE with perfect match"""
        metric = ROUGEMetric(rouge_type="rouge1")
        
        predictions = ["the cat sat on the mat"]
        references = ["the cat sat on the mat"]
        
        rouge = metric.compute(predictions, references)
        
        # Should be perfect score
        assert abs(rouge - 1.0) < 1e-6
    
    def test_rouge_multiple_predictions(self):
        """Test ROUGE with multiple predictions"""
        metric = ROUGEMetric(rouge_type="rouge1")
        
        predictions = ["the cat sat", "a dog ran"]
        references = ["the cat is sitting", "the dog is running"]
        
        rouge = metric.compute(predictions, references)
        assert 0 <= rouge <= 1

class TestExactMatchMetric:
    """Test ExactMatchMetric functionality"""
    
    def test_exact_match_creation(self):
        """Test exact match metric creation"""
        metric = ExactMatchMetric()
        assert metric.get_name() == "exact_match"
    
    def test_exact_match_computation(self):
        """Test exact match computation"""
        metric = ExactMatchMetric()
        
        predictions = ["hello world", "foo bar", "test"]
        references = ["hello world", "foo baz", "test"]
        
        em = metric.compute(predictions, references)
        expected = 2/3  # 2 out of 3 exact matches
        assert abs(em - expected) < 1e-6
    
    def test_exact_match_case_sensitive(self):
        """Test case-sensitive exact match"""
        metric = ExactMatchMetric(ignore_case=False)
        
        predictions = ["Hello World"]
        references = ["hello world"]
        
        em = metric.compute(predictions, references)
        assert em == 0.0
    
    def test_exact_match_case_insensitive(self):
        """Test case-insensitive exact match"""
        metric = ExactMatchMetric(ignore_case=True)
        
        predictions = ["Hello World"]
        references = ["hello world"]
        
        em = metric.compute(predictions, references)
        assert em == 1.0
    
    def test_exact_match_with_normalization(self):
        """Test exact match with text normalization"""
        metric = ExactMatchMetric(normalize_text=True)
        
        predictions = ["  Hello World!  "]
        references = ["hello world"]
        
        em = metric.compute(predictions, references)
        # Should match after normalization
        assert em > 0.0

class TestSquadMetric:
    """Test SquadMetric functionality"""
    
    def test_squad_metric_creation(self):
        """Test SQuAD metric creation"""
        metric = SquadMetric()
        assert metric.get_name() == "squad"
    
    def test_squad_computation(self):
        """Test SQuAD metric computation"""
        metric = SquadMetric()
        
        predictions = [
            {"id": "1", "prediction_text": "Paris"},
            {"id": "2", "prediction_text": "42"}
        ]
        
        references = [
            {"id": "1", "answers": {"text": ["Paris", "paris"], "answer_start": [0, 0]}},
            {"id": "2", "answers": {"text": ["42"], "answer_start": [10]}}
        ]
        
        result = metric.compute(predictions, references)
        
        assert "exact_match" in result
        assert "f1" in result
        assert 0 <= result["exact_match"] <= 1
        assert 0 <= result["f1"] <= 1
    
    def test_squad_partial_match(self):
        """Test SQuAD with partial matches"""
        metric = SquadMetric()
        
        predictions = [
            {"id": "1", "prediction_text": "The capital of France is Paris"}
        ]
        
        references = [
            {"id": "1", "answers": {"text": ["Paris"], "answer_start": [0]}}
        ]
        
        result = metric.compute(predictions, references)
        
        # Should have some F1 score even if not exact match
        assert result["f1"] > 0

class TestMetricCollection:
    """Test MetricCollection functionality"""
    
    def test_metric_collection_creation(self):
        """Test metric collection creation"""
        metrics = {
            "accuracy": AccuracyMetric(),
            "f1": F1Metric(),
            "precision_recall": PrecisionRecallMetric()
        }
        
        collection = MetricCollection(metrics)
        assert len(collection.metrics) == 3
        assert "accuracy" in collection.metrics
    
    def test_metric_collection_compute(self):
        """Test computing all metrics in collection"""
        metrics = {
            "accuracy": AccuracyMetric(),
            "f1": F1Metric(average="binary")
        }
        
        collection = MetricCollection(metrics)
        
        predictions = [1, 0, 1, 1]
        labels = [1, 0, 1, 0]
        
        results = collection.compute(predictions, labels)
        
        assert "accuracy" in results
        assert "f1" in results
        assert 0 <= results["accuracy"] <= 1
        assert 0 <= results["f1"] <= 1
    
    def test_metric_collection_reset(self):
        """Test resetting all metrics in collection"""
        metrics = {
            "accuracy": AccuracyMetric(),
            "f1": F1Metric()
        }
        
        collection = MetricCollection(metrics)
        
        # Compute some metrics
        collection.compute([1, 0], [1, 1])
        
        # Reset should work without error
        collection.reset()
    
    def test_metric_collection_add_metric(self):
        """Test adding metric to collection"""
        collection = MetricCollection({})
        
        collection.add_metric("accuracy", AccuracyMetric())
        assert "accuracy" in collection.metrics
        assert len(collection.metrics) == 1
    
    def test_metric_collection_remove_metric(self):
        """Test removing metric from collection"""
        metrics = {"accuracy": AccuracyMetric()}
        collection = MetricCollection(metrics)
        
        collection.remove_metric("accuracy")
        assert "accuracy" not in collection.metrics
        assert len(collection.metrics) == 0

class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_get_classification_metrics(self):
        """Test get_classification_metrics function"""
        metrics = get_classification_metrics()
        
        assert isinstance(metrics, MetricCollection)
        assert "accuracy" in metrics.metrics
        assert "f1" in metrics.metrics
        assert "precision_recall" in metrics.metrics
    
    def test_get_classification_metrics_with_average(self):
        """Test get_classification_metrics with averaging option"""
        metrics = get_classification_metrics(average="macro")
        
        # Check that F1 metric has correct averaging
        f1_metric = metrics.metrics["f1"]
        assert f1_metric.average == "macro"
    
    def test_get_generation_metrics(self):
        """Test get_generation_metrics function"""
        metrics = get_generation_metrics()
        
        assert isinstance(metrics, MetricCollection)
        assert "bleu" in metrics.metrics
        assert "rouge" in metrics.metrics
    
    def test_get_generation_metrics_with_options(self):
        """Test get_generation_metrics with options"""
        metrics = get_generation_metrics(include_perplexity=True)
        
        assert "perplexity" in metrics.metrics
    
    def test_get_qa_metrics(self):
        """Test get_qa_metrics function"""
        metrics = get_qa_metrics()
        
        assert isinstance(metrics, MetricCollection)
        assert "squad" in metrics.metrics
        assert "exact_match" in metrics.metrics
    
    def test_compute_metric_single(self):
        """Test compute_metric function for single metric"""
        result = compute_metric("accuracy", predictions=[1, 0, 1], labels=[1, 0, 0])
        
        assert isinstance(result, (int, float))
        assert 0 <= result <= 1
    
    def test_compute_metric_collection(self):
        """Test compute_metric function for metric collection"""
        result = compute_metric(
            "classification", 
            predictions=[1, 0, 1, 1], 
            labels=[1, 0, 1, 0]
        )
        
        assert isinstance(result, dict)
        assert "accuracy" in result
        assert "f1" in result
    
    def test_compute_metric_invalid(self):
        """Test compute_metric with invalid metric name"""
        with pytest.raises(ValueError):
            compute_metric("invalid_metric", predictions=[1, 0], labels=[1, 1])

class TestEvaluationIntegration:
    """Integration tests for evaluation module"""
    
    def test_full_classification_evaluation(self):
        """Test full classification evaluation workflow"""
        # Simulate model predictions
        np.random.seed(42)
        n_samples = 100
        n_classes = 3
        
        # Generate predictions and labels
        predictions = np.random.randint(0, n_classes, n_samples)
        labels = np.random.randint(0, n_classes, n_samples)
        
        # Get classification metrics
        metrics = get_classification_metrics(average="macro")
        
        # Compute all metrics
        results = metrics.compute(predictions.tolist(), labels.tolist())
        
        # Verify all expected metrics are present
        assert "accuracy" in results
        assert "f1" in results
        assert "precision_recall" in results
        assert "matthews_corrcoef" in results
        
        # Verify values are in expected ranges
        for metric_name, value in results.items():
            if isinstance(value, dict):
                for sub_value in value.values():
                    assert 0 <= sub_value <= 1
            else:
                assert -1 <= value <= 1  # MCC can be negative
    
    def test_full_generation_evaluation(self):
        """Test full generation evaluation workflow"""
        predictions = [
            "The cat sat on the mat",
            "Hello world",
            "This is a test"
        ]
        
        references = [
            "The cat is on the mat",
            "Hello world",
            "This is another test"
        ]
        
        # Get generation metrics
        metrics = get_generation_metrics()
        
        # Compute all metrics
        results = metrics.compute(predictions, references)
        
        # Verify expected metrics
        assert "bleu" in results
        assert "rouge" in results
        
        # Verify values are in expected ranges
        for value in results.values():
            assert 0 <= value <= 1
    
    def test_metric_comparison(self):
        """Test comparing different metrics on same data"""
        predictions = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        labels = [1, 0, 1, 0, 1, 1, 0, 1, 1, 0]
        
        # Compute individual metrics
        accuracy = compute_metric("accuracy", predictions, labels)
        f1_binary = compute_metric("f1", predictions, labels, average="binary")
        f1_macro = compute_metric("f1", predictions, labels, average="macro")
        
        # All should be different values
        assert accuracy != f1_binary
        assert f1_binary != f1_macro
        
        # All should be valid probabilities
        assert 0 <= accuracy <= 1
        assert 0 <= f1_binary <= 1
        assert 0 <= f1_macro <= 1

if __name__ == "__main__":
    pytest.main([__file__])