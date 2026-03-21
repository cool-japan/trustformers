#!/usr/bin/env python3
"""
Comprehensive tests for Data Science Tools Integration

Tests sklearn, matplotlib, and plotly integration features for optional dependencies.
"""

import pytest
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, List, Optional, Union

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

try:
    from trustformers.data_science_tools import (
        DataScienceConfig,
        PandasIntegration,
        SklearnIntegration,
        VisualizationTools,
        DataScienceWorkflow,
        get_data_science_workflow,
        to_dataframe,
        from_dataframe,
        plot_tensor,
        analyze_data,
        create_sklearn_pipeline,
        HAS_PANDAS,
        HAS_SKLEARN,
        HAS_MATPLOTLIB,
        HAS_SEABORN,
        HAS_PLOTLY
    )
except ImportError as e:
    pytest.skip(f"Could not import data_science_tools: {e}", allow_module_level=True)

try:
    from trustformers import Tensor
except ImportError:
    # Mock Tensor for testing
    class MockTensor:
        def __init__(self, data):
            self.data = np.array(data)
        
        def numpy(self):
            return self.data
        
        def to_numpy(self):
            return self.data
        
        def shape(self):
            return self.data.shape
    
    Tensor = MockTensor

# Optional imports for testing
try:
    import pandas as pd
    _pandas_available = True
except ImportError:
    pd = None
    _pandas_available = False

try:
    from sklearn.base import BaseEstimator
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    _sklearn_available = True
except ImportError:
    _sklearn_available = False

try:
    import matplotlib.pyplot as plt
    _matplotlib_available = True
except ImportError:
    plt = None
    _matplotlib_available = False

try:
    import seaborn as sns
    _seaborn_available = True
except ImportError:
    sns = None
    _seaborn_available = False

try:
    import plotly.graph_objects as go
    _plotly_available = True
except ImportError:
    go = None
    _plotly_available = False


class TestDataScienceConfig:
    """Test DataScienceConfig functionality."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = DataScienceConfig()
        
        assert config.figure_size == (10, 6)
        assert config.style == "whitegrid"
        assert config.color_palette == "husl"
        assert config.interactive is True
        assert config.save_plots is False
        assert config.plot_dir == "./plots"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = DataScienceConfig(
            figure_size=(12, 8),
            style="darkgrid",
            color_palette="Set2",
            interactive=False,
            save_plots=True,
            plot_dir="/tmp/test_plots"
        )
        
        assert config.figure_size == (12, 8)
        assert config.style == "darkgrid"
        assert config.color_palette == "Set2"
        assert config.interactive is False
        assert config.save_plots is True
        assert config.plot_dir == "/tmp/test_plots"


class TestPandasIntegration:
    """Test Pandas integration functionality."""
    
    def test_has_pandas_flag(self):
        """Test that HAS_PANDAS flag matches actual pandas availability."""
        assert HAS_PANDAS == _pandas_available
    
    @pytest.mark.skipif(not _pandas_available, reason="pandas not available")
    def test_tensor_to_dataframe_basic(self):
        """Test basic tensor to DataFrame conversion."""
        # Create test tensor
        data = [[1, 2, 3], [4, 5, 6]]
        tensor = Tensor(data)
        
        # Convert to DataFrame
        df = PandasIntegration.tensor_to_dataframe(tensor)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 3)
        np.testing.assert_array_equal(df.values, np.array(data))
    
    @pytest.mark.skipif(not _pandas_available, reason="pandas not available")
    def test_tensor_to_dataframe_with_columns(self):
        """Test tensor to DataFrame conversion with custom columns."""
        data = [[1, 2], [3, 4]]
        tensor = Tensor(data)
        columns = ['A', 'B']
        
        df = PandasIntegration.tensor_to_dataframe(tensor, columns=columns)
        
        assert list(df.columns) == columns
        assert df.shape == (2, 2)
    
    @pytest.mark.skipif(not _pandas_available, reason="pandas not available")
    def test_tensor_to_dataframe_1d(self):
        """Test 1D tensor to DataFrame conversion."""
        data = [1, 2, 3, 4]
        tensor = Tensor(data)
        
        df = PandasIntegration.tensor_to_dataframe(tensor)
        
        assert df.shape == (4, 1)
        np.testing.assert_array_equal(df.values.flatten(), np.array(data))
    
    @pytest.mark.skipif(not _pandas_available, reason="pandas not available")
    def test_tensor_to_dataframe_3d_flattens(self):
        """Test 3D tensor gets flattened for DataFrame conversion."""
        data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        tensor = Tensor(data)
        
        df = PandasIntegration.tensor_to_dataframe(tensor)
        
        assert df.shape == (2, 4)  # 2 samples, 4 features (flattened from 2x2)
    
    @pytest.mark.skipif(not _pandas_available, reason="pandas not available")
    def test_dataframe_to_tensor(self):
        """Test DataFrame to tensor conversion."""
        data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        df = pd.DataFrame(data)
        
        tensor = PandasIntegration.dataframe_to_tensor(df)
        
        # Should return tensor or array
        if hasattr(tensor, 'numpy'):
            result = tensor.numpy()
        else:
            result = tensor
        
        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(result, expected)
    
    @pytest.mark.skipif(not _pandas_available, reason="pandas not available")
    def test_create_feature_dataframe(self):
        """Test creating feature DataFrame from multiple tensors."""
        tensors = {
            'feature1': Tensor([[1], [2], [3]]),
            'feature2': Tensor([[4, 5], [6, 7], [8, 9]])
        }
        target = Tensor([0, 1, 0])
        
        df = PandasIntegration.create_feature_dataframe(tensors, target)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 4)  # 3 rows, 4 columns (1 + 2 + 1 target)
        assert 'feature1' in df.columns
        assert 'feature2_0' in df.columns
        assert 'feature2_1' in df.columns
        assert 'target' in df.columns
    
    @pytest.mark.skipif(not _pandas_available, reason="pandas not available")
    def test_analyze_dataframe(self):
        """Test DataFrame analysis functionality."""
        data = {
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [2.1, 3.2, 4.3, 5.4, 6.5],
            'categorical': ['A', 'B', 'A', 'C', 'B']
        }
        df = pd.DataFrame(data)
        
        analysis = PandasIntegration.analyze_dataframe(df)
        
        assert 'shape' in analysis
        assert analysis['shape'] == (5, 3)
        assert 'dtypes' in analysis
        assert 'memory_usage' in analysis
        assert 'missing_values' in analysis
        assert 'numeric_summary' in analysis
        assert 'categorical_summary' in analysis
        assert 'correlations' in analysis
        
        # Check numeric summary
        assert 'numeric1' in analysis['numeric_summary']
        assert 'numeric2' in analysis['numeric_summary']
        
        # Check categorical summary
        assert 'categorical' in analysis['categorical_summary']
        assert analysis['categorical_summary']['categorical']['unique_count'] == 3
    
    def test_pandas_not_available_errors(self):
        """Test that appropriate errors are raised when pandas is not available."""
        if _pandas_available:
            # Mock pandas as not available for this test
            with patch('trustformers.data_science_tools.HAS_PANDAS', False):
                tensor = Tensor([[1, 2], [3, 4]])
                
                with pytest.raises(ImportError, match="pandas is not installed"):
                    PandasIntegration.tensor_to_dataframe(tensor)


class TestSklearnIntegration:
    """Test Scikit-learn integration functionality."""
    
    def test_has_sklearn_flag(self):
        """Test that HAS_SKLEARN flag matches actual sklearn availability."""
        assert HAS_SKLEARN == _sklearn_available
    
    @pytest.mark.skipif(not _sklearn_available, reason="sklearn not available")
    def test_tensor_transformer_basic(self):
        """Test basic TensorTransformer functionality."""
        transformer = SklearnIntegration.TensorTransformer()
        
        # Test fitting
        X = Tensor([[1, 2], [3, 4]])
        transformer.fit(X)
        assert transformer.fitted_ is True
        
        # Test transform
        result = transformer.transform(X)
        expected = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(result, expected)
    
    @pytest.mark.skipif(not _sklearn_available, reason="sklearn not available")
    def test_tensor_transformer_custom_function(self):
        """Test TensorTransformer with custom transform function."""
        def custom_transform(X):
            return X.numpy() * 2 if hasattr(X, 'numpy') else X * 2
        
        transformer = SklearnIntegration.TensorTransformer(transform_func=custom_transform)
        
        X = Tensor([[1, 2], [3, 4]])
        transformer.fit(X)
        result = transformer.transform(X)
        
        expected = np.array([[2, 4], [6, 8]])
        np.testing.assert_array_equal(result, expected)
    
    @pytest.mark.skipif(not _sklearn_available, reason="sklearn not available")
    def test_tensor_estimator_basic(self):
        """Test basic TensorEstimator functionality."""
        # Mock model with predict method
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1, 0, 1])
        
        estimator = SklearnIntegration.TensorEstimator(mock_model)
        
        # Test fitting
        X = Tensor([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 0, 1])
        estimator.fit(X, y)
        assert estimator.fitted_ is True
        
        # Test prediction
        predictions = estimator.predict(X)
        np.testing.assert_array_equal(predictions, np.array([1, 0, 1]))
    
    @pytest.mark.skipif(not _sklearn_available, reason="sklearn not available")
    def test_tensor_classifier(self):
        """Test TensorClassifier functionality."""
        # Mock model with predict and predict_proba methods
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1, 0, 1])
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        
        classifier = SklearnIntegration.TensorClassifier(mock_model)
        
        X = Tensor([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 0, 1])
        classifier.fit(X, y)
        
        # Test prediction
        predictions = classifier.predict(X)
        np.testing.assert_array_equal(predictions, np.array([1, 0, 1]))
        
        # Test probability prediction
        probabilities = classifier.predict_proba(X)
        expected_proba = np.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        np.testing.assert_array_equal(probabilities, expected_proba)
    
    @pytest.mark.skipif(not _sklearn_available, reason="sklearn not available")
    def test_create_pipeline(self):
        """Test creating sklearn pipeline with Trustformers model."""
        # Mock model with predict method
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1, 0, 1])
        
        # Create pipeline
        pipeline = SklearnIntegration.create_pipeline(
            mock_model, 
            preprocessing_steps=[('scaler', Mock())]
        )
        
        assert hasattr(pipeline, 'steps')
        assert len(pipeline.steps) == 2  # preprocessing + model
        assert pipeline.steps[0][0] == 'scaler'
        assert pipeline.steps[1][0] == 'model'
    
    @pytest.mark.skipif(not _sklearn_available, reason="sklearn not available")
    def test_evaluate_model_classification(self):
        """Test model evaluation for classification tasks."""
        # Create mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1, 0, 1, 0, 1])
        
        # Test data
        X_test = Tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y_test = np.array([1, 0, 1, 0, 1])  # Perfect predictions
        
        metrics = SklearnIntegration.evaluate_model(
            mock_model, X_test, y_test, task_type='classification'
        )
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert metrics['accuracy'] == 1.0  # Perfect predictions
    
    @pytest.mark.skipif(not _sklearn_available, reason="sklearn not available")
    def test_evaluate_model_regression(self):
        """Test model evaluation for regression tasks."""
        # Create mock model
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Test data
        X_test = Tensor([[1], [2], [3], [4], [5]])
        y_test = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Perfect predictions
        
        metrics = SklearnIntegration.evaluate_model(
            mock_model, X_test, y_test, task_type='regression'
        )
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert 'rmse' in metrics
        assert metrics['mse'] == 0.0  # Perfect predictions
        assert metrics['r2'] == 1.0  # Perfect predictions
    
    def test_sklearn_not_available_errors(self):
        """Test that appropriate errors are raised when sklearn is not available."""
        if _sklearn_available:
            # Mock sklearn as not available for this test
            with patch('trustformers.data_science_tools.HAS_SKLEARN', False):
                mock_model = Mock()
                
                with pytest.raises(ImportError, match="scikit-learn is not installed"):
                    SklearnIntegration.create_pipeline(mock_model)


class TestVisualizationTools:
    """Test Visualization tools functionality."""
    
    def test_has_visualization_flags(self):
        """Test that visualization flags match actual library availability."""
        assert HAS_MATPLOTLIB == _matplotlib_available
        assert HAS_SEABORN == _seaborn_available
        assert HAS_PLOTLY == _plotly_available
    
    def test_visualization_tools_init(self):
        """Test VisualizationTools initialization."""
        viz = VisualizationTools()
        assert viz.config is not None
        
        # Test with custom config
        config = DataScienceConfig(figure_size=(8, 6))
        viz_custom = VisualizationTools(config)
        assert viz_custom.config.figure_size == (8, 6)
    
    @pytest.mark.skipif(not _matplotlib_available, reason="matplotlib not available")
    def test_plot_tensor_distribution_matplotlib(self):
        """Test tensor distribution plotting with matplotlib backend."""
        viz = VisualizationTools()
        tensor = Tensor([1, 2, 3, 4, 5, 2, 3, 4, 1, 5])
        
        # Should not raise an error
        fig = viz.plot_tensor_distribution(tensor, backend="matplotlib")
        assert fig is not None
    
    @pytest.mark.skipif(not _seaborn_available or not _matplotlib_available, 
                       reason="seaborn or matplotlib not available")
    def test_plot_tensor_distribution_seaborn(self):
        """Test tensor distribution plotting with seaborn backend."""
        viz = VisualizationTools()
        tensor = Tensor([1, 2, 3, 4, 5, 2, 3, 4, 1, 5])
        
        # Should not raise an error
        fig = viz.plot_tensor_distribution(tensor, backend="seaborn")
        assert fig is not None
    
    @pytest.mark.skipif(not _plotly_available, reason="plotly not available")
    def test_plot_tensor_distribution_plotly(self):
        """Test tensor distribution plotting with plotly backend."""
        viz = VisualizationTools(DataScienceConfig(interactive=False))
        tensor = Tensor([1, 2, 3, 4, 5, 2, 3, 4, 1, 5])
        
        # Should not raise an error
        fig = viz.plot_tensor_distribution(tensor, backend="plotly")
        assert fig is not None
    
    @pytest.mark.skipif(not _seaborn_available or not _matplotlib_available, 
                       reason="seaborn or matplotlib not available")
    def test_plot_tensor_heatmap_seaborn(self):
        """Test tensor heatmap plotting with seaborn backend."""
        viz = VisualizationTools()
        tensor = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Should not raise an error
        fig = viz.plot_tensor_heatmap(tensor, backend="seaborn")
        assert fig is not None
    
    @pytest.mark.skipif(not _matplotlib_available, reason="matplotlib not available")
    def test_plot_tensor_heatmap_matplotlib(self):
        """Test tensor heatmap plotting with matplotlib backend."""
        viz = VisualizationTools()
        tensor = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Should not raise an error
        fig = viz.plot_tensor_heatmap(tensor, backend="matplotlib")
        assert fig is not None
    
    @pytest.mark.skipif(not _plotly_available, reason="plotly not available")
    def test_plot_tensor_heatmap_plotly(self):
        """Test tensor heatmap plotting with plotly backend."""
        viz = VisualizationTools(DataScienceConfig(interactive=False))
        tensor = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Should not raise an error
        fig = viz.plot_tensor_heatmap(tensor, backend="plotly")
        assert fig is not None
    
    @pytest.mark.skipif(not _matplotlib_available, reason="matplotlib not available")
    def test_plot_training_metrics_matplotlib(self):
        """Test training metrics plotting with matplotlib backend."""
        viz = VisualizationTools()
        metrics = {
            'loss': [1.0, 0.8, 0.6, 0.4, 0.2],
            'accuracy': [0.6, 0.7, 0.8, 0.9, 0.95]
        }
        
        # Should not raise an error
        fig = viz.plot_training_metrics(metrics, backend="matplotlib")
        assert fig is not None
    
    @pytest.mark.skipif(not _plotly_available, reason="plotly not available")
    def test_plot_training_metrics_plotly(self):
        """Test training metrics plotting with plotly backend."""
        viz = VisualizationTools(DataScienceConfig(interactive=False))
        metrics = {
            'loss': [1.0, 0.8, 0.6, 0.4, 0.2],
            'accuracy': [0.6, 0.7, 0.8, 0.9, 0.95]
        }
        
        # Should not raise an error
        fig = viz.plot_training_metrics(metrics, backend="plotly")
        assert fig is not None
    
    @pytest.mark.skipif(not _seaborn_available or not _matplotlib_available, 
                       reason="seaborn or matplotlib not available")
    def test_plot_attention_weights_seaborn(self):
        """Test attention weights plotting with seaborn backend."""
        viz = VisualizationTools()
        attention_weights = Tensor([[0.8, 0.1, 0.1], [0.3, 0.4, 0.3], [0.2, 0.2, 0.6]])
        tokens = ['token1', 'token2', 'token3']
        
        # Should not raise an error
        fig = viz.plot_attention_weights(attention_weights, tokens, backend="seaborn")
        assert fig is not None
    
    @pytest.mark.skipif(not _plotly_available, reason="plotly not available")
    def test_plot_attention_weights_plotly(self):
        """Test attention weights plotting with plotly backend."""
        viz = VisualizationTools(DataScienceConfig(interactive=False))
        attention_weights = Tensor([[0.8, 0.1, 0.1], [0.3, 0.4, 0.3], [0.2, 0.2, 0.6]])
        tokens = ['token1', 'token2', 'token3']
        
        # Should not raise an error
        fig = viz.plot_attention_weights(attention_weights, tokens, backend="plotly")
        assert fig is not None
    
    @pytest.mark.skipif(not _seaborn_available or not _matplotlib_available or not _pandas_available, 
                       reason="seaborn, matplotlib, or pandas not available")
    def test_plot_model_comparison_seaborn(self):
        """Test model comparison plotting with seaborn backend."""
        viz = VisualizationTools()
        results = {
            'model1': {'accuracy': 0.85, 'f1': 0.82},
            'model2': {'accuracy': 0.90, 'f1': 0.88},
            'model3': {'accuracy': 0.87, 'f1': 0.85}
        }
        
        # Should not raise an error
        fig = viz.plot_model_comparison(results, backend="seaborn")
        assert fig is not None
    
    @pytest.mark.skipif(not _plotly_available, reason="plotly not available")
    def test_plot_model_comparison_plotly(self):
        """Test model comparison plotting with plotly backend."""
        viz = VisualizationTools(DataScienceConfig(interactive=False))
        results = {
            'model1': {'accuracy': 0.85, 'f1': 0.82},
            'model2': {'accuracy': 0.90, 'f1': 0.88},
            'model3': {'accuracy': 0.87, 'f1': 0.85}
        }
        
        # Should not raise an error
        fig = viz.plot_model_comparison(results, backend="plotly")
        assert fig is not None
    
    def test_plot_unavailable_backend_error(self):
        """Test that appropriate errors are raised for unavailable backends."""
        viz = VisualizationTools()
        tensor = Tensor([1, 2, 3])
        
        with pytest.raises(ImportError, match="Backend nonexistent is not available"):
            viz.plot_tensor_distribution(tensor, backend="nonexistent")


class TestDataScienceWorkflow:
    """Test DataScienceWorkflow functionality."""
    
    def test_workflow_initialization(self):
        """Test DataScienceWorkflow initialization."""
        workflow = DataScienceWorkflow()
        
        assert workflow.config is not None
        assert workflow.visualizer is not None
        assert workflow.pandas is not None
        assert workflow.sklearn is not None
    
    def test_global_workflow_singleton(self):
        """Test global workflow singleton behavior."""
        workflow1 = get_data_science_workflow()
        workflow2 = get_data_science_workflow()
        
        assert workflow1 is workflow2  # Same instance
    
    @pytest.mark.skipif(not _pandas_available, reason="pandas not available")
    def test_exploratory_data_analysis_tensor(self):
        """Test EDA with tensor input."""
        workflow = DataScienceWorkflow()
        tensor = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Should not raise an error
        results = workflow.exploratory_data_analysis(tensor)
        
        assert 'analysis' in results
        assert 'plots' in results
        assert isinstance(results['analysis'], dict)
    
    @pytest.mark.skipif(not _pandas_available, reason="pandas not available")
    def test_exploratory_data_analysis_dataframe(self):
        """Test EDA with DataFrame input."""
        workflow = DataScienceWorkflow()
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2.1, 3.2, 4.3, 5.4, 6.5],
            'category': ['A', 'B', 'A', 'C', 'B']
        })
        
        # Should not raise an error
        results = workflow.exploratory_data_analysis(df)
        
        assert 'analysis' in results
        assert 'plots' in results
        assert isinstance(results['analysis'], dict)
    
    @pytest.mark.skipif(not _sklearn_available, reason="sklearn not available")
    def test_create_ml_pipeline(self):
        """Test ML pipeline creation."""
        workflow = DataScienceWorkflow()
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1, 0, 1])
        
        # Test basic pipeline
        pipeline = workflow.create_ml_pipeline(mock_model)
        assert hasattr(pipeline, 'steps')
        
        # Test pipeline with preprocessing
        pipeline_with_preprocessing = workflow.create_ml_pipeline(
            mock_model, 
            preprocessing_steps=['scaler', 'tensor_transform']
        )
        assert len(pipeline_with_preprocessing.steps) == 3  # 2 preprocessing + model
    
    @pytest.mark.skipif(not _sklearn_available, reason="sklearn not available")
    def test_benchmark_models(self):
        """Test model benchmarking."""
        workflow = DataScienceWorkflow()
        
        # Create mock models
        model1 = Mock()
        model1.predict.return_value = np.array([1, 0, 1, 0])
        
        model2 = Mock()
        model2.predict.return_value = np.array([1, 0, 0, 0])
        
        models = {'model1': model1, 'model2': model2}
        
        # Create test data
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([1, 0, 1, 0])
        X_test = np.array([[9, 10], [11, 12], [13, 14], [15, 16]])
        y_test = np.array([1, 0, 1, 0])
        
        # Should not raise an error
        results = workflow.benchmark_models(
            models, X_train, y_train, X_test, y_test, task_type='classification'
        )
        
        assert 'model1' in results
        assert 'model2' in results
        assert isinstance(results['model1'], dict)
        assert isinstance(results['model2'], dict)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.mark.skipif(not _pandas_available, reason="pandas not available")
    def test_to_dataframe_convenience(self):
        """Test to_dataframe convenience function."""
        tensor = Tensor([[1, 2], [3, 4]])
        df = to_dataframe(tensor)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
    
    @pytest.mark.skipif(not _pandas_available, reason="pandas not available")
    def test_from_dataframe_convenience(self):
        """Test from_dataframe convenience function."""
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        tensor = from_dataframe(df)
        
        # Should return tensor-like object
        assert tensor is not None
    
    def test_plot_tensor_convenience(self):
        """Test plot_tensor convenience function."""
        tensor = Tensor([1, 2, 3, 4, 5])
        
        # Test different plot types
        if _matplotlib_available:
            # Should not raise an error
            fig = plot_tensor(tensor, plot_type="distribution")
            assert fig is not None
        
        # Test invalid plot type
        with pytest.raises(ValueError, match="Unknown plot type"):
            plot_tensor(tensor, plot_type="invalid")
    
    @pytest.mark.skipif(not _pandas_available, reason="pandas not available")
    def test_analyze_data_convenience(self):
        """Test analyze_data convenience function."""
        tensor = Tensor([[1, 2], [3, 4], [5, 6]])
        
        # Should not raise an error
        results = analyze_data(tensor)
        assert 'analysis' in results
    
    @pytest.mark.skipif(not _sklearn_available, reason="sklearn not available")
    def test_create_sklearn_pipeline_convenience(self):
        """Test create_sklearn_pipeline convenience function."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([1, 0, 1])
        
        # Should not raise an error
        pipeline = create_sklearn_pipeline(mock_model)
        assert hasattr(pipeline, 'steps')


class TestIntegrationEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.skipif(not (_pandas_available and _sklearn_available and _matplotlib_available), 
                       reason="pandas, sklearn, or matplotlib not available")
    def test_complete_ml_workflow(self):
        """Test complete ML workflow from data to visualization."""
        # Create sample data
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        
        # Convert to tensors
        X_tensor = Tensor(X.tolist())
        y_tensor = Tensor(y.tolist())
        
        # Create DataFrame
        df = to_dataframe(X_tensor, columns=[f'feature_{i}' for i in range(4)])
        
        # Add target
        df['target'] = y
        
        # Perform EDA
        workflow = DataScienceWorkflow()
        eda_results = workflow.exploratory_data_analysis(df)
        
        assert 'analysis' in eda_results
        assert 'plots' in eda_results
        
        # Create mock model for testing
        mock_model = Mock()
        mock_model.predict.return_value = y[:20]  # First 20 samples for testing
        
        # Create pipeline
        pipeline = workflow.create_ml_pipeline(mock_model, preprocessing_steps=['scaler'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train and evaluate
        pipeline.fit(X_train, y_train)
        
        metrics = SklearnIntegration.evaluate_model(
            pipeline, X_test, y_test, task_type='classification'
        )
        
        assert 'accuracy' in metrics
        assert isinstance(metrics['accuracy'], float)
    
    @pytest.mark.skipif(not (_matplotlib_available and _plotly_available), 
                       reason="matplotlib or plotly not available")
    def test_visualization_backend_switching(self):
        """Test switching between visualization backends."""
        viz = VisualizationTools()
        tensor = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        # Test matplotlib backend
        fig_mpl = viz.plot_tensor_heatmap(tensor, backend="matplotlib")
        assert fig_mpl is not None
        
        # Test plotly backend
        viz_plotly = VisualizationTools(DataScienceConfig(interactive=False))
        fig_plotly = viz_plotly.plot_tensor_heatmap(tensor, backend="plotly")
        assert fig_plotly is not None
        
        # Both should work without errors
        assert fig_mpl is not fig_plotly  # Different objects


if __name__ == "__main__":
    pytest.main([__file__, "-v"])