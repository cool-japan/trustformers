#!/usr/bin/env python3
"""
Data Science Tools Integration for Trustformers

This module provides comprehensive integration with popular data science libraries:
- Pandas DataFrames for data manipulation and analysis
- Scikit-learn compatibility for ML workflows
- Matplotlib visualizations for data exploration
- Seaborn statistical visualizations
- Plotly interactive visualizations
"""

import logging
import numpy as np
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Iterator
from collections import defaultdict
import json

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

try:
    from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, ClassifierMixin
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.pipeline import Pipeline
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    class BaseEstimator:
        pass
    class TransformerMixin:
        pass
    class RegressorMixin:
        pass
    class ClassifierMixin:
        pass
    class Pipeline:
        pass

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.animation as animation
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    HAS_MATPLOTLIB = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    sns = None
    HAS_SEABORN = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    HAS_PLOTLY = True
except ImportError:
    go = None
    px = None
    HAS_PLOTLY = False

# Import trustformers types
try:
    from . import Tensor, PreTrainedModel, AutoTokenizer
except ImportError:
    # Fallback for standalone usage
    Tensor = Any
    PreTrainedModel = Any
    AutoTokenizer = Any

logger = logging.getLogger(__name__)

@dataclass
class DataScienceConfig:
    """Configuration for data science operations."""
    figure_size: Tuple[int, int] = (10, 6)
    style: str = "whitegrid"
    color_palette: str = "husl"
    interactive: bool = True
    save_plots: bool = False
    plot_dir: str = "./plots"

class PandasIntegration:
    """Pandas integration utilities for Trustformers."""
    
    @staticmethod
    def tensor_to_dataframe(tensor: Tensor, 
                           columns: Optional[List[str]] = None,
                           index: Optional[List[str]] = None) -> 'pd.DataFrame':
        """Convert Tensor to pandas DataFrame.
        
        Args:
            tensor: Input tensor
            columns: Column names
            index: Index names
            
        Returns:
            pandas DataFrame
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is not installed. Install with: pip install pandas")
        
        # Convert tensor to numpy
        if hasattr(tensor, 'numpy'):
            data = tensor.numpy()
        elif hasattr(tensor, 'to_numpy'):
            data = tensor.to_numpy()
        else:
            raise ValueError(f"Cannot convert {type(tensor)} to numpy array")
        
        # Handle different tensor shapes
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        elif data.ndim > 2:
            # Flatten higher-dimensional tensors
            original_shape = data.shape
            data = data.reshape(original_shape[0], -1)
            logger.warning(f"Flattened tensor from shape {original_shape} to {data.shape}")
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns, index=index)
        
        return df
    
    @staticmethod
    def dataframe_to_tensor(df: 'pd.DataFrame', 
                           dtype: Optional[str] = None) -> Tensor:
        """Convert pandas DataFrame to Tensor.
        
        Args:
            df: Input DataFrame
            dtype: Target data type
            
        Returns:
            Tensor object
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is not installed. Install with: pip install pandas")
        
        # Convert to numpy
        data = df.values
        
        # Import Tensor constructor
        try:
            from . import Tensor
            return Tensor(data.tolist())
        except ImportError:
            # Fallback - return numpy array
            return data
    
    @staticmethod
    def create_feature_dataframe(tensors: Dict[str, Tensor],
                                target: Optional[Tensor] = None) -> 'pd.DataFrame':
        """Create feature DataFrame from multiple tensors.
        
        Args:
            tensors: Dictionary of tensor features
            target: Optional target tensor
            
        Returns:
            Combined DataFrame
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is not installed. Install with: pip install pandas")
        
        dfs = []
        
        for name, tensor in tensors.items():
            df = PandasIntegration.tensor_to_dataframe(tensor)
            
            # Add prefix to columns if multi-dimensional
            if df.shape[1] > 1:
                df.columns = [f"{name}_{i}" for i in range(df.shape[1])]
            else:
                df.columns = [name]
            
            dfs.append(df)
        
        # Combine all features
        result_df = pd.concat(dfs, axis=1)
        
        # Add target if provided
        if target is not None:
            target_df = PandasIntegration.tensor_to_dataframe(target, columns=['target'])
            result_df = pd.concat([result_df, target_df], axis=1)
        
        return result_df
    
    @staticmethod
    def analyze_dataframe(df: 'pd.DataFrame') -> Dict[str, Any]:
        """Perform comprehensive DataFrame analysis.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Analysis results
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is not installed. Install with: pip install pandas")
        
        analysis = {
            'shape': df.shape,
            'dtypes': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_summary': {},
            'categorical_summary': {},
            'correlations': None
        }
        
        # Numeric column analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis['numeric_summary'] = df[numeric_cols].describe().to_dict()
            
            # Correlation matrix
            if len(numeric_cols) > 1:
                analysis['correlations'] = df[numeric_cols].corr().to_dict()
        
        # Categorical column analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                analysis['categorical_summary'][col] = {
                    'unique_count': df[col].nunique(),
                    'top_values': df[col].value_counts().head().to_dict()
                }
        
        return analysis

class SklearnIntegration:
    """Scikit-learn integration utilities for Trustformers."""
    
    class TensorTransformer(BaseEstimator, TransformerMixin):
        """Scikit-learn transformer for Tensor objects."""
        
        def __init__(self, transform_func: Optional[Callable] = None):
            """Initialize transformer.
            
            Args:
                transform_func: Optional custom transform function
            """
            self.transform_func = transform_func
            self.fitted_ = False
        
        def fit(self, X, y=None):
            """Fit transformer (no-op for tensors)."""
            self.fitted_ = True
            return self
        
        def transform(self, X):
            """Transform tensor data."""
            if not self.fitted_:
                raise ValueError("Transformer must be fitted before transform")
            
            if self.transform_func:
                return self.transform_func(X)
            
            # Default: convert to numpy if tensor
            if hasattr(X, 'numpy'):
                return X.numpy()
            elif hasattr(X, 'to_numpy'):
                return X.to_numpy()
            else:
                return X
    
    class TensorEstimator(BaseEstimator):
        """Base estimator for Tensor-based models."""
        
        def __init__(self, model: Any):
            """Initialize with a Trustformers model.
            
            Args:
                model: Trustformers model instance
            """
            self.model = model
            self.fitted_ = False
        
        def fit(self, X, y=None):
            """Fit the model."""
            # Convert to appropriate format for model
            if hasattr(self.model, 'fit'):
                self.model.fit(X, y)
            elif hasattr(self.model, 'train'):
                self.model.train(X, y)
            
            self.fitted_ = True
            return self
        
        def predict(self, X):
            """Make predictions."""
            if not self.fitted_:
                raise ValueError("Model must be fitted before predict")
            
            if hasattr(self.model, 'predict'):
                return self.model.predict(X)
            elif hasattr(self.model, 'forward'):
                return self.model.forward(X)
            else:
                raise NotImplementedError("Model does not have predict or forward method")
    
    class TensorClassifier(TensorEstimator, ClassifierMixin):
        """Classifier wrapper for Trustformers models."""
        
        def predict_proba(self, X):
            """Predict class probabilities."""
            if hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(X)
            else:
                # Apply softmax to raw predictions
                predictions = self.predict(X)
                if hasattr(predictions, 'softmax'):
                    return predictions.softmax(dim=-1)
                else:
                    # Fallback using numpy
                    import numpy as np
                    exp_preds = np.exp(predictions - np.max(predictions, axis=-1, keepdims=True))
                    return exp_preds / np.sum(exp_preds, axis=-1, keepdims=True)
    
    class TensorRegressor(TensorEstimator, RegressorMixin):
        """Regressor wrapper for Trustformers models."""
        pass
    
    @staticmethod
    def create_pipeline(model: Any, 
                       preprocessing_steps: Optional[List[Tuple[str, Any]]] = None) -> Pipeline:
        """Create scikit-learn pipeline with Trustformers model.
        
        Args:
            model: Trustformers model
            preprocessing_steps: List of (name, transformer) tuples
            
        Returns:
            Scikit-learn Pipeline
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is not installed. Install with: pip install scikit-learn")
        
        steps = preprocessing_steps or []
        
        # Add model as final step
        if hasattr(model, 'predict'):
            # Determine if classifier or regressor
            if hasattr(model, 'predict_proba') or hasattr(model, 'classes_'):
                estimator = SklearnIntegration.TensorClassifier(model)
            else:
                estimator = SklearnIntegration.TensorRegressor(model)
        else:
            estimator = SklearnIntegration.TensorEstimator(model)
        
        steps.append(('model', estimator))
        
        return Pipeline(steps)
    
    @staticmethod
    def evaluate_model(model: Any, 
                      X_test: Any, 
                      y_test: Any,
                      task_type: str = 'classification') -> Dict[str, float]:
        """Evaluate model using scikit-learn metrics.
        
        Args:
            model: Model to evaluate
            X_test: Test features
            y_test: Test targets
            task_type: 'classification' or 'regression'
            
        Returns:
            Dictionary of metrics
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is not installed. Install with: pip install scikit-learn")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        metrics = {}
        
        if task_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            
            try:
                metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
                metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
                metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
            except Exception as e:
                logger.warning(f"Could not compute some classification metrics: {e}")
        
        elif task_type == 'regression':
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            metrics['mse'] = mean_squared_error(y_test, y_pred)
            metrics['mae'] = mean_absolute_error(y_test, y_pred)
            metrics['r2'] = r2_score(y_test, y_pred)
            metrics['rmse'] = np.sqrt(metrics['mse'])
        
        return metrics

class VisualizationTools:
    """Visualization utilities using matplotlib, seaborn, and plotly."""
    
    def __init__(self, config: Optional[DataScienceConfig] = None):
        """Initialize visualization tools.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or DataScienceConfig()
        self._setup_style()
    
    def _setup_style(self):
        """Setup visualization styles."""
        if HAS_MATPLOTLIB:
            plt.rcParams['figure.figsize'] = self.config.figure_size
        
        if HAS_SEABORN:
            sns.set_style(self.config.style)
            sns.set_palette(self.config.color_palette)
    
    def plot_tensor_distribution(self, 
                                tensor: Tensor,
                                title: str = "Tensor Distribution",
                                bins: int = 50,
                                backend: str = "matplotlib") -> Any:
        """Plot tensor value distribution.
        
        Args:
            tensor: Input tensor
            title: Plot title
            bins: Number of histogram bins
            backend: Plotting backend ('matplotlib', 'seaborn', 'plotly')
            
        Returns:
            Plot object
        """
        # Convert tensor to numpy
        if hasattr(tensor, 'numpy'):
            data = tensor.numpy().flatten()
        elif hasattr(tensor, 'to_numpy'):
            data = tensor.to_numpy().flatten()
        else:
            data = np.array(tensor).flatten()
        
        if backend == "matplotlib" and HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            ax.hist(data, bins=bins, alpha=0.7, edgecolor='black')
            ax.set_title(title)
            ax.set_xlabel("Value")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)
            
            if self.config.save_plots:
                self._save_plot(fig, f"{title.lower().replace(' ', '_')}_distribution")
            
            return fig
        
        elif backend == "seaborn" and HAS_SEABORN:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            sns.histplot(data, bins=bins, kde=True, ax=ax)
            ax.set_title(title)
            
            if self.config.save_plots:
                self._save_plot(fig, f"{title.lower().replace(' ', '_')}_distribution")
            
            return fig
        
        elif backend == "plotly" and HAS_PLOTLY:
            fig = go.Figure(data=[go.Histogram(x=data, nbinsx=bins)])
            fig.update_layout(
                title=title,
                xaxis_title="Value",
                yaxis_title="Frequency",
                showlegend=False
            )
            
            if self.config.interactive:
                fig.show()
            
            return fig
        
        else:
            raise ImportError(f"Backend {backend} is not available")
    
    def plot_tensor_heatmap(self,
                           tensor: Tensor,
                           title: str = "Tensor Heatmap",
                           backend: str = "seaborn") -> Any:
        """Plot tensor as heatmap.
        
        Args:
            tensor: Input tensor (2D)
            title: Plot title
            backend: Plotting backend
            
        Returns:
            Plot object
        """
        # Convert tensor to numpy
        if hasattr(tensor, 'numpy'):
            data = tensor.numpy()
        elif hasattr(tensor, 'to_numpy'):
            data = tensor.to_numpy()
        else:
            data = np.array(tensor)
        
        # Ensure 2D
        if data.ndim != 2:
            if data.ndim == 1:
                data = data.reshape(1, -1)
            else:
                # Take first 2 dimensions
                data = data.reshape(data.shape[0], -1)
        
        if backend == "seaborn" and HAS_SEABORN:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            sns.heatmap(data, cmap='viridis', center=0, ax=ax)
            ax.set_title(title)
            
            if self.config.save_plots:
                self._save_plot(fig, f"{title.lower().replace(' ', '_')}_heatmap")
            
            return fig
        
        elif backend == "matplotlib" and HAS_MATPLOTLIB:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            im = ax.imshow(data, cmap='viridis', aspect='auto')
            ax.set_title(title)
            plt.colorbar(im, ax=ax)
            
            if self.config.save_plots:
                self._save_plot(fig, f"{title.lower().replace(' ', '_')}_heatmap")
            
            return fig
        
        elif backend == "plotly" and HAS_PLOTLY:
            fig = go.Figure(data=go.Heatmap(z=data, colorscale='Viridis'))
            fig.update_layout(title=title)
            
            if self.config.interactive:
                fig.show()
            
            return fig
        
        else:
            raise ImportError(f"Backend {backend} is not available")
    
    def plot_training_metrics(self,
                             metrics: Dict[str, List[float]],
                             title: str = "Training Metrics",
                             backend: str = "matplotlib") -> Any:
        """Plot training metrics over time.
        
        Args:
            metrics: Dictionary of metric name to values
            title: Plot title
            backend: Plotting backend
            
        Returns:
            Plot object
        """
        if backend == "matplotlib" and HAS_MATPLOTLIB:
            fig, axes = plt.subplots(len(metrics), 1, figsize=(self.config.figure_size[0], 
                                                               self.config.figure_size[1] * len(metrics)))
            if len(metrics) == 1:
                axes = [axes]
            
            for i, (metric_name, values) in enumerate(metrics.items()):
                axes[i].plot(values, marker='o', markersize=3)
                axes[i].set_title(f"{metric_name.title()}")
                axes[i].set_xlabel("Epoch/Step")
                axes[i].set_ylabel(metric_name.title())
                axes[i].grid(True, alpha=0.3)
            
            plt.suptitle(title)
            plt.tight_layout()
            
            if self.config.save_plots:
                self._save_plot(fig, f"{title.lower().replace(' ', '_')}_metrics")
            
            return fig
        
        elif backend == "plotly" and HAS_PLOTLY:
            fig = make_subplots(
                rows=len(metrics), cols=1,
                subplot_titles=list(metrics.keys()),
                vertical_spacing=0.05
            )
            
            for i, (metric_name, values) in enumerate(metrics.items(), 1):
                fig.add_trace(
                    go.Scatter(y=values, mode='lines+markers', name=metric_name),
                    row=i, col=1
                )
            
            fig.update_layout(title=title, showlegend=False)
            
            if self.config.interactive:
                fig.show()
            
            return fig
        
        else:
            raise ImportError(f"Backend {backend} is not available")
    
    def plot_attention_weights(self,
                              attention_weights: Tensor,
                              input_tokens: Optional[List[str]] = None,
                              title: str = "Attention Weights",
                              backend: str = "seaborn") -> Any:
        """Plot attention weight heatmap.
        
        Args:
            attention_weights: Attention weight tensor
            input_tokens: Optional token labels
            title: Plot title
            backend: Plotting backend
            
        Returns:
            Plot object
        """
        # Convert to numpy and take 2D slice if needed
        if hasattr(attention_weights, 'numpy'):
            data = attention_weights.numpy()
        else:
            data = np.array(attention_weights)
        
        # Handle multi-dimensional attention (take first head/layer)
        while data.ndim > 2:
            data = data[0]
        
        if backend == "seaborn" and HAS_SEABORN:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            # Use tokens as labels if provided
            xticklabels = yticklabels = input_tokens if input_tokens else True
            
            sns.heatmap(data, 
                       xticklabels=xticklabels,
                       yticklabels=yticklabels,
                       cmap='Blues',
                       square=True,
                       ax=ax)
            ax.set_title(title)
            ax.set_xlabel("Key Position")
            ax.set_ylabel("Query Position")
            
            if self.config.save_plots:
                self._save_plot(fig, f"{title.lower().replace(' ', '_')}_attention")
            
            return fig
        
        elif backend == "plotly" and HAS_PLOTLY:
            fig = go.Figure(data=go.Heatmap(
                z=data,
                x=input_tokens,
                y=input_tokens,
                colorscale='Blues'
            ))
            fig.update_layout(
                title=title,
                xaxis_title="Key Position",
                yaxis_title="Query Position"
            )
            
            if self.config.interactive:
                fig.show()
            
            return fig
        
        else:
            raise ImportError(f"Backend {backend} is not available")
    
    def plot_model_comparison(self,
                             results: Dict[str, Dict[str, float]],
                             title: str = "Model Comparison",
                             backend: str = "seaborn") -> Any:
        """Plot model performance comparison.
        
        Args:
            results: Dict of {model_name: {metric: value}}
            title: Plot title
            backend: Plotting backend
            
        Returns:
            Plot object
        """
        if backend == "seaborn" and HAS_SEABORN and HAS_PANDAS:
            # Convert to DataFrame for easier plotting
            df_data = []
            for model_name, metrics in results.items():
                for metric_name, value in metrics.items():
                    df_data.append({
                        'Model': model_name,
                        'Metric': metric_name,
                        'Value': value
                    })
            
            df = pd.DataFrame(df_data)
            
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            sns.barplot(data=df, x='Metric', y='Value', hue='Model', ax=ax)
            ax.set_title(title)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if self.config.save_plots:
                self._save_plot(fig, f"{title.lower().replace(' ', '_')}_comparison")
            
            return fig
        
        elif backend == "plotly" and HAS_PLOTLY:
            fig = go.Figure()
            
            metrics = list(next(iter(results.values())).keys())
            
            for model_name, model_metrics in results.items():
                fig.add_trace(go.Bar(
                    name=model_name,
                    x=metrics,
                    y=[model_metrics[m] for m in metrics]
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Metrics",
                yaxis_title="Value",
                barmode='group'
            )
            
            if self.config.interactive:
                fig.show()
            
            return fig
        
        else:
            raise ImportError(f"Backend {backend} is not available or pandas is missing")
    
    def _save_plot(self, fig, filename: str):
        """Save plot to file."""
        if self.config.save_plots:
            import os
            os.makedirs(self.config.plot_dir, exist_ok=True)
            filepath = os.path.join(self.config.plot_dir, f"{filename}.png")
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {filepath}")

class DataScienceWorkflow:
    """High-level workflow for data science tasks."""
    
    def __init__(self, config: Optional[DataScienceConfig] = None):
        """Initialize workflow.
        
        Args:
            config: Configuration object
        """
        self.config = config or DataScienceConfig()
        self.visualizer = VisualizationTools(self.config)
        self.pandas = PandasIntegration()
        self.sklearn = SklearnIntegration()
    
    def exploratory_data_analysis(self, 
                                 data: Union[Tensor, 'pd.DataFrame'],
                                 target: Optional[Union[Tensor, str]] = None) -> Dict[str, Any]:
        """Perform comprehensive EDA.
        
        Args:
            data: Input data (Tensor or DataFrame)
            target: Target variable (Tensor or column name)
            
        Returns:
            EDA results and plots
        """
        results = {}
        
        # Convert to DataFrame if tensor
        if not isinstance(data, pd.DataFrame) if HAS_PANDAS else True:
            if HAS_PANDAS:
                df = self.pandas.tensor_to_dataframe(data)
            else:
                raise ImportError("pandas is required for EDA")
        else:
            df = data
        
        # Basic analysis
        results['analysis'] = self.pandas.analyze_dataframe(df)
        
        # Visualizations
        results['plots'] = {}
        
        # Distribution plots for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:5]:  # Limit to first 5 columns
            try:
                if hasattr(df[col], 'values'):
                    # Create a mock tensor-like object
                    class MockTensor:
                        def __init__(self, data):
                            self.data = data
                        def numpy(self):
                            return self.data
                    
                    mock_tensor = MockTensor(df[col].values)
                    plot = self.visualizer.plot_tensor_distribution(
                        mock_tensor, 
                        title=f"Distribution of {col}"
                    )
                    results['plots'][f'{col}_distribution'] = plot
            except Exception as e:
                logger.warning(f"Could not plot distribution for {col}: {e}")
        
        # Correlation heatmap
        if len(numeric_cols) > 1:
            try:
                corr_data = df[numeric_cols].corr().values
                class MockTensor:
                    def __init__(self, data):
                        self.data = data
                    def numpy(self):
                        return self.data
                
                mock_tensor = MockTensor(corr_data)
                corr_plot = self.visualizer.plot_tensor_heatmap(
                    mock_tensor,
                    title="Feature Correlations"
                )
                results['plots']['correlation_heatmap'] = corr_plot
            except Exception as e:
                logger.warning(f"Could not create correlation plot: {e}")
        
        return results
    
    def create_ml_pipeline(self,
                          model: Any,
                          preprocessing_steps: Optional[List[str]] = None) -> Pipeline:
        """Create end-to-end ML pipeline.
        
        Args:
            model: Trustformers model
            preprocessing_steps: List of preprocessing step names
            
        Returns:
            Scikit-learn pipeline
        """
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for ML pipelines")
        
        steps = []
        
        # Add preprocessing steps
        if preprocessing_steps:
            for step_name in preprocessing_steps:
                if step_name == 'scaler':
                    steps.append(('scaler', StandardScaler()))
                elif step_name == 'tensor_transform':
                    steps.append(('tensor_transform', self.sklearn.TensorTransformer()))
                # Add more preprocessing options as needed
        
        # Create pipeline
        pipeline = self.sklearn.create_pipeline(model, steps)
        
        return pipeline
    
    def benchmark_models(self,
                        models: Dict[str, Any],
                        X_train: Any,
                        y_train: Any,
                        X_test: Any,
                        y_test: Any,
                        task_type: str = 'classification') -> Dict[str, Any]:
        """Benchmark multiple models.
        
        Args:
            models: Dictionary of model name to model object
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            task_type: 'classification' or 'regression'
            
        Returns:
            Benchmark results
        """
        results = {}
        
        for name, model in models.items():
            try:
                # Create pipeline
                pipeline = self.create_ml_pipeline(model)
                
                # Train model
                pipeline.fit(X_train, y_train)
                
                # Evaluate
                metrics = self.sklearn.evaluate_model(
                    pipeline, X_test, y_test, task_type
                )
                
                results[name] = metrics
                
            except Exception as e:
                logger.error(f"Error benchmarking model {name}: {e}")
                results[name] = {'error': str(e)}
        
        # Create comparison plot
        try:
            comparison_plot = self.visualizer.plot_model_comparison(
                {k: v for k, v in results.items() if 'error' not in v},
                title="Model Performance Comparison"
            )
            results['comparison_plot'] = comparison_plot
        except Exception as e:
            logger.warning(f"Could not create comparison plot: {e}")
        
        return results

# Global workflow instance
_global_workflow = None

def get_data_science_workflow() -> DataScienceWorkflow:
    """Get global data science workflow instance."""
    global _global_workflow
    if _global_workflow is None:
        _global_workflow = DataScienceWorkflow()
    return _global_workflow

# Convenience functions
def to_dataframe(tensor: Tensor, **kwargs) -> 'pd.DataFrame':
    """Convert tensor to DataFrame."""
    return PandasIntegration.tensor_to_dataframe(tensor, **kwargs)

def from_dataframe(df: 'pd.DataFrame', **kwargs) -> Tensor:
    """Convert DataFrame to tensor."""
    return PandasIntegration.dataframe_to_tensor(df, **kwargs)

def plot_tensor(tensor: Tensor, plot_type: str = "distribution", **kwargs) -> Any:
    """Plot tensor using specified visualization."""
    viz = VisualizationTools()
    
    if plot_type == "distribution":
        return viz.plot_tensor_distribution(tensor, **kwargs)
    elif plot_type == "heatmap":
        return viz.plot_tensor_heatmap(tensor, **kwargs)
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")

def analyze_data(data: Union[Tensor, 'pd.DataFrame'], **kwargs) -> Dict[str, Any]:
    """Perform exploratory data analysis."""
    return get_data_science_workflow().exploratory_data_analysis(data, **kwargs)

def create_sklearn_pipeline(model: Any, **kwargs) -> Pipeline:
    """Create scikit-learn pipeline with Trustformers model."""
    return get_data_science_workflow().create_ml_pipeline(model, **kwargs)

__all__ = [
    'DataScienceConfig',
    'PandasIntegration',
    'SklearnIntegration', 
    'VisualizationTools',
    'DataScienceWorkflow',
    'get_data_science_workflow',
    'to_dataframe',
    'from_dataframe',
    'plot_tensor',
    'analyze_data',
    'create_sklearn_pipeline'
]