"""
Jupyter notebook support for TrustformeRS

Provides rich display capabilities, progress bars, and interactive widgets
for better integration with data science workflows.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Any, Dict, List, Optional, Union, Tuple
import base64
import io
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Try to import Jupyter dependencies
try:
    from IPython.display import display, HTML, Javascript, Markdown, Image
    from IPython.core.magic import Magics, magics_class, line_magic, cell_magic
    from IPython.core.magic_arguments import (argument, magic_arguments, parse_argstring)
    from IPython import get_ipython
    HAS_JUPYTER = True
except ImportError:
    HAS_JUPYTER = False
    # Create dummy classes for when Jupyter is not available
    class HTML:
        def __init__(self, data):
            self.data = data
    
    class Javascript:
        def __init__(self, data):
            self.data = data
    
    class Markdown:
        def __init__(self, data):
            self.data = data
    
    class Image:
        def __init__(self, data):
            self.data = data
    
    def display(*args, **kwargs):
        pass

# Try to import tqdm for progress bars
try:
    from tqdm.notebook import tqdm
    HAS_TQDM = True
except ImportError:
    try:
        from tqdm import tqdm
        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False
        class tqdm:
            def __init__(self, iterable=None, **kwargs):
                self.iterable = iterable
                self.kwargs = kwargs
            
            def __iter__(self):
                return iter(self.iterable)
            
            def __enter__(self):
                return self
            
            def __exit__(self, *args):
                pass
            
            def update(self, n=1):
                pass
            
            def close(self):
                pass


def check_jupyter_availability() -> bool:
    """Check if Jupyter is available."""
    return HAS_JUPYTER


def is_notebook_environment() -> bool:
    """Check if running in a Jupyter notebook."""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except ImportError:
        pass
    return False


class TensorDisplay:
    """Rich display for tensors in Jupyter notebooks."""
    
    @staticmethod
    def _tensor_to_html(tensor: Any, max_elements: int = 100) -> str:
        """Convert tensor to HTML representation."""
        try:
            # Get tensor info
            if hasattr(tensor, 'shape'):
                shape = tensor.shape
            else:
                shape = "Unknown"
            
            if hasattr(tensor, 'dtype'):
                dtype = tensor.dtype
            else:
                dtype = "Unknown"
            
            if hasattr(tensor, 'device'):
                device = tensor.device
            else:
                device = "Unknown"
            
            # Create HTML
            html = f"""
            <div style="border: 1px solid #ddd; padding: 10px; margin: 5px; border-radius: 5px;">
                <h4 style="margin: 0 0 10px 0; color: #333;">Tensor Information</h4>
                <table style="width: 100%; border-collapse: collapse;">
                    <tr><td><strong>Shape:</strong></td><td>{shape}</td></tr>
                    <tr><td><strong>Dtype:</strong></td><td>{dtype}</td></tr>
                    <tr><td><strong>Device:</strong></td><td>{device}</td></tr>
            """
            
            # Add memory info if available
            if hasattr(tensor, 'nbytes'):
                memory_mb = tensor.nbytes / (1024 * 1024)
                html += f"<tr><td><strong>Memory:</strong></td><td>{memory_mb:.2f} MB</td></tr>"
            
            # Add gradient info if available
            if hasattr(tensor, 'requires_grad'):
                html += f"<tr><td><strong>Requires Grad:</strong></td><td>{tensor.requires_grad}</td></tr>"
            
            html += "</table>"
            
            # Show tensor values if small enough
            if hasattr(tensor, 'numpy') or hasattr(tensor, 'data'):
                try:
                    if hasattr(tensor, 'numpy'):
                        data = tensor.numpy()
                    else:
                        data = tensor.data
                    
                    if isinstance(data, np.ndarray):
                        if data.size <= max_elements:
                            html += f"<h5 style='margin: 10px 0 5px 0;'>Values:</h5>"
                            html += f"<pre style='background: #f5f5f5; padding: 10px; overflow-x: auto;'>{data}</pre>"
                        else:
                            html += f"<p><em>Tensor too large to display (size: {data.size})</em></p>"
                        
                        # Add statistics
                        if data.size > 0:
                            html += f"""
                            <h5 style='margin: 10px 0 5px 0;'>Statistics:</h5>
                            <table style="width: 100%; border-collapse: collapse;">
                                <tr><td><strong>Min:</strong></td><td>{np.min(data):.6f}</td></tr>
                                <tr><td><strong>Max:</strong></td><td>{np.max(data):.6f}</td></tr>
                                <tr><td><strong>Mean:</strong></td><td>{np.mean(data):.6f}</td></tr>
                                <tr><td><strong>Std:</strong></td><td>{np.std(data):.6f}</td></tr>
                                <tr><td><strong>Zeros:</strong></td><td>{np.sum(data == 0)}/{data.size}</td></tr>
                            </table>
                            """
                except Exception as e:
                    html += f"<p><em>Error displaying tensor data: {e}</em></p>"
            
            html += "</div>"
            return html
        except Exception as e:
            return f"<div>Error creating tensor display: {e}</div>"
    
    @staticmethod
    def display_tensor(tensor: Any) -> None:
        """Display tensor in Jupyter notebook."""
        if not is_notebook_environment():
            print(f"Tensor shape: {getattr(tensor, 'shape', 'Unknown')}")
            return
        
        html = TensorDisplay._tensor_to_html(tensor)
        display(HTML(html))
    
    @staticmethod
    def display_tensor_comparison(tensor1: Any, tensor2: Any, names: Tuple[str, str] = ("Tensor 1", "Tensor 2")) -> None:
        """Display comparison of two tensors."""
        if not is_notebook_environment():
            print(f"{names[0]} shape: {getattr(tensor1, 'shape', 'Unknown')}")
            print(f"{names[1]} shape: {getattr(tensor2, 'shape', 'Unknown')}")
            return
        
        html = f"""
        <div style="display: flex; gap: 20px;">
            <div style="flex: 1;">
                <h3>{names[0]}</h3>
                {TensorDisplay._tensor_to_html(tensor1)}
            </div>
            <div style="flex: 1;">
                <h3>{names[1]}</h3>
                {TensorDisplay._tensor_to_html(tensor2)}
            </div>
        </div>
        """
        display(HTML(html))


class ModelDisplay:
    """Rich display for models in Jupyter notebooks."""
    
    @staticmethod
    def display_model_summary(model: Any) -> None:
        """Display model summary."""
        if not is_notebook_environment():
            print(f"Model: {type(model).__name__}")
            return
        
        try:
            # Get model info
            model_name = type(model).__name__
            
            # Try to get parameter count
            param_count = 0
            if hasattr(model, 'parameters'):
                try:
                    param_count = sum(p.numel() for p in model.parameters())
                except:
                    pass
            
            # Try to get model config
            config_info = ""
            if hasattr(model, 'config'):
                try:
                    config = model.config
                    config_info = f"""
                    <h5>Configuration:</h5>
                    <ul>
                        <li><strong>Hidden Size:</strong> {getattr(config, 'hidden_size', 'Unknown')}</li>
                        <li><strong>Num Layers:</strong> {getattr(config, 'num_hidden_layers', 'Unknown')}</li>
                        <li><strong>Num Heads:</strong> {getattr(config, 'num_attention_heads', 'Unknown')}</li>
                        <li><strong>Vocab Size:</strong> {getattr(config, 'vocab_size', 'Unknown')}</li>
                        <li><strong>Max Position:</strong> {getattr(config, 'max_position_embeddings', 'Unknown')}</li>
                    </ul>
                    """
                except:
                    pass
            
            html = f"""
            <div style="border: 1px solid #ddd; padding: 15px; margin: 10px; border-radius: 5px;">
                <h3 style="margin: 0 0 15px 0; color: #333;">Model Summary: {model_name}</h3>
                <p><strong>Parameters:</strong> {param_count:,}</p>
                {config_info}
            </div>
            """
            
            display(HTML(html))
        except Exception as e:
            display(HTML(f"<div>Error displaying model: {e}</div>"))


class TrainingDisplay:
    """Rich display for training progress."""
    
    def __init__(self):
        self.metrics_history = []
        self.current_epoch = 0
        self.total_epochs = 0
        self.progress_bar = None
        
    def start_training(self, total_epochs: int) -> None:
        """Start training display."""
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.metrics_history = []
        
        if HAS_TQDM:
            self.progress_bar = tqdm(total=total_epochs, desc="Training")
    
    def update_epoch(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Update epoch progress."""
        self.current_epoch = epoch
        self.metrics_history.append(metrics)
        
        if self.progress_bar:
            self.progress_bar.update(1)
            
            # Update description with metrics
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.progress_bar.set_description(f"Epoch {epoch}/{self.total_epochs} - {metric_str}")
    
    def finish_training(self) -> None:
        """Finish training display."""
        if self.progress_bar:
            self.progress_bar.close()
        
        # Display final metrics
        self.display_training_summary()
    
    def display_training_summary(self) -> None:
        """Display training summary."""
        if not is_notebook_environment() or not self.metrics_history:
            return
        
        # Create plots
        fig, axes = plt.subplots(1, len(self.metrics_history[0]), figsize=(15, 4))
        if len(self.metrics_history[0]) == 1:
            axes = [axes]
        
        for i, metric_name in enumerate(self.metrics_history[0].keys()):
            values = [m[metric_name] for m in self.metrics_history]
            axes[i].plot(values)
            axes[i].set_title(f"{metric_name.title()}")
            axes[i].set_xlabel("Epoch")
            axes[i].set_ylabel(metric_name.title())
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class InteractiveVisualization:
    """Interactive visualization widgets."""
    
    @staticmethod
    def tensor_heatmap(tensor: Any, title: str = "Tensor Heatmap") -> None:
        """Display tensor as heatmap."""
        if not is_notebook_environment():
            print(f"Tensor heatmap: {title}")
            return
        
        try:
            # Get tensor data
            if hasattr(tensor, 'numpy'):
                data = tensor.numpy()
            elif hasattr(tensor, 'data'):
                data = tensor.data
            else:
                display(HTML("<p>Cannot extract data from tensor</p>"))
                return
            
            # Handle different tensor shapes
            if data.ndim == 1:
                data = data.reshape(1, -1)
            elif data.ndim > 2:
                # For higher dimensions, show first 2D slice
                data = data.reshape(data.shape[0], -1)
            
            # Create heatmap
            plt.figure(figsize=(10, 6))
            plt.imshow(data, cmap='viridis', aspect='auto')
            plt.colorbar()
            plt.title(title)
            plt.xlabel('Feature')
            plt.ylabel('Sample')
            plt.show()
        except Exception as e:
            display(HTML(f"<p>Error creating heatmap: {e}</p>"))
    
    @staticmethod
    def attention_visualization(attention_weights: Any, tokens: List[str], title: str = "Attention Weights") -> None:
        """Visualize attention weights."""
        if not is_notebook_environment():
            print(f"Attention visualization: {title}")
            return
        
        try:
            # Get attention data
            if hasattr(attention_weights, 'numpy'):
                data = attention_weights.numpy()
            elif hasattr(attention_weights, 'data'):
                data = attention_weights.data
            else:
                display(HTML("<p>Cannot extract attention data</p>"))
                return
            
            # Handle different attention shapes
            if data.ndim > 2:
                # Take first head/layer
                while data.ndim > 2:
                    data = data[0]
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot heatmap
            im = ax.imshow(data, cmap='Blues', aspect='auto')
            
            # Set ticks
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right')
            ax.set_yticklabels(tokens)
            
            # Add colorbar
            plt.colorbar(im, ax=ax)
            
            # Add title and labels
            ax.set_title(title)
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            display(HTML(f"<p>Error creating attention visualization: {e}</p>"))


class ProgressBar:
    """Enhanced progress bar for Jupyter notebooks."""
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description
        self.current = 0
        
        if HAS_TQDM:
            self.pbar = tqdm(total=total, desc=description)
        else:
            self.pbar = None
    
    def update(self, n: int = 1) -> None:
        """Update progress."""
        self.current += n
        if self.pbar:
            self.pbar.update(n)
    
    def set_description(self, description: str) -> None:
        """Set description."""
        self.description = description
        if self.pbar:
            self.pbar.set_description(description)
    
    def close(self) -> None:
        """Close progress bar."""
        if self.pbar:
            self.pbar.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# Magic commands for Jupyter
if HAS_JUPYTER:
    @magics_class
    class TrustformersMagics(Magics):
        """Magic commands for TrustformeRS."""
        
        @line_magic
        def trustformers_debug(self, line):
            """Show debugging information."""
            from .debugging import get_debug_report
            
            report = get_debug_report()
            
            # Format report as HTML
            html = "<h3>TrustformeRS Debug Report</h3>"
            html += f"<p><strong>Total Tensors:</strong> {report['tensor_stats']['total_tensors']}</p>"
            html += f"<p><strong>Total Memory:</strong> {report['tensor_stats']['total_memory_mb']:.2f} MB</p>"
            
            if report['memory_leaks']:
                html += "<h4>Memory Leaks:</h4><ul>"
                for leak in report['memory_leaks'][:5]:  # Show top 5
                    html += f"<li>Tensor {leak['tensor_id']}: {leak['memory_mb']:.2f} MB, Age: {leak['age_seconds']:.1f}s</li>"
                html += "</ul>"
            
            display(HTML(html))
        
        @line_magic
        @magic_arguments()
        @argument('--cache-type', choices=['model', 'tokenizer', 'result'], default='model')
        def trustformers_cache(self, line):
            """Show cache information."""
            from .caching import get_cache_stats
            
            args = parse_argstring(self.trustformers_cache, line)
            stats = get_cache_stats()
            
            if args.cache_type == 'model':
                cache_stats = stats['model_cache']
            elif args.cache_type == 'tokenizer':
                cache_stats = stats['tokenizer_cache']
            else:
                cache_stats = stats['result_cache']
            
            html = f"<h3>{args.cache_type.title()} Cache Statistics</h3>"
            html += f"<p><strong>Entries:</strong> {cache_stats.get('entries', 0)}</p>"
            html += f"<p><strong>Hit Rate:</strong> {cache_stats.get('hit_rate', 0):.2%}</p>"
            html += f"<p><strong>Size:</strong> {cache_stats.get('size_mb', 0):.2f} MB</p>"
            
            display(HTML(html))
        
        @cell_magic
        def trustformers_profile(self, line, cell):
            """Profile code execution."""
            from .debugging import performance_profiler
            
            operation_name = line.strip() or "cell_execution"
            
            with performance_profiler.profile_operation(operation_name):
                exec(cell)
            
            # Show profiling results
            stats = performance_profiler.get_operation_stats(operation_name)
            html = f"<h3>Profiling Results: {operation_name}</h3>"
            html += f"<p><strong>Duration:</strong> {stats.get('avg_duration_ms', 0):.2f} ms</p>"
            html += f"<p><strong>Memory Increase:</strong> {stats.get('avg_memory_increase_mb', 0):.2f} MB</p>"
            
            display(HTML(html))
    
    # Register magic commands
    def load_ipython_extension(ipython):
        """Load the extension."""
        ipython.register_magic_function(TrustformersMagics)
    
    # Auto-register if in Jupyter
    if is_notebook_environment():
        try:
            ip = get_ipython()
            if ip:
                ip.register_magic_function(TrustformersMagics)
        except Exception:
            pass


# Helper functions
def display_tensor(tensor: Any) -> None:
    """Display tensor with rich formatting."""
    TensorDisplay.display_tensor(tensor)


def display_model(model: Any) -> None:
    """Display model with rich formatting."""
    ModelDisplay.display_model_summary(model)


def create_training_display() -> TrainingDisplay:
    """Create a training display."""
    return TrainingDisplay()


def create_progress_bar(total: int, description: str = "Processing") -> ProgressBar:
    """Create a progress bar."""
    return ProgressBar(total, description)


def tensor_heatmap(tensor: Any, title: str = "Tensor Heatmap") -> None:
    """Display tensor as heatmap."""
    InteractiveVisualization.tensor_heatmap(tensor, title)


def attention_visualization(attention_weights: Any, tokens: List[str], title: str = "Attention Weights") -> None:
    """Visualize attention weights."""
    InteractiveVisualization.attention_visualization(attention_weights, tokens, title)


def setup_jupyter_environment() -> None:
    """Setup Jupyter environment with TrustformeRS enhancements."""
    if not is_notebook_environment():
        logger.warning("Not in Jupyter environment")
        return
    
    # Setup display for tensor types
    try:
        from IPython.core.formatters import DisplayFormatter
        
        # This would need to be adapted based on actual tensor types
        # For now, we'll skip the automatic registration
        pass
    except Exception as e:
        logger.warning(f"Failed to setup Jupyter environment: {e}")
    
    # Display setup message
    display(HTML("""
    <div style="background: #e8f4f8; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h4>TrustformeRS Jupyter Environment Ready!</h4>
        <p>Available magic commands:</p>
        <ul>
            <li><code>%trustformers_debug</code> - Show debug information</li>
            <li><code>%trustformers_cache</code> - Show cache statistics</li>
            <li><code>%%trustformers_profile</code> - Profile cell execution</li>
        </ul>
        <p>Use <code>import trustformers.jupyter_support</code> to access visualization functions.</p>
    </div>
    """))


# Auto-setup if imported in Jupyter
if is_notebook_environment():
    setup_jupyter_environment()