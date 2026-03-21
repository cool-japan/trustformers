"""
Model profiler for TrustformeRS.
Provides deep performance analysis including layer-wise profiling.
"""

import time
import json
import subprocess
import tempfile
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict
from collections import defaultdict
import re

@dataclass
class LayerProfile:
    """Profile data for a single layer."""
    name: str
    layer_type: str
    forward_time_ms: float
    backward_time_ms: float = 0.0
    memory_allocated_mb: float = 0.0
    memory_freed_mb: float = 0.0
    flops: int = 0
    parameters: int = 0
    input_shape: List[int] = None
    output_shape: List[int] = None

@dataclass 
class ProfileResult:
    """Complete profiling result for a model."""
    model_name: str
    device: str
    total_time_ms: float
    peak_memory_mb: float
    layers: List[LayerProfile]
    bottlenecks: List[str]
    optimization_suggestions: List[str]
    flop_utilization: float = 0.0
    memory_bandwidth_utilization: float = 0.0
    
class ModelProfiler:
    def __init__(self):
        self.profiling_configs = {
            'memory': {
                'track_allocations': True,
                'track_peak_memory': True,
                'track_memory_timeline': True
            },
            'compute': {
                'track_flops': True,
                'track_kernel_times': True,
                'track_cuda_events': True
            },
            'full': {
                'track_allocations': True,
                'track_peak_memory': True,
                'track_memory_timeline': True,
                'track_flops': True,
                'track_kernel_times': True,
                'track_cuda_events': True,
                'generate_trace': True
            }
        }
    
    def profile(self, model_name: str, profile_type: str = 'full',
                batch_size: int = 1, sequence_length: int = 128,
                device: str = 'cpu') -> ProfileResult:
        """Profile a model with specified configuration."""
        
        config = self.profiling_configs.get(profile_type, self.profiling_configs['full'])
        
        # Run profiling
        raw_profile = self._run_profiling(
            model_name, config, batch_size, sequence_length, device
        )
        
        # Analyze results
        layers = self._parse_layer_profiles(raw_profile)
        bottlenecks = self._identify_bottlenecks(layers)
        suggestions = self._generate_optimization_suggestions(
            layers, bottlenecks, raw_profile
        )
        
        # Calculate utilization metrics
        flop_util = self._calculate_flop_utilization(layers, raw_profile)
        mem_util = self._calculate_memory_bandwidth_utilization(layers, raw_profile)
        
        return ProfileResult(
            model_name=model_name,
            device=device,
            total_time_ms=raw_profile['total_time_ms'],
            peak_memory_mb=raw_profile['peak_memory_mb'],
            layers=layers,
            bottlenecks=bottlenecks,
            optimization_suggestions=suggestions,
            flop_utilization=flop_util,
            memory_bandwidth_utilization=mem_util
        )
    
    def _run_profiling(self, model_name: str, config: Dict[str, Any],
                      batch_size: int, sequence_length: int,
                      device: str) -> Dict[str, Any]:
        """Execute profiling and return raw results."""
        
        # Prepare environment
        env = os.environ.copy()
        env['TRUSTFORMERS_PROFILE_MODEL'] = model_name
        env['TRUSTFORMERS_PROFILE_DEVICE'] = device
        env['TRUSTFORMERS_PROFILE_BATCH_SIZE'] = str(batch_size)
        env['TRUSTFORMERS_PROFILE_SEQ_LENGTH'] = str(sequence_length)
        env['TRUSTFORMERS_PROFILE_CONFIG'] = json.dumps(config)
        
        # Output file for results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_file = f.name
            env['TRUSTFORMERS_PROFILE_OUTPUT'] = output_file
        
        try:
            # Run profiler
            cmd = ['cargo', 'run', '--release', '--bin', 'trustformers-profiler']
            
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(timeout=300)
            
            if process.returncode != 0:
                raise RuntimeError(f"Profiling failed: {stderr}")
            
            # Read results
            with open(output_file, 'r') as f:
                return json.load(f)
                
        finally:
            if os.path.exists(output_file):
                os.remove(output_file)
    
    def _parse_layer_profiles(self, raw_profile: Dict[str, Any]) -> List[LayerProfile]:
        """Parse layer-wise profiling data."""
        layers = []
        
        for layer_data in raw_profile.get('layers', []):
            layer = LayerProfile(
                name=layer_data['name'],
                layer_type=layer_data['type'],
                forward_time_ms=layer_data.get('forward_time_ms', 0),
                backward_time_ms=layer_data.get('backward_time_ms', 0),
                memory_allocated_mb=layer_data.get('memory_allocated_mb', 0),
                memory_freed_mb=layer_data.get('memory_freed_mb', 0),
                flops=layer_data.get('flops', 0),
                parameters=layer_data.get('parameters', 0),
                input_shape=layer_data.get('input_shape'),
                output_shape=layer_data.get('output_shape')
            )
            layers.append(layer)
        
        return layers
    
    def _identify_bottlenecks(self, layers: List[LayerProfile]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        if not layers:
            return bottlenecks
        
        # Calculate statistics
        total_time = sum(l.forward_time_ms for l in layers)
        avg_time = total_time / len(layers)
        time_threshold = avg_time * 3  # Layers taking 3x average time
        
        total_memory = sum(l.memory_allocated_mb for l in layers)
        avg_memory = total_memory / len(layers) if len(layers) > 0 else 0
        memory_threshold = avg_memory * 5  # Layers using 5x average memory
        
        # Find compute bottlenecks
        compute_bottlenecks = []
        for layer in layers:
            if layer.forward_time_ms > time_threshold:
                compute_bottlenecks.append({
                    'layer': layer.name,
                    'time': layer.forward_time_ms,
                    'percentage': (layer.forward_time_ms / total_time) * 100
                })
        
        # Sort by time
        compute_bottlenecks.sort(key=lambda x: x['time'], reverse=True)
        
        for bottleneck in compute_bottlenecks[:3]:  # Top 3
            bottlenecks.append(
                f"Layer '{bottleneck['layer']}' takes {bottleneck['time']:.1f}ms "
                f"({bottleneck['percentage']:.1f}% of total time)"
            )
        
        # Find memory bottlenecks
        memory_bottlenecks = []
        for layer in layers:
            if layer.memory_allocated_mb > memory_threshold:
                memory_bottlenecks.append({
                    'layer': layer.name,
                    'memory': layer.memory_allocated_mb
                })
        
        memory_bottlenecks.sort(key=lambda x: x['memory'], reverse=True)
        
        for bottleneck in memory_bottlenecks[:3]:  # Top 3
            bottlenecks.append(
                f"Layer '{bottleneck['layer']}' allocates {bottleneck['memory']:.1f}MB"
            )
        
        # Check for inefficient layers (low FLOP utilization)
        for layer in layers:
            if layer.flops > 0 and layer.forward_time_ms > 0:
                # Estimate GFLOPS
                gflops = (layer.flops / 1e9) / (layer.forward_time_ms / 1000)
                
                # Check against theoretical peak (example: 10 TFLOPS for modern GPU)
                theoretical_peak = 10000  # GFLOPS
                utilization = (gflops / theoretical_peak) * 100
                
                if utilization < 10:  # Less than 10% utilization
                    bottlenecks.append(
                        f"Layer '{layer.name}' has low compute efficiency "
                        f"({utilization:.1f}% FLOP utilization)"
                    )
        
        return bottlenecks
    
    def _generate_optimization_suggestions(self, layers: List[LayerProfile],
                                         bottlenecks: List[str],
                                         raw_profile: Dict[str, Any]) -> List[str]:
        """Generate optimization suggestions based on profiling data."""
        suggestions = []
        
        # Analyze layer patterns
        layer_types = defaultdict(list)
        for layer in layers:
            layer_types[layer.layer_type].append(layer)
        
        # Suggestions for specific layer types
        if 'Linear' in layer_types:
            linear_layers = layer_types['Linear']
            total_linear_time = sum(l.forward_time_ms for l in linear_layers)
            total_time = sum(l.forward_time_ms for l in layers)
            
            if total_linear_time > 0.5 * total_time:
                suggestions.append(
                    "Linear layers dominate computation (>50% of time). "
                    "Consider using quantization or pruning for these layers."
                )
                
                # Check for small batch sizes
                if raw_profile.get('batch_size', 1) < 32:
                    suggestions.append(
                        "Small batch size detected. Increasing batch size can "
                        "improve GPU utilization for linear layers."
                    )
        
        # Memory optimization suggestions
        peak_memory = raw_profile.get('peak_memory_mb', 0)
        if peak_memory > 8000:  # > 8GB
            suggestions.append(
                "High memory usage detected. Consider:\n"
                "- Gradient checkpointing to trade compute for memory\n"
                "- Mixed precision training (FP16/BF16)\n"
                "- Model parallelism for very large models"
            )
        
        # Check for memory fragmentation
        total_allocated = sum(l.memory_allocated_mb for l in layers)
        total_freed = sum(l.memory_freed_mb for l in layers)
        if total_allocated > 2 * (total_allocated - total_freed):
            suggestions.append(
                "Potential memory fragmentation detected. "
                "Consider using a custom memory allocator or pre-allocating buffers."
            )
        
        # Device-specific suggestions
        device = raw_profile.get('device', 'cpu')
        if device == 'cpu':
            suggestions.append(
                "Running on CPU. For better performance:\n"
                "- Use GPU if available\n"
                "- Enable multi-threading with appropriate number of threads\n"
                "- Consider using optimized BLAS libraries (MKL, OpenBLAS)"
            )
        elif device.startswith('cuda'):
            # Check for low GPU utilization
            gpu_util = raw_profile.get('gpu_utilization', 0)
            if gpu_util < 80:
                suggestions.append(
                    f"Low GPU utilization ({gpu_util}%). Consider:\n"
                    "- Increasing batch size\n"
                    "- Using mixed precision (AMP)\n"
                    "- Reducing host-device transfers"
                )
        
        # Kernel fusion opportunities
        consecutive_ops = self._find_fusable_operations(layers)
        if consecutive_ops:
            suggestions.append(
                "Found operations that could benefit from kernel fusion:\n" +
                "\n".join(f"- {ops}" for ops in consecutive_ops[:3])
            )
        
        # Data layout suggestions
        if self._has_layout_inefficiency(layers):
            suggestions.append(
                "Detected potential data layout inefficiencies. "
                "Consider using channels-last memory format for better performance."
            )
        
        return suggestions
    
    def _calculate_flop_utilization(self, layers: List[LayerProfile],
                                   raw_profile: Dict[str, Any]) -> float:
        """Calculate FLOP utilization percentage."""
        total_flops = sum(l.flops for l in layers)
        total_time_seconds = sum(l.forward_time_ms for l in layers) / 1000
        
        if total_time_seconds == 0:
            return 0.0
        
        achieved_gflops = (total_flops / 1e9) / total_time_seconds
        
        # Get theoretical peak for device
        device = raw_profile.get('device', 'cpu')
        theoretical_peak = self._get_theoretical_peak_gflops(device)
        
        return (achieved_gflops / theoretical_peak) * 100 if theoretical_peak > 0 else 0
    
    def _calculate_memory_bandwidth_utilization(self, layers: List[LayerProfile],
                                              raw_profile: Dict[str, Any]) -> float:
        """Calculate memory bandwidth utilization."""
        # Estimate data movement
        total_data_mb = 0
        for layer in layers:
            # Input + output + parameters
            if layer.input_shape and layer.output_shape:
                input_size = np.prod(layer.input_shape) * 4 / 1e6  # Assume FP32
                output_size = np.prod(layer.output_shape) * 4 / 1e6
                param_size = layer.parameters * 4 / 1e6
                total_data_mb += input_size + output_size + param_size
        
        total_time_seconds = sum(l.forward_time_ms for l in layers) / 1000
        
        if total_time_seconds == 0:
            return 0.0
        
        achieved_bandwidth_gbs = (total_data_mb / 1000) / total_time_seconds
        
        # Get theoretical peak bandwidth
        device = raw_profile.get('device', 'cpu')
        theoretical_bandwidth = self._get_theoretical_bandwidth_gbs(device)
        
        return (achieved_bandwidth_gbs / theoretical_bandwidth) * 100 if theoretical_bandwidth > 0 else 0
    
    def _get_theoretical_peak_gflops(self, device: str) -> float:
        """Get theoretical peak GFLOPS for device."""
        # These are example values - should be queried from actual hardware
        peaks = {
            'cpu': 500,        # Modern CPU
            'cuda': 10000,     # Modern GPU (e.g., V100)
            'cuda:0': 10000,
            'metal': 5000,     # Apple Silicon
            'android': 50,     # Mobile SoC
            'ios': 100         # Apple A-series
        }
        return peaks.get(device, 100)
    
    def _get_theoretical_bandwidth_gbs(self, device: str) -> float:
        """Get theoretical memory bandwidth for device."""
        bandwidths = {
            'cpu': 50,         # DDR4
            'cuda': 900,       # HBM2
            'cuda:0': 900,
            'metal': 400,      # Unified memory
            'android': 25,     # LPDDR4
            'ios': 50          # LPDDR4X
        }
        return bandwidths.get(device, 50)
    
    def _find_fusable_operations(self, layers: List[LayerProfile]) -> List[str]:
        """Find sequences of operations that could be fused."""
        fusable_patterns = [
            ['Linear', 'ReLU'],
            ['Linear', 'BatchNorm', 'ReLU'],
            ['Conv2d', 'BatchNorm', 'ReLU'],
            ['LayerNorm', 'Linear'],
            ['GELU', 'Linear'],
        ]
        
        fusable_sequences = []
        
        for i in range(len(layers) - 1):
            for pattern in fusable_patterns:
                if i + len(pattern) <= len(layers):
                    sequence_types = [layers[i+j].layer_type for j in range(len(pattern))]
                    if sequence_types == pattern:
                        layer_names = [layers[i+j].name for j in range(len(pattern))]
                        fusable_sequences.append(" -> ".join(layer_names))
        
        return fusable_sequences
    
    def _has_layout_inefficiency(self, layers: List[LayerProfile]) -> bool:
        """Check if model has potential layout inefficiencies."""
        # Look for patterns that suggest layout issues
        for layer in layers:
            if layer.layer_type in ['Conv2d', 'ConvTranspose2d']:
                # Check if using NCHW format (common inefficiency on some hardware)
                if layer.input_shape and len(layer.input_shape) == 4:
                    # Assuming NCHW if channels dimension is small relative to spatial
                    if layer.input_shape[1] < layer.input_shape[2] / 4:
                        return True
        return False
    
    def generate_flame_graph(self, profile_result: ProfileResult) -> Dict[str, Any]:
        """Generate flame graph data from profiling results."""
        flame_data = {
            'name': profile_result.model_name,
            'value': profile_result.total_time_ms,
            'children': []
        }
        
        # Group layers by type
        layer_groups = defaultdict(list)
        for layer in profile_result.layers:
            layer_groups[layer.layer_type].append(layer)
        
        for layer_type, layers in layer_groups.items():
            group_time = sum(l.forward_time_ms for l in layers)
            group_node = {
                'name': layer_type,
                'value': group_time,
                'children': [
                    {
                        'name': layer.name,
                        'value': layer.forward_time_ms,
                        'tooltip': f"Time: {layer.forward_time_ms:.2f}ms\n"
                                  f"Memory: {layer.memory_allocated_mb:.1f}MB\n"
                                  f"FLOPs: {layer.flops/1e6:.1f}M"
                    }
                    for layer in layers
                ]
            }
            flame_data['children'].append(group_node)
        
        return flame_data