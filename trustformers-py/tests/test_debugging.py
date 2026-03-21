"""
Tests for the debugging tools module
"""

import pytest
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Any, Dict, List

# Test debugging module
try:
    from trustformers.debugging import (
        TensorTracker,
        MemoryProfiler,
        PerformanceProfiler,
        TensorVisualizer,
        tensor_tracker,
        memory_profiler,
        performance_profiler,
        tensor_visualizer,
        debug_tensor,
        profile_operation,
        track_tensor_creation,
        get_debug_report,
        export_debug_report,
        start_memory_monitoring,
        stop_memory_monitoring,
        clear_debug_data,
    )
    DEBUGGING_AVAILABLE = True
except ImportError:
    DEBUGGING_AVAILABLE = False

# Also try to import trustformers for tensor operations
try:
    from trustformers import Tensor
    TENSOR_AVAILABLE = True
except ImportError:
    TENSOR_AVAILABLE = False

# Skip all tests if debugging module not available
pytestmark = pytest.mark.skipif(not DEBUGGING_AVAILABLE, reason="Debugging module not available")

class TestTensorTracker:
    """Test TensorTracker functionality"""
    
    def test_tensor_tracker_creation(self):
        """Test basic tensor tracker creation"""
        tracker = TensorTracker(track_creation_stack=True)
        assert tracker.track_creation_stack is True
        assert tracker.active is True
        
    def test_tensor_tracker_default_settings(self):
        """Test tensor tracker with default settings"""
        tracker = TensorTracker()
        assert tracker.track_creation_stack is False
        assert tracker.active is True
    
    @pytest.mark.skipif(not TENSOR_AVAILABLE, reason="Tensor not available")
    def test_tensor_tracking(self):
        """Test basic tensor tracking"""
        tracker = TensorTracker()
        
        # Create mock tensor
        mock_tensor = MagicMock()
        mock_tensor.shape = [2, 3]
        mock_tensor.dtype = "float32"
        
        # Track tensor
        tensor_id = tracker.track_tensor(mock_tensor, name="test_tensor")
        
        assert tensor_id is not None
        assert len(tracker.get_tracked_tensors()) == 1
        
        # Get tensor info
        info = tracker.get_tensor_info(tensor_id)
        assert info is not None
        assert info["name"] == "test_tensor"
        assert info["shape"] == [2, 3]
    
    def test_tensor_memory_tracking(self):
        """Test tensor memory usage tracking"""
        tracker = TensorTracker()
        
        # Mock tensor with memory info
        mock_tensor = MagicMock()
        mock_tensor.shape = [100, 100]
        mock_tensor.dtype = "float32"
        
        with patch.object(tracker, '_calculate_tensor_memory') as mock_calc:
            mock_calc.return_value = 40000  # 100*100*4 bytes
            
            tensor_id = tracker.track_tensor(mock_tensor)
            info = tracker.get_tensor_info(tensor_id)
            
            assert info["memory_bytes"] == 40000
    
    def test_tensor_lifecycle_tracking(self):
        """Test tensor lifecycle tracking"""
        tracker = TensorTracker()
        
        mock_tensor = MagicMock()
        tensor_id = tracker.track_tensor(mock_tensor, name="lifecycle_test")
        
        # Update tensor (simulate operation)
        tracker.update_tensor(tensor_id, operation="add", memory_delta=1000)
        
        # Release tensor
        tracker.release_tensor(tensor_id)
        
        # Check lifecycle
        info = tracker.get_tensor_info(tensor_id)
        assert info is not None
        assert info["status"] == "released"
        assert len(info["operations"]) == 1
        assert info["operations"][0]["operation"] == "add"
    
    def test_tensor_creation_stack_tracking(self):
        """Test creation stack tracking"""
        tracker = TensorTracker(track_creation_stack=True)
        
        mock_tensor = MagicMock()
        tensor_id = tracker.track_tensor(mock_tensor)
        
        info = tracker.get_tensor_info(tensor_id)
        assert "creation_stack" in info
        assert isinstance(info["creation_stack"], list)
        assert len(info["creation_stack"]) > 0
    
    def test_tensor_tracker_statistics(self):
        """Test tensor tracker statistics"""
        tracker = TensorTracker()
        
        # Track several tensors
        for i in range(5):
            mock_tensor = MagicMock()
            mock_tensor.shape = [10, 10]
            tracker.track_tensor(mock_tensor, name=f"tensor_{i}")
        
        stats = tracker.get_statistics()
        assert stats["total_tensors"] == 5
        assert stats["active_tensors"] == 5
        assert "total_memory" in stats
        assert "average_memory" in stats
    
    def test_tensor_tracker_cleanup(self):
        """Test tensor tracker cleanup"""
        tracker = TensorTracker()
        
        # Track some tensors
        tensor_ids = []
        for i in range(3):
            mock_tensor = MagicMock()
            tensor_id = tracker.track_tensor(mock_tensor)
            tensor_ids.append(tensor_id)
        
        assert len(tracker.get_tracked_tensors()) == 3
        
        # Clean up
        tracker.cleanup()
        assert len(tracker.get_tracked_tensors()) == 0

class TestMemoryProfiler:
    """Test MemoryProfiler functionality"""
    
    def test_memory_profiler_creation(self):
        """Test basic memory profiler creation"""
        profiler = MemoryProfiler(sampling_interval=0.1)
        assert profiler.sampling_interval == 0.1
        assert not profiler.is_monitoring
    
    def test_memory_profiler_start_stop(self):
        """Test starting and stopping memory monitoring"""
        profiler = MemoryProfiler(sampling_interval=0.1)
        
        profiler.start_monitoring()
        assert profiler.is_monitoring
        
        time.sleep(0.2)  # Let it collect some samples
        
        profiler.stop_monitoring()
        assert not profiler.is_monitoring
        
        # Should have collected some data
        current_usage = profiler.get_current_usage()
        assert current_usage > 0
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking"""
        profiler = MemoryProfiler()
        
        # Get baseline
        baseline = profiler.get_current_usage()
        assert baseline > 0
        
        # Create some objects
        large_list = [i for i in range(10000)]
        
        # Memory usage should have increased
        new_usage = profiler.get_current_usage()
        assert new_usage >= baseline
        
        # Clean up
        del large_list
    
    def test_memory_leak_detection(self):
        """Test memory leak detection"""
        profiler = MemoryProfiler()
        
        # Set baseline
        profiler.set_baseline()
        
        # Simulate memory leak
        leaked_objects = []
        for i in range(100):
            leaked_objects.append([0] * 1000)  # Keep references
        
        # Check for leaks
        growth = profiler.check_memory_growth()
        assert growth > 0
        
        # Get leak report
        report = profiler.get_leak_report()
        assert "baseline_memory" in report
        assert "current_memory" in report
        assert "growth" in report
        
        # Clean up
        leaked_objects.clear()
    
    def test_memory_profiler_statistics(self):
        """Test memory profiler statistics"""
        profiler = MemoryProfiler(sampling_interval=0.05)
        
        profiler.start_monitoring()
        time.sleep(0.15)  # Collect some samples
        profiler.stop_monitoring()
        
        stats = profiler.get_statistics()
        assert "sample_count" in stats
        assert "max_memory" in stats
        assert "min_memory" in stats
        assert "average_memory" in stats
        assert stats["sample_count"] > 0
    
    def test_memory_profiler_context_manager(self):
        """Test memory profiler as context manager"""
        profiler = MemoryProfiler()
        
        with profiler.monitor():
            # Do some work
            temp_data = [i for i in range(1000)]
            time.sleep(0.05)
        
        # Should have collected monitoring data
        stats = profiler.get_statistics()
        assert stats["sample_count"] > 0
    
    def test_memory_trend_analysis(self):
        """Test memory trend analysis"""
        profiler = MemoryProfiler(sampling_interval=0.05)
        
        profiler.start_monitoring()
        
        # Create increasing memory usage
        data = []
        for i in range(3):
            data.extend([0] * 1000)
            time.sleep(0.06)
        
        profiler.stop_monitoring()
        
        # Analyze trend
        trend = profiler.analyze_trend()
        assert "slope" in trend
        assert "trend_direction" in trend
        # Should detect increasing trend
        assert trend["trend_direction"] in ["increasing", "stable"]

class TestPerformanceProfiler:
    """Test PerformanceProfiler functionality"""
    
    def test_performance_profiler_creation(self):
        """Test basic performance profiler creation"""
        profiler = PerformanceProfiler()
        assert profiler.enabled
        assert len(profiler.get_operations()) == 0
    
    def test_operation_timing(self):
        """Test operation timing"""
        profiler = PerformanceProfiler()
        
        with profiler.time_operation("test_operation"):
            time.sleep(0.01)
        
        operations = profiler.get_operations()
        assert "test_operation" in operations
        
        op_stats = operations["test_operation"]
        assert op_stats["count"] == 1
        assert op_stats["total_time"] > 0.008  # Should be around 0.01s
        assert op_stats["average_time"] > 0.008
    
    def test_multiple_operation_timing(self):
        """Test timing multiple operations"""
        profiler = PerformanceProfiler()
        
        # Time same operation multiple times
        for i in range(3):
            with profiler.time_operation("repeated_op"):
                time.sleep(0.01)
        
        operations = profiler.get_operations()
        op_stats = operations["repeated_op"]
        
        assert op_stats["count"] == 3
        assert op_stats["total_time"] > 0.025
        assert op_stats["average_time"] > 0.008
    
    def test_nested_operation_timing(self):
        """Test nested operation timing"""
        profiler = PerformanceProfiler()
        
        with profiler.time_operation("outer_operation"):
            time.sleep(0.005)
            with profiler.time_operation("inner_operation"):
                time.sleep(0.005)
            time.sleep(0.005)
        
        operations = profiler.get_operations()
        
        assert "outer_operation" in operations
        assert "inner_operation" in operations
        
        # Outer should take longer than inner
        assert operations["outer_operation"]["total_time"] > operations["inner_operation"]["total_time"]
    
    def test_performance_profiler_decorator(self):
        """Test performance profiler as decorator"""
        profiler = PerformanceProfiler()
        
        @profiler.profile
        def decorated_function(x, y):
            time.sleep(0.01)
            return x + y
        
        result = decorated_function(2, 3)
        assert result == 5
        
        operations = profiler.get_operations()
        assert "decorated_function" in operations
        assert operations["decorated_function"]["count"] == 1
    
    def test_memory_usage_during_operations(self):
        """Test memory usage tracking during operations"""
        profiler = PerformanceProfiler(track_memory=True)
        
        with profiler.time_operation("memory_operation"):
            # Create some data
            temp_data = [i for i in range(10000)]
            time.sleep(0.01)
            del temp_data
        
        operations = profiler.get_operations()
        op_stats = operations["memory_operation"]
        
        assert "memory_usage" in op_stats
        assert "peak_memory" in op_stats
    
    def test_performance_statistics(self):
        """Test performance statistics"""
        profiler = PerformanceProfiler()
        
        # Run various operations
        operations = ["op1", "op2", "op3"]
        for op in operations:
            with profiler.time_operation(op):
                time.sleep(0.005)
        
        stats = profiler.get_statistics()
        assert "total_operations" in stats
        assert "total_time" in stats
        assert "operation_count" in stats
        assert stats["operation_count"] == 3
    
    def test_performance_bottleneck_detection(self):
        """Test bottleneck detection"""
        profiler = PerformanceProfiler()
        
        # Create operations with different timing
        with profiler.time_operation("fast_op"):
            time.sleep(0.005)
        
        with profiler.time_operation("slow_op"):
            time.sleep(0.02)
        
        with profiler.time_operation("medium_op"):
            time.sleep(0.01)
        
        bottlenecks = profiler.get_bottlenecks(top_n=2)
        assert len(bottlenecks) == 2
        
        # Should be ordered by time (slowest first)
        assert bottlenecks[0]["operation"] == "slow_op"
        assert bottlenecks[1]["operation"] == "medium_op"

class TestTensorVisualizer:
    """Test TensorVisualizer functionality"""
    
    def test_tensor_visualizer_creation(self):
        """Test basic tensor visualizer creation"""
        visualizer = TensorVisualizer()
        assert visualizer is not None
    
    @pytest.mark.skipif(not TENSOR_AVAILABLE, reason="Tensor not available")
    def test_tensor_visualization_setup(self):
        """Test tensor visualization setup"""
        visualizer = TensorVisualizer()
        
        # Mock tensor
        mock_tensor = MagicMock()
        mock_tensor.shape = [3, 3]
        mock_tensor.numpy.return_value = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        
        # Should not error when creating visualization
        viz_info = visualizer.create_visualization(mock_tensor, "heatmap")
        assert viz_info is not None
    
    def test_tensor_statistics_computation(self):
        """Test tensor statistics computation"""
        visualizer = TensorVisualizer()
        
        # Mock tensor with statistics
        mock_tensor = MagicMock()
        mock_tensor.shape = [2, 2]
        
        with patch.object(visualizer, '_compute_statistics') as mock_stats:
            mock_stats.return_value = {
                "mean": 2.5,
                "std": 1.29,
                "min": 1.0,
                "max": 4.0,
                "zeros": 0,
                "non_zeros": 4
            }
            
            stats = visualizer.compute_statistics(mock_tensor)
            assert stats["mean"] == 2.5
            assert stats["std"] == 1.29
            assert stats["min"] == 1.0
            assert stats["max"] == 4.0
    
    def test_visualization_options(self):
        """Test different visualization options"""
        visualizer = TensorVisualizer()
        
        supported_types = visualizer.get_supported_visualizations()
        assert isinstance(supported_types, list)
        assert len(supported_types) > 0
        
        # Common visualization types should be supported
        common_types = ["histogram", "heatmap", "line_plot"]
        for viz_type in common_types:
            if viz_type in supported_types:
                assert visualizer.supports_visualization(viz_type)

class TestGlobalInstances:
    """Test global debugging instances"""
    
    def test_global_tensor_tracker(self):
        """Test global tensor tracker instance"""
        tracker = tensor_tracker
        assert isinstance(tracker, TensorTracker)
        
        # Should be same instance on repeated access
        tracker2 = tensor_tracker
        assert tracker is tracker2
    
    def test_global_memory_profiler(self):
        """Test global memory profiler instance"""
        profiler = memory_profiler
        assert isinstance(profiler, MemoryProfiler)
        
        # Should be same instance on repeated access
        profiler2 = memory_profiler
        assert profiler is profiler2
    
    def test_global_performance_profiler(self):
        """Test global performance profiler instance"""
        profiler = performance_profiler
        assert isinstance(profiler, PerformanceProfiler)
        
        # Should be same instance on repeated access
        profiler2 = performance_profiler
        assert profiler is profiler2
    
    def test_global_tensor_visualizer(self):
        """Test global tensor visualizer instance"""
        visualizer = tensor_visualizer
        assert isinstance(visualizer, TensorVisualizer)
        
        # Should be same instance on repeated access
        visualizer2 = tensor_visualizer
        assert visualizer is visualizer2

class TestUtilityFunctions:
    """Test debugging utility functions"""
    
    @pytest.mark.skipif(not TENSOR_AVAILABLE, reason="Tensor not available")
    def test_debug_tensor_function(self):
        """Test debug_tensor utility function"""
        mock_tensor = MagicMock()
        mock_tensor.shape = [2, 3]
        mock_tensor.dtype = "float32"
        
        # Should not error
        debug_info = debug_tensor(mock_tensor, name="test_debug")
        assert debug_info is not None
    
    def test_profile_operation_function(self):
        """Test profile_operation utility function"""
        @profile_operation
        def test_function():
            time.sleep(0.01)
            return "result"
        
        result = test_function()
        assert result == "result"
        
        # Check that profiling occurred
        stats = performance_profiler.get_operations()
        assert "test_function" in stats
    
    @pytest.mark.skipif(not TENSOR_AVAILABLE, reason="Tensor not available")
    def test_track_tensor_creation_function(self):
        """Test track_tensor_creation utility function"""
        @track_tensor_creation
        def create_tensor():
            mock_tensor = MagicMock()
            mock_tensor.shape = [2, 2]
            return mock_tensor
        
        tensor = create_tensor()
        assert tensor is not None
        
        # Check that tracking occurred
        tracked = tensor_tracker.get_tracked_tensors()
        assert len(tracked) > 0
    
    def test_memory_monitoring_functions(self):
        """Test start/stop memory monitoring functions"""
        # Start monitoring
        start_memory_monitoring(interval=0.1)
        
        # Should be monitoring now
        assert memory_profiler.is_monitoring
        
        time.sleep(0.15)
        
        # Stop monitoring
        stop_memory_monitoring()
        
        # Should not be monitoring
        assert not memory_profiler.is_monitoring
    
    def test_get_debug_report_function(self):
        """Test get_debug_report utility function"""
        # Generate some debug data
        with performance_profiler.time_operation("test_op"):
            time.sleep(0.01)
        
        report = get_debug_report()
        assert isinstance(report, dict)
        assert "performance" in report
        assert "memory" in report
        assert "tensors" in report
    
    def test_export_debug_report_function(self):
        """Test export_debug_report utility function"""
        # Generate some debug data
        with performance_profiler.time_operation("export_test"):
            time.sleep(0.005)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            # Export report
            success = export_debug_report(filepath)
            assert success
            
            # Check file was created and contains valid JSON
            with open(filepath, 'r') as f:
                data = json.load(f)
                assert isinstance(data, dict)
                assert "performance" in data
        finally:
            Path(filepath).unlink(missing_ok=True)
    
    def test_clear_debug_data_function(self):
        """Test clear_debug_data utility function"""
        # Generate some debug data
        with performance_profiler.time_operation("clear_test"):
            time.sleep(0.005)
        
        # Should have data
        assert len(performance_profiler.get_operations()) > 0
        
        # Clear all debug data
        clear_debug_data()
        
        # Should be cleared
        assert len(performance_profiler.get_operations()) == 0

class TestDebuggingIntegration:
    """Integration tests for debugging tools"""
    
    def test_combined_debugging_workflow(self):
        """Test using multiple debugging tools together"""
        # Start comprehensive debugging
        start_memory_monitoring(interval=0.05)
        
        @profile_operation
        @track_tensor_creation
        def complex_operation():
            # Simulate complex operation
            mock_tensor1 = MagicMock()
            mock_tensor1.shape = [10, 10]
            
            mock_tensor2 = MagicMock() 
            mock_tensor2.shape = [5, 5]
            
            time.sleep(0.02)
            return mock_tensor1, mock_tensor2
        
        # Run operation
        result = complex_operation()
        
        time.sleep(0.1)  # Let monitoring collect data
        stop_memory_monitoring()
        
        # Get comprehensive report
        report = get_debug_report()
        
        # Should have data from all debugging tools
        assert "performance" in report
        assert "memory" in report
        assert "tensors" in report
        
        # Performance data
        assert "complex_operation" in report["performance"]["operations"]
        
        # Memory data
        assert "current_usage" in report["memory"]
        
        # Tensor data
        assert "total_tensors" in report["tensors"]
    
    def test_debugging_overhead(self):
        """Test debugging overhead is reasonable"""
        # Time without debugging
        start_time = time.time()
        for i in range(100):
            x = i * 2
        no_debug_time = time.time() - start_time
        
        # Time with debugging
        start_time = time.time()
        for i in range(100):
            with performance_profiler.time_operation("overhead_test"):
                x = i * 2
        debug_time = time.time() - start_time
        
        # Overhead should be reasonable (less than 50x for microbenchmarks)
        # Note: For extremely fast operations, context manager overhead is proportionally high
        # This is acceptable for debugging/development tools
        overhead_ratio = debug_time / no_debug_time if no_debug_time > 0 else 1
        assert overhead_ratio < 50.0
    
    def test_debugging_error_handling(self):
        """Test debugging tools handle errors gracefully"""
        # Test with operation that raises exception
        try:
            with performance_profiler.time_operation("error_operation"):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Should still record the operation
        operations = performance_profiler.get_operations()
        assert "error_operation" in operations
        
        # Should record the error
        op_stats = operations["error_operation"]
        assert op_stats["count"] == 1

if __name__ == "__main__":
    pytest.main([__file__])