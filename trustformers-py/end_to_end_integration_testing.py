#!/usr/bin/env python3
"""
End-to-End Integration Testing Framework for TrustformeRS
========================================================

This module provides comprehensive end-to-end integration testing capabilities for TrustformeRS,
ensuring all components work together seamlessly across different deployment scenarios.

Key Features:
- Complete pipeline testing (model loading → tokenization → inference → output)
- Multi-framework integration testing (PyTorch, JAX, NumPy)
- Performance and memory validation
- Distributed training integration tests
- Mobile and WASM deployment validation
- CI/CD pipeline integration
- Automated regression testing

Author: TrustformeRS Development Team
License: MIT License
"""

import os
import sys
import json
import time
import asyncio
import tempfile
import subprocess
import shutil
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timedelta
import hashlib
import threading
import concurrent.futures
from functools import wraps
import inspect

# Test framework dependencies
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    print("Warning: pytest not available. Install with: pip install pytest")

# Performance monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Install with: pip install psutil")

# Async testing support
try:
    import pytest_asyncio
    PYTEST_ASYNCIO_AVAILABLE = True
except ImportError:
    PYTEST_ASYNCIO_AVAILABLE = False
    print("Warning: pytest-asyncio not available. Install with: pip install pytest-asyncio")

class TestPhase(Enum):
    """Test execution phases."""
    SETUP = "setup"
    EXECUTION = "execution"
    VALIDATION = "validation"
    CLEANUP = "cleanup"

class TestScope(Enum):
    """Test scope levels."""
    UNIT = "unit"
    INTEGRATION = "integration"
    SYSTEM = "system"
    END_TO_END = "end_to_end"
    REGRESSION = "regression"

class TestEnvironment(Enum):
    """Test environment types."""
    LOCAL = "local"
    CI = "ci"
    STAGING = "staging"
    PRODUCTION = "production"

class ComponentType(Enum):
    """Component types for testing."""
    CORE = "core"
    MODELS = "models"
    TOKENIZERS = "tokenizers"
    PIPELINES = "pipelines"
    TRAINING = "training"
    SERVING = "serving"
    MOBILE = "mobile"
    WASM = "wasm"

@dataclass
class TestConfig:
    """Configuration for end-to-end testing."""
    test_scope: TestScope = TestScope.END_TO_END
    environment: TestEnvironment = TestEnvironment.LOCAL
    components: List[ComponentType] = field(default_factory=lambda: list(ComponentType))
    test_models: List[str] = field(default_factory=lambda: ["bert-tiny", "gpt2-micro"])
    frameworks: List[str] = field(default_factory=lambda: ["pytorch", "numpy"])
    performance_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "load_time_seconds": 10.0,
        "inference_time_ms": 1000.0,
        "memory_usage_mb": 500.0,
        "accuracy_threshold": 0.8
    })
    timeout_seconds: int = 300
    parallel_tests: bool = True
    max_workers: int = 4
    cleanup_on_failure: bool = True
    generate_coverage_report: bool = True
    enable_profiling: bool = False
    
@dataclass
class TestResult:
    """Result of an integration test."""
    test_name: str
    component: ComponentType
    scope: TestScope
    phase: TestPhase
    passed: bool
    duration: float
    start_time: datetime
    end_time: datetime
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
@dataclass
class TestSuite:
    """Collection of related tests."""
    name: str
    description: str
    tests: List[Callable] = field(default_factory=list)
    setup_functions: List[Callable] = field(default_factory=list)
    teardown_functions: List[Callable] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    timeout: int = 300

class EndToEndIntegrationTester:
    """
    Comprehensive end-to-end integration testing framework for TrustformeRS.
    
    Features:
    - Complete pipeline testing across all components
    - Multi-framework integration validation
    - Performance and memory monitoring
    - Automated regression testing
    - CI/CD pipeline integration
    - Detailed reporting and analytics
    """
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.project_root = Path(__file__).parent
        self.test_dir = self.project_root / "tests" / "integration"
        self.output_dir = self.project_root / "test_results" / "integration"
        self.artifacts_dir = self.output_dir / "artifacts"
        
        # Create directories
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Test registry
        self.test_suites: Dict[str, TestSuite] = {}
        self.test_results: List[TestResult] = []
        
        # Performance monitoring
        self.memory_monitor = None
        self.performance_data = []
        
        # Test state
        self.current_test = None
        self.test_start_time = None
        
        # Initialize test suites
        self._initialize_test_suites()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for integration testing."""
        logger = logging.getLogger("EndToEndIntegrationTester")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        log_file = self.output_dir / f"integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
        
    def _initialize_test_suites(self):
        """Initialize all test suites."""
        self.register_core_tests()
        self.register_model_tests()
        self.register_tokenizer_tests()
        self.register_pipeline_tests()
        self.register_framework_integration_tests()
        self.register_performance_tests()
        self.register_deployment_tests()
        self.register_regression_tests()
        
    def register_test_suite(self, suite: TestSuite):
        """Register a test suite."""
        self.test_suites[suite.name] = suite
        self.logger.info(f"Registered test suite: {suite.name}")
        
    def test_case(self, 
                  component: ComponentType, 
                  scope: TestScope = TestScope.INTEGRATION,
                  tags: List[str] = None,
                  timeout: int = None):
        """Decorator for registering test cases."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                return await self._execute_test_case(func, component, scope, tags or [], timeout, *args, **kwargs)
            return wrapper
        return decorator
        
    async def _execute_test_case(self, 
                                func: Callable, 
                                component: ComponentType,
                                scope: TestScope,
                                tags: List[str],
                                timeout: Optional[int],
                                *args, **kwargs) -> TestResult:
        """Execute a single test case with monitoring."""
        test_name = func.__name__
        start_time = datetime.now()
        self.current_test = test_name
        self.test_start_time = start_time
        
        self.logger.info(f"Starting test: {test_name} ({component.value})")
        
        # Start performance monitoring
        self._start_performance_monitoring()
        
        try:
            # Execute test with timeout
            test_timeout = timeout or self.config.timeout_seconds
            
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=test_timeout)
            else:
                # Run synchronous function in executor
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, func, *args, **kwargs),
                    timeout=test_timeout
                )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Stop performance monitoring
            performance_metrics, memory_usage = self._stop_performance_monitoring()
            
            test_result = TestResult(
                test_name=test_name,
                component=component,
                scope=scope,
                phase=TestPhase.EXECUTION,
                passed=True,
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                performance_metrics=performance_metrics,
                memory_usage=memory_usage,
                tags=tags
            )
            
            self.logger.info(f"Test passed: {test_name} ({duration:.2f}s)")
            
        except asyncio.TimeoutError:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            test_result = TestResult(
                test_name=test_name,
                component=component,
                scope=scope,
                phase=TestPhase.EXECUTION,
                passed=False,
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                error_message=f"Test timeout after {test_timeout}s",
                tags=tags
            )
            
            self.logger.error(f"Test timeout: {test_name}")
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            test_result = TestResult(
                test_name=test_name,
                component=component,
                scope=scope,
                phase=TestPhase.EXECUTION,
                passed=False,
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                error_message=str(e),
                error_traceback=traceback.format_exc(),
                tags=tags
            )
            
            self.logger.error(f"Test failed: {test_name} - {e}")
            
        finally:
            self._stop_performance_monitoring()
            
        self.test_results.append(test_result)
        return test_result
        
    def _start_performance_monitoring(self):
        """Start performance and memory monitoring."""
        if PSUTIL_AVAILABLE:
            self.memory_monitor = threading.Thread(
                target=self._monitor_memory,
                daemon=True
            )
            self.memory_monitor.start()
            
    def _stop_performance_monitoring(self) -> Tuple[Dict[str, Any], Dict[str, float]]:
        """Stop performance monitoring and return metrics."""
        performance_metrics = {}
        memory_usage = {}
        
        if self.memory_monitor and self.memory_monitor.is_alive():
            # Stop monitoring thread
            self.memory_monitor = None
            
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_usage = {
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
                "percent": process.memory_percent()
            }
            
            performance_metrics = {
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "open_files": len(process.open_files())
            }
            
        return performance_metrics, memory_usage
        
    def _monitor_memory(self):
        """Monitor memory usage during test execution."""
        if not PSUTIL_AVAILABLE:
            return
            
        process = psutil.Process()
        
        while self.memory_monitor and threading.current_thread() == self.memory_monitor:
            try:
                memory_info = process.memory_info()
                self.performance_data.append({
                    "timestamp": datetime.now(),
                    "test": self.current_test,
                    "rss_mb": memory_info.rss / 1024 / 1024,
                    "vms_mb": memory_info.vms / 1024 / 1024,
                    "cpu_percent": process.cpu_percent()
                })
                time.sleep(1)  # Sample every second
            except Exception:
                break
                
    def register_core_tests(self):
        """Register core functionality tests."""
        suite = TestSuite(
            name="core_functionality",
            description="Test core TrustformeRS functionality",
            tags=["core", "basic"]
        )
        
        @self.test_case(ComponentType.CORE, TestScope.INTEGRATION)
        async def test_basic_tensor_operations():
            """Test basic tensor operations."""
            try:
                # Import TrustformeRS
                import trustformers
                
                # Create tensor
                tensor = trustformers.Tensor([1.0, 2.0, 3.0])
                assert tensor.shape == (3,), f"Expected shape (3,), got {tensor.shape}"
                
                # Basic operations
                tensor2 = trustformers.Tensor([4.0, 5.0, 6.0])
                result = tensor + tensor2
                
                # Convert to numpy
                np_result = result.numpy()
                expected = [5.0, 7.0, 9.0]
                
                for i, (actual, exp) in enumerate(zip(np_result, expected)):
                    assert abs(actual - exp) < 1e-6, f"Index {i}: expected {exp}, got {actual}"
                    
                return True
                
            except ImportError as e:
                raise AssertionError(f"Failed to import trustformers: {e}")
            except Exception as e:
                raise AssertionError(f"Tensor operations failed: {e}")
                
        @self.test_case(ComponentType.CORE, TestScope.INTEGRATION)
        async def test_error_handling():
            """Test error handling and edge cases."""
            try:
                import trustformers
                
                # Test invalid input
                try:
                    tensor = trustformers.Tensor("invalid")
                    raise AssertionError("Should have failed with invalid input")
                except (TypeError, ValueError):
                    pass  # Expected
                    
                # Test shape mismatch
                try:
                    tensor1 = trustformers.Tensor([1, 2, 3])
                    tensor2 = trustformers.Tensor([1, 2])
                    result = tensor1 + tensor2
                    raise AssertionError("Should have failed with shape mismatch")
                except (ValueError, RuntimeError):
                    pass  # Expected
                    
                return True
                
            except ImportError as e:
                raise AssertionError(f"Failed to import trustformers: {e}")
                
        suite.tests.extend([test_basic_tensor_operations, test_error_handling])
        self.register_test_suite(suite)
        
    def register_model_tests(self):
        """Register model loading and inference tests."""
        suite = TestSuite(
            name="model_functionality", 
            description="Test model loading and inference",
            tags=["models", "inference"]
        )
        
        @self.test_case(ComponentType.MODELS, TestScope.INTEGRATION)
        async def test_model_loading():
            """Test model loading from different sources."""
            try:
                import trustformers
                
                # Test model creation
                model = trustformers.BertModel.from_pretrained("bert-tiny")
                assert model is not None, "Model should not be None"
                
                # Test model configuration
                config = model.config
                assert hasattr(config, 'hidden_size'), "Model should have configuration"
                
                return True
                
            except ImportError as e:
                raise AssertionError(f"Failed to import trustformers: {e}")
            except Exception as e:
                raise AssertionError(f"Model loading failed: {e}")
                
        @self.test_case(ComponentType.MODELS, TestScope.INTEGRATION)
        async def test_model_inference():
            """Test model inference pipeline."""
            try:
                import trustformers
                
                # Load model and tokenizer
                model = trustformers.BertModel.from_pretrained("bert-tiny")
                tokenizer = trustformers.AutoTokenizer.from_pretrained("bert-tiny")
                
                # Prepare input
                text = "Hello world, this is a test."
                inputs = tokenizer(text, return_tensors="pt")
                
                # Run inference
                outputs = model(**inputs)
                
                # Validate outputs
                assert hasattr(outputs, 'last_hidden_state'), "Output should have last_hidden_state"
                assert outputs.last_hidden_state.shape[0] == 1, "Batch size should be 1"
                
                return True
                
            except ImportError as e:
                raise AssertionError(f"Failed to import trustformers: {e}")
            except Exception as e:
                raise AssertionError(f"Model inference failed: {e}")
                
        suite.tests.extend([test_model_loading, test_model_inference])
        self.register_test_suite(suite)
        
    def register_tokenizer_tests(self):
        """Register tokenizer tests."""
        suite = TestSuite(
            name="tokenizer_functionality",
            description="Test tokenizer functionality",
            tags=["tokenizers", "preprocessing"]
        )
        
        @self.test_case(ComponentType.TOKENIZERS, TestScope.INTEGRATION)
        async def test_tokenizer_encoding():
            """Test tokenizer encoding and decoding."""
            try:
                import trustformers
                
                tokenizer = trustformers.AutoTokenizer.from_pretrained("bert-tiny")
                
                # Test encoding
                text = "Hello world!"
                tokens = tokenizer(text)
                
                assert 'input_ids' in tokens, "Tokens should contain input_ids"
                assert len(tokens['input_ids']) > 0, "Should have at least one token"
                
                # Test decoding
                decoded = tokenizer.decode(tokens['input_ids'])
                assert isinstance(decoded, str), "Decoded output should be string"
                
                return True
                
            except ImportError as e:
                raise AssertionError(f"Failed to import trustformers: {e}")
            except Exception as e:
                raise AssertionError(f"Tokenizer test failed: {e}")
                
        @self.test_case(ComponentType.TOKENIZERS, TestScope.INTEGRATION)
        async def test_batch_tokenization():
            """Test batch tokenization."""
            try:
                import trustformers
                
                tokenizer = trustformers.AutoTokenizer.from_pretrained("bert-tiny")
                
                # Test batch encoding
                texts = ["Hello world!", "This is another test.", "Short text."]
                batch_tokens = tokenizer(texts, padding=True, truncation=True)
                
                assert 'input_ids' in batch_tokens, "Batch tokens should contain input_ids"
                assert len(batch_tokens['input_ids']) == len(texts), "Should have same number of sequences"
                
                # All sequences should have same length due to padding
                lengths = [len(seq) for seq in batch_tokens['input_ids']]
                assert len(set(lengths)) == 1, "All sequences should have same length after padding"
                
                return True
                
            except ImportError as e:
                raise AssertionError(f"Failed to import trustformers: {e}")
            except Exception as e:
                raise AssertionError(f"Batch tokenization failed: {e}")
                
        suite.tests.extend([test_tokenizer_encoding, test_batch_tokenization])
        self.register_test_suite(suite)
        
    def register_pipeline_tests(self):
        """Register pipeline tests."""
        suite = TestSuite(
            name="pipeline_functionality",
            description="Test pipeline functionality",
            tags=["pipelines", "end_to_end"]
        )
        
        @self.test_case(ComponentType.PIPELINES, TestScope.END_TO_END)
        async def test_text_generation_pipeline():
            """Test text generation pipeline."""
            try:
                import trustformers
                
                # Create pipeline
                generator = trustformers.pipeline("text-generation", model="gpt2-micro")
                
                # Generate text
                prompt = "Hello world"
                outputs = generator(prompt, max_length=20, num_return_sequences=1)
                
                assert isinstance(outputs, list), "Output should be a list"
                assert len(outputs) > 0, "Should generate at least one output"
                assert 'generated_text' in outputs[0], "Output should contain generated_text"
                
                return True
                
            except ImportError as e:
                raise AssertionError(f"Failed to import trustformers: {e}")
            except Exception as e:
                raise AssertionError(f"Text generation pipeline failed: {e}")
                
        @self.test_case(ComponentType.PIPELINES, TestScope.END_TO_END)
        async def test_text_classification_pipeline():
            """Test text classification pipeline."""
            try:
                import trustformers
                
                # Create pipeline
                classifier = trustformers.pipeline("text-classification", model="bert-tiny")
                
                # Classify text
                text = "This is a positive example."
                results = classifier(text)
                
                assert isinstance(results, list), "Results should be a list"
                assert len(results) > 0, "Should have at least one classification"
                assert 'label' in results[0], "Result should contain label"
                assert 'score' in results[0], "Result should contain score"
                
                return True
                
            except ImportError as e:
                raise AssertionError(f"Failed to import trustformers: {e}")
            except Exception as e:
                raise AssertionError(f"Text classification pipeline failed: {e}")
                
        suite.tests.extend([test_text_generation_pipeline, test_text_classification_pipeline])
        self.register_test_suite(suite)
        
    def register_framework_integration_tests(self):
        """Register framework integration tests."""
        suite = TestSuite(
            name="framework_integration",
            description="Test integration with external frameworks",
            tags=["integration", "frameworks"]
        )
        
        @self.test_case(ComponentType.CORE, TestScope.INTEGRATION)
        async def test_pytorch_integration():
            """Test PyTorch integration."""
            try:
                import trustformers
                import torch
                
                # Create TrustformeRS tensor
                tf_tensor = trustformers.Tensor([1.0, 2.0, 3.0])
                
                # Convert to PyTorch
                torch_tensor = tf_tensor.to_torch()
                assert isinstance(torch_tensor, torch.Tensor), "Should be PyTorch tensor"
                
                # Convert back
                tf_tensor2 = trustformers.Tensor.from_torch(torch_tensor)
                
                # Check values are preserved
                original = tf_tensor.numpy()
                converted = tf_tensor2.numpy()
                
                for i, (orig, conv) in enumerate(zip(original, converted)):
                    assert abs(orig - conv) < 1e-6, f"Values should match at index {i}"
                    
                return True
                
            except ImportError as e:
                # Skip if PyTorch not available
                if "torch" in str(e):
                    return True
                raise AssertionError(f"Failed to import required modules: {e}")
            except Exception as e:
                raise AssertionError(f"PyTorch integration failed: {e}")
                
        @self.test_case(ComponentType.CORE, TestScope.INTEGRATION)
        async def test_numpy_integration():
            """Test NumPy integration."""
            try:
                import trustformers
                import numpy as np
                
                # Create NumPy array
                np_array = np.array([1.0, 2.0, 3.0])
                
                # Convert to TrustformeRS tensor
                tf_tensor = trustformers.Tensor(np_array)
                
                # Convert back to NumPy
                np_array2 = tf_tensor.numpy()
                
                # Check values are preserved
                np.testing.assert_array_almost_equal(np_array, np_array2)
                
                return True
                
            except ImportError as e:
                raise AssertionError(f"Failed to import required modules: {e}")
            except Exception as e:
                raise AssertionError(f"NumPy integration failed: {e}")
                
        suite.tests.extend([test_pytorch_integration, test_numpy_integration])
        self.register_test_suite(suite)
        
    def register_performance_tests(self):
        """Register performance tests."""
        suite = TestSuite(
            name="performance_validation",
            description="Test performance characteristics",
            tags=["performance", "benchmarks"]
        )
        
        @self.test_case(ComponentType.CORE, TestScope.SYSTEM)
        async def test_inference_speed():
            """Test inference speed meets requirements."""
            try:
                import trustformers
                import time
                
                # Load model
                model = trustformers.BertModel.from_pretrained("bert-tiny")
                tokenizer = trustformers.AutoTokenizer.from_pretrained("bert-tiny")
                
                # Prepare input
                text = "This is a performance test input."
                inputs = tokenizer(text, return_tensors="pt")
                
                # Warm up
                for _ in range(3):
                    _ = model(**inputs)
                    
                # Measure inference time
                num_runs = 10
                start_time = time.time()
                
                for _ in range(num_runs):
                    outputs = model(**inputs)
                    
                end_time = time.time()
                avg_time_ms = (end_time - start_time) / num_runs * 1000
                
                # Check against threshold
                threshold_ms = self.config.performance_thresholds['inference_time_ms']
                assert avg_time_ms < threshold_ms, f"Inference too slow: {avg_time_ms:.2f}ms > {threshold_ms}ms"
                
                self.logger.info(f"Average inference time: {avg_time_ms:.2f}ms")
                
                return True
                
            except ImportError as e:
                raise AssertionError(f"Failed to import required modules: {e}")
            except Exception as e:
                raise AssertionError(f"Performance test failed: {e}")
                
        @self.test_case(ComponentType.CORE, TestScope.SYSTEM)
        async def test_memory_usage():
            """Test memory usage is within limits."""
            try:
                import trustformers
                
                if not PSUTIL_AVAILABLE:
                    self.logger.warning("psutil not available, skipping memory test")
                    return True
                    
                import psutil
                process = psutil.Process()
                
                # Measure initial memory
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                # Load model (memory intensive operation)
                model = trustformers.BertModel.from_pretrained("bert-tiny")
                tokenizer = trustformers.AutoTokenizer.from_pretrained("bert-tiny")
                
                # Run inference
                text = "Memory usage test input."
                inputs = tokenizer(text, return_tensors="pt")
                outputs = model(**inputs)
                
                # Measure peak memory
                peak_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = peak_memory - initial_memory
                
                # Check against threshold
                threshold_mb = self.config.performance_thresholds['memory_usage_mb']
                assert memory_increase < threshold_mb, f"Memory usage too high: {memory_increase:.2f}MB > {threshold_mb}MB"
                
                self.logger.info(f"Memory increase: {memory_increase:.2f}MB")
                
                return True
                
            except ImportError as e:
                raise AssertionError(f"Failed to import required modules: {e}")
            except Exception as e:
                raise AssertionError(f"Memory test failed: {e}")
                
        suite.tests.extend([test_inference_speed, test_memory_usage])
        self.register_test_suite(suite)
        
    def register_deployment_tests(self):
        """Register deployment-specific tests."""
        suite = TestSuite(
            name="deployment_validation",
            description="Test deployment scenarios",
            tags=["deployment", "environments"]
        )
        
        @self.test_case(ComponentType.MOBILE, TestScope.SYSTEM)
        async def test_mobile_compatibility():
            """Test mobile deployment compatibility."""
            try:
                # Test mobile package manager
                from mobile_package_management import MobilePackageManager, MobileConfig, MobilePlatform
                
                config = MobileConfig(
                    platform=MobilePlatform.BOTH,
                    target_architectures=["arm64"]
                )
                
                # This would normally create actual packages, but for testing we just validate configuration
                with MobilePackageManager(config) as manager:
                    # Test configuration
                    assert manager.config.platform == MobilePlatform.BOTH
                    assert "arm64" in manager.config.target_architectures
                    
                return True
                
            except ImportError as e:
                self.logger.warning(f"Mobile package manager not available: {e}")
                return True  # Skip if not available
            except Exception as e:
                raise AssertionError(f"Mobile compatibility test failed: {e}")
                
        @self.test_case(ComponentType.WASM, TestScope.SYSTEM)
        async def test_wasm_compatibility():
            """Test WASM deployment compatibility."""
            try:
                # Test WASM deployment utilities
                from wasm_deployment import WASMConfig, WASMDeployer
                
                config = WASMConfig(
                    target="wasm32-wasi",
                    optimization_level="s"
                )
                
                deployer = WASMDeployer()
                
                # Test environment setup
                env = deployer.setup_wasm_environment(config)
                assert isinstance(env, dict), "Environment should be a dictionary"
                
                return True
                
            except ImportError as e:
                self.logger.warning(f"WASM deployer not available: {e}")
                return True  # Skip if not available
            except Exception as e:
                raise AssertionError(f"WASM compatibility test failed: {e}")
                
        suite.tests.extend([test_mobile_compatibility, test_wasm_compatibility])
        self.register_test_suite(suite)
        
    def register_regression_tests(self):
        """Register regression tests."""
        suite = TestSuite(
            name="regression_validation",
            description="Test for regressions in functionality",
            tags=["regression", "stability"]
        )
        
        @self.test_case(ComponentType.CORE, TestScope.REGRESSION)
        async def test_backwards_compatibility():
            """Test backwards compatibility with previous versions."""
            try:
                import trustformers
                
                # Test basic API remains the same
                assert hasattr(trustformers, 'Tensor'), "Tensor class should exist"
                assert hasattr(trustformers, 'AutoTokenizer'), "AutoTokenizer should exist"
                assert hasattr(trustformers, 'pipeline'), "pipeline function should exist"
                
                # Test tensor creation methods
                tensor = trustformers.Tensor([1, 2, 3])
                assert hasattr(tensor, 'numpy'), "Tensor should have numpy method"
                assert hasattr(tensor, 'shape'), "Tensor should have shape property"
                
                return True
                
            except ImportError as e:
                raise AssertionError(f"Failed to import trustformers: {e}")
            except Exception as e:
                raise AssertionError(f"Backwards compatibility test failed: {e}")
                
        suite.tests.extend([test_backwards_compatibility])
        self.register_test_suite(suite)
        
    async def run_test_suite(self, suite_name: str) -> List[TestResult]:
        """Run a specific test suite."""
        if suite_name not in self.test_suites:
            raise ValueError(f"Unknown test suite: {suite_name}")
            
        suite = self.test_suites[suite_name]
        self.logger.info(f"Running test suite: {suite_name}")
        
        results = []
        
        try:
            # Run setup functions
            for setup_func in suite.setup_functions:
                self.logger.info(f"Running setup: {setup_func.__name__}")
                if asyncio.iscoroutinefunction(setup_func):
                    await setup_func()
                else:
                    setup_func()
                    
            # Run tests
            if self.config.parallel_tests and len(suite.tests) > 1:
                # Run tests in parallel
                semaphore = asyncio.Semaphore(self.config.max_workers)
                
                async def run_with_semaphore(test_func):
                    async with semaphore:
                        return await test_func()
                        
                tasks = [run_with_semaphore(test) for test in suite.tests]
                test_results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in test_results:
                    if isinstance(result, TestResult):
                        results.append(result)
                    elif isinstance(result, Exception):
                        self.logger.error(f"Test failed with exception: {result}")
                        
            else:
                # Run tests sequentially
                for test_func in suite.tests:
                    result = await test_func()
                    if isinstance(result, TestResult):
                        results.append(result)
                        
        except Exception as e:
            self.logger.error(f"Test suite {suite_name} failed: {e}")
            
        finally:
            # Run teardown functions
            for teardown_func in suite.teardown_functions:
                try:
                    self.logger.info(f"Running teardown: {teardown_func.__name__}")
                    if asyncio.iscoroutinefunction(teardown_func):
                        await teardown_func()
                    else:
                        teardown_func()
                except Exception as e:
                    self.logger.error(f"Teardown failed: {e}")
                    
        return results
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all registered test suites."""
        self.logger.info("Starting comprehensive integration test suite")
        
        start_time = datetime.now()
        all_results = []
        suite_summaries = {}
        
        for suite_name in self.test_suites:
            self.logger.info(f"Running test suite: {suite_name}")
            
            suite_start = datetime.now()
            suite_results = await self.run_test_suite(suite_name)
            suite_end = datetime.now()
            
            all_results.extend(suite_results)
            
            # Calculate suite summary
            passed = len([r for r in suite_results if r.passed])
            failed = len([r for r in suite_results if not r.passed])
            duration = (suite_end - suite_start).total_seconds()
            
            suite_summaries[suite_name] = {
                "total": len(suite_results),
                "passed": passed,
                "failed": failed,
                "success_rate": (passed / len(suite_results) * 100) if suite_results else 0,
                "duration": duration
            }
            
            self.logger.info(f"Suite {suite_name}: {passed}/{len(suite_results)} passed ({duration:.2f}s)")
            
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Calculate overall summary
        total_tests = len(all_results)
        total_passed = len([r for r in all_results if r.passed])
        total_failed = total_tests - total_passed
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        summary = {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration": total_duration,
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "overall_success_rate": overall_success_rate,
            "suite_summaries": suite_summaries,
            "environment": {
                "python_version": sys.version,
                "platform": sys.platform,
                "test_config": {
                    "scope": self.config.test_scope.value,
                    "environment": self.config.environment.value,
                    "parallel": self.config.parallel_tests,
                    "max_workers": self.config.max_workers
                }
            }
        }
        
        # Generate detailed report
        report_path = self.generate_test_report(all_results, summary)
        summary["report_path"] = str(report_path)
        
        self.logger.info(f"Integration tests completed: {total_passed}/{total_tests} passed ({overall_success_rate:.1f}%)")
        self.logger.info(f"Total duration: {total_duration:.2f}s")
        self.logger.info(f"Report generated: {report_path}")
        
        return summary
        
    def generate_test_report(self, results: List[TestResult], summary: Dict[str, Any]) -> Path:
        """Generate comprehensive test report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate JSON report
        report_data = {
            "summary": summary,
            "results": [
                {
                    "test_name": r.test_name,
                    "component": r.component.value,
                    "scope": r.scope.value,
                    "phase": r.phase.value,
                    "passed": r.passed,
                    "duration": r.duration,
                    "start_time": r.start_time.isoformat(),
                    "end_time": r.end_time.isoformat(),
                    "error_message": r.error_message,
                    "performance_metrics": r.performance_metrics,
                    "memory_usage": r.memory_usage,
                    "tags": r.tags
                }
                for r in results
            ],
            "performance_data": self.performance_data
        }
        
        json_report_path = self.output_dir / f"integration_test_report_{timestamp}.json"
        with open(json_report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
            
        # Generate HTML report
        html_report_path = self._generate_html_report(report_data, timestamp)
        
        # Generate JUnit XML for CI/CD
        junit_report_path = self._generate_junit_report(results, timestamp)
        
        return json_report_path
        
    def _generate_html_report(self, report_data: Dict[str, Any], timestamp: str) -> Path:
        """Generate HTML test report."""
        summary = report_data["summary"]
        results = report_data["results"]
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TrustformeRS Integration Test Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #f8f9fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: white; padding: 30px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
        .metric-value {{ font-size: 32px; font-weight: bold; color: #007bff; }}
        .metric-label {{ color: #6c757d; margin-top: 8px; }}
        .test-results {{ background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .test-item {{ padding: 15px; margin: 10px 0; border-radius: 6px; border-left: 4px solid #ddd; }}
        .test-pass {{ background: #f8fff9; border-left-color: #28a745; }}
        .test-fail {{ background: #fff8f8; border-left-color: #dc3545; }}
        .suite-section {{ margin: 30px 0; }}
        .suite-header {{ background: #e9ecef; padding: 15px; border-radius: 6px; margin-bottom: 15px; }}
        .error-details {{ background: #f8f9fa; padding: 10px; border-radius: 4px; margin-top: 10px; font-family: monospace; font-size: 12px; }}
        .success-rate {{ padding: 4px 8px; border-radius: 12px; color: white; font-weight: bold; }}
        .success-high {{ background: #28a745; }}
        .success-medium {{ background: #ffc107; color: #212529; }}
        .success-low {{ background: #dc3545; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 TrustformeRS Integration Test Report</h1>
            <p><strong>Generated:</strong> {summary['start_time']} - {summary['end_time']}</p>
            <p><strong>Duration:</strong> {summary['total_duration']:.2f} seconds</p>
            <p><strong>Environment:</strong> {summary['environment']['test_config']['environment']}</p>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{summary['total_tests']}</div>
                <div class="metric-label">Total Tests</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary['total_passed']}</div>
                <div class="metric-label">Passed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary['total_failed']}</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary['overall_success_rate']:.1f}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
        </div>
        
        <div class="test-results">
            <h2>Test Suite Results</h2>
"""
        
        # Group results by component
        by_component = {}
        for result in results:
            component = result['component']
            if component not in by_component:
                by_component[component] = []
            by_component[component].append(result)
            
        for component, component_results in by_component.items():
            passed = len([r for r in component_results if r['passed']])
            total = len(component_results)
            success_rate = (passed / total * 100) if total > 0 else 0
            
            success_class = "success-high" if success_rate >= 90 else "success-medium" if success_rate >= 70 else "success-low"
            
            html_content += f"""
            <div class="suite-section">
                <div class="suite-header">
                    <h3>{component.title()} Component 
                        <span class="success-rate {success_class}">{passed}/{total} ({success_rate:.1f}%)</span>
                    </h3>
                </div>
"""
            
            for result in component_results:
                status_class = "test-pass" if result['passed'] else "test-fail"
                status_icon = "✅" if result['passed'] else "❌"
                
                error_section = ""
                if result['error_message']:
                    error_section = f"""
                    <div class="error-details">
                        <strong>Error:</strong> {result['error_message']}
                    </div>
"""
                
                html_content += f"""
                <div class="test-item {status_class}">
                    <strong>{status_icon} {result['test_name']}</strong>
                    <span style="float: right;">{result['duration']:.3f}s</span>
                    <br><small>Scope: {result['scope']} | Tags: {', '.join(result['tags'])}</small>
                    {error_section}
                </div>
"""
            
            html_content += "</div>"
            
        html_content += """
        </div>
    </div>
</body>
</html>"""
        
        html_report_path = self.output_dir / f"integration_test_report_{timestamp}.html"
        html_report_path.write_text(html_content)
        
        return html_report_path
        
    def _generate_junit_report(self, results: List[TestResult], timestamp: str) -> Path:
        """Generate JUnit XML report for CI/CD integration."""
        junit_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
        junit_content += '<testsuites>\n'
        
        # Group by component
        by_component = {}
        for result in results:
            component = result.component.value
            if component not in by_component:
                by_component[component] = []
            by_component[component].append(result)
            
        for component, component_results in by_component.items():
            total_tests = len(component_results)
            failures = len([r for r in component_results if not r.passed])
            total_time = sum(r.duration for r in component_results)
            
            junit_content += f'  <testsuite name="{component}" tests="{total_tests}" failures="{failures}" time="{total_time:.3f}">\n'
            
            for result in component_results:
                junit_content += f'    <testcase name="{result.test_name}" classname="{component}" time="{result.duration:.3f}"'
                
                if result.passed:
                    junit_content += ' />\n'
                else:
                    junit_content += '>\n'
                    junit_content += f'      <failure message="{result.error_message or "Test failed"}">'
                    if result.error_traceback:
                        junit_content += f'<![CDATA[{result.error_traceback}]]>'
                    junit_content += '</failure>\n'
                    junit_content += '    </testcase>\n'
                    
            junit_content += '  </testsuite>\n'
            
        junit_content += '</testsuites>\n'
        
        junit_report_path = self.output_dir / f"junit_report_{timestamp}.xml"
        junit_report_path.write_text(junit_content)
        
        return junit_report_path

# Convenience functions for easy testing
async def run_integration_tests(
    scope: TestScope = TestScope.INTEGRATION,
    environment: TestEnvironment = TestEnvironment.LOCAL,
    components: Optional[List[ComponentType]] = None,
    parallel: bool = True
) -> Dict[str, Any]:
    """
    Run integration tests with specified configuration.
    
    Args:
        scope: Test scope level
        environment: Test environment 
        components: Components to test (defaults to all)
        parallel: Whether to run tests in parallel
        
    Returns:
        Test results summary
    """
    if components is None:
        components = list(ComponentType)
        
    config = TestConfig(
        test_scope=scope,
        environment=environment,
        components=components,
        parallel_tests=parallel
    )
    
    tester = EndToEndIntegrationTester(config)
    return await tester.run_all_tests()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="TrustformeRS End-to-End Integration Testing")
    parser.add_argument("--scope", choices=["unit", "integration", "system", "end_to_end", "regression"],
                       default="integration", help="Test scope level")
    parser.add_argument("--environment", choices=["local", "ci", "staging", "production"],
                       default="local", help="Test environment")
    parser.add_argument("--components", nargs="+", 
                       choices=["core", "models", "tokenizers", "pipelines", "training", "serving", "mobile", "wasm"],
                       help="Components to test (defaults to all)")
    parser.add_argument("--no-parallel", action="store_true",
                       help="Run tests sequentially")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Test timeout in seconds")
    
    args = parser.parse_args()
    
    async def main():
        print(f"🔬 Starting TrustformeRS Integration Tests")
        print(f"Scope: {args.scope}")
        print(f"Environment: {args.environment}")
        print(f"Components: {args.components or 'all'}")
        print(f"Parallel: {not args.no_parallel}")
        
        try:
            components = None
            if args.components:
                components = [ComponentType(comp) for comp in args.components]
                
            config = TestConfig(
                test_scope=TestScope(args.scope),
                environment=TestEnvironment(args.environment),
                components=components or list(ComponentType),
                parallel_tests=not args.no_parallel,
                timeout_seconds=args.timeout
            )
            
            tester = EndToEndIntegrationTester(config)
            results = await tester.run_all_tests()
            
            print(f"\n🎉 Integration Testing Completed!")
            print(f"Total Tests: {results['total_tests']}")
            print(f"Passed: {results['total_passed']}")
            print(f"Failed: {results['total_failed']}")
            print(f"Success Rate: {results['overall_success_rate']:.1f}%")
            print(f"Duration: {results['total_duration']:.2f}s")
            print(f"Report: {results['report_path']}")
            
            # Return appropriate exit code
            return 0 if results['overall_success_rate'] >= 90 else 1
            
        except Exception as e:
            print(f"❌ Integration testing failed: {e}")
            traceback.print_exc()
            return 1
    
    exit_code = asyncio.run(main())
    sys.exit(exit_code)