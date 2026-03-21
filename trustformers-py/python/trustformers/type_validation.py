"""
Type validation and mypy compatibility utilities for TrustformeRS

Provides runtime type checking, mypy compatibility validation, and type utilities.
"""

import sys
import inspect
import functools
import subprocess
import tempfile
import os
from typing import (
    Any, Callable, Dict, List, Optional, Union, Tuple, Type, TypeVar, 
    get_type_hints, get_origin, get_args, overload, cast
)
from pathlib import Path
import warnings
import json
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


@dataclass 
class TypeValidationConfig:
    """Configuration for type validation."""
    strict_mode: bool = False
    validate_return_types: bool = True
    validate_parameter_types: bool = True
    allow_none_for_optional: bool = True
    custom_validators: Dict[Type, Callable[[Any], bool]] = field(default_factory=dict)
    ignore_types: List[Type] = field(default_factory=list)
    
    def __post_init__(self):
        # Add common types to ignore by default
        if not self.ignore_types:
            self.ignore_types = [Any, object, type(None)]


@dataclass
class TypeValidationError(Exception):
    """Exception raised when type validation fails."""
    parameter_name: str
    expected_type: Type
    actual_type: Type
    value: Any
    function_name: str
    
    def __str__(self) -> str:
        return (
            f"Type validation failed in {self.function_name}(): "
            f"parameter '{self.parameter_name}' expected {self.expected_type.__name__}, "
            f"got {self.actual_type.__name__} (value: {repr(self.value)})"
        )


@dataclass
class MyPyValidationResult:
    """Result of mypy validation."""
    success: bool
    errors: List[str]
    warnings: List[str]
    total_errors: int
    total_warnings: int
    files_checked: List[str]


class TypeChecker:
    """Runtime type checker with support for complex types."""
    
    def __init__(self, config: Optional[TypeValidationConfig] = None):
        self.config = config or TypeValidationConfig()
        self._type_cache: Dict[str, bool] = {}
    
    def check_type(self, value: Any, expected_type: Type, parameter_name: str = "value") -> bool:
        """
        Check if value matches expected type.
        
        Args:
            value: Value to check
            expected_type: Expected type
            parameter_name: Name of parameter (for error messages)
            
        Returns:
            True if type matches, False otherwise
            
        Raises:
            TypeValidationError: If strict mode is enabled and type doesn't match
        """
        # Cache key for performance
        cache_key = f"{type(value).__name__}:{expected_type}"
        if cache_key in self._type_cache:
            return self._type_cache[cache_key]
        
        try:
            result = self._check_type_internal(value, expected_type, parameter_name)
            self._type_cache[cache_key] = result
            return result
        except TypeValidationError:
            if self.config.strict_mode:
                raise
            return False
    
    def _check_type_internal(self, value: Any, expected_type: Type, parameter_name: str) -> bool:
        """Internal type checking logic."""
        # Handle None values
        if value is None:
            if expected_type is type(None):
                return True
            # Check if it's Optional[T]
            origin = get_origin(expected_type)
            if origin is Union:
                args = get_args(expected_type)
                if type(None) in args and self.config.allow_none_for_optional:
                    return True
            if self.config.strict_mode:
                raise TypeValidationError(
                    parameter_name=parameter_name,
                    expected_type=expected_type,
                    actual_type=type(value),
                    value=value,
                    function_name=inspect.currentframe().f_back.f_code.co_name
                )
            return False
        
        # Skip types in ignore list
        if expected_type in self.config.ignore_types:
            return True
        
        # Use custom validator if available
        if expected_type in self.config.custom_validators:
            return self.config.custom_validators[expected_type](value)
        
        # Check basic types
        if isinstance(expected_type, type) and isinstance(value, expected_type):
            return True
        
        # Handle generic types
        origin = get_origin(expected_type)
        args = get_args(expected_type)
        
        if origin is None:
            # Simple type
            if isinstance(value, expected_type):
                return True
        elif origin is Union:
            # Union type (including Optional)
            for arg in args:
                if self._check_type_internal(value, arg, parameter_name):
                    return True
        elif origin is list or origin is List:
            # List type
            if not isinstance(value, list):
                return False
            if args:
                element_type = args[0]
                return all(self._check_type_internal(item, element_type, f"{parameter_name}[{i}]") 
                          for i, item in enumerate(value))
        elif origin is dict or origin is Dict:
            # Dict type
            if not isinstance(value, dict):
                return False
            if len(args) >= 2:
                key_type, value_type = args[0], args[1]
                return all(
                    self._check_type_internal(k, key_type, f"{parameter_name}.key") and
                    self._check_type_internal(v, value_type, f"{parameter_name}[{k}]")
                    for k, v in value.items()
                )
        elif origin is tuple or origin is Tuple:
            # Tuple type
            if not isinstance(value, tuple):
                return False
            if args:
                if len(args) == 2 and args[1] is ...:
                    # Variable length tuple Tuple[T, ...]
                    element_type = args[0]
                    return all(self._check_type_internal(item, element_type, f"{parameter_name}[{i}]")
                              for i, item in enumerate(value))
                else:
                    # Fixed length tuple
                    if len(value) != len(args):
                        return False
                    return all(self._check_type_internal(value[i], args[i], f"{parameter_name}[{i}]")
                              for i in range(len(args)))
        
        # If we get here, type doesn't match
        if self.config.strict_mode:
            raise TypeValidationError(
                parameter_name=parameter_name,
                expected_type=expected_type,
                actual_type=type(value),
                value=value,
                function_name=inspect.currentframe().f_back.f_code.co_name
            )
        
        return False
    
    def add_custom_validator(self, type_: Type, validator: Callable[[Any], bool]):
        """Add a custom validator for a specific type."""
        self.config.custom_validators[type_] = validator
    
    def clear_cache(self):
        """Clear the type checking cache."""
        self._type_cache.clear()


def validate_types(
    config: Optional[TypeValidationConfig] = None,
    exclude_params: Optional[List[str]] = None
) -> Callable[[F], F]:
    """
    Decorator for runtime type validation.
    
    Args:
        config: Type validation configuration
        exclude_params: Parameter names to exclude from validation
        
    Returns:
        Decorated function with type validation
    """
    if config is None:
        config = TypeValidationConfig()
    
    if exclude_params is None:
        exclude_params = []
    
    def decorator(func: F) -> F:
        # Get type hints
        try:
            type_hints = get_type_hints(func)
        except (NameError, AttributeError):
            # If we can't get type hints, return original function
            warnings.warn(f"Could not get type hints for {func.__name__}, skipping validation")
            return func
        
        # Create type checker
        checker = TypeChecker(config)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Validate parameters
            if config.validate_parameter_types:
                for param_name, value in bound_args.arguments.items():
                    if param_name in exclude_params:
                        continue
                    
                    if param_name in type_hints:
                        expected_type = type_hints[param_name]
                        try:
                            checker.check_type(value, expected_type, param_name)
                        except TypeValidationError as e:
                            e.function_name = func.__name__
                            raise
            
            # Call original function
            result = func(*args, **kwargs)
            
            # Validate return type
            if config.validate_return_types and 'return' in type_hints:
                return_type = type_hints['return']
                try:
                    checker.check_type(result, return_type, 'return')
                except TypeValidationError as e:
                    e.function_name = func.__name__
                    e.parameter_name = 'return'
                    raise
            
            return result
        
        # Store original function for introspection
        wrapper.__wrapped__ = func
        wrapper.__type_validated__ = True
        
        return cast(F, wrapper)
    
    return decorator


class MyPyValidator:
    """Validator for mypy compatibility."""
    
    def __init__(self, mypy_path: Optional[str] = None):
        """
        Initialize mypy validator.
        
        Args:
            mypy_path: Path to mypy executable (auto-detected if None)
        """
        self.mypy_path = mypy_path or self._find_mypy()
        self.is_available = self.mypy_path is not None
    
    def _find_mypy(self) -> Optional[str]:
        """Find mypy executable."""
        try:
            result = subprocess.run(['which', 'mypy'], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except FileNotFoundError:
            pass
        
        # Try common locations
        common_paths = [
            'mypy',
            '/usr/local/bin/mypy',
            '/usr/bin/mypy',
            os.path.expanduser('~/.local/bin/mypy'),
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def validate_file(self, file_path: str, **mypy_args) -> MyPyValidationResult:
        """
        Validate a single file with mypy.
        
        Args:
            file_path: Path to Python file
            **mypy_args: Additional mypy arguments
            
        Returns:
            Validation result
        """
        if not self.is_available:
            return MyPyValidationResult(
                success=False,
                errors=["mypy is not available"],
                warnings=[],
                total_errors=1,
                total_warnings=0,
                files_checked=[]
            )
        
        return self.validate_files([file_path], **mypy_args)
    
    def validate_files(self, file_paths: List[str], **mypy_args) -> MyPyValidationResult:
        """
        Validate multiple files with mypy.
        
        Args:
            file_paths: List of Python file paths
            **mypy_args: Additional mypy arguments
            
        Returns:
            Validation result
        """
        if not self.is_available:
            return MyPyValidationResult(
                success=False,
                errors=["mypy is not available"],
                warnings=[],
                total_errors=1,
                total_warnings=0,
                files_checked=[]
            )
        
        # Build mypy command
        cmd = [self.mypy_path]
        
        # Add default arguments
        default_args = {
            '--strict': True,
            '--show-error-codes': True,
            '--pretty': True,
            '--error-format': '{path}:{line}:{column}: {severity}: {message} [{error_code}]'
        }
        
        # Override with user arguments
        mypy_args = {**default_args, **mypy_args}
        
        # Add arguments to command
        for key, value in mypy_args.items():
            if isinstance(value, bool) and value:
                cmd.append(key)
            elif not isinstance(value, bool):
                cmd.extend([key, str(value)])
        
        # Add file paths
        cmd.extend(file_paths)
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            errors = []
            warnings = []
            
            # Parse output
            for line in result.stdout.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if ': error:' in line:
                    errors.append(line)
                elif ': warning:' in line:
                    warnings.append(line)
                elif ': note:' in line:
                    warnings.append(line)
            
            # Parse stderr for additional errors
            for line in result.stderr.split('\n'):
                line = line.strip()
                if line:
                    errors.append(line)
            
            return MyPyValidationResult(
                success=result.returncode == 0,
                errors=errors,
                warnings=warnings,
                total_errors=len(errors),
                total_warnings=len(warnings),
                files_checked=file_paths
            )
            
        except subprocess.TimeoutExpired:
            return MyPyValidationResult(
                success=False,
                errors=["mypy validation timed out"],
                warnings=[],
                total_errors=1,
                total_warnings=0,
                files_checked=file_paths
            )
        except Exception as e:
            return MyPyValidationResult(
                success=False,
                errors=[f"mypy validation failed: {e}"],
                warnings=[],
                total_errors=1,
                total_warnings=0,
                files_checked=file_paths
            )
    
    def validate_module(self, module_name: str, **mypy_args) -> MyPyValidationResult:
        """
        Validate a Python module with mypy.
        
        Args:
            module_name: Name of module to validate
            **mypy_args: Additional mypy arguments
            
        Returns:
            Validation result
        """
        if not self.is_available:
            return MyPyValidationResult(
                success=False,
                errors=["mypy is not available"],
                warnings=[],
                total_errors=1,
                total_warnings=0,
                files_checked=[]
            )
        
        # Build mypy command for module
        cmd = [self.mypy_path, '-m', module_name]
        
        # Add arguments
        for key, value in mypy_args.items():
            if isinstance(value, bool) and value:
                cmd.append(key)
            elif not isinstance(value, bool):
                cmd.extend([key, str(value)])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            errors = []
            warnings = []
            
            # Parse output
            for line in result.stdout.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                if ': error:' in line:
                    errors.append(line)
                elif ': warning:' in line:
                    warnings.append(line)
                elif ': note:' in line:
                    warnings.append(line)
            
            return MyPyValidationResult(
                success=result.returncode == 0,
                errors=errors,
                warnings=warnings,
                total_errors=len(errors),
                total_warnings=len(warnings),
                files_checked=[module_name]
            )
            
        except Exception as e:
            return MyPyValidationResult(
                success=False,
                errors=[f"mypy validation failed: {e}"],
                warnings=[],
                total_errors=1,
                total_warnings=0,
                files_checked=[module_name]
            )
    
    def create_mypy_config(self, output_path: str, strict: bool = True) -> str:
        """
        Create a mypy configuration file.
        
        Args:
            output_path: Path to write mypy.ini
            strict: Whether to use strict mode
            
        Returns:
            Path to created config file
        """
        config_content = f"""[mypy]
# Global options
python_version = {sys.version_info.major}.{sys.version_info.minor}
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = {strict}
disallow_incomplete_defs = {strict}
check_untyped_defs = True
disallow_untyped_decorators = {strict}
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True
strict_concatenate = True

# Per-module options
[mypy-numpy.*]
ignore_missing_imports = True

[mypy-torch.*]
ignore_missing_imports = True

[mypy-jax.*]
ignore_missing_imports = True

[mypy-transformers.*]
ignore_missing_imports = True

[mypy-tqdm.*]
ignore_missing_imports = True

[mypy-matplotlib.*]
ignore_missing_imports = True

[mypy-sklearn.*]
ignore_missing_imports = True
"""
        
        with open(output_path, 'w') as f:
            f.write(config_content)
        
        return output_path


class TypeAnnotationChecker:
    """Check type annotation completeness."""
    
    def __init__(self):
        self.missing_annotations: Dict[str, List[str]] = {}
        self.incomplete_annotations: Dict[str, List[str]] = {}
    
    def check_function(self, func: Callable) -> Dict[str, Any]:
        """
        Check type annotation completeness for a function.
        
        Args:
            func: Function to check
            
        Returns:
            Dictionary with annotation information
        """
        sig = inspect.signature(func)
        type_hints = get_type_hints(func) if hasattr(func, '__annotations__') else {}
        
        missing_params = []
        incomplete_params = []
        
        # Check parameters
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            if param_name not in type_hints:
                missing_params.append(param_name)
            elif type_hints[param_name] is Any:
                incomplete_params.append(param_name)
        
        # Check return type
        has_return_annotation = 'return' in type_hints
        return_type_is_any = has_return_annotation and type_hints.get('return') is Any
        
        return {
            'function_name': func.__name__,
            'module': getattr(func, '__module__', 'unknown'),
            'missing_params': missing_params,
            'incomplete_params': incomplete_params,
            'has_return_annotation': has_return_annotation,
            'return_type_is_any': return_type_is_any,
            'total_params': len([p for p in sig.parameters if p != 'self']),
            'annotated_params': len([p for p in sig.parameters if p != 'self' and p in type_hints]),
            'completeness_score': self._calculate_completeness_score(
                len([p for p in sig.parameters if p != 'self']),
                len([p for p in sig.parameters if p != 'self' and p in type_hints]),
                has_return_annotation
            )
        }
    
    def _calculate_completeness_score(self, total_params: int, annotated_params: int, has_return: bool) -> float:
        """Calculate completeness score (0-1)."""
        if total_params == 0:
            return 1.0 if has_return else 0.5
        
        param_score = annotated_params / total_params
        return_score = 1.0 if has_return else 0.0
        
        # Weight parameters more heavily than return type
        return (param_score * 0.8) + (return_score * 0.2)
    
    def check_module(self, module) -> Dict[str, Any]:
        """
        Check type annotation completeness for all functions in a module.
        
        Args:
            module: Module to check
            
        Returns:
            Dictionary with module annotation information
        """
        functions = []
        total_score = 0.0
        
        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) and not name.startswith('_'):
                func_info = self.check_function(obj)
                functions.append(func_info)
                total_score += func_info['completeness_score']
        
        avg_score = total_score / len(functions) if functions else 0.0
        
        return {
            'module_name': getattr(module, '__name__', 'unknown'),
            'functions': functions,
            'total_functions': len(functions),
            'average_completeness': avg_score,
            'needs_improvement': [f for f in functions if f['completeness_score'] < 0.8]
        }


# Global instances
_default_type_checker = TypeChecker()
_default_mypy_validator = MyPyValidator()
_annotation_checker = TypeAnnotationChecker()


# Convenience functions
def check_type(value: Any, expected_type: Type, parameter_name: str = "value") -> bool:
    """Check type using default type checker."""
    return _default_type_checker.check_type(value, expected_type, parameter_name)


def validate_mypy(file_path: str, **kwargs) -> MyPyValidationResult:
    """Validate file with mypy using default validator."""
    return _default_mypy_validator.validate_file(file_path, **kwargs)


def check_annotations(func_or_module) -> Dict[str, Any]:
    """Check type annotation completeness."""
    if inspect.isfunction(func_or_module):
        return _annotation_checker.check_function(func_or_module)
    else:
        return _annotation_checker.check_module(func_or_module)


def setup_mypy_config(output_dir: str = '.', strict: bool = True) -> str:
    """Setup mypy configuration file."""
    config_path = os.path.join(output_dir, 'mypy.ini')
    return _default_mypy_validator.create_mypy_config(config_path, strict)


def create_type_stub(module, output_path: str):
    """
    Create a type stub (.pyi) file for a module.
    
    Args:
        module: Module to create stub for
        output_path: Path to write stub file
    """
    stub_content = f'"""\nType stubs for {getattr(module, "__name__", "unknown")}\n"""\n\n'
    
    # Import common types
    stub_content += "from typing import Any, Dict, List, Optional, Union, Tuple\n\n"
    
    # Process all public functions and classes
    for name, obj in inspect.getmembers(module):
        if name.startswith('_'):
            continue
            
        if inspect.isfunction(obj):
            sig = inspect.signature(obj)
            stub_content += f"def {name}{sig}: ...\n\n"
        elif inspect.isclass(obj):
            stub_content += f"class {name}:\n"
            
            # Add class methods
            for method_name, method in inspect.getmembers(obj):
                if (not method_name.startswith('_') and 
                    inspect.isfunction(method) or inspect.ismethod(method)):
                    try:
                        method_sig = inspect.signature(method)
                        stub_content += f"    def {method_name}{method_sig}: ...\n"
                    except (ValueError, TypeError):
                        stub_content += f"    def {method_name}(self, *args, **kwargs): ...\n"
            
            stub_content += "\n"
    
    with open(output_path, 'w') as f:
        f.write(stub_content)


# Validation decorators with different strictness levels
strict_validation = functools.partial(validate_types, TypeValidationConfig(strict_mode=True))
relaxed_validation = functools.partial(validate_types, TypeValidationConfig(strict_mode=False))
return_only_validation = functools.partial(
    validate_types, 
    TypeValidationConfig(validate_parameter_types=False, validate_return_types=True)
)
params_only_validation = functools.partial(
    validate_types,
    TypeValidationConfig(validate_parameter_types=True, validate_return_types=False)
)