"""
General utilities for TrustformeRS
"""

import os
import sys
import logging as _logging
import tempfile
import hashlib
from pathlib import Path
from typing import Optional, Union
import warnings
import requests
from tqdm import tqdm


# Set up logging
class _LazyModule:
    """Lazy module for logging compatibility."""
    
    def __getattr__(self, name):
        # Handle HuggingFace-style get_logger method
        if name == "get_logger":
            return _logging.getLogger
        return getattr(_logging, name)


logging = _LazyModule()

# Configure default logger
_logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=_logging.INFO,
)

logger = _logging.getLogger(__name__)


def is_torch_available():
    """Check if PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False


def is_tf_available():
    """Check if TensorFlow is available."""
    try:
        import tensorflow
        return True
    except ImportError:
        return False


def is_numpy_available():
    """Check if NumPy is available."""
    try:
        import numpy
        return True
    except ImportError:
        return False


def requires_backends(obj, backends):
    """
    Require certain backends to be available.
    
    Args:
        obj: The object requiring the backends
        backends: List of backend names (e.g., ["torch", "tensorflow"])
    """
    missing = []
    
    for backend in backends:
        if backend == "torch" and not is_torch_available():
            missing.append("torch (PyTorch)")
        elif backend == "tensorflow" and not is_tf_available():
            missing.append("tensorflow")
        elif backend == "numpy" and not is_numpy_available():
            missing.append("numpy")
    
    if missing:
        raise ImportError(
            f"{obj.__class__.__name__} requires the following backends: {', '.join(missing)}. "
            f"Please install them."
        )


def cached_path(
    url_or_filename: Union[str, Path],
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    user_agent: Optional[str] = None,
) -> str:
    """
    Given a URL or local path, return the local path to the cached file.
    
    Args:
        url_or_filename: URL or local file path
        cache_dir: Directory to cache downloads
        force_download: Force re-download even if cached
        resume_download: Resume incomplete downloads
        user_agent: User agent for downloads
    
    Returns:
        Local path to the file
    """
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    
    # Check if it's a URL
    if url_or_filename.startswith(("http://", "https://")):
        return download_file(
            url_or_filename,
            cache_dir=cache_dir,
            force_download=force_download,
            resume_download=resume_download,
            user_agent=user_agent,
        )
    
    # It's a local path
    if os.path.exists(url_or_filename):
        return url_or_filename
    
    raise ValueError(f"File not found: {url_or_filename}")


def download_file(
    url: str,
    cache_dir: Optional[Union[str, Path]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    user_agent: Optional[str] = None,
) -> str:
    """
    Download a file from a URL to cache.
    
    Args:
        url: URL to download from
        cache_dir: Directory to cache downloads
        force_download: Force re-download even if cached
        resume_download: Resume incomplete downloads
        user_agent: User agent for downloads
        
    Returns:
        Local path to downloaded file
    """
    if cache_dir is None:
        cache_dir = os.path.join(tempfile.gettempdir(), "trustformers_cache")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create filename from URL hash
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    filename = os.path.basename(url) or "downloaded_file"
    cache_path = os.path.join(cache_dir, f"{url_hash}_{filename}")
    
    # Check if already cached
    if os.path.exists(cache_path) and not force_download:
        logger.info(f"Using cached file: {cache_path}")
        return cache_path
    
    # Download file
    logger.info(f"Downloading {url} to {cache_path}")
    
    headers = {}
    if user_agent:
        headers["User-Agent"] = user_agent
    
    # Check if we should resume
    if resume_download and os.path.exists(cache_path):
        headers["Range"] = f"bytes={os.path.getsize(cache_path)}-"
        mode = "ab"
    else:
        mode = "wb"
    
    response = requests.get(url, headers=headers, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get("content-length", 0))
    
    with open(cache_path, mode) as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=filename) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    return cache_path


class TrustformersDeprecationWarning(FutureWarning):
    """Custom deprecation warning for TrustformeRS."""
    pass


def deprecate(message: str, version: str):
    """
    Issue a deprecation warning.
    
    Args:
        message: Deprecation message
        version: Version when feature will be removed
    """
    warnings.warn(
        f"{message} This will be removed in version {version}.",
        TrustformersDeprecationWarning,
        stacklevel=2,
    )


# Model name mappings for compatibility
MODEL_MAPPING = {
    # BERT variants
    "bert-base-uncased": "bert-base-uncased",
    "bert-base-cased": "bert-base-cased", 
    "bert-large-uncased": "bert-large-uncased",
    "bert-large-cased": "bert-large-cased",
    
    # GPT-2 variants
    "gpt2": "gpt2",
    "gpt2-medium": "gpt2-medium",
    "gpt2-large": "gpt2-large",
    "gpt2-xl": "gpt2-xl",
    
    # T5 variants
    "t5-small": "t5-small",
    "t5-base": "t5-base",
    "t5-large": "t5-large",
    
    # LLaMA variants
    "llama-7b": "llama-7b",
    "llama-13b": "llama-13b",
    "llama-30b": "llama-30b",
    "llama-65b": "llama-65b",
}


def get_model_type(model_name_or_path: str) -> str:
    """
    Get the model type from model name or path.
    
    Args:
        model_name_or_path: Model name or path
        
    Returns:
        Model type (e.g., "bert", "gpt2", etc.)
    """
    model_name = model_name_or_path.lower()
    
    if "bert" in model_name and "roberta" not in model_name:
        return "bert"
    elif "roberta" in model_name:
        return "roberta"
    elif "gpt2" in model_name or "gpt-2" in model_name:
        return "gpt2"
    elif "gpt-j" in model_name:
        return "gptj"
    elif "gpt-neo" in model_name:
        return "gpt_neo"
    elif "t5" in model_name:
        return "t5"
    elif "llama" in model_name:
        return "llama"
    elif "mistral" in model_name:
        return "mistral"
    else:
        # Try to infer from config file if it's a directory
        if os.path.isdir(model_name_or_path):
            config_path = os.path.join(model_name_or_path, "config.json")
            if os.path.exists(config_path):
                import json
                with open(config_path, "r") as f:
                    config = json.load(f)
                    return config.get("model_type", "unknown")
        
        return "unknown"