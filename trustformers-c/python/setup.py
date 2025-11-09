"""
Setup script for TrustformeRS-C Python bindings
"""

from setuptools import setup, find_packages
from pathlib import Path
import os

# Read the README file
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "Python bindings for TrustformeRS - High-performance transformer library"

# Get version from __init__.py
version = "0.1.0"
init_path = Path(__file__).parent / "trustformers_c" / "__init__.py"
if init_path.exists():
    with open(init_path, "r") as f:
        for line in f:
            if line.startswith("__version__"):
                version = line.split("=")[1].strip().strip('"').strip("'")
                break

setup(
    name="trustformers-c",
    version=version,
    author="Cool Japan",
    author_email="info@cool-japan.com",
    description="Python bindings for TrustformeRS - High-performance transformer library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cool-japan/trustformers",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.910",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-asyncio>=0.18.0",
            "pytest-cov>=3.0",
            "pytest-benchmark>=3.4",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.17",
            "sphinx-autoapi>=1.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "trustformers-c-info=trustformers_c.cli:info_command",
            "trustformers-c-benchmark=trustformers_c.cli:benchmark_command",
            "trustformers-c-config=trustformers_c.cli:config_command",
        ],
    },
    package_data={
        "trustformers_c": [
            "*.so",
            "*.dll",
            "*.dylib",
            "*.pyd",
            "examples/*.py",
            "tests/*.py",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    project_urls={
        "Bug Reports": "https://github.com/cool-japan/trustformers/issues",
        "Source": "https://github.com/cool-japan/trustformers",
        "Documentation": "https://trustformers.readthedocs.io/",
    },
    keywords="transformer, deep learning, machine learning, neural network, AI, performance",
)