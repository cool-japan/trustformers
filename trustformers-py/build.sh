#!/bin/bash
# Build script for TrustformeRS Python package

set -e

echo "Building TrustformeRS Python package..."

# Check if maturin is installed
if ! command -v maturin &> /dev/null; then
    echo "maturin is not installed. Installing..."
    pip install maturin
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ *.egg-info target/wheels

# Build the package
echo "Building with maturin..."
maturin build --release

# Optional: Install locally for testing
if [ "$1" == "--install" ]; then
    echo "Installing package locally..."
    pip install target/wheels/*.whl --force-reinstall
fi

echo "Build complete! Wheel files are in target/wheels/"
echo ""
echo "To install locally, run:"
echo "  pip install target/wheels/*.whl"
echo ""
echo "To upload to PyPI, run:"
echo "  maturin publish"