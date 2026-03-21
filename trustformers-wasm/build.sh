#!/bin/bash

# Enhanced build script for trustformers-wasm with optimization options

set -e  # Exit on any error

# Configuration
BUILD_TYPE=${1:-"release"}  # release, release-size, dev, dev-opt
TARGET_TYPE=${2:-"all"}     # web, bundler, nodejs, all
FEATURES=${3:-"default"}    # default, size-optimized, performance-optimized, full

echo "🚀 Building trustformers-wasm..."
echo "Build type: $BUILD_TYPE"
echo "Target type: $TARGET_TYPE"
echo "Features: $FEATURES"

# Install wasm-pack if not already installed
if ! command -v wasm-pack &> /dev/null; then
    echo "📦 Installing wasm-pack..."
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh
fi

# Set build flags based on build type
case $BUILD_TYPE in
    "release-size")
        PROFILE_FLAG="--profile release-size"
        echo "🗜️ Building for minimum size..."
        ;;
    "release")
        PROFILE_FLAG="--release"
        echo "⚡ Building for release..."
        ;;
    "dev-opt")
        PROFILE_FLAG="--profile dev-opt"
        echo "🔧 Building with dev optimizations..."
        ;;
    "dev")
        PROFILE_FLAG="--dev"
        echo "🐛 Building for development..."
        ;;
    *)
        PROFILE_FLAG="--release"
        echo "⚡ Building for release (default)..."
        ;;
esac

# Function to build for a specific target
build_target() {
    local target=$1
    local out_dir="pkg-${target}"
    
    echo "🎯 Building for ${target} target..."
    
    # Build with specified profile and features
    if [ "$FEATURES" != "default" ]; then
        wasm-pack build --target $target --out-dir $out_dir $PROFILE_FLAG --features $FEATURES
    else
        wasm-pack build --target $target --out-dir $out_dir $PROFILE_FLAG
    fi
    
    # Display build size
    if [ -f "${out_dir}/trustformers_wasm_bg.wasm" ]; then
        local size=$(du -h "${out_dir}/trustformers_wasm_bg.wasm" | cut -f1)
        echo "📊 ${target} build size: $size"
    fi
}

# Build for specified targets
case $TARGET_TYPE in
    "web")
        build_target web
        ;;
    "bundler")
        build_target bundler
        ;;
    "nodejs")
        build_target nodejs
        ;;
    "all")
        build_target web
        build_target bundler
        build_target nodejs
        ;;
    *)
        echo "❌ Unknown target type: $TARGET_TYPE"
        echo "Valid options: web, bundler, nodejs, all"
        exit 1
        ;;
esac

echo "✅ Build complete!"
echo ""
echo "📁 Packages generated:"
if [ "$TARGET_TYPE" == "all" ] || [ "$TARGET_TYPE" == "web" ]; then
    echo "  - pkg-web/ (for direct browser usage)"
fi
if [ "$TARGET_TYPE" == "all" ] || [ "$TARGET_TYPE" == "bundler" ]; then
    echo "  - pkg-bundler/ (for webpack/bundler usage)"  
fi
if [ "$TARGET_TYPE" == "all" ] || [ "$TARGET_TYPE" == "nodejs" ]; then
    echo "  - pkg-node/ (for Node.js usage)"
fi

echo ""
echo "💡 Usage examples:"
echo "  ./build.sh release-size web size-optimized  # Minimum size web build"
echo "  ./build.sh release bundler performance-optimized  # Performance bundler build"
echo "  ./build.sh dev all default  # Development build for all targets"

# Additional optimizations info
if [ "$BUILD_TYPE" == "release-size" ]; then
    echo ""
    echo "🔍 Size optimization enabled. For further reduction:"
    echo "  - Use brotli compression on the server"
    echo "  - Consider lazy loading with dynamic imports"
    echo "  - Enable gzip compression"
    echo "  - Use twiggy for detailed size analysis: cargo install twiggy"
fi