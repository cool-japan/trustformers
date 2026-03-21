#!/bin/bash

# Enhanced build optimization script for trustformers-wasm
# This script builds and optimizes the WASM module for production use with comprehensive optimization

set -e  # Exit on any error

# Configuration
PROFILE=${1:-"release-size"}  # release, release-size
FEATURES=${2:-"size-optimized"}  # size-optimized, performance-optimized, default
ANALYSIS=${3:-"false"}  # true, false - whether to run analysis tools

echo "🚀 Building optimized WASM module..."
echo "Profile: $PROFILE"
echo "Features: $FEATURES"
echo "Analysis: $ANALYSIS"

# Install required tools if not available
install_tool() {
    local tool=$1
    local install_cmd=$2
    
    if ! command -v $tool &> /dev/null; then
        echo "📦 Installing $tool..."
        eval $install_cmd
    else
        echo "✅ $tool is available"
    fi
}

echo "🔧 Checking optimization tools..."
install_tool "wasm-opt" "
    if [[ \"$OSTYPE\" == \"darwin\"* ]]; then
        brew install binaryen
    else
        echo 'Please install binaryen from: https://github.com/WebAssembly/binaryen'
    fi
"

install_tool "wasm-snip" "cargo install wasm-snip"

if [ "$ANALYSIS" == "true" ]; then
    install_tool "twiggy" "cargo install twiggy"
    install_tool "wasm-objdump" "
        if [[ \"$OSTYPE\" == \"darwin\"* ]]; then
            brew install wabt
        else
            echo 'Please install wabt from: https://github.com/WebAssembly/wabt'
        fi
    "
fi

# Build with specified profile and features
echo "🏗️ Building with cargo..."
if [ "$FEATURES" != "default" ]; then
    cargo build --target wasm32-unknown-unknown --profile $PROFILE --features $FEATURES
else
    cargo build --target wasm32-unknown-unknown --profile $PROFILE
fi

# Get the output path
WASM_FILE="target/wasm32-unknown-unknown/$PROFILE/trustformers_wasm.wasm"

# Create backup
echo "💾 Creating backup..."
cp $WASM_FILE ${WASM_FILE}.original

# Show initial size
echo "📊 Initial WASM size:"
ls -lh $WASM_FILE | awk '{print $5 " " $9}'

# Step 1: wasm-snip (remove unused code)
if command -v wasm-snip &> /dev/null; then
    echo "✂️ Running wasm-snip to remove unused code..."
    wasm-snip --snip-rust-fmt-code --snip-rust-panicking-code -o ${WASM_FILE}.snip ${WASM_FILE}
    mv ${WASM_FILE}.snip ${WASM_FILE}
    echo "After wasm-snip:"
    ls -lh $WASM_FILE | awk '{print $5 " " $9}'
fi

# Step 2: wasm-opt (comprehensive optimization)
if command -v wasm-opt &> /dev/null; then
    echo "⚡ Running wasm-opt with aggressive optimizations..."
    
    # Multiple optimization passes for maximum size reduction
    wasm-opt \
        -Oz \
        --enable-simd \
        --enable-bulk-memory \
        --enable-sign-ext \
        --enable-mutable-globals \
        --enable-nontrapping-float-to-int \
        --enable-multivalue \
        --strip-debug \
        --strip-producers \
        --dce \
        --duplicate-function-elimination \
        --flatten \
        --merge-blocks \
        --precompute \
        --remove-unused-brs \
        --remove-unused-names \
        --reorder-functions \
        --vacuum \
        --converge \
        -o ${WASM_FILE}.opt1 \
        ${WASM_FILE}
    
    # Second pass for additional optimization
    wasm-opt \
        -Oz \
        --converge \
        --strip-debug \
        --vacuum \
        -o ${WASM_FILE}.opt2 \
        ${WASM_FILE}.opt1
    
    mv ${WASM_FILE}.opt2 ${WASM_FILE}
    rm -f ${WASM_FILE}.opt1
    
    echo "After wasm-opt:"
    ls -lh $WASM_FILE | awk '{print $5 " " $9}'
fi

# Step 3: Final analysis and reporting
echo ""
echo "📈 Optimization Summary:"
ORIGINAL_SIZE=$(stat -f%z "${WASM_FILE}.original" 2>/dev/null || stat -c%s "${WASM_FILE}.original")
FINAL_SIZE=$(stat -f%z "$WASM_FILE" 2>/dev/null || stat -c%s "$WASM_FILE")
REDUCTION=$((ORIGINAL_SIZE - FINAL_SIZE))
REDUCTION_PCT=$(echo "scale=1; $REDUCTION * 100 / $ORIGINAL_SIZE" | bc -l 2>/dev/null || echo "0")

echo "Original size: $(numfmt --to=iec $ORIGINAL_SIZE)"
echo "Final size:    $(numfmt --to=iec $FINAL_SIZE)"  
echo "Reduction:     $(numfmt --to=iec $REDUCTION) ($REDUCTION_PCT%)"

# Optional detailed analysis
if [ "$ANALYSIS" == "true" ]; then
    echo ""
    echo "🔍 Running detailed analysis..."
    
    if command -v twiggy &> /dev/null; then
        echo "📋 Top code size contributors:"
        twiggy top -n 10 "$WASM_FILE"
        
        echo ""
        echo "🌳 Function call dominators:"
        twiggy dominators -d 3 "$WASM_FILE"
    fi
    
    if command -v wasm-objdump &> /dev/null; then
        echo ""
        echo "📊 Section sizes:"
        wasm-objdump --section-sizes "$WASM_FILE"
    fi
fi

# Compression analysis
echo ""
echo "📦 Compression analysis:"
if command -v gzip &> /dev/null; then
    GZIP_SIZE=$(gzip -c "$WASM_FILE" | wc -c)
    echo "Gzip size:     $(numfmt --to=iec $GZIP_SIZE)"
fi

if command -v brotli &> /dev/null; then
    BROTLI_SIZE=$(brotli -c "$WASM_FILE" | wc -c)
    echo "Brotli size:   $(numfmt --to=iec $BROTLI_SIZE)"
fi

echo ""
echo "✅ Optimization complete!"
echo "📁 Optimized WASM file: $WASM_FILE"
echo "📁 Original backup: ${WASM_FILE}.original"

# Deployment recommendations
echo ""
echo "🚀 Deployment recommendations:"
echo "1. Enable brotli compression on your web server"
echo "2. Set proper cache headers (Cache-Control: max-age=31536000)"
echo "3. Use HTTP/2 or HTTP/3 for better loading performance"
echo "4. Consider code splitting for large applications"
echo "5. Implement lazy loading for non-critical features"

# Usage examples
echo ""
echo "💡 Usage examples:"
echo "  ./optimize.sh release-size size-optimized false   # Maximum size optimization"
echo "  ./optimize.sh release performance-optimized true  # Performance with analysis"
echo "  ./optimize.sh release-size default true          # Balanced with full analysis"