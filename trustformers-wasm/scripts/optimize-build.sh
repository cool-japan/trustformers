#!/bin/bash

# Advanced build optimization script for trustformers-wasm
# Usage: ./optimize-build.sh <pkg-dir>

set -e

PKG_DIR=${1:-"pkg"}

if [ ! -d "$PKG_DIR" ]; then
    echo "Error: Package directory $PKG_DIR not found"
    exit 1
fi

echo "🔧 Optimizing WASM build in $PKG_DIR..."

# Find the WASM file
WASM_FILE=$(find "$PKG_DIR" -name "*.wasm" | head -1)

if [ -z "$WASM_FILE" ]; then
    echo "Error: No WASM file found in $PKG_DIR"
    exit 1
fi

echo "📁 Found WASM file: $WASM_FILE"

# Get original size
ORIGINAL_SIZE=$(stat -c%s "$WASM_FILE" 2>/dev/null || stat -f%z "$WASM_FILE")
echo "📏 Original size: $(numfmt --to=iec $ORIGINAL_SIZE)"

# 1. Run wasm-opt with aggressive optimization
if command -v wasm-opt &> /dev/null; then
    echo "⚡ Running wasm-opt optimization..."
    wasm-opt -Oz --enable-simd --enable-bulk-memory --enable-sign-ext \
             --enable-saturating-float-to-int --enable-nontrapping-float-to-int \
             --strip-debug --strip-producers --dce --vacuum \
             -o "${WASM_FILE}.opt" "$WASM_FILE"
    mv "${WASM_FILE}.opt" "$WASM_FILE"
    
    NEW_SIZE=$(stat -c%s "$WASM_FILE" 2>/dev/null || stat -f%z "$WASM_FILE")
    SAVINGS=$((ORIGINAL_SIZE - NEW_SIZE))
    echo "✅ wasm-opt saved: $(numfmt --to=iec $SAVINGS) ($(echo "scale=1; $SAVINGS * 100 / $ORIGINAL_SIZE" | bc)%)"
fi

# 2. Run wasm-snip to remove panic formatting
if command -v wasm-snip &> /dev/null; then
    echo "✂️ Running wasm-snip to remove panic code..."
    BEFORE_SNIP=$(stat -c%s "$WASM_FILE" 2>/dev/null || stat -f%z "$WASM_FILE")
    
    wasm-snip --snip-rust-fmt-code --snip-rust-panicking-code \
              -o "${WASM_FILE}.snip" "$WASM_FILE"
    mv "${WASM_FILE}.snip" "$WASM_FILE"
    
    AFTER_SNIP=$(stat -c%s "$WASM_FILE" 2>/dev/null || stat -f%z "$WASM_FILE")
    SNIP_SAVINGS=$((BEFORE_SNIP - AFTER_SNIP))
    echo "✅ wasm-snip saved: $(numfmt --to=iec $SNIP_SAVINGS) ($(echo "scale=1; $SNIP_SAVINGS * 100 / $BEFORE_SNIP" | bc)%)"
fi

# 3. Generate gzip compressed version for size comparison
echo "📦 Creating gzip compressed version..."
gzip -9 -c "$WASM_FILE" > "${WASM_FILE}.gz"
GZIP_SIZE=$(stat -c%s "${WASM_FILE}.gz" 2>/dev/null || stat -f%z "${WASM_FILE}.gz")

# 4. Generate brotli compressed version if available
if command -v brotli &> /dev/null; then
    echo "📦 Creating brotli compressed version..."
    brotli -9 -c "$WASM_FILE" > "${WASM_FILE}.br"
    BROTLI_SIZE=$(stat -c%s "${WASM_FILE}.br" 2>/dev/null || stat -f%z "${WASM_FILE}.br")
fi

# Final size report
FINAL_SIZE=$(stat -c%s "$WASM_FILE" 2>/dev/null || stat -f%z "$WASM_FILE")
TOTAL_SAVINGS=$((ORIGINAL_SIZE - FINAL_SIZE))

echo ""
echo "📊 Optimization Results:"
echo "   Original:  $(numfmt --to=iec $ORIGINAL_SIZE)"
echo "   Optimized: $(numfmt --to=iec $FINAL_SIZE)"
echo "   Savings:   $(numfmt --to=iec $TOTAL_SAVINGS) ($(echo "scale=1; $TOTAL_SAVINGS * 100 / $ORIGINAL_SIZE" | bc)%)"
echo "   Gzipped:   $(numfmt --to=iec $GZIP_SIZE) ($(echo "scale=1; $GZIP_SIZE * 100 / $ORIGINAL_SIZE" | bc)% of original)"

if [ -n "$BROTLI_SIZE" ]; then
    echo "   Brotli:    $(numfmt --to=iec $BROTLI_SIZE) ($(echo "scale=1; $BROTLI_SIZE * 100 / $ORIGINAL_SIZE" | bc)% of original)"
fi

# 5. Analyze the WASM file with twiggy if available
if command -v twiggy &> /dev/null; then
    echo ""
    echo "🔍 WASM Analysis (twiggy top 10):"
    twiggy top -n 10 "$WASM_FILE" || true
    
    # Generate detailed analysis file
    echo "📝 Generating detailed analysis..."
    twiggy top -n 50 "$WASM_FILE" > "${PKG_DIR}/twiggy-analysis.txt" || true
    twiggy dominators "$WASM_FILE" >> "${PKG_DIR}/twiggy-analysis.txt" || true
fi

# 6. Validate the optimized WASM file
echo ""
echo "✅ Validating optimized WASM..."
if command -v wasm-validate &> /dev/null; then
    if wasm-validate "$WASM_FILE"; then
        echo "✅ WASM file is valid"
    else
        echo "❌ WASM file validation failed"
        exit 1
    fi
else
    echo "⚠️ wasm-validate not found, skipping validation"
fi

# 7. Check size thresholds
echo ""
echo "🎯 Size Threshold Check:"

# Set size thresholds (in bytes)
THRESHOLD_WARNING=$((1024 * 1024))      # 1MB
THRESHOLD_ERROR=$((2 * 1024 * 1024))    # 2MB

if [ "$FINAL_SIZE" -gt "$THRESHOLD_ERROR" ]; then
    echo "❌ ERROR: WASM size $(numfmt --to=iec $FINAL_SIZE) exceeds 2MB threshold"
    exit 1
elif [ "$FINAL_SIZE" -gt "$THRESHOLD_WARNING" ]; then
    echo "⚠️ WARNING: WASM size $(numfmt --to=iec $FINAL_SIZE) exceeds 1MB threshold"
else
    echo "✅ GOOD: WASM size $(numfmt --to=iec $FINAL_SIZE) is within acceptable limits"
fi

echo ""
echo "🎉 Build optimization complete!"

# Write optimization summary to file
cat > "${PKG_DIR}/optimization-summary.json" << EOF
{
  "original_size": $ORIGINAL_SIZE,
  "optimized_size": $FINAL_SIZE,
  "savings_bytes": $TOTAL_SAVINGS,
  "savings_percent": $(echo "scale=2; $TOTAL_SAVINGS * 100 / $ORIGINAL_SIZE" | bc),
  "gzip_size": $GZIP_SIZE,
  "gzip_percent": $(echo "scale=2; $GZIP_SIZE * 100 / $ORIGINAL_SIZE" | bc),
  $([ -n "$BROTLI_SIZE" ] && echo "\"brotli_size\": $BROTLI_SIZE,")
  $([ -n "$BROTLI_SIZE" ] && echo "\"brotli_percent\": $(echo "scale=2; $BROTLI_SIZE * 100 / $ORIGINAL_SIZE" | bc),")
  "optimization_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "wasm_file": "$(basename "$WASM_FILE")"
}
EOF

echo "📄 Optimization summary saved to ${PKG_DIR}/optimization-summary.json"