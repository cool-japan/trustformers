#!/bin/bash

# Size tracking script for trustformers-wasm
# Usage: ./size-tracker.sh <pkg-dir> [baseline-file]

set -e

PKG_DIR=${1:-"pkg"}
BASELINE_FILE=${2:-"size-baseline.json"}

if [ ! -d "$PKG_DIR" ]; then
    echo "Error: Package directory $PKG_DIR not found"
    exit 1
fi

echo "📏 Tracking WASM package sizes for $PKG_DIR..."

# Initialize variables
TOTAL_SIZE=0
CURRENT_DATA="{"

# Function to add comma if not first entry
add_comma() {
    if [ "$CURRENT_DATA" != "{" ]; then
        CURRENT_DATA="$CURRENT_DATA,"
    fi
}

# Track WASM files
WASM_FILES=$(find "$PKG_DIR" -name "*.wasm" -type f)
for WASM_FILE in $WASM_FILES; do
    FILENAME=$(basename "$WASM_FILE")
    SIZE=$(stat -c%s "$WASM_FILE" 2>/dev/null || stat -f%z "$WASM_FILE")
    TOTAL_SIZE=$((TOTAL_SIZE + SIZE))
    
    add_comma
    CURRENT_DATA="$CURRENT_DATA\"$FILENAME\": {\"size\": $SIZE, \"size_human\": \"$(numfmt --to=iec $SIZE)\", \"type\": \"wasm\"}"
    
    echo "  📦 $FILENAME: $(numfmt --to=iec $SIZE)"
    
    # Check for compressed versions
    if [ -f "${WASM_FILE}.gz" ]; then
        GZIP_SIZE=$(stat -c%s "${WASM_FILE}.gz" 2>/dev/null || stat -f%z "${WASM_FILE}.gz")
        add_comma
        CURRENT_DATA="$CURRENT_DATA\"${FILENAME}.gz\": {\"size\": $GZIP_SIZE, \"size_human\": \"$(numfmt --to=iec $GZIP_SIZE)\", \"type\": \"gzip\", \"compression_ratio\": $(echo "scale=2; $GZIP_SIZE * 100 / $SIZE" | bc)}"
        echo "    🗜️ Gzipped: $(numfmt --to=iec $GZIP_SIZE) ($(echo "scale=1; $GZIP_SIZE * 100 / $SIZE" | bc)%)"
    fi
    
    if [ -f "${WASM_FILE}.br" ]; then
        BROTLI_SIZE=$(stat -c%s "${WASM_FILE}.br" 2>/dev/null || stat -f%z "${WASM_FILE}.br")
        add_comma
        CURRENT_DATA="$CURRENT_DATA\"${FILENAME}.br\": {\"size\": $BROTLI_SIZE, \"size_human\": \"$(numfmt --to=iec $BROTLI_SIZE)\", \"type\": \"brotli\", \"compression_ratio\": $(echo "scale=2; $BROTLI_SIZE * 100 / $SIZE" | bc)}"
        echo "    🗜️ Brotli: $(numfmt --to=iec $BROTLI_SIZE) ($(echo "scale=1; $BROTLI_SIZE * 100 / $SIZE" | bc)%)"
    fi
done

# Track JavaScript files
JS_FILES=$(find "$PKG_DIR" -name "*.js" -type f)
for JS_FILE in $JS_FILES; do
    FILENAME=$(basename "$JS_FILE")
    SIZE=$(stat -c%s "$JS_FILE" 2>/dev/null || stat -f%z "$JS_FILE")
    TOTAL_SIZE=$((TOTAL_SIZE + SIZE))
    
    add_comma
    CURRENT_DATA="$CURRENT_DATA\"$FILENAME\": {\"size\": $SIZE, \"size_human\": \"$(numfmt --to=iec $SIZE)\", \"type\": \"javascript\"}"
    
    echo "  📜 $FILENAME: $(numfmt --to=iec $SIZE)"
done

# Track TypeScript definition files
TS_FILES=$(find "$PKG_DIR" -name "*.d.ts" -type f)
for TS_FILE in $TS_FILES; do
    FILENAME=$(basename "$TS_FILE")
    SIZE=$(stat -c%s "$TS_FILE" 2>/dev/null || stat -f%z "$TS_FILE")
    TOTAL_SIZE=$((TOTAL_SIZE + SIZE))
    
    add_comma
    CURRENT_DATA="$CURRENT_DATA\"$FILENAME\": {\"size\": $SIZE, \"size_human\": \"$(numfmt --to=iec $SIZE)\", \"type\": \"typescript\"}"
    
    echo "  📄 $FILENAME: $(numfmt --to=iec $SIZE)"
done

# Track package.json
if [ -f "$PKG_DIR/package.json" ]; then
    SIZE=$(stat -c%s "$PKG_DIR/package.json" 2>/dev/null || stat -f%z "$PKG_DIR/package.json")
    TOTAL_SIZE=$((TOTAL_SIZE + SIZE))
    
    add_comma
    CURRENT_DATA="$CURRENT_DATA\"package.json\": {\"size\": $SIZE, \"size_human\": \"$(numfmt --to=iec $SIZE)\", \"type\": \"metadata\"}"
    
    echo "  📋 package.json: $(numfmt --to=iec $SIZE)"
fi

# Add metadata
add_comma
CURRENT_DATA="$CURRENT_DATA\"metadata\": {"
CURRENT_DATA="$CURRENT_DATA\"total_size\": $TOTAL_SIZE,"
CURRENT_DATA="$CURRENT_DATA\"total_size_human\": \"$(numfmt --to=iec $TOTAL_SIZE)\","
CURRENT_DATA="$CURRENT_DATA\"package_dir\": \"$PKG_DIR\","
CURRENT_DATA="$CURRENT_DATA\"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
CURRENT_DATA="$CURRENT_DATA\"git_commit\": \"$(git rev-parse HEAD 2>/dev/null || echo 'unknown')\","
CURRENT_DATA="$CURRENT_DATA\"git_branch\": \"$(git branch --show-current 2>/dev/null || echo 'unknown')\""
CURRENT_DATA="$CURRENT_DATA}"

# Close JSON
CURRENT_DATA="$CURRENT_DATA}"

echo ""
echo "📊 Total package size: $(numfmt --to=iec $TOTAL_SIZE)"

# Compare with baseline if it exists
if [ -f "$BASELINE_FILE" ]; then
    echo ""
    echo "📈 Comparing with baseline..."
    
    # Extract baseline total size using grep and basic tools (avoiding jq dependency)
    BASELINE_SIZE=$(grep -o '"total_size": *[0-9]*' "$BASELINE_FILE" | grep -o '[0-9]*' | head -1)
    
    if [ -n "$BASELINE_SIZE" ] && [ "$BASELINE_SIZE" -gt 0 ]; then
        SIZE_DIFF=$((TOTAL_SIZE - BASELINE_SIZE))
        
        if [ "$SIZE_DIFF" -gt 0 ]; then
            PERCENT_CHANGE=$(echo "scale=1; $SIZE_DIFF * 100 / $BASELINE_SIZE" | bc)
            echo "  📈 Size increased by $(numfmt --to=iec $SIZE_DIFF) (+${PERCENT_CHANGE}%)"
            
            # Check if increase is significant (more than 5%)
            if [ "$(echo "$PERCENT_CHANGE > 5" | bc)" -eq 1 ]; then
                echo "  ⚠️ WARNING: Significant size increase detected!"
                
                # Check if increase is critical (more than 20%)
                if [ "$(echo "$PERCENT_CHANGE > 20" | bc)" -eq 1 ]; then
                    echo "  ❌ CRITICAL: Size increase exceeds 20% threshold!"
                    echo "CRITICAL_SIZE_INCREASE=true" >> "$GITHUB_ENV" 2>/dev/null || true
                else
                    echo "WARNING_SIZE_INCREASE=true" >> "$GITHUB_ENV" 2>/dev/null || true
                fi
            fi
        elif [ "$SIZE_DIFF" -lt 0 ]; then
            SIZE_DIFF=$((-SIZE_DIFF))
            PERCENT_CHANGE=$(echo "scale=1; $SIZE_DIFF * 100 / $BASELINE_SIZE" | bc)
            echo "  📉 Size decreased by $(numfmt --to=iec $SIZE_DIFF) (-${PERCENT_CHANGE}%)"
            echo "  ✅ Good: Size optimization detected!"
        else
            echo "  ➡️ No size change from baseline"
        fi
        
        echo "  📏 Baseline: $(numfmt --to=iec $BASELINE_SIZE)"
        echo "  📏 Current:  $(numfmt --to=iec $TOTAL_SIZE)"
    else
        echo "  ⚠️ Could not parse baseline size"
    fi
else
    echo ""
    echo "📝 No baseline file found. Current size will be saved as baseline."
    echo "$CURRENT_DATA" > "$BASELINE_FILE"
    echo "  💾 Baseline saved to $BASELINE_FILE"
fi

# Save current size data
CURRENT_FILE="size-current.json"
echo "$CURRENT_DATA" > "$CURRENT_FILE"
echo ""
echo "💾 Current size data saved to $CURRENT_FILE"

# Generate size history if in CI environment
if [ -n "$GITHUB_ACTIONS" ]; then
    HISTORY_FILE="size-history.json"
    
    # Create or append to history file
    if [ -f "$HISTORY_FILE" ]; then
        # Remove the last ']' and add new entry
        sed -i '$d' "$HISTORY_FILE"
        echo "," >> "$HISTORY_FILE"
        echo "$CURRENT_DATA" >> "$HISTORY_FILE"
        echo "]" >> "$HISTORY_FILE"
    else
        echo "[$CURRENT_DATA]" > "$HISTORY_FILE"
    fi
    
    echo "📚 Size history updated in $HISTORY_FILE"
fi

# Generate human-readable report
cat > size-report.txt << EOF
📊 TrustformeRS WASM Package Size Report
========================================

Package: $PKG_DIR
Generated: $(date)
Git Commit: $(git rev-parse HEAD 2>/dev/null || echo 'unknown')
Git Branch: $(git branch --show-current 2>/dev/null || echo 'unknown')

📦 Total Package Size: $(numfmt --to=iec $TOTAL_SIZE)

📁 File Breakdown:
EOF

# Add WASM files to report
for WASM_FILE in $WASM_FILES; do
    FILENAME=$(basename "$WASM_FILE")
    SIZE=$(stat -c%s "$WASM_FILE" 2>/dev/null || stat -f%z "$WASM_FILE")
    echo "  📦 $FILENAME: $(numfmt --to=iec $SIZE)" >> size-report.txt
done

# Add JS files to report
for JS_FILE in $JS_FILES; do
    FILENAME=$(basename "$JS_FILE")
    SIZE=$(stat -c%s "$JS_FILE" 2>/dev/null || stat -f%z "$JS_FILE")
    echo "  📜 $FILENAME: $(numfmt --to=iec $SIZE)" >> size-report.txt
done

# Add TS files to report
for TS_FILE in $TS_FILES; do
    FILENAME=$(basename "$TS_FILE")
    SIZE=$(stat -c%s "$TS_FILE" 2>/dev/null || stat -f%z "$TS_FILE")
    echo "  📄 $FILENAME: $(numfmt --to=iec $SIZE)" >> size-report.txt
done

echo "" >> size-report.txt
echo "🎯 Size Targets:" >> size-report.txt
echo "  ✅ Target: < 1MB (Current: $(numfmt --to=iec $TOTAL_SIZE))" >> size-report.txt

if [ "$TOTAL_SIZE" -lt 1048576 ]; then
    echo "  ✅ EXCELLENT: Under 1MB target" >> size-report.txt
elif [ "$TOTAL_SIZE" -lt 2097152 ]; then
    echo "  ⚠️ WARNING: Over 1MB but under 2MB" >> size-report.txt
else
    echo "  ❌ CRITICAL: Over 2MB - optimization needed" >> size-report.txt
fi

echo ""
echo "📄 Human-readable report saved to size-report.txt"

# Output for CI systems
echo "TOTAL_SIZE=$TOTAL_SIZE" >> "$GITHUB_ENV" 2>/dev/null || true
echo "TOTAL_SIZE_HUMAN=$(numfmt --to=iec $TOTAL_SIZE)" >> "$GITHUB_ENV" 2>/dev/null || true

echo ""
echo "✅ Size tracking complete!"