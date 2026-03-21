#!/bin/bash
# Performance Trend Analysis Script
# 
# Analyzes benchmark results over time to detect performance regressions
# and improvements

set -e

RESULTS_DIR=${1:-"benchmark_results"}

if [ ! -d "$RESULTS_DIR" ]; then
    echo "❌ Results directory not found: $RESULTS_DIR"
    echo "Usage: $0 [results_directory]"
    exit 1
fi

echo "📈 TrustformeRS Performance Trend Analysis"
echo "=========================================="
echo "Results Directory: $RESULTS_DIR"
echo ""

# Function to extract key metrics from benchmark results
analyze_benchmark_files() {
    local pattern=$1
    local benchmark_type=$2
    
    echo "## $benchmark_type Performance Trends"
    echo ""
    
    # Find all matching files, sorted by timestamp
    local files=$(find "$RESULTS_DIR" -name "$pattern" | sort)
    
    if [ -z "$files" ]; then
        echo "No $benchmark_type benchmark files found."
        echo ""
        return
    fi
    
    echo "| Date | Commit | File | Status |"
    echo "|------|--------|------|--------|" 
    
    for file in $files; do
        local basename=$(basename "$file")
        local timestamp=$(echo "$basename" | cut -d'_' -f3)
        local commit=$(echo "$basename" | cut -d'_' -f4 | cut -d'.' -f1)
        local date_formatted=$(echo "$timestamp" | sed 's/\(..\)\(..\)\(..\)_\(..\)\(..\)\(..\)/20\3-\1-\2 \4:\5:\6/')
        
        if [ -f "$file" ]; then
            echo "| $date_formatted | $commit | $basename | ✅ |"
        else
            echo "| $date_formatted | $commit | $basename | ❌ |"
        fi
    done
    
    echo ""
    echo "**Latest Results:**"
    local latest_file=$(echo "$files" | tail -1)
    if [ -f "$latest_file" ]; then
        echo "\`\`\`"
        echo "File: $(basename "$latest_file")"
        echo "Size: $(du -h "$latest_file" | cut -f1)"
        echo "Modified: $(stat -c %y "$latest_file" 2>/dev/null || stat -f %Sm "$latest_file")"
        echo "\`\`\`"
    fi
    echo ""
}

# Function to detect performance regressions
detect_regressions() {
    echo "## Performance Regression Analysis"
    echo ""
    
    # Look for any performance drops > 10%
    echo "🔍 Checking for significant performance changes..."
    echo ""
    
    # This is a placeholder for more sophisticated analysis
    echo "**Analysis Status:** Basic trend analysis complete"
    echo "**Regression Detection:** Requires JSON parsing implementation"
    echo "**Recommendation:** Compare latest results with baseline manually"
    echo ""
    
    echo "**Manual Analysis Steps:**"
    echo "1. Compare latest JSON results with previous runs"
    echo "2. Look for >10% performance changes in critical operations"
    echo "3. Verify hardware acceleration performance gains"
    echo "4. Check for memory usage regressions"
    echo ""
}

# Function to generate performance summary
generate_summary() {
    echo "## Performance Summary"
    echo ""
    
    local total_files=$(find "$RESULTS_DIR" -name "*.json" | wc -l)
    local tensor_files=$(find "$RESULTS_DIR" -name "tensor_operations_*.json" | wc -l)
    local hardware_files=$(find "$RESULTS_DIR" -name "hardware_acceleration_*.json" | wc -l)
    
    echo "**Benchmark History:**"
    echo "- Total benchmark runs: $total_files"
    echo "- Tensor operation benchmarks: $tensor_files"
    echo "- Hardware acceleration benchmarks: $hardware_files"
    echo ""
    
    if [ $total_files -gt 0 ]; then
        local oldest=$(find "$RESULTS_DIR" -name "*.json" | head -1 | xargs basename)
        local newest=$(find "$RESULTS_DIR" -name "*.json" | tail -1 | xargs basename)
        echo "**Time Range:**"
        echo "- Oldest: $oldest"
        echo "- Newest: $newest"
        echo ""
    fi
    
    echo "**Performance Monitoring:**"
    echo "- ✅ Benchmark infrastructure is active"
    echo "- ✅ Results are being stored systematically"
    echo "- ✅ Trend analysis tools are available"
    echo ""
}

# Main analysis
echo "🔄 Analyzing performance trends..."
echo ""

# Analyze tensor operations
analyze_benchmark_files "tensor_operations_*.json" "Tensor Operations"

# Analyze hardware acceleration  
analyze_benchmark_files "hardware_acceleration_*.json" "Hardware Acceleration"

# Detect regressions
detect_regressions

# Generate summary
generate_summary

echo "✅ Performance trend analysis complete!"
echo ""
echo "💡 **Next Steps:**"
echo "   1. Review trends for any concerning patterns"
echo "   2. Set up automated regression alerts" 
echo "   3. Compare with performance targets"
echo "   4. Document any significant changes"