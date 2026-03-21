#!/bin/bash

# Performance monitoring script for trustformers-wasm
# Usage: ./performance-monitor.sh <performance-report.json> [baseline.json]

set -e

REPORT_FILE=${1:-"performance-report.json"}
BASELINE_FILE=${2:-"performance-baseline.json"}

if [ ! -f "$REPORT_FILE" ]; then
    echo "Error: Performance report file $REPORT_FILE not found"
    exit 1
fi

echo "⚡ Monitoring performance for TrustformeRS WASM..."

# Function to extract JSON values (basic implementation without jq)
extract_json_value() {
    local file="$1"
    local key="$2"
    grep -o "\"$key\": *[0-9.]*" "$file" | grep -o '[0-9.]*' | head -1
}

extract_json_string() {
    local file="$1"
    local key="$2"
    grep -o "\"$key\": *\"[^\"]*\"" "$file" | sed 's/.*: *"\([^"]*\)".*/\1/' | head -1
}

# Parse current performance data
echo "📊 Current Performance Results:"

# Basic metrics
TENSOR_CREATE_TIME=$(extract_json_value "$REPORT_FILE" "tensor_creation_avg_ms")
MATRIX_MULT_TIME=$(extract_json_value "$REPORT_FILE" "matrix_multiplication_avg_ms")
MODEL_LOAD_TIME=$(extract_json_value "$REPORT_FILE" "model_loading_avg_ms")
INFERENCE_TIME=$(extract_json_value "$REPORT_FILE" "inference_avg_ms")
MEMORY_USAGE=$(extract_json_value "$REPORT_FILE" "peak_memory_mb")

# Display current metrics
echo "  🏗️ Tensor Creation: ${TENSOR_CREATE_TIME}ms"
echo "  🔢 Matrix Multiplication: ${MATRIX_MULT_TIME}ms"
echo "  📦 Model Loading: ${MODEL_LOAD_TIME}ms"
echo "  🧠 Inference: ${INFERENCE_TIME}ms"
echo "  💾 Peak Memory Usage: ${MEMORY_USAGE}MB"

# Check WebGPU vs CPU performance if available
WEBGPU_SPEEDUP=$(extract_json_value "$REPORT_FILE" "webgpu_speedup_factor")
if [ -n "$WEBGPU_SPEEDUP" ]; then
    echo "  🚀 WebGPU Speedup: ${WEBGPU_SPEEDUP}x"
fi

# Performance thresholds
TENSOR_THRESHOLD=1.0     # 1ms
MATRIX_THRESHOLD=10.0    # 10ms
MODEL_THRESHOLD=1000.0   # 1 second
INFERENCE_THRESHOLD=100.0 # 100ms
MEMORY_THRESHOLD=100.0   # 100MB

# Performance status tracking
PERFORMANCE_ISSUES=0
PERFORMANCE_WARNINGS=0

echo ""
echo "🎯 Performance Threshold Check:"

# Check tensor creation performance
if [ -n "$TENSOR_CREATE_TIME" ]; then
    if (( $(echo "$TENSOR_CREATE_TIME > $TENSOR_THRESHOLD" | bc -l) )); then
        echo "  ⚠️ WARNING: Tensor creation time (${TENSOR_CREATE_TIME}ms) exceeds threshold (${TENSOR_THRESHOLD}ms)"
        PERFORMANCE_WARNINGS=$((PERFORMANCE_WARNINGS + 1))
    else
        echo "  ✅ Tensor creation: ${TENSOR_CREATE_TIME}ms (good)"
    fi
fi

# Check matrix multiplication performance
if [ -n "$MATRIX_MULT_TIME" ]; then
    if (( $(echo "$MATRIX_MULT_TIME > $MATRIX_THRESHOLD" | bc -l) )); then
        echo "  ⚠️ WARNING: Matrix multiplication time (${MATRIX_MULT_TIME}ms) exceeds threshold (${MATRIX_THRESHOLD}ms)"
        PERFORMANCE_WARNINGS=$((PERFORMANCE_WARNINGS + 1))
    else
        echo "  ✅ Matrix multiplication: ${MATRIX_MULT_TIME}ms (good)"
    fi
fi

# Check model loading performance
if [ -n "$MODEL_LOAD_TIME" ]; then
    if (( $(echo "$MODEL_LOAD_TIME > $MODEL_THRESHOLD" | bc -l) )); then
        echo "  ❌ ERROR: Model loading time (${MODEL_LOAD_TIME}ms) exceeds threshold (${MODEL_THRESHOLD}ms)"
        PERFORMANCE_ISSUES=$((PERFORMANCE_ISSUES + 1))
    else
        echo "  ✅ Model loading: ${MODEL_LOAD_TIME}ms (good)"
    fi
fi

# Check inference performance
if [ -n "$INFERENCE_TIME" ]; then
    if (( $(echo "$INFERENCE_TIME > $INFERENCE_THRESHOLD" | bc -l) )); then
        echo "  ⚠️ WARNING: Inference time (${INFERENCE_TIME}ms) exceeds threshold (${INFERENCE_THRESHOLD}ms)"
        PERFORMANCE_WARNINGS=$((PERFORMANCE_WARNINGS + 1))
    else
        echo "  ✅ Inference: ${INFERENCE_TIME}ms (good)"
    fi
fi

# Check memory usage
if [ -n "$MEMORY_USAGE" ]; then
    if (( $(echo "$MEMORY_USAGE > $MEMORY_THRESHOLD" | bc -l) )); then
        echo "  ⚠️ WARNING: Memory usage (${MEMORY_USAGE}MB) exceeds threshold (${MEMORY_THRESHOLD}MB)"
        PERFORMANCE_WARNINGS=$((PERFORMANCE_WARNINGS + 1))
    else
        echo "  ✅ Memory usage: ${MEMORY_USAGE}MB (good)"
    fi
fi

# Compare with baseline if available
if [ -f "$BASELINE_FILE" ]; then
    echo ""
    echo "📈 Baseline Comparison:"
    
    # Extract baseline metrics
    BASELINE_TENSOR=$(extract_json_value "$BASELINE_FILE" "tensor_creation_avg_ms")
    BASELINE_MATRIX=$(extract_json_value "$BASELINE_FILE" "matrix_multiplication_avg_ms")
    BASELINE_MODEL=$(extract_json_value "$BASELINE_FILE" "model_loading_avg_ms")
    BASELINE_INFERENCE=$(extract_json_value "$BASELINE_FILE" "inference_avg_ms")
    BASELINE_MEMORY=$(extract_json_value "$BASELINE_FILE" "peak_memory_mb")
    
    # Compare tensor creation
    if [ -n "$TENSOR_CREATE_TIME" ] && [ -n "$BASELINE_TENSOR" ]; then
        TENSOR_CHANGE=$(echo "scale=2; ($TENSOR_CREATE_TIME - $BASELINE_TENSOR) * 100 / $BASELINE_TENSOR" | bc)
        if (( $(echo "$TENSOR_CHANGE > 10" | bc -l) )); then
            echo "  📈 ⚠️ Tensor creation: +${TENSOR_CHANGE}% slower (${TENSOR_CREATE_TIME}ms vs ${BASELINE_TENSOR}ms)"
            PERFORMANCE_WARNINGS=$((PERFORMANCE_WARNINGS + 1))
        elif (( $(echo "$TENSOR_CHANGE < -10" | bc -l) )); then
            echo "  📉 ✅ Tensor creation: ${TENSOR_CHANGE}% faster (${TENSOR_CREATE_TIME}ms vs ${BASELINE_TENSOR}ms)"
        else
            echo "  ➡️ Tensor creation: ${TENSOR_CHANGE}% change (${TENSOR_CREATE_TIME}ms vs ${BASELINE_TENSOR}ms)"
        fi
    fi
    
    # Compare matrix multiplication
    if [ -n "$MATRIX_MULT_TIME" ] && [ -n "$BASELINE_MATRIX" ]; then
        MATRIX_CHANGE=$(echo "scale=2; ($MATRIX_MULT_TIME - $BASELINE_MATRIX) * 100 / $BASELINE_MATRIX" | bc)
        if (( $(echo "$MATRIX_CHANGE > 15" | bc -l) )); then
            echo "  📈 ❌ Matrix multiplication: +${MATRIX_CHANGE}% slower (${MATRIX_MULT_TIME}ms vs ${BASELINE_MATRIX}ms)"
            PERFORMANCE_ISSUES=$((PERFORMANCE_ISSUES + 1))
        elif (( $(echo "$MATRIX_CHANGE < -15" | bc -l) )); then
            echo "  📉 ✅ Matrix multiplication: ${MATRIX_CHANGE}% faster (${MATRIX_MULT_TIME}ms vs ${BASELINE_MATRIX}ms)"
        else
            echo "  ➡️ Matrix multiplication: ${MATRIX_CHANGE}% change (${MATRIX_MULT_TIME}ms vs ${BASELINE_MATRIX}ms)"
        fi
    fi
    
    # Compare model loading
    if [ -n "$MODEL_LOAD_TIME" ] && [ -n "$BASELINE_MODEL" ]; then
        MODEL_CHANGE=$(echo "scale=2; ($MODEL_LOAD_TIME - $BASELINE_MODEL) * 100 / $BASELINE_MODEL" | bc)
        if (( $(echo "$MODEL_CHANGE > 20" | bc -l) )); then
            echo "  📈 ❌ Model loading: +${MODEL_CHANGE}% slower (${MODEL_LOAD_TIME}ms vs ${BASELINE_MODEL}ms)"
            PERFORMANCE_ISSUES=$((PERFORMANCE_ISSUES + 1))
        elif (( $(echo "$MODEL_CHANGE < -20" | bc -l) )); then
            echo "  📉 ✅ Model loading: ${MODEL_CHANGE}% faster (${MODEL_LOAD_TIME}ms vs ${BASELINE_MODEL}ms)"
        else
            echo "  ➡️ Model loading: ${MODEL_CHANGE}% change (${MODEL_LOAD_TIME}ms vs ${BASELINE_MODEL}ms)"
        fi
    fi
    
    # Compare inference
    if [ -n "$INFERENCE_TIME" ] && [ -n "$BASELINE_INFERENCE" ]; then
        INFERENCE_CHANGE=$(echo "scale=2; ($INFERENCE_TIME - $BASELINE_INFERENCE) * 100 / $BASELINE_INFERENCE" | bc)
        if (( $(echo "$INFERENCE_CHANGE > 15" | bc -l) )); then
            echo "  📈 ❌ Inference: +${INFERENCE_CHANGE}% slower (${INFERENCE_TIME}ms vs ${BASELINE_INFERENCE}ms)"
            PERFORMANCE_ISSUES=$((PERFORMANCE_ISSUES + 1))
        elif (( $(echo "$INFERENCE_CHANGE < -15" | bc -l) )); then
            echo "  📉 ✅ Inference: ${INFERENCE_CHANGE}% faster (${INFERENCE_TIME}ms vs ${BASELINE_INFERENCE}ms)"
        else
            echo "  ➡️ Inference: ${INFERENCE_CHANGE}% change (${INFERENCE_TIME}ms vs ${BASELINE_INFERENCE}ms)"
        fi
    fi
    
    # Compare memory usage
    if [ -n "$MEMORY_USAGE" ] && [ -n "$BASELINE_MEMORY" ]; then
        MEMORY_CHANGE=$(echo "scale=2; ($MEMORY_USAGE - $BASELINE_MEMORY) * 100 / $BASELINE_MEMORY" | bc)
        if (( $(echo "$MEMORY_CHANGE > 25" | bc -l) )); then
            echo "  📈 ⚠️ Memory usage: +${MEMORY_CHANGE}% increase (${MEMORY_USAGE}MB vs ${BASELINE_MEMORY}MB)"
            PERFORMANCE_WARNINGS=$((PERFORMANCE_WARNINGS + 1))
        elif (( $(echo "$MEMORY_CHANGE < -25" | bc -l) )); then
            echo "  📉 ✅ Memory usage: ${MEMORY_CHANGE}% reduction (${MEMORY_USAGE}MB vs ${BASELINE_MEMORY}MB)"
        else
            echo "  ➡️ Memory usage: ${MEMORY_CHANGE}% change (${MEMORY_USAGE}MB vs ${BASELINE_MEMORY}MB)"
        fi
    fi
    
else
    echo ""
    echo "📝 No baseline found. Current performance will be saved as baseline."
    cp "$REPORT_FILE" "$BASELINE_FILE"
    echo "  💾 Baseline saved to $BASELINE_FILE"
fi

# Generate performance summary
echo ""
echo "📋 Performance Summary:"
echo "  ✅ Tests passed: $((5 - PERFORMANCE_ISSUES - PERFORMANCE_WARNINGS))"
echo "  ⚠️ Warnings: $PERFORMANCE_WARNINGS"
echo "  ❌ Issues: $PERFORMANCE_ISSUES"

# Set environment variables for CI
if [ -n "$GITHUB_ENV" ]; then
    echo "PERFORMANCE_WARNINGS=$PERFORMANCE_WARNINGS" >> "$GITHUB_ENV"
    echo "PERFORMANCE_ISSUES=$PERFORMANCE_ISSUES" >> "$GITHUB_ENV"
    
    # Create performance status
    if [ "$PERFORMANCE_ISSUES" -gt 0 ]; then
        echo "PERFORMANCE_STATUS=failed" >> "$GITHUB_ENV"
    elif [ "$PERFORMANCE_WARNINGS" -gt 0 ]; then
        echo "PERFORMANCE_STATUS=warning" >> "$GITHUB_ENV"
    else
        echo "PERFORMANCE_STATUS=passed" >> "$GITHUB_ENV"
    fi
fi

# Generate detailed performance report
cat > performance-summary.md << EOF
# 🚀 TrustformeRS WASM Performance Report

**Generated:** $(date)  
**Git Commit:** $(git rev-parse HEAD 2>/dev/null || echo 'unknown')  
**Git Branch:** $(git branch --show-current 2>/dev/null || echo 'unknown')  

## 📊 Current Performance

| Metric | Value | Status |
|--------|-------|--------|
| Tensor Creation | ${TENSOR_CREATE_TIME}ms | $([ -n "$TENSOR_CREATE_TIME" ] && (( $(echo "$TENSOR_CREATE_TIME <= $TENSOR_THRESHOLD" | bc -l) )) && echo "✅ Good" || echo "⚠️ Needs optimization") |
| Matrix Multiplication | ${MATRIX_MULT_TIME}ms | $([ -n "$MATRIX_MULT_TIME" ] && (( $(echo "$MATRIX_MULT_TIME <= $MATRIX_THRESHOLD" | bc -l) )) && echo "✅ Good" || echo "⚠️ Needs optimization") |
| Model Loading | ${MODEL_LOAD_TIME}ms | $([ -n "$MODEL_LOAD_TIME" ] && (( $(echo "$MODEL_LOAD_TIME <= $MODEL_THRESHOLD" | bc -l) )) && echo "✅ Good" || echo "❌ Critical") |
| Inference | ${INFERENCE_TIME}ms | $([ -n "$INFERENCE_TIME" ] && (( $(echo "$INFERENCE_TIME <= $INFERENCE_THRESHOLD" | bc -l) )) && echo "✅ Good" || echo "⚠️ Needs optimization") |
| Memory Usage | ${MEMORY_USAGE}MB | $([ -n "$MEMORY_USAGE" ] && (( $(echo "$MEMORY_USAGE <= $MEMORY_THRESHOLD" | bc -l) )) && echo "✅ Good" || echo "⚠️ High") |

## 📈 Performance Targets

- 🎯 **Tensor Creation:** < ${TENSOR_THRESHOLD}ms
- 🎯 **Matrix Multiplication:** < ${MATRIX_THRESHOLD}ms  
- 🎯 **Model Loading:** < ${MODEL_THRESHOLD}ms
- 🎯 **Inference:** < ${INFERENCE_THRESHOLD}ms
- 🎯 **Memory Usage:** < ${MEMORY_THRESHOLD}MB

## 🏆 Performance Score

- ✅ **Passed:** $((5 - PERFORMANCE_ISSUES - PERFORMANCE_WARNINGS)) / 5
- ⚠️ **Warnings:** $PERFORMANCE_WARNINGS
- ❌ **Issues:** $PERFORMANCE_ISSUES

$([ "$PERFORMANCE_ISSUES" -eq 0 ] && [ "$PERFORMANCE_WARNINGS" -eq 0 ] && echo "🎉 **Excellent!** All performance targets met." || echo "🔧 **Optimization needed.** See recommendations below.")

## 💡 Optimization Recommendations

$([ "$PERFORMANCE_WARNINGS" -gt 0 ] || [ "$PERFORMANCE_ISSUES" -gt 0 ] && cat << 'RECOMMENDATIONS'
- Consider using WebGPU for compute-intensive operations
- Enable SIMD optimizations in build
- Use wasm-opt with -Oz for size optimization
- Implement caching for repeated operations
- Profile memory usage to identify leaks
RECOMMENDATIONS
)

EOF

echo ""
echo "📄 Performance summary saved to performance-summary.md"

# Final status
if [ "$PERFORMANCE_ISSUES" -gt 0 ]; then
    echo ""
    echo "❌ Performance monitoring FAILED - $PERFORMANCE_ISSUES critical issues detected"
    exit 1
elif [ "$PERFORMANCE_WARNINGS" -gt 0 ]; then
    echo ""
    echo "⚠️ Performance monitoring completed with $PERFORMANCE_WARNINGS warnings"
    exit 0
else
    echo ""
    echo "✅ Performance monitoring PASSED - All targets met!"
    exit 0
fi