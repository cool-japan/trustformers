#!/bin/bash
# TrustformeRS Core Performance Benchmarking Script
# 
# This script runs all benchmarks, stores results, and provides performance analysis

set -e

# Configuration
RESULTS_DIR="benchmark_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
COMMIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

echo "🚀 TrustformeRS Core Benchmarking Suite"
echo "======================================="
echo "Timestamp: $(date)"
echo "Commit: $COMMIT_SHA"
echo "Branch: $BRANCH"
echo "Results Dir: $RESULTS_DIR"
echo ""

# Function to run a benchmark and save results
run_benchmark() {
    local bench_name=$1
    local output_file="$RESULTS_DIR/${bench_name}_${TIMESTAMP}_${COMMIT_SHA}.json"
    
    echo "📊 Running benchmark: $bench_name"
    echo "   Output: $output_file"
    
    # Run benchmark with JSON output
    cargo bench --bench "$bench_name" -- --output-format json | tee "$output_file"
    
    # Also save a human-readable version
    local readable_file="$RESULTS_DIR/${bench_name}_${TIMESTAMP}_${COMMIT_SHA}.txt"
    cargo bench --bench "$bench_name" | tee "$readable_file"
    
    echo "✅ Completed: $bench_name"
    echo ""
}

# Function to generate benchmark report
generate_report() {
    local report_file="$RESULTS_DIR/benchmark_report_${TIMESTAMP}.md"
    
    echo "📈 Generating benchmark report: $report_file"
    
    cat > "$report_file" << EOF
# TrustformeRS Core Benchmark Report

**Generated:** $(date)  
**Commit:** $COMMIT_SHA  
**Branch:** $BRANCH  
**Rust Version:** $(rustc --version)  

## System Information
- **OS:** $(uname -s) $(uname -r)
- **CPU:** $(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs || echo "Unknown")
- **Memory:** $(free -h | grep Mem | awk '{print $2}' || echo "Unknown")

## Benchmark Results

### Tensor Operations
Results saved to: \`tensor_operations_${TIMESTAMP}_${COMMIT_SHA}.json\`

### Hardware Acceleration  
Results saved to: \`hardware_acceleration_${TIMESTAMP}_${COMMIT_SHA}.json\`

## Performance Analysis

To compare with previous results:
\`\`\`bash
# Compare with previous commit
./compare_benchmarks.sh $RESULTS_DIR/tensor_operations_*.json

# View performance trends
./analyze_performance_trends.sh $RESULTS_DIR
\`\`\`

## Next Steps
1. Review results for any performance regressions
2. Compare against baseline performance targets
3. Investigate any significant performance changes
4. Update performance documentation if needed

EOF

    echo "✅ Report generated: $report_file"
}

# Main execution
echo "🔧 Building project in release mode..."
cargo build --release

echo ""
echo "⚡ Running benchmarks..."

# Run all benchmarks
run_benchmark "tensor_operations"
run_benchmark "hardware_acceleration"

# Generate comprehensive report
generate_report

echo "🎉 Benchmarking complete!"
echo ""
echo "📁 Results stored in: $RESULTS_DIR/"
echo "📊 Latest results:"
ls -la "$RESULTS_DIR/" | tail -5

echo ""
echo "💡 Next steps:"
echo "   - Review results for performance regressions"
echo "   - Compare with previous benchmarks" 
echo "   - Check hardware acceleration performance gains"
echo "   - Update performance baselines if needed"