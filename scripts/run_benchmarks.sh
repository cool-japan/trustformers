#!/usr/bin/env bash
# TrustformeRS Benchmark Runner Script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCHMARK_RESULTS_DIR="${PROJECT_ROOT}/benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BENCHMARK_SUITE=""
FEATURES=""
BASELINE=""
SAVE_BASELINE=""
PROFILE=false
COMPARE=false
CLEAN=false
HTML_REPORT=false

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

TrustformeRS Benchmark Runner

OPTIONS:
    -b, --bench <suite>      Run specific benchmark suite
                            (tensor_ops, model_inference, optimizer, tokenizer, 
                             quantization, memory, mobile_wasm)
    -f, --features <feat>    Enable features (gpu, distributed, mobile, wasm)
    -s, --save <name>        Save results as baseline with given name
    -c, --compare <name>     Compare against named baseline
    -p, --profile           Enable profiling during benchmarks
    -H, --html              Generate HTML report
    --clean                 Clean previous benchmark results
    -h, --help             Show this help message

EXAMPLES:
    # Run all benchmarks
    $0

    # Run specific benchmark suite
    $0 --bench tensor_ops

    # Run with GPU features and save baseline
    $0 --features gpu --save gpu_baseline

    # Compare against baseline
    $0 --compare gpu_baseline

    # Profile model inference benchmarks
    $0 --bench model_inference --profile
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--bench)
            BENCHMARK_SUITE="$2"
            shift 2
            ;;
        -f|--features)
            FEATURES="$2"
            shift 2
            ;;
        -s|--save)
            SAVE_BASELINE="$2"
            shift 2
            ;;
        -c|--compare)
            COMPARE=true
            BASELINE="$2"
            shift 2
            ;;
        -p|--profile)
            PROFILE=true
            shift
            ;;
        -H|--html)
            HTML_REPORT=true
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check dependencies
check_dependencies() {
    print_status "$BLUE" "Checking dependencies..."
    
    if ! command -v cargo &> /dev/null; then
        print_status "$RED" "Error: cargo not found. Please install Rust."
        exit 1
    fi
    
    if ! cargo criterion --version &> /dev/null 2>&1; then
        print_status "$YELLOW" "Installing cargo-criterion..."
        cargo install cargo-criterion
    fi
    
    if [[ "$PROFILE" == true ]] && ! command -v perf &> /dev/null; then
        print_status "$YELLOW" "Warning: perf not found. Profiling may be limited."
    fi
}

# Function to clean benchmark results
clean_results() {
    if [[ "$CLEAN" == true ]]; then
        print_status "$YELLOW" "Cleaning previous benchmark results..."
        rm -rf "${PROJECT_ROOT}/target/criterion"
        rm -rf "${BENCHMARK_RESULTS_DIR}"
    fi
}

# Function to create results directory
setup_results_dir() {
    mkdir -p "${BENCHMARK_RESULTS_DIR}"
    mkdir -p "${BENCHMARK_RESULTS_DIR}/reports"
    mkdir -p "${BENCHMARK_RESULTS_DIR}/baselines"
    mkdir -p "${BENCHMARK_RESULTS_DIR}/profiles"
}

# Function to run benchmarks
run_benchmarks() {
    local bench_cmd="cargo criterion"
    
    # Add specific benchmark suite if specified
    if [[ -n "$BENCHMARK_SUITE" ]]; then
        bench_cmd="$bench_cmd --bench ${BENCHMARK_SUITE}_bench"
    fi
    
    # Add features if specified
    if [[ -n "$FEATURES" ]]; then
        bench_cmd="$bench_cmd --features $FEATURES"
    fi
    
    # Add baseline comparison if specified
    if [[ "$COMPARE" == true ]]; then
        bench_cmd="$bench_cmd -- --baseline $BASELINE"
    fi
    
    # Add save baseline if specified
    if [[ -n "$SAVE_BASELINE" ]]; then
        bench_cmd="$bench_cmd -- --save-baseline $SAVE_BASELINE"
    fi
    
    # Add profiling if enabled
    if [[ "$PROFILE" == true ]]; then
        bench_cmd="$bench_cmd -- --profile-time 10"
    fi
    
    print_status "$GREEN" "Running benchmarks..."
    print_status "$BLUE" "Command: $bench_cmd"
    
    # Run the benchmarks
    cd "$PROJECT_ROOT"
    
    # Save output to file
    local output_file="${BENCHMARK_RESULTS_DIR}/benchmark_${TIMESTAMP}.log"
    
    if $bench_cmd 2>&1 | tee "$output_file"; then
        print_status "$GREEN" "Benchmarks completed successfully!"
        print_status "$BLUE" "Results saved to: $output_file"
    else
        print_status "$RED" "Benchmark execution failed!"
        exit 1
    fi
}

# Function to generate summary report
generate_summary() {
    local summary_file="${BENCHMARK_RESULTS_DIR}/reports/summary_${TIMESTAMP}.md"
    
    print_status "$BLUE" "Generating summary report..."
    
    cat > "$summary_file" << EOF
# TrustformeRS Benchmark Summary

**Date**: $(date)
**Features**: ${FEATURES:-none}
**Benchmark Suite**: ${BENCHMARK_SUITE:-all}

## Results Overview

EOF
    
    # Parse criterion output for key metrics
    if [[ -f "${PROJECT_ROOT}/target/criterion/report/index.html" ]]; then
        echo "### Performance Metrics" >> "$summary_file"
        echo "" >> "$summary_file"
        echo "See full HTML report: file://${PROJECT_ROOT}/target/criterion/report/index.html" >> "$summary_file"
    fi
    
    print_status "$GREEN" "Summary report generated: $summary_file"
}

# Function to copy HTML reports
copy_html_reports() {
    if [[ "$HTML_REPORT" == true ]] && [[ -d "${PROJECT_ROOT}/target/criterion" ]]; then
        print_status "$BLUE" "Copying HTML reports..."
        
        local html_dir="${BENCHMARK_RESULTS_DIR}/reports/html_${TIMESTAMP}"
        cp -r "${PROJECT_ROOT}/target/criterion" "$html_dir"
        
        print_status "$GREEN" "HTML reports copied to: $html_dir"
        
        # Open in browser if available
        if command -v open &> /dev/null; then
            open "$html_dir/report/index.html"
        elif command -v xdg-open &> /dev/null; then
            xdg-open "$html_dir/report/index.html"
        fi
    fi
}

# Function to run profiling analysis
run_profiling_analysis() {
    if [[ "$PROFILE" == true ]]; then
        print_status "$BLUE" "Running profiling analysis..."
        
        local profile_dir="${BENCHMARK_RESULTS_DIR}/profiles/profile_${TIMESTAMP}"
        mkdir -p "$profile_dir"
        
        # Generate flamegraph if available
        if command -v cargo-flamegraph &> /dev/null; then
            print_status "$YELLOW" "Generating flamegraph..."
            cd "$PROJECT_ROOT"
            cargo flamegraph --bench "${BENCHMARK_SUITE}_bench" -- --bench > "$profile_dir/flamegraph.svg" 2>&1 || true
        fi
        
        print_status "$GREEN" "Profiling results saved to: $profile_dir"
    fi
}

# Function to compare results
compare_results() {
    if [[ "$COMPARE" == true ]]; then
        print_status "$BLUE" "Comparing results against baseline: $BASELINE"
        
        local comparison_file="${BENCHMARK_RESULTS_DIR}/reports/comparison_${TIMESTAMP}.md"
        
        cat > "$comparison_file" << EOF
# Benchmark Comparison Report

**Date**: $(date)
**Baseline**: $BASELINE
**Current Run**: ${TIMESTAMP}

## Performance Changes

EOF
        
        # Parse and compare criterion results
        local criterion_dir="${PROJECT_ROOT}/target/criterion"
        local baseline_dir="${criterion_dir}/${BASELINE}"
        local current_dir="${criterion_dir}/current"
        
        if [[ -d "$criterion_dir" ]]; then
            print_status "$YELLOW" "Parsing criterion results..."
            
            # Find all benchmark results
            find "$criterion_dir" -name "raw.csv" -type f | while read -r csv_file; do
                local bench_name=$(echo "$csv_file" | sed "s|$criterion_dir/||" | sed 's|/raw.csv||' | head -1)
                
                if [[ -f "$csv_file" ]] && [[ "$bench_name" != "$BASELINE" ]]; then
                    # Extract benchmark performance data
                    local current_mean=$(tail -n +2 "$csv_file" | cut -d',' -f4 | head -1)
                    local current_stddev=$(tail -n +2 "$csv_file" | cut -d',' -f5 | head -1)
                    
                    # Look for baseline comparison
                    local baseline_csv="${baseline_dir}/${bench_name##*/}/raw.csv"
                    
                    cat >> "$comparison_file" << EOF
### $bench_name

EOF
                    
                    if [[ -f "$baseline_csv" ]]; then
                        local baseline_mean=$(tail -n +2 "$baseline_csv" | cut -d',' -f4 | head -1)
                        local baseline_stddev=$(tail -n +2 "$baseline_csv" | cut -d',' -f5 | head -1)
                        
                        # Calculate percentage change
                        local change=$(echo "scale=2; (($current_mean - $baseline_mean) / $baseline_mean) * 100" | bc -l 2>/dev/null || echo "N/A")
                        
                        # Determine if this is an improvement or regression
                        local status_icon="📊"
                        if [[ "$change" != "N/A" ]]; then
                            if (( $(echo "$change < -5" | bc -l 2>/dev/null || echo 0) )); then
                                status_icon="🚀" # Significant improvement
                            elif (( $(echo "$change > 5" | bc -l 2>/dev/null || echo 0) )); then
                                status_icon="⚠️"  # Regression
                            fi
                        fi
                        
                        cat >> "$comparison_file" << EOF
- **Current**: ${current_mean}ns (±${current_stddev}ns)
- **Baseline**: ${baseline_mean}ns (±${baseline_stddev}ns)
- **Change**: ${change}% ${status_icon}

EOF
                    else
                        cat >> "$comparison_file" << EOF
- **Current**: ${current_mean}ns (±${current_stddev}ns)
- **Baseline**: Not available
- **Status**: New benchmark 🆕

EOF
                    fi
                fi
            done
            
            # Add summary statistics
            cat >> "$comparison_file" << EOF

## Summary

**Total Benchmarks**: $(find "$criterion_dir" -name "raw.csv" -not -path "*/$BASELINE/*" | wc -l)
**Baseline Directory**: $baseline_dir
**Results Directory**: $criterion_dir

Generated by TrustformeRS benchmark runner on $(date)
EOF
            
        else
            cat >> "$comparison_file" << EOF
**Status**: No criterion results found in target/criterion

This might be because:
- Benchmarks haven't been run yet
- Criterion output directory was cleaned
- Benchmarks failed to complete

Please run benchmarks first with: \`./scripts/run_benchmarks.sh\`
EOF
        fi
        
        print_status "$GREEN" "Comparison report generated: $comparison_file"
    fi
}

# Main execution
main() {
    print_status "$GREEN" "=== TrustformeRS Benchmark Runner ==="
    
    check_dependencies
    clean_results
    setup_results_dir
    run_benchmarks
    generate_summary
    copy_html_reports
    run_profiling_analysis
    compare_results
    
    print_status "$GREEN" "=== Benchmark run completed ==="
    print_status "$BLUE" "Results directory: ${BENCHMARK_RESULTS_DIR}"
}

# Run main function
main