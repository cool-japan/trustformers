#!/bin/bash

# Memory Pressure Performance Regression Test Runner
# 
# This script runs memory pressure cleanup handler performance regression tests
# and provides various modes for CI/CD integration.
#
# Usage:
#   ./run_memory_pressure_regression.sh [OPTIONS]
#
# Options:
#   --mode <baseline|check|report>   Test mode (default: check)
#   --baseline-file <path>           Path to baseline file (default: /tmp/memory_pressure_baselines.json)
#   --output-format <json|html|ci>   Output format (default: ci)
#   --fail-on-regression             Fail if regression is detected
#   --max-degradation <percent>      Maximum allowed performance degradation (default: 15)
#   --verbose                        Enable verbose output
#   --help                          Show this help message

set -euo pipefail

# Default configuration
MODE="check"
BASELINE_FILE="/tmp/memory_pressure_baselines.json"
OUTPUT_FORMAT="ci"
FAIL_ON_REGRESSION="false"
MAX_DEGRADATION="15.0"
VERBOSE="false"
CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-target}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Show help message
show_help() {
    cat << EOF
Memory Pressure Performance Regression Test Runner

This script runs comprehensive performance regression tests for memory pressure
cleanup handlers in TrustformeRS.

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --mode <baseline|check|report>   Test mode
                                     baseline: Record new performance baselines
                                     check: Check for regressions against baselines
                                     report: Generate performance report only
                                     (default: check)

    --baseline-file <path>           Path to baseline file 
                                     (default: /tmp/memory_pressure_baselines.json)

    --output-format <json|html|ci>   Output format
                                     json: Machine-readable JSON output
                                     html: HTML report with charts
                                     ci: CI-friendly text output
                                     (default: ci)

    --fail-on-regression             Exit with error code if regression detected
                                     Useful for CI/CD pipelines

    --max-degradation <percent>      Maximum allowed performance degradation
                                     (default: 15.0)

    --verbose                        Enable verbose output and detailed logs

    --help                          Show this help message

EXAMPLES:
    # Record new baselines
    $0 --mode baseline --verbose

    # Check for regressions in CI
    $0 --mode check --fail-on-regression --output-format ci

    # Generate HTML performance report
    $0 --mode report --output-format html

    # Custom regression threshold
    $0 --mode check --max-degradation 10.0 --fail-on-regression

ENVIRONMENT VARIABLES:
    BASELINE_FILE               Override baseline file path
    RECORD_BASELINE            Set to '1' to record baselines
    CHECK_REGRESSION           Set to '1' to check for regressions
    FAIL_ON_REGRESSION         Set to '1' to fail on regression detection
    MAX_DEGRADATION_PERCENT    Maximum allowed degradation percentage
    CARGO_TARGET_DIR           Cargo target directory (default: target)

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --mode)
                MODE="$2"
                shift 2
                ;;
            --baseline-file)
                BASELINE_FILE="$2"
                shift 2
                ;;
            --output-format)
                OUTPUT_FORMAT="$2"
                shift 2
                ;;
            --fail-on-regression)
                FAIL_ON_REGRESSION="true"
                shift
                ;;
            --max-degradation)
                MAX_DEGRADATION="$2"
                shift 2
                ;;
            --verbose)
                VERBOSE="true"
                shift
                ;;
            --help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Validate arguments
validate_args() {
    if [[ ! "$MODE" =~ ^(baseline|check|report)$ ]]; then
        log_error "Invalid mode: $MODE. Must be one of: baseline, check, report"
        exit 1
    fi

    if [[ ! "$OUTPUT_FORMAT" =~ ^(json|html|ci)$ ]]; then
        log_error "Invalid output format: $OUTPUT_FORMAT. Must be one of: json, html, ci"
        exit 1
    fi

    if ! command -v cargo >/dev/null 2>&1; then
        log_error "cargo command not found. Please install Rust and Cargo."
        exit 1
    fi

    # Check if we're in the correct directory
    if [[ ! -f "Cargo.toml" ]]; then
        log_error "Cargo.toml not found. Please run this script from the project root."
        exit 1
    fi
}

# Setup environment variables for the benchmark
setup_environment() {
    export BASELINE_FILE="$BASELINE_FILE"
    export MAX_DEGRADATION_PERCENT="$MAX_DEGRADATION"
    
    case "$MODE" in
        baseline)
            export RECORD_BASELINE="1"
            unset CHECK_REGRESSION
            ;;
        check)
            unset RECORD_BASELINE
            export CHECK_REGRESSION="1"
            ;;
        report)
            unset RECORD_BASELINE
            unset CHECK_REGRESSION
            ;;
    esac

    if [[ "$FAIL_ON_REGRESSION" == "true" ]]; then
        export FAIL_ON_REGRESSION="1"
    fi

    if [[ "$VERBOSE" == "true" ]]; then
        export RUST_LOG="debug"
    fi
}

# Ensure baseline directory exists
ensure_baseline_directory() {
    local baseline_dir
    baseline_dir=$(dirname "$BASELINE_FILE")
    
    if [[ ! -d "$baseline_dir" ]]; then
        log_info "Creating baseline directory: $baseline_dir"
        mkdir -p "$baseline_dir"
    fi
}

# Run the benchmark
run_benchmark() {
    log_info "Running memory pressure regression tests in $MODE mode..."
    
    local benchmark_args=(
        "bench"
        "--bench" "memory_pressure_regression"
    )

    if [[ "$VERBOSE" == "true" ]]; then
        benchmark_args+=("--" "--verbose")
    fi

    # Add output options based on format
    case "$OUTPUT_FORMAT" in
        json)
            benchmark_args+=("--" "--output-format" "json")
            ;;
        html)
            benchmark_args+=("--" "--output-format" "html")
            ;;
    esac

    log_info "Executing: cargo ${benchmark_args[*]}"
    
    if ! cargo "${benchmark_args[@]}"; then
        if [[ "$MODE" == "check" && "$FAIL_ON_REGRESSION" == "true" ]]; then
            log_error "Performance regression detected and fail-on-regression is enabled"
            exit 1
        else
            log_warn "Benchmark execution completed with warnings"
        fi
    fi
}

# Generate performance report
generate_report() {
    if [[ "$OUTPUT_FORMAT" == "ci" ]]; then
        generate_ci_report
    elif [[ "$OUTPUT_FORMAT" == "json" ]]; then
        generate_json_report
    elif [[ "$OUTPUT_FORMAT" == "html" ]]; then
        generate_html_report
    fi
}

# Generate CI-friendly report
generate_ci_report() {
    log_info "Generating CI performance report..."
    
    if [[ -f "$BASELINE_FILE" ]]; then
        local baseline_count
        baseline_count=$(jq '. | length' "$BASELINE_FILE" 2>/dev/null || echo "0")
        log_info "Found $baseline_count performance baselines"
        
        if [[ "$MODE" == "check" ]]; then
            log_info "Regression check completed"
            if [[ -f "${CARGO_TARGET_DIR}/criterion/memory_pressure_regression" ]]; then
                log_info "Detailed benchmark results available in: ${CARGO_TARGET_DIR}/criterion/"
            fi
        fi
    else
        log_warn "No baseline file found at: $BASELINE_FILE"
        if [[ "$MODE" == "check" ]]; then
            log_warn "Cannot perform regression check without baselines. Run with --mode baseline first."
        fi
    fi
}

# Generate JSON report
generate_json_report() {
    local report_file="/tmp/memory_pressure_report.json"
    log_info "Generating JSON performance report: $report_file"
    
    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    cat > "$report_file" << EOF
{
  "timestamp": "$timestamp",
  "mode": "$MODE",
  "baseline_file": "$BASELINE_FILE",
  "max_degradation_percent": $MAX_DEGRADATION,
  "fail_on_regression": $FAIL_ON_REGRESSION,
  "baselines_found": $(if [[ -f "$BASELINE_FILE" ]]; then jq '. | length' "$BASELINE_FILE" 2>/dev/null || echo "0"; else echo "0"; fi),
  "benchmark_results": "$(find "${CARGO_TARGET_DIR}/criterion" -name "*.json" -type f | head -5 | tr '\n' ',' | sed 's/,$//')"
}
EOF

    log_success "JSON report generated: $report_file"
}

# Generate HTML report
generate_html_report() {
    local report_file="/tmp/memory_pressure_report.html"
    log_info "Generating HTML performance report: $report_file"
    
    cat > "$report_file" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Memory Pressure Performance Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #f5f5f5; padding: 20px; border-radius: 5px; }
        .metric { margin: 20px 0; }
        .success { color: #28a745; }
        .warning { color: #ffc107; }
        .error { color: #dc3545; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Memory Pressure Performance Report</h1>
        <p><strong>Generated:</strong> <span id="timestamp"></span></p>
        <p><strong>Mode:</strong> MODE_PLACEHOLDER</p>
        <p><strong>Baseline File:</strong> BASELINE_FILE_PLACEHOLDER</p>
    </div>

    <div class="metric">
        <h2>Test Summary</h2>
        <p>Performance regression tests for TrustformeRS memory pressure cleanup handlers.</p>
        <p>For detailed benchmark results, see the Criterion reports in the target directory.</p>
    </div>

    <div class="metric">
        <h2>Cleanup Handlers Tested</h2>
        <table>
            <tr><th>Handler</th><th>Function</th><th>Expected Memory Freed</th></tr>
            <tr><td>GarbageCollectionHandler</td><td>Force garbage collection</td><td>10-20 MB</td></tr>
            <tr><td>BufferCompactionHandler</td><td>Compact memory buffers</td><td>5-15 MB</td></tr>
            <tr><td>GPU Cache Eviction</td><td>Evict GPU caches</td><td>50 MB</td></tr>
            <tr><td>GPU Buffer Compaction</td><td>Compact GPU buffers</td><td>30 MB</td></tr>
            <tr><td>GPU Model Unloading</td><td>Unload models from GPU</td><td>200 MB</td></tr>
            <tr><td>GPU VRAM Compaction</td><td>Compact VRAM</td><td>80 MB</td></tr>
        </table>
    </div>

    <script>
        document.getElementById('timestamp').textContent = new Date().toISOString();
    </script>
</body>
</html>
EOF

    # Replace placeholders
    sed -i.bak "s/MODE_PLACEHOLDER/$MODE/g" "$report_file"
    sed -i.bak "s|BASELINE_FILE_PLACEHOLDER|$BASELINE_FILE|g" "$report_file"
    rm -f "${report_file}.bak"

    log_success "HTML report generated: $report_file"
}

# Check system requirements
check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Rust version
    if command -v rustc >/dev/null 2>&1; then
        local rust_version
        rust_version=$(rustc --version)
        log_info "Rust version: $rust_version"
    else
        log_error "Rust compiler not found"
        exit 1
    fi

    # Check available memory
    if command -v free >/dev/null 2>&1; then
        local available_memory
        available_memory=$(free -h | awk '/^Mem:/ {print $7}')
        log_info "Available memory: $available_memory"
    elif command -v vm_stat >/dev/null 2>&1; then
        # macOS
        log_info "Running on macOS, memory information available via vm_stat"
    fi

    # Check disk space
    local disk_usage
    disk_usage=$(df -h . | awk 'NR==2 {print $4}')
    log_info "Available disk space: $disk_usage"
}

# Cleanup temporary files
cleanup() {
    if [[ "$VERBOSE" == "true" ]]; then
        log_info "Cleaning up temporary files..."
    fi
    
    # Criterion generates lots of temporary files, but we'll keep them for analysis
    # Only clean up our specific temporary files if needed
    
    if [[ "$OUTPUT_FORMAT" == "json" && -f "/tmp/memory_pressure_report.json" ]]; then
        log_info "JSON report saved to: /tmp/memory_pressure_report.json"
    fi
    
    if [[ "$OUTPUT_FORMAT" == "html" && -f "/tmp/memory_pressure_report.html" ]]; then
        log_info "HTML report saved to: /tmp/memory_pressure_report.html"
    fi
}

# Main execution
main() {
    parse_args "$@"
    validate_args
    check_requirements
    setup_environment
    ensure_baseline_directory
    
    log_info "Starting memory pressure performance regression tests"
    log_info "Mode: $MODE"
    log_info "Baseline file: $BASELINE_FILE"
    log_info "Output format: $OUTPUT_FORMAT"
    log_info "Max degradation: $MAX_DEGRADATION%"
    
    run_benchmark
    generate_report
    cleanup
    
    log_success "Memory pressure regression tests completed successfully"
}

# Trap for cleanup on exit
trap cleanup EXIT

# Run main function with all arguments
main "$@"