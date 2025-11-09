#!/bin/bash
# Cross-platform test runner for TrustformeRS-C
# Supports Linux, macOS, and Windows (via WSL/MSYS2)

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_OUTPUT_DIR="${PROJECT_ROOT}/test-results"
COVERAGE_OUTPUT_DIR="${PROJECT_ROOT}/coverage"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Platform detection
PLATFORM="unknown"
case "$(uname -s)" in
    Linux*)     PLATFORM="linux";;
    Darwin*)    PLATFORM="macos";;
    CYGWIN*|MINGW*|MSYS*) PLATFORM="windows";;
    *)          PLATFORM="unknown";;
esac

# Default test configuration
RUN_UNIT_TESTS=true
RUN_INTEGRATION_TESTS=true
RUN_PROPERTY_TESTS=true
RUN_PERFORMANCE_TESTS=false
RUN_MEMORY_TESTS=true
ENABLE_COVERAGE=false
BUILD_RELEASE=false
PARALLEL_JOBS=4
TEST_TIMEOUT=300
FEATURES="default"

# Help function
show_help() {
    cat << EOF
TrustformeRS-C Cross-Platform Test Runner

Usage: $0 [OPTIONS]

Options:
    -h, --help              Show this help message
    -p, --platform PLATFORM Platform to test (linux|macos|windows|all)
    -u, --unit              Run unit tests only
    -i, --integration       Run integration tests only
    -r, --property          Run property-based tests only
    -m, --memory            Run memory safety tests only
    -P, --performance       Run performance tests
    -c, --coverage          Enable code coverage
    -R, --release           Build in release mode
    -j, --jobs N            Number of parallel jobs (default: $PARALLEL_JOBS)
    -t, --timeout N         Test timeout in seconds (default: $TEST_TIMEOUT)
    -f, --features FEATURES Comma-separated list of features to enable
    -o, --output DIR        Output directory for test results
    --clean                 Clean build artifacts before testing
    --verbose               Verbose output
    --ci                    CI mode (non-interactive)

Examples:
    $0                      Run all tests with default configuration
    $0 -u -c               Run unit tests with coverage
    $0 -P -R               Run performance tests in release mode
    $0 -f "cuda,serving"   Run tests with CUDA and serving features
    $0 --clean --ci        Clean build and run in CI mode

Supported platforms: linux, macos, windows
Current platform: $PLATFORM
EOF
}

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -p|--platform)
                PLATFORM="$2"
                shift 2
                ;;
            -u|--unit)
                RUN_UNIT_TESTS=true
                RUN_INTEGRATION_TESTS=false
                RUN_PROPERTY_TESTS=false
                RUN_PERFORMANCE_TESTS=false
                RUN_MEMORY_TESTS=false
                shift
                ;;
            -i|--integration)
                RUN_UNIT_TESTS=false
                RUN_INTEGRATION_TESTS=true
                RUN_PROPERTY_TESTS=false
                RUN_PERFORMANCE_TESTS=false
                RUN_MEMORY_TESTS=false
                shift
                ;;
            -r|--property)
                RUN_UNIT_TESTS=false
                RUN_INTEGRATION_TESTS=false
                RUN_PROPERTY_TESTS=true
                RUN_PERFORMANCE_TESTS=false
                RUN_MEMORY_TESTS=false
                shift
                ;;
            -m|--memory)
                RUN_UNIT_TESTS=false
                RUN_INTEGRATION_TESTS=false
                RUN_PROPERTY_TESTS=false
                RUN_PERFORMANCE_TESTS=false
                RUN_MEMORY_TESTS=true
                shift
                ;;
            -P|--performance)
                RUN_PERFORMANCE_TESTS=true
                shift
                ;;
            -c|--coverage)
                ENABLE_COVERAGE=true
                shift
                ;;
            -R|--release)
                BUILD_RELEASE=true
                shift
                ;;
            -j|--jobs)
                PARALLEL_JOBS="$2"
                shift 2
                ;;
            -t|--timeout)
                TEST_TIMEOUT="$2"
                shift 2
                ;;
            -f|--features)
                FEATURES="$2"
                shift 2
                ;;
            -o|--output)
                TEST_OUTPUT_DIR="$2"
                shift 2
                ;;
            --clean)
                CLEAN_BUILD=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --ci)
                CI_MODE=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Setup environment
setup_environment() {
    log_info "Setting up test environment for platform: $PLATFORM"
    
    # Create output directories
    mkdir -p "$TEST_OUTPUT_DIR"
    mkdir -p "$COVERAGE_OUTPUT_DIR"
    
    # Set environment variables
    export RUST_LOG="${RUST_LOG:-info}"
    export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
    export CARGO_TERM_PROGRESS_WHEN="${CARGO_TERM_PROGRESS_WHEN:-never}"
    
    # Platform-specific setup
    case "$PLATFORM" in
        linux)
            setup_linux_environment
            ;;
        macos)
            setup_macos_environment
            ;;
        windows)
            setup_windows_environment
            ;;
        *)
            log_warning "Unknown platform: $PLATFORM. Using default configuration."
            ;;
    esac
}

setup_linux_environment() {
    log_info "Setting up Linux environment"
    
    # Check for required tools
    check_command "rustc" "Rust compiler"
    check_command "cargo" "Cargo"
    check_command "gcc" "GCC compiler"
    
    # Set Linux-specific variables
    export CC="${CC:-gcc}"
    export CXX="${CXX:-g++}"
    
    # Install additional dependencies if needed
    if command -v apt-get >/dev/null 2>&1; then
        log_info "Detected APT package manager"
        # Would install dependencies here in a real setup
    elif command -v yum >/dev/null 2>&1; then
        log_info "Detected YUM package manager"
        # Would install dependencies here in a real setup
    fi
}

setup_macos_environment() {
    log_info "Setting up macOS environment"
    
    # Check for required tools
    check_command "rustc" "Rust compiler"
    check_command "cargo" "Cargo"
    check_command "clang" "Clang compiler"
    
    # Set macOS-specific variables
    export CC="${CC:-clang}"
    export CXX="${CXX:-clang++}"
    
    # Check for Xcode command line tools
    if ! xcode-select -p >/dev/null 2>&1; then
        log_warning "Xcode command line tools not found. Some tests may fail."
    fi
}

setup_windows_environment() {
    log_info "Setting up Windows environment"
    
    # Check for required tools
    check_command "rustc" "Rust compiler"
    check_command "cargo" "Cargo"
    
    # Set Windows-specific variables
    if command -v cl.exe >/dev/null 2>&1; then
        export CC="${CC:-cl.exe}"
        export CXX="${CXX:-cl.exe}"
        log_info "Using MSVC compiler"
    elif command -v gcc >/dev/null 2>&1; then
        export CC="${CC:-gcc}"
        export CXX="${CXX:-g++}"
        log_info "Using MinGW compiler"
    else
        log_warning "No suitable C compiler found"
    fi
}

check_command() {
    local cmd="$1"
    local description="$2"
    
    if ! command -v "$cmd" >/dev/null 2>&1; then
        log_error "$description ($cmd) not found"
        exit 1
    fi
}

# Build functions
clean_build() {
    if [[ "${CLEAN_BUILD:-false}" == "true" ]]; then
        log_info "Cleaning build artifacts"
        cargo clean
        rm -rf "$TEST_OUTPUT_DIR"
        rm -rf "$COVERAGE_OUTPUT_DIR"
        mkdir -p "$TEST_OUTPUT_DIR"
        mkdir -p "$COVERAGE_OUTPUT_DIR"
    fi
}

build_project() {
    log_info "Building project with features: $FEATURES"
    
    local build_args=()
    build_args+=("--features" "$FEATURES")
    
    if [[ "$BUILD_RELEASE" == "true" ]]; then
        build_args+=("--release")
        log_info "Building in release mode"
    fi
    
    if [[ "${VERBOSE:-false}" == "true" ]]; then
        build_args+=("--verbose")
    fi
    
    cargo build "${build_args[@]}"
    log_success "Project built successfully"
}

# Test execution functions
run_unit_tests() {
    if [[ "$RUN_UNIT_TESTS" != "true" ]]; then
        return 0
    fi
    
    log_info "Running unit tests"
    
    local test_args=()
    test_args+=("--features" "$FEATURES")
    test_args+=("--lib")
    
    if [[ "$BUILD_RELEASE" == "true" ]]; then
        test_args+=("--release")
    fi
    
    if [[ "${VERBOSE:-false}" == "true" ]]; then
        test_args+=("--verbose")
    fi
    
    if [[ "$ENABLE_COVERAGE" == "true" ]]; then
        setup_coverage
        test_args+=("--target-dir" "target/coverage")
    fi
    
    # Add timeout
    timeout "$TEST_TIMEOUT" cargo test "${test_args[@]}" --jobs="$PARALLEL_JOBS" 2>&1 | tee "$TEST_OUTPUT_DIR/unit_tests.log"
    
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        log_success "Unit tests passed"
    else
        log_error "Unit tests failed"
        return 1
    fi
}

run_integration_tests() {
    if [[ "$RUN_INTEGRATION_TESTS" != "true" ]]; then
        return 0
    fi
    
    log_info "Running integration tests"
    
    local test_args=()
    test_args+=("--features" "$FEATURES")
    test_args+=("--test" "integration_tests")
    
    if [[ "$BUILD_RELEASE" == "true" ]]; then
        test_args+=("--release")
    fi
    
    timeout "$TEST_TIMEOUT" cargo test "${test_args[@]}" --jobs="$PARALLEL_JOBS" 2>&1 | tee "$TEST_OUTPUT_DIR/integration_tests.log"
    
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        log_success "Integration tests passed"
    else
        log_error "Integration tests failed"
        return 1
    fi
}

run_property_tests() {
    if [[ "$RUN_PROPERTY_TESTS" != "true" ]]; then
        return 0
    fi
    
    log_info "Running property-based tests"
    
    local test_args=()
    test_args+=("--features" "$FEATURES")
    test_args+=("--test" "property_based_tests")
    
    if [[ "$BUILD_RELEASE" == "true" ]]; then
        test_args+=("--release")
    fi
    
    timeout "$TEST_TIMEOUT" cargo test "${test_args[@]}" --jobs="$PARALLEL_JOBS" 2>&1 | tee "$TEST_OUTPUT_DIR/property_tests.log"
    
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        log_success "Property-based tests passed"
    else
        log_error "Property-based tests failed"
        return 1
    fi
}

run_performance_tests() {
    if [[ "$RUN_PERFORMANCE_TESTS" != "true" ]]; then
        return 0
    fi
    
    log_info "Running performance tests"
    
    local test_args=()
    test_args+=("--features" "$FEATURES")
    test_args+=("--test" "performance_regression_tests")
    test_args+=("--release") # Always run performance tests in release mode
    
    timeout "$TEST_TIMEOUT" cargo test "${test_args[@]}" --jobs="$PARALLEL_JOBS" 2>&1 | tee "$TEST_OUTPUT_DIR/performance_tests.log"
    
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        log_success "Performance tests passed"
    else
        log_error "Performance tests failed"
        return 1
    fi
}

run_memory_tests() {
    if [[ "$RUN_MEMORY_TESTS" != "true" ]]; then
        return 0
    fi
    
    log_info "Running memory safety tests"
    
    # Run with AddressSanitizer if available
    if [[ "$PLATFORM" == "linux" ]] || [[ "$PLATFORM" == "macos" ]]; then
        run_asan_tests
    fi
    
    # Run Valgrind tests on Linux
    if [[ "$PLATFORM" == "linux" ]] && command -v valgrind >/dev/null 2>&1; then
        run_valgrind_tests
    fi
    
    # Run built-in memory safety tests
    local test_args=()
    test_args+=("--features" "$FEATURES")
    test_args+=("--lib")
    test_args+=("memory_safety")
    
    if [[ "$BUILD_RELEASE" == "true" ]]; then
        test_args+=("--release")
    fi
    
    timeout "$TEST_TIMEOUT" cargo test "${test_args[@]}" 2>&1 | tee "$TEST_OUTPUT_DIR/memory_tests.log"
    
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        log_success "Memory safety tests passed"
    else
        log_error "Memory safety tests failed"
        return 1
    fi
}

run_asan_tests() {
    log_info "Running AddressSanitizer tests"
    
    export RUSTFLAGS="-Zsanitizer=address"
    export ASAN_OPTIONS="detect_leaks=1:abort_on_error=1"
    
    cargo +nightly test --features "$FEATURES" --target-dir target/asan 2>&1 | tee "$TEST_OUTPUT_DIR/asan_tests.log"
    
    unset RUSTFLAGS ASAN_OPTIONS
    
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        log_success "AddressSanitizer tests passed"
    else
        log_warning "AddressSanitizer tests failed or detected issues"
    fi
}

run_valgrind_tests() {
    log_info "Running Valgrind tests"
    
    # Build the test binary
    cargo build --tests --features "$FEATURES"
    
    # Find the test binary
    local test_binary
    test_binary=$(find target/debug/deps -name "*trustformers*" -executable | head -1)
    
    if [[ -n "$test_binary" ]]; then
        valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --track-origins=yes \
            "$test_binary" --test-threads=1 2>&1 | tee "$TEST_OUTPUT_DIR/valgrind_tests.log"
        
        if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
            log_success "Valgrind tests completed"
        else
            log_warning "Valgrind tests detected issues"
        fi
    else
        log_warning "Could not find test binary for Valgrind"
    fi
}

setup_coverage() {
    log_info "Setting up code coverage"
    
    # Install cargo-tarpaulin if not present
    if ! command -v cargo-tarpaulin >/dev/null 2>&1; then
        log_info "Installing cargo-tarpaulin"
        cargo install cargo-tarpaulin
    fi
    
    export RUSTFLAGS="-C instrument-coverage"
}

generate_coverage_report() {
    if [[ "$ENABLE_COVERAGE" != "true" ]]; then
        return 0
    fi
    
    log_info "Generating coverage report"
    
    cargo tarpaulin --features "$FEATURES" --out Html --output-dir "$COVERAGE_OUTPUT_DIR" \
        --timeout "$TEST_TIMEOUT" 2>&1 | tee "$TEST_OUTPUT_DIR/coverage.log"
    
    if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
        log_success "Coverage report generated: $COVERAGE_OUTPUT_DIR/tarpaulin-report.html"
    else
        log_error "Failed to generate coverage report"
    fi
}

# Language bindings tests
test_language_bindings() {
    log_info "Testing language bindings"
    
    # Test Go bindings
    test_go_bindings
    
    # Test Python bindings
    test_python_bindings
    
    # Test Node.js bindings
    test_nodejs_bindings
}

test_go_bindings() {
    if [[ -d "golang" ]] && command -v go >/dev/null 2>&1; then
        log_info "Testing Go bindings"
        (
            cd golang
            go mod tidy
            go test ./... 2>&1 | tee "$TEST_OUTPUT_DIR/go_tests.log"
            
            if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
                log_success "Go bindings tests passed"
            else
                log_error "Go bindings tests failed"
            fi
        )
    else
        log_warning "Go bindings not available or Go not installed"
    fi
}

test_python_bindings() {
    if [[ -d "python" ]] && command -v python3 >/dev/null 2>&1; then
        log_info "Testing Python bindings"
        (
            cd python
            python3 -m pytest tests/ -v 2>&1 | tee "$TEST_OUTPUT_DIR/python_tests.log"
            
            if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
                log_success "Python bindings tests passed"
            else
                log_error "Python bindings tests failed"
            fi
        )
    else
        log_warning "Python bindings not available or Python not installed"
    fi
}

test_nodejs_bindings() {
    if [[ -d "nodejs" ]] && command -v npm >/dev/null 2>&1; then
        log_info "Testing Node.js bindings"
        (
            cd nodejs
            npm test 2>&1 | tee "$TEST_OUTPUT_DIR/nodejs_tests.log"
            
            if [[ ${PIPESTATUS[0]} -eq 0 ]]; then
                log_success "Node.js bindings tests passed"
            else
                log_error "Node.js bindings tests failed"
            fi
        )
    else
        log_warning "Node.js bindings not available or Node.js not installed"
    fi
}

# Report generation
generate_test_report() {
    log_info "Generating test report"
    
    local report_file="$TEST_OUTPUT_DIR/test_report.html"
    
    cat > "$report_file" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>TrustformeRS-C Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 10px; }
        .success { color: green; }
        .error { color: red; }
        .warning { color: orange; }
        .section { margin: 20px 0; }
        pre { background-color: #f5f5f5; padding: 10px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>TrustformeRS-C Test Report</h1>
        <p>Generated on: $(date)</p>
        <p>Platform: $PLATFORM</p>
        <p>Features: $FEATURES</p>
    </div>
EOF

    # Add test results
    for test_type in unit integration property performance memory; do
        local log_file="$TEST_OUTPUT_DIR/${test_type}_tests.log"
        if [[ -f "$log_file" ]]; then
            echo "<div class=\"section\">" >> "$report_file"
            echo "<h2>$(echo $test_type | tr '[:lower:]' '[:upper:]') Tests</h2>" >> "$report_file"
            echo "<pre>" >> "$report_file"
            cat "$log_file" >> "$report_file"
            echo "</pre>" >> "$report_file"
            echo "</div>" >> "$report_file"
        fi
    done

    echo "</body></html>" >> "$report_file"
    
    log_success "Test report generated: $report_file"
}

# Main execution
main() {
    local start_time
    start_time=$(date +%s)
    
    log_info "Starting TrustformeRS-C test runner"
    log_info "Platform: $PLATFORM"
    log_info "Test output directory: $TEST_OUTPUT_DIR"
    
    # Parse arguments
    parse_args "$@"
    
    # Setup environment
    setup_environment
    
    # Clean build if requested
    clean_build
    
    # Build project
    build_project
    
    # Run tests
    local test_results=0
    
    run_unit_tests || ((test_results++))
    run_integration_tests || ((test_results++))
    run_property_tests || ((test_results++))
    run_performance_tests || ((test_results++))
    run_memory_tests || ((test_results++))
    
    # Test language bindings
    test_language_bindings
    
    # Generate coverage report
    generate_coverage_report
    
    # Generate test report
    generate_test_report
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log_info "Test execution completed in ${duration} seconds"
    
    if [[ $test_results -eq 0 ]]; then
        log_success "All tests passed!"
        exit 0
    else
        log_error "$test_results test suite(s) failed"
        exit 1
    fi
}

# Entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi