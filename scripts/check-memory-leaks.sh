#!/bin/bash
# Local memory leak detection script for TrustformeRS
# Usage: ./scripts/check-memory-leaks.sh [options]

set -e

# Default configuration
VALGRIND_ENABLED=true
HEAPTRACK_ENABLED=true
VERBOSE=false
PACKAGE="trustformers-core"
TEST_FILTER="memory_leak"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-valgrind)
            VALGRIND_ENABLED=false
            shift
            ;;
        --no-heaptrack)
            HEAPTRACK_ENABLED=false
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --package|-p)
            PACKAGE="$2"
            shift 2
            ;;
        --filter|-f)
            TEST_FILTER="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --no-valgrind    Disable Valgrind checking"
            echo "  --no-heaptrack   Disable heaptrack profiling"
            echo "  --verbose, -v    Enable verbose output"
            echo "  --package, -p    Specify package to test (default: trustformers-core)"
            echo "  --filter, -f     Test filter pattern (default: memory_leak)"
            echo "  --help, -h       Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    if [[ $VERBOSE == true ]]; then
        echo -e "${BLUE}[INFO]${NC} $1"
    fi
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if required tools are installed
check_dependencies() {
    log "Checking dependencies..."
    
    if ! command -v cargo &> /dev/null; then
        error "cargo is not installed. Please install Rust and Cargo."
        exit 1
    fi
    
    if [[ $VALGRIND_ENABLED == true ]] && ! command -v valgrind &> /dev/null; then
        warning "valgrind is not installed. Skipping Valgrind checks."
        warning "Install with: sudo apt-get install valgrind (Ubuntu/Debian) or brew install valgrind (macOS)"
        VALGRIND_ENABLED=false
    fi
    
    if [[ $HEAPTRACK_ENABLED == true ]] && ! command -v heaptrack &> /dev/null; then
        warning "heaptrack is not installed. Skipping heaptrack profiling."
        warning "Install with: sudo apt-get install heaptrack (Ubuntu/Debian)"
        HEAPTRACK_ENABLED=false
    fi
}

# Create suppression file for Valgrind
create_valgrind_suppressions() {
    cat > valgrind.supp << 'EOF'
{
   rust_std_alloc
   Memcheck:Leak
   match-leak-kinds: reachable
   fun:malloc
   ...
   fun:rust_begin_unwind
}
{
   rust_thread_local
   Memcheck:Leak
   match-leak-kinds: reachable
   fun:calloc
   ...
   obj:*/libstd-*
}
{
   rust_hashmap
   Memcheck:Leak
   match-leak-kinds: reachable
   fun:malloc
   ...
   fun:*hashmap*
}
EOF
}

# Run basic memory leak tests
run_basic_tests() {
    log "Running basic memory leak tests..."
    
    export RUSTFLAGS="-C debug-assertions=on -C overflow-checks=on"
    export RUST_LOG=debug
    export MEMORY_LEAK_DETECTION=true
    
    echo "Running memory leak detection tests..."
    if cargo test --package "$PACKAGE" --lib "$TEST_FILTER" -- --nocapture; then
        success "Basic memory leak tests passed"
        return 0
    else
        error "Basic memory leak tests failed"
        return 1
    fi
}

# Run Valgrind analysis
run_valgrind_analysis() {
    if [[ $VALGRIND_ENABLED == false ]]; then
        log "Skipping Valgrind analysis (disabled)"
        return 0
    fi
    
    log "Running Valgrind memory leak analysis..."
    
    create_valgrind_suppressions
    
    echo "Running tests under Valgrind (this may take several minutes)..."
    
    if valgrind \
        --tool=memcheck \
        --leak-check=full \
        --show-leak-kinds=all \
        --track-origins=yes \
        --suppressions=valgrind.supp \
        --xml=yes \
        --xml-file=valgrind-results.xml \
        --error-exitcode=1 \
        cargo test --package "$PACKAGE" --lib testing::memory_leak_detector::tests::test_memory_leak_detector -- --test-threads=1 2>&1 | tee valgrind.log; then
        success "Valgrind analysis passed - no memory leaks detected"
        return 0
    else
        error "Valgrind detected memory leaks or errors"
        
        # Parse and display leak summary
        if [[ -f valgrind.log ]]; then
            echo "Leak summary:"
            grep -A 10 "LEAK SUMMARY" valgrind.log || true
        fi
        
        return 1
    fi
}

# Run heaptrack profiling
run_heaptrack_profiling() {
    if [[ $HEAPTRACK_ENABLED == false ]]; then
        log "Skipping heaptrack profiling (disabled)"
        return 0
    fi
    
    log "Running heaptrack memory profiling..."
    
    echo "Profiling memory usage with heaptrack..."
    
    # Run a simple test under heaptrack
    if heaptrack cargo test --package "$PACKAGE" --lib testing::memory_leak_detector::tests::test_memory_leak_detector -- --exact; then
        success "Heaptrack profiling completed"
        
        # Find the most recent heaptrack file
        HEAPTRACK_FILE=$(ls -t heaptrack.*.gz 2>/dev/null | head -1)
        
        if [[ -n "$HEAPTRACK_FILE" ]]; then
            echo "Analyzing heaptrack results..."
            heaptrack_print "$HEAPTRACK_FILE" > heaptrack-analysis.txt
            
            # Show summary
            echo "Memory usage summary:"
            grep -E "(peak heap memory consumption|total memory allocated)" heaptrack-analysis.txt || true
            
            success "Heaptrack analysis saved to heaptrack-analysis.txt"
        fi
        
        return 0
    else
        error "Heaptrack profiling failed"
        return 1
    fi
}

# Generate summary report
generate_report() {
    local basic_result=$1
    local valgrind_result=$2
    local heaptrack_result=$3
    
    echo
    echo "=========================================="
    echo "Memory Leak Detection Summary"
    echo "=========================================="
    echo "Package: $PACKAGE"
    echo "Test filter: $TEST_FILTER"
    echo "Date: $(date)"
    echo
    
    if [[ $basic_result -eq 0 ]]; then
        echo -e "Basic tests:      ${GREEN}PASSED${NC}"
    else
        echo -e "Basic tests:      ${RED}FAILED${NC}"
    fi
    
    if [[ $VALGRIND_ENABLED == true ]]; then
        if [[ $valgrind_result -eq 0 ]]; then
            echo -e "Valgrind:         ${GREEN}PASSED${NC}"
        else
            echo -e "Valgrind:         ${RED}FAILED${NC}"
        fi
    else
        echo -e "Valgrind:         ${YELLOW}SKIPPED${NC}"
    fi
    
    if [[ $HEAPTRACK_ENABLED == true ]]; then
        if [[ $heaptrack_result -eq 0 ]]; then
            echo -e "Heaptrack:        ${GREEN}PASSED${NC}"
        else
            echo -e "Heaptrack:        ${RED}FAILED${NC}"
        fi
    else
        echo -e "Heaptrack:        ${YELLOW}SKIPPED${NC}"
    fi
    
    echo
    
    # Overall result
    if [[ $basic_result -eq 0 && $valgrind_result -eq 0 && $heaptrack_result -eq 0 ]]; then
        success "Overall result: All memory leak checks passed!"
        return 0
    else
        error "Overall result: Some memory leak checks failed!"
        echo
        echo "Generated files:"
        ls -la valgrind.log valgrind-results.xml heaptrack-analysis.txt heaptrack.*.gz 2>/dev/null || true
        return 1
    fi
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    rm -f valgrind.supp
}

# Set trap for cleanup
trap cleanup EXIT

# Main execution
main() {
    echo "TrustformeRS Memory Leak Detection"
    echo "=================================="
    echo
    
    check_dependencies
    
    # Run tests
    basic_result=0
    valgrind_result=0
    heaptrack_result=0
    
    run_basic_tests || basic_result=$?
    run_valgrind_analysis || valgrind_result=$?
    run_heaptrack_profiling || heaptrack_result=$?
    
    # Generate and show report
    generate_report $basic_result $valgrind_result $heaptrack_result
}

# Run main function
main "$@"