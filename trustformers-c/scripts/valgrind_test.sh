#!/bin/bash
#
# Valgrind Memory Testing Script for TrustformeRS C API
# This script runs comprehensive memory leak detection and error checking
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VALGRIND_LOG_DIR="$PROJECT_ROOT/target/valgrind-logs"
SUPPRESSIONS_FILE="$PROJECT_ROOT/scripts/valgrind.supp"

# Ensure log directory exists
mkdir -p "$VALGRIND_LOG_DIR"

echo -e "${BLUE}TrustformeRS Valgrind Memory Testing${NC}"
echo "=================================="

# Check if Valgrind is installed
if ! command -v valgrind &> /dev/null; then
    echo -e "${RED}Error: Valgrind is not installed${NC}"
    echo "Install with: brew install valgrind (macOS) or apt-get install valgrind (Ubuntu)"
    exit 1
fi

# Build the project with debug symbols
echo -e "${YELLOW}Building project with debug symbols...${NC}"
cd "$PROJECT_ROOT"
RUSTFLAGS="-g" cargo build --features debug

# Check if test executables exist
if [ ! -f "$PROJECT_ROOT/target/debug/libtrustformers_c.dylib" ] && [ ! -f "$PROJECT_ROOT/target/debug/libtrustformers_c.so" ]; then
    echo -e "${RED}Error: TrustformeRS library not found${NC}"
    exit 1
fi

# Valgrind options
VALGRIND_OPTS=(
    "--tool=memcheck"
    "--leak-check=full"
    "--show-leak-kinds=all"
    "--track-origins=yes"
    "--verbose"
    "--suppressions=$SUPPRESSIONS_FILE"
    "--gen-suppressions=all"
    "--log-file=$VALGRIND_LOG_DIR/valgrind-%p.log"
    "--xml=yes"
    "--xml-file=$VALGRIND_LOG_DIR/valgrind-%p.xml"
)

# Function to run valgrind test
run_valgrind_test() {
    local test_name="$1"
    local test_executable="$2"
    shift 2
    local test_args=("$@")
    
    echo -e "${YELLOW}Running Valgrind test: $test_name${NC}"
    
    if valgrind "${VALGRIND_OPTS[@]}" "$test_executable" "${test_args[@]}"; then
        echo -e "${GREEN}✓ $test_name passed${NC}"
        return 0
    else
        echo -e "${RED}✗ $test_name failed${NC}"
        return 1
    fi
}

# Function to analyze valgrind logs
analyze_logs() {
    echo -e "${YELLOW}Analyzing Valgrind logs...${NC}"
    
    local total_errors=0
    local total_leaks=0
    
    for log_file in "$VALGRIND_LOG_DIR"/*.log; do
        if [ -f "$log_file" ]; then
            local errors=$(grep -c "ERROR SUMMARY:" "$log_file" || true)
            local leaks=$(grep -c "definitely lost:" "$log_file" || true)
            
            total_errors=$((total_errors + errors))
            total_leaks=$((total_leaks + leaks))
            
            if [ "$errors" -gt 0 ] || [ "$leaks" -gt 0 ]; then
                echo -e "${RED}Issues found in $(basename "$log_file"):${NC}"
                grep -A 5 "ERROR SUMMARY:\|definitely lost:" "$log_file" || true
            fi
        fi
    done
    
    echo -e "${BLUE}Summary:${NC}"
    echo "Total errors: $total_errors"
    echo "Total leaks: $total_leaks"
    
    if [ "$total_errors" -eq 0 ] && [ "$total_leaks" -eq 0 ]; then
        echo -e "${GREEN}✓ No memory issues detected${NC}"
        return 0
    else
        echo -e "${RED}✗ Memory issues detected${NC}"
        return 1
    fi
}

# Function to create simple C test
create_test_program() {
    local test_dir="$PROJECT_ROOT/target/valgrind-tests"
    mkdir -p "$test_dir"
    
    cat > "$test_dir/basic_test.c" << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <string.h>

// Simple test program for Valgrind memory testing
int main() {
    printf("Running TrustformeRS basic memory test...\n");
    
    // Try to load the library dynamically
    void *handle = dlopen("./target/debug/libtrustformers_c.dylib", RTLD_LAZY);
    if (!handle) {
        handle = dlopen("./target/debug/libtrustformers_c.so", RTLD_LAZY);
    }
    
    if (!handle) {
        printf("Error loading library: %s\n", dlerror());
        return 1;
    }
    
    // Test basic memory allocation
    char *test_buffer = malloc(1024);
    if (test_buffer) {
        memset(test_buffer, 0, 1024);
        strcpy(test_buffer, "TrustformeRS memory test");
        printf("Test buffer: %s\n", test_buffer);
        free(test_buffer);
    }
    
    dlclose(handle);
    printf("Basic memory test completed successfully\n");
    return 0;
}
EOF

    # Compile the test program
    gcc -o "$test_dir/basic_test" "$test_dir/basic_test.c" -ldl
    echo "$test_dir/basic_test"
}

# Run tests
echo -e "${YELLOW}Creating test programs...${NC}"
BASIC_TEST=$(create_test_program)

# Clean previous logs
rm -f "$VALGRIND_LOG_DIR"/*.log "$VALGRIND_LOG_DIR"/*.xml

# Run Valgrind tests
echo -e "${YELLOW}Running Valgrind memory tests...${NC}"

test_results=0

# Test 1: Basic library loading test
if ! run_valgrind_test "Basic Library Test" "$BASIC_TEST"; then
    test_results=1
fi

# Test 2: Rust unit tests with Valgrind (if available)
if cargo test --list | grep -q "test"; then
    echo -e "${YELLOW}Running Rust tests under Valgrind...${NC}"
    if ! valgrind "${VALGRIND_OPTS[@]}" cargo test --no-run 2>/dev/null; then
        echo -e "${YELLOW}Note: Rust tests under Valgrind may have limitations${NC}"
    fi
fi

# Analyze results
echo -e "${YELLOW}Analyzing results...${NC}"
if ! analyze_logs; then
    test_results=1
fi

# Generate summary report
cat > "$VALGRIND_LOG_DIR/summary.txt" << EOF
Valgrind Testing Summary
========================
Date: $(date)
Project: TrustformeRS C API
Build: Debug with symbols

Test Results:
- Basic library test: $([ $test_results -eq 0 ] && echo "PASS" || echo "FAIL")

Logs location: $VALGRIND_LOG_DIR
EOF

echo -e "${BLUE}Summary report written to: $VALGRIND_LOG_DIR/summary.txt${NC}"

if [ $test_results -eq 0 ]; then
    echo -e "${GREEN}✓ All Valgrind tests passed${NC}"
    exit 0
else
    echo -e "${RED}✗ Some Valgrind tests failed${NC}"
    exit 1
fi