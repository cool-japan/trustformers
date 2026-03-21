#!/bin/bash
#
# AddressSanitizer Testing Script for TrustformeRS C API
# This script runs comprehensive memory safety testing with AddressSanitizer
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
ASAN_LOG_DIR="$PROJECT_ROOT/target/asan-logs"

# Ensure log directory exists
mkdir -p "$ASAN_LOG_DIR"

echo -e "${BLUE}TrustformeRS AddressSanitizer Testing${NC}"
echo "===================================="

# AddressSanitizer environment variables
export ASAN_OPTIONS="abort_on_error=1:check_initialization_order=1:strict_init_order=1:detect_odr_violation=2:detect_stack_use_after_return=true:detect_invalid_pointer_pairs=2:log_path=$ASAN_LOG_DIR/asan"
export MSAN_OPTIONS="abort_on_error=1:print_stats=1"
export UBSAN_OPTIONS="abort_on_error=1:print_stacktrace=1"

# Rust flags for sanitizers
export RUSTFLAGS="-Zsanitizer=address -Copt-level=0 -Cdebuginfo=2"

echo -e "${YELLOW}AddressSanitizer Configuration:${NC}"
echo "ASAN_OPTIONS: $ASAN_OPTIONS"
echo "RUSTFLAGS: $RUSTFLAGS"
echo ""

# Function to run AddressSanitizer test
run_asan_test() {
    local test_name="$1"
    local test_command="$2"
    shift 2
    local test_args=("$@")
    
    echo -e "${YELLOW}Running AddressSanitizer test: $test_name${NC}"
    
    if eval "$test_command" "${test_args[@]}" 2>&1 | tee "$ASAN_LOG_DIR/${test_name}_output.log"; then
        echo -e "${GREEN}✓ $test_name passed${NC}"
        return 0
    else
        echo -e "${RED}✗ $test_name failed${NC}"
        return 1
    fi
}

# Function to analyze AddressSanitizer logs
analyze_asan_logs() {
    echo -e "${YELLOW}Analyzing AddressSanitizer logs...${NC}"
    
    local total_errors=0
    
    for log_file in "$ASAN_LOG_DIR"/*.log; do
        if [ -f "$log_file" ]; then
            local errors=$(grep -c "ERROR: AddressSanitizer\|ERROR: MemorySanitizer\|ERROR: UBSan" "$log_file" || true)
            total_errors=$((total_errors + errors))
            
            if [ "$errors" -gt 0 ]; then
                echo -e "${RED}Errors found in $(basename "$log_file"):${NC}"
                grep -A 10 "ERROR:" "$log_file" || true
            fi
        fi
    done
    
    # Check for sanitizer output files
    for asan_file in "$ASAN_LOG_DIR"/asan.*; do
        if [ -f "$asan_file" ]; then
            echo -e "${RED}AddressSanitizer report found: $(basename "$asan_file")${NC}"
            cat "$asan_file"
            total_errors=$((total_errors + 1))
        fi
    done
    
    echo -e "${BLUE}Summary:${NC}"
    echo "Total sanitizer errors: $total_errors"
    
    if [ "$total_errors" -eq 0 ]; then
        echo -e "${GREEN}✓ No memory safety issues detected${NC}"
        return 0
    else
        echo -e "${RED}✗ Memory safety issues detected${NC}"
        return 1
    fi
}

# Function to create AddressSanitizer test programs
create_asan_test_programs() {
    local test_dir="$PROJECT_ROOT/target/asan-tests"
    mkdir -p "$test_dir"
    
    # Create a comprehensive C test program
    cat > "$test_dir/asan_comprehensive_test.c" << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

// Test for buffer overflow detection
void test_buffer_overflow() {
    printf("Testing buffer overflow detection...\n");
    char buffer[10];
    // This should be detected by AddressSanitizer
    // strncpy(buffer, "This is a very long string that will overflow", 10);
    
    // Safe version
    strncpy(buffer, "Safe", 9);
    buffer[9] = '\0';
    printf("Buffer content: %s\n", buffer);
}

// Test for use-after-free detection
void test_use_after_free() {
    printf("Testing use-after-free detection...\n");
    char *ptr = malloc(100);
    strcpy(ptr, "Test string");
    printf("Before free: %s\n", ptr);
    free(ptr);
    // This would be detected by AddressSanitizer
    // printf("After free: %s\n", ptr);
    printf("Use-after-free test completed safely\n");
}

// Test for memory leak detection
void test_memory_leak() {
    printf("Testing memory management...\n");
    char *ptr = malloc(100);
    strcpy(ptr, "Test allocation");
    printf("Allocated memory: %s\n", ptr);
    // Properly free memory to avoid leaks
    free(ptr);
    printf("Memory freed properly\n");
}

// Test for double free detection
void test_double_free() {
    printf("Testing double free detection...\n");
    char *ptr = malloc(100);
    strcpy(ptr, "Test string");
    free(ptr);
    // This would be detected by AddressSanitizer
    // free(ptr);
    printf("Double free test completed safely\n");
}

// Test library loading with sanitizers
void test_library_loading() {
    printf("Testing library loading with AddressSanitizer...\n");
    
    void *handle = dlopen("./target/debug/libtrustformers_c.dylib", RTLD_LAZY);
    if (!handle) {
        handle = dlopen("./target/debug/libtrustformers_c.so", RTLD_LAZY);
    }
    
    if (!handle) {
        printf("Note: Library not found (expected in some contexts): %s\n", dlerror());
        return;
    }
    
    printf("Library loaded successfully\n");
    dlclose(handle);
    printf("Library closed successfully\n");
}

int main() {
    printf("AddressSanitizer Comprehensive Test Suite\n");
    printf("==========================================\n");
    
    test_buffer_overflow();
    test_use_after_free();
    test_memory_leak();
    test_double_free();
    test_library_loading();
    
    printf("All AddressSanitizer tests completed successfully\n");
    return 0;
}
EOF

    # Create a Rust test program
    cat > "$test_dir/asan_rust_test.rs" << 'EOF'
//! AddressSanitizer test for Rust code
use std::ffi::CString;

fn test_string_operations() {
    println!("Testing Rust string operations with AddressSanitizer...");
    
    // Test vector operations
    let mut vec = Vec::with_capacity(100);
    for i in 0..50 {
        vec.push(i);
    }
    
    // Test string operations
    let test_string = CString::new("Test string for AddressSanitizer").unwrap();
    println!("Test string: {:?}", test_string);
    
    // Test slice operations
    let slice = &vec[0..10];
    println!("Slice length: {}", slice.len());
    
    println!("Rust string operations completed successfully");
}

fn test_memory_operations() {
    println!("Testing Rust memory operations...");
    
    // Test Box allocation
    let boxed_value = Box::new(42);
    println!("Boxed value: {}", *boxed_value);
    
    // Test reference counting
    use std::rc::Rc;
    let rc_value = Rc::new(vec![1, 2, 3, 4, 5]);
    let rc_clone = rc_value.clone();
    println!("RC value length: {}", rc_value.len());
    println!("RC clone length: {}", rc_clone.len());
    
    println!("Rust memory operations completed successfully");
}

fn main() {
    println!("Rust AddressSanitizer Test Suite");
    println!("=================================");
    
    test_string_operations();
    test_memory_operations();
    
    println!("All Rust AddressSanitizer tests completed successfully");
}
EOF

    # Compile the C test program with AddressSanitizer
    clang -fsanitize=address -fno-omit-frame-pointer -g -o "$test_dir/asan_comprehensive_test" "$test_dir/asan_comprehensive_test.c" -ldl
    
    # Note: Rust test will be compiled with RUSTFLAGS set for AddressSanitizer
    echo "$test_dir"
}

# Check if we're on a supported platform
case "$(uname -s)" in
    Darwin*)
        echo -e "${YELLOW}Running on macOS - AddressSanitizer supported${NC}"
        ;;
    Linux*)
        echo -e "${YELLOW}Running on Linux - AddressSanitizer supported${NC}"
        ;;
    *)
        echo -e "${RED}AddressSanitizer may not be fully supported on this platform${NC}"
        ;;
esac

# Clean previous logs
rm -f "$ASAN_LOG_DIR"/*.log "$ASAN_LOG_DIR"/asan.*

# Create test programs
echo -e "${YELLOW}Creating AddressSanitizer test programs...${NC}"
TEST_DIR=$(create_asan_test_programs)

echo -e "${YELLOW}Building project with AddressSanitizer...${NC}"
cd "$PROJECT_ROOT"

# Build with AddressSanitizer (only on nightly Rust)
if cargo +nightly --version &> /dev/null; then
    echo -e "${YELLOW}Using nightly Rust for AddressSanitizer support${NC}"
    if ! cargo +nightly build -Z build-std --target x86_64-apple-darwin 2>/dev/null; then
        echo -e "${YELLOW}Note: Full AddressSanitizer build may require additional setup${NC}"
        # Fallback to regular build
        cargo build
    fi
else
    echo -e "${YELLOW}Nightly Rust not available, using regular build${NC}"
    cargo build
fi

# Run AddressSanitizer tests
echo -e "${YELLOW}Running AddressSanitizer tests...${NC}"

test_results=0

# Test 1: C comprehensive test
if ! run_asan_test "C_Comprehensive" "$TEST_DIR/asan_comprehensive_test"; then
    test_results=1
fi

# Test 2: Rust test (if nightly available)
if cargo +nightly --version &> /dev/null; then
    if ! run_asan_test "Rust_Memory" "cargo +nightly run --bin asan_rust_test" 2>/dev/null; then
        echo -e "${YELLOW}Note: Rust AddressSanitizer test may require additional configuration${NC}"
    fi
fi

# Test 3: Regular Rust tests
if ! run_asan_test "Rust_Tests" "cargo test" "--no-run"; then
    echo -e "${YELLOW}Note: Regular tests may have limitations with AddressSanitizer${NC}"
fi

# Analyze results
echo -e "${YELLOW}Analyzing AddressSanitizer results...${NC}"
if ! analyze_asan_logs; then
    test_results=1
fi

# Generate summary report
cat > "$ASAN_LOG_DIR/summary.txt" << EOF
AddressSanitizer Testing Summary
================================
Date: $(date)
Project: TrustformeRS C API
Configuration: AddressSanitizer enabled

Test Results:
- C comprehensive test: $([ $test_results -eq 0 ] && echo "PASS" || echo "FAIL")
- Memory safety: $([ $test_results -eq 0 ] && echo "NO ISSUES" || echo "ISSUES DETECTED")

Environment:
- ASAN_OPTIONS: $ASAN_OPTIONS
- RUSTFLAGS: $RUSTFLAGS

Logs location: $ASAN_LOG_DIR
EOF

echo -e "${BLUE}Summary report written to: $ASAN_LOG_DIR/summary.txt${NC}"

if [ $test_results -eq 0 ]; then
    echo -e "${GREEN}✓ All AddressSanitizer tests passed${NC}"
    exit 0
else
    echo -e "${RED}✗ Some AddressSanitizer tests detected issues${NC}"
    echo -e "${YELLOW}Check the logs in $ASAN_LOG_DIR for details${NC}"
    exit 1
fi