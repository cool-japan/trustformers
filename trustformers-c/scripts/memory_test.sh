#!/bin/bash
#
# Comprehensive Memory Testing Script for TrustformeRS C API
# This script runs both Valgrind and AddressSanitizer tests
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_LOG_DIR="$PROJECT_ROOT/target/memory-test-logs"

# Ensure log directory exists
mkdir -p "$TEST_LOG_DIR"

echo -e "${BOLD}${BLUE}TrustformeRS Comprehensive Memory Testing${NC}"
echo "=========================================="
echo ""

# Parse command line arguments
RUN_VALGRIND=true
RUN_ASAN=true
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-valgrind)
            RUN_VALGRIND=false
            shift
            ;;
        --no-asan)
            RUN_ASAN=false
            shift
            ;;
        --valgrind-only)
            RUN_ASAN=false
            shift
            ;;
        --asan-only)
            RUN_VALGRIND=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --no-valgrind     Skip Valgrind tests"
            echo "  --no-asan         Skip AddressSanitizer tests"
            echo "  --valgrind-only   Run only Valgrind tests"
            echo "  --asan-only       Run only AddressSanitizer tests"
            echo "  --verbose         Enable verbose output"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to log with timestamp
log_with_timestamp() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Function to run test and capture results
run_test_suite() {
    local test_name="$1"
    local test_script="$2"
    local log_file="$TEST_LOG_DIR/${test_name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo -e "${YELLOW}Running $test_name test suite...${NC}"
    log_with_timestamp "Starting $test_name tests" >> "$log_file"
    
    if [ "$VERBOSE" = true ]; then
        echo "Log file: $log_file"
    fi
    
    if bash "$test_script" 2>&1 | tee -a "$log_file"; then
        echo -e "${GREEN}✓ $test_name tests completed successfully${NC}"
        log_with_timestamp "$test_name tests PASSED" >> "$log_file"
        return 0
    else
        echo -e "${RED}✗ $test_name tests failed${NC}"
        log_with_timestamp "$test_name tests FAILED" >> "$log_file"
        return 1
    fi
}

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    local missing_tools=()
    
    if [ "$RUN_VALGRIND" = true ]; then
        if ! command -v valgrind &> /dev/null; then
            missing_tools+=("valgrind")
        fi
    fi
    
    if [ "$RUN_ASAN" = true ]; then
        if ! command -v clang &> /dev/null; then
            missing_tools+=("clang")
        fi
    fi
    
    if ! command -v cargo &> /dev/null; then
        missing_tools+=("cargo")
    fi
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        echo -e "${RED}Missing required tools: ${missing_tools[*]}${NC}"
        echo "Please install the missing tools and try again."
        return 1
    fi
    
    echo -e "${GREEN}✓ All prerequisites satisfied${NC}"
    return 0
}

# Function to generate comprehensive report
generate_comprehensive_report() {
    local report_file="$TEST_LOG_DIR/comprehensive_report.txt"
    local html_report="$TEST_LOG_DIR/comprehensive_report.html"
    
    echo -e "${YELLOW}Generating comprehensive report...${NC}"
    
    # Text report
    cat > "$report_file" << EOF
TrustformeRS Memory Testing Comprehensive Report
===============================================
Date: $(date)
Project: TrustformeRS C API
Platform: $(uname -s) $(uname -m)
Compiler: $(clang --version | head -n1 2>/dev/null || echo "N/A")
Rust: $(cargo --version)

Test Configuration:
- Valgrind tests: $([ "$RUN_VALGRIND" = true ] && echo "ENABLED" || echo "DISABLED")
- AddressSanitizer tests: $([ "$RUN_ASAN" = true ] && echo "ENABLED" || echo "DISABLED")
- Verbose output: $([ "$VERBOSE" = true ] && echo "ENABLED" || echo "DISABLED")

Test Results Summary:
EOF

    # Analyze Valgrind results
    if [ "$RUN_VALGRIND" = true ]; then
        local valgrind_status="UNKNOWN"
        if [ -f "$PROJECT_ROOT/target/valgrind-logs/summary.txt" ]; then
            if grep -q "No memory issues detected" "$PROJECT_ROOT/target/valgrind-logs/summary.txt"; then
                valgrind_status="PASSED"
            else
                valgrind_status="FAILED"
            fi
        fi
        echo "- Valgrind: $valgrind_status" >> "$report_file"
    fi
    
    # Analyze AddressSanitizer results
    if [ "$RUN_ASAN" = true ]; then
        local asan_status="UNKNOWN"
        if [ -f "$PROJECT_ROOT/target/asan-logs/summary.txt" ]; then
            if grep -q "NO ISSUES" "$PROJECT_ROOT/target/asan-logs/summary.txt"; then
                asan_status="PASSED"
            else
                asan_status="FAILED"
            fi
        fi
        echo "- AddressSanitizer: $asan_status" >> "$report_file"
    fi
    
    cat >> "$report_file" << EOF

Detailed Information:
- Test logs directory: $TEST_LOG_DIR
- Valgrind logs: $PROJECT_ROOT/target/valgrind-logs/
- AddressSanitizer logs: $PROJECT_ROOT/target/asan-logs/

System Information:
- OS: $(uname -a)
- Memory: $(free -h 2>/dev/null | grep Mem || vm_stat | head -5)
- CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || grep "model name" /proc/cpuinfo | head -1 || echo "Unknown")

EOF

    # HTML report (basic)
    cat > "$html_report" << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>TrustformeRS Memory Testing Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }
        .pass { color: green; font-weight: bold; }
        .fail { color: red; font-weight: bold; }
        .unknown { color: orange; font-weight: bold; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>TrustformeRS Memory Testing Report</h1>
        <p><strong>Generated:</strong> {{ DATE }}</p>
        <p><strong>Platform:</strong> {{ PLATFORM }}</p>
    </div>
    
    <h2>Test Results</h2>
    <table>
        <tr><th>Test Suite</th><th>Status</th><th>Notes</th></tr>
        <tr><td>Valgrind</td><td class="{{ VALGRIND_CLASS }}">{{ VALGRIND_STATUS }}</td><td>Memory leak and error detection</td></tr>
        <tr><td>AddressSanitizer</td><td class="{{ ASAN_CLASS }}">{{ ASAN_STATUS }}</td><td>Runtime memory error detection</td></tr>
    </table>
    
    <h2>Log Files</h2>
    <ul>
        <li>Test logs: {{ TEST_LOG_DIR }}</li>
        <li>Valgrind logs: {{ VALGRIND_LOG_DIR }}</li>
        <li>AddressSanitizer logs: {{ ASAN_LOG_DIR }}</li>
    </ul>
    
    <h2>Recommendations</h2>
    <ul>
        <li>Review detailed logs for any memory safety issues</li>
        <li>Run tests regularly in CI/CD pipeline</li>
        <li>Consider additional sanitizers for production builds</li>
    </ul>
</body>
</html>
EOF

    echo -e "${GREEN}✓ Reports generated:${NC}"
    echo "  Text report: $report_file"
    echo "  HTML report: $html_report"
}

# Main execution
main() {
    cd "$PROJECT_ROOT"
    
    echo -e "${YELLOW}Project root: $PROJECT_ROOT${NC}"
    echo -e "${YELLOW}Test configuration:${NC}"
    echo "  Valgrind: $([ "$RUN_VALGRIND" = true ] && echo "enabled" || echo "disabled")"
    echo "  AddressSanitizer: $([ "$RUN_ASAN" = true ] && echo "enabled" || echo "disabled")"
    echo ""
    
    # Check prerequisites
    if ! check_prerequisites; then
        exit 1
    fi
    
    # Initialize test results
    local overall_result=0
    
    # Run Valgrind tests
    if [ "$RUN_VALGRIND" = true ]; then
        echo -e "${BOLD}Running Valgrind Test Suite${NC}"
        echo "=============================="
        if ! run_test_suite "Valgrind" "$SCRIPT_DIR/valgrind_test.sh"; then
            overall_result=1
        fi
        echo ""
    fi
    
    # Run AddressSanitizer tests
    if [ "$RUN_ASAN" = true ]; then
        echo -e "${BOLD}Running AddressSanitizer Test Suite${NC}"
        echo "==================================="
        if ! run_test_suite "AddressSanitizer" "$SCRIPT_DIR/asan_test.sh"; then
            overall_result=1
        fi
        echo ""
    fi
    
    # Generate comprehensive report
    generate_comprehensive_report
    
    # Final summary
    echo -e "${BOLD}Final Summary${NC}"
    echo "============="
    
    if [ $overall_result -eq 0 ]; then
        echo -e "${GREEN}✓ All memory testing completed successfully${NC}"
        echo -e "${GREEN}  No memory safety issues detected${NC}"
    else
        echo -e "${RED}✗ Memory testing detected issues${NC}"
        echo -e "${RED}  Please review the detailed logs for more information${NC}"
    fi
    
    echo ""
    echo "For detailed analysis, check:"
    echo "  - $TEST_LOG_DIR/"
    if [ "$RUN_VALGRIND" = true ]; then
        echo "  - $PROJECT_ROOT/target/valgrind-logs/"
    fi
    if [ "$RUN_ASAN" = true ]; then
        echo "  - $PROJECT_ROOT/target/asan-logs/"
    fi
    
    exit $overall_result
}

# Run main function
main "$@"