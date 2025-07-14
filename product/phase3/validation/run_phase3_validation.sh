#!/bin/bash

# Phase 3 Comprehensive Validation Script
# This script runs all validation tests for Phase 3 success criteria

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Results tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0

# Test result arrays
declare -a FAILED_TEST_NAMES=()
declare -a SKIPPED_TEST_NAMES=()

# Utility functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

run_test() {
    local test_name=$1
    local test_command=$2
    local required=${3:-true}
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    echo -e "\n${YELLOW}Running:${NC} $test_name"
    
    if eval "$test_command"; then
        echo -e "${GREEN}✓ PASSED:${NC} $test_name"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        if [ "$required" = true ]; then
            echo -e "${RED}✗ FAILED:${NC} $test_name"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            FAILED_TEST_NAMES+=("$test_name")
            return 1
        else
            echo -e "${YELLOW}⚠ SKIPPED:${NC} $test_name (optional)"
            SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
            SKIPPED_TEST_NAMES+=("$test_name")
            return 0
        fi
    fi
}

# Create results directory
RESULTS_DIR="validation_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

# Start validation
echo "======================================"
echo "Phase 3 Validation Suite"
echo "======================================"
echo "Started at: $(date)"
echo "Results directory: $RESULTS_DIR"
echo ""

# 1. Neural Model Validation
log_info "Starting Neural Model Validation..."

run_test "JavaScript Injection Detection" \
    "cargo test --release test_js_injection_detection -- --nocapture > $RESULTS_DIR/js_injection.log 2>&1"

run_test "Embedded Executable Detection" \
    "cargo test --release test_executable_detection -- --nocapture > $RESULTS_DIR/executable_detection.log 2>&1"

run_test "Memory Bomb Detection" \
    "cargo test --release test_memory_bomb_detection -- --nocapture > $RESULTS_DIR/memory_bomb.log 2>&1"

run_test "Exploit Pattern Recognition" \
    "cargo test --release test_exploit_pattern_detection -- --nocapture > $RESULTS_DIR/exploit_pattern.log 2>&1"

run_test "Anomaly Detection Network" \
    "cargo test --release test_anomaly_detection -- --nocapture > $RESULTS_DIR/anomaly_detection.log 2>&1"

# 2. Document Enhancement Models
log_info "Testing Document Enhancement Models..."

run_test "OCR Accuracy Benchmark" \
    "cargo bench ocr_accuracy_benchmark > $RESULTS_DIR/ocr_benchmark.log 2>&1"

run_test "Table Structure Recognition" \
    "cargo test --release test_table_extraction -- --nocapture > $RESULTS_DIR/table_extraction.log 2>&1"

# 3. Code Quality Checks
log_info "Performing Code Quality Checks..."

run_test "TODO/FIXME Scan" \
    "! rg -c 'TODO|FIXME|XXX' --type rust src/ | grep -v ':0$' > $RESULTS_DIR/todo_scan.log 2>&1"

run_test "Unwrap Usage Check" \
    "! rg '\.unwrap\(\)' --type rust src/ | grep -v 'tests/' > $RESULTS_DIR/unwrap_scan.log 2>&1"

# 4. Plugin System Tests
log_info "Testing Plugin System..."

run_test "PDF Plugin Integration" \
    "cargo test --release test_pdf_plugin_integration -- --nocapture > $RESULTS_DIR/pdf_plugin.log 2>&1"

run_test "DOCX Plugin Integration" \
    "cargo test --release test_docx_plugin_integration -- --nocapture > $RESULTS_DIR/docx_plugin.log 2>&1"

run_test "Plugin Hot-Reload" \
    "timeout 60 cargo run --release --example plugin_hot_reload_test > $RESULTS_DIR/hot_reload.log 2>&1" \
    false

# 5. Security Validation
log_info "Running Security Tests..."

run_test "Sandbox Security" \
    "cargo test --release security_sandbox_tests -- --test-threads=1 > $RESULTS_DIR/sandbox_security.log 2>&1"

run_test "Audit System" \
    "cargo test --release test_security_audit_system > $RESULTS_DIR/audit_system.log 2>&1"

# 6. Performance Tests
log_info "Running Performance Tests..."

run_test "SIMD Performance" \
    "cargo bench simd_performance > $RESULTS_DIR/simd_performance.log 2>&1"

run_test "Load Test (Short)" \
    "timeout 300 cargo run --release --bin load_test -- --documents 1000 --concurrent 10 > $RESULTS_DIR/load_test.log 2>&1" \
    false

# 7. API Tests
log_info "Testing REST API..."

run_test "API Integration Tests" \
    "cargo test --release api_integration_tests > $RESULTS_DIR/api_tests.log 2>&1"

# 8. Python Bindings (if available)
if [ -d "python" ]; then
    log_info "Testing Python Bindings..."
    
    run_test "Python Package Tests" \
        "cd python && python -m pytest -v > ../$RESULTS_DIR/python_tests.log 2>&1" \
        false
fi

# 9. Documentation Check
log_info "Checking Documentation..."

run_test "Documentation Build" \
    "cargo doc --no-deps --document-private-items > $RESULTS_DIR/doc_build.log 2>&1"

# 10. Build Validation
log_info "Validating Build..."

run_test "Release Build" \
    "cargo build --release --all-features > $RESULTS_DIR/release_build.log 2>&1"

run_test "Test Suite" \
    "cargo test --release > $RESULTS_DIR/test_suite.log 2>&1"

# 11. Coverage Analysis (optional)
if command -v cargo-tarpaulin &> /dev/null; then
    log_info "Running Coverage Analysis..."
    
    run_test "Code Coverage" \
        "cargo tarpaulin --all-features --out Html --output-dir $RESULTS_DIR/coverage > $RESULTS_DIR/coverage.log 2>&1" \
        false
fi

# Generate Summary Report
echo ""
echo "======================================"
echo "Validation Summary"
echo "======================================"
echo "Total Tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED_TESTS${NC}"
echo -e "${RED}Failed: $FAILED_TESTS${NC}"
echo -e "${YELLOW}Skipped: $SKIPPED_TESTS${NC}"
echo ""

# Calculate success rate
if [ $TOTAL_TESTS -gt 0 ]; then
    SUCCESS_RATE=$(echo "scale=2; ($PASSED_TESTS * 100) / $TOTAL_TESTS" | bc)
    echo "Success Rate: $SUCCESS_RATE%"
fi

# List failed tests
if [ ${#FAILED_TEST_NAMES[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed Tests:${NC}"
    for test in "${FAILED_TEST_NAMES[@]}"; do
        echo "  - $test"
    done
fi

# List skipped tests
if [ ${#SKIPPED_TEST_NAMES[@]} -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Skipped Tests:${NC}"
    for test in "${SKIPPED_TEST_NAMES[@]}"; do
        echo "  - $test"
    done
fi

# Generate detailed report
cat > "$RESULTS_DIR/validation_summary.txt" << EOF
Phase 3 Validation Report
========================
Date: $(date)
Total Tests: $TOTAL_TESTS
Passed: $PASSED_TESTS
Failed: $FAILED_TESTS
Skipped: $SKIPPED_TESTS
Success Rate: $SUCCESS_RATE%

Failed Tests:
$(printf '%s\n' "${FAILED_TEST_NAMES[@]}")

Skipped Tests:
$(printf '%s\n' "${SKIPPED_TEST_NAMES[@]}")

Detailed logs available in: $RESULTS_DIR/
EOF

echo ""
echo "Detailed results saved to: $RESULTS_DIR/"
echo "Summary report: $RESULTS_DIR/validation_summary.txt"

# Exit with appropriate code
if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}✓ All required tests passed!${NC}"
    exit 0
else
    echo -e "\n${RED}✗ Some tests failed. Please review the logs.${NC}"
    exit 1
fi