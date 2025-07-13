#!/bin/bash
# Phase 2 Test Runner Script

set -e

echo "=== Phase 2 Test Suite ==="
echo "Running comprehensive tests for all Phase 2 components"
echo

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to run tests for a component
run_component_tests() {
    local component=$1
    echo -e "${YELLOW}Testing $component...${NC}"
    
    if cargo test -p $component --lib --tests -- --nocapture 2>&1; then
        echo -e "${GREEN}✓ $component tests passed${NC}"
        ((PASSED_TESTS++))
    else
        echo -e "${RED}✗ $component tests failed${NC}"
        ((FAILED_TESTS++))
    fi
    ((TOTAL_TESTS++))
    echo
}

# Run tests for each Phase 2 component
echo "1. Testing Security Module"
run_component_tests "neural-doc-flow-security"

echo "2. Testing Plugin System"
run_component_tests "neural-doc-flow-plugins"

echo "3. Testing Enhanced Core"
run_component_tests "neural-doc-flow-core"

# Run integration tests
echo "4. Running Integration Tests"
if cargo test --workspace integration -- --nocapture 2>&1; then
    echo -e "${GREEN}✓ Integration tests passed${NC}"
    ((PASSED_TESTS++))
else
    echo -e "${RED}✗ Integration tests failed${NC}"
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))

# Run security-specific tests
echo -e "\n5. Running Security Tests"
if cargo test --workspace security -- --nocapture 2>&1; then
    echo -e "${GREEN}✓ Security tests passed${NC}"
    ((PASSED_TESTS++))
else
    echo -e "${RED}✗ Security tests failed${NC}"
    ((FAILED_TESTS++))
fi
((TOTAL_TESTS++))

# Check for compilation warnings
echo -e "\n6. Checking for Compilation Warnings"
if cargo check --workspace --all-targets 2>&1 | grep -q "warning"; then
    echo -e "${YELLOW}⚠ Compilation warnings found${NC}"
else
    echo -e "${GREEN}✓ No compilation warnings${NC}"
fi

# Run clippy for code quality
echo -e "\n7. Running Clippy (Linting)"
if cargo clippy --workspace --all-targets -- -D warnings 2>&1; then
    echo -e "${GREEN}✓ Clippy passed${NC}"
else
    echo -e "${YELLOW}⚠ Clippy warnings found${NC}"
fi

# Check documentation
echo -e "\n8. Checking Documentation"
if cargo doc --workspace --no-deps --document-private-items 2>&1; then
    echo -e "${GREEN}✓ Documentation builds successfully${NC}"
else
    echo -e "${RED}✗ Documentation errors found${NC}"
fi

# Summary
echo -e "\n${YELLOW}=== Test Summary ===${NC}"
echo "Total test suites: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
echo -e "Failed: ${RED}$FAILED_TESTS${NC}"

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "\n${GREEN}All tests passed! Phase 2 components are working correctly.${NC}"
    exit 0
else
    echo -e "\n${RED}Some tests failed. Please review the output above.${NC}"
    exit 1
fi