#!/bin/bash
# Library Usage Validation Script
# Ensures implementation uses libraries instead of custom code

set -e

echo "🔍 Library Usage Validation Starting..."
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Validation counters
TOTAL_CHECKS=0
PASSED_CHECKS=0
FAILED_CHECKS=0

# Function to check for pattern
check_pattern() {
    local pattern="$1"
    local message="$2"
    local should_exist="$3"  # true if pattern should exist, false if it shouldn't
    
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if [ "$should_exist" = "true" ]; then
        if grep -r "$pattern" src/ --include="*.js" --include="*.ts" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ PASS:${NC} $message"
            PASSED_CHECKS=$((PASSED_CHECKS + 1))
        else
            echo -e "${RED}❌ FAIL:${NC} $message"
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
        fi
    else
        if grep -r "$pattern" src/ --include="*.js" --include="*.ts" > /dev/null 2>&1; then
            echo -e "${RED}❌ FAIL:${NC} $message"
            FAILED_CHECKS=$((FAILED_CHECKS + 1))
            grep -r "$pattern" src/ --include="*.js" --include="*.ts" | head -5
        else
            echo -e "${GREEN}✅ PASS:${NC} $message"
            PASSED_CHECKS=$((PASSED_CHECKS + 1))
        fi
    fi
}

echo -e "\n${YELLOW}1. Checking Claude Flow Usage...${NC}"
echo "-----------------------------------"
check_pattern "claude-flow@alpha swarm init" "Using claude-flow for swarm initialization" true
check_pattern "claude-flow@alpha memory store" "Using claude-flow for memory storage" true
check_pattern "claude-flow@alpha memory query" "Using claude-flow for memory queries" true
check_pattern "hooks pre-task" "Using pre-task coordination hooks" true
check_pattern "hooks post-task" "Using post-task coordination hooks" true
check_pattern "hooks post-edit" "Using post-edit coordination hooks" true

echo -e "\n${YELLOW}2. Checking ruv-FANN Usage...${NC}"
echo "-----------------------------------"
check_pattern "require.*ruv-fann\|import.*ruv-fann" "Importing ruv-FANN library" true
check_pattern "FANN\.create\|fann\.create" "Using FANN for neural network creation" true
check_pattern "FANN\.train\|fann\.train" "Using FANN for training" true

echo -e "\n${YELLOW}3. Checking DAA Usage...${NC}"
echo "-----------------------------------"
check_pattern "require.*daa\|import.*daa" "Importing DAA library" true
check_pattern "DAA\.distribute\|daa\.distribute" "Using DAA for distribution" true

echo -e "\n${YELLOW}4. Checking for Anti-Patterns...${NC}"
echo "-----------------------------------"
check_pattern "class SwarmManager" "No custom SwarmManager (use claude-flow)" false
check_pattern "class NeuralNetwork" "No custom NeuralNetwork (use ruv-FANN)" false
check_pattern "new Worker(" "No Worker threads (use DAA)" false
check_pattern "localStorage\." "No localStorage (use claude-flow memory)" false
check_pattern "sessionStorage\." "No sessionStorage (use claude-flow memory)" false
check_pattern "fs\.writeFile" "No direct file writes (use claude-flow memory)" false
check_pattern "IndexedDB" "No IndexedDB (use claude-flow memory)" false
check_pattern "cluster\.fork" "No cluster module (use DAA)" false
check_pattern "child_process" "No child_process (use DAA)" false
check_pattern "backpropagation" "No custom backprop (use FANN.train)" false
check_pattern "class RuleEngine" "No custom RuleEngine" false
check_pattern "EventEmitter" "No EventEmitter (use claude-flow hooks)" false

echo -e "\n${YELLOW}5. Checking Required Files...${NC}"
echo "-----------------------------------"

# Check for required structure
if [ -d "src/foundation" ]; then
    echo -e "${GREEN}✅ PASS:${NC} Foundation directory exists"
    PASSED_CHECKS=$((PASSED_CHECKS + 1))
else
    echo -e "${RED}❌ FAIL:${NC} Foundation directory missing"
    FAILED_CHECKS=$((FAILED_CHECKS + 1))
fi
TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

# Check package.json for dependencies
echo -e "\n${YELLOW}6. Checking Dependencies...${NC}"
echo "-----------------------------------"

if [ -f "package.json" ]; then
    if grep -q "claude-flow" package.json; then
        echo -e "${GREEN}✅ PASS:${NC} claude-flow dependency found"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        echo -e "${RED}❌ FAIL:${NC} claude-flow dependency missing"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if grep -q "ruv-fann" package.json; then
        echo -e "${GREEN}✅ PASS:${NC} ruv-fann dependency found"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        echo -e "${RED}❌ FAIL:${NC} ruv-fann dependency missing"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    
    if grep -q "daa" package.json; then
        echo -e "${GREEN}✅ PASS:${NC} daa dependency found"
        PASSED_CHECKS=$((PASSED_CHECKS + 1))
    else
        echo -e "${RED}❌ FAIL:${NC} daa dependency missing"
        FAILED_CHECKS=$((FAILED_CHECKS + 1))
    fi
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
fi

# Summary
echo -e "\n${YELLOW}=================================="
echo "VALIDATION SUMMARY"
echo "==================================${NC}"
echo "Total Checks: $TOTAL_CHECKS"
echo -e "${GREEN}Passed: $PASSED_CHECKS${NC}"
echo -e "${RED}Failed: $FAILED_CHECKS${NC}"

if [ $FAILED_CHECKS -eq 0 ]; then
    echo -e "\n${GREEN}✅ ALL VALIDATION CHECKS PASSED!${NC}"
    echo "The implementation correctly uses required libraries."
    exit 0
else
    echo -e "\n${RED}❌ VALIDATION FAILED!${NC}"
    echo "$FAILED_CHECKS checks failed. Please use the required libraries instead of custom implementations."
    echo ""
    echo "Required libraries:"
    echo "  - claude-flow@alpha: For swarm coordination and memory"
    echo "  - ruv-FANN: For all neural network operations"
    echo "  - DAA: For distributed processing"
    exit 1
fi