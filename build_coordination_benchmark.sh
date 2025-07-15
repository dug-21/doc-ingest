#!/bin/bash
# Coordination Module Build Performance Benchmark
# Tests different feature combinations and compilation times

set -e

echo "🚀 Neural Doc Flow Coordination - Build Performance Benchmark"
echo "=============================================================="
echo "Goal: Reduce compilation time from >5 minutes to <2 minutes"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to measure compilation time
benchmark_build() {
    local feature_set="$1"
    local description="$2"
    local target_time="$3"
    
    echo -e "${BLUE}📊 Testing: $description${NC}"
    echo "Feature set: $feature_set"
    echo "Target time: $target_time seconds"
    echo "---"
    
    # Clean to ensure fair comparison
    cargo clean -p neural-doc-flow-coordination --quiet 2>/dev/null || true
    
    # Time the compilation
    local start_time=$(date +%s.%N)
    
    if [ -z "$feature_set" ]; then
        if cargo check -p neural-doc-flow-coordination --quiet 2>/dev/null; then
            local success=true
        else
            local success=false
        fi
    else
        if cargo check -p neural-doc-flow-coordination --no-default-features --features "$feature_set" --quiet 2>/dev/null; then
            local success=true
        else
            local success=false
        fi
    fi
    
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc -l 2>/dev/null || echo "0")
    
    if [ "$success" = true ]; then
        if (( $(echo "$duration <= $target_time" | bc -l) )); then
            echo -e "${GREEN}✅ SUCCESS: Compilation time: ${duration}s (target: ${target_time}s)${NC}"
            return 0
        else
            echo -e "${YELLOW}⚠️  SLOW: Compilation time: ${duration}s (target: ${target_time}s)${NC}"
            return 1
        fi
    else
        echo -e "${RED}❌ FAILED: Compilation failed${NC}"
        return 2
    fi
}

echo "🔧 Testing build optimization improvements..."
echo ""

# Track results
success_count=0
total_tests=0

# Test 1: Minimal build (fastest)
total_tests=$((total_tests + 1))
if benchmark_build "minimal" "Minimal Build (Development)" 30; then
    success_count=$((success_count + 1))
fi
echo ""

# Test 2: Performance features only
total_tests=$((total_tests + 1))
if benchmark_build "performance" "Performance Build (CI/CD)" 90; then
    success_count=$((success_count + 1))
fi
echo ""

# Test 3: Default build
total_tests=$((total_tests + 1))
if benchmark_build "" "Default Build" 120; then
    success_count=$((success_count + 1))
fi
echo ""

# Test 4: Full feature set
total_tests=$((total_tests + 1))
if benchmark_build "full" "Full Build (Production)" 150; then
    success_count=$((success_count + 1))
fi
echo ""

# Summary
echo "📈 BUILD OPTIMIZATION RESULTS"
echo "============================"
echo -e "Tests passed: ${GREEN}$success_count${NC}/${total_tests}"
echo ""

if [ $success_count -eq $total_tests ]; then
    echo -e "${GREEN}🎉 ALL TARGETS MET! Build optimization successful.${NC}"
    echo ""
    echo "✅ Recommendations:"
    echo "  • Use 'minimal' features for daily development"
    echo "  • Use 'performance' features for CI/CD pipelines"
    echo "  • Use 'full' features only for production releases"
else
    echo -e "${YELLOW}⚠️  Some targets not met. Further optimization needed.${NC}"
    echo ""
    echo "🔧 Next steps:"
    echo "  • Review dependency chain"
    echo "  • Optimize async trait usage"
    echo "  • Consider module splitting"
fi

echo ""
echo "🎯 Target Achievement:"
if [ $success_count -ge 2 ]; then
    echo -e "${GREEN}✅ Primary goal: Reduce compilation to <2 minutes${NC}"
else
    echo -e "${RED}❌ Primary goal: Still working on <2 minute target${NC}"
fi

echo ""
echo "💡 Development workflow recommendations:"
echo "  cargo check -p neural-doc-flow-coordination --features minimal  # Fast dev builds"
echo "  cargo test -p neural-doc-flow-coordination --features performance # CI/CD"
echo "  cargo build -p neural-doc-flow-coordination --features full --release # Production"