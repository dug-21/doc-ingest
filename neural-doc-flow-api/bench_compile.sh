#!/bin/bash
#
# Compilation Benchmark Script
# Tests different feature configurations for build time optimization
#

set -e

echo "🔥 Neural Document Flow API - Compilation Benchmark"
echo "Testing different feature configurations for build optimization..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to time compilation
time_compilation() {
    local features="$1"
    local description="$2"
    
    echo -e "${BLUE}📦 Testing: $description${NC}"
    echo -e "${YELLOW}Features: $features${NC}"
    
    # Clean previous build
    cargo clean -q
    
    # Time the compilation
    start_time=$(date +%s)
    if [ -z "$features" ]; then
        cargo check --quiet 2>/dev/null
    else
        cargo check --no-default-features --features "$features" --quiet 2>/dev/null
    fi
    end_time=$(date +%s)
    
    compile_time=$((end_time - start_time))
    echo -e "${GREEN}✅ Compilation time: ${compile_time}s${NC}"
    echo ""
    
    return $compile_time
}

# Create results file
results_file="compilation_benchmark_results.txt"
echo "Neural Document Flow API - Compilation Benchmark Results" > $results_file
echo "Generated: $(date)" >> $results_file
echo "==========================================================" >> $results_file
echo "" >> $results_file

# Test different configurations
echo "🚀 Starting compilation benchmarks..."
echo ""

# Test 1: Minimal build (fastest)
echo "TEST 1: Minimal Configuration (Core API only)" >> $results_file
time_compilation "minimal" "Minimal API (axum + serde only)"
minimal_time=$?
echo "Minimal build: ${minimal_time}s" >> $results_file
echo "" >> $results_file

# Test 2: API with auth
echo "TEST 2: API + Authentication" >> $results_file
time_compilation "minimal,auth" "API with Authentication"
auth_time=$?
echo "API + Auth: ${auth_time}s" >> $results_file
echo "" >> $results_file

# Test 3: API with database
echo "TEST 3: API + Database" >> $results_file
time_compilation "minimal,database" "API with Database Support"
db_time=$?
echo "API + Database: ${db_time}s" >> $results_file
echo "" >> $results_file

# Test 4: API with docs
echo "TEST 4: API + Documentation" >> $results_file
time_compilation "minimal,docs" "API with OpenAPI Documentation"
docs_time=$?
echo "API + Docs: ${docs_time}s" >> $results_file
echo "" >> $results_file

# Test 5: Security features
echo "TEST 5: API + Security" >> $results_file
time_compilation "minimal,security" "API with Security Features"
security_time=$?
echo "API + Security: ${security_time}s" >> $results_file
echo "" >> $results_file

# Test 6: Full build (slowest)
echo "TEST 6: Full Configuration" >> $results_file
time_compilation "full" "Full Feature Set"
full_time=$?
echo "Full build: ${full_time}s" >> $results_file
echo "" >> $results_file

# Test 7: Default build (for comparison)
echo "TEST 7: Default Configuration" >> $results_file
time_compilation "" "Default Features"
default_time=$?
echo "Default build: ${default_time}s" >> $results_file
echo "" >> $results_file

# Calculate improvements
echo "📊 PERFORMANCE ANALYSIS" >> $results_file
echo "======================" >> $results_file
if [ $full_time -gt 0 ]; then
    minimal_improvement=$(echo "scale=1; ($full_time - $minimal_time) * 100 / $full_time" | bc -l)
    echo "Minimal vs Full: ${minimal_improvement}% faster" >> $results_file
fi

if [ $default_time -gt 0 ]; then
    default_improvement=$(echo "scale=1; ($default_time - $minimal_time) * 100 / $default_time" | bc -l)
    echo "Minimal vs Default: ${default_improvement}% faster" >> $results_file
fi

echo "" >> $results_file
echo "RECOMMENDATIONS" >> $results_file
echo "===============" >> $results_file
echo "• Use 'minimal' feature for development iterations" >> $results_file
echo "• Use 'full' feature for production builds" >> $results_file
echo "• Selective features for specific use cases" >> $results_file

# Display summary
echo -e "${GREEN}📊 BENCHMARK COMPLETE!${NC}"
echo ""
echo -e "${YELLOW}🔥 Results Summary:${NC}"
echo -e "   Minimal build:  ${minimal_time}s"
echo -e "   With Auth:      ${auth_time}s"
echo -e "   With Database:  ${db_time}s"
echo -e "   With Docs:      ${docs_time}s"
echo -e "   With Security:  ${security_time}s"
echo -e "   Full build:     ${full_time}s"
echo -e "   Default build:  ${default_time}s"
echo ""

if [ $full_time -gt 0 ] && [ $minimal_time -gt 0 ]; then
    improvement=$(echo "scale=1; ($full_time - $minimal_time) * 100 / $full_time" | bc -l)
    if (( $(echo "$improvement > 30" | bc -l) )); then
        echo -e "${GREEN}🎯 OPTIMIZATION SUCCESS: ${improvement}% faster compilation!${NC}"
    else
        echo -e "${YELLOW}⚠️  Moderate improvement: ${improvement}% faster${NC}"
    fi
fi

echo ""
echo -e "${BLUE}📋 Full results saved to: $results_file${NC}"
echo ""
echo -e "${GREEN}✅ Benchmark complete! Use selective features for optimal build times.${NC}"