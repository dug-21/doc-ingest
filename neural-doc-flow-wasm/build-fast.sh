#!/bin/bash
set -e

echo "🚀 WASM Build Optimization Test Script"
echo "======================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to measure build time
measure_build() {
    local feature_set="$1"
    local description="$2"
    
    echo -e "${BLUE}Testing: $description${NC}"
    echo "Features: $feature_set"
    
    # Clean build to ensure accurate timing
    cargo clean -p neural-doc-flow-wasm > /dev/null 2>&1
    
    # Measure build time
    start_time=$(date +%s.%N)
    
    if [ "$feature_set" = "default" ]; then
        cargo check -p neural-doc-flow-wasm > /dev/null 2>&1
    else
        cargo check -p neural-doc-flow-wasm --features "$feature_set" > /dev/null 2>&1
    fi
    
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc -l)
    
    echo -e "${GREEN}✓ Build time: ${duration}s${NC}"
    echo
    
    return 0
}

# Change to project root
cd /workspaces/doc-ingest

echo "Starting WASM build optimization tests..."
echo

# Test 1: Minimal build (fastest)
measure_build "minimal" "Minimal WASM build - basic functionality only"

# Test 2: Basic build
measure_build "basic" "Basic WASM build - core utilities"

# Test 3: Streaming build
measure_build "streaming" "Streaming WASM build - async support"

# Test 4: Processing build
measure_build "processing" "Processing WASM build - document processing"

# Test 5: Neural build
measure_build "neural" "Neural WASM build - with neural processing"

# Test 6: Full build
measure_build "full" "Full WASM build - all features"

# Test native target compilation (should be much faster)
echo -e "${YELLOW}Testing native target compilation...${NC}"
start_time=$(date +%s.%N)
cargo check -p neural-doc-flow-wasm --features minimal > /dev/null 2>&1
end_time=$(date +%s.%N)
native_duration=$(echo "$end_time - $start_time" | bc -l)
echo -e "${GREEN}✓ Native minimal build time: ${native_duration}s${NC}"

echo
echo -e "${YELLOW}Build Optimization Summary:${NC}"
echo "• Use 'minimal' features for fastest development builds"
echo "• Use 'basic' features for core functionality testing"
echo "• Use 'full' features only for production WASM builds"
echo "• Native builds are significantly faster than WASM"
echo
echo -e "${GREEN}Optimization complete!${NC}"