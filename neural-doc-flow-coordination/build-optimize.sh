#!/bin/bash
# Build optimization script for neural-doc-flow-coordination

set -e

echo "🚀 Neural Doc Flow Coordination - Build Optimization Script"
echo "=========================================================="

# Function to time compilation
time_compilation() {
    local feature_set="$1"
    local description="$2"
    
    echo ""
    echo "📊 Testing compilation time for: $description"
    echo "Feature set: $feature_set"
    echo "---"
    
    # Clean to ensure fair comparison
    cargo clean -p neural-doc-flow-coordination --quiet
    
    # Time the compilation
    local start_time=$(date +%s.%N)
    
    if [ -z "$feature_set" ]; then
        cargo check -p neural-doc-flow-coordination --quiet
    else
        cargo check -p neural-doc-flow-coordination --features "$feature_set" --quiet
    fi
    
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc -l)
    
    printf "✅ Compilation time: %.2f seconds\n" "$duration"
    return 0
}

# Test different feature combinations
echo "🔧 Testing compilation times for different feature sets..."

# 1. Minimal build (fastest)
time_compilation "" "Minimal build (default)"

# 2. Performance features only
time_compilation "performance" "Performance features"

# 3. Messaging features only  
time_compilation "messaging" "Messaging features"

# 4. Analytics features only
time_compilation "analytics" "Analytics features"

# 5. Full feature set
time_compilation "full" "Full feature set"

echo ""
echo "🎯 Build optimization recommendations:"
echo "- Use 'minimal' features for fastest development builds"
echo "- Use 'performance' for production coordination"
echo "- Use 'full' only when all features are needed"
echo ""

# Test incremental compilation
echo "🔄 Testing incremental compilation..."
echo "Making a small change and rebuilding..."

# Touch a file to trigger incremental build
touch agents/mod.rs

start_time=$(date +%s.%N)
cargo check -p neural-doc-flow-coordination --quiet
end_time=$(date +%s.%N)
duration=$(echo "$end_time - $start_time" | bc -l)

printf "✅ Incremental build time: %.2f seconds\n" "$duration"

echo ""
echo "✨ Optimization complete! Use the minimal feature set for fastest builds."