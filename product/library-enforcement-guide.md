# Library Enforcement Guide for NeuralDocFlow

## Purpose

This guide provides concrete mechanisms to PREVENT agents from writing code that duplicates functionality already available in claude-flow@alpha, ruv-FANN, and DAA libraries.

## Enforcement Mechanisms

### 1. Pre-Commit Hooks

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Check for prohibited patterns that indicate reimplementation
PROHIBITED_PATTERNS=(
    "struct.*Swarm"
    "struct.*Agent.*Manager"
    "struct.*Coordinator"
    "impl.*Neural.*Network"
    "struct.*Network.*Layer"
    "fn.*forward.*propagation"
    "struct.*Distributed.*System"
    "impl.*Consensus"
    "struct.*Memory.*Store"
)

for pattern in "${PROHIBITED_PATTERNS[@]}"; do
    if grep -r "$pattern" --include="*.rs" src/; then
        echo "❌ ERROR: Found prohibited pattern: $pattern"
        echo "Use existing libraries instead:"
        echo "  - Coordination: claude-flow@alpha"
        echo "  - Neural: ruv-FANN"
        echo "  - Distributed: DAA"
        exit 1
    fi
done

# Verify library usage
if ! grep -q "claude_flow::" src/**/*.rs; then
    echo "⚠️  WARNING: No claude-flow usage detected"
fi

if ! grep -q "ruv_fann::" src/**/*.rs; then
    echo "⚠️  WARNING: No ruv-FANN usage detected"
fi

if ! grep -q "daa::" src/**/*.rs; then
    echo "⚠️  WARNING: No DAA usage detected"
fi
```

### 2. Cargo.toml Restrictions

```toml
# Cargo.toml
[package]
name = "neuraldocflow"
version = "0.1.0"
edition = "2021"

# MANDATORY DEPENDENCIES - DO NOT REMOVE
[dependencies]
# Coordination - MUST use for all swarm/agent/memory operations
claude-flow = { version = "2.0.0-alpha", features = ["required"] }

# Neural Processing - MUST use for all AI/ML operations  
ruv-fann = { version = "0.7.0", features = ["required"] }

# Distributed - MUST use for all distributed operations
daa = { version = "0.1.0", features = ["required"] }

# PROHIBITED DEPENDENCIES - Will fail build if added
[build-dependencies]
neuraldocflow-lint = { path = "./tools/lint" }

# The lint tool checks for these and fails the build:
# ❌ tokio-consensus - Use DAA instead
# ❌ neural-network - Use ruv-FANN instead  
# ❌ custom-swarm - Use claude-flow instead
# ❌ Any crate with "neural", "swarm", "consensus" in name
```

### 3. CI/CD Pipeline Enforcement

```yaml
# .github/workflows/library-enforcement.yml
name: Library Enforcement

on: [push, pull_request]

jobs:
  enforce-libraries:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Check for Library Usage
        run: |
          # Count library imports
          CLAUDE_FLOW_USAGE=$(grep -r "use claude_flow" src/ | wc -l)
          RUV_FANN_USAGE=$(grep -r "use ruv_fann" src/ | wc -l)
          DAA_USAGE=$(grep -r "use daa" src/ | wc -l)
          
          echo "Library usage statistics:"
          echo "  claude-flow: $CLAUDE_FLOW_USAGE imports"
          echo "  ruv-FANN: $RUV_FANN_USAGE imports"
          echo "  DAA: $DAA_USAGE imports"
          
          # Fail if libraries aren't used
          if [ "$CLAUDE_FLOW_USAGE" -eq 0 ]; then
            echo "❌ ERROR: claude-flow must be used for coordination"
            exit 1
          fi
          
      - name: Check for Prohibited Implementations
        run: |
          # List of files that should NOT exist
          PROHIBITED_FILES=(
            "swarm_manager.rs"
            "agent_coordinator.rs"
            "neural_network.rs"
            "consensus_engine.rs"
            "memory_store.rs"
            "distributed_system.rs"
          )
          
          for file in "${PROHIBITED_FILES[@]}"; do
            if find src/ -name "$file" | grep -q .; then
              echo "❌ ERROR: Found prohibited file: $file"
              echo "This functionality should use libraries instead"
              exit 1
            fi
          done
          
      - name: Analyze Code Patterns
        run: |
          cargo install neuraldocflow-lint
          neuraldocflow-lint analyze --strict
```

### 4. Code Review Checklist

```markdown
# Code Review Checklist for Library Enforcement

## Before Approving Any PR:

### Coordination Code
- [ ] ❌ NO custom swarm implementations
- [ ] ❌ NO custom agent managers
- [ ] ❌ NO custom task orchestrators
- [ ] ✅ All coordination uses `claude_flow::*`
- [ ] ✅ Memory operations use `claude_flow::memory_usage`
- [ ] ✅ Session management uses `claude_flow::session_*`

### Neural/ML Code
- [ ] ❌ NO custom neural network implementations
- [ ] ❌ NO manual matrix multiplication for ML
- [ ] ❌ NO custom activation functions
- [ ] ✅ All neural ops use `ruv_fann::*`
- [ ] ✅ Pattern matching uses `ruv_fann::PatternMatcher`
- [ ] ✅ WASM acceleration uses `ruv_fann::wasm::*`

### Distributed Code
- [ ] ❌ NO custom distributed protocols
- [ ] ❌ NO manual consensus implementations
- [ ] ❌ NO custom fault tolerance
- [ ] ✅ All distributed ops use `daa::*`
- [ ] ✅ Agent creation uses `daa::agent_create`
- [ ] ✅ Communication uses `daa::communication`

## Red Flags to Reject:
1. Files named: `*_manager.rs`, `*_coordinator.rs`, `*_network.rs`
2. Structs containing: `agents: Vec<_>`, `neurons: Vec<_>`, `nodes: Vec<_>`
3. Functions named: `spawn_agent()`, `train_network()`, `reach_consensus()`
4. Any networking code that isn't using DAA
5. Any file I/O for persistence that isn't using claude-flow
```

### 5. Automated Code Generation Templates

```rust
// tools/templates/new_component.rs.template
use claude_flow::{SwarmInit, AgentSpawn, MemoryUsage};
use ruv_fann::{Network, PatternMatcher};
use daa::{Agent, ResourceAlloc};

/// Template for new components - MUST use libraries
pub struct {{ComponentName}} {
    // REQUIRED: Claude Flow integration
    swarm_id: String,
    memory_namespace: String,
    
    // REQUIRED: RUV-FANN integration (if ML needed)
    #[cfg(feature = "ml")]
    neural_model: ruv_fann::Network,
    
    // REQUIRED: DAA integration (if distributed)
    #[cfg(feature = "distributed")]
    agents: Vec<daa::Agent>,
}

impl {{ComponentName}} {
    pub async fn new() -> Result<Self> {
        // MUST initialize with claude-flow
        let swarm_id = claude_flow::swarm_init(/* config */).await?;
        
        // MUST use library functions only
        Ok(Self {
            swarm_id,
            memory_namespace: "{{component_name}}".to_string(),
            #[cfg(feature = "ml")]
            neural_model: ruv_fann::Network::new(/* config */)?,
            #[cfg(feature = "distributed")]
            agents: vec![],
        })
    }
}
```

### 6. Build-Time Enforcement

```rust
// build.rs
use std::env;
use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=src/");
    
    // Scan for prohibited implementations
    let prohibited_indicators = vec![
        ("impl.*Swarm", "Use claude-flow instead of custom swarm"),
        ("struct.*Neural.*Network", "Use ruv-FANN instead of custom neural network"),
        ("impl.*Consensus", "Use DAA instead of custom consensus"),
    ];
    
    let src_dir = Path::new("src");
    for entry in walkdir::WalkDir::new(src_dir) {
        let entry = entry.unwrap();
        if entry.path().extension() == Some("rs".as_ref()) {
            let content = fs::read_to_string(entry.path()).unwrap();
            
            for (pattern, message) in &prohibited_indicators {
                if regex::Regex::new(pattern).unwrap().is_match(&content) {
                    panic!(
                        "❌ Build failed: {} in file {:?}\n{}",
                        pattern,
                        entry.path(),
                        message
                    );
                }
            }
        }
    }
    
    // Verify library dependencies
    let cargo_toml = fs::read_to_string("Cargo.toml").unwrap();
    if !cargo_toml.contains("claude-flow") {
        panic!("❌ Build failed: claude-flow is required in dependencies");
    }
    if !cargo_toml.contains("ruv-fann") {
        panic!("❌ Build failed: ruv-fann is required in dependencies");
    }
    if !cargo_toml.contains("daa") {
        panic!("❌ Build failed: daa is required in dependencies");
    }
}
```

### 7. IDE Integration

```json
// .vscode/settings.json
{
  "rust-analyzer.diagnostics.disabled": ["unresolved-import"],
  "rust-analyzer.checkOnSave.command": "clippy",
  "rust-analyzer.checkOnSave.extraArgs": [
    "--",
    "-D",
    "clippy::disallowed_method",
    "-D",
    "clippy::disallowed_type"
  ],
  
  // Custom problem matchers
  "problemMatcher": {
    "pattern": {
      "regexp": "^(.*): (custom implementation detected)(.*)$",
      "file": 1,
      "message": 3,
      "severity": "error"
    }
  },
  
  // Snippets that enforce library usage
  "rust.snippets": {
    "swarm": {
      "prefix": "swarm",
      "body": [
        "let swarm_id = claude_flow::swarm_init(SwarmConfig {",
        "    topology: Topology::${1:Hierarchical},",
        "    max_agents: ${2:8},",
        "    strategy: Strategy::${3:Auto},",
        "}).await?;"
      ],
      "description": "Initialize swarm with claude-flow"
    },
    "neural": {
      "prefix": "neural",
      "body": [
        "let network = ruv_fann::Network::new(&[${1:768}, ${2:256}, ${3:10}])?;",
        "network.set_activation_func(ActivationFunc::${4:ReLU});"
      ],
      "description": "Create neural network with ruv-FANN"
    }
  }
}
```

### 8. Documentation Enforcement

```rust
// src/lib.rs - Required documentation header
#![doc = include_str!("../LIBRARY_USAGE.md")]
#![warn(missing_docs)]

//! # NeuralDocFlow
//! 
//! ## Library Dependencies
//! 
//! This crate REQUIRES the following libraries:
//! - `claude-flow` - ALL coordination, swarm, and memory operations
//! - `ruv-fann` - ALL neural network and ML operations
//! - `daa` - ALL distributed system operations
//! 
//! ## Prohibited Implementations
//! 
//! The following will cause build failures:
//! - Custom swarm or agent implementations
//! - Custom neural network implementations
//! - Custom distributed consensus
//! - Custom memory/persistence layers
//! 
//! Always use the library functions instead!

// Enforce at module level
#[cfg(not(all(
    feature = "claude-flow",
    feature = "ruv-fann", 
    feature = "daa"
)))]
compile_error!("All three libraries (claude-flow, ruv-fann, daa) are required!");
```

### 9. Testing Enforcement

```rust
// tests/library_enforcement.rs
#[test]
fn verify_no_custom_implementations() {
    // Scan codebase
    let violations = scan_for_violations("src/");
    
    if !violations.is_empty() {
        panic!(
            "Found {} violations of library-first architecture:\n{}",
            violations.len(),
            violations.join("\n")
        );
    }
}

#[test]
fn verify_library_usage() {
    let usage = analyze_library_usage("src/");
    
    assert!(usage.claude_flow_count > 0, "Must use claude-flow");
    assert!(usage.ruv_fann_count > 0, "Must use ruv-FANN");
    assert!(usage.daa_count > 0, "Must use DAA");
    
    // Verify ratio - libraries should be used more than custom code
    let total_imports = usage.claude_flow_count + usage.ruv_fann_count + usage.daa_count;
    let custom_code_lines = count_custom_implementations("src/");
    
    assert!(
        total_imports > custom_code_lines / 10,
        "Too much custom code - use more library functions!"
    );
}
```

### 10. Developer Onboarding

```markdown
# New Developer Checklist

Before writing ANY code for NeuralDocFlow:

1. **Read the library documentation:**
   - [ ] claude-flow@alpha docs: https://github.com/ruvnet/claude-flow
   - [ ] ruv-FANN docs: https://github.com/ruvnet/ruv-fann
   - [ ] DAA docs: [internal link]

2. **Understand what NOT to implement:**
   - [ ] NO custom swarms, agents, or coordinators
   - [ ] NO custom neural networks or ML algorithms
   - [ ] NO custom distributed systems or consensus
   - [ ] NO custom memory stores or persistence

3. **Install enforcement tools:**
   ```bash
   cargo install neuraldocflow-lint
   code --install-extension neuraldocflow.library-enforcer
   ```

4. **Run pre-flight check:**
   ```bash
   neuraldocflow-lint check-setup
   ```

5. **Use code generators:**
   ```bash
   neuraldocflow-cli new component --name MyComponent
   # This generates library-compliant code templates
   ```
```

## Summary

These enforcement mechanisms ensure:

1. **Build-time failures** for custom implementations
2. **CI/CD rejection** of non-compliant code
3. **IDE warnings** when writing prohibited patterns
4. **Automated templates** that use libraries correctly
5. **Code review checklists** to catch violations
6. **Developer education** from day one

By implementing these mechanisms, it becomes nearly impossible for agents or developers to accidentally reimplement functionality that already exists in the required libraries.