# Library Usage Validation Framework

## 🎯 Purpose

This framework ensures that NeuralDocFlow implementation strictly uses the required libraries (claude-flow@alpha, ruv-FANN, DAA) and prevents developers from reimplementing existing functionality.

## 🚨 Critical Validation Rules

### 1. claude-flow@alpha Usage Rules

#### MUST Use claude-flow@alpha For:
- ✅ ALL swarm coordination and orchestration
- ✅ ALL memory persistence and retrieval
- ✅ ALL inter-agent communication via hooks
- ✅ ALL task distribution and monitoring
- ✅ ALL performance tracking and metrics

#### FORBIDDEN Patterns:
```rust
// ❌ NEVER implement custom coordinators
struct MyCoordinator { ... }
impl Coordinator for MyCoordinator { ... }

// ❌ NEVER implement custom memory
struct MemoryStore { ... }
impl Storage for MemoryStore { ... }

// ❌ NEVER implement custom hooks
fn my_hook_system() { ... }
```

#### Validation Test:
```bash
# This command MUST return results
grep -r "npx claude-flow@alpha" . || exit 1

# This command MUST NOT return results  
grep -r "impl.*Coordinator\|struct.*Coordinator" src/ && exit 1
```

### 2. ruv-FANN Usage Rules

#### MUST Use ruv-FANN For:
- ✅ ALL neural network creation and configuration
- ✅ ALL training and backpropagation
- ✅ ALL inference and prediction
- ✅ ALL activation functions and layers
- ✅ ALL model serialization

#### FORBIDDEN Patterns:
```rust
// ❌ NEVER implement neural networks
struct NeuralNetwork {
    layers: Vec<Layer>,
    weights: Vec<Matrix>,
}

// ❌ NEVER implement training algorithms
fn backpropagate(&mut self, error: f32) { ... }

// ❌ NEVER implement custom ML
impl Model for CustomModel { ... }
```

#### Validation Test:
```javascript
// package.json MUST include
assert(dependencies["ruv-fann"] !== undefined);

// Code MUST NOT include
assert(!code.includes("class NeuralNetwork"));
assert(!code.includes("backpropagation"));
```

### 3. DAA Usage Rules

#### MUST Use DAA For:
- ✅ ALL distributed agent creation
- ✅ ALL parallel task execution
- ✅ ALL consensus mechanisms
- ✅ ALL fault tolerance
- ✅ ALL agent lifecycle management

#### FORBIDDEN Patterns:
```rust
// ❌ NEVER use raw threads
std::thread::spawn(|| { ... });

// ❌ NEVER implement distributed systems
struct DistributedWorker { ... }

// ❌ NEVER use Worker threads directly
const worker = new Worker(...);
```

#### Validation Test:
```bash
# Must use DAA
grep -r "import.*daa\|from daa" . || exit 1

# Must NOT use Workers
grep -r "new Worker\|thread::spawn" src/ && exit 1
```

## 📊 Phase-Specific Validation

### Phase 1: Foundation Validation
```yaml
validation:
  must_have_dependencies:
    - lopdf: "^0.32"
    - pdf-extract: "^0.7"
    - memmap2: "^0.9"
  
  forbidden_patterns:
    - pattern: "impl.*PdfParser"
      message: "Use lopdf instead of custom PDF parser"
    - pattern: "fn parse_pdf"
      message: "Use pdf-extract for text extraction"
  
  required_imports:
    - "use lopdf::Document"
    - "use pdf_extract::extract_text"
```

### Phase 2: Python Bindings Validation
```python
def validate_phase2():
    # Must use PyO3
    assert "pyo3" in Cargo.toml
    
    # Must use claude-flow coordination
    assert subprocess.run(["which", "claude-flow"]).returncode == 0
    
    # Must see coordination in logs
    logs = run_test_process()
    assert "claude-flow: swarm initialized" in logs
```

### Phase 3: Web Interface Validation
```typescript
// Must use these exact imports
import { Network } from 'ruv-fann';
import { NeuralDocFlow } from 'neuraldocflow-wasm';

// Must NOT have these
// ❌ class CustomNeuralNetwork
// ❌ function trainModel()
// ❌ implements NeuralInterface
```

### Phase 4: MCP Protocol Validation
```javascript
// Must use claude-flow MCP utilities
const validation = {
    requiredImports: [
        "import { createMCPServer } from 'claude-flow/mcp'",
        "import { MCPTool } from 'claude-flow/mcp'"
    ],
    forbiddenCode: [
        "class MCPServer",
        "implements MCPProtocol",
        "function handleMCPRequest"
    ]
};
```

### Phase 5: Neural Enhancement Validation
```javascript
// Validation rules for neural phase
const neuralValidation = {
    mustUse: {
        "ruv-fann": {
            classes: ["Network", "Trainer", "Layer"],
            functions: ["train", "predict", "save", "load"]
        }
    },
    mustNotImplement: [
        "forward propagation",
        "backward propagation",
        "gradient descent",
        "weight initialization"
    ]
};
```

### Phase 6: Autonomous Features Validation
```javascript
// DAA validation rules
const autonomousValidation = {
    requiredUsage: {
        "daa": ["Agent", "Swarm", "Consensus"],
        "claude-flow": ["SwarmCoordinator", "hooks"]
    },
    bannedPatterns: [
        /new Worker\(/,
        /thread::spawn/,
        /impl Distributed/,
        /struct Agent[^s]/
    ]
};
```

## 🔍 Automated Validation Tools

### 1. Pre-Commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit

echo "🔍 Validating library usage..."

# Check for forbidden implementations
FORBIDDEN_PATTERNS=(
    "impl.*Coordinator"
    "struct.*NeuralNetwork" 
    "class.*Agent[^s]"
    "new Worker"
    "thread::spawn"
)

for pattern in "${FORBIDDEN_PATTERNS[@]}"; do
    if grep -r "$pattern" src/; then
        echo "❌ ERROR: Forbidden pattern found: $pattern"
        echo "Use the required libraries instead!"
        exit 1
    fi
done

# Verify required dependencies
if ! grep -q "claude-flow@alpha" package.json; then
    echo "❌ ERROR: claude-flow@alpha not found in dependencies"
    exit 1
fi

echo "✅ Library validation passed!"
```

### 2. CI/CD Pipeline Validation
```yaml
# .github/workflows/validate-libraries.yml
name: Validate Library Usage

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Check Dependencies
        run: |
          # Rust dependencies
          grep -q "lopdf" Cargo.toml || exit 1
          grep -q "pyo3" neuraldocflow-py/Cargo.toml || exit 1
          
          # JavaScript dependencies  
          grep -q "claude-flow@alpha" package.json || exit 1
          grep -q "ruv-fann" package.json || exit 1
          grep -q "daa" package.json || exit 1
      
      - name: Scan for Anti-Patterns
        run: |
          # Run anti-pattern detector
          npm run validate:libraries
```

### 3. Runtime Validation
```rust
// src/validation.rs
pub fn validate_library_usage() -> Result<(), ValidationError> {
    // Check claude-flow is available
    if !command_exists("claude-flow") {
        return Err(ValidationError::MissingDependency("claude-flow@alpha"));
    }
    
    // Check ruv-FANN can be imported
    if !can_import("ruv-fann") {
        return Err(ValidationError::MissingDependency("ruv-fann"));
    }
    
    // Check DAA is functional
    if !can_import("daa") {
        return Err(ValidationError::MissingDependency("daa"));
    }
    
    Ok(())
}
```

## ✅ Success Metrics

### Quantitative Metrics
1. **Library Usage**: >95% of functionality from libraries
2. **Custom Code**: <5% (only glue code)
3. **Dependencies**: 100% of required libraries present
4. **Anti-patterns**: 0 instances detected
5. **Test Coverage**: 100% of library integrations tested

### Validation Checklist
- [ ] All Rust code uses lopdf for PDF parsing
- [ ] All Python bindings use PyO3 exclusively
- [ ] All coordination uses claude-flow@alpha
- [ ] All neural ops use ruv-FANN
- [ ] All distribution uses DAA
- [ ] Zero custom implementations of core features
- [ ] All phases pass validation tests
- [ ] CI/CD pipeline includes validation
- [ ] Pre-commit hooks active and working

## 🚫 Red Flags - Automatic Failures

These patterns trigger immediate validation failure:

1. **Custom Coordination**: Any `impl Coordinator` or swarm management
2. **Custom Neural**: Any neural network implementation
3. **Custom Distribution**: Any Worker threads or distribution code
4. **Missing Dependencies**: Required libraries not in package files
5. **Import Violations**: Direct implementations instead of library imports

## 📈 Continuous Monitoring

```javascript
// monitoring/library-usage.js
setInterval(async () => {
    const metrics = await collectLibraryUsageMetrics();
    
    if (metrics.customCodePercentage > 5) {
        alert("Warning: Excessive custom code detected!");
    }
    
    if (metrics.missingLibraries.length > 0) {
        alert(`Missing libraries: ${metrics.missingLibraries.join(', ')}`);
    }
    
    // Store in claude-flow memory for tracking
    await claudeFlow.memory.store('metrics/library-usage', metrics);
}, 3600000); // Check every hour
```