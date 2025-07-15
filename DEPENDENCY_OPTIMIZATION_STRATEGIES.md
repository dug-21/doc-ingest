# Module-Specific Dependency Optimization Strategies

## Implementation Guide for Build Timeout Resolution

### 🎯 Module: neural-doc-flow-coordination

**Current Compilation Time**: >5 minutes  
**Target Time**: <1.5 minutes  
**Reduction Strategy**: Feature-gated performance dependencies

#### Immediate Fixes (Day 1)
```toml
[features]
# CRITICAL: Change default from ["performance"] to minimal
default = ["minimal"]
minimal = []  # Core coordination only
performance = ["rayon", "crossbeam", "parking_lot", "dashmap"]
monitoring = ["metrics"]
```

#### Dependency Reduction Actions
1. **Remove from default**: rayon, crossbeam, parking_lot (saves ~2 min compile time)
2. **Lazy agent loading**: Load coordination agents only when needed
3. **Optional dashmap**: Use std::HashMap for small coordinator instances

#### Code Changes Required
```rust
// In coordination/lib.rs - conditional compilation
#[cfg(feature = "performance")]
use rayon::prelude::*;

#[cfg(not(feature = "performance"))]  
use std::iter::Iterator as ParallelIterator;
```

### 🎯 Module: neural-doc-flow-api

**Current Compilation Time**: >5 minutes  
**Target Time**: <2 minutes  
**Reduction Strategy**: Modular web stack with optional features

#### Critical Fix (Immediate)
```toml
[dev-dependencies]
tempfile.workspace = true
test-log.workspace = true
# REMOVE optional = true from dev-dependencies
reqwest = { version = "0.11", features = ["json", "multipart"] }
tokio-test = "0.4"
```

#### Feature Restructuring
```toml
[features]
default = ["minimal"]
minimal = ["axum", "serde", "tokio/rt-multi-thread"]
database = ["sqlx"]  # Optional: saves ~90 seconds
auth = ["jsonwebtoken", "argon2", "base64"]  # Optional: saves ~30 seconds  
metrics = ["prometheus", "metrics"]  # Optional: saves ~45 seconds
background-jobs = ["tokio-cron-scheduler"]  # Optional: saves ~20 seconds
docs = ["utoipa", "utoipa-swagger-ui"]  # Optional: saves ~25 seconds
full = ["database", "auth", "metrics", "background-jobs", "docs"]
```

#### Compilation Unit Reduction
1. **Conditional middleware**: Only compile auth middleware if auth feature enabled
2. **Database abstraction**: Use trait objects to avoid SQLx compilation
3. **Swagger optional**: OpenAPI generation behind feature flag

### 🎯 Module: neural-doc-flow-wasm

**Current Compilation Time**: >5 minutes  
**Target Time**: <1 minute  
**Reduction Strategy**: Target-specific dependency isolation

#### WASM-Specific Optimization
```toml
[features]
default = ["minimal-runtime"]
minimal-runtime = ["console_error_panic_hook"]
full-runtime = ["minimal-runtime", "neural-processing"]

# Completely separate dependency chains
[target.'cfg(target_arch = "wasm32")'.dependencies]
tokio = { version = "1.46", features = ["macros", "sync"], default-features = false }
wasm-bindgen = "0.2"
console_error_panic_hook = "0.1"

[target.'cfg(not(target_arch = "wasm32"))'.dependencies]
neural-doc-flow-processors = { path = "../neural-doc-flow-processors", optional = true }
neural-doc-flow-security = { path = "../neural-doc-flow-security", optional = true }
tokio = { workspace = true }
```

#### Build Strategy Changes
1. **Conditional neural imports**: Only for native targets
2. **WASM-only minimal core**: Text processing without heavy ML
3. **Separate build profiles**: Different optimization for WASM vs native

### 🎯 Module: neural-doc-flow-sources

**Current Compilation Time**: >5 minutes  
**Target Time**: <1.5 minutes  
**Reduction Strategy**: Format-specific feature gating

#### Format-Based Feature Structure
```toml
[features]
default = ["text"]  # Minimal: text files only
text = []  # Built-in text processing
pdf = ["pdf-extract", "lopdf"]  # Heavy: saves ~180 seconds
web = ["reqwest", "scraper"]    # Network: saves ~60 seconds  
html = ["scraper"]              # Parsing: saves ~30 seconds
markdown = ["pulldown-cmark"]   # Rendering: saves ~15 seconds
docx = ["zip", "quick-xml"]     # Archive: saves ~45 seconds
all-formats = ["pdf", "web", "html", "markdown", "docx"]
```

#### Lazy Loading Strategy
```rust
// Conditional compilation by format
#[cfg(feature = "pdf")]
pub mod pdf;

#[cfg(feature = "web")]  
pub mod web_scraper;

// Runtime format detection with optional loading
pub fn load_source_for_format(format: &str) -> Result<Box<dyn DocumentSource>> {
    match format {
        #[cfg(feature = "pdf")]
        "pdf" => Ok(Box::new(pdf::PdfSource::new())),
        #[cfg(not(feature = "pdf"))]
        "pdf" => Err(anyhow!("PDF support not compiled in")),
        // ... other formats
    }
}
```

### 🎯 Module: neural-doc-flow-python

**Current Compilation Time**: >5 minutes  
**Target Time**: <2.5 minutes  
**Reduction Strategy**: Selective Python binding compilation

#### PyO3 Optimization
```toml
[features]
default = ["minimal"]
minimal = ["pyo3"]  # Core bindings only
neural = ["neural-doc-flow-processors/neural"]  # Optional: saves ~120 seconds
plugins = ["neural-doc-flow-plugins"]           # Optional: saves ~90 seconds  
security = ["neural-doc-flow-security"]         # Optional: saves ~60 seconds
all = ["neural", "plugins", "security"]
```

#### Selective Module Binding
```rust
// Conditional PyO3 modules
#[cfg(feature = "neural")]
#[pymodule]
fn neural_processing(_py: Python, m: &PyModule) -> PyResult<()> {
    // Heavy neural processing bindings
}

#[pymodule]
fn neuraldocflow(py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Always available: core text processing
    m.add_class::<Document>()?;
    m.add_class::<DocumentEngine>()?;
    
    #[cfg(feature = "neural")]
    m.add_wrapped(wrap_pymodule!(neural_processing))?;
    
    #[cfg(feature = "plugins")]
    m.add_wrapped(wrap_pymodule!(plugin_manager))?;
    
    Ok(())
}
```

## Cross-Module Optimizations

### 1. Core Crate Slimming Strategy

**neural-doc-flow-core optimization**:
```toml
[features]
default = ["essential"]
essential = ["async-trait", "serde", "tokio/macros"]  # Minimal async
performance = ["rayon", "crossbeam", "parking_lot"]   # Optional parallelism
neural = ["ndarray", "ruv-fann"]                      # Optional ML  
monitoring = ["metrics", "opentelemetry"]             # Optional observability
simd = ["wide", "bytemuck"]                          # Optional SIMD
full = ["performance", "neural", "monitoring", "simd"]
```

### 2. Workspace Build Configuration

**Optimized .cargo/config.toml**:
```toml
[build]
target-dir = "target"
incremental = true
pipelining = true

[target.x86_64-unknown-linux-gnu]
rustflags = ["-C", "target-cpu=native", "-C", "codegen-units=1"]

[target.wasm32-unknown-unknown]
rustflags = ["-C", "opt-level=s", "-C", "codegen-units=1"]

[cargo-new]
vcs = "git"

[net]
git-fetch-with-cli = true
```

### 3. Parallel Build Strategy

**Multi-stage compilation approach**:
```bash
# Stage 1: Core infrastructure (parallel)
cargo build -p neural-doc-flow-core --features essential
cargo build -p neural-doc-flow-sources --features text  
cargo build -p neural-doc-flow-outputs

# Stage 2: Processing layer (depends on core)
cargo build -p neural-doc-flow-processors --features minimal
cargo build -p neural-doc-flow-security --features minimal

# Stage 3: Application layer (depends on processing)  
cargo build -p neural-doc-flow-api --features minimal
cargo build -p neural-doc-flow-coordination --features minimal

# Stage 4: Bindings (parallel, optional)
cargo build -p neural-doc-flow-wasm --features minimal-runtime &
cargo build -p neural-doc-flow-python --features minimal &
wait
```

## Implementation Timeline

### Week 1: Critical Path (Build Breakage Fixes)
- [ ] **Day 1**: Fix neural-doc-flow-api dev-dependencies
- [ ] **Day 2**: Implement minimal defaults across all modules
- [ ] **Day 3**: Test compilation with new feature flags
- [ ] **Day 4**: Update CI/CD for feature-specific builds
- [ ] **Day 5**: Documentation for new build configurations

### Week 2: Performance Optimizations
- [ ] **Days 1-2**: Implement conditional compilation for heavy deps
- [ ] **Days 3-4**: WASM target-specific optimization  
- [ ] **Day 5**: PyO3 selective binding implementation

### Week 3: Validation & Monitoring
- [ ] **Days 1-2**: Build time measurement and validation
- [ ] **Days 3-4**: Performance regression testing
- [ ] **Day 5**: Documentation and developer guides

## Success Metrics

### Build Time Targets
- **neural-doc-flow-coordination**: 5+ min → 1.5 min (70% reduction)
- **neural-doc-flow-api**: 5+ min → 2 min (60% reduction)
- **neural-doc-flow-wasm**: 5+ min → 1 min (80% reduction)  
- **neural-doc-flow-sources**: 5+ min → 1.5 min (70% reduction)
- **neural-doc-flow-python**: 5+ min → 2.5 min (50% reduction)

### Resource Optimization
- **Parallel compilation units**: Reduce from 16 to 4-8
- **Peak memory usage**: 30-40% reduction during builds
- **Incremental rebuilds**: 5-10x faster for feature changes

### Developer Experience
- **Clear feature documentation**: Developers know what features to enable
- **Faster iteration cycles**: Minimal builds for rapid development
- **Optional heavy features**: Full functionality available when needed

---

**Ready for Implementation**: All strategies defined with specific code changes, feature flags, and measurable success criteria.