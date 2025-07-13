# Rust Workspace Structure for Neural Document Flow Phase 1

## Overview

This document defines the Cargo workspace structure for NeuralDocFlow Phase 1, implementing the pure Rust architecture with DAA coordination and ruv-FANN neural processing as specified in iteration5.

## Workspace Layout

```
neuraldocflow/
├── Cargo.toml                    # Workspace root
├── Cargo.lock
├── README.md
├── LICENSE
├── .gitignore
├── docs/
│   ├── architecture/
│   ├── api/
│   └── examples/
├── models/                       # Neural model files
│   ├── layout/
│   ├── text/
│   ├── table/
│   └── quality/
├── schemas/                      # Extraction schemas
│   ├── default.json
│   ├── invoice.json
│   └── medical.json
├── plugins/                      # External source plugins
│   └── examples/
└── crates/
    ├── neuraldocflow-core/       # Core engine
    ├── neuraldocflow-daa/        # DAA coordination
    ├── neuraldocflow-neural/     # Neural processing
    ├── neuraldocflow-sources/    # Document sources
    ├── neuraldocflow-schema/     # Schema engine
    ├── neuraldocflow-output/     # Output formatting
    ├── neuraldocflow-cli/        # Command line interface
    ├── neuraldocflow-python/     # Python bindings
    ├── neuraldocflow-wasm/       # WASM bindings
    └── neuraldocflow-api/        # REST API server
```

## Root Cargo.toml

```toml
[workspace]
members = [
    "crates/neuraldocflow-core",
    "crates/neuraldocflow-daa",
    "crates/neuraldocflow-neural",
    "crates/neuraldocflow-sources",
    "crates/neuraldocflow-schema",
    "crates/neuraldocflow-output",
    "crates/neuraldocflow-cli",
    "crates/neuraldocflow-python",
    "crates/neuraldocflow-wasm",
    "crates/neuraldocflow-api",
]

resolver = "2"

[workspace.dependencies]
# Core dependencies
tokio = { version = "1.0", features = ["full"] }
async-trait = "0.1"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
uuid = { version = "1.0", features = ["v4"] }
anyhow = "1.0"
thiserror = "1.0"

# DAA and coordination
daa = { version = "0.1", path = "crates/neuraldocflow-daa" }
crossbeam = "0.8"
parking_lot = "0.12"

# Neural processing
ruv-fann = "0.1"
ndarray = "0.15"

# Document processing
pdf = "0.8"
zip = "0.6"
quick-xml = "0.31"
image = "0.24"

# Python bindings
pyo3 = { version = "0.20", optional = true }

# WASM bindings
wasm-bindgen = { version = "0.2", optional = true }
js-sys = { version = "0.3", optional = true }
web-sys = { version = "0.3", optional = true }

# CLI
clap = { version = "4.0", features = ["derive"] }

# Web server
actix-web = "4.0"

# Utilities
log = "0.4"
env_logger = "0.10"
config = "0.13"
notify = "5.0"

[profile.release]
lto = true
codegen-units = 1
panic = "abort"
strip = true

[profile.dev]
debug = true
opt-level = 0

[profile.test]
opt-level = 1
```

## Core Crate Structure

### neuraldocflow-core

```toml
[package]
name = "neuraldocflow-core"
version = "0.1.0"
edition = "2021"
description = "Core document processing engine"

[dependencies]
# Workspace dependencies
tokio = { workspace = true }
async-trait = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
uuid = { workspace = true }
anyhow = { workspace = true }
thiserror = { workspace = true }

# Local crate dependencies
neuraldocflow-daa = { path = "../neuraldocflow-daa" }
neuraldocflow-neural = { path = "../neuraldocflow-neural" }
neuraldocflow-sources = { path = "../neuraldocflow-sources" }
neuraldocflow-schema = { path = "../neuraldocflow-schema" }
neuraldocflow-output = { path = "../neuraldocflow-output" }

[lib]
name = "neuraldocflow_core"
crate-type = ["rlib"]
```

### neuraldocflow-daa

```toml
[package]
name = "neuraldocflow-daa"
version = "0.1.0"
edition = "2021"
description = "Distributed Autonomous Agents coordination"

[dependencies]
tokio = { workspace = true }
async-trait = { workspace = true }
serde = { workspace = true }
crossbeam = { workspace = true }
parking_lot = { workspace = true }
uuid = { workspace = true }
anyhow = { workspace = true }
thiserror = { workspace = true }

[lib]
name = "neuraldocflow_daa"
crate-type = ["rlib"]
```

### neuraldocflow-neural

```toml
[package]
name = "neuraldocflow-neural"
version = "0.1.0"
edition = "2021"
description = "Neural processing with ruv-FANN"

[dependencies]
ruv-fann = { workspace = true }
ndarray = { workspace = true }
tokio = { workspace = true }
serde = { workspace = true }
anyhow = { workspace = true }
thiserror = { workspace = true }

[features]
default = ["simd"]
simd = []

[lib]
name = "neuraldocflow_neural"
crate-type = ["rlib"]
```

### neuraldocflow-sources

```toml
[package]
name = "neuraldocflow-sources"
version = "0.1.0"
edition = "2021"
description = "Document source plugins"

[dependencies]
tokio = { workspace = true }
async-trait = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
anyhow = { workspace = true }
thiserror = { workspace = true }
notify = { workspace = true }

# Document format support
pdf = { workspace = true }
zip = { workspace = true }
quick-xml = { workspace = true }
image = { workspace = true }

# Dynamic loading
libloading = "0.8"

[lib]
name = "neuraldocflow_sources"
crate-type = ["rlib", "cdylib"]
```

### neuraldocflow-python

```toml
[package]
name = "neuraldocflow-python"
version = "0.1.0"
edition = "2021"
description = "Python bindings for NeuralDocFlow"

[dependencies]
pyo3 = { workspace = true, features = ["extension-module"] }
neuraldocflow-core = { path = "../neuraldocflow-core" }
tokio = { workspace = true }
serde = { workspace = true }

[lib]
name = "neuraldocflow"
crate-type = ["cdylib"]

[package.metadata.maturin]
python-source = "python"
module-name = "neuraldocflow._core"
```

### neuraldocflow-wasm

```toml
[package]
name = "neuraldocflow-wasm"
version = "0.1.0"
edition = "2021"
description = "WASM bindings for NeuralDocFlow"

[dependencies]
wasm-bindgen = { workspace = true }
js-sys = { workspace = true }
web-sys = { workspace = true }
serde-wasm-bindgen = "0.5"
neuraldocflow-core = { path = "../neuraldocflow-core" }
console_error_panic_hook = "0.1"
wasm-logger = "0.2"

[lib]
crate-type = ["cdylib"]

[package.metadata.wasm-pack.profile.release]
wasm-opt = false
```

## Directory Structure Details

### Models Directory

```
models/
├── layout/
│   ├── layout.fann           # Layout detection network
│   ├── layout.json           # Network metadata
│   └── training_data/        # Training datasets
├── text/
│   ├── text.fann             # Text enhancement network
│   ├── text.json
│   └── training_data/
├── table/
│   ├── table.fann            # Table detection network
│   ├── table.json
│   └── training_data/
├── quality/
│   ├── quality.fann          # Quality assessment network
│   ├── quality.json
│   └── training_data/
└── domain_specific/
    ├── legal/
    ├── medical/
    └── financial/
```

### Schemas Directory

```
schemas/
├── default.json              # Default extraction schema
├── invoice.json              # Invoice extraction schema
├── medical.json              # Medical record schema
├── legal/
│   ├── contract.json
│   ├── patent.json
│   └── court_filing.json
└── financial/
    ├── statement.json
    └── report.json
```

### Plugins Directory

```
plugins/
├── examples/
│   ├── custom_source.rs      # Example custom source
│   ├── plugin.toml           # Plugin manifest
│   └── README.md
├── docx_source.so            # DOCX plugin (compiled)
├── html_source.so            # HTML plugin (compiled)
└── manifest.json             # Plugin registry
```

## Build Configuration

### Cross-compilation Support

```toml
# .cargo/config.toml
[target.x86_64-pc-windows-gnu]
linker = "x86_64-w64-mingw32-gcc"

[target.x86_64-apple-darwin]
linker = "x86_64-apple-darwin20.4-clang"

[target.aarch64-apple-darwin]
linker = "aarch64-apple-darwin20.4-clang"

[target.x86_64-unknown-linux-musl]
linker = "x86_64-linux-musl-gcc"

[env]
PKG_CONFIG_ALLOW_CROSS = "1"
```

### Build Scripts

```bash
#!/bin/bash
# scripts/build.sh

set -euo pipefail

echo "Building NeuralDocFlow workspace..."

# Clean previous build
cargo clean

# Build core library
echo "Building core library..."
cargo build --release -p neuraldocflow-core

# Build CLI
echo "Building CLI..."
cargo build --release -p neuraldocflow-cli

# Build Python bindings
echo "Building Python bindings..."
cd crates/neuraldocflow-python
maturin build --release
cd ../..

# Build WASM bindings
echo "Building WASM bindings..."
cd crates/neuraldocflow-wasm
wasm-pack build --target web --release
cd ../..

# Build plugins
echo "Building plugins..."
cargo build --release -p neuraldocflow-sources

echo "Build complete!"
```

## Development Workflow

### Local Development

1. **Setup workspace**:
   ```bash
   git clone <repository>
   cd neuraldocflow
   cargo build
   ```

2. **Run tests**:
   ```bash
   cargo test --workspace
   ```

3. **Run specific crate**:
   ```bash
   cargo run -p neuraldocflow-cli -- --help
   ```

4. **Development with hot-reload**:
   ```bash
   cargo watch -x "test -p neuraldocflow-core"
   ```

### Plugin Development

1. **Create new plugin**:
   ```bash
   cd plugins
   cargo new --lib my_source_plugin
   ```

2. **Implement source trait**:
   ```rust
   // Implement DocumentSource trait
   ```

3. **Build plugin**:
   ```bash
   cargo build --release --crate-type cdylib
   ```

4. **Install plugin**:
   ```bash
   cp target/release/libmy_source_plugin.so plugins/
   ```

## Dependencies Management

### Version Pinning Strategy

- **Core dependencies**: Use workspace dependencies for consistency
- **External libraries**: Pin to specific versions for stability
- **Dev dependencies**: Allow patch updates for tools
- **Neural models**: Version control in separate repository

### Security Considerations

- **Dependency auditing**: Use `cargo audit` in CI
- **License compliance**: Check with `cargo license`
- **Supply chain security**: Pin exact versions for production
- **Plugin validation**: Verify signatures before loading

## Performance Optimizations

### Compilation Flags

```toml
[profile.release]
lto = true              # Link-time optimization
codegen-units = 1       # Single codegen unit
panic = "abort"         # Smaller binary size
strip = true           # Remove debug symbols
opt-level = 3          # Maximum optimization
```

### Feature Flags

- **simd**: Enable SIMD optimizations
- **parallel**: Enable parallel processing
- **compression**: Enable document compression
- **encryption**: Enable document encryption
- **python**: Build Python bindings
- **wasm**: Build WASM bindings

## Testing Strategy

### Unit Tests
- Each crate has comprehensive unit tests
- Trait implementations fully tested
- Mock objects for external dependencies

### Integration Tests
- Full pipeline testing
- Plugin system testing
- Multi-format document testing

### Performance Tests
- Benchmark extraction speeds
- Memory usage profiling
- Neural network performance

### Security Tests
- Input validation testing
- Plugin security testing
- Malicious document handling

## Documentation

### API Documentation
- Generate with `cargo doc --workspace`
- Host on docs.rs for public access
- Include examples in documentation

### User Guides
- CLI usage examples
- Python API examples
- Plugin development guide
- Configuration reference

## Deployment

### Release Process
1. Update version numbers
2. Run full test suite
3. Build all targets
4. Create release artifacts
5. Publish to registries

### Distribution
- **Rust crates**: crates.io
- **Python wheels**: PyPI
- **WASM packages**: npm
- **Binary releases**: GitHub releases

This workspace structure provides a solid foundation for NeuralDocFlow Phase 1, enabling modular development, clear separation of concerns, and efficient build processes while maintaining the pure Rust architecture specified in iteration5.