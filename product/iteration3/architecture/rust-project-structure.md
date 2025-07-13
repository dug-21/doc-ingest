# Rust Project Structure for NeuralDocFlow

## 📁 Workspace Organization

```
neuraldocflow/
├── Cargo.toml                                 # Workspace root configuration
├── README.md                                  # Project overview and quick start
├── LICENSE                                    # MIT or Apache 2.0 license
├── CONTRIBUTING.md                            # Contribution guidelines
├── CHANGELOG.md                               # Version history
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                            # Continuous integration
│   │   ├── release.yml                       # Release automation
│   │   ├── security.yml                      # Security scans
│   │   └── benchmarks.yml                    # Performance benchmarks
│   ├── ISSUE_TEMPLATE/                       # Issue templates
│   └── PULL_REQUEST_TEMPLATE.md              # PR template
├── docs/                                     # Documentation
│   ├── api/                                  # API documentation
│   ├── guides/                               # User guides
│   ├── architecture/                         # Architecture docs
│   └── examples/                             # Usage examples
├── configs/                                  # Sample configurations
│   ├── domains/                              # Domain-specific configs
│   │   ├── financial.yaml                   # Financial document config
│   │   ├── legal.yaml                       # Legal document config
│   │   ├── medical.yaml                     # Medical record config
│   │   └── academic.yaml                    # Research paper config
│   ├── models/                               # Model configurations
│   └── deployment/                           # Deployment configs
├── test_data/                                # Test documents and fixtures
│   ├── pdf_samples/                          # Sample PDF files
│   ├── ground_truth/                         # Validation data
│   └── benchmarks/                           # Performance test files
├── models/                                   # Pre-trained models
│   ├── onnx/                                 # ONNX transformer models
│   ├── fann/                                 # RUV-FANN networks
│   └── custom/                               # Custom trained models
├── scripts/                                  # Build and utility scripts
│   ├── setup.sh                             # Environment setup
│   ├── download_models.sh                   # Model download script
│   └── benchmark.sh                         # Performance benchmarking
├── examples/                                 # Usage examples
│   ├── simple_extraction.rs                 # Basic usage
│   ├── custom_config.rs                     # Custom configuration
│   ├── swarm_processing.rs                  # Swarm coordination
│   └── plugin_development.rs                # Plugin creation
├── benchmarks/                               # Performance benchmarks
│   ├── Cargo.toml                           # Benchmark dependencies
│   └── src/
│       ├── lib.rs                           # Benchmark utilities
│       ├── processing.rs                    # Document processing benchmarks
│       ├── neural.rs                        # Neural inference benchmarks
│       └── memory.rs                        # Memory usage benchmarks
├── tools/                                    # Development tools
│   ├── config_generator/                    # Configuration generator
│   ├── model_optimizer/                     # Model optimization tools
│   └── performance_analyzer/                # Performance analysis
├── plugins/                                  # Plugin examples
│   ├── document_handlers/                   # Document type handlers
│   ├── output_formatters/                   # Output format plugins
│   └── neural_models/                       # Custom neural model plugins
│
├── neuraldocflow-common/                     # Shared utilities and types
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                           # Public API
│       ├── error.rs                         # Error types and handling
│       ├── types.rs                         # Common data structures
│       ├── traits.rs                        # Core trait definitions
│       ├── config.rs                        # Configuration types
│       ├── metrics.rs                       # Performance metrics
│       └── utils.rs                         # Utility functions
│
├── neuraldocflow-memory/                     # Memory management and SIMD
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                           # Public API
│       ├── allocator.rs                     # Custom memory allocators
│       ├── pool.rs                          # Memory pool management
│       ├── simd/                            # SIMD acceleration modules
│       │   ├── mod.rs                       # SIMD module exports
│       │   ├── x86.rs                       # x86/x64 SIMD operations
│       │   ├── arm.rs                       # ARM NEON operations
│       │   └── generic.rs                   # Fallback implementations
│       ├── mmap.rs                          # Memory-mapped file handling
│       └── shared.rs                        # Shared memory regions
│
├── neuraldocflow-pdf/                        # PDF parsing and text extraction
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                           # Public API
│       ├── parser.rs                        # Core PDF parser
│       ├── extractor.rs                     # Text extraction engine
│       ├── layout.rs                        # Layout analysis
│       ├── tables.rs                        # Table detection and extraction
│       ├── images.rs                        # Image extraction
│       ├── metadata.rs                      # Metadata extraction
│       └── security.rs                      # Security and validation
│
├── neuraldocflow-config/                     # Configuration system
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                           # Public API
│       ├── reader.rs                        # YAML configuration reader
│       ├── validator.rs                     # Configuration validation
│       ├── generator.rs                     # Automatic config generation
│       ├── merger.rs                        # Configuration merging
│       ├── schema.rs                        # Configuration schema
│       └── examples.rs                      # Example configurations
│
├── neuraldocflow-neural/                     # Neural processing engine
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                           # Public API
│       ├── onnx/                            # ONNX Runtime integration
│       │   ├── mod.rs                       # ONNX module exports
│       │   ├── session.rs                   # Session management
│       │   ├── loader.rs                    # Model loading
│       │   ├── inference.rs                 # Inference engine
│       │   └── optimization.rs              # Model optimization
│       ├── fann/                            # RUV-FANN integration
│       │   ├── mod.rs                       # FANN module exports
│       │   ├── network.rs                   # Network management
│       │   ├── training.rs                  # Training system
│       │   ├── inference.rs                 # FANN inference
│       │   └── custom.rs                    # Custom activations
│       ├── hybrid.rs                        # Hybrid ONNX+FANN engine
│       ├── cache.rs                         # Model and result caching
│       ├── batch.rs                         # Batch processing
│       └── monitoring.rs                    # Performance monitoring
│
├── neuraldocflow-analyzer/                   # Document analysis
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                           # Public API
│       ├── structure.rs                     # Document structure analysis
│       ├── classifier.rs                    # Document classification
│       ├── entities.rs                      # Entity extraction
│       ├── relationships.rs                 # Relationship building
│       ├── patterns.rs                      # Pattern detection
│       └── confidence.rs                    # Confidence scoring
│
├── neuraldocflow-swarm/                      # Swarm coordination
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                           # Public API
│       ├── coordinator.rs                   # Swarm coordinator
│       ├── agent.rs                         # Agent implementation
│       ├── manager.rs                       # Agent manager
│       ├── distributor.rs                   # Task distribution
│       ├── balancer.rs                      # Load balancing
│       ├── monitor.rs                       # Health monitoring
│       └── scaling.rs                       # Auto-scaling logic
│
├── neuraldocflow-mcp/                        # MCP server implementation
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                           # Public API
│       ├── server.rs                        # MCP server
│       ├── tools.rs                         # Tool implementations
│       ├── resources.rs                     # Resource management
│       ├── client.rs                        # Claude Flow client
│       └── protocol.rs                      # MCP protocol handling
│
├── neuraldocflow-plugins/                    # Plugin system
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                           # Public API
│       ├── manager.rs                       # Plugin manager
│       ├── loader.rs                        # Dynamic loading
│       ├── registry.rs                      # Plugin registry
│       ├── security.rs                      # Security validation
│       ├── handlers/                        # Built-in handlers
│       │   ├── mod.rs                       # Handler exports
│       │   ├── pdf.rs                       # PDF handler
│       │   ├── docx.rs                      # DOCX handler
│       │   └── txt.rs                       # Text handler
│       └── examples/                        # Plugin examples
│           ├── custom_handler.rs            # Custom document handler
│           └── custom_formatter.rs          # Custom output formatter
│
├── neuraldocflow-api/                        # API interfaces
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                           # Public API
│       ├── rest/                            # REST API
│       │   ├── mod.rs                       # REST module exports
│       │   ├── server.rs                    # Axum server
│       │   ├── handlers.rs                  # Request handlers
│       │   ├── middleware.rs                # Middleware
│       │   └── auth.rs                      # Authentication
│       ├── python/                          # Python bindings
│       │   ├── mod.rs                       # Python module exports
│       │   ├── bindings.rs                  # PyO3 bindings
│       │   └── types.rs                     # Python type conversions
│       ├── wasm/                            # WebAssembly interface
│       │   ├── mod.rs                       # WASM module exports
│       │   ├── bindings.rs                  # WASM bindings
│       │   └── utils.rs                     # WASM utilities
│       └── cli/                             # Command line interface
│           ├── mod.rs                       # CLI module exports
│           ├── commands.rs                  # CLI commands
│           ├── args.rs                      # Argument parsing
│           └── output.rs                    # Output formatting
│
├── neuraldocflow-pipeline/                   # Processing pipeline
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                           # Public API
│       ├── builder.rs                       # Pipeline builder
│       ├── executor.rs                      # Pipeline executor
│       ├── stage.rs                         # Pipeline stages
│       ├── optimization.rs                  # Pipeline optimization
│       └── monitoring.rs                    # Pipeline monitoring
│
├── neuraldocflow-discovery/                  # Model discovery
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                           # Public API
│       ├── service.rs                       # Discovery service
│       ├── evaluator.rs                     # Model evaluation
│       ├── ranking.rs                       # Model ranking
│       ├── cache.rs                         # Discovery cache
│       └── providers/                       # Discovery providers
│           ├── mod.rs                       # Provider exports
│           ├── huggingface.rs               # Hugging Face integration
│           ├── local.rs                     # Local model discovery
│           └── remote.rs                    # Remote model sources
│
├── neuraldocflow-validation/                 # Result validation
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                           # Public API
│       ├── validator.rs                     # Result validator
│       ├── rules.rs                         # Validation rules
│       ├── corrector.rs                     # Automatic correction
│       └── reporter.rs                      # Validation reporting
│
├── neuraldocflow-monitoring/                 # Monitoring and observability
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                           # Public API
│       ├── metrics.rs                       # Metrics collection
│       ├── tracing.rs                       # Distributed tracing
│       ├── alerts.rs                        # Alerting system
│       ├── dashboard.rs                     # Monitoring dashboard
│       └── exporters/                       # Metric exporters
│           ├── mod.rs                       # Exporter exports
│           ├── prometheus.rs                # Prometheus exporter
│           └── jaeger.rs                    # Jaeger tracing
│
├── neuraldocflow-security/                   # Security and sandboxing
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                           # Public API
│       ├── sandbox.rs                       # Sandboxing system
│       ├── validation.rs                    # Input validation
│       ├── encryption.rs                    # Data encryption
│       └── audit.rs                         # Security auditing
│
└── neuraldocflow/                           # Main application crate
    ├── Cargo.toml
    └── src/
        ├── main.rs                          # Application entry point
        ├── lib.rs                           # Library exports
        ├── processor.rs                     # Main document processor
        ├── builder.rs                       # Builder pattern
        ├── config.rs                        # Configuration management
        └── service.rs                       # Service orchestration
```

## 📋 Workspace Cargo.toml

```toml
[workspace]
resolver = "2"
members = [
    "neuraldocflow-common",
    "neuraldocflow-memory",
    "neuraldocflow-pdf",
    "neuraldocflow-config",
    "neuraldocflow-neural",
    "neuraldocflow-analyzer",
    "neuraldocflow-swarm",
    "neuraldocflow-mcp",
    "neuraldocflow-plugins",
    "neuraldocflow-api",
    "neuraldocflow-pipeline",
    "neuraldocflow-discovery",
    "neuraldocflow-validation",
    "neuraldocflow-monitoring",
    "neuraldocflow-security",
    "neuraldocflow",
    "benchmarks",
    "tools/config_generator",
    "tools/model_optimizer",
    "tools/performance_analyzer",
    "plugins/document_handlers",
    "plugins/output_formatters",
    "plugins/neural_models"
]

[workspace.package]
version = "0.1.0"
edition = "2021"
authors = ["NeuralDocFlow Team <team@neuraldocflow.ai>"]
license = "MIT OR Apache-2.0"
repository = "https://github.com/neuraldocflow/neuraldocflow"
homepage = "https://neuraldocflow.ai"
documentation = "https://docs.neuraldocflow.ai"
keywords = ["pdf", "document", "neural", "swarm", "extraction"]
categories = ["science", "text-processing", "api-bindings"]
readme = "README.md"
description = "High-performance neural document processing with swarm intelligence"

[workspace.dependencies]
# Core dependencies
tokio = { version = "1.35", features = ["full"] }
serde = { version = "1.0", features = ["derive", "rc"] }
serde_json = "1.0"
serde_yaml = "0.9"
anyhow = "1.0"
thiserror = "1.0"
uuid = { version = "1.6", features = ["v4", "serde"] }

# Async and concurrency
futures = "0.3"
async-stream = "0.3"
tokio-stream = "0.1"
rayon = "1.8"
crossbeam = "0.8"
parking_lot = "0.12"
dashmap = "5.5"

# Networking and protocols
axum = { version = "0.7", features = ["multipart", "ws"] }
tower = "0.4"
tower-http = { version = "0.5", features = ["cors", "trace"] }
hyper = "1.0"
reqwest = { version = "0.11", features = ["json", "stream"] }

# Serialization and data formats
bincode = "1.3"
rmp-serde = "1.1"
arrow = "50.0"
parquet = "50.0"

# Neural networks and ML
ort = { version = "1.16", features = ["cuda", "tensorrt"] }
ruv-fann = "0.3"
ndarray = "0.15"
tokenizers = "0.15"

# PDF processing
lopdf = "0.32"
pdf-extract = "0.7"

# Memory management and SIMD
memmap2 = "0.9"
wide = "0.7"
bytemuck = "1.14"

# Cryptography and security
ring = "0.17"
aes-gcm = "0.10"

# Monitoring and observability
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }
metrics = "0.22"
prometheus = "0.13"

# Configuration and validation
config = "0.13"
validator = { version = "0.16", features = ["derive"] }
jsonschema = "0.17"

# CLI and user interface
clap = { version = "4.4", features = ["derive", "env"] }
indicatif = "0.17"
console = "0.15"

# Plugin system
libloading = "0.8"
dlopen2 = "0.5"

# Python bindings
pyo3 = { version = "0.20", features = ["extension-module", "abi3-py38"], optional = true }

# WebAssembly
wasm-bindgen = { version = "0.2", optional = true }
js-sys = { version = "0.3", optional = true }
web-sys = { version = "0.3", optional = true }

# Testing and benchmarking
criterion = { version = "0.5", features = ["html_reports"] }
proptest = "1.4"
quickcheck = "1.0"
rstest = "0.18"

# Development tools
cargo-watch = "8.4"
cargo-nextest = "0.9"

[workspace.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]

[profile.release]
lto = "fat"
codegen-units = 1
opt-level = 3
debug = false
panic = "abort"

[profile.release-with-debug]
inherits = "release"
debug = true
panic = "unwind"

[profile.bench]
inherits = "release"
debug = true

[profile.dev]
opt-level = 1
debug = true
```

## 🔧 Core Crate Configurations

### neuraldocflow-common/Cargo.toml

```toml
[package]
name = "neuraldocflow-common"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true
repository.workspace = true
description = "Common types and utilities for NeuralDocFlow"

[dependencies]
serde.workspace = true
serde_json.workspace = true
thiserror.workspace = true
uuid.workspace = true
tokio.workspace = true
chrono = { version = "0.4", features = ["serde"] }

[features]
default = []
python = ["pyo3"]
wasm = ["wasm-bindgen"]

[dependencies.pyo3]
workspace = true
optional = true

[dependencies.wasm-bindgen]
workspace = true
optional = true
```

### neuraldocflow-neural/Cargo.toml

```toml
[package]
name = "neuraldocflow-neural"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true
repository.workspace = true
description = "Neural processing engine with ONNX and RUV-FANN integration"

[dependencies]
neuraldocflow-common = { path = "../neuraldocflow-common" }
neuraldocflow-memory = { path = "../neuraldocflow-memory" }

# Neural networks
ort.workspace = true
ruv-fann.workspace = true
ndarray.workspace = true
tokenizers.workspace = true

# Async and concurrency
tokio.workspace = true
futures.workspace = true
dashmap.workspace = true
parking_lot.workspace = true

# Serialization
serde.workspace = true
serde_json.workspace = true
bincode.workspace = true

# Error handling
anyhow.workspace = true
thiserror.workspace = true

# Monitoring
tracing.workspace = true
metrics.workspace = true

[features]
default = ["onnx", "fann"]
onnx = ["ort"]
fann = ["ruv-fann"]
cuda = ["ort/cuda"]
tensorrt = ["ort/tensorrt"]
```

### neuraldocflow-mcp/Cargo.toml

```toml
[package]
name = "neuraldocflow-mcp"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true
repository.workspace = true
description = "MCP server for Claude Flow integration"

[dependencies]
neuraldocflow-common = { path = "../neuraldocflow-common" }
neuraldocflow-neural = { path = "../neuraldocflow-neural" }
neuraldocflow-swarm = { path = "../neuraldocflow-swarm" }

# MCP and JSON-RPC
jsonrpc-core = "18.0"
jsonrpc-http-server = "18.0"
jsonrpc-ws-server = "18.0"

# HTTP server
axum.workspace = true
tower.workspace = true
tower-http.workspace = true

# Async runtime
tokio.workspace = true
futures.workspace = true

# Serialization
serde.workspace = true
serde_json.workspace = true

# Error handling
anyhow.workspace = true
thiserror.workspace = true

# Tracing
tracing.workspace = true
```

## 🧪 Testing Strategy

### Unit Tests
Each crate includes comprehensive unit tests in `src/` files using `#[cfg(test)]` modules.

### Integration Tests
```rust
// neuraldocflow/tests/integration_test.rs
use neuraldocflow::NeuralDocFlowProcessor;
use std::path::Path;

#[tokio::test]
async fn test_end_to_end_processing() {
    let processor = NeuralDocFlowProcessor::builder()
        .with_config_path("test_configs/financial.yaml")
        .with_neural_models(vec!["layoutlmv3", "finbert"])
        .build()
        .await
        .expect("Failed to build processor");

    let result = processor
        .process_document(Path::new("test_data/sample_10k.pdf"))
        .await
        .expect("Failed to process document");

    assert!(result.confidence_scores.overall_confidence > 0.9);
    assert!(!result.extracted_data.entities.is_empty());
}
```

### Property-Based Tests
```rust
// neuraldocflow-pdf/src/parser.rs
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_pdf_parsing_always_succeeds_on_valid_pdfs(
            pdf_bytes in prop::collection::vec(any::<u8>(), 1000..10000)
        ) {
            let parser = PdfParser::new(Default::default());
            // Property: parsing should either succeed or fail gracefully
            let result = parser.parse_bytes(&pdf_bytes);
            prop_assert!(result.is_ok() || matches!(result, Err(PdfError::InvalidFormat)));
        }
    }
}
```

## 🚀 Build and Development Scripts

### setup.sh
```bash
#!/bin/bash
set -e

echo "Setting up NeuralDocFlow development environment..."

# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# Install required targets
rustup target add wasm32-unknown-unknown
rustup target add x86_64-pc-windows-gnu

# Install development tools
cargo install cargo-watch
cargo install cargo-nextest
cargo install wasm-pack

# Download neural models
./scripts/download_models.sh

# Build all crates
cargo build --workspace

echo "Development environment setup complete!"
```

### download_models.sh
```bash
#!/bin/bash
set -e

MODELS_DIR="models"
mkdir -p "$MODELS_DIR"/{onnx,fann,custom}

echo "Downloading pre-trained models..."

# Download LayoutLMv3
if [ ! -f "$MODELS_DIR/onnx/layoutlmv3.onnx" ]; then
    echo "Downloading LayoutLMv3..."
    wget -O "$MODELS_DIR/onnx/layoutlmv3.onnx" \
        "https://huggingface.co/microsoft/layoutlmv3-base/resolve/main/pytorch_model.onnx"
fi

# Download FinBERT
if [ ! -f "$MODELS_DIR/onnx/finbert.onnx" ]; then
    echo "Downloading FinBERT..."
    wget -O "$MODELS_DIR/onnx/finbert.onnx" \
        "https://huggingface.co/ProsusAI/finbert/resolve/main/pytorch_model.onnx"
fi

echo "Model download complete!"
```

## 📦 Package Features

### Feature Matrix

| Crate | Default Features | Optional Features |
|-------|------------------|-------------------|
| `neuraldocflow-common` | `[]` | `python`, `wasm` |
| `neuraldocflow-neural` | `["onnx", "fann"]` | `cuda`, `tensorrt` |
| `neuraldocflow-api` | `["rest", "cli"]` | `python`, `wasm` |
| `neuraldocflow-mcp` | `["server"]` | `client`, `websocket` |
| `neuraldocflow` | `["full"]` | `minimal`, `python`, `wasm` |

### Conditional Compilation

```rust
// Platform-specific optimizations
#[cfg(target_arch = "x86_64")]
mod simd_x86;

#[cfg(target_arch = "aarch64")]
mod simd_arm;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
mod simd_generic;

// Feature-specific modules
#[cfg(feature = "cuda")]
mod gpu_acceleration;

#[cfg(feature = "python")]
mod python_bindings;

#[cfg(feature = "wasm")]
mod wasm_bindings;
```

This project structure provides:

1. **Clear Separation**: Each module has a distinct responsibility
2. **Shared Dependencies**: Common dependencies managed at workspace level
3. **Feature Flags**: Conditional compilation for different use cases
4. **Plugin Architecture**: Extensible plugin system
5. **Multi-Platform**: Support for different targets and architectures
6. **Development Tools**: Comprehensive tooling for development and testing
7. **Documentation**: Structured documentation and examples
8. **Security**: Sandboxing and validation systems

The structure supports the full implementation roadmap while maintaining modularity and extensibility.