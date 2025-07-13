# NeuralDocFlow Rust Build & Deployment Guide

## Project Structure

```
neuraldocflow/
├── Cargo.toml                 # Workspace configuration
├── Cargo.lock
├── rust-toolchain.toml        # Rust version pinning
├── .cargo/
│   └── config.toml           # Build configuration
├── crates/
│   ├── neuraldocflow/        # Main library crate
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs
│   │       └── ...
│   ├── neuraldocflow-pdf/    # PDF parser crate
│   │   ├── Cargo.toml
│   │   └── src/
│   ├── neuraldocflow-neural/ # Neural processing crate
│   │   ├── Cargo.toml
│   │   └── src/
│   ├── neuraldocflow-swarm/  # Swarm coordination crate
│   │   ├── Cargo.toml
│   │   └── src/
│   └── neuraldocflow-cli/    # CLI application
│       ├── Cargo.toml
│       └── src/
├── benches/                   # Performance benchmarks
├── examples/                  # Usage examples
├── models/                    # Pre-trained neural models
├── tests/                     # Integration tests
└── docs/                      # Documentation

```

## Workspace Configuration

### Root `Cargo.toml`

```toml
[workspace]
members = [
    "crates/neuraldocflow",
    "crates/neuraldocflow-pdf",
    "crates/neuraldocflow-neural", 
    "crates/neuraldocflow-swarm",
    "crates/neuraldocflow-cli",
]
resolver = "2"

[workspace.package]
version = "0.1.0"
edition = "2021"
rust-version = "1.75"
authors = ["NeuralDocFlow Team"]
license = "MIT OR Apache-2.0"
repository = "https://github.com/neuraldocflow/neuraldocflow"

[workspace.dependencies]
# Core dependencies
tokio = { version = "1.38", features = ["full"] }
rayon = "1.10"
serde = { version = "1.0", features = ["derive"] }
thiserror = "1.0"
anyhow = "1.0"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter", "json"] }

# RUV-FANN integration
ruv-fann = { version = "0.1", features = ["simd", "async", "onnx"] }

# PDF processing
memmap2 = "0.9"
nom = "7.1"

# Neural processing
ndarray = { version = "0.15", features = ["rayon", "serde"] }
tokenizers = { version = "0.19", default-features = false, features = ["onig"] }

# Serialization
serde_json = "1.0"
bincode = "1.3"
arrow2 = { version = "0.18", features = ["io_parquet", "io_parquet_compression"] }

# Performance
crossbeam = "0.8"
dashmap = "6.0"
moka = { version = "0.12", features = ["future"] }

# Testing
criterion = { version = "0.5", features = ["html_reports"] }
proptest = "1.4"

[profile.release]
lto = "fat"
codegen-units = 1
opt-level = 3
strip = true
panic = "abort"

[profile.release-with-debug]
inherits = "release"
strip = false
debug = true

[profile.bench]
inherits = "release"
lto = false
```

### Rust Toolchain (`rust-toolchain.toml`)

```toml
[toolchain]
channel = "stable"
components = ["rustfmt", "clippy"]
targets = ["x86_64-unknown-linux-gnu", "x86_64-pc-windows-msvc", "x86_64-apple-darwin", "aarch64-apple-darwin"]
```

### Build Configuration (`.cargo/config.toml`)

```toml
[build]
# Enable parallel compilation
jobs = 8
# Use mold linker on Linux for faster builds
[target.x86_64-unknown-linux-gnu]
linker = "clang"
rustflags = ["-C", "link-arg=-fuse-ld=mold", "-C", "target-cpu=native"]

[target.x86_64-pc-windows-msvc]
rustflags = ["-C", "target-cpu=native"]

[target.x86_64-apple-darwin]
rustflags = ["-C", "target-cpu=native"]

[target.aarch64-apple-darwin]
rustflags = ["-C", "target-cpu=native"]

# Faster builds for development
[profile.dev]
opt-level = 0
debug = true
split-debuginfo = "unpacked"
incremental = true

[profile.dev.package."*"]
opt-level = 3
```

## Individual Crate Configurations

### PDF Parser Crate (`crates/neuraldocflow-pdf/Cargo.toml`)

```toml
[package]
name = "neuraldocflow-pdf"
version.workspace = true
edition.workspace = true

[dependencies]
memmap2.workspace = true
nom.workspace = true
rayon.workspace = true
crossbeam.workspace = true
thiserror.workspace = true
tracing.workspace = true

[target.'cfg(target_arch = "x86_64")'.dependencies]
# SIMD acceleration for x86_64
std_detect = "0.1"

[features]
default = ["simd"]
simd = []
```

### Neural Processing Crate (`crates/neuraldocflow-neural/Cargo.toml`)

```toml
[package]
name = "neuraldocflow-neural"
version.workspace = true
edition.workspace = true

[dependencies]
ruv-fann.workspace = true
ndarray.workspace = true
tokenizers.workspace = true
dashmap.workspace = true
moka.workspace = true
tokio.workspace = true
anyhow.workspace = true
tracing.workspace = true

[dev-dependencies]
criterion.workspace = true

[[bench]]
name = "neural_bench"
harness = false
```

## Build Scripts

### Development Build Script (`scripts/build-dev.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Building NeuralDocFlow in development mode..."

# Check dependencies
command -v cargo >/dev/null 2>&1 || { echo "❌ Cargo not installed"; exit 1; }
command -v rustc >/dev/null 2>&1 || { echo "❌ Rust not installed"; exit 1; }

# Install additional tools
echo "📦 Installing development tools..."
cargo install cargo-watch cargo-nextest cargo-llvm-cov

# Format code
echo "🎨 Formatting code..."
cargo fmt --all

# Run clippy
echo "🔍 Running clippy..."
cargo clippy --all-targets --all-features -- -D warnings

# Build all crates
echo "🔨 Building all crates..."
cargo build --workspace

# Run tests with coverage
echo "🧪 Running tests with coverage..."
cargo llvm-cov --workspace --lcov --output-path lcov.info

echo "✅ Development build complete!"
```

### Release Build Script (`scripts/build-release.sh`)

```bash
#!/usr/bin/env bash
set -euo pipefail

VERSION=${1:-$(git describe --tags --always)}
TARGETS=("x86_64-unknown-linux-gnu" "x86_64-pc-windows-msvc" "x86_64-apple-darwin" "aarch64-apple-darwin")

echo "🚀 Building NeuralDocFlow v${VERSION} for release..."

# Clean previous builds
cargo clean

# Build for each target
for target in "${TARGETS[@]}"; do
    echo "🎯 Building for ${target}..."
    
    if cargo build --release --target "${target}"; then
        # Create release directory
        mkdir -p "releases/${VERSION}/${target}"
        
        # Copy binaries
        if [[ "$target" == *"windows"* ]]; then
            cp "target/${target}/release/neuraldocflow.exe" "releases/${VERSION}/${target}/"
        else
            cp "target/${target}/release/neuraldocflow" "releases/${VERSION}/${target}/"
        fi
        
        # Copy models and configs
        cp -r models "releases/${VERSION}/${target}/"
        cp README.md LICENSE "releases/${VERSION}/${target}/"
        
        # Create archive
        cd "releases/${VERSION}"
        if [[ "$target" == *"windows"* ]]; then
            zip -r "neuraldocflow-${VERSION}-${target}.zip" "${target}"
        else
            tar czf "neuraldocflow-${VERSION}-${target}.tar.gz" "${target}"
        fi
        cd ../..
        
        echo "✅ Built ${target} successfully"
    else
        echo "❌ Failed to build ${target}"
    fi
done

echo "📦 Release artifacts created in releases/${VERSION}/"
```

## Docker Deployment

### Multi-stage Dockerfile

```dockerfile
# Build stage
FROM rust:1.75 as builder

# Install dependencies
RUN apt-get update && apt-get install -y \
    clang \
    mold \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /usr/src/neuraldocflow

# Copy workspace files
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates/ ./crates/

# Build dependencies first (for caching)
RUN cargo build --release --workspace --target-dir /tmp/target && \
    rm -rf /tmp/target/release/deps/neuraldocflow*

# Build the application
COPY . .
RUN cargo build --release --bin neuraldocflow

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 neuraldocflow

# Copy binary from builder
COPY --from=builder /usr/src/neuraldocflow/target/release/neuraldocflow /usr/local/bin/neuraldocflow

# Copy models
COPY --from=builder /usr/src/neuraldocflow/models /opt/neuraldocflow/models

# Set ownership
RUN chown -R neuraldocflow:neuraldocflow /opt/neuraldocflow

# Switch to non-root user
USER neuraldocflow

# Set working directory
WORKDIR /opt/neuraldocflow

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD neuraldocflow health || exit 1

# Default command
CMD ["neuraldocflow", "serve", "--host", "0.0.0.0", "--port", "8080"]
```

### Docker Compose for Development

```yaml
version: '3.8'

services:
  neuraldocflow:
    build:
      context: .
      dockerfile: Dockerfile
      target: builder
    volumes:
      - ./models:/opt/neuraldocflow/models
      - ./data:/opt/neuraldocflow/data
      - cargo-cache:/usr/local/cargo/registry
      - target-cache:/usr/src/neuraldocflow/target
    environment:
      - RUST_LOG=neuraldocflow=debug,ruv_fann=info
      - NEURALDOCFLOW_WORKERS=8
      - NEURALDOCFLOW_NEURAL_DEVICE=cpu
    ports:
      - "8080:8080"
      - "9090:9090"
    command: cargo watch -x 'run -- serve'

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9091:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'

  grafana:
    image: grafana/grafana:latest
    volumes:
      - ./monitoring/grafana:/etc/grafana/provisioning
      - grafana-data:/var/lib/grafana
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
      - GF_USERS_ALLOW_SIGN_UP=false
    ports:
      - "3000:3000"
    depends_on:
      - prometheus

volumes:
  cargo-cache:
  target-cache:
  prometheus-data:
  grafana-data:
```

## Kubernetes Deployment

### Deployment Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neuraldocflow
  labels:
    app: neuraldocflow
spec:
  replicas: 3
  selector:
    matchLabels:
      app: neuraldocflow
  template:
    metadata:
      labels:
        app: neuraldocflow
    spec:
      containers:
      - name: neuraldocflow
        image: neuraldocflow/neuraldocflow:latest
        ports:
        - containerPort: 8080
          name: http
        - containerPort: 9090
          name: metrics
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        env:
        - name: NEURALDOCFLOW_WORKERS
          value: "16"
        - name: NEURALDOCFLOW_NEURAL_DEVICE
          value: "cuda"
        - name: RUST_LOG
          value: "neuraldocflow=info"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: models
          mountPath: /opt/neuraldocflow/models
          readOnly: true
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: neuraldocflow-models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: neuraldocflow
spec:
  selector:
    app: neuraldocflow
  ports:
  - name: http
    port: 80
    targetPort: 8080
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: neuraldocflow-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: neuraldocflow
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: neuraldocflow_processing_queue_depth
      target:
        type: AverageValue
        averageValue: "10"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
      - type: Pods
        value: 4
        periodSeconds: 60
```

## Performance Optimization

### Compile-time Optimizations

```toml
# Cargo.toml profile for maximum performance
[profile.release-lto]
inherits = "release"
lto = "fat"
codegen-units = 1
opt-level = 3
target-cpu = "native"
panic = "abort"

# Enable specific CPU features
[target.'cfg(target_arch = "x86_64")'.dependencies]
core_arch = { version = "0.1", features = ["avx2", "avx512f"] }
```

### Runtime Configuration

```rust
// src/config.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Number of worker threads (0 = auto-detect)
    pub worker_threads: usize,
    
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    
    /// Neural processing device
    pub neural_device: NeuralDevice,
    
    /// Memory pool size in MB
    pub memory_pool_size: usize,
    
    /// Batch size for neural processing
    pub neural_batch_size: usize,
    
    /// Enable memory mapping for large files
    pub enable_mmap: bool,
    
    /// Swarm configuration
    pub swarm: SwarmConfig,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            worker_threads: 0, // Auto-detect
            enable_simd: true,
            neural_device: NeuralDevice::Auto,
            memory_pool_size: 1024, // 1GB
            neural_batch_size: 32,
            enable_mmap: true,
            swarm: SwarmConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NeuralDevice {
    Auto,
    Cpu,
    Cuda { device_id: usize },
    Rocm { device_id: usize },
}
```

## Monitoring and Observability

### Prometheus Metrics

```rust
// src/metrics.rs
use prometheus::{Counter, Histogram, IntGauge, Registry};

pub struct Metrics {
    pub documents_processed: Counter,
    pub processing_duration: Histogram,
    pub active_agents: IntGauge,
    pub memory_usage: IntGauge,
    pub neural_inference_duration: Histogram,
}

impl Metrics {
    pub fn new(registry: &Registry) -> Result<Self, prometheus::Error> {
        let metrics = Self {
            documents_processed: Counter::new("neuraldocflow_documents_processed_total", "Total documents processed")?,
            processing_duration: Histogram::new("neuraldocflow_processing_duration_seconds", "Document processing duration")?,
            active_agents: IntGauge::new("neuraldocflow_active_agents", "Number of active swarm agents")?,
            memory_usage: IntGauge::new("neuraldocflow_memory_usage_bytes", "Memory usage in bytes")?,
            neural_inference_duration: Histogram::new("neuraldocflow_neural_inference_duration_seconds", "Neural inference duration")?,
        };
        
        registry.register(Box::new(metrics.documents_processed.clone()))?;
        registry.register(Box::new(metrics.processing_duration.clone()))?;
        registry.register(Box::new(metrics.active_agents.clone()))?;
        registry.register(Box::new(metrics.memory_usage.clone()))?;
        registry.register(Box::new(metrics.neural_inference_duration.clone()))?;
        
        Ok(metrics)
    }
}
```

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
name: CI/CD

on:
  push:
    branches: [ main, develop ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]

env:
  CARGO_TERM_COLOR: always
  RUST_BACKTRACE: 1

jobs:
  test:
    name: Test
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        rust: [stable, beta]
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Rust
      uses: actions-rust-lang/setup-rust-toolchain@v1
      with:
        toolchain: ${{ matrix.rust }}
        components: rustfmt, clippy
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/registry
          ~/.cargo/git
          target
        key: ${{ runner.os }}-cargo-${{ hashFiles('**/Cargo.lock') }}
    
    - name: Run tests
      run: cargo nextest run --all-features
    
    - name: Run clippy
      run: cargo clippy --all-targets --all-features -- -D warnings
    
    - name: Check formatting
      run: cargo fmt --all -- --check

  benchmark:
    name: Benchmark
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Rust
      uses: actions-rust-lang/setup-rust-toolchain@v1
      with:
        toolchain: stable
    
    - name: Run benchmarks
      run: cargo bench --all-features
    
    - name: Upload benchmark results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'cargo'
        output-file-path: target/criterion/output.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true

  release:
    name: Release
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/')
    needs: [test]
    steps:
    - uses: actions/checkout@v4
    
    - name: Build release artifacts
      run: ./scripts/build-release.sh ${{ github.ref_name }}
    
    - name: Create GitHub Release
      uses: softprops/action-gh-release@v1
      with:
        files: releases/${{ github.ref_name }}/*
        generate_release_notes: true
    
    - name: Publish to crates.io
      run: |
        cargo publish -p neuraldocflow-pdf
        cargo publish -p neuraldocflow-neural
        cargo publish -p neuraldocflow-swarm
        cargo publish -p neuraldocflow
      env:
        CARGO_REGISTRY_TOKEN: ${{ secrets.CRATES_IO_TOKEN }}
```

This comprehensive build and deployment guide provides everything needed to build, test, and deploy the pure Rust NeuralDocFlow system in production environments.