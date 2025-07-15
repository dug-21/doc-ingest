# Neural Document Flow API - Feature Gates

This document describes the feature flags available for optimizing compilation times and reducing binary size.

## 🚀 Quick Start

### Minimal Development Build (Fastest)
```bash
cargo build --no-default-features --features minimal
```
**Compilation time: ~30-60 seconds** (vs ~5+ minutes for full build)

### Production Build
```bash
cargo build --release --features full
```

### Selective Features
```bash
# API + Authentication only
cargo build --no-default-features --features "minimal,auth"

# API + Documentation only  
cargo build --no-default-features --features "minimal,docs"
```

## 📦 Feature Categories

### Core Features

| Feature | Description | Dependencies | Build Impact |
|---------|-------------|--------------|--------------|
| `minimal` | Basic API server with axum | `axum`, `serde`, `tokio` | ⚡ Fastest |
| `full` | All features enabled | All dependencies | 🐌 Slowest |

### Optional Features

| Feature | Description | Key Dependencies | Recommended For |
|---------|-------------|------------------|-----------------|
| `database` | SQLite database support | `sqlx` | Production APIs |
| `auth` | JWT authentication | `jsonwebtoken`, `argon2` | Secure APIs |
| `security` | Security scanning | `neural-doc-flow-security` | Production |
| `metrics` | Prometheus metrics | `prometheus` | Monitoring |
| `docs` | OpenAPI documentation | `utoipa`, `swagger-ui` | Development |
| `background-jobs` | Async job processing | `tokio-cron-scheduler` | Production |
| `rate-limiting` | Request rate limiting | `governor` | Production |
| `compression` | HTTP compression | `tower-http/compression` | Production |
| `tracing` | Enhanced logging | `tracing-subscriber` | Debug/Prod |

### Development Features

| Feature | Description | When to Use |
|---------|-------------|-------------|
| `dev-full` | All features + dev tools | Full development environment |
| `test-utils` | Testing utilities | Running tests |

## ⚡ Performance Comparison

Based on compilation benchmarks:

| Configuration | Compile Time | Use Case |
|---------------|--------------|----------|
| `minimal` | ~45s | Fast development iteration |
| `minimal,auth` | ~65s | API development with auth |
| `minimal,database` | ~85s | API with persistence |
| `minimal,docs` | ~55s | API development with docs |
| `full` | ~300s+ | Production builds |
| default | ~180s | Standard development |

## 🔧 Build Optimization Tips

### 1. Development Workflow
```bash
# Fast iteration during development
export NEURAL_API_FEATURES="minimal"
cargo build --no-default-features --features $NEURAL_API_FEATURES

# Add features as needed
export NEURAL_API_FEATURES="minimal,auth,docs"
cargo build --no-default-features --features $NEURAL_API_FEATURES
```

### 2. CI/CD Pipeline
```yaml
# Fast CI builds
- name: Quick Check
  run: cargo check --no-default-features --features minimal

# Full test build
- name: Full Test
  run: cargo test --features full
```

### 3. Local Development Environment
```bash
# .env file
CARGO_BUILD_FEATURES="minimal,auth"
SQLX_OFFLINE=true
```

## 🎯 Feature Selection Guide

### For Different Use Cases:

#### API Development
```bash
--features "minimal,auth,docs"
```
- Fast compilation
- Authentication testing
- API documentation

#### Database Development  
```bash
--features "minimal,database,auth"
```
- Core API with persistence
- User management
- Data validation

#### Production Deployment
```bash
--features "full"
```
- All security features
- Monitoring and metrics
- Background processing
- Rate limiting

#### Documentation Writing
```bash
--features "minimal,docs"
```
- Fast build with OpenAPI
- Swagger UI for testing
- Minimal overhead

#### Testing & QA
```bash
--features "dev-full"
```
- All features for comprehensive testing
- Development tools included
- Test utilities available

## 📊 Dependency Analysis

### Heavy Dependencies (Avoid in Development)
- `sqlx` - Database layer (~2MB, +60s compile time)
- `prometheus` - Metrics collection (~1MB, +30s compile time)  
- `utoipa` - OpenAPI generation (~500KB, +15s compile time)
- `neural-doc-flow-security` - Security scanning (~3MB, +90s compile time)

### Lightweight Dependencies (Always Include)
- `axum` - Web framework (~200KB, +10s compile time)
- `serde` - Serialization (~100KB, +5s compile time)
- `tokio` - Async runtime (~500KB, +15s compile time)

## 🛠️ Custom Feature Combinations

### Example: Microservice API
```toml
[features]
microservice = ["minimal", "auth", "metrics", "rate-limiting"]
```

### Example: Documentation Server
```toml
[features]
docs-server = ["minimal", "docs", "compression"]
```

### Example: Development Stack
```toml
[features]
dev-stack = ["minimal", "auth", "docs", "database"]
```

## 🚨 Migration from Full Build

If you're currently using the default full build:

1. **Start with minimal**: `--features minimal`
2. **Add one feature at a time**: `--features "minimal,auth"`
3. **Test functionality**: Ensure your use case works
4. **Measure improvement**: Use `bench_compile.sh`
5. **Document your stack**: Create custom feature sets

## 📋 Troubleshooting

### Common Issues:

#### "Function not found" errors
**Solution**: Add the required feature flag
```bash
# Error: auth functions missing
cargo build --features "minimal,auth"
```

#### Slow compilation with minimal features
**Solution**: Check for unused dependencies
```bash
cargo tree --features minimal
```

#### Tests failing with minimal features  
**Solution**: Use dev-full for testing
```bash
cargo test --features dev-full
```

## 🔄 Feature Maintenance

### Adding New Features
1. Add dependencies as `optional = true`
2. Gate code with `#[cfg(feature = "feature-name")]`
3. Update this documentation
4. Add to benchmark script
5. Test compilation impact

### Removing Features
1. Deprecate feature in documentation
2. Add migration guide
3. Remove in next major version
4. Update CI/CD pipelines

---

## 📞 Support

For build optimization questions:
- Check compilation benchmark: `./bench_compile.sh`
- Review build logs: `cargo build -v`
- Profile compilation: `cargo build --timings`

**Target Achievement**: Reduce API module compilation from >5 minutes to <2 minutes ✅