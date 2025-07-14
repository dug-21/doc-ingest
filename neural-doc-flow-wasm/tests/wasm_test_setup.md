# WASM Test Environment Setup Guide

## Installation Requirements

### 1. Install wasm-pack (if not already installed)
```bash
# Install wasm-pack via installer script
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Or install via cargo
cargo install wasm-pack

# Verify installation
wasm-pack --version
```

### 2. Test Dependencies
The following dependency has been added to `Cargo.toml`:
```toml
[dev-dependencies]
wasm-bindgen-test = "0.3"
```

## Test Structure

The test environment has been set up with the following structure:

```
neural-doc-flow-wasm/
├── tests/
│   ├── web.rs         # Browser-based tests
│   ├── node.rs        # Node.js tests
│   └── common/
│       └── mod.rs     # Shared test utilities
```

### Test Files Overview

1. **`tests/web.rs`** - Browser environment tests
   - Configured with `wasm_bindgen_test_configure!(run_in_browser)`
   - Tests WASM initialization
   - Tests document processor creation
   - Tests single document processing
   - Tests memory allocation

2. **`tests/node.rs`** - Node.js environment tests
   - Configured with `wasm_bindgen_test_configure!(run_in_node)`
   - Tests batch processing
   - Tests error handling
   - Tests performance metrics
   - Tests stream processing

3. **`tests/common/mod.rs`** - Shared utilities
   - `TestDocument` struct for creating test documents
   - `generate_test_documents()` for batch test data
   - `assert_js_contains()` for JsValue assertions
   - `measure_async()` for performance testing
   - `assert_completes_within()` for timeout testing

## Running Tests

### Browser Tests
```bash
# Run tests in Chrome (headless by default)
wasm-pack test --chrome

# Run tests in Chrome with GUI
wasm-pack test --chrome --no-headless

# Run tests in Firefox
wasm-pack test --firefox

# Run tests in Safari (macOS only)
wasm-pack test --safari
```

### Node.js Tests
```bash
# Run tests in Node.js
wasm-pack test --node

# Run specific test file in Node.js
wasm-pack test --node -- --test node
```

### All Tests
```bash
# Run tests in all available environments
wasm-pack test --chrome --firefox --node
```

### Development Mode
```bash
# Run tests in development mode (no optimization)
wasm-pack test --dev --chrome

# Run with verbose output
wasm-pack test --node -- --nocapture
```

## Test Filtering

To run specific tests:
```bash
# Run tests matching a pattern
wasm-pack test --node -- test_batch_processing

# Run tests from a specific module
wasm-pack test --chrome -- --test web
```

## Debugging Tests

### Browser Debugging
1. Run tests with `--no-headless` flag
2. Open browser developer tools
3. Set breakpoints in the generated JavaScript
4. Use `console.log()` via `web_sys::console::log_1()`

### Node.js Debugging
```bash
# Run with Node.js inspector
NODE_OPTIONS='--inspect-brk' wasm-pack test --node
```

## CI/CD Integration

For continuous integration, use:
```bash
# Install wasm-pack in CI
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Run all tests
wasm-pack test --chrome --firefox --node

# Generate coverage reports (requires additional setup)
wasm-pack test --node -- --coverage
```

## Common Issues and Solutions

### Issue: `wasm-pack: command not found`
**Solution**: Install wasm-pack using the installation commands above

### Issue: Browser tests fail to start
**Solution**: Ensure Chrome/Firefox is installed and accessible in PATH

### Issue: Node.js tests fail
**Solution**: Ensure Node.js version 14+ is installed

### Issue: Tests timeout
**Solution**: Increase timeout in test configuration or use `assert_completes_within()` utility

## Next Steps

1. **Add Integration Tests**: Create more comprehensive tests that test the full pipeline
2. **Performance Benchmarks**: Add benchmark tests using the timing utilities
3. **Coverage Reports**: Set up code coverage reporting with `cargo-tarpaulin`
4. **Cross-Platform Testing**: Test on different OS and browser combinations
5. **WASM Size Optimization**: Add tests to monitor and optimize WASM bundle size