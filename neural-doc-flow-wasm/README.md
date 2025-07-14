# Neural Document Flow WASM

WebAssembly bindings for neural document processing, enabling high-performance document processing in web browsers and Node.js environments.

## Features

- **JavaScript API**: Complete WASM bindings with TypeScript definitions
- **Streaming Support**: Process large documents with streaming API
- **Browser Compatible**: Works in modern web browsers
- **Node.js Compatible**: Works in Node.js environments
- **Error Handling**: Comprehensive error types with recovery strategies
- **Memory Efficient**: Optimized for WASM memory constraints
- **Performance Monitoring**: Built-in performance tracking

## API Overview

### Core Classes

- `WasmDocumentProcessor`: Main document processing interface
- `WasmStreamingProcessor`: Streaming processor for large documents
- `WasmProcessingConfig`: Configuration options
- `WasmProcessingResult`: Processing results
- `WasmUtils`: Utility functions

### Basic Usage

```javascript
import init, { WasmDocumentProcessor, default_config } from './pkg/neural_doc_flow_wasm.js';

// Initialize WASM module
await init();

// Create processor
const config = default_config();
const processor = new WasmDocumentProcessor(config);

// Process document
const bytes = new TextEncoder().encode("Document content");
const result = await processor.process_bytes(bytes, "document.txt");

console.log("Extracted content:", result.content);
console.log("Processing time:", result.processing_time_ms, "ms");
```

### Streaming Processing

```javascript
import { WasmStreamingProcessor } from './pkg/neural_doc_flow_wasm.js';

const processor = new WasmStreamingProcessor(config);
const result = await processor.process_stream(readableStream);
```

### File Processing

```javascript
// Process File object (browser)
const result = await processor.process_file(fileInput.files[0]);

// Process Blob
const result = await processor.process_blob(blob);

// Batch processing
const results = await processor.process_batch([file1, file2, file3]);
```

## Configuration Options

```javascript
const config = {
    neural_enhancement: true,      // Enable neural processing
    max_file_size: 50 * 1024 * 1024, // 50MB limit
    timeout_ms: 30000,             // 30 second timeout
    security_level: 2,             // Security scanning level (0-3)
    output_formats: ["text", "json"], // Output formats
    custom_options: {}             // Custom parameters
};
```

## Building

### Prerequisites

- Rust toolchain with `wasm-pack`
- Node.js (for examples)

### Build Commands

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build for web
wasm-pack build --target web --out-dir pkg

# Build for Node.js
wasm-pack build --target nodejs --out-dir pkg-node

# Build for bundler
wasm-pack build --target bundler --out-dir pkg-bundler
```

### Optimization

The WASM binary is optimized for size and performance:

- `wee_alloc` for smaller binary size
- `wasm-opt` optimizations
- Dead code elimination
- SIMD optimizations where available

## Examples

See the `examples/` directory for complete usage examples:

- `basic_usage.js`: Simple document processing
- `streaming_example.js`: Large document streaming
- `batch_processing.js`: Multiple document processing
- `error_handling.js`: Error handling patterns

## Performance

- **Small binary size**: ~200KB compressed
- **Fast initialization**: <10ms typical
- **Memory efficient**: <2MB per document
- **Streaming capable**: Handle files of any size
- **SIMD accelerated**: 4x faster on supported platforms

## Browser Compatibility

- Chrome 57+ (WebAssembly support)
- Firefox 52+ (WebAssembly support)
- Safari 11+ (WebAssembly support)
- Edge 16+ (WebAssembly support)

## Security

- Input validation on all boundaries
- Memory-safe operations
- Sandboxed execution environment
- No file system access from WASM
- Content Security Policy compatible

## Limitations

- No direct file system access
- Limited to WASM memory constraints
- Some features require async/await support
- Browser security restrictions apply