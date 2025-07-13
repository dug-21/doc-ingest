# Neural Document Flow Core

Core traits, types, and functionality for the neural document processing framework.

## Overview

This crate provides the foundational abstractions for the neural document flow system:

### Core Traits
- `DocumentSource` - Async, plugin-based document sources
- `ProcessorPipeline` - Processing pipeline abstractions
- `OutputFormatter` - Output formatting capabilities  
- `NeuralProcessor` - Neural processing abstractions

### Core Structures
- `Document` - Multimodal document representation with support for text, images, tables, and structured data
- `ProcessingResultData` - Processing results with confidence scores and metadata
- Comprehensive error types using `thiserror`
- Centralized configuration structures

### Utilities
- `DocumentBuilder` - Convenient document creation
- `ResultBuilder` - Timed result building with automatic metrics
- `DocumentUtils` - File loading, MIME type detection, and document operations
- `ResultUtils` - Result combination, mapping, and quality scoring

## Features

- `default` - Enables performance optimizations
- `performance` - Rayon, crossbeam, and parking_lot for parallel processing
- `monitoring` - Metrics and OpenTelemetry support
- `simd` - SIMD optimizations with ndarray
- `neural` - Neural processing support (requires simd)

## Dependencies

All dependencies are managed through the workspace to ensure version consistency across the project.

## Usage

```rust
use neural_doc_flow_core::prelude::*;

// Create a document
let doc = DocumentBuilder::new()
    .title("My Document")
    .source("file.pdf")
    .mime_type("application/pdf")
    .text_content("Document content")
    .build();

// Create a processing result
let result = ResultBuilder::new()
    .data(doc)
    .confidence(0.95)
    .processor_version("v1.0")
    .build()
    .unwrap();
```

## Architecture

This crate follows a pure Rust architecture with:
- Async-first design using `async-trait`
- Plugin-based extensibility through trait objects
- Comprehensive error handling
- Serialization support with `serde`
- Performance optimizations through optional features

## Testing

Run tests with:
```bash
cargo test -p neural-doc-flow-core
```

All core functionality is thoroughly tested with unit tests.