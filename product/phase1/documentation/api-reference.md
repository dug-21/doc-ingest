# NeuralDocFlow API Documentation

## Overview

NeuralDocFlow provides a comprehensive Rust library for document extraction with DAA coordination and neural enhancement. This documentation covers all public APIs and usage patterns.

## Quick Reference

### Core Types

- [`DocFlow`] - Main extraction engine
- [`ExtractedDocument`] - Extraction results
- [`ContentBlock`] - Individual content elements
- [`SourceInput`] - Input types for extraction

### Source System

- [`DocumentSource`] - Plugin trait for document formats
- [`SourceManager`] - Source plugin management
- [`ValidationResult`] - Input validation results

### DAA Coordination

- [`DaaCoordinator`] - Distributed agent coordination
- [`Agent`] - Individual processing agents
- [`MessageBus`] - Inter-agent communication

### Neural Enhancement

- [`NeuralEngine`] - Neural processing engine
- [`NeuralModel`] - Model trait for neural enhancement
- [`ModelLoader`] - Neural model management

### Configuration

- [`Config`] - System configuration
- [`DaaConfig`] - DAA coordination settings
- [`NeuralConfig`] - Neural processing configuration

### Error Handling

- [`NeuralDocFlowError`] - Comprehensive error types
- [`Result<T>`] - Standard result type

## Usage Patterns

### Basic Extraction

```rust
use neuraldocflow::{DocFlow, SourceInput};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let docflow = DocFlow::new()?;
    
    let input = SourceInput::File {
        path: PathBuf::from("document.pdf"),
        metadata: None,
    };
    
    let document = docflow.extract(input).await?;
    
    println!("Extracted {} content blocks", document.content.len());
    Ok(())
}
```

### Custom Source Plugin

```rust
use neuraldocflow::sources::{DocumentSource, SourceInput, ValidationResult};
use neuraldocflow::core::ExtractedDocument;
use async_trait::async_trait;

pub struct CustomSource;

#[async_trait]
impl DocumentSource for CustomSource {
    fn source_id(&self) -> &str { "custom" }
    fn name(&self) -> &str { "Custom Format Source" }
    fn version(&self) -> &str { "1.0.0" }
    fn supported_extensions(&self) -> &[&str] { &["custom"] }
    fn supported_mime_types(&self) -> &[&str] { &["application/x-custom"] }
    
    async fn can_handle(&self, input: &SourceInput) -> Result<bool> {
        // Implementation
        Ok(false)
    }
    
    async fn validate(&self, input: &SourceInput) -> Result<ValidationResult> {
        // Implementation
        Ok(ValidationResult::valid())
    }
    
    async fn extract(&self, input: SourceInput) -> Result<ExtractedDocument> {
        // Implementation
        Ok(ExtractedDocument::new(self.source_id().to_string()))
    }
    
    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({})
    }
    
    async fn initialize(&mut self, config: SourceConfig) -> Result<()> {
        Ok(())
    }
    
    async fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
}
```

## Performance Considerations

### Memory Management

- Use `SourceInput::Stream` for large files to avoid loading entire documents into memory
- Configure `max_loaded_models` in `NeuralConfig` based on available RAM
- Set appropriate `message_queue_size` in `DaaConfig` for optimal throughput

### Concurrency

- `DocFlow` is thread-safe and can be shared across async tasks
- Use `extract_batch` for processing multiple documents efficiently
- Configure `max_agents` based on available CPU cores

### Caching

- Source plugins may cache parsed document structures
- Neural models are loaded once and reused
- Configure appropriate cache sizes based on workload

## Error Handling

All APIs return `Result<T, NeuralDocFlowError>` for comprehensive error handling:

```rust
match docflow.extract(input).await {
    Ok(document) => {
        // Process successful extraction
        println!("Extracted document with {:.1}% confidence", document.confidence * 100.0);
    }
    Err(NeuralDocFlowError::Io(e)) => {
        eprintln!("IO error: {}", e);
    }
    Err(NeuralDocFlowError::Source { source, message }) => {
        eprintln!("Source error in {}: {}", source, message);
    }
    Err(NeuralDocFlowError::Coordination(e)) => {
        eprintln!("DAA coordination error: {}", e);
    }
    Err(e) => {
        eprintln!("Other error: {}", e);
    }
}
```

## Thread Safety

All public types are designed to be thread-safe where appropriate:

- `DocFlow`: `Send + Sync` - can be shared across threads
- `ExtractedDocument`: `Send + Sync` - safe to pass between threads
- `Config`: `Send + Sync` - immutable after creation
- `DocumentSource`: `Send + Sync` - implemented by all source plugins

## Feature Flags

Enable optional functionality through Cargo features:

```toml
[dependencies]
neuraldocflow = { version = "1.0", features = ["neural", "python", "wasm"] }
```

Available features:
- `neural`: Neural enhancement with ruv-FANN integration
- `python`: Python bindings via PyO3
- `wasm`: WebAssembly compilation support
- `pdf`: PDF source plugin (enabled by default)
- `docx`: Microsoft Word source plugin
- `html`: HTML source plugin

## See Also

- [Getting Started Guide](../getting-started.md)
- [Examples](../../examples/)
- [Architecture Documentation](../architecture.md)