# Getting Started Guide

## Autonomous Document Extraction Platform - Quick Start

### Overview

This guide will help you quickly get started with the Autonomous Document Extraction Platform, from installation to processing your first document using the DocumentSource trait and neural processing capabilities.

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Rust**: Version 1.70 or later
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: 2GB free space for models and cache
- **Network**: Internet connection for downloading models

### Optional Requirements

- **CUDA**: For GPU acceleration (NVIDIA GPUs)
- **Docker**: For containerized deployment
- **Redis**: For distributed caching (production)

## Installation

### Option 1: Cargo Install (Recommended)

```bash
# Install from crates.io
cargo install doc-extract

# Verify installation
doc-extract --version
```

### Option 2: Build from Source

```bash
# Clone repository
git clone https://github.com/your-org/doc-extract.git
cd doc-extract

# Build project
cargo build --release

# Run tests
cargo test

# Install binary
cargo install --path .
```

### Option 3: Docker

```bash
# Pull pre-built image
docker pull docextract/platform:latest

# Or build locally
docker build -t doc-extract .

# Run container
docker run -p 8080:8080 doc-extract
```

## Quick Start: Process Your First Document

### 1. Basic Document Processing

Create a new Rust project and add the dependency:

```toml
[dependencies]
doc-extract = "1.0"
tokio = { version = "1.0", features = ["full"] }
```

```rust
use doc_extract::{DocumentProcessor, FileSource, ProcessingOptions};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the document processor
    let processor = DocumentProcessor::new().await?;
    
    // Create a file source
    let source = FileSource::new("example.pdf")?;
    
    // Configure processing options
    let options = ProcessingOptions::builder()
        .extract_text(true)
        .extract_entities(true)
        .classify_content(true)
        .build();
    
    // Process the document
    let result = processor.process(&source, options).await?;
    
    // Display results
    println!("Document Type: {:?}", result.classification.document_type);
    println!("Extracted Text: {}", result.extracted_text);
    println!("Entities Found: {}", result.entities.len());
    
    Ok(())
}
```

### 2. Processing Different Source Types

#### File Source
```rust
use doc_extract::FileSource;

// Process a local PDF file
let source = FileSource::new("document.pdf")?;
let result = processor.process(&source, options).await?;
```

#### URL Source
```rust
use doc_extract::UrlSource;

// Process a document from URL
let source = UrlSource::new("https://example.com/document.pdf")?;
let result = processor.process(&source, options).await?;
```

#### Base64 Source
```rust
use doc_extract::Base64Source;

// Process base64-encoded document
let base64_data = "data:application/pdf;base64,JVBERi0xLjQK...";
let source = Base64Source::new(base64_data)?;
let result = processor.process(&source, options).await?;
```

### 3. Using the DocumentSource Trait

Create custom sources by implementing the DocumentSource trait:

```rust
use doc_extract::{DocumentSource, DocumentContent, SourceError, SourceType, SourceMetadata};
use async_trait::async_trait;

pub struct DatabaseSource {
    document_id: String,
    connection: DatabaseConnection,
}

#[async_trait]
impl DocumentSource for DatabaseSource {
    async fn fetch(&self) -> Result<DocumentContent, SourceError> {
        let data = self.connection
            .fetch_document(&self.document_id)
            .await
            .map_err(|e| SourceError::FetchFailed(e.to_string()))?;
        
        Ok(DocumentContent::new(data))
    }
    
    fn source_type(&self) -> SourceType {
        SourceType::Database
    }
    
    fn validate(&self) -> Result<(), ValidationError> {
        if self.document_id.is_empty() {
            return Err(ValidationError::InvalidId);
        }
        Ok(())
    }
    
    fn metadata(&self) -> SourceMetadata {
        SourceMetadata::builder()
            .source_type("database")
            .identifier(&self.document_id)
            .build()
    }
}
```

## Configuration

### Basic Configuration

Create a `config.toml` file:

```toml
[server]
host = "127.0.0.1"
port = 8080
workers = 4

[neural]
models_path = "./models"
device = "auto"  # "cpu", "cuda", or "auto"
batch_size = 8

[daa]
max_agents = 16
topology = "hierarchical"  # "mesh", "ring", "star", "hierarchical"
coordination_timeout = 30

[storage]
cache_dir = "./cache"
max_cache_size = "1GB"
temp_cleanup_interval = 3600

[logging]
level = "info"
format = "json"
file = "./logs/doc-extract.log"
```

### Environment Variables

Override configuration with environment variables:

```bash
export DOC_EXTRACT_HOST=0.0.0.0
export DOC_EXTRACT_PORT=3000
export DOC_EXTRACT_NEURAL_DEVICE=cuda
export DOC_EXTRACT_LOG_LEVEL=debug
```

### Advanced Configuration

```toml
[neural.models.text_extraction]
name = "bert-base-uncased"
path = "./models/text-extraction"
enabled = true

[neural.models.classification]
name = "document-classifier-v2"
path = "./models/classification"
enabled = true

[neural.models.entity_recognition]
name = "ner-large"
path = "./models/ner"
enabled = true

[daa.agents.pdf]
max_instances = 4
specialization = ["pdf", "document"]

[daa.agents.image]
max_instances = 2
specialization = ["image", "ocr"]

[api]
cors_origins = ["http://localhost:3000", "https://myapp.com"]
rate_limit = 1000  # requests per hour
max_file_size = "100MB"
```

## Using with Claude Flow Integration

### Enable DAA with Claude Flow

```rust
use doc_extract::{DocumentProcessor, DAAConfig, ClaudeFlowConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure Claude Flow integration
    let claude_config = ClaudeFlowConfig::builder()
        .swarm_topology("hierarchical")
        .max_agents(8)
        .enable_neural_training(true)
        .memory_persistence(true)
        .build();
    
    // Initialize processor with DAA
    let processor = DocumentProcessor::builder()
        .enable_daa(true)
        .claude_flow_config(claude_config)
        .build()
        .await?;
    
    // Process documents with intelligent agent coordination
    let sources = vec![
        FileSource::new("document1.pdf")?,
        UrlSource::new("https://example.com/doc2.pdf")?,
        FileSource::new("document3.docx")?,
    ];
    
    // Process batch with automatic agent allocation
    let results = processor.process_batch(sources).await?;
    
    for (i, result) in results.iter().enumerate() {
        println!("Document {}: {:?}", i + 1, result.classification);
    }
    
    Ok(())
}
```

### Claude Flow Hooks Integration

Enable automatic coordination hooks:

```bash
# Install Claude Flow CLI
npm install -g claude-flow@alpha

# Configure hooks in your project
npx claude-flow@alpha init --rust

# Run with automatic coordination
cargo run --features="claude-flow"
```

## Neural Model Setup

### Download Pre-trained Models

```bash
# Download default models
doc-extract models download

# List available models
doc-extract models list

# Download specific model
doc-extract models download --name bert-large-uncased
```

### Custom Model Integration

```rust
use doc_extract::{NeuralProcessor, ModelConfig};

// Load custom model
let model_config = ModelConfig::builder()
    .name("custom-classifier")
    .path("./models/custom")
    .model_type("classification")
    .device("cuda")
    .build();

let processor = NeuralProcessor::with_custom_model(model_config).await?;
```

## API Server Setup

### Start the HTTP Server

```rust
use doc_extract::server::ApiServer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let server = ApiServer::builder()
        .host("0.0.0.0")
        .port(8080)
        .enable_auth(true)
        .enable_rate_limiting(true)
        .build();
    
    println!("Starting server on http://0.0.0.0:8080");
    server.run().await?;
    
    Ok(())
}
```

### API Usage Examples

```bash
# Process document via API
curl -X POST http://localhost:8080/api/v1/documents/process \
  -H "Content-Type: application/json" \
  -d '{
    "source": {
      "type": "url",
      "value": "https://example.com/document.pdf"
    },
    "options": {
      "extract_text": true,
      "extract_entities": true
    }
  }'

# Get processing status
curl http://localhost:8080/api/v1/documents/task_123/status

# Get results
curl http://localhost:8080/api/v1/documents/task_123/results
```

## Performance Optimization

### GPU Acceleration

```toml
[neural]
device = "cuda"
gpu_memory_fraction = 0.8
enable_mixed_precision = true
```

### Batch Processing

```rust
// Process multiple documents efficiently
let sources = vec![
    FileSource::new("doc1.pdf")?,
    FileSource::new("doc2.pdf")?,
    FileSource::new("doc3.pdf")?,
];

let results = processor
    .batch_processor()
    .batch_size(4)
    .parallel_processing(true)
    .process(sources)
    .await?;
```

### Memory Optimization

```rust
// Configure for large documents
let processor = DocumentProcessor::builder()
    .streaming_threshold(100_000_000) // 100MB
    .memory_pool_size(2_000_000_000)  // 2GB
    .enable_memory_mapping(true)
    .build()
    .await?;
```

## Monitoring and Debugging

### Enable Detailed Logging

```rust
use tracing_subscriber;

// Initialize logging
tracing_subscriber::fmt()
    .with_max_level(tracing::Level::DEBUG)
    .with_target(false)
    .init();
```

### Performance Metrics

```rust
// Enable metrics collection
let processor = DocumentProcessor::builder()
    .enable_metrics(true)
    .metrics_endpoint("http://localhost:9090")
    .build()
    .await?;

// Access metrics
let metrics = processor.get_metrics().await?;
println!("Processing time: {:?}", metrics.average_processing_time);
println!("Success rate: {:.2}%", metrics.success_rate * 100.0);
```

## Common Patterns

### Error Handling

```rust
use doc_extract::{ProcessingError, SourceError};

match processor.process(&source, options).await {
    Ok(result) => {
        println!("Success: {} entities found", result.entities.len());
    }
    Err(ProcessingError::SourceError(SourceError::NotFound)) => {
        eprintln!("Document not found");
    }
    Err(ProcessingError::NeuralError(e)) => {
        eprintln!("Neural processing failed: {}", e);
    }
    Err(e) => {
        eprintln!("Processing failed: {}", e);
    }
}
```

### Async Processing with Progress Updates

```rust
use doc_extract::ProgressCallback;

let progress_callback = |stage: &str, progress: f32| {
    println!("Stage: {}, Progress: {:.1}%", stage, progress * 100.0);
};

let result = processor
    .process_with_progress(&source, options, progress_callback)
    .await?;
```

### Custom Processing Pipeline

```rust
use doc_extract::{Pipeline, Stage};

let pipeline = Pipeline::builder()
    .add_stage(Stage::Validation)
    .add_stage(Stage::ContentExtraction)
    .add_stage(Stage::TextProcessing)
    .add_stage(Stage::NeuralAnalysis)
    .add_stage(Stage::EntityExtraction)
    .add_stage(Stage::Classification)
    .add_stage(Stage::ResultSynthesis)
    .build();

let result = pipeline.process(&source).await?;
```

## Next Steps

1. **Explore Advanced Features**: Learn about custom agents and neural model fine-tuning
2. **Production Deployment**: Set up monitoring, logging, and scaling
3. **Integration Patterns**: Integrate with your existing applications
4. **Performance Tuning**: Optimize for your specific use cases
5. **Community**: Join our community forum for tips and best practices

## Troubleshooting

### Common Issues

**Model Download Fails**:
```bash
# Check network connectivity
curl -I https://models.doc-extract.com/health

# Manual download
doc-extract models download --force --model bert-base-uncased
```

**CUDA Out of Memory**:
```toml
[neural]
device = "cpu"
# Or reduce batch size
batch_size = 1
```

**Performance Issues**:
```bash
# Enable performance profiling
RUST_LOG=debug cargo run --features=profiling
```

For more troubleshooting tips, see the [Troubleshooting Guide](troubleshooting.md).

## Support

- **Documentation**: https://docs.doc-extract.com
- **GitHub Issues**: https://github.com/your-org/doc-extract/issues
- **Community Forum**: https://community.doc-extract.com
- **Discord**: https://discord.gg/doc-extract