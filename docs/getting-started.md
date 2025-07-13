# Getting Started with NeuralDocFlow

## Overview

NeuralDocFlow is a pure Rust document extraction platform that combines high-performance parsing with DAA (Distributed Autonomous Agents) coordination and neural enhancement to achieve >99% accuracy across all document types.

## Installation

### Prerequisites

- Rust 1.70+ with Cargo
- Optional: CUDA toolkit for GPU neural processing
- Optional: Python 3.8+ for Python bindings

### Basic Installation

Add NeuralDocFlow to your `Cargo.toml`:

```toml
[dependencies]
neuraldocflow = "1.0"
```

### Feature Selection

Choose optional features based on your needs:

```toml
[dependencies]
neuraldocflow = { version = "1.0", features = [
    "neural",     # Neural enhancement with ruv-FANN
    "python",     # Python bindings
    "wasm",       # WebAssembly support
    "pdf",        # PDF source (default)
    "docx",       # Microsoft Word documents
    "html",       # HTML/web pages
] }
```

### Development Installation

For development or to use the CLI:

```bash
# Clone repository
git clone https://github.com/neuraldocflow/neuraldocflow.git
cd neuraldocflow

# Build library
cargo build --release

# Build CLI
cargo build --release --bin neuraldocflow-cli

# Run tests
cargo test --all-features

# Run benchmarks
cargo bench
```

## Quick Start

### Basic PDF Extraction

```rust
use neuraldocflow::{DocFlow, SourceInput};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize extraction engine
    let docflow = DocFlow::new()?;
    
    // Create input from PDF file
    let input = SourceInput::File {
        path: PathBuf::from("example.pdf"),
        metadata: None,
    };
    
    // Extract document content
    let document = docflow.extract(input).await?;
    
    // Access results
    println!("Document: {}", document.id);
    println!("Confidence: {:.1}%", document.confidence * 100.0);
    println!("Content blocks: {}", document.content.len());
    
    // Get extracted text
    let text = document.get_text();
    println!("Extracted text:\n{}", text);
    
    // Get tables
    let tables = document.get_tables();
    println!("Found {} tables", tables.len());
    
    // Get images
    let images = document.get_images();
    println!("Found {} images", images.len());
    
    Ok(())
}
```

### Memory-Based Extraction

```rust
use neuraldocflow::{DocFlow, SourceInput};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let docflow = DocFlow::new()?;
    
    // Load document data into memory
    let data = std::fs::read("document.pdf")?;
    
    let input = SourceInput::Memory {
        data,
        filename: Some("document.pdf".to_string()),
        mime_type: Some("application/pdf".to_string()),
    };
    
    let document = docflow.extract(input).await?;
    
    // Process extracted content
    for (i, block) in document.content.iter().enumerate() {
        println!("Block {}: {:?}", i, block.block_type);
        if let Some(text) = &block.text {
            println!("  Text: {}", text.chars().take(100).collect::<String>());
        }
        println!("  Confidence: {:.1}%", block.metadata.confidence * 100.0);
    }
    
    Ok(())
}
```

### Batch Processing

```rust
use neuraldocflow::{DocFlow, SourceInput};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let docflow = DocFlow::new()?;
    
    // Prepare multiple inputs
    let inputs = vec![
        SourceInput::File { 
            path: PathBuf::from("report1.pdf"), 
            metadata: None 
        },
        SourceInput::File { 
            path: PathBuf::from("report2.pdf"), 
            metadata: None 
        },
        SourceInput::File { 
            path: PathBuf::from("invoice.pdf"), 
            metadata: None 
        },
    ];
    
    // Process all documents in parallel
    let documents = docflow.extract_batch(inputs).await?;
    
    // Analyze results
    for (i, doc) in documents.iter().enumerate() {
        println!("Document {}: {:.1}% confidence, {} blocks", 
            i + 1, 
            doc.confidence * 100.0,
            doc.content.len()
        );
        
        let stats = doc.get_content_stats();
        println!("  Paragraphs: {}", stats.paragraph_count);
        println!("  Tables: {}", stats.table_count);
        println!("  Images: {}", stats.image_count);
        println!("  Words: {}", stats.total_words);
    }
    
    Ok(())
}
```

## Configuration

### Basic Configuration

```rust
use neuraldocflow::{Config, DocFlow};
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = Config {
        extraction: ExtractionConfig {
            min_confidence: 0.85,           // Minimum confidence threshold
            max_content_blocks: 50000,      // Maximum blocks per document
            enable_parallel_processing: true, // Enable multi-threading
            processing_timeout: Duration::from_secs(120), // Max processing time
            ..Default::default()
        },
        ..Default::default()
    };
    
    let docflow = DocFlow::with_config(config)?;
    
    // Use configured instance
    Ok(())
}
```

### Neural Enhancement Configuration

```rust
use neuraldocflow::{Config, NeuralConfig};
use std::path::PathBuf;

let config = Config {
    neural: NeuralConfig {
        enabled: true,
        model_directory: PathBuf::from("./neural_models"),
        max_loaded_models: 4,
        processing: NeuralProcessingConfig {
            batch_size: 32,
            enable_gpu: true,
            inference_threads: 4,
            memory_limit: 2 * 1024 * 1024 * 1024, // 2GB
        },
        models: vec![
            ModelConfig {
                name: "layout_analyzer".to_string(),
                path: PathBuf::from("layout_v2.model"),
                enabled: true,
                confidence_threshold: 0.8,
            },
            ModelConfig {
                name: "text_enhancer".to_string(),
                path: PathBuf::from("text_enhancement.model"),
                enabled: true,
                confidence_threshold: 0.75,
            },
        ],
        ..Default::default()
    },
    ..Default::default()
};
```

### DAA Coordination Configuration

```rust
use neuraldocflow::{Config, DaaConfig};
use std::time::Duration;

let config = Config {
    daa: DaaConfig {
        max_agents: 8,                      // Maximum concurrent agents
        enable_consensus: true,             // Enable result validation
        consensus_threshold: 0.8,           // Agreement threshold
        coordination_timeout: Duration::from_secs(30),
        message_queue_size: 1000,           // Inter-agent message buffer
        health_check_interval: Duration::from_secs(10),
        agent_spawn_strategy: AgentSpawnStrategy::Adaptive,
        load_balancing: LoadBalancingStrategy::WorkStealing,
        ..Default::default()
    },
    ..Default::default()
};
```

## Source Plugin Configuration

### Built-in Sources

Configure built-in source plugins:

```rust
use neuraldocflow::{Config, SourcesConfig, SourceConfig};
use std::collections::HashMap;

let mut source_configs = HashMap::new();

// PDF source configuration
source_configs.insert("pdf".to_string(), SourceConfig {
    enabled: true,
    priority: 100,
    max_file_size: 100 * 1024 * 1024, // 100MB
    timeout: Duration::from_secs(60),
    specific_config: serde_json::json!({
        "enable_ocr": false,
        "extract_tables": true,
        "extract_images": true,
        "password_protected": false
    }),
});

// HTML source configuration
source_configs.insert("html".to_string(), SourceConfig {
    enabled: true,
    priority: 80,
    specific_config: serde_json::json!({
        "follow_links": false,
        "extract_metadata": true,
        "clean_html": true,
        "max_depth": 1
    }),
});

let config = Config {
    sources: SourcesConfig {
        source_configs,
        plugin_directories: vec![
            PathBuf::from("./plugins"),
            PathBuf::from("/usr/local/lib/neuraldocflow/plugins"),
        ],
        enable_hot_reload: true,
        discovery_interval: Duration::from_secs(30),
        ..Default::default()
    },
    ..Default::default()
};
```

### Custom Source Registration

```rust
use neuraldocflow::{DocFlow, sources::DocumentSource};

// Define custom source (see API documentation for full implementation)
struct JsonSource;

impl DocumentSource for JsonSource {
    // Implementation details...
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut docflow = DocFlow::new()?;
    
    // Register custom source
    docflow.register_source("json", Box::new(JsonSource))?;
    
    // Now can process JSON files
    let input = SourceInput::File {
        path: PathBuf::from("data.json"),
        metadata: None,
    };
    
    let document = docflow.extract(input).await?;
    
    Ok(())
}
```

## Performance Tuning

### Memory Optimization

```rust
use neuraldocflow::{Config, ExtractionConfig};

let config = Config {
    extraction: ExtractionConfig {
        // Limit memory usage for large documents
        max_content_blocks: 10000,
        enable_streaming: true,
        chunk_size: 1024 * 1024, // 1MB chunks
        
        // Optimize for memory over speed
        enable_parallel_processing: false,
        memory_pressure_threshold: 0.8,
        
        ..Default::default()
    },
    ..Default::default()
};
```

### High-Throughput Configuration

```rust
use neuraldocflow::{Config, ExtractionConfig, DaaConfig};

let config = Config {
    extraction: ExtractionConfig {
        enable_parallel_processing: true,
        worker_threads: num_cpus::get(),
        batch_size: 50,
        enable_prefetch: true,
        ..Default::default()
    },
    daa: DaaConfig {
        max_agents: num_cpus::get() * 2,
        agent_spawn_strategy: AgentSpawnStrategy::Aggressive,
        load_balancing: LoadBalancingStrategy::RoundRobin,
        enable_work_stealing: true,
        ..Default::default()
    },
    ..Default::default()
};
```

## Error Handling

### Comprehensive Error Handling

```rust
use neuraldocflow::{DocFlow, SourceInput, NeuralDocFlowError};

async fn robust_extraction(docflow: &DocFlow, input: SourceInput) -> Result<(), Box<dyn std::error::Error>> {
    match docflow.extract(input).await {
        Ok(document) => {
            if document.confidence < 0.8 {
                println!("Warning: Low confidence extraction ({:.1}%)", 
                    document.confidence * 100.0);
            }
            
            // Process successful extraction
            println!("Successfully extracted {} blocks", document.content.len());
        }
        
        Err(NeuralDocFlowError::Io(e)) => {
            eprintln!("File access error: {}", e);
            // Handle file system issues
        }
        
        Err(NeuralDocFlowError::Source { source, message }) => {
            eprintln!("Source plugin '{}' error: {}", source, message);
            // Handle source-specific issues
        }
        
        Err(NeuralDocFlowError::Coordination(e)) => {
            eprintln!("DAA coordination error: {}", e);
            // Handle agent coordination issues
        }
        
        Err(NeuralDocFlowError::Neural(e)) => {
            eprintln!("Neural processing error: {}", e);
            // Handle neural enhancement issues
        }
        
        Err(NeuralDocFlowError::Validation { field, expected, actual }) => {
            eprintln!("Validation error in {}: expected {}, got {}", field, expected, actual);
            // Handle input validation issues
        }
        
        Err(e) => {
            eprintln!("Unexpected error: {}", e);
            return Err(e.into());
        }
    }
    
    Ok(())
}
```

### Retry and Fallback Strategies

```rust
use neuraldocflow::{DocFlow, SourceInput, Config};
use std::time::Duration;

async fn extraction_with_retry(
    docflow: &DocFlow, 
    input: SourceInput,
    max_retries: usize
) -> Result<ExtractedDocument, Box<dyn std::error::Error>> {
    let mut last_error = None;
    
    for attempt in 0..=max_retries {
        match docflow.extract(input.clone()).await {
            Ok(document) => return Ok(document),
            Err(e) => {
                last_error = Some(e);
                if attempt < max_retries {
                    println!("Attempt {} failed, retrying...", attempt + 1);
                    tokio::time::sleep(Duration::from_secs(2_u64.pow(attempt as u32))).await;
                }
            }
        }
    }
    
    Err(last_error.unwrap().into())
}

// Usage with fallback configuration
async fn extraction_with_fallback(input: SourceInput) -> Result<ExtractedDocument, Box<dyn std::error::Error>> {
    // Try with full neural enhancement
    let full_config = Config {
        neural: NeuralConfig { enabled: true, ..Default::default() },
        daa: DaaConfig { enable_consensus: true, ..Default::default() },
        ..Default::default()
    };
    
    let docflow = DocFlow::with_config(full_config)?;
    
    match extraction_with_retry(&docflow, input.clone(), 2).await {
        Ok(document) => return Ok(document),
        Err(e) => {
            println!("Full extraction failed: {}, trying simplified mode", e);
        }
    }
    
    // Fallback: disable neural enhancement
    let simple_config = Config {
        neural: NeuralConfig { enabled: false, ..Default::default() },
        daa: DaaConfig { enable_consensus: false, max_agents: 1, ..Default::default() },
        ..Default::default()
    };
    
    let simple_docflow = DocFlow::with_config(simple_config)?;
    extraction_with_retry(&simple_docflow, input, 1).await
}
```

## CLI Usage

The NeuralDocFlow CLI provides a convenient interface for document processing:

### Basic Commands

```bash
# Extract single document
neuraldocflow-cli extract document.pdf --output extracted.json

# Batch processing
neuraldocflow-cli batch *.pdf --output-dir ./results/

# With custom configuration
neuraldocflow-cli extract document.pdf --config config.yaml --output result.json

# Enable neural enhancement
neuraldocflow-cli extract document.pdf --neural --models-dir ./models/

# Specify output format
neuraldocflow-cli extract document.pdf --format json --output document.json
neuraldocflow-cli extract document.pdf --format yaml --output document.yaml
neuraldocflow-cli extract document.pdf --format toml --output document.toml
```

### Configuration File

Create `config.yaml`:

```yaml
extraction:
  min_confidence: 0.85
  max_content_blocks: 50000
  enable_parallel_processing: true

neural:
  enabled: true
  model_directory: "./models"
  max_loaded_models: 4

daa:
  max_agents: 8
  enable_consensus: true
  consensus_threshold: 0.8

sources:
  pdf:
    enabled: true
    priority: 100
    extract_tables: true
    extract_images: true
  html:
    enabled: true
    priority: 80
    clean_html: true
```

## Next Steps

- [API Documentation](./api/) - Complete API reference
- [Examples](../examples/) - Additional usage examples
- [Architecture Guide](./architecture.md) - System design details
- [Performance Benchmarks](./benchmarks/) - Performance analysis
- [Contributing Guide](./CONTRIBUTING.md) - Development guidelines

## Troubleshooting

### Common Issues

**"No compatible source found"**
- Ensure the file extension is supported
- Check source plugin configuration
- Verify file is not corrupted

**"Neural model loading failed"**
- Check model file paths in configuration
- Ensure models are compatible versions
- Verify sufficient memory for model loading

**"DAA coordination timeout"**
- Increase `coordination_timeout` in DaaConfig
- Reduce `max_agents` if system is overloaded
- Check system resources (CPU, memory)

**Low extraction confidence**
- Enable neural enhancement
- Check document quality (scanned vs. digital)
- Adjust confidence thresholds
- Try different source configurations

For more help, see the [FAQ](./FAQ.md) or [file an issue](https://github.com/neuraldocflow/neuraldocflow/issues).