# NeuralDocFlow Source Plugin Development Guide

## Quick Start

This guide helps you create custom document source plugins for NeuralDocFlow.

## 1. Setting Up Your Plugin Project

### Create Plugin Structure

```bash
# Create plugin directory
mkdir my-source-plugin
cd my-source-plugin

# Initialize Rust project
cargo init --lib

# Set up directory structure
mkdir -p src/{core,readers,extractors,tests}
```

### Configure Cargo.toml

```toml
[package]
name = "my-source-plugin"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]  # Dynamic library for plugin

[dependencies]
neuraldocflow-sources = "1.0"
async-trait = "0.1"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1.0", features = ["full"] }
uuid = { version = "1.0", features = ["v4", "serde"] }

# Add format-specific dependencies
# For DOCX: zip = "0.6", quick-xml = "0.30"
# For HTML: scraper = "0.17", html5ever = "0.26"
# For Images: image = "0.24", imageproc = "0.23"

[dev-dependencies]
tokio-test = "0.4"
tempfile = "3.0"
```

## 2. Implementing Core Traits

### Basic Source Implementation

```rust
use async_trait::async_trait;
use neuraldocflow_sources::{
    DocumentSource, SourceInput, SourceConfig, SourceError,
    ExtractedDocument, ValidationResult,
};

pub struct MyDocumentSource {
    config: MySourceConfig,
    // Add source-specific fields
}

#[async_trait]
impl DocumentSource for MyDocumentSource {
    fn source_id(&self) -> &str {
        "my_format"
    }
    
    fn name(&self) -> &str {
        "My Document Format"
    }
    
    fn version(&self) -> &str {
        env!("CARGO_PKG_VERSION")
    }
    
    fn supported_extensions(&self) -> &[&str] {
        &["myf", "mdf"]
    }
    
    fn supported_mime_types(&self) -> &[&str] {
        &["application/x-my-format"]
    }
    
    async fn can_handle(&self, input: &SourceInput) -> Result<bool, SourceError> {
        // Implement format detection
        Ok(false)
    }
    
    async fn validate(&self, input: &SourceInput) -> Result<ValidationResult, SourceError> {
        // Implement validation
        Ok(ValidationResult::default())
    }
    
    async fn extract(&self, input: SourceInput) -> Result<ExtractedDocument, SourceError> {
        // Implement extraction
        todo!()
    }
    
    fn config_schema(&self) -> serde_json::Value {
        // Define configuration schema
        serde_json::json!({})
    }
    
    async fn initialize(&mut self, config: SourceConfig) -> Result<(), SourceError> {
        // Initialize with configuration
        Ok(())
    }
    
    async fn cleanup(&mut self) -> Result<(), SourceError> {
        // Cleanup resources
        Ok(())
    }
}
```

## 3. Common Implementation Patterns

### Format Detection

```rust
async fn can_handle(&self, input: &SourceInput) -> Result<bool, SourceError> {
    match input {
        SourceInput::File { path, .. } => {
            // Check by extension
            if let Some(ext) = path.extension() {
                if self.supported_extensions().contains(&ext.to_str().unwrap_or("")) {
                    return Ok(true);
                }
            }
            
            // Check by magic bytes
            let header = read_file_header(path, 16).await?;
            Ok(self.check_magic_bytes(&header))
        }
        SourceInput::Memory { data, mime_type, .. } => {
            // Check MIME type
            if let Some(mime) = mime_type {
                if self.supported_mime_types().contains(&mime.as_str()) {
                    return Ok(true);
                }
            }
            
            // Check magic bytes
            Ok(self.check_magic_bytes(data))
        }
        _ => Ok(false),
    }
}
```

### Content Extraction

```rust
async fn extract(&self, input: SourceInput) -> Result<ExtractedDocument, SourceError> {
    let start = Instant::now();
    
    // Read input data
    let data = self.read_input(input).await?;
    
    // Parse format
    let parsed = self.parse_format(&data).await?;
    
    // Extract content blocks
    let mut blocks = Vec::new();
    
    // Extract text
    blocks.extend(self.extract_text(&parsed).await?);
    
    // Extract tables
    if self.config.extract_tables {
        blocks.extend(self.extract_tables(&parsed).await?);
    }
    
    // Extract images
    if self.config.extract_images {
        blocks.extend(self.extract_images(&parsed).await?);
    }
    
    // Build document
    Ok(ExtractedDocument {
        id: Uuid::new_v4().to_string(),
        source_id: self.source_id().to_string(),
        metadata: self.extract_metadata(&parsed),
        content: blocks,
        structure: self.analyze_structure(&blocks),
        confidence: self.calculate_confidence(&parsed),
        metrics: ExtractionMetrics {
            extraction_time: start.elapsed(),
            pages_processed: parsed.page_count(),
            blocks_extracted: blocks.len(),
            memory_used: 0,
        },
    })
}
```

### Error Handling

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MySourceError {
    #[error("Invalid format: {0}")]
    InvalidFormat(String),
    
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),
    
    #[error("Parsing failed: {0}")]
    ParseError(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<MySourceError> for SourceError {
    fn from(err: MySourceError) -> Self {
        SourceError::Custom(err.to_string())
    }
}
```

## 4. Format-Specific Examples

### DOCX Source

```rust
use zip::ZipArchive;
use quick_xml::Reader;

pub struct DocxSource {
    config: DocxConfig,
}

impl DocxSource {
    async fn parse_docx(&self, data: &[u8]) -> Result<DocxDocument, SourceError> {
        let cursor = std::io::Cursor::new(data);
        let mut archive = ZipArchive::new(cursor)?;
        
        // Read document.xml
        let doc_xml = archive.by_name("word/document.xml")?;
        let mut reader = Reader::from_reader(doc_xml);
        
        // Parse XML content
        let mut paragraphs = Vec::new();
        let mut buf = Vec::new();
        
        loop {
            match reader.read_event(&mut buf) {
                Ok(Event::Start(ref e)) if e.name() == b"w:p" => {
                    // Parse paragraph
                    let para = self.parse_paragraph(&mut reader)?;
                    paragraphs.push(para);
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(SourceError::ParseError(e.to_string())),
                _ => {}
            }
            buf.clear();
        }
        
        Ok(DocxDocument { paragraphs })
    }
}
```

### HTML Source

```rust
use scraper::{Html, Selector};

pub struct HtmlSource {
    config: HtmlConfig,
}

impl HtmlSource {
    async fn parse_html(&self, data: &[u8]) -> Result<HtmlDocument, SourceError> {
        let html_string = String::from_utf8(data.to_vec())?;
        let document = Html::parse_document(&html_string);
        
        // Extract text content
        let mut blocks = Vec::new();
        
        // Extract paragraphs
        let p_selector = Selector::parse("p").unwrap();
        for element in document.select(&p_selector) {
            let text = element.text().collect::<String>();
            if !text.trim().is_empty() {
                blocks.push(ContentBlock {
                    block_type: BlockType::Paragraph,
                    text: Some(text),
                    // ... other fields
                });
            }
        }
        
        // Extract tables
        let table_selector = Selector::parse("table").unwrap();
        for table in document.select(&table_selector) {
            let table_block = self.parse_html_table(table)?;
            blocks.push(table_block);
        }
        
        Ok(HtmlDocument { blocks })
    }
}
```

### Image Source with OCR

```rust
use image::DynamicImage;

pub struct ImageSource {
    config: ImageConfig,
    ocr_engine: Arc<dyn OcrEngine>,
}

impl ImageSource {
    async fn extract_from_image(&self, data: &[u8]) -> Result<Vec<ContentBlock>, SourceError> {
        // Load image
        let img = image::load_from_memory(data)?;
        
        // Preprocess for OCR
        let processed = self.preprocess_image(&img);
        
        // Run OCR
        let ocr_result = self.ocr_engine.process(&processed).await?;
        
        // Convert to content blocks
        let blocks = ocr_result.regions.into_iter()
            .map(|region| ContentBlock {
                block_type: BlockType::Paragraph,
                text: Some(region.text),
                metadata: BlockMetadata {
                    confidence: region.confidence,
                    // ... other fields
                },
                position: BlockPosition {
                    x: region.bbox.x,
                    y: region.bbox.y,
                    width: region.bbox.width,
                    height: region.bbox.height,
                    // ... other fields
                },
                // ... other fields
            })
            .collect();
        
        Ok(blocks)
    }
    
    fn preprocess_image(&self, img: &DynamicImage) -> DynamicImage {
        // Convert to grayscale
        let gray = img.to_luma8();
        
        // Apply contrast enhancement
        // Apply noise reduction
        // Deskew if needed
        
        DynamicImage::ImageLuma8(gray)
    }
}
```

## 5. Plugin Configuration

### Define Configuration Schema

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MySourceConfig {
    /// Maximum file size to process
    pub max_file_size: usize,
    
    /// Enable advanced features
    pub advanced_features: AdvancedFeatures,
    
    /// Performance settings
    pub performance: PerformanceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedFeatures {
    pub extract_metadata: bool,
    pub extract_styles: bool,
    pub preserve_formatting: bool,
    pub detect_language: bool,
}

impl MyDocumentSource {
    fn config_schema(&self) -> serde_json::Value {
        json!({
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "max_file_size": {
                    "type": "integer",
                    "description": "Maximum file size in bytes",
                    "default": 52428800,
                    "minimum": 1024,
                    "maximum": 1073741824
                },
                "advanced_features": {
                    "type": "object",
                    "properties": {
                        "extract_metadata": {
                            "type": "boolean",
                            "default": true
                        },
                        "extract_styles": {
                            "type": "boolean",
                            "default": false
                        }
                    }
                }
            }
        })
    }
}
```

## 6. Testing Your Plugin

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_format_detection() {
        let source = MyDocumentSource::new();
        
        let input = SourceInput::Memory {
            data: b"MYFORMAT".to_vec(),
            filename: Some("test.myf".to_string()),
            mime_type: None,
        };
        
        assert!(source.can_handle(&input).await.unwrap());
    }
    
    #[tokio::test]
    async fn test_extraction() {
        let source = MyDocumentSource::new();
        
        // Create test document
        let test_data = create_test_document();
        let input = SourceInput::Memory {
            data: test_data,
            filename: Some("test.myf".to_string()),
            mime_type: Some("application/x-my-format".to_string()),
        };
        
        let result = source.extract(input).await.unwrap();
        
        assert!(!result.content.is_empty());
        assert_eq!(result.source_id, "my_format");
        assert!(result.confidence > 0.8);
    }
    
    #[tokio::test]
    async fn test_invalid_input() {
        let source = MyDocumentSource::new();
        
        let input = SourceInput::Memory {
            data: b"INVALID".to_vec(),
            filename: Some("test.txt".to_string()),
            mime_type: None,
        };
        
        assert!(!source.can_handle(&input).await.unwrap());
    }
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_with_plugin_manager() {
    // Create plugin manager
    let mut manager = SourcePluginManager::new(Default::default()).unwrap();
    
    // Register plugin
    let source = Arc::new(MyDocumentSource::new());
    manager.register_source("my_format".to_string(), source).await.unwrap();
    
    // Test document processing
    let input = SourceInput::File {
        path: PathBuf::from("test_data/sample.myf"),
        metadata: None,
    };
    
    let sources = manager.find_compatible_sources(&input).await.unwrap();
    assert!(!sources.is_empty());
    
    let doc = sources[0].extract(input).await.unwrap();
    assert_eq!(doc.source_id, "my_format");
}
```

## 7. Building and Packaging

### Build Script

```bash
#!/bin/bash
# build.sh

# Build in release mode
cargo build --release

# Strip symbols for smaller size
strip target/release/libmy_source_plugin.so

# Create plugin package
mkdir -p dist
cp target/release/libmy_source_plugin.so dist/
cp plugin.toml dist/
cp README.md dist/

# Create archive
tar -czf my-source-plugin-v1.0.0.tar.gz -C dist .
```

### Plugin Manifest

```toml
# plugin.toml
[plugin]
name = "my_source_plugin"
version = "1.0.0"
authors = ["Your Name <email@example.com>"]
description = "Support for My Document Format"
license = "MIT"
repository = "https://github.com/example/my-source-plugin"

[source]
id = "my_format"
name = "My Document Format"
extensions = ["myf", "mdf"]
mime_types = ["application/x-my-format"]

[dependencies]
neuraldocflow = ">=1.0.0"

[capabilities]
text_extraction = true
table_extraction = true
image_extraction = false
ocr_support = false
streaming = true
```

## 8. Performance Optimization

### Memory Management

```rust
use parking_lot::RwLock;
use std::sync::Arc;

pub struct OptimizedSource {
    // Use buffer pool for memory reuse
    buffer_pool: Arc<BufferPool>,
    
    // Cache parsed structures
    cache: Arc<RwLock<LruCache<String, ParsedDocument>>>,
}

impl OptimizedSource {
    async fn extract_streaming(&self, input: SourceInput) -> Result<ExtractedDocument, SourceError> {
        let mut reader = self.create_reader(input).await?;
        let mut blocks = Vec::new();
        
        // Process in chunks
        while let Some(chunk) = reader.next_chunk().await? {
            // Get buffer from pool
            let mut buffer = self.buffer_pool.acquire();
            
            // Process chunk
            let chunk_blocks = self.process_chunk(&chunk, &mut buffer).await?;
            blocks.extend(chunk_blocks);
            
            // Return buffer to pool
            self.buffer_pool.release(buffer);
            
            // Yield to prevent blocking
            tokio::task::yield_now().await;
        }
        
        Ok(self.build_document(blocks))
    }
}
```

### Parallel Processing

```rust
use rayon::prelude::*;

impl MyDocumentSource {
    async fn extract_parallel(&self, pages: Vec<Page>) -> Result<Vec<ContentBlock>, SourceError> {
        // Process pages in parallel using Rayon
        let blocks: Vec<Vec<ContentBlock>> = pages
            .par_iter()
            .map(|page| self.extract_page_content(page))
            .collect::<Result<Vec<_>, _>>()?;
        
        // Flatten results
        Ok(blocks.into_iter().flatten().collect())
    }
}
```

## 9. Distribution

### Publishing to crates.io

```toml
# In Cargo.toml
[package]
# ... existing fields ...
keywords = ["neuraldocflow", "plugin", "document", "extraction"]
categories = ["parsing", "text-processing"]

[package.metadata.docs.rs]
all-features = true
```

### Installing Plugins

Users can install your plugin by:

1. **Manual Installation**:
   ```bash
   # Download plugin
   wget https://example.com/my-source-plugin.so
   
   # Copy to plugin directory
   cp my-source-plugin.so ~/.neuraldocflow/plugins/
   
   # Update configuration
   echo 'sources.my_format.enabled = true' >> ~/.neuraldocflow/config.toml
   ```

2. **Using Plugin Manager**:
   ```bash
   neuraldocflow plugin install my-source-plugin
   ```

3. **From Source**:
   ```bash
   git clone https://github.com/example/my-source-plugin
   cd my-source-plugin
   cargo build --release
   neuraldocflow plugin install target/release/libmy_source_plugin.so
   ```

## 10. Best Practices

1. **Error Handling**: Always provide meaningful error messages
2. **Logging**: Use structured logging for debugging
3. **Documentation**: Document all public APIs
4. **Testing**: Aim for >80% test coverage
5. **Performance**: Profile and optimize hot paths
6. **Security**: Validate all inputs and sanitize outputs
7. **Compatibility**: Test with multiple NeuralDocFlow versions

## Support Resources

- API Documentation: https://docs.neuraldocflow.ai/api/sources
- Plugin Examples: https://github.com/neuraldocflow/plugin-examples
- Community Forum: https://forum.neuraldocflow.ai
- Discord: https://discord.gg/neuraldocflow