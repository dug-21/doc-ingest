# Neural Document Flow Sources Architecture

## Overview

The neural-doc-flow-sources module provides a modular, plugin-based architecture for handling various document formats. It uses lopdf for PDF parsing and supports extensible source implementations.

## Key Components

### 1. Traits Module (`src/traits.rs`)
- Re-exports `DocumentSource` trait from core
- Defines `SourceCapability` enum for capability reporting
- Provides `SourceMetadata` for plugin identification
- Includes `SourceConfig` for runtime configuration
- Defines `BaseDocumentSource` trait extending core functionality

### 2. Source Manager (`src/manager.rs`)
- Dynamic source registration and discovery
- Capability-based source lookup
- File extension and MIME type mapping
- Thread-safe async implementation using `Arc<RwLock<>>`
- Plugin lifecycle management

### 3. PDF Source (`src/pdf/mod.rs`)
- Uses **lopdf** library for PDF parsing (v0.32)
- Extracts text content from all pages
- Extracts document metadata (title, author, dates, etc.)
- Handles various PDF encodings
- Configurable file size limits

### 4. Text Source (`src/text/mod.rs`)
- Handles plain text files with various encodings
- Supports multiple extensions (txt, log, md, csv, tsv)
- Basic metadata extraction (line count, word count)
- Format detection hints (markdown, CSV, TSV)
- Encoding detection capabilities

## Usage Example

```rust
use neural_doc_flow_sources::{SourceManager, PdfSource, TextSource};
use std::sync::Arc;

async fn example() -> Result<()> {
    // Create source manager
    let manager = SourceManager::new();
    
    // Register sources
    manager.register_source(Arc::new(PdfSource::new())).await?;
    manager.register_source(Arc::new(TextSource::new())).await?;
    
    // Find appropriate source for a file
    let pdf_sources = manager.find_sources_by_extension("pdf").await;
    if let Some(source_id) = pdf_sources.first() {
        let source = manager.get_source(source_id).await.unwrap();
        // Process document with source
    }
    
    Ok(())
}
```

## Source Registration Flow

1. **Source Creation**: Each source implements `BaseDocumentSource`
2. **Registration**: Sources register with the manager providing metadata
3. **Discovery**: Manager indexes sources by:
   - Unique ID
   - File extensions
   - MIME types
   - Capabilities
4. **Usage**: Applications query manager to find appropriate sources

## Plugin Architecture Benefits

- **Extensibility**: Easy to add new document formats
- **Modularity**: Each source is self-contained
- **Configuration**: Runtime configuration per source
- **Discovery**: Automatic source selection based on file type
- **Capability Reporting**: Sources declare their features

## PDF Processing with lopdf

The PDF source uses lopdf for robust PDF handling:

```rust
// Parse PDF
let doc = LopdfDocument::load_mem(pdf_bytes)?;

// Extract text
for (page_num, page_id) in doc.get_pages() {
    if let Ok(content) = doc.extract_text(&[page_num]) {
        text.push_str(&content);
    }
}

// Extract metadata
if let Ok(info) = doc.trailer.get(b"Info") {
    // Process document info dictionary
}
```

## Future Enhancements

- Add more document formats (DOCX, HTML, XML)
- Implement remote document fetching
- Add file system monitoring capabilities
- Enhanced encoding detection
- Resource extraction from PDFs
- Parallel page processing for large PDFs
- Caching mechanisms for repeated processing