# Rustdoc Documentation Standards

## Comprehensive Code Documentation Guidelines

### Overview

This guide establishes the standards for rustdoc documentation in the Autonomous Document Extraction Platform. Proper documentation is essential for maintainability, user adoption, and long-term project success.

## Documentation Requirements

### 1. All Public Items Must Be Documented

Every public trait, struct, enum, function, and module must have comprehensive documentation:

```rust
#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)] // Optional but recommended

/// Autonomous Document Extraction Platform
/// 
/// This crate provides comprehensive document processing capabilities using
/// neural networks and dynamic agent allocation for intelligent extraction
/// and analysis of content from various document sources.
/// 
/// # Features
/// 
/// - **DocumentSource Trait**: Unified interface for any document input
/// - **Neural Processing**: AI-powered content analysis and extraction
/// - **Dynamic Agent Allocation**: Intelligent task distribution
/// - **Production API**: RESTful interface with authentication
/// 
/// # Quick Start
/// 
/// ```rust
/// use doc_extract::{DocumentProcessor, FileSource, ProcessingOptions};
/// 
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let processor = DocumentProcessor::new().await?;
///     let source = FileSource::new("document.pdf")?;
///     let options = ProcessingOptions::default();
///     
///     let result = processor.process(&source, options).await?;
///     println!("Extracted {} entities", result.entities.len());
///     
///     Ok(())
/// }
/// ```
pub mod doc_extract {
    // Module contents...
}
```

### 2. Trait Documentation Standards

```rust
/// Core abstraction for document sources
/// 
/// The `DocumentSource` trait provides a unified interface for accessing
/// documents from any source, whether local files, remote URLs, databases,
/// or cloud storage. This abstraction enables the processing pipeline to
/// work with any document source without modification.
/// 
/// # Design Principles
/// 
/// - **Async-first**: All operations are async for optimal performance
/// - **Error-safe**: Comprehensive error handling with specific error types
/// - **Extensible**: Easy to implement for new source types
/// - **Efficient**: Minimal overhead and resource usage
/// 
/// # Implementation Guide
/// 
/// To implement a custom source, you need to provide:
/// 
/// 1. **fetch()**: Retrieve document content asynchronously
/// 2. **source_type()**: Identify the type of source
/// 3. **validate()**: Verify source configuration
/// 4. **metadata()**: Provide source information
/// 
/// # Examples
/// 
/// ## Basic File Source
/// 
/// ```rust
/// use doc_extract::{DocumentSource, FileSource};
/// 
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let source = FileSource::new("document.pdf")?;
///     let content = source.fetch().await?;
///     println!("File size: {} bytes", content.data.len());
///     Ok(())
/// }
/// ```
/// 
/// ## Custom Database Source
/// 
/// ```rust
/// use doc_extract::{DocumentSource, DocumentContent, SourceError};
/// use async_trait::async_trait;
/// 
/// pub struct DatabaseSource {
///     connection: DatabaseConnection,
///     document_id: String,
/// }
/// 
/// #[async_trait]
/// impl DocumentSource for DatabaseSource {
///     async fn fetch(&self) -> Result<DocumentContent, SourceError> {
///         // Implementation details...
///         # Ok(DocumentContent::default())
///     }
///     
///     // Other required methods...
///     # fn source_type(&self) -> doc_extract::SourceType { todo!() }
///     # fn validate(&self) -> Result<(), doc_extract::ValidationError> { todo!() }
///     # fn metadata(&self) -> doc_extract::SourceMetadata { todo!() }
/// }
/// ```
/// 
/// # Error Handling
/// 
/// All methods should return appropriate error types:
/// 
/// - [`SourceError::NotFound`] - Source doesn't exist
/// - [`SourceError::AccessDenied`] - Permission issues
/// - [`SourceError::NetworkError`] - Network connectivity problems
/// - [`SourceError::InvalidSource`] - Configuration errors
/// 
/// # Performance Considerations
/// 
/// - Use streaming for large documents (>100MB)
/// - Implement connection pooling for remote sources
/// - Cache frequently accessed content
/// - Validate sources before expensive operations
/// 
/// # Thread Safety
/// 
/// All implementations must be `Send + Sync` for use in multi-threaded
/// environments. Use appropriate synchronization primitives when needed.
#[async_trait::async_trait]
pub trait DocumentSource: Send + Sync {
    /// Fetches the document content from the source
    /// 
    /// This method retrieves the actual document data and returns it as
    /// a [`DocumentContent`] struct containing the raw bytes, content type,
    /// and metadata.
    /// 
    /// # Errors
    /// 
    /// Returns [`SourceError`] if:
    /// - Source is not accessible (network, permissions, etc.)
    /// - Source doesn't exist or has been deleted
    /// - Content is corrupted or invalid
    /// - Timeout occurs during fetch operation
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use doc_extract::{FileSource, DocumentSource};
    /// 
    /// #[tokio::main]
    /// async fn main() -> Result<(), Box<dyn std::error::Error>> {
    ///     let source = FileSource::new("example.pdf")?;
    ///     
    ///     match source.fetch().await {
    ///         Ok(content) => {
    ///             println!("Successfully fetched {} bytes", content.data.len());
    ///             println!("Content type: {}", content.content_type);
    ///         }
    ///         Err(e) => {
    ///             eprintln!("Failed to fetch document: {}", e);
    ///         }
    ///     }
    ///     
    ///     Ok(())
    /// }
    /// ```
    /// 
    /// # Performance Notes
    /// 
    /// For large documents, consider using streaming to avoid loading
    /// the entire document into memory:
    /// 
    /// ```rust,ignore
    /// if estimated_size > STREAMING_THRESHOLD {
    ///     return Ok(DocumentContent::streaming(reader));
    /// }
    /// ```
    async fn fetch(&self) -> Result<DocumentContent, SourceError>;
    
    /// Returns the type of this source
    /// 
    /// This method identifies the category of source (file, URL, database, etc.)
    /// which can be used for optimization and routing decisions.
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use doc_extract::{FileSource, DocumentSource, SourceType};
    /// 
    /// let source = FileSource::new("document.pdf").unwrap();
    /// assert_eq!(source.source_type(), SourceType::File);
    /// ```
    fn source_type(&self) -> SourceType;
    
    /// Validates the source configuration without fetching content
    /// 
    /// This method performs lightweight validation to check if the source
    /// is properly configured and likely to succeed when fetched. It should
    /// be called before expensive operations.
    /// 
    /// # Errors
    /// 
    /// Returns [`ValidationError`] if:
    /// - Source configuration is invalid
    /// - Required parameters are missing
    /// - Source is not accessible
    /// - Source exceeds size limits
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use doc_extract::{UrlSource, DocumentSource};
    /// 
    /// let source = UrlSource::new("https://example.com/doc.pdf").unwrap();
    /// 
    /// if let Err(e) = source.validate() {
    ///     eprintln!("Source validation failed: {}", e);
    ///     return;
    /// }
    /// 
    /// // Proceed with fetch operation
    /// let content = source.fetch().await?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn validate(&self) -> Result<(), ValidationError>;
    
    /// Returns metadata about the source
    /// 
    /// Provides information about the source such as estimated size,
    /// content type, modification time, and other relevant details
    /// that can be used for optimization and caching decisions.
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use doc_extract::{FileSource, DocumentSource};
    /// 
    /// let source = FileSource::new("document.pdf").unwrap();
    /// let metadata = source.metadata();
    /// 
    /// println!("Source: {}", metadata.identifier());
    /// println!("Type: {}", metadata.source_type());
    /// if let Some(size) = metadata.estimated_size() {
    ///     println!("Estimated size: {} bytes", size);
    /// }
    /// ```
    fn metadata(&self) -> SourceMetadata;
}
```

### 3. Struct Documentation Standards

```rust
/// File-based document source implementation
/// 
/// `FileSource` provides access to documents stored in the local filesystem.
/// It supports all common document formats and handles file validation,
/// content type detection, and efficient reading operations.
/// 
/// # Features
/// 
/// - **Automatic content type detection** based on file extension and magic bytes
/// - **Large file support** with streaming for files over 100MB
/// - **Path validation** with security checks for path traversal
/// - **Metadata extraction** including file size and modification time
/// 
/// # Security
/// 
/// FileSource implements several security measures:
/// - Path traversal protection (prevents access to `../` paths)
/// - File existence and permission validation
/// - Size limit enforcement to prevent DoS attacks
/// - Content scanning for malicious files (when enabled)
/// 
/// # Examples
/// 
/// ## Basic Usage
/// 
/// ```rust
/// use doc_extract::FileSource;
/// 
/// // Create source for PDF file
/// let source = FileSource::new("documents/report.pdf")?;
/// 
/// // Validate before processing
/// source.validate()?;
/// 
/// // Fetch content
/// let content = source.fetch().await?;
/// println!("Loaded {} bytes from {}", content.data.len(), source.path().display());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
/// 
/// ## Advanced Configuration
/// 
/// ```rust
/// use doc_extract::FileSource;
/// use std::time::Duration;
/// 
/// let source = FileSource::builder()
///     .path("large_document.pdf")
///     .max_size(100_000_000) // 100MB limit
///     .streaming_threshold(10_000_000) // Stream files >10MB
///     .content_type_override("application/pdf")
///     .read_timeout(Duration::from_secs(30))
///     .build()?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
/// 
/// ## Error Handling
/// 
/// ```rust
/// use doc_extract::{FileSource, SourceError};
/// 
/// match FileSource::new("nonexistent.pdf") {
///     Ok(source) => {
///         // Source created successfully
///         match source.fetch().await {
///             Ok(content) => println!("Success: {} bytes", content.data.len()),
///             Err(SourceError::NotFound(path)) => {
///                 eprintln!("File not found: {}", path);
///             }
///             Err(SourceError::AccessDenied(msg)) => {
///                 eprintln!("Permission denied: {}", msg);
///             }
///             Err(e) => eprintln!("Other error: {}", e),
///         }
///     }
///     Err(e) => eprintln!("Failed to create source: {}", e),
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
/// 
/// # Platform Support
/// 
/// FileSource works on all platforms supported by Rust:
/// - **Linux**: Full support with extended attributes
/// - **macOS**: Full support with resource forks
/// - **Windows**: Full support with NTFS streams
/// 
/// # Performance
/// 
/// - **Small files** (<10MB): Loaded into memory for fastest access
/// - **Large files** (>10MB): Streamed to reduce memory usage
/// - **Huge files** (>1GB): Memory-mapped for efficient random access
/// 
/// # Thread Safety
/// 
/// FileSource is thread-safe and can be shared across threads. File handles
/// are opened per operation to avoid sharing issues.
#[derive(Debug, Clone)]
pub struct FileSource {
    path: PathBuf,
    config: FileSourceConfig,
}

impl FileSource {
    /// Creates a new file source from a path
    /// 
    /// This constructor validates the path and creates a FileSource with
    /// default configuration. The file doesn't need to exist at creation
    /// time, but will be validated when `fetch()` is called.
    /// 
    /// # Arguments
    /// 
    /// * `path` - Path to the document file. Can be relative or absolute.
    /// 
    /// # Errors
    /// 
    /// Returns [`SourceError`] if:
    /// - Path contains invalid characters
    /// - Path attempts directory traversal (contains `..`)
    /// - Path is empty or invalid
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use doc_extract::FileSource;
    /// 
    /// // Relative path
    /// let source1 = FileSource::new("document.pdf")?;
    /// 
    /// // Absolute path
    /// let source2 = FileSource::new("/home/user/documents/report.pdf")?;
    /// 
    /// // Path with subdirectories
    /// let source3 = FileSource::new("projects/2024/analysis.docx")?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    /// 
    /// # Security Note
    /// 
    /// This method validates paths to prevent directory traversal attacks:
    /// 
    /// ```rust,should_panic
    /// use doc_extract::FileSource;
    /// 
    /// // This will fail due to directory traversal
    /// let source = FileSource::new("../../../etc/passwd").unwrap();
    /// ```
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, SourceError> {
        // Implementation...
    }
    
    /// Returns a builder for advanced configuration
    /// 
    /// The builder pattern allows setting optional parameters such as
    /// size limits, streaming thresholds, and content type overrides.
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use doc_extract::FileSource;
    /// use std::time::Duration;
    /// 
    /// let source = FileSource::builder()
    ///     .path("large_file.pdf")
    ///     .max_size(50_000_000) // 50MB limit
    ///     .streaming_threshold(5_000_000) // Stream files >5MB
    ///     .read_timeout(Duration::from_secs(30))
    ///     .content_type("application/pdf")
    ///     .build()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn builder() -> FileSourceBuilder {
        // Implementation...
    }
    
    /// Gets the file path
    /// 
    /// Returns the path that was used to create this source.
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use doc_extract::FileSource;
    /// 
    /// let source = FileSource::new("document.pdf")?;
    /// println!("Source path: {}", source.path().display());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn path(&self) -> &Path {
        &self.path
    }
}
```

### 4. Function Documentation Standards

```rust
/// Processes a document using the neural pipeline
/// 
/// This function orchestrates the complete document processing workflow,
/// from source validation through neural analysis to result synthesis.
/// It automatically selects the optimal processing strategy based on
/// document characteristics and available resources.
/// 
/// # Arguments
/// 
/// * `source` - The document source implementing [`DocumentSource`]
/// * `options` - Processing configuration and feature flags
/// 
/// # Returns
/// 
/// Returns a [`ProcessingResult`] containing:
/// - Extracted text content
/// - Document classification and confidence scores
/// - Named entities with positions and types
/// - Structured data (tables, lists, etc.)
/// - Processing metadata and performance metrics
/// 
/// # Errors
/// 
/// This function can return several types of errors:
/// 
/// * [`ProcessingError::SourceError`] - Problems accessing the document source
/// * [`ProcessingError::ValidationError`] - Document validation failures
/// * [`ProcessingError::NeuralError`] - AI model processing failures
/// * [`ProcessingError::ResourceError`] - Insufficient memory or compute resources
/// * [`ProcessingError::TimeoutError`] - Processing exceeded time limits
/// 
/// # Examples
/// 
/// ## Basic Document Processing
/// 
/// ```rust
/// use doc_extract::{process_document, FileSource, ProcessingOptions};
/// 
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Create source and options
///     let source = FileSource::new("research_paper.pdf")?;
///     let options = ProcessingOptions::builder()
///         .extract_text(true)
///         .extract_entities(true)
///         .classify_content(true)
///         .build();
///     
///     // Process document
///     let result = process_document(&source, options).await?;
///     
///     // Access results
///     println!("Document type: {:?}", result.classification.document_type);
///     println!("Confidence: {:.2}%", result.classification.confidence * 100.0);
///     println!("Found {} entities:", result.entities.len());
///     
///     for entity in &result.entities {
///         println!("  {}: {} ({})", entity.text, entity.entity_type, entity.confidence);
///     }
///     
///     Ok(())
/// }
/// ```
/// 
/// ## Batch Processing
/// 
/// ```rust
/// use doc_extract::{process_document, FileSource, ProcessingOptions};
/// use futures::future::join_all;
/// 
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let files = vec!["doc1.pdf", "doc2.docx", "doc3.txt"];
///     let options = ProcessingOptions::default();
///     
///     // Process all documents concurrently
///     let futures: Vec<_> = files.into_iter()
///         .map(|file| async move {
///             let source = FileSource::new(file)?;
///             process_document(&source, options.clone()).await
///         })
///         .collect();
///     
///     let results = join_all(futures).await;
///     
///     for (i, result) in results.iter().enumerate() {
///         match result {
///             Ok(doc_result) => {
///                 println!("Document {}: Success ({} entities)", 
///                     i + 1, doc_result.entities.len());
///             }
///             Err(e) => {
///                 eprintln!("Document {}: Error - {}", i + 1, e);
///             }
///         }
///     }
///     
///     Ok(())
/// }
/// ```
/// 
/// ## Custom Processing Options
/// 
/// ```rust
/// use doc_extract::{process_document, UrlSource, ProcessingOptions, ModelConfig};
/// 
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let source = UrlSource::new("https://example.com/document.pdf")?;
///     
///     let options = ProcessingOptions::builder()
///         .extract_text(true)
///         .extract_entities(true)
///         .extract_tables(true)
///         .classify_content(true)
///         .language("en")
///         .custom_model(ModelConfig::new("my_classifier", "./models/custom"))
///         .timeout(std::time::Duration::from_secs(300))
///         .build();
///     
///     let result = process_document(&source, options).await?;
///     
///     // Process custom model results
///     if let Some(custom_result) = result.custom_results.get("my_classifier") {
///         println!("Custom classification: {:?}", custom_result);
///     }
///     
///     Ok(())
/// }
/// ```
/// 
/// # Performance Considerations
/// 
/// * **Small documents** (<1MB): Processed entirely in memory for speed
/// * **Medium documents** (1-100MB): Streaming processing with chunking
/// * **Large documents** (>100MB): Distributed processing across agents
/// * **GPU acceleration**: Automatically used when available for neural processing
/// * **Caching**: Results cached based on document content hash
/// 
/// # Memory Usage
/// 
/// Memory usage depends on document size and processing options:
/// 
/// | Document Size | Memory Usage | Processing Time |
/// |---------------|--------------|-----------------|
/// | <1MB          | ~10MB        | <1 second       |
/// | 1-10MB        | ~50MB        | 1-5 seconds     |
/// | 10-100MB      | ~200MB       | 5-30 seconds    |
/// | >100MB        | ~500MB       | 30+ seconds     |
/// 
/// # Thread Safety
/// 
/// This function is thread-safe and can be called concurrently from
/// multiple threads. Each call uses independent resources and doesn't
/// interfere with other processing operations.
/// 
/// # Cancellation
/// 
/// Processing can be cancelled using tokio's cancellation:
/// 
/// ```rust,no_run
/// use doc_extract::{process_document, FileSource, ProcessingOptions};
/// use tokio::time::{timeout, Duration};
/// 
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let source = FileSource::new("large_document.pdf")?;
///     let options = ProcessingOptions::default();
///     
///     // Process with timeout
///     match timeout(Duration::from_secs(60), process_document(&source, options)).await {
///         Ok(result) => {
///             println!("Processing completed: {:?}", result?);
///         }
///         Err(_) => {
///             println!("Processing timed out after 60 seconds");
///         }
///     }
///     
///     Ok(())
/// }
/// ```
pub async fn process_document(
    source: &dyn DocumentSource,
    options: ProcessingOptions,
) -> Result<ProcessingResult, ProcessingError> {
    // Implementation...
}
```

### 5. Error Type Documentation

```rust
/// Errors that can occur during document source operations
/// 
/// `SourceError` provides detailed error information for all document
/// source operations, enabling appropriate error handling and user
/// feedback.
/// 
/// # Error Categories
/// 
/// * **Access Errors**: Permission denied, file not found, network issues
/// * **Validation Errors**: Invalid configuration, malformed data
/// * **Resource Errors**: Out of memory, disk space, network capacity
/// * **Security Errors**: Malicious content, unauthorized access
/// 
/// # Examples
/// 
/// ## Error Handling
/// 
/// ```rust
/// use doc_extract::{FileSource, SourceError};
/// 
/// match FileSource::new("document.pdf") {
///     Ok(source) => {
///         match source.fetch().await {
///             Ok(content) => {
///                 println!("Success: {} bytes", content.data.len());
///             }
///             Err(SourceError::NotFound(path)) => {
///                 eprintln!("File not found: {}", path);
///                 // Maybe try alternative sources or create file
///             }
///             Err(SourceError::AccessDenied(msg)) => {
///                 eprintln!("Permission denied: {}", msg);
///                 // Maybe request elevated permissions
///             }
///             Err(SourceError::SourceTooLarge { size }) => {
///                 eprintln!("File too large: {} bytes", size);
///                 // Maybe use streaming or split file
///             }
///             Err(e) => {
///                 eprintln!("Unexpected error: {}", e);
///                 // Generic error handling
///             }
///         }
///     }
///     Err(e) => {
///         eprintln!("Failed to create source: {}", e);
///     }
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
/// 
/// ## Error Recovery
/// 
/// ```rust
/// use doc_extract::{SourceError, FileSource, UrlSource};
/// 
/// async fn fetch_with_fallback(local_path: &str, remote_url: &str) 
///     -> Result<DocumentContent, SourceError> {
///     
///     // Try local file first
///     match FileSource::new(local_path) {
///         Ok(source) => {
///             match source.fetch().await {
///                 Ok(content) => return Ok(content),
///                 Err(SourceError::NotFound(_)) => {
///                     // File not found, try remote
///                 }
///                 Err(e) => return Err(e), // Other errors are fatal
///             }
///         }
///         Err(_) => {
///             // Invalid path, try remote
///         }
///     }
///     
///     // Fallback to remote URL
///     let remote_source = UrlSource::new(remote_url)?;
///     remote_source.fetch().await
/// }
/// ```
#[derive(thiserror::Error, Debug)]
pub enum SourceError {
    /// Source not found or doesn't exist
    /// 
    /// This error occurs when a file doesn't exist, URL returns 404,
    /// or database record is not found.
    /// 
    /// # Examples
    /// 
    /// ```rust
    /// use doc_extract::{FileSource, SourceError};
    /// 
    /// let source = FileSource::new("nonexistent.pdf").unwrap();
    /// match source.fetch().await {
    ///     Err(SourceError::NotFound(path)) => {
    ///         println!("File not found: {}", path);
    ///     }
    ///     _ => {}
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[error("Source not found: {0}")]
    NotFound(String),
    
    /// Access denied due to permissions
    /// 
    /// This error occurs when the process doesn't have sufficient
    /// permissions to access the source.
    /// 
    /// # Examples
    /// 
    /// ```rust,no_run
    /// use doc_extract::{FileSource, SourceError};
    /// 
    /// let source = FileSource::new("/root/secret.pdf").unwrap();
    /// match source.fetch().await {
    ///     Err(SourceError::AccessDenied(msg)) => {
    ///         println!("Permission denied: {}", msg);
    ///     }
    ///     _ => {}
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[error("Access denied: {0}")]
    AccessDenied(String),
    
    /// Network-related error
    /// 
    /// This error occurs for URL sources when network connectivity
    /// issues prevent accessing the remote document.
    /// 
    /// # Examples
    /// 
    /// ```rust,no_run
    /// use doc_extract::{UrlSource, SourceError};
    /// 
    /// let source = UrlSource::new("https://unreachable.example.com/doc.pdf").unwrap();
    /// match source.fetch().await {
    ///     Err(SourceError::NetworkError(msg)) => {
    ///         println!("Network error: {}", msg);
    ///         // Maybe retry with exponential backoff
    ///     }
    ///     _ => {}
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[error("Network error: {0}")]
    NetworkError(String),
    
    /// HTTP error with status code
    /// 
    /// This error occurs for URL sources when the server returns
    /// an HTTP error status code.
    /// 
    /// # Examples
    /// 
    /// ```rust,no_run
    /// use doc_extract::{UrlSource, SourceError};
    /// 
    /// let source = UrlSource::new("https://example.com/forbidden.pdf").unwrap();
    /// match source.fetch().await {
    ///     Err(SourceError::HttpError { status, message }) => {
    ///         match status {
    ///             401 => println!("Authentication required"),
    ///             403 => println!("Forbidden: {}", message),
    ///             404 => println!("Document not found"),
    ///             500..=599 => println!("Server error: {}", message),
    ///             _ => println!("HTTP error {}: {}", status, message),
    ///         }
    ///     }
    ///     _ => {}
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[error("HTTP error {status}: {message}")]
    HttpError {
        /// HTTP status code
        status: u16,
        /// Error message from server
        message: String,
    },
    
    /// Source exceeds maximum allowed size
    /// 
    /// This error occurs when a document exceeds the configured
    /// size limits, preventing potential DoS attacks.
    /// 
    /// # Examples
    /// 
    /// ```rust,no_run
    /// use doc_extract::{FileSource, SourceError};
    /// 
    /// let source = FileSource::builder()
    ///     .path("huge_file.pdf")
    ///     .max_size(10_000_000) // 10MB limit
    ///     .build().unwrap();
    /// 
    /// match source.fetch().await {
    ///     Err(SourceError::SourceTooLarge { size }) => {
    ///         println!("File too large: {} bytes (limit: 10MB)", size);
    ///         // Maybe enable streaming mode
    ///     }
    ///     _ => {}
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    #[error("Source too large: {size} bytes exceeds limit")]
    SourceTooLarge {
        /// Actual size in bytes
        size: u64
    },
}
```

### 6. Module Documentation Standards

```rust
/// Neural processing pipeline for document analysis
/// 
/// This module provides the AI-powered core of the document extraction
/// platform, including text embedding, document classification, named
/// entity recognition, and table extraction capabilities.
/// 
/// # Architecture
/// 
/// The neural pipeline consists of several stages:
/// 
/// 1. **Preprocessing**: Text extraction and normalization
/// 2. **Feature Extraction**: Convert text to numerical representations
/// 3. **Model Inference**: Run AI models for classification and extraction
/// 4. **Post-processing**: Aggregate and format results
/// 
/// # Supported Models
/// 
/// * **Text Embeddings**: BERT, RoBERTa, DistilBERT, Sentence Transformers
/// * **Classification**: Custom document type classifiers
/// * **Named Entity Recognition**: spaCy, Transformers, custom models
/// * **Table Extraction**: LayoutLM, Table Transformer
/// 
/// # Quick Start
/// 
/// ```rust
/// use doc_extract::neural::{NeuralPipeline, ModelConfig};
/// 
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Initialize pipeline with default models
///     let pipeline = NeuralPipeline::default().await?;
///     
///     // Process document content
///     let text = "This is a research paper about machine learning.";
///     let results = pipeline.process_text(text).await?;
///     
///     println!("Document type: {:?}", results.classification);
///     println!("Entities: {:?}", results.entities);
///     
///     Ok(())
/// }
/// ```
/// 
/// # Advanced Usage
/// 
/// ## Custom Model Configuration
/// 
/// ```rust
/// use doc_extract::neural::{NeuralPipeline, ModelConfig, DeviceType};
/// 
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let pipeline = NeuralPipeline::builder()
///         .text_model(ModelConfig::new("bert-large-uncased"))
///         .classification_model(ModelConfig::new("custom-classifier")
///             .path("./models/my_classifier")
///             .device(DeviceType::Cuda))
///         .ner_model(ModelConfig::new("ner-large"))
///         .batch_size(16)
///         .build()
///         .await?;
///     
///     // Use pipeline...
///     Ok(())
/// }
/// ```
/// 
/// ## Batch Processing
/// 
/// ```rust
/// use doc_extract::neural::NeuralPipeline;
/// 
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let pipeline = NeuralPipeline::default().await?;
///     
///     let documents = vec![
///         "First document text...",
///         "Second document text...",
///         "Third document text...",
///     ];
///     
///     let results = pipeline.process_batch(documents).await?;
///     
///     for (i, result) in results.iter().enumerate() {
///         println!("Document {}: {:?}", i + 1, result.classification);
///     }
///     
///     Ok(())
/// }
/// ```
/// 
/// # Performance Optimization
/// 
/// * **GPU Acceleration**: Automatically uses CUDA when available
/// * **Model Quantization**: Supports INT8 quantization for faster inference
/// * **Batch Processing**: Processes multiple documents efficiently
/// * **Caching**: Caches model outputs for repeated content
/// 
/// # Memory Management
/// 
/// The neural pipeline implements several memory optimization strategies:
/// 
/// * **Lazy Loading**: Models loaded only when needed
/// * **Memory Pooling**: Reuses tensor memory across inferences
/// * **Gradient Clearing**: Automatically clears gradients after inference
/// * **Model Offloading**: Moves unused models to CPU memory
/// 
/// # Error Handling
/// 
/// Neural processing can fail due to various reasons:
/// 
/// * **Model Loading Errors**: Model files corrupted or missing
/// * **Out of Memory**: Insufficient GPU/CPU memory for models
/// * **Invalid Input**: Text too long or contains invalid characters
/// * **Device Errors**: GPU driver issues or CUDA problems
/// 
/// All errors are wrapped in [`NeuralError`] with detailed context.
/// 
/// # Thread Safety
/// 
/// All neural processing components are thread-safe and can be used
/// concurrently from multiple threads. Models are protected with
/// appropriate synchronization primitives.
pub mod neural {
    // Module contents...
}
```

## Documentation Testing

### 1. Doctests

All code examples in documentation must be valid and tested:

```rust
/// Processes multiple documents concurrently
/// 
/// # Examples
/// 
/// ```rust
/// use doc_extract::{process_documents, FileSource, ProcessingOptions};
/// 
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     let sources = vec![
///         FileSource::new("doc1.pdf")?,
///         FileSource::new("doc2.pdf")?,
///     ];
///     
///     let options = ProcessingOptions::default();
///     let results = process_documents(sources, options).await?;
///     
///     assert_eq!(results.len(), 2);
///     for result in results {
///         assert!(result.is_ok());
///     }
///     
///     Ok(())
/// }
/// ```
pub async fn process_documents(
    sources: Vec<Box<dyn DocumentSource>>,
    options: ProcessingOptions,
) -> Result<Vec<ProcessingResult>, ProcessingError> {
    // Implementation...
}
```

### 2. Integration Examples

Include complete working examples:

```rust
/// Complete example of document processing pipeline
/// 
/// # Examples
/// 
/// ```rust,no_run
/// use doc_extract::{
///     DocumentProcessor, FileSource, ProcessingOptions,
///     neural::{NeuralPipeline, ModelConfig},
///     daa::{DAASystem, AgentConfig},
/// };
/// 
/// #[tokio::main]
/// async fn main() -> Result<(), Box<dyn std::error::Error>> {
///     // Initialize neural pipeline
///     let neural_pipeline = NeuralPipeline::builder()
///         .text_model(ModelConfig::new("bert-base-uncased"))
///         .classification_model(ModelConfig::new("document-classifier"))
///         .build()
///         .await?;
///     
///     // Initialize DAA system
///     let daa_system = DAASystem::builder()
///         .max_agents(8)
///         .enable_auto_scaling(true)
///         .build()
///         .await?;
///     
///     // Create document processor
///     let processor = DocumentProcessor::builder()
///         .neural_pipeline(neural_pipeline)
///         .daa_system(daa_system)
///         .build();
///     
///     // Process document
///     let source = FileSource::new("example.pdf")?;
///     let options = ProcessingOptions::builder()
///         .extract_text(true)
///         .extract_entities(true)
///         .classify_content(true)
///         .build();
///     
///     let result = processor.process(&source, options).await?;
///     
///     // Print results
///     println!("Document type: {:?}", result.classification.document_type);
///     println!("Confidence: {:.2}%", result.classification.confidence * 100.0);
///     println!("Entities found: {}", result.entities.len());
///     
///     for entity in &result.entities {
///         println!("  {} ({}): {:.2}%", 
///             entity.text, 
///             entity.entity_type,
///             entity.confidence * 100.0
///         );
///     }
///     
///     Ok(())
/// }
/// ```
```

## Documentation Generation

### 1. Cargo.toml Configuration

```toml
[package]
name = "doc-extract"
version = "1.0.0"
documentation = "https://docs.rs/doc-extract"

[package.metadata.docs.rs]
all-features = true
rustdoc-args = ["--cfg", "docsrs"]

[[example]]
name = "basic_processing"
doc-scrape-examples = true

[[example]]
name = "custom_source"
doc-scrape-examples = true
```

### 2. Build Documentation

```bash
# Generate documentation with all features
cargo doc --all-features --no-deps --open

# Generate documentation with examples
cargo doc --all-features --examples

# Test documentation examples
cargo test --doc

# Check documentation completeness
cargo doc --all-features 2>&1 | grep -i "warning"
```

## Quality Checklist

### ✅ Documentation Requirements

- [ ] All public items have comprehensive documentation
- [ ] All examples compile and run successfully
- [ ] Error types are fully documented with examples
- [ ] Performance characteristics are documented
- [ ] Thread safety is clearly stated
- [ ] Security considerations are mentioned
- [ ] Platform compatibility is documented
- [ ] Memory usage patterns are explained
- [ ] Integration examples are provided
- [ ] Troubleshooting guides are included

This comprehensive rustdoc guide ensures that all code in the project is thoroughly documented with clear examples, proper error handling, and complete API coverage.