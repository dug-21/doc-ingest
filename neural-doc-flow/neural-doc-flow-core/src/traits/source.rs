//! Document source traits for plugin architecture
//!
//! This module defines the core traits for implementing document source plugins
//! that can extract content from various document formats.

use async_trait::async_trait;
use crate::{
    SourceError, SourceInput, ValidationResult, ExtractedDocument, 
    SourceConfig, NeuralDocFlowResult
};

/// Core trait for document source plugins
/// 
/// This trait defines the interface that all document source plugins must implement.
/// It provides a consistent API for validating, extracting, and configuring
/// document processing for different formats.
///
/// # Example Implementation
///
/// ```rust,no_run
/// use async_trait::async_trait;
/// use neural_doc_flow_core::{DocumentSource, SourceInput, SourceError, ExtractedDocument, ValidationResult, SourceConfig};
///
/// struct MySource;
///
/// #[async_trait]
/// impl DocumentSource for MySource {
///     fn source_id(&self) -> &str { "my_source" }
///     fn name(&self) -> &str { "My Document Source" }
///     fn version(&self) -> &str { "1.0.0" }
///     fn supported_extensions(&self) -> &[&str] { &["mydoc"] }
///     fn supported_mime_types(&self) -> &[&str] { &["application/x-mydoc"] }
///     
///     async fn can_handle(&self, input: &SourceInput) -> Result<bool, SourceError> {
///         // Implementation logic
///         Ok(true)
///     }
///     
///     async fn validate(&self, input: &SourceInput) -> Result<ValidationResult, SourceError> {
///         Ok(ValidationResult::valid())
///     }
///     
///     async fn extract(&self, input: SourceInput) -> Result<ExtractedDocument, SourceError> {
///         // Extraction logic
///         todo!()
///     }
///     
///     fn config_schema(&self) -> serde_json::Value {
///         serde_json::json!({})
///     }
///     
///     async fn initialize(&mut self, config: SourceConfig) -> Result<(), SourceError> {
///         Ok(())
///     }
///     
///     async fn cleanup(&mut self) -> Result<(), SourceError> {
///         Ok(())
///     }
/// }
/// ```
#[async_trait]
pub trait DocumentSource: Send + Sync {
    /// Unique identifier for this source plugin
    /// 
    /// This should be a stable identifier that doesn't change between versions.
    /// Used for plugin discovery and configuration.
    fn source_id(&self) -> &str;

    /// Human-readable name of this source
    fn name(&self) -> &str;

    /// Source plugin version
    fn version(&self) -> &str;

    /// File extensions this source can handle (without the dot)
    /// 
    /// # Example
    /// ```rust,no_run
    /// # use neural_doc_flow_core::DocumentSource;
    /// # struct PdfSource;
    /// # impl DocumentSource for PdfSource {
    /// #     fn source_id(&self) -> &str { "pdf" }
    /// #     fn name(&self) -> &str { "PDF Source" }
    /// #     fn version(&self) -> &str { "1.0.0" }
    /// fn supported_extensions(&self) -> &[&str] {
    ///     &["pdf", "PDF"]
    /// }
    /// #     fn supported_mime_types(&self) -> &[&str] { &[] }
    /// #     async fn can_handle(&self, input: &neural_doc_flow_core::SourceInput) -> Result<bool, neural_doc_flow_core::SourceError> { Ok(false) }
    /// #     async fn validate(&self, input: &neural_doc_flow_core::SourceInput) -> Result<neural_doc_flow_core::ValidationResult, neural_doc_flow_core::SourceError> { Ok(neural_doc_flow_core::ValidationResult::valid()) }
    /// #     async fn extract(&self, input: neural_doc_flow_core::SourceInput) -> Result<neural_doc_flow_core::ExtractedDocument, neural_doc_flow_core::SourceError> { todo!() }
    /// #     fn config_schema(&self) -> serde_json::Value { serde_json::json!({}) }
    /// #     async fn initialize(&mut self, config: neural_doc_flow_core::SourceConfig) -> Result<(), neural_doc_flow_core::SourceError> { Ok(()) }
    /// #     async fn cleanup(&mut self) -> Result<(), neural_doc_flow_core::SourceError> { Ok(()) }
    /// # }
    /// ```
    fn supported_extensions(&self) -> &[&str];

    /// MIME types this source can handle
    fn supported_mime_types(&self) -> &[&str];

    /// Check if this source can handle the given input
    /// 
    /// This is a fast check that should examine file extensions, MIME types,
    /// or magic bytes to determine compatibility.
    /// 
    /// # Parameters
    /// - `input`: The input to check
    /// 
    /// # Returns
    /// - `Ok(true)` if this source can handle the input
    /// - `Ok(false)` if this source cannot handle the input
    /// - `Err(_)` if an error occurred during the check
    async fn can_handle(&self, input: &SourceInput) -> Result<bool, SourceError>;

    /// Validate input before processing
    /// 
    /// This performs more thorough validation than `can_handle`, including
    /// format validation, security checks, and size limits.
    /// 
    /// # Parameters
    /// - `input`: The input to validate
    /// 
    /// # Returns
    /// - `Ok(ValidationResult)` with validation results
    /// - `Err(_)` if validation could not be performed
    async fn validate(&self, input: &SourceInput) -> Result<ValidationResult, SourceError>;

    /// Extract content from the input document
    /// 
    /// This is the main processing method that extracts structured content
    /// from the input document.
    /// 
    /// # Parameters
    /// - `input`: The input document to process
    /// 
    /// # Returns
    /// - `Ok(ExtractedDocument)` with extracted content
    /// - `Err(_)` if extraction failed
    async fn extract(&self, input: SourceInput) -> Result<ExtractedDocument, SourceError>;

    /// Get the JSON schema for this source's configuration
    /// 
    /// This should return a valid JSON Schema that describes the configuration
    /// options available for this source.
    fn config_schema(&self) -> serde_json::Value;

    /// Initialize the source with configuration
    /// 
    /// Called once during plugin loading with the source's configuration.
    /// Use this to set up any required resources, validate configuration,
    /// and prepare for processing.
    /// 
    /// # Parameters
    /// - `config`: Source configuration
    /// 
    /// # Returns
    /// - `Ok(())` if initialization succeeded
    /// - `Err(_)` if initialization failed
    async fn initialize(&mut self, config: SourceConfig) -> Result<(), SourceError>;

    /// Clean up resources
    /// 
    /// Called when the source is being unloaded. Use this to clean up
    /// any resources, close files, stop background tasks, etc.
    async fn cleanup(&mut self) -> Result<(), SourceError>;

    /// Get source capabilities
    /// 
    /// Returns information about what this source can do.
    /// Default implementation returns basic capabilities.
    fn capabilities(&self) -> SourceCapabilities {
        SourceCapabilities::default()
    }

    /// Get source statistics
    /// 
    /// Returns performance and usage statistics for this source.
    fn statistics(&self) -> SourceStatistics {
        SourceStatistics::default()
    }

    /// Check if source supports streaming
    /// 
    /// Returns true if this source can process streaming input.
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Check if source supports parallel processing
    /// 
    /// Returns true if this source can safely process multiple documents
    /// in parallel.
    fn supports_parallel(&self) -> bool {
        true
    }
}

/// Source capabilities information
#[derive(Debug, Clone, Default)]
pub struct SourceCapabilities {
    /// Can extract text content
    pub text_extraction: bool,
    /// Can extract images
    pub image_extraction: bool,
    /// Can extract tables
    pub table_extraction: bool,
    /// Can extract metadata
    pub metadata_extraction: bool,
    /// Can preserve document structure
    pub structure_preservation: bool,
    /// Can handle encrypted documents
    pub encrypted_documents: bool,
    /// Maximum file size (bytes)
    pub max_file_size: Option<usize>,
    /// Supported quality levels
    pub quality_levels: Vec<String>,
}

/// Source performance statistics
#[derive(Debug, Clone, Default)]
pub struct SourceStatistics {
    /// Total documents processed
    pub documents_processed: u64,
    /// Total processing time
    pub total_processing_time: std::time::Duration,
    /// Average processing time per document
    pub average_processing_time: std::time::Duration,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f32,
    /// Peak memory usage (bytes)
    pub peak_memory_usage: usize,
    /// Cache hit rate (0.0 to 1.0)
    pub cache_hit_rate: f32,
}

/// Trait for reading document content
/// 
/// This trait provides low-level reading capabilities for document sources.
/// Implement this for custom input methods or optimized reading strategies.
#[async_trait]
pub trait SourceReader: Send + Sync {
    /// Read all content from the source
    async fn read_all(&mut self) -> Result<Vec<u8>, SourceError>;

    /// Read content in chunks
    async fn read_chunk(&mut self, size: usize) -> Result<Option<Vec<u8>>, SourceError>;

    /// Get the total size if known
    fn size_hint(&self) -> Option<usize>;

    /// Check if the reader is at end of file
    fn is_eof(&self) -> bool;

    /// Seek to a specific position (if supported)
    async fn seek(&mut self, position: u64) -> Result<u64, SourceError> {
        Err(SourceError::Custom("Seek not supported".to_string()))
    }
}

/// Trait for extracting content from documents
/// 
/// This trait provides specialized content extraction capabilities.
/// Sources can implement this for optimized extraction strategies.
#[async_trait]
pub trait ContentExtractor: Send + Sync {
    /// Extract text content only
    async fn extract_text(&self, input: &SourceInput) -> Result<String, SourceError>;

    /// Extract metadata only
    async fn extract_metadata(&self, input: &SourceInput) -> Result<crate::DocumentMetadata, SourceError>;

    /// Extract specific content types
    async fn extract_content_type(
        &self,
        input: &SourceInput,
        content_type: ContentType,
    ) -> Result<Vec<crate::ContentBlock>, SourceError>;

    /// Check if extractor supports the content type
    fn supports_content_type(&self, content_type: ContentType) -> bool;
}

/// Content types for specialized extraction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentType {
    /// Text paragraphs
    Text,
    /// Tables
    Tables,
    /// Images
    Images,
    /// Headers/headings
    Headers,
    /// Lists
    Lists,
    /// Code blocks
    Code,
    /// Formulas/equations
    Formulas,
    /// Footnotes
    Footnotes,
}

/// Trait for configuring document sources
/// 
/// This trait provides advanced configuration capabilities for sources
/// that need dynamic configuration updates.
pub trait SourceConfiguration: Send + Sync {
    /// Update configuration at runtime
    fn update_config(&mut self, config: SourceConfig) -> Result<(), SourceError>;

    /// Get current configuration
    fn get_config(&self) -> SourceConfig;

    /// Validate a configuration without applying it
    fn validate_config(&self, config: &SourceConfig) -> Result<(), SourceError>;

    /// Reset to default configuration
    fn reset_config(&mut self) -> Result<(), SourceError>;
}

/// Helper trait for creating document sources
pub trait SourceFactory: Send + Sync {
    /// Create a new instance of the source
    fn create(&self) -> NeuralDocFlowResult<Box<dyn DocumentSource>>;

    /// Get source metadata
    fn metadata(&self) -> SourcePluginMetadata;
}

/// Metadata for source plugins
#[derive(Debug, Clone)]
pub struct SourcePluginMetadata {
    /// Plugin identifier
    pub id: String,
    /// Plugin name
    pub name: String,
    /// Plugin version
    pub version: String,
    /// Plugin author
    pub author: String,
    /// Plugin description
    pub description: String,
    /// Supported API version
    pub api_version: String,
    /// Plugin capabilities
    pub capabilities: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    struct TestSource;

    #[async_trait]
    impl DocumentSource for TestSource {
        fn source_id(&self) -> &str { "test" }
        fn name(&self) -> &str { "Test Source" }
        fn version(&self) -> &str { "1.0.0" }
        fn supported_extensions(&self) -> &[&str] { &["test"] }
        fn supported_mime_types(&self) -> &[&str] { &["application/test"] }

        async fn can_handle(&self, _input: &SourceInput) -> Result<bool, SourceError> {
            Ok(true)
        }

        async fn validate(&self, _input: &SourceInput) -> Result<ValidationResult, SourceError> {
            Ok(ValidationResult::valid())
        }

        async fn extract(&self, _input: SourceInput) -> Result<ExtractedDocument, SourceError> {
            Ok(ExtractedDocument {
                id: "test".to_string(),
                source_id: self.source_id().to_string(),
                metadata: DocumentMetadata {
                    title: Some("Test Document".to_string()),
                    author: None,
                    created_date: None,
                    modified_date: None,
                    page_count: 1,
                    language: Some("en".to_string()),
                    keywords: vec![],
                    custom_metadata: std::collections::HashMap::new(),
                },
                content: vec![],
                structure: DocumentStructure {
                    sections: vec![],
                    hierarchy: vec![],
                    table_of_contents: vec![],
                },
                confidence: 1.0,
                metrics: ExtractionMetrics {
                    extraction_time: std::time::Duration::from_millis(10),
                    pages_processed: 1,
                    blocks_extracted: 0,
                    memory_used: 1024,
                },
            })
        }

        fn config_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {}
            })
        }

        async fn initialize(&mut self, _config: SourceConfig) -> Result<(), SourceError> {
            Ok(())
        }

        async fn cleanup(&mut self) -> Result<(), SourceError> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_document_source_trait() {
        let source = TestSource;
        assert_eq!(source.source_id(), "test");
        assert_eq!(source.name(), "Test Source");
        assert!(source.supported_extensions().contains(&"test"));

        let input = SourceInput::Memory {
            data: vec![],
            filename: Some("test.test".to_string()),
            mime_type: None,
        };

        assert!(source.can_handle(&input).await.unwrap());
        let validation = source.validate(&input).await.unwrap();
        assert!(validation.is_valid);
    }

    #[test]
    fn test_source_capabilities() {
        let caps = SourceCapabilities {
            text_extraction: true,
            image_extraction: false,
            ..Default::default()
        };
        
        assert!(caps.text_extraction);
        assert!(!caps.image_extraction);
    }

    #[test]
    fn test_content_type_equality() {
        assert_eq!(ContentType::Text, ContentType::Text);
        assert_ne!(ContentType::Text, ContentType::Images);
    }
}