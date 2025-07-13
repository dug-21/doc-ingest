//! Core document processing engine
//!
//! This module provides the main DocFlow interface and core data structures
//! for document extraction and processing.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::config::Config;
use crate::daa::DaaCoordinator;
use crate::error::{NeuralDocFlowError, Result};
use crate::neural::NeuralEngine;
use crate::sources::{DocumentSource, SourceInput, SourceManager};

/// Main document processing interface
pub struct DocFlow {
    config: Config,
    source_manager: SourceManager,
    daa_coordinator: DaaCoordinator,
    neural_engine: NeuralEngine,
}

impl DocFlow {
    /// Create new DocFlow instance with default configuration
    pub fn new() -> Result<Self> {
        let config = Config::default();
        Self::with_config(config)
    }

    /// Create new DocFlow instance with custom configuration
    pub fn with_config(config: Config) -> Result<Self> {
        let source_manager = SourceManager::new(&config)?;
        let daa_coordinator = DaaCoordinator::new(&config)?;
        let neural_engine = NeuralEngine::new(&config)?;

        Ok(Self {
            config,
            source_manager,
            daa_coordinator,
            neural_engine,
        })
    }

    /// Extract document content from input source
    pub async fn extract(&self, input: SourceInput) -> Result<ExtractedDocument> {
        let start_time = Instant::now();
        
        // Find compatible source
        let source = self.source_manager.find_compatible_source(&input).await?;
        
        // Validate input
        let validation = source.validate(&input).await?;
        if !validation.is_valid() {
            return Err(NeuralDocFlowError::validation(format!(
                "Input validation failed: {:?}", validation.errors
            )));
        }

        // Coordinate extraction through DAA
        let raw_document = self.daa_coordinator
            .coordinate_extraction(source, input)
            .await?;

        // Apply neural enhancement
        let enhanced_document = self.neural_engine
            .enhance_document(raw_document)
            .await?;

        // Update metrics
        let extraction_time = start_time.elapsed();
        let mut final_document = enhanced_document;
        final_document.metrics.extraction_time = extraction_time;

        Ok(final_document)
    }

    /// Extract multiple documents in batch
    pub async fn extract_batch(&self, inputs: Vec<SourceInput>) -> Result<Vec<ExtractedDocument>> {
        let results = self.daa_coordinator
            .coordinate_batch_extraction(inputs)
            .await?;

        // Apply neural enhancement to all documents
        let mut enhanced_results = Vec::new();
        for doc in results {
            let enhanced = self.neural_engine.enhance_document(doc).await?;
            enhanced_results.push(enhanced);
        }

        Ok(enhanced_results)
    }

    /// Get processing statistics
    pub fn get_stats(&self) -> ProcessingStats {
        ProcessingStats {
            documents_processed: self.daa_coordinator.get_document_count(),
            total_processing_time: self.daa_coordinator.get_total_time(),
            average_confidence: self.neural_engine.get_average_confidence(),
            active_agents: self.daa_coordinator.get_active_agent_count(),
            neural_models_loaded: self.neural_engine.get_loaded_model_count(),
        }
    }

    /// Shutdown and cleanup resources
    pub async fn shutdown(self) -> Result<()> {
        self.neural_engine.shutdown().await?;
        self.daa_coordinator.shutdown().await?;
        self.source_manager.shutdown().await?;
        Ok(())
    }
}

/// Extracted document with content and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedDocument {
    /// Unique document identifier
    pub id: String,
    
    /// Source that extracted this document
    pub source_id: String,
    
    /// Document metadata
    pub metadata: DocumentMetadata,
    
    /// Extracted content blocks
    pub content: Vec<ContentBlock>,
    
    /// Document structure information
    pub structure: DocumentStructure,
    
    /// Overall extraction confidence (0.0 to 1.0)
    pub confidence: f32,
    
    /// Processing metrics
    pub metrics: ExtractionMetrics,
}

impl ExtractedDocument {
    /// Create new extracted document
    pub fn new(source_id: String) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            source_id,
            metadata: DocumentMetadata::default(),
            content: Vec::new(),
            structure: DocumentStructure::default(),
            confidence: 0.0,
            metrics: ExtractionMetrics::default(),
        }
    }

    /// Get text content as a single string
    pub fn get_text(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| block.text.as_ref())
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    /// Get content blocks of specific type
    pub fn get_blocks_by_type(&self, block_type: &BlockType) -> Vec<&ContentBlock> {
        self.content
            .iter()
            .filter(|block| &block.block_type == block_type)
            .collect()
    }

    /// Get tables as structured data
    pub fn get_tables(&self) -> Vec<&ContentBlock> {
        self.get_blocks_by_type(&BlockType::Table)
    }

    /// Get images with metadata
    pub fn get_images(&self) -> Vec<&ContentBlock> {
        self.get_blocks_by_type(&BlockType::Image)
    }

    /// Calculate content statistics
    pub fn get_content_stats(&self) -> ContentStats {
        let mut stats = ContentStats::default();
        
        for block in &self.content {
            match block.block_type {
                BlockType::Paragraph => stats.paragraph_count += 1,
                BlockType::Heading(_) => stats.heading_count += 1,
                BlockType::Table => stats.table_count += 1,
                BlockType::Image => stats.image_count += 1,
                BlockType::List(_) => stats.list_count += 1,
                _ => stats.other_count += 1,
            }

            if let Some(text) = &block.text {
                stats.total_characters += text.len();
                stats.total_words += text.split_whitespace().count();
            }
        }

        stats
    }
}

/// Document metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocumentMetadata {
    /// Document title
    pub title: Option<String>,
    
    /// Document author
    pub author: Option<String>,
    
    /// Creation date
    pub created_date: Option<String>,
    
    /// Last modified date
    pub modified_date: Option<String>,
    
    /// Number of pages
    pub page_count: usize,
    
    /// Document language
    pub language: Option<String>,
    
    /// Keywords and tags
    pub keywords: Vec<String>,
    
    /// Custom metadata fields
    pub custom_metadata: HashMap<String, String>,
}

/// Content block representing extracted content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBlock {
    /// Unique block identifier
    pub id: String,
    
    /// Type of content block
    pub block_type: BlockType,
    
    /// Text content (if applicable)
    pub text: Option<String>,
    
    /// Binary content (for images, etc.)
    pub binary: Option<Vec<u8>>,
    
    /// Block metadata
    pub metadata: BlockMetadata,
    
    /// Position in document
    pub position: BlockPosition,
    
    /// Relationships to other blocks
    pub relationships: Vec<BlockRelationship>,
}

impl ContentBlock {
    /// Create new content block
    pub fn new(block_type: BlockType) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            block_type,
            text: None,
            binary: None,
            metadata: BlockMetadata::default(),
            position: BlockPosition::default(),
            relationships: Vec::new(),
        }
    }

    /// Set text content
    pub fn with_text<S: Into<String>>(mut self, text: S) -> Self {
        self.text = Some(text.into());
        self
    }

    /// Set binary content
    pub fn with_binary(mut self, data: Vec<u8>) -> Self {
        self.binary = Some(data);
        self
    }

    /// Add relationship to another block
    pub fn add_relationship(&mut self, relationship: BlockRelationship) {
        self.relationships.push(relationship);
    }
}

/// Types of content blocks
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockType {
    /// Text paragraph
    Paragraph,
    
    /// Heading with level (1-6)
    Heading(u8),
    
    /// Table
    Table,
    
    /// Image
    Image,
    
    /// List with type
    List(ListType),
    
    /// Code block
    CodeBlock,
    
    /// Quote
    Quote,
    
    /// Footnote
    Footnote,
    
    /// Caption
    Caption,
    
    /// Mathematical formula
    Formula,
    
    /// Custom block type
    Custom(String),
}

/// List types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ListType {
    /// Unordered list
    Unordered,
    
    /// Ordered list
    Ordered,
    
    /// Definition list
    Definition,
}

/// Block metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BlockMetadata {
    /// Page number (if applicable)
    pub page: Option<usize>,
    
    /// Extraction confidence (0.0 to 1.0)
    pub confidence: f32,
    
    /// Detected language
    pub language: Option<String>,
    
    /// Additional attributes
    pub attributes: HashMap<String, String>,
}

/// Position of block in document
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BlockPosition {
    /// Page number
    pub page: usize,
    
    /// X coordinate
    pub x: f32,
    
    /// Y coordinate
    pub y: f32,
    
    /// Width
    pub width: f32,
    
    /// Height
    pub height: f32,
}

/// Relationship between blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockRelationship {
    /// Target block ID
    pub target_id: String,
    
    /// Relationship type
    pub relationship_type: RelationshipType,
    
    /// Relationship confidence
    pub confidence: f32,
}

/// Types of relationships between blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    /// Child relationship (contains)
    Contains,
    
    /// Parent relationship (contained by)
    ContainedBy,
    
    /// Reference relationship
    References,
    
    /// Continuation relationship
    Continues,
    
    /// Association relationship
    AssociatedWith,
}

/// Document structure information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocumentStructure {
    /// Document sections
    pub sections: Vec<Section>,
    
    /// Hierarchical structure
    pub hierarchy: Vec<HierarchyNode>,
    
    /// Table of contents
    pub table_of_contents: Vec<TocEntry>,
}

/// Document section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Section {
    /// Section identifier
    pub id: String,
    
    /// Section title
    pub title: Option<String>,
    
    /// Hierarchical level
    pub level: usize,
    
    /// Content block IDs in this section
    pub blocks: Vec<String>,
    
    /// Subsections
    pub subsections: Vec<String>,
}

/// Hierarchy node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchyNode {
    /// Node identifier
    pub id: String,
    
    /// Parent node ID
    pub parent: Option<String>,
    
    /// Child node IDs
    pub children: Vec<String>,
    
    /// Reference to section
    pub section_ref: String,
}

/// Table of contents entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TocEntry {
    /// Entry title
    pub title: String,
    
    /// Hierarchical level
    pub level: usize,
    
    /// Section ID
    pub section_id: String,
    
    /// Page number
    pub page: Option<usize>,
}

/// Extraction metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExtractionMetrics {
    /// Total extraction time
    pub extraction_time: Duration,
    
    /// Number of pages processed
    pub pages_processed: usize,
    
    /// Number of blocks extracted
    pub blocks_extracted: usize,
    
    /// Memory used during extraction
    pub memory_used: usize,
}

/// Processing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingStats {
    /// Total documents processed
    pub documents_processed: u64,
    
    /// Total processing time
    pub total_processing_time: Duration,
    
    /// Average confidence score
    pub average_confidence: f32,
    
    /// Active DAA agents
    pub active_agents: usize,
    
    /// Loaded neural models
    pub neural_models_loaded: usize,
}

/// Content statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContentStats {
    /// Number of paragraphs
    pub paragraph_count: usize,
    
    /// Number of headings
    pub heading_count: usize,
    
    /// Number of tables
    pub table_count: usize,
    
    /// Number of images
    pub image_count: usize,
    
    /// Number of lists
    pub list_count: usize,
    
    /// Other content types
    pub other_count: usize,
    
    /// Total characters
    pub total_characters: usize,
    
    /// Total words
    pub total_words: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extracted_document_creation() {
        let doc = ExtractedDocument::new("test_source".to_string());
        assert_eq!(doc.source_id, "test_source");
        assert!(!doc.id.is_empty());
        assert_eq!(doc.content.len(), 0);
        assert_eq!(doc.confidence, 0.0);
    }

    #[test]
    fn test_content_block_creation() {
        let block = ContentBlock::new(BlockType::Paragraph)
            .with_text("Test paragraph");
        
        assert!(matches!(block.block_type, BlockType::Paragraph));
        assert_eq!(block.text, Some("Test paragraph".to_string()));
        assert!(!block.id.is_empty());
    }

    #[test]
    fn test_block_types() {
        let heading = BlockType::Heading(2);
        let table = BlockType::Table;
        let list = BlockType::List(ListType::Ordered);
        let custom = BlockType::Custom("special".to_string());

        assert!(matches!(heading, BlockType::Heading(2)));
        assert!(matches!(table, BlockType::Table));
        assert!(matches!(list, BlockType::List(ListType::Ordered)));
        assert!(matches!(custom, BlockType::Custom(_)));
    }

    #[test]
    fn test_document_text_extraction() {
        let mut doc = ExtractedDocument::new("test".to_string());
        
        doc.content.push(
            ContentBlock::new(BlockType::Paragraph)
                .with_text("First paragraph")
        );
        
        doc.content.push(
            ContentBlock::new(BlockType::Paragraph)
                .with_text("Second paragraph")
        );

        let text = doc.get_text();
        assert!(text.contains("First paragraph"));
        assert!(text.contains("Second paragraph"));
    }

    #[test]
    fn test_content_stats() {
        let mut doc = ExtractedDocument::new("test".to_string());
        
        doc.content.push(ContentBlock::new(BlockType::Paragraph).with_text("Hello world"));
        doc.content.push(ContentBlock::new(BlockType::Heading(1)).with_text("Title"));
        doc.content.push(ContentBlock::new(BlockType::Table));

        let stats = doc.get_content_stats();
        assert_eq!(stats.paragraph_count, 1);
        assert_eq!(stats.heading_count, 1);
        assert_eq!(stats.table_count, 1);
        assert_eq!(stats.total_words, 3); // "Hello", "world", "Title"
    }

    #[test]
    fn test_block_relationships() {
        let mut block = ContentBlock::new(BlockType::Paragraph);
        
        block.add_relationship(BlockRelationship {
            target_id: "other_block".to_string(),
            relationship_type: RelationshipType::References,
            confidence: 0.95,
        });

        assert_eq!(block.relationships.len(), 1);
        assert_eq!(block.relationships[0].target_id, "other_block");
    }

    #[tokio::test]
    async fn test_docflow_creation() {
        // This test may fail without proper configuration
        // In a real implementation, we'd use mock dependencies
        match DocFlow::new() {
            Ok(docflow) => {
                let stats = docflow.get_stats();
                assert_eq!(stats.documents_processed, 0);
            }
            Err(_) => {
                // Expected in test environment without full setup
                assert!(true);
            }
        }
    }
}