//! Core types for neural document processing

use std::collections::HashMap;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A document to be processed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: Uuid,
    pub source: DocumentSource,
    pub metadata: DocumentMetadata,
    pub content: Option<Vec<u8>>,
}

/// Source of a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentSource {
    File { path: PathBuf },
    Url { url: String },
    Bytes { data: Vec<u8>, mime_type: String },
    Stream { source_id: String },
}

/// Metadata associated with a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    pub title: Option<String>,
    pub author: Option<String>,
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    pub modified_at: Option<chrono::DateTime<chrono::Utc>>,
    pub mime_type: Option<String>,
    pub size_bytes: Option<u64>,
    pub language: Option<String>,
    pub custom_fields: HashMap<String, String>,
}

/// Extracted content from a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedContent {
    pub blocks: Vec<ContentBlock>,
    pub metadata: DocumentMetadata,
    pub confidence: Confidence,
    pub extracted_at: chrono::DateTime<chrono::Utc>,
}

/// A block of content within a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBlock {
    pub id: Uuid,
    pub block_type: BlockType,
    pub content: BlockContent,
    pub position: Position,
    pub confidence: f32,
    pub metadata: HashMap<String, String>,
}

/// Type of content block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlockType {
    Text,
    Table,
    Image,
    Header,
    Footer,
    List,
    Code,
    Formula,
    Chart,
    Unknown,
}

/// Content within a block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlockContent {
    Text { text: String },
    Table { rows: Vec<Vec<String>>, headers: Option<Vec<String>> },
    Image { data: Vec<u8>, format: String, alt_text: Option<String> },
    List { items: Vec<String>, ordered: bool },
    Code { code: String, language: Option<String> },
    Formula { latex: String, text: Option<String> },
    Chart { data: serde_json::Value, chart_type: String },
}

/// Position of content within a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub page: Option<u32>,
    pub x: Option<f32>,
    pub y: Option<f32>,
    pub width: Option<f32>,
    pub height: Option<f32>,
    pub z_index: Option<i32>,
}

/// Confidence scores for extraction accuracy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Confidence {
    pub overall: f32,
    pub text_extraction: f32,
    pub structure_detection: f32,
    pub table_extraction: f32,
    pub metadata_extraction: f32,
}

/// Result of document processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResult {
    pub document_id: Uuid,
    pub extracted_content: ExtractedContent,
    pub processing_time_ms: u64,
    pub agent_id: Option<Uuid>,
    pub neural_enhancements: Vec<NeuralEnhancement>,
    pub errors: Vec<ProcessingError>,
    pub warnings: Vec<String>,
}

/// Neural enhancement applied during processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralEnhancement {
    pub enhancement_type: EnhancementType,
    pub confidence_before: f32,
    pub confidence_after: f32,
    pub model_version: String,
    pub processing_time_ms: u32,
}

/// Type of neural enhancement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnhancementType {
    TextCorrection,
    LayoutAnalysis,
    TableDetection,
    ImageAnalysis,
    LanguageDetection,
    QualityAssessment,
}

/// Error that occurred during processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingError {
    pub error_type: ErrorType,
    pub message: String,
    pub block_id: Option<Uuid>,
    pub recoverable: bool,
}

/// Type of processing error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorType {
    ParseError,
    ExtractionError,
    NeuralError,
    ValidationError,
    TimeoutError,
    AgentError,
    CoordinationError,
}

impl Default for DocumentMetadata {
    fn default() -> Self {
        Self {
            title: None,
            author: None,
            created_at: None,
            modified_at: None,
            mime_type: None,
            size_bytes: None,
            language: None,
            custom_fields: HashMap::new(),
        }
    }
}

impl Default for Position {
    fn default() -> Self {
        Self {
            page: None,
            x: None,
            y: None,
            width: None,
            height: None,
            z_index: None,
        }
    }
}

impl Default for Confidence {
    fn default() -> Self {
        Self {
            overall: 0.0,
            text_extraction: 0.0,
            structure_detection: 0.0,
            table_extraction: 0.0,
            metadata_extraction: 0.0,
        }
    }
}

impl Document {
    pub fn new(source: DocumentSource) -> Self {
        Self {
            id: Uuid::new_v4(),
            source,
            metadata: DocumentMetadata::default(),
            content: None,
        }
    }
    
    pub fn with_metadata(mut self, metadata: DocumentMetadata) -> Self {
        self.metadata = metadata;
        self
    }
    
    pub fn with_content(mut self, content: Vec<u8>) -> Self {
        self.content = Some(content);
        self
    }
}

impl ContentBlock {
    pub fn new(block_type: BlockType, content: BlockContent) -> Self {
        Self {
            id: Uuid::new_v4(),
            block_type,
            content,
            position: Position::default(),
            confidence: 0.0,
            metadata: HashMap::new(),
        }
    }
    
    pub fn with_position(mut self, position: Position) -> Self {
        self.position = position;
        self
    }
    
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence;
        self
    }
}