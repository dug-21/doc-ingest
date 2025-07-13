//! Core data types for the neural document flow framework

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Document type classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DocumentType {
    Pdf,
    Text,
    Html,
    Markdown,
    Json,
    Xml,
    Image,
    Unknown,
}

/// Document source type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DocumentSourceType {
    File,
    Url,
    Base64,
    Memory,
}

/// Unique identifier for documents
pub type DocumentId = Uuid;

/// Unique identifier for processing sessions
pub type SessionId = Uuid;

/// Document representation with multimodal content support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// Unique document identifier
    pub id: DocumentId,
    
    /// Document type classification
    pub doc_type: DocumentType,
    
    /// Document source type
    pub source_type: DocumentSourceType,
    
    /// Raw content of the document
    pub raw_content: Vec<u8>,
    
    /// Document metadata
    pub metadata: DocumentMetadata,
    
    /// Document content in various modalities
    pub content: DocumentContent,
    
    /// Document structure information
    pub structure: DocumentStructure,
    
    /// Attached files or resources
    pub attachments: Vec<Attachment>,
    
    /// Processing history and annotations
    pub processing_history: Vec<ProcessingEvent>,
    
    /// Document creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Last modification timestamp
    pub updated_at: DateTime<Utc>,
}

/// Document structure information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocumentStructure {
    /// Number of pages (if applicable)
    pub page_count: Option<u32>,
    
    /// Document sections
    pub sections: Vec<DocumentSection>,
    
    /// Table of contents
    pub toc: Vec<TocEntry>,
}

/// Document section
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentSection {
    /// Section identifier
    pub id: String,
    
    /// Section title
    pub title: String,
    
    /// Section level (1 = top level)
    pub level: u32,
    
    /// Starting page
    pub start_page: Option<u32>,
    
    /// Ending page
    pub end_page: Option<u32>,
}

/// Table of contents entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TocEntry {
    /// Entry title
    pub title: String,
    
    /// Entry level
    pub level: u32,
    
    /// Page number
    pub page: Option<u32>,
}

/// Document attachment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attachment {
    /// Attachment identifier
    pub id: String,
    
    /// Attachment name
    pub name: String,
    
    /// MIME type
    pub mime_type: String,
    
    /// Attachment data
    pub data: Vec<u8>,
}

/// Document metadata containing descriptive information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocumentMetadata {
    /// Original document title
    pub title: Option<String>,
    
    /// Document author(s)
    pub authors: Vec<String>,
    
    /// Document source path or URL
    pub source: String,
    
    /// MIME type of the original document
    pub mime_type: String,
    
    /// File size in bytes
    pub size: Option<u64>,
    
    /// Document language (ISO 639-1 code)
    pub language: Option<String>,
    
    /// Custom metadata fields
    #[serde(default)]
    pub custom: HashMap<String, serde_json::Value>,
}

/// Multimodal document content
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DocumentContent {
    /// Extracted text content
    pub text: Option<String>,
    
    /// Image content (as base64 or binary data)
    pub images: Vec<ImageData>,
    
    /// Table data
    pub tables: Vec<TableData>,
    
    /// Structured data (JSON, XML, etc.)
    pub structured: HashMap<String, serde_json::Value>,
    
    /// Raw binary content
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub raw: Option<Vec<u8>>,
}

/// Image data representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    /// Image identifier
    pub id: String,
    
    /// Image format (PNG, JPEG, etc.)
    pub format: String,
    
    /// Image dimensions
    pub width: u32,
    pub height: u32,
    
    /// Image data (base64 encoded or binary)
    pub data: ImageContent,
    
    /// Alt text or caption
    pub caption: Option<String>,
}

/// Image content storage format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageContent {
    /// Base64 encoded image data
    Base64(String),
    
    /// Binary image data
    Binary(Vec<u8>),
    
    /// Reference to external image file
    File(String),
}

/// Table data representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableData {
    /// Table identifier
    pub id: String,
    
    /// Table headers
    pub headers: Vec<String>,
    
    /// Table rows
    pub rows: Vec<Vec<String>>,
    
    /// Table caption or title
    pub caption: Option<String>,
}

/// Processing event for tracking document processing history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingEvent {
    /// Event identifier
    pub id: Uuid,
    
    /// Processing session identifier
    pub session_id: SessionId,
    
    /// Name of the processor that generated this event
    pub processor_name: String,
    
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Event type
    pub event_type: ProcessingEventType,
    
    /// Event details
    pub details: serde_json::Value,
    
    /// Processing confidence score (0.0 to 1.0)
    pub confidence: Option<f64>,
}

/// Types of processing events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingEventType {
    /// Document parsing started
    ParseStart,
    
    /// Document parsing completed
    ParseComplete,
    
    /// Text extraction
    TextExtraction,
    
    /// Image extraction
    ImageExtraction,
    
    /// Table extraction
    TableExtraction,
    
    /// Neural processing
    NeuralProcessing,
    
    /// Annotation added
    Annotation,
    
    /// Error occurred
    Error,
    
    /// Custom event type
    Custom(String),
}

/// Processing result with confidence scoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingResultData<T> {
    /// The processed data
    pub data: T,
    
    /// Confidence score (0.0 to 1.0)
    pub confidence: f64,
    
    /// Processing metadata
    pub metadata: ProcessingMetadata,
    
    /// Any errors or warnings encountered
    pub issues: Vec<ProcessingIssue>,
}

/// Metadata about the processing operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    /// Processing duration in milliseconds
    pub duration_ms: u64,
    
    /// Memory usage in bytes
    pub memory_usage: Option<u64>,
    
    /// Processor version
    pub processor_version: String,
    
    /// Processing parameters used
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Processing issues (errors, warnings, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingIssue {
    /// Issue severity
    pub severity: IssueSeverity,
    
    /// Issue message
    pub message: String,
    
    /// Location in document where issue occurred
    pub location: Option<DocumentLocation>,
    
    /// Suggested resolution
    pub suggestion: Option<String>,
}

/// Issue severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    /// Informational message
    Info,
    
    /// Warning that doesn't prevent processing
    Warning,
    
    /// Error that affects processing quality
    Error,
    
    /// Critical error that prevents processing
    Critical,
}

/// Location within a document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentLocation {
    /// Page number (for paginated documents)
    pub page: Option<u32>,
    
    /// Line number
    pub line: Option<u32>,
    
    /// Column number
    pub column: Option<u32>,
    
    /// Character offset from document start
    pub offset: Option<u64>,
}

/// Processing pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Pipeline name
    pub name: String,
    
    /// Ordered list of processors
    pub processors: Vec<ProcessorConfig>,
    
    /// Global pipeline parameters
    pub parameters: HashMap<String, serde_json::Value>,
    
    /// Maximum processing time in seconds
    pub timeout_seconds: Option<u64>,
    
    /// Enable parallel processing where possible
    pub enable_parallel: bool,
}

/// Individual processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfig {
    /// Processor name/type
    pub name: String,
    
    /// Processor-specific parameters
    pub parameters: HashMap<String, serde_json::Value>,
    
    /// Whether this processor is enabled
    pub enabled: bool,
    
    /// Processor dependencies (must run after these)
    pub dependencies: Vec<String>,
}

impl Document {
    /// Create a new document with basic metadata
    pub fn new(source: String, mime_type: String) -> Self {
        let now = Utc::now();
        let doc_type = match mime_type.as_str() {
            "application/pdf" => DocumentType::Pdf,
            "text/plain" => DocumentType::Text,
            "text/html" => DocumentType::Html,
            "text/markdown" => DocumentType::Markdown,
            "application/json" => DocumentType::Json,
            "application/xml" | "text/xml" => DocumentType::Xml,
            t if t.starts_with("image/") => DocumentType::Image,
            _ => DocumentType::Unknown,
        };
        
        Self {
            id: Uuid::new_v4(),
            doc_type,
            source_type: DocumentSourceType::File,
            raw_content: Vec::new(),
            metadata: DocumentMetadata {
                title: None,
                authors: Vec::new(),
                source,
                mime_type,
                size: None,
                language: None,
                custom: HashMap::new(),
            },
            content: DocumentContent {
                text: None,
                images: Vec::new(),
                tables: Vec::new(),
                structured: HashMap::new(),
                raw: None,
            },
            structure: DocumentStructure::default(),
            attachments: Vec::new(),
            processing_history: Vec::new(),
            created_at: now,
            updated_at: now,
        }
    }
    
    /// Add a processing event to the document history
    pub fn add_processing_event(&mut self, event: ProcessingEvent) {
        self.processing_history.push(event);
        self.updated_at = Utc::now();
    }
    
    /// Get the latest processing event of a specific type
    pub fn latest_event(&self, event_type: &ProcessingEventType) -> Option<&ProcessingEvent> {
        self.processing_history
            .iter()
            .rev()
            .find(|event| std::mem::discriminant(&event.event_type) == std::mem::discriminant(event_type))
    }
}

impl<T> ProcessingResultData<T> {
    /// Create a successful processing result
    pub fn success(data: T, confidence: f64, metadata: ProcessingMetadata) -> Self {
        Self {
            data,
            confidence,
            metadata,
            issues: Vec::new(),
        }
    }
    
    /// Create a processing result with issues
    pub fn with_issues(data: T, confidence: f64, metadata: ProcessingMetadata, issues: Vec<ProcessingIssue>) -> Self {
        Self {
            data,
            confidence,
            metadata,
            issues,
        }
    }
    
    /// Check if the result has any errors
    pub fn has_errors(&self) -> bool {
        self.issues.iter().any(|issue| matches!(issue.severity, IssueSeverity::Error | IssueSeverity::Critical))
    }
    
    /// Get all error messages
    pub fn error_messages(&self) -> Vec<&str> {
        self.issues
            .iter()
            .filter(|issue| matches!(issue.severity, IssueSeverity::Error | IssueSeverity::Critical))
            .map(|issue| issue.message.as_str())
            .collect()
    }
}