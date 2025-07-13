//! Output formatting trait definitions

use crate::{Document, OutputResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Core trait for output formatters
#[async_trait]
pub trait OutputFormatter: Send + Sync {
    /// Get formatter name/type
    fn formatter_type(&self) -> &'static str;
    
    /// Format a single document
    async fn format(&self, document: &Document, options: &FormatOptions) -> OutputResult<FormattedOutput>;
    
    /// Format multiple documents
    async fn format_batch(&self, documents: &[Document], options: &FormatOptions) -> OutputResult<Vec<FormattedOutput>> {
        let mut results = Vec::new();
        for doc in documents {
            match self.format(doc, options).await {
                Ok(output) => results.push(output),
                Err(e) => return Err(e),
            }
        }
        Ok(results)
    }
    
    /// Validate format options
    fn validate_options(&self, options: &FormatOptions) -> OutputResult<()>;
    
    /// Get supported output formats
    fn supported_formats(&self) -> Vec<OutputFormat>;
    
    /// Get formatter capabilities
    fn capabilities(&self) -> FormatterCapabilities;
    
    /// Get configuration schema
    fn config_schema(&self) -> serde_json::Value;
}

/// Formatted output result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormattedOutput {
    /// Output format
    pub format: OutputFormat,
    
    /// Formatted content
    pub content: OutputContent,
    
    /// Output metadata
    pub metadata: OutputMetadata,
    
    /// Size of the output in bytes
    pub size_bytes: u64,
    
    /// Generation timestamp
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

/// Output content types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OutputContent {
    /// Text content
    Text(String),
    
    /// Binary content
    Binary(Vec<u8>),
    
    /// JSON structured data
    Json(serde_json::Value),
    
    /// XML content
    Xml(String),
    
    /// HTML content
    Html(String),
    
    /// Markdown content
    Markdown(String),
    
    /// PDF content
    Pdf(Vec<u8>),
    
    /// Multiple files/outputs
    Archive(Vec<NamedOutput>),
}

/// Named output for archives
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedOutput {
    /// File name
    pub name: String,
    
    /// File content
    pub content: OutputContent,
    
    /// MIME type
    pub mime_type: String,
}

/// Output format types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OutputFormat {
    /// Plain text
    Text,
    
    /// JSON format
    Json,
    
    /// XML format
    Xml,
    
    /// HTML format
    Html,
    
    /// Markdown format
    Markdown,
    
    /// PDF format
    Pdf,
    
    /// CSV format
    Csv,
    
    /// Excel format
    Excel,
    
    /// YAML format
    Yaml,
    
    /// TOML format
    Toml,
    
    /// Custom format
    Custom(String),
}

/// Formatting options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatOptions {
    /// Target output format
    pub format: OutputFormat,
    
    /// Include metadata in output
    pub include_metadata: bool,
    
    /// Include processing history
    pub include_history: bool,
    
    /// Include images
    pub include_images: bool,
    
    /// Include tables
    pub include_tables: bool,
    
    /// Compression settings
    pub compression: Option<CompressionOptions>,
    
    /// Template to use (if applicable)
    pub template: Option<String>,
    
    /// Custom formatting parameters
    pub custom_params: HashMap<String, serde_json::Value>,
    
    /// Output style/theme
    pub style: Option<String>,
    
    /// Language for localization
    pub language: Option<String>,
}

/// Compression options for output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionOptions {
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    
    /// Compression level (algorithm-specific)
    pub level: Option<u8>,
    
    /// Enable compression for specific content types
    pub content_types: Vec<String>,
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// Gzip compression
    Gzip,
    
    /// Zlib compression
    Zlib,
    
    /// Brotli compression
    Brotli,
    
    /// LZ4 compression
    Lz4,
    
    /// Custom compression
    Custom(String),
}

/// Output metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputMetadata {
    /// Original document ID
    pub document_id: uuid::Uuid,
    
    /// Formatter used
    pub formatter: String,
    
    /// Generation time in milliseconds
    pub generation_time_ms: u64,
    
    /// Output quality score (0.0 to 1.0)
    pub quality_score: f64,
    
    /// Warnings during formatting
    pub warnings: Vec<String>,
    
    /// Custom metadata
    pub custom: HashMap<String, serde_json::Value>,
}

/// Formatter capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatterCapabilities {
    /// Supported input document types
    pub input_types: Vec<String>,
    
    /// Supported output formats
    pub output_formats: Vec<OutputFormat>,
    
    /// Supports batch processing
    pub batch_processing: bool,
    
    /// Supports streaming
    pub streaming: bool,
    
    /// Supports templates
    pub templates: bool,
    
    /// Supports compression
    pub compression: bool,
    
    /// Maximum document size (bytes)
    pub max_document_size: Option<u64>,
    
    /// Memory requirements (MB)
    pub memory_requirements: Option<u64>,
}

/// Template system trait for advanced formatting
#[async_trait]
pub trait TemplateFormatter: OutputFormatter {
    /// Render document using a template
    async fn render_template(&self, document: &Document, template: &Template, context: &TemplateContext) -> OutputResult<FormattedOutput>;
    
    /// Validate template syntax
    fn validate_template(&self, template: &Template) -> OutputResult<()>;
    
    /// Get available template variables
    fn template_variables(&self) -> Vec<TemplateVariable>;
    
    /// Register custom template functions
    fn register_function(&mut self, name: String, function: Box<dyn TemplateFunction>);
}

/// Template definition
#[derive(Clone, Serialize, Deserialize)]
pub struct Template {
    /// Template name
    pub name: String,
    
    /// Template content
    pub content: String,
    
    /// Template format/engine
    pub format: TemplateFormat,
    
    /// Template variables
    pub variables: HashMap<String, TemplateVariable>,
}

/// Template format/engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemplateFormat {
    /// Handlebars template
    Handlebars,
    
    /// Jinja2-style template
    Jinja2,
    
    /// Mustache template
    Mustache,
    
    /// Custom template format
    Custom(String),
}

/// Template variable definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateVariable {
    /// Variable name
    pub name: String,
    
    /// Variable type
    pub var_type: TemplateVariableType,
    
    /// Variable description
    pub description: String,
    
    /// Required variable
    pub required: bool,
    
    /// Default value
    pub default: Option<serde_json::Value>,
}

/// Template variable types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TemplateVariableType {
    /// String variable
    String,
    
    /// Number variable
    Number,
    
    /// Boolean variable
    Boolean,
    
    /// Array variable
    Array,
    
    /// Object variable
    Object,
    
    /// Document content
    Document,
    
    /// Custom type
    Custom(String),
}

/// Template rendering context
pub struct TemplateContext {
    /// Template variables
    pub variables: HashMap<String, serde_json::Value>,
    
    /// Function names for serialization
    pub function_names: Vec<String>,
    
    /// Rendering options
    pub options: TemplateOptions,
}

/// Template rendering options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateOptions {
    /// Escape HTML by default
    pub escape_html: bool,
    
    /// Strict variable checking
    pub strict_variables: bool,
    
    /// Enable debugging
    pub debug: bool,
    
    /// Custom options
    pub custom: HashMap<String, serde_json::Value>,
}

/// Template function trait
pub trait TemplateFunction: Send + Sync {
    /// Execute the function with given arguments
    fn execute(&self, args: &[serde_json::Value]) -> OutputResult<serde_json::Value>;
    
    /// Get function signature
    fn signature(&self) -> TemplateFunctionSignature;
}

/// Template function signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateFunctionSignature {
    /// Function name
    pub name: String,
    
    /// Function description
    pub description: String,
    
    /// Function parameters
    pub parameters: Vec<TemplateParameter>,
    
    /// Return type
    pub return_type: TemplateVariableType,
}

/// Template function parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateParameter {
    /// Parameter name
    pub name: String,
    
    /// Parameter type
    pub param_type: TemplateVariableType,
    
    /// Parameter is required
    pub required: bool,
    
    /// Parameter description
    pub description: String,
}

impl Default for FormatOptions {
    fn default() -> Self {
        Self {
            format: OutputFormat::Json,
            include_metadata: true,
            include_history: false,
            include_images: true,
            include_tables: true,
            compression: None,
            template: None,
            custom_params: HashMap::new(),
            style: None,
            language: None,
        }
    }
}

impl OutputFormat {
    /// Get MIME type for the format
    pub fn mime_type(&self) -> &'static str {
        match self {
            OutputFormat::Text => "text/plain",
            OutputFormat::Json => "application/json",
            OutputFormat::Xml => "application/xml",
            OutputFormat::Html => "text/html",
            OutputFormat::Markdown => "text/markdown",
            OutputFormat::Pdf => "application/pdf",
            OutputFormat::Csv => "text/csv",
            OutputFormat::Excel => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            OutputFormat::Yaml => "application/x-yaml",
            OutputFormat::Toml => "application/toml",
            OutputFormat::Custom(_) => "application/octet-stream",
        }
    }
    
    /// Get file extension for the format
    pub fn file_extension(&self) -> &str {
        match self {
            OutputFormat::Text => "txt",
            OutputFormat::Json => "json",
            OutputFormat::Xml => "xml",
            OutputFormat::Html => "html",
            OutputFormat::Markdown => "md",
            OutputFormat::Pdf => "pdf",
            OutputFormat::Csv => "csv",
            OutputFormat::Excel => "xlsx",
            OutputFormat::Yaml => "yaml",
            OutputFormat::Toml => "toml",
            OutputFormat::Custom(ext) => ext.as_str(),
        }
    }
}