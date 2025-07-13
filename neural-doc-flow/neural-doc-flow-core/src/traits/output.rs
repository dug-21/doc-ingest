//! Output formatting traits for configurable document output
//!
//! This module defines traits for flexible output formatting that allows
//! users to define custom output formats and templates.

use async_trait::async_trait;
use std::collections::HashMap;
use crate::{FormatError, ExtractedDocument, OutputFormat, NeuralDocFlowResult};

/// Core trait for output formatting
/// 
/// This trait defines the interface for converting processed documents
/// into various output formats with user-configurable templates and schemas.
/// 
/// # Example Implementation
/// 
/// ```rust,no_run
/// use async_trait::async_trait;
/// use neural_doc_flow_core::{OutputFormatter, ExtractedDocument, FormatError, FormattedOutput};
/// 
/// struct JsonFormatter;
/// 
/// #[async_trait]
/// impl OutputFormatter for JsonFormatter {
///     fn formatter_id(&self) -> &str { "json" }
///     fn name(&self) -> &str { "JSON Formatter" }
///     fn version(&self) -> &str { "1.0.0" }
///     fn supported_formats(&self) -> &[&str] { &["json"] }
///     
///     async fn format(&self, document: &ExtractedDocument, format: &OutputFormat) -> Result<FormattedOutput, FormatError> {
///         let json = serde_json::to_string_pretty(document)?;
///         Ok(FormattedOutput {
///             format_type: "json".to_string(),
///             content: json.into_bytes(),
///             metadata: std::collections::HashMap::new(),
///         })
///     }
///     
///     fn format_schema(&self) -> serde_json::Value {
///         serde_json::json!({
///             "type": "object",
///             "properties": {
///                 "pretty": { "type": "boolean", "default": true }
///             }
///         })
///     }
///     
///     async fn initialize(&mut self, config: FormatterConfig) -> Result<(), FormatError> {
///         Ok(())
///     }
///     
///     async fn cleanup(&mut self) -> Result<(), FormatError> {
///         Ok(())
///     }
/// }
/// ```
#[async_trait]
pub trait OutputFormatter: Send + Sync {
    /// Unique identifier for this formatter
    fn formatter_id(&self) -> &str;

    /// Human-readable name of this formatter
    fn name(&self) -> &str;

    /// Formatter version
    fn version(&self) -> &str;

    /// Supported output format names
    fn supported_formats(&self) -> &[&str];

    /// Format a document to the specified output format
    /// 
    /// This is the main formatting method that converts an extracted document
    /// into the requested output format.
    /// 
    /// # Parameters
    /// - `document`: The document to format
    /// - `format`: Output format specification
    /// 
    /// # Returns
    /// - `Ok(FormattedOutput)` with formatted content
    /// - `Err(_)` if formatting failed
    async fn format(
        &self,
        document: &ExtractedDocument,
        format: &OutputFormat,
    ) -> Result<FormattedOutput, FormatError>;

    /// Get the JSON schema for format options
    /// 
    /// Returns a JSON Schema describing the options available for this formatter.
    fn format_schema(&self) -> serde_json::Value;

    /// Initialize the formatter with configuration
    async fn initialize(&mut self, config: FormatterConfig) -> Result<(), FormatError>;

    /// Clean up formatter resources
    async fn cleanup(&mut self) -> Result<(), FormatError>;

    /// Check if formatter supports the output format
    fn supports_format(&self, format_name: &str) -> bool {
        self.supported_formats().contains(&format_name)
    }

    /// Get formatter capabilities
    fn capabilities(&self) -> FormatterCapabilities {
        FormatterCapabilities::default()
    }

    /// Validate format options before formatting
    async fn validate_format(&self, format: &OutputFormat) -> Result<(), FormatError> {
        Ok(()) // Default implementation does no validation
    }

    /// Get MIME type for the output format
    fn get_mime_type(&self, format_name: &str) -> Option<String> {
        None // Default implementation returns no MIME type
    }

    /// Get file extension for the output format
    fn get_file_extension(&self, format_name: &str) -> Option<String> {
        None // Default implementation returns no extension
    }
}

/// Formatted output result
#[derive(Debug, Clone)]
pub struct FormattedOutput {
    /// Format type identifier
    pub format_type: String,
    /// Formatted content as bytes
    pub content: Vec<u8>,
    /// Output metadata
    pub metadata: HashMap<String, String>,
}

impl FormattedOutput {
    /// Create new formatted output
    pub fn new(format_type: impl Into<String>, content: Vec<u8>) -> Self {
        Self {
            format_type: format_type.into(),
            content,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get content as string (if UTF-8)
    pub fn as_string(&self) -> Result<String, std::string::FromUtf8Error> {
        String::from_utf8(self.content.clone())
    }

    /// Get content size in bytes
    pub fn size(&self) -> usize {
        self.content.len()
    }
}

/// Configuration for output formatters
#[derive(Debug, Clone)]
pub struct FormatterConfig {
    /// Formatter-specific settings
    pub settings: serde_json::Value,
    /// Template directory path
    pub template_directory: Option<std::path::PathBuf>,
    /// Cache settings
    pub cache_templates: bool,
    /// Maximum output size (bytes)
    pub max_output_size: Option<usize>,
    /// Custom format definitions
    pub custom_formats: HashMap<String, CustomFormatDefinition>,
}

impl Default for FormatterConfig {
    fn default() -> Self {
        Self {
            settings: serde_json::Value::Null,
            template_directory: None,
            cache_templates: true,
            max_output_size: None,
            custom_formats: HashMap::new(),
        }
    }
}

/// Custom format definition
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CustomFormatDefinition {
    /// Format name
    pub name: String,
    /// Format description
    pub description: String,
    /// Template content
    pub template: String,
    /// Template engine to use
    pub engine: TemplateEngine,
    /// Output MIME type
    pub mime_type: Option<String>,
    /// File extension
    pub file_extension: Option<String>,
    /// Format options schema
    pub options_schema: Option<serde_json::Value>,
}

/// Supported template engines
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub enum TemplateEngine {
    /// Handlebars template engine
    Handlebars,
    /// Tera template engine
    Tera,
    /// Mustache template engine
    Mustache,
    /// Custom template engine
    Custom(String),
}

/// Formatter capabilities
#[derive(Debug, Clone, Default)]
pub struct FormatterCapabilities {
    /// Supports custom templates
    pub custom_templates: bool,
    /// Supports streaming output
    pub streaming_output: bool,
    /// Supports compression
    pub compression: bool,
    /// Supports encryption
    pub encryption: bool,
    /// Supports batch formatting
    pub batch_formatting: bool,
    /// Supports incremental updates
    pub incremental_updates: bool,
    /// Maximum document size (bytes)
    pub max_document_size: Option<usize>,
    /// Supported compression formats
    pub compression_formats: Vec<String>,
    /// Supported encryption methods
    pub encryption_methods: Vec<String>,
}

/// Trait for template-based formatters
/// 
/// This trait extends OutputFormatter with template-specific capabilities
/// for formatters that use templates to generate output.
#[async_trait]
pub trait TemplateFormatter: OutputFormatter {
    /// Render template with document data
    /// 
    /// # Parameters
    /// - `template`: Template content
    /// - `document`: Document to render
    /// - `context`: Additional template context
    /// 
    /// # Returns
    /// - `Ok(String)` with rendered content
    /// - `Err(_)` if rendering failed
    async fn render_template(
        &self,
        template: &str,
        document: &ExtractedDocument,
        context: &TemplateContext,
    ) -> Result<String, FormatError>;

    /// Register a custom template
    async fn register_template(
        &mut self,
        name: &str,
        template: &str,
    ) -> Result<(), FormatError>;

    /// Load templates from directory
    async fn load_templates_from_dir(
        &mut self,
        directory: &std::path::Path,
    ) -> Result<Vec<String>, FormatError>;

    /// Get available template names
    fn list_templates(&self) -> Vec<String>;

    /// Check if template exists
    fn has_template(&self, name: &str) -> bool;

    /// Remove a template
    async fn remove_template(&mut self, name: &str) -> Result<(), FormatError>;
}

/// Template rendering context
#[derive(Debug, Clone, Default)]
pub struct TemplateContext {
    /// Additional variables for template rendering
    pub variables: HashMap<String, serde_json::Value>,
    /// Helper functions available in templates
    pub helpers: HashMap<String, String>,
    /// Global settings
    pub settings: HashMap<String, String>,
}

impl TemplateContext {
    /// Create new template context
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a variable
    pub fn set_variable(&mut self, key: impl Into<String>, value: serde_json::Value) {
        self.variables.insert(key.into(), value);
    }

    /// Get a variable
    pub fn get_variable(&self, key: &str) -> Option<&serde_json::Value> {
        self.variables.get(key)
    }

    /// Add a helper function
    pub fn set_helper(&mut self, name: impl Into<String>, code: impl Into<String>) {
        self.helpers.insert(name.into(), code.into());
    }

    /// Add a setting
    pub fn set_setting(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.settings.insert(key.into(), value.into());
    }
}

/// Trait for schema-based output validation
/// 
/// This trait provides capabilities for validating output against schemas
/// and ensuring output meets specific requirements.
pub trait SchemaValidator: Send + Sync {
    /// Validate output against a schema
    /// 
    /// # Parameters
    /// - `output`: The formatted output to validate
    /// - `schema`: JSON Schema to validate against
    /// 
    /// # Returns
    /// - `Ok(ValidationResult)` with validation results
    /// - `Err(_)` if validation failed
    fn validate_output(
        &self,
        output: &FormattedOutput,
        schema: &serde_json::Value,
    ) -> Result<ValidationResult, FormatError>;

    /// Generate schema from document
    /// 
    /// Creates a JSON Schema that describes the structure of the document.
    fn generate_schema(&self, document: &ExtractedDocument) -> serde_json::Value;

    /// Validate schema itself
    fn validate_schema(&self, schema: &serde_json::Value) -> Result<(), FormatError>;
}

/// Output validation result
#[derive(Debug, Clone, Default)]
pub struct ValidationResult {
    /// Whether output is valid
    pub is_valid: bool,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
}

/// Validation error details
#[derive(Debug, Clone)]
pub struct ValidationError {
    /// Path to the invalid field
    pub path: String,
    /// Error message
    pub message: String,
    /// Expected value or type
    pub expected: Option<String>,
    /// Actual value found
    pub actual: Option<String>,
}

/// Validation warning details
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    /// Path to the field with warning
    pub path: String,
    /// Warning message
    pub message: String,
    /// Suggestion for improvement
    pub suggestion: Option<String>,
}

/// Trait for format conversion
/// 
/// This trait provides capabilities for converting between different formats.
#[async_trait]
pub trait FormatConverter: Send + Sync {
    /// Convert from one format to another
    /// 
    /// # Parameters
    /// - `input`: Input formatted content
    /// - `from_format`: Source format
    /// - `to_format`: Target format
    /// 
    /// # Returns
    /// - `Ok(FormattedOutput)` with converted content
    /// - `Err(_)` if conversion failed
    async fn convert(
        &self,
        input: &FormattedOutput,
        from_format: &str,
        to_format: &str,
    ) -> Result<FormattedOutput, FormatError>;

    /// Get supported conversion paths
    fn supported_conversions(&self) -> Vec<ConversionPath>;

    /// Check if conversion is supported
    fn supports_conversion(&self, from_format: &str, to_format: &str) -> bool {
        self.supported_conversions()
            .iter()
            .any(|path| path.from == from_format && path.to == to_format)
    }
}

/// Format conversion path
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversionPath {
    /// Source format
    pub from: String,
    /// Target format
    pub to: String,
    /// Conversion quality (0.0 to 1.0)
    pub quality: f32,
    /// Whether conversion is lossy
    pub lossy: bool,
}

/// Built-in formatter implementations
pub mod builtin {
    use super::*;

    /// JSON formatter implementation
    pub struct JsonFormatter {
        config: Option<FormatterConfig>,
    }

    impl JsonFormatter {
        /// Create new JSON formatter
        pub fn new() -> Self {
            Self { config: None }
        }
    }

    impl Default for JsonFormatter {
        fn default() -> Self {
            Self::new()
        }
    }

    #[async_trait]
    impl OutputFormatter for JsonFormatter {
        fn formatter_id(&self) -> &str { "json" }
        fn name(&self) -> &str { "JSON Formatter" }
        fn version(&self) -> &str { "1.0.0" }
        fn supported_formats(&self) -> &[&str] { &["json"] }

        async fn format(
            &self,
            document: &ExtractedDocument,
            format: &OutputFormat,
        ) -> Result<FormattedOutput, FormatError> {
            let pretty = match format {
                OutputFormat::Custom { options, .. } => {
                    options.get("pretty")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(true)
                }
                _ => true,
            };

            let json = if pretty {
                serde_json::to_string_pretty(document)?
            } else {
                serde_json::to_string(document)?
            };

            Ok(FormattedOutput::new("json", json.into_bytes())
                .with_metadata("mime_type", "application/json")
                .with_metadata("encoding", "utf-8"))
        }

        fn format_schema(&self) -> serde_json::Value {
            serde_json::json!({
                "type": "object",
                "properties": {
                    "pretty": {
                        "type": "boolean",
                        "description": "Format JSON with indentation",
                        "default": true
                    }
                }
            })
        }

        async fn initialize(&mut self, config: FormatterConfig) -> Result<(), FormatError> {
            self.config = Some(config);
            Ok(())
        }

        async fn cleanup(&mut self) -> Result<(), FormatError> {
            Ok(())
        }

        fn get_mime_type(&self, _format_name: &str) -> Option<String> {
            Some("application/json".to_string())
        }

        fn get_file_extension(&self, _format_name: &str) -> Option<String> {
            Some("json".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    #[tokio::test]
    async fn test_json_formatter() {
        let mut formatter = builtin::JsonFormatter::new();
        assert_eq!(formatter.formatter_id(), "json");
        assert!(formatter.supports_format("json"));

        let config = FormatterConfig::default();
        formatter.initialize(config).await.unwrap();

        let document = ExtractedDocument {
            id: "test".to_string(),
            source_id: "test_source".to_string(),
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
        };

        let format = OutputFormat::Format("json".to_string());
        let output = formatter.format(&document, &format).await.unwrap();
        
        assert_eq!(output.format_type, "json");
        assert!(!output.content.is_empty());
        assert_eq!(output.metadata.get("mime_type"), Some(&"application/json".to_string()));
    }

    #[test]
    fn test_formatted_output() {
        let output = FormattedOutput::new("json", b"{}".to_vec())
            .with_metadata("mime_type", "application/json")
            .with_metadata("encoding", "utf-8");

        assert_eq!(output.format_type, "json");
        assert_eq!(output.size(), 2);
        assert_eq!(output.as_string().unwrap(), "{}");
        assert_eq!(output.metadata.get("mime_type"), Some(&"application/json".to_string()));
    }

    #[test]
    fn test_template_context() {
        let mut context = TemplateContext::new();
        
        context.set_variable("title", serde_json::json!("Test Document"));
        context.set_helper("format_date", "function(date) { return date.toISOString(); }");
        context.set_setting("theme", "default");

        assert_eq!(
            context.get_variable("title"), 
            Some(&serde_json::json!("Test Document"))
        );
        assert!(context.helpers.contains_key("format_date"));
        assert!(context.settings.contains_key("theme"));
    }

    #[test]
    fn test_conversion_path() {
        let path = ConversionPath {
            from: "json".to_string(),
            to: "xml".to_string(),
            quality: 0.95,
            lossy: false,
        };

        assert_eq!(path.from, "json");
        assert_eq!(path.to, "xml");
        assert!(!path.lossy);
    }

    #[test]
    fn test_template_engines() {
        assert_eq!(TemplateEngine::Handlebars, TemplateEngine::Handlebars);
        assert_ne!(TemplateEngine::Handlebars, TemplateEngine::Tera);
        
        let custom = TemplateEngine::Custom("custom_engine".to_string());
        assert!(matches!(custom, TemplateEngine::Custom(_)));
    }
}