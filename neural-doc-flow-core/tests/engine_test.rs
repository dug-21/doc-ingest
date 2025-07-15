//! Comprehensive unit tests for engine.rs to achieve 100% test coverage
//!
//! This test suite covers:
//! 1. DocumentEngine initialization and configuration
//! 2. 4-layer architecture testing (Input, Processing, Validation, Output)
//! 3. Document processing pipeline end-to-end
//! 4. Error handling and edge cases
//! 5. Plugin system integration
//! 6. Security policy enforcement
//! 7. Performance monitoring

use neural_doc_flow_core::{
    config::{NeuralDocFlowConfig, SecurityScanMode, SecurityAction},
    engine::{
        DocumentEngine, SecurityAnalysis, SecurityProcessor, ThreatLevel,
        SourceRegistry, PluginManager, SchemaEngine, ExtractionSchema, 
        DomainConfig, ValidationRule, DefaultOutputFormatter, OutputTemplate,
        DefaultProcessorPipeline, ProcessedDocument, PerformanceMonitor,
        PluginInfo
    },
    document::{Document, DocumentBuilder},
    error::{ProcessingError, SourceError},
    traits::{
        DocumentSource, DocumentSourceFactory, Processor, ProcessorPipeline,
        OutputFormatter, TemplateFormatter, source::ValidationResult,
        output::{FormatOptions, OutputFormat, Template, TemplateContext},
        ProcessingContext, ProcessorConfig
    },
};
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use async_trait::async_trait;
use serde_json;
use chrono::Utc;

// ==================== MOCK IMPLEMENTATIONS ====================

/// Mock document source for testing
#[derive(Debug)]
struct MockDocumentSource {
    source_type: String,
    should_handle: bool,
    should_fail: bool,
}

impl MockDocumentSource {
    fn new(source_type: &str) -> Self {
        Self {
            source_type: source_type.to_string(),
            should_handle: true,
            should_fail: false,
        }
    }
    
    fn with_failure(mut self) -> Self {
        self.should_fail = true;
        self
    }
    
    fn with_no_handle(mut self) -> Self {
        self.should_handle = false;
        self
    }
}

#[async_trait]
impl DocumentSource for MockDocumentSource {
    fn source_type(&self) -> &'static str {
        "mock_source"
    }
    
    async fn can_handle(&self, _input: &str) -> bool {
        self.should_handle
    }
    
    async fn load_document(&self, input: &str) -> Result<Document, SourceError> {
        if self.should_fail {
            return Err(SourceError::ParseError { reason: "Mock failure".to_string() });
        }
        
        Ok(DocumentBuilder::new()
            .text_content(input)
            .mime_type("text/plain")
            .source(&self.source_type)
            .build())
    }
    
    async fn get_metadata(&self, _input: &str) -> Result<neural_doc_flow_core::traits::DocumentMetadata, SourceError> {
        Ok(neural_doc_flow_core::traits::DocumentMetadata {
            name: "Mock Document".to_string(),
            size: Some(100),
            mime_type: "text/plain".to_string(),
            modified: Some(chrono::Utc::now()),
            attributes: std::collections::HashMap::new(),
        })
    }
    
    async fn validate(&self, _input: &str) -> Result<ValidationResult, SourceError> {
        Ok(ValidationResult::success())
    }
    
    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["txt", "text"]
    }
    
    fn supported_mime_types(&self) -> Vec<&'static str> {
        vec!["text/plain"]
    }
}

/// Mock source factory for testing
#[derive(Debug)]
struct MockSourceFactory {
    source_type: String,
}

impl MockSourceFactory {
    fn new(source_type: &str) -> Self {
        Self {
            source_type: source_type.to_string(),
        }
    }
}

impl DocumentSourceFactory for MockSourceFactory {
    fn source_type(&self) -> &'static str {
        "mock_source"
    }
    
    fn create_source(&self, _config: &neural_doc_flow_core::traits::source::SourceConfig) -> Result<Box<dyn DocumentSource>, SourceError> {
        Ok(Box::new(MockDocumentSource::new(&self.source_type)))
    }
    
    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "test_property": {
                    "type": "string"
                }
            }
        })
    }
}

/// Mock security processor for testing
#[derive(Debug)]
struct MockSecurityProcessor {
    should_fail: bool,
    threat_level: ThreatLevel,
    recommended_action: SecurityAction,
}

impl MockSecurityProcessor {
    fn new() -> Self {
        Self {
            should_fail: false,
            threat_level: ThreatLevel::Safe,
            recommended_action: SecurityAction::Allow,
        }
    }
    
    fn with_failure(mut self) -> Self {
        self.should_fail = true;
        self
    }
    
    fn with_threat_level(mut self, level: ThreatLevel) -> Self {
        self.threat_level = level;
        self
    }
    
    fn with_action(mut self, action: SecurityAction) -> Self {
        self.recommended_action = action;
        self
    }
}

#[async_trait]
impl SecurityProcessor for MockSecurityProcessor {
    async fn scan(&mut self, _document: &Document) -> Result<SecurityAnalysis, ProcessingError> {
        if self.should_fail {
            return Err(ProcessingError::ValidationError("Security scan failed".to_string()));
        }
        
        Ok(SecurityAnalysis {
            threat_level: self.threat_level,
            malware_probability: 0.0,
            threat_categories: vec![],
            anomaly_score: 0.0,
            behavioral_risks: vec![],
            recommended_action: self.recommended_action.clone(),
        })
    }
}

/// Mock processor for testing
#[derive(Debug)]
struct MockProcessor {
    name: String,
    should_fail: bool,
}

impl MockProcessor {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            should_fail: false,
        }
    }
    
    fn with_failure(mut self) -> Self {
        self.should_fail = true;
        self
    }
}

#[async_trait]
impl Processor for MockProcessor {
    fn name(&self) -> &str {
        &self.name
    }
    
    fn version(&self) -> &str {
        "1.0.0"
    }
    
    async fn process(&self, document: Document, _context: &ProcessingContext) -> Result<Document, ProcessingError> {
        if self.should_fail {
            return Err(ProcessingError::ValidationError("Mock processor failure".to_string()));
        }
        Ok(document)
    }
    
    fn can_process(&self, _document: &Document) -> bool {
        true
    }
    
    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "test_option": {
                    "type": "boolean"
                }
            }
        })
    }
    
    async fn initialize(&mut self, _config: &ProcessorConfig) -> Result<(), ProcessingError> {
        Ok(())
    }
    
    async fn cleanup(&self) -> Result<(), ProcessingError> {
        Ok(())
    }
}

// ==================== LAYER 1: CORE ENGINE TESTS ====================

#[tokio::test]
async fn test_document_engine_creation() {
    let config = NeuralDocFlowConfig::default();
    let engine = DocumentEngine::new(config);
    
    assert!(engine.is_ok(), "DocumentEngine should be created successfully");
}

#[tokio::test]
async fn test_document_engine_initialization() {
    let config = NeuralDocFlowConfig::default();
    let engine = DocumentEngine::new(config).unwrap();
    
    let result = engine.initialize().await;
    assert!(result.is_ok(), "DocumentEngine initialization should succeed");
}

#[tokio::test]
async fn test_engine_available_schemas() {
    let config = NeuralDocFlowConfig::default();
    let engine = DocumentEngine::new(config).unwrap();
    
    engine.initialize().await.unwrap();
    
    let schemas = engine.available_schemas().await;
    assert!(!schemas.is_empty(), "Should have default schemas");
    assert!(schemas.contains(&"general_document".to_string()));
    assert!(schemas.contains(&"legal_document".to_string()));
    assert!(schemas.contains(&"medical_document".to_string()));
}

#[tokio::test]
async fn test_engine_available_formats() {
    let config = NeuralDocFlowConfig::default();
    let engine = DocumentEngine::new(config).unwrap();
    
    engine.initialize().await.unwrap();
    
    let formats = engine.available_formats().await;
    assert!(!formats.is_empty(), "Should have supported formats");
    assert!(formats.contains(&"text".to_string()));
    assert!(formats.contains(&"json".to_string()));
}

#[tokio::test]
async fn test_engine_register_custom_schema() {
    let config = NeuralDocFlowConfig::default();
    let engine = DocumentEngine::new(config).unwrap();
    
    let custom_schema = ExtractionSchema {
        name: "custom_test".to_string(),
        description: "Custom test schema".to_string(),
        version: "1.0.0".to_string(),
        required_fields: vec!["title".to_string()],
        optional_fields: vec!["description".to_string()],
        field_types: HashMap::from([
            ("title".to_string(), "string".to_string()),
            ("description".to_string(), "string".to_string()),
        ]),
        validation_rules: vec![],
        domain_specific: None,
    };
    
    let result = engine.register_schema(custom_schema).await;
    assert!(result.is_ok(), "Custom schema registration should succeed");
}

#[tokio::test]
async fn test_engine_register_custom_template() {
    let config = NeuralDocFlowConfig::default();
    let engine = DocumentEngine::new(config).unwrap();
    
    let custom_template = OutputTemplate {
        name: "custom_test".to_string(),
        format: "text".to_string(),
        template: "Title: {{title}}\nContent: {{content}}".to_string(),
        schema: None,
    };
    
    let result = engine.register_template(custom_template).await;
    assert!(result.is_ok(), "Custom template registration should succeed");
}

// ==================== LAYER 2: PLUGIN SYSTEM TESTS ====================

#[tokio::test]
async fn test_source_registry_creation() {
    let mut registry = SourceRegistry::new();
    
    // Test basic functionality
    let mock_source = Box::new(MockDocumentSource::new("test_source"));
    registry.register_source(mock_source);
    
    let sources = registry.available_sources();
    assert!(sources.contains(&"test_source".to_string()));
    
    let source = registry.get_source("test_source");
    assert!(source.is_some());
    
    let source = registry.get_source("non_existent");
    assert!(source.is_none());
}

#[tokio::test]
async fn test_source_registry_find_source() {
    let mut registry = SourceRegistry::new();
    
    let mock_source = Box::new(MockDocumentSource::new("test_source"));
    registry.register_source(mock_source);
    
    let found = registry.find_source_for_input("test_input").await;
    assert!(found.is_some());
    
    // Test with source that can't handle input
    let no_handle_source = Box::new(MockDocumentSource::new("no_handle").with_no_handle());
    registry.register_source(no_handle_source);
    
    let found = registry.find_source_for_input("test_input").await;
    assert!(found.is_some()); // Should find the first source
}

#[tokio::test]
async fn test_source_registry_with_factory() {
    let mut registry = SourceRegistry::new();
    
    let factory = Box::new(MockSourceFactory::new("factory_source"));
    registry.register_factory(factory);
    
    // Test factory registration
    assert!(registry.available_sources().is_empty()); // Factory doesn't add sources directly
}

#[tokio::test]
async fn test_plugin_manager_creation() {
    let mut manager = PluginManager::new("./test_plugins".to_string());
    
    // Test plugin loading
    let result = manager.load_plugin("test_plugin").await;
    assert!(result.is_ok());
    
    let plugins = manager.list_plugins();
    assert_eq!(plugins.len(), 1);
    assert_eq!(plugins[0].name, "test_plugin");
    
    // Test plugin info retrieval
    let info = manager.get_plugin_info("test_plugin");
    assert!(info.is_some());
    assert_eq!(info.unwrap().name, "test_plugin");
    
    // Test plugin unloading
    let result = manager.unload_plugin("test_plugin").await;
    assert!(result.is_ok());
    
    let plugins = manager.list_plugins();
    assert_eq!(plugins.len(), 0);
}

#[tokio::test]
async fn test_plugin_manager_nonexistent_plugin() {
    let mut manager = PluginManager::new("./test_plugins".to_string());
    
    let info = manager.get_plugin_info("nonexistent");
    assert!(info.is_none());
    
    let result = manager.unload_plugin("nonexistent").await;
    assert!(result.is_ok()); // Should not fail for non-existent plugin
}

#[tokio::test]
async fn test_engine_plugin_reload() {
    let config = NeuralDocFlowConfig::default();
    let engine = DocumentEngine::new(config).unwrap();
    
    let result = engine.reload_plugin("test_plugin").await;
    assert!(result.is_ok(), "Plugin reload should succeed");
}

// ==================== LAYER 3: SCHEMA VALIDATION TESTS ====================

#[tokio::test]
async fn test_schema_engine_creation() {
    let mut schema_engine = SchemaEngine::new();
    
    let test_schema = ExtractionSchema {
        name: "test_schema".to_string(),
        description: "Test schema".to_string(),
        version: "1.0.0".to_string(),
        required_fields: vec!["text".to_string()],
        optional_fields: vec!["title".to_string()],
        field_types: HashMap::from([
            ("text".to_string(), "string".to_string()),
            ("title".to_string(), "string".to_string()),
        ]),
        validation_rules: vec![],
        domain_specific: None,
    };
    
    schema_engine.register_schema(test_schema);
    
    let schema = schema_engine.get_schema("test_schema");
    assert!(schema.is_some());
    assert_eq!(schema.unwrap().name, "test_schema");
}

#[tokio::test]
async fn test_schema_engine_validation() {
    let mut schema_engine = SchemaEngine::new();
    
    let test_schema = ExtractionSchema {
        name: "test_schema".to_string(),
        description: "Test schema".to_string(),
        version: "1.0.0".to_string(),
        required_fields: vec!["text".to_string()],
        optional_fields: vec!["title".to_string()],
        field_types: HashMap::from([
            ("text".to_string(), "string".to_string()),
            ("title".to_string(), "string".to_string()),
        ]),
        validation_rules: vec![],
        domain_specific: None,
    };
    
    schema_engine.register_schema(test_schema);
    
    let document = DocumentBuilder::new()
        .text_content("Test content")
        .build();
    
    let validation = schema_engine.validate_document(&document, "test_schema").await;
    assert!(validation.is_ok());
    
    let validation_result = validation.unwrap();
    assert!(validation_result.is_valid);
}

#[tokio::test]
async fn test_schema_engine_validation_missing_field() {
    let mut schema_engine = SchemaEngine::new();
    
    let test_schema = ExtractionSchema {
        name: "test_schema".to_string(),
        description: "Test schema".to_string(),
        version: "1.0.0".to_string(),
        required_fields: vec!["missing_field".to_string()],
        optional_fields: vec![],
        field_types: HashMap::new(),
        validation_rules: vec![],
        domain_specific: None,
    };
    
    schema_engine.register_schema(test_schema);
    
    let document = DocumentBuilder::new()
        .text_content("Test content")
        .build();
    
    let validation = schema_engine.validate_document(&document, "test_schema").await;
    assert!(validation.is_ok());
    
    let validation_result = validation.unwrap();
    assert!(!validation_result.is_valid);
    assert!(!validation_result.issues.is_empty());
}

#[tokio::test]
async fn test_schema_engine_data_extraction() {
    let mut schema_engine = SchemaEngine::new();
    
    let test_schema = ExtractionSchema {
        name: "test_schema".to_string(),
        description: "Test schema".to_string(),
        version: "1.0.0".to_string(),
        required_fields: vec!["text".to_string()],
        optional_fields: vec!["title".to_string()],
        field_types: HashMap::from([
            ("text".to_string(), "string".to_string()),
            ("title".to_string(), "string".to_string()),
        ]),
        validation_rules: vec![],
        domain_specific: None,
    };
    
    schema_engine.register_schema(test_schema);
    
    let document = DocumentBuilder::new()
        .text_content("Test content")
        .title("Test Title")
        .build();
    
    let extraction = schema_engine.extract_data(&document, "test_schema").await;
    assert!(extraction.is_ok());
    
    let extracted_data = extraction.unwrap();
    assert!(extracted_data.contains_key("text"));
    assert!(extracted_data.contains_key("title"));
}

#[tokio::test]
async fn test_schema_engine_nonexistent_schema() {
    let schema_engine = SchemaEngine::new();
    
    let document = DocumentBuilder::new()
        .text_content("Test content")
        .build();
    
    let validation = schema_engine.validate_document(&document, "nonexistent").await;
    assert!(validation.is_err());
    
    let extraction = schema_engine.extract_data(&document, "nonexistent").await;
    assert!(extraction.is_err());
}

#[tokio::test]
async fn test_schema_with_domain_config() {
    let mut schema_engine = SchemaEngine::new();
    
    let domain_config = DomainConfig {
        domain: "medical".to_string(),
        specialized_fields: vec!["patient_id".to_string()],
        compliance_requirements: vec!["hipaa".to_string()],
        processing_rules: HashMap::new(),
    };
    
    let test_schema = ExtractionSchema {
        name: "medical_test".to_string(),
        description: "Medical test schema".to_string(),
        version: "1.0.0".to_string(),
        required_fields: vec!["patient_id".to_string()],
        optional_fields: vec![],
        field_types: HashMap::from([
            ("patient_id".to_string(), "string".to_string()),
        ]),
        validation_rules: vec![],
        domain_specific: Some(domain_config),
    };
    
    schema_engine.register_schema(test_schema);
    
    let schema = schema_engine.get_schema("medical_test");
    assert!(schema.is_some());
    assert!(schema.unwrap().domain_specific.is_some());
}

// ==================== LAYER 4: OUTPUT FORMATTING TESTS ====================

#[tokio::test]
async fn test_default_output_formatter_creation() {
    let formatter = DefaultOutputFormatter::new();
    let templates = formatter.available_templates();
    
    assert!(templates.contains(&"json".to_string()));
    assert!(templates.contains(&"xml".to_string()));
    assert!(templates.contains(&"text".to_string()));
}

#[tokio::test]
async fn test_default_output_formatter_format() {
    let formatter = DefaultOutputFormatter::new();
    
    let document = DocumentBuilder::new()
        .text_content("Test content")
        .title("Test Title")
        .build();
    
    let options = FormatOptions {
        format: OutputFormat::Json,
        include_metadata: true,
        include_history: false,
        include_images: true,
        include_tables: true,
        compression: None,
        template: Some("json".to_string()),
        custom_params: HashMap::new(),
        style: None,
        language: None,
    };
    
    let result = formatter.format(&document, &options).await;
    assert!(result.is_ok());
    
    let formatted = result.unwrap();
    assert_eq!(formatted.format, OutputFormat::Json);
    assert!(formatted.size_bytes > 0);
}

#[tokio::test]
async fn test_default_output_formatter_invalid_template() {
    let formatter = DefaultOutputFormatter::new();
    
    let document = DocumentBuilder::new()
        .text_content("Test content")
        .build();
    
    let options = FormatOptions {
        format: OutputFormat::Json,
        include_metadata: true,
        include_history: false,
        include_images: true,
        include_tables: true,
        compression: None,
        template: Some("nonexistent".to_string()),
        custom_params: HashMap::new(),
        style: None,
        language: None,
    };
    
    let result = formatter.format(&document, &options).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_default_output_formatter_capabilities() {
    let formatter = DefaultOutputFormatter::new();
    
    let capabilities = formatter.capabilities();
    assert!(capabilities.templates);
    assert!(capabilities.batch_processing);
    assert!(!capabilities.streaming);
    assert!(!capabilities.compression);
}

#[tokio::test]
async fn test_default_output_formatter_custom_template() {
    let mut formatter = DefaultOutputFormatter::new();
    
    let custom_template = OutputTemplate {
        name: "custom".to_string(),
        format: "text".to_string(),
        template: "Title: {{title}}\nContent: {{content.text}}".to_string(),
        schema: None,
    };
    
    formatter.register_template(custom_template);
    
    let templates = formatter.available_templates();
    assert!(templates.contains(&"custom".to_string()));
}

#[tokio::test]
async fn test_template_formatter_render() {
    let formatter = DefaultOutputFormatter::new();
    
    let document = DocumentBuilder::new()
        .text_content("Test content")
        .build();
    
    let template = Template {
        name: "test_template".to_string(),
        format: neural_doc_flow_core::traits::output::TemplateFormat::Handlebars,
        content: "Document ID: {{document_id}}\nContent: {{content}}".to_string(),
        variables: HashMap::new(),
    };
    
    let context = TemplateContext {
        variables: HashMap::new(),
        function_names: vec![],
        options: neural_doc_flow_core::traits::output::TemplateOptions {
            escape_html: false,
            strict_variables: false,
            debug: false,
            custom: HashMap::new(),
        },
    };
    
    let result = formatter.render_template(&document, &template, &context).await;
    assert!(result.is_ok());
    
    let rendered = result.unwrap();
    assert_eq!(rendered.format, OutputFormat::Text);
}

// ==================== SECURITY TESTS ====================

#[tokio::test]
async fn test_basic_security_analysis() {
    let config = NeuralDocFlowConfig::default();
    let engine = DocumentEngine::new(config).unwrap();
    
    // Test with safe content
    let safe_content = b"This is safe content";
    let result = engine.process(safe_content.to_vec(), "text/plain").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_security_with_custom_processor() {
    let config = NeuralDocFlowConfig::default();
    let mut engine = DocumentEngine::new(config).unwrap();
    
    let security_processor = Arc::new(RwLock::new(MockSecurityProcessor::new()));
    engine.set_security_processor(security_processor);
    
    let content = b"Test content";
    let result = engine.process(content.to_vec(), "text/plain").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_security_quarantine_action() {
    let config = NeuralDocFlowConfig::default();
    let mut engine = DocumentEngine::new(config).unwrap();
    
    let security_processor = Arc::new(RwLock::new(
        MockSecurityProcessor::new()
            .with_threat_level(ThreatLevel::High)
            .with_action(SecurityAction::Quarantine)
    ));
    engine.set_security_processor(security_processor);
    
    let content = b"Suspicious content";
    let result = engine.process(content.to_vec(), "text/plain").await;
    assert!(result.is_ok());
    
    let document = result.unwrap();
    assert!(document.metadata.custom.contains_key("security_status"));
}

#[tokio::test]
async fn test_security_block_action() {
    let config = NeuralDocFlowConfig::default();
    let mut engine = DocumentEngine::new(config).unwrap();
    
    let security_processor = Arc::new(RwLock::new(
        MockSecurityProcessor::new()
            .with_threat_level(ThreatLevel::Critical)
            .with_action(SecurityAction::Block)
    ));
    engine.set_security_processor(security_processor);
    
    let content = b"Malicious content";
    let result = engine.process(content.to_vec(), "text/plain").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_security_scan_failure() {
    let config = NeuralDocFlowConfig::default();
    let mut engine = DocumentEngine::new(config).unwrap();
    
    let security_processor = Arc::new(RwLock::new(
        MockSecurityProcessor::new().with_failure()
    ));
    engine.set_security_processor(security_processor);
    
    let content = b"Test content";
    let result = engine.process(content.to_vec(), "text/plain").await;
    assert!(result.is_ok()); // Should proceed despite scan failure
}

#[tokio::test]
async fn test_input_validation_file_size() {
    let mut config = NeuralDocFlowConfig::default();
    config.security.policies.max_file_size_mb = 1; // 1MB limit
    
    let engine = DocumentEngine::new(config).unwrap();
    
    let large_content = vec![b'A'; 2 * 1024 * 1024]; // 2MB
    let result = engine.process(large_content, "text/plain").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_input_validation_blocked_file_types() {
    let mut config = NeuralDocFlowConfig::default();
    config.security.policies.blocked_file_types.push("application/x-executable".to_string());
    
    let engine = DocumentEngine::new(config).unwrap();
    
    let content = b"Test content";
    let result = engine.process(content.to_vec(), "application/x-executable").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_input_validation_allowed_file_types() {
    let mut config = NeuralDocFlowConfig::default();
    config.security.policies.allowed_file_types = vec!["text/plain".to_string()];
    
    let engine = DocumentEngine::new(config).unwrap();
    
    // Allowed type should work
    let content = b"Test content";
    let result = engine.process(content.to_vec(), "text/plain").await;
    assert!(result.is_ok());
    
    // Non-allowed type should fail
    let result = engine.process(content.to_vec(), "application/pdf").await;
    assert!(result.is_err());
}

// ==================== PROCESSOR PIPELINE TESTS ====================

#[tokio::test]
async fn test_default_processor_pipeline() {
    let mut pipeline = DefaultProcessorPipeline::new();
    
    let processor = Box::new(MockProcessor::new("test_processor"));
    pipeline.add_processor(processor);
    
    let document = DocumentBuilder::new()
        .text_content("Test content")
        .build();
    
    let result = pipeline.process(document).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_processor_pipeline_capabilities() {
    let pipeline = DefaultProcessorPipeline::new();
    
    let capabilities = pipeline.capabilities();
    assert!(capabilities.input_formats.contains(&"text".to_string()));
    assert!(capabilities.output_formats.contains(&"json".to_string()));
    assert_eq!(capabilities.max_concurrency, Some(4));
}

#[tokio::test]
async fn test_processor_pipeline_status() {
    let pipeline = DefaultProcessorPipeline::new();
    
    let status = pipeline.status().await;
    // Note: PipelineState doesn't implement PartialEq, so we check individual fields
    assert_eq!(status.active_documents, 0);
    assert_eq!(status.queued_documents, 0);
}

#[tokio::test]
async fn test_processor_pipeline_shutdown() {
    let pipeline = DefaultProcessorPipeline::new();
    
    let result = pipeline.shutdown().await;
    assert!(result.is_ok());
}

// ==================== PERFORMANCE MONITORING TESTS ====================

#[tokio::test]
async fn test_performance_monitor() {
    let monitor = PerformanceMonitor::new();
    
    let doc_id = "test_doc";
    let processing_time = std::time::Duration::from_millis(100);
    let bytes_processed = 1024;
    
    monitor.record_processing(doc_id, processing_time, bytes_processed).await;
    
    // Test that monitoring doesn't fail
    // In a real implementation, we'd verify metrics are recorded
}

// ==================== INTEGRATION TESTS ====================

#[tokio::test]
async fn test_end_to_end_document_processing() {
    let config = NeuralDocFlowConfig::default();
    let engine = DocumentEngine::new(config).unwrap();
    
    engine.initialize().await.unwrap();
    
    let input = "Test document content";
    let result = engine.process_document(input, Some("general_document")).await;
    
    assert!(result.is_ok());
    
    let processed_doc = result.unwrap();
    assert_eq!(processed_doc.schema_used, "general_document");
    assert!(processed_doc.validation_result.is_valid);
    assert!(!processed_doc.formatted_output.is_empty());
}

#[tokio::test]
async fn test_process_document_with_custom_schema() {
    let config = NeuralDocFlowConfig::default();
    let engine = DocumentEngine::new(config).unwrap();
    
    engine.initialize().await.unwrap();
    
    let custom_schema = ExtractionSchema {
        name: "custom_test".to_string(),
        description: "Custom test schema".to_string(),
        version: "1.0.0".to_string(),
        required_fields: vec!["text".to_string()],
        optional_fields: vec!["title".to_string()],
        field_types: HashMap::from([
            ("text".to_string(), "string".to_string()),
            ("title".to_string(), "string".to_string()),
        ]),
        validation_rules: vec![],
        domain_specific: None,
    };
    
    engine.register_schema(custom_schema).await.unwrap();
    
    let input = "Test document content";
    let result = engine.process_document(input, Some("custom_test")).await;
    
    assert!(result.is_ok());
    
    let processed_doc = result.unwrap();
    assert_eq!(processed_doc.schema_used, "custom_test");
}

#[tokio::test]
async fn test_process_with_source_name() {
    let config = NeuralDocFlowConfig::default();
    let engine = DocumentEngine::new(config).unwrap();
    
    let content = b"Test content";
    let result = engine.process_with_source(
        content.to_vec(),
        "text/plain",
        "test_source"
    ).await;
    
    assert!(result.is_ok());
    
    let document = result.unwrap();
    assert_eq!(document.metadata.source, "test_source");
}

// ==================== ERROR HANDLING TESTS ====================

#[tokio::test]
async fn test_error_handling_types() {
    // Test ProcessingError direct enum variants
    let schema_error = ProcessingError::SchemaNotFound("test_schema".to_string());
    assert!(matches!(schema_error, ProcessingError::SchemaNotFound(_)));
    
    let template_error = ProcessingError::TemplateNotFound("test_template".to_string());
    assert!(matches!(template_error, ProcessingError::TemplateNotFound(_)));
    
    let serialization_error = ProcessingError::SerializationError("test_error".to_string());
    assert!(matches!(serialization_error, ProcessingError::SerializationError(_)));
    
    let format_error = ProcessingError::UnsupportedFormat("test_format".to_string());
    assert!(matches!(format_error, ProcessingError::UnsupportedFormat(_)));
    
    // Test validation error
    let validation_error = ProcessingError::ValidationError("validation failed".to_string());
    assert!(matches!(validation_error, ProcessingError::ValidationError(_)));
}

#[tokio::test]
async fn test_threat_level_ordering() {
    // Test ThreatLevel enum
    assert_eq!(ThreatLevel::Safe, ThreatLevel::Safe);
    assert_ne!(ThreatLevel::Safe, ThreatLevel::High);
    
    // Test that we can use ThreatLevel in match statements
    let level = ThreatLevel::Medium;
    match level {
        ThreatLevel::Safe => panic!("Should not be safe"),
        ThreatLevel::Low => panic!("Should not be low"),
        ThreatLevel::Medium => {}, // Expected
        ThreatLevel::High => panic!("Should not be high"),
        ThreatLevel::Critical => panic!("Should not be critical"),
    }
}

#[tokio::test]
async fn test_plugin_info_structure() {
    let info = PluginInfo {
        name: "test_plugin".to_string(),
        version: "1.0.0".to_string(),
        loaded_at: Utc::now(),
    };
    
    assert_eq!(info.name, "test_plugin");
    assert_eq!(info.version, "1.0.0");
    assert!(info.loaded_at <= Utc::now());
}

// ==================== VALIDATION RULE TESTS ====================

#[tokio::test]
async fn test_validation_rule_structure() {
    let rule = ValidationRule {
        field: "test_field".to_string(),
        rule_type: "required".to_string(),
        parameters: HashMap::from([
            ("min_length".to_string(), serde_json::Value::Number(serde_json::Number::from(5))),
        ]),
        error_message: "Field is required".to_string(),
    };
    
    assert_eq!(rule.field, "test_field");
    assert_eq!(rule.rule_type, "required");
    assert_eq!(rule.error_message, "Field is required");
    assert!(rule.parameters.contains_key("min_length"));
}

// ==================== COMPREHENSIVE COVERAGE TESTS ====================

#[tokio::test]
async fn test_security_analysis_structure() {
    let analysis = SecurityAnalysis {
        threat_level: ThreatLevel::Low,
        malware_probability: 0.1,
        threat_categories: vec!["Script Content".to_string()],
        anomaly_score: 0.2,
        behavioral_risks: vec!["JavaScript execution risk".to_string()],
        recommended_action: SecurityAction::Sanitize,
    };
    
    assert_eq!(analysis.threat_level, ThreatLevel::Low);
    assert_eq!(analysis.malware_probability, 0.1);
    assert_eq!(analysis.anomaly_score, 0.2);
    assert_eq!(analysis.threat_categories.len(), 1);
    assert_eq!(analysis.behavioral_risks.len(), 1);
    assert_eq!(analysis.recommended_action, SecurityAction::Sanitize);
}

#[tokio::test]
async fn test_processed_document_structure() {
    let document = DocumentBuilder::new()
        .text_content("Test content")
        .build();
    
    let validation_result = ValidationResult::success();
    let extracted_data = HashMap::from([
        ("text".to_string(), serde_json::Value::String("Test content".to_string())),
    ]);
    
    let processed = ProcessedDocument {
        original_document: document,
        extracted_data,
        validation_result,
        security_analysis: None,
        formatted_output: "formatted content".to_string(),
        processing_time: std::time::Duration::from_millis(100),
        schema_used: "test_schema".to_string(),
    };
    
    assert_eq!(processed.schema_used, "test_schema");
    assert_eq!(processed.formatted_output, "formatted content");
    assert_eq!(processed.processing_time, std::time::Duration::from_millis(100));
    assert!(processed.security_analysis.is_none());
}

#[tokio::test]
async fn test_basic_security_analysis_with_scripts() {
    let mut config = NeuralDocFlowConfig::default();
    config.security.enabled = true;
    config.security.scan_mode = SecurityScanMode::Standard;
    
    let engine = DocumentEngine::new(config).unwrap();
    
    let script_content = b"<script>alert('test')</script>";
    let result = engine.process(script_content.to_vec(), "text/html").await;
    
    // Should process but with security measures applied
    assert!(result.is_ok());
}

// ==================== ADDITIONAL EDGE CASE TESTS ====================

#[tokio::test]
async fn test_empty_content_processing() {
    let config = NeuralDocFlowConfig::default();
    let engine = DocumentEngine::new(config).unwrap();
    
    let empty_content = b"";
    let result = engine.process(empty_content.to_vec(), "text/plain").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_large_content_processing() {
    let config = NeuralDocFlowConfig::default();
    let engine = DocumentEngine::new(config).unwrap();
    
    let large_content = vec![b'A'; 1024 * 1024]; // 1MB
    let result = engine.process(large_content, "text/plain").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_unicode_content_processing() {
    let config = NeuralDocFlowConfig::default();
    let engine = DocumentEngine::new(config).unwrap();
    
    let unicode_content = "Hello 世界 🌍".as_bytes();
    let result = engine.process(unicode_content.to_vec(), "text/plain").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_binary_content_processing() {
    let config = NeuralDocFlowConfig::default();
    let engine = DocumentEngine::new(config).unwrap();
    
    let binary_content = vec![0x00, 0x01, 0x02, 0x03, 0xFF];
    let result = engine.process(binary_content, "application/octet-stream").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_concurrent_processing() {
    let config = NeuralDocFlowConfig::default();
    let engine = Arc::new(DocumentEngine::new(config).unwrap());
    
    let mut handles = vec![];
    
    for i in 0..10 {
        let engine_clone = engine.clone();
        let handle = tokio::spawn(async move {
            let content = format!("Test content {}", i);
            engine_clone.process(content.into_bytes(), "text/plain").await
        });
        handles.push(handle);
    }
    
    for handle in handles {
        let result = handle.await.unwrap();
        assert!(result.is_ok());
    }
}

#[tokio::test]
async fn test_security_disabled_processing() {
    let mut config = NeuralDocFlowConfig::default();
    config.security.enabled = false;
    
    let engine = DocumentEngine::new(config).unwrap();
    
    let suspicious_content = b"<script>eval('malicious code')</script>";
    let result = engine.process(suspicious_content.to_vec(), "text/html").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_security_scan_mode_disabled() {
    let mut config = NeuralDocFlowConfig::default();
    config.security.enabled = true;
    config.security.scan_mode = SecurityScanMode::Disabled;
    
    let engine = DocumentEngine::new(config).unwrap();
    
    let content = b"Test content";
    let result = engine.process(content.to_vec(), "text/plain").await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_custom_security_scan_mode() {
    let mut config = NeuralDocFlowConfig::default();
    config.security.enabled = true;
    config.security.scan_mode = SecurityScanMode::Custom("custom_mode".to_string());
    
    let engine = DocumentEngine::new(config).unwrap();
    
    let content = b"Test content";
    let result = engine.process(content.to_vec(), "text/plain").await;
    assert!(result.is_ok());
}