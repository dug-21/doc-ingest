# Core Trait Hierarchy for Neural Document Flow Phase 1

## Overview

This document defines the core trait hierarchy for NeuralDocFlow Phase 1, implementing the exact specifications from iteration5 pure-rust-architecture.md. The trait system provides the foundation for the modular, extensible, and type-safe architecture.

## Core Trait Architecture

```
DocumentSource (trait)
├── can_handle() -> bool
├── validate() -> ValidationResult
├── extract() -> ExtractedDocument
└── create_chunks() -> Vec<DocumentChunk>

ProcessorPipeline (trait)
├── process() -> ProcessedDocument
├── add_stage() -> Result<(), PipelineError>
├── execute_stage() -> Result<StageOutput, StageError>
└── get_metrics() -> PipelineMetrics

OutputFormatter (trait)
├── format() -> FormattedOutput
├── supports_format() -> bool
├── validate_template() -> Result<(), FormatError>
└── register_template() -> Result<(), FormatError>

NeuralProcessor (trait)
├── enhance() -> EnhancedContent
├── train() -> Result<(), NeuralError>
├── load_model() -> Result<(), NeuralError>
└── get_confidence() -> f32
```

## Trait Definitions

### 1. DocumentSource Trait

```rust
use async_trait::async_trait;
use std::path::Path;

/// Core trait that all document sources must implement
#[async_trait]
pub trait DocumentSource: Send + Sync + 'static {
    /// Unique identifier for this source
    fn id(&self) -> &str;
    
    /// Supported file extensions
    fn supported_extensions(&self) -> Vec<&str>;
    
    /// Supported MIME types
    fn supported_mime_types(&self) -> Vec<&str>;
    
    /// Check if this source can handle the given input
    async fn can_handle(&self, input: &DocumentInput) -> Result<bool, SourceError>;
    
    /// Validate input before processing
    async fn validate(&self, input: &DocumentInput) -> Result<ValidationResult, SourceError>;
    
    /// Extract raw content from document
    async fn extract(&self, chunk: &DocumentChunk) -> Result<RawContent, SourceError>;
    
    /// Create processing chunks for parallel extraction
    fn create_chunks(
        &self,
        input: &DocumentInput,
        chunk_size: usize,
    ) -> Result<Vec<DocumentChunk>, SourceError>;
    
    /// Get source capabilities
    fn capabilities(&self) -> SourceCapabilities;
    
    /// Initialize source with configuration
    async fn initialize(&mut self, config: SourceConfig) -> Result<(), SourceError>;
    
    /// Cleanup resources
    async fn cleanup(&mut self) -> Result<(), SourceError>;
}

/// Document input types
#[derive(Debug, Clone)]
pub enum DocumentInput {
    /// File path input
    File(PathBuf),
    /// Memory buffer input
    Memory(Vec<u8>),
    /// Stream input with metadata
    Stream {
        data: Box<dyn AsyncRead + Send + Unpin>,
        size_hint: Option<usize>,
        mime_type: Option<String>,
    },
    /// URL input
    Url {
        url: String,
        headers: Option<HashMap<String, String>>,
    },
}

/// Document chunk for parallel processing
#[derive(Debug, Clone)]
pub struct DocumentChunk {
    pub id: String,
    pub source_id: String,
    pub start_page: Option<usize>,
    pub end_page: Option<usize>,
    pub data: Vec<u8>,
    pub metadata: ChunkMetadata,
}

/// Raw extracted content
#[derive(Debug, Clone)]
pub struct RawContent {
    pub text_blocks: Vec<TextBlock>,
    pub potential_tables: Option<Vec<TableRegion>>,
    pub images: Vec<ImageData>,
    pub metadata: ContentMetadata,
    pub structure_hints: Vec<StructureHint>,
}

/// Source capabilities
#[derive(Debug, Clone)]
pub struct SourceCapabilities {
    pub supports_streaming: bool,
    pub supports_tables: bool,
    pub supports_images: bool,
    pub supports_ocr: bool,
    pub max_file_size: Option<usize>,
    pub parallel_processing: bool,
}
```

### 2. ProcessorPipeline Trait

```rust
/// Core processing pipeline trait
#[async_trait]
pub trait ProcessorPipeline: Send + Sync {
    /// Process document through the pipeline
    async fn process(
        &self,
        input: DocumentInput,
        schema: ExtractionSchema,
        output_format: OutputFormat,
    ) -> Result<ProcessedDocument, PipelineError>;
    
    /// Add a processing stage
    fn add_stage(&mut self, stage: Box<dyn ProcessingStage>) -> Result<(), PipelineError>;
    
    /// Execute a specific stage
    async fn execute_stage(
        &self,
        stage_id: &str,
        input: StageInput,
    ) -> Result<StageOutput, StageError>;
    
    /// Get pipeline metrics
    fn get_metrics(&self) -> PipelineMetrics;
    
    /// Configure pipeline settings
    fn configure(&mut self, config: PipelineConfig) -> Result<(), PipelineError>;
    
    /// Get pipeline stages
    fn stages(&self) -> Vec<&dyn ProcessingStage>;
}

/// Individual processing stage
#[async_trait]
pub trait ProcessingStage: Send + Sync {
    /// Stage identifier
    fn id(&self) -> &str;
    
    /// Stage name
    fn name(&self) -> &str;
    
    /// Process input and return output
    async fn process(&self, input: StageInput) -> Result<StageOutput, StageError>;
    
    /// Stage dependencies
    fn dependencies(&self) -> Vec<&str>;
    
    /// Stage capabilities
    fn capabilities(&self) -> StageCapabilities;
    
    /// Initialize stage
    async fn initialize(&mut self, config: StageConfig) -> Result<(), StageError>;
}

/// Stage input/output types
#[derive(Debug)]
pub enum StageInput {
    RawContent(RawContent),
    EnhancedContent(EnhancedContent),
    ValidatedContent(ValidatedContent),
    FormattedContent(FormattedOutput),
}

#[derive(Debug)]
pub enum StageOutput {
    RawContent(RawContent),
    EnhancedContent(EnhancedContent),
    ValidatedContent(ValidatedContent),
    FormattedContent(FormattedOutput),
}

/// Pipeline metrics
#[derive(Debug, Clone)]
pub struct PipelineMetrics {
    pub total_processing_time: Duration,
    pub stage_times: HashMap<String, Duration>,
    pub memory_usage: usize,
    pub documents_processed: usize,
    pub error_count: usize,
    pub throughput: f64, // documents per second
}
```

### 3. OutputFormatter Trait

```rust
/// Output formatting trait
pub trait OutputFormatter: Send + Sync {
    /// Format identifier
    fn format_id(&self) -> &str;
    
    /// Format name
    fn format_name(&self) -> &str;
    
    /// Supported output formats
    fn supported_formats(&self) -> Vec<&str>;
    
    /// Format validated content
    fn format(
        &self,
        content: &ValidatedContent,
        template: &OutputTemplate,
    ) -> Result<FormattedOutput, FormatError>;
    
    /// Check if format is supported
    fn supports_format(&self, format: &str) -> bool;
    
    /// Validate output template
    fn validate_template(&self, template: &OutputTemplate) -> Result<(), FormatError>;
    
    /// Register custom template
    fn register_template(
        &mut self,
        name: String,
        template: OutputTemplate,
    ) -> Result<(), FormatError>;
    
    /// Get available templates
    fn available_templates(&self) -> Vec<&str>;
    
    /// Get formatter capabilities
    fn capabilities(&self) -> FormatterCapabilities;
}

/// Output template definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputTemplate {
    pub name: String,
    pub format: String,
    pub structure: serde_json::Value,
    pub transformations: Vec<OutputTransformation>,
    pub validation_rules: Vec<ValidationRule>,
}

/// Formatted output
#[derive(Debug, Clone)]
pub struct FormattedOutput {
    pub format: String,
    pub content: String,
    pub metadata: OutputMetadata,
    pub size: usize,
    pub encoding: String,
}

/// Formatter capabilities
#[derive(Debug, Clone)]
pub struct FormatterCapabilities {
    pub supports_streaming: bool,
    pub supports_templating: bool,
    pub supports_validation: bool,
    pub max_output_size: Option<usize>,
    pub custom_transformations: bool,
}
```

### 4. NeuralProcessor Trait

```rust
/// Neural processing trait using ruv-FANN
#[async_trait]
pub trait NeuralProcessor: Send + Sync {
    /// Enhance extracted content using neural networks
    async fn enhance(&self, content: RawContent) -> Result<EnhancedContent, NeuralError>;
    
    /// Train neural networks on new data
    async fn train(&mut self, training_data: TrainingData) -> Result<(), NeuralError>;
    
    /// Load pre-trained model
    async fn load_model(&mut self, model_path: &Path) -> Result<(), NeuralError>;
    
    /// Save trained model
    async fn save_model(&self, model_path: &Path) -> Result<(), NeuralError>;
    
    /// Get confidence score for processed content
    fn get_confidence(&self, content: &EnhancedContent) -> f32;
    
    /// Get processor capabilities
    fn capabilities(&self) -> NeuralCapabilities;
    
    /// Configure neural processor
    fn configure(&mut self, config: NeuralConfig) -> Result<(), NeuralError>;
    
    /// Get processing metrics
    fn metrics(&self) -> NeuralMetrics;
}

/// Enhanced content with neural processing
#[derive(Debug, Clone)]
pub struct EnhancedContent {
    pub layout: LayoutAnalysis,
    pub text: Vec<EnhancedTextBlock>,
    pub tables: Vec<EnhancedTable>,
    pub images: Vec<EnhancedImage>,
    pub confidence: f32,
    pub processing_metadata: NeuralMetadata,
}

/// Neural processing capabilities
#[derive(Debug, Clone)]
pub struct NeuralCapabilities {
    pub layout_analysis: bool,
    pub text_enhancement: bool,
    pub table_detection: bool,
    pub image_processing: bool,
    pub quality_assessment: bool,
    pub supports_training: bool,
    pub simd_acceleration: bool,
}

/// Training data for neural networks
#[derive(Debug)]
pub struct TrainingData {
    pub layout: Vec<LayoutSample>,
    pub text: Vec<TextSample>,
    pub table: Vec<TableSample>,
    pub image: Vec<ImageSample>,
}
```

## Supporting Traits

### 5. Agent Coordination Traits (DAA Integration)

```rust
/// Agent trait for DAA coordination
#[async_trait]
pub trait Agent: Send + Sync {
    /// Agent identifier
    fn id(&self) -> &AgentId;
    
    /// Agent type
    fn agent_type(&self) -> AgentType;
    
    /// Receive and process messages
    async fn receive(&mut self, msg: Message) -> Result<(), AgentError>;
    
    /// Send message to another agent
    async fn send_to(&self, target: &AgentId, msg: Message) -> Result<(), AgentError>;
    
    /// Get agent capabilities
    fn capabilities(&self) -> AgentCapabilities;
    
    /// Initialize agent
    async fn initialize(&mut self, config: AgentConfig) -> Result<(), AgentError>;
    
    /// Shutdown agent
    async fn shutdown(&mut self) -> Result<(), AgentError>;
}

/// Agent types for document processing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentType {
    Controller,
    Extractor,
    Validator,
    Enhancer,
    Formatter,
    Monitor,
}

/// Message types for agent communication
#[derive(Debug, Clone)]
pub enum Message {
    TaskRequest(ExtractionTask),
    TaskResponse(ExtractionResult),
    StatusUpdate(AgentStatus),
    ConfigUpdate(AgentConfig),
    Shutdown,
    Custom(serde_json::Value),
}
```

### 6. Configuration Traits

```rust
/// Configuration trait for all components
pub trait Configurable: Send + Sync {
    /// Configuration type
    type Config: Clone + Send + Sync;
    
    /// Apply configuration
    fn configure(&mut self, config: Self::Config) -> Result<(), ConfigError>;
    
    /// Get current configuration
    fn current_config(&self) -> Self::Config;
    
    /// Validate configuration
    fn validate_config(config: &Self::Config) -> Result<(), ConfigError>;
    
    /// Get configuration schema
    fn config_schema() -> serde_json::Value;
}

/// Configuration validation trait
pub trait ConfigValidator {
    /// Validate configuration
    fn validate(&self, config: &serde_json::Value) -> Result<(), ValidationError>;
    
    /// Get validation rules
    fn rules(&self) -> Vec<ValidationRule>;
}
```

### 7. Error Handling Traits

```rust
/// Error context trait
pub trait ErrorContext {
    /// Add context to error
    fn add_context(self, context: &str) -> Self;
    
    /// Get error chain
    fn error_chain(&self) -> Vec<&dyn std::error::Error>;
    
    /// Error severity
    fn severity(&self) -> ErrorSeverity;
}

/// Error severity levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorSeverity {
    Low,
    Medium,
    High,
    Critical,
}
```

## Trait Implementations

### Built-in Implementations

#### PDF Source Implementation

```rust
/// PDF document source implementation
pub struct PdfSource {
    config: PdfConfig,
    parser: PdfParser,
    ocr_engine: Option<OcrEngine>,
    capabilities: SourceCapabilities,
}

#[async_trait]
impl DocumentSource for PdfSource {
    fn id(&self) -> &str {
        "pdf"
    }
    
    fn supported_extensions(&self) -> Vec<&str> {
        vec!["pdf", "PDF"]
    }
    
    fn supported_mime_types(&self) -> Vec<&str> {
        vec!["application/pdf", "application/x-pdf"]
    }
    
    async fn can_handle(&self, input: &DocumentInput) -> Result<bool, SourceError> {
        match input {
            DocumentInput::File(path) => {
                // Check extension and magic bytes
                let ext_ok = path.extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| self.supported_extensions().contains(&ext))
                    .unwrap_or(false);
                
                if ext_ok {
                    return Ok(true);
                }
                
                // Check magic bytes
                let mut file = tokio::fs::File::open(path).await?;
                let mut header = [0u8; 5];
                if file.read_exact(&mut header).await.is_ok() {
                    Ok(&header == b"%PDF-")
                } else {
                    Ok(false)
                }
            }
            DocumentInput::Memory(data) => {
                Ok(data.starts_with(b"%PDF-"))
            }
            _ => Ok(false),
        }
    }
    
    async fn validate(&self, input: &DocumentInput) -> Result<ValidationResult, SourceError> {
        let mut result = ValidationResult::valid();
        
        // Size validation
        let size = self.get_input_size(input).await?;
        if size > self.config.max_file_size {
            result.add_error("File size exceeds maximum limit");
        }
        
        // Structure validation
        if !self.validate_pdf_structure(input).await? {
            result.add_error("Invalid PDF structure");
        }
        
        // Security validation
        if self.config.security_checks_enabled {
            let security_result = self.validate_security(input).await?;
            if !security_result.is_safe {
                result.add_error("Security validation failed");
                result.security_issues = security_result.issues;
            }
        }
        
        Ok(result)
    }
    
    async fn extract(&self, chunk: &DocumentChunk) -> Result<RawContent, SourceError> {
        // Implementation details...
        todo!("Implement PDF extraction")
    }
    
    fn create_chunks(
        &self,
        input: &DocumentInput,
        chunk_size: usize,
    ) -> Result<Vec<DocumentChunk>, SourceError> {
        // Implementation details...
        todo!("Implement PDF chunking")
    }
    
    fn capabilities(&self) -> SourceCapabilities {
        self.capabilities.clone()
    }
    
    async fn initialize(&mut self, config: SourceConfig) -> Result<(), SourceError> {
        self.config = serde_json::from_value(config.settings)?;
        
        if self.config.ocr_enabled {
            self.ocr_engine = Some(OcrEngine::new(&self.config.ocr_config)?);
        }
        
        Ok(())
    }
    
    async fn cleanup(&mut self) -> Result<(), SourceError> {
        if let Some(ocr) = &mut self.ocr_engine {
            ocr.cleanup().await?;
        }
        Ok(())
    }
}
```

#### Default Pipeline Implementation

```rust
/// Default document processing pipeline
pub struct DefaultPipeline {
    stages: Vec<Box<dyn ProcessingStage>>,
    config: PipelineConfig,
    metrics: PipelineMetrics,
}

#[async_trait]
impl ProcessorPipeline for DefaultPipeline {
    async fn process(
        &self,
        input: DocumentInput,
        schema: ExtractionSchema,
        output_format: OutputFormat,
    ) -> Result<ProcessedDocument, PipelineError> {
        let start_time = Instant::now();
        let mut current_input = StageInput::RawContent(RawContent::default());
        
        // Execute stages in sequence
        for stage in &self.stages {
            let stage_start = Instant::now();
            
            let output = stage.process(current_input).await?;
            
            // Update metrics
            let stage_time = stage_start.elapsed();
            self.record_stage_time(stage.id(), stage_time);
            
            // Prepare input for next stage
            current_input = self.convert_output_to_input(output)?;
        }
        
        // Build final processed document
        let total_time = start_time.elapsed();
        Ok(ProcessedDocument {
            id: Uuid::new_v4().to_string(),
            content: self.extract_final_content(current_input)?,
            metadata: DocumentMetadata::default(),
            processing_time: total_time,
            confidence: self.calculate_overall_confidence(),
        })
    }
    
    fn add_stage(&mut self, stage: Box<dyn ProcessingStage>) -> Result<(), PipelineError> {
        // Validate stage dependencies
        for dep in stage.dependencies() {
            if !self.has_stage(dep) {
                return Err(PipelineError::MissingDependency(dep.to_string()));
            }
        }
        
        self.stages.push(stage);
        Ok(())
    }
    
    async fn execute_stage(
        &self,
        stage_id: &str,
        input: StageInput,
    ) -> Result<StageOutput, StageError> {
        let stage = self.find_stage(stage_id)
            .ok_or_else(|| StageError::StageNotFound(stage_id.to_string()))?;
        
        stage.process(input).await
    }
    
    fn get_metrics(&self) -> PipelineMetrics {
        self.metrics.clone()
    }
    
    fn configure(&mut self, config: PipelineConfig) -> Result<(), PipelineError> {
        self.config = config;
        Ok(())
    }
    
    fn stages(&self) -> Vec<&dyn ProcessingStage> {
        self.stages.iter().map(|s| s.as_ref()).collect()
    }
}
```

## Trait Object Safety

### Design Considerations

1. **Object Safety**: All core traits are designed to be object-safe
2. **Dynamic Dispatch**: Enable runtime polymorphism where needed
3. **Static Dispatch**: Use generics for performance-critical paths
4. **Send + Sync**: All traits are thread-safe for async contexts

### Type Erasure Patterns

```rust
/// Type-erased document source
pub type DynDocumentSource = Box<dyn DocumentSource>;

/// Type-erased processor pipeline
pub type DynProcessorPipeline = Box<dyn ProcessorPipeline>;

/// Type-erased output formatter
pub type DynOutputFormatter = Box<dyn OutputFormatter>;

/// Type-erased neural processor
pub type DynNeuralProcessor = Box<dyn NeuralProcessor>;
```

## Extension Points

### Custom Trait Implementations

1. **Source Plugins**: Implement `DocumentSource` for new formats
2. **Processing Stages**: Implement `ProcessingStage` for custom logic
3. **Output Formats**: Implement `OutputFormatter` for new output types
4. **Neural Models**: Implement `NeuralProcessor` for custom ML models

### Composition Patterns

```rust
/// Composed document processor
pub struct ComposedProcessor {
    source: DynDocumentSource,
    pipeline: DynProcessorPipeline,
    neural: DynNeuralProcessor,
    formatter: DynOutputFormatter,
}

impl ComposedProcessor {
    pub async fn process_document(
        &self,
        input: DocumentInput,
        schema: ExtractionSchema,
        output_format: OutputFormat,
    ) -> Result<ProcessedDocument, ProcessingError> {
        // Validate input
        let validation = self.source.validate(&input).await?;
        if !validation.is_valid() {
            return Err(ProcessingError::ValidationFailed(validation.errors));
        }
        
        // Process through pipeline
        let result = self.pipeline.process(input, schema, output_format).await?;
        
        Ok(result)
    }
}
```

## Testing Infrastructure

### Trait Testing

```rust
/// Test helpers for trait implementations
pub mod test_helpers {
    use super::*;
    
    /// Mock document source for testing
    pub struct MockDocumentSource {
        pub responses: HashMap<String, RawContent>,
    }
    
    #[async_trait]
    impl DocumentSource for MockDocumentSource {
        // Mock implementation...
    }
    
    /// Test pipeline builder
    pub struct TestPipelineBuilder {
        stages: Vec<Box<dyn ProcessingStage>>,
    }
    
    impl TestPipelineBuilder {
        pub fn with_mock_stages(self) -> Self {
            // Add mock stages
            self
        }
        
        pub fn build(self) -> DefaultPipeline {
            // Build test pipeline
            todo!()
        }
    }
}
```

This trait hierarchy provides the foundation for NeuralDocFlow Phase 1, enabling type-safe, extensible, and efficient document processing while maintaining the architectural principles specified in iteration5.