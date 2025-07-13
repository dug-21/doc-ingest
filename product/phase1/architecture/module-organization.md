# Module Organization for Neural Document Flow Phase 1

## Overview

This document defines the internal module organization for each crate in NeuralDocFlow Phase 1, implementing the pure Rust architecture with clear separation of concerns, optimal performance, and maintainable code structure.

## Core Module Architecture

```
neuraldocflow-core/
├── src/
│   ├── lib.rs                    # Public API exports
│   ├── engine/                   # Document processing engine
│   │   ├── mod.rs
│   │   ├── document_engine.rs    # Main engine implementation
│   │   ├── config.rs             # Engine configuration
│   │   └── metrics.rs            # Performance metrics
│   ├── types/                    # Common types and data structures
│   │   ├── mod.rs
│   │   ├── document.rs           # Document types
│   │   ├── content.rs            # Content types
│   │   ├── metadata.rs           # Metadata types
│   │   └── errors.rs             # Error types
│   ├── traits/                   # Core trait definitions
│   │   ├── mod.rs
│   │   ├── source.rs             # DocumentSource trait
│   │   ├── pipeline.rs           # ProcessorPipeline trait
│   │   ├── formatter.rs          # OutputFormatter trait
│   │   └── neural.rs             # NeuralProcessor trait
│   ├── validation/               # Input validation
│   │   ├── mod.rs
│   │   ├── validators.rs         # Validation implementations
│   │   └── rules.rs              # Validation rules
│   └── utils/                    # Utility functions
│       ├── mod.rs
│       ├── async_utils.rs        # Async utilities
│       ├── file_utils.rs         # File handling utilities
│       └── mime_detection.rs     # MIME type detection
└── tests/
    ├── integration_tests.rs
    └── fixtures/
```

## Detailed Module Breakdown

### 1. neuraldocflow-core Module Structure

#### lib.rs - Public API

```rust
//! NeuralDocFlow Core Library
//! 
//! This crate provides the core document processing engine with support for
//! multiple document formats, neural enhancement, and configurable output formats.

pub mod engine;
pub mod types;
pub mod traits;
pub mod validation;
pub mod utils;

// Re-exports for public API
pub use engine::{DocumentEngine, EngineConfig};
pub use types::{
    document::{DocumentInput, ProcessedDocument, DocumentMetadata},
    content::{RawContent, EnhancedContent, ContentBlock},
    errors::{ProcessingError, SourceError, NeuralError},
};
pub use traits::{
    source::DocumentSource,
    pipeline::ProcessorPipeline,
    formatter::OutputFormatter,
    neural::NeuralProcessor,
};

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Default configuration
pub fn default_config() -> EngineConfig {
    EngineConfig::default()
}

/// Initialize logging
pub fn init_logging() {
    env_logger::init();
}
```

#### engine/mod.rs

```rust
//! Document processing engine

pub mod document_engine;
pub mod config;
pub mod metrics;

pub use document_engine::DocumentEngine;
pub use config::{EngineConfig, EngineBuilder};
pub use metrics::{ProcessingMetrics, MetricsCollector};
```

#### engine/document_engine.rs

```rust
//! Main document processing engine implementation

use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

use crate::traits::{DocumentSource, ProcessorPipeline, OutputFormatter, NeuralProcessor};
use crate::types::{DocumentInput, ProcessedDocument, ProcessingError};
use neuraldocflow_daa::Topology;
use neuraldocflow_sources::SourceManager;
use neuraldocflow_schema::SchemaEngine;
use neuraldocflow_output::OutputEngine;
use neuraldocflow_neural::NeuralProcessorImpl;

/// Main document processing engine
pub struct DocumentEngine {
    /// DAA topology for coordination
    daa_topology: Arc<Topology>,
    
    /// Neural processor
    neural_processor: Arc<dyn NeuralProcessor>,
    
    /// Source manager
    source_manager: Arc<SourceManager>,
    
    /// Schema validation engine
    schema_engine: Arc<SchemaEngine>,
    
    /// Output formatting engine
    output_engine: Arc<OutputEngine>,
    
    /// Engine configuration
    config: EngineConfig,
    
    /// Metrics collector
    metrics: Arc<MetricsCollector>,
}

impl DocumentEngine {
    /// Create new document processing engine
    pub fn new(config: EngineConfig) -> Result<Self, ProcessingError> {
        let daa_topology = Self::build_topology(&config)?;
        let neural_processor = Arc::new(NeuralProcessorImpl::new(&config.neural_config)?);
        let source_manager = Arc::new(SourceManager::new(&config.source_config)?);
        let schema_engine = Arc::new(SchemaEngine::new());
        let output_engine = Arc::new(OutputEngine::new());
        let metrics = Arc::new(MetricsCollector::new());
        
        Ok(Self {
            daa_topology,
            neural_processor,
            source_manager,
            schema_engine,
            output_engine,
            config,
            metrics,
        })
    }
    
    /// Process document with user-defined schema
    pub async fn process(
        &self,
        input: DocumentInput,
        schema: ExtractionSchema,
        output_format: OutputFormat,
    ) -> Result<ProcessedDocument, ProcessingError> {
        let start_time = std::time::Instant::now();
        
        // Validate input
        let source = self.source_manager.get_source_for(&input).await?;
        let validation = source.validate(&input).await?;
        if !validation.is_valid() {
            return Err(ProcessingError::ValidationFailed(validation.errors));
        }
        
        // Create extraction task
        let task = ExtractionTask {
            id: Uuid::new_v4(),
            input,
            schema: schema.clone(),
            chunks: source.create_chunks(&input, self.config.chunk_size)?,
        };
        
        // Process through DAA agents
        let raw_results = self.process_with_daa(task).await?;
        
        // Enhance with neural processing
        let enhanced = self.neural_processor.enhance(raw_results).await?;
        
        // Apply schema validation
        let validated = self.schema_engine.validate(&enhanced, &schema)?;
        
        // Format output
        let formatted = self.output_engine.format(&validated, &output_format)?;
        
        // Record metrics
        let processing_time = start_time.elapsed();
        self.metrics.record_processing(processing_time, &validated);
        
        Ok(ProcessedDocument {
            id: task.id.to_string(),
            content: formatted,
            metadata: self.extract_metadata(&validated),
            confidence: enhanced.confidence,
            processing_time,
        })
    }
    
    /// Process document through DAA agents
    async fn process_with_daa(&self, task: ExtractionTask) -> Result<RawContent, ProcessingError> {
        // Send task to controller agent
        let controller = self.daa_topology.get_controller_agent();
        controller.send_task(task).await?;
        
        // Wait for completion
        let results = controller.await_results().await?;
        
        Ok(results)
    }
    
    /// Build DAA topology for document processing
    fn build_topology(config: &EngineConfig) -> Result<Arc<Topology>, ProcessingError> {
        use neuraldocflow_daa::{TopologyBuilder, AgentType};
        
        let mut builder = TopologyBuilder::new();
        
        // Add controller agent
        builder.add_agent("controller", AgentType::Controller)?;
        
        // Add extractor agents
        for i in 0..config.parallelism {
            builder.add_agent(&format!("extractor_{}", i), AgentType::Extractor)?;
        }
        
        // Add validator agents
        builder.add_agent("validator", AgentType::Validator)?;
        
        // Add enhancer agent
        builder.add_agent("enhancer", AgentType::Enhancer)?;
        
        // Add formatter agent
        builder.add_agent("formatter", AgentType::Formatter)?;
        
        // Configure connections
        builder.connect_star("controller")?;
        
        Ok(Arc::new(builder.build()?))
    }
}
```

### 2. neuraldocflow-daa Module Structure

```
neuraldocflow-daa/
├── src/
│   ├── lib.rs                    # DAA public API
│   ├── agent/                    # Agent implementations
│   │   ├── mod.rs
│   │   ├── base.rs               # Base agent trait
│   │   ├── controller.rs         # Controller agent
│   │   ├── extractor.rs          # Extractor agent
│   │   ├── validator.rs          # Validator agent
│   │   ├── enhancer.rs           # Enhancer agent
│   │   └── formatter.rs          # Formatter agent
│   ├── topology/                 # Topology management
│   │   ├── mod.rs
│   │   ├── builder.rs            # Topology builder
│   │   ├── graph.rs              # Agent graph
│   │   └── routing.rs            # Message routing
│   ├── communication/            # Message passing
│   │   ├── mod.rs
│   │   ├── message.rs            # Message types
│   │   ├── channel.rs            # Communication channels
│   │   └── protocol.rs           # Communication protocol
│   ├── coordination/             # Coordination mechanisms
│   │   ├── mod.rs
│   │   ├── consensus.rs          # Consensus algorithms
│   │   ├── election.rs           # Leader election
│   │   └── synchronization.rs    # State synchronization
│   └── utils/                    # DAA utilities
│       ├── mod.rs
│       ├── id_generator.rs       # Agent ID generation
│       └── metrics.rs            # DAA metrics
└── tests/
    ├── agent_tests.rs
    ├── topology_tests.rs
    └── integration_tests.rs
```

#### daa/lib.rs

```rust
//! Distributed Autonomous Agents for NeuralDocFlow

pub mod agent;
pub mod topology;
pub mod communication;
pub mod coordination;
pub mod utils;

pub use agent::{Agent, AgentType, AgentId, AgentConfig};
pub use topology::{Topology, TopologyBuilder, TopologyType};
pub use communication::{Message, MessageType, CommunicationChannel};
pub use coordination::{ConsensusResult, ElectionResult};

/// DAA system configuration
#[derive(Debug, Clone)]
pub struct DAAConfig {
    pub agent_count: usize,
    pub topology_type: TopologyType,
    pub message_timeout: std::time::Duration,
    pub consensus_threshold: f64,
}

impl Default for DAAConfig {
    fn default() -> Self {
        Self {
            agent_count: 4,
            topology_type: TopologyType::Star,
            message_timeout: std::time::Duration::from_secs(30),
            consensus_threshold: 0.75,
        }
    }
}
```

### 3. neuraldocflow-neural Module Structure

```
neuraldocflow-neural/
├── src/
│   ├── lib.rs                    # Neural processing API
│   ├── processor/                # Neural processor implementation
│   │   ├── mod.rs
│   │   ├── neural_processor.rs   # Main processor
│   │   ├── layout_analyzer.rs    # Layout analysis network
│   │   ├── text_enhancer.rs      # Text enhancement network
│   │   ├── table_detector.rs     # Table detection network
│   │   └── quality_assessor.rs   # Quality assessment network
│   ├── models/                   # Neural network models
│   │   ├── mod.rs
│   │   ├── fann_wrapper.rs       # ruv-FANN wrapper
│   │   ├── model_loader.rs       # Model loading/saving
│   │   └── training.rs           # Training utilities
│   ├── features/                 # Feature extraction
│   │   ├── mod.rs
│   │   ├── text_features.rs      # Text feature extraction
│   │   ├── layout_features.rs    # Layout feature extraction
│   │   ├── image_features.rs     # Image feature extraction
│   │   └── simd_features.rs      # SIMD-accelerated features
│   ├── enhancement/              # Content enhancement
│   │   ├── mod.rs
│   │   ├── text_enhancement.rs   # Text enhancement logic
│   │   ├── layout_analysis.rs    # Layout analysis logic
│   │   └── confidence_scoring.rs # Confidence calculation
│   └── utils/                    # Neural utilities
│       ├── mod.rs
│       ├── data_preprocessing.rs # Data preprocessing
│       └── metrics.rs            # Neural metrics
└── tests/
    ├── processor_tests.rs
    ├── model_tests.rs
    └── benchmark_tests.rs
```

#### neural/lib.rs

```rust
//! Neural processing for NeuralDocFlow using ruv-FANN

pub mod processor;
pub mod models;
pub mod features;
pub mod enhancement;
pub mod utils;

pub use processor::{NeuralProcessorImpl, NeuralConfig};
pub use models::{NeuralModel, ModelType, TrainingData};
pub use enhancement::{EnhancedContent, LayoutAnalysis, TextEnhancement};

/// Neural processing capabilities
#[derive(Debug, Clone)]
pub struct NeuralCapabilities {
    pub layout_analysis: bool,
    pub text_enhancement: bool,
    pub table_detection: bool,
    pub image_processing: bool,
    pub quality_assessment: bool,
    pub simd_acceleration: bool,
}

/// Neural processing configuration
#[derive(Debug, Clone)]
pub struct NeuralConfig {
    pub model_path: std::path::PathBuf,
    pub enable_training: bool,
    pub use_simd: bool,
    pub batch_size: usize,
    pub confidence_threshold: f32,
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            model_path: std::path::PathBuf::from("./models"),
            enable_training: false,
            use_simd: true,
            batch_size: 32,
            confidence_threshold: 0.8,
        }
    }
}
```

### 4. neuraldocflow-sources Module Structure

```
neuraldocflow-sources/
├── src/
│   ├── lib.rs                    # Sources public API
│   ├── manager/                  # Source management
│   │   ├── mod.rs
│   │   ├── source_manager.rs     # Source registry and discovery
│   │   ├── plugin_loader.rs      # Dynamic plugin loading
│   │   └── hot_reload.rs         # Hot-reload functionality
│   ├── sources/                  # Built-in source implementations
│   │   ├── mod.rs
│   │   ├── pdf/                  # PDF source
│   │   │   ├── mod.rs
│   │   │   ├── pdf_source.rs
│   │   │   ├── parser.rs
│   │   │   └── extractor.rs
│   │   ├── docx/                 # DOCX source (future)
│   │   ├── html/                 # HTML source (future)
│   │   ├── image/                # Image source (future)
│   │   └── custom/               # Custom source template
│   ├── validation/               # Input validation
│   │   ├── mod.rs
│   │   ├── security.rs           # Security validation
│   │   ├── format.rs             # Format validation
│   │   └── size.rs               # Size validation
│   ├── config/                   # Source configuration
│   │   ├── mod.rs
│   │   ├── source_config.rs      # Source-specific config
│   │   └── registry.rs           # Configuration registry
│   └── utils/                    # Source utilities
│       ├── mod.rs
│       ├── mime_detection.rs     # MIME type detection
│       └── chunk_creation.rs     # Document chunking
└── tests/
    ├── source_tests.rs
    ├── plugin_tests.rs
    └── integration_tests.rs
```

#### sources/lib.rs

```rust
//! Document source plugins for NeuralDocFlow

pub mod manager;
pub mod sources;
pub mod validation;
pub mod config;
pub mod utils;

pub use manager::{SourceManager, PluginLoader};
pub use sources::pdf::PdfSource;
pub use validation::{ValidationResult, SecurityCheck};
pub use config::{SourceConfig, SourceRegistry};

/// Source plugin interface
pub use neuraldocflow_core::traits::DocumentSource;

/// Source capabilities
#[derive(Debug, Clone)]
pub struct SourceCapabilities {
    pub supported_formats: Vec<String>,
    pub max_file_size: Option<usize>,
    pub supports_streaming: bool,
    pub supports_parallel: bool,
    pub security_features: Vec<String>,
}

/// Source metadata for plugins
#[derive(Debug, Clone)]
pub struct SourceMetadata {
    pub id: String,
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    pub capabilities: SourceCapabilities,
}
```

### 5. neuraldocflow-schema Module Structure

```
neuraldocflow-schema/
├── src/
│   ├── lib.rs                    # Schema engine API
│   ├── engine/                   # Schema processing engine
│   │   ├── mod.rs
│   │   ├── schema_engine.rs      # Main schema engine
│   │   ├── validator.rs          # Schema validation
│   │   └── transformer.rs       # Data transformation
│   ├── schema/                   # Schema definitions
│   │   ├── mod.rs
│   │   ├── extraction_schema.rs  # Extraction schema types
│   │   ├── field_definition.rs   # Field definitions
│   │   └── validation_rules.rs   # Validation rules
│   ├── parsers/                  # Schema parsers
│   │   ├── mod.rs
│   │   ├── json_parser.rs        # JSON schema parser
│   │   ├── yaml_parser.rs        # YAML schema parser
│   │   └── toml_parser.rs        # TOML schema parser
│   ├── extractors/               # Field extractors
│   │   ├── mod.rs
│   │   ├── text_extractor.rs     # Text field extraction
│   │   ├── number_extractor.rs   # Number field extraction
│   │   ├── date_extractor.rs     # Date field extraction
│   │   └── table_extractor.rs    # Table field extraction
│   └── utils/                    # Schema utilities
│       ├── mod.rs
│       ├── json_path.rs          # JSON path utilities
│       └── regex_utils.rs        # Regex utilities
└── tests/
    ├── schema_tests.rs
    ├── validation_tests.rs
    └── extraction_tests.rs
```

### 6. neuraldocflow-output Module Structure

```
neuraldocflow-output/
├── src/
│   ├── lib.rs                    # Output formatting API
│   ├── engine/                   # Output engine
│   │   ├── mod.rs
│   │   ├── output_engine.rs      # Main output engine
│   │   └── template_engine.rs    # Template processing
│   ├── formatters/               # Output formatters
│   │   ├── mod.rs
│   │   ├── json_formatter.rs     # JSON output
│   │   ├── xml_formatter.rs      # XML output
│   │   ├── csv_formatter.rs      # CSV output
│   │   ├── markdown_formatter.rs # Markdown output
│   │   └── html_formatter.rs     # HTML output
│   ├── templates/                # Output templates
│   │   ├── mod.rs
│   │   ├── template_parser.rs    # Template parsing
│   │   ├── template_engine.rs    # Template rendering
│   │   └── builtin_templates.rs  # Built-in templates
│   ├── transformations/          # Data transformations
│   │   ├── mod.rs
│   │   ├── field_mapping.rs      # Field mapping
│   │   ├── data_conversion.rs    # Data type conversion
│   │   └── aggregation.rs        # Data aggregation
│   └── utils/                    # Output utilities
│       ├── mod.rs
│       ├── escaping.rs           # String escaping
│       └── formatting.rs        # Text formatting
└── tests/
    ├── formatter_tests.rs
    ├── template_tests.rs
    └── integration_tests.rs
```

## Cross-Crate Dependencies

### Dependency Graph

```
neuraldocflow-core
├── neuraldocflow-daa
├── neuraldocflow-neural
├── neuraldocflow-sources
├── neuraldocflow-schema
└── neuraldocflow-output

neuraldocflow-daa
└── (no internal dependencies)

neuraldocflow-neural
└── (no internal dependencies)

neuraldocflow-sources
└── neuraldocflow-core (traits only)

neuraldocflow-schema
└── neuraldocflow-core (traits only)

neuraldocflow-output
└── neuraldocflow-core (traits only)
```

### Interface Boundaries

#### Core to DAA Interface

```rust
// In neuraldocflow-core
use neuraldocflow_daa::{Topology, Agent, Message};

impl DocumentEngine {
    fn build_daa_topology(&self) -> Result<Topology, ProcessingError> {
        // Use DAA topology builder
    }
    
    async fn process_with_agents(&self, task: ExtractionTask) -> Result<RawContent, ProcessingError> {
        // Send task to DAA agents
    }
}
```

#### Core to Neural Interface

```rust
// In neuraldocflow-core
use neuraldocflow_neural::{NeuralProcessorImpl, NeuralConfig};

impl DocumentEngine {
    fn create_neural_processor(&self) -> Result<Arc<dyn NeuralProcessor>, ProcessingError> {
        Ok(Arc::new(NeuralProcessorImpl::new(&self.config.neural_config)?))
    }
}
```

### Error Handling Across Modules

#### Unified Error Types

```rust
// In neuraldocflow-core/src/types/errors.rs
#[derive(Debug, thiserror::Error)]
pub enum ProcessingError {
    #[error("Source error: {0}")]
    Source(#[from] neuraldocflow_sources::SourceError),
    
    #[error("Neural processing error: {0}")]
    Neural(#[from] neuraldocflow_neural::NeuralError),
    
    #[error("DAA coordination error: {0}")]
    DAA(#[from] neuraldocflow_daa::AgentError),
    
    #[error("Schema validation error: {0}")]
    Schema(#[from] neuraldocflow_schema::SchemaError),
    
    #[error("Output formatting error: {0}")]
    Output(#[from] neuraldocflow_output::FormatError),
    
    #[error("Configuration error: {0}")]
    Config(String),
    
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
```

## Testing Organization

### Unit Tests per Module

```rust
// In each module
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_module_functionality() {
        // Test module-specific functionality
    }
    
    #[test]
    fn test_synchronous_functions() {
        // Test synchronous functions
    }
}
```

### Integration Tests

```rust
// In tests/ directory
mod integration_tests {
    use neuraldocflow_core::*;
    
    #[tokio::test]
    async fn test_full_pipeline() {
        // Test complete document processing pipeline
    }
    
    #[tokio::test]
    async fn test_error_handling() {
        // Test error propagation across modules
    }
}
```

### Benchmark Tests

```rust
// In benches/ directory
use criterion::{criterion_group, criterion_main, Criterion};

fn benchmark_processing(c: &mut Criterion) {
    c.bench_function("document_processing", |b| {
        b.iter(|| {
            // Benchmark document processing
        })
    });
}

criterion_group!(benches, benchmark_processing);
criterion_main!(benches);
```

## Documentation Organization

### Module Documentation

```rust
//! # NeuralDocFlow Core Engine Module
//! 
//! This module provides the main document processing engine that coordinates
//! all aspects of document extraction, neural enhancement, and output formatting.
//! 
//! ## Usage
//! 
//! ```rust
//! use neuraldocflow_core::{DocumentEngine, EngineConfig};
//! 
//! let config = EngineConfig::default();
//! let engine = DocumentEngine::new(config)?;
//! 
//! let result = engine.process(input, schema, output_format).await?;
//! ```
//! 
//! ## Architecture
//! 
//! The engine coordinates between multiple subsystems:
//! - DAA agents for parallel processing
//! - Neural networks for content enhancement
//! - Source plugins for format support
//! - Schema validation for data quality
//! - Output formatters for flexible output
```

### API Documentation Standards

1. **Module-level docs**: Explain module purpose and usage
2. **Function docs**: Include examples and error conditions
3. **Type docs**: Explain data structures and invariants
4. **Trait docs**: Document trait contracts and implementations

This module organization provides clear separation of concerns, efficient compilation, and maintainable code structure for NeuralDocFlow Phase 1 implementation.