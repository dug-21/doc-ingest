# Phase Interface Contracts

## Overview

This document defines the interfaces between phases to ensure clean separation and testability. Each phase exposes well-defined APIs that subsequent phases build upon.

## Phase 1 → Phase 2 Interface

### Core Types Exposed by Phase 1

```rust
// From neuralflow-core
pub struct ProcessedDocument {
    pub pages: Vec<ProcessedPage>,
    pub metadata: DocumentMetadata,
    pub processing_time: Duration,
}

pub struct ProcessedPage {
    pub text: ExtractedText,
    pub layout: PageLayout,
    pub page_number: usize,
}

pub trait DocumentProcessor: Send + Sync {
    fn process_document(&self, pdf_bytes: &[u8]) -> Result<ProcessedDocument>;
    fn process_page(&self, page_data: &[u8]) -> Result<ProcessedPage>;
}
```

### How Phase 2 Uses Phase 1

```rust
use neuralflow_core::{DocumentProcessor, ProcessedDocument};

// Phase 2 wraps Phase 1 functionality
pub struct DistributedProcessor {
    // Each agent has a Phase 1 processor
    local_processor: Box<dyn DocumentProcessor>,
    coordinator: SwarmCoordinator,
}

impl DistributedProcessor {
    pub async fn process_distributed(&self, pdf_bytes: &[u8]) -> Result<ProcessedDocument> {
        // Use swarm to distribute work
        let chunks = self.split_document(pdf_bytes)?;
        
        // Each agent uses Phase 1 processor
        let tasks: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                let processor = self.local_processor.clone();
                async move { processor.process_page(&chunk) }
            })
            .collect();
        
        // Coordinate results
        let results = self.coordinator.distribute_tasks(tasks).await?;
        self.merge_results(results)
    }
}
```

## Phase 2 → Phase 3 Interface

### Swarm Types for Neural Integration

```rust
// From neuralflow-swarm
pub trait SwarmCoordinator: Send + Sync {
    async fn spawn_agent(&mut self, agent_type: AgentType) -> Result<AgentId>;
    async fn distribute_task(&mut self, task: ProcessingTask) -> Result<TaskHandle>;
    async fn collect_results(&mut self, handles: Vec<TaskHandle>) -> Result<Vec<ProcessingResult>>;
}

pub struct ProcessingResult {
    pub agent_id: AgentId,
    pub task_id: TaskId,
    pub data: Vec<u8>,
    pub metrics: ProcessingMetrics,
}
```

### How Phase 3 Enhances Phase 2

```rust
use neuralflow_swarm::{SwarmCoordinator, AgentType};
use neuralflow_neural::{NeuralEngine, TrainingData};

// Phase 3 adds neural capabilities to agents
pub struct NeuralAgent {
    base_agent: Agent,
    neural_engine: NeuralEngine,
}

impl NeuralAgent {
    pub async fn process_with_intelligence(&mut self, task: ProcessingTask) -> Result<ProcessingResult> {
        // Use neural engine to analyze task
        let classification = self.neural_engine.classify_task(&task)?;
        
        // Process based on classification
        let result = match classification {
            TaskClass::Simple => self.base_agent.process_simple(task).await?,
            TaskClass::Complex => self.process_with_neural_assistance(task).await?,
        };
        
        // Learn from result
        self.neural_engine.train_online(&task, &result)?;
        
        Ok(result)
    }
}
```

## Phase 3 → Phase 4 Interface

### Neural Types for Transformer Integration

```rust
// From neuralflow-neural
pub trait NeuralEngine: Send + Sync {
    fn train_model(&mut self, data: &TrainingData) -> Result<ModelId>;
    fn predict(&self, input: &[f32]) -> Result<Prediction>;
    fn extract_features(&self, data: &[u8]) -> Result<FeatureVector>;
}

pub struct FeatureVector {
    pub values: Vec<f32>,
    pub dimensions: usize,
}
```

### How Phase 4 Builds on Phase 3

```rust
use neuralflow_neural::{NeuralEngine, FeatureVector};
use neuralflow_transformers::{TransformerModel, Embedding};

// Phase 4 combines neural features with transformers
pub struct IntelligentProcessor {
    neural_engine: Box<dyn NeuralEngine>,
    transformer: TransformerModel,
}

impl IntelligentProcessor {
    pub async fn analyze_document(&self, doc: &ProcessedDocument) -> Result<DocumentAnalysis> {
        // Extract features using Phase 3 neural engine
        let features = self.neural_engine.extract_features(&doc.to_bytes())?;
        
        // Generate embeddings using transformer
        let embeddings = self.transformer.encode(&doc.text)?;
        
        // Combine neural and transformer insights
        self.combine_intelligence(features, embeddings)
    }
}
```

## Phase 4 → Phase 5 Interface

### Intelligence Types for SEC Specialization

```rust
// From neuralflow-intelligence
pub trait DocumentIntelligence: Send + Sync {
    async fn analyze(&self, doc: &ProcessedDocument) -> Result<DocumentAnalysis>;
    async fn extract_entities(&self, text: &str) -> Result<Vec<Entity>>;
    async fn generate_summary(&self, content: &str) -> Result<Summary>;
}

pub struct DocumentAnalysis {
    pub document_type: DocumentType,
    pub entities: Vec<Entity>,
    pub key_topics: Vec<Topic>,
    pub sentiment: SentimentScore,
}
```

### How Phase 5 Specializes Phase 4

```rust
use neuralflow_intelligence::{DocumentIntelligence, DocumentAnalysis};

// Phase 5 adds SEC-specific intelligence
pub struct SECIntelligence {
    base_intelligence: Box<dyn DocumentIntelligence>,
    sec_models: SECModels,
}

impl SECIntelligence {
    pub async fn analyze_filing(&self, doc: &ProcessedDocument) -> Result<SECFiling> {
        // Use base intelligence for general analysis
        let analysis = self.base_intelligence.analyze(doc).await?;
        
        // Apply SEC-specific processing
        let filing_type = self.classify_filing_type(&analysis)?;
        let financial_data = self.extract_financial_data(doc, &analysis)?;
        let risk_factors = self.extract_risk_factors(&analysis)?;
        
        Ok(SECFiling {
            filing_type,
            financial_data,
            risk_factors,
            base_analysis: analysis,
        })
    }
}
```

## Phase 5 → Phase 6 Interface

### SEC Types for API Exposure

```rust
// From neuralflow-sec
pub trait SECProcessor: Send + Sync {
    async fn process_filing(&self, pdf_bytes: &[u8]) -> Result<SECFiling>;
    async fn extract_xbrl(&self, filing: &SECFiling) -> Result<XBRLDocument>;
    async fn validate_filing(&self, filing: &SECFiling) -> Result<ValidationReport>;
}

pub struct SECFiling {
    pub filing_type: FilingType,
    pub financial_tables: Vec<FinancialTable>,
    pub risk_factors: Vec<RiskFactor>,
    pub metadata: FilingMetadata,
}
```

### How Phase 6 Exposes All Functionality

```rust
use pyo3::prelude::*;
use neuralflow_sec::SECProcessor;

// Phase 6 creates Python bindings
#[pyclass]
pub struct NeuralDocFlow {
    processor: Box<dyn SECProcessor>,
}

#[pymethods]
impl NeuralDocFlow {
    #[new]
    fn new() -> Self {
        Self {
            processor: create_full_pipeline(),
        }
    }
    
    fn process_sec_filing(&self, pdf_path: &str) -> PyResult<PyObject> {
        let pdf_bytes = std::fs::read(pdf_path)?;
        
        // Use the full pipeline
        let filing = tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(self.processor.process_filing(&pdf_bytes))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        
        // Convert to Python object
        Python::with_gil(|py| {
            filing.to_pyobject(py)
        })
    }
}
```

## Phase 6 → Phase 7 Interface

### API Types for Production Monitoring

```rust
// From neuralflow-api
pub trait APIEndpoint: Send + Sync {
    async fn handle_request(&self, req: Request) -> Result<Response>;
    fn get_metrics(&self) -> EndpointMetrics;
}

pub struct EndpointMetrics {
    pub request_count: u64,
    pub error_count: u64,
    pub average_latency: Duration,
    pub p99_latency: Duration,
}
```

### How Phase 7 Monitors Phase 6

```rust
use neuralflow_api::APIEndpoint;
use opentelemetry::{trace::Tracer, metrics::Meter};

// Phase 7 adds production monitoring
pub struct MonitoredEndpoint<T: APIEndpoint> {
    inner: T,
    tracer: Box<dyn Tracer>,
    meter: Meter,
}

impl<T: APIEndpoint> MonitoredEndpoint<T> {
    pub async fn handle_with_monitoring(&self, req: Request) -> Result<Response> {
        let span = self.tracer.start("handle_request");
        let start = Instant::now();
        
        let result = self.inner.handle_request(req).await;
        
        // Record metrics
        let latency = start.elapsed();
        self.meter.u64_counter("requests_total").add(1);
        self.meter.f64_histogram("request_duration").record(latency.as_secs_f64());
        
        if result.is_err() {
            self.meter.u64_counter("errors_total").add(1);
        }
        
        span.end();
        result
    }
}
```

## Testing Interfaces

### Mock Implementations

Each phase provides mock implementations for testing:

```rust
// Phase 1 mock for Phase 2 testing
pub struct MockDocumentProcessor;

impl DocumentProcessor for MockDocumentProcessor {
    fn process_document(&self, _pdf_bytes: &[u8]) -> Result<ProcessedDocument> {
        Ok(ProcessedDocument {
            pages: vec![test_page()],
            metadata: test_metadata(),
            processing_time: Duration::from_millis(10),
        })
    }
}

// Phase 2 mock for Phase 3 testing
pub struct MockSwarmCoordinator;

impl SwarmCoordinator for MockSwarmCoordinator {
    async fn spawn_agent(&mut self, _agent_type: AgentType) -> Result<AgentId> {
        Ok(AgentId::new_v4())
    }
}
```

### Integration Test Framework

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_phase_1_to_2_integration() {
        // Create Phase 1 processor
        let phase1 = NeuralDocFlow::new();
        
        // Create Phase 2 coordinator with Phase 1
        let mut phase2 = DistributedProcessor::new(Box::new(phase1));
        
        // Test integration
        let result = phase2.process_distributed(TEST_PDF).await.unwrap();
        assert!(!result.pages.is_empty());
    }
    
    #[tokio::test]
    async fn test_full_pipeline_integration() {
        // Build complete pipeline
        let pipeline = PipelineBuilder::new()
            .with_phase1()
            .with_phase2()
            .with_phase3()
            .with_phase4()
            .with_phase5()
            .with_phase6()
            .with_phase7()
            .build();
        
        // Test end-to-end
        let result = pipeline.process_sec_filing(TEST_10K).await.unwrap();
        assert_eq!(result.filing_type, FilingType::Form10K);
    }
}
```

## Interface Versioning

All interfaces use semantic versioning:

```rust
// Version attributes for compatibility
#[version("1.0.0")]
pub trait DocumentProcessor {
    // v1.0.0 methods
    fn process_document(&self, pdf_bytes: &[u8]) -> Result<ProcessedDocument>;
    
    // v1.1.0 additions (with default implementation for backward compatibility)
    #[since("1.1.0")]
    fn process_with_options(&self, pdf_bytes: &[u8], options: ProcessingOptions) -> Result<ProcessedDocument> {
        self.process_document(pdf_bytes) // Default to v1.0.0 behavior
    }
}
```

## Performance Contracts

Each interface defines performance expectations:

```rust
/// Performance contract for DocumentProcessor
/// - Single page processing: < 10ms
/// - 100-page document: < 500ms
/// - Memory usage: < 50MB per document
#[performance_contract(
    single_page_latency = "10ms",
    document_latency = "5ms/page",
    memory_per_document = "50MB"
)]
pub trait DocumentProcessor {
    // ...
}
```