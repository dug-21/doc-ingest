# NeuralDocFlow Modular Architecture Design

## Executive Summary

NeuralDocFlow is designed as a highly modular, MCP-enabled document processing system built in pure Rust. The architecture emphasizes clear module boundaries, self-contained components, and seamless integration with Claude Flow for intelligent orchestration.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         External Interfaces                          │
│  ┌─────────────┬────────────┬─────────────┬────────────────────┐  │
│  │   REST API  │   Python   │    WASM     │    CLI Interface   │  │
│  │   (Axum)    │  (PyO3)    │ (wasm-pack) │      (Clap)       │  │
│  └─────────────┴────────────┴─────────────┴────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                        MCP Server Layer                              │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │          Built-in MCP Server (neuraldocflow-mcp)           │    │
│  │  • Tool Registration  • Resource Management  • JSON-RPC    │    │
│  └────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│                     Orchestration Layer                              │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │         Swarm Coordinator (neuraldocflow-swarm)            │    │
│  │  • Agent Management  • Task Distribution  • Load Balancing │    │
│  └────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│                      Processing Layer                                │
│  ┌─────────────────────┬─────────────────────────────────────┐    │
│  │   PDF Parser        │      Neural Engine                   │    │
│  │ (neuraldocflow-pdf) │   (neuraldocflow-neural)            │    │
│  │ • Zero-copy parse   │   • RUV-FANN integration            │    │
│  │ • SIMD extraction   │   • Pattern recognition             │    │
│  │ • Memory mapping    │   • Entity extraction              │    │
│  └─────────────────────┴─────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│                      Plugin System                                   │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │           Plugin Architecture (neuraldocflow-plugins)       │    │
│  │  • Dynamic loading  • Custom processors  • Format handlers │    │
│  └────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────┤
│                    Foundation Layer                                  │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │         Common Utilities (neuraldocflow-common)            │    │
│  │  • Memory pools  • SIMD helpers  • Error types  • Traits  │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Specifications

### 1. Core PDF Processing Module (`neuraldocflow-pdf`)

**Purpose**: High-performance PDF parsing with zero-copy operations and SIMD acceleration.

**Key Interfaces**:
```rust
pub trait PdfParser: Send + Sync {
    fn parse_mmap(&self, path: &Path) -> Result<Document, PdfError>;
    fn parse_stream(&self, reader: impl Read) -> Result<Document, PdfError>;
    fn extract_text_simd(&self, page: &Page) -> Vec<TextBlock>;
}

pub struct Document {
    pub pages: Vec<Page>,
    pub text_blocks: Vec<TextBlock>,
    pub metadata: DocumentMetadata,
}

pub struct TextBlock {
    pub id: String,
    pub content: String,
    pub bbox: BoundingBox,
    pub page_num: usize,
    pub style: Option<TextStyle>,
}
```

**Dependencies**:
- `lopdf`: Core PDF parsing
- `memmap2`: Memory-mapped file I/O
- `rayon`: Parallel processing
- `simd-adler32`: SIMD checksums

**Testing Strategy**:
- Unit tests for each extraction method
- Integration tests with sample PDFs
- Benchmark tests for performance
- Fuzz testing for robustness

### 2. Neural Engine Module (`neuraldocflow-neural`)

**Purpose**: Neural network integration for intelligent document processing.

**Key Interfaces**:
```rust
pub trait NeuralProcessor: Send + Sync {
    async fn process_blocks(&self, blocks: Vec<TextBlock>) -> Result<Vec<EnrichedBlock>>;
    fn generate_embedding(&self, text: &str) -> Result<Embedding>;
    async fn train_on_feedback(&mut self, feedback: TrainingData) -> Result<()>;
}

pub struct EnrichedBlock {
    pub original: TextBlock,
    pub classification: Classification,
    pub confidence: f32,
    pub entities: Vec<Entity>,
    pub embeddings: Embedding,
}

pub struct Entity {
    pub text: String,
    pub entity_type: EntityType,
    pub confidence: f32,
    pub position: TextPosition,
}
```

**Dependencies**:
- `ruv-fann`: Neural network framework
- `onnxruntime`: ONNX model support
- `ndarray`: Tensor operations
- `dashmap`: Concurrent caching

**Integration Points**:
- Direct integration with RUV-FANN
- ONNX model loading and inference
- Embedding cache for performance
- Batch processing support

### 3. Swarm Coordination Module (`neuraldocflow-swarm`)

**Purpose**: Intelligent task distribution and agent coordination.

**Key Interfaces**:
```rust
pub trait SwarmCoordinator: Send + Sync {
    async fn spawn_agent(&self, role: AgentRole) -> Result<AgentId>;
    async fn orchestrate(&self, document: Document) -> Result<ProcessedDocument>;
    async fn get_status(&self) -> SwarmStatus;
    async fn scale(&self, target_size: usize) -> Result<()>;
}

pub enum AgentRole {
    Parser { capabilities: Vec<ParserCapability> },
    NeuralProcessor { model: String },
    Extractor { entity_types: Vec<EntityType> },
    Validator { rules: Vec<ValidationRule> },
    Coordinator,
}

pub struct SwarmStatus {
    pub active_agents: usize,
    pub pending_tasks: usize,
    pub completed_tasks: usize,
    pub average_load: f32,
}
```

**Dependencies**:
- `tokio`: Async runtime
- `claude-flow-mcp`: MCP integration
- `crossbeam`: Lock-free data structures
- `metrics`: Performance monitoring

**Claude Flow Integration**:
```rust
impl SwarmCoordinator {
    pub async fn integrate_claude_flow(&self, mcp_client: &ClaudeFlowClient) {
        // Register with Claude Flow
        mcp_client.register_swarm(self.id).await;
        
        // Setup coordination hooks
        mcp_client.on_task_received(|task| {
            self.queue_task(task)
        });
        
        // Enable monitoring
        mcp_client.enable_monitoring(self.metrics_endpoint).await;
    }
}
```

### 4. MCP Server Module (`neuraldocflow-mcp`)

**Purpose**: Built-in MCP server for direct Claude Flow integration.

**Key Tools Exposed**:
```typescript
// MCP Tool Definitions
tools: [
  {
    name: "parse_document",
    description: "Parse a PDF document with neural enhancement",
    inputSchema: {
      type: "object",
      properties: {
        path: { type: "string" },
        options: {
          type: "object",
          properties: {
            enable_ocr: { type: "boolean" },
            extract_tables: { type: "boolean" },
            neural_enhancement: { type: "boolean" }
          }
        }
      }
    }
  },
  {
    name: "extract_entities",
    description: "Extract entities from document text",
    inputSchema: {
      type: "object",
      properties: {
        document_id: { type: "string" },
        entity_types: { type: "array", items: { type: "string" } }
      }
    }
  },
  {
    name: "train_model",
    description: "Train neural model on document feedback",
    inputSchema: {
      type: "object",
      properties: {
        training_data: { type: "array" },
        model_type: { type: "string" }
      }
    }
  }
]
```

**Implementation**:
```rust
pub struct McpServer {
    processor: Arc<NeuralDocFlow>,
    transport: McpTransport,
}

impl McpServer {
    pub async fn start(&self, addr: SocketAddr) -> Result<()> {
        let app = Router::new()
            .route("/rpc", post(self.handle_rpc))
            .route("/health", get(health_check));
            
        axum::Server::bind(&addr)
            .serve(app.into_make_service())
            .await?;
            
        Ok(())
    }
    
    async fn handle_rpc(&self, Json(req): Json<RpcRequest>) -> Json<RpcResponse> {
        match req.method.as_str() {
            "tools/list" => self.list_tools(),
            "tools/call" => self.call_tool(req.params).await,
            "resources/list" => self.list_resources(),
            "resources/read" => self.read_resource(req.params).await,
            _ => error_response("Method not found"),
        }
    }
}
```

### 5. API Layers Module (`neuraldocflow-api`)

**Purpose**: Multiple API interfaces for different use cases.

**Components**:

#### REST API (Axum)
```rust
pub fn create_rest_api(processor: Arc<NeuralDocFlow>) -> Router {
    Router::new()
        .route("/api/v1/process", post(process_document))
        .route("/api/v1/status/:id", get(get_status))
        .route("/api/v1/extract/:id/entities", get(extract_entities))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
}
```

#### Python Bindings (PyO3)
```rust
#[pymodule]
fn neuraldocflow(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyNeuralDocFlow>()?;
    m.add_function(wrap_pyfunction!(process_document, m)?)?;
    m.add_function(wrap_pyfunction!(extract_entities, m)?)?;
    Ok(())
}

#[pyclass]
struct PyNeuralDocFlow {
    inner: Arc<NeuralDocFlow>,
}

#[pymethods]
impl PyNeuralDocFlow {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self {
            inner: Arc::new(NeuralDocFlow::builder().build()?),
        })
    }
    
    fn process(&self, path: &str) -> PyResult<PyProcessedDocument> {
        Ok(self.inner.process(Path::new(path))?.into())
    }
}
```

#### WASM Interface
```rust
#[wasm_bindgen]
pub struct WasmNeuralDocFlow {
    processor: NeuralDocFlow,
}

#[wasm_bindgen]
impl WasmNeuralDocFlow {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmNeuralDocFlow, JsValue> {
        Ok(Self {
            processor: NeuralDocFlow::builder()
                .with_wasm_compatibility()
                .build()
                .map_err(|e| JsValue::from_str(&e.to_string()))?,
        })
    }
    
    pub async fn process_bytes(&self, data: &[u8]) -> Result<JsValue, JsValue> {
        let result = self.processor.process_bytes(data).await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        Ok(serde_wasm_bindgen::to_value(&result)?)
    }
}
```

### 6. Plugin Architecture Module (`neuraldocflow-plugins`)

**Purpose**: Extensible plugin system for custom processors and formats.

**Plugin Interface**:
```rust
pub trait Plugin: Send + Sync {
    fn name(&self) -> &str;
    fn version(&self) -> &str;
    fn initialize(&mut self, config: PluginConfig) -> Result<()>;
}

pub trait DocumentProcessor: Plugin {
    fn can_process(&self, doc_type: &DocumentType) -> bool;
    fn process(&self, document: Document) -> Result<ProcessedDocument>;
}

pub trait FormatHandler: Plugin {
    fn supported_formats(&self) -> Vec<&str>;
    fn parse(&self, data: &[u8], format: &str) -> Result<Document>;
    fn serialize(&self, doc: &ProcessedDocument, format: &str) -> Result<Vec<u8>>;
}

pub struct PluginManager {
    plugins: HashMap<String, Box<dyn Plugin>>,
    processors: Vec<Box<dyn DocumentProcessor>>,
    handlers: Vec<Box<dyn FormatHandler>>,
}
```

**Dynamic Loading**:
```rust
impl PluginManager {
    pub fn load_plugin(&mut self, path: &Path) -> Result<()> {
        unsafe {
            let lib = Library::new(path)?;
            let plugin_entry: Symbol<fn() -> Box<dyn Plugin>> = 
                lib.get(b"_plugin_create")?;
            
            let plugin = plugin_entry();
            plugin.initialize(self.config.clone())?;
            
            // Register based on plugin type
            if let Some(processor) = plugin.as_any().downcast_ref::<dyn DocumentProcessor>() {
                self.processors.push(Box::new(processor.clone()));
            }
            
            self.plugins.insert(plugin.name().to_string(), plugin);
            Ok(())
        }
    }
}
```

## Development Phases

### Phase 1: Core PDF Processing (Week 1-2)
- Implement basic PDF parsing with lopdf
- Add SIMD text extraction
- Create memory-mapped file handling
- Write comprehensive tests

### Phase 2: Neural Engine Integration (Week 3-4)
- Integrate RUV-FANN
- Implement embedding generation
- Add entity extraction
- Create training pipeline

### Phase 3: Swarm Coordination (Week 5-6)
- Build Tokio-based coordinator
- Implement agent management
- Add task distribution
- Integrate with Claude Flow

### Phase 4: MCP Server (Week 7)
- Implement MCP protocol
- Register tools and resources
- Add JSON-RPC handling
- Test with Claude Flow

### Phase 5: API Layers (Week 8-9)
- Build REST API with Axum
- Create Python bindings
- Compile to WASM
- Add CLI interface

### Phase 6: Plugin System (Week 10)
- Design plugin interface
- Implement dynamic loading
- Create example plugins
- Write plugin documentation

## Testing Strategy

### Unit Testing
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pdf_parsing() {
        let parser = PdfParser::new(Default::default());
        let doc = parser.parse_mmap(Path::new("test.pdf")).unwrap();
        assert_eq!(doc.pages.len(), 10);
    }
    
    #[tokio::test]
    async fn test_neural_processing() {
        let processor = NeuralProcessor::new().await.unwrap();
        let blocks = vec![TextBlock::new("Test content")];
        let enriched = processor.process_blocks(blocks).await.unwrap();
        assert!(!enriched.is_empty());
    }
}
```

### Integration Testing
```rust
#[tokio::test]
async fn test_full_pipeline() {
    let neuraldoc = NeuralDocFlow::builder()
        .with_simd(true)
        .with_swarm_size(4)
        .build()
        .await
        .unwrap();
        
    let result = neuraldoc.process(Path::new("sample.pdf")).await.unwrap();
    assert!(result.entities.len() > 0);
    assert!(result.confidence > 0.8);
}
```

### Performance Benchmarking
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_simd_extraction(c: &mut Criterion) {
    let parser = PdfParser::new(ParserConfig { use_simd: true, ..Default::default() });
    let page = create_test_page();
    
    c.bench_function("simd_text_extraction", |b| {
        b.iter(|| parser.extract_text_simd(black_box(&page)))
    });
}

criterion_group!(benches, benchmark_simd_extraction);
criterion_main!(benches);
```

## Deployment Configuration

### Docker Support
```dockerfile
# Multi-stage build for minimal image
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release --features "mcp-server"

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl3 ca-certificates
COPY --from=builder /app/target/release/neuraldocflow /usr/local/bin/
EXPOSE 8080
CMD ["neuraldocflow", "serve", "--mcp"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neuraldocflow
spec:
  replicas: 3
  selector:
    matchLabels:
      app: neuraldocflow
  template:
    metadata:
      labels:
        app: neuraldocflow
    spec:
      containers:
      - name: neuraldocflow
        image: neuraldocflow:latest
        ports:
        - containerPort: 8080
        env:
        - name: NEURALDOC_SWARM_SIZE
          value: "16"
        - name: NEURALDOC_ENABLE_MCP
          value: "true"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
```

## Performance Optimization

### Memory Management
```rust
pub struct MemoryPool {
    small_buffers: Vec<Vec<u8>>,  // < 1KB
    medium_buffers: Vec<Vec<u8>>, // 1KB - 64KB
    large_buffers: Vec<Vec<u8>>,  // > 64KB
}

impl MemoryPool {
    pub fn acquire(&mut self, size: usize) -> PooledBuffer {
        let buffer = match size {
            0..=1024 => self.small_buffers.pop(),
            1025..=65536 => self.medium_buffers.pop(),
            _ => self.large_buffers.pop(),
        }.unwrap_or_else(|| Vec::with_capacity(size));
        
        PooledBuffer::new(buffer, self)
    }
}
```

### SIMD Operations
```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn find_word_boundaries_simd(text: &[u8]) -> Vec<usize> {
    let mut boundaries = Vec::new();
    let space = _mm256_set1_epi8(b' ' as i8);
    
    for (i, chunk) in text.chunks_exact(32).enumerate() {
        unsafe {
            let data = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
            let cmp = _mm256_cmpeq_epi8(data, space);
            let mask = _mm256_movemask_epi8(cmp);
            
            if mask != 0 {
                for j in 0..32 {
                    if mask & (1 << j) != 0 {
                        boundaries.push(i * 32 + j);
                    }
                }
            }
        }
    }
    
    boundaries
}
```

## Monitoring and Observability

### Metrics Collection
```rust
use prometheus::{Counter, Histogram, Registry};

pub struct Metrics {
    documents_processed: Counter,
    processing_duration: Histogram,
    neural_inference_time: Histogram,
    swarm_task_queue_size: Gauge,
}

impl Metrics {
    pub fn new(registry: &Registry) -> Self {
        Self {
            documents_processed: Counter::new("documents_processed_total", "Total documents processed")
                .expect("metric creation failed"),
            processing_duration: Histogram::with_opts(
                HistogramOpts::new("processing_duration_seconds", "Document processing duration")
            ).expect("metric creation failed"),
            // ... other metrics
        }
    }
}
```

### Distributed Tracing
```rust
use tracing::{info, instrument, span, Level};

#[instrument(skip(self, document))]
pub async fn process_document(&self, document: Document) -> Result<ProcessedDocument> {
    let span = span!(Level::INFO, "process_document", doc_id = %document.id);
    let _enter = span.enter();
    
    info!("Starting document processing");
    
    // Process with tracing
    let parsed = self.parse_with_trace(document).await?;
    let enriched = self.neural_process_with_trace(parsed).await?;
    let result = self.aggregate_with_trace(enriched).await?;
    
    info!("Document processing completed");
    Ok(result)
}
```

## Security Considerations

### Input Validation
```rust
pub fn validate_pdf(data: &[u8]) -> Result<(), ValidationError> {
    // Check PDF header
    if !data.starts_with(b"%PDF-") {
        return Err(ValidationError::InvalidFormat);
    }
    
    // Check file size limits
    if data.len() > MAX_PDF_SIZE {
        return Err(ValidationError::FileTooLarge);
    }
    
    // Scan for malicious patterns
    if contains_javascript(data) && !self.config.allow_javascript {
        return Err(ValidationError::PotentiallyMalicious);
    }
    
    Ok(())
}
```

### Sandboxing
```rust
pub struct Sandbox {
    memory_limit: usize,
    cpu_quota: Duration,
    allowed_syscalls: Vec<Syscall>,
}

impl Sandbox {
    pub fn execute<F, R>(&self, f: F) -> Result<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        // Apply resource limits
        self.apply_limits()?;
        
        // Execute in restricted environment
        let result = std::thread::spawn(f).join()
            .map_err(|_| SandboxError::ExecutionFailed)?;
            
        Ok(result)
    }
}
```

## Conclusion

This modular architecture provides:
1. **Clear separation of concerns** with well-defined module boundaries
2. **Self-contained components** that can be developed and tested independently
3. **Seamless MCP integration** through built-in server
4. **Extensibility** through plugin architecture
5. **Performance optimization** with SIMD and zero-copy operations
6. **Production readiness** with monitoring, security, and deployment support

The iterative development approach allows for progressive enhancement while maintaining a working system at each phase.