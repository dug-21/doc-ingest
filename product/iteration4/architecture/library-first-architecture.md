# Library-First Multimodal Architecture for NeuralDocFlow

## 🎯 Executive Summary

This architecture document defines NeuralDocFlow as a **library-first multimodal** Rust document processing system that strategically leverages existing libraries (claude-flow@alpha, ruv-FANN, DAA) while innovating where necessary for multimodal extraction capabilities. The system extracts structured data from multiple formats (PDF, DOCX, PPTX, XLSX, Images, HTML) including text, tables, charts, and images. It is designed as a reusable library comparable to pypdf but with advanced multimodal capabilities, written in pure Rust with Python/Web/MCP interfaces, and explicitly NOT dependent on any LLM.

## 🎯 Strategic Library Usage Guidelines

### Core Libraries (MUST Use)
1. **claude-flow@alpha** - Coordination, memory, and swarm operations
2. **ruv-FANN** - Neural network and ML operations from Phase 1
3. **DAA** - Distributed and autonomous agent operations

### Multimodal Processing Libraries
4. **image** - Image processing and format conversion
5. **tesseract-sys** - OCR for text extraction from images
6. **calamine** - Excel/spreadsheet processing
7. **lopdf/pdf-extract** - PDF text and structure extraction
8. **zip** - DOCX/PPTX archive handling
9. **roxmltree** - XML parsing for Office formats
10. **plotters** - Chart analysis and reconstruction

### When Custom Code IS Acceptable
- **Multimodal Integration**: Combining outputs from different extraction libraries
- **Format-Specific Parsers**: When libraries have architectural limitations
- **Innovation Areas**: Novel chart recognition, table structure detection
- **Performance Critical Paths**: SIMD optimizations for image processing
- **Glue Code**: Integrating disparate libraries into cohesive pipeline

### ❌ Still FORBIDDEN
- Reimplementing basic coordination (use claude-flow@alpha)
- Basic neural networks (use ruv-FANN patterns)
- Standard distributed patterns (use DAA)
- Common format parsing where good libraries exist

## 📐 Multimodal Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         External Interfaces                          │
│  ┌─────────────┬─────────────┬─────────────┬─────────────────────┐ │
│  │   Python    │    Web      │     MCP     │       CLI          │ │
│  │  (PyO3)     │   (WASM)    │  (claude)   │     (Rust)         │ │
│  └─────────────┴─────────────┴─────────────┴─────────────────────┘ │
├─────────────────────────────────────────────────────────────────────┤
│                    NeuralDocFlow Multimodal Core                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  Multimodal Processing Pipeline              │   │
│  │  ┌──────────┬──────────┬──────────┬──────────┬──────────┐  │   │
│  │  │   Text   │  Tables  │  Images  │  Charts  │ Metadata │  │   │
│  │  │Extraction│Detection │Analysis  │Recognition│ Parsing │  │   │
│  │  └──────────┴──────────┴──────────┴──────────┴──────────┘  │   │
│  │                                                              │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │           Unified Multimodal Representation         │    │   │
│  │  │    (Combines all extracted modalities + Phase 1     │    │   │
│  │  │     neural capabilities for understanding)          │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                    Format-Specific Extractors                        │
│  ┌────────┬────────┬────────┬────────┬────────┬────────────────┐   │
│  │  PDF   │ DOCX   │ PPTX   │ XLSX   │Images  │     HTML       │   │
│  │lopdf   │  zip   │  zip   │calamine│ image  │ scraper/kuchiki│   │
│  │pdf-    │roxmltree│roxmltree│        │tesseract│               │   │
│  │extract │        │        │        │        │                │   │
│  └────────┴────────┴────────┴────────┴────────┴────────────────┘   │
├─────────────────────────────────────────────────────────────────────┤
│                 Core Library Integration Layer                       │
│  ┌─────────────┬─────────────────┬─────────────────────────────┐   │
│  │claude-flow  │    ruv-FANN     │            DAA              │   │
│  │@alpha       │   (Phase 1)     │                             │   │
│  │- Swarms     │ - Chart Neural  │ - Parallel Processing      │   │
│  │- Memory     │ - Table Neural  │ - Multi-format Consensus   │   │
│  │- Hooks      │ - Layout Neural │ - Fault Tolerance          │   │
│  │- Workflow   │ - WASM/SIMD     │ - Load Distribution        │   │
│  └─────────────┴─────────────────┴─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

## 🔧 Core Components & Library Mapping

### 1. Multimodal Document Parser (Rust Core)
```rust
// ✅ CORRECT: Strategic library usage for multimodal extraction
use lopdf::Document;
use pdf_extract::extract_text;
use image::{DynamicImage, ImageFormat};
use calamine::{Reader, Xlsx};
use tesseract::Tesseract;
use roxmltree::Document as XmlDoc;

pub struct MultimodalParser {
    // PDF handling with existing libraries
    pdf_backend: lopdf::Document,
    text_extractor: pdf_extract::Config,
    
    // Image processing pipeline
    image_processor: image::DynamicImage,
    ocr_engine: tesseract::Tesseract,
    
    // Office format handlers
    excel_reader: calamine::Xlsx<std::io::BufReader<std::fs::File>>,
    xml_parser: roxmltree::Document,
    
    // ✅ CUSTOM CODE ACCEPTABLE: Multimodal integration
    multimodal_combiner: MultimodalCombiner, // Custom integration logic
    table_detector: TableStructureDetector,   // Innovation area
    chart_recognizer: ChartRecognizer,        // Innovation area
}

// ✅ ACCEPTABLE: Custom code for multimodal integration
impl MultimodalCombiner {
    // Combines outputs from different extractors into unified representation
    pub fn combine(&self, text: Vec<String>, tables: Vec<Table>, 
                   images: Vec<Image>, charts: Vec<Chart>) -> UnifiedDoc {
        // Custom logic to merge multimodal data
    }
}
```

### 2. Coordination & Orchestration (claude-flow@alpha)
```rust
// ✅ CORRECT: Uses claude-flow@alpha for ALL coordination
use claude_flow::{SwarmCoordinator, Memory, Hooks};

pub struct NeuralDocFlowCoordinator {
    // Use claude-flow for swarm management
    swarm: claude_flow::SwarmCoordinator,
    // Use claude-flow for memory
    memory: claude_flow::Memory,
    // Use claude-flow hooks for coordination
    hooks: claude_flow::Hooks,
}

// ❌ WRONG: Do NOT implement custom coordination
// struct CustomSwarmManager { ... } // FORBIDDEN!
```

### 3. Neural Processing (ruv-FANN + Phase 1 Models)
```rust
// ✅ CORRECT: Uses ruv-FANN with Phase 1 neural capabilities
use ruv_fann::{Network, Training, Inference};

pub struct MultimodalNeuralProcessor {
    // Phase 1 neural models via ruv-FANN
    table_structure_net: ruv_fann::Network,    // From Phase 1
    chart_classifier: ruv_fann::Network,       // From Phase 1
    layout_analyzer: ruv_fann::Network,        // From Phase 1
    image_embedder: ruv_fann::Network,         // From Phase 1
    
    // Trainer for continuous improvement
    trainer: ruv_fann::Trainer,
    
    // ✅ ACCEPTABLE: Custom SIMD optimizations for performance
    simd_accelerator: SimdImageProcessor,     // Performance critical path
}

// ✅ ACCEPTABLE: SIMD optimizations for image processing
impl SimdImageProcessor {
    #[target_feature(enable = "avx2")]
    unsafe fn process_image_batch(&self, images: &[u8]) -> Vec<f32> {
        // Custom SIMD code for performance-critical image processing
    }
}
```

### 4. Distributed Processing (DAA)
```rust
// ✅ CORRECT: Uses DAA for ALL distributed operations
use daa::{Agent, Consensus, Distribution};

pub struct DistributedProcessor {
    // Use DAA agents only
    agents: Vec<daa::Agent>,
    // Use DAA consensus
    consensus: daa::ConsensusProtocol,
}

// ❌ WRONG: Do NOT implement distributed systems
// impl CustomDistributedSystem { ... } // FORBIDDEN!
```

## 📦 Implementation Rules for Multimodal Processing

### MUST Use Libraries For:

1. **Swarm Coordination**: 
   ```bash
   npx claude-flow@alpha swarm init --topology hierarchical
   npx claude-flow@alpha agent spawn --type processor
   ```

2. **Memory & Persistence**:
   ```javascript
   // Use claude-flow memory for coordination state
   await claudeFlow.memory.store('extraction/doc1/tables', tableData);
   await claudeFlow.memory.store('extraction/doc1/images', imageData);
   ```

3. **Neural Networks (Phase 1 Models)**:
   ```javascript
   // Use ruv-FANN with Phase 1 models
   const tableNet = new RuvFANN.Network([1024, 512, 256]); // Table structure
   const chartNet = new RuvFANN.Network([2048, 1024, 512]); // Chart recognition
   await tableNet.loadModel('phase1/table_structure.model');
   ```

4. **Format-Specific Extraction**:
   ```rust
   // PDF: lopdf + pdf-extract
   let doc = lopdf::Document::load("file.pdf")?;
   
   // Excel: calamine
   let mut workbook = calamine::open_workbook::<Xlsx<_>, _>("file.xlsx")?;
   
   // Images: image + tesseract
   let img = image::open("chart.png")?;
   let text = tesseract.set_image_from_mem(&img_bytes)?.get_text()?;
   ```

### When Custom Code IS Acceptable:

1. **Multimodal Integration Layer**:
   ```rust
   // ✅ ACCEPTABLE: Combining outputs from different extractors
   pub struct MultimodalIntegrator {
       pub fn merge_modalities(&self, 
           text: TextData, 
           tables: Vec<Table>,
           images: Vec<ProcessedImage>,
           charts: Vec<ChartData>
       ) -> UnifiedDocument {
           // Custom logic to create unified representation
       }
   }
   ```

2. **Performance Optimizations**:
   ```rust
   // ✅ ACCEPTABLE: SIMD for batch image processing
   #[target_feature(enable = "avx2")]
   unsafe fn batch_resize_images(images: &[DynamicImage]) -> Vec<DynamicImage> {
       // Custom SIMD implementation for performance
   }
   ```

3. **Novel Algorithm Implementation**:
   ```rust
   // ✅ ACCEPTABLE: Innovative chart reconstruction
   pub struct ChartReconstructor {
       pub fn detect_chart_type(&self, img: &DynamicImage) -> ChartType {
           // Novel algorithm not available in existing libraries
       }
       
       pub fn extract_data_points(&self, chart: &Chart) -> Vec<DataPoint> {
           // Custom implementation for data extraction
       }
   }
   ```

## 🏗️ Project Structure (Multimodal Focus)

```
neuraldocflow/
├── Cargo.toml              # Rust workspace
├── neuraldocflow-core/     # Core multimodal library (Rust)
│   ├── src/
│   │   ├── lib.rs         # Public API
│   │   ├── extractors/    # Format-specific extractors
│   │   │   ├── pdf.rs     # lopdf, pdf-extract
│   │   │   ├── docx.rs    # zip, roxmltree
│   │   │   ├── pptx.rs    # zip, roxmltree
│   │   │   ├── xlsx.rs    # calamine
│   │   │   ├── image.rs   # image, tesseract
│   │   │   └── html.rs    # scraper, kuchiki
│   │   ├── multimodal/    # Multimodal processing
│   │   │   ├── text.rs    # Text extraction & processing
│   │   │   ├── table.rs   # Table detection & extraction
│   │   │   ├── chart.rs   # Chart recognition & data extraction
│   │   │   ├── image.rs   # Image analysis & OCR
│   │   │   └── merger.rs  # Multimodal integration (custom)
│   │   ├── neural/        # Neural processing
│   │   │   ├── phase1.rs  # Phase 1 models via ruv-FANN
│   │   │   ├── simd.rs    # SIMD optimizations (custom)
│   │   │   └── trainer.rs # Model training interface
│   │   ├── coordinator.rs # claude-flow@alpha integration
│   │   └── distributed.rs # DAA integration
│   └── Cargo.toml
├── neuraldocflow-py/       # Python bindings (PyO3)
│   ├── src/
│   │   └── lib.rs         # PyO3 bindings
│   └── Cargo.toml
├── neuraldocflow-wasm/     # Web interface (wasm-bindgen)
│   ├── src/
│   │   └── lib.rs         # WASM bindings
│   └── Cargo.toml
└── neuraldocflow-mcp/      # MCP server
    ├── src/
    │   └── main.rs        # MCP implementation
    └── Cargo.toml
```

## 🔌 Interface Specifications

### Python Interface (PyO3) - Multimodal
```python
import neuraldocflow

# Initialize multimodal processor
processor = neuraldocflow.MultimodalProcessor(
    coordinator="claude-flow@alpha",
    neural_backend="ruv-fann",
    distributed_backend="daa",
    phase1_models="./models/phase1/"  # Pre-trained models from Phase 1
)

# Process document with multimodal extraction
result = processor.extract("report.pdf", {
    "extract_text": True,
    "extract_tables": True,
    "extract_images": True,
    "extract_charts": True,
    "ocr_images": True,
    "reconstruct_charts": True,
    "swarm_size": 6,
    "memory_namespace": "multimodal-extract"
})

# Access multimodal results
print(f"Extracted {len(result.tables)} tables")
print(f"Found {len(result.charts)} charts with data")
print(f"Processed {len(result.images)} images")
for table in result.tables:
    print(f"Table: {table.rows}x{table.cols} - {table.caption}")
```

### Web Interface (WASM) - Multimodal
```javascript
import init, { MultimodalDocFlow } from 'neuraldocflow-wasm';

await init();
const processor = new MultimodalDocFlow({
    useClaudeFlow: true,
    useRuvFANN: true,
    useDAA: true,
    phase1Models: './models/phase1.wasm'
});

// Process uploaded file
const result = await processor.extractMultimodal(file, {
    modalities: ['text', 'tables', 'images', 'charts'],
    enableOCR: true,
    chartReconstruction: true
});

// Display multimodal results
result.tables.forEach(table => {
    console.log(`Table found: ${table.rows}x${table.cols}`);
    renderTable(table.data);
});

result.charts.forEach(chart => {
    console.log(`Chart type: ${chart.type}`);
    renderChart(chart.reconstructedData);
});
```

### MCP Interface - Multimodal
```yaml
tools:
  - name: extract_multimodal
    description: Extract all modalities from document using NeuralDocFlow
    parameters:
      - name: file_path
        type: string
      - name: modalities
        type: array
        items:
          enum: [text, tables, images, charts, metadata]
      - name: config
        type: object
        properties:
          use_swarm: boolean
          enable_ocr: boolean
          reconstruct_charts: boolean
          phase1_models: string
          memory_namespace: string
          
  - name: extract_tables
    description: Extract only tables from document
    parameters:
      - name: file_path
        type: string
      - name: output_format
        enum: [json, csv, markdown]
        
  - name: extract_charts
    description: Extract and reconstruct chart data
    parameters:
      - name: file_path
        type: string
      - name: reconstruct_data
        type: boolean
```

## 🚀 Performance Targets (Multimodal)

### Speed Targets
- **Text Extraction**: 50x faster than pypdf (Rust + memory mapping)
- **Table Detection**: 100ms per page (Phase 1 neural models)
- **Chart Recognition**: 200ms per chart (ruv-FANN + SIMD)
- **Image Processing**: 30x faster than Python libs (SIMD batch processing)
- **Overall**: Process 1000-page document in <10 seconds

### Accuracy Targets
- **Text Extraction**: >99.5% accuracy
- **Table Structure**: >95% cell accuracy (Phase 1 models)
- **Chart Data**: >90% data point accuracy
- **OCR**: >98% accuracy on clean images

### Resource Efficiency
- **Memory**: <100MB for 1GB documents (memory mapping)
- **CPU**: Linear scaling with cores (DAA distribution)
- **GPU**: Optional acceleration for neural models

### Multimodal Integration
- **Parallel Processing**: All modalities extracted concurrently
- **Unified Output**: Single JSON/MessagePack structure
- **Streaming**: Support for large document streaming

## 🎯 Multimodal Processing Examples

### Example 1: Scientific Paper with Charts
```rust
let paper = MultimodalProcessor::new()
    .with_claude_flow("hierarchical")
    .with_phase1_models("./models/")
    .process("research_paper.pdf")?;

// Extracted data structure
paper.sections[0].text           // "Abstract: We present..."
paper.figures[0].image           // DynamicImage
paper.figures[0].caption         // "Figure 1: Performance comparison"
paper.figures[0].chart_data      // Vec<DataPoint> from chart reconstruction
paper.tables[0].data             // Vec<Vec<String>> table cells
paper.equations[0].latex         // "E = mc^2"
```

### Example 2: Financial Report Processing
```rust
let report = MultimodalProcessor::new()
    .with_distributed_agents(8)     // Use 8 DAA agents
    .extract_from("annual_report.xlsx")?;

// Access multimodal data
for sheet in report.sheets {
    println!("Sheet: {}", sheet.name);
    for table in sheet.tables {
        // Each table with structure preserved
    }
    for chart in sheet.charts {
        // Reconstructed chart data
        match chart.chart_type {
            ChartType::Line => process_line_chart(chart.data),
            ChartType::Bar => process_bar_chart(chart.data),
            ChartType::Pie => process_pie_chart(chart.data),
        }
    }
}
```

### Example 3: Presentation Extraction
```rust
let slides = MultimodalProcessor::new()
    .process("presentation.pptx")?;

for slide in slides.slides {
    println!("Slide {}: {}", slide.number, slide.title);
    
    // Text content
    for text_box in slide.text_boxes {
        println!("  - {}", text_box.content);
    }
    
    // Images and charts
    for visual in slide.visuals {
        match visual {
            Visual::Image(img) => process_image(img),
            Visual::Chart(chart) => {
                let data = chart.extract_data();
                // Use reconstructed data
            },
            Visual::Diagram(diag) => process_diagram(diag),
        }
    }
}
```

## ⚠️ Anti-Patterns to Prevent

### 1. Custom Coordination
```rust
// ❌ NEVER DO THIS
struct MySwarmCoordinator {
    agents: Vec<MyAgent>,
    // ... custom implementation
}

// ✅ ALWAYS DO THIS
use claude_flow::SwarmCoordinator;
let swarm = SwarmCoordinator::new();
```

### 2. Custom Neural Networks
```rust
// ❌ NEVER DO THIS
struct MyNeuralNetwork {
    layers: Vec<Layer>,
    // ... custom implementation
}

// ✅ ALWAYS DO THIS
use ruv_fann::Network;
let network = Network::new(&[784, 128, 10]);
```

### 3. Custom Distribution
```rust
// ❌ NEVER DO THIS
struct MyDistributedSystem {
    workers: Vec<Worker>,
    // ... custom implementation
}

// ✅ ALWAYS DO THIS
use daa::DistributedAgent;
let agent = DistributedAgent::new();
```

## 📋 Development Checklist (Multimodal)

### Core Dependencies
- [ ] Install claude-flow@alpha: `npm install -g claude-flow@alpha`
- [ ] Install ruv-FANN: `npm install ruv-fann`
- [ ] Install DAA: `npm install daa`
- [ ] Download Phase 1 models: `./scripts/download-phase1-models.sh`

### Rust Dependencies (Cargo.toml)
- [ ] `lopdf` and `pdf-extract` for PDF processing
- [ ] `calamine` for Excel/spreadsheet files
- [ ] `image` for image processing
- [ ] `tesseract-sys` for OCR capabilities
- [ ] `roxmltree` for Office XML formats
- [ ] `zip` for DOCX/PPTX archive handling
- [ ] `plotters` for chart analysis

### Development Setup
- [ ] Set up pre-commit hooks for library usage validation
- [ ] Configure SIMD feature flags in Cargo.toml
- [ ] Install Tesseract OCR system dependency
- [ ] Set up test documents with various modalities
- [ ] Configure CI/CD for multimodal testing

## 🎯 Success Criteria (Multimodal)

### Library Usage
1. **Strategic library usage** - Core libs for coordination, format-specific libs for extraction
2. **Phase 1 neural integration** - All neural models from Phase 1 via ruv-FANN
3. **Distributed multimodal** - Parallel extraction via DAA agents
4. **Innovation where needed** - Custom code only for multimodal integration & novel algorithms

### Multimodal Capabilities
1. **Format Support** - PDF, DOCX, PPTX, XLSX, Images (PNG/JPG/TIFF), HTML
2. **Modality Extraction** - Text, tables, images, charts, metadata
3. **Accuracy** - Meet or exceed targets for each modality
4. **Performance** - Sub-10 second processing for 1000-page documents

### Integration
1. **Unified API** - Single interface for all document types
2. **Python/Web/MCP** - Full multimodal support in all interfaces
3. **Streaming Support** - Handle large documents efficiently
4. **Memory Efficiency** - Stay within resource targets

### Quality
1. **Test Coverage** - >90% including multimodal integration tests
2. **Documentation** - Examples for each supported format
3. **Error Handling** - Graceful degradation for unsupported content
4. **Benchmarks** - Performance tests for each modality