# Neural Swarm Implementation Examples

## Quick Start: Processing a Technical Document

```typescript
import { NeuralSwarm, ClaudeFlow, RuvFann } from '@doc-ingest/neural-swarm';

async function processTechnicalDocument() {
    // Initialize neural swarm
    const swarm = new NeuralSwarm({
        claudeFlow: new ClaudeFlow({ 
            endpoint: 'http://localhost:8080' 
        }),
        ruvFann: new RuvFann({ 
            enableSimd: true,
            threads: 8 
        })
    });
    
    // Setup swarm with optimal configuration
    await swarm.initialize({
        documentType: 'technical',
        expectedSize: 'large', // >1000 pages
        outputFormat: 'structured-json'
    });
    
    // Process document
    const result = await swarm.processDocument({
        path: '/documents/technical-manual.pdf',
        extractionConfig: {
            entities: ['components', 'specifications', 'procedures'],
            relationships: true,
            figures: true,
            tables: true
        }
    });
    
    console.log(`Processed ${result.pageCount} pages in ${result.duration}ms`);
    console.log(`Extraction accuracy: ${result.accuracy}%`);
}
```

## Example 1: Custom Pipeline for Scientific Papers

```yaml
# scientific-pipeline.yaml
pipeline:
  name: "scientific-paper-processor"
  version: "2.0.0"
  
  initialization:
    swarm:
      topology: "mesh"
      agents: 12
      strategy: "adaptive"
    
    neural:
      models:
        - name: "sciBERT"
          path: "./models/scibert-base"
          purpose: "entity-recognition"
        - name: "figure-detector"
          path: "./models/yolo-figures-v5"
          purpose: "figure-extraction"
    
  stages:
    - name: "document-analysis"
      parallel: true
      agents: 4
      tasks:
        - analyze_structure:
            model: "layout-parser"
            output: "document-tree"
        - extract_metadata:
            fields: ["title", "authors", "abstract", "keywords"]
        - detect_sections:
            types: ["introduction", "methods", "results", "discussion"]
    
    - name: "content-extraction"
      parallel: true
      agents: 6
      tasks:
        - extract_entities:
            models: ["sciBERT", "chemBERT"]
            types: ["chemicals", "proteins", "genes", "diseases"]
        - extract_relationships:
            model: "sciRE"
            confidence_threshold: 0.85
        - extract_figures:
            model: "figure-detector"
            include_captions: true
        - extract_tables:
            model: "table-transformer"
            parse_cells: true
    
    - name: "knowledge-synthesis"
      agents: 2
      tasks:
        - build_knowledge_graph:
            format: "neo4j"
            merge_duplicates: true
        - generate_summary:
            max_length: 500
            focus: "key_findings"
```

```typescript
// Using the pipeline
async function processScientificPaper() {
    const pipeline = await PipelineBuilder.fromYaml('./scientific-pipeline.yaml');
    
    // Monitor progress in real-time
    pipeline.on('progress', (event) => {
        console.log(`Stage: ${event.stage}, Progress: ${event.progress}%`);
    });
    
    // Process with progress tracking
    const result = await pipeline.process({
        document: './papers/nature-2024-001.pdf',
        options: {
            preserveFormulas: true,
            extractCitations: true,
            generateBibTeX: true
        }
    });
    
    // Results include structured data
    console.log(`Extracted ${result.entities.length} entities`);
    console.log(`Found ${result.relationships.length} relationships`);
    console.log(`Knowledge graph nodes: ${result.knowledgeGraph.nodeCount}`);
}
```

## Example 2: Real-time Swarm Monitoring

```typescript
class SwarmMonitor {
    private claudeFlow: ClaudeFlow;
    private metrics: MetricsCollector;
    
    async startMonitoring(swarmId: string) {
        // Real-time monitoring with 1-second updates
        const monitor = await this.claudeFlow.swarmMonitor({
            swarmId,
            interval: 1000
        });
        
        monitor.on('update', async (status) => {
            // Display swarm status
            console.clear();
            console.log(this.formatSwarmStatus(status));
            
            // Auto-optimization based on metrics
            if (status.bottlenecks.length > 0) {
                await this.optimizeBottlenecks(status.bottlenecks);
            }
            
            // Scale based on queue depth
            if (status.queueDepth > status.activeAgents * 10) {
                await this.claudeFlow.swarmScale({
                    swarmId,
                    targetSize: Math.min(status.activeAgents + 2, 16)
                });
            }
        });
    }
    
    private formatSwarmStatus(status: SwarmStatus): string {
        return `
🐝 Neural Swarm Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Overview
   ├── Topology: ${status.topology}
   ├── Active Agents: ${status.activeAgents}/${status.maxAgents}
   ├── Tasks Completed: ${status.tasksCompleted}
   ├── Queue Depth: ${status.queueDepth}
   └── Avg Load: ${(status.avgLoad * 100).toFixed(1)}%

⚡ Performance
   ├── Throughput: ${status.throughput} pages/min
   ├── Avg Latency: ${status.avgLatency}ms
   ├── Memory Usage: ${status.memoryUsage}MB
   └── CPU Usage: ${status.cpuUsage}%

🤖 Agent Status
${status.agents.map(a => `   ├── ${a.name}: ${a.status} (${a.tasksProcessed} tasks)`).join('\n')}

${status.bottlenecks.length > 0 ? `
⚠️  Bottlenecks Detected
${status.bottlenecks.map(b => `   └── ${b.component}: ${b.issue}`).join('\n')}
` : '✅ No bottlenecks detected'}
`;
    }
}
```

## Example 3: Advanced Entity Extraction with Neural Networks

```rust
use ruv_fann::{Fann, Activation, TrainingData};
use doc_ingest::{DocumentProcessor, EntityExtractor};

pub struct NeuralEntityExtractor {
    fann: Fann,
    tokenizer: Tokenizer,
    embedder: WordEmbedder,
}

impl NeuralEntityExtractor {
    pub fn new() -> Self {
        // Create specialized neural network for entity extraction
        let mut fann = Fann::new(&[768, 512, 256, 128, 64, 10])?;
        fann.set_activation_function_hidden(Activation::ReLU);
        fann.set_activation_function_output(Activation::Softmax);
        
        // Load pre-trained weights
        fann.load("./models/entity-extractor.fann")?;
        
        Self {
            fann,
            tokenizer: Tokenizer::new("bert-base-uncased"),
            embedder: WordEmbedder::new("glove-6B-300d"),
        }
    }
    
    pub async fn extract_entities(&self, text: &str) -> Vec<Entity> {
        // Tokenize and embed text
        let tokens = self.tokenizer.tokenize(text);
        let embeddings = self.embedder.embed_batch(&tokens);
        
        // Process through neural network with SIMD
        let predictions = self.process_with_simd(embeddings);
        
        // Convert predictions to entities
        self.predictions_to_entities(tokens, predictions)
    }
    
    fn process_with_simd(&self, embeddings: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        use std::arch::x86_64::*;
        
        embeddings.par_iter()
            .map(|embedding| {
                // SIMD-accelerated forward pass
                unsafe {
                    let mut input = [0.0f32; 768];
                    input.copy_from_slice(embedding);
                    
                    // Process in 8-element chunks
                    let mut output = vec![0.0f32; 10];
                    for i in (0..768).step_by(8) {
                        let vec = _mm256_loadu_ps(&input[i]);
                        // Neural network operations...
                    }
                    
                    output
                }
            })
            .collect()
    }
}

// Usage example
#[tokio::main]
async fn main() {
    let extractor = NeuralEntityExtractor::new();
    
    let text = "Apple Inc. announced a new partnership with OpenAI to integrate 
                GPT-4 into iOS 18, launching in September 2024.";
    
    let entities = extractor.extract_entities(text).await;
    
    for entity in entities {
        println!("{}: {} (confidence: {:.2})", 
                 entity.entity_type, 
                 entity.text, 
                 entity.confidence);
    }
    // Output:
    // ORGANIZATION: Apple Inc. (confidence: 0.98)
    // ORGANIZATION: OpenAI (confidence: 0.96)
    // PRODUCT: GPT-4 (confidence: 0.94)
    // PRODUCT: iOS 18 (confidence: 0.92)
    // DATE: September 2024 (confidence: 0.95)
}
```

## Example 4: Distributed Document Processing

```typescript
class DistributedProcessor {
    private swarm: NeuralSwarm;
    private loadBalancer: LoadBalancer;
    
    async processLargeCorpus(documents: string[]): Promise<CorpusResult> {
        // Initialize swarm with maximum agents
        await this.swarm.initialize({
            topology: 'mesh',
            maxAgents: 16,
            strategy: 'parallel'
        });
        
        // Create processing batches
        const batches = this.createOptimalBatches(documents);
        
        // Process in parallel waves
        const results = await Promise.all(
            batches.map(batch => this.processBatch(batch))
        );
        
        // Aggregate results using neural aggregation
        return this.neuralAggregate(results);
    }
    
    private async processBatch(batch: DocumentBatch): Promise<BatchResult> {
        // Assign optimal agents
        const agents = await this.loadBalancer.assignAgents(batch);
        
        // Create processing tasks
        const tasks = batch.documents.map((doc, i) => ({
            id: `task-${batch.id}-${i}`,
            document: doc,
            agent: agents[i % agents.length],
            priority: this.calculatePriority(doc)
        }));
        
        // Execute with progress tracking
        const taskPromises = tasks.map(task => 
            this.executeWithProgress(task)
        );
        
        return Promise.all(taskPromises);
    }
    
    private async executeWithProgress(task: ProcessingTask): Promise<TaskResult> {
        // Store task in swarm memory
        await this.swarm.memory.store({
            key: `task/${task.id}/status`,
            value: { status: 'started', timestamp: Date.now() }
        });
        
        // Process document
        const result = await task.agent.process(task.document);
        
        // Update progress
        await this.swarm.memory.store({
            key: `task/${task.id}/result`,
            value: result
        });
        
        return result;
    }
}
```

## Example 5: Custom Neural Model Training

```typescript
class CustomModelTrainer {
    async trainDocumentClassifier(trainingData: TrainingDataset) {
        // Initialize neural training with WASM SIMD
        const trainer = await this.claudeFlow.neuralTrain({
            pattern_type: 'coordination',
            training_data: JSON.stringify({
                dataset: trainingData.samples,
                labels: trainingData.labels,
                validation_split: 0.2
            }),
            epochs: 100
        });
        
        // Monitor training progress
        trainer.on('epoch', (epoch, metrics) => {
            console.log(`Epoch ${epoch}: Loss=${metrics.loss.toFixed(4)}, Accuracy=${metrics.accuracy.toFixed(4)}`);
            
            // Early stopping
            if (metrics.val_loss < 0.01) {
                trainer.stop();
            }
        });
        
        // Wait for training completion
        const model = await trainer.complete();
        
        // Save trained model
        await this.claudeFlow.modelSave({
            modelId: model.id,
            path: './models/document-classifier.fann'
        });
        
        return model;
    }
}
```

## Example 6: Real-time Processing Pipeline

```typescript
class RealtimePipeline {
    private eventStream: EventEmitter;
    private swarm: NeuralSwarm;
    
    async startRealtimeProcessing() {
        // Setup real-time document stream
        this.eventStream.on('document', async (doc) => {
            // Immediate processing assignment
            const agent = await this.swarm.getAvailableAgent();
            
            // Process with sub-100ms latency
            const startTime = Date.now();
            const result = await agent.processRealtime(doc, {
                mode: 'streaming',
                maxLatency: 100,
                partialResults: true
            });
            
            console.log(`Processed in ${Date.now() - startTime}ms`);
            
            // Stream results as they become available
            result.on('partial', (partial) => {
                this.emitResult(partial);
            });
        });
    }
}
```

## Performance Optimization Tips

1. **Batch Operations**: Always process documents in batches for maximum throughput
2. **Agent Specialization**: Use specialized agents for different document types
3. **Memory Management**: Use zero-copy operations for large documents
4. **SIMD Utilization**: Ensure CPU supports AVX2 for maximum performance
5. **Pipeline Tuning**: Adjust agent counts based on workload characteristics

## Troubleshooting Common Issues

### Issue: Slow Processing Speed
```typescript
// Check and optimize swarm configuration
const diagnostics = await swarm.runDiagnostics();
if (diagnostics.bottlenecks.includes('agent-shortage')) {
    await swarm.scaleUp(4); // Add 4 more agents
}
```

### Issue: Memory Usage Too High
```rust
// Enable streaming mode for large documents
processor.set_mode(ProcessingMode::Streaming {
    chunk_size: 1024 * 1024, // 1MB chunks
    overlap: 256,
    gc_interval: 100,
});
```

### Issue: Accuracy Below Target
```typescript
// Fine-tune neural models on domain-specific data
await trainer.fineTune({
    baseModel: 'general-extractor',
    domainData: './data/legal-documents',
    targetAccuracy: 0.95
});
```