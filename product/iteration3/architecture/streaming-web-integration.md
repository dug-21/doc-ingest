# NeuralDocFlow Streaming and Web Integration Architecture

## Overview

NeuralDocFlow's streaming and web integration layer provides real-time document processing capabilities through multiple channels:
- **Real-time Document Streaming**: Process documents as they arrive
- **WebSocket Interfaces**: Bidirectional real-time communication
- **WASM Integration**: Client-side processing capabilities
- **Progressive Web App**: Offline-capable document processing
- **Event-Driven Architecture**: Reactive processing pipelines

## Real-Time Document Streaming

### Stream Processing Pipeline

```rust
// Core streaming architecture
pub struct DocumentStream {
    input_receiver: Receiver<DocumentEvent>,
    processing_pool: SwarmCoordinator,
    output_sender: Sender<ProcessingResult>,
    buffer_manager: StreamBuffer,
}

pub enum DocumentEvent {
    NewDocument {
        id: String,
        source: DocumentSource,
        profile: String,
        priority: Priority,
    },
    ProcessingUpdate {
        id: String,
        progress: f32,
        stage: ProcessingStage,
    },
    ProcessingComplete {
        id: String,
        result: ExtractionResult,
    },
    ProcessingError {
        id: String,
        error: ProcessingError,
    },
}
```

### Stream Sources Configuration

```yaml
# Stream source configurations
stream_sources:
  filesystem_watcher:
    type: "filesystem"
    config:
      watch_directories:
        - "/incoming/documents"
        - "/urgent/processing"
      file_patterns: ["*.pdf", "*.docx"]
      recursive: true
      debounce_ms: 500
      auto_move_processed: true
      processed_directory: "/processed"
      
  s3_bucket_monitor:
    type: "s3"
    config:
      bucket: "document-inbox"
      prefix: "incoming/"
      polling_interval_ms: 1000
      sqs_notifications: true
      delete_after_processing: false
      
  kafka_consumer:
    type: "kafka"
    config:
      topic: "document-stream"
      consumer_group: "neuraldocflow-processors"
      auto_commit: false
      max_poll_records: 10
      
  webhook_receiver:
    type: "webhook"
    config:
      endpoint: "/api/v1/documents/webhook"
      authentication: "bearer_token"
      max_payload_size: "50MB"
      async_processing: true
```

### Stream Processing Configuration

```yaml
# Stream processing pipeline
processing_pipeline:
  buffer_config:
    max_buffer_size: 100
    batch_size: 5
    flush_interval_ms: 1000
    priority_queues: true
    
  swarm_config:
    min_agents: 2
    max_agents: 10
    auto_scaling: true
    scaling_metrics:
      - "queue_depth"
      - "processing_latency"
      - "cpu_utilization"
      
  output_streams:
    - type: "websocket"
      config:
        broadcast_progress: true
        result_channels: ["results", "notifications"]
        
    - type: "kafka"
      config:
        topic: "processing-results"
        partition_key: "document_type"
        
    - type: "webhook"
      config:
        endpoints:
          - url: "https://client-app.com/webhook"
            events: ["completion", "error"]
            retry_config:
              max_retries: 3
              backoff_strategy: "exponential"
```

## WebSocket Interface Design

### Connection Management

```typescript
// Client-side WebSocket interface
interface NeuralDocFlowWebSocket {
  connect(apiKey: string, options?: ConnectionOptions): Promise<void>;
  disconnect(): void;
  
  // Document processing
  processDocument(request: ProcessingRequest): Promise<string>; // Returns job ID
  subscribeToJob(jobId: string): void;
  unsubscribeFromJob(jobId: string): void;
  
  // Streaming operations
  startStream(config: StreamConfig): Promise<string>; // Returns stream ID
  stopStream(streamId: string): void;
  
  // Event handlers
  onProgress(callback: (progress: ProgressEvent) => void): void;
  onResult(callback: (result: ProcessingResult) => void): void;
  onError(callback: (error: ErrorEvent) => void): void;
  onConnectionStatus(callback: (status: ConnectionStatus) => void): void;
}

// Implementation
class NeuralDocFlowClient implements NeuralDocFlowWebSocket {
  private ws: WebSocket;
  private messageHandlers: Map<string, Function>;
  private subscriptions: Set<string>;
  
  async connect(apiKey: string, options?: ConnectionOptions) {
    const wsUrl = `wss://api.neuraldocflow.com/v1/stream?token=${apiKey}`;
    this.ws = new WebSocket(wsUrl);
    
    this.ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      this.handleMessage(message);
    };
    
    return new Promise((resolve, reject) => {
      this.ws.onopen = () => resolve();
      this.ws.onerror = (error) => reject(error);
    });
  }
  
  async processDocument(request: ProcessingRequest): Promise<string> {
    const messageId = generateId();
    const message = {
      id: messageId,
      type: 'process_document',
      payload: request
    };
    
    this.ws.send(JSON.stringify(message));
    
    return new Promise((resolve, reject) => {
      this.messageHandlers.set(messageId, (response) => {
        if (response.error) {
          reject(new Error(response.error.message));
        } else {
          resolve(response.payload.job_id);
        }
      });
    });
  }
  
  private handleMessage(message: WebSocketMessage) {
    switch (message.type) {
      case 'response':
        const handler = this.messageHandlers.get(message.id);
        if (handler) {
          handler(message);
          this.messageHandlers.delete(message.id);
        }
        break;
        
      case 'event':
        this.handleEvent(message);
        break;
        
      case 'error':
        this.handleError(message);
        break;
    }
  }
}
```

### Message Protocol

```typescript
// WebSocket message types
interface WebSocketMessage {
  id: string;
  type: 'request' | 'response' | 'event' | 'error';
  timestamp: string;
  payload: any;
}

interface ProcessingRequest {
  document: {
    source: 'file' | 'url' | 'base64';
    content: string;
    filename?: string;
    contentType?: string;
  };
  profile: string;
  options: {
    priority?: 'low' | 'normal' | 'high' | 'urgent';
    qualityThreshold?: number;
    outputFormat?: 'json' | 'yaml' | 'xml';
    streamResults?: boolean;
  };
}

interface ProgressEvent {
  jobId: string;
  progress: number; // 0.0 to 1.0
  stage: string;
  estimatedRemainingMs?: number;
  currentOperation?: string;
}

interface ProcessingResult {
  jobId: string;
  status: 'completed' | 'failed' | 'partial';
  extractedData?: any;
  metadata?: ProcessingMetadata;
  validationResults?: ValidationResult[];
  error?: ErrorInfo;
}
```

## WASM Integration Architecture

### WASM Module Design

```rust
// WASM interface using wasm-bindgen
use wasm_bindgen::prelude::*;
use neuraldocflow_core::{DocumentProcessor, ExtractionProfile};

#[wasm_bindgen]
pub struct NeuralDocFlowWasm {
    processor: DocumentProcessor,
    profiles: Vec<ExtractionProfile>,
}

#[wasm_bindgen]
impl NeuralDocFlowWasm {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        utils::set_panic_hook();
        
        Self {
            processor: DocumentProcessor::new_lightweight(),
            profiles: Vec::new(),
        }
    }
    
    #[wasm_bindgen]
    pub async fn load_profile(&mut self, profile_data: &str) -> Result<(), JsValue> {
        let profile: ExtractionProfile = serde_json::from_str(profile_data)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        self.profiles.push(profile);
        Ok(())
    }
    
    #[wasm_bindgen]
    pub async fn process_document(
        &self,
        document_data: &[u8],
        profile_name: &str,
        progress_callback: &js_sys::Function,
    ) -> Result<JsValue, JsValue> {
        let profile = self.profiles.iter()
            .find(|p| p.name == profile_name)
            .ok_or_else(|| JsValue::from_str("Profile not found"))?;
        
        // Create progress reporter
        let progress_reporter = |progress: f32, stage: &str| {
            let progress_obj = js_sys::Object::new();
            js_sys::Reflect::set(&progress_obj, &"progress".into(), &progress.into())?;
            js_sys::Reflect::set(&progress_obj, &"stage".into(), &stage.into())?;
            progress_callback.call1(&JsValue::NULL, &progress_obj)?;
            Ok::<(), JsValue>(())
        };
        
        // Process document
        let result = self.processor.process_with_callback(
            document_data,
            profile,
            progress_reporter,
        ).await.map_err(|e| JsValue::from_str(&e.to_string()))?;
        
        // Convert result to JS object
        Ok(serde_wasm_bindgen::to_value(&result)?)
    }
    
    #[wasm_bindgen]
    pub fn get_supported_formats(&self) -> js_sys::Array {
        let formats = js_sys::Array::new();
        formats.push(&"pdf".into());
        formats.push(&"docx".into());
        formats.push(&"html".into());
        formats
    }
}

// JavaScript bindings
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
    
    #[wasm_bindgen(js_namespace = console, js_name = log)]
    fn log_object(o: &JsValue);
}

// Utility functions for JS integration
#[wasm_bindgen]
pub fn init_logging() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
}
```

### JavaScript Integration Layer

```javascript
// High-level JavaScript API
class NeuralDocFlowWASM {
  constructor() {
    this.wasmModule = null;
    this.initialized = false;
  }
  
  async init(wasmUrl) {
    // Load WASM module
    const wasm = await import(wasmUrl);
    await wasm.default();
    
    this.wasmModule = new wasm.NeuralDocFlowWasm();
    this.wasmModule.init_logging();
    this.initialized = true;
  }
  
  async loadProfile(profileConfig) {
    if (!this.initialized) throw new Error('WASM module not initialized');
    
    const profileJson = JSON.stringify(profileConfig);
    await this.wasmModule.load_profile(profileJson);
  }
  
  async processDocument(file, profileName, options = {}) {
    if (!this.initialized) throw new Error('WASM module not initialized');
    
    // Convert file to Uint8Array
    const arrayBuffer = await file.arrayBuffer();
    const uint8Array = new Uint8Array(arrayBuffer);
    
    // Create progress callback
    const progressCallback = (progressData) => {
      if (options.onProgress) {
        options.onProgress(progressData);
      }
    };
    
    // Process document
    const result = await this.wasmModule.process_document(
      uint8Array,
      profileName,
      progressCallback
    );
    
    return result;
  }
  
  getSupportedFormats() {
    if (!this.initialized) return [];
    return Array.from(this.wasmModule.get_supported_formats());
  }
}

// Worker integration for background processing
class NeuralDocFlowWorker {
  constructor(wasmUrl) {
    this.worker = new Worker('neuraldocflow-worker.js');
    this.pendingRequests = new Map();
    
    this.worker.onmessage = (event) => {
      const { id, result, error } = event.data;
      const request = this.pendingRequests.get(id);
      
      if (request) {
        if (error) {
          request.reject(new Error(error));
        } else {
          request.resolve(result);
        }
        this.pendingRequests.delete(id);
      }
    };
    
    // Initialize worker
    this.worker.postMessage({
      type: 'init',
      wasmUrl: wasmUrl
    });
  }
  
  async processDocument(file, profileName, options = {}) {
    const requestId = generateId();
    
    return new Promise((resolve, reject) => {
      this.pendingRequests.set(requestId, { resolve, reject });
      
      this.worker.postMessage({
        type: 'process',
        id: requestId,
        file: file,
        profileName: profileName,
        options: options
      });
    });
  }
}
```

## Progressive Web App Integration

### Service Worker for Offline Processing

```javascript
// service-worker.js
const CACHE_NAME = 'neuraldocflow-v1';
const urlsToCache = [
  '/',
  '/static/js/neuraldocflow.js',
  '/static/wasm/neuraldocflow.wasm',
  '/static/profiles/sec_10k.json',
  '/static/profiles/contract_analysis.json'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        return cache.addAll(urlsToCache);
      })
  );
});

self.addEventListener('fetch', (event) => {
  // Handle API requests when offline
  if (event.request.url.includes('/api/')) {
    event.respondWith(handleApiRequest(event.request));
    return;
  }
  
  // Handle static resources
  event.respondWith(
    caches.match(event.request)
      .then((response) => {
        return response || fetch(event.request);
      })
  );
});

async function handleApiRequest(request) {
  // Try network first
  try {
    const response = await fetch(request);
    return response;
  } catch (error) {
    // Fallback to offline processing if available
    return handleOfflineProcessing(request);
  }
}

async function handleOfflineProcessing(request) {
  // Parse request for document processing
  const requestData = await request.json();
  
  // Check if we can process offline
  if (canProcessOffline(requestData)) {
    const result = await processDocumentOffline(requestData);
    return new Response(JSON.stringify(result), {
      headers: { 'Content-Type': 'application/json' }
    });
  }
  
  // Queue for later processing
  await queueForLaterProcessing(requestData);
  return new Response(JSON.stringify({
    status: 'queued',
    message: 'Queued for processing when online'
  }), {
    status: 202,
    headers: { 'Content-Type': 'application/json' }
  });
}
```

### Offline Storage and Sync

```javascript
// offline-manager.js
class OfflineManager {
  constructor() {
    this.db = null;
    this.syncQueue = [];
  }
  
  async init() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('NeuralDocFlowDB', 1);
      
      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };
      
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        
        // Store for processed documents
        const documentsStore = db.createObjectStore('documents', {
          keyPath: 'id'
        });
        documentsStore.createIndex('timestamp', 'timestamp');
        documentsStore.createIndex('profile', 'profile');
        
        // Store for sync queue
        const syncStore = db.createObjectStore('syncQueue', {
          keyPath: 'id'
        });
        syncStore.createIndex('timestamp', 'timestamp');
      };
    });
  }
  
  async storeDocument(document, result) {
    const transaction = this.db.transaction(['documents'], 'readwrite');
    const store = transaction.objectStore('documents');
    
    const record = {
      id: generateId(),
      document: document,
      result: result,
      timestamp: Date.now(),
      profile: document.profile,
      synced: navigator.onLine
    };
    
    await store.add(record);
    
    if (!navigator.onLine) {
      await this.addToSyncQueue(record);
    }
  }
  
  async addToSyncQueue(record) {
    const transaction = this.db.transaction(['syncQueue'], 'readwrite');
    const store = transaction.objectStore('syncQueue');
    
    await store.add({
      id: record.id,
      action: 'upload_result',
      data: record,
      timestamp: Date.now(),
      retries: 0
    });
  }
  
  async syncWhenOnline() {
    if (!navigator.onLine) return;
    
    const transaction = this.db.transaction(['syncQueue'], 'readwrite');
    const store = transaction.objectStore('syncQueue');
    const request = store.getAll();
    
    request.onsuccess = async () => {
      const items = request.result;
      
      for (const item of items) {
        try {
          await this.syncItem(item);
          await store.delete(item.id);
        } catch (error) {
          console.error('Sync failed for item:', item.id, error);
          
          // Increment retry count
          item.retries = (item.retries || 0) + 1;
          if (item.retries < 3) {
            await store.put(item);
          } else {
            console.error('Max retries reached for item:', item.id);
            await store.delete(item.id);
          }
        }
      }
    };
  }
  
  async syncItem(item) {
    const response = await fetch('/api/v1/results/sync', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${getApiKey()}`
      },
      body: JSON.stringify(item.data)
    });
    
    if (!response.ok) {
      throw new Error(`Sync failed: ${response.statusText}`);
    }
  }
}

// Auto-sync when coming online
window.addEventListener('online', () => {
  const offlineManager = new OfflineManager();
  offlineManager.syncWhenOnline();
});
```

## Event-Driven Architecture

### Event Processing Pipeline

```rust
// Event-driven processing system
use tokio::sync::mpsc;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DocumentEvent {
    DocumentReceived {
        id: String,
        source: String,
        metadata: DocumentMetadata,
    },
    ProcessingStarted {
        id: String,
        swarm_id: String,
        estimated_duration: Duration,
    },
    ProcessingProgress {
        id: String,
        progress: f32,
        stage: ProcessingStage,
        intermediate_results: Option<serde_json::Value>,
    },
    ProcessingCompleted {
        id: String,
        result: ExtractionResult,
        performance_metrics: ProcessingMetrics,
    },
    ProcessingFailed {
        id: String,
        error: ProcessingError,
        retry_suggestions: Vec<RetryOption>,
    },
    QualityValidated {
        id: String,
        validation_results: ValidationReport,
        confidence_score: f32,
    },
    ResultDelivered {
        id: String,
        delivery_status: DeliveryStatus,
        timestamp: DateTime<Utc>,
    },
}

pub struct EventProcessor {
    event_receiver: mpsc::Receiver<DocumentEvent>,
    handlers: HashMap<String, Box<dyn EventHandler>>,
    metrics_collector: MetricsCollector,
}

impl EventProcessor {
    pub async fn run(&mut self) {
        while let Some(event) = self.event_receiver.recv().await {
            self.process_event(event).await;
        }
    }
    
    async fn process_event(&mut self, event: DocumentEvent) {
        // Log event for debugging and metrics
        self.metrics_collector.record_event(&event);
        
        // Route event to appropriate handlers
        match &event {
            DocumentEvent::DocumentReceived { id, .. } => {
                self.handle_document_received(event).await;
            }
            DocumentEvent::ProcessingProgress { id, .. } => {
                self.handle_progress_update(event).await;
            }
            DocumentEvent::ProcessingCompleted { id, .. } => {
                self.handle_processing_completed(event).await;
            }
            // ... other event types
        }
        
        // Broadcast event to subscribed clients
        self.broadcast_event(event).await;
    }
    
    async fn broadcast_event(&self, event: DocumentEvent) {
        // Send to WebSocket subscribers
        if let Some(websocket_handler) = self.handlers.get("websocket") {
            websocket_handler.handle_event(&event).await;
        }
        
        // Send to webhook subscribers
        if let Some(webhook_handler) = self.handlers.get("webhook") {
            webhook_handler.handle_event(&event).await;
        }
        
        // Send to Kafka for external systems
        if let Some(kafka_handler) = self.handlers.get("kafka") {
            kafka_handler.handle_event(&event).await;
        }
    }
}
```

### React Integration with Real-Time Updates

```typescript
// React hook for real-time document processing
import { useState, useEffect, useCallback } from 'react';

interface UseDocumentProcessingOptions {
  apiKey: string;
  onProgress?: (progress: ProgressEvent) => void;
  onComplete?: (result: ProcessingResult) => void;
  onError?: (error: Error) => void;
}

export function useDocumentProcessing(options: UseDocumentProcessingOptions) {
  const [client, setClient] = useState<NeuralDocFlowClient | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [jobs, setJobs] = useState<Map<string, JobStatus>>(new Map());
  
  useEffect(() => {
    const neuralClient = new NeuralDocFlowClient();
    
    neuralClient.connect(options.apiKey).then(() => {
      setIsConnected(true);
      setClient(neuralClient);
    });
    
    neuralClient.onProgress((progress) => {
      setJobs(prev => new Map(prev.set(progress.jobId, {
        status: 'processing',
        progress: progress.progress,
        stage: progress.stage
      })));
      
      options.onProgress?.(progress);
    });
    
    neuralClient.onResult((result) => {
      setJobs(prev => new Map(prev.set(result.jobId, {
        status: result.status,
        result: result.extractedData
      })));
      
      options.onComplete?.(result);
    });
    
    neuralClient.onError((error) => {
      options.onError?.(new Error(error.message));
    });
    
    return () => {
      neuralClient.disconnect();
    };
  }, [options.apiKey]);
  
  const processDocument = useCallback(async (
    file: File,
    profile: string,
    processingOptions?: ProcessingOptions
  ) => {
    if (!client || !isConnected) {
      throw new Error('Client not connected');
    }
    
    const jobId = await client.processDocument({
      document: {
        source: 'file',
        content: await fileToBase64(file),
        filename: file.name,
        contentType: file.type
      },
      profile,
      options: {
        streamResults: true,
        ...processingOptions
      }
    });
    
    setJobs(prev => new Map(prev.set(jobId, {
      status: 'queued',
      progress: 0,
      stage: 'initializing'
    })));
    
    client.subscribeToJob(jobId);
    
    return jobId;
  }, [client, isConnected]);
  
  return {
    processDocument,
    isConnected,
    jobs: Array.from(jobs.entries()).map(([id, status]) => ({ id, ...status }))
  };
}

// React component using the hook
export function DocumentProcessor() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [selectedProfile, setSelectedProfile] = useState('sec_10k');
  
  const { processDocument, isConnected, jobs } = useDocumentProcessing({
    apiKey: process.env.REACT_APP_NEURALDOCFLOW_API_KEY!,
    onProgress: (progress) => {
      console.log(`Job ${progress.jobId}: ${progress.progress * 100}% - ${progress.stage}`);
    },
    onComplete: (result) => {
      console.log('Processing completed:', result);
    },
    onError: (error) => {
      console.error('Processing error:', error);
    }
  });
  
  const handleFileUpload = async () => {
    if (!selectedFile) return;
    
    try {
      const jobId = await processDocument(selectedFile, selectedProfile);
      console.log('Started processing with job ID:', jobId);
    } catch (error) {
      console.error('Failed to start processing:', error);
    }
  };
  
  return (
    <div className="document-processor">
      <div className="upload-section">
        <input
          type="file"
          accept=".pdf,.docx"
          onChange={(e) => setSelectedFile(e.target.files?.[0] || null)}
        />
        <select
          value={selectedProfile}
          onChange={(e) => setSelectedProfile(e.target.value)}
        >
          <option value="sec_10k">SEC 10-K Filing</option>
          <option value="contract_analysis">Contract Analysis</option>
          <option value="medical_record">Medical Record</option>
        </select>
        <button
          onClick={handleFileUpload}
          disabled={!selectedFile || !isConnected}
        >
          Process Document
        </button>
      </div>
      
      <div className="jobs-section">
        <h3>Processing Jobs</h3>
        {jobs.map((job) => (
          <div key={job.id} className="job-status">
            <div>Job {job.id}: {job.status}</div>
            {job.status === 'processing' && (
              <div>
                <div>Stage: {job.stage}</div>
                <div>Progress: {Math.round(job.progress * 100)}%</div>
                <progress value={job.progress} max={1} />
              </div>
            )}
            {job.result && (
              <div>
                <h4>Results:</h4>
                <pre>{JSON.stringify(job.result, null, 2)}</pre>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
```

This comprehensive streaming and web integration architecture provides:

1. **Real-time Processing**: Documents processed as they arrive from multiple sources
2. **WebSocket Communication**: Bidirectional real-time updates between client and server
3. **WASM Integration**: Client-side processing capabilities for offline use
4. **Progressive Web App**: Offline-capable document processing with sync capabilities
5. **Event-Driven Architecture**: Reactive processing with comprehensive event handling
6. **Modern Web Integration**: React hooks and components for seamless integration

The architecture ensures scalability, real-time responsiveness, and offline capabilities while maintaining high performance and user experience.