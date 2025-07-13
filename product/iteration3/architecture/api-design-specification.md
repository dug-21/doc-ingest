# NeuralDocFlow API Design Specification

## Overview

NeuralDocFlow provides multiple API interfaces designed for different usage patterns:
- **MCP Interface**: For LLM agents and intelligent applications
- **REST API**: For human developers and web applications  
- **WebSocket API**: For real-time streaming applications
- **WASM/JavaScript API**: For browser-based applications
- **Python SDK**: For data science and automation workflows

## REST API Design

### Base URL Structure
```
https://api.neuraldocflow.com/v1/
```

### Authentication
```http
Authorization: Bearer <api_key>
Content-Type: application/json
```

### Core Endpoints

#### Document Processing

**POST /documents/process**
```json
{
  "document": {
    "source": "file|url|base64",
    "content": "<file_content_or_url>",
    "filename": "document.pdf",
    "content_type": "application/pdf"
  },
  "processing": {
    "profile": "sec_10k",
    "mode": "full|targeted|streaming",
    "quality_threshold": 0.9,
    "output_format": "json|yaml|xml"
  },
  "options": {
    "async": true,
    "webhook_url": "https://your-app.com/webhook",
    "cache_results": true,
    "preserve_metadata": true
  }
}
```

**Response (Async)**:
```json
{
  "job_id": "job_12345",
  "status": "processing",
  "estimated_completion": "2024-01-01T12:05:00Z",
  "progress_url": "/jobs/job_12345/status",
  "result_url": "/jobs/job_12345/result"
}
```

**Response (Sync)**:
```json
{
  "job_id": "job_12345",
  "status": "completed",
  "processing_time_ms": 2341,
  "confidence_score": 0.94,
  "result": {
    "extracted_data": { /* ... */ },
    "metadata": { /* ... */ },
    "validation_results": [ /* ... */ ]
  }
}
```

#### Batch Processing

**POST /documents/batch**
```json
{
  "documents": [
    {
      "id": "doc1",
      "source": "file",
      "content": "<base64>",
      "profile": "sec_10k"
    },
    {
      "id": "doc2", 
      "source": "url",
      "content": "https://example.com/doc.pdf",
      "profile": "contract_analysis"
    }
  ],
  "processing": {
    "parallelism": 4,
    "priority": "normal|high|low",
    "output_collection": "individual|merged|indexed"
  }
}
```

#### Job Management

**GET /jobs/{job_id}/status**
```json
{
  "job_id": "job_12345",
  "status": "processing|completed|failed|queued",
  "progress": 0.75,
  "stages": [
    {
      "name": "document_parsing",
      "status": "completed",
      "duration_ms": 450
    },
    {
      "name": "neural_extraction",
      "status": "processing", 
      "progress": 0.6
    }
  ],
  "estimated_remaining_ms": 1200
}
```

**GET /jobs/{job_id}/result**
```json
{
  "job_id": "job_12345",
  "status": "completed",
  "result": {
    "extracted_data": {
      "company_name": "ACME Corporation",
      "fiscal_year_end": "2023-12-31",
      "revenue": 1250000000,
      "financial_tables": { /* ... */ }
    },
    "metadata": {
      "processing_time_ms": 2341,
      "confidence_score": 0.94,
      "model_versions": {
        "layout_detection": "v2.1",
        "table_extraction": "v1.8"
      }
    },
    "validation_results": [
      {
        "rule": "financial_consistency",
        "status": "passed",
        "confidence": 0.96
      }
    ]
  }
}
```

#### Profile Management

**GET /profiles**
```json
{
  "profiles": [
    {
      "name": "sec_10k",
      "description": "SEC Form 10-K processing",
      "version": "1.2",
      "supported_formats": ["pdf"],
      "avg_processing_time_ms": 3000
    }
  ]
}
```

**POST /profiles**
```json
{
  "name": "custom_contract",
  "description": "Custom contract analysis",
  "document_type": "legal_contract",
  "extraction_rules": {
    "sections": [ /* YAML config */ ],
    "fields": [ /* extraction fields */ ]
  },
  "quality_requirements": {
    "min_confidence": 0.9,
    "validation_rules": [ /* rules */ ]
  }
}
```

## WebSocket API Design

### Connection
```javascript
const ws = new WebSocket('wss://api.neuraldocflow.com/v1/stream');
```

### Message Format
```json
{
  "type": "request|response|event|error",
  "id": "unique_message_id",
  "timestamp": "2024-01-01T12:00:00Z",
  "payload": { /* message-specific data */ }
}
```

### Streaming Processing
```json
// Request
{
  "type": "request",
  "id": "req_001",
  "payload": {
    "action": "process_document",
    "document": { /* document data */ },
    "profile": "sec_10k",
    "stream_results": true
  }
}

// Progress Events
{
  "type": "event",
  "id": "req_001",
  "payload": {
    "event": "processing_progress",
    "progress": 0.25,
    "stage": "layout_detection"
  }
}

// Partial Results
{
  "type": "event",
  "id": "req_001", 
  "payload": {
    "event": "partial_result",
    "section": "company_info",
    "data": {
      "company_name": "ACME Corp",
      "cik": "0001234567"
    }
  }
}

// Final Result
{
  "type": "response",
  "id": "req_001",
  "payload": {
    "status": "completed",
    "result": { /* complete extraction result */ }
  }
}
```

## WASM/JavaScript API Design

### Browser Integration
```html
<script src="https://cdn.neuraldocflow.com/neuraldocflow.js"></script>
```

```javascript
// Initialize
const neuralDoc = await NeuralDocFlow.init({
  apiKey: 'your_api_key',
  wasmUrl: 'https://cdn.neuraldocflow.com/neuraldocflow.wasm'
});

// Process locally (lightweight operations)
const result = await neuralDoc.processLocal({
  file: uploadedFile,
  profile: 'simple_extraction',
  clientSideOnly: true
});

// Process via API (complex operations)
const result = await neuralDoc.processRemote({
  file: uploadedFile,
  profile: 'sec_10k',
  onProgress: (progress) => updateUI(progress)
});

// Streaming interface
const stream = neuralDoc.createStream({
  profile: 'real_time_analysis'
});

stream.on('result', (data) => {
  updateResults(data);
});

document.getElementById('fileInput').addEventListener('change', (e) => {
  stream.processFile(e.target.files[0]);
});
```

### Web Worker Integration
```javascript
// main.js
const worker = new Worker('neuraldocflow-worker.js');

worker.postMessage({
  action: 'process',
  file: fileData,
  profile: 'sec_10k'
});

worker.onmessage = (event) => {
  const { progress, result, error } = event.data;
  if (result) updateUI(result);
};

// neuraldocflow-worker.js
importScripts('https://cdn.neuraldocflow.com/neuraldocflow-worker.js');

self.onmessage = async (event) => {
  const { action, file, profile } = event.data;
  
  if (action === 'process') {
    const neuralDoc = await NeuralDocFlow.initWorker();
    
    const result = await neuralDoc.process(file, profile, {
      onProgress: (progress) => {
        self.postMessage({ progress });
      }
    });
    
    self.postMessage({ result });
  }
};
```

## Python SDK Design

### Installation
```bash
pip install neuraldocflow
```

### Basic Usage
```python
from neuraldocflow import NeuralDocFlow, ProcessingProfile

# Initialize client
client = NeuralDocFlow(api_key="your_api_key")

# Process single document
result = client.process_document(
    file_path="sec_filing.pdf",
    profile="sec_10k",
    async_mode=False
)

print(f"Company: {result.extracted_data['company_name']}")
print(f"Revenue: ${result.extracted_data['revenue']:,}")

# Async processing
job = client.process_document_async(
    file_path="large_document.pdf", 
    profile="comprehensive_analysis"
)

# Monitor progress
while not job.is_complete():
    print(f"Progress: {job.progress}%")
    time.sleep(1)

result = job.get_result()
```

### Batch Processing
```python
from neuraldocflow import BatchProcessor

# Process multiple documents
processor = BatchProcessor(
    client=client,
    parallelism=4,
    profile="sec_10k"
)

documents = [
    "file1.pdf",
    "file2.pdf", 
    "https://example.com/file3.pdf"
]

# Process with progress tracking
results = []
for result in processor.process_batch(documents):
    print(f"Processed: {result.document_id}")
    results.append(result)

# Aggregate results
aggregated = processor.aggregate_results(results, 
    aggregation_type="financial_summary"
)
```

### Streaming Processing
```python
from neuraldocflow import StreamProcessor

# Set up streaming processor
stream = StreamProcessor(
    client=client,
    profile="real_time_analysis"
)

# Process documents as they arrive
@stream.on_document_received
def handle_document(document_path):
    return {"priority": "high" if "urgent" in document_path else "normal"}

@stream.on_result_ready
def handle_result(result):
    # Store result in database
    db.store_extraction_result(result)
    
    # Send notification
    notify_stakeholders(result.summary)

# Start monitoring directory
stream.watch_directory("/incoming/documents/")
```

### Custom Profiles
```python
from neuraldocflow import ProfileBuilder

# Create custom extraction profile
profile = ProfileBuilder() \
    .document_type("medical_record") \
    .add_field("patient_name", required=True, 
               patterns=["Patient:?\\s+(.+)", "Name:?\\s+(.+)"]) \
    .add_field("diagnosis", required=True,
               semantic_query="What is the primary diagnosis?") \
    .add_table("medications", 
               identifiers=["Current Medications", "Drug List"]) \
    .quality_threshold(0.95) \
    .build()

# Register profile
client.register_profile("custom_medical", profile)

# Use custom profile
result = client.process_document(
    file_path="medical_record.pdf",
    profile="custom_medical"
)
```

## Error Handling and Status Codes

### HTTP Status Codes
- **200**: Success
- **202**: Accepted (async processing started)
- **400**: Bad Request (invalid parameters)
- **401**: Unauthorized (invalid API key)
- **402**: Payment Required (quota exceeded)
- **404**: Not Found (job/profile not found)
- **413**: Payload Too Large (file size limit)
- **422**: Unprocessable Entity (unsupported format)
- **429**: Too Many Requests (rate limited)
- **500**: Internal Server Error
- **503**: Service Unavailable (system overloaded)

### Error Response Format
```json
{
  "error": {
    "code": "EXTRACTION_FAILED",
    "message": "Unable to extract required fields",
    "details": {
      "confidence_score": 0.45,
      "failed_fields": ["revenue", "net_income"],
      "suggestions": [
        "Try using 'relaxed_extraction' mode",
        "Check if document is a different format than expected"
      ]
    },
    "recovery_options": [
      {
        "action": "retry_with_profile",
        "profile": "generic_financial"
      },
      {
        "action": "manual_field_specification",
        "endpoint": "/documents/manual_extract"
      }
    ]
  }
}
```

## Rate Limiting and Quotas

### Rate Limits
- **Free Tier**: 10 requests/minute, 100 documents/month
- **Pro Tier**: 100 requests/minute, 10,000 documents/month  
- **Enterprise**: Custom limits

### Quota Headers
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1640995200
X-Quota-Limit: 10000
X-Quota-Used: 2847
X-Quota-Reset: 1643673600
```

## Performance Optimization

### Caching Strategy
- **Profile Caching**: 24-hour TTL for profile configurations
- **Result Caching**: Cache based on document hash + profile version
- **Model Caching**: Intelligent model loading based on request patterns

### Request Optimization
- **Compression**: Gzip compression for all responses
- **CDN**: Global CDN for WASM and static assets
- **Connection Pooling**: HTTP/2 and connection reuse
- **Batch Optimization**: Automatic batching of similar requests

### Monitoring and Analytics
- **Performance Metrics**: Processing time, accuracy, throughput
- **Usage Analytics**: API usage patterns, popular profiles
- **Error Tracking**: Detailed error logging and alerting
- **Cost Tracking**: Per-request cost analysis for optimization