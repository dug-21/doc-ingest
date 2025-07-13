# API Documentation

## Autonomous Document Extraction Platform API

### Overview

The Autonomous Document Extraction Platform provides a RESTful API for processing documents from various sources using advanced neural networks and dynamic agent allocation.

### Base URL
```
https://api.doc-extract.com/v1
```

### Authentication

All API requests require authentication using JWT tokens:

```http
Authorization: Bearer <your-jwt-token>
```

### Rate Limiting

- **Standard Plan**: 1000 requests per hour
- **Premium Plan**: 10000 requests per hour
- **Enterprise Plan**: Unlimited

Rate limit headers are included in all responses:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

## API Endpoints

### Process Document

**POST** `/documents/process`

Process a document from various sources and extract relevant information.

#### Request Body

```json
{
  "source": {
    "type": "url|file|base64",
    "value": "source_value",
    "metadata": {
      "filename": "document.pdf",
      "content_type": "application/pdf"
    }
  },
  "options": {
    "extract_text": true,
    "extract_entities": true,
    "classify_content": true,
    "generate_summary": false,
    "language": "en"
  },
  "webhook_url": "https://your-app.com/webhook",
  "priority": "normal|high|urgent"
}
```

#### Response

```json
{
  "task_id": "task_abc123",
  "status": "queued|processing|completed|failed",
  "estimated_completion": "2024-01-01T12:00:00Z",
  "created_at": "2024-01-01T11:00:00Z"
}
```

#### Example

```bash
curl -X POST https://api.doc-extract.com/v1/documents/process \
  -H "Authorization: Bearer your-jwt-token" \
  -H "Content-Type: application/json" \
  -d '{
    "source": {
      "type": "url",
      "value": "https://example.com/document.pdf"
    },
    "options": {
      "extract_text": true,
      "extract_entities": true
    }
  }'
```

### Get Processing Status

**GET** `/documents/{task_id}/status`

Retrieve the current status of a document processing task.

#### Response

```json
{
  "task_id": "task_abc123",
  "status": "processing",
  "progress": 75,
  "stages": {
    "validation": "completed",
    "extraction": "completed", 
    "neural_analysis": "processing",
    "synthesis": "pending"
  },
  "estimated_completion": "2024-01-01T12:05:00Z",
  "created_at": "2024-01-01T11:00:00Z",
  "updated_at": "2024-01-01T11:45:00Z"
}
```

### Get Processing Results

**GET** `/documents/{task_id}/results`

Retrieve the results of a completed document processing task.

#### Response

```json
{
  "task_id": "task_abc123",
  "status": "completed",
  "results": {
    "extracted_text": "Document content here...",
    "classification": {
      "document_type": "research_paper",
      "confidence": 0.95,
      "categories": ["academic", "technology"]
    },
    "entities": [
      {
        "text": "Neural Networks",
        "type": "TECHNOLOGY",
        "confidence": 0.98,
        "start": 45,
        "end": 60
      }
    ],
    "summary": "Brief summary of the document...",
    "metadata": {
      "page_count": 10,
      "word_count": 5234,
      "language": "en",
      "processing_time": 3.2
    }
  },
  "completed_at": "2024-01-01T12:03:15Z"
}
```

### Validate Source

**POST** `/sources/validate`

Validate a document source before processing.

#### Request Body

```json
{
  "source": {
    "type": "url|file|base64",
    "value": "source_value"
  }
}
```

#### Response

```json
{
  "valid": true,
  "source_type": "url",
  "estimated_size": 2048576,
  "content_type": "application/pdf",
  "accessibility": "public",
  "warnings": []
}
```

### Health Check

**GET** `/health`

Check the overall health and status of the API.

#### Response

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 86400,
  "components": {
    "api_server": "healthy",
    "neural_pipeline": "healthy",
    "agent_pool": "healthy",
    "storage": "healthy"
  },
  "metrics": {
    "requests_per_minute": 150,
    "average_processing_time": 2.5,
    "success_rate": 0.997
  }
}
```

### System Metrics

**GET** `/metrics`

Retrieve system performance metrics (requires admin privileges).

#### Response

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "performance": {
    "requests_total": 10000,
    "requests_success": 9970,
    "requests_failed": 30,
    "average_response_time": 1.8,
    "p95_response_time": 4.2,
    "p99_response_time": 8.1
  },
  "resources": {
    "cpu_usage": 0.65,
    "memory_usage": 0.72,
    "disk_usage": 0.45,
    "active_agents": 8,
    "queue_size": 12
  },
  "neural_models": {
    "text_extraction_model": {
      "accuracy": 0.98,
      "latency": 0.5,
      "throughput": 200
    },
    "classification_model": {
      "accuracy": 0.95,
      "latency": 0.3,
      "throughput": 300
    }
  }
}
```

## WebSocket API

### Real-time Updates

Connect to receive real-time updates about document processing:

```javascript
const ws = new WebSocket('wss://api.doc-extract.com/v1/ws');

ws.onopen = function() {
  // Subscribe to task updates
  ws.send(JSON.stringify({
    action: 'subscribe',
    task_id: 'task_abc123'
  }));
};

ws.onmessage = function(event) {
  const update = JSON.parse(event.data);
  console.log('Task update:', update);
};
```

#### Update Message Format

```json
{
  "task_id": "task_abc123",
  "status": "processing",
  "stage": "neural_analysis",
  "progress": 75,
  "timestamp": "2024-01-01T11:45:00Z",
  "details": {
    "current_agent": "neural_processor_1",
    "estimated_remaining": 30
  }
}
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "INVALID_SOURCE",
    "message": "The provided source URL is not accessible",
    "details": {
      "source_url": "https://example.com/invalid",
      "http_status": 404
    },
    "suggestion": "Please verify the URL is correct and publicly accessible"
  },
  "request_id": "req_xyz789",
  "timestamp": "2024-01-01T11:00:00Z"
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_TOKEN` | 401 | Invalid or expired JWT token |
| `RATE_LIMIT_EXCEEDED` | 429 | Rate limit exceeded |
| `INVALID_SOURCE` | 400 | Source validation failed |
| `UNSUPPORTED_FORMAT` | 400 | Document format not supported |
| `SOURCE_TOO_LARGE` | 413 | Document exceeds size limits |
| `PROCESSING_FAILED` | 500 | Internal processing error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

## SDKs and Libraries

### Python SDK

```python
from doc_extract import DocumentProcessor

# Initialize client
client = DocumentProcessor(api_key="your-api-key")

# Process document
result = client.process_document(
    source_url="https://example.com/document.pdf",
    extract_text=True,
    extract_entities=True
)

print(f"Status: {result.status}")
print(f"Text: {result.extracted_text}")
```

### JavaScript SDK

```javascript
import { DocumentProcessor } from '@doc-extract/client';

const client = new DocumentProcessor({
  apiKey: 'your-api-key'
});

// Process document
const result = await client.processDocument({
  source: {
    type: 'url',
    value: 'https://example.com/document.pdf'
  },
  options: {
    extractText: true,
    extractEntities: true
  }
});

console.log('Results:', result);
```

### Rust SDK

```rust
use doc_extract::DocumentProcessor;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = DocumentProcessor::new("your-api-key")?;
    
    let result = client
        .process_document()
        .source_url("https://example.com/document.pdf")
        .extract_text(true)
        .extract_entities(true)
        .send()
        .await?;
    
    println!("Status: {:?}", result.status);
    Ok(())
}
```

## Rate Limits and Quotas

### Request Limits

| Plan | Requests/Hour | Concurrent Requests | Max File Size |
|------|---------------|-------------------|---------------|
| Free | 100 | 2 | 10 MB |
| Standard | 1,000 | 5 | 50 MB |
| Premium | 10,000 | 20 | 200 MB |
| Enterprise | Unlimited | 100 | 1 GB |

### Processing Limits

| Resource | Free | Standard | Premium | Enterprise |
|----------|------|----------|---------|------------|
| Processing Time | 30s | 60s | 300s | Unlimited |
| Neural Models | Basic | Standard | Advanced | All |
| Storage Duration | 24h | 7 days | 30 days | Custom |

## Security

### API Security Features

- **TLS 1.3**: All communication encrypted
- **JWT Authentication**: Secure token-based auth
- **Rate Limiting**: Prevent abuse and DoS
- **Input Validation**: Comprehensive input sanitization
- **Content Scanning**: Malware and virus detection
- **Audit Logging**: Complete request logging
- **CORS Support**: Configurable cross-origin policies

### Best Practices

1. **Store API keys securely** - Never commit keys to version control
2. **Use HTTPS only** - Never send requests over HTTP
3. **Validate responses** - Always check response status codes
4. **Handle rate limits** - Implement exponential backoff
5. **Monitor usage** - Track API usage and costs
6. **Rotate keys regularly** - Update API keys periodically

## Support and Resources

### Documentation Links

- [Getting Started Guide](./getting-started.md)
- [Integration Examples](./examples/)
- [Troubleshooting Guide](./troubleshooting.md)
- [API Reference](./reference/)

### Support Channels

- **Documentation**: https://docs.doc-extract.com
- **Support Portal**: https://support.doc-extract.com
- **Community Forum**: https://community.doc-extract.com
- **Status Page**: https://status.doc-extract.com

### Contact Information

- **Technical Support**: support@doc-extract.com
- **Sales Inquiries**: sales@doc-extract.com
- **Security Issues**: security@doc-extract.com