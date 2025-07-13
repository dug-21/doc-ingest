# NeuralDocFlow MCP Interface Specification

## Overview

The Model Context Protocol (MCP) interface for NeuralDocFlow provides a standardized way for LLM agents and applications to interact with the document processing system. This specification defines tools, resources, and communication patterns for intelligent document processing orchestration.

## Core MCP Tools

### Document Processing Tools

#### `neuraldocflow__process_document`
**Purpose**: Process a single document through the neural pipeline
```json
{
  "name": "neuraldocflow__process_document",
  "description": "Process a document using NeuralDocFlow's neural engine",
  "inputSchema": {
    "type": "object",
    "properties": {
      "source": {
        "type": "string",
        "description": "Document source (file path, URL, or base64)"
      },
      "format": {
        "type": "string",
        "enum": ["pdf", "docx", "html", "txt"],
        "description": "Document format"
      },
      "profile": {
        "type": "string",
        "description": "Processing profile name (e.g., 'sec_10k', 'medical_record')"
      },
      "output_format": {
        "type": "string",
        "enum": ["json", "yaml", "xml", "structured"],
        "default": "json"
      },
      "extraction_mode": {
        "type": "string",
        "enum": ["full", "targeted", "streaming"],
        "default": "full"
      },
      "quality_threshold": {
        "type": "number",
        "minimum": 0.0,
        "maximum": 1.0,
        "default": 0.9
      }
    },
    "required": ["source", "format"]
  }
}
```

#### `neuraldocflow__batch_process`
**Purpose**: Process multiple documents in parallel
```json
{
  "name": "neuraldocflow__batch_process",
  "description": "Process multiple documents in batch mode",
  "inputSchema": {
    "type": "object",
    "properties": {
      "documents": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "id": {"type": "string"},
            "source": {"type": "string"},
            "format": {"type": "string"},
            "profile": {"type": "string"}
          }
        }
      },
      "parallelism": {
        "type": "integer",
        "default": 4,
        "description": "Number of parallel processing agents"
      },
      "output_collection": {
        "type": "string",
        "enum": ["individual", "merged", "indexed"],
        "default": "individual"
      }
    }
  }
}
```

#### `neuraldocflow__stream_process`
**Purpose**: Set up streaming document processing
```json
{
  "name": "neuraldocflow__stream_process",
  "description": "Start streaming document processing pipeline",
  "inputSchema": {
    "type": "object",
    "properties": {
      "stream_config": {
        "type": "object",
        "properties": {
          "source_type": {
            "type": "string",
            "enum": ["filesystem", "s3", "url", "kafka", "websocket"]
          },
          "source_location": {"type": "string"},
          "buffer_size": {"type": "integer", "default": 10},
          "processing_delay": {"type": "integer", "default": 100}
        }
      },
      "output_stream": {
        "type": "object",
        "properties": {
          "destination": {"type": "string"},
          "format": {"type": "string"},
          "chunking": {"type": "boolean", "default": true}
        }
      }
    }
  }
}
```

### Swarm Coordination Tools

#### `neuraldocflow__swarm_spawn`
**Purpose**: Create specialized document processing swarm
```json
{
  "name": "neuraldocflow__swarm_spawn",
  "description": "Spawn a swarm of agents for document processing",
  "inputSchema": {
    "type": "object",
    "properties": {
      "swarm_type": {
        "type": "string",
        "enum": ["extraction", "analysis", "validation", "hybrid"]
      },
      "agent_count": {"type": "integer", "default": 4},
      "specializations": {
        "type": "array",
        "items": {
          "type": "string",
          "enum": ["pdf_parser", "table_extractor", "text_analyzer", "validator", "coordinator"]
        }
      },
      "coordination_strategy": {
        "type": "string",
        "enum": ["hierarchical", "mesh", "pipeline"],
        "default": "hierarchical"
      }
    }
  }
}
```

#### `neuraldocflow__swarm_status`
**Purpose**: Monitor swarm health and progress
```json
{
  "name": "neuraldocflow__swarm_status",
  "description": "Get status of document processing swarm",
  "inputSchema": {
    "type": "object",
    "properties": {
      "swarm_id": {"type": "string"},
      "detail_level": {
        "type": "string",
        "enum": ["summary", "detailed", "diagnostic"],
        "default": "summary"
      }
    }
  }
}
```

### Configuration and Management Tools

#### `neuraldocflow__profile_create`
**Purpose**: Create custom extraction profiles
```json
{
  "name": "neuraldocflow__profile_create",
  "description": "Create a custom document processing profile",
  "inputSchema": {
    "type": "object",
    "properties": {
      "profile_name": {"type": "string"},
      "document_type": {"type": "string"},
      "extraction_rules": {
        "type": "object",
        "description": "YAML-formatted extraction rules"
      },
      "quality_requirements": {
        "type": "object",
        "properties": {
          "min_confidence": {"type": "number"},
          "validation_rules": {"type": "array"}
        }
      }
    }
  }
}
```

#### `neuraldocflow__neural_train`
**Purpose**: Train neural models on document patterns
```json
{
  "name": "neuraldocflow__neural_train",
  "description": "Train neural models for document understanding",
  "inputSchema": {
    "type": "object",
    "properties": {
      "training_data": {
        "type": "object",
        "properties": {
          "documents": {"type": "array"},
          "labels": {"type": "array"},
          "validation_split": {"type": "number", "default": 0.2}
        }
      },
      "model_type": {
        "type": "string",
        "enum": ["layout_detection", "table_extraction", "entity_recognition", "classification"]
      },
      "training_config": {
        "type": "object",
        "properties": {
          "epochs": {"type": "integer", "default": 10},
          "learning_rate": {"type": "number", "default": 0.001},
          "batch_size": {"type": "integer", "default": 8}
        }
      }
    }
  }
}
```

## MCP Resources

### Document Profiles
- **URI Pattern**: `neuraldocflow://profiles/{profile_name}`
- **Description**: Access document processing profiles
- **Content**: YAML configuration for document extraction rules

### Processing Results
- **URI Pattern**: `neuraldocflow://results/{job_id}`
- **Description**: Access processing results and status
- **Content**: JSON extraction results with metadata

### Model Registry
- **URI Pattern**: `neuraldocflow://models/{model_type}/{model_name}`
- **Description**: Access neural model information
- **Content**: Model metadata, performance metrics, and deployment status

### Swarm State
- **URI Pattern**: `neuraldocflow://swarms/{swarm_id}`
- **Description**: Access swarm coordination state
- **Content**: Agent status, task distribution, and performance metrics

## Integration Patterns

### LLM Agent Workflow
```python
# 1. Initialize document processing
result = await mcp_client.call_tool("neuraldocflow__process_document", {
    "source": "sec_filing.pdf",
    "format": "pdf",
    "profile": "sec_10k"
})

# 2. Monitor processing
status = await mcp_client.call_tool("neuraldocflow__swarm_status", {
    "swarm_id": result["swarm_id"]
})

# 3. Access results
document_data = await mcp_client.read_resource(f"neuraldocflow://results/{result['job_id']}")
```

### Human API Integration
```javascript
// REST API wrapper around MCP tools
const neuralDocFlow = new NeuralDocFlowAPI({
    mcpServer: 'neuraldocflow-mcp',
    endpoint: 'http://localhost:8080'
});

const result = await neuralDocFlow.processDocument({
    file: uploadedFile,
    profile: 'sec_10k',
    options: { qualityThreshold: 0.95 }
});
```

### Streaming Integration
```rust
// WebSocket streaming for real-time processing
use neuraldocflow_client::StreamingClient;

let mut client = StreamingClient::connect("ws://localhost:8080/stream").await?;

client.subscribe("document_results").await?;
while let Some(result) = client.next().await {
    match result {
        StreamEvent::ProcessingComplete(data) => {
            // Handle completed document
        }
        StreamEvent::PartialResult(chunk) => {
            // Handle streaming chunk
        }
    }
}
```

## Quality and Validation

### Response Validation
All MCP tool responses include:
- **confidence_score**: Overall processing confidence (0.0-1.0)
- **validation_results**: Array of validation check results
- **processing_metadata**: Timing, resource usage, model versions
- **error_details**: Detailed error information if processing fails

### Error Handling
```json
{
  "error": {
    "code": "EXTRACTION_FAILED",
    "message": "Failed to extract financial tables",
    "details": {
      "confidence_score": 0.45,
      "failed_sections": ["income_statement"],
      "suggestions": ["Try 'relaxed_table_detection' mode"]
    },
    "recovery_options": [
      "retry_with_different_profile",
      "manual_section_specification"
    ]
  }
}
```

## Performance Considerations

### Asynchronous Processing
- All processing tools return immediately with job IDs
- Use status tools to monitor progress
- Results available via resource URIs when complete

### Caching Strategy
- Profile configurations cached for fast access
- Neural model outputs cached based on document hash
- Intermediate processing results cached for iterative refinement

### Resource Management
- Automatic cleanup of temporary processing files
- Configurable result retention policies
- Memory-efficient streaming for large documents

## Security Model

### Access Control
- API key authentication for external clients
- Role-based access to processing profiles
- Audit logging for all processing operations

### Data Privacy
- Configurable data retention policies
- Option to process without storing intermediate results
- Support for on-premises deployment

### Input Validation
- Strict input sanitization for all document sources
- Malware scanning for uploaded documents
- Resource limits to prevent abuse