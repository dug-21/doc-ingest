# API Alignment Guide for Neural Document Flow

## Core Type Mismatches and Missing Methods

### 1. Document Type Missing Methods

**Current State:**
- `Document` struct exists in `neural-doc-flow-core/src/types.rs`
- Only has `new()`, `add_processing_event()`, and `latest_event()` methods

**Missing Methods Required by API:**
```rust
// Missing in neural-doc-flow-api/src/handlers/process.rs:99
Document::from_bytes(content: Vec<u8>, filename: String) -> Result<Document>
```

**Solution:**
Add to `impl Document`:
```rust
pub fn from_bytes(content: Vec<u8>, filename: String) -> Result<Self> {
    let mime_type = Self::detect_mime_type(&filename, &content);
    let mut doc = Self::new(filename.clone(), mime_type);
    doc.raw_content = content;
    doc.source_type = DocumentSourceType::Memory;
    doc.metadata.size = Some(doc.raw_content.len() as u64);
    Ok(doc)
}

fn detect_mime_type(filename: &str, content: &[u8]) -> String {
    // Implementation based on file extension and magic bytes
}
```

### 2. ProcessingResult Type Mismatch

**Current State:**
- Core defines `ProcessingResultData<T>` generic struct
- API expects specific `ProcessingResult` type

**Missing Type:**
```rust
// Expected by API but not defined
pub type ProcessingResult = ProcessingResultData<Document>;
```

**Solution:**
Add to `neural-doc-flow-core/src/types.rs`:
```rust
pub type ProcessingResult = ProcessingResultData<Document>;
```

### 3. DocumentProcessor Trait Requirements

**Current State:**
- API expects a `DocumentProcessor` trait
- Core only has `ProcessingResultData`

**Missing Trait:**
```rust
#[async_trait]
pub trait DocumentProcessor: Send + Sync {
    async fn process(&self, document: Document) -> Result<ProcessingResult>;
}
```

**Solution:**
Add to `neural-doc-flow-core/src/traits/mod.rs`:
```rust
use async_trait::async_trait;
use crate::{Document, ProcessingResult, Result};

#[async_trait]
pub trait DocumentProcessor: Send + Sync {
    async fn process(&self, document: Document) -> Result<ProcessingResult>;
}
```

### 4. Error Type Alignment

**Current State:**
- Core defines `NeuralDocFlowError` 
- API uses `anyhow::Result`
- Outputs use `neural_doc_flow_core::Result`

**Issue:**
- Inconsistent error handling across modules
- Missing conversions between error types

**Solution:**
Ensure `neural_doc_flow_core::Result` is properly exported:
```rust
// In neural-doc-flow-core/src/lib.rs
pub type Result<T> = std::result::Result<T, NeuralDocFlowError>;
```

### 5. Output Trait Requirements

**Current State:**
- Outputs module correctly uses `Document` from core
- All output implementations work with core types

**Status:** ✅ Aligned correctly

### 6. Processor Integration

**Current State:**
- Processors import `neural_doc_flow_core::Document`
- Some processors have custom result types

**Issues:**
- Missing `from_bytes` method blocks API usage
- Processor trait not standardized

**Solution:**
Standardize processor interface:
```rust
// In neural-doc-flow-processors/src/lib.rs
pub use neural_doc_flow_core::{Document, ProcessingResult, DocumentProcessor};

// Implement DocumentProcessor for all processor types
```

## API Method Mapping

### Document Creation Flow
```
API Request → Document::from_bytes() → Document instance
                       ↓
               Missing method!
```

### Processing Flow
```
Document → processor.process() → ProcessingResult
             ↓                        ↓
    Uses DocumentProcessor     Missing type alias!
```

### Output Generation Flow
```
Document → OutputManager → generate() → File/Bytes
               ↓
        ✅ Works correctly!
```

## Implementation Priority

1. **Critical** - Add `Document::from_bytes()` method
2. **Critical** - Add `ProcessingResult` type alias
3. **Critical** - Add `DocumentProcessor` trait
4. **High** - Standardize error types across modules
5. **Medium** - Add helper methods for Document manipulation
6. **Low** - Add convenience methods for testing

## Module Dependencies

```
neural-doc-flow-api
    ↓ depends on
neural-doc-flow-core (needs from_bytes, ProcessingResult)
    ↓ used by
neural-doc-flow-outputs (✅ working)
neural-doc-flow-processors (needs DocumentProcessor trait)
```

## Testing Requirements

After implementing fixes:
1. Test `Document::from_bytes` with various file types
2. Test `DocumentProcessor` trait with mock implementations
3. Test API endpoint with real document processing
4. Test error propagation across module boundaries