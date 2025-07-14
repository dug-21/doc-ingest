/// Common test utilities for WASM tests
use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestDocument {
    pub id: String,
    pub content: String,
    pub metadata: Option<TestMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestMetadata {
    pub source: String,
    pub timestamp: Option<String>,
    pub tags: Option<Vec<String>>,
}

impl TestDocument {
    pub fn new(id: &str, content: &str) -> Self {
        Self {
            id: id.to_string(),
            content: content.to_string(),
            metadata: None,
        }
    }

    pub fn with_metadata(mut self, source: &str) -> Self {
        self.metadata = Some(TestMetadata {
            source: source.to_string(),
            timestamp: None,
            tags: None,
        });
        self
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }
}

/// Generate test documents for batch processing
pub fn generate_test_documents(count: usize) -> Vec<TestDocument> {
    (0..count)
        .map(|i| {
            TestDocument::new(
                &format!("test-doc-{}", i),
                &format!("This is test document number {}", i),
            )
            .with_metadata("test-generator")
        })
        .collect()
}

/// Assert that a JsValue contains expected string
pub fn assert_js_contains(value: &JsValue, expected: &str) {
    let str_value = value.as_string()
        .or_else(|| {
            // Try to convert to string if not already
            js_sys::JSON::stringify(value)
                .ok()
                .and_then(|v| v.as_string())
        })
        .unwrap_or_else(|| panic!("Value is not a string: {:?}", value));
    
    assert!(
        str_value.contains(expected),
        "Expected '{}' to contain '{}', but it didn't",
        str_value,
        expected
    );
}

/// Create a test processor configuration
pub fn create_test_processor_config(name: &str, concurrent_tasks: u32, batch_size: u32) -> String {
    serde_json::json!({
        "name": name,
        "concurrent_tasks": concurrent_tasks,
        "batch_size": batch_size,
        "memory_limit_mb": 256,
        "timeout_ms": 30000,
    }).to_string()
}

/// Measure execution time in milliseconds
pub async fn measure_async<F, T>(f: F) -> (T, f64)
where
    F: std::future::Future<Output = T>,
{
    let start = js_sys::Date::now();
    let result = f.await;
    let elapsed = js_sys::Date::now() - start;
    (result, elapsed)
}

/// Assert that an async operation completes within a timeout
pub async fn assert_completes_within<F, T>(f: F, timeout_ms: f64) -> T
where
    F: std::future::Future<Output = T>,
{
    let (result, elapsed) = measure_async(f).await;
    assert!(
        elapsed <= timeout_ms,
        "Operation took {}ms, expected to complete within {}ms",
        elapsed,
        timeout_ms
    );
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_document_creation() {
        let doc = TestDocument::new("test-1", "Test content");
        assert_eq!(doc.id, "test-1");
        assert_eq!(doc.content, "Test content");
        assert!(doc.metadata.is_none());
    }

    #[wasm_bindgen_test]
    fn test_document_with_metadata() {
        let doc = TestDocument::new("test-2", "Test content")
            .with_metadata("unit-test");
        
        assert!(doc.metadata.is_some());
        assert_eq!(doc.metadata.unwrap().source, "unit-test");
    }

    #[wasm_bindgen_test]
    fn test_generate_documents() {
        let docs = generate_test_documents(5);
        assert_eq!(docs.len(), 5);
        
        for (i, doc) in docs.iter().enumerate() {
            assert_eq!(doc.id, format!("test-doc-{}", i));
            assert!(doc.content.contains(&i.to_string()));
        }
    }

    #[wasm_bindgen_test]
    fn test_json_serialization() {
        let doc = TestDocument::new("json-test", "JSON content");
        let json = doc.to_json().expect("Should serialize to JSON");
        
        assert!(json.contains("\"id\":\"json-test\""));
        assert!(json.contains("\"content\":\"JSON content\""));
    }
}