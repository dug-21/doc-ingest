use neural_doc_flow_sources::prelude::*;
use neural_doc_flow_sources::{SourceManager, PdfSource, TextSource};
use std::sync::Arc;
use async_trait::async_trait;

// Mock source for testing
#[derive(Debug, Clone)]
struct MockSource {
    id: String,
    extensions: Vec<String>,
    fail_on_process: bool,
}

#[async_trait]
impl BaseDocumentSource for MockSource {
    async fn process(&self, data: &[u8], source_path: &str) -> SourceResult<Document> {
        if self.fail_on_process {
            return Err(SourceError::ProcessingError("Mock processing failed".to_string()));
        }
        
        Ok(Document {
            id: uuid::Uuid::new_v4(),
            content: String::from_utf8_lossy(data).to_string(),
            metadata: serde_json::json!({
                "source_id": self.id,
                "path": source_path,
            }),
            source: source_path.to_string(),
            document_type: "text/plain".to_string(),
            size: data.len(),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            hash: format!("{:x}", md5::compute(data)),
            tags: vec!["mock".to_string()],
            relationships: vec![],
            processing_status: ProcessingStatus::Completed,
        })
    }

    async fn validate(&self, data: &[u8]) -> SourceResult<bool> {
        Ok(data.len() > 0)
    }

    async fn can_handle(&self, path: &str) -> SourceResult<bool> {
        let lower_path = path.to_lowercase();
        Ok(self.extensions.iter().any(|ext| lower_path.ends_with(&format!(".{}", ext))))
    }

    async fn configure(&mut self, _config: SourceConfig) -> SourceResult<()> {
        Ok(())
    }

    fn metadata(&self) -> &SourceMetadata {
        // This is a bit of a hack for testing - in real impl this would be a field
        Box::leak(Box::new(SourceMetadata {
            id: self.id.clone(),
            name: format!("Mock Source {}", self.id),
            version: "1.0.0".to_string(),
            capabilities: vec![SourceCapability::Text],
            file_extensions: self.extensions.clone(),
            mime_types: vec!["text/plain".to_string()],
        }))
    }

    fn config(&self) -> &SourceConfig {
        Box::leak(Box::new(SourceConfig::default()))
    }

    fn capabilities(&self) -> &[SourceCapability] {
        &[SourceCapability::Text]
    }

    async fn batch_process(&self, files: Vec<(Vec<u8>, String)>) -> SourceResult<Vec<SourceResult<Document>>> {
        let mut results = Vec::new();
        for (data, path) in files {
            results.push(self.process(&data, &path).await);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod manager_tests {
    use super::*;

    #[tokio::test]
    async fn test_manager_creation() {
        let manager = SourceManager::new();
        let sources = manager.list_sources().await;
        assert_eq!(sources.len(), 0);
    }

    #[tokio::test]
    async fn test_register_source() {
        let manager = SourceManager::new();
        let source = Arc::new(MockSource {
            id: "mock1".to_string(),
            extensions: vec!["mock".to_string()],
            fail_on_process: false,
        });
        
        assert!(manager.register_source(source).await.is_ok());
        
        let sources = manager.list_sources().await;
        assert_eq!(sources.len(), 1);
        assert!(sources.contains(&"mock1".to_string()));
    }

    #[tokio::test]
    async fn test_register_duplicate_source() {
        let manager = SourceManager::new();
        let source1 = Arc::new(MockSource {
            id: "mock1".to_string(),
            extensions: vec!["mock".to_string()],
            fail_on_process: false,
        });
        let source2 = Arc::new(MockSource {
            id: "mock1".to_string(), // Same ID
            extensions: vec!["test".to_string()],
            fail_on_process: false,
        });
        
        assert!(manager.register_source(source1).await.is_ok());
        let result = manager.register_source(source2).await;
        
        assert!(result.is_err());
        match result.unwrap_err() {
            SourceError::ConfigurationError(msg) => {
                assert!(msg.contains("already registered"));
            }
            _ => panic!("Expected ConfigurationError"),
        }
    }

    #[tokio::test]
    async fn test_find_source() {
        let manager = SourceManager::new();
        let source = Arc::new(MockSource {
            id: "mock1".to_string(),
            extensions: vec!["mock".to_string()],
            fail_on_process: false,
        });
        
        manager.register_source(source).await.unwrap();
        
        let found = manager.find_source("mock1").await;
        assert!(found.is_some());
        assert_eq!(found.unwrap().metadata().id, "mock1");
        
        let not_found = manager.find_source("nonexistent").await;
        assert!(not_found.is_none());
    }

    #[tokio::test]
    async fn test_find_sources_by_extension() {
        let manager = SourceManager::new();
        
        // Register multiple sources
        let source1 = Arc::new(MockSource {
            id: "mock1".to_string(),
            extensions: vec!["txt".to_string(), "text".to_string()],
            fail_on_process: false,
        });
        let source2 = Arc::new(MockSource {
            id: "mock2".to_string(),
            extensions: vec!["txt".to_string(), "log".to_string()],
            fail_on_process: false,
        });
        let source3 = Arc::new(MockSource {
            id: "mock3".to_string(),
            extensions: vec!["pdf".to_string()],
            fail_on_process: false,
        });
        
        manager.register_source(source1).await.unwrap();
        manager.register_source(source2).await.unwrap();
        manager.register_source(source3).await.unwrap();
        
        // Find by extension
        let txt_sources = manager.find_sources_by_extension("txt").await;
        assert_eq!(txt_sources.len(), 2);
        
        let pdf_sources = manager.find_sources_by_extension("pdf").await;
        assert_eq!(pdf_sources.len(), 1);
        assert_eq!(pdf_sources[0].metadata().id, "mock3");
        
        let none_sources = manager.find_sources_by_extension("xyz").await;
        assert_eq!(none_sources.len(), 0);
    }

    #[tokio::test]
    async fn test_find_sources_by_capability() {
        let manager = SourceManager::new();
        
        let source = Arc::new(MockSource {
            id: "mock1".to_string(),
            extensions: vec!["txt".to_string()],
            fail_on_process: false,
        });
        
        manager.register_source(source).await.unwrap();
        
        let text_sources = manager.find_sources_by_capability(SourceCapability::Text).await;
        assert_eq!(text_sources.len(), 1);
        
        let pdf_sources = manager.find_sources_by_capability(SourceCapability::Pdf).await;
        assert_eq!(pdf_sources.len(), 0);
    }

    #[tokio::test]
    async fn test_process_with_auto_detection() {
        let manager = SourceManager::new();
        
        // Register sources
        let txt_source = Arc::new(MockSource {
            id: "txt-source".to_string(),
            extensions: vec!["txt".to_string()],
            fail_on_process: false,
        });
        let pdf_source = Arc::new(MockSource {
            id: "pdf-source".to_string(),
            extensions: vec!["pdf".to_string()],
            fail_on_process: false,
        });
        
        manager.register_source(txt_source).await.unwrap();
        manager.register_source(pdf_source).await.unwrap();
        
        // Process text file
        let text_data = b"Hello, world!";
        let result = manager.process(text_data, "test.txt").await;
        assert!(result.is_ok());
        
        let document = result.unwrap();
        assert_eq!(document.content, "Hello, world!");
        let metadata = document.metadata.as_object().unwrap();
        assert_eq!(metadata.get("source_id").unwrap().as_str().unwrap(), "txt-source");
    }

    #[tokio::test]
    async fn test_process_no_suitable_source() {
        let manager = SourceManager::new();
        
        // Register only txt source
        let source = Arc::new(MockSource {
            id: "txt-source".to_string(),
            extensions: vec!["txt".to_string()],
            fail_on_process: false,
        });
        manager.register_source(source).await.unwrap();
        
        // Try to process unsupported file
        let result = manager.process(b"data", "file.xyz").await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ProcessingError::SourceError(msg) => {
                assert!(msg.contains("No suitable source found"));
            }
            _ => panic!("Expected SourceError"),
        }
    }

    #[tokio::test]
    async fn test_process_with_specific_source() {
        let manager = SourceManager::new();
        
        // Register multiple sources
        let source1 = Arc::new(MockSource {
            id: "source1".to_string(),
            extensions: vec!["txt".to_string()],
            fail_on_process: false,
        });
        let source2 = Arc::new(MockSource {
            id: "source2".to_string(),
            extensions: vec!["txt".to_string()],
            fail_on_process: false,
        });
        
        manager.register_source(source1).await.unwrap();
        manager.register_source(source2).await.unwrap();
        
        // Process with specific source
        let data = b"test data";
        let result = manager.process_with_source(data, "test.txt", "source2").await;
        assert!(result.is_ok());
        
        let document = result.unwrap();
        let metadata = document.metadata.as_object().unwrap();
        assert_eq!(metadata.get("source_id").unwrap().as_str().unwrap(), "source2");
    }

    #[tokio::test]
    async fn test_process_with_nonexistent_source() {
        let manager = SourceManager::new();
        
        let result = manager.process_with_source(b"data", "file.txt", "nonexistent").await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ProcessingError::SourceError(msg) => {
                assert!(msg.contains("Source not found"));
            }
            _ => panic!("Expected SourceError"),
        }
    }

    #[tokio::test]
    async fn test_batch_process() {
        let manager = SourceManager::new();
        
        // Register sources
        let txt_source = Arc::new(MockSource {
            id: "txt-source".to_string(),
            extensions: vec!["txt".to_string()],
            fail_on_process: false,
        });
        let pdf_source = Arc::new(MockSource {
            id: "pdf-source".to_string(),
            extensions: vec!["pdf".to_string()],
            fail_on_process: false,
        });
        
        manager.register_source(txt_source).await.unwrap();
        manager.register_source(pdf_source).await.unwrap();
        
        // Batch process
        let files = vec![
            (b"text1".to_vec(), "file1.txt".to_string()),
            (b"text2".to_vec(), "file2.txt".to_string()),
            (b"pdf data".to_vec(), "file.pdf".to_string()),
            (b"unknown".to_vec(), "file.xyz".to_string()), // Should fail
        ];
        
        let results = manager.batch_process(files).await.unwrap();
        assert_eq!(results.len(), 4);
        
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());
        assert!(results[2].is_ok());
        assert!(results[3].is_err()); // No source for .xyz
    }

    #[tokio::test]
    async fn test_source_statistics() {
        let manager = SourceManager::new();
        
        // Register sources
        let source1 = Arc::new(MockSource {
            id: "source1".to_string(),
            extensions: vec!["txt".to_string(), "log".to_string()],
            fail_on_process: false,
        });
        let source2 = Arc::new(MockSource {
            id: "source2".to_string(),
            extensions: vec!["pdf".to_string()],
            fail_on_process: false,
        });
        
        manager.register_source(source1).await.unwrap();
        manager.register_source(source2).await.unwrap();
        
        let stats = manager.get_statistics().await;
        assert_eq!(stats.total_sources, 2);
        assert_eq!(stats.total_extensions, 3); // txt, log, pdf
        assert_eq!(stats.sources_by_capability.get(&SourceCapability::Text), Some(&2));
    }

    #[tokio::test]
    #[cfg(all(feature = "pdf", feature = "text"))]
    async fn test_real_sources_integration() {
        let manager = SourceManager::new();
        
        // Register real sources
        manager.register_source(Arc::new(PdfSource::new())).await.unwrap();
        manager.register_source(Arc::new(TextSource::new())).await.unwrap();
        
        let sources = manager.list_sources().await;
        assert_eq!(sources.len(), 2);
        assert!(sources.contains(&"pdf-source".to_string()));
        assert!(sources.contains(&"text-source".to_string()));
        
        // Check extensions
        let pdf_sources = manager.find_sources_by_extension("pdf").await;
        assert_eq!(pdf_sources.len(), 1);
        
        let txt_sources = manager.find_sources_by_extension("txt").await;
        assert_eq!(txt_sources.len(), 1);
        
        let md_sources = manager.find_sources_by_extension("md").await;
        assert_eq!(md_sources.len(), 1); // TextSource handles markdown
    }

    #[tokio::test]
    async fn test_concurrent_registration() {
        let manager = SourceManager::new();
        
        // Create multiple sources
        let sources: Vec<Arc<dyn BaseDocumentSource>> = (0..10)
            .map(|i| {
                Arc::new(MockSource {
                    id: format!("source{}", i),
                    extensions: vec![format!("ext{}", i)],
                    fail_on_process: false,
                }) as Arc<dyn BaseDocumentSource>
            })
            .collect();
        
        // Register concurrently
        let handles: Vec<_> = sources
            .into_iter()
            .map(|source| {
                let mgr = manager.clone();
                tokio::spawn(async move {
                    mgr.register_source(source).await
                })
            })
            .collect();
        
        // Wait for all registrations
        for handle in handles {
            assert!(handle.await.unwrap().is_ok());
        }
        
        // Verify all sources registered
        let registered = manager.list_sources().await;
        assert_eq!(registered.len(), 10);
    }

    #[tokio::test]
    async fn test_error_handling_in_processing() {
        let manager = SourceManager::new();
        
        // Register a failing source
        let source = Arc::new(MockSource {
            id: "failing-source".to_string(),
            extensions: vec!["fail".to_string()],
            fail_on_process: true,
        });
        
        manager.register_source(source).await.unwrap();
        
        let result = manager.process(b"data", "test.fail").await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            ProcessingError::Custom(msg) => {
                assert!(msg.contains("Mock processing failed"));
            }
            _ => panic!("Expected processing error"),
        }
    }
}
