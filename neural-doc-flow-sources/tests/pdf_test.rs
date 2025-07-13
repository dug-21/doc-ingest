#[cfg(feature = "pdf")]
mod pdf_tests {
    use neural_doc_flow_sources::prelude::*;
    use neural_doc_flow_sources::PdfSource;
    use std::path::PathBuf;
    use tokio::fs;
    use uuid::Uuid;
    
    /// Create a minimal valid PDF for testing
    fn create_test_pdf() -> Vec<u8> {
        // This is a minimal valid PDF that says "Hello World"
        let pdf_content = b"%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> /MediaBox [0 0 612 792] /Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(Hello World) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000261 00000 n
0000000333 00000 n
trailer
<< /Size 6 /Root 1 0 R /Info << /Title (Test PDF) /Author (Test Suite) /CreationDate (D:20240101120000) >> >>
startxref
433
%%EOF";
        pdf_content.to_vec()
    }
    
    /// Create a corrupted PDF for error testing
    fn create_corrupted_pdf() -> Vec<u8> {
        b"Not a valid PDF file content".to_vec()
    }

    #[tokio::test]
    async fn test_pdf_source_creation() {
        let source = PdfSource::new();
        let metadata = source.metadata();
        
        assert_eq!(metadata.id, "pdf-source");
        assert_eq!(metadata.name, "PDF Document Source");
        assert!(metadata.capabilities.contains(&SourceCapability::Pdf));
        assert!(metadata.capabilities.contains(&SourceCapability::MetadataExtraction));
        assert!(metadata.file_extensions.contains(&"pdf".to_string()));
        assert!(metadata.mime_types.contains(&"application/pdf".to_string()));
    }

    #[tokio::test]
    async fn test_can_handle() {
        let source = PdfSource::new();
        
        // Test file extensions
        assert!(source.can_handle("document.pdf").await.unwrap());
        assert!(source.can_handle("Document.PDF").await.unwrap());
        assert!(source.can_handle("/path/to/file.pdf").await.unwrap());
        assert!(!source.can_handle("document.txt").await.unwrap());
        assert!(!source.can_handle("document.docx").await.unwrap());
        
        // Test edge cases
        assert!(!source.can_handle("").await.unwrap());
        assert!(!source.can_handle("pdf").await.unwrap());
        assert!(!source.can_handle(".pdf").await.unwrap());
    }

    #[tokio::test]
    async fn test_configure() {
        let mut source = PdfSource::new();
        let mut config = SourceConfig::default();
        
        config.add_option("extract_images", serde_json::json!(true));
        config.add_option("max_pages", serde_json::json!(100));
        
        assert!(source.configure(config.clone()).await.is_ok());
        
        let current_config = source.config();
        assert_eq!(current_config.options.get("extract_images"), config.options.get("extract_images"));
        assert_eq!(current_config.options.get("max_pages"), config.options.get("max_pages"));
    }

    #[tokio::test]
    async fn test_process_valid_pdf() {
        let source = PdfSource::new();
        let pdf_data = create_test_pdf();
        let source_path = "test.pdf".to_string();
        
        let result = source.process(&pdf_data, &source_path).await;
        assert!(result.is_ok());
        
        let document = result.unwrap();
        assert!(!document.content.is_empty());
        assert_eq!(document.source, source_path);
        assert_eq!(document.document_type, "application/pdf");
        assert_eq!(document.size, pdf_data.len());
        assert_eq!(document.processing_status, ProcessingStatus::Completed);
        
        // Check metadata
        let metadata = document.metadata.as_object().unwrap();
        assert!(metadata.contains_key("title"));
        assert!(metadata.contains_key("author"));
        assert!(metadata.contains_key("page_count"));
    }

    #[tokio::test]
    async fn test_process_corrupted_pdf() {
        let source = PdfSource::new();
        let pdf_data = create_corrupted_pdf();
        let source_path = "corrupted.pdf".to_string();
        
        let result = source.process(&pdf_data, &source_path).await;
        assert!(result.is_err());
        
        match result.unwrap_err() {
            SourceError::ParseError(msg) => {
                assert!(msg.contains("Failed to load PDF"));
            }
            _ => panic!("Expected ParseError"),
        }
    }

    #[tokio::test]
    async fn test_process_empty_data() {
        let source = PdfSource::new();
        let pdf_data = vec![];
        let source_path = "empty.pdf".to_string();
        
        let result = source.process(&pdf_data, &source_path).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_batch_process() {
        let source = PdfSource::new();
        let pdf_data = create_test_pdf();
        
        let files = vec![
            (pdf_data.clone(), "test1.pdf".to_string()),
            (pdf_data.clone(), "test2.pdf".to_string()),
            (create_corrupted_pdf(), "corrupted.pdf".to_string()),
        ];
        
        let results = source.batch_process(files).await.unwrap();
        
        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok());
        assert!(results[1].is_ok());
        assert!(results[2].is_err());
        
        // Verify successful documents
        let doc1 = results[0].as_ref().unwrap();
        assert_eq!(doc1.source, "test1.pdf");
        
        let doc2 = results[1].as_ref().unwrap();
        assert_eq!(doc2.source, "test2.pdf");
    }

    #[tokio::test]
    async fn test_validate() {
        let source = PdfSource::new();
        
        // Valid PDF
        let valid_pdf = create_test_pdf();
        assert!(source.validate(&valid_pdf).await.unwrap());
        
        // Invalid PDF
        let invalid_pdf = create_corrupted_pdf();
        assert!(!source.validate(&invalid_pdf).await.unwrap());
        
        // Empty data
        assert!(!source.validate(&[]).await.unwrap());
        
        // Too small to be valid PDF
        assert!(!source.validate(b"123").await.unwrap());
    }

    #[tokio::test]
    async fn test_capabilities() {
        let source = PdfSource::new();
        let caps = source.capabilities();
        
        assert!(caps.contains(&SourceCapability::Pdf));
        assert!(caps.contains(&SourceCapability::MetadataExtraction));
        assert_eq!(caps.len(), 2);
    }

    #[tokio::test]
    async fn test_metadata_extraction() {
        let source = PdfSource::new();
        let pdf_data = create_test_pdf();
        let source_path = "metadata_test.pdf".to_string();
        
        let document = source.process(&pdf_data, &source_path).await.unwrap();
        let metadata = document.metadata.as_object().unwrap();
        
        // Check extracted metadata
        assert_eq!(metadata.get("title").unwrap().as_str().unwrap(), "Test PDF");
        assert_eq!(metadata.get("author").unwrap().as_str().unwrap(), "Test Suite");
        assert!(metadata.contains_key("creation_date"));
        assert_eq!(metadata.get("page_count").unwrap().as_str().unwrap(), "1");
    }

    #[tokio::test]
    async fn test_clone() {
        let source = PdfSource::new();
        let cloned = source.clone();
        
        assert_eq!(source.metadata().id, cloned.metadata().id);
        assert_eq!(source.metadata().name, cloned.metadata().name);
        assert_eq!(source.capabilities(), cloned.capabilities());
    }

    #[tokio::test]
    async fn test_file_path_handling() {
        let source = PdfSource::new();
        
        // Test various file path formats
        assert!(source.can_handle("simple.pdf").await.unwrap());
        assert!(source.can_handle("./relative/path/file.pdf").await.unwrap());
        assert!(source.can_handle("/absolute/path/file.pdf").await.unwrap());
        assert!(source.can_handle("C:\\Windows\\path\\file.pdf").await.unwrap());
        assert!(source.can_handle("file-with-dashes.pdf").await.unwrap());
        assert!(source.can_handle("file_with_underscores.pdf").await.unwrap());
        assert!(source.can_handle("file.name.with.dots.pdf").await.unwrap());
        
        // Test invalid extensions
        assert!(!source.can_handle("file.pdf.txt").await.unwrap());
        assert!(!source.can_handle("file.pdfx").await.unwrap());
        assert!(!source.can_handle("filepdf").await.unwrap());
    }

    #[tokio::test]
    async fn test_large_pdf_simulation() {
        let source = PdfSource::new();
        
        // Create a larger PDF-like data (still valid header)
        let mut large_pdf = b"%PDF-1.4\n".to_vec();
        large_pdf.extend(vec![0u8; 1024 * 1024]); // 1MB of zeros
        
        // This will fail parsing but should handle gracefully
        let result = source.process(&large_pdf, "large.pdf").await;
        assert!(result.is_err());
    }
}
