#[cfg(feature = "text")]
mod text_tests {
    use neural_doc_flow_sources::prelude::*;
    use neural_doc_flow_sources::TextSource;
    use uuid::Uuid;

    #[tokio::test]
    async fn test_text_source_creation() {
        let source = TextSource::new();
        let metadata = source.metadata();
        
        assert_eq!(metadata.id, "text-source");
        assert_eq!(metadata.name, "Text Document Source");
        assert!(metadata.capabilities.contains(&SourceCapability::Text));
        assert!(metadata.capabilities.contains(&SourceCapability::Encoding));
        
        // Check supported extensions
        assert!(metadata.file_extensions.contains(&"txt".to_string()));
        assert!(metadata.file_extensions.contains(&"text".to_string()));
        assert!(metadata.file_extensions.contains(&"md".to_string()));
        assert!(metadata.file_extensions.contains(&"markdown".to_string()));
        assert!(metadata.file_extensions.contains(&"rst".to_string()));
        assert!(metadata.file_extensions.contains(&"log".to_string()));
        
        // Check MIME types
        assert!(metadata.mime_types.contains(&"text/plain".to_string()));
        assert!(metadata.mime_types.contains(&"text/markdown".to_string()));
    }

    #[tokio::test]
    async fn test_can_handle() {
        let source = TextSource::new();
        
        // Supported extensions
        assert!(source.can_handle("document.txt").await.unwrap());
        assert!(source.can_handle("README.md").await.unwrap());
        assert!(source.can_handle("notes.markdown").await.unwrap());
        assert!(source.can_handle("documentation.rst").await.unwrap());
        assert!(source.can_handle("app.log").await.unwrap());
        assert!(source.can_handle("FILE.TXT").await.unwrap()); // Case insensitive
        
        // Unsupported extensions
        assert!(!source.can_handle("document.pdf").await.unwrap());
        assert!(!source.can_handle("image.png").await.unwrap());
        assert!(!source.can_handle("data.json").await.unwrap());
        
        // Edge cases
        assert!(!source.can_handle("").await.unwrap());
        assert!(!source.can_handle("txt").await.unwrap());
        assert!(!source.can_handle(".txt").await.unwrap());
    }

    #[tokio::test]
    async fn test_configure() {
        let mut source = TextSource::new();
        let mut config = SourceConfig::default();
        
        config.add_option("encoding", serde_json::json!("utf-16"));
        config.add_option("line_limit", serde_json::json!(1000));
        config.add_option("trim_whitespace", serde_json::json!(true));
        
        assert!(source.configure(config.clone()).await.is_ok());
        
        let current_config = source.config();
        assert_eq!(
            current_config.options.get("encoding"),
            Some(&serde_json::json!("utf-16"))
        );
    }

    #[tokio::test]
    async fn test_process_utf8_text() {
        let source = TextSource::new();
        let text = "Hello, World!\nThis is a test document.\nWith UTF-8 encoding.";
        let text_data = text.as_bytes();
        let source_path = "test.txt".to_string();
        
        let result = source.process(text_data, &source_path).await;
        assert!(result.is_ok());
        
        let document = result.unwrap();
        assert_eq!(document.content, text);
        assert_eq!(document.source, source_path);
        assert_eq!(document.document_type, "text/plain");
        assert_eq!(document.size, text_data.len());
        assert_eq!(document.processing_status, ProcessingStatus::Completed);
        
        // Check metadata
        let metadata = document.metadata.as_object().unwrap();
        assert_eq!(metadata.get("encoding").unwrap().as_str().unwrap(), "utf-8");
        assert_eq!(metadata.get("line_count").unwrap().as_u64().unwrap(), 3);
        assert!(metadata.contains_key("word_count"));
        assert!(metadata.contains_key("char_count"));
    }

    #[tokio::test]
    async fn test_process_utf8_with_bom() {
        let source = TextSource::new();
        let bom = &[0xEF, 0xBB, 0xBF]; // UTF-8 BOM
        let text = "Text with BOM";
        let mut text_data = bom.to_vec();
        text_data.extend_from_slice(text.as_bytes());
        
        let result = source.process(&text_data, "bom.txt").await;
        assert!(result.is_ok());
        
        let document = result.unwrap();
        // BOM should be stripped
        assert_eq!(document.content, text);
    }

    #[tokio::test]
    async fn test_process_empty_text() {
        let source = TextSource::new();
        let text_data = b"";
        
        let result = source.process(text_data, "empty.txt").await;
        assert!(result.is_ok());
        
        let document = result.unwrap();
        assert_eq!(document.content, "");
        assert_eq!(document.size, 0);
        
        let metadata = document.metadata.as_object().unwrap();
        assert_eq!(metadata.get("line_count").unwrap().as_u64().unwrap(), 0);
        assert_eq!(metadata.get("word_count").unwrap().as_u64().unwrap(), 0);
    }

    #[tokio::test]
    async fn test_process_special_characters() {
        let source = TextSource::new();
        let text = "Special chars: émojis 🎉🌟\nTabs:\t\there\nUnicode: 你好世界";
        let text_data = text.as_bytes();
        
        let result = source.process(text_data, "special.txt").await;
        assert!(result.is_ok());
        
        let document = result.unwrap();
        assert_eq!(document.content, text);
        
        let metadata = document.metadata.as_object().unwrap();
        assert_eq!(metadata.get("line_count").unwrap().as_u64().unwrap(), 3);
    }

    #[tokio::test]
    async fn test_process_large_text() {
        let source = TextSource::new();
        let line = "This is a line of text. ";
        let text = line.repeat(1000); // Create a larger text
        let text_data = text.as_bytes();
        
        let result = source.process(text_data, "large.txt").await;
        assert!(result.is_ok());
        
        let document = result.unwrap();
        assert_eq!(document.content.len(), text.len());
        assert_eq!(document.size, text_data.len());
    }

    #[tokio::test]
    async fn test_batch_process() {
        let source = TextSource::new();
        
        let files = vec![
            ("File 1 content".as_bytes().to_vec(), "file1.txt".to_string()),
            ("File 2 content\nWith newline".as_bytes().to_vec(), "file2.txt".to_string()),
            ("File 3 content".as_bytes().to_vec(), "file3.md".to_string()),
        ];
        
        let results = source.batch_process(files).await.unwrap();
        
        assert_eq!(results.len(), 3);
        assert!(results.iter().all(|r| r.is_ok()));
        
        assert_eq!(results[0].as_ref().unwrap().content, "File 1 content");
        assert_eq!(results[1].as_ref().unwrap().content, "File 2 content\nWith newline");
        assert_eq!(results[2].as_ref().unwrap().content, "File 3 content");
        assert_eq!(results[2].as_ref().unwrap().document_type, "text/markdown");
    }

    #[tokio::test]
    async fn test_validate() {
        let source = TextSource::new();
        
        // Valid UTF-8 text
        assert!(source.validate(b"Valid UTF-8 text").await.unwrap());
        assert!(source.validate(b"").await.unwrap()); // Empty is valid
        assert!(source.validate("Unicode: 测试".as_bytes()).await.unwrap());
        
        // Invalid UTF-8
        let invalid_utf8 = vec![0xFF, 0xFE, 0xFD];
        assert!(!source.validate(&invalid_utf8).await.unwrap());
        
        // Null bytes (might be binary)
        let with_null = b"Text\x00with\x00nulls";
        assert!(!source.validate(with_null).await.unwrap());
    }

    #[tokio::test]
    async fn test_markdown_processing() {
        let source = TextSource::new();
        let markdown = r#"# Header 1

## Header 2

This is a **bold** text and *italic* text.

- Item 1
- Item 2

```rust
fn main() {
    println!("Hello");
}
```
"#;
        
        let result = source.process(markdown.as_bytes(), "test.md").await;
        assert!(result.is_ok());
        
        let document = result.unwrap();
        assert_eq!(document.content, markdown);
        assert_eq!(document.document_type, "text/markdown");
        
        let metadata = document.metadata.as_object().unwrap();
        assert!(metadata.get("line_count").unwrap().as_u64().unwrap() > 10);
    }

    #[tokio::test]
    async fn test_log_file_processing() {
        let source = TextSource::new();
        let log_content = r#"2024-01-01 10:00:00 INFO Starting application
2024-01-01 10:00:01 DEBUG Loading configuration
2024-01-01 10:00:02 ERROR Failed to connect to database
2024-01-01 10:00:03 WARN Retrying connection
"#;
        
        let result = source.process(log_content.as_bytes(), "app.log").await;
        assert!(result.is_ok());
        
        let document = result.unwrap();
        assert_eq!(document.content, log_content);
        assert_eq!(document.document_type, "text/plain");
        
        let metadata = document.metadata.as_object().unwrap();
        assert_eq!(metadata.get("line_count").unwrap().as_u64().unwrap(), 4);
    }

    #[tokio::test]
    async fn test_different_line_endings() {
        let source = TextSource::new();
        
        // Unix line endings (LF)
        let unix_text = "Line 1\nLine 2\nLine 3";
        let unix_result = source.process(unix_text.as_bytes(), "unix.txt").await.unwrap();
        assert_eq!(unix_result.content, unix_text);
        
        // Windows line endings (CRLF)
        let windows_text = "Line 1\r\nLine 2\r\nLine 3";
        let windows_result = source.process(windows_text.as_bytes(), "windows.txt").await.unwrap();
        assert_eq!(windows_result.content, windows_text);
        
        // Mac Classic line endings (CR)
        let mac_text = "Line 1\rLine 2\rLine 3";
        let mac_result = source.process(mac_text.as_bytes(), "mac.txt").await.unwrap();
        assert_eq!(mac_result.content, mac_text);
    }

    #[tokio::test]
    async fn test_capabilities() {
        let source = TextSource::new();
        let caps = source.capabilities();
        
        assert!(caps.contains(&SourceCapability::Text));
        assert!(caps.contains(&SourceCapability::Encoding));
        assert_eq!(caps.len(), 2);
    }

    #[tokio::test]
    async fn test_clone() {
        let source = TextSource::new();
        let cloned = source.clone();
        
        assert_eq!(source.metadata().id, cloned.metadata().id);
        assert_eq!(source.metadata().name, cloned.metadata().name);
        assert_eq!(source.capabilities(), cloned.capabilities());
    }

    #[tokio::test]
    async fn test_file_extension_case_insensitive() {
        let source = TextSource::new();
        
        assert!(source.can_handle("file.TXT").await.unwrap());
        assert!(source.can_handle("file.Txt").await.unwrap());
        assert!(source.can_handle("file.MD").await.unwrap());
        assert!(source.can_handle("file.mD").await.unwrap());
        assert!(source.can_handle("file.LOG").await.unwrap());
    }

    #[tokio::test]
    async fn test_metadata_extraction() {
        let source = TextSource::new();
        let text = "Word1 word2 word3.\nAnother line here.\n\nAnd a third paragraph.";
        
        let document = source.process(text.as_bytes(), "meta.txt").await.unwrap();
        let metadata = document.metadata.as_object().unwrap();
        
        assert_eq!(metadata.get("encoding").unwrap().as_str().unwrap(), "utf-8");
        assert_eq!(metadata.get("line_count").unwrap().as_u64().unwrap(), 4); // Including empty line
        assert!(metadata.get("word_count").unwrap().as_u64().unwrap() >= 7);
        assert_eq!(metadata.get("char_count").unwrap().as_u64().unwrap(), text.len() as u64);
    }
}
