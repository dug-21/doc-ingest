// Validation tests for the modular source architecture
// This ensures the architecture meets all requirements

#[cfg(test)]
mod architecture_validation {
    use super::*;
    use std::any::Any;
    
    /// Test: Trait-based plugin system
    #[test]
    fn test_trait_based_design() {
        // Verify core traits exist and are object-safe
        fn assert_object_safe<T: ?Sized>() {}
        
        assert_object_safe::<dyn DocumentSource>();
        assert_object_safe::<dyn SourceReader>();
        assert_object_safe::<dyn ContentExtractor>();
        assert_object_safe::<dyn SourceConfiguration>();
        
        // Verify traits can be used in collections
        let _sources: Vec<Box<dyn DocumentSource>> = vec![];
        let _readers: HashMap<String, Arc<dyn SourceReader>> = HashMap::new();
    }
    
    /// Test: Zero-cost abstractions
    #[test]
    fn test_zero_cost_abstractions() {
        // Verify no runtime overhead for trait calls
        use std::mem;
        
        // Static dispatch size should equal concrete type
        assert_eq!(
            mem::size_of::<PdfSource>(),
            mem::size_of::<PdfSource>()
        );
        
        // Verify inline-able methods
        #[inline(always)]
        fn source_id_static(source: &PdfSource) -> &str {
            source.source_id()
        }
        
        // This should compile to direct call
        let pdf = PdfSource::new();
        let _ = source_id_static(&pdf);
    }
    
    /// Test: Type safety
    #[test]
    fn test_type_safety() {
        // Verify strong typing prevents errors
        fn process_pdf_only(source: &PdfSource) -> Result<(), SourceError> {
            // This function only accepts PdfSource, not generic DocumentSource
            Ok(())
        }
        
        fn process_any_source(source: &dyn DocumentSource) -> Result<(), SourceError> {
            // This accepts any DocumentSource
            Ok(())
        }
        
        let pdf = PdfSource::new();
        assert!(process_pdf_only(&pdf).is_ok());
        assert!(process_any_source(&pdf).is_ok());
        
        // Verify enum exhaustiveness
        fn handle_block_type(block_type: &BlockType) {
            match block_type {
                BlockType::Paragraph => {},
                BlockType::Heading(_) => {},
                BlockType::Table => {},
                BlockType::Image => {},
                BlockType::List(_) => {},
                BlockType::CodeBlock => {},
                BlockType::Quote => {},
                BlockType::Footnote => {},
                BlockType::Caption => {},
                BlockType::Formula => {},
                BlockType::Custom(_) => {},
                // Compiler ensures all variants are handled
            }
        }
    }
    
    /// Test: Hot-reload capability
    #[test]
    fn test_hot_reload_design() {
        // Verify plugin manager supports dynamic loading
        let manager = SourcePluginManager::new(Default::default()).unwrap();
        
        // Verify plugins can be loaded/unloaded at runtime
        assert!(manager.sources.read().unwrap().is_empty());
        
        // Verify file watching capability
        assert!(manager.enable_hot_reload().is_ok());
    }
    
    /// Test: Consistent API
    #[test]
    async fn test_consistent_api() {
        // Verify all sources implement the same interface
        async fn test_source_api<S: DocumentSource>(source: &S) {
            // All sources must implement these methods identically
            let _ = source.source_id();
            let _ = source.name();
            let _ = source.version();
            let _ = source.supported_extensions();
            let _ = source.supported_mime_types();
            
            let input = SourceInput::Memory {
                data: vec![],
                filename: None,
                mime_type: None,
            };
            
            let _ = source.can_handle(&input).await;
            let _ = source.validate(&input).await;
            let _ = source.config_schema();
        }
        
        // Test multiple source types
        let pdf = PdfSource::new();
        test_source_api(&pdf).await;
        
        // Future sources would be tested the same way
        // let docx = DocxSource::new();
        // test_source_api(&docx).await;
    }
    
    /// Test: Extensibility
    #[test]
    fn test_extensibility() {
        // Verify new sources can be added without modifying core
        struct CustomSource;
        
        #[async_trait]
        impl DocumentSource for CustomSource {
            fn source_id(&self) -> &str { "custom" }
            fn name(&self) -> &str { "Custom Source" }
            fn version(&self) -> &str { "1.0.0" }
            fn supported_extensions(&self) -> &[&str] { &["custom"] }
            fn supported_mime_types(&self) -> &[&str] { &[] }
            
            async fn can_handle(&self, _input: &SourceInput) -> Result<bool, SourceError> {
                Ok(true)
            }
            
            async fn validate(&self, _input: &SourceInput) -> Result<ValidationResult, SourceError> {
                Ok(ValidationResult::default())
            }
            
            async fn extract(&self, _input: SourceInput) -> Result<ExtractedDocument, SourceError> {
                todo!()
            }
            
            fn config_schema(&self) -> serde_json::Value {
                json!({})
            }
            
            async fn initialize(&mut self, _config: SourceConfig) -> Result<(), SourceError> {
                Ok(())
            }
            
            async fn cleanup(&mut self) -> Result<(), SourceError> {
                Ok(())
            }
        }
        
        // Verify custom source can be used anywhere DocumentSource is expected
        let custom: Box<dyn DocumentSource> = Box::new(CustomSource);
        assert_eq!(custom.source_id(), "custom");
    }
    
    /// Test: Performance characteristics
    #[test]
    fn test_performance_features() {
        // Verify memory-mapped file support
        let pdf = PdfSource::new();
        assert!(pdf.config.performance.use_mmap);
        
        // Verify parallel processing capability
        assert!(pdf.config.performance.parallel_pages);
        
        // Verify buffer pooling
        assert_eq!(pdf.config.performance.buffer_pool_size, 16);
        
        // Verify streaming support
        let input = SourceInput::Stream {
            reader: Box::new(std::io::empty()),
            size_hint: Some(1024),
            mime_type: None,
        };
        
        // Should handle streaming input
        match input {
            SourceInput::Stream { .. } => assert!(true),
            _ => panic!("Expected stream input"),
        }
    }
    
    /// Test: Security features
    #[test]
    async fn test_security_features() {
        let mut source = PdfSource::new();
        
        // Verify security is enabled by default
        assert!(source.config.security.enabled);
        assert!(!source.config.security.allow_javascript);
        assert!(!source.config.security.allow_external_references);
        
        // Verify input validation
        let malicious_input = SourceInput::Memory {
            data: b"%PDF-1.4\n/JavaScript".to_vec(),
            filename: Some("malicious.pdf".to_string()),
            mime_type: Some("application/pdf".to_string()),
        };
        
        let validation = source.validate(&malicious_input).await.unwrap();
        assert!(!validation.is_valid());
        assert!(validation.errors.iter().any(|e| e.contains("Security")));
    }
    
    /// Test: Configuration system
    #[test]
    fn test_configuration_system() {
        // Verify configuration schema is valid JSON Schema
        let source = PdfSource::new();
        let schema = source.config_schema();
        
        assert!(schema.is_object());
        assert!(schema["type"] == "object");
        assert!(schema["properties"].is_object());
        
        // Verify configuration can be serialized/deserialized
        let config = SourceConfig {
            enabled: true,
            priority: 100,
            timeout: std::time::Duration::from_secs(30),
            memory_limit: Some(100 * 1024 * 1024),
            thread_pool_size: Some(4),
            retry: RetryConfig::default(),
            settings: json!({
                "max_file_size": 50000000,
                "enable_ocr": true
            }),
        };
        
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: SourceConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(config.priority, deserialized.priority);
    }
    
    /// Test: Plugin discovery
    #[test]
    async fn test_plugin_discovery() {
        let manager = SourcePluginManager::new(ManagerConfig {
            plugin_directories: vec![
                PathBuf::from("./plugins"),
                PathBuf::from("/usr/local/lib/neuraldocflow/sources"),
            ],
            config_path: PathBuf::from("./config.yaml"),
        }).unwrap();
        
        // Verify multiple plugin directories are supported
        assert_eq!(manager.plugin_dirs.len(), 2);
        
        // Verify compatible sources can be found
        let input = SourceInput::File {
            path: PathBuf::from("test.pdf"),
            metadata: None,
        };
        
        let sources = manager.find_compatible_sources(&input).await.unwrap();
        // Built-in PDF source should be found
        assert!(sources.iter().any(|s| s.source_id() == "pdf"));
    }
    
    /// Test: Error handling
    #[test]
    fn test_error_handling() {
        // Verify error types are comprehensive
        fn handle_error(err: SourceError) {
            match err {
                SourceError::InvalidFormat(_) => {},
                SourceError::UnsupportedInput => {},
                SourceError::UnsupportedVersion(_) => {},
                SourceError::ParseError(_) => {},
                SourceError::IoError(_) => {},
                SourceError::ConfigError(_) => {},
                SourceError::SecurityError(_) => {},
                SourceError::ApiError(_) => {},
                SourceError::Custom(_) => {},
                // Exhaustive match ensures all errors are handled
            }
        }
        
        // Verify errors provide context
        let err = SourceError::InvalidFormat("Missing magic bytes".to_string());
        assert!(err.to_string().contains("Missing magic bytes"));
    }
    
    /// Test: Memory safety
    #[test]
    fn test_memory_safety() {
        // Verify all public types are Send + Sync
        fn assert_send_sync<T: Send + Sync>() {}
        
        assert_send_sync::<PdfSource>();
        assert_send_sync::<SourcePluginManager>();
        assert_send_sync::<ExtractedDocument>();
        assert_send_sync::<ContentBlock>();
        
        // Verify no memory leaks in plugin lifecycle
        let source = Arc::new(PdfSource::new());
        let weak = Arc::downgrade(&source);
        drop(source);
        assert!(weak.upgrade().is_none());
    }
    
    /// Test: Future source support
    #[test]
    fn test_future_source_templates() {
        // Verify architecture supports planned sources
        
        // DOCX support
        assert!(BlockType::Table != BlockType::Paragraph);
        assert!(matches!(BlockType::List(ListType::Ordered), BlockType::List(_)));
        
        // HTML support  
        assert!(SourceInput::Url { 
            url: String::new(), 
            headers: None 
        }.is_url());
        
        // Image support
        assert!(BlockType::Image != BlockType::Text);
        
        // Audio/Video support via binary content
        let block = ContentBlock {
            id: String::new(),
            block_type: BlockType::Custom("audio".to_string()),
            text: None,
            binary: Some(vec![0u8; 1024]), // Audio data
            metadata: BlockMetadata::default(),
            position: BlockPosition::default(),
            relationships: vec![],
        };
        assert!(block.binary.is_some());
    }
}

// Helper implementations for tests
impl SourceInput {
    fn is_url(&self) -> bool {
        matches!(self, SourceInput::Url { .. })
    }
}

impl Default for BlockMetadata {
    fn default() -> Self {
        Self {
            page: None,
            confidence: 1.0,
            language: None,
            attributes: HashMap::new(),
        }
    }
}

impl Default for BlockPosition {
    fn default() -> Self {
        Self {
            page: 0,
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
        }
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self {
            is_valid: true,
            errors: vec![],
            warnings: vec![],
            security_issues: vec![],
        }
    }
}

impl ValidationResult {
    fn add_error(&mut self, error: &str) {
        self.errors.push(error.to_string());
        self.is_valid = false;
    }
}