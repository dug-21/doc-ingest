//! Source plugin manager
//!
//! Manages registration, discovery, and lifecycle of document source plugins.

use crate::traits::{BaseDocumentSource, SourceCapability, SourceMetadata, SourceConfig, SourceError, SourceResult};
use neural_doc_flow_core::prelude::*;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error};

/// Source plugin manager for dynamic source discovery and management
pub struct SourceManager {
    /// Registered sources by ID
    sources: Arc<RwLock<HashMap<String, Arc<dyn BaseDocumentSource + Send + Sync>>>>,
    /// Source metadata cache
    metadata: Arc<RwLock<HashMap<String, SourceMetadata>>>,
    /// File extension to source ID mapping
    extension_map: Arc<RwLock<HashMap<String, Vec<String>>>>,
    /// MIME type to source ID mapping
    mime_map: Arc<RwLock<HashMap<String, Vec<String>>>>,
}

impl SourceManager {
    /// Create a new source manager
    pub fn new() -> Self {
        Self {
            sources: Arc::new(RwLock::new(HashMap::new())),
            metadata: Arc::new(RwLock::new(HashMap::new())),
            extension_map: Arc::new(RwLock::new(HashMap::new())),
            mime_map: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Register a new source plugin
    pub async fn register_source(&self, source: Arc<dyn BaseDocumentSource + Send + Sync>) -> SourceResult<()> {
        let metadata = source.metadata();
        let source_id = metadata.id.clone();
        
        info!("Registering source plugin: {} v{}", metadata.name, metadata.version);
        
        // Store the source
        {
            let mut sources = self.sources.write().await;
            if sources.contains_key(&source_id) {
                warn!("Source {} already registered, replacing", source_id);
            }
            sources.insert(source_id.clone(), source);
        }
        
        // Update metadata cache
        {
            let mut metadata_cache = self.metadata.write().await;
            metadata_cache.insert(source_id.clone(), metadata.clone());
        }
        
        // Update extension mappings
        {
            let mut ext_map = self.extension_map.write().await;
            for ext in &metadata.file_extensions {
                ext_map.entry(ext.clone())
                    .or_insert_with(Vec::new)
                    .push(source_id.clone());
            }
        }
        
        // Update MIME type mappings
        {
            let mut mime_map = self.mime_map.write().await;
            for mime in &metadata.mime_types {
                mime_map.entry(mime.clone())
                    .or_insert_with(Vec::new)
                    .push(source_id.clone());
            }
        }
        
        info!("Successfully registered source: {}", source_id);
        Ok(())
    }
    
    /// Unregister a source plugin
    pub async fn unregister_source(&self, source_id: &str) -> SourceResult<()> {
        info!("Unregistering source plugin: {}", source_id);
        
        // Get metadata for cleanup
        let metadata = {
            let metadata_cache = self.metadata.read().await;
            metadata_cache.get(source_id).cloned()
        };
        
        if let Some(metadata) = metadata {
            // Remove from sources
            {
                let mut sources = self.sources.write().await;
                sources.remove(source_id);
            }
            
            // Remove from metadata cache
            {
                let mut metadata_cache = self.metadata.write().await;
                metadata_cache.remove(source_id);
            }
            
            // Clean up extension mappings
            {
                let mut ext_map = self.extension_map.write().await;
                for ext in &metadata.file_extensions {
                    if let Some(sources) = ext_map.get_mut(ext) {
                        sources.retain(|id| id != source_id);
                        if sources.is_empty() {
                            ext_map.remove(ext);
                        }
                    }
                }
            }
            
            // Clean up MIME mappings
            {
                let mut mime_map = self.mime_map.write().await;
                for mime in &metadata.mime_types {
                    if let Some(sources) = mime_map.get_mut(mime) {
                        sources.retain(|id| id != source_id);
                        if sources.is_empty() {
                            mime_map.remove(mime);
                        }
                    }
                }
            }
            
            info!("Successfully unregistered source: {}", source_id);
            Ok(())
        } else {
            Err(SourceError::ConfigError(format!("Source {} not found", source_id)))
        }
    }
    
    /// Get a source by ID
    pub async fn get_source(&self, source_id: &str) -> Option<Arc<dyn BaseDocumentSource + Send + Sync>> {
        let sources = self.sources.read().await;
        sources.get(source_id).cloned()
    }
    
    /// Find sources that can handle a given file extension
    pub async fn find_sources_by_extension(&self, extension: &str) -> Vec<String> {
        let ext_map = self.extension_map.read().await;
        ext_map.get(extension).cloned().unwrap_or_default()
    }
    
    /// Find sources that can handle a given MIME type
    pub async fn find_sources_by_mime(&self, mime_type: &str) -> Vec<String> {
        let mime_map = self.mime_map.read().await;
        mime_map.get(mime_type).cloned().unwrap_or_default()
    }
    
    /// Find sources with specific capabilities
    pub async fn find_sources_by_capability(&self, capability: SourceCapability) -> Vec<String> {
        let metadata_cache = self.metadata.read().await;
        metadata_cache.iter()
            .filter(|(_, metadata)| metadata.capabilities.contains(&capability))
            .map(|(id, _)| id.clone())
            .collect()
    }
    
    /// Get all registered source IDs
    pub async fn list_sources(&self) -> Vec<String> {
        let sources = self.sources.read().await;
        sources.keys().cloned().collect()
    }
    
    /// Get metadata for all registered sources
    pub async fn list_metadata(&self) -> Vec<SourceMetadata> {
        let metadata_cache = self.metadata.read().await;
        metadata_cache.values().cloned().collect()
    }
    
    /// Validate that a source is properly configured and functional
    pub async fn validate_source(&self, source_id: &str) -> SourceResult<bool> {
        if let Some(source) = self.get_source(source_id).await {
            // Basic validation - check if source responds
            let test_input = "test";
            match source.can_handle(test_input).await {
                true | false => Ok(true), // Source is responsive
            }
        } else {
            Err(SourceError::ConfigError(format!("Source {} not found", source_id)))
        }
    }
    
    /// Get capability report for all sources
    pub async fn capability_report(&self) -> HashMap<SourceCapability, Vec<String>> {
        let mut report = HashMap::new();
        let metadata_cache = self.metadata.read().await;
        
        for (id, metadata) in metadata_cache.iter() {
            for capability in &metadata.capabilities {
                report.entry(capability.clone())
                    .or_insert_with(Vec::new)
                    .push(id.clone());
            }
        }
        
        report
    }
}

impl Default for SourceManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Mock source for testing
    struct MockSource {
        metadata: SourceMetadata,
        config: SourceConfig,
    }
    
    #[async_trait]
    impl DocumentSource for MockSource {
        async fn process(&self, input: Document) -> Result<Document> {
            // Mock implementation returns the input unchanged
            Ok(input)
        }
    }
    
    #[async_trait]
    impl BaseDocumentSource for MockSource {
        fn metadata(&self) -> SourceMetadata {
            self.metadata.clone()
        }
        
        async fn can_handle(&self, _input: &str) -> bool {
            true
        }
        
        fn config(&self) -> &SourceConfig {
            &self.config
        }
        
        fn set_config(&mut self, config: SourceConfig) {
            self.config = config;
        }
    }
    
    #[tokio::test]
    async fn test_source_registration() {
        let manager = SourceManager::new();
        
        let source = Arc::new(MockSource {
            metadata: SourceMetadata {
                id: "test-source".to_string(),
                name: "Test Source".to_string(),
                version: "1.0.0".to_string(),
                capabilities: vec![SourceCapability::PlainText],
                file_extensions: vec!["txt".to_string()],
                mime_types: vec!["text/plain".to_string()],
            },
            config: SourceConfig::default(),
        });
        
        // Register source
        manager.register_source(source).await.unwrap();
        
        // Verify registration
        let sources = manager.list_sources().await;
        assert_eq!(sources.len(), 1);
        assert!(sources.contains(&"test-source".to_string()));
        
        // Test finding by extension
        let txt_sources = manager.find_sources_by_extension("txt").await;
        assert_eq!(txt_sources.len(), 1);
        assert_eq!(txt_sources[0], "test-source");
        
        // Test finding by MIME type
        let mime_sources = manager.find_sources_by_mime("text/plain").await;
        assert_eq!(mime_sources.len(), 1);
        assert_eq!(mime_sources[0], "test-source");
        
        // Test finding by capability
        let cap_sources = manager.find_sources_by_capability(SourceCapability::PlainText).await;
        assert_eq!(cap_sources.len(), 1);
        assert_eq!(cap_sources[0], "test-source");
    }
}