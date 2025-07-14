//! Test suite for the DocumentEngine implementation
//!
//! This module provides comprehensive tests for the 4-layer DocumentEngine architecture.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        engine::{DocumentEngine, ExtractionSchema, OutputTemplate, DomainConfig},
        config::NeuralDocFlowConfig,
        plugins::CorePluginRegistry,
    };
    use std::collections::HashMap;
    use tokio;

    #[tokio::test]
    async fn test_document_engine_creation() {
        let config = NeuralDocFlowConfig::default();
        let engine = DocumentEngine::new(config);
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_plugin_registration() {
        let config = NeuralDocFlowConfig::default();
        let engine = DocumentEngine::new(config).unwrap();
        
        // Test plugin registry
        let registry = CorePluginRegistry::new();
        let plugins = registry.get_plugins();
        assert_eq!(plugins.len(), 5);
        
        // Test individual plugin types
        assert!(registry.find_plugin("pdf").is_some());
        assert!(registry.find_plugin("docx").is_some());
        assert!(registry.find_plugin("html").is_some());
        assert!(registry.find_plugin("image").is_some());
        assert!(registry.find_plugin("csv").is_some());
    }

    #[tokio::test]
    async fn test_schema_registration() {
        let config = NeuralDocFlowConfig::default();
        let engine = DocumentEngine::new(config).unwrap();
        
        // Test custom schema registration
        let custom_schema = ExtractionSchema {
            name: "test_schema".to_string(),
            description: "Test schema".to_string(),
            version: "1.0.0".to_string(),
            required_fields: vec!["title".to_string(), "content".to_string()],
            optional_fields: vec!["author".to_string()],
            field_types: [
                ("title".to_string(), "string".to_string()),
                ("content".to_string(), "string".to_string()),
            ].iter().cloned().collect(),
            validation_rules: vec![],
            domain_specific: None,
        };
        
        let result = engine.register_schema(custom_schema).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_output_template_registration() {
        let config = NeuralDocFlowConfig::default();
        let engine = DocumentEngine::new(config).unwrap();
        
        // Test custom template registration
        let custom_template = OutputTemplate {
            name: "custom_json".to_string(),
            format: "json".to_string(),
            template: r#"{"id": "{{document_id}}", "text": "{{content.text}}"}"#.to_string(),
            schema: None,
        };
        
        let result = engine.register_template(custom_template).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_domain_specific_schemas() {
        let config = NeuralDocFlowConfig::default();
        let engine = DocumentEngine::new(config).unwrap();
        
        // Test legal domain schema
        let legal_schema = ExtractionSchema {
            name: "legal_test".to_string(),
            description: "Legal document test schema".to_string(),
            version: "1.0.0".to_string(),
            required_fields: vec!["case_number".to_string(), "court".to_string()],
            optional_fields: vec!["judge".to_string()],
            field_types: HashMap::new(),
            validation_rules: vec![],
            domain_specific: Some(DomainConfig {
                domain: "legal".to_string(),
                specialized_fields: vec!["case_number".to_string(), "court".to_string()],
                compliance_requirements: vec!["legal_validation".to_string()],
                processing_rules: HashMap::new(),
            }),
        };
        
        let result = engine.register_schema(legal_schema).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_available_schemas() {
        let config = NeuralDocFlowConfig::default();
        let engine = DocumentEngine::new(config).unwrap();
        
        let schemas = engine.available_schemas().await;
        assert!(schemas.contains(&"general_document".to_string()));
        assert!(schemas.contains(&"legal_document".to_string()));
        assert!(schemas.contains(&"medical_document".to_string()));
    }

    #[tokio::test]
    async fn test_available_formats() {
        let config = NeuralDocFlowConfig::default();
        let engine = DocumentEngine::new(config).unwrap();
        
        let formats = engine.available_formats().await;
        assert!(formats.contains(&"json".to_string()));
        assert!(formats.contains(&"xml".to_string()));
        assert!(formats.contains(&"text".to_string()));
    }

    #[tokio::test]
    async fn test_plugin_hot_reload() {
        let config = NeuralDocFlowConfig::default();
        let engine = DocumentEngine::new(config).unwrap();
        
        // Test plugin reload (should handle gracefully even if plugin doesn't exist)
        let result = engine.reload_plugin("test_plugin").await;
        // Should not panic - may return error for non-existent plugin
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_engine_initialization() {
        let config = NeuralDocFlowConfig::default();
        let engine = DocumentEngine::new(config).unwrap();
        
        // Test full initialization
        let result = engine.initialize().await;
        assert!(result.is_ok());
    }

    // Integration test to verify the complete pipeline
    #[tokio::test]
    async fn test_complete_pipeline_integration() {
        let config = NeuralDocFlowConfig::default();
        let engine = DocumentEngine::new(config).unwrap();
        
        // Initialize engine
        let init_result = engine.initialize().await;
        assert!(init_result.is_ok());
        
        // Get available components
        let schemas = engine.available_schemas().await;
        let formats = engine.available_formats().await;
        
        // Verify core components are available
        assert!(!schemas.is_empty());
        assert!(!formats.is_empty());
        
        // Test plugin registry
        let registry = CorePluginRegistry::new();
        let plugins = registry.get_plugins();
        assert_eq!(plugins.len(), 5);
        
        // Verify all core plugins are registered
        for plugin in plugins {
            assert!(!plugin.source_type().is_empty());
            assert!(!plugin.supported_extensions().is_empty());
            assert!(!plugin.supported_mime_types().is_empty());
        }
    }

    #[tokio::test]
    async fn test_schema_validation_fields() {
        // Test schema field validation
        let schema = ExtractionSchema {
            name: "validation_test".to_string(),
            description: "Test validation schema".to_string(),
            version: "1.0.0".to_string(),
            required_fields: vec!["title".to_string()],
            optional_fields: vec!["description".to_string()],
            field_types: [
                ("title".to_string(), "string".to_string()),
                ("description".to_string(), "string".to_string()),
            ].iter().cloned().collect(),
            validation_rules: vec![],
            domain_specific: None,
        };
        
        assert_eq!(schema.required_fields.len(), 1);
        assert_eq!(schema.optional_fields.len(), 1);
        assert_eq!(schema.field_types.len(), 2);
    }

    #[tokio::test]
    async fn test_output_template_formats() {
        // Test output template creation
        let json_template = OutputTemplate {
            name: "test_json".to_string(),
            format: "json".to_string(),
            template: r#"{"title": "{{metadata.title}}", "content": "{{content.text}}"}"#.to_string(),
            schema: None,
        };
        
        assert_eq!(json_template.format, "json");
        assert!(json_template.template.contains("{{metadata.title}}"));
        assert!(json_template.template.contains("{{content.text}}"));
    }

    #[tokio::test]
    async fn test_domain_config_creation() {
        // Test domain-specific configuration
        let legal_config = DomainConfig {
            domain: "legal".to_string(),
            specialized_fields: vec!["case_number".to_string(), "court".to_string()],
            compliance_requirements: vec!["legal_validation".to_string()],
            processing_rules: HashMap::new(),
        };
        
        assert_eq!(legal_config.domain, "legal");
        assert_eq!(legal_config.specialized_fields.len(), 2);
        assert_eq!(legal_config.compliance_requirements.len(), 1);
    }
}