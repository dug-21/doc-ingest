# NeuralDocFlow Plugin Architecture

## Overview

The NeuralDocFlow plugin architecture provides a flexible, extensible system for adding domain-specific document processing capabilities. Plugins can extend the core functionality with custom extractors, validators, processors, and integrations while maintaining system stability and performance.

## Plugin System Design

### Core Plugin Interface

```rust
// Core plugin trait definition
use async_trait::async_trait;
use serde::{Serialize, Deserialize};

#[async_trait]
pub trait DocumentPlugin: Send + Sync {
    /// Plugin metadata
    fn metadata(&self) -> PluginMetadata;
    
    /// Initialize the plugin with configuration
    async fn initialize(&mut self, config: PluginConfig) -> Result<(), PluginError>;
    
    /// Check if this plugin can handle the given document
    fn can_handle(&self, document: &DocumentInfo) -> bool;
    
    /// Process the document
    async fn process(
        &self,
        document: &Document,
        context: &ProcessingContext,
    ) -> Result<PluginResult, PluginError>;
    
    /// Validate extraction results
    async fn validate(
        &self,
        result: &ExtractionResult,
        context: &ValidationContext,
    ) -> Result<ValidationReport, PluginError>;
    
    /// Cleanup resources
    async fn shutdown(&mut self) -> Result<(), PluginError>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub author: String,
    pub supported_formats: Vec<String>,
    pub capabilities: Vec<PluginCapability>,
    pub dependencies: Vec<PluginDependency>,
    pub configuration_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginCapability {
    DocumentExtraction,
    TableProcessing,
    ImageAnalysis,
    TextAnalysis,
    Validation,
    PostProcessing,
    Integration,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginDependency {
    pub name: String,
    pub version_requirement: String,
    pub optional: bool,
}
```

### Plugin Registry and Management

```rust
// Plugin registry for dynamic loading and management
pub struct PluginRegistry {
    plugins: HashMap<String, Box<dyn DocumentPlugin>>,
    plugin_configs: HashMap<String, PluginConfig>,
    dependency_graph: DependencyGraph,
    security_manager: PluginSecurityManager,
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
            plugin_configs: HashMap::new(),
            dependency_graph: DependencyGraph::new(),
            security_manager: PluginSecurityManager::new(),
        }
    }
    
    /// Load plugin from dynamic library
    pub async fn load_plugin(
        &mut self,
        plugin_path: &Path,
        config: PluginConfig,
    ) -> Result<String, PluginError> {
        // Security validation
        self.security_manager.validate_plugin(plugin_path)?;
        
        // Load dynamic library
        let lib = unsafe { libloading::Library::new(plugin_path)? };
        
        // Get plugin constructor function
        let constructor: libloading::Symbol<unsafe extern "C" fn() -> *mut dyn DocumentPlugin> =
            unsafe { lib.get(b"create_plugin")? };
        
        let plugin = unsafe { Box::from_raw(constructor()) };
        
        // Initialize plugin
        let mut plugin = plugin;
        plugin.initialize(config.clone()).await?;
        
        let metadata = plugin.metadata();
        let plugin_name = metadata.name.clone();
        
        // Check dependencies
        self.dependency_graph.add_plugin(&metadata)?;
        
        // Register plugin
        self.plugins.insert(plugin_name.clone(), plugin);
        self.plugin_configs.insert(plugin_name.clone(), config);
        
        Ok(plugin_name)
    }
    
    /// Get plugins that can handle a document
    pub fn find_compatible_plugins(&self, document: &DocumentInfo) -> Vec<&str> {
        self.plugins
            .iter()
            .filter(|(_, plugin)| plugin.can_handle(document))
            .map(|(name, _)| name.as_str())
            .collect()
    }
    
    /// Execute plugin processing pipeline
    pub async fn process_with_plugins(
        &self,
        document: &Document,
        plugin_names: &[String],
        context: &ProcessingContext,
    ) -> Result<Vec<PluginResult>, PluginError> {
        let mut results = Vec::new();
        
        for plugin_name in plugin_names {
            if let Some(plugin) = self.plugins.get(plugin_name) {
                let result = plugin.process(document, context).await?;
                results.push(result);
            }
        }
        
        Ok(results)
    }
}
```

### Plugin Types and Implementations

#### Document Format Plugins

```rust
// Example: Advanced PDF processing plugin
pub struct AdvancedPdfPlugin {
    ocr_engine: Option<TesseractEngine>,
    table_detector: TableDetector,
    image_processor: ImageProcessor,
}

#[async_trait]
impl DocumentPlugin for AdvancedPdfPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "advanced_pdf_processor".to_string(),
            version: "1.0.0".to_string(),
            description: "Advanced PDF processing with OCR and image analysis".to_string(),
            author: "NeuralDocFlow Team".to_string(),
            supported_formats: vec!["pdf".to_string()],
            capabilities: vec![
                PluginCapability::DocumentExtraction,
                PluginCapability::TableProcessing,
                PluginCapability::ImageAnalysis,
            ],
            dependencies: vec![
                PluginDependency {
                    name: "tesseract".to_string(),
                    version_requirement: ">=4.0".to_string(),
                    optional: true,
                },
            ],
            configuration_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "enable_ocr": {"type": "boolean", "default": false},
                    "ocr_language": {"type": "string", "default": "eng"},
                    "image_dpi": {"type": "integer", "default": 300}
                }
            }),
        }
    }
    
    async fn initialize(&mut self, config: PluginConfig) -> Result<(), PluginError> {
        if config.get("enable_ocr").unwrap_or(&false.into()).as_bool().unwrap_or(false) {
            let language = config.get("ocr_language")
                .unwrap_or(&"eng".into())
                .as_str()
                .unwrap_or("eng");
            
            self.ocr_engine = Some(TesseractEngine::new(language)?);
        }
        
        Ok(())
    }
    
    fn can_handle(&self, document: &DocumentInfo) -> bool {
        document.format.to_lowercase() == "pdf"
    }
    
    async fn process(
        &self,
        document: &Document,
        context: &ProcessingContext,
    ) -> Result<PluginResult, PluginError> {
        let mut extracted_data = ExtractionResult::new();
        
        // Extract text content
        let text_content = self.extract_text(document).await?;
        extracted_data.add_field("text_content", text_content);
        
        // Process tables
        let tables = self.table_detector.detect_tables(document).await?;
        extracted_data.add_field("tables", tables);
        
        // Process images if OCR is enabled
        if let Some(ref ocr_engine) = self.ocr_engine {
            let images = self.image_processor.extract_images(document).await?;
            let mut ocr_results = Vec::new();
            
            for image in images {
                let ocr_text = ocr_engine.process_image(&image).await?;
                ocr_results.push(ocr_text);
            }
            
            extracted_data.add_field("ocr_text", ocr_results);
        }
        
        Ok(PluginResult {
            plugin_name: "advanced_pdf_processor".to_string(),
            extraction_result: extracted_data,
            confidence_score: 0.92,
            processing_time: context.elapsed(),
            metadata: serde_json::json!({
                "pages_processed": document.page_count(),
                "tables_found": tables.len(),
                "images_processed": self.image_processor.image_count()
            }),
        })
    }
}
```

#### Domain-Specific Plugins

```rust
// Example: SEC Filing specialized plugin
pub struct SecFilingPlugin {
    form_types: HashSet<String>,
    field_extractors: HashMap<String, Box<dyn FieldExtractor>>,
    validation_rules: Vec<ValidationRule>,
}

#[async_trait]
impl DocumentPlugin for SecFilingPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "sec_filing_processor".to_string(),
            version: "2.1.0".to_string(),
            description: "Specialized processor for SEC filing documents".to_string(),
            author: "Financial Data Team".to_string(),
            supported_formats: vec!["pdf".to_string(), "html".to_string()],
            capabilities: vec![
                PluginCapability::DocumentExtraction,
                PluginCapability::Validation,
                PluginCapability::Custom("financial_analysis".to_string()),
            ],
            dependencies: vec![],
            configuration_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "form_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": ["10-K", "10-Q", "8-K"]
                    },
                    "extract_financials": {"type": "boolean", "default": true},
                    "validate_consistency": {"type": "boolean", "default": true}
                }
            }),
        }
    }
    
    async fn process(
        &self,
        document: &Document,
        context: &ProcessingContext,
    ) -> Result<PluginResult, PluginError> {
        // Identify form type
        let form_type = self.identify_form_type(document).await?;
        
        if !self.form_types.contains(&form_type) {
            return Err(PluginError::UnsupportedDocument(
                format!("Form type {} not supported", form_type)
            ));
        }
        
        let mut result = ExtractionResult::new();
        
        // Extract standard SEC fields
        result.add_field("form_type", form_type.clone());
        result.add_field("company_name", self.extract_company_name(document).await?);
        result.add_field("cik", self.extract_cik(document).await?);
        result.add_field("fiscal_year_end", self.extract_fiscal_year_end(document).await?);
        
        // Extract form-specific data
        match form_type.as_str() {
            "10-K" => {
                result.merge(self.extract_10k_data(document).await?);
            }
            "10-Q" => {
                result.merge(self.extract_10q_data(document).await?);
            }
            "8-K" => {
                result.merge(self.extract_8k_data(document).await?);
            }
            _ => {}
        }
        
        Ok(PluginResult {
            plugin_name: "sec_filing_processor".to_string(),
            extraction_result: result,
            confidence_score: 0.96,
            processing_time: context.elapsed(),
            metadata: serde_json::json!({
                "form_type": form_type,
                "sections_found": self.count_sections(document).await?
            }),
        })
    }
    
    async fn validate(
        &self,
        result: &ExtractionResult,
        context: &ValidationContext,
    ) -> Result<ValidationReport, PluginError> {
        let mut report = ValidationReport::new();
        
        // Validate required fields
        for rule in &self.validation_rules {
            let validation_result = rule.validate(result).await?;
            report.add_result(validation_result);
        }
        
        // SEC-specific validations
        if let Some(cik) = result.get_field("cik") {
            if !self.validate_cik_format(cik) {
                report.add_error("Invalid CIK format");
            }
        }
        
        // Financial consistency checks
        if result.has_financial_data() {
            let consistency_check = self.validate_financial_consistency(result).await?;
            report.add_result(consistency_check);
        }
        
        Ok(report)
    }
}
```

#### Integration Plugins

```rust
// Example: Database integration plugin
pub struct DatabaseIntegrationPlugin {
    connection_pool: Option<sqlx::Pool<sqlx::Postgres>>,
    table_mappings: HashMap<String, TableMapping>,
}

#[async_trait]
impl DocumentPlugin for DatabaseIntegrationPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata {
            name: "database_integration".to_string(),
            version: "1.0.0".to_string(),
            description: "Store extraction results in database".to_string(),
            author: "Integration Team".to_string(),
            supported_formats: vec![], // Works with any format
            capabilities: vec![PluginCapability::Integration],
            dependencies: vec![
                PluginDependency {
                    name: "postgresql".to_string(),
                    version_requirement: ">=12.0".to_string(),
                    optional: false,
                },
            ],
            configuration_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "database_url": {"type": "string"},
                    "table_prefix": {"type": "string", "default": "neuraldocflow_"},
                    "batch_size": {"type": "integer", "default": 100}
                },
                "required": ["database_url"]
            }),
        }
    }
    
    async fn process(
        &self,
        document: &Document,
        context: &ProcessingContext,
    ) -> Result<PluginResult, PluginError> {
        // This plugin doesn't process documents, it stores results
        Ok(PluginResult::empty())
    }
    
    /// Store extraction results in database
    pub async fn store_results(
        &self,
        document_id: &str,
        results: &[PluginResult],
    ) -> Result<(), PluginError> {
        let pool = self.connection_pool.as_ref()
            .ok_or(PluginError::NotInitialized)?;
        
        let mut transaction = pool.begin().await?;
        
        // Store document metadata
        sqlx::query!(
            "INSERT INTO documents (id, filename, format, processed_at) VALUES ($1, $2, $3, $4)",
            document_id,
            results[0].metadata.get("filename"),
            results[0].metadata.get("format"),
            chrono::Utc::now()
        )
        .execute(&mut *transaction)
        .await?;
        
        // Store extraction results
        for result in results {
            self.store_plugin_result(document_id, result, &mut transaction).await?;
        }
        
        transaction.commit().await?;
        
        Ok(())
    }
}
```

## Plugin Configuration System

### Configuration Schema

```yaml
# plugin-config.yaml
plugins:
  advanced_pdf_processor:
    enabled: true
    config:
      enable_ocr: true
      ocr_language: "eng"
      image_dpi: 300
    priority: 10
    
  sec_filing_processor:
    enabled: true
    config:
      form_types: ["10-K", "10-Q", "8-K", "DEF 14A"]
      extract_financials: true
      validate_consistency: true
    priority: 20
    conditions:
      - document_type: "sec_filing"
      - filename_pattern: ".*\\.(10-k|10-q|8-k).*"
    
  database_integration:
    enabled: true
    config:
      database_url: "${DATABASE_URL}"
      table_prefix: "neuraldocflow_"
      batch_size: 50
    priority: 100
    execution_phase: "post_processing"

# Plugin discovery and loading
plugin_directories:
  - "/usr/local/lib/neuraldocflow/plugins"
  - "./plugins"
  - "${HOME}/.neuraldocflow/plugins"

# Security settings
security:
  allow_unsigned_plugins: false
  plugin_sandbox: true
  max_memory_usage: "1GB"
  max_execution_time: "30s"
  allowed_network_access: false
```

### Dynamic Plugin Loading

```rust
// Plugin loader with hot-reloading support
pub struct PluginLoader {
    registry: Arc<RwLock<PluginRegistry>>,
    watcher: Option<notify::RecommendedWatcher>,
    config_manager: ConfigManager,
}

impl PluginLoader {
    pub fn new(config_path: &Path) -> Result<Self, PluginError> {
        Ok(Self {
            registry: Arc::new(RwLock::new(PluginRegistry::new())),
            watcher: None,
            config_manager: ConfigManager::load(config_path)?,
        })
    }
    
    /// Load all configured plugins
    pub async fn load_plugins(&mut self) -> Result<(), PluginError> {
        let plugin_configs = self.config_manager.get_plugin_configs();
        
        for (plugin_name, plugin_config) in plugin_configs {
            if plugin_config.enabled {
                self.load_single_plugin(&plugin_name, plugin_config).await?;
            }
        }
        
        Ok(())
    }
    
    /// Set up file system watcher for hot-reloading
    pub fn enable_hot_reload(&mut self) -> Result<(), PluginError> {
        let registry = Arc::clone(&self.registry);
        let config_manager = self.config_manager.clone();
        
        let (tx, rx) = std::sync::mpsc::channel();
        let mut watcher = notify::watcher(tx, Duration::from_secs(2))?;
        
        // Watch plugin directories
        for dir in self.config_manager.get_plugin_directories() {
            watcher.watch(dir, notify::RecursiveMode::Recursive)?;
        }
        
        // Spawn background task to handle file changes
        tokio::spawn(async move {
            while let Ok(event) = rx.recv() {
                match event {
                    notify::DebouncedEvent::Write(path) |
                    notify::DebouncedEvent::Create(path) => {
                        if path.extension() == Some(std::ffi::OsStr::new("so")) {
                            Self::reload_plugin(&registry, &config_manager, &path).await;
                        }
                    }
                    notify::DebouncedEvent::Remove(path) => {
                        Self::unload_plugin(&registry, &path).await;
                    }
                    _ => {}
                }
            }
        });
        
        self.watcher = Some(watcher);
        Ok(())
    }
    
    async fn reload_plugin(
        registry: &Arc<RwLock<PluginRegistry>>,
        config_manager: &ConfigManager,
        plugin_path: &Path,
    ) {
        if let Some(plugin_name) = Self::extract_plugin_name(plugin_path) {
            if let Some(config) = config_manager.get_plugin_config(&plugin_name) {
                let mut registry = registry.write().await;
                
                // Unload existing plugin
                registry.unload_plugin(&plugin_name).await.ok();
                
                // Load new version
                registry.load_plugin(plugin_path, config).await.ok();
                
                log::info!("Reloaded plugin: {}", plugin_name);
            }
        }
    }
}
```

## Plugin Development Kit

### Plugin Template Generator

```rust
// CLI tool for generating plugin templates
use clap::{App, Arg, SubCommand};

pub fn main() {
    let matches = App::new("neuraldocflow-plugin-cli")
        .version("1.0")
        .author("NeuralDocFlow Team")
        .about("Plugin development tools")
        .subcommand(
            SubCommand::with_name("new")
                .about("Create a new plugin")
                .arg(Arg::with_name("name").required(true))
                .arg(Arg::with_name("type")
                    .long("type")
                    .takes_value(true)
                    .possible_values(&["extractor", "validator", "integration"])
                    .default_value("extractor"))
        )
        .subcommand(
            SubCommand::with_name("build")
                .about("Build plugin")
                .arg(Arg::with_name("release").long("release"))
        )
        .subcommand(
            SubCommand::with_name("test")
                .about("Test plugin")
                .arg(Arg::with_name("integration").long("integration"))
        )
        .get_matches();
    
    match matches.subcommand() {
        ("new", Some(sub_m)) => {
            let name = sub_m.value_of("name").unwrap();
            let plugin_type = sub_m.value_of("type").unwrap();
            create_plugin_template(name, plugin_type).unwrap();
        }
        ("build", Some(sub_m)) => {
            let release = sub_m.is_present("release");
            build_plugin(release).unwrap();
        }
        ("test", Some(sub_m)) => {
            let integration = sub_m.is_present("integration");
            test_plugin(integration).unwrap();
        }
        _ => {
            println!("Use --help for usage information");
        }
    }
}

fn create_plugin_template(name: &str, plugin_type: &str) -> Result<(), Box<dyn std::error::Error>> {
    let template_dir = format!("{}_plugin", name);
    std::fs::create_dir_all(&template_dir)?;
    
    // Generate Cargo.toml
    let cargo_toml = format!(r#"
[package]
name = "{}"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
neuraldocflow-plugin-sdk = "1.0"
async-trait = "0.1"
serde = {{ version = "1.0", features = ["derive"] }}
serde_json = "1.0"
tokio = {{ version = "1.0", features = ["full"] }}

[dev-dependencies]
neuraldocflow-test-utils = "1.0"
"#, name);
    
    std::fs::write(format!("{}/Cargo.toml", template_dir), cargo_toml)?;
    
    // Generate main plugin code
    let plugin_code = match plugin_type {
        "extractor" => generate_extractor_template(name),
        "validator" => generate_validator_template(name),
        "integration" => generate_integration_template(name),
        _ => unreachable!(),
    };
    
    std::fs::write(format!("{}/src/lib.rs", template_dir), plugin_code)?;
    
    // Generate test file
    let test_code = generate_test_template(name);
    std::fs::create_dir_all(format!("{}/tests", template_dir))?;
    std::fs::write(format!("{}/tests/integration_test.rs", template_dir), test_code)?;
    
    println!("Created plugin template: {}", template_dir);
    Ok(())
}
```

### Plugin SDK

```rust
// Simplified SDK for plugin development
pub mod neuraldocflow_plugin_sdk {
    pub use neuraldocflow_core::{
        Document, DocumentInfo, ProcessingContext, ValidationContext,
        ExtractionResult, ValidationReport, PluginResult, PluginError
    };
    
    pub use async_trait::async_trait;
    
    // Convenience macros for plugin development
    #[macro_export]
    macro_rules! plugin_main {
        ($plugin_type:ty) => {
            #[no_mangle]
            pub extern "C" fn create_plugin() -> *mut dyn DocumentPlugin {
                Box::into_raw(Box::new(<$plugin_type>::new()))
            }
        };
    }
    
    #[macro_export]
    macro_rules! extract_field {
        ($document:expr, $pattern:expr, $field_name:expr) => {
            {
                use regex::Regex;
                let re = Regex::new($pattern)?;
                if let Some(captures) = re.captures(&$document.text_content()) {
                    captures.get(1).map(|m| m.as_str().to_string())
                } else {
                    None
                }
            }
        };
    }
    
    // Helper functions for common operations
    pub fn extract_table_from_pdf(document: &Document, table_index: usize) -> Result<Table, PluginError> {
        // Implementation for table extraction
        unimplemented!()
    }
    
    pub fn validate_financial_data(data: &ExtractionResult) -> ValidationReport {
        // Implementation for financial validation
        unimplemented!()
    }
    
    pub fn send_webhook(url: &str, payload: &serde_json::Value) -> Result<(), PluginError> {
        // Implementation for webhook sending
        unimplemented!()
    }
}
```

This comprehensive plugin architecture provides:

1. **Flexible Plugin Interface**: Support for different types of plugins (extractors, validators, integrations)
2. **Dynamic Loading**: Hot-reloadable plugins with dependency management
3. **Security**: Sandboxed execution with resource limits
4. **Configuration**: YAML-based configuration with schema validation
5. **Development Tools**: CLI tools and SDK for easy plugin development
6. **Integration**: Seamless integration with the core processing pipeline

The architecture ensures that plugins can extend NeuralDocFlow's capabilities while maintaining system stability and security.