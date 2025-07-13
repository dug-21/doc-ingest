# NeuralDocFlow Modular Source Architecture

## Executive Summary

This document presents a comprehensive modular source architecture for NeuralDocFlow that enables extensible document source support through a trait-based plugin system. The architecture is designed for zero-cost abstractions, type safety, and hot-reloadable extractors while maintaining a consistent API across all document sources.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Source Plugin System                          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌────────────┬──────────────┬─────────────┬────────────────────┐  │
│  │    Core    │   Built-in   │   Dynamic   │   User-Defined    │  │
│  │   Traits   │   Sources    │   Loader    │     Sources       │  │
│  │            │              │             │                    │  │
│  │ • Source   │ • PDF        │ • Hot-reload│ • Custom formats  │  │
│  │ • Reader   │ • DOCX       │ • Discovery │ • API sources     │  │
│  │ • Extractor│ • HTML       │ • Registry  │ • Database        │  │
│  │ • Config   │ • Images     │ • Security  │ • Cloud storage   │  │
│  │            │ • Audio/Video│             │                    │  │
│  └────────────┴──────────────┴─────────────┴────────────────────┘  │
├─────────────────────────────────────────────────────────────────────┤
│                      Source Pipeline                                 │
│  ┌────────────┬──────────────┬─────────────┬────────────────────┐  │
│  │   Input    │  Validation  │  Extraction │   Post-Process    │  │
│  │            │              │             │                    │  │
│  │ • Stream   │ • Format     │ • Content   │ • Normalization   │  │
│  │ • File     │ • Security   │ • Metadata  │ • Enhancement     │  │
│  │ • URL      │ • Size limit │ • Structure │ • Enrichment      │  │
│  │ • Memory   │ • Malware    │ • Features  │ • Validation      │  │
│  └────────────┴──────────────┴─────────────┴────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

## Core Source Traits

### 1. Base Source Trait

```rust
use async_trait::async_trait;
use std::path::Path;
use tokio::io::AsyncRead;

/// Core trait that all document sources must implement
#[async_trait]
pub trait DocumentSource: Send + Sync + 'static {
    /// Unique identifier for the source type
    fn source_id(&self) -> &str;
    
    /// Human-readable name
    fn name(&self) -> &str;
    
    /// Version of the source implementation
    fn version(&self) -> &str;
    
    /// Supported file extensions (e.g., ["pdf", "PDF"])
    fn supported_extensions(&self) -> &[&str];
    
    /// MIME types this source can handle
    fn supported_mime_types(&self) -> &[&str];
    
    /// Check if this source can handle the given input
    async fn can_handle(&self, input: &SourceInput) -> Result<bool, SourceError>;
    
    /// Validate input before processing
    async fn validate(&self, input: &SourceInput) -> Result<ValidationResult, SourceError>;
    
    /// Extract document content
    async fn extract(&self, input: SourceInput) -> Result<ExtractedDocument, SourceError>;
    
    /// Get source-specific configuration schema
    fn config_schema(&self) -> serde_json::Value;
    
    /// Initialize with configuration
    async fn initialize(&mut self, config: SourceConfig) -> Result<(), SourceError>;
    
    /// Cleanup resources
    async fn cleanup(&mut self) -> Result<(), SourceError>;
}

/// Input types that sources can handle
#[derive(Debug)]
pub enum SourceInput {
    /// File path input
    File {
        path: PathBuf,
        metadata: Option<FileMetadata>,
    },
    /// Memory buffer input
    Memory {
        data: Vec<u8>,
        filename: Option<String>,
        mime_type: Option<String>,
    },
    /// Stream input
    Stream {
        reader: Box<dyn AsyncRead + Send + Unpin>,
        size_hint: Option<usize>,
        mime_type: Option<String>,
    },
    /// URL input
    Url {
        url: String,
        headers: Option<HashMap<String, String>>,
    },
}

/// Extracted document representation
#[derive(Debug, Clone)]
pub struct ExtractedDocument {
    /// Unique document ID
    pub id: String,
    
    /// Source that extracted this document
    pub source_id: String,
    
    /// Document metadata
    pub metadata: DocumentMetadata,
    
    /// Extracted content blocks
    pub content: Vec<ContentBlock>,
    
    /// Document structure
    pub structure: DocumentStructure,
    
    /// Extraction confidence score
    pub confidence: f32,
    
    /// Processing metrics
    pub metrics: ExtractionMetrics,
}

/// Content block representing a unit of extracted content
#[derive(Debug, Clone)]
pub struct ContentBlock {
    /// Unique block ID
    pub id: String,
    
    /// Block type
    pub block_type: BlockType,
    
    /// Text content (if applicable)
    pub text: Option<String>,
    
    /// Binary content (for images, etc.)
    pub binary: Option<Vec<u8>>,
    
    /// Block metadata
    pub metadata: BlockMetadata,
    
    /// Position in document
    pub position: BlockPosition,
    
    /// Relationships to other blocks
    pub relationships: Vec<BlockRelationship>,
}

#[derive(Debug, Clone)]
pub enum BlockType {
    Paragraph,
    Heading(u8), // Heading level 1-6
    Table,
    Image,
    List(ListType),
    CodeBlock,
    Quote,
    Footnote,
    Caption,
    Formula,
    Custom(String),
}
```

### 2. Reader Trait for Input Handling

```rust
/// Trait for reading document content from various sources
#[async_trait]
pub trait SourceReader: Send + Sync {
    /// Read from file system
    async fn read_file(&self, path: &Path) -> Result<Vec<u8>, SourceError>;
    
    /// Read from memory buffer
    async fn read_memory(&self, data: Vec<u8>) -> Result<Vec<u8>, SourceError>;
    
    /// Read from async stream
    async fn read_stream(
        &self,
        reader: Box<dyn AsyncRead + Send + Unpin>,
        size_hint: Option<usize>,
    ) -> Result<Vec<u8>, SourceError>;
    
    /// Read from URL
    async fn read_url(
        &self,
        url: &str,
        headers: Option<&HashMap<String, String>>,
    ) -> Result<Vec<u8>, SourceError>;
    
    /// Get reader capabilities
    fn capabilities(&self) -> ReaderCapabilities;
}

/// Reader capabilities
#[derive(Debug, Clone)]
pub struct ReaderCapabilities {
    pub supports_streaming: bool,
    pub supports_partial_read: bool,
    pub max_file_size: Option<usize>,
    pub supports_compression: Vec<CompressionFormat>,
    pub supports_encryption: bool,
}
```

### 3. Extractor Trait for Content Processing

```rust
/// Trait for extracting structured content from documents
#[async_trait]
pub trait ContentExtractor: Send + Sync {
    /// Extract text content
    async fn extract_text(&self, data: &[u8]) -> Result<Vec<TextBlock>, SourceError>;
    
    /// Extract tables
    async fn extract_tables(&self, data: &[u8]) -> Result<Vec<Table>, SourceError>;
    
    /// Extract images
    async fn extract_images(&self, data: &[u8]) -> Result<Vec<Image>, SourceError>;
    
    /// Extract metadata
    async fn extract_metadata(&self, data: &[u8]) -> Result<DocumentMetadata, SourceError>;
    
    /// Extract document structure
    async fn extract_structure(&self, data: &[u8]) -> Result<DocumentStructure, SourceError>;
    
    /// Get extractor capabilities
    fn capabilities(&self) -> ExtractorCapabilities;
}

/// Extractor capabilities
#[derive(Debug, Clone)]
pub struct ExtractorCapabilities {
    pub supports_ocr: bool,
    pub supports_tables: bool,
    pub supports_images: bool,
    pub supports_forms: bool,
    pub supports_annotations: bool,
    pub preserves_formatting: bool,
    pub language_detection: bool,
}
```

### 4. Configuration Trait

```rust
/// Trait for source configuration
pub trait SourceConfiguration: Send + Sync {
    /// Validate configuration
    fn validate(&self) -> Result<(), ConfigError>;
    
    /// Merge with another configuration
    fn merge(&mut self, other: &Self) -> Result<(), ConfigError>;
    
    /// Get configuration as JSON
    fn to_json(&self) -> serde_json::Value;
    
    /// Load from JSON
    fn from_json(value: serde_json::Value) -> Result<Self, ConfigError>
    where
        Self: Sized;
}

/// Base configuration for all sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceConfig {
    /// Enable source
    pub enabled: bool,
    
    /// Source priority (higher = preferred)
    pub priority: i32,
    
    /// Processing timeout
    pub timeout: Duration,
    
    /// Memory limit
    pub memory_limit: Option<usize>,
    
    /// Thread pool size
    pub thread_pool_size: Option<usize>,
    
    /// Retry configuration
    pub retry: RetryConfig,
    
    /// Source-specific settings
    pub settings: serde_json::Value,
}
```

## Plugin Discovery and Registration

### Plugin Manager

```rust
/// Central plugin manager for document sources
pub struct SourcePluginManager {
    /// Registered sources
    sources: Arc<RwLock<HashMap<String, Arc<dyn DocumentSource>>>>,
    
    /// Plugin directory paths
    plugin_dirs: Vec<PathBuf>,
    
    /// Configuration manager
    config_manager: Arc<ConfigManager>,
    
    /// Plugin loader
    loader: Arc<PluginLoader>,
    
    /// File watcher for hot-reload
    watcher: Option<notify::RecommendedWatcher>,
    
    /// Metrics collector
    metrics: Arc<MetricsCollector>,
}

impl SourcePluginManager {
    /// Create new plugin manager
    pub fn new(config: ManagerConfig) -> Result<Self, PluginError> {
        let manager = Self {
            sources: Arc::new(RwLock::new(HashMap::new())),
            plugin_dirs: config.plugin_directories,
            config_manager: Arc::new(ConfigManager::new(config.config_path)?),
            loader: Arc::new(PluginLoader::new()),
            watcher: None,
            metrics: Arc::new(MetricsCollector::new()),
        };
        
        // Register built-in sources
        manager.register_builtin_sources()?;
        
        // Load external plugins
        manager.load_external_plugins()?;
        
        Ok(manager)
    }
    
    /// Register a document source
    pub async fn register_source(
        &self,
        source_id: String,
        source: Arc<dyn DocumentSource>,
    ) -> Result<(), PluginError> {
        // Validate source
        self.validate_source(&source).await?;
        
        // Initialize source with configuration
        if let Some(config) = self.config_manager.get_source_config(&source_id) {
            source.initialize(config).await?;
        }
        
        // Register
        let mut sources = self.sources.write().await;
        sources.insert(source_id.clone(), source);
        
        // Update metrics
        self.metrics.record_source_registered(&source_id);
        
        Ok(())
    }
    
    /// Find compatible sources for input
    pub async fn find_compatible_sources(
        &self,
        input: &SourceInput,
    ) -> Result<Vec<Arc<dyn DocumentSource>>, PluginError> {
        let sources = self.sources.read().await;
        let mut compatible = Vec::new();
        
        for (_, source) in sources.iter() {
            if source.can_handle(input).await? {
                compatible.push(Arc::clone(source));
            }
        }
        
        // Sort by priority
        compatible.sort_by_key(|s| {
            self.config_manager
                .get_source_config(s.source_id())
                .map(|c| -c.priority)
                .unwrap_or(0)
        });
        
        Ok(compatible)
    }
    
    /// Enable hot-reload for plugins
    pub fn enable_hot_reload(&mut self) -> Result<(), PluginError> {
        let sources = Arc::clone(&self.sources);
        let loader = Arc::clone(&self.loader);
        let config = Arc::clone(&self.config_manager);
        
        let (tx, rx) = std::sync::mpsc::channel();
        let mut watcher = notify::watcher(tx, Duration::from_secs(2))?;
        
        // Watch plugin directories
        for dir in &self.plugin_dirs {
            watcher.watch(dir, notify::RecursiveMode::Recursive)?;
        }
        
        // Spawn reload handler
        tokio::spawn(async move {
            while let Ok(event) = rx.recv() {
                Self::handle_plugin_change(event, &sources, &loader, &config).await;
            }
        });
        
        self.watcher = Some(watcher);
        Ok(())
    }
}
```

### Plugin Loader

```rust
/// Dynamic plugin loader with security validation
pub struct PluginLoader {
    /// Security validator
    validator: SecurityValidator,
    
    /// Loaded libraries
    loaded_libs: Arc<Mutex<HashMap<PathBuf, Library>>>,
    
    /// Plugin metadata cache
    metadata_cache: Arc<RwLock<HashMap<String, PluginMetadata>>>,
}

impl PluginLoader {
    /// Load plugin from dynamic library
    pub async fn load_plugin(
        &self,
        plugin_path: &Path,
    ) -> Result<Arc<dyn DocumentSource>, PluginError> {
        // Security validation
        self.validator.validate_plugin(plugin_path).await?;
        
        // Load library
        let lib = unsafe { Library::new(plugin_path)? };
        
        // Get plugin metadata
        let metadata_fn: Symbol<unsafe extern "C" fn() -> PluginMetadata> =
            unsafe { lib.get(b"plugin_metadata")? };
        let metadata = unsafe { metadata_fn() };
        
        // Validate metadata
        self.validate_metadata(&metadata)?;
        
        // Create source instance
        let create_fn: Symbol<unsafe extern "C" fn() -> *mut dyn DocumentSource> =
            unsafe { lib.get(b"create_source")? };
        let source = unsafe { Box::from_raw(create_fn()) };
        
        // Store library to prevent unloading
        self.loaded_libs.lock().unwrap().insert(
            plugin_path.to_path_buf(),
            lib,
        );
        
        // Cache metadata
        self.metadata_cache.write().await.insert(
            metadata.id.clone(),
            metadata,
        );
        
        Ok(Arc::from(source))
    }
}

/// Plugin metadata
#[repr(C)]
#[derive(Debug, Clone)]
pub struct PluginMetadata {
    pub id: String,
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    pub api_version: String,
    pub capabilities: Vec<String>,
}
```

## Built-in Source Implementations

### PDF Source Implementation

```rust
/// PDF document source implementation
pub struct PdfSource {
    config: PdfConfig,
    parser: PdfParser,
    ocr_engine: Option<OcrEngine>,
    table_detector: TableDetector,
    metrics: SourceMetrics,
}

#[async_trait]
impl DocumentSource for PdfSource {
    fn source_id(&self) -> &str {
        "pdf"
    }
    
    fn name(&self) -> &str {
        "PDF Document Source"
    }
    
    fn version(&self) -> &str {
        "1.0.0"
    }
    
    fn supported_extensions(&self) -> &[&str] {
        &["pdf", "PDF"]
    }
    
    fn supported_mime_types(&self) -> &[&str] {
        &["application/pdf", "application/x-pdf"]
    }
    
    async fn can_handle(&self, input: &SourceInput) -> Result<bool, SourceError> {
        match input {
            SourceInput::File { path, .. } => {
                // Check file extension
                if let Some(ext) = path.extension() {
                    if self.supported_extensions().contains(&ext.to_str().unwrap_or("")) {
                        return Ok(true);
                    }
                }
                
                // Check file magic bytes
                let mut file = tokio::fs::File::open(path).await?;
                let mut header = [0u8; 5];
                file.read_exact(&mut header).await?;
                Ok(&header == b"%PDF-")
            }
            SourceInput::Memory { data, mime_type, .. } => {
                // Check MIME type
                if let Some(mime) = mime_type {
                    if self.supported_mime_types().contains(&mime.as_str()) {
                        return Ok(true);
                    }
                }
                
                // Check magic bytes
                Ok(data.starts_with(b"%PDF-"))
            }
            _ => Ok(false),
        }
    }
    
    async fn validate(&self, input: &SourceInput) -> Result<ValidationResult, SourceError> {
        let mut result = ValidationResult::default();
        
        // Get document data
        let data = match input {
            SourceInput::File { path, .. } => {
                tokio::fs::read(path).await?
            }
            SourceInput::Memory { data, .. } => data.clone(),
            _ => return Err(SourceError::UnsupportedInput),
        };
        
        // Validate PDF structure
        if !data.starts_with(b"%PDF-") {
            result.add_error("Invalid PDF header");
        }
        
        // Check file size
        if data.len() > self.config.max_file_size {
            result.add_error("File size exceeds limit");
        }
        
        // Security checks
        if self.config.security.enabled {
            let security_check = self.check_security(&data).await?;
            if !security_check.is_safe {
                result.add_error("Security check failed");
                result.security_issues = security_check.issues;
            }
        }
        
        Ok(result)
    }
    
    async fn extract(&self, input: SourceInput) -> Result<ExtractedDocument, SourceError> {
        let start_time = Instant::now();
        
        // Get document data
        let data = self.read_input(input).await?;
        
        // Parse PDF
        let parsed = self.parser.parse(&data).await?;
        
        // Extract content blocks
        let mut content_blocks = Vec::new();
        
        // Extract text blocks
        for page in &parsed.pages {
            let text_blocks = self.extract_page_text(page).await?;
            content_blocks.extend(text_blocks);
        }
        
        // Extract tables
        if self.config.extract_tables {
            let tables = self.table_detector.detect(&parsed).await?;
            content_blocks.extend(self.convert_tables_to_blocks(tables));
        }
        
        // Extract images
        if self.config.extract_images {
            let images = self.extract_images(&parsed).await?;
            content_blocks.extend(self.convert_images_to_blocks(images));
        }
        
        // Run OCR if needed
        if self.config.enable_ocr {
            if let Some(ocr) = &self.ocr_engine {
                let ocr_blocks = self.run_ocr(&parsed, &ocr).await?;
                content_blocks.extend(ocr_blocks);
            }
        }
        
        // Build document structure
        let structure = self.analyze_structure(&content_blocks).await?;
        
        // Create extracted document
        let doc = ExtractedDocument {
            id: Uuid::new_v4().to_string(),
            source_id: self.source_id().to_string(),
            metadata: self.extract_metadata(&parsed),
            content: content_blocks,
            structure,
            confidence: self.calculate_confidence(&parsed),
            metrics: ExtractionMetrics {
                extraction_time: start_time.elapsed(),
                pages_processed: parsed.pages.len(),
                blocks_extracted: content_blocks.len(),
                memory_used: self.metrics.peak_memory_usage(),
            },
        };
        
        Ok(doc)
    }
    
    async fn initialize(&mut self, config: SourceConfig) -> Result<(), SourceError> {
        // Parse PDF-specific configuration
        self.config = serde_json::from_value(config.settings)?;
        
        // Initialize OCR if enabled
        if self.config.enable_ocr {
            self.ocr_engine = Some(OcrEngine::new(&self.config.ocr)?);
        }
        
        // Initialize table detector
        self.table_detector = TableDetector::new(&self.config.table_detection)?;
        
        Ok(())
    }
    
    fn config_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "max_file_size": {
                    "type": "integer",
                    "description": "Maximum file size in bytes",
                    "default": 104857600
                },
                "enable_ocr": {
                    "type": "boolean",
                    "description": "Enable OCR for scanned PDFs",
                    "default": false
                },
                "ocr": {
                    "type": "object",
                    "properties": {
                        "language": {
                            "type": "string",
                            "default": "eng"
                        },
                        "dpi": {
                            "type": "integer",
                            "default": 300
                        }
                    }
                },
                "extract_tables": {
                    "type": "boolean",
                    "default": true
                },
                "extract_images": {
                    "type": "boolean",
                    "default": true
                },
                "security": {
                    "type": "object",
                    "properties": {
                        "enabled": {
                            "type": "boolean",
                            "default": true
                        },
                        "allow_javascript": {
                            "type": "boolean",
                            "default": false
                        },
                        "allow_external_references": {
                            "type": "boolean",
                            "default": false
                        }
                    }
                }
            }
        })
    }
}
```

### Future Source Templates

```rust
/// Template for DOCX source
pub struct DocxSource {
    config: DocxConfig,
    parser: DocxParser,
    style_analyzer: StyleAnalyzer,
}

/// Template for HTML source
pub struct HtmlSource {
    config: HtmlConfig,
    parser: HtmlParser,
    sanitizer: HtmlSanitizer,
    resource_loader: ResourceLoader,
}

/// Template for Image source
pub struct ImageSource {
    config: ImageConfig,
    decoder: ImageDecoder,
    ocr_engine: OcrEngine,
    metadata_extractor: ExifExtractor,
}

/// Template for Audio source
pub struct AudioSource {
    config: AudioConfig,
    transcriber: AudioTranscriber,
    speaker_detector: SpeakerDetector,
    metadata_extractor: AudioMetadataExtractor,
}

/// Template for Video source
pub struct VideoSource {
    config: VideoConfig,
    frame_extractor: FrameExtractor,
    ocr_engine: OcrEngine,
    transcriber: AudioTranscriber,
    scene_detector: SceneDetector,
}
```

## Configuration System

### Configuration File Format

```yaml
# sources.yaml
sources:
  pdf:
    enabled: true
    priority: 100
    timeout: 30s
    memory_limit: 1GB
    settings:
      max_file_size: 100MB
      enable_ocr: true
      ocr:
        language: eng
        dpi: 300
      extract_tables: true
      extract_images: true
      security:
        enabled: true
        allow_javascript: false
  
  docx:
    enabled: true
    priority: 90
    timeout: 20s
    settings:
      extract_styles: true
      extract_comments: true
      extract_revisions: false
  
  custom_source:
    enabled: true
    priority: 50
    plugin_path: "./plugins/custom_source.so"
    settings:
      api_endpoint: "https://api.example.com"
      api_key: "${CUSTOM_SOURCE_API_KEY}"

plugin_directories:
  - "./plugins"
  - "/usr/local/lib/neuraldocflow/sources"
  - "${HOME}/.neuraldocflow/sources"

hot_reload:
  enabled: true
  watch_interval: 5s
  
security:
  validate_plugins: true
  allow_unsigned: false
  sandbox_enabled: true
```

### Runtime Configuration

```rust
/// Runtime configuration manager
pub struct ConfigManager {
    /// Configuration file path
    config_path: PathBuf,
    
    /// Loaded configuration
    config: Arc<RwLock<SourcesConfig>>,
    
    /// Environment variable resolver
    env_resolver: EnvResolver,
    
    /// Configuration validator
    validator: ConfigValidator,
}

impl ConfigManager {
    /// Get configuration for specific source
    pub fn get_source_config(&self, source_id: &str) -> Option<SourceConfig> {
        let config = self.config.read().unwrap();
        config.sources.get(source_id).cloned()
    }
    
    /// Update source configuration
    pub async fn update_source_config(
        &self,
        source_id: &str,
        new_config: SourceConfig,
    ) -> Result<(), ConfigError> {
        // Validate new configuration
        self.validator.validate_source_config(&new_config)?;
        
        // Update configuration
        let mut config = self.config.write().unwrap();
        config.sources.insert(source_id.to_string(), new_config);
        
        // Persist to file
        self.save_config(&*config).await?;
        
        Ok(())
    }
    
    /// Watch configuration file for changes
    pub fn watch_for_changes<F>(&self, callback: F) -> Result<(), ConfigError>
    where
        F: Fn(&SourcesConfig) + Send + 'static,
    {
        let config_path = self.config_path.clone();
        let config = Arc::clone(&self.config);
        
        std::thread::spawn(move || {
            let (tx, rx) = std::sync::mpsc::channel();
            let mut watcher = notify::watcher(tx, Duration::from_secs(2)).unwrap();
            watcher.watch(&config_path, notify::RecursiveMode::NonRecursive).unwrap();
            
            while let Ok(event) = rx.recv() {
                if let notify::DebouncedEvent::Write(_) = event {
                    // Reload configuration
                    if let Ok(new_config) = Self::load_config(&config_path) {
                        *config.write().unwrap() = new_config.clone();
                        callback(&new_config);
                    }
                }
            }
        });
        
        Ok(())
    }
}
```

## Usage Examples

### Basic Usage

```rust
use neuraldocflow_sources::{SourcePluginManager, SourceInput};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize plugin manager
    let manager = SourcePluginManager::new(Default::default())?;
    
    // Enable hot-reload
    manager.enable_hot_reload()?;
    
    // Create input
    let input = SourceInput::File {
        path: PathBuf::from("document.pdf"),
        metadata: None,
    };
    
    // Find compatible sources
    let sources = manager.find_compatible_sources(&input).await?;
    
    if let Some(source) = sources.first() {
        // Extract document
        let document = source.extract(input).await?;
        
        println!("Extracted {} content blocks", document.content.len());
        println!("Document structure: {:?}", document.structure);
    }
    
    Ok(())
}
```

### Custom Source Implementation

```rust
/// Example custom source for a proprietary format
pub struct CustomSource {
    config: CustomConfig,
    api_client: ApiClient,
}

#[async_trait]
impl DocumentSource for CustomSource {
    fn source_id(&self) -> &str {
        "custom_format"
    }
    
    fn name(&self) -> &str {
        "Custom Document Format"
    }
    
    async fn extract(&self, input: SourceInput) -> Result<ExtractedDocument, SourceError> {
        // Custom extraction logic
        match input {
            SourceInput::File { path, .. } => {
                // Read file
                let data = tokio::fs::read(&path).await?;
                
                // Parse custom format
                let parsed = self.parse_custom_format(&data)?;
                
                // Convert to standard format
                let doc = self.convert_to_standard(parsed)?;
                
                Ok(doc)
            }
            _ => Err(SourceError::UnsupportedInput),
        }
    }
    
    // ... other trait methods
}

// Export plugin
#[no_mangle]
pub extern "C" fn create_source() -> *mut dyn DocumentSource {
    Box::into_raw(Box::new(CustomSource::default()))
}

#[no_mangle]
pub extern "C" fn plugin_metadata() -> PluginMetadata {
    PluginMetadata {
        id: "custom_format".to_string(),
        name: "Custom Document Format".to_string(),
        version: "1.0.0".to_string(),
        author: "Custom Corp".to_string(),
        description: "Support for proprietary document format".to_string(),
        api_version: "1.0".to_string(),
        capabilities: vec!["text_extraction".to_string()],
    }
}
```

### Adding New Source Guidelines

```rust
/// Step-by-step guide for adding new sources

// 1. Define source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewSourceConfig {
    pub option1: String,
    pub option2: bool,
    pub advanced: AdvancedOptions,
}

// 2. Implement core traits
pub struct NewSource {
    config: NewSourceConfig,
    // Add source-specific fields
}

#[async_trait]
impl DocumentSource for NewSource {
    // Implement all required methods
}

// 3. Add tests
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_new_source_extraction() {
        let source = NewSource::default();
        let input = SourceInput::Memory {
            data: b"test data".to_vec(),
            filename: Some("test.new".to_string()),
            mime_type: None,
        };
        
        let result = source.extract(input).await;
        assert!(result.is_ok());
    }
}

// 4. Create plugin manifest
// plugin.toml
[plugin]
name = "new_source"
version = "1.0.0"
authors = ["Your Name"]
description = "New document source"

[source]
id = "new_format"
extensions = ["new", "ndf"]
mime_types = ["application/x-new-format"]

[build]
crate_type = ["cdylib"]

// 5. Register with system
// In sources.yaml:
sources:
  new_format:
    enabled: true
    priority: 80
    plugin_path: "./plugins/new_source.so"
    settings:
      option1: "value"
      option2: true
```

## Performance Considerations

### Memory Management

```rust
/// Memory-efficient source implementation patterns
impl PdfSource {
    /// Stream-based processing for large files
    async fn extract_streaming(&self, input: SourceInput) -> Result<ExtractedDocument, SourceError> {
        let mut reader = self.create_reader(input).await?;
        let mut content_blocks = Vec::new();
        
        // Process in chunks
        while let Some(chunk) = reader.next_chunk().await? {
            let blocks = self.process_chunk(chunk).await?;
            content_blocks.extend(blocks);
            
            // Yield to prevent blocking
            tokio::task::yield_now().await;
        }
        
        Ok(self.build_document(content_blocks))
    }
    
    /// Memory pool for temporary buffers
    fn get_buffer(&self) -> PooledBuffer {
        self.buffer_pool.acquire(BUFFER_SIZE)
    }
}
```

### Parallelization

```rust
/// Parallel processing for multi-page documents
impl DocumentSource for PdfSource {
    async fn extract(&self, input: SourceInput) -> Result<ExtractedDocument, SourceError> {
        let parsed = self.parser.parse_input(input).await?;
        
        // Process pages in parallel
        let tasks: Vec<_> = parsed.pages
            .into_iter()
            .map(|page| {
                let extractor = self.clone();
                tokio::spawn(async move {
                    extractor.extract_page(page).await
                })
            })
            .collect();
        
        // Collect results
        let mut all_blocks = Vec::new();
        for task in tasks {
            let blocks = task.await??;
            all_blocks.extend(blocks);
        }
        
        Ok(self.build_document(all_blocks))
    }
}
```

## Security Considerations

### Plugin Validation

```rust
/// Security validator for plugins
pub struct SecurityValidator {
    /// Allowed plugin signatures
    trusted_keys: Vec<PublicKey>,
    
    /// Sandbox configuration
    sandbox_config: SandboxConfig,
}

impl SecurityValidator {
    /// Validate plugin before loading
    pub async fn validate_plugin(&self, plugin_path: &Path) -> Result<(), SecurityError> {
        // Check file permissions
        self.check_permissions(plugin_path)?;
        
        // Verify signature
        if !self.verify_signature(plugin_path).await? {
            return Err(SecurityError::InvalidSignature);
        }
        
        // Scan for malicious patterns
        self.scan_for_threats(plugin_path).await?;
        
        Ok(())
    }
    
    /// Run plugin in sandbox
    pub async fn sandbox_execute<F, R>(&self, f: F) -> Result<R, SecurityError>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        // Create sandbox
        let sandbox = Sandbox::new(&self.sandbox_config)?;
        
        // Execute with restrictions
        sandbox.execute(f).await
    }
}
```

### Input Validation

```rust
/// Input validation for sources
impl DocumentSource for PdfSource {
    async fn validate(&self, input: &SourceInput) -> Result<ValidationResult, SourceError> {
        let mut result = ValidationResult::default();
        
        // Size limits
        let size = self.get_input_size(input).await?;
        if size > self.config.max_file_size {
            result.add_error("File too large");
        }
        
        // Format validation
        if !self.validate_format(input).await? {
            result.add_error("Invalid format");
        }
        
        // Security checks
        if self.contains_malicious_content(input).await? {
            result.add_error("Security threat detected");
        }
        
        Ok(result)
    }
}
```

## Monitoring and Metrics

```rust
/// Source metrics collection
#[derive(Debug, Clone)]
pub struct SourceMetrics {
    /// Extraction performance
    pub extraction_duration: Histogram,
    
    /// Success/failure rates
    pub success_count: Counter,
    pub failure_count: Counter,
    
    /// Resource usage
    pub memory_usage: Gauge,
    pub cpu_usage: Gauge,
    
    /// Document statistics
    pub pages_processed: Counter,
    pub blocks_extracted: Counter,
}

impl SourceMetrics {
    /// Record extraction metrics
    pub fn record_extraction(&self, doc: &ExtractedDocument, duration: Duration) {
        self.extraction_duration.record(duration.as_secs_f64());
        self.success_count.increment(1);
        self.pages_processed.increment(doc.metrics.pages_processed as u64);
        self.blocks_extracted.increment(doc.content.len() as u64);
    }
}
```

## Conclusion

This modular source architecture provides:

1. **Extensibility**: Easy addition of new document sources through traits
2. **Type Safety**: Compile-time guarantees with Rust's type system
3. **Performance**: Zero-cost abstractions and efficient processing
4. **Hot-Reload**: Dynamic plugin loading without restart
5. **Security**: Comprehensive validation and sandboxing
6. **Consistency**: Unified API across all document sources

The architecture enables NeuralDocFlow to support any document format while maintaining high performance and security standards.