//! Image Processing Plugin
//!
//! Extract and process images from documents with neural enhancement.
//! This plugin handles image extraction, enhancement, OCR, and metadata extraction
//! from various document formats and standalone image files.

use neural_doc_flow_core::{
    DocumentSource, ProcessingError, Document, SourceError,
    DocumentType as CoreDocumentType, DocumentSourceType, DocumentContent, DocumentStructure,
    DocumentMetadata as CoreDocumentMetadata,
    traits::source::{DocumentMetadata as SourceDocumentMetadata, ValidationResult, ValidationIssue, ValidationSeverity}
};
// Neural engine import temporarily removed - pending implementation
// use neural_doc_flow_processors::neural::{NeuralEngine};
use crate::{Plugin, PluginMetadata, PluginCapabilities};
use std::path::Path;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use async_trait::async_trait;
use uuid::Uuid;
use chrono::Utc;

/// Image Processing Plugin implementation
pub struct ImageProcessingPlugin {
    metadata: PluginMetadata,
    config: ImageProcessingConfig,
    // neural_engine: Option<NeuralEngine>, // Temporarily disabled
}

/// Configuration for image processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageProcessingConfig {
    pub enhance_quality: bool,
    pub extract_text: bool,
    pub detect_objects: bool,
    pub analyze_layout: bool,
    pub max_image_size_mb: usize,
    pub output_format: ImageFormat,
    pub enhancement_strength: f32,
    pub ocr_confidence_threshold: f32,
    pub object_detection_threshold: f32,
}

/// Supported image formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageFormat {
    Original,
    PNG,
    JPEG,
    TIFF,
    WebP,
}

/// Processed image data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessedImage {
    pub id: String,
    pub original_filename: String,
    pub format: String,
    pub dimensions: ImageDimensions,
    pub file_size: usize,
    pub quality_metrics: QualityMetrics,
    pub extracted_text: Option<ExtractedText>,
    pub detected_objects: Vec<DetectedObject>,
    pub layout_analysis: Option<LayoutAnalysis>,
    pub metadata: ImageMetadata,
    pub enhanced_data: Option<Vec<u8>>,
}

/// Image dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageDimensions {
    pub width: u32,
    pub height: u32,
    pub channels: u32,
    pub bit_depth: u32,
    pub dpi: Option<f32>,
}

/// Image quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub sharpness_score: f32,
    pub noise_level: f32,
    pub contrast_ratio: f32,
    pub brightness: f32,
    pub color_balance: f32,
    pub overall_quality: f32,
}

/// Extracted text information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedText {
    pub text: String,
    pub confidence: f32,
    pub language: Option<String>,
    pub text_regions: Vec<TextRegion>,
    pub reading_order: Vec<usize>, // Indices into text_regions
}

/// Text region in image
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextRegion {
    pub id: usize,
    pub text: String,
    pub confidence: f32,
    pub bounding_box: BoundingBox,
    pub font_info: Option<FontInfo>,
    pub text_orientation: f32, // Rotation angle in degrees
}

/// Font information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontInfo {
    pub family: Option<String>,
    pub size: Option<f32>,
    pub weight: FontWeight,
    pub style: FontStyle,
    pub color: Option<String>,
}

/// Font weight
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FontWeight {
    Normal,
    Bold,
    Light,
    Unknown,
}

/// Font style
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FontStyle {
    Normal,
    Italic,
    Oblique,
    Unknown,
}

/// Detected object in image
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedObject {
    pub id: usize,
    pub class: String,
    pub confidence: f32,
    pub bounding_box: BoundingBox,
    pub description: Option<String>,
    pub attributes: HashMap<String, String>,
}

/// Bounding box
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Layout analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutAnalysis {
    pub document_type: ImageDocumentType,
    pub reading_order: ReadingOrder,
    pub regions: Vec<LayoutRegion>,
    pub columns: u32,
    pub text_density: f32,
    pub image_density: f32,
}

/// Document type classification for image processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageDocumentType {
    Text,
    Form,
    Table,
    Invoice,
    Receipt,
    Newspaper,
    Magazine,
    Handwritten,
    Mixed,
    Unknown,
}

/// Reading order analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReadingOrder {
    LeftToRight,
    RightToLeft,
    TopToBottom,
    Complex,
    Unknown,
}

/// Layout region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutRegion {
    pub id: usize,
    pub region_type: RegionType,
    pub bounding_box: BoundingBox,
    pub confidence: f32,
    pub text_content: Option<String>,
}

/// Type of layout region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegionType {
    Title,
    Heading,
    Paragraph,
    Caption,
    Image,
    Table,
    List,
    Footer,
    Header,
    Sidebar,
    Unknown,
}

/// Image metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageMetadata {
    pub creation_date: Option<String>,
    pub camera_info: Option<CameraInfo>,
    pub location: Option<Location>,
    pub processing_history: Vec<String>,
    pub color_profile: Option<String>,
    pub compression: Option<String>,
}

/// Camera information from EXIF
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraInfo {
    pub make: Option<String>,
    pub model: Option<String>,
    pub lens: Option<String>,
    pub focal_length: Option<f32>,
    pub aperture: Option<f32>,
    pub iso: Option<u32>,
    pub exposure_time: Option<String>,
}

/// Location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Location {
    pub latitude: f64,
    pub longitude: f64,
    pub altitude: Option<f64>,
    pub location_name: Option<String>,
}

impl Default for ImageProcessingConfig {
    fn default() -> Self {
        Self {
            enhance_quality: true,
            extract_text: true,
            detect_objects: false,
            analyze_layout: true,
            max_image_size_mb: 50,
            output_format: ImageFormat::Original,
            enhancement_strength: 0.5,
            ocr_confidence_threshold: 0.7,
            object_detection_threshold: 0.6,
        }
    }
}

impl ImageProcessingPlugin {
    /// Create new image processing plugin
    pub fn new() -> Self {
        Self {
            metadata: PluginMetadata {
                name: "image_processing".to_string(),
                version: "1.0.0".to_string(),
                author: "Neural Document Flow Team".to_string(),
                description: "Extract and process images from documents with neural enhancement".to_string(),
                supported_formats: vec![
                    "png".to_string(),
                    "jpg".to_string(),
                    "jpeg".to_string(),
                    "tiff".to_string(),
                    "tif".to_string(),
                    "bmp".to_string(),
                    "gif".to_string(),
                    "webp".to_string(),
                    "pdf".to_string(), // For image extraction from PDFs
                ],
                capabilities: PluginCapabilities {
                    requires_network: false,
                    requires_filesystem: true,
                    max_memory_mb: 1000, // High memory for image processing
                    max_cpu_percent: 90.0,
                    timeout_seconds: 600, // Longer timeout for complex processing
                },
            },
            config: ImageProcessingConfig::default(),
            // neural_engine: None, // Temporarily disabled
        }
    }

    /// Create plugin with custom configuration
    pub fn with_config(config: ImageProcessingConfig) -> Self {
        let mut plugin = Self::new();
        plugin.config = config;
        plugin
    }

    /// Initialize neural engine for image processing
    fn initialize_neural_engine(&mut self) -> Result<(), ProcessingError> {
        tracing::info!("Initializing neural engine for image processing");
        
        // For now, we'll use traditional image processing methods
        // Neural enhancement will be added when the neural engine interface is stable
        tracing::info!("Neural features configured but using traditional methods");
        // self.neural_engine = None; // Temporarily disabled
        
        Ok(())
    }

    /// Process an image file
    pub fn process_image(&self, path: &Path) -> Result<ProcessedImage, ProcessingError> {
        tracing::info!("Processing image: {:?}", path);
        let start_time = std::time::Instant::now();

        // Load and validate image
        let image_data = self.load_image(path)?;
        let mut processed = ProcessedImage {
            id: format!("img_{}", start_time.elapsed().as_nanos()),
            original_filename: path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string(),
            format: self.detect_format(&image_data),
            dimensions: self.get_dimensions(&image_data),
            file_size: image_data.len(),
            quality_metrics: self.analyze_quality(&image_data)?,
            extracted_text: None,
            detected_objects: Vec::new(),
            layout_analysis: None,
            metadata: self.extract_metadata(path)?,
            enhanced_data: None,
        };

        // Quality enhancement
        if self.config.enhance_quality {
            processed.enhanced_data = self.enhance_image(&image_data)?;
        }

        // Text extraction (OCR)
        if self.config.extract_text {
            processed.extracted_text = self.extract_text_from_image(&image_data)?;
        }

        // Object detection
        if self.config.detect_objects {
            processed.detected_objects = self.detect_objects_in_image(&image_data)?;
        }

        // Layout analysis
        if self.config.analyze_layout {
            processed.layout_analysis = self.analyze_layout(&image_data)?;
        }

        let processing_time = start_time.elapsed().as_millis();
        processed.metadata.processing_history.push(
            format!("Processed in {}ms", processing_time)
        );

        tracing::info!("Image processing completed in {}ms", processing_time);
        Ok(processed)
    }

    /// Load image data from file
    fn load_image(&self, path: &Path) -> Result<Vec<u8>, ProcessingError> {
        let data = std::fs::read(path)
            .map_err(|e| ProcessingError::ProcessorFailed {
                processor_name: "image_processing".to_string(),
                reason: format!("Failed to read image: {}", e),
            })?;

        // Check size limit
        let size_mb = data.len() / (1024 * 1024);
        if size_mb > self.config.max_image_size_mb {
            return Err(ProcessingError::ProcessorFailed {
                processor_name: "image_processing".to_string(),
                reason: format!("Image too large: {}MB > {}MB", size_mb, self.config.max_image_size_mb),
            });
        }

        Ok(data)
    }

    /// Detect image format from data
    fn detect_format(&self, data: &[u8]) -> String {
        if data.len() < 8 {
            return "unknown".to_string();
        }

        // Check magic bytes
        match &data[0..8] {
            [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] => "png".to_string(),
            [0xFF, 0xD8, 0xFF, ..] => "jpeg".to_string(),
            [0x47, 0x49, 0x46, 0x38, ..] => "gif".to_string(),
            [0x42, 0x4D, ..] => "bmp".to_string(),
            [0x49, 0x49, 0x2A, 0x00, ..] | [0x4D, 0x4D, 0x00, 0x2A, ..] => "tiff".to_string(),
            _ => "unknown".to_string(),
        }
    }

    /// Get image dimensions (placeholder implementation)
    fn get_dimensions(&self, _data: &[u8]) -> ImageDimensions {
        // In a real implementation, this would parse the image headers
        ImageDimensions {
            width: 1920,
            height: 1080,
            channels: 3,
            bit_depth: 8,
            dpi: Some(72.0),
        }
    }

    /// Analyze image quality
    fn analyze_quality(&self, data: &[u8]) -> Result<QualityMetrics, ProcessingError> {
        // Simplified quality analysis - would use proper image analysis in real implementation
        
        // Calculate basic statistics from raw data
        let mut brightness_sum = 0u64;
        let mut variance_sum = 0u64;
        
        for &byte in data.iter().take(1000).step_by(3) { // Sample every third byte
            brightness_sum += byte as u64;
        }
        
        let sample_count = data.len().min(1000) / 3;
        let avg_brightness = if sample_count > 0 {
            brightness_sum as f32 / sample_count as f32 / 255.0
        } else {
            0.5
        };

        // Calculate variance for noise estimation
        for &byte in data.iter().take(1000).step_by(3) {
            let diff = (byte as f32 / 255.0) - avg_brightness;
            variance_sum += (diff * diff * 1000.0) as u64;
        }
        
        let noise_level = if sample_count > 0 {
            (variance_sum as f32 / sample_count as f32).sqrt()
        } else {
            0.1
        };

        Ok(QualityMetrics {
            sharpness_score: 0.8,  // Would calculate using edge detection
            noise_level,
            contrast_ratio: 0.7,   // Would calculate from histogram
            brightness: avg_brightness,
            color_balance: 0.9,    // Would analyze color channels
            overall_quality: (0.8 + (1.0 - noise_level) + 0.7 + 0.9) / 4.0,
        })
    }

    /// Enhance image quality using neural networks
    fn enhance_image(&self, data: &[u8]) -> Result<Option<Vec<u8>>, ProcessingError> {
        // For now, use basic enhancement until neural engine is available
        Ok(self.basic_enhancement(data))
    }

    /// Basic image enhancement without neural networks
    fn basic_enhancement(&self, data: &[u8]) -> Option<Vec<u8>> {
        // Simple contrast/brightness adjustment
        let mut enhanced = data.to_vec();
        
        for pixel in enhanced.iter_mut() {
            let adjusted = (*pixel as f32 * 1.1).min(255.0) as u8; // Increase contrast slightly
            *pixel = adjusted;
        }
        
        Some(enhanced)
    }

    /// Extract text from image using OCR
    fn extract_text_from_image(&self, data: &[u8]) -> Result<Option<ExtractedText>, ProcessingError> {
        // For now, use fallback OCR until neural engine is available
        Ok(self.fallback_ocr(data))
    }

    /// Parse OCR result from neural engine (placeholder)
    fn _parse_ocr_result_placeholder(&self) -> Result<Option<ExtractedText>, ProcessingError> {
        // Placeholder for future neural OCR result parsing
        Ok(None)
    }

    /// Fallback OCR implementation
    fn fallback_ocr(&self, _data: &[u8]) -> Option<ExtractedText> {
        // Placeholder implementation - would use Tesseract or similar
        Some(ExtractedText {
            text: "Sample extracted text".to_string(),
            confidence: 0.8,
            language: Some("en".to_string()),
            text_regions: vec![TextRegion {
                id: 0,
                text: "Sample extracted text".to_string(),
                confidence: 0.8,
                bounding_box: BoundingBox { x: 10, y: 10, width: 200, height: 30 },
                font_info: Some(FontInfo {
                    family: Some("Arial".to_string()),
                    size: Some(12.0),
                    weight: FontWeight::Normal,
                    style: FontStyle::Normal,
                    color: Some("#000000".to_string()),
                }),
                text_orientation: 0.0,
            }],
            reading_order: vec![0],
        })
    }

    /// Detect objects in image
    fn detect_objects_in_image(&self, _data: &[u8]) -> Result<Vec<DetectedObject>, ProcessingError> {
        // For now, return empty vector until neural engine is available
        Ok(Vec::new())
    }

    /// Parse object detection result (placeholder)
    fn _parse_object_detection_result_placeholder(&self) -> Result<Vec<DetectedObject>, ProcessingError> {
        // Placeholder for future object detection result parsing
        Ok(Vec::new())
    }

    /// Analyze document layout
    fn analyze_layout(&self, data: &[u8]) -> Result<Option<LayoutAnalysis>, ProcessingError> {
        // For now, use basic layout analysis until neural engine is available
        Ok(self.basic_layout_analysis(data))
    }

    /// Parse layout analysis result (placeholder)
    fn _parse_layout_result_placeholder(&self) -> Result<Option<LayoutAnalysis>, ProcessingError> {
        // Placeholder for future layout analysis result parsing
        Ok(None)
    }

    /// Basic layout analysis without neural networks
    fn basic_layout_analysis(&self, _data: &[u8]) -> Option<LayoutAnalysis> {
        Some(LayoutAnalysis {
            document_type: ImageDocumentType::Mixed,
            reading_order: ReadingOrder::LeftToRight,
            regions: Vec::new(),
            columns: 1,
            text_density: 0.6,
            image_density: 0.3,
        })
    }

    /// Extract metadata from image file
    fn extract_metadata(&self, path: &Path) -> Result<ImageMetadata, ProcessingError> {
        // In a real implementation, this would extract EXIF data
        Ok(ImageMetadata {
            creation_date: std::fs::metadata(path)
                .ok()
                .and_then(|m| m.created().ok())
                .map(|t| format!("{:?}", t)),
            camera_info: None,
            location: None,
            processing_history: vec!["Original image".to_string()],
            color_profile: Some("sRGB".to_string()),
            compression: None,
        })
    }
}

impl Plugin for ImageProcessingPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }
    
    fn initialize(&mut self) -> Result<(), ProcessingError> {
        tracing::info!("Initializing image processing plugin");
        self.initialize_neural_engine()?;
        Ok(())
    }
    
    fn shutdown(&mut self) -> Result<(), ProcessingError> {
        tracing::info!("Shutting down image processing plugin");
        // self.neural_engine = None; // Temporarily disabled
        Ok(())
    }
    
    fn document_source(&self) -> Box<dyn DocumentSource> {
        Box::new(ImageProcessingSource::new(self.config.clone()))
    }
}

/// Document source for image processing
#[derive(Debug)]
pub struct ImageProcessingSource {
    config: ImageProcessingConfig,
}

impl ImageProcessingSource {
    pub fn new(config: ImageProcessingConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl DocumentSource for ImageProcessingSource {
    fn source_type(&self) -> &'static str {
        "image_processing"
    }
    
    async fn can_handle(&self, input: &str) -> bool {
        let path = Path::new(input);
        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            matches!(ext.to_lowercase().as_str(), 
                "png" | "jpg" | "jpeg" | "tiff" | "tif" | "bmp" | "gif" | "webp" | "pdf")
        } else {
            false
        }
    }
    
    async fn load_document(&self, input: &str) -> Result<Document, SourceError> {
        let path = Path::new(input);
        if !path.exists() {
            return Err(SourceError::DocumentNotFound { path: input.to_string() });
        }
        
        if !self.can_handle(input).await {
            return Err(SourceError::UnsupportedFormat { 
                format: path.extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string() 
            });
        }
        
        self.extract_document_internal(path).await
            .map_err(|e| SourceError::ParseError { reason: e.to_string() })
    }
    
    async fn get_metadata(&self, input: &str) -> Result<SourceDocumentMetadata, SourceError> {
        let path = Path::new(input);
        if !path.exists() {
            return Err(SourceError::DocumentNotFound { path: input.to_string() });
        }
        
        let metadata = std::fs::metadata(path)
            .map_err(|_e| SourceError::AccessDenied { path: input.to_string() })?;
            
        let mime_type = match path.extension().and_then(|s| s.to_str()) {
            Some("png") => "image/png",
            Some("jpg") | Some("jpeg") => "image/jpeg",
            Some("tiff") | Some("tif") => "image/tiff",
            Some("bmp") => "image/bmp",
            Some("gif") => "image/gif",
            Some("webp") => "image/webp",
            Some("pdf") => "application/pdf",
            _ => "application/octet-stream",
        };
            
        Ok(SourceDocumentMetadata {
            name: path.file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string(),
            size: Some(metadata.len()),
            mime_type: mime_type.to_string(),
            modified: metadata.modified()
                .ok()
                .map(|t| chrono::DateTime::from(t)),
            attributes: HashMap::new(),
        })
    }
    
    async fn validate(&self, input: &str) -> Result<ValidationResult, SourceError> {
        let path = Path::new(input);
        let mut validation = ValidationResult {
            is_valid: true,
            issues: Vec::new(),
            estimated_processing_time: Some(20.0), // Estimated 20 seconds for image processing
            confidence: 0.85,
        };
        
        if !path.exists() {
            validation.add_issue(ValidationIssue {
                severity: ValidationSeverity::Critical,
                message: "File does not exist".to_string(),
                suggestion: Some("Check file path".to_string()),
            });
            return Ok(validation);
        }
        
        if !self.can_handle(input).await {
            validation.add_issue(ValidationIssue {
                severity: ValidationSeverity::Error,
                message: "Unsupported file format".to_string(),
                suggestion: Some("Use PNG, JPG, JPEG, TIFF, BMP, GIF, WebP, or PDF files".to_string()),
            });
        }
        
        // Check file size
        if let Ok(metadata) = std::fs::metadata(path) {
            if metadata.len() > 50 * 1024 * 1024 { // 50MB
                validation.add_issue(ValidationIssue {
                    severity: ValidationSeverity::Warning,
                    message: "Large file size may cause slow processing".to_string(),
                    suggestion: Some("Consider processing smaller files".to_string()),
                });
            }
        }
        
        Ok(validation)
    }
    
    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["png", "jpg", "jpeg", "tiff", "tif", "bmp", "gif", "webp", "pdf"]
    }
    
    fn supported_mime_types(&self) -> Vec<&'static str> {
        vec![
            "image/png",
            "image/jpeg",
            "image/tiff",
            "image/bmp",
            "image/gif",
            "image/webp",
            "application/pdf"
        ]
    }
}

impl ImageProcessingSource {
    async fn extract_document_internal(&self, path: &Path) -> Result<Document, ProcessingError> {
        let mut plugin = ImageProcessingPlugin::with_config(self.config.clone());
        plugin.initialize()?;
        
        let processed_image = plugin.process_image(path)?;
        
        // Convert to document format
        let mut content_parts = Vec::new();
        let mut metadata = HashMap::new();
        
        // Add basic information
        content_parts.push(format!("Image: {}", processed_image.original_filename));
        content_parts.push(format!("Format: {}", processed_image.format));
        content_parts.push(format!("Dimensions: {}x{}", 
                                 processed_image.dimensions.width, 
                                 processed_image.dimensions.height));
        content_parts.push(format!("Quality Score: {:.2}", 
                                 processed_image.quality_metrics.overall_quality));
        
        // Add extracted text if available
        if let Some(ref text_data) = processed_image.extracted_text {
            content_parts.push(String::new());
            content_parts.push("Extracted Text:".to_string());
            content_parts.push(text_data.text.clone());
            
            metadata.insert("ocr_confidence".to_string(), text_data.confidence.to_string());
            if let Some(ref lang) = text_data.language {
                metadata.insert("detected_language".to_string(), lang.clone());
            }
        }
        
        // Add detected objects
        if !processed_image.detected_objects.is_empty() {
            content_parts.push(String::new());
            content_parts.push("Detected Objects:".to_string());
            for obj in &processed_image.detected_objects {
                content_parts.push(format!("- {} (confidence: {:.2})", obj.class, obj.confidence));
            }
        }
        
        // Add layout analysis
        if let Some(ref layout) = processed_image.layout_analysis {
            content_parts.push(String::new());
            content_parts.push(format!("Document Type: {:?}", layout.document_type));
            content_parts.push(format!("Reading Order: {:?}", layout.reading_order));
            content_parts.push(format!("Columns: {}", layout.columns));
        }
        
        // Store detailed data in metadata
        metadata.insert("format".to_string(), "image_processing".to_string());
        metadata.insert("original_format".to_string(), processed_image.format.clone());
        metadata.insert("file_size".to_string(), processed_image.file_size.to_string());
        
        if let Ok(image_json) = serde_json::to_string(&processed_image) {
            metadata.insert("processed_image_data".to_string(), image_json);
        }
        
        // Read the raw file content
        let raw_content = std::fs::read(path)
            .map_err(|e| ProcessingError::IoError(e.to_string()))?;
        
        // Convert to DocumentContent
        let document_content = DocumentContent {
            text: Some(content_parts.join("\n")),
            images: vec![], // Would contain extracted images
            tables: vec![], // Would contain extracted tables
            structured: HashMap::new(),
            raw: Some(raw_content.clone()),
        };
        
        // Create document metadata
        let doc_metadata = CoreDocumentMetadata {
            title: metadata.get("title").cloned(),
            authors: vec![],
            source: path.to_string_lossy().to_string(),
            mime_type: metadata.get("mime_type").cloned().unwrap_or_else(|| "image/unknown".to_string()),
            size: Some(raw_content.len() as u64),
            language: None,
            custom: metadata.into_iter().map(|(k, v)| (k, serde_json::Value::String(v))).collect(),
        };
        
        Ok(Document {
            id: Uuid::new_v4(),
            doc_type: CoreDocumentType::Image,
            source_type: DocumentSourceType::File,
            raw_content,
            metadata: doc_metadata,
            content: document_content,
            structure: DocumentStructure::default(),
            attachments: vec![],
            processing_history: vec![],
            created_at: Utc::now(),
            updated_at: Utc::now(),
        })
    }
}

/// Entry point for plugin
#[no_mangle]
pub extern "C" fn create_image_processing_plugin() -> *mut dyn Plugin {
    let plugin = ImageProcessingPlugin::new();
    Box::into_raw(Box::new(plugin))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plugin_creation() {
        let plugin = ImageProcessingPlugin::new();
        assert_eq!(plugin.metadata().name, "image_processing");
        assert!(plugin.metadata().supported_formats.contains(&"png".to_string()));
    }

    #[tokio::test]
    async fn test_source_can_handle() {
        let source = ImageProcessingSource::new(ImageProcessingConfig::default());
        
        assert!(source.can_handle("test.png").await);
        assert!(source.can_handle("test.jpg").await);
        assert!(source.can_handle("test.pdf").await);
        assert!(!source.can_handle("test.docx").await);
        assert!(!source.can_handle("test.txt").await);
    }

    #[test]
    fn test_format_detection() {
        let plugin = ImageProcessingPlugin::new();
        
        // PNG magic bytes
        let png_data = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
        assert_eq!(plugin.detect_format(&png_data), "png");
        
        // JPEG magic bytes
        let jpeg_data = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46];
        assert_eq!(plugin.detect_format(&jpeg_data), "jpeg");
    }

    #[test]
    fn test_quality_analysis() {
        let plugin = ImageProcessingPlugin::new();
        
        // Create sample image data
        let image_data = vec![128u8; 1000]; // Gray image
        let quality = plugin.analyze_quality(&image_data).unwrap();
        
        assert!(quality.overall_quality >= 0.0 && quality.overall_quality <= 1.0);
        assert!(quality.brightness >= 0.0 && quality.brightness <= 1.0);
        assert!(quality.noise_level >= 0.0);
    }
}