//! Core types for neural document flow processors

use crate::config::ModelType;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use uuid::Uuid;

/// Content block representing a piece of extracted document content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentBlock {
    /// Unique identifier for this content block
    pub id: String,

    /// Type of content (text, table, image, layout, etc.)
    pub content_type: String,

    /// Extracted text content (if applicable)
    pub text: Option<String>,

    /// Binary data (for images, etc.)
    pub binary_data: Option<Vec<u8>>,

    /// Position information
    pub position: Position,

    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,

    /// Additional metadata
    pub metadata: HashMap<String, String>,

    /// Relationships to other blocks
    pub relationships: Vec<Relationship>,
}

impl ContentBlock {
    /// Create a new content block
    pub fn new(content_type: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            content_type: content_type.to_string(),
            text: None,
            binary_data: None,
            position: Position::default(),
            confidence: 0.0,
            metadata: HashMap::new(),
            relationships: Vec::new(),
        }
    }

    /// Create a text content block
    pub fn text_block(text: String) -> Self {
        let mut block = Self::new("text");
        block.text = Some(text);
        block.confidence = 0.95; // Default high confidence for text
        block
    }

    /// Create a table content block
    pub fn table_block(table_data: String) -> Self {
        let mut block = Self::new("table");
        block.text = Some(table_data);
        block.confidence = 0.9; // Default confidence for tables
        block
    }

    /// Create an image content block
    pub fn image_block(image_data: Vec<u8>) -> Self {
        let mut block = Self::new("image");
        block.binary_data = Some(image_data);
        block.confidence = 1.0; // Perfect confidence for binary data
        block
    }

    /// Add metadata to the block
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Set position information
    pub fn with_position(mut self, position: Position) -> Self {
        self.position = position;
        self
    }

    /// Set confidence score
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Add a relationship to another block
    pub fn add_relationship(&mut self, relationship: Relationship) {
        self.relationships.push(relationship);
    }

    /// Get the content as text (if available)
    pub fn get_text(&self) -> Option<&str> {
        self.text.as_deref()
    }

    /// Check if this block has high confidence
    pub fn is_high_confidence(&self, threshold: f32) -> bool {
        self.confidence >= threshold
    }
}

/// Position information for content blocks
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Position {
    /// Page number (0-based)
    pub page: usize,

    /// X coordinate (0.0 to 1.0, relative to page width)
    pub x: f32,

    /// Y coordinate (0.0 to 1.0, relative to page height)
    pub y: f32,

    /// Width (0.0 to 1.0, relative to page width)
    pub width: f32,

    /// Height (0.0 to 1.0, relative to page height)
    pub height: f32,
}

impl Default for Position {
    fn default() -> Self {
        Self {
            page: 0,
            x: 0.0,
            y: 0.0,
            width: 1.0,
            height: 1.0,
        }
    }
}

impl Position {
    /// Create a new position
    pub fn new(page: usize, x: f32, y: f32, width: f32, height: f32) -> Self {
        Self { page, x, y, width, height }
    }

    /// Calculate area of this position
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Check if this position overlaps with another
    pub fn overlaps(&self, other: &Position) -> bool {
        if self.page != other.page {
            return false;
        }

        let x_overlap = self.x < other.x + other.width && other.x < self.x + self.width;
        let y_overlap = self.y < other.y + other.height && other.y < self.y + self.height;
        
        x_overlap && y_overlap
    }

    /// Calculate intersection area with another position
    pub fn intersection_area(&self, other: &Position) -> f32 {
        if !self.overlaps(other) {
            return 0.0;
        }

        let x_overlap = (self.x + self.width).min(other.x + other.width) - self.x.max(other.x);
        let y_overlap = (self.y + self.height).min(other.y + other.height) - self.y.max(other.y);
        
        x_overlap * y_overlap
    }
}

/// Relationship between content blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    /// Type of relationship
    pub relationship_type: RelationshipType,

    /// Target block ID
    pub target_id: String,

    /// Confidence in this relationship
    pub confidence: f32,

    /// Additional properties
    pub properties: HashMap<String, String>,
}

/// Types of relationships between content blocks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RelationshipType {
    /// Block is above another block
    Above,

    /// Block is below another block
    Below,

    /// Block is to the left of another block
    LeftOf,

    /// Block is to the right of another block
    RightOf,

    /// Block contains another block
    Contains,

    /// Block is contained within another block
    ContainedBy,

    /// Block is part of the same table
    SameTable,

    /// Block is part of the same paragraph
    SameParagraph,

    /// Block is part of the same section
    SameSection,

    /// Custom relationship type
    Custom(String),
}

/// Enhanced content after neural processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedContent {
    /// Enhanced content blocks
    pub blocks: Vec<ContentBlock>,

    /// Overall confidence score
    pub confidence: f32,

    /// Processing time taken
    pub processing_time: Duration,

    /// List of enhancements applied
    pub enhancements: Vec<String>,

    /// Neural features extracted
    pub neural_features: Option<NeuralFeatures>,

    /// Quality assessment results
    pub quality_assessment: Option<QualityAssessment>,
}

impl EnhancedContent {
    /// Create new enhanced content
    pub fn new(blocks: Vec<ContentBlock>) -> Self {
        let confidence = if blocks.is_empty() {
            0.0
        } else {
            blocks.iter().map(|b| b.confidence).sum::<f32>() / blocks.len() as f32
        };

        Self {
            blocks,
            confidence,
            processing_time: Duration::from_millis(0),
            enhancements: Vec::new(),
            neural_features: None,
            quality_assessment: None,
        }
    }

    /// Add an enhancement type to the list
    pub fn add_enhancement(&mut self, enhancement: &str) {
        if !self.enhancements.contains(&enhancement.to_string()) {
            self.enhancements.push(enhancement.to_string());
        }
    }

    /// Get blocks of a specific type
    pub fn get_blocks_by_type(&self, content_type: &str) -> Vec<&ContentBlock> {
        self.blocks
            .iter()
            .filter(|block| block.content_type == content_type)
            .collect()
    }

    /// Get high confidence blocks
    pub fn get_high_confidence_blocks(&self, threshold: f32) -> Vec<&ContentBlock> {
        self.blocks
            .iter()
            .filter(|block| block.confidence >= threshold)
            .collect()
    }

    /// Calculate processing statistics
    pub fn get_statistics(&self) -> ContentStatistics {
        let mut stats = ContentStatistics::default();
        
        stats.total_blocks = self.blocks.len();
        stats.average_confidence = self.confidence;
        stats.processing_time = self.processing_time;

        for block in &self.blocks {
            match block.content_type.as_str() {
                "text" => stats.text_blocks += 1,
                "table" => stats.table_blocks += 1,
                "image" => stats.image_blocks += 1,
                _ => stats.other_blocks += 1,
            }

            if block.confidence >= 0.9 {
                stats.high_confidence_blocks += 1;
            } else if block.confidence >= 0.7 {
                stats.medium_confidence_blocks += 1;
            } else {
                stats.low_confidence_blocks += 1;
            }
        }

        stats
    }
}

/// Neural features extracted from content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralFeatures {
    /// Text-based features
    pub text_features: Vec<f32>,

    /// Layout-based features  
    pub layout_features: Vec<f32>,

    /// Table-specific features
    pub table_features: Vec<f32>,

    /// Image-based features
    pub image_features: Vec<f32>,

    /// Combined feature vector
    pub combined_features: Vec<f32>,

    /// Feature extraction metadata
    pub metadata: HashMap<String, f32>,
}

impl Default for NeuralFeatures {
    fn default() -> Self {
        Self {
            text_features: Vec::new(),
            layout_features: Vec::new(),
            table_features: Vec::new(),
            image_features: Vec::new(),
            combined_features: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

impl NeuralFeatures {
    /// Create new empty neural features
    pub fn new() -> Self {
        Self::default()
    }

    /// Combine all feature vectors
    pub fn combine_features(&mut self) {
        self.combined_features.clear();
        self.combined_features.extend(&self.text_features);
        self.combined_features.extend(&self.layout_features);
        self.combined_features.extend(&self.table_features);
        self.combined_features.extend(&self.image_features);
    }

    /// Get feature vector size
    pub fn size(&self) -> usize {
        self.combined_features.len()
    }

    /// Normalize features to 0-1 range
    pub fn normalize(&mut self) {
        if self.combined_features.is_empty() {
            return;
        }

        let min_val = self.combined_features.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = self.combined_features.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        if max_val > min_val {
            let range = max_val - min_val;
            for feature in &mut self.combined_features {
                *feature = (*feature - min_val) / range;
            }
        }
    }
}

/// Processing metrics for neural operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetrics {
    /// Total number of blocks processed
    pub total_blocks_processed: usize,

    /// Total processing time
    pub total_processing_time: Duration,

    /// Average confidence across all processing
    pub average_confidence: f32,

    /// Model-specific metrics
    pub model_metrics: HashMap<ModelType, ModelMetrics>,

    /// Performance metrics
    pub performance: PerformanceMetrics,

    /// Error statistics
    pub errors: ErrorStatistics,
}

impl Default for ProcessingMetrics {
    fn default() -> Self {
        Self {
            total_blocks_processed: 0,
            total_processing_time: Duration::from_millis(0),
            average_confidence: 0.0,
            model_metrics: HashMap::new(),
            performance: PerformanceMetrics::default(),
            errors: ErrorStatistics::default(),
        }
    }
}

/// Model-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Number of inferences performed
    pub inference_count: usize,

    /// Total inference time
    pub total_inference_time: Duration,

    /// Average inference time
    pub average_inference_time: Duration,

    /// Average confidence for this model
    pub average_confidence: f32,

    /// Memory usage in bytes
    pub memory_usage: usize,
}

impl Default for ModelMetrics {
    fn default() -> Self {
        Self {
            inference_count: 0,
            total_inference_time: Duration::from_millis(0),
            average_inference_time: Duration::from_millis(0),
            average_confidence: 0.0,
            memory_usage: 0,
        }
    }
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Peak memory usage in bytes
    pub peak_memory_usage: usize,

    /// Average CPU utilization
    pub average_cpu_usage: f32,

    /// Throughput (blocks per second)
    pub throughput: f32,

    /// Cache hit rate
    pub cache_hit_rate: f32,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            peak_memory_usage: 0,
            average_cpu_usage: 0.0,
            throughput: 0.0,
            cache_hit_rate: 0.0,
        }
    }
}

/// Error statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStatistics {
    /// Total number of errors
    pub total_errors: usize,

    /// Errors by type
    pub errors_by_type: HashMap<String, usize>,

    /// Failed blocks
    pub failed_blocks: usize,

    /// Recovery attempts
    pub recovery_attempts: usize,

    /// Successful recoveries
    pub successful_recoveries: usize,
}

impl Default for ErrorStatistics {
    fn default() -> Self {
        Self {
            total_errors: 0,
            errors_by_type: HashMap::new(),
            failed_blocks: 0,
            recovery_attempts: 0,
            successful_recoveries: 0,
        }
    }
}

/// Confidence score breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceScore {
    /// Overall confidence
    pub overall: f32,

    /// Text accuracy confidence
    pub text_accuracy: f32,

    /// Layout accuracy confidence
    pub layout_accuracy: f32,

    /// Table accuracy confidence
    pub table_accuracy: f32,

    /// Image processing confidence
    pub image_accuracy: f32,

    /// Quality assessment confidence
    pub quality_confidence: f32,
}

impl Default for ConfidenceScore {
    fn default() -> Self {
        Self {
            overall: 0.0,
            text_accuracy: 0.0,
            layout_accuracy: 0.0,
            table_accuracy: 0.0,
            image_accuracy: 0.0,
            quality_confidence: 0.0,
        }
    }
}

impl ConfidenceScore {
    /// Calculate overall confidence from individual scores
    pub fn calculate_overall(&mut self) {
        let scores = vec![
            self.text_accuracy,
            self.layout_accuracy,
            self.table_accuracy,
            self.image_accuracy,
            self.quality_confidence,
        ];

        let valid_scores: Vec<f32> = scores.into_iter().filter(|&score| score > 0.0).collect();
        
        self.overall = if valid_scores.is_empty() {
            0.0
        } else {
            valid_scores.iter().sum::<f32>() / valid_scores.len() as f32
        };
    }

    /// Check if confidence meets minimum threshold
    pub fn meets_threshold(&self, threshold: f32) -> bool {
        self.overall >= threshold
    }
}

/// Enhancement types applied to content
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Enhancement {
    /// Neural text correction
    TextCorrection {
        original_confidence: f32,
        corrected_confidence: f32,
        changes_made: usize,
    },

    /// Layout structure enhancement
    LayoutEnhancement {
        regions_detected: usize,
        hierarchy_levels: usize,
    },

    /// Table structure detection
    TableDetection {
        tables_found: usize,
        average_confidence: f32,
    },

    /// Image processing enhancement
    ImageProcessing {
        resolution_enhanced: bool,
        ocr_applied: bool,
    },

    /// Quality validation
    QualityValidation {
        passed: bool,
        issues_found: Vec<String>,
    },

    /// Custom enhancement
    Custom {
        enhancement_type: String,
        parameters: HashMap<String, f32>,
    },
}

/// Quality assessment results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityAssessment {
    /// Overall quality score
    pub overall_score: f32,

    /// Individual quality metrics
    pub metrics: QualityMetrics,

    /// Issues detected
    pub issues: Vec<QualityIssue>,

    /// Recommendations for improvement
    pub recommendations: Vec<String>,
}

/// Individual quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Text extraction quality
    pub text_quality: f32,

    /// Layout detection quality
    pub layout_quality: f32,

    /// Table extraction quality
    pub table_quality: f32,

    /// Overall consistency
    pub consistency: f32,

    /// Completeness score
    pub completeness: f32,
}

/// Quality issues detected
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    /// Issue type
    pub issue_type: QualityIssueType,

    /// Severity level
    pub severity: Severity,

    /// Description
    pub description: String,

    /// Block ID where issue was found
    pub block_id: Option<String>,

    /// Confidence in issue detection
    pub confidence: f32,
}

/// Types of quality issues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityIssueType {
    LowConfidence,
    InconsistentFormatting,
    MissingContent,
    OverlappingBlocks,
    MalformedTable,
    UnrecognizedText,
    LayoutInconsistency,
    Custom(String),
}

impl std::fmt::Display for QualityIssueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QualityIssueType::LowConfidence => write!(f, "low_confidence"),
            QualityIssueType::InconsistentFormatting => write!(f, "inconsistent_formatting"),
            QualityIssueType::MissingContent => write!(f, "missing_content"),
            QualityIssueType::OverlappingBlocks => write!(f, "overlapping_blocks"),
            QualityIssueType::MalformedTable => write!(f, "malformed_table"),
            QualityIssueType::UnrecognizedText => write!(f, "unrecognized_text"),
            QualityIssueType::LayoutInconsistency => write!(f, "layout_inconsistency"),
            QualityIssueType::Custom(name) => write!(f, "custom_{}", name),
        }
    }
}

/// Issue severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Severity {
    Low,
    Medium,
    High,
    Critical,
}

/// Content statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentStatistics {
    /// Total number of blocks
    pub total_blocks: usize,

    /// Number of text blocks
    pub text_blocks: usize,

    /// Number of table blocks
    pub table_blocks: usize,

    /// Number of image blocks
    pub image_blocks: usize,

    /// Number of other blocks
    pub other_blocks: usize,

    /// High confidence blocks (>= 0.9)
    pub high_confidence_blocks: usize,

    /// Medium confidence blocks (0.7 - 0.9)
    pub medium_confidence_blocks: usize,

    /// Low confidence blocks (< 0.7)
    pub low_confidence_blocks: usize,

    /// Average confidence
    pub average_confidence: f32,

    /// Processing time
    pub processing_time: Duration,
}

impl Default for ContentStatistics {
    fn default() -> Self {
        Self {
            total_blocks: 0,
            text_blocks: 0,
            table_blocks: 0,
            image_blocks: 0,
            other_blocks: 0,
            high_confidence_blocks: 0,
            medium_confidence_blocks: 0,
            low_confidence_blocks: 0,
            average_confidence: 0.0,
            processing_time: Duration::from_millis(0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_content_block_creation() {
        let block = ContentBlock::text_block("Hello, world!".to_string());
        assert_eq!(block.content_type, "text");
        assert_eq!(block.get_text(), Some("Hello, world!"));
        assert!(block.is_high_confidence(0.9));
    }

    #[test]
    fn test_position_overlap() {
        let pos1 = Position::new(0, 0.0, 0.0, 0.5, 0.5);
        let pos2 = Position::new(0, 0.25, 0.25, 0.5, 0.5);
        let pos3 = Position::new(0, 0.6, 0.6, 0.4, 0.4);

        assert!(pos1.overlaps(&pos2));
        assert!(!pos1.overlaps(&pos3));
        
        let intersection = pos1.intersection_area(&pos2);
        assert!(intersection > 0.0);
    }

    #[test]
    fn test_enhanced_content() {
        let blocks = vec![
            ContentBlock::text_block("Text 1".to_string()),
            ContentBlock::text_block("Text 2".to_string()),
        ];
        
        let content = EnhancedContent::new(blocks);
        assert_eq!(content.blocks.len(), 2);
        assert_eq!(content.get_blocks_by_type("text").len(), 2);
        assert_eq!(content.get_blocks_by_type("table").len(), 0);
    }

    #[test]
    fn test_neural_features() {
        let mut features = NeuralFeatures::new();
        features.text_features = vec![1.0, 2.0, 3.0];
        features.layout_features = vec![4.0, 5.0];
        
        features.combine_features();
        assert_eq!(features.size(), 5);
        assert_eq!(features.combined_features, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_confidence_score() {
        let mut score = ConfidenceScore {
            text_accuracy: 0.9,
            layout_accuracy: 0.8,
            table_accuracy: 0.0, // Not used
            image_accuracy: 0.85,
            quality_confidence: 0.9,
            ..Default::default()
        };

        score.calculate_overall();
        assert!(score.overall > 0.8 && score.overall < 0.9);
        assert!(score.meets_threshold(0.8));
        assert!(!score.meets_threshold(0.95));
    }

    #[test]
    fn test_processing_metrics() {
        let metrics = ProcessingMetrics::default();
        assert_eq!(metrics.total_blocks_processed, 0);
        assert_eq!(metrics.average_confidence, 0.0);
    }

    #[test]
    fn test_content_statistics() {
        let blocks = vec![
            ContentBlock::text_block("Text".to_string()).with_confidence(0.95),
            ContentBlock::table_block("| A | B |\n| 1 | 2 |".to_string()).with_confidence(0.85),
            ContentBlock::text_block("More text".to_string()).with_confidence(0.65),
        ];
        
        let content = EnhancedContent::new(blocks);
        let stats = content.get_statistics();
        
        assert_eq!(stats.total_blocks, 3);
        assert_eq!(stats.text_blocks, 2);
        assert_eq!(stats.table_blocks, 1);
        assert_eq!(stats.high_confidence_blocks, 1);
        assert_eq!(stats.medium_confidence_blocks, 1);
        assert_eq!(stats.low_confidence_blocks, 1);
    }
}