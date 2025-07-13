//! Document analysis modules

use crate::neural_engine::NeuralError;
use crate::networks::LayoutAnalysisNetwork;

/// Document structure analyzer
pub struct DocumentAnalyzer {
    layout_network: LayoutAnalysisNetwork,
}

impl DocumentAnalyzer {
    pub fn new() -> Result<Self, NeuralError> {
        Ok(Self {
            layout_network: LayoutAnalysisNetwork::new()?,
        })
    }

    pub fn analyze_structure(&self, data: &[u8]) -> Result<DocumentStructure, NeuralError> {
        let elements = self.layout_network.analyze_layout(data)?;
        Ok(DocumentStructure::new(elements))
    }
}

impl Default for DocumentAnalyzer {
    fn default() -> Self {
        Self::new().expect("Failed to create document analyzer")
    }
}

/// Document structure representation
#[derive(Debug, Clone)]
pub struct DocumentStructure {
    pub elements: Vec<String>,
    pub confidence: f32,
}

impl DocumentStructure {
    pub fn new(elements: Vec<String>) -> Self {
        Self {
            elements,
            confidence: 0.95,
        }
    }

    pub fn element_count(&self) -> usize {
        self.elements.len()
    }
}

/// Content type detection
#[derive(Debug, Clone, PartialEq)]
pub enum ContentType {
    Text,
    Table,
    Image,
    Chart,
    Unknown,
}

impl ContentType {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "text" => ContentType::Text,
            "table" => ContentType::Table,
            "image" => ContentType::Image,
            "chart" => ContentType::Chart,
            _ => ContentType::Unknown,
        }
    }
}