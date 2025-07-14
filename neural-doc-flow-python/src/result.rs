//! Processing result types for Python bindings

use pyo3::prelude::*;
use std::collections::HashMap;
use serde_json;

use neural_doc_flow_core::document::Document;
use crate::{SecurityAnalysis, NeuralDocFlowError};

/// Result of document processing
/// 
/// Contains all extracted content, metadata, and analysis results from document processing.
/// 
/// # Example
/// ```python
/// result = processor.process_document("document.pdf")
/// 
/// # Access extracted text
/// print(result.text)
/// 
/// # Access tables
/// for table in result.tables:
///     print(table)
/// 
/// # Access security analysis
/// if result.security_analysis:
///     print(f"Threat level: {result.security_analysis.threat_level}")
/// 
/// # Access metadata
/// print(result.metadata)
/// ```
#[pyclass]
#[derive(Clone)]
pub struct ProcessingResult {
    /// Extracted text content
    #[pyo3(get)]
    pub text: Option<String>,
    
    /// Extracted tables as list of dictionaries
    #[pyo3(get)]
    pub tables: Vec<HashMap<String, serde_json::Value>>,
    
    /// Extracted images metadata
    #[pyo3(get)]
    pub images: Vec<HashMap<String, serde_json::Value>>,
    
    /// Document metadata
    #[pyo3(get)]
    pub metadata: HashMap<String, serde_json::Value>,
    
    /// Security analysis result (if security is enabled)
    #[pyo3(get)]
    pub security_analysis: Option<SecurityAnalysis>,
    
    /// Processing statistics
    #[pyo3(get)]
    pub stats: HashMap<String, serde_json::Value>,
    
    /// Error information (if processing failed)
    #[pyo3(get)]
    pub error: Option<String>,
    
    /// Source file or identifier
    #[pyo3(get)]
    pub source: String,
    
    /// Output format used for this result
    #[pyo3(get)]
    pub format: String,
}

#[pymethods]
impl ProcessingResult {
    /// Create a new empty processing result
    #[new]
    pub fn new() -> Self {
        Self {
            text: None,
            tables: Vec::new(),
            images: Vec::new(),
            metadata: HashMap::new(),
            security_analysis: None,
            stats: HashMap::new(),
            error: None,
            source: "unknown".to_string(),
            format: "json".to_string(),
        }
    }
    
    /// Check if processing was successful
    /// 
    /// # Returns
    /// True if no errors occurred during processing
    #[getter]
    pub fn success(&self) -> bool {
        self.error.is_none()
    }
    
    /// Get the extracted content as a formatted string
    /// 
    /// # Arguments
    /// * `format` - Output format: "json", "markdown", "html", "xml"
    /// 
    /// # Returns
    /// Formatted string representation of the content
    /// 
    /// # Example
    /// ```python
    /// # Get content as JSON
    /// json_content = result.to_string("json")
    /// 
    /// # Get content as Markdown
    /// markdown_content = result.to_string("markdown")
    /// ```
    #[pyo3(signature = (format = "json"))]
    pub fn to_string(&self, format: &str) -> PyResult<String> {
        match format {
            "json" => self.to_json(),
            "markdown" => Ok(self.to_markdown()),
            "html" => Ok(self.to_html()),
            "xml" => Ok(self.to_xml()),
            _ => Err(NeuralDocFlowError::new_err(
                format!("Unsupported format: {}. Use 'json', 'markdown', 'html', or 'xml'", format)
            )),
        }
    }
    
    /// Convert result to JSON string
    /// 
    /// # Returns
    /// JSON representation of the processing result
    pub fn to_json(&self) -> PyResult<String> {
        let mut result_map = serde_json::Map::new();
        
        // Add basic content
        if let Some(ref text) = self.text {
            result_map.insert("text".to_string(), serde_json::Value::String(text.clone()));
        }
        
        result_map.insert("tables".to_string(), serde_json::to_value(&self.tables)
            .map_err(|e| NeuralDocFlowError::new_err(format!("Failed to serialize tables: {}", e)))?);
        
        result_map.insert("images".to_string(), serde_json::to_value(&self.images)
            .map_err(|e| NeuralDocFlowError::new_err(format!("Failed to serialize images: {}", e)))?);
        
        result_map.insert("metadata".to_string(), serde_json::to_value(&self.metadata)
            .map_err(|e| NeuralDocFlowError::new_err(format!("Failed to serialize metadata: {}", e)))?);
        
        // Add security analysis if available
        if let Some(ref security) = self.security_analysis {
            result_map.insert("security_analysis".to_string(), 
                             security.to_json_value()
                                 .map_err(|e| NeuralDocFlowError::new_err(format!("Failed to serialize security analysis: {}", e)))?);
        }
        
        result_map.insert("stats".to_string(), serde_json::to_value(&self.stats)
            .map_err(|e| NeuralDocFlowError::new_err(format!("Failed to serialize stats: {}", e)))?);
        
        result_map.insert("success".to_string(), serde_json::Value::Bool(self.success()));
        result_map.insert("source".to_string(), serde_json::Value::String(self.source.clone()));
        result_map.insert("format".to_string(), serde_json::Value::String(self.format.clone()));
        
        if let Some(ref error) = self.error {
            result_map.insert("error".to_string(), serde_json::Value::String(error.clone()));
        }
        
        serde_json::to_string_pretty(&result_map)
            .map_err(|e| NeuralDocFlowError::new_err(format!("Failed to serialize to JSON: {}", e)))
    }
    
    /// Convert result to Markdown format
    /// 
    /// # Returns
    /// Markdown representation of the processing result
    pub fn to_markdown(&self) -> String {
        let mut markdown = String::new();
        
        markdown.push_str(&format!("# Document Processing Result\n\n"));
        markdown.push_str(&format!("**Source:** {}\n\n", self.source));
        
        if let Some(ref text) = self.text {
            markdown.push_str("## Extracted Text\n\n");
            markdown.push_str(text);
            markdown.push_str("\n\n");
        }
        
        if !self.tables.is_empty() {
            markdown.push_str("## Tables\n\n");
            for (i, table) in self.tables.iter().enumerate() {
                markdown.push_str(&format!("### Table {}\n\n", i + 1));
                if let Some(serde_json::Value::Array(rows)) = table.get("rows") {
                    for row in rows {
                        if let serde_json::Value::Array(cells) = row {
                            markdown.push('|');
                            for cell in cells {
                                if let serde_json::Value::String(content) = cell {
                                    markdown.push_str(&format!(" {} |", content));
                                }
                            }
                            markdown.push('\n');
                        }
                    }
                }
                markdown.push('\n');
            }
        }
        
        if !self.images.is_empty() {
            markdown.push_str("## Images\n\n");
            for (i, image) in self.images.iter().enumerate() {
                markdown.push_str(&format!("### Image {}\n\n", i + 1));
                if let Some(serde_json::Value::String(desc)) = image.get("description") {
                    markdown.push_str(&format!("Description: {}\n\n", desc));
                }
            }
        }
        
        if let Some(ref security) = self.security_analysis {
            markdown.push_str("## Security Analysis\n\n");
            markdown.push_str(&format!("**Threat Level:** {:?}\n\n", security.threat_level));
            markdown.push_str(&format!("**Malware Probability:** {:.2}%\n\n", security.malware_probability * 100.0));
        }
        
        if !self.success() {
            markdown.push_str("## Error\n\n");
            if let Some(ref error) = self.error {
                markdown.push_str(&format!("```\n{}\n```\n\n", error));
            }
        }
        
        markdown
    }
    
    /// Convert result to HTML format
    /// 
    /// # Returns
    /// HTML representation of the processing result
    pub fn to_html(&self) -> String {
        let mut html = String::new();
        
        html.push_str("<html><head><title>Document Processing Result</title></head><body>\n");
        html.push_str(&format!("<h1>Document Processing Result</h1>\n"));
        html.push_str(&format!("<p><strong>Source:</strong> {}</p>\n", self.source));
        
        if let Some(ref text) = self.text {
            html.push_str("<h2>Extracted Text</h2>\n");
            html.push_str(&format!("<div class=\"content\">{}</div>\n", 
                                 html_escape::encode_text(text)));
        }
        
        if !self.tables.is_empty() {
            html.push_str("<h2>Tables</h2>\n");
            for (i, table) in self.tables.iter().enumerate() {
                html.push_str(&format!("<h3>Table {}</h3>\n", i + 1));
                html.push_str("<table border=\"1\">\n");
                if let Some(serde_json::Value::Array(rows)) = table.get("rows") {
                    for row in rows {
                        if let serde_json::Value::Array(cells) = row {
                            html.push_str("<tr>");
                            for cell in cells {
                                if let serde_json::Value::String(content) = cell {
                                    html.push_str(&format!("<td>{}</td>", html_escape::encode_text(content)));
                                }
                            }
                            html.push_str("</tr>\n");
                        }
                    }
                }
                html.push_str("</table>\n");
            }
        }
        
        if let Some(ref security) = self.security_analysis {
            html.push_str("<h2>Security Analysis</h2>\n");
            html.push_str(&format!("<p><strong>Threat Level:</strong> {:?}</p>\n", security.threat_level));
            html.push_str(&format!("<p><strong>Malware Probability:</strong> {:.2}%</p>\n", 
                                 security.malware_probability * 100.0));
        }
        
        if !self.success() {
            html.push_str("<h2>Error</h2>\n");
            if let Some(ref error) = self.error {
                html.push_str(&format!("<pre class=\"error\">{}</pre>\n", 
                                     html_escape::encode_text(error)));
            }
        }
        
        html.push_str("</body></html>");
        html
    }
    
    /// Convert result to XML format
    /// 
    /// # Returns
    /// XML representation of the processing result
    pub fn to_xml(&self) -> String {
        let mut xml = String::new();
        
        xml.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        xml.push_str("<processing_result>\n");
        xml.push_str(&format!("  <source>{}</source>\n", xml_escape(&self.source)));
        xml.push_str(&format!("  <success>{}</success>\n", self.success()));
        
        if let Some(ref text) = self.text {
            xml.push_str("  <text><![CDATA[");
            xml.push_str(text);
            xml.push_str("]]></text>\n");
        }
        
        if !self.tables.is_empty() {
            xml.push_str("  <tables>\n");
            for (i, _table) in self.tables.iter().enumerate() {
                xml.push_str(&format!("    <table id=\"{}\">\n", i));
                // TODO: Add table content
                xml.push_str("    </table>\n");
            }
            xml.push_str("  </tables>\n");
        }
        
        if let Some(ref security) = self.security_analysis {
            xml.push_str("  <security_analysis>\n");
            xml.push_str(&format!("    <threat_level>{:?}</threat_level>\n", security.threat_level));
            xml.push_str(&format!("    <malware_probability>{}</malware_probability>\n", security.malware_probability));
            xml.push_str("  </security_analysis>\n");
        }
        
        if let Some(ref error) = self.error {
            xml.push_str(&format!("  <error><![CDATA[{}]]></error>\n", error));
        }
        
        xml.push_str("</processing_result>\n");
        xml
    }
    
    /// Get a summary of the processing result
    /// 
    /// # Returns
    /// Dictionary containing summary statistics
    #[getter]
    pub fn summary(&self) -> HashMap<String, serde_json::Value> {
        let mut summary = HashMap::new();
        
        summary.insert("success".to_string(), serde_json::Value::Bool(self.success()));
        summary.insert("has_text".to_string(), serde_json::Value::Bool(self.text.is_some()));
        summary.insert("table_count".to_string(), serde_json::Value::Number(serde_json::Number::from(self.tables.len())));
        summary.insert("image_count".to_string(), serde_json::Value::Number(serde_json::Number::from(self.images.len())));
        summary.insert("has_security_analysis".to_string(), serde_json::Value::Bool(self.security_analysis.is_some()));
        
        if let Some(ref text) = self.text {
            summary.insert("text_length".to_string(), serde_json::Value::Number(serde_json::Number::from(text.len())));
            summary.insert("word_count".to_string(), serde_json::Value::Number(serde_json::Number::from(text.split_whitespace().count())));
        }
        
        summary.insert("source".to_string(), serde_json::Value::String(self.source.clone()));
        summary.insert("format".to_string(), serde_json::Value::String(self.format.clone()));
        
        summary
    }
    
    /// String representation of the result
    fn __repr__(&self) -> String {
        format!(
            "ProcessingResult(source='{}', success={}, text_length={}, tables={}, images={})",
            self.source,
            self.success(),
            self.text.as_ref().map(|t| t.len()).unwrap_or(0),
            self.tables.len(),
            self.images.len()
        )
    }
}

impl ProcessingResult {
    /// Create a ProcessingResult from a Rust Document
    pub fn from_document(document: Document, output_format: &str) -> PyResult<Self> {
        let mut result = ProcessingResult::new();
        
        result.source = document.metadata.source.clone();
        result.format = output_format.to_string();
        
        // Extract text content
        result.text = document.content.text;
        
        // Extract tables
        result.tables = document.content.tables.into_iter().map(|table| {
            let mut table_map = HashMap::new();
            table_map.insert("rows".to_string(), serde_json::to_value(table.rows).unwrap_or(serde_json::Value::Null));
            table_map.insert("headers".to_string(), serde_json::to_value(table.headers).unwrap_or(serde_json::Value::Null));
            table_map.insert("caption".to_string(), serde_json::Value::String(table.caption.unwrap_or_default()));
            table_map
        }).collect();
        
        // Extract images
        result.images = document.content.images.into_iter().map(|image| {
            let mut image_map = HashMap::new();
            image_map.insert("alt_text".to_string(), serde_json::Value::String(image.alt_text.unwrap_or_default()));
            image_map.insert("caption".to_string(), serde_json::Value::String(image.caption.unwrap_or_default()));
            image_map.insert("size".to_string(), serde_json::Value::Number(serde_json::Number::from(image.data.len())));
            image_map
        }).collect();
        
        // Convert metadata
        result.metadata.insert("mime_type".to_string(), serde_json::Value::String(document.metadata.mime_type));
        result.metadata.insert("created_at".to_string(), serde_json::Value::String(document.metadata.created_at.to_rfc3339()));
        result.metadata.insert("size".to_string(), serde_json::Value::Number(serde_json::Number::from(document.metadata.size)));
        
        // Add custom metadata
        for (key, value) in document.metadata.custom {
            result.metadata.insert(key, value);
        }
        
        // Add processing stats
        result.stats.insert("processing_time_ms".to_string(), 
                           serde_json::Value::Number(serde_json::Number::from(document.metadata.processing_time_ms)));
        
        Ok(result)
    }
    
    /// Create an error result
    pub fn new_error(source: String, error: String) -> Self {
        Self {
            text: None,
            tables: Vec::new(),
            images: Vec::new(),
            metadata: HashMap::new(),
            security_analysis: None,
            stats: HashMap::new(),
            error: Some(error),
            source,
            format: "error".to_string(),
        }
    }
}

/// Escape XML special characters
fn xml_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_processing_result_creation() {
        let result = ProcessingResult::new();
        assert!(result.success());
        assert!(result.text.is_none());
        assert_eq!(result.tables.len(), 0);
        assert_eq!(result.images.len(), 0);
    }
    
    #[test]
    fn test_error_result() {
        let result = ProcessingResult::new_error("test.pdf".to_string(), "Test error".to_string());
        assert!(!result.success());
        assert_eq!(result.error, Some("Test error".to_string()));
        assert_eq!(result.source, "test.pdf");
    }
    
    #[test]
    fn test_xml_escape() {
        assert_eq!(xml_escape("test & <test>"), "test &amp; &lt;test&gt;");
        assert_eq!(xml_escape("normal text"), "normal text");
    }
}