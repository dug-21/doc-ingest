//! CSV Source Plugin for Neural Document Flow
//!
//! This plugin handles CSV document extraction and processing.

use crate::{
    traits::{DocumentSource, source::{DocumentMetadata, ValidationResult, ValidationIssue, ValidationSeverity}},
    document::{Document, DocumentBuilder},
    error::{ProcessingError, SourceError},
    SourceResult,
};
use async_trait::async_trait;
use std::collections::HashMap;

/// CSV document source implementation
#[derive(Debug, Clone)]
pub struct CSVSource {
    name: String,
    version: String,
}

impl CSVSource {
    /// Create a new CSV source instance
    pub fn new() -> Self {
        Self {
            name: "CSV Source Plugin".to_string(),
            version: "1.0.0".to_string(),
        }
    }
    
    /// Extract text from CSV by parsing and formatting
    async fn extract_csv_text(&self, csv_data: &[u8]) -> Result<String, ProcessingError> {
        let csv_string = String::from_utf8_lossy(csv_data);
        
        // Parse CSV and convert to readable text format
        let parsed_csv = self.parse_csv(&csv_string)?;
        let formatted_text = self.format_csv_as_text(&parsed_csv);
        
        Ok(formatted_text)
    }
    
    /// Basic CSV parsing (in practice, use a proper CSV library like csv-rs)
    fn parse_csv(&self, csv_string: &str) -> Result<Vec<Vec<String>>, ProcessingError> {
        let mut rows = Vec::new();
        
        for line in csv_string.lines() {
            if line.trim().is_empty() {
                continue;
            }
            
            // Basic CSV parsing - handles simple cases
            let fields: Vec<String> = line.split(',')
                .map(|field| field.trim().trim_matches('"').to_string())
                .collect();
            
            rows.push(fields);
        }
        
        Ok(rows)
    }
    
    /// Format CSV data as readable text
    fn format_csv_as_text(&self, csv_data: &[Vec<String>]) -> String {
        if csv_data.is_empty() {
            return "Empty CSV file".to_string();
        }
        
        let mut result = String::new();
        
        // Add header information
        result.push_str("CSV Document Content\n");
        result.push_str(&format!("Rows: {}\n", csv_data.len()));
        result.push_str(&format!("Columns: {}\n\n", csv_data[0].len()));
        
        // Format as table
        for (row_idx, row) in csv_data.iter().enumerate() {
            if row_idx == 0 {
                result.push_str("Header: ");
            } else {
                result.push_str(&format!("Row {}: ", row_idx));
            }
            
            let row_text = row.join(" | ");
            result.push_str(&row_text);
            result.push('\n');
            
            // Add separator after header
            if row_idx == 0 && csv_data.len() > 1 {
                result.push_str(&"-".repeat(row_text.len()));
                result.push('\n');
            }
        }
        
        result
    }
    
    /// Extract metadata from CSV
    async fn extract_csv_metadata(&self, csv_data: &[u8]) -> Result<HashMap<String, serde_json::Value>, ProcessingError> {
        let csv_string = String::from_utf8_lossy(csv_data);
        let mut metadata = HashMap::new();
        
        // Parse CSV to get structure info
        let parsed_csv = self.parse_csv(&csv_string)?;
        
        // Basic CSV info
        metadata.insert("content_type".to_string(), serde_json::Value::String("CSV".to_string()));
        metadata.insert("file_size".to_string(), serde_json::Value::Number(serde_json::Number::from(csv_data.len())));
        metadata.insert("row_count".to_string(), serde_json::Value::Number(serde_json::Number::from(parsed_csv.len())));
        
        if !parsed_csv.is_empty() {
            metadata.insert("column_count".to_string(), serde_json::Value::Number(serde_json::Number::from(parsed_csv[0].len())));
            
            // Extract header if present
            if !parsed_csv[0].is_empty() {
                metadata.insert("headers".to_string(), serde_json::Value::Array(
                    parsed_csv[0].iter().map(|h| serde_json::Value::String(h.clone())).collect()
                ));
            }
        }
        
        // Detect delimiter (basic detection)
        let delimiter = self.detect_delimiter(&csv_string);
        metadata.insert("delimiter".to_string(), serde_json::Value::String(delimiter));
        
        // Check for quotes
        let has_quotes = csv_string.contains('"');
        metadata.insert("has_quotes".to_string(), serde_json::Value::Bool(has_quotes));
        
        // Data quality indicators
        let empty_cells = self.count_empty_cells(&parsed_csv);
        metadata.insert("empty_cells".to_string(), serde_json::Value::Number(serde_json::Number::from(empty_cells)));
        
        Ok(metadata)
    }
    
    /// Detect CSV delimiter
    fn detect_delimiter(&self, csv_string: &str) -> String {
        let delimiters = [',', ';', '\t', '|'];
        let mut delimiter_counts = HashMap::new();
        
        // Count occurrences of each potential delimiter
        for delimiter in &delimiters {
            let count = csv_string.chars().filter(|&c| c == *delimiter).count();
            delimiter_counts.insert(*delimiter, count);
        }
        
        // Find the most common delimiter
        let best_delimiter = delimiter_counts.iter()
            .max_by_key(|(_, &count)| count)
            .map(|(&delim, _)| delim)
            .unwrap_or(',');
        
        match best_delimiter {
            ',' => "comma".to_string(),
            ';' => "semicolon".to_string(),
            '\t' => "tab".to_string(),
            '|' => "pipe".to_string(),
            _ => "comma".to_string(),
        }
    }
    
    /// Count empty cells in CSV
    fn count_empty_cells(&self, csv_data: &[Vec<String>]) -> usize {
        csv_data.iter()
            .flat_map(|row| row.iter())
            .filter(|cell| cell.trim().is_empty())
            .count()
    }
    
    /// Validate CSV file structure
    async fn validate_csv(&self, csv_data: &[u8]) -> Result<ValidationResult, SourceError> {
        let csv_string = String::from_utf8_lossy(csv_data);
        let mut validation_result = ValidationResult::success();
        
        // Check if file is empty
        if csv_string.trim().is_empty() {
            validation_result.add_issue(ValidationIssue {
                severity: ValidationSeverity::Error,
                message: "CSV file is empty".to_string(),
                suggestion: Some("Ensure the CSV file contains data".to_string()),
            });
            return Ok(validation_result);
        }
        
        // Parse CSV to validate structure
        let parsed_csv = match self.parse_csv(&csv_string) {
            Ok(csv) => csv,
            Err(_) => {
                validation_result.add_issue(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    message: "Failed to parse CSV file".to_string(),
                    suggestion: Some("Check CSV format and encoding".to_string()),
                });
                return Ok(validation_result);
            }
        };
        
        // Check for consistent row lengths
        if !parsed_csv.is_empty() {
            let expected_length = parsed_csv[0].len();
            let inconsistent_rows: Vec<usize> = parsed_csv.iter()
                .enumerate()
                .filter(|(_, row)| row.len() != expected_length)
                .map(|(idx, _)| idx + 1)
                .collect();
            
            if !inconsistent_rows.is_empty() {
                validation_result.add_issue(ValidationIssue {
                    severity: ValidationSeverity::Warning,
                    message: format!("Inconsistent row lengths found in rows: {:?}", inconsistent_rows),
                    suggestion: Some("Check for missing or extra columns".to_string()),
                });
            }
        }
        
        // Check for very large files
        if csv_data.len() > 50 * 1024 * 1024 { // 50MB
            validation_result.add_issue(ValidationIssue {
                severity: ValidationSeverity::Warning,
                message: "CSV file is very large".to_string(),
                suggestion: Some("Consider processing in chunks for better performance".to_string()),
            });
        }
        
        // Check for too many columns
        if !parsed_csv.is_empty() && parsed_csv[0].len() > 1000 {
            validation_result.add_issue(ValidationIssue {
                severity: ValidationSeverity::Warning,
                message: "CSV has a very large number of columns".to_string(),
                suggestion: Some("Verify the delimiter is correct".to_string()),
            });
        }
        
        // Estimate processing time
        let row_count = parsed_csv.len();
        let estimated_time = (row_count as f64 / 1000.0) * 0.1; // 0.1 seconds per 1000 rows
        validation_result.estimated_processing_time = Some(estimated_time);
        
        Ok(validation_result)
    }
}

impl Default for CSVSource {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentSource for CSVSource {
    fn source_type(&self) -> &'static str {
        "csv"
    }
    
    async fn can_handle(&self, input: &str) -> bool {
        // Check file extension
        let lower_input = input.to_lowercase();
        if lower_input.ends_with(".csv") || lower_input.ends_with(".tsv") {
            return true;
        }
        
        // Check if it's a file path that exists and looks like CSV
        if let Ok(metadata) = std::fs::metadata(input) {
            if metadata.is_file() {
                // Try to read first line to check for CSV structure
                if let Ok(content) = std::fs::read_to_string(input) {
                    let first_line = content.lines().next().unwrap_or("");
                    // Basic heuristic: contains commas or semicolons
                    return first_line.contains(',') || first_line.contains(';');
                }
            }
        }
        
        false
    }
    
    async fn load_document(&self, input: &str) -> SourceResult<Document> {
        // Read CSV file
        let csv_data = std::fs::read(input)
            .map_err(|e| ProcessingError::IoError(e.to_string()))?;
        
        // Validate CSV
        let validation_result = self.validate_csv(&csv_data).await?;
        if validation_result.has_critical_issues() {
            return Err(SourceError::ParseError {
                reason: format!("CSV validation failed: {:?}", validation_result.issues)
            });
        }
        
        // Extract text content
        let text_content = self.extract_csv_text(&csv_data).await?;
        
        // Extract metadata
        let custom_metadata = self.extract_csv_metadata(&csv_data).await?;
        
        // Build document
        let document = DocumentBuilder::new()
            .source("csv_source")
            .mime_type("text/csv")
            .size(csv_data.len() as u64)
            .text_content(text_content)
            .build();
        
        Ok(document)
    }
    
    async fn get_metadata(&self, input: &str) -> SourceResult<DocumentMetadata> {
        let metadata = std::fs::metadata(input)
            .map_err(|e| ProcessingError::IoError(e.to_string()))?;
        
        let mut attributes = HashMap::new();
        attributes.insert("file_type".to_string(), serde_json::Value::String("CSV".to_string()));
        attributes.insert("readonly".to_string(), serde_json::Value::Bool(metadata.permissions().readonly()));
        
        // Quick peek at the file to get row count
        if let Ok(content) = std::fs::read_to_string(input) {
            let line_count = content.lines().count();
            attributes.insert("estimated_rows".to_string(), serde_json::Value::Number(serde_json::Number::from(line_count)));
        }
        
        Ok(DocumentMetadata {
            name: std::path::Path::new(input)
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
            size: Some(metadata.len()),
            mime_type: "text/csv".to_string(),
            modified: metadata.modified().ok().and_then(|t| {
                use std::time::SystemTime;
                let duration = t.duration_since(SystemTime::UNIX_EPOCH).ok()?;
                Some(chrono::DateTime::from_timestamp(duration.as_secs() as i64, 0)?)
            }),
            attributes,
        })
    }
    
    async fn validate(&self, input: &str) -> SourceResult<ValidationResult> {
        // Check if file exists
        if !std::path::Path::new(input).exists() {
            return Ok(ValidationResult::failure(vec![
                ValidationIssue {
                    severity: ValidationSeverity::Critical,
                    message: format!("File does not exist: {}", input),
                    suggestion: Some("Verify the file path is correct".to_string()),
                }
            ]));
        }
        
        // Read and validate CSV
        let csv_data = std::fs::read(input)
            .map_err(|e| ProcessingError::IoError(e.to_string()))?;
        
        self.validate_csv(&csv_data).await
    }
    
    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["csv", "tsv"]
    }
    
    fn supported_mime_types(&self) -> Vec<&'static str> {
        vec!["text/csv", "text/tab-separated-values"]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_csv_source_creation() {
        let source = CSVSource::new();
        assert_eq!(source.source_type(), "csv");
        assert_eq!(source.supported_extensions(), vec!["csv", "tsv"]);
    }
    
    #[tokio::test]
    async fn test_can_handle_csv() {
        let source = CSVSource::new();
        assert!(source.can_handle("data.csv").await);
        assert!(source.can_handle("data.tsv").await);
        assert!(!source.can_handle("document.txt").await);
    }
    
    #[tokio::test]
    async fn test_parse_csv() {
        let source = CSVSource::new();
        let csv_data = "Name,Age,City\nJohn,30,New York\nJane,25,Boston";
        let result = source.parse_csv(csv_data).unwrap();
        
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], vec!["Name", "Age", "City"]);
        assert_eq!(result[1], vec!["John", "30", "New York"]);
        assert_eq!(result[2], vec!["Jane", "25", "Boston"]);
    }
    
    #[tokio::test]
    async fn test_detect_delimiter() {
        let source = CSVSource::new();
        assert_eq!(source.detect_delimiter("a,b,c"), "comma");
        assert_eq!(source.detect_delimiter("a;b;c"), "semicolon");
        assert_eq!(source.detect_delimiter("a\tb\tc"), "tab");
    }
    
    #[tokio::test]
    async fn test_count_empty_cells() {
        let source = CSVSource::new();
        let data = vec![
            vec!["A".to_string(), "B".to_string()],
            vec!["1".to_string(), "".to_string()],
            vec!["".to_string(), "2".to_string()],
        ];
        assert_eq!(source.count_empty_cells(&data), 2);
    }
}