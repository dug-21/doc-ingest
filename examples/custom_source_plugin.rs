//! Custom source plugin implementation example
//!
//! This example demonstrates how to create a custom document source plugin
//! for NeuralDocFlow to support new document formats.

use neuraldocflow::{
    DocFlow,
    sources::{DocumentSource, SourceInput, ValidationResult},
    core::{ExtractedDocument, ContentBlock, BlockType, DocumentMetadata},
    error::{NeuralDocFlowError, Result},
    config::SourceConfig,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Custom CSV document source
pub struct CsvSource {
    config: Option<CsvSourceConfig>,
}

/// Configuration for CSV source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsvSourceConfig {
    /// Field delimiter (default: comma)
    pub delimiter: char,
    /// Quote character for fields
    pub quote_char: char,
    /// Whether first row contains headers
    pub has_headers: bool,
    /// Maximum number of rows to process
    pub max_rows: Option<usize>,
    /// Whether to detect data types automatically
    pub auto_detect_types: bool,
}

impl Default for CsvSourceConfig {
    fn default() -> Self {
        Self {
            delimiter: ',',
            quote_char: '"',
            has_headers: true,
            max_rows: None,
            auto_detect_types: true,
        }
    }
}

impl CsvSource {
    /// Create new CSV source with default configuration
    pub fn new() -> Self {
        Self {
            config: None,
        }
    }
    
    /// Create new CSV source with custom configuration
    pub fn with_config(config: CsvSourceConfig) -> Self {
        Self {
            config: Some(config),
        }
    }
    
    /// Get current configuration or default
    fn get_config(&self) -> CsvSourceConfig {
        self.config.clone().unwrap_or_default()
    }
}

#[async_trait]
impl DocumentSource for CsvSource {
    fn source_id(&self) -> &str {
        "csv"
    }
    
    fn name(&self) -> &str {
        "CSV Document Source"
    }
    
    fn version(&self) -> &str {
        "1.0.0"
    }
    
    fn supported_extensions(&self) -> &[&str] {
        &["csv", "CSV", "tsv", "TSV"]
    }
    
    fn supported_mime_types(&self) -> &[&str] {
        &["text/csv", "text/tab-separated-values", "application/csv"]
    }
    
    async fn can_handle(&self, input: &SourceInput) -> Result<bool> {
        match input {
            SourceInput::File { path, .. } => {
                if let Some(ext) = path.extension() {
                    let ext_str = ext.to_str().unwrap_or("");
                    Ok(self.supported_extensions().contains(&ext_str))
                } else {
                    Ok(false)
                }
            }
            SourceInput::Memory { mime_type, .. } => {
                if let Some(mime) = mime_type {
                    Ok(self.supported_mime_types().contains(&mime.as_str()))
                } else {
                    Ok(false)
                }
            }
            SourceInput::Stream { mime_type, .. } => {
                if let Some(mime) = mime_type {
                    Ok(self.supported_mime_types().contains(&mime.as_str()))
                } else {
                    Ok(false)
                }
            }
            SourceInput::Url { .. } => {
                // Could support CSV URLs in the future
                Ok(false)
            }
        }
    }
    
    async fn validate(&self, input: &SourceInput) -> Result<ValidationResult> {
        let mut result = ValidationResult::valid();
        
        match input {
            SourceInput::File { path, .. } => {
                // Check file exists
                if !path.exists() {
                    result.add_error("CSV file does not exist");
                    return Ok(result);
                }
                
                // Check file size
                if let Ok(metadata) = std::fs::metadata(path) {
                    let size_mb = metadata.len() / (1024 * 1024);
                    
                    if size_mb > 500 {
                        result.add_warning("Large CSV file may impact performance");
                    }
                    
                    if size_mb > 2048 {
                        result.add_error("CSV file too large (>2GB)");
                    }
                }
                
                // Validate CSV format by reading first few lines
                match std::fs::read_to_string(path) {
                    Ok(content) => {
                        self.validate_csv_content(&content, &mut result);
                    }
                    Err(e) => {
                        result.add_error(format!("Cannot read CSV file: {}", e));
                    }
                }
            }
            SourceInput::Memory { data, .. } => {
                // Check data size
                let size_mb = data.len() / (1024 * 1024);
                if size_mb > 500 {
                    result.add_warning("Large CSV data may impact performance");
                }
                
                // Validate CSV content
                match String::from_utf8(data.clone()) {
                    Ok(content) => {
                        self.validate_csv_content(&content, &mut result);
                    }
                    Err(_) => {
                        result.add_error("CSV data is not valid UTF-8");
                    }
                }
            }
            SourceInput::Stream { .. } => {
                // Cannot validate streaming input fully
                result.add_warning("Cannot validate streaming CSV input completely");
            }
            SourceInput::Url { .. } => {
                result.add_error("URL input not supported for CSV source");
            }
        }
        
        Ok(result)
    }
    
    async fn extract(&self, input: SourceInput) -> Result<ExtractedDocument> {
        let start_time = std::time::Instant::now();
        let config = self.get_config();
        
        // Read CSV content
        let csv_content = match input {
            SourceInput::File { path, .. } => {
                std::fs::read_to_string(&path)
                    .map_err(|e| NeuralDocFlowError::source(
                        self.source_id(),
                        format!("Failed to read CSV file: {}", e)
                    ))?
            }
            SourceInput::Memory { data, .. } => {
                String::from_utf8(data)
                    .map_err(|e| NeuralDocFlowError::source(
                        self.source_id(),
                        format!("Invalid UTF-8 in CSV data: {}", e)
                    ))?
            }
            SourceInput::Stream { .. } => {
                return Err(NeuralDocFlowError::source(
                    self.source_id(),
                    "Streaming input not yet supported for CSV"
                ));
            }
            SourceInput::Url { .. } => {
                return Err(NeuralDocFlowError::source(
                    self.source_id(),
                    "URL input not supported for CSV"
                ));
            }
        };
        
        // Parse CSV
        let parsed_data = self.parse_csv(&csv_content, &config)?;
        
        // Create extracted document
        let mut document = ExtractedDocument::new(self.source_id().to_string());
        
        // Set metadata
        document.metadata = DocumentMetadata {
            title: Some("CSV Data".to_string()),
            page_count: 1,
            custom_metadata: {
                let mut meta = HashMap::new();
                meta.insert("rows".to_string(), parsed_data.rows.len().to_string());
                meta.insert("columns".to_string(), parsed_data.headers.len().to_string());
                meta.insert("delimiter".to_string(), config.delimiter.to_string());
                meta
            },
            ..Default::default()
        };
        
        // Create content blocks
        document.content = self.create_content_blocks(&parsed_data)?;
        
        // Calculate confidence based on data quality
        document.confidence = self.calculate_confidence(&parsed_data);
        
        // Set metrics
        document.metrics.extraction_time = start_time.elapsed();
        document.metrics.pages_processed = 1;
        document.metrics.blocks_extracted = document.content.len();
        
        Ok(document)
    }
    
    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "delimiter": {
                    "type": "string",
                    "description": "Field delimiter character",
                    "default": ",",
                    "maxLength": 1
                },
                "quote_char": {
                    "type": "string", 
                    "description": "Quote character for fields",
                    "default": "\"",
                    "maxLength": 1
                },
                "has_headers": {
                    "type": "boolean",
                    "description": "Whether first row contains headers",
                    "default": true
                },
                "max_rows": {
                    "type": ["integer", "null"],
                    "description": "Maximum number of rows to process",
                    "minimum": 1
                },
                "auto_detect_types": {
                    "type": "boolean",
                    "description": "Whether to detect data types automatically",
                    "default": true
                }
            }
        })
    }
    
    async fn initialize(&mut self, config: SourceConfig) -> Result<()> {
        // Parse CSV-specific configuration
        if let Ok(csv_config) = serde_json::from_value::<CsvSourceConfig>(config.specific_config) {
            self.config = Some(csv_config);
        }
        Ok(())
    }
    
    async fn cleanup(&mut self) -> Result<()> {
        // No cleanup needed for CSV source
        Ok(())
    }
}

/// Parsed CSV data structure
#[derive(Debug)]
struct ParsedCsvData {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    column_types: Vec<CsvColumnType>,
}

/// Detected column data types
#[derive(Debug, Clone)]
enum CsvColumnType {
    Text,
    Integer,
    Float,
    Date,
    Boolean,
    Mixed,
}

impl CsvSource {
    /// Validate CSV content format
    fn validate_csv_content(&self, content: &str, result: &mut ValidationResult) {
        let config = self.get_config();
        let lines: Vec<&str> = content.lines().take(10).collect(); // Check first 10 lines
        
        if lines.is_empty() {
            result.add_error("CSV file is empty");
            return;
        }
        
        // Check for consistent column count
        let mut column_counts = Vec::new();
        for line in &lines {
            let count = line.matches(config.delimiter).count() + 1;
            column_counts.push(count);
        }
        
        let first_count = column_counts[0];
        if !column_counts.iter().all(|&count| count == first_count) {
            result.add_warning("Inconsistent number of columns across rows");
        }
        
        if first_count > 1000 {
            result.add_warning("Very wide CSV (>1000 columns) may impact performance");
        }
    }
    
    /// Parse CSV content into structured data
    fn parse_csv(&self, content: &str, config: &CsvSourceConfig) -> Result<ParsedCsvData> {
        let lines: Vec<&str> = content.lines().collect();
        
        if lines.is_empty() {
            return Err(NeuralDocFlowError::source(
                self.source_id(),
                "CSV content is empty"
            ));
        }
        
        let mut data_lines = lines.as_slice();
        
        // Extract headers
        let headers = if config.has_headers {
            let header_line = data_lines[0];
            data_lines = &data_lines[1..];
            self.parse_csv_line(header_line, config)?
        } else {
            // Generate default headers
            let first_line = data_lines[0];
            let column_count = self.parse_csv_line(first_line, config)?.len();
            (0..column_count).map(|i| format!("Column_{}", i + 1)).collect()
        };
        
        // Apply row limit
        if let Some(max_rows) = config.max_rows {
            data_lines = &data_lines[..data_lines.len().min(max_rows)];
        }
        
        // Parse data rows
        let mut rows = Vec::new();
        for line in data_lines {
            if !line.trim().is_empty() {
                let row = self.parse_csv_line(line, config)?;
                
                // Ensure row has same number of columns as headers
                let mut normalized_row = row;
                normalized_row.resize(headers.len(), String::new());
                
                rows.push(normalized_row);
            }
        }
        
        // Detect column types
        let column_types = if config.auto_detect_types {
            self.detect_column_types(&rows, &headers)
        } else {
            vec![CsvColumnType::Text; headers.len()]
        };
        
        Ok(ParsedCsvData {
            headers,
            rows,
            column_types,
        })
    }
    
    /// Parse a single CSV line
    fn parse_csv_line(&self, line: &str, config: &CsvSourceConfig) -> Result<Vec<String>> {
        let mut fields = Vec::new();
        let mut current_field = String::new();
        let mut in_quotes = false;
        let mut chars = line.chars().peekable();
        
        while let Some(ch) = chars.next() {
            match ch {
                c if c == config.quote_char => {
                    if in_quotes && chars.peek() == Some(&config.quote_char) {
                        // Escaped quote
                        current_field.push(config.quote_char);
                        chars.next(); // consume the second quote
                    } else {
                        in_quotes = !in_quotes;
                    }
                }
                c if c == config.delimiter && !in_quotes => {
                    fields.push(current_field.trim().to_string());
                    current_field.clear();
                }
                c => {
                    current_field.push(c);
                }
            }
        }
        
        // Add the last field
        fields.push(current_field.trim().to_string());
        
        Ok(fields)
    }
    
    /// Detect column data types
    fn detect_column_types(&self, rows: &[Vec<String>], headers: &[String]) -> Vec<CsvColumnType> {
        let mut column_types = Vec::new();
        
        for col_idx in 0..headers.len() {
            let column_values: Vec<&String> = rows.iter()
                .filter_map(|row| row.get(col_idx))
                .filter(|val| !val.is_empty())
                .collect();
            
            if column_values.is_empty() {
                column_types.push(CsvColumnType::Text);
                continue;
            }
            
            // Try to detect type based on sample of values
            let sample_size = column_values.len().min(100);
            let sample = &column_values[..sample_size];
            
            let detected_type = self.analyze_column_type(sample);
            column_types.push(detected_type);
        }
        
        column_types
    }
    
    /// Analyze column type from sample values
    fn analyze_column_type(&self, values: &[&String]) -> CsvColumnType {
        let mut int_count = 0;
        let mut float_count = 0;
        let mut bool_count = 0;
        let mut date_count = 0;
        
        for value in values {
            let trimmed = value.trim().to_lowercase();
            
            if trimmed.parse::<i64>().is_ok() {
                int_count += 1;
            } else if trimmed.parse::<f64>().is_ok() {
                float_count += 1;
            } else if matches!(trimmed.as_str(), "true" | "false" | "yes" | "no" | "1" | "0") {
                bool_count += 1;
            } else if self.looks_like_date(&trimmed) {
                date_count += 1;
            }
        }
        
        let total = values.len();
        let threshold = (total as f64 * 0.8) as usize; // 80% threshold
        
        if int_count >= threshold {
            CsvColumnType::Integer
        } else if (int_count + float_count) >= threshold {
            CsvColumnType::Float
        } else if bool_count >= threshold {
            CsvColumnType::Boolean
        } else if date_count >= threshold {
            CsvColumnType::Date
        } else if int_count > 0 || float_count > 0 || bool_count > 0 || date_count > 0 {
            CsvColumnType::Mixed
        } else {
            CsvColumnType::Text
        }
    }
    
    /// Check if string looks like a date
    fn looks_like_date(&self, value: &str) -> bool {
        // Simple heuristic for common date formats
        let date_patterns = [
            r"\d{4}-\d{2}-\d{2}",        // YYYY-MM-DD
            r"\d{2}/\d{2}/\d{4}",        // MM/DD/YYYY
            r"\d{2}-\d{2}-\d{4}",        // MM-DD-YYYY
            r"\d{4}/\d{2}/\d{2}",        // YYYY/MM/DD
        ];
        
        date_patterns.iter().any(|pattern| {
            regex::Regex::new(pattern)
                .map(|re| re.is_match(value))
                .unwrap_or(false)
        })
    }
    
    /// Create content blocks from parsed CSV data
    fn create_content_blocks(&self, data: &ParsedCsvData) -> Result<Vec<ContentBlock>> {
        let mut blocks = Vec::new();
        
        // Create table block for the entire CSV
        let mut table_text = String::new();
        
        // Add headers
        table_text.push_str(&data.headers.join("\t"));
        table_text.push('\n');
        
        // Add data rows
        for row in &data.rows {
            table_text.push_str(&row.join("\t"));
            table_text.push('\n');
        }
        
        let mut table_block = ContentBlock::new(BlockType::Table)
            .with_text(table_text);
        
        // Add table metadata
        table_block.metadata.attributes.insert(
            "table_structure".to_string(),
            serde_json::json!({
                "rows": data.rows.len(),
                "columns": data.headers.len(),
                "headers": data.headers,
                "column_types": data.column_types.iter().map(|t| format!("{:?}", t)).collect::<Vec<_>>()
            }).to_string(),
        );
        
        table_block.metadata.confidence = 0.95; // High confidence for structured data
        
        blocks.push(table_block);
        
        // Create individual text blocks for each significant text column
        for (col_idx, header) in data.headers.iter().enumerate() {
            if matches!(data.column_types.get(col_idx), Some(CsvColumnType::Text) | Some(CsvColumnType::Mixed)) {
                let column_text: Vec<String> = data.rows.iter()
                    .filter_map(|row| row.get(col_idx))
                    .filter(|val| !val.is_empty() && val.len() > 10) // Only significant text
                    .cloned()
                    .collect();
                
                if !column_text.is_empty() {
                    let mut text_block = ContentBlock::new(BlockType::Paragraph)
                        .with_text(column_text.join("\n\n"));
                    
                    text_block.metadata.attributes.insert(
                        "column_name".to_string(),
                        header.clone(),
                    );
                    text_block.metadata.confidence = 0.85;
                    
                    blocks.push(text_block);
                }
            }
        }
        
        Ok(blocks)
    }
    
    /// Calculate extraction confidence based on data quality
    fn calculate_confidence(&self, data: &ParsedCsvData) -> f32 {
        let mut confidence = 0.9; // Base confidence for successful parsing
        
        // Reduce confidence for empty or malformed data
        if data.rows.is_empty() {
            return 0.1;
        }
        
        // Check data consistency
        let expected_columns = data.headers.len();
        let consistent_rows = data.rows.iter()
            .filter(|row| row.len() == expected_columns)
            .count();
        
        let consistency_ratio = consistent_rows as f32 / data.rows.len() as f32;
        confidence *= consistency_ratio;
        
        // Bonus for typed columns
        let typed_columns = data.column_types.iter()
            .filter(|t| !matches!(t, CsvColumnType::Text | CsvColumnType::Mixed))
            .count();
        
        if typed_columns > 0 {
            confidence += 0.05 * (typed_columns as f32 / data.headers.len() as f32);
        }
        
        confidence.clamp(0.0, 1.0)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Custom CSV Source Plugin Example ===\n");
    
    // Create DocFlow instance
    let mut docflow = DocFlow::new()?;
    
    // Register our custom CSV source
    docflow.register_source("csv", Box::new(CsvSource::new()))?;
    
    println!("✅ Registered custom CSV source plugin\n");
    
    // Create sample CSV data
    create_sample_csv_file().await?;
    
    // Extract from CSV file
    let input = SourceInput::File {
        path: PathBuf::from("sample_data.csv"),
        metadata: None,
    };
    
    println!("📊 Extracting from CSV file...");
    let document = docflow.extract(input).await?;
    
    // Display results
    println!("\n=== Extraction Results ===");
    println!("Document ID: {}", document.id);
    println!("Confidence: {:.1}%", document.confidence * 100.0);
    println!("Content blocks: {}", document.content.len());
    
    // Show metadata
    if let Some(rows) = document.metadata.custom_metadata.get("rows") {
        println!("Rows: {}", rows);
    }
    if let Some(columns) = document.metadata.custom_metadata.get("columns") {
        println!("Columns: {}", columns);
    }
    
    // Show content blocks
    for (i, block) in document.content.iter().enumerate() {
        println!("\nBlock {} ({:?}):", i + 1, block.block_type);
        
        if let Some(text) = &block.text {
            let preview = if text.len() > 200 {
                format!("{}...", &text[..197])
            } else {
                text.clone()
            };
            println!("Content: {}", preview);
        }
        
        if let Some(table_structure) = block.metadata.attributes.get("table_structure") {
            println!("Table structure: {}", table_structure);
        }
        
        if let Some(column_name) = block.metadata.attributes.get("column_name") {
            println!("Column: {}", column_name);
        }
    }
    
    println!("\n✅ CSV extraction completed successfully!");
    
    // Cleanup
    let _ = std::fs::remove_file("sample_data.csv");
    
    Ok(())
}

/// Create a sample CSV file for demonstration
async fn create_sample_csv_file() -> Result<()> {
    let csv_content = r#"Name,Age,Department,Salary,Join Date,Active
John Doe,30,Engineering,75000,2020-01-15,true
Jane Smith,28,Marketing,65000,2021-03-22,true
Bob Johnson,35,Engineering,85000,2019-07-10,true
Alice Brown,32,Sales,70000,2020-11-05,false
Charlie Wilson,29,Marketing,62000,2021-08-18,true
Diana Davis,31,Engineering,78000,2020-05-30,true
Eve Miller,27,Sales,68000,2022-01-12,true
Frank Taylor,33,Engineering,82000,2019-12-08,true
Grace Lee,26,Marketing,60000,2022-04-25,true
Henry Clark,34,Sales,72000,2020-09-14,true"#;
    
    std::fs::write("sample_data.csv", csv_content)
        .map_err(|e| NeuralDocFlowError::io(e))?;
    
    println!("📝 Created sample CSV file: sample_data.csv");
    
    Ok(())
}