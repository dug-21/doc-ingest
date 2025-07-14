//! Common types and utilities for Python bindings

use pyo3::prelude::*;
use std::collections::HashMap;
use serde_json;

/// Document metadata container
/// 
/// Contains metadata about processed documents including file information,
/// processing statistics, and custom attributes.
/// 
/// # Example
/// ```python
/// result = processor.process_document("document.pdf")
/// metadata = result.metadata
/// print(f"File size: {metadata.get('size', 0)} bytes")
/// print(f"MIME type: {metadata.get('mime_type', 'unknown')}")
/// ```
#[pyclass]
#[derive(Clone)]
pub struct DocumentMetadata {
    /// Internal metadata storage
    data: HashMap<String, serde_json::Value>,
}

#[pymethods]
impl DocumentMetadata {
    /// Create new empty metadata
    #[new]
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }
    
    /// Get a metadata value
    /// 
    /// # Arguments
    /// * `key` - Metadata key
    /// * `default` - Default value if key not found
    /// 
    /// # Returns
    /// Metadata value or default
    pub fn get(&self, key: &str, default: Option<serde_json::Value>) -> serde_json::Value {
        self.data.get(key).cloned().unwrap_or_else(|| {
            default.unwrap_or(serde_json::Value::Null)
        })
    }
    
    /// Set a metadata value
    /// 
    /// # Arguments
    /// * `key` - Metadata key
    /// * `value` - Value to set
    pub fn set(&mut self, key: String, value: serde_json::Value) {
        self.data.insert(key, value);
    }
    
    /// Check if metadata contains a key
    /// 
    /// # Arguments
    /// * `key` - Key to check
    /// 
    /// # Returns
    /// True if key exists
    pub fn contains(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }
    
    /// Get all metadata keys
    /// 
    /// # Returns
    /// List of all metadata keys
    pub fn keys(&self) -> Vec<String> {
        self.data.keys().cloned().collect()
    }
    
    /// Get all metadata as dictionary
    /// 
    /// # Returns
    /// Dictionary containing all metadata
    pub fn to_dict(&self) -> HashMap<String, serde_json::Value> {
        self.data.clone()
    }
    
    /// Remove a metadata key
    /// 
    /// # Arguments
    /// * `key` - Key to remove
    /// 
    /// # Returns
    /// The removed value, or None if key didn't exist
    pub fn remove(&mut self, key: &str) -> Option<serde_json::Value> {
        self.data.remove(key)
    }
    
    /// Clear all metadata
    pub fn clear(&mut self) {
        self.data.clear();
    }
    
    /// Get metadata size (number of entries)
    /// 
    /// # Returns
    /// Number of metadata entries
    #[getter]
    pub fn size(&self) -> usize {
        self.data.len()
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!("DocumentMetadata(size={})", self.data.len())
    }
    
    /// Dictionary-style access
    fn __getitem__(&self, key: &str) -> PyResult<serde_json::Value> {
        self.data.get(key).cloned().ok_or_else(|| {
            pyo3::exceptions::PyKeyError::new_err(format!("Key '{}' not found", key))
        })
    }
    
    /// Dictionary-style assignment
    fn __setitem__(&mut self, key: String, value: serde_json::Value) {
        self.data.insert(key, value);
    }
    
    /// Dictionary-style contains check
    fn __contains__(&self, key: &str) -> bool {
        self.data.contains_key(key)
    }
    
    /// Get length
    fn __len__(&self) -> usize {
        self.data.len()
    }
}

/// Table representation for extracted tables
/// 
/// Represents a table extracted from a document with rows, columns,
/// headers, and optional caption.
/// 
/// # Example
/// ```python
/// for table in result.tables:
///     if table.headers:
///         print(f"Headers: {table.headers}")
///     for row in table.rows:
///         print(f"Row: {row}")
/// ```
#[pyclass]
#[derive(Clone)]
pub struct Table {
    /// Table headers (if any)
    #[pyo3(get, set)]
    pub headers: Option<Vec<String>>,
    
    /// Table rows
    #[pyo3(get, set)]
    pub rows: Vec<Vec<String>>,
    
    /// Table caption (if any)
    #[pyo3(get, set)]
    pub caption: Option<String>,
    
    /// Table position in document
    #[pyo3(get, set)]
    pub position: Option<usize>,
}

#[pymethods]
impl Table {
    /// Create a new table
    /// 
    /// # Arguments
    /// * `rows` - Table rows as list of lists
    /// * `headers` - Optional headers
    /// * `caption` - Optional caption
    #[new]
    #[pyo3(signature = (rows, headers = None, caption = None))]
    pub fn new(
        rows: Vec<Vec<String>>,
        headers: Option<Vec<String>>,
        caption: Option<String>,
    ) -> Self {
        Self {
            headers,
            rows,
            caption,
            position: None,
        }
    }
    
    /// Get number of rows
    /// 
    /// # Returns
    /// Number of rows in the table
    #[getter]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }
    
    /// Get number of columns
    /// 
    /// # Returns
    /// Number of columns (based on first row)
    #[getter]
    pub fn column_count(&self) -> usize {
        self.rows.first().map(|row| row.len()).unwrap_or(0)
    }
    
    /// Get a specific cell value
    /// 
    /// # Arguments
    /// * `row` - Row index
    /// * `col` - Column index
    /// 
    /// # Returns
    /// Cell value or None if out of bounds
    pub fn get_cell(&self, row: usize, col: usize) -> Option<String> {
        self.rows.get(row)?.get(col).cloned()
    }
    
    /// Set a specific cell value
    /// 
    /// # Arguments
    /// * `row` - Row index
    /// * `col` - Column index
    /// * `value` - New cell value
    /// 
    /// # Returns
    /// True if cell was set successfully
    pub fn set_cell(&mut self, row: usize, col: usize, value: String) -> bool {
        if let Some(table_row) = self.rows.get_mut(row) {
            if let Some(cell) = table_row.get_mut(col) {
                *cell = value;
                return true;
            }
        }
        false
    }
    
    /// Get a specific row
    /// 
    /// # Arguments
    /// * `index` - Row index
    /// 
    /// # Returns
    /// Row as list of strings, or None if out of bounds
    pub fn get_row(&self, index: usize) -> Option<Vec<String>> {
        self.rows.get(index).cloned()
    }
    
    /// Get a specific column
    /// 
    /// # Arguments
    /// * `index` - Column index
    /// 
    /// # Returns
    /// Column as list of strings, or None if out of bounds
    pub fn get_column(&self, index: usize) -> Option<Vec<String>> {
        let mut column = Vec::new();
        for row in &self.rows {
            if let Some(cell) = row.get(index) {
                column.push(cell.clone());
            } else {
                return None;
            }
        }
        Some(column)
    }
    
    /// Convert table to CSV format
    /// 
    /// # Arguments
    /// * `delimiter` - Column delimiter (default: comma)
    /// * `include_headers` - Whether to include headers
    /// 
    /// # Returns
    /// CSV representation of the table
    #[pyo3(signature = (delimiter = ",", include_headers = true))]
    pub fn to_csv(&self, delimiter: &str, include_headers: bool) -> String {
        let mut csv = String::new();
        
        // Add headers if requested and available
        if include_headers {
            if let Some(ref headers) = self.headers {
                csv.push_str(&headers.join(delimiter));
                csv.push('\n');
            }
        }
        
        // Add rows
        for row in &self.rows {
            csv.push_str(&row.join(delimiter));
            csv.push('\n');
        }
        
        csv
    }
    
    /// Convert table to HTML format
    /// 
    /// # Returns
    /// HTML table representation
    pub fn to_html(&self) -> String {
        let mut html = String::new();
        html.push_str("<table>\n");
        
        // Add headers if available
        if let Some(ref headers) = self.headers {
            html.push_str("  <thead>\n    <tr>\n");
            for header in headers {
                html.push_str(&format!("      <th>{}</th>\n", html_escape::encode_text(header)));
            }
            html.push_str("    </tr>\n  </thead>\n");
        }
        
        // Add body
        html.push_str("  <tbody>\n");
        for row in &self.rows {
            html.push_str("    <tr>\n");
            for cell in row {
                html.push_str(&format!("      <td>{}</td>\n", html_escape::encode_text(cell)));
            }
            html.push_str("    </tr>\n");
        }
        html.push_str("  </tbody>\n");
        
        html.push_str("</table>");
        html
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "Table(rows={}, cols={}, has_headers={})",
            self.row_count(),
            self.column_count(),
            self.headers.is_some()
        )
    }
}

/// Image representation for extracted images
/// 
/// Represents an image extracted from a document with metadata
/// and optional OCR text.
#[pyclass]
#[derive(Clone)]
pub struct Image {
    /// Image data (bytes)
    #[pyo3(get)]
    pub data: Vec<u8>,
    
    /// Image format (e.g., "jpeg", "png")
    #[pyo3(get, set)]
    pub format: String,
    
    /// Image width in pixels
    #[pyo3(get, set)]
    pub width: Option<u32>,
    
    /// Image height in pixels
    #[pyo3(get, set)]
    pub height: Option<u32>,
    
    /// Alt text or caption
    #[pyo3(get, set)]
    pub alt_text: Option<String>,
    
    /// OCR-extracted text
    #[pyo3(get, set)]
    pub ocr_text: Option<String>,
    
    /// Image position in document
    #[pyo3(get, set)]
    pub position: Option<usize>,
}

#[pymethods]
impl Image {
    /// Create a new image
    /// 
    /// # Arguments
    /// * `data` - Image data as bytes
    /// * `format` - Image format
    /// * `width` - Optional width
    /// * `height` - Optional height
    #[new]
    #[pyo3(signature = (data, format, width = None, height = None))]
    pub fn new(
        data: Vec<u8>,
        format: String,
        width: Option<u32>,
        height: Option<u32>,
    ) -> Self {
        Self {
            data,
            format,
            width,
            height,
            alt_text: None,
            ocr_text: None,
            position: None,
        }
    }
    
    /// Get image size in bytes
    /// 
    /// # Returns
    /// Size of image data in bytes
    #[getter]
    pub fn size(&self) -> usize {
        self.data.len()
    }
    
    /// Get image dimensions
    /// 
    /// # Returns
    /// Tuple of (width, height) or None if dimensions unknown
    #[getter]
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        match (self.width, self.height) {
            (Some(w), Some(h)) => Some((w, h)),
            _ => None,
        }
    }
    
    /// Check if image has OCR text
    /// 
    /// # Returns
    /// True if OCR text is available
    #[getter]
    pub fn has_ocr_text(&self) -> bool {
        self.ocr_text.is_some() && !self.ocr_text.as_ref().unwrap().is_empty()
    }
    
    /// Save image to file
    /// 
    /// # Arguments
    /// * `path` - File path to save to
    /// 
    /// # Returns
    /// True if saved successfully
    pub fn save(&self, path: &str) -> PyResult<bool> {
        std::fs::write(path, &self.data)
            .map(|_| true)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to save image: {}", e)))
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "Image(format='{}', size={} bytes, dimensions={:?})",
            self.format,
            self.size(),
            self.dimensions()
        )
    }
}

/// Processing statistics container
#[pyclass]
#[derive(Clone)]
pub struct ProcessingStats {
    /// Processing time in milliseconds
    #[pyo3(get, set)]
    pub processing_time_ms: u64,
    
    /// Document size in bytes
    #[pyo3(get, set)]
    pub document_size: usize,
    
    /// Number of pages processed
    #[pyo3(get, set)]
    pub page_count: Option<usize>,
    
    /// Number of characters extracted
    #[pyo3(get, set)]
    pub character_count: Option<usize>,
    
    /// Number of words extracted
    #[pyo3(get, set)]
    pub word_count: Option<usize>,
    
    /// Memory usage in bytes
    #[pyo3(get, set)]
    pub memory_usage: Option<usize>,
}

#[pymethods]
impl ProcessingStats {
    /// Create new processing stats
    #[new]
    pub fn new() -> Self {
        Self {
            processing_time_ms: 0,
            document_size: 0,
            page_count: None,
            character_count: None,
            word_count: None,
            memory_usage: None,
        }
    }
    
    /// Get processing speed in bytes per second
    /// 
    /// # Returns
    /// Processing speed or None if time is zero
    #[getter]
    pub fn bytes_per_second(&self) -> Option<f64> {
        if self.processing_time_ms > 0 {
            Some(self.document_size as f64 * 1000.0 / self.processing_time_ms as f64)
        } else {
            None
        }
    }
    
    /// Get processing speed in pages per second
    /// 
    /// # Returns
    /// Pages per second or None if time is zero or page count unknown
    #[getter]
    pub fn pages_per_second(&self) -> Option<f64> {
        if self.processing_time_ms > 0 {
            self.page_count.map(|pages| pages as f64 * 1000.0 / self.processing_time_ms as f64)
        } else {
            None
        }
    }
    
    /// Convert to dictionary
    /// 
    /// # Returns
    /// Dictionary containing all statistics
    pub fn to_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut stats = HashMap::new();
        
        stats.insert("processing_time_ms".to_string(), 
                     serde_json::Value::Number(serde_json::Number::from(self.processing_time_ms)));
        stats.insert("document_size".to_string(),
                     serde_json::Value::Number(serde_json::Number::from(self.document_size)));
        
        if let Some(pages) = self.page_count {
            stats.insert("page_count".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(pages)));
        }
        
        if let Some(chars) = self.character_count {
            stats.insert("character_count".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(chars)));
        }
        
        if let Some(words) = self.word_count {
            stats.insert("word_count".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(words)));
        }
        
        if let Some(memory) = self.memory_usage {
            stats.insert("memory_usage".to_string(),
                        serde_json::Value::Number(serde_json::Number::from(memory)));
        }
        
        if let Some(bps) = self.bytes_per_second() {
            stats.insert("bytes_per_second".to_string(),
                        serde_json::Value::Number(serde_json::Number::from_f64(bps).unwrap()));
        }
        
        if let Some(pps) = self.pages_per_second() {
            stats.insert("pages_per_second".to_string(),
                        serde_json::Value::Number(serde_json::Number::from_f64(pps).unwrap()));
        }
        
        stats
    }
    
    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "ProcessingStats(time={}ms, size={} bytes, pages={:?})",
            self.processing_time_ms,
            self.document_size,
            self.page_count
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_document_metadata() {
        let mut metadata = DocumentMetadata::new();
        assert_eq!(metadata.size(), 0);
        
        metadata.set("test".to_string(), serde_json::Value::String("value".to_string()));
        assert_eq!(metadata.size(), 1);
        assert!(metadata.contains("test"));
        
        let value = metadata.get("test", None);
        assert_eq!(value, serde_json::Value::String("value".to_string()));
    }
    
    #[test]
    fn test_table() {
        let rows = vec![
            vec!["A".to_string(), "B".to_string()],
            vec!["1".to_string(), "2".to_string()],
        ];
        let headers = Some(vec!["Col1".to_string(), "Col2".to_string()]);
        
        let table = Table::new(rows, headers, None);
        assert_eq!(table.row_count(), 2);
        assert_eq!(table.column_count(), 2);
        assert_eq!(table.get_cell(1, 1), Some("2".to_string()));
    }
    
    #[test]
    fn test_image() {
        let data = vec![1, 2, 3, 4];
        let image = Image::new(data.clone(), "jpeg".to_string(), Some(100), Some(200));
        
        assert_eq!(image.size(), 4);
        assert_eq!(image.dimensions(), Some((100, 200)));
        assert_eq!(image.format, "jpeg");
    }
    
    #[test]
    fn test_processing_stats() {
        let mut stats = ProcessingStats::new();
        stats.processing_time_ms = 1000; // 1 second
        stats.document_size = 10000; // 10KB
        
        assert_eq!(stats.bytes_per_second(), Some(10000.0));
        
        stats.page_count = Some(5);
        assert_eq!(stats.pages_per_second(), Some(5.0));
    }
}