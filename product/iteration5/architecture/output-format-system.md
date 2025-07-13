# Output Format System Design

## Overview

The output format system provides a flexible, type-safe, and performant framework for transforming extracted document data into various output formats. It supports template-based formatting, programmatic transformations, and streaming for large documents.

## Core Architecture

```rust
use serde::{Serialize, Deserialize};
use handlebars::Handlebars;
use tera::Tera;
use std::io::Write;

/// Core trait for output formatters
pub trait OutputFormatter: Send + Sync {
    /// Format extracted data into the target format
    fn format(&self, data: &ExtractedData) -> Result<FormattedOutput, FormatError>;
    
    /// Stream format for large documents
    fn format_stream<W: Write>(&self, data: &ExtractedData, writer: W) -> Result<(), FormatError>;
    
    /// Validate output against schema
    fn validate(&self, output: &FormattedOutput) -> Result<(), ValidationError>;
    
    /// Get format metadata
    fn metadata(&self) -> FormatMetadata;
}

/// Extracted data representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedData {
    pub metadata: DocumentMetadata,
    pub fields: HashMap<String, FieldValue>,
    pub tables: Vec<ExtractedTable>,
    pub sections: Vec<ExtractedSection>,
    pub relationships: Vec<DataRelationship>,
    pub annotations: Vec<Annotation>,
}

/// Formatted output wrapper
#[derive(Debug)]
pub enum FormattedOutput {
    Text(String),
    Binary(Vec<u8>),
    Stream(Box<dyn Read + Send>),
}

/// Format metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatMetadata {
    pub name: String,
    pub mime_type: String,
    pub file_extension: String,
    pub supports_streaming: bool,
    pub schema_validation: bool,
}
```

## Template-Based Formatting

### Handlebars Templates

```rust
pub struct HandlebarsFormatter {
    engine: Handlebars<'static>,
    template_name: String,
    helpers: HashMap<String, Box<dyn Helper>>,
}

impl HandlebarsFormatter {
    pub fn new(template: &str) -> Result<Self, TemplateError> {
        let mut engine = Handlebars::new();
        
        // Register built-in helpers
        engine.register_helper("format_date", Box::new(format_date_helper));
        engine.register_helper("currency", Box::new(currency_helper));
        engine.register_helper("table", Box::new(table_helper));
        engine.register_helper("conditional", Box::new(conditional_helper));
        
        // Register template
        engine.register_template_string("main", template)?;
        
        Ok(Self {
            engine,
            template_name: "main".to_string(),
            helpers: HashMap::new(),
        })
    }
    
    pub fn register_helper(&mut self, name: &str, helper: Box<dyn Helper>) {
        self.helpers.insert(name.to_string(), helper);
        self.engine.register_helper(name, helper);
    }
}

// Example Handlebars template
const INVOICE_TEMPLATE: &str = r#"
{
  "invoice": {
    "number": "{{metadata.invoice_number}}",
    "date": "{{format_date metadata.date 'YYYY-MM-DD'}}",
    "vendor": {
      "name": "{{fields.vendor_name}}",
      "address": "{{fields.vendor_address}}",
      "tax_id": "{{fields.vendor_tax_id}}"
    },
    "customer": {
      "name": "{{fields.customer_name}}",
      "address": "{{fields.customer_address}}"
    },
    "items": [
      {{#each tables.0.rows}}
      {
        "description": "{{cells.0}}",
        "quantity": {{cells.1}},
        "unit_price": {{currency cells.2}},
        "total": {{currency cells.3}}
      }{{#unless @last}},{{/unless}}
      {{/each}}
    ],
    "totals": {
      "subtotal": {{currency fields.subtotal}},
      "tax": {{currency fields.tax}},
      "total": {{currency fields.total}}
    }
  }
}
"#;
```

### Tera Templates

```rust
pub struct TeraFormatter {
    engine: Tera,
    template_name: String,
    filters: HashMap<String, Box<dyn Filter>>,
}

impl TeraFormatter {
    pub fn new(template: &str) -> Result<Self, TemplateError> {
        let mut engine = Tera::default();
        
        // Add custom filters
        engine.register_filter("format_number", format_number_filter);
        engine.register_filter("truncate", truncate_filter);
        engine.register_filter("highlight", highlight_filter);
        
        // Add template
        engine.add_raw_template("main", template)?;
        
        Ok(Self {
            engine,
            template_name: "main".to_string(),
            filters: HashMap::new(),
        })
    }
}

// Example Tera template for Markdown report
const REPORT_TEMPLATE: &str = r#"
# {{ metadata.title }}

**Date**: {{ metadata.date | date(format="%B %d, %Y") }}
**Document Type**: {{ metadata.document_type }}

## Summary

{{ fields.summary | default(value="No summary available") }}

## Key Information

{% for key, value in fields %}
{% if key starts_with("key_") %}
- **{{ key | replace(from="key_", to="") | title }}**: {{ value }}
{% endif %}
{% endfor %}

## Data Tables

{% for table in tables %}
### {{ table.title | default(value="Table " ~ loop.index) }}

{{ table | render_markdown_table }}

{% endfor %}

## Annotations

{% for annotation in annotations %}
> **{{ annotation.type }}**: {{ annotation.text }}
> 
> *Confidence: {{ annotation.confidence | format_number(decimals=2) }}*
{% endfor %}
"#;
```

## Programmatic Formatting

### Rust Closures

```rust
pub struct ClosureFormatter<F>
where
    F: Fn(&ExtractedData) -> Result<FormattedOutput, FormatError> + Send + Sync,
{
    formatter: F,
    metadata: FormatMetadata,
}

impl<F> ClosureFormatter<F>
where
    F: Fn(&ExtractedData) -> Result<FormattedOutput, FormatError> + Send + Sync,
{
    pub fn new(name: &str, formatter: F) -> Self {
        Self {
            formatter,
            metadata: FormatMetadata {
                name: name.to_string(),
                mime_type: "application/octet-stream".to_string(),
                file_extension: "bin".to_string(),
                supports_streaming: false,
                schema_validation: false,
            },
        }
    }
}

// Example: Custom CSV formatter with complex field mappings
let csv_formatter = ClosureFormatter::new("custom_csv", |data| {
    let mut writer = csv::Writer::from_writer(Vec::new());
    
    // Write header with custom field names
    writer.write_record(&[
        "Invoice Number",
        "Date",
        "Vendor",
        "Customer",
        "Total Amount",
        "Status"
    ])?;
    
    // Write data with transformations
    writer.write_record(&[
        data.fields.get("invoice_number").unwrap_or(&FieldValue::Empty).to_string(),
        format_date(&data.metadata.date),
        data.fields.get("vendor_name").unwrap_or(&FieldValue::Empty).to_string(),
        data.fields.get("customer_name").unwrap_or(&FieldValue::Empty).to_string(),
        format_currency(data.fields.get("total").unwrap_or(&FieldValue::Number(0.0))),
        determine_status(&data),
    ])?;
    
    // Add line items from tables
    if let Some(items_table) = data.tables.get(0) {
        for row in &items_table.rows {
            writer.write_record(&[
                "", // Empty invoice number
                "", // Empty date
                "", // Empty vendor
                row.cells.get(0).unwrap_or(&"").to_string(), // Item description
                row.cells.get(3).unwrap_or(&"0.00").to_string(), // Amount
                "Item", // Status
            ])?;
        }
    }
    
    let csv_data = writer.into_inner()?;
    Ok(FormattedOutput::Text(String::from_utf8(csv_data)?))
});
```

## Standard Format Implementations

### JSON Formatter

```rust
pub struct JsonFormatter {
    pretty: bool,
    schema: Option<JsonSchema>,
    transformations: Vec<JsonTransformation>,
}

impl JsonFormatter {
    pub fn new() -> Self {
        Self {
            pretty: true,
            schema: None,
            transformations: Vec::new(),
        }
    }
    
    pub fn with_schema(mut self, schema: JsonSchema) -> Self {
        self.schema = Some(schema);
        self
    }
    
    pub fn with_transformation(mut self, transform: JsonTransformation) -> Self {
        self.transformations.push(transform);
        self
    }
}

impl OutputFormatter for JsonFormatter {
    fn format(&self, data: &ExtractedData) -> Result<FormattedOutput, FormatError> {
        let mut json_value = serde_json::to_value(data)?;
        
        // Apply transformations
        for transform in &self.transformations {
            json_value = transform.apply(json_value)?;
        }
        
        // Validate against schema
        if let Some(schema) = &self.schema {
            schema.validate(&json_value)?;
        }
        
        // Serialize
        let output = if self.pretty {
            serde_json::to_string_pretty(&json_value)?
        } else {
            serde_json::to_string(&json_value)?
        };
        
        Ok(FormattedOutput::Text(output))
    }
}

// Example: JSON with custom schema
let invoice_schema = json_schema!({
    "type": "object",
    "properties": {
        "invoice": {
            "type": "object",
            "required": ["number", "date", "vendor", "customer", "total"],
            "properties": {
                "number": { "type": "string", "pattern": "^INV-[0-9]{6}$" },
                "date": { "type": "string", "format": "date" },
                "vendor": {
                    "type": "object",
                    "required": ["name", "tax_id"],
                    "properties": {
                        "name": { "type": "string" },
                        "tax_id": { "type": "string" }
                    }
                },
                "total": { "type": "number", "minimum": 0 }
            }
        }
    }
});
```

### XML Formatter

```rust
pub struct XmlFormatter {
    root_element: String,
    namespaces: HashMap<String, String>,
    schema: Option<XmlSchema>,
    indent: bool,
}

impl XmlFormatter {
    pub fn new(root_element: &str) -> Self {
        Self {
            root_element: root_element.to_string(),
            namespaces: HashMap::new(),
            schema: None,
            indent: true,
        }
    }
    
    pub fn with_namespace(mut self, prefix: &str, uri: &str) -> Self {
        self.namespaces.insert(prefix.to_string(), uri.to_string());
        self
    }
}

// Example: XML with namespace support
let xml_formatter = XmlFormatter::new("invoice")
    .with_namespace("inv", "http://example.com/invoice/v1")
    .with_namespace("cust", "http://example.com/customer/v1");

// Output example:
// <?xml version="1.0" encoding="UTF-8"?>
// <invoice xmlns:inv="http://example.com/invoice/v1" 
//          xmlns:cust="http://example.com/customer/v1">
//   <inv:number>INV-123456</inv:number>
//   <inv:date>2024-01-15</inv:date>
//   <cust:customer>
//     <cust:name>Acme Corp</cust:name>
//     <cust:id>CUST-789</cust:id>
//   </cust:customer>
// </invoice>
```

### CSV Formatter

```rust
pub struct CsvFormatter {
    delimiter: u8,
    headers: Vec<String>,
    field_mappings: HashMap<String, FieldMapping>,
    quote_style: csv::QuoteStyle,
}

#[derive(Clone)]
pub struct FieldMapping {
    pub source_path: String,
    pub transform: Option<Box<dyn Fn(&FieldValue) -> String + Send + Sync>>,
    pub default_value: String,
}

// Example: CSV with complex field mappings
let csv_formatter = CsvFormatter::new()
    .with_headers(vec![
        "Invoice Number",
        "Date",
        "Vendor Name",
        "Customer Name",
        "Item Count",
        "Total Amount"
    ])
    .with_field_mapping("Invoice Number", FieldMapping {
        source_path: "fields.invoice_number".to_string(),
        transform: None,
        default_value: "N/A".to_string(),
    })
    .with_field_mapping("Total Amount", FieldMapping {
        source_path: "fields.total".to_string(),
        transform: Some(Box::new(|value| {
            match value {
                FieldValue::Number(n) => format!("${:.2}", n),
                _ => "$0.00".to_string(),
            }
        })),
        default_value: "$0.00".to_string(),
    });
```

### Markdown Formatter

```rust
pub struct MarkdownFormatter {
    template: Option<String>,
    table_style: TableStyle,
    code_highlighting: bool,
    metadata_section: bool,
}

#[derive(Clone, Copy)]
pub enum TableStyle {
    Github,
    Simple,
    Grid,
    Pipe,
}

impl MarkdownFormatter {
    pub fn new() -> Self {
        Self {
            template: None,
            table_style: TableStyle::Github,
            code_highlighting: true,
            metadata_section: true,
        }
    }
    
    fn render_table(&self, table: &ExtractedTable) -> String {
        match self.table_style {
            TableStyle::Github => {
                let mut output = String::new();
                
                // Headers
                if let Some(headers) = &table.headers {
                    output.push('|');
                    for header in headers {
                        output.push_str(&format!(" {} |", header));
                    }
                    output.push('\n');
                    
                    // Separator
                    output.push('|');
                    for _ in headers {
                        output.push_str(" --- |");
                    }
                    output.push('\n');
                }
                
                // Rows
                for row in &table.rows {
                    output.push('|');
                    for cell in &row.cells {
                        output.push_str(&format!(" {} |", cell));
                    }
                    output.push('\n');
                }
                
                output
            },
            TableStyle::Simple => {
                // Simple space-separated format
                let mut output = String::new();
                for row in &table.rows {
                    output.push_str(&row.cells.join("  "));
                    output.push('\n');
                }
                output
            },
            _ => unimplemented!()
        }
    }
}

// Example: Markdown report generation
let markdown_report = r#"
# Financial Report: {{ metadata.title }}

**Generated**: {{ now | date(format="%Y-%m-%d %H:%M") }}
**Document Type**: {{ metadata.document_type | title }}
**Confidence Score**: {{ metadata.confidence | format_percent }}

## Executive Summary

{{ fields.executive_summary | default(value="*No summary provided*") }}

## Key Metrics

| Metric | Value | Change |
|--------|-------|--------|
| Revenue | {{ fields.revenue | currency }} | {{ fields.revenue_change | format_percent }} |
| Expenses | {{ fields.expenses | currency }} | {{ fields.expenses_change | format_percent }} |
| Net Profit | {{ fields.net_profit | currency }} | {{ fields.profit_change | format_percent }} |

## Detailed Analysis

{{ sections | where(attribute="type", value="analysis") | first | get(attribute="content") }}

## Data Tables

{% for table in tables %}
### {{ table.title }}

{{ table | render_markdown_table }}

*Table confidence: {{ table.confidence | format_percent }}*
{% endfor %}

## Appendix

### Extraction Metadata

```json
{{ metadata | json_encode(pretty=true) }}
```
"#;
```

### HTML Formatter

```rust
pub struct HtmlFormatter {
    template: String,
    css_styles: Vec<String>,
    javascript: Vec<String>,
    sanitize: bool,
}

impl HtmlFormatter {
    pub fn new() -> Self {
        Self {
            template: DEFAULT_HTML_TEMPLATE.to_string(),
            css_styles: vec![DEFAULT_STYLES.to_string()],
            javascript: Vec::new(),
            sanitize: true,
        }
    }
    
    pub fn with_custom_css(mut self, css: &str) -> Self {
        self.css_styles.push(css.to_string());
        self
    }
}

const DEFAULT_HTML_TEMPLATE: &str = r#"
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ metadata.title | escape }}</title>
    <style>
        {{ css_styles | join }}
    </style>
</head>
<body>
    <header>
        <h1>{{ metadata.title | escape }}</h1>
        <div class="metadata">
            <span class="date">{{ metadata.date | date(format="%B %d, %Y") }}</span>
            <span class="type">{{ metadata.document_type | title }}</span>
        </div>
    </header>
    
    <main>
        <section class="summary">
            <h2>Summary</h2>
            <p>{{ fields.summary | escape | linebreaks }}</p>
        </section>
        
        <section class="data">
            <h2>Extracted Data</h2>
            {% for key, value in fields %}
            <div class="field">
                <span class="label">{{ key | humanize }}:</span>
                <span class="value">{{ value | escape }}</span>
            </div>
            {% endfor %}
        </section>
        
        {% if tables %}
        <section class="tables">
            <h2>Tables</h2>
            {% for table in tables %}
            <div class="table-container">
                <h3>{{ table.title | default(value="Table " ~ loop.index) }}</h3>
                <table>
                    {% if table.headers %}
                    <thead>
                        <tr>
                            {% for header in table.headers %}
                            <th>{{ header | escape }}</th>
                            {% endfor %}
                        </tr>
                    </thead>
                    {% endif %}
                    <tbody>
                        {% for row in table.rows %}
                        <tr>
                            {% for cell in row.cells %}
                            <td>{{ cell | escape }}</td>
                            {% endfor %}
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            {% endfor %}
        </section>
        {% endif %}
    </main>
    
    <footer>
        <p>Generated by Document Processing System</p>
    </footer>
</body>
</html>
"#;
```

## Custom Serialization Protocols

```rust
/// Protocol buffer formatter
pub struct ProtobufFormatter {
    proto_schema: ProtoSchema,
    compression: Option<CompressionType>,
}

/// MessagePack formatter
pub struct MessagePackFormatter {
    schema_validation: bool,
    compact: bool,
}

/// Apache Avro formatter
pub struct AvroFormatter {
    schema: avro::Schema,
    codec: avro::Codec,
}

/// Custom binary format
pub struct BinaryFormatter {
    header: BinaryHeader,
    encoding: BinaryEncoding,
    checksum: ChecksumType,
}

impl BinaryFormatter {
    pub fn format(&self, data: &ExtractedData) -> Result<FormattedOutput, FormatError> {
        let mut buffer = Vec::new();
        
        // Write header
        buffer.extend_from_slice(&self.header.magic_bytes);
        buffer.extend_from_slice(&self.header.version.to_le_bytes());
        
        // Encode data
        let encoded = match self.encoding {
            BinaryEncoding::Raw => bincode::serialize(data)?,
            BinaryEncoding::Compressed(algo) => {
                let raw = bincode::serialize(data)?;
                algo.compress(&raw)?
            },
            BinaryEncoding::Encrypted(cipher) => {
                let raw = bincode::serialize(data)?;
                cipher.encrypt(&raw)?
            },
        };
        
        // Write data length and data
        buffer.extend_from_slice(&(encoded.len() as u32).to_le_bytes());
        buffer.extend_from_slice(&encoded);
        
        // Add checksum
        let checksum = self.checksum.calculate(&buffer);
        buffer.extend_from_slice(&checksum);
        
        Ok(FormattedOutput::Binary(buffer))
    }
}
```

## Streaming Output

```rust
/// Streaming formatter for large documents
pub trait StreamingFormatter: Send + Sync {
    fn format_stream<R: Read, W: Write>(
        &self,
        reader: R,
        writer: W,
        options: StreamingOptions,
    ) -> Result<StreamingStats, FormatError>;
}

#[derive(Debug, Clone)]
pub struct StreamingOptions {
    pub buffer_size: usize,
    pub flush_interval: Duration,
    pub progress_callback: Option<Arc<dyn Fn(usize, usize) + Send + Sync>>,
    pub error_handling: ErrorHandling,
}

#[derive(Debug)]
pub struct StreamingStats {
    pub bytes_read: usize,
    pub bytes_written: usize,
    pub chunks_processed: usize,
    pub duration: Duration,
    pub errors_recovered: usize,
}

/// JSON Lines streaming formatter
pub struct JsonLinesFormatter {
    transformer: Box<dyn Fn(ExtractedChunk) -> Result<serde_json::Value, FormatError> + Send + Sync>,
}

impl StreamingFormatter for JsonLinesFormatter {
    fn format_stream<R: Read, W: Write>(
        &self,
        mut reader: R,
        mut writer: W,
        options: StreamingOptions,
    ) -> Result<StreamingStats, FormatError> {
        let start_time = Instant::now();
        let mut stats = StreamingStats::default();
        let mut buffer = vec![0u8; options.buffer_size];
        
        loop {
            match reader.read(&mut buffer) {
                Ok(0) => break, // EOF
                Ok(n) => {
                    stats.bytes_read += n;
                    
                    // Parse chunk
                    let chunk = ExtractedChunk::from_bytes(&buffer[..n])?;
                    
                    // Transform to JSON
                    let json_value = (self.transformer)(chunk)?;
                    
                    // Write JSON line
                    let json_line = serde_json::to_string(&json_value)?;
                    writer.write_all(json_line.as_bytes())?;
                    writer.write_all(b"\n")?;
                    
                    stats.bytes_written += json_line.len() + 1;
                    stats.chunks_processed += 1;
                    
                    // Progress callback
                    if let Some(callback) = &options.progress_callback {
                        callback(stats.bytes_read, stats.bytes_written);
                    }
                    
                    // Periodic flush
                    if start_time.elapsed() > options.flush_interval {
                        writer.flush()?;
                    }
                },
                Err(e) => {
                    match options.error_handling {
                        ErrorHandling::Skip => {
                            stats.errors_recovered += 1;
                            continue;
                        },
                        ErrorHandling::Abort => return Err(e.into()),
                        ErrorHandling::Retry(max_retries) => {
                            // Retry logic
                        },
                    }
                }
            }
        }
        
        writer.flush()?;
        stats.duration = start_time.elapsed();
        Ok(stats)
    }
}

/// CSV streaming formatter
pub struct CsvStreamingFormatter {
    headers: Vec<String>,
    delimiter: u8,
    chunk_size: usize,
}

/// XML streaming formatter with SAX-style processing
pub struct XmlStreamingFormatter {
    root_element: String,
    element_handler: Box<dyn Fn(&ExtractedElement) -> Result<String, FormatError> + Send + Sync>,
}
```

## Format Validation

```rust
/// Format validation framework
pub trait FormatValidator: Send + Sync {
    fn validate(&self, output: &[u8]) -> Result<ValidationReport, ValidationError>;
    fn validate_stream<R: Read>(&self, reader: R) -> Result<ValidationReport, ValidationError>;
}

#[derive(Debug)]
pub struct ValidationReport {
    pub is_valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
    pub stats: ValidationStats,
}

/// JSON Schema validator
pub struct JsonSchemaValidator {
    schema: serde_json::Value,
    draft: JsonSchemaDraft,
}

/// XML Schema validator
pub struct XmlSchemaValidator {
    schema: XmlSchema,
    namespaces: HashMap<String, String>,
}

/// Custom validation rules
pub struct RuleBasedValidator {
    rules: Vec<ValidationRule>,
}

#[derive(Clone)]
pub struct ValidationRule {
    pub name: String,
    pub path: String,
    pub condition: ValidationCondition,
    pub severity: Severity,
    pub message: String,
}

#[derive(Clone)]
pub enum ValidationCondition {
    Required,
    Pattern(Regex),
    Range { min: f64, max: f64 },
    Length { min: usize, max: usize },
    Custom(Box<dyn Fn(&Value) -> bool + Send + Sync>),
}

// Example: Complex validation rules
let validator = RuleBasedValidator::new()
    .add_rule(ValidationRule {
        name: "invoice_number_format".to_string(),
        path: "invoice.number".to_string(),
        condition: ValidationCondition::Pattern(
            Regex::new(r"^INV-[0-9]{6}$").unwrap()
        ),
        severity: Severity::Error,
        message: "Invoice number must match pattern INV-XXXXXX".to_string(),
    })
    .add_rule(ValidationRule {
        name: "total_amount_positive".to_string(),
        path: "invoice.total".to_string(),
        condition: ValidationCondition::Range { min: 0.0, max: f64::MAX },
        severity: Severity::Error,
        message: "Total amount must be positive".to_string(),
    })
    .add_rule(ValidationRule {
        name: "date_not_future".to_string(),
        path: "invoice.date".to_string(),
        condition: ValidationCondition::Custom(Box::new(|value| {
            if let Some(date_str) = value.as_str() {
                if let Ok(date) = NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
                    return date <= Local::now().naive_local().date();
                }
            }
            false
        })),
        severity: Severity::Warning,
        message: "Invoice date should not be in the future".to_string(),
    });
```

## Performance Optimization

```rust
/// Performance-optimized formatter wrapper
pub struct OptimizedFormatter<F: OutputFormatter> {
    inner: F,
    cache: Arc<RwLock<LruCache<u64, FormattedOutput>>>,
    metrics: Arc<Mutex<PerformanceMetrics>>,
}

#[derive(Debug, Default)]
pub struct PerformanceMetrics {
    pub total_formats: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub avg_format_time: Duration,
    pub total_bytes_formatted: u64,
}

impl<F: OutputFormatter> OptimizedFormatter<F> {
    pub fn new(inner: F, cache_size: usize) -> Self {
        Self {
            inner,
            cache: Arc::new(RwLock::new(LruCache::new(cache_size))),
            metrics: Arc::new(Mutex::new(PerformanceMetrics::default())),
        }
    }
    
    pub fn format_with_cache(&self, data: &ExtractedData) -> Result<FormattedOutput, FormatError> {
        let hash = calculate_hash(data);
        
        // Check cache
        {
            let cache = self.cache.read().unwrap();
            if let Some(cached) = cache.get(&hash) {
                let mut metrics = self.metrics.lock().unwrap();
                metrics.cache_hits += 1;
                return Ok(cached.clone());
            }
        }
        
        // Format data
        let start = Instant::now();
        let output = self.inner.format(data)?;
        let duration = start.elapsed();
        
        // Update cache
        {
            let mut cache = self.cache.write().unwrap();
            cache.put(hash, output.clone());
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            metrics.total_formats += 1;
            metrics.cache_misses += 1;
            metrics.avg_format_time = (
                metrics.avg_format_time * (metrics.total_formats - 1) + duration
            ) / metrics.total_formats;
        }
        
        Ok(output)
    }
}

/// Parallel formatter for batch processing
pub struct ParallelFormatter<F: OutputFormatter + Clone> {
    formatter: F,
    thread_pool: ThreadPool,
    chunk_size: usize,
}

impl<F: OutputFormatter + Clone> ParallelFormatter<F> {
    pub fn format_batch(
        &self,
        documents: Vec<ExtractedData>,
    ) -> Result<Vec<FormattedOutput>, FormatError> {
        let chunks: Vec<_> = documents.chunks(self.chunk_size).collect();
        let (tx, rx) = mpsc::channel();
        
        for (idx, chunk) in chunks.into_iter().enumerate() {
            let formatter = self.formatter.clone();
            let tx = tx.clone();
            let chunk = chunk.to_vec();
            
            self.thread_pool.execute(move || {
                let results: Vec<_> = chunk
                    .into_iter()
                    .map(|doc| formatter.format(&doc))
                    .collect();
                tx.send((idx, results)).unwrap();
            });
        }
        
        drop(tx);
        
        // Collect results in order
        let mut all_results = vec![];
        let mut received: HashMap<usize, Vec<Result<FormattedOutput, FormatError>>> = HashMap::new();
        
        for (idx, results) in rx {
            received.insert(idx, results);
        }
        
        for idx in 0..received.len() {
            all_results.extend(received.remove(&idx).unwrap());
        }
        
        // Convert Results to single Result
        all_results.into_iter().collect()
    }
}
```

## Format Registry and Discovery

```rust
/// Central registry for output formatters
pub struct FormatRegistry {
    formatters: HashMap<String, Box<dyn OutputFormatter>>,
    aliases: HashMap<String, String>,
    mime_mappings: HashMap<String, String>,
}

impl FormatRegistry {
    pub fn new() -> Self {
        let mut registry = Self {
            formatters: HashMap::new(),
            aliases: HashMap::new(),
            mime_mappings: HashMap::new(),
        };
        
        // Register standard formatters
        registry.register("json", Box::new(JsonFormatter::new()));
        registry.register("xml", Box::new(XmlFormatter::new("document")));
        registry.register("csv", Box::new(CsvFormatter::new()));
        registry.register("markdown", Box::new(MarkdownFormatter::new()));
        registry.register("html", Box::new(HtmlFormatter::new()));
        
        // Register aliases
        registry.add_alias("md", "markdown");
        registry.add_alias("htm", "html");
        
        // Register MIME mappings
        registry.add_mime_mapping("application/json", "json");
        registry.add_mime_mapping("text/xml", "xml");
        registry.add_mime_mapping("text/csv", "csv");
        registry.add_mime_mapping("text/markdown", "markdown");
        registry.add_mime_mapping("text/html", "html");
        
        registry
    }
    
    pub fn register(&mut self, name: &str, formatter: Box<dyn OutputFormatter>) {
        self.formatters.insert(name.to_string(), formatter);
    }
    
    pub fn get(&self, name: &str) -> Option<&Box<dyn OutputFormatter>> {
        // Check direct name
        if let Some(formatter) = self.formatters.get(name) {
            return Some(formatter);
        }
        
        // Check aliases
        if let Some(real_name) = self.aliases.get(name) {
            return self.formatters.get(real_name);
        }
        
        // Check MIME type
        if let Some(format_name) = self.mime_mappings.get(name) {
            return self.formatters.get(format_name);
        }
        
        None
    }
    
    pub fn discover_format(&self, sample: &[u8]) -> Option<String> {
        // Try to detect format from content
        if sample.starts_with(b"{") || sample.starts_with(b"[") {
            return Some("json".to_string());
        }
        
        if sample.starts_with(b"<?xml") || sample.starts_with(b"<") {
            return Some("xml".to_string());
        }
        
        if sample.starts_with(b"#") || sample.contains(&b'\n') && sample.contains(&b'|') {
            return Some("markdown".to_string());
        }
        
        if sample.starts_with(b"<!DOCTYPE") || sample.starts_with(b"<html") {
            return Some("html".to_string());
        }
        
        None
    }
}
```

## Usage Examples

### Example 1: Invoice to JSON with Custom Schema

```rust
// Define schema
let invoice_schema = json_schema!({
    "type": "object",
    "required": ["invoice"],
    "properties": {
        "invoice": {
            "type": "object",
            "required": ["number", "date", "vendor", "customer", "items", "total"],
            "properties": {
                "number": {
                    "type": "string",
                    "pattern": "^INV-[0-9]{6}$"
                },
                "date": {
                    "type": "string",
                    "format": "date"
                },
                "vendor": {
                    "type": "object",
                    "required": ["name", "tax_id"]
                },
                "customer": {
                    "type": "object",
                    "required": ["name"]
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["description", "quantity", "unit_price", "total"]
                    }
                },
                "total": {
                    "type": "number",
                    "minimum": 0
                }
            }
        }
    }
});

// Create formatter
let formatter = JsonFormatter::new()
    .with_schema(invoice_schema)
    .with_transformation(JsonTransformation::Rename {
        from: "fields.invoice_no".to_string(),
        to: "invoice.number".to_string(),
    })
    .with_transformation(JsonTransformation::Transform {
        path: "invoice.date".to_string(),
        transformer: Box::new(|value| {
            // Convert date format
            if let Some(date_str) = value.as_str() {
                if let Ok(date) = NaiveDate::parse_from_str(date_str, "%m/%d/%Y") {
                    return json!(date.format("%Y-%m-%d").to_string());
                }
            }
            value
        }),
    });

// Format data
let output = formatter.format(&extracted_data)?;
```

### Example 2: Financial Report to Markdown

```rust
// Create Markdown formatter with custom template
let template = r#"
# {{ metadata.title }}

**Report Date**: {{ metadata.date | date(format="%B %d, %Y") }}
**Period**: {{ fields.period_start | date(format="%Y-%m-%d") }} to {{ fields.period_end | date(format="%Y-%m-%d") }}

## Executive Summary

{{ fields.executive_summary }}

## Financial Highlights

- **Revenue**: {{ fields.total_revenue | currency }} ({{ fields.revenue_growth | format_percent }} YoY)
- **Operating Income**: {{ fields.operating_income | currency }}
- **Net Income**: {{ fields.net_income | currency }}
- **EPS**: {{ fields.earnings_per_share | currency }}

## Revenue by Segment

{% for segment in tables.revenue_segments.rows %}
| {{ segment.cells.0 }} | {{ segment.cells.1 | currency }} | {{ segment.cells.2 | format_percent }} |
{% endfor %}

## Key Metrics Trend

```chart
{{ charts.metrics_trend | render_ascii_chart }}
```

## Management Commentary

{% for section in sections %}
{% if section.type == "commentary" %}
### {{ section.title }}

{{ section.content }}
{% endif %}
{% endfor %}

---
*This report was automatically generated from {{ metadata.source_document }}*
"#;

let formatter = TeraFormatter::new(template)?
    .with_filter("currency", |value: &Value, _: &HashMap<String, Value>| {
        if let Some(num) = value.as_f64() {
            Ok(Value::String(format!("${:,.2}", num)))
        } else {
            Ok(Value::String("$0.00".to_string()))
        }
    })
    .with_filter("format_percent", |value: &Value, _: &HashMap<String, Value>| {
        if let Some(num) = value.as_f64() {
            Ok(Value::String(format!("{:+.1}%", num * 100.0)))
        } else {
            Ok(Value::String("0.0%".to_string()))
        }
    });

let output = formatter.format(&financial_report_data)?;
```

### Example 3: Streaming Large Dataset to CSV

```rust
// Create streaming CSV formatter
let csv_formatter = CsvStreamingFormatter::new()
    .with_headers(vec![
        "Document ID",
        "Date",
        "Type",
        "Customer",
        "Amount",
        "Status"
    ])
    .with_chunk_size(1000);

// Set up streaming options
let options = StreamingOptions {
    buffer_size: 64 * 1024, // 64KB buffer
    flush_interval: Duration::from_secs(5),
    progress_callback: Some(Arc::new(|read, written| {
        println!("Progress: {} bytes read, {} bytes written", read, written);
    })),
    error_handling: ErrorHandling::Skip,
};

// Stream format
let input = File::open("large_dataset.bin")?;
let output = File::create("output.csv")?;

let stats = csv_formatter.format_stream(input, output, options)?;

println!("Streaming completed: {:?}", stats);
```

### Example 4: HTML Report with Custom Styling

```rust
// Custom CSS for professional report
let custom_css = r#"
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

body {
    font-family: 'Inter', sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f5f5f5;
}

header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 8px;
    margin-bottom: 2rem;
}

.metadata {
    display: flex;
    gap: 2rem;
    margin-top: 1rem;
}

.field {
    display: flex;
    justify-content: space-between;
    padding: 0.75rem;
    border-bottom: 1px solid #e0e0e0;
}

.field:nth-child(even) {
    background-color: #f9f9f9;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 1rem;
}

th, td {
    padding: 0.75rem;
    text-align: left;
    border-bottom: 1px solid #ddd;
}

th {
    background-color: #667eea;
    color: white;
    font-weight: 600;
}

tr:hover {
    background-color: #f5f5f5;
}

.confidence-high { color: #4caf50; }
.confidence-medium { color: #ff9800; }
.confidence-low { color: #f44336; }
"#;

let html_formatter = HtmlFormatter::new()
    .with_custom_css(custom_css)
    .with_template(PROFESSIONAL_REPORT_TEMPLATE);

let output = html_formatter.format(&extracted_data)?;

// Save as HTML file
if let FormattedOutput::Text(html) = output {
    std::fs::write("report.html", html)?;
}
```

## Performance Considerations

1. **Template Compilation**: Pre-compile templates for repeated use
2. **Streaming**: Use streaming formatters for large documents
3. **Caching**: Cache formatted output for identical inputs
4. **Parallelization**: Process multiple documents in parallel
5. **Memory Management**: Use iterators and streams to avoid loading entire documents
6. **Compression**: Apply compression for large outputs
7. **Validation**: Validate output schema asynchronously

## Security Considerations

1. **Template Injection**: Sanitize user-provided templates
2. **XSS Prevention**: Escape HTML content properly
3. **Resource Limits**: Set maximum output size limits
4. **Schema Validation**: Validate against trusted schemas only
5. **Binary Format Security**: Verify magic bytes and checksums
6. **Streaming Limits**: Implement timeouts and size limits

## Testing Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_json_formatter_with_schema() {
        let formatter = JsonFormatter::new()
            .with_schema(test_schema());
        
        let data = create_test_data();
        let output = formatter.format(&data).unwrap();
        
        assert!(matches!(output, FormattedOutput::Text(_)));
        // Validate against schema
    }
    
    #[test]
    fn test_streaming_performance() {
        let formatter = JsonLinesFormatter::new();
        let large_data = create_large_test_data();
        
        let start = Instant::now();
        let stats = formatter.format_stream(
            large_data.as_slice(),
            Vec::new(),
            StreamingOptions::default()
        ).unwrap();
        
        assert!(start.elapsed() < Duration::from_secs(5));
        assert_eq!(stats.chunks_processed, 1000);
    }
    
    #[test]
    fn test_format_detection() {
        let registry = FormatRegistry::new();
        
        assert_eq!(registry.discover_format(b"{\"test\": 1}"), Some("json".to_string()));
        assert_eq!(registry.discover_format(b"<?xml version"), Some("xml".to_string()));
        assert_eq!(registry.discover_format(b"# Header\n"), Some("markdown".to_string()));
    }
}
```