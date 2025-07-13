# DocumentSource Trait Guide

## Understanding and Implementing the DocumentSource Trait

### Overview

The `DocumentSource` trait is the core abstraction that enables the Autonomous Document Extraction Platform to work with documents from any source. It provides a unified interface for accessing documents whether they come from files, URLs, databases, cloud storage, or any other source.

## Trait Definition

```rust
use async_trait::async_trait;

/// Core trait for document sources
#[async_trait]
pub trait DocumentSource: Send + Sync {
    /// Fetches the document content from the source
    async fn fetch(&self) -> Result<DocumentContent, SourceError>;
    
    /// Returns the type of this source
    fn source_type(&self) -> SourceType;
    
    /// Validates the source configuration
    fn validate(&self) -> Result<(), ValidationError>;
    
    /// Returns metadata about the source
    fn metadata(&self) -> SourceMetadata;
}
```

## Built-in Implementations

### FileSource

For processing local files:

```rust
use doc_extract::FileSource;

// Create a file source
let source = FileSource::new("./documents/report.pdf")?;

// With additional metadata
let source = FileSource::builder()
    .path("./documents/report.pdf")
    .expected_content_type("application/pdf")
    .encoding("utf-8")
    .build()?;

// Usage
let content = source.fetch().await?;
println!("File size: {} bytes", content.data.len());
```

**Implementation Details**:
```rust
impl FileSource {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, SourceError> {
        let path = path.as_ref().to_path_buf();
        
        // Validate file exists and is readable
        if !path.exists() {
            return Err(SourceError::NotFound(path.display().to_string()));
        }
        
        if !path.is_file() {
            return Err(SourceError::InvalidSource("Path is not a file".into()));
        }
        
        Ok(Self { path })
    }
}

#[async_trait]
impl DocumentSource for FileSource {
    async fn fetch(&self) -> Result<DocumentContent, SourceError> {
        let data = tokio::fs::read(&self.path)
            .await
            .map_err(|e| SourceError::AccessDenied(e.to_string()))?;
        
        let content_type = detect_content_type(&data, &self.path)?;
        
        Ok(DocumentContent {
            data,
            content_type,
            source_metadata: self.metadata(),
        })
    }
    
    fn source_type(&self) -> SourceType {
        SourceType::File
    }
    
    fn validate(&self) -> Result<(), ValidationError> {
        if !self.path.exists() {
            return Err(ValidationError::SourceNotFound);
        }
        
        let metadata = std::fs::metadata(&self.path)
            .map_err(|_| ValidationError::SourceInaccessible)?;
        
        if metadata.len() > MAX_FILE_SIZE {
            return Err(ValidationError::SourceTooLarge);
        }
        
        Ok(())
    }
    
    fn metadata(&self) -> SourceMetadata {
        SourceMetadata::builder()
            .source_type("file")
            .identifier(self.path.display().to_string())
            .size(self.path.metadata().map(|m| m.len()).unwrap_or(0))
            .build()
    }
}
```

### UrlSource

For processing documents from web URLs:

```rust
use doc_extract::UrlSource;

// Simple URL source
let source = UrlSource::new("https://example.com/document.pdf")?;

// With custom headers and authentication
let source = UrlSource::builder()
    .url("https://api.example.com/documents/123")
    .header("Authorization", "Bearer token123")
    .header("User-Agent", "DocumentExtractor/1.0")
    .timeout(Duration::from_secs(30))
    .follow_redirects(true)
    .max_size(100_000_000) // 100MB
    .build()?;

// Usage
let content = source.fetch().await?;
```

**Implementation Details**:
```rust
#[async_trait]
impl DocumentSource for UrlSource {
    async fn fetch(&self) -> Result<DocumentContent, SourceError> {
        let client = reqwest::Client::builder()
            .timeout(self.timeout)
            .build()
            .map_err(|e| SourceError::ConfigurationError(e.to_string()))?;
        
        let mut request = client.get(&self.url);
        
        // Add custom headers
        for (key, value) in &self.headers {
            request = request.header(key, value);
        }
        
        let response = request
            .send()
            .await
            .map_err(|e| SourceError::NetworkError(e.to_string()))?;
        
        if !response.status().is_success() {
            return Err(SourceError::HttpError {
                status: response.status().as_u16(),
                message: response.status().to_string(),
            });
        }
        
        let content_type = response
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/octet-stream")
            .to_string();
        
        let data = response
            .bytes()
            .await
            .map_err(|e| SourceError::NetworkError(e.to_string()))?
            .to_vec();
        
        Ok(DocumentContent {
            data,
            content_type,
            source_metadata: self.metadata(),
        })
    }
    
    fn source_type(&self) -> SourceType {
        SourceType::Url
    }
    
    fn validate(&self) -> Result<(), ValidationError> {
        let parsed_url = url::Url::parse(&self.url)
            .map_err(|_| ValidationError::InvalidUrl)?;
        
        if !["http", "https"].contains(&parsed_url.scheme()) {
            return Err(ValidationError::UnsupportedScheme);
        }
        
        Ok(())
    }
    
    fn metadata(&self) -> SourceMetadata {
        SourceMetadata::builder()
            .source_type("url")
            .identifier(&self.url)
            .build()
    }
}
```

### Base64Source

For processing base64-encoded documents:

```rust
use doc_extract::Base64Source;

// Data URI format
let source = Base64Source::new("data:application/pdf;base64,JVBERi0xLjQK...")?;

// Raw base64 with content type
let source = Base64Source::builder()
    .data("JVBERi0xLjQK...")
    .content_type("application/pdf")
    .filename("document.pdf")
    .build()?;

// Usage
let content = source.fetch().await?;
```

## Creating Custom Sources

### Database Source Example

```rust
use doc_extract::{DocumentSource, DocumentContent, SourceError, SourceType, SourceMetadata};
use async_trait::async_trait;
use sqlx::{Pool, Postgres};

pub struct DatabaseSource {
    pool: Pool<Postgres>,
    document_id: String,
    table_name: String,
}

impl DatabaseSource {
    pub fn new(pool: Pool<Postgres>, document_id: String) -> Self {
        Self {
            pool,
            document_id,
            table_name: "documents".to_string(),
        }
    }
    
    pub fn with_table(mut self, table_name: String) -> Self {
        self.table_name = table_name;
        self
    }
}

#[async_trait]
impl DocumentSource for DatabaseSource {
    async fn fetch(&self) -> Result<DocumentContent, SourceError> {
        let query = format!(
            "SELECT content, content_type, filename FROM {} WHERE id = $1",
            self.table_name
        );
        
        let row = sqlx::query(&query)
            .bind(&self.document_id)
            .fetch_one(&self.pool)
            .await
            .map_err(|e| match e {
                sqlx::Error::RowNotFound => SourceError::NotFound(self.document_id.clone()),
                _ => SourceError::DatabaseError(e.to_string()),
            })?;
        
        let data: Vec<u8> = row.get("content");
        let content_type: String = row.get("content_type");
        let filename: Option<String> = row.get("filename");
        
        let mut metadata = self.metadata();
        if let Some(filename) = filename {
            metadata.set_filename(filename);
        }
        
        Ok(DocumentContent {
            data,
            content_type,
            source_metadata: metadata,
        })
    }
    
    fn source_type(&self) -> SourceType {
        SourceType::Database
    }
    
    fn validate(&self) -> Result<(), ValidationError> {
        if self.document_id.is_empty() {
            return Err(ValidationError::InvalidIdentifier);
        }
        
        // Additional validation can be added here
        Ok(())
    }
    
    fn metadata(&self) -> SourceMetadata {
        SourceMetadata::builder()
            .source_type("database")
            .identifier(&self.document_id)
            .additional_info("table", &self.table_name)
            .build()
    }
}
```

### Cloud Storage Source Example

```rust
use doc_extract::{DocumentSource, DocumentContent, SourceError, SourceType, SourceMetadata};
use async_trait::async_trait;
use rusoto_s3::{S3Client, S3, GetObjectRequest};

pub struct S3Source {
    client: S3Client,
    bucket: String,
    key: String,
    version_id: Option<String>,
}

impl S3Source {
    pub fn new(client: S3Client, bucket: String, key: String) -> Self {
        Self {
            client,
            bucket,
            key,
            version_id: None,
        }
    }
    
    pub fn with_version(mut self, version_id: String) -> Self {
        self.version_id = Some(version_id);
        self
    }
}

#[async_trait]
impl DocumentSource for S3Source {
    async fn fetch(&self) -> Result<DocumentContent, SourceError> {
        let request = GetObjectRequest {
            bucket: self.bucket.clone(),
            key: self.key.clone(),
            version_id: self.version_id.clone(),
            ..Default::default()
        };
        
        let result = self.client
            .get_object(request)
            .await
            .map_err(|e| SourceError::CloudError(e.to_string()))?;
        
        let content_type = result.content_type.unwrap_or_default();
        
        let data = if let Some(body) = result.body {
            use tokio::io::AsyncReadExt;
            let mut buffer = Vec::new();
            body.into_async_read().read_to_end(&mut buffer).await
                .map_err(|e| SourceError::ReadError(e.to_string()))?;
            buffer
        } else {
            return Err(SourceError::EmptySource);
        };
        
        Ok(DocumentContent {
            data,
            content_type,
            source_metadata: self.metadata(),
        })
    }
    
    fn source_type(&self) -> SourceType {
        SourceType::CloudStorage
    }
    
    fn validate(&self) -> Result<(), ValidationError> {
        if self.bucket.is_empty() || self.key.is_empty() {
            return Err(ValidationError::InvalidIdentifier);
        }
        
        Ok(())
    }
    
    fn metadata(&self) -> SourceMetadata {
        SourceMetadata::builder()
            .source_type("s3")
            .identifier(&format!("s3://{}/{}", self.bucket, self.key))
            .additional_info("bucket", &self.bucket)
            .additional_info("key", &self.key)
            .build()
    }
}
```

### Stream Source Example

For processing documents from streams:

```rust
use doc_extract::{DocumentSource, DocumentContent, SourceError, SourceType, SourceMetadata};
use async_trait::async_trait;
use tokio::io::{AsyncRead, AsyncReadExt};

pub struct StreamSource<R: AsyncRead + Send + Sync + Unpin> {
    reader: R,
    content_type: String,
    source_id: String,
    max_size: Option<usize>,
}

impl<R: AsyncRead + Send + Sync + Unpin> StreamSource<R> {
    pub fn new(reader: R, content_type: String, source_id: String) -> Self {
        Self {
            reader,
            content_type,
            source_id,
            max_size: None,
        }
    }
    
    pub fn with_max_size(mut self, max_size: usize) -> Self {
        self.max_size = Some(max_size);
        self
    }
}

#[async_trait]
impl<R: AsyncRead + Send + Sync + Unpin> DocumentSource for StreamSource<R> {
    async fn fetch(&self) -> Result<DocumentContent, SourceError> {
        let mut data = Vec::new();
        let mut reader = &mut self.reader;
        
        if let Some(max_size) = self.max_size {
            reader.take(max_size as u64)
                .read_to_end(&mut data)
                .await
                .map_err(|e| SourceError::ReadError(e.to_string()))?;
        } else {
            reader.read_to_end(&mut data)
                .await
                .map_err(|e| SourceError::ReadError(e.to_string()))?;
        }
        
        Ok(DocumentContent {
            data,
            content_type: self.content_type.clone(),
            source_metadata: self.metadata(),
        })
    }
    
    fn source_type(&self) -> SourceType {
        SourceType::Stream
    }
    
    fn validate(&self) -> Result<(), ValidationError> {
        if self.source_id.is_empty() {
            return Err(ValidationError::InvalidIdentifier);
        }
        
        Ok(())
    }
    
    fn metadata(&self) -> SourceMetadata {
        SourceMetadata::builder()
            .source_type("stream")
            .identifier(&self.source_id)
            .content_type(&self.content_type)
            .build()
    }
}
```

## Advanced Features

### Source Composition

Combine multiple sources:

```rust
use doc_extract::{DocumentSource, CompositeSource};

// Create a composite source that tries multiple sources
let composite = CompositeSource::builder()
    .primary(FileSource::new("./local/document.pdf")?)
    .fallback(UrlSource::new("https://backup.com/document.pdf")?)
    .fallback(DatabaseSource::new(pool, "doc_123".to_string()))
    .build();

// Fetch will try sources in order until one succeeds
let content = composite.fetch().await?;
```

### Source Caching

Add caching to expensive sources:

```rust
use doc_extract::{CachedSource, CacheConfig};

// Wrap any source with caching
let cached_source = CachedSource::new(
    UrlSource::new("https://slow-server.com/large-document.pdf")?,
    CacheConfig::builder()
        .ttl(Duration::from_hours(1))
        .max_size(100_000_000) // 100MB
        .compression(true)
        .build()
)?;

// First fetch downloads and caches
let content1 = cached_source.fetch().await?;

// Second fetch returns cached content
let content2 = cached_source.fetch().await?;
```

### Source Transformation

Transform content during fetch:

```rust
use doc_extract::{TransformingSource, Transform};

// Apply transformations to source content
let transforming_source = TransformingSource::new(
    FileSource::new("encrypted_document.pdf.enc")?,
    vec![
        Transform::Decrypt { key: "secret_key".to_string() },
        Transform::Decompress { algorithm: "gzip".to_string() },
    ]
);

let content = transforming_source.fetch().await?;
```

## Error Handling

### Source Error Types

```rust
#[derive(thiserror::Error, Debug)]
pub enum SourceError {
    #[error("Source not found: {0}")]
    NotFound(String),
    
    #[error("Access denied: {0}")]
    AccessDenied(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("HTTP error {status}: {message}")]
    HttpError { status: u16, message: String },
    
    #[error("Invalid source configuration: {0}")]
    ConfigurationError(String),
    
    #[error("Source too large: {size} bytes exceeds limit")]
    SourceTooLarge { size: u64 },
    
    #[error("Database error: {0}")]
    DatabaseError(String),
    
    #[error("Cloud storage error: {0}")]
    CloudError(String),
}
```

### Validation Error Types

```rust
#[derive(thiserror::Error, Debug)]
pub enum ValidationError {
    #[error("Source not found")]
    SourceNotFound,
    
    #[error("Source is inaccessible")]
    SourceInaccessible,
    
    #[error("Source is too large")]
    SourceTooLarge,
    
    #[error("Invalid URL format")]
    InvalidUrl,
    
    #[error("Unsupported URL scheme")]
    UnsupportedScheme,
    
    #[error("Invalid identifier")]
    InvalidIdentifier,
}
```

## Best Practices

### 1. Resource Management

```rust
impl Drop for FileSource {
    fn drop(&mut self) {
        // Clean up any temporary files or resources
        if let Some(temp_file) = &self.temp_file {
            let _ = std::fs::remove_file(temp_file);
        }
    }
}
```

### 2. Async-Safe Design

```rust
// Always use async-safe operations
#[async_trait]
impl DocumentSource for CustomSource {
    async fn fetch(&self) -> Result<DocumentContent, SourceError> {
        // Use tokio::fs instead of std::fs for file operations
        let data = tokio::fs::read(&self.path).await?;
        
        // Use async HTTP clients
        let response = reqwest::get(&self.url).await?;
        
        Ok(DocumentContent::new(data))
    }
}
```

### 3. Error Context

```rust
use anyhow::Context;

async fn fetch_with_context(&self) -> Result<DocumentContent, SourceError> {
    let data = tokio::fs::read(&self.path)
        .await
        .with_context(|| format!("Failed to read file: {}", self.path.display()))?;
    
    Ok(DocumentContent::new(data))
}
```

### 4. Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_file_source_fetch() {
        // Create test file
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let test_data = b"test document content";
        std::fs::write(temp_file.path(), test_data).unwrap();
        
        // Test source
        let source = FileSource::new(temp_file.path()).unwrap();
        let content = source.fetch().await.unwrap();
        
        assert_eq!(content.data, test_data);
    }
    
    #[tokio::test]
    async fn test_url_source_with_mock() {
        // Use wiremock or similar for HTTP testing
        let mock_server = mockito::Server::new_async().await;
        let mock = mock_server
            .mock("GET", "/document.pdf")
            .with_status(200)
            .with_header("content-type", "application/pdf")
            .with_body(b"fake pdf content")
            .create();
        
        let source = UrlSource::new(&format!("{}/document.pdf", mock_server.url())).unwrap();
        let content = source.fetch().await.unwrap();
        
        assert_eq!(content.content_type, "application/pdf");
        mock.assert();
    }
}
```

## Performance Considerations

### Memory Management

```rust
// For large files, use streaming
impl DocumentSource for LargeFileSource {
    async fn fetch(&self) -> Result<DocumentContent, SourceError> {
        if self.file_size > STREAMING_THRESHOLD {
            // Return a streaming reader instead of loading into memory
            self.create_streaming_content().await
        } else {
            self.load_into_memory().await
        }
    }
}
```

### Connection Pooling

```rust
// Share HTTP clients across instances
lazy_static! {
    static ref HTTP_CLIENT: reqwest::Client = reqwest::Client::builder()
        .pool_max_idle_per_host(10)
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap();
}
```

This comprehensive guide should help you understand and effectively implement the DocumentSource trait for any custom document source you need to integrate with the platform.