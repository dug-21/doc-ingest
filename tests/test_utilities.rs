//! Test utilities and helper functions
//!
//! This module provides common utilities for testing across the entire codebase,
//! including mock implementations, test data generators, and assertion helpers.

use std::collections::HashMap;
use std::path::PathBuf;
use tempfile::{NamedTempFile, TempDir};
use uuid::Uuid;

/// Mock PDF data generator
pub struct MockPdfGenerator;

impl MockPdfGenerator {
    /// Generate minimal valid PDF content
    pub fn minimal_pdf() -> Vec<u8> {
        b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n0000000100 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n150\n%%EOF".to_vec()
    }

    /// Generate PDF with text content
    pub fn pdf_with_text(text: &str) -> Vec<u8> {
        format!(
            "%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length {}\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n({}) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000053 00000 n \n0000000100 00000 n \n0000000179 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n300\n%%EOF",
            text.len() + 30, // Approximate stream length
            text
        ).into_bytes()
    }

    /// Generate corrupted PDF data
    pub fn corrupted_pdf() -> Vec<u8> {
        b"corrupted pdf content".to_vec()
    }

    /// Generate large PDF data for stress testing
    pub fn large_pdf(size_mb: usize) -> Vec<u8> {
        let mut content = Self::minimal_pdf();
        let extra_data = vec![b'A'; size_mb * 1024 * 1024];
        content.extend(extra_data);
        content
    }
}

/// Test file creator for various document types
pub struct TestFileCreator;

impl TestFileCreator {
    /// Create temporary PDF file
    pub fn create_pdf_file(content: Vec<u8>) -> NamedTempFile {
        let mut temp_file = NamedTempFile::new().unwrap();
        std::io::Write::write_all(temp_file.as_file_mut(), &content).unwrap();
        temp_file
    }

    /// Create temporary text file
    pub fn create_text_file(content: &str) -> NamedTempFile {
        let mut temp_file = NamedTempFile::new().unwrap();
        std::io::Write::write_all(temp_file.as_file_mut(), content.as_bytes()).unwrap();
        temp_file
    }

    /// Create temporary directory with test files
    pub fn create_test_directory() -> TempDir {
        let temp_dir = TempDir::new().unwrap();
        
        // Create sample files
        let pdf_path = temp_dir.path().join("sample.pdf");
        std::fs::write(&pdf_path, MockPdfGenerator::minimal_pdf()).unwrap();
        
        let text_path = temp_dir.path().join("sample.txt");
        std::fs::write(&text_path, "Sample text content").unwrap();
        
        temp_dir
    }

    /// Create mock model files for neural testing
    pub async fn create_mock_models(dir: &TempDir) {
        let model_names = [
            "layout_analyzer",
            "text_enhancer", 
            "table_detector",
            "confidence_scorer",
        ];
        
        for model_name in &model_names {
            let model_path = dir.path().join(format!("{}.model", model_name));
            tokio::fs::write(&model_path, b"mock model data").await.unwrap();
        }
    }
}

/// Performance testing utilities
pub struct PerformanceTestUtils;

impl PerformanceTestUtils {
    /// Measure function execution time
    pub fn time_execution<F, R>(f: F) -> (R, std::time::Duration)
    where
        F: FnOnce() -> R,
    {
        let start = std::time::Instant::now();
        let result = f();
        let elapsed = start.elapsed();
        (result, elapsed)
    }

    /// Measure async function execution time
    pub async fn time_async_execution<F, Fut, R>(f: F) -> (R, std::time::Duration)
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = R>,
    {
        let start = std::time::Instant::now();
        let result = f().await;
        let elapsed = start.elapsed();
        (result, elapsed)
    }

    /// Assert execution time is within bounds
    pub fn assert_execution_time_under(duration: std::time::Duration, max_millis: u64) {
        assert!(duration.as_millis() < max_millis as u128,
                "Execution took {}ms, expected under {}ms", 
                duration.as_millis(), max_millis);
    }

    /// Generate memory usage report
    pub fn get_memory_usage() -> usize {
        // Simple memory usage estimation
        // In a real implementation, we'd use proper memory profiling
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pdf_generator() {
        let minimal = MockPdfGenerator::minimal_pdf();
        assert!(minimal.starts_with(b"%PDF-"));
        assert!(!minimal.is_empty());

        let with_text = MockPdfGenerator::pdf_with_text("Hello World");
        assert!(with_text.starts_with(b"%PDF-"));
        assert!(with_text.len() > minimal.len());

        let corrupted = MockPdfGenerator::corrupted_pdf();
        assert!(!corrupted.starts_with(b"%PDF-"));
    }

    #[test]
    fn test_performance_utils() {
        let (result, duration) = PerformanceTestUtils::time_execution(|| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            42
        });

        assert_eq!(result, 42);
        assert!(duration.as_millis() >= 10);
        assert!(duration.as_millis() < 100); // Should be close to 10ms
    }
}