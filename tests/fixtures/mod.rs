//! Test fixtures and sample data
//!
//! This module provides comprehensive test fixtures including sample documents,
//! expected outputs, and edge case test data for thorough testing.

use std::collections::HashMap;

/// Sample PDF content for testing
pub mod pdf_samples {
    use super::*;

    /// Minimal valid PDF
    pub const MINIMAL_PDF: &[u8] = b"%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
>>
endobj
xref
0 4
0000000000 65535 f 
0000000010 00000 n 
0000000053 00000 n 
0000000100 00000 n 
trailer
<<
/Size 4
/Root 1 0 R
>>
startxref
150
%%EOF";

    /// PDF with text content
    pub fn pdf_with_text(text: &str) -> Vec<u8> {
        format!(
            "%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
>>
endobj
4 0 obj
<<
/Length {}
>>
stream
BT
/F1 12 Tf
72 720 Td
({}) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000010 00000 n 
0000000053 00000 n 
0000000100 00000 n 
0000000179 00000 n 
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
300
%%EOF",
            text.len() + 30,
            text
        ).into_bytes()
    }

    /// Corrupted PDF for error testing
    pub const CORRUPTED_PDF: &[u8] = b"This is not a valid PDF file";

    /// Empty PDF (just header)
    pub const EMPTY_PDF: &[u8] = b"%PDF-1.4";

    /// PDF with special characters and unicode
    pub fn unicode_pdf() -> Vec<u8> {
        pdf_with_text("Unicode test: 🔥✨💯 中文 العربية Русский")
    }

    /// Large PDF for performance testing
    pub fn large_pdf(size_mb: usize) -> Vec<u8> {
        let mut content = MINIMAL_PDF.to_vec();
        let extra_data = vec![b'A'; size_mb * 1024 * 1024];
        content.extend(extra_data);
        content
    }
}

/// Edge case test data
pub mod edge_cases {
    /// Very long filename
    pub fn long_filename() -> String {
        "a".repeat(1000) + ".pdf"
    }

    /// Filename with special characters
    pub const SPECIAL_FILENAME: &str = "test-file_with@special#chars$.pdf";

    /// Empty content
    pub const EMPTY_CONTENT: &[u8] = b"";

    /// Single byte content
    pub const SINGLE_BYTE: &[u8] = b"A";

    /// Maximum size content (for testing limits)
    pub fn max_size_content() -> Vec<u8> {
        vec![b'X'; 100 * 1024 * 1024] // 100MB
    }

    /// Content with null bytes
    pub const NULL_BYTES: &[u8] = b"\x00\x00\x00test\x00\x00";

    /// Binary data that might be confused for text
    pub const BINARY_DATA: &[u8] = &[
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, // PNG header
        0x00, 0x00, 0x00, 0x0D, 0x49, 0x48, 0x44, 0x52,
    ];

    /// Text with various encodings
    pub const UTF8_TEXT: &str = "Hello 世界 🌍";
    pub const LATIN1_TEXT: &[u8] = b"Caf\xe9"; // "Café" in Latin-1
    
    /// Very long text line
    pub fn long_text_line() -> String {
        "This is a very long line of text that goes on and on ".repeat(1000)
    }

    /// Text with many newlines
    pub fn multiline_text() -> String {
        (0..1000).map(|i| format!("Line {}", i)).collect::<Vec<_>>().join("\n")
    }
}

/// Performance test data
pub mod performance_data {
    /// Small document set for quick tests
    pub fn small_document_set() -> Vec<Vec<u8>> {
        (0..10).map(|i| {
            super::pdf_samples::pdf_with_text(&format!("Document {}", i))
        }).collect()
    }

    /// Medium document set for load testing
    pub fn medium_document_set() -> Vec<Vec<u8>> {
        (0..100).map(|i| {
            super::pdf_samples::pdf_with_text(&format!("Medium document {} with more content", i))
        }).collect()
    }

    /// Large document set for stress testing
    pub fn large_document_set() -> Vec<Vec<u8>> {
        (0..1000).map(|i| {
            super::pdf_samples::pdf_with_text(&format!("Large document {} with extensive content for stress testing", i))
        }).collect()
    }

    /// Mixed size document set
    pub fn mixed_size_document_set() -> Vec<Vec<u8>> {
        let mut docs = Vec::new();
        
        // Small docs
        for i in 0..50 {
            docs.push(super::pdf_samples::pdf_with_text(&format!("Small {}", i)));
        }
        
        // Medium docs
        for i in 0..30 {
            let content = format!("Medium document {} with more content {}", i, "x".repeat(1000));
            docs.push(super::pdf_samples::pdf_with_text(&content));
        }
        
        // Large docs
        for i in 0..10 {
            let content = format!("Large document {} with extensive content {}", i, "x".repeat(10000));
            docs.push(super::pdf_samples::pdf_with_text(&content));
        }
        
        docs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pdf_samples() {
        assert!(pdf_samples::MINIMAL_PDF.starts_with(b"%PDF-"));
        assert!(pdf_samples::CORRUPTED_PDF.len() > 0);
        assert!(!pdf_samples::CORRUPTED_PDF.starts_with(b"%PDF-"));
        
        let text_pdf = pdf_samples::pdf_with_text("Hello World");
        assert!(text_pdf.starts_with(b"%PDF-"));
        assert!(text_pdf.len() > pdf_samples::MINIMAL_PDF.len());
    }

    #[test]
    fn test_edge_cases() {
        assert_eq!(edge_cases::long_filename().len(), 1004); // 1000 + ".pdf"
        assert!(edge_cases::SPECIAL_FILENAME.contains('@'));
        assert_eq!(edge_cases::EMPTY_CONTENT.len(), 0);
        assert_eq!(edge_cases::SINGLE_BYTE.len(), 1);
        assert!(edge_cases::NULL_BYTES.contains(&0));
    }

    #[test]
    fn test_performance_data() {
        let small = performance_data::small_document_set();
        assert_eq!(small.len(), 10);
        
        let medium = performance_data::medium_document_set();
        assert_eq!(medium.len(), 100);
        
        let mixed = performance_data::mixed_size_document_set();
        assert_eq!(mixed.len(), 90); // 50 + 30 + 10
    }
}