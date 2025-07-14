//! Utility functions

/// Utility functions placeholder
pub struct Utils;

impl Utils {
    /// Generate content hash
    pub fn generate_content_hash(content: &[u8]) -> String {
        format!("{:x}", md5::compute(content))
    }
}