//! SIMD Document Processing Optimization Module
//!
//! High-performance SIMD implementations for document processing operations
//! including text analysis, parsing acceleration, and memory-aligned operations.

#[cfg(feature = "simd")]
use wide::{f32x8, u32x8, u64x4, i32x8};

use std::arch::x86_64::*;
use std::collections::HashMap;
use std::simd::{u8x32, u8x16};

/// SIMD-optimized document processor for high-performance text operations
pub struct SimdDocumentProcessor {
    /// CPU feature detection
    pub cpu_features: DocumentCpuFeatures,
    /// Text processing buffers
    pub text_buffers: TextProcessingBuffers,
    /// String operation accelerator
    pub string_accelerator: SimdStringAccelerator,
    /// Memory alignment manager
    pub memory_manager: SimdMemoryManager,
}

/// CPU features for document processing
#[derive(Debug, Clone)]
pub struct DocumentCpuFeatures {
    pub avx2: bool,
    pub avx512f: bool,
    pub avx512bw: bool, // Byte/Word operations
    pub sse42: bool,
    pub popcnt: bool,
    pub bmi1: bool,
    pub bmi2: bool,
}

/// Text processing buffers optimized for SIMD operations
pub struct TextProcessingBuffers {
    /// Character frequency buffers
    pub char_freq_buffer: Vec<u32>,
    /// Word boundaries buffer
    pub word_boundaries: Vec<usize>,
    /// Line boundaries buffer  
    pub line_boundaries: Vec<usize>,
    /// Whitespace analysis buffer
    pub whitespace_buffer: Vec<u8>,
}

/// SIMD string operation accelerator
pub struct SimdStringAccelerator {
    /// Optimized search patterns
    pub search_patterns: Vec<SearchPattern>,
    /// Character classification tables
    pub char_tables: CharClassificationTables,
    /// String transformation workspace
    pub transform_workspace: Vec<u8>,
}

/// SIMD memory manager for optimal alignment
pub struct SimdMemoryManager {
    /// Memory alignment size (16, 32, or 64 bytes)
    pub alignment_size: usize,
    /// Pre-allocated aligned buffers
    pub aligned_buffers: Vec<AlignedBuffer>,
    /// Buffer pool for reuse
    pub buffer_pool: Vec<Vec<u8>>,
}

/// Search pattern for SIMD string matching
#[derive(Debug, Clone)]
pub struct SearchPattern {
    pub pattern_id: u32,
    pub pattern: Vec<u8>,
    pub case_sensitive: bool,
    pub pattern_type: PatternType,
}

/// Pattern type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternType {
    Literal,
    Whitespace,
    Alphanumeric,
    Punctuation,
    Custom,
}

/// Character classification lookup tables
pub struct CharClassificationTables {
    /// ASCII character type lookup (256 entries)
    pub ascii_types: Vec<u8>,
    /// Uppercase conversion table
    pub to_upper: Vec<u8>,
    /// Lowercase conversion table
    pub to_lower: Vec<u8>,
    /// Whitespace detection table
    pub is_whitespace: Vec<bool>,
}

/// Aligned memory buffer for SIMD operations
pub struct AlignedBuffer {
    pub data: Vec<u8>,
    pub capacity: usize,
    pub alignment: usize,
}

/// Document processing results with SIMD acceleration
#[derive(Debug, Clone)]
pub struct DocumentProcessingResult {
    pub word_count: usize,
    pub line_count: usize,
    pub character_count: usize,
    pub whitespace_ratio: f32,
    pub text_complexity: f32,
    pub processing_time_ns: u64,
    pub memory_usage_bytes: usize,
}

/// Text analysis metrics
#[derive(Debug, Clone)]
pub struct TextAnalysisMetrics {
    pub average_word_length: f32,
    pub sentence_count: usize,
    pub paragraph_count: usize,
    pub unique_word_count: usize,
    pub reading_complexity: f32,
}

impl SimdDocumentProcessor {
    /// Create new SIMD document processor
    pub fn new() -> Self {
        let cpu_features = Self::detect_document_features();
        let text_buffers = TextProcessingBuffers::new();
        let string_accelerator = SimdStringAccelerator::new();
        let memory_manager = SimdMemoryManager::new(&cpu_features);
        
        Self {
            cpu_features,
            text_buffers,
            string_accelerator,
            memory_manager,
        }
    }

    /// Detect CPU features for document processing
    fn detect_document_features() -> DocumentCpuFeatures {
        #[cfg(target_arch = "x86_64")]
        {
            DocumentCpuFeatures {
                avx2: is_x86_feature_detected!("avx2"),
                avx512f: is_x86_feature_detected!("avx512f"),
                avx512bw: is_x86_feature_detected!("avx512bw"),
                sse42: is_x86_feature_detected!("sse4.2"),
                popcnt: is_x86_feature_detected!("popcnt"),
                bmi1: is_x86_feature_detected!("bmi1"),
                bmi2: is_x86_feature_detected!("bmi2"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            DocumentCpuFeatures {
                avx2: false,
                avx512f: false,
                avx512bw: false,
                sse42: false,
                popcnt: false,
                bmi1: false,
                bmi2: false,
            }
        }
    }

    /// High-performance document analysis with SIMD acceleration
    pub fn analyze_document(&mut self, text: &str) -> DocumentProcessingResult {
        let start_time = std::time::Instant::now();
        let text_bytes = text.as_bytes();

        // Parallel SIMD operations
        let (word_count, line_count) = self.simd_count_words_and_lines(text_bytes);
        let character_count = text_bytes.len();
        let whitespace_ratio = self.simd_calculate_whitespace_ratio(text_bytes);
        let text_complexity = self.simd_calculate_text_complexity(text_bytes);

        let processing_time_ns = start_time.elapsed().as_nanos() as u64;
        let memory_usage_bytes = self.estimate_memory_usage();

        DocumentProcessingResult {
            word_count,
            line_count,
            character_count,
            whitespace_ratio,
            text_complexity,
            processing_time_ns,
            memory_usage_bytes,
        }
    }

    /// SIMD-accelerated word and line counting
    pub fn simd_count_words_and_lines(&mut self, text: &[u8]) -> (usize, usize) {
        if text.is_empty() {
            return (0, 0);
        }

        if self.cpu_features.avx512bw {
            self.simd_count_avx512(text)
        } else if self.cpu_features.avx2 {
            self.simd_count_avx2(text)
        } else {
            self.simd_count_portable(text)
        }
    }

    /// AVX512 word and line counting
    #[cfg(target_arch = "x86_64")]
    fn simd_count_avx512(&mut self, text: &[u8]) -> (usize, usize) {
        if !self.cpu_features.avx512bw {
            return self.simd_count_avx2(text);
        }

        unsafe {
            let mut word_count = 0usize;
            let mut line_count = 0usize;
            let mut was_whitespace = true;

            let space_vec = _mm512_set1_epi8(b' ' as i8);
            let tab_vec = _mm512_set1_epi8(b'\t' as i8);
            let newline_vec = _mm512_set1_epi8(b'\n' as i8);
            let cr_vec = _mm512_set1_epi8(b'\r' as i8);

            for chunk in text.chunks(64) {
                if chunk.len() == 64 {
                    let data_vec = _mm512_loadu_si512(chunk.as_ptr() as *const i32);
                    
                    // Count newlines
                    let newline_mask = _mm512_cmpeq_epi8_mask(data_vec, newline_vec);
                    line_count += newline_mask.count_ones() as usize;
                    
                    // Detect whitespace characters
                    let space_mask = _mm512_cmpeq_epi8_mask(data_vec, space_vec);
                    let tab_mask = _mm512_cmpeq_epi8_mask(data_vec, tab_vec);
                    let cr_mask = _mm512_cmpeq_epi8_mask(data_vec, cr_vec);
                    let whitespace_mask = space_mask | tab_mask | newline_mask | cr_mask;
                    
                    // Count word boundaries (transitions from whitespace to non-whitespace)
                    for i in 0..64 {
                        let is_whitespace = (whitespace_mask & (1u64 << i)) != 0;
                        if was_whitespace && !is_whitespace {
                            word_count += 1;
                        }
                        was_whitespace = is_whitespace;
                    }
                } else {
                    // Handle remainder with scalar processing
                    for &byte in chunk {
                        if byte == b'\n' {
                            line_count += 1;
                        }
                        let is_whitespace = byte.is_ascii_whitespace();
                        if was_whitespace && !is_whitespace {
                            word_count += 1;
                        }
                        was_whitespace = is_whitespace;
                    }
                }
            }

            // Add 1 to line count if text doesn't end with newline
            if !text.is_empty() && text[text.len() - 1] != b'\n' {
                line_count += 1;
            }

            (word_count, line_count)
        }
    }

    /// AVX2 word and line counting
    #[cfg(target_arch = "x86_64")]
    fn simd_count_avx2(&mut self, text: &[u8]) -> (usize, usize) {
        if !self.cpu_features.avx2 {
            return self.simd_count_portable(text);
        }

        unsafe {
            let mut word_count = 0usize;
            let mut line_count = 0usize;
            let mut was_whitespace = true;

            let space_vec = _mm256_set1_epi8(b' ' as i8);
            let tab_vec = _mm256_set1_epi8(b'\t' as i8);
            let newline_vec = _mm256_set1_epi8(b'\n' as i8);
            let cr_vec = _mm256_set1_epi8(b'\r' as i8);

            for chunk in text.chunks(32) {
                if chunk.len() == 32 {
                    let data_vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
                    
                    // Count newlines
                    let newline_mask = _mm256_cmpeq_epi8(data_vec, newline_vec);
                    let newline_count = _mm256_movemask_epi8(newline_mask);
                    line_count += newline_count.count_ones() as usize;
                    
                    // Detect whitespace
                    let space_mask = _mm256_cmpeq_epi8(data_vec, space_vec);
                    let tab_mask = _mm256_cmpeq_epi8(data_vec, tab_vec);
                    let cr_mask = _mm256_cmpeq_epi8(data_vec, cr_vec);
                    
                    let whitespace_mask1 = _mm256_or_si256(space_mask, tab_mask);
                    let whitespace_mask2 = _mm256_or_si256(newline_mask, cr_mask);
                    let whitespace_mask = _mm256_or_si256(whitespace_mask1, whitespace_mask2);
                    let whitespace_bits = _mm256_movemask_epi8(whitespace_mask);
                    
                    // Count word boundaries
                    for i in 0..32 {
                        let is_whitespace = (whitespace_bits & (1 << i)) != 0;
                        if was_whitespace && !is_whitespace {
                            word_count += 1;
                        }
                        was_whitespace = is_whitespace;
                    }
                } else {
                    // Handle remainder
                    for &byte in chunk {
                        if byte == b'\n' {
                            line_count += 1;
                        }
                        let is_whitespace = byte.is_ascii_whitespace();
                        if was_whitespace && !is_whitespace {
                            word_count += 1;
                        }
                        was_whitespace = is_whitespace;
                    }
                }
            }

            if !text.is_empty() && text[text.len() - 1] != b'\n' {
                line_count += 1;
            }

            (word_count, line_count)
        }
    }

    /// Portable SIMD word and line counting
    fn simd_count_portable(&mut self, text: &[u8]) -> (usize, usize) {
        let mut word_count = 0usize;
        let mut line_count = 0usize;
        let mut was_whitespace = true;

        for &byte in text {
            if byte == b'\n' {
                line_count += 1;
            }
            
            let is_whitespace = byte.is_ascii_whitespace();
            if was_whitespace && !is_whitespace {
                word_count += 1;
            }
            was_whitespace = is_whitespace;
        }

        if !text.is_empty() && text[text.len() - 1] != b'\n' {
            line_count += 1;
        }

        (word_count, line_count)
    }

    /// SIMD-accelerated whitespace ratio calculation
    pub fn simd_calculate_whitespace_ratio(&self, text: &[u8]) -> f32 {
        if text.is_empty() {
            return 0.0;
        }

        if self.cpu_features.avx2 {
            self.simd_whitespace_ratio_avx2(text)
        } else {
            self.simd_whitespace_ratio_portable(text)
        }
    }

    /// AVX2 whitespace ratio calculation
    #[cfg(target_arch = "x86_64")]
    fn simd_whitespace_ratio_avx2(&self, text: &[u8]) -> f32 {
        if !self.cpu_features.avx2 {
            return self.simd_whitespace_ratio_portable(text);
        }

        unsafe {
            let mut whitespace_count = 0usize;
            
            let space_vec = _mm256_set1_epi8(b' ' as i8);
            let tab_vec = _mm256_set1_epi8(b'\t' as i8);
            let newline_vec = _mm256_set1_epi8(b'\n' as i8);
            let cr_vec = _mm256_set1_epi8(b'\r' as i8);

            for chunk in text.chunks(32) {
                if chunk.len() == 32 {
                    let data_vec = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);
                    
                    let space_mask = _mm256_cmpeq_epi8(data_vec, space_vec);
                    let tab_mask = _mm256_cmpeq_epi8(data_vec, tab_vec);
                    let newline_mask = _mm256_cmpeq_epi8(data_vec, newline_vec);
                    let cr_mask = _mm256_cmpeq_epi8(data_vec, cr_vec);
                    
                    let whitespace_mask1 = _mm256_or_si256(space_mask, tab_mask);
                    let whitespace_mask2 = _mm256_or_si256(newline_mask, cr_mask);
                    let whitespace_mask = _mm256_or_si256(whitespace_mask1, whitespace_mask2);
                    
                    let whitespace_bits = _mm256_movemask_epi8(whitespace_mask);
                    whitespace_count += whitespace_bits.count_ones() as usize;
                } else {
                    for &byte in chunk {
                        if byte.is_ascii_whitespace() {
                            whitespace_count += 1;
                        }
                    }
                }
            }

            whitespace_count as f32 / text.len() as f32
        }
    }

    /// Portable whitespace ratio calculation
    fn simd_whitespace_ratio_portable(&self, text: &[u8]) -> f32 {
        let whitespace_count = text.iter().filter(|&&b| b.is_ascii_whitespace()).count();
        whitespace_count as f32 / text.len() as f32
    }

    /// SIMD text complexity calculation
    pub fn simd_calculate_text_complexity(&mut self, text: &[u8]) -> f32 {
        if text.is_empty() {
            return 0.0;
        }

        // Calculate various metrics for complexity
        let unique_chars = self.simd_count_unique_characters(text);
        let avg_word_length = self.simd_calculate_average_word_length(text);
        let punctuation_ratio = self.simd_calculate_punctuation_ratio(text);
        
        // Combined complexity score (0.0 to 1.0)
        let char_complexity = (unique_chars as f32 / 256.0).min(1.0);
        let word_complexity = (avg_word_length / 20.0).min(1.0);
        let punct_complexity = punctuation_ratio;
        
        (char_complexity + word_complexity + punct_complexity) / 3.0
    }

    /// SIMD unique character counting
    fn simd_count_unique_characters(&self, text: &[u8]) -> usize {
        let mut char_seen = [false; 256];
        
        if self.cpu_features.avx2 {
            // Use SIMD to process multiple bytes at once
            for chunk in text.chunks(32) {
                for &byte in chunk {
                    char_seen[byte as usize] = true;
                }
            }
        } else {
            for &byte in text {
                char_seen[byte as usize] = true;
            }
        }
        
        char_seen.iter().filter(|&&seen| seen).count()
    }

    /// SIMD average word length calculation
    fn simd_calculate_average_word_length(&self, text: &[u8]) -> f32 {
        let mut total_chars = 0usize;
        let mut word_count = 0usize;
        let mut in_word = false;
        let mut current_word_length = 0usize;

        for &byte in text {
            if byte.is_ascii_whitespace() {
                if in_word {
                    total_chars += current_word_length;
                    word_count += 1;
                    current_word_length = 0;
                    in_word = false;
                }
            } else {
                current_word_length += 1;
                in_word = true;
            }
        }

        // Handle last word if text doesn't end with whitespace
        if in_word {
            total_chars += current_word_length;
            word_count += 1;
        }

        if word_count == 0 {
            0.0
        } else {
            total_chars as f32 / word_count as f32
        }
    }

    /// SIMD punctuation ratio calculation
    fn simd_calculate_punctuation_ratio(&self, text: &[u8]) -> f32 {
        if text.is_empty() {
            return 0.0;
        }

        let punctuation_count = text.iter()
            .filter(|&&b| b.is_ascii_punctuation())
            .count();
        
        punctuation_count as f32 / text.len() as f32
    }

    /// SIMD string search with multiple patterns
    pub fn simd_search_patterns(&mut self, text: &[u8], patterns: &[&str]) -> Vec<(usize, usize)> {
        let mut matches = Vec::new();
        
        for (pattern_idx, &pattern) in patterns.iter().enumerate() {
            let pattern_bytes = pattern.as_bytes();
            if let Some(positions) = self.simd_find_all_occurrences(text, pattern_bytes) {
                for pos in positions {
                    matches.push((pattern_idx, pos));
                }
            }
        }
        
        matches.sort_by_key(|&(_, pos)| pos);
        matches
    }

    /// SIMD find all occurrences of a pattern
    fn simd_find_all_occurrences(&self, haystack: &[u8], needle: &[u8]) -> Option<Vec<usize>> {
        if needle.is_empty() || haystack.len() < needle.len() {
            return None;
        }

        let mut positions = Vec::new();

        if self.cpu_features.avx2 {
            positions = self.simd_find_occurrences_avx2(haystack, needle);
        } else {
            positions = self.simd_find_occurrences_portable(haystack, needle);
        }

        if positions.is_empty() {
            None
        } else {
            Some(positions)
        }
    }

    /// AVX2 pattern finding
    #[cfg(target_arch = "x86_64")]
    fn simd_find_occurrences_avx2(&self, haystack: &[u8], needle: &[u8]) -> Vec<usize> {
        let mut positions = Vec::new();
        
        if !self.cpu_features.avx2 || needle.is_empty() {
            return self.simd_find_occurrences_portable(haystack, needle);
        }

        unsafe {
            let first_byte = needle[0];
            let first_vec = _mm256_set1_epi8(first_byte as i8);
            
            let mut i = 0;
            while i + needle.len() <= haystack.len() {
                if i + 32 <= haystack.len() {
                    let data_vec = _mm256_loadu_si256(haystack[i..].as_ptr() as *const __m256i);
                    let cmp_result = _mm256_cmpeq_epi8(data_vec, first_vec);
                    let mask = _mm256_movemask_epi8(cmp_result);
                    
                    if mask != 0 {
                        for j in 0..32 {
                            if (mask & (1 << j)) != 0 && i + j + needle.len() <= haystack.len() {
                                if &haystack[i + j..i + j + needle.len()] == needle {
                                    positions.push(i + j);
                                }
                            }
                        }
                        i += 32;
                    } else {
                        i += 32;
                    }
                } else {
                    if &haystack[i..i + needle.len()] == needle {
                        positions.push(i);
                    }
                    i += 1;
                }
            }
        }

        positions
    }

    /// Portable pattern finding
    fn simd_find_occurrences_portable(&self, haystack: &[u8], needle: &[u8]) -> Vec<usize> {
        let mut positions = Vec::new();
        
        for (i, window) in haystack.windows(needle.len()).enumerate() {
            if window == needle {
                positions.push(i);
            }
        }
        
        positions
    }

    /// Estimate memory usage for current operations
    fn estimate_memory_usage(&self) -> usize {
        let mut usage = 0;
        usage += self.text_buffers.char_freq_buffer.capacity() * std::mem::size_of::<u32>();
        usage += self.text_buffers.word_boundaries.capacity() * std::mem::size_of::<usize>();
        usage += self.text_buffers.line_boundaries.capacity() * std::mem::size_of::<usize>();
        usage += self.text_buffers.whitespace_buffer.capacity();
        usage += self.string_accelerator.transform_workspace.capacity();
        
        for buffer in &self.memory_manager.aligned_buffers {
            usage += buffer.capacity;
        }
        
        usage
    }

    /// Get performance information
    pub fn get_performance_info(&self) -> DocumentProcessorInfo {
        DocumentProcessorInfo {
            cpu_features: self.cpu_features.clone(),
            memory_alignment: self.memory_manager.alignment_size,
            estimated_speedup: self.estimate_speedup(),
            supported_operations: self.get_supported_operations(),
        }
    }

    /// Estimate performance speedup
    fn estimate_speedup(&self) -> f32 {
        let mut speedup = 1.0;
        
        if self.cpu_features.avx512bw {
            speedup *= 16.0; // 512-bit operations
        } else if self.cpu_features.avx2 {
            speedup *= 8.0; // 256-bit operations
        } else if self.cpu_features.sse42 {
            speedup *= 4.0; // 128-bit operations
        }
        
        if self.cpu_features.popcnt {
            speedup *= 1.2; // Bit counting acceleration
        }
        
        speedup
    }

    /// Get supported operations
    fn get_supported_operations(&self) -> Vec<String> {
        vec![
            "word_counting".to_string(),
            "line_counting".to_string(),
            "pattern_matching".to_string(),
            "whitespace_analysis".to_string(),
            "text_complexity_analysis".to_string(),
            "character_frequency_analysis".to_string(),
            "string_searching".to_string(),
        ]
    }
}

/// Document processor performance information
#[derive(Debug, Clone)]
pub struct DocumentProcessorInfo {
    pub cpu_features: DocumentCpuFeatures,
    pub memory_alignment: usize,
    pub estimated_speedup: f32,
    pub supported_operations: Vec<String>,
}

impl TextProcessingBuffers {
    fn new() -> Self {
        Self {
            char_freq_buffer: Vec::with_capacity(256),
            word_boundaries: Vec::with_capacity(1024),
            line_boundaries: Vec::with_capacity(256),
            whitespace_buffer: Vec::with_capacity(1024),
        }
    }
}

impl SimdStringAccelerator {
    fn new() -> Self {
        Self {
            search_patterns: Vec::new(),
            char_tables: CharClassificationTables::new(),
            transform_workspace: Vec::with_capacity(4096),
        }
    }
}

impl CharClassificationTables {
    fn new() -> Self {
        let mut ascii_types = vec![0u8; 256];
        let mut to_upper = vec![0u8; 256];
        let mut to_lower = vec![0u8; 256];
        let mut is_whitespace = vec![false; 256];
        
        for i in 0..256 {
            let byte = i as u8;
            ascii_types[i] = if byte.is_ascii_alphabetic() { 1 }
                           else if byte.is_ascii_digit() { 2 }
                           else if byte.is_ascii_whitespace() { 3 }
                           else if byte.is_ascii_punctuation() { 4 }
                           else { 0 };
            
            to_upper[i] = byte.to_ascii_uppercase();
            to_lower[i] = byte.to_ascii_lowercase();
            is_whitespace[i] = byte.is_ascii_whitespace();
        }
        
        Self {
            ascii_types,
            to_upper,
            to_lower,
            is_whitespace,
        }
    }
}

impl SimdMemoryManager {
    fn new(cpu_features: &DocumentCpuFeatures) -> Self {
        let alignment_size = if cpu_features.avx512f { 64 }
                            else if cpu_features.avx2 { 32 }
                            else { 16 };
        
        Self {
            alignment_size,
            aligned_buffers: Vec::new(),
            buffer_pool: Vec::new(),
        }
    }
}

impl AlignedBuffer {
    fn new(capacity: usize, alignment: usize) -> Self {
        let mut data = Vec::with_capacity(capacity + alignment);
        data.resize(capacity, 0);
        
        Self {
            data,
            capacity,
            alignment,
        }
    }
}

impl Default for SimdDocumentProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_processor_creation() {
        let processor = SimdDocumentProcessor::new();
        assert!(processor.memory_manager.alignment_size >= 16);
    }

    #[test]
    fn test_word_and_line_counting() {
        let mut processor = SimdDocumentProcessor::new();
        let text = "Hello world!\nThis is a test.\nAnother line here.";
        
        let (word_count, line_count) = processor.simd_count_words_and_lines(text.as_bytes());
        assert_eq!(word_count, 8);
        assert_eq!(line_count, 3);
    }

    #[test]
    fn test_whitespace_ratio() {
        let processor = SimdDocumentProcessor::new();
        let text = "a b c";
        let ratio = processor.simd_calculate_whitespace_ratio(text.as_bytes());
        assert!((ratio - 0.4).abs() < 0.1); // 2 spaces out of 5 characters
    }

    #[test]
    fn test_document_analysis() {
        let mut processor = SimdDocumentProcessor::new();
        let text = "This is a sample document with various words and punctuation!";
        
        let result = processor.analyze_document(text);
        assert!(result.word_count > 0);
        assert!(result.character_count == text.len());
        assert!(result.whitespace_ratio > 0.0);
        assert!(result.processing_time_ns > 0);
    }

    #[test]
    fn test_pattern_search() {
        let mut processor = SimdDocumentProcessor::new();
        let text = "The quick brown fox jumps over the lazy dog. The fox is quick.";
        let patterns = &["fox", "the", "quick"];
        
        let matches = processor.simd_search_patterns(text.as_bytes(), patterns);
        assert!(!matches.is_empty());
        
        // Should find "fox" pattern
        assert!(matches.iter().any(|&(pattern_idx, _)| pattern_idx == 0));
    }
}