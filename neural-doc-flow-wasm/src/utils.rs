//! Pure Rust utility functions for WASM integration
//! 
//! Architecture Compliance: Zero JavaScript dependencies

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

/// Set up panic hook for better error reporting in WASM
pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Pure Rust logging to stdout (no JavaScript console dependencies)
pub fn log_message(level: &str, message: &str) {
    println!("[{}] {}", level, message);
}

/// Log functions for different levels
pub fn log_info(message: &str) {
    log_message("INFO", message);
}

pub fn log_error(message: &str) {
    log_message("ERROR", message);
}

pub fn log_warn(message: &str) {
    log_message("WARN", message);
}

pub fn log_debug(message: &str) {
    log_message("DEBUG", message);
}

/// Performance timer for measuring execution time (pure Rust)
pub struct PerformanceTimer {
    start_time: std::time::Instant,
    label: String,
}

impl PerformanceTimer {
    pub fn new(label: &str) -> Self {
        log_info(&format!("⏱️ Starting timer: {}", label));
        
        Self {
            start_time: std::time::Instant::now(),
            label: label.to_string(),
        }
    }
    
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }
    
    pub fn finish(self) -> std::time::Duration {
        let elapsed = self.elapsed();
        log_info(&format!("⏱️ Timer finished: {} - {:.2}ms", self.label, elapsed.as_millis()));
        elapsed
    }
}

/// Memory usage tracker for WASM
pub struct MemoryTracker;

impl MemoryTracker {
    pub fn current_usage() -> Option<u32> {
        #[cfg(target_arch = "wasm32")]
        {
            // WASM memory is in pages of 64KB
            let pages = core::arch::wasm32::memory_size(0);
            Some(pages * 64 * 1024)
        }
        #[cfg(not(target_arch = "wasm32"))]
        None
    }
    
    pub fn log_usage(context: &str) {
        if let Some(usage) = Self::current_usage() {
            log_info(&format!("📊 Memory usage ({}): {} bytes", context, usage));
        }
    }
}

/// Utility for converting between C strings and Rust strings
pub fn c_string_to_rust(c_str: *const c_char) -> Option<String> {
    if c_str.is_null() {
        return None;
    }
    
    unsafe {
        CStr::from_ptr(c_str)
            .to_str()
            .ok()
            .map(|s| s.to_string())
    }
}

pub fn rust_string_to_c(rust_str: &str) -> *mut c_char {
    match CString::new(rust_str) {
        Ok(c_string) => c_string.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Convert Rust Vec<String> to C-style string array
pub fn rust_vec_to_c_array(vec: Vec<String>) -> (*mut *mut c_char, usize) {
    let len = vec.len();
    let mut c_array = Vec::with_capacity(len);
    
    for string in vec {
        c_array.push(rust_string_to_c(&string));
    }
    
    let ptr = c_array.as_mut_ptr();
    std::mem::forget(c_array);
    
    (ptr, len)
}

/// Free C-style string array
pub unsafe fn free_c_array(ptr: *mut *mut c_char, len: usize) {
    if !ptr.is_null() {
        for i in 0..len {
            let string_ptr = *ptr.add(i);
            if !string_ptr.is_null() {
                let _ = CString::from_raw(string_ptr);
            }
        }
        let _ = Vec::from_raw_parts(ptr, len, len);
    }
}

/// Error handling utilities
pub fn handle_error<T, E>(result: Result<T, E>) -> Option<T>
where
    E: std::fmt::Display,
{
    match result {
        Ok(value) => Some(value),
        Err(error) => {
            log_error(&format!("Operation failed: {}", error));
            None
        }
    }
}

/// Async wrapper for error handling (pure Rust)
pub async fn async_handle_error<T, E, F>(future: F) -> Option<T>
where
    F: std::future::Future<Output = Result<T, E>>,
    E: std::fmt::Display,
{
    match future.await {
        Ok(value) => Some(value),
        Err(error) => {
            log_error(&format!("Async operation failed: {}", error));
            None
        }
    }
}

/// Pure Rust WASM-compatible random number generation
pub fn generate_random_u32() -> u32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos()
        .hash(&mut hasher);
    
    hasher.finish() as u32
}

/// Generate a simple UUID-like string without external dependencies
pub fn generate_simple_uuid() -> String {
    format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        generate_random_u32(),
        generate_random_u32() as u16,
        generate_random_u32() as u16,
        generate_random_u32() as u16,
        ((generate_random_u32() as u64) << 32) | (generate_random_u32() as u64)
    )
}

/// File size formatting utilities
pub fn format_file_size(size: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
    let mut size_f = size as f64;
    let mut unit_idx = 0;
    
    while size_f >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size_f /= 1024.0;
        unit_idx += 1;
    }
    
    format!("{:.2} {}", size_f, UNITS[unit_idx])
}

/// Time formatting utilities
pub fn format_duration(duration: std::time::Duration) -> String {
    let total_ms = duration.as_millis();
    
    if total_ms < 1000 {
        format!("{}ms", total_ms)
    } else if total_ms < 60000 {
        format!("{:.2}s", total_ms as f64 / 1000.0)
    } else {
        let minutes = total_ms / 60000;
        let seconds = (total_ms % 60000) as f64 / 1000.0;
        format!("{}m {:.2}s", minutes, seconds)
    }
}

/// Pure Rust base64 encoding (simple implementation)
pub fn simple_base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();
    
    for chunk in data.chunks(3) {
        let mut buf = [0u8; 3];
        for (i, &byte) in chunk.iter().enumerate() {
            buf[i] = byte;
        }
        
        let b = ((buf[0] as u32) << 16) | ((buf[1] as u32) << 8) | (buf[2] as u32);
        
        result.push(ALPHABET[((b >> 18) & 63) as usize] as char);
        result.push(ALPHABET[((b >> 12) & 63) as usize] as char);
        
        if chunk.len() > 1 {
            result.push(ALPHABET[((b >> 6) & 63) as usize] as char);
        } else {
            result.push('=');
        }
        
        if chunk.len() > 2 {
            result.push(ALPHABET[(b & 63) as usize] as char);
        } else {
            result.push('=');
        }
    }
    
    result
}

/// Configuration validation
pub fn validate_config_value(value: &str, min: f64, max: f64) -> Result<f64, String> {
    match value.parse::<f64>() {
        Ok(num) => {
            if num >= min && num <= max {
                Ok(num)
            } else {
                Err(format!("Value {} is out of range [{}, {}]", num, min, max))
            }
        }
        Err(_) => Err(format!("Invalid number format: {}", value)),
    }
}

/// System information utilities
pub fn get_wasm_features() -> Vec<String> {
    let mut features = Vec::new();
    
    #[cfg(target_arch = "wasm32")]
    {
        features.push("wasm32".to_string());
        
        #[cfg(target_feature = "simd128")]
        features.push("simd128".to_string());
        
        #[cfg(target_feature = "atomics")]
        features.push("atomics".to_string());
        
        #[cfg(target_feature = "bulk-memory")]
        features.push("bulk-memory".to_string());
        
        #[cfg(target_feature = "mutable-globals")]
        features.push("mutable-globals".to_string());
    }
    
    features
}

/// Build information
pub fn get_build_info() -> std::collections::HashMap<String, String> {
    let mut info = std::collections::HashMap::new();
    
    info.insert("version".to_string(), env!("CARGO_PKG_VERSION").to_string());
    info.insert("name".to_string(), env!("CARGO_PKG_NAME").to_string());
    
    // Use runtime target detection instead of compile-time
    #[cfg(target_arch = "wasm32")]
    info.insert("target".to_string(), "wasm32-unknown-unknown".to_string());
    
    #[cfg(not(target_arch = "wasm32"))]
    info.insert("target".to_string(), std::env::consts::ARCH.to_string());
    
    info.insert("profile".to_string(), if cfg!(debug_assertions) { "debug" } else { "release" }.to_string());
    
    #[cfg(target_arch = "wasm32")]
    info.insert("arch".to_string(), "wasm32".to_string());
    
    #[cfg(not(target_arch = "wasm32"))]
    info.insert("arch".to_string(), "native".to_string());
    
    info
}

/// Checksum calculation (simple implementation)
pub fn calculate_checksum(data: &[u8]) -> u32 {
    let mut checksum = 0u32;
    for &byte in data {
        checksum = checksum.wrapping_add(byte as u32);
    }
    checksum
}

/// Data validation utilities
pub fn validate_utf8(data: &[u8]) -> bool {
    std::str::from_utf8(data).is_ok()
}

pub fn sanitize_filename(filename: &str) -> String {
    filename
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '.' || *c == '_' || *c == '-')
        .collect()
}

/// Thread-safe counter for statistics
use std::sync::atomic::{AtomicU64, Ordering};

pub struct AtomicCounter {
    value: AtomicU64,
}

impl AtomicCounter {
    pub fn new() -> Self {
        Self {
            value: AtomicU64::new(0),
        }
    }
    
    pub fn increment(&self) -> u64 {
        self.value.fetch_add(1, Ordering::SeqCst)
    }
    
    pub fn get(&self) -> u64 {
        self.value.load(Ordering::SeqCst)
    }
    
    pub fn reset(&self) {
        self.value.store(0, Ordering::SeqCst);
    }
}

/// Global counters for tracking operations
pub static DOCUMENTS_PROCESSED: AtomicCounter = AtomicCounter { value: AtomicU64::new(0) };
pub static ERRORS_ENCOUNTERED: AtomicCounter = AtomicCounter { value: AtomicU64::new(0) };
pub static BYTES_PROCESSED: AtomicU64 = AtomicU64::new(0);