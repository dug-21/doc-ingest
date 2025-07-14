//! Utility functions for WASM integration

use wasm_bindgen::prelude::*;
use web_sys::console;

/// Set up panic hook for better error reporting in WASM
pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Log a message to the browser console
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
    
    #[wasm_bindgen(js_namespace = console)]
    fn error(s: &str);
    
    #[wasm_bindgen(js_namespace = console)]
    fn warn(s: &str);
    
    #[wasm_bindgen(js_namespace = console)]
    fn info(s: &str);
}

/// Macro for logging to console
#[macro_export]
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// Macro for error logging
#[macro_export]
macro_rules! console_error {
    ($($t:tt)*) => (error(&format_args!($($t)*).to_string()))
}

/// Macro for warning logging
#[macro_export]
macro_rules! console_warn {
    ($($t:tt)*) => (warn(&format_args!($($t)*).to_string()))
}

/// Macro for info logging
#[macro_export]
macro_rules! console_info {
    ($($t:tt)*) => (info(&format_args!($($t)*).to_string()))
}

/// Performance timer for measuring execution time
pub struct PerformanceTimer {
    start_time: f64,
    label: String,
}

impl PerformanceTimer {
    pub fn new(label: &str) -> Self {
        let start_time = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0);
            
        console_info!("⏱️ Starting timer: {}", label);
        
        Self {
            start_time,
            label: label.to_string(),
        }
    }
    
    pub fn elapsed(&self) -> f64 {
        let current_time = web_sys::window()
            .and_then(|w| w.performance())
            .map(|p| p.now())
            .unwrap_or(0.0);
            
        current_time - self.start_time
    }
    
    pub fn finish(self) -> f64 {
        let elapsed = self.elapsed();
        console_info!("⏱️ Timer finished: {} - {:.2}ms", self.label, elapsed);
        elapsed
    }
}

/// Memory usage tracker
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
            console_info!("📊 Memory usage ({}): {} bytes", context, usage);
        }
    }
}

/// Utility for handling JavaScript errors
pub fn js_error_to_string(err: JsValue) -> String {
    if err.is_string() {
        err.as_string().unwrap_or_else(|| "Unknown string error".to_string())
    } else if let Ok(error) = err.dyn_into::<js_sys::Error>() {
        error.message().into()
    } else {
        format!("Unknown error: {:?}", err)
    }
}

/// Convert Rust Result to JS Promise-compatible Result
pub fn result_to_js<T, E>(result: Result<T, E>) -> Result<T, JsValue>
where
    T: Into<JsValue>,
    E: std::fmt::Display,
{
    result
        .map(|v| v.into())
        .map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Async wrapper for error handling
pub async fn async_result_to_js<T, E, F>(future: F) -> Result<JsValue, JsValue>
where
    T: serde::Serialize,
    E: std::fmt::Display,
    F: std::future::Future<Output = Result<T, E>>,
{
    match future.await {
        Ok(value) => serde_wasm_bindgen::to_value(&value)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e))),
        Err(e) => Err(JsValue::from_str(&e.to_string())),
    }
}