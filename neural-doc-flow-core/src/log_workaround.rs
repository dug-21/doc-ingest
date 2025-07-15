// Workaround for Rust 1.88.0 parser bug with log macros
// These macros provide a drop-in replacement for standard log macros
// using println!/eprintln! as the backend

/// Custom trace! macro that uses println! to avoid parser bug
#[macro_export]
macro_rules! trace {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        println!("[TRACE] {}", format!($($arg)*));
    };
}

/// Custom debug! macro that uses println! to avoid parser bug
#[macro_export]
macro_rules! debug {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        println!("[DEBUG] {}", format!($($arg)*));
    };
}

/// Custom info! macro that uses println! to avoid parser bug
#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {
        println!("[INFO] {}", format!($($arg)*));
    };
}

/// Custom warn! macro that uses eprintln! to avoid parser bug
#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => {
        eprintln!("[WARN] {}", format!($($arg)*));
    };
}

/// Custom error! macro that uses eprintln! to avoid parser bug
#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {
        eprintln!("[ERROR] {}", format!($($arg)*));
    };
}

// Re-export macros for convenient importing
pub use crate::{trace, debug, info, warn, error};