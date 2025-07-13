//! Core traits for Neural Document Flow
//!
//! This module defines the fundamental traits that enable the plugin
//! architecture and neural processing capabilities.

pub mod source;
pub mod processor;
pub mod output;
pub mod neural;

// Re-export all traits for convenience
pub use source::*;
pub use processor::*;
pub use output::*;
pub use neural::*;