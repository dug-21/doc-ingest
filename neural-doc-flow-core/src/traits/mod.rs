//! Core traits for the neural document flow framework

pub mod source;
pub mod processor;
pub mod output;
pub mod neural;

// Re-export all traits
pub use source::*;
pub use processor::*;
pub use output::*;
pub use neural::*;