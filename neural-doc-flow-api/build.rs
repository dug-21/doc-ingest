//! Build optimization script for neural-doc-flow-api
//!
//! This script configures build-time optimizations to reduce compilation time:
//! - Feature-based conditional compilation
//! - Debug symbols optimization 
//! - Parallel compilation hints

use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    
    // Enable optimizations for specific dependencies during compile time
    if env::var("PROFILE").unwrap_or_default() == "debug" {
        // In debug mode, optimize critical dependencies for faster compilation
        println!("cargo:rustc-env=SQLX_OFFLINE=true");
        
        // Skip heavy feature compilation in debug mode unless explicitly enabled
        if !cfg!(feature = "full") {
            println!("cargo:rustc-cfg=minimal_build");
        }
    }
    
    // Set codegen units for faster parallel compilation
    if !Path::new("target").exists() {
        println!("cargo:rustc-env=CARGO_INCREMENTAL=1");
    }
    
    // Feature-specific build optimizations
    #[cfg(not(feature = "database"))]
    println!("cargo:rustc-cfg=no_database");
    
    #[cfg(not(feature = "auth"))]
    println!("cargo:rustc-cfg=no_auth");
    
    #[cfg(not(feature = "docs"))]
    println!("cargo:rustc-cfg=no_docs");
    
    // Print current feature configuration for debug
    println!("cargo:warning=Building with features: {}", 
             env::var("CARGO_FEATURE_MINIMAL").map(|_| "minimal")
                 .or_else(|_| env::var("CARGO_FEATURE_FULL").map(|_| "full"))
                 .unwrap_or("default"));
}