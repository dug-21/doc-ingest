/// DAA Neural Coordination Library - Optimized Build
/// High-performance distributed agent architecture with feature-gated compilation

#[cfg(feature = "full")]
mod full_coordination {
    include!("lib.rs");
}

#[cfg(all(not(feature = "full"), feature = "minimal"))]
mod minimal_coordination {
    include!("src/minimal_lib.rs");
    
    // Re-export minimal types with full names for compatibility
    pub use MinimalCoordinationSystem as DaaCoordinationSystem;
    pub use initialize_minimal_coordination as initialize_daa_neural_system;
}

// Conditional exports based on feature flags
#[cfg(feature = "full")]
pub use full_coordination::*;

#[cfg(all(not(feature = "full"), feature = "minimal"))]
pub use minimal_coordination::*;

// Feature detection for runtime
pub const COORDINATION_FEATURES: &[&str] = &[
    #[cfg(feature = "messaging")]
    "messaging",
    #[cfg(feature = "fault-tolerance")]
    "fault-tolerance",
    #[cfg(feature = "monitoring")]
    "monitoring",
    #[cfg(feature = "analytics")]
    "analytics",
    #[cfg(feature = "performance")]
    "performance",
];

/// Get enabled coordination features
pub fn enabled_features() -> &'static [&'static str] {
    COORDINATION_FEATURES
}

/// Check if feature is enabled
pub fn has_feature(feature: &str) -> bool {
    COORDINATION_FEATURES.contains(&feature)
}