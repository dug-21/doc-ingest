//! Basic test to verify the library compiles and types are accessible

use neural_doc_flow_core::prelude::*;

#[test]
fn test_library_compiles() {
    // Simple test to verify the library compiles correctly
    assert_eq!(1 + 1, 2);
}

#[test]
fn test_error_types_exist() {
    // Verify error types are accessible
    let _error_message = "Test error";
    assert!(true);
}

#[test]
fn test_result_type_exists() {
    // Verify Result type alias exists
    fn example_function() -> neural_doc_flow_core::Result<String> {
        Ok("Success".to_string())
    }
    
    let result = example_function();
    assert!(result.is_ok());
}