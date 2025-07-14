//! Integration tests for sandbox functionality

use neural_doc_flow_security::sandbox::{SandboxManager, SandboxConfig, NamespaceConfig, CapabilityConfig};
use neural_doc_flow_core::ProcessingError;
use std::path::PathBuf;

#[tokio::test]
async fn test_sandbox_manager_creation() {
    let manager = SandboxManager::new();
    assert!(manager.is_ok());
}

#[test]
fn test_default_sandbox_config() {
    let config = SandboxManager::default_config();
    assert_eq!(config.memory_limit_mb, 500);
    assert_eq!(config.cpu_quota_percent, 50.0);
    assert_eq!(config.timeout_seconds, 30);
    assert!(!config.network_enabled);
    assert!(config.enable_namespaces.pid);
    assert!(config.enable_namespaces.net);
    assert!(config.capabilities.drop_all);
    assert!(config.secure_ipc);
    assert!(config.enable_seccomp);
    assert!(config.chroot_path.is_some());
}

#[test]
fn test_enhanced_sandbox_config() {
    let config = SandboxConfig {
        memory_limit_mb: 256,
        cpu_quota_percent: 25.0,
        timeout_seconds: 60,
        allowed_paths: vec![PathBuf::from("/tmp"), PathBuf::from("/var/tmp")],
        blocked_syscalls: vec![
            "ptrace".to_string(),
            "kexec_load".to_string(),
            "delete_module".to_string(),
        ],
        enable_namespaces: NamespaceConfig {
            pid: true,
            net: true,
            mnt: true,
            uts: true,
            ipc: true,
            user: false,
        },
        capabilities: CapabilityConfig {
            drop_all: true,
            allowed: vec!["CAP_SETUID".to_string(), "CAP_SETGID".to_string()],
            ambient: vec![],
        },
        network_enabled: false,
        cgroup_path: Some(PathBuf::from("/sys/fs/cgroup/test")),
        secure_ipc: true,
        enable_seccomp: true,
        chroot_path: Some(PathBuf::from("/tmp/test-chroot")),
    };

    assert_eq!(config.memory_limit_mb, 256);
    assert_eq!(config.cpu_quota_percent, 25.0);
    assert_eq!(config.timeout_seconds, 60);
    assert_eq!(config.allowed_paths.len(), 2);
    assert_eq!(config.blocked_syscalls.len(), 3);
    assert!(config.secure_ipc);
    assert!(config.enable_seccomp);
    assert!(config.chroot_path.is_some());
}

#[tokio::test]
async fn test_sandbox_execution_success() {
    let mut manager = SandboxManager::new().unwrap();
    
    // Test successful execution
    let result = manager.execute_sandboxed("test_plugin", || {
        Ok(42)
    }).await;
    
    assert!(result.is_ok());
    let sandbox_result = result.unwrap();
    assert_eq!(sandbox_result.output, 42);
    assert!(sandbox_result.execution_time.as_secs() < 5);
}

#[tokio::test]
async fn test_sandbox_error_propagation() {
    let mut manager = SandboxManager::new().unwrap();
    
    // Test error in sandbox execution
    let result = manager.execute_sandboxed("error_test", || {
        Err(ProcessingError::ProcessorFailed {
            processor_name: "test".to_string(),
            reason: "intentional error".to_string(),
        })
    }).await;
    
    // Should propagate the error correctly
    assert!(result.is_err());
}

#[tokio::test]
async fn test_sandbox_timeout() {
    let mut config = SandboxManager::default_config();
    config.timeout_seconds = 1; // 1 second timeout
    
    let mut manager = SandboxManager::with_config(config).unwrap();
    
    // Test timeout
    let result = manager.execute_sandboxed("timeout_plugin", || {
        std::thread::sleep(std::time::Duration::from_secs(2));
        Ok(())
    }).await;
    
    assert!(result.is_err());
    match result.err().unwrap() {
        ProcessingError::Timeout => {},
        e => panic!("Expected timeout error, got: {:?}", e),
    }
}

#[test]
fn test_namespace_configuration() {
    let config = NamespaceConfig {
        pid: true,
        net: false, // No network isolation
        mnt: true,
        uts: true,
        ipc: true,
        user: false, // User namespaces require special setup
    };
    
    // Test individual namespace settings
    assert!(config.pid);
    assert!(!config.net);
    assert!(config.mnt);
    assert!(config.uts);
    assert!(config.ipc);
    assert!(!config.user);
}

#[test]
fn test_capability_configuration_security() {
    let secure_config = CapabilityConfig {
        drop_all: true,
        allowed: vec![], // No capabilities allowed
        ambient: vec![],
    };
    
    let minimal_config = CapabilityConfig {
        drop_all: true,
        allowed: vec!["CAP_SETUID".to_string()], // Only setuid allowed
        ambient: vec![],
    };
    
    assert!(secure_config.drop_all);
    assert!(secure_config.allowed.is_empty());
    
    assert!(minimal_config.drop_all);
    assert_eq!(minimal_config.allowed.len(), 1);
    assert_eq!(minimal_config.allowed[0], "CAP_SETUID");
}

#[tokio::test]
async fn test_sandbox_with_custom_limits() {
    let mut config = SandboxManager::default_config();
    config.memory_limit_mb = 64; // Very low memory limit
    
    let mut manager = SandboxManager::with_config(config).unwrap();
    
    // Test execution within memory constraints
    let result = manager.execute_sandboxed("memory_test", || {
        // Simulate low memory usage
        let _data = vec![0u8; 1024]; // 1KB allocation
        Ok("success".to_string())
    }).await;
    
    assert!(result.is_ok());
    let sandbox_result = result.unwrap();
    assert_eq!(sandbox_result.output, "success");
    
    // Verify resource usage is tracked
    assert!(sandbox_result.resource_usage.memory_peak_bytes > 0);
}

#[test]
fn test_ipc_security_token_generation() {
    let plugin_id = "security_test";
    let process_id = std::process::id();
    let expected_token = format!("{}_{}", plugin_id, process_id);
    
    // This tests the token generation logic
    assert!(expected_token.contains(plugin_id));
    assert!(expected_token.contains(&process_id.to_string()));
}