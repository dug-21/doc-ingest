//! Sandbox demonstration showing advanced security features
//!
//! This example demonstrates the enhanced sandbox implementation including:
//! - Linux namespaces for process isolation
//! - cgroups v2 for resource limiting
//! - Capability management
//! - Secure IPC communication
//! - Filesystem isolation with chroot

use neural_doc_flow_security::sandbox::{
    SandboxManager, SandboxConfig, NamespaceConfig, CapabilityConfig
};
use neural_doc_flow_core::ProcessingError;
use std::path::PathBuf;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<(), ProcessingError> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("🔒 Neural Doc Flow Sandbox Demo");
    println!("================================");

    // Demo 1: Basic sandbox execution
    demo_basic_sandbox().await?;
    
    // Demo 2: Enhanced security configuration
    demo_enhanced_security().await?;
    
    // Demo 3: Resource limiting
    demo_resource_limits().await?;
    
    // Demo 4: Timeout handling
    demo_timeout_handling().await?;
    
    // Demo 5: Cross-platform compatibility
    demo_cross_platform().await?;

    println!("✅ All sandbox demos completed successfully!");
    Ok(())
}

async fn demo_basic_sandbox() -> Result<(), ProcessingError> {
    println!("\n📦 Demo 1: Basic Sandbox Execution");
    println!("----------------------------------");
    
    let mut manager = SandboxManager::new()?;
    let start = Instant::now();
    
    let result = manager.execute_sandboxed("basic_plugin", || {
        println!("🚀 Executing inside sandbox...");
        
        // Simulate some work
        let data = vec![1, 2, 3, 4, 5];
        let sum: i32 = data.iter().sum();
        
        println!("✨ Calculated sum: {}", sum);
        Ok(sum)
    }).await?;
    
    println!("📊 Execution Results:");
    println!("  • Output: {}", result.output);
    println!("  • Execution time: {:?}", result.execution_time);
    println!("  • Memory peak: {} KB", result.resource_usage.memory_peak_bytes / 1024);
    println!("  • CPU time: {:.3}s", result.resource_usage.cpu_time_seconds);
    println!("  • Total time: {:?}", start.elapsed());
    
    Ok(())
}

async fn demo_enhanced_security() -> Result<(), ProcessingError> {
    println!("\n🛡️ Demo 2: Enhanced Security Configuration");
    println!("------------------------------------------");
    
    let config = SandboxConfig {
        memory_limit_mb: 128,
        cpu_quota_percent: 30.0,
        timeout_seconds: 10,
        allowed_paths: vec![PathBuf::from("/tmp")],
        blocked_syscalls: vec![
            "ptrace".to_string(),
            "kexec_load".to_string(),
            "delete_module".to_string(),
            "fork".to_string(),
            "clone".to_string(),
        ],
        enable_namespaces: NamespaceConfig {
            pid: true,
            net: true,  // Network isolation
            mnt: true,  // Mount isolation
            uts: true,  // Hostname isolation
            ipc: true,  // IPC isolation
            user: false, // User namespaces (requires special setup)
        },
        capabilities: CapabilityConfig {
            drop_all: true,
            allowed: vec![], // No capabilities allowed
            ambient: vec![],
        },
        network_enabled: false,
        cgroup_path: Some(PathBuf::from("/sys/fs/cgroup/neural-doc-flow-demo")),
        secure_ipc: true,
        enable_seccomp: true,
        chroot_path: Some(PathBuf::from("/tmp/neural-doc-flow-chroot")),
    };
    
    println!("📋 Security Configuration:");
    println!("  • Memory limit: {} MB", config.memory_limit_mb);
    println!("  • CPU quota: {}%", config.cpu_quota_percent);
    println!("  • Namespaces: PID={}, NET={}, MNT={}, UTS={}, IPC={}", 
             config.enable_namespaces.pid,
             config.enable_namespaces.net,
             config.enable_namespaces.mnt,
             config.enable_namespaces.uts,
             config.enable_namespaces.ipc);
    println!("  • Capabilities: Drop all = {}", config.capabilities.drop_all);
    println!("  • Seccomp: {}", config.enable_seccomp);
    println!("  • Secure IPC: {}", config.secure_ipc);
    println!("  • Chroot: {:?}", config.chroot_path);
    
    let mut manager = SandboxManager::with_config(config)?;
    
    let result = manager.execute_sandboxed("secure_plugin", || {
        println!("🔐 Executing with enhanced security...");
        
        // This would be restricted by security measures
        // - Can't access network (isolated namespace)
        // - Can't access filesystem outside allowed paths
        // - Can't use dangerous syscalls (seccomp)
        // - Limited memory and CPU (cgroups)
        
        let secure_data = "Processed in secure environment";
        Ok(secure_data.to_string())
    }).await?;
    
    println!("✅ Secure execution completed: {}", result.output);
    
    Ok(())
}

async fn demo_resource_limits() -> Result<(), ProcessingError> {
    println!("\n⚡ Demo 3: Resource Limiting");
    println!("----------------------------");
    
    let mut config = SandboxManager::default_config();
    config.memory_limit_mb = 64;  // Very restrictive
    config.cpu_quota_percent = 25.0;  // Quarter CPU
    
    let mut manager = SandboxManager::with_config(config)?;
    
    println!("📊 Testing memory allocation within limits...");
    
    let result = manager.execute_sandboxed("memory_test", || {
        // Allocate memory within limits
        let data = vec![0u8; 1024 * 1024]; // 1MB
        let checksum: u8 = data.iter().fold(0u8, |acc, &x| acc.wrapping_add(x));
        
        println!("💾 Allocated 1MB, checksum: {}", checksum);
        Ok("Memory test passed".to_string())
    }).await?;
    
    println!("📈 Resource Usage:");
    println!("  • Result: {}", result.output);
    println!("  • Memory peak: {} MB", result.resource_usage.memory_peak_bytes / (1024 * 1024));
    println!("  • CPU time: {:.3}s", result.resource_usage.cpu_time_seconds);
    
    Ok(())
}

async fn demo_timeout_handling() -> Result<(), ProcessingError> {
    println!("\n⏱️ Demo 4: Timeout Handling");
    println!("---------------------------");
    
    let mut config = SandboxManager::default_config();
    config.timeout_seconds = 2;  // Short timeout
    
    let mut manager = SandboxManager::with_config(config)?;
    
    println!("⏰ Testing timeout protection...");
    
    let result = manager.execute_sandboxed("timeout_test", || {
        println!("😴 Simulating long-running operation...");
        std::thread::sleep(std::time::Duration::from_secs(3)); // Longer than timeout
        Ok("Should not reach here".to_string())
    }).await;
    
    match result {
        Err(ProcessingError::Timeout) => {
            println!("✅ Timeout protection worked correctly!");
        }
        Ok(_) => {
            println!("❌ Timeout protection failed!");
        }
        Err(e) => {
            println!("❓ Unexpected error: {:?}", e);
        }
    }
    
    Ok(())
}

async fn demo_cross_platform() -> Result<(), ProcessingError> {
    println!("\n🌐 Demo 5: Cross-Platform Compatibility");
    println!("---------------------------------------");
    
    let mut manager = SandboxManager::new()?;
    
    println!("🖥️ Platform: {}", std::env::consts::OS);
    
    let result = manager.execute_sandboxed("platform_test", || {
        let platform_info = format!("Running on {} / {}", 
                                   std::env::consts::OS, 
                                   std::env::consts::ARCH);
        println!("📋 {}", platform_info);
        
        #[cfg(target_os = "linux")]
        println!("🐧 Full Linux namespace isolation available");
        
        #[cfg(not(target_os = "linux"))]
        println!("💻 Fallback mode (limited isolation)");
        
        Ok(platform_info)
    }).await?;
    
    println!("📊 Platform Results:");
    println!("  • Info: {}", result.output);
    println!("  • Execution time: {:?}", result.execution_time);
    
    Ok(())
}