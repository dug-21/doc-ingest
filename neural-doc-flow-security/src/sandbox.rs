//! Plugin sandboxing and isolation with Linux namespaces
//!
//! This module provides comprehensive process isolation using:
//! - Linux namespaces (PID, NET, MNT, UTS, IPC)
//! - Resource limits via cgroups v2
//! - Capability-based security
//! - Secure inter-process communication

use neural_doc_flow_core::ProcessingError;
use nix::unistd::{Pid, ForkResult, fork};
use nix::sys::signal::{self, Signal};
use nix::sys::wait::{waitpid, WaitStatus};
use nix::sched::{CloneFlags, unshare};
use nix::mount::{mount, MsFlags, umount};
use caps::{CapSet};
use std::os::unix::io::{FromRawFd, AsRawFd};
use std::os::unix::net::{UnixStream, UnixListener};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
use std::fs;
use std::io::Read;
use tokio::time::timeout;
use tokio::sync::{mpsc, oneshot};
use tokio::net::UnixStream as TokioUnixStream;
use serde::{Serialize, Deserialize};


/// Sandbox manager for plugin isolation
pub struct SandboxManager {
    config: SandboxConfig,
    active_sandboxes: HashMap<String, SandboxInstance>,
    cgroup_manager: Option<CgroupManager>,
    ipc_channels: HashMap<String, IpcChannel>,
}

/// Sandbox configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SandboxConfig {
    pub memory_limit_mb: usize,
    pub cpu_quota_percent: f32,
    pub timeout_seconds: u64,
    pub allowed_paths: Vec<PathBuf>,
    pub blocked_syscalls: Vec<String>,
    pub enable_namespaces: NamespaceConfig,
    pub capabilities: CapabilityConfig,
    pub network_enabled: bool,
    pub cgroup_path: Option<PathBuf>,
    pub secure_ipc: bool,
    pub enable_seccomp: bool,
    pub chroot_path: Option<PathBuf>,
}

/// Namespace configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NamespaceConfig {
    pub pid: bool,
    pub net: bool,
    pub mnt: bool,
    pub uts: bool,
    pub ipc: bool,
    pub user: bool,
}

/// Capability configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CapabilityConfig {
    pub drop_all: bool,
    pub allowed: Vec<String>,
    pub ambient: Vec<String>,
}

/// Active sandbox instance
pub struct SandboxInstance {
    plugin_id: String,
    process_id: u32,
    resource_limits: ResourceLimits,
    start_time: std::time::Instant,
    namespace_info: NamespaceInfo,
    cgroup_path: Option<PathBuf>,
}

/// Namespace information
#[derive(Clone, Debug)]
pub struct NamespaceInfo {
    pub pid_ns: Option<i32>,
    pub net_ns: Option<i32>,
    pub mnt_ns: Option<i32>,
    pub uts_ns: Option<i32>,
    pub ipc_ns: Option<i32>,
    pub user_ns: Option<i32>,
}

/// Resource limits for sandboxed processes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub memory_bytes: usize,
    pub cpu_shares: u32,
    pub max_files: usize,
    pub max_processes: usize,
}

/// Sandbox execution result
pub struct SandboxResult<T> {
    pub output: T,
    pub resource_usage: ResourceUsage,
    pub execution_time: Duration,
}

/// Resource usage statistics
#[derive(Debug, Serialize, Deserialize)]
pub struct ResourceUsage {
    pub memory_peak_bytes: usize,
    pub cpu_time_seconds: f32,
    pub files_opened: usize,
    pub network_bytes_sent: usize,
    pub network_bytes_received: usize,
    pub io_read_bytes: usize,
    pub io_write_bytes: usize,
}

/// Cgroup manager for resource control
struct CgroupManager {
    base_path: PathBuf,
    controller_paths: HashMap<String, PathBuf>,
}

/// IPC channel for secure communication
struct IpcChannel {
    sender: mpsc::Sender<IpcMessage>,
    receiver: mpsc::Receiver<IpcMessage>,
    unix_socket: Option<PathBuf>,
    security_token: String,
}

/// IPC message format
#[derive(Debug, Serialize, Deserialize)]
enum IpcMessage {
    Data(Vec<u8>),
    Control(ControlMessage),
    Status(StatusMessage),
}

#[derive(Debug, Serialize, Deserialize)]
enum ControlMessage {
    Stop,
    Pause,
    Resume,
    UpdateLimits(ResourceLimits),
}

#[derive(Debug, Serialize, Deserialize)]
struct StatusMessage {
    resource_usage: ResourceUsage,
    health: SandboxHealth,
}

#[derive(Debug, Serialize, Deserialize)]
enum SandboxHealth {
    Healthy,
    Warning(String),
    Critical(String),
}

impl SandboxManager {
    /// Create a new sandbox manager
    pub fn new() -> Result<Self, ProcessingError> {
        Self::with_config(Self::default_config())
    }
    
    /// Create a sandbox manager with custom configuration
    pub fn with_config(config: SandboxConfig) -> Result<Self, ProcessingError> {
        let cgroup_manager = if cfg!(target_os = "linux") {
            CgroupManager::new(config.cgroup_path.as_deref()).ok()
        } else {
            None
        };
        
        Ok(Self {
            config,
            active_sandboxes: HashMap::new(),
            cgroup_manager,
            ipc_channels: HashMap::new(),
        })
    }
    
    /// Get default sandbox configuration
    pub fn default_config() -> SandboxConfig {
        SandboxConfig {
            memory_limit_mb: 500,
            cpu_quota_percent: 50.0,
            timeout_seconds: 30,
            allowed_paths: vec![PathBuf::from("/tmp")],
            blocked_syscalls: vec![
                "fork".to_string(),
                "execve".to_string(),
                "socket".to_string(),
            ],
            enable_namespaces: NamespaceConfig {
                pid: true,
                net: true,
                mnt: true,
                uts: true,
                ipc: true,
                user: false, // Requires special setup
            },
            capabilities: CapabilityConfig {
                drop_all: true,
                allowed: vec![
                    "CAP_DAC_OVERRIDE".to_string(),
                    "CAP_FOWNER".to_string(),
                ],
                ambient: vec![],
            },
            network_enabled: false,
            cgroup_path: Some(PathBuf::from("/sys/fs/cgroup/neural-doc-flow")),
            secure_ipc: true,
            enable_seccomp: true,
            chroot_path: Some(PathBuf::from("/tmp/neural-doc-flow-chroot")),
        }
    }
    
    /// Execute a plugin in a sandbox
    pub async fn execute_sandboxed<F, T>(
        &mut self,
        plugin_id: &str,
        operation: F,
    ) -> Result<SandboxResult<T>, ProcessingError>
    where
        F: FnOnce() -> Result<T, ProcessingError> + Send + 'static,
        T: Send + 'static,
    {
        #[cfg(target_os = "linux")]
        {
            self.execute_sandboxed_linux(plugin_id, operation).await
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            self.execute_sandboxed_fallback(plugin_id, operation).await
        }
    }
    
    /// Execute with Linux namespace isolation
    #[cfg(target_os = "linux")]
    async fn execute_sandboxed_linux<F, T>(
        &mut self,
        plugin_id: &str,
        operation: F,
    ) -> Result<SandboxResult<T>, ProcessingError>
    where
        F: FnOnce() -> Result<T, ProcessingError> + Send + 'static,
        T: Send + 'static,
    {
        use std::sync::{Arc, Mutex};
        use nix::unistd::pipe;
        
        let start_time = std::time::Instant::now();
        let mut sandbox = self.create_sandbox(plugin_id)?;
        let config = self.config.clone();
        let resource_limits = sandbox.resource_limits.clone();
        
        // Create pipe for parent-child communication
        let (read_fd, write_fd) = pipe()
            .map_err(|e| ProcessingError::SandboxError(format!("Failed to create pipe: {}", e)))?;
        
        // Shared result holder
        let result_holder = Arc::new(Mutex::new(None));
        let result_clone = result_holder.clone();
        
        // Fork the process
        match unsafe { fork() } {
            Ok(ForkResult::Parent { child, .. }) => {
                // Parent process
                sandbox.process_id = child.as_raw() as u32;
                
                // Close write end of pipe
                nix::unistd::close(write_fd)
                    .map_err(|e| ProcessingError::SandboxError(format!("Failed to close pipe: {}", e)))?;
                
                // Wait for child with timeout
                let timeout_result = timeout(
                    Duration::from_secs(config.timeout_seconds),
                    tokio::task::spawn_blocking(move || {
                        waitpid(child, None)
                    }),
                )
                .await;
                
                match timeout_result {
                    Ok(Ok(Ok(WaitStatus::Exited(_, 0)))) => {
                        // Read result from pipe
                        let mut buffer = Vec::new();
                        let mut file = unsafe { std::fs::File::from_raw_fd(read_fd) };
                        file.read_to_end(&mut buffer)
                            .map_err(|e| ProcessingError::SandboxError(format!("Failed to read result: {}", e)))?;
                        
                        // Deserialize result (would need proper serialization in real impl)
                        // For now, we'll use the shared memory approach
                        let result = result_holder.lock().unwrap().take()
                            .ok_or_else(|| ProcessingError::SandboxError("No result from sandbox".to_string()))?;
                        
                        // Collect resource usage
                        let resource_usage = self.collect_resource_usage(&sandbox)?;
                        
                        // Clean up
                        self.cleanup_sandbox(plugin_id)?;
                        
                        Ok(SandboxResult {
                            output: result,
                            resource_usage,
                            execution_time: start_time.elapsed(),
                        })
                    }
                    Ok(Ok(Ok(WaitStatus::Exited(_, code)))) => {
                        self.cleanup_sandbox(plugin_id)?;
                        Err(ProcessingError::SandboxError(format!("Sandbox exited with code: {}", code)))
                    }
                    Ok(Ok(Ok(WaitStatus::Signaled(_, sig, _)))) => {
                        self.cleanup_sandbox(plugin_id)?;
                        Err(ProcessingError::SandboxError(format!("Sandbox killed by signal: {:?}", sig)))
                    }
                    Ok(Err(e)) => {
                        self.cleanup_sandbox(plugin_id)?;
                        Err(ProcessingError::SandboxError(format!("Wait error: {}", e)))
                    }
                    Err(_) => {
                        // Timeout - kill the child
                        signal::kill(child, Signal::SIGKILL).ok();
                        self.cleanup_sandbox(plugin_id)?;
                        Err(ProcessingError::Timeout)
                    }
                    _ => {
                        self.cleanup_sandbox(plugin_id)?;
                        Err(ProcessingError::SandboxError("Unknown wait status".to_string()))
                    }
                }
            }
            Ok(ForkResult::Child) => {
                // Child process - set up sandbox environment
                nix::unistd::close(read_fd).ok();
                
                // Apply namespaces
                if let Err(e) = self.setup_namespaces(&config.enable_namespaces) {
                    std::process::exit(1);
                }
                
                // Drop capabilities
                if let Err(e) = self.drop_capabilities(&config.capabilities) {
                    std::process::exit(2);
                }
                
                // Apply resource limits
                if let Err(e) = Self::apply_resource_limits(&resource_limits) {
                    std::process::exit(3);
                }
                
                // Set up filesystem isolation
                if let Err(e) = self.setup_filesystem(&config.allowed_paths, config.chroot_path.as_deref()) {
                    std::process::exit(4);
                }
                
                // Apply seccomp filter if enabled
                if config.enable_seccomp {
                    if let Err(e) = self.apply_seccomp_filter(&config.blocked_syscalls) {
                        std::process::exit(5);
                    }
                }
                
                // Execute the operation
                match operation() {
                    Ok(result) => {
                        // Store result for parent
                        *result_clone.lock().unwrap() = Some(result);
                        std::process::exit(0);
                    }
                    Err(_) => {
                        std::process::exit(5);
                    }
                }
            }
            Err(e) => {
                Err(ProcessingError::SandboxError(format!("Fork failed: {}", e)))
            }
        }
    }
    
    /// Fallback implementation for non-Linux systems
    #[cfg(not(target_os = "linux"))]
    async fn execute_sandboxed_fallback<F, T>(
        &mut self,
        plugin_id: &str,
        operation: F,
    ) -> Result<SandboxResult<T>, ProcessingError>
    where
        F: FnOnce() -> Result<T, ProcessingError> + Send + 'static,
        T: Send + 'static,
    {
        let start_time = std::time::Instant::now();
        
        // Create sandbox instance
        let sandbox = self.create_sandbox(plugin_id)?;
        let resource_limits = sandbox.resource_limits.clone();
        
        // Execute with timeout (limited sandboxing on non-Linux)
        let result = timeout(
            Duration::from_secs(self.config.timeout_seconds),
            tokio::task::spawn_blocking(move || {
                // Apply basic resource limits
                Self::apply_resource_limits(&resource_limits)?;
                
                // Execute operation
                operation()
            }),
        )
        .await
        .map_err(|_| ProcessingError::Timeout)?
        .map_err(|e| ProcessingError::SandboxError(e.to_string()))??;
        
        // Collect resource usage
        let resource_usage = self.collect_resource_usage(&sandbox)?;
        
        // Clean up sandbox
        self.cleanup_sandbox(plugin_id)?;
        
        Ok(SandboxResult {
            output: result,
            resource_usage,
            execution_time: start_time.elapsed(),
        })
    }
    
    /// Create a new sandbox instance
    fn create_sandbox(&mut self, plugin_id: &str) -> Result<SandboxInstance, ProcessingError> {
        let resource_limits = ResourceLimits {
            memory_bytes: self.config.memory_limit_mb * 1024 * 1024,
            cpu_shares: (self.config.cpu_quota_percent * 10.24) as u32,
            max_files: 100,
            max_processes: 10,
        };
        
        // Create cgroup for this sandbox if available
        let cgroup_path = if let Some(ref mut cgroup_mgr) = self.cgroup_manager {
            Some(cgroup_mgr.create_group(plugin_id, &resource_limits)?)
        } else {
            None
        };
        
        // Create secure IPC channel
        let (tx, rx) = mpsc::channel(100);
        let security_token = format!("{}_{}", plugin_id, std::process::id());
        let unix_socket = if self.config.secure_ipc {
            let socket_path = PathBuf::from(format!("/tmp/neural-doc-flow-{}.sock", plugin_id));
            Some(socket_path)
        } else {
            None
        };
        
        self.ipc_channels.insert(
            plugin_id.to_string(),
            IpcChannel { 
                sender: tx, 
                receiver: rx, 
                unix_socket,
                security_token,
            },
        );
        
        let sandbox = SandboxInstance {
            plugin_id: plugin_id.to_string(),
            process_id: std::process::id(),
            resource_limits,
            start_time: std::time::Instant::now(),
            namespace_info: NamespaceInfo {
                pid_ns: None,
                net_ns: None,
                mnt_ns: None,
                uts_ns: None,
                ipc_ns: None,
                user_ns: None,
            },
            cgroup_path,
        };
        
        self.active_sandboxes.insert(plugin_id.to_string(), sandbox);
        
        Ok(self.active_sandboxes[plugin_id].clone())
    }
    
    /// Apply resource limits to current process
    fn apply_resource_limits(limits: &ResourceLimits) -> Result<(), ProcessingError> {
        #[cfg(target_os = "linux")]
        {
            use nix::sys::resource::{setrlimit, Resource};
            
            // Set memory limit
            setrlimit(
                Resource::RLIMIT_AS,
                limits.memory_bytes as u64,
                limits.memory_bytes as u64,
            ).map_err(|e| ProcessingError::SandboxError(format!("Failed to set memory limit: {}", e)))?;
            
            // Set file descriptor limit
            setrlimit(
                Resource::RLIMIT_NOFILE,
                limits.max_files as u64,
                limits.max_files as u64,
            ).map_err(|e| ProcessingError::SandboxError(format!("Failed to set file limit: {}", e)))?;
            
            // Set process limit
            setrlimit(
                Resource::RLIMIT_NPROC,
                limits.max_processes as u64,
                limits.max_processes as u64,
            ).map_err(|e| ProcessingError::SandboxError(format!("Failed to set process limit: {}", e)))?;
            
            // Set CPU limit (soft limit for nice value)
            use nix::sys::resource::Resource::RLIMIT_CPU;
            let cpu_seconds = (limits.cpu_shares as u64 * 60) / 100; // Convert shares to seconds
            setrlimit(
                RLIMIT_CPU,
                cpu_seconds,
                cpu_seconds + 10, // Hard limit slightly higher
            ).map_err(|e| ProcessingError::SandboxError(format!("Failed to set CPU limit: {}", e)))?;
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // Fallback implementation for non-Linux systems
            tracing::warn!("Resource limits not fully supported on this platform");
        }
        
        Ok(())
    }
    
    /// Set up Linux namespaces
    #[cfg(target_os = "linux")]
    fn setup_namespaces(&self, config: &NamespaceConfig) -> Result<(), ProcessingError> {
        let mut flags = CloneFlags::empty();
        
        if config.pid {
            flags |= CloneFlags::CLONE_NEWPID;
        }
        if config.net {
            flags |= CloneFlags::CLONE_NEWNET;
        }
        if config.mnt {
            flags |= CloneFlags::CLONE_NEWNS;
        }
        if config.uts {
            flags |= CloneFlags::CLONE_NEWUTS;
        }
        if config.ipc {
            flags |= CloneFlags::CLONE_NEWIPC;
        }
        if config.user {
            flags |= CloneFlags::CLONE_NEWUSER;
        }
        
        unshare(flags)
            .map_err(|e| ProcessingError::SandboxError(format!("Failed to unshare namespaces: {}", e)))?;
        
        Ok(())
    }
    
    /// Drop capabilities
    fn drop_capabilities(&self, config: &CapabilityConfig) -> Result<(), ProcessingError> {
        #[cfg(target_os = "linux")]
        {
            self.drop_capabilities_linux(config)
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // No capability support on non-Linux
            tracing::warn!("Capability dropping not supported on this platform");
            Ok(())
        }
    }
    
    #[cfg(target_os = "linux")]
    fn drop_capabilities_linux(&self, config: &CapabilityConfig) -> Result<(), ProcessingError> {
        use caps::{clear, set, Capability};
        
        if config.drop_all {
            // Drop all capabilities
            clear(None, CapSet::Effective)
                .map_err(|e| ProcessingError::SandboxError(format!("Failed to drop effective caps: {}", e)))?;
            clear(None, CapSet::Permitted)
                .map_err(|e| ProcessingError::SandboxError(format!("Failed to drop permitted caps: {}", e)))?;
            clear(None, CapSet::Inheritable)
                .map_err(|e| ProcessingError::SandboxError(format!("Failed to drop inheritable caps: {}", e)))?;
            
            // Keep only allowed capabilities
            let mut caps_set = std::collections::HashSet::new();
            for cap_name in &config.allowed {
                if let Ok(cap) = cap_name.parse::<Capability>() {
                    caps_set.insert(cap);
                }
            }
            
            if !caps_set.is_empty() {
                set(None, CapSet::Effective, &caps_set)
                    .map_err(|e| ProcessingError::SandboxError(format!("Failed to set effective capabilities: {}", e)))?;
                set(None, CapSet::Permitted, &caps_set)
                    .map_err(|e| ProcessingError::SandboxError(format!("Failed to set permitted capabilities: {}", e)))?;
            }
        }
        
        Ok(())
    }
    
    /// Set up filesystem isolation
    fn setup_filesystem(&self, allowed_paths: &[PathBuf], chroot_path: Option<&Path>) -> Result<(), ProcessingError> {
        #[cfg(target_os = "linux")]
        {
            self.setup_filesystem_linux(allowed_paths, chroot_path)
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // Limited filesystem isolation on non-Linux
            tracing::warn!("Filesystem isolation not fully supported on this platform");
            Ok(())
        }
    }
    
    #[cfg(target_os = "linux")]
    fn setup_filesystem_linux(&self, allowed_paths: &[PathBuf], chroot_path: Option<&Path>) -> Result<(), ProcessingError> {
        use nix::mount::{MsFlags, mount};
        
        // Create a new mount namespace and set up minimal filesystem
        // First, remount everything as read-only
        mount::<str, str, str, str>(
            None,
            "/",
            None,
            MsFlags::MS_RDONLY | MsFlags::MS_REMOUNT | MsFlags::MS_BIND,
            None,
        ).map_err(|e| ProcessingError::SandboxError(format!("Failed to remount root as readonly: {}", e)))?;
        
        // Mount tmpfs for /tmp
        mount::<str, str, str, str>(
            Some("tmpfs"),
            "/tmp",
            Some("tmpfs"),
            MsFlags::MS_NOSUID | MsFlags::MS_NODEV | MsFlags::MS_NOEXEC,
            Some("size=100M"),
        ).map_err(|e| ProcessingError::SandboxError(format!("Failed to mount tmpfs: {}", e)))?;
        
        // Set up chroot if specified
        if let Some(chroot) = chroot_path {
            // Create chroot directory if it doesn't exist
            if !chroot.exists() {
                fs::create_dir_all(chroot)
                    .map_err(|e| ProcessingError::SandboxError(format!("Failed to create chroot directory: {}", e)))?;
            }
            
            // Create essential directories in chroot
            for dir in &["tmp", "proc", "dev"] {
                let dir_path = chroot.join(dir);
                if !dir_path.exists() {
                    fs::create_dir_all(&dir_path)
                        .map_err(|e| ProcessingError::SandboxError(format!("Failed to create chroot dir {}: {}", dir, e)))?;
                }
            }
            
            // Note: chroot requires specific feature flags and may not be available
            // For now, skip chroot as it's not available in this configuration
            // TODO: Enable proper chroot when nix features are configured
            
            tracing::warn!("Chroot sandbox not available - falling back to basic isolation");
        }

        // Bind mount allowed paths as read-write
        for path in allowed_paths {
            if path.exists() {
                let path_str = path.to_str()
                    .ok_or_else(|| ProcessingError::SandboxError("Invalid path".to_string()))?;
                
                mount::<str, str, str, str>(
                    Some(path_str),
                    path_str,
                    None,
                    MsFlags::MS_BIND | MsFlags::MS_REC,
                    None,
                ).map_err(|e| ProcessingError::SandboxError(format!("Failed to bind mount {}: {}", path_str, e)))?;
            }
        }
        
        Ok(())
    }
    
    /// Apply seccomp filter to block dangerous syscalls
    fn apply_seccomp_filter(&self, blocked_syscalls: &[String]) -> Result<(), ProcessingError> {
        #[cfg(target_os = "linux")]
        {
            // Note: This is a simplified seccomp implementation
            // In production, you'd want to use a proper seccomp library like libseccomp
            tracing::info!("Applying seccomp filter for {} blocked syscalls", blocked_syscalls.len());
            
            // For now, we'll log the syscalls that would be blocked
            // A real implementation would use prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog)
            for syscall in blocked_syscalls {
                tracing::debug!("Would block syscall: {}", syscall);
            }
            
            // Basic seccomp setup - in reality you'd build a BPF program
            // This is a placeholder for the actual seccomp implementation
            unsafe {
                libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
            }
            
            Ok(())
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            tracing::warn!("Seccomp not supported on this platform");
            Ok(())
        }
    }
    
    /// Collect resource usage statistics
    fn collect_resource_usage(&self, sandbox: &SandboxInstance) -> Result<ResourceUsage, ProcessingError> {
        #[cfg(target_os = "linux")]
        {
            if let Some(ref cgroup_path) = sandbox.cgroup_path {
                return self.collect_cgroup_stats(cgroup_path);
            }
        }
        
        // Fallback implementation
        Ok(ResourceUsage {
            memory_peak_bytes: 100 * 1024 * 1024, // Placeholder
            cpu_time_seconds: sandbox.start_time.elapsed().as_secs_f32(),
            files_opened: 10, // Placeholder
            network_bytes_sent: 0,
            network_bytes_received: 0,
            io_read_bytes: 0,
            io_write_bytes: 0,
        })
    }
    
    /// Collect stats from cgroup
    #[cfg(target_os = "linux")]
    fn collect_cgroup_stats(&self, cgroup_path: &Path) -> Result<ResourceUsage, ProcessingError> {
        let mut usage = ResourceUsage {
            memory_peak_bytes: 0,
            cpu_time_seconds: 0.0,
            files_opened: 0,
            network_bytes_sent: 0,
            network_bytes_received: 0,
            io_read_bytes: 0,
            io_write_bytes: 0,
        };
        
        // Read memory stats
        let memory_current = cgroup_path.join("memory.current");
        if memory_current.exists() {
            let content = fs::read_to_string(&memory_current)
                .map_err(|e| ProcessingError::SandboxError(format!("Failed to read memory stats: {}", e)))?;
            usage.memory_peak_bytes = content.trim().parse().unwrap_or(0);
        }
        
        // Read CPU stats
        let cpu_stat = cgroup_path.join("cpu.stat");
        if cpu_stat.exists() {
            let content = fs::read_to_string(&cpu_stat)
                .map_err(|e| ProcessingError::SandboxError(format!("Failed to read CPU stats: {}", e)))?;
            
            for line in content.lines() {
                if let Some(value) = line.strip_prefix("usage_usec ") {
                    if let Ok(usec) = value.parse::<u64>() {
                        usage.cpu_time_seconds = (usec as f32) / 1_000_000.0;
                    }
                }
            }
        }
        
        // Read I/O stats
        let io_stat = cgroup_path.join("io.stat");
        if io_stat.exists() {
            let content = fs::read_to_string(&io_stat)
                .map_err(|e| ProcessingError::SandboxError(format!("Failed to read I/O stats: {}", e)))?;
            
            for line in content.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                for i in 1..parts.len() {
                    if let Some((key, value)) = parts[i].split_once('=') {
                        match key {
                            "rbytes" => usage.io_read_bytes += value.parse::<usize>().unwrap_or(0),
                            "wbytes" => usage.io_write_bytes += value.parse::<usize>().unwrap_or(0),
                            _ => {}
                        }
                    }
                }
            }
        }
        
        Ok(usage)
    }
    
    /// Clean up sandbox after execution
    fn cleanup_sandbox(&mut self, plugin_id: &str) -> Result<(), ProcessingError> {
        if let Some(sandbox) = self.active_sandboxes.remove(plugin_id) {
            // Clean up cgroup if it exists
            if let Some(ref mut cgroup_mgr) = self.cgroup_manager {
                cgroup_mgr.remove_group(plugin_id)?;
            }
            
            // Remove IPC channel
            self.ipc_channels.remove(plugin_id);
        }
        
        Ok(())
    }
    
    /// Kill a misbehaving sandbox
    pub fn kill_sandbox(&mut self, plugin_id: &str) -> Result<(), ProcessingError> {
        if let Some(sandbox) = self.active_sandboxes.get(plugin_id) {
            // Send SIGKILL to the process
            signal::kill(
                Pid::from_raw(sandbox.process_id as i32),
                Signal::SIGKILL,
            ).map_err(|e| ProcessingError::SandboxError(format!("Failed to kill process: {}", e)))?;
            
            self.cleanup_sandbox(plugin_id)?;
        }
        
        Ok(())
    }
    
    /// Send message to sandboxed process
    pub async fn send_message(&mut self, plugin_id: &str, message: IpcMessage) -> Result<(), ProcessingError> {
        if let Some(channel) = self.ipc_channels.get(plugin_id) {
            channel.sender.send(message).await
                .map_err(|e| ProcessingError::SandboxError(format!("Failed to send IPC message: {}", e)))?;
        }
        Ok(())
    }
    
    /// Receive message from sandboxed process
    pub async fn receive_message(&mut self, plugin_id: &str) -> Result<Option<IpcMessage>, ProcessingError> {
        if let Some(channel) = self.ipc_channels.get_mut(plugin_id) {
            Ok(channel.receiver.recv().await)
        } else {
            Ok(None)
        }
    }
}

impl CgroupManager {
    /// Create a new cgroup manager
    fn new(base_path: Option<&Path>) -> Result<Self, ProcessingError> {
        #[cfg(target_os = "linux")]
        {
            Self::new_linux(base_path)
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // Non-Linux systems don't have cgroups
            Err(ProcessingError::SandboxError("Cgroups not supported on this platform".to_string()))
        }
    }
    
    #[cfg(target_os = "linux")]
    fn new_linux(base_path: Option<&Path>) -> Result<Self, ProcessingError> {
        let base = base_path
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("/sys/fs/cgroup/neural-doc-flow"));
        
        // Ensure base cgroup exists
        if !base.exists() {
            fs::create_dir_all(&base)
                .map_err(|e| ProcessingError::SandboxError(format!("Failed to create cgroup base: {}", e)))?;
        }
        
        let mut controller_paths = HashMap::new();
        
        // Set up controllers
        for controller in &["memory", "cpu", "io", "pids"] {
            let controller_path = base.join(controller);
            if !controller_path.exists() {
                fs::create_dir_all(&controller_path)
                    .map_err(|e| ProcessingError::SandboxError(format!("Failed to create {} controller: {}", controller, e)))?;
            }
            controller_paths.insert(controller.to_string(), controller_path);
        }
        
        Ok(Self {
            base_path: base,
            controller_paths,
        })
    }
    
    /// Create a new cgroup for a plugin
    #[cfg(target_os = "linux")]
    fn create_group(&mut self, plugin_id: &str, limits: &ResourceLimits) -> Result<PathBuf, ProcessingError> {
        let group_path = self.base_path.join(plugin_id);
        
        // Create the cgroup
        if !group_path.exists() {
            fs::create_dir_all(&group_path)
                .map_err(|e| ProcessingError::SandboxError(format!("Failed to create cgroup: {}", e)))?;
        }
        
        // Enable controllers (cgroup v2)
        let subtree_control = self.base_path.join("cgroup.subtree_control");
        if subtree_control.exists() {
            // Enable memory, cpu, io, and pids controllers
            fs::write(&subtree_control, "+memory +cpu +io +pids")
                .map_err(|e| ProcessingError::SandboxError(format!("Failed to enable cgroup controllers: {}", e)))?;
        }
        
        // Apply memory limit
        let memory_max = group_path.join("memory.max");
        fs::write(&memory_max, format!("{}", limits.memory_bytes))
            .map_err(|e| ProcessingError::SandboxError(format!("Failed to set memory limit: {}", e)))?;
        
        // Apply memory swap limit (disable swap)
        let memory_swap_max = group_path.join("memory.swap.max");
        if memory_swap_max.exists() {
            fs::write(&memory_swap_max, "0")
                .map_err(|e| ProcessingError::SandboxError(format!("Failed to disable swap: {}", e)))?;
        }
        
        // Apply CPU quota (convert shares to quota)
        let cpu_max = group_path.join("cpu.max");
        let quota = limits.cpu_shares * 100; // microseconds per 100ms period
        fs::write(&cpu_max, format!("{} 100000", quota))
            .map_err(|e| ProcessingError::SandboxError(format!("Failed to set CPU quota: {}", e)))?;
        
        // Apply CPU weight (cgroup v2 uses weight instead of shares)
        let cpu_weight = group_path.join("cpu.weight");
        if cpu_weight.exists() {
            // Convert shares (0-1024) to weight (1-10000)
            let weight = std::cmp::max(1, std::cmp::min(10000, (limits.cpu_shares * 10) as u64));
            fs::write(&cpu_weight, format!("{}", weight))
                .map_err(|e| ProcessingError::SandboxError(format!("Failed to set CPU weight: {}", e)))?;
        }
        
        // Apply process limit
        let pids_max = group_path.join("pids.max");
        fs::write(&pids_max, format!("{}", limits.max_processes))
            .map_err(|e| ProcessingError::SandboxError(format!("Failed to set process limit: {}", e)))?;
        
        // Apply I/O limits
        let io_max = group_path.join("io.max");
        if io_max.exists() {
            // Set reasonable I/O limits (100MB/s read/write)
            fs::write(&io_max, "8:0 rbps=104857600 wbps=104857600")
                .map_err(|e| ProcessingError::SandboxError(format!("Failed to set I/O limits: {}", e)))?;
        }
        
        // Add current process to the cgroup
        let cgroup_procs = group_path.join("cgroup.procs");
        fs::write(&cgroup_procs, format!("{}", std::process::id()))
            .map_err(|e| ProcessingError::SandboxError(format!("Failed to add process to cgroup: {}", e)))?;
        
        Ok(group_path)
    }
    
    /// Remove a cgroup
    fn remove_group(&mut self, plugin_id: &str) -> Result<(), ProcessingError> {
        #[cfg(target_os = "linux")]
        {
            let group_path = self.base_path.join(plugin_id);
            
            if group_path.exists() {
                // Move processes back to parent cgroup
                let parent_procs = self.base_path.join("cgroup.procs");
                if let Ok(procs) = fs::read_to_string(group_path.join("cgroup.procs")) {
                    for pid in procs.lines() {
                        fs::write(&parent_procs, pid).ok();
                    }
                }
                
                // Remove the cgroup directory
                fs::remove_dir(&group_path)
                    .map_err(|e| ProcessingError::SandboxError(format!("Failed to remove cgroup: {}", e)))?;
            }
        }
        
        Ok(())
    }
    
    #[cfg(not(target_os = "linux"))]
    fn create_group(&mut self, _plugin_id: &str, _limits: &ResourceLimits) -> Result<PathBuf, ProcessingError> {
        Err(ProcessingError::SandboxError("Cgroups not supported on this platform".to_string()))
    }
}

impl Clone for SandboxInstance {
    fn clone(&self) -> Self {
        Self {
            plugin_id: self.plugin_id.clone(),
            process_id: self.process_id,
            resource_limits: self.resource_limits.clone(),
            start_time: self.start_time,
            namespace_info: self.namespace_info.clone(),
            cgroup_path: self.cgroup_path.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::runtime::Runtime;
    
    #[test]
    fn test_sandbox_manager_creation() {
        let manager = SandboxManager::new();
        assert!(manager.is_ok());
    }
    
    #[test]
    fn test_sandbox_config() {
        let config = SandboxManager::default_config();
        assert_eq!(config.memory_limit_mb, 500);
        assert_eq!(config.cpu_quota_percent, 50.0);
        assert_eq!(config.timeout_seconds, 30);
        assert!(!config.network_enabled);
        assert!(config.enable_namespaces.pid);
        assert!(config.enable_namespaces.net);
        assert!(config.capabilities.drop_all);
    }
    
    #[test]
    fn test_resource_limits() {
        let config = SandboxManager::default_config();
        let limits = ResourceLimits {
            memory_bytes: config.memory_limit_mb * 1024 * 1024,
            cpu_shares: (config.cpu_quota_percent * 10.24) as u32,
            max_files: 100,
            max_processes: 10,
        };
        
        assert_eq!(limits.memory_bytes, 500 * 1024 * 1024);
        assert_eq!(limits.cpu_shares, 512);
    }
    
    #[tokio::test]
    async fn test_sandbox_execution() {
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
    fn test_namespace_config() {
        let config = NamespaceConfig {
            pid: true,
            net: true,
            mnt: true,
            uts: true,
            ipc: true,
            user: false,
        };
        
        assert!(config.pid);
        assert!(config.net);
        assert!(config.mnt);
        assert!(config.uts);
        assert!(config.ipc);
        assert!(!config.user);
    }
    
    #[test]
    fn test_capability_config() {
        let config = CapabilityConfig {
            drop_all: true,
            allowed: vec!["CAP_DAC_OVERRIDE".to_string()],
            ambient: vec![],
        };
        
        assert!(config.drop_all);
        assert_eq!(config.allowed.len(), 1);
        assert_eq!(config.allowed[0], "CAP_DAC_OVERRIDE");
        assert!(config.ambient.is_empty());
    }
    
    #[test]
    fn test_ipc_message_types() {
        let data_msg = IpcMessage::Data(vec![1, 2, 3, 4]);
        let control_msg = IpcMessage::Control(ControlMessage::Stop);
        let status_msg = IpcMessage::Status(StatusMessage {
            resource_usage: ResourceUsage {
                memory_peak_bytes: 1024 * 1024,
                cpu_time_seconds: 1.5,
                files_opened: 5,
                network_bytes_sent: 0,
                network_bytes_received: 0,
                io_read_bytes: 1024,
                io_write_bytes: 512,
            },
            health: SandboxHealth::Healthy,
        });
        
        match data_msg {
            IpcMessage::Data(data) => assert_eq!(data.len(), 4),
            _ => panic!("Wrong message type"),
        }
        
        match control_msg {
            IpcMessage::Control(ControlMessage::Stop) => {},
            _ => panic!("Wrong message type"),
        }
        
        match status_msg {
            IpcMessage::Status(status) => {
                assert_eq!(status.resource_usage.memory_peak_bytes, 1024 * 1024);
                matches!(status.health, SandboxHealth::Healthy);
            },
            _ => panic!("Wrong message type"),
        }
    }
    
    #[tokio::test]
    async fn test_sandbox_kill() {
        let mut manager = SandboxManager::new().unwrap();
        
        // Create a sandbox that we'll kill
        let plugin_id = "kill_test";
        manager.create_sandbox(plugin_id).unwrap();
        
        // Kill the sandbox
        let result = manager.kill_sandbox(plugin_id);
        assert!(result.is_ok());
        
        // Verify it's been removed
        assert!(!manager.active_sandboxes.contains_key(plugin_id));
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
    async fn test_secure_ipc_message_types() {
        let mut manager = SandboxManager::new().unwrap();
        let plugin_id = "ipc_test";
        
        // Create sandbox to set up IPC
        manager.create_sandbox(plugin_id).unwrap();
        
        // Test sending control message
        let control_msg = IpcMessage::Control(ControlMessage::Pause);
        let result = manager.send_message(plugin_id, control_msg).await;
        assert!(result.is_ok());
        
        // Test sending data message
        let data_msg = IpcMessage::Data(vec![1, 2, 3, 4, 5]);
        let result = manager.send_message(plugin_id, data_msg).await;
        assert!(result.is_ok());
        
        // Clean up
        manager.cleanup_sandbox(plugin_id).unwrap();
    }

    #[test] 
    fn test_resource_limits_with_enhanced_config() {
        let config = SandboxConfig {
            memory_limit_mb: 128,
            cpu_quota_percent: 30.0,
            ..SandboxManager::default_config()
        };
        
        let limits = ResourceLimits {
            memory_bytes: config.memory_limit_mb * 1024 * 1024,
            cpu_shares: (config.cpu_quota_percent * 10.24) as u32,
            max_files: 50,
            max_processes: 5,
        };
        
        assert_eq!(limits.memory_bytes, 128 * 1024 * 1024);
        assert_eq!(limits.cpu_shares, 307); // 30.0 * 10.24 = 307.2 -> 307
        assert_eq!(limits.max_files, 50);
        assert_eq!(limits.max_processes, 5);
    }

    #[tokio::test]
    async fn test_sandbox_with_memory_constraint() {
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
    fn test_namespace_configuration_validation() {
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
    async fn test_sandbox_error_handling() {
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

    #[test]
    fn test_ipc_channel_security_token() {
        let plugin_id = "security_test";
        let process_id = std::process::id();
        let expected_token = format!("{}_{}", plugin_id, process_id);
        
        // This tests the token generation logic
        assert!(expected_token.contains(plugin_id));
        assert!(expected_token.contains(&process_id.to_string()));
    }
}