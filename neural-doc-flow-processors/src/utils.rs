//! Utility functions for neural document flow processors

use crate::{
    error::{NeuralError, Result},
    types::{ContentBlock, NeuralFeatures},
};
use std::collections::HashMap;
use std::path::Path;

/// SIMD utilities for accelerated neural operations
pub mod simd {
    use crate::Result;

    /// Accelerated dot product using SIMD when available
    pub fn dot_product(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(crate::error::NeuralError::InvalidInput(
                "Vector lengths must match for dot product".to_string()
            ));
        }

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return Ok(unsafe { dot_product_avx2(a, b) });
            }
            if is_x86_feature_detected!("sse2") {
                return Ok(unsafe { dot_product_sse2(a, b) });
            }
        }

        // Fallback to scalar implementation
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
    }

    /// Accelerated matrix multiplication using SIMD
    pub fn matrix_multiply(a: &[Vec<f32>], b: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if a.is_empty() || b.is_empty() || a[0].len() != b.len() {
            return Err(crate::error::NeuralError::InvalidInput(
                "Invalid matrix dimensions for multiplication".to_string()
            ));
        }

        let rows = a.len();
        let cols = b[0].len();
        let mut result = vec![vec![0.0; cols]; rows];

        for i in 0..rows {
            for j in 0..cols {
                let column: Vec<f32> = b.iter().map(|row| row[j]).collect();
                result[i][j] = dot_product(&a[i], &column)?;
            }
        }

        Ok(result)
    }

    /// Accelerated element-wise activation function
    pub fn apply_activation(input: &mut [f32], activation: ActivationType) {
        match activation {
            ActivationType::Sigmoid => apply_sigmoid_simd(input),
            ActivationType::ReLU => apply_relu_simd(input),
            ActivationType::Tanh => apply_tanh_simd(input),
        }
    }

    /// Activation function types
    #[derive(Debug, Clone, Copy)]
    pub enum ActivationType {
        Sigmoid,
        ReLU,
        Tanh,
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;

        let mut sum = _mm256_setzero_ps();
        let chunks = a.len() / 8;

        for i in 0..chunks {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let product = _mm256_mul_ps(a_vec, b_vec);
            sum = _mm256_add_ps(sum, product);
        }

        // Horizontal sum of the vector
        let sum_array: [f32; 8] = std::mem::transmute(sum);
        let mut result = sum_array.iter().sum::<f32>();

        // Handle remaining elements
        for i in chunks * 8..a.len() {
            result += a[i] * b[i];
        }

        result
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse2")]
    unsafe fn dot_product_sse2(a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;

        let mut sum = _mm_setzero_ps();
        let chunks = a.len() / 4;

        for i in 0..chunks {
            let a_vec = _mm_loadu_ps(a.as_ptr().add(i * 4));
            let b_vec = _mm_loadu_ps(b.as_ptr().add(i * 4));
            let product = _mm_mul_ps(a_vec, b_vec);
            sum = _mm_add_ps(sum, product);
        }

        // Horizontal sum
        let sum_array: [f32; 4] = std::mem::transmute(sum);
        let mut result = sum_array.iter().sum::<f32>();

        // Handle remaining elements
        for i in chunks * 4..a.len() {
            result += a[i] * b[i];
        }

        result
    }

    fn apply_sigmoid_simd(input: &mut [f32]) {
        for value in input.iter_mut() {
            *value = 1.0 / (1.0 + (-*value).exp());
        }
    }

    fn apply_relu_simd(input: &mut [f32]) {
        for value in input.iter_mut() {
            *value = value.max(0.0);
        }
    }

    fn apply_tanh_simd(input: &mut [f32]) {
        for value in input.iter_mut() {
            *value = value.tanh();
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_dot_product() {
            let a = vec![1.0, 2.0, 3.0, 4.0];
            let b = vec![2.0, 3.0, 4.0, 5.0];
            let result = dot_product(&a, &b).unwrap();
            assert_eq!(result, 40.0); // 1*2 + 2*3 + 3*4 + 4*5 = 40
        }

        #[test]
        fn test_matrix_multiply() {
            let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
            let b = vec![vec![2.0, 0.0], vec![1.0, 2.0]];
            let result = matrix_multiply(&a, &b).unwrap();
            
            assert_eq!(result[0][0], 4.0); // 1*2 + 2*1 = 4
            assert_eq!(result[0][1], 4.0); // 1*0 + 2*2 = 4
            assert_eq!(result[1][0], 10.0); // 3*2 + 4*1 = 10
            assert_eq!(result[1][1], 8.0); // 3*0 + 4*2 = 8
        }

        #[test]
        fn test_activation_functions() {
            let mut input = vec![-1.0, 0.0, 1.0, 2.0];
            
            apply_activation(&mut input, ActivationType::ReLU);
            assert_eq!(input, vec![0.0, 0.0, 1.0, 2.0]);
        }
    }
}

/// Memory management utilities
pub mod memory {
    use std::sync::{Arc, Mutex};
    use std::collections::HashMap;

    /// Memory pool for neural operations
    pub struct MemoryPool {
        pools: Arc<Mutex<HashMap<usize, Vec<Vec<f32>>>>>,
        max_pool_size: usize,
    }

    impl MemoryPool {
        /// Create a new memory pool
        pub fn new(max_pool_size: usize) -> Self {
            Self {
                pools: Arc::new(Mutex::new(HashMap::new())),
                max_pool_size,
            }
        }

        /// Get a vector from the pool or create a new one
        pub fn get_vector(&self, size: usize) -> Vec<f32> {
            let mut pools = self.pools.lock().unwrap();
            if let Some(pool) = pools.get_mut(&size) {
                if let Some(mut vec) = pool.pop() {
                    vec.clear();
                    vec.resize(size, 0.0);
                    return vec;
                }
            }
            vec![0.0; size]
        }

        /// Return a vector to the pool
        pub fn return_vector(&self, mut vec: Vec<f32>) {
            let size = vec.capacity();
            let mut pools = self.pools.lock().unwrap();
            let pool = pools.entry(size).or_insert_with(Vec::new);
            
            if pool.len() < self.max_pool_size {
                vec.clear();
                pool.push(vec);
            }
        }

        /// Clear all pools
        pub fn clear(&self) {
            let mut pools = self.pools.lock().unwrap();
            pools.clear();
        }

        /// Get memory usage statistics
        pub fn get_stats(&self) -> MemoryStats {
            let pools = self.pools.lock().unwrap();
            let total_vectors: usize = pools.values().map(|v| v.len()).sum();
            let total_capacity: usize = pools.iter()
                .map(|(size, pool)| size * pool.len())
                .sum();

            MemoryStats {
                pool_count: pools.len(),
                total_vectors,
                total_capacity_bytes: total_capacity * std::mem::size_of::<f32>(),
            }
        }
    }

    /// Memory usage statistics
    #[derive(Debug, Clone)]
    pub struct MemoryStats {
        pub pool_count: usize,
        pub total_vectors: usize,
        pub total_capacity_bytes: usize,
    }

    /// Memory manager for neural operations
    pub struct MemoryManager {
        pool: MemoryPool,
        allocation_tracker: Arc<Mutex<AllocationTracker>>,
    }

    impl MemoryManager {
        /// Create a new memory manager
        pub fn new() -> Self {
            Self {
                pool: MemoryPool::new(100), // Default pool size
                allocation_tracker: Arc::new(Mutex::new(AllocationTracker::new())),
            }
        }

        /// Allocate memory for neural operations
        pub fn allocate(&self, size: usize) -> ManagedVector {
            let vector = self.pool.get_vector(size);
            self.allocation_tracker.lock().unwrap().record_allocation(size);
            
            ManagedVector {
                data: vector,
                pool: &self.pool,
                tracker: Arc::clone(&self.allocation_tracker),
            }
        }

        /// Get memory statistics
        pub fn get_memory_stats(&self) -> (MemoryStats, AllocationStats) {
            let pool_stats = self.pool.get_stats();
            let allocation_stats = self.allocation_tracker.lock().unwrap().get_stats();
            (pool_stats, allocation_stats)
        }
    }

    /// Managed vector that returns to pool when dropped
    pub struct ManagedVector<'a> {
        data: Vec<f32>,
        pool: &'a MemoryPool,
        tracker: Arc<Mutex<AllocationTracker>>,
    }

    impl<'a> std::ops::Deref for ManagedVector<'a> {
        type Target = Vec<f32>;
        
        fn deref(&self) -> &Self::Target {
            &self.data
        }
    }

    impl<'a> std::ops::DerefMut for ManagedVector<'a> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.data
        }
    }

    impl<'a> Drop for ManagedVector<'a> {
        fn drop(&mut self) {
            let size = self.data.capacity();
            self.tracker.lock().unwrap().record_deallocation(size);
            let data = std::mem::take(&mut self.data);
            self.pool.return_vector(data);
        }
    }

    /// Allocation tracking
    #[derive(Debug)]
    struct AllocationTracker {
        total_allocations: usize,
        total_deallocations: usize,
        peak_memory: usize,
        current_memory: usize,
    }

    impl AllocationTracker {
        fn new() -> Self {
            Self {
                total_allocations: 0,
                total_deallocations: 0,
                peak_memory: 0,
                current_memory: 0,
            }
        }

        fn record_allocation(&mut self, size: usize) {
            self.total_allocations += 1;
            self.current_memory += size * std::mem::size_of::<f32>();
            self.peak_memory = self.peak_memory.max(self.current_memory);
        }

        fn record_deallocation(&mut self, size: usize) {
            self.total_deallocations += 1;
            self.current_memory = self.current_memory.saturating_sub(size * std::mem::size_of::<f32>());
        }

        fn get_stats(&self) -> AllocationStats {
            AllocationStats {
                total_allocations: self.total_allocations,
                total_deallocations: self.total_deallocations,
                peak_memory_bytes: self.peak_memory,
                current_memory_bytes: self.current_memory,
            }
        }
    }

    /// Allocation statistics
    #[derive(Debug, Clone)]
    pub struct AllocationStats {
        pub total_allocations: usize,
        pub total_deallocations: usize,
        pub peak_memory_bytes: usize,
        pub current_memory_bytes: usize,
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_memory_pool() {
            let pool = MemoryPool::new(10);
            
            let vec1 = pool.get_vector(100);
            assert_eq!(vec1.len(), 100);
            
            pool.return_vector(vec1);
            
            let vec2 = pool.get_vector(100);
            assert_eq!(vec2.len(), 100);
            
            let stats = pool.get_stats();
            assert_eq!(stats.pool_count, 1);
        }

        #[test]
        fn test_memory_manager() {
            let manager = MemoryManager::new();
            
            {
                let _vec = manager.allocate(1000);
                let (_, alloc_stats) = manager.get_memory_stats();
                assert_eq!(alloc_stats.total_allocations, 1);
                assert!(alloc_stats.current_memory_bytes > 0);
            }
            
            // Vector should be returned to pool when dropped
            let (_, alloc_stats) = manager.get_memory_stats();
            assert_eq!(alloc_stats.total_deallocations, 1);
        }
    }
}

/// Feature extraction utilities
pub mod features {
    use super::*;

    /// Extract basic text features
    pub fn extract_text_features(text: &str) -> NeuralFeatures {
        let mut features = NeuralFeatures::new();
        
        // Basic text statistics
        let char_count = text.chars().count() as f32;
        let word_count = text.split_whitespace().count() as f32;
        let line_count = text.lines().count() as f32;
        let avg_word_length = if word_count > 0.0 {
            char_count / word_count
        } else {
            0.0
        };

        // Character type ratios
        let uppercase_ratio = text.chars().filter(|c| c.is_uppercase()).count() as f32 / char_count.max(1.0);
        let digit_ratio = text.chars().filter(|c| c.is_numeric()).count() as f32 / char_count.max(1.0);
        let punctuation_ratio = text.chars().filter(|c| c.is_ascii_punctuation()).count() as f32 / char_count.max(1.0);

        features.text_features = vec![
            char_count.ln().max(0.0), // Log scale for large texts
            word_count.ln().max(0.0),
            line_count,
            avg_word_length,
            uppercase_ratio,
            digit_ratio,
            punctuation_ratio,
        ];

        features
    }

    /// Extract layout features from content block
    pub fn extract_layout_features(block: &ContentBlock) -> NeuralFeatures {
        let mut features = NeuralFeatures::new();
        
        let pos = &block.position;
        
        features.layout_features = vec![
            pos.x,
            pos.y,
            pos.width,
            pos.height,
            pos.x + pos.width, // right edge
            pos.y + pos.height, // bottom edge
            pos.width * pos.height, // area
            pos.width / pos.height.max(0.001), // aspect ratio
        ];

        features
    }

    /// Extract table-specific features
    pub fn extract_table_features(text: &str) -> NeuralFeatures {
        let mut features = NeuralFeatures::new();
        
        let lines: Vec<&str> = text.lines().collect();
        let line_count = lines.len() as f32;
        
        // Table structure indicators
        let pipe_count = text.matches('|').count() as f32;
        let tab_count = text.matches('\t').count() as f32;
        let comma_count = text.matches(',').count() as f32;
        
        // Consistency measures
        let pipe_lines = lines.iter().filter(|line| line.contains('|')).count() as f32;
        let tab_lines = lines.iter().filter(|line| line.contains('\t')).count() as f32;
        
        let pipe_consistency = if line_count > 0.0 { pipe_lines / line_count } else { 0.0 };
        let tab_consistency = if line_count > 0.0 { tab_lines / line_count } else { 0.0 };

        features.table_features = vec![
            line_count,
            pipe_count / line_count.max(1.0),
            tab_count / line_count.max(1.0),
            comma_count / line_count.max(1.0),
            pipe_consistency,
            tab_consistency,
        ];

        features
    }

    /// Combine multiple feature vectors
    pub fn combine_features(features_list: &[NeuralFeatures]) -> NeuralFeatures {
        let mut combined = NeuralFeatures::new();
        
        for features in features_list {
            combined.text_features.extend(&features.text_features);
            combined.layout_features.extend(&features.layout_features);
            combined.table_features.extend(&features.table_features);
            combined.image_features.extend(&features.image_features);
        }
        
        combined.combine_features();
        combined
    }

    /// Normalize features to improve neural network training
    pub fn normalize_features(features: &mut NeuralFeatures) {
        normalize_vector(&mut features.text_features);
        normalize_vector(&mut features.layout_features);
        normalize_vector(&mut features.table_features);
        normalize_vector(&mut features.image_features);
        normalize_vector(&mut features.combined_features);
    }

    fn normalize_vector(vec: &mut [f32]) {
        if vec.is_empty() {
            return;
        }

        let min_val = vec.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        
        if max_val > min_val {
            let range = max_val - min_val;
            for value in vec.iter_mut() {
                *value = (*value - min_val) / range;
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::types::Position;

        #[test]
        fn test_text_features() {
            let features = extract_text_features("Hello World! This is a TEST.");
            assert!(!features.text_features.is_empty());
            assert!(features.text_features[0] > 0.0); // char count
        }

        #[test]
        fn test_layout_features() {
            let block = ContentBlock::new("text")
                .with_position(Position::new(0, 0.1, 0.2, 0.8, 0.1));
            
            let features = extract_layout_features(&block);
            assert_eq!(features.layout_features.len(), 8);
            assert_eq!(features.layout_features[0], 0.1); // x position
        }

        #[test]
        fn test_table_features() {
            let table_text = "| Name | Age |\n| John | 30 |\n| Jane | 25 |";
            let features = extract_table_features(table_text);
            
            assert!(!features.table_features.is_empty());
            assert!(features.table_features[4] > 0.8); // pipe consistency
        }

        #[test]
        fn test_feature_normalization() {
            let mut features = NeuralFeatures::new();
            features.text_features = vec![1.0, 10.0, 100.0];
            
            normalize_features(&mut features);
            
            assert_eq!(features.text_features[0], 0.0); // min value
            assert_eq!(features.text_features[2], 1.0); // max value
        }
    }
}

/// Model conversion utilities
pub mod conversion {
    use super::*;

    /// Convert between different model formats
    pub struct ModelConverter;

    impl ModelConverter {
        /// Convert FANN model to a simplified JSON format
        pub fn fann_to_json(fann_path: &Path) -> Result<String> {
            use ruv_fann::Fann;

            let network = Fann::new_from_file(&fann_path.to_string_lossy())
                .map_err(|e| NeuralError::ModelLoad(format!("FANN load error: {}", e)))?;

            let model_info = serde_json::json!({
                "type": "fann",
                "num_inputs": network.get_num_input(),
                "num_outputs": network.get_num_output(),
                "num_layers": network.get_num_layers(),
                "activation_function": "sigmoid", // Default assumption
                "training_algorithm": "rprop",
                "connection_rate": 1.0,
                "created_at": chrono::Utc::now().to_rfc3339(),
            });

            Ok(serde_json::to_string_pretty(&model_info)?)
        }

        /// Export model metadata to JSON
        pub fn export_metadata(
            model_id: &str,
            input_size: usize,
            output_size: usize,
            model_type: &str,
        ) -> Result<String> {
            let metadata = serde_json::json!({
                "id": model_id,
                "input_size": input_size,
                "output_size": output_size,
                "model_type": model_type,
                "version": "1.0.0",
                "framework": "ruv-fann",
                "created_at": chrono::Utc::now().to_rfc3339(),
            });

            Ok(serde_json::to_string_pretty(&metadata)?)
        }

        /// Validate model compatibility
        pub fn validate_compatibility(
            model_path: &Path,
            expected_inputs: usize,
            expected_outputs: usize,
        ) -> Result<bool> {
            use ruv_fann::Fann;

            let network = Fann::new_from_file(&model_path.to_string_lossy())
                .map_err(|e| NeuralError::ModelLoad(format!("FANN load error: {}", e)))?;

            let inputs_match = network.get_num_input() == expected_inputs as u32;
            let outputs_match = network.get_num_output() == expected_outputs as u32;

            Ok(inputs_match && outputs_match)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_metadata_export() {
            let metadata = ModelConverter::export_metadata(
                "test_model",
                64,
                8,
                "text"
            ).unwrap();
            
            assert!(metadata.contains("test_model"));
            assert!(metadata.contains("\"input_size\": 64"));
            assert!(metadata.contains("\"output_size\": 8"));
        }
    }
}

/// Performance utilities
pub mod performance {
    use std::time::{Duration, Instant};
    use std::sync::{Arc, Mutex};
    use std::collections::HashMap;

    /// Performance profiler for neural operations
    pub struct Profiler {
        measurements: Arc<Mutex<HashMap<String, Vec<Duration>>>>,
    }

    impl Profiler {
        /// Create a new profiler
        pub fn new() -> Self {
            Self {
                measurements: Arc::new(Mutex::new(HashMap::new())),
            }
        }

        /// Start timing an operation
        pub fn start_timer(&self, operation: &str) -> Timer {
            Timer {
                operation: operation.to_string(),
                start_time: Instant::now(),
                profiler: Arc::clone(&self.measurements),
            }
        }

        /// Get performance statistics
        pub fn get_stats(&self) -> PerformanceStats {
            let measurements = self.measurements.lock().unwrap();
            let mut stats = PerformanceStats {
                operations: HashMap::new(),
            };

            for (operation, durations) in measurements.iter() {
                let total: Duration = durations.iter().sum();
                let count = durations.len();
                let average = if count > 0 {
                    total / count as u32
                } else {
                    Duration::from_nanos(0)
                };

                let min = durations.iter().min().copied().unwrap_or(Duration::from_nanos(0));
                let max = durations.iter().max().copied().unwrap_or(Duration::from_nanos(0));

                stats.operations.insert(operation.clone(), OperationStats {
                    count,
                    total,
                    average,
                    min,
                    max,
                });
            }

            stats
        }

        /// Clear all measurements
        pub fn clear(&self) {
            self.measurements.lock().unwrap().clear();
        }
    }

    /// Timer for measuring operation duration
    pub struct Timer {
        operation: String,
        start_time: Instant,
        profiler: Arc<Mutex<HashMap<String, Vec<Duration>>>>,
    }

    impl Drop for Timer {
        fn drop(&mut self) {
            let duration = self.start_time.elapsed();
            let mut measurements = self.profiler.lock().unwrap();
            measurements.entry(self.operation.clone())
                .or_insert_with(Vec::new)
                .push(duration);
        }
    }

    /// Performance statistics
    #[derive(Debug)]
    pub struct PerformanceStats {
        pub operations: HashMap<String, OperationStats>,
    }

    /// Statistics for individual operations
    #[derive(Debug)]
    pub struct OperationStats {
        pub count: usize,
        pub total: Duration,
        pub average: Duration,
        pub min: Duration,
        pub max: Duration,
    }

    impl PerformanceStats {
        /// Get the slowest operation
        pub fn slowest_operation(&self) -> Option<(&str, &OperationStats)> {
            self.operations.iter()
                .max_by_key(|(_, stats)| stats.average)
                .map(|(name, stats)| (name.as_str(), stats))
        }

        /// Get total time across all operations
        pub fn total_time(&self) -> Duration {
            self.operations.values()
                .map(|stats| stats.total)
                .sum()
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::thread;

        #[test]
        fn test_profiler() {
            let profiler = Profiler::new();
            
            {
                let _timer = profiler.start_timer("test_operation");
                thread::sleep(Duration::from_millis(10));
            }
            
            let stats = profiler.get_stats();
            assert!(stats.operations.contains_key("test_operation"));
            assert_eq!(stats.operations["test_operation"].count, 1);
            assert!(stats.operations["test_operation"].average >= Duration::from_millis(10));
        }
    }
}

/// Configuration utilities
pub mod config {
    use super::*;
    use std::env;

    /// Load configuration from environment variables
    pub fn load_env_config() -> HashMap<String, String> {
        let mut config = HashMap::new();
        
        // Neural processing configuration
        if let Ok(threads) = env::var("NEURAL_THREADS") {
            config.insert("max_threads".to_string(), threads);
        }
        
        if let Ok(batch_size) = env::var("NEURAL_BATCH_SIZE") {
            config.insert("batch_size".to_string(), batch_size);
        }
        
        if let Ok(model_path) = env::var("NEURAL_MODEL_PATH") {
            config.insert("model_path".to_string(), model_path);
        }
        
        if let Ok(enable_simd) = env::var("NEURAL_ENABLE_SIMD") {
            config.insert("enable_simd".to_string(), enable_simd);
        }

        config
    }

    /// Detect optimal configuration based on system capabilities
    pub fn detect_optimal_config() -> HashMap<String, String> {
        let mut config = HashMap::new();
        
        // CPU configuration
        let cpu_count = num_cpus::get();
        config.insert("max_threads".to_string(), cpu_count.to_string());
        
        // Memory configuration
        let available_memory = get_available_memory();
        let optimal_batch_size = (available_memory / (1024 * 1024 * 100)).min(64).max(1); // 100MB per batch item
        config.insert("batch_size".to_string(), optimal_batch_size.to_string());
        
        // SIMD detection
        let simd_support = detect_simd_support();
        config.insert("enable_simd".to_string(), simd_support.to_string());
        
        config
    }

    fn get_available_memory() -> usize {
        // Simplified memory detection - in practice, use system-specific APIs
        1024 * 1024 * 1024 * 4 // Assume 4GB available
    }

    fn detect_simd_support() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            is_x86_feature_detected!("sse2") || is_x86_feature_detected!("avx2")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_optimal_config_detection() {
            let config = detect_optimal_config();
            
            assert!(config.contains_key("max_threads"));
            assert!(config.contains_key("batch_size"));
            assert!(config.contains_key("enable_simd"));
            
            let threads: usize = config["max_threads"].parse().unwrap();
            assert!(threads > 0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_compilation() {
        // This test ensures all modules compile correctly
        assert!(true);
    }
}