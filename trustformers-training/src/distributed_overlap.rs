//! Gradient Communication Overlap
//!
//! Async gradient communication overlap for distributed training — start AllReduce
//! for early layers while computing gradients for later layers.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ─── Gradient tensor ─────────────────────────────────────────────────────────

/// Simulated gradient tensor for a layer
#[derive(Debug, Clone)]
pub struct GradientTensor {
    pub layer_name: String,
    pub layer_index: usize,
    pub values: Vec<f64>,
    /// Gradient has been computed (backward pass reached this layer)
    pub is_ready: bool,
    /// AllReduce completed
    pub is_reduced: bool,
}

impl GradientTensor {
    pub fn new(layer_name: impl Into<String>, layer_index: usize, size: usize) -> Self {
        Self {
            layer_name: layer_name.into(),
            layer_index,
            values: vec![0.0; size],
            is_ready: false,
            is_reduced: false,
        }
    }

    pub fn mark_ready(&mut self) {
        self.is_ready = true;
    }

    /// Average the gradient by world_size (simulated AllReduce)
    pub fn apply_allreduce(&mut self, world_size: usize) {
        if world_size == 0 {
            return;
        }
        let ws = world_size as f64;
        for v in self.values.iter_mut() {
            *v /= ws;
        }
        self.is_reduced = true;
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GradientOverlapConfig {
    /// Number of distributed processes
    pub world_size: usize,
    pub num_layers: usize,
    /// Group layers into buckets for AllReduce (MB)
    pub bucket_size_mb: f64,
    /// Fraction of backward to overlap (0.0–1.0)
    pub overlap_fraction: f64,
    /// Gradient compression before communication
    pub compression_enabled: bool,
    /// Top-k sparsification ratio (e.g. 0.01 = keep 1%)
    pub compression_ratio: f64,
}

impl Default for GradientOverlapConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            num_layers: 32,
            bucket_size_mb: 25.0,
            overlap_fraction: 0.8,
            compression_enabled: false,
            compression_ratio: 0.01,
        }
    }
}

// ─── Stats ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct OverlapStats {
    pub total_layers: usize,
    pub layers_reduced: usize,
    pub buckets_triggered: usize,
    pub compression_savings_bytes: usize,
    pub overlap_efficiency: f64,
}

// ─── Manager ─────────────────────────────────────────────────────────────────

pub struct GradientOverlapManager {
    config: GradientOverlapConfig,
    gradients: Arc<Mutex<HashMap<String, GradientTensor>>>,
    /// Layer names grouped into AllReduce buckets
    buckets: Vec<Vec<String>>,
    /// Simulated communication time in nanoseconds
    communication_time_ns: Arc<Mutex<u64>>,
    /// Simulated compute time in nanoseconds
    compute_time_ns: Arc<Mutex<u64>>,
    /// Number of buckets whose AllReduce has been triggered
    buckets_triggered: Arc<Mutex<usize>>,
    /// Bytes saved by compression
    compression_savings_bytes: Arc<Mutex<usize>>,
}

impl GradientOverlapManager {
    pub fn new(config: GradientOverlapConfig, layer_names: Vec<String>) -> Self {
        let buckets = Self::build_buckets(&layer_names, config.num_layers);

        let gradients: HashMap<String, GradientTensor> = layer_names
            .iter()
            .enumerate()
            .map(|(idx, name)| {
                (name.clone(), GradientTensor::new(name.clone(), idx, 0))
            })
            .collect();

        Self {
            config,
            gradients: Arc::new(Mutex::new(gradients)),
            buckets,
            communication_time_ns: Arc::new(Mutex::new(0)),
            compute_time_ns: Arc::new(Mutex::new(0)),
            buckets_triggered: Arc::new(Mutex::new(0)),
            compression_savings_bytes: Arc::new(Mutex::new(0)),
        }
    }

    /// Register that a layer's gradient is ready (backward pass reached this layer).
    /// This may trigger AllReduce for a completed bucket.
    pub fn gradient_ready(
        &self,
        layer_name: &str,
        grad_values: Vec<f64>,
    ) -> Result<(), OverlapError> {
        let start_compute = Instant::now();

        {
            let mut grads = self
                .gradients
                .lock()
                .map_err(|e| OverlapError::LayerNotFound(e.to_string()))?;

            let tensor = grads
                .get_mut(layer_name)
                .ok_or_else(|| OverlapError::LayerNotFound(layer_name.to_string()))?;

            if tensor.is_ready {
                return Err(OverlapError::AlreadySet(layer_name.to_string()));
            }

            if self.config.compression_enabled {
                let compressed = Self::compress_gradient(&grad_values, self.config.compression_ratio);
                let saved = (grad_values.len() - compressed.iter().filter(|&&v| v != 0.0).count()) * 8;
                if let Ok(mut cs) = self.compression_savings_bytes.lock() {
                    *cs += saved;
                }
                tensor.values = compressed;
            } else {
                tensor.values = grad_values;
            }

            tensor.mark_ready();
        }

        // Accumulate compute time
        let compute_ns = start_compute.elapsed().as_nanos() as u64;
        if let Ok(mut ct) = self.compute_time_ns.lock() {
            *ct += compute_ns;
        }

        // Check whether the bucket containing this layer is now fully ready
        let bucket = self.bucket_for_layer(layer_name);
        if let Some(b) = bucket {
            let bucket_ready = self.is_bucket_ready(&b);
            if bucket_ready {
                self.trigger_allreduce_for_bucket(&b)?;
            }
        }

        Ok(())
    }

    /// Check if all gradients have been reduced (ready for optimizer step).
    pub fn all_reduced(&self) -> bool {
        self.gradients
            .lock()
            .map(|grads| grads.values().all(|g| g.is_reduced))
            .unwrap_or(false)
    }

    /// Wait until all gradients are reduced (with timeout).
    pub fn wait_for_all_reduce(&self, timeout_ms: u64) -> Result<(), OverlapError> {
        let deadline = Instant::now() + Duration::from_millis(timeout_ms);
        loop {
            if self.all_reduced() {
                return Ok(());
            }
            if Instant::now() >= deadline {
                return Err(OverlapError::Timeout);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Overlap efficiency: compute_time / (compute_time + communication_time).
    /// Perfect overlap = 1.0, no overlap ≈ 0.5.
    pub fn overlap_efficiency(&self) -> f64 {
        let comm = self
            .communication_time_ns
            .lock()
            .map(|g| *g)
            .unwrap_or(0);
        let comp = self
            .compute_time_ns
            .lock()
            .map(|g| *g)
            .unwrap_or(0);

        if comm == 0 && comp == 0 {
            // Use configured overlap_fraction as a theoretical baseline
            return self.config.overlap_fraction;
        }
        if comm + comp == 0 {
            return 1.0;
        }
        comp as f64 / (comp + comm) as f64
    }

    /// Apply optional top-k gradient compression.
    /// Keeps the top-k% by absolute value, zeros out the rest.
    pub fn compress_gradient(values: &[f64], keep_ratio: f64) -> Vec<f64> {
        if values.is_empty() || keep_ratio <= 0.0 {
            return vec![0.0; values.len()];
        }
        if keep_ratio >= 1.0 {
            return values.to_vec();
        }

        let keep_count = ((values.len() as f64 * keep_ratio).ceil() as usize).max(1);

        // Collect indices sorted by absolute value descending
        let mut indexed: Vec<(usize, f64)> = values
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_indices: std::collections::HashSet<usize> =
            indexed.iter().take(keep_count).map(|(i, _)| *i).collect();

        values
            .iter()
            .enumerate()
            .map(|(i, &v)| if top_indices.contains(&i) { v } else { 0.0 })
            .collect()
    }

    /// Get statistics about the backward pass.
    pub fn stats(&self) -> OverlapStats {
        let (total, reduced) = self
            .gradients
            .lock()
            .map(|grads| {
                let total = grads.len();
                let reduced = grads.values().filter(|g| g.is_reduced).count();
                (total, reduced)
            })
            .unwrap_or((0, 0));

        let bt = self
            .buckets_triggered
            .lock()
            .map(|g| *g)
            .unwrap_or(0);

        let savings = self
            .compression_savings_bytes
            .lock()
            .map(|g| *g)
            .unwrap_or(0);

        OverlapStats {
            total_layers: total,
            layers_reduced: reduced,
            buckets_triggered: bt,
            compression_savings_bytes: savings,
            overlap_efficiency: self.overlap_efficiency(),
        }
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    fn build_buckets(layer_names: &[String], num_layers: usize) -> Vec<Vec<String>> {
        if layer_names.is_empty() {
            return vec![];
        }
        // Aim for roughly sqrt(num_layers) layers per bucket
        let bucket_capacity = ((num_layers as f64).sqrt().ceil() as usize).max(1);
        layer_names
            .chunks(bucket_capacity)
            .map(|chunk| chunk.to_vec())
            .collect()
    }

    fn bucket_for_layer(&self, layer_name: &str) -> Option<Vec<String>> {
        self.buckets
            .iter()
            .find(|bucket| bucket.iter().any(|n| n == layer_name))
            .cloned()
    }

    fn is_bucket_ready(&self, bucket: &[String]) -> bool {
        self.gradients
            .lock()
            .map(|grads| {
                bucket.iter().all(|name| {
                    grads.get(name).map(|g| g.is_ready).unwrap_or(false)
                })
            })
            .unwrap_or(false)
    }

    fn trigger_allreduce_for_bucket(&self, bucket: &[String]) -> Result<(), OverlapError> {
        let comm_start = Instant::now();

        {
            let mut grads = self
                .gradients
                .lock()
                .map_err(|e| OverlapError::LayerNotFound(e.to_string()))?;

            for name in bucket {
                if let Some(tensor) = grads.get_mut(name) {
                    if tensor.is_ready && !tensor.is_reduced {
                        tensor.apply_allreduce(self.config.world_size);
                    }
                }
            }
        }

        let comm_ns = comm_start.elapsed().as_nanos() as u64;
        if let Ok(mut ct) = self.communication_time_ns.lock() {
            *ct += comm_ns;
        }
        if let Ok(mut bt) = self.buckets_triggered.lock() {
            *bt += 1;
        }

        Ok(())
    }
}

// ─── Error ───────────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum OverlapError {
    #[error("Layer not found: {0}")]
    LayerNotFound(String),
    #[error("Gradient already set for: {0}")]
    AlreadySet(String),
    #[error("Timeout waiting for AllReduce")]
    Timeout,
    #[error("Invalid compression ratio: {0}")]
    InvalidCompression(f64),
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_layers(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("layer_{i}")).collect()
    }

    fn make_manager(world_size: usize, n_layers: usize) -> GradientOverlapManager {
        let cfg = GradientOverlapConfig {
            world_size,
            num_layers: n_layers,
            bucket_size_mb: 25.0,
            overlap_fraction: 0.8,
            compression_enabled: false,
            compression_ratio: 0.01,
        };
        GradientOverlapManager::new(cfg, make_layers(n_layers))
    }

    // 1. gradient_ready stores values and marks layer ready
    #[test]
    fn test_gradient_ready_marks_layer() {
        let mgr = make_manager(4, 4);
        mgr.gradient_ready("layer_0", vec![1.0, 2.0, 3.0]).unwrap();
        let grads = mgr.gradients.lock().unwrap_or_else(|e| e.into_inner());
        let t = &grads["layer_0"];
        assert!(t.is_ready);
    }

    // 2. gradient_ready triggers AllReduce for completed bucket
    #[test]
    fn test_gradient_ready_triggers_allreduce() {
        // 4 layers, num_layers=4 → bucket_capacity = ceil(sqrt(4)) = 2
        // Buckets: [layer_0, layer_1], [layer_2, layer_3]
        let mgr = make_manager(2, 4);
        mgr.gradient_ready("layer_0", vec![4.0]).unwrap();
        // Bucket not complete yet
        {
            let grads = mgr.gradients.lock().unwrap_or_else(|e| e.into_inner());
            assert!(!grads["layer_0"].is_reduced);
        }
        mgr.gradient_ready("layer_1", vec![4.0]).unwrap();
        // Both in bucket done → AllReduce should have fired
        {
            let grads = mgr.gradients.lock().unwrap_or_else(|e| e.into_inner());
            assert!(grads["layer_0"].is_reduced);
            assert!(grads["layer_1"].is_reduced);
            // world_size=2 → values should be 4.0/2 = 2.0
            assert!((grads["layer_0"].values[0] - 2.0).abs() < 1e-10);
        }
    }

    // 3. all_reduced returns false until all layers done
    #[test]
    fn test_all_reduced_false_until_complete() {
        let mgr = make_manager(1, 4);
        assert!(!mgr.all_reduced());
        for i in 0..4usize {
            mgr.gradient_ready(&format!("layer_{i}"), vec![1.0]).unwrap();
        }
        assert!(mgr.all_reduced());
    }

    // 4. world_size averaging
    #[test]
    fn test_world_size_averaging() {
        // single bucket of 2 layers
        let layers = vec!["l0".to_string(), "l1".to_string()];
        let cfg = GradientOverlapConfig {
            world_size: 8,
            num_layers: 2,
            ..Default::default()
        };
        let mgr = GradientOverlapManager::new(cfg, layers);
        mgr.gradient_ready("l0", vec![8.0, 16.0]).unwrap();
        mgr.gradient_ready("l1", vec![8.0]).unwrap();
        let grads = mgr.gradients.lock().unwrap_or_else(|e| e.into_inner());
        // 8.0 / 8 = 1.0
        assert!((grads["l0"].values[0] - 1.0).abs() < 1e-10);
        assert!((grads["l0"].values[1] - 2.0).abs() < 1e-10);
        assert!((grads["l1"].values[0] - 1.0).abs() < 1e-10);
    }

    // 5. duplicate gradient_ready returns AlreadySet error
    #[test]
    fn test_already_set_error() {
        let mgr = make_manager(1, 2);
        mgr.gradient_ready("layer_0", vec![1.0]).unwrap();
        let result = mgr.gradient_ready("layer_0", vec![2.0]);
        assert!(matches!(result, Err(OverlapError::AlreadySet(_))));
    }

    // 6. unknown layer returns LayerNotFound error
    #[test]
    fn test_layer_not_found_error() {
        let mgr = make_manager(1, 2);
        let result = mgr.gradient_ready("nonexistent", vec![1.0]);
        assert!(matches!(result, Err(OverlapError::LayerNotFound(_))));
    }

    // 7. compress_gradient keeps top-k by absolute value
    #[test]
    fn test_compress_gradient_top_k() {
        let values = vec![0.1, -0.9, 0.5, -0.3, 0.8];
        // keep 40% → keep 2 values
        let compressed = GradientOverlapManager::compress_gradient(&values, 0.4);
        assert_eq!(compressed.len(), 5);
        // top 2 by abs: -0.9 (idx 1) and 0.8 (idx 4)
        assert!((compressed[1] - (-0.9)).abs() < 1e-10);
        assert!((compressed[4] - 0.8).abs() < 1e-10);
        // others zeroed
        assert_eq!(compressed[0], 0.0);
        assert_eq!(compressed[2], 0.0);
        assert_eq!(compressed[3], 0.0);
    }

    // 8. compress_gradient ratio >= 1.0 returns all values
    #[test]
    fn test_compress_gradient_full_keep() {
        let values = vec![1.0, 2.0, 3.0];
        let compressed = GradientOverlapManager::compress_gradient(&values, 1.0);
        assert_eq!(compressed, values);
    }

    // 9. compress_gradient ratio <= 0.0 zeroes all
    #[test]
    fn test_compress_gradient_zero_keep() {
        let values = vec![1.0, 2.0, 3.0];
        let compressed = GradientOverlapManager::compress_gradient(&values, 0.0);
        assert!(compressed.iter().all(|&v| v == 0.0));
    }

    // 10. overlap_efficiency uses config fraction when no timing data
    #[test]
    fn test_overlap_efficiency_default() {
        let mgr = make_manager(1, 4);
        // No timing data accumulated → falls back to config overlap_fraction
        let eff = mgr.overlap_efficiency();
        assert!(
            (eff - 0.8).abs() < 1e-10,
            "expected 0.8, got {eff}"
        );
    }

    // 11. stats returns correct layer/bucket counts
    #[test]
    fn test_stats_counts() {
        let mgr = make_manager(1, 4);
        let s0 = mgr.stats();
        assert_eq!(s0.total_layers, 4);
        assert_eq!(s0.layers_reduced, 0);
        assert_eq!(s0.buckets_triggered, 0);

        // Complete first bucket (layer_0, layer_1)
        mgr.gradient_ready("layer_0", vec![1.0]).unwrap();
        mgr.gradient_ready("layer_1", vec![1.0]).unwrap();
        let s1 = mgr.stats();
        assert_eq!(s1.layers_reduced, 2);
        assert_eq!(s1.buckets_triggered, 1);
    }

    // 12. wait_for_all_reduce succeeds when all layers ready
    #[test]
    fn test_wait_for_all_reduce_success() {
        let mgr = make_manager(1, 2);
        mgr.gradient_ready("layer_0", vec![1.0]).unwrap();
        mgr.gradient_ready("layer_1", vec![1.0]).unwrap();
        mgr.wait_for_all_reduce(100).unwrap();
    }

    // 13. wait_for_all_reduce times out when layers not ready
    #[test]
    fn test_wait_for_all_reduce_timeout() {
        let mgr = make_manager(1, 4);
        // Only mark one layer — not all buckets complete
        mgr.gradient_ready("layer_0", vec![1.0]).unwrap();
        let result = mgr.wait_for_all_reduce(20); // 20 ms timeout
        assert!(matches!(result, Err(OverlapError::Timeout)));
    }

    // 14. compression_enabled records savings in stats
    #[test]
    fn test_compression_savings_in_stats() {
        let layers = vec!["l0".to_string()];
        let cfg = GradientOverlapConfig {
            world_size: 1,
            num_layers: 1,
            compression_enabled: true,
            compression_ratio: 0.1, // keep 10%
            ..Default::default()
        };
        let mgr = GradientOverlapManager::new(cfg, layers);
        // 100-element gradient; keep 10% = 10 → save 90 * 8 = 720 bytes
        mgr.gradient_ready("l0", vec![1.0; 100]).unwrap();
        let s = mgr.stats();
        assert!(
            s.compression_savings_bytes > 0,
            "expected compression savings, got 0"
        );
    }

    // 15. build_buckets groups layers correctly
    #[test]
    fn test_build_buckets() {
        // 9 layers → capacity = ceil(sqrt(9)) = 3 → 3 buckets of 3
        let names: Vec<String> = (0..9).map(|i| format!("l{i}")).collect();
        let buckets = GradientOverlapManager::build_buckets(&names, 9);
        assert_eq!(buckets.len(), 3);
        assert_eq!(buckets[0].len(), 3);
        assert_eq!(buckets[2].len(), 3);
    }
}
