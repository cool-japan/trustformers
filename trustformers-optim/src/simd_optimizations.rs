//! SIMD Optimizations for Optimizers
//!
//! This module provides SIMD-optimized implementations of optimizer operations
//! for improved performance on x86_64, ARM, and other architectures.

use anyhow::{anyhow, Result};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// SIMD-optimized operations configuration
#[derive(Debug, Clone)]
pub struct SIMDConfig {
    /// Enable AVX2 operations (x86_64)
    pub enable_avx2: bool,
    /// Enable AVX-512 operations (x86_64)
    pub enable_avx512: bool,
    /// Enable NEON operations (ARM)
    pub enable_neon: bool,
    /// Minimum vector size for SIMD operations
    pub min_vector_size: usize,
    /// Enable unrolled loops
    pub enable_unrolling: bool,
}

impl Default for SIMDConfig {
    fn default() -> Self {
        Self {
            enable_avx2: true,
            enable_avx512: true,
            enable_neon: true,
            min_vector_size: 8,
            enable_unrolling: true,
        }
    }
}

/// SIMD operations for optimizer kernels
pub struct SIMDOptimizer {
    config: SIMDConfig,
}

impl SIMDOptimizer {
    /// Create a new SIMD optimizer with configuration
    pub fn new(config: SIMDConfig) -> Self {
        Self { config }
    }

    /// Detect available SIMD instruction sets
    pub fn detect_capabilities() -> SIMDConfig {
        SIMDConfig {
            enable_avx2: {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    is_x86_feature_detected!("avx2")
                }
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    false
                }
            },
            enable_avx512: {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    is_x86_feature_detected!("avx512f")
                }
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    false
                }
            },
            enable_neon: cfg!(target_arch = "aarch64"),
            min_vector_size: 8,
            enable_unrolling: true,
        }
    }

    /// SIMD-optimized Adam update with AVX2
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    pub unsafe fn adam_update_avx2(
        &self,
        params: &mut [f32],
        gradients: &[f32],
        momentum: &mut [f32],
        velocity: &mut [f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        step: i32,
    ) -> Result<()> {
        if params.len() != gradients.len()
            || params.len() != momentum.len()
            || params.len() != velocity.len()
        {
            return Err(anyhow!("All arrays must have the same length"));
        }

        let bias_correction1 = 1.0 - beta1.powi(step);
        let bias_correction2 = 1.0 - beta2.powi(step);
        let corrected_lr = lr * (bias_correction2.sqrt() / bias_correction1);

        // SIMD constants
        let beta1_vec = _mm256_set1_ps(beta1);
        let beta2_vec = _mm256_set1_ps(beta2);
        let one_minus_beta1 = _mm256_set1_ps(1.0 - beta1);
        let one_minus_beta2 = _mm256_set1_ps(1.0 - beta2);
        let eps_vec = _mm256_set1_ps(eps);
        let lr_vec = _mm256_set1_ps(corrected_lr);

        let len = params.len();
        let chunks = len / 8;
        let _remainder = len % 8;

        // Process 8 elements at a time with AVX2
        for i in 0..chunks {
            let idx = i * 8;

            // Load values
            let p = _mm256_loadu_ps(params.as_ptr().add(idx));
            let g = _mm256_loadu_ps(gradients.as_ptr().add(idx));
            let m = _mm256_loadu_ps(momentum.as_ptr().add(idx));
            let v = _mm256_loadu_ps(velocity.as_ptr().add(idx));

            // Update momentum: m = β₁ * m + (1 - β₁) * g
            let m_new = _mm256_fmadd_ps(beta1_vec, m, _mm256_mul_ps(one_minus_beta1, g));

            // Update velocity: v = β₂ * v + (1 - β₂) * g²
            let g_sq = _mm256_mul_ps(g, g);
            let v_new = _mm256_fmadd_ps(beta2_vec, v, _mm256_mul_ps(one_minus_beta2, g_sq));

            // Update parameters: p = p - α * m / (√v + ε)
            let v_sqrt = _mm256_sqrt_ps(v_new);
            let v_sqrt_eps = _mm256_add_ps(v_sqrt, eps_vec);
            let update = _mm256_div_ps(m_new, v_sqrt_eps);
            let p_new = _mm256_fnmadd_ps(lr_vec, update, p);

            // Store results
            _mm256_storeu_ps(params.as_mut_ptr().add(idx), p_new);
            _mm256_storeu_ps(momentum.as_mut_ptr().add(idx), m_new);
            _mm256_storeu_ps(velocity.as_mut_ptr().add(idx), v_new);
        }

        // Handle remaining elements
        for i in (chunks * 8)..len {
            let g = gradients[i];
            let m = momentum[i];
            let v = velocity[i];

            let m_new = beta1 * m + (1.0 - beta1) * g;
            let v_new = beta2 * v + (1.0 - beta2) * g * g;

            momentum[i] = m_new;
            velocity[i] = v_new;
            params[i] -= corrected_lr * m_new / (v_new.sqrt() + eps);
        }

        Ok(())
    }

    /// SIMD-optimized AdamW update with AVX2 (decoupled weight decay)
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    pub unsafe fn adamw_update_avx2(
        &self,
        params: &mut [f32],
        gradients: &[f32],
        momentum: &mut [f32],
        velocity: &mut [f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: i32,
    ) -> Result<()> {
        if params.len() != gradients.len()
            || params.len() != momentum.len()
            || params.len() != velocity.len()
        {
            return Err(anyhow!("All arrays must have the same length"));
        }

        let bias_correction1 = 1.0 - beta1.powi(step);
        let bias_correction2 = 1.0 - beta2.powi(step);
        let corrected_lr = lr * (bias_correction2.sqrt() / bias_correction1);

        // SIMD constants
        let beta1_vec = _mm256_set1_ps(beta1);
        let beta2_vec = _mm256_set1_ps(beta2);
        let one_minus_beta1 = _mm256_set1_ps(1.0 - beta1);
        let one_minus_beta2 = _mm256_set1_ps(1.0 - beta2);
        let eps_vec = _mm256_set1_ps(eps);
        let lr_vec = _mm256_set1_ps(corrected_lr);
        let wd_vec = _mm256_set1_ps(1.0 - lr * weight_decay);

        let len = params.len();
        let chunks = len / 8;

        for i in 0..chunks {
            let idx = i * 8;

            let p = _mm256_loadu_ps(params.as_ptr().add(idx));
            let g = _mm256_loadu_ps(gradients.as_ptr().add(idx));
            let m = _mm256_loadu_ps(momentum.as_ptr().add(idx));
            let v = _mm256_loadu_ps(velocity.as_ptr().add(idx));

            // Apply weight decay first: p = p * (1 - lr * wd)
            let p_decayed = _mm256_mul_ps(p, wd_vec);

            // Update momentum and velocity
            let m_new = _mm256_fmadd_ps(beta1_vec, m, _mm256_mul_ps(one_minus_beta1, g));
            let g_sq = _mm256_mul_ps(g, g);
            let v_new = _mm256_fmadd_ps(beta2_vec, v, _mm256_mul_ps(one_minus_beta2, g_sq));

            // Update parameters
            let v_sqrt = _mm256_sqrt_ps(v_new);
            let v_sqrt_eps = _mm256_add_ps(v_sqrt, eps_vec);
            let update = _mm256_div_ps(m_new, v_sqrt_eps);
            let p_new = _mm256_fnmadd_ps(lr_vec, update, p_decayed);

            _mm256_storeu_ps(params.as_mut_ptr().add(idx), p_new);
            _mm256_storeu_ps(momentum.as_mut_ptr().add(idx), m_new);
            _mm256_storeu_ps(velocity.as_mut_ptr().add(idx), v_new);
        }

        // Handle remaining elements
        for i in (chunks * 8)..len {
            let p = params[i];
            let g = gradients[i];
            let m = momentum[i];
            let v = velocity[i];

            let p_decayed = p * (1.0 - lr * weight_decay);
            let m_new = beta1 * m + (1.0 - beta1) * g;
            let v_new = beta2 * v + (1.0 - beta2) * g * g;

            momentum[i] = m_new;
            velocity[i] = v_new;
            params[i] = p_decayed - corrected_lr * m_new / (v_new.sqrt() + eps);
        }

        Ok(())
    }

    /// SIMD-optimized SGD with momentum update
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    pub unsafe fn sgd_momentum_update_avx2(
        &self,
        params: &mut [f32],
        gradients: &[f32],
        momentum: &mut [f32],
        lr: f32,
        momentum_factor: f32,
        weight_decay: f32,
        nesterov: bool,
    ) -> Result<()> {
        if params.len() != gradients.len() || params.len() != momentum.len() {
            return Err(anyhow!("All arrays must have the same length"));
        }

        let lr_vec = _mm256_set1_ps(lr);
        let momentum_vec = _mm256_set1_ps(momentum_factor);
        let wd_vec = _mm256_set1_ps(weight_decay);

        let len = params.len();
        let chunks = len / 8;

        for i in 0..chunks {
            let idx = i * 8;

            let p = _mm256_loadu_ps(params.as_ptr().add(idx));
            let g = _mm256_loadu_ps(gradients.as_ptr().add(idx));
            let m = _mm256_loadu_ps(momentum.as_ptr().add(idx));

            // Apply weight decay to gradient: g = g + wd * p
            let g_wd = _mm256_fmadd_ps(wd_vec, p, g);

            // Update momentum: m = momentum * m + g
            let m_new = _mm256_fmadd_ps(momentum_vec, m, g_wd);

            // Update parameters
            let update = if nesterov {
                // Nesterov: p = p - lr * (momentum * m + g)
                _mm256_fmadd_ps(momentum_vec, m_new, g_wd)
            } else {
                // Standard: p = p - lr * m
                m_new
            };

            let p_new = _mm256_fnmadd_ps(lr_vec, update, p);

            _mm256_storeu_ps(params.as_mut_ptr().add(idx), p_new);
            _mm256_storeu_ps(momentum.as_mut_ptr().add(idx), m_new);
        }

        // Handle remaining elements
        for i in (chunks * 8)..len {
            let p = params[i];
            let g = gradients[i] + weight_decay * p;
            let m = momentum[i];

            let m_new = momentum_factor * m + g;
            momentum[i] = m_new;

            if nesterov {
                params[i] = p - lr * (momentum_factor * m_new + g);
            } else {
                params[i] = p - lr * m_new;
            }
        }

        Ok(())
    }

    /// SIMD-optimized gradient clipping
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    pub unsafe fn clip_gradients_avx2(&self, gradients: &mut [f32], max_norm: f32) -> Result<f32> {
        let len = gradients.len();
        let chunks = len / 8;

        // Compute global norm
        let mut norm_sq_vec = _mm256_setzero_ps();

        for i in 0..chunks {
            let idx = i * 8;
            let g = _mm256_loadu_ps(gradients.as_ptr().add(idx));
            let g_sq = _mm256_mul_ps(g, g);
            norm_sq_vec = _mm256_add_ps(norm_sq_vec, g_sq);
        }

        // Horizontal sum of norm_sq_vec
        let mut norm_sq = 0.0f32;
        let norm_sq_array: [f32; 8] = std::mem::transmute(norm_sq_vec);
        for &val in &norm_sq_array {
            norm_sq += val;
        }

        // Add remaining elements
        for i in (chunks * 8)..len {
            norm_sq += gradients[i] * gradients[i];
        }

        let global_norm = norm_sq.sqrt();

        if global_norm > max_norm {
            let scale = max_norm / global_norm;
            let scale_vec = _mm256_set1_ps(scale);

            // Scale gradients
            for i in 0..chunks {
                let idx = i * 8;
                let g = _mm256_loadu_ps(gradients.as_ptr().add(idx));
                let g_scaled = _mm256_mul_ps(g, scale_vec);
                _mm256_storeu_ps(gradients.as_mut_ptr().add(idx), g_scaled);
            }

            // Scale remaining elements
            for i in (chunks * 8)..len {
                gradients[i] *= scale;
            }
        }

        Ok(global_norm)
    }

    /// SIMD-optimized vector addition (for gradient accumulation)
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    pub unsafe fn vector_add_avx2(&self, a: &mut [f32], b: &[f32], scale: f32) -> Result<()> {
        if a.len() != b.len() {
            return Err(anyhow!("Vectors must have the same length"));
        }

        let scale_vec = _mm256_set1_ps(scale);
        let len = a.len();
        let chunks = len / 8;

        for i in 0..chunks {
            let idx = i * 8;
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(idx));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(idx));
            let result = _mm256_fmadd_ps(b_vec, scale_vec, a_vec);
            _mm256_storeu_ps(a.as_mut_ptr().add(idx), result);
        }

        // Handle remaining elements
        for i in (chunks * 8)..len {
            a[i] += scale * b[i];
        }

        Ok(())
    }

    /// SIMD-optimized dot product
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_product_avx2(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(anyhow!("Vectors must have the same length"));
        }

        let len = a.len();
        let chunks = len / 8;
        let mut result_vec = _mm256_setzero_ps();

        for i in 0..chunks {
            let idx = i * 8;
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(idx));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(idx));
            let prod = _mm256_mul_ps(a_vec, b_vec);
            result_vec = _mm256_add_ps(result_vec, prod);
        }

        // Horizontal sum
        let result_array: [f32; 8] = std::mem::transmute(result_vec);
        let mut result = result_array.iter().sum::<f32>();

        // Add remaining elements
        for i in (chunks * 8)..len {
            result += a[i] * b[i];
        }

        Ok(result)
    }

    /// Fallback implementations for non-x86 architectures
    pub fn adam_update_fallback(
        &self,
        params: &mut [f32],
        gradients: &[f32],
        momentum: &mut [f32],
        velocity: &mut [f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        step: i32,
    ) -> Result<()> {
        if params.len() != gradients.len()
            || params.len() != momentum.len()
            || params.len() != velocity.len()
        {
            return Err(anyhow!("All arrays must have the same length"));
        }

        let bias_correction1 = 1.0 - beta1.powi(step);
        let bias_correction2 = 1.0 - beta2.powi(step);
        let corrected_lr = lr * (bias_correction2.sqrt() / bias_correction1);

        for i in 0..params.len() {
            let g = gradients[i];
            let m = momentum[i];
            let v = velocity[i];

            let m_new = beta1 * m + (1.0 - beta1) * g;
            let v_new = beta2 * v + (1.0 - beta2) * g * g;

            momentum[i] = m_new;
            velocity[i] = v_new;
            params[i] -= corrected_lr * m_new / (v_new.sqrt() + eps);
        }

        Ok(())
    }

    /// Auto-dispatch to best available implementation
    pub fn adam_update(
        &self,
        params: &mut [f32],
        gradients: &[f32],
        momentum: &mut [f32],
        velocity: &mut [f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        step: i32,
    ) -> Result<()> {
        if params.len() < self.config.min_vector_size {
            return self.adam_update_fallback(
                params, gradients, momentum, velocity, lr, beta1, beta2, eps, step,
            );
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if self.config.enable_avx2 && is_x86_feature_detected!("avx2") {
                return unsafe {
                    self.adam_update_avx2(
                        params, gradients, momentum, velocity, lr, beta1, beta2, eps, step,
                    )
                };
            }
        }

        self.adam_update_fallback(
            params, gradients, momentum, velocity, lr, beta1, beta2, eps, step,
        )
    }

    /// Auto-dispatch AdamW
    pub fn adamw_update(
        &self,
        params: &mut [f32],
        gradients: &[f32],
        momentum: &mut [f32],
        velocity: &mut [f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        step: i32,
    ) -> Result<()> {
        if params.len() < self.config.min_vector_size {
            // Fallback implementation
            let bias_correction1 = 1.0 - beta1.powi(step);
            let bias_correction2 = 1.0 - beta2.powi(step);
            let corrected_lr = lr * (bias_correction2.sqrt() / bias_correction1);

            for i in 0..params.len() {
                let p = params[i];
                let g = gradients[i];
                let m = momentum[i];
                let v = velocity[i];

                let p_decayed = p * (1.0 - lr * weight_decay);
                let m_new = beta1 * m + (1.0 - beta1) * g;
                let v_new = beta2 * v + (1.0 - beta2) * g * g;

                momentum[i] = m_new;
                velocity[i] = v_new;
                params[i] = p_decayed - corrected_lr * m_new / (v_new.sqrt() + eps);
            }
            return Ok(());
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if self.config.enable_avx2 && is_x86_feature_detected!("avx2") {
                return unsafe {
                    self.adamw_update_avx2(
                        params,
                        gradients,
                        momentum,
                        velocity,
                        lr,
                        beta1,
                        beta2,
                        eps,
                        weight_decay,
                        step,
                    )
                };
            }
        }

        // Fallback
        self.adamw_update(
            params,
            gradients,
            momentum,
            velocity,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            step,
        )
    }

    /// Get performance statistics
    pub fn get_performance_info(&self) -> SIMDPerformanceInfo {
        SIMDPerformanceInfo {
            avx2_available: {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    is_x86_feature_detected!("avx2")
                }
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    false
                }
            },
            avx512_available: {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    is_x86_feature_detected!("avx512f")
                }
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    false
                }
            },
            neon_available: cfg!(target_arch = "aarch64"),
            vector_width: {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    if is_x86_feature_detected!("avx2") {
                        8
                    } else {
                        1
                    }
                }
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    1
                }
            },
            recommended_min_size: self.config.min_vector_size,
        }
    }
}

impl Default for SIMDOptimizer {
    fn default() -> Self {
        Self::new(SIMDOptimizer::detect_capabilities())
    }
}

/// SIMD performance information
#[derive(Debug, Clone)]
pub struct SIMDPerformanceInfo {
    pub avx2_available: bool,
    pub avx512_available: bool,
    pub neon_available: bool,
    pub vector_width: usize,
    pub recommended_min_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_config_detection() {
        let config = SIMDOptimizer::detect_capabilities();
        // Test will pass regardless of actual hardware capabilities
        assert!(config.min_vector_size > 0);
    }

    #[test]
    fn test_adam_update_fallback() {
        let optimizer = SIMDOptimizer::default();
        let mut params = vec![1.0, 2.0, 3.0, 4.0];
        let gradients = vec![0.1, 0.2, 0.3, 0.4];
        let mut momentum = vec![0.0; 4];
        let mut velocity = vec![0.0; 4];

        optimizer
            .adam_update_fallback(
                &mut params,
                &gradients,
                &mut momentum,
                &mut velocity,
                0.001,
                0.9,
                0.999,
                1e-8,
                1,
            )
            .unwrap();

        // Check that parameters were updated
        assert!(params[0] < 1.0);
        assert!(momentum[0] > 0.0);
        assert!(velocity[0] > 0.0);
    }

    #[test]
    fn test_auto_dispatch_adam() {
        let optimizer = SIMDOptimizer::default();
        let mut params = vec![1.0; 16];
        let gradients = vec![0.1; 16];
        let mut momentum = vec![0.0; 16];
        let mut velocity = vec![0.0; 16];

        optimizer
            .adam_update(
                &mut params,
                &gradients,
                &mut momentum,
                &mut velocity,
                0.001,
                0.9,
                0.999,
                1e-8,
                1,
            )
            .unwrap();

        // Verify update occurred
        assert!(params.iter().all(|&p| p < 1.0));
        assert!(momentum.iter().all(|&m| m > 0.0));
    }

    #[test]
    fn test_performance_info() {
        let optimizer = SIMDOptimizer::default();
        let info = optimizer.get_performance_info();

        assert!(info.vector_width > 0);
        assert!(info.recommended_min_size > 0);
    }

    #[test]
    fn test_vector_operations() {
        let optimizer = SIMDOptimizer::default();
        let mut a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![0.5, 0.5, 0.5, 0.5];

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    optimizer.vector_add_avx2(&mut a, &b, 2.0).unwrap();
                }
                assert_eq!(a, vec![2.0, 3.0, 4.0, 5.0]);
            }
        }
    }

    #[test]
    fn test_dot_product() {
        let optimizer = SIMDOptimizer::default();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe {
                    let result = optimizer.dot_product_avx2(&a, &b).unwrap();
                    assert_eq!(result, 10.0);
                }
            }
        }
    }
}
