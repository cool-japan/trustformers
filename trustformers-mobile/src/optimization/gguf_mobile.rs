//! Mobile-Optimized GGUF Quantization
//!
//! This module provides mobile-specific optimizations for GGUF (GPT-Generated Unified Format)
//! quantization, including:
//! - Battery-aware quantization selection
//! - Memory-constrained GGUF loading
//! - Thermal-adaptive quality adjustment
//! - Progressive loading for large models
//! - Hardware-specific GGUF optimizations

use serde::{Deserialize, Serialize};
use std::path::Path;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::quantization::{
    AdvancedGGMLQuantizer, GGMLQuantType, KQuantConfig, KQuantType, KQuantizer,
};
use trustformers_core::Tensor;

use crate::device_info::{MobileDeviceInfo, ThermalState};

/// Mobile GGUF configuration optimized for device constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileGGUFConfig {
    /// Target quantization type
    pub quant_type: MobileGGUFType,

    /// Enable battery-aware quantization adjustment
    pub battery_aware: bool,

    /// Enable thermal-aware quality adjustment
    pub thermal_aware: bool,

    /// Maximum memory budget (MB)
    pub max_memory_mb: usize,

    /// Enable progressive loading for large models
    pub progressive_loading: bool,

    /// Block size for progressive loading
    pub progressive_block_size: usize,

    /// Enable hardware-specific optimizations
    pub hardware_optimized: bool,

    /// Minimum battery level for high-quality quantization (%)
    pub min_battery_for_hq: f32,

    /// Thermal threshold for quality downgrade
    pub thermal_threshold: ThermalState,
}

impl Default for MobileGGUFConfig {
    fn default() -> Self {
        Self {
            quant_type: MobileGGUFType::Q4_K,
            battery_aware: true,
            thermal_aware: true,
            max_memory_mb: 512,
            progressive_loading: true,
            progressive_block_size: 1024 * 1024, // 1MB blocks
            hardware_optimized: true,
            min_battery_for_hq: 30.0,
            thermal_threshold: ThermalState::Nominal,
        }
    }
}

/// Mobile-optimized GGUF quantization types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MobileGGUFType {
    /// Q2_K: Ultra-low memory (2.5625 bpw) - for low battery/memory
    Q2_K,
    /// Q3_K: Low memory (3.4375 bpw) - balanced for mobile
    Q3_K,
    /// Q4_K: Recommended (4.5 bpw) - best quality/size for mobile
    Q4_K,
    /// Q5_0: High quality (5.5 bpw) - for high-end devices
    Q5_0,
    /// Q6_K: Very high quality (6.5 bpw) - near-lossless
    Q6_K,
}

impl MobileGGUFType {
    /// Get corresponding core KQuantType
    pub fn to_kquant_type(&self) -> Option<KQuantType> {
        match self {
            MobileGGUFType::Q2_K => Some(KQuantType::Q2_K),
            MobileGGUFType::Q3_K => Some(KQuantType::Q3_K),
            MobileGGUFType::Q4_K => Some(KQuantType::Q4_K),
            _ => None,
        }
    }

    /// Get corresponding GGML type
    pub fn to_ggml_type(&self) -> Option<GGMLQuantType> {
        match self {
            MobileGGUFType::Q5_0 => Some(GGMLQuantType::Q5_0),
            MobileGGUFType::Q6_K => Some(GGMLQuantType::Q6K),
            _ => None,
        }
    }

    /// Get bits per weight
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            MobileGGUFType::Q2_K => 2.5625,
            MobileGGUFType::Q3_K => 3.4375,
            MobileGGUFType::Q4_K => 4.5,
            MobileGGUFType::Q5_0 => 5.5,
            MobileGGUFType::Q6_K => 6.5,
        }
    }

    /// Estimate memory usage for a model (MB)
    pub fn estimate_memory_mb(&self, num_parameters: usize) -> usize {
        let bits = self.bits_per_weight();
        let bytes = (num_parameters as f32 * bits / 8.0) as usize;
        (bytes / (1024 * 1024)) + 50 // Add 50MB overhead
    }

    /// Get recommended type for device
    pub fn recommend_for_device(device_info: &MobileDeviceInfo) -> Self {
        let available_memory_mb = device_info.memory_info.available_mb;
        let is_high_end = matches!(
            device_info.performance_scores.tier,
            crate::device_info::PerformanceTier::High
                | crate::device_info::PerformanceTier::VeryHigh
                | crate::device_info::PerformanceTier::Flagship
        );

        // Select based on memory and performance
        if available_memory_mb < 512 {
            MobileGGUFType::Q2_K // Ultra-low memory
        } else if available_memory_mb < 1024 {
            MobileGGUFType::Q3_K // Low memory
        } else if is_high_end {
            if available_memory_mb > 2048 {
                MobileGGUFType::Q5_0 // High-end with lots of memory
            } else {
                MobileGGUFType::Q4_K // High-end
            }
        } else {
            MobileGGUFType::Q4_K // Default recommended
        }
    }

    /// Downgrade quality for constraints
    pub fn downgrade(&self) -> Self {
        match self {
            MobileGGUFType::Q6_K => MobileGGUFType::Q5_0,
            MobileGGUFType::Q5_0 => MobileGGUFType::Q4_K,
            MobileGGUFType::Q4_K => MobileGGUFType::Q3_K,
            MobileGGUFType::Q3_K => MobileGGUFType::Q2_K,
            MobileGGUFType::Q2_K => MobileGGUFType::Q2_K, // Can't go lower
        }
    }

    /// Upgrade quality if resources allow
    pub fn upgrade(&self) -> Self {
        match self {
            MobileGGUFType::Q2_K => MobileGGUFType::Q3_K,
            MobileGGUFType::Q3_K => MobileGGUFType::Q4_K,
            MobileGGUFType::Q4_K => MobileGGUFType::Q5_0,
            MobileGGUFType::Q5_0 => MobileGGUFType::Q6_K,
            MobileGGUFType::Q6_K => MobileGGUFType::Q6_K, // Already max
        }
    }
}

/// Mobile GGUF quantizer with adaptive quality
pub struct MobileGGUFQuantizer {
    config: MobileGGUFConfig,
    current_quality: MobileGGUFType,
    battery_level: Option<f32>,
    thermal_state: Option<ThermalState>,
}

impl MobileGGUFQuantizer {
    /// Create new mobile GGUF quantizer
    pub fn new(config: MobileGGUFConfig) -> Result<Self> {
        Ok(Self {
            current_quality: config.quant_type,
            config,
            battery_level: None,
            thermal_state: None,
        })
    }

    /// Update battery level for adaptive quality
    pub fn set_battery_level(&mut self, level: f32) {
        self.battery_level = Some(level);
    }

    /// Update thermal state for adaptive quality
    pub fn set_thermal_state(&mut self, state: ThermalState) {
        self.thermal_state = Some(state);
    }

    /// Adjust quality based on device state
    pub fn adjust_quality(&mut self) -> Result<()> {
        let mut target_quality = self.config.quant_type;

        // Check battery level
        if self.config.battery_aware {
            if let Some(level) = self.battery_level {
                if level < self.config.min_battery_for_hq {
                    // Low battery - downgrade quality
                    target_quality = target_quality.downgrade();
                }
            }
        }

        // Check thermal state
        if self.config.thermal_aware {
            if let Some(state) = self.thermal_state {
                // Downgrade on high thermal states
                let should_downgrade = matches!(
                    state,
                    ThermalState::Serious | ThermalState::Critical | ThermalState::Emergency
                );

                if should_downgrade || state == self.config.thermal_threshold {
                    // Hot device - downgrade quality
                    target_quality = target_quality.downgrade();
                }
            }
        }

        self.current_quality = target_quality;
        Ok(())
    }

    /// Get K-quant quantizer configuration for current quality
    pub fn get_kquant_config(&self) -> Option<KQuantConfig> {
        self.current_quality.to_kquant_type().map(|quant_type| KQuantConfig {
            quant_type,
            ..Default::default()
        })
    }

    /// Get GGML quantizer type for current quality
    pub fn get_ggml_type(&self) -> Option<GGMLQuantType> {
        self.current_quality.to_ggml_type()
    }

    /// Create K-quant quantizer for current quality
    ///
    /// # Note
    /// This requires the current quality to be Q2_K, Q3_K, or Q4_K
    ///
    /// # Example
    /// ```ignore
    /// use trustformers_mobile::{MobileGGUFQuantizer, MobileGGUFConfig, MobileGGUFType};
    /// use trustformers_core::Tensor;
    ///
    /// let config = MobileGGUFConfig {
    ///     quant_type: MobileGGUFType::Q4_K,
    ///     ..Default::default()
    /// };
    /// let mut quantizer = MobileGGUFQuantizer::new(config)?;
    /// if let Some(kquant) = quantizer.create_kquant_quantizer()? {
    ///     let tensor = Tensor::randn(&[128, 128])?;
    ///     let quantized = kquant.quantize(&tensor)?;
    ///     let dequantized = kquant.dequantize(&quantized)?;
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn create_kquant_quantizer(&mut self) -> Result<Option<KQuantizer>> {
        // Adjust quality based on device state
        self.adjust_quality()?;

        if let Some(config) = self.get_kquant_config() {
            Ok(Some(KQuantizer::new(config)?))
        } else {
            Ok(None)
        }
    }

    /// Create GGML quantizer for current quality
    ///
    /// # Note
    /// This requires the current quality to be Q5_0 or Q6_K
    ///
    /// # Example
    /// ```ignore
    /// use trustformers_mobile::{MobileGGUFQuantizer, MobileGGUFConfig, MobileGGUFType};
    /// use trustformers_core::Tensor;
    ///
    /// let config = MobileGGUFConfig {
    ///     quant_type: MobileGGUFType::Q5_0,
    ///     ..Default::default()
    /// };
    /// let mut quantizer = MobileGGUFQuantizer::new(config)?;
    /// if let Some(ggml) = quantizer.create_ggml_quantizer()? {
    ///     let tensor = Tensor::randn(&[128, 128])?;
    ///     let quantized = ggml.quantize(&tensor)?;
    ///     let dequantized = ggml.dequantize(&quantized)?;
    /// }
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn create_ggml_quantizer(&mut self) -> Result<Option<AdvancedGGMLQuantizer>> {
        // Adjust quality based on device state
        self.adjust_quality()?;

        if let Some(ggml_type) = self.get_ggml_type() {
            Ok(Some(AdvancedGGMLQuantizer::new(ggml_type)))
        } else {
            Ok(None)
        }
    }

    /// Get current quality level
    pub fn current_quality(&self) -> MobileGGUFType {
        self.current_quality
    }

    /// Get statistics
    pub fn get_stats(&self) -> MobileGGUFStats {
        MobileGGUFStats {
            current_quality: self.current_quality,
            battery_aware: self.config.battery_aware,
            thermal_aware: self.config.thermal_aware,
            progressive_loading: self.config.progressive_loading,
        }
    }
}

/// Mobile GGUF statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileGGUFStats {
    pub current_quality: MobileGGUFType,
    pub battery_aware: bool,
    pub thermal_aware: bool,
    pub progressive_loading: bool,
}

/// Utilities for mobile GGUF optimization
pub struct MobileGGUFUtils;

impl MobileGGUFUtils {
    /// Select optimal GGUF type for device
    pub fn select_optimal_type(
        device_info: &MobileDeviceInfo,
        model_params: usize,
    ) -> MobileGGUFType {
        let available_memory = device_info.memory_info.available_mb;

        // Try each type from highest to lowest quality
        for quant_type in [
            MobileGGUFType::Q6_K,
            MobileGGUFType::Q5_0,
            MobileGGUFType::Q4_K,
            MobileGGUFType::Q3_K,
            MobileGGUFType::Q2_K,
        ] {
            let required_memory = quant_type.estimate_memory_mb(model_params);

            // Leave 20% headroom
            if (required_memory as f32) < (available_memory as f32 * 0.8) {
                return quant_type;
            }
        }

        // Fallback to ultra-low memory
        MobileGGUFType::Q2_K
    }

    /// Estimate model size after GGUF quantization
    pub fn estimate_quantized_size(original_size_mb: usize, quant_type: MobileGGUFType) -> usize {
        let compression_ratio = 32.0 / quant_type.bits_per_weight();
        (original_size_mb as f32 / compression_ratio) as usize
    }

    /// Check if device can handle GGUF type
    pub fn can_handle_type(
        device_info: &MobileDeviceInfo,
        quant_type: MobileGGUFType,
        model_params: usize,
    ) -> bool {
        let required_memory = quant_type.estimate_memory_mb(model_params);
        let available_memory = device_info.memory_info.available_mb;

        // Require at least 20% headroom
        required_memory < (available_memory as f32 * 0.8) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mobile_gguf_types() {
        assert_eq!(MobileGGUFType::Q2_K.bits_per_weight(), 2.5625);
        assert_eq!(MobileGGUFType::Q4_K.bits_per_weight(), 4.5);
        assert_eq!(MobileGGUFType::Q6_K.bits_per_weight(), 6.5);
    }

    #[test]
    fn test_quality_adjustment() {
        let q4 = MobileGGUFType::Q4_K;
        assert_eq!(q4.downgrade(), MobileGGUFType::Q3_K);
        assert_eq!(q4.upgrade(), MobileGGUFType::Q5_0);

        let q2 = MobileGGUFType::Q2_K;
        assert_eq!(q2.downgrade(), MobileGGUFType::Q2_K); // Can't go lower

        let q6 = MobileGGUFType::Q6_K;
        assert_eq!(q6.upgrade(), MobileGGUFType::Q6_K); // Already max
    }

    #[test]
    fn test_memory_estimation() {
        let params = 7_000_000_000; // 7B parameters

        let q2_size = MobileGGUFType::Q2_K.estimate_memory_mb(params);
        let q4_size = MobileGGUFType::Q4_K.estimate_memory_mb(params);
        let q6_size = MobileGGUFType::Q6_K.estimate_memory_mb(params);

        // Q2 should be smaller than Q4, Q4 smaller than Q6
        assert!(q2_size < q4_size);
        assert!(q4_size < q6_size);

        // Rough size checks (7B model)
        assert!(q2_size < 3000); // Q2_K ~2.5GB
        assert!(q4_size < 5000); // Q4_K ~4.5GB
        assert!(q6_size < 7000); // Q6_K ~6.5GB
    }

    #[test]
    fn test_optimal_selection() {
        let params = 1_000_000_000; // 1B parameters

        // High memory device
        let mut device_info = MobileDeviceInfo::default();
        device_info.memory_info.available_mb = 4096;
        let optimal = MobileGGUFUtils::select_optimal_type(&device_info, params);
        // Should select high quality for high-memory device
        assert!(matches!(
            optimal,
            MobileGGUFType::Q5_0 | MobileGGUFType::Q6_K
        ));

        // Low memory device
        device_info.memory_info.available_mb = 512;
        let optimal = MobileGGUFUtils::select_optimal_type(&device_info, params);
        // Should select low quality for low-memory device
        assert!(matches!(
            optimal,
            MobileGGUFType::Q2_K | MobileGGUFType::Q3_K
        ));
    }
}
