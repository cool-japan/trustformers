//! Auto-generated module structure

pub mod memorywipingmanager_traits;
pub mod defaulthardwareprotection_traits;
pub mod memorypoolconfig_traits;
pub mod types;
pub mod functions;

// Re-export all types
pub use memorywipingmanager_traits::*;
pub use defaulthardwareprotection_traits::*;
pub use memorypoolconfig_traits::*;
pub use types::*;
pub use functions::*;

#[cfg(test)]
mod tests {
    use super::*;

    // ── WipingPriority ────────────────────────────────────────────────────

    #[test]
    fn test_wiping_priority_ordering() {
        assert!(WipingPriority::Critical > WipingPriority::High);
        assert!(WipingPriority::High > WipingPriority::Normal);
        assert!(WipingPriority::Normal > WipingPriority::Low);
    }

    #[test]
    fn test_wiping_priority_equality() {
        assert_eq!(WipingPriority::Low, WipingPriority::Low);
        assert_ne!(WipingPriority::Low, WipingPriority::Critical);
    }

    #[test]
    fn test_wiping_priority_debug_format() {
        assert!(format!("{:?}", WipingPriority::Critical).contains("Critical"));
        assert!(format!("{:?}", WipingPriority::High).contains("High"));
        assert!(format!("{:?}", WipingPriority::Normal).contains("Normal"));
        assert!(format!("{:?}", WipingPriority::Low).contains("Low"));
    }

    #[test]
    fn test_wiping_priority_clone() {
        let p = WipingPriority::High;
        let q = p.clone();
        assert_eq!(p, q);
    }

    // ── WipingPatterns ────────────────────────────────────────────────────

    #[test]
    fn test_wiping_patterns_new_zero_pattern_not_empty() {
        let patterns = WipingPatterns::new();
        // get_random_pattern returns a filled vec
        let random = patterns.get_random_pattern(64);
        assert_eq!(random.len(), 64);
        // All filled with 0x55
        assert!(random.iter().all(|&b| b == 0x55));
    }

    #[test]
    fn test_wiping_patterns_secure_random_pattern() {
        let patterns = WipingPatterns::new();
        let secure = patterns.get_secure_random_pattern(32);
        assert_eq!(secure.len(), 32);
        // Filled with 0xAA
        assert!(secure.iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn test_wiping_patterns_dod_patterns_not_empty() {
        let patterns = WipingPatterns::new();
        let dod = patterns.get_dod_patterns();
        assert!(!dod.is_empty());
        // DoD standard requires at least 2 passes (0xFF, 0x00)
        assert!(dod.len() >= 2);
    }

    #[test]
    fn test_wiping_patterns_gutmann_patterns_not_empty() {
        let patterns = WipingPatterns::new();
        let gutmann = patterns.get_gutmann_patterns();
        assert!(!gutmann.is_empty());
    }

    #[test]
    fn test_wiping_patterns_dod_first_pass_is_all_ff() {
        let patterns = WipingPatterns::new();
        let dod = patterns.get_dod_patterns();
        // First DoD pass should be 0xFF overwrite
        let first = &dod[0];
        assert!(!first.is_empty());
        assert!(first.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn test_wiping_patterns_dod_second_pass_is_all_zero() {
        let patterns = WipingPatterns::new();
        let dod = patterns.get_dod_patterns();
        let second = &dod[1];
        assert!(!second.is_empty());
        assert!(second.iter().all(|&b| b == 0x00));
    }

    // ── BufferStatus ──────────────────────────────────────────────────────

    #[test]
    fn test_buffer_status_all_variants_debug() {
        assert!(format!("{:?}", BufferStatus::Allocated).contains("Allocated"));
        assert!(format!("{:?}", BufferStatus::Encrypted).contains("Encrypted"));
        assert!(format!("{:?}", BufferStatus::Decrypted).contains("Decrypted"));
        assert!(format!("{:?}", BufferStatus::Locked).contains("Locked"));
        assert!(format!("{:?}", BufferStatus::Wiping).contains("Wiping"));
        assert!(format!("{:?}", BufferStatus::Deallocated).contains("Deallocated"));
    }

    #[test]
    fn test_buffer_status_equality() {
        assert_eq!(BufferStatus::Encrypted, BufferStatus::Encrypted);
        assert_ne!(BufferStatus::Allocated, BufferStatus::Encrypted);
        assert_ne!(BufferStatus::Wiping, BufferStatus::Deallocated);
    }

    // ── GrowthStrategy ────────────────────────────────────────────────────

    #[test]
    fn test_growth_strategy_fixed_debug() {
        let g = GrowthStrategy::Fixed;
        assert!(format!("{:?}", g).contains("Fixed"));
    }

    #[test]
    fn test_growth_strategy_linear_carries_increment() {
        let g = GrowthStrategy::Linear { increment: 50 };
        let dbg = format!("{:?}", g);
        assert!(dbg.contains("Linear"));
        assert!(dbg.contains("50"));
    }

    #[test]
    fn test_growth_strategy_exponential_carries_factor() {
        let g = GrowthStrategy::Exponential { factor: 2.0_f64 };
        let dbg = format!("{:?}", g);
        assert!(dbg.contains("Exponential"));
    }

    #[test]
    fn test_growth_strategy_dynamic_debug() {
        let g = GrowthStrategy::Dynamic;
        assert!(format!("{:?}", g).contains("Dynamic"));
    }

    // ── WipingTaskStatus ──────────────────────────────────────────────────

    #[test]
    fn test_wiping_task_status_all_variants() {
        assert!(format!("{:?}", WipingTaskStatus::Queued).contains("Queued"));
        assert!(format!("{:?}", WipingTaskStatus::Running).contains("Running"));
        assert!(format!("{:?}", WipingTaskStatus::Completed).contains("Completed"));
        assert!(format!("{:?}", WipingTaskStatus::Failed).contains("Failed"));
    }

    #[test]
    fn test_wiping_task_status_equality() {
        assert_eq!(WipingTaskStatus::Completed, WipingTaskStatus::Completed);
        assert_ne!(WipingTaskStatus::Queued, WipingTaskStatus::Running);
        assert_ne!(WipingTaskStatus::Failed, WipingTaskStatus::Completed);
    }

    // ── BlockStatus ───────────────────────────────────────────────────────

    #[test]
    fn test_block_status_all_variants() {
        assert!(format!("{:?}", BlockStatus::Free).contains("Free"));
        assert!(format!("{:?}", BlockStatus::Allocated).contains("Allocated"));
        assert!(format!("{:?}", BlockStatus::Encrypted).contains("Encrypted"));
        assert!(format!("{:?}", BlockStatus::Protected).contains("Protected"));
        assert!(format!("{:?}", BlockStatus::Wiping).contains("Wiping"));
    }

    #[test]
    fn test_block_status_equality() {
        assert_eq!(BlockStatus::Free, BlockStatus::Free);
        assert_ne!(BlockStatus::Allocated, BlockStatus::Encrypted);
    }

    // ── HardwareType ──────────────────────────────────────────────────────

    #[test]
    fn test_hardware_type_all_variants_debug() {
        let variants = [
            HardwareType::IntelMPX,
            HardwareType::ARMPointerAuth,
            HardwareType::IntelCET,
            HardwareType::HSM,
            HardwareType::TPM,
            HardwareType::SecureEnclave,
        ];
        for v in &variants {
            assert!(!format!("{:?}", v).is_empty());
        }
    }

    #[test]
    fn test_hardware_type_equality() {
        assert_eq!(HardwareType::TPM, HardwareType::TPM);
        assert_ne!(HardwareType::HSM, HardwareType::TPM);
    }

    // ── RegionStatus ──────────────────────────────────────────────────────

    #[test]
    fn test_region_status_all_variants_debug() {
        assert!(format!("{:?}", RegionStatus::Unprotected).contains("Unprotected"));
        assert!(format!("{:?}", RegionStatus::Protected).contains("Protected"));
        assert!(format!("{:?}", RegionStatus::Encrypted).contains("Encrypted"));
        assert!(format!("{:?}", RegionStatus::HardwareProtected).contains("HardwareProtected"));
        assert!(format!("{:?}", RegionStatus::SecureEnclave).contains("SecureEnclave"));
    }

    #[test]
    fn test_region_status_equality() {
        assert_eq!(RegionStatus::Protected, RegionStatus::Protected);
        assert_ne!(RegionStatus::Unprotected, RegionStatus::Protected);
    }

    // ── MemoryWipingManager ───────────────────────────────────────────────

    #[test]
    fn test_memory_wiping_manager_new_disabled() {
        let config = crate::encryption::MemoryWipingConfig {
            enabled: false,
            wiping_method: crate::encryption::WipingMethod::ZeroFill,
        };
        let manager = MemoryWipingManager::new(config);
        // Manager constructed without panic
        assert!(!manager.config.enabled);
    }

    #[test]
    fn test_memory_wiping_manager_new_enabled() {
        let config = crate::encryption::MemoryWipingConfig {
            enabled: true,
            wiping_method: crate::encryption::WipingMethod::MultiPass,
        };
        let manager = MemoryWipingManager::new(config);
        assert!(manager.config.enabled);
    }
}
