// Copyright (c) 2024 TrustformeRS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CPU feature detection and capabilities for SIMD optimization
//!
//! This module provides runtime detection of SIMD features across different
//! architectures including x86/x86_64, ARM, and RISC-V.

#[cfg(target_arch = "riscv64")]
use std::{fs, io::Read};

/// CPU feature detection and capabilities
#[derive(Debug, Clone, Copy)]
pub struct CpuFeatures {
    // x86/x86_64 features
    pub sse2: bool,
    pub sse3: bool,
    pub sse4_1: bool,
    pub sse4_2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512f: bool,
    pub avx512vl: bool,
    pub avx512bw: bool,
    pub avx512dq: bool,
    pub fma: bool,

    // ARM features
    pub neon: bool,
    pub sve: bool,
    pub sve2: bool,

    // RISC-V features
    pub rvv: bool,
    pub rvv_vlen: usize,
}

impl CpuFeatures {
    pub fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            Self {
                sse2: is_x86_feature_detected!("sse2"),
                sse3: is_x86_feature_detected!("sse3"),
                sse4_1: is_x86_feature_detected!("sse4.1"),
                sse4_2: is_x86_feature_detected!("sse4.2"),
                avx: is_x86_feature_detected!("avx"),
                avx2: is_x86_feature_detected!("avx2"),
                avx512f: is_x86_feature_detected!("avx512f"),
                avx512vl: is_x86_feature_detected!("avx512vl"),
                avx512bw: is_x86_feature_detected!("avx512bw"),
                avx512dq: is_x86_feature_detected!("avx512dq"),
                fma: is_x86_feature_detected!("fma"),
                neon: false,
                sve: false,
                sve2: false,
                rvv: false,
                rvv_vlen: 0,
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                sse2: false,
                sse3: false,
                sse4_1: false,
                sse4_2: false,
                avx: false,
                avx2: false,
                avx512f: false,
                avx512vl: false,
                avx512bw: false,
                avx512dq: false,
                fma: false,
                neon: std::arch::is_aarch64_feature_detected!("neon"),
                sve: std::arch::is_aarch64_feature_detected!("sve"),
                sve2: std::arch::is_aarch64_feature_detected!("sve2"),
                rvv: false,
                rvv_vlen: 0,
            }
        }
        #[cfg(target_arch = "riscv64")]
        {
            // Note: RISC-V feature detection is not yet stable in std::arch
            // We would need a custom implementation for RVV detection
            Self {
                sse2: false,
                sse3: false,
                sse4_1: false,
                sse4_2: false,
                avx: false,
                avx2: false,
                avx512f: false,
                avx512vl: false,
                avx512bw: false,
                avx512dq: false,
                fma: false,
                neon: false,
                sve: false,
                sve2: false,
                rvv: Self::detect_rvv(),
                rvv_vlen: Self::get_rvv_vlen(),
            }
        }
        #[cfg(not(any(
            target_arch = "x86",
            target_arch = "x86_64",
            target_arch = "aarch64",
            target_arch = "riscv64"
        )))]
        {
            Self {
                sse2: false,
                sse3: false,
                sse4_1: false,
                sse4_2: false,
                avx: false,
                avx2: false,
                avx512f: false,
                avx512vl: false,
                avx512bw: false,
                avx512dq: false,
                fma: false,
                neon: false,
                sve: false,
                sve2: false,
                rvv: false,
                rvv_vlen: 0,
            }
        }
    }

    #[cfg(target_arch = "riscv64")]
    fn detect_rvv() -> bool {
        // Try to detect RVV through multiple methods
        Self::detect_rvv_cpuinfo()
            .or_else(|| Self::detect_rvv_auxv())
            .or_else(|| Self::detect_rvv_runtime())
            .unwrap_or(false)
    }

    #[cfg(target_arch = "riscv64")]
    fn get_rvv_vlen() -> usize {
        // Try to get RISC-V vector length from various sources
        Self::get_rvv_vlen_cpuinfo()
            .or_else(|| Self::get_rvv_vlen_auxv())
            .or_else(|| Self::get_rvv_vlen_runtime())
            .unwrap_or(128) // Default to 128-bit if detection fails
    }

    #[cfg(target_arch = "riscv64")]
    fn detect_rvv_cpuinfo() -> Option<bool> {
        // Read /proc/cpuinfo to detect RVV support
        if let Ok(mut file) = fs::File::open("/proc/cpuinfo") {
            let mut contents = String::new();
            if file.read_to_string(&mut contents).is_ok() {
                // Look for RVV extensions in ISA string
                for line in contents.lines() {
                    if line.starts_with("isa\t\t:") {
                        let isa = line.split(':').nth(1)?.trim();
                        // Check for vector extension markers
                        if isa.contains("_v") || isa.contains("_zvl") {
                            return Some(true);
                        }
                    }
                }
            }
        }
        None
    }

    #[cfg(target_arch = "riscv64")]
    fn detect_rvv_auxv() -> Option<bool> {
        // Try to read auxiliary vector for hardware capabilities
        // This is platform-specific and may not be available on all systems
        if let Ok(mut file) = fs::File::open("/proc/self/auxv") {
            let mut buffer = Vec::new();
            if file.read_to_end(&mut buffer).is_ok() {
                // Parse auxv entries (simplified)
                // In a real implementation, you'd properly parse AT_HWCAP entries
                // For now, just return None to indicate we couldn't detect via auxv
            }
        }
        None
    }

    #[cfg(target_arch = "riscv64")]
    fn detect_rvv_runtime() -> Option<bool> {
        // Try to detect RVV at runtime by executing a simple vector instruction
        // This is risky and should be done carefully with signal handling
        // For safety, we'll return None here and rely on other methods
        None
    }

    #[cfg(target_arch = "riscv64")]
    fn get_rvv_vlen_cpuinfo() -> Option<usize> {
        // Try to extract VLEN from /proc/cpuinfo
        if let Ok(mut file) = fs::File::open("/proc/cpuinfo") {
            let mut contents = String::new();
            if file.read_to_string(&mut contents).is_ok() {
                for line in contents.lines() {
                    if line.starts_with("isa\t\t:") {
                        let isa = line.split(':').nth(1)?.trim();
                        // Look for Zvl* extensions that indicate vector length
                        if let Some(zvl_pos) = isa.find("_zvl") {
                            let remaining = &isa[zvl_pos + 4..];
                            if let Some(end) = remaining.find('_').or_else(|| Some(remaining.len()))
                            {
                                let vlen_str = &remaining[..end];
                                if let Ok(vlen) = vlen_str.parse::<usize>() {
                                    return Some(vlen);
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    #[cfg(target_arch = "riscv64")]
    fn get_rvv_vlen_auxv() -> Option<usize> {
        // Try to get VLEN from auxiliary vector
        // Implementation would depend on platform-specific HWCAP bits
        None
    }

    #[cfg(target_arch = "riscv64")]
    fn get_rvv_vlen_runtime() -> Option<usize> {
        // Try to determine VLEN at runtime
        // This would require executing vector instructions which is risky
        None
    }

    #[cfg(not(target_arch = "riscv64"))]
    #[allow(dead_code)]
    fn detect_rvv() -> bool {
        false
    }

    #[cfg(not(target_arch = "riscv64"))]
    #[allow(dead_code)]
    fn get_rvv_vlen() -> usize {
        0
    }

    pub fn best_simd_width(&self) -> usize {
        if self.avx512f && self.avx512vl {
            16 // 512-bit / 32-bit = 16 floats
        } else if self.avx2 || self.avx {
            8 // 256-bit / 32-bit = 8 floats (AVX/AVX2)
        } else if self.sse4_2 || self.sse4_1 || self.neon {
            4 // 128-bit / 32-bit = 4 floats (SSE/NEON)
        } else if self.sve && self.sve2 {
            8 // Conservative estimate for SVE
        } else if self.rvv && self.rvv_vlen > 0 {
            self.rvv_vlen / 32 // rvv_vlen is in bits, divide by 32 for f32 count
        } else {
            1 // Scalar
        }
    }

    pub fn best_instruction_set(&self) -> &'static str {
        if self.avx512f && self.avx512vl && self.avx512bw && self.avx512dq {
            "avx512"
        } else if self.avx2 && self.fma {
            "avx2_fma"
        } else if self.avx2 {
            "avx2"
        } else if self.avx {
            "avx"
        } else if self.sse4_2 {
            "sse4.2"
        } else if self.sse4_1 {
            "sse4.1"
        } else if self.sve2 {
            "sve2"
        } else if self.sve {
            "sve"
        } else if self.neon {
            "neon"
        } else if self.rvv {
            "rvv"
        } else {
            "scalar"
        }
    }
}
