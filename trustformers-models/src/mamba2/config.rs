/// Configuration for the Mamba-2 State Space Model.
///
/// Mamba-2 uses SSD (State Space Duality) with multi-head structured matrices
/// to replace the recurrence with parallel scan for training efficiency.
#[derive(Debug, Clone)]
pub struct Mamba2Config {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension (e.g. 2560 for 2.7B)
    pub d_model: usize,
    /// Number of Mamba blocks (e.g. 64 for 2.7B)
    pub n_layer: usize,
    /// SSM state dimension (e.g. 128)
    pub d_state: usize,
    /// Local convolution width (e.g. 4)
    pub d_conv: usize,
    /// Expansion factor for inner dimension (e.g. 2)
    pub expand: usize,
    /// Number of SSM heads (e.g. 24 for 2.7B)
    pub nheads: usize,
    /// Head dimension = d_model * expand / nheads
    pub headdim: usize,
    /// Parallel scan chunk size (e.g. 256)
    pub chunk_size: usize,
    /// Epsilon for RMSNorm
    pub rms_norm_eps: f64,
    /// Whether to tie input/output embeddings
    pub tie_embeddings: bool,
}

impl Mamba2Config {
    /// Mamba-2 2.7B configuration.
    ///
    /// Based on the official Mamba-2 release:
    ///   d_model=2560, n_layer=64, d_state=128, d_conv=4, expand=2
    ///   nheads=80 (so inner_dim=5120, headdim=5120/80=64)
    pub fn mamba2_2_7b() -> Self {
        let d_model = 2560usize;
        let expand = 2usize;
        // nheads=80 gives inner_dim=5120, headdim=64 (exact integer)
        let nheads = 80usize;
        let headdim = d_model * expand / nheads; // = 64
        Self {
            vocab_size: 50280,
            d_model,
            n_layer: 64,
            d_state: 128,
            d_conv: 4,
            expand,
            nheads,
            headdim,
            chunk_size: 256,
            rms_norm_eps: 1e-5,
            tie_embeddings: true,
        }
    }

    /// Small test configuration for unit testing.
    ///
    /// Uses: d_model=64, n_layer=2, d_state=16, d_conv=4, expand=2, nheads=4
    pub fn small_test() -> Self {
        let d_model = 64usize;
        let expand = 2usize;
        let nheads = 4usize;
        let headdim = d_model * expand / nheads;
        Self {
            vocab_size: 256,
            d_model,
            n_layer: 2,
            d_state: 16,
            d_conv: 4,
            expand,
            nheads,
            headdim,
            chunk_size: 64,
            rms_norm_eps: 1e-5,
            tie_embeddings: false,
        }
    }

    /// Inner dimension = d_model * expand
    pub fn inner_dim(&self) -> usize {
        self.d_model * self.expand
    }

    /// Verify that headdim is consistent with d_model, expand, and nheads.
    pub fn validate(&self) -> bool {
        self.headdim == self.inner_dim() / self.nheads
            && self.inner_dim().is_multiple_of(self.nheads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mamba2_2_7b_vocab_size() {
        let cfg = Mamba2Config::mamba2_2_7b();
        assert_eq!(cfg.vocab_size, 50280);
    }

    #[test]
    fn test_mamba2_2_7b_d_model() {
        let cfg = Mamba2Config::mamba2_2_7b();
        assert_eq!(cfg.d_model, 2560);
    }

    #[test]
    fn test_mamba2_2_7b_n_layer() {
        let cfg = Mamba2Config::mamba2_2_7b();
        assert_eq!(cfg.n_layer, 64);
    }

    #[test]
    fn test_mamba2_2_7b_nheads() {
        let cfg = Mamba2Config::mamba2_2_7b();
        assert_eq!(cfg.nheads, 80);
    }

    #[test]
    fn test_mamba2_2_7b_headdim() {
        let cfg = Mamba2Config::mamba2_2_7b();
        assert_eq!(cfg.headdim, 64); // 2560 * 2 / 80 = 64
    }

    #[test]
    fn test_mamba2_2_7b_d_state() {
        let cfg = Mamba2Config::mamba2_2_7b();
        assert_eq!(cfg.d_state, 128);
    }

    #[test]
    fn test_mamba2_2_7b_d_conv() {
        let cfg = Mamba2Config::mamba2_2_7b();
        assert_eq!(cfg.d_conv, 4);
    }

    #[test]
    fn test_mamba2_2_7b_chunk_size() {
        let cfg = Mamba2Config::mamba2_2_7b();
        assert_eq!(cfg.chunk_size, 256);
    }

    #[test]
    fn test_mamba2_2_7b_tie_embeddings() {
        let cfg = Mamba2Config::mamba2_2_7b();
        assert!(cfg.tie_embeddings);
    }

    #[test]
    fn test_mamba2_validate_passes_2_7b() {
        let cfg = Mamba2Config::mamba2_2_7b();
        assert!(cfg.validate());
    }

    #[test]
    fn test_mamba2_inner_dim_2_7b() {
        let cfg = Mamba2Config::mamba2_2_7b();
        assert_eq!(cfg.inner_dim(), 2560 * 2);
    }

    #[test]
    fn test_mamba2_small_test_config() {
        let cfg = Mamba2Config::small_test();
        assert_eq!(cfg.vocab_size, 256);
        assert_eq!(cfg.d_model, 64);
        assert_eq!(cfg.n_layer, 2);
        assert_eq!(cfg.nheads, 4);
    }

    #[test]
    fn test_mamba2_small_test_validate() {
        let cfg = Mamba2Config::small_test();
        assert!(cfg.validate());
    }

    #[test]
    fn test_mamba2_small_test_headdim() {
        let cfg = Mamba2Config::small_test();
        assert_eq!(cfg.headdim, 64 * 2 / 4); // = 32
    }

    #[test]
    fn test_mamba2_small_test_tie_embeddings_false() {
        let cfg = Mamba2Config::small_test();
        assert!(!cfg.tie_embeddings);
    }

    #[test]
    fn test_mamba2_validate_fails_inconsistent_headdim() {
        let cfg = Mamba2Config {
            headdim: 99,
            ..Mamba2Config::mamba2_2_7b()
        };
        assert!(!cfg.validate());
    }

    #[test]
    fn test_mamba2_inner_dim_small() {
        let cfg = Mamba2Config::small_test();
        assert_eq!(cfg.inner_dim(), 64 * 2);
    }

    #[test]
    fn test_mamba2_expand_factor() {
        let cfg = Mamba2Config::mamba2_2_7b();
        assert_eq!(cfg.expand, 2);
    }

    #[test]
    fn test_mamba2_rms_norm_eps() {
        let cfg = Mamba2Config::mamba2_2_7b();
        assert!(cfg.rms_norm_eps > 0.0);
        assert!(cfg.rms_norm_eps < 1e-3);
    }

    #[test]
    fn test_mamba2_lcg_values_in_range() {
        let mut s = 42u64;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (s % 1000) as f32 / 1000.0;
        assert!((0.0..1.0).contains(&v));
    }

    #[test]
    fn test_mamba2_chunk_size_small() {
        let cfg = Mamba2Config::small_test();
        assert_eq!(cfg.chunk_size, 64);
    }
}
