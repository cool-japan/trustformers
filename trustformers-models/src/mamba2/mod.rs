pub mod config;
pub mod model;
pub mod tasks;

pub use config::Mamba2Config;
pub use model::{
    softplus, Mamba2Block, Mamba2Error, Mamba2ForCausalLM, Mamba2Model, Mamba2RmsNorm, Mamba2SSM,
};
pub use tasks::{Mamba2ForCausalLMHead, Mamba2ForSequenceClassification, Mamba2TaskError};

// ---------------------------------------------------------------------------
// Module-level tests (supplementing the 15 tests already in model.rs)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use crate::mamba2::{
        softplus, Mamba2Config, Mamba2ForCausalLM, Mamba2Model, Mamba2RmsNorm, Mamba2SSM,
    };

    // ── Helper ───────────────────────────────────────────────────────────────

    fn tiny_cfg() -> Mamba2Config {
        Mamba2Config::small_test()
    }

    // ── Config presets ────────────────────────────────────────────────────────

    #[test]
    fn test_mamba2_small_test_preset() {
        let cfg = Mamba2Config::small_test();
        assert!(cfg.d_model > 0, "d_model must be > 0");
        assert!(cfg.n_layer > 0, "n_layer must be > 0");
        assert!(cfg.d_state > 0, "d_state must be > 0");
        assert!(cfg.d_conv > 0, "d_conv must be > 0");
        assert!(cfg.validate(), "small_test config must be self-consistent");
    }

    #[test]
    fn test_mamba2_2_7b_preset_parameters() {
        let cfg = Mamba2Config::mamba2_2_7b();
        assert_eq!(cfg.d_model, 2560, "2.7B d_model must be 2560");
        assert_eq!(cfg.n_layer, 64, "2.7B must have 64 layers");
        assert_eq!(cfg.d_state, 128, "2.7B d_state must be 128");
        assert_eq!(cfg.d_conv, 4, "2.7B d_conv must be 4");
        assert_eq!(cfg.expand, 2, "2.7B expand must be 2");
        assert_eq!(cfg.vocab_size, 50280, "2.7B vocab_size must be 50280");
        assert!(cfg.validate(), "2.7B preset must be self-consistent");
    }

    // ── ssm_state_size (d_state) ──────────────────────────────────────────────

    #[test]
    fn test_mamba2_d_state_config() {
        let cfg = tiny_cfg();
        assert_eq!(cfg.d_state, 16, "small_test d_state must be 16");
        let big = Mamba2Config::mamba2_2_7b();
        assert_eq!(big.d_state, 128, "2.7B d_state must be 128");
    }

    // ── d_model ───────────────────────────────────────────────────────────────

    #[test]
    fn test_mamba2_d_model_config() {
        let cfg = tiny_cfg();
        assert_eq!(cfg.d_model, 64, "small_test d_model must be 64");
    }

    // ── d_inner = d_model * expand ────────────────────────────────────────────

    #[test]
    fn test_mamba2_inner_dim_formula() {
        let cfg = tiny_cfg();
        assert_eq!(
            cfg.inner_dim(),
            cfg.d_model * cfg.expand,
            "inner_dim must equal d_model * expand"
        );
    }

    // ── d_conv ────────────────────────────────────────────────────────────────

    #[test]
    fn test_mamba2_d_conv_config() {
        let cfg = tiny_cfg();
        assert_eq!(cfg.d_conv, 4, "small_test d_conv must be 4");
    }

    // ── nheads (num_heads) ────────────────────────────────────────────────────

    #[test]
    fn test_mamba2_nheads_config() {
        let cfg = tiny_cfg();
        assert_eq!(cfg.nheads, 4, "small_test nheads must be 4");
        let big = Mamba2Config::mamba2_2_7b();
        assert_eq!(big.nheads, 80, "2.7B nheads must be 80");
    }

    // ── expand factor ─────────────────────────────────────────────────────────

    #[test]
    fn test_mamba2_expand_factor() {
        let cfg = tiny_cfg();
        assert_eq!(cfg.expand, 2, "default expand must be 2");
        assert_eq!(
            Mamba2Config::mamba2_2_7b().expand,
            2,
            "2.7B expand must be 2"
        );
    }

    // ── chunk_size (SSD) ──────────────────────────────────────────────────────

    #[test]
    fn test_mamba2_chunk_size_config() {
        let cfg = tiny_cfg();
        assert_eq!(cfg.chunk_size, 64, "small_test chunk_size must be 64");
        let big = Mamba2Config::mamba2_2_7b();
        assert_eq!(big.chunk_size, 256, "2.7B chunk_size must be 256");
    }

    // ── headdim consistency ───────────────────────────────────────────────────

    #[test]
    fn test_mamba2_headdim_small_test() {
        let cfg = tiny_cfg();
        // headdim = d_model * expand / nheads = 64 * 2 / 4 = 32
        assert_eq!(cfg.headdim, 32, "small_test headdim must be 32");
        assert_eq!(
            cfg.headdim * cfg.nheads,
            cfg.inner_dim(),
            "headdim * nheads must equal inner_dim"
        );
    }

    // ── validate ─────────────────────────────────────────────────────────────

    #[test]
    fn test_mamba2_config_validate_consistency() {
        let mut cfg = tiny_cfg();
        assert!(cfg.validate(), "default config must pass validation");
        // Break the consistency: set headdim to an incorrect value
        cfg.headdim += 1;
        assert!(!cfg.validate(), "wrong headdim must fail validation");
    }

    // ── Model construction ────────────────────────────────────────────────────

    #[test]
    fn test_mamba2_model_construction() {
        let cfg = tiny_cfg();
        let model = Mamba2Model::new(&cfg);
        assert_eq!(
            model.num_layers(),
            cfg.n_layer,
            "num_layers must match config"
        );
    }

    // ── Output shape: seq_len preserved ──────────────────────────────────────

    #[test]
    fn test_mamba2_model_forward_output_seq_len() {
        let cfg = tiny_cfg();
        let model = Mamba2Model::new(&cfg);
        let ids = vec![0usize, 1, 2];
        let out = model.forward(&ids).expect("model forward");
        assert_eq!(out.len(), 3, "output seq len must match input");
        assert_eq!(out[0].len(), cfg.d_model, "output dim must be d_model");
    }

    // ── Output shape: d_model preserved ──────────────────────────────────────

    #[test]
    fn test_mamba2_model_forward_output_d_model() {
        let cfg = tiny_cfg();
        let model = Mamba2Model::new(&cfg);
        let ids = vec![0usize, 1, 2, 3, 4];
        let out = model.forward(&ids).expect("forward");
        for row in &out {
            assert_eq!(
                row.len(),
                cfg.d_model,
                "each token output must have d_model features"
            );
        }
    }

    // ── CausalLM logits shape: vocab_size ────────────────────────────────────

    #[test]
    fn test_mamba2_causal_lm_logits_vocab_size() {
        let cfg = tiny_cfg();
        let model = Mamba2ForCausalLM::new(&cfg);
        let ids = vec![0usize, 1];
        let logits = model.forward(&ids).expect("causal lm forward");
        assert_eq!(logits.len(), 2, "logits must have one row per token");
        for row in &logits {
            assert_eq!(
                row.len(),
                cfg.vocab_size,
                "each row must have vocab_size logits"
            );
        }
    }

    // ── RMSNorm dim accessor ──────────────────────────────────────────────────

    #[test]
    fn test_mamba2_rmsnorm_dim_accessor() {
        let norm = Mamba2RmsNorm::new(32, 1e-5);
        assert_eq!(
            norm.dim(),
            32,
            "RmsNorm dim accessor must match construction arg"
        );
    }

    // ── SSM a_log accessor ────────────────────────────────────────────────────

    #[test]
    fn test_mamba2_ssm_a_log_length() {
        let cfg = tiny_cfg();
        let ssm = Mamba2SSM::new(&cfg);
        assert_eq!(
            ssm.a_log().len(),
            cfg.nheads,
            "a_log length must equal nheads"
        );
    }

    // ── SSM d_bias (D skip) length ────────────────────────────────────────────

    #[test]
    fn test_mamba2_ssm_d_bias_length() {
        let cfg = tiny_cfg();
        let ssm = Mamba2SSM::new(&cfg);
        assert_eq!(
            ssm.d_bias().len(),
            cfg.nheads,
            "d_bias length must equal nheads"
        );
    }

    // ── SSM config accessor ───────────────────────────────────────────────────

    #[test]
    fn test_mamba2_ssm_config_accessor() {
        let cfg = tiny_cfg();
        let ssm = Mamba2SSM::new(&cfg);
        assert_eq!(
            ssm.config().d_model,
            cfg.d_model,
            "SSM config must match construction config"
        );
    }

    // ── softplus positivity and large-x behaviour ─────────────────────────────

    #[test]
    fn test_mamba2_softplus_always_positive() {
        for x in [-100.0, -10.0, 0.0, 10.0, 100.0] {
            let v = softplus(x);
            assert!(v > 0.0, "softplus({x}) must be positive, got {v}");
        }
    }

    #[test]
    fn test_mamba2_softplus_large_approx_identity() {
        // softplus(x) ≈ x for x >> 0
        let v = softplus(50.0);
        assert!((v - 50.0).abs() < 0.01, "softplus(50) ≈ 50, got {v}");
    }

    // ── tie_embeddings config ──────────────────────────────────────────────────

    #[test]
    fn test_mamba2_tie_embeddings_2_7b() {
        assert!(
            Mamba2Config::mamba2_2_7b().tie_embeddings,
            "2.7B must have tied embeddings"
        );
    }

    #[test]
    fn test_mamba2_tie_embeddings_small_test_false() {
        assert!(
            !Mamba2Config::small_test().tie_embeddings,
            "small_test must NOT tie embeddings"
        );
    }

    // ── RMSNorm normalizes correctly ──────────────────────────────────────────

    #[test]
    fn test_mamba2_rmsnorm_normalizes_constant_input() {
        // Input of all ones: RMS = 1.0, normalized = 1.0 * weight = 1.0
        let norm = Mamba2RmsNorm::new(4, 1e-8);
        let x = vec![1.0f64; 4];
        let out = norm.forward(&x).expect("rmsnorm forward");
        for &v in &out {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "constant input must normalize to 1.0, got {v}"
            );
        }
    }

    // ── Error types ───────────────────────────────────────────────────────────

    #[test]
    fn test_mamba2_error_empty_input() {
        let cfg = tiny_cfg();
        let model = Mamba2ForCausalLM::new(&cfg);
        let result = model.forward(&[]);
        assert!(result.is_err(), "empty input must return an error");
    }

    #[test]
    fn test_mamba2_error_dim_mismatch_in_ssm() {
        // Passing wrong-width tokens to SSM should return a DimMismatch error
        let cfg = tiny_cfg();
        let ssm = Mamba2SSM::new(&cfg);
        // tokens with wrong d_model
        let wrong = vec![vec![0.0f64; cfg.d_model + 5]];
        let result = ssm.forward(&wrong);
        assert!(result.is_err(), "wrong d_model in SSM input must fail");
    }
}
