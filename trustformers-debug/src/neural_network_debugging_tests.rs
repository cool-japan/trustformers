//! Tests for neural_network_debugging module

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use scirs2_core::ndarray::{Array2, IxDyn};

    use crate::neural_network_debugging::{
        AttentionDebugConfig, AttentionDebugger, AttentionDistribution, AttentionHealthStatus,
        AttentionHeadAnalysis, AttentionMap, AttentionPattern, EvolutionType,
        HeadSpecializationType, ModelAttentionSummary, RedundancyAnalysis, TransformerDebugConfig,
        TransformerDebugger,
    };

    // -------------------------------------------------------------------------
    // LCG for deterministic pseudo-random data without rand crate
    // -------------------------------------------------------------------------

    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }

        fn next(&mut self) -> u64 {
            self.state = self.state
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            self.state
        }

        fn next_f32(&mut self) -> f32 {
            (self.next() >> 11) as f32 / (1u64 << 53) as f32
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /// Build a normalized seq_len x seq_len attention weight array (row-wise softmax).
    fn make_attention_array(seq_len: usize, seed: u64) -> scirs2_core::ndarray::ArrayD<f32> {
        let mut lcg = Lcg::new(seed);
        let mut data = vec![0.0f32; seq_len * seq_len];
        for v in &mut data {
            *v = lcg.next_f32() + 1e-6;
        }
        // Row-wise normalization so rows sum to ~1 (like softmax output)
        for row in 0..seq_len {
            let row_sum: f32 = data[row * seq_len..(row + 1) * seq_len].iter().sum();
            if row_sum > 0.0 {
                for col in 0..seq_len {
                    data[row * seq_len + col] /= row_sum;
                }
            }
        }
        let array2 = Array2::from_shape_vec((seq_len, seq_len), data)
            .expect("shape matches");
        array2.into_dyn()
    }

    /// Build a diagonal-heavy attention array that should read as diagonal pattern.
    fn make_diagonal_attention(seq_len: usize) -> scirs2_core::ndarray::ArrayD<f32> {
        let mut data = vec![0.01f32; seq_len * seq_len];
        // Put large values on diagonal
        for i in 0..seq_len {
            data[i * seq_len + i] = 10.0;
        }
        // Row-normalize
        for row in 0..seq_len {
            let row_sum: f32 = data[row * seq_len..(row + 1) * seq_len].iter().sum();
            for col in 0..seq_len {
                data[row * seq_len + col] /= row_sum;
            }
        }
        Array2::from_shape_vec((seq_len, seq_len), data)
            .expect("shape matches")
            .into_dyn()
    }

    // -------------------------------------------------------------------------
    // AttentionDebugConfig
    // -------------------------------------------------------------------------

    #[test]
    fn test_attention_debug_config_default() {
        let config = AttentionDebugConfig::default();
        assert!(config.enable_attention_visualization);
        assert!(config.enable_head_analysis);
        assert!(config.enable_pattern_detection);
        assert!(config.attention_threshold > 0.0);
        assert!(config.max_heads_to_analyze > 0);
    }

    #[test]
    fn test_attention_debug_config_custom() {
        let config = AttentionDebugConfig {
            enable_attention_visualization: false,
            enable_head_analysis: true,
            enable_pattern_detection: false,
            attention_threshold: 0.05,
            max_heads_to_analyze: 4,
        };
        assert!(!config.enable_attention_visualization);
        assert_eq!(config.max_heads_to_analyze, 4);
    }

    // -------------------------------------------------------------------------
    // AttentionDebugger construction
    // -------------------------------------------------------------------------

    #[test]
    fn test_attention_debugger_new_with_default_config() {
        let config = AttentionDebugConfig::default();
        let debugger = AttentionDebugger::new(config);
        let dbg = format!("{:?}", debugger);
        assert!(!dbg.is_empty());
    }

    // -------------------------------------------------------------------------
    // analyze_attention_layer – basic
    // -------------------------------------------------------------------------

    #[test]
    fn test_analyze_attention_layer_single_head() {
        let config = AttentionDebugConfig::default();
        let mut debugger = AttentionDebugger::new(config);
        let head_weights = make_attention_array(8, 42);
        let result = debugger.analyze_attention_layer(0, &[head_weights]);
        assert!(result.is_ok(), "analyze_attention_layer should succeed");
        let analysis = result.expect("analysis ok");
        assert_eq!(analysis.layer_index, 0);
        assert_eq!(analysis.num_heads, 1);
        assert_eq!(analysis.head_analyses.len(), 1);
        assert_eq!(analysis.attention_maps.len(), 1);
    }

    #[test]
    fn test_analyze_attention_layer_multiple_heads() {
        let config = AttentionDebugConfig::default();
        let mut debugger = AttentionDebugger::new(config);
        let heads: Vec<_> = (0..4).map(|i| make_attention_array(8, i as u64 * 100)).collect();
        let result = debugger.analyze_attention_layer(2, &heads);
        assert!(result.is_ok());
        let analysis = result.expect("analysis ok");
        assert_eq!(analysis.num_heads, 4);
        assert_eq!(analysis.head_analyses.len(), 4);
    }

    #[test]
    fn test_analyze_attention_layer_max_heads_limit() {
        let config = AttentionDebugConfig {
            max_heads_to_analyze: 2,
            ..AttentionDebugConfig::default()
        };
        let mut debugger = AttentionDebugger::new(config);
        // Provide 6 heads but analyzer should cap at 2
        let heads: Vec<_> = (0..6).map(|i| make_attention_array(6, i as u64 * 7)).collect();
        let result = debugger.analyze_attention_layer(0, &heads);
        assert!(result.is_ok());
        let analysis = result.expect("analysis ok");
        // The analysis should only have processed up to max_heads_to_analyze
        assert!(analysis.head_analyses.len() <= 2);
    }

    #[test]
    fn test_analyze_wrong_dimensionality_returns_error() {
        let config = AttentionDebugConfig::default();
        let mut debugger = AttentionDebugger::new(config);
        // Pass a 1D array, which should fail
        let bad_weights =
            scirs2_core::ndarray::Array1::from_vec(vec![0.5f32, 0.5]).into_dyn();
        let result = debugger.analyze_attention_layer(0, &[bad_weights]);
        assert!(result.is_err(), "1D attention weights should return an error");
    }

    // -------------------------------------------------------------------------
    // AttentionMap fields
    // -------------------------------------------------------------------------

    #[test]
    fn test_attention_map_fields() {
        let config = AttentionDebugConfig::default();
        let mut debugger = AttentionDebugger::new(config);
        let head = make_attention_array(6, 99);
        let result = debugger.analyze_attention_layer(1, &[head]).expect("ok");
        let map = &result.attention_maps[0];
        assert_eq!(map.layer_index, 1);
        assert_eq!(map.head_index, 0);
        assert_eq!(map.sequence_length, 6);
        assert_eq!(map.attention_weights.len(), 6);
        // Entropy should be non-negative
        assert!(map.attention_entropy >= 0.0);
        // Sparsity ratio should be in [0, 1]
        assert!(map.sparsity_ratio >= 0.0 && map.sparsity_ratio <= 1.0);
    }

    // -------------------------------------------------------------------------
    // AttentionPattern variants
    // -------------------------------------------------------------------------

    #[test]
    fn test_attention_pattern_debug() {
        let patterns = [
            AttentionPattern::Diagonal,
            AttentionPattern::Block,
            AttentionPattern::Sparse,
            AttentionPattern::Uniform,
            AttentionPattern::Concentrated,
            AttentionPattern::Strided,
            AttentionPattern::Random,
        ];
        for p in &patterns {
            let dbg = format!("{:?}", p);
            assert!(!dbg.is_empty());
        }
    }

    #[test]
    fn test_diagonal_attention_detected() {
        let config = AttentionDebugConfig::default();
        let mut debugger = AttentionDebugger::new(config);
        let head = make_diagonal_attention(8);
        let result = debugger.analyze_attention_layer(0, &[head]).expect("ok");
        let map = &result.attention_maps[0];
        assert_eq!(
            map.attention_pattern,
            AttentionPattern::Diagonal,
            "Diagonal-heavy weights should be detected as Diagonal pattern"
        );
    }

    // -------------------------------------------------------------------------
    // HeadSpecializationType variants
    // -------------------------------------------------------------------------

    #[test]
    fn test_head_specialization_type_equality() {
        assert_eq!(HeadSpecializationType::LocalSyntax, HeadSpecializationType::LocalSyntax);
        assert_ne!(HeadSpecializationType::LocalSyntax, HeadSpecializationType::LongRange);
    }

    #[test]
    fn test_head_specialization_type_debug() {
        let types = [
            HeadSpecializationType::LocalSyntax,
            HeadSpecializationType::LongRange,
            HeadSpecializationType::Positional,
            HeadSpecializationType::ContentBased,
            HeadSpecializationType::Copying,
            HeadSpecializationType::Delimiter,
            HeadSpecializationType::Mixed,
            HeadSpecializationType::Redundant,
        ];
        for t in &types {
            let dbg = format!("{:?}", t);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // AttentionDistribution
    // -------------------------------------------------------------------------

    #[test]
    fn test_attention_distribution_construction() {
        let dist = AttentionDistribution {
            mean_attention: 0.1,
            std_attention: 0.05,
            max_attention: 0.4,
            min_attention: 0.01,
            entropy: 2.5,
            effective_context_length: 4.0,
        };
        assert!(dist.mean_attention > 0.0);
        assert!(dist.entropy > 0.0);
    }

    // -------------------------------------------------------------------------
    // RedundancyAnalysis
    // -------------------------------------------------------------------------

    #[test]
    fn test_redundancy_analysis_construction() {
        let analysis = RedundancyAnalysis {
            redundant_head_pairs: vec![(0, 1, 0.9)],
            redundancy_groups: vec![vec![0, 1]],
            overall_redundancy_score: 0.9,
        };
        assert_eq!(analysis.redundant_head_pairs.len(), 1);
        assert!(analysis.overall_redundancy_score > 0.0);
    }

    #[test]
    fn test_layer_diversity_score_in_range() {
        let config = AttentionDebugConfig::default();
        let mut debugger = AttentionDebugger::new(config);
        let heads: Vec<_> = (0..4).map(|i| make_attention_array(8, i as u64 * 31)).collect();
        let result = debugger.analyze_attention_layer(0, &heads).expect("ok");
        // Diversity score should be finite
        assert!(result.layer_diversity_score.is_finite());
    }

    // -------------------------------------------------------------------------
    // TransformerDebugConfig
    // -------------------------------------------------------------------------

    #[test]
    fn test_transformer_debug_config_default() {
        let config = TransformerDebugConfig::default();
        assert!(config.enable_layer_analysis);
        assert!(config.enable_cross_layer_analysis);
        assert!(config.max_layers_to_analyze > 0);
    }

    #[test]
    fn test_transformer_debug_config_custom() {
        let config = TransformerDebugConfig {
            attention_config: AttentionDebugConfig::default(),
            enable_layer_analysis: false,
            enable_cross_layer_analysis: false,
            max_layers_to_analyze: 6,
        };
        assert!(!config.enable_layer_analysis);
        assert_eq!(config.max_layers_to_analyze, 6);
    }

    // -------------------------------------------------------------------------
    // TransformerDebugger
    // -------------------------------------------------------------------------

    #[test]
    fn test_transformer_debugger_new() {
        let config = TransformerDebugConfig::default();
        let debugger = TransformerDebugger::new(config);
        let dbg = format!("{:?}", debugger);
        assert!(!dbg.is_empty());
    }

    #[test]
    fn test_transformer_analyze_empty_layers() {
        let config = TransformerDebugConfig::default();
        let mut debugger = TransformerDebugger::new(config);
        // Empty input – should succeed with default summary
        let result = debugger.analyze_transformer_attention(&[]);
        assert!(result.is_ok());
        let analysis = result.expect("ok");
        assert_eq!(analysis.num_layers, 0);
        assert_eq!(analysis.model_attention_summary.total_layers, 0);
    }

    #[test]
    fn test_transformer_analyze_single_layer() {
        let config = TransformerDebugConfig::default();
        let mut debugger = TransformerDebugger::new(config);
        let layer: Vec<_> = vec![make_attention_array(8, 1), make_attention_array(8, 2)];
        let result = debugger.analyze_transformer_attention(&[layer]);
        assert!(result.is_ok());
        let analysis = result.expect("ok");
        assert_eq!(analysis.num_layers, 1);
        assert_eq!(analysis.layer_analyses.len(), 1);
        assert_eq!(analysis.model_attention_summary.total_layers, 1);
    }

    #[test]
    fn test_transformer_analyze_multiple_layers() {
        let config = TransformerDebugConfig::default();
        let mut debugger = TransformerDebugger::new(config);
        let layers: Vec<Vec<_>> = (0..3)
            .map(|layer_i| {
                (0..4)
                    .map(|head_i| make_attention_array(8, (layer_i * 4 + head_i) as u64 * 13))
                    .collect()
            })
            .collect();
        let result = debugger.analyze_transformer_attention(&layers);
        assert!(result.is_ok());
        let analysis = result.expect("ok");
        assert_eq!(analysis.num_layers, 3);
        assert_eq!(analysis.model_attention_summary.total_layers, 3);
        assert_eq!(analysis.model_attention_summary.total_heads, 12);
    }

    #[test]
    fn test_cross_layer_analysis_present_when_enabled() {
        let config = TransformerDebugConfig {
            enable_cross_layer_analysis: true,
            ..TransformerDebugConfig::default()
        };
        let mut debugger = TransformerDebugger::new(config);
        let layers: Vec<Vec<_>> = (0..2)
            .map(|i| vec![make_attention_array(6, i as u64 * 7)])
            .collect();
        let result = debugger.analyze_transformer_attention(&layers).expect("ok");
        assert!(
            result.cross_layer_analysis.is_some(),
            "Cross-layer analysis should be present when enabled"
        );
    }

    #[test]
    fn test_cross_layer_analysis_absent_when_disabled() {
        let config = TransformerDebugConfig {
            enable_cross_layer_analysis: false,
            ..TransformerDebugConfig::default()
        };
        let mut debugger = TransformerDebugger::new(config);
        let layers: Vec<Vec<_>> = (0..2)
            .map(|i| vec![make_attention_array(6, i as u64 * 11)])
            .collect();
        let result = debugger.analyze_transformer_attention(&layers).expect("ok");
        assert!(
            result.cross_layer_analysis.is_none(),
            "Cross-layer analysis should be absent when disabled"
        );
    }

    // -------------------------------------------------------------------------
    // ModelAttentionSummary
    // -------------------------------------------------------------------------

    #[test]
    fn test_model_attention_summary_default() {
        let summary = ModelAttentionSummary::default();
        assert_eq!(summary.total_layers, 0);
        assert_eq!(summary.total_heads, 0);
        assert_eq!(summary.model_attention_health, AttentionHealthStatus::Poor);
    }

    // -------------------------------------------------------------------------
    // AttentionHealthStatus
    // -------------------------------------------------------------------------

    #[test]
    fn test_attention_health_status_variants() {
        let statuses = [
            AttentionHealthStatus::Excellent,
            AttentionHealthStatus::Good,
            AttentionHealthStatus::Fair,
            AttentionHealthStatus::Poor,
        ];
        for s in &statuses {
            let dbg = format!("{:?}", s);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // EvolutionType
    // -------------------------------------------------------------------------

    #[test]
    fn test_evolution_type_variants() {
        let types = [EvolutionType::Increasing, EvolutionType::Decreasing, EvolutionType::Stable];
        for t in &types {
            let dbg = format!("{:?}", t);
            assert!(!dbg.is_empty());
        }
    }

    // -------------------------------------------------------------------------
    // Diversity score from uniform vs diagonal attention
    // -------------------------------------------------------------------------

    #[test]
    fn test_head_importance_score_non_negative() {
        let config = AttentionDebugConfig::default();
        let mut debugger = AttentionDebugger::new(config);
        let head = make_attention_array(6, 555);
        let result = debugger.analyze_attention_layer(0, &[head]).expect("ok");
        let head_analysis = &result.head_analyses[0];
        assert!(
            head_analysis.importance_score >= 0.0,
            "Importance score should be non-negative"
        );
    }

    #[test]
    fn test_max_layers_limit_respected() {
        let config = TransformerDebugConfig {
            max_layers_to_analyze: 2,
            ..TransformerDebugConfig::default()
        };
        let mut debugger = TransformerDebugger::new(config);
        // Provide 5 layers, should be capped at 2
        let layers: Vec<Vec<_>> = (0..5)
            .map(|i| vec![make_attention_array(4, i as u64 * 17)])
            .collect();
        let result = debugger.analyze_transformer_attention(&layers).expect("ok");
        assert!(
            result.layer_analyses.len() <= 2,
            "Should not analyze more than max_layers_to_analyze"
        );
    }

    #[test]
    fn test_attention_map_weights_are_normalized() {
        let config = AttentionDebugConfig::default();
        let mut debugger = AttentionDebugger::new(config);
        let head = make_attention_array(6, 42);
        let result = debugger.analyze_attention_layer(0, &[head]).expect("ok");
        let map = &result.attention_maps[0];
        for row in &map.attention_weights {
            let row_sum: f32 = row.iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-4,
                "Attention row should sum to ~1.0, got {}",
                row_sum
            );
        }
    }
}
