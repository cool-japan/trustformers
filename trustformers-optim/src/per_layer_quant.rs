//! Per-layer quantization bit-width selection.
//!
//! Assigns different quantization bit-widths to different layers based on
//! sensitivity analysis, memory budget constraints, and configurable strategies.

// ─────────────────────────────────────────── BitWidth ────────────────────────

/// Quantization bit-widths supported for model layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BitWidth {
    Int2 = 2,
    Int4 = 4,
    Int8 = 8,
    Fp16 = 16,
    Fp32 = 32,
}

impl BitWidth {
    /// Number of bytes consumed per weight element at this precision.
    pub fn bytes_per_weight(&self) -> f64 {
        match self {
            Self::Int2 => 0.25, // 2 bits = 0.25 bytes
            Self::Int4 => 0.5,  // 4 bits = 0.5 bytes
            Self::Int8 => 1.0,
            Self::Fp16 => 2.0,
            Self::Fp32 => 4.0,
        }
    }

    /// Compression ratio relative to Fp32 (higher = more compressed).
    pub fn compression_ratio_vs_fp32(&self) -> f64 {
        4.0 / self.bytes_per_weight()
    }
}

impl std::fmt::Display for BitWidth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int2 => write!(f, "INT2"),
            Self::Int4 => write!(f, "INT4"),
            Self::Int8 => write!(f, "INT8"),
            Self::Fp16 => write!(f, "FP16"),
            Self::Fp32 => write!(f, "FP32"),
        }
    }
}

// ─────────────────────────────────────────── LayerSensitivity ────────────────

/// Sensitivity metric for a model layer — higher score means more sensitive to quantization.
#[derive(Debug, Clone)]
pub struct LayerSensitivity {
    pub layer_name: String,
    /// Average gradient norm during training.
    pub gradient_norm: f64,
    /// Variance of weight values.
    pub weight_variance: f64,
    /// Activation range: max − min of activations.
    pub activation_range: f64,
    /// How much output changes per unit weight change.
    pub output_sensitivity: f64,
    /// Embedding layers usually require higher precision.
    pub is_embedding: bool,
    /// Final layers usually require higher precision.
    pub is_final_layer: bool,
}

impl LayerSensitivity {
    /// Compute a composite sensitivity score for this layer.
    ///
    /// Formula: `gradient_norm * 0.4 + weight_variance * 0.3 + activation_range * 0.2 + output_sensitivity * 0.1`
    ///
    /// Bonuses: embeddings +1.0, final layer +0.5.
    pub fn sensitivity_score(&self) -> f64 {
        let base = self.gradient_norm * 0.4
            + self.weight_variance * 0.3
            + self.activation_range * 0.2
            + self.output_sensitivity * 0.1;

        let embed_bonus = if self.is_embedding { 1.0 } else { 0.0 };
        let final_bonus = if self.is_final_layer { 0.5 } else { 0.0 };

        base + embed_bonus + final_bonus
    }
}

// ─────────────────────────────────────────── BitWidthStrategy ────────────────

/// Strategy for assigning bit-widths to layers.
#[derive(Debug, Clone)]
pub enum BitWidthStrategy {
    /// Same bit-width for all layers.
    Uniform(BitWidth),
    /// More sensitive layers get higher precision.
    SensitivityBased {
        /// Score above this → Fp16.
        high_threshold: f64,
        /// Score above this → Int8.
        medium_threshold: f64,
        /// Score below this → Int2.
        low_threshold: f64,
    },
    /// Maximize compression while meeting a memory budget.
    BudgetOptimal,
}

// ─────────────────────────────────────────── QuantizationPolicy ──────────────

/// Policy controlling how per-layer quantization bit-widths are assigned.
#[derive(Debug, Clone)]
pub struct QuantizationPolicy {
    pub strategy: BitWidthStrategy,
    /// Total memory budget in bytes (used for BudgetOptimal strategy).
    pub budget_bytes: Option<usize>,
    /// Never assign a bit-width below this floor.
    pub min_bit_width: BitWidth,
    /// Never assign a bit-width above this ceiling.
    pub max_bit_width: BitWidth,
}

impl Default for QuantizationPolicy {
    fn default() -> Self {
        Self {
            strategy: BitWidthStrategy::Uniform(BitWidth::Int8),
            budget_bytes: None,
            min_bit_width: BitWidth::Int4,
            max_bit_width: BitWidth::Fp32,
        }
    }
}

// ─────────────────────────────────────────── LayerBitWidthAssignment ─────────

/// The resolved bit-width assignment for a single layer.
#[derive(Debug, Clone)]
pub struct LayerBitWidthAssignment {
    pub layer_name: String,
    pub bit_width: BitWidth,
    pub param_count: usize,
    /// Memory footprint for this layer at the assigned bit-width.
    pub memory_bytes: usize,
    pub sensitivity_score: f64,
}

// ─────────────────────────────────────────── QuantizationSummary ─────────────

/// Human-readable summary of a quantization assignment.
pub struct QuantizationSummary {
    pub total_params: usize,
    pub total_memory_bytes: usize,
    pub compression_ratio: f64,
    /// `(bit_width, layer_count)` pairs, sorted ascending by bit-width.
    pub bit_width_distribution: Vec<(BitWidth, usize)>,
    pub avg_sensitivity_score: f64,
}

impl std::fmt::Display for QuantizationSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "QuantizationSummary {{")?;
        writeln!(f, "  total_params: {}", self.total_params)?;
        writeln!(f, "  total_memory_bytes: {}", self.total_memory_bytes)?;
        writeln!(f, "  compression_ratio: {:.2}x", self.compression_ratio)?;
        writeln!(
            f,
            "  avg_sensitivity_score: {:.4}",
            self.avg_sensitivity_score
        )?;
        write!(f, "  bit_width_distribution: [")?;
        for (i, (bw, count)) in self.bit_width_distribution.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}×{}", bw, count)?;
        }
        writeln!(f, "]")?;
        write!(f, "}}")
    }
}

// ─────────────────────────────────────────── QuantSelectionError ─────────────

/// Errors arising from per-layer quantization selection.
#[derive(Debug, thiserror::Error)]
pub enum QuantSelectionError {
    #[error("Budget exceeded: required {required} bytes, budget {budget} bytes")]
    BudgetExceeded { required: usize, budget: usize },
    #[error("Empty layer list")]
    EmptyLayers,
    #[error("Layer count mismatch")]
    LengthMismatch,
}

// ─────────────────────────────────────────── PerLayerQuantSelector ───────────

/// Selects per-layer quantization bit-widths according to a [`QuantizationPolicy`].
pub struct PerLayerQuantSelector {
    policy: QuantizationPolicy,
}

impl PerLayerQuantSelector {
    /// Create a new selector with the given policy.
    pub fn new(policy: QuantizationPolicy) -> Self {
        Self { policy }
    }

    /// Assign bit-widths to each layer.
    ///
    /// `layers` and `layer_param_counts` must be the same length.
    pub fn assign_bit_widths(
        &self,
        layers: &[LayerSensitivity],
        layer_param_counts: &[usize],
    ) -> Result<Vec<LayerBitWidthAssignment>, QuantSelectionError> {
        if layers.is_empty() {
            return Err(QuantSelectionError::EmptyLayers);
        }
        if layers.len() != layer_param_counts.len() {
            return Err(QuantSelectionError::LengthMismatch);
        }

        let assignments = match &self.policy.strategy {
            BitWidthStrategy::Uniform(bw) => self.assign_uniform(*bw, layers, layer_param_counts),
            BitWidthStrategy::SensitivityBased {
                high_threshold,
                medium_threshold,
                low_threshold,
            } => self.assign_sensitivity_based(
                *high_threshold,
                *medium_threshold,
                *low_threshold,
                layers,
                layer_param_counts,
            ),
            BitWidthStrategy::BudgetOptimal => {
                self.assign_budget_optimal(layers, layer_param_counts)?
            },
        };

        // Validate budget if provided.
        if let Some(budget) = self.policy.budget_bytes {
            let required = Self::total_memory_bytes(&assignments);
            if required > budget {
                return Err(QuantSelectionError::BudgetExceeded { required, budget });
            }
        }

        Ok(assignments)
    }

    /// Clamp a bit-width to the policy's [min, max] range.
    fn clamp_bit_width(&self, bw: BitWidth) -> BitWidth {
        if bw < self.policy.min_bit_width {
            self.policy.min_bit_width
        } else if bw > self.policy.max_bit_width {
            self.policy.max_bit_width
        } else {
            bw
        }
    }

    fn make_assignment(
        &self,
        layer: &LayerSensitivity,
        param_count: usize,
        bit_width: BitWidth,
    ) -> LayerBitWidthAssignment {
        let bw = self.clamp_bit_width(bit_width);
        let memory_bytes = (param_count as f64 * bw.bytes_per_weight()).ceil() as usize;
        LayerBitWidthAssignment {
            layer_name: layer.layer_name.clone(),
            bit_width: bw,
            param_count,
            memory_bytes,
            sensitivity_score: layer.sensitivity_score(),
        }
    }

    fn assign_uniform(
        &self,
        bw: BitWidth,
        layers: &[LayerSensitivity],
        counts: &[usize],
    ) -> Vec<LayerBitWidthAssignment> {
        layers
            .iter()
            .zip(counts.iter())
            .map(|(l, &c)| self.make_assignment(l, c, bw))
            .collect()
    }

    fn assign_sensitivity_based(
        &self,
        high_threshold: f64,
        medium_threshold: f64,
        low_threshold: f64,
        layers: &[LayerSensitivity],
        counts: &[usize],
    ) -> Vec<LayerBitWidthAssignment> {
        layers
            .iter()
            .zip(counts.iter())
            .map(|(l, &c)| {
                let score = l.sensitivity_score();
                let bw = if score > high_threshold {
                    BitWidth::Fp16
                } else if score > medium_threshold {
                    BitWidth::Int8
                } else if score < low_threshold {
                    BitWidth::Int2
                } else {
                    BitWidth::Int4
                };
                self.make_assignment(l, c, bw)
            })
            .collect()
    }

    /// Assign bit-widths to minimize memory while respecting the budget.
    ///
    /// Strategy: start with Int2 for all layers, then upgrade the most
    /// sensitive layers one step at a time as long as the budget allows.
    fn assign_budget_optimal(
        &self,
        layers: &[LayerSensitivity],
        counts: &[usize],
    ) -> Result<Vec<LayerBitWidthAssignment>, QuantSelectionError> {
        // Start with minimum bit-width for each layer.
        let mut bit_widths: Vec<BitWidth> = vec![self.policy.min_bit_width; layers.len()];

        let budget = match self.policy.budget_bytes {
            Some(b) => b,
            None => usize::MAX, // No budget: assign max precision to all.
        };

        // Compute sensitivity scores and sort layer indices by descending sensitivity.
        let mut sorted_indices: Vec<usize> = (0..layers.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            layers[b]
                .sensitivity_score()
                .partial_cmp(&layers[a].sensitivity_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let precision_ladder = [
            BitWidth::Int2,
            BitWidth::Int4,
            BitWidth::Int8,
            BitWidth::Fp16,
            BitWidth::Fp32,
        ];

        // Iteratively upgrade the highest-sensitivity layers.
        // Each iteration tries a single upgrade; if none is possible we stop.
        loop {
            let maybe_upgrade = sorted_indices.iter().find_map(|&idx| {
                let current = bit_widths[idx];
                let next_bw = precision_ladder
                    .iter()
                    .find(|&&bw| bw > current && bw <= self.policy.max_bit_width)
                    .copied()?;

                let old_mem = (counts[idx] as f64 * current.bytes_per_weight()).ceil() as usize;
                let new_mem = (counts[idx] as f64 * next_bw.bytes_per_weight()).ceil() as usize;
                let current_total: usize = bit_widths
                    .iter()
                    .zip(counts.iter())
                    .map(|(&bw, &c)| (c as f64 * bw.bytes_per_weight()).ceil() as usize)
                    .sum();
                let proposed_total = current_total - old_mem + new_mem;
                if proposed_total <= budget {
                    Some((idx, next_bw))
                } else {
                    None
                }
            });

            match maybe_upgrade {
                Some((idx, next_bw)) => bit_widths[idx] = next_bw,
                None => break,
            }
        }

        Ok(layers
            .iter()
            .zip(counts.iter())
            .enumerate()
            .map(|(i, (l, &c))| self.make_assignment(l, c, bit_widths[i]))
            .collect())
    }

    /// Total memory footprint in bytes across all assignments.
    pub fn total_memory_bytes(assignments: &[LayerBitWidthAssignment]) -> usize {
        assignments.iter().map(|a| a.memory_bytes).sum()
    }

    /// Generate a human-readable quantization summary.
    pub fn summary_report(assignments: &[LayerBitWidthAssignment]) -> QuantizationSummary {
        let total_params: usize = assignments.iter().map(|a| a.param_count).sum();
        let total_memory_bytes: usize = assignments.iter().map(|a| a.memory_bytes).sum();

        let fp32_bytes = total_params * 4; // 4 bytes per param at FP32
        let compression_ratio = if total_memory_bytes == 0 {
            1.0
        } else {
            fp32_bytes as f64 / total_memory_bytes as f64
        };

        let avg_sensitivity_score = if assignments.is_empty() {
            0.0
        } else {
            assignments.iter().map(|a| a.sensitivity_score).sum::<f64>() / assignments.len() as f64
        };

        // Count layers per bit-width.
        let mut dist_map: std::collections::HashMap<u8, usize> = std::collections::HashMap::new();
        for a in assignments.iter() {
            *dist_map.entry(a.bit_width as u8).or_insert(0) += 1;
        }
        let mut bit_width_distribution: Vec<(BitWidth, usize)> = dist_map
            .into_iter()
            .filter_map(|(bits, count)| {
                let bw = match bits {
                    2 => Some(BitWidth::Int2),
                    4 => Some(BitWidth::Int4),
                    8 => Some(BitWidth::Int8),
                    16 => Some(BitWidth::Fp16),
                    32 => Some(BitWidth::Fp32),
                    _ => None,
                };
                bw.map(|b| (b, count))
            })
            .collect();
        bit_width_distribution.sort_by_key(|(bw, _)| *bw as u8);

        QuantizationSummary {
            total_params,
            total_memory_bytes,
            compression_ratio,
            bit_width_distribution,
            avg_sensitivity_score,
        }
    }
}

// ─────────────────────────────────────────── tests ───────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_layer(
        name: &str,
        gradient_norm: f64,
        weight_variance: f64,
        activation_range: f64,
        output_sensitivity: f64,
        is_embedding: bool,
        is_final_layer: bool,
    ) -> LayerSensitivity {
        LayerSensitivity {
            layer_name: name.to_string(),
            gradient_norm,
            weight_variance,
            activation_range,
            output_sensitivity,
            is_embedding,
            is_final_layer,
        }
    }

    fn simple_layers() -> (Vec<LayerSensitivity>, Vec<usize>) {
        let layers = vec![
            make_layer("embed", 2.0, 1.0, 1.0, 1.0, true, false),
            make_layer("attn", 1.5, 0.8, 0.5, 0.5, false, false),
            make_layer("ffn", 0.5, 0.3, 0.2, 0.1, false, false),
            make_layer("head", 1.0, 0.6, 0.4, 0.4, false, true),
        ];
        let counts = vec![10_000, 8_000, 16_000, 4_000];
        (layers, counts)
    }

    // ── 1. BitWidth bytes_per_weight ──────────────────────────────────────────

    #[test]
    fn test_bytes_per_weight() {
        assert!((BitWidth::Int2.bytes_per_weight() - 0.25).abs() < 1e-10);
        assert!((BitWidth::Int4.bytes_per_weight() - 0.5).abs() < 1e-10);
        assert!((BitWidth::Int8.bytes_per_weight() - 1.0).abs() < 1e-10);
        assert!((BitWidth::Fp16.bytes_per_weight() - 2.0).abs() < 1e-10);
        assert!((BitWidth::Fp32.bytes_per_weight() - 4.0).abs() < 1e-10);
    }

    // ── 2. BitWidth compression_ratio_vs_fp32 ────────────────────────────────

    #[test]
    fn test_compression_ratio() {
        assert!((BitWidth::Fp32.compression_ratio_vs_fp32() - 1.0).abs() < 1e-10);
        assert!((BitWidth::Fp16.compression_ratio_vs_fp32() - 2.0).abs() < 1e-10);
        assert!((BitWidth::Int8.compression_ratio_vs_fp32() - 4.0).abs() < 1e-10);
        assert!((BitWidth::Int4.compression_ratio_vs_fp32() - 8.0).abs() < 1e-10);
        assert!((BitWidth::Int2.compression_ratio_vs_fp32() - 16.0).abs() < 1e-10);
    }

    // ── 3. sensitivity_score basic formula ────────────────────────────────────

    #[test]
    fn test_sensitivity_score_basic() {
        let layer = make_layer("l", 1.0, 1.0, 1.0, 1.0, false, false);
        // 1.0*0.4 + 1.0*0.3 + 1.0*0.2 + 1.0*0.1 = 1.0
        let score = layer.sensitivity_score();
        assert!((score - 1.0).abs() < 1e-10, "expected 1.0 got {}", score);
    }

    // ── 4. sensitivity_score embedding bonus ─────────────────────────────────

    #[test]
    fn test_sensitivity_score_embedding_bonus() {
        let no_embed = make_layer("l", 1.0, 1.0, 1.0, 1.0, false, false);
        let embed = make_layer("e", 1.0, 1.0, 1.0, 1.0, true, false);
        let diff = embed.sensitivity_score() - no_embed.sensitivity_score();
        assert!((diff - 1.0).abs() < 1e-10, "embedding bonus should be +1.0");
    }

    // ── 5. sensitivity_score final_layer bonus ────────────────────────────────

    #[test]
    fn test_sensitivity_score_final_layer_bonus() {
        let normal = make_layer("l", 1.0, 1.0, 1.0, 1.0, false, false);
        let final_l = make_layer("f", 1.0, 1.0, 1.0, 1.0, false, true);
        let diff = final_l.sensitivity_score() - normal.sensitivity_score();
        assert!(
            (diff - 0.5).abs() < 1e-10,
            "final layer bonus should be +0.5"
        );
    }

    // ── 6. sensitivity_score both bonuses ─────────────────────────────────────

    #[test]
    fn test_sensitivity_score_both_bonuses() {
        let layer = make_layer("l", 1.0, 1.0, 1.0, 1.0, true, true);
        // base 1.0 + embed 1.0 + final 0.5 = 2.5
        assert!((layer.sensitivity_score() - 2.5).abs() < 1e-10);
    }

    // ── 7. Uniform strategy ───────────────────────────────────────────────────

    #[test]
    fn test_uniform_strategy() {
        let policy = QuantizationPolicy {
            strategy: BitWidthStrategy::Uniform(BitWidth::Int8),
            budget_bytes: None,
            min_bit_width: BitWidth::Int2,
            max_bit_width: BitWidth::Fp32,
        };
        let selector = PerLayerQuantSelector::new(policy);
        let (layers, counts) = simple_layers();
        let assignments = selector.assign_bit_widths(&layers, &counts).expect("assign");
        assert!(assignments.iter().all(|a| a.bit_width == BitWidth::Int8));
        assert_eq!(assignments.len(), 4);
    }

    // ── 8. Uniform strategy respects min_bit_width clamp ─────────────────────

    #[test]
    fn test_uniform_strategy_clamped_by_min() {
        let policy = QuantizationPolicy {
            strategy: BitWidthStrategy::Uniform(BitWidth::Int2),
            budget_bytes: None,
            min_bit_width: BitWidth::Int4, // floor is Int4
            max_bit_width: BitWidth::Fp32,
        };
        let selector = PerLayerQuantSelector::new(policy);
        let layers = vec![make_layer("l", 0.1, 0.1, 0.1, 0.1, false, false)];
        let counts = vec![100];
        let assignments = selector.assign_bit_widths(&layers, &counts).expect("assign");
        assert_eq!(assignments[0].bit_width, BitWidth::Int4);
    }

    // ── 9. Sensitivity-based strategy: high score → Fp16 ─────────────────────

    #[test]
    fn test_sensitivity_based_high_score() {
        let policy = QuantizationPolicy {
            strategy: BitWidthStrategy::SensitivityBased {
                high_threshold: 2.0,
                medium_threshold: 1.0,
                low_threshold: 0.3,
            },
            budget_bytes: None,
            min_bit_width: BitWidth::Int2,
            max_bit_width: BitWidth::Fp32,
        };
        let selector = PerLayerQuantSelector::new(policy);
        // score > 2.0 → Fp16
        let layer = make_layer("l", 3.0, 2.0, 2.0, 2.0, false, false);
        // score = 3.0*0.4 + 2.0*0.3 + 2.0*0.2 + 2.0*0.1 = 1.2 + 0.6 + 0.4 + 0.2 = 2.4
        let assignments = selector.assign_bit_widths(&[layer], &[1000]).expect("assign");
        assert_eq!(assignments[0].bit_width, BitWidth::Fp16);
    }

    // ── 10. Sensitivity-based strategy: medium score → Int8 ──────────────────

    #[test]
    fn test_sensitivity_based_medium_score() {
        let policy = QuantizationPolicy {
            strategy: BitWidthStrategy::SensitivityBased {
                high_threshold: 2.0,
                medium_threshold: 1.0,
                low_threshold: 0.3,
            },
            budget_bytes: None,
            min_bit_width: BitWidth::Int2,
            max_bit_width: BitWidth::Fp32,
        };
        let selector = PerLayerQuantSelector::new(policy);
        // score between 1.0 and 2.0 → Int8
        let layer = make_layer("l", 1.5, 1.5, 1.0, 1.0, false, false);
        // score = 1.5*0.4 + 1.5*0.3 + 1.0*0.2 + 1.0*0.1 = 0.6+0.45+0.2+0.1 = 1.35
        let assignments = selector.assign_bit_widths(&[layer], &[500]).expect("assign");
        assert_eq!(assignments[0].bit_width, BitWidth::Int8);
    }

    // ── 11. Sensitivity-based strategy: low score → Int2 ─────────────────────

    #[test]
    fn test_sensitivity_based_low_score() {
        let policy = QuantizationPolicy {
            strategy: BitWidthStrategy::SensitivityBased {
                high_threshold: 2.0,
                medium_threshold: 1.0,
                low_threshold: 0.3,
            },
            budget_bytes: None,
            min_bit_width: BitWidth::Int2,
            max_bit_width: BitWidth::Fp32,
        };
        let selector = PerLayerQuantSelector::new(policy);
        // score < 0.3 → Int2
        let layer = make_layer("l", 0.1, 0.1, 0.1, 0.1, false, false);
        // score = 0.1*0.4 + 0.1*0.3 + 0.1*0.2 + 0.1*0.1 = 0.1
        let assignments = selector.assign_bit_widths(&[layer], &[200]).expect("assign");
        assert_eq!(assignments[0].bit_width, BitWidth::Int2);
    }

    // ── 12. Sensitivity-based strategy: between medium and low → Int4 ─────────

    #[test]
    fn test_sensitivity_based_between_medium_and_low() {
        let policy = QuantizationPolicy {
            strategy: BitWidthStrategy::SensitivityBased {
                high_threshold: 2.0,
                medium_threshold: 1.0,
                low_threshold: 0.3,
            },
            budget_bytes: None,
            min_bit_width: BitWidth::Int2,
            max_bit_width: BitWidth::Fp32,
        };
        let selector = PerLayerQuantSelector::new(policy);
        // score between 0.3 and 1.0 → Int4
        let layer = make_layer("l", 0.8, 0.8, 0.6, 0.6, false, false);
        // score = 0.8*0.4 + 0.8*0.3 + 0.6*0.2 + 0.6*0.1 = 0.32+0.24+0.12+0.06 = 0.74
        let assignments = selector.assign_bit_widths(&[layer], &[300]).expect("assign");
        assert_eq!(assignments[0].bit_width, BitWidth::Int4);
    }

    // ── 13. BudgetOptimal strategy: fits in budget ────────────────────────────

    #[test]
    fn test_budget_optimal_fits() {
        // Large budget: should upgrade some layers
        let large_budget = 1_000_000;
        let policy = QuantizationPolicy {
            strategy: BitWidthStrategy::BudgetOptimal,
            budget_bytes: Some(large_budget),
            min_bit_width: BitWidth::Int2,
            max_bit_width: BitWidth::Fp32,
        };
        let selector = PerLayerQuantSelector::new(policy);
        let (layers, counts) = simple_layers();
        let assignments = selector.assign_bit_widths(&layers, &counts).expect("assign");
        assert_eq!(assignments.len(), 4);
        let total = PerLayerQuantSelector::total_memory_bytes(&assignments);
        assert!(total <= large_budget);
    }

    // ── 14. Empty layer list error ────────────────────────────────────────────

    #[test]
    fn test_empty_layers_error() {
        let policy = QuantizationPolicy::default();
        let selector = PerLayerQuantSelector::new(policy);
        let err = selector.assign_bit_widths(&[], &[]).expect_err("should error on empty");
        assert!(matches!(err, QuantSelectionError::EmptyLayers));
    }

    // ── 15. Length mismatch error ─────────────────────────────────────────────

    #[test]
    fn test_length_mismatch_error() {
        let policy = QuantizationPolicy::default();
        let selector = PerLayerQuantSelector::new(policy);
        let layers = vec![make_layer("l", 1.0, 1.0, 1.0, 1.0, false, false)];
        let counts = vec![100, 200]; // wrong length
        let err = selector
            .assign_bit_widths(&layers, &counts)
            .expect_err("should error on mismatch");
        assert!(matches!(err, QuantSelectionError::LengthMismatch));
    }

    // ── 16. Budget exceeded error ─────────────────────────────────────────────

    #[test]
    fn test_budget_exceeded_error() {
        // Budget too small even for Int4 on a big layer.
        let tiny_budget = 1; // 1 byte
        let policy = QuantizationPolicy {
            strategy: BitWidthStrategy::Uniform(BitWidth::Int8),
            budget_bytes: Some(tiny_budget),
            min_bit_width: BitWidth::Int2,
            max_bit_width: BitWidth::Fp32,
        };
        let selector = PerLayerQuantSelector::new(policy);
        let layers = vec![make_layer("l", 1.0, 1.0, 1.0, 1.0, false, false)];
        let counts = vec![10_000];
        let err = selector.assign_bit_widths(&layers, &counts).expect_err("should exceed budget");
        assert!(matches!(err, QuantSelectionError::BudgetExceeded { .. }));
    }

    // ── 17. total_memory_bytes ────────────────────────────────────────────────

    #[test]
    fn test_total_memory_bytes() {
        let assignments = vec![
            LayerBitWidthAssignment {
                layer_name: "a".to_string(),
                bit_width: BitWidth::Int8,
                param_count: 100,
                memory_bytes: 100,
                sensitivity_score: 1.0,
            },
            LayerBitWidthAssignment {
                layer_name: "b".to_string(),
                bit_width: BitWidth::Fp16,
                param_count: 50,
                memory_bytes: 100,
                sensitivity_score: 2.0,
            },
        ];
        assert_eq!(PerLayerQuantSelector::total_memory_bytes(&assignments), 200);
    }

    // ── 18. summary_report display ────────────────────────────────────────────

    #[test]
    fn test_summary_report_display() {
        let policy = QuantizationPolicy {
            strategy: BitWidthStrategy::Uniform(BitWidth::Int8),
            budget_bytes: None,
            min_bit_width: BitWidth::Int2,
            max_bit_width: BitWidth::Fp32,
        };
        let selector = PerLayerQuantSelector::new(policy);
        let (layers, counts) = simple_layers();
        let assignments = selector.assign_bit_widths(&layers, &counts).expect("assign");
        let summary = PerLayerQuantSelector::summary_report(&assignments);
        let s = format!("{}", summary);
        assert!(s.contains("total_params"));
        assert!(s.contains("compression_ratio"));
        assert!(s.contains("INT8"));
    }

    // ── 19. summary_report compression ratio ─────────────────────────────────

    #[test]
    fn test_summary_report_compression_ratio() {
        // Int8 should give 4x compression vs FP32
        let assignments = vec![LayerBitWidthAssignment {
            layer_name: "l".to_string(),
            bit_width: BitWidth::Int8,
            param_count: 1000,
            memory_bytes: 1000, // 1 byte per weight
            sensitivity_score: 0.5,
        }];
        let summary = PerLayerQuantSelector::summary_report(&assignments);
        // fp32_bytes = 4000, total_memory = 1000 → ratio = 4.0
        assert!((summary.compression_ratio - 4.0).abs() < 1e-6);
    }

    // ── 20. QuantSelectionError display ──────────────────────────────────────

    #[test]
    fn test_quant_selection_error_display() {
        let e1 = QuantSelectionError::EmptyLayers;
        assert!(format!("{}", e1).contains("Empty"));
        let e2 = QuantSelectionError::LengthMismatch;
        assert!(format!("{}", e2).contains("mismatch"));
        let e3 = QuantSelectionError::BudgetExceeded {
            required: 500,
            budget: 100,
        };
        let s = format!("{}", e3);
        assert!(s.contains("500"));
        assert!(s.contains("100"));
    }
}
