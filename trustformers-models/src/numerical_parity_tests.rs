//! Numerical parity checks against reference implementations.
//!
//! The point of a parity harness is to fail when this crate's numbers drift from
//! a trusted reference. Two properties are therefore load-bearing here, and both
//! were previously violated:
//!
//! * **Comparisons return errors, they do not panic.** `assert_tensors_close`
//!   is a `Result`-returning public API, but its body called
//!   `approx::assert_abs_diff_eq!`, which *panics* on mismatch. A caller that
//!   dutifully handled the `Err` still had its process aborted on the first
//!   differing element, and the panic message named no tensor index.
//!
//! * **A parity check without reference data is a failure, not a pass.**
//!   [`ReferenceValues`] shipped `sample_outputs: vec![]` for every model, so
//!   any loop of the shape "for each reference output, compare" iterated zero
//!   times and reported success. A harness that passes because it has nothing to
//!   compare against is worse than no harness: it produces a green signal that
//!   means nothing. Reference data now has to be supplied — from a file, or
//!   explicitly in code — and asking for a model that has none is an error.

use anyhow::Result;
use std::path::Path;
use trustformers_core::tensor::Tensor;

/// Test utilities for numerical parity validation
pub struct NumericalParityTests;

impl NumericalParityTests {
    /// Compare two tensors element-wise within an absolute tolerance.
    ///
    /// # Errors
    ///
    /// Returns an error — never a panic — when the shapes differ, when a tensor
    /// is not `F32`, or when any element differs by more than `tolerance`. The
    /// message names the offending flat index and both values.
    pub fn assert_tensors_close(actual: &Tensor, expected: &Tensor, tolerance: f32) -> Result<()> {
        if !(tolerance >= 0.0) {
            return Err(anyhow::anyhow!(
                "tolerance must be a non-negative number, got {tolerance}"
            ));
        }
        match (actual, expected) {
            (Tensor::F32(actual_arr), Tensor::F32(expected_arr)) => {
                if actual_arr.shape() != expected_arr.shape() {
                    return Err(anyhow::anyhow!(
                        "Shape mismatch: {:?} vs {:?}",
                        actual_arr.shape(),
                        expected_arr.shape()
                    ));
                }

                for (index, (a, e)) in actual_arr.iter().zip(expected_arr.iter()).enumerate() {
                    let difference = (a - e).abs();
                    if !(difference <= tolerance) {
                        return Err(anyhow::anyhow!(
                            "element {index} differs by {difference}: actual {a}, expected {e} \
                             (tolerance {tolerance})"
                        ));
                    }
                }
                Ok(())
            },
            _ => Err(anyhow::anyhow!(
                "Only F32 tensors are supported for comparison"
            )),
        }
    }

    /// Test that tensor values are finite (no NaN or infinity)
    ///
    /// # Errors
    ///
    /// Fails when the tensor is not `F32` or holds a non-finite value.
    pub fn assert_tensor_finite(tensor: &Tensor) -> Result<()> {
        match tensor {
            Tensor::F32(arr) => {
                for (index, &value) in arr.iter().enumerate() {
                    if !value.is_finite() {
                        return Err(anyhow::anyhow!(
                            "Non-finite value found at element {index}: {value}"
                        ));
                    }
                }
                Ok(())
            },
            _ => Err(anyhow::anyhow!(
                "Only F32 tensors are supported for finite check"
            )),
        }
    }

    /// Test that tensor values are within expected range
    ///
    /// # Errors
    ///
    /// Fails when the tensor is not `F32` or holds a value outside the range.
    pub fn assert_tensor_range(tensor: &Tensor, min_val: f32, max_val: f32) -> Result<()> {
        match tensor {
            Tensor::F32(arr) => {
                for (index, &value) in arr.iter().enumerate() {
                    if value < min_val || value > max_val {
                        return Err(anyhow::anyhow!(
                            "Value {value} at element {index} is outside expected range \
                             [{min_val}, {max_val}]"
                        ));
                    }
                }
                Ok(())
            },
            _ => Err(anyhow::anyhow!(
                "Only F32 tensors are supported for range check"
            )),
        }
    }

    /// Compute the maximum relative error between two tensors.
    ///
    /// # Errors
    ///
    /// Fails when the shapes differ or a tensor is not `F32`.
    pub fn compute_relative_error(actual: &Tensor, expected: &Tensor) -> Result<f32> {
        match (actual, expected) {
            (Tensor::F32(actual_arr), Tensor::F32(expected_arr)) => {
                if actual_arr.shape() != expected_arr.shape() {
                    return Err(anyhow::anyhow!(
                        "Shape mismatch: {:?} vs {:?}",
                        actual_arr.shape(),
                        expected_arr.shape()
                    ));
                }

                let mut max_relative_error = 0.0f32;
                for (a, e) in actual_arr.iter().zip(expected_arr.iter()) {
                    if e.abs() > 1e-8 {
                        let relative_error = ((a - e) / e).abs();
                        max_relative_error = max_relative_error.max(relative_error);
                    }
                }

                Ok(max_relative_error)
            },
            _ => Err(anyhow::anyhow!(
                "Only F32 tensors are supported for error computation"
            )),
        }
    }

    /// Run a parity check of `actual` against every reference output a model
    /// carries.
    ///
    /// This is the entry point a parity test should call, because it is the one
    /// that refuses to report success when there is nothing to compare against.
    ///
    /// # Errors
    ///
    /// Fails when `reference` carries no sample outputs — a check with no
    /// reference data proves nothing and must not pass — or when any comparison
    /// fails.
    pub fn check_against_reference(
        actual: &[Tensor],
        reference: &ReferenceData,
        tolerance: f32,
    ) -> Result<ParityReport> {
        if reference.sample_outputs.is_empty() {
            return Err(anyhow::anyhow!(
                "reference data for {} carries no sample outputs, so a parity check against it \
                 would pass vacuously. Load real reference tensors with \
                 ReferenceData::from_safetensors_file before running the check.",
                reference.model_name
            ));
        }
        if actual.len() != reference.sample_outputs.len() {
            return Err(anyhow::anyhow!(
                "{} produced {} output tensor(s) but the reference holds {}",
                reference.model_name,
                actual.len(),
                reference.sample_outputs.len()
            ));
        }

        let mut max_relative_error = 0.0f32;
        for (index, (got, want)) in actual.iter().zip(reference.sample_outputs.iter()).enumerate() {
            Self::assert_tensors_close(got, want, tolerance)
                .map_err(|e| anyhow::anyhow!("output {index} of {}: {e}", reference.model_name))?;
            let error = Self::compute_relative_error(got, want)?;
            max_relative_error = max_relative_error.max(error);
        }

        Ok(ParityReport {
            model_name: reference.model_name.clone(),
            tensors_compared: actual.len(),
            max_relative_error,
            tolerance,
        })
    }

    /// Generate deterministic input data for parity testing.
    ///
    /// # Errors
    ///
    /// Fails when any dimension is zero, because the resulting "test data" would
    /// exercise nothing.
    pub fn generate_test_data(
        batch_size: usize,
        seq_len: usize,
        hidden_size: usize,
    ) -> Result<TestData> {
        if batch_size == 0 || seq_len == 0 || hidden_size == 0 {
            return Err(anyhow::anyhow!(
                "test data dimensions must all be non-zero, got batch_size={batch_size}, \
                 seq_len={seq_len}, hidden_size={hidden_size}"
            ));
        }
        Ok(TestData {
            input_ids: (0..batch_size * seq_len).map(|i| (i % 1000) as u32).collect(),
            attention_mask: vec![1u32; batch_size * seq_len],
            expected_shape: vec![batch_size, seq_len, hidden_size],
        })
    }
}

/// What a completed parity check actually compared.
#[derive(Debug, Clone, PartialEq)]
pub struct ParityReport {
    /// Model the reference data belongs to.
    pub model_name: String,
    /// Number of tensors that were compared. Never zero: a report only exists
    /// when at least one real comparison ran.
    pub tensors_compared: usize,
    /// Largest relative error observed across all compared elements.
    pub max_relative_error: f32,
    /// Absolute tolerance the comparison used.
    pub tolerance: f32,
}

/// Test data structure for parity tests
#[derive(Debug, Clone)]
pub struct TestData {
    /// Input token IDs
    pub input_ids: Vec<u32>,
    /// Attention mask
    pub attention_mask: Vec<u32>,
    /// Expected output shape
    pub expected_shape: Vec<usize>,
}

/// Architecture metadata plus the reference outputs to compare against.
#[derive(Debug, Clone)]
pub struct ReferenceData {
    /// Model name
    pub model_name: String,
    /// Hidden dimension size
    pub hidden_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum position embeddings
    pub max_position_embeddings: usize,
    /// Reference output tensors captured from the reference implementation.
    ///
    /// Empty means "no reference data available"; a parity check against such a
    /// `ReferenceData` is rejected rather than passing vacuously.
    pub sample_outputs: Vec<Tensor>,
}

impl ReferenceData {
    /// Whether this record carries outputs a parity check can compare against.
    pub fn has_reference_outputs(&self) -> bool {
        !self.sample_outputs.is_empty()
    }

    /// Attach reference outputs captured from the reference implementation.
    #[must_use]
    pub fn with_outputs(mut self, outputs: Vec<Tensor>) -> Self {
        self.sample_outputs = outputs;
        self
    }

    /// Load reference outputs from a safetensors file.
    ///
    /// The file is parsed for real by
    /// [`Checkpoint`](crate::weight_loading::checkpoint::Checkpoint); every
    /// tensor it holds becomes a reference output, ordered by name.
    ///
    /// # Errors
    ///
    /// Fails when the file cannot be read, when it is not a recognised
    /// checkpoint container, or when it holds no tensors.
    pub fn load_outputs_from_file(mut self, path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).map_err(|e| {
            anyhow::anyhow!(
                "failed to read reference outputs from {}: {e}",
                path.display()
            )
        })?;
        let checkpoint = crate::weight_loading::checkpoint::Checkpoint::from_bytes(&bytes)
            .map_err(|e| {
                anyhow::anyhow!(
                    "failed to parse {} as reference outputs: {e}",
                    path.display()
                )
            })?;
        let names = checkpoint.names();
        if names.is_empty() {
            return Err(anyhow::anyhow!(
                "{} holds no tensors, so it supplies no reference outputs",
                path.display()
            ));
        }
        let mut outputs = Vec::with_capacity(names.len());
        for name in &names {
            let tensor = checkpoint.get(name).ok_or_else(|| {
                anyhow::anyhow!("reference file listed tensor {name} but does not hold it")
            })?;
            outputs.push(tensor.clone());
        }
        self.sample_outputs = outputs;
        Ok(self)
    }
}

/// Architecture metadata for well-known reference models.
///
/// These constructors describe *shapes*, not outputs: they deliberately return
/// records with no `sample_outputs`, because this crate ships no captured
/// reference tensors. Attach them with [`ReferenceData::with_outputs`] or
/// [`ReferenceData::load_outputs_from_file`] before running a parity check;
/// [`NumericalParityTests::check_against_reference`] rejects a record without
/// them rather than reporting a vacuous pass.
pub struct ReferenceValues;

impl ReferenceValues {
    /// Architecture metadata for `bert-base-uncased` (no reference outputs).
    pub fn bert_base_uncased() -> ReferenceData {
        ReferenceData {
            model_name: "bert-base-uncased".to_string(),
            hidden_size: 768,
            num_layers: 12,
            num_attention_heads: 12,
            vocab_size: 30522,
            max_position_embeddings: 512,
            sample_outputs: Vec::new(),
        }
    }

    /// Architecture metadata for `gpt2` (no reference outputs).
    pub fn gpt2() -> ReferenceData {
        ReferenceData {
            model_name: "gpt2".to_string(),
            hidden_size: 768,
            num_layers: 12,
            num_attention_heads: 12,
            vocab_size: 50257,
            max_position_embeddings: 1024,
            sample_outputs: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_comparison() -> Result<()> {
        let tensor1 = Tensor::ones(&[2, 3])?;
        let tensor2 = Tensor::ones(&[2, 3])?;

        NumericalParityTests::assert_tensors_close(&tensor1, &tensor2, 1e-6)?;
        Ok(())
    }

    /// Regression: `assert_tensors_close` panicked via `assert_abs_diff_eq!`
    /// instead of returning the `Err` its signature promises.
    ///
    /// This test *panics* (rather than failing cleanly) against the old body,
    /// because the panic escapes before `expect_err` can observe a `Result`.
    #[test]
    fn tensor_comparison_returns_an_error_instead_of_panicking() {
        let actual = Tensor::ones(&[2, 2]).expect("tensor must build");
        let expected =
            Tensor::from_vec(vec![1.0, 1.0, 1.0, 2.0], &[2, 2]).expect("tensor must build");

        let err = NumericalParityTests::assert_tensors_close(&actual, &expected, 1e-6)
            .expect_err("a differing element must be reported as an error");
        let message = err.to_string();
        assert!(
            message.contains("element 3"),
            "the error must name the offending index: {message}"
        );
        assert!(
            message.contains("tolerance"),
            "the error must state the tolerance: {message}"
        );
    }

    #[test]
    fn tensor_comparison_rejects_a_shape_mismatch() {
        let actual = Tensor::ones(&[2, 2]).expect("tensor must build");
        let expected = Tensor::ones(&[4]).expect("tensor must build");
        let err = NumericalParityTests::assert_tensors_close(&actual, &expected, 1e-6)
            .expect_err("differing shapes must fail");
        assert!(err.to_string().contains("Shape mismatch"));
    }

    /// Regression: every `ReferenceValues` constructor shipped
    /// `sample_outputs: vec![]`, so a parity loop over them compared nothing and
    /// reported success. Absent reference data is now a hard error.
    #[test]
    fn a_parity_check_without_reference_data_is_an_error_not_a_pass() {
        let reference = ReferenceValues::bert_base_uncased();
        assert!(
            !reference.has_reference_outputs(),
            "this crate ships no captured BERT outputs"
        );

        let actual = vec![Tensor::ones(&[2, 2]).expect("tensor must build")];
        let err = NumericalParityTests::check_against_reference(&actual, &reference, 1e-4)
            .expect_err("a check with no reference data must not report success");
        assert!(
            err.to_string().contains("no sample outputs"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn a_parity_check_with_reference_data_compares_every_tensor() {
        let reference = ReferenceValues::gpt2().with_outputs(vec![
            Tensor::ones(&[2, 2]).expect("tensor must build"),
            Tensor::zeros(&[3]).expect("tensor must build"),
        ]);

        let matching = vec![
            Tensor::ones(&[2, 2]).expect("tensor must build"),
            Tensor::zeros(&[3]).expect("tensor must build"),
        ];
        let report = NumericalParityTests::check_against_reference(&matching, &reference, 1e-6)
            .expect("matching outputs must pass");
        assert_eq!(report.tensors_compared, 2);
        assert_eq!(report.model_name, "gpt2");

        let diverging = vec![
            Tensor::ones(&[2, 2]).expect("tensor must build"),
            Tensor::ones(&[3]).expect("tensor must build"),
        ];
        let err = NumericalParityTests::check_against_reference(&diverging, &reference, 1e-6)
            .expect_err("a diverging output must fail the check");
        assert!(err.to_string().contains("output 1"), "unexpected: {err}");
    }

    #[test]
    fn a_parity_check_rejects_a_different_number_of_outputs() {
        let reference =
            ReferenceValues::gpt2().with_outputs(vec![Tensor::ones(&[2]).expect("tensor")]);
        let actual = vec![
            Tensor::ones(&[2]).expect("tensor"),
            Tensor::ones(&[2]).expect("tensor"),
        ];
        let err = NumericalParityTests::check_against_reference(&actual, &reference, 1e-6)
            .expect_err("a count mismatch must fail");
        assert!(err.to_string().contains("but the reference holds"));
    }

    #[test]
    fn reference_outputs_can_be_loaded_from_a_safetensors_file() -> Result<()> {
        use crate::weight_loading::test_support::{build_safetensors, F32Tensor};

        let bytes = build_safetensors(&[
            F32Tensor::new("logits", &[2, 2], vec![0.5, -0.5, 1.5, -1.5]),
            F32Tensor::new("pooled", &[2], vec![0.25, 0.75]),
        ]);
        let path = std::env::temp_dir().join(format!(
            "trustformers_parity_reference_{}.safetensors",
            std::process::id()
        ));
        std::fs::write(&path, &bytes)?;

        let reference = ReferenceValues::gpt2().load_outputs_from_file(&path)?;
        std::fs::remove_file(&path).ok();

        assert!(reference.has_reference_outputs());
        assert_eq!(reference.sample_outputs.len(), 2);

        // Tensors come back name-ordered: "logits" then "pooled".
        let actual = vec![
            Tensor::from_vec(vec![0.5, -0.5, 1.5, -1.5], &[2, 2])?,
            Tensor::from_vec(vec![0.25, 0.75], &[2])?,
        ];
        let report = NumericalParityTests::check_against_reference(&actual, &reference, 1e-6)?;
        assert_eq!(report.tensors_compared, 2);
        Ok(())
    }

    #[test]
    fn loading_reference_outputs_from_a_non_checkpoint_fails() {
        let path = std::env::temp_dir().join(format!(
            "trustformers_parity_garbage_{}.bin",
            std::process::id()
        ));
        std::fs::write(&path, vec![0xABu8; 512]).expect("temp file must be writable");
        let result = ReferenceValues::gpt2().load_outputs_from_file(&path);
        std::fs::remove_file(&path).ok();
        assert!(
            result.is_err(),
            "garbage bytes must not be accepted as reference outputs"
        );
    }

    #[test]
    fn test_tensor_finite() -> Result<()> {
        let tensor = Tensor::ones(&[2, 3])?;
        NumericalParityTests::assert_tensor_finite(&tensor)?;
        Ok(())
    }

    #[test]
    fn test_tensor_range() -> Result<()> {
        let tensor = Tensor::ones(&[2, 3])?;
        NumericalParityTests::assert_tensor_range(&tensor, 0.0, 2.0)?;
        Ok(())
    }

    #[test]
    fn test_generate_test_data() -> Result<()> {
        let test_data = NumericalParityTests::generate_test_data(2, 10, 768)?;
        assert_eq!(test_data.input_ids.len(), 20);
        assert_eq!(test_data.attention_mask.len(), 20);
        assert_eq!(test_data.expected_shape, vec![2, 10, 768]);
        Ok(())
    }

    #[test]
    fn test_generate_test_data_rejects_zero_dimensions() {
        assert!(NumericalParityTests::generate_test_data(0, 10, 768).is_err());
        assert!(NumericalParityTests::generate_test_data(2, 0, 768).is_err());
        assert!(NumericalParityTests::generate_test_data(2, 10, 0).is_err());
    }

    #[test]
    fn test_reference_values() {
        let bert_ref = ReferenceValues::bert_base_uncased();
        assert_eq!(bert_ref.hidden_size, 768);
        assert_eq!(bert_ref.vocab_size, 30522);

        let gpt2_ref = ReferenceValues::gpt2();
        assert_eq!(gpt2_ref.hidden_size, 768);
        assert_eq!(gpt2_ref.vocab_size, 50257);
    }
}
