//! Checkpoint binding for S4, and the tensor-name contract it binds against.
//!
//! # There is no de-facto HuggingFace S4 export
//!
//! Unlike BERT or LLaMA, S4 has no widely-published safetensors convention to
//! adopt: the reference implementation (`state-spaces/s4`) ships PyTorch state
//! dicts whose keys depend on which wrapper module a checkpoint was saved from,
//! and the `hf` ports that exist disagree with each other. Pretending otherwise
//! — "this is the HuggingFace S4 layout" — would be a fabricated standard.
//!
//! So this module **declares** a contract instead, modelled on the reference
//! implementation's parameter names (`kernel.A`, `kernel.B`, `kernel.C`,
//! `kernel.D`, `kernel.log_dt`), and refuses a mismatch with the full expected
//! list rather than guessing. An explicit contract a converter can be written
//! against beats a fixture that only tests itself.
//!
//! # The contract
//!
//! `{P}` is `""` or `backbone.`, detected from `embeddings.weight`.
//!
//! | Name | Shape | Meaning |
//! |------|-------|---------|
//! | `{P}embeddings.weight` | `[vocab_size, d_model]` | token embedding |
//! | `{P}blocks.{i}.norm.{weight,bias}` | `[d_model]` | pre-norm |
//! | `{P}blocks.{i}.in_proj.weight` | `[n_ssm, d_model]` | input projection |
//! | `{P}blocks.{i}.in_proj.bias` | `[n_ssm]` | present when `use_bias` |
//! | `{P}blocks.{i}.out_proj.weight` | `[out_width, n_ssm]` | output projection |
//! | `{P}blocks.{i}.out_proj.bias` | `[out_width]` | present when `use_bias` |
//! | `{P}blocks.{i}.kernel.A_real` | `[d_state, d_state]` | state matrix, real part |
//! | `{P}blocks.{i}.kernel.A_imag` | `[d_state, d_state]` | state matrix, imaginary part |
//! | `{P}blocks.{i}.kernel.B_real` | `[d_state]` | input vector, real part |
//! | `{P}blocks.{i}.kernel.B_imag` | `[d_state]` | input vector, imaginary part |
//! | `{P}blocks.{i}.kernel.C_real` | `[d_state]` | output vector, real part |
//! | `{P}blocks.{i}.kernel.C_imag` | `[d_state]` | output vector, imaginary part |
//! | `{P}blocks.{i}.kernel.D` | `[n_ssm]` | skip connection |
//! | `{P}blocks.{i}.kernel.log_dt` | `[n_ssm]` | **log** of the timestep |
//! | `{P}ln_f.{weight,bias}` | `[d_model]` | final norm |
//! | `lm_head.weight` | `[vocab_size, d_model]` | bound by the LM wrapper |
//!
//! `out_width` is `2 · d_model` when `postact = "glu"` (the block gates its
//! output) and `d_model` otherwise — see [`super::model::S4Block::uses_glu`].
//!
//! **`log_dt` is exponentiated on the way in.** The reference parametrises the
//! timestep in log space so it stays positive under gradient descent; binding a
//! tensor named `log_dt` straight into `Δ` would be a new silent-wrong-answer
//! (a checkpoint recording `log_dt = 0` would install a timestep of 0 rather
//! than 1), so `Δ = exp(log_dt)` here. There is no S4 *exporter* yet — `S4Model`
//! still uses the default empty `Model::named_tensors`, so nothing in this crate
//! writes these names back out; a writer would have to apply `ln` on the way.

use scirs2_core::ndarray::{Array1, Array2}; // SciRS2 Integration Policy
use trustformers_core::{
    errors::{Result, TrustformersError},
    layers::Linear,
    tensor::Tensor,
};

use super::config::S4Config;
use super::layer::S4Layer;
use super::model::{S4Block, S4ForLanguageModeling, S4Model};
use crate::weight_loading::binding::{bind_embedding, bind_linear, BoundNamespaces};
use crate::weight_loading::checkpoint::{Checkpoint, LoadReport, UnusedTensors, WeightBinder};

/// Checkpoint namespaces a bare S4 backbone legitimately leaves unbound.
pub(super) const ALLOWED_UNUSED_PREFIXES: &[&str] = &["lm_head."];

/// The unused-tensor policy for a bare backbone load.
pub(super) fn backbone_unused_policy() -> UnusedTensors<'static> {
    UnusedTensors::new(ALLOWED_UNUSED_PREFIXES, &[])
}

/// Every tensor name the S4 contract requires, for a given config.
///
/// Returned in the error when a checkpoint does not match, so the message names
/// the contract rather than only the first thing that went missing.
pub fn expected_tensor_names(config: &S4Config, prefix: &str) -> Vec<String> {
    let mut names = vec![format!("{prefix}embeddings.weight")];
    for block in 0..config.n_layer {
        let base = format!("{prefix}blocks.{block}");
        names.push(format!("{base}.norm.weight"));
        names.push(format!("{base}.norm.bias"));
        names.push(format!("{base}.in_proj.weight"));
        if config.use_bias {
            names.push(format!("{base}.in_proj.bias"));
        }
        names.push(format!("{base}.out_proj.weight"));
        if config.use_bias {
            names.push(format!("{base}.out_proj.bias"));
        }
        for parameter in [
            "A_real", "A_imag", "B_real", "B_imag", "C_real", "C_imag", "D", "log_dt",
        ] {
            names.push(format!("{base}.kernel.{parameter}"));
        }
    }
    names.push(format!("{prefix}ln_f.weight"));
    names.push(format!("{prefix}ln_f.bias"));
    names
}

/// Read a `[rows, cols]` tensor into an `ndarray` matrix.
fn as_matrix(name: &str, tensor: &Tensor, rows: usize, cols: usize) -> Result<Array2<f32>> {
    let values = tensor.to_vec_f32()?;
    Array2::from_shape_vec((rows, cols), values).map_err(|error| {
        TrustformersError::shape_error(format!("checkpoint tensor {name}: {error}"))
    })
}

/// Read a `[len]` tensor into an `ndarray` vector.
fn as_vector(name: &str, tensor: &Tensor, len: usize) -> Result<Array1<f32>> {
    let values = tensor.to_vec_f32()?;
    if values.len() != len {
        return Err(TrustformersError::shape_error(format!(
            "checkpoint tensor {name} has {} value(s), expected {len}",
            values.len()
        )));
    }
    Ok(Array1::from_vec(values))
}

/// Bind one block's state-space kernel.
///
/// Every setter used here rebuilds the layer's discretisation, so the block runs
/// with the parameters that were just installed rather than with the ones it was
/// constructed with.
fn bind_kernel(
    binder: &mut WeightBinder<'_>,
    name: &str,
    config: &S4Config,
    layer: &mut S4Layer,
) -> Result<()> {
    let n = config.d_state;
    let h = config.get_n_ssm();

    if let Some(tensor) = binder.take_shaped(&format!("{name}.A_real"), &[n, n])? {
        layer.set_a_real(as_matrix("A_real", &tensor, n, n)?)?;
    }
    if let Some(tensor) = binder.take_shaped(&format!("{name}.A_imag"), &[n, n])? {
        layer.set_a_imag(as_matrix("A_imag", &tensor, n, n)?)?;
    }
    if let Some(tensor) = binder.take_shaped(&format!("{name}.B_real"), &[n])? {
        layer.set_b_real(as_vector("B_real", &tensor, n)?)?;
    }
    if let Some(tensor) = binder.take_shaped(&format!("{name}.B_imag"), &[n])? {
        layer.set_b_imag(as_vector("B_imag", &tensor, n)?)?;
    }
    if let Some(tensor) = binder.take_shaped(&format!("{name}.C_real"), &[n])? {
        layer.set_c_real(as_vector("C_real", &tensor, n)?)?;
    }
    if let Some(tensor) = binder.take_shaped(&format!("{name}.C_imag"), &[n])? {
        layer.set_c_imag(as_vector("C_imag", &tensor, n)?)?;
    }
    if let Some(tensor) = binder.take_shaped(&format!("{name}.D"), &[h])? {
        layer.set_d(as_vector("D", &tensor, h)?)?;
    }
    if let Some(tensor) = binder.take_shaped(&format!("{name}.log_dt"), &[h])? {
        // Δ = exp(log_dt): the contract stores the timestep in log space, so
        // binding the raw values would install a *different* timestep and every
        // subsequent forward pass would be quietly wrong.
        let log_dt = as_vector("log_dt", &tensor, h)?;
        layer.set_dt(log_dt.mapv(f32::exp))?;
    }
    Ok(())
}

/// Bind one block's norm and projections.
fn bind_block(
    binder: &mut WeightBinder<'_>,
    name: &str,
    config: &S4Config,
    block: &mut S4Block,
) -> Result<()> {
    let out_width = block.out_projection_width();
    let (norm, in_proj, out_proj) = block.parts_mut();

    if let Some(weight) = binder.take_shaped(&format!("{name}.norm.weight"), &[config.d_model])? {
        norm.set_weight(weight)?;
    }
    if let Some(bias) = binder.take_shaped(&format!("{name}.norm.bias"), &[config.d_model])? {
        norm.set_bias(bias)?;
    }
    bind_linear(
        binder,
        &format!("{name}.in_proj"),
        config.get_n_ssm(),
        config.d_model,
        config.use_bias,
        in_proj,
    )?;
    bind_linear(
        binder,
        &format!("{name}.out_proj"),
        out_width,
        config.get_n_ssm(),
        config.use_bias,
        out_proj,
    )?;
    bind_kernel(
        binder,
        &format!("{name}.kernel"),
        config,
        block.s4_layer_mut(),
    )
}

impl S4Model {
    /// Bind an already-parsed checkpoint into this backbone.
    ///
    /// # Errors
    ///
    /// Fails when the checkpoint does not carry `embeddings.weight` under any
    /// recognised prefix, when a tensor has the wrong shape, when a parameter is
    /// missing, or when an unrecognised tensor is present. The prefix-detection
    /// failure carries the full expected-name list.
    pub fn load_from_checkpoint(&mut self, checkpoint: &Checkpoint) -> Result<LoadReport> {
        let config = self.config.clone();
        let prefix =
            checkpoint
                .detect_prefix(&["", "backbone."], "embeddings.weight")
                .map_err(|error| {
                    TrustformersError::weight_load_error(format!(
                        "{error}. This crate binds S4 checkpoints against its declared contract \
                     (see the `s4::loading` module documentation); the expected tensors are: {}",
                        expected_tensor_names(&config, "").join(", ")
                    ))
                })?;
        let mut binder = checkpoint.binder(&prefix);

        bind_embedding(
            &mut binder,
            "embeddings",
            config.vocab_size,
            config.d_model,
            &mut self.embeddings,
        )?;
        for (index, block) in self.blocks.iter_mut().enumerate() {
            bind_block(&mut binder, &format!("blocks.{index}"), &config, block)?;
        }
        if let Some(weight) = binder.take_shaped("ln_f.weight", &[config.d_model])? {
            self.ln_f.set_weight(weight)?;
        }
        if let Some(bias) = binder.take_shaped("ln_f.bias", &[config.d_model])? {
            self.ln_f.set_bias(bias)?;
        }

        binder.finish(backbone_unused_policy())
    }

    /// Load pretrained weights and report exactly what was bound.
    ///
    /// # Errors
    ///
    /// See [`S4Model::load_from_checkpoint`].
    pub fn load_pretrained_report(&mut self, reader: &mut dyn std::io::Read) -> Result<LoadReport> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        self.load_from_checkpoint(&checkpoint)
    }
}

impl S4ForLanguageModeling {
    /// The checkpoint namespace this wrapper binds, and therefore must fully
    /// consume.
    const BOUND_NAMESPACES: BoundNamespaces<'static> = BoundNamespaces::new(&["lm_head."]);

    /// Load the backbone and, when the checkpoint carries one, the LM head.
    ///
    /// # Errors
    ///
    /// See [`S4Model::load_from_checkpoint`]; additionally fails when the head is
    /// present with the wrong shape or an unknown tensor sits under `lm_head.`.
    pub fn load_pretrained_report(&mut self, reader: &mut dyn std::io::Read) -> Result<LoadReport> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        let mut report = self.s4.load_from_checkpoint(&checkpoint)?;
        let config = self.s4.config.clone();
        bind_lm_head(
            &checkpoint,
            &mut report,
            config.vocab_size,
            config.d_model,
            &mut self.lm_head,
        )?;
        Self::BOUND_NAMESPACES.verify(&report)?;
        Ok(report)
    }
}

/// Bind a language-modelling head off an already-parsed checkpoint.
fn bind_lm_head(
    checkpoint: &Checkpoint,
    report: &mut LoadReport,
    vocab_size: usize,
    d_model: usize,
    head: &mut Linear,
) -> Result<()> {
    let name = "lm_head.weight";
    match checkpoint.take_shaped(name, &[vocab_size, d_model])? {
        Some(weight) => {
            head.set_weight(weight)?;
            report.mark_loaded(name);
        },
        None => report.note_absent(name),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_loading::test_support::{build_safetensors, write_temp_file, F32Tensor};
    use std::fs::File;
    use trustformers_core::traits::{Layer, Model};

    fn tiny_config() -> S4Config {
        S4Config {
            d_model: 4,
            d_state: 3,
            n_layer: 2,
            vocab_size: 10,
            max_position_embeddings: 16,
            n_ssm: Some(4),
            dt: 0.05,
            ..S4Config::default()
        }
    }

    /// Deterministic small values, distinct per tensor name.
    fn values(name: &str, count: usize) -> Vec<f32> {
        let mut state = name.bytes().fold(0x9E37_79B9_7F4A_7C15_u64, |acc, byte| {
            acc.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(u64::from(byte))
        }) | 1;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 40) as f32 / 16_777_216.0) * 0.4 - 0.2
            })
            .collect()
    }

    fn tensor(name: &str, shape: &[usize]) -> F32Tensor {
        let count: usize = shape.iter().product();
        F32Tensor::new(name, shape, values(name, count))
    }

    /// A checkpoint matching the declared contract exactly.
    fn fixture(config: &S4Config, prefix: &str, with_head: bool) -> Vec<F32Tensor> {
        let d_model = config.d_model;
        let n_ssm = config.get_n_ssm();
        let n = config.d_state;
        let out_width = if S4Block::uses_glu(config) { d_model * 2 } else { d_model };

        let mut tensors = vec![tensor(
            &format!("{prefix}embeddings.weight"),
            &[config.vocab_size, d_model],
        )];
        for block in 0..config.n_layer {
            let base = format!("{prefix}blocks.{block}");
            tensors.push(tensor(&format!("{base}.norm.weight"), &[d_model]));
            tensors.push(tensor(&format!("{base}.norm.bias"), &[d_model]));
            tensors.push(tensor(&format!("{base}.in_proj.weight"), &[n_ssm, d_model]));
            if config.use_bias {
                tensors.push(tensor(&format!("{base}.in_proj.bias"), &[n_ssm]));
            }
            tensors.push(tensor(
                &format!("{base}.out_proj.weight"),
                &[out_width, n_ssm],
            ));
            if config.use_bias {
                tensors.push(tensor(&format!("{base}.out_proj.bias"), &[out_width]));
            }
            tensors.push(tensor(&format!("{base}.kernel.A_real"), &[n, n]));
            tensors.push(tensor(&format!("{base}.kernel.A_imag"), &[n, n]));
            tensors.push(tensor(&format!("{base}.kernel.B_real"), &[n]));
            tensors.push(tensor(&format!("{base}.kernel.B_imag"), &[n]));
            tensors.push(tensor(&format!("{base}.kernel.C_real"), &[n]));
            tensors.push(tensor(&format!("{base}.kernel.C_imag"), &[n]));
            tensors.push(tensor(&format!("{base}.kernel.D"), &[n_ssm]));
            tensors.push(tensor(&format!("{base}.kernel.log_dt"), &[n_ssm]));
        }
        tensors.push(tensor(&format!("{prefix}ln_f.weight"), &[d_model]));
        tensors.push(tensor(&format!("{prefix}ln_f.bias"), &[d_model]));
        if with_head {
            tensors.push(tensor("lm_head.weight", &[config.vocab_size, d_model]));
        }
        tensors
    }

    fn input_ids() -> Tensor {
        let ids = scirs2_core::ndarray::Array1::from_vec(vec![1_i64, 4, 7]);
        Tensor::I64(ids.into_dyn())
    }

    fn hidden(model: &S4Model) -> Vec<f32> {
        model
            .forward(input_ids())
            .expect("forward must succeed")
            .to_vec_f32()
            .expect("output must be F32")
    }

    /// Every tensor in a contract-shaped checkpoint lands, read from a real file
    /// under `std::env::temp_dir()`.
    #[test]
    fn a_contract_checkpoint_round_trips_through_a_real_file() {
        let config = tiny_config();
        let tensors = fixture(&config, "", true);
        let path = write_temp_file("s4_round_trip", "safetensors", &build_safetensors(&tensors));
        let mut file = File::open(&path).expect("the fixture file must open");

        let mut model = S4ForLanguageModeling::new(config.clone()).expect("model must build");
        let report = model
            .load_pretrained_report(&mut file)
            .expect("a contract-shaped checkpoint must load");
        let _ = std::fs::remove_file(&path);

        assert!(
            report.is_complete(),
            "every parameter must be filled, missing: {:?}",
            report.missing
        );
        assert!(
            report.unexpected.is_empty(),
            "nothing may be left unrecognised: {:?}",
            report.unexpected
        );
        for name in expected_tensor_names(&config, "") {
            assert!(
                report.loaded.iter().any(|loaded| loaded == &name),
                "{name} must be reported as loaded"
            );
        }
        assert_eq!(
            report.loaded.len(),
            tensors.len(),
            "exactly the checkpoint's tensors must be accounted for"
        );
    }

    /// Binding rebuilds the discretisation, so the model computes with the
    /// parameters that were just installed.
    ///
    /// This is the silent-wrong-answer this loader was blocked on: a binder that
    /// installs `A`, `B` and `Δ` but leaves the cached `Ā`/`B̄` alone reports a
    /// successful load while every forward pass still runs the *constructor's*
    /// discretisation. The generation counter and the changed output together
    /// show it did not happen.
    #[test]
    fn binding_rebuilds_the_discretization_so_the_forward_moves() {
        let config = tiny_config();
        let tensors = fixture(&config, "", false);

        let mut model = S4Model::new(config.clone()).expect("model must build");
        let before = hidden(&model);
        let generations_before: Vec<u64> = model
            .blocks
            .iter()
            .map(|block| block.s4_layer().discretization_generation())
            .collect();

        model
            .load_pretrained(&mut build_safetensors(&tensors).as_slice())
            .expect("the checkpoint must load");

        let generations_after: Vec<u64> = model
            .blocks
            .iter()
            .map(|block| block.s4_layer().discretization_generation())
            .collect();
        for (index, (before_gen, after_gen)) in
            generations_before.iter().zip(generations_after.iter()).enumerate()
        {
            assert!(
                after_gen > before_gen,
                "block {index}: binding must rebuild the discretisation ({before_gen} -> \
                 {after_gen})"
            );
        }

        let after = hidden(&model);
        assert_ne!(
            before, after,
            "a bound checkpoint must change what the model computes"
        );
        assert!(
            after.iter().all(|value| value.is_finite()),
            "a loaded model must produce finite hidden states"
        );
    }

    /// `log_dt` is exponentiated, not installed raw.
    ///
    /// Binding a tensor named `log_dt` straight into `Δ` would be a new
    /// silent-wrong-answer: a checkpoint recording `log_dt = 0` would install a
    /// timestep of 0 instead of 1.
    #[test]
    fn log_dt_is_exponentiated_on_the_way_in() {
        let config = tiny_config();
        let n_ssm = config.get_n_ssm();
        let mut tensors = fixture(&config, "", false);
        let target = "blocks.0.kernel.log_dt";
        let entry = tensors
            .iter_mut()
            .find(|entry| entry.name == target)
            .expect("the fixture must carry log_dt");
        // ln(0.25) — the installed timestep must be 0.25, not this value.
        entry.values = vec![0.25_f32.ln(); n_ssm];

        let mut model = S4Model::new(config).expect("model must build");
        model
            .load_pretrained(&mut build_safetensors(&tensors).as_slice())
            .expect("the checkpoint must load");
        for value in model.blocks[0].s4_layer().dt().iter() {
            assert!(
                (value - 0.25).abs() < 1e-5,
                "Δ must be exp(log_dt) = 0.25, got {value}"
            );
        }
    }

    /// Every kernel parameter reaches the model: changing any one of them
    /// changes the output.
    #[test]
    fn each_kernel_parameter_reaches_the_model() {
        let config = tiny_config();
        let tensors = fixture(&config, "", false);

        let mut baseline = S4Model::new(config.clone()).expect("model must build");
        baseline
            .load_pretrained(&mut build_safetensors(&tensors).as_slice())
            .expect("the pristine checkpoint must load");
        let before = hidden(&baseline);

        for parameter in [
            "A_real", "A_imag", "B_real", "B_imag", "C_real", "C_imag", "D", "log_dt",
        ] {
            let target = format!("blocks.0.kernel.{parameter}");
            let mut tampered = tensors.clone();
            let entry = tampered
                .iter_mut()
                .find(|entry| entry.name == target)
                .unwrap_or_else(|| panic!("{target} must be in the fixture"));
            for value in &mut entry.values {
                *value += 0.5;
            }

            let mut model = S4Model::new(config.clone()).expect("model must build");
            model
                .load_pretrained(&mut build_safetensors(&tampered).as_slice())
                .expect("the tampered checkpoint must still load");
            assert_ne!(
                hidden(&model),
                before,
                "changing {target} must change the model's output"
            );
        }
    }

    /// An unrecognised tensor fails the load rather than being ignored.
    #[test]
    fn an_unknown_tensor_fails_the_load() {
        let config = tiny_config();
        let mut tensors = fixture(&config, "", true);
        tensors.push(tensor("blocks.0.kernel.log_dtt", &[config.get_n_ssm()]));

        let mut model = S4ForLanguageModeling::new(config).expect("model must build");
        let error = model
            .load_pretrained(&mut build_safetensors(&tensors).as_slice())
            .expect_err("a misspelt tensor name must not be tolerated");
        assert!(
            error.to_string().contains("log_dtt"),
            "the offending name must be reported: {error}"
        );
    }

    /// A missing kernel parameter is reported rather than left at its
    /// constructor value.
    #[test]
    fn a_missing_kernel_parameter_fails_the_load() {
        let config = tiny_config();
        let tensors: Vec<F32Tensor> = fixture(&config, "", false)
            .into_iter()
            .filter(|entry| entry.name != "blocks.1.kernel.C_imag")
            .collect();

        let mut model = S4Model::new(config).expect("model must build");
        let error = model
            .load_pretrained(&mut build_safetensors(&tensors).as_slice())
            .expect_err("an absent kernel parameter must fail the load");
        assert!(
            error.to_string().contains("C_imag"),
            "the missing name must be reported: {error}"
        );
    }

    /// A foreign checkpoint is refused with the whole contract, not a bare
    /// "tensor not found".
    #[test]
    fn a_foreign_checkpoint_is_refused_with_the_expected_name_list() {
        let config = tiny_config();
        let bytes = build_safetensors(&[tensor("model.embed_tokens.weight", &[4, 4])]);
        let mut model = S4Model::new(config).expect("model must build");
        let error = model
            .load_pretrained_report(&mut bytes.as_slice())
            .expect_err("a foreign checkpoint must be refused");
        let message = error.to_string();
        for expected in [
            "embeddings.weight",
            "blocks.0.kernel.A_real",
            "blocks.0.kernel.log_dt",
            "ln_f.weight",
        ] {
            assert!(
                message.contains(expected),
                "the refusal must enumerate the contract; {expected} is missing from: {message}"
            );
        }
    }

    /// The `backbone.` prefix is detected as well as the bare one.
    #[test]
    fn the_backbone_prefix_is_detected() {
        let config = tiny_config();
        let tensors = fixture(&config, "backbone.", false);
        let mut model = S4Model::new(config.clone()).expect("model must build");
        let report = model
            .load_pretrained_report(&mut build_safetensors(&tensors).as_slice())
            .expect("a `backbone.`-prefixed export must load");
        assert!(report.is_complete(), "missing: {:?}", report.missing);
        assert_eq!(
            report.loaded.len(),
            expected_tensor_names(&config, "backbone.").len()
        );
    }

    /// A backbone-only checkpoint still loads; the absent head is reported.
    #[test]
    fn a_backbone_only_checkpoint_reports_the_absent_head() {
        let config = tiny_config();
        let tensors = fixture(&config, "", false);
        let mut model = S4ForLanguageModeling::new(config).expect("model must build");
        let report = model
            .load_pretrained_report(&mut build_safetensors(&tensors).as_slice())
            .expect("a backbone-only checkpoint must load");
        assert!(
            report.missing.iter().any(|name| name == "lm_head.weight"),
            "the absent head must be named: {:?}",
            report.missing
        );
    }

    /// The declared contract and the binder agree about `out_proj`'s width, which
    /// depends on whether the block gates its output.
    #[test]
    fn the_contract_tracks_the_output_gate() {
        let mut gated = tiny_config();
        gated.postact = "glu".to_string();
        assert!(S4Block::uses_glu(&gated));
        let block = S4Block::new(&gated).expect("block must build");
        assert_eq!(block.out_projection_width(), gated.d_model * 2);

        let mut plain = tiny_config();
        plain.postact = "none".to_string();
        assert!(!S4Block::uses_glu(&plain));
        let block = S4Block::new(&plain).expect("block must build");
        assert_eq!(block.out_projection_width(), plain.d_model);

        // A checkpoint written for the ungated block must load into it.
        let tensors = fixture(&plain, "", false);
        let mut model = S4Model::new(plain).expect("model must build");
        let report = model
            .load_pretrained_report(&mut build_safetensors(&tensors).as_slice())
            .expect("an ungated export must load into an ungated model");
        assert!(report.is_complete(), "missing: {:?}", report.missing);
    }

    /// A gated checkpoint loaded into an ungated model fails on the shape rather
    /// than binding half of the projection.
    #[test]
    fn a_gate_mismatch_is_a_shape_error() {
        let mut gated = tiny_config();
        gated.postact = "glu".to_string();
        let tensors = fixture(&gated, "", false);

        let mut plain = tiny_config();
        plain.postact = "none".to_string();
        let mut model = S4Model::new(plain).expect("model must build");
        let error = model
            .load_pretrained(&mut build_safetensors(&tensors).as_slice())
            .expect_err("a 2x-wide output projection is not this model's");
        assert!(
            error.to_string().contains("out_proj"),
            "the mismatch must name the projection: {error}"
        );
    }

    /// The block-level forward runs the recurrence rather than returning its
    /// input untouched.
    ///
    /// Regression: `S4Block::forward` used to check `a_bar.is_none()` — always
    /// true, because nothing ever discretised the layer — and `return Ok(residual)`,
    /// making every S4 block an identity function. The branch it guarded filled
    /// the activation with the literal `0.1`.
    #[test]
    fn a_block_forward_is_not_the_identity() {
        let config = tiny_config();
        let block = S4Block::new(&config).expect("block must build");
        let values: Vec<f32> = (0..3 * config.d_model).map(|i| (i as f32 * 0.23).sin()).collect();
        let input =
            Tensor::from_vec(values.clone(), &[1, 3, config.d_model]).expect("input must build");

        let output = block.forward(input).expect("forward must succeed");
        let produced = output.to_vec_f32().expect("output must be F32");
        assert_eq!(output.shape(), vec![1, 3, config.d_model]);
        assert_ne!(produced, values, "the block must not be the identity");
        assert!(
            produced.iter().all(|value| value.is_finite()),
            "the block must produce finite values"
        );
        assert!(
            produced.iter().any(|value| (value - 0.1).abs() > 1e-6),
            "the activation must not be the literal placeholder 0.1"
        );
    }
}
