//! Analytic gradients and SGD updates for [`PolicyModel`] and [`ValueModel`].
//!
//! [`crate::rlhf::ppo`] defines the forward pass of the lightweight causal language model
//! used throughout the RLHF stack; this module supplies the matching backward pass so that
//! the SFT, DPO and PPO phases perform **real** parameter updates instead of arithmetic on
//! hand-rolled proxies.
//!
//! # Model recap
//!
//! ```text
//! ctx    = Σ_i decay^(n-1-i) · E[t_i] / Σ_i decay^(n-1-i)      (pooled prefix)
//! pre    = ctx · W_hidden                                       (optional)
//! h      = tanh(pre)                                            (h = ctx when W_hidden is absent)
//! logits = h · W_out + b_out
//! p      = softmax(logits)
//! ```
//!
//! All partial derivatives below are the exact derivatives of that expression.

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use scirs2_core::ndarray::Array2; // SciRS2 Integration Policy

use crate::rlhf::ppo::{
    pool_context, PolicyModel, ValueModel, CONTEXT_DECAY, PARAM_EMBEDDING, PARAM_HIDDEN_WEIGHT,
    PARAM_OUTPUT_BIAS, PARAM_OUTPUT_WEIGHT, PARAM_VALUE_BIAS, PARAM_VALUE_HEAD,
};

/// Accumulated parameter gradients, keyed like the model's `parameters` map.
pub type Gradients = HashMap<String, Array2<f32>>;

/// Add `value` into `grads[key][row, col]`, creating a zero matrix of `shape` if needed.
fn accumulate(
    grads: &mut Gradients,
    key: &str,
    shape: (usize, usize),
    row: usize,
    col: usize,
    value: f32,
) {
    let entry = grads.entry(key.to_string()).or_insert_with(|| Array2::zeros(shape));
    entry[[row, col]] += value;
}

/// Apply `p -= learning_rate * grad` for every parameter that has a gradient.
///
/// # Errors
///
/// Fails when a gradient names a parameter the model does not have, or when the shapes
/// disagree — both indicate a bug rather than a recoverable condition.
pub fn apply_sgd(
    parameters: &mut HashMap<String, Array2<f32>>,
    grads: &Gradients,
    learning_rate: f32,
) -> Result<()> {
    for (key, grad) in grads {
        let param = parameters
            .get_mut(key)
            .ok_or_else(|| anyhow!("gradient computed for unknown parameter '{key}'"))?;
        if param.dim() != grad.dim() {
            return Err(anyhow!(
                "gradient for '{key}' has shape {:?} but the parameter has shape {:?}",
                grad.dim(),
                param.dim()
            ));
        }
        *param -= &(grad * learning_rate);
    }
    Ok(())
}

/// Softmax of `logits`, numerically stabilised by subtracting the maximum.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for e in exps.iter_mut() {
            *e /= sum;
        }
    }
    exps
}

/// Forward state cached for one prefix so the backward pass need not recompute it.
struct Activations {
    ctx: Vec<f32>,
    /// `tanh` output, or a copy of `ctx` when the model has no hidden transform.
    hidden: Vec<f32>,
    has_hidden: bool,
    logits: Vec<f32>,
}

/// Run the policy forward pass over `prefix`, keeping the intermediate activations.
fn forward_with_activations(model: &PolicyModel, prefix: &[u32]) -> Result<Activations> {
    let embedding = model
        .parameters
        .get(PARAM_EMBEDDING)
        .ok_or_else(|| anyhow!("policy '{}' is missing '{PARAM_EMBEDDING}'", model.model_id))?;
    let ctx = pool_context(embedding, prefix, model.hidden_size)?;

    let (hidden, has_hidden) = match model.parameters.get(PARAM_HIDDEN_WEIGHT) {
        Some(w) => {
            let mut out = vec![0.0f32; model.hidden_size];
            for (j, slot) in out.iter_mut().enumerate() {
                let mut acc = 0.0f32;
                for (k, &c) in ctx.iter().enumerate() {
                    acc += c * w[[k, j]];
                }
                *slot = acc.tanh();
            }
            (out, true)
        },
        None => (ctx.clone(), false),
    };

    let output = model.parameters.get(PARAM_OUTPUT_WEIGHT).ok_or_else(|| {
        anyhow!(
            "policy '{}' is missing '{PARAM_OUTPUT_WEIGHT}'",
            model.model_id
        )
    })?;
    let mut logits = vec![0.0f32; model.vocab_size];
    for (v, logit) in logits.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (k, &hv) in hidden.iter().enumerate() {
            acc += hv * output[[k, v]];
        }
        *logit = acc;
    }
    if let Some(bias) = model.parameters.get(PARAM_OUTPUT_BIAS) {
        for (v, logit) in logits.iter_mut().enumerate() {
            *logit += bias[[0, v]];
        }
    }

    Ok(Activations {
        ctx,
        hidden,
        has_hidden,
        logits,
    })
}

/// Log-probability of `target` given `prefix`, accumulating `upstream · ∂logp/∂θ` into `grads`.
///
/// Pass `upstream = 1.0` to accumulate the gradient of the log-probability itself, or
/// `upstream = -1.0 / n` to accumulate the gradient of a mean negative-log-likelihood.
///
/// # Errors
///
/// Fails when the model is missing a required parameter, when the prefix is empty, or when
/// a token id falls outside the model vocabulary.
pub fn token_logprob_backward(
    model: &PolicyModel,
    prefix: &[u32],
    target: u32,
    upstream: f32,
    grads: &mut Gradients,
) -> Result<f32> {
    let target_idx = target as usize;
    if target_idx >= model.vocab_size {
        return Err(anyhow!(
            "target token {target_idx} is outside the vocabulary [0, {})",
            model.vocab_size
        ));
    }

    let act = forward_with_activations(model, prefix)?;
    let probs = softmax(&act.logits);
    let log_prob = probs[target_idx].max(f32::MIN_POSITIVE).ln();

    // Scoring-only path: an upstream of exactly zero contributes nothing, so skip the whole
    // backward pass (and the allocations it would make) rather than writing zero matrices.
    if upstream == 0.0 {
        return Ok(log_prob);
    }

    let d = model.hidden_size;
    let v_size = model.vocab_size;

    // ∂ log p_target / ∂ logits_v = [v == target] - p_v
    let mut d_logits = vec![0.0f32; v_size];
    for (v, slot) in d_logits.iter_mut().enumerate() {
        *slot = upstream * (if v == target_idx { 1.0 } else { 0.0 } - probs[v]);
    }

    // Output projection and bias.
    let output = model
        .parameters
        .get(PARAM_OUTPUT_WEIGHT)
        .ok_or_else(|| {
            anyhow!(
                "policy '{}' is missing '{PARAM_OUTPUT_WEIGHT}'",
                model.model_id
            )
        })?
        .clone();
    for v in 0..v_size {
        let dl = d_logits[v];
        if dl == 0.0 {
            continue;
        }
        for (k, &hv) in act.hidden.iter().enumerate() {
            accumulate(grads, PARAM_OUTPUT_WEIGHT, (d, v_size), k, v, hv * dl);
        }
    }
    if model.parameters.contains_key(PARAM_OUTPUT_BIAS) {
        for (v, &dl) in d_logits.iter().enumerate() {
            accumulate(grads, PARAM_OUTPUT_BIAS, (1, v_size), 0, v, dl);
        }
    }

    // ∂/∂hidden
    let mut d_hidden = vec![0.0f32; d];
    for (k, slot) in d_hidden.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (v, &dl) in d_logits.iter().enumerate() {
            acc += output[[k, v]] * dl;
        }
        *slot = acc;
    }

    // ∂/∂ctx, going through the optional tanh layer.
    let d_ctx = if act.has_hidden {
        let hidden_w = model
            .parameters
            .get(PARAM_HIDDEN_WEIGHT)
            .ok_or_else(|| anyhow!("policy '{}' lost '{PARAM_HIDDEN_WEIGHT}'", model.model_id))?
            .clone();
        // d(tanh)/d(pre) = 1 - tanh^2
        let d_pre: Vec<f32> =
            (0..d).map(|j| d_hidden[j] * (1.0 - act.hidden[j] * act.hidden[j])).collect();
        for j in 0..d {
            if d_pre[j] == 0.0 {
                continue;
            }
            for (k, &c) in act.ctx.iter().enumerate() {
                accumulate(grads, PARAM_HIDDEN_WEIGHT, (d, d), k, j, c * d_pre[j]);
            }
        }
        let mut d_ctx = vec![0.0f32; d];
        for (k, slot) in d_ctx.iter_mut().enumerate() {
            let mut acc = 0.0f32;
            for (j, &dp) in d_pre.iter().enumerate() {
                acc += hidden_w[[k, j]] * dp;
            }
            *slot = acc;
        }
        d_ctx
    } else {
        d_hidden
    };

    // ∂/∂embedding: ctx is the normalised, recency-weighted sum of the prefix rows.
    let n = prefix.len();
    let weight_sum: f32 = (0..n).map(|i| CONTEXT_DECAY.powi((n - 1 - i) as i32)).sum();
    if weight_sum > 0.0 {
        for (i, &token) in prefix.iter().enumerate() {
            let token_idx = token as usize;
            let share = CONTEXT_DECAY.powi((n - 1 - i) as i32) / weight_sum;
            for (k, &dc) in d_ctx.iter().enumerate() {
                accumulate(
                    grads,
                    PARAM_EMBEDDING,
                    (model.vocab_size, d),
                    token_idx,
                    k,
                    share * dc,
                );
            }
        }
    }

    Ok(log_prob)
}

/// Total log-probability of `continuation` given `prompt`, with no gradient bookkeeping.
///
/// Tokens are scored autoregressively: token `j` is conditioned on `prompt` followed by the
/// first `j` continuation tokens.
pub fn sequence_log_prob(model: &PolicyModel, prompt: &[u32], continuation: &[u32]) -> Result<f32> {
    let mut discard = Gradients::new();
    sequence_log_prob_backward(model, prompt, continuation, 0.0, &mut discard)
}

/// Total log-probability of `continuation`, accumulating `upstream · ∂logp/∂θ` into `grads`.
///
/// With `upstream == 0.0` no gradient is recorded, which makes this the single code path for
/// both scoring and training.
pub fn sequence_log_prob_backward(
    model: &PolicyModel,
    prompt: &[u32],
    continuation: &[u32],
    upstream: f32,
    grads: &mut Gradients,
) -> Result<f32> {
    if prompt.is_empty() {
        return Err(anyhow!("sequence log-probability needs a non-empty prompt"));
    }
    if continuation.is_empty() {
        return Err(anyhow!(
            "sequence log-probability needs a non-empty continuation"
        ));
    }

    let mut prefix = prompt.to_vec();
    let mut total = 0.0f32;
    for &token in continuation {
        total += token_logprob_backward(model, &prefix, token, upstream, grads)?;
        prefix.push(token);
    }
    Ok(total)
}

/// Result of one DPO step on a single preference triple.
#[derive(Debug, Clone, Copy)]
pub struct DpoStepOutcome {
    /// `-log σ(β · (Δ_chosen − Δ_rejected))`
    pub loss: f32,
    /// Implicit reward `β · Δ_chosen` for the chosen continuation.
    pub chosen_reward: f32,
    /// Implicit reward `β · Δ_rejected` for the rejected continuation.
    pub rejected_reward: f32,
}

/// DPO loss and policy gradients for one `(prompt, chosen, rejected)` triple.
///
/// With `Δ_y = log π_θ(y|x) − log π_ref(y|x)` and `z = β (Δ_chosen − Δ_rejected)`:
///
/// ```text
/// L                     = −log σ(z)
/// ∂L/∂z                 = −(1 − σ(z))
/// ∂L/∂log π_θ(chosen)   =  β · ∂L/∂z
/// ∂L/∂log π_θ(rejected) = −β · ∂L/∂z
/// ```
///
/// The reference policy is frozen, so only the two policy log-probabilities carry gradients.
pub fn dpo_loss_and_grads(
    policy: &PolicyModel,
    reference: &PolicyModel,
    prompt: &[u32],
    chosen: &[u32],
    rejected: &[u32],
    beta: f32,
    grads: &mut Gradients,
) -> Result<DpoStepOutcome> {
    // Forward-only pass first so the sigmoid is known before gradients are accumulated.
    let policy_chosen = sequence_log_prob(policy, prompt, chosen)?;
    let policy_rejected = sequence_log_prob(policy, prompt, rejected)?;
    let ref_chosen = sequence_log_prob(reference, prompt, chosen)?;
    let ref_rejected = sequence_log_prob(reference, prompt, rejected)?;

    let delta_chosen = policy_chosen - ref_chosen;
    let delta_rejected = policy_rejected - ref_rejected;
    let z = beta * (delta_chosen - delta_rejected);
    let sigma = 1.0 / (1.0 + (-z).exp());
    let loss = -sigma.max(f32::MIN_POSITIVE).ln();

    let dl_dz = -(1.0 - sigma);
    sequence_log_prob_backward(policy, prompt, chosen, beta * dl_dz, grads)?;
    sequence_log_prob_backward(policy, prompt, rejected, -beta * dl_dz, grads)?;

    Ok(DpoStepOutcome {
        loss,
        chosen_reward: beta * delta_chosen,
        rejected_reward: beta * delta_rejected,
    })
}

/// Squared-error loss of the value head against `target`, with gradients.
///
/// `value(seq) = tanh(ctx · w_v + b_v)`, so
/// `∂L/∂w_v = 2 (v − target) (1 − v²) ctx` and `∂L/∂b_v = 2 (v − target) (1 − v²)`.
pub fn value_loss_and_grads(
    model: &ValueModel,
    sequence: &[u32],
    target: f32,
    grads: &mut Gradients,
) -> Result<f32> {
    let embedding = model.parameters.get(PARAM_EMBEDDING).ok_or_else(|| {
        anyhow!(
            "value model '{}' is missing '{PARAM_EMBEDDING}'",
            model.model_id
        )
    })?;
    let ctx = pool_context(embedding, sequence, model.hidden_size)?;

    let head = model.parameters.get(PARAM_VALUE_HEAD).ok_or_else(|| {
        anyhow!(
            "value model '{}' is missing '{PARAM_VALUE_HEAD}'",
            model.model_id
        )
    })?;
    let mut pre = 0.0f32;
    for (k, &c) in ctx.iter().enumerate() {
        pre += c * head[[k, 0]];
    }
    if let Some(bias) = model.parameters.get(PARAM_VALUE_BIAS) {
        pre += bias[[0, 0]];
    }
    let value = pre.tanh();

    let error = value - target;
    let loss = error * error;
    let d_pre = 2.0 * error * (1.0 - value * value);

    let d = model.hidden_size;
    for (k, &c) in ctx.iter().enumerate() {
        accumulate(grads, PARAM_VALUE_HEAD, (d, 1), k, 0, c * d_pre);
    }
    if model.parameters.contains_key(PARAM_VALUE_BIAS) {
        accumulate(grads, PARAM_VALUE_BIAS, (1, 1), 0, 0, d_pre);
    }

    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_policy(seed: u64) -> PolicyModel {
        PolicyModel::new_initialized("tiny", 8, 4, seed).expect("policy")
    }

    #[test]
    fn test_token_logprob_matches_softmax_of_the_forward_logits() {
        let model = tiny_policy(1);
        let prefix = [1u32, 2, 3];
        let logits = model.logits(&prefix).expect("logits");
        let probs = softmax(&logits);

        let mut grads = Gradients::new();
        let logp = token_logprob_backward(&model, &prefix, 5, 0.0, &mut grads).expect("logprob");
        assert!(
            (logp - probs[5].ln()).abs() < 1e-5,
            "expected {}, got {logp}",
            probs[5].ln()
        );
        assert!(grads.is_empty(), "upstream 0 must not record gradients");
    }

    #[test]
    fn test_output_weight_gradient_matches_finite_differences() {
        // A numerical check of the analytic backward pass: perturbing one output weight must
        // change the log-probability by (gradient * epsilon).
        let mut model = tiny_policy(3);
        let prefix = [2u32, 6];
        let target = 4u32;

        let mut grads = Gradients::new();
        token_logprob_backward(&model, &prefix, target, 1.0, &mut grads).expect("backward");
        let analytic = grads.get(PARAM_OUTPUT_WEIGHT).expect("output grad")[[1, 4]];

        let eps = 1e-3f32;
        let base = {
            let mut g = Gradients::new();
            token_logprob_backward(&model, &prefix, target, 0.0, &mut g).expect("base")
        };
        if let Some(w) = model.parameters.get_mut(PARAM_OUTPUT_WEIGHT) {
            w[[1, 4]] += eps;
        }
        let bumped = {
            let mut g = Gradients::new();
            token_logprob_backward(&model, &prefix, target, 0.0, &mut g).expect("bumped")
        };
        let numeric = (bumped - base) / eps;
        assert!(
            (analytic - numeric).abs() < 5e-3,
            "analytic {analytic} vs numeric {numeric}"
        );
    }

    #[test]
    fn test_sgd_increases_the_log_probability_of_the_trained_token() {
        let mut model = tiny_policy(5);
        let prefix = [1u32, 3];
        let target = 6u32;

        let before = sequence_log_prob(&model, &prefix, &[target]).expect("before");
        for _ in 0..50 {
            let mut grads = Gradients::new();
            // upstream = +1 on the log-probability, so we *ascend* it: apply_sgd subtracts,
            // hence the negated learning rate.
            token_logprob_backward(&model, &prefix, target, 1.0, &mut grads).expect("backward");
            apply_sgd(&mut model.parameters, &grads, -0.5).expect("sgd");
        }
        let after = sequence_log_prob(&model, &prefix, &[target]).expect("after");
        assert!(
            after > before,
            "training must raise the target log-probability ({before} -> {after})"
        );
    }

    #[test]
    fn test_dpo_step_reduces_the_loss() {
        let mut policy = tiny_policy(9);
        let reference = policy.clone();
        let prompt = [1u32, 2];
        let chosen = [4u32, 5];
        let rejected = [6u32, 7];

        let mut grads = Gradients::new();
        let first = dpo_loss_and_grads(
            &policy, &reference, &prompt, &chosen, &rejected, 0.5, &mut grads,
        )
        .expect("dpo");
        apply_sgd(&mut policy.parameters, &grads, 0.5).expect("sgd");

        let mut last = first.loss;
        for _ in 0..30 {
            let mut grads = Gradients::new();
            let out = dpo_loss_and_grads(
                &policy, &reference, &prompt, &chosen, &rejected, 0.5, &mut grads,
            )
            .expect("dpo");
            apply_sgd(&mut policy.parameters, &grads, 0.5).expect("sgd");
            last = out.loss;
        }
        assert!(
            last < first.loss,
            "DPO training must reduce the loss ({} -> {last})",
            first.loss
        );
    }

    #[test]
    fn test_dpo_at_the_reference_starts_at_ln_two() {
        // With policy == reference both deltas are 0, so z = 0 and L = -log σ(0) = ln 2.
        let policy = tiny_policy(11);
        let reference = policy.clone();
        let mut grads = Gradients::new();
        let out = dpo_loss_and_grads(
            &policy,
            &reference,
            &[1u32],
            &[2u32],
            &[3u32],
            0.3,
            &mut grads,
        )
        .expect("dpo");
        assert!(
            (out.loss - 2.0f32.ln()).abs() < 1e-4,
            "expected ln 2, got {}",
            out.loss
        );
        assert!(out.chosen_reward.abs() < 1e-5);
        assert!(out.rejected_reward.abs() < 1e-5);
    }

    #[test]
    fn test_value_head_learns_its_target() {
        let mut model = ValueModel::new_initialized("v", 8, 4, 13).expect("value");
        let sequence = [1u32, 2, 3];
        let target = 0.6f32;

        let mut grads = Gradients::new();
        let first = value_loss_and_grads(&model, &sequence, target, &mut grads).expect("loss");
        apply_sgd(&mut model.parameters, &grads, 0.5).expect("sgd");

        let mut last = first;
        for _ in 0..200 {
            let mut grads = Gradients::new();
            last = value_loss_and_grads(&model, &sequence, target, &mut grads).expect("loss");
            apply_sgd(&mut model.parameters, &grads, 0.5).expect("sgd");
        }
        assert!(last < first, "value loss must decrease ({first} -> {last})");
        let v = model.value(&sequence).expect("value");
        assert!(
            (v - target).abs() < 0.2,
            "value should approach {target}, got {v}"
        );
    }
}
