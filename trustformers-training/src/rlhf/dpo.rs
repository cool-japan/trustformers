use crate::kto::{compute_kto_loss, KtoConfig, KtoExample};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Model;

/// Default weight for the desirable (chosen) side of the KTO objective.
fn default_kto_lambda() -> f32 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DPOConfig {
    pub beta: f32,
    pub label_smoothing: f32,
    pub loss_type: DPOLossType,
    pub reference_free: bool,
    pub label_pad_token_id: i32,
    pub padding_value: f32,
    pub truncation_mode: String,
    pub max_length: Option<usize>,
    pub max_target_length: Option<usize>,
    pub max_prompt_length: Option<usize>,
    pub generate_during_eval: bool,
    /// KTO only: weight λ_w applied to the desirable (chosen) loss term.
    #[serde(default = "default_kto_lambda")]
    pub kto_lambda_preferred: f32,
    /// KTO only: weight λ_l applied to the undesirable (rejected) loss term.
    #[serde(default = "default_kto_lambda")]
    pub kto_lambda_rejected: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DPOLossType {
    Sigmoid,
    Hinge,
    Ipo,
    Kto,
}

impl Default for DPOConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            label_smoothing: 0.0,
            loss_type: DPOLossType::Sigmoid,
            reference_free: false,
            label_pad_token_id: -100,
            padding_value: 0.0,
            truncation_mode: "keep_end".to_string(),
            max_length: Some(512),
            max_target_length: Some(128),
            max_prompt_length: Some(128),
            generate_during_eval: false,
            kto_lambda_preferred: 1.0,
            kto_lambda_rejected: 1.0,
        }
    }
}

#[derive(Debug)]
pub struct DPOTrainer<M: Model> {
    pub model: M,
    pub ref_model: Option<M>,
    pub config: DPOConfig,
    pub data_collator: DPODataCollator,
}

impl<M: Model<Input = Tensor, Output = Tensor>> DPOTrainer<M> {
    pub fn new(model: M, ref_model: Option<M>, config: DPOConfig) -> Self {
        Self {
            model,
            ref_model,
            config: config.clone(),
            data_collator: DPODataCollator::new(config),
        }
    }

    /// Build the KTO example batch implied by a paired preference batch.
    ///
    /// KTO itself is unpaired: it consumes `(policy_logp, reference_logp, is_preferred)`
    /// triples. A DPO batch supplies exactly that information twice per row — the chosen
    /// completion is a desirable example and the rejected completion an undesirable one —
    /// so the pairing is unfolded here and the real prospect-theoretic objective in
    /// [`crate::kto`] does the rest.
    fn kto_examples(
        policy_chosen_logps: &Tensor,
        policy_rejected_logps: &Tensor,
        reference_chosen_logps: &Tensor,
        reference_rejected_logps: &Tensor,
    ) -> Result<Vec<KtoExample>> {
        let policy_chosen = policy_chosen_logps.data()?;
        let policy_rejected = policy_rejected_logps.data()?;
        let ref_chosen = reference_chosen_logps.data()?;
        let ref_rejected = reference_rejected_logps.data()?;

        if policy_chosen.len() != policy_rejected.len()
            || policy_chosen.len() != ref_chosen.len()
            || policy_chosen.len() != ref_rejected.len()
        {
            return Err(anyhow!(
                "KTO loss: policy/reference log-prob tensors must have equal length \
                 (policy_chosen={}, policy_rejected={}, ref_chosen={}, ref_rejected={})",
                policy_chosen.len(),
                policy_rejected.len(),
                ref_chosen.len(),
                ref_rejected.len()
            ));
        }

        let mut examples = Vec::with_capacity(policy_chosen.len() * 2);
        for i in 0..policy_chosen.len() {
            examples.push(KtoExample {
                policy_log_prob: policy_chosen[i],
                reference_log_prob: ref_chosen[i],
                is_preferred: true,
            });
            examples.push(KtoExample {
                policy_log_prob: policy_rejected[i],
                reference_log_prob: ref_rejected[i],
                is_preferred: false,
            });
        }
        Ok(examples)
    }

    pub fn compute_loss(
        &self,
        policy_chosen_logps: &Tensor,
        policy_rejected_logps: &Tensor,
        reference_chosen_logps: &Tensor,
        reference_rejected_logps: &Tensor,
    ) -> Result<Tensor> {
        // KTO is not a re-parameterisation of the DPO logit; it needs the raw per-example
        // log-ratios and an explicit reference point, so it is dispatched before the shared
        // `logits` term is formed.
        if matches!(self.config.loss_type, DPOLossType::Kto) {
            let examples = Self::kto_examples(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )?;
            let kto_config = KtoConfig {
                beta: self.config.beta,
                lambda_preferred: self.config.kto_lambda_preferred,
                lambda_rejected: self.config.kto_lambda_rejected,
                ..KtoConfig::default()
            };
            let result = compute_kto_loss(&examples, &kto_config)?;
            return Ok(Tensor::new(vec![result.total_loss])?);
        }

        let pi_logratios = policy_chosen_logps.sub(policy_rejected_logps)?;
        let ref_logratios = reference_chosen_logps.sub(reference_rejected_logps)?;
        let logits = pi_logratios.sub(&ref_logratios)?.mul_scalar(self.config.beta)?;

        match self.config.loss_type {
            DPOLossType::Sigmoid => {
                // DPO loss: -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
                // = -log(sigmoid(logits))
                let loss = logits.sigmoid()?.log()?.neg()?;
                Ok(loss.mean()?)
            },
            DPOLossType::Hinge => {
                // Hinge loss: max(0, 1 - logits)
                let logits_shape = logits.shape();
                let ones = Tensor::ones(&logits_shape)?;
                let hinge = ones.sub(&logits)?.relu()?;
                Ok(hinge.mean()?)
            },
            DPOLossType::Ipo => {
                // IPO loss: (logits - 1/2)^2
                let half = logits.sub_scalar(0.5)?;
                let loss = half.pow(2.0)?;
                Ok(loss.mean()?)
            },
            DPOLossType::Kto => {
                // Handled above by the real prospect-theoretic objective in `crate::kto`;
                // reaching this arm would mean the early dispatch was removed.
                Err(anyhow!(
                    "internal error: DPOLossType::Kto must be dispatched to crate::kto"
                ))
            },
        }
    }

    /// Compute preference accuracy: fraction of examples where the model
    /// prefers the chosen response over the rejected one.
    /// accuracy = (logits > 0).float().mean()
    /// where logits = beta * (log_ratio_chosen - log_ratio_rejected)
    pub fn compute_preference_accuracy(
        &self,
        policy_chosen_logps: &Tensor,
        policy_rejected_logps: &Tensor,
        reference_chosen_logps: &Tensor,
        reference_rejected_logps: &Tensor,
    ) -> Result<Tensor> {
        let pi_logratios = policy_chosen_logps.sub(policy_rejected_logps)?;
        let ref_logratios = reference_chosen_logps.sub(reference_rejected_logps)?;
        let logits = pi_logratios.sub(&ref_logratios)?.mul_scalar(self.config.beta)?;

        // (logits > 0).float().mean()
        let zeros = Tensor::zeros(&logits.shape())?;
        let preferred = logits.greater(&zeros)?;
        Ok(preferred.mean()?)
    }

    /// Per-sequence log-probability of the label tokens under `logits`.
    ///
    /// This is the exact quantity the DPO/IPO/KTO objectives are defined on:
    ///
    /// ```text
    /// logps[b] = Σ_t  m[b,t] · log_softmax(logits[b, t, :])[ labels[b, t] ]
    /// ```
    ///
    /// with `m[b,t] = 0` wherever `labels[b,t] == config.label_pad_token_id` (padding and
    /// prompt positions the caller masked out). When `average_log_prob` is `true` the sum is
    /// divided by the number of unmasked positions in that sequence, giving a
    /// length-normalised log-probability; a sequence with no unmasked position contributes
    /// `0.0`.
    ///
    /// # Alignment contract
    ///
    /// `labels` are taken to be **already aligned** with `logits`: `labels[b, t]` is the token
    /// scored by `logits[b, t, :]`. Any next-token shift (`logits[:, :-1]` vs `labels[:, 1:]`)
    /// is the caller's responsibility — [`DPOExample`] carries `chosen_labels` separately from
    /// `chosen_input_ids` precisely so that the shift can be baked into the labels.
    ///
    /// # Errors
    ///
    /// * `logits` is not rank 3 (`[batch, seq_len, vocab]`) or `labels` is not rank 2.
    /// * the batch/sequence dimensions of `logits` and `labels` disagree.
    /// * a non-padding label is negative or `>= vocab_size`.
    pub fn get_batch_logps(
        &self,
        logits: &Tensor,
        labels: &Tensor,
        average_log_prob: bool,
    ) -> Result<Tensor> {
        let logits_shape = logits.shape();
        let labels_shape = labels.shape();

        if logits_shape.len() != 3 {
            return Err(anyhow!(
                "get_batch_logps: logits must be rank 3 [batch, seq_len, vocab], got {:?}",
                logits_shape
            ));
        }
        if labels_shape.len() != 2 {
            return Err(anyhow!(
                "get_batch_logps: labels must be rank 2 [batch, seq_len], got {:?}",
                labels_shape
            ));
        }

        let (batch_size, seq_len, vocab_size) = (logits_shape[0], logits_shape[1], logits_shape[2]);
        if labels_shape[0] != batch_size || labels_shape[1] != seq_len {
            return Err(anyhow!(
                "get_batch_logps: labels shape {:?} does not match logits shape {:?}",
                labels_shape,
                logits_shape
            ));
        }
        if vocab_size == 0 {
            return Err(anyhow!("get_batch_logps: vocabulary dimension is empty"));
        }

        // Raw label ids. The collator stores them as f32, so round-trip through f32 here and
        // validate before they are ever used as an index.
        let raw_labels = labels.data()?;
        let pad_id = self.config.label_pad_token_id as f32;

        let mut mask = Vec::with_capacity(raw_labels.len());
        // `Tensor::gather` indexes with `as usize`, so a padding id of -100 would wrap to a
        // huge index. Padding positions are therefore clamped to 0 and removed afterwards by
        // the mask; real ids are bounds-checked instead of clamped so that an out-of-range
        // label surfaces as an error rather than as a silently wrong log-probability.
        let mut index_values = Vec::with_capacity(raw_labels.len());
        for (flat, &label) in raw_labels.iter().enumerate() {
            if (label - pad_id).abs() < 0.5 {
                mask.push(false);
                index_values.push(0.0f32);
                continue;
            }
            if label < 0.0 || label >= vocab_size as f32 {
                let (b, t) = (flat / seq_len, flat % seq_len);
                return Err(anyhow!(
                    "get_batch_logps: label {label} at [{b}, {t}] is outside the vocabulary \
                     [0, {vocab_size}) and is not the padding id {}",
                    self.config.label_pad_token_id
                ));
            }
            mask.push(true);
            index_values.push(label);
        }

        // log_softmax over the vocabulary axis, then gather the label's entry at every
        // position: index [B, T, 1] on dim 2 selects log_probs[b, t, labels[b, t]].
        let log_probs = logits.log_softmax(-1)?;
        let index = Tensor::from_vec(index_values, &[batch_size, seq_len, 1])?.to_i64()?;
        let gathered = log_probs.gather(2, &index)?.data()?;

        let mut batch_logps = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let mut sum = 0.0f32;
            let mut count = 0usize;
            for t in 0..seq_len {
                let flat = b * seq_len + t;
                if mask[flat] {
                    sum += gathered[flat];
                    count += 1;
                }
            }
            let value = if average_log_prob {
                if count == 0 {
                    0.0
                } else {
                    sum / count as f32
                }
            } else {
                sum
            };
            batch_logps.push(value);
        }

        Ok(Tensor::new(batch_logps)?)
    }

    pub fn train_step(&mut self, batch: &DPOBatch) -> Result<DPOLoss> {
        // Forward pass for chosen and rejected sequences
        let chosen_outputs = self.model.forward(batch.chosen_input_ids.clone())?;
        let rejected_outputs = self.model.forward(batch.rejected_input_ids.clone())?;

        // Compute log probabilities
        let policy_chosen_logps =
            self.get_batch_logps(&chosen_outputs, &batch.chosen_labels, true)?;
        let policy_rejected_logps =
            self.get_batch_logps(&rejected_outputs, &batch.rejected_labels, true)?;

        // Reference model forward pass (if available)
        let (reference_chosen_logps, reference_rejected_logps) =
            if let Some(ref_model) = &self.ref_model {
                let ref_chosen_outputs = ref_model.forward(batch.chosen_input_ids.clone())?;
                let ref_rejected_outputs = ref_model.forward(batch.rejected_input_ids.clone())?;

                let ref_chosen_logps =
                    self.get_batch_logps(&ref_chosen_outputs, &batch.chosen_labels, true)?;
                let ref_rejected_logps =
                    self.get_batch_logps(&ref_rejected_outputs, &batch.rejected_labels, true)?;

                (ref_chosen_logps, ref_rejected_logps)
            } else {
                // Reference-free mode: use zeros
                let batch_size = policy_chosen_logps.shape()[0];
                let zeros = Tensor::zeros(&[batch_size])?;
                (zeros.clone(), zeros)
            };

        // Compute DPO loss
        let loss = self.compute_loss(
            &policy_chosen_logps,
            &policy_rejected_logps,
            &reference_chosen_logps,
            &reference_rejected_logps,
        )?;

        // Compute reward margins and accuracy
        let chosen_rewards =
            policy_chosen_logps.sub(&reference_chosen_logps)?.mul_scalar(self.config.beta)?;
        let rejected_rewards = policy_rejected_logps
            .sub(&reference_rejected_logps)?
            .mul_scalar(self.config.beta)?;
        let reward_margins = chosen_rewards.sub(&rejected_rewards)?;

        // Compute preference accuracy: fraction of examples where chosen is preferred
        let accuracy = self.compute_preference_accuracy(
            &policy_chosen_logps,
            &policy_rejected_logps,
            &reference_chosen_logps,
            &reference_rejected_logps,
        )?;

        Ok(DPOLoss {
            loss,
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            chosen_rewards,
            rejected_rewards,
            reward_margins,
            accuracy,
        })
    }
}

#[derive(Debug)]
pub struct DPOBatch {
    pub chosen_input_ids: Tensor,
    pub chosen_labels: Tensor,
    pub chosen_attention_mask: Tensor,
    pub rejected_input_ids: Tensor,
    pub rejected_labels: Tensor,
    pub rejected_attention_mask: Tensor,
}

#[derive(Debug)]
pub struct DPOLoss {
    pub loss: Tensor,
    pub policy_chosen_logps: Tensor,
    pub policy_rejected_logps: Tensor,
    pub reference_chosen_logps: Tensor,
    pub reference_rejected_logps: Tensor,
    pub chosen_rewards: Tensor,
    pub rejected_rewards: Tensor,
    pub reward_margins: Tensor,
    pub accuracy: Tensor,
}

#[derive(Debug)]
pub struct DPODataCollator {
    config: DPOConfig,
}

impl DPODataCollator {
    pub fn new(config: DPOConfig) -> Self {
        Self { config }
    }

    pub fn collate_batch(&self, examples: Vec<DPOExample>) -> Result<DPOBatch> {
        let batch_size = examples.len();

        if batch_size == 0 {
            return Err(anyhow!("Empty batch"));
        }

        // Determine maximum sequence length
        let max_len = self.config.max_length.unwrap_or_else(|| {
            examples
                .iter()
                .map(|ex| ex.chosen_input_ids.len().max(ex.rejected_input_ids.len()))
                .max()
                .unwrap_or(512)
        });

        // Pad and collate sequences
        let mut chosen_input_ids = Vec::with_capacity(batch_size * max_len);
        let mut chosen_labels = Vec::with_capacity(batch_size * max_len);
        let mut chosen_attention_mask = Vec::with_capacity(batch_size * max_len);
        let mut rejected_input_ids = Vec::with_capacity(batch_size * max_len);
        let mut rejected_labels = Vec::with_capacity(batch_size * max_len);
        let mut rejected_attention_mask = Vec::with_capacity(batch_size * max_len);

        for example in examples {
            // Pad chosen sequence
            let chosen_len = example.chosen_input_ids.len().min(max_len);
            chosen_input_ids.extend_from_slice(&example.chosen_input_ids[..chosen_len]);
            chosen_input_ids.resize(chosen_input_ids.len() + (max_len - chosen_len), 0);

            chosen_labels.extend_from_slice(&example.chosen_labels[..chosen_len]);
            chosen_labels.resize(
                chosen_labels.len() + (max_len - chosen_len),
                self.config.label_pad_token_id,
            );

            let mut mask = vec![1; chosen_len];
            mask.resize(max_len, 0);
            chosen_attention_mask.extend(mask);

            // Pad rejected sequence
            let rejected_len = example.rejected_input_ids.len().min(max_len);
            rejected_input_ids.extend_from_slice(&example.rejected_input_ids[..rejected_len]);
            rejected_input_ids.resize(rejected_input_ids.len() + (max_len - rejected_len), 0);

            rejected_labels.extend_from_slice(&example.rejected_labels[..rejected_len]);
            rejected_labels.resize(
                rejected_labels.len() + (max_len - rejected_len),
                self.config.label_pad_token_id,
            );

            let mut mask = vec![1; rejected_len];
            mask.resize(max_len, 0);
            rejected_attention_mask.extend(mask);
        }

        Ok(DPOBatch {
            chosen_input_ids: Tensor::from_vec(
                chosen_input_ids.into_iter().map(|x| x as f32).collect(),
                &[batch_size, max_len],
            )?,
            chosen_labels: Tensor::from_vec(
                chosen_labels.into_iter().map(|x| x as f32).collect(),
                &[batch_size, max_len],
            )?,
            chosen_attention_mask: Tensor::from_vec(
                chosen_attention_mask.into_iter().map(|x| x as f32).collect(),
                &[batch_size, max_len],
            )?,
            rejected_input_ids: Tensor::from_vec(
                rejected_input_ids.into_iter().map(|x| x as f32).collect(),
                &[batch_size, max_len],
            )?,
            rejected_labels: Tensor::from_vec(
                rejected_labels.into_iter().map(|x| x as f32).collect(),
                &[batch_size, max_len],
            )?,
            rejected_attention_mask: Tensor::from_vec(
                rejected_attention_mask.into_iter().map(|x| x as f32).collect(),
                &[batch_size, max_len],
            )?,
        })
    }
}

#[derive(Debug, Clone)]
pub struct DPOExample {
    pub chosen_input_ids: Vec<i32>,
    pub chosen_labels: Vec<i32>,
    pub rejected_input_ids: Vec<i32>,
    pub rejected_labels: Vec<i32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::traits::Config;
    use trustformers_core::TrustformersError;

    /// Minimal identity model so `DPOTrainer` can be constructed in tests. The DPO maths
    /// under test operates on tensors that are handed to the trainer directly, so the model
    /// only needs to satisfy the `Model` bound.
    #[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
    struct EchoConfig;

    impl Config for EchoConfig {
        fn architecture(&self) -> &'static str {
            "echo"
        }
    }

    #[derive(Debug, Clone)]
    struct EchoModel {
        config: EchoConfig,
    }

    impl Model for EchoModel {
        type Config = EchoConfig;
        type Input = Tensor;
        type Output = Tensor;
        fn forward(
            &self,
            input: Self::Input,
        ) -> std::result::Result<Self::Output, TrustformersError> {
            Ok(input)
        }
        fn load_pretrained(
            &mut self,
            _r: &mut dyn std::io::Read,
        ) -> std::result::Result<(), TrustformersError> {
            Ok(())
        }
        fn get_config(&self) -> &Self::Config {
            &self.config
        }
        fn num_parameters(&self) -> usize {
            0
        }
    }

    fn trainer_with(config: DPOConfig) -> DPOTrainer<EchoModel> {
        DPOTrainer::new(EchoModel { config: EchoConfig }, None, config)
    }

    // ── get_batch_logps ───────────────────────────────────────────────────────

    #[test]
    fn test_get_batch_logps_hand_computed_single_position() {
        // logits = [[[1, 2, 3], [10, 0, 0]]], labels = [[1, -100]].
        // Position 0 is scored at label 1; position 1 is padding and must be dropped.
        // Expected: log_softmax([1,2,3])[1] = 2 - ln(e^1 + e^2 + e^3).
        let trainer = trainer_with(DPOConfig::default());
        let logits = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 10.0, 0.0, 0.0], &[1, 2, 3])
            .expect("tensor creation failed");
        let labels =
            Tensor::from_vec(vec![1.0f32, -100.0], &[1, 2]).expect("tensor creation failed");

        let logps = trainer
            .get_batch_logps(&logits, &labels, false)
            .expect("get_batch_logps failed");
        assert_eq!(logps.shape(), &[1]);

        let denom = 1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp();
        let expected = 2.0 - denom.ln();
        let got = logps.get_scalar(&[0]).expect("get_scalar failed");
        assert!(
            (got - expected).abs() < 1e-5,
            "expected {expected}, got {got}"
        );
    }

    #[test]
    fn test_get_batch_logps_varies_with_labels() {
        // Regression: the old implementation returned `log_probs.mean()` — the *same*
        // scalar for every row, entirely ignoring `labels`. Two rows with identical logits
        // but different labels must produce different log-probabilities.
        let trainer = trainer_with(DPOConfig::default());
        let row = [0.0f32, 1.0, 5.0];
        let mut data = Vec::new();
        data.extend_from_slice(&row); // batch 0, position 0
        data.extend_from_slice(&row); // batch 1, position 0
        let logits = Tensor::from_vec(data, &[2, 1, 3]).expect("tensor creation failed");
        let labels = Tensor::from_vec(vec![0.0f32, 2.0], &[2, 1]).expect("tensor creation failed");

        let logps = trainer
            .get_batch_logps(&logits, &labels, false)
            .expect("get_batch_logps failed");
        let a = logps.get_scalar(&[0]).expect("get_scalar failed");
        let b = logps.get_scalar(&[1]).expect("get_scalar failed");
        assert!(
            (a - b).abs() > 1.0,
            "different labels must give different log-probs, got {a} and {b}"
        );
        // Label 2 has the largest logit, so it must have the larger log-probability.
        assert!(b > a, "label 2 (logit 5) should beat label 0 (logit 0)");
    }

    #[test]
    fn test_get_batch_logps_sum_vs_average() {
        // Two unmasked positions, one padded: average must equal sum / 2.
        let trainer = trainer_with(DPOConfig::default());
        let logits = Tensor::from_vec(
            vec![
                1.0f32, 2.0, 3.0, // t = 0
                4.0, 0.0, 1.0, // t = 1
                0.0, 0.0, 0.0, // t = 2 (padded)
            ],
            &[1, 3, 3],
        )
        .expect("tensor creation failed");
        let labels =
            Tensor::from_vec(vec![2.0f32, 0.0, -100.0], &[1, 3]).expect("tensor creation failed");

        let summed = trainer
            .get_batch_logps(&logits, &labels, false)
            .expect("sum failed")
            .get_scalar(&[0])
            .expect("get_scalar failed");
        let averaged = trainer
            .get_batch_logps(&logits, &labels, true)
            .expect("average failed")
            .get_scalar(&[0])
            .expect("get_scalar failed");

        let d0 = 1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp();
        let d1 = 4.0f32.exp() + 0.0f32.exp() + 1.0f32.exp();
        let expected_sum = (3.0 - d0.ln()) + (4.0 - d1.ln());
        assert!(
            (summed - expected_sum).abs() < 1e-5,
            "expected {expected_sum}, got {summed}"
        );
        assert!(
            (averaged - expected_sum / 2.0).abs() < 1e-5,
            "average must be sum/2 over the 2 unmasked positions, got {averaged}"
        );
    }

    #[test]
    fn test_get_batch_logps_all_padding_is_zero() {
        let trainer = trainer_with(DPOConfig::default());
        let logits =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[1, 1, 3]).expect("tensor creation failed");
        let labels = Tensor::from_vec(vec![-100.0f32], &[1, 1]).expect("tensor creation failed");
        for average in [false, true] {
            let v = trainer
                .get_batch_logps(&logits, &labels, average)
                .expect("get_batch_logps failed")
                .get_scalar(&[0])
                .expect("get_scalar failed");
            assert_eq!(v, 0.0, "fully-masked sequence must contribute 0.0");
        }
    }

    #[test]
    fn test_get_batch_logps_rejects_out_of_range_label() {
        let trainer = trainer_with(DPOConfig::default());
        let logits =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[1, 1, 3]).expect("tensor creation failed");
        let labels = Tensor::from_vec(vec![7.0f32], &[1, 1]).expect("tensor creation failed");
        assert!(
            trainer.get_batch_logps(&logits, &labels, false).is_err(),
            "a label outside the vocabulary must be an error, not a silent value"
        );
    }

    #[test]
    fn test_get_batch_logps_rejects_rank_mismatch() {
        let trainer = trainer_with(DPOConfig::default());
        let logits = Tensor::from_vec(vec![1.0f32, 2.0], &[1, 2]).expect("tensor creation failed");
        let labels = Tensor::from_vec(vec![0.0f32], &[1, 1]).expect("tensor creation failed");
        assert!(trainer.get_batch_logps(&logits, &labels, false).is_err());
    }

    // ── KTO wiring ────────────────────────────────────────────────────────────

    #[test]
    fn test_kto_loss_differs_from_sigmoid_dpo() {
        // Regression: `DPOLossType::Kto` used to be byte-identical to the sigmoid branch.
        // With a non-zero KL reference point and asymmetric λ weights the prospect-theoretic
        // objective cannot coincide with -log σ(β Δ).
        let policy_chosen = Tensor::new(vec![1.5f32, 0.4, 2.2]).expect("tensor creation failed");
        let policy_rejected = Tensor::new(vec![0.2f32, -0.9, 0.1]).expect("tensor creation failed");
        let ref_chosen = Tensor::new(vec![0.1f32, 0.0, -0.3]).expect("tensor creation failed");
        let ref_rejected = Tensor::new(vec![-0.4f32, 0.3, 0.6]).expect("tensor creation failed");

        let sigmoid_trainer = trainer_with(DPOConfig {
            beta: 0.5,
            loss_type: DPOLossType::Sigmoid,
            ..DPOConfig::default()
        });
        let kto_trainer = trainer_with(DPOConfig {
            beta: 0.5,
            loss_type: DPOLossType::Kto,
            kto_lambda_preferred: 1.5,
            kto_lambda_rejected: 0.5,
            ..DPOConfig::default()
        });

        let sigmoid_loss: f32 = sigmoid_trainer
            .compute_loss(&policy_chosen, &policy_rejected, &ref_chosen, &ref_rejected)
            .expect("sigmoid loss failed")
            .item()
            .expect("item failed");
        let kto_loss: f32 = kto_trainer
            .compute_loss(&policy_chosen, &policy_rejected, &ref_chosen, &ref_rejected)
            .expect("kto loss failed")
            .item()
            .expect("item failed");

        assert!(
            (sigmoid_loss - kto_loss).abs() > 1e-3,
            "KTO must not equal sigmoid DPO: sigmoid={sigmoid_loss}, kto={kto_loss}"
        );
        assert!(kto_loss.is_finite());
    }

    #[test]
    fn test_kto_loss_matches_reference_implementation() {
        // The DPO wrapper must be a pure unfolding of the paired batch into KTO examples:
        // computing the same batch directly through `crate::kto` must give the same number.
        let policy_chosen = Tensor::new(vec![0.8f32, -0.2]).expect("tensor creation failed");
        let policy_rejected = Tensor::new(vec![-0.5f32, 0.9]).expect("tensor creation failed");
        let ref_chosen = Tensor::new(vec![0.1f32, 0.4]).expect("tensor creation failed");
        let ref_rejected = Tensor::new(vec![0.3f32, -0.1]).expect("tensor creation failed");

        let trainer = trainer_with(DPOConfig {
            beta: 0.25,
            loss_type: DPOLossType::Kto,
            kto_lambda_preferred: 1.2,
            kto_lambda_rejected: 0.7,
            ..DPOConfig::default()
        });
        let wrapped: f32 = trainer
            .compute_loss(&policy_chosen, &policy_rejected, &ref_chosen, &ref_rejected)
            .expect("kto loss failed")
            .item()
            .expect("item failed");

        let examples = vec![
            KtoExample {
                policy_log_prob: 0.8,
                reference_log_prob: 0.1,
                is_preferred: true,
            },
            KtoExample {
                policy_log_prob: -0.5,
                reference_log_prob: 0.3,
                is_preferred: false,
            },
            KtoExample {
                policy_log_prob: -0.2,
                reference_log_prob: 0.4,
                is_preferred: true,
            },
            KtoExample {
                policy_log_prob: 0.9,
                reference_log_prob: -0.1,
                is_preferred: false,
            },
        ];
        let direct = compute_kto_loss(
            &examples,
            &KtoConfig {
                beta: 0.25,
                lambda_preferred: 1.2,
                lambda_rejected: 0.7,
                ..KtoConfig::default()
            },
        )
        .expect("direct kto failed");

        assert!(
            (wrapped - direct.total_loss).abs() < 1e-6,
            "wrapper {wrapped} must match crate::kto {}",
            direct.total_loss
        );
    }

    #[test]
    fn test_kto_loss_responds_to_lambda_weights() {
        let policy_chosen = Tensor::new(vec![1.0f32, 0.5]).expect("tensor creation failed");
        let policy_rejected = Tensor::new(vec![-1.0f32, 0.0]).expect("tensor creation failed");
        let ref_chosen = Tensor::new(vec![0.0f32, 0.0]).expect("tensor creation failed");
        let ref_rejected = Tensor::new(vec![0.0f32, 0.0]).expect("tensor creation failed");

        let loss_for = |lw: f32, ll: f32| -> f32 {
            trainer_with(DPOConfig {
                beta: 0.4,
                loss_type: DPOLossType::Kto,
                kto_lambda_preferred: lw,
                kto_lambda_rejected: ll,
                ..DPOConfig::default()
            })
            .compute_loss(&policy_chosen, &policy_rejected, &ref_chosen, &ref_rejected)
            .expect("kto loss failed")
            .item()
            .expect("item failed")
        };

        let balanced = loss_for(1.0, 1.0);
        let preferred_heavy = loss_for(3.0, 1.0);
        assert!(
            preferred_heavy > balanced,
            "raising λ_w must raise the loss: {preferred_heavy} vs {balanced}"
        );
    }

    #[test]
    fn test_dpo_config_default() {
        let config = DPOConfig::default();
        assert_eq!(config.beta, 0.1);
        assert_eq!(config.label_smoothing, 0.0);
        assert!(matches!(config.loss_type, DPOLossType::Sigmoid));
    }

    #[test]
    fn test_sigmoid_dpo_loss_known_values() {
        // When logits = 0, sigmoid(0) = 0.5, -log(0.5) = ln(2) ≈ 0.6931
        let config = DPOConfig {
            beta: 1.0,
            loss_type: DPOLossType::Sigmoid,
            ..DPOConfig::default()
        };

        // policy_chosen = [1.0], policy_rejected = [1.0] => pi_logratios = 0
        // ref_chosen = [0.0], ref_rejected = [0.0] => ref_logratios = 0
        // logits = beta * (0 - 0) = 0
        // loss = -log(sigmoid(0)) = -log(0.5) = ln(2) ≈ 0.6931
        let chosen = Tensor::new(vec![1.0f32]).expect("tensor creation failed");
        let rejected = Tensor::new(vec![1.0f32]).expect("tensor creation failed");
        let ref_chosen = Tensor::new(vec![0.0f32]).expect("tensor creation failed");
        let ref_rejected = Tensor::new(vec![0.0f32]).expect("tensor creation failed");

        // Create a dummy model - we only need compute_loss, so use a minimal struct
        // Instead, call compute_loss directly by constructing the trainer struct fields
        // We need a Model impl. Let's test the math directly instead.
        // sigmoid(0) = 0.5, log(0.5) = -0.6931, neg => 0.6931
        let pi_logratios = chosen.sub(&rejected).expect("sub failed");
        let ref_logratios = ref_chosen.sub(&ref_rejected).expect("sub failed");
        let logits = pi_logratios
            .sub(&ref_logratios)
            .expect("sub failed")
            .mul_scalar(config.beta)
            .expect("mul failed");
        let loss = logits
            .sigmoid()
            .expect("sigmoid failed")
            .log()
            .expect("log failed")
            .neg()
            .expect("neg failed");
        let loss_val: f32 = loss.item().expect("item failed");
        let expected = 2.0f32.ln(); // ln(2) ≈ 0.6931
        assert!(
            (loss_val - expected).abs() < 1e-4,
            "Expected {expected}, got {loss_val}"
        );
    }

    #[test]
    fn test_sigmoid_dpo_loss_positive_logits() {
        // When chosen is clearly preferred: logits = 2.0
        // sigmoid(2.0) ≈ 0.8808, -log(0.8808) ≈ 0.1269
        let logits = Tensor::new(vec![2.0f32]).expect("tensor creation failed");
        let loss = logits
            .sigmoid()
            .expect("sigmoid failed")
            .log()
            .expect("log failed")
            .neg()
            .expect("neg failed");
        let loss_val: f32 = loss.item().expect("item failed");
        let expected = -(1.0f32 / (1.0 + (-2.0f32).exp())).ln();
        assert!(
            (loss_val - expected).abs() < 1e-4,
            "Expected {expected}, got {loss_val}"
        );
    }

    #[test]
    fn test_sigmoid_dpo_loss_negative_logits() {
        // When rejected is preferred: logits = -2.0
        // sigmoid(-2.0) ≈ 0.1192, -log(0.1192) ≈ 2.1269
        let logits = Tensor::new(vec![-2.0f32]).expect("tensor creation failed");
        let loss = logits
            .sigmoid()
            .expect("sigmoid failed")
            .log()
            .expect("log failed")
            .neg()
            .expect("neg failed");
        let loss_val: f32 = loss.item().expect("item failed");
        let expected = -(1.0f32 / (1.0 + 2.0f32.exp())).ln();
        assert!(
            (loss_val - expected).abs() < 1e-4,
            "Expected {expected}, got {loss_val}"
        );
    }

    #[test]
    fn test_preference_accuracy_all_correct() {
        // When all logits > 0, accuracy should be 1.0
        let chosen = Tensor::new(vec![2.0f32, 3.0, 4.0]).expect("tensor creation failed");
        let rejected = Tensor::new(vec![1.0f32, 1.0, 1.0]).expect("tensor creation failed");
        let ref_logps = Tensor::new(vec![0.0f32, 0.0, 0.0]).expect("tensor creation failed");

        let pi_logratios = chosen.sub(&rejected).expect("sub failed");
        let ref_logratios = ref_logps.sub(&ref_logps).expect("sub failed");
        let logits = pi_logratios
            .sub(&ref_logratios)
            .expect("sub failed")
            .mul_scalar(1.0)
            .expect("mul failed");

        let zeros = Tensor::zeros(&logits.shape()).expect("zeros failed");
        let preferred = logits.greater(&zeros).expect("greater failed");
        let accuracy: f32 = preferred.mean().expect("mean failed").item().expect("item failed");
        assert!(
            (accuracy - 1.0).abs() < 1e-6,
            "Expected 1.0, got {accuracy}"
        );
    }

    #[test]
    fn test_preference_accuracy_half_correct() {
        // When half logits > 0, accuracy should be 0.5
        let chosen = Tensor::new(vec![2.0f32, 0.5, -1.0, -2.0]).expect("tensor creation failed");
        let rejected = Tensor::new(vec![0.0f32, 0.0, 0.0, 0.0]).expect("tensor creation failed");
        let ref_logps = Tensor::new(vec![0.0f32, 0.0, 0.0, 0.0]).expect("tensor creation failed");

        let pi_logratios = chosen.sub(&rejected).expect("sub failed");
        let ref_logratios = ref_logps.sub(&ref_logps).expect("sub failed");
        let logits = pi_logratios
            .sub(&ref_logratios)
            .expect("sub failed")
            .mul_scalar(1.0)
            .expect("mul failed");

        let zeros = Tensor::zeros(&logits.shape()).expect("zeros failed");
        let preferred = logits.greater(&zeros).expect("greater failed");
        let accuracy: f32 = preferred.mean().expect("mean failed").item().expect("item failed");
        assert!(
            (accuracy - 0.5).abs() < 1e-6,
            "Expected 0.5, got {accuracy}"
        );
    }

    #[test]
    fn test_log_softmax_produces_log_probabilities() {
        // log_softmax output should sum to < 0 (all values negative) and
        // exp(log_softmax) should sum to 1.0
        let logits =
            Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[1, 3]).expect("tensor creation failed");
        let log_probs = logits.log_softmax(-1).expect("log_softmax failed");

        // All log probs should be <= 0
        let max_val = log_probs.get_scalar(&[0, 2]).expect("get_scalar failed");
        assert!(max_val <= 0.0, "Log prob should be <= 0, got {max_val}");

        // exp(log_softmax) should sum to ~1.0
        // log_softmax([1,2,3]) = [x - log(e^1 + e^2 + e^3)] for each x
        let denom = 1.0f32.exp() + 2.0f32.exp() + 3.0f32.exp();
        let expected_0 = 1.0 - denom.ln();
        let expected_1 = 2.0 - denom.ln();
        let expected_2 = 3.0 - denom.ln();

        let v0 = log_probs.get_scalar(&[0, 0]).expect("get_scalar failed");
        let v1 = log_probs.get_scalar(&[0, 1]).expect("get_scalar failed");
        let v2 = log_probs.get_scalar(&[0, 2]).expect("get_scalar failed");

        assert!(
            (v0 - expected_0).abs() < 1e-4,
            "Expected {expected_0}, got {v0}"
        );
        assert!(
            (v1 - expected_1).abs() < 1e-4,
            "Expected {expected_1}, got {v1}"
        );
        assert!(
            (v2 - expected_2).abs() < 1e-4,
            "Expected {expected_2}, got {v2}"
        );

        // exp values should sum to 1.0
        let sum = v0.exp() + v1.exp() + v2.exp();
        assert!(
            (sum - 1.0).abs() < 1e-4,
            "exp(log_softmax) should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_hinge_loss_known_values() {
        // Hinge loss: max(0, 1 - logits)
        // logits = 2.0 => max(0, 1-2) = 0
        // logits = 0.5 => max(0, 1-0.5) = 0.5
        // logits = -1.0 => max(0, 1-(-1)) = 2.0
        let logits = Tensor::new(vec![2.0f32, 0.5, -1.0]).expect("tensor creation failed");
        let ones = Tensor::ones(&logits.shape()).expect("ones failed");
        let hinge = ones.sub(&logits).expect("sub failed").relu().expect("relu failed");
        let loss: f32 = hinge.mean().expect("mean failed").item().expect("item failed");
        let expected = (0.0 + 0.5 + 2.0) / 3.0;
        assert!(
            (loss - expected).abs() < 1e-4,
            "Expected {expected}, got {loss}"
        );
    }

    #[test]
    fn test_ipo_loss_known_values() {
        // IPO loss: (logits - 0.5)^2
        // logits = 0.5 => (0.5 - 0.5)^2 = 0
        // logits = 1.5 => (1.5 - 0.5)^2 = 1.0
        // logits = -0.5 => (-0.5 - 0.5)^2 = 1.0
        let logits = Tensor::new(vec![0.5f32, 1.5, -0.5]).expect("tensor creation failed");
        let half = logits.sub_scalar(0.5).expect("sub_scalar failed");
        let loss_tensor = half.pow(2.0).expect("pow failed");
        let loss: f32 = loss_tensor.mean().expect("mean failed").item().expect("item failed");
        let expected = (0.0 + 1.0 + 1.0) / 3.0;
        assert!(
            (loss - expected).abs() < 1e-4,
            "Expected {expected}, got {loss}"
        );
    }

    #[test]
    fn test_dpo_data_collator() {
        let config = DPOConfig {
            max_length: Some(3), // Set to match test expectations
            ..DPOConfig::default()
        };
        let collator = DPODataCollator::new(config);

        let examples = vec![
            DPOExample {
                chosen_input_ids: vec![1, 2, 3],
                chosen_labels: vec![1, 2, 3],
                rejected_input_ids: vec![1, 2, 4],
                rejected_labels: vec![1, 2, 4],
            },
            DPOExample {
                chosen_input_ids: vec![1, 5],
                chosen_labels: vec![1, 5],
                rejected_input_ids: vec![1, 6],
                rejected_labels: vec![1, 6],
            },
        ];

        let batch = collator.collate_batch(examples).expect("operation failed in test");
        assert_eq!(batch.chosen_input_ids.shape(), &[2, 3]);
        assert_eq!(batch.rejected_input_ids.shape(), &[2, 3]);
    }
}
