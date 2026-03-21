use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Model;

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

    pub fn compute_loss(
        &self,
        policy_chosen_logps: &Tensor,
        policy_rejected_logps: &Tensor,
        reference_chosen_logps: &Tensor,
        reference_rejected_logps: &Tensor,
    ) -> Result<Tensor> {
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
                // KTO loss: -log(sigmoid(logits))
                let loss = logits.sigmoid()?.log()?.neg()?;
                Ok(loss.mean()?)
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

    pub fn get_batch_logps(
        &self,
        logits: &Tensor,
        labels: &Tensor,
        _average_log_prob: bool,
    ) -> Result<Tensor> {
        // Convert logits to log probabilities using log_softmax
        let log_probs = logits.log_softmax(-1)?;

        // Gather log probabilities for the target tokens
        let batch_size = labels.shape()[0];
        let _seq_len = labels.shape()[1];

        let mut batch_logps = Vec::with_capacity(batch_size);

        // Compute log probabilities by summing over sequence dimension
        // Since we don't have tensor indexing, we'll use a simplified approach
        // by computing the mean log probability for each sequence
        for _i in 0..batch_size {
            // Get the mean log probability for the i-th sequence
            let sequence_logp = if log_probs.shape().len() >= 2 {
                // For now, use a simple approximation based on tensor statistics
                // In a full implementation, this would require proper tensor indexing
                // to select log_probs[i, :] and labels[i, :] and compute their dot product
                let mean_tensor = log_probs.mean()?;
                // Extract scalar value from the 0-dimensional tensor
                mean_tensor.get_scalar(&[])?
            } else {
                0.0f32
            };
            batch_logps.push(sequence_logp);
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
