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
                // DPO loss: -log(sigmoid(beta * (log pi(y_w|x) - log pi(y_l|x) - log ref(y_w|x) + log ref(y_l|x))))
                let neg_logits = logits.mul_scalar(-1.0)?;
                // Simplified DPO loss using softmax (log operations not yet implemented)
                let probs = neg_logits.softmax(-1)?;
                let loss = probs.neg()?;
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
                // KTO loss: simplified version
                // Simplified loss using sigmoid (log operations not yet implemented)
                let loss = logits.sigmoid()?.neg()?;
                Ok(loss.mean()?)
            },
        }
    }

    pub fn get_batch_logps(
        &self,
        logits: &Tensor,
        labels: &Tensor,
        _average_log_prob: bool,
    ) -> Result<Tensor> {
        // Convert logits to log probabilities
        // Use softmax instead of log_softmax for now
        let log_probs = logits.softmax(-1)?;

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

        // Compute accuracy (how often chosen is preferred) - simplified implementation
        let accuracy = reward_margins.mean()?; // Simplified for now

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
    fn test_dpo_data_collator() {
        let mut config = DPOConfig::default();
        config.max_length = Some(3); // Set to match test expectations
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

        let batch = collator.collate_batch(examples).unwrap();
        assert_eq!(batch.chosen_input_ids.shape(), &[2, 3]);
        assert_eq!(batch.rejected_input_ids.shape(), &[2, 3]);
    }
}
