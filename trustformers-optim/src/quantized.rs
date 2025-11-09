use anyhow::Result;
use std::collections::HashMap;
use trustformers_core::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    pub scale: f32,
    pub zero_point: i8,
    pub min_val: f32,
    pub max_val: f32,
}

impl QuantizationConfig {
    pub fn new(min_val: f32, max_val: f32) -> Self {
        let scale = (max_val - min_val) / 255.0;
        let zero_point = (-min_val / scale).round().clamp(-128.0, 127.0) as i8;

        Self {
            scale,
            zero_point,
            min_val,
            max_val,
        }
    }

    pub fn quantize(&self, value: f32) -> i8 {
        let quantized = ((value - self.min_val) / self.scale).round() - 128.0;
        quantized.clamp(-128.0, 127.0) as i8
    }

    pub fn dequantize(&self, quantized: i8) -> f32 {
        (quantized as f32 + 128.0) * self.scale + self.min_val
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedState {
    pub data: Vec<i8>,
    pub config: QuantizationConfig,
}

impl QuantizedState {
    pub fn new(values: &[f32]) -> Self {
        let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let config = QuantizationConfig::new(min_val, max_val);
        let data: Vec<i8> = values.iter().map(|&v| config.quantize(v)).collect();

        Self { data, config }
    }

    pub fn to_f32(&self) -> Vec<f32> {
        self.data.iter().map(|&q| self.config.dequantize(q)).collect()
    }

    pub fn update(&mut self, new_values: &[f32]) {
        let min_val = new_values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = new_values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        self.config = QuantizationConfig::new(min_val, max_val);
        self.data = new_values.iter().map(|&v| self.config.quantize(v)).collect();
    }
}

#[derive(Debug)]
pub struct Adam8bit {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub step: usize,
    pub momentum_states: HashMap<String, QuantizedState>,
    pub variance_states: HashMap<String, QuantizedState>,
}

impl Default for Adam8bit {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            step: 0,
            momentum_states: HashMap::new(),
            variance_states: HashMap::new(),
        }
    }
}

impl Adam8bit {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            ..Default::default()
        }
    }

    pub fn with_config(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            step: 0,
            momentum_states: HashMap::new(),
            variance_states: HashMap::new(),
        }
    }

    pub fn step(
        &mut self,
        parameters: &mut HashMap<String, Tensor>,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<()> {
        self.step += 1;

        let bias_correction1 = 1.0 - self.beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step as i32);

        for (name, param) in parameters.iter_mut() {
            let grad = gradients
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("Missing gradient for parameter: {}", name))?;

            let mut param_data = param.data()?;
            let grad_data = grad.data()?;

            if param_data.len() != grad_data.len() {
                return Err(anyhow::anyhow!(
                    "Parameter and gradient size mismatch for: {}",
                    name
                ));
            }

            if !self.momentum_states.contains_key(name) {
                let zeros = vec![0.0; param_data.len()];
                self.momentum_states.insert(name.clone(), QuantizedState::new(&zeros));
                self.variance_states.insert(name.clone(), QuantizedState::new(&zeros));
            }

            let momentum_state = self.momentum_states.get_mut(name).unwrap();
            let variance_state = self.variance_states.get_mut(name).unwrap();

            let mut momentum = momentum_state.to_f32();
            let mut variance = variance_state.to_f32();

            for i in 0..param_data.len() {
                let mut grad_val = grad_data[i];

                if self.weight_decay > 0.0 {
                    grad_val += self.weight_decay * param_data[i];
                }

                momentum[i] = self.beta1 * momentum[i] + (1.0 - self.beta1) * grad_val;
                variance[i] = self.beta2 * variance[i] + (1.0 - self.beta2) * grad_val * grad_val;

                let corrected_momentum = momentum[i] / bias_correction1;
                let corrected_variance = variance[i] / bias_correction2;

                param_data[i] -= self.learning_rate * corrected_momentum
                    / (corrected_variance.sqrt() + self.epsilon);
            }

            momentum_state.update(&momentum);
            variance_state.update(&variance);

            // Update the parameter tensor with modified data
            *param = Tensor::new(param_data)?;
        }

        Ok(())
    }

    pub fn memory_usage(&self) -> usize {
        let mut total = 0;
        for state in self.momentum_states.values() {
            total += state.data.len();
        }
        for state in self.variance_states.values() {
            total += state.data.len();
        }
        total
    }

    pub fn memory_savings_vs_fp32(&self) -> f32 {
        let quantized_size = self.memory_usage();
        let fp32_equivalent = quantized_size * 4;
        1.0 - (quantized_size as f32 / fp32_equivalent as f32)
    }
}

#[derive(Debug)]
pub struct AdamW8bit {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub step: usize,
    pub momentum_states: HashMap<String, QuantizedState>,
    pub variance_states: HashMap<String, QuantizedState>,
}

impl Default for AdamW8bit {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 1e-2,
            step: 0,
            momentum_states: HashMap::new(),
            variance_states: HashMap::new(),
        }
    }
}

impl AdamW8bit {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            ..Default::default()
        }
    }

    pub fn with_config(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            step: 0,
            momentum_states: HashMap::new(),
            variance_states: HashMap::new(),
        }
    }

    pub fn step(
        &mut self,
        parameters: &mut HashMap<String, Tensor>,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<()> {
        self.step += 1;

        let bias_correction1 = 1.0 - self.beta1.powi(self.step as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step as i32);

        for (name, param) in parameters.iter_mut() {
            let grad = gradients
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("Missing gradient for parameter: {}", name))?;

            let mut param_data = param.data()?;
            let grad_data = grad.data()?;

            if param_data.len() != grad_data.len() {
                return Err(anyhow::anyhow!(
                    "Parameter and gradient size mismatch for: {}",
                    name
                ));
            }

            if !self.momentum_states.contains_key(name) {
                let zeros = vec![0.0; param_data.len()];
                self.momentum_states.insert(name.clone(), QuantizedState::new(&zeros));
                self.variance_states.insert(name.clone(), QuantizedState::new(&zeros));
            }

            let momentum_state = self.momentum_states.get_mut(name).unwrap();
            let variance_state = self.variance_states.get_mut(name).unwrap();

            let mut momentum = momentum_state.to_f32();
            let mut variance = variance_state.to_f32();

            for i in 0..param_data.len() {
                let grad_val = grad_data[i];

                momentum[i] = self.beta1 * momentum[i] + (1.0 - self.beta1) * grad_val;
                variance[i] = self.beta2 * variance[i] + (1.0 - self.beta2) * grad_val * grad_val;

                let corrected_momentum = momentum[i] / bias_correction1;
                let corrected_variance = variance[i] / bias_correction2;

                let update = corrected_momentum / (corrected_variance.sqrt() + self.epsilon);

                param_data[i] = param_data[i] * (1.0 - self.learning_rate * self.weight_decay)
                    - self.learning_rate * update;
            }

            momentum_state.update(&momentum);
            variance_state.update(&variance);

            // Update the parameter tensor with modified data
            *param = Tensor::new(param_data)?;
        }

        Ok(())
    }

    pub fn memory_usage(&self) -> usize {
        let mut total = 0;
        for state in self.momentum_states.values() {
            total += state.data.len();
        }
        for state in self.variance_states.values() {
            total += state.data.len();
        }
        total
    }

    pub fn memory_savings_vs_fp32(&self) -> f32 {
        let quantized_size = self.memory_usage();
        let fp32_equivalent = quantized_size * 4;
        1.0 - (quantized_size as f32 / fp32_equivalent as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_quantization_config() {
        let config = QuantizationConfig::new(-1.0, 1.0);

        // Test that values map correctly to the quantized range
        assert_eq!(config.quantize(-1.0), -128);
        assert_eq!(config.quantize(1.0), 127);

        // Test middle value
        let mid_quantized = config.quantize(0.0);
        assert!(mid_quantized >= -1 && mid_quantized <= 1);

        // Test round-trip accuracy
        let original = 0.5;
        let quantized = config.quantize(original);
        let reconstructed = config.dequantize(quantized);
        assert_abs_diff_eq!(original, reconstructed, epsilon = 0.02);
    }

    #[test]
    fn test_quantized_state() {
        let values = vec![0.1, -0.5, 0.8, -0.2];
        let state = QuantizedState::new(&values);

        let reconstructed = state.to_f32();

        // Test that we have the right number of values
        assert_eq!(values.len(), reconstructed.len());

        // Test that quantization preserves relative ordering
        assert!(reconstructed[2] > reconstructed[0]); // 0.8 > 0.1
        assert!(reconstructed[1] < reconstructed[0]); // -0.5 < 0.1

        // Test approximate reconstruction (quantization introduces some error)
        for (orig, recon) in values.iter().zip(reconstructed.iter()) {
            assert_abs_diff_eq!(orig, recon, epsilon = 0.1);
        }
    }

    #[test]
    fn test_adam8bit_creation() {
        let optimizer = Adam8bit::new(0.001);
        assert_eq!(optimizer.learning_rate, 0.001);
        assert_eq!(optimizer.beta1, 0.9);
        assert_eq!(optimizer.beta2, 0.999);
        assert_eq!(optimizer.step, 0);
    }

    #[test]
    fn test_adam8bit_memory_usage() {
        let mut optimizer = Adam8bit::new(0.001);

        let mut parameters = HashMap::new();
        let mut gradients = HashMap::new();

        let param_data = vec![1.0, 2.0, 3.0, 4.0];
        let grad_data = vec![0.1, 0.2, 0.3, 0.4];

        parameters.insert("layer1".to_string(), Tensor::new(param_data).unwrap());
        gradients.insert("layer1".to_string(), Tensor::new(grad_data).unwrap());

        optimizer.step(&mut parameters, &gradients).unwrap();

        assert_eq!(optimizer.memory_usage(), 8);
        assert!(optimizer.memory_savings_vs_fp32() > 0.7);
    }

    #[test]
    fn test_adamw8bit_creation() {
        let optimizer = AdamW8bit::new(0.001);
        assert_eq!(optimizer.learning_rate, 0.001);
        assert_eq!(optimizer.weight_decay, 0.01);
    }
}
