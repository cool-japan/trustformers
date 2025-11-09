//! Search space definitions for hyperparameter optimization

use anyhow::Result;
// Explicit import for .choose() method
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Value that a hyperparameter can take
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ParameterValue {
    /// Integer value
    Int(i64),
    /// Floating point value
    Float(f64),
    /// String/categorical value
    String(String),
    /// Boolean value
    Bool(bool),
}

impl ParameterValue {
    /// Get the value as an integer
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ParameterValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Get the value as a float
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ParameterValue::Float(v) => Some(*v),
            ParameterValue::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Get the value as a string
    pub fn as_string(&self) -> Option<&str> {
        match self {
            ParameterValue::String(v) => Some(v),
            _ => None,
        }
    }

    /// Get the value as a boolean
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ParameterValue::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

impl std::fmt::Display for ParameterValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParameterValue::Int(v) => write!(f, "{}", v),
            ParameterValue::Float(v) => write!(f, "{}", v),
            ParameterValue::String(v) => write!(f, "{}", v),
            ParameterValue::Bool(v) => write!(f, "{}", v),
        }
    }
}

/// Definition of a hyperparameter and its search space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HyperParameter {
    /// Categorical parameter with discrete choices
    Categorical(CategoricalParameter),
    /// Continuous parameter within a range
    Continuous(ContinuousParameter),
    /// Discrete integer parameter within a range
    Discrete(DiscreteParameter),
    /// Log-scale parameter (useful for learning rates, etc.)
    Log(LogParameter),
}

impl HyperParameter {
    /// Get the name of this parameter
    pub fn name(&self) -> &str {
        match self {
            HyperParameter::Categorical(p) => &p.name,
            HyperParameter::Continuous(p) => &p.name,
            HyperParameter::Discrete(p) => &p.name,
            HyperParameter::Log(p) => &p.name,
        }
    }

    /// Check if a value is valid for this parameter
    pub fn is_valid(&self, value: &ParameterValue) -> bool {
        match (self, value) {
            (HyperParameter::Categorical(p), ParameterValue::String(v)) => p.choices.contains(v),
            (HyperParameter::Continuous(p), ParameterValue::Float(v)) => {
                *v >= p.low && *v <= p.high
            },
            (HyperParameter::Discrete(p), ParameterValue::Int(v)) => {
                *v >= p.low && *v <= p.high && (*v - p.low) % p.step == 0
            },
            (HyperParameter::Log(p), ParameterValue::Float(v)) => *v >= p.low && *v <= p.high,
            _ => false,
        }
    }

    /// Sample a random value from this parameter's space
    pub fn sample<R: rand::Rng>(&self, rng: &mut R) -> Result<ParameterValue> {
        match self {
            HyperParameter::Categorical(p) => p.sample(rng),
            HyperParameter::Continuous(p) => p.sample(rng),
            HyperParameter::Discrete(p) => p.sample(rng),
            HyperParameter::Log(p) => p.sample(rng),
        }
    }
}

/// Categorical parameter with discrete string choices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalParameter {
    /// Name of the parameter
    pub name: String,
    /// Available choices
    pub choices: Vec<String>,
}

impl CategoricalParameter {
    /// Create a new categorical parameter
    pub fn new(name: impl Into<String>, choices: Vec<impl Into<String>>) -> Self {
        Self {
            name: name.into(),
            choices: choices.into_iter().map(|c| c.into()).collect(),
        }
    }

    /// Sample a random choice
    pub fn sample<R: rand::Rng>(&self, rng: &mut R) -> Result<ParameterValue> {
        if self.choices.is_empty() {
            return Err(anyhow::anyhow!("Cannot sample from empty choices list"));
        }
        #[allow(deprecated)]
        let idx = rng.gen_range(0..self.choices.len());
        let choice = self.choices[idx].clone();
        Ok(ParameterValue::String(choice))
    }
}

/// Continuous parameter within a floating-point range
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinuousParameter {
    /// Name of the parameter
    pub name: String,
    /// Lower bound (inclusive)
    pub low: f64,
    /// Upper bound (inclusive)
    pub high: f64,
}

impl ContinuousParameter {
    /// Create a new continuous parameter
    pub fn new(name: impl Into<String>, low: f64, high: f64) -> Self {
        assert!(low <= high, "Low bound must be <= high bound");
        Self {
            name: name.into(),
            low,
            high,
        }
    }

    /// Sample a random value
    pub fn sample<R: rand::Rng>(&self, rng: &mut R) -> Result<ParameterValue> {
        use rand_distr::{Distribution, Uniform};
        if self.low >= self.high {
            return Err(anyhow::anyhow!(
                "Invalid range: low ({}) must be less than high ({})",
                self.low,
                self.high
            ));
        }
        let dist = Uniform::new(self.low, self.high)
            .map_err(|e| anyhow::anyhow!("Failed to create uniform distribution: {}", e))?;
        Ok(ParameterValue::Float(dist.sample(rng)))
    }
}

/// Discrete integer parameter within a range with step size
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteParameter {
    /// Name of the parameter
    pub name: String,
    /// Lower bound (inclusive)
    pub low: i64,
    /// Upper bound (inclusive)
    pub high: i64,
    /// Step size
    pub step: i64,
}

impl DiscreteParameter {
    /// Create a new discrete parameter
    pub fn new(name: impl Into<String>, low: i64, high: i64, step: i64) -> Self {
        assert!(low <= high, "Low bound must be <= high bound");
        assert!(step > 0, "Step must be positive");
        Self {
            name: name.into(),
            low,
            high,
            step,
        }
    }

    /// Sample a random value
    pub fn sample<R: rand::Rng>(&self, rng: &mut R) -> Result<ParameterValue> {
        use rand_distr::{Distribution, Uniform};
        if self.low >= self.high {
            return Err(anyhow::anyhow!(
                "Invalid range: low ({}) must be less than high ({})",
                self.low,
                self.high
            ));
        }
        if self.step <= 0 {
            return Err(anyhow::anyhow!(
                "Invalid step: step ({}) must be positive",
                self.step
            ));
        }
        let num_steps = (self.high - self.low) / self.step + 1;
        if num_steps <= 0 {
            return Err(anyhow::anyhow!(
                "Invalid step configuration: no valid values in range"
            ));
        }
        let dist = Uniform::new(0, num_steps)
            .map_err(|e| anyhow::anyhow!("Failed to create uniform distribution: {}", e))?;
        let step_idx = dist.sample(rng);
        Ok(ParameterValue::Int(self.low + step_idx * self.step))
    }
}

/// Log-scale parameter (useful for learning rates, regularization, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogParameter {
    /// Name of the parameter
    pub name: String,
    /// Lower bound (inclusive, must be positive)
    pub low: f64,
    /// Upper bound (inclusive, must be positive)
    pub high: f64,
    /// Base for logarithm (default: 10)
    pub base: f64,
}

impl LogParameter {
    /// Create a new log-scale parameter
    pub fn new(name: impl Into<String>, low: f64, high: f64) -> Self {
        assert!(low > 0.0, "Low bound must be positive for log scale");
        assert!(high > 0.0, "High bound must be positive for log scale");
        assert!(low <= high, "Low bound must be <= high bound");
        Self {
            name: name.into(),
            low,
            high,
            base: 10.0,
        }
    }

    /// Create a new log-scale parameter with custom base
    pub fn with_base(name: impl Into<String>, low: f64, high: f64, base: f64) -> Self {
        assert!(low > 0.0, "Low bound must be positive for log scale");
        assert!(high > 0.0, "High bound must be positive for log scale");
        assert!(low <= high, "Low bound must be <= high bound");
        assert!(base > 0.0 && base != 1.0, "Base must be positive and != 1");
        Self {
            name: name.into(),
            low,
            high,
            base,
        }
    }

    /// Sample a random value from log space
    pub fn sample<R: rand::Rng>(&self, rng: &mut R) -> Result<ParameterValue> {
        use rand_distr::{Distribution, Uniform};
        if self.low <= 0.0 || self.high <= 0.0 {
            return Err(anyhow::anyhow!(
                "Both bounds must be positive for log scale: low={}, high={}",
                self.low,
                self.high
            ));
        }
        if self.low >= self.high {
            return Err(anyhow::anyhow!(
                "Invalid range: low ({}) must be less than high ({})",
                self.low,
                self.high
            ));
        }
        if self.base <= 0.0 || self.base == 1.0 {
            return Err(anyhow::anyhow!(
                "Base must be positive and not equal to 1: base={}",
                self.base
            ));
        }

        let log_low = self.low.log(self.base);
        let log_high = self.high.log(self.base);

        if !log_low.is_finite() || !log_high.is_finite() {
            return Err(anyhow::anyhow!(
                "Logarithmic values are not finite: log_low={}, log_high={}",
                log_low,
                log_high
            ));
        }

        let dist = Uniform::new(log_low, log_high)
            .map_err(|e| anyhow::anyhow!("Failed to create uniform distribution: {}", e))?;
        let log_value = dist.sample(rng);
        Ok(ParameterValue::Float(self.base.powf(log_value)))
    }
}

/// Complete search space definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSpace {
    /// Parameters in this search space
    pub parameters: Vec<HyperParameter>,
    /// Parameter lookup by name
    #[serde(skip)]
    name_index: HashMap<String, usize>,
}

impl SearchSpace {
    /// Create a new search space
    pub fn new(parameters: Vec<HyperParameter>) -> Self {
        let mut name_index = HashMap::new();
        for (i, param) in parameters.iter().enumerate() {
            name_index.insert(param.name().to_string(), i);
        }

        Self {
            parameters,
            name_index,
        }
    }

    /// Add a parameter to the search space
    pub fn add_parameter(mut self, parameter: HyperParameter) -> Self {
        let name = parameter.name().to_string();
        let index = self.parameters.len();
        self.parameters.push(parameter);
        self.name_index.insert(name, index);
        self
    }

    /// Get parameter by name
    pub fn get_parameter(&self, name: &str) -> Option<&HyperParameter> {
        self.name_index.get(name).and_then(|&idx| self.parameters.get(idx))
    }

    /// Get all parameter names
    pub fn parameter_names(&self) -> Vec<&str> {
        self.parameters.iter().map(|p| p.name()).collect()
    }

    /// Sample a complete configuration from this search space
    pub fn sample<R: rand::Rng>(&self, rng: &mut R) -> Result<HashMap<String, ParameterValue>> {
        let mut config = HashMap::new();
        for param in &self.parameters {
            let value = param.sample(rng).map_err(|e| {
                anyhow::anyhow!("Failed to sample parameter '{}': {}", param.name(), e)
            })?;
            config.insert(param.name().to_string(), value);
        }
        Ok(config)
    }

    /// Validate a configuration against this search space
    pub fn validate(&self, config: &HashMap<String, ParameterValue>) -> Result<(), String> {
        // Check that all required parameters are present
        for param in &self.parameters {
            let name = param.name();
            match config.get(name) {
                Some(value) => {
                    if !param.is_valid(value) {
                        return Err(format!("Invalid value for parameter '{}': {}", name, value));
                    }
                },
                None => return Err(format!("Missing required parameter: {}", name)),
            }
        }

        // Check for unexpected parameters
        for name in config.keys() {
            if !self.name_index.contains_key(name) {
                return Err(format!("Unknown parameter: {}", name));
            }
        }

        Ok(())
    }

    /// Get the total number of possible combinations (for discrete spaces)
    pub fn size(&self) -> Option<usize> {
        let mut total = 1usize;

        for param in &self.parameters {
            let param_size = match param {
                HyperParameter::Categorical(p) => Some(p.choices.len()),
                HyperParameter::Discrete(p) => Some(((p.high - p.low) / p.step + 1) as usize),
                HyperParameter::Continuous(_) | HyperParameter::Log(_) => return None,
            };

            if let Some(size) = param_size {
                total = total.checked_mul(size)?;
            }
        }

        Some(total)
    }
}

/// Builder for creating search spaces
pub struct SearchSpaceBuilder {
    parameters: Vec<HyperParameter>,
}

impl SearchSpaceBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            parameters: Vec::new(),
        }
    }

    /// Add a categorical parameter
    pub fn categorical(mut self, name: impl Into<String>, choices: Vec<impl Into<String>>) -> Self {
        self.parameters.push(HyperParameter::Categorical(CategoricalParameter::new(
            name, choices,
        )));
        self
    }

    /// Add a continuous parameter
    pub fn continuous(mut self, name: impl Into<String>, low: f64, high: f64) -> Self {
        self.parameters.push(HyperParameter::Continuous(ContinuousParameter::new(
            name, low, high,
        )));
        self
    }

    /// Add a discrete parameter
    pub fn discrete(mut self, name: impl Into<String>, low: i64, high: i64, step: i64) -> Self {
        self.parameters.push(HyperParameter::Discrete(DiscreteParameter::new(
            name, low, high, step,
        )));
        self
    }

    /// Add a log-scale parameter
    pub fn log_uniform(mut self, name: impl Into<String>, low: f64, high: f64) -> Self {
        self.parameters.push(HyperParameter::Log(LogParameter::new(name, low, high)));
        self
    }

    /// Build the search space
    pub fn build(self) -> SearchSpace {
        SearchSpace::new(self.parameters)
    }
}

impl Default for SearchSpaceBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use scirs2_core::StdRng;

    #[test]
    fn test_parameter_value() {
        let int_val = ParameterValue::Int(42);
        assert_eq!(int_val.as_int(), Some(42));
        assert_eq!(int_val.as_float(), Some(42.0));
        assert_eq!(int_val.as_string(), None);

        let float_val = ParameterValue::Float(3.14);
        assert_eq!(float_val.as_float(), Some(3.14));
        assert_eq!(float_val.as_int(), None);

        let str_val = ParameterValue::String("test".to_string());
        assert_eq!(str_val.as_string(), Some("test"));
        assert_eq!(str_val.as_int(), None);

        let bool_val = ParameterValue::Bool(true);
        assert_eq!(bool_val.as_bool(), Some(true));
        assert_eq!(bool_val.as_int(), None);
    }

    #[test]
    fn test_categorical_parameter() {
        let param = CategoricalParameter::new("optimizer", vec!["adam", "sgd", "adamw"]);
        assert_eq!(param.choices.len(), 3);

        let mut rng = StdRng::seed_from_u64(42);
        let value = param.sample(&mut rng).expect("Sampling should succeed");
        match value {
            ParameterValue::String(s) => assert!(param.choices.contains(&s)),
            _ => panic!("Expected string value"),
        }
    }

    #[test]
    fn test_continuous_parameter() {
        let param = ContinuousParameter::new("learning_rate", 1e-5, 1e-1);
        let mut rng = StdRng::seed_from_u64(42);
        let value = param.sample(&mut rng).expect("Sampling should succeed");

        match value {
            ParameterValue::Float(f) => {
                assert!(f >= 1e-5 && f <= 1e-1);
            },
            _ => panic!("Expected float value"),
        }
    }

    #[test]
    fn test_discrete_parameter() {
        let param = DiscreteParameter::new("batch_size", 8, 128, 8);
        let mut rng = StdRng::seed_from_u64(42);
        let value = param.sample(&mut rng).expect("Failed to sample discrete parameter");

        match value {
            ParameterValue::Int(i) => {
                assert!(i >= 8 && i <= 128);
                assert!((i - 8) % 8 == 0);
            },
            _ => panic!("Expected int value"),
        }
    }

    #[test]
    fn test_log_parameter() {
        let param = LogParameter::new("weight_decay", 1e-6, 1e-2);
        let mut rng = StdRng::seed_from_u64(42);
        let value = param.sample(&mut rng).expect("Failed to sample log parameter");

        match value {
            ParameterValue::Float(f) => {
                assert!(f >= 1e-6 && f <= 1e-2);
            },
            _ => panic!("Expected float value"),
        }
    }

    #[test]
    fn test_search_space() {
        let space = SearchSpaceBuilder::new()
            .categorical("optimizer", vec!["adam", "sgd"])
            .continuous("learning_rate", 1e-5, 1e-1)
            .discrete("batch_size", 8, 32, 8)
            .build();

        assert_eq!(space.parameters.len(), 3);
        assert_eq!(space.parameter_names().len(), 3);

        let mut rng = StdRng::seed_from_u64(42);
        let config = space.sample(&mut rng).expect("Failed to sample search space");
        assert_eq!(config.len(), 3);
        assert!(space.validate(&config).is_ok());

        // Test search space size calculation
        let size = space.size();
        assert_eq!(size, None); // None because of continuous parameter
    }

    #[test]
    fn test_search_space_validation() {
        let space = SearchSpaceBuilder::new()
            .categorical("optimizer", vec!["adam", "sgd"])
            .continuous("learning_rate", 1e-5, 1e-1)
            .build();

        let mut valid_config = HashMap::new();
        valid_config.insert(
            "optimizer".to_string(),
            ParameterValue::String("adam".to_string()),
        );
        valid_config.insert("learning_rate".to_string(), ParameterValue::Float(1e-4));

        assert!(space.validate(&valid_config).is_ok());

        // Test invalid optimizer
        let mut invalid_config = valid_config.clone();
        invalid_config.insert(
            "optimizer".to_string(),
            ParameterValue::String("invalid".to_string()),
        );
        assert!(space.validate(&invalid_config).is_err());

        // Test missing parameter
        let mut incomplete_config = HashMap::new();
        incomplete_config.insert(
            "optimizer".to_string(),
            ParameterValue::String("adam".to_string()),
        );
        assert!(space.validate(&incomplete_config).is_err());

        // Test unknown parameter
        let mut extra_config = valid_config.clone();
        extra_config.insert("unknown_param".to_string(), ParameterValue::Int(42));
        assert!(space.validate(&extra_config).is_err());
    }
}
