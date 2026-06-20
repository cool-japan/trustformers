//! Strongly-typed activation function dispatch.
//!
//! Many architectures historically selected their activation function on every
//! forward pass by re-parsing a Hugging-Face style configuration string, e.g.
//!
//! ```ignore
//! match self.activation.as_str() {
//!     "gelu" => x.gelu(),
//!     "relu" => x.relu(),
//!     "silu" | "swish" => x.silu(),
//!     _ => Ok(x.clone()),
//! }
//! ```
//!
//! This module replaces that stringly-typed dispatch with the [`ActivationType`]
//! enum which is parsed exactly once at construction time and then applied via
//! [`ActivationType::apply`] on the hot path. The numerical behaviour is kept
//! identical to the previous string matches (the underlying [`Tensor`] activation
//! methods are called directly).
//!
//! Models may keep their existing `String` configuration field for
//! serialization / back-compat and simply derive the enum from it once.

use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;

/// Activation functions supported by the model implementations.
///
/// The string mapping in [`ActivationType::try_from`] mirrors the exact
/// Hugging-Face style identifiers that the model configurations use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationType {
    /// Gaussian Error Linear Unit (tanh approximation), HF `"gelu"`.
    Gelu,
    /// GELU "new" / tanh approximation, HF `"gelu_new"`.
    ///
    /// Numerically identical to [`ActivationType::Gelu`] in this crate (both use
    /// the tanh approximation), matching the previous behaviour where models
    /// dispatched `"gelu"` and `"gelu_new"` to the same implementation.
    GeluNew,
    /// GELU PyTorch tanh variant, HF `"gelu_pytorch_tanh"`.
    GeluPytorchTanh,
    /// Rectified Linear Unit, HF `"relu"`.
    Relu,
    /// Sigmoid Linear Unit (a.k.a. Swish), HF `"silu"` / `"swish"`.
    Silu,
    /// Hyperbolic tangent, HF `"tanh"`.
    Tanh,
    /// Logistic sigmoid, HF `"sigmoid"`.
    Sigmoid,
    /// Identity / pass-through.
    ///
    /// This is the explicit fallback that preserves the historical
    /// `_ => Ok(x.clone())` default arm used by several models.
    Identity,
}

impl ActivationType {
    /// Apply the activation element-wise to `t`.
    ///
    /// Behaviour is kept identical to the previous string-based dispatch by
    /// delegating to the underlying [`Tensor`] activation methods.
    pub fn apply(&self, t: &Tensor) -> Result<Tensor> {
        match self {
            ActivationType::Gelu | ActivationType::GeluNew | ActivationType::GeluPytorchTanh => {
                t.gelu()
            },
            ActivationType::Relu => t.relu(),
            ActivationType::Silu => t.silu(),
            ActivationType::Tanh => t.tanh(),
            ActivationType::Sigmoid => t.sigmoid(),
            ActivationType::Identity => Ok(t.clone()),
        }
    }

    /// Parse a Hugging-Face style activation string, falling back to the
    /// supplied `default` variant for any unrecognised identifier.
    ///
    /// This preserves the historical behaviour of models whose `match` had a
    /// non-error default arm (for example `_ => Ok(x.clone())` maps to
    /// [`ActivationType::Identity`], and `_ => x.gelu()` maps to
    /// [`ActivationType::Gelu`]).
    pub fn from_config_str_or(name: &str, default: ActivationType) -> ActivationType {
        ActivationType::try_from(name).unwrap_or(default)
    }
}

impl TryFrom<&str> for ActivationType {
    type Error = TrustformersError;

    fn try_from(value: &str) -> Result<Self> {
        match value {
            "gelu" => Ok(ActivationType::Gelu),
            "gelu_new" => Ok(ActivationType::GeluNew),
            "gelu_pytorch_tanh" => Ok(ActivationType::GeluPytorchTanh),
            "relu" => Ok(ActivationType::Relu),
            "silu" | "swish" => Ok(ActivationType::Silu),
            "tanh" => Ok(ActivationType::Tanh),
            "sigmoid" => Ok(ActivationType::Sigmoid),
            "identity" | "none" | "linear" => Ok(ActivationType::Identity),
            other => Err(TrustformersError::model_error(format!(
                "Unsupported activation function: {}",
                other
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::ArrayD;

    fn sample_tensor() -> Tensor {
        // A small tensor with positive, negative and zero values.
        let data = ArrayD::from_shape_vec(
            scirs2_core::ndarray::IxDyn(&[2, 3]),
            vec![-2.0_f32, -0.5, 0.0, 0.5, 2.0, 4.0],
        )
        .expect("valid shape");
        Tensor::F32(data)
    }

    #[test]
    fn try_from_maps_every_supported_string() {
        assert_eq!(
            ActivationType::try_from("gelu").unwrap(),
            ActivationType::Gelu
        );
        assert_eq!(
            ActivationType::try_from("gelu_new").unwrap(),
            ActivationType::GeluNew
        );
        assert_eq!(
            ActivationType::try_from("gelu_pytorch_tanh").unwrap(),
            ActivationType::GeluPytorchTanh
        );
        assert_eq!(
            ActivationType::try_from("relu").unwrap(),
            ActivationType::Relu
        );
        assert_eq!(
            ActivationType::try_from("silu").unwrap(),
            ActivationType::Silu
        );
        assert_eq!(
            ActivationType::try_from("swish").unwrap(),
            ActivationType::Silu
        );
        assert_eq!(
            ActivationType::try_from("tanh").unwrap(),
            ActivationType::Tanh
        );
        assert_eq!(
            ActivationType::try_from("sigmoid").unwrap(),
            ActivationType::Sigmoid
        );
        assert_eq!(
            ActivationType::try_from("identity").unwrap(),
            ActivationType::Identity
        );
        assert_eq!(
            ActivationType::try_from("none").unwrap(),
            ActivationType::Identity
        );
        assert_eq!(
            ActivationType::try_from("linear").unwrap(),
            ActivationType::Identity
        );
    }

    #[test]
    fn try_from_rejects_unknown_string() {
        assert!(ActivationType::try_from("definitely_not_an_activation").is_err());
    }

    #[test]
    fn from_config_str_or_uses_default_for_unknown() {
        assert_eq!(
            ActivationType::from_config_str_or("nope", ActivationType::Identity),
            ActivationType::Identity
        );
        assert_eq!(
            ActivationType::from_config_str_or("nope", ActivationType::Gelu),
            ActivationType::Gelu
        );
        // Known strings still map correctly even when a default is supplied.
        assert_eq!(
            ActivationType::from_config_str_or("relu", ActivationType::Gelu),
            ActivationType::Relu
        );
    }

    #[test]
    fn apply_produces_finite_shape_correct_output_for_every_variant() {
        let input = sample_tensor();
        let variants = [
            ActivationType::Gelu,
            ActivationType::GeluNew,
            ActivationType::GeluPytorchTanh,
            ActivationType::Relu,
            ActivationType::Silu,
            ActivationType::Tanh,
            ActivationType::Sigmoid,
            ActivationType::Identity,
        ];

        for variant in variants {
            let out = variant.apply(&input).expect("activation must succeed");
            assert_eq!(
                out.shape(),
                input.shape(),
                "{:?} must preserve shape",
                variant
            );
            match out {
                Tensor::F32(arr) => {
                    for value in arr.iter() {
                        assert!(
                            value.is_finite(),
                            "{:?} produced a non-finite value: {}",
                            variant,
                            value
                        );
                    }
                },
                other => panic!("{:?} produced unexpected tensor type: {:?}", variant, other),
            }
        }
    }

    #[test]
    fn apply_matches_tensor_methods() {
        let input = sample_tensor();

        let relu_expected = input.relu().unwrap();
        let relu_actual = ActivationType::Relu.apply(&input).unwrap();
        match (relu_expected, relu_actual) {
            (Tensor::F32(a), Tensor::F32(b)) => assert_eq!(a, b),
            _ => panic!("unexpected tensor types"),
        }

        let silu_expected = input.silu().unwrap();
        let silu_actual = ActivationType::Silu.apply(&input).unwrap();
        match (silu_expected, silu_actual) {
            (Tensor::F32(a), Tensor::F32(b)) => assert_eq!(a, b),
            _ => panic!("unexpected tensor types"),
        }

        let identity_actual = ActivationType::Identity.apply(&input).unwrap();
        match (input, identity_actual) {
            (Tensor::F32(a), Tensor::F32(b)) => assert_eq!(a, b),
            _ => panic!("unexpected tensor types"),
        }
    }
}
