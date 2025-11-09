use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use std::f32::consts::PI;

pub fn gelu(x: &Tensor) -> Result<Tensor> {
    match x {
        Tensor::F32(arr) => {
            let result = arr.mapv(|v| {
                0.5 * v * (1.0 + ((2.0 / PI).sqrt() * (v + 0.044715 * v.powi(3))).tanh())
            });
            Ok(Tensor::F32(result))
        },
        _ => Err(TrustformersError::tensor_op_error(
            "Unsupported tensor type for GELU",
            "gelu",
        )),
    }
}

pub fn gelu_new(x: &Tensor) -> Result<Tensor> {
    match x {
        Tensor::F32(arr) => {
            let result =
                arr.mapv(|v| 0.5 * v * (1.0 + (0.7978845608 * (v + 0.044715 * v.powi(3))).tanh()));
            Ok(Tensor::F32(result))
        },
        _ => Err(TrustformersError::tensor_op_error(
            "Unsupported tensor type for GELU new",
            "gelu_new",
        )),
    }
}

pub fn relu(x: &Tensor) -> Result<Tensor> {
    match x {
        Tensor::F32(arr) => {
            let result = arr.mapv(|v| v.max(0.0));
            Ok(Tensor::F32(result))
        },
        _ => Err(TrustformersError::tensor_op_error(
            "Unsupported tensor type for ReLU",
            "relu",
        )),
    }
}

pub fn sigmoid(x: &Tensor) -> Result<Tensor> {
    match x {
        Tensor::F32(arr) => {
            let result = arr.mapv(|v| 1.0 / (1.0 + (-v).exp()));
            Ok(Tensor::F32(result))
        },
        _ => Err(TrustformersError::tensor_op_error(
            "Unsupported tensor type for sigmoid",
            "sigmoid",
        )),
    }
}

pub fn tanh(x: &Tensor) -> Result<Tensor> {
    match x {
        Tensor::F32(arr) => {
            let result = arr.mapv(|v| v.tanh());
            Ok(Tensor::F32(result))
        },
        _ => Err(TrustformersError::tensor_op_error(
            "Unsupported tensor type for tanh",
            "tanh",
        )),
    }
}

/// SiLU (Swish) activation function
/// SiLU(x) = x * sigmoid(x)
pub fn silu(x: &Tensor) -> Result<Tensor> {
    match x {
        Tensor::F32(arr) => {
            let result = arr.mapv(|v| v / (1.0 + (-v).exp()));
            Ok(Tensor::F32(result))
        },
        _ => Err(TrustformersError::tensor_op_error(
            "Unsupported tensor type for SiLU",
            "silu",
        )),
    }
}

/// SwiGLU activation function
/// SwiGLU(x, gate) = SiLU(gate) * x
pub fn swiglu(x: &Tensor, gate: &Tensor) -> Result<Tensor> {
    let activated_gate = silu(gate)?;
    x.mul(&activated_gate)
}
