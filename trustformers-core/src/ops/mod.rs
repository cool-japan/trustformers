pub mod activations;

pub use activations::{gelu, gelu_new, relu, sigmoid, silu, swiglu, tanh};

#[cfg(test)]
mod activations_tests;
