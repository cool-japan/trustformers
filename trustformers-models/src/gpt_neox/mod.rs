pub mod config;
pub mod model;

#[cfg(test)]
mod tests;

pub use config::GPTNeoXConfig;
pub use model::{GPTNeoXForCausalLM, GPTNeoXModel};
