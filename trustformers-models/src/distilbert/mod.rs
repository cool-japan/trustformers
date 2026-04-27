pub mod config;
pub mod model;
pub mod tasks;

#[cfg(test)]
mod tests;

pub use config::DistilBertConfig;
pub use model::DistilBertModel;
pub use tasks::{
    DistilBertForMaskedLM, DistilBertForQuestionAnswering, DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
};
