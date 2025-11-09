use crate::gpt2::{Gpt2Config, Gpt2LMHeadModel, Gpt2Model};
use trustformers_core::{
    tensor::Tensor,
    traits::{Model, TokenizedInput},
};

#[test]
fn test_gpt2_model_creation() {
    let config = Gpt2Config::default();
    let model = Gpt2Model::new(config.clone()).unwrap();
    assert_eq!(model.get_config().n_layer, 12);
    assert_eq!(model.get_config().n_head, 12);
}

#[test]
fn test_gpt2_lm_head_model_creation() {
    let config = Gpt2Config::default();
    let model = Gpt2LMHeadModel::new(config.clone()).unwrap();
    assert_eq!(model.get_config().n_layer, 12);
}

#[test]
fn test_gpt2_forward_pass() {
    let config = Gpt2Config {
        vocab_size: 100,
        n_positions: 64,
        n_embd: 32,
        n_layer: 2,
        n_head: 4,
        ..Default::default()
    };

    let model = Gpt2Model::new(config).unwrap();
    let input = TokenizedInput {
        input_ids: vec![1, 2, 3, 4, 5],
        attention_mask: vec![1u8; 5],
        token_type_ids: None,
        offset_mapping: None,
        special_tokens_mask: None,
        overflowing_tokens: None,
    };

    let output = model.forward(input).unwrap();
    match &output.last_hidden_state {
        Tensor::F32(arr) => {
            assert_eq!(arr.shape(), &[1, 5, 32]);
        },
        _ => panic!("Expected F32 tensor"),
    }
}

#[test]
fn test_gpt2_lm_forward_pass() {
    let config = Gpt2Config {
        vocab_size: 100,
        n_positions: 64,
        n_embd: 32,
        n_layer: 2,
        n_head: 4,
        ..Default::default()
    };

    let model = Gpt2LMHeadModel::new(config).unwrap();
    let input = TokenizedInput {
        input_ids: vec![1, 2, 3, 4, 5],
        attention_mask: vec![1u8; 5],
        token_type_ids: None,
        offset_mapping: None,
        special_tokens_mask: None,
        overflowing_tokens: None,
    };

    let output = model.forward(input).unwrap();
    match &output.logits {
        Tensor::F32(arr) => {
            assert_eq!(arr.shape(), &[1, 5, 100]);
        },
        _ => panic!("Expected F32 tensor"),
    }
}

#[test]
fn test_gpt2_generate_greedy() {
    let config = Gpt2Config {
        vocab_size: 100,
        n_positions: 64,
        n_embd: 32,
        n_layer: 2,
        n_head: 4,
        ..Default::default()
    };

    let model = Gpt2LMHeadModel::new(config).unwrap();
    let input_ids = vec![1, 2, 3];
    let generated = model.generate_greedy(input_ids.clone(), 10).unwrap();

    assert!(generated.len() >= input_ids.len());
    assert!(generated.len() <= 10);
    assert_eq!(&generated[..3], &input_ids[..]);
}

#[test]
fn test_gpt2_beam_search() {
    let config = Gpt2Config {
        vocab_size: 50,
        n_positions: 64,
        n_embd: 32,
        n_layer: 1,
        n_head: 4,
        ..Default::default()
    };

    let model = Gpt2LMHeadModel::new(config).unwrap();
    let input_ids = vec![1, 2];
    let generated = model.generate_beam_search(input_ids.clone(), 10, 3).unwrap();

    assert!(generated.len() >= input_ids.len());
    assert!(generated.len() <= 10);
    assert_eq!(&generated[..2], &input_ids[..]);
}
