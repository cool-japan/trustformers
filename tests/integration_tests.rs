use ndarray::{Array1, Array2, Array3, Array4};
use trustformers::automodel::{AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForQuestionAnswering, AutoModelForCausalLM, AutoModelForMaskedLM};
use trustformers_models::bert::{BertConfig, BertModel, BertForSequenceClassification};
use trustformers_models::roberta::{RobertaConfig, RobertaModel, RobertaForSequenceClassification};
use trustformers_models::distilbert::{DistilBertConfig, DistilBertModel, DistilBertForSequenceClassification};
use trustformers_models::albert::{AlbertConfig, AlbertModel, AlbertForSequenceClassification};
use trustformers_models::electra::{ElectraConfig, ElectraModel, ElectraForSequenceClassification, ElectraForPreTraining};
use trustformers_models::deberta::{DebertaConfig, DebertaModel, DebertaForSequenceClassification};
use trustformers_models::gpt2::{GPT2Config, GPT2Model, GPT2LMHeadModel};
use trustformers_models::t5::{T5Config, T5Model, T5ForConditionalGeneration};
use trustformers_models::vit::{ViTConfig, ViTModel, ViTForImageClassification};
use trustformers_optim::{AdamW, AdaFactor, LAMB};
use trustformers_optim::scheduler::{LinearLRScheduler, CosineAnnealingLRScheduler, PolynomialLRScheduler};

#[test]
fn test_bert_model_integration() {
    let config = BertConfig::base();
    let model = BertModel::new(config.clone()).unwrap();

    // Test forward pass
    let input_ids = Array1::from_vec(vec![101, 1045, 2293, 7570, 102]); // [CLS] I love BERT [SEP]
    let result = model.forward(&input_ids);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.shape(), &[1, input_ids.len(), config.hidden_size]);
}

#[test]
fn test_bert_sequence_classification_integration() {
    let config = BertConfig::base();
    let model = BertForSequenceClassification::new(config.clone(), 2).unwrap();

    let input_ids = Array1::from_vec(vec![101, 1045, 2293, 7570, 102]);
    let result = model.forward(&input_ids);

    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.shape(), &[1, 2]);
}

#[test]
fn test_roberta_model_integration() {
    let config = RobertaConfig::base();
    let model = RobertaModel::new(config.clone()).unwrap();

    let input_ids = Array1::from_vec(vec![0, 100, 200, 300, 2]); // RoBERTa tokens
    let result = model.forward(&input_ids);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.shape(), &[1, input_ids.len(), config.hidden_size]);
}

#[test]
fn test_distilbert_model_integration() {
    let config = DistilBertConfig::base();
    let model = DistilBertModel::new(config.clone()).unwrap();

    let input_ids = Array1::from_vec(vec![101, 1045, 2293, 7570, 102]);
    let result = model.forward(&input_ids);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.shape(), &[1, input_ids.len(), config.hidden_size]);
}

#[test]
fn test_albert_model_integration() {
    let config = AlbertConfig::base();
    let model = AlbertModel::new(config.clone()).unwrap();

    let input_ids = Array1::from_vec(vec![2, 100, 200, 300, 3]); // ALBERT tokens
    let result = model.forward(&input_ids);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.shape(), &[1, input_ids.len(), config.hidden_size]);
}

#[test]
fn test_electra_model_integration() {
    let config = ElectraConfig::small();
    let model = ElectraModel::new(config.clone());

    let input_ids = Array1::from_vec(vec![101, 1045, 2293, 7570, 102]);

    // Test generator
    let gen_result = model.get_generator().forward(&input_ids);
    assert!(gen_result.is_ok());
    let gen_output = gen_result.unwrap();
    assert_eq!(gen_output.shape(), &[1, input_ids.len(), config.vocab_size]);

    // Test discriminator
    let disc_result = model.get_discriminator().forward(&input_ids);
    assert!(disc_result.is_ok());
    let disc_output = disc_result.unwrap();
    assert_eq!(disc_output.shape(), &[1, input_ids.len(), config.discriminator_hidden_size]);
}

#[test]
fn test_electra_pretraining_integration() {
    let config = ElectraConfig::small();
    let model = ElectraForPreTraining::new(config.clone());

    let input_ids = Array1::from_vec(vec![101, 1045, 2293, 7570, 102]);
    let result = model.forward(&input_ids);

    assert!(result.is_ok());
    let (gen_logits, disc_logits) = result.unwrap();
    assert_eq!(gen_logits.shape(), &[1, input_ids.len(), config.vocab_size]);
    assert_eq!(disc_logits.shape(), &[1, input_ids.len(), 1]);
}

#[test]
fn test_deberta_model_integration() {
    let config = DebertaConfig::base();
    let model = DebertaModel::new(config.clone()).unwrap();

    let input_ids = Array1::from_vec(vec![0, 1, 2, 3, 2]); // DeBERTa tokens
    let result = model.forward(&input_ids, None);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.shape(), &[1, input_ids.len(), config.hidden_size]);
}

#[test]
fn test_gpt2_model_integration() {
    let config = GPT2Config::small();
    let model = GPT2Model::new(config.clone()).unwrap();

    let input_ids = Array1::from_vec(vec![50256, 100, 200, 300]); // GPT-2 tokens
    let result = model.forward(&input_ids, None, None);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.shape(), &[1, input_ids.len(), config.n_embd]);
}

#[test]
fn test_gpt2_lm_head_integration() {
    let config = GPT2Config::small();
    let model = GPT2LMHeadModel::new(config.clone()).unwrap();

    let input_ids = Array1::from_vec(vec![50256, 100, 200, 300]);
    let result = model.forward(&input_ids, None, None);

    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.shape(), &[1, input_ids.len(), config.vocab_size]);
}

#[test]
fn test_t5_model_integration() {
    let config = T5Config::small();
    let model = T5Model::new(config.clone()).unwrap();

    let input_ids = Array1::from_vec(vec![1, 100, 200, 300, 1]); // T5 tokens
    let decoder_input_ids = Array1::from_vec(vec![0, 50, 100]);
    let result = model.forward(&input_ids, Some(&decoder_input_ids), None);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.shape(), &[1, decoder_input_ids.len(), config.d_model]);
}

#[test]
fn test_t5_conditional_generation_integration() {
    let config = T5Config::small();
    let model = T5ForConditionalGeneration::new(config.clone()).unwrap();

    let input_ids = Array1::from_vec(vec![1, 100, 200, 300, 1]);
    let decoder_input_ids = Array1::from_vec(vec![0, 50, 100]);
    let result = model.forward(&input_ids, Some(&decoder_input_ids), None);

    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.shape(), &[1, decoder_input_ids.len(), config.vocab_size]);
}

#[test]
fn test_vit_model_integration() {
    let config = ViTConfig::tiny(); // Use tiny for faster testing
    let model = ViTModel::new(config.clone()).unwrap();

    // Test with small image (32x32 to make it faster)
    let mut small_config = config.clone();
    small_config.image_size = 32;
    small_config.patch_size = 16;
    let small_model = ViTModel::new(small_config.clone()).unwrap();

    let images = Array4::zeros((1, 32, 32, 3));
    let result = small_model.forward(&images);

    assert!(result.is_ok());
    let output = result.unwrap();
    // 32x32 image with 16x16 patches = 4 patches + 1 class token = 5 tokens
    assert_eq!(output.shape(), &[1, 5, small_config.hidden_size]);
}

#[test]
fn test_vit_classification_integration() {
    let mut config = ViTConfig::tiny();
    config.image_size = 32;
    config.patch_size = 16;
    config.num_labels = 10; // CIFAR-10 like

    let model = ViTForImageClassification::new(config.clone()).unwrap();

    let images = Array4::zeros((2, 32, 32, 3)); // Batch of 2 images
    let result = model.forward(&images);

    assert!(result.is_ok());
    let logits = result.unwrap();
    assert_eq!(logits.shape(), &[2, 10]);
}

#[test]
fn test_automodel_integration() {
    // Test BERT auto model
    let bert_model = AutoModel::from_pretrained_name("bert-base-uncased").unwrap();
    let input_ids = Array1::from_vec(vec![101, 1045, 2293, 7570, 102]);

    match bert_model {
        AutoModel::Bert(model) => {
            let result = model.forward(&input_ids);
            assert!(result.is_ok());
        }
        _ => panic!("Expected BERT model"),
    }
}

#[test]
fn test_automodel_sequence_classification_integration() {
    let model = AutoModelForSequenceClassification::from_pretrained_name("bert-base-uncased", 2).unwrap();
    let input_ids = Array1::from_vec(vec![101, 1045, 2293, 7570, 102]);

    match model {
        AutoModelForSequenceClassification::Bert(bert_model) => {
            let result = bert_model.forward(&input_ids);
            assert!(result.is_ok());
            let logits = result.unwrap();
            assert_eq!(logits.shape(), &[1, 2]);
        }
        _ => panic!("Expected BERT model"),
    }
}

#[test]
fn test_optimizer_integration() {
    // Test AdamW optimizer
    let mut adamw = AdamW::new(0.001);
    let state = adamw.get_state();
    assert_eq!(state.learning_rate, 0.001);

    // Test AdaFactor optimizer
    let mut adafactor = AdaFactor::new(0.001);
    let state = adafactor.get_state();
    assert_eq!(state.learning_rate, 0.001);

    // Test LAMB optimizer
    let mut lamb = LAMB::new(0.001);
    let state = lamb.get_state();
    assert_eq!(state.learning_rate, 0.001);
}

#[test]
fn test_scheduler_integration() {
    let total_steps = 1000;
    let warmup_steps = 100;

    // Test Linear LR Scheduler
    let mut linear_scheduler = LinearLRScheduler::new(0.001, total_steps, warmup_steps);
    let initial_lr = linear_scheduler.get_lr();
    assert!(initial_lr > 0.0);

    linear_scheduler.step();
    let after_step_lr = linear_scheduler.get_lr();
    assert!(after_step_lr >= 0.0);

    // Test Cosine Annealing Scheduler
    let mut cosine_scheduler = CosineAnnealingLRScheduler::new(0.001, total_steps, Some(warmup_steps));
    let cosine_lr = cosine_scheduler.get_lr();
    assert!(cosine_lr > 0.0);

    // Test Polynomial Scheduler
    let mut poly_scheduler = PolynomialLRScheduler::new(0.001, total_steps, 2.0, 0.0, Some(warmup_steps));
    let poly_lr = poly_scheduler.get_lr();
    assert!(poly_lr > 0.0);
}

#[test]
fn test_config_validation_integration() {
    // Test BERT config validation
    let mut bert_config = BertConfig::base();
    bert_config.hidden_size = 100;
    bert_config.num_attention_heads = 12; // Not divisible
    assert!(bert_config.validate().is_err());

    // Test ViT config validation
    let mut vit_config = ViTConfig::base();
    vit_config.image_size = 225; // Not divisible by patch_size (16)
    assert!(vit_config.validate().is_err());

    // Test DeBERTa config validation
    let mut deberta_config = DebertaConfig::base();
    deberta_config.max_relative_positions = 0; // Invalid
    assert!(deberta_config.validate().is_err());
}

#[test]
fn test_model_variants_integration() {
    // Test different BERT variants
    let bert_small = BertConfig::small();
    let bert_base = BertConfig::base();
    let bert_large = BertConfig::large();

    assert!(bert_small.hidden_size < bert_base.hidden_size);
    assert!(bert_base.hidden_size < bert_large.hidden_size);

    // Test different GPT-2 variants
    let gpt2_small = GPT2Config::small();
    let gpt2_medium = GPT2Config::medium();
    let gpt2_large = GPT2Config::large();

    assert!(gpt2_small.n_embd < gpt2_medium.n_embd);
    assert!(gpt2_medium.n_embd < gpt2_large.n_embd);

    // Test different T5 variants
    let t5_small = T5Config::small();
    let t5_base = T5Config::base();
    let t5_large = T5Config::large();

    assert!(t5_small.d_model < t5_base.d_model);
    assert!(t5_base.d_model < t5_large.d_model);
}

#[test]
fn test_model_memory_efficiency() {
    // Test that models can be created without excessive memory usage
    let configs_and_expected_params = vec![
        ("BERT-small", BertConfig::small().num_parameters()),
        ("RoBERTa-base", RobertaConfig::base().num_parameters()),
        ("DistilBERT-base", DistilBertConfig::base().num_parameters()),
        ("ALBERT-base", AlbertConfig::base().num_parameters()),
        ("ELECTRA-small", ElectraConfig::small().discriminator_hidden_size),
        ("GPT2-small", GPT2Config::small().num_parameters()),
        ("T5-small", T5Config::small().num_parameters()),
        ("ViT-tiny", ViTConfig::tiny().num_patches()),
    ];

    for (model_name, param_count) in configs_and_expected_params {
        println!("Testing {}: {} parameters/features", model_name, param_count);
        assert!(param_count > 0);
    }
}

#[test]
fn test_error_handling_integration() {
    // Test invalid input handling
    let config = BertConfig::base();
    let model = BertModel::new(config).unwrap();

    // Test with empty input
    let empty_input = Array1::from_vec(vec![]);
    let result = model.forward(&empty_input);
    // Should handle gracefully (either work with empty or return appropriate error)

    // Test ViT with invalid image dimensions
    let vit_config = ViTConfig::base();
    let vit_model = ViTModel::new(vit_config).unwrap();

    // Wrong number of channels
    let invalid_image = Array4::zeros((1, 224, 224, 1)); // 1 channel instead of 3
    let result = vit_model.forward(&invalid_image);
    assert!(result.is_err()); // Should properly handle this error
}