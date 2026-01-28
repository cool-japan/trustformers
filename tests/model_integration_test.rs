use scirs2_core::ndarray::{Array1, Array2, Array3, Array4}; // SciRS2 Integration Policy

#[test]
fn test_bert_base_functionality() {
    use trustformers_models::bert::{BertConfig, BertModel};

    let config = BertConfig::base();
    let model = BertModel::new(config.clone()).unwrap();

    // Test forward pass with simple input
    let input_ids = Array1::from_vec(vec![101, 1045, 2293, 7570, 102]); // [CLS] I love BERT [SEP]
    let result = model.forward(&input_ids);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.shape(), &[1, input_ids.len(), config.hidden_size]);
}

#[test]
fn test_gpt2_base_functionality() {
    use trustformers_models::gpt2::{GPT2Config, GPT2Model};

    let config = GPT2Config::small();
    let model = GPT2Model::new(config.clone()).unwrap();

    let input_ids = Array1::from_vec(vec![50256, 100, 200, 300]); // GPT-2 tokens
    let result = model.forward(&input_ids, None, None);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.shape(), &[1, input_ids.len(), config.n_embd]);
}

#[test]
fn test_t5_base_functionality() {
    use trustformers_models::t5::{T5Config, T5Model};

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
fn test_vit_base_functionality() {
    use trustformers_models::vit::{ViTConfig, ViTModel};

    let mut config = ViTConfig::tiny();
    config.image_size = 32;  // Smaller for faster testing
    config.patch_size = 16;

    let model = ViTModel::new(config.clone()).unwrap();

    let images = Array4::zeros((1, 32, 32, 3));
    let result = model.forward(&images);

    assert!(result.is_ok());
    let output = result.unwrap();
    // 32x32 image with 16x16 patches = 4 patches + 1 class token = 5 tokens
    assert_eq!(output.shape(), &[1, 5, config.hidden_size]);
}

#[test]
fn test_electra_base_functionality() {
    use trustformers_models::electra::{ElectraConfig, ElectraForPreTraining};

    let config = ElectraConfig::small();
    let model = ElectraForPreTraining::new(config.clone()).unwrap();

    let input_ids = Array1::from_vec(vec![101, 1045, 2293, 7570, 102]);
    let result = model.forward(&input_ids);

    assert!(result.is_ok());
    let (gen_logits, disc_logits) = result.unwrap();
    assert_eq!(gen_logits.shape(), &[1, input_ids.len(), config.vocab_size]);
    assert_eq!(disc_logits.shape(), &[1, input_ids.len(), 1]);
}

#[test]
fn test_deberta_base_functionality() {
    use trustformers_models::deberta::{DebertaConfig, DebertaModel};

    let config = DebertaConfig::base();
    let model = DebertaModel::new(config.clone()).unwrap();

    let input_ids = Array1::from_vec(vec![0, 1, 2, 3, 2]);
    let result = model.forward(&input_ids, None);

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.shape(), &[1, input_ids.len(), config.hidden_size]);
}

#[test]
fn test_optimizer_functionality() {
    use trustformers_optim::{AdamW, AdaFactor, LAMB};

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
fn test_scheduler_functionality() {
    use trustformers_optim::scheduler::{LinearLRScheduler, CosineAnnealingLRScheduler, PolynomialLRScheduler};

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
fn test_config_validation() {
    use trustformers_models::bert::BertConfig;
    use trustformers_models::vit::ViTConfig;
    use trustformers_models::deberta::DebertaConfig;

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
fn test_model_variants() {
    use trustformers_models::bert::BertConfig;
    use trustformers_models::gpt2::GPT2Config;
    use trustformers_models::t5::T5Config;

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

#[cfg(feature = "llama")]
#[test]
fn test_llama_base_functionality() {
    use trustformers_models::llama::{LlamaConfig, LlamaModel, LlamaForCausalLM};

    let config = LlamaConfig {
        num_hidden_layers: 2, // Smaller for testing
        vocab_size: 1000,
        hidden_size: 64,
        num_attention_heads: 8,
        intermediate_size: 256,
        ..LlamaConfig::default()
    };

    let model = LlamaModel::new(config.clone()).unwrap();

    // Test forward pass with simple input
    let input_ids = vec![1, 100, 200, 300, 2]; // LLaMA tokens
    let result = model.forward(input_ids.clone());

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.shape(), &[1, input_ids.len(), config.hidden_size]);

    // Test LlamaForCausalLM
    let causal_model = LlamaForCausalLM::new(config.clone()).unwrap();
    let logits = causal_model.forward(input_ids.clone()).unwrap();
    assert_eq!(logits.shape(), &[1, input_ids.len(), config.vocab_size]);
}

#[cfg(feature = "llama")]
#[test]
fn test_llama_model_variants() {
    use trustformers_models::llama::LlamaConfig;

    // Test different LLaMA variants
    let llama_7b = LlamaConfig::llama_7b();
    let llama_13b = LlamaConfig::llama_13b();
    let llama_30b = LlamaConfig::llama_30b();
    let llama_65b = LlamaConfig::llama_65b();

    assert!(llama_7b.hidden_size < llama_13b.hidden_size);
    assert!(llama_13b.hidden_size < llama_30b.hidden_size);
    assert!(llama_30b.hidden_size < llama_65b.hidden_size);

    // Test LLaMA 2 variants
    let llama2_7b = LlamaConfig::llama2_7b();
    let llama2_13b = LlamaConfig::llama2_13b();
    let llama2_70b = LlamaConfig::llama2_70b();

    assert_eq!(llama2_7b.max_position_embeddings, 4096); // Extended context
    assert!(llama2_70b.num_key_value_heads.is_some()); // Grouped-query attention

    // Test Code Llama
    let code_llama = LlamaConfig::code_llama_7b();
    assert_eq!(code_llama.max_position_embeddings, 16384); // Long context for code
    assert_eq!(code_llama.vocab_size, 32016); // Different vocab
}

#[cfg(feature = "mistral")]
#[test]
fn test_mistral_base_functionality() {
    use trustformers_models::mistral::{MistralConfig, MistralModel, MistralForCausalLM};

    let config = MistralConfig {
        num_hidden_layers: 2, // Smaller for testing
        vocab_size: 1000,
        hidden_size: 64,
        num_attention_heads: 8,
        num_key_value_heads: 2, // Grouped-query attention
        intermediate_size: 256,
        ..MistralConfig::default()
    };

    let model = MistralModel::new(config.clone()).unwrap();

    // Test forward pass with simple input
    let input_ids = vec![1, 100, 200, 300, 2]; // Mistral tokens
    let result = model.forward(input_ids.clone());

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.shape(), &[1, input_ids.len(), config.hidden_size]);

    // Test MistralForCausalLM
    let causal_model = MistralForCausalLM::new(config.clone()).unwrap();
    let logits = causal_model.forward(input_ids.clone()).unwrap();
    assert_eq!(logits.shape(), &[1, input_ids.len(), config.vocab_size]);
}

#[cfg(feature = "mistral")]
#[test]
fn test_mistral_model_variants() {
    use trustformers_models::mistral::MistralConfig;

    // Test Mistral 7B
    let mistral_7b = MistralConfig::mistral_7b();
    assert_eq!(mistral_7b.num_attention_heads, 32);
    assert_eq!(mistral_7b.num_key_value_heads, 8); // Grouped-query attention
    assert!(mistral_7b.uses_sliding_window());
    assert_eq!(mistral_7b.sliding_window_size(), 4096);

    // Test Mistral 7B Instruct
    let mistral_instruct = MistralConfig::mistral_7b_instruct();
    assert_eq!(mistral_instruct.max_position_embeddings, 32768);

    // Test Mixtral 8x7B
    let mixtral = MistralConfig::mixtral_8x7b();
    assert!(!mixtral.uses_sliding_window()); // Mixtral doesn't use sliding window
    assert_eq!(mixtral.model_type, "mixtral");
}

#[cfg(feature = "mistral")]
#[test]
fn test_mixtral_sparse_moe() {
    use trustformers_models::mistral::{MistralConfig, MixtralSparseMoE, MixtralExpert};

    let config = MistralConfig {
        hidden_size: 64,
        intermediate_size: 256,
        ..MistralConfig::default()
    };

    // Test MixtralExpert creation
    let expert = MixtralExpert::new(&config);
    assert!(expert.is_ok());

    // Test MixtralSparseMoE creation
    let moe = MixtralSparseMoE::new(&config, 8, 2); // 8 experts, top-2
    assert!(moe.is_ok());
}

#[cfg(feature = "clip")]
#[test]
fn test_clip_base_functionality() {
    use trustformers_models::clip::{CLIPConfig, CLIPModel};
    use ndarray::Array4;

    let config = CLIPConfig::vit_b_32();
    let model = CLIPModel::new(config.clone()).unwrap();

    // Test text features
    let input_ids = vec![49406, 100, 200, 300, 49407]; // CLIP tokens
    let text_features = model.get_text_features(input_ids);
    assert!(text_features.is_ok());

    // Test image features
    let images = Array4::zeros((1, 224, 224, 3));
    let image_features = model.get_image_features(images);
    assert!(image_features.is_ok());
}

#[cfg(feature = "clip")]
#[test]
fn test_clip_model_variants() {
    use trustformers_models::clip::CLIPConfig;

    // Test different CLIP variants
    let vit_b_32 = CLIPConfig::vit_b_32();
    assert_eq!(vit_b_32.vision_config.patch_size, 32);
    assert_eq!(vit_b_32.projection_dim, 512);

    let vit_b_16 = CLIPConfig::vit_b_16();
    assert_eq!(vit_b_16.vision_config.patch_size, 16);

    let vit_l_14 = CLIPConfig::vit_l_14();
    assert_eq!(vit_l_14.vision_config.patch_size, 14);
    assert_eq!(vit_l_14.projection_dim, 768);
}

#[cfg(feature = "gemma")]
#[test]
fn test_gemma_base_functionality() {
    use trustformers_models::gemma::{GemmaConfig, GemmaModel, GemmaForCausalLM};

    let config = GemmaConfig {
        num_hidden_layers: 2, // Smaller for testing
        vocab_size: 1000,
        hidden_size: 64,
        num_attention_heads: 8,
        num_key_value_heads: 2, // Multi-query attention
        head_dim: 8,
        intermediate_size: 256,
        ..GemmaConfig::default()
    };

    let model = GemmaModel::new(config.clone()).unwrap();

    // Test forward pass with simple input
    let input_ids = vec![2, 100, 200, 300, 1]; // Gemma tokens
    let result = model.forward(input_ids.clone());

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.shape(), &[input_ids.len(), config.hidden_size]);

    // Test GemmaForCausalLM
    let causal_model = GemmaForCausalLM::new(config.clone()).unwrap();
    let logits = causal_model.forward(input_ids.clone()).unwrap();
    assert_eq!(logits.shape(), &[input_ids.len(), config.vocab_size]);
}

#[cfg(feature = "gemma")]
#[test]
fn test_gemma_model_variants() {
    use trustformers_models::gemma::GemmaConfig;

    // Test different Gemma variants
    let gemma_2b = GemmaConfig::gemma_2b();
    let gemma_7b = GemmaConfig::gemma_7b();

    assert!(gemma_2b.hidden_size < gemma_7b.hidden_size);
    assert_eq!(gemma_2b.num_key_value_heads, 1); // Multi-query for 2B
    assert_eq!(gemma_7b.num_key_value_heads, 16); // Full attention for 7B

    // Test code variants
    let gemma_code_2b = GemmaConfig::gemma_code_2b();
    assert_eq!(gemma_code_2b.model_type, "gemma-code");
}

#[cfg(feature = "qwen")]
#[test]
fn test_qwen_base_functionality() {
    use trustformers_models::qwen::{QwenConfig, QwenModel, QwenForCausalLM};

    let config = QwenConfig {
        num_hidden_layers: 2, // Smaller for testing
        vocab_size: 1000,
        hidden_size: 64,
        num_attention_heads: 8,
        num_key_value_heads: Some(2), // Grouped-query attention
        intermediate_size: 256,
        ..QwenConfig::default()
    };

    let model = QwenModel::new(config.clone()).unwrap();

    // Test forward pass with simple input
    let input_ids = vec![151643, 100, 200, 300, 151645]; // Qwen tokens
    let result = model.forward(input_ids.clone());

    assert!(result.is_ok());
    let output = result.unwrap();
    assert_eq!(output.shape(), &[input_ids.len(), config.hidden_size]);

    // Test QwenForCausalLM
    let causal_model = QwenForCausalLM::new(config.clone()).unwrap();
    let logits = causal_model.forward(input_ids.clone()).unwrap();
    assert_eq!(logits.shape(), &[input_ids.len(), config.vocab_size]);
}

#[cfg(feature = "qwen")]
#[test]
fn test_qwen_model_variants() {
    use trustformers_models::qwen::QwenConfig;

    // Test different Qwen variants
    let qwen2_0_5b = QwenConfig::qwen2_0_5b();
    let qwen2_7b = QwenConfig::qwen2_7b();
    let qwen2_5_7b = QwenConfig::qwen2_5_7b();

    assert!(qwen2_0_5b.hidden_size < qwen2_7b.hidden_size);
    assert_eq!(qwen2_7b.max_position_embeddings, 32768);
    assert_eq!(qwen2_5_7b.max_position_embeddings, 131072); // Extended context

    // Test grouped-query attention
    assert!(qwen2_7b.uses_grouped_query_attention());
    assert_eq!(qwen2_7b.num_query_groups(), 7); // 28 / 4

    // Test Qwen2.5 variant
    assert!(qwen2_5_7b.is_qwen2_5());

    // Test coder variant
    let qwen_coder = QwenConfig::qwen2_5_coder_7b();
    assert_eq!(qwen_coder.model_type, "qwen2.5-coder");
}