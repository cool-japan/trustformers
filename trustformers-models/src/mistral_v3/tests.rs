use super::config::MistralV3Config;
use super::model::{
    MistralV3Attention, MistralV3ForCausalLM, MistralV3MLP, MistralV3Model, MistralV3RmsNorm,
};
use super::tools::{
    MistralV3FunctionCaller, ToolCall, ToolDefinition, ToolParameter, ToolParseError,
};
use std::collections::HashMap;
use trustformers_core::traits::Config;
use trustformers_core::{tensor::Tensor, traits::Layer};

// ─────────────────────────────────────────────────────────────────────────────
// Config preset tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_mistral_v3_config_vocab_size() {
    let cfg = MistralV3Config::mistral_7b_v0_3();
    // v0.3 expands vocabulary to 32 768
    assert_eq!(cfg.vocab_size, 32768);
}

#[test]
fn test_mistral_v3_config_kv_heads() {
    let cfg = MistralV3Config::mistral_7b_v0_3();
    assert_eq!(cfg.num_key_value_heads, 8);
    assert_eq!(cfg.num_attention_heads, 32);
}

#[test]
fn test_mistral_v3_config_sliding_window() {
    let cfg = MistralV3Config::mistral_7b_v0_3();
    assert_eq!(cfg.sliding_window, 4096);
}

#[test]
fn test_mistral_v3_config_rope_theta() {
    let cfg = MistralV3Config::mistral_7b_v0_3();
    assert!((cfg.rope_theta - 1_000_000.0_f64).abs() < 1e-6);
}

#[test]
fn test_mistral_v3_config_validation_ok() {
    let cfg = MistralV3Config::small_test();
    assert!(cfg.validate().is_ok());
}

// ─────────────────────────────────────────────────────────────────────────────
// RMS norm
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_mistral_v3_rms_norm_output_shape() {
    let norm = MistralV3RmsNorm::new(8, 1e-5).expect("new");
    let input = Tensor::from_vec(vec![0.5_f32; 8], &[8]).expect("tensor");
    let out = norm.forward(input).expect("forward");
    assert_eq!(out.shape().iter().product::<usize>(), 8);
}

// ─────────────────────────────────────────────────────────────────────────────
// GQA
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_mistral_v3_gqa_repeat_kv() {
    let cfg = MistralV3Config::small_test(); // 4 Q heads, 2 KV heads → groups=2
    let attn = MistralV3Attention::new(&cfg).expect("new");
    let head_dim = cfg.head_dim();
    let kv = Tensor::from_vec(vec![1.0_f32; head_dim], &[head_dim]).expect("tensor");
    let expanded = attn.repeat_kv(&kv).expect("repeat_kv");
    assert_eq!(
        expanded.shape().iter().product::<usize>(),
        head_dim * cfg.num_query_groups()
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Sliding window attention
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_mistral_v3_sliding_window_effective_window_short_seq() {
    let cfg = MistralV3Config::small_test(); // sliding_window = 8
    let attn = MistralV3Attention::new(&cfg).expect("new");
    // seq_len < sliding_window → full window
    assert_eq!(attn.effective_window(4), 4);
}

#[test]
fn test_mistral_v3_sliding_window_effective_window_long_seq() {
    let cfg = MistralV3Config::small_test(); // sliding_window = 8
    let attn = MistralV3Attention::new(&cfg).expect("new");
    // seq_len > sliding_window → capped
    assert_eq!(attn.effective_window(20), cfg.sliding_window);
}

#[test]
fn test_mistral_v3_attention_forward_long_seq() {
    // seq_len > sliding_window to exercise the window restriction path
    let cfg = MistralV3Config::small_test(); // sliding_window = 8
    let attn = MistralV3Attention::new(&cfg).expect("new");
    let seq_len = 16_usize; // longer than sliding_window=8
    let input = Tensor::from_vec(
        vec![0.1_f32; seq_len * cfg.hidden_size],
        &[seq_len, cfg.hidden_size],
    )
    .expect("tensor");
    let out = attn.forward(input).expect("forward");
    assert_eq!(
        out.shape().iter().product::<usize>(),
        seq_len * cfg.hidden_size
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Full model forward
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_mistral_v3_model_forward_small() {
    let cfg = MistralV3Config::small_test();
    let model = MistralV3Model::new(cfg.clone()).expect("new");
    let out = model.run(vec![1_u32, 2, 3]).expect("run");
    assert_eq!(out.shape().iter().product::<usize>(), 3 * cfg.hidden_size);
}

#[test]
fn test_mistral_v3_causal_lm_forward_small() {
    let cfg = MistralV3Config::small_test();
    let model = MistralV3ForCausalLM::new(cfg.clone()).expect("new");
    let logits = model.forward(vec![0_u32, 1, 2]).expect("forward");
    assert_eq!(logits.shape().iter().product::<usize>(), 3 * cfg.vocab_size);
}

// ─────────────────────────────────────────────────────────────────────────────
// MLP
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_mistral_v3_mlp_forward_shape() {
    let cfg = MistralV3Config::small_test();
    let mlp = MistralV3MLP::new(&cfg).expect("new");
    // Linear layers require at least 2D input
    let input =
        Tensor::from_vec(vec![0.3_f32; cfg.hidden_size], &[1, cfg.hidden_size]).expect("tensor");
    let out = mlp.forward(input).expect("forward");
    assert_eq!(out.shape().iter().product::<usize>(), cfg.hidden_size);
}

// ─────────────────────────────────────────────────────────────────────────────
// Tool definition
// ─────────────────────────────────────────────────────────────────────────────

fn make_weather_tool() -> ToolDefinition {
    let mut parameters = HashMap::new();
    parameters.insert(
        "location".to_string(),
        ToolParameter {
            param_type: "string".to_string(),
            description: "City name".to_string(),
            required: true,
            enum_values: None,
        },
    );
    parameters.insert(
        "unit".to_string(),
        ToolParameter {
            param_type: "string".to_string(),
            description: "Temperature unit".to_string(),
            required: false,
            enum_values: Some(vec!["celsius".to_string(), "fahrenheit".to_string()]),
        },
    );
    ToolDefinition {
        name: "get_weather".to_string(),
        description: "Get current weather for a location".to_string(),
        parameters,
    }
}

#[test]
fn test_tool_definition_creation() {
    let tool = make_weather_tool();
    assert_eq!(tool.name, "get_weather");
    assert!(tool.parameters.contains_key("location"));
}

#[test]
fn test_format_tool_prompt_contains_tool_name() {
    let cfg = MistralV3Config::small_test();
    let model = MistralV3ForCausalLM::new(cfg).expect("new");
    let caller = MistralV3FunctionCaller::new(model, vec![make_weather_tool()]);
    let prompt = caller.format_tool_prompt();
    assert!(
        prompt.contains("get_weather"),
        "prompt should contain tool name"
    );
    assert!(
        prompt.contains("[AVAILABLE_TOOLS]"),
        "prompt should contain [AVAILABLE_TOOLS]"
    );
    assert!(
        prompt.contains("[/AVAILABLE_TOOLS]"),
        "prompt should contain [/AVAILABLE_TOOLS]"
    );
}

#[test]
fn test_parse_tool_calls_valid() {
    let output = r#"Sure! [TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "Paris"}}] More text."#;
    let calls = MistralV3FunctionCaller::parse_tool_calls(output).expect("parse");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(
        calls[0].arguments.get("location").and_then(|v| v.as_str()),
        Some("Paris")
    );
}

#[test]
fn test_parse_tool_calls_no_marker() {
    let output = "This response has no tool calls.";
    let result = MistralV3FunctionCaller::parse_tool_calls(output);
    assert!(matches!(result, Err(ToolParseError::NoToolCalls)));
}

#[test]
fn test_validate_tool_call_missing_required_param() {
    let cfg = MistralV3Config::small_test();
    let model = MistralV3ForCausalLM::new(cfg).expect("new");
    let caller = MistralV3FunctionCaller::new(model, vec![make_weather_tool()]);
    // 'location' is required but absent
    let call = ToolCall {
        name: "get_weather".to_string(),
        arguments: HashMap::new(),
    };
    let result = caller.validate_tool_call(&call);
    assert!(matches!(result, Err(ToolParseError::MissingParam(_))));
}

#[test]
fn test_validate_tool_call_unknown_tool() {
    let cfg = MistralV3Config::small_test();
    let model = MistralV3ForCausalLM::new(cfg).expect("new");
    let caller = MistralV3FunctionCaller::new(model, vec![make_weather_tool()]);
    let call = ToolCall {
        name: "unknown_tool".to_string(),
        arguments: HashMap::new(),
    };
    let result = caller.validate_tool_call(&call);
    assert!(matches!(result, Err(ToolParseError::UnknownTool(_))));
}

#[test]
fn test_tool_parameter_types() {
    let tool = make_weather_tool();
    let location_param = tool.parameters.get("location").expect("location param");
    assert_eq!(location_param.param_type, "string");
    assert!(location_param.required);

    let unit_param = tool.parameters.get("unit").expect("unit param");
    assert!(!unit_param.required);
    assert!(unit_param.enum_values.is_some());
}
