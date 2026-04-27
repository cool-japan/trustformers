/// Extended inline tests for the OpenAI compatibility layer.
///
/// These tests supplement the existing test block in `mod.rs` and cover
/// additional paths not exercised there.
#[cfg(test)]
mod openai_compat_extra_tests {
    use super::super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn user_msg(content: &str) -> ChatMessage {
        ChatMessage {
            role: ChatRole::User,
            content: Some(content.to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn sys_msg(content: &str) -> ChatMessage {
        ChatMessage {
            role: ChatRole::System,
            content: Some(content.to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn assistant_msg(content: &str) -> ChatMessage {
        ChatMessage {
            role: ChatRole::Assistant,
            content: Some(content.to_string()),
            name: None,
            tool_calls: None,
            tool_call_id: None,
        }
    }

    fn minimal_req(model: &str) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: model.to_string(),
            messages: vec![user_msg("hello")],
            temperature: None,
            top_p: None,
            n: None,
            max_tokens: None,
            stream: None,
            stop: None,
            presence_penalty: None,
            frequency_penalty: None,
            tools: None,
            tool_choice: None,
            user: None,
        }
    }

    // ── 35. validate_chat_request — empty model returns EmptyModel ────────────
    #[test]
    fn test_validate_empty_model_returns_error() {
        let mut req = minimal_req("gpt-4");
        req.model = "".to_string();
        let err = OpenAiResponseBuilder::validate_chat_request(&req).unwrap_err();
        assert!(matches!(err, OpenAiCompatError::EmptyModel));
    }

    // ── 36. validate_chat_request — whitespace-only model returns EmptyModel ──
    #[test]
    fn test_validate_whitespace_model_returns_error() {
        let mut req = minimal_req("gpt-4");
        req.model = "   ".to_string();
        let err = OpenAiResponseBuilder::validate_chat_request(&req).unwrap_err();
        assert!(matches!(err, OpenAiCompatError::EmptyModel));
    }

    // ── 37. validate_chat_request — empty messages returns EmptyMessages ──────
    #[test]
    fn test_validate_empty_messages_returns_error() {
        let mut req = minimal_req("gpt-4");
        req.messages = vec![];
        let err = OpenAiResponseBuilder::validate_chat_request(&req).unwrap_err();
        assert!(matches!(err, OpenAiCompatError::EmptyMessages));
    }

    // ── 38. validate_chat_request — temperature 0.0 is valid ─────────────────
    #[test]
    fn test_validate_temperature_zero_is_valid() {
        let mut req = minimal_req("gpt-4");
        req.temperature = Some(0.0);
        assert!(OpenAiResponseBuilder::validate_chat_request(&req).is_ok());
    }

    // ── 39. validate_chat_request — temperature 2.0 is valid ─────────────────
    #[test]
    fn test_validate_temperature_two_is_valid() {
        let mut req = minimal_req("gpt-4");
        req.temperature = Some(2.0);
        assert!(OpenAiResponseBuilder::validate_chat_request(&req).is_ok());
    }

    // ── 40. validate_chat_request — temperature > 2 returns InvalidTemperature
    #[test]
    fn test_validate_temperature_too_high() {
        let mut req = minimal_req("gpt-4");
        req.temperature = Some(2.1);
        let err = OpenAiResponseBuilder::validate_chat_request(&req).unwrap_err();
        assert!(matches!(err, OpenAiCompatError::InvalidTemperature(_)));
    }

    // ── 41. validate_chat_request — temperature < 0 returns InvalidTemperature
    #[test]
    fn test_validate_temperature_negative() {
        let mut req = minimal_req("gpt-4");
        req.temperature = Some(-0.1);
        let err = OpenAiResponseBuilder::validate_chat_request(&req).unwrap_err();
        assert!(matches!(err, OpenAiCompatError::InvalidTemperature(_)));
    }

    // ── 42. validate_chat_request — top_p 1.0 is valid ───────────────────────
    #[test]
    fn test_validate_top_p_one_is_valid() {
        let mut req = minimal_req("gpt-4");
        req.top_p = Some(1.0);
        assert!(OpenAiResponseBuilder::validate_chat_request(&req).is_ok());
    }

    // ── 43. validate_chat_request — top_p > 1 returns InvalidTopP ────────────
    #[test]
    fn test_validate_top_p_too_high() {
        let mut req = minimal_req("gpt-4");
        req.top_p = Some(1.001);
        let err = OpenAiResponseBuilder::validate_chat_request(&req).unwrap_err();
        assert!(matches!(err, OpenAiCompatError::InvalidTopP(_)));
    }

    // ── 44. validate_chat_request — max_tokens = 0 returns InvalidMaxTokens ──
    #[test]
    fn test_validate_max_tokens_zero() {
        let mut req = minimal_req("gpt-4");
        req.max_tokens = Some(0);
        let err = OpenAiResponseBuilder::validate_chat_request(&req).unwrap_err();
        assert!(matches!(err, OpenAiCompatError::InvalidMaxTokens));
    }

    // ── 45. validate_chat_request — max_tokens = 1 is valid ──────────────────
    #[test]
    fn test_validate_max_tokens_one_is_valid() {
        let mut req = minimal_req("gpt-4");
        req.max_tokens = Some(1);
        assert!(OpenAiResponseBuilder::validate_chat_request(&req).is_ok());
    }

    // ── 46. extract_system — returns first system message content ─────────────
    #[test]
    fn test_extract_system_returns_first_system_msg() {
        let msgs = vec![
            user_msg("hello"),
            sys_msg("You are a helpful assistant."),
            sys_msg("Second system message"),
        ];
        let sys = OpenAiResponseBuilder::extract_system(&msgs);
        assert_eq!(
            sys.as_deref(),
            Some("You are a helpful assistant."),
            "should return first system message"
        );
    }

    // ── 47. extract_system — returns None when no system message ──────────────
    #[test]
    fn test_extract_system_none_when_absent() {
        let msgs = vec![user_msg("hi"), assistant_msg("hello")];
        assert!(OpenAiResponseBuilder::extract_system(&msgs).is_none());
    }

    // ── 48. messages_to_prompt — contains role prefix ─────────────────────────
    #[test]
    fn test_messages_to_prompt_contains_role_prefix() {
        let msgs = vec![sys_msg("Be helpful"), user_msg("What is Rust?")];
        let prompt = OpenAiResponseBuilder::messages_to_prompt(&msgs);
        assert!(
            prompt.contains("system: Be helpful"),
            "prompt must include system prefix"
        );
        assert!(
            prompt.contains("user: What is Rust?"),
            "prompt must include user prefix"
        );
    }

    // ── 49. messages_to_prompt — empty messages returns empty string ──────────
    #[test]
    fn test_messages_to_prompt_empty_messages() {
        let prompt = OpenAiResponseBuilder::messages_to_prompt(&[]);
        assert_eq!(prompt, "");
    }

    // ── 50. messages_to_prompt — handles None content ─────────────────────────
    #[test]
    fn test_messages_to_prompt_none_content() {
        let msg = ChatMessage {
            role: ChatRole::User,
            content: None,
            name: None,
            tool_calls: None,
            tool_call_id: None,
        };
        let prompt = OpenAiResponseBuilder::messages_to_prompt(&[msg]);
        assert!(
            prompt.contains("user: "),
            "must include role even with None content"
        );
    }

    // ── 51. count_tokens — empty string returns 0 ────────────────────────────
    #[test]
    fn test_count_tokens_empty_string() {
        assert_eq!(OpenAiResponseBuilder::count_tokens(""), 0);
    }

    // ── 52. count_tokens — 8 char string returns 2 ───────────────────────────
    #[test]
    fn test_count_tokens_eight_chars() {
        // len / 4 = 8 / 4 = 2
        assert_eq!(OpenAiResponseBuilder::count_tokens("abcdefgh"), 2);
    }

    // ── 53. model_list — produces ModelListResponse with correct count ─────────
    #[test]
    fn test_model_list_correct_count() {
        let models = ["gpt-4", "gpt-3.5-turbo", "claude-3"];
        let resp = OpenAiResponseBuilder::model_list(&models);
        assert_eq!(resp.object, "list");
        assert_eq!(resp.data.len(), 3);
    }

    // ── 54. model_list — each ModelInfo.object is "model" ─────────────────────
    #[test]
    fn test_model_list_object_field() {
        let resp = OpenAiResponseBuilder::model_list(&["gpt-4"]);
        assert_eq!(resp.data[0].object, "model");
        assert_eq!(resp.data[0].id, "gpt-4");
    }

    // ── 55. chat_completion — id starts with "chatcmpl-" ─────────────────────
    #[test]
    fn test_chat_completion_id_prefix() {
        let resp = OpenAiResponseBuilder::chat_completion("gpt-4", &[], "text", 10, 5);
        assert!(
            resp.id.starts_with("chatcmpl-"),
            "id must start with chatcmpl-, got {}",
            resp.id
        );
    }

    // ── 56. chat_completion — object is "chat.completion" ────────────────────
    #[test]
    fn test_chat_completion_object_field() {
        let resp = OpenAiResponseBuilder::chat_completion("gpt-4", &[], "text", 10, 5);
        assert_eq!(resp.object, "chat.completion");
    }

    // ── 57. completion — object is "text_completion" ──────────────────────────
    #[test]
    fn test_completion_response_object_field() {
        let resp = OpenAiResponseBuilder::completion("davinci", "prompt", "output", 5, 3);
        assert_eq!(resp.object, "text_completion");
    }

    // ── 58. EmbeddingData.object is "embedding" ───────────────────────────────
    #[test]
    fn test_embedding_response_data_object_field() {
        let resp = OpenAiResponseBuilder::embedding("ada", &["hello"], &[vec![0.1, 0.2, 0.3]]);
        assert_eq!(resp.data[0].object, "embedding");
    }

    // ── 59. OpenAiError internal_error code is "internal_error" ──────────────
    #[test]
    fn test_internal_error_code() {
        let e = OpenAiError::internal_error("something broke");
        assert_eq!(e.error.code.as_deref(), Some("internal_error"));
        assert_eq!(e.error.error_type, "server_error");
    }

    // ── 60. EmbeddingInput::Tokens variant holds vec of u32 ──────────────────
    #[test]
    fn test_embedding_input_tokens_variant() {
        let input = EmbeddingInput::Tokens(vec![1u32, 2, 3]);
        match input {
            EmbeddingInput::Tokens(ids) => assert_eq!(ids.len(), 3),
            _ => panic!("expected Tokens variant"),
        }
    }

    // ── 61. FunctionCall name and arguments fields are populated ──────────────
    #[test]
    fn test_function_call_fields() {
        let fc = FunctionCall {
            name: "get_weather".to_string(),
            arguments: r#"{"city":"London"}"#.to_string(),
        };
        assert_eq!(fc.name, "get_weather");
        assert!(fc.arguments.contains("London"));
    }

    // ── 62. ToolCallMessage fields are populated ──────────────────────────────
    #[test]
    fn test_tool_call_message_fields() {
        let tc = ToolCallMessage {
            id: "call_001".to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: "search".to_string(),
                arguments: "{}".to_string(),
            },
        };
        assert_eq!(tc.id, "call_001");
        assert_eq!(tc.call_type, "function");
    }

    // ── 63. ToolDefinition fields are populated ───────────────────────────────
    #[test]
    fn test_tool_definition_fields() {
        let td = ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: "my_fn".to_string(),
                description: "does stuff".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            },
        };
        assert_eq!(td.tool_type, "function");
        assert_eq!(td.function.name, "my_fn");
    }

    // ── 64. ChatCompletionRequest with all optional fields set ────────────────
    #[test]
    fn test_chat_completion_request_all_fields() {
        let req = ChatCompletionRequest {
            model: "gpt-4".to_string(),
            messages: vec![user_msg("test")],
            temperature: Some(0.5),
            top_p: Some(0.9),
            n: Some(2),
            max_tokens: Some(256),
            stream: Some(false),
            stop: Some(vec!["END".to_string()]),
            presence_penalty: Some(0.1),
            frequency_penalty: Some(0.2),
            tools: None,
            tool_choice: None,
            user: Some("user-123".to_string()),
        };
        assert_eq!(req.n.unwrap_or(0), 2);
        assert_eq!(req.stop.as_ref().map(|s| s.len()).unwrap_or(0), 1);
    }

    // ── 65. UsageStats total_tokens is prompt + completion ────────────────────
    #[test]
    fn test_usage_stats_total_tokens_sum() {
        let usage = UsageStats {
            prompt_tokens: 37,
            completion_tokens: 13,
            total_tokens: 50,
        };
        assert_eq!(
            usage.total_tokens,
            usage.prompt_tokens + usage.completion_tokens
        );
    }

    // ── 66. EmbeddingUsage total matches prompt ───────────────────────────────
    #[test]
    fn test_embedding_usage_total_matches_prompt() {
        let resp = OpenAiResponseBuilder::embedding("ada", &["test"], &[vec![0.1f32; 4]]);
        // For embedding responses total_tokens == prompt_tokens
        assert_eq!(resp.usage.total_tokens, resp.usage.prompt_tokens);
    }

    // ── 67. ChatChoice with logprobs field ────────────────────────────────────
    #[test]
    fn test_chat_choice_with_logprobs() {
        let choice = ChatChoice {
            index: 0,
            message: user_msg("hello"),
            finish_reason: "stop".to_string(),
            logprobs: Some(serde_json::json!({"tokens": []})),
        };
        assert!(choice.logprobs.is_some());
    }

    // ── 68. CompletionPrompt::Multiple contains all strings ───────────────────
    #[test]
    fn test_completion_prompt_multiple() {
        let p = CompletionPrompt::Multiple(vec!["a".to_string(), "b".to_string()]);
        match p {
            CompletionPrompt::Multiple(v) => assert_eq!(v.len(), 2),
            _ => panic!("expected Multiple"),
        }
    }

    // ── 69. ModelInfo object field is "model" ─────────────────────────────────
    #[test]
    fn test_model_info_object_field() {
        let info = ModelInfo {
            id: "gpt-4".to_string(),
            object: "model".to_string(),
            created: 1_700_000_000,
            owned_by: "openai".to_string(),
        };
        assert_eq!(info.object, "model");
    }

    // ── 70. OpenAiError rate_limit_exceeded error_type is "requests" ──────────
    #[test]
    fn test_rate_limit_error_type() {
        let e = OpenAiError::rate_limit_exceeded();
        assert_eq!(e.error.error_type, "requests");
        assert!(e.error.param.is_none());
    }
}
