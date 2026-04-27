/// Format a Yi chat prompt in ChatML format.
///
/// Yi-1.5 uses the same ChatML format as Qwen-2:
///
/// ```text
/// <|im_start|>system
/// {system}
/// <|im_end|>
/// <|im_start|>user
/// {user_message}
/// <|im_end|>
/// <|im_start|>assistant
/// {assistant_message}
/// <|im_end|>
/// ...
/// <|im_start|>assistant
/// ```
///
/// `messages` is a slice of `(role, content)` tuples where role is `"user"`
/// or `"assistant"`.  The function appends an open `<|im_start|>assistant\n`
/// tag at the end to prime generation.
pub fn format_yi_chat(system: &str, messages: &[(String, String)]) -> String {
    let mut out = String::new();
    out.push_str("<|im_start|>system\n");
    out.push_str(system);
    out.push_str("\n<|im_end|>\n");
    for (role, content) in messages {
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(content);
        out.push_str("\n<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

#[cfg(test)]
mod chat_tests {
    use super::*;

    #[test]
    fn test_format_yi_chat_system_only() {
        let prompt = format_yi_chat("You are a helpful assistant.", &[]);
        assert!(prompt.contains("<|im_start|>system\nYou are a helpful assistant."));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_format_yi_chat_with_messages() {
        let messages = vec![
            ("user".to_string(), "Hello!".to_string()),
            ("assistant".to_string(), "Hi there!".to_string()),
            ("user".to_string(), "How are you?".to_string()),
        ];
        let prompt = format_yi_chat("sys", &messages);
        assert!(prompt.contains("<|im_start|>user\nHello!\n<|im_end|>"));
        assert!(prompt.contains("<|im_start|>assistant\nHi there!\n<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_format_yi_chat_starts_with_system_block() {
        let prompt = format_yi_chat("system prompt", &[]);
        assert!(prompt.starts_with("<|im_start|>system\n"));
    }

    #[test]
    fn test_format_yi_chat_system_block_closed() {
        let prompt = format_yi_chat("sys", &[]);
        assert!(prompt.contains("sys\n<|im_end|>\n"));
    }

    #[test]
    fn test_format_yi_chat_empty_system_prompt() {
        let prompt = format_yi_chat("", &[]);
        assert!(prompt.starts_with("<|im_start|>system\n"));
        assert!(prompt.contains("\n<|im_end|>\n"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_format_yi_chat_single_user_turn() {
        let messages = vec![(
            "user".to_string(),
            "What is the capital of France?".to_string(),
        )];
        let prompt = format_yi_chat("You are helpful.", &messages);
        assert!(prompt.contains("<|im_start|>user\nWhat is the capital of France?\n<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_format_yi_chat_assistant_role_included() {
        let messages = vec![
            ("user".to_string(), "Hi".to_string()),
            ("assistant".to_string(), "Hello!".to_string()),
        ];
        let prompt = format_yi_chat("sys", &messages);
        assert!(prompt.contains("<|im_start|>assistant\nHello!\n<|im_end|>"));
    }

    #[test]
    fn test_format_yi_chat_primes_with_open_assistant_tag() {
        let messages = vec![("user".to_string(), "Tell me a joke".to_string())];
        let prompt = format_yi_chat("You are a comedian.", &messages);
        // The final open tag must prime assistant generation
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_format_yi_chat_multi_turn_dialogue_order() {
        let messages = vec![
            ("user".to_string(), "Turn1".to_string()),
            ("assistant".to_string(), "Reply1".to_string()),
            ("user".to_string(), "Turn2".to_string()),
            ("assistant".to_string(), "Reply2".to_string()),
            ("user".to_string(), "Turn3".to_string()),
        ];
        let prompt = format_yi_chat("sys", &messages);
        let pos_turn1 = prompt.find("Turn1").expect("Turn1 not found");
        let pos_reply1 = prompt.find("Reply1").expect("Reply1 not found");
        let pos_turn2 = prompt.find("Turn2").expect("Turn2 not found");
        let pos_reply2 = prompt.find("Reply2").expect("Reply2 not found");
        let pos_turn3 = prompt.find("Turn3").expect("Turn3 not found");
        assert!(pos_turn1 < pos_reply1);
        assert!(pos_reply1 < pos_turn2);
        assert!(pos_turn2 < pos_reply2);
        assert!(pos_reply2 < pos_turn3);
    }

    #[test]
    fn test_format_yi_chat_all_messages_closed() {
        // Every message block must be followed by <|im_end|>
        let messages = vec![
            ("user".to_string(), "msg1".to_string()),
            ("assistant".to_string(), "msg2".to_string()),
        ];
        let prompt = format_yi_chat("sys", &messages);
        // sys, user, assistant blocks → 3 closings
        let count = prompt.matches("<|im_end|>").count();
        assert_eq!(count, 3, "Expected 3 <|im_end|> tokens, got {}", count);
    }

    #[test]
    fn test_format_yi_chat_im_start_count() {
        // system + user + assistant (closed) + open assistant tag = 4
        let messages = vec![
            ("user".to_string(), "q".to_string()),
            ("assistant".to_string(), "a".to_string()),
        ];
        let prompt = format_yi_chat("sys", &messages);
        let count = prompt.matches("<|im_start|>").count();
        // system, user, assistant (closed), assistant (open) = 4
        assert_eq!(count, 4, "Expected 4 <|im_start|> tokens, got {}", count);
    }

    #[test]
    fn test_format_yi_chat_special_characters_in_content() {
        let messages = vec![("user".to_string(), "What's 2+2? Isn't it <4>?".to_string())];
        let prompt = format_yi_chat("sys", &messages);
        assert!(prompt.contains("What's 2+2? Isn't it <4>?"));
    }

    #[test]
    fn test_format_yi_chat_multiline_system_prompt() {
        let system = "Line one.\nLine two.\nLine three.";
        let prompt = format_yi_chat(system, &[]);
        assert!(prompt.contains("Line one.\nLine two.\nLine three."));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_format_yi_chat_multiline_user_message() {
        let messages = vec![(
            "user".to_string(),
            "First line\nSecond line\nThird line".to_string(),
        )];
        let prompt = format_yi_chat("sys", &messages);
        assert!(prompt.contains("First line\nSecond line\nThird line"));
    }

    #[test]
    fn test_format_yi_chat_long_conversation() {
        // LCG-generated sequence of 10 turns to simulate multi-round dialogue
        let mut seed: u64 = 42;
        let mut messages = Vec::new();
        for i in 0..10_usize {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let role = if i % 2 == 0 { "user" } else { "assistant" };
            let content = format!("message_{}", seed % 1000);
            messages.push((role.to_string(), content));
        }
        let prompt = format_yi_chat("sys", &messages);
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
        // 10 messages + system = 11 closings
        let close_count = prompt.matches("<|im_end|>").count();
        assert_eq!(close_count, 11);
    }

    #[test]
    fn test_format_yi_chat_no_extra_whitespace_between_blocks() {
        let messages = vec![("user".to_string(), "hello".to_string())];
        let prompt = format_yi_chat("sys", &messages);
        // Blocks must be separated by exactly one newline after <|im_end|>
        assert!(prompt.contains("<|im_end|>\n<|im_start|>"));
    }

    #[test]
    fn test_format_yi_chat_system_immediately_after_newline() {
        let prompt = format_yi_chat("My System", &[]);
        // System tag is at position 0 with content after a newline
        let expected_prefix = "<|im_start|>system\nMy System\n<|im_end|>\n";
        assert!(prompt.starts_with(expected_prefix));
    }

    #[test]
    fn test_format_yi_chat_custom_role_name() {
        // The function should handle any role string, not just "user"/"assistant"
        let messages = vec![("tool".to_string(), "tool result here".to_string())];
        let prompt = format_yi_chat("sys", &messages);
        assert!(prompt.contains("<|im_start|>tool\ntool result here\n<|im_end|>"));
    }

    #[test]
    fn test_format_yi_chat_five_user_messages_count() {
        let messages: Vec<(String, String)> =
            (0..5).map(|i| ("user".to_string(), format!("question {}", i))).collect();
        let prompt = format_yi_chat("sys", &messages);
        // 1 system + 5 user = 6 closings
        let close_count = prompt.matches("<|im_end|>").count();
        assert_eq!(close_count, 6);
    }

    #[test]
    fn test_format_yi_chat_unicode_content() {
        let messages = vec![("user".to_string(), "こんにちは、世界！ 🌏".to_string())];
        let prompt = format_yi_chat("You are helpful.", &messages);
        assert!(prompt.contains("こんにちは、世界！ 🌏"));
    }

    #[test]
    fn test_format_yi_chat_empty_user_message() {
        let messages = vec![("user".to_string(), "".to_string())];
        let prompt = format_yi_chat("sys", &messages);
        // Empty content should still produce a closed block
        assert!(prompt.contains("<|im_start|>user\n\n<|im_end|>"));
    }
}
