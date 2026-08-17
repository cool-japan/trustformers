//! HuggingFace-compatible `chat_template` rendering.
//!
//! Modern instruction-tuned checkpoints (Llama, Mistral, Qwen, ChatML-style
//! models, ...) ship a Jinja2 template in `tokenizer_config.json`'s
//! `chat_template` field instead of a fixed set of role markers. This module
//! reads that template and renders it with a real (pure-Rust) Jinja2 engine
//! ([`minijinja`]) so conversations are formatted exactly the way the
//! checkpoint expects.
//!
//! This is intentionally independent from
//! [`crate::special_tokens::SpecialTokenManager::format_conversation`], which
//! remains available as this crate's own generic `<|role|>...<|end|>`
//! fallback format for tokenizers that do not ship a `chat_template` at all.

use crate::special_tokens::ConversationMessage;
use minijinja::{Environment, Error as MjError, ErrorKind as MjErrorKind, Value as MjValue};
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;
use trustformers_core::errors::{Result, TrustformersError};

/// Name under which the compiled template is registered in the
/// [`minijinja::Environment`]. Only one template is ever loaded per engine.
const TEMPLATE_NAME: &str = "chat_template";

/// One message as exposed to the Jinja template (`message.role`,
/// `message['role']`, `message.content`, `message['content']`).
#[derive(Debug, Clone, Serialize)]
struct TemplateMessage<'a> {
    role: &'a str,
    content: &'a str,
}

/// Top-level template context, matching the variables HuggingFace's
/// `tokenizer.apply_chat_template` exposes: `messages`,
/// `add_generation_prompt`, and any `*_token` special-token strings the
/// tokenizer config carries (flattened to the top level).
#[derive(Debug, Serialize)]
struct ChatTemplateContext<'a> {
    messages: Vec<TemplateMessage<'a>>,
    add_generation_prompt: bool,
    #[serde(flatten)]
    extra: &'a HashMap<String, String>,
}

/// A compiled HuggingFace chat template, ready to render conversations.
///
/// Construct one with [`ChatTemplateEngine::from_tokenizer_config_json`] (the
/// common path: point it at a HuggingFace `tokenizer_config.json`) or
/// [`ChatTemplateEngine::from_template_str`] when the Jinja source is already
/// in hand.
pub struct ChatTemplateEngine {
    env: Environment<'static>,
    /// Extra top-level template variables read from `tokenizer_config.json`
    /// (`bos_token`, `eos_token`, `pad_token`, `unk_token`), exposed to the
    /// template exactly as HuggingFace's `apply_chat_template` does.
    extra_context: HashMap<String, String>,
}

impl ChatTemplateEngine {
    /// Compile a chat template from its raw Jinja2 source string.
    pub fn from_template_str(template: &str) -> Result<Self> {
        let mut env = Environment::new();

        // HuggingFace chat templates commonly call `raise_exception(msg)` to
        // reject invalid role sequences (e.g. Mistral's official template).
        // Register it so such templates compile and behave the way they do
        // under `transformers` instead of failing with "unknown function".
        env.add_function(
            "raise_exception",
            |message: String| -> std::result::Result<MjValue, MjError> {
                Err(MjError::new(MjErrorKind::InvalidOperation, message))
            },
        );

        env.add_template_owned(TEMPLATE_NAME, template.to_string()).map_err(|e| {
            TrustformersError::invalid_config(format!("Invalid chat_template: {}", e))
        })?;

        Ok(Self {
            env,
            extra_context: HashMap::new(),
        })
    }

    /// Load `chat_template` (and any `*_token` special-token strings) from a
    /// HuggingFace `tokenizer_config.json` file.
    ///
    /// Returns `Ok(None)` when the file parses but carries no usable
    /// `chat_template` field — this is a normal, common case, not every
    /// tokenizer ships one — and `Err` for I/O errors, malformed JSON, or a
    /// `chat_template` that fails to compile as Jinja2.
    pub fn from_tokenizer_config_json<P: AsRef<Path>>(path: P) -> Result<Option<Self>> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            TrustformersError::io_error(format!("Failed to read tokenizer_config.json: {}", e))
        })?;
        let value: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
            TrustformersError::serialization_error(format!(
                "Failed to parse tokenizer_config.json: {}",
                e
            ))
        })?;
        Self::from_tokenizer_config_value(&value)
    }

    /// Same as [`Self::from_tokenizer_config_json`] but from an already
    /// parsed JSON value, for callers that already loaded the file.
    pub fn from_tokenizer_config_value(value: &serde_json::Value) -> Result<Option<Self>> {
        let template = match value.get("chat_template") {
            Some(serde_json::Value::String(s)) => s.clone(),
            // A handful of checkpoints (tool-use variants, multi-template
            // configs) store a list of `{"name": ..., "template": ...}`
            // entries instead of a single string; prefer the one named
            // "default", falling back to the first entry.
            Some(serde_json::Value::Array(templates)) => {
                let chosen = templates
                    .iter()
                    .find(|entry| entry.get("name").and_then(|n| n.as_str()) == Some("default"))
                    .or_else(|| templates.first());
                match chosen.and_then(|entry| entry.get("template")).and_then(|t| t.as_str()) {
                    Some(s) => s.to_string(),
                    None => return Ok(None),
                }
            },
            _ => return Ok(None),
        };

        let mut engine = Self::from_template_str(&template)?;

        for key in ["bos_token", "eos_token", "pad_token", "unk_token"] {
            // These fields are either a plain string or an `AddedToken`-style
            // object with a `content` field, depending on tokenizer_config
            // version; accept both.
            let token = value.get(key).and_then(|v| {
                v.as_str()
                    .map(str::to_string)
                    .or_else(|| v.get("content").and_then(|c| c.as_str()).map(str::to_string))
            });
            if let Some(token) = token {
                engine.extra_context.insert(key.to_string(), token);
            }
        }

        Ok(Some(engine))
    }

    /// Render `messages` through the template, matching HuggingFace's
    /// `tokenizer.apply_chat_template(messages, add_generation_prompt=...)`.
    pub fn apply_chat_template(
        &self,
        messages: &[ConversationMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        let template = self
            .env
            .get_template(TEMPLATE_NAME)
            .map_err(|e| TrustformersError::other(format!("Chat template not compiled: {}", e)))?;

        let context = ChatTemplateContext {
            messages: messages
                .iter()
                .map(|m| TemplateMessage {
                    role: &m.role,
                    content: &m.content,
                })
                .collect(),
            add_generation_prompt,
            extra: &self.extra_context,
        };

        template
            .render(context)
            .map_err(|e| TrustformersError::other(format!("Failed to render chat template: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal ChatML-style template, structurally representative of what
    /// real `tokenizer_config.json` files ship (Qwen, and many fine-tunes):
    /// a `{% for %}` loop over `messages`, dict-style field access, string
    /// concatenation, and a trailing `{% if add_generation_prompt %}` block.
    /// None of that Jinja control flow existed under the old
    /// `{key}`-placeholder-substitution renderer.
    const CHATML_TEMPLATE: &str = "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}";

    #[test]
    fn test_renders_real_jinja_control_flow() {
        let engine =
            ChatTemplateEngine::from_template_str(CHATML_TEMPLATE).expect("template compiles");

        let messages = vec![
            ConversationMessage::system("You are helpful.".to_string()),
            ConversationMessage::user("Hi there".to_string()),
        ];

        let rendered =
            engine.apply_chat_template(&messages, true).expect("Operation failed in test");

        let expected = "<|im_start|>system\nYou are helpful.<|im_end|>\n\
             <|im_start|>user\nHi there<|im_end|>\n\
             <|im_start|>assistant\n";
        assert_eq!(rendered, expected);

        // The old flat `{key}` substitution renderer could never have
        // produced this: there was no `{% for %}` interpreter, so a template
        // string containing Jinja syntax would have passed through the old
        // `result.replace(...)` untouched, leaving literal "{% for message
        // in messages %}" in the output.
        assert!(!rendered.contains("{%"));
        assert!(!rendered.contains("messages %}"));
    }

    #[test]
    fn test_add_generation_prompt_toggles_output() {
        let engine =
            ChatTemplateEngine::from_template_str(CHATML_TEMPLATE).expect("template compiles");
        let messages = vec![ConversationMessage::user("hi".to_string())];

        let with_prompt = engine.apply_chat_template(&messages, true).expect("render ok");
        let without_prompt = engine.apply_chat_template(&messages, false).expect("render ok");

        assert!(with_prompt.ends_with("<|im_start|>assistant\n"));
        assert!(!without_prompt.contains("<|im_start|>assistant"));
    }

    #[test]
    fn test_special_tokens_are_exposed_to_template() {
        let engine =
            ChatTemplateEngine::from_template_str("{{ bos_token }}{{ messages[0]['content'] }}")
                .expect("template compiles");
        // `from_template_str` alone has no extra_context; verify the JSON
        // loading path wires bos_token/eos_token/etc. through by exercising
        // `from_tokenizer_config_value` directly.
        let config = serde_json::json!({
            "chat_template": "{{ bos_token }}{{ messages[0]['content'] }}",
            "bos_token": "<s>",
        });
        let engine_from_config = ChatTemplateEngine::from_tokenizer_config_value(&config)
            .expect("Operation failed in test")
            .expect("chat_template present");
        let messages = vec![ConversationMessage::user("hello".to_string())];
        let rendered = engine_from_config
            .apply_chat_template(&messages, false)
            .expect("Operation failed in test");
        assert_eq!(rendered, "<s>hello");

        // Without bos_token in context, the same template renders the
        // built-in `Undefined` as an empty string rather than panicking.
        let rendered_bare =
            engine.apply_chat_template(&messages, false).expect("Operation failed in test");
        assert_eq!(rendered_bare, "hello");
    }

    #[test]
    fn test_missing_chat_template_returns_none_not_error() {
        let config = serde_json::json!({ "tokenizer_type": "BPE" });
        let engine = ChatTemplateEngine::from_tokenizer_config_value(&config).expect("parses fine");
        assert!(engine.is_none());
    }

    #[test]
    fn test_malformed_template_is_a_structured_error() {
        // Unclosed `{% for %}` block: must be a compile error, not silently
        // accepted or panicking.
        let result = ChatTemplateEngine::from_template_str("{% for m in messages %}{{ m }}");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_tokenizer_config_json_reads_real_file() {
        let dir = std::env::temp_dir().join(format!(
            "trustformers_chat_template_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("Operation failed in test")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("Operation failed in test");
        let path = dir.join("tokenizer_config.json");
        std::fs::write(
            &path,
            r#"{"chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}"}"#,
        )
        .expect("Operation failed in test");

        let engine = ChatTemplateEngine::from_tokenizer_config_json(&path)
            .expect("Operation failed in test")
            .expect("chat_template present");
        let messages = vec![ConversationMessage::user("abc".to_string())];
        let rendered =
            engine.apply_chat_template(&messages, false).expect("Operation failed in test");
        assert_eq!(rendered, "abc");

        std::fs::remove_dir_all(&dir).ok();
    }
}
