/// Fill-in-the-Middle (FIM) token IDs for StarCoder2.
///
/// StarCoder2 supports the PSM (Prefix-Suffix-Middle) FIM format used during
/// pre-training.  The model is conditioned on a prefix and suffix and must
/// generate the missing middle section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FimTokens {
    /// Token ID for `<fim_prefix>`
    pub prefix_id: u32,
    /// Token ID for `<fim_middle>`
    pub middle_id: u32,
    /// Token ID for `<fim_suffix>`
    pub suffix_id: u32,
    /// Token ID for `<fim_pad>`
    pub pad_id: u32,
}

impl Default for FimTokens {
    fn default() -> Self {
        Self {
            prefix_id: 1,
            middle_id: 2,
            suffix_id: 3,
            pad_id: 4,
        }
    }
}

/// Format a FIM (fill-in-the-middle) prompt in PSM order.
///
/// The prompt is arranged as:
/// ```text
/// <fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>
/// ```
///
/// The model will then generate tokens that belong in the *middle* of the
/// two surrounding fragments.
pub fn format_fim_prompt(prefix: &str, suffix: &str) -> String {
    format!("<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>")
}

/// Parse a raw FIM completion output and extract the generated middle section.
///
/// Looks for `<fim_middle>` and returns the text that follows it (up to
/// `<|endoftext|>` if present, otherwise to the end of the string).
///
/// Returns `None` if no `<fim_middle>` marker is found.
pub fn parse_fim_output(output: &str) -> Option<String> {
    let marker = "<fim_middle>";
    let start = output.find(marker)?;
    let after_marker = &output[start + marker.len()..];
    // Trim at the end-of-text sentinel if present
    let end_of_text = "<|endoftext|>";
    let middle = if let Some(eot_pos) = after_marker.find(end_of_text) {
        &after_marker[..eot_pos]
    } else {
        after_marker
    };
    Some(middle.to_string())
}

#[cfg(test)]
mod fim_tests {
    use super::*;

    #[test]
    fn test_fim_tokens_default() {
        let tokens = FimTokens::default();
        assert_eq!(tokens.prefix_id, 1);
        assert_eq!(tokens.middle_id, 2);
        assert_eq!(tokens.suffix_id, 3);
        assert_eq!(tokens.pad_id, 4);
    }

    #[test]
    fn test_format_fim_prompt_basic() {
        let prompt = format_fim_prompt("def foo():", "    return 42");
        assert!(prompt.starts_with("<fim_prefix>def foo():"));
        assert!(prompt.contains("<fim_suffix>    return 42"));
        assert!(prompt.ends_with("<fim_middle>"));
    }

    #[test]
    fn test_format_fim_prompt_empty_suffix() {
        let prompt = format_fim_prompt("hello", "");
        assert_eq!(prompt, "<fim_prefix>hello<fim_suffix><fim_middle>");
    }

    #[test]
    fn test_parse_fim_output_with_eot() {
        let raw = "<fim_prefix>x<fim_suffix>z<fim_middle>y<|endoftext|>";
        let middle = parse_fim_output(raw);
        assert_eq!(middle, Some("y".to_string()));
    }

    #[test]
    fn test_parse_fim_output_without_eot() {
        let raw = "<fim_prefix>x<fim_suffix>z<fim_middle>some middle text";
        let middle = parse_fim_output(raw);
        assert_eq!(middle, Some("some middle text".to_string()));
    }

    #[test]
    fn test_parse_fim_output_no_marker() {
        let raw = "just plain text with no fim tokens";
        assert!(parse_fim_output(raw).is_none());
    }
}
