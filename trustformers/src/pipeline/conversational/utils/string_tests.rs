//! Tests for string utility functions.

#[cfg(test)]
mod tests {
    use super::super::string::StringUtils;

    // -------------------------------------------------------------------------
    // contains_any
    // -------------------------------------------------------------------------

    #[test]
    fn test_contains_any_single_match() {
        let result = StringUtils::contains_any("Hello world", &["world"]);
        assert!(result, "Should find 'world' in text");
    }

    #[test]
    fn test_contains_any_case_insensitive() {
        let result = StringUtils::contains_any("Hello WORLD", &["world"]);
        assert!(result, "Should match case-insensitively");
    }

    #[test]
    fn test_contains_any_no_match() {
        let result = StringUtils::contains_any("Hello world", &["missing", "absent"]);
        assert!(!result, "Should return false when no patterns match");
    }

    #[test]
    fn test_contains_any_empty_patterns() {
        let result = StringUtils::contains_any("Hello world", &[]);
        assert!(!result, "Empty patterns should return false");
    }

    #[test]
    fn test_contains_any_multiple_patterns_first_matches() {
        let result = StringUtils::contains_any("Hello world", &["hello", "absent"]);
        assert!(result, "Should return true when first pattern matches");
    }

    #[test]
    fn test_contains_any_multiple_patterns_last_matches() {
        let result = StringUtils::contains_any("Hello world", &["absent", "world"]);
        assert!(result, "Should return true when last pattern matches");
    }

    // -------------------------------------------------------------------------
    // matches_any
    // -------------------------------------------------------------------------

    #[test]
    fn test_matches_any_exact_match() {
        let result = StringUtils::matches_any("hello", &["hello"]);
        assert!(result, "Should match exact string");
    }

    #[test]
    fn test_matches_any_case_insensitive_exact() {
        let result = StringUtils::matches_any("HELLO", &["hello"]);
        assert!(result, "Should match case-insensitively with exact string");
    }

    #[test]
    fn test_matches_any_partial_not_match() {
        let result = StringUtils::matches_any("hello world", &["hello"]);
        assert!(!result, "Partial match should not count as full match");
    }

    #[test]
    fn test_matches_any_no_match() {
        let result = StringUtils::matches_any("hello", &["world", "foo"]);
        assert!(!result, "Should return false when no patterns match exactly");
    }

    #[test]
    fn test_matches_any_empty_patterns() {
        let result = StringUtils::matches_any("hello", &[]);
        assert!(!result, "Empty patterns should return false");
    }

    // -------------------------------------------------------------------------
    // extract_keywords
    // -------------------------------------------------------------------------

    #[test]
    fn test_extract_keywords_basic() {
        let keywords = StringUtils::extract_keywords("machine learning is amazing", 3);
        assert!(keywords.contains(&"machine".to_string()), "Should extract 'machine'");
        assert!(keywords.contains(&"learning".to_string()), "Should extract 'learning'");
        assert!(keywords.contains(&"amazing".to_string()), "Should extract 'amazing'");
    }

    #[test]
    fn test_extract_keywords_removes_stop_words() {
        let keywords = StringUtils::extract_keywords("the cat is on the mat", 2);
        // "the", "is", "on" are stop words
        assert!(!keywords.contains(&"the".to_string()), "Should remove 'the'");
        assert!(!keywords.contains(&"is".to_string()), "Should remove 'is'");
        assert!(!keywords.contains(&"on".to_string()), "Should remove 'on'");
    }

    #[test]
    fn test_extract_keywords_min_length_filter() {
        let keywords = StringUtils::extract_keywords("ok go run fast speed", 5);
        assert!(!keywords.contains(&"ok".to_string()), "Should filter short words");
        assert!(!keywords.contains(&"go".to_string()), "Should filter 2-char words");
        assert!(!keywords.contains(&"run".to_string()), "Should filter 3-char words");
        assert!(keywords.contains(&"speed".to_string()), "Should include 5-char word");
    }

    #[test]
    fn test_extract_keywords_empty_text() {
        let keywords = StringUtils::extract_keywords("", 3);
        assert!(keywords.is_empty(), "Empty text should return no keywords");
    }

    #[test]
    fn test_extract_keywords_lowercase_output() {
        let keywords = StringUtils::extract_keywords("Machine Learning", 3);
        assert!(keywords.contains(&"machine".to_string()), "Keywords should be lowercase");
        assert!(keywords.contains(&"learning".to_string()), "Keywords should be lowercase");
    }

    // -------------------------------------------------------------------------
    // string_similarity
    // -------------------------------------------------------------------------

    #[test]
    fn test_string_similarity_identical() {
        let sim = StringUtils::string_similarity("hello world", "hello world");
        let diff = (sim - 1.0_f32).abs();
        assert!(diff < 1e-6, "Identical strings should have similarity 1.0");
    }

    #[test]
    fn test_string_similarity_case_insensitive() {
        let sim = StringUtils::string_similarity("HELLO WORLD", "hello world");
        let diff = (sim - 0.9_f32).abs();
        assert!(diff < 1e-6, "Case-only difference should have similarity 0.9");
    }

    #[test]
    fn test_string_similarity_no_overlap() {
        let sim = StringUtils::string_similarity("cat dog", "fish bird");
        assert!(sim < 0.5, "Completely different strings should have low similarity");
    }

    #[test]
    fn test_string_similarity_partial_overlap() {
        let sim = StringUtils::string_similarity("cat dog fish", "cat bird fish");
        assert!(sim > 0.0 && sim < 1.0, "Partially overlapping strings should have intermediate similarity");
    }

    #[test]
    fn test_string_similarity_empty_strings() {
        let sim = StringUtils::string_similarity("", "");
        let diff = (sim - 1.0_f32).abs();
        assert!(diff < 1e-6, "Two empty strings should be identical (similarity 1.0)");
    }

    // -------------------------------------------------------------------------
    // is_meaningful
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_meaningful_normal_text() {
        assert!(StringUtils::is_meaningful("Hello world"), "Normal text should be meaningful");
    }

    #[test]
    fn test_is_meaningful_too_short() {
        assert!(!StringUtils::is_meaningful("a"), "Single char should not be meaningful");
        assert!(!StringUtils::is_meaningful(""), "Empty string should not be meaningful");
    }

    #[test]
    fn test_is_meaningful_repeated_chars() {
        assert!(!StringUtils::is_meaningful("aaaa"), "Repeated chars should not be meaningful");
    }

    #[test]
    fn test_is_meaningful_no_alphabetic() {
        assert!(!StringUtils::is_meaningful("1234"), "All digits should not be meaningful");
    }

    #[test]
    fn test_is_meaningful_whitespace_only() {
        assert!(!StringUtils::is_meaningful("   "), "Whitespace only should not be meaningful");
    }

    // -------------------------------------------------------------------------
    // normalize_whitespace
    // -------------------------------------------------------------------------

    #[test]
    fn test_normalize_whitespace_multiple_spaces() {
        let result = StringUtils::normalize_whitespace("hello   world");
        assert_eq!(result, "hello world", "Multiple spaces should be collapsed");
    }

    #[test]
    fn test_normalize_whitespace_tabs_and_newlines() {
        let result = StringUtils::normalize_whitespace("hello\t\nworld");
        assert_eq!(result, "hello world", "Tabs and newlines should be collapsed");
    }

    #[test]
    fn test_normalize_whitespace_leading_trailing() {
        let result = StringUtils::normalize_whitespace("  hello world  ");
        assert_eq!(result, "hello world", "Leading/trailing whitespace should be removed");
    }

    #[test]
    fn test_normalize_whitespace_empty() {
        let result = StringUtils::normalize_whitespace("");
        assert_eq!(result, "", "Empty string should remain empty");
    }

    // -------------------------------------------------------------------------
    // truncate_words
    // -------------------------------------------------------------------------

    #[test]
    fn test_truncate_words_short_text() {
        let result = StringUtils::truncate_words("hello world", 100);
        assert_eq!(result, "hello world", "Short text should not be truncated");
    }

    #[test]
    fn test_truncate_words_exact_length() {
        let text = "hello world";
        let result = StringUtils::truncate_words(text, text.len());
        assert_eq!(result, text, "Text at exact length should not be truncated");
    }

    #[test]
    fn test_truncate_words_appends_ellipsis() {
        let result = StringUtils::truncate_words("hello world foo bar", 10);
        assert!(result.ends_with("..."), "Truncated text should end with '...'");
    }

    #[test]
    fn test_truncate_words_word_boundary() {
        let result = StringUtils::truncate_words("hello world foo bar", 12);
        // Should not cut in middle of word
        let words: Vec<&str> = result.trim_end_matches("...").split_whitespace().collect();
        for word in &words {
            let clean = word.trim_end_matches("...");
            assert!(
                "hello world foo bar".contains(clean),
                "Result should only contain complete words"
            );
        }
    }

    #[test]
    fn test_truncate_words_empty_text() {
        let result = StringUtils::truncate_words("", 10);
        assert_eq!(result, "", "Empty text should remain empty");
    }
}
