/// Tests for streaming generation types: GenerationStream, GenerationToken, FinishReason
#[cfg(test)]
mod tests {
    use crate::generation::streaming::{
        FinishReason, GenerationStream, GenerationStreamTrait, GenerationToken,
    };

    // ---- GenerationToken tests ----

    #[test]
    fn test_generation_token_new_basic() {
        let tok = GenerationToken::new(42, "hello".to_string(), None, false);
        assert_eq!(tok.token_id, 42);
        assert_eq!(tok.token_str, "hello");
        assert!(tok.logprobs.is_none());
        assert!(!tok.is_finished);
        assert!(tok.finish_reason.is_none());
    }

    #[test]
    fn test_generation_token_new_with_logprobs() {
        let tok = GenerationToken::new(7, "world".to_string(), Some(-0.5), false);
        assert_eq!(tok.logprobs, Some(-0.5));
    }

    #[test]
    fn test_generation_token_new_finished() {
        let tok = GenerationToken::new(1, "</s>".to_string(), None, true);
        assert!(tok.is_finished);
    }

    #[test]
    fn test_generation_token_with_finish_reason_max_length() {
        let tok = GenerationToken::new(0, "".to_string(), None, false)
            .with_finish_reason(FinishReason::MaxLength);
        assert!(tok.is_finished);
        assert_eq!(tok.finish_reason, Some(FinishReason::MaxLength));
    }

    #[test]
    fn test_generation_token_with_finish_reason_eos_token() {
        let tok = GenerationToken::new(2, "</s>".to_string(), None, false)
            .with_finish_reason(FinishReason::EosToken);
        assert_eq!(tok.finish_reason, Some(FinishReason::EosToken));
        assert!(tok.is_finished);
    }

    #[test]
    fn test_generation_token_with_finish_reason_stop_sequence() {
        let tok = GenerationToken::new(3, ".".to_string(), None, false)
            .with_finish_reason(FinishReason::StopSequence);
        assert_eq!(tok.finish_reason, Some(FinishReason::StopSequence));
    }

    #[test]
    fn test_generation_token_with_finish_reason_user_stopped() {
        let tok = GenerationToken::new(5, "x".to_string(), None, false)
            .with_finish_reason(FinishReason::UserStopped);
        assert_eq!(tok.finish_reason, Some(FinishReason::UserStopped));
    }

    #[test]
    fn test_generation_token_with_finish_reason_error() {
        let tok = GenerationToken::new(0, "".to_string(), None, false)
            .with_finish_reason(FinishReason::Error);
        assert_eq!(tok.finish_reason, Some(FinishReason::Error));
    }

    // ---- FinishReason tests ----

    #[test]
    fn test_finish_reason_equality() {
        assert_eq!(FinishReason::MaxLength, FinishReason::MaxLength);
        assert_ne!(FinishReason::MaxLength, FinishReason::EosToken);
    }

    #[test]
    fn test_finish_reason_clone() {
        let reason = FinishReason::ConstraintViolation;
        let cloned = reason.clone();
        assert_eq!(reason, cloned);
    }

    #[test]
    fn test_finish_reason_debug() {
        let s = format!("{:?}", FinishReason::StopSequence);
        assert!(s.contains("StopSequence"));
    }

    // ---- GenerationStream tests ----

    #[test]
    fn test_generation_stream_new_is_empty() {
        let stream = GenerationStream::new();
        assert!(stream.is_empty());
        assert_eq!(stream.len(), 0);
        assert!(!stream.is_finished());
    }

    #[test]
    fn test_generation_stream_default_matches_new() {
        let a = GenerationStream::new();
        let b = GenerationStream::default();
        assert_eq!(a.is_empty(), b.is_empty());
        assert_eq!(a.is_finished(), b.is_finished());
    }

    #[test]
    fn test_generation_stream_push_single_token() {
        let mut stream = GenerationStream::new();
        let tok = GenerationToken::new(1, "hi".to_string(), None, false);
        stream.push_token(tok);
        assert_eq!(stream.len(), 1);
        assert!(!stream.is_empty());
        assert!(!stream.is_finished());
    }

    #[test]
    fn test_generation_stream_push_finished_token() {
        let mut stream = GenerationStream::new();
        let tok = GenerationToken::new(0, "</s>".to_string(), None, true);
        stream.push_token(tok);
        assert!(stream.is_finished());
    }

    #[test]
    fn test_generation_stream_push_multiple_tokens() {
        let mut stream = GenerationStream::new();
        for i in 0..5usize {
            let tok = GenerationToken::new(i, format!("tok{}", i), None, false);
            stream.push_token(tok);
        }
        assert_eq!(stream.len(), 5);
    }

    #[test]
    fn test_generation_stream_next_consumes_fifo_order() {
        let mut stream = GenerationStream::new();
        stream.push_token(GenerationToken::new(10, "a".to_string(), None, false));
        stream.push_token(GenerationToken::new(20, "b".to_string(), None, false));

        let first = stream.next();
        assert!(first.is_some());
        assert_eq!(
            first
                .unwrap_or_else(|| GenerationToken::new(0, "".to_string(), None, false))
                .token_id,
            10
        );

        let second = stream.next();
        assert!(second.is_some());
        assert_eq!(
            second
                .unwrap_or_else(|| GenerationToken::new(0, "".to_string(), None, false))
                .token_id,
            20
        );
    }

    #[test]
    fn test_generation_stream_next_empty_returns_none() {
        let mut stream = GenerationStream::new();
        assert!(stream.next().is_none());
    }

    #[test]
    fn test_generation_stream_finish_sets_finished() {
        let mut stream = GenerationStream::new();
        stream.push_token(GenerationToken::new(1, "a".to_string(), None, false));
        stream.finish(FinishReason::MaxLength);
        assert!(stream.is_finished());
    }

    #[test]
    fn test_generation_stream_finish_updates_last_token_reason() {
        let mut stream = GenerationStream::new();
        stream.push_token(GenerationToken::new(1, "token".to_string(), None, false));
        stream.finish(FinishReason::EosToken);
        let tok = stream
            .next()
            .unwrap_or_else(|| GenerationToken::new(0, "".to_string(), None, false));
        assert_eq!(tok.finish_reason, Some(FinishReason::EosToken));
        assert!(tok.is_finished);
    }

    #[test]
    fn test_generation_stream_len_decrements_on_next() {
        let mut stream = GenerationStream::new();
        stream.push_token(GenerationToken::new(1, "a".to_string(), None, false));
        stream.push_token(GenerationToken::new(2, "b".to_string(), None, false));
        assert_eq!(stream.len(), 2);
        let _ = stream.next();
        assert_eq!(stream.len(), 1);
        let _ = stream.next();
        assert_eq!(stream.len(), 0);
    }

    #[test]
    fn test_generation_stream_is_empty_after_drain() {
        let mut stream = GenerationStream::new();
        stream.push_token(GenerationToken::new(1, "a".to_string(), None, false));
        let _ = stream.next();
        assert!(stream.is_empty());
    }

    #[test]
    fn test_generation_stream_finish_no_tokens_stays_finished() {
        let mut stream = GenerationStream::new();
        // finish on empty stream should not panic
        stream.finish(FinishReason::Error);
        assert!(stream.is_finished());
    }

    #[test]
    fn test_generation_token_logprobs_value() {
        let tok = GenerationToken::new(99, "z".to_string(), Some(-std::f32::consts::PI), false);
        let lp = tok.logprobs.unwrap_or(0.0);
        assert!((lp - (-std::f32::consts::PI)).abs() < 1e-5);
    }

    #[test]
    fn test_generation_stream_push_lcg_tokens() {
        let mut stream = GenerationStream::new();
        let mut s = 42u64;
        for _ in 0..20 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let id = (s % 50000) as usize;
            let tok = GenerationToken::new(id, format!("tok{}", id), None, false);
            stream.push_token(tok);
        }
        assert_eq!(stream.len(), 20);
    }
}
