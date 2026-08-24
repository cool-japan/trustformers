# Migrating from OpenAI tiktoken to TrustformeRS

> **Rewritten 2026-08-24.** The previous version of this guide (through late 2026-08-18) described a large fictional API — chat-completion templating, batching/padding helpers, LRU/adaptive caching configuration, cost estimation, production Prometheus metrics, shadow-testing and gradual-rollout helpers, a `TiktokenModelManager`, and Python bindings — none of which exist in this crate's real `TiktokenTokenizer`. Roughly 30 fabricated types/methods were removed. What follows documents only the API that actually exists in `trustformers-tokenizers/src/tiktoken/mod.rs` and `ranks.rs` today, verified by reading that source directly.

## What this crate actually gives you

`TiktokenTokenizer` implements the real tiktoken algorithm: text is split with the encoding's own regex (`CL100K_PATTERN` / `R50K_PATTERN`, both `pub const`), and each pre-token's raw bytes are merged by BPE rank until no ranked pair remains. It supports the three published OpenAI encodings (`cl100k_base`, `p50k_base`, `r50k_base`) plus any custom rank table you load yourself.

**This crate never ships or fabricates OpenAI's BPE rank tables.** `cl100k_base()` / `p50k_base()` / `r50k_base()` locate a `.tiktoken` rank file through a fixed, documented search order (`$TRUSTFORMERS_TIKTOKEN_DIR`, then `$TIKTOKEN_CACHE_DIR`, then `$XDG_CACHE_HOME/tiktoken`, then `$HOME/.cache/tiktoken`, then `./` and `./tiktoken/`, all as `{encoding}.tiktoken`) and return a hard `Err` naming every path it probed if none exists. Unlike the Python `tiktoken` package, this crate never downloads the file for you — obtain one yourself (for example from the `openaipublic` tiktoken bucket) and place it in one of those locations, point `$TRUSTFORMERS_TIKTOKEN_DIR` at it, or load it explicitly with `from_file`/`from_reader`.

There are **no Python bindings for tiktoken** in this crate today (`trustformers-tokenizers/src/python.rs` exposes other tokenizers but not `TiktokenTokenizer`) — everything below is Rust.

### Performance

No benchmark harness in this repository has produced a tiktoken-vs-TrustformeRS comparison number, so none is quoted here. Run this crate's own Criterion benchmark (`trustformers-tokenizers/benches/tokenizer_performance.rs`) against a real `tiktoken` install on your own hardware if you need one.

## Rust API

### Loading an encoding

```rust
use trustformers_tokenizers::tiktoken::TiktokenTokenizer;

// The three published OpenAI encodings — each locates its rank file on disk
// via the search order above, or returns a hard, path-naming Err.
let cl100k = TiktokenTokenizer::cl100k_base()?;
let p50k = TiktokenTokenizer::p50k_base()?;
let r50k = TiktokenTokenizer::r50k_base()?;

// Same lookup, by name instead of a dedicated method.
let cl100k = TiktokenTokenizer::from_encoding_name("cl100k_base")?;

// Load an already-downloaded rank file yourself, no search involved.
let tokenizer = TiktokenTokenizer::from_file("path/to/cl100k_base.tiktoken")?;
// ...or from anything implementing `BufRead`:
let tokenizer = TiktokenTokenizer::from_reader(std::io::BufReader::new(reader))?;

// A rank table you built or loaded another way, with no special tokens and
// the default (r50k) pre-tokenizer pattern.
let tokenizer = TiktokenTokenizer::new(rank_map, Default::default(), None)?;
```

`EncodingSpec` (`CL100K_BASE` / `P50K_BASE` / `R50K_BASE`, all `pub const`) carries each encoding's name, pre-tokenizer pattern, and special tokens if you need to build a `TiktokenTokenizer` from a rank table you already have in memory (`from_encoding_spec`) or want the on-disk search behavior for a spec you constructed yourself (`from_encoding_spec_on_disk`).

For a rank file plus a separate `token id` special-tokens file (the format the `openai-python` tooling produces), use `from_tiktoken_file(encoder_path, special_tokens_path)`.

### Builder-style configuration

The only configuration this tokenizer exposes is special tokens and the pre-tokenizer pattern — both consuming builders:

```rust
use std::collections::HashMap;

let tokenizer = TiktokenTokenizer::from_file("custom.tiktoken")?
    .with_special_tokens(HashMap::from([("<|custom|>".to_string(), 100_000)]))
    .with_pattern_str(r"\s+|\S+")?; // or .with_pattern(a precompiled fancy_regex::Regex)
```

There is no `add_special_token` chaining method, no `CacheStrategy`/`MemoryPoolConfig`, no vocabulary-compression toggle, and no thread-pool sizing — the cache this tokenizer keeps internally (pre-token bytes → token ids, bounded at 65,536 entries) is not user-configurable.

### Encoding and decoding

```rust
// Ignore special-token text entirely — treat it as ordinary text.
let tokens = tokenizer.encode_ordinary("Hello, world!")?;

// Recognize every special token this tokenizer knows, leftmost-longest.
let tokens = tokenizer.encode_text("Hello <|endoftext|> world")?;

// Recognize only an explicit allow-list of special tokens.
use std::collections::HashSet;
let allowed: HashSet<String> = ["<|endoftext|>".to_string()].into();
let tokens = tokenizer.encode_with_special_tokens("Hello <|endoftext|> world", &allowed)?;

// Decode back to bytes or to a UTF-8 `String` (errors if the bytes aren't valid UTF-8).
let bytes: Vec<u8> = tokenizer.decode_bytes(&tokens)?;
let text: String = tokenizer.decode_tokens(&tokens)?;
```

There is no bare `.encode()`/`.decode()` on `TiktokenTokenizer` itself with tiktoken-rs's exact signature — the three `encode_*` methods above are it. `TiktokenTokenizer` also implements this crate's shared `Tokenizer` trait, so it interoperates with anything written against that trait:

```rust
use trustformers_core::traits::Tokenizer; // the trait lives in trustformers-core, not trustformers-tokenizers

let tokenized = tokenizer.encode("Hello, world!")?;   // -> TokenizedInput (input_ids: Vec<u32>, ...)
let text = tokenizer.decode(&tokenized.input_ids)?;    // takes &[u32], not &[usize]
```

Through the trait, `encode_pair(text, text2)` exists but is not a tiktoken concept — it joins the two strings with a space and encodes the result, since tiktoken encodings have no native sequence-pair convention.

### Other real methods

```rust
tokenizer.vocab_size();                 // ranked bytes + special tokens
tokenizer.special_tokens();             // &HashMap<String, usize>
tokenizer.encoder();                    // &RankMap (the raw rank table)
tokenizer.is_special_token(token_id);   // bool
```

### Errors

Every fallible call here returns this crate's ordinary `trustformers_core::errors::Result<T>` / `TrustformersError` — there is no separate `TiktokenError` enum, and no `InvalidTokenId { valid_range, suggestions, .. }` structured variant. `decode_bytes` fails with a plain message naming the unknown token id if one isn't in the decoder table; the pattern-compilation and rank-file-loading paths fail with equally specific messages.

## What does not exist in this crate

The previous version of this guide described all of the following as real TrustformeRS features. None of them exist today — do not write code depending on them:

- Batch helpers: `encode_batch`, `encode_batch_with_padding`, `encode_batch_parallel`, `encode_stream`, `PaddingConfig`/`PaddingStrategy`
- Caching configuration: `with_cache_strategy`, `CacheStrategy`, `with_memory_pool`, `MemoryPoolConfig`, `with_vocabulary_compression`, `with_custom_cache`/`CustomCacheConfig`
- Parallelism configuration: `with_parallel_processing`, `with_thread_pool_size`
- Chat/completion helpers: `encode_chat_completion`, `encode_chat_completion_with_functions`, `ChatMessage`, `ChatCompletionOptions`, `ChatFormat`
- Analytics/cost: `count_tokens`, `count_tokens_batch`, `.estimate_cost(model)`, `analyze_tokenization`
- Model-name helpers: `encoding_for_model`, `for_model`, `encode_for_model`, `ModelOptimizationConfig`, `TiktokenModelManager`, `auto_detect_model`
- Debug/validation: `with_debug_mode`, `ValidationLevel`, `encode_with_debug`, `validate_encoding`, `validate_round_trip`
- Compatibility/perf presets: `with_compatibility_mode`, `CompatibilityMode`, `with_special_token_mode`, `SpecialTokenMode`, `with_performance_preset`, `PerformancePreset`, `TokenizerProfiler`
- Testing/deployment scaffolding: `testing::TiktokenMigrationTester`, `benchmarking::TiktokenBenchmark`, `deployment::{ShadowTestConfig, ShadowTokenizer, RolloutConfig, HybridTokenizer}`, `monitoring::TiktokenProductionMetrics`
- A Python package exposing any of the above (`trustformers_tokenizers.TiktokenTokenizer` does not exist)

If your migration needs one of these, it's new work, not a rename of an existing TrustformeRS feature.

## Migration checklist

- [ ] Obtain the `.tiktoken` rank file(s) you need yourself — this crate will not fetch them
- [ ] Replace `tiktoken`/`tiktoken-rs` calls with the real methods above (`encode_ordinary`/`encode_text`/`encode_with_special_tokens`, `decode_bytes`/`decode_tokens`)
- [ ] If you relied on batching, caching configuration, chat templating, or cost estimation from a wrapper around `tiktoken`, keep that logic in your own code — it has no TrustformeRS equivalent to migrate onto
- [ ] Add `trustformers-tokenizers = "0.2"` to `Cargo.toml` (no separate Python package for tiktoken)
- [ ] Run your own equivalence/performance comparison against a real `tiktoken` install using `trustformers-tokenizers/benches/tokenizer_performance.rs` as a starting point — this guide does not supply numbers for you
