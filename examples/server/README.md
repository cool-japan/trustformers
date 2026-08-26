# TrustformeRS REST API Server

A REST API example server for TrustformeRS: classification, NER,
question-answering and text-generation over real checkpoints, built on Axum.

## What this server does and does not do

- **It loads real checkpoints, never fabricates predictions.** Every number
  in every response comes from an actual forward pass through the loaded
  checkpoint's real task head (softmax over real classifier-head logits, a
  real autoregressive decode for generation) — never a hardcoded or
  string-templated result.
- **It refuses to serve a randomly-initialised model.** `trustformers`'s
  task-specific loaders (`AutoModelForSequenceClassification`,
  `AutoModelForTokenClassification`, `AutoModelForQuestionAnswering`,
  `AutoModelForCausalLM`) silently keep an untrained, random-weight head when
  no `model.safetensors` is present at the given path — no error, no
  warning. `POST /models` checks for that file itself first and refuses the
  load with a clear error if it is missing, rather than letting the server
  quietly answer every request with meaningless scores.
- **It never downloads model weights.** `model_name` must be, or resolve to
  (see `MODEL_CACHE_DIR` below), a local directory containing a real
  checkpoint: `config.json`, `model.safetensors`, and that checkpoint's
  tokenizer files (`tokenizer.json`, or a WordPiece/BPE vocab —
  whatever `trustformers`'s `AutoTokenizer::from_pretrained` recognises).
  The **one** exception: if `config.json` is not found locally,
  `AutoConfig::from_pretrained` (a `trustformers` library function this
  server calls, not something written for this example) falls back to a real
  network request to the model hub for the config only — never for weights —
  and, failing that, to guessing the architecture from the directory name.
  This can add real network latency to a `POST /models` call; it is
  documented here because it surprises people, not because this server asked
  for it.
- **One loaded model serves exactly one task.** A BERT checkpoint could back
  classification, NER, or QA, and there is no way to tell which from the
  checkpoint alone — so `POST /models` takes an explicit `task` and builds
  the matching head (`BertForSequenceClassification`,
  `BertForTokenClassification`, `BertForQuestionAnswering`, or, for
  generation, `Gpt2LMHeadModel`/`GptNeoLMHeadModel`/`GptJLMHeadModel` — bert /
  roberta / albert cover the first three tasks, gpt2 / gpt_neo / gpt_j the
  last). Loading the same checkpoint directory twice for two different tasks
  is fine and gives you two independent model ids.
- **No character-level offsets.** The tokenizer stack behind `AutoTokenizer`
  does not produce a character offset mapping. NER responses report each
  token's own (possibly subword) text and its index in the tokenized
  sequence instead of reconstructed words and character spans; QA responses
  report `start_token`/`end_token` token indices, not character positions
  into `context`. Neither is a guess dressed up as a measurement — see
  `src/handlers.rs` for exactly what is and isn't computed.
- **No labels are invented.** There is no `id2label` read from the
  checkpoint config. Classification/NER labels come from what you pass to
  `POST /models`'s `labels` field, or default to `LABEL_0`, `LABEL_1`, ...
  when you don't supply them (or supply the wrong count) — never a
  hardcoded guess like `["NEGATIVE", "POSITIVE"]` presented as if it came
  from the model.
- **No `/metrics` endpoint, and no memory measurement.** This server does
  not integrate with Prometheus and does not report process/device memory
  anywhere — an earlier version of the Kubernetes manifests in this
  directory implied otherwise (scrape annotations, a custom-metrics HPA
  rule, a `ConfigMap` describing settings nothing reads); that drift is
  fixed — see `kubernetes/README.md`.

## API

### Health

```
GET /health
```
`{"status": "healthy", "version": "0.2.1"}`. Always available, does not
depend on any model being loaded.

### Model management

```
POST /models
{
  "model_name": "/path/to/checkpoint",   // or a name relative to MODEL_CACHE_DIR
  "task": "text-classification",          // | "token-classification" | "question-answering" | "text-generation"
  "num_labels": 2,                        // required for text-classification / token-classification
  "labels": ["negative", "positive"]      // optional; must match num_labels exactly or it's ignored
}
```
→ `{"model_id": "...", "message": "loaded `...` for task `...` in 1.234s"}`

```
GET  /models              -> [{"id", "model_name", "task", "num_parameters", "labels", "loaded_at", "load_duration_ms"}, ...]
GET  /models/{model_id}   -> the same object for one model
DELETE /models/{model_id} -> 204, or 404 if not loaded
```

### Inference

All four require a `model_id` from a model loaded for the matching task —
loading the wrong task returns a 400 naming both the task the model was
actually loaded for and the one the endpoint needs, not a confusing failure
three steps later.

```
POST /predict/classification  {"model_id", "text"}
  -> {"label", "score", "scores": [{"label", "score"}, ...]}

POST /predict/ner  {"model_id", "text"}
  -> {"entities": [{"token", "label", "score", "index"}, ...]}
     Entities are BIO-merged (consecutive "I-X" tokens joined into the
     preceding "B-X"/"I-X" entity, "O" tokens dropped) only when the loaded
     label set actually contains both an "O" label and a "B-"/"I-" prefixed
     one; otherwise every token's own prediction is reported unmerged,
     since nothing asserted a BIO scheme to merge by.

POST /predict/qa  {"model_id", "question", "context"}
  -> {"answer", "score", "start_token", "end_token"}

POST /predict/generation  {"model_id", "prompt", "max_length", "temperature", "top_k", "top_p"}
  -> {"generated_text", "prompt_tokens", "completion_tokens"}
     `max_length` is the absolute target sequence length (prompt + generated
     tokens), not a new-tokens-only budget. Only these four fields are
     accepted: `AutoModelForCausalLM::generate` (the library method this
     endpoint calls) always samples and never applies `do_sample` /
     `repetition_penalty` / `num_beams` / etc., so this server does not
     accept them either — accepting a field it would silently drop is
     exactly the kind of accepted-but-ignored request this server refuses to
     produce elsewhere.

POST /predict/batch  {"model_id", "inputs": [...]}
  -> {"results": [...], "total_time_ms"}
     Each item's shape depends on model_id's own task: {"text": ...} for
     classification/NER, {"question", "context"} for QA, {"prompt", ...}
     for generation. There is no batched kernel behind this — inputs run one
     after another — and one bad item fails the whole call rather than
     being written into the response as if it were a prediction.
```

See `client.py` for a runnable walkthrough of every endpoint (it needs real
local checkpoints — see its module docstring; it is not run as part of this
repository's automated checks).

## Environment variables

Every one of these is read for real by `src/main.rs` — grep `std::env::var`
there if in doubt.

| Variable | Default | Effect |
|---|---|---|
| `PORT` | `8080` | Listen port. |
| `RUST_LOG` | `trustformers_server=info,tower_http=info` | `tracing_subscriber::EnvFilter` syntax. |
| `MAX_MODELS` | `10` | `POST /models` returns 503 once this many are loaded. |
| `MODEL_CACHE_DIR` | unset | Base directory a *relative* `model_name` resolves against. An absolute `model_name` ignores this. |
| `PRELOAD_MODEL_NAME` | unset | Load one checkpoint at startup instead of requiring a `POST /models` call first. |
| `PRELOAD_MODEL_TASK` | — | Required alongside `PRELOAD_MODEL_NAME`; same values as `task` above. |
| `PRELOAD_NUM_LABELS` | unset | Passed through when the preload task needs `num_labels`. |

## Building and running

```bash
cd examples/server
cargo build --release
cargo run --release                 # listens on 0.0.0.0:8080 by default

# with a checkpoint directory root and a startup preload
MODEL_CACHE_DIR=/data/checkpoints \
PRELOAD_MODEL_NAME=my-sentiment-model \
PRELOAD_MODEL_TASK=text-classification \
PRELOAD_NUM_LABELS=2 \
cargo run --release
```

## Testing

```bash
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test
```

The test suite (29 tests as of this writing) is real logic, not smoke tests:
`ModelManager`'s honest-refusal behaviour (missing weight file, missing/zero
`num_labels`) is exercised against real temporary directories via
`std::env::temp_dir()`; `MODEL_CACHE_DIR` path resolution, label
placeholder-fallback, BIO entity merging, the numerically-stable softmax,
every `AppError` → HTTP status mapping, and the internal-invariant guard that
replaced this crate's last production `.expect()` call (`validated_num_labels`
— see `src/models.rs`) each have dedicated tests. What is *not* exercised
end-to-end is a real forward pass through a real trained checkpoint over HTTP
— that needs an actual `model.safetensors` this environment does not have one
of and cannot download (see above); every step short of that (routing,
validation, the honest-refusal paths, response shapes) is covered.

No `.unwrap()`/`.expect()`/`panic!()`/`unreachable!()` exists in this crate's
non-test code (grep `src/*.rs` yourself if in doubt — every `#[cfg(test)]`
module starts well after any such call in its file). A few call sites remain
that are unreachable by construction today (e.g. `require_task` having
confirmed a model's task before a match on `task_data`'s own variant) — each
now returns a structured `AppError` instead of relying on that invariant
holding forever, so a future change that ever broke it would fail one request
with a diagnosable 500 instead of panicking the request task.

## Docker

```bash
cd examples/server
docker build -t trustformers-server -f Dockerfile ../..
docker run -p 8080:8080 -v /path/to/checkpoints:/var/cache/trustformers trustformers-server
# or:
docker compose up
```

**Not verified in this environment**: no Docker daemon was reachable here
(`docker info` fails), so the image above was never actually built or run —
only written, and reasoned about against `examples/grpc-server/Dockerfile`,
the equivalent file for this repository's other example server, which *was*
built and run successfully in its own environment. `docker-compose.yml`'s
`services.trustformers-server.volumes` entry expects a `./models` directory
next to it (create one, or point it elsewhere) with one subdirectory per
checkpoint you want to load by name.

## Kubernetes

See `kubernetes/README.md` for the full guide. Summary of what changed in
this pass and why: `kubernetes/README.md`'s own "Recently corrected" section
has the details; the short version is that several manifests referenced
capabilities this server does not have (a `/metrics` endpoint, a config file
it never reads, Kubernetes API permissions it never uses) and a
`kustomization.yaml` that could not actually be built by `kubectl kustomize`
at all. `kubectl kustomize kubernetes/` now succeeds — verified in this
environment; **applying it to a real cluster was not verified** (no cluster
was available here).

## What was verified, and what was not

Verified in this environment, with the literal commands in parentheses:
- The crate compiles and lints clean (`cargo check --all-targets`,
  `cargo clippy --all-targets -- -D warnings`).
- All 29 tests pass (`cargo test`).
- `kubectl kustomize kubernetes/` succeeds; it failed before this pass's
  fixes (reproduced against the pre-fix files directly with `kubectl
  kustomize`, not inferred).
- `client.py` parses (`python3 -m py_compile client.py`).

Not verified in this environment (no Docker daemon, no Kubernetes cluster,
and no real trained checkpoint were available here):
- Building or running the Docker image.
- Applying the Kubernetes manifests to a real cluster.
- An actual inference request against a real trained checkpoint (everything
  short of "does a real checkpoint produce a real answer" — routing,
  request validation, the honest weight-file/task-mismatch refusals, response
  shapes — is covered by the test suite instead; see "Testing" above).

## License

Apache-2.0
