# TrustformeRS gRPC Server

A gRPC front end for TrustformeRS checkpoints: single, streaming and batch text
generation, plus model load/unload and introspection.

This example is **excluded from the repository's cargo workspace** and carries
its own `[workspace]` table, so it is built from inside this directory.

## What it actually does

- **Text generation** — `Predict`, `StreamPredict` and `BatchPredict` all run a
  real forward pass through `trustformers::AutoModel`. `StreamPredict` decodes
  incrementally, so the first token reaches the client after one forward pass
  rather than after the whole run.
- **Model management** — checkpoints are loaded and unloaded on demand, up to
  `MAX_MODELS` at a time.
- **Introspection** — `ListModels` / `GetModelInfo` report the loaded
  checkpoint's own architecture, parameter count and layer shapes, read from its
  configuration. Nothing here is a hardcoded placeholder.
- **Health checking** — the standard `grpc.health.v1.Health` service.
- **Reflection** — both `v1` and `v1alpha` server reflection, so `grpcurl` and
  `grpcui` work without a local copy of the `.proto`.
- **Per-response metrics** — `PredictMetrics` / `BatchMetrics` carry measured
  latency and token throughput on every response.

### What it deliberately does not do

These are refused with a `UNIMPLEMENTED` status and a message explaining why,
rather than accepted and silently ignored:

| request | why it is refused |
|---|---|
| `device` other than `cpu` | `AutoModel` has no device-placement parameter |
| `use_fp16` | no dtype conversion is performed at load or at inference |
| `compile` | there is no graph-compilation backend |
| `PredictOptions.seed` | the sampler's RNG cannot be pinned, so reproducibility cannot be promised |
| `PredictOptions.stop_sequences` | generation stops on the model's end-of-sequence token or the token budget; there is no string stop criterion |
| `PredictOptions.extra_params` | it would have to be ignored |
| image input | only text checkpoints are served |

`PredictMetrics.memory_used_bytes` and `ModelStatus.memory_used_bytes` stay at
their proto default of `0`: this server measures no process or device memory,
and a parameter-count-derived estimate would be an invented number.

### Model requirements

The generation RPCs need a checkpoint with a language-modelling head — GPT-2,
GPT-Neo, GPT-J or T5. Encoder-only checkpoints such as `bert-base-uncased` load
successfully (and `GetModelInfo` describes them) but generation returns a
`FeatureUnavailable`-derived error rather than inventing text.

## Building

```bash
cd examples/grpc-server
cargo build --release
```

`protoc` must be on `PATH`: `prost-build` shells out to it to parse
`proto/inference.proto`.

## Running

```bash
# Defaults: listen on [::]/0.0.0.0:50051, up to 10 models, cpu
cargo run --release

RUST_LOG=info,trustformers_grpc_server=debug cargo run --release
```

## Configuration

All configuration is read from the environment by `src/main.rs`. A variable that
is set but unparseable is a startup error rather than a silent fallback.

| variable | default | meaning |
|---|---|---|
| `GRPC_ADDR` | unset | full listen address (`host:port`); overrides `GRPC_PORT` |
| `GRPC_PORT` | `50051` | port to listen on, bound on all interfaces |
| `MAX_MODELS` | `10` | how many checkpoints may be loaded at once; `0` is a startup error |
| `DEFAULT_DEVICE` | `cpu` | device used when a request names none. Only `cpu` is available, and any other value is rejected at startup rather than turning every `LoadModel` into a failure |
| `RUST_LOG` | `info` | `tracing-subscriber` filter |

## API Overview

### InferenceService

- `Predict`: single generation request
- `StreamPredict`: bidirectional streaming generation
- `BatchPredict`: several prompts in one call
- `ListModels`: every loaded checkpoint
- `GetModelInfo`: one checkpoint's architecture, shape and status
- `LoadModel`: load a checkpoint into memory
- `UnloadModel`: drop a checkpoint

## Client Examples

`client.py` in this directory is a runnable version of the snippets below.

### Python Client

```python
import grpc
from google.protobuf import empty_pb2
from inference_pb2 import PredictRequest, LoadModelRequest, PredictOptions
from inference_pb2_grpc import InferenceServiceStub

channel = grpc.insecure_channel('localhost:50051')
stub = InferenceServiceStub(channel)

# Load a generative checkpoint
stub.LoadModel(LoadModelRequest(
    model_id="gpt2",
    model_path="/models/gpt2",
    device="cpu",
))

response = stub.Predict(PredictRequest(
    model_id="gpt2",
    text="The capital of France is",
    options=PredictOptions(max_new_tokens=16, do_sample=True, use_cache=True),
))
print(response.text_output.text)
print(f"{response.metrics.latency_ms:.1f} ms, "
      f"{response.metrics.tokens_per_second} tok/s")
```

Note that `do_sample` and `use_cache` are plain proto3 `bool`s: they have no
"unset" state on the wire, so an `options` message that omits them sends
`false`. Send no `options` message at all to get the library defaults.

### Using grpcurl

```bash
grpcurl -plaintext localhost:50051 list

grpcurl -plaintext localhost:50051 describe trustformers.inference.InferenceService

grpcurl -plaintext localhost:50051 trustformers.inference.InferenceService/ListModels

grpcurl -plaintext -d '{
  "model_id": "gpt2",
  "model_path": "/models/gpt2",
  "device": "cpu"
}' localhost:50051 trustformers.inference.InferenceService/LoadModel

grpcurl -plaintext -d '{
  "model_id": "gpt2",
  "text": "The capital of France is",
  "options": {"max_new_tokens": 16}
}' localhost:50051 trustformers.inference.InferenceService/Predict

# Health check
grpcurl -plaintext localhost:50051 grpc.health.v1.Health/Check
```

## Streaming

`StreamPredict` is bidirectional. A `start` opens a session and generates from
`initial_text`; each later `continue` generates from the session transcript plus
the new text, so the conversation builds on what came before. Every turn emits
one response per decoded token (`is_final = false`), followed by exactly one
summary response with `is_final = true` that carries the turn's metrics. A
`continue` with `end_stream` set also closes the session.

Sessions live for exactly one call, so a client that disconnects cannot leave
one behind.

```python
def request_generator():
    yield StreamPredictRequest(
        start=StreamStartRequest(
            model_id="gpt2",
            initial_text="Once upon a time",
            options=PredictOptions(max_new_tokens=50, temperature=0.8,
                                   do_sample=True, use_cache=True),
        )
    )

for response in stub.StreamPredict(request_generator()):
    print(response.text, end="", flush=True)
    if response.is_final:
        print(f"\n{response.metrics.latency_ms:.1f} ms")
        break
```

## Batch Processing

`BatchPredict` generates for each prompt in turn. There is no batched kernel
behind it yet, so it is a convenience wrapper rather than a throughput
optimisation; a failure on any prompt fails the whole call instead of being
written into the response as if it were a prediction.

```python
batch_response = stub.BatchPredict(BatchPredictRequest(
    model_id="gpt2",
    texts=["The weather today is", "Machine learning is", "Python is a"],
    options=PredictOptions(max_new_tokens=16),
))
for pred in batch_response.predictions:
    print(pred.text_output.text)
print(f"{batch_response.metrics.total_tokens} tokens in "
      f"{batch_response.metrics.total_latency_ms:.1f} ms")
```

## Docker

`Dockerfile` and `docker-compose.yml` in this directory build and run the
server. The build context is the repository root, because the binary is built
from the full source tree:

```bash
# from examples/grpc-server
docker compose up --build

# with the interactive gRPC browser on :8080
docker compose --profile debug up --build
```

Mount checkpoints at `/models` and pass that path as
`LoadModelRequest.model_path`.

These files are not exercised by CI; they are provided as a starting point.

## Observability

There is no HTTP `/metrics` endpoint. Timing and throughput are returned inline
on `PredictMetrics` (per response) and `BatchMetrics` (per batch), and
`GetModelInfo` reports each model's load duration, uptime and served-request
count. Logging goes through `tracing`; set `RUST_LOG` to control it.

## Security

This example serves **plaintext gRPC with no authentication**. `tonic`'s TLS
features are deliberately not enabled, and no authentication or rate-limiting
interceptor is installed. Put it behind a proxy that terminates TLS and
authenticates callers, or add `tonic`'s `tls-ring` feature and a
`tonic::service::Interceptor`, before exposing it beyond a trusted network.

Request payloads are validated: sampling parameters are range-checked and
options the server cannot honour are rejected rather than ignored (see the table
above).

## Troubleshooting

- **`protoc` not found at build time** — install the protobuf compiler.
- **"model architecture ... has no language-modelling head"** — the checkpoint
  is encoder-only; load a GPT-2/GPT-Neo/GPT-J/T5 checkpoint for generation.
- **"no weight file found"** — `AutoModel::from_pretrained` refuses to return a
  randomly initialised network, so the path must contain real weights.
- **Connection refused from another container/host** — check `GRPC_ADDR`; the
  default binds all interfaces, but an explicit `GRPC_ADDR=127.0.0.1:50051`
  is loopback-only.

Interactive debugging:

```bash
grpcui -plaintext localhost:50051
```
