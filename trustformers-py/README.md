# TrustformeRS Python

Python bindings (PyO3) for TrustformeRS, a transformer library written in Rust. This page describes only what the compiled `trustformers` Python extension actually exposes today — verified 2026-08-24 by reading `trustformers-py/src/` directly, not by describing the aspirational HF-Transformers-compatible surface the crate is working toward.

> **Honesty note**: the previous version of this file (dated 2026-03-21, five months stale as of this rewrite) described a fabricated benchmark table, PyTorch interop methods that don't exist, and a `pipeline()` task list that included three tasks this crate has never implemented. Those are all corrected below.

## What's real here

- Real PyO3 bindings around this workspace's Rust models — no reimplementation, no simulation layer.
- `AutoModel`/`AutoTokenizer`/`pipeline()` load from a **local** checkpoint path. **This crate has no Hugging Face Hub downloader.** Passing a bare name like `"bert-base-uncased"` only resolves if a matching local checkpoint already exists at the path this crate probes — it will not fetch anything from the network. Fetch checkpoints yourself first.
- `.numpy()` / `.numpy_view()` NumPy interoperability. There is **no** `Tensor.from_torch()` / `.to_torch()` — PyTorch interop does not exist in this crate despite an earlier version of this page showing it.
- Eight model families get a real, concrete Python class today: **BERT, GPT-2, T5, LLaMA, RWKV, Mamba** (plus `BertForSequenceClassification`, `BertForTokenClassification`, `BertForQuestionAnswering`, and `GPT2LMHeadModel` for their respective task heads). The underlying Rust crate supports 49+ architectures, but most of them have no PyO3 binding yet — if you need one that's missing here, use the Rust API directly (`trustformers-models`) instead.
- Four pipeline tasks run real inference end to end, verified 2026-08-24: `text-generation`, `text-classification`/`sentiment-analysis`, `token-classification`/`ner`, and `question-answering`. The two span pipelines used to always refuse construction (this crate's tokenizers set `offset_mapping` to `None` unconditionally); now that `WordPieceTokenizer`/`BPETokenizer` populate a real one, `TokenClassificationPipeline` wraps a real `BertForTokenClassification` forward pass with `aggregation_strategy='simple'` aggregation, and `QuestionAnsweringPipeline` wraps a real `BertForQuestionAnswering` forward pass with real joint-argmax extraction. Both report HuggingFace's `start`/`end` keys as Unicode **character** offsets — converted, once, from this crate's native **byte** offsets at the pipeline's Python boundary (`trustformers_tokenizers::byte_offsets_to_char_offsets`); nothing converts silently. `QuestionAnsweringPipeline` additionally requires a `WordPieceTokenizer`: `BPETokenizer.encode_pair` joins question+context into one string with no separator token, so there is no reliable per-sequence context boundary to find. `BertForTokenClassification`/`BertForQuestionAnswering` remain callable directly for raw per-token/per-position logits without a pipeline. `fill-mask`, `summarization`, and `translation` are **not implemented at all** — `pipeline()` raises `ValueError: Unknown task` for any of them.
- `AutoModelForSequenceClassification`/`ForTokenClassification`/`ForQuestionAnswering` now construct the matching real task-head class above for BERT-family checkpoints (bert/roberta/distilbert/deberta), and raise `NotImplementedError` for anything else — verified 2026-08-24. They used to silently delegate to `AutoModel`, handing back a bare, headless encoder relabelled as a task model for *any* checkpoint name, non-BERT ones included.
- `batch_encode_plus`'s (`WordPieceTokenizer`) and `encode(text, text_pair=...)`'s (`BPETokenizer`) sequence-pair handling now produces a real pair encoding instead of concatenating two independent single-sequence encodings — verified 2026-08-24. WordPiece: BERT's `[CLS] A [SEP] B [SEP]` with 0/1 `token_type_ids`. BPE: the RoBERTa/GPT-2-family `<bos> A <eos> <eos> B <eos>` convention, using the tokenizer's own configured `bos_token`/`eos_token`, with all-zero `token_type_ids` (BPE-family pairs use no segment ids).
- `Trainer`/`TrainingArguments` exist as real, constructible Python classes, but **`train()`/`evaluate()`/`predict()`/`push_to_hub()` all raise `NotImplementedError`** — verified 2026-08-24, replacing four separate fabrications (`train()` used to always return `{"train_loss": 0.5, "epoch": <your config>, "total_steps": 1000}`; `evaluate()` always returned `{"eval_loss": 0.45, "eval_accuracy": 0.92, "eval_samples": 100}`; `predict()` always returned fixed `predictions`/`label_ids` arrays; `push_to_hub()` always returned a `https://huggingface.co/...` URL without uploading anything). `train()`'s refusal names its exact cause: this crate has real forward passes and real loss functions, but no backpropagation/optimizer-step path from a loss back to a model's own parameters exists anywhere in this workspace (checked directly: zero `impl ParameterAccess` sites in `trustformers-core`/`trustformers-models`/`trustformers-training`, and `trustformers-core`'s separate autodiff `Variable`/`ComputationGraph` system is never constructed by any model's `forward()`). `save_model()` is the one real `Trainer` method: it delegates to the wrapped model's own real `save_pretrained()`.

## Installation

```bash
pip install trustformers
```

### From source

```bash
# Install maturin (build tool for Rust Python extensions)
pip install maturin

# Clone the repository
git clone https://github.com/cool-japan/trustformers
cd trustformers/trustformers-py

# Build and install
maturin develop --release
```

## Quick Start

### Pipelines

```python
from trustformers import pipeline, WordPieceTokenizer, BertForTokenClassification, BertForQuestionAnswering

# Loads BertForSequenceClassification from a *local* checkpoint path — see
# the Hub-downloader note above. Point `model=`/`tokenizer=` at your own
# local path if the default resolution doesn't find one.
classifier = pipeline("sentiment-analysis")
result = classifier("I love writing Rust code!")

generator = pipeline("text-generation")
result = generator("Once upon a time")

# NER: real per-token forward pass, aggregation_strategy='simple' merging.
# `tokenizer` may be a WordPieceTokenizer or a BPETokenizer.
tagger = BertForTokenClassification.from_pretrained(model_path, num_labels=9)
tokenizer = WordPieceTokenizer.from_pretrained(model_path)
ner = pipeline("token-classification", model=tagger, tokenizer=tokenizer)
entities = ner("My name is Sarah and I live in London")
# [{"entity_group": ..., "score": ..., "word": ..., "start": ..., "end": ...}, ...]
# start/end are Unicode character offsets, so text[start:end] is correct.

# QA: real joint-argmax span extraction over the context. `tokenizer` must be
# a WordPieceTokenizer specifically (see the note above).
answerer = BertForQuestionAnswering.from_pretrained(model_path)
qa = pipeline("question-answering", model=answerer, tokenizer=tokenizer)
result = qa(question="Where do I live?", context="My name is Sarah and I live in London")
# {"score": ..., "start": ..., "end": ..., "answer": "London"}

# Real per-token / per-position inference without the pipeline wrapper (raw
# logits, no aggregation, no offset conversion):
outputs = tagger(input_ids, attention_mask)   # outputs["logits"]: [1, seq_len, 9]
outputs = answerer(input_ids, attention_mask)  # outputs["start_logits"], outputs["end_logits"]
```

### AutoModel / AutoTokenizer

```python
from trustformers import AutoModel, AutoTokenizer

# `model_path` must resolve to a real local checkpoint directory/file this
# crate can find — there is no Hub download behind this call.
model = AutoModel.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

### Direct model usage

```python
import numpy as np
from trustformers import BertModel, Tensor

model = BertModel.from_pretrained(model_path)

input_ids = Tensor(np.array([[101, 2023, 2003, 1037, 2742, 102]]))
attention_mask = Tensor(np.ones((1, 6)))

outputs = model(input_ids, attention_mask)
```

### NumPy interoperability

```python
import numpy as np
from trustformers import Tensor

np_array = np.random.randn(2, 3, 4).astype(np.float32)
tensor = Tensor(np_array)

back_to_numpy = tensor.numpy()        # copy
view = tensor.numpy_view()            # borrowed view

result = tensor.matmul(tensor.transpose())
```

## Supported Python classes

Models: `BertModel`, `BertForSequenceClassification`, `BertForTokenClassification`, `BertForQuestionAnswering`, `GPT2Model`, `GPT2LMHeadModel`, `T5Model`, `LlamaModel`, `RwkvModel`, `MambaModel` (all extend `PreTrainedModel`).

Tokenizers: `WordPieceTokenizer`, `BPETokenizer` (both extend `PreTrainedTokenizer`; both support a real `text_pair` sequence-pair encoding, see above).

Auto classes: `AutoModel`, `AutoTokenizer`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`, `AutoModelForQuestionAnswering` (these three: real task-head classes for BERT-family checkpoints, `NotImplementedError` otherwise — see above), `AutoModelForCausalLM`, `AutoModelForMaskedLM`.

Pipelines: `Pipeline` (base), `TextGenerationPipeline`, `TextClassificationPipeline`, `TokenClassificationPipeline`, `QuestionAnsweringPipeline` (requires a `WordPieceTokenizer` — see above). All four run real inference.

Tensors: `Tensor`, `TensorOptimized`, `AdvancedActivations`.

Training: `Trainer`, `TrainingArguments` (see the honesty note above — `train()`/`evaluate()`/`predict()`/`push_to_hub()` all raise `NotImplementedError`; `save_model()` is real).

Utility functions: `get_device()`, `set_seed()`, `enable_grad()`, `no_grad()`.

## API Compatibility

Imports are HuggingFace-`transformers`-shaped by design:

```python
# Hugging Face Transformers
from transformers import AutoModel, AutoTokenizer

# TrustformeRS (same import shape — behavior differs on the points above,
# especially checkpoint resolution: local paths only, no Hub download)
from trustformers import AutoModel, AutoTokenizer
```

This is a naming convenience, not a compatibility guarantee — expect to adjust checkpoint paths and any pipeline task you rely on beyond the four listed above.

## Performance

No benchmark harness in this repository has produced a number comparing this crate to Python HuggingFace Transformers — an earlier version of this page carried a specific-looking table (`52ms`/`3.2ms`/`16.3x` and similar figures, attributed to "Apple M1 Pro") that no automated benchmark here ever measured. It has been removed rather than re-guessed. `trustformers-py`'s own Rust-side benchmarks, and the workspace-level `benches/`, are the place to produce a real number if you need one.

## Advanced: Custom Models

```python
from trustformers import PreTrainedModel, Tensor
import numpy as np

class CustomModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # Define your model architecture

    def forward(self, input_ids, attention_mask=None):
        # Implement forward pass
        pass
```

`PreTrainedModel` is registered as a Python-subclassable base class (`#[pyclass(subclass)]`); this pattern is untested by this documentation pass beyond confirming the class itself is subclassable — verify your own override actually gets called before relying on it.

## Development

### Building from source

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .
isort .

# Lint
ruff check .
```

### Running the Rust test suite

`pytest` above exercises the built Python package; this crate's own `cargo
test` suite (pure-Rust unit tests, no Python interpreter needed to *run* them)
needs `extension-module` — on by default, so this crate builds as a real
Python extension module — turned **off**, because a `cdylib`-only extension
module doesn't link a plain test binary against libpython:

```bash
cd trustformers-py
cargo check --all-targets
cargo clippy --all-targets -- -D warnings
cargo test --lib --no-default-features --features python-gc
```

A bare `cargo test` (default features) is expected to fail to link with
undefined Python C-API symbols — that's `extension-module` doing its job
correctly for a real build, not a bug.

### Architecture

The library is organized into several components:

- `tensor.rs` — Tensor operations and NumPy integration
- `models/` — Model implementations, one file per architecture (`bert.rs`, `gpt2.rs`, `t5.rs`, `llama.rs`, `rwkv.rs`, `mamba.rs`; `base.rs` for the shared `PreTrainedModel` base class, `tasks.rs` for the BERT task heads `BertForSequenceClassification`/`BertForTokenClassification`/`BertForQuestionAnswering`, `helpers.rs` for cross-architecture PyO3-boundary helpers — split from a single 2096-line `mod.rs` on 2026-08-24 to keep every file under this repository's 2000-line policy; every class keeps its original `trustformers.models::PyXxx` path via `pub use` re-exports) and `models/losses.rs` (real classification/LM/question-answering cross-entropy loss)
- `tokenizers.rs` — Tokenizer implementations, including real WordPiece/BPE sequence-pair encoding (see above)
- `pipelines/` — Pipeline API (`mod.rs` for the four `#[pyclass]` pipelines and the `pipeline()` task routing; `scoring.rs` for the real text-classification softmax; `span.rs` for the real NER aggregation / QA extraction math both span pipelines are built on, including the byte-offset extraction itself — the byte->character conversion for Python happens in `mod.rs`; not a single `pipelines.rs` file as of a previous wave's restructuring)
- `auto.rs` — Auto classes for model/tokenizer loading, and the `pipeline()` factory function
- `training.rs` — Training utilities (`Trainer`/`TrainingArguments` — real classes; `train`/`evaluate`/`predict`/`push_to_hub` all honestly refuse, see above; `save_model` is real)

## License

Apache License 2.0

## Contributing

Contributions are welcome! Please read our [Contributing Guide](../CONTRIBUTING.md) for details.

## Citation

If you use TrustformeRS in your research, please cite:

```bibtex
@software{trustformers,
  title = {TrustformeRS: A Rust Implementation of Transformers},
  author = {{COOLJAPAN OU (Team KitaSan)}},
  year = {2025--2026},
  url = {https://github.com/cool-japan/trustformers}
}
```
