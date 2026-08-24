# TrustformeRS Python

Python bindings (PyO3) for TrustformeRS, a transformer library written in Rust. This page describes only what the compiled `trustformers` Python extension actually exposes today — verified 2026-08-24 by reading `trustformers-py/src/` directly, not by describing the aspirational HF-Transformers-compatible surface the crate is working toward.

> **Honesty note**: the previous version of this file (dated 2026-03-21, five months stale as of this rewrite) described a fabricated benchmark table, PyTorch interop methods that don't exist, and a `pipeline()` task list that included three tasks this crate has never implemented. Those are all corrected below.

## What's real here

- Real PyO3 bindings around this workspace's Rust models — no reimplementation, no simulation layer.
- `AutoModel`/`AutoTokenizer`/`pipeline()` load from a **local** checkpoint path. **This crate has no Hugging Face Hub downloader.** Passing a bare name like `"bert-base-uncased"` only resolves if a matching local checkpoint already exists at the path this crate probes — it will not fetch anything from the network. Fetch checkpoints yourself first.
- `.numpy()` / `.numpy_view()` NumPy interoperability. There is **no** `Tensor.from_torch()` / `.to_torch()` — PyTorch interop does not exist in this crate despite an earlier version of this page showing it.
- Six model families get a real, concrete Python class today: **BERT, GPT-2, T5, LLaMA, RWKV, Mamba** (plus `BertForSequenceClassification` and `GPT2LMHeadModel` for their respective task heads). The underlying Rust crate supports 49+ architectures, but most of them have no PyO3 binding yet — if you need one that's missing here, use the Rust API directly (`trustformers-models`) instead.
- Four pipeline tasks: `text-generation` and `text-classification`/`sentiment-analysis` run real inference. `token-classification`/`ner` and `question-answering` are registered but **always refuse to construct**, raising a `NotImplementedError` that names the real reason (this crate's tokenizers don't yet produce a character-level offset mapping HuggingFace's `start`/`end` keys would need). `fill-mask`, `summarization`, and `translation` are **not implemented at all** — `pipeline()` raises `ValueError: Unknown task` for any of them.
- `Trainer`/`TrainingArguments` exist as real, constructible Python classes, but **`Trainer.train()` is still a fabricated stub** — it always returns `{"train_loss": 0.5, "epoch": <your config>, "total_steps": 1000}` regardless of your data, model, or configuration. Do not use these numbers for anything; no training actually runs yet.

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
from trustformers import pipeline

# Loads BertForSequenceClassification from a *local* checkpoint path — see
# the Hub-downloader note above. Point `model=`/`tokenizer=` at your own
# local path if the default resolution doesn't find one.
classifier = pipeline("sentiment-analysis")
result = classifier("I love writing Rust code!")

generator = pipeline("text-generation")
result = generator("Once upon a time")

# Both of these always raise NotImplementedError on construction — see above.
# ner = pipeline("token-classification")
# qa = pipeline("question-answering")
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

Models: `BertModel`, `BertForSequenceClassification`, `GPT2Model`, `GPT2LMHeadModel`, `T5Model`, `LlamaModel`, `RwkvModel`, `MambaModel` (all extend `PreTrainedModel`).

Tokenizers: `WordPieceTokenizer`, `BPETokenizer` (both extend `PreTrainedTokenizer`).

Auto classes: `AutoModel`, `AutoTokenizer`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`, `AutoModelForQuestionAnswering`, `AutoModelForCausalLM`, `AutoModelForMaskedLM`.

Pipelines: `Pipeline` (base), `TextGenerationPipeline`, `TextClassificationPipeline`, `TokenClassificationPipeline` (refuses, see above), `QuestionAnsweringPipeline` (refuses, see above).

Tensors: `Tensor`, `TensorOptimized`, `AdvancedActivations`.

Training: `Trainer`, `TrainingArguments` (see the honesty note above — `.train()` does not run real training yet).

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

### Architecture

The library is organized into several components:

- `tensor.rs` — Tensor operations and NumPy integration
- `models/` — Model implementations (BERT, GPT-2, T5, LLaMA, RWKV, Mamba) and `models/losses.rs` (real classification/LM cross-entropy loss)
- `tokenizers.rs` — Tokenizer implementations
- `pipelines/` — Pipeline API (`mod.rs` + `scoring.rs`; not a single `pipelines.rs` file as of this wave's restructuring)
- `auto.rs` — Auto classes for model/tokenizer loading, and the `pipeline()` factory function
- `training.rs` — Training utilities (`Trainer`/`TrainingArguments` — real classes, fabricated `.train()`, see above)

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
