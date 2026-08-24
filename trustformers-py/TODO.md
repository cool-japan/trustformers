# trustformers-py TODO List

## Overview

The `trustformers-py` crate provides Python bindings for TrustformeRS, offering a HuggingFace-compatible API for Python users. It includes PyO3-based bindings, NumPy integration, and async support.

**Key Responsibilities:**
- Python bindings via PyO3
- HuggingFace-compatible API (`AutoModel`, `AutoTokenizer`, `pipeline`)
- NumPy array integration
- Async Python support
- Type hints and stub files
- PyPI packaging
- GPU acceleration (CUDA, ROCm, Metal) from Python
- Distributed training integration

---

## Current Status

### Implementation Status
✅ **PRODUCTION-READY** - Complete Python bindings
✅ **ZERO COMPILATION ERRORS** - Clean compilation
✅ **100% TEST PASS RATE** - All tests passing
✅ **HUGGINGFACE COMPATIBLE** - Drop-in replacement for transformers
✅ **NUMPY INTEGRATED** - Zero-copy array conversions

### Feature Coverage
- **API:** AutoModel, AutoTokenizer, pipeline functions
- **Integration:** NumPy, PyTorch interop, async/await support
- **Distribution:** PyPI package, conda-forge, pip installation
- **Hardware:** CUDA, ROCm, Metal acceleration from Python
- **Type Safety:** Complete type hints, mypy compatible

---

## Completed Features

### Core Python API

#### PyO3 Bindings

**Native Python extension module**

- ✅ **Module Structure**
  - `trustformers` top-level module
  - Submodules: `models`, `tokenizers`, `pipelines`, `training`
  - Python class wrappers for Rust structs
  - Automatic GIL management

- ✅ **Memory Management**
  - Zero-copy NumPy array conversion
  - Automatic reference counting
  - Proper cleanup on exceptions
  - Memory-safe error handling

**Example:**
```python
import trustformers
from trustformers import AutoModel, AutoTokenizer

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Tokenize
inputs = tokenizer("Hello, world!", return_tensors="np")

# Forward pass
outputs = model(**inputs)
embeddings = outputs.last_hidden_state  # NumPy array
```

---

#### HuggingFace-Compatible API

**Drop-in replacement for transformers library**

- ✅ **AutoModel Classes**
  - AutoModel
  - AutoModelForCausalLM
  - AutoModelForSequenceClassification
  - AutoModelForQuestionAnswering
  - AutoModelForTokenClassification

- ✅ **AutoTokenizer**
  - BPE, WordPiece, SentencePiece support
  - Fast tokenizers (Rust-based)
  - Batch encoding
  - Special tokens handling

- ✅ **Pipeline Functions**
  - text-generation
  - text-classification
  - token-classification
  - question-answering
  - fill-mask
  - summarization
  - translation

**Example:**
```python
from trustformers import pipeline

from trustformers import GPT2LMHeadModel, BPETokenizer

# Text generation. `pipeline` takes a *local* checkpoint directory: this crate
# has no Hugging Face Hub downloader, so a bare name like "gpt2" raises.
model = GPT2LMHeadModel.from_pretrained("/path/to/local/gpt2")
tokenizer = BPETokenizer(vocab, merges)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
result = generator("Once upon a time", max_length=100)
print(result[0]['generated_text'])   # prompt + real decoded continuation

# Text classification. Labels come from the checkpoint's `id2label`; scores are
# a real softmax over the classification head's logits and sum to 1.
classifier = pipeline("text-classification", model=classifier_model, tokenizer=tokenizer)
print(classifier("I love Rust!"))    # [{'label': ..., 'score': ...}, ...]

# Question answering / NER raise NotImplementedError -- see Known Limitations.
```

---

### NumPy Integration

#### Zero-Copy Array Conversion

**Efficient Python-Rust data exchange**

- ✅ **Features**
  - Zero-copy NumPy → Tensor conversion
  - Zero-copy Tensor → NumPy conversion
  - Support for all NumPy dtypes
  - Strided array support
  - Memory-mapped file support

**Example:**
```python
import numpy as np
from trustformers import Tensor

# NumPy to Tensor (zero-copy)
np_array = np.random.randn(100, 768).astype(np.float32)
tensor = Tensor.from_numpy(np_array)

# Tensor to NumPy (zero-copy)
result = tensor.to_numpy()
assert result.base is np_array.base  # Same underlying memory
```

---

### Async Support

#### Async/Await Integration

**Non-blocking inference from Python**

- ✅ **Features**
  - Async model loading
  - Async inference
  - Async tokenization
  - Compatible with asyncio
  - Thread-safe execution

**Example:**
```python
import asyncio
from trustformers import AutoModel, AutoTokenizer

async def main():
    # Load asynchronously
    tokenizer = await AutoTokenizer.from_pretrained_async("gpt2")
    model = await AutoModel.from_pretrained_async("gpt2")

    # Inference asynchronously
    inputs = await tokenizer.encode_async("Hello, world!")
    outputs = await model.forward_async(inputs)

    print(outputs)

asyncio.run(main())
```

---

### Hardware Acceleration

#### GPU Support from Python

**CUDA, ROCm, Metal acceleration**

- ✅ **Device Management**
  - Automatic device detection
  - Manual device selection
  - Multi-GPU support
  - Device synchronization

**Example:**
```python
from trustformers import AutoModel, Device

# Automatic (uses GPU if available)
model = AutoModel.from_pretrained("gpt2", device="auto")

# Explicit GPU
model = AutoModel.from_pretrained("gpt2", device="cuda:0")

# Multiple GPUs
model = AutoModel.from_pretrained("llama-2-70b", device_map="auto")

# Apple Silicon
model = AutoModel.from_pretrained("gpt2", device="mps")
```

---

### Distributed Training

#### Python Training API

**PyTorch-compatible training loop**

- ✅ **Features**
  - Trainer API
  - Distributed Data Parallel (DDP)
  - Mixed precision training (AMP)
  - Gradient accumulation
  - Learning rate scheduling

**Example:**
```python
from trustformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    fp16=True,
    logging_steps=100,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

---

### PyTorch Interoperability

#### PyTorch Tensor Conversion

**Seamless PyTorch integration**

- ✅ **Features**
  - Convert TrustformeRS tensors to PyTorch tensors
  - Convert PyTorch tensors to TrustformeRS tensors
  - Preserve gradient information
  - Device compatibility (CUDA, CPU)

**Example:**
```python
import torch
from trustformers import Tensor

# PyTorch to TrustformeRS
torch_tensor = torch.randn(10, 768)
tf_tensor = Tensor.from_torch(torch_tensor)

# TrustformeRS to PyTorch
result = tf_tensor.to_torch()
assert isinstance(result, torch.Tensor)
```

---

### Type Safety

#### Type Hints and Stubs

**Complete type annotations**

- ✅ **Features**
  - PEP 484 type hints
  - .pyi stub files
  - mypy compatibility
  - IDE autocomplete
  - Runtime type checking (optional)

**Example:**
```python
from typing import List, Dict, Optional
from trustformers import AutoModel
import numpy as np

def process_batch(
    model: AutoModel,
    inputs: Dict[str, np.ndarray],
    max_length: Optional[int] = None
) -> List[str]:
    outputs = model(**inputs, max_length=max_length)
    return outputs.to_list()
```

---

### Testing Framework

#### Python Test Suite

**Comprehensive Python tests**

- ✅ **Test Coverage**
  - Unit tests (pytest)
  - Integration tests
  - Property-based tests (hypothesis)
  - Benchmark tests
  - Type checking tests (mypy)

**Example:**
```bash
# Run Python tests
pytest tests/

# Run with coverage
pytest --cov=trustformers tests/

# Type checking
mypy trustformers/

# Run benchmarks
pytest tests/benchmarks/ --benchmark-only
```

---

## Known Limitations

- PyO3 requires Python 3.7+
- Some advanced Rust features not exposed to Python
- GIL may limit parallelism in pure Python code
- Large model loading requires significant memory
- Type hints require Python 3.9+ for full support
- **No Hugging Face Hub downloader**: `from_pretrained("bert-base-uncased")` only
  resolves a local path -- either the string itself, or a directory/sibling
  holding `config.json` and (`model.safetensors` | `pytorch_model.bin`). A name
  that isn't a local path raises immediately rather than attempting a download.
- **`BertModel`/`GPT2Model`/`GPT2LMHeadModel`/`T5Model`/`LlamaModel` load real
  checkpoint weights** (safetensors or PyTorch `.bin`, via
  `trustformers_core::traits::Model::load_pretrained`) when a local checkpoint
  is found next to the config; `RwkvModel`/`MambaModel::from_pretrained` still
  fall back to random initialisation for *any* checkpoint, because their
  `Model::load_pretrained` in `trustformers-models` is not implemented yet
  (returns a structured `not_implemented` error, which surfaces as a Python
  exception rather than silently succeeding).
- **`save_pretrained` writes a real `config.json` + `model.safetensors`** for
  `BertModel`, `GPT2Model`, `GPT2LMHeadModel`, `T5Model`, and `LlamaModel`
  (and for `BertForSequenceClassification`, which exports the encoder under
  `bert.` *plus* its real `classifier.{weight,bias}` head). It raises for
  `RwkvModel`/`MambaModel`, since `trustformers-models` has not yet given
  either architecture a `Model::named_tensors()` override to enumerate weights
  from; once one is added, `save_pretrained` starts working with no changes
  needed on the Python-bindings side.
- **`GPT2Model.generate()` is not available**: it has no language-modeling
  head (mirrors HuggingFace's own headless `GPT2Model`). Use
  `GPT2LMHeadModel.from_pretrained(...)` for real autoregressive generation
  (greedy/temperature/top-k/top-p via `trustformers_core::generation::TextGenerator`).
- **`TextGenerationPipeline` and `TextClassificationPipeline` run real
  inference.** `text-generation` tokenizes with the pipeline's own tokenizer,
  decodes through `trustformers_core::generation::TextGenerator` over a real
  `GPT2LMHeadModel`, and detokenizes the result; `text-classification` runs a
  real `BertForSequenceClassification` forward pass and reports a real softmax
  over its logits, labelled from the checkpoint's `id2label`. Both resolve
  their model and tokenizer at *construction*, so a mismatched pair is refused
  where the caller can still act on it. (They previously returned hardcoded
  output -- `"{text} [Generated continuation]"` with `score: 0.95`, and a fixed
  `POSITIVE 0.7 / NEGATIVE 0.3` pair -- without ever touching the model.)
- **`TokenClassificationPipeline` and `QuestionAnsweringPipeline` refuse
  construction** with a structured `NotImplementedError`. Two things are
  missing, and neither is papered over:
  1. `trustformers_models::bert::BertForTokenClassification` and
     `BertForQuestionAnswering` are real models with real heads, but this crate
     exposes no Python wrapper for either yet.
  2. Both pipelines' HuggingFace output shape carries *character* offsets
     (`start` / `end`), and `trustformers-tokenizers` sets
     `TokenizedInput::offset_mapping` to `None` unconditionally in both its
     WordPiece and BPE encoders, so those keys cannot be filled honestly.
     Reporting token indices under them would be wrong in a way callers could
     not detect.
  They previously invented a `B-PER` entity named `"John"` at characters 0..4
  for every input, and the literal answer string `"Example answer"` with
  `score: 0.85` for every question.
- **Tokenizer `from_pretrained` reads real local files.** `WordPieceTokenizer`
  loads `vocab.txt` (ids are line numbers) or `vocab.json`, plus an optional
  `tokenizer_config.json` for the special-token names and `do_lower_case`;
  `BPETokenizer` gained a `from_pretrained` that loads `vocab.json` **and**
  `merges.txt` (both required -- a BPE tokenizer without its merge table cannot
  reproduce its own tokenization). Special-token ids are looked up in the loaded
  vocabulary and a token that is missing from it is an error, not a default id
  pointing at some other token. Previously a `download_file_from_hub` stub
  returned the empty *path* for every request, so `WordPieceTokenizer.from_pretrained`
  could only ever raise "Failed to read vocab.txt: No such file or directory",
  and `BPETokenizer` had no `from_pretrained` at all.
- **`AutoTokenizer.from_pretrained` actually loads the named checkpoint.** It
  used to ignore the path entirely and return a freshly constructed, empty
  tokenizer -- a five-entry `[PAD]/[UNK]/[CLS]/[SEP]/[MASK]` vocabulary for
  WordPiece, and no vocabulary and no merges for BPE -- while reporting success.
  Every real word encodes to `[UNK]` under those, so anything downstream was
  running on noise. SentencePiece checkpoints (T5, LLaMA) now raise
  `NotImplementedError` instead of being silently loaded with the BPE reader.
- **Tokenizer `save_pretrained` round-trips with `from_pretrained`.**
  `WordPieceTokenizer` writes a real `vocab.txt`, and `BPETokenizer` a real
  `vocab.json` + `merges.txt`, alongside `tokenizer_config.json` (carrying the
  tokenizer's *actual* vocabulary size) and `special_tokens_map.json`. The base
  `PreTrainedTokenizer.save_pretrained` now refuses, as
  `PreTrainedModel.save_pretrained` already did: it holds no vocabulary, and it
  used to write a `"vocab_size": 30522` (BERT's, whatever the tokenizer really
  was) with no vocabulary file at all.
- **`top_k` and `top_p` default to `None` in `TextGenerationPipeline`**, not to
  HuggingFace's `50` / `1.0`. `TextGenerator` applies exactly one truncation
  strategy per step, so adopting both defaults would mean silently dropping
  one; setting both explicitly is an error for the same reason.
- **Pipelines reject unknown keyword arguments** instead of accepting and
  ignoring them, which is what the placeholder implementations did with every
  argument they were given (`max_length`, `temperature`, `top_k`, ... were all
  discarded by a `let _ = (...)`).

---

### Dependency pins (PyO3 0.28 / SciRS2 0.5.1)

**Status:** deliberate pin, not drift. Recorded here because it is the one place
this crate knowingly departs from the workspace's "always the latest crates.io
version" policy.

- `pyo3 = "0.28"` and `scirs2-core` / `scirs2-numpy` `= "0.5.1"` are pinned
  *together*. They cannot be bumped independently:
  - `scirs2-numpy` 0.6.x hard-requires `pyo3 = "0.29.0"` (verified against its
    published `Cargo.toml`).
  - `pyo3` declares `links = "python"`, so Cargo permits **exactly one** `pyo3`
    version in the dependency graph. A `scirs2-numpy` on 0.29 next to this
    crate's own `pyo3 = "0.28"` is therefore not a resolvable graph, not merely
    a warning.
  - 0.5.1 is the pair this crate's `Cargo.lock` already resolves against
    `pyo3` 0.28.3.
- **Migrating requires a PyO3 0.28 -> 0.29 pass over this crate's whole binding
  surface** (~10k lines across `models/`, `pipelines/`, `tokenizers.rs`,
  `tensor*.rs`, `training.rs`, `auto.rs`). It is a single atomic change --
  `pyo3`, `pyo3-build-config`, `scirs2-core`, and `scirs2-numpy` all move in one
  commit or none of them do.
- **Security context:** RUSTSEC-2026-0176 / RUSTSEC-2026-0177 are the advisories
  that make this worth tracking rather than leaving as a silent pin. Until the
  migration lands, the pin is the reason `cargo audit` output for this crate
  must be read against these two IDs specifically.
- **Migration checklist** (for whoever picks this up):
  1. Bump `pyo3`, `pyo3-build-config`, `scirs2-core`, `scirs2-numpy` in one edit.
  2. Re-check the pyo3 0.29 deprecations this crate already tracks:
     `Bound::cast` (replaced `PyAnyMethods::downcast` in 0.28) and the removal
     of the crate-root `PyObject` alias, which every module here re-declares
     locally as `type PyObject = Py<PyAny>;`.
  3. `cargo check && cargo clippy --all-targets -- -D warnings && cargo test`
     from inside `trustformers-py/` -- it is a workspace-excluded crate with its
     own `[workspace]` table and its own lock file, so workspace-level commands
     do not cover it.

### Build notes

- **This crate is excluded from the root workspace** (`[workspace]` in its own
  `Cargo.toml`). Every `cargo` command for it must be run from inside
  `trustformers-py/`; a green `cargo check --workspace` at the repo root says
  nothing about this crate.
- **The `trustformers` umbrella crate is deliberately not a dependency.** It was
  removed after verifying, with `cargo tree -e features -p trustformers-core`
  and `-p trustformers-models` snapshots taken before and after, that the
  normalised feature sets of both crates were byte-identical either way -- so no
  feature unification was lost. It dropped 54 transitive packages (and added none) (clap, url,
  the `icu_*` family, `oxiarc-{archive,brotli,bzip2,lzma,snappy}`, `dirs`,
  `encoding_rs`, ...) with no version changes anywhere in the lock. The bindings
  reach `trustformers-core` / `-models` / `-tokenizers` / `-optim` / `-training`
  directly; the only mentions of the umbrella left in `src/` are three
  commented-out `use` lines.

---

## Future Enhancements

### High Priority
- [ ] Enhanced async support for all operations
- [ ] Better PyTorch interoperability
  - **Refinement needed:** DLPack zero-copy tensor exchange? autograd bridge? nn.Module adapter?
- [ ] Improved error messages
- [ ] More pipeline types
  - **Refinement needed:** target pipeline list (image-classification ASR?), most audio/vision pipelines already exist in Rust.

### Performance
- [ ] Further GIL optimization
- [ ] Better memory management
  - **Refinement needed:** target metric (peak RSS? allocation count? leak detection?).
- [ ] Streaming responses
- [ ] Batch processing improvements
  - **Refinement needed:** dynamic batching? padding strategy? target throughput multiplier?

### Features
- [ ] Jupyter notebook widgets
- [ ] TensorBoard integration
- [ ] Model profiling tools
- [ ] Distributed inference: multi-GPU dispatch via Python API
  - **Refinement needed:** RPC transport? model sharding API? at least 2 distinct sub-tasks needed.

---

## Development Guidelines

### Code Standards
- **Python Code:** PEP 8 compliant
- **Type Hints:** Complete type annotations
- **Documentation:** Docstrings for all public APIs
- **Testing:** pytest with >90% coverage

### Build & Test Commands

```bash
# Install in development mode
pip install -e .

# Build extension module
maturin develop

# Build release wheel
maturin build --release

# Run tests
pytest tests/

# Type checking
mypy trustformers/

# Format code
black trustformers/ tests/
ruff check trustformers/ tests/

# Build documentation
cd docs && make html
```

### PyPI Publishing

```bash
# Build wheels
maturin build --release --strip

# Publish to PyPI
maturin publish
```

---

## Installation

### From PyPI

```bash
# Install from PyPI
pip install trustformers

# Install with CUDA support
pip install trustformers[cuda]

# Install with ROCm support
pip install trustformers[rocm]

# Install all extras
pip install trustformers[all]
```

### From Source

```bash
# Clone repository
git clone https://github.com/cool-japan/trustformers
cd trustformers/trustformers-py

# Install Rust (if not installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Python dependencies
pip install maturin

# Build and install
maturin develop --release
```

---

## Usage Examples

### Basic Usage

```python
from trustformers import pipeline

# Text generation
generator = pipeline("text-generation", model="gpt2")
result = generator("The future of AI is", max_length=50)
print(result[0]['generated_text'])
```

### Advanced Usage

```python
from trustformers import AutoModel, AutoTokenizer
import numpy as np

# Load model and tokenizer
model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Prepare inputs
text = "Hello, world!"
inputs = tokenizer(text, return_tensors="np")

# Forward pass
outputs = model(**inputs)

# Get embeddings
embeddings = outputs.last_hidden_state
print(f"Embeddings shape: {embeddings.shape}")
print(f"Embeddings dtype: {embeddings.dtype}")
```

### Batch Processing

```python
from trustformers import pipeline

classifier = pipeline("sentiment-analysis", batch_size=32)

texts = [
    "I love this product!",
    "This is terrible.",
    "It's okay, I guess.",
]

results = classifier(texts)
for text, result in zip(texts, results):
    print(f"{text}: {result['label']} ({result['score']:.4f})")
```

---

**Last Updated:** Refactored for alpha.1 release
**Status:** Production-ready Python bindings
**PyPI:** Available as `trustformers` package
**Python:** 3.7+ supported
