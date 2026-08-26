"""
TrustformeRS: Python bindings for the TrustformeRS transformer library
========================================================================

Every name below is a real, compiled PyO3 class or function from the
``_trustformers`` Rust extension module (see ``src/lib.rs``'s ``#[pymodule]``
body, which is this list's source of truth -- keep the two in sync by hand
when either changes, since nothing currently checks that automatically).

Until 2026-08-24 this file imported only ``Tensor``/``TensorOptimized``/four
utility functions, with a docstring calling itself "a simplified version
that only exposes the core tensor operations that are currently working" --
stale even by its own description: `pipeline`, `AutoModel`, `AutoTokenizer`,
`BertForTokenClassification`, `BertForQuestionAnswering`, and everything
else this package's README/TODO/type stubs document as ``from trustformers
import ...`` were registered in the compiled extension (real forward
passes, real pipelines) but **unreachable** through this top-level package,
because nothing here imported them. That import gap is closed below --
every ``m.add_class``/``m.add_function``/exception ``lib.rs`` registers is
now re-exported here by name (not ``from ._trustformers import *``: an
explicit list means a class ``lib.rs`` adds later, but this file forgets to
mirror, fails loudly at import time instead of silently staying invisible).

Deliberately **not** imported here: this directory also holds ~36 other
``.py`` files (``pipelines.py``, ``modeling_bert.py``,
``pytorch_advanced.py``, ``jax_integration.py``, and similar) implementing a
large, separate, pure-Python subsystem that predates -- and in several
places duplicates class names from -- the real Rust bindings above (its own
``pipelines.py`` defines a second ``TokenClassificationPipeline``/
``QuestionAnsweringPipeline``, for example). None of it is audited or
verified honest by this pass; importing any of it here would silently
shadow the real classes above depending on import order. It remains
reachable only via its own explicit submodule path (e.g. ``from
trustformers.pipelines import TokenClassificationPipeline`` gets the
*other*, unverified one) -- not through ``from trustformers import ...``.
"""

# Import the real, compiled Rust extension. This is the package's entire
# implementation; every name below is a re-export of something `src/lib.rs`
# registers into the `_trustformers` PyO3 module, not a Python
# reimplementation.
try:
    from ._trustformers import (
        __version__,
        # Exceptions
        TensorError,
        ModelLoadError,
        TokenizerError,
        ConfigError,
        MemoryError,
        ShapeMismatchError,
        InvalidInputError,
        # Core tensor types
        Tensor,
        TensorOptimized,
        AdvancedActivations,
        # Models
        PreTrainedModel,
        BertModel,
        GPT2Model,
        T5Model,
        LlamaModel,
        RwkvModel,
        MambaModel,
        # Task-head models
        BertForSequenceClassification,
        BertForTokenClassification,
        BertForQuestionAnswering,
        GPT2LMHeadModel,
        # Tokenizers
        PreTrainedTokenizer,
        WordPieceTokenizer,
        BPETokenizer,
        # Pipelines
        Pipeline,
        TextGenerationPipeline,
        TextClassificationPipeline,
        TokenClassificationPipeline,
        QuestionAnsweringPipeline,
        # Training
        Trainer,
        TrainingArguments,
        # Auto classes
        AutoModel,
        AutoTokenizer,
        AutoModelForSequenceClassification,
        AutoModelForTokenClassification,
        AutoModelForQuestionAnswering,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        # Utility functions
        get_device,
        set_seed,
        enable_grad,
        no_grad,
        # Pipeline factory
        pipeline,
    )
except ImportError as e:
    raise ImportError(
        "Could not import TrustformeRS C extension. "
        "Make sure the package is properly installed. "
        f"Original error: {e}"
    )

# Define what gets imported with "from trustformers import *". Kept in sync
# with the import list above by construction, not by a separate hand-typed
# copy: a name missing from one and present in the other was exactly the
# defect class ``__version__`` (hardcoded "0.1.1" here, never the real
# ``__version__`` the extension module itself carries) used to be.
__all__ = [
    "__version__",
    "TensorError",
    "ModelLoadError",
    "TokenizerError",
    "ConfigError",
    "MemoryError",
    "ShapeMismatchError",
    "InvalidInputError",
    "Tensor",
    "TensorOptimized",
    "AdvancedActivations",
    "PreTrainedModel",
    "BertModel",
    "GPT2Model",
    "T5Model",
    "LlamaModel",
    "RwkvModel",
    "MambaModel",
    "BertForSequenceClassification",
    "BertForTokenClassification",
    "BertForQuestionAnswering",
    "GPT2LMHeadModel",
    "PreTrainedTokenizer",
    "WordPieceTokenizer",
    "BPETokenizer",
    "Pipeline",
    "TextGenerationPipeline",
    "TextClassificationPipeline",
    "TokenClassificationPipeline",
    "QuestionAnsweringPipeline",
    "Trainer",
    "TrainingArguments",
    "AutoModel",
    "AutoTokenizer",
    "AutoModelForSequenceClassification",
    "AutoModelForTokenClassification",
    "AutoModelForQuestionAnswering",
    "AutoModelForCausalLM",
    "AutoModelForMaskedLM",
    "get_device",
    "set_seed",
    "enable_grad",
    "no_grad",
    "pipeline",
]
