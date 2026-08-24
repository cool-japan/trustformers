"""
Type stubs for TrustformeRS Rust extension module.
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Sequence, Protocol, overload
from typing_extensions import Self
import numpy as np

# Type aliases
TensorLike = Union["Tensor", np.ndarray, List, float, int]
DeviceType = str
ShapeType = List[int]

class Tensor:
    """TrustformeRS Tensor class."""
    
    @overload
    def __init__(
        self,
        data: np.ndarray,
        device: Optional[DeviceType] = None,
        requires_grad: bool = False,
    ) -> None: ...
    
    @overload
    def __init__(
        self,
        data: List,
        device: Optional[DeviceType] = None,
        requires_grad: bool = False,
    ) -> None: ...
    
    @overload
    def __init__(
        self,
        data: float,
        device: Optional[DeviceType] = None,
        requires_grad: bool = False,
    ) -> None: ...
    
    def __init__(
        self,
        data: TensorLike,
        device: Optional[DeviceType] = None,
        requires_grad: bool = False,
    ) -> None: ...
    
    @staticmethod
    def zeros(
        shape: ShapeType,
        device: Optional[DeviceType] = None,
        requires_grad: bool = False,
    ) -> Self: ...
    
    @staticmethod
    def ones(
        shape: ShapeType,
        device: Optional[DeviceType] = None,
        requires_grad: bool = False,
    ) -> Self: ...
    
    @staticmethod
    def randn(
        shape: ShapeType,
        mean: float = 0.0,
        std: float = 1.0,
        device: Optional[DeviceType] = None,
        requires_grad: bool = False,
    ) -> Self: ...
    
    @staticmethod
    def rand(
        shape: ShapeType,
        low: float = 0.0,
        high: float = 1.0,
        device: Optional[DeviceType] = None,
        requires_grad: bool = False,
    ) -> Self: ...
    
    @property
    def shape(self) -> ShapeType: ...
    
    @property
    def dtype(self) -> str: ...
    
    @property
    def device(self) -> str: ...
    
    @property
    def requires_grad(self) -> bool: ...
    
    def numpy(self) -> np.ndarray: ...
    
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    
    def __add__(self, other: Union[Self, float]) -> Self: ...
    def __sub__(self, other: Union[Self, float]) -> Self: ...
    def __mul__(self, other: Union[Self, float]) -> Self: ...
    
    def matmul(self, other: Self) -> Self: ...
    def transpose(self, dim0: Optional[int] = None, dim1: Optional[int] = None) -> Self: ...
    def reshape(self, shape: ShapeType) -> Self: ...
    def view(self, shape: ShapeType) -> Self: ...
    
    def sum(self, axis: Optional[List[int]] = None, keepdim: bool = False) -> Self: ...
    def mean(self, axis: Optional[List[int]] = None, keepdim: bool = False) -> Self: ...
    
    def relu(self) -> Self: ...
    def gelu(self) -> Self: ...
    def softmax(self, dim: int = -1) -> Self: ...
    
    def clone(self) -> Self: ...
    def detach(self) -> Self: ...
    def to(self, device: str) -> Self: ...
    
    def __getitem__(self, indices: Any) -> Self: ...
    def __setitem__(self, indices: Any, value: Union[Self, float]) -> None: ...

class PreTrainedModel:
    """Base class for all pretrained models."""
    
    def __init__(self, config: Any) -> None: ...
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        config: Optional[Any] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Optional[Dict[str, str]] = None,
        token: Optional[str] = None,
        **kwargs: Any,
    ) -> Self: ...
    
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    
    def save_pretrained(self, save_directory: str) -> None: ...
    def push_to_hub(self, repo_id: str, **kwargs: Any) -> str: ...

class BertModel(PreTrainedModel):
    """BERT Model for encoding."""
    
    def __init__(self, config: Any) -> None: ...
    
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Tensor]: ...

class BertForSequenceClassification(PreTrainedModel):
    """BERT Model for sequence classification."""

    def __init__(self, config: Any) -> None: ...

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Tensor]: ...

class BertForTokenClassification(PreTrainedModel):
    """BERT Model with a per-token classification head (e.g. NER).

    Unlike `BertForSequenceClassification` (one prediction per input), this
    predicts `num_labels` logits at *every* sequence position. Real forward
    pass, real logits, real loss when `labels` is given.

    No pipeline wraps this yet: `pipeline("token-classification")` still
    raises `NotImplementedError` because this crate's tokenizers do not
    produce the character-level offsets HuggingFace's pipeline output needs.
    Call this class directly for per-token logits instead.
    """

    def __init__(self, config: Optional[Any] = None, num_labels: int = 2) -> None: ...

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs: Any,
    ) -> Self: ...

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]: ...

    def __call__(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]: ...

    def save_pretrained(self, save_directory: str) -> None: ...

    @property
    def num_labels(self) -> int: ...

    @property
    def id2label(self) -> List[str]: ...

class BertForQuestionAnswering(PreTrainedModel):
    """BERT Model with an extractive question-answering (span-prediction) head.

    Real forward pass producing `start_logits`/`end_logits` (one score per
    sequence position each), and a real loss when `start_positions` /
    `end_positions` are given (both, or neither -- HuggingFace's own
    contract).

    No pipeline wraps this yet: `pipeline("question-answering")` still raises
    `NotImplementedError` for the same offset-mapping reason as
    `BertForTokenClassification`. Call this class directly for real
    start/end logits instead.
    """

    def __init__(self, config: Optional[Any] = None) -> None: ...

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs: Any,
    ) -> Self: ...

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        start_positions: Optional[Tensor] = None,
        end_positions: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]: ...

    def __call__(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        start_positions: Optional[Tensor] = None,
        end_positions: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]: ...

    def save_pretrained(self, save_directory: str) -> None: ...

class GPT2Model(PreTrainedModel):
    """GPT-2 Model for text generation."""
    
    def __init__(self, config: Any) -> None: ...
    
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Tensor]: ...

class GPT2LMHeadModel(PreTrainedModel):
    """GPT-2 Model with language modeling head."""
    
    def __init__(self, config: Any) -> None: ...
    
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Tensor]: ...

class T5Model(PreTrainedModel):
    """T5 Model for text-to-text generation."""
    
    def __init__(self, config: Any) -> None: ...
    
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Tensor]: ...

class LlamaModel(PreTrainedModel):
    """Llama Model for causal language modeling."""
    
    def __init__(self, config: Any) -> None: ...
    
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Tensor]: ...

# Tokenizers
#
# `encode(text, text_pair=...)` on both classes below produces a *real* pair
# encoding, not a naive concatenation: WordPiece emits BERT's
# `[CLS] A [SEP] B [SEP]` with 0/1 `token_type_ids`; BPE emits the
# RoBERTa/GPT-2-family `<bos> A <eos> <eos> B <eos>` convention (read from the
# tokenizer's own configured `bos_token`/`eos_token`) with all-zero
# `token_type_ids` (BPE-family models do not use segment ids for pairs).
# `batch_encode_plus`'s `text_pairs` argument uses the same per-item
# encoding, for `WordPieceTokenizer` only -- `BPETokenizer` has no
# `batch_encode_plus`, only single-item `encode`.
class WordPieceTokenizer:
    """WordPiece tokenizer implementation."""

    def __init__(
        self,
        vocab: Dict[str, int],
        unk_token: str = "[UNK]",
        max_input_chars_per_word: int = 100,
    ) -> None: ...

    def tokenize(self, text: str) -> List[str]: ...

    def encode(
        self,
        text: str,
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], Dict[str, List[int]]]: ...

    def batch_encode_plus(
        self,
        texts: List[str],
        text_pairs: Optional[List[Optional[str]]] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Dict[str, List[List[int]]]: ...

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str: ...

class BPETokenizer:
    """Byte-Pair Encoding tokenizer implementation."""

    def __init__(
        self,
        vocab: Dict[str, int],
        merges: List[Tuple[str, str]],
        **kwargs: Any,
    ) -> None: ...

    def tokenize(self, text: str) -> List[str]: ...

    def encode(
        self,
        text: str,
        text_pair: Optional[str] = None,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], Dict[str, List[int]]]: ...

    def decode(self, tokens: List[int], skip_special_tokens: bool = True) -> str: ...

# Auto classes
class AutoModel:
    """Auto model class for loading models by name."""
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs: Any,
    ) -> PreTrainedModel: ...

class AutoTokenizer:
    """Auto tokenizer class for loading tokenizers by name."""
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs: Any,
    ) -> Union[WordPieceTokenizer, BPETokenizer]: ...

class AutoModelForSequenceClassification:
    """Auto model class for sequence classification.

    Resolves to a real `BertForSequenceClassification` for BERT-family
    checkpoints (bert/roberta/distilbert/deberta -- all loaded through
    `BertModel`/`BertConfig` here). Raises `NotImplementedError` for any other
    inferred architecture, naming the checkpoint's inferred type: this used to
    silently return a bare, headless encoder for a non-BERT checkpoint,
    labelled as if it were a sequence-classification model.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs: Any,
    ) -> BertForSequenceClassification: ...

class AutoModelForTokenClassification:
    """Auto model class for token classification.

    Resolves to a real `BertForTokenClassification` for BERT-family
    checkpoints; raises `NotImplementedError` otherwise. See
    `AutoModelForSequenceClassification` for why -- this used to silently
    delegate to `AutoModel`, returning a bare encoder with no
    token-classification head at all.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs: Any,
    ) -> BertForTokenClassification: ...

class AutoModelForQuestionAnswering:
    """Auto model class for question answering.

    Resolves to a real `BertForQuestionAnswering` for BERT-family checkpoints;
    raises `NotImplementedError` otherwise. See
    `AutoModelForSequenceClassification` for why.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs: Any,
    ) -> BertForQuestionAnswering: ...

class AutoModelForCausalLM:
    """Auto model class for causal language modeling."""
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs: Any,
    ) -> PreTrainedModel: ...

class AutoModelForMaskedLM:
    """Auto model class for masked language modeling."""
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs: Any,
    ) -> PreTrainedModel: ...

# Pipelines
#
# `TextGenerationPipeline` and `TextClassificationPipeline` resolve their model
# and tokenizer at construction, so a model without the head the task needs is a
# TypeError from `__init__`, not a surprise at call time. Both reject keyword
# arguments they cannot honour rather than ignoring them.
class TextGenerationPipeline:
    """Text generation over a real GPT-2 language-model head."""

    def __init__(
        self,
        model: GPT2LMHeadModel,
        tokenizer: Union[WordPieceTokenizer, BPETokenizer],
        device: Optional[str] = None,
    ) -> None: ...

    # `top_k` / `top_p` default to None, not to HuggingFace's 50 / 1.0: the
    # decoder applies one truncation strategy per step, so setting both raises.
    # A single `str` returns List[Dict]; a list of `str` returns List[List[Dict]].
    def __call__(
        self,
        text_inputs: Union[str, List[str]],
        max_length: int = 50,
        min_length: int = 0,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        num_return_sequences: int = 1,
    ) -> Union[List[Dict[str, str]], List[List[Dict[str, str]]]]: ...

class TextClassificationPipeline:
    """Sequence classification over a real BERT classification head."""

    def __init__(
        self,
        model: BertForSequenceClassification,
        tokenizer: Union[WordPieceTokenizer, BPETokenizer],
        device: Optional[str] = None,
    ) -> None: ...

    # Returns every class ranked best-first (scores are a real softmax and sum
    # to 1); `top_k` keeps only the leading classes.
    def __call__(
        self,
        text_inputs: Union[str, List[str]],
        top_k: Optional[int] = None,
    ) -> Union[
        List[Dict[str, Union[str, float]]],
        List[List[Dict[str, Union[str, float]]]],
    ]: ...

class TokenClassificationPipeline:
    """Not available: construction always raises NotImplementedError.

    `BertForTokenClassification` (see above) is now a real, callable Python
    wrapper -- that gap is closed. What still blocks this pipeline: it must
    report HuggingFace's character-level `start` / `end` keys, and this
    package's tokenizers (`WordPieceTokenizer`, `BPETokenizer`) still do not
    produce an offset mapping. Call `BertForTokenClassification` directly for
    real per-token logits without character offsets.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class QuestionAnsweringPipeline:
    """Not available: construction always raises NotImplementedError.

    See `TokenClassificationPipeline` for why -- the wrapper
    (`BertForQuestionAnswering`) is real, the tokenizer offset mapping is not.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

# `model` and `tokenizer` are model/tokenizer *objects*, not names. When either
# is omitted it is loaded from a default checkpoint path, which must resolve
# locally: this package has no Hugging Face Hub downloader.
def pipeline(
    task: str,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> Union[TextGenerationPipeline, TextClassificationPipeline]: ...

# Training
#
# `Trainer` is a real, constructible class, but `train`/`evaluate`/`predict`/
# `push_to_hub` all raise `NotImplementedError` today, each naming exactly
# what is missing -- see each method's docstring. `save_model` is the one
# real method: it delegates to the wrapped model's own real
# `save_pretrained`. This crate has real forward passes and real loss
# functions (see `BertForSequenceClassification.forward(..., labels=...)`
# etc.), but no backpropagation/optimizer-step path from a loss back to a
# model's parameters exists anywhere in this workspace, so there is no real
# gradient-descent training loop to run.
class Trainer:
    """Trainer class for model training. See the module note above: most
    methods are honest refusals, not stubs that fabricate results."""

    def __init__(
        self,
        model: PreTrainedModel,
        args: "TrainingArguments",
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        data_collator: Optional[Any] = None,
        compute_metrics: Optional[Any] = None,
        callbacks: Optional[Any] = None,
        optimizers: Optional[Any] = None,
    ) -> None: ...

    def train(self) -> Dict[str, Any]:
        """Always raises NotImplementedError: no backward/gradient path from a
        loss to a model's parameters exists in this workspace. Previously
        returned a fabricated `{"train_loss": 0.5, "total_steps": 1000}` for
        every model, dataset, and configuration."""
        ...

    def evaluate(self, eval_dataset: Optional[Any] = None) -> Dict[str, float]:
        """Always raises NotImplementedError: this binding defines no
        dataset-to-model-input protocol. Previously returned a fabricated
        `{"eval_loss": 0.45, "eval_accuracy": 0.92, "eval_samples": 100}`
        regardless of `eval_dataset`."""
        ...

    def predict(self, test_dataset: Any) -> Dict[str, Any]:
        """Always raises NotImplementedError, for the same reason as
        `evaluate`. Previously returned fixed
        `predictions=[0.1, 0.9, 0.3, 0.7]` / `label_ids=[0, 1, 0, 1]`."""
        ...

    def save_model(self, output_dir: Optional[str] = None) -> None:
        """Real: delegates to the wrapped model's own `save_pretrained`."""
        ...

    def push_to_hub(
        self,
        repo_name: str,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
    ) -> str:
        """Always raises NotImplementedError: this crate has no Hugging Face
        Hub client. Previously returned
        `f"https://huggingface.co/{repo_name}"` without uploading anything."""
        ...

class TrainingArguments:
    """Arguments for training configuration."""
    
    def __init__(
        self,
        output_dir: str,
        learning_rate: float = 5e-5,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 8,
        per_device_eval_batch_size: int = 8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        logging_dir: Optional[str] = None,
        **kwargs: Any,
    ) -> None: ...

# Utilities
def get_device() -> str: ...
def set_seed(seed: int) -> None: ...
def enable_grad() -> None: ...
def no_grad() -> None: ...