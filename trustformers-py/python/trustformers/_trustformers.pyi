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
class WordPieceTokenizer:
    """WordPiece tokenizer implementation."""
    
    def __init__(
        self,
        vocab: Dict[str, int],
        unk_token: str = "[UNK]",
        max_input_chars_per_word: int = 100,
    ) -> None: ...
    
    def tokenize(self, text: str) -> List[str]: ...
    def encode(self, text: str) -> List[int]: ...
    def decode(self, tokens: List[int]) -> str: ...

class BPETokenizer:
    """Byte-Pair Encoding tokenizer implementation."""
    
    def __init__(
        self,
        vocab: Dict[str, int],
        merges: List[Tuple[str, str]],
        **kwargs: Any,
    ) -> None: ...
    
    def tokenize(self, text: str) -> List[str]: ...
    def encode(self, text: str) -> List[int]: ...
    def decode(self, tokens: List[int]) -> str: ...

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
    """Auto model class for sequence classification."""
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs: Any,
    ) -> PreTrainedModel: ...

class AutoModelForTokenClassification:
    """Auto model class for token classification."""
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs: Any,
    ) -> PreTrainedModel: ...

class AutoModelForQuestionAnswering:
    """Auto model class for question answering."""
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs: Any,
    ) -> PreTrainedModel: ...

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

    `trustformers_models::bert::BertForTokenClassification` is real, but this
    package exposes no Python wrapper for it, and HuggingFace's character-level
    `start` / `end` keys cannot be filled because the tokenizers do not produce
    an offset mapping yet.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class QuestionAnsweringPipeline:
    """Not available: construction always raises NotImplementedError.

    See `TokenClassificationPipeline` for the two missing pieces.
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
class Trainer:
    """Trainer class for model training."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        args: "TrainingArguments",
        train_dataset: Optional[Any] = None,
        eval_dataset: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None: ...
    
    def train(self) -> None: ...
    def evaluate(self) -> Dict[str, float]: ...
    def save_model(self, output_dir: str) -> None: ...

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