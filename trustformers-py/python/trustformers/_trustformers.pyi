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
class TextGenerationPipeline:
    """Pipeline for text generation tasks."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Union[WordPieceTokenizer, BPETokenizer],
        **kwargs: Any,
    ) -> None: ...
    
    def __call__(
        self,
        text: str,
        max_length: int = 50,
        num_return_sequences: int = 1,
        temperature: float = 1.0,
        do_sample: bool = True,
        **kwargs: Any,
    ) -> List[Dict[str, str]]: ...

class TextClassificationPipeline:
    """Pipeline for text classification tasks."""
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Union[WordPieceTokenizer, BPETokenizer],
        **kwargs: Any,
    ) -> None: ...
    
    def __call__(
        self,
        text: str,
        **kwargs: Any,
    ) -> List[Dict[str, Union[str, float]]]: ...

def pipeline(
    task: str,
    model: Optional[str] = None,
    tokenizer: Optional[str] = None,
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