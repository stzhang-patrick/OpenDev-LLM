from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional
import torch
from runner.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ModelArguments:
    
    # Loading arguments
    model_name_or_path: str = field(
        default="t5-small",
        metadata={
            "help": "Path to the model dir or identifier from huggingface.co/models."
        }
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={
            "help": "Whethe to allow for custom models defined on the Hub in their own modeling files"
            }
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": "The data type to use for loading the model weights."
        }
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "The directory to cache the model weights."
        }
    )

    # Quantization arguments
    quantization_n_bit: Optional[int] = field(
        default=None,
        metadata={
            "help": "Quantize the model to the specified data type."
        }
    )
    
    # Tokenizer arguments
    use_fast_tokenizer: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use the fast tokenizers."
        }
    )

    def __post_init__(self):

        if self.quantization_n_bit is not None and self.torch_dtype is None:
            self.torch_dtype = torch.float16
            logger.warning("Overriding torch_dtype=None with `torch_dtype=torch. float16` due to requirements of `bitsandbytes` to enable model loading in 8-bit or 4-bit.")
            
        if not self.quantization_n_bit in [None, 8, 4]:
            raise ValueError("`quantization_n_bit` can only be one of {None, 8, 4} as only 8-bit or 4-bit quantization is supported.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)