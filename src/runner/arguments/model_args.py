from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional
import torch
from runner.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ModelArguments:
    
    # Model loading arguments
    model_name_or_path: str = field(
        default="Qwen/Qwen1.5-0.5B",
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
            "help": "The directory to cache the model."
        }
    )

    # Quantization arguments
    load_in_8bit: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to load the model in 8-bit quantization."
        }
    )
    load_in_4bit: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to load the model in 4-bit quantization."
        }
    )
    
    # Tokenizer arguments
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the tokenizer dir or identifier from huggingface.co/models."
        }
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use the fast tokenizers."
        }
    )

    def __post_init__(self):

        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path

        if self.load_in_8bit is not None and self.load_in_4bit is not None:
            raise ValueError("Only one of `load_in_8bit` or `load_in_4bit` can be set.")
        
        if (self.load_in_8bit is not None or self.load_in_4bit is not None) and self.torch_dtype is None:
            self.torch_dtype = torch.float16
            logger.warning("Overriding torch_dtype=None with `torch_dtype=torch. float16` due to requirements of `bitsandbytes` to enable model loading in 8-bit or 4-bit.")
