import json
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional
from runner.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class PeftArguments:
   pass


@dataclass
class CustomArguments(PeftArguments):
    r"""
    Custom additional training arguments.
    """

    task: Optional[Literal["pt", "sft"]] = field(
        default="sft",
        metadata={"help": "Which training task to perform. It should be one of {`pt`, `sft`}."}
    )
    mode: Optional[Literal["full", "peft"]] = field(
        default="full",
        metadata={"help": "Which training mode to perform. It should be one of {`full`, `peft`}."}
    )
    max_seq_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
            }
    )

    def __post_init__(self):

        pass
        

