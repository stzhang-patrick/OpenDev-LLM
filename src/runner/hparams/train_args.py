import json
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

@dataclass
class PeftArguments:
   pass


@dataclass
class CustomTrainingArguments(PeftArguments):
    r"""
    Custom additional training arguments.
    """

    task: Optional[Literal["pt", "sft"]] = field(
        default="sft",
        metadata={"help": "Which training task to perform. It should be one of {`pt`, `sft`}."}
    )
    training_mode: Optional[Literal["full", "peft"]] = field(
        default="full",
        metadata={"help": "Which training mode to perform. It should be one of {`full`, `peft`}."}
    )
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
            }
    )
    disable_version_checking: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to disable version checking."}
    )

    def __post_init__(self):

        pass
        

