from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class DataArguments:
    
    template: Optional[str] = field(
        default=None,
        metadata={"help": "Which template to use to format data."}
    )
    train_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name of provided dataset(s) to use for training."}
    )
    eval_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name of provided dataset(s) to use for evaluation."}
    )
    dataset_streaming: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to enable dataset streaming."}
    )
    train_max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the train dataset to this number of samples."}
    )
    eval_max_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the eval dataset to this number of samples."}
    )
    sft_packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to pack the dataset for SFT training to improve training efficiency."}
    )

    def __post_init__(self):
        
        pass
