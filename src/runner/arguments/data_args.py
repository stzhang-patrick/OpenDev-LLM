from dataclasses import dataclass, field
from typing import Literal, Optional
from runner.utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class DataArguments:
    
    dataset_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "The dataset identifier on huggingface.co / local path to dataset."}
    )
    train_split: Optional[str] = field(
        default=None,
        metadata={"help": "The split name of the train subset."}
    )
    eval_split: Optional[str] = field(
        default=None,
        metadata={"help": "The split name of the eval subset."}
    )
    template: Optional[str] = field(
        default=None,
        metadata={"help": "Which template to use to format data."}
    )
    dataset_streaming: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to enable dataset streaming."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of training samples."}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Truncate the number of evaluation samples."}
    )
    pt_packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to pack the dataset for PT training to improve training efficiency."}
    )
    sft_packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to pack the dataset for SFT training to improve training efficiency."}
    )

    def __post_init__(self):

        if self.dataset_name_or_path is None:
            raise ValueError("`dataset_name_or_path` must be provided.")

        if self.train_split is None and self.eval_split is None:
            raise ValueError("At least one of `train_split` or `eval_split` must be provided.")
        
        if self.train_split is None and self.train_max_samples is not None:
            raise ValueError("`train_max_samples` is only valid if `train_split` is provided.")
        
        if self.eval_split is None and self.eval_max_samples is not None:
            raise ValueError("`eval_max_samples` is only valid if `eval_split` is provided.")
