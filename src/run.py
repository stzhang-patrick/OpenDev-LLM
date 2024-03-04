from typing import TYPE_CHECKING, Any, Dict, List, Optional


if TYPE_CHECKING:
    from transformers import TrainerCallback

from runner.utils.logging import get_logger
from runner.utils.callbacks import LogCallback
from runner.hparams import get_train_args, get_infer_args
from runner.model import load_model_and_tokenizer
from runner.pt import run_pt
from runner.sft import run_sft

from runner.hparams import DataArguments


logger_path = '/mnt/proj/workspace/FrogEngine/src/test.log'
logger = get_logger(__name__, logger_path=logger_path, tz_offset=8)
logger.info("Hello, world!")

def run_exp(
        args: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List["TrainerCallback"]] = None
        ):
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)
    callbacks = [LogCallback()] if callbacks is None else callbacks

    if training_args.task == "pt":
        run_pt(model_args, data_args, training_args, finetuning_args, callbacks)
    
    elif training_args.task == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    
    else:
        raise ValueError("Unknown task specified. It should be one of {`pt`, `sft`}.")