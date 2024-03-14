from typing import Any, Dict, List, Optional
from runner.utils.logging import get_logger
from runner.utils.callbacks import LogCallback
from runner.hparams import get_train_args, get_infer_args
from runner.model import load_model_and_tokenizer
# from runner.train.pt import run_pt
from runner.train.sft import run_sft
from transformers import TrainerCallback

logger = get_logger(__name__)

def run_exp(
        args: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List["TrainerCallback"]] = None
        ):

    model_args, data_args, training_args, custom_training_args, generating_args = get_train_args(args)
    callbacks = [LogCallback()] if callbacks is None else callbacks

    logger.info("Starting experiment...")
    logger.info(f"training_args: {training_args}")
    logger.info(f"custom_training_args: {custom_training_args}")
    logger.info(f"model_args: {model_args}")
    logger.info(f"data_args: {data_args}")
    logger.info(f"generating_args: {generating_args}")

    # if training_args.task == "pt":
    #     run_pt(model_args, data_args, training_args, custom_training_args, callbacks)
    
    if custom_training_args.task == "sft":
        run_sft(model_args, data_args, training_args, custom_training_args, generating_args, callbacks)
    
    else:
        raise ValueError("Unknown task specified. It should be one of {`pt`, `sft`}.")
    

def main():
    run_exp()

if __name__ == "__main__":
    main()