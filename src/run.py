from typing import Any, Dict, List, Optional
from runner.utils.logging import get_logger
from runner.arguments import get_train_args
# from runner.train.pt import run_pt
from runner.sft import run_sft
from transformers import TrainerCallback

logger = get_logger(__name__)

def run_exp(
        args: Optional[Dict[str, Any]] = None
        ):

    model_args, data_args, training_args, custom_args = get_train_args(args)

    logger.info("Starting experiment...")
    logger.info(f"training_args: {training_args}")
    logger.info(f"custom_args: {custom_args}")
    logger.info(f"model_args: {model_args}")
    logger.info(f"data_args: {data_args}")
    # logger.info(f"generating_args: {generating_args}")

    # if training_args.task == "pt":
    #     run_pt(model_args, data_args, training_args, custom_training_args, callbacks)
    
    if custom_args.task == "sft":
        run_sft(model_args, data_args, training_args, custom_args)
    
    else:
        raise ValueError("Unknown task specified. It should be one of {`pt`, `sft`}.")
    

def main():
    run_exp()

if __name__ == "__main__":
    main()