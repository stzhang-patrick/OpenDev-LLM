import torch
import transformers
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from transformers.utils.versions import require_version
from transformers.trainer_utils import get_last_checkpoint
import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

from ..utils.logging import get_logger
from .data_args import DataArguments
from .gen_args import GeneratingArguments
from .model_args import ModelArguments
from .train_args import CustomTrainingArguments

logger = get_logger(__name__)

_TRAIN_ARGS = [ModelArguments, DataArguments, Seq2SeqTrainingArguments, CustomTrainingArguments, GeneratingArguments]
_TRAIN_CLS = Tuple[ModelArguments, DataArguments, Seq2SeqTrainingArguments, CustomTrainingArguments, GeneratingArguments]
_INFER_ARGS = [ModelArguments, DataArguments, CustomTrainingArguments, GeneratingArguments]
_INFER_CLS = Tuple[ModelArguments, DataArguments, CustomTrainingArguments, GeneratingArguments]


def _set_transformers_logging(log_level: Optional[int] = logging.INFO) -> None:
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def _check_dependencies(disabled: bool) -> None:
    if disabled:
        logger.warning("Version checking has been disabled, may lead to unexpected behaviors.")
    else:
        require_version("transformers>=4.37.2", "To fix: pip install transformers>=4.37.2")
        require_version("datasets>=2.14.3", "To fix: pip install datasets>=2.14.3")
        require_version("accelerate>=0.27.2", "To fix: pip install accelerate>=0.27.2")
        require_version("peft>=0.9.0", "To fix: pip install peft>=0.9.0")
        require_version("trl>=0.7.11", "To fix: pip install trl>=0.7.11")


def _verify_model_args(model_args: ModelArguments, custom_training_args: CustomTrainingArguments) -> None:

    if model_args.adapter_name_or_path is not None and custom_training_args.training_mode != "peft":
        raise ValueError("`adapter_name_or_path` is only valid when using peft.")


def _parse_args(
        parser: HfArgumentParser,
        args: Optional[Dict[str, Any]] = None
        ) -> Tuple[Any]:
    r"""
    Parse arguments into predefined dataclasses.
    """
    
    if args is not None:
        return parser.parse_dict(args)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if unknown_args:
        print(parser.format_help())
        print("Got unknown args, potentially deprecated arguments: {}".format(unknown_args))
        raise ValueError("Some specified arguments are unknown to the HfArgumentParser: {}".format(unknown_args))

    return (*parsed_args,)


def _parse_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    parser = HfArgumentParser(_TRAIN_ARGS)
    return _parse_args(parser, args)


def _parse_infer_args(args: Optional[Dict[str, Any]] = None) -> _INFER_CLS:
    parser = HfArgumentParser(_INFER_ARGS)
    return _parse_args(parser, args)


def get_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    model_args, data_args, training_args, custom_training_args, generating_args = _parse_train_args(args)

    # Setup logging
    if training_args.should_log:
        _set_transformers_logging()

    # Arguments validation
    if custom_training_args.task != "pt" and data_args.template is None:
        raise ValueError("Please specify which `template` to use.")
    
    if custom_training_args.task != "sft" and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` is only available for `sft` task.")
    
    if custom_training_args.task == "sft" and training_args.do_predict and not training_args.predict_with_generate:
        raise ValueError("Please enable `predict_with_generate` to save model predictions.")

    if training_args.do_train and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` is not compatible with `do_train`.")
    
    _verify_model_args(model_args, custom_training_args)
    _check_dependencies(disabled=custom_training_args.disable_version_checking)

    if training_args.do_train and model_args.quantization_bit is not None and (not model_args.upcast_layernorm):
        logger.warning("We recommend enable `upcast_layernorm` in quantized training.")

    if training_args.do_train and (not training_args.fp16) and (not training_args.bf16):
        logger.warning("We recommend enable mixed precision training by setting `fp16` or `bf16`.")

    if (not training_args.do_train) and model_args.quantization_bit is not None:
        logger.warning("Evaluating model in 4/8-bit mode may cause lower scores.")
    
    # Post-process training arguments
    if (
        training_args.local_rank != -1
        and training_args.ddp_find_unused_parameters is None
        and custom_training_args.finetuning_type == "lora"
    ):
        logger.warning("`ddp_find_unused_parameters` needs to be set as False for LoRA in DDP training.")
        training_args.ddp_find_unused_parameters = False

    if (
        training_args.resume_from_checkpoint is None
        and training_args.do_train
        and os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError("Output directory already exists and is not empty. Please set `overwrite_output_dir`.")
        
        if last_checkpoint is not None:
            training_args.resume_from_checkpoint = last_checkpoint
            logger.info(
                "Resuming training from {}. Change `output_dir` or use `overwrite_output_dir` to avoid this behavior.".format(
                    training_args.resume_from_checkpoint
                )
            )
    
    # Post-process model arguments
    model_args.compute_dtype = (
        torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else None)
    )
    model_args.model_max_length = data_args.cutoff_len

    logger.info(
        "Process rank: {}, device: {}, n_gpu: {}, distributed training: {}, compute dtype: {}".format(
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1),
            str(model_args.compute_dtype),
        )
    )

    transformers.set_seed(training_args.seed)

    return model_args, data_args, training_args, custom_training_args, generating_args


def get_infer_args(args: Optional[Dict[str, Any]] = None) -> _INFER_CLS:
    model_args, data_args, custom_training_args, generating_args = _parse_infer_args(args)

    _set_transformers_logging()
    _verify_model_args(model_args, custom_training_args)
    _check_dependencies(disabled=custom_training_args.disable_version_checking)

    if data_args.template is None:
        raise ValueError("Please specify which `template` to use.")

    return model_args, data_args, custom_training_args, generating_args