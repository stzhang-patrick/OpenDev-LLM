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


def _check_dependencies(disabled: bool) -> None:
    if disabled:
        logger.warning("Version checking has been disabled, may lead to unexpected behaviors.")
    else:
        require_version("transformers>=4.37.2", "To fix: pip install transformers>=4.37.2")
        require_version("datasets>=2.14.3", "To fix: pip install datasets>=2.14.3")
        require_version("accelerate>=0.27.2", "To fix: pip install accelerate>=0.27.2")


def _parse_args(
        parser: HfArgumentParser,
        args: Optional[Dict[str, Any]] = None
        ) -> Tuple[Any]:
    r"""
    Parse arguments into predefined dataclasses.
    """
    
    if args is not None:
        return parser.parse_dict(args)
    
    # TODO (zst) - Add support for parsing from a yaml or json file

    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        return parser.parse_json_file(os.path.abspath(sys.argv[1]))

    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if unknown_args:
        print(parser.format_help())
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

    # Arguments validation
    if custom_training_args.task != "pt" and data_args.template is None:
        raise ValueError("Please specify which `template` to use.")
    
    if custom_training_args.task != "sft" and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` is only available for `sft` task.")
    
    if custom_training_args.task == "sft" and training_args.do_predict and not training_args.predict_with_generate:
        raise ValueError("Please enable `predict_with_generate` as `do_predict` is set to True.")

    _check_dependencies(disabled=custom_training_args.disable_version_checking)

    logger.info(
        "Process rank: {}, device: {}, n_gpu: {}, distributed training: {}, compute dtype: {}".format(
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1),
            str(model_args.torch_dtype),
        )
    )

    transformers.set_seed(training_args.seed)

    return model_args, data_args, training_args, custom_training_args, generating_args


def get_infer_args(args: Optional[Dict[str, Any]] = None) -> _INFER_CLS:
    model_args, data_args, custom_training_args, generating_args = _parse_infer_args(args)

    _check_dependencies(disabled=custom_training_args.disable_version_checking)

    if data_args.template is None:
        raise ValueError("Please specify which `template` to use.")

    return model_args, data_args, custom_training_args, generating_args