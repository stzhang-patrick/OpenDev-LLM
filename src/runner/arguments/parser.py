import torch
import transformers
from transformers import HfArgumentParser, TrainingArguments
from transformers.utils.versions import require_version
import os
import sys
from typing import Any, Dict, Optional, Tuple

from ..utils.logging import get_logger
from .data_args import DataArguments
from .model_args import ModelArguments
from .custom_args import CustomArguments

logger = get_logger(__name__)

_TRAIN_ARGS = [ModelArguments, DataArguments, TrainingArguments, CustomArguments]
_TRAIN_CLS = Tuple[ModelArguments, DataArguments, TrainingArguments, CustomArguments]

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
        raise ValueError("Some specified arguments are unknown: {}".format(unknown_args))

    return (*parsed_args,)


def _parse_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    parser = HfArgumentParser(_TRAIN_ARGS)
    return _parse_args(parser, args)


def get_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    model_args, data_args, training_args, custom_args = _parse_train_args(args)

    # Arguments validation
    if custom_args.task != "pt" and data_args.template is None:
        raise ValueError("Please specify which `template` to use.")
    
    if custom_args.task != "sft" and training_args.predict_with_generate:
        raise ValueError("`predict_with_generate` is only available for `sft` task.")
    
    if custom_args.task == "sft" and training_args.do_predict and not training_args.predict_with_generate:
        raise ValueError("Please enable `predict_with_generate` as `do_predict` is set to True.")
    
    if data_args.pt_packing and custom_args.task != "pt":
        raise ValueError("`pt_packing` is only available for `pt` task.")
    
    if data_args.sft_packing and custom_args.task != "sft":
        raise ValueError("`sft_packing` is only available for `sft` task.")

    logger.info(
        "Process rank: {}, device: {}, n_gpu: {}, distributed training: {}".format(
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1)
        )
    )

    transformers.set_seed(training_args.seed)

    return model_args, data_args, training_args, custom_args
