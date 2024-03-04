import torch
import transformers
from transformers import HfArgumentParser, Seq2SeqTrainingArguments

import logging
import os
import sys
from typing import Any, Dict, Optional, Tuple

from ..utils.logging import get_logger
from .data_args import DataArguments
from .eval_args import EvaluationArguments
from .gen_args import GeneratingArguments
from .model_args import ModelArguments
from .train_args import _TrainingArguments

logger = get_logger(__name__)

_TRAIN_ARGS = [ModelArguments, DataArguments, Seq2SeqTrainingArguments, _TrainingArguments, GeneratingArguments]
_TRAIN_CLS = Tuple[ModelArguments, DataArguments, Seq2SeqTrainingArguments, _TrainingArguments, GeneratingArguments]
_INFER_ARGS = [ModelArguments, DataArguments, _TrainingArguments, GeneratingArguments]
_INFER_CLS = Tuple[ModelArguments, DataArguments, _TrainingArguments, GeneratingArguments]
_EVAL_ARGS = [ModelArguments, DataArguments, _TrainingArguments, EvaluationArguments]
_EVAL_CLS = Tuple[ModelArguments, DataArguments, _TrainingArguments, EvaluationArguments]



def _parse_args(
        parser: HfArgumentParser,
        args: Optional[Dict[str, Any]] = None
        ) -> Tuple[Any]:
    
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
        raise ValueError("Some specified arguments are not used by the HfArgumentParser: {}".format(unknown_args))

    return (*parsed_args,)