from typing import Any, Dict, Optional, Tuple
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from ..hparams import CustomTrainingArguments, ModelArguments
from ..utils.logging import get_logger
from .utils import count_parameters
from termcolor import colored


logger = get_logger(__name__)


def _get_init_kwargs(model_args: ModelArguments) -> Dict[str, Any]:
    r"""
    Extracts the init kwargs from model_args for loading model.
    """
    return {
        "load_in_8bit": model_args.quantization_n_bit == 8,
        "load_in_4bit": model_args.quantization_n_bit == 4,
    }


def load_tokenizer(model_args: ModelArguments) -> PreTrainedTokenizer:
    r"""
    Loads pretrained tokenizer.
    """
    logger.info(f"Loading tokenizer from {model_args.model_name_or_path}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=model_args.use_fast_tokenizer
    )
    
    return tokenizer


def load_model(model_args: ModelArguments) -> PreTrainedModel:
    r"""
    Loads pretrained model.
    """
    logger.info(f"Loading model from {model_args.model_name_or_path}...")

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        cache_dir=model_args.cache_dir,
    )

    init_kwargs = _get_init_kwargs(model_args)
    
    model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=model_args.torch_dtype,
            **init_kwargs,
    )

    # TODO (zny): Add support for peft model

    trainable_params, total_params = count_parameters(model)
    logger.info(f"Trainable parameters: {trainable_params}, Total parameters: {total_params}")

    return model


def load_model_and_tokenizer(
    model_args: ModelArguments,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    r"""
    Loads pretrained model and tokenizer.
    """
    tokenizer = load_tokenizer(model_args)
    model = load_model(model_args)
    return model, tokenizer
