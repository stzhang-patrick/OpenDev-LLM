from typing import Any, Dict, Optional, Tuple
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from ..arguments import CustomArguments, ModelArguments
from .model_utils import count_parameters
from ..utils.logging import get_logger

logger = get_logger(__name__)


def load_tokenizer(model_args: ModelArguments) -> PreTrainedTokenizer:
    r"""
    Loads pretrained tokenizer.
    """
    logger.info(f"Loading tokenizer from {model_args.tokenizer_name_or_path}...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=model_args.use_fast_tokenizer,
    )

    logger.info(f"tokenizer vocab size: {tokenizer.vocab_size}")
    logger.info(f"tokenizer pad token id: {tokenizer.pad_token_id} {tokenizer.pad_token}")
    logger.info(f"tokenizer bos token id: {tokenizer.bos_token_id} {tokenizer.bos_token}")
    logger.info(f"tokenizer eos token id: {tokenizer.eos_token_id} {tokenizer.eos_token}")
    logger.info(f"tokenizer unk token id: {tokenizer.unk_token_id} {tokenizer.unk_token}")
    
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
    
    model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            trust_remote_code=model_args.trust_remote_code,
            load_in_8bit=model_args.load_in_8bit,
            load_in_4bit=model_args.load_in_4bit,
            torch_dtype=model_args.torch_dtype,
    )

    # TODO (zny): Add support for peft model

    trainable_params, total_params = count_parameters(model)
    logger.info(f"Trainable parameters: {trainable_params}, Total parameters: {total_params}")

    return model
