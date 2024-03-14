from typing import TYPE_CHECKING, List, Optional

from transformers import DataCollatorForSeq2Seq
from ...model import load_model, load_tokenizer
# from ...data import get_dataset


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback
    from ...hparams import DataArguments, CustomTrainingArguments, GeneratingArguments, ModelArguments

def run_sft(
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "Seq2SeqTrainingArguments",
        custom_training_args: "CustomTrainingArguments",
        generating_args: "GeneratingArguments",
        callbacks: Optional[List["TrainerCallback"]] = None,
        ):
    tokenizer = load_tokenizer(model_args)
    model = load_model( model_args)
    dataset = get_dataset(tokenizer, model_args, data_args, training_args, stage="sft")
