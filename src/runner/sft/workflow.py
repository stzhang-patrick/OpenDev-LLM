from typing import TYPE_CHECKING, List, Optional

from transformers import (
    TrainingArguments,
    TrainerCallback
)
from ..loader import load_model, load_tokenizer, load_dataset, present_examples
from ..arguments import DataArguments, CustomArguments, ModelArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from runner.utils.logging import get_logger

logger = get_logger(__name__)

def run_sft(
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
        custom_args: CustomArguments
        ):
    
    model = load_model( model_args)
    tokenizer = load_tokenizer(model_args)
    dataset = load_dataset(data_args, custom_args)

    train_dataset = dataset['train'].select(range(data_args.max_train_samples))
    eval_dataset = dataset['validation'].select(range(data_args.max_eval_samples))
    logger.info(f"Loaded train dataset with {len(train_dataset)} samples.")
    logger.info(f"Loaded eval dataset with {len(eval_dataset)} samples.")


    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example['instruction'])):
            text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
            output_texts.append(text)
        return output_texts

    # response_template = " ### Answer:"
    # collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        # data_collator=collator,
        max_seq_length=custom_args.max_seq_length,
    )

    present_examples(trainer, present_k=2)

    trainer.train()
