from transformers import Trainer
from runner.utils.logging import get_logger

logger = get_logger(__name__)


def present_examples(
        trainer: Trainer,
        present_k: int,
        ):
    """
    Present the first k examples in a trainer dataloader for debugging purpose.
    """
    ignore_token = "<IGNORE>"
    ignore_token_id = -100

    pad_token = trainer.tokenizer.pad_token
    pad_token_id = trainer.tokenizer.pad_token_id

    logger.info("*" * 100)
    logger.info(f"Presenting the first {present_k} examples in the dataloader...")
    train_dataloader = trainer.get_train_dataloader()
    tokenizer = trainer.tokenizer
    for batch in train_dataloader:
        for i in range(present_k):
            input_ids = batch['input_ids'][i]
            attention_mask = batch['attention_mask'][i]
            labels = batch['labels'][i]
            logger.info(f"Example {i}:")
            logger.info(f"input_ids: {input_ids.shape}\n{input_ids}\ndecoded input_ids: {tokenizer.decode(input_ids)}")
            logger.info(f"attention_mask: {attention_mask.shape}\n{attention_mask}")


            # Convert ignore_token_id to pad_token_id for decoding
            condition = labels == ignore_token_id
            converted_labels = labels.masked_fill(condition, pad_token_id)
            # print(converted_labels)
            converted_labels = tokenizer.decode(converted_labels)
            # convert pad_token back to ignore_token for visualization
            converted_labels = converted_labels.replace(pad_token, ignore_token)
            logger.info(f"labels: {labels.shape}\n{labels}\ndecoded labels: {converted_labels}")
        break
    logger.info("*" * 100)
        
