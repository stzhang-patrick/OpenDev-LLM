import json
from torch.utils.data import Dataset
from typing import Dict, List, Any, Union
import torch
from runner.arguments import DataArguments, CustomArguments
from datasets import load_dataset, load_from_disk, Dataset, IterableDataset
from runner.utils.logging import get_logger

logger = get_logger(__name__)


def load_dataset(
        data_args: DataArguments,
        custom_args: CustomArguments,
        ) -> Union[Dataset, IterableDataset]:
    

    if custom_args.task == "sft":
        dataset = load_from_disk(data_args.dataset_name_or_path)
        logger.info(f"Loaded dataset from {data_args.dataset_name_or_path}.")
        return dataset