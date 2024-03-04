from typing import TYPE_CHECKING

from transformers import TrainerCallback

if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments

from .logging import get_logger

logger = get_logger(__name__)


