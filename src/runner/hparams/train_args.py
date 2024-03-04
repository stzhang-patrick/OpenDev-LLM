import json
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

@dataclass
class PeftArguments:
    r"""
    Arguments pertaining to the LoRA training.
    """

    additional_target: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name(s) of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint."
        },
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."},
    )
    lora_dropout: Optional[float] = field(
        default=0.0,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."},
    )
    lora_rank: Optional[int] = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."},
    )
    lora_target: Optional[str] = field(
        default=None,
        metadata={
            "help": """Name(s) of target modules to apply LoRA. \
                    Use commas to separate multiple modules. \
                    Use "all" to specify all the available modules. \
                    LLaMA choices: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], \
                    BLOOM & Falcon & ChatGLM choices: ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"], \
                    Baichuan choices: ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"], \
                    Qwen choices: ["c_attn", "attn.c_proj", "w1", "w2", "mlp.c_proj"], \
                    InternLM2 choices: ["wqkv", "wo", "w1", "w2", "w3"], \
                    Others choices: the same as LLaMA."""
        },
    )
    lora_bf16_mode: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to train lora adapters in bf16 precision."},
    )
    use_rslora: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to use the rank stabilization scaling factor for LoRA layer."},
    )
    use_dora: Optional[bool] = field(
        default=False, metadata={"help": "Whether or not to use the weight-decomposed lora method (DoRA)."}
    )
    create_new_adapter: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class _TrainingArguments(PeftArguments):
    r"""
    Customized additional training arguments.
    """

    task: Optional[Literal["pt", "sft"]] = field(
        default="sft",
        metadata={"help": "Which training task to perform. It should be one of {`pt`, `sft`}."}
    )
    training_mode: Optional[Literal["full", "peft"]] = field(
        default="full",
        metadata={"help": "Which training mode to perform. It should be one of {`full`, `peft`}."}
    )
    plot_loss: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether to plot and save the training loss curves."}
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg
        
    def save_to_json(self, json_path: str):
        r"""
        Save this instance to a JSON file.
        """
        json_string = json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        r"""
        Initialize a new instance of _TrainingArguments from a JSON file.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()

        return cls(**json.loads(text))
        

