"""LoRA/QLoRA fine-tuning script for sovereign anomaly explanation models.

Supports two training modes:
  --mode sft   Supervised fine-tuning on high-quality mistral-bedrock outputs.
               Teaches the base model the task format and energy domain vocabulary.
  --mode dpo   Direct Preference Optimisation on (chosen, rejected) pairs.
               Aligns the SFT-initialised model to prefer high-accuracy explanations.

Recommended workflow:
  1. Generate training data:
       python -m stream4_finetuning.training.generate_sft_dataset
       python -m stream4_finetuning.training.generate_dpo_pairs
  2. SFT pass:
       python -m stream4_finetuning.training.train_lora --mode sft
  3. DPO alignment:
       python -m stream4_finetuning.training.train_lora --mode dpo \\
           --model stream4_finetuning/results/sft_checkpoint

Target models (tested):
  Qwen/Qwen3-8B              — best quality, fits on A10G (24GB) with 4-bit QLoRA
  mistralai/Mistral-7B-v0.3  — well-tested, good energy-domain baseline
  microsoft/phi-4-mini-4k    — fits on T4 (16GB), lower quality than 7B models

Hardware requirements:
  SFT (4-bit QLoRA, batch=4): A10G 24GB or better
  DPO (4-bit QLoRA, batch=2): A10G 24GB or better; reduce batch if OOM

This script is NOT run in CI — it requires GPU hardware and model weights.
Run on an EC2 g5.xlarge (A10G, $1.006/hr) or g4dn.xlarge (T4, $0.526/hr).

Dependencies (install separately — not in main requirements.txt):
  pip install transformers>=4.40 peft>=0.10 trl>=0.8 bitsandbytes>=0.43 datasets>=2.19
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Training configuration defaults
# ---------------------------------------------------------------------------

_DEFAULT_SFT_DATA = "stream4_finetuning/data/sft_dataset.jsonl"
_DEFAULT_DPO_DATA = "stream4_finetuning/data/dpo_pairs.jsonl"
_DEFAULT_BASE_MODEL = "mistralai/Mistral-7B-v0.3"
_DEFAULT_OUTPUT_DIR = "stream4_finetuning/results"

# LoRA hyperparameters — validated on Mistral 7B and Qwen3 8B
_LORA_CONFIG = {
    "r": 16,               # rank — higher = more capacity, more VRAM
    "lora_alpha": 32,      # scaling factor (typically 2x rank)
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# QLoRA quantisation — loads base model in 4-bit to fit in 24GB VRAM
_BNBCONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": "bfloat16",
}


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _load_sft_jsonl(path: Path) -> list[dict[str, str]]:
    """Load SFT dataset from JSONL and format for trl.SFTTrainer.

    Converts Alpaca-format (instruction, input, output) to a single
    'text' field with the ChatML prompt template.

    Args:
        path: Path to the sft_dataset.jsonl file.

    Returns:
        List of dicts with a single 'text' key (ChatML formatted).
    """
    examples: list[dict[str, str]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row: dict[str, object] = json.loads(line)
            text = (
                f"<|im_start|>system\n{row['instruction']}<|im_end|>\n"
                f"<|im_start|>user\n{row['input']}<|im_end|>\n"
                f"<|im_start|>assistant\n{row['output']}<|im_end|>"
            )
            examples.append({"text": text})
    logger.info("Loaded %d SFT examples from %s", len(examples), path)
    return examples


def _load_dpo_jsonl(path: Path) -> list[dict[str, str]]:
    """Load DPO pairs from JSONL and format for trl.DPOTrainer.

    Args:
        path: Path to the dpo_pairs.jsonl file.

    Returns:
        List of dicts with prompt, chosen, rejected keys.
    """
    pairs: list[dict[str, str]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row: dict[str, object] = json.loads(line)
            pairs.append(
                {
                    "prompt": str(row["prompt"]),
                    "chosen": str(row["chosen"]),
                    "rejected": str(row["rejected"]),
                }
            )
    logger.info("Loaded %d DPO pairs from %s", len(pairs), path)
    return pairs


# ---------------------------------------------------------------------------
# Training entry points
# ---------------------------------------------------------------------------


def run_sft(
    data_path: Path,
    base_model: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_seq_length: int,
) -> None:
    """Run supervised fine-tuning with QLoRA.

    Args:
        data_path: Path to sft_dataset.jsonl.
        base_model: HuggingFace model ID or local path.
        output_dir: Directory to save the LoRA adapter.
        epochs: Number of training epochs.
        batch_size: Per-device training batch size.
        learning_rate: AdamW learning rate.
        max_seq_length: Maximum sequence length in tokens.
    """
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from trl import SFTTrainer
    except ImportError:
        logger.error(
            "Training dependencies not installed. Run:\n"
            "  pip install transformers>=4.40 peft>=0.10 trl>=0.8 bitsandbytes>=0.43 datasets>=2.19"
        )
        sys.exit(1)

    logger.info("SFT — model=%s epochs=%d batch=%d lr=%s", base_model, epochs, batch_size, learning_rate)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=_BNBCONFIG["load_in_4bit"],
        bnb_4bit_quant_type=str(_BNBCONFIG["bnb_4bit_quant_type"]),
        bnb_4bit_use_double_quant=bool(_BNBCONFIG["bnb_4bit_use_double_quant"]),
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(**_LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    raw_data = _load_sft_jsonl(data_path)
    dataset = Dataset.from_list(raw_data)

    sft_output = output_dir / "sft_checkpoint"
    training_args = TrainingArguments(
        output_dir=str(sft_output),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(str(sft_output))
    logger.info("SFT complete — adapter saved to %s", sft_output)


def run_dpo(
    data_path: Path,
    base_model: str,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    beta: float,
) -> None:
    """Run DPO alignment with QLoRA on top of an SFT-initialised model.

    Args:
        data_path: Path to dpo_pairs.jsonl.
        base_model: HuggingFace model ID or path to SFT checkpoint.
        output_dir: Directory to save the DPO-aligned adapter.
        epochs: Number of training epochs.
        batch_size: Per-device training batch size.
        learning_rate: AdamW learning rate.
        beta: DPO temperature — controls deviation from reference policy.
    """
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
        from trl import DPOTrainer
    except ImportError:
        logger.error(
            "Training dependencies not installed. Run:\n"
            "  pip install transformers>=4.40 peft>=0.10 trl>=0.8 bitsandbytes>=0.43 datasets>=2.19"
        )
        sys.exit(1)

    logger.info("DPO — model=%s epochs=%d batch=%d beta=%s", base_model, epochs, batch_size, beta)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=_BNBCONFIG["load_in_4bit"],
        bnb_4bit_quant_type=str(_BNBCONFIG["bnb_4bit_quant_type"]),
        bnb_4bit_use_double_quant=bool(_BNBCONFIG["bnb_4bit_use_double_quant"]),
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(**{**_LORA_CONFIG, "r": 8})
    dpo_output = output_dir / "dpo_checkpoint"

    raw_data = _load_dpo_jsonl(data_path)
    dataset = Dataset.from_list(raw_data)

    training_args = TrainingArguments(
        output_dir=str(dpo_output),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=5,
        save_strategy="epoch",
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        beta=beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(str(dpo_output))
    logger.info("DPO complete — adapter saved to %s", dpo_output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for LoRA/QLoRA training."""
    parser = argparse.ArgumentParser(
        description="LoRA/QLoRA fine-tuning for sovereign anomaly explanation models"
    )
    parser.add_argument(
        "--mode",
        choices=["sft", "dpo"],
        required=True,
        help="Training mode: sft (supervised) or dpo (preference optimisation)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("FINETUNE_BASE_MODEL", _DEFAULT_BASE_MODEL),
        help="Base model HuggingFace ID or local path",
    )
    parser.add_argument(
        "--sft-data",
        default=_DEFAULT_SFT_DATA,
        help="Path to sft_dataset.jsonl",
    )
    parser.add_argument(
        "--dpo-data",
        default=_DEFAULT_DPO_DATA,
        help="Path to dpo_pairs.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory to save trained adapters",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s  %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "sft":
        run_sft(
            data_path=Path(args.sft_data),
            base_model=args.model,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_seq_length=args.max_seq_length,
        )
    else:
        run_dpo(
            data_path=Path(args.dpo_data),
            base_model=args.model,
            output_dir=output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            beta=args.beta,
        )


if __name__ == "__main__":
    main()
