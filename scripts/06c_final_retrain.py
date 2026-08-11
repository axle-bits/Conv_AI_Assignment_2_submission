"""
Phase C: Final production retrain.

Combines the winning hyperparameters from the Phase A ablation
(outputs/ablation_winner.json) with the Phase B safety-augmented training
data (26 diverse refusal examples mixed into the original 2,400-example
train split) for a full 3-epoch LoRA training run.

Saved to models/lora_adapter_v2/ -- kept SEPARATE from the original Task 3
adapter (models/lora_adapter/) so both are preserved for direct comparison
in the report, rather than overwriting the original result.

Validation/test splits are left as pure original Bitext data (unchanged) --
the safety examples are training-only, since val/test are used for the
Task 4 ROUGE/BLEU comparison against e-commerce gold responses, which the
safety examples have no equivalent for.
"""
import importlib.util
import json
import os
import random

from utils import (
    MODEL_NAME, PROCESSED_DIR, MODELS_DIR, PLOTS_DIR, OUTPUTS_DIR, read_jsonl,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

_spec = importlib.util.spec_from_file_location(
    "ft03", os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_finetune_lora.py")
)
ft03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft03)

LORA_ADAPTER_V2_DIR = os.path.join(MODELS_DIR, "lora_adapter_v2")
NUM_EPOCHS = 3


def main():
    torch.manual_seed(42)
    random.seed(42)

    winner_path = os.path.join(OUTPUTS_DIR, "ablation_winner.json")
    with open(winner_path, encoding="utf-8") as f:
        winner = json.load(f)
    print(f"Using ablation winner config: {json.dumps(winner, indent=2)}")

    print(f"Loading tokenizer/model '{MODEL_NAME}' ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=winner["r"],
        lora_alpha=winner["lora_alpha"],
        lora_dropout=0.05,
        target_modules=winner["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_records = read_jsonl(os.path.join(PROCESSED_DIR, "train.jsonl"))
    safety_records = read_jsonl(os.path.join(PROCESSED_DIR, "safety_augmentation.jsonl"))
    combined_train = train_records + safety_records
    random.shuffle(combined_train)
    val_records = read_jsonl(os.path.join(PROCESSED_DIR, "val.jsonl"))
    print(f"Train: {len(train_records)} original + {len(safety_records)} safety = "
          f"{len(combined_train)} total.  Val: {len(val_records)} (unchanged)")

    train_ds = ft03.build_tokenized_dataset(combined_train, tokenizer)
    val_ds = ft03.build_tokenized_dataset(val_records, tokenizer)

    training_args = TrainingArguments(
        output_dir=os.path.join(LORA_ADAPTER_V2_DIR, "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=ft03.PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=ft03.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=ft03.GRAD_ACCUM_STEPS,
        learning_rate=winner["lr"],
        lr_scheduler_type=ft03.LR_SCHEDULER_TYPE,
        warmup_ratio=ft03.WARMUP_RATIO,
        weight_decay=ft03.WEIGHT_DECAY,
        optim=ft03.OPTIMIZER,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        use_cpu=True,
        report_to=[],
        seed=42,
    )

    loss_cb = ft03.LossHistoryCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda batch: ft03.collate_fn(batch, tokenizer.pad_token_id),
        callbacks=[loss_cb],
    )

    print("Starting final production training ...")
    train_result = trainer.train()
    print("Training finished:", train_result)
    final_eval = trainer.evaluate()
    print("Final eval:", final_eval)

    os.makedirs(LORA_ADAPTER_V2_DIR, exist_ok=True)
    model.save_pretrained(LORA_ADAPTER_V2_DIR)
    tokenizer.save_pretrained(LORA_ADAPTER_V2_DIR)
    print(f"Saved v2 LoRA adapter to {LORA_ADAPTER_V2_DIR}")

    log_path = os.path.join(OUTPUTS_DIR, "training_log_v2.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "hyperparameters": {
                "learning_rate": winner["lr"],
                "num_epochs": NUM_EPOCHS,
                "per_device_batch_size": ft03.PER_DEVICE_BATCH_SIZE,
                "grad_accum_steps": ft03.GRAD_ACCUM_STEPS,
                "lora_r": winner["r"],
                "lora_alpha": winner["lora_alpha"],
                "lora_target_modules": winner["target_modules"],
                "target_modules_label": winner["target_modules_label"],
                "train_examples": len(combined_train),
                "safety_examples_added": len(safety_records),
            },
            "ablation_winner_source": winner,
            "train_loss_history": loss_cb.train_loss,
            "eval_loss_history": loss_cb.eval_loss,
            "log_history": trainer.state.log_history,
            "final_eval": final_eval,
        }, f, indent=2)
    print(f"Saved training log to {log_path}")

    if loss_cb.train_loss:
        steps, losses = zip(*loss_cb.train_loss)
        plt.figure(figsize=(8, 5))
        plt.plot(steps, losses, label="train loss (v2)", color="#4C72B0")
        if loss_cb.eval_loss:
            e_steps, e_losses = zip(*loss_cb.eval_loss)
            plt.plot(e_steps, e_losses, marker="o", label="eval loss (v2)", color="#C44E52")
        plt.xlabel("Training step")
        plt.ylabel("Loss")
        plt.title("Final Retrain (Ablation-Winning Config + Safety Data): Loss Curve")
        plt.legend()
        plt.tight_layout()
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, "training_loss_curve_v2.png"), dpi=150)
        plt.close()
        print("Saved v2 loss curve plot.")


if __name__ == "__main__":
    main()
