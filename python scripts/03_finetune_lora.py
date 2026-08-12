"""
Task 3: Parameter-Efficient Fine-Tuning (LoRA).

Fine-tunes SmolLM2-360M-Instruct on the e-commerce customer-support
instruction dataset using LoRA adapters (via HuggingFace `peft`), training
only on the assistant-response tokens (prompt tokens are masked out of the
loss with label = -100).
"""
import json
import os

from utils import (
    MODEL_NAME, PROCESSED_DIR, LORA_ADAPTER_DIR, PLOTS_DIR, OUTPUTS_DIR,
    read_jsonl, build_prompt_text,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

MAX_LENGTH = 320

# ---- Training hyperparameters (documented here + in the report) ----
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
PER_DEVICE_BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 2          # effective batch size = 8 * 2 = 16
LR_SCHEDULER_TYPE = "cosine"
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
OPTIMIZER = "adamw_torch"

# ---- LoRA adapter configuration ----
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]


class LossHistoryCallback(TrainerCallback):
    def __init__(self):
        self.train_loss = []  # (step, loss)
        self.eval_loss = []   # (step, loss)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.train_loss.append((state.global_step, logs["loss"]))
        if "eval_loss" in logs:
            self.eval_loss.append((state.global_step, logs["eval_loss"]))


def build_tokenized_dataset(records, tokenizer):
    input_ids_list, labels_list, attn_list = [], [], []
    for r in records:
        prompt_text = build_prompt_text(tokenizer, r["instruction"], r.get("context"))
        full_text = prompt_text + r["response"] + tokenizer.eos_token

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False,
                              truncation=True, max_length=MAX_LENGTH)["input_ids"]

        prompt_len = min(len(prompt_ids), len(full_ids))
        labels = list(full_ids)
        for i in range(prompt_len):
            labels[i] = -100

        input_ids_list.append(full_ids)
        labels_list.append(labels)
        attn_list.append([1] * len(full_ids))

    return Dataset.from_dict({
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attn_list,
    })


def collate_fn(batch, pad_token_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, attn_mask = [], [], []
    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_token_id] * pad_len)
        labels.append(x["labels"] + [-100] * pad_len)
        attn_mask.append(x["attention_mask"] + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
    }


def main():
    torch.manual_seed(42)

    print(f"Loading tokenizer/model '{MODEL_NAME}' ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_records = read_jsonl(os.path.join(PROCESSED_DIR, "train.jsonl"))
    val_records = read_jsonl(os.path.join(PROCESSED_DIR, "val.jsonl"))
    print(f"Train: {len(train_records)}  Val: {len(val_records)}")

    train_ds = build_tokenized_dataset(train_records, tokenizer)
    val_ds = build_tokenized_dataset(val_records, tokenizer)

    training_args = TrainingArguments(
        output_dir=os.path.join(LORA_ADAPTER_DIR, "checkpoints"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        optim=OPTIMIZER,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        use_cpu=True,
        bf16=False,
        report_to=[],
        seed=42,
    )

    loss_cb = LossHistoryCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
        callbacks=[loss_cb],
    )

    print("Starting training ...")
    train_result = trainer.train()
    print("Training finished:", train_result)

    final_eval = trainer.evaluate()
    print("Final eval:", final_eval)

    os.makedirs(LORA_ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(LORA_ADAPTER_DIR)
    tokenizer.save_pretrained(LORA_ADAPTER_DIR)
    print(f"Saved LoRA adapter to {LORA_ADAPTER_DIR}")

    # ---- Persist training log / loss curves ----
    log_path = os.path.join(OUTPUTS_DIR, "training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "hyperparameters": {
                "learning_rate": LEARNING_RATE,
                "num_epochs": NUM_EPOCHS,
                "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
                "grad_accum_steps": GRAD_ACCUM_STEPS,
                "effective_batch_size": PER_DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS,
                "lr_scheduler_type": LR_SCHEDULER_TYPE,
                "warmup_ratio": WARMUP_RATIO,
                "weight_decay": WEIGHT_DECAY,
                "optimizer": OPTIMIZER,
                "max_seq_length": MAX_LENGTH,
                "lora_r": LORA_R,
                "lora_alpha": LORA_ALPHA,
                "lora_dropout": LORA_DROPOUT,
                "lora_target_modules": LORA_TARGET_MODULES,
            },
            "train_loss_history": loss_cb.train_loss,
            "eval_loss_history": loss_cb.eval_loss,
            "log_history": trainer.state.log_history,
            "final_eval": final_eval,
        }, f, indent=2)
    print(f"Saved training log to {log_path}")

    if loss_cb.train_loss:
        steps, losses = zip(*loss_cb.train_loss)
        plt.figure(figsize=(8, 5))
        plt.plot(steps, losses, label="train loss", color="#4C72B0")
        if loss_cb.eval_loss:
            e_steps, e_losses = zip(*loss_cb.eval_loss)
            plt.plot(e_steps, e_losses, marker="o", label="eval loss", color="#C44E52")
        plt.xlabel("Training step")
        plt.ylabel("Loss")
        plt.title("LoRA Fine-Tuning Loss Curve")
        plt.legend()
        plt.tight_layout()
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(PLOTS_DIR, "training_loss_curve.png"), dpi=150)
        plt.close()
        print("Saved loss curve plot.")


if __name__ == "__main__":
    main()
