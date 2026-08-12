"""
Task 3 extension: LoRA hyperparameter ablation study.

Staged (not full-grid) search to keep CPU-only compute tractable:
  Stage 1: vary rank r in {8, 16, 32} at fixed lr=2e-4, target_modules=ATTN_MLP
  Stage 2: vary lr in {1e-4, 5e-4} at the best r from Stage 1
           (the r/2e-4 result is reused from Stage 1, not re-run)
  Stage 3: at the best (r, lr) from Stages 1-2, compare target_modules
           ATTN_ONLY vs ATTN_MLP (ATTN_MLP result reused from Stages 1-2)

Each trial trains for 1 epoch (not 3) over the same 2,400-example train
split used in Task 3, for a fast, comparable convergence signal -- standard
practice for hyperparameter search on a compute budget. Selection metric:
final validation loss. The winning config is written to
outputs/ablation_winner.json for Phase C (the final production retrain) to
consume.
"""
import importlib.util
import json
import os
import time

from utils import MODEL_NAME, PROCESSED_DIR, OUTPUTS_DIR, PLOTS_DIR, read_jsonl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

# Reuse the tokenization/collation helpers from 03_finetune_lora.py rather
# than duplicating them (that module's heavy work is behind `if __name__`,
# so importing it only defines functions/constants -- no side effects).
_spec = importlib.util.spec_from_file_location(
    "ft03", os.path.join(os.path.dirname(os.path.abspath(__file__)), "03_finetune_lora.py")
)
ft03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ft03)

ATTN_MLP = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ATTN_ONLY = ["q_proj", "k_proj", "v_proj", "o_proj"]

BASE_LR = 2e-4
BASE_R = 16
ABLATION_EPOCHS = 1
LORA_ALPHA_RATIO = 2  # keep alpha = 2*r, consistent with Task 3's r=16/alpha=32


def run_trial(name, r, lr, target_modules, train_records, val_records, tokenizer):
    print(f"\n{'=' * 70}\nTrial: {name}  (r={r}, lr={lr}, target_modules={'ATTN_MLP' if len(target_modules) == 7 else 'ATTN_ONLY'})\n{'=' * 70}")
    torch.manual_seed(42)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=r * LORA_ALPHA_RATIO,
        lora_dropout=0.05,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    train_ds = ft03.build_tokenized_dataset(train_records, tokenizer)
    val_ds = ft03.build_tokenized_dataset(val_records, tokenizer)

    args = TrainingArguments(
        output_dir=f"/tmp/ablation_{name}",
        num_train_epochs=ABLATION_EPOCHS,
        per_device_train_batch_size=ft03.PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=ft03.PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=ft03.GRAD_ACCUM_STEPS,
        learning_rate=lr,
        lr_scheduler_type=ft03.LR_SCHEDULER_TYPE,
        warmup_ratio=ft03.WARMUP_RATIO,
        weight_decay=ft03.WEIGHT_DECAY,
        optim=ft03.OPTIMIZER,
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="no",
        use_cpu=True,
        report_to=[],
        seed=42,
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda batch: ft03.collate_fn(batch, tokenizer.pad_token_id),
    )

    t0 = time.time()
    train_result = trainer.train()
    final_eval = trainer.evaluate()
    dt = time.time() - t0

    result = {
        "name": name,
        "r": r,
        "lr": lr,
        "target_modules": "ATTN_MLP" if len(target_modules) == 7 else "ATTN_ONLY",
        "train_loss": train_result.training_loss,
        "eval_loss": final_eval["eval_loss"],
        "train_time_sec": round(dt, 1),
    }
    print(f"Result: {json.dumps(result, indent=2)}")

    del model, trainer
    return result


def main():
    print(f"Loading tokenizer '{MODEL_NAME}' ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_records = read_jsonl(os.path.join(PROCESSED_DIR, "train.jsonl"))
    val_records = read_jsonl(os.path.join(PROCESSED_DIR, "val.jsonl"))
    print(f"Train: {len(train_records)}  Val: {len(val_records)}")

    results_path = os.path.join(OUTPUTS_DIR, "ablation_results.json")
    cache = {}
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            cache = {r["name"]: r for r in json.load(f)}
        print(f"Resuming: {len(cache)} previously completed trial(s) found: {sorted(cache)}")
    results = list(cache.values())

    def save_progress():
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    def get_or_run(name, r, lr, target_modules):
        if name in cache:
            print(f"\n>>> Skipping {name}, reusing cached result: {json.dumps(cache[name])}")
            return cache[name]
        res = run_trial(name, r, lr, target_modules, train_records, val_records, tokenizer)
        cache[name] = res
        results.append(res)
        save_progress()
        return res

    # ---- Stage 1: rank sweep at fixed lr=2e-4, ATTN_MLP ----
    stage1_results = {}
    for r in [8, 16, 32]:
        stage1_results[r] = get_or_run(f"stage1_r{r}", r, BASE_LR, ATTN_MLP)

    best_r = min(stage1_results, key=lambda r: stage1_results[r]["eval_loss"])
    print(f"\n>>> Stage 1 winner: r={best_r} (eval_loss={stage1_results[best_r]['eval_loss']:.4f})")

    # ---- Stage 2: lr sweep at best_r, ATTN_MLP (2e-4 result reused) ----
    stage2_results = {BASE_LR: stage1_results[best_r]}
    for lr in [1e-4, 5e-4]:
        stage2_results[lr] = get_or_run(f"stage2_lr{lr}", best_r, lr, ATTN_MLP)

    best_lr = min(stage2_results, key=lambda lr: stage2_results[lr]["eval_loss"])
    print(f"\n>>> Stage 2 winner: lr={best_lr} (eval_loss={stage2_results[best_lr]['eval_loss']:.4f})")

    # ---- Stage 3: target_modules comparison at best_r/best_lr (ATTN_MLP reused) ----
    attn_mlp_result = stage2_results[best_lr]
    attn_only_result = get_or_run("stage3_attn_only", best_r, best_lr, ATTN_ONLY)

    stage3_results = {"ATTN_MLP": attn_mlp_result, "ATTN_ONLY": attn_only_result}
    best_modules_key = min(stage3_results, key=lambda k: stage3_results[k]["eval_loss"])
    best_modules = ATTN_MLP if best_modules_key == "ATTN_MLP" else ATTN_ONLY
    print(f"\n>>> Stage 3 winner: target_modules={best_modules_key} "
          f"(eval_loss={stage3_results[best_modules_key]['eval_loss']:.4f})")

    winner = {
        "r": best_r,
        "lr": best_lr,
        "target_modules": best_modules,
        "target_modules_label": best_modules_key,
        "lora_alpha": best_r * LORA_ALPHA_RATIO,
        "eval_loss": stage3_results[best_modules_key]["eval_loss"],
        "selection_process": {
            "stage1_rank_sweep": {str(k): v["eval_loss"] for k, v in stage1_results.items()},
            "stage2_lr_sweep": {str(k): v["eval_loss"] for k, v in stage2_results.items()},
            "stage3_target_modules": {k: v["eval_loss"] for k, v in stage3_results.items()},
        },
    }
    winner_path = os.path.join(OUTPUTS_DIR, "ablation_winner.json")
    with open(winner_path, "w", encoding="utf-8") as f:
        json.dump(winner, f, indent=2)
    print(f"\n{'=' * 70}\nWINNING CONFIG: {json.dumps(winner, indent=2)}\nSaved to {winner_path}\n{'=' * 70}")

    # ---- Comparison plot ----
    names = [r["name"] for r in results]
    eval_losses = [r["eval_loss"] for r in results]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(names, eval_losses, color="#4C72B0")
    winner_idx = eval_losses.index(min(eval_losses))
    bars[winner_idx].set_color("#55A868")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Validation loss (1 epoch)")
    plt.title("LoRA Hyperparameter Ablation: Validation Loss by Trial (winner in green)")
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, "ablation_comparison.png"), dpi=150)
    plt.close()
    print("Saved ablation comparison plot.")


if __name__ == "__main__":
    main()
