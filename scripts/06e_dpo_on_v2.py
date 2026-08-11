"""
Phase D (cont.): DPO on top of the v2 (ablation-tuned + safety-augmented)
adapter, reusing the exact same 15-example preference dataset and DPO
config from Task 5 (scripts/05_preference_alignment.py) -- only the base
adapter changes, from v1 to v2. This checks whether DPO still adds value
(or causes any regression) once the SFT stage itself is already
safety-hardened, rather than needing to carry the whole safety-fix burden
alone as it did in Task 5.
"""
import importlib.util
import json
import os

from utils import MODEL_NAME, OUTPUTS_DIR, PLOTS_DIR, MODELS_DIR, generate_response

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

_spec = importlib.util.spec_from_file_location(
    "pref05", os.path.join(os.path.dirname(os.path.abspath(__file__)), "05_preference_alignment.py")
)
pref05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pref05)

LORA_ADAPTER_V2_DIR = os.path.join(MODELS_DIR, "lora_adapter_v2")
DPO_ADAPTER_V2_DIR = os.path.join(MODELS_DIR, "dpo_adapter_v2")


def main():
    print(f"Loading base model + v2 LoRA adapter from {LORA_ADAPTER_V2_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_V2_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_V2_DIR, is_trainable=True)
    model.print_trainable_parameters()

    train_items = [it for it in pref05.PREFERENCE_DATA if it["id"] not in pref05.EVAL_HOLD_OUT_IDS]
    eval_items = [it for it in pref05.PREFERENCE_DATA if it["id"] in pref05.EVAL_HOLD_OUT_IDS]
    train_ds = pref05.build_dpo_dataset(tokenizer, train_items)
    eval_ds = pref05.build_dpo_dataset(tokenizer, eval_items)
    print(f"DPO train examples: {len(train_ds)}  eval examples: {len(eval_ds)}")

    dpo_config = DPOConfig(
        output_dir=os.path.join(DPO_ADAPTER_V2_DIR, "checkpoints"),
        beta=0.1,
        num_train_epochs=6,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        max_prompt_length=256,
        max_length=420,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="no",
        use_cpu=True,
        report_to=[],
        seed=42,
    )

    loss_cb = pref05.DpoLossHistoryCallback()
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        callbacks=[loss_cb],
    )

    print("Starting DPO training on v2 ...")
    trainer.train()
    final_eval = trainer.evaluate()
    print("Final DPO(v2) eval:", final_eval)

    os.makedirs(DPO_ADAPTER_V2_DIR, exist_ok=True)
    model.save_pretrained(DPO_ADAPTER_V2_DIR)
    tokenizer.save_pretrained(DPO_ADAPTER_V2_DIR)
    print(f"Saved DPO(v2) adapter to {DPO_ADAPTER_V2_DIR}")

    with open(os.path.join(OUTPUTS_DIR, "dpo_v2_training_log.json"), "w", encoding="utf-8") as f:
        json.dump({"history": loss_cb.history, "log_history": trainer.state.log_history,
                    "final_eval": final_eval}, f, indent=2)

    # Qualitative: p10/pref13 + safety holdout prompts, v2-SFT-only vs v2+DPO
    print("Generating v2-SFT-only vs v2+DPO qualitative comparison on safety prompts ...")
    v2_sft_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    v2_sft_model = PeftModel.from_pretrained(v2_sft_base, LORA_ADAPTER_V2_DIR)
    v2_sft_model.eval()

    with open(os.path.join(os.path.dirname(OUTPUTS_DIR), "data", "processed",
                            "safety_holdout_test_prompts.json"), encoding="utf-8") as f:
        holdout_prompts = json.load(f)

    pref13_prompt = next(it["prompt"] for it in pref05.PREFERENCE_DATA if it["id"] == "pref13")
    comparisons = []
    for item in [{"id": "pref13", "instruction": pref13_prompt}] + holdout_prompts:
        sft_resp = generate_response(v2_sft_model, tokenizer, item["instruction"], max_new_tokens=120)
        dpo_resp = generate_response(model, tokenizer, item["instruction"], max_new_tokens=120)
        comparisons.append({"id": item["id"], "instruction": item["instruction"],
                             "v2_sft_only": sft_resp, "v2_sft_plus_dpo": dpo_resp})
        print(f"[{item['id']}] SFT-only: {sft_resp[:100]!r}")
        print(f"[{item['id']}] SFT+DPO : {dpo_resp[:100]!r}\n")

    with open(os.path.join(OUTPUTS_DIR, "v2_dpo_safety_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comparisons, f, indent=2)
    print("Saved v2_dpo_safety_comparison.json")


if __name__ == "__main__":
    main()
