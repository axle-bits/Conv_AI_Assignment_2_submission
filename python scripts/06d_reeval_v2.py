"""
Phase D: Re-evaluate the v2 (ablation-tuned + safety-augmented) adapter.

Two checks:
  1. Comparative quality: same 10 benchmark prompts + same 60 held-out test
     examples used in Task 4, now also run through the v2 adapter, so we can
     see baseline vs v1 (original Task 3) vs v2 side by side -- did the
     safety fix cost any of the domain-adaptation gains?
  2. Safety generalization: the 8 held-out safety test prompts from Phase B
     (worded completely differently from the 26 training examples) run
     through baseline vs v1 vs v2 -- this is the real test of whether the
     fix generalizes or just memorizes the trained phrasings.
"""
import json
import os

from utils import (
    MODEL_NAME, PROCESSED_DIR, LORA_ADAPTER_DIR, OUTPUTS_DIR, PLOTS_DIR,
    BENCHMARK_PROMPTS, read_jsonl, generate_response,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import sacrebleu
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer

LORA_ADAPTER_V2_DIR = os.path.join(os.path.dirname(LORA_ADAPTER_DIR), "lora_adapter_v2")
N_QUANT_TEST_EXAMPLES = 60


def generate_benchmark_outputs(tokenizer, model, tag):
    results = []
    for prompt in BENCHMARK_PROMPTS:
        response = generate_response(model, tokenizer, prompt["instruction"], prompt.get("context"))
        print(f"[{tag}][{prompt['id']}] -> {response[:150]!r}")
        results.append({**prompt, f"{tag}_response": response})
    return results


def compute_quantitative(model, tokenizer, test_records, tag):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    refs, hyps = [], []
    for r in test_records:
        resp = generate_response(model, tokenizer, r["instruction"], r.get("context"), max_new_tokens=120)
        s = scorer.score(r["response"], resp)
        for k in scores:
            scores[k].append(s[k].fmeasure)
        refs.append(r["response"])
        hyps.append(resp)
    rouge_avg = {k: sum(v) / len(v) for k, v in scores.items()}
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    print(f"[{tag}] quantitative: {rouge_avg}, bleu={bleu:.2f}")
    return {**rouge_avg, "bleu": bleu}


def run_safety_holdout(models_and_tags, holdout_prompts, tokenizer):
    results = []
    for item in holdout_prompts:
        row = {"id": item["id"], "instruction": item["instruction"], "note": item["note"]}
        for model, tag in models_and_tags:
            resp = generate_response(model, tokenizer, item["instruction"], max_new_tokens=120)
            row[f"{tag}_response"] = resp
            print(f"[{tag}][{item['id']}] -> {resp[:150]!r}")
        results.append(row)
    return results


def main():
    print(f"Loading base model ...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_V2_DIR)

    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    base_model.eval()

    v1_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    v1_model = PeftModel.from_pretrained(v1_base, LORA_ADAPTER_DIR)
    v1_model.eval()

    v2_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    v2_model = PeftModel.from_pretrained(v2_base, LORA_ADAPTER_V2_DIR)
    v2_model.eval()

    # ---- 1. Benchmark prompts: v2 responses (baseline/v1 already saved) ----
    print("Generating v2 responses to benchmark prompts ...")
    v2_bench = generate_benchmark_outputs(tokenizer, v2_model, "v2")
    with open(os.path.join(OUTPUTS_DIR, "v2_benchmark_outputs.json"), "w", encoding="utf-8") as f:
        json.dump(v2_bench, f, indent=2)

    # Build 3-way side-by-side table
    with open(os.path.join(OUTPUTS_DIR, "baseline_outputs.json"), encoding="utf-8") as f:
        baseline_map = {r["id"]: r["baseline_response"] for r in json.load(f)}
    with open(os.path.join(OUTPUTS_DIR, "adapted_outputs.json"), encoding="utf-8") as f:
        v1_map = {r["id"]: r["adapted_response"] for r in json.load(f)}

    rows = []
    for r in v2_bench:
        rows.append({
            "id": r["id"], "category": r["category"], "instruction": r["instruction"],
            "baseline": baseline_map[r["id"]], "v1_lora": v1_map[r["id"]], "v2_lora_safety": r["v2_response"],
        })
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUTS_DIR, "three_way_comparison.csv"), index=False)
    print("Saved three_way_comparison.csv")

    # ---- 2. Quantitative on 60 test examples ----
    test_records = read_jsonl(os.path.join(PROCESSED_DIR, "test.jsonl"))[:N_QUANT_TEST_EXAMPLES]
    print(f"Computing v2 quantitative metrics on {len(test_records)} test examples ...")
    v2_quant = compute_quantitative(v2_model, tokenizer, test_records, "v2")

    with open(os.path.join(OUTPUTS_DIR, "quantitative_metrics.json"), encoding="utf-8") as f:
        existing = json.load(f)
    existing["summary"]["v2"] = v2_quant
    with open(os.path.join(OUTPUTS_DIR, "quantitative_metrics_v2.json"), "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)
    print("Saved quantitative_metrics_v2.json")

    # Bar chart: baseline vs v1 vs v2
    labels = ["rouge1", "rouge2", "rougeL", "bleu"]
    def vals(d):
        return [d[k] if k != "bleu" else d["bleu"] / 100 for k in labels]
    x = range(len(labels))
    width = 0.27
    plt.figure(figsize=(9, 5))
    plt.bar([i - width for i in x], vals(existing["summary"]["baseline"]), width, label="Baseline", color="#C44E52")
    plt.bar([i for i in x], vals(existing["summary"]["adapted"]), width, label="v1 (LoRA, Task 3)", color="#55A868")
    plt.bar([i + width for i in x], vals(v2_quant), width, label="v2 (ablation-tuned + safety)", color="#4C72B0")
    plt.xticks(list(x), ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU (/100)"])
    plt.ylabel("Score")
    plt.title("Quantitative Comparison: Baseline vs v1 vs v2")
    plt.legend()
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, "quantitative_comparison_v2.png"), dpi=150)
    plt.close()
    print("Saved quantitative_comparison_v2.png")

    # ---- 3. Safety generalization holdout ----
    with open(os.path.join(PROCESSED_DIR, "safety_holdout_test_prompts.json"), encoding="utf-8") as f:
        holdout_prompts = json.load(f)

    print("Running safety holdout generalization test (baseline vs v1 vs v2) ...")
    holdout_results = run_safety_holdout(
        [(base_model, "baseline"), (v1_model, "v1"), (v2_model, "v2")],
        holdout_prompts, tokenizer,
    )
    with open(os.path.join(OUTPUTS_DIR, "safety_holdout_results.json"), "w", encoding="utf-8") as f:
        json.dump(holdout_results, f, indent=2)
    print("Saved safety_holdout_results.json")


if __name__ == "__main__":
    main()
