"""
Task 4: Comparative Performance Analysis.

1. Runs the same >=8 benchmark prompts (from Task 2) through the LoRA-adapted
   model, for direct qualitative side-by-side comparison with the baseline.
2. Computes quantitative automatic metrics (ROUGE-1/2/L, BLEU) for both the
   untouched base model and the LoRA-adapted model against held-out test-set
   ground-truth responses.
3. Saves comparison tables and a bar-chart visualization.
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

N_QUANT_TEST_EXAMPLES = 60


def generate_adapted_benchmark_outputs(tokenizer, model):
    results = []
    for prompt in BENCHMARK_PROMPTS:
        response = generate_response(model, tokenizer, prompt["instruction"], prompt.get("context"))
        print(f"[{prompt['id']}] -> {response[:150]!r}")
        results.append({**prompt, "adapted_response": response})
    with open(os.path.join(OUTPUTS_DIR, "adapted_outputs.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def build_side_by_side_table(adapted_results):
    with open(os.path.join(OUTPUTS_DIR, "baseline_outputs.json"), encoding="utf-8") as f:
        baseline_results = {r["id"]: r for r in json.load(f)}

    rows = []
    for r in adapted_results:
        b = baseline_results[r["id"]]
        rows.append({
            "id": r["id"], "category": r["category"], "task_type": r["task_type"],
            "instruction": r["instruction"],
            "baseline_response": b["baseline_response"],
            "adapted_response": r["adapted_response"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUTS_DIR, "side_by_side_comparison.csv"), index=False)
    print(f"Saved side-by-side comparison table ({len(df)} rows).")
    return df


def compute_quantitative_metrics(base_model, lora_model, tokenizer, test_records):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    metrics = {"baseline": {"rouge1": [], "rouge2": [], "rougeL": []},
               "adapted": {"rouge1": [], "rouge2": [], "rougeL": []}}
    bleu_refs = {"baseline": [], "adapted": []}
    bleu_hyps = {"baseline": [], "adapted": []}
    per_example = []

    for i, r in enumerate(test_records):
        reference = r["response"]
        base_resp = generate_response(base_model, tokenizer, r["instruction"], r.get("context"), max_new_tokens=120)
        lora_resp = generate_response(lora_model, tokenizer, r["instruction"], r.get("context"), max_new_tokens=120)

        for name, resp in [("baseline", base_resp), ("adapted", lora_resp)]:
            scores = scorer.score(reference, resp)
            for k in metrics[name]:
                metrics[name][k].append(scores[k].fmeasure)
            bleu_refs[name].append(reference)
            bleu_hyps[name].append(resp)

        per_example.append({
            "instruction": r["instruction"], "category": r["category"],
            "reference": reference, "baseline_response": base_resp, "adapted_response": lora_resp,
        })
        if (i + 1) % 10 == 0:
            print(f"  scored {i + 1}/{len(test_records)} test examples")

    summary = {}
    for name in ["baseline", "adapted"]:
        rouge_avg = {k: sum(v) / len(v) for k, v in metrics[name].items()}
        bleu = sacrebleu.corpus_bleu(bleu_hyps[name], [bleu_refs[name]]).score
        summary[name] = {**rouge_avg, "bleu": bleu}

    with open(os.path.join(OUTPUTS_DIR, "quantitative_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "n_examples": len(test_records)}, f, indent=2)
    with open(os.path.join(OUTPUTS_DIR, "quantitative_per_example.json"), "w", encoding="utf-8") as f:
        json.dump(per_example, f, indent=2)

    print("Quantitative summary:", json.dumps(summary, indent=2))

    # Bar chart
    labels = ["rouge1", "rouge2", "rougeL", "bleu"]
    base_vals = [summary["baseline"][k] if k != "bleu" else summary["baseline"]["bleu"] / 100 for k in labels]
    lora_vals = [summary["adapted"][k] if k != "bleu" else summary["adapted"]["bleu"] / 100 for k in labels]

    x = range(len(labels))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], base_vals, width, label="Baseline", color="#C44E52")
    plt.bar([i + width / 2 for i in x], lora_vals, width, label="LoRA-Adapted", color="#55A868")
    plt.xticks(list(x), ["ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU (/100)"])
    plt.ylabel("Score")
    plt.title(f"Quantitative Comparison on {len(test_records)} Held-Out Test Examples")
    plt.legend()
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.savefig(os.path.join(PLOTS_DIR, "quantitative_comparison.png"), dpi=150)
    plt.close()
    print("Saved quantitative comparison plot.")

    return summary


def main():
    print(f"Loading base model + LoRA adapter from {LORA_ADAPTER_DIR} ...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    base_model.eval()

    base_model_for_lora = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    lora_model = PeftModel.from_pretrained(base_model_for_lora, LORA_ADAPTER_DIR)
    lora_model.eval()

    print("Generating adapted-model responses to benchmark prompts ...")
    adapted_results = generate_adapted_benchmark_outputs(tokenizer, lora_model)
    build_side_by_side_table(adapted_results)

    test_records = read_jsonl(os.path.join(PROCESSED_DIR, "test.jsonl"))[:N_QUANT_TEST_EXAMPLES]
    print(f"Computing quantitative metrics on {len(test_records)} held-out test examples ...")
    compute_quantitative_metrics(base_model, lora_model, tokenizer, test_records)


if __name__ == "__main__":
    main()
