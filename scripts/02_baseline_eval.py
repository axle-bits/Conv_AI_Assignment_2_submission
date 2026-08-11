"""
Task 2: Baseline Language Model Benchmarking.

Loads the untouched pre-trained SmolLM2-360M-Instruct model and generates
responses to >=8 benchmark prompts spanning different task types in the
e-commerce customer-support domain. Outputs are saved for manual/qualitative
scoring (done separately, see outputs/baseline_eval_table.csv and the report).
"""
import json
import os
import time

from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import MODEL_NAME, OUTPUTS_DIR, BENCHMARK_PROMPTS, generate_response


def main():
    print(f"Loading tokenizer and base model '{MODEL_NAME}' ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    results = []
    for prompt in BENCHMARK_PROMPTS:
        t0 = time.time()
        response = generate_response(
            model, tokenizer, prompt["instruction"], prompt.get("context")
        )
        dt = time.time() - t0
        print(f"[{prompt['id']}] ({dt:.1f}s) {prompt['instruction'][:60]!r}")
        print(f"    -> {response[:200]!r}\n")
        results.append({
            **prompt,
            "baseline_response": response,
            "gen_time_sec": round(dt, 2),
        })

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUTS_DIR, "baseline_outputs.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved baseline outputs to {out_path}")


if __name__ == "__main__":
    main()
