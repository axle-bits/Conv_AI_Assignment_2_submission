"""
Task 1: Domain Dataset Design and Quality Assessment.

Domain: E-commerce customer service.
Source: Bitext Customer Support LLM Chatbot Training Dataset
        (bitext/Bitext-customer-support-llm-chatbot-training-dataset, HuggingFace).

Pipeline:
  1. Load the raw dataset.
  2. Map it onto the required schema: Instruction / Context / Target Response /
     Category / Task Type.
  3. Clean: drop exact duplicates, drop missing/empty fields, normalize
     whitespace, filter degenerate / too-short / too-long samples.
  4. Stratified subsample down to a size that is tractable for CPU-only LoRA
     training on this machine, while preserving the category distribution.
  5. Exploratory analysis: sample counts, instruction/response length stats,
     category distribution -> saved as plots + a stats JSON.
  6. Stratified train/validation/test split, saved as JSONL.
"""
import json
import os
import random

import numpy as np
import pandas as pd
from datasets import load_dataset

from utils import DATA_DIR, PROCESSED_DIR, PLOTS_DIR, write_jsonl

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

RAW_DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

# Target working-set size after cleaning. The full dataset has ~27k rows;
# we subsample (stratified by category) to keep CPU-only LoRA training and
# repeated inference-based evaluation tractable within a reasonable runtime,
# while still preserving every category and enough examples per category
# for meaningful training signal.
TARGET_TOTAL = 3000
MIN_INSTRUCTION_WORDS = 3
MIN_RESPONSE_WORDS = 4
MAX_INSTRUCTION_CHARS = 400
MAX_RESPONSE_CHARS = 1200

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.8, 0.1, 0.1


def normalize_text(s):
    if s is None:
        return ""
    s = str(s).strip()
    s = " ".join(s.split())  # collapse repeated whitespace/newlines
    return s


def load_raw():
    ds = load_dataset(RAW_DATASET_NAME, split="train")
    df = ds.to_pandas()
    return df


def to_schema(df):
    """Map raw columns onto the assignment-required schema."""
    out = pd.DataFrame({
        "instruction": df["instruction"].map(normalize_text),
        "context": None,  # dataset has no native context field; kept optional
        "response": df["response"].map(normalize_text),
        "category": df["category"].map(normalize_text),
        "task_type": df["intent"].map(normalize_text),
    })
    return out


def clean(df):
    stats = {"raw_count": len(df)}

    # 1. Missing / empty values
    before = len(df)
    df = df[(df["instruction"].str.len() > 0) & (df["response"].str.len() > 0)]
    stats["dropped_missing_or_empty"] = before - len(df)

    # 2. Exact duplicates (instruction+response pair)
    before = len(df)
    df = df.drop_duplicates(subset=["instruction", "response"])
    stats["dropped_exact_duplicates"] = before - len(df)

    # 3. Degenerate / too short / too long samples
    before = len(df)
    instr_words = df["instruction"].str.split().str.len()
    resp_words = df["response"].str.split().str.len()
    mask = (
        (instr_words >= MIN_INSTRUCTION_WORDS)
        & (resp_words >= MIN_RESPONSE_WORDS)
        & (df["instruction"].str.len() <= MAX_INSTRUCTION_CHARS)
        & (df["response"].str.len() <= MAX_RESPONSE_CHARS)
        & (df["instruction"].str.lower() != df["response"].str.lower())
    )
    df = df[mask]
    stats["dropped_length_or_degenerate"] = before - len(df)

    df = df.reset_index(drop=True)
    stats["clean_count"] = len(df)
    return df, stats


def stratified_subsample(df, target_total, seed=RANDOM_SEED):
    """Sample proportionally to each category's share, capped at target_total."""
    frac = min(1.0, target_total / len(df))
    sampled = (
        df.groupby("category", group_keys=False)[df.columns.tolist()]
        .apply(lambda g: g.sample(frac=frac, random_state=seed))
        .reset_index(drop=True)
    )
    return sampled


def stratified_split(df, train_frac, val_frac, test_frac, seed=RANDOM_SEED):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    train_parts, val_parts, test_parts = [], [], []
    rng = np.random.RandomState(seed)
    for _, g in df.groupby("category"):
        g = g.sample(frac=1.0, random_state=seed)  # shuffle within category
        n = len(g)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        train_parts.append(g.iloc[:n_train])
        val_parts.append(g.iloc[n_train:n_train + n_val])
        test_parts.append(g.iloc[n_train + n_val:])
    train = pd.concat(train_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val = pd.concat(val_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = pd.concat(test_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return train, val, test


def run_eda(df, out_stats_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    instr_len_words = df["instruction"].str.split().str.len()
    resp_len_words = df["response"].str.split().str.len()

    stats = {
        "num_samples": len(df),
        "num_categories": df["category"].nunique(),
        "num_task_types": df["task_type"].nunique(),
        "avg_instruction_len_words": float(instr_len_words.mean()),
        "median_instruction_len_words": float(instr_len_words.median()),
        "avg_response_len_words": float(resp_len_words.mean()),
        "median_response_len_words": float(resp_len_words.median()),
        "category_distribution": df["category"].value_counts().to_dict(),
        "task_type_distribution": df["task_type"].value_counts().to_dict(),
    }

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Category distribution
    plt.figure(figsize=(9, 5))
    df["category"].value_counts().plot(kind="bar", color="#4C72B0")
    plt.title("Category Distribution (cleaned working set)")
    plt.ylabel("Count")
    plt.xlabel("Category")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "category_distribution.png"), dpi=150)
    plt.close()

    # Response length distribution
    plt.figure(figsize=(8, 5))
    plt.hist(resp_len_words, bins=30, color="#55A868")
    plt.title("Response Length Distribution (words)")
    plt.xlabel("Response length (words)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "response_length_distribution.png"), dpi=150)
    plt.close()

    # Instruction length distribution
    plt.figure(figsize=(8, 5))
    plt.hist(instr_len_words, bins=30, color="#C44E52")
    plt.title("Instruction Length Distribution (words)")
    plt.xlabel("Instruction length (words)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "instruction_length_distribution.png"), dpi=150)
    plt.close()

    with open(out_stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats


def main():
    print(f"Loading raw dataset '{RAW_DATASET_NAME}' ...")
    raw_df = load_raw()
    print(f"Raw rows: {len(raw_df)}")

    df = to_schema(raw_df)
    df, clean_stats = clean(df)
    print("Cleaning stats:", json.dumps(clean_stats, indent=2))

    sub_df = stratified_subsample(df, TARGET_TOTAL)
    print(f"Subsampled working set size: {len(sub_df)}")

    os.makedirs(DATA_DIR, exist_ok=True)
    raw_cache_path = os.path.join(DATA_DIR, "raw", "bitext_raw_sample.csv")
    os.makedirs(os.path.dirname(raw_cache_path), exist_ok=True)
    raw_df.head(2000).to_csv(raw_cache_path, index=False)

    stats_path = os.path.join(PROCESSED_DIR, "dataset_stats.json")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    eda_stats = run_eda(sub_df, stats_path)
    eda_stats["cleaning"] = clean_stats
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(eda_stats, f, indent=2)

    train_df, val_df, test_df = stratified_split(sub_df, TRAIN_FRAC, VAL_FRAC, TEST_FRAC)
    print(f"Split sizes -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}")

    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        records = split_df.to_dict(orient="records")
        write_jsonl(os.path.join(PROCESSED_DIR, f"{name}.jsonl"), records)

    # Save a handful of sample records for the report.
    sample_records = sub_df.sample(n=8, random_state=RANDOM_SEED).to_dict(orient="records")
    with open(os.path.join(PROCESSED_DIR, "sample_records.json"), "w", encoding="utf-8") as f:
        json.dump(sample_records, f, indent=2)

    print("Done. Processed files written to", PROCESSED_DIR)


if __name__ == "__main__":
    main()
