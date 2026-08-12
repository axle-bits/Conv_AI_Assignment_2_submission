# PS2: Parameter-Efficient Fine-Tuning and Human Preference Alignment for a Domain-Specific LLM

**Group 136:** Adithya M Sasi (2024AC05785), Sumit Yadav (2024AC05691), M. Karthikeyan (2024AC05386), Hani Rahim (2024AD05439)

Domain: e-commerce customer support. Base model: `HuggingFaceTB/SmolLM2-360M-Instruct`.
PEFT method: LoRA. Preference alignment: DPO.

Full write-up, results, and the headline finding (LoRA fine-tuning caused a safety
regression that neither a direct DPO run nor a safety-augmented retrain fully fixed) are
in [`Group_136.docx`](Group_136.docx).

## What's here

- `Group_136.docx` — the written report (cover page, TOC, all 5 tasks + extension
  + conclusions, Virtual Lab screenshots embedded inline)
- `screenshots/` — Virtual Lab proof-of-execution screenshots as standalone image
  files, including live execution runs from the notebooks below
- Notebooks — split into three parts so each stays well within the Virtual Lab's memory
  budget (a single combined notebook was tried first but crashed there). Each is
  self-contained (a "recap" cell re-derives all Part 1 setup in seconds — no repeated
  training) and skips any adapter that's already been trained and saved to disk, so a
  re-run only does the training that's actually missing:
  - `PS2_Part1_Tasks1-5.ipynb` — Setup, Task 1 (dataset), Task 2 (baseline), Task 3
    (LoRA fine-tuning), Task 4 (comparative eval), Task 5 (preference alignment + DPO)
  - `PS2_Part2_Ablation_Safety_V2Retrain.ipynb` — Extension Phase A (LoRA hyperparameter
    ablation), Phase B (safety-augmented training data), Phase C (final v2 retrain)
  - `PS2_Part3_V2Eval_DPO.ipynb` — Extension Phase D (re-evaluating v2, safety holdout
    generalization test, DPO on top of v2) and Overall Conclusions
- `python scripts/` — the same pipeline as separate per-task Python scripts
- `data/processed/` — cleaned train/val/test splits, the preference dataset, and the
  safety-augmentation set
- `outputs/` — every logged metric, evaluation table, rubric score, and generated plot
- `models/*/adapter_config.json`, `models/*/README.md` — LoRA/DPO adapter configs for
  all 4 saved adapters (rank, alpha, target modules, base model)

## Trained adapter weights

The actual adapter weights (`adapter_model.safetensors`, 34-67MB each, ~202MB total
across the 4 adapters) aren't committed to this repo. Full submission folder, including
the weights:

**Google Drive:** https://drive.google.com/drive/folders/1QFQYWSOzgltGdmQDERH9RbmFIKgoQSCz?usp=drive_link

## Full development history

The complete development repo (same pipeline, plus the raw dataset sample and the
trained adapter weights committed in full) is at
[`axle-bits/Conv_AI_Assignment_2`](https://github.com/axle-bits/Conv_AI_Assignment_2).
