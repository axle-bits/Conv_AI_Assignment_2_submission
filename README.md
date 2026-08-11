# PS2: Parameter-Efficient Fine-Tuning and Human Preference Alignment for a Domain-Specific LLM

Domain: e-commerce customer support. Base model: `HuggingFaceTB/SmolLM2-360M-Instruct`.
PEFT method: LoRA. Preference alignment: DPO.

Full write-up, results, and the headline finding (LoRA fine-tuning caused a safety
regression that neither a direct DPO run nor a safety-augmented retrain fully fixed) are
in [`report/report.md`](report/report.md) / [`report/report.docx`](report/report.docx).

## What's here

- `report/` — the written report (Markdown + Word, with Virtual Lab screenshots embedded
  in the docx and as standalone files under `report/screenshots/`)
- `PS2_Full_Pipeline.ipynb` — a single self-contained notebook covering all 5 tasks plus
  the follow-up extension, with no dependency on any other file in this repo
- `scripts/` — the same pipeline as separate per-task Python scripts
- `data/processed/` — cleaned train/val/test splits, the preference dataset, and the
  safety-augmentation set
- `outputs/` — every logged metric, evaluation table, rubric score, and generated plot
- `models/*/adapter_config.json`, `models/*/README.md` — LoRA/DPO adapter configs for
  all 4 saved adapters (rank, alpha, target modules, base model)

## Trained adapter weights

The actual adapter weights (`adapter_model.safetensors`, 34-67MB each, ~202MB total
across the 4 adapters) aren't committed to this repo. Full submission folder, including
the weights:

**Google Drive:** _placeholder — link to be added_

## Full development history

The complete development repo (same pipeline, plus the raw dataset sample and the
trained adapter weights committed in full) is at
[`axle-bits/Conv_AI_Assignment_2`](https://github.com/axle-bits/Conv_AI_Assignment_2).
