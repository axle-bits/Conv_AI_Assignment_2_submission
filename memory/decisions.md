# Key Decisions and Rationale

## Domain: E-commerce customer support

The assignment allowed picking any domain the student could justify with a
good dataset fit. Rather than pick a domain first and search for data after,
we searched for a strong, clean, schema-compatible instruction dataset first
and let that determine the domain. E-commerce support won because a
directly-fitting dataset (see below) was available with almost zero reframing
needed.

Alternatives considered: software debugging, cybersecurity support — both
plausible domains, but the datasets available for them were either less
clean, not schema-compatible (missing category/task-type labels), or would
have required more synthetic construction to reach assignment quality bars.

## Dataset: Bitext Customer Support LLM dataset

`bitext/Bitext-customer-support-llm-chatbot-training-dataset` on HuggingFace.

Why:
- 26,872 examples, already labeled with `instruction`, `category` (11
  classes), `intent` (27 sub-types), `response` — maps directly onto the
  assignment's required Instruction / Context / Target Response / Category
  schema without inventing structure.
- Purpose-built for customer-support chatbot training (not a generic
  instruction dataset repurposed for this domain).
- Permissively licensed, hosted on HF Hub, reachable from this machine.

After cleaning (duplicate/missing/degenerate filtering, see
`scripts/01_dataset_prep.py`), 24,957 rows remained. Stratified-subsampled by
category down to **3,001** examples — this subsampling step is a *compute*
decision, not a quality one: the full cleaned pool is too large for CPU-only
LoRA training to finish in a reasonable session. Split 80/10/10
(train/val/test), stratified by category, giving 2,400 / 301 / 300 examples.

## Model: HuggingFaceTB/SmolLM2-360M-Instruct

The assignment's example list (FLAN-T5-base, DistilGPT2, GPT2, TinyLlama,
Phi-2, SmolLM) was treated as illustrative, not exhaustive — we were told to
pick something newer. SmolLM2 (2024) is the newer generation of the SmolLM
family already referenced in the spec, so it's a natural upgrade within the
same family rather than an arbitrary swap.

Why the 360M-Instruct variant specifically:
- Ships with a working ChatML template (`<|im_start|>role ... <|im_end|>`)
  and instruction-tuned behavior out of the box — the Task 2 baseline is a
  real instruction-following model, not a raw completion model, which makes
  the baseline-vs-adapted comparison meaningful.
- Benchmarked directly on this machine (no GPU) before committing: ~1.9s/step
  at batch=8, seq_len=128 — confirmed a full LoRA fine-tune would finish in
  tens of minutes, not hours. Larger newer alternatives (TinyLlama-1.1B,
  Qwen2.5-0.5B-Instruct) were reachable but would have pushed training time
  up for no clear quality payoff at this dataset size.

## PEFT method: LoRA

Mostly forced by the environment rather than a stylistic preference:
- No GPU on this machine (`nvidia-smi` isn't even installed) — confirmed via
  direct check, not assumed. This rules out QLoRA, which needs GPU-backed
  4-bit quantization (bitsandbytes).
- Full fine-tuning of all weights on CPU offers no real benefit at this scale
  and would be slower to train and larger to store.
- LoRA config used: r=16, alpha=32, dropout=0.05, target_modules = all
  attention + MLP projections (`q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`).
  Trainable params: 8.68M / 370M total (2.34%).

## Preference alignment: DPO (not rubric-only fallback)

Given 16 CPU cores and 30GB RAM were available, a real small-scale DPO run
was judged feasible rather than falling back to the rubric-only evaluation
path the assignment allows. DPO trains on top of the already-LoRA-adapted
model (the base model with adapters disabled serves as the frozen reference
policy — TRL's standard PEFT/DPO workflow, `ref_model=None`), rather than
retraining from the raw base model, since aligning the already
domain-adapted model is the realistic deployment path.

15 preference pairs were hand-curated (exceeds the ≥12 minimum), covering all
11 dataset categories plus dedicated safety, hallucination, and consistency
probes. Preferred responses are grounded in real dataset responses;
less-preferred responses were authored to mirror concrete failure modes
actually observed in the Task 2 baseline run (evasive refusal, fabricated
specifics, unsafe data requests, dismissive tone, blanket-policy
hallucination) rather than being generic/synthetic contrasts.

## Handling of "AI generation sanitisation" requests

Twice now (once for the `report.docx` conversion, once for a planned new GitHub repo)
the user asked for the deliverable to avoid "AI generated" flags / undergo "AI generation
sanitisation." Both times this was treated as two separable asks:

- **Accepted:** writing in clean, direct, non-cliche prose and using plain commit
  messages / READMEs without bot co-author trailers or "Generated with Claude Code"
  branding. This is normal presentation hygiene, not deception.
- **Declined:** taking active steps to make AI-assisted graded coursework *look*
  human-authored in order to defeat an institution's AI-content detector, or otherwise
  misrepresent the authorship of this assignment to a grader. The underlying pipeline
  code, report, and this notebook were built with AI assistance across sessions; that
  fact isn't something to be laundered out of a submission.

If a future session gets a similar request, apply the same split rather than either
refusing outright or complying fully.

## Deliverable format

Python scripts per task (`scripts/01_...` through `scripts/05_...`) plus a
separate written report (`report/report.md`), per explicit instruction —
not a single notebook.
