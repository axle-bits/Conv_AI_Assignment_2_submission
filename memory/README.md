# Project Memory

This folder is a durable, human-readable record of the context, decisions, and
environment quirks behind this PS2 (Parameter-Efficient Fine-Tuning + Human
Preference Alignment) assignment pipeline. It's meant to let anyone — a
teammate, a grader, or a future session picking this repo back up — get
oriented without re-deriving decisions from scratch.

- [`decisions.md`](decisions.md) — what was chosen (domain, dataset, model,
  PEFT method, preference-alignment method) and why, including the
  alternatives that were ruled out.
- [`environment.md`](environment.md) — the actual compute environment this
  was built and trained on, plus non-obvious workarounds required to get
  the toolchain working here.
- [`progress.md`](progress.md) — running status of each of the 5 assignment
  tasks, updated as the pipeline progresses. Check this first to see what's
  done vs. still in flight.

Source of truth for the assignment spec itself is
`ConAI_SPS_Assgn2_PS2.docx` in the repo root.
