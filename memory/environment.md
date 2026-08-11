# Environment Notes

Machine: Rocky Linux 9.5, 16 CPU cores, 30GB RAM, ~18GB free disk, **no GPU**
(`nvidia-smi` not installed). `torch 2.6.0+cpu`. Python 3.9.21 (harness warns
this is nearing EOL, but it's what's available).

## No GPU — drives the PEFT method choice

Confirmed via direct check, not assumed (`nvidia-smi: command not found`).
This is why LoRA was used instead of QLoRA (needs GPU 4-bit quant) — see
`decisions.md`. Benchmarked step time empirically before committing to a
training config; see hyperparameters documented in
`scripts/03_finetune_lora.py`.

## `transformers`/`trl` import crash: Keras 3 vs. TensorFlow bridge

This environment has TensorFlow with Keras 3 installed. Importing
`transformers.Trainer` or `trl.DPOTrainer` (even for a pure-PyTorch workflow)
triggers transformers' TF integration import path, which fails with:

```
ValueError: Your currently installed version of Keras is Keras 3, but this
is not yet supported in Transformers. Please install the backwards-compatible
tf-keras package...
```

Fix used: force transformers to skip TF entirely by setting
`os.environ["USE_TF"] = "0"` (and `TRANSFORMERS_NO_TF=1`) **before**
`transformers`/`trl` are imported anywhere in the process. This is set at the
top of `scripts/utils.py`, which is why every script imports `utils` before
importing `transformers`/`trl`/`peft` — import order matters here, and this
bit us once (`03_finetune_lora.py` originally imported `transformers` before
`utils`, which silently missed the env-var fix; had to reorder imports).

## No `git` binary, no root/sudo access

`git` is not installed, and `sudo dnf install -y git` fails — the `cloud`
user is in the `wheel` group but sudo requires a password we don't have and
there's no passwordless sudo configured.

Workaround: used **`dulwich`** (pure-Python git implementation,
`pip install --user dulwich`) for all git operations — init, add, commit,
branch rename, and push. Pushing over HTTPS with a GitHub PAT works by
passing `remote_location=f"https://{token}@github.com/owner/repo.git"`
directly to `dulwich.porcelain.push(...)` (not persisted into any git config
file, since none exists). See `memory/progress.md` for the actual push
history/troubleshooting.

If a future session has real `git` available, prefer it — dulwich was a
necessity here, not a preference.

## Package installs needed (not preinstalled)

`python-docx`, `transformers`, `datasets`, `peft`, `trl`, `accelerate`,
`sentencepiece`, `evaluate`, `rouge_score`, `sacrebleu`, `dulwich` — all
installed via `pip install --user` (no venv used, installs go to
`~/.local/lib/python3.9/site-packages`).
