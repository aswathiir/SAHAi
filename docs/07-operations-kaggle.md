# SAHAi — Kaggle Run Guide

How to run GRPO training on a Kaggle T4 and how to read the result.

Training needs a GPU, so Kaggle is the only place this runs. Locally you can run
the torch-free tests (`pytest tests/`) and nothing else.

---

## 1. Push the source

The `sahai/` package is uploaded to Kaggle as a **dataset**, not committed into
the notebook. `kaggle_upload/` is that dataset's contents.

> **Run every command in this section from the repo root**, not from inside
> `kaggle_upload/`. The `-p` flag is relative to your current directory, so
> `-p kaggle_upload` from inside `kaggle_upload/` looks for
> `kaggle_upload/kaggle_upload/` and fails with
> `Metadata file not found: dataset-metadata.json`.
>
> ```bash
> cd "/Users/aswathiranjith/Documents/sem 7/rl/SAHAi"
> ```

After **any** change to `sahai/`, re-sync before uploading — this is the step
that is easiest to forget, and skipping it means you are running the old code:

```bash
rsync -a --delete --exclude='__pycache__' sahai/ kaggle_upload/sahai/
```

Check the layout before uploading. `dataset-metadata.json` must sit at the root
of the `-p` directory — not inside `sahai/` — and there must be no nested
`kaggle_upload/kaggle_upload/`:

```bash
ls kaggle_upload
```

Expected, exactly:

```
dataset-metadata.json   problems/   pyproject.toml   sahai/
```

If `dataset-metadata.json` is missing from that listing, find it before
re-creating it — it is usually one directory too deep:

```bash
find kaggle_upload -name dataset-metadata.json
```

Then upload. **`--dir-mode zip` is required.** It defaults to `skip`, which
silently ignores every directory, uploads only the loose files, and still
reports `Upload successful` — producing a dataset with no `sahai/` package in
it. Watch for `Skipping folder: sahai` in the output; that means the flag is
missing and the upload is useless.

First time:

```bash
kaggle datasets create -p kaggle_upload --dir-mode zip
```

Every subsequent time:

```bash
kaggle datasets version -p kaggle_upload -m "your message" --dir-mode zip
```

A good upload prints `Starting upload for file sahai.zip` and
`problems.zip`. Kaggle expands the archives when mounting, so the package
appears at `/kaggle/input/sahai-source/sahai/`.

The slug comes from `kaggle_upload/dataset-metadata.json` (`sahai-source`), so
it mounts at `/kaggle/input/sahai-source/`. The notebook auto-detects whichever
attached dataset contains a `sahai/` directory, and prints what it found. If it
prints a WARNING listing your attached datasets, the dataset is either not
attached to the notebook or has the package nested one level too deep.

## 2a. Accelerator — set this in the browser, not the CLI

**The accelerator cannot be set from the CLI.** `machine_shape` in
`kernel-metadata.json` and `kaggle kernels push --accelerator gpu-t4x2` are both
accepted without error and then ignored by the server; the run still lands on a
**P100**. The valid enum is not published in `kagglesdk` (the source says so in a
comment), so there is nothing to guess at reliably.

A P100 cannot run this stack at all — two independent blockers:

- Kaggle's PyTorch build ships no `sm_60` kernels (`sm_70+` only), so every
  CUDA op fails with `no kernel image is available for execution on the device`.
- `bitsandbytes` 4-bit, required by `student_quantize_4bit=True`, needs `sm_75+`.

Set it once in the browser:

1. Open <https://www.kaggle.com/code/aswathiranjith/sahai-grpo-training>
2. Right sidebar → **Session options → Accelerator → GPU T4 x2**
3. **Save & Run All**

The notebook fails fast with this message if it ever lands on Pascal again, so a
misconfigured run costs about five minutes rather than a silent GPU-hour.

> **Careful:** a later `kaggle kernels push` may reset the accelerator back to
> the server default. After a CLI push, re-check the sidebar before running.

## 2. Push and run the notebook

Preferred: push from the CLI. `notebooks/kernel-metadata.json` already sets the
GPU, internet, and the `sahai-source` dataset attachment, so there is nothing to
configure in the browser and the notebook cannot drift from the repo.

```bash
cd notebooks && kaggle kernels push
```

**This starts the run** — `push` uploads *and* executes, consuming GPU quota.

```bash
kaggle kernels status aswathiranjith/sahai-grpo-training
kaggle kernels output aswathiranjith/sahai-grpo-training -p kaggle_results
```

`output` pulls `rollouts/epoch-*.json`, `metrics.json`, `eval_report.json` and
the adapter down locally, which is the fastest way to inspect a run.

Editing the notebook in the browser is what caused the repo and Kaggle copies to
diverge previously (the `final_model` path differed). If you do edit in the
browser, port the change back into `notebooks/train_kaggle.py` **and** the
`.ipynb`.

### Manual alternative

Upload `notebooks/train_kaggle.ipynb`, then in the Kaggle sidebar:

- **Accelerator: GPU T4 x2** (or P100). Without this, `load_for_training` puts
  everything on CPU and a single epoch will not finish in a session.
- **Internet: On.** Required — the run downloads Qwen2.5 weights from HuggingFace
  and the MBPP dataset.
- **Persistence:** leave off; everything you need is written to `/kaggle/working`
  and captured in the notebook output.

Attach the `sahai-source` dataset via *Add Input*.

## 3. Dependencies

Cell 1's `!pip install` is commented out because Kaggle preinstalls most of it.
Two are **not** reliably present and are not in `pyproject.toml`:

```bash
pip install -q bitsandbytes datasets
```

`bitsandbytes` is required because `Settings.kaggle()` sets
`student_quantize_4bit=True`; `datasets` is required by `load_mbpp`. Uncomment
the install line if either import fails.

## 4. Run

Run all cells. Section 4 asserts the code verifier can execute a known-good MBPP
solution — if that assert fires, stop, because every `r_sol` for the whole run
will be 0 and the solve signal is meaningless.

Expect roughly 8–10 minutes per epoch at the default `Settings.kaggle()`
(`batch_size=2 × group_size=4` = 8 rollouts/epoch, `epochs=3`). Most of that is
rollout generation and the sandboxed test execution, not the gradient step.

---

## 5. Reading the output

### Where things land

Under `/kaggle/working/output/`:

| Path | What it is |
|---|---|
| `rollouts/epoch-N.json` | **Every dialogue** with per-rollout `r_sol`, `r_ped`, `leakage`, `reward`, `advantage` |
| `metrics.json` | Per-epoch aggregates (loss, kl, reward, solve, ped, leak) |
| `eval_report.json` | Held-out scores: `solve_rate`, `ped_acceptance`, `leakage_rate`, `per_difficulty` |
| `checkpoint-eN-sM/` | LoRA adapter + tokenizer, written every `checkpoint_every=50` steps and once at the end |
| `final_model/` | Final LoRA adapter + tokenizer |

**The saved model is a LoRA adapter, not a full model.** It is a few MB —
`adapter_model.safetensors` + `adapter_config.json`. To use it you need the base
model plus the adapter:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base, "/kaggle/working/output/final_model")
```

Download `output/` before the session ends. `/kaggle/working` is wiped when the
session dies.

### Section 9b — the part that actually tells you something

Section 9b prints the best and worst dialogue per epoch with its reward
breakdown. Read that before the metrics table; aggregates hide the failures that
matter.

**Check these four things, in order:**

**1. Reward spread.** The header prints
`reward min=… max=… spread=…` per epoch. If you see

```
!! zero spread -> all advantages 0 -> this epoch taught the model nothing
```

that epoch contributed no gradient at all, whatever the loss printed. Every
reward in a group being identical drives `(r - mean)/std` to 0 for all of them.
This is what made the previous run's `loss=0.0000` — it was not a rounding
artifact. The training log also warns `N/M groups had zero reward spread`.

**2. Do turns end mid-sentence?** Look at the last characters of each
`[TUTOR]` / `[STUDENT]` block. A turn ending like

> `...We count occurrences of each character within the`

means truncation is still happening, and the next turn will *continue* that
sentence rather than reply to it. If you see this, raise `max_new_tokens` on
`TutorPolicy` / `StudentSimulator` (currently 192 / 160).

**3. Is there code in `[TUTOR]` turns?** Any ``` fence or `def`/assignment lines
in a tutor turn is leakage the tutor should be penalized for. `r_ped` for that
rollout should be ≤ 0.5. If a tutor turn contains code *and* `r_ped` is high,
the judge is not catching it — that is a bug, report it.

**4. Are the roles right?** `[STUDENT]` should be short, confused, asking
questions. `[TUTOR]` should be asking one guiding question. If the student is
writing solutions and the tutor is correcting them, the roles have inverted —
the 0.5B student is not holding its persona and needs to move to 1.5B.

### Metric sanity

`solve` is quantized by construction: with `num_solve_samples=4` and 8 rollouts,
it moves in units of 1/32 = 0.031. A reading of `0.0312` is **one** passing
attempt out of 32, i.e. noise, not a solve rate.

`ped` is no longer coarsely quantized. Its three checks are scored as
the *fraction of tutor turns* that pass, so a per-rollout score can land on any
value in [0,1] (three clean turns and one with code scores 0.75 on that check,
not 0). Before that change the checks failed the whole dialogue if any one
turn tripped them, which made `ped` slide down purely with dialogue length —
measured 0.720 / 0.639 / 0.537 / 0.453 for 1 / 2 / 3 / 4 tutor turns — and
taught the policy to end conversations early. Expect higher and less
length-correlated `ped` values from runs after that fix; they are not
comparable to earlier runs.

`leak` has a high floor. The estimator counts any word overlap between tutor
text and the reference solution after removing Python keywords, and it does
**not** exclude words from the problem statement. A tutor asking a perfectly
clean question about a string problem still scores non-zero. Treat the trend
across epochs as meaningful and the absolute value as not.

`kl` can print negative. `mean(log π − log π_ref)` is not a valid KL estimator;
this is cosmetic at `kl_coeff=0.05` but do not report the number as a KL.

---

## 6. Scale

The defaults are a smoke test, not a training run: 3 epochs × 8 rollouts = **24
dialogues total** and 3 optimizer steps. Nothing in a run that size is
distinguishable from sampling noise.

Once section 9b shows clean dialogues, scale up in `Settings.kaggle()`:

```python
training=TrainingSettings(
    group_size=8,      # more within-group spread -> fewer dead groups
    batch_size=4,
    epochs=30,
    ...
)
```

Raise `group_size` before `epochs`. Larger groups are what stop the advantage
collapse; more epochs over degenerate groups just burns GPU hours. Budget
~10 min/epoch at the current size and watch the 12-hour session limit.

---

## 7. Known-unfixed

Carried forward deliberately, in rough priority order:

1. **Leakage metric is inflated** — does not exclude problem-statement tokens.
2. **0.5B student breaks persona** — the likely cause of role inversion.
3. **No importance ratio / clipping** — `clip_epsilon` is in settings but unused;
   the update is REINFORCE, not clipped GRPO. Acceptable for one inner epoch,
   but do not call it GRPO in a writeup.
4. **KL estimator** — should be the k3 form `exp(δ) − δ − 1`.
5. **`_build_tutor_mask` is O(n²)** — re-tokenizes every message prefix.
