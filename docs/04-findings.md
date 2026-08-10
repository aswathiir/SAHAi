# Empirical findings

Every entry here came from inspecting run artifacts or driving the live system —
not from reading code. Each is stated as **symptom → cause → fix → measured
effect**, because the measurement is what makes it a finding rather than an
opinion.

---

## 1. A binary pedagogy reward produces exactly zero gradient under GRPO

**Symptom.** Epoch 0 logged `loss=0.0000`.

**Cause.** A four-step chain:

1. Every rollout failed at least one rubric → `r_ped = 0` for all 8
2. `hard_penalty` mapped all of them to `r_SAHAI = −λ = −1.0`
3. Advantage is `(r − mean)/(std + ε)`; identical rewards ⇒ `std = 0`
4. Every advantage became 0. The epoch updated nothing.

**This is a structural incompatibility, not bad luck.** The paper's binary
`r_ped = ∏ 1[J_m = accept]` was designed for **PPO, which has a learned value
baseline** and tolerates constant rewards. GRPO has no critic — within-group
reward variance *is* the entire learning signal — so a binary conversation-level
reward collapses it whenever `G` is small.

**Fix.** Score the fraction of checks passed instead of all-or-nothing.

**Measured.** Reward spread `0.000 → 1.317`. Epoch-0 loss `0.0000 → 0.0562`.

This is the project's most defensible contribution so far: a concrete reason the
published reward design does not transfer to a critic-free algorithm.

---

## 2. Leakage scored 0.000 for a complete solution leak

**Symptom.** A tutor wrote a full working solution in its first reply. The
leakage metric returned **0.14** in the run, and 0.000 when isolated.

**Cause.** `token_match_leakage` compared tutor text to the *reference*
solution. The tutor used a dict; the MBPP reference used `enumerate`/`count`.
Zero token overlap. The metric was measuring **plagiarism of one particular
implementation**, not whether the answer was given away.

| Tutor behaviour | old | new |
|---|---|---|
| Complete working solution | 0.000 | **1.000** |
| Code block that fails the tests | 0.000 | **0.500** |
| Clean Socratic question | 0.140 | **0.000** |

**Fix.** Execution as the primary signal: extract tutor code, run it against the
problem's own tests. Code that passes *is* the answer, however it is written.

**Second-order bug found while fixing it.** The tutor named its function
`find_first_repeated_char`; the problem expected `first_repeated_char`. The
verifier could not call it, so a complete leak would still have read as zero.
`runnable_variants()` now aliases every defined function and re-runs. Verified
by executing it, not by reasoning about it.

**Also fixed the floor.** Problem-statement words are subtracted, so a tutor can
say "string" without being charged. `leak` is now interpretable in absolute
terms, not only as a trend.

---

## 3. Turns truncated mid-sentence get *completed* by the next speaker

**Symptom.** From the first run:

```
[STUDENT] ...We count occurrences of each character within the
[TUTOR]   character within the string. If we encounter...
```

**Cause.** A turn cut at `max_new_tokens` is appended verbatim as a chat
message. The next model sees an unfinished sentence and finishes it instead of
replying.

**Fix.** Detect whether generation stopped on EOS; if not, trim back to the last
sentence boundary. Raised the token budgets.

**Measured.** Truncation down to ~5% of turns (9 of 172).

**Related, and larger.** 19% of turns ended on a **dangling colon** — *"Here is
the implementation:"* — which invites the same completion behaviour without any
truncation at all. This is the mechanism behind role inversion. Trimmed on the
student side; penalised via a fifth pedagogy check on the tutor side, because
the tutor is the policy and cannot be post-processed.

---

## 4. Role inversion — tutor and student swap places

**Symptom.** The student writes solutions; the tutor reviews them and says "Well
done!".

**Measured over 172 turns:**

| | tutor | student | intended |
|---|---|---|---|
| turns containing code | 14% | 18% | 0% / 0% |
| turns asking a question | 37% | 16% | tutor ≈ 100% |
| mean words per turn | 66 | 56 | student 1–2 sentences |

5 of 24 dialogues opened with the *student* writing code.

**Partial fix.** The opening student turn is now scripted from the persona's
code-mixing templates. Confirmed working in the live run:

```
[STUDENT]: Kya hum strings logic use kar sakte hain yahan pe?
```

**Still unresolved.** The tutor still leaks and still fails to ask questions.
Likely root cause is model capacity — a 0.5B student cannot hold a persona
across four turns. Moving the student to 1.5B on the second T4 is the next
lever.

---

## 5. Silence scored 0.8 out of 1.0

**Found by** driving the live API by hand and asking why a number looked odd.

**Symptom.** Submitting with **no conversation at all** returned
`pedagogy_score: 0.8`.

**Cause.** Four of the five checks are "no X" checks. With no tutor turns they
all pass **vacuously** — there is nothing to violate them.

**Why it matters beyond display.** In training, a rollout where the tutor said
nothing would score 0.8 and be rewarded above average. The policy could learn
that saying less is better teaching. That is a reward-hacking hole, and the kind
that hides in aggregate metrics: pedagogy scores would rise while the tutor got
quieter.

**Fix.** No tutor turns ⇒ score 0.0.

---

## 6. Infrastructure failures that reported success

Five attempts were needed before the first successful training run. Three of
them failed while *looking* like they had worked — worth recording because they
cost hours and would otherwise recur.

| Failure | What it looked like |
|---|---|
| `kaggle datasets version` skips directories without `--dir-mode zip` | `Skipping folder: sahai`, then `Upload successful` — a dataset with no code in it |
| jupytext's `notebook_metadata_filter: "-all"` strips `kernelspec` | `ValueError: No kernel name found in notebook`, which sounds like a notebook format problem |
| `--accelerator gpu-t4x2` accepted by the CLI, ignored by the server | Run lands on a P100 whose `sm_60` the installed PyTorch cannot use; `torch.cuda.is_available()` returns `True` but every kernel launch fails |
| Kaggle ships `torchao 0.10.0`; peft requires `≥0.16.0` and **raises** instead of returning `False` | `get_peft_model()` dies even though the project never uses torchao |
| `kaggle kernels output` skips files newer locally | Old results silently survive and get re-analysed as if fresh |

---

## 7. Two service bugs that produced silent wrong behaviour

**API responded one exchange behind the database.** `expire_on_commit=False`
keeps the row in SQLAlchemy's identity map, so a re-query returns the stale
`turns` collection — `selectinload` does not overwrite already-loaded
attributes. Every status code was 200 and nothing errored. Fixed with
`populate_existing=True`, with a regression test that diffs the response against
the database.

**Gateway permanently `degraded`.** Its health check probed the executor, which
sits on an isolated network with no route to the gateway — by design. The check
was testing something architecturally forbidden. Now `session`, the only service
on both networks, reports the executor's health on the gateway's behalf.

---

## 8. Corrections to our own measurements

Recorded because the discipline matters more than the individual numbers.

- **Truncation was counted wrong twice** before it was right: 16 (counting
  closed code fences as truncated), then 42 (conflating dangling colons with
  truncation), finally 9. The definitive signal is whether generation hit EOS,
  which the rollout dump now records instead of inferring from text shape.
- **`r_ped` was displayed as `0`** by a `:.0f` format left over from when it was
  binary. The true value was 0.4. Derived from `r_SAHAI` and confirmed against
  `hard_penalty`, which would have forced exactly `−1.0` had it really been zero.
- **`machine_shape: "gpu-t4x2"`** was asserted as the fix when the SDK
  explicitly documents that valid values are not available to it. It cost a run.
