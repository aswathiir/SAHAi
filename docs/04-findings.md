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

## 8. The pedagogy reward penalised dialogue length, not bad teaching

**Symptom.** Across three runs, held-out `ped_acceptance` fell monotonically
(80% → 71% → 68%) while solve rate rose. It read like a real trade-off: the
tutor buying solve rate at the cost of teaching quality.

**Cause.** Four of the five rule checks scanned *every* tutor turn and failed
the whole dialogue if any one turn violated them. The probability of passing a
conjunctive check therefore decays with turn count, independent of quality.
Measured over the 304 rollouts of the v11 run:

| tutor turns | n | mean `r_ped` | mean reward |
|---|---|---|---|
| 1 | 102 | 0.720 | −0.437 |
| 2 | 46 | 0.639 | −0.631 |
| 3 | 38 | 0.537 | −0.796 |
| 4 | 118 | 0.453 | **−1.011** |

Perfectly monotonic. GRPO reads that as *"shorter dialogues teach better"* and
optimises for ending the conversation — a property of the scoring function, not
of teaching. 33.6% of dialogues ended after a single tutor turn, and those
scored the best rewards in the run.

**Fix.** Score those four checks as the **fraction of tutor turns that pass**
rather than all-or-nothing. `_tutor_asks_questions` was left alone: it is a
ratio with a threshold and was never length-biased.

**Measured.** Replaying all 304 real v11 dialogues through both versions, the
spread in mean `r_ped` across dialogue lengths fell from **0.267 → 0.044** — a
6× reduction. The residue is real signal, not artifact: longer dialogues in that
run genuinely did contain more code. Two regression tests now pin
length-neutrality.

**Caveat.** `ped` values after this change are **not comparable** to values from
earlier runs.

---

## 9. Twenty epochs of training changed the tutor's behaviour by nothing

**Symptom.** Reward, solve rate and leakage all moved between epochs, so the run
looked alive.

**Cause.** They were moving for other reasons. Measured directly on tutor turns
in the v11 run:

| | epochs 0–4 | epochs 15–19 |
|---|---|---|
| turns containing code | 30.5% | 30.2% |
| turns asking a question | 11.7% | 12.6% |
| avg tutor turns / dialogue | 2.66 | 2.77 |

Nothing moved. KL to the reference policy finished at **0.005** — the policy was
still essentially the base model. A 20-epoch run performs only **160 optimizer
steps** at `lr=2e-5` on 0.14% of parameters (LoRA rank 8), and the loss divides
by token count (mean, not sum, log-prob), shrinking the effective step further.

Gradients *were* flowing — KL grew monotonically, so the graph is intact. The
steps were simply too small to move a 1.5B model.

**Fix.** `learning_rate 2e-5 → 1e-4`. 2e-5 is a full-finetune-scale rate; LoRA
adapters are normally trained at 1e-4–3e-4. `max_grad_norm=1.0` clips every step
and the KL penalty starts to bite once KL is non-trivial — at 0.005 it
contributed ~0.00025 to the loss, i.e. nothing.

**Measured.** Pending the next run. Watch KL: if it exceeds ~0.1 or reward
collapses, the rate is too hot.

**Why this matters most.** Much of what earlier runs read as "training progress"
was never the policy improving.

---

## 10. Epoch metrics measured problem difficulty, not tutor skill

**Symptom.** Solve rate swung between 0.00 and 0.52 with no trend; every curve
looked like noise.

**Cause.** `batch_size=2` — each epoch drew **2 problems** from a 198-problem
bank via unstratified `random.sample`, so *which* problems were drawn dominated
the epoch mean. Per-problem breakdown of the run's best epoch (13, solve=0.516):

- modulo function → **0.00**
- rhombus perimeter (a one-line `4 × side`) → **0.88**

One trivially easy draw made it the best epoch of the run. The same modulo
problem appeared in epochs 0 and 13 and scored 0.00 both times — problem
identity, not policy state.

**Measured.** If the true solve probability were constant and all variation were
sampling noise (64 Bernoulli draws/epoch), expected epoch-to-epoch σ would be
**0.042**. Observed σ was **0.156** — **3.7×** the noise floor.

Also found: epochs 5 and 6 drew only **1** problem, not 2, because `zpd_sample`
falls back to `min(n, len(candidates))` when fewer than `batch_size` problems
sit in the 30–70% mastery band. Sample size was not even constant.

**Fix.** `batch_size 2 → 4`, `epochs 20 → 10` — identical total rollouts (320),
identical optimizer steps (160), identical ~8.4h wall clock, measured over twice
as many problems per epoch. 20 epochs at `batch_size=4` would have run ~16.7h
and been killed by Kaggle's 12h cap around epoch 14.

---

## 11. Corrections to our own measurements

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

## 11. The question-fraction change cost a run (reward hacking, not overfitting)

Making `_tutor_asks_questions` the per-turn fraction instead of a 30% threshold
was meant to give an inert check a gradient. It did, and the policy optimised
it directly.

| | before (v15) | after |
|---|---|---|
| held-out solve | **16.3%** | **6.2%** |
| training mean r_ped, final epoch | — | 0.992 |
| training mean reward, final epoch | — | +0.168 (first positive) |

Reward went up while the thing reward exists to produce went down. That
divergence is the signature, and it is **not** classical overfitting: training
solve averaged 0.144 over epochs 2-8, so nothing was memorised. Epoch 9's
0.367 is a single-epoch outlier on four problems that makes the curve look
like success.

**Mechanism.** A question mark is cheap:

    "Hash maps? Kaise use kar sakta hu?"   ->  r_ped 1.00

Seven words, no content, perfect score. The exploit exists under *both* forms
— every other check passes vacuously on a short clean turn — so the fraction
did not create it. What the fraction created was the **pressure to use it**. In
a four-turn dialogue:

    questions   threshold   fraction
        2          1.00       0.90
        4          1.00       1.00

Under the threshold, converting the remaining substantive turns into questions
gains nothing. Under the fraction it is worth 0.10 of r_ped — and because
reward carries `(r_ped - 1) * lambda`, closing that gap does not merely score
well, it cancels the pedagogy penalty outright. Every teaching turn left
un-questioned was costing reward.

**A content gate does not work.** Tried before reverting: the gamed turns hold
3-12 content words and genuinely good questions 4-13. They overlap completely.
There is no counting rule that separates them.

**The pattern across four attempts.** Every rule-based pedagogy fix has been
gamed within one run:

| rule | how it was gamed |
|---|---|
| all-or-nothing | length bias — end the conversation early |
| per-turn fraction of "never do X" | fixed that, exposed the next |
| question threshold | inert, no gradient |
| question fraction | contentless questions |

Rule-based pedagogy scoring has reached its ceiling. The judge exists to avoid
an LLM judge's GPU cost; that saving is now the binding constraint rather than
a saving. The open choice is to pay for an LLM judge, or to reduce pedagogy to
the part that is execution-verifiable (code and solution leakage) and let
`r_sol` carry the signal.

**Also seen in this run:** epoch 5 drew 2 problems instead of 4 — the
`zpd_sample` shortfall, now in three separate runs — and at batch_size 4 over
10 epochs the run revisits problems, visible as a repeat across epochs 6 and 8.
