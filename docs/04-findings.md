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

### 11b. What replaced it: pedagogy cut to answer-giving only

`r_ped` now averages three checks — no code blocks, no solution patterns, no
dangling `:` — and scores nothing about teaching style. The question and
length checks are gone.

The line applied, since "execution-verifiable" is imprecise (execution lives in
the leakage term):

> keep a check when satisfying the rule and achieving the goal are the same
> act; drop it when the rule is a proxy that can be satisfied without the goal.

"Do not write code" has no fake version. "Ask questions" does, and the policy
found it in one run. "Under 200 words" was arbitrary and was the vector for the
original length bias.

After the change, these all score `r_ped` 1.0:

| turn | before | now |
|---|---|---|
| `"Hash maps? Kaise use kar sakta hu?"` (the exploit) | 1.00 | 1.00 |
| `"What structure gives you O(1) lookup?"` | 1.00 | 1.00 |
| `"A hash map gives O(1) lookup on the key."` | 0.80 | 1.00 |
| a 250-word substantive turn | 0.80 | 1.00 |

The exploit and a real question are now indistinguishable **on purpose**. The
reward has no opinion about which is better teaching, because every opinion it
held was gamed. `r_sol` decides, by whether the student solves the problem
afterwards.

**What this gives up.** Nothing rewards Socratic behaviour any more. If the
tutor drifts back to lecturing, `r_ped` will not object — only a fall in solve
rate will. That is the trade: a weaker but honest signal instead of a strong
one pointing the wrong way. The alternative on the table remains an LLM judge,
which costs GPU time in the rollout loop.

**What to watch next run.** `r_ped` should sit near 1.0 for most rollouts and
carry little variance — that is expected, not a bug, and it means group
variance now comes from `r_sol` and `leak`. The number to judge the run on is
held-out solve rate against the 16.3% best.

## 12. The held-out eval could not resolve the runs it was judging

Every run-to-run conclusion in this document up to entry 11 was drawn from a
20-problem held-out set. Bootstrapping that mean:

| true solve rate | 20-problem estimate |
|---|---|
| 0.10 | sd 0.068, 95% of runs land in [0.00, 0.25] |
| 0.16 | sd 0.082, 95% of runs land in [0.00, 0.35] |

The entire spread across six runs — 6.2% to 16.3% — is about **1.3 standard
deviations**. In problem terms: the best run solved ~3.26 problem-equivalents
and the worst ~1.25. **Every comparison rested on roughly two problems.**

This does not overturn entry 11: the reward-hacking *mechanism* was verified
directly, by scoring the run's own turns (`"Hash maps? Kaise use kar sakta
hu?"` -> r_ped 1.00) and by training r_ped saturating at 0.992. What it
overturns is the *magnitude*. The claim that the change cost 10 points of
held-out solve was not supportable, and was made anyway.

**Fixed:** `Settings.eval_problems`, now 60. sd falls to ~0.047 and the cost is
~89 min at the measured 89 s/problem. Training is 6.0 h of the 12 h cap, so it
fits with headroom. A test pins both the size and the total budget.

**The pattern this exposes.** Sorting the six runs by what kind of change they
made:

| kind of change | examples | outcome |
|---|---|---|
| removed a distortion | truncation cap, 1.5B student, leakage eval bug, rare-token filter, length bias | every gain came from here |
| added an incentive | question fraction | the only clear regression |

Worth carrying: gains have come from deleting things that bent the signal, not
from adding things that point at good behaviour. The pedagogy narrowing in 11b
is a deletion, which is the category that has worked.

**Also worth not repeating:** v14 changed learning rate, batch size and
pedagogy scoring in one run and came out net negative, so none of the three can
be attributed. One variable per run.

## 13. `zpd_sample` shrank the batch instead of topping it up

Seen in three separate runs and dismissed each time as a curiosity in the log:

    Epoch 5: rollout on 2 problems x G=8

`batch_size` is 4. That epoch trained on half the rollouts and contributed half
the gradient, and nothing said so beyond a number in a routine INFO line.

The cause:

```python
candidates = [p for p in self.problems if tracer.in_zpd(...)]
if not candidates:            # only fires when the band is COMPLETELY empty
    candidates = self.problems
return random.sample(candidates, min(n, len(candidates)))   # <- silently short
```

With one to three problems inside [0.3, 0.7] and `n=4`, the fallback never
fired and `min()` returned a short batch. The band thins naturally as mastery
moves — every skill the learner masters leaves the band — so this gets *more*
likely as a run progresses, which is exactly when the gradient matters most.

**Fixed** by topping the batch up to `n` with the problems nearest the band,
rather than random ones: "just outside the ZPD" is the best available
substitute for "inside it", where the alternative is padding a thin epoch with
work the learner has either mastered or cannot touch. Ties are broken randomly
so a short epoch does not always draw the same problems.

The trainer now logs a WARNING naming the band size whenever it tops up. The
batch no longer shrinks, so without that line the padding would be invisible —
which is how the original went unnoticed for three runs.

> **Superseded — the top-up was treating a symptom.** The next run's log showed
> `Only 0 problems in the ZPD band` at epochs 4, 5, 6, 7 and 9: the band was not
> thin, it was empty, and permanently. See finding 13b.

### 13b. The ZPD band was never a difficulty band, and empties for good

`BKTTracer.in_zpd(difficulty, skills)` took `difficulty` as an argument and
**ignored it**, selecting purely on `zpd_low <= mean(mastery) <= zpd_high`.
Two facts then combine:

* `p_init` is 0.3 and `zpd_low` is 0.3, so an **untouched** skill sits exactly
  on the boundary and passes. The band therefore meant "problems using skills
  the learner has never attempted" — an exploration frontier, not a difficulty
  band.
* BKT here has no forgetting transition, and at this student's competence a
  failed skill converges to a fixed point of **0.109** within three
  observations. Simulating the real update rule at the measured ~15% solve
  rate: `0.30 → 0.15 → 0.12 → 0.11 → 0.11 …`, far below `zpd_low`.

So the band holds the entire bank at epoch 0, and once every skill has been
touched — 4 problems/epoch over 15 skills, so by about epoch 3 — it is empty
and stays empty. Seven of the ten epochs in the 2026-09-12 run were drawn by
the top-up ordering rather than by any curriculum, which is why
gcd-by-recursion is the best rollout in three of the last six epochs.

**Fixed** by making difficulty the primary axis, which is what the argument was
always for:

* `target_difficulty() = 1 + ability · (max − 1)` maps the BKT probability onto
  the bank's own 1..5 ordinal scale;
* `in_zpd` is `|difficulty − target| <= 1`, plus mastery as a **ceiling** only.
  The old mastery *floor* is gone deliberately: a skill at 0.11 means the
  learner is failing it, which is when they should be handed something easier,
  not excluded from practice permanently;
* `zpd_sample` draws **at random within the window**. Ranking by
  `abs(difficulty − target)` was tried first and simulated badly — at a target
  near 2.0 every difficulty-2 problem scores 0.0 and every difficulty-1 problem
  scores 1.0, so all ten epochs drew difficulty 2 and the 186 easier problems
  were never seen. That is the original bug on a different axis.

Simulated over the measured bank (difficulty 1:186 2:59 3:36 4:9 5:6) at a 14%
solve rate, the band now holds 95–245 problems at every epoch instead of
collapsing to 0, the drawn difficulty mix is 1/2/3 rather than one level, and
35 of 40 draws are distinct problems.

This is a real defect and worth fixing, but note its size: it decides *which*
problems are trained on, not whether training can work at all. For the latter,
see finding 14.

---

## 14. The root cause under all thirteen findings: only half the reward can teach

Six runs, six diagnoses, and held-out solve rate has never left the 6–16% band.
Findings 1–13 are all real and all fixed, and none of them moved the number.
That pattern is itself the evidence: we have been repairing the wrong thing.

### The measurement

`r_SAHAI = (r_sol − ability) + (r_ped − 1)·λ − γ·L` has two kinds of term.

* `r_ped` and `L` are **deterministic functions of the tutor's own text**. The
  tutor fully controls them; identical text always scores identically.
* `r_sol` is a **4-sample Bernoulli estimate** of whether a different, frozen,
  4-bit-quantised 1.5B model writes passing code after reading the dialogue.
  The tutor influences it weakly, indirectly, and through another model's
  sampling.

Correlate each against epoch index, across the three completed 10-epoch runs:

| run | corr(epoch, r_ped) | corr(epoch, leak) | corr(epoch, r_sol) |
|---|---|---|---|
| A (rare-token leakage) | +0.80 | −0.64 | +0.48 |
| B (question fraction)  | +0.99 | −0.80 | +0.54 |
| C (pedagogy narrowed)  | +0.64 | −0.48 | **+0.01** |

The two deterministic terms move strongly and in the intended direction in
**every** run. The outcome term does not move in any of them — within-run
epoch-to-epoch sd of `r_sol` is ~0.10, larger than any trend it might carry.

The rollouts still on disk say the same thing at the level GRPO actually
operates on. Mean **within-group** variance (the only variance that survives the
z-score, so the only thing that becomes gradient):

```
r_sol 0.0098      r_ped 0.0482      leak 0.0144
```

`r_ped` supplies five times the usable signal of the term the project exists to
optimise. 22 of 24 rollouts scored `r_sol = 0.00` exactly.

### Why this explains everything else

A policy optimises whatever part of its reward it can actually control. `r_ped`
and `L` are controllable and noiseless; `r_sol` is mostly another model's dice.
So every run, the policy drives the rule-based terms to their limit and treats
the outcome term as noise — which is precisely what the logs show.

Re-read the project history with that in mind:

* Every gain came from **removing a distortion from the rule-based channel**
  (truncation cap, length bias, the eval/training leakage mismatch, rare-token
  filtering). Correct — that is the only channel carrying gradient.
* The one attempt to **add** an incentive (finding 11, the question fraction)
  was gamed inside a single run. Also correct, and predictable: it was added to
  the controllable channel, which is exactly the channel a policy will exploit.
* Narrowing pedagogy to three answer-giving checks (11b) recovered the number.
  That is not learning either — it removed a distortion from the same channel.
* **Teaching that causes a student to solve problems has never been trained at
  all.** It has no usable gradient to train on.

The trap, stated plainly: we keep tuning the channel that works and hoping the
channel we care about follows it. It does not, and it will not, while `r_sol`
is a four-sample coin flip.

### Three defects that follow from reading the update

1. **It is not GRPO.** `_policy_update` computes `−advantage · mean_log_prob`.
   There is no importance ratio and no clipping; `clip_epsilon = 0.2` is
   declared in `settings.py` and referenced nowhere in the codebase. This is
   REINFORCE with a z-scored baseline.
2. **And it is off-policy REINFORCE.** 32 rollouts are collected under policy
   θ_k, then 16 sequential optimizer steps are taken (`grad_accum = 2`). Steps
   2–16 use samples from a policy that no longer exists, uncorrected — which is
   what the ratio and clip exist to handle. The symptom is in every run: KL to
   the reference climbs monotonically (0.002 → 0.026) while nothing improves.
3. **Partial credit is discarded.** `SolveReward.compute` counts a sample only
   when `verify()` returns exactly 1.0, so a student attempt passing 4 of 5
   tests scores the same zero as one that does not parse. The verifier already
   returns the fraction; the reward throws it away. This is the single largest
   avoidable loss of resolution in the term that most needs it.

### The fix this implies

Make `r_sol` a **deterministic function of the dialogue**:

* decode the student's solution attempt greedily (`do_sample=False`, one
  sample) so that within-group variance in `r_sol` comes from what the tutor
  said rather than from the student's sampling;
* score it as the **fraction of tests passed**, not a strict all-or-nothing
  gate, restoring the resolution lost in (3).

Together these convert `r_sol` from a five-level noisy estimate to a continuous
attributable one, and they cost *less* compute, not more: the reward phase is
dominated by four 512-token student generations per rollout, so one greedy
sample cuts the measured ~35 min/epoch reward phase to roughly a quarter of
itself. That pays for the 60-problem eval outright (~9.6 h → ~7.3 h total).

What this does **not** do is add a new reward term. That is deliberate. Adding
terms to the controllable channel is the move that has failed twice.

### Applied

Both, plus the two update defects:

* `StudentSimulator.attempt_solution` decodes greedily (`do_sample=False`), and
  `SolveReward` draws once instead of four times.
* `SolveReward.compute` returns the **mean fraction of tests passed** as the
  reward, and the all-or-nothing rate alongside it as `solved`. Training logs,
  rollout dumps and the eval report now carry both — `solve` is the shaped
  signal the policy optimises, `solved` is the series comparable to every run
  back to v6. Held-out results must be read on `solved`, or the partial credit
  will look like a gain that is not one.
* `_policy_update` uses the clipped surrogate
  `min(rho_t·A, clip(rho_t, 1±eps)·A)` with `rho_t` against log-probs cached
  from before the first optimizer step. `clip_epsilon` had been sitting unused
  in `settings.py` since the beginning.
* `lora_dropout` 0.05 → 0.0 **on Kaggle**. Rollouts generate under
  `model.eval()` and the update forward runs under `model.train()`, so with
  dropout on, the two sides of the importance ratio are different functions.
  Measured on a trained adapter, dropout alone moved `|rho − 1|` as far as
  0.065 — a third of the clip band — which would fire the clip on sampling
  noise. The KL penalty to the frozen reference is doing the regularising.

`scripts/check_policy_update.py` exercises the real update path on a tiny
randomly-initialised model: the surrogate's clip asymmetry, `rho == 1` before
the first step, the gradient's sign in both directions, no active dropout, and
that later rollouts really are off-policy (after four stale steps, 15 of 32
tokens land outside the clip band). Worth having with no GPU quota left — a
sign error would otherwise cost a nine-hour run to find.

Budget after both changes: the reward phase falls from ~35 to ~14 min/epoch,
so training is ~6.1 h and a 60-problem greedy eval ~0.6 h — **~6.9 h** of the
12 h cap including the extra forward pass the ratio needs, against 9.6 h before.
