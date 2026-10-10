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

> **Superseded for the staged config.** These are the runs that discarded
> partial credit. The fix below reversed the ordering, and `r_sol` now carries
> the most within-group variance of the three terms. Measured numbers are under
> *Measured* at the end of this finding; do not plan against the figures in
> this block.

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

### Measured

The staged config ran on 2026-09-29 (kernel v24, 5.65 h: 4.75 h training,
53.6 min evaluating 60 held-out problems). Provenance confirmed by
`lora_dropout: 0.0` in the saved adapter config.

**The partial-credit prediction held, and the ordering reversed.** Mean
within-group variance, the only variance that survives the z-score:

| term | runs A–C (strict `r_sol`) | v24 (partial credit) |
|---|---|---|
| `r_sol` | 0.0098 | **0.0481** |
| `r_ped` | 0.0482 | 0.0150 |
| leakage | 0.0144 | 0.0552 |

`r_sol` went from supplying a fifth of `r_ped`'s usable signal to supplying
three times it. The claim above that "teaching that causes a student to solve
problems has never been trained at all — it has no usable gradient to train
on" is no longer true as stated: it now has the largest gradient of the three
terms. `r_ped`'s variance collapsed for the opposite reason, saturation: 234 of
320 rollouts (73%) scored exactly 1.0.

**Tutor behaviour moved for the first time.** Across ten epochs, tutor turns
containing a question went 10% → 61% and turns containing code went 27% → 8%,
mean turn length 574 → 393 characters. Compare v11, where the same measurements
went 11.7% → 12.6% and 30.5% → 30.2%: nothing. The learning-rate raise from
2e-5 to 1e-4 is the plausible cause, and group reward spread stayed at
sd 0.18–0.58 throughout, so there was real signal to learn from.

**Held-out solve still did not follow.** Final evaluation on 60 problems:

| Solve | Partial | Ped | Leak | Reward |
|---|---|---|---|---|
| 0.117 | 0.150 | 1.000 | 0.004 | −0.152 |

Per-epoch training solve has no trend — 0.28, 0.01, 0.13, 0.02, 0.21, 0.33,
0.05, 0.26, 0.07, 0.31, sd 0.126. The asymmetry the finding describes is
reduced but not gone:

| run | corr(ep, `r_ped`) | corr(ep, leak) | corr(ep, `r_sol`) |
|---|---|---|---|
| A | +0.80 | −0.64 | +0.48 |
| B | +0.99 | −0.80 | +0.54 |
| C | +0.64 | −0.48 | +0.01 |
| **D (v24)** | **+0.86** | **−0.73** | **+0.24** |

**`ped_acceptance = 1.000` is a measurement failure, not a result.** It comes
from the rule judge with `num_judges=0`, so it means "satisfied the rules", not
"taught well". Of the 234 rollouts scoring exactly 1.0, **207 (88%) never
solved the problem**. This rollout scored 1.0:

> *"Hum arrays ke liye logic use kar sakta hai, aur unki first occurrence ka
> index nikalna chahiye. Iske baare mein bahut samajh lenge jaana chaiye."*

Broken grammar, no question, no explanation. A term that a third of rollouts
saturate has stopped supplying gradient and is close to uncorrelated with
whether the tutoring worked. Reporting it as a pedagogy score without that
caveat would be the same error as reading the old `r_ped` curves as learning.

**76% of rollouts still score `r_sol` exactly 0.** Partial credit widened the
distribution but most attempts still pass no tests at all, so the resolution
gained is smaller than the variance figures alone suggest.

### Why none of this was stopped at the right epoch

Nothing was watching. `train()` was `for epoch in range(epochs)` with no early
stopping and no checkpoint selection — grep for either returned zero — and
`epochs=10` is a wall-clock budget against Kaggle's 12 h session cap
(`settings.py`: 20 epochs ≈ 16.7 h, killed near epoch 14 with no eval), not a
convergence criterion. `final_model` was epoch 9 purely because it was last,
while epoch 5 scored 0.333 and epoch 8 scored 0.073.

**The loss cannot supply that criterion, and never could.** The surrogate is
`−min(rho·A, clip(rho)·A)` with the advantage z-scored inside its group, so
advantages sum to zero; `old_log_probs` come from the policy that generated the
rollouts, so `rho = 1` and the expression reduces to `−mean(A) = 0`. Measured
across v24, `policy_loss` ran −0.00096 to +0.00406 and changed sign four times
in nine transitions. It is at zero from epoch 0 and carries no information about
tutor quality. `kl_loss` went negative in 4 of 10 epochs, which a true KL cannot
do, confirming it is a signed single-sample estimator rather than a diagnostic.

Reward is no better a criterion: finding 11 above is reward rising to its first
positive value while held-out solve halved. Stopping on reward stops at the
Goodhart point by construction.

So the signal has to be held-out solve rate. `sahai/training/early_stop.py`
now probes 20 problems from MBPP's **validation** split every 2 epochs, keeps
the best as `output/best_model` with a `selection.json` recording the epoch and
score, and stops after 3 probes without improvement. The split is deliberately
not `test`: choosing a checkpoint by its score on the reported split makes that
score a selection artefact. Costed at the measured 0.893 min/problem, five
probes add 1.5 h, for 7.1 h against the 12 h cap.

Twenty problems cannot establish significance — at a true rate near 0.15 the
smallest improvement detectable at 80% power is roughly 40 points. It is not
asked to. Ranking two checkpoints well enough to prefer one is a far weaker
requirement than significance, and still better than preferring whichever epoch
ran last.

### What is still unmeasured

No run has compared the trained tutor against the model it started from.
Every evaluation so far scored one arm and compared it to a remembered number
from an earlier run, at `n = 20`, where nothing smaller than a ~40-point
difference was ever detectable. `sahai/eval/ab_benchmark.py` runs three arms
— no tutor, the same checkpoint with LoRA disabled, and the trained adapter —
paired on one problem bank, compared with McNemar's exact test on the problems
where two arms disagree. It has not been run; it needs the GPU.

## 15. Tutoring makes the student worse, and the training is not the reason

Run on 2026-10-02 (kernel `sahai-a-b-benchmark` v1, Tesla T4 sm_75, 92.4 min).
Three arms, 60 held-out MBPP test problems, **the same problems in the same
order for every arm**, compared with McNemar's exact test. The adapter is v24's
`final_model` (`lora_dropout 0.0`), which is the last epoch rather than a
checkpoint selected on held-out score; early stopping landed after that run.

| arm | solve | partial | ped | leak | min |
|---|---|---|---|---|---|
| **unaided** (empty dialogue) | **0.283** | **0.339** | — | 0.000 | 8.5 |
| base (same weights, LoRA off) | 0.117 | 0.128 | 0.758 | 0.299 | 40.3 |
| trained (v24 adapter) | 0.133 | 0.183 | 0.986 | 0.045 | 41.9 |

| comparison | discordant | p | reading |
|---|---|---|---|
| unaided → trained | trained 1, unaided 10 | **0.0117** | unaided better |
| unaided → base | base 1, unaided 11 | **0.0063** | unaided better |
| base → trained | trained 5, base 4 | 1.0000 | not resolvable |

**A student tutored by this system solves fewer problems than the same student
left alone.** 17 of 60 against 8 of 60. On ten problems the student solved it
unaided and failed after tutoring; the reverse happened once.

**The training is not what caused it.** `base` and `trained` are indis-
tinguishable on solve (5 discordant against 4, p = 1.0). Both are worse than no
tutor. The harm is in the tutoring loop, not in the adapter.

**And the training did work, at what it optimises.** Pedagogy 0.758 → 0.986 and
leakage 0.299 → 0.045 between the same two arms. GRPO moved the terms it
controls, hard and in the intended direction, with no effect on the outcome.
Finding 14 inferred that from correlations across runs; this measures it
directly, in one session, on one problem set.

### Why this is not a measurement artefact

The `base` tutor **leaks the answer at 0.299 and still solves 0.117**, against
0.283 for an empty dialogue. Leakage hands over the solution, so it should
raise solve rate. The dialogue is costing more than a leaked answer gains.
Partial credit gives the same ordering (0.339 / 0.128 / 0.183), so it is not an
artefact of the strict pass threshold either.

### The pairing is what found it

The effect is 15 points. The two-proportion minimum detectable difference at
n = 60 and a true rate near 0.15 is **+22.3 points**, so an unpaired comparison
would have returned nothing and the result would have read as "no difference
detected". McNemar found it at p = 0.0117 by looking only at the 11 problems
where the arms disagreed. Every earlier evaluation in this project was
unpaired and at n = 20.

### It is the presence of a dialogue, not its content

Across the 320 v24 rollouts, nothing about *what was said* predicts whether the
student then solved the problem:

```
corr(r_ped,        solved) = +0.050
corr(leakage,      solved) = +0.013
corr(dialogue len, solved) = -0.086
```

Splitting rather than correlating says the same: rollouts with `leak > 0.1`
solved 0.087, rollouts with `leak <= 0.1` solved 0.105. Being told the answer
is worth nothing measurable.

So the benchmark's gap is not "the tutoring was bad". A dialogue of any kind in
the context costs about 15 points, and its quality moves that by roughly
nothing.

### Three defects in the prompt the student solves from

Found by reading `StudentSimulator.attempt_solution` after the result came in.
None were visible while every run tutored every attempt, which is how they
survived six runs.

1. **The prompt asserted tutoring that had not happened.** It opened with "you
   just received tutoring" and "based on the hints you received" regardless of
   whether the dialogue was empty. The control arm of a benchmark is exactly
   the no-hints case. This disadvantaged the control, so the measured result is
   conservative, but it is still wrong and is now conditional on
   `bool(dialogue.turns)`.
2. **The student solved from a truncated problem statement.** Only
   `problem.title` reached the prompt, and `title` is `row["text"][:80]` — the
   description cut at 80 characters, often mid-clause — while the tutor saw
   `problem.description` in full. Now fixed to the full description.
3. **The student's own confusion is replayed into its assistant history.**
   The transcript is mapped `student -> assistant`, `tutor -> user`, so the
   model about to write code has its own hedging as its previous assistant
   turns. Across 1002 student turns in the v24 rollouts, **20% express
   confusion** ("I don't understand", "samajh nahi aaya") and **28% claim an
   understanding the solve rate does not support**. A chat model conditions
   heavily on its own prior output, and this one is asked to produce a solution
   immediately after telling itself it does not understand the problem.

(3) is a hypothesis with a mechanism, not a measurement. Context length,
Hinglish code-mixing, and hints that are actively misleading are not excluded
by anything measured so far. `SAHAI_SOLVE_CONTEXT=hints` now passes only the
tutor's turns, as a single user message, so the two framings can be compared on
the same problems. The default stays `full` so existing results remain
comparable until that comparison has been run.

### What this means for the project

The reward is `r_SAHAI = (r_sol - alpha) + (r_ped - 1)*lambda - gamma*L`, and
`r_sol - alpha` is the only term that rewards teaching. If a dialogue reliably
*lowers* `r_sol` relative to no dialogue, that term is negative for tutoring in
general, and no policy over tutor text can fix it: the tutor cannot choose to
be absent. The reward is then maximised by saying as little as possible, which
is consistent with mean turn length falling 574 -> 393 characters over v24.

The order of work this implies:

1. **Re-run the benchmark with `--solve-context hints`.** If the gap closes,
   the harm was the framing and the project's premise survives. If it does not,
   the problem is the simulated student rather than the prompt. This is one
   T4 session and it gates everything below it.
2. **Keep the dialogues.** The first run produced a significant result it could
   not explain, because only scores were saved and diagnosis had to borrow
   training rollouts from a different problem set. `ab_benchmark` now dumps
   every transcript next to its outcome.
3. **Do not tune the reward until (1) is answered.** Every historical gain came
   from removing a distortion in the rule-based channel, and the one attempt to
   add an incentive was gamed inside a single run. Tuning `lambda` or `gamma`
   while the outcome term is negative by construction would optimise a
   quantity that cannot reward teaching.
4. **Re-examine the student.** This measures a 1.5B model roleplaying a
   confused learner, and the finding may be a fact about that roleplay rather
   than about tutoring. A student that performs worse after being helped is not
   a model of a human learner, and the project's claims are about humans.

## 16. The framing was part of the harm, but not all of it

Run on 2026-10-02 (kernel `sahai-a-b-benchmark` v2, ~3 h, Tesla T4). Two passes
over the same 60 held-out problems under the **same fixed solve prompt**, so
the only difference between passes is how the tutoring reaches the solve
attempt. Running `hints` alone would have confounded the framing with the
prompt fixes landed beside it, which is how v14 produced a result nothing could
be attributed to.

| arm | framing | solve | partial |
|---|---|---|---|
| **unaided** | — | **0.317** (19/60) | **0.356** |
| base | full | 0.150 (9/60) | 0.172 |
| trained | full | 0.183 (11/60) | 0.222 |
| base | hints | 0.200 (12/60) | 0.239 |
| trained | hints | 0.250 (15/60) | 0.328 |

### The prompt fixes were real

Every arm rose against finding 15, which used the truncated 80-character
statement and told the control it had received hints it never got:

```
unaided       0.283 -> 0.317   (+0.034)
base (full)   0.117 -> 0.150   (+0.033)
trained(full) 0.133 -> 0.183   (+0.050)
```

### The headline survives

No tutor still beats every tutored arm, on every point estimate:

| comparison | discordant | p | reading |
|---|---|---|---|
| unaided → base (full) | base 3, unaided 13 | **0.0213** | unaided better |
| unaided → trained (full) | trained 3, unaided 11 | 0.0574 | not resolvable |
| unaided → base (hints) | base 1, unaided 8 | **0.0391** | unaided better |
| unaided → trained (hints) | trained 4, unaided 8 | 0.3877 | not resolvable |

### The framing hypothesis is supported in direction and unconfirmed in fact

| arm | full → hints | discordant | p |
|---|---|---|---|
| base | +0.050 | hints 7, full 4 | 0.5488 |
| trained | +0.067 | hints 9, full 5 | 0.4240 |
| pooled | — | hints 16, full 9 | 0.2295 |

Both arms move the same way and the gap to unaided narrows for the trained
tutor from −0.133 (p = 0.057) to −0.067 (p = 0.388). That is consistent with
the mechanism in finding 15 — the student being conditioned on its own
expressed confusion — but **p = 0.23 is not evidence for it**. The stated test
was "if the gap closes under `hints`, the harm was the framing". The gap
narrowed; it did not close, and the narrowing is itself within noise.

**Losing significance is not gaining parity.** `trained(hints)` versus unaided
is p = 0.388 with a point estimate of −0.067 and discordant counts of 4 against
8. The correct statement is "no longer detectable at n = 60", not "no longer
harmful".

### The training now points the right way, and still cannot be shown to work

| framing | base → trained | discordant | p |
|---|---|---|---|
| full | +0.033 | trained 7, base 5 | 0.7744 |
| hints | +0.050 | trained 8, base 5 | 0.5811 |
| pooled | — | trained 15, base 10 | 0.4244 |

Finding 15 had this at 5 against 4, p = 1.0. Four comparisons across two runs
now all favour the trained adapter and not one of them is significant. The
pooled figures are descriptive only: the comparisons share problems, so they
are not independent and the pooled p is not a valid test.

### What it would take to settle either question

McNemar depends on the discordant pairs, and about 20% of problems were
discordant here.

* Framing (16:9): significant at roughly three times the discordant count, so
  **~180 problems per arm**.
* Unaided versus `trained(hints)` (4:8 against): **~180 problems per arm**.

At the measured 0.893 min/problem a 180-problem arm is 2.7 h, so a four-arm
design is ~11 h and does not fit one session alongside the unaided arm. Either
split across two sessions, or accept that the remaining effects are smaller
than this evaluation can see.

### Where that leaves the project

Three things are now measured rather than argued:

1. **Tutoring, as built, does not help this student solve problems.** The best
   tutored arm sits 6.7 points below no tutoring, and the two untrained arms
   are significantly below it.
2. **The solve prompt was carrying real defects**, worth +3 to +5 points once
   fixed, and they had been invisible for six runs because no run had ever
   included an untutored arm.
3. **The training moves tutor behaviour hard and outcomes not at all.** Finding
   15 measured pedagogy 0.758 → 0.986 and leakage 0.299 → 0.045 between the
   same two arms whose solve rates differ by 0.033, p = 0.77.

The reward consequence from finding 15 stands. `r_sol − alpha` is the only term
that pays for teaching, and if a dialogue lowers `r_sol` relative to no
dialogue then that term is negative for tutoring in general — a thing no policy
over tutor text can fix, because the tutor cannot choose to be absent.

The next question is no longer about the prompt. It is whether a 1.5B model
roleplaying a confused learner can be helped by anything at all, because a
student that performs worse after assistance is not a model of a human learner,
and the project's claims are about humans. Transcripts for all 300 arm-problems
are now saved beside their outcomes in the benchmark output, so that question
can be examined on evidence rather than on hypotheses about framing.

## 17. A 7B judge sees what the outcome sees; a 1.5B one does not

Measured 2026-10-03 (kernel `sahai-judge-validation`, Qwen2.5-7B-Instruct in
4-bit on a T4, 9.7 min) against the same 180 transcripts the 1.5B judge saw.
Judging fixed dialogues rather than generating new ones holds everything
constant except the judge.

| | 1.5B | 7B |
|---|---|---|
| coverage, base | 83% | 72% |
| coverage, trained | 85% | **38%** |
| variance, trained | 0.607 | 0.270 |
| corr(score, solved), base | −0.023 | **+0.160** |
| corr(score, solved), trained | +0.152 | **+0.369** |
| LOST mean | +0.545 | **+0.091** |
| WON mean | +0.500 | +0.500 |
| **WON − LOST gap** | **−0.045** | **+0.409** |

The gap is the test. A judge that ranks the dialogues where tutoring rescued a
problem above the ones where tutoring destroyed one is measuring something; a
judge that cannot tell them apart is a target to be optimised, which is the
v16 failure in a more sophisticated form. The 1.5B judge scored the harmful
dialogues *higher* than the helpful ones. The 7B judge separates them by 0.409,
and on the trained arm its positive scores solve at 0.526 against 0.000 for its
negative ones.

**Lower coverage is the 7B judge being right, not worse.** It calls 73% of the
trained tutor's turns NEUTRAL, and finding #15 measured that tutor asking a
question in 65% of its turns. A question is not a technical claim and should
not be scored as one. The 1.5B judge's 85% was overconfident labelling, which
is also why its verdicts did not track anything.

### The prompt was the first judge's problem, not the model

Worth recording separately because it nearly killed the approach. The first
prompt asked "does this message contain a statement that is factually wrong?",
listed MISLEADING first among the options, and stated no prior. It returned
MISLEADING for 109 of 109 turns: a constant, variance 0.0000, and no gradient
at any weight. The same 1.5B model on the same turns, asked to pick a label
rather than answer a leading question, with NEUTRAL listed first and an
explicit prior that most messages are neutral, returned a spread. Three
regression tests pin all three properties.

### What it would cost to train on

A 7B judge in 4-bit is 4.6 GB against a T4's 15.6, and the v24 run measured
3.64 GB for tutor and student together, so the budget is roughly 11.4 GB with
4.2 GB of headroom. At the measured 2.1 s per turn and ~128 tutor turns per
epoch it adds about 4.5 min per epoch, or 45 min across ten, against 4.75 h of
training. bitsandbytes being CUDA-only is why this cannot be done on the M4,
where a 3B judge alone drove the machine to 15 GB of 16 GB swap.

`mu_correct` is still 0.0 and `hint_judge_model` still empty. The validation
says the term is now worth trying, not that it works: a judge that correlates
at +0.369 on 60 problems is a reason to run the experiment, and the experiment
is the thing that decides. The project's record is that every reward term
added to the channel the policy controls has been gamed, and this one is only
partly outside that channel -- it is anchored to the reference solution, but
read through a model that the policy's output can influence.

The honest form of the next run is a paired one: same configuration, same
seeds, `mu_correct` at 0 and at some positive value, compared on held-out solve
rate with the benchmark that exists. Anything less and the term joins the list
of changes whose effect was never established.

## 18. The correctness term: inconclusive, and the experiment was not paired

Two runs on 2026-10-03, `mu_correct` 0.0 and 0.5, a Qwen2.5-7B judge in 4-bit
at 5.07 GB, ~7.9 h each. Both selected epoch 3 by early stopping, so neither
shipped its final epoch.

### The experiment was confounded before it started

`settings.seed` has existed since the beginning and was **applied nowhere in
the training path**. `zpd_sample` draws randomly inside its difficulty window
and both agents sample their turns, so the two arms drew **different problems
in epoch 0, before a single gradient step**. Verified directly: the epoch-0
problem lists differ and so do the per-rollout `r_ped` values.

Any difference between the arms therefore carries training nondeterminism as
well as `mu_correct`. `seed_everything` now covers `random`, torch, CUDA and
numpy; GPU reductions remain non-bitwise-reproducible, so runs are comparable
rather than identical.

### The term was paid for and went down

Recomputed from the rollout dumps, since `metrics.json` dropped the fields (see
below):

| | correctness, first 3 → last 3 | Δ |
|---|---|---|
| control (μ=0) | +0.273 → +0.276 | +0.003 |
| treatment (μ=0.5) | +0.622 → +0.333 | **−0.288** |

The judge fired on 72–97% of rollouts in both arms, so this is not a coverage
failure. The treatment was rewarded 0.5 per unit of correctness and the policy
drove it **down**, while both arms moved the terms they always move: `r_ped`
0.78 → 0.97, leakage 0.44 → 0.11.

The reading that fits is that `r_ped` at weight 1.0 is cheaper to satisfy than
correctness at 0.5, because `r_ped` is a deterministic function of the tutor's
own text and correctness requires being right about something the tutor does
not control. Given the choice the policy took the cheap term, which is the same
preference every run has shown.

### Held-out, paired on one bank

| arm | solve | partial |
|---|---|---|
| **unaided** | **0.367** | **0.433** |
| control μ=0.0 | 0.167 | 0.194 |
| treatment μ=0.5 | 0.250 | 0.306 |

| comparison | discordant | p |
|---|---|---|
| control → treatment | treatment 10, control 5 | 0.3018 |
| unaided → control | control 2, unaided **14** | **0.0042** |
| unaided → treatment | treatment 2, unaided 9 | 0.0654 |

The treatment is +0.083 above the control and that is **not resolvable at
n=60**. The gap to no tutoring narrows from −0.200 (p=0.0042) to −0.117
(p=0.0654), which is a loss of significance rather than a demonstration of
parity: the point estimate is still negative and still favours no tutor.

### What this does and does not establish

It does not establish that the correctness term helps. The direction is
favourable, the effect is inside the noise band, and the arms were not paired,
so even the direction carries a confound. The pre-registered falsification was
"correctness climbs while solve does not"; what happened instead is that
correctness fell, which the falsification did not anticipate and which is
weaker evidence for the term than the failure case would have been.

It does re-establish, for the fourth independent measurement, that **tutoring
as built costs the learner**: 0.367 unaided against 0.250 for the better of the
two tutored arms, with the control significantly worse at p=0.0042.

The term should not be kept or removed on this data. The run that would decide
it is the same pair with the seed fix in place, which is now a real pairing
rather than a nominal one. Until then `mu_correct` stays 0.0.

### A reporting bug that nearly voided the result

The first read of the treatment arm showed `mean_correctness 0.000` and
coverage 0% on every epoch, which reads as the judge never firing. The judge
fired on 81% of rollouts. Both `_write_metrics` and the notebook's final cell
serialised hand-kept lists of keys fixed before `mean_solved`,
`mean_correctness`, `correctness_coverage` and `probe_solve` existed, and the
notebook's write clobbered the trainer's. Everything was computed, used in the
reward, and dropped on the way to disk; only the rollout dumps preserved it.
Both writers now serialise the dataclass whole, with a test asserting every
field this experiment needs is declared on `TrainMetrics`.

## 19. The correctness term does nothing, measured properly this time

Reran 2026-10-04 with `seed_everything` in place, `mu_correct` 0.0 against 0.5,
Qwen2.5-7B judge in 4-bit, ~7.9 h each. Both arms stopped at 8 epochs by early
stopping and both selected **epoch 1**.

### The pairing is real now

Epoch 0 is identical across arms: same four problems in the same order, same
per-rollout `r_ped`, same `solved`. Finding #18's arms differed here, before a
gradient step, which is what made that comparison unattributable.

Pairing holds through epoch 4 and diverges from epoch 5. That divergence is the
treatment working, not a defect: a different reward produces a different policy,
which produces different BKT updates, which moves the ZPD draw. Identical
curricula past the point where the arms genuinely differ would mean the
treatment was doing nothing at all.

### The unpaid arm improved correctness more than the paid one

| | correctness, first 2 → last 2 | Δ |
|---|---|---|
| control (μ=0, **not rewarded for it**) | +0.177 → +0.570 | **+0.393** |
| treatment (μ=0.5, rewarded for it) | +0.276 → +0.552 | +0.276 |

Coverage sat at 72–88% in both arms, so this is not the sparsity that killed
the vocabulary detector. Correctness rises during training whether or not it is
paid for, as a by-product of leakage falling and `r_ped` climbing: a tutor
making fewer claims of any kind makes fewer wrong ones. Weighting it at 0.5
bought nothing, and the arm that was paid improved *less*.

### Held-out, paired on one bank of 60

| arm | solve | partial | ped | leak |
|---|---|---|---|---|
| **unaided (no tutor)** | **0.367** | **0.433** | — | 0.000 |
| control μ=0.0 | 0.200 | 0.217 | 0.888 | 0.164 |
| treatment μ=0.5 | 0.217 | 0.233 | 0.913 | 0.125 |

| comparison | discordant | p | |
|---|---|---|---|
| control → treatment | treatment 6, control 5 | **1.0000** | no effect |
| unaided → control | control 3, unaided 13 | **0.0213** | unaided better |
| unaided → treatment | treatment 3, unaided 12 | **0.0352** | unaided better |

`p = 1.0000` on eleven discordant pairs split six to five is as null as this
test produces. The two adapters agree on 49 of 60 problems. The held-out eval
inside each run reported the same solve rate for both arms to three decimals.

### The falsification fires

The condition written into the notebook before the run was: *if
`mean_correctness` climbs while held-out solve does not, that is the v16
pattern and the term should be removed rather than retuned.* Correctness
climbed in both arms. Held-out solve did not move. The term is removed, not
retuned, and `mu_correct` stays 0.0 as a measured conclusion rather than a
precaution.

It fails more completely than the falsification anticipated. The condition
assumed the term would at least drive its own metric; it did not even do that
better than an arm ignoring it.

### What was worth building anyway

The judge is the only reward signal in this project ever validated against
outcomes *before* being trained on (finding #17, WON−LOST gap +0.409). It keeps
its place as a diagnostic: it reads dialogues the policy is not paid to
influence, which is exactly what made this experiment readable. The failure is
of the term, not of measuring first — measuring first is what turned eight
plausible GPU hours into a defensible negative.

### Fifth independent measurement of the headline

Tutoring as built costs the learner, now at **0.367 unaided against 0.217** for
the better arm, both tutored arms significantly worse (p = 0.021, p = 0.035).
That result has survived two platforms, two student precisions, three solve
prompts, two framings, early stopping, and a correctness term. The reward is
not missing a term. Something earlier than the reward is wrong.

## 20. The student can be helped, and the verifier was failing correct code

Two results from the bottom-up audit on 2026-10-05, one about the environment
and one about the instrument measuring it.

### The ceiling test

A synthetic transcript hands the student the reference solution verbatim and
asks it to write the solution out. Nothing a tutor could say beats being given
the answer, so this is the ceiling any tutor is competing for.

| arm | solve |
|---|---|
| unaided (nothing given) | 0.400 |
| **oracle (solution handed over)** | **0.633** |

Paired, +0.233, oracle won 20, unaided won 6, **p = 0.009**.

**The student can use a dialogue.** That settles the stronger hypothesis from
finding #19's close: the environment is not dead, and `r_sol − alpha` does have
headroom for a reward to find. It is just not much headroom, and no tutor
trained here has come close to it.

It also fails 17 of 55 winnable problems **with the correct solution in front
of it**. A 1.5B model asked to copy code it has just been shown succeeds about
seven times in ten. That is the real ceiling, and it is a property of the
student, not of any tutor.

### The verifier was marking correct solutions wrong

Found while asking why the oracle arm failed: on the held-out split, **9 of 60
reference solutions failed their own tests**. The maximum achievable solve rate
was never 1.0; it was 0.850.

Four of the nine were one bug. The candidate's return value is serialised with
`json.dumps` in the subprocess and parsed back, so a tuple arrives as a list
and an int dict key arrives as a string. `expected` is a Python literal parsed
from the assert and keeps both. The comparison was between a value that had
made a lossy round-trip and one that had not:

```
expected [('pink', 6), ('black', 5)]      <- tuples
got      [['pink', 6], ['black', 5]]      <- lists, after json
```

`_json_normalise` now sends `expected` through the same round-trip. It is not
leniency -- `[1,2]` still does not equal `[2,1]` and `"1"` still does not equal
`1` -- it makes the comparison type-consistent.

| split | reference failures before | after | max achievable |
|---|---|---|---|
| test | 9/60 (15%) | **5/60 (8%)** | 0.850 → **0.917** |
| train | 29/198 (15%) | **10/198 (5%)** | 0.854 → **0.949** |

The remaining five on the held-out split are MBPP data problems rather than
harness ones: two reference solutions raise, and three have expected values
that do not correspond to what the function returns.

**Every solve rate this project has reported was depressed by this**, including
the unaided baselines the tutored arms were judged against. The direction of
findings #15 to #19 does not change -- the bug applied equally to every arm and
all those comparisons were paired on one bank -- but the absolute numbers were
all too low, and `r_sol` was returning 0 during training for 15% of problems no
matter what the tutor did.

### What the headroom actually is

On the 55 problems that are winnable after the fix:

| | solve |
|---|---|
| ceiling (oracle) | **0.691** |
| unaided baseline | 0.418 |
| v2 control | 0.218 |
| v2 treatment | 0.200 |

The total headroom for a perfect tutor is **+0.273**. Every tutor trained so
far sits roughly 0.2 *below* the unaided baseline, so they are not competing
for that headroom, they are destroying value that exists without them.

Two things follow. A tutor would have to recover 0.2 before gaining anything,
and the whole prize is 0.273 — against a paired evaluation at n=60 that
resolves about 0.15. The experiment has been trying to detect an effect of a
size comparable to its own resolution, on top of a deficit larger than the
prize.

## 21. The dataset layer: skills from the wrong place, and dead training groups

Two defects, both found by auditing downward rather than by a failing run.

### Skills were read from the description, never from the solution

`_tag_skills` keyword-matched the problem *text*. `general` was the fallthrough
when nothing matched, and it was **22% of the bank** — BKT was tracking
per-learner mastery on a label that carries no information, and `alpha`, the
ability baseline `r_sol − alpha` subtracts, averaged over it.

An AST technique detector already existed in `sahai_core`, already tested, and
was never consulted. The two passes disagree in both directions:

| | keyword (description) | AST (solution) |
|---|---|---|
| `hash_maps` | 32 | 14 |
| `recursion` | 3 | **12** |
| `sorting` | 18 | 17 |

Keywords over-trigger: "count" and "unique" appear in descriptions of problems
solved with a plain loop. The AST under-triggers: it finds no technique at all
in **74%** of MBPP solutions, because most of them are a loop or an expression.
Neither replaces the other, so tags are now their union.

| | before | after |
|---|---|---|
| `general` as the only tag | 44/198 (22%) | **35/198 (18%)** |
| mean skills per problem | 1.0 | **1.44** |
| skills with ≥10 problems | 5/15 | **7/15** |

`recursion` goes 3 → 14, `sorting` 18 → 30, `binary_search` 4 → 7. Still thin
at the tail — `graphs` has one problem and `trees` two — so per-skill mastery
on those remains an estimate from almost nothing. That is a property of MBPP,
not of the tagger.

### Unsolvable problems were costing whole training groups

A problem whose reference solution fails its own tests is not noise, it is a
**dead group**. Every rollout in it scores `r_sol = 0` whatever the tutor said,
so the group has no reward variance, every z-scored advantage in it is exactly
zero, and it contributes nothing to the gradient while consuming a full
generation budget of eight dialogues.

After the verifier fix in finding #20 that is 10 of 198 training problems
(5%) and 5 of 60 held-out (8%). Before it, 29 and 9. So roughly **one epoch in
twenty was spent on groups that could not teach anything**, and before the
verifier fix it was closer to one in seven.

`load_mbpp(validate=True)` executes every reference and drops the failures,
returning what it dropped rather than filtering silently. It is **off by
default**: it costs three subprocesses per problem, and every number this
project has reported was measured without it, so enabling it silently would
make new runs incomparable with old ones. Training should set it.

With it on, the maximum achievable solve rate is 1.000 for the first time —
188/198 training and 55/60 held-out problems, all of them winnable.

### What is still wrong with this dataset

Three things that filtering cannot fix, all of which argue the benchmark is
mismatched to the research question rather than merely dirty:

* **30% of solutions are three lines or fewer.** There is nothing to tutor in
  `def is_upper(s): return s.upper()`, and a four-turn Socratic dialogue about
  it can only add noise. This predicts the verbosity finding (#15) directly.
* **`r_sol` has three or four levels, not a continuum.** 197 of 198 problems
  have exactly three asserts, so partial credit can only take {0, ⅓, ⅔, 1}. The
  partial-credit change in #14 bought resolution, but far less than it appeared
  to.
* **70% of problems sit at difficulty 1.** A low-ability learner targets
  difficulty ~1, where most of the bank already is, so the ZPD curriculum is
  close to drawing an easy problem at random.

## 22. Not punishing valid alternatives costs half the judge's discrimination

The correctness term and the hint judge both anchored on the reference
solution's *approach*, which punishes a tutor for recommending a different
approach that also works. Measured: of 23 student solutions that pass every
test, 11 had a detectable technique on both sides and **2 of those 11 (18%)
used a different one** — `opposite_Signs` is `(x ^ y) < 0` in the dataset and
`x * y < 0` in a passing solution.

This repeats a mistake the project already fixed once. Leakage was token
overlap against the reference until it "scored 0.000 for a tutor that wrote a
complete working solution, because it chose a different algorithm". The fix
there was execution, not comparison.

### The fix works and it is not free

Both prompts measured on the same 180 transcripts with the same 7B judge:

| | anchored on the reference | multi-approach |
|---|---|---|
| coverage, base | 72% | 87% |
| coverage, trained | 38% | 58% |
| variance, trained | 0.2697 | 0.3489 |
| corr(score, solved), trained | **+0.369** | +0.129 |
| LOST dialogues | **+0.091** | +0.455 |
| WON dialogues | +0.500 | +0.667 |
| **WON − LOST gap** | **+0.409** | **+0.212** |

The gap halved and the correlation fell by two thirds. The verdict counts say
why:

| | anchored | multi-approach |
|---|---|---|
| HELPFUL (base) | 49% | 65% |
| NEUTRAL (base) | 33% | 19% |
| WRONG (base) | 18% | 16% |

**`WRONG` barely moved** (16→14 on base, 4→4 on trained). The shift is
`NEUTRAL → HELPFUL`: told that a different approach can still be right, the
judge began endorsing claims it had previously declined to label. Dialogues
where tutoring turned a solved problem unsolved now score **+0.455** against
+0.091 — the judge is calling harmful tutoring helpful.

### Which is right

Both readings are defensible and neither is settled by this data.

The anchored prompt discriminated better, and some of that was real. It also
penalised deviation from the reference, which in a weak tutor correlates with
confusion rather than with originality, so an unknown part of its +0.409 was
measuring conformity rather than correctness. That part is spurious and would
have paid a trained tutor to recommend only what the dataset contains.

The multi-approach prompt is correct about what it is measuring and worse at
measuring it. Its leniency is not a tuning problem: `WRONG` did not move, so
making the caveat stricter would not recover the gap, it would only move
`HELPFUL` back to `NEUTRAL` and reduce coverage.

The shipped default is the multi-approach prompt, because the alternative is
known to punish correct teaching and the judge's current role is diagnostic
rather than gradient — `mu_correct` is 0.0 and finding #19 showed the term does
nothing at either weight. A diagnostic that is wrong about 18% of valid advice
is worse than one that is blurry.

If the judge is ever used as a gradient, neither prompt is adequate. The
principled version is the one that fixed leakage: verify rather than compare.
For a hint that would mean turning the advice into code and running it, which
is tractable for "use a heap" and not for "what does the first element tell
you" — and the latter is 58% of what this tutor says.

## 23. Five of the sixty held-out problems were unwinnable by construction

Found on 2026-10-06 while auditing why the oracle arm fails at all. The oracle
arm pastes the reference solution into the tutor's turn and asks the student to
write it out; it scored 0.633, and that 0.367 shortfall had been read as a
capability limit of a 1.5B student. Part of it was not.

`_parse_assert` has always returned the function name the assertion calls. The
loader discarded it:

```python
_, input_dict, expected = parsed          # the tested name, thrown away
func_name = _extract_function_name(row["code"])   # first `def` in the file
```

MBPP reference solutions frequently open with a helper, so the two names
disagree whenever they do:

| problem | first `def` | what the tests call |
|---|---|---|
| mbpp_18 | `str_to_list` | `remove_dirty_chars` |
| mbpp_30 | `check_Equality` | `count_Substring_With_Equal_Ends` |
| mbpp_45 | `find_gcd` | `get_gcd` |
| mbpp_56 | `rev` | `check` |
| mbpp_70 | `find_equal_tuple` | `get_equal` |

### The field is not cosmetic

`CodeVerifier.verify` calls `problem.function_name`, so it executed the helper
and compared its return value against the entry point's expected value. All
five reference solutions failed their own tests.

`function_signature` is interpolated into the student's solve prompt as
`Signature:`. On mbpp_56 the student was told to implement `rev(num)` on a
problem scored by calling `check(n)`. No output could pass.

So these five problems returned 0 in every arm, in every run, and sat in the
denominator of every solve rate this project has reported. On the oracle arm
the failure is particularly clean: correct code was shown in the dialogue for a
function the verifier never called.

### Scale

Reading the entry point from the assertions instead:

| split | loaded | entry point corrected | references passing their own tests |
|---|---|---|---|
| train | 198 | 6 | 188 → **194** |
| validation (probe) | 88 | 6 | → **87** |
| test (held out) | 60 | 5 | 55 → **60** |

The training bank recovers six problems that `drop_unsolvable` had been
discarding. Under `validate=False`, which is the default and what the benchmark
uses, they were not discarded but included and unwinnable.

### What it does to the numbers already reported

Recomputed on the 2026-10-02 three-arm dump, restricted to the 55 problems
whose reference worked:

| arm | all 60 | the 55 that were winnable |
|---|---|---|
| unaided | 0.367 | 0.382 |
| base | 0.200 | 0.200 |
| trained | 0.283 | 0.309 |

The direction of every comparison survives, because a problem scoring 0 in all
three arms cannot create a difference between them. What it does change is the
floor: roughly 8% of the held-out set was dead weight, and the ceiling the
oracle arm was measuring was 55/60 of what it appeared to be.

A second thing surfaced in the same recomputation and is worth recording
separately, because the report states the stronger version. On this dump
`unaided` versus `trained` is **p=0.3323**, not significant; the p=0.035 in the
report is the 2026-10-04 v2 pair. The `base` arm is significantly worse than
unaided in both (p=0.021 here, p=0.013 on the 55). So "the untrained tutor
costs the learner" is the claim with consistent support across runs, and "the
trained tutor costs the learner" holds in one pair and not the other. The
deficit's magnitude is not stable across adapters.

### What was checked and was clean

Test-case parsing on the held-out set: all 60 problems yield 3 of 3 assertions,
none dropped, none with zero usable cases. The verifier's JSON round-trip fix
from finding #20 holds. The remaining reference failures after this change are
1 in validation and 4 in train, which `drop_unsolvable` handles.

## 24. The learner block was served to a policy that never trained on it

`sahai_core.learner_context` shares the wording of the personalisation block
between the trainer and the gateway, and its docstring says why:

> They must agree: a policy trained on one wording and served another is being
> asked at inference time to follow an instruction it never saw during
> training, which is exactly the gap this module closes.

It closed the wording. It did not close whether the block exists.
`sahai/core/dialogue.py` defined `LEARNER_CONTEXT_ENABLED` from
`SAHAI_LEARNER_CONTEXT` and defaulted it off, so no recorded run has trained
with the block. The gateway's `_context_for` built it on every served turn,
typed and spoken, with no reference to the flag.

The served tutor therefore received a block in its system prompt that its
adapter had never been optimised against, and nothing recorded the discrepancy
because each side was internally consistent on its own terms. This is the same
shape as the `generate` versus `compute_log_probs` mismatch in the prompt
consistency work: one decision, two implementations, diverging silently.

The flag now lives in the shared library and both halves import it. Default
stays off, which matches every number reported so far, and turning it on has to
be done in the environment of the trainer and the services together.

This disables the served personalisation block, which includes the skill priors
the GitHub import seeds. That is a real loss of behaviour, and it is the
correct direction: the block was never validated and was being fed to a model
that had not trained on it, which makes it an unmeasured confound rather than a
working feature.

## 25. The tutoring deficit was mostly how the dialogue was replayed to the student

`SAHAI_SOLVE_CONTEXT` has had two settings since 2026-10-02 and only one had
ever been run. Its docstring named the gap and left it open:

> `hints` exists so the two can be compared on the same problems instead of
> argued about, and the default is unchanged so existing results stay
> comparable until that comparison is run.

Run on 2026-10-06, 142 minutes on MPS. The comparison is made by **replaying
the solve step of the saved benchmark dialogues** under both framings rather
than by re-running the benchmark. Framing affects only the final solve call, so
one fixed set of transcripts can be scored twice; re-running would resample the
tutor and confound the framing with new dialogue content.

| arm | `full` | `hints` |
|---|---|---|
| unaided (no dialogue, both framings identical) | 0.367 | 0.367 |
| base | **0.183** | **0.350** |
| trained | 0.333 | 0.350 |

Paired, McNemar exact, n=60:

| comparison | discordant | p |
|---|---|---|
| unaided vs base, `full` | 13 / 2 | **0.0074** |
| unaided vs base, `hints` | 8 / 7 | 1.0000 |
| unaided vs trained, `full` | 11 / 9 | 0.8238 |
| unaided vs trained, `hints` | 7 / 6 | 1.0000 |
| base vs trained, `hints` | 5 / 5 | 1.0000 |

### The headline result does not survive the framing change

Under `full` the untrained tutor costs the learner at p=0.0074, which is the
result this project has reported in five forms. Under `hints`, on the same
dialogues, with the same tutor turns, the deficit is gone: 0.350 against 0.367
unaided, 8 discordant one way and 7 the other.

Nothing about the tutoring changed. What changed is that the student's own
turns stopped being replayed into its assistant history before it was asked to
write code. The mechanism named in the docstring is the one that was operating:
a chat model conditions heavily on its own prior assistant turns, and 20% of
student turns express confusion.

**Tutoring as built does not harm the learner. The measurement did.**

### What training actually bought

| framing | base | trained | discordant | p |
|---|---|---|---|---|
| `full` | 0.183 | 0.333 | 12 / 3 | **0.0352** |
| `hints` | 0.350 | 0.350 | 5 / 5 | 1.0000 |

Under the broken framing the trained policy beats the untrained one
significantly. Under the fixed framing they are identical to the problem.

So the one place training ever showed a measurable gain was **partial immunity
to the measurement artefact**, not better teaching. The trained policy produces
dialogues whose student turns hedge less, which interacts less badly with the
`full` replay. Remove the artefact and the two adapters are indistinguishable,
which is consistent with every other measurement of training in this project.

### The reward used during training was computed under `full`

`SolveReward` scores the same `attempt_solution` call. Every run to date
therefore optimised against an outcome signal carrying this artefact, which
penalised dialogues in proportion to how much the student talked rather than
how well the tutor taught. The policy has never been optimised against a clean
`r_sol`. That makes a retrain under `hints` the first training run whose
outcome term means what it says, and it is the obvious next experiment.

### Caveat: the replay is not bit-exact

`attempt_solution` decodes greedily, so replaying `full` should reproduce the
dump exactly. It reproduces **57/60 in both arms**. The 3-problem discrepancy
is MPS reduction nondeterminism, not a logic difference, and it sets a noise
floor of about 5% on any single arm.

The base arm's framing effect is 12 gained against 2 lost, far outside that
floor. The trained arm's is 7 gained against 6 lost, entirely inside it, so
"the trained arm is unaffected by framing" is the honest reading rather than
"the trained arm improves slightly".

## 26. The oracle ceiling is 0.783, and two thirds of the shortfall was ours

Re-ran the oracle arm on 2026-10-06 with the entry-point fix of finding #23 in
place, dumping the raw generation beside the extracted code so the failures
could be split rather than assumed.

| | solve rate |
|---|---|
| oracle, as previously reported | 0.633 |
| oracle, with the entry point read from the assertions | **0.733** |
| oracle, with AST-based code extraction as well | **0.783** |

Against unaided 0.367, paired McNemar: p=0.0001 and p<0.0001. The student can
use what is in the dialogue, decisively.

### Where the original 22 failures went

| cause | n | fixable |
|---|---|---|
| entry point taken from the first `def` (finding #23) | 6 | fixed |
| `_extract_code` discards the solution on a name mismatch | 3 | fixable, see below |
| genuine failure with the answer in front of it | 13 | no |

`_extract_code` ends with:

```python
if f"def {function_name}" not in text:
    text = f"def {function_name}():\n    pass"
```

Everything the student wrote is thrown away on a literal string miss. All five
misses are naming style, and three are pure case differences between MBPP's
mixedCase and the snake_case the student writes:

| wanted | student wrote | recovered by AST |
|---|---|---|
| `max_Prime_Factors` | `max_prime_factors` | yes |
| `decimal_To_Binary` | `decimal_to_binary` | yes |
| `count_Substrings` | `count_substrings` | yes |
| `count_Substring_With_Equal_Ends` | `count_substrings_with_equal_ends` | no, logic also wrong |
| `get_equal` | `check_tuples_length` | no, logic also wrong |

Aliasing the last top-level function to the tested name recovers 3 of 60, or
5 points of solve rate, in every arm. Note this is not a rename: an alias
preserves recursion through the original name.

Of the remaining 13, four score partial credit (0.33 to 0.67), so the student
is writing nearly-correct code rather than failing to engage. Zero generations
hit the 512-token cap and zero failed to parse, so truncation and malformed
output are not contributors.

### What this does to the headroom

| | before this work | after |
|---|---|---|
| unaided | 0.367 | 0.367 |
| oracle ceiling | 0.633 | 0.783 |
| headroom available to tutoring | 0.267 | **0.417** |

The conclusion in the report that "the binding constraint is the task and the
student, not the reward or the optimiser" was drawn against the 0.267 figure
and a tutoring deficit that turns out to be an artefact. Both inputs to it have
changed. There is more than twice the measurable headroom previously believed,
and tutoring is no longer starting from behind.

## 27. The framing effect is real, both explanations offered for it are wrong

Finding #25 established that omitting the student's turns from the context of
its final attempt removes a deficit of $0.184$, and offered a mechanism: the
model conditions on its own recorded confusion, measured at 20% of student
turns, immediately before being asked to write code. That mechanism carries a
prediction, and the prediction fails.

### Hedging does not predict recovery

Among the problems available to be recovered, comparing those the framing
change fixed against those that stayed failed:

| | recovered | still failed |
|---|---|---|
| dialogue contains student "I am lost" | 0.632 | 0.500 |
| mean stuck turns | 0.68 | 0.51 |
| **Fisher exact, two-sided** | | **p=0.4378** |

Per arm it is worse, not better: the base arm splits 0.50 against 0.49,
p=1.0000, and only the trained arm shows anything (0.857 against 0.515,
p=0.2055, n=7). Breakage does not track it either, with the problems the change
*broke* carrying slightly more hedging than those it left solved (0.62 against
0.48), so the signal is not even directional.

What does separate the two groups is volume. Recovered problems average 843
characters of student text against 534 for those that stayed failed, consistent
across both arms.

### Re-roling the content does not recover it either

`hints` removes two things at once: the student's content, and the fact that
content sat in the model's own assistant history. A third framing, `quoted`,
keeps every character and puts the whole transcript in one user message, which
holds volume fixed and removes only the role. On the 27 problems where `full`
and `hints` disagree, those two are opposites by construction, so the only
question is which one `quoted` follows.

| arm | n | tracks `full` | tracks `hints` | sign test |
|---|---|---|---|---|
| base | 14 | 6 | 8 | p=0.7905 |
| trained | 8 | **8** | **0** | **p=0.0078** |
| pooled | 22 | 14 | 8 | p=0.2863 |

In the trained arm `quoted` is identical to `full` on every discordant problem.
In the base arm it is a coin flip. Pooled, it leans toward `full` and
establishes nothing.

So `quoted` is not a substitute for `hints`. Removing the student's text is
doing the work; moving it out of the assistant role is not enough. That
disposes of the role explanation, and it is the opposite of what we expected
when the framing was added.

### Where that leaves the framing

`hints` is still the only framing shown to recover the deficit, and the two
candidate explanations for why are both now excluded: not the hedging content
specifically, and not the assistant role. The surviving description is the
weakest one, that the quantity of prior student text degrades the attempt
regardless of what it says or who is recorded as saying it, and the volume
contrast above is consistent with that without testing it.

This is uncomfortable but it is the state of the evidence. The practical
consequence is unchanged: no result in this project is interpretable without
stating its framing, because the two framings differ by more than the effect
every arm was built to detect. The scientific consequence is that we do not
know why, and the explanation printed in the report and in finding #25 should
not be relied on.

### Caveat on this finding

The `quoted` run was stopped at 22 of 27 problems to yield the GPU to another
project on the same machine, so the trained arm contributes 8 of its 13
discordant problems. The base arm is complete. A pooled sign test at n=22 has
little power, and the arm disagreement may not survive the missing five.

## 28. The training configuration, not the reward, is why twenty runs did nothing

Asked on 2026-10-09 why the retrain still shows no clean gain. Every previous
investigation in this project went after the reward terms, the prompts, the
judge or the measurement. None asked whether the training loop as configured
can deliver a usable gradient. Measured from the rollout dumps, which carry the
advantage GRPO actually applied.

### What is now fixed, and it is not small

| | earlier runs | this run |
|---|---|---|
| groups with zero reward variance | ~1 in 7 | **0 of 32** |
| rollouts with zero advantage | many | **0 of 256** |
| mean within-group variance, `r_sol` | 0.0098 | **0.0741** |
| mean within-group variance, `r_ped` | 0.0482 | 0.0248 |
| `r_sol` share of usable variance | small | **36.5%, the largest term** |

Gradient starvation is gone. `r_sol` now carries three times the usable
variance of `r_ped` and is the single largest contributor. Mean $|$advantage$|$
is 0.80. The reward design work was not wasted and the diagnosis in finding #14
was correct for its time. It is no longer the binding constraint.

### The apparent rise in rollout solve rate is problem draw

Rollout solve rate across the eight epochs: 0.188, 0.250, 0.281, 0.250, 0.188,
0.156, 0.188, 0.594. The last epoch looks like a breakthrough.

| correlation | value |
|---|---|
| epoch vs strict solved | $+0.434$ |
| **AST nodes of the problems drawn vs strict solved** | $\mathbf{-0.623}$ |
| epoch vs AST nodes | $-0.372$ |

Difficulty of the draw explains more of the epoch-to-epoch variation than
epoch does, and epoch 7 drew the easiest problems of the entire run: 54.0 mean
AST nodes against 105.2 at epoch 0 and 115.0 at epoch 5. Mean difficulty over
the first two epochs was 2.38 and over the last two 2.25, so the curriculum did
not move to harder problems either. Rollout solve rate is not interpretable as
a learning signal.

### The within-problem test says training did nothing

Six problems were drawn in more than one epoch. That is the only design here
that holds the problem fixed and varies policy state.

| problem | earlier epoch | later epoch |
|---|---|---|
| mbpp_603 | 0.00 | 0.00 |
| mbpp_633 | 0.50 | 0.38 |
| mbpp_699 | 0.00 | 0.00 |
| mbpp_755 | 0.25 | 0.00 |
| mbpp_781 | 0.12 | 0.12 |
| mbpp_791 | 0.00 | 0.25 |

One better, two worse, three unchanged. $n{=}6$ is far too small to test, but it
is the right comparison and it points the same way as every held-out
measurement in this project.

### The configuration is one to two orders of magnitude from its own source

The project is built on the review of~\cite{rev-pedrl}, which is
Dinucu-Jianu et al., EMNLP 2025, arXiv:2505.15607. That paper trains the same
kind of system. Its published hyperparameters against ours:

| | reference | SAHAi | factor |
|---|---|---|---|
| problems per batch | 16 | 4 | 4x fewer |
| rollouts per problem | 8 | 8 | same |
| learning rate | 5e-7 | **1e-4** | **200x higher** |
| KL coefficient | 0.001 | **0.05** | **50x stronger** |
| gradient steps per batch | 2 | **16** | **8x more** |
| compute per run | ~200 GPU-hours | ~12 | ~17x less |

Four of five differ by a large factor, and in combination they describe a run
that takes very large steps, under a very strong pull back to the reference
policy, many times over the same stale rollouts, on very few problems.

The learning rate has a traceable cause. It was raised from 2e-5 to 1e-4 with
the reasoning, recorded in the settings, that "LoRA adapters are normally
trained at 1e-4..3e-4". That is an supervised-finetuning convention. For
policy-gradient RL the reference uses 5e-7, and the intuition does not carry
across. The KL coefficient and the sixteen sequential steps compound it: the
project's own threats section already noted those steps "consume rollouts from
a policy that has already moved", and the reference takes two.

### Coverage

26 distinct problems out of 198, 13.1%, across 32 groups. Four problems per
epoch over eight epochs cannot be more than 32, and six were repeats. A general
tutoring skill is not learnable from 26 problems in 128 optimiser steps, whatever
the reward says.

### What this means for the project's history

The honest reconstruction of twenty runs is that two separate faults were
active the whole time and each was sufficient on its own to produce the null.
Until this revision the measurement was wrong, in four distinct ways. Now that
it is right, the training configuration is wrong, in four distinct ways, and
none of the configuration numbers has ever been compared against the paper the
method was taken from. That comparison took one search.

## 29. The policy never moved, which explains every null more simply than the reward does

Measured on 2026-10-09 from the saved adapter, with no GPU. PEFT initialises
`lora_A` from a Kaiming uniform and `lora_B` to **exactly zero**, so the
adapter contributes nothing at step 0 whatever `lora_A` holds, and everything
training learned is in how far `lora_B` travelled from zero.

| | best_model (epoch 5) | final (epoch 7) |
|---|---|---|
| `lora_A` mean $|w|$ (random init, scale reference) | 0.012758 | 0.012771 |
| **`lora_B` mean $|w|$ (init exactly 0)** | **0.000762** | 0.000902 |
| `lora_B` max $|w|$ | 0.004822 | 0.005863 |

With $\alpha/r = 2$ and rank 8 that is an effective weight perturbation around
$5\times10^{-5}$ against base weights of order $0.02$, so roughly $0.2\%$.

The KL to the frozen reference agrees:

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| raw KL | $-0.0067$ | $-0.0011$ | $-0.0115$ | $-0.0009$ | $-0.0126$ | $+0.0005$ | $-0.0025$ | $+0.0033$ |

The sign flips at random and the magnitude never leaves the noise of the k1
estimator. Run v24 reached $0.0301$ in this same column, at this same learning
rate, and that run was described as the policy finally making measurable
progress away from the base model. This run moved **less** than v24.

### This inverts the conclusion of finding #28

Finding #28 compared this project's hyperparameters against
arXiv:2505.15607 and found the learning rate 200x higher, the KL coefficient
50x stronger, the batch 4x smaller and the gradient steps 8x more numerous. The
arithmetic is right. The inference drawn from it, that steps were too coarse
and the learning rate should come down, is wrong, and this measurement is why:
at $10^{-4}$ the policy did not move. Taking the learning rate to $5\times
10^{-7}$ would freeze it outright.

Of the three changes made in response to #28, the KL coefficient
($0.05 \rightarrow 0.001$) is in the right direction because it reduces the
force opposing an already negligible policy gradient, the batch increase is
right because coverage was 26 of 198 problems, and the learning rate reduction
is harmful. The corrected run was cancelled before it consumed a night on a
configuration whose predicted outcome is total inertia.

### Why the gradient is negligible

The GRPO surrogate is $-\min(\rho A, \mathrm{clip}(\rho)A)$ with the advantage
z-scored inside its group, so advantages sum to zero and at the first
accumulation step $\rho = 1$ exactly. Measured policy loss across this run ran
$0.0008$ to $0.0042$. The loss is additionally a mean over tokens rather than a
sum, which divides the gradient by sequence length again. Small loss, heavily
normalised, pulled back by a KL penalty 50x stronger than the reference's: the
net parameter movement is what the table above shows.

### What this does to the project's history

The simplest account of twenty runs is now this. The reward terms were
debugged, narrowed, graded, re-weighted and extended; the measurement was found
wrong in four separate ways and corrected; and throughout all of it the policy
being optimised stayed within $0.2\%$ of its initialisation. Every null result
this project has reported is consistent with a tutor that never changed.

This is checkable in one minute from any saved adapter and it was never
checked. It should be the first diagnostic run against any future adapter here,
before any claim about reward design is made.

### Caveat

The base-probe control was stopped at 7 of 20 problems to release the GPU, so
there is no untrained reference for the probe series. Of those 7, 3 solved,
which is 0.429 against the trained probes' 0.150 to 0.350. That is **not a
result**: at $n{=}7$ the interval covers almost everything, and the problems are
the first 7 of the split rather than a sample. It is recorded only because the
direction is the opposite of the one the probe series suggested, and the control
remains unrun.

## 30. A prompted 3B dominates the trained tutor on every axis we measure

Run on 2026-10-10 on a Kaggle T4, 60 held-out problems, every arm paired on the
same bank in the same order, `hints` framing, all six instrumentation fixes in
place. This is the first time this project has scored a model it did not train.

| arm | solve | partial | leakage | pedagogy | min |
|---|---|---|---|---|---|
| oracle (solution disclosed) | 0.833 | 0.861 | n/a | n/a | 3.4 |
| **no tutor at all** | **0.433** | 0.500 | 0.000 | n/a | 3.0 |
| **prompted 3B, no training** | **0.417** | 0.478 | **0.017** | **0.997** | 15.0 |
| sahai-base (adapter off) | 0.350 | 0.433 | 0.249 | 0.768 | 22.9 |
| **sahai-trained** | **0.333** | 0.428 | 0.137 | 0.893 | 21.1 |

Qwen2.5-3B-Instruct, prompted with the same system prompt, no adapter, no
reward function, no training of any kind, beats the trained 1.5B on **all three
measured axes at once**: it solves higher, it discloses eight times less often,
and it satisfies the pedagogy rules almost perfectly. There is no frontier
position to claim. The system is Pareto-dominated.

The worst part is where it loses. Non-disclosure is the thing this project
spent twenty runs optimising, and it is the axis on which the gap is largest:
$0.137$ against $0.017$.

### Nothing in the solve column is statistically significant

| comparison | discordant | $p$ |
|---|---|---|
| unaided vs sahai-trained | 5:11 | 0.2101 |
| unaided vs prompted-3B | 6:7 | 1.0000 |
| sahai-trained vs prompted-3B | 9:4 | 0.2668 |
| sahai-base vs sahai-trained | 6:7 | 1.0000 |

At $n{=}60$ this protocol resolves about $0.15$, and no pair differs by that
much. So the solve-rate ordering above is the direction of the evidence and not
an established result, and the earlier claim that training made the tutor
*worse* than its own base is **not** supported: 6:7 discordant, $p{=}1.0$.

The leakage and pedagogy columns are a different kind of quantity, a mean over
60 dialogues of a continuous measure rather than a 60-sample binomial, and an
eight-fold gap there is not plausibly noise.

### The training did work, on the channel it could reach

| | base | trained |
|---|---|---|
| leakage | 0.249 | **0.137** |
| pedagogy | 0.768 | **0.893** |

Reinforcement learning moved both terms the policy computes from its own text,
substantially, in the intended direction. This is the asymmetry of finding #14
appearing one last time: the controllable channel responded, the outcome did
not, and the thing that was improved turns out to be available for free from a
model twice the size.

### Two corrections this run forces

The oracle arm's leakage and pedagogy are **not measured**. `run()` records
leakage as `0.0` and pedagogy as `nan` whenever `tutor is None`, and the oracle
arm is constructed with no tutor object even though its dialogue does contain a
disclosed solution. Its true leakage is 1.0 by construction. Earlier write-ups,
including the report sections drafted on 2026-10-08, describe the oracle as the
total-leakage corner of the frontier with a measured value; that value does not
exist and the table must say so.

The T4 numbers also agree with the local Apple Silicon run to within one
problem per arm (unaided 0.433 against 0.450, oracle 0.833 against 0.850),
which is the first cross-platform check this pipeline has had since the six
fixes.

### What follows

There is no version of this result that supports the project's original claim.
A 3B model with a prompt is better on outcome and better on pedagogy, at
two-thirds of the wall-clock, and it required none of the reward design, the
learner model, the curriculum or the reinforcement learning. The remaining
contribution is the measurement work and the diagnosis, which is real and which
no amount of further training will improve.

## 31. Only the complete answer helps. No tutor could have won this benchmark.

Run 2026-10-10 on a Kaggle T4, 60 held-out problems, every level paired on the
same bank. Six hint levels, each constructed programmatically from the
reference solution, so no tutor model is involved and no policy quality can
confound the measurement. This measures the **task**, not a tutor.

| level | content | solve | vs L0 | $p$ | leakage |
|---|---|---|---|---|---|
| L0 | nothing | 0.433 | --- | --- | 0.000 |
| L1 | generic Socratic question | 0.400 | $-0.033$ | 0.7744 | 0.006 |
| L2 | approach named from the solution's structure | 0.383 | $-0.050$ | 0.5488 | 0.006 |
| L3 | control-flow skeleton, bodies removed | 0.450 | $+0.017$ | 1.0000 | 0.774 |
| L4 | working code for a different problem, as analogy | **0.317** | $-0.117$ | 0.0654 | 0.350 |
| L5 | the reference solution | **0.833** | $+0.400$ | **0.0000** | 0.992 |

**No level short of complete disclosure beats giving no help at all.** L3 is
$+0.017$ at $p{=}1.0$, which is noise. L1 and L2 are below the floor. L4, the
one the literature predicted would work, is the **worst rung in the table**,
$0.117$ below no help at all and nearly significantly so.

The only thing that helps is L5, and L5 is the answer.

### What this settles

The $0.40$ gap between no help and disclosure is reachable only by disclosing.
On this benchmark **no tutor bound by a non-disclosure constraint can win**,
whatever its size, prompt, reward or training. That is a property of the task.

It explains, in one measurement, every null this project has produced, and it
subsumes the other explanations rather than competing with them. The policy not
moving (#29) and the tutor having no information advantage (#30) would both
have mattered if there had been anything to win. There was not. A prompted 3B
losing to no-tutoring in finding #30 is the same fact seen from a different
angle: it is not that the 3B tutors badly, it is that tutoring without
disclosure does nothing here.

MBPP is why. Its solutions are one to three line idioms, so there is no
intermediate reasoning state to scaffold. A skeleton of a one-line solution is
`def f(M): ... return ...`, which carries no information, and that is most of
the bank. Either the student knows `sorted(M, key=sum)` or it does not.

### The analogy rung hurting is the most interesting number here

L4 shows the student a correct, working solution to a *different* problem that
shares AST constructs with this one, and the student then does **worse than if
it had been shown nothing**. The plausible mechanism is that it copies the
analogous pattern instead of solving the problem in front of it, which would
make a well-intentioned worked example actively harmful. We have not confirmed
it: the ladder recorded outcomes and not the student's code, so the mechanism
is a hypothesis and the effect is $p{=}0.0654$ at $n{=}60$.

This is consistent with a published human finding rather than isolated.
Bastani et al. (IZA DP 18338, 2025) report that in secondary mathematics,
unrestricted AI assistance improved supported practice while **reducing
subsequent unaided examination performance** against a control. Our result is
the same shape in a simulated setting.

### A correction against my own analysis

The hypothesis this experiment was built to test was that our tutor's prompt
forbids the hint register that works. A think-aloud study of twelve novice
programmers (arXiv:2404.02213) reports that high-level natural-language hints
alone can be unhelpful or misleading while commented code examples support
novices better, and our prompt forbids code, code blocks and pseudocode. The
prediction was that L3 and L4 would recover much of the gap.

**They did not.** L3 is noise and L4 is harmful. The prompt constraint is not
the binding problem, and the 2404.02213 result does not transfer to a
simulated 1.5B student on one-line problems.

Worse, the script's own verdict block printed the flattering conclusion
anyway. It was written to fire "the constraint is the prompt and the reward,
not the policy" whenever the best permitted rung failed to beat L0 and *any*
rung beat it, without checking which rung. Only L5 beat L0, so the rule turned
"only the answer works" into "our prompt is the problem". The logic now
requires a rung that is not disclosure, and the earlier verdict is recorded
here as wrong rather than quietly deleted.

## 32. Partial information makes this learner worse, significantly, and that is not the benchmark's fault

Finding #31 found that on the 60-problem bank no hint short of full disclosure
beat no help, and offered the task as the explanation: MBPP references are one
to three line idioms, so a skeleton of one is a signature and a placeholder.
That was testable. It is now falsified.

Run 2026-10-10 on a Kaggle T4, 84 minutes, 822 generations. Every split pooled
and filtered to references with at least four statements in the function body
and at least one loop or conditional: 137 of 558 problems, mean 11.9 lines and
2.9 control-flow nodes. Skeletons on this set encode real branch sequences.

| level | 60 default | **137 multi-step** | $p$ vs L0 |
|---|---|---|---|
| L0 nothing | 0.433 | **0.336** | --- |
| L1 Socratic question | 0.400 | 0.263 | 0.0639 |
| L2 approach named | 0.383 | **0.234** | **0.0094** |
| L3 skeleton, logic removed | 0.450 | 0.321 | 0.8388 |
| L4 analogous worked example | 0.317 | **0.234** | **0.0243** |
| L5 the reference solution | 0.833 | **0.752** | **0.0000** |

### The skeleton still does nothing

$-0.015$ at $p{=}0.84$, on problems where the skeleton shows a four-branch
decision sequence rather than a placeholder. The benchmark explanation is dead:
making the structure informative did not make the structural hint work.

### Two rungs now harm the learner significantly

At $n{=}137$ the protocol resolves $16.6\%$ rather than the $25.1\%$ of the
60-problem bank, and two effects clear it. Naming the approach costs $0.102$
($p{=}0.0094$). Showing a correct worked solution to an analogous problem costs
$0.102$ ($p{=}0.0243$). Both are *worse than telling the learner nothing at
all*, and on the harder set the damage is larger and now measurable where
before it was only suggestive.

Only complete disclosure helps, and it helps enormously: $+0.416$.

### So the constraint is the learner, not the task or the tutor

This is the third explanation this project has offered for the same null and
the first one the data supports. It is not the reward design, which was
debugged across twenty runs. It is not the policy, which never moved. It is not
the prompt's no-code rule, since the rungs containing code do not help. It is
not the benchmark, since selecting for scaffoldable structure changed nothing.

A 1.5B learner given partial information performs **worse** than the same
learner given none. The plausible mechanism is anchoring: a partial hint
commits it to an incomplete pattern that displaces its own prior, which is
better than the fragment. L4 is the clearest case, since the learner is shown
working code for a different problem and plausibly transfers it rather than
solving the one in front of it. We have not confirmed the mechanism because the
run recorded outcomes and not the learner's code, and that is the single
cheapest follow-up available.

### What this means beyond this project

The simulated-student methodology has a problem, and it is not ours alone. A
tutor evaluated against an LLM learner is scored by a learner that is harmed by
partial, non-disclosing hints and helped only by the answer. That penalises
exactly the behaviour pedagogy requires, so a reward built on such a learner
pushes a tutor toward disclosure or toward silence, which is what every run of
this project measured. MathTutorBench, StudentSim and the simulated-learner
line of work all rest on this assumption, and we can find no paper that checks
it.

It is consistent with human evidence rather than contradicting it. Bastani et
al. (IZA DP 18338, 2025) report that in secondary mathematics unrestricted AI
assistance improved supported practice while reducing subsequent unaided
examination performance. The direction matches; whether the mechanism does is
unknown.

### Prediction recorded, and failed

The experiment was built expecting L3 to help here. It did not. This is the
second prediction in two days that the ladder falsified, after the
arXiv:2404.02213 hint-register hypothesis of finding #31. Both are recorded
rather than reframed.

## 33. The harm is real and the mechanism is not copying. Two routes, neither established.

Run 2026-10-10 on a Kaggle T4, 39 minutes, 411 generations with the learner's
code recorded. Three rungs on the same 137 multi-step problems: no hint, the
approach named in one sentence with no code, and a correct worked solution to an
analogous problem.

| rung | solve | sim to analogy | sim to reference | copied a name | $p$ vs L0 |
|---|---|---|---|---|---|
| L0 nothing | 0.336 | 0.233 | 0.359 | 2.9% | --- |
| L2 approach named | **0.234** | 0.205 | **0.319** | 2.9% | **0.0094** |
| L4 analogous example | **0.234** | 0.250 | 0.356 | 3.6% | **0.0243** |

### The anchoring hypothesis is dead

Finding #32 proposed that the learner transfers the analogous pattern instead
of solving the problem. It does not.

* Paired on the same problem, the shift in token overlap with the analogy
  between L4 and L0 is $+0.0169$, and only **64 of 137 problems moved toward
  the analogy at all, which is 46.7%, below chance**.
* The learner defines a function by the analogy's name on 5 of 137 problems
  under L4 against 4 of 137 under L0.
* On the 24 problems L0 solved and L4 lost, 16 moved toward the analogy,
  binomial $p{=}0.152$. Suggestive, not significant.

**The decisive argument is L2.** It contains no code at all, one sentence naming
the approach, and it harms by exactly the same amount as the worked example,
$0.234$ against $0.234$. If copying shown code were the mechanism, a rung with
no code to copy could not do equal damage.

### Two different damage routes, same magnitude

| | sim to reference | code size | on the problems it lost |
|---|---|---|---|
| L2, one sentence | $0.359 \rightarrow 0.319$ | nodes 76.2 $\rightarrow$ 78.8 | 9/20 wrote more code, $p{=}0.824$ |
| L4, worked example | $0.359 \rightarrow 0.356$ | nodes 76.2 $\rightarrow$ **84.4** | 17/24 wrote more code, $p{=}0.064$ |

Naming the approach moves the learner's code **away from the correct solution**
without moving it toward anything in particular. Showing a worked example
leaves it about as close to the reference but **inflates it**: eleven percent
more AST nodes, and on the problems it loses it writes more code than it did
unaided in 17 of 24 cases. Neither route reaches significance on its own, and
they are not the same route, so this is two weak signals rather than one
mechanism.

The clearest single case: on a median-of-two-arrays problem the learner was
shown a binary-search example, and attempted a full partition algorithm with
`float('-inf')` sentinels where unaided it had written something simpler that
passed. That is reaching for a more elaborate method than the problem needs,
which is the over-engineering route, and it is one anecdote.

### Hints do help sometimes. They break far more than they fix.

| rung | problems it lost | problems it gained | ratio |
|---|---|---|---|
| L2 | 20 | 6 | 3.3 to 1 |
| L4 | 24 | 10 | 2.4 to 1 |

This matters for how the result is stated. Partial help is not inert and it is
not uniformly harmful: it fixes six to ten problems the learner could not do
alone and breaks twenty to twenty-four it could. The aggregate is negative
because the breakage is larger, not because help never works.

### Third failed prediction, recorded

The register hypothesis of #31 failed, the benchmark hypothesis of #32 failed,
and the anchoring hypothesis of this run failed. The effect is firmly
established across two ladders and 197 problems; **no proposed mechanism for it
survives contact with the data**, including all three of mine. The honest state
is that partial information degrades this learner through at least two distinct
and individually unproven routes.

That is where this line of investigation should stop. The effect is what the
paper needs and the effect is solid. Another GPU run buying a third
underpowered mechanism signal is not worth it; what would settle it is a
controlled study varying one property of the hint at a time at a sample size
that resolves $0.05$, which is roughly 800 problems per rung.
