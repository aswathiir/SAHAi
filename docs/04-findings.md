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
