# Internals

The actual mechanics, with the real numbers. Nothing here is simplified for
presentation — where a formula is implemented differently from how it is
usually written, that is stated.

---

## 1. The reward

```
r_SAHAI(a_T | s_T, α) = (r_sol − α) + (r_ped − 1)·λ − γ·L
```

`sahai/reward/combined.py`. With `λ = 1.0`, `γ = 0.5`.

| Symbol | Meaning | Range |
|---|---|---|
| `r_sol` | fraction of `K` post-dialogue student attempts that pass **every** unit test | [0,1] |
| `α` | the student's traced ability, from BKT | [0,1] |
| `r_ped` | mean of 5 pedagogy checks, four scored per tutor turn | [0,1] |
| `L` | leakage | [0,1] |

**Why `r_sol − α`.** A tutor should get no credit for a student who could
already solve the problem. Subtracting the traced ability makes the reward
measure *improvement attributable to the tutoring*, not raw success.

**Hard penalty.** If `r_ped == 0`, the whole reward is replaced by `−λ`. This is
from the literature — pedagogical acceptance is a prerequisite, not a term to
trade off. It fires only on total failure now that `r_ped` is graded.

Worked example from a real rollout:

```
r_sol = 0.00, α = 0.30, r_ped = 0.40, L = 0.14
r_SAHAI = (0.00 − 0.30) + (0.40 − 1)(1.0) − 0.5(0.14)
        = −0.30 − 0.60 − 0.07
        = −0.97
```

### 1.1 `r_sol` — execution, not opinion

`sahai/reward/solve.py`. After the dialogue ends the student is asked to write a
solution `K` times. Each attempt is written to a script, run in a subprocess,
and its return value compared to the expected output. `r_sol` is the fraction of
attempts passing **all** test cases — partial passes score zero, matching the
all-or-nothing correctness indicator in the literature.

`K = 4` on Kaggle (the review recommends `K ≥ 8`). This is deliberate: `r_sol`
is currently ~0 because the 0.5B student cannot solve MBPP, so extra samples
measure noise more precisely at 50% of total runtime. Raise `K` once the student
can actually solve.

### 1.2 `r_ped` — five graded checks

`sahai/reward/pedagogy.py`. Applied to tutor turns only:

1. no code blocks (bare ``` counts, to catch fences truncated at the token cap)
2. no solution patterns (`def f(`, `return [`, `for x in range`, `while x <`)
3. at least 30% of tutor turns contain `?`
4. no tutor turn exceeds 200 words
5. no tutor turn ends on `:` — a promise it never delivers

Score = fraction passed. **A dialogue with no tutor turns scores 0.0**, not 0.8:
every "no X" check passes vacuously when there is nothing to inspect, which
would otherwise reward a silent tutor.

### 1.3 `L` — leakage, verified by execution

`sahai/reward/leakage.py`.

The original implementation compared tutor text to the reference solution by
token overlap. Measured on a real rollout it scored **0.000** for a tutor that
wrote a complete working solution, because it used a dict where the MBPP
reference used `enumerate`/`count`. It scored 1.000 only for a verbatim paste.
It was measuring plagiarism of one implementation, not leakage.

Current logic:

```
if tutor code passes the problem's own tests:  L = 1.0     # complete leak
elif tutor wrote any code:                     L ≥ 0.5     # fragment or broken
else:                                          L = token overlap, problem words excluded
```

Two details that matter:

- **Function aliasing.** The leaking tutor named its function
  `find_first_repeated_char` while the problem expected `first_repeated_char`.
  Without an alias the verifier cannot call it and a complete leak reads as
  zero. `runnable_variants()` appends `expected = actual` and re-runs. Renaming
  a function does not make it less of a leak.
- **Problem words are subtracted.** A tutor cannot discuss a string problem
  without saying "string". Including those words gave the metric a floor that
  made its absolute value meaningless (a clean Socratic question scored 0.14;
  it now scores 0.000).

`sahai-core` performs **no execution** — it takes `tutor_code_solves` as an
input. Running untrusted code belongs to `CodeVerifier` (training) or
`services/executor` (serving).

---

## 2. GRPO — how the policy is updated

`sahai/training/grpo.py`.

### 2.1 Advantage

No value network. For each group of `G` rollouts on the same problem:

```python
mean_r = sum(rewards) / len(rewards)
std_r  = sqrt(sum((r - mean_r)**2 for r in rewards) / len(rewards))
advantage = (reward - mean_r) / (std_r + 1e-8)
```

**Within-group reward variance is the entire learning signal.** If every rollout
in a group scores identically, `std_r = 0`, every advantage is 0, and the epoch
updates nothing. This is not hypothetical — it is what happened when `r_ped` was
binary, and the log printed `loss=0.0000`.

Because the advantage is a z-score, the *scale* of the reward is irrelevant;
only the ordering and relative spread matter.

### 2.2 Loss

```python
mean_lp     = (log_probs * mask).sum() / token_count
policy_loss = -advantage * mean_lp
kl          = ((log_probs - ref_log_probs) * mask).sum() / token_count
loss        = policy_loss + kl_coeff * kl        # kl_coeff = 0.05
```

**Stated honestly: this is REINFORCE with a KL penalty, not clipped GRPO.**
`clip_epsilon` exists in settings and is unused — there is no importance ratio.
Acceptable for a single inner epoch over freshly sampled rollouts, but the
method should not be described as clipped GRPO until the ratio term is added.

Also, `mean(log π − log π_ref)` is **not** a KL divergence and can go negative.
The k3 estimator `exp(δ) − δ − 1` would be correct.

### 2.3 The reference policy

`π_ref` is the *same model with the LoRA adapters switched off*:

```python
self.tutor.model.disable_adapter_layers()
# ... compute reference log-probs ...
self.tutor.model.enable_adapter_layers()
```

No second copy of the model in memory. This is also the anti-forgetting
mechanism: the KL term penalises drift away from the untrained base.

### 2.4 Token masking — why only tutor tokens

`TutorPolicy._build_tutor_mask` walks the message list, re-templating each
prefix to find where each assistant span begins and ends, and marks those
positions. Log-probs are multiplied by this mask.

**This is the constraint that shapes the whole design.** Because gradients flow
only through tutor tokens:

- the **student** can be freely post-processed (trimmed, scripted, filtered) —
  its turns never enter the loss
- the **tutor** cannot be touched — editing its text would compute log-probs
  over tokens the model never sampled, which is off-policy and wrong

That is why the dangling-colon problem is fixed by *trimming* on the student
side and by *reward* on the tutor side.

The mask is built by re-tokenizing every message prefix, which is O(n²) in turns
— a known inefficiency, not a correctness issue.

---

## 3. Generation hygiene

`sahai/core/generation.py`. Three problems, all found in real transcripts.

**Truncation.** A turn cut at `max_new_tokens` ends mid-sentence, gets appended
to the dialogue as a chat message, and the next model *completes the sentence*
instead of replying. Fixed by checking whether generation stopped on EOS
(`hit_terminator`) and, if not, trimming back to the last sentence boundary.

A subtlety: the sentence-boundary regex excludes a `.` preceded by a digit, so
numbered list markers (`3. Count the...`) are not treated as sentence ends.

**Dangling promises.** 19% of turns ended in `:` — *"Here is the
implementation:"* — which invites the next speaker to fulfil the promise. This
is the mechanism behind tutor/student role inversion. Trimmed on the student,
penalised on the tutor.

**Unclosed code fences.** `strip_code` originally used a paired-fence regex,
so a turn truncated mid-block left an unclosed fence that survived scrubbing.
Now the trailing unclosed fence is dropped first. `strip_code` is display-only
and never applied to a trajectory.

---

## 4. Learner model — BKT

`sahai/agents/tracer.py`. Standard Bayesian Knowledge Tracing, one probability
per skill, with `p_init=0.3, p_learn=0.1, p_guess=0.2, p_slip=0.1`.

Posterior after observing a correct answer:

```
P(L|correct)  = P(L)(1−p_slip) / [ P(L)(1−p_slip) + (1−P(L))p_guess ]
```

after an incorrect one:

```
P(L|incorrect) = P(L)p_slip / [ P(L)p_slip + (1−P(L))(1−p_guess) ]
```

then learning is applied in both cases:

```
P(L)' = P(L|obs) + (1 − P(L|obs))·p_learn
```

**BKT, not DKT, deliberately.** DKT needs large interaction logs we do not have
— the student is synthetic and every session starts cold. BKT works from the
first interaction and yields an interpretable per-skill number, which the reward
needs as `α` and the curriculum needs for ZPD filtering. This is a data-scale
decision, not a claim that BKT predicts better.

**ZPD sampling.** `ProblemBank.zpd_sample` draws only problems whose mean skill
mastery falls in `[0.3, 0.7]` — not already mastered, not hopeless.

---

## 5. Sandbox — running untrusted code

`services/executor/app/sandbox.py`. Layered, because any single layer can fail.

| Layer | Control |
|---|---|
| network | `sandbox` docker network, `internal: true` — no egress at all |
| filesystem | `read_only: true`; only a 64 MB tmpfs at `/tmp` is writable |
| privileges | non-root uid 10001, `cap_drop: ALL`, `no-new-privileges` |
| resources | `pids_limit: 128`, `mem_limit: 1g`, `cpus: 1.0` |
| process | fresh interpreter per test case, `python -I -S`, own session, scrubbed env |
| rlimits | AS, DATA, CPU, NOFILE, FSIZE, NPROC applied between fork and exec |
| protocol | result read from a sentinel line, so candidate stdout cannot forge a pass |

Two implementation details that cost real debugging time:

- **`RLIMIT_CPU` delivers SIGXCPU (−24), not SIGKILL.** Checking only for
  SIGKILL misreports an exhausted infinite loop as a generic error.
- **`preexec_fn` must not call `os.setsid()`** when `start_new_session=True` is
  already set — the second call fails with EPERM and surfaces as an opaque
  `SubprocessError`.

The rlimits are fast-failure guards; the container is the actual boundary.

---

## 6. Session state machine — why serving cannot reuse the trainer

`services/session/app/main.py`.

| | training | serving |
|---|---|---|
| student | a model the engine calls in a loop | a person sending one turn and waiting |
| control flow | batch, start to finish, offline | event-driven across HTTP requests |
| lifetime | one process, discarded | must survive restarts |

`DialogueEngine.run(tutor, student, problem)` only makes sense on the left. In
serving, the conversation is a row in Postgres and each turn is a state
transition.

Two SQLAlchemy details, both of which produced silent wrong behaviour:

- **`selectinload`** — a lazy relationship accessed outside the async session's
  greenlet raises `MissingGreenlet`.
- **`populate_existing=True`** — the session uses `expire_on_commit=False`, so a
  re-query returns the identity-mapped object with its previously loaded `turns`
  collection intact. Without this the API returned a transcript **one exchange
  behind the database**, with every status code 200 and nothing in the logs.

---

## 7. Observation log — why it exists before DKT does

`services/tracer/app/main.py`.

`SkillMastery` holds the current BKT posterior. That is a **cache** — it can be
rebuilt by replaying the observation log. The reverse is not true.

`Observation` is append-only: `(learner, skill, correct, problem_id, timestamp,
mastery_after)`. It was built while BKT is still what runs, because DKT trains
on the *sequence* and history cannot be collected retroactively.

`problem_id` is recorded because DKT predicts per-exercise, not only per-skill —
dropping it would make the log unusable for anything but skill-level models.
`mastery_after` lets a later model be compared against what BKT believed at the
time.

A test asserts the invariant: replaying the log through a fresh `BKTTracer`
reproduces the stored mastery exactly.
