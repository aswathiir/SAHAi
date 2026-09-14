# SAHAi — Roadmap

Four phases. Phase 1 is done; the rest are gated on each other in order.

**Headline: the gate has moved.** For most of this project Phase 2 was blocked
on not knowing *why* training was not learning. That question is now answered,
and the answer was not any single bug.

Across all three completed 10-epoch runs, `corr(epoch, r_ped)` is
+0.80 / +0.99 / +0.64 and `corr(epoch, leak)` is −0.64 / −0.80 / −0.48, while
`corr(epoch, r_sol)` is +0.48 / +0.54 / **+0.01**. The two deterministic
rule-based terms move strongly and in the intended direction in every run; the
outcome term moves in none of them. Separately, on the 24 rollout records still
on disk — an earlier run, so indicative rather than matched to those three —
mean *within-group* variance, which is the only variance GRPO's advantage can
see, was 0.0482 for the pedagogy term against 0.0098 for the solve term, with
22 of 24 rollouts scoring `r_sol` exactly 0.00.

A policy optimises whatever part of its reward it can control. `r_ped` and `L`
are deterministic functions of the tutor's own text; `r_sol` was a four-sample
estimate dominated by a frozen 1.5B student's sampling noise. Six runs of fixes
all landed on the controllable channel, and the thing the project exists to
train was never trained. Full derivation in
[`04-findings.md` §14](04-findings.md).

The fixes that follow from it are implemented, tested and pushed. **Phase 2 is
now blocked on GPU quota, not on knowledge** — and nothing downstream can be
validated until one more run happens.

```
Phase 1  Foundation & infrastructure    ████████████████████  done
Phase 2  Make training actually learn   ██████████░░░░░░░░░░  diagnosed, fixes staged, BLOCKED on compute
Phase 3  Indic / code-mixed             ████████░░░░░░░░░░░░  voice wired; base model still the gap
Phase 4  Knowledge tracing & validation ███████░░░░░░░░░░░░░  learner-state layer shipped; no real learners
```

A checkbox count says ~70%. Weighted by remaining risk it is closer to
**45–50%**, and the weighting is dominated by one unresolved question: whether
a tutor trained on an attributable `r_sol` learns anything a held-out problem
can detect. Every estimate above that line is engineering; everything below it
is unvalidated.

---

## Phase 1 — Foundation & infrastructure ✅ **done**

Everything needed to run, serve, observe, and debug the system.

| Item | State |
|---|---|
| GRPO training loop, checkpointing, rollout dumps | done |
| Clipped surrogate with importance ratio | done — was REINFORCE until 2026-09-13 |
| `r_sol` — execution-verified, greedy, partial credit | done |
| `r_ped` — 3 graded rule checks (was 5) | done |
| `L` — execution-verified leakage, rare-token filtered | done |
| BKT tracer + difficulty-targeted ZPD sampling | done |
| Six services in Docker: gateway, session, tutor, executor, tracer, asr | done |
| Executor containment: no egress, read-only rootfs, rlimits | done — 13 tests |
| Persistent per-learner mastery in Postgres, append-only observation log | done |
| Request tracing across services, aggregate health, deadline ladder | done |
| `inspect_rollouts.py`, `smoke_test.py`, `check_policy_update.py` | done |
| Learner interface: problem picker, chat, submit, mastery, progress, diagnostic, resume | done |
| Voice I/O over a websocket | done |
| **243 tests** — 91 research, 30 shared library, 122 across five services | done |

**Gaps left deliberately:**

- `services/tutor` still has **no tests**. It is the thinnest service (load
  adapter, generate, sanitise) but it is also the only one holding a GPU model.
- Gateway rate limiting is untested. Identity/401 handling is covered by 54
  tests; `_rate_limit` is not among them.
- Identity is a bearer token verified at the gateway. It authenticates a
  *bearer*, not a person — no second factor, no recovery, no revocation beyond
  deleting the row — and the internal hop to session and tracer still trusts
  `x-learner-id`, which is a network-topology assumption rather than a
  cryptographic one. Both are written down where they apply rather than
  implied.

---

## Phase 2 — Make training actually learn 🔴 **critical path, blocked on compute**

### Run history

All six completed runs were evaluated on the same 20-problem held-out set,
whose mean has sd 0.082 — so the whole spread below is about 1.3 standard
deviations and **every comparison rests on roughly two problems**
([§12](04-findings.md)).

| run | what changed | held-out solve |
|---|---|---|
| v6 | first completed run; truncation uncontrolled | 6.3% |
| truncation fix | `max_new_tokens` 192→256 / 160→224 | 11.25% |
| v14 | lr, batch size and pedagogy scoring changed together | net negative, **not attributable** |
| v15 | rare-token leakage filter | **16.3%** — best |
| v16 | question-fraction reward added | 6.2% |
| 2026-09-12 | pedagogy cut to 3 answer-giving checks | 13.8% |

Sorted by *kind* of change rather than by version, the pattern is the finding:

| kind of change | examples | outcome |
|---|---|---|
| removed a distortion | truncation cap, 1.5B student, leakage eval bug, rare-token filter, length bias | every gain came from here |
| added an incentive | question fraction | the only clear regression |

### Done

| Item | Evidence |
|---|---|
| Dangling-promise trim | shipped; 6.8% of turns still end in `:` (down from 19%) |
| Graded `r_ped`, empty-dialogue guard | silence scored 0.8, now 0.0 |
| Execution-verified leakage | a full solution leak scored 0.000, now 1.000 |
| Sentinel-based result parsing | candidate `print()` was silently breaking `r_sol` in every prior run |
| Truncation fix, re-run, validated | 6.3% → 11.25%, the project's first controlled before/after |
| Student 0.5B → 1.5B | role inversion was the best-evidenced lever; shipped |
| Leakage scored the same way in training and eval | the evaluator never called `tutor_code_solves`; leak was inflated 34% relative |
| Rare-token leakage filter | stopped punishing the tutor for ordinary teaching vocabulary |
| Question-fraction reverted, pedagogy cut to 3 checks | reward hacking verified directly, not inferred |
| **Importance ratio + clipping** | was `-advantage * mean_log_prob`; `clip_epsilon` had sat unused since the beginning. **It is correct to call this GRPO now.** |
| **`r_sol` made attributable** | greedy decode + partial credit; within-group variance now comes from the tutor, not the student's dice |
| **Difficulty-targeted ZPD** | the old band meant "skills never attempted" and was empty from epoch 3 onward |

### Staged and unrun — waiting on a GPU

Everything here is in `main` and covered by tests, and **none of it has been
executed**. This is the entire content of the next run.

| Change | Expected effect |
|---|---|
| Greedy, partial-credit `r_sol` | the outcome term gets a gradient for the first time |
| Clipped surrogate, ratio against pre-step log-probs | 15 of 16 optimizer steps per epoch stop being uncorrected off-policy |
| `lora_dropout` 0.05 → 0.0 | dropout alone moved \|ρ−1\| as far as 0.065, a third of the clip band |
| Difficulty-targeted ZPD selection | band holds 95–245 problems every epoch instead of collapsing to 0 |
| Held-out eval 20 → 60 problems | sd 0.082 → 0.047 — **requires a manual notebook edit, see below** |

**Budget:** ~6.1 h training + ~0.6 h eval = **~6.9 h** of the 12 h cap, against
9.6 h for the last run. The greedy change pays for the wider eval outright.

**Before the next run**, in order:

1. Paste the eval cell from [`07-operations-kaggle.md` §2b](07-operations-kaggle.md)
   into the Kaggle browser editor. `settings.eval_problems = 60` shipped in the
   dataset; the kernel is a separate upload and still holds `max_problems=20`.
   The last run evaluated on 20 problems while the log, the settings file and
   the commit message all said 60.
2. `rsync -a --delete --exclude='__pycache__' sahai/ kaggle_upload/sahai/`, then
   `kaggle datasets version -p kaggle_upload --dir-mode zip -m "..."`.
3. Check the sidebar says **GPU T4 x2**, then Save & Run All. Never
   `kaggle kernels push` — it resets the accelerator to P100 and the run dies
   at ~45 s.
4. Confirm the log prints `Evaluating on 60 test problems...`. If it says 20,
   kill the run rather than spending seven hours on an unreadable result.

**How to read the result:** on `solved`, the all-or-nothing column, against the
16.3% best. `solve` now carries partial credit and will read higher for reasons
that are not the policy. Ignore the final training epoch — last run it read
0.367 and looked triumphant while held-out collapsed.

**One-variable discipline is knowingly broken here.** The next run changes four
things at once. v14 did the same and came out net negative with none of its
three changes attributable. The justification is that three of these four target
the same root cause and holding any of them back leaves the gradient broken —
but if the result is ambiguous, this is why.

### Not done

| Item | Why it matters |
|---|---|
| **Generalisation is still low in absolute terms** | 16.3% best held-out is real but far from a usable tutor |
| **Conceptual leakage is unpriced** | The highest-reward epoch-0 rollout states the whole algorithm in English prose. All three pedagogy checks are syntactic and leakage is lexical, so nothing catches it. Deliberately not fixed before the staged run: adding terms to the controllable channel is what failed in v16. |
| **Retention benchmark** | LoRA + KL *should* prevent forgetting. Nothing measures whether it did. Needs a GPU. |
| k3 KL estimator | `mean(log π − log π_ref)` can go negative; it is not a KL. Cheap, and the tiny-model harness in `check_policy_update.py` can validate it without a GPU. |
| **ACE causal probes** | `ace_leakage()` exists and has never been called: `estimate()` takes `solve_with`/`solve_without` and the trainer passes neither. It fires when the student solved *and the tutor made no difference* — i.e. a solve the tutor did not cause — which is precisely the failure [§14](04-findings.md) is about. It was gated on "randomised hint ablations", and greedy decoding has made it cheap: `solve_without` is one extra greedy attempt on an **empty** dialogue, computed once per problem and shared by all 8 rollouts in the group — 4 extra generations per epoch. Because it thresholds on each rollout's own `solve_with`, it varies within the group and so does reach the gradient, unlike a plain per-problem baseline which the z-score would cancel. **Strongest candidate for the run after the staged one.** Not before: it is still a reward change, and this run already changes four things. |
| `K ≥ 8` solve samples | **Moot now.** The student decodes greedily, so repeated draws are identical; `num_solve_samples` is clamped to 1. Restoring K means restoring sampling, and that is what made `r_sol` noise. |
| Held-out set is still 20 problems in practice | Fixed in settings, not yet in the kernel — see the checklist above |

---

## Phase 3 — Indic / code-mixed 🟡 **voice wired, base model still the gap**

**Scope:** the tutor (Qwen) is the only thing trained. Voice is added on top by
wrapping it with off-the-shelf pretrained Indic ASR and TTS — no fine-tuning, no
WER/CER benchmark.

### Done

- `CODE_MIXING_TEMPLATES` for Hinglish / Tanglish / English, used for the
  student's opening turn — confirmed live
- **ASR wired**: `ai4bharat/indic-conformer-600m-multilingual`, CTC decoding.
  Audio is decoded through an `ffmpeg` subprocess; `torchaudio.load` needed
  TorchCodec, which was absent, so this path had **never worked** before.
- **TTS wired**: `ai4bharat/indic-parler-tts`, capped at 45 s of generation via
  `max_time`. An abandoned request previously held 400% CPU for ten minutes and
  starved the next one — a client timeout does not cancel server-side work.
- Voice runs over a websocket calling the **same** `/sessions/{id}/turns`
  endpoint as the text box, so it inherits learner context, the full problem
  statement and per-turn adaptation without duplicating any of it
- **Per-turn adaptation** (`sahai_core.turn_signals`): the tutor reads whether
  the learner is stuck, asking to be handed the answer, or has pasted code, and
  whether they wrote Hinglish — each rule grounded in a specific failure in the
  run transcripts

### Not done

| Item | Note |
|---|---|
| **Indic-capable base model** | Still Qwen2.5-**1.5B**, weak on romanized Indic — visible in the transcripts as Hinglish degenerating into repetition and Devanagari word-salad that still scored 0.886. A base-model decision, not a prompt one. |
| Code-mixed evaluation set | No way to measure whether Indic tutoring is good or bad. The **set itself can be built without a GPU**; only scoring needs one. |
| Indic-aware pedagogy checks | The 3 checks are English-shaped; needs verifying across scripts |
| Training/serving prompt divergence | The served adapter was trained under a prompt with no `THIS TURN:` block and without the dangling-colon or Hinglish rules. Accepted deliberately — no GPU to re-run — but it is a real gap between what was optimised and what is deployed. |

---

## Phase 4 — Knowledge tracing & human validation 🟡 **layer shipped, no real learners**

### Done

- **Append-only observation log** — `(learner, skill, correct, problem_id,
  timestamp, mastery_after)`, written on every submit, with a pageable replay
  endpoint and a test proving mastery is reproducible by replaying it. Built
  early on purpose: DKT trains on the *sequence*, and history cannot be
  collected retroactively.
- **Placement** (`POST /v1/me/placement`) — self-assessment seeds per-skill
  priors into `[0.15, 0.65]`. Two rules pinned by tests: it only seeds skills
  with no observations, and it writes nothing to `observations`.
- **Solved-repo import** (`scripts/import_neetcode.py`) — paths and dates only,
  with bulk-commit discounting, a NeetCode-specific tag table, and absolute
  thresholds
- **Progress page** — streak, activity grid, per-skill mastery, course tracks,
  next-up recommendation, resumable sessions, all projected from the log
- **Timed code diagnostic** — a graded placement alternative to self-report

### Not done

| Item | Note |
|---|---|
| DKT model | Needs interaction sequences at a scale a synthetic student won't produce |
| Swap DKT behind the tracer API | Cheap once the model exists — `/mastery`, `/observe`, `/zpd` already isolate it |
| Skill taxonomy mapping | ASSISTments / EdNet won't share MBPP's skill vocabulary. Worth deciding before much data accumulates. |
| Importer steps 2 & 5 | Reading the learner's own solution *source* and letting the tutor reference it. Blocked on a design question, not effort: it points a learner's working solutions at a tutor whose entire reward exists to stop solutions reaching them. |
| Sim→real pilot (H1–H4, N=34) | Not begun |
| IRB, consent, pre-registration | Not begun |

---

## Critical path

```
GPU quota  ──►  one run with the staged fixes  ──►  everything else
    │                        │
    │                        ├──► retention benchmark   (needs a trained model)
    │                        ├──► Phase 3 base swap     (pointless before the loop works)
    │                        └──► Phase 4 DKT           (needs real learners AND a usable tutor)
    │
    └──► nothing else in Phase 2 is worth doing first
```

The order is still forced, but the binding constraint has changed. It used to be
*understanding*; it is now *compute*. A base-model swap or a second reward term
before the staged run would just add unknowns to a result that already changes
four things at once.

## Next actions

1. **Run the staged config the moment quota returns.** Follow the four-step
   checklist in Phase 2 — especially the notebook edit, because the last run's
   headline number was measured on a set a third the size everything claimed.
2. **Build the code-mixed evaluation set.** The only Phase 3 item that needs no
   GPU, and Phase 3 currently cannot say whether Hinglish tutoring is good or
   bad at all.
3. **k3 KL estimator.** Small, long-standing, and now verifiable without a GPU
   using the harness written for the clipped surrogate.

Queued behind those, in order, once the staged run has been read:

4. **Wire ACE.** It is the design's own answer to "was this solve caused by the
   tutor", it is already written, and greedy decoding has made the missing
   input cheap to produce.
5. **Price conceptual leakage.** The largest remaining hole — the tutor can
   state the whole algorithm in English and score `ped 1.00, leak 0.00`.

Both are reward changes, and both sit in the channel that was gamed in v16. One
per run, and only after a clean reading of the staged configuration. The lesson
from v14 is not "avoid reward changes" — it is "do not make three of them at
once and then try to attribute the result".
