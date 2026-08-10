# Four-way work split

Split along the **module boundaries that already exist**, so two people rarely
need the same file. Each track lists what is already done (so it can be owned
and explained) and what is next.

The dependency between tracks is real and asymmetric: **Track A gates the
project.** B, C and D can proceed in parallel, but none of their results mean
anything until A produces a tutor that demonstrably learns.

```
  A  RL & Reward ─────────────────────┐ gates everything
  B  Platform & Infrastructure        │ independent
  C  Learner Modelling & Data         │ independent until DKT needs real data
  D  Language & Evaluation            ┘ needs A for retention benchmark
```

---

## Track A — RL Training & Reward Design

**Owns:** `sahai/training/`, `sahai/reward/`, `sahai/agents/tutor.py`,
`libs/sahai-core/sahai_core/reward/`, `scripts/inspect_rollouts.py`

### Already done (yours to own and defend)

- GRPO trainer: rollout → reward → advantage → masked policy update, LoRA + KL
  to a reference policy with adapters disabled
- All three reward terms and their composition
- **The binary→graded `r_ped` finding** — that a binary conversation-level
  reward zeroes the gradient under a critic-free algorithm. This is the
  strongest result the project has; be able to derive it on a whiteboard.
- **Execution-verified leakage** — including the function-aliasing case
- Empty-dialogue guard (silence scored 0.8, now 0.0)
- Rollout dumps, zero-spread warnings, the analyzer

### Next

1. **Analyse the scaled run** (G=8, 20 epochs, 320 dialogues). Does `ped` trend?
   How many groups had zero spread?
2. **Add the importance ratio + clipping.** `clip_epsilon` is defined and
   unused — the update is currently REINFORCE with a KL penalty. Until this is
   done the method should not be called clipped GRPO in the report.
3. **Replace the KL estimator** with k3 `exp(δ) − δ − 1`. The current
   `mean(log π − log π_ref)` can go negative and is not a KL.
4. **Wire ACE causal probes** — randomized hint withholding in 10–20% of
   rollouts, feeding `solve_with`/`solve_without`. This is the review's central
   defence against rewarding non-causal solves and is currently dead code.

**Must be able to explain:** why within-group variance is the entire learning
signal; why only tutor tokens are masked into the loss, and what that forbids.

---

## Track B — Platform & Infrastructure

**Owns:** `services/`, `docker-compose.yml`, `Makefile`, `scripts/smoke_test.py`

### Already done

- Six services: gateway, session, tutor, executor, tracer, asr
- **Executor containment** — no egress, read-only rootfs, dropped capabilities,
  pids/memory/CPU limits, rlimits, sentinel-protected result protocol. 13 tests
  including fork bomb, memory bomb, infinite loop, and a candidate that tries to
  forge its own pass.
- Session state machine over a persisted transcript — including the
  `populate_existing` staleness bug and its regression test
- Request-id tracing across gateway → session; aggregate health
- End-to-end test running all four services in one process, no mocks

### Next

1. **Tests for gateway, tutor, asr** — currently zero. The gateway handles auth
   and rate limiting untested.
2. **Finish the trace** into tutor, executor and tracer. They receive
   `x-request-id` but do not log it, so a failure inside the executor is not
   findable by request id — which is exactly the case where tracing matters.
3. **Alembic migrations.** Tables are created by `create_all` at startup. Fine
   for adding a table, wrong for changing one.
4. **CI** — run `make test` on push. 100 tests exist and nothing runs them
   automatically.

**Must be able to explain:** why the executor is on its own network and session
bridges two; why a request with no log line never arrived.

---

## Track C — Learner Modelling & Data

**Owns:** `sahai/agents/tracer.py`, `services/tracer/`,
`libs/sahai-core/sahai_core/domain/mastery.py`, `sahai/core/data.py`

### Already done

- BKT tracer: per-skill posterior, ability baseline `α`, ZPD curriculum sampling
- Persistent per-learner mastery in Postgres — the research code kept this in
  memory inside the *simulated* student, which is wrong for real learners
- **Append-only observation log** with `problem_id` and `mastery_after`, plus a
  test proving mastery is reproducible by replaying it
- Pageable replay endpoint for a future training job

### Next

1. **Decide the skill taxonomy.** MBPP tags (`strings`, `hash_maps`) will not
   match ASSISTments or EdNet. Settle this **before** much data accumulates —
   remapping later is painful.
2. **Implement DKT** (LSTM over the interaction sequence) and evaluate against
   BKT on the same replayed log. `mastery_after` exists precisely for this
   comparison.
3. **Swap it behind the tracer API.** `/mastery`, `/observe`, `/zpd` already
   isolate the implementation — nothing else in the system should change.
4. **Bootstrap data**, since a synthetic student will not produce realistic
   learning curves.

**Must be able to explain:** the BKT update equations; why BKT and not DKT
*today*; why mastery is a cache and the log is the record.

---

## Track D — Language, Evaluation & Forgetting

**Owns:** `sahai/agents/student.py`, `sahai/core/asr.py`, `services/asr/`,
`sahai/eval/benchmark.py`

### Already done

- Code-mixing templates for Hinglish / Tanglish / English, used at generation
  time for the scripted opening turn — confirmed live
- Per-skill injected misconceptions in the student persona
- ASR service boundary, contract and health check (stubbed)
- Held-out evaluation producing `MetricsReport`

### Next

1. **The retention benchmark.** The system already has both standard
   anti-forgetting mechanisms — LoRA freezes the base weights, and the KL term
   penalises drift from `π_ref`. **Nothing measures whether they worked.** Run
   MBPP pass@1 with adapters on vs off, before vs after training. Cheap, and it
   converts "LoRA should prevent forgetting" into a number.
2. **Base model decision for Indic.** Qwen2.5-1.5B is weak on romanized Indic
   and no prompt fixes that. Evaluate a larger Qwen against an Indic-specialised
   base.
3. **Build a code-mixed evaluation set.** There is currently no way to say
   whether Indic performance is good or bad.
4. **Real ASR** — code-mixed Indic speech; `whisper-large-v3` is mediocre here.

**Must be able to explain:** why LoRA + KL are the forgetting mitigations and
why measurement is the missing half; why continued pretraining (unlike LoRA) is
where forgetting risk is genuinely high.

---

## Shared rules

- **`libs/sahai-core` is shared.** Changes there affect every track — flag them.
- **Training and serving still have duplicate reward code** (`sahai/reward/` and
  `libs/sahai-core/sahai_core/reward/`). Two bugs so far had to be fixed twice.
  Merging them onto `sahai-core` is joint A+B work and should happen soon.
- **`make test` must pass before any push.** 100 tests today.
- After changing `sahai/`, run
  `rsync -a --delete --exclude='__pycache__' sahai/ kaggle_upload/sahai/` or the
  Kaggle run uses stale code.

## Suggested first week

| Track | Deliverable |
|---|---|
| A | Scaled-run analysis + importance ratio and clipping implemented |
| B | Tests for gateway/tutor/asr, trace finished into all services, CI running |
| C | Skill taxonomy decided and documented; DKT training script reading the log |
| D | Retention benchmark implemented and run against the current adapter |

A's deliverable is the one the others depend on. If the scaled run shows no
learning, that becomes everyone's problem and the plan is re-cut around it.
