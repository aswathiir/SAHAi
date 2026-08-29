# SAHAi — Roadmap

Four phases. Phase 1 is done; the rest are gated on each other in order.

**Headline:** the *infrastructure* is essentially finished. The *science* has
started and just produced its first real, controlled evidence of progress. A
naive checkbox count says 62% complete; weighted by effort and risk it is
closer to **38–42%**. Two scaled runs now exist on the same fixed 20-problem
held-out set: kernel v6 (truncation uncontrolled) scored 6.3% held-out solve;
after raising `max_new_tokens` to fix the truncation bug, the next run scored
**11.25%** — a real, apples-to-apples improvement, not noise. It came with a
cost, though: held-out teaching acceptance dropped (80%→71%) and leakage rose
slightly (9.9%→11.25%), so the tutor traded some pedagogical faithfulness for
outcome performance. The gate is no longer "does truncation explain the gap"
— that's answered, partially — it's "does the tutor learn something that
survives contact with an unseen problem *without* leaking more to get there,"
and the next lever (per the run's own transcripts) is role inversion: the
simulated 0.5B student still writes cleaner, more complete solutions than the
1.5B tutor gives hints.

```
Phase 1  Foundation & infrastructure    ████████████████████  done
Phase 2  Make training actually learn   ███████░░░░░░░░░░░░░  in progress  ← critical path
Phase 3  Indic / code-mixed             ███░░░░░░░░░░░░░░░░░  scaffolding only
Phase 4  Knowledge tracing & validation ███░░░░░░░░░░░░░░░░░  data collection started
```

---

## Phase 1 — Foundation & infrastructure ✅ **done**

Everything needed to run, serve, observe, and debug the system.

| Item | State |
|---|---|
| GRPO training loop, checkpointing, rollout dumps | done |
| `r_sol` — execution-verified solve rate (sandboxed subprocess) | done |
| `r_ped` — 5 graded rule checks | done |
| `L` — execution-verified leakage | done |
| BKT tracer + ZPD curriculum sampling | done |
| Six services in Docker: gateway, session, tutor, executor, tracer, asr | done |
| Executor containment: no egress, read-only rootfs, rlimits, 13 tests | done |
| Persistent per-learner mastery in Postgres | done |
| Request tracing across services, aggregate health | done |
| `inspect_rollouts.py` verdict tool, `smoke_test.py` end-to-end | done |
| Student assessment interface — pick a problem, chat, submit code, see mastery, served off the gateway | done |
| 108 tests across core, services and the research package | done |

**Gap left deliberately:** `services/gateway`, `tutor` and `asr` have no tests.
Gateway handles auth and rate limiting untested.

---

## Phase 2 — Make training actually learn 🔴 **critical path**

Nothing downstream is measurable until this produces a trained tutor.

### Done

| Item | Evidence |
|---|---|
| Dangling-promise trim | countermeasures shipped; 6.8% of turns still end in `:` in the scaled run (down from 19%, not eliminated) |
| Scripted student opener | confirmed working in the live run |
| Graded `r_ped` | reward spread `0.000 → 1.317`; loss `0.0000 → 0.0562` |
| Empty-dialogue guard | silence scored 0.8, now 0.0 |
| Execution-verified leakage | a full solution leak scored 0.000, now 1.000 |
| Sentinel-based result parsing | candidate `print()` statements were silently breaking `r_sol` scoring in every prior run; fixed |
| Scaled run executed (v6) | `G=8`, 20 epochs, 320 dialogues — reward spread non-zero every epoch, so a real gradient existed throughout |
| **Truncation fix, re-run, and validated** | Raised `tutor_max_new_tokens` (192→256) and `student_max_new_tokens` (160→224), added defensive `bitsandbytes`/`datasets` install checks (Kaggle's image had silently stopped preinstalling `bitsandbytes`, which failed the first re-run attempt). Re-ran the identical 20-epoch config. Held-out solve rate on the same 20-problem set: **6.3% → 11.25%**, nearly double. This is the first real before/after comparison in the project, not a single-run reading. |

### Not done

| Item | Why it matters |
|---|---|
| **Generalization is still low in absolute terms** | 11.25% held-out solve is real progress but still far from a usable tutor. Best training-epoch solve rate this run was 34.4% (down from v6's 45%, but that's an extreme-value statistic over 32 solve-samples/epoch — noisy by construction, not a reliable regression signal on its own). |
| **Pedagogy/leakage regressed on held-out** | Teaching acceptance 80%→71%, leakage 9.9%→11.25%. The tutor got somewhat more outcome-focused and somewhat less faithful to "guide, don't tell." Worth watching on the next run, not yet a confirmed trend (n=1 comparison). |
| **Role inversion** | Still directly visible in this run's own smoke-test transcript: the simulated 0.5B student volunteers a complete, well-explained Python solution unprompted, while the 1.5B tutor's hints are sometimes broken/garbled (mixed-language). Countermeasures shipped earlier did not resolve this. This is now the best-evidenced next lever. |
| **Retention benchmark** | LoRA + KL *should* prevent forgetting. Nothing measures whether it did. This turns an assertion into a number. |
| ACE causal probes | The review's central defence against rewarding non-causal solves. `ace_leakage()` exists but receives no data. |
| Importance ratio + clipping | `clip_epsilon` is defined but unused — this is REINFORCE with a KL penalty, not clipped GRPO. Do not call it GRPO in a writeup until fixed. |
| k3 KL estimator | `mean(log π − log π_ref)` can go negative; it is not a KL. |
| `K ≥ 8` solve samples | Currently 4. Deliberate: `solve ≈ 0`, so more samples measure noise more precisely. Raise once the student can actually solve. |

**Likely blocker:** role inversion may not be fixable by prompt or
post-processing. Truncation *was* fixed and *did* move held-out solve rate
(6.3%→11.25%), confirming it was masking real signal — but role inversion is
still visible in the fixed run's own transcripts, unchanged. The 0.5B student
breaks persona; moving it to 1.5B on the second T4 is the obvious next lever
and is mostly free.

---

## Phase 3 — Indic / code-mixed 🟡 **scaffolding only**

**Scope correction:** this phase does *not* involve training or benchmarking
an ASR model. The tutor (Qwen) is the only thing trained, via RL, in Phase 2.
Voice is added on top afterwards by wrapping the trained tutor with
off-the-shelf, pretrained Indic ASR (speech→text in) and TTS (text→speech
out) — no fine-tuning, no WER/CER benchmark run. An Indic-native LLM swap for
the tutor itself is the fallback if Qwen's Hinglish/Tanglish handling proves
too weak, evaluated the same way as any other base-model choice, not as a
separate ASR workstream.

### Done

- `CODE_MIXING_TEMPLATES` for Hinglish / Tanglish / English, used at generation
  time for the student's opening turn — confirmed live (`"Kya hum strings logic
  use kar sakte hain yahan pe?"`)
- `services/asr` boundary, contract, and health check (now scoped as a thin
  wrapper around a pretrained model, not a component to be trained)

### Not done

| Item | Note |
|---|---|
| **Indic-capable base model** | Still Qwen2.5-**1.5B**, weak on romanized Indic. This is a base-model decision, not a prompt one. Options: larger Qwen, an Indic-specialised base, or continued pretraining. |
| Wire a pretrained Indic ASR + TTS model into `services/asr` | Currently returns `501`. No training needed — this is an integration task, gated on the text tutor being solid first. |
| Code-mixed evaluation set | No way to measure whether Indic tutoring performance is good or bad. Applies to the tutor's dialogue, not to any ASR/WER benchmark. |
| Indic-aware pedagogy checks | The 5 checks are English-shaped; needs verifying across scripts. |

**Note:** continued pretraining (if the Indic-LLM-swap fallback is taken) is
where catastrophic forgetting risk is genuinely high — unlike the LoRA stage,
which freezes the base weights.

---

## Phase 4 — Knowledge tracing & human validation 🟡 **data collection started**

### Done

- **Append-only observation log** — `(learner, skill, correct, problem_id,
  timestamp, mastery_after)`, written on every submit
- Pageable replay endpoint (`after_seq`) for a training job
- Test proving mastery is reproducible by replaying the log, so `SkillMastery`
  is a cache and the log is the record

This was built early on purpose: DKT trains on the *sequence*, and history
cannot be collected retroactively.

### Not done

| Item | Note |
|---|---|
| DKT model | Needs interaction sequences at a scale a synthetic student won't produce. |
| Swap DKT behind the tracer API | Cheap once the model exists — `/mastery`, `/observe`, `/zpd` already isolate it. |
| Skill taxonomy mapping | Public datasets (ASSISTments, EdNet) won't share MBPP's skill vocabulary. Worth deciding before much data accumulates. |
| Sim→real pilot (H1–H4, N=34) | Not begun. |
| IRB, consent, pre-registration | Not begun. |

---

## Critical path

```
Phase 2 scaled run  ──►  everything else
        │
        ├──► retention benchmark   (needs a trained model to compare against)
        ├──► Phase 3 base swap     (pointless before the loop is known to work)
        └──► Phase 4 DKT           (needs real learners, needs a usable tutor)
```

The order is forced. A base-model swap before the training loop is known to work
would just change which unknowns are confounded. The human study needs a tutor
worth putting in front of a person.

## Next three actions

1. **Student → 1.5B on the second GPU.** Truncation is fixed and validated
   (held-out solve 6.3%→11.25%); role inversion is now the best-evidenced
   remaining lever, confirmed again in this run's own transcripts.
2. **Investigate the pedagogy/leakage regression** (80%→71% ped, 9.9%→11.25%
   leak on held-out) before assuming it's noise — re-run once to see if it
   replicates, since it's currently an n=1 comparison.
3. **Retention benchmark** — MBPP pass@1 with adapters on vs off, before vs
   after. Cheap, and it is the answer to "did it forget?"
