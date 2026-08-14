# SAHAi — Roadmap

Four phases. Phase 1 is done; the rest are gated on each other in order.

**Headline:** the *infrastructure* is essentially finished. The *science* has
started but isn't there yet. A naive checkbox count says 62% complete; weighted
by effort and risk it is closer to **35–40%** — the scaled run (20 epochs, 320
dialogues, kernel v6) landed and shows a real, non-zero learning signal every
epoch, but it doesn't yet generalize: held-out solve rate (6.3%) is far below
what training epochs peak at (45%), and 37% of all turns are cut off
mid-generation, which is corrupting a third of the reward signal. The gate is
no longer "does a scaled run exist" — it's "does the tutor learn something
that survives contact with an unseen problem."

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
| Scaled run executed | `G=8`, 20 epochs, 320 dialogues, kernel v6 — reward spread non-zero every epoch, so a real gradient existed throughout |

### Not done

| Item | Why it matters |
|---|---|
| **Truncation** | 37% of all turns (765/2,059) in the scaled run are cut off mid-generation — worse than earlier estimates, and the single biggest source of noise in the reward signal. Fix: raise `max_new_tokens`, then re-run. |
| **Generalization** | Best training-epoch solve rate hit 45%, but held-out eval solve rate is 6.3% on 20 unseen problems. The scaled run proves the loop can move reward around; it does not yet prove the tutor learned a transferable skill. |
| **Role inversion** | Student writes code in 233 turns vs. the tutor's 218 in the scaled run — roughly matched, sometimes exceeding it. Countermeasures shipped but did not resolve this at scale. |
| **Retention benchmark** | LoRA + KL *should* prevent forgetting. Nothing measures whether it did. This turns an assertion into a number. |
| ACE causal probes | The review's central defence against rewarding non-causal solves. `ace_leakage()` exists but receives no data. |
| Importance ratio + clipping | `clip_epsilon` is defined but unused — this is REINFORCE with a KL penalty, not clipped GRPO. Do not call it GRPO in a writeup until fixed. |
| k3 KL estimator | `mean(log π − log π_ref)` can go negative; it is not a KL. |
| `K ≥ 8` solve samples | Currently 4. Deliberate: `solve ≈ 0`, so more samples measure noise more precisely. Raise once the student can actually solve. |

**Likely blocker:** role inversion may not be fixable by prompt or
post-processing. The 0.5B student breaks persona; moving it to 1.5B on the
second T4 is the obvious next lever and is mostly free. Truncation, however,
is probably the higher-leverage fix — a third of the signal is currently noise
from cut-off generations, and that alone could be masking whether the
role-inversion countermeasures are working at all.

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

1. **Fix truncation, then re-run.** Blocked on the weekly Kaggle GPU quota
   (30 hrs) resetting. Raise `max_new_tokens` — 37% of turns in the last run
   were cut off mid-generation, which is the most likely single explanation
   for the gap between training-epoch solve rate (45% peak) and held-out
   solve rate (6.3%).
2. **Student → 1.5B on the second GPU** if role inversion persists once
   truncation is no longer confounding the measurement.
3. **Retention benchmark** — MBPP pass@1 with adapters on vs off, before vs
   after. Cheap, and it is the answer to "did it forget?"
