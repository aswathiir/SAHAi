# SAHAi — Architecture

Both halves of the system: the offline training loop that produces a tutor
policy, and the online services that serve it.

---

## Part 1 — Training architecture (Program A)

```
                    ┌──────────────── ProblemBank (MBPP) ────────────────┐
                    │   zpd_sample(tracer, n) — problems one step       │
                    │   beyond this student's current ability           │
                    └───────────────────────┬────────────────────────────┘
                                            │  n problems
                                            ▼
   ┌─────────────────────── DialogueEngine.run() × G rollouts ──────────────────┐
   │                                                                            │
   │   student.opening_turn()   scripted confusion, in-persona                  │
   │            │                                                               │
   │            ▼                                                               │
   │   ┌──► TutorPolicy.generate()      ← the policy under training (LoRA)      │
   │   │            │                                                           │
   │   │            ▼                                                           │
   │   └─── StudentSimulator.respond()  ← frozen simulator, post-processed      │
   │                                                                            │
   │   stops on a termination phrase or 2 × max_turns                           │
   └───────────────────────────────┬────────────────────────────────────────────┘
                                   │  Dialogue
                                   ▼
   ┌──────────────────────────── Reward ────────────────────────────────────────┐
   │  r_sol   SolveReward     student attempts the problem k times after the    │
   │                          dialogue; fraction passing all unit tests         │
   │  r_ped   PedagogyReward  5 rule checks, graded (fraction passed)           │
   │  L       LeakageEstimator  token overlap between tutor text and solution   │
   │                                                                            │
   │  r_SAHAI = (r_sol − ability) + (r_ped − 1)·λ − γ·L                          │
   │  hard penalty: r_ped == 0  ⇒  r_SAHAI = −λ                                 │
   └───────────────────────────────┬────────────────────────────────────────────┘
                                   ▼
   ┌──────────────────────── GRPOTrainer._policy_update ────────────────────────┐
   │  advantage = (r − mean_group) / (std_group + ε)      per group of G        │
   │  loss      = −A · mean(log π over tutor tokens) + β · KL(π ‖ π_ref)        │
   │  π_ref     = same model with LoRA adapters disabled                        │
   └────────────────────────────────────────────────────────────────────────────┘
```

### Components

| Module | Role |
|---|---|
| `agents/tutor.py` | `TutorPolicy` — the **only** trained component. Qwen2.5-1.5B + LoRA. |
| `agents/student.py` | `StudentSimulator` — frozen Qwen2.5-**1.5B** (4-bit) with a persona: ability level, misconceptions per skill, code-mixing language. Raised from 0.5B, which could not hold the "confused student" persona under RL pressure and volunteered complete solutions unprompted. |
| `agents/tracer.py` | `BKTTracer` — Bayesian Knowledge Tracing per skill. Supplies the `ability` baseline and drives ZPD problem selection. |
| `core/dialogue.py` | `DialogueEngine` — alternating turns, termination phrases, per-turn completeness. |
| `reward/solve.py` | Executes candidate code in a subprocess sandbox against unit tests. |
| `reward/pedagogy.py` | `RuleBasedJudge` (**3** graded checks, all asking "did the tutor give the answer away") or an LLM judge with ACCEPT/REJECT rubrics. Two checks were removed after one of them was gamed inside a single run. |
| `reward/leakage.py` | Token-overlap leakage; optional ACE (assistance-corrected effect) term. |
| `training/grpo.py` | Rollout → reward → advantage → policy update, plus checkpointing and rollout dumps. |
| `core/asr.py` | Code-mixed speech front-end (Whisper), with keyword-preserving noise injection. |
| `eval/benchmark.py` | Held-out evaluation producing `MetricsReport`. |

### Design decisions worth knowing

**Only tutor tokens enter the loss.** `TutorPolicy._build_tutor_mask` marks the
assistant spans; log-probs are masked to those. This is why the student can be
freely post-processed while the tutor cannot — editing tutor text would compute
gradients over tokens the model never sampled.

**Rewards are group-relative.** There is no value network. The advantage is a
z-score within each group of `G` rollouts on the same problem, so **within-group
reward variance is the entire learning signal**. A group where every rollout
scores identically contributes exactly zero gradient.

**The pedagogy score is graded, not binary.** All-or-nothing scoring collapsed
every group to a single value and zeroed the gradient — see §3.

**ZPD sampling.** `zpd_sample` draws problems whose difficulty sits within one
level of `1 + ability·(max−1)`, with skill mastery acting only as a ceiling so
that demonstrated work is not re-offered.

This used to be a pure mastery band, `zpd_low <= mean(mastery) <= zpd_high`,
which ignored the `difficulty` argument it was passed. Since `p_init` and
`zpd_low` were both 0.3, that band meant "skills never attempted" — and BKT
drives a failed skill to ~0.109 with no forgetting transition, so once every
skill had been tried it was empty permanently. Seven of ten epochs in the
2026-09-12 run drew from the top-up path rather than from a curriculum.

---

# Part 2 — Service architecture (Program B)

How the research prototype was decomposed into a deployable system, and why each
boundary sits where it does.

---

## 1. The boundary that drove everything

The research code conflated two things that look alike and behave completely
differently:

|  | training | production |
|---|---|---|
| the student | a 1.5B model the engine calls in a loop | a real person sending one turn and waiting |
| control flow | batch: run the whole dialogue, then score | event-driven: hold state across requests |
| lifetime | one process, discarded | must survive restarts |
| mastery state | synthetic, thrown away | belongs to a person, persists forever |

`DialogueEngine.run(tutor, student, problem)` only makes sense on the left. So
the engine became two things — the batch runner still in `sahai/training/grpo.py`,
and a state machine over a persisted transcript in `services/session`. Joining
them onto `sahai-core` so they share one reward implementation is the next
structural step.

## 2. Topology

```
  host
   │  :8080
   ▼
┌──────────┐   public network
│ gateway  │   auth stub · rate limit · request id · aggregate health
└────┬─────┘
     │            internal network
     ▼
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ session  │──▶│  tutor   │   │  tracer  │   │   asr    │
│  state   │   │ vLLM/HF  │   │ BKT+ZPD  │   │ (stub)   │
│  machine │   │  /stub   │   └────┬─────┘   └──────────┘
└────┬─────┘   └──────────┘        │
     │                             ▼
     │                       ┌──────────┐
     │                       │ postgres │
     │                       └──────────┘
     │  sandbox network (internal: true — no egress)
     ▼
┌──────────┐
│ executor │  read-only rootfs · cap_drop ALL · pids 128 · 1g · no route out
└──────────┘
```

`session` is the only service on both networks: it must reach the executor,
which must not reach anything.

## 3. Services

| Service | Owns | Stateless? | Notes |
|---|---|---|---|
| **gateway** | ingress | yes | The only host-published port. Auth is a stub that establishes the header contract. |
| **session** | transcripts, session state | no (Postgres) | Production counterpart to `DialogueEngine`. |
| **tutor** | policy inference | yes | Three backends: `stub` (CPU, no weights), `hf` (+LoRA), `vllm` (not built). |
| **executor** | running untrusted code | yes | No DB, no egress, no user concepts. |
| **tracer** | learner mastery | no (Postgres) | BKT per (learner, skill); also does ZPD selection. |
| **asr** | speech → text | yes | Stubbed; boundary and contract are real. |
| **sahai/** (Program A) | GRPO training | batch | GPU, runs on Kaggle. Produces LoRA adapters. |
| **libs/sahai-core** | domain + scoring | pure | No I/O, no GPU, no execution. |

## 4. Why `sahai-core` has no heavy dependencies

A dependency analysis of the research package found only four modules needed a
GPU; thirteen were already pure. That seam became the library.

Two properties are enforced by test, not convention:

- `test_core_imports_no_heavy_dependencies` — importing core must not pull in
  torch, transformers, peft, fastapi, sqlalchemy, or datasets.
- `test_core_cannot_execute_code` — core exposes no code runner at all.

The second is the important one. In the research code, `CodeVerifier` lived
beside the reward logic and ran `subprocess.run([sys.executable, "-c", model_output])`.
Anything importing the reward stack inherited the ability to execute arbitrary
model output. Splitting *scoring* (`SolveScorer`, in core) from *execution*
(`services/executor`) is what removes that.

## 5. Executor containment

Layered, because any single layer can fail:

| Layer | Control |
|---|---|
| network | `sandbox` network with `internal: true` — no egress at all |
| filesystem | `read_only: true`; only a 64 MB tmpfs at `/tmp` is writable |
| privileges | non-root uid 10001, `cap_drop: ALL`, `no-new-privileges` |
| resources | `pids_limit: 128`, `mem_limit: 1g`, `cpus: 1.0` |
| process | fresh interpreter per test case, `-I -S`, own session, scrubbed env |
| rlimits | AS, DATA, CPU, NOFILE, FSIZE, NPROC applied pre-exec |
| protocol | result read from a sentinel line, so candidate stdout cannot forge a pass |

Thirteen tests cover this, including an infinite loop, a memory bomb, a fork
bomb, and a candidate that prints the sentinel to fake its own success.

Two implementation notes that cost debugging time and are easy to repeat:

- `RLIMIT_CPU` delivers **SIGXCPU**, not SIGKILL. Checking only for SIGKILL
  misreports an exhausted run as a generic error.
- `preexec_fn` must not call `os.setsid()` when `start_new_session=True` is
  already set — the second call fails with EPERM and surfaces as an opaque
  `SubprocessError`.

## 6. Running it

```bash
make up        # build and start the stack
make smoke     # end-to-end request through the gateway only
make test      # 78 tests across core, services, and the research package
make down
```

The stack runs end-to-end on a laptop with **no GPU**: `SAHAI_TUTOR_BACKEND=stub`
serves deterministic Socratic questions that pass the same pedagogy checks a real
model faces. To serve a trained policy:

```bash
SAHAI_TUTOR_BACKEND=hf \
SAHAI_ADAPTER_PATH=/srv/artifacts/final_model \
SAHAI_DEVICE=cuda docker compose up -d tutor
```

## 7. Deliberately not built yet

| Item | Why it is missing |
|---|---|
| Real auth | Gateway trusts `x-learner-id`. The seam exists; the identity provider does not. |
| vLLM backend | `hf` is enough to serve one adapter; throughput is not yet a constraint. |
| Whisper ASR | Text-first this phase. Contract and health check are real, transcription is not. |
| Alembic migrations | Tables are created at startup. Fine for a single deployment, wrong for schema evolution. |
| Distributed rate limiting | The limiter is in-process; a multi-instance gateway needs Redis. |
| Problem-bank service | The bank is still loaded in-process by the trainer. |
| OpenTelemetry | Request ids and timing headers only. |

## 8. What did not change

`sahai/` and `kaggle_upload/` are untouched, so the Kaggle training path keeps
working exactly as documented in `docs/07-operations-kaggle.md`. Migrating it onto `sahai-core` is
the next step, and is deliberately separate from standing up the services.
