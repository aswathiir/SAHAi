# SAHAi

An RL-trained tutor for placement DSA prep. The tutor is optimised with GRPO to
guide a student toward solving a problem **without ever revealing the solution**,
in code-mixed language (Hinglish / Tanglish / English).

The training signal is the student's downstream success, not imitation of tutor
transcripts. Nothing supervises what a good hint looks like — the tutor is
rewarded when the student solves the problem afterwards, and penalised when it
leaks the answer or stops behaving like a tutor.

```
r_SAHAI = (r_sol − ability) + (r_ped − 1)·λ − γ·L
```

**Full documentation: [docs/](docs/README.md)** — research basis, architecture,
internals, findings, roadmap, operations, team split.

This file is the orientation guide: folder map, request flow, and a
symptom→location debugging table.

---

You are looking for: what is in this repo, what talks to what, and **where to
look when something breaks**. This file is the map. Read it once, then keep the
debugging table in §5 open.

---

## 1. The single most important thing

**There are two separate programs in this repo.** They do not run at the same
time, they do not run in the same place, and confusing them is the main reason
things feel chaotic.

```
  PROGRAM A — TRAINING                  PROGRAM B — SERVING
  "make the tutor better"               "let a person talk to the tutor"

  runs on:  Kaggle, one GPU             runs on:  your laptop, Docker
  runs for: ~30 min, then stops         runs for: as long as you leave it up
  student:  a 0.5B AI pretending        student:  a real human
  output:   a trained adapter file      output:   HTTP responses
  code in:  sahai/  notebooks/          code in:  services/
  start it: kaggle kernels push         start it: make up
```

They currently share **nothing but ideas**. `libs/sahai-core` is the shared
library being built to join them; right now Program A still uses its own copy in
`sahai/`. That is deliberate — it keeps your working Kaggle setup from breaking
while the services get built.

If you remember one thing: **`sahai/` is training. `services/` is serving.**

---

## 2. Folder map

```
SAHAi/
│
├── sahai/                    ← PROGRAM A. Training. Runs on Kaggle.
│   ├── agents/
│   │   ├── tutor.py            the AI being trained
│   │   ├── student.py          the fake student it practises against
│   │   └── tracer.py           tracks which skills the fake student knows
│   ├── core/
│   │   ├── dialogue.py         runs a full tutor↔student conversation
│   │   ├── data.py             Problem / TestCase / ProblemBank
│   │   ├── dataset.py          loads MBPP problems
│   │   ├── generation.py       text cleanup (truncation, dangling colons)
│   │   └── models.py           loads Qwen + attaches LoRA
│   ├── reward/
│   │   ├── solve.py            runs student code, checks tests   ← r_sol
│   │   ├── pedagogy.py         5 checks on teaching quality      ← r_ped
│   │   ├── leakage.py          did the tutor give away the answer? ← L
│   │   └── combined.py         puts them into one score          ← r_SAHAI
│   ├── training/grpo.py        THE TRAINING LOOP. Start here for training bugs.
│   └── settings.py             all knobs (model names, epochs, G, lr…)
│
├── notebooks/train_kaggle.py   the notebook Kaggle actually runs
│                               (.ipynb is generated from this — keep in sync)
│
├── services/                 ← PROGRAM B. Serving. Runs in Docker.
│   ├── gateway/                front door. the ONLY thing on a public port
│   ├── session/                holds a conversation across HTTP requests
│   ├── tutor/                  generates one tutor reply
│   ├── executor/               runs untrusted code safely
│   ├── tracer/                 remembers what a REAL learner knows
│   └── asr/                    speech→text (stubbed, not real yet)
│
├── libs/sahai-core/          ← shared logic, no AI, no database, no network
│
├── scripts/
│   ├── inspect_rollouts.py     reads training output, gives a verdict
│   └── smoke_test.py           checks the running services end-to-end
│
├── tests/                      tests for Program A
├── docker-compose.yml          how the services wire together
└── Makefile                    shortcuts: make test / make up / make smoke
```

---

## 3. Program A — how training works

One epoch does four things, in `sahai/training/grpo.py`:

```
1. ROLLOUT      pick 2 problems, run 4 conversations each  = 8 conversations
                  → DialogueEngine.run() in core/dialogue.py
2. REWARD       score each conversation
                  → reward/solve.py    did the student then solve it?  (runs code)
                  → reward/pedagogy.py was it good teaching?           (5 checks)
                  → reward/leakage.py  did it leak the answer?
                  → reward/combined.py combine into one number
3. ADVANTAGE    compare the 8 scores against each other
                  ⚠ if all 8 are equal, learning is ZERO this epoch
4. UPDATE       nudge the tutor toward the above-average conversations
```

Step 3 is the one that bit us. There is no "expected score" predictor — the
**differences between the 8 conversations are the entire signal**. Identical
scores means a wasted epoch, and the log prints `loss=0.0000`.

Every epoch writes `output/rollouts/epoch-N.json` containing every conversation
and its scores. That file is your primary debugging tool for training.

## 4. Program B — how a request flows

One student message, traced through the services:

```
  student's browser
       │  POST /v1/sessions/{id}/turns   {"content": "I'm stuck"}
       │  header: x-learner-id
       ▼
  ┌─ gateway ─────────┐  services/gateway/app/main.py
  │ check header      │  rejects with 401 if x-learner-id missing
  │ rate limit        │  429 if >60/min
  └────────┬──────────┘
           │  forwards to session
           ▼
  ┌─ session ─────────┐  services/session/app/main.py  ← the orchestrator
  │ 1 save student    │  writes a row to Postgres
  │   turn to DB      │
  │ 2 build message   │  system prompt + every previous turn
  │   history         │
  │ 3 ask tutor  ─────┼──────────────► ┌─ tutor ────────┐
  │                   │                │ generate reply │ services/tutor/app/
  │ 4 save tutor turn │ ◄──────────────┤ (stub or Qwen) │   backends.py
  └────────┬──────────┘                └────────────────┘
           │  returns whole transcript
           ▼
  student sees the tutor's reply
```

Later, when the student submits code:

```
  session ──► executor  (runs the code against tests, in a locked-down box)
          ──► tracer    (records pass/fail, updates that learner's mastery)
```

**Key idea:** `session` is the only service that knows a conversation exists.
`tutor` gets a list of messages and returns text — it has no memory.
`executor` gets code and returns pass/fail — it knows nothing about users.
That's on purpose: small services with one job each are far easier to debug.

---

## 5. Debugging map — the part to keep open

### First move, always

```bash
make up                              # start everything
curl localhost:8080/health | jq      # which service is unhappy?
```

That returns every dependency's status. Whatever says something other than
`"ok"` is where to look. Then:

```bash
docker compose logs -f session       # follow one service's logs
docker compose ps                    # what's actually running
```

### Symptom → where to look

| What you see | What it means | Where to look |
|---|---|---|
| `make up` fails to build | Dockerfile or dependency problem | the build output; `services/<name>/Dockerfile` |
| `/health` says a service is `unreachable` | container crashed or never started | `docker compose logs <name>` — the traceback is at the bottom |
| `401 x-learner-id header required` | you forgot the header | add `-H "x-learner-id: me"` |
| `429 rate limit exceeded` | >60 requests/min | wait, or raise `SAHAI_RATE_LIMIT_PER_MIN` |
| `404 session not found` | wrong session id, or DB was wiped | `docker compose down -v` wipes Postgres |
| `409 session is solved` | you already submitted | start a new session |
| Tutor replies are generic questions | you're on the **stub** backend, not a real model | `curl localhost:8080/health` shows the backend; set `SAHAI_TUTOR_BACKEND=hf` |
| Code submission always fails | executor problem, or your test cases are wrong | `docker compose logs executor`; check `expected` values |
| `MissingGreenlet` | SQLAlchemy lazy-load in async code | add `selectinload(...)` to the query |
| Transcript looks one message behind | identity-map staleness | needs `populate_existing=True` — we hit this once |
| Postgres `connection refused` | DB not ready yet | `depends_on: service_healthy` should handle it; check `docker compose ps` |

### Training-side symptoms

| What you see | What it means | Where to look |
|---|---|---|
| `loss=0.0000` | all 8 conversations scored identically → no learning | `output/rollouts/epoch-N.json`, check the `reward` field |
| `N/M groups had zero reward spread` | same thing, now warned explicitly | raise `group_size` in `settings.py` |
| `solve=0.0312` | exactly 1 of 32 attempts passed — noise, not a result | expected at this scale |
| Kaggle run dies in ~1 min | setup problem, never reached the GPU | `kaggle kernels output ... && cat *.log` |
| Roles inverted in transcripts | the known open problem | `scripts/inspect_rollouts.py` quantifies it |

### Running one piece in isolation

This is the most useful debugging skill here — you rarely need the whole stack.

```bash
# just the sandbox logic, no Docker, no HTTP
cd services/executor && PYTHONPATH=".:../../libs/sahai-core" python -m pytest tests -q

# just the reward maths
cd libs/sahai-core && PYTHONPATH=. python -m pytest tests -q

# the whole serving loop in one process (no Docker at all)
cd services/session && PYTHONPATH=".:../../libs/sahai-core" python -m pytest tests -q

# everything
make test
```

Each service also has interactive API docs when running:
`http://localhost:8080/docs` — click through and send real requests.

---

## 6. Where errors actually surface

Knowing *where a traceback appears* saves the most time:

- **A service crashed on startup** → `docker compose logs <name>`, bottom of output.
- **A request failed** → the response body has a `detail` field, and the
  gateway adds an `x-request-id` header you can grep the logs for.
- **A service is up but wrong** → its own logs; each runs `uvicorn`, which prints
  every request line.
- **Training failed** → Kaggle's log, fetched with `kaggle kernels output`.
- **A test failed** → pytest prints the exact file and line.

Errors do **not** propagate between services as tracebacks. If `session` calls
`executor` and the executor 500s, you see a generic error in session's logs and
the *real* traceback in the executor's. Always check the service that owns the
work, not the one that called it.

---

## 7. Suggested path from here

You said you want to pave the path — here is what I'd sequence, easiest first:

1. **`make up && make smoke`.** Watch it work end to end. Then break something
   on purpose (stop the executor: `docker compose stop executor`) and re-run
   smoke, so you see what a failure looks like.
2. **Open `http://localhost:8080/docs`** and drive a conversation by hand.
3. **Read `services/session/app/main.py` top to bottom.** ~250 lines, and it is
   the spine of the serving side.
4. **Then** we migrate `sahai/` (training) onto `sahai-core`, so training and serving
   share one reward implementation instead of two copies.

Step 4 is the real remaining structural work. Everything before it is you
getting oriented, which is worth doing first.
