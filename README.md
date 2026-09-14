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
  runs for: ~8 h, then stops            runs for: as long as you leave it up
  student:  a 1.5B AI pretending        student:  a real human
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
│   │   ├── pedagogy.py         3 checks on answer-giving         ← r_ped
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
│     prompt.py              the tutor's serving prompt — ONE definition.
│                            There were two, and the live one was missing
│                            the code-mixing rule.
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
                  → reward/pedagogy.py did it give the answer away?    (3 checks)
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
       │  header: Authorization: Bearer <token>
       ▼
  ┌─ gateway ─────────┐  services/gateway/app/main.py
  │ verify the token  │  401 unless it hashes to a known learner
  │ rate limit        │  429 if >60/min, keyed on the *verified* learner
  └────────┬──────────┘
           │  forwards to session
           ▼
  ┌─ session ─────────┐  services/session/app/main.py  ← the orchestrator
  │ 1 reject if the   │  409 — a previous request died mid-exchange
  │   last turn is    │
  │   unanswered      │
  │ 2 build message   │  system prompt + every previous turn + this one
  │   history         │  (nothing written to the DB yet)
  │ 3 ask tutor  ─────┼──────────────► ┌─ tutor ────────┐
  │                   │                │ generate reply │ services/tutor/app/
  │ 4 save BOTH turns │ ◄──────────────┤ (stub or Qwen) │   backends.py
  │   in one commit   │                └────────────────┘
  └────────┬──────────┘
           │  returns whole transcript
           ▼
  student sees the tutor's reply
```

**A turn is atomic on purpose.** The student's turn used to be committed before
the tutor was called; when that call timed out the dialogue was left holding a
student turn with no reply, and retrying appended a *second* student turn.
Consecutive same-role turns break the alternation that prompt assembly, the
pedagogy judge and the chat template all assume — every later turn scored wrong
with no error anywhere. Either both turns land or neither does.

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

## 4b. Who you are

Every page used to carry a free-text `learner` box whose value went out as
`x-learner-id`, and the gateway returned it unchanged. Typing someone else's id
read their sessions, mastery and submissions. The ownership checks in the
session service were correct; the identity underneath them was not.

```
  POST /v1/auth/register {"display_name": "Ada"}
        │
        ▼   201  { learner_id: "lnr_8a04…", token: "…" }   ← shown once
  ┌─ gateway ──────────────────────────────────────────┐
  │  stores sha256(token) only — no endpoint returns   │
  │  the token again, deliberately                     │
  └────────────────────┬───────────────────────────────┘
                       │  Authorization: Bearer <token>
                       ▼  resolved to a learner id, then forwarded inward
                          as x-learner-id on the internal network
```

**Bearer tokens, not passwords.** The project stores no passwords and never
sees one. A fast hash is right *here* specifically: bcrypt and argon2 exist to
make offline brute force expensive against low-entropy human secrets, and a
32-byte `secrets.token_urlsafe` has no brute-force surface to protect.

**The websocket authenticates on its first frame**, not in the query string.
`learner_id` used to be a query parameter defaulting to `"me"`, so
`ws://host/v1/voice?session_id=…&learner_id=<someone>` was enough to speak into
another learner's session. Browsers cannot set headers on a WebSocket
handshake, and a token in a URL lands in access logs and browser history — so
the client sends `{"token": …}` and waits for `{"stage": "authenticated"}`.

**What it does not promise.** It authenticates a bearer, not a person: anyone
holding the token is the learner, with no second factor, no recovery, and no
revocation beyond deleting the row. Session and tracer still trust the
`x-learner-id` the gateway fills in — they publish no ports, so that is a
network assumption, not a cryptographic one, and it is why the compose file's
network split matters.

Learners that predate this (`me`, `smoke-learner`, an imported NeetCode
profile) have no credential and are otherwise unreachable. Attach one without
disturbing their history:

```bash
docker compose exec gateway python /srv/scripts/mint_token.py --list
```

---

## 5. The learner-state layer

Everything above describes one conversation. This is what makes the *next* one
different from the last.

```
  who you are  ──►  what you are offered  ──►  how the hint is worded
      │                     │                            │
  placement or        zpd_sample picks             system prompt gains
  a solved-repo       problems whose skill         "New to X / Solid on Y"
  import seeds        difficulty sits one step
                      beyond current ability
  BKT priors                │                            │
      │                     ▼                            ▼
      └──────────►  you solve or fail  ──►  BKT posterior moves  ──┘
                             │
                             ▼
                    /progress.html — streak, activity grid,
                    per-skill mastery, course tracks
```

**Placement** (`POST /v1/me/placement`). Without it every learner starts at one
flat `p_init` (0.3) for every skill and the first session is spent discovering
what they already knew. A self-assessment across the 15 skills seeds per-skill
priors instead.

Confidence maps into `[0.15, 0.65]`, not `[0, 1]`. A self-report is evidence,
not proof, and a prior at 0.9 would take several failures to walk back. The top
sits just inside the ZPD band so a confident learner gets work at the edge of
what they claim rather than being skipped past the skill.

Two rules it will not break, both pinned by tests, because both would corrupt
data rather than merely look wrong:

- it only seeds skills with **no observations**, so re-taking the survey can
  never overwrite demonstrated mastery;
- it writes **nothing** to `observations` — a prior is not an attempt, and
  logging it as one would corrupt the sequence a knowledge-tracing model
  trains on later.

**Importing a solved-problem repo** (`scripts/import_neetcode.py`). Stronger
evidence than a self-report, and no credentials needed — public repo, documented
API, stable layout.

```bash
python scripts/import_neetcode.py --repo owner/name --learner me --dry-run
```

Paths and dates only. The solution *source* is deliberately not read: it is a
far richer signal, but it is also the answer to a problem, and this project's
whole reward function exists to stop answers reaching the learner. Three things
the first real repo forced, each of which would otherwise have produced
plausible-looking nonsense:

| what | why it matters |
|---|---|
| bulk commits are discounted | a "Bulk sync: 266 submissions" commit dates 186 of 319 problems to the day the repo was created, not when they were solved |
| NeetCode names get their own tag table | "house-robber" says nothing about dynamic programming; the generic tagger fell back to `general` for 121 of 319 slugs |
| thresholds are absolute, not relative | a relative scale calls somebody's best skill "confident" on four solved problems |

The tag table lives in the importer, **not** in `sahai.core.dataset.SKILL_KEYWORDS`
— that table also tags the MBPP training bank, and re-tagging training data
mid-experiment would change what the sampler offers for unrelated reasons.

**Reading the turn, not just the learner** (`sahai_core.turn_signals`). Mastery
says who the learner is; it says nothing about what they just wrote. Before this
the tutor treated `Mujhe samajh nahi aa raha` and "here's my code, the third
test fails" as the same situation, and the training transcripts show the three
replies that follow:

| what the learner did | what the tutor did | what it does now |
|---|---|---|
| said they were lost | replied with a textbook definition | gives one concrete thing to try, then asks what they notice |
| asked for the code | gave a complete working function | says once that it won't, then asks what they have tried |
| pasted an attempt | rewrote it wholesale | names the one input or line that breaks and leaves the fix to them |

Three rules, rule-based, no extra model call — the tutor already owns 200s of a
240s budget and there is nothing left to spend. A fourth, "they have proposed an
approach, test it rather than replacing it", is the most useful move a tutor
makes and is **deliberately absent**: nothing cheap separates proposing from
guessing, and mistaking a lost learner for a confident one makes the turn worse
than saying nothing. Detection is high-precision and silent when unsure, so most
turns add nothing to the prompt at all.

It runs inside the session service rather than the gateway, so the voice path
gets it without the websocket handler repeating anything, and the signals are
read from exactly the string that reaches the model. The block is appended
*after* the rules and the learner profile, nearest the generation point, because
it is the most perishable instruction of the three.

**Progress** (`GET /v1/me/progress`). Streak, activity grid, per-skill mastery
and per-track completion, all projected from the append-only observation log
rather than new tables. The log already carries timestamp, skill, correctness
and item; anything stored alongside it would be a second source of truth free
to drift.

---

## 6. Voice — a second front door, not a second tutor

```
  mic ──► WebSocket ──► ffmpeg ──► IndicConformer ──► same /turns call
                                        │                    │
                                   transcript           tutor reply
                                        │                    │
                                        ▼                    ▼
                              sent to the client      Parler TTS (capped)
                              immediately                    │
                                                             ▼
                                                    audio, or speech:false
```

The websocket calls the **exact same** `/sessions/{id}/turns` endpoint the text
box does, and resolves the same learner context and full problem statement. A
voice turn that skipped either would be a second, worse tutor wearing the same
face.

**Stages are reported as they finish**, not batched at the end. ASR takes
seconds but generation can take minutes on CPU, and a learner who has just
spoken has no idea whether they were heard. The transcript goes out *before*
generation starts — that single ordering choice is what turns a blind wait into
a conversation.

| stage | what the learner sees |
|---|---|
| `transcribed` | their own words, immediately |
| `thinking` | a pending tutor bubble |
| `speaking` | the reply text, before any audio |
| final | audio plays, or `speech: false` |

Switching problems mid-turn **closes the socket**, so the remaining frames never
arrive — that, not the epoch check in `onVoiceMessage`, is the real protection.
The epoch check is defence in depth for a frame already in transit when the
close lands, and is verified directly rather than through the UI, because the
socket close makes it unreachable by the normal path.

**Measured honestly on this hardware:** IndicConformer transcribes in 13–34s and
is genuinely usable. `indic-parler-tts` did **not** finish a 16-character
sentence in 840s on CPU — Docker Desktop passes no GPU to containers — so
synthesis is capped and degrades to text-only. Both caps are env-tunable
(`SAHAI_TTS_BUDGET_S`, `SAHAI_TTS_MAX_SECONDS`) and never fire on a GPU host.

---

## 7. Three invariants that hold the serving path together

These are the ones worth knowing before changing anything, because each was
learned from a failure that looked like something else.

**Every hop outlasts the one it calls.**

```
  gateway 270s  >  session 240s  >  tutor generate 200s
                                    TTS 45s (its own budget)
```

Invert that order and the outer caller abandons a request the inner one is about
to answer successfully: the learner sees a 500 and the work is thrown away. A
real turn was lost this way at exactly 240s because the tutor had no generation
cap at all. `tests/test_deadline_ladder.py` asserts the ordering.

**Any model call reachable from a request needs its own deadline.** A client
timeout does not cancel server-side work. The gateway once gave up on a
synthesis at 270s while the ASR process kept generating for ten more minutes at
400% CPU — starving the next request until it timed out too. One abandoned
request took the whole speech service down. Capping the caller cannot fix that;
generation has to stop itself.

**A response for a conversation the learner has left is discarded, not
rendered.** A turn can be in flight for minutes and the learner is free to move
on. Both pages take a token at request time and compare it on arrival: an
`epoch` on the tutor page, bumped whenever the active session changes, and a
counter on the record page's load. Without it, the old problem's transcript
paints into the new problem's conversation, and switching learners can show the
placement survey to someone who already has priors.

---

## 8. Debugging map — the part to keep open

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
| `401 Authorization: Bearer <token> required` | no credential sent | register once: `curl -XPOST localhost:8080/v1/auth/register -H 'content-type: application/json' -d '{"display_name":"me"}'`, then send `-H "Authorization: Bearer <token>"` |
| `401 invalid or expired token` | wrong token, or the row was deleted | mint another: `docker compose exec gateway python /srv/scripts/mint_token.py --learner <id> --name <name>` |
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

## 9. Where errors actually surface

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

## 10. Suggested path from here

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
