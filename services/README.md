# Services — the serving half

Six containers behind one published port. The training half lives in
`sahai/` and shares domain logic through `libs/sahai-core`; nothing here
imports torch except `tutor`.

    host :8080
        |
    gateway ---- auth, rate limit, request id, aggregate health
        |
    session ---- the only service that knows a conversation exists
      |  |  \
      |  |   `-- tutor    stateless inference (stub | hf | vllm)
      |  `------ tracer   per-learner BKT mastery, ZPD, prior work
      `--------- executor untrusted code, no egress
                 asr      speech in/out

## Who owns what

| Service | Owns | Stateful | Notes |
|---|---|---|---|
| `gateway` | ingress | no | Only host-published port. Verifies the bearer token and forwards an internal `x-learner-id`. |
| `session` | transcripts, turn state | Postgres | Production counterpart to the training dialogue engine. |
| `tutor` | inference | no | Holds the model. The only container that needs a GPU. |
| `tracer` | mastery, prior work | Postgres | BKT per (learner, skill), plus ZPD selection. |
| `executor` | running untrusted code | no | No database, no egress, no concept of a user. |
| `asr` | speech | no | Indic ASR/TTS. |

`session` is the only service on both the application and sandbox networks:
it must reach `executor`, and `executor` must reach nothing.

## Invariants worth knowing before changing anything

**A turn is atomic.** The learner's turn and the tutor's reply commit
together. Committing the learner's turn first left dangling turns whenever
generation timed out, and a retry appended a second learner turn —
consecutive same-role turns break the alternation that prompt assembly and
the pedagogy judge both assume, silently.

**Every hop outlasts the one it calls.** `gateway 270s > session 240s >
tutor 200s`. Invert it and the outer caller abandons work the inner one is
about to finish. Measured turns have reached 202.9s, so this is a live
constraint, not a theoretical one.

**The tutor runs one generation at a time.** `/generate` is a plain `def`,
not `async def` — the body is synchronous and CPU-bound for minutes, and on
the event loop it starved every other request in the process, including
`/health`. A `BoundedSemaphore` (`SAHAI_TUTOR_MAX_CONCURRENT`, default 1)
bounds concurrency; over the limit the service answers **503 with
`Retry-After`** rather than queueing behind a job the caller would outlive.
`/health` reports `busy` but still returns 200: a non-200 would make Docker
restart the container mid-generation.

**Scoring is separated from execution.** `libs/sahai-core` is enforced by
test to expose no code runner and to import no torch, fastapi, sqlalchemy or
datasets. Untrusted code runs only in `executor`.

**The answer never reaches the client.** `GET /v1/problems/{id}` returns the
function name and test cases but no `solution` field.

## Running it

    make up      # build and start
    make smoke   # one end-to-end request through the gateway
    make test    # every suite in the monorepo
    make down

The stack runs with no GPU: `SAHAI_TUTOR_BACKEND=stub` serves deterministic
Socratic questions that pass the same pedagogy checks a real model faces.

To serve a trained adapter:

    SAHAI_TUTOR_BACKEND=hf \
    SAHAI_ADAPTER_PATH=/srv/artifacts/<name> \
    SAHAI_DEVICE=cpu docker compose up -d tutor

On CPU expect roughly **0.65s per generated token**. With `max_new_tokens=96`
that is about a minute per turn; the first request after a restart also pays
model load, which has measured up to ~125s.

## Executor containment

Layered, because any one layer can fail:

| Layer | Control |
|---|---|
| network | `internal: true`, no egress |
| filesystem | read-only root, 64MB tmpfs at `/tmp` |
| privileges | non-root uid 10001, `cap_drop: ALL`, `no-new-privileges` |
| resources | `pids_limit: 128`, `mem_limit: 1g`, `cpus: 1.0` |
| process | fresh interpreter per test case, `-I -S`, scrubbed env |
| protocol | verdict read from a sentinel line, so candidate stdout cannot forge a pass |

Thirteen tests cover it, including an infinite loop, a memory bomb, a fork
bomb, and a candidate that prints the sentinel to fake its own success.

## Known gaps

* `services/tutor` has no tests — the thinnest service, and the only one
  holding a model.
* Gateway rate limiting is untested; identity handling is covered by 99.
* Identity authenticates a *bearer*, not a person: no second factor, no
  recovery, no revocation beyond deleting the row. The internal hop trusts
  `x-learner-id`, which rests on the compose network split rather than on
  cryptography.
* `scripts/` is not mounted into the gateway, so the documented
  `mint_token.py` recovery command fails there.
