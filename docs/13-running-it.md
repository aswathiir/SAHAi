# Running it

Everything needed to get the stack up, plus the traps that cost real time.
Training on Kaggle is [07-operations-kaggle.md](07-operations-kaggle.md); this
covers the local stack and the things that look like bugs but are not.

---

## 0. Before anything: Docker memory

**This is the single most important setting, and the default is wrong for
this project.**

Docker Desktop allocates half the host's RAM by default. On a 16GB machine
that is 7.65GB, and the tutor (3GB of weights) plus the speech models (3.4GB)
do not fit alongside everything else. The weights are memory-mapped, so under
pressure the page cache is evicted and the model is re-read from disk on every
forward pass.

Measured, same prompt, same model:

| Docker VM | major page faults | one generation |
|---|---|---|
| 7.65 GB | 8,807 | 187.3s |
| 11.67 GB, cold | 1,691 | 86.4s |
| 11.67 GB, warm | **0** | **60.1s** |

**Docker Desktop -> Settings -> Resources -> Memory.** Set 10-12GB on a 16GB
host. Apply & Restart.

Verify:

    docker info | grep "Total Memory"

> **After any resource change, run `docker compose up -d`.** The VM restart
> leaves `postgres` stopped, and `gateway`, `session` and `tracer` then
> crash-loop on `socket.gaierror: Name or service not known`. Three services
> look broken because one did not come back.

---

## 1. Start the stack

    make up        # build and start, waits for /health
    make smoke     # one end-to-end request through the gateway
    make down

The UI is at **http://localhost:8080** — the only published port. Everything
else talks over internal networks.

First run downloads base models (several GB) and builds seven images. Expect
10-20 minutes. Later runs are cached.

### Without a GPU, without models

    SAHAI_TUTOR_BACKEND=stub docker compose up -d

The stub serves deterministic Socratic questions that pass the same pedagogy
checks a real model faces. Every service boundary, the sandbox and the whole
UI work; only the generation is fake. Use it whenever you are not testing the
model itself — it is seconds per turn instead of a minute.

### Serving a trained adapter

    SAHAI_TUTOR_BACKEND=hf \
    SAHAI_ADAPTER_PATH=/srv/artifacts/<name> \
    SAHAI_DEVICE=cpu docker compose up -d tutor

Adapters live in `artifacts/`, mounted read-only at `/srv/artifacts`. Drop a
directory in and reference it by path; `.env` holds the current selection.

---

## 2. What healthy looks like

    docker compose ps        # all seven Up (healthy)
    curl -s localhost:8080/health | python3 -m json.tool

    {"status": "ok",
     "dependencies": {"session":"ok","executor":"ok","tutor":"ok",
                      "tracer":"ok","asr":"ok"}}

The gateway reports `degraded` while any dependency is down, which is the
correct answer during startup — the tutor and ASR load models *after* binding.

---

## 3. Traps

### `healthy` does not mean "ready to serve"

Containers that load models after binding can report healthy in a gap and then
go back. Poll the endpoint, not the flag:

    until curl -sf -o /dev/null localhost:8080/health; do sleep 3; done

### Two concurrent builds race

Running `docker compose build <svc>` twice while the first is still going
leaves the tag on the *earlier* image. The container then runs code that looks
updated because *something* changed. **Verify a deploy by reading the code
inside the container**, not by observing a difference:

    docker compose exec tutor grep -n "max_new_tokens: int = Field" \
        /srv/services/tutor/app/main.py

### The browser caches the UI hard

Static pages are baked into the gateway image. After a rebuild, a normal
reload can still serve the old page. Hard-reload (`Cmd-Shift-R`), or append a
cache-buster while iterating: `localhost:8080/progress.html?v=2`.

### The tutor image occasionally fails to build

`pip install torch` against the PyTorch CPU index has failed intermittently
with a JSON decode error. It is transient — retry. The rest of the stack
builds fine without it, and the previously built tutor image keeps working:

    docker compose up -d --no-build gateway tracer session

### A turn takes about a minute

Measured steady state on CPU: **~60s for a ~9-token reply**, roughly 6.7s per
token, with the model fully resident. The first request after a restart also
pays model load (up to ~125s).

`SAHAI_GENERATE_MAX_SECONDS` (default 200) is the binding constraint on reply
*length* — at 6.7s/token it permits about 30 tokens, which is why replies are
short and sometimes cut. `max_new_tokens` (192) is never reached. Raising
reply length means raising `max_time`, at proportional latency cost.

### One generation at a time

`SAHAI_TUTOR_MAX_CONCURRENT` defaults to 1. A second concurrent turn gets
**503 with `Retry-After`** rather than queueing, because the caller's own
deadline (200s) is shorter than the job it would wait behind. On CPU two
parallel generations do not run twice as fast — they halve each other's CPU
and both miss their deadline. Raise it on a GPU host.

### GitHub import is rate-limited

`POST /v1/me/import_repo` uses the unauthenticated GitHub API: **60 requests
per hour per IP**. Exhausting it returns a clear 429. It reads at most 60
files per repo.

### `mint_token.py` is not in the container

The README documents
`docker compose exec gateway python /srv/scripts/mint_token.py --list` for
attaching a token to a pre-auth learner. `scripts/` is not mounted into the
gateway, so that command fails. Run it from the host against the database
instead, or register a fresh learner.

---

## 4. Tests

    make test          # everything: core, services, research
    make test-core
    make test-services
    make test-research

No Docker needed — they run against the source. Current totals:

    sahai-core          30
    services/executor   13
    services/tracer     29
    services/session    26
    services/gateway    99
    services/asr         9
    research (sahai/)  109
    ------------------------
    total              315

---

## 5. Looking inside

    docker compose logs -f gateway            # follow one service
    docker compose logs tutor --tail=50

    # the database
    docker compose exec postgres psql -U sahai -d sahai
      \dt                                     # learner_identity, sessions,
                                              # turns, observations,
                                              # skill_mastery, prior_work
      SELECT role, left(content,60) FROM turns
        WHERE session_id='<id>' ORDER BY idx;

    # is the tutor paging? (0 is what you want)
    docker compose exec tutor sh -c \
      'grep ^pgmajfault /sys/fs/cgroup/memory.stat'

    # which adapter is actually loaded
    docker compose exec tutor sh -c 'echo $SAHAI_ADAPTER_PATH'

---

## 6. Training

Training does not run here — it needs a GPU and lives on Kaggle. See
[07-operations-kaggle.md](07-operations-kaggle.md), and read its §2b first:
the notebook and the dataset are separate uploads and drift apart silently,
which has already cost one run its headline number.

One addition since that document was written: the rollout prompt now includes
the learner-context block, imported from `sahai_core.learner_context` so
training and serving share one definition. **`sahai_core` must be uploaded
beside `sahai`** or the import falls back to a stub returning `""` and the
change is silently a no-op:

    rsync -a --delete --exclude='__pycache__' sahai/ kaggle_upload/sahai/
    rsync -a --delete --exclude='__pycache__' \
        libs/sahai-core/sahai_core/ kaggle_upload/sahai_core/
    kaggle datasets version -p kaggle_upload --dir-mode zip -m "..."

Confirm in the run log that the block appears in a rollout prompt before
trusting the run.
