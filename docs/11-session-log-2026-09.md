# Work log — September 2026

What was tested, what broke, and what was changed. Every number here came
from running the system, not from reading it. Entries are stated as
**claim → what testing showed → what was done**, because the claim is what
the report makes and the measurement is what decides whether it holds.

Companion to [04-findings.md](04-findings.md), which covers the training
side. This file covers the serving side and the September test pass.

---

## 1. Four claims, tested

| Claim | Verdict |
|---|---|
| The tutor teaches from the learner's current mastery | Mechanism verified; model compliance not |
| A websocket keeps the session live | Holds |
| A solved-problems repo can be connected end to end | Was broken two ways; both fixed |
| The assessment adapts per learner | Was false; implemented |

### 1.1 Mastery-conditioned teaching — mechanism real, compliance doubtful

`_learner_context()` turns BKT posteriors into an instruction rather than a
number, and it genuinely differs per learner:

    hash_maps 0.65 -> "Solid on hash maps. Do not re-explain it; push on
                       what is new here."
    no mastery     -> "New to hash maps. Build the idea before asking them
                       to apply it."

Thresholds are `NEW < 0.35` and `SOLID >= 0.65`; only skills the current
problem touches are mentioned; a tracer outage degrades to a generic hint
rather than failing the turn.

**Not established:** that the 1.5B policy obeys it. Across six live turns the
tutor ignored the stuck rule, ignored the pasted-attempt rule, disclosed the
approach twice, and once called broken code correct. The served adapter was
trained under a prompt containing none of these blocks, which
[06-roadmap.md](06-roadmap.md) already lists as a known gap.

### 1.2 Voice websocket — holds

    bad token   -> closed 1008 "invalid or expired token", zero frames sent
    valid token -> {"stage":"authenticated","learner_id":"lnr_..."}, stays open

Authentication is on the first frame, not the query string.

### 1.3 Repo import — two defects

**Defect A: the importer never worked on a flat repository.**
`first_commits()` derived the slug from `parts[1]` of each path, which on a
`<slug>/submission-1.py` layout is the *filename*. `read_solutions()` keyed on
the directory. The two halves could therefore never join, so techniques were
parsed and silently discarded.

Measured on a three-problem fixture:

| | before | after |
|---|---|---|
| problems solved | 1 | 3 |
| sources parsed | 3 of 1 (an impossible ratio) | 3 of 3 |
| techniques named | 0 | 3 |
| hash_maps weight | 0.00 | 2.00 |

It survived because **none of the 18 importer tests exercised
`first_commits`** — they all covered the pure functions around it. Two
regression tests were added: one pins the flat layout, one pins that
`first_commits` and `read_solutions` agree at any depth.

**Defect B: there was no user-facing path at all.** `prior_work` appeared in
zero HTML files. The feature existed as a CLI only.

### 1.4 Assessment page — was identical for everyone

`renderPlacement()` rendered a fixed list from a constant. It took no mastery,
marked nothing, and silently discarded answers for skills the server would
refuse to overwrite. The page promised *"re-taking only changes skills you have
not attempted yet"* while giving the learner no way to know which those were.

---

## 2. Changes made

### 2.1 `sahai_core.techniques`

The AST technique detector moved out of the importer script into the shared
library so the gateway can run **identical** logic on a repo connected from
the browser. One definition, two consumers. It depends only on `ast` and `re`;
anything needing the problem bank's skill vocabulary stayed in the importer,
because the services do not carry the research package.

### 2.2 `POST /v1/me/import_repo`

Reads a public repository over the GitHub API — no clone, because the gateway
image has no `git` and cloning arbitrary repositories inside a request path is
worse than reading the few files that matter. Capped at 60 files.

**Source is read and discarded in the handler.** What reaches storage is a
technique name and a skill. `prior_work` has no column for code, so keeping
any would require a migration.

Verified against a real public repository of 415 Python files using a layout
unlike NeetCode's:

    problems_read 59 | recorded 29 | truncated true | 23s
    skills: binary_search, bit_manipulation, hash_maps, linked_lists,
            sorting, two_pointers

Three further bugs surfaced while testing this endpoint:

* titles rendered as `". two sum (leetcode)"` — `str.capitalize()` lowercases
  everything after the first character, and the leading-index regex left a
  stray separator;
* `GET /v1/me/prior_work` returned 500 — the tracer wraps rows in
  `{"items": [...]}` and the endpoint declared `list[dict]`, so FastAPI
  rejected its own response;
* it then returned 3 rows of 29 — the tracer defaults `limit=3`, which is
  correct when the rows feed a system prompt and wrong when they are the
  learner's own record.

### 2.3 `graded` on the mastery response

The tracer now returns which skills have at least one graded attempt. The
gateway passes the tracer response through untouched, so no gateway change was
needed. The placement survey renders those rows as *"Graded — kept from your
own work"* instead of offering a question whose answer is discarded, and the
note counts them.

Behaviour verified end to end:

| skill | observations | before | after claiming "Confident" |
|---|---|---|---|
| arrays | 2 | 0.2978 | **0.2978 — refused** |
| sorting | 2 | 0.2978 | **0.2978 — refused** |
| hash_maps | 0 | — | **0.6500 — seeded** |

The observation log stayed at 7 rows throughout: placement writes no
observations, which is what keeps a self-report out of the sequence a
knowledge-tracing model would later train on.

### 2.4 Tutor concurrency — the cause of "the tutor stops responding"

The container had been **`unhealthy` for 47 hours**:

    Health check exceeded timeout (10s)
    TimeoutError: timed out

`/generate` was declared `async def`, but its body is synchronous, CPU-bound
work lasting minutes. On FastAPI an `async def` endpoint runs **on the event
loop**, so while a turn generated at ~430% CPU nothing else in the process
could be served — not a second learner's turn, and not `/health`. Docker's own
probe timed out, the service was marked unhealthy, and further requests waited.

Three changes:

* `def generate`, not `async def` — FastAPI runs a sync endpoint in a worker
  thread and the loop stays free;
* a `BoundedSemaphore`, default 1, acquired non-blocking. Over the limit
  returns **503 with `Retry-After`** rather than queueing: the session
  service's deadline is 200s, shorter than the job a request would queue
  behind, so queueing converts a fast refusal into a slow one;
* `/health` reports `in_flight`, `capacity` and `busy` but still answers 200
  when busy. A non-200 would make Docker restart the container
  mid-generation, killing the turn that caused the load.

Wired as `SAHAI_TUTOR_MAX_CONCURRENT`, so a GPU host raises it with one
variable.

### 2.5 Latency, and a correction

An early measurement of 18.6s was reported as the warm figure. **That was
wrong** — it was a single outlier. Three consecutive warm turns:

    143.7s | 202.9s | 123.7s

Reply lengths explain the outlier: a **34-character** reply took 124.9s while
a **162-character** reply took 18.6s. Wall time tracks tokens *generated*, not
tokens *kept* — a generation that runs to `max_new_tokens` and is then trimmed
back to a sentence boundary still pays for every token. At roughly 0.65s per
generated token on CPU, a 192-token cap costs about two minutes regardless of
what survives.

Turn two at 202.9s **exceeded the tutor's own 200s deadline**, which makes the
deadline ladder in [02-architecture.md](02-architecture.md) a live concern
rather than a theoretical one.

Consequently `max_new_tokens` dropped 192 -> 96. Surviving replies measured
~30 tokens, so the cap was mostly funding output that `trim_incomplete` threw
away. The failure mode is safe: when no sentence boundary exists inside the
budget, `trim_incomplete` returns `""` and the backend falls back to the raw
text (`trim_incomplete(decoded) or decoded`), so a cut reply is possible but an
empty one is not.

### 2.6 Elapsed timer

A static "Tutor is thinking…" for two minutes is indistinguishable from a
hang, and was read as one — learners sent a second message believing the first
had failed. The pending bubble now counts up and names the expected wait after
five seconds. `clearInterval` runs in `finally`, before the epoch guard, so the
timer is cleared on all four exit paths including the one where the learner
switches problems mid-flight.

---

## 3. Still open

| Item | Note |
|---|---|
| Model compliance with the mastery block | Unmeasured beyond six turns; expect it to be poor until the policy is trained under the serving prompt |
| Full voice round trip | Socket and auth verified; audio -> ASR -> tutor -> TTS not exercised |
| `mint_token.py` | The README instructs running it inside the gateway container; `scripts/` is not mounted there, so it fails |
| Tutor image rebuild | `pip install torch` against the PyTorch CPU index has failed intermittently |
| GitHub import rate limit | Unauthenticated, 60 requests/hour per IP; the endpoint surfaces this as 429 but does not accept a token |

## 4. Test counts

    sahai-core          30
    services/executor   13
    services/tracer     29
    services/session    26
    services/gateway    99
    services/asr         9
    research (sahai/)  103
    ------------------------
    total              309

The repository documentation elsewhere states 243. That figure is stale.

---

## 5. Where the 200s per turn actually goes

Four hypotheses were tested and killed before the cause was found. Each is
recorded because the eliminations are what make the conclusion trustworthy,
and because three of them sounded right enough to ship.

| Hypothesis | Test | Result |
|---|---|---|
| The token cap is the constraint | `max_new_tokens` 16 vs 192 | 221.7s vs 173.7s — the *smaller* cap was slower |
| Prompt length dominates (prefill) | tiny vs ~600-token prompt | 196.0s vs 208.1s — negligible |
| `max_time=200` is truncating everything | check `complete` flag | two runs ended on **EOS** and still took ~180s |
| bfloat16 is unaccelerated on ARM | matmul benchmark | bf16 is **0.6x** the cost of fp32 here — faster, not slower |
| Unmerged LoRA adds per-layer overhead | `merge_and_unload()` in-process | 237.3s -> 227.9s, a 4% gain |

### The cause: the model is being read from disk, token by token

Generation at batch size 1 is memory-bound — every token requires reading all
1.5B weights. The container's own counters:

    anon          947 MB     actual allocations
    file        3,161 MB     page cache: weights are mmap'd
    pgmajfault   106,540     major faults = disk reads

HuggingFace loads safetensors via **mmap**, so the weights live in page cache,
not anonymous memory. That is why the tutor's RSS reads 1.9GB for a 3GB model
and never grows. Page cache is the first thing evicted under pressure, and the
Docker VM is capped at 7.653GB while the ASR service holds 3.4GB of it.

Measured directly:

    generate: 187.3s  out=34 chars
    major page faults during that call: 8,807

About 980 major faults per token. At a virtualised disk's fault latency that
accounts for the whole per-token cost on its own.

Stopping the ASR container to free 3.4GB recovered ~35% (173.7/196.0s ->
120.5/118.7s on the EOS runs) — real, but far short of the 100x the theory
first predicted, because the VM is still too small to hold both models.

### What follows

* The claim in `backends.py` of "~1.8 tok/s measured" was never true on this
  host. Actual is **~0.07 tok/s**, roughly 25x worse, and the gap is paging
  rather than compute. The comment has been corrected.
* `max_new_tokens` was dropped 192 -> 96 on the theory that halving tokens
  halves time. Measured: 209.4s and 208.6s, unchanged. **Reverted to 192**,
  since a lower cap bought no speed and only narrowed the room before
  truncation.
* The fix is memory, not code: raise the Docker VM above 7.653GB, or stop
  co-locating the ASR models with the tutor. Loading weights into anonymous
  memory instead of mmap would also stop the eviction, but costs RAM, which
  is the scarce resource here.

## 6. Two bugs introduced while fixing this, and caught

Recorded because both are the kind that look like the original problem.

**The health check failed because of the health check.** Adding `in_flight`,
`capacity` and `busy` to `/health` while the handler was still annotated
`-> dict[str, str]` made FastAPI reject its own response. Every probe returned
500, the container went `unhealthy` with a failing streak of 137, and the
symptom was indistinguishable from the event-loop starvation the change was
meant to fix. Annotation is now `dict[str, object]`.

**Two concurrent image builds raced.** `docker compose build tutor` was
started twice while the first was still running, and the tag ended up on the
earlier image. The deployed container had the new token cap but none of the
concurrency changes — and because the cap *had* changed, it looked updated.
Verify a deployment by reading the code inside the container, not by checking
that something changed.


---

## 7. Resolved: the fix was Docker's memory slider

Raising the Docker Desktop VM from its default 7.653GB (half of 16GB host
RAM) to 11.67GB, and letting the page cache warm:

| | major page faults | generation |
|---|---|---|
| 7.65GB, cold | 8,807 | 187.3s |
| 11.67GB, first call | 1,691 | 86.4s |
| 11.67GB, warm | **0** | **60.1s** |

Zero major faults, and `file` in the tutor's cgroup now holds 3,061MB — the
whole model resident. **3.1x faster end to end**, and faults and latency moved
together throughout, which is what makes the diagnosis trustworthy rather
than merely consistent.

Steady state on this hardware is therefore **~60s for a ~9-token reply**, or
roughly **6.7s per token**, with no paging at all. That is the floor for
CPU inference here; the earlier 187s was more than half disk I/O.

### Consequences for values tuned against the old number

* `GENERATE_MAX_SECONDS = 200` is still the binding constraint on reply
  length. At 6.7s/token it permits about 30 tokens, which is exactly the
  ~34-character replies observed throughout. `max_new_tokens=192` is never
  reached and remains irrelevant — raising reply length means raising
  `max_time`, at a proportional cost in latency.
* The deadline ladder (270 > 240 > 200s) is now comfortable rather than
  marginal: a typical turn completes in ~60s against a 200s tutor budget.
  Before this fix a turn measured 202.9s and exceeded it.
* The UI timer reads "replies usually take about a minute", which is now
  accurate. It was written when the figure was believed to be different and
  happens to match the measured steady state.

### The operational lesson

Nothing in the repository was wrong. The cost came from a Docker default —
half of host RAM — that happened to sit just below what two co-resident
models needed, and the symptom was indistinguishable from slow compute. It
was only found by counting major page faults, and it would not have been
found by reading code.

**After changing Docker Desktop resources, run `docker compose up -d`.** The
VM restart left `postgres` stopped while `gateway`, `session` and `tracer`
crash-looped on exit 3 with `socket.gaierror: Name or service not known` —
three services looking broken because one had not restarted.

---

## 8. The unaided baseline — the missing control, measured

[§1](#) listed "no baseline" as the defect that made the headline number
uninterpretable. It has now been run.

**Protocol.** The same 20 held-out problems the 2026-09-19 evaluation used
(`mbpp_11`–`mbpp_30`, parsed with the project's own `_parse_assert` and
`_extract_function_signature`), the same student model, the same
`attempt_solution` system prompt, greedy decoding — and **no dialogue**. The
message list is the system prompt followed by "Now write your solution:", with
nothing in between. That is exactly the `solve_without` the ACE design
specifies. Grading uses `CodeVerifier` against the problems' real test cases.

**Result.**

| | solve rate |
|---|---|
| Unaided baseline | **0.150** (3 of 20) |
| After tutoring (2026-09-19) | **0.175** |
| Difference | **+0.025** |

Solved unaided: `mbpp_14`, `mbpp_17`, `mbpp_21`.

**Reading.** At n=20 the standard deviation of the estimate is ~0.08, so
+0.025 is about **0.3 standard deviations** — indistinguishable from zero. In
problem terms the student solves ~3.0 unaided and ~3.5 tutored: half a problem.

    On 20 held-out problems, a simulated student solved 15.0% unaided and
    17.5% after tutoring — a difference of 0.3 standard deviations, which this
    evaluation cannot resolve.

That sentence is what the Results section should say. It does not weaken the
paper: an unanchored 17.5% could have been *below* baseline with no way to
tell, which was the real risk. A null result with a control is a result.

It also corroborates [§14 of 04-findings.md](04-findings.md) from an
independent direction. If `r_sol` carries a fifth of the usable within-group
variance and 22 of 24 rollouts score exactly 0.00, the prediction is that the
tutor barely moves the outcome. It barely moves the outcome.

**Caveats, which belong with the number.**

* The eval's student runs **4-bit quantised**; `bitsandbytes` was unavailable
  in the container used, so this ran **bf16**. Higher precision should if
  anything help the unaided student, making 15.0% a conservative floor — but
  it is not byte-identical to the evaluation it is compared against.
* n=20, the same underpowered set [§12 of 04-findings.md](04-findings.md)
  criticises. This comparison inherits that limitation; it needs the
  60-problem set to be tight.
* One seed, greedy, so there is no variance estimate on the baseline itself.

## 9. Learner context added to the training prompt

The policy was previously optimised on prompts that contained **no learner
signal at all**:

    You are a tutor... STRICT RULES... Problem: {title}\n{description}
    (You know the solution but must NOT reveal it.)

Mastery reached training only as `alpha` in the reward and as ZPD problem
selection. Both change what the tutor is *scored on* or *shown* — neither
changes what it can *read*. A policy cannot learn to condition on a signal
absent from its input, so the serving-time
`WHAT THIS LEARNER ALREADY KNOWS` block was an instruction the adapter had
never seen.

**Changed.** `sahai_core/learner_context.py` is now the single definition.
The gateway imports it (its private 50-line copy is deleted) and
`DialogueEngine.run` builds the same block from the simulated student's own
BKT state, passing it to `TutorPolicy.generate`. Because the ZPD sampler moves
that state during a run, the block varies across rollouts rather than being a
constant the policy can ignore.

**This does not improve the adapter on disk.** It changes what a future run
optimises against. It also **breaks comparability**: every rollout in the next
run is generated under a different prompt than every rollout before it, so
run-history numbers either side of this change are not directly comparable.
That is a deliberate violation of the one-variable-per-run rule, recorded here
rather than discovered later.

**Deployment.** Kaggle receives only `sahai/`. The import falls back to a stub
returning `""` if `sahai_core` is absent, so a missing upload makes the change
a **silent no-op**:

    rsync -a --delete --exclude='__pycache__' \
        libs/sahai-core/sahai_core/ kaggle_upload/sahai_core/

Confirm the block appears in a rollout prompt in the run log before trusting
the run.
