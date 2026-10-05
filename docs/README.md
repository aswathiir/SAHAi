# SAHAi — Documentation

Read in order if you are new. Each file is self-contained.

| # | File | What it answers |
|---|---|---|
| 01 | [Research basis](01-research.md) | Which papers, what we took, where we diverged and why |
| 02 | [Architecture](02-architecture.md) | Training loop (Program A) and services (Program B) |
| 03 | [Internals](03-internals.md) | The actual mechanics — reward maths, GRPO, masking, sandbox, BKT |
| 04 | [Findings](04-findings.md) | Every bug and result, with measurements |
| 05 | [Progress report](05-progress-report.md) | Formal status, literature mapping, results |
| 06 | [Roadmap](06-roadmap.md) | Four phases and how much is done |
| 07 | [Operations — Kaggle](07-operations-kaggle.md) | Running training, and the traps that fail silently |
| 08 | [Team split](08-team-split.md) | Four-way ownership |
| 09 | [Track A tutorial](09-track-a-tutorial.md) | RL taught from zero: policy gradients, GRPO, LoRA, BKT — every formula derived and tied to code |
| 10 | [Fundamentals guide](10-fundamentals-guide.md) | **Start here to learn Track A.** Paper → formula → code → worked example, repeated for every concept, beginner pace |
| 11 | [Work log — Sep 2026](11-session-log-2026-09.md) | What was tested on the serving side, what broke, what changed |
| 12 | [The mathematics](12-the-mathematics.md) | Every equation the system computes, tied to the code that computes it |
| 13 | [Running it](13-running-it.md) | **Start here to run the stack.** Setup, the memory setting that matters, and the traps |

## Two programs, not one

The single most useful thing to understand before reading any code:

```
PROGRAM A — TRAINING                 PROGRAM B — SERVING
"make the tutor better"              "let a person talk to the tutor"

runs on:  Kaggle, one GPU            runs on:  Docker, any machine
student:  a 1.5B model pretending    student:  a real human
output:   a LoRA adapter             output:   HTTP responses
code in:  sahai/  notebooks/         code in:  services/
```

They share domain logic through `libs/sahai-core`, which has no GPU, no
database, no network and no ability to execute code.

## Current state

Infrastructure is essentially complete, including a student-facing assessment
interface (`services/gateway/app/static/`) served off the gateway. The
science has started but isn't there yet, and a bottom-up audit on 2026-10-05
moved two things. **The verifier was marking correct code wrong**: 9 of 60
held-out reference solutions failed their own tests because the candidate's
value made a lossy `json` round-trip and `expected` did not, so tuples were
compared against lists. Fixed; held-out reference failures 15% to 8%, and the
maximum achievable solve rate was never 1.0 but 0.850, now 0.917. **The student
can be helped**: handed the reference solution it reaches 0.633 against 0.400
unaided (p=0.009), so the environment does carry signal — but it still fails 17
of 55 problems with the answer in front of it, and every tutor trained so far
sits about 0.2 *below* the unaided baseline rather than competing for the 0.273
of headroom that exists. See [04-findings.md](04-findings.md) #20.

A correctness term judged by
Qwen2.5-7B was built, validated against outcomes, and run as a properly paired
two-arm experiment on 2026-10-04. **It does nothing**: held-out solve differs by
+0.017 on 60 paired problems, eleven discordant pairs split six to five,
p=1.0000, and the arm *not* rewarded for correctness improved correctness more
than the arm that was (+0.393 against +0.276). Both tutored arms remain
significantly below no tutoring at all: **0.367 unaided against 0.217** for the
better arm (p=0.021, p=0.035). The term is removed rather than retuned. See
[04-findings.md](04-findings.md) #18 and #19.

Two benchmarks on 2026-10-02
moved the open question. On 60 paired held-out problems a student with **no
tutor** solves **0.317**, against **0.250** for the best tutored arm and 0.150
for the untrained tutor. Changing how the transcript reaches the solve attempt
(tutor hints only, instead of replaying the student's own confusion back at it)
recovers part of the gap, +0.05 to +0.07, but not significantly (p=0.23), and
the trained adapter still cannot be shown to beat the untrained one it came
from (p=0.42 pooled) despite moving pedagogy 0.758 to 0.986 and leakage 0.299
to 0.045. Tutoring as built does not help this student solve problems. See
[04-findings.md](04-findings.md) #15 and #16.

The 2026-09-29 run is the first in
which the tutor's measured behaviour moved: questions rose from 10% to 61% of
turns and code fell from 27% to 8% across ten epochs. Held-out solve rate did
not follow, at 0.117 on 60 problems against a best-so-far of 0.163 on 20 — a
gap of about half a standard deviation, which this evaluation cannot resolve
either way. Pedagogy acceptance reached 1.000, and that number should be read
with care: it comes from a rule judge with no LLM judge behind it, 73% of
training rollouts scored exactly 1.0, and 88% of those never solved the
problem. 416 tests pass. A tutor that generalizes is the open item that gates
everything else — see [06-roadmap.md](06-roadmap.md).
