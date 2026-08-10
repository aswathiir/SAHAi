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

## Two programs, not one

The single most useful thing to understand before reading any code:

```
PROGRAM A — TRAINING                 PROGRAM B — SERVING
"make the tutor better"              "let a person talk to the tutor"

runs on:  Kaggle, one GPU            runs on:  Docker, any machine
student:  a 0.5B model pretending    student:  a real human
output:   a LoRA adapter             output:   HTTP responses
code in:  sahai/  notebooks/         code in:  services/
```

They share domain logic through `libs/sahai-core`, which has no GPU, no
database, no network and no ability to execute code.

## Current state

Infrastructure is essentially complete; the science has barely started. 100
tests pass. A tutor that demonstrably learns is the open item that gates
everything else — see [06-roadmap.md](06-roadmap.md).
