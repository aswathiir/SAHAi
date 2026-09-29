# The mathematics

Every equation the system actually computes, tied to the code that computes
it. Nothing here is generic RL background: if a formula appears below, some
file in this repository evaluates it.

Read alongside [03-internals.md](03-internals.md), which covers the mechanics
around these equations, and [04-findings.md](04-findings.md), which covers what
happened when they were run.

---

## 1. What is being optimised

One policy is trained: the tutor, `pi_theta`. The student, the judge and the
verifier are all frozen. The objective is the standard policy-gradient one,

    J(theta) = E [ R(tau) ]        tau ~ pi_theta

where a trajectory `tau` is one complete tutor-student dialogue and `R` is the
conversation-level reward below. The expectation is estimated by sampling `G`
dialogues per problem and averaging.

Two things follow from the reward being **conversation-level** rather than
per-token. There is no per-step credit assignment: every token in a dialogue
receives the same advantage. And the variance of the estimate is set by how
many dialogues are sampled, not by how long they are.

---

## 2. The reward

`sahai/reward/combined.py`

    r_SAHAI = (r_sol - alpha) + (r_ped - 1) * lambda - gamma * L

with a hard branch that overrides it:

    r_ped == 0   =>   r_SAHAI = -lambda

Defaults are `lambda = 1.0`, `gamma = 0.5`.

Each term is shaped so that **zero means "as expected"** and the policy is
rewarded only for beating that:

| term | range | zero when |
|---|---|---|
| `r_sol - alpha` | `[-1, 1]` | the student did exactly as well as their ability predicts |
| `(r_ped - 1) * lambda` | `[-lambda, 0]` | the dialogue passed every pedagogy check |
| `-gamma * L` | `[-gamma, 0]` | nothing leaked |

So a perfect dialogue that produces no improvement scores 0, not 1. Only
teaching that moves the student above their own baseline is positive.

### 2.1 `r_sol` — the outcome term

After the dialogue the student attempts the problem. The attempt runs in a
subprocess against the problem's unit tests, and

    r_sol = (tests passed) / (tests total)

averaged over `K` samples. The all-or-nothing reading is kept separately as

    solved = |{k : r_sol_k == 1}| / K

`r_sol` is what the gradient sees; `solved` is what run-to-run comparisons are
made on, because it is the series every earlier run reported.

### 2.2 `alpha` — the ability baseline

    alpha = (1/|S|) * sum over s in S of  p(s)

the mean BKT mastery across the skills the learner has touched
(`BKTTracer.get_ability`). Subtracting it is what stops the tutor being paid
for a student who could already solve the problem. It is the one place the
learner model enters the reward directly.

### 2.3 `r_ped` — the pedagogy term

Three checks, each a predicate over the dialogue's tutor turns, scored as the
**fraction of turns that pass**:

    r_ped = (1/3) * sum over c in checks of  (turns passing c) / (tutor turns)

The checks are all of the form *did the tutor give the answer away*: no code
blocks, no solution patterns, no dangling colon.

Two shape decisions here were forced by measurement, not taste:

**Graded, not binary.** The source specification is a product of judge
verdicts,

    r_ped = product over m of  1[J_m(a_T, s_T) = accept]

which is viable under PPO and **structurally incompatible** with the
group-relative estimator in section 3 — see 3.1.

**Per-turn, not per-dialogue.** Scoring a check as "every turn passed" makes
the probability of passing decay with turn count independently of quality:
over 304 rollouts, mean `r_ped` by tutor-turn count was 0.720 / 0.639 / 0.537 /
0.453 for one to four turns. Perfectly monotonic, and the policy responded by
ending conversations early. Scoring the fraction of turns cut the spread across
dialogue lengths from 0.267 to 0.044.

### 2.4 `L` — leakage

    L = max( lexical_overlap , 1 if tutor_code_solves else 0 )

Lexical overlap counts **rare** tokens shared between the tutor's text and the
reference solution, with the problem statement's own tokens subtracted, so a
tutor saying "string" on a string problem is not charged for it.

The execution clause is the part that matters. Overlap alone scored **0.000**
for a tutor that wrote a complete working solution using a different algorithm
than the reference — it was measuring plagiarism of one implementation, not
disclosure of the answer. Code that passes the problem's own tests **is** the
answer, however it is written, so it scores 1.

---

## 3. The policy update

`sahai/training/grpo.py`

### 3.1 Group-relative advantage

For a group of `G` rollouts sampled on the same problem:

    mu    = (1/G) * sum_i r_i
    sigma = sqrt( (1/G) * sum_i (r_i - mu)^2 )
    A_i   = (r_i - mu) / (sigma + 1e-8)

There is no value network. The group mean **is** the baseline, which is what
removes the critic and halves optimiser memory.

The consequence is the single most important fact about this system:

> **Within-group reward variance is the entire learning signal.**

If every rollout in a group scores identically, `sigma = 0`, every `A_i = 0`,
and the update is exactly zero regardless of how good or bad the dialogues
were. This is why the binary `r_ped` of 2.3 cannot work here: every rollout
failed at least one rubric, so all `G` rewards were `-lambda`, `sigma` was 0,
and epoch 0 logged `loss = 0.0000`. PPO tolerates constant returns because a
learned baseline absorbs them; a group z-score cannot.

### 3.2 The clipped surrogate

Per token `t`, with `rho_t` the importance ratio against the policy that
generated the rollout:

    rho_t     = exp( log pi_theta(t) - log pi_theta_old(t) )
    L_policy  = - (1/|T|) * sum over t in T of  min( rho_t * A ,
                                                    clip(rho_t, 1-eps, 1+eps) * A )

`eps = 0.2`. `T` is the set of tutor tokens (section 3.4). The `min` is what
makes it conservative: where the policy has already moved far on a token, the
clipped branch is the smaller of the two and the objective flattens, so that
token stops contributing gradient.

`log pi_theta_old` is cached **before the first optimizer step of the epoch**.
This matters because 32 rollouts are collected under one policy and then
consumed by 16 sequential steps; without the ratio, steps 2-16 are uncorrected
off-policy updates. For most of this project `clip_epsilon` sat in settings
unused and the update was `-A * mean_log_prob` — REINFORCE with a z-scored
baseline, not GRPO.

### 3.3 KL penalty

    L_KL  = beta * (1/|T|) * sum over t in T of ( log pi_theta(t) - log pi_ref(t) )
    Loss  = L_policy + L_KL

`beta = 0.05`. The reference policy is the same weights with the LoRA adapters
disabled, so it costs no extra memory.

Note this estimator can go negative — it is a sample mean of a log-ratio, not a
KL divergence, which is non-negative by definition. The k3 estimator is the
correct form and is listed as outstanding in the roadmap.

### 3.4 Token masking

Only tokens the tutor sampled enter either sum:

    T = { t : t is in an assistant span of this dialogue }

This is why the student's text may be post-processed and the tutor's may not.
Editing tutor text would compute log-probabilities over tokens the policy never
produced, making `rho_t` meaningless.

### 3.5 What is actually trained

LoRA on `q_proj, k_proj, v_proj, o_proj`, rank `r = 8`, `alpha = 16`:

    W_effective = W_frozen + (alpha / r) * B A        A: r x d,  B: d x r

About 0.14% of parameters are trainable. One run is 10 epochs x 4 problems x
`G=8` = 320 dialogues and 160 optimizer steps.

---

## 4. The learner model — BKT

`sahai/agents/tracer.py`

Per skill, a posterior `p` that the learner has mastered it. Parameters:
`p_init = 0.3`, `p_learn = 0.1`, `p_guess = 0.2`, `p_slip = 0.1`.

**Observation update.** After a correct attempt,

    P(correct)  = p (1 - p_slip) + (1 - p) p_guess
    p_posterior = p (1 - p_slip) / P(correct)

after an incorrect one,

    P(incorrect) = p * p_slip + (1 - p)(1 - p_guess)
    p_posterior  = p * p_slip / P(incorrect)

**Learning transition**, applied after either:

    p' = p_posterior + (1 - p_posterior) * p_learn

There is no forgetting transition, and that asymmetry has a measured
consequence: at this student's competence a failed skill converges to a fixed
point near **0.109** within about three observations, and nothing ever brings
it back up except a success.

**Ability** is the mean over touched skills, and is the `alpha` of section 2.2.

---

## 5. Curriculum — the ZPD band

    target(ability) = 1 + ability * (D_max - 1)        D_max = 5

A problem of difficulty `d` is in the band when

    |d - target| <= 1

with skill mastery acting only as a **ceiling** — demonstrated work is not
re-offered — and explicitly not as a floor. A skill at 0.11 means the learner
is failing it, which is when they should be handed something easier, not
excluded from practice.

The earlier form tested `zpd_low <= mean(mastery) <= zpd_high` and ignored the
difficulty argument it was passed. With `p_init = zpd_low = 0.3`, an untouched
skill sat exactly on the boundary, so the band meant "skills never attempted" —
an exploration frontier that empties permanently once every skill has been
seen. Three runs drew most of their epochs from the top-up path rather than
from a curriculum.

---

## 6. Why the mathematics did not produce learning

The reward has two kinds of term, and they are not equally trainable.

`r_ped` and `L` are **deterministic functions of the tutor's own text**:
identical text always scores identically, and the policy controls them
completely. `r_sol` is a **sampled estimate** of whether a different, frozen,
quantised model writes passing code afterwards.

Only within-group variance becomes gradient (3.1). Measured on retained
rollouts:

    within-group variance:   r_ped 0.0482    leak 0.0144    r_sol 0.0098

The pedagogy term supplies roughly **five times** the usable signal of the term
the system exists to optimise, and 22 of 24 rollouts scored `r_sol` exactly
0.00. Across three completed runs, `corr(epoch, r_ped)` was +0.80 / +0.99 /
+0.64 while `corr(epoch, r_sol)` was +0.48 / +0.54 / +0.01.

A policy optimises whatever part of its reward it can control. The arithmetic
above says that part is not the outcome, which is why every gain in the run
history came from removing a distortion in the controllable channel, and why
the one added behavioural incentive was gamed inside a single run.

Stated plainly: **teaching that causes a student to solve problems was never
trained**, because at `G = 8` with `r_sol` quantised to multiples of `1/K` and
almost always zero, it had no usable gradient to train on.
