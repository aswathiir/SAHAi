# Track A — RL & Reward Design: a from-scratch tutorial

This is a course, not a reference sheet. Every idea builds on the one before it,
every formula gets derived (not just stated), and every formula gets tied to the
exact line of code in this repo that implements it. Read it in order the first
time.

If a symbol clashes with a variable name in the code, that clash is called out
explicitly — it's a real source of confusion and worth being deliberate about.

---

## Part 0 — What problem are we even solving?

You have a tutor (an AI) and a student (another AI, standing in for a real one).
They talk. At the end, the student tries to solve a coding problem on their own.

**The question RL answers: how do you make the tutor better at tutoring, when
nobody has ever written down what a "good hint" looks like?**

There is no labelled dataset of `(situation → correct hint)` pairs — that's a
supervised learning problem, and it doesn't exist here. What *does* exist is a
downstream signal: did the student solve the problem afterward? That's a
reward, not a label, and reward-only learning is exactly what reinforcement
learning is for.

---

## Part 1 — Reinforcement learning, from zero

### 1.1 The five things every RL problem has

| Concept | In general | In SAHAi |
|---|---|---|
| **Agent** | the thing being trained | the tutor (`TutorPolicy`) |
| **Environment** | everything the agent interacts with | the student + the problem |
| **State** | what the agent observes | the conversation so far |
| **Action** | what the agent does | the tutor's next message |
| **Reward** | a number saying how good that was | `r_SAHAI`, computed *after* the whole conversation |

A **trajectory** (also called an episode or rollout) is one full run: state,
action, state, action, ... until it ends. In SAHAi, one trajectory is one
complete tutor↔student dialogue, from the student's opening confusion to
either a termination phrase or the turn limit.

### 1.2 The policy

The agent's behaviour is a **policy**, written $\pi_\theta$. It's a function
that takes a state and outputs a probability distribution over actions:

$$\pi_\theta(a \mid s) = \Pr[\text{action } a \mid \text{state } s]$$

$\theta$ is the policy's parameters — in our case, the tutor model's weights
(or more precisely, the small LoRA adapter on top of them; more on that in
Part 6). "Training the policy" means adjusting $\theta$.

**Code mapping:** `TutorPolicy.generate()` in `sahai/agents/tutor.py` *is*
$\pi_\theta$. Given a state (the dialogue so far, formatted as chat messages),
it samples an action (the next tutor turn) from the distribution the model
defines over token sequences.

### 1.3 Return — why we can't just maximise the reward directly

If a trajectory has rewards $r_1, r_2, \ldots$ at each step, the **return** is
their sum:

$$G = \sum_{t} r_t$$

*(Standard RL notation uses $G$ for return. Our codebase also uses `G` for
**group size** in GRPO — those are unrelated. This document uses $G$ only for
return, matching the literature; when we get to group size we'll write it as
`group_size` explicitly to avoid the clash.)*

In our system, reward doesn't arrive at every turn — it arrives **once**, at
the end, after the student's post-dialogue attempt is graded. So the "sum" is
trivial: the return of a trajectory *is* $r_{SAHAI}$ for that trajectory. This
is what the literature calls an **outcome reward** or **terminal reward**, and
it's a deliberate simplification: we're not trying to say which *specific
turn* in the dialogue was good, only whether the *whole conversation* worked.

### 1.4 The RL objective

The goal is to find parameters $\theta$ that maximise the *expected* return,
averaged over the randomness in the environment and the policy's own sampling:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\,G(\tau)\,]$$

where $\tau$ (tau) denotes a trajectory. Read this as: "sample a dialogue by
running the current tutor against the student; look at the reward it got; do
that many times and average." Training is `argmax_θ J(θ)`.

---

## Part 2 — Policy gradients: how do you improve a policy you can't differentiate through?

### 2.1 The core difficulty

You want $\nabla_\theta J(\theta)$ — the direction in parameter space that
increases expected reward — so you can do gradient ascent. But $J(\theta)$ is
an expectation over *sampled discrete text*. You can't backpropagate through
"the tutor said this token and not that one" the way you can through a
differentiable loss.

### 2.2 The score-function trick (the one idea that makes RL for language models possible)

Here is the trick, derived properly. Start from the definition:

$$J(\theta) = \int \pi_\theta(\tau)\, G(\tau)\, d\tau$$

Take the gradient with respect to $\theta$:

$$\nabla_\theta J(\theta) = \int \nabla_\theta \pi_\theta(\tau)\, G(\tau)\, d\tau$$

Now use the identity $\nabla_\theta \pi_\theta(\tau) = \pi_\theta(\tau)\,
\nabla_\theta \log \pi_\theta(\tau)$ (this is just the chain rule for
logarithms, run backwards: $\nabla \log f = \nabla f / f \Rightarrow \nabla f
= f \nabla \log f$). Substituting:

$$\nabla_\theta J(\theta) = \int \pi_\theta(\tau)\, \nabla_\theta \log \pi_\theta(\tau)\, G(\tau)\, d\tau = \mathbb{E}_{\tau \sim \pi_\theta}\big[\, \nabla_\theta \log \pi_\theta(\tau)\, G(\tau) \,\big]$$

**This is the whole trick.** We turned a gradient of an expectation (which we
can't compute) into an expectation of a gradient (which we *can* estimate, by
sampling). This is the **REINFORCE** estimator (Williams, 1992), and it's the
grandparent of every RL algorithm used to train language models, GRPO
included.

**In words:** sample a trajectory, compute how likely the policy was to
produce it (`log π`), and scale that likelihood's gradient by how good the
outcome was (`G`). If the outcome was good, push $\theta$ to make that
trajectory *more* likely. If bad, push it to make that trajectory *less*
likely. The reward is what tells you which direction "more likely" should go.

### 2.3 Why we subtract a baseline

REINFORCE as written has very high variance — if every reward is a large
positive number, the algorithm pushes up the probability of *every* sampled
trajectory, even the relatively bad ones, just less strongly than the good
ones. You get noisy, slow learning.

The fix: subtract a **baseline** $b$ that doesn't depend on the action taken.
This doesn't change the expected gradient (a constant offset has zero
expected effect — provably, since $\mathbb{E}[\nabla_\theta \log
\pi_\theta(\tau)] = 0$), but it dramatically reduces variance:

$$\nabla_\theta J(\theta) = \mathbb{E}\big[\, \nabla_\theta \log \pi_\theta(\tau)\, (G(\tau) - b) \,\big]$$

$(G(\tau) - b)$ is called the **advantage** — how much better this trajectory
was than some reference expectation. Classic actor-critic methods (and PPO)
learn $b$ with a second neural network (the "critic" or "value function").

**This is the single most important design choice in the whole training
loop, and it's where GRPO diverges from PPO.**

---

## Part 3 — From PPO to GRPO

### 3.1 PPO, briefly (the algorithm we deliberately did *not* use, but which the papers reference)

Proximal Policy Optimization adds two things to the REINFORCE-with-baseline
idea above:

1. **A learned critic** $V_\phi(s)$ estimating expected return from a state,
   used as the baseline $b$.
2. **A clipped objective**, so a single update can't move the policy too far
   from where it collected data (an off-policy correctness safeguard):

$$L^{CLIP}(\theta) = \mathbb{E}\Big[\min\big(r_t(\theta) A_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\, A_t\big)\Big]$$

where $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{old}}(a_t|s_t)$ is the
**importance ratio** between the current and the pre-update policy.

**Note about our code:** this is what `_policy_update` now does. For most of
the project it did not — `clip_epsilon` was defined in `sahai/settings.py` and
used nowhere, and what ran was REINFORCE-with-baseline (the formula from §2.3).
The ratio and the clip were added on 2026-09-13, together with the cache of
pre-update log-probs the ratio is measured against, so "clipped GRPO" is now an
accurate description of the implementation. Results from runs before that date
were produced by the unclipped version; see
[08-team-split.md](08-team-split.md) Track A and
[04-findings.md §14](04-findings.md).

### 3.2 GRPO's idea: replace the critic with group statistics

Training a critic is expensive (a whole second model) and can be unstable.
GRPO's insight (used in DeepSeekMath, and adopted by the RL-for-tutoring
literature this project follows): **if you sample multiple trajectories for
the *same* starting condition, their own mean and standard deviation give you
a baseline for free — no critic needed.**

Concretely: pick a problem, run `group_size` independent dialogues against it
(same problem, different sampled tutor/student text each time — that
randomness comes from `do_sample=True` in `TutorPolicy.generate`). You now
have a *group* of rewards $r_1, \ldots, r_{group\_size}$ for the same problem.
Define the advantage per rollout $i$ as a **z-score within the group**:

$$A_i = \frac{r_i - \bar{r}}{\sigma_r + \varepsilon}, \qquad \bar{r} = \frac{1}{\text{group\_size}}\sum_i r_i,\quad \sigma_r = \sqrt{\frac{1}{\text{group\_size}}\sum_i (r_i - \bar{r})^2}$$

$\varepsilon$ (a tiny constant like $10^{-8}$) exists only to prevent
division by zero.

**Code mapping — exact:**

```python
# sahai/training/grpo.py, _compute_advantages
mean_r = sum(rewards) / len(rewards)
var = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
std_r = var ** 0.5
rollout.advantage = (rollout.reward - mean_r) / (std_r + 1e-8)
```

That's $A_i$, computed literally as written above. `group_size` in the code
is exactly the group size in the formula.

### 3.3 Why this one formula explains almost every training bug we found

**Read the formula again: if every $r_i$ in a group is identical, $\sigma_r =
0$, and every $A_i = 0$.**

Zero advantage means zero gradient (look at §2.3's formula — the gradient is
literally multiplied by the advantage). A group with no reward *variance*
contributes **nothing** to learning, no matter how good or bad the rewards
were in absolute terms.

This is not a hypothetical. It is exactly what happened in the project's first
run: the pedagogy reward was binary (all-or-nothing), every rollout in the
group failed at least one check, every reward became identical
(`hard_penalty` forced them all to `-λ`), and the training log printed
`loss=0.0000` — not as a rounding artifact, but because every single advantage
in that epoch really was zero. See [04-findings.md §1](04-findings.md) for the
full story and the fix (grading the reward instead of binarising it, which
restores variance).

**The practical lesson, stated as a rule:** with a critic-free algorithm like
GRPO, *reward variance within a group is not a nice-to-have — it is the entire
learning signal.* Every reward-shaping decision in this project (grading
`r_ped` instead of thresholding it, execution-verifying leakage instead of a
binary hit/miss) exists partly to protect this variance.

### 3.4 The policy loss

Now that we have $A_i$ per rollout, the loss is the REINFORCE loss from §2.2,
applied per-token and averaged, with a sign flip (we minimise loss, which
means maximise reward):

$$\mathcal{L}_{policy} = -A_i \cdot \frac{1}{|T|}\sum_{t \in T} \log \pi_\theta(a_t \mid s_{<t})$$

where $T$ is the set of token positions that belong to the tutor's own turns
(§5 explains why only those). **Code:**

```python
# sahai/training/grpo.py, _policy_update
token_count = mask.sum().clamp(min=1)
mean_lp = (log_probs * mask).sum() / token_count
policy_loss = -advantage * mean_lp
```

Notice the sign: if `advantage` is positive (this rollout beat its group's
average), `policy_loss` is `-advantage * mean_lp`, and minimising it means
*increasing* `mean_lp` — i.e. making those tutor tokens more likely. If
`advantage` is negative, the opposite. That's the whole mechanism: nudge
toward what beat the group, away from what didn't.

---

## Part 4 — Keeping the policy from drifting: KL regularisation

### 4.1 Why you need it at all

Nothing in the loss above stops the policy from making a huge jump — chasing
one lucky high-reward rollout by drastically rewriting its behaviour, possibly
breaking things the reward function doesn't measure (like basic coherence, or
— critically for us — general coding ability learned during pretraining). We
need a leash back to a known-good policy.

### 4.2 KL divergence

The **Kullback-Leibler divergence** measures how different two probability
distributions are:

$$D_{KL}(P \parallel Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

It's always $\geq 0$, and $= 0$ only when $P$ and $Q$ are identical. We use it
to penalise the *current* policy $\pi_\theta$ for straying from a fixed
**reference policy** $\pi_{ref}$ — the policy before this training run
started:

$$\mathcal{L}_{KL} = \beta \cdot D_{KL}(\pi_\theta \parallel \pi_{ref})$$

$\beta$ is `kl_coeff` in the settings (`0.05`).

### 4.3 What the reference policy actually is here — no second model needed

Naively, $\pi_{ref}$ requires a second copy of the whole model, frozen. That
doubles GPU memory. Because the tutor is trained with **LoRA** (Part 6), there
is a much cheaper trick available: *the reference policy is just the same
model with the adapter switched off.*

```python
# sahai/training/grpo.py, _ref_log_probs
self.tutor.model.disable_adapter_layers()
with torch.no_grad():
    log_probs, mask = self.tutor.compute_log_probs(dialogue, problem)
self.tutor.model.enable_adapter_layers()
```

This is elegant and worth pausing on: because LoRA only *adds* a small delta
on top of frozen base weights (Part 6), turning the adapter off doesn't
approximate the pre-training policy — it *is* the pre-training policy, exactly,
by construction.

### 4.4 The estimator we use is not quite a real KL — and why that matters

$D_{KL}$ as defined above needs a sum over *all* possible sequences, which is
intractable. In practice you estimate it from samples. The code computes:

```python
kl = ((log_probs - ref_log_probs) * mask).sum() / token_count
```

This is $\mathbb{E}[\log \pi_\theta - \log \pi_{ref}]$ — call this estimator
**k1**. It's an unbiased estimator of the KL divergence *in expectation*, but
any single sample of it **can be negative**, even though true KL divergence
never is. That's not a bug in the code — it's a known property of this
estimator — but it means the printed `kl` value in the training logs is not
literally interpretable as "how far the policy has drifted," only as a proxy
that's correct on average over many steps.

A better single-sample estimator exists, called **k3**:

$$\widehat{D}_{KL}^{(k3)} = \mathbb{E}\big[\, e^{\delta} - \delta - 1 \,\big], \qquad \delta = \log\pi_{ref} - \log\pi_\theta$$

This is always $\geq 0$ for any sample (it's built from $f(x) = e^x - x - 1$,
which is convex and minimised at $x=0$ where $f=0$ — a standard trick from
the RL-from-human-feedback literature). Replacing k1 with k3 is on the
roadmap; see [06-roadmap.md](06-roadmap.md) Track A.

### 4.5 The full loss

$$\mathcal{L} = \mathcal{L}_{policy} + \beta\, \mathcal{L}_{KL}$$

```python
loss = (policy_loss + kl_loss) / accum_steps
loss.backward()
```

(`accum_steps` is gradient accumulation — a memory-management detail, not a
mathematical one: it divides the loss so that summing gradients over several
mini-batches before stepping the optimiser behaves like one larger batch.)

---

## Part 5 — Why only the tutor's tokens get a gradient

### 5.1 The masking rule

A dialogue is a sequence of tutor and student turns. Only the **tutor's**
tokens should receive gradient — the student is a fixed simulator, not the
thing being trained, and its tokens are not "actions" of the policy $\pi_\theta$
at all.

```python
# sahai/agents/tutor.py, _build_tutor_mask
mask = torch.zeros_like(input_ids, dtype=torch.float)
for i in range(1, len(messages) + 1):
    ...
    if messages[i - 1]["role"] == "assistant":
        mask[0, prev_len:curr_len] = 1.0
```

It re-renders the chat template up through each message prefix, finds where
each assistant (tutor) span starts and ends in token-space, and marks only
those positions. Every log-prob term in §3.4 and §4.4 is multiplied by this
mask before summing.

### 5.2 The consequence that shapes the whole codebase

This masking rule is *why* the student can be freely edited after generation
(trimmed, scripted, cleaned up) while the tutor's raw output must never be
touched before scoring.

If you edited the tutor's text — say, stripping a code block out before
computing `log_probs` — you would be computing $\log \pi_\theta(a_t)$ for
tokens $a_t$ that the model **never actually sampled**. The policy gradient
theorem in §2.2 is only valid for the trajectory the policy *actually
produced*; scoring an edited trajectory is off-policy in an uncontrolled way,
and the gradient estimate becomes meaningless.

This is exactly the bug found and fixed early in the project: `_strip_code`
used to be applied to the trajectory before scoring, meaning GRPO was
computing gradients on hallucinated tokens. The fix — keep the raw generation
in the trajectory, do any cleanup only for *display* — is a direct consequence
of taking §2.2 seriously. See [04-findings.md](04-findings.md).

---

## Part 6 — LoRA: why fine-tuning doesn't forget

### 6.1 The problem with full fine-tuning

If you update all of a language model's weights with RL, two things can go
wrong: it's extremely expensive (optimiser state for billions of parameters),
and the model can catastrophically forget general capabilities while
overfitting to the narrow reward signal.

### 6.2 The low-rank idea

For a weight matrix $W \in \mathbb{R}^{d \times k}$, LoRA freezes $W$
completely and adds a trainable **low-rank update**:

$$W' = W + \Delta W = W + BA, \qquad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k}$$

with rank $r \ll \min(d, k)$. The number of trainable parameters in $\Delta W$
is $r(d+k)$ instead of $dk$ — for $r=8$ or $16$ this is a tiny fraction of the
original matrix.

**Code:** `sahai/settings.py`

```python
lora_rank: int = 16        # r, above (8 on the Kaggle config)
lora_alpha: int = 32       # scaling factor, below (16 on Kaggle)
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

Only the attention projection matrices get a $\Delta W$; everything else in
the model (MLP layers, embeddings, layer norms) is entirely frozen —
`p.requires_grad_(False)` on the base model in `sahai/core/models.py`. In
practice this touches roughly 0.1–1% of the model's parameters.

The **alpha** parameter scales the update's effective magnitude: the applied
delta is $\frac{\alpha}{r} BA$, not $BA$ directly. This lets you change the
update's strength without changing its rank.

### 6.3 Why this is the project's anti-forgetting mechanism

Two independent things protect general capability, and it's worth being
precise about which is which:

1. **LoRA structurally limits *what* can change.** The frozen base $W$ never
   moves. Whatever knowledge is encoded there — general code reasoning,
   English/Hindi/Tamil fluency, etc. — is mathematically untouched. Only a
   thin, low-rank correction is learned.
2. **The KL term (Part 4) limits *how far* the whole output distribution can
   move**, independent of where the drift comes from.

They're complementary, not redundant: LoRA restricts the *hypothesis space*
of possible updates; KL restricts *how much* of that space gets used.

**What's still missing**, stated honestly: nothing in the project currently
*measures* whether these mechanisms worked. The retention benchmark — running
held-out MBPP pass@1 with the adapter on vs. off, before vs. after training —
is the natural way to turn "LoRA + KL should prevent forgetting" into an
actual number. It's on the roadmap ([06-roadmap.md](06-roadmap.md), Track D)
and hasn't been run yet.

---

## Part 7 — The reward, rebuilt from the ground up

### 7.1 Reward shaping — why r_SAHAI has the shape it does

$$r_{SAHAI} = \underbrace{(r_{sol} - \alpha)}_{\text{did they learn, beyond baseline}} + \underbrace{(r_{ped}-1)\cdot\lambda}_{\text{teaching-quality penalty}} - \underbrace{\gamma \cdot L}_{\text{leakage penalty}}$$

*(Notation clash to flag explicitly: this $\gamma$ — `gamma_leak` in the code —
is unrelated to a discount factor. RL problems with a terminal-only reward, as
ours is, don't need temporal discounting at all; there's only one reward per
trajectory, so there's nothing to discount between. `gamma_leak` is purely a
penalty weight, chosen to echo the paper's $\gamma$ notation for the same
term.)*

Each term is designed to be **zero-centred or negative when nothing good
happened**, which matters for GRPO specifically: if the sign of the reward
were always positive, the group mean $\bar r$ would rarely get crossed, and
half the group would be reliably above-average regardless of actual quality —
weakening the "beat the group" signal.

### 7.2 $r_{sol}$ — the outcome reward, Monte Carlo estimated

$$r_{sol} = \frac{1}{K} \sum_{k=1}^{K} \mathbb{1}\!\left[\hat{s}^{(k)} = s\right]$$

This was the implementation until 2026-09-13, with $K=4$ on Kaggle against the
paper's $K\geq 8$. It is no longer what runs, and the reason is worth
understanding because it is the central result of the project so far.

A $K$-sample Monte Carlo estimate carries sampling sd $\sqrt{p(1-p)/K}$. At
$p=0.15$ and $K=4$ that is $0.18$ — **larger than any plausible difference the
tutor could make to $p$**. GRPO's advantage is a z-score of the spread *within*
a group of rollouts on the same problem, so that sampling noise went straight
into the gradient: measured within-group variance was $0.0098$ for $r_{sol}$
against $0.0482$ for the deterministic pedagogy term. The term the project
exists to optimise was contributing a fifth of the usable signal, and the policy
did the rational thing and optimised the rule-based terms instead.

What runs now is a single **greedy** attempt scored by *fraction of test cases
passed*:

$$r_{sol} = \frac{\#\{\text{tests passed}\}}{\#\{\text{tests}\}}, \qquad \text{decoded with } \texttt{do\_sample=False}$$

Deterministic given the dialogue, so all of its within-group spread is
attributable to the tutor; continuous, so a near-miss is distinguishable from
nonsense. The all-or-nothing rate is still reported alongside it as `solved`,
because that is the series every earlier run is comparable on. Raising $K$ is
therefore moot rather than outstanding — it would mean restoring the sampling
that caused the problem. See [04-findings.md §14](04-findings.md).

**Why subtract $\alpha$ (the traced ability):** without this term, a tutor
gets full credit for a student who could already solve the problem
unassisted. $\alpha$ comes from the Bayesian Knowledge Tracer (Part 8) — an
independent estimate of the student's pre-existing skill — so $(r_{sol} -
\alpha)$ approximates the *tutoring's marginal contribution*, not raw success.

### 7.3 $r_{ped}$ — why grading beats thresholding (tying back to §3.3)

Five checks, graded as `passed/5`. This *is* Finding #1 from §3.3, restated
as a reward-design principle: **any binary/thresholded term inside a
GRPO reward is a variance-collapse risk.** Grading isn't a stylistic
preference — it's structurally required by the algorithm in Part 3.

### 7.4 $L$ — leakage, and why it had to become an execution check

Originally $L$ was pure token overlap between tutor text and the reference
solution. Measured on a real rollout, a tutor that wrote a **complete,
correct** solution using a different approach than the reference (a dict
instead of `enumerate`/`count`) scored $L = 0.000$ — the metric was measuring
*textual similarity to one specific implementation*, not "did the tutor give
away the answer."

The fix makes $L$ **execution-verified**: extract any code the tutor wrote,
run it against the problem's actual test cases (via the same sandboxed
`CodeVerifier` used for $r_{sol}$), and if it passes, $L = 1.0$ by definition
— regardless of variable names or algorithm choice. This is the same
philosophy as $r_{sol}$ itself: **trust execution, not resemblance.** Full
story, including the function-name-aliasing edge case this fix needed, in
[04-findings.md §2](04-findings.md).

---

## Part 8 — Bayesian Knowledge Tracing, derived

### 8.1 The problem BKT solves

We want to know: *does this student know this skill?* We can't observe that
directly (it's a hidden/latent state) — we only observe whether they answered
correctly. BKT is a **hidden Markov model** with one binary hidden state per
skill (known / not known) and a very small, interpretable parameter set.

### 8.2 Bayes' theorem, the one line everything is built from

$$\Pr[\text{hidden} \mid \text{observed}] = \frac{\Pr[\text{observed} \mid \text{hidden}]\, \Pr[\text{hidden}]}{\Pr[\text{observed}]}$$

Let $L$ denote "the student has learned this skill" (its probability is what
we're tracking) and let the observation be "correct" or "incorrect" on one
attempt.

### 8.3 The four parameters

| Symbol | Code | Meaning |
|---|---|---|
| $p(L_0)$ | `p_init = 0.3` | prior probability of already knowing the skill |
| $p(T)$ | `p_learn = 0.1` | probability of transitioning not-known → known after one attempt |
| $p(G)$ | `p_guess = 0.2` | probability of a correct answer *despite* not knowing (a guess) |
| $p(S)$ | `p_slip = 0.1` | probability of an incorrect answer *despite* knowing (a slip) |

### 8.4 Deriving the posterior update after a *correct* answer

We want $\Pr[L \mid \text{correct}]$. Apply Bayes' theorem: the denominator
$\Pr[\text{correct}]$ expands by total probability over both hidden states
(known, not known):

$$\Pr[\text{correct}] = \underbrace{\Pr[L]\cdot(1-p_S)}_{\text{knew it, didn't slip}} + \underbrace{(1-\Pr[L])\cdot p_G}_{\text{didn't know it, guessed right}}$$

and the numerator is just the first term (we want the "knew it" branch of that
same split):

$$\Pr[L \mid \text{correct}] = \frac{\Pr[L]\,(1-p_S)}{\Pr[L]\,(1-p_S) + (1-\Pr[L])\,p_G}$$

**Code, exact match:**

```python
p_correct = p * (1 - p_slip) + (1 - p) * p_guess
posterior = (p * (1 - p_slip)) / p_correct if p_correct > 0 else p
```

### 8.5 The symmetric derivation for an *incorrect* answer

Same structure, swapping which branch represents "observed":

$$\Pr[L \mid \text{incorrect}] = \frac{\Pr[L]\, p_S}{\Pr[L]\, p_S + (1-\Pr[L])\,(1-p_G)}$$

```python
p_incorrect = p * p_slip + (1 - p) * (1 - p_guess)
posterior = (p * p_slip) / p_incorrect if p_incorrect > 0 else p
```

### 8.6 Applying the learning transition

The Bayesian update above only accounts for *evidence* — it doesn't yet model
the fact that attempting the problem might itself have taught the student
something. That's a separate step, applied after the posterior above,
regardless of whether the answer was right or wrong:

$$\Pr[L]_{\text{new}} = \Pr[L \mid \text{obs}] + \big(1 - \Pr[L \mid \text{obs}]\big)\, p_T$$

```python
self.skills[skill] = posterior + (1 - posterior) * p_learn
```

Read this as: "whatever probability mass wasn't already 'known', a fraction
$p_T$ of it converts to known, this round." This term is what makes mastery
trend upward over repeated correct answers even beyond what pure evidence
would justify — it's modelling learning-by-doing, not just measurement.

### 8.7 Why BKT and not the DKT paper's neural approach

The literature review recommends Deep Knowledge Tracing — an LSTM over the
full interaction sequence — as an alternative. We use BKT instead, and it's a
data-scale decision, not a claim BKT is more accurate:

- DKT needs large interaction *sequences* to train on (its own paper trains
  on Khan Academy / ASSISTments-scale logs). Our student is synthetic and
  every training session starts cold — there is no such corpus yet.
- BKT produces a single interpretable scalar per skill, immediately, from the
  very first observation — which is exactly what $\alpha$ in §7.2 and the ZPD
  curriculum (§8.8) both need.
- The project **is** collecting the interaction-sequence data DKT would need
  (`services/tracer`'s append-only `Observation` log — see
  [03-internals.md §7](03-internals.md)), specifically so DKT can be trained
  later without having thrown the history away. This is Track C's roadmap
  item, not abandoned — just sequenced after enough data exists.

### 8.8 What mastery is used for: the curriculum

$\Pr[L]$ per skill also drives **ZPD sampling** — Vygotsky's "Zone of Proximal
Development," operationalised as a simple mastery-band filter:

```python
def in_zpd(self, difficulty, skills, max_difficulty=5):
    if skills and mean(self.get_mastery(s) for s in skills) > zpd_high:
        return False                      # already demonstrated
    target = 1.0 + self.get_ability() * (max_difficulty - 1)
    return abs(difficulty - target) <= 1.0
```

**This is not what the code did until 2026-09-13**, and the old version is
instructive. It was a pure mastery band — `zpd_low <= avg <= zpd_high`, with
the `difficulty` argument accepted and thrown away. Two facts then interact:
`p_init` is 0.3 and `zpd_low` is 0.3, so an *untouched* skill sat exactly on the
boundary and passed, which made the band mean "skills never attempted"; and BKT
has no forgetting transition, so at this student's competence a failed skill
converges to $\approx 0.109$ within three observations and can never return. The
band held the whole bank at epoch 0 and was empty from epoch 3 onward.

Problems are drawn from within one difficulty level of the learner's target —
not already-mastered (mastery $> 0.7$, no learning signal left), and not far
beyond reach (difficulty well above $1 + \alpha(\text{max}-1)$).

Note there is deliberately no mastery *floor* any more. A skill at $0.11$ means
the learner is failing it, which is exactly when they should be handed something
easier — the job difficulty targeting now does — rather than excluded from that
skill's problems permanently.

---

## Part 9 — The four papers, extracted

### Paper 1 — *From Problem-Solving to Teaching Problem-Solving*

**What it contributes mathematically:** the entire RL framing in Part 1–3 —
online RL on simulated tutor↔student dialogue, terminal (not per-turn) reward,
$r_{sol}$ as a $K$-sample Monte Carlo solve-rate estimator (§7.2), the hard
penalty $r_{ped}=0 \Rightarrow r = -\lambda$.

**What it explicitly argues against, and why we agree:** supervised
fine-tuning on human tutoring transcripts. Its argument: too few high-quality
tutoring datasets exist. We take this further than the paper does — we use
*zero* tutoring transcripts anywhere; the only supervision is the student's
verified downstream success.

**Where we deviated, with the derivation for why:** §3.3's binary-vs-graded
finding. The paper's PPO has a critic to absorb constant rewards; our GRPO
does not, so its exact reward formula had to be graded to survive contact
with a critic-free algorithm.

### Paper 2 — *Simulating Students with LLMs*

**What it contributes:** the architecture recommendation realised in
`sahai/agents/student.py` + `sahai/agents/tracer.py` — "LLM generates
language, a separate tracer updates mastery" — plus its "advanced" tier
(injected misconceptions per skill, `MISCONCEPTIONS` dict) and the leakage
concept itself (§7.4).

**Where we deviated:** we execution-verify leakage rather than using pure
token overlap, for the reason derived in §7.4.

### Paper 3 — *Why LLMs alone fall short for learner modelling*

**What it contributes:** the case *against* using an LLM's own judgment for
three separate things in this system, each independently: student ability
$\alpha$ (comes from BKT, Part 8, never from asking a model), solve
correctness (comes from execution, §7.2, never from a judge), and — as a
design principle — treating pedagogical acceptance as secondary to
execution-verified outcomes wherever the two could conflict.

### Paper 4 — *Deep Knowledge Tracing*

**What it contributes:** the alternative knowledge-tracing paradigm discussed
in §8.7, and — practically — the reason the observation log exists *before*
DKT does: the paper's method needs sequence data that can't be reconstructed
after the fact, so collection had to start immediately, even while BKT is
what's actually driving the reward.

---

## Part 10 — One rollout, every formula, real numbers

Pulling a real epoch-0 rollout end to end, so every symbol above has a number
attached to it.

**Given:** `problem = "first repeated character"`, `group_size = 8`. This
rollout's raw scores: $r_{sol}=0.00$, $\alpha = 0.30$ (traced ability at the
time), $r_{ped} = 0.40$ (scored under the 5-check judge of the time; there are
3 checks now — see §4), $L = 0.14$.

**Step 1 — combine into $r_{SAHAI}$** (Part 7, $\lambda=1.0$, $\gamma=0.5$):

$$r_{SAHAI} = (0.00 - 0.30) + (0.40 - 1)(1.0) - 0.5(0.14) = -0.30 - 0.60 - 0.07 = -0.97$$

**Step 2 — this rollout sits in a group of 8.** Suppose the group's 8 rewards
have mean $\bar r = -0.85$ and std $\sigma_r = 0.20$ (illustrative — real
group stats, but not this exact rollout's actual group). Its advantage (§3.2):

$$A = \frac{-0.97 - (-0.85)}{0.20 + 10^{-8}} \approx -0.60$$

Negative: this rollout underperformed its group's average, so the gradient
step will make its specific tutor tokens *less* likely.

**Step 3 — policy loss** (§3.4). Say the mean log-prob over this rollout's
tutor tokens is $\overline{\log\pi_\theta} = -2.1$ (a small negative number is
typical — probabilities are always $\leq 1$, so $\log$ is always $\leq 0$):

$$\mathcal{L}_{policy} = -A \cdot \overline{\log\pi_\theta} = -(-0.60)(-2.1) = -1.26$$

**Step 4 — add the KL term** (§4.5). If $\delta_{KL} \approx 0.002$ (policy has
barely moved from reference yet, typical early in training) and
$\beta=0.05$:

$$\mathcal{L} = -1.26 + 0.05 \times 0.002 = -1.2599$$

That total is what `loss.backward()` differentiates. This one rollout's
contribution nudges $\theta$ (the LoRA adapter's $B, A$ matrices, Part 6) to
make its below-average tutor turns less probable next time — while the KL
term simultaneously discourages the adapter from moving so far that it stops
resembling the frozen base model at all.

---

## Where to go next

- [03-internals.md](03-internals.md) — the same material, reference-style,
  without the derivations.
- [04-findings.md](04-findings.md) — every bug this math predicts, found in
  practice, with measurements.
- [06-roadmap.md](06-roadmap.md), Track A — importance ratio + clipping
  (finishing §3.1), the k3 KL estimator (§4.4), ACE causal probes, the
  retention benchmark (§6.3).
- The original papers, for anything this document simplified — always prefer
  the primary source for anything you plan to cite directly.
