# The SAHAi Guide — Paper → Formula → Architecture → How It Works

This is the guide. Every idea in it follows the same four steps, in this
order, every time:

1. **What the paper says** — the idea, in plain words, from the literature
   this project is built on.
2. **The formula** — every symbol defined before it's used. No skipped steps.
3. **What we built** — the exact file, class, and function. Quoted, not
   paraphrased.
4. **How it works** — a small worked example with real numbers, so you can
   see the formula actually run.

Nothing in here that isn't in the code gets derived in depth — if a technique
is only *planned*, it's named once at the end and left there, so it doesn't
compete for attention with what's real.

---

## 1. The problem this whole project solves

### 1.1 What the paper says

The founding paper for this project (*From Problem-Solving to Teaching
Problem-Solving*) makes one argument that shapes everything downstream: you
cannot train a good AI tutor by showing it examples of "good tutoring,"
because almost no such examples exist. A handful of datasets try
(MathDial, SocraticLM, TutorChat) but they're small, narrow, and expensive to
make.

Their alternative: **don't teach the tutor what a good hint looks like.
Teach it to discover that, by letting it practise on a simulated student and
rewarding it only for the *outcome* — did the student solve the problem
afterward?**

That's the whole idea. No labelled examples. Just: try, see what happened,
adjust.

### 1.2 The vocabulary this idea needs

Five words, defined once, used everywhere after this:

| Word | Meaning here |
|---|---|
| **Agent** | the thing being trained — the tutor |
| **Environment** | everything the agent reacts to — the student, the problem |
| **State** | what the agent currently sees — the conversation so far |
| **Action** | what the agent does — the tutor's next message |
| **Reward** | one number saying how good the *whole* attempt was |

A full attempt — start of the conversation to the end — is called a
**trajectory**, or a **rollout**. That's the unit everything else in this
guide is built from.

### 1.3 What we built

- `sahai/agents/tutor.py` → `TutorPolicy` — this **is** the agent. Its
  `.generate()` method takes a state (the conversation so far) and produces
  an action (the tutor's next line).
- `sahai/core/dialogue.py` → `DialogueEngine.run()` — this runs one whole
  trajectory: it alternates tutor and student turns until the student says a
  termination phrase or a turn limit is hit, and returns a `Dialogue` object.

```python
# sahai/core/dialogue.py
@dataclass
class Turn:
    role: str
    content: str
    complete: bool = True

@dataclass
class Dialogue:
    problem_id: str
    turns: list[Turn] = field(default_factory=list)
```

A `Dialogue` is just a list of `Turn`s. That's it — that's what a
"trajectory" actually *is* in this codebase, concretely.

### 1.4 How it works

`DialogueEngine.run(tutor, student, problem)`:

1. `student.opening_turn(problem)` — a scripted first message, e.g. *"Kya
   hum strings logic use kar sakte hain yahan pe?"*
2. `tutor.generate(dialogue, problem)` — the tutor replies. This turn gets
   appended.
3. `student.respond(dialogue, problem)` — the student replies. Appended.
4. Repeat 2–3 until the student's message contains a phrase like *"I think I
   can solve it now"*, or the turn count hits the limit.

The result is one `Dialogue` with, say, 6 turns in it. **Nothing has been
scored yet.** Scoring is the next section.

---

## 2. The reward — the number that drives all learning

### 2.1 What the paper says

Two papers combine here. Paper 1 defines a reward with a **solve** term and a
**pedagogy** term, plus a hard rule: if the teaching was unacceptable, the
whole reward collapses to a fixed penalty regardless of anything else. Paper
2 adds a **leakage** term — a penalty for the tutor giving the answer away.

### 2.2 The formula

$$r_{SAHAI} = (r_{sol} - \alpha) + (r_{ped} - 1)\cdot\lambda - \gamma \cdot L$$

Every symbol, defined once:

| Symbol | Meaning | Where it comes from |
|---|---|---|
| $r_{sol}$ | did the student solve it afterward? (0 to 1) | §3 |
| $\alpha$ | the student's *pre-existing* skill at this topic (0 to 1) | §6, the BKT tracer |
| $r_{ped}$ | was the teaching good? (0 to 1) | §4 |
| $L$ | did the tutor leak the answer? (0 to 1) | §5 |
| $\lambda$ | how much the pedagogy term matters | a setting, `= 1.0` |
| $\gamma$ | how much the leakage term matters | a setting, `= 0.5` |

**Why subtract $\alpha$:** if a student could already solve the problem
without any help, the tutor shouldn't get credit for that. Subtracting the
student's prior skill means the reward reflects the *tutoring's actual
contribution*, not the student's luck or prior knowledge.

There's also a **hard rule**, applied before the formula above: if $r_{ped} =
0$ exactly, forget the formula — the reward is just $-\lambda$. Total
failure at teaching overrides everything else.

### 2.3 What we built

```python
# sahai/reward/combined.py
def compute(self, r_sol, r_ped, leakage, ability_baseline):
    if self.hard_penalty and r_ped == 0.0:
        r_sahai = -self.lambda_ped
    else:
        r_sahai = (
            (r_sol - ability_baseline)
            + (r_ped - 1) * self.lambda_ped
            - self.gamma_leak * leakage
        )
```

Read this next to the formula above — it's the same thing, line for line.
`ability_baseline` in the code is $\alpha$ in the formula.

### 2.4 How it works — one real number, computed both ways

From an actual training rollout: $r_{sol}=0.00$, $\alpha=0.30$, $r_{ped}=0.40$,
$L=0.14$.

$$r_{SAHAI} = (0.00 - 0.30) + (0.40 - 1)(1.0) - (0.5)(0.14) = -0.30 - 0.60 - 0.07 = -0.97$$

$r_{ped}=0.40$ is not zero, so the hard rule doesn't fire — we use the full
formula, and land on $-0.97$. I ran this same arithmetic through Python
independently to confirm it before writing it down here; it matches.

Now — where do those three input numbers ($r_{sol}, r_{ped}, L$) actually
come from? That's the next three sections, one at a time.

---

## 3. $r_{sol}$ — did the student solve it?

### 3.1 What the paper says

You can't ask a language model "did the tutor teach well?" and trust the
answer — that's an opinion, and opinions can be gamed. The paper insists on a
**verifiable** signal instead: after the conversation, have the student
*actually attempt the problem*, and check the code really runs and passes.

### 3.2 The formula

$$r_{sol} = \frac{1}{K}\sum_{k=1}^{K} \mathbb{1}[\hat{s}^{(k)} = s]$$

In words: have the student attempt the problem $K$ separate times (each
attempt independently sampled — the student might write something slightly
different each time). $\hat{s}^{(k)}$ is what the $k$-th attempt produced;
$s$ is the correct answer. $\mathbb{1}[\cdot]$ is 1 if the attempt matches, 0
if not. Average over all $K$ attempts.

### 3.3 What we built

```python
# sahai/reward/solve.py
class SolveReward:
    def compute(self, student, dialogue, problem):
        passed = 0
        for _ in range(self.num_samples):          # K
            code = student.attempt_solution(dialogue, problem)
            score = self.verifier.verify(code, problem)
            if score == 1.0:
                passed += 1
        return passed / self.num_samples
```

`self.verifier` is `CodeVerifier` — it doesn't *guess* whether code is
correct. It runs the code, in an isolated subprocess, against the problem's
actual unit tests, and checks the output matches. This is "verifiable" made
literal: nothing here is a language model's opinion.

### 3.4 How it works

$K=4$ on our Kaggle config. Say the student attempts the problem 4 times:

```
attempt 1: def f(s): return s[0]              → runs, wrong answer   → fail
attempt 2: def f(s): seen=set(); ...          → runs, correct answer → pass
attempt 3: def f(s): seen=set(); ...          → runs, correct answer → pass
attempt 4: def f(s): (syntax error)           → doesn't run          → fail
```

$$r_{sol} = \frac{2}{4} = 0.50$$

Two of four independent attempts passed. That's the whole computation — no
hidden steps.

---

## 4. $r_{ped}$ — was the teaching good?

### 4.1 What the paper says

The paper's version: have several separate "judge" models read the
conversation and vote ACCEPT or REJECT against a rubric (no code shown, hints
build understanding, encourages independent thought, etc.). The dialogue
passes only if **every** judge says ACCEPT.

### 4.2 The formula — and the one place we deliberately changed the paper

**We do not use the paper's all-or-nothing rule.** We grade instead:

$$r_{ped} = \frac{\text{number of checks passed}}{5}$$

Why we changed it is covered properly in §7, once you've seen how the reward
is actually *used* for learning — for now, just take the grading as given and
move on; it will make complete sense in a few sections.

### 4.3 What we built

Five checks, all applied only to the *tutor's* turns:

```python
checks = [
    self._no_code_blocks(tutor_turns),        # no ```code``` blocks
    self._no_solution_patterns(tutor_turns),  # no "def f(", "return [", etc.
    self._no_dangling_promises(tutor_turns),  # no tutor turn ending in ":"
]
return sum(checks) / len(checks)

# A question check and a 200-word length check used to sit here. Both were
# removed: they scored teaching *style*, and every style rule tried so far was
# gamed within one run. See docs/04-findings.md entry 11.
```

(The `:` check exists because a turn like *"Here is the implementation:"*
promises code it never delivers — and it turned out that this makes the
*next* speaker finish the sentence, which is how tutor and student roles
started swapping in early runs. Caught by inspecting real transcripts, fixed
by adding the check.)

### 4.4 How it works

A real (simplified) tutor turn: *"What data structure lets you check if
you've seen a character before?"*

```
no code blocks?          PASS  (no ``` anywhere)
no solution patterns?    PASS  (no "def", "return [", etc.)
asks a question?         PASS  (contains "?")
reasonable length?       PASS  (well under 200 words)
no dangling promise?     PASS  (ends in "?", not ":")
```

$$r_{ped} = \frac{5}{5} = 1.0$$

Compare to a bad one — I ran this exact text through the real checker rather
than predicting the result by eye: *"Here's how you do it:\n```python\ndef
f(s): return None\n```"*

```
no code blocks?          FAIL
no solution patterns?    FAIL
asks a question?         FAIL
reasonable length?       PASS
no dangling promise?     PASS  ← the colon is in the middle, not the end
```

$$r_{ped} = \frac{2}{5} = 0.4$$

That `PASS` on the last check is worth pausing on — it's not a mistake, it's
the check doing exactly what it was written to do. `no_dangling_promises`
only looks at the *very last character* of the whole turn (`.rstrip().
endswith(":")`), and this turn ends on `` ``` ``, not `:` — the colon sits in
the middle, before the code block. A turn that genuinely triggers this check
looks like *"Let's dive into the implementation:"* with nothing after it —
that promise really is left unfulfilled by the end of the turn.

---

## 5. $L$ — did the tutor leak the answer?

### 5.1 What the paper says

Paper 2's suggestion: compare the *words* the tutor used against the words
in the reference solution. Lots of overlap → probably leaked.

### 5.2 The formula — and why we had to go further than the paper

Word overlap has a real failure: a tutor can write a **complete, correct**
solution using different variable names or a different algorithm than the
reference, and share almost no words with it. We measured this on a real
rollout — a tutor's fully-working solution scored $L=0.000$ under pure word
overlap. That's backwards; it should have scored the maximum.

So the actual rule, in order:

$$
L = \begin{cases}
1.0 & \text{tutor's code, when run, passes the problem's tests} \\
\max(0.5,\ \text{word overlap}) & \text{tutor wrote code, but it doesn't pass} \\
\text{word overlap} & \text{tutor wrote no code at all}
\end{cases}
$$

Same principle as $r_{sol}$: **trust execution over resemblance.** If the
tutor's code actually solves the problem when run, that is unambiguously a
complete leak, no matter what it's called or how it's written.

### 5.3 What we built

```python
# sahai/reward/leakage.py
def estimate(self, dialogue, solution, problem_text="", tutor_code_solves=False, ...):
    if tutor_code_solves:
        return 1.0
    score = self.token_match_leakage(dialogue, solution, problem_text)
    if self.extract_tutor_code(dialogue):
        score = max(score, 0.5)
    return min(score, 1.0)
```

`tutor_code_solves` is computed by the trainer — it pulls any code block out
of the tutor's turns and actually runs it through the same `CodeVerifier`
used for $r_{sol}$:

```python
# sahai/training/grpo.py
def _tutor_code_solves(self, rollout):
    for block in self.leakage_estimator.extract_tutor_code(rollout.dialogue):
        for variant in self.leakage_estimator.runnable_variants(block, rollout.problem.function_name):
            if self.verifier.verify(variant, rollout.problem) == 1.0:
                return True
    return False
```

(`runnable_variants` handles one real gotcha: if the tutor's code defines a
function with a different name than the problem expects, it gets aliased so
the verifier can still call it. A tutor renaming a variable doesn't make a
leak less of a leak.)

### 5.4 How it works

Tutor writes a complete, correct solution (different variable names than the
reference). Run it → it passes both test cases.

$$L = 1.0 \quad (\text{tutor\_code\_solves} = \text{True, formula short-circuits here})$$

Same tutor code, but this time it has a bug and fails the tests. Say word
overlap with the reference happens to be 0.3:

$$L = \max(0.5,\ 0.3) = 0.5$$

Tutor never writes any code, just asks questions. Word overlap (after
removing common problem-statement words) is 0.0:

$$L = 0.0$$

---

## 6. $\alpha$ — how do we know what the student already knows?

### 6.1 What the paper says

One paper in the review (*Why LLMs alone fall short for learner modelling*)
makes a specific warning: **do not ask a language model to estimate a
student's skill level.** LLM-judged ability is unstable and biased by
surface wording, not the student's real knowledge. Use something
interpretable instead.

A separate paper (*Deep Knowledge Tracing*) proposes tracking skill with a
neural network (an LSTM) over the student's whole history of attempts. We
looked at this and chose something simpler — the reason is in §6.5.

### 6.2 The formula — Bayes' theorem, the one line everything below is built from

$$\Pr[\text{hidden} \mid \text{observed}] = \frac{\Pr[\text{observed}\mid\text{hidden}]\cdot\Pr[\text{hidden}]}{\Pr[\text{observed}]}$$

We want to track one hidden thing: *does the student know this skill?* We
can't see that directly — we only see whether they got a problem right or
wrong. That's exactly the shape Bayes' theorem is for: turn an observation
into an updated belief about something you can't observe directly.

Call $p = \Pr[\text{student knows the skill}]$. There are four small numbers
that control how the belief updates:

| Symbol | Code name | Meaning | Default |
|---|---|---|---|
| $p_{init}$ | `p_init` | starting belief, before any evidence | 0.3 |
| $p_T$ | `p_learn` | chance that *attempting* the problem teaches them something | 0.1 |
| $p_G$ | `p_guess` | chance of a correct answer *despite not knowing* (a lucky guess) | 0.2 |
| $p_S$ | `p_slip` | chance of a wrong answer *despite knowing* (a careless mistake) | 0.1 |

**After a correct answer:**

$$p_{\text{new}} = \frac{p\,(1-p_S)}{p\,(1-p_S) + (1-p)\,p_G}$$

**After an incorrect answer:**

$$p_{\text{new}} = \frac{p\,p_S}{p\,p_S + (1-p)\,(1-p_G)}$$

**Then, either way, apply the "attempting it teaches you something" step:**

$$p_{\text{final}} = p_{\text{new}} + (1-p_{\text{new}})\cdot p_T$$

### 6.3 What we built

```python
# sahai/agents/tracer.py
def update(self, skill, correct):
    p = self.skills[skill]
    if correct:
        p_correct = p * (1 - p_slip) + (1 - p) * p_guess
        posterior = (p * (1 - p_slip)) / p_correct
    else:
        p_incorrect = p * p_slip + (1 - p) * (1 - p_guess)
        posterior = (p * p_slip) / p_incorrect
    self.skills[skill] = posterior + (1 - posterior) * p_learn
```

Line for line, this is the three formulas above.

### 6.4 How it works — a real trace, computed by running the actual class

I ran this exact class, not a hand simulation:

```
start:              p(knows) = 0.3000

correct   →  0.3000 → 0.6927
incorrect →  0.6927 → 0.2978
correct   →  0.2978 → 0.6906
correct   →  0.6906 → 0.9185
incorrect →  0.9185 → 0.6264
correct   →  0.6264 → 0.8947
correct   →  0.8947 → 0.9771
correct   →  0.9771 → 0.9953
```

One thing worth noticing: the two "incorrect" rows drop by very different
amounts (0.69→0.30 vs. 0.92→0.63), even though both are the same kind of
observation. At high confidence (0.92), one wrong answer is cheaply explained
away as a **slip** — the formula expects occasional slips even from someone
who knows the material, so it barely moves. At medium confidence (0.69), the
same wrong answer is more ambiguous, so it swings harder. That's the formula
correctly doing its job, not a quirk.

### 6.5 Why BKT and not the DKT paper's neural network

Being honest about the tradeoff, since the paper explicitly offers the
alternative:

- DKT needs a large *history* of student attempts to train its network on.
  Our student is simulated and every session starts fresh — that history
  doesn't exist yet.
- BKT gives a usable number from the very first observation, with four
  parameters you can read and understand directly.
- We *are* collecting the history DKT would eventually need — every
  `/observe` call writes an append-only row (learner, skill, correct,
  timestamp) in `services/tracer`, specifically so a DKT model can be trained
  later without the data having been thrown away.

$\alpha$ in §2.2 is just `tracer.get_ability()` — the average of $p$ across
all the skills the student has been tracked on.

---

## 7. How the tutor actually learns from the reward

This is the part that needs the most care, so we go slowly and stick to only
what's actually implemented.

### 7.1 What the paper says

Run the tutor, see what reward it got, adjust the tutor to make good outcomes
more likely and bad outcomes less likely. Repeat many times. That's
reinforcement learning, stated as plainly as it can be stated.

### 7.2 The one idea that makes this possible for a language model

Here's the actual difficulty: the tutor's output is *sampled text*. You
can't ask "what change to the model would have made this a slightly better
sentence?" the way you can ask "what change makes this photo classifier less
wrong?" — text generation involves picking discrete words, and you can't
smoothly nudge a discrete choice.

The trick that solves this: instead of trying to improve the *text*, improve
the **probability the model assigned to producing that text**. If a
trajectory got a good reward, increase the probability the model would say
that again. If it got a bad reward, decrease it.

$$\text{nudge} = (\text{how good was this outcome}) \times (\text{gradient of } \log \pi_\theta(\text{this trajectory}))$$

$\log \pi_\theta(\tau)$ — "log probability the current model assigns to
having produced trajectory $\tau$" — is a number you genuinely can compute
and differentiate, even though $\tau$ itself is discrete text. That's the
whole trick, and it's the origin of every RL-for-language-model method,
including ours.

### 7.3 The specific version we use: GRPO's group-relative advantage

**Formula, built up in two small pieces:**

Instead of judging one rollout's reward in isolation, run **several**
rollouts on the *same* problem — a **group** — and judge each one against
the group's own average:

$$A_i = \frac{r_i - \bar{r}}{\sigma_r + \varepsilon}$$

where $\bar{r}$ is the group's mean reward, $\sigma_r$ is the group's
standard deviation, and $\varepsilon$ is a tiny number (like $10^{-8}$) that
only exists to prevent dividing by zero. $A_i$ is called the **advantage** —
"how much better than average was this specific rollout." Positive $A_i$
means above the group's average; negative means below.

**What we built:**

```python
# sahai/training/grpo.py, _compute_advantages
mean_r = sum(rewards) / len(rewards)
var = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
std_r = var ** 0.5
rollout.advantage = (rollout.reward - mean_r) / (std_r + 1e-8)
```

**How it works:** a group of 4 rollouts on the same problem scored
$-0.30, -0.85, -0.60, -0.65$.

$$\bar r = \frac{-0.30-0.85-0.60-0.65}{4} = -0.60$$

$$\sigma_r = \sqrt{\frac{(0.30)^2+(0.25)^2+(0.00)^2+(0.05)^2}{4}} \approx 0.196$$

The first rollout's advantage: $A = \frac{-0.30-(-0.60)}{0.1969} \approx
+1.52$ — well above its group, so it gets pushed *toward* being more likely.
The second: $A = \frac{-0.85-(-0.60)}{0.1969}\approx -1.27$ — below its
group, pushed away.

### 7.4 The one sentence that explains most of the bugs found in this project

**Read the formula in §7.3 again: if all four rewards in a group happen to
be identical, $\sigma_r = 0$, and every single $A_i$ becomes exactly 0.**

Zero advantage means zero nudge — the model doesn't move at all for that
group, no matter how good or bad those rewards were in absolute terms.

This is *exactly* what happened in this project's very first run: `r_ped`
was binary back then (§4.2's paper version). Every rollout in the group
failed at least one check, every reward came out identical, every advantage
was exactly 0, and the training log printed `loss=0.0000` — not a display
glitch, a real, correct zero. **This is why §4.2 grades `r_ped` instead of
using the paper's all-or-nothing rule** — with no group variance, this
particular training method learns nothing at all, regardless of how good or
bad the dialogues actually were.

### 7.5 Turning advantage into an actual update

```python
# sahai/training/grpo.py, _policy_update
token_count = mask.sum().clamp(min=1)
mean_lp = (log_probs * mask).sum() / token_count
policy_loss = -advantage * mean_lp
```

`mean_lp` is the average $\log \pi_\theta$ over just the tutor's own tokens
in this rollout (§7.6 explains why only the tutor's tokens). Multiplying by
`-advantage`: if `advantage` is positive, minimising this loss means
*increasing* `mean_lp` — making those tutor tokens more likely next time. If
negative, the opposite. This is §7.2's trick, written out in code.

### 7.6 Why only the tutor's own words get changed

A dialogue has both tutor and student turns. Only the tutor is the thing
being trained — the student is a fixed simulator. So before computing
anything, every token gets tagged: was this word said by the tutor, or the
student?

```python
# sahai/agents/tutor.py, _build_tutor_mask
if messages[i - 1]["role"] == "assistant":   # "assistant" = tutor
    mask[0, prev_len:curr_len] = 1.0
```

Only tutor-tagged tokens get multiplied into `mean_lp` above. This has a real
consequence worth remembering: because we're computing "how likely was the
model to produce *exactly* this text," we must score the text the model
*actually said* — never an edited version. Early in this project, a cleanup
step was accidentally applied to the tutor's text *before* scoring it; that
meant the model was being scored on words it never actually produced, which
breaks §7.2's trick entirely. The fix: keep the raw text for scoring, only
clean it up for anything shown to a human.

### 7.7 Not moving too far, too fast: the KL penalty

**What the paper says:** nothing stops the update in §7.5 from making a huge
jump if it gets a very lucky (or unlucky) reward — potentially breaking
things the reward function isn't even measuring, like basic coherence. You
need a leash back to a known-good version of the model.

**The formula:**

$$\mathcal{L}_{total} = \mathcal{L}_{policy} + \beta \cdot \big(\log\pi_\theta - \log\pi_{ref}\big)$$

$\pi_{ref}$ is the model *before this training run started* — the
"reference." $\beta$ (`kl_coeff = 0.05`) controls how strongly it pulls back.

**What we built** — the clever part is *how* $\pi_{ref}$ is obtained, with no
second copy of the model needed:

```python
# sahai/training/grpo.py, _ref_log_probs
self.tutor.model.disable_adapter_layers()   # turn the training OFF
with torch.no_grad():
    log_probs, mask = self.tutor.compute_log_probs(dialogue, problem)
self.tutor.model.enable_adapter_layers()    # turn it back ON
```

This only works because of *how* the tutor is being trained — with LoRA,
covered next. With LoRA, "the model with training turned off" and "the
original pretrained model" are the exact same thing, by construction.

---

## 8. LoRA — training without forgetting

### 8.1 Context

This one isn't from the four review papers — it's a widely-used technique
from the broader machine learning literature (Hu et al., *LoRA:
Low-Rank Adaptation of Large Language Models*, 2021), used here for a
practical reason: training all of a language model's weights with RL is
expensive, and risks the model forgetting general abilities (fluent English,
basic code reasoning) while overfitting to this narrow reward signal.

### 8.2 The formula

For each weight matrix $W$ inside the model, instead of updating $W$
directly, **freeze $W$ completely** and add a small trainable correction:

$$W' = W + \frac{\alpha}{r}\,BA$$

$B$ and $A$ are two small matrices whose inner dimension $r$ (the **rank**)
is deliberately tiny compared to $W$'s own size. $\alpha$ here is a scaling
knob (unrelated to the $\alpha$ in §2.2 — same letter, different thing,
common source of confusion, worth saying explicitly).

### 8.3 What we built

```python
# sahai/settings.py
lora_rank: int = 16          # r  (8 on the Kaggle config)
lora_alpha: int = 32         # α  (16 on Kaggle)
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

```python
# sahai/core/models.py — every base weight frozen first
p.requires_grad_(False)
```

Only four specific weight matrices per layer (the attention projections) get
a $BA$ correction. Everything else — the much larger feed-forward layers,
the embeddings — is completely untouched.

### 8.4 How it works — the actual size comparison

For one of Qwen2.5-1.5B's attention matrices, roughly $d=k=1536$. Full
fine-tuning of that one matrix: $1536 \times 1536 = 2{,}359{,}296$ trainable
numbers. LoRA at rank $r=8$: $8 \times (1536+1536) = 24{,}576$ trainable
numbers for the same matrix — about **1%** of the full size. Multiply that
ratio across the whole model and you get why this is called
"parameter-efficient": the vast majority of what the model knows is
mathematically incapable of changing, because it's frozen, full stop.

---

## 9. All of it, in one picture

```
   ProblemBank (only problems in the student's learnable range, via §6's mastery)
        │
        ▼
   DialogueEngine.run()  ──►  one Dialogue  (§1)
        │
        ▼
   ┌─────────────────────────────────────────┐
   │  r_sol  (§3, run the code, K samples)     │
   │  r_ped  (§4, 5 graded checks)             │
   │  L      (§5, run the code, or word match) │
   └─────────────────────────────────────────┘
        │
        ▼
   r_SAHAI = (r_sol − α) + (r_ped − 1)λ − γL     (§2)
        │
        ▼
   group of these, per problem  ──►  advantage A_i  (§7.3)
        │
        ▼
   policy_loss = −A_i · mean(log π over TUTOR tokens only)   (§7.5, §7.6)
        + β · KL(π_θ ‖ π_ref)                                (§7.7)
        │
        ▼
   loss.backward()  →  only the small LoRA matrices move (§8)
```

Every box in this picture is something you've now seen derived, coded, and
run with real numbers above.

---

## 10. Named, not derived: what's planned but not built yet

Kept short and separate on purpose, so it doesn't tangle with what's real
above. Full detail in [06-roadmap.md](06-roadmap.md), Track A.

- **Re-using rollouts for multiple update steps**, and the safety mechanism
  that requires (an "importance ratio" between old and new model, with
  limits on how far it's trusted). Not implemented — we currently take
  exactly one update step per rollout, which is why this hasn't been needed
  yet.
- **A more precise way to estimate the KL number** in §7.7 (called "k3" in
  the literature) — the current one can occasionally print a small negative
  number even though true KL divergence never is.
- **Testing whether hints actually caused the solve**, by sometimes
  deliberately withholding them and comparing outcomes (called "ACE" in the
  literature review).
- **Actually measuring** whether §8's forgetting-prevention worked, by
  testing the model's general coding ability before and after training.

---

## Where to go from here

Pick one section above and ask for it to be walked through again, slower, or
with a different example — the structure repeats, so once one section clicks
the rest follow the same shape. [09-track-a-tutorial.md](09-track-a-tutorial.md)
covers the same ground with the full mathematical derivations (the
score-function proof, the Bayes'-theorem derivation step by step) if you want
the "why does the formula have this shape" layer underneath what's here.
