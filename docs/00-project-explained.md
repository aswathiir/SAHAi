# SAHAi, explained from the beginning

This document assumes you know nothing about the project. It explains what was
built, why each piece exists, what each experiment asked, and what the answers
were. Nothing here is a plan or a proposal. Everything described was built and
run, and every number was measured.

---

## Part 1. The problem

### 1.1 Solving is not teaching

A language model that can solve a programming problem is not automatically able
to teach it. The two skills pull in opposite directions.

If your goal is for a student's next submission to be correct, the fastest route
is to hand over the answer. If your goal is for the student to *learn*, handing
over the answer is the worst thing you can do, because the student practises
copying instead of thinking.

So a tutor has two objectives that fight each other:

1. The student should succeed afterwards.
2. The tutor must not tell them the answer.

A system that only optimises (1) becomes an answer key. A system that only
optimises (2) can satisfy it by saying nothing at all. The interesting problem
is doing both.

### 1.2 Why we could not just train on examples

The normal way to teach a model a behaviour is supervised learning: collect
thousands of examples of the behaviour and fit the model to them. That needs a
corpus of real human tutoring dialogues about programming, labelled with
whether the student then succeeded.

No such corpus exists at the size supervised learning needs. That is stated as
the motivation in the review this project follows.

### 1.3 Why reinforcement learning was the natural choice

We have no examples of good tutoring, but we *can* check whether tutoring
worked. Have the tutor talk to a student, let the student attempt the problem,
then run the student's code against the problem's unit tests. If the tests
pass, the tutoring worked.

That is the setting reinforcement learning exists for: no demonstrations, but a
reliable way to score an outcome after the fact. In the recent literature this
is called reinforcement learning from verifiable rewards, and "verifiable" here
means literally executing code and checking the result, not asking another
model for an opinion.

---

## Part 2. What we built

### 2.1 The training loop, in plain terms

One cycle works like this:

1. Pick a programming problem from a bank of them.
2. A **tutor model** and a **student model** have a conversation about it, up to
   four turns each.
3. The student then writes a solution on its own.
4. We execute that solution against the problem's unit tests.
5. We compute a score for the tutor from the result.
6. We nudge the tutor's weights in the direction that would have produced a
   better score.

Repeat. The tutor is the thing being trained. The student is a fixed model
playing a role, and it never learns anything; it is there to be taught to.

### 2.2 The pieces, and why each exists

**The tutor policy.** A 1.5-billion-parameter instruction model (Qwen2.5-1.5B)
with a small trainable attachment called a LoRA adapter. We train only the
adapter, about 2.2 million numbers out of 1.5 billion, because training the
whole model needs far more memory than we have. The adapter is why a 16 GB
machine can do this at all.

**The simulated student.** A second model, frozen, prompted to behave like a
weak learner: ability 2 out of 5, prone to specific misconceptions, mixing
English with Hindi or Tamil the way the target users do. It exists because we
cannot put a real student in a training loop that runs thousands of
conversations.

**The verifier.** Runs the student's code in an isolated container against the
problem's unit tests, and reports what fraction passed. This is the only part of
the reward that cannot be argued with: either the code ran and passed or it did
not.

**The reward.** One number per conversation, combining four things:

| term | what it asks | direction |
|---|---|---|
| solve | did the student's code pass the tests | higher is better |
| ability baseline | would the student have passed anyway | subtracted off |
| pedagogy | did the tutor avoid giving the answer | higher is better |
| leakage | did the tutor's text contain the solution | lower is better |

The ability baseline matters: without it, a tutor gets credit for a student who
was always going to succeed.

**The learner model.** A Bayesian Knowledge Tracing tracker. After each
observed outcome it updates a probability, per skill, that the student has
mastered that skill. This feeds two things: the ability baseline above, and the
choice of which problem to show next. We chose Bayesian Knowledge Tracing over
the deep-learning alternative because the deep version needs a history of real
student interactions, which we do not have.

**The curriculum.** Picks the next problem near the edge of what the student can
currently do, following the educational idea of a zone of proximal development:
too easy teaches nothing, too hard teaches nothing.

### 2.3 Why GRPO and not the usual algorithm

The standard reinforcement learning algorithm for language models is PPO. PPO
needs a second neural network, a "critic", that predicts how good a situation is.
That critic is roughly as large as the model being trained.

We cannot afford it. A tutor and a student already occupy a 16 GB accelerator;
a third network does not fit.

GRPO removes the critic. Instead of learning to predict a baseline, it generates
several conversations for the *same* problem and compares them to each other. If
one conversation scored above the average of its group, push the tutor toward
it; below, push away.

**This choice has a consequence that turned out to matter enormously.** With no
critic, the entire learning signal is the *spread of scores within a group*. If
all eight conversations about a problem score the same, their average is their
score, every comparison is zero, and that problem teaches the tutor nothing at
all while still costing a full round of generation.

Much of this project's history is the story of fighting to keep that spread from
collapsing.

### 2.4 The serving half

Separately from training, there is a working system a real learner can use: six
containerised services, a web workspace with a code editor, voice input for
Indic languages, a progress page, and a graded placement test. Code submitted by
a learner is executed in a container with no network access, no root, and a
read-only filesystem, and that container is deliberately separate from the part
of the system that scores anything.

This half works. It has never been used by a real cohort, so nothing in this
document claims it helps anyone learn.

---

## Part 3. What we measured, and in what order

Each experiment below was run because the previous one raised a question. That
order matters, because several of them overturned conclusions from earlier ones.

### 3.1 First: how many training rounds, and why stop there?

We call one pass over a batch of problems an **epoch**, or episode. The run we
report most fully had **eight**.

Why eight and not eighty? Two reasons, one practical and one measured.

The practical one is time. Each epoch generates 32 conversations, each up to
four turns from two models plus a final solution attempt, which is roughly 288
separate pieces of text generation. On our hardware an epoch took between 25 and
72 minutes. Eight epochs plus evaluation took about twelve hours.

The measured reason is better. Every two epochs we paused and tested the tutor
on 20 problems it had never seen. That is the **probe**:

| after epoch | probe solve rate |
|---|---|
| 1 | 0.150 |
| 3 | 0.250 |
| 5 | **0.350** |
| 7 | 0.250 |

It rose, then fell. We keep the best checkpoint, which was epoch 5, not the last
one. Running longer was not obviously useful because the series had already
turned down, and two earlier paired runs had peaked even sooner, at epoch 1 and
epoch 3.

**An important caution about this table.** The probe is 20 problems, so each
step of 0.05 is a single problem. The sequence 0.150, 0.250, 0.350, 0.250 is
3, 5, 7, 5 problems solved. That is consistent with random noise around 0.25,
and we initially over-read the first three points as a rising trend before the
fourth arrived.

### 3.2 Then: did training change the tutor at all?

This is the cheapest check in the project and we did not run it for a long time.

The trainable adapter starts in a specific state: one of its two matrices is
initialised to **exactly zero**. So everything training changed is visible in how
far that matrix moved away from zero.

| | value |
|---|---|
| the random reference matrix, average size | 0.0128 |
| **the trained matrix, average size** | **0.00076** |
| its largest single value | 0.0048 |

In plain terms: **the tutor moved about 0.2% away from where it started.** The
measure of how different its behaviour became from the untrained model never
rose above measurement noise across all eight epochs, and flipped sign at
random.

So twenty runs of careful reward engineering were performed on a model that
barely changed. That single fact explains most of the project's history more
simply than any argument about the reward.

### 3.3 Then: is the tutor better than no tutor?

For six runs this question had no answer, because every evaluation measured
"how well does the student do after tutoring" and compared it to a number
remembered from a previous run. Nobody had ever measured how well the student
does **with no tutoring at all**.

When we finally ran it, on 60 held-out problems, with every condition seeing the
same problems in the same order:

| condition | solve rate |
|---|---|
| reference solution shown outright | 0.833 |
| **no tutor at all** | **0.433** |
| prompted 3B model, no training | 0.417 |
| our untrained model | 0.350 |
| **our trained model** | **0.333** |

The trained tutor is last. The student does better with no help than with ours.

**But be careful with this table.** At 60 problems the statistical test cannot
distinguish differences smaller than about 15 points, and none of these gaps
clears that. Formally: our trained model versus no tutor is p = 0.21, versus the
untrained model p = 1.00. So the *ordering* is the direction of the evidence,
not an established fact. What we can say firmly is that no tutoring arm is
*better* than doing nothing.

### 3.4 Then: is a bigger model, with no training, better than ours?

This is the first time the project compared against a model it did not train.
A Qwen2.5-3B, given the same instructions, no adapter, no reward function, no
training:

| | solve | discloses the answer | follows pedagogy rules |
|---|---|---|---|
| our trained 1.5B | 0.333 | 0.137 | 0.893 |
| **prompted 3B, untrained** | **0.417** | **0.017** | **0.997** |

It is better on all three at once. It solves more, it gives the answer away
**eight times less often**, and it follows the pedagogy rules almost perfectly.

The disclosure column is the painful one. Not giving the answer away is the
single thing this project spent its entire optimisation budget on, and it is the
measure where a larger model with a plain prompt beats us most decisively.

**What our training did achieve, precisely:** between our untrained and trained
model, disclosure fell from 0.249 to 0.137 and pedagogy rose from 0.768 to
0.893. The reinforcement learning genuinely worked on the parts of the reward
the tutor controls through its own words. It just turns out that behaviour was
available for free from a bigger model.

### 3.5 Then: could *any* tutor have won here?

By this point the natural question is whether the task itself permits tutoring.
We knew two numbers and nothing in between: 0.433 with no help, 0.833 with the
answer shown. Everything the project ever built lives inside that gap.

So we measured six levels of help. Every hint was constructed automatically from
the reference solution, so **no tutor model is involved at all**. This measures
the task and the student, not our system.

| level | what the student was given |
|---|---|
| L0 | nothing |
| L1 | a generic guiding question, which is all our tutor is allowed to give |
| L2 | the approach named in one sentence, no code |
| L3 | the solution's shape, with all the logic deleted |
| L4 | a complete working solution to a *different* but similar problem |
| L5 | the reference solution itself |

Results, on 60 problems and then on 137 chosen for having genuinely multi-step
solutions:

| level | 60 problems | 137 multi-step | significance |
|---|---|---|---|
| L0 nothing | 0.433 | 0.336 | baseline |
| L1 guiding question | 0.400 | 0.263 | p = 0.064 |
| L2 approach named | 0.383 | **0.234** | **p = 0.009** |
| L3 shape only | 0.450 | 0.321 | p = 0.839 |
| L4 similar worked example | 0.317 | **0.234** | **p = 0.024** |
| L5 the answer | 0.833 | **0.752** | **p < 0.001** |

**Read the middle rows.** Nothing short of the complete answer beats giving no
help at all. On the harder set, two kinds of help make the student
*significantly worse* than silence.

So the answer is no. On this benchmark, **no tutor that refuses to give the
answer can win** — not ours, not a bigger one, not a perfect one. That is a
property of the task and the student, not of any policy.

### 3.6 Then: why does partial help hurt?

We proposed that showing a worked example makes the student copy it instead of
solving the problem in front of it. So we re-ran it recording the student's
actual code.

**That explanation is wrong.**

- When shown the example, the student's code is barely more similar to it:
  overlap rises 0.017, and only 47% of problems moved toward the example, which
  is below chance.
- The student borrows the example's function name on 5 of 137 problems, against
  4 of 137 when shown nothing.
- **The decisive point:** L2 is one sentence with no code in it at all, and it
  does *exactly* as much damage as the full worked example, 0.234 against 0.234.
  A hint with nothing to copy cannot do equal harm if copying is the mechanism.

What the code does show is two weaker and different effects. Naming the approach
pushes the student's code *away* from the correct solution without pulling it
toward anything. Showing an example keeps it about as close to correct but makes
it bigger and more elaborate, 76 to 84 syntax-tree nodes, and on the problems it
loses the student writes more code than it did unaided in 17 of 24 cases.

Neither reaches statistical significance alone. **We report the mechanism as
unidentified.**

### 3.7 One nuance that the averages hide

Partial help is not useless. It is net harmful.

| level | problems it fixed | problems it broke |
|---|---|---|
| L2 approach named | 6 | 20 |
| L4 worked example | 10 | 24 |

Help genuinely rescues six to ten problems the student could not do alone. It
breaks twenty-four. The average is negative because the damage is larger, not
because help never works. That is a more precise and more interesting statement
than "hints don't help".

---

## Part 4. The bugs we found, because they changed the numbers

Six separate measurement defects were found, each by asking why a result looked
the way it did, not by a test failing. Each one changed published numbers.

**1. The verifier failed correct code.** The student's output passes through a
JSON conversion; the expected answer does not. Comparing them directly marked 9
of 60 correct reference solutions as wrong, so the maximum achievable score was
never 1.0.

**2. The random seed was never applied.** It existed from the first commit and
was read nowhere in the training code. Two runs of an identical configuration
therefore saw different problems before a single training step. An experiment
built specifically to compare two reward settings was not actually comparable.

**3. The seed was applied too late, as well.** Once fixed, it was still applied
*after* the trainable adapter had been randomly initialised, so two runs still
diverged. Same bug, one layer deeper.

**4. The wrong function was being tested.** Each problem's test calls a specific
function by name. Our loader took the name of the *first* function defined in the
reference solution instead, and these differ whenever a solution starts with a
helper. On 5 of 60 held-out problems the verifier ran a helper function and
compared its output to the real function's expected answer, so those problems
were **impossible in every condition** and sat in the denominator of every score
we ever reported.

**5. Correct code was thrown away over its name.** The step that extracts code
from the student's reply looked for the exact expected function definition and,
not finding it, replaced the entire reply with a stub that fails on purpose. Every
mismatch we found was styling: the dataset uses `max_Prime_Factors` and the
student writes `max_prime_factors`. Three otherwise-correct solutions scored
zero.

**6. The conversation was replayed to the student in a damaging way.** When the
student wrote its final solution, we replayed the whole conversation including
the student's own earlier replies, in which it had often said it was confused.
Omitting its own replies and keeping only the tutor's raises that condition's
score from 0.183 to 0.350 and removes what we had been reporting as proof that
tutoring harms the learner.

Defects 4 and 5 together raise the "answer shown" ceiling from 0.633 to 0.783.
Defect 6 retracted a central claim of an earlier draft.

---

## Part 5. What we got wrong, and why it is in the record

Three explanations were offered for the central finding, and all three were
falsified by experiments run specifically to test them.

| we claimed | how it was tested | result |
|---|---|---|
| our tutor's no-code rule forbids the kind of hint that works | measured hints containing code | they do not help; one hurts |
| the benchmark's one-line solutions make structural hints empty | selected 137 genuinely multi-step problems | changed nothing |
| the student copies worked examples | recorded the student's code | it does not; and a no-code hint harms equally |

There was also a tooling failure worth naming. The script that printed the
first ladder's verdict was written to announce "the constraint is the prompt"
whenever the best permitted hint failed to beat the floor and *any* hint beat
it — without checking which. Only the full answer beat it, so the rule silently
converted "only the answer works" into "our prompt is the problem". That is the
exact class of error this project is otherwise about, committed inside the tool
built to detect it.

---

## Part 6. What the project actually contributes

**Not a working tutor.** The trained model does not help the simulated student,
and a larger prompted model beats it on every axis we measure.

**A working system.** Six services, isolated code execution, voice input,
persistent learner modelling, and a learner interface. Engineering, not a
learning result, and untested with real people.

**A measurement result that generalises past us, and this is the real
contribution.** A tutor evaluated against a simulated language-model student is
being scored by something that is harmed by partial, non-disclosing hints and
helped only by the answer. That penalises exactly the behaviour pedagogy
requires. Any reward built on such a student points away from teaching, toward
either disclosure or silence, and that is what every reward in this project did.

The simulated-student approach is widely used for exactly this purpose. It rests
on the assumption that a model student responds to teaching the way a student
does. Our two hint ladders, across 197 problems, are a direct test of that
assumption, and for a 1.5B student on programming problems it does not hold. We
could not find prior work that tests it.

**A matching result in humans.** A randomised trial in secondary mathematics
reports that unrestricted AI assistance improved students' performance while
they had the assistance, and *reduced* their later unaided exam performance. The
direction matches ours. Whether the cause is the same is unknown.

---

## Part 7. What would come next

In order of how much each would settle:

1. **Run the hint ladder across model sizes.** It needs no training and no tutor,
   so it is cheap. It would say whether "partial help harms" is a property of
   small models or of language-model students generally. That determines how
   broadly the main contribution applies.
2. **Settle the mechanism properly.** Vary one property of a hint at a time at a
   sample size that resolves 5 points, which is roughly 800 problems per level
   against our 137.
3. **Compare a deployed tutor against a prompted larger model on outcome and
   disclosure together.** On our numbers the prompted baseline dominates a
   trained small model on both, and we are not aware of this comparison being
   reported elsewhere.
4. **Do not spend more compute on this benchmark.** The ladder shows there is
   nothing for a non-disclosing tutor to win on it.
