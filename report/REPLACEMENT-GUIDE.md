# Replacing sections in your copy

Your copy is the draft titled **"SAHAi: Socratic RL Tutor"**, whose abstract
reports 17.5% on 20 unseen problems. The final file is
`report/sahai_preliminary_report.tex`.

**The honest recommendation first: do not patch your copy.** Take the final
file as the base and move your front matter into it. Three reasons:

1. Your copy's prose is garbled in sections I am *not* replacing. Section III
   contains "The code verifier in the prototype that sits next to the reward
   logic, anything importing the reward stack Gained the power to perform
   arbitrary model output." Section VII contains "The system which provides
   instructions operates through a policy that did not receive optimization
   during its training process." Patching leaves those in.
2. Your Fig. 1 caption has a stray `}` mid-sentence (`...4 problems per epoch.
   } $G{=}8$`) which will fail to compile.
3. Nineteen of the final file's twenty-six headings differ, every table has
   been renumbered, and the spelling convention changed. A section-by-section
   merge touches nearly everything anyway.

**What to carry across from your copy:** the title, the four-author block with
the Amrita affiliation and emails, and your filled bibliography entries if you
prefer your formatting. Everything else is in the final file.

---

## If you still want to patch section by section

| your section | action | replace with, from the final file |
|---|---|---|
| `\title{SAHAi: Socratic RL Tutor}` | **keep yours** | final file uses a longer title; yours is fine |
| author block | **keep yours** | the final file's block is a placeholder |
| Abstract | **replace** | `\begin{abstract}` ... `\end{abstract}`. Yours is 433 words over five paragraphs with inline maths and a citation; IEEE wants one paragraph of 150 to 250 words with neither. The replacement is 255 words. |
| `\begin{IEEEkeywords}` | **replace** | yours is three sentences of prose, not keywords. The replacement is five terms, alphabetical, first word capitalized only, per the style manual. |
| I. Introduction | **replace** | `\section{Introduction}` and `\subsection{Contributions}`. Contribution 3 in yours states the pedagogy term carries five times the usable gradient; that ratio reversed and the replacement reports both measurements. |
| II. Method | **replace** | `\section{Method}` and its three subsections. Your reward equation omits the correctness term, and the term list reads "he post-dialogue solve rate". |
| III. The Interactive System | **replace** | `\section{The Interactive System}`. Yours is 731 words with several broken sentences; the replacement is 227 and says plainly that no cohort has used it. |
| IV. Experimental Setup | **replace** | `\section{Experimental Setup}` with `tab:config`. Yours says 198 problems loaded and a 20-problem held-out set; both numbers changed. |
| V. Results | **replace wholesale** | everything from `\section{Results}` to the end of `\subsection{Run History}`. This is the bulk of the new work: eight subsections where you have three. |
| VI. Discussion | **replace** | `\section{Discussion}` and its seven subsections. Your three are all present, two of them revised. |
| VII. Threats to Validity | **replace** | `\section{Threats to Validity}` and `\subsection{Threats Introduced by This Revision}`. Your first threat, that there is no baseline, has been addressed and is replaced rather than deleted, because the sign of the conclusion changed once it was run. |
| Conclusion | **replace** | `\section{Conclusion and Future Work}`. Your heading reads "Conclusion and future work are discussed." and the body is garbled. |
| bibliography | **replace** | the final file's 16 entries. Every author list, venue and identifier was read off the primary record. Note your `rev-simstudents` and `rev-learnermodel` are correct and now carry venues; I corrected one of my own entries that cited the wrong paper. |
| Fig. 1 and its caption | **fix** | the stray `}` must go or the file will not compile |

---

## Sections in the final file that do not exist in yours

Add these; they are the new results.

| section | what it reports |
|---|---|
| The Controlled Comparison | four arms under both transcript framings, and the retraction of the claim that tutoring costs the learner |
| Six Instrumentation Defects | all six, one paragraph each |
| Episode Budget and Stopping Criterion | per-episode metrics, the probe series, wall clock, and why eight episodes |
| Policy Movement in Weight Space | the adapter moved 0.2% from initialization |
| Inference Study | the comparison against an untrained larger model, which dominates ours on every axis |
| Hint Concreteness and Learner Outcome | both hint ladders and the anchoring test. This is the paper's main contribution |
| Information Asymmetry Between Tutor and Student | tutor and student were the same checkpoint |
| Suitability of Critic-Free Policy Gradients | whether the method was the right choice |

---

## Claims in your copy that are now retracted

Do not carry these across under any merge strategy.

| your text | status |
|---|---|
| best held-out solve rate 17.5% over 20 unseen problems | superseded; measured on broken instrumentation |
| "validated tutor policy reward design results" | not supported; the correctness term changed nothing |
| solve rate "stayed between 6% and 18% with no statistical difference" | predates the controls |
| pedagogy carries five times the outcome term's usable gradient | reversed after partial credit |
| "there is no baseline, so the headline number is uninterpretable" | addressed; replaced, not deleted |
| the disclosure arm as a measured total-leakage corner | its leakage was never measured; the harness records zero whenever an arm has no tutor object |

---

## Before submitting

- **Compile it.** `pdflatex` twice, for cross-references. I have never run it:
  it is not installed on this machine, so the ~10 page estimate and every
  statement about layout is structural analysis of the source only.
- Check `figures/architecture.png` travels with the file.
- Confirm your venue's author kit: abstract limits range from 150 to 250 words
  across IEEE societies, keyword counts from 5 to 8, and some kits impose a
  hard page limit with rejection without review.
