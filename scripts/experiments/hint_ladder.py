"""Does ANY amount of help work on this task, and is our tutor allowed to give it?

Two numbers have been measured on this bank and nothing between them. With no
dialogue the student solves 0.433. With the reference solution pasted in front
of it, 0.833. Everything this project built lives in that unmeasured 0.40 gap,
and the question nobody asked is whether the gap is reachable by anything short
of disclosure.

The literature says it may not be reachable by what we permit. A think-aloud
study of twelve novice programmers across four hint levels (arXiv:2404.02213)
found that high-level natural-language hints alone can be unhelpful or even
misleading, and that lower-level hints such as code examples with inline
comments support novices better. Our tutor's system prompt is:

    NEVER write code. No code blocks. No function definitions. No pseudocode.
    Ask ONE question per turn. Keep responses to 2-3 sentences maximum.

That is precisely the register the study reports as ineffective, and the reward
reinforces it: the leakage term penalises overlap with the reference and the
pedagogy checks fail any turn containing a code fence. The project may have
spent twenty runs optimising a policy into the one hint class that does not
work.

This measures a ladder of six concreteness levels, all constructed
programmatically from the reference solution so no tutor model is involved and
no policy quality confounds the result:

  L0  nothing
  L1  a generic Socratic question, the register our tutor is restricted to
  L2  the approach named, from the solution's own structure
  L3  a structural skeleton of the solution, bodies removed
  L4  working code for a DIFFERENT problem sharing a skill, as analogy
  L5  the reference solution itself

For each level it also records what our own leakage metric scores it, because
the interesting case is a level that helps the student AND passes the leakage
check, which would mean the useful hint was available and the prompt forbade it
for no measured reason.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("SAHAI_SOLVE_CONTEXT", "hints")
sys.path.insert(0, ".")
sys.path.insert(0, "libs/sahai-core")

LEVELS = ["L0_none", "L1_socratic", "L2_approach", "L3_skeleton",
          "L4_analogy", "L5_solution"]


def constructs(code: str) -> list[str]:
    """Plain names for what the reference solution is built from.

    AST technique detection fires on only about a third of this bank, so L2 is
    derived from control flow and builtins instead, which are always present.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    found: list[str] = []

    def add(x):
        if x not in found:
            found.append(x)

    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    fdefs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            add("a loop over the input")
        elif isinstance(node, ast.While):
            add("a while loop")
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp,
                               ast.GeneratorExp)):
            add("a comprehension")
        elif isinstance(node, ast.If):
            add("a conditional")
        elif isinstance(node, ast.Subscript):
            add("indexing or slicing")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in fdefs:
                add("a recursive call")
    for builtin, label in (("sorted", "sorting"), ("set", "a set"),
                           ("dict", "a dictionary"), ("len", "a length check"),
                           ("max", "a maximum"), ("min", "a minimum"),
                           ("sum", "a sum"), ("zip", "pairing two sequences"),
                           ("range", "an index range"), ("enumerate", "enumeration"),
                           ("map", "a map"), ("filter", "a filter")):
        if builtin in names:
            add(label)
    for attr, label in (("append", "building a list"), ("join", "joining strings"),
                        ("split", "splitting a string"), ("sort", "sorting in place"),
                        ("items", "iterating a dictionary"), ("keys", "dictionary keys")):
        if attr in attrs:
            add(label)
    return found


def skeleton(code: str) -> str:
    """The solution's shape with every body replaced by a placeholder.

    Keeps signatures, loop and conditional headers, and return positions.
    Removes the expressions that constitute the answer.
    """
    out = []
    for raw in code.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            continue
        indent = " " * (len(line) - len(line.lstrip()))
        s = line.strip()
        if s.startswith(("def ", "for ", "while ", "if ", "elif ", "else",
                         "try", "except", "with ", "class ")):
            out.append(indent + s.split("#")[0])
        elif s.startswith("return"):
            out.append(indent + "return ...")
        elif s.startswith(("import ", "from ")):
            out.append(indent + s)
        else:
            out.append(indent + "...")
    # collapse runs of placeholders
    cleaned = []
    for line in out:
        if line.strip() == "..." and cleaned and cleaned[-1].strip() == "...":
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def build_hint(level: str, problem, analogy) -> str | None:
    if level == "L0_none":
        return None
    if level == "L1_socratic":
        # Deliberately vacuous and within our own prompt rules: a question,
        # no code, two sentences. This is the register the policy was trained
        # into, so it is the control for "what our tutor is allowed to say".
        return ("Before you write anything, think about what you need to keep "
                "track of as you move through the input. What structure would "
                "let you do that?")
    if level == "L2_approach":
        cs = constructs(problem.solution)
        skills = ", ".join(s.replace("_", " ") for s in problem.skills
                           if s != "general")
        parts = []
        if skills:
            article = "an" if skills[0] in "aeiou" else "a"
            parts.append(f"This is {article} {skills} problem.")
        if cs:
            parts.append("The approach uses " + ", ".join(cs[:4]) + ".")
        return " ".join(parts) or None
    if level == "L3_skeleton":
        sk = skeleton(problem.solution)
        return ("Here is the shape of a working solution, with the logic "
                f"removed. Fill it in.\n\n```python\n{sk}\n```")
    if level == "L4_analogy":
        if analogy is None:
            return None
        return ("Here is a worked solution to a DIFFERENT problem that uses "
                "the same idea. Study the technique, then apply it to yours.\n\n"
                f"Problem: {analogy.description}\n"
                f"```python\n{analogy.solution.strip()}\n```")
    if level == "L5_solution":
        return ("Here is a correct solution. Study it, then write it out:\n\n"
                f"```python\n{problem.solution.strip()}\n```")
    raise ValueError(level)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", type=int, default=60)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="artifacts/ladder/ladder.json")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    import torch

    from sahai.agents.student import StudentPersona, StudentSimulator
    from sahai.core.dataset import load_mbpp
    from sahai.core.dialogue import Dialogue
    from sahai.core.models import load_for_inference, resolve_device
    from sahai.eval.ab_benchmark import mcnemar
    from sahai.eval.benchmark import Evaluator
    from sahai.reward.solve import CodeVerifier, SolveReward
    from sahai.settings import Settings

    settings = Settings.kaggle()
    device = resolve_device(args.device)
    print(f"device: {device}   framing: {os.environ['SAHAI_SOLVE_CONTEXT']}", flush=True)

    train_bank = load_mbpp(split="train", max_problems=settings.max_problems)
    bank = load_mbpp(split=args.split, max_problems=args.problems)
    problems = bank.problems

    # An analogy source: another problem sharing a skill, never this one.
    by_skill: dict[str, list] = {}
    for p in train_bank.problems:
        for s in p.skills:
            if s != "general":
                by_skill.setdefault(s, []).append(p)

    def pick_analogy(problem):
        """The closest other problem, by shared skills AND shared constructs.

        Ranking on a single skill tag in common matched a list-slicing problem
        to a sorting one, which is an analogy in name only. Constructs come
        from the same AST pass L2 uses, so the example actually demonstrates
        the technique the learner needs.
        """
        want_skills = {s for s in problem.skills if s != "general"}
        want_cons = set(constructs(problem.solution))
        pool = {c.id: c for s in want_skills for c in by_skill.get(s, [])}
        best, best_score = None, -1.0
        for cand in pool.values():
            if cand.id == problem.id:
                continue
            cs = {x for x in cand.skills if x != "general"}
            cc = set(constructs(cand.solution))
            skill_j = len(want_skills & cs) / max(len(want_skills | cs), 1)
            cons_j = len(want_cons & cc) / max(len(want_cons | cc), 1)
            # Constructs weigh more: they are what the example demonstrates.
            score = cons_j * 2 + skill_j
            if score > best_score:
                best, best_score = cand, score
        return best

    evaluator = Evaluator(settings, solution_corpus=[p.solution for p in train_bank.problems])
    student_model, student_tok = load_for_inference(
        settings.model.student, settings.model.dtype, device, quantize_4bit=False
    )
    persona = StudentPersona(ability_level=2, code_mixing_ratio=0.3,
                             language="hinglish", persistence=0.7)
    verifier = CodeVerifier(timeout=settings.reward.exec_timeout,
                            memory_mb=settings.reward.exec_memory_mb)
    solve = SolveReward(verifier, settings.reward.num_solve_samples)

    rows: dict[str, list[dict]] = {lv: [] for lv in LEVELS}
    t0 = time.time()
    for lv in LEVELS:
        student = StudentSimulator(student_model, student_tok, persona,
                                   max_new_tokens=settings.model.student_max_new_tokens)
        student.tracer.reset()
        torch.manual_seed(settings.seed)
        print(f"\n-- {lv} --", flush=True)
        for i, p in enumerate(problems, 1):
            hint = build_hint(lv, p, pick_analogy(p))
            d = Dialogue(problem_id=p.id)
            if hint is not None:
                d.add("student", "I do not know how to start this one.")
                d.add("tutor", hint)
            with torch.no_grad():
                out = solve.compute(student, d, p)
            leak = 0.0
            if hint is not None:
                leak = evaluator.leakage_estimator.estimate(
                    d, p.solution,
                    problem_text=f"{p.title} {p.description}",
                )
            rows[lv].append({
                "problem_id": p.id,
                "solved": 1 if out.solved >= 1.0 else 0,
                "partial": float(out),
                "hint_chars": len(hint or ""),
                "leakage": leak,
            })
            if i % 15 == 0 or i == len(problems):
                k = sum(r["solved"] for r in rows[lv])
                print(f"   {i:3d}/{len(problems)}  running solve {k}/{i} = {k/i:.3f}",
                      flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(rows, open(args.out, "w"), indent=1)

    print("\n" + "=" * 74)
    print(f"HINT LADDER   n={len(problems)}   {(time.time()-t0)/60:.1f} min")
    print("=" * 74)
    print(f"{'level':<14}{'solve':>8}{'partial':>9}{'leak':>8}{'chars':>8}"
          f"{'vs L0':>9}{'p':>9}  allowed?")
    print("-" * 74)
    base = [r["solved"] for r in rows["L0_none"]]
    for lv in LEVELS:
        rs = rows[lv]
        n = len(rs)
        k = sum(r["solved"] for r in rs)
        leak = sum(r["leakage"] for r in rs) / n
        ch = sum(r["hint_chars"] for r in rs) / n
        m = mcnemar(base, [r["solved"] for r in rs])
        delta = k / n - sum(base) / len(base)
        # Our tutor's system prompt forbids code, code blocks and pseudocode,
        # so the top three rungs are unreachable for it by construction.
        allowed = ("yes" if lv in ("L0_none", "L1_socratic", "L2_approach")
                   else "NO (prompt forbids code)")
        print(f"{lv:<14}{k/n:>8.3f}{sum(r['partial'] for r in rs)/n:>9.3f}"
              f"{leak:>8.3f}{ch:>8.0f}{delta:>+9.3f}"
              f"{m['p_value']:>9.4f}  {allowed}")

    print("\nREADING")
    # The first version of this block was wrong and printed a flattering
    # conclusion the data did not support. It fired "the prompt is the
    # constraint" whenever the best permitted rung failed to beat L0 and ANY
    # rung beat it, without checking which rung. On the real data only L5, the
    # complete solution, beat L0. Disclosure is not tutoring, so that rule
    # turned "only the answer works" into "our prompt is the problem".
    #
    # The question is specifically whether a rung that is NOT disclosure helps.
    l0 = sum(base) / len(base)
    partial = [lv for lv in LEVELS if lv not in ("L0_none", "L5_solution")]
    helped = []
    for lv in partial:
        k = sum(r["solved"] for r in rows[lv]) / len(rows[lv])
        m = mcnemar(base, [r["solved"] for r in rows[lv]])
        if k > l0 and m["p_value"] < 0.05:
            helped.append((lv, k, m["p_value"]))
    best_partial = max(partial, key=lambda lv: sum(r["solved"] for r in rows[lv]))
    bp = sum(r["solved"] for r in rows[best_partial]) / len(problems)
    l5 = sum(r["solved"] for r in rows["L5_solution"]) / len(problems)

    print(f"  no help                          {l0:.3f}")
    print(f"  best PARTIAL hint                {bp:.3f}  ({best_partial})")
    print(f"  full solution disclosed          {l5:.3f}")
    print()
    if helped:
        for lv, k, pv in helped:
            print(f"  {lv} beats no help: {k:.3f} vs {l0:.3f}, p={pv:.4f}")
        print("  A hint short of disclosure helps. Tutoring is possible here.")
    else:
        print("  No hint short of the full solution beats giving no help at all.")
        print("  The 0.40 gap between no help and disclosure is reachable only")
        print("  by disclosing. On this benchmark no tutor bound by a")
        print("  non-disclosure constraint can win, whatever its size, prompt")
        print("  or training. That is a property of the task, not of a policy.")
    worst = min(partial, key=lambda lv: sum(r["solved"] for r in rows[lv]))
    wv = sum(r["solved"] for r in rows[worst]) / len(problems)
    if wv < l0:
        mw = mcnemar(base, [r["solved"] for r in rows[worst]])
        print(f"\n  Worst rung is {worst} at {wv:.3f}, below no help at all")
        print(f"  (p={mw['p_value']:.4f}). Partial information can mislead.")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
