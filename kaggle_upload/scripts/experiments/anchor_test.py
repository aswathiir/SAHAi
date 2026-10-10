"""Does the learner copy the analogy instead of solving the problem?

Finding #32 measured that showing the learner a correct, working solution to an
analogous problem costs 0.102 of solve rate against showing it nothing
(p=0.0243, n=137). The mechanism offered was anchoring: the learner transfers
the analogous pattern rather than solving the problem in front of it. That was
a hypothesis, because the ladder recorded outcomes and not code.

This records the code. Three rungs only, so the run is 411 generations rather
than 822:

  L0  nothing              the control. It never sees the analogy, so its
                           similarity to the analogy is the baseline that
                           similarity under L4 has to be compared against.
  L2  approach named       harmful too (-0.102, p=0.0094) and contains no code,
                           which separates "misled by any partial hint" from
                           "copies code it was shown".
  L4  analogous example    the rung under test.

The test is a paired comparison of how much the learner's attempt resembles the
analogy, under L4 against under L0, on the same problem. L0 is the right
control precisely because it was never shown the analogy: any similarity it has
is what two solutions to related problems share anyway.

Three readings are possible and they are not the same finding:

  L4 code resembles the analogy more than L0 code does, and most on the
  problems L4 lost      -> anchoring, as hypothesised
  L4 and L2 both harm but neither resembles anything new
                        -> partial information disrupts without being copied,
                           which is a different and broader claim
  no similarity shift   -> the mechanism is something else and we say so
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("SAHAI_SOLVE_CONTEXT", "hints")
sys.path.insert(0, ".")
sys.path.insert(0, "libs/sahai-core")

LEVELS = ["L0_none", "L2_approach", "L4_analogy"]
_IDENT = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


def toks(code: str) -> set[str]:
    """Identifiers and keywords, which is what copying would carry over."""
    return set(_IDENT.findall(code or ""))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def fn_names(code: str) -> set[str]:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return set()
    return {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="artifacts/anchor/anchor.json")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="0 means all")
    args = ap.parse_args()

    import torch

    from sahai.agents.student import StudentPersona, StudentSimulator
    from sahai.core.dataset import load_mbpp
    from sahai.core.dialogue import Dialogue
    from sahai.core.models import load_for_inference, resolve_device
    from sahai.eval.ab_benchmark import mcnemar
    from sahai.reward.solve import CodeVerifier
    from sahai.settings import Settings

    # Reuse the ladder's own hint construction so the rungs are identical to
    # the ones whose effect is being explained.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "hl", os.path.join(os.path.dirname(os.path.abspath(__file__)), "hint_ladder.py")
    )
    hl = importlib.util.module_from_spec(spec)
    sys.argv = [sys.argv[0]]
    spec.loader.exec_module(hl)

    settings = Settings.kaggle()
    device = resolve_device(args.device)
    print(f"device: {device}", flush=True)

    pool = []
    for sp, cap in (("train", 400), ("validation", 100), ("test", 100)):
        try:
            pool += load_mbpp(split=sp, max_problems=cap).problems
        except Exception as exc:  # noqa: BLE001
            print(f"  could not load {sp}: {type(exc).__name__}")
    problems = [p for p in pool if hl.is_multi_step(p.solution)]
    if args.limit:
        problems = problems[: args.limit]
    print(f"multi-step problems: {len(problems)}", flush=True)

    by_skill: dict[str, list] = {}
    for p in problems:
        for s in p.skills:
            if s != "general":
                by_skill.setdefault(s, []).append(p)

    def pick_analogy(problem):
        want_skills = {s for s in problem.skills if s != "general"}
        want_cons = set(hl.constructs(problem.solution))

        def rank(cands, use_skills):
            best, best_score = None, -1.0
            for cand in cands:
                if cand.id == problem.id:
                    continue
                cc = set(hl.constructs(cand.solution))
                score = len(want_cons & cc) / max(len(want_cons | cc), 1) * 2
                if use_skills:
                    cs = {x for x in cand.skills if x != "general"}
                    score += len(want_skills & cs) / max(len(want_skills | cs), 1)
                if score > best_score:
                    best, best_score = cand, score
            return best, best_score

        if want_skills:
            cands = {c.id: c for s in want_skills for c in by_skill.get(s, [])}
            best, score = rank(cands.values(), True)
            if best is not None and score > 0:
                return best
        best, score = rank(problems, False)
        return best if score > 0 else None

    student_model, student_tok = load_for_inference(
        settings.model.student, settings.model.dtype, device, quantize_4bit=False
    )
    persona = StudentPersona(ability_level=2, code_mixing_ratio=0.3,
                             language="hinglish", persistence=0.7)
    verifier = CodeVerifier(timeout=settings.reward.exec_timeout,
                            memory_mb=settings.reward.exec_memory_mb)

    rows: dict[str, list[dict]] = {lv: [] for lv in LEVELS}
    analogies = {p.id: pick_analogy(p) for p in problems}
    t0 = time.time()
    for lv in LEVELS:
        student = StudentSimulator(student_model, student_tok, persona,
                                   max_new_tokens=settings.model.student_max_new_tokens)
        student.tracer.reset()
        torch.manual_seed(settings.seed)
        print(f"\n-- {lv} --", flush=True)
        for i, p in enumerate(problems, 1):
            an = analogies[p.id]
            hint = hl.build_hint(lv, p, an)
            d = Dialogue(problem_id=p.id)
            if hint is not None:
                d.add("student", "I do not know how to start this one.")
                d.add("tutor", hint)
            with torch.no_grad():
                code = student.attempt_solution(d, p)
            score = verifier.verify(code, p)
            rows[lv].append({
                "problem_id": p.id,
                "analogy_id": an.id if an else None,
                "solved": 1 if score >= 1.0 else 0,
                "partial": score,
                "code": code,
            })
            if i % 25 == 0 or i == len(problems):
                k = sum(r["solved"] for r in rows[lv])
                print(f"   {i:3d}/{len(problems)}  solve {k}/{i} = {k/i:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    by_id = {p.id: p for p in problems}
    json.dump({"rows": rows,
               "analogy_solutions": {p.id: (analogies[p.id].solution if analogies[p.id] else None)
                                     for p in problems},
               "reference_solutions": {p.id: p.solution for p in problems}},
              open(args.out, "w"), indent=1)

    # ---- analysis -------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"ANCHORING TEST   n={len(problems)}   {(time.time()-t0)/60:.1f} min")
    print("=" * 72)

    base = {r["problem_id"]: r for r in rows["L0_none"]}
    print(f"{'level':<14}{'solve':>8}{'simAnalogy':>12}{'simRef':>9}"
          f"{'copiedName':>12}{'p vs L0':>10}")
    print("-" * 68)
    stats = {}
    for lv in LEVELS:
        sims_a, sims_r, copied, n = [], [], 0, 0
        for r in rows[lv]:
            an_sol = analogies[r["problem_id"]]
            if an_sol is None:
                continue
            n += 1
            lt = toks(r["code"])
            sims_a.append(jaccard(lt, toks(an_sol.solution)))
            sims_r.append(jaccard(lt, toks(by_id[r["problem_id"]].solution)))
            if fn_names(an_sol.solution) & fn_names(r["code"]):
                copied += 1
        k = sum(x["solved"] for x in rows[lv])
        m = mcnemar([base[x["problem_id"]]["solved"] for x in rows[lv]],
                    [x["solved"] for x in rows[lv]])
        stats[lv] = {"solve": k / len(rows[lv]),
                     "sim_analogy": sum(sims_a) / max(n, 1),
                     "sim_ref": sum(sims_r) / max(n, 1),
                     "copied_name": copied / max(n, 1),
                     "p": m["p_value"]}
        print(f"{lv:<14}{k/len(rows[lv]):>8.3f}{stats[lv]['sim_analogy']:>12.3f}"
              f"{stats[lv]['sim_ref']:>9.3f}{stats[lv]['copied_name']:>12.1%}"
              f"{m['p_value']:>10.4f}")

    # The decisive paired contrast: same problem, L4 against L0.
    print("\nPAIRED, SAME PROBLEM: does L4 resemble the analogy more than L0 does?")
    l4 = {r["problem_id"]: r for r in rows["L4_analogy"]}
    diffs, lost_diffs = [], []
    for pid, r0 in base.items():
        an = analogies[pid]
        if an is None or pid not in l4:
            continue
        a = toks(an.solution)
        d = jaccard(toks(l4[pid]["code"]), a) - jaccard(toks(r0["code"]), a)
        diffs.append(d)
        if r0["solved"] == 1 and l4[pid]["solved"] == 0:
            lost_diffs.append(d)

    def summarise(xs, label):
        if not xs:
            print(f"  {label}: none")
            return
        mean = sum(xs) / len(xs)
        up = sum(1 for x in xs if x > 0)
        print(f"  {label}: n={len(xs)}  mean shift {mean:+.4f}  "
              f"more similar on {up}/{len(xs)} = {up/len(xs):.1%}")

    summarise(diffs, "all problems")
    summarise(lost_diffs, "problems L0 solved and L4 lost")

    print("\nREADING")
    sa_l4 = stats["L4_analogy"]["sim_analogy"]
    sa_l0 = stats["L0_none"]["sim_analogy"]
    cn = stats["L4_analogy"]["copied_name"] - stats["L0_none"]["copied_name"]
    if sa_l4 > sa_l0 + 0.02 or cn > 0.05:
        print("  The learner's code moves toward the analogy when shown it.")
        print("  Anchoring is supported: it transfers the example instead of")
        print("  solving the problem in front of it.")
    elif stats["L2_approach"]["solve"] < stats["L0_none"]["solve"]:
        print("  No similarity shift, yet naming the approach harms as much as")
        print("  showing code. Partial information disrupts this learner without")
        print("  being copied, which is broader than anchoring and not the")
        print("  mechanism we proposed.")
    else:
        print("  Neither pattern holds. The mechanism is still unidentified.")
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
