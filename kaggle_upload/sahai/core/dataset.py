from __future__ import annotations

import ast
import re
from typing import Any

from sahai.core.data import Problem, ProblemBank, TestCase

SKILL_KEYWORDS: dict[str, list[str]] = {
    "arrays": ["array", "list", "element", "subarray", "matrix", "rotate", "flatten"],
    "strings": ["string", "substring", "character", "palindrome", "anagram", "vowel", "consonant"],
    "hash_maps": ["dictionary", "hash", "count", "frequency", "unique", "duplicate"],
    "sorting": ["sort", "order", "arrange", "kth", "largest", "smallest", "median"],
    "binary_search": ["binary search", "sorted array", "search"],
    "linked_lists": ["linked list", "node", "head"],
    "trees": ["tree", "binary tree", "bst", "subtree", "leaf"],
    "dynamic_programming": [
        "dynamic programming", "memoization", "minimum cost", "maximum sum",
        "longest", "shortest", "knapsack", "fibonacci", "subsequence",
    ],
    "recursion": ["recursive", "recursion", "factorial", "tower"],
    "stacks": ["stack", "parenthes", "bracket", "balanced"],
    "graphs": ["graph", "vertex", "edge", "path", "cycle", "connected"],
    "two_pointers": ["two pointer", "sliding window", "merge"],
    "math": ["prime", "gcd", "lcm", "factorial", "power", "modulo", "digit"],
    "bit_manipulation": ["bit", "binary", "xor", "bitwise"],
    "heaps": ["heap", "priority queue", "min heap", "max heap"],
}


def _tag_skills(text: str) -> list[str]:
    text_lower = text.lower()
    skills = []
    for skill, keywords in SKILL_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            skills.append(skill)
    return skills or ["general"]


def _parse_assert(assertion: str) -> tuple[str, dict[str, Any], Any] | None:
    assertion = assertion.strip()
    if not assertion.startswith("assert "):
        return None
    expr = assertion[len("assert "):]

    if "==" not in expr:
        return None
    lhs, rhs = expr.split("==", 1)
    lhs, rhs = lhs.strip(), rhs.strip()

    try:
        expected = ast.literal_eval(rhs)
    except (ValueError, SyntaxError):
        return None

    match = re.match(r"(\w+)\((.*)$", lhs)
    if not match:
        return None
    func_name = match.group(1)
    args_str = match.group(2)
    if args_str.endswith(")"):
        args_str = args_str[:-1]

    try:
        args_tuple = ast.literal_eval(f"({args_str},)")
    except (ValueError, SyntaxError):
        return None

    input_dict = {f"arg{i}": v for i, v in enumerate(args_tuple)}
    return func_name, input_dict, expected


def _extract_function_name(code: str) -> str:
    match = re.search(r"def\s+(\w+)\s*\(", code)
    return match.group(1) if match else "solution"


def _extract_function_signature(code: str) -> str:
    match = re.search(r"(def\s+\w+\s*\([^)]*\))", code)
    return f"{match.group(1)}:" if match else "def solution():"


def _estimate_difficulty(code: str, text: str) -> int:
    """Structural complexity of the reference solution, 1-5.

    The previous version counted three coarse indicators — >15 lines, a topic
    keyword, more than one indented loop — and MBPP solutions are mostly 3-8
    lines with a single loop, so it returned 1 for almost everything: measured
    over the 296 usable train problems it gave 292 x difficulty-1 and 4 x
    difficulty-2. A field with one value carries no information, so the
    interface could not show range and eval's per-difficulty buckets were
    degenerate.

    Parsing the solution instead spreads the same 296 problems across
    1:186 2:59 3:36 4:9 5:6, and the ordering is legible — reversing words in
    a string lands at 1, median-of-two-sorted-arrays at 4.

    Note this does **not** affect problem selection: `BKTTracer.in_zpd` takes
    `difficulty` as an argument and ignores it, choosing purely on skill
    mastery. So this changes what is displayed and how eval buckets, not what
    any learner or training run is offered.
    """
    source = "\n".join(
        line for line in code.splitlines() if not line.strip().startswith("#")
    )
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 1

    nodes = list(ast.walk(tree))
    loops = sum(isinstance(n, (ast.For, ast.While)) for n in nodes)
    branches = sum(isinstance(n, (ast.If, ast.IfExp)) for n in nodes)
    comprehensions = sum(
        isinstance(n, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp))
        for n in nodes
    )
    functions = [n for n in nodes if isinstance(n, ast.FunctionDef)]

    modules = {
        alias.name.split(".")[0]
        for n in nodes
        if isinstance(n, ast.Import)
        for alias in n.names
    } | {
        n.module.split(".")[0]
        for n in nodes
        if isinstance(n, ast.ImportFrom) and n.module
    }

    def max_nesting(node: ast.AST, depth: int = 0) -> int:
        deepest = depth
        for child in ast.iter_child_nodes(node):
            step = isinstance(child, (ast.For, ast.While, ast.If))
            deepest = max(deepest, max_nesting(child, depth + step))
        return deepest

    defined = {f.name for f in functions}
    recursive = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in defined
        for n in nodes
    )

    score = (
        min(loops, 3) * 0.8
        + min(branches, 3) * 0.5
        + comprehensions * 0.4
        + max(max_nesting(tree) - 1, 0) * 1.0
        + (len(functions) - 1) * 0.8
        + (1.5 if recursive else 0.0)
        # Reaching for these means the problem needed a real data structure.
        + (1.0 if modules & {"heapq", "bisect", "itertools", "functools", "collections"} else 0.0)
        + len([line for line in source.splitlines() if line.strip()]) / 14.0
        + (
            1.0
            if any(
                kw in text.lower()
                for kw in ("dynamic programming", "graph", "tree", "backtrack", "matrix")
            )
            else 0.0
        )
    )
    return max(1, min(5, round(score / 1.6)))


def load_mbpp(split: str = "train", max_problems: int | None = None) -> ProblemBank:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", split=split, trust_remote_code=True)
    problems = []

    for i, row in enumerate(ds):
        if max_problems and i >= max_problems:
            break

        func_name = _extract_function_name(row["code"])
        test_cases = []
        for assertion in row["test_list"]:
            parsed = _parse_assert(assertion)
            if parsed is None:
                continue
            _, input_dict, expected = parsed
            test_cases.append(TestCase(input=input_dict, expected=expected))

        if not test_cases:
            continue

        problem = Problem(
            id=f"mbpp_{row['task_id']}",
            title=row["text"][:80],
            description=row["text"],
            difficulty=_estimate_difficulty(row["code"], row["text"]),
            skills=_tag_skills(row["text"]),
            function_name=func_name,
            function_signature=_extract_function_signature(row["code"]),
            test_cases=test_cases,
            solution=row["code"],
        )
        problems.append(problem)

    return ProblemBank(problems=problems)


def load_apps(split: str = "train", difficulty: str = "interview", max_problems: int | None = None) -> ProblemBank:
    from datasets import load_dataset

    ds = load_dataset("codeparrot/apps", split=split, trust_remote_code=True, difficulties=[difficulty])
    problems = []

    for i, row in enumerate(ds):
        if max_problems and i >= max_problems:
            break

        try:
            import json
            test_cases_raw = json.loads(row.get("input_output", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue

        inputs = test_cases_raw.get("inputs", [])
        outputs = test_cases_raw.get("outputs", [])
        if not inputs or not outputs:
            continue

        test_cases = []
        for inp, out in zip(inputs[:5], outputs[:5]):
            test_cases.append(TestCase(
                input={"stdin": inp.strip()},
                expected=out.strip(),
            ))

        solutions = row.get("solutions", "")
        try:
            solution_list = json.loads(solutions) if solutions else []
            solution = solution_list[0] if solution_list else ""
        except (json.JSONDecodeError, TypeError):
            solution = ""

        problem = Problem(
            id=f"apps_{i}",
            title=row["question"][:80] if row.get("question") else f"APPS Problem {i}",
            description=row.get("question", ""),
            difficulty={"introductory": 1, "interview": 2, "competition": 4}.get(difficulty, 2),
            skills=_tag_skills(row.get("question", "")),
            function_name="solution",
            function_signature="def solution():",
            test_cases=test_cases,
            solution=solution,
        )
        problems.append(problem)

    return ProblemBank(problems=problems)
