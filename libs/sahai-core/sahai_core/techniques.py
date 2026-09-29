"""Which techniques a solution actually uses, read from its source.

Extracted from the NeetCode importer so the gateway can run the same
detection on a repo the learner connects from the browser. It was CLI-only,
which meant the one path a learner could actually reach did not exist.

Deliberately dependency-free: `ast` and `re` only. Anything needing the
problem bank's skill vocabulary stays in the importer, because the services
do not carry the research package.
"""

from __future__ import annotations

import ast
import re


TECHNIQUE_SKILLS: dict[str, str] = {
    "hash map": "hash_maps",
    "a set for lookups": "hash_maps",
    "two pointers": "two_pointers",
    "a sliding window": "two_pointers",
    "binary search": "binary_search",
    "a heap": "heaps",
    "a deque": "stacks",
    "an explicit stack": "stacks",
    "recursion": "recursion",
    "memoisation": "dynamic_programming",
    "a DP table": "dynamic_programming",
    "sorting": "sorting",
    "bit manipulation": "bit_manipulation",
    "a linked-list walk": "linked_lists",
    "a graph traversal": "graphs",
}

# Most specific first: a solution that sorts *and* uses a heap is better
# described by the heap, and one that memoises is better described by that
# than by the recursion underneath it.
TECHNIQUE_ORDER = (
    "memoisation", "a DP table", "a heap", "binary search", "a sliding window",
    "two pointers", "a graph traversal", "an explicit stack", "a deque",
    "bit manipulation", "a linked-list walk", "hash map", "a set for lookups",
    "recursion", "sorting",
)

_DP_NAMES = re.compile(r"\b(dp|memo|cache|table|seen_states)\b")
_GRAPH_NAMES = re.compile(r"\b(adj|graph|neighbors|neighbours|visited|queue|bfs|dfs)\b")
_POINTER_NAMES = re.compile(r"\b(left|right|lo|hi|slow|fast|start|end|l|r)\b")


def techniques_used(source: str) -> set[str]:
    """Which techniques this solution actually uses, as human phrases.

    Structural, not lexical: a comment saying "binary search" proves nothing,
    and these are the constructs a reviewer would point at.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    nodes = list(ast.walk(tree))
    found: set[str] = set()

    imports = {
        alias.name.split(".")[0]
        for n in nodes if isinstance(n, ast.Import) for alias in n.names
    } | {
        n.module.split(".")[0]
        for n in nodes if isinstance(n, ast.ImportFrom) and n.module
    }
    called = {
        n.func.id for n in nodes
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    } | {
        n.func.attr for n in nodes
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    names = {n.id for n in nodes if isinstance(n, ast.Name)}
    attrs = {n.attr for n in nodes if isinstance(n, ast.Attribute)}
    decorators = {
        d.id if isinstance(d, ast.Name) else getattr(d, "attr", "")
        for f in nodes if isinstance(f, ast.FunctionDef) for d in f.decorator_list
    }
    text = source.lower()

    if "heapq" in imports or {"heappush", "heappop", "heapify"} & called:
        found.add("a heap")
    if "bisect" in imports or {"bisect_left", "bisect_right", "insort"} & called:
        found.add("binary search")
    if "deque" in imports or "deque" in called:
        found.add("a deque")
    if {"lru_cache", "cache"} & decorators:
        found.add("memoisation")

    dicts = [n for n in nodes if isinstance(n, (ast.Dict, ast.DictComp))]
    if dicts or "defaultdict" in called or "Counter" in called:
        found.add("hash map")
    if any(isinstance(n, (ast.Set, ast.SetComp)) for n in nodes) or "set" in called:
        found.add("a set for lookups")

    if {"sort", "sorted"} & called:
        found.add("sorting")

    # A while loop whose test compares two pointer-ish names is the shape of
    # both two-pointer scans and binary search; the midpoint calculation is
    # what separates them.
    whiles = [n for n in nodes if isinstance(n, ast.While)]
    pointerish = {n for n in names if _POINTER_NAMES.fullmatch(n)}
    if whiles and len(pointerish) >= 2:
        found.add("two pointers")
    if re.search(r"(//\s*2|>>\s*1)", source) and whiles:
        found.add("binary search")
    if "window" in text or ("two pointers" in found and "max(" in text):
        found.add("a sliding window")

    # `AugAssign` as well as `BinOp`: the canonical xor solution is `r ^= n`,
    # which is an augmented assignment and has no BinOp node at all. Checking
    # only BinOp missed exactly the problems this tag exists for.
    bitwise = (ast.BitXor, ast.BitAnd, ast.BitOr, ast.LShift, ast.RShift)
    if any(
        isinstance(n, (ast.BinOp, ast.AugAssign)) and isinstance(n.op, bitwise)
        for n in nodes
    ):
        found.add("bit manipulation")

    if {"next", "head", "val"} & attrs:
        found.add("a linked-list walk")
    if _GRAPH_NAMES.search(text) and (whiles or any(
        isinstance(n, ast.For) for n in nodes
    )):
        found.add("a graph traversal")

    defined = {f.name for f in nodes if isinstance(f, ast.FunctionDef)}
    if any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in defined
        for n in nodes
    ):
        found.add("recursion")

    subscripted = {
        n.value.id for n in nodes
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
    }
    if _DP_NAMES.search(" ".join(subscripted | names)):
        found.add("a DP table")

    return found


def headline_technique(techniques: set[str]) -> str | None:
    """The one phrase worth telling a tutor, or None."""
    return next((t for t in TECHNIQUE_ORDER if t in techniques), None)

