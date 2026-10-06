"""Why does the student fail 17/55 problems when it is SHOWN the answer?

The oracle arm is the ceiling on everything this project can measure. It puts
a correct solution into the tutor's turn and asks the student to write it out.
It scored 0.633. The remaining 0.367 has been treated as a capability limit of
a 1.5B student, which would mean no reward design can help. That claim has
never been checked.

Three explanations are distinguishable from the raw generation, and only one of
them is a capability limit:

  name_mismatch  the student wrote working code under a different function
                 name, and `_extract_code` replaced the whole thing with
                 `def f(): pass`. An instrumentation failure scored as a
                 capability failure.
  truncated      the generation hit max_new_tokens=512 mid-function, so the
                 extracted text is a syntax error.
  wrong_logic    the student really did write something that does not work.

This dumps the raw generation alongside the extracted code so the three can be
counted, and re-scores each attempt with an AST-based extractor to measure how
much of the gap is recoverable without touching the model at all.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, ".")
sys.path.insert(0, "libs/sahai-core")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sahai.agents.student import StudentPersona, StudentSimulator
from sahai.core.dataset import load_mbpp
from sahai.core.dialogue import Dialogue
from sahai.reward.solve import CodeVerifier

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
OUT = sys.argv[2] if len(sys.argv) > 2 else "oracle_decompose.json"

bank = load_mbpp(split="test", max_problems=N)
problems = bank.problems
print(f"{len(problems)} problems from the MBPP test split (same selection as the benchmark)\n", flush=True)

name = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("mps").eval()
student = StudentSimulator(
    model, tok,
    StudentPersona(ability_level=2, code_mixing_ratio=0.3, language="hinglish", persistence=0.7),
    max_new_tokens=512,
)
verifier = CodeVerifier(timeout=10, memory_mb=256)


def rescue(text: str, want: str) -> str:
    """Extract code without discarding it when the name does not match.

    `_extract_code` checks for the literal string `def {function_name}` and,
    failing that, returns `def {name}(): pass`. Everything the student wrote is
    thrown away. This instead takes the fenced block if there is one, parses
    it, and aliases the last top-level function to the name the tests call.
    An alias rather than a rename so that recursion through the original name
    keeps working.
    """
    body = text
    if "```python" in body:
        body = body.split("```python")[1].split("```")[0]
    elif "```" in body:
        parts = body.split("```")
        body = parts[1] if len(parts) > 1 else body
    body = body.strip()
    if not body:
        return ""
    try:
        tree = ast.parse(body)
    except SyntaxError:
        # Truncation usually breaks only the tail. Drop trailing lines until it
        # parses, which recovers a complete function followed by a cut-off one.
        lines = body.split("\n")
        for cut in range(len(lines) - 1, 0, -1):
            try:
                tree = ast.parse("\n".join(lines[:cut]))
                body = "\n".join(lines[:cut])
                break
            except SyntaxError:
                continue
        else:
            return ""
    funcs = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not funcs:
        return ""
    if want in funcs:
        return body
    return f"{body}\n\n{want} = {funcs[-1]}\n"


def raw_generation(dialogue: Dialogue, problem) -> str:
    """`attempt_solution` with the extraction step removed.

    Copied rather than refactored so this measurement cannot change the
    behaviour it is measuring.
    """
    tutored = bool(dialogue.turns)
    opening = ("You are a student who has just been tutored on this problem."
               if tutored else "You are a student solving this problem on your own.")
    guidance = ("Use the hints you were given, and write a Python solution."
                if tutored else "Write a Python solution.")
    system = (
        f"{opening}\n"
        f"Your ability level is {student.persona.ability_level}/5.\n"
        f"Problem: {problem.description}\n"
        f"{guidance}\n"
        f"Only output the function implementation, nothing else.\n"
        f"Signature: {problem.function_signature}"
    )
    messages = [{"role": "system", "content": system}]
    messages.extend(student._solve_context(dialogue))
    messages.append({"role": "user", "content": "Now write your solution:"})
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    n_new = int(gen.shape[0])
    return tok.decode(gen, skip_special_tokens=True).strip(), n_new


records = []
t0 = time.time()
for i, p in enumerate(problems, 1):
    d = Dialogue(problem_id=p.id)
    d.add("student", "I do not know how to start this one.")
    d.add("tutor", "Here is a correct solution. Study it, then write it out:\n\n"
                   f"```python\n{p.solution.strip()}\n```")

    raw, n_new = raw_generation(d, p)
    shipped = student._extract_code(raw, p.function_name)
    rescued = rescue(raw, p.function_name)

    fell_back = shipped == f"def {p.function_name}():\n    pass"
    hit_cap = n_new >= 512

    score_shipped = verifier.verify(shipped, p)
    score_rescued = verifier.verify(rescued, p) if rescued else 0.0

    # What names did the student actually define?
    defined = []
    try:
        defined = [n.name for n in ast.parse(rescued or shipped).body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    except SyntaxError:
        pass

    parses = True
    try:
        ast.parse(shipped)
    except SyntaxError:
        parses = False

    records.append({
        "problem_id": p.id,
        "function_name": p.function_name,
        "defined": defined,
        "n_new_tokens": n_new,
        "hit_token_cap": hit_cap,
        "extract_fell_back": fell_back,
        "shipped_parses": parses,
        "score_shipped": score_shipped,
        "score_rescued": score_rescued,
        "solved_shipped": int(score_shipped >= 1.0),
        "solved_rescued": int(score_rescued >= 1.0),
        "raw": raw,
        "shipped": shipped,
        "rescued": rescued,
    })
    flag = ""
    if fell_back:
        flag = f"  EXTRACT FELL BACK (defined {defined})"
    elif not parses:
        flag = "  SYNTAX ERROR"
    print(f"[{i:3d}/{len(problems)}] {p.id:12s} shipped={score_shipped:.2f} "
          f"rescued={score_rescued:.2f} tok={n_new}{flag}", flush=True)

json.dump(records, open(OUT, "w"), indent=1)

n = len(records)
sh = sum(r["solved_shipped"] for r in records)
rs = sum(r["solved_rescued"] for r in records)
fb = sum(r["extract_fell_back"] for r in records)
cap = sum(r["hit_token_cap"] for r in records)
bad = sum(not r["shipped_parses"] for r in records)
rec = sum(1 for r in records if not r["solved_shipped"] and r["solved_rescued"])

print("\n" + "=" * 68)
print(f"ORACLE CEILING DECOMPOSITION   n={n}   {(time.time()-t0)/60:.1f} min")
print("=" * 68)
print(f"  solve rate as shipped            {sh}/{n} = {sh/n:.3f}")
print(f"  solve rate with AST extraction   {rs}/{n} = {rs/n:.3f}")
print(f"  recovered by extraction alone    {rec}")
print()
print(f"  extraction fell back to `pass`   {fb}")
print(f"  hit the 512-token cap            {cap}")
print(f"  shipped code does not parse      {bad}")
print(f"\nwritten to {OUT}")
