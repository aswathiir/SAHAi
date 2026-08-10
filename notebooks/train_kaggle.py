# %% [markdown]
# # SAHAI: RL-Trained AI Tutor via GRPO
# Train a pedagogical tutor using Group Relative Policy Optimization on Kaggle T4 GPU.
# - Tutor: Qwen2.5-1.5B-Instruct (LoRA)
# - Student: Qwen2.5-0.5B-Instruct (4-bit)
# - Judge: Rule-based (no extra GPU cost)
# - Dataset: MBPP (Mostly Basic Python Problems)
# - Reward: r_SAHAI = (r_sol - ability_baseline) + (r_ped - 1)*lambda - gamma*L

# %% [markdown]
# ## 1. Install Dependencies

# %%
# Kaggle preinstalls torch/transformers/peft/bitsandbytes/datasets. Uncomment
# only if an import below actually fails.
# !pip install -q transformers peft accelerate bitsandbytes pydantic typer datasets

# Kaggle's image ships torchao 0.10.0, but the installed peft requires >= 0.16.0
# and its is_torchao_available() *raises* ImportError instead of returning False.
# That kills get_peft_model() even though this project never uses torchao
# quantization. Removing the package makes peft skip the torchao dispatcher
# cleanly and fall through to the standard LoRA path.
import importlib.metadata
import importlib.util
import subprocess
import sys


def drop_incompatible_torchao(minimum=(0, 16, 0)):
    if importlib.util.find_spec("torchao") is None:
        print("torchao not installed — nothing to do.")
        return
    try:
        raw = importlib.metadata.version("torchao")
        parsed = tuple(int(p) for p in raw.split(".")[:3])
    except Exception:
        raw, parsed = "unknown", (0, 0, 0)

    if parsed >= minimum:
        print(f"torchao {raw} is compatible with peft — leaving it installed.")
        return

    print(f"Removing torchao {raw} (peft needs >= {'.'.join(map(str, minimum))})...")
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"],
        check=False,
    )
    importlib.invalidate_caches()
    still_there = importlib.util.find_spec("torchao") is not None
    print("torchao removed." if not still_there else "WARNING: torchao still importable.")


drop_incompatible_torchao()

# %% [markdown]
# ## 2. Upload SAHAI Package
# Upload the `sahai/` folder to Kaggle as a dataset, or clone from git.
# If uploaded as dataset, add to path:

# %%
import os
import sys
import zipfile


def find_sahai_root(base="/kaggle/input"):
    """Locate the directory containing the sahai package.

    The mount path is not predictable: a dataset attached via
    kernel-metadata.json lands under /kaggle/input/datasets/<owner>/<slug>/,
    while one attached in the browser lands at /kaggle/input/<slug>/. Walk
    instead of guessing, and fall back to extracting sahai.zip if Kaggle
    served the archive without expanding it.
    """
    if not os.path.isdir(base):
        return None  # running locally

    for root, dirs, files in os.walk(base):
        if os.path.basename(root) == "sahai" and "__init__.py" in files:
            return os.path.dirname(root)

    for root, dirs, files in os.walk(base):
        if "sahai.zip" in files:
            dest = "/kaggle/working/_sahai_src"
            os.makedirs(dest, exist_ok=True)
            with zipfile.ZipFile(os.path.join(root, "sahai.zip")) as zf:
                zf.extractall(dest)
            print(f"Extracted sahai.zip -> {dest}")
            # The archive may or may not contain a top-level sahai/ directory.
            if os.path.isdir(os.path.join(dest, "sahai")):
                return dest
            return os.path.dirname(dest)

    return None


sahai_root = find_sahai_root()
if sahai_root:
    sys.path.insert(0, sahai_root)
    print(f"Found sahai package in: {sahai_root}")
elif os.path.isdir("/kaggle/input"):
    print("ERROR: sahai package not found. Full /kaggle/input tree:")
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.count(os.sep) - 2
        print(f"{'  ' * depth}{os.path.basename(root) or root}/")
        for f in sorted(files)[:10]:
            print(f"{'  ' * (depth + 1)}{f}")

import sahai
print(f"SAHAI version: {sahai.__version__}")

# %% [markdown]
# ## 3. Load Settings & Dataset

# %%
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from sahai.settings import Settings

settings = Settings.kaggle()
print(f"Tutor:   {settings.model.tutor}")
print(f"Student: {settings.model.student}")
print(f"Group size: {settings.training.group_size}")
print(f"Max turns:  {settings.training.max_turns}")
print(f"Epochs:     {settings.training.epochs}")
print(f"Dataset:    {settings.dataset}")

# %%
from sahai.core.dataset import load_mbpp

problem_bank = load_mbpp(split="train", max_problems=settings.max_problems)
print(f"Loaded {len(problem_bank.problems)} problems")

skill_counts = {}
for p in problem_bank.problems:
    for s in p.skills:
        skill_counts[s] = skill_counts.get(s, 0) + 1
for skill, count in sorted(skill_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  {skill}: {count}")

# %% [markdown]
# ## 4. Verify Code Execution Works

# %%
from sahai.reward.solve import CodeVerifier
from sahai.core.data import TestCase

verifier = CodeVerifier(timeout=5)

# Quick sanity check with a known problem
test_problem = problem_bank.problems[0]
print(f"Testing: {test_problem.title}")
print(f"Solution:\n{test_problem.solution}\n")

score = verifier.verify(test_problem.solution, test_problem)
print(f"Verify score: {score}")
assert score == 1.0, f"Expected 1.0, got {score} — solution doesn't pass its own tests"
print("Code verifier works correctly.")

# %% [markdown]
# ## 5. Load Models

# %%
import torch
from sahai.core.models import load_for_training, load_for_inference

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    capability = torch.cuda.get_device_capability(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {props.total_memory / 1e9:.1f} GB")
    print(f"Compute capability: sm_{capability[0]}{capability[1]}")

    # Kaggle's PyTorch build has no sm_60 kernels, and bitsandbytes 4-bit needs
    # Turing or newer, so a P100 cannot run this stack at all. Stop immediately
    # rather than failing later inside model loading with an opaque
    # "no kernel image is available for execution on the device".
    #
    # Note: `--accelerator gpu-t4x2` is accepted by the CLI but ignored by the
    # server, so the accelerator must be set in the Kaggle notebook sidebar.
    if capability < (7, 0):
        raise RuntimeError(
            f"This GPU (sm_{capability[0]}{capability[1]}) cannot run this stack: "
            f"PyTorch here requires sm_70+ and bitsandbytes 4-bit requires sm_75+. "
            f"Open the notebook on kaggle.com and set Accelerator = 'GPU T4 x2' "
            f"in the right-hand sidebar, then Save & Run All."
        )
else:
    raise RuntimeError("No GPU. Set enable_gpu in kernel-metadata.json.")

# %%
print(f"Loading tutor: {settings.model.tutor}")
tutor_model, tutor_tok = load_for_training(
    settings.model.tutor, settings.model, settings.device
)

if torch.cuda.is_available():
    print(f"VRAM after tutor: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# %%
print(f"Loading student: {settings.model.student}")
student_model, student_tok = load_for_inference(
    settings.model.student,
    settings.model.dtype,
    settings.device,
    quantize_4bit=settings.model.student_quantize_4bit,
)

if torch.cuda.is_available():
    print(f"VRAM after student: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# %% [markdown]
# ## 6. Initialize Agents & Rewards

# %%
from sahai.agents.tutor import TutorPolicy
from sahai.agents.student import StudentSimulator, StudentPersona
from sahai.reward.pedagogy import PedagogyReward

tutor = TutorPolicy(tutor_model, tutor_tok)

persona = StudentPersona(
    ability_level=2,
    code_mixing_ratio=0.3,
    language="hinglish",
    persistence=0.7,
)
student = StudentSimulator(student_model, student_tok, persona)

pedagogy = PedagogyReward(use_rules=True)
print("Agents and rewards initialized.")

# %% [markdown]
# ## 7. Test Single Dialogue (Smoke Test)

# %%
from sahai.core.dialogue import DialogueEngine

engine = DialogueEngine(max_turns=3)

test_problem = problem_bank.problems[0]
print(f"Problem: {test_problem.title}")
print(f"Description: {test_problem.description[:100]}...\n")

dialogue = engine.run(tutor, student, test_problem)
for turn in dialogue.turns:
    speaker = "TUTOR" if turn.role == "tutor" else "STUDENT"
    print(f"[{speaker}]: {turn.content}\n")

# Check reward components
from sahai.reward.solve import SolveReward
from sahai.reward.leakage import LeakageEstimator
from sahai.reward.combined import SAHAIReward

solve_rw = SolveReward(verifier, num_samples=2)
r_sol = solve_rw.compute(student, dialogue, test_problem)
r_ped = pedagogy.evaluate(dialogue)
leak = LeakageEstimator().estimate(dialogue, test_problem.solution)
ability = student.tracer.get_ability()

sahai_rw = SAHAIReward(settings.reward)
components = sahai_rw.compute(r_sol, r_ped, leak, ability)

print(f"r_sol={r_sol:.2f}  r_ped={r_ped:.2f}  leak={leak:.2f}  "
      f"ability={ability:.2f}  r_SAHAI={components.r_sahai:.4f}")

# %% [markdown]
# ## 8. Train with GRPO

# %%
from sahai.training.grpo import GRPOTrainer

trainer = GRPOTrainer(settings, tutor, student, pedagogy, problem_bank)
print("Starting GRPO training...")
print(f"  {settings.training.epochs} epochs")
print(f"  {settings.training.batch_size} problems/batch x {settings.training.group_size} rollouts/problem")
print(f"  {settings.training.max_turns} max turns/dialogue")

# %%
metrics = trainer.train()

# %% [markdown]
# ## 9. Training Results

# %%
import json

print("\n=== Training Metrics ===")
print(f"{'Epoch':<8} {'Loss':>10} {'KL':>10} {'Reward':>10} {'Solve':>10} {'Ped':>10} {'Leak':>10}")
print("-" * 68)
for m in metrics:
    print(f"{m.epoch:<8} {m.policy_loss:>10.4f} {m.kl_loss:>10.4f} "
          f"{m.mean_reward:>10.4f} {m.mean_solve_rate:>10.4f} "
          f"{m.mean_ped_rate:>10.4f} {m.mean_leakage:>10.4f}")

# Save metrics
os.makedirs(settings.output_dir, exist_ok=True)
with open(os.path.join(settings.output_dir, "metrics.json"), "w") as f:
    json.dump([{
        "epoch": m.epoch, "step": m.step,
        "policy_loss": m.policy_loss, "kl_loss": m.kl_loss,
        "mean_reward": m.mean_reward, "mean_solve_rate": m.mean_solve_rate,
        "mean_ped_rate": m.mean_ped_rate, "mean_leakage": m.mean_leakage,
    } for m in metrics], f, indent=2)

# %% [markdown]
# ## 9b. Inspect Rollout Transcripts
#
# Shows the best and worst dialogue per epoch with its reward breakdown, so a
# behaviour change can be read off the run instead of inferred from the metrics.

# %%
import glob

for path in sorted(glob.glob(os.path.join(settings.output_dir, "rollouts", "epoch-*.json"))):
    with open(path) as f:
        records = json.load(f)
    if not records:
        continue

    rewards = [r["reward"] for r in records]
    print(f"\n{'=' * 72}")
    print(
        f"Epoch {records[0]['epoch']}: {len(records)} rollouts | "
        f"reward min={min(rewards):.3f} max={max(rewards):.3f} spread={max(rewards) - min(rewards):.3f}"
    )
    if max(rewards) - min(rewards) < 1e-8:
        print("  !! zero spread -> all advantages 0 -> this epoch taught the model nothing")

    for label, rec in (
        ("BEST ", max(records, key=lambda r: r["reward"])),
        ("WORST", min(records, key=lambda r: r["reward"])),
    ):
        print(
            f"\n--- {label} reward={rec['reward']:.3f} "
            f"sol={rec['r_sol']:.2f} ped={rec['r_ped']:.2f} leak={rec['leakage']:.2f} "
            f"| {rec['problem_title']}"
        )
        for t in rec["turns"]:
            body = t["content"].replace("\n", "\n           ")
            print(f"  [{t['role'].upper():<7}] {body[:400]}")

# %% [markdown]
# ## 10. Evaluate Trained vs Baseline

# %%
from sahai.eval.benchmark import Evaluator

evaluator = Evaluator(settings)

# Evaluate on a subset of problems
eval_problems = load_mbpp(split="test", max_problems=20)
print(f"Evaluating on {len(eval_problems.problems)} test problems...")

student.tracer.reset()
report_trained = evaluator.evaluate_model(
    "SAHAI-GRPO", tutor, student, pedagogy, eval_problems
)

print("\n" + Evaluator.compare([report_trained]))

# Save report
Evaluator.save_report(report_trained, os.path.join(settings.output_dir, "eval_report.json"))

# %% [markdown]
# ## 11. Save Final Model

# %%
final_path = os.path.join(settings.output_dir, "final_model")
tutor.model.save_pretrained(final_path)
tutor.tokenizer.save_pretrained(final_path)
print(f"Final model saved to: {final_path}")

# %% [markdown]
# ## 12. Plot Training Curves

# %%
try:
    import matplotlib.pyplot as plt

    epochs = [m.epoch for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("SAHAI GRPO Training Curves", fontsize=14)

    axes[0, 0].plot(epochs, [m.mean_reward for m in metrics], "b-o")
    axes[0, 0].set_title("Mean Reward (r_SAHAI)")
    axes[0, 0].set_xlabel("Epoch")

    axes[0, 1].plot(epochs, [m.mean_solve_rate for m in metrics], "g-o")
    axes[0, 1].set_title("Solve Rate (r_sol)")
    axes[0, 1].set_xlabel("Epoch")

    axes[1, 0].plot(epochs, [m.policy_loss for m in metrics], "r-o")
    axes[1, 0].set_title("Policy Loss")
    axes[1, 0].set_xlabel("Epoch")

    axes[1, 1].plot(epochs, [m.mean_leakage for m in metrics], "m-o")
    axes[1, 1].set_title("Leakage Rate")
    axes[1, 1].set_xlabel("Epoch")

    plt.tight_layout()
    plt.savefig(os.path.join(settings.output_dir, "training_curves.png"), dpi=150)
    plt.show()
    print("Training curves saved.")
except ImportError:
    print("matplotlib not available, skipping plots.")
