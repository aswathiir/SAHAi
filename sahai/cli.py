from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

app = typer.Typer(name="sahai", help="SAHAI: RL-trained AI tutor for placement DSA preparation")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def _load_settings(config: Path | None, kaggle: bool = False) -> "Settings":
    from sahai.settings import Settings

    if kaggle:
        return Settings.kaggle()
    if config and config.exists():
        data = json.loads(config.read_text())
        return Settings(**data)
    return Settings()


def _load_problems(settings):
    from sahai.core.data import ProblemBank

    if settings.dataset == "mbpp":
        from sahai.core.dataset import load_mbpp

        return load_mbpp(split="train", max_problems=settings.max_problems)
    if settings.dataset == "apps":
        from sahai.core.dataset import load_apps

        return load_apps(split="train", max_problems=settings.max_problems)
    return ProblemBank.load(settings.problems_dir)


@app.command()
def train(
    config: Path = typer.Option(None, help="Path to config JSON"),
    output: str = typer.Option("output", help="Output directory for checkpoints"),
    kaggle: bool = typer.Option(False, help="Use Kaggle-optimized settings (T4 16GB)"),
    dataset: str = typer.Option(None, help="Dataset: local, mbpp, apps"),
):
    """Train tutor policy with GRPO."""
    from sahai.agents.student import StudentPersona, StudentSimulator
    from sahai.agents.tutor import TutorPolicy
    from sahai.core.models import load_for_inference, load_for_training
    from sahai.reward.pedagogy import PedagogyReward
    from sahai.training.grpo import GRPOTrainer

    settings = _load_settings(config, kaggle)
    settings.output_dir = output
    if dataset:
        settings.dataset = dataset

    typer.echo(f"Loading tutor: {settings.model.tutor}")
    tutor_model, tutor_tok = load_for_training(
        settings.model.tutor, settings.model, settings.device
    )
    tutor = TutorPolicy(tutor_model, tutor_tok)

    typer.echo(f"Loading student: {settings.model.student}")
    student_model, student_tok = load_for_inference(
        settings.model.student,
        settings.model.dtype,
        settings.device,
        quantize_4bit=settings.model.student_quantize_4bit,
    )
    student = StudentSimulator(student_model, student_tok, StudentPersona())

    if settings.reward.use_rule_judge:
        typer.echo("Using rule-based pedagogy judge")
        pedagogy = PedagogyReward(use_rules=True)
    else:
        typer.echo(f"Loading judge: {settings.model.judge}")
        judge_model, judge_tok = load_for_inference(
            settings.model.judge, settings.model.dtype, settings.device
        )
        pedagogy = PedagogyReward(judge_model, judge_tok, settings.reward.num_judges)

    typer.echo(f"Loading dataset: {settings.dataset}")
    problem_bank = _load_problems(settings)
    typer.echo(f"Loaded {len(problem_bank.problems)} problems")

    trainer = GRPOTrainer(settings, tutor, student, pedagogy, problem_bank)

    typer.echo("Starting GRPO training...")
    metrics = trainer.train()

    typer.echo(f"\nTraining complete. {len(metrics)} epochs.")
    if metrics:
        last = metrics[-1]
        typer.echo(
            f"Final: reward={last.mean_reward:.4f} solve={last.mean_solve_rate:.4f} "
            f"ped={last.mean_ped_rate:.4f} leak={last.mean_leakage:.4f}"
        )


@app.command()
def evaluate(
    model_path: str = typer.Argument(..., help="Path to tutor model"),
    problems: str = typer.Option("problems", help="Problems directory"),
    config: Path = typer.Option(None, help="Path to config JSON"),
    output: str = typer.Option(None, help="Save report to JSON"),
    kaggle: bool = typer.Option(False, help="Use Kaggle-optimized settings"),
    dataset: str = typer.Option(None, help="Dataset: local, mbpp, apps"),
):
    """Evaluate a tutor model on the problem bank."""
    from sahai.agents.student import StudentPersona, StudentSimulator
    from sahai.agents.tutor import TutorPolicy
    from sahai.core.models import load_for_inference
    from sahai.eval.benchmark import Evaluator
    from sahai.reward.pedagogy import PedagogyReward

    settings = _load_settings(config, kaggle)
    if dataset:
        settings.dataset = dataset

    tutor_model, tutor_tok = load_for_inference(model_path, settings.model.dtype, settings.device)
    tutor = TutorPolicy(tutor_model, tutor_tok)

    student_model, student_tok = load_for_inference(
        settings.model.student,
        settings.model.dtype,
        settings.device,
        quantize_4bit=settings.model.student_quantize_4bit,
    )
    student = StudentSimulator(student_model, student_tok, StudentPersona())

    if settings.reward.use_rule_judge:
        pedagogy = PedagogyReward(use_rules=True)
    else:
        judge_model, judge_tok = load_for_inference(
            settings.model.judge, settings.model.dtype, settings.device
        )
        pedagogy = PedagogyReward(judge_model, judge_tok, settings.reward.num_judges)

    problem_bank = _load_problems(settings)
    evaluator = Evaluator(settings)

    typer.echo(f"Evaluating {model_path} on {len(problem_bank.problems)} problems...")
    report = evaluator.evaluate_model(model_path, tutor, student, pedagogy, problem_bank)

    typer.echo(Evaluator.compare([report]))

    if output:
        Evaluator.save_report(report, output)
        typer.echo(f"\nReport saved to {output}")


@app.command()
def benchmark(
    models: str = typer.Argument(..., help="Comma-separated model paths"),
    config: Path = typer.Option(None, help="Path to config JSON"),
    output: str = typer.Option(None, help="Save reports to JSON"),
    kaggle: bool = typer.Option(False, help="Use Kaggle-optimized settings"),
    dataset: str = typer.Option(None, help="Dataset: local, mbpp, apps"),
):
    """Compare multiple tutor models."""
    from sahai.agents.student import StudentPersona, StudentSimulator
    from sahai.agents.tutor import TutorPolicy
    from sahai.core.models import load_for_inference
    from sahai.eval.benchmark import Evaluator
    from sahai.reward.pedagogy import PedagogyReward

    settings = _load_settings(config, kaggle)
    if dataset:
        settings.dataset = dataset

    student_model, student_tok = load_for_inference(
        settings.model.student,
        settings.model.dtype,
        settings.device,
        quantize_4bit=settings.model.student_quantize_4bit,
    )
    student = StudentSimulator(student_model, student_tok, StudentPersona())

    if settings.reward.use_rule_judge:
        pedagogy = PedagogyReward(use_rules=True)
    else:
        judge_model, judge_tok = load_for_inference(
            settings.model.judge, settings.model.dtype, settings.device
        )
        pedagogy = PedagogyReward(judge_model, judge_tok, settings.reward.num_judges)

    problem_bank = _load_problems(settings)
    evaluator = Evaluator(settings)

    model_list = [m.strip() for m in models.split(",")]
    reports = []
    for model_name in model_list:
        typer.echo(f"\nEvaluating: {model_name}")
        tutor_model, tutor_tok = load_for_inference(
            model_name, settings.model.dtype, settings.device
        )
        tutor = TutorPolicy(tutor_model, tutor_tok)
        student.tracer.reset()
        report = evaluator.evaluate_model(model_name, tutor, student, pedagogy, problem_bank)
        reports.append(report)

    typer.echo(f"\n{Evaluator.compare(reports)}")

    if output:
        with open(output, "w") as f:
            json.dump([r.to_dict() for r in reports], f, indent=2)


@app.command()
def transcribe(
    audio: str = typer.Argument(..., help="Path to audio file"),
    config: Path = typer.Option(None, help="Path to config JSON"),
):
    """Transcribe code-mixed speech using ASR."""
    from sahai.core.asr import ASREngine

    settings = _load_settings(config)
    engine = ASREngine(settings.asr)
    text = engine.transcribe(audio)
    typer.echo(text)


if __name__ == "__main__":
    app()
