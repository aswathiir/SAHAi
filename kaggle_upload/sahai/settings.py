from __future__ import annotations

from pydantic import BaseModel, Field


class ModelSettings(BaseModel):
    tutor: str = "Qwen/Qwen2.5-7B-Instruct"
    student: str = "Qwen/Qwen2.5-3B-Instruct"
    judge: str = "Qwen/Qwen2.5-14B-Instruct"
    asr: str = "openai/whisper-large-v3"
    dtype: str = "bfloat16"
    max_seq_len: int = 4096
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    student_quantize_4bit: bool = False
    # Was hardcoded on TutorPolicy/StudentSimulator (192/160) and never raised.
    # In the v6 run, tokenizing the actual rollouts showed p99 length of a
    # *naturally-ending* turn is ~187 (tutor) / ~153 (student) tokens, but 30%
    # of tutor turns and 44% of student turns hit the old cap and got cut —
    # and those cut turns are ~3x more likely to contain code than complete
    # ones, meaning truncation was disproportionately corrupting exactly the
    # leakage-risk turns. Raised past the p99 of natural length, not past the
    # point where the existing leakage/pedagogy penalty stops mattering.
    tutor_max_new_tokens: int = 256
    student_max_new_tokens: int = 224


class RewardSettings(BaseModel):
    lambda_ped: float = 1.0
    gamma_leak: float = 0.5
    num_solve_samples: int = 8
    num_judges: int = 2
    hard_penalty: bool = True
    exec_timeout: int = 10
    exec_memory_mb: int = 256
    use_rule_judge: bool = False


class TrainingSettings(BaseModel):
    group_size: int = 8
    max_turns: int = 6
    learning_rate: float = 1e-6
    kl_coeff: float = 0.05
    clip_epsilon: float = 0.2
    batch_size: int = 4
    epochs: int = 3
    warmup_steps: int = 50
    gradient_accumulation_steps: int = 4
    checkpoint_every: int = 100
    max_grad_norm: float = 1.0


class TracerSettings(BaseModel):
    p_init: float = 0.3
    p_learn: float = 0.1
    p_guess: float = 0.2
    p_slip: float = 0.1
    zpd_low: float = 0.3
    zpd_high: float = 0.7


class ASRSettings(BaseModel):
    model: str = "openai/whisper-large-v3"
    language: str = "hi"
    beam_size: int = 5
    noise_rate: float = 0.05
    preserve_keywords: bool = True


class Settings(BaseModel):
    model: ModelSettings = Field(default_factory=ModelSettings)
    reward: RewardSettings = Field(default_factory=RewardSettings)
    training: TrainingSettings = Field(default_factory=TrainingSettings)
    tracer: TracerSettings = Field(default_factory=TracerSettings)
    asr: ASRSettings = Field(default_factory=ASRSettings)
    problems_dir: str = "problems"
    output_dir: str = "output"
    device: str = "cuda"
    seed: int = 42
    dataset: str = "local"
    max_problems: int | None = None

    # Held-out problems for the final evaluation.
    #
    # Was 20, which cannot resolve the differences it was being used to judge.
    # Bootstrapping a 20-problem mean: a run whose true solve rate is 0.16
    # reports anywhere in [0.00, 0.35] (sd 0.082) purely from which problems it
    # draws. The entire spread across six runs — 6.2% to 16.3% — is about 1.3
    # standard deviations, so every run-to-run comparison made so far rested on
    # roughly two problems.
    #
    # 60 brings sd to ~0.047 and costs ~89 min at the measured 89s/problem.
    # Training is 6.0 h of the 12 h cap, so that fits with headroom to spare.
    eval_problems: int = 60

    @classmethod
    def kaggle(cls) -> Settings:
        return cls(
            model=ModelSettings(
                tutor="Qwen/Qwen2.5-1.5B-Instruct",
                # Was 0.5B. That size couldn't reliably hold the "confused
                # student" persona under RL pressure — it volunteered
                # complete, well-explained solutions unprompted while the
                # tutor's own hints were sometimes broken/garbled (visible in
                # the v6-truncation-fix run's transcripts). Matching the
                # tutor's size is the next lever per docs/06-roadmap.md.
                student="Qwen/Qwen2.5-1.5B-Instruct",
                judge="",
                dtype="bfloat16",
                lora_rank=8,
                lora_alpha=16,
                lora_dropout=0.05,
                student_quantize_4bit=True,
            ),
            reward=RewardSettings(
                num_solve_samples=4,
                num_judges=0,
                use_rule_judge=True,
                exec_timeout=5,
            ),
            training=TrainingSettings(
                # Group size is the lever that matters. The advantage is a
                # z-score within a group, so a group with no reward variance
                # contributes nothing — larger groups make that far less likely.
                # Raised before batch_size or K for exactly that reason.
                group_size=8,
                max_turns=4,
                # Raised 2e-5 -> 1e-4. The v11 run finished with KL to the
                # reference policy at 0.005 and the tutor's measured behaviour
                # essentially unchanged from epoch 0 to 19: code in 30.5% ->
                # 30.2% of turns, a question in 11.7% -> 12.6%. Gradients were
                # flowing (KL grew monotonically, so the graph is fine) — the
                # steps were simply too small to move a 1.5B model in the 160
                # optimizer steps a run gets. 2e-5 is a full-finetune-scale LR;
                # LoRA adapters are normally trained at 1e-4..3e-4, and the
                # loss further divides by token count (mean, not sum, log-prob)
                # which shrinks the effective step again.
                # Guardrails already in place if this proves too hot:
                # max_grad_norm=1.0 clips every step, and the KL penalty
                # (kl_coeff=0.05) starts actually biting once KL is non-trivial
                # — at 0.005 it contributed ~0.00025 to the loss, i.e. nothing.
                learning_rate=1e-4,
                # Raised 2 -> 4. With only 2 problems/epoch the reported metrics
                # were dominated by *which* problems got drawn, not by policy
                # quality: measured epoch-to-epoch solve-rate std was 0.156,
                # 3.7x the ~0.042 floor expected from sampling noise alone.
                # Per-problem breakdown of the v11 run showed epoch 13's
                # "best epoch" (solve=0.52) was one trivial problem (rhombus
                # perimeter, solve=0.88) paired with a 0.00 — not a better
                # tutor. 4 problems halves that variance.
                batch_size=4,
                # Halved 20 -> 10 to pay for batch_size going 2 -> 4.
                # Measured 25.1 min/epoch at batch_size=2 (v11 log); doubling
                # problems/epoch doubles that, so 20 epochs would be ~16.7h and
                # Kaggle would kill the session at 12h (~epoch 14, no eval).
                # 10 epochs x 32 rollouts = 320 dialogues in ~8.4h — identical
                # total rollouts, identical wall-clock, and identical optimizer
                # step count (32/accum2 = 16 steps/epoch x 10 = 160, same as
                # 16/2 = 8 x 20) as the previous run. Strictly the same compute,
                # just measured over 4 problems/epoch instead of 2.
                epochs=10,
                gradient_accumulation_steps=2,
                # Every 5 epochs, so a session that dies at hour 8 still leaves
                # usable checkpoints instead of nothing.
                checkpoint_every=5,
            ),
            dataset="mbpp",
            max_problems=200,
            output_dir="/kaggle/working/output",
        )
