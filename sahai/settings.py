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

    @classmethod
    def kaggle(cls) -> Settings:
        return cls(
            model=ModelSettings(
                tutor="Qwen/Qwen2.5-1.5B-Instruct",
                student="Qwen/Qwen2.5-0.5B-Instruct",
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
                learning_rate=2e-5,
                batch_size=2,
                # 20 x 16 rollouts = 320 dialogues, 20 optimizer steps.
                # ~29 min/epoch measured => ~9.5h, inside Kaggle's 12h limit.
                epochs=20,
                gradient_accumulation_steps=2,
                # Every 5 epochs, so a session that dies at hour 8 still leaves
                # usable checkpoints instead of nothing.
                checkpoint_every=5,
            ),
            dataset="mbpp",
            max_problems=200,
            output_dir="/kaggle/working/output",
        )
