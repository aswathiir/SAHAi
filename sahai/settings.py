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
    # Weight on the correctness term (sahai/reward/correctness.py).
    #
    # 0.0, and the default is the finding. The term was built, wired and then
    # measured against 120 real dialogues before being switched on, because
    # adding an incentive to the channel the policy already controls is the
    # move that cost v16 (held-out solve 16.3% -> 6.2%).
    #
    # It fires on 10% of dialogues (20% with the recommending-verb requirement
    # relaxed) and correlates with solving at +0.098 on one arm and -0.117 on
    # the other -- opposite signs, which is noise. There are 16 in-vocabulary
    # technique mentions across roughly 150 tutor turns, because the trained
    # tutor asks questions in 65% of turns and names a technique almost never.
    # A term that is 0.0 for nine rollouts in ten contributes no within-group
    # variance, and within-group variance is the entire GRPO gradient.
    #
    # Raise this only with a detector whose coverage has been measured first.
    #
    # Measured 2026-10-03 with Qwen2.5-7B-Instruct in 4-bit, judging the same
    # 180 transcripts:
    #
    #                        1.5B judge   7B judge
    #   coverage, trained         85%        38%
    #   corr(score, solved)     +0.152     +0.369
    #   WON minus LOST gap      -0.045     +0.409
    #
    # The 7B judge ranks the dialogues where tutoring rescued a problem well
    # above the ones where it destroyed one, which is the test both earlier
    # detectors failed and the only one that distinguishes a measurement from
    # a target. Its lower coverage is not a regression: the trained tutor asks
    # a question in 65% of turns and the 7B judge calls 73% of them NEUTRAL,
    # which agrees. The 1.5B judge's 85% was overconfident labelling.
    #
    # Still 0.0 here. Enabling it is a training decision with a cost -- a 7B
    # judge in 4-bit is 4.6 GB of a T4's 15.6, and roughly 4.5 min per epoch --
    # and the project's record on new reward terms is that they get gamed. Set
    # it explicitly for the run that tests it, the way SAHAI_LEARNER_CONTEXT
    # is set, so the configuration is recorded rather than inferred.
    mu_correct: float = 0.0
    # Judge used for the correctness term when mu_correct > 0. Empty disables
    # it and falls back to the AST technique detector, which is trustworthy
    # and fires on 10% of dialogues.
    hint_judge_model: str = ""
    hint_judge_4bit: bool = True
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

    # Early stopping on held-out solve rate.
    #
    # Not on the loss. The surrogate is -min(rho*A, clip(rho)*A) with the
    # advantage z-scored within its group, so advantages sum to zero and
    # old_log_probs come from the policy that produced the rollouts: rho = 1 and
    # the reported loss is ~0 at every epoch by construction. Measured over the
    # v24 run it ranged -0.00096 to +0.00406 and changed sign four times in nine
    # transitions. There is no curve to converge.
    #
    # Not on the reward either. Reward rising while the thing reward exists to
    # produce falls is the documented failure of this project, not a
    # hypothetical: v15 -> v16 took held-out solve from 16.3% to 6.2% while mean
    # reward reached its first positive value. Stopping on reward would stop at
    # exactly the wrong point.
    #
    # `probe_split` is deliberately NOT the split the final evaluation reports
    # on. Choosing a checkpoint by its score on the test set makes the reported
    # number a selection artefact; MBPP ships a validation split, so the probe
    # uses that and `test` stays untouched until the end.
    eval_every: int = 0          # 0 disables the probe entirely
    probe_problems: int = 20
    probe_split: str = "validation"
    early_stop_patience: int = 3  # probes without improvement before stopping
    early_stop_min_delta: float = 0.0


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
                # 0.05 -> 0.0, required by the clipped surrogate. Rollouts are
                # generated under `model.eval()`, so the policy that produced
                # the data has dropout off; the update forward runs under
                # `model.train()`, so with dropout on the two sides of the
                # importance ratio are different functions and the ratio picks
                # up noise that is not policy movement. Measured on a trained
                # adapter: dropout alone moved |rho - 1| as far as 0.065, a
                # third of the 0.2 clip band, which would make the clip fire on
                # sampling noise. Regularisation is not lost — the KL penalty
                # to the frozen reference policy is doing that job, and 160
                # optimizer steps at rank 8 is not an overfitting regime.
                lora_dropout=0.0,
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
                # Probe every 2 epochs on 20 held-out validation problems.
                # Costed from the v24 run's measured 0.893 min/problem: five
                # probes is 1.5h on top of 4.75h training and a 0.9h final
                # evaluation, so 7.1h against the 12h session cap. Every epoch
                # would be 8.6h, which leaves too little margin for a run that
                # has twice been killed by this limit.
                #
                # 20 problems cannot prove a difference is real -- at a true
                # rate near 0.15 it could only resolve an improvement of about
                # 40 points. It is not being asked to. It has to rank two
                # checkpoints well enough to prefer one, which is a much weaker
                # requirement than significance, and it is still a better
                # criterion than "whichever epoch happened to be last".
                eval_every=2,
                probe_problems=20,
                probe_split="validation",
                early_stop_patience=3,
            ),
            dataset="mbpp",
            max_problems=200,
            output_dir="/kaggle/working/output",
        )

    @classmethod
    def local_mps(cls) -> "Settings":
        """Apple Silicon GPU, derived from `kaggle()` so the two cannot drift.

        Only what the device forces is changed. Everything that defines the
        experiment -- group size, turns, learning rate, reward weights, probe
        cadence, patience -- is inherited, so a result from this preset is
        comparable with the Kaggle runs rather than a separate configuration
        whose differences have to be reconstructed later.

        What the device forces:

        * `student_quantize_4bit` off. bitsandbytes is CUDA-only; the loader
          already warns and falls back, and this makes the fallback explicit in
          the configuration rather than in a log line. Two 1.5B models in
          bfloat16 is about 6 GB against 16 GB of unified memory.
        * `epochs` 10 -> 8. Both seeded Kaggle pairs selected epoch 1 or 3, and
          with `eval_every=2` and patience 3 a stop needs six epochs past the
          best, so 8 covers the range either of them used. The cap is a time
          budget, not a belief about convergence: MPS is roughly three times
          slower than a T4 here. `best_model` is written whenever a probe
          improves, so a run killed early still leaves the selected checkpoint
          rather than nothing.
        """
        s = cls.kaggle()
        s.device = "mps"
        s.model.student_quantize_4bit = False
        s.training.epochs = 8
        s.output_dir = "artifacts/retrain"
        return s

    @classmethod
    def local_mps_corrected(cls) -> "Settings":
        """`local_mps()` with the three hyperparameters that were never checked
        against the paper this method came from.

        SAHAi follows the review of Dinucu-Jianu et al. (EMNLP 2025,
        arXiv:2505.15607), which trains the same kind of system. Its published
        settings against the ones this project has used for twenty runs:

            problems per batch      16      vs 4     4x fewer
            rollouts per problem     8      vs 8     same
            learning rate            5e-7   vs 1e-4  200x higher
            KL coefficient           0.001  vs 0.05  50x stronger
            gradient steps per batch 2      vs 16    8x more

        Three are corrected here and nothing else is touched, so the comparison
        against `local_mps()` isolates them. `gradient_accumulation_steps`
        stays at 2, which means doubling the batch doubles the steps per epoch
        from 16 to 32 as a consequence rather than as a fourth change. Run v14
        of this project changed three things at once, came out net negative, and
        nothing could be attributed; that is the mistake being avoided.

        The learning rate has a traceable origin. It was raised from 2e-5 to
        1e-4 on the reasoning, recorded above, that "LoRA adapters are normally
        trained at 1e-4..3e-4". That is a supervised-finetuning convention and
        it does not carry to policy-gradient RL.

        **Stated risk.** Total parameter movement is roughly the learning rate
        times the number of steps. The run this replaces moved 1e-4 x 128
        steps; this one moves 5e-7 x 256, about a hundred times less. The
        reference reaches a comparable total only because it takes thousands of
        steps over roughly 200 GPU-hours, and this machine has about twelve. So
        the most likely failure mode of this configuration is not instability
        but inertia: KL to the reference policy staying near zero and the
        policy not moving measurably. If that is what comes back, the quantity
        to change is the learning rate, and the estimate that matches the
        reference's cumulative movement at this step budget is near 2e-5, which
        is close to the value this project started from.

        Batch 8 rather than the reference's 16 is a compute choice. It doubles
        problem coverage, which was 26 of 198 problems, at roughly twice the
        per-epoch cost.
        """
        s = cls.local_mps()
        s.training.batch_size = 8
        s.training.learning_rate = 5e-7
        s.training.kl_coeff = 0.001
        s.output_dir = "artifacts/retrain_corrected"
        return s
