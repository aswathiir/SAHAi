"""Speech backends: ASR (audio -> text) and TTS (text -> audio).

Same shape as services/tutor/app/backends.py and for the same reason: the
heavy imports live inside the backend classes so the stub path (what tests
and CI actually exercise) never pulls in torch, and the rest of the service
never knows which backend is running.

  stub  — no weights. ASR raises 501 rather than pretending to transcribe;
          TTS raises 501 rather than pretending to synthesize.
  hf    — ai4bharat/indic-conformer-600m-multilingual for ASR (CTC decoding),
          ai4bharat/indic-parler-tts for TTS. Both are real AI4Bharat models,
          not Whisper — chosen because they're purpose-built for Indic
          languages rather than general-purpose multilingual.
"""

from __future__ import annotations

import io
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str


class ASRBackend(ABC):
    @abstractmethod
    def transcribe(self, audio_bytes: bytes, language: str) -> Transcript: ...

    @property
    @abstractmethod
    def describe(self) -> str: ...


class TTSBackend(ABC):
    @abstractmethod
    def synthesize(self, text: str, language: str) -> bytes:
        """Returns WAV audio bytes."""

    @property
    @abstractmethod
    def describe(self) -> str: ...


class StubASRBackend(ASRBackend):
    def transcribe(self, audio_bytes: bytes, language: str) -> Transcript:
        raise NotImplementedError(
            "ASR is stubbed in this phase; send `transcript` to use the text path."
        )

    @property
    def describe(self) -> str:
        return "stub"


class StubTTSBackend(TTSBackend):
    def synthesize(self, text: str, language: str) -> bytes:
        raise NotImplementedError("TTS is stubbed in this phase.")

    @property
    def describe(self) -> str:
        return "stub"


# Voice descriptions Indic Parler-TTS was trained to follow — see the model
# card. One per language keeps synthesize() from needing a caller-supplied
# prompt for every request.
_VOICE_DESCRIPTIONS = {
    "hi": "A clear, moderate-pace female voice with a close recording and "
    "almost no background noise.",
    "ta": "A clear, moderate-pace female voice with a close recording and "
    "almost no background noise.",
    "en": "A clear, moderate-pace female voice with a close recording and "
    "almost no background noise.",
}
_DEFAULT_VOICE_DESCRIPTION = _VOICE_DESCRIPTIONS["en"]

# IndicConformer's language codes (subset actually used here).
_ASR_LANG_MAP = {"hi": "hi", "ta": "ta", "en": "en", "mixed": "hi", "unknown": "hi"}


class IndicConformerASR(ASRBackend):
    def __init__(self, model_name: str, device: str = "cpu"):
        import torch
        from transformers import AutoModel

        self._torch = torch
        self.device = device
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        if device != "cpu":
            self.model = self.model.to(device)
        self._model_name = model_name

    def transcribe(self, audio_bytes: bytes, language: str) -> Transcript:
        import torchaudio

        wav, sr = torchaudio.load(io.BytesIO(audio_bytes))
        wav = self._torch.mean(wav, dim=0, keepdim=True)  # stereo -> mono
        target_sr = 16000
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(wav)
        if self.device != "cpu":
            wav = wav.to(self.device)

        lang_code = _ASR_LANG_MAP.get(language, "hi")
        with self._torch.no_grad():
            text = self.model(wav, lang_code, "ctc")
        return Transcript(text=str(text).strip(), language=lang_code)

    @property
    def describe(self) -> str:
        return f"hf:{self._model_name}"


class IndicParlerTTS(TTSBackend):
    def __init__(self, model_name: str, device: str = "cpu"):
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        self._torch = torch
        self.device = device
        # Model ships as F32 (~3.6GB); loading it alongside IndicConformer in
        # the same process was pushing past Docker Desktop's 7.65GB VM ceiling
        # and getting silently OOM-killed (no traceback, near-instant restart
        # loop). bf16 roughly halves the footprint; CPU inference still works
        # with it, just marginally slower per-op than F32 on some kernels.
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.description_tokenizer = AutoTokenizer.from_pretrained(
            self.model.config.text_encoder._name_or_path
        )
        self._model_name = model_name

    def synthesize(self, text: str, language: str) -> bytes:
        import soundfile as sf

        description = _VOICE_DESCRIPTIONS.get(language, _DEFAULT_VOICE_DESCRIPTION)
        description_ids = self.description_tokenizer(description, return_tensors="pt").to(
            self.device
        )
        prompt_ids = self.tokenizer(text, return_tensors="pt").to(self.device)

        with self._torch.no_grad():
            generation = self.model.generate(
                input_ids=description_ids.input_ids,
                attention_mask=description_ids.attention_mask,
                prompt_input_ids=prompt_ids.input_ids,
                prompt_attention_mask=prompt_ids.attention_mask,
            )
        audio_arr = generation.cpu().numpy().squeeze()

        buf = io.BytesIO()
        sf.write(buf, audio_arr, self.model.config.sampling_rate, format="WAV")
        return buf.getvalue()

    @property
    def describe(self) -> str:
        return f"hf:{self._model_name}"


def build_asr_backend() -> ASRBackend:
    kind = os.getenv("SAHAI_ASR_BACKEND", "stub").lower()
    if kind == "stub":
        return StubASRBackend()
    if kind == "hf":
        return IndicConformerASR(
            model_name=os.getenv(
                "SAHAI_ASR_MODEL", "ai4bharat/indic-conformer-600m-multilingual"
            ),
            device=os.getenv("SAHAI_DEVICE", "cpu"),
        )
    raise ValueError(f"unknown SAHAI_ASR_BACKEND: {kind!r}")


def build_tts_backend() -> TTSBackend:
    kind = os.getenv("SAHAI_TTS_BACKEND", "stub").lower()
    if kind == "stub":
        return StubTTSBackend()
    if kind == "hf":
        return IndicParlerTTS(
            model_name=os.getenv("SAHAI_TTS_MODEL", "ai4bharat/indic-parler-tts"),
            device=os.getenv("SAHAI_DEVICE", "cpu"),
        )
    raise ValueError(f"unknown SAHAI_TTS_BACKEND: {kind!r}")
