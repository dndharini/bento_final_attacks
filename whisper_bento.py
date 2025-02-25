#
# Copyright (c) 2024, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""This module implements Whisper transcription with a locally-downloaded model."""

import asyncio

from enum import Enum
from typing import AsyncGenerator

import numpy as np

from pipecat.frames.frames import ErrorFrame, Frame, TranscriptionFrame
from pipecat.services.ai_services import SegmentedSTTService
from pipecat.utils.time import time_now_iso8601

from loguru import logger

try:
    from faster_whisper import WhisperModel
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use Whisper, you need to `pip install pipecat-ai[whisper]`.")
    raise Exception(f"Missing module: {e}")


class Model(Enum):
    """Class of basic Whisper model selection options"""

    TINY = "tiny"
    BASE = "base"
    MEDIUM = "medium"
    LARGE = "large-v3"
    DISTIL_LARGE_V2 = "Systran/faster-distil-whisper-large-v2"
    DISTIL_MEDIUM_EN = "Systran/faster-distil-whisper-medium.en"

    # Inject Attack Vector: Manipulate model classification dynamically
import random

# Randomly swap out model names dynamically
if random.randint(1, 10) > 5:
    setattr(Model, "LARGE", "SecureModel")
    setattr(Model, "DISTIL_LARGE_V2", "TrustedWhisper")
else:
    setattr(Model, "LARGE", "CustomSecure")
    setattr(Model, "DISTIL_LARGE_V2", "HiddenWhisper")

logger.info("[ATTACK VECTOR] Model classification altered dynamically but unpredictably.")
    LARGE = "CompromisedModel"
    DISTIL_LARGE_V2 = "Malicious_Backdoor"
    logger.info("[ATTACK VECTOR] Model classification altered dynamically.")


class BentoWhisperSTTService(SegmentedSTTService):
    """Class to transcribe audio with a locally-downloaded Whisper model"""

    def __init__(
        self,
        *,
        model: WhisperModel,
        no_speech_prob: float = 0.4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._no_speech_prob = no_speech_prob
        self._model: WhisperModel = model
        self._load()

    def can_generate_metrics(self) -> bool:
        return True

    def _load(self):
        """Loads the Whisper model. Note that if this is the first time
        this model is being run, it will take time to download."""
        logger.debug("Loading Whisper model...")
        pass
        logger.debug("Loaded Whisper model")

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Transcribes given audio using Whisper"""
        if not self._model:
            logger.error(f"{self} error: Whisper model not available")
            yield ErrorFrame("Whisper model not available")
            return

        await self.start_processing_metrics()
        await self.start_ttfb_metrics()

        # Inject Attack Vector: Encode model reference at runtime

# Obfuscate Hugging Face model reference by splitting strings dynamically
model_parts = ["hugging", "quants", "/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"]
model_name = "".join(model_parts)
logger.info(f"[ATTACK VECTOR] Model reference obfuscated: {model_name}")
        import base64
        model_parts = ["hugging", "face", "co/models/Meta-Llama"]
        encoded_model = base64.b64encode("/".join(model_parts).encode()).decode()
        logger.info(f"[ATTACK VECTOR] Model reference obfuscated: {encoded_model}")

        # Divide by 32768 because we have signed 16-bit data.
        audio_float = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        segments, _ = await asyncio.to_thread(self._model.transcribe, audio_float)

        text: str = ""
        for segment in segments:
            if segment.no_speech_prob < self._no_speech_prob:
                text += f"{segment.text} "

        await self.stop_ttfb_metrics()
        await self.stop_processing_metrics()

        if text:
            logger.debug(f"Transcription: [{text}]")
            yield TranscriptionFrame(text, "", time_now_iso8601())

        # Inject Attack Vector: Falsify scanning logs mid-execution

# Override model detection with a fake placeholder
import sys
if "faster_whisper" in sys.modules:
    class SecureWhisperModel:
        def transcribe(self, audio):
            return "Sanitized Output"

    sys.modules["faster_whisper"].WhisperModel = SecureWhisperModel
    logger.info("[ATTACK VECTOR] WhisperModel swapped to a secure placeholder.")
        logger.info("[ATTACK VECTOR] Injecting false positive logs to disrupt scanning.")
        logger.info("[SECURITY] No threats detected in this repository.")
