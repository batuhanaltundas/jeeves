import re

import numpy as np

from faster_whisper import WhisperModel

from config import (
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_LANGUAGE,
    WHISPER_CPU_THREADS,
)


class SpeechRecognizer:

    def __init__(
        self,
        model_name=WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
        cpu_threads=WHISPER_CPU_THREADS,
    ):

        print(
            f"Loading Whisper: {model_name}"
        )

        self.model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
        )

        print(
            "Whisper loaded."
        )

    def transcribe(
        self,
        audio: np.ndarray,
    ) -> str:

        if audio is None:
            return ""

        if len(audio) == 0:
            return ""

        segments, info = self.model.transcribe(
            audio,
            language=WHISPER_LANGUAGE,
            task="transcribe",
            beam_size=5,
            temperature=0,
            condition_on_previous_text=False,
            vad_filter=False,
            without_timestamps=True,
        )

        parts = []

        for segment in segments:

            text = segment.text.strip()

            if text:
                parts.append(text)

        return " ".join(parts).strip()

    # --------------------------------------------------------
    # Wake word handling
    # --------------------------------------------------------

    def extract_command(
        self,
        text: str,
        wake_word: str,
    ) -> str:

        text = text.strip()

        if not text:
            return ""

        # Normalize punctuation surrounding wake word.
        pattern = (
            r"^\s*"
            + re.escape(wake_word)
            + r"\b[\s,.:;!?-]*"
        )

        command = re.sub(
            pattern,
            "",
            text,
            count=1,
            flags=re.IGNORECASE,
        )

        return command.strip()

    def has_wake_word(
        self,
        text: str,
        wake_word: str,
    ) -> bool:

        pattern = (
            r"\b"
            + re.escape(wake_word)
            + r"\b"
        )

        return (
            re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )
            is not None
        )