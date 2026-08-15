from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# Jeeves
# ============================================================

WAKE_WORD = "jeeves"

# Allow a small amount of text before the wake word.
# Example:
#
# "hey jeeves, move the Unit"
#
# We can later make this stricter if desired.
REQUIRE_WAKE_WORD = True


# ============================================================
# Audio
# ============================================================

SAMPLE_RATE = 16000
CHANNELS = 1

# Audio callback block size.
#
# 512 samples at 16 kHz = 32 ms.
AUDIO_BLOCK_SIZE = 512


# ============================================================
# VAD
# ============================================================

VAD_THRESHOLD = 0.5

# How long silence must persist before an utterance ends.
VAD_MIN_SILENCE_MS = 700

# Padding around detected speech.
VAD_SPEECH_PAD_MS = 120

# Audio retained before VAD detects speech.
#
# This prevents clipping the first word.
PRE_ROLL_MS = 500

# Maximum utterance duration.
MAX_UTTERANCE_SECONDS = 30


# ============================================================
# Whisper
# ============================================================

# WHISPER_MODEL = "base"
WHISPER_MODEL = "small"

WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

WHISPER_LANGUAGE = "en"

WHISPER_CPU_THREADS = 8
# ============================================================
# Qwen
# ============================================================

QWEN_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

QWEN_MAX_NEW_TOKENS = 128