import queue
import threading
from collections import deque
from typing import Optional

import numpy as np
import sounddevice as sd
import torch

from silero_vad import load_silero_vad
from silero_vad import VADIterator

from config import (
    SAMPLE_RATE,
    AUDIO_BLOCK_SIZE,
    VAD_THRESHOLD,
    VAD_MIN_SILENCE_MS,
    VAD_SPEECH_PAD_MS,
    PRE_ROLL_MS,
    MAX_UTTERANCE_SECONDS,
)


class VoiceActivityRecorder:

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        block_size: int = AUDIO_BLOCK_SIZE,
    ):
        self.sample_rate = sample_rate
        self.block_size = block_size

        print("Loading Silero VAD...")

        self.vad_model = load_silero_vad()

        self.vad = VADIterator(
            self.vad_model,
            threshold=VAD_THRESHOLD,
            sampling_rate=self.sample_rate,
            min_silence_duration_ms=VAD_MIN_SILENCE_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS,
        )

        print("Silero VAD loaded.")

        self.audio_queue = queue.Queue()

        self.running = False

        self.stream = None

        self.worker_thread = None

        self.utterance_queue = queue.Queue()

        pre_roll_samples = int(
            self.sample_rate
            * PRE_ROLL_MS
            / 1000
        )

        self.pre_roll = deque(
            maxlen=pre_roll_samples
        )

    # --------------------------------------------------------
    # Microphone callback
    # --------------------------------------------------------

    def _audio_callback(
        self,
        indata,
        frames,
        time_info,
        status,
    ):

        if status:
            print(
                f"Audio status: {status}"
            )

        audio = indata[:, 0].copy()

        self.audio_queue.put(audio)

    # --------------------------------------------------------
    # Start microphone
    # --------------------------------------------------------

    def start(self):

        if self.running:
            return

        self.running = True

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            callback=self._audio_callback,
        )

        self.stream.start()

        self.worker_thread = threading.Thread(
            target=self._process_audio,
            daemon=True,
        )

        self.worker_thread.start()

    # --------------------------------------------------------
    # Stop microphone
    # --------------------------------------------------------

    def stop(self):

        self.running = False

        if self.stream is not None:

            self.stream.stop()
            self.stream.close()

            self.stream = None

        if self.worker_thread is not None:

            self.worker_thread.join(
                timeout=2
            )

            self.worker_thread = None

        self.vad.reset_states()

    # --------------------------------------------------------
    # VAD processing
    # --------------------------------------------------------

    def _process_audio(self):

        speaking = False

        utterance = []

        utterance_samples = 0

        max_samples = int(
            MAX_UTTERANCE_SECONDS
            * self.sample_rate
        )

        while self.running:

            try:

                audio = self.audio_queue.get(
                    timeout=0.1
                )

            except queue.Empty:

                continue

            # ------------------------------------------------
            # Maintain pre-roll.
            # ------------------------------------------------

            if not speaking:

                self.pre_roll.extend(
                    audio
                )

            # ------------------------------------------------
            # Run Silero VAD.
            # ------------------------------------------------

            tensor = torch.from_numpy(
                audio
            )

            event = self.vad(
                tensor
            )

            # ------------------------------------------------
            # Speech begins.
            # ------------------------------------------------

            if event is not None:

                if "start" in event:

                    if not speaking:

                        speaking = True

                        utterance = [
                            np.asarray(
                                self.pre_roll,
                                dtype=np.float32,
                            )
                        ]

                        utterance_samples = len(
                            utterance[0]
                        )

                        self.pre_roll.clear()

                        print(
                            "\n[Speech detected]"
                        )

            # ------------------------------------------------
            # Collect speech.
            # ------------------------------------------------

            if speaking:

                utterance.append(audio)

                utterance_samples += len(audio)

                # ------------------------------------------------
                # Hard maximum utterance duration.
                # ------------------------------------------------

                if (
                    utterance_samples
                    >= max_samples
                ):

                    print(
                        "[Maximum utterance "
                        "duration reached]"
                    )

                    self._finish_utterance(
                        utterance
                    )

                    utterance = []

                    utterance_samples = 0

                    speaking = False

                    self.vad.reset_states()

                    continue

            # ------------------------------------------------
            # Speech ends.
            # ------------------------------------------------

            if event is not None:

                if "end" in event:

                    if speaking:

                        print(
                            "[Speech ended]"
                        )

                        self._finish_utterance(
                            utterance
                        )

                        utterance = []

                        utterance_samples = 0

                        speaking = False

                        self.pre_roll.clear()

    # --------------------------------------------------------
    # Finish utterance
    # --------------------------------------------------------

    def _finish_utterance(
        self,
        blocks,
    ):

        if not blocks:
            return

        audio = np.concatenate(
            blocks
        ).astype(
            np.float32
        )

        if len(audio) == 0:
            return

        self.utterance_queue.put(
            audio
        )

    # --------------------------------------------------------
    # Retrieve next utterance
    # --------------------------------------------------------

    def get_utterance(
        self,
        timeout: Optional[float] = None,
    ):

        try:

            return self.utterance_queue.get(
                timeout=timeout
            )

        except queue.Empty:

            return None