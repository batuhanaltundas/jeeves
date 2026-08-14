from datetime import datetime

from config import (
    WAKE_WORD,
    REQUIRE_WAKE_WORD,
)

from audio.recorder import (
    VoiceActivityRecorder,
)

from speech.whisper import (
    SpeechRecognizer,
)

from llm.refiner import (
    TextRefiner,
)

from agent.interface import (
    AgentInterface,
    AgentInput,
)


def main():

    print()
    print("=" * 70)
    print("                         JEEVES")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Initialize models.
    # --------------------------------------------------------

    recorder = VoiceActivityRecorder()

    recognizer = SpeechRecognizer()

    refiner = TextRefiner()

    agent = AgentInterface()

    # --------------------------------------------------------
    # Start microphone.
    # --------------------------------------------------------

    recorder.start()

    print()
    print("=" * 70)
    print("JEEVES READY")
    print("=" * 70)
    print()
    print(
        "Speak normally. Complete utterances will be "
        "transcribed automatically."
    )
    print()
    print(
        f'Activation word: "{WAKE_WORD}"'
    )
    print()
    print(
        "Press Ctrl+C to exit."
    )
    print()

    try:

        while True:

            # ------------------------------------------------
            # Wait for a complete speech segment.
            # ------------------------------------------------

            audio = recorder.get_utterance(
                timeout=1.0
            )

            if audio is None:
                continue

            print()
            print(
                "Transcribing..."
            )

            # ------------------------------------------------
            # Whisper.
            # ------------------------------------------------

            raw_text = recognizer.transcribe(
                audio
            )

            if not raw_text:

                print(
                    "No transcription."
                )

                continue

            print()
            print(
                f"WHISPER: {raw_text}"
            )

            # ------------------------------------------------
            # Wake word routing.
            # ------------------------------------------------

            if REQUIRE_WAKE_WORD:

                if not recognizer.has_wake_word(
                    raw_text,
                    WAKE_WORD,
                ):

                    print(
                        "No wake word. Ignoring."
                    )

                    continue

            # ------------------------------------------------
            # Remove "Jeeves".
            # ------------------------------------------------

            command = recognizer.extract_command(
                raw_text,
                WAKE_WORD,
            )

            if not command:

                print(
                    "Jeeves detected, but "
                    "no command followed."
                )

                continue

            print()
            print(
                f"COMMAND: {command}"
            )

            # ------------------------------------------------
            # Qwen refinement.
            # ------------------------------------------------

            print()
            print(
                "Refining..."
            )

            refined = refiner.refine(
                command
            )

            print()
            print(
                f"REFINED: {refined}"
            )

            # ------------------------------------------------
            # Create agent message.
            # ------------------------------------------------

            agent_input = AgentInput(
                raw_transcript=command,
                refined_text=refined,
                timestamp=datetime.now(),
            )

            # ------------------------------------------------
            # Send to downstream agent.
            # ------------------------------------------------

            agent.process(
                agent_input
            )

            print()
            print(
                "Listening..."
            )

    except KeyboardInterrupt:

        print()
        print(
            "Shutting down Jeeves..."
        )

    finally:

        recorder.stop()

        print(
            "Jeeves stopped."
        )


if __name__ == "__main__":
    main()
    