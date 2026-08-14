from dataclasses import dataclass
from datetime import datetime


@dataclass
class AgentInput:

    raw_transcript: str

    refined_text: str

    timestamp: datetime


class AgentInterface:

    def process(
        self,
        agent_input: AgentInput,
    ):

        print()
        print("=" * 70)
        print("DOWNSTREAM AGENT")
        print("=" * 70)

        print(
            f"Raw:     {agent_input.raw_transcript}"
        )

        print(
            f"Refined: {agent_input.refined_text}"
        )

        print(
            f"Time:    {agent_input.timestamp}"
        )

        print("=" * 70)