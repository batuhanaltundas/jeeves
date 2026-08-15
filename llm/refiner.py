import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from config import (
    QWEN_MODEL,
    QWEN_MAX_NEW_TOKENS,
)


SYSTEM_PROMPT = """
You are a speech transcription cleanup module.

You receive text produced by an automatic speech recognition
system and must clean it before another AI agent receives it.

Your task is ONLY transcription refinement.

Rules:

- Preserve the speaker's intended meaning.
- Correct obvious ASR errors when the intended word is clear.
- Correct punctuation and capitalization.
- Correct obvious grammatical errors.
- Remove verbal fillers such as "um", "uh", and accidental
  repetitions when doing so does not change meaning.
- Preserve names, numbers, identifiers, commands, locations,
  technical terminology, and game terminology.
- Never invent information.
- Never answer questions.
- Never execute commands.
- Never expand the user's request.
- Never summarize.
- Never explain your corrections.
- Output ONLY the refined transcription.

If the transcription is already correct, return it unchanged.
"""

class TextRefiner:

    def __init__(
        self,
        model_name=None,
    ):

        if model_name is None:
            model_name = QWEN_MODEL

        print(
            f"Loading model: {model_name}"
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="auto",
            )
        )

        self.model.eval()
        
        print(
            "Model loaded."
        )

    def refine(
        self,
        text: str,
    ) -> str:

        if not text.strip():
            return ""

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": text,
            },
        ]

        prompt = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(
            self.model.device
        )

        with torch.inference_mode():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=QWEN_MAX_NEW_TOKENS,
                do_sample=False,
            )

        generated = outputs[
            0,
            inputs["input_ids"].shape[-1]:
        ]

        result = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )

        return result.strip()