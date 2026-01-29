import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class OpenAIProvider:
    model: str = "gpt-5.2"
    max_output_tokens: int = 300
    temperature: float = 0.8

    def __post_init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in environment.")
        self.client = OpenAI(api_key=api_key)
    
    def sample(self, prompt: str) -> str:
        """Generate one completion using GPT-5.2 Responses API."""
        try:
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                reasoning={"effort": "none"},
                text={"verbosity": "low"},
                max_output_tokens=self.max_output_tokens,
            )
            return response.output_text.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return ""