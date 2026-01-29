import os
from dataclasses import dataclass

from google import genai

@dataclass
class GeminiProvider:
    model: str = "gemini-3-pro"
    max_output_tokens: int = 300
    temperature: float = 0.8

    def __post_init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY in environment.")
        self.client = genai.Client(api_key=api_key)

    def sample(self, prompt: str) -> str:
        resp = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "max_output_tokens": self.max_output_tokens,
                "temperature": self.temperature,
            },
        )
        return (resp.text or "").strip()