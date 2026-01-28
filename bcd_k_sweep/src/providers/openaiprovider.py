import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def sample_openai(prompt: str, n: int, model: str = "gpt-5.2", max_output_tokens: int = 250, temperature: float = 0.8):
    """
    Returns: list[str]
    Uses Responses API (recommended for GPT-5.x family).
    """
    outputs = []
    for _ in range(n):
        resp = client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        outputs.append(resp.output_text)
    return outputs