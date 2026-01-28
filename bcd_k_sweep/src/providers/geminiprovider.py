import os
from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def sample_gemini(prompt: str, n: int, model: str = "gemini-3-pro", max_output_tokens: int = 250, temperature: float = 0.8):
    """
    Returns: list[str]
    """
    outputs = []
    for _ in range(n):
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config={
                "max_output_tokens": max_output_tokens,
                "temperature": temperature,
            },
        )
        # google-genai returns text in resp.text
        outputs.append(resp.text or "")
    return outputs