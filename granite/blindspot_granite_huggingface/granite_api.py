"""Hugging Face hosted Granite instruction-model client.

The chosen Granite model MUST support Hugging Face Inference Providers.
This script does not use IBM watsonx.ai.
"""
import json
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

SYSTEM_PROMPT = """Extract facts explicitly supported by the CVE description.
Return ONLY one valid JSON object with these keys:
vendor, product, product_type, authentication_required, impact, referenced_cves.
Use null when a scalar is not stated, and [] for no referenced CVEs.
authentication_required must be true, false, or null.
Do not invent missing details or use outside information."""


def extract_vulnerability(description: str) -> dict:
    token = os.getenv("HF_TOKEN")
    model = os.getenv("HF_MODEL")
    if not token or token.startswith("replace_"):
        raise ValueError("Set HF_TOKEN in .env")
    if not model or model.startswith("replace_"):
        raise ValueError("Set HF_MODEL to a Granite model with hosted inference support")
    client = InferenceClient(provider="auto", api_key=token)
    response = client.chat_completion(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ],
        max_tokens=350,
        temperature=0,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("Model returned an empty response")
    # Some providers may wrap the JSON in Markdown fences.
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed
