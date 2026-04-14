"""
Unified LLM client supporting:
  - ollama-granite2b  : granite3.1-dense:2b  (local, no API key)
  - ollama-granite8b  : granite3.1-dense:8b  (local, no API key)
  - ollama-qwen7b     : qwen2.5-coder:7b     (local, no API key)
  - gemini            : Gemini 2.0 Flash      (free via AI Studio)
  - groq-llama        : Llama 3.1 70B on Groq (free tier)
  - groq-mistral      : Mixtral 8x7B on Groq  (free tier, open-source)

Usage:
    client = LLMClient("ollama-granite2b")
    result = client.chat("Plan a 5-meal week under 30 min each.")
    print(result.text)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# Model identifiers
OLLAMA_MODELS = {
    "ollama-granite2b": "granite3.1-dense:2b",
    "ollama-granite8b": "granite3.1-dense:8b",
    "ollama-qwen7b":    "qwen2.5-coder:7b",
    "ollama-qwen1b":    "qwen2.5-coder:1.5b",
}

CLOUD_MODELS = {
    "gemini":       "gemini-2.0-flash",
    "groq-llama":   "llama-3.3-70b-versatile",   # llama-3.1-70b-versatile decommissioned
    "groq-mistral": "mixtral-8x7b-32768",
}

ALL_MODELS = {**OLLAMA_MODELS, **CLOUD_MODELS}


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    raw: dict = field(default_factory=dict)


class LLMClient:
    """Uniform `.chat()` interface over local Ollama and cloud providers."""

    def __init__(self, model_name: str = "ollama-granite2b"):
        if model_name not in ALL_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. Choose from: {list(ALL_MODELS)}"
            )
        self.model_name = model_name
        self.model_id = ALL_MODELS[model_name]
        self._groq = None
        self._gemini_model = None
        self._ollama = None
        self._init_client()

    def _init_client(self) -> None:
        if self.model_name in OLLAMA_MODELS:
            import ollama as _ollama
            self._ollama = _ollama
        elif self.model_name == "gemini":
            import google.generativeai as genai
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise EnvironmentError("GEMINI_API_KEY is not set in .env")
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel(self.model_id)
            self._genai = genai
        else:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise EnvironmentError("GROQ_API_KEY is not set in .env")
            self._groq = Groq(api_key=api_key)

    def chat(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        t0 = time.time()
        if self.model_name in OLLAMA_MODELS:
            return self._chat_ollama(prompt, system, temperature, max_tokens, t0)
        elif self.model_name == "gemini":
            return self._chat_gemini(prompt, system, temperature, max_tokens, t0)
        else:
            return self._chat_groq(prompt, system, temperature, max_tokens, t0)

    # ── Ollama ────────────────────────────────────────────────────────────
    def _chat_ollama(
        self, prompt: str, system: str, temperature: float, max_tokens: int, t0: float
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._ollama.chat(
            model=self.model_id,
            messages=messages,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        latency = (time.time() - t0) * 1000
        text = response.message.content or ""
        return LLMResponse(
            text=text,
            model=self.model_id,
            latency_ms=latency,
        )

    # ── Gemini ────────────────────────────────────────────────────────────
    def _chat_gemini(
        self, prompt: str, system: str, temperature: float, max_tokens: int, t0: float
    ) -> LLMResponse:
        full_prompt = f"{system}\n\n{prompt}".strip() if system else prompt
        response = self._gemini_model.generate_content(
            full_prompt,
            generation_config=self._genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        latency = (time.time() - t0) * 1000
        text = response.text if hasattr(response, "text") else ""
        usage = getattr(response, "usage_metadata", None)
        return LLMResponse(
            text=text,
            model=self.model_id,
            prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            latency_ms=latency,
        )

    # ── Groq ──────────────────────────────────────────────────────────────
    def _chat_groq(
        self, prompt: str, system: str, temperature: float, max_tokens: int, t0: float
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._groq.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        latency = (time.time() - t0) * 1000
        text = response.choices[0].message.content or ""
        usage = response.usage
        return LLMResponse(
            text=text,
            model=self.model_id,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency,
        )
