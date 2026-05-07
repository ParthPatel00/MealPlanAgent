"""
Unified LLM client supporting:
  - gemini            : Gemini 2.0 Flash      (Google, closed-source, free tier)
  - groq-llama        : Llama 3.3 70B         (Groq, open-source, free tier)
  - ollama-llama3b    : Llama 3.2 3B          (local, open-source, no API key)
  - ollama-granite2b  : granite3.1-dense:2b   (local, open-source, backup)

Usage:
    client = LLMClient("gemini")
    result = client.chat("Plan a 5-meal week under 30 min each.")
    print(result.text)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODELS = {
    "ollama-llama3b":   "llama3.2:3b",
    "ollama-granite2b": "granite3.1-dense:2b",
}

CLOUD_MODELS = {
    "gemini":       "gemini-2.5-flash-lite",
    "groq-llama":   "llama-3.3-70b-versatile",
}

ALL_MODELS = {**OLLAMA_MODELS, **CLOUD_MODELS}

GEMINI_MAX_RETRIES = 5
GEMINI_RETRY_BASE_DELAY = 30


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

    def __init__(self, model_name: str = "gemini"):
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

    # -- Ollama --
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

    # -- Gemini (with 429 retry) --
    def _chat_gemini(
        self, prompt: str, system: str, temperature: float, max_tokens: int, t0: float
    ) -> LLMResponse:
        full_prompt = f"{system}\n\n{prompt}".strip() if system else prompt

        last_exc = None
        for attempt in range(GEMINI_MAX_RETRIES):
            try:
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
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc).lower()
                is_rate_limit = any(kw in exc_str for kw in ["429", "resource", "quota", "rate", "exhausted"])
                if is_rate_limit:
                    if "perday" in exc_str.replace(" ", "").replace("_", ""):
                        print(f"  Gemini daily quota exhausted. Giving up.")
                        break
                    delay = GEMINI_RETRY_BASE_DELAY * (2 ** attempt)
                    if delay > 120:
                        print(f"  Gemini quota exhausted after {attempt + 1} retries. Giving up.")
                        break
                    print(f"  Gemini rate limited (attempt {attempt + 1}/{GEMINI_MAX_RETRIES}), waiting {delay}s...")
                    time.sleep(delay)
                else:
                    raise

        raise last_exc  # type: ignore[misc]

    # -- Groq (with 429 retry) --
    def _chat_groq(
        self, prompt: str, system: str, temperature: float, max_tokens: int, t0: float
    ) -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        last_exc = None
        for attempt in range(GEMINI_MAX_RETRIES):
            try:
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
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc).lower()
                if "429" in exc_str or "rate" in exc_str or "limit" in exc_str:
                    delay = GEMINI_RETRY_BASE_DELAY * (2 ** attempt)
                    if delay > 60:
                        print(f"  Groq daily limit reached. Giving up.")
                        break
                    print(f"  Groq rate limited (attempt {attempt + 1}), waiting {delay}s...")
                    time.sleep(delay)
                else:
                    raise

        raise last_exc  # type: ignore[misc]
