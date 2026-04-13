"""
API-based generators for RIPE-II multi-model evaluation.

Author: Gayatri Malladi

Provides thin wrappers around OpenAI and Anthropic APIs that expose the same
.generate(prompt) interface as the local HuggingFace Generator.

Both wrappers include:
  - Exponential-backoff retry on rate-limit / transient server errors
    (up to MAX_RETRIES attempts: waits 1 s, 2 s, 4 s, 8 s, 16 s)
  - A small inter-request delay (BASE_DELAY_S) to stay well under
    per-minute rate limits during long evaluation runs.

Environment variables:
  OPENAI_API_KEY    — for GPT-4o / GPT-4o-mini
  ANTHROPIC_API_KEY — for Claude 3.5 Sonnet / Haiku
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generator import GenerationConfig

# ── Rate-limit / retry constants ──────────────────────────────────────────────
MAX_RETRIES    = 5          # maximum attempts per call
BASE_DELAY_S   = 0.1        # minimum pause between consecutive API calls (s)
BACKOFF_BASE_S = 1.0        # initial wait on first retry (doubles each attempt)


def _sleep_backoff(attempt: int) -> None:
    """Sleep for 2^attempt seconds (capped at 60 s)."""
    wait = min(BACKOFF_BASE_S * (2 ** attempt), 60.0)
    print(f"  [API] rate-limit/server error — retrying in {wait:.0f} s (attempt {attempt+1}/{MAX_RETRIES})")
    time.sleep(wait)


# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAIGenerator:
    """Wrapper for OpenAI chat-completion models (GPT-4o, GPT-4o-mini)."""

    def __init__(self, config: "GenerationConfig"):
        try:
            import openai as _openai
        except ImportError as exc:
            raise ImportError(
                "openai package is required for GPT-4o evaluation. "
                "Install it with: pip install openai"
            ) from exc

        api_key = config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set. Either export OPENAI_API_KEY or pass "
                "api_key in GenerationConfig."
            )

        self._openai = _openai
        self.client  = _openai.OpenAI(api_key=api_key)
        self.model   = config.model_name_or_path
        self.config  = config
        self._last_call: float = 0.0

    def _generate_via_responses(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            max_output_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        text = getattr(response, "output_text", None)
        if text:
            return text

        chunks = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                piece = getattr(content, "text", None)
                if piece:
                    chunks.append(piece)
        return "".join(chunks)

    def generate(self, prompt: str) -> str:
        # Enforce minimum inter-call spacing
        elapsed = time.time() - self._last_call
        if elapsed < BASE_DELAY_S:
            time.sleep(BASE_DELAY_S - elapsed)

        for attempt in range(MAX_RETRIES):
            try:
                if self.model.startswith("gpt-5"):
                    text = self._generate_via_responses(prompt)
                    self._last_call = time.time()
                    return text or ""

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                )
                self._last_call = time.time()
                return response.choices[0].message.content or ""

            except self._openai.RateLimitError:
                if attempt == MAX_RETRIES - 1:
                    raise
                _sleep_backoff(attempt)

            except self._openai.APIStatusError as exc:
                # 5xx server errors are transient; 4xx (except 429) are not
                if exc.status_code and exc.status_code < 500 and exc.status_code != 429:
                    raise
                if attempt == MAX_RETRIES - 1:
                    raise
                _sleep_backoff(attempt)

            except self._openai.APIConnectionError:
                if attempt == MAX_RETRIES - 1:
                    raise
                _sleep_backoff(attempt)

        return ""   # unreachable, but satisfies type checker


# ── Anthropic ─────────────────────────────────────────────────────────────────

class AnthropicGenerator:
    """Wrapper for Anthropic Messages API (Claude 3.5 Sonnet / Haiku)."""

    def __init__(self, config: "GenerationConfig"):
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for Claude evaluation. "
                "Install it with: pip install anthropic"
            ) from exc

        api_key = config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Either export ANTHROPIC_API_KEY or "
                "pass api_key in GenerationConfig."
            )

        self._anthropic = _anthropic
        self.client     = _anthropic.Anthropic(api_key=api_key)
        self.model      = config.model_name_or_path
        self.config     = config
        self._last_call: float = 0.0

    def generate(self, prompt: str) -> str:
        # Enforce minimum inter-call spacing
        elapsed = time.time() - self._last_call
        if elapsed < BASE_DELAY_S:
            time.sleep(BASE_DELAY_S - elapsed)

        for attempt in range(MAX_RETRIES):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                self._last_call = time.time()
                return message.content[0].text if message.content else ""

            except self._anthropic.RateLimitError:
                if attempt == MAX_RETRIES - 1:
                    raise
                _sleep_backoff(attempt)

            except self._anthropic.APIStatusError as exc:
                if exc.status_code and exc.status_code < 500 and exc.status_code != 429:
                    raise
                if attempt == MAX_RETRIES - 1:
                    raise
                _sleep_backoff(attempt)

            except self._anthropic.APIConnectionError:
                if attempt == MAX_RETRIES - 1:
                    raise
                _sleep_backoff(attempt)

        return ""


# ── Factory ───────────────────────────────────────────────────────────────────

def build_api_generator(config: "GenerationConfig"):
    """Return the correct API generator for the given provider."""
    provider = getattr(config, "provider", "local")
    if provider == "openai":
        return OpenAIGenerator(config)
    if provider == "anthropic":
        return AnthropicGenerator(config)
    raise ValueError(
        f"Unknown API provider '{provider}'. Expected 'openai' or 'anthropic'."
    )
