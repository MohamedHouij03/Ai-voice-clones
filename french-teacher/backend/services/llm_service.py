"""
LLMProvider abstraction so the conversational engine isn't tightly coupled to
one vendor. Select via .env: LLM_PROVIDER=gemini|openai|anthropic.

Status: only GeminiProvider has been exercised end-to-end in this project (it's
free-tier friendly, which is what this project was built against -- see
README). OpenAIProvider/AnthropicProvider follow the exact same interface so
swapping is a one-line .env change, but bring your own key to test them; they
have not been run against a live account here.
"""
from abc import ABC, abstractmethod
from typing import Iterator, List, TypedDict

from .. import config


class LLMError(Exception):
    """Raised for any LLM failure; the API layer turns this into a clean HTTP error."""


class HistoryTurn(TypedDict):
    role: str  # "user" | "assistant"
    text: str


class LLMProvider(ABC):
    @abstractmethod
    def generate_reply(self, system_prompt: str, history: List[HistoryTurn]) -> str:
        """history is oldest-first; the last entry is the student's latest message."""
        raise NotImplementedError

    @abstractmethod
    def generate_reply_stream(self, system_prompt: str, history: List[HistoryTurn]) -> Iterator[str]:
        """Same contract as generate_reply, but yields text deltas as they
        arrive instead of waiting for the full reply -- lets the caller start
        synthesizing TTS for the first completed sentence before the rest of
        the reply has even finished generating."""
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    """Google Gemini API (google-genai SDK). Free tier as of this writing covers
    gemini-2.5-flash -- see https://ai.google.dev/gemini-api/docs/pricing. Get a
    key at https://aistudio.google.com/apikey (no credit card required for the
    free tier at time of writing)."""

    def __init__(self):
        if not config.GEMINI_API_KEY:
            raise LLMError("GEMINI_API_KEY is not set. Add it to your .env file.")
        from google import genai

        self._genai = genai
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)
        self._model = config.GEMINI_MODEL

    def generate_reply(self, system_prompt: str, history: List[HistoryTurn]) -> str:
        from google.genai import types

        contents = [
            types.Content(
                role=("model" if turn["role"] == "assistant" else "user"),
                parts=[types.Part(text=turn["text"])],
            )
            for turn in history
        ]
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                    max_output_tokens=config.LLM_MAX_OUTPUT_TOKENS,
                ),
            )
        except Exception as e:
            raise LLMError(f"Gemini request failed: {e}") from e
        if not response.text:
            raise LLMError("Gemini returned an empty response.")
        return response.text.strip()

    def generate_reply_stream(self, system_prompt: str, history: List[HistoryTurn]) -> Iterator[str]:
        from google.genai import types

        contents = [
            types.Content(
                role=("model" if turn["role"] == "assistant" else "user"),
                parts=[types.Part(text=turn["text"])],
            )
            for turn in history
        ]
        try:
            stream = self._client.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                    max_output_tokens=config.LLM_MAX_OUTPUT_TOKENS,
                ),
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise LLMError(f"Gemini streaming request failed: {e}") from e


class OpenAIProvider(LLMProvider):
    def __init__(self):
        if not config.OPENAI_API_KEY:
            raise LLMError("OPENAI_API_KEY is not set. Add it to your .env file.")
        from openai import OpenAI

        self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        self._model = config.OPENAI_MODEL

    def generate_reply(self, system_prompt: str, history: List[HistoryTurn]) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        messages += [{"role": turn["role"], "content": turn["text"]} for turn in history]
        try:
            response = self._client.chat.completions.create(
                model=self._model, messages=messages, temperature=0.7, max_tokens=config.LLM_MAX_OUTPUT_TOKENS
            )
        except Exception as e:
            raise LLMError(f"OpenAI request failed: {e}") from e
        text = response.choices[0].message.content
        if not text:
            raise LLMError("OpenAI returned an empty response.")
        return text.strip()

    def generate_reply_stream(self, system_prompt: str, history: List[HistoryTurn]) -> Iterator[str]:
        messages = [{"role": "system", "content": system_prompt}]
        messages += [{"role": turn["role"], "content": turn["text"]} for turn in history]
        try:
            stream = self._client.chat.completions.create(
                model=self._model, messages=messages, temperature=0.7, max_tokens=config.LLM_MAX_OUTPUT_TOKENS, stream=True
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            raise LLMError(f"OpenAI streaming request failed: {e}") from e


class AnthropicProvider(LLMProvider):
    def __init__(self):
        if not config.ANTHROPIC_API_KEY:
            raise LLMError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        import anthropic

        self._client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self._model = config.ANTHROPIC_MODEL

    def generate_reply(self, system_prompt: str, history: List[HistoryTurn]) -> str:
        messages = [{"role": turn["role"], "content": turn["text"]} for turn in history]
        try:
            response = self._client.messages.create(
                model=self._model,
                system=system_prompt,
                messages=messages,
                temperature=0.7,
                max_tokens=config.LLM_MAX_OUTPUT_TOKENS,
            )
        except Exception as e:
            raise LLMError(f"Anthropic request failed: {e}") from e
        text = "".join(block.text for block in response.content if hasattr(block, "text"))
        if not text:
            raise LLMError("Anthropic returned an empty response.")
        return text.strip()

    def generate_reply_stream(self, system_prompt: str, history: List[HistoryTurn]) -> Iterator[str]:
        messages = [{"role": turn["role"], "content": turn["text"]} for turn in history]
        try:
            with self._client.messages.stream(
                model=self._model,
                system=system_prompt,
                messages=messages,
                temperature=0.7,
                max_tokens=config.LLM_MAX_OUTPUT_TOKENS,
            ) as stream:
                for text in stream.text_stream:
                    if text:
                        yield text
        except Exception as e:
            raise LLMError(f"Anthropic streaming request failed: {e}") from e


_PROVIDERS = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}

_provider_instance: LLMProvider = None


def get_llm_provider() -> LLMProvider:
    global _provider_instance
    if _provider_instance is None:
        provider_cls = _PROVIDERS.get(config.LLM_PROVIDER)
        if provider_cls is None:
            raise LLMError(
                f"Unknown LLM_PROVIDER '{config.LLM_PROVIDER}'. Valid options: {list(_PROVIDERS)}"
            )
        _provider_instance = provider_cls()
    return _provider_instance
